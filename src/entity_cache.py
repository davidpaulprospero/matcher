"""
Cross-Project Entity Image Cache

Manages a global cache of entity images that can be shared across projects.
Enables fuzzy entity name matching using difflib.SequenceMatcher.
"""

from __future__ import annotations

import json
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

from .cache import BaseCache, CacheEntry

if TYPE_CHECKING:
    from src.config import EntityCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class CachedEntity:
    """A cached entity with its images"""
    entity_name: str
    entity_type: str
    images: List[str]  # Absolute paths in cache
    source_project: str
    cached_at: str
    query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CachedEntity':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EntityCache(BaseCache):
    """
    Global cache for entity images.

    Features:
    - Fuzzy entity name matching (difflib.SequenceMatcher)
    - JSON index for fast lookups
    - Age-based expiration
    - Multiple cache strategies (copy, symlink, reference)
    """

    IMAGES_SUBDIR = "images"

    def __init__(self, config: 'EntityCacheConfig'):
        """
        Args:
            config: EntityCacheConfig object with cache settings
        """
        self.enabled = getattr(config, 'enabled', False)
        cache_dir = Path(os.path.expanduser(getattr(config, 'cache_dir', '~/.matcher_entity_cache')))
        self.fuzzy_threshold = getattr(config, 'fuzzy_threshold', 0.85)
        self.max_age_days = getattr(config, 'max_age_days', 0)
        self.cache_strategy = getattr(config, 'cache_strategy', 'copy')

        # Convert max_age_days to TTL seconds (0 = no expiration)
        ttl_seconds = self.max_age_days * 24 * 3600 if self.max_age_days > 0 else 0

        # Initialize BaseCache
        super().__init__(
            cache_dir=cache_dir,
            index_name="entity_cache_index.json",
            ttl_seconds=ttl_seconds,
            auto_save=True
        )

        if self.enabled:
            self._ensure_cache_dir()

    def _serialize_entry(self, entry: CacheEntry) -> dict:
        """Serialize CachedEntity to dict"""
        return {
            'data': entry.data,  # CachedEntity.to_dict()
            'cached_at': entry.cached_at,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: dict) -> CacheEntry:
        """Deserialize dict to CachedEntity entry

        Handles both new BaseCache format and legacy format for migration.
        """
        # New format: {"data": {...}, "cached_at": ..., "metadata": {}}
        if 'data' in data and isinstance(data['data'], dict):
            return CacheEntry(
                data=data['data'],
                cached_at=data['cached_at'],
                key='',
                metadata=data.get('metadata', {})
            )
        # Legacy format: entity data directly (migration path)
        # Assume it's a CachedEntity dict with entity_name, entity_type, images, etc.
        elif 'entity_name' in data:
            return CacheEntry(
                data=data,  # Use the entity data directly
                cached_at=data.get('cached_at', ''),
                key='',
                metadata={}
            )
        else:
            # Unknown format, log and return empty entry
            logger.warning(f"Unknown cache entry format, skipping: {list(data.keys())}")
            return CacheEntry(data={}, cached_at='', key='', metadata={})

    def _load_index(self) -> None:
        """Load index with migration support for old format"""
        if not self.index_path.exists():
            self.index = {}
            logger.debug(f"Created new cache index at {self.index_path}")
            return

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if this is the old format with "entities" key
            if 'entities' in data and isinstance(data['entities'], dict):
                logger.info(f"Migrating old entity cache format to BaseCache format")
                # Migrate old format to new format
                migrated_index = {}
                for entity_name, entity_data in data['entities'].items():
                    # Wrap in new format
                    migrated_index[entity_name] = {
                        'data': entity_data,
                        'cached_at': entity_data.get('cached_at', ''),
                        'metadata': {}
                    }
                self.index = migrated_index
                # Save migrated format
                self._save_index()
                logger.info(f"Migrated {len(migrated_index)} entities to new format")
            else:
                # Already in new format
                self.index = data
                logger.debug(f"Loaded cache index: {len(self.index)} entries from {self.index_path}")

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache index: {e}. Starting fresh.")
            self.index = {}

    def _ensure_cache_dir(self) -> None:
        """Create cache directory structure if it doesn't exist"""
        (self.cache_dir / self.IMAGES_SUBDIR).mkdir(exist_ok=True)

    def find_entity(self, entity_name: str, entity_type: str = "") -> Optional[CachedEntity]:
        """
        Find an entity in the cache with fuzzy matching.

        Args:
            entity_name: Entity name to search for
            entity_type: Optional entity type filter (PERSON, GPE, etc.)

        Returns:
            CachedEntity if found, None otherwise
        """
        if not self.enabled:
            return None

        # Get all entries and convert to CachedEntity objects
        all_entries = self.get_all()

        # Exact match first (case-insensitive)
        name_lower = entity_name.lower()
        for cached_name, entry in all_entries.items():
            cached = CachedEntity.from_dict(entry.data)
            if cached_name.lower() == name_lower:
                if not entity_type or cached.entity_type == entity_type:
                    if self._is_valid(cached):
                        logger.debug(f"Cache hit (exact): '{entity_name}' -> '{cached_name}'")
                        return cached

        # Fuzzy match
        best_match = None
        best_score = 0.0

        for name, entry in all_entries.items():
            cached = CachedEntity.from_dict(entry.data)
            if entity_type and cached.entity_type != entity_type:
                continue

            score = self._fuzzy_similarity(entity_name, name)
            if score >= self.fuzzy_threshold and score > best_score:
                if self._is_valid(cached):
                    best_match = cached
                    best_score = score

        if best_match:
            logger.debug(f"Cache hit (fuzzy {best_score:.2f}): '{entity_name}' -> '{best_match.entity_name}'")

        return best_match

    def _fuzzy_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two strings (0.0-1.0)"""
        # Normalize strings
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()

        if s1 == s2:
            return 1.0

        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Use SequenceMatcher for similarity ratio
        return SequenceMatcher(None, s1, s2).ratio()

    def _is_valid(self, cached: CachedEntity) -> bool:
        """Check if cached entry is still valid (not expired, files exist)"""
        # Check age
        if self.max_age_days > 0:
            try:
                cached_time = datetime.fromisoformat(cached.cached_at)
                if datetime.now() - cached_time > timedelta(days=self.max_age_days):
                    logger.debug(f"Cache entry expired: {cached.entity_name}")
                    return False
            except ValueError:
                pass  # Invalid date format, consider valid

        # Check at least one file exists
        valid_files = [p for p in cached.images if Path(p).exists()]
        if not valid_files:
            logger.debug(f"Cache entry has no valid files: {cached.entity_name}")
            return False

        return True

    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        image_paths: List[str],
        source_project: str,
        query: str = ""
    ) -> None:
        """
        Add an entity to the cache.

        Args:
            entity_name: Entity name
            entity_type: Entity type (PERSON, GPE, etc.)
            image_paths: List of image file paths (will be copied to cache)
            source_project: Name of project that downloaded these images
            query: Search query used
        """
        if not self.enabled:
            return

        if not image_paths:
            return

        # Create entity-specific subdirectory in cache
        safe_name = self._safe_name(entity_name)
        cache_entity_dir = self.cache_dir / self.IMAGES_SUBDIR / safe_name
        cache_entity_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to cache
        cached_paths = []
        for src_path in image_paths:
            src = Path(src_path)
            if src.exists():
                dst = cache_entity_dir / src.name
                if not dst.exists():
                    try:
                        shutil.copy2(src, dst)
                        logger.debug(f"Cached image: {src.name}")
                    except Exception as e:
                        logger.warning(f"Failed to cache image {src}: {e}")
                        continue
                cached_paths.append(str(dst))

        if cached_paths:
            entity = CachedEntity(
                entity_name=entity_name,
                entity_type=entity_type,
                images=cached_paths,
                source_project=source_project,
                cached_at=datetime.now().isoformat(),
                query=query
            )
            # Use BaseCache's set method (auto-saves)
            self.set(entity_name, entity.to_dict())
            logger.info(f"Cached entity '{entity_name}': {len(cached_paths)} images")

    def _safe_name(self, name: str) -> str:
        """Convert name to filesystem-safe string"""
        # Replace problematic characters
        safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in name)
        safe = safe.strip().replace(' ', '_')
        return safe[:50]  # Limit length

    def get_images_for_project(
        self,
        cached: CachedEntity,
        project_images_dir: str
    ) -> List[str]:
        """
        Get image paths for use in a project.

        Depending on cache_strategy:
        - "copy": Copies images to project folder
        - "symlink": Creates symlinks to cache
        - "reference": Returns absolute paths to cache

        Args:
            cached: CachedEntity to use
            project_images_dir: Project's images directory

        Returns:
            List of image paths usable in the project
        """
        project_dir = Path(project_images_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        result_paths = []

        for cache_path in cached.images:
            cache_file = Path(cache_path)
            if not cache_file.exists():
                continue

            if self.cache_strategy == "reference":
                # Use absolute path directly
                result_paths.append(str(cache_file.resolve()))

            elif self.cache_strategy == "symlink":
                # Create symlink (may fail on Windows without admin)
                link_path = project_dir / cache_file.name
                if not link_path.exists():
                    try:
                        link_path.symlink_to(cache_file.resolve())
                    except OSError as e:
                        logger.warning(f"Symlink failed (falling back to copy): {e}")
                        # Fall back to copy
                        shutil.copy2(cache_file, link_path)
                result_paths.append(str(link_path))

            else:  # "copy" (default)
                dest_path = project_dir / cache_file.name
                if not dest_path.exists():
                    try:
                        shutil.copy2(cache_file, dest_path)
                    except Exception as e:
                        logger.warning(f"Failed to copy cached image: {e}")
                        continue
                result_paths.append(str(dest_path))

        return result_paths

    def cleanup(self) -> int:
        """
        Remove expired entries and entries with missing files.

        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0

        removed = []

        # Get all entries and check validity
        for name, entry in list(self.get_all().items()):
            cached = CachedEntity.from_dict(entry.data)
            if not self._is_valid(cached):
                removed.append(name)
                self.delete(name)  # Use BaseCache's delete method

        if removed:
            logger.info(f"Cleaned up {len(removed)} expired/invalid cache entries")

        return len(removed)

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}

        # Get all entries and convert to CachedEntity objects
        all_entries = self.get_all()
        entities = [CachedEntity.from_dict(entry.data) for entry in all_entries.values()]

        total_images = sum(len(e.images) for e in entities)
        valid_entries = sum(1 for e in entities if self._is_valid(e))

        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_entities': len(entities),
            'valid_entities': valid_entries,
            'total_images': total_images,
            'fuzzy_threshold': self.fuzzy_threshold,
            'max_age_days': self.max_age_days,
            'cache_strategy': self.cache_strategy
        }
