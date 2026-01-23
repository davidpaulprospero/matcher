"""Cache for voiceover analysis results.

Caches keyword extraction, chapter detection, and entity extraction
based on voiceover file content hash. Enables reusing expensive LLM
analysis across fresh runs when the voiceover hasn't changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCache, CacheEntry

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeResult:
    """Cached analysis result."""
    keywords: List[str]
    entities: List[Dict[str, Any]]
    topic_context: str
    segments: List[Dict[str, Any]]
    chapters: List[Dict[str, Any]]
    is_listicle: bool
    location_chapters: List[Dict[str, Any]]
    # Config hash to invalidate if settings change
    config_hash: str = ""
    # Source file info
    file_hash: str = ""
    file_path: str = ""
    file_size: int = 0


class AnalyzeCache(BaseCache[AnalyzeResult]):
    """
    Cache for voiceover analysis results.

    Key: MD5 hash of voiceover file content + config hash
    Value: AnalyzeResult with all extracted data

    Usage:
        cache = AnalyzeCache(project_dir / ".cache" / "analyze")

        # Check cache before running analysis
        result = cache.get_by_file(voiceover_path, config)
        if result:
            # Use cached result
            state.keywords = result.keywords
            ...
        else:
            # Run analysis
            ...
            # Cache result
            cache.set_by_file(voiceover_path, config, result)
    """

    # Config keys that affect analysis output
    CONFIG_KEYS = [
        'keyword.max_keywords',
        'matching.chapter_matching_enabled',
        'chapter_detection.enabled',
        'matching.location_matching.enabled',
    ]

    def __init__(
        self,
        cache_dir: Path | str,
        ttl_days: int = 30
    ):
        """
        Initialize analyze cache.

        Args:
            cache_dir: Directory for cache files
            ttl_days: Time-to-live in days (0 = no expiration)
        """
        super().__init__(
            cache_dir=cache_dir,
            index_name="analyze_index.json",
            ttl_seconds=ttl_days * 86400 if ttl_days > 0 else 0,
            auto_save=True
        )

    def _serialize_entry(self, entry: CacheEntry[AnalyzeResult]) -> Dict[str, Any]:
        """Serialize analyze result to JSON."""
        result = entry.data
        return {
            'cached_at': entry.cached_at,
            'key': entry.key,
            'metadata': entry.metadata,
            'keywords': result.keywords,
            'entities': result.entities,
            'topic_context': result.topic_context,
            'segments': result.segments,
            'chapters': result.chapters,
            'is_listicle': result.is_listicle,
            'location_chapters': result.location_chapters,
            'config_hash': result.config_hash,
            'file_hash': result.file_hash,
            'file_path': result.file_path,
            'file_size': result.file_size,
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry[AnalyzeResult]:
        """Deserialize JSON to analyze result."""
        result = AnalyzeResult(
            keywords=data.get('keywords', []),
            entities=data.get('entities', []),
            topic_context=data.get('topic_context', ''),
            segments=data.get('segments', []),
            chapters=data.get('chapters', []),
            is_listicle=data.get('is_listicle', False),
            location_chapters=data.get('location_chapters', []),
            config_hash=data.get('config_hash', ''),
            file_hash=data.get('file_hash', ''),
            file_path=data.get('file_path', ''),
            file_size=data.get('file_size', 0),
        )
        return CacheEntry(
            data=result,
            cached_at=data.get('cached_at', 0),
            key=data.get('key', ''),
            metadata=data.get('metadata', {})
        )

    @staticmethod
    def compute_file_hash(file_path: Path | str) -> str:
        """
        Compute MD5 hash of file content.

        Args:
            file_path: Path to file

        Returns:
            MD5 hex digest
        """
        path = Path(file_path)
        if not path.exists():
            return ""

        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_config_hash(config: Any) -> str:
        """
        Compute hash of config settings that affect analysis.

        Args:
            config: Config object

        Returns:
            MD5 hex digest of relevant config values
        """
        config_values = {}

        for key_path in AnalyzeCache.CONFIG_KEYS:
            parts = key_path.split('.')
            value = config
            try:
                for part in parts:
                    if hasattr(value, part):
                        value = getattr(value, part)
                    elif isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break
                if value is not None:
                    config_values[key_path] = str(value)
            except (AttributeError, KeyError, TypeError):
                pass

        # Sort for consistent hashing
        config_str = json.dumps(config_values, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def make_cache_key(self, file_path: Path | str, config: Any) -> str:
        """
        Create cache key from file hash and config hash.

        Args:
            file_path: Path to voiceover file
            config: Config object

        Returns:
            Cache key string
        """
        file_hash = self.compute_file_hash(file_path)
        config_hash = self.compute_config_hash(config)
        return f"{file_hash}_{config_hash}"

    def get_by_file(
        self,
        file_path: Path | str,
        config: Any
    ) -> Optional[AnalyzeResult]:
        """
        Get cached analysis for file.

        Args:
            file_path: Path to voiceover file
            config: Config object

        Returns:
            AnalyzeResult if cached and valid, None otherwise
        """
        cache_key = self.make_cache_key(file_path, config)
        entry = self.get(cache_key)

        if entry is None:
            return None

        # Validate file hash still matches (file hasn't changed)
        current_hash = self.compute_file_hash(file_path)
        if current_hash != entry.data.file_hash:
            logger.info(f"Analyze cache invalid: file content changed")
            self.delete(cache_key)
            return None

        logger.info(f"Analyze cache HIT: {Path(file_path).name} ({len(entry.data.keywords)} keywords)")
        return entry.data

    def set_by_file(
        self,
        file_path: Path | str,
        config: Any,
        result: AnalyzeResult
    ) -> None:
        """
        Cache analysis result for file.

        Args:
            file_path: Path to voiceover file
            config: Config object
            result: Analysis result to cache
        """
        path = Path(file_path)
        cache_key = self.make_cache_key(file_path, config)

        # Update result with file info
        result.file_hash = self.compute_file_hash(file_path)
        result.file_path = str(path)
        result.file_size = path.stat().st_size if path.exists() else 0
        result.config_hash = self.compute_config_hash(config)

        self.set(cache_key, result, metadata={
            'file_name': path.name,
            'keywords_count': len(result.keywords),
            'segments_count': len(result.segments),
            'chapters_count': len(result.chapters),
        })

        logger.info(f"Analyze cache SET: {path.name} ({len(result.keywords)} keywords, {len(result.chapters)} chapters)")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = super().get_stats()

        # Add analyze-specific stats
        total_keywords = 0
        total_segments = 0
        total_chapters = 0

        for key, entry_data in self.index.items():
            total_keywords += len(entry_data.get('keywords', []))
            total_segments += len(entry_data.get('segments', []))
            total_chapters += len(entry_data.get('chapters', []))

        stats.update({
            'total_keywords_cached': total_keywords,
            'total_segments_cached': total_segments,
            'total_chapters_cached': total_chapters,
        })

        return stats
