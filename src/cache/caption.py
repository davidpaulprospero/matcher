"""
Caption cache for caption-first pipeline mode.

Caches downloaded caption files and their metadata to avoid
redundant downloads and provide consistent access patterns.

Part of cache consolidation - extends BaseCache for unified behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseCache, CacheEntry

logger = logging.getLogger(__name__)


@dataclass
class CaptionCacheEntry:
    """Cached caption metadata"""
    video_id: str
    file: str  # Path to .srt/.vtt file
    language: str
    is_auto_generated: bool
    format: str = "srt"
    segment_count: int = 0
    duration_covered: float = 0.0  # Total caption duration
    cached_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'video_id': self.video_id,
            'file': self.file,
            'language': self.language,
            'is_auto_generated': self.is_auto_generated,
            'format': self.format,
            'segment_count': self.segment_count,
            'duration_covered': self.duration_covered,
            'cached_at': self.cached_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaptionCacheEntry':
        return cls(
            video_id=data.get('video_id', ''),
            file=data.get('file', ''),
            language=data.get('language', 'en'),
            is_auto_generated=data.get('is_auto_generated', True),
            format=data.get('format', 'srt'),
            segment_count=data.get('segment_count', 0),
            duration_covered=data.get('duration_covered', 0.0),
            cached_at=data.get('cached_at', 0.0),
        )


class CaptionCache(BaseCache[CaptionCacheEntry]):
    """
    Cache for downloaded YouTube captions.

    Features:
    - Stores metadata for each video's caption file
    - Validates caption file existence on retrieval
    - Supports TTL for cache expiration
    - Tracks caption statistics (language, auto-generated, coverage)

    Usage:
        cache = CaptionCache(cache_dir=".cache/captions")

        # Store caption metadata
        cache.set("video_id", CaptionCacheEntry(...))

        # Retrieve
        entry = cache.get("video_id")
        if entry:
            # entry.data is CaptionCacheEntry
            print(f"Caption file: {entry.data.file}")

        # Check existence
        if cache.has_valid_caption("video_id"):
            ...
    """

    def __init__(
        self,
        cache_dir: Path | str,
        ttl_hours: int = 0,  # No expiration by default
        auto_save: bool = True
    ):
        """
        Initialize caption cache.

        Args:
            cache_dir: Directory for caption files and index
            ttl_hours: Hours until cache entries expire (0 = no expiration)
            auto_save: Auto-save index after modifications
        """
        super().__init__(
            cache_dir=cache_dir,
            index_name="caption_index.json",
            ttl_seconds=ttl_hours * 3600 if ttl_hours > 0 else 0,
            auto_save=auto_save
        )

    def _serialize_entry(self, entry: CacheEntry[CaptionCacheEntry]) -> Dict[str, Any]:
        """Serialize cache entry for JSON storage"""
        data = entry.data
        if isinstance(data, CaptionCacheEntry):
            data_dict = data.to_dict()
        elif isinstance(data, dict):
            data_dict = data
        else:
            data_dict = {}

        return {
            'cached_at': entry.cached_at,
            'key': entry.key,
            'metadata': entry.metadata,
            **data_dict
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry[CaptionCacheEntry]:
        """Deserialize JSON data to cache entry"""
        caption_entry = CaptionCacheEntry.from_dict(data)
        caption_entry.cached_at = data.get('cached_at', 0.0)

        return CacheEntry(
            data=caption_entry,
            cached_at=data.get('cached_at', 0.0),
            key=data.get('key', data.get('video_id', '')),
            metadata=data.get('metadata', {})
        )

    def _is_valid_entry(self, entry: CacheEntry[CaptionCacheEntry]) -> bool:
        """
        Validate cache entry - check TTL and file existence.

        Args:
            entry: Cache entry to validate

        Returns:
            True if valid (not expired and file exists)
        """
        # Check TTL first
        if not super()._is_valid_entry(entry):
            return False

        # Check if caption file still exists
        if entry.data and hasattr(entry.data, 'file'):
            file_path = Path(entry.data.file)
            if not file_path.exists():
                logger.debug(f"Caption file missing: {file_path}")
                return False

        return True

    def has_valid_caption(self, video_id: str) -> bool:
        """
        Check if a valid caption exists for the video.

        Args:
            video_id: YouTube video ID

        Returns:
            True if valid caption cached and file exists
        """
        entry = self.get(video_id)
        return entry is not None

    def get_caption_file(self, video_id: str) -> Optional[str]:
        """
        Get caption file path for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Path to caption file or None
        """
        entry = self.get(video_id)
        if entry and entry.data:
            return entry.data.file
        return None

    def get_caption_info(self, video_id: str) -> Optional[CaptionCacheEntry]:
        """
        Get full caption info for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            CaptionCacheEntry or None
        """
        entry = self.get(video_id)
        if entry:
            return entry.data
        return None

    def set_caption(
        self,
        video_id: str,
        file: str,
        language: str,
        is_auto_generated: bool,
        format: str = "srt",
        segment_count: int = 0,
        duration_covered: float = 0.0
    ) -> None:
        """
        Store caption metadata.

        Args:
            video_id: YouTube video ID
            file: Path to caption file
            language: Language code (e.g., "en", "en-US")
            is_auto_generated: True if auto-generated caption
            format: Caption format (srt, vtt)
            segment_count: Number of caption segments
            duration_covered: Total duration covered by captions
        """
        entry = CaptionCacheEntry(
            video_id=video_id,
            file=file,
            language=language,
            is_auto_generated=is_auto_generated,
            format=format,
            segment_count=segment_count,
            duration_covered=duration_covered,
        )
        self.set(video_id, entry)
        logger.debug(f"Cached caption for {video_id}: {language}, {segment_count} segments")

    def find_caption_by_language(
        self,
        video_id: str,
        languages: List[str]
    ) -> Optional[CaptionCacheEntry]:
        """
        Find cached caption matching one of the preferred languages.

        Args:
            video_id: YouTube video ID
            languages: List of preferred language codes

        Returns:
            CaptionCacheEntry if found for any language, None otherwise
        """
        entry = self.get(video_id)
        if not entry or not entry.data:
            return None

        # Check if cached language matches any preferred
        caption_lang = entry.data.language
        for lang in languages:
            # Match exact or prefix (e.g., "en" matches "en-US")
            if caption_lang == lang or caption_lang.startswith(lang):
                return entry.data

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics
        """
        stats = self.get_stats()

        # Count by type
        manual_count = 0
        auto_count = 0
        total_segments = 0
        total_duration = 0.0

        for key in self.index:
            entry = self.get(key)
            if entry and entry.data:
                if entry.data.is_auto_generated:
                    auto_count += 1
                else:
                    manual_count += 1
                total_segments += entry.data.segment_count
                total_duration += entry.data.duration_covered

        stats.update({
            'manual_captions': manual_count,
            'auto_captions': auto_count,
            'total_segments': total_segments,
            'total_duration_hours': total_duration / 3600,
        })

        return stats

    def cleanup_orphaned_files(self) -> int:
        """
        Remove caption files that are not in the index.

        Returns:
            Number of orphaned files removed
        """
        removed = 0

        # Get indexed video IDs
        indexed_ids = set(self.index.keys())

        # Scan directory for caption files
        for file_path in self.cache_dir.glob("*.srt"):
            # Extract video ID from filename (assume format: video_id.lang.srt)
            parts = file_path.stem.split(".")
            if parts:
                video_id = parts[0]
                if video_id not in indexed_ids:
                    try:
                        file_path.unlink()
                        removed += 1
                        logger.debug(f"Removed orphaned caption: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")

        # Also check .vtt files
        for file_path in self.cache_dir.glob("*.vtt"):
            parts = file_path.stem.split(".")
            if parts:
                video_id = parts[0]
                if video_id not in indexed_ids:
                    try:
                        file_path.unlink()
                        removed += 1
                        logger.debug(f"Removed orphaned caption: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")

        if removed > 0:
            logger.info(f"Cleaned up {removed} orphaned caption files")

        return removed

    def migrate_from_files(self) -> int:
        """
        Migrate existing caption files to indexed cache.

        Scans cache directory for caption files and adds them to the index.
        Used to migrate from ad-hoc file-based caching to indexed caching.

        Returns:
            Number of files migrated
        """
        migrated = 0

        # Find caption files not in index
        for file_path in self.cache_dir.glob("*.srt"):
            parts = file_path.stem.split(".")
            if len(parts) >= 2:
                video_id = parts[0]
                language = parts[1] if len(parts) > 1 else "en"
                is_auto = "-auto" in file_path.name or ".auto." in file_path.name

                if video_id not in self.index:
                    # Count segments by reading file
                    segment_count = 0
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        # Count segment separators
                        segment_count = content.count('\n\n')
                    except Exception:
                        pass

                    self.set_caption(
                        video_id=video_id,
                        file=str(file_path),
                        language=language.replace("-auto", "").replace("-orig", ""),
                        is_auto_generated=is_auto,
                        format="srt",
                        segment_count=segment_count,
                    )
                    migrated += 1
                    logger.debug(f"Migrated caption: {video_id} ({language})")

        if migrated > 0:
            logger.info(f"Migrated {migrated} caption files to indexed cache")

        return migrated
