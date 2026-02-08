"""
Delta-aware indexing for transcription.

Tracks which videos have already been transcribed to avoid redundant work.
Uses set-based storage for fast membership checks.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Set, Tuple

from src.utils import normalize_path
from .utils import extract_video_id

logger = logging.getLogger(__name__)


class DeltaAwareIndex:
    """
    Track transcribed videos to enable delta indexing.

    Stores set of video paths and video IDs that have been successfully transcribed.
    Supports video ID matching for audio-first mode (segments match audio transcripts).
    """

    def __init__(self, cache_dir: str):
        """
        Initialize delta-aware index.

        Args:
            cache_dir: Cache directory for index storage
        """
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "delta_index.json"
        self.indexed_videos: Set[str] = set()
        self.indexed_video_ids: Set[str] = set()
        self._load()

    def _load(self):
        """Load the index of previously processed videos."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    # Normalize paths when loading for consistent matching
                    raw_paths = data.get('indexed', [])
                    self.indexed_videos = set(normalize_path(p) for p in raw_paths)
                    self.indexed_video_ids = set(data.get('indexed_ids', []))

                    # Rebuild video IDs from paths if not stored (migration)
                    if not self.indexed_video_ids:
                        for vp in raw_paths:
                            vid_id = extract_video_id(vp)
                            if vid_id:
                                self.indexed_video_ids.add(vid_id)
            except Exception as e:
                logger.debug(f"Could not load delta index: {e}")
                self.indexed_videos = set()
                self.indexed_video_ids = set()

    def _save(self):
        """Save the index to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, 'w') as f:
                json.dump({
                    'indexed': list(self.indexed_videos),
                    'indexed_ids': list(self.indexed_video_ids),
                    'updated_at': time.time()
                }, f)
        except Exception as e:
            logger.debug(f"Could not save delta index: {e}")

    def is_indexed(self, video_path: str) -> bool:
        """
        Check if a video has been indexed.

        Checks both normalized path and video ID (for segments).

        Args:
            video_path: Path to video file

        Returns:
            True if video has been indexed, False otherwise
        """
        # Normalize path for consistent matching
        normalized = normalize_path(video_path)

        # Check normalized path match
        if normalized in self.indexed_videos:
            return True

        # Check video ID match (for segment files matching audio)
        vid_id = extract_video_id(video_path)
        if vid_id and vid_id in self.indexed_video_ids:
            return True

        return False

    def mark_indexed(self, video_path: str):
        """
        Mark a video as indexed.

        Args:
            video_path: Path to video file
        """
        normalized = normalize_path(video_path)
        self.indexed_videos.add(normalized)
        vid_id = extract_video_id(video_path)
        if vid_id:
            self.indexed_video_ids.add(vid_id)
        self._save()

    def mark_indexed_batch(self, video_paths: List[str]):
        """
        Mark multiple videos as indexed.

        Args:
            video_paths: List of video paths
        """
        for vp in video_paths:
            normalized = normalize_path(vp)
            self.indexed_videos.add(normalized)
            vid_id = extract_video_id(vp)
            if vid_id:
                self.indexed_video_ids.add(vid_id)
        self._save()

    def get_new_videos(self, video_paths: List[str]) -> List[str]:
        """
        Get list of videos that haven't been indexed yet.

        Args:
            video_paths: List of video paths to check

        Returns:
            List of video paths that need indexing
        """
        return [vp for vp in video_paths if not self.is_indexed(vp)]

    def check_staleness(self, cache_entry_count: int) -> Tuple[bool, int, int]:
        """
        Check if the delta index is stale compared to actual cache entries.

        The index is considered stale if the cache has >10% more entries
        than the index tracks, suggesting videos were cached outside
        normal indexing (manual edits, interrupted writes, etc.).

        Args:
            cache_entry_count: Number of actual entries in the transcription cache

        Returns:
            Tuple of (is_stale, indexed_count, cache_count)
        """
        indexed_count = len(self.indexed_videos)
        is_stale = (
            cache_entry_count > 0
            and cache_entry_count > indexed_count * 1.1
        )
        return is_stale, indexed_count, cache_entry_count

    def rebuild_from_cache(self, cache_video_paths: List[str]) -> None:
        """
        Rebuild the delta index from actual cache entries.

        Args:
            cache_video_paths: List of video paths found in the cache
        """
        self.indexed_videos = set()
        self.indexed_video_ids = set()
        for vp in cache_video_paths:
            normalized = normalize_path(vp)
            self.indexed_videos.add(normalized)
            vid_id = extract_video_id(vp)
            if vid_id:
                self.indexed_video_ids.add(vid_id)
        self._save()
        logger.info(
            f"Delta index rebuilt: {len(self.indexed_videos)} videos, "
            f"{len(self.indexed_video_ids)} video IDs"
        )

    def clear(self):
        """Clear the index (force full reprocess)."""
        self.indexed_videos = set()
        self.indexed_video_ids = set()
        self._save()
