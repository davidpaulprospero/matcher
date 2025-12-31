"""
Match Index Module - Delta-Aware Matching Support

Tracks which videos have been matched and caches match results to enable
incremental matching. On subsequent runs, only new videos are processed.

Usage:
    from src.match_index import MatchAwareIndex

    index = MatchAwareIndex(project_dir)
    new_videos = index.get_new_videos(all_video_paths)

    if new_videos:
        # Process only new videos
        new_matches = match_new_videos(new_videos, ...)

        # Merge with cached matches
        all_matches = index.merge_matches(new_matches)

        # Save updated state
        index.save_matches(all_matches)
        index.mark_matched_batch(new_videos)
"""

import json
import logging
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class MatchedVideoInfo:
    """Information about a matched video"""
    video_path: str
    video_hash: str
    matched_at: float  # timestamp
    segment_count: int  # number of segments in this video

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchedVideoInfo":
        return cls(
            video_path=data.get('video_path', ''),
            video_hash=data.get('video_hash', ''),
            matched_at=data.get('matched_at', 0.0),
            segment_count=data.get('segment_count', 0)
        )


class MatchAwareIndex:
    """
    Tracks which videos have been matched to enable delta-aware processing.

    Features:
    - Tracks matched videos with their hashes (detect modifications)
    - Caches match results for reuse
    - Detects new, modified, and deleted videos
    - Supports voiceover change detection
    """

    VERSION = "1.0"

    def __init__(self, project_dir: str):
        """
        Initialize the match index.

        Args:
            project_dir: Project directory (match_index.json stored in .cache/)
        """
        self.project_dir = Path(project_dir)
        self.cache_dir = self.project_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.cache_dir / "match_index.json"
        self.matches_path = self.cache_dir / "cached_matches.json"

        # Index data
        self.matched_videos: Dict[str, MatchedVideoInfo] = {}
        self.voiceover_hash: str = ""
        self.config_hash: str = ""
        self.updated_at: float = 0.0

        self._load()

    def _load(self):
        """Load the match index from disk"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)

                # Load matched videos
                for path, info in data.get('matched_videos', {}).items():
                    self.matched_videos[path] = MatchedVideoInfo.from_dict(info)

                self.voiceover_hash = data.get('voiceover_hash', '')
                self.config_hash = data.get('config_hash', '')
                self.updated_at = data.get('updated_at', 0.0)

                logger.debug(f"Loaded match index with {len(self.matched_videos)} videos")

            except Exception as e:
                logger.warning(f"Could not load match index: {e}")
                self._reset()
        else:
            self._reset()

    def _reset(self):
        """Reset index to empty state"""
        self.matched_videos = {}
        self.voiceover_hash = ""
        self.config_hash = ""
        self.updated_at = 0.0

    def _save(self):
        """Save the match index to disk"""
        try:
            data = {
                'version': self.VERSION,
                'updated_at': time.time(),
                'voiceover_hash': self.voiceover_hash,
                'config_hash': self.config_hash,
                'matched_videos': {
                    path: info.to_dict()
                    for path, info in self.matched_videos.items()
                }
            }

            # Atomic write
            temp_path = self.index_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.index_path)

            self.updated_at = data['updated_at']

        except Exception as e:
            logger.error(f"Could not save match index: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        """Get fast hash based on path, size, and mtime"""
        try:
            p = Path(file_path)
            stat = p.stat()
            hash_input = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            return ""

    # =========================================================================
    # VIDEO TRACKING
    # =========================================================================

    def is_matched(self, video_path: str) -> bool:
        """Check if a video has been matched"""
        return str(video_path) in self.matched_videos

    def is_video_modified(self, video_path: str) -> bool:
        """Check if a previously matched video has been modified"""
        video_path = str(video_path)
        if video_path not in self.matched_videos:
            return False

        current_hash = self._get_file_hash(video_path)
        stored_hash = self.matched_videos[video_path].video_hash

        return current_hash != stored_hash

    def mark_matched(self, video_path: str, segment_count: int = 0):
        """Mark a video as matched"""
        video_path = str(video_path)
        self.matched_videos[video_path] = MatchedVideoInfo(
            video_path=video_path,
            video_hash=self._get_file_hash(video_path),
            matched_at=time.time(),
            segment_count=segment_count
        )
        self._save()

    def mark_matched_batch(self, video_paths: List[str], segment_counts: Dict[str, int] = None):
        """Mark multiple videos as matched"""
        segment_counts = segment_counts or {}

        for video_path in video_paths:
            video_path = str(video_path)
            self.matched_videos[video_path] = MatchedVideoInfo(
                video_path=video_path,
                video_hash=self._get_file_hash(video_path),
                matched_at=time.time(),
                segment_count=segment_counts.get(video_path, 0)
            )

        self._save()

    def get_new_videos(self, video_paths: List[str]) -> List[str]:
        """Get list of videos that haven't been matched yet"""
        return [vp for vp in video_paths if not self.is_matched(vp)]

    def get_modified_videos(self, video_paths: List[str]) -> List[str]:
        """Get list of previously matched videos that have been modified"""
        return [vp for vp in video_paths if self.is_video_modified(vp)]

    def get_deleted_videos(self, current_video_paths: List[str]) -> List[str]:
        """Get list of videos in index but no longer on disk"""
        current_set = set(str(vp) for vp in current_video_paths)
        return [
            vp for vp in self.matched_videos.keys()
            if vp not in current_set
        ]

    def remove_videos(self, video_paths: List[str]):
        """Remove videos from the index (e.g., when deleted)"""
        for vp in video_paths:
            self.matched_videos.pop(str(vp), None)
        self._save()

    # =========================================================================
    # VOICEOVER & CONFIG TRACKING
    # =========================================================================

    def set_voiceover_hash(self, voiceover_path: str):
        """Set the voiceover hash for change detection"""
        self.voiceover_hash = self._get_file_hash(voiceover_path)
        self._save()

    def is_voiceover_changed(self, voiceover_path: str) -> bool:
        """Check if voiceover has changed since last match"""
        if not self.voiceover_hash:
            return False  # First run

        current_hash = self._get_file_hash(voiceover_path)
        return current_hash != self.voiceover_hash

    def set_config_hash(self, config_hash: str):
        """Set config hash for change detection"""
        self.config_hash = config_hash
        self._save()

    def is_config_changed(self, config_hash: str) -> bool:
        """Check if matching config has changed"""
        if not self.config_hash:
            return False  # First run
        return config_hash != self.config_hash

    # =========================================================================
    # MATCH RESULT CACHING
    # =========================================================================

    def get_cached_matches(self) -> Optional[List[dict]]:
        """
        Load cached match results.

        Returns:
            List of match result dicts, or None if no cache exists
        """
        if not self.matches_path.exists():
            return None

        try:
            with open(self.matches_path, 'r') as f:
                data = json.load(f)
                return data.get('matches', [])
        except Exception as e:
            logger.warning(f"Could not load cached matches: {e}")
            return None

    def save_matches(self, matches: List[dict], voiceover_path: str = None):
        """
        Save match results to cache.

        Saves to two locations:
        1. cached_matches.json - for delta matching (gets overwritten)
        2. matches_YYYYMMDD_HHMMSS.json - timestamped archive (never overwritten)

        Args:
            matches: List of MatchResult dicts (from MatchResult.to_dict())
            voiceover_path: Optional voiceover path for metadata
        """
        from datetime import datetime

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = {
                'version': self.VERSION,
                'saved_at': time.time(),
                'timestamp': timestamp,
                'voiceover_path': str(voiceover_path) if voiceover_path else '',
                'voiceover_hash': self.voiceover_hash,
                'match_count': len(matches),
                'matches': matches
            }

            # Save to main cache file (for delta matching)
            temp_path = self.matches_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f)
            temp_path.replace(self.matches_path)

            # Also save timestamped version (never overwritten)
            timestamped_path = self.cache_dir / f"matches_{timestamp}.json"
            with open(timestamped_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(matches)} matches to cache")
            logger.info(f"Match results saved to: {timestamped_path.name}")

        except Exception as e:
            logger.error(f"Could not save matches to cache: {e}")

    def clear_matches(self):
        """Clear cached matches"""
        if self.matches_path.exists():
            self.matches_path.unlink()

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clear(self):
        """Clear the entire index (force full rematch)"""
        self._reset()
        self._save()
        self.clear_matches()
        logger.info("Match index cleared - will rematch all videos")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'matched_video_count': len(self.matched_videos),
            'has_cached_matches': self.matches_path.exists(),
            'voiceover_hash': self.voiceover_hash[:16] if self.voiceover_hash else None,
            'config_hash': self.config_hash[:16] if self.config_hash else None,
            'updated_at': self.updated_at
        }

    def __repr__(self) -> str:
        return f"MatchAwareIndex(videos={len(self.matched_videos)}, cached={self.matches_path.exists()})"
