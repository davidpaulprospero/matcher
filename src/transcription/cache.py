"""
Transcript caching with multi-strategy lookup.

Provides caching for transcription results with multiple lookup strategies:
- Source file path matching
- Video ID matching (for segment files)
- Hash-based matching (path + size)

NOTE: Does NOT use BaseCache due to complex multi-strategy lookup
and dynamic index building by scanning cache directory.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional

from src.utils import normalize_path
from .utils import extract_video_id

logger = logging.getLogger(__name__)


class TranscriptCache:
    """
    Multi-strategy transcript cache.

    Lookup strategies:
    1. Source file path (most reliable)
    2. Video ID matching (for segment files in audio-first mode)
    3. Hash-based (path + size fallback)
    """

    def __init__(self, cache_dir: str):
        """
        Initialize transcript cache.

        Args:
            cache_dir: Base cache directory (will use transcriptions/ subdirectory)
        """
        base_dir = Path(cache_dir)

        # Check both possible folder names
        self.cache_dir = base_dir / "transcriptions"
        self.alt_cache_dir = base_dir / "transcripts"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_map: Dict[str, Path] = {}
        self._video_id_map: Dict[str, Path] = {}

        logger.debug(f"Primary cache: {self.cache_dir}")
        logger.debug(f"Alt cache: {self.alt_cache_dir}")

        # Build reverse lookup by reading source_file from each cache file
        self._build_source_map()

    def _build_source_map(self):
        """
        Build mapping from video path to cache file.

        Reads source_file from each cached transcript to build index.
        """
        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue

            cache_files = list(folder.glob("*.json"))
            logger.debug(f"Scanning {len(cache_files)} cache files in {folder.name}...")

            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract source_file from the data
                    source_file = None

                    if isinstance(data, list) and len(data) > 0:
                        # List of segments - get source_file from first segment
                        source_file = data[0].get('source_file', '')
                    elif isinstance(data, dict):
                        # Dict format - check various keys
                        source_file = data.get('source_file', data.get('video', data.get('video_path', '')))
                        if not source_file and 'segments' in data:
                            segs = data['segments']
                            if segs and len(segs) > 0:
                                source_file = segs[0].get('source_file', '')

                    if source_file:
                        # Normalize path for matching (consistent across platforms)
                        normalized = normalize_path(source_file)
                        self._source_map[normalized] = cache_file

                        # Also add just the filename as key for partial matching
                        filename = Path(source_file).name.lower()
                        self._source_map[filename] = cache_file

                        # Extract video ID for segment matching (audio-first mode support)
                        video_id = extract_video_id(filename)
                        if video_id and video_id not in self._video_id_map:
                            self._video_id_map[video_id] = cache_file

                except Exception as e:
                    continue

        logger.debug(f"Built source map with {len(self._source_map)} entries, {len(self._video_id_map)} video IDs")

    def _get_video_hash(self, video_path: str) -> str:
        """
        Get hash for a video file based on path and size.

        Args:
            video_path: Path to video file

        Returns:
            MD5 hash string
        """
        path = Path(video_path)
        size = path.stat().st_size if path.exists() else 0
        key = f"{path.name}:{size}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, video_path: str) -> Optional[List[dict]]:
        """
        Get cached transcript for a video.

        Tries multiple strategies:
        1. Source path matching (normalized)
        2. Filename matching
        3. Video ID matching (for segments)
        4. Hash-based matching (fallback)

        Args:
            video_path: Path to video file

        Returns:
            List of segment dicts with 'start', 'end', 'text', or None if not cached
        """
        # Normalize path for consistent matching
        normalized_path = normalize_path(video_path)
        video_name = Path(video_path).name.lower()

        # Try source map lookup first (most reliable)
        cache_file = None
        if normalized_path in self._source_map:
            cache_file = self._source_map[normalized_path]
        elif video_name in self._source_map:
            cache_file = self._source_map[video_name]

        # Fallback to video ID lookup (for segment files matching audio transcripts)
        if not cache_file:
            video_id = extract_video_id(video_name)
            if video_id and video_id in self._video_id_map:
                cache_file = self._video_id_map[video_id]

        # Fallback to hash-based lookup
        if not cache_file:
            video_hash = self._get_video_hash(video_path)
            for folder in [self.cache_dir, self.alt_cache_dir]:
                if folder and folder.exists():
                    exact = folder / f"{video_hash}.json"
                    if exact.exists():
                        cache_file = exact
                        break

        if not cache_file or not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different cache formats
            if isinstance(data, list):
                segments = data
            elif isinstance(data, dict):
                segments = data.get('segments', data.get('transcripts', []))
            else:
                return None

            # Normalize segment format
            normalized = []
            for seg in segments:
                if isinstance(seg, dict):
                    normalized.append({
                        'start': seg.get('start', seg.get('start_time', 0)),
                        'end': seg.get('end', seg.get('end_time', 0)),
                        'text': seg.get('text', '')
                    })

            return normalized if normalized else None

        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None

    def set(self, video_path: str, segments: List[dict]):
        """
        Cache transcript for a video.

        Args:
            video_path: Path to video file
            segments: List of segment dicts with 'start', 'end', 'text'
        """
        video_hash = self._get_video_hash(video_path)
        cache_file = self.cache_dir / f"{video_hash}.json"

        # Convert to format matching existing cache
        cache_data = []
        for i, seg in enumerate(segments):
            cache_data.append({
                'index': i + 1,
                'start_time': seg.get('start', seg.get('start_time', 0)),
                'end_time': seg.get('end', seg.get('end_time', 0)),
                'text': seg.get('text', ''),
                'source_file': str(video_path)
            })

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)

            # Update source map
            video_path_resolved = str(Path(video_path).resolve())
            self._source_map[video_path_resolved] = cache_file
            self._source_map[Path(video_path).name] = cache_file

        except Exception as e:
            logger.debug(f"Could not cache transcript: {e}")
