"""
Clip and source reuse tracking for timeline variety.

Migrated from matching.py lines 41-216.
Provides two tracking mechanisms:
- TimelineVarietyTracker: Prevents source repetition within time windows
- GlobalClipTracker: Hard-blocks clips from reuse across entire timeline
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict

from ..utils import SRTSegment


class TimelineVarietyTracker:
    """
    Tracks source video usage per track to enforce timeline variety.

    Prevents the same source video from appearing more than max_repeats times
    within a timeline_window (e.g., 10 minutes).

    This solves the problem of one video dominating 90% of a 30-minute timeline.

    Performance: Uses sorted list with binary search for O(log n + k) lookups
    where k is the number of items in the window (typically small).
    """

    def __init__(self, timeline_window: float = 600.0, max_repeats: int = 1):
        """
        Args:
            timeline_window: Time window in seconds (default 600 = 10 minutes)
            max_repeats: Max times same source can appear in window (default 1)
        """
        self.timeline_window = timeline_window
        self.max_repeats = max_repeats

        # Track source usage per track: {track_name: [(timeline_position, source_file), ...]}
        # Sorted by position for binary search
        self.track_usage: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

    def get_excluded_sources(self, track: str, current_timeline_pos: float) -> Set[str]:
        """
        Get set of source files that should be excluded for this track at this position.

        Uses binary search to find window boundaries efficiently.

        Args:
            track: Track name (e.g., "V1", "V2", etc.)
            current_timeline_pos: Current position in timeline (seconds from start)

        Returns:
            Set of source file paths that should be excluded
        """
        usages = self.track_usage[track]
        if not usages:
            return set()

        window_start = current_timeline_pos - self.timeline_window

        # Binary search to find start of window
        # Find first index where position >= window_start
        left, right = 0, len(usages)
        while left < right:
            mid = (left + right) // 2
            if usages[mid][0] < window_start:
                left = mid + 1
            else:
                right = mid
        start_idx = left

        # Count occurrences of each source within the window
        source_counts: Dict[str, int] = defaultdict(int)

        for i in range(start_idx, len(usages)):
            pos, source_file = usages[i]
            if pos >= current_timeline_pos:
                break
            source_counts[source_file] += 1

        # Exclude sources that have reached max_repeats
        return {src for src, count in source_counts.items() if count >= self.max_repeats}

    def record_usage(self, track: str, source_file: str, timeline_pos: float):
        """
        Record that a source file was used on a track at a timeline position.

        Maintains sorted order for efficient lookups.

        Args:
            track: Track name (e.g., "V1", "V2", etc.)
            source_file: Path to the source video file
            timeline_pos: Position in timeline where this clip starts (seconds)
        """
        usages = self.track_usage[track]
        # Insert in sorted order (typically appending since timeline is sequential)
        # Use bisect for insertion point
        entry = (timeline_pos, source_file)
        if not usages or usages[-1][0] <= timeline_pos:
            # Fast path: append at end (most common case)
            usages.append(entry)
        else:
            # Binary search for insertion point
            left, right = 0, len(usages)
            while left < right:
                mid = (left + right) // 2
                if usages[mid][0] < timeline_pos:
                    left = mid + 1
                else:
                    right = mid
            usages.insert(left, entry)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about source usage per track."""
        stats = {}
        for track, usages in self.track_usage.items():
            source_counts = defaultdict(int)
            for _, source_file in usages:
                source_counts[Path(source_file).name] += 1

            stats[track] = {
                "total_clips": len(usages),
                "unique_sources": len(source_counts),
                "top_sources": sorted(source_counts.items(), key=lambda x: -x[1])[:5]
            }
        return stats


class GlobalClipTracker:
    """
    Hard-block clip reuse across entire timeline.

    Tracks all clips used across ALL segments and ALL tracks (V1-V7+).
    Prevents the same clip from ever appearing twice in the timeline.

    For audio-first segment files (e.g., abc12345678_0045.mp4), the clip ID
    is calculated using the original video coordinates to properly detect
    overlapping segments from the same source video.
    """

    def __init__(self):
        self.used_clips: Set[str] = set()
        self.clip_track_map: Dict[str, str] = {}  # clip_id -> "V1@S003"

    def get_clip_id(self, segment: SRTSegment) -> str:
        """
        Generate unique clip ID using ORIGINAL video coordinates.

        For audio-first segment files (e.g., abc12345678_0045.mp4):
        - Extract video ID from filename
        - Add file offset to segment times to get original coords

        Format: "{video_id}:{original_start:.2f}-{original_end:.2f}"
        """
        file_path = segment.source_file
        filename = Path(file_path).stem

        # Check for audio-first segment file pattern: {video_id}_{offset:04d}
        match = re.match(r'^([a-zA-Z0-9_-]{11})_(\d{4})$', filename)
        if match:
            video_id = match.group(1)
            file_offset = float(match.group(2))
            original_start = file_offset + segment.start_time
            original_end = file_offset + segment.end_time
            return f"{video_id}:{original_start:.2f}-{original_end:.2f}"

        # Fallback for regular video files
        path = file_path.replace('\\', '/').lower()
        return f"{path}:{segment.start_time:.2f}-{segment.end_time:.2f}"

    def is_used(self, segment: SRTSegment) -> bool:
        """Check if this exact clip has been used anywhere in the timeline."""
        return self.get_clip_id(segment) in self.used_clips

    def record_usage(self, segment: SRTSegment, track: str, segment_idx: int):
        """Record that a clip was used on a specific track."""
        clip_id = self.get_clip_id(segment)
        self.used_clips.add(clip_id)
        self.clip_track_map[clip_id] = f"{track}@S{segment_idx:03d}"

    def get_used_clips(self) -> Set[str]:
        """Get all used clip IDs for filtering."""
        return self.used_clips.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for logging."""
        return {
            "total_clips_used": len(self.used_clips),
            "tracks_used": len(set(v.split('@')[0] for v in self.clip_track_map.values()))
        }
