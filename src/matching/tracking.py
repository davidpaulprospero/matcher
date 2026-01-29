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

    def __init__(self, timeline_window: float = 600.0, max_repeats: int = 1) -> None:
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

    def record_usage(self, track: str, source_file: str, timeline_pos: float) -> None:
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
        """
        Get statistics about source usage per track.

        Returns:
            Dict mapping track names to statistics including total_clips,
            unique_sources, and top_sources list.
        """
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

    This tracker ensures that the same video segment is never used twice in
    the output timeline, across ALL voiceover segments and ALL tracks (V1-V7+).
    Unlike TimelineVarietyTracker which allows limited repeats within time
    windows, this provides absolute deduplication.

    Key Concepts:
        Clip ID: A unique identifier for a specific time range within a video.
            Format: "{video_source}:{start_time:.2f}-{end_time:.2f}"

        Audio-First Mode: When videos are pre-segmented (e.g., abc12345678_0045.mp4),
            the clip ID is calculated using ORIGINAL video coordinates, not the
            segment file's timestamps. This correctly detects overlapping segments.

        Used Clips Set: All recorded clips are stored in a set for O(1) lookup.
            Before selecting any clip, callers check is_used() to filter it out.

    Why This Matters:
        Without global tracking, the same compelling 5-second clip might be
        selected for multiple voiceover segments (e.g., V1@S001, V2@S005, V3@S010).
        This creates jarring repetition for viewers. GlobalClipTracker ensures
        each unique moment in the source footage appears at most once.

    Attributes:
        used_clips: Set of clip IDs that have been used in the timeline.
        clip_track_map: Maps clip_id to location string (e.g., "V1@S003") for
            debugging and statistics.
    """

    def __init__(self) -> None:
        """Initialize empty clip tracker with no used clips."""
        self.used_clips: Set[str] = set()
        self.clip_track_map: Dict[str, str] = {}  # clip_id -> "V1@S003"
        # Clip ID cache: segment object id -> clip_id string
        # Avoids recomputing clip IDs for same segment in tight matching loops
        self._clip_id_cache: Dict[int, str] = {}

    def get_clip_id(self, segment: SRTSegment) -> str:
        """
        Generate unique clip ID using ORIGINAL video coordinates with caching.

        Memoizes clip IDs by segment object id to avoid repeated regex matching
        and path operations in tight matching loops. Each segment is typically
        checked multiple times (once per strategy track).

        This method handles two distinct video file patterns:

        1. Audio-First Segment Files (e.g., abc12345678_0045.mp4):
           - Filename pattern: {11-char YouTube video ID}_{4-digit offset}
           - The offset indicates where this segment starts in the original video
           - Example: "dQw4w9WgXcQ_0045.mp4" means segment starts at 45 seconds
           - Clip ID calculation:
             * Extract video_id: "dQw4w9WgXcQ"
             * Extract file_offset: 45.0 seconds
             * original_start = file_offset + segment.start_time
             * original_end = file_offset + segment.end_time
           - Result: "dQw4w9WgXcQ:47.50-52.30"

        2. Regular Video Files (full downloads or other formats):
           - Uses normalized file path as the video identifier
           - Clip ID: "{normalized_path}:{start:.2f}-{end:.2f}"
           - Path is lowercased and backslashes converted to forward slashes

        Why Original Coordinates Matter:
            Consider segment files "xyz_0030.mp4" and "xyz_0040.mp4".
            If segment.start_time=5, segment.end_time=10 for both:
            - Without original coords: both would be "xyz:5.00-10.00" (collision!)
            - With original coords: "xyz:35.00-40.00" and "xyz:45.00-50.00"
            This correctly identifies them as different clips.

        Args:
            segment: SRTSegment containing source_file path and start/end times.

        Returns:
            Unique string identifier for this clip in format:
            "{source}:{start_time:.2f}-{end_time:.2f}"

        Notes:
            The regex pattern expects exactly 11 alphanumeric characters for the
            video ID (standard YouTube format) followed by underscore and 4 digits.
        """
        # Check cache first (uses object id as key for O(1) lookup)
        seg_id = id(segment)
        if seg_id in self._clip_id_cache:
            return self._clip_id_cache[seg_id]

        file_path = segment.source_file
        filename = Path(file_path).stem

        # Check for audio-first segment file pattern: {video_id}_{offset:04d}
        match = re.match(r'^([a-zA-Z0-9_-]{11})_(\d{4})$', filename)
        if match:
            video_id = match.group(1)
            file_offset = float(match.group(2))
            original_start = file_offset + segment.start_time
            original_end = file_offset + segment.end_time
            clip_id = f"{video_id}:{original_start:.2f}-{original_end:.2f}"
        else:
            # Fallback for regular video files
            path = file_path.replace('\\', '/').lower()
            clip_id = f"{path}:{segment.start_time:.2f}-{segment.end_time:.2f}"

        # Cache and return
        self._clip_id_cache[seg_id] = clip_id
        return clip_id

    def is_used(self, segment: SRTSegment) -> bool:
        """
        Check if this exact clip has been used anywhere in the timeline.

        This is the primary filter method used during matching. Before selecting
        any candidate clip for a voiceover segment, callers should check is_used()
        to prevent duplicate selections.

        Typical Usage in StrategyMatcher:
            for candidate in candidates:
                if global_clip_tracker.is_used(candidate):
                    continue  # Skip - already used elsewhere
                # ... score and potentially select candidate

        Performance:
            O(1) lookup via set membership test. The clip ID is computed
            each time (involves regex match and string formatting), but this
            is negligible compared to other matching operations.

        Args:
            segment: The SRT segment to check, containing source_file and
                start_time/end_time. Note: the segment's text content is
                ignored - only the video coordinates matter for deduplication.

        Returns:
            True if this exact clip (same video source and time range) has
            been recorded via record_usage(). False if the clip is available.
        """
        return self.get_clip_id(segment) in self.used_clips

    def record_usage(self, segment: SRTSegment, track: str, segment_idx: int) -> None:
        """
        Record that a clip was used on a specific track.

        After a clip is selected for the timeline, this method must be called
        to mark it as used. Subsequent calls to is_used() for the same clip
        will then return True, preventing reuse.

        Recording Process:
            1. Generate clip_id via get_clip_id()
            2. Add to used_clips set (enables O(1) lookup)
            3. Store location in clip_track_map (for debugging/stats)

        Location Format:
            clip_track_map stores "{track}@S{segment_idx:03d}" format.
            Example: "V1@S003" means the clip is used on V1 for voiceover
            segment 3. This helps debugging when investigating why certain
            clips were excluded.

        Args:
            segment: The SRT segment containing the clip's video source and
                time coordinates.
            track: Track name where the clip will appear (e.g., "V1", "V2",
                "V3", "V4", "V5", "V6", "V7" for embedding-diversity, "V8"
                for B-roll only).
            segment_idx: Zero-based index of the voiceover segment this clip
                is matched to. Used for the location string.

        Notes:
            This method should only be called once per clip. Calling it
            multiple times with the same segment is safe (set.add is
            idempotent) but would overwrite the clip_track_map entry.
        """
        clip_id = self.get_clip_id(segment)
        self.used_clips.add(clip_id)
        self.clip_track_map[clip_id] = f"{track}@S{segment_idx:03d}"

    def get_used_clips(self) -> Set[str]:
        """
        Get all used clip IDs for bulk filtering.

        Returns a copy of the used_clips set for external filtering operations.
        This is useful when multiple components need to filter candidates
        without having access to the SRTSegment objects.

        Use Case - StrategyMatcher:
            When get_strategy_matches() is called, it passes get_used_clips()
            to filtering methods so they can quickly check membership without
            repeatedly calling is_used() with segment objects.

        Returns:
            A copy of the used_clips set. Modifications to the returned set
            do not affect the tracker's internal state.

        Notes:
            Returns a copy to prevent external code from accidentally
            modifying the tracker's state. The copy operation is O(n) but
            typically the set contains at most thousands of clips.
        """
        return self.used_clips.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about clip usage for logging and debugging.

        Provides a summary of how many unique clips were used and across
        how many tracks. Useful for pipeline logging and understanding
        timeline diversity.

        Example Output:
            {
                "total_clips_used": 145,
                "tracks_used": 6
            }

        Interpretation:
            - total_clips_used: Number of unique video segments in the timeline.
              Higher numbers indicate more variety.
            - tracks_used: Number of distinct tracks (V1, V2, etc.) that have
              clips. If only 1 track is used, alternatives weren't generated.

        Returns:
            Dict containing:
            - "total_clips_used" (int): Count of unique clips across all tracks
            - "tracks_used" (int): Count of distinct track names (e.g., V1, V2)

        Notes:
            The tracks_used count is derived from clip_track_map by extracting
            the track name before the "@" in location strings like "V1@S003".
        """
        return {
            "total_clips_used": len(self.used_clips),
            "tracks_used": len(set(v.split('@')[0] for v in self.clip_track_map.values()))
        }
