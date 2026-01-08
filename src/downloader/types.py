"""
Dataclass types for downloader module.

Migrated from downloader.py lines 71-119.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MatchedSegment:
    """
    A segment from matching that needs video download.

    Migrated from downloader.py lines 72-83.

    Created during matching phase, before video segments are downloaded.
    """
    video_id: str
    video_url: str
    start_time: float     # Original timestamp in source video
    end_time: float       # Original end timestamp
    track: str            # "V1", "V4", etc.
    voiceover_segment_idx: int
    keyword: str = ""     # Source keyword (for organizing downloads)


@dataclass
class MergedSegment:
    """
    Merged segments ready for download.

    Migrated from downloader.py lines 87-98.

    Multiple close matches are merged with buffer applied to reduce
    download requests.
    """
    video_id: str
    video_url: str
    start_time: float     # After buffer applied
    end_time: float       # After buffer applied
    original_matches: List[MatchedSegment]  # Which matches this covers
    keyword: str = ""


@dataclass
class DownloadedSegment:
    """
    Downloaded video segment with timing info for OTIO mapping.

    Migrated from downloader.py lines 102-119.

    The file contains a portion of the original video, from original_start
    to original_end. To find a match within this file, calculate:
        offset = match.start_time - original_start
    """
    file: str             # Path to downloaded segment file
    video_id: str
    original_start: float # Start time in source video
    original_end: float   # End time in source video
    file_duration: float  # Actual file duration
    matches: List[MatchedSegment]  # Matches contained in this segment
    keyword: str = ""

    def get_offset(self, match_time: float) -> float:
        """Get offset within this file for a match timestamp."""
        return match_time - self.original_start
