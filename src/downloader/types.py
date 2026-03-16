"""
Dataclass types for downloader module.

Migrated from downloader.py lines 71-119.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple


class DownloadError(Exception):
    """
    Base exception for download errors with retry context.

    Carries information about retry attempts so healers can make
    informed decisions about escalation (skip redundant backoff
    if retries are already exhausted).

    Attributes:
        message: Error description
        retry_count: Number of retries already attempted (0 = no retries yet)
        max_retries: Maximum retries that were configured
        error_type: Category of error ('transient', 'permanent', 'timeout', 'unknown')
        original_error: The underlying error message from yt-dlp
    """

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        max_retries: int = 3,
        error_type: str = 'unknown',
        original_error: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.error_type = error_type
        self.original_error = original_error or message

    @property
    def retries_exhausted(self) -> bool:
        """Check if all retry attempts have been used."""
        return self.retry_count >= self.max_retries

    def __str__(self) -> str:
        if self.retry_count > 0:
            return f"{self.message} (retried {self.retry_count}/{self.max_retries})"
        return self.message


class EscalationTier(IntEnum):
    """4-tier escalation levels for yt-dlp bypass."""
    IMPERSONATE_ONLY = 1
    EXTRACTOR_ARGS = 2
    FULL_BYPASS = 3
    VPN_ROTATION = 4


@dataclass
class EscalationState:
    """Per-keyword escalation progression state.

    Tracks consecutive 403 errors and manages tier advancement
    for the 4-tier bypass system. Also tracks consecutive successes
    for de-escalation.
    """
    current_tier: EscalationTier = EscalationTier.IMPERSONATE_ONLY
    consecutive_403s: int = 0
    consecutive_successes: int = 0
    last_escalation_time: Optional[float] = None
    extractor_args_index: int = 0
    escalation_history: List[Tuple[float, EscalationTier]] = field(default_factory=list)

    def should_escalate(self, threshold: int = 2) -> bool:
        """Check if consecutive 403s have reached the escalation threshold."""
        return self.consecutive_403s >= threshold

    def escalate(self) -> None:
        """Advance to the next tier and reset the 403 counter."""
        if self.current_tier < EscalationTier.VPN_ROTATION:
            self.current_tier = EscalationTier(self.current_tier + 1)
        now = time.time()
        self.last_escalation_time = now
        self.escalation_history.append((now, self.current_tier))
        self.consecutive_403s = 0
        self.consecutive_successes = 0

    def record_success(
        self,
        de_escalation_threshold: int = 5,
        de_escalation_enabled: bool = True,
    ) -> bool:
        """Record a success, potentially de-escalating on sustained success.

        Resets the 403 counter and increments consecutive_successes.
        If consecutive_successes reaches de_escalation_threshold and
        de_escalation is enabled, decreases tier by 1.

        Args:
            de_escalation_threshold: Number of consecutive successes needed
                to trigger de-escalation. Default: 5.
            de_escalation_enabled: Whether de-escalation is enabled. Default: True.

        Returns:
            True if de-escalation occurred, False otherwise.
        """
        self.consecutive_403s = 0
        self.consecutive_successes += 1

        if (
            de_escalation_enabled
            and self.consecutive_successes >= de_escalation_threshold
            and self.current_tier > EscalationTier.IMPERSONATE_ONLY
        ):
            self.current_tier = EscalationTier(self.current_tier - 1)
            self.consecutive_successes = 0
            return True
        return False


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


@dataclass
class SegmentDownloadProgress:
    """Real-time download progress for a video segment.

    Parsed from yt-dlp progress output during segment download.
    Provides percent complete, bytes downloaded, and download speed.

    Attributes:
        video_id: YouTube video ID being downloaded.
        percent: Download progress as percentage (0.0 to 100.0).
        bytes_downloaded: Total bytes downloaded so far.
        speed: Download speed in bytes per second (may be 0 if unknown).
        eta_seconds: Estimated seconds remaining (may be None if unknown).
    """
    video_id: str
    percent: float
    bytes_downloaded: int
    speed: float
    eta_seconds: Optional[float] = None


import re

# Pre-compiled regex for yt-dlp progress parsing
# Matches lines like: [download]  50.0% of 10.00MiB at  5.00MiB/s ETA 00:01
_YTDLP_PROGRESS_PATTERN = re.compile(
    r'\[download\]\s+(\d+\.?\d*)%\s+of\s+~?(\d+\.?\d*)(Ki?B|Mi?B|Gi?B|B)\s+'
    r'at\s+(\d+\.?\d*)(Ki?B|Mi?B|Gi?B|B)/s(?:\s+ETA\s+(\d+:\d+(?::\d+)?))?',
    re.IGNORECASE
)


def _convert_size_to_bytes(value: float, unit: str) -> int:
    """Convert size value with unit to bytes."""
    unit_lower = unit.lower()
    if unit_lower in ('kib', 'kb'):
        return int(value * 1024)
    elif unit_lower in ('mib', 'mb'):
        return int(value * 1024 * 1024)
    elif unit_lower in ('gib', 'gb'):
        return int(value * 1024 * 1024 * 1024)
    return int(value)


def _parse_eta_to_seconds(eta_str: str) -> Optional[float]:
    """Parse ETA string (HH:MM:SS or MM:SS) to seconds."""
    if not eta_str:
        return None
    parts = eta_str.split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return None


def parse_ytdlp_progress(line: str, video_id: str) -> Optional[SegmentDownloadProgress]:
    """Parse a yt-dlp progress line into SegmentDownloadProgress.

    Args:
        line: A line from yt-dlp stderr output.
        video_id: The video ID being downloaded.

    Returns:
        SegmentDownloadProgress if line is a progress line, None otherwise.
    """
    match = _YTDLP_PROGRESS_PATTERN.search(line)
    if not match:
        return None

    percent = float(match.group(1))
    size_value = float(match.group(2))
    size_unit = match.group(3)
    speed_value = float(match.group(4))
    speed_unit = match.group(5)
    eta_str = match.group(6)

    # Calculate bytes downloaded from percent and total size
    total_bytes = _convert_size_to_bytes(size_value, size_unit)
    bytes_downloaded = int(total_bytes * percent / 100.0) if percent > 0 else 0
    speed = _convert_size_to_bytes(speed_value, speed_unit)
    eta_seconds = _parse_eta_to_seconds(eta_str) if eta_str else None

    return SegmentDownloadProgress(
        video_id=video_id,
        percent=percent,
        bytes_downloaded=bytes_downloaded,
        speed=float(speed),
        eta_seconds=eta_seconds
    )
