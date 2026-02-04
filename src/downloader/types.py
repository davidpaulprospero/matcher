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
