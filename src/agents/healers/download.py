"""
Download Healer - Video/audio download error recovery.

Handles:
- YouTube rate limiting (429)
- Video unavailable
- Format extraction failures
- Network timeouts
- Partial downloads
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Set

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class DownloadHealer(Healer):
    """
    Heals download-related errors.

    Recovery strategies:
    1. Rate limit: Exponential backoff
    2. Video unavailable: Skip and continue
    3. Format error: Try alternate format
    4. Network timeout: Retry with longer timeout
    5. Partial download: Resume or restart
    """

    name = "download-healer"
    description = "Fix video download errors"

    error_patterns = [
        "yt-dlp",
        "youtube",
        "download",
        "429",
        "rate limit",
        "unavailable",
        "private",
        "removed",
        "format",
        "extract",
        "timeout",
        "connection",
        "incomplete",
        "partial",
    ]

    # Backoff configuration
    INITIAL_BACKOFF = 10.0
    MAX_BACKOFF = 600.0  # 10 minutes
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0
        self.skipped_videos: Set[str] = set()

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix download-related errors."""
        error_str = str(error).lower()

        # Rate limiting
        if any(p in error_str for p in ["429", "rate limit", "too many"]):
            return self._handle_rate_limit(error, state)

        # Video unavailable
        if any(p in error_str for p in ["unavailable", "private", "removed", "not found"]):
            return self._handle_unavailable(error, state)

        # Format extraction error
        if any(p in error_str for p in ["format", "extract", "no video"]):
            return self._handle_format_error(error, state)

        # Network/timeout errors
        if any(p in error_str for p in ["timeout", "connection", "network"]):
            return self._handle_network_error(error, state)

        # Partial/incomplete download
        if any(p in error_str for p in ["incomplete", "partial", "corrupt"]):
            return self._handle_incomplete(error, state)

        # Generic download error - try with backoff
        return self._handle_generic_error(error, state)

    def _handle_rate_limit(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle YouTube rate limiting."""
        self.log_attempt(f"Rate limited by YouTube, waiting {self.backoff_time:.0f}s...")

        time.sleep(self.backoff_time)

        old_backoff = self.backoff_time
        self.backoff_time = min(self.backoff_time * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
        self.retry_count += 1

        self.log_success(f"Waited {old_backoff:.0f}s, resuming downloads")

        return HealerResult.fixed(
            f"YouTube rate limit: waited {old_backoff:.0f}s",
            action=HealerAction.RETRY,
            backoff_seconds=old_backoff,
            retry_count=self.retry_count
        )

    def _handle_unavailable(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle unavailable video by skipping."""
        self.log_attempt("Video unavailable, marking to skip...")

        # Extract video ID from error if possible
        video_id = self._extract_video_id(str(error))

        if video_id:
            self.skipped_videos.add(video_id)
            self.log_success(f"Skipping unavailable video: {video_id}")
            return HealerResult.fixed(
                f"Skipped unavailable video: {video_id}",
                action=HealerAction.SKIP,
                skipped_video=video_id,
                total_skipped=len(self.skipped_videos)
            )

        # Can't identify video - skip stage and continue
        return HealerResult.fixed(
            "Video unavailable, continuing with remaining videos",
            action=HealerAction.RETRY,
            skip_current=True
        )

    def _handle_format_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle format extraction errors by trying alternate formats."""
        self.log_attempt("Format extraction failed, trying alternate formats...")

        download_config = getattr(self.config, 'download', None)
        if not download_config:
            return HealerResult.failed("No download config available")

        # Try progressively simpler format strings
        format_fallbacks = [
            "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "bestvideo+bestaudio/best",
            "best",
        ]

        current_format = getattr(download_config, 'format', None)

        try:
            current_idx = format_fallbacks.index(current_format) if current_format else -1
        except ValueError:
            current_idx = -1

        next_idx = current_idx + 1
        if next_idx >= len(format_fallbacks):
            return HealerResult.failed("All format options exhausted")

        new_format = format_fallbacks[next_idx]

        if hasattr(download_config, 'format'):
            download_config.format = new_format
        elif isinstance(download_config, dict):
            download_config['format'] = new_format

        self.log_success(f"Switched format: {current_format} -> {new_format}")
        return HealerResult.config_changed(
            f"Switched to simpler format: {new_format}",
            old_format=current_format,
            new_format=new_format
        )

    def _handle_network_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle network/timeout errors."""
        self.log_attempt("Network error, waiting before retry...")

        time.sleep(self.INITIAL_BACKOFF)

        # Try increasing socket timeout
        download_config = getattr(self.config, 'download', None)
        if download_config:
            current_timeout = getattr(download_config, 'socket_timeout', 30)
            new_timeout = min(current_timeout * 2, 120)

            if hasattr(download_config, 'socket_timeout'):
                download_config.socket_timeout = new_timeout
            elif isinstance(download_config, dict):
                download_config['socket_timeout'] = new_timeout

            self.log_success(f"Increased socket timeout: {current_timeout}s -> {new_timeout}s")
            return HealerResult.config_changed(
                f"Network error, increased timeout to {new_timeout}s",
                old_timeout=current_timeout,
                new_timeout=new_timeout
            )

        return HealerResult.fixed("Network error, retrying", action=HealerAction.RETRY)

    def _handle_incomplete(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle incomplete/partial downloads."""
        self.log_attempt("Incomplete download detected...")

        # Enable resume in yt-dlp config
        download_config = getattr(self.config, 'download', None)
        if download_config:
            if hasattr(download_config, 'continue_dl'):
                download_config.continue_dl = True
            elif isinstance(download_config, dict):
                download_config['continue_dl'] = True

        self.log_success("Enabled download resume, retrying")
        return HealerResult.fixed(
            "Enabled download resume for incomplete file",
            action=HealerAction.RETRY,
            resume_enabled=True
        )

    def _handle_generic_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle generic download errors with backoff."""
        self.log_attempt("Download error, retrying with backoff...")

        time.sleep(self.INITIAL_BACKOFF)
        self.retry_count += 1

        if self.retry_count > 5:
            return HealerResult.failed(f"Download failed after {self.retry_count} retries")

        return HealerResult.fixed(
            f"Download error, retry attempt {self.retry_count}",
            action=HealerAction.RETRY,
            retry_count=self.retry_count
        )

    def _extract_video_id(self, error_str: str) -> str | None:
        """Extract video ID from error message."""
        import re

        # YouTube video ID pattern (11 characters)
        patterns = [
            r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:\?|&|$|/)',
            r'Video ID: ([a-zA-Z0-9_-]{11})',
            r"'([a-zA-Z0-9_-]{11})'",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str)
            if match:
                return match.group(1)

        return None

    def reset_backoff(self):
        """Reset backoff state."""
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0

    def get_skipped_videos(self) -> Set[str]:
        """Get set of skipped video IDs."""
        return self.skipped_videos.copy()
