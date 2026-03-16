"""
Download Healer - Video/audio download error recovery.

Handles:
- YouTube rate limiting (429) with cookie rotation and VPN switching
- Video unavailable
- Format extraction failures
- Network timeouts
- Partial downloads

Integration with core retry mechanism:
- VideoDownloader._run_download_cmd() already implements exponential backoff retry
- DownloadHealer checks if retries are exhausted before applying additional backoff
- When retries exhausted, healer proceeds directly to cookie rotation/VPN switch
- Clear logging shows handoff between core retry and healer escalation
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Set, Union

from ..base import Healer, HealerResult, HealerAction, HealerEvent, HealerEventData, get_config_value, set_config_value
from ...downloader.cookie_rotator import CookieRotator
from ...downloader.vpn_manager import VPNManager
from ...downloader.types import DownloadError

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState
    from ...downloader.escalation_manager import EscalationManager

logger = logging.getLogger(__name__)


class DownloadHealer(Healer):
    """
    Heals download-related errors.

    Recovery strategies (in order for rate limits):
    1. Cookie rotation: Switch to a different cookie file
    2. VPN switch: Change IP address via VPN
    3. Exponential backoff: Wait and retry

    Other strategies:
    - Video unavailable: Skip and continue
    - Format error: Try alternate format
    - Network timeout: Retry with longer timeout
    - Partial download: Resume or restart
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
        "sign in",
        "login required",
        "bot detection",
    ]

    # Backoff configuration
    INITIAL_BACKOFF = 10.0
    MAX_BACKOFF = 600.0  # 10 minutes
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, config: 'Config', project_dir: Union[str, Any], escalation_manager: Optional['EscalationManager'] = None) -> None:
        super().__init__(config, project_dir)
        self.backoff_time: float = self.INITIAL_BACKOFF
        self.retry_count: int = 0
        self.skipped_videos: Set[str] = set()

        # Store shared escalation manager from pipeline's VideoDownloader
        self.escalation_manager: Optional['EscalationManager'] = escalation_manager
        if self.escalation_manager:
            logger.info("DownloadHealer: Using shared EscalationManager from pipeline")

        # Initialize cookie rotator only if no escalation manager is provided.
        # When escalation_manager is available, its Tier 3 handles cookie rotation,
        # so we skip creating a duplicate CookieRotator here.
        self.cookie_rotator: Optional[CookieRotator] = None
        download_config = getattr(config, 'download', None)
        if not self.escalation_manager:
            if download_config:
                cookie_rotation_config = getattr(download_config, 'cookie_rotation', None)
                if cookie_rotation_config and getattr(cookie_rotation_config, 'enabled', False):
                    self.cookie_rotator = CookieRotator(cookie_rotation_config)
                    if self.cookie_rotator.is_enabled:
                        logger.info(f"DownloadHealer: Cookie rotation enabled with {self.cookie_rotator.available_cookies} cookies")

        # Initialize VPN manager if configured
        self.vpn_manager: Optional[VPNManager] = None
        if download_config:
            vpn_config = getattr(download_config, 'vpn', None)
            if vpn_config and getattr(vpn_config, 'enabled', False):
                self.vpn_manager = VPNManager(vpn_config)
                if self.vpn_manager.is_enabled:
                    logger.info("DownloadHealer: VPN manager enabled")

    def can_handle(self, error: Exception, stage_name: str) -> bool:
        """Check if this healer can handle the given error."""
        return super().can_handle(error, stage_name)

    def _get_retry_context(self, error: Exception) -> tuple[int, int, bool]:
        """
        Extract retry context from error if available.

        Args:
            error: The exception that occurred

        Returns:
            Tuple of (retry_count, max_retries, retries_exhausted).
            Returns (0, 3, False) if error is not a DownloadError.
        """
        if isinstance(error, DownloadError):
            return (error.retry_count, error.max_retries, error.retries_exhausted)
        return (0, 3, False)

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """
        Attempt to fix download-related errors.

        Checks for retry context from DownloadError to avoid redundant backoff.
        If retries are already exhausted by the core retry mechanism, proceeds
        directly to escalation (cookie rotation, VPN switch) without additional backoff.

        When an escalation_manager is available, records success/failure to
        update the shared escalation tier state.
        """
        error_str = str(error).lower()

        # Get retry context to avoid redundant backoff
        retry_count, max_retries, retries_exhausted = self._get_retry_context(error)
        if retries_exhausted:
            logger.info(f"[{self.name}] Core retry exhausted ({retry_count}/{max_retries}), proceeding to escalation")

        # Rate limiting (includes 403 Forbidden - cookie/auth issue)
        if any(p in error_str for p in ["429", "rate limit", "too many", "403", "forbidden"]):
            return self._handle_rate_limit(error, state, retries_exhausted)

        # Video unavailable
        if any(p in error_str for p in ["unavailable", "private", "removed", "not found"]):
            return self._handle_unavailable(error, state)

        # Format extraction error
        if any(p in error_str for p in ["format", "extract", "no video"]):
            return self._handle_format_error(error, state)

        # Network/timeout errors
        if any(p in error_str for p in ["timeout", "connection", "network"]):
            result = self._handle_network_error(error, state, retries_exhausted)
            # Record failure with escalation manager for network errors
            # (these may indicate rate limiting or IP blocking)
            if self.escalation_manager and not result.success:
                keyword = self._extract_keyword_from_error(error)
                self.escalation_manager.record_failure(keyword, str(error))
            return result

        # Partial/incomplete download
        if any(p in error_str for p in ["incomplete", "partial", "corrupt"]):
            return self._handle_incomplete(error, state)

        # Generic download error - try with backoff
        return self._handle_generic_error(error, state, retries_exhausted)

    def _extract_keyword_from_error(self, error: Exception) -> Optional[str]:
        """
        Extract keyword or video ID from error for escalation manager tracking.

        Falls back to 'unknown' if no keyword can be extracted.
        """
        # Try to extract from DownloadError attributes
        if isinstance(error, DownloadError):
            keyword = getattr(error, 'keyword', None)
            if keyword:
                return keyword

        # Try to extract video ID from error message
        video_id = self._extract_video_id(str(error))
        return video_id or 'healer_unknown'

    def _handle_rate_limit(
        self,
        error: Exception,
        state: 'PipelineState',
        retries_exhausted: bool = False
    ) -> HealerResult:
        """
        Handle YouTube rate limiting with escalation manager, cookie rotation, VPN, then backoff.

        Recovery order:
        1. Consult escalation manager for current tier args (if available)
        2. Try cookie rotation (if enabled and available, skipped when escalation manager handles it)
        3. Try VPN switch (if enabled and cookies exhausted)
        4. Fall back to exponential backoff (SKIPPED if retries already exhausted)

        Args:
            error: The exception that occurred
            state: Pipeline state
            retries_exhausted: If True, core retry already exhausted - skip additional backoff
        """
        error_str = str(error)
        keyword = self._extract_keyword_from_error(error)
        self._last_keyword = keyword  # Track for success recording on reset_backoff

        # 0. Record failure with escalation manager (advances tier state)
        if self.escalation_manager:
            self.escalation_manager.record_failure(keyword, error_str)
            self.log_attempt(f"Recorded failure with EscalationManager for '{keyword}'")

        # 1. Consult escalation manager for current tier args
        if self.escalation_manager:
            escalation_result = self.escalation_manager.get_escalation_args(keyword)
            tier_name = escalation_result.tier.name if hasattr(escalation_result.tier, 'name') else str(escalation_result.tier)
            self.log_attempt(f"EscalationManager recommends tier {tier_name} for '{keyword}'")

            # Cookie rotation is handled by escalation manager at Tier 3
            if escalation_result.rotate_cookies:
                self.log_attempt("Escalation tier 3: cookie rotation delegated to EscalationManager")

            return HealerResult.fixed(
                f"Rate limit: escalation tier {tier_name} applied for retry",
                action=HealerAction.RETRY,
                escalation_tier=tier_name,
                escalation_args=escalation_result.args,
                rotate_cookies=escalation_result.rotate_cookies,
                retry_count=self.retry_count,
                core_retries_exhausted=retries_exhausted
            )

        # 2. Try cookie rotation first (fallback when no escalation manager)
        if self._try_cookie_rotation(error_str):
            return HealerResult.fixed(
                "Rate limit: rotated to new cookie",
                action=HealerAction.RETRY,
                cookie_rotated=True,
                retry_count=self.retry_count,
                core_retries_exhausted=retries_exhausted
            )

        # 3. Try VPN switch if cookies exhausted
        if self._try_vpn_switch():
            return HealerResult.fixed(
                "Rate limit: switched VPN server",
                action=HealerAction.RETRY,
                vpn_switched=True,
                retry_count=self.retry_count,
                core_retries_exhausted=retries_exhausted
            )

        # 4. Fall back to exponential backoff
        # SKIP if core retry already exhausted - avoid redundant waiting
        if retries_exhausted:
            self.log_attempt("Skipping healer backoff (core retry already exhausted)")
            return HealerResult.failed(
                "Rate limit: all escalation options exhausted (cookie rotation + VPN + core retries)",
                core_retries_exhausted=True,
                healer_retry_count=self.retry_count
            )

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

    def _try_cookie_rotation(self, error_str: str) -> bool:
        """
        Attempt to rotate cookie based on error.

        Args:
            error_str: Error string from yt-dlp

        Returns:
            True if cookie was rotated successfully
        """
        if not self.cookie_rotator or not self.cookie_rotator.is_enabled:
            return False

        if not self.cookie_rotator.should_rotate(error_str):
            return False

        if not self.cookie_rotator.can_rotate():
            self.log_attempt("Cookie rotation exhausted, trying other methods...")
            return False

        new_cookie = self.cookie_rotator.rotate()
        if new_cookie:
            self.log_success(f"Rotated to cookie: {new_cookie}")
            # Reset backoff after successful rotation
            self.backoff_time = self.INITIAL_BACKOFF
            return True

        return False

    def _try_vpn_switch(self) -> bool:
        """
        Attempt to switch VPN server.

        Returns:
            True if VPN was switched successfully
        """
        if not self.vpn_manager or not self.vpn_manager.is_enabled:
            return False

        if not self.vpn_manager.can_switch():
            self.log_attempt("VPN switch limit reached, falling back to backoff...")
            return False

        self.log_attempt("Switching VPN server...")
        success = self.vpn_manager.switch()

        if success:
            self.log_success(f"VPN switched (total: {self.vpn_manager.switch_count})")
            # Reset cookie rotator after VPN switch (new IP = fresh start)
            if self.cookie_rotator:
                self.cookie_rotator.reset()
            # Reset backoff after successful VPN switch
            self.backoff_time = self.INITIAL_BACKOFF
            return True

        self.log_attempt("VPN switch failed, falling back to backoff...")
        return False

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

        current_format = get_config_value(download_config, 'format', None)

        try:
            current_idx = format_fallbacks.index(current_format) if current_format else -1
        except ValueError:
            current_idx = -1

        next_idx = current_idx + 1
        if next_idx >= len(format_fallbacks):
            return HealerResult.failed("All format options exhausted")

        new_format = format_fallbacks[next_idx]
        set_config_value(download_config, 'format', new_format)

        self.log_success(f"Switched format: {current_format} -> {new_format}")
        return HealerResult.config_changed(
            f"Switched to simpler format: {new_format}",
            old_format=current_format,
            new_format=new_format
        )

    def _handle_network_error(
        self,
        error: Exception,
        state: 'PipelineState',
        retries_exhausted: bool = False
    ) -> HealerResult:
        """
        Handle network/timeout errors.

        Args:
            error: The exception that occurred
            state: Pipeline state
            retries_exhausted: If True, core retry already exhausted - skip additional backoff
        """
        # Skip wait if retries already exhausted
        if not retries_exhausted:
            self.log_attempt("Network error, waiting before retry...")
            time.sleep(self.INITIAL_BACKOFF)
        else:
            self.log_attempt("Network error, skipping backoff (core retry already exhausted)")

        # Try increasing socket timeout
        download_config = getattr(self.config, 'download', None)
        if download_config:
            current_timeout = get_config_value(download_config, 'socket_timeout', 30)
            new_timeout = min(current_timeout * 2, 120)

            set_config_value(download_config, 'socket_timeout', new_timeout)

            self.log_success(f"Increased socket timeout: {current_timeout}s -> {new_timeout}s")
            return HealerResult.config_changed(
                f"Network error, increased timeout to {new_timeout}s",
                old_timeout=current_timeout,
                new_timeout=new_timeout,
                core_retries_exhausted=retries_exhausted
            )

        return HealerResult.fixed(
            "Network error, retrying",
            action=HealerAction.RETRY,
            core_retries_exhausted=retries_exhausted
        )

    def _handle_incomplete(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle incomplete/partial downloads."""
        self.log_attempt("Incomplete download detected...")

        # Enable resume in yt-dlp config
        download_config = getattr(self.config, 'download', None)
        if download_config:
            set_config_value(download_config, 'continue_dl', True)

        self.log_success("Enabled download resume, retrying")
        return HealerResult.fixed(
            "Enabled download resume for incomplete file",
            action=HealerAction.RETRY,
            resume_enabled=True
        )

    def _handle_generic_error(
        self,
        error: Exception,
        state: 'PipelineState',
        retries_exhausted: bool = False
    ) -> HealerResult:
        """
        Handle generic download errors with backoff.

        Args:
            error: The exception that occurred
            state: Pipeline state
            retries_exhausted: If True, core retry already exhausted - skip additional backoff
        """
        # If core retry already exhausted, don't add redundant backoff - fail fast
        if retries_exhausted:
            self.log_attempt("Download error, skipping healer backoff (core retry already exhausted)")
            return HealerResult.failed(
                f"Download failed after core retries exhausted",
                core_retries_exhausted=True,
                healer_retry_count=self.retry_count
            )

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

    def handle_event(self, event_data: HealerEventData) -> None:
        """Handle cross-healer coordination events.

        Reacts to:
        - CONFIG_CHANGED: Reset backoff since config may have fixed the issue
        - RATE_LIMITED: Increase backoff preemptively
        - PROVIDER_SWITCHED: Reset backoff for fresh start with new provider
        """
        if event_data.event == HealerEvent.CONFIG_CHANGED:
            self.reset_backoff()
            logger.debug(f"[{self.name}] Reset backoff due to config change from {event_data.source_healer}")
        elif event_data.event == HealerEvent.RATE_LIMITED:
            self.backoff_time = min(self.backoff_time * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
            logger.debug(f"[{self.name}] Increased backoff to {self.backoff_time:.1f}s due to rate limit from {event_data.source_healer}")
        elif event_data.event == HealerEvent.PROVIDER_SWITCHED:
            self.reset_backoff()
            logger.debug(f"[{self.name}] Reset backoff due to provider switch from {event_data.source_healer}")

    def reset_backoff(self):
        """Reset backoff state.

        Called by ResilientRunner after a successful stage run.
        If an escalation_manager is available, records success for
        any keywords the healer was tracking.
        """
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0

        # Record success with escalation manager to update tier state
        if self.escalation_manager and self._last_keyword:
            self.escalation_manager.record_success(self._last_keyword)
            logger.debug(f"DownloadHealer: Recorded success for '{self._last_keyword}'")
            self._last_keyword = None

    @property
    def _last_keyword(self) -> Optional[str]:
        """Last keyword the healer operated on (for success tracking)."""
        return getattr(self, '_tracked_keyword', None)

    @_last_keyword.setter
    def _last_keyword(self, value: Optional[str]):
        self._tracked_keyword = value

    def get_skipped_videos(self) -> Set[str]:
        """Get set of skipped video IDs."""
        return self.skipped_videos.copy()
