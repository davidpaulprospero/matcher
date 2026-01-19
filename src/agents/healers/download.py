"""
Download Healer - Video/audio download error recovery.

Handles:
- YouTube rate limiting (429) with exponential backoff
- Access denied (403) with tier escalation
- Bot detection with tier escalation
- Video unavailable (skip and continue)
- Format extraction failures
- Network timeouts
"""

from __future__ import annotations

import logging
import re
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Set, Optional

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class ErrorCategory(IntEnum):
    """Categories of download errors for routing to correct handler."""
    RATE_LIMIT = 1       # 429, "too many requests"
    ACCESS_DENIED = 2    # 403, age-gate, geo-block
    BOT_CHECK = 3        # "Sign in to confirm", bot detection
    UNAVAILABLE = 4      # Private, removed, not found
    FORMAT = 5           # Format extraction failed
    NETWORK = 6          # Timeout, connection error
    INCOMPLETE = 7       # Partial download
    CONNECTION_HANG = 8  # Connection hang (no output, likely tier issue)
    UNKNOWN = 99


class DownloadHealer(Healer):
    """
    Heals download-related errors with tiered bypass strategies.

    Recovery strategies by error type:
    - Connection hang (no output): Escalate to next bypass tier
    - Rate limit (429): Exponential backoff, stay on current tier
    - Access denied (403): Escalate to next bypass tier
    - Bot check: Escalate to next bypass tier
    - Video unavailable: Skip and continue
    - Format error: Try simpler format
    - Network timeout: Increase timeout, retry
    - Partial download: Resume
    """

    name = "download-healer"
    description = "Fix video download errors with bypass tier escalation"

    error_patterns = [
        "yt-dlp", "youtube", "download",
        "429", "rate limit", "too many",
        "403", "forbidden", "age", "geo",
        "sign in", "confirm", "bot",
        "unavailable", "private", "removed",
        "format", "extract",
        "timeout", "connection", "network", "hang",
        "incomplete", "partial",
    ]

    # Backoff configuration
    INITIAL_BACKOFF = 10.0
    MAX_BACKOFF = 300.0  # 5 minutes max
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, config: 'Config', project_dir):
        super().__init__(config, project_dir)
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0
        self.skipped_videos: Set[str] = set()

        # Tier escalation tracking
        self.tier_attempts = {1: 0, 2: 0, 3: 0}
        self.tier_successes = {1: 0, 2: 0, 3: 0}
        self.escalation_history: list = []

    def _get_bypass_config(self):
        """Get bypass config, handling both dataclass and dict."""
        download_config = getattr(self.config, 'download', None)
        if not download_config:
            return None
        return getattr(download_config, 'rate_limit_bypass', None)

    def _get_current_tier(self) -> int:
        """Get current bypass tier."""
        bypass = self._get_bypass_config()
        if not bypass:
            return 1
        if isinstance(bypass, dict):
            return bypass.get('_current_tier', 1)
        return bypass._current_tier

    def _set_current_tier(self, tier: int):
        """Set current bypass tier."""
        bypass = self._get_bypass_config()
        if not bypass:
            return
        if isinstance(bypass, dict):
            bypass['_current_tier'] = tier
        else:
            bypass._current_tier = tier

    def _categorize_error(self, error_str: str) -> ErrorCategory:
        """Categorize error for routing to correct handler."""
        error_lower = error_str.lower()

        # Order matters - check most specific first
        # Connection hang is most specific - check before general timeout/network
        if "connection hang" in error_lower or ("hang" in error_lower and "no output" in error_lower):
            return ErrorCategory.CONNECTION_HANG
        if any(p in error_lower for p in ["sign in", "confirm your", "bot"]):
            return ErrorCategory.BOT_CHECK
        if "429" in error_lower or "rate limit" in error_lower or "too many" in error_lower:
            return ErrorCategory.RATE_LIMIT
        if "403" in error_lower or "forbidden" in error_lower:
            return ErrorCategory.ACCESS_DENIED
        if any(p in error_lower for p in ["unavailable", "private", "removed", "not found"]):
            return ErrorCategory.UNAVAILABLE
        if any(p in error_lower for p in ["format", "extract", "no video"]):
            return ErrorCategory.FORMAT
        if any(p in error_lower for p in ["timeout", "connection", "network", "timed out"]):
            return ErrorCategory.NETWORK
        if any(p in error_lower for p in ["incomplete", "partial", "corrupt"]):
            return ErrorCategory.INCOMPLETE

        return ErrorCategory.UNKNOWN

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix download-related errors."""
        error_str = str(error)
        category = self._categorize_error(error_str)
        current_tier = self._get_current_tier()

        logger.debug(f"[HEALER] Error category: {category.name}, current tier: {current_tier}")
        self.tier_attempts[current_tier] = self.tier_attempts.get(current_tier, 0) + 1

        # Route to appropriate handler
        if category == ErrorCategory.CONNECTION_HANG:
            return self._handle_connection_hang(error, state)
        elif category == ErrorCategory.RATE_LIMIT:
            return self._handle_rate_limit(error, state)
        elif category == ErrorCategory.ACCESS_DENIED:
            return self._handle_access_denied(error, state)
        elif category == ErrorCategory.BOT_CHECK:
            return self._handle_bot_check(error, state)
        elif category == ErrorCategory.UNAVAILABLE:
            return self._handle_unavailable(error, state)
        elif category == ErrorCategory.FORMAT:
            return self._handle_format_error(error, state)
        elif category == ErrorCategory.NETWORK:
            return self._handle_network_error(error, state)
        elif category == ErrorCategory.INCOMPLETE:
            return self._handle_incomplete(error, state)
        else:
            return self._handle_generic_error(error, state)

    def _handle_rate_limit(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle 429 rate limiting with exponential backoff (stay on current tier)."""
        current_tier = self._get_current_tier()

        self.log_attempt(f"[Tier {current_tier}] Rate limited (429), backing off {self.backoff_time:.0f}s")
        logger.warning(f"[HEALER] YouTube 429 rate limit, waiting {self.backoff_time:.0f}s before retry")

        time.sleep(self.backoff_time)

        old_backoff = self.backoff_time
        self.backoff_time = min(self.backoff_time * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
        self.retry_count += 1

        self.log_success(f"[Tier {current_tier}] Waited {old_backoff:.0f}s, retrying (attempt {self.retry_count})")

        return HealerResult.fixed(
            f"Rate limit: waited {old_backoff:.0f}s (tier {current_tier})",
            action=HealerAction.RETRY,
            backoff_seconds=old_backoff,
            retry_count=self.retry_count,
            tier=current_tier
        )

    def _handle_access_denied(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle 403 Forbidden by escalating tier (age-gate, geo-block)."""
        bypass = self._get_bypass_config()
        current_tier = self._get_current_tier()

        # Check if escalation is enabled for 403
        escalate = True
        if bypass:
            if isinstance(bypass, dict):
                escalate = bypass.get('escalate_on_403', True)
            else:
                escalate = getattr(bypass, 'escalate_on_403', True)

        if not escalate:
            logger.info("[HEALER] 403 escalation disabled, skipping video")
            return self._skip_current_video(error, "403 Forbidden (escalation disabled)")

        return self._escalate_tier(
            reason="403 Forbidden",
            detail="Age-gated or geo-blocked content",
            error=error
        )

    def _handle_bot_check(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle bot detection by escalating tier."""
        bypass = self._get_bypass_config()

        # Check if escalation is enabled for bot check
        escalate = True
        if bypass:
            if isinstance(bypass, dict):
                escalate = bypass.get('escalate_on_bot_check', True)
            else:
                escalate = getattr(bypass, 'escalate_on_bot_check', True)

        if not escalate:
            logger.info("[HEALER] Bot check escalation disabled, skipping video")
            return self._skip_current_video(error, "Bot detection (escalation disabled)")

        return self._escalate_tier(
            reason="Bot detection",
            detail="YouTube requires sign-in confirmation",
            error=error
        )

    def _handle_connection_hang(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle connection hang by escalating tier.

        Connection hangs occur when yt-dlp produces no output before timeout.
        This typically indicates:
        - Tier 1 (--impersonate): curl_cffi TLS handshake failure with YouTube
        - Tier 2 (browser auth): Browser cookies expired or invalid
        - Network issue preventing connection establishment

        Strategy: Escalate to next tier immediately (no backoff needed).
        """
        current_tier = self._get_current_tier()

        logger.warning(
            f"[HEALER] Connection hang detected on Tier {current_tier}. "
            f"yt-dlp produced no output before timeout - likely tier compatibility issue."
        )

        return self._escalate_tier(
            reason="Connection hang",
            detail=f"No output from yt-dlp (Tier {current_tier} connection failed)",
            error=error
        )

    def _escalate_tier(self, reason: str, detail: str, error: Exception) -> HealerResult:
        """Escalate to next bypass tier."""
        current_tier = self._get_current_tier()
        bypass = self._get_bypass_config()

        # Check max tier
        max_tier = 3
        if bypass:
            if isinstance(bypass, dict):
                max_tier = bypass.get('max_tier', 3)
            else:
                max_tier = getattr(bypass, 'max_tier', 3)

        if current_tier >= max_tier:
            logger.error(f"[HEALER] All bypass tiers exhausted (max={max_tier})")
            return HealerResult.failed(
                f"All bypass tiers exhausted: {reason}. "
                f"Tried tiers 1-{max_tier}. Try again later or check video availability."
            )

        # Validate next tier is usable
        next_tier = current_tier + 1
        if next_tier == 2:
            # Validate browser auth is possible
            validation_error = self._validate_browser_auth()
            if validation_error:
                logger.warning(f"[HEALER] Cannot use Tier 2: {validation_error}")
                # Skip to tier 3
                next_tier = 3

        # Escalate
        old_tier = current_tier
        self._set_current_tier(next_tier)

        # Record escalation
        self.escalation_history.append({
            'from_tier': old_tier,
            'to_tier': next_tier,
            'reason': reason,
            'detail': detail
        })

        tier_names = {1: "Impersonate", 2: "Browser Auth", 3: "Standard"}
        logger.warning(
            f"[HEALER] Tier escalation: {old_tier} ({tier_names[old_tier]}) → "
            f"{next_tier} ({tier_names[next_tier]})"
        )
        logger.info(f"[HEALER] Reason: {reason} - {detail}")

        self.log_success(f"Escalated to Tier {next_tier} ({tier_names[next_tier]})")

        # Small backoff before retry with new tier
        time.sleep(2.0)

        return HealerResult.config_changed(
            f"Tier escalation: {reason} → Tier {next_tier} ({tier_names[next_tier]})",
            old_tier=old_tier,
            new_tier=next_tier,
            reason=reason
        )

    def _validate_browser_auth(self) -> Optional[str]:
        """Validate that browser auth can work. Returns error message or None."""
        import shutil

        bypass = self._get_bypass_config()
        if not bypass:
            return "No bypass config"

        if isinstance(bypass, dict):
            browser = bypass.get('tier2_browser', 'firefox')
        else:
            browser = getattr(bypass, 'tier2_browser', 'firefox')

        # Check if browser executable exists (basic check)
        browser_executables = {
            'firefox': ['firefox', 'firefox.exe'],
            'chrome': ['chrome', 'google-chrome', 'chrome.exe'],
            'edge': ['msedge', 'microsoft-edge', 'msedge.exe'],
            'brave': ['brave', 'brave-browser', 'brave.exe'],
        }

        executables = browser_executables.get(browser, [browser])
        found = any(shutil.which(exe) for exe in executables)

        if not found:
            return f"Browser '{browser}' not found in PATH"

        return None

    def _skip_current_video(self, error: Exception, reason: str) -> HealerResult:
        """Skip the current video and continue."""
        video_id = self._extract_video_id(str(error))

        if video_id:
            self.skipped_videos.add(video_id)
            logger.info(f"[HEALER] Skipping video {video_id}: {reason}")
            return HealerResult.fixed(
                f"Skipped video {video_id}: {reason}",
                action=HealerAction.SKIP,
                skipped_video=video_id
            )

        return HealerResult.fixed(
            f"Skipping current video: {reason}",
            action=HealerAction.RETRY,
            skip_current=True
        )

    def _handle_unavailable(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle unavailable video by skipping."""
        self.log_attempt("Video unavailable, marking to skip")
        return self._skip_current_video(error, "Video unavailable")

    def _handle_format_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle format extraction errors by trying alternate formats."""
        self.log_attempt("Format extraction failed, trying simpler format")

        download_config = getattr(self.config, 'download', None)
        if not download_config:
            return HealerResult.failed("No download config available")

        format_fallbacks = [
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
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

        logger.info(f"[HEALER] Format fallback: {current_format} → {new_format}")
        self.log_success(f"Switched format to {new_format}")

        return HealerResult.config_changed(
            f"Format fallback: {new_format}",
            old_format=current_format,
            new_format=new_format
        )

    def _handle_network_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle network/timeout errors."""
        self.log_attempt("Network error, waiting before retry")

        time.sleep(self.INITIAL_BACKOFF)

        download_config = getattr(self.config, 'download', None)
        if download_config:
            current_timeout = getattr(download_config, 'socket_timeout', 30)
            new_timeout = min(current_timeout * 2, 120)

            if hasattr(download_config, 'socket_timeout'):
                download_config.socket_timeout = new_timeout
            elif isinstance(download_config, dict):
                download_config['socket_timeout'] = new_timeout

            logger.info(f"[HEALER] Socket timeout: {current_timeout}s → {new_timeout}s")
            self.log_success(f"Increased timeout to {new_timeout}s")

            return HealerResult.config_changed(
                f"Network error, timeout increased to {new_timeout}s",
                old_timeout=current_timeout,
                new_timeout=new_timeout
            )

        return HealerResult.fixed("Network error, retrying", action=HealerAction.RETRY)

    def _handle_incomplete(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle incomplete/partial downloads."""
        self.log_attempt("Incomplete download, enabling resume")

        download_config = getattr(self.config, 'download', None)
        if download_config:
            if hasattr(download_config, 'continue_dl'):
                download_config.continue_dl = True
            elif isinstance(download_config, dict):
                download_config['continue_dl'] = True

        self.log_success("Enabled download resume")
        return HealerResult.fixed(
            "Enabled download resume",
            action=HealerAction.RETRY,
            resume_enabled=True
        )

    def _handle_generic_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle generic download errors with backoff."""
        self.log_attempt("Download error, retrying with backoff")

        time.sleep(self.INITIAL_BACKOFF)
        self.retry_count += 1

        if self.retry_count > 5:
            return HealerResult.failed(f"Download failed after {self.retry_count} retries")

        return HealerResult.fixed(
            f"Download error, retry {self.retry_count}",
            action=HealerAction.RETRY,
            retry_count=self.retry_count
        )

    def _extract_video_id(self, error_str: str) -> Optional[str]:
        """Extract video ID from error message."""
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

    def reset(self):
        """Reset all state for new download batch."""
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0
        self._set_current_tier(1)
        logger.debug("[HEALER] Reset: backoff, retry count, and tier")

    def mark_success(self):
        """Mark current tier as successful (call after successful download)."""
        current_tier = self._get_current_tier()
        self.tier_successes[current_tier] = self.tier_successes.get(current_tier, 0) + 1

        # Reset backoff on success
        self.backoff_time = self.INITIAL_BACKOFF
        logger.debug(f"[HEALER] Tier {current_tier} success recorded, backoff reset")

    def get_statistics(self) -> dict:
        """Get healer statistics for reporting."""
        return {
            'current_tier': self._get_current_tier(),
            'retry_count': self.retry_count,
            'skipped_videos': len(self.skipped_videos),
            'tier_attempts': dict(self.tier_attempts),
            'tier_successes': dict(self.tier_successes),
            'escalation_count': len(self.escalation_history),
            'escalations': self.escalation_history[-5:],  # Last 5
        }

    def get_skipped_videos(self) -> Set[str]:
        """Get set of skipped video IDs."""
        return self.skipped_videos.copy()
