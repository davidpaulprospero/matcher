"""
Streamlined VideoDownloader core orchestrator.

Refactored from monolithic downloader.py (2,906 lines) into modular architecture.
This core orchestrator (~650 lines) delegates to specialized modules:
- CheckpointManager: Checkpoint/resume and duration tier management
- TranscodingManager: FFmpeg transcoding and codec detection
- TitleFilter: LLM-based title filtering
- SpeechScreener: Whisper VAD speech detection
- SearchOptimizer: Download search optimization with adaptive pools and keyword alternatives
- AudioFirstPipeline: Audio-first download workflow
- segment_utils, utils: Helper functions

Original methods extracted: ~2,050 lines across 8 modules (71% reduction).
Core orchestration logic retained: download_all, download_for_keyword, _download_single,
_download_by_ids, _run_download_cmd (subprocess management, transcoding workflow).
"""

from __future__ import annotations

import os
import json
import subprocess
import time
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from ..config import Config, get_config
from ..state import DownloadedVideo
from ..global_cache import GlobalCacheManager
from ..config.sections.test_mode import is_mock_rate_limits_enabled, get_mock_delay_seconds

from .checkpoint import CheckpointManager, DownloadCheckpoint
from .transcoding import TranscodingManager
from .title_filter import TitleFilter
from .speech_screening import SpeechScreener
from .keyword_remix import SearchOptimizer
from .audio_first import AudioFirstPipeline
from .cookie_rotator import CookieRotator
from .cookie_method_fallback import CookieMethodFallback
from .impersonation import ImpersonationManager
from .escalation_manager import EscalationManager, EscalationResult, is_escalation_trigger
from .vpn_manager import VPNManager
from .mullvad_vpn import MullvadVPN
from .speed_tracker import DownloadSpeedTracker, DownloadSpeedConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .pause_calculator import RegionSuccessTracker
from .retry_queue import RetryQueue, BatchRetryConfig
from .rate_limit_metrics import RateLimitMetrics
from .rate_limit_budget import RateLimitBudget
from .metrics_exporter import DownloadMetricsExporter, DownloadMetricsConfig, create_metrics_exporter
from . import utils
from .utils import SUBPROCESS_FLAGS
from .errors import log_error, log_download_error, get_error_code
from src.logging_templates import log_error_with_context, log_rate_limit
from ..rate_limit.coordinator import (
    GlobalRateLimitCoordinator,
    build_coordinator_rate_limit_config,
)

# Lazy import to avoid circular dependency
# DownloadOrchestrator is imported at runtime in download_all()

logger = logging.getLogger(__name__)


def _log_download_failure(
    video_id: str,
    url: str,
    tier: str,
    error_type: str,
    error_message: str,
    keyword: str = None
) -> None:
    """Log download failure with standardized error codes for observability.

    Args:
        video_id: YouTube video ID or keyword|tier fallback
        url: YouTube URL if available
        tier: Duration tier (short/medium/long)
        error_type: Classified error type (transient, permanent, auth, etc.)
        error_message: Error message or stderr excerpt
        keyword: Search keyword (optional, for context)
    """
    # Map error_type to DL-xxx error codes
    error_code_map = {
        "timeout": "DL-002",
        "transient": "DL-002",
        "format": "DL-003",
        "unavailable": "DL-003",
        "geo_blocked": "DL-003",
        "auth": "DL-004",
        "cookies": "DL-004",
        "cookie_exhausted": "DL-004",
        "login_required": "DL-004",
        "vpn": "DL-005",
        "rate_limit": "DL-006",
        "quota": "DL-006",
    }
    error_code = error_code_map.get(error_type.lower() if error_type else "", "DL-001")

    # Truncate error message for logging
    msg = f"Download failed: {error_message[:200]}" if error_message else "Download failed"

    log_error_with_context(
        logger,
        error_code,
        msg,
        video_id=video_id,
        url=url,
        tier=tier,
        error_type=error_type,
        keyword=keyword
    )


def _log_retry_attempt(
    keyword: str,
    tier: str,
    attempt: int,
    max_retries: int,
    delay: float,
    reason: str,
    video_id: str = None
) -> None:
    """Log retry attempt with structured context.

    Args:
        keyword: Search keyword
        tier: Duration tier
        attempt: Current attempt number (0-indexed)
        max_retries: Maximum retries allowed
        delay: Delay before retry in seconds
        reason: Reason for retry (timeout, transient, etc.)
        video_id: YouTube video ID if known
    """
    log_data = {
        "event_type": "retry_attempt",
        "keyword": keyword,
        "tier": tier,
        "attempt": attempt + 1,  # Convert to 1-indexed for display
        "max_retries": max_retries,
        "countdown_seconds": delay,
        "reason": reason,
    }
    if video_id:
        log_data["video_id"] = video_id

    logger.info("retry_attempt", extra=log_data)


def _log_escalation_event(
    keyword: str,
    tier: str,
    event: str,
    before_state: str,
    after_state: str,
    details: dict = None
) -> None:
    """Log escalation event with before/after state for observability.

    Args:
        keyword: Search keyword
        tier: Current duration tier
        event: Event type (tier_change, vpn_activation, cookie_rotation)
        before_state: State before the event
        after_state: State after the event
        details: Additional context dict
    """
    log_data = {
        "event_type": "escalation_event",
        "keyword": keyword,
        "tier": tier,
        "escalation_event": event,
        "before_state": before_state,
        "after_state": after_state,
    }
    if details:
        log_data.update(details)

    logger.info("escalation_event", extra=log_data)


# US-52-006: Error classification delegated to shared module
# US-89-012: Adaptive severity tracking
# US-93-006: Structured error responses with error codes and suggestions
from .error_classification import (
    ERROR_SEVERITY_PATTERNS,
    SEVERITY_MULTIPLIERS,
    classify_error_severity,
    classify_error_category,
    init_adaptive_tracker,
    parse_retry_after,
)
from .errors import (
    ClassifiedDownloadError,
    StructuredDownloadError,
)


@dataclass
class TierRateLimitState:
    """Per-tier rate limit state tracking.

    Maintains independent backoff state for each duration tier (short, medium, long, longer).
    When per_tier_isolation is enabled, rate limiting on one tier doesn't affect others.

    Attributes:
        backoff_count: Number of backoff attempts for this tier
        total_delay: Cumulative delay applied for this tier (seconds)
        in_recovery: Whether this tier is in cooldown recovery mode
        last_event_time: Timestamp of last rate limit event for this tier
    """
    backoff_count: int = 0
    total_delay: float = 0.0
    in_recovery: bool = False
    last_event_time: Optional[str] = None

    def reset(self) -> None:
        """Reset backoff state after successful download or cookie rotation."""
        self.backoff_count = 0
        self.total_delay = 0.0
        # Don't reset in_recovery - that's session-level from checkpoint

    def to_dict(self) -> dict:
        """Serialize state for checkpoint."""
        return {
            'backoff_count': self.backoff_count,
            'total_delay': self.total_delay,
            'in_recovery': self.in_recovery,
            'last_event_time': self.last_event_time
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TierRateLimitState':
        """Create from checkpoint data."""
        if not data:
            return cls()
        return cls(
            backoff_count=data.get('backoff_count', 0),
            total_delay=data.get('total_delay', 0.0),
            in_recovery=data.get('in_recovery', False),
            last_event_time=data.get('last_event_time')
        )


class VideoDownloader:
    """
    Advanced video downloader with modular architecture.

    Features:
    - Three duration tiers (short/medium/long/longer)
    - Parallel downloads with rate limiting
    - DaVinci Resolve transcoding (optional)
    - LLM title filtering (optional)
    - Speech screening for B-roll detection (optional)
    - Keyword remixing when searches fail
    - Audio-first pipeline for bandwidth savings
    - Checkpoint/resume support
    - Source attribution tracking
    """

    def __init__(self, config: Config = None):
        """
        Initialize VideoDownloader with all specialized managers.

        Args:
            config: Config object (defaults to get_config() if None)
        """
        self.config = config or get_config()
        self.download_config = self.config.download
        self.download_coordinator = None

        # Initialize checkpoint manager
        checkpoint_file = Path(self.config.cache_dir) / "download_checkpoint.json"
        sources_file = Path(self.config.downloaded_videos_dir) / "sources.json"

        self.checkpoint_mgr = CheckpointManager(
            config=self.config,
            checkpoint_file=checkpoint_file,
            sources_file=sources_file
        )

        # Initialize transcoding manager
        self.transcoding_mgr = TranscodingManager(self.config)

        # Initialize title filter (needs cookies and tier functions)
        cookies_args = utils.get_cookies_args(self.config)
        self.title_filter = TitleFilter(
            config=self.config,
            cookies_args=cookies_args,
            get_tier_value_func=self.checkpoint_mgr.get_tier_value
        )

        # Initialize speech screener
        self.speech_screener = SpeechScreener(
            config=self.config,
            cookies_args=cookies_args
        )

        # Initialize search optimizer (needs LLM functions from title_filter)
        self.search_optimizer = SearchOptimizer(
            config=self.config,
            llm_call_gemini_func=self.title_filter._call_gemini,
            llm_call_anthropic_func=self.title_filter._call_anthropic
        )

        # Initialize audio-first pipeline
        self._lock = threading.RLock()
        self.tier_download_counts: Dict[str, int] = {
            'short': 0, 'medium': 0, 'long': 0, 'longer': 0
        }

        self.audio_first = AudioFirstPipeline(
            config=self.config,
            get_tier_value_func=self.checkpoint_mgr.get_tier_value,
            search_metadata_func=self.title_filter.search_video_metadata,
            filter_titles_func=self.title_filter.filter_titles_with_llm,
            cleanup_partial_func=self._cleanup_partial_files,
            tier_download_counts=self.tier_download_counts,
            lock=self._lock
        )

        # US-89-012: Initialize adaptive severity tracker from config
        adaptive_config = getattr(self.download_config, 'adaptive_severity', None)
        if adaptive_config:
            init_adaptive_tracker(
                enabled=getattr(adaptive_config, 'enabled', True),
                escalation_threshold=getattr(adaptive_config, 'escalation_threshold', 3),
                max_severity=getattr(adaptive_config, 'max_severity', 'high'),
                window_size=getattr(adaptive_config, 'window_size', 10),
                reset_on_success=getattr(adaptive_config, 'reset_on_success', True),
            )

        # Core state
        self.sources: List[DownloadedVideo] = self.checkpoint_mgr.load_sources()
        self.checkpoint: Optional[DownloadCheckpoint] = None
        self.DURATION_TIERS = self.checkpoint_mgr.duration_tiers
        self._last_download_timed_out = False
        self._last_download_rate_limited = False  # Track rate limit failures for batch retry

        # Cookie authentication
        self._cookies_from_browser = getattr(self.download_config, 'cookies_from_browser', '')
        self._cookies_path = self._find_cookies_file() if not self._cookies_from_browser else None

        if self._cookies_from_browser:
            logger.info(f"Using cookies from browser: {self._cookies_from_browser}")
        elif self._cookies_path:
            logger.info(f"Found cookies file: {self._cookies_path}")
        else:
            logger.warning("No cookies configured - YouTube downloads may fail!")

        # Cookie rotation (for rate limit evasion)
        cookie_rotation_config = getattr(self.download_config, 'cookie_rotation', None)
        if cookie_rotation_config and getattr(cookie_rotation_config, 'enabled', False):
            self.cookie_rotator = CookieRotator(cookie_rotation_config)
            if self.cookie_rotator.is_enabled:
                logger.info(f"Cookie rotation enabled with {self.cookie_rotator.available_cookies} cookies")
                # Share cookie rotator with audio-first pipeline
                self.audio_first.cookie_rotator = self.cookie_rotator
        else:
            self.cookie_rotator = None

        # Cookie method fallback chain (browser → file1 → file2 → no-cookies)
        self.method_fallback = CookieMethodFallback(self.download_config)

        # Browser impersonation (TLS fingerprint bypass)
        impersonation_config = getattr(self.download_config, 'impersonation', None)
        impersonation_enabled = getattr(impersonation_config, 'enabled', False) if impersonation_config else False
        if impersonation_enabled:
            preferred = getattr(impersonation_config, 'preferred_targets', []) or []
            detect_startup = getattr(impersonation_config, 'detect_at_startup', True)
            timeout = getattr(impersonation_config, 'detection_timeout', 10)
            min_success_rate = getattr(impersonation_config, 'min_success_rate', 0.2)
            enable_success_filtering = getattr(impersonation_config, 'enable_success_filtering', True)
            fallback_order = getattr(impersonation_config, 'impersonation_fallback_order', None)
            adaptive = getattr(impersonation_config, 'adaptive_profile_selection', False)
            success_window = getattr(impersonation_config, 'profile_success_window', 20)
            self.impersonation_manager = ImpersonationManager(
                preferred_targets=preferred,
                detect_at_startup=detect_startup,
                detection_timeout=timeout,
                min_success_rate=min_success_rate,
                enable_success_filtering=enable_success_filtering,
                fallback_order=fallback_order,
                adaptive_profile_selection=adaptive,
                profile_success_window=success_window,
            )
            # Share impersonation manager with audio-first pipeline
            self.audio_first.impersonation_manager = self.impersonation_manager
            # Share with auxiliary modules (title filter, speech screener)
            self.title_filter.impersonation_manager = self.impersonation_manager
            self.speech_screener.impersonation_manager = self.impersonation_manager
        else:
            self.impersonation_manager = None

        # Escalation manager (3-tier bypass: impersonation → extractor-args → full bypass)
        extractor_args_config = getattr(self.download_config, 'extractor_args', None)
        if self.impersonation_manager:
            self.escalation_manager = EscalationManager(
                impersonation_manager=self.impersonation_manager,
                extractor_args_config=extractor_args_config,
            )
            # Share escalation manager with audio-first pipeline and auxiliary modules
            self.audio_first.escalation_manager = self.escalation_manager
            self.title_filter.escalation_manager = self.escalation_manager
            self.speech_screener.escalation_manager = self.escalation_manager
            logger.info("Escalation manager enabled (3-tier bypass)")
        else:
            self.escalation_manager = None

        # VPN manager (for IP rotation)
        vpn_config = getattr(self.download_config, 'vpn', None)
        if vpn_config and getattr(vpn_config, 'enabled', False):
            self.vpn_manager = VPNManager(vpn_config)
            if self.vpn_manager.is_enabled:
                logger.info("VPN manager enabled for IP rotation")
        else:
            self.vpn_manager = None

        # Mullvad VPN manager (Tier 4 bypass - IP rotation via Mullvad)
        mullvad_config = getattr(self.download_config, 'mullvad', None)
        if mullvad_config and getattr(mullvad_config, 'enabled', False):
            self.mullvad_vpn = MullvadVPN(mullvad_config)
            logger.info("Mullvad VPN enabled for Tier 4 IP rotation")
            # Connect Mullvad to escalation manager for Tier 4 escalation
            if self.escalation_manager:
                self.escalation_manager.set_mullvad_vpn(self.mullvad_vpn)
                logger.info("Escalation manager upgraded to 4-tier bypass (with Mullvad VPN)")
        else:
            self.mullvad_vpn = None

        # Rate limit backoff state (progressive delay before cookie rotation)
        self._rate_limit_backoff_count = 0  # Current backoff attempt count (global fallback)
        self._rate_limit_total_delay = 0.0  # Cumulative delay applied (global fallback)
        self._rate_limit_event_count = 0  # Count of rate limit events this session
        self._in_cooldown_recovery_mode = False  # True if resumed within cooldown period

        # Per-tier rate limit state (US-001: tier isolation)
        # Each tier maintains independent backoff state when per_tier_isolation is enabled
        self._tier_rate_limit_states: Dict[str, TierRateLimitState] = {
            'short': TierRateLimitState(),
            'medium': TierRateLimitState(),
            'long': TierRateLimitState(),
            'longer': TierRateLimitState(),
        }
        # Check if per-tier isolation is enabled in config
        rate_limit_config = getattr(self.download_config, 'rate_limit', None)
        self._per_tier_isolation = getattr(rate_limit_config, 'per_tier_isolation', True) if rate_limit_config else True

        # Validate retry configuration bounds (US-58-004)
        max_retries = getattr(self.download_config, 'max_retries', 3)
        retry_backoff = getattr(self.download_config, 'retry_backoff', 2.0)
        retry_delay = getattr(self.download_config, 'retry_delay', 2.0)

        if not isinstance(max_retries, int) or max_retries < 0 or max_retries > 20:
            raise ValueError(
                f"max_retries must be an integer in range [0, 20], got {max_retries}"
            )
        if not isinstance(retry_backoff, (int, float)) or retry_backoff < 1.0 or retry_backoff > 10.0:
            raise ValueError(
                f"retry_backoff must be a number in range [1.0, 10.0], got {retry_backoff}"
            )

        # Warn if computed max backoff time exceeds 600 seconds
        if max_retries > 0:
            max_backoff_time = sum(
                retry_delay * (retry_backoff ** attempt)
                for attempt in range(max_retries)
            )
            if max_backoff_time > 600.0:
                logger.warning(
                    f"Retry configuration produces high max backoff time: "
                    f"{max_backoff_time:.1f}s (threshold: 600s). "
                    f"max_retries={max_retries}, retry_delay={retry_delay}, "
                    f"retry_backoff={retry_backoff}"
                )

        # Speed tracker (for adaptive timeouts)
        speed_tracking_config = getattr(self.download_config, 'speed_tracking', None)
        speed_tracking_enabled = False
        if speed_tracking_config:
            try:
                enabled_val = getattr(speed_tracking_config, 'enabled', False)
                speed_tracking_enabled = enabled_val is True  # Strict check, not just truthy
            except (TypeError, ValueError):
                speed_tracking_enabled = False

        if speed_tracking_enabled:
            # Get config values with safe defaults (handles MagicMock in tests)
            try:
                window_size = int(getattr(speed_tracking_config, 'window_size', 5))
            except (TypeError, ValueError):
                window_size = 5
            try:
                min_speed = float(getattr(speed_tracking_config, 'min_speed_mbps', 1.0))
            except (TypeError, ValueError):
                min_speed = 1.0
            try:
                max_mult = float(getattr(speed_tracking_config, 'max_timeout_multiplier', 2.0))
            except (TypeError, ValueError):
                max_mult = 2.0
            try:
                adaptive_val = getattr(speed_tracking_config, 'enable_adaptive_timeout', True)
                adaptive = adaptive_val is True
            except (TypeError, ValueError):
                adaptive = True

            # Rate limit signal detection config (US-002)
            try:
                rate_limit_threshold = float(getattr(speed_tracking_config, 'rate_limit_signal_threshold', 0.1))
            except (TypeError, ValueError):
                rate_limit_threshold = 0.1
            try:
                consecutive_slow = int(getattr(speed_tracking_config, 'consecutive_slow_samples', 3))
            except (TypeError, ValueError):
                consecutive_slow = 3

            # Variance detection config (US-93-012)
            try:
                variance_threshold = float(getattr(speed_tracking_config, 'variance_threshold', 0.5))
            except (TypeError, ValueError):
                variance_threshold = 0.5
            try:
                slow_warning_threshold = float(getattr(speed_tracking_config, 'slow_download_warning_threshold', 0.5))
            except (TypeError, ValueError):
                slow_warning_threshold = 0.5
            try:
                variance_enabled = getattr(speed_tracking_config, 'enable_variance_detection', True)
                variance_enabled = variance_enabled is True
            except (TypeError, ValueError):
                variance_enabled = True

            self.speed_tracker = DownloadSpeedTracker(
                DownloadSpeedConfig(
                    enabled=True,
                    window_size=window_size,
                    min_speed_mbps=min_speed,
                    max_timeout_multiplier=max_mult,
                    enable_adaptive_timeout=adaptive,
                    rate_limit_signal_threshold=rate_limit_threshold,
                    consecutive_slow_samples=consecutive_slow,
                    variance_threshold=variance_threshold,
                    slow_download_warning_threshold=slow_warning_threshold,
                    enable_variance_detection=variance_enabled
                )
            )
            logger.debug("Speed tracker enabled for adaptive timeouts")
        else:
            self.speed_tracker = DownloadSpeedTracker(DownloadSpeedConfig(enabled=False))

        # Share speed tracker with audio-first pipeline for speed-based escalation
        self.audio_first.speed_tracker = self.speed_tracker

        # Circuit breaker (for search failure protection)
        circuit_breaker_config = getattr(self.download_config, 'circuit_breaker', None)
        circuit_breaker_enabled = False
        if circuit_breaker_config:
            try:
                enabled_val = getattr(circuit_breaker_config, 'enabled', False)
                circuit_breaker_enabled = enabled_val is True
            except (TypeError, ValueError):
                circuit_breaker_enabled = False

        if circuit_breaker_enabled:
            # Get config values with safe defaults
            try:
                threshold = int(getattr(circuit_breaker_config, 'consecutive_failures_threshold', 5))
            except (TypeError, ValueError):
                threshold = 5
            try:
                pause_secs = float(getattr(circuit_breaker_config, 'pause_seconds', 60.0))
            except (TypeError, ValueError):
                pause_secs = 60.0

            self.circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    enabled=True,
                    consecutive_failures_threshold=threshold,
                    pause_seconds=pause_secs
                )
            )
            logger.debug(
                f"Circuit breaker enabled: trips after {threshold} failures, "
                f"pauses for {pause_secs:.0f}s"
            )
        else:
            self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig(enabled=False))

        # Batch retry queue (for rate-limited videos)
        batch_retry_config = getattr(self.download_config, 'batch_retry', None)
        batch_retry_enabled = False
        if batch_retry_config:
            try:
                enabled_val = getattr(batch_retry_config, 'enabled', False)
                batch_retry_enabled = enabled_val is True
            except (TypeError, ValueError):
                batch_retry_enabled = False

        if batch_retry_enabled:
            # Get config values with safe defaults
            try:
                delay_secs = float(getattr(batch_retry_config, 'delay_seconds', 120.0))
            except (TypeError, ValueError):
                delay_secs = 120.0
            try:
                max_passes = int(getattr(batch_retry_config, 'max_passes', 2))
            except (TypeError, ValueError):
                max_passes = 2
            try:
                respect_cb_val = getattr(batch_retry_config, 'respect_circuit_breaker', True)
                respect_circuit_breaker = respect_cb_val is True
            except (TypeError, ValueError):
                respect_circuit_breaker = True
            try:
                wait_cookie_val = getattr(batch_retry_config, 'wait_for_cookie_cooldown', True)
                wait_for_cookie_cooldown = wait_cookie_val is True
            except (TypeError, ValueError):
                wait_for_cookie_cooldown = True

            self.retry_queue = RetryQueue(
                BatchRetryConfig(
                    enabled=True,
                    delay_seconds=delay_secs,
                    max_passes=max_passes,
                    respect_circuit_breaker=respect_circuit_breaker,
                    wait_for_cookie_cooldown=wait_for_cookie_cooldown
                )
            )
            logger.debug(
                f"Batch retry enabled: {delay_secs:.0f}s delay, "
                f"max {max_passes} passes, respect_circuit_breaker={respect_circuit_breaker}, "
                f"wait_for_cookie_cooldown={wait_for_cookie_cooldown}"
            )
        else:
            self.retry_queue = RetryQueue(BatchRetryConfig(enabled=False))

        # Link circuit breaker to retry queue for coordination (US-003)
        self.retry_queue.set_circuit_breaker(self.circuit_breaker)

        # Link cookie rotator to retry queue for cooldown coordination (US-007)
        if self.cookie_rotator and self.cookie_rotator.is_enabled:
            self.retry_queue.set_cookie_rotator(self.cookie_rotator)

        # US-129-002: Link retry budget tracker to retry queue for per-video limits
        from .retry_queue import DownloadRetryBudget
        retry_budget_config = getattr(self.download_config, 'retry_budget', None)
        self._retry_budget = DownloadRetryBudget(retry_budget_config)
        self.retry_queue.processor.set_retry_budget(self._retry_budget)
        if self._retry_budget.enabled:
            logger.debug(
                f"Download retry budget enabled: max_attempts={retry_budget_config.max_attempts}, "
                f"max_backoff_time_seconds={retry_budget_config.max_backoff_time_seconds}s"
            )

        # Rate limiting metrics tracking (US-010)
        self.rate_limit_metrics = RateLimitMetrics()

        # Cross-keyword rate limit budget tracking (US-004, US-009)
        # Read from dedicated RateLimitBudgetConfig if available, fall back to scattered configs
        budget_config = getattr(self.download_config, 'rate_limit_budget', None)
        if budget_config is not None:
            # New path: use dedicated RateLimitBudgetConfig (US-009)
            self._share_budget_across_keywords = getattr(budget_config, 'enabled', True)
            self.rate_limit_budget = RateLimitBudget.from_config(budget_config)
        else:
            # Legacy path: read limits from scattered config sections
            self._share_budget_across_keywords = getattr(
                rate_limit_config, 'share_budget_across_keywords', True
            ) if rate_limit_config else True

            self.rate_limit_budget = RateLimitBudget()
            if rate_limit_config:
                try:
                    max_backoff_budget = float(getattr(rate_limit_config, 'max_backoff_budget', 300.0))
                except (TypeError, ValueError):
                    max_backoff_budget = 300.0
                self.rate_limit_budget.max_backoff_time = max_backoff_budget

            if cookie_rotation_config:
                max_rotations = getattr(cookie_rotation_config, 'max_rotations_per_session', 0)
                self.rate_limit_budget.max_rotations = max_rotations

            if vpn_config:
                max_vpn_switches = getattr(vpn_config, 'max_switches_per_session', 10)
                self.rate_limit_budget.max_vpn_switches = max_vpn_switches

        if self._share_budget_across_keywords:
            logger.debug("Cross-keyword rate limit budget sharing enabled")

        # Wire budget into escalation manager for budget-aware escalation (US-002)
        if self.escalation_manager is not None:
            self.escalation_manager._budget = self.rate_limit_budget
            logger.debug("Rate limit budget wired into escalation manager")

        # Global rate limit coordinator (US-35-002: unified slot-based rate limiting)
        # Reads config from the top-level rate_limit section so all yt-dlp paths share one budget.
        pipeline_rate_limit_config = None
        if isinstance(self.config, dict):
            pipeline_rate_limit_config = self.config.get('rate_limit')
        else:
            config_dict = getattr(self.config, '__dict__', {})
            if 'rate_limit' in config_dict or hasattr(type(self.config), 'rate_limit'):
                pipeline_rate_limit_config = getattr(self.config, 'rate_limit', None)

        self.rate_limit_coordinator = GlobalRateLimitCoordinator(
            build_coordinator_rate_limit_config(pipeline_rate_limit_config)
        )
        self.audio_first.rate_limit_coordinator = self.rate_limit_coordinator
        if self.rate_limit_coordinator.is_enabled():
            logger.debug(
                f"Global rate limit coordinator enabled: "
                f"{self.rate_limit_coordinator._config.slots_per_second} slots/sec, "
                f"burst={self.rate_limit_coordinator._config.burst_size}"
            )

        # US-93-011: Initialize download metrics exporter
        self.metrics_exporter: Optional[DownloadMetricsExporter] = None
        download_metrics_config = getattr(self.download_config, 'metrics_exporter', None)
        if download_metrics_config:
            try:
                self.metrics_exporter = create_metrics_exporter(
                    config={
                        'enabled': getattr(download_metrics_config, 'enabled', True),
                        'export_interval_seconds': getattr(download_metrics_config, 'export_interval_seconds', 60.0),
                        'output_dir': getattr(download_metrics_config, 'output_dir', 'output/download_metrics'),
                        'export_prometheus': getattr(download_metrics_config, 'export_prometheus', True),
                        'export_json': getattr(download_metrics_config, 'export_json', True),
                    },
                    rate_limit_metrics=self.rate_limit_metrics,
                    coordinator=self.rate_limit_coordinator if self.rate_limit_coordinator.is_enabled() else None,
                )
                self.metrics_exporter.start()
                logger.info("Download metrics exporter initialized")
            except Exception as e:
                log_error_with_context(
                    logger, "DL-001",
                    f"Failed to initialize metrics exporter: {e}",
                    component="metrics_exporter"
                )

        # US-129-011: Initialize global cache for cross-project deduplication
        global_cache_config = getattr(self.config, 'global_cache', None)
        use_global_cache = True  # Default to enabled
        if global_cache_config:
            use_global_cache = getattr(global_cache_config, 'enabled', True)
        if use_global_cache:
            try:
                cache_dir = str(getattr(global_cache_config, 'cache_dir', '~/.matcher_global_cache'))
                self._global_cache = GlobalCacheManager(cache_dir=cache_dir, config=global_cache_config)
                logger.debug("Global cache initialized for cross-project deduplication")
            except Exception as e:
                log_error_with_context(
                    logger, "DL-001",
                    f"Failed to initialize global cache: {e}",
                    component="global_cache"
                )
                self._global_cache = None
        else:
            self._global_cache = None
            logger.debug("Global cache disabled")

    # =========================================================================
    # DELEGATION METHODS (Delegate to specialized managers)
    # =========================================================================

    def _get_tier_value(self, tier: str, key: str, default: int = 0) -> int:
        """Delegate to CheckpointManager."""
        return self.checkpoint_mgr.get_tier_value(tier, key, default)

    def _load_checkpoint(self) -> Optional[DownloadCheckpoint]:
        """Delegate to CheckpointManager."""
        return self.checkpoint_mgr.load_checkpoint()

    def _save_checkpoint(self):
        """Delegate to CheckpointManager, including speed tracker state and metrics."""
        if self.checkpoint:
            # Include speed tracker state in checkpoint for resume
            self.checkpoint.speed_tracker_state = self.speed_tracker.to_checkpoint_dict()
            # Include rate limit metrics in checkpoint for cross-session analysis (US-010)
            self.checkpoint.rate_limit_metrics = self.rate_limit_metrics.to_dict()
            # Include cross-keyword rate limit budget in checkpoint (US-004)
            if self._share_budget_across_keywords:
                self.checkpoint.rate_limit_budget = self.rate_limit_budget.to_dict()
            # Include VPN manager state for switch count persistence (US-005)
            if self.vpn_manager and self.vpn_manager.is_enabled:
                self.checkpoint.vpn_manager_state = self.vpn_manager.to_checkpoint_state()
            # Include MullvadVPN state for rotation limit persistence (US-129-007)
            if self.mullvad_vpn and self.mullvad_vpn.is_enabled:
                self.checkpoint.mullvad_vpn_state = self.mullvad_vpn.to_checkpoint_state()
            # Include escalation manager state for resume support (Sprint 10 US-007)
            if self.escalation_manager is not None:
                self.checkpoint.escalation_state = self.escalation_manager.to_dict()
            # Persist impersonation manager state for resume support
            if self.impersonation_manager is not None:
                self.checkpoint.impersonation_state = self.impersonation_manager.to_dict()
            # Include per-tier backoff state for resume support (Sprint 12 US-003)
            if self._per_tier_isolation:
                self.checkpoint.tier_backoff_state = {
                    'tiers': {k: v.to_dict() for k, v in self._tier_rate_limit_states.items()},
                    'saved_at': datetime.now().isoformat()
                }
            # US-129-002: Include retry budget state for per-video limits
            if hasattr(self, '_retry_budget') and self._retry_budget:
                self.checkpoint.retry_budget_state = self._retry_budget.to_checkpoint_dict()
            # US-136-007: Include region success tracking state for dynamic region backoff
            if RegionSuccessTracker.is_enabled():
                self.checkpoint.region_success_state = RegionSuccessTracker.to_checkpoint_state()
            self.checkpoint_mgr.save_checkpoint(self.checkpoint)

    def _clear_checkpoint(self):
        """Delegate to CheckpointManager."""
        self.checkpoint_mgr.clear_checkpoint()
        self.checkpoint = None

    def _save_sources(self):
        """Delegate to CheckpointManager."""
        self.checkpoint_mgr.save_sources(self.sources)

    def _search_video_metadata(self, keyword: str, tier: str, max_results: int = 50):
        """Delegate to TitleFilter."""
        return self.title_filter.search_video_metadata(keyword, tier, max_results)

    # =========================================================================
    # RATE LIMIT COORDINATION (US-35-002)
    # =========================================================================

    def acquire_download_slot(self, timeout: float = 30.0) -> bool:
        """Acquire a rate limit slot before downloading.

        Uses the global rate limit coordinator to ensure downloads
        don't exceed the configured rate limit across all operations.

        Args:
            timeout: Maximum time to wait for slot (seconds)

        Returns:
            True if slot acquired, False if timeout
        """
        acquired = self.rate_limit_coordinator.acquire_slot('download', timeout=timeout)
        if acquired:
            logger.debug("Download slot acquired")
        else:
            log_error_with_context(
                logger, "DL-006",
                f"Failed to acquire download slot after {timeout}s",
                timeout_seconds=timeout
            )
            self.rate_limit_metrics.record_slot_timeout()
        return acquired

    def release_download_slot(self) -> None:
        """Release a rate limit slot after download completes."""
        self.rate_limit_coordinator.release_slot('download')
        logger.debug("Download slot released")

    def _filter_titles_with_llm(self, videos, keyword, topic=""):
        """Delegate to TitleFilter."""
        return self.title_filter.filter_titles_with_llm(videos, keyword, topic)

    def _screen_approved_videos(self, approved_videos, keyword):
        """Delegate to SpeechScreener."""
        return self.speech_screener.screen_approved_videos(approved_videos, keyword)

    def _get_remix_keyword(self, keyword, topic=""):
        """Delegate to SearchOptimizer."""
        return self.search_optimizer.get_remix_keyword(keyword, topic)

    def _get_retry_keyword(self, keyword: str, retry_count: int) -> str:
        """Delegate to SearchOptimizer."""
        return self.search_optimizer.get_retry_keyword(keyword, retry_count)

    def _get_adaptive_search_pool(self, keyword, max_downloads):
        """Delegate to SearchOptimizer."""
        return self.search_optimizer.get_adaptive_search_pool(keyword, max_downloads)

    def _record_search_pass_rate(self, keyword, searched, approved):
        """Delegate to SearchOptimizer."""
        self.search_optimizer.record_search_pass_rate(keyword, searched, approved)

    def _record_source_for_keyword(self, keyword: str, video_id: str):
        """Delegate to SearchOptimizer."""
        self.search_optimizer.record_source_for_keyword(keyword, video_id)

    def log_source_diversity_report(self):
        """Delegate to SearchOptimizer."""
        self.search_optimizer.log_source_diversity_report()

    def get_rate_limit_metrics(self) -> RateLimitMetrics:
        """
        Get aggregated rate limiting metrics for reporting.

        Combines metrics from the internal tracker with stats from subsystems
        (circuit breaker, retry queue, speed tracker).

        Returns:
            RateLimitMetrics with all aggregated statistics.
        """
        # Update metrics from subsystems
        self.rate_limit_metrics.update_from_circuit_breaker(self.circuit_breaker.get_stats())
        self.rate_limit_metrics.update_from_retry_queue(self.retry_queue.get_stats())
        self.rate_limit_metrics.update_from_speed_tracker(self.speed_tracker.get_speed_stats())

        # Get cookie rotation and VPN counts from rotators if available
        if self.cookie_rotator and self.cookie_rotator.is_enabled:
            status = self.cookie_rotator.get_status()
            self.rate_limit_metrics.cookie_rotations = status.get('rotations', 0)
        if self.vpn_manager and self.vpn_manager.is_enabled:
            status = self.vpn_manager.get_status()
            self.rate_limit_metrics.vpn_switches = status.get('switches', 0)

        # Get speed escalation count from escalation manager
        if hasattr(self, 'escalation_manager') and self.escalation_manager is not None:
            esc_metrics = self.escalation_manager.get_metrics()
            self.rate_limit_metrics.speed_escalations = esc_metrics.get('speed_escalations', 0)

        return self.rate_limit_metrics

    def close(self) -> None:
        """Clean up resources including the metrics exporter."""
        if self.metrics_exporter is not None:
            try:
                self.metrics_exporter.stop()
                logger.info("Download metrics exporter stopped")
            except Exception as e:
                log_error_with_context(
                    logger, "DL-001",
                    f"Error stopping metrics exporter: {e}",
                    component="metrics_exporter"
                )

    def get_speed_stats(self) -> dict:
        """Get download speed statistics for reporting."""
        return self.speed_tracker.get_speed_stats()

    # =========================================================================
    # SELF-REGULATION THROTTLING (US-123-010)
    # =========================================================================

    def _get_self_regulate_config(self):
        """Get self-regulate config from download config.

        Returns:
            SelfRegulateConfig or None if not enabled
        """
        regulate_config = getattr(self.download_config, 'self_regulate', None)
        if regulate_config is None:
            return None
        # Check if enabled
        if hasattr(regulate_config, 'enabled'):
            if not regulate_config.enabled:
                return None
        return regulate_config

    def _get_download_coordinator(self):
        """Return the active DownloadCoordinator if one has been assigned."""
        coordinator = getattr(self, 'download_coordinator', None)
        if coordinator is None:
            logger.debug("Self-regulation skipped: DownloadCoordinator is not initialized")
        return coordinator

    def _apply_self_regulation(self, error_type: str = "rate_limit") -> None:
        """Apply self-regulation throttling based on error patterns.

        Called when errors occur to record them and potentially throttle.

        Args:
            error_type: Type of error (e.g., 'rate_limit', '429', '403')
        """
        regulate_config = self._get_self_regulate_config()
        if regulate_config is None:
            return

        download_coordinator = self._get_download_coordinator()
        if download_coordinator is None:
            return

        # Record the error
        self.speed_tracker.record_error(error_type)

        # Get current concurrency from coordinator
        current_concurrency = download_coordinator.get_current_concurrency()

        # Check if we should throttle
        throttle_signal = self.speed_tracker.should_throttle(
            threshold=regulate_config.rate_limit_prevention_threshold,
            window_seconds=regulate_config.rate_limit_prevention_window_seconds,
            current_concurrency=current_concurrency,
            throttle_factor=regulate_config.throttle_factor,
            min_concurrent=regulate_config.min_concurrent
        )

        if throttle_signal.should_throttle:
            # Adjust coordinator concurrency
            download_coordinator.set_concurrency(throttle_signal.new_concurrency)
            logger.info(f"Self-regulation throttled: {throttle_signal.reason}")

    def _check_self_regulation_recovery(self) -> None:
        """Check if we should recover (increase) concurrency after error-free period.

        Called periodically or after successful downloads to gradually restore speed.
        """
        regulate_config = self._get_self_regulate_config()
        if regulate_config is None:
            return

        download_coordinator = self._get_download_coordinator()
        if download_coordinator is None:
            return

        current_concurrency = download_coordinator.get_current_concurrency()

        # Get max recovery concurrency
        max_recovery = regulate_config.max_recovery_concurrency
        if max_recovery <= 0:
            # Use original max from config
            max_recovery = getattr(self.download_config, 'max_concurrent', 4)
            if max_recovery is None:
                max_recovery = getattr(self.download_config, 'parallel_workers', 4)

        # Check if we should recover
        recovery_signal = self.speed_tracker.should_recover(
            recovery_window_seconds=regulate_config.recovery_window_seconds,
            current_concurrency=current_concurrency,
            recovery_factor=regulate_config.recovery_factor,
            max_recovery_concurrency=max_recovery
        )

        if recovery_signal.should_recover:
            # Adjust coordinator concurrency
            download_coordinator.set_concurrency(recovery_signal.new_concurrency)
            logger.info(f"Self-regulation recovered: {recovery_signal.reason}")

    def _build_format_string(self):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.build_format_string()

    def _build_filter_string(self, tier):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.build_filter_string(tier, self.DURATION_TIERS)

    def _needs_transcoding(self, video_path):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.needs_transcoding(video_path)

    def _get_ffmpeg_transcode_cmd(self, input_path, output_path):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.get_ffmpeg_transcode_cmd(input_path, output_path)

    # =========================================================================
    # CORE ORCHESTRATION METHODS (Keep in core - complex coordination logic)
    # =========================================================================

    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if yt-dlp and ffmpeg are installed.

        Returns:
            Tuple of (success: bool, message: str)
        """
        messages = []

        # Check yt-dlp
        try:
            result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True, encoding='utf-8', errors='replace', **SUBPROCESS_FLAGS)
            messages.append(f"✓ yt-dlp {result.stdout.strip()}")
        except FileNotFoundError:
            return False, "✗ yt-dlp not found! Install with: pip install yt-dlp"

        # Check ffmpeg (required for DaVinci mode)
        if self.download_config.davinci_mode:
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, **SUBPROCESS_FLAGS)
                hw_accel = self.transcoding_mgr.hw_accel
                hw_names = {
                    'nvidia': 'NVIDIA NVENC',
                    'amd': 'AMD AMF',
                    'intel': 'Intel QuickSync',
                    'mac': 'Apple VideoToolbox',
                    'none': 'CPU'
                }
                messages.append(f"✓ FFmpeg ({hw_names.get(hw_accel, 'Unknown')})")
            except FileNotFoundError:
                return False, "✗ FFmpeg not found! Required for DaVinci mode."

        return True, "\n".join(messages)

    def _cleanup_partial_files(self, directory: Path, video_id: str) -> None:
        """
        Remove partial download files for a video ID.

        Cleans up .part, .ytdl, and other temporary files left by failed/timed out downloads.
        """
        if not directory.exists():
            return

        patterns = [f"{video_id}.*part*", f"{video_id}.*.ytdl", f"{video_id}.ytdl"]
        for pattern in patterns:
            for partial_file in directory.glob(pattern):
                try:
                    partial_file.unlink()
                    logger.debug(f"Cleaned up partial file: {partial_file.name}")
                except Exception as e:
                    logger.debug(f"Could not remove {partial_file.name}: {e}")

    def _find_cookies_file(self) -> Optional[Path]:
        """
        Find cookies.txt file for YouTube authentication.

        Searches in order:
        1. Explicit path from config (download.cookies_path)
        2. Install directory (same as config.yaml)
        3. Project directory
        4. Current working directory
        5. User home directory
        """
        search_locations = []

        # 0. Check if explicit path is set in config
        if hasattr(self.download_config, 'cookies_path') and self.download_config.cookies_path:
            explicit_path = Path(self.download_config.cookies_path)
            if explicit_path.exists():
                return explicit_path
            else:
                log_error_with_context(
                    logger, "DL-004",
                    f"Configured cookies_path does not exist: {explicit_path}",
                    cookies_path=str(explicit_path)
                )

        # 1. Install directory (where config.yaml is)
        if hasattr(self.config, '_config_path') and self.config._config_path:
            install_dir = Path(self.config._config_path).parent
            search_locations.append(install_dir / 'cookies.txt')

        # 2. Project directory
        if hasattr(self.config, 'project_dir') and self.config.project_dir:
            search_locations.append(Path(self.config.project_dir) / 'cookies.txt')

        # 3. Current working directory
        search_locations.append(Path.cwd() / 'cookies.txt')

        # 4. User home directory
        search_locations.append(Path.home() / 'cookies.txt')

        for path in search_locations:
            if path.exists():
                return path

        return None

    def _add_impersonation_to_cmd(self, cmd: list) -> Optional[str]:
        """Add browser impersonation args to yt-dlp command.

        Injects --impersonate with the next rotated target from the
        ImpersonationManager. Must be called BEFORE _add_cookies_to_cmd
        to maintain correct yt-dlp argument ordering.

        When impersonation is disabled or no targets are available,
        this is a no-op (command unchanged).

        Returns:
            The impersonation target string, or None if not applied.
        """
        if self.impersonation_manager:
            args, target = self.impersonation_manager.get_impersonate_args_with_target()
            if args:
                # DEBUG-level logging for impersonation changes
                logger.debug(f"Applying impersonation args: {args}")
                cmd.extend(args)
            return target
        return None

    def _add_escalation_to_cmd(self, cmd: list, keyword: str) -> Tuple[Optional[EscalationResult], Optional[str]]:
        """Add escalation-aware bypass args to yt-dlp command.

        Uses EscalationManager when available (respects current tier per keyword).
        Falls back to direct impersonation when escalation is not available.

        Args:
            cmd: The yt-dlp command list to extend.
            keyword: The download keyword or video ID for per-keyword escalation.

        Returns:
            Tuple of (EscalationResult or None, impersonation_target or None).
        """
        if self.escalation_manager:
            result = self.escalation_manager.get_escalation_args(keyword)
            if result.args:
                cmd.extend(result.args)
            return result, getattr(result, 'impersonation_target', None)
        # Fallback: direct impersonation only (no escalation manager)
        target = self._add_impersonation_to_cmd(cmd)
        return None, target

    def _add_cookies_to_cmd(self, cmd: list) -> None:
        """Add cookie authentication to yt-dlp command."""
        # Use cookie rotator if enabled, otherwise fall back to static cookie
        if self.cookie_rotator and self.cookie_rotator.is_enabled:
            current_cookie = self.cookie_rotator.get_current_cookie()
            if current_cookie:
                cmd.extend(['--cookies', current_cookie])
                return

        # Fallback to static cookie configuration
        if self._cookies_from_browser:
            cmd.extend(['--cookies-from-browser', self._cookies_from_browser])
        elif self._cookies_path:
            cmd.extend(['--cookies', str(self._cookies_path)])

    def _add_bandwidth_throttle_to_cmd(self, cmd: list, estimated_size_mb: float = None) -> None:
        """Add bandwidth throttling to yt-dlp command via --downloader-args.

        Uses yt-dlp's --downloader-args to pass ffmpeg bandwidth limiting option.
        The -bt (bitrate) option limits the bandwidth used by ffmpeg for downloads.

        Supports:
        - Size-based throttling (bypass small files)
        - Time-based throttling (peak/offpeak hours)
        - Integer kbps config (bandwidth_limit_kbps)
        - String format (global_limit like "5M")

        Args:
            cmd: The yt-dlp command list to modify
            estimated_size_mb: Estimated file size in MB (None to bypass size check)
        """
        # Get bandwidth throttle config
        bandwidth_config = getattr(self.download_config, 'bandwidth_throttle', None)
        if not bandwidth_config:
            return

        # Check if throttling is enabled
        if not bandwidth_config.enabled:
            return

        # Use the new unified get_limit method that handles both size and time
        limit = bandwidth_config.get_limit(file_size_mb=estimated_size_mb)

        if not limit:
            if estimated_size_mb:
                logger.debug(f"    Bandwidth throttling bypassed for small file ({estimated_size_mb:.1f}MB)")
            return

        # Add the downloader-args for ffmpeg bandwidth limiting
        # The -bt option sets the target bitrate for ffmpeg
        cmd.extend(['--downloader-args', f'ffmpeg:-bt {limit}'])

        if estimated_size_mb:
            logger.info(f"    Bandwidth throttled to {limit} for {estimated_size_mb:.1f}MB file")
        else:
            logger.info(f"    Bandwidth throttled to {limit}")

    def _replace_cookie_args_in_cmd(self, cmd: List[str]) -> List[str]:
        """Strip existing cookie args from cmd and append current fallback method's args."""
        cleaned = []
        skip_next = False
        for arg in cmd:
            if skip_next:
                skip_next = False
                continue
            if arg in ('--cookies', '--cookies-from-browser'):
                skip_next = True  # skip the arg and its value
                continue
            cleaned.append(arg)
        cleaned.extend(self.method_fallback.get_cmd_args())
        return cleaned

    def rotate_cookie_on_error(self, error_message: str, keyword: str = None) -> bool:
        """
        Attempt to rotate cookie based on error message.

        Args:
            error_message: Error string from yt-dlp
            keyword: Search keyword for budget tracking (optional)

        Returns:
            True if cookie was rotated, False otherwise
        """
        if not self.cookie_rotator:
            return False

        if self.cookie_rotator.should_rotate(error_message):
            # DEBUG-level logging for cookie rotation
            logger.debug(f"Cookie rotation triggered, error: {error_message[:100]}")
            new_cookie = self.cookie_rotator.rotate()
            if new_cookie:
                logger.info(f"Rotated to new cookie: {Path(new_cookie).name}")
                self.rate_limit_metrics.record_cookie_rotation()
                # Record in cross-keyword budget (US-004)
                if self._share_budget_across_keywords:
                    self.rate_limit_budget.record_rotation(keyword=keyword)
                return True
            else:
                # Get keyword from available context if possible
                kw = keyword if 'keyword' in dir() else None
                log_error_with_context(logger, "DL-004", "Cookie rotation exhausted", keyword=kw)

        return False

    def switch_vpn_on_error(self, keyword: str = None) -> bool:
        """
        Attempt to switch VPN server.

        Should be called after cookie rotation is exhausted.

        Args:
            keyword: Search keyword for budget tracking (optional)

        Returns:
            True if VPN was switched, False otherwise
        """
        if not self.vpn_manager:
            return False

        # Check cross-keyword budget if enabled (US-004)
        if self._share_budget_across_keywords:
            if not self.rate_limit_budget.can_switch_vpn():
                remaining = self.rate_limit_budget.vpn_switches_remaining()
                logger.warning(
                    f"VPN switch budget exhausted ({self.rate_limit_budget.vpn_switches_used} used, "
                    f"{remaining or 0} remaining)"
                )
                return False

        if self.vpn_manager.can_switch():
            success = self.vpn_manager.switch()
            if success:
                # Reset cookie rotator after VPN switch (new IP = fresh start)
                if self.cookie_rotator:
                    self.cookie_rotator.reset()
                self.rate_limit_metrics.record_vpn_switch()
                # Record in cross-keyword budget (US-004)
                if self._share_budget_across_keywords:
                    self.rate_limit_budget.record_vpn_switch(keyword=keyword)
                return True

        return False

    def handle_rate_limit_error(
        self, error_message: str, tier: str = None, keyword: str = None
    ) -> bool:
        """
        Handle rate limit or authentication error with progressive backoff.

        Tries in order:
        1. Progressive exponential backoff (until max_backoff_before_rotate reached)
        2. Cookie rotation (if enabled and available)
        3. VPN switch (if enabled and cookies exhausted)

        The backoff phase handles brief rate-limit windows without exhausting
        cookies too quickly.

        When in cooldown recovery mode (resumed within cooldown period), uses
        longer initial backoff and faster escalation to cookie/VPN rotation.

        When per_tier_isolation is enabled, each tier maintains independent
        backoff state. Rate limiting on 'long' tier won't affect 'short' tier.

        When share_budget_across_keywords is enabled (default), uses shared budget
        to skip exhausted escalation levels. If keyword A exhausted all cookie
        rotations, keyword B skips directly to VPN switching.

        Args:
            error_message: Error string from yt-dlp
            tier: Duration tier (short, medium, long, longer) for tier-specific tracking
            keyword: Search keyword for budget tracking (optional)

        Returns:
            True if recovery was attempted (backoff or rotation), False if no options left
        """
        # Record rate limit event for cross-session tracking and metrics
        self._rate_limit_event_count += 1
        self._record_rate_limit_event()
        self.rate_limit_metrics.record_rate_limit_event(tier=tier, keyword=keyword)

        # Get rate limit config settings
        rate_limit_config = getattr(self.download_config, 'rate_limit', None)
        initial_backoff = getattr(rate_limit_config, 'initial_backoff_seconds', 5.0) if rate_limit_config else 5.0
        max_backoff = getattr(rate_limit_config, 'max_backoff_before_rotate', 60.0) if rate_limit_config else 60.0
        backoff_multiplier = getattr(rate_limit_config, 'backoff_multiplier', 2.0) if rate_limit_config else 2.0
        adaptive_multiplier_enabled = getattr(rate_limit_config, 'adaptive_multiplier', True) if rate_limit_config else True

        # Classify error severity and adjust multiplier if adaptive mode enabled (US-008)
        severity = classify_error_severity(error_message)
        if adaptive_multiplier_enabled:
            backoff_multiplier = SEVERITY_MULTIPLIERS.get(severity, backoff_multiplier)
            log_rate_limit(
                logger, "backoff", "download", "error_classified",
                severity=severity, multiplier=backoff_multiplier
            )
        else:
            log_rate_limit(
                logger, "backoff", "download", "adaptive_disabled"
            )

        # US-114-010: Check for Retry-After header in error message
        # If present, use it instead of exponential backoff
        retry_after = parse_retry_after(error_message)
        using_retry_after = retry_after is not None

        # Get tier-specific state if isolation enabled
        if self._per_tier_isolation and tier and tier in self._tier_rate_limit_states:
            tier_state = self._tier_rate_limit_states[tier]
            backoff_count = tier_state.backoff_count
            total_delay = tier_state.total_delay
            in_recovery = tier_state.in_recovery or self._in_cooldown_recovery_mode
            tier_label = f" [{tier}]"
        else:
            # Fallback to global state
            tier_state = None
            backoff_count = self._rate_limit_backoff_count
            total_delay = self._rate_limit_total_delay
            in_recovery = self._in_cooldown_recovery_mode
            tier_label = ""

        # In recovery mode, use more aggressive settings
        if in_recovery:
            # Longer initial delay, shorter max before escalation
            initial_backoff = initial_backoff * 2
            max_backoff = max_backoff * 0.5  # Escalate faster to cookie/VPN rotation
            logger.debug(
                f"Cooldown recovery mode{tier_label}: initial_backoff={initial_backoff:.1f}s, "
                f"max_backoff={max_backoff:.1f}s"
            )

        # Check cross-keyword budget if enabled (US-004)
        # If budget sharing enabled and budget exhausted, skip lower-level escalations
        skip_backoff = False
        skip_rotation = False

        if self._share_budget_across_keywords:
            # Check if backoff budget is exhausted across keywords
            if not self.rate_limit_budget.can_backoff(initial_backoff):
                skip_backoff = True
                logger.info(
                    f"Backoff budget exhausted (spent: {self.rate_limit_budget.backoff_time_spent:.1f}s / "
                    f"{self.rate_limit_budget.max_backoff_time:.0f}s max), skipping backoff"
                )

            # Check if rotation budget is exhausted across keywords
            if not self.rate_limit_budget.can_rotate():
                skip_rotation = True
                remaining = self.rate_limit_budget.rotations_remaining()
                logger.info(
                    f"Cookie rotation budget exhausted ({self.rate_limit_budget.rotations_used} used, "
                    f"{remaining or 0} remaining), skipping to VPN"
                )

        # Check if we should try backoff first (before cookie rotation)
        if not skip_backoff and total_delay < max_backoff:
            # US-114-010: If Retry-After header was provided, use it instead of exponential backoff
            if using_retry_after and retry_after is not None:
                delay = retry_after
                # Cap based on cross-keyword budget if enabled
                if self._share_budget_across_keywords:
                    budget_remaining = self.rate_limit_budget.backoff_time_remaining()
                    if budget_remaining is not None:
                        delay = min(delay, budget_remaining)
                # Still cap at max_backoff to avoid excessively long waits
                remaining = max_backoff - total_delay
                delay = min(delay, remaining)
            else:
                # Calculate next backoff delay: initial * (multiplier ^ attempt)
                delay = initial_backoff * (backoff_multiplier ** backoff_count)

                # Cap delay so we don't exceed max_backoff total
                remaining = max_backoff - total_delay
                delay = min(delay, remaining)

                # Also cap based on cross-keyword budget if enabled
                if self._share_budget_across_keywords:
                    budget_remaining = self.rate_limit_budget.backoff_time_remaining()
                    if budget_remaining is not None:
                        delay = min(delay, budget_remaining)

            if delay > 0:
                # Update tier-specific or global state
                # Note: For Retry-After, we don't increment backoff_count as it's server-specified
                if not using_retry_after:
                    if tier_state:
                        tier_state.backoff_count += 1
                        tier_state.total_delay += delay
                        tier_state.last_event_time = datetime.now().isoformat()
                    else:
                        self._rate_limit_backoff_count += 1
                        self._rate_limit_total_delay += delay

                self.rate_limit_metrics.record_backoff(delay, severity=severity)

                # Record in cross-keyword budget (US-004)
                if self._share_budget_across_keywords:
                    self.rate_limit_budget.record_backoff(delay, keyword=keyword)

                recovery_note = " (recovery mode)" if in_recovery else ""
                severity_note = f" [{severity}]" if adaptive_multiplier_enabled else ""
                retry_after_note = " (Retry-After)" if using_retry_after else ""
                new_count = tier_state.backoff_count if tier_state else self._rate_limit_backoff_count
                new_total = tier_state.total_delay if tier_state else self._rate_limit_total_delay
                logger.info(
                    f"Rate limit backoff{tier_label}{severity_note} {new_count}{recovery_note}{retry_after_note}: "
                    f"waiting {delay:.1f}s (total: {new_total:.1f}s / {max_backoff:.0f}s max)"
                )
                self._mock_sleep(delay)
                return True

        # Backoff exhausted - reset counters and escalate to cookie rotation
        if total_delay > 0:
            logger.info(
                f"Rate limit backoff{tier_label} exhausted after {total_delay:.1f}s total delay, "
                "escalating to cookie rotation"
            )
            self._reset_rate_limit_backoff(tier=tier)

        # Try cookie rotation (unless budget exhausted)
        if not skip_rotation and self.rotate_cookie_on_error(error_message, keyword=keyword):
            return True

        # Try VPN switch if cookies exhausted
        if self.switch_vpn_on_error(keyword=keyword):
            return True

        return False

    def _record_rate_limit_event(self) -> None:
        """Record rate limit event to checkpoint for cross-session tracking."""
        if self.checkpoint:
            self.checkpoint.last_rate_limit_timestamp = datetime.now().isoformat()
            self.checkpoint.rate_limit_event_count = self._rate_limit_event_count
            self._save_checkpoint()

    def _reset_rate_limit_backoff(self, tier: str = None) -> None:
        """Reset rate limit backoff state after successful download or cookie rotation.

        When per_tier_isolation is enabled, only resets the specified tier's state.
        Otherwise, resets global state.

        Args:
            tier: Duration tier to reset (short, medium, long, longer), or None for global
        """
        if self._per_tier_isolation and tier and tier in self._tier_rate_limit_states:
            self._tier_rate_limit_states[tier].reset()
            logger.debug(f"Reset rate limit backoff for tier '{tier}'")
        else:
            # Reset global state
            self._rate_limit_backoff_count = 0
            self._rate_limit_total_delay = 0.0

    def _mock_sleep(self, delay: float) -> None:
        """Apply mock delay for rate limiting in test mode.

        When mock rate limits are enabled (via config or MOCK_RATE_LIMITS env var),
        this sleeps for mock_delay_seconds instead of the actual delay.

        Args:
            delay: The intended delay in seconds (used for logging)
        """
        test_mode_config = getattr(self.config, 'test_mode', None)
        test_mode_active = bool(getattr(self.config, '_test_mode', False))
        if test_mode_active and is_mock_rate_limits_enabled(test_mode_config):
            mock_delay = get_mock_delay_seconds(test_mode_config)
            if mock_delay > 0:
                logger.debug(f"Mock rate limit: sleeping {mock_delay:.3f}s instead of {delay:.1f}s")
                time.sleep(mock_delay)
        else:
            # Normal delay
            time.sleep(delay)

    def _restore_tier_backoff_state(self, tier_state_data: dict) -> None:
        """Restore per-tier backoff state from checkpoint with staleness de-escalation.

        If checkpoint is older than 30 minutes, backoff delays are halved (staleness
        de-escalation), since YouTube rate limits may have relaxed.

        Args:
            tier_state_data: Dict with 'tiers' and 'saved_at' keys from checkpoint
        """
        try:
            tiers_data = tier_state_data.get('tiers', {})
            saved_at_str = tier_state_data.get('saved_at')

            # Calculate staleness factor
            staleness_factor = 1.0
            if saved_at_str:
                try:
                    saved_at = datetime.fromisoformat(saved_at_str)
                    age_minutes = (datetime.now() - saved_at).total_seconds() / 60
                    if age_minutes > 30:
                        staleness_factor = 0.5
                        logger.info(
                            f"Tier backoff state is {age_minutes:.0f}min old — "
                            f"applying 50% staleness de-escalation"
                        )
                except (ValueError, TypeError):
                    log_error_with_context(
                        logger, "DL-006",
                        "Invalid saved_at in tier_backoff_state, using fresh state",
                        component="tier_backoff_state"
                    )
                    return

            restored_count = 0
            for tier_name, state_data in tiers_data.items():
                if tier_name in self._tier_rate_limit_states:
                    state = TierRateLimitState.from_dict(state_data)
                    # Apply staleness de-escalation
                    if staleness_factor < 1.0:
                        state.total_delay *= staleness_factor
                    self._tier_rate_limit_states[tier_name] = state
                    if state.backoff_count > 0:
                        restored_count += 1

            if restored_count > 0:
                logger.info(
                    f"Restored tier backoff state: "
                    f"{restored_count} tiers with active backoff"
                )
            else:
                logger.debug("Restored tier backoff state: all tiers clean")
        except Exception as e:
            log_error_with_context(
                logger, "DL-006",
                f"Could not restore tier backoff state: {e} — starting fresh",
                component="tier_backoff_state"
            )

    def _check_rate_limit_cooldown(self, checkpoint: DownloadCheckpoint) -> bool:
        """
        Check if last rate limit was within cooldown period.

        If within cooldown, enables recovery mode with longer delays and
        faster escalation to cookie/VPN rotation.

        Args:
            checkpoint: Loaded checkpoint with rate limit history

        Returns:
            True if within cooldown period, False otherwise
        """
        if not checkpoint.last_rate_limit_timestamp:
            return False

        try:
            last_rate_limit = datetime.fromisoformat(checkpoint.last_rate_limit_timestamp)
        except (ValueError, TypeError):
            logger.debug("Invalid rate limit timestamp in checkpoint, ignoring cooldown")
            return False

        # Get cooldown config
        rate_limit_config = getattr(self.download_config, 'rate_limit', None)
        cooldown_minutes = getattr(rate_limit_config, 'resume_cooldown_minutes', 15.0) if rate_limit_config else 15.0

        # Calculate time since last rate limit
        now = datetime.now()
        minutes_since = (now - last_rate_limit).total_seconds() / 60

        if minutes_since < cooldown_minutes:
            logger.info(
                f"⚠ Rate limit cooldown: Last rate limit was {minutes_since:.1f} min ago "
                f"(cooldown: {cooldown_minutes:.0f} min). "
                f"Previous session had {checkpoint.rate_limit_event_count} rate limit events."
            )
            logger.info(
                "  Enabling recovery mode: longer delays, faster escalation to cookie/VPN rotation"
            )
            return True
        else:
            logger.info(
                f"✓ Rate limit cooldown cleared: {minutes_since:.1f} min since last rate limit "
                f"(cooldown: {cooldown_minutes:.0f} min)"
            )
            return False

    def download_all(
        self,
        keywords: List[str],
        output_dir: Path,
        max_concurrent: int = None,
        resume: bool = False,
        topic: str = ""
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Download videos for all keywords.

        Delegates to DownloadOrchestrator for batch coordination.

        Args:
            keywords: List of search keywords
            output_dir: Output directory
            max_concurrent: Max concurrent downloads (uses config.download.parallel_workers if None)
            resume: Whether to resume from checkpoint
            topic: Topic context for LLM title filtering

        Returns:
            Tuple of (downloaded_videos, failed_keywords)
        """
        # Lazy import to avoid circular dependency
        from .orchestrator import DownloadOrchestrator
        orchestrator = DownloadOrchestrator(self)
        return orchestrator.download_all(
            keywords=keywords,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            resume=resume,
            topic=topic
        )

    def download_for_keyword(
        self,
        keyword: str,
        output_dir: Path,
        tiers: List[str] = None,
        topic: str = ""
    ) -> List[DownloadedVideo]:
        """
        Download videos for a keyword across specified tiers.

        If download times out, will retry with modified keyword (up to 2 retries).

        Args:
            keyword: Search keyword
            output_dir: Output directory
            tiers: List of tier names (defaults to all tiers)
            topic: Topic context for LLM filtering

        Returns:
            List of DownloadedVideo objects
        """
        if tiers is None:
            tiers = list(self.DURATION_TIERS.keys())

        all_downloaded = []

        for tier in tiers:
            # Skip tiers with per_keyword=0
            per_kw = self._get_tier_value(tier, 'per_keyword', 5)
            if per_kw <= 0:
                logger.debug(f"  [{tier}] Skipped (0/kw)")
                continue

            # Check max_total limit for this tier
            max_total = self._get_tier_value(tier, 'max_total', 0)
            if max_total > 0 and self.tier_download_counts.get(tier, 0) >= max_total:
                logger.debug(f"  [{tier}] Skipped (max_total={max_total} reached)")
                continue

            # FILE-BASED SKIP: Check if videos already exist
            max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
            safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
            safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
            tier_short = tier[0]
            keyword_dir = output_dir / f"{safe_keyword}_{tier_short}"

            if keyword_dir.exists():
                existing_videos = [f for f in os.listdir(keyword_dir)
                                   if f.endswith(('.mp4', '.mkv', '.webm'))]
                per_kw = self._get_tier_value(tier, 'per_keyword', 5)
                if len(existing_videos) >= per_kw:
                    logger.info(f"  [{tier}] Already have {len(existing_videos)} videos (skipping search + filter)")
                    # Count existing toward tier total
                    with self._lock:
                        self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(existing_videos)
                    # Mark as completed in checkpoint to avoid re-processing
                    if self.checkpoint:
                        self.checkpoint.completed_videos.append(f"{keyword}|{tier}")
                        self._save_checkpoint()
                    continue
                elif existing_videos:
                    logger.debug(f"  [{tier}] Found {len(existing_videos)} existing, need {per_kw - len(existing_videos)} more")

            # Checkpoint-based skip (secondary check)
            if self.checkpoint:
                video_key = f"{keyword}|{tier}"
                if video_key in self.checkpoint.completed_videos:
                    logger.debug(f"Skipping {keyword} ({tier}) - checkpoint says completed")
                    continue

            logger.debug(f"  [{tier}] Downloading...")
            downloaded = self._download_single(keyword, tier, output_dir, topic)

            # Check for timeout and retry with modified keywords
            retry_attempt = 0
            while not downloaded and getattr(self, '_last_download_timed_out', False) and retry_attempt < 2:
                alt_keyword = self._get_retry_keyword(keyword, retry_attempt)
                if alt_keyword and alt_keyword != keyword:
                    logger.debug(f"  [{tier}] Timeout - retrying with: '{alt_keyword}'")
                    downloaded = self._download_single(alt_keyword, tier, output_dir, topic)
                    retry_attempt += 1
                else:
                    break

            if downloaded:
                logger.debug(f"  [{tier}] ✓ {len(downloaded)} video(s)")
                all_downloaded.extend(downloaded)

                # Update tier download count
                with self._lock:
                    self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(downloaded)

                # Update sources
                with self._lock:
                    self.sources.extend(downloaded)
                    self._save_sources()
            else:
                # Check if failure was due to rate limiting - add to batch retry queue
                if getattr(self, '_last_download_rate_limited', False):
                    self.retry_queue.add(
                        video_id=f"{keyword}|{tier}",  # Use keyword|tier as ID
                        keyword=keyword,
                        tier=tier,
                        error_message="Rate limited after retries exhausted"
                    )
                    logger.debug(f"  [{tier}] Added to batch retry queue")
                else:
                    # ZERO-DOWNLOAD REMIX: Try LLM-based keyword remix when 0 results
                    remix_keyword = self._get_remix_keyword(keyword, topic)
                    if remix_keyword and remix_keyword != keyword:
                        logger.info(f"  [{tier}] No results - trying remix: '{remix_keyword}'")
                        downloaded = self._download_single(remix_keyword, tier, output_dir, topic)
                        if downloaded:
                            logger.debug(f"  [{tier}] ✓ Remix success: {len(downloaded)} video(s)")
                            all_downloaded.extend(downloaded)
                            with self._lock:
                                self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(downloaded)
                                self.sources.extend(downloaded)
                                self._save_sources()
                        else:
                            logger.debug(f"  [{tier}] Remix also returned no results")
                    else:
                        logger.debug(f"  [{tier}] No results")

            # Update checkpoint
            if self.checkpoint:
                logger.debug(f"  [{tier}] Saving checkpoint...")
                self.checkpoint.completed_videos.append(f"{keyword}|{tier}")
                self._save_checkpoint()
                logger.debug(f"  [{tier}] Checkpoint saved")

        return all_downloaded

    def _download_single(
        self,
        keyword: str,
        tier: str,
        output_dir: Path,
        topic: str = ""
    ) -> List[DownloadedVideo]:
        """
        Download videos for a single keyword and tier with optional LLM filtering.

        Implements two flows:
        1. LLM filter enabled: Search metadata → LLM filter → Speech screen → Download by IDs
        2. LLM filter disabled: Direct yt-dlp search with built-in filters

        Args:
            keyword: Search keyword
            tier: Duration tier
            output_dir: Output directory
            topic: Topic context for LLM

        Returns:
            List of DownloadedVideo objects
        """
        max_downloads = self._get_tier_value(tier, 'per_keyword', 5)

        # ADAPTIVE SEARCH POOL: Adjust based on historical pass rates
        search_pool = self._get_adaptive_search_pool(keyword, max_downloads)

        # Get path length settings from config
        max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)

        # Create keyword subdirectory
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
        safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
        tier_short = tier[0]
        keyword_dir = output_dir / f"{safe_keyword}_{tier_short}"
        keyword_dir.mkdir(parents=True, exist_ok=True)

        # Get existing files
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()

        # Check if LLM filtering is enabled
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        use_llm_filter = llm_config and getattr(llm_config, 'enabled', False)

        if use_llm_filter:
            # NEW FLOW: Search metadata first, filter with LLM, then download specific videos
            logger.debug(f"    Searching {search_pool} videos for LLM filtering...")

            # Check circuit breaker before searching (may pause if tripped)
            self.circuit_breaker.check_and_wait()

            # Search with timeout-based remix fallback (max 2 remix attempts)
            search_result = self._search_video_metadata(keyword, tier, search_pool)
            current_keyword = keyword
            remix_attempts = 0
            max_remix_attempts = 2

            # If search timed out, try LLM keyword remix before giving up
            while search_result.timed_out and remix_attempts < max_remix_attempts:
                logger.info(f"    Search timeout for '{current_keyword}' - trying keyword remix ({remix_attempts + 1}/{max_remix_attempts})")

                remix_keyword = self._get_remix_keyword(current_keyword, topic)
                if remix_keyword and remix_keyword != current_keyword:
                    logger.info(f"    Remixed keyword: '{current_keyword}' → '{remix_keyword}'")
                    current_keyword = remix_keyword
                    search_result = self._search_video_metadata(remix_keyword, tier, search_pool)
                    remix_attempts += 1
                else:
                    logger.debug(f"    No remix available for '{current_keyword}'")
                    break

            # Final check after all remix attempts
            if search_result.timed_out:
                log_error_with_context(
                    logger, "DL-002",
                    f"Search timeout for '{keyword}' after {remix_attempts} remix attempts",
                    keyword=keyword,
                    remix_attempts=remix_attempts
                )
                self.circuit_breaker.record_failure()
                # US-123-010: Apply self-regulation on errors
                self._apply_self_regulation(error_type="timeout")
                return []

            videos = search_result.videos
            if not videos:
                logger.debug(f"    No videos found for '{keyword}'")
                self.circuit_breaker.record_failure()
                # US-123-010: Apply self-regulation on errors
                self._apply_self_regulation(error_type="no_results")
                return []

            # Search returned results - record success to reset circuit breaker
            self.circuit_breaker.record_success()
            logger.debug(f"    Found {len(videos)} candidate videos")

            # Apply blacklist filter first (fast, no API cost)
            title_blacklist = getattr(self.download_config, 'title_blacklist', [])
            if title_blacklist:
                before_count = len(videos)
                videos = [v for v in videos if not any(
                    term.lower() in v['title'].lower() for term in title_blacklist
                )]
                if before_count > len(videos):
                    logger.debug(f"    Blacklist filter: {len(videos)}/{before_count} passed")

            # Apply LLM filter
            approved_videos = self._filter_titles_with_llm(videos[:search_pool], keyword, topic)

            # Track pass rate for adaptive pool sizing
            searched_count = min(len(videos), search_pool)
            approved_count = len(approved_videos) if approved_videos else 0
            self._record_search_pass_rate(keyword, searched_count, approved_count)

            if not approved_videos:
                logger.debug(f"    No videos passed LLM filter for '{keyword}'")
                return []

            # Speech screening: filter out videos with speech
            speech_config = getattr(self.download_config, 'speech_screening', None)
            speech_enabled = speech_config and getattr(speech_config, 'enabled', False)
            speech_tiers = getattr(speech_config, 'tiers', ['long', 'longer']) if speech_config else []

            if speech_enabled and tier in speech_tiers:
                logger.info(f"[SPEECH SCREEN] Tier '{tier}' - screening {len(approved_videos)} approved videos...")
                approved_videos = self._screen_approved_videos(approved_videos, keyword)
                logger.info(f"[SPEECH SCREEN] After screening: {len(approved_videos)} videos remain")

                if not approved_videos:
                    logger.debug(f"    No videos passed speech screening for '{keyword}'")
                    return []

            # Download only approved videos (by ID)
            video_ids = [v['id'] for v in approved_videos[:max_downloads]]
            logger.debug(f"    Downloading {len(video_ids)} approved videos...")

            return self._download_by_ids(video_ids, keyword_dir, output_dir, keyword, tier)

        else:
            # ORIGINAL FLOW: Direct search and download with yt-dlp filters

            # Check circuit breaker before searching (may pause if tripped)
            self.circuit_breaker.check_and_wait()

            cmd = [
                'yt-dlp',
                '--ignore-config',
                f'ytsearch{search_pool}:{keyword}',
                '-f', self._build_format_string(),
                '--match-filter', self._build_filter_string(tier),
                '--max-downloads', str(max_downloads),
                '--merge-output-format', 'mp4',
                '--no-playlist',
                '--write-info-json',
                '--restrict-filenames',
                '--no-overwrites',
                '--socket-timeout', '10',
                '--retries', '10',
                '--fragment-retries', '10',
                '--throttled-rate', '100K',
                '--force-ipv4',
                '--http-chunk-size', '10M',
                '--skip-unavailable-fragments',
                '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
                '--progress',
                '--newline',
            ]

            esc_result, imp_target = self._add_escalation_to_cmd(cmd, keyword)
            self._add_cookies_to_cmd(cmd)

            # US-93-007: Add bandwidth throttling if configured (for search+download)
            tier_size_estimates = {
                'short': 10.0,
                'medium': 30.0,
                'long': 90.0,
                'longer': 200.0,
            }
            estimated_size = tier_size_estimates.get(tier, 50.0)
            self._add_bandwidth_throttle_to_cmd(cmd, estimated_size)

            logger.debug(f"    Search: ytsearch{search_pool}, Max: {max_downloads}")

            slot_acquired = self.acquire_download_slot(timeout=30.0)
            if not slot_acquired:
                log_rate_limit(logger, "slot_acquisition", "download", "timeout", keyword=keyword, tier=tier)
                return []

            try:
                downloaded = self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)
            finally:
                self.release_download_slot()

            # Update circuit breaker based on search results
            if downloaded:
                self.circuit_breaker.record_success()
                if self.impersonation_manager and imp_target:
                    self.impersonation_manager.record_success(imp_target)
                # US-123-010: Check for recovery after successful download
                self._check_self_regulation_recovery()
            else:
                self.circuit_breaker.record_failure()
                if self.impersonation_manager and imp_target:
                    self.impersonation_manager.record_failure(imp_target)
                # US-123-010: Apply self-regulation on errors
                self._apply_self_regulation(error_type="download_failed")

            return downloaded

    def _download_by_ids(
        self,
        video_ids: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str
    ) -> List[DownloadedVideo]:
        """
        Download specific videos by their YouTube IDs.

        Automatically skips videos that already exist on disk (crash-resilient).

        Args:
            video_ids: List of YouTube video IDs
            keyword_dir: Keyword-specific directory
            output_dir: Base output directory
            keyword: Search keyword
            tier: Duration tier

        Returns:
            List of DownloadedVideo objects (including already-downloaded ones)
        """
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()

        # Check which video IDs are already downloaded (file-based, not checkpoint-based)
        already_downloaded = []
        missing_ids = []

        for vid_id in video_ids:
            # Check if any file contains this video ID
            found = False
            for existing_file in existing_before:
                if vid_id in existing_file and existing_file.endswith(('.mp4', '.mkv', '.webm')):
                    found = True
                    logger.debug(f"    Skipping {vid_id} - already exists: {existing_file}")
                    # Create DownloadedVideo for existing file
                    video_path = keyword_dir / existing_file
                    already_downloaded.append(DownloadedVideo(
                        file=str(video_path.relative_to(output_dir)),
                        url=f"https://www.youtube.com/watch?v={vid_id}",
                        title=existing_file,
                        channel="",
                        upload_date="",
                        duration=0,
                        duration_tier=tier,
                        keyword=keyword,
                        download_date="",
                        license="Unknown"
                    ))
                    break
            if not found:
                missing_ids.append(vid_id)

        if already_downloaded:
            logger.debug(f"    {len(already_downloaded)} already downloaded, {len(missing_ids)} to fetch")

        # US-129-011: Check global cache for missing videos
        global_cache_hits = []
        if self._global_cache and missing_ids:
            for vid_id in missing_ids[:]:  # Copy list to allow modification during iteration
                try:
                    entry = self._global_cache.find_by_youtube_id(vid_id)
                    if entry:
                        # Generate destination filename similar to yt-dlp format
                        dest_filename = f"{entry.filename[:max_fn_len]}_{vid_id}.mp4"
                        copied_path = self._global_cache.copy_to_project(
                            entry.video_hash,
                            keyword_dir,
                            dest_filename
                        )
                        if copied_path:
                            logger.info(f"    Copied from global cache: {vid_id} -> {copied_path}")
                            global_cache_hits.append(DownloadedVideo(
                                file=str(Path(copied_path).relative_to(output_dir)),
                                url=f"https://www.youtube.com/watch?v={vid_id}",
                                title=entry.filename,
                                channel="",
                                upload_date="",
                                duration=entry.duration,
                                duration_tier=tier,
                                keyword=keyword,
                                download_date=entry.last_used,
                                license="Unknown"
                            ))
                            missing_ids.remove(vid_id)
                except Exception as e:
                    logger.debug(f"    Global cache check failed for {vid_id}: {e}")

            if global_cache_hits:
                logger.info(f"    {len(global_cache_hits)} videos copied from global cache")
                already_downloaded.extend(global_cache_hits)

        if not missing_ids:
            # All videos already exist (locally or from global cache)
            return already_downloaded

        # Get filename length from config
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)

        # Download videos one at a time to avoid timeout on large batches
        # Each video gets its own timeout, and partial progress is preserved
        newly_downloaded = []
        total_to_download = len(missing_ids)
        for i, vid_id in enumerate(missing_ids, 1):
            logger.info(f"    Downloading video {i}/{total_to_download}: {vid_id}")

            # Acquire rate limit slot before download (US-35-002)
            # Uses global coordinator to ensure downloads don't exceed rate limit
            slot_acquired = self.acquire_download_slot(timeout=30.0)
            if not slot_acquired:
                log_rate_limit(logger, "slot_acquisition", "download", "timeout", video_id=vid_id)
                continue

            try:
                # Track existing files before this download
                existing_now = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()

                url = f"https://www.youtube.com/watch?v={vid_id}"
                cmd = [
                    'yt-dlp',
                    '--ignore-config',
                    '-f', self._build_format_string(),
                    '--merge-output-format', 'mp4',
                    '--no-playlist',
                    '--write-info-json',
                    '--restrict-filenames',
                    '--no-overwrites',
                    '--socket-timeout', '10',
                    '--retries', '10',
                    '--fragment-retries', '10',
                    '--throttled-rate', '100K',
                    '--force-ipv4',
                    '--http-chunk-size', '10M',
                    '--skip-unavailable-fragments',
                    '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
                    '--progress',
                    '--newline',
                    url
                ]

                esc_result, imp_target = self._add_escalation_to_cmd(cmd, keyword)
                self._add_cookies_to_cmd(cmd)

                # US-93-007: Add bandwidth throttling if configured
                # Estimate file size based on tier
                tier_size_estimates = {
                    'short': 10.0,
                    'medium': 30.0,
                    'long': 90.0,
                    'longer': 200.0,
                }
                estimated_size = tier_size_estimates.get(tier, 50.0)
                self._add_bandwidth_throttle_to_cmd(cmd, estimated_size)

                downloaded = self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_now)
                if downloaded:
                    newly_downloaded.extend(downloaded)
                    if self.impersonation_manager and imp_target:
                        self.impersonation_manager.record_success(imp_target)
                    logger.debug(f"    ✓ Downloaded {vid_id}")
                else:
                    if self.impersonation_manager and imp_target:
                        self.impersonation_manager.record_failure(imp_target)
                    logger.debug(f"    ✗ Failed to download {vid_id}")
            finally:
                # Always release slot after download attempt (US-35-002)
                self.release_download_slot()

        # Combine already downloaded + newly downloaded
        return already_downloaded + newly_downloaded

    # Error patterns for retry classification
    # Transient errors: worth retrying with exponential backoff
    TRANSIENT_ERROR_PATTERNS = [
        '429',                  # Rate limit
        'rate limit',
        'too many requests',
        'http error 403',       # Forbidden - cookie/auth issue
        '403: forbidden',
        'connection reset',
        'connection refused',
        'connection timed out',
        'temporary failure',
        'network unreachable',
        'service unavailable',
        'http error 503',
        'http error 502',
        'socket timeout',
        'read timed out',
        'getaddrinfo failed',       # DNS resolution failure
        'name or service not known', # DNS resolution failure (Linux)
        'nodename nor servname',     # DNS resolution failure (macOS)
    ]

    # Permanent errors: fail immediately, no retry
    PERMANENT_ERROR_PATTERNS = [
        'video unavailable',
        'private video',
        'this video is private',
        'age-restricted',
        'sign in to confirm your age',
        'video has been removed',
        'this video is no longer available',
        'copyright claim',
        'blocked in your country',
        'members-only',
        'join this channel',
        'premiere will begin',
        'is not available',
        'video is unavailable',
        'account has been terminated',
    ]

    # Auth errors: retrying with the same cookies is futile — advance method immediately
    AUTH_ERROR_PATTERNS = [
        'http error 403',
        '403: forbidden',
        'sign in to confirm',
        'login required',
        'confirm you\'re not a bot',
    ]

    def _is_transient_error(self, stderr: str) -> bool:
        """Check if error is transient (worth retrying)."""
        stderr_lower = stderr.lower()
        return any(pattern in stderr_lower for pattern in self.TRANSIENT_ERROR_PATTERNS)

    def _is_auth_error(self, stderr: str) -> bool:
        """Check if error is an auth/cookie error (advance method, don't retry)."""
        stderr_lower = stderr.lower()
        return any(pattern in stderr_lower for pattern in self.AUTH_ERROR_PATTERNS)

    def _is_permanent_error(self, stderr: str) -> bool:
        """Check if error is permanent (should not retry)."""
        stderr_lower = stderr.lower()
        return any(pattern in stderr_lower for pattern in self.PERMANENT_ERROR_PATTERNS)

    def _wait_for_process_with_progress(
        self,
        process: subprocess.Popen,
        stall_timeout: int,
        max_timeout: int,
        keyword: str,
        tier: str
    ) -> Tuple[str, str, bool]:
        """Wait for process with progress-aware timeout.

        Instead of a hard total timeout (subprocess.communicate), this monitors
        stderr output from yt-dlp. The process is only killed if:
        1. No stderr output for stall_timeout seconds (download stalled), OR
        2. Total elapsed time exceeds max_timeout (absolute cap)

        This allows slow-but-progressing downloads to continue instead of being
        killed by a fixed timeout.

        Args:
            process: Running subprocess
            stall_timeout: Seconds of no output before declaring stall
            max_timeout: Absolute maximum seconds to wait
            keyword: For logging
            tier: For logging

        Returns:
            tuple: (stdout, stderr, timeout_type) where timeout_type is
                None (no timeout), 'stall' (no output), or 'max_timeout' (too slow).
                Both string values are truthy for backward-compatible `if timed_out:` checks.
        """
        stderr_lines = []
        stdout_lines = []
        last_activity = time.time()
        lock = threading.Lock()
        stderr_done = threading.Event()
        stdout_done = threading.Event()

        def read_stderr():
            nonlocal last_activity
            try:
                while True:
                    line = process.stderr.readline()
                    if not isinstance(line, str) or not line:
                        break
                    with lock:
                        stderr_lines.append(line)
                        last_activity = time.time()
            except (OSError, ValueError, UnicodeDecodeError):
                pass  # Pipe closed, invalid, or encoding error
            finally:
                stderr_done.set()

        def read_stdout():
            """Drain stdout to prevent pipe deadlock."""
            try:
                while True:
                    line = process.stdout.readline()
                    if not isinstance(line, str) or not line:
                        break
                    with lock:
                        stdout_lines.append(line)
                        last_activity = time.time()
            except (OSError, ValueError, UnicodeDecodeError):
                pass  # Pipe closed, invalid, or encoding error
            finally:
                stdout_done.set()

        stderr_reader = threading.Thread(target=read_stderr, daemon=True)
        stdout_reader = threading.Thread(target=read_stdout, daemon=True)
        stderr_reader.start()
        stdout_reader.start()

        logger.debug(
            f"Download process started for '{keyword}' ({tier}) — "
            f"stall_timeout={stall_timeout}s, max_timeout={max_timeout}s, pid={process.pid}"
        )

        start_time = time.time()
        timeout_type = None  # None, 'stall', or 'max_timeout'

        while process.poll() is None:
            time.sleep(1)
            now = time.time()

            with lock:
                stall_duration = now - last_activity
                lines_so_far = len(stderr_lines) + len(stdout_lines)

            if stall_duration > stall_timeout:
                logger.info(
                    f"Download stalled for '{keyword}' ({tier}) — "
                    f"no output for {stall_timeout}s (received {lines_so_far} lines before stall), "
                    f"killing pid {process.pid}"
                )
                timeout_type = 'stall'
                break

            if now - start_time > max_timeout:
                logger.info(
                    f"Download hit max timeout for '{keyword}' ({tier}) — "
                    f"{int(now - start_time)}s total ({lines_so_far} lines received), "
                    f"killing pid {process.pid}"
                )
                timeout_type = 'max_timeout'
                break

        if timeout_type:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # Close pipes BEFORE joining reader threads — unblocks readline()
            # which may hang indefinitely on Windows after process.kill()
            try:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            except Exception:
                pass
            # Log what stderr/stdout we did receive for debugging
            with lock:
                if stderr_lines:
                    last_stderr = stderr_lines[-1].strip()
                    logger.debug(f"Last stderr before kill: {last_stderr[:200]}")
                if stdout_lines:
                    last_stdout = stdout_lines[-1].strip()
                    logger.debug(f"Last stdout before kill: {last_stdout[:200]}")
                if not stderr_lines and not stdout_lines:
                    logger.info(
                        f"yt-dlp produced ZERO output for '{keyword}' ({tier}) — "
                        f"process may be hanging during connection/metadata extraction"
                    )
        else:
            # Process finished normally — wait for it to fully clean up
            process.wait()
            elapsed = time.time() - start_time
            with lock:
                lines_total = len(stderr_lines) + len(stdout_lines)
            logger.debug(
                f"Download process finished for '{keyword}' ({tier}) in {elapsed:.1f}s "
                f"({lines_total} output lines, exit code {process.returncode})"
            )

        # Wait for reader threads to finish
        stderr_done.wait(timeout=5)
        stdout_done.wait(timeout=5)
        stderr_reader.join(timeout=2)
        stdout_reader.join(timeout=2)

        # Close pipes to avoid ResourceWarning (no-op if already closed above)
        try:
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
            if process.stderr and not process.stderr.closed:
                process.stderr.close()
        except Exception:
            pass

        stderr = ''.join(stderr_lines)
        stdout = ''.join(stdout_lines)

        return stdout, stderr, timeout_type

    def _run_download_cmd(
        self,
        cmd: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set,
        timeout_override: int = None,
        _is_method_retry: bool = False
    ) -> List[DownloadedVideo]:
        """
        Execute download command and process results with exponential backoff retry.

        This is the most complex method in the core orchestrator (199 lines in original).
        Handles:
        - Subprocess execution with timeout
        - Exponential backoff retry for transient errors
        - Cookie method fallback on auth error exhaustion
        - Immediate failure for permanent errors
        - Cleanup of partial downloads
        - Transcoding if DaVinci mode enabled
        - Metadata extraction from info.json
        - Filename sanitization for NLE compatibility
        - Source tracking for diversity analysis

        Args:
            cmd: yt-dlp command to execute
            keyword_dir: Keyword-specific directory
            output_dir: Base output directory
            keyword: Search keyword
            tier: Duration tier
            existing_before: Set of files that existed before download
            timeout_override: Optional timeout override
            _is_method_retry: Internal flag — True when retrying with a new cookie method

        Returns:
            List of DownloadedVideo objects, or empty list on failure/timeout.
            Sets self._last_download_timed_out = True if timeout occurred.

        Notes:
            Caller owns global rate-limit slot acquisition/release for the yt-dlp request.
        """
        # Clean up any partial downloads from previous crashes
        if keyword_dir.exists():
            for part_file in keyword_dir.glob('*.part'):
                try:
                    part_file.unlink()
                    logger.debug(f"Cleaned up partial download: {part_file.name}")
                except Exception:
                    pass
            for ytdl_file in keyword_dir.glob('*.ytdl'):
                try:
                    ytdl_file.unlink()
                except Exception:
                    pass

        # Use override, tier-specific timeout, config default, or fallback
        if timeout_override:
            base_timeout = timeout_override
        else:
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            if tier and tier in tier_timeouts:
                base_timeout = tier_timeouts[tier]
            else:
                base_timeout = getattr(self.download_config, 'download_timeout', 120)

        # US-93-005: Apply adaptive timeout based on video file size estimation
        # Calculate: adaptive_timeout_base + (estimated_size_mb * size_multiplier)
        adaptive_enabled = getattr(self.download_config, 'adaptive_timeout_enabled', True)
        if adaptive_enabled:
            adaptive_base = getattr(self.download_config, 'adaptive_timeout_base', 30)
            adaptive_multiplier = getattr(self.download_config, 'adaptive_timeout_multiplier', 0.5)
            adaptive_max = getattr(self.download_config, 'adaptive_timeout_max', 600)

            # Estimate file size based on tier (tier correlates with duration)
            # Tier durations: short=<2min, medium=2-10min, long=10-25min, longer=25+min
            tier_size_estimates = {
                'short': 10.0,    # ~10 MB for <2 min video
                'medium': 30.0,   # ~30 MB for 2-10 min video
                'long': 90.0,     # ~90 MB for 10-25 min video
                'longer': 200.0,  # ~200 MB for 25+ min video
            }
            estimated_size_mb = tier_size_estimates.get(tier, 50.0)

            # Get quality from config for more accurate estimation
            quality = getattr(self.download_config, 'quality', '1080p')
            # Extract resolution number (1080p -> 1080)
            quality_res = ''.join(c for c in quality if c.isdigit()) or '1080'

            # Get size estimates from config if available
            size_estimates = getattr(self.download_config, 'adaptive_timeout_size_estimates', None)
            if size_estimates:
                estimated_size_mb = utils.estimate_video_size_mb(
                    duration_seconds=int(estimated_size_mb * 2),  # Convert MB estimate back to duration proxy
                    quality=quality_res,
                    size_estimates=size_estimates
                )

            # Calculate adaptive timeout
            size_based_timeout = utils.calculate_adaptive_timeout(
                estimated_size_mb=estimated_size_mb,
                base_timeout=adaptive_base,
                multiplier=adaptive_multiplier,
                max_timeout=adaptive_max
            )

            # Log timeout values for debugging (always log adaptive timeout)
            logger.debug(
                f"Adaptive timeout: tier={tier}, estimated_size={estimated_size_mb:.0f}MB, "
                f"tier_timeout={base_timeout}s, size_based={size_based_timeout}s, "
                f"adaptive_multiplier={adaptive_multiplier}, max={adaptive_max}s"
            )

            # Use the larger of tier-based or size-based timeout
            if size_based_timeout > base_timeout:
                base_timeout = size_based_timeout

        # Apply adaptive timeout based on network speed
        download_timeout = self.speed_tracker.get_adjusted_timeout(base_timeout)
        if download_timeout > base_timeout:
            self.rate_limit_metrics.record_timeout_extension()

        # Track download timing for speed measurement
        download_start_time = time.time()

        # Get retry settings from config
        max_retries = getattr(self.download_config, 'max_retries', 3)
        retry_delay = getattr(self.download_config, 'retry_delay', 2.0)
        retry_backoff = getattr(self.download_config, 'retry_backoff', 2.0)

        # Track timeout and rate limit for retry logic
        self._last_download_timed_out = False
        self._last_download_rate_limited = False
        process = None
        last_stderr = ""

        # Check if circuit breaker should block download retries (US-011)
        block_download_retries = getattr(
            self.circuit_breaker.config, 'block_download_retries', True
        )

        # Record download attempt for metrics
        self.rate_limit_metrics.record_download_attempt()

        # US-003: Check budget before starting retry loop
        if self._share_budget_across_keywords:
            recommended = self.rate_limit_budget.get_recommended_escalation()
            if recommended == "exhausted":
                log_error(
                    logger, "Rate limit budget",
                    f"exhausted for '{keyword}' ({tier}) — "
                    f"skipping keyword entirely (backoff: {self.rate_limit_budget.backoff_time_spent:.1f}s / "
                    f"{self.rate_limit_budget.max_backoff_time:.0f}s, "
                    f"rotations: {self.rate_limit_budget.rotations_used})",
                    error_code="E201"
                )
                self.rate_limit_budget.record_failure(keyword=keyword)
                self.rate_limit_metrics.record_download_failure()
                # US-93-011: Record to metrics exporter
                if self.metrics_exporter:
                    self.metrics_exporter.record_download_error()
                # Add to batch retry queue so it can be retried later with fresh budget
                self.retry_queue.add(
                    video_id=f"{keyword}|{tier}",
                    keyword=keyword,
                    tier=tier,
                    error_message="Rate limit budget exhausted"
                )
                return []

        # Reset cookie method fallback to last working method (skip on recursive retry)
        if not _is_method_retry:
            self.method_fallback.reset_for_next_download()

        # Retry loop with exponential backoff
        return self._run_download_retry_loop(
            cmd=cmd,
            keyword_dir=keyword_dir,
            output_dir=output_dir,
            keyword=keyword,
            tier=tier,
            existing_before=existing_before,
            timeout_override=timeout_override,
            _is_method_retry=_is_method_retry,
            download_timeout=download_timeout,
            download_start_time=download_start_time,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff=retry_backoff,
            block_download_retries=block_download_retries
        )

    def _run_download_retry_loop(
        self,
        cmd: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set,
        timeout_override: int,
        _is_method_retry: bool,
        download_timeout: int,
        download_start_time: float,
        max_retries: int,
        retry_delay: float,
        retry_backoff: float,
        block_download_retries: bool
    ) -> List[DownloadedVideo]:
        """Execute the download retry loop (extracted for slot management).

        Args:
            All parameters from _run_download_cmd plus computed values

        Returns:
            List of DownloadedVideo objects
        """
        process = None
        last_stderr = ""
        # US-162-004: Track retry timing for duration logging
        retry_start_time = time.time()
        # Track if we've had to retry (for recovery logging)
        had_retry = False

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            # US-011: Check circuit breaker state before retry attempts (not first attempt)
            if attempt > 0 and block_download_retries and self.circuit_breaker.is_open:
                wait_time = self.circuit_breaker.wait_for_recovery_if_needed(
                    context=f"download retry {attempt}/{max_retries} for '{keyword}' ({tier})"
                )
                if wait_time > 0:
                    self.rate_limit_metrics.record_circuit_breaker_wait(wait_time)
            try:
                # Log the actual command for debugging stalls
                safe_cmd = ' '.join(cmd[:6])  # First 6 args (yt-dlp, url/search, -f, format)
                logger.debug(f"yt-dlp command for '{keyword}' ({tier}): {safe_cmd} ... ({len(cmd)} args total)")

                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    **SUBPROCESS_FLAGS
                )

                # Verify encoding is applied (debug charmap issue)
                logger.debug(f"Popen stderr encoding: {process.stderr.encoding}, errors: {process.stderr.errors}")

                # Use progress-aware timeout: only kill if download stalls
                # (no stderr/stdout output for stall_timeout seconds), not on total elapsed time.
                # stall_timeout=0 means use the tier timeout as the stall timeout.
                # max_timeout is 1.5x tier timeout as an absolute cap.
                # With resume-on-retry, we kill stuck downloads sooner and
                # resume from the last completed fragment on the next attempt.
                configured_stall = getattr(self.download_config, 'stall_timeout', 0)
                stall_timeout = configured_stall if configured_stall > 0 else download_timeout
                max_timeout = int(download_timeout * 1.5)
                stdout, stderr, timeout_type = self._wait_for_process_with_progress(
                    process, stall_timeout, max_timeout, keyword, tier
                )
                last_stderr = stderr

                if timeout_type:
                    # max_timeout means download was progressing but too slow —
                    # retrying won't help much since speed won't improve.
                    # Cap at 1 retry (resume may finish a nearly-done download).
                    # Stall timeout means download stopped producing output —
                    # resume-on-retry can recover, so use full max_retries.
                    effective_max_retries = 1 if timeout_type == 'max_timeout' else max_retries
                    timeout_label = 'max timeout' if timeout_type == 'max_timeout' else 'stall'

                    if attempt < effective_max_retries:
                        delay = retry_delay * (retry_backoff ** attempt)
                        # US-003: Check budget before retry backoff
                        if self._share_budget_across_keywords and not self.rate_limit_budget.can_backoff(delay):
                            logger.warning(
                                f"Backoff budget exhausted for '{keyword}' ({tier}) timeout retry — "
                                f"adding to batch retry queue"
                            )
                            self._last_download_rate_limited = True
                            self.rate_limit_budget.record_failure(keyword=keyword)
                            self.retry_queue.add(
                                video_id=f"{keyword}|{tier}",
                                keyword=keyword,
                                tier=tier,
                                error_message="Timeout - backoff budget exhausted"
                            )
                            self.rate_limit_metrics.record_download_failure()
                            return []
                        logger.info(
                            f"Timeout downloading '{keyword}' ({tier}) — "
                            f"{timeout_label} after {stall_timeout if timeout_type == 'stall' else max_timeout}s, "
                            f"retry {attempt + 1}/{effective_max_retries} in {delay:.1f}s"
                        )
                        # Structured logging for retry attempt (US-159-005)
                        _log_retry_attempt(
                            keyword=keyword,
                            tier=tier,
                            attempt=attempt,
                            max_retries=effective_max_retries,
                            delay=delay,
                            reason=f"timeout_{timeout_type}",
                            video_id=f"{keyword}|{tier}"
                        )
                        # US-162-004: Track that a retry was needed
                        had_retry = True
                        self.rate_limit_metrics.record_retry('timeout')
                        # US-93-011: Record retry to metrics exporter
                        if self.metrics_exporter:
                            self.metrics_exporter.record_retry()
                        self._mock_sleep(delay)
                        # US-003: Record backoff time in budget
                        if self._share_budget_across_keywords:
                            self.rate_limit_budget.record_backoff(delay, keyword=keyword)
                        continue
                    else:
                        logger.warning(
                            f"Timeout downloading '{keyword}' ({tier}) — "
                            f"{timeout_label}, all {effective_max_retries} retries exhausted "
                            f"(tier timeout: {download_timeout}s)"
                        )
                        self._last_download_timed_out = True
                        self.rate_limit_metrics.record_retries_exhausted()
                        self.rate_limit_metrics.record_download_failure()
                        return []

                # Check for errors in stderr
                if stderr and process.returncode != 0:
                    # Record escalation trigger (403/bot-detection) before error classification
                    if self.escalation_manager and is_escalation_trigger(stderr):
                        self.escalation_manager.record_failure(keyword, stderr)
                        # Check escalation state - may trigger Tier 3 (cookies) or Tier 4 (VPN)
                        esc_result = self.escalation_manager.get_escalation_args(keyword)

                        # Tier 4: VPN rotation (takes precedence - changes IP which resets other limits)
                        # US-113-012: Check if VPN is available (not degraded) before attempting
                        if esc_result.rotate_vpn and self.mullvad_vpn:
                            # Check if VPN can be used (not degraded)
                            if not self.mullvad_vpn.can_use_vpn():
                                degr_status = self.mullvad_vpn.get_degradation_status()
                                logger.warning(
                                    f"VPN unavailable for '{keyword}' ({tier}) — "
                                    f"degraded={degr_status['is_degraded']}, "
                                    f"consecutive_failures={degr_status['consecutive_failures']}/"
                                    f"{degr_status['max_consecutive_vpn_failures']}. "
                                    f"Falling back to cookie-only mode."
                                )
                                # Fall through to cookie rotation below
                            else:
                                logger.info(f"Tier 4 escalation for '{keyword}' ({tier}) — rotating VPN server")
                                # Structured logging for escalation event (US-159-005)
                                _log_escalation_event(
                                    keyword=keyword,
                                    tier=tier,
                                    event="vpn_activation",
                                    before_state="tier3_cookie_rotation",
                                    after_state="tier4_vpn_rotation",
                                    details={"reason": "escalation_trigger"}
                                )
                                if self.mullvad_vpn.rotate_server():
                                    # Record VPN success - resets failure counter
                                    self.mullvad_vpn.record_vpn_success()
                                    # Record VPN rotation in budget
                                    if self._share_budget_across_keywords:
                                        self.rate_limit_budget.record_vpn_rotation()
                                        # Reset cookie/backoff budgets after IP change
                                        self.rate_limit_budget.reset_on_ip_change()
                                    # Reset circuit breaker after successful VPN rotation
                                    if self.circuit_breaker:
                                        self.circuit_breaker.reset()
                                    logger.info(f"VPN rotation successful — circuit breaker reset, budgets refreshed")
                                    # Structured logging for escalation event (US-159-005)
                                    _log_escalation_event(
                                        keyword=keyword,
                                        tier=tier,
                                        event="vpn_rotation_success",
                                        before_state="tier4_vpn_rotation",
                                        after_state="vpn_active",
                                        details={"consecutive_failures": 0}
                                    )
                                else:
                                    # Record VPN failure for degradation tracking (US-113-012)
                                    self.mullvad_vpn.record_vpn_failure()
                                    degr_status = self.mullvad_vpn.get_degradation_status()
                                    logger.warning(
                                        f"VPN rotation failed for '{keyword}' ({tier}) — "
                                        f"consecutive failures: {degr_status['consecutive_failures']}/"
                                        f"{degr_status['max_consecutive_vpn_failures']}"
                                    )
                                    if degr_status['is_degraded']:
                                        logger.warning(
                                            f"VPN degraded to cookie-only mode after {degr_status['consecutive_failures']} "
                                            f"consecutive failures. Use enable_vpn(override_degradation=True) to override."
                                        )

                        # Tier 3: Cookie rotation (if not doing VPN rotation)
                        elif esc_result.rotate_cookies and self.cookie_rotator and self.cookie_rotator.is_enabled:
                            # DEBUG-level logging for cookie rotation
                            logger.debug(f"Cookie rotation triggered for '{keyword}' ({tier})")
                            new_cookie = self.cookie_rotator.rotate()
                            if new_cookie:
                                # Structured logging for escalation event (US-159-005)
                                _log_escalation_event(
                                    keyword=keyword,
                                    tier=tier,
                                    event="cookie_rotation",
                                    before_state="tier2_cookie_exhausted",
                                    after_state="tier3_new_cookie",
                                    details={"new_cookie": Path(new_cookie).name if new_cookie else "unknown"}
                                )
                                for i, arg in enumerate(cmd):
                                    if arg == '--cookies' and i + 1 < len(cmd):
                                        cmd[i + 1] = new_cookie
                                        break

                    # Check for permanent errors - fail immediately
                    if self._is_permanent_error(stderr):
                        logger.debug(f"Permanent error for '{keyword}' ({tier}): {stderr[:200]}")
                        # Structured logging for download failure (US-159-005)
                        _log_download_failure(
                            video_id=f"{keyword}|{tier}",
                            url="",
                            tier=tier,
                            error_type="permanent",
                            error_message=stderr[:200],
                            keyword=keyword
                        )
                        self.rate_limit_metrics.record_download_failure()
                        return []

                    # Auth errors (403/sign-in): advance cookie method immediately, no retries
                    if self._is_auth_error(stderr):
                        logger.info(f"Auth error for '{keyword}' ({tier}) — advancing cookie method (retries won't help)")
                        # WARNING-level logging when falling back between download methods
                        current_method = self.method_fallback.current_method.label
                        if self.method_fallback.advance():
                            new_method = self.method_fallback.current_method.label
                            logger.warning(
                                f"Cookie method fallback for '{keyword}' ({tier}): "
                                f"{current_method} -> {new_method}"
                            )
                            cmd = self._replace_cookie_args_in_cmd(cmd)
                            return self._run_download_cmd(
                                cmd, keyword_dir, output_dir, keyword, tier,
                                existing_before, timeout_override,
                                _is_method_retry=True
                            )
                        # All cookie methods exhausted
                        log_error_with_context(logger, "DL-004", f"Auth error - all cookie methods exhausted", keyword=keyword, tier=tier)
                        # Structured logging for download failure (US-159-005)
                        _log_download_failure(
                            video_id=f"{keyword}|{tier}",
                            url="",
                            tier=tier,
                            error_type="auth_cookie_exhausted",
                            error_message="All cookie methods exhausted after auth error",
                            keyword=keyword
                        )
                        self._last_download_rate_limited = True
                        self.rate_limit_metrics.record_retries_exhausted()
                        self.rate_limit_metrics.record_download_failure()
                        return []

                    # Other transient errors (429, network): retry with backoff
                    if self._is_transient_error(stderr):
                        self.handle_rate_limit_error(stderr, tier=tier, keyword=keyword)
                        # Update cmd with new cookie if rotated
                        if self.cookie_rotator and self.cookie_rotator.is_enabled:
                            current_cookie = self.cookie_rotator.get_current_cookie()
                            if current_cookie:
                                for i, arg in enumerate(cmd):
                                    if arg == '--cookies' and i + 1 < len(cmd):
                                        cmd[i + 1] = current_cookie
                                        break

                        if attempt < max_retries:
                            delay = retry_delay * (retry_backoff ** attempt)
                            # US-003: Check budget before retry backoff
                            if self._share_budget_across_keywords and not self.rate_limit_budget.can_backoff(delay):
                                logger.warning(
                                    f"Backoff budget exhausted for '{keyword}' ({tier}) transient error retry — "
                                    f"adding to batch retry queue"
                                )
                                self._last_download_rate_limited = True
                                self.rate_limit_budget.record_failure(keyword=keyword)
                                self.retry_queue.add(
                                    video_id=f"{keyword}|{tier}",
                                    keyword=keyword,
                                    tier=tier,
                                    error_message="Transient error - backoff budget exhausted"
                                )
                                self.rate_limit_metrics.record_download_failure()
                                return []
                            logger.info(f"Transient error for '{keyword}' ({tier}) - retry {attempt + 1}/{max_retries} in {delay:.1f}s")
                            # Structured logging for retry attempt (US-159-005)
                            _log_retry_attempt(
                                keyword=keyword,
                                tier=tier,
                                attempt=attempt,
                                max_retries=max_retries,
                                delay=delay,
                                reason="transient_error",
                                video_id=f"{keyword}|{tier}"
                            )
                            # US-162-004: Track that a retry was needed
                            had_retry = True
                            logger.debug(f"  Error: {stderr[:200]}")
                            self.rate_limit_metrics.record_retry('transient')
                            self._mock_sleep(delay)
                            # US-003: Record backoff time in budget
                            if self._share_budget_across_keywords:
                                self.rate_limit_budget.record_backoff(delay, keyword=keyword)
                            continue
                        else:
                            # Exhausted retries — try next cookie method
                            # WARNING-level logging when falling back between download methods
                            current_method = self.method_fallback.current_method.label
                            if self.method_fallback.advance():
                                new_method = self.method_fallback.current_method.label
                                logger.warning(
                                    f"Cookie method fallback for '{keyword}' ({tier}): "
                                    f"{current_method} -> {new_method}"
                                )
                                cmd = self._replace_cookie_args_in_cmd(cmd)
                                return self._run_download_cmd(
                                    cmd, keyword_dir, output_dir, keyword, tier,
                                    existing_before, timeout_override,
                                    _is_method_retry=True
                                )
                            # All cookie methods exhausted - mark for batch retry
                            log_rate_limit(
                                logger, "cookie_rotation", "download", "exhausted",
                                keyword=keyword, tier=tier, retries=max_retries
                            )
                            # Structured logging for download failure (US-159-005)
                            _log_download_failure(
                                video_id=f"{keyword}|{tier}",
                                url="",
                                tier=tier,
                                error_type="rate_limit_cookie_exhausted",
                                error_message=f"Rate limit after {max_retries} retries, all cookie methods exhausted",
                                keyword=keyword
                            )
                            self._last_download_rate_limited = True
                            self.rate_limit_metrics.record_retries_exhausted()
                            self.rate_limit_metrics.record_download_failure()
                            return []

                # Only log actual errors (not retried)
                if stderr:
                    for line in stderr.strip().split('\n'):
                        if line and 'WARNING' not in line and 'ERROR' in line:
                            logger.warning(f"    yt-dlp: {line}")

                # Success or non-retryable error - break out of retry loop
                break

            except Exception as e:
                # Ensure process is cleaned up on unexpected exception
                if process is not None and process.poll() is None:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                # Log full traceback for encoding errors to find the source
                if isinstance(e, UnicodeDecodeError):
                    import traceback
                    log_error(logger, "Unicode decode", f"for '{keyword}': {traceback.format_exc()}", error_code="E902", error=e)
                # Handle unexpected exceptions with retry
                if attempt < max_retries:
                    delay = retry_delay * (retry_backoff ** attempt)
                    # US-003: Check budget before retry backoff
                    if self._share_budget_across_keywords and not self.rate_limit_budget.can_backoff(delay):
                        logger.warning(
                            f"Backoff budget exhausted for '{keyword}' ({tier}) exception retry — "
                            f"adding to batch retry queue"
                        )
                        self._last_download_rate_limited = True
                        self.rate_limit_budget.record_failure(keyword=keyword)
                        self.retry_queue.add(
                            video_id=f"{keyword}|{tier}",
                            keyword=keyword,
                            tier=tier,
                            error_message=f"Exception - backoff budget exhausted: {e}"
                        )
                        self.rate_limit_metrics.record_download_failure()
                        return []
                    # US-93-006: Use structured error for better logging
                    error_str = str(e)
                    classified = classify_error_category(error_str)
                    structured = StructuredDownloadError(classified, keyword=keyword)
                    logger.info(
                        f"Error downloading '{keyword}' ({tier}): {structured} - "
                        f"retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                    )
                    # Structured logging for retry attempt (US-159-005)
                    _log_retry_attempt(
                        keyword=keyword,
                        tier=tier,
                        attempt=attempt,
                        max_retries=max_retries,
                        delay=delay,
                        reason=f"exception_{classified}",
                        video_id=f"{keyword}|{tier}"
                    )
                    # US-162-004: Track that a retry was needed
                    had_retry = True
                    self.rate_limit_metrics.record_retry('network')
                    self._mock_sleep(delay)
                    # US-003: Record backoff time in budget
                    if self._share_budget_across_keywords:
                        self.rate_limit_budget.record_backoff(delay, keyword=keyword)
                    continue
                else:
                    # US-93-006: Use structured error with suggestions for better user feedback
                    error_str = str(e)
                    classified = classify_error_category(error_str)
                    structured = StructuredDownloadError(classified, keyword=keyword)
                    error_code = structured.error_code.value
                    log_error(
                        logger, "Download",
                        f"failed for '{keyword}' ({tier}): {structured} - "
                        f"all {max_retries} retries exhausted. "
                        f"Suggestions: {', '.join(structured.suggestions[:2])}",
                        error_code=error_code
                    )
                    # Structured logging for download failure (US-159-005)
                    _log_download_failure(
                        video_id=f"{keyword}|{tier}",
                        url="",
                        tier=tier,
                        error_type=f"exception_{classified}",
                        error_message=error_str[:500],
                        keyword=keyword
                    )
                    self.rate_limit_metrics.record_retries_exhausted()
                    self.rate_limit_metrics.record_download_failure()
                    return []

        # After retry loop - process downloaded files
        # Find new files
        existing_after = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
        new_files = existing_after - existing_before

        # Filter to video files
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
        new_videos = [f for f in new_files if Path(f).suffix.lower() in video_extensions]

        if new_videos:
            # US-162-004: Log successful recovery after retry with duration
            if had_retry:
                retry_duration = time.time() - retry_start_time
                logger.info(
                    f"Download recovered after retries for '{keyword}' ({tier}) "
                    f"({retry_duration:.1f}s total)"
                )
            logger.debug(f"    Downloaded {len(new_videos)} video(s)")
            # Reset rate limit backoff on successful download
            self._reset_rate_limit_backoff()
            # Remember working cookie method for next download
            self.method_fallback.mark_success()
            # Record successful download for metrics
            self.rate_limit_metrics.record_download_success()
            # US-93-011: Record to metrics exporter
            if self.metrics_exporter:
                self.metrics_exporter.record_download_complete(
                    duration=download_duration,
                    size_bytes=total_bytes,
                    success=True,
                )
            # Record success for escalation manager (resets 403 counter, keeps tier)
            if self.escalation_manager:
                self.escalation_manager.record_success(keyword)
            # US-003: Record success in rate limit budget for success rate tracking
            if self._share_budget_across_keywords:
                self.rate_limit_budget.record_success(keyword=keyword)

            # Record download speed for adaptive timeout tracking
            download_duration = time.time() - download_start_time
            total_bytes = sum(
                (keyword_dir / f).stat().st_size
                for f in new_videos
                if (keyword_dir / f).exists()
            )
            if total_bytes > 0 and download_duration > 0:
                # Record per-video average for more consistent speed tracking
                for video_file in new_videos:
                    video_path = keyword_dir / video_file
                    if video_path.exists():
                        file_size = video_path.stat().st_size
                        # Proportional time based on file size
                        video_duration = download_duration * (file_size / total_bytes) if total_bytes > 0 else download_duration / len(new_videos)
                        # Extract video ID from filename (format: title_VIDEOID.ext)
                        video_id = Path(video_file).stem.split('_')[-1] if '_' in video_file else video_file
                        self.speed_tracker.record_download(
                            video_id=video_id,
                            bytes_downloaded=file_size,
                            duration_seconds=video_duration,
                            tier=tier
                        )
                        # Check for slow download warning (US-93-012)
                        if self.speed_tracker.config.enabled:
                            warning = self.speed_tracker.check_slow_download_warning(video_id)
                            if warning.is_slow:
                                logger.info(warning.message)
                        # Record speed sample for metrics
                        speed_mbps = (file_size / 1024 / 1024) / video_duration if video_duration > 0 else 0
                        self.rate_limit_metrics.record_speed_sample(speed_mbps)

            # Check for rate limit signals after recording speeds (US-002)
            # Proactive detection: slow speeds often precede hard rate limit errors
            if self.speed_tracker.config.enabled:
                signal = self.speed_tracker.detect_rate_limit_signals()
                if signal.detected:
                    # Trigger preemptive backoff via circuit breaker
                    if self.circuit_breaker.is_enabled:
                        logger.info(
                            f"Proactive rate limit response: {signal.consecutive_slow_count} "
                            f"consecutive slow downloads detected, triggering circuit breaker"
                        )
                        self.circuit_breaker.record_failure()
                        # US-123-010: Apply self-regulation on rate limit signals
                        self._apply_self_regulation(error_type="rate_limit_signal")

                    # Signal escalation manager for preemptive tier escalation (US-006)
                    if hasattr(self, 'escalation_manager') and self.escalation_manager is not None:
                        avg_speed = self.speed_tracker.get_average_speed_mbps()
                        self.escalation_manager.record_slow_speed(keyword, speed_mbps=avg_speed)

            # Variance detection for network issue identification (US-93-012)
            # Detect flaky vs slow-but-consistent networks
            if self.speed_tracker.config.enabled and self.speed_tracker.config.enable_variance_detection:
                variance_signal = self.speed_tracker.detect_variance_signals()
                if variance_signal.is_flaky:
                    logger.warning(
                        f"Network flakiness detected for keyword '{keyword}': "
                        f"CV={variance_signal.coefficient_of_variation:.2f}, "
                        f"category={variance_signal.variance_category}"
                    )
                    # Signal escalation manager for potential tier escalation on flaky network
                    if hasattr(self, 'escalation_manager') and self.escalation_manager is not None:
                        self.escalation_manager.record_slow_speed(keyword, speed_mbps=variance_signal.mean_speed_mbps)

        # US-162-004: Log final failure after all retries exhausted with total duration
        if not new_videos and had_retry:
            retry_duration = time.time() - retry_start_time
            logger.warning(
                f"Download failed after all retries for '{keyword}' ({tier}) "
                f"({retry_duration:.1f}s total)"
            )

        downloaded = []

        for idx, video_file in enumerate(new_videos, 1):
            video_path = keyword_dir / video_file

            # Try to get metadata from info.json
            info_file = video_path.with_suffix('.info.json')
            metadata = {}
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except (OSError, IOError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    # JSON metadata is optional - log and continue without it
                    logger.debug(f"Could not load metadata from {info_file}: {e}")

            # Transcode for DaVinci if enabled AND necessary
            final_path = video_path
            if self.download_config.davinci_mode:
                needs_transcode, reason = self._needs_transcoding(str(video_path))

                if not needs_transcode:
                    logger.debug(f"    No transcode needed: {reason}")
                    final_path = video_path
                else:
                    logger.debug(f"    Transcoding {video_file[:40]}...")
                    transcode_cmd, output_path = self._get_ffmpeg_transcode_cmd(
                        str(video_path), str(video_path)
                    )
                    temp_output = Path(output_path).with_stem(Path(output_path).stem + '_davinci')
                    transcode_cmd[-1] = str(temp_output)

                    transcode_process = None
                    try:
                        transcode_process = subprocess.Popen(
                            transcode_cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8',
                            errors='replace',
                            **SUBPROCESS_FLAGS
                        )

                        try:
                            _, stderr = transcode_process.communicate(timeout=download_timeout)
                            if transcode_process.returncode != 0:
                                log_error_with_context(logger, "DL-001", f"FFmpeg transcode failed", video_file=str(video_file), error=stderr[-500:] if stderr else "unknown")
                        except subprocess.TimeoutExpired:
                            transcode_process.kill()
                            transcode_process.communicate()
                            log_error_with_context(logger, "DL-002", f"Transcode timeout", video_file=str(video_file))
                        finally:
                            if transcode_process is not None and transcode_process.poll() is None:
                                transcode_process.kill()
                                try:
                                    transcode_process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    pass

                        if temp_output.exists() and temp_output.stat().st_size > 0:
                            if self.download_config.delete_original:
                                video_path.unlink()
                            final_path = temp_output.rename(temp_output.with_stem(
                                temp_output.stem.replace('_davinci', '')
                            ))
                            logger.info(f"    ↳ ✓ Transcode complete")
                        else:
                            logger.warning(f"    ↳ Transcode produced no output, using original")
                            final_path = video_path
                    except Exception as e:
                        log_error_with_context(logger, "DL-001", f"Transcode failed", video_file=str(video_file), error=str(e))
                        final_path = video_path

            # Sanitize filename for NLE compatibility
            final_path = utils.sanitize_filename_for_nle(final_path)

            # US-93-002: Verify file integrity via hash after download
            expected_hash = metadata.get('hash')  # Could be provided by yt-dlp or external source
            hash_passed, actual_hash = utils.verify_download_hash(final_path, expected_hash)

            # Create source record
            source = DownloadedVideo(
                file=str(final_path.relative_to(output_dir)),
                url=metadata.get('webpage_url', metadata.get('url', 'Unknown')),
                title=metadata.get('title', video_file),
                channel=metadata.get('uploader', metadata.get('channel', 'Unknown')),
                upload_date=metadata.get('upload_date', 'Unknown'),
                duration=metadata.get('duration', 0),
                duration_tier=tier,
                keyword=keyword,
                download_date=datetime.now().strftime('%Y-%m-%d'),
                license=metadata.get('license', 'Unknown'),
                video_hash=actual_hash or ""  # US-93-002: Store computed hash
            )

            downloaded.append(source)

            # US-129-011: Register downloaded video in global cache for cross-project reuse
            if self._global_cache and final_path.exists():
                try:
                    # Extract YouTube ID from URL
                    youtube_url = metadata.get('webpage_url', '')
                    youtube_id = ''
                    if 'youtube.com/watch?v=' in youtube_url:
                        youtube_id = youtube_url.split('watch?v=')[1].split('&')[0]
                    elif 'youtu.be/' in youtube_url:
                        youtube_id = youtube_url.split('youtu.be/')[1].split('?')[0]

                    self._global_cache.register_video(
                        video_path=str(final_path),
                        download_keyword=keyword,
                        topics=[],  # Will be updated after transcription
                        youtube_id=youtube_id,
                        youtube_url=youtube_url,
                        original_title=metadata.get('title', ''),
                        project_id='',  # Project info can be added later
                        duration=metadata.get('duration', 0)
                    )
                    logger.debug(f"Registered in global cache: {final_path.name}")
                except Exception as e:
                    logger.debug(f"Failed to register in global cache: {e}")

            # Track source for inter-keyword diversity analysis
            video_id = metadata.get('id', '')
            if video_id:
                self._record_source_for_keyword(keyword, video_id)

            # Clean up info.json
            if info_file.exists():
                info_file.unlink()

        return downloaded

    # =========================================================================
    # REPORTING METHODS
    # =========================================================================

    def get_download_estimate(self, num_keywords: int) -> Dict:
        """
        Estimate download stats.

        Args:
            num_keywords: Number of keywords to download

        Returns:
            Dict with estimates (videos, storage, time)
        """
        videos_per_keyword = sum(
            self._get_tier_value(t, 'per_keyword', 0)
            for t in self.DURATION_TIERS.keys()
        )
        total_videos = num_keywords * videos_per_keyword

        # Rough estimates
        avg_size_mb = 150
        avg_download_time = 30

        return {
            'keywords': num_keywords,
            'videos_per_keyword': videos_per_keyword,
            'total_videos': total_videos,
            'est_storage_gb': round(total_videos * avg_size_mb / 1024, 1),
            'est_time_minutes': round(total_videos * avg_download_time / 60, 0),
            'tiers': {
                name: f"{self._get_tier_value(name, 'min', 0)}s-{self._get_tier_value(name, 'max', 120)}s ({self._get_tier_value(name, 'per_keyword', 5)}/kw)"
                for name in self.DURATION_TIERS.keys()
            }
        }

    def get_inventory_report(self, output_dir: Path) -> Dict:
        """
        Report on existing footage.

        Args:
            output_dir: Output directory to scan

        Returns:
            Dict with inventory stats
        """
        output_dir = Path(output_dir)

        if not output_dir.exists():
            return {'total_videos': 0, 'total_duration': 0, 'keywords_covered': []}

        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
        videos = []

        for ext in video_extensions:
            videos.extend(output_dir.rglob(f'*{ext}'))

        # Get unique keywords from directory names
        keywords_covered = set()
        for video in videos:
            parent = video.parent.name
            # Extract keyword from dir name (remove tier suffix)
            for tier in self.DURATION_TIERS.keys():
                if parent.endswith(f'_{tier[0]}'):  # Match tier short code
                    keyword = parent[:-2]  # Remove _s, _m, _l
                    keywords_covered.add(keyword)
                    break

        # Calculate total duration from sources.json
        total_duration = sum(s.duration for s in self.sources) if self.sources else 0

        return {
            'total_videos': len(videos),
            'total_duration_hours': round(total_duration / 3600, 2),
            'keywords_covered': list(keywords_covered),
            'num_keywords_covered': len(keywords_covered)
        }

    def get_speed_stats(self) -> Dict:
        """Get download speed statistics for debugging.

        Returns:
            Dict with speed stats from the speed tracker:
            - samples: Number of downloads tracked
            - avg_speed_mbps: Average speed in MB/s
            - min_speed_mbps: Minimum speed in MB/s
            - max_speed_mbps: Maximum speed in MB/s
            - total_bytes: Total bytes downloaded
            - total_duration: Total download duration in seconds
            - records: List of recent download records
        """
        return self.speed_tracker.get_speed_stats()
