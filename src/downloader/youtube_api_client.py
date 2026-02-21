"""YouTube Data API client abstraction layer.

Provides programmatic access to YouTube Data API v3 for:
- Video search
- Video metadata retrieval
- Caption availability checking
- Quota tracking and budget management
- Error handling with exponential backoff retry
- Circuit breaker pattern for failure isolation
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import threading
import asyncio
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple
from urllib.parse import urlencode

import requests

import aiohttp
import asyncio
from dataclasses import dataclass, field

from .errors import (
    APIError,
    QuotaExceededError,
    InvalidCredentialsError,
    RateLimitError,
    YouTubeAPIError,
    YouTubeAPIQuotaExceededError,
    YouTubeAPIRateLimitedError,
    YouTubeAPIInvalidKeyError,
    YouTubeAPIPermissionDeniedError,
    YouTubeAPINetworkError,
    YouTubeAPITemporaryError,
    YouTubeAPIQuotaError,
    get_error_description,
    get_error_category,
    get_error_reason_info,
    parse_youtube_api_error_response,
)
from .api_fallback_handler import get_youtube_quota_reset_requested
from .youtube_retry_budget import YouTubeAPIRetryBudget, YouTubeAPIRetryBudgetConfig
from .youtube_api_cache import YouTubeAPISQLCache
from .rate_limit_predictor import RateLimitPredictor, PredictionConfidence
from .per_channel_circuit_breaker import PerChannelCircuitBreaker, PerChannelCircuitBreakerConfig
from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase

logger = logging.getLogger(__name__)


# US-152-4: YouTube API Circuit Breaker
# Track failures per endpoint type (search, videos, captions)


@dataclass
class YouTubeAPICircuitBreakerConfig:
    """Configuration for YouTube API circuit breaker.

    Attributes:
        enabled: Enable circuit breaker tracking
        consecutive_failures_threshold: Failures before circuit trips (default: 5)
        pause_seconds: Duration to pause when circuit is open (default: 60)
    """
    enabled: bool = True
    consecutive_failures_threshold: int = 5
    pause_seconds: float = 60.0


@dataclass
class YouTubeAPICircuitState:
    """State for a single endpoint's circuit breaker."""
    consecutive_failures: int = 0
    is_open: bool = False
    opened_at: Optional[float] = None
    total_trips: int = 0
    total_paused_seconds: float = 0.0


class YouTubeAPICircuitBreaker:
    """Circuit breaker for YouTube API endpoints.

    Tracks failures per endpoint type (search, videos, captions) and trips
    the circuit after consecutive failures threshold is reached.

    State transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED

    Usage:
        cb = YouTubeAPICircuitBreaker()

        # Before making an API call
        cb.check_and_wait("search")

        # After call succeeds
        cb.record_success("search")

        # After call fails
        cb.record_failure("search")
    """

    def __init__(self, config: Optional[YouTubeAPICircuitBreakerConfig] = None):
        """Initialize the circuit breaker.

        Args:
            config: Configuration. If None, uses defaults.
        """
        self._config = config or YouTubeAPICircuitBreakerConfig()
        # Track state per endpoint type
        self._states: Dict[str, YouTubeAPICircuitState] = {}

    @property
    def config(self) -> YouTubeAPICircuitBreakerConfig:
        return self._config

    def _get_state(self, endpoint: str) -> YouTubeAPICircuitState:
        """Get or create state for an endpoint."""
        if endpoint not in self._states:
            self._states[endpoint] = YouTubeAPICircuitState()
        return self._states[endpoint]

    def check_and_wait(self, endpoint: str) -> bool:
        """Check endpoint's circuit state and wait if necessary.

        Args:
            endpoint: The endpoint type ("search", "videos", "captions")

        Returns:
            True if operation should proceed, False if circuit breaker is disabled.
        """
        if not self._config.enabled:
            return True

        state = self._get_state(endpoint)

        if not state.is_open:
            return True

        # Circuit is open - check if pause duration has elapsed
        pause = self._config.pause_seconds
        elapsed = time.time() - state.opened_at
        remaining = pause - elapsed

        if remaining > 0:
            logger.info(
                f"YouTube API Circuit Breaker OPEN: endpoint '{endpoint}' pausing {remaining:.1f}s "
                f"(trip #{state.total_trips}, {state.consecutive_failures} consecutive failures)"
            )
            time.sleep(remaining)
            state.total_paused_seconds += remaining

        # Transition to half-open (ready to test)
        logger.info(
            f"YouTube API Circuit Breaker: endpoint '{endpoint}' pause complete, "
            f"allowing request (half-open)"
        )
        state.is_open = False
        state.opened_at = None

        return True

    def record_success(self, endpoint: str) -> None:
        """Record a successful call for an endpoint.

        Resets the consecutive failure counter and closes the circuit.

        Args:
            endpoint: The endpoint type ("search", "videos", "captions")
        """
        if not self._config.enabled:
            return

        state = self._get_state(endpoint)

        if state.consecutive_failures > 0:
            logger.debug(
                f"YouTube API Circuit Breaker: endpoint '{endpoint}' succeeded after "
                f"{state.consecutive_failures} failures, resetting counter"
            )

        state.consecutive_failures = 0
        state.is_open = False
        state.opened_at = None

    def record_failure(self, endpoint: str) -> bool:
        """Record a failure for an endpoint.

        Increments the consecutive failure counter. If threshold is reached,
        trips the circuit (opens it) and pauses future operations.

        Args:
            endpoint: The endpoint type ("search", "videos", "captions")

        Returns:
            True if circuit tripped (opened) as a result of this failure,
            False otherwise.
        """
        if not self._config.enabled:
            return False

        state = self._get_state(endpoint)
        state.consecutive_failures += 1

        threshold = self._config.consecutive_failures_threshold
        logger.debug(
            f"YouTube API Circuit Breaker: endpoint '{endpoint}' failure "
            f"({state.consecutive_failures}/{threshold})"
        )

        # Check if we've hit the threshold
        if state.consecutive_failures >= threshold:
            self._trip(endpoint, state)
            return True

        return False

    def _trip(self, endpoint: str, state: YouTubeAPICircuitState) -> None:
        """Trip the circuit breaker (open it).

        Args:
            endpoint: The endpoint type
            state: The circuit state for this endpoint
        """
        state.is_open = True
        state.opened_at = time.time()
        state.total_trips += 1

        logger.warning(
            f"YouTube API Circuit Breaker TRIPPED: endpoint '{endpoint}' after "
            f"{state.consecutive_failures} consecutive failures. "
            f"Pausing for {self._config.pause_seconds:.0f}s (trip #{state.total_trips})"
        )

    def get_state(self, endpoint: str) -> str:
        """Get the current circuit breaker state for an endpoint.

        Args:
            endpoint: The endpoint type

        Returns:
            'closed' - Normal operation, no failures or circuit reset
            'open' - Circuit is tripped, operations are paused
            'half_open' - Circuit was open but pause elapsed, testing recovery
        """
        state = self._get_state(endpoint)
        if state.is_open:
            return 'open'
        elif state.consecutive_failures > 0:
            return 'half_open'
        else:
            return 'closed'

    def get_stats(self, endpoint: str) -> dict:
        """Get circuit breaker statistics for an endpoint.

        Args:
            endpoint: The endpoint type

        Returns:
            Dict with circuit breaker statistics
        """
        state = self._get_state(endpoint)
        return {
            'enabled': self._config.enabled,
            'endpoint': endpoint,
            'state': self.get_state(endpoint),
            'is_open': state.is_open,
            'consecutive_failures': state.consecutive_failures,
            'total_trips': state.total_trips,
            'total_paused_seconds': round(state.total_paused_seconds, 3),
            'threshold': self._config.consecutive_failures_threshold,
            'pause_seconds': self._config.pause_seconds,
        }

    def get_all_stats(self) -> dict:
        """Get statistics for all endpoints.

        Returns:
            Dict mapping endpoint to stats
        """
        return {
            endpoint: self.get_stats(endpoint)
            for endpoint in self._states
        }

    def get_circuit_breaker_state(self) -> dict:
        """Get circuit breaker state for all endpoints (US-156-006).

        This method provides a convenient way to retrieve the current circuit
        breaker state for all tracked endpoints.

        Returns:
            Dict mapping endpoint to circuit breaker state with:
            - state: 'closed', 'open', or 'half_open'
            - is_open: Boolean indicating if circuit is open
            - consecutive_failures: Current failure count
            - total_trips: Total number of times circuit has tripped
            - total_paused_seconds: Total time spent paused due to open circuits
        """
        return self.get_all_stats()

    def reset(self, endpoint: str) -> None:
        """Manually reset the circuit breaker for an endpoint.

        Args:
            endpoint: The endpoint type to reset
        """
        state = self._get_state(endpoint)
        state.consecutive_failures = 0
        state.is_open = False
        state.opened_at = None
        logger.debug(f"YouTube API Circuit Breaker: manually reset endpoint '{endpoint}'")

    def reset_all(self) -> None:
        """Reset all endpoint circuits."""
        for state in self._states.values():
            state.consecutive_failures = 0
            state.is_open = False
            state.opened_at = None
        logger.debug("YouTube API Circuit Breaker: all circuits manually reset")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize circuit breaker state to dictionary for persistence.

        Returns:
            Dict with circuit breaker state for all endpoints
        """
        return {
            "config": {
                "enabled": self._config.enabled,
                "consecutive_failures_threshold": self._config.consecutive_failures_threshold,
                "pause_seconds": self._config.pause_seconds,
            },
            "states": {
                endpoint: {
                    "consecutive_failures": state.consecutive_failures,
                    "is_open": state.is_open,
                    "opened_at": state.opened_at,
                    "total_trips": state.total_trips,
                    "total_paused_seconds": state.total_paused_seconds,
                }
                for endpoint, state in self._states.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load circuit breaker state from dictionary.

        Args:
            state_dict: Dict with circuit breaker state (from to_dict)
        """
        if not state_dict:
            return

        # Load config if present
        if "config" in state_dict:
            config = state_dict["config"]
            self._config.enabled = config.get("enabled", self._config.enabled)
            self._config.consecutive_failures_threshold = config.get(
                "consecutive_failures_threshold", self._config.consecutive_failures_threshold
            )
            self._config.pause_seconds = config.get("pause_seconds", self._config.pause_seconds)

        # Load endpoint states
        if "states" in state_dict:
            for endpoint, endpoint_state in state_dict["states"].items():
                state = self._get_state(endpoint)
                state.consecutive_failures = endpoint_state.get("consecutive_failures", 0)
                state.is_open = endpoint_state.get("is_open", False)
                state.opened_at = endpoint_state.get("opened_at")
                state.total_trips = endpoint_state.get("total_trips", 0)
                state.total_paused_seconds = endpoint_state.get("total_paused_seconds", 0.0)

                # If circuit was open, check if pause time has elapsed
                if state.is_open and state.opened_at:
                    elapsed = time.time() - state.opened_at
                    if elapsed >= self._config.pause_seconds:
                        # Circuit pause has elapsed, allow requests
                        state.is_open = False
                        state.opened_at = None
                        logger.info(
                            f"YouTube API Circuit Breaker: restored state for '{endpoint}' "
                            f"({state.consecutive_failures} failures), circuit now closed"
                        )
                    else:
                        remaining = self._config.pause_seconds - elapsed
                        logger.info(
                            f"YouTube API Circuit Breaker: restored state for '{endpoint}' "
                            f"circuit still open, {remaining:.1f}s remaining"
                        )


# US-152-8: Token bucket rate limiter for YouTube API
# Limits requests to specified requests per second (default: 10 RPS as per YouTube guidelines)


class YouTubeAPITokenBucket:
    """Token bucket rate limiter for YouTube API requests.

    Implements the token bucket algorithm to limit requests per second.
    YouTube recommends a limit of 10 requests/second for the Data API.

    The token bucket algorithm allows bursts up to the bucket capacity
    while enforcing the average rate over time.
    """

    def __init__(self, rate: float = 10.0, capacity: Optional[float] = None):
        """Initialize the token bucket.

        Args:
            rate: Number of tokens (requests) added per second (default: 10 RPS)
            capacity: Maximum bucket capacity (default: equal to rate for 1 second burst)
        """
        self._rate = rate
        self._capacity = capacity if capacity is not None else rate
        self._tokens = self._capacity
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

        # Statistics
        self._total_waits: float = 0.0
        self._wait_count: int = 0

    @property
    def rate(self) -> float:
        """Get the rate (requests per second)."""
        return self._rate

    @property
    def capacity(self) -> float:
        """Get the bucket capacity."""
        return self._capacity

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last update."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_update = now

    def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket, waiting if necessary.

        Implements the token bucket algorithm:
        - If enough tokens are available, returns immediately (0 wait)
        - If not enough tokens, waits until enough tokens accumulate

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            Time waited in seconds (0 if tokens were immediately available)
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time for tokens to become available
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._rate

            logger.debug(
                f"Rate limiter: waiting {wait_time:.3f}s for {tokens} token(s) "
                f"(available: {self._tokens:.2f}, rate: {self._rate} RPS)"
            )

            # Wait for tokens to become available
            time.sleep(wait_time)

            # Update statistics
            self._total_waits += wait_time
            self._wait_count += 1

            # Consume the tokens
            self._tokens = 0.0
            self._last_update = time.monotonic()

            return wait_time

    def get_stats(self) -> dict:
        """Get rate limiter statistics.

        Returns:
            Dict with rate limiter stats
        """
        with self._lock:
            return {
                'rate': self._rate,
                'capacity': self._capacity,
                'available_tokens': round(self._tokens, 2),
                'total_waits': round(self._total_waits, 3),
                'wait_count': self._wait_count,
                'average_wait': round(self._total_waits / self._wait_count, 3) if self._wait_count > 0 else 0.0,
            }

    def reset(self) -> None:
        """Reset the token bucket."""
        with self._lock:
            self._tokens = self._capacity
            self._last_update = time.monotonic()
            self._total_waits = 0.0
            self._wait_count = 0


# US-150-8: Session-level cache for API key validation results
# Caches validation result to avoid repeated health check calls in same session
_youtube_api_validation_cache: Dict[str, tuple[bool, str, dict, float]] = {}
_youtube_api_validation_cache_lock = threading.Lock()


def get_cached_validation(api_key: str) -> Optional[tuple[bool, str, dict]]:
    """Get cached validation result for an API key.

    Returns:
        Tuple of (is_valid, error_message, quota_info) if cached, None otherwise.
        Cache is valid for the current session only.
    """
    # Use first 8 chars of key as cache key for security
    cache_key = api_key[:8] if api_key else ""
    with _youtube_api_validation_cache_lock:
        if cache_key in _youtube_api_validation_cache:
            is_valid, error_msg, quota_info, cached_at = _youtube_api_validation_cache[cache_key]
            # Cache valid for 5 minutes
            if time.time() - cached_at < 300:
                return (is_valid, error_msg, quota_info)
    return None


def set_cached_validation(api_key: str, is_valid: bool, error_message: str, quota_info: dict):
    """Cache validation result for an API key.

    Args:
        api_key: The API key that was validated
        is_valid: Whether the key is valid
        error_message: Error message if not valid
        quota_info: Quota information dict
    """
    cache_key = api_key[:8] if api_key else ""
    with _youtube_api_validation_cache_lock:
        _youtube_api_validation_cache[cache_key] = (is_valid, error_message, quota_info, time.time())


# API endpoint
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Quota costs per operation
QUOTA_COST_SEARCH = 100
QUOTA_COST_VIDEOS = 1
QUOTA_COST_CHANNELS = 1
QUOTA_COST_CAPTIONS = 50


@dataclass
class VideoSearchResult:
    """Result from YouTube API video search."""
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    published_at: str
    description: str = ""
    # Additional fields available via API
    thumbnail_url: str = ""
    view_count: int = 0
    # Channel metadata (enriched via channels.list API)
    subscriber_count: int = 0
    total_views: int = 0
    channel_created_date: str = ""
    # US-155-4: Date range info from search
    search_date_range: Optional[Dict[str, str]] = None
    # US-155-005: Video category filter info
    video_category_id: Optional[str] = None


@dataclass
class VideoDetails:
    """Detailed video metadata from YouTube API."""
    video_id: str
    duration: str  # ISO 8601 duration (e.g., "PT1H2M10S")
    duration_seconds: int
    tags: List[str] = field(default_factory=list)
    category_id: str = ""
    topic_details: Dict[str, Any] = field(default_factory=dict)
    # US-150-6: Dedicated topic_categories field for easier access
    topic_categories: List[str] = field(default_factory=list)
    caption_available: bool = False
    dimension: str = ""
    definition: str = ""
    # US-148-8: Engagement metrics from videos.list API
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0

    # US-158-010: Quality score based on engagement metrics
    # Calculated as: viewCount * 0.7 + likeCount * 0.2 + commentCount * 0.1
    quality_score: float = 0.0

    # US-158-007: Channel metadata for subscriber filtering
    channel_id: str = ""
    channel_title: str = ""
    subscriber_count: int = 0


@dataclass
class CaptionInfo:
    """Caption track information."""
    language: str
    track_id: str
    is_auto_generated: bool


@dataclass
class YouTubeAPIMetrics:
    """Metrics for YouTube Data API usage tracking.

    Tracks API call counts, quota consumption, and fallback events
    for observability and cost analysis.
    """

    # API call counts by endpoint
    search_calls: int = 0
    videos_calls: int = 0
    channels_calls: int = 0
    captions_calls: int = 0

    # API call errors by endpoint
    search_errors: int = 0
    videos_errors: int = 0
    channels_errors: int = 0
    captions_errors: int = 0

    # API call errors by HTTP status code (US-152-7)
    errors_403: int = 0
    errors_429: int = 0
    errors_500: int = 0
    errors_other: int = 0

    # Fallback events: API failed, yt-dlp succeeded
    fallback_to_ytdlp: int = 0
    fallback_events: List[Dict[str, Any]] = field(default_factory=list)

    # Quota consumption tracking with timestamps for aggregates
    quota_history: List[Dict[str, Any]] = field(default_factory=list)

    # Session-level totals (not reset on export)
    session_quota_used: int = 0

    # Session tracking (US-152-7)
    session_start_time: Optional[float] = None
    session_end_time: Optional[float] = None

    # US-149-9: Cache hit/miss metrics for SQLite query cache
    cache_hits: int = 0
    cache_misses: int = 0

    # US-150-4: Caption enumeration savings tracking
    caption_checks_api_success: int = 0  # Videos checked via captions.list API
    caption_checks_skipped_yt_dlp: int = 0  # Videos where we skipped yt-dlp (no captions)

    # Baseline cost of yt-dlp caption fetch (in arbitrary "cost units")
    # Using captions.list (50 units) + yt-dlp fetch vs just captions.list
    YTDLP_COST_BASELINE = 100  # Estimated cost units saved per avoided yt-dlp call

    # US-155-003: Abnormal quota usage rate warnings
    abnormal_rate_warnings: int = 0
    abnormal_rate_history: List[Dict[str, Any]] = field(default_factory=list)

    # US-155-5: Parallel batch execution metrics
    parallel_batch_operations: int = 0  # Number of batch operations using parallel execution
    sequential_batch_operations: int = 0  # Number of batch operations using sequential execution
    parallel_batch_time_saved_ms: float = 0.0  # Time saved by using parallel execution (ms)

    # US-153-10: Latency tracking per endpoint (in milliseconds)
    search_latencies: List[float] = field(default_factory=list)
    videos_latencies: List[float] = field(default_factory=list)
    channels_latencies: List[float] = field(default_factory=list)
    captions_latencies: List[float] = field(default_factory=list)

    # US-153-10: Key rotation events tracking
    key_rotation_events: List[Dict[str, Any]] = field(default_factory=list)

    # US-154-6: Key utilization balance tracking (ratio of min/avg remaining quota)
    key_utilization_balance_history: List[float] = field(default_factory=list)

    # US-154-10: Pagination metrics
    search_pages_total: int = 0  # Total pages fetched across all searches
    search_queries_with_pagination: int = 0  # Number of searches that required pagination
    search_page_timeouts: int = 0  # Number of times pagination timed out
    duplicate_page_tokens: int = 0  # Duplicate pageToken occurrences

    # US-157-011: Pagination efficiency metrics
    pagination_results_requested: int = 0  # Total results requested across all paginated searches
    pagination_results_returned: int = 0  # Total results returned from paginated searches

    # US-153-10: Quota usage by operation type
    search_quota: int = 0
    videos_quota: int = 0
    channels_quota: int = 0
    captions_quota: int = 0

    # US-155-3: Caption language distribution tracking
    caption_language_distribution: Dict[str, int] = field(default_factory=dict)

    # US-155-008: In-session request deduplication tracking
    deduplicated_requests: int = 0  # Count of requests deduplicated within session

    # US-155-005: Parallel video details fetching metrics
    video_details_parallel_batches: int = 0  # Number of batch operations using parallel execution
    video_details_sequential_batches: int = 0  # Number of batch operations using sequential execution
    video_details_parallel_time_ms: float = 0.0  # Total time spent in parallel mode (ms)
    video_details_sequential_time_ms: float = 0.0  # Total time spent in sequential mode (ms)
    video_details_parallel_videos: int = 0  # Total videos processed in parallel mode
    video_details_sequential_videos: int = 0  # Total videos processed in sequential mode

    # US-155-010: Error categorization metrics for debugging
    # Track specific error types for debugging and analysis
    error_categories: Dict[str, int] = field(default_factory=lambda: {
        "quota_exceeded": 0,
        "rate_limited": 0,
        "invalid_key": 0,
        "permission_denied": 0,
        "network_error": 0,
        "timeout": 0,
        "temporary_error": 0,
        "unknown": 0,
    })

    def reset(self) -> None:
        """Reset interval-based counters (keeps session totals)."""
        self.search_calls = 0
        self.videos_calls = 0
        self.channels_calls = 0
        self.captions_calls = 0
        self.search_errors = 0
        self.videos_errors = 0
        self.channels_errors = 0
        self.captions_errors = 0
        self.caption_checks_api_success = 0
        self.caption_checks_skipped_yt_dlp = 0
        # US-153-10: Reset latency tracking
        self.search_latencies.clear()
        self.videos_latencies.clear()
        self.channels_latencies.clear()
        self.captions_latencies.clear()
        # US-153-10: Reset quota by endpoint
        self.search_quota = 0
        self.videos_quota = 0
        self.channels_quota = 0
        self.captions_quota = 0
        # US-154-6: Reset key utilization balance history
        self.key_utilization_balance_history.clear()
        # US-154-10: Reset pagination metrics
        self.search_pages_total = 0
        self.search_queries_with_pagination = 0
        # US-157-011: Reset pagination efficiency metrics
        self.pagination_results_requested = 0
        self.pagination_results_returned = 0
        # US-155-003: Reset abnormal rate warning tracking
        self.abnormal_rate_warnings = 0
        self.abnormal_rate_history.clear()

    def record_api_call(self, endpoint: str, success: bool = True) -> None:
        """Record an API call.

        Args:
            endpoint: API endpoint name (search, videos, channels, captions)
            success: Whether the call succeeded
        """
        if endpoint == "search":
            self.search_calls += 1
            if not success:
                self.search_errors += 1
        elif endpoint == "videos":
            self.videos_calls += 1
            if not success:
                self.videos_errors += 1
        elif endpoint == "channels":
            self.channels_calls += 1
            if not success:
                self.channels_errors += 1
        elif endpoint == "captions":
            self.captions_calls += 1
            if not success:
                self.captions_errors += 1

    def record_fallback(self, endpoint: str, reason: str, query: str = "") -> None:
        """Record a fallback event where API failed and yt-dlp was used.

        Args:
            endpoint: API endpoint that failed
            reason: Reason for fallback (quota_exceeded, error, circuit_breaker)
            query: Optional search query or video ID
        """
        self.fallback_to_ytdlp += 1
        self.fallback_events.append({
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "reason": reason,
            "query": query,
        })

    def record_caption_api_success(self, has_captions: bool) -> None:
        """Record successful caption enumeration via captions.list API.

        US-150-4: Track quota savings vs yt-dlp approach.

        Args:
            has_captions: True if video has captions (we skip yt-dlp fetch),
                         False if no captions (we skip yt-dlp entirely)
        """
        self.caption_checks_api_success += 1
        if not has_captions:
            # No captions found - we avoided a full yt-dlp fetch
            self.caption_checks_skipped_yt_dlp += 1

    def record_caption_language(self, language: str) -> None:
        """Record caption language selection for metrics tracking.

        US-155-3: Track caption language distribution.

        Args:
            language: Language code of the selected caption track
        """
        self.caption_language_distribution[language] = self.caption_language_distribution.get(language, 0) + 1

    def get_caption_language_distribution(self) -> Dict[str, int]:
        """Get caption language distribution for metrics.

        US-155-3: Returns the distribution of caption languages selected.

        Returns:
            Dictionary mapping language codes to counts
        """
        return dict(self.caption_language_distribution)

    def get_quota_savings(self) -> Dict[str, Any]:
        """Calculate quota savings from caption enumeration API.

        US-150-4: Returns savings metrics comparing captions.list vs yt-dlp.

        Returns:
            Dictionary with savings metrics:
            - caption_checks_api_success: Number of videos checked via API
            - caption_checks_skipped_yt_dlp: Videos where yt-dlp was avoided
            - estimated_quota_saved: Estimated quota units saved
            - api_quota_used: Quota units consumed by captions.list calls
        """
        api_quota_used = self.caption_checks_api_success * QUOTA_COST_CAPTIONS
        # Savings = avoided yt-dlp calls * baseline cost - API cost
        estimated_quota_saved = (self.caption_checks_skipped_yt_dlp * self.YTDLP_COST_BASELINE) - api_quota_used

        return {
            "caption_checks_api_success": self.caption_checks_api_success,
            "caption_checks_skipped_yt_dlp": self.caption_checks_skipped_yt_dlp,
            "estimated_quota_saved": max(0, estimated_quota_saved),  # Can't be negative
            "api_quota_used": api_quota_used,
            "ytdlp_baseline_cost": self.YTDLP_COST_BASELINE,
        }

    def get_pagination_stats(self) -> Dict[str, Any]:
        """Get pagination statistics (US-154-10).

        Returns:
            Dictionary with pagination metrics:
            - search_pages_total: Total pages fetched across all searches
            - search_queries_with_pagination: Number of searches requiring pagination
            - average_pages_per_search: Average pages per search (0 if no paginated searches)
            - search_page_timeouts: Number of pagination timeouts
            - duplicate_page_tokens: Number of duplicate pageToken occurrences
        """
        avg_pages = 0.0
        if self.search_queries_with_pagination > 0:
            avg_pages = self.search_pages_total / self.search_queries_with_pagination

        return {
            "search_pages_total": self.search_pages_total,
            "search_queries_with_pagination": self.search_queries_with_pagination,
            "average_pages_per_search": round(avg_pages, 2),
            "search_page_timeouts": self.search_page_timeouts,
            "duplicate_page_tokens": self.duplicate_page_tokens,
            "pagination_results_requested": self.pagination_results_requested,
            "pagination_results_returned": self.pagination_results_returned,
            "pagination_efficiency": round(
                self.pagination_results_returned / self.pagination_results_requested
                if self.pagination_results_requested > 0 else 0.0,
                3
            ),
        }

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record API call latency (US-153-10).

        Args:
            endpoint: API endpoint name (search, videos, channels, captions)
            latency_ms: Latency in milliseconds
        """
        if endpoint == "search":
            self.search_latencies.append(latency_ms)
        elif endpoint == "videos":
            self.videos_latencies.append(latency_ms)
        elif endpoint == "channels":
            self.channels_latencies.append(latency_ms)
        elif endpoint == "captions":
            self.captions_latencies.append(latency_ms)

    def record_quota_usage_by_endpoint(self, endpoint: str, quota_cost: int) -> None:
        """Record quota usage by endpoint type (US-153-10).

        Args:
            endpoint: API endpoint name (search, videos, channels, captions)
            quota_cost: Quota cost for this operation
        """
        if endpoint == "search":
            self.search_quota += quota_cost
        elif endpoint == "videos":
            self.videos_quota += quota_cost
        elif endpoint == "channels":
            self.channels_quota += quota_cost
        elif endpoint == "captions":
            self.captions_quota += quota_cost

    def record_key_rotation(self, from_key_index: int, to_key_index: int, reason: str) -> None:
        """Record API key rotation event (US-153-10).

        Args:
            from_key_index: Index of the previous API key
            to_key_index: Index of the new API key
            reason: Reason for rotation (quota_exceeded, error, health_check)
        """
        self.key_rotation_events.append({
            "timestamp": datetime.now().isoformat(),
            "from_key_index": from_key_index,
            "to_key_index": to_key_index,
            "reason": reason,
        })

    def record_key_utilization_balance(self, balance_score: float) -> None:
        """Record key utilization balance score (US-154-6).

        Records the ratio of minimum to average remaining quota across all keys.
        A score of 1.0 means perfect balance, lower scores indicate imbalance.

        Args:
            balance_score: Ratio of min_remaining / avg_remaining (0.0 to 1.0)
        """
        self.key_utilization_balance_history.append({
            "timestamp": datetime.now().isoformat(),
            "balance_score": balance_score,
        })

    def _calculate_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate percentiles from latency list.

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            Dictionary with p50, p95, p99 percentiles
        """
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(n * p)
            if idx >= n:
                idx = n - 1
            return sorted_latencies[idx]

        return {
            "p50": round(percentile(0.50), 2),
            "p95": round(percentile(0.95), 2),
            "p99": round(percentile(0.99), 2),
        }

    def _calculate_success_rate(self, calls: int, errors: int) -> float:
        """Calculate success rate percentage.

        Args:
            calls: Total number of calls
            errors: Number of failed calls

        Returns:
            Success rate as percentage (0-100)
        """
        if calls == 0:
            return 0.0
        return round(((calls - errors) / calls) * 100, 2)

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage.

        Returns:
            Cache hit rate as percentage (0-100)
        """
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return round((self.cache_hits / total) * 100, 2)

    def record_quota_usage(self, quota_used: int) -> None:
        """Record quota usage with timestamp for historical tracking.

        Args:
            quota_used: Current quota used
        """
        self.quota_history.append({
            "timestamp": datetime.now().isoformat(),
            "quota_used": quota_used,
        })
        self.session_quota_used = quota_used

    def record_abnormal_rate_warning(self, rate_ratio: float) -> None:
        """Record an abnormal quota usage rate warning (US-155-003).

        Args:
            rate_ratio: Ratio of current rate to average rate
        """
        self.abnormal_rate_warnings += 1
        self.abnormal_rate_history.append({
            "timestamp": datetime.now().isoformat(),
            "rate_ratio": rate_ratio,
        })

    def get_total_calls(self) -> int:
        """Get total API calls across all endpoints."""
        return (self.search_calls + self.videos_calls +
                self.channels_calls + self.captions_calls)

    def get_total_errors(self) -> int:
        """Get total errors across all endpoints."""
        return (self.search_errors + self.videos_errors +
                self.channels_errors + self.captions_errors)

    def get_errors_by_type(self) -> Dict[str, int]:
        """Get errors grouped by HTTP status code (US-152-7).

        Returns:
            Dictionary with error counts by HTTP status code.
        """
        return {
            "403": self.errors_403,
            "429": self.errors_429,
            "500": self.errors_500,
            "other": self.errors_other,
            "total": self.errors_403 + self.errors_429 + self.errors_500 + self.errors_other,
        }

    def record_error_by_status(self, status_code: int) -> None:
        """Record an API error by HTTP status code (US-152-7).

        Args:
            status_code: HTTP status code from the error response
        """
        if status_code == 403:
            self.errors_403 += 1
        elif status_code == 429:
            self.errors_429 += 1
        elif status_code >= 500:
            self.errors_500 += 1
        else:
            self.errors_other += 1

    def record_error_category(self, error_type: str) -> None:
        """Record an error by category for debugging (US-155-010).

        Args:
            error_type: The specific error type (quota_exceeded, rate_limited,
                       invalid_key, permission_denied, network_error, timeout,
                       temporary_error, unknown)
        """
        if error_type in self.error_categories:
            self.error_categories[error_type] += 1
        else:
            self.error_categories["unknown"] += 1

    def get_error_category_counts(self) -> Dict[str, int]:
        """Get error category counts for debugging.

        Returns:
            Dictionary mapping error types to their counts
        """
        return dict(self.error_categories)

    def start_session(self) -> None:
        """Record session start time (US-152-7)."""
        if self.session_start_time is None:
            self.session_start_time = time.time()

    def end_session(self) -> None:
        """Record session end time (US-152-7)."""
        self.session_end_time = time.time()

    def get_session_duration_seconds(self) -> float:
        """Get session duration in seconds (US-152-7).

        Returns:
            Session duration in seconds, or 0 if session hasn't ended.
        """
        if self.session_start_time is None:
            return 0.0
        end_time = self.session_end_time if self.session_end_time else time.time()
        return end_time - self.session_start_time

    def get_api_calls_per_minute(self) -> float:
        """Calculate API calls per minute (US-152-7).

        Returns:
            Average API calls per minute over the session.
        """
        duration = self.get_session_duration_seconds()
        if duration <= 0:
            return 0.0
        total_calls = self.get_total_calls()
        return (total_calls / duration) * 60.0

    def get_quota_aggregates(self) -> Dict[str, Any]:
        """Calculate daily/weekly/monthly quota aggregates.

        Returns:
            Dictionary with quota aggregates by time period
        """
        if not self.quota_history:
            return {
                "daily": {"max": 0, "avg": 0},
                "weekly": {"max": 0, "avg": 0},
                "monthly": {"max": 0, "avg": 0},
                "session_total": self.session_quota_used,
            }

        # Group by date for daily calculations
        daily_usage: Dict[str, List[int]] = defaultdict(list)
        for entry in self.quota_history:
            date = entry["timestamp"][:10]  # YYYY-MM-DD
            daily_usage[date].append(entry["quota_used"])

        daily_max = max((max(v) for v in daily_usage.values()), default=0)
        daily_avg = sum(sum(v) for v in daily_usage.values()) / max(len(daily_usage), 1)

        # Weekly (approximate - use 7-day windows)
        weekly_max = daily_max * 7
        weekly_avg = daily_avg * 7

        # Monthly (approximate - use 30-day windows)
        monthly_max = daily_max * 30
        monthly_avg = daily_avg * 30

        return {
            "daily": {"max": daily_max, "avg": round(daily_avg, 2)},
            "weekly": {"max": weekly_max, "avg": round(weekly_avg, 2)},
            "monthly": {"max": monthly_max, "avg": round(monthly_avg, 2)},
            "session_total": self.session_quota_used,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "api_calls": {
                "search": self.search_calls,
                "videos": self.videos_calls,
                "channels": self.channels_calls,
                "captions": self.captions_calls,
                "total": self.get_total_calls(),
            },
            "api_errors": {
                "search": self.search_errors,
                "videos": self.videos_errors,
                "channels": self.channels_errors,
                "captions": self.captions_errors,
                "total": self.get_total_errors(),
            },
            # US-153-10: Success rate per endpoint
            "success_rate": {
                "search": self._calculate_success_rate(self.search_calls, self.search_errors),
                "videos": self._calculate_success_rate(self.videos_calls, self.videos_errors),
                "channels": self._calculate_success_rate(self.channels_calls, self.channels_errors),
                "captions": self._calculate_success_rate(self.captions_calls, self.captions_errors),
                "overall": self._calculate_success_rate(self.get_total_calls(), self.get_total_errors()),
            },
            # US-153-10: Latency percentiles per endpoint
            "latency_percentiles": {
                "search": self._calculate_percentiles(self.search_latencies),
                "videos": self._calculate_percentiles(self.videos_latencies),
                "channels": self._calculate_percentiles(self.channels_latencies),
                "captions": self._calculate_percentiles(self.captions_latencies),
            },
            # US-153-10: Key rotation events
            "key_rotation_events": list(self.key_rotation_events),
            # US-153-10: Quota usage by operation type
            "quota_by_operation": {
                "search": self.search_quota,
                "videos": self.videos_quota,
                "channels": self.channels_quota,
                "captions": self.captions_quota,
                "total": self.search_quota + self.videos_quota + self.channels_quota + self.captions_quota,
            },
            "errors_by_type": self.get_errors_by_type(),  # US-152-7
            # US-155-010: Error categorization for debugging
            "error_categories": self.get_error_category_counts(),
            "fallback_events": {
                "total": self.fallback_to_ytdlp,
                "events": list(self.fallback_events),
            },
            "quota_aggregates": self.get_quota_aggregates(),
            "cache_metrics": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self._calculate_cache_hit_rate(),
            },
            # US-152-7: Session metrics
            "session": {
                "duration_seconds": round(self.get_session_duration_seconds(), 2),
                "api_calls_per_minute": round(self.get_api_calls_per_minute(), 2),
            },
            # US-154-6: Key utilization balance metrics
            "key_utilization_balance": {
                "history": list(self.key_utilization_balance_history),
                "latest": self.key_utilization_balance_history[-1].get("balance_score") if self.key_utilization_balance_history else None,
                "average": sum(h.get("balance_score", 0) for h in self.key_utilization_balance_history) / len(self.key_utilization_balance_history) if self.key_utilization_balance_history else None,
            },
            # US-154-10: Pagination metrics
            "pagination": self.get_pagination_stats(),
            # US-155-3: Caption language distribution
            "caption_language_distribution": self.get_caption_language_distribution(),
            # US-155-003: Abnormal quota usage rate warnings
            "abnormal_rate_warnings": {
                "count": self.abnormal_rate_warnings,
                "history": list(self.abnormal_rate_history),
            },
        }

    def record_cache_hit(self) -> None:
        """Record a cache hit event."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss event."""
        self.cache_misses += 1


# US-152-8: Token bucket rate limiter for YouTube API requests
# US-155-008: Adaptive rate limiting based on response times
class YouTubeAPIRateLimiter:
    """Token bucket rate limiter for YouTube API requests.

    Implements the token bucket algorithm to control the rate of API requests.
    The bucket has a capacity (burst_size) and is refilled at a steady rate (rate).

    This ensures requests stay within YouTube's recommended limit of 10 requests/second
    while allowing short bursts above the steady-state rate.

    Adaptive Rate Limiting (US-155-008):
    - Monitors API response latency continuously
    - Reduces RPS when average latency exceeds high threshold (default 500ms)
    - Increases RPS gradually when latency is below low threshold (default 200ms)
    - Thread-safe rate adjustment

    Usage:
        limiter = YouTubeAPIRateLimiter(rate=10.0, burst_size=20)

        # Before making an API call
        limiter.acquire()

        # Record response latency after API call
        limiter.record_latency(0.150)  # 150ms response time

        # Make API call...
    """

    def __init__(
        self,
        rate: float = 10.0,
        burst_size: int = 20,
        adaptive_enabled: bool = True,
        latency_high_threshold_ms: float = 500.0,
        latency_low_threshold_ms: float = 200.0,
        rate_decrease_factor: float = 0.8,
        rate_increase_factor: float = 1.1,
        min_adaptive_rate: float = 1.0,
        max_adaptive_rate: float = 20.0,
        latency_smoothing_window: int = 10,
        min_requests_before_adjustment: int = 5,
        # US-156-008: Burst handling options
        burst_allowance: int = 10,
        smooth_start: bool = False,
        smooth_start_duration: float = 5.0,
    ):
        """Initialize the token bucket rate limiter.

        Args:
            rate: Refill rate in tokens per second (default: 10.0 for YouTube recommended limit)
            burst_size: Maximum number of tokens in the bucket (default: 20)
            adaptive_enabled: Enable adaptive rate limiting based on latency (default: True)
            latency_high_threshold_ms: Threshold above which to reduce RPS (default: 500ms)
            latency_low_threshold_ms: Threshold below which to increase RPS (default: 200ms)
            rate_decrease_factor: Factor to reduce rate by when latency is high (default: 0.8)
            rate_increase_factor: Factor to increase rate by when latency is low (default: 1.1)
            min_adaptive_rate: Minimum RPS allowed (default: 1.0)
            max_adaptive_rate: Maximum RPS allowed (default: 20.0)
            latency_smoothing_window: Number of samples for latency averaging (default: 10)
            min_requests_before_adjustment: Minimum requests before rate adjustment (default: 5)
            burst_allowance: Additional tokens available for short bursts above rate limit (default: 10)
            smooth_start: Enable gradual rate ramp-up at start (default: False)
            smooth_start_duration: Duration in seconds for smooth start ramp-up (default: 5.0)
        """
        self._initial_rate = rate
        self._rate = rate
        self._burst_size = burst_size
        self._tokens = float(burst_size + burst_allowance)
        self._last_update = time.time()
        self._lock = threading.Lock()

        # Metrics tracking
        self._total_waits = 0
        self._total_wait_time = 0.0
        self._max_wait_time = 0.0

        # US-155-008: Adaptive rate limiting configuration
        self._adaptive_enabled = adaptive_enabled
        self._latency_high_threshold_ms = latency_high_threshold_ms
        self._latency_low_threshold_ms = latency_low_threshold_ms
        self._rate_decrease_factor = rate_decrease_factor
        self._rate_increase_factor = rate_increase_factor
        self._min_adaptive_rate = min_adaptive_rate
        self._max_adaptive_rate = max_adaptive_rate
        self._latency_smoothing_window = latency_smoothing_window
        self._min_requests_before_adjustment = min_requests_before_adjustment

        # US-155-008: Latency tracking
        self._latency_samples: List[float] = []
        self._total_requests = 0
        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0
        self._rate_adjustments = 0

        # US-156-008: Burst handling configuration
        self._burst_allowance = burst_allowance
        self._smooth_start = smooth_start
        self._smooth_start_duration = smooth_start_duration
        self._burst_usage = 0
        self._burst_exhausted_warnings = 0
        self._smooth_start_active = smooth_start
        self._smooth_start_start_time = time.time() if smooth_start else None
        # Start at 10% of target rate for smooth start ramp-up
        self._smooth_start_current_rate = rate * 0.1 if smooth_start else rate

    @property
    def rate(self) -> float:
        """Get the refill rate."""
        return self._rate

    @property
    def burst_size(self) -> int:
        """Get the burst size."""
        return self._burst_size

    @property
    def adaptive_enabled(self) -> bool:
        """Check if adaptive rate limiting is enabled."""
        return self._adaptive_enabled

    @property
    def burst_allowance(self) -> int:
        """Get the burst allowance."""
        return self._burst_allowance

    @property
    def smooth_start(self) -> bool:
        """Check if smooth start is enabled."""
        return self._smooth_start

    @property
    def burst_usage(self) -> int:
        """Get the current burst usage (tokens consumed from allowance)."""
        return self._burst_usage

    def get_current_rate(self) -> float:
        """Get the current effective rate including smooth start adjustment.

        Returns:
            Current rate in tokens per second
        """
        with self._lock:
            if self._smooth_start_active:
                return self._smooth_start_current_rate
            return self._rate

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # US-156-008: Handle smooth start
        effective_rate = self._rate
        if self._smooth_start_active:
            # Calculate progress through smooth start period
            elapsed_since_start = now - self._smooth_start_start_time
            if elapsed_since_start >= self._smooth_start_duration:
                # Smooth start complete
                self._smooth_start_active = False
                self._smooth_start_current_rate = self._rate
            else:
                # Calculate gradual ramp-up
                progress = elapsed_since_start / self._smooth_start_duration
                self._smooth_start_current_rate = self._initial_rate * progress + self._rate * (1 - progress)
                effective_rate = self._smooth_start_current_rate

        # Add tokens based on elapsed time
        new_tokens = elapsed * effective_rate
        max_tokens = self._burst_size + self._burst_allowance
        self._tokens = min(max_tokens, self._tokens + new_tokens)

    def record_latency(self, latency_seconds: float) -> None:
        """Record API response latency for adaptive rate limiting.

        Args:
            latency_seconds: Response time in seconds (e.g., 0.150 for 150ms)
        """
        if not self._adaptive_enabled:
            return

        latency_ms = latency_seconds * 1000.0

        with self._lock:
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > self._latency_smoothing_window:
                self._latency_samples.pop(0)

            self._total_requests += 1
            self._total_latency_ms += latency_ms
            self._max_latency_ms = max(self._max_latency_ms, latency_ms)

            # Check if we should adjust rate
            self._maybe_adjust_rate()

    def _maybe_adjust_rate(self) -> None:
        """Adjust rate based on latency if conditions are met."""
        # Need minimum samples before adjusting
        if len(self._latency_samples) < self._min_requests_before_adjustment:
            return

        # Calculate average latency
        avg_latency = sum(self._latency_samples) / len(self._latency_samples)
        old_rate = self._rate

        # Determine rate adjustment
        if avg_latency > self._latency_high_threshold_ms:
            # High latency - reduce rate
            new_rate = max(self._min_adaptive_rate, self._rate * self._rate_decrease_factor)
            if new_rate < self._rate:
                self._rate = new_rate
                self._rate_adjustments += 1
                logger.info(
                    f"Adaptive rate limit: reducing RPS from {old_rate:.2f} to {self._rate:.2f} "
                    f"(avg latency {avg_latency:.1f}ms exceeds {self._latency_high_threshold_ms:.1f}ms threshold)"
                )
        elif avg_latency < self._latency_low_threshold_ms:
            # Low latency - increase rate
            new_rate = min(self._max_adaptive_rate, self._rate * self._rate_increase_factor)
            if new_rate > self._rate:
                self._rate = new_rate
                self._rate_adjustments += 1
                logger.info(
                    f"Adaptive rate limit: increasing RPS from {old_rate:.2f} to {self._rate:.2f} "
                    f"(avg latency {avg_latency:.1f}ms below {self._latency_low_threshold_ms:.1f}ms threshold)"
                )

        # Update burst size proportionally
        self._burst_size = max(int(self._min_adaptive_rate * 2), int(self._rate * 2))

    def get_average_latency_ms(self) -> float:
        """Get average latency in milliseconds.

        Returns:
            Average latency in ms, or 0.0 if no samples
        """
        with self._lock:
            if not self._latency_samples:
                return 0.0
            return sum(self._latency_samples) / len(self._latency_samples)

    def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket, waiting if necessary.

        Supports burst allowance for short bursts above steady-state rate,
        and smooth start for gradual rate ramp-up at startup.

        Args:
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            Time waited in seconds (0.0 if no wait was needed)
        """
        with self._lock:
            # US-156-008: Handle smooth start - ramp up gradually
            if self._smooth_start_active:
                elapsed = time.time() - self._smooth_start_start_time
                if elapsed >= self._smooth_start_duration:
                    # Smooth start complete, use full rate
                    self._smooth_start_active = False
                    self._smooth_start_current_rate = self._rate
                    logger.info(
                        f"Smooth start complete: transitioned from {self._smooth_start_current_rate:.2f} "
                        f"to {self._rate:.2f} RPS over {elapsed:.1f}s"
                    )
                else:
                    # Calculate ramped rate based on elapsed time (linear ramp)
                    progress = elapsed / self._smooth_start_duration
                    self._smooth_start_current_rate = self._rate * (0.1 + 0.9 * progress)

            self._refill()

            if self._tokens >= tokens:
                # Enough tokens available, consume them
                # US-156-008: Track burst usage - tokens beyond burst_size come from allowance
                tokens_from_allowance = max(0, tokens - self._burst_size)
                if tokens_from_allowance > 0:
                    self._burst_usage += int(tokens_from_allowance)

                self._tokens -= tokens
                return 0.0

            # Not enough tokens, calculate wait time
            # US-156-008: All tokens from bucket used, track allowance usage
            tokens_from_allowance = max(0, self._tokens - self._burst_size)
            if tokens_from_allowance > 0:
                self._burst_usage += int(tokens_from_allowance)

            # US-156-008: Warn when burst allowance exhausted
            if self._burst_usage >= self._burst_allowance and self._burst_exhausted_warnings == 0:
                logger.warning(
                    f"YouTube API rate limiter: burst allowance exhausted "
                    f"(used {self._burst_usage}/{self._burst_allowance}). "
                    f"Consider increasing burst_allowance or reducing request rate."
                )
                self._burst_exhausted_warnings += 1

            tokens_needed = tokens - self._tokens
            effective_rate = self._smooth_start_current_rate if self._smooth_start_active else self._rate
            wait_time = tokens_needed / effective_rate

            # Consume all tokens (we'll wait until we have enough)
            self._tokens = 0.0
            self._last_update = time.time()

        # Wait outside the lock to allow other threads to proceed
        if wait_time > 0:
            logger.debug(
                f"YouTube API rate limiter: waiting {wait_time:.3f}s for tokens "
                f"(rate={self._rate}, burst={self._burst_size}, smooth_start={self._smooth_start_active})"
            )
            time.sleep(wait_time)

            # Update metrics (outside the lock)
            with self._lock:
                self._total_waits += 1
                self._total_wait_time += wait_time
                self._max_wait_time = max(self._max_wait_time, wait_time)

            # Refill after waiting
            with self._lock:
                self._refill()
                # Consume the tokens we waited for
                self._tokens = max(0, self._tokens - tokens)

        return wait_time

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics.

        Returns:
            Dictionary with rate limiter statistics
        """
        with self._lock:
            avg_wait = self._total_wait_time / self._total_waits if self._total_waits > 0 else 0.0
            avg_latency = sum(self._latency_samples) / len(self._latency_samples) if self._latency_samples else 0.0

            return {
                "rate": self._rate,
                "burst_size": self._burst_size,
                "current_tokens": round(self._tokens, 2),
                "total_waits": self._total_waits,
                "total_wait_time": round(self._total_wait_time, 3),
                "avg_wait_time": round(avg_wait, 3),
                "max_wait_time": round(self._max_wait_time, 3),
                # US-155-008: Adaptive rate limiting metrics
                "adaptive_enabled": self._adaptive_enabled,
                "avg_latency_ms": round(avg_latency, 2),
                "max_latency_ms": round(self._max_latency_ms, 2),
                "latency_samples": len(self._latency_samples),
                "rate_adjustments": self._rate_adjustments,
                # US-156-008: Burst handling metrics
                "burst_allowance": self._burst_allowance,
                "burst_usage": self._burst_usage,
                "burst_exhausted_warnings": self._burst_exhausted_warnings,
                "smooth_start_enabled": self._smooth_start,
                "smooth_start_active": self._smooth_start_active,
                "current_rate": self.get_current_rate(),
            }

    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self._lock:
            self._tokens = float(self._burst_size + self._burst_allowance)
            self._last_update = time.time()
            self._total_waits = 0
            self._total_wait_time = 0.0
            self._max_wait_time = 0.0
            # Reset adaptive rate limiting state
            self._latency_samples.clear()
            self._total_requests = 0
            self._total_latency_ms = 0.0
            self._max_latency_ms = 0.0
            self._rate_adjustments = 0
            # US-156-008: Reset burst handling state
            self._burst_usage = 0
            self._burst_exhausted_warnings = 0
            self._smooth_start_active = self._smooth_start
            self._smooth_start_start_time = time.time() if self._smooth_start else None
            # Start at 10% of target rate for smooth start ramp-up
            self._smooth_start_current_rate = self._initial_rate * 0.1 if self._smooth_start else self._rate


class YouTubeAPIClient:
    """YouTube Data API v3 client with quota tracking and multi-key support.

    Provides methods for video search, metadata retrieval, and caption
    enumeration with automatic quota management and fallback support.
    Supports multiple API keys for higher quota limits through automatic key rotation.

    Example:
        # Single key
        client = YouTubeAPIClient(api_key="your-key")
        results = client.search_videos("nature documentary", max_results=10)
        for video in results:
            print(f"{video.video_id}: {video.title}")

        # Multiple keys (auto-rotation)
        client = YouTubeAPIClient(api_keys=["key1", "key2", "key3"])
        results = client.search_videos("nature documentary", max_results=10)
    """

    def __init__(
        self,
        api_key: str = "",
        api_keys: Optional[List[str]] = None,
        quota_limit: int = 10000,
        warn_at_percent: int = 80,
        quota_fallback_threshold_percent: int = 10,  # US-150-7: Proactive fallback threshold
        quota_fallback_prediction_minutes: int = 30,  # US-155-3: Trigger fallback when predicted exhaustion < N minutes
        quota_fallback_adaptive_enabled: bool = True,  # US-155-3: Enable adaptive threshold based on time of day
        quota_fallback_peak_multiplier: float = 1.5,  # US-155-3: Peak hours threshold multiplier
        quota_fallback_peak_start_hour: int = 9,  # US-155-3: Peak hours start (9 AM)
        quota_fallback_peak_end_hour: int = 21,  # US-155-3: Peak hours end (9 PM)
        quota_abnormal_rate_warning_enabled: bool = True,  # US-155-3: Enable abnormal rate warning
        quota_abnormal_rate_threshold: float = 2.0,  # US-155-3: Abnormal rate threshold (2x = warn if 2x average)
        quota_exhausted_action: str = "fallback",  # US-152-3: Action when quota exhausted
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: int = 30,
        cache_ttl: int = 3600,
        channel_cache_ttl: int = 604800,  # US-153-6: Default 7 days
        cache_ttl_days: int = 7,  # US-149-9: SQLite cache TTL
        cache_ttl_seconds: int = 0,  # US-158-012: TTL in seconds (0 = use cache_ttl_days)
        auto_invalidate_on_error: bool = True,  # US-156-003: Auto-invalidate cache on stale/empty data
        circuit_breaker_threshold: float = 0.5,
        circuit_breaker_pause: int = 60,
        circuit_breaker_window: int = 10,
        # US-152-8: Rate limiting (requests per second)
        rate_limit_rps: float = 10.0,
        # Project size parameters for auto-scaling quota
        keyword_count: int = 0,
        voiceover_segments: int = 0,
        auto_scale_quota: bool = True,
        # US-153-2: Rotation strategy for multi-key support
        rotation_strategy: str = "sequential",
        # US-153-5: Max concurrent requests for async batch operations
        max_concurrent_requests: int = 5,
        # US-155-5: Batch chunk size for parallel video details fetching
        batch_chunk_size: int = 50,
        # US-155-5: Enable parallel batch operations
        parallel_batch_enabled: bool = True,
        # US-153-9: Rate limit prediction based on time-of-day patterns
        rate_limit_prediction_enabled: bool = True,
        prediction_window_hours: int = 24,
        backoff_multiplier: float = 2.0,
        prediction_sensitivity: float = 0.5,
        # US-154-12: Mock mode for offline testing
        mock_mode: bool = False,
        mock_fixtures_path: Optional[str] = None,
        # US-154-10: Pagination timeout to prevent infinite loops
        pagination_timeout: int = 300,  # 5 minutes default
        # US-155-3: Caption language preference and filtering
        preferred_caption_language: str = "en",
        caption_language_fallback: bool = True,
        caption_language_metrics: bool = True,
        # US-155-010: Per-channel API usage tracking and limits
        per_channel_tracking_enabled: bool = True,
        per_channel_rate_limit: int = 100,
        per_channel_circuit_breaker_enabled: bool = True,
        per_channel_circuit_breaker_threshold: int = 5,
        per_channel_circuit_breaker_pause: float = 60.0,
        # US-155-007: Quota alert webhook notifications
        webhook_enabled: bool = False,
        webhook_urls: Optional[List[str]] = None,
        webhook_timeout: int = 10,
        webhook_retry_count: int = 3,
        # US-155-005: Parallel video details fetching
        parallel_video_details_enabled: bool = True,
        video_details_chunk_size: int = 50,
        video_details_max_workers: int = 5,
        # US-155-008: Adaptive rate limiting based on response latency
        adaptive_rate_limiting_enabled: bool = True,
        latency_high_threshold_ms: float = 500.0,
        latency_low_threshold_ms: float = 200.0,
        rate_decrease_factor: float = 0.8,
        rate_increase_factor: float = 1.1,
        min_adaptive_rate: float = 1.0,
        max_adaptive_rate: float = 20.0,
        latency_smoothing_window: int = 10,
        min_requests_before_adjustment: int = 5,
        # US-156-003: Cache auto-invalidation on stale/empty results
        cache_auto_invalidate_on_error: bool = True,
        cache_auto_invalidate_threshold: int = 3,
        # US-156-005: Search query sanitization and deduplication
        deduplicate_searches: bool = True,
        # US-156-007: Partial failure handling for batch operations
        max_partial_failure_percent: float = 20.0,
        # US-156-011: Maximum pages per query to limit pagination
        max_pages_per_query: int = 10,
        # US-157-011: Results per page for pagination (max 50 for YouTube search API)
        results_per_page: int = 50,
        # US-157-003: Metadata enrichment cache TTL (default 24 hours)
        metadata_enrichment_cache_ttl: int = 86400,
        # US-158-002: Search results ordering (relevance, date, viewCount, rating, videoCount)
        order_by: str = "relevance",
        # US-158-003: Video duration filter (any, short, medium, long)
        video_duration: str = "any",
        # US-158-004: Region code for localized search results (ISO 3166-1 alpha-2)
        region_code: str = "US",
        # US-158-005: Safe search level for family-friendly results
        safe_search: str = "moderate",
        # US-158-006: Batch caption fetching for multiple videos in single call
        caption_batch_size: int = 10,
        # US-158-010: Video quality signals integration for result ranking
        quality_boost_enabled: bool = False,
        quality_view_count_weight: float = 0.7,
        quality_like_count_weight: float = 0.2,
        quality_comment_count_weight: float = 0.1,
    ):
        """Initialize YouTube API client.

        Args:
            api_key: YouTube Data API key from Google Cloud Console (single key)
            api_keys: List of API keys for higher quota limits with auto-rotation
            quota_limit: Daily quota limit per key (default: 10000)
            warn_at_percent: Warn when quota reaches this percentage
            quota_fallback_threshold_percent: Trigger fallback when remaining quota below this %
            quota_exhausted_action: Action when quota exhausted ("fallback" or "pause")
            max_retries: Maximum retry attempts for transient errors
            retry_delay: Initial retry delay in seconds (exponential backoff)
            timeout: Request timeout in seconds
            cache_ttl: Cache TTL in seconds for API responses
            channel_cache_ttl: Cache TTL in seconds for channel metadata (default 7 days / 604800s)
            circuit_breaker_threshold: Failure rate threshold (0.5 = 50%)
            circuit_breaker_pause: Seconds to pause when circuit open
            circuit_breaker_window: Number of recent calls to track
            rate_limit_rps: Maximum requests per second (default: 10.0, YouTube recommended limit)
            keyword_count: Number of keywords to search (for quota auto-scaling)
            voiceover_segments: Number of voiceover segments (for quota auto-scaling)
            auto_scale_quota: Whether to auto-scale quota based on project size
            rotation_strategy: API key rotation strategy ("sequential", "random", "least_used", "weighted", "smart")
            max_concurrent_requests: Max concurrent requests for async batch operations (default: 5)
            batch_chunk_size: Chunk size for splitting batch requests (default: 50, max 50 per API)
            parallel_batch_enabled: Enable parallel execution of batch chunks (default: True)
            rate_limit_prediction_enabled: Enable predictive rate limiting using historical patterns
            prediction_window_hours: Historical window for prediction analysis (default: 24 hours)
            backoff_multiplier: Delay multiplier when high failure rate predicted (default: 2.0)
            prediction_sensitivity: Sensitivity 0.0-1.0 (default: 0.5)
            pagination_timeout: Maximum time in seconds for pagination (default: 300 = 5 minutes)
            mock_mode: Enable mock mode for offline testing (reads from fixtures)
            mock_fixtures_path: Path to mock fixtures directory (defaults to tests/fixtures/youtube_api)
            preferred_caption_language: Preferred caption language for API calls (ISO 639-1)
            caption_language_fallback: Enable fallback chain: preferred -> en -> auto -> any
            caption_language_metrics: Track caption language distribution in metrics
            # US-155-007: Quota alert webhook notifications
            webhook_enabled: bool = False,
            webhook_urls: Optional[List[str]] = None,
            webhook_timeout: int = 10,
            webhook_retry_count: int = 3,
            # US-155-008: Adaptive rate limiting based on response latency
            adaptive_rate_limiting_enabled: Enable adaptive rate limiting based on latency
            latency_high_threshold_ms: Threshold above which to reduce RPS (default: 500ms)
            latency_low_threshold_ms: Threshold below which to increase RPS (default: 200ms)
            rate_decrease_factor: Factor to reduce rate by when latency is high (default: 0.8)
            rate_increase_factor: Factor to increase rate by when latency is low (default: 1.1)
            min_adaptive_rate: Minimum RPS allowed (default: 1.0)
            max_adaptive_rate: Maximum RPS allowed (default: 20.0)
            latency_smoothing_window: Number of samples for latency averaging (default: 10)
            min_requests_before_adjustment: Minimum requests before rate adjustment (default: 5)
            max_pages_per_query: Maximum pages to fetch per query (default: 10, YouTube API limit is ~1000 results)
            # US-158-002: Search results ordering
            order_by: Search results ordering (relevance, date, viewCount, rating, videoCount)
            # US-158-003: Video duration filter
            video_duration: Filter by video duration (any, short, medium, long)
            # US-158-004: Region code for localized search results
            region_code: ISO 3166-1 alpha-2 region code for localized search results
            # US-158-005: Safe search level for family-friendly results
            safe_search: Safe search level (none, moderate, strict)
            # US-158-006: Batch caption fetching configuration
            caption_batch_size: Number of videos to process in a single batch (default: 10)
        """
        # Handle multi-key support
        self._api_keys: List[str] = []
        if api_keys:
            self._api_keys = [k for k in api_keys if k]  # Filter empty keys
        elif api_key:
            self._api_keys = [api_key]

        if not self._api_keys:
            raise ValueError("At least one API key must be provided (api_key or api_keys)")

        # US-153-2: Rotation strategy for multi-key support
        # Options: "sequential" (default), "random", "least_used", "smart"
        valid_strategies = {"sequential", "random", "least_used", "smart", "weighted"}
        if rotation_strategy not in valid_strategies:
            raise ValueError(f"rotation_strategy must be one of {valid_strategies}, got: {rotation_strategy}")
        self._rotation_strategy = rotation_strategy

        # Current active key index
        self._current_key_index: int = 0

        # Per-key quota tracking: maps key_index -> quota_used
        self._key_quota_used: Dict[int, int] = {i: 0 for i in range(len(self._api_keys))}
        self._key_quota_warned: Dict[int, bool] = {i: False for i in range(len(self._api_keys))}
        # US-152-3: Track per-key warning at quota_fallback_threshold_percent
        self._key_fallback_threshold_warned: Dict[int, bool] = {i: False for i in range(len(self._api_keys))}

        # Track exhausted keys
        self._exhausted_keys: set = set()

        # US-155-009: Per-key health status tracking
        # Status values: "healthy", "quota_warning", "exhausted"
        self._key_health_status: Dict[int, str] = {i: "healthy" for i in range(len(self._api_keys))}
        # Track when each key was exhausted (for daily reset)
        self._key_exhausted_at: Dict[int, float] = {}
        # US-155-009: Quota reset time (UTC midnight)
        self._quota_reset_hour: int = 0  # UTC midnight = reset time

        # Per-key error tracking: maps key_index -> {"403": count, "429": count, "other": count}
        # US-150-3: Multi-key API rotation with health monitoring
        self._key_errors: Dict[int, Dict[str, int]] = {
            i: {"403": 0, "429": 0, "other": 0, "total": 0} for i in range(len(self._api_keys))
        }
        # Per-key success tracking for error rate calculation
        self._key_successes: Dict[int, int] = {i: 0 for i in range(len(self._api_keys))}

        # Total quota limit (same for all keys)
        self._total_quota_limit = quota_limit
        self.warn_at_percent = warn_at_percent
        # US-155-007: Quota alert webhook notifications
        self._webhook_enabled = webhook_enabled
        self._webhook_urls = webhook_urls or []
        self._webhook_timeout = webhook_timeout
        self._webhook_retry_count = webhook_retry_count
        # US-149-3: Predictive quota exhaustion warning settings
        self.warn_soon_threshold_percent = 10  # Warn when <10% remaining
        self.warn_soon_threshold_minutes = 30  # Warn when predicted to exhaust within 30 min
        # US-150-7: Proactive fallback threshold - trigger fallback when remaining < this %
        self._proactive_fallback_threshold_percent = quota_fallback_threshold_percent
        # US-155-003: Predictive fallback threshold - trigger fallback when predicted exhaustion < N minutes
        self._quota_fallback_prediction_minutes = quota_fallback_prediction_minutes
        # US-155-003: Adaptive threshold based on time of day
        self._quota_fallback_adaptive_enabled = quota_fallback_adaptive_enabled
        self._quota_fallback_peak_multiplier = quota_fallback_peak_multiplier
        self._quota_fallback_peak_start_hour = quota_fallback_peak_start_hour
        self._quota_fallback_peak_end_hour = quota_fallback_peak_end_hour
        # US-155-003: Abnormal rate warning settings
        self._quota_abnormal_rate_warning_enabled = quota_abnormal_rate_warning_enabled
        self._quota_abnormal_rate_threshold = quota_abnormal_rate_threshold
        # US-152-3: Action when quota exhausted - "fallback" (use yt-dlp) or "pause" (stop)
        self._quota_exhausted_action = quota_exhausted_action
        self._warn_soon_warned: Dict[int, bool] = {i: False for i in range(len(self._api_keys))}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        # US-157-003: Cache for enriched video metadata (24 hours default)
        self._metadata_enrichment_cache: Dict[str, tuple[Any, float]] = {}
        self._metadata_enrichment_cache_ttl = metadata_enrichment_cache_ttl

        # Circuit breaker settings (per-key)
        self._circuit_breaker_threshold = circuit_breaker_threshold
        self._circuit_breaker_pause = circuit_breaker_pause
        self._circuit_breaker_window = circuit_breaker_window
        self._circuit_breaker_open: bool = False
        self._circuit_breaker_open_until: float = 0.0
        self._recent_calls: deque = deque(maxlen=circuit_breaker_window)

        # US-152-4: Per-endpoint circuit breaker (replaces simple rate-based breaker)
        self._endpoint_circuit_breaker = YouTubeAPICircuitBreaker(
            YouTubeAPICircuitBreakerConfig(
                enabled=True,
                consecutive_failures_threshold=5,  # Trip after 5 consecutive failures
                pause_seconds=60,  # 60 second pause
            )
        )

        # US-155-010: Per-channel circuit breaker for tracking API calls per channel
        self._per_channel_tracking_enabled = per_channel_tracking_enabled
        self._per_channel_rate_limit = per_channel_rate_limit
        self._per_channel_circuit_breaker = PerChannelCircuitBreaker(
            PerChannelCircuitBreakerConfig(
                enabled=per_channel_circuit_breaker_enabled,
                consecutive_failures_threshold=per_channel_circuit_breaker_threshold,
                pause_seconds=per_channel_circuit_breaker_pause,
            )
        )

        # US-152-8: Token bucket rate limiter (default: 10 RPS, YouTube recommended limit)
        # US-155-008: Adaptive rate limiting based on response latency
        self._rate_limit_rps = rate_limit_rps
        self._rate_limiter = YouTubeAPIRateLimiter(
            rate=rate_limit_rps,
            burst_size=int(rate_limit_rps * 2),
            adaptive_enabled=adaptive_rate_limiting_enabled,
            latency_high_threshold_ms=latency_high_threshold_ms,
            latency_low_threshold_ms=latency_low_threshold_ms,
            rate_decrease_factor=rate_decrease_factor,
            rate_increase_factor=rate_increase_factor,
            min_adaptive_rate=min_adaptive_rate,
            max_adaptive_rate=max_adaptive_rate,
            latency_smoothing_window=latency_smoothing_window,
            min_requests_before_adjustment=min_requests_before_adjustment,
        )

        # Simple in-memory cache
        self._cache: Dict[str, tuple[Any, float]] = {}

        # Separate cache for channel metadata (longer TTL since it changes infrequently)
        self._channel_cache: Dict[str, tuple[Any, float]] = {}
        self._channel_cache_ttl: int = channel_cache_ttl

        # US-154-9: Intelligent channel metadata cache with adaptive TTL
        # Tracks subscriber count and video count changes for smart invalidation
        self._channel_cache_metadata: Dict[str, Dict[str, Any]] = {}
        # Per-channel cache hit/miss metrics
        self._channel_cache_hits: Dict[str, int] = {}
        self._channel_cache_misses: Dict[str, int] = {}
        # Track channel update frequency for adaptive TTL
        self._channel_update_history: Dict[str, List[float]] = {}  # timestamps of updates

        # Default TTL multipliers by channel activity level
        self._channel_ttl_multipliers = {
            'daily': 0.25,      # 25% of base TTL for daily uploaders
            'weekly': 0.5,      # 50% of base TTL for weekly uploaders
            'monthly': 1.0,     # 100% of base TTL for monthly uploaders
            'rare': 2.0,        # 200% of base TTL for rare uploaders
        }

        # US-149-9: SQLite-based persistent query cache
        self._cache_ttl_days: int = cache_ttl_days
        self._cache_ttl_seconds: int = cache_ttl_seconds  # US-158-012: TTL in seconds
        self._auto_invalidate_on_error: bool = auto_invalidate_on_error  # US-156-003
        self._quota_exhaustion_cache_threshold: int = 1000  # US-158-012: Quota threshold for cache invalidation
        # US-158-012: Pass cache_ttl_seconds to cache (overrides ttl_days if > 0)
        self._query_cache: Optional[YouTubeAPISQLCache] = YouTubeAPISQLCache(
            ttl_days=cache_ttl_days,
            cache_ttl_seconds=cache_ttl_seconds if cache_ttl_seconds > 0 else None
        )

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "voiceover-matcher/1.0"
        })

        # API metrics tracking (US-146-12)
        self._metrics = YouTubeAPIMetrics()
        self._metrics.start_session()  # US-152-7: Start session tracking

        # Quota persistence (US-148-7)
        self._quota_file_path = self._get_quota_file_path()
        self._last_reset_date: Optional[str] = None

        # US-149-3: Quota usage velocity tracking for predictive exhaustion warning
        # Stores (timestamp, quota_used) tuples - tracks usage over time
        self._quota_usage_timestamps: deque = deque(maxlen=100)

        # US-155-003: Quota prediction accuracy tracking
        # Stores predictions for later accuracy calculation
        self._quota_predictions: deque = deque(maxlen=50)

        # Check if CLI flag requested reset (US-148-7)
        # Import here to avoid circular imports
        cli_reset_requested = get_youtube_quota_reset_requested()
        self._force_reset = cli_reset_requested  # Set to True to force reset via CLI

        # Load persisted quota and check for daily reset
        self._load_quota()
        self._check_daily_reset()

        # Auto-scale quota based on project size (US-148-9)
        if auto_scale_quota:
            self._auto_scale_quota(keyword_count, voiceover_segments)

        # US-149-4: Retry budget to prevent infinite retry loops
        self._retry_budget: YouTubeAPIRetryBudget = YouTubeAPIRetryBudget()

        # US-149-8: Async batch enrichment with parallel requests
        # Semaphore for concurrency limiting (controls max parallel requests)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        # Thread pool executor for running sync requests in async context
        self._executor: Optional[ThreadPoolExecutor] = None
        # Max concurrent requests for async operations (default: 5, US-153-5)
        self._max_concurrent_requests: int = max_concurrent_requests
        # US-155-5: Batch chunk size and parallel execution
        self._batch_chunk_size: int = min(batch_chunk_size, 50)  # YouTube API max is 50
        self._parallel_batch_enabled: bool = parallel_batch_enabled

        # US-155-005: Parallel video details fetching
        self._parallel_video_details_enabled: bool = parallel_video_details_enabled
        self._video_details_chunk_size: int = min(video_details_chunk_size, 50)  # YouTube API max is 50
        self._video_details_max_workers: int = video_details_max_workers

        # US-153-9: Rate limit prediction based on time-of-day patterns
        self._rate_limit_prediction_enabled = rate_limit_prediction_enabled
        self._prediction_window_hours = prediction_window_hours
        self._backoff_multiplier = backoff_multiplier
        self._prediction_sensitivity = prediction_sensitivity

        # US-154-10: Pagination timeout to prevent infinite loops
        self._pagination_timeout = pagination_timeout

        # Initialize the rate limit predictor
        self._predictor: Optional[RateLimitPredictor] = None
        if rate_limit_prediction_enabled:
            self._predictor = RateLimitPredictor(
                max_history_days=max(1, prediction_window_hours // 24),
                sensitivity=prediction_sensitivity
            )
            logger.info(
                f"Rate limit prediction enabled: window={prediction_window_hours}h, "
                f"backoff_multiplier={backoff_multiplier}, sensitivity={prediction_sensitivity}"
            )

        # US-153-9: Metrics for predicted vs actual rate limits
        self._predicted_rate_limits: int = 0
        self._actual_rate_limits: int = 0
        # US-157-9: Track unnecessary backoffs (predicted but didn't occur)
        self._unnecessary_backoffs: int = 0
        # Track whether a backoff was applied in the last request
        self._last_backoff_applied: bool = False

        # US-154-12: Mock mode for offline testing
        self._mock_mode = mock_mode
        if mock_fixtures_path:
            self._mock_fixtures_path = mock_fixtures_path
        else:
            # Default to tests/fixtures/youtube_api
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self._mock_fixtures_path = os.path.join(project_root, "tests", "fixtures", "youtube_api")

        if mock_mode:
            self._mock_responses: Dict[str, Any] = {}
            self._load_mock_fixtures()
            logger.info(f"YouTubeAPIClient initialized in MOCK MODE, fixtures from: {self._mock_fixtures_path}")

        # US-155-3: Caption language preference and filtering
        self._preferred_caption_language = preferred_caption_language
        self._caption_language_fallback = caption_language_fallback
        self._caption_language_metrics = caption_language_metrics

        # Track caption language distribution for metrics
        self._caption_language_counts: Dict[str, int] = {}

        # US-155-008: In-session request deduplication cache
        # Maps request hash -> cached results (cleared on pipeline start)
        self._deduplication_cache: Dict[str, List[VideoSearchResult]] = {}

        # US-156-005: Search query sanitization and deduplication
        # Track recent normalized queries to avoid redundant API calls
        self._deduplicate_searches_enabled = deduplicate_searches
        self._recent_queries: Set[str] = set()

        # US-156-007: Partial failure handling for batch operations
        self._max_partial_failure_percent: float = max_partial_failure_percent

        # US-156-011: Pagination state management
        # max_pages_per_query: Maximum pages to fetch per query (default: 10)
        self._max_pages_per_query: int = max_pages_per_query
        # US-157-011: Results per page for pagination (max 50 for YouTube search API)
        self._results_per_page: int = min(results_per_page, 50)  # Enforce max 50
        # US-158-002: Search results ordering
        self._order_by: str = order_by
        # Validate order_by value
        valid_orders = ("relevance", "date", "viewCount", "rating", "videoCount")
        if self._order_by not in valid_orders:
            logger.warning(f"US-158-002: Invalid order_by '{order_by}', falling back to 'relevance'")
            self._order_by = "relevance"
        else:
            logger.info(f"US-158-002: Search ordering set to '{self._order_by}'")
        # US-158-003: Video duration filter
        self._video_duration: str = video_duration
        # Validate video_duration value
        valid_durations = ("any", "short", "medium", "long")
        if self._video_duration not in valid_durations:
            logger.warning(f"US-158-003: Invalid video_duration '{video_duration}', falling back to 'any'")
            self._video_duration = "any"
        else:
            logger.info(f"US-158-003: Video duration filter set to '{self._video_duration}'")
        # US-158-004: Region code for localized search results
        self._region_code: str = region_code.upper() if region_code else ""
        if self._region_code and len(self._region_code) != 2:
            logger.warning(f"US-158-004: Invalid region_code '{region_code}', must be 2-letter ISO code. Using empty (no filter).")
            self._region_code = ""
        elif self._region_code:
            logger.info(f"US-158-004: Region code set to '{self._region_code}' for localized search results")
        # US-158-005: Safe search level for family-friendly results
        self._safe_search: str = safe_search.lower() if safe_search else "moderate"
        valid_safe_search = ("none", "moderate", "strict")
        if self._safe_search not in valid_safe_search:
            logger.warning(f"US-158-005: Invalid safe_search '{safe_search}', falling back to 'moderate'")
            self._safe_search = "moderate"
        else:
            logger.info(f"US-158-005: Safe search level set to '{self._safe_search}' for family-friendly results")
        # US-158-006: Batch caption fetching configuration
        self._caption_batch_size: int = max(1, caption_batch_size)  # Ensure at least 1
        if caption_batch_size != self._caption_batch_size:
            logger.warning(f"US-158-006: caption_batch_size must be >= 1, got {caption_batch_size}, using {self._caption_batch_size}")

        # US-158-010: Video quality signals integration for result ranking
        self._quality_boost_enabled: bool = quality_boost_enabled
        self._quality_view_count_weight: float = quality_view_count_weight
        self._quality_like_count_weight: float = quality_like_count_weight
        self._quality_comment_count_weight: float = quality_comment_count_weight

        # Track pagination state: query -> {page_token -> results_count}
        self._pagination_state: Dict[str, Dict[str, int]] = {}
        # Page token cache to resume interrupted pagination: query -> list of (page_token, results_count)
        self._page_token_cache: Dict[str, List[Tuple[str, int]]] = {}

        # US-155-010: Per-channel API usage tracking and rate limiting
        # Track API calls per channel to avoid rate-limiting specific channels
        self._per_channel_tracking_enabled: bool = False
        self._per_channel_rate_limit: int = 100
        self._per_channel_circuit_breaker_enabled: bool = False
        self._per_channel_circuit_breaker_threshold: int = 5
        self._per_channel_circuit_breaker_pause: float = 60.0
        self._per_channel_graceful_no_videos: bool = True

        # Per-channel API call tracking: channel_id -> call_count
        self._channel_api_calls: Dict[str, int] = {}
        # Per-channel circuit breaker state: channel_id -> (is_open, opened_at, consecutive_failures)
        self._channel_circuit_state: Dict[str, Dict[str, Any]] = {}
        # Channels with no published videos tracked for graceful handling
        self._channels_no_videos: Set[str] = set()

        # US-155-010: Channel query distribution metrics
        # Track distribution of queries across channels for metrics
        self._channel_query_counts: Dict[str, int] = {}  # channel_id -> number of queries
        self._channel_video_counts: Dict[str, int] = {}  # channel_id -> number of videos returned

        logger.info(f"YouTubeAPIClient initialized with {len(self._api_keys)} API keys")

        # US-156-005: Query sanitization and deduplication
        # Track recent queries to avoid duplicate API calls within a session
        self._recent_queries: Set[str] = set()
        # US-156-005: Config for deduplication (passed from VideoSearchConfig)
        self._deduplicate_searches: bool = True

    # ==================== US-156-005: Query Sanitization Methods ====================

    def sanitize_query(self, query: str) -> str:
        """Sanitize a search query by cleaning whitespace and special characters.

        Removes leading/trailing whitespace, collapses multiple spaces into single space,
        and removes potentially problematic special characters while preserving
        meaningful punctuation.

        Args:
            query: Raw search query string

        Returns:
            Sanitized query string suitable for API calls
        """
        if not query:
            return ""

        # Strip leading/trailing whitespace
        query = query.strip()

        # Collapse multiple whitespace characters into single space
        query = ' '.join(query.split())

        # Remove control characters and non-printable characters (keep visible ASCII)
        # Keep alphanumeric, spaces, and common punctuation/symbols
        query = ''.join(char for char in query if ord(char) >= 33 or char == ' ')

        # Collapse any resulting multiple spaces again
        query = ' '.join(query.split())

        return query

    def query_normalize(self, query: str) -> str:
        """Normalize a query for consistent comparison.

        Converts to lowercase, removes extra whitespace, and handles
        common variations to ensure consistent query formatting.

        Args:
            query: Query string to normalize

        Returns:
            Normalized query string
        """
        if not query:
            return ""

        # Start with sanitization
        query = self.sanitize_query(query)

        # Convert to lowercase for consistent comparison
        query = query.lower()

        # Remove common variations that don't affect search results
        # Remove extra whitespace again after lowercasing
        query = ' '.join(query.split())

        return query

    def deduplicate_queries(self, queries: List[str]) -> List[str]:
        """Remove duplicate queries from a list while preserving order.

        Uses query_normalize for comparison to catch duplicates that
        differ only in case or whitespace.

        Args:
            queries: List of query strings (may contain duplicates)

        Returns:
            Deduplicated list of queries preserving original order
        """
        if not queries:
            return []

        seen_normalized: Set[str] = set()
        result: List[str] = []

        for query in queries:
            if not query:
                continue

            normalized = self.query_normalize(query)
            if normalized and normalized not in seen_normalized:
                seen_normalized.add(normalized)
                result.append(query)

        return result

    def is_query_recent(self, query: str) -> bool:
        """Check if a query was recently executed in this session.

        Args:
            query: Query string to check

        Returns:
            True if the normalized query was recently executed
        """
        if not query or not self._deduplicate_searches:
            return False

        normalized = self.query_normalize(query)
        return normalized in self._recent_queries

    def mark_query_executed(self, query: str) -> None:
        """Mark a query as executed in this session.

        Args:
            query: Query string that was executed
        """
        if not query or not self._deduplicate_searches:
            return

        normalized = self.query_normalize(query)
        if normalized:
            self._recent_queries.add(normalized)

    def clear_recent_queries(self) -> None:
        """Clear the recent queries cache.

        Call this at the start of a new pipeline run to reset deduplication.
        """
        self._recent_queries.clear()

    # ==================== US-156-011: Pagination State Management ====================

    def _cache_page_token(self, query: str, page_token: str, results_count: int) -> None:
        """Cache page token for resuming interrupted pagination (US-156-011).

        Args:
            query: The search query
            page_token: The next page token to resume from
            results_count: Number of results collected so far
        """
        # Normalize query for consistent cache key
        normalized_query = self.query_normalize(query)
        if normalized_query not in self._page_token_cache:
            self._page_token_cache[normalized_query] = []
        # Add page token with results count
        self._page_token_cache[normalized_query].append((page_token, results_count))
        logger.debug(
            f"US-156-011: Cached page token for query '{query[:30]}...': "
            f"token={page_token[:20]}..., results_so_far={results_count}"
        )

    def get_cached_page_token(self, query: str) -> Optional[Tuple[str, int]]:
        """Get cached page token to resume interrupted pagination (US-156-011).

        Args:
            query: The search query

        Returns:
            Tuple of (page_token, results_count) if cached, None otherwise
        """
        normalized_query = self.query_normalize(query)
        if normalized_query in self._page_token_cache and self._page_token_cache[normalized_query]:
            cached = self._page_token_cache[normalized_query][-1]
            logger.info(
                f"US-156-011: Resuming pagination for '{query[:30]}...' "
                f"from token={cached[0][:20]}... with {cached[1]} results already fetched"
            )
            return cached
        return None

    def clear_page_token_cache(self, query: Optional[str] = None) -> None:
        """Clear page token cache (US-156-011).

        Args:
            query: Specific query to clear, or None to clear all
        """
        if query:
            normalized_query = self.query_normalize(query)
            if normalized_query in self._page_token_cache:
                del self._page_token_cache[normalized_query]
                logger.debug(f"US-156-011: Cleared page token cache for query '{query[:30]}...'")
        else:
            self._page_token_cache.clear()
            logger.debug("US-156-011: Cleared all page token caches")

    def track_pagination_state(self, query: str, page_token: str, results_count: int) -> None:
        """Track pagination state for a query (US-156-011).

        This helps detect duplicate page tokens and other API issues.

        Args:
            query: The search query
            page_token: The page token received from API
            results_count: Number of results for this page
        """
        normalized_query = self.query_normalize(query)
        if normalized_query not in self._pagination_state:
            self._pagination_state[normalized_query] = {}

        # Track this page token
        if page_token in self._pagination_state[normalized_query]:
            # Duplicate detected - this indicates an API bug
            logger.warning(
                f"US-156-011: Duplicate page token detected for query '{query[:30]}...': "
                f"token={page_token[:20]}..., previous_results={self._pagination_state[normalized_query][page_token]}, "
                f"new_results={results_count}"
            )
            self._metrics.duplicate_page_tokens += 1
        else:
            self._pagination_state[normalized_query][page_token] = results_count

    def get_pagination_state(self, query: str) -> Dict[str, int]:
        """Get pagination state for a query (US-156-011).

        Args:
            query: The search query

        Returns:
            Dictionary mapping page_token -> results_count
        """
        normalized_query = self.query_normalize(query)
        return self._pagination_state.get(normalized_query, {}).copy()

    def clear_pagination_state(self, query: Optional[str] = None) -> None:
        """Clear pagination state (US-156-011).

        Args:
            query: Specific query to clear, or None to clear all
        """
        if query:
            normalized_query = self.query_normalize(query)
            if normalized_query in self._pagination_state:
                del self._pagination_state[normalized_query]
        else:
            self._pagination_state.clear()

    def _get_deduplication_hash(
        self,
        query: str,
        max_results: int = 50,
        published_after: str = "",
        published_before: str = "",
        video_category_id: str = "",
    ) -> str:
        """Generate a hash for request deduplication.

        Args:
            query: Search query string
            max_results: Maximum results requested
            published_after: Date filter
            published_before: Date filter
            video_category_id: Category filter

        Returns:
            A hash string that uniquely identifies this request combination
        """
        # Create a stable string representation of all parameters
        params_str = "|".join([
            query,
            str(max_results),
            published_after or "",
            published_before or "",
            video_category_id or "",
        ])
        import hashlib
        return hashlib.md5(params_str.encode('utf-8')).hexdigest()

    def clear_deduplication_cache(self) -> None:
        """Clear the in-session deduplication cache.

        Should be called at pipeline start to ensure fresh session.
        """
        self._deduplication_cache.clear()
        self._recent_queries.clear()
        logger.debug("Deduplication cache and recent queries cleared")

    # US-156-005: Query sanitization and normalization
    def query_normalize(self, query: str) -> str:
        """Normalize a search query for consistent formatting.

        Converts to lowercase, strips whitespace, and removes extra internal spaces.

        Args:
            query: Raw search query string

        Returns:
            Normalized query string
        """
        if not query:
            return ""
        # Strip leading/trailing whitespace and normalize internal whitespace
        normalized = " ".join(query.lower().strip().split())
        return normalized

    def sanitize_query(self, query: str) -> str:
        """Clean a search query by removing special characters and extra whitespace.

        Removes characters that might cause issues with the YouTube API or reduce
        search effectiveness (e.g., extra punctuation, non-alphanumeric chars).

        Args:
            query: Raw search query string

        Returns:
            Sanitized query string
        """
        if not query:
            return ""
        import re
        # First normalize whitespace
        normalized = " ".join(query.split())
        # Remove special characters that could cause issues
        # Keep alphanumeric, spaces, and common punctuation that helps search
        # Remove control characters and excessive punctuation
        sanitized = re.sub(r'[^\w\s\-:,\.]', '', normalized)
        # Collapse multiple spaces to single space
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized

    def deduplicate_queries(self, queries: List[str]) -> List[str]:
        """Remove duplicate search queries after normalization.

        Uses query_normalize to convert queries to a consistent format before
        checking for duplicates. Also checks against recent queries to avoid
        redundant API calls within a session.

        Args:
            queries: List of raw search query strings

        Returns:
            Deduplicated list of query strings (original casing preserved for first occurrence)
        """
        if not queries:
            return []
        seen_normalized: Set[str] = set()
        unique_queries: List[str] = []
        for query in queries:
            if not query:
                continue
            normalized = self.query_normalize(query)
            # Skip if we've seen this normalized query already
            if normalized in seen_normalized:
                logger.debug(f"US-156-005: Skipping duplicate query: {query}")
                continue
            # Skip if we've recently searched this query
            if self._deduplicate_searches_enabled and normalized in self._recent_queries:
                logger.debug(f"US-156-005: Skipping recent query: {query}")
                continue
            seen_normalized.add(normalized)
            # Also add to recent queries
            if self._deduplicate_searches_enabled:
                self._recent_queries.add(normalized)
            unique_queries.append(query)
        if len(unique_queries) < len(queries):
            logger.info(f"US-156-005: Deduplicated {len(queries) - len(unique_queries)} queries from {len(queries)} total")
        return unique_queries

    # US-155-010: Per-channel tracking state initialization
    def configure_per_channel_tracking(self, enabled: bool = False, rate_limit: int = 100,
            circuit_breaker_enabled: bool = False, circuit_breaker_threshold: int = 5,
            circuit_breaker_pause: float = 60.0, graceful_no_videos: bool = True) -> None:
        """Configure per-channel API tracking and rate limiting."""
        self._per_channel_tracking_enabled = enabled
        self._per_channel_rate_limit = rate_limit
        self._per_channel_circuit_breaker_enabled = circuit_breaker_enabled
        self._per_channel_circuit_breaker_threshold = circuit_breaker_threshold
        self._per_channel_circuit_breaker_pause = circuit_breaker_pause
        self._per_channel_graceful_no_videos = graceful_no_videos

    def track_channel_api_call(self, channel_id: str) -> bool:
        """Track an API call for a specific channel. Returns True if allowed."""
        if not self._per_channel_tracking_enabled or not channel_id:
            return True
        current = self._channel_api_calls.get(channel_id, 0)
        if current >= self._per_channel_rate_limit:
            return False
        self._channel_api_calls[channel_id] = current + 1
        if channel_id not in self._channel_circuit_state:
            self._channel_circuit_state[channel_id] = {'is_open': False, 'opened_at': None, 'consecutive_failures': 0}
        return True

    def check_channel_circuit_breaker(self, channel_id: str) -> bool:
        """Check if channel's circuit breaker allows API calls."""
        if not self._per_channel_tracking_enabled or not self._per_channel_circuit_breaker_enabled or not channel_id:
            return True
        if channel_id not in self._channel_circuit_state:
            self._channel_circuit_state[channel_id] = {'is_open': False, 'opened_at': None, 'consecutive_failures': 0}
        state = self._channel_circuit_state[channel_id]
        if state['is_open'] and state['opened_at']:
            elapsed = time.time() - state['opened_at']
            remaining = self._per_channel_circuit_breaker_pause - elapsed
            if remaining > 0:
                time.sleep(remaining)
            state['is_open'] = False
        return True

    def record_channel_success(self, channel_id: str) -> None:
        """Record a successful API call for a channel."""
        if not self._per_channel_tracking_enabled or not channel_id:
            return
        if channel_id in self._channel_circuit_state:
            state = self._channel_circuit_state[channel_id]
            state['consecutive_failures'] = 0
            state['is_open'] = False

    def record_channel_failure(self, channel_id: str) -> bool:
        """Record a failed API call for a channel. Returns True if circuit breaker trips."""
        if not self._per_channel_tracking_enabled or not self._per_channel_circuit_breaker_enabled or not channel_id:
            return False
        if channel_id not in self._channel_circuit_state:
            self._channel_circuit_state[channel_id] = {'is_open': False, 'opened_at': None, 'consecutive_failures': 0}
        state = self._channel_circuit_state[channel_id]
        state['consecutive_failures'] += 1
        if state['consecutive_failures'] >= self._per_channel_circuit_breaker_threshold:
            state['is_open'] = True
            state['opened_at'] = time.time()
            return True
        return False

    def mark_channel_no_videos(self, channel_id: str) -> None:
        """Mark a channel as having no published videos."""
        if self._per_channel_graceful_no_videos and channel_id:
            self._channels_no_videos.add(channel_id)

    def is_channel_marked_no_videos(self, channel_id: str) -> bool:
        """Check if a channel is known to have no published videos."""
        if not self._per_channel_graceful_no_videos:
            return False
        return channel_id in self._channels_no_videos

    def get_channel_metrics(self) -> Dict[str, Any]:
        """Get per-channel usage metrics."""
        if not self._per_channel_tracking_enabled:
            return {'enabled': False}
        top_channels = sorted(self._channel_api_calls.items(), key=lambda x: x[1], reverse=True)[:20]
        open_circuits = [ch_id for ch_id, state in self._channel_circuit_state.items() if state.get('is_open', False)]
        return {'enabled': True, 'total_channels_queried': len(self._channel_api_calls),
            'total_api_calls': sum(self._channel_api_calls.values()),
            'top_channels_by_api_calls': dict(top_channels),
            'channels_with_open_circuit_breaker': open_circuits,
            'channels_no_videos': list(self._channels_no_videos)}

    def clear_channel_tracking(self) -> None:
        """Clear all per-channel tracking state."""
        self._channel_api_calls.clear()
        self._channel_circuit_state.clear()
        self._channels_no_videos.clear()
        logger.debug("US-155-010: Channel tracking state cleared")

    def _load_mock_fixtures(self) -> None:
        """Load mock API response fixtures from the fixtures directory.

        Reads JSON files from the fixtures directory:
        - search_{query}.json: Search results for a query
        - video_{video_id}.json: Video metadata
        - channel_{channel_id}.json: Channel metadata
        - captions_{video_id}.json: Caption data
        - error_{code}.json: Error responses

        The fixture loader stores responses keyed by type and query/video_id.
        """
        import os
        import json

        if not os.path.exists(self._mock_fixtures_path):
            logger.warning(f"Mock fixtures path does not exist: {self._mock_fixtures_path}")
            return

        # Load all JSON files in the fixtures directory
        for filename in os.listdir(self._mock_fixtures_path):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self._mock_fixtures_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Parse filename to determine type and key
                # Format: {type}_{key}.json (e.g., search_nature.json, video_abc123.json)
                base_name = filename[:-5]  # Remove .json
                if '_' in base_name:
                    parts = base_name.split('_', 1)
                    if len(parts) == 2:
                        fixture_type, key = parts
                        self._mock_responses[f"{fixture_type}:{key}"] = data
                        logger.debug(f"Loaded mock fixture: {fixture_type}:{key}")
            except Exception as e:
                logger.warning(f"Failed to load mock fixture {filename}: {e}")

    def _get_mock_response(self, fixture_type: str, key: str) -> Optional[Any]:
        """Get a mock response for a given type and key.

        Args:
            fixture_type: Type of fixture (search, video, channel, captions, error)
            key: The key to look up (query, video_id, channel_id, etc.)

        Returns:
            Mock response data if found, None otherwise
        """
        if not self._mock_mode:
            return None

        lookup_key = f"{fixture_type}:{key}"
        return self._mock_responses.get(lookup_key)

    def _get_mock_search_results(self, query: str, max_results: int, video_category_id: str = "") -> List[VideoSearchResult]:
        """Get mock search results for a query.

        Args:
            query: Search query
            max_results: Maximum number of results
            video_category_id: Video category ID filter

        Returns:
            List of VideoSearchResult objects
        """
        from .youtube_api_fixtures import (
            create_mock_youtube_search_result,
            create_mock_search_list_response,
            youtube_search_to_video_search_result,
        )

        # Try to load from fixtures first
        mock_data = self._get_mock_response("search", query)

        if mock_data is None:
            # Generate mock data using the fixture factory
            logger.debug(f"Generating mock search results for query: {query}")
            items = mock_data.get("items") if mock_data else None
            if items:
                results = []
                for item in items[:max_results]:
                    result = youtube_search_to_video_search_result(item, keyword=query)
                    results.append(VideoSearchResult(**result))
                return results
            else:
                # Use fixture factory to generate results
                search_response = create_mock_search_list_response(
                    results=[
                        create_mock_youtube_search_result(
                            video_id=f"mock_{i:03d}",
                            title=f"Mock Video {i} for {query}",
                            description=f"This is a mock video result for the search query: {query}",
                        )
                        for i in range(min(max_results, 5))
                    ],
                    total_results=max_results,
                )
                results = []
                for item in search_response.get("items", []):
                    result = youtube_search_to_video_search_result(item, keyword=query)
                    results.append(VideoSearchResult(**result))
                return results

        # Process loaded mock data
        results = []
        items = mock_data.get("items", [])
        for item in items[:max_results]:
            result = youtube_search_to_video_search_result(item, keyword=query)
            results.append(VideoSearchResult(**result))
        return results

    def _get_mock_video_details(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get mock video details for video IDs.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            Dict mapping video_id to video metadata
        """
        from .youtube_api_fixtures import create_mock_youtube_video_metadata

        result = {}
        for video_id in video_ids:
            mock_data = self._get_mock_response("video", video_id)
            if mock_data:
                result[video_id] = mock_data
            else:
                # Generate mock video metadata
                result[video_id] = create_mock_youtube_video_metadata(video_id=video_id)
        return result

    def _get_mock_channel_metadata(self, channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get mock channel metadata.

        Args:
            channel_ids: List of YouTube channel IDs

        Returns:
            Dict mapping channel_id to channel metadata
        """
        from .youtube_api_fixtures import create_mock_youtube_channel

        result = {}
        for channel_id in channel_ids:
            mock_data = self._get_mock_response("channel", channel_id)
            if mock_data:
                result[channel_id] = mock_data
            else:
                # Generate mock channel metadata
                result[channel_id] = create_mock_youtube_channel(channel_id=channel_id)
        return result

    def _get_mock_captions(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get mock caption data for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Mock caption data or None
        """
        from .youtube_api_fixtures import create_mock_caption_response

        mock_data = self._get_mock_response("captions", video_id)
        if mock_data:
            return mock_data

        # Generate mock caption data
        return create_mock_caption_response(video_id=video_id)

    def is_mock_mode(self) -> bool:
        """Check if client is in mock mode.

        Returns:
            True if mock mode is enabled
        """
        return self._mock_mode

    def configure_retry_budget(
        self,
        retry_budget_config: Optional[YouTubeAPIRetryBudgetConfig] = None,
        batch_size: int = 0,
    ) -> None:
        """Configure retry budget from config and optionally set batch size.

        Args:
            retry_budget_config: Configuration for retry budget
            batch_size: Number of videos/items for auto-scaling
        """
        if retry_budget_config:
            self._retry_budget = YouTubeAPIRetryBudget.from_config(retry_budget_config)

        if batch_size > 0:
            self._retry_budget.set_batch_size(batch_size)

    def set_retry_budget_batch_size(self, batch_size: int) -> None:
        """Set batch size for retry budget auto-scaling.

        Args:
            batch_size: Number of videos/items in the batch
        """
        self._retry_budget.set_batch_size(batch_size)

    def is_retry_budget_exhausted(self) -> bool:
        """Check if retry budget is exhausted.

        Returns:
            True if budget is exhausted, False otherwise
        """
        return self._retry_budget.budget_exhausted()

    def get_retry_budget_stats(self) -> Dict[str, Any]:
        """Get current retry budget statistics.

        Returns:
            Dict with budget statistics
        """
        return self._retry_budget.get_stats()

    # US-149-9: Query cache methods
    def enable_query_cache(self, cache_dir: Optional[str] = None) -> None:
        """Enable the SQLite query cache for search results.

        Args:
            cache_dir: Optional custom directory for cache storage
        """
        if self._query_cache is not None:
            logger.debug("Query cache already enabled")
            return

        from pathlib import Path
        cache_path = Path(cache_dir) if cache_dir else None
        self._query_cache = YouTubeAPISQLCache(
            cache_dir=cache_path,
            ttl_days=self._cache_ttl_days
        )
        logger.info(f"Query cache enabled with TTL={self._cache_ttl_days} days")

    def disable_query_cache(self) -> None:
        """Disable the SQLite query cache."""
        self._query_cache = None
        logger.info("Query cache disabled")

    def invalidate_query_cache(self) -> int:
        """Invalidate all cache entries (e.g., when quota resets).

        Returns:
            Number of entries invalidated
        """
        if self._query_cache is None:
            return 0
        return self._query_cache.invalidate_all()

    def get_query_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics.

        Returns:
            Dict with cache hits, misses, hit_rate, entries, etc.
        """
        if self._query_cache is None:
            return {'enabled': False}
        return self._query_cache.get_stats()

    def _auto_scale_quota(self, keyword_count: int, voiceover_segments: int) -> None:
        """Auto-scale quota based on project size (US-148-9).

        Adjusts quota_limit and warn_at_percent based on the expected API usage
        for the project. This ensures larger projects have sufficient quota
        while preventing runaway costs on small projects.

        Scaling formula:
        - Base quota: 10000 units (default Google Cloud free tier)
        - Per keyword: ~100 units (search cost) + ~50 units (caption check)
        - Per segment: ~50 units additional for caption fetching
        - Uses max of keyword-based or segment-based estimate

        Args:
            keyword_count: Number of search keywords
            voiceover_segments: Number of voiceover segments
        """
        if keyword_count == 0 and voiceover_segments == 0:
            logger.debug("No project size info provided, using default quota")
            return

        # Constants for scaling calculation
        QUOTA_PER_KEYWORD = 150  # 100 (search) + 50 (caption check)
        QUOTA_PER_SEGMENT = 50   # Additional for caption fetching
        MIN_QUOTA_FLOOR = 1000   # Minimum quota for small projects
        MAX_QUOTA_CEILING = 100000  # Maximum quota to prevent runaway
        SCALE_MULTIPLIER = 1.5   # Safety margin

        # Calculate estimated quota needs
        keyword_estimate = keyword_count * QUOTA_PER_KEYWORD * SCALE_MULTIPLIER
        segment_estimate = voiceover_segments * QUOTA_PER_SEGMENT * SCALE_MULTIPLIER

        # Use the larger estimate
        estimated_quota = max(keyword_estimate, segment_estimate)

        # Apply floor and ceiling
        scaled_quota = max(MIN_QUOTA_FLOOR, min(estimated_quota, MAX_QUOTA_CEILING))

        # Round to nearest 100 for cleaner numbers
        scaled_quota = round(scaled_quota / 100) * 100

        # Adjust warn_at_percent proportionally (keep same absolute warning threshold)
        # Original: quota_limit * 0.80 = warning at 8000
        # Scaled: scaled_quota * warn_ratio should equal same absolute threshold
        original_warning_threshold = self._total_quota_limit * (self.warn_at_percent / 100)
        new_warn_percent = min(95, max(50, round(original_warning_threshold / scaled_quota * 100)))

        # Log scaling decision
        if scaled_quota != self._total_quota_limit:
            logger.info(
                f"YouTube API quota auto-scaled: {self._total_quota_limit} -> {scaled_quota} "
                f"(keywords={keyword_count}, segments={voiceover_segments}, "
                f"warn_at_percent: {self.warn_at_percent} -> {new_warn_percent})"
            )
        else:
            logger.debug(
                f"YouTube API quota unchanged at {scaled_quota} "
                f"(keywords={keyword_count}, segments={voiceover_segments})"
            )

        # Apply the scaled values
        self._total_quota_limit = scaled_quota
        self.warn_at_percent = new_warn_percent

    @property
    def metrics(self) -> YouTubeAPIMetrics:
        """Get the API metrics object."""
        return self._metrics

    def get_pagination_stats(self) -> Dict[str, Any]:
        """Get pagination statistics including state tracking info (US-156-011).

        Returns:
            Dictionary with pagination metrics:
            - search_pages_total: Total pages fetched across all searches
            - search_queries_with_pagination: Number of searches requiring pagination
            - average_pages_per_search: Average pages per search (0 if no paginated searches)
            - search_page_timeouts: Number of pagination timeouts
            - duplicate_page_tokens: Number of duplicate pageToken occurrences
            - cached_queries_with_page_tokens: Number of queries with cached page tokens
        """
        # Get base stats from metrics
        base_stats = self._metrics.get_pagination_stats()
        # Add pagination state tracking info
        base_stats["cached_queries_with_page_tokens"] = len(self._page_token_cache)
        base_stats["tracked_queries"] = len(self._pagination_state)
        return base_stats

    @property
    def api_key(self) -> str:
        """Get the current API key being used (alias for current_api_key)."""
        return self.current_api_key

    @property
    def current_api_key(self) -> str:
        """Get the current API key being used."""
        return self._api_keys[self._current_key_index]

    @property
    def quota_used(self) -> int:
        """Get current quota usage for active key."""
        return self._key_quota_used[self._current_key_index]

    @property
    def quota_remaining(self) -> int:
        """Get remaining quota for active key."""
        return max(0, self._total_quota_limit - self._key_quota_used[self._current_key_index])

    def invalidate_cache_on_quota_exhaustion(self, threshold: Optional[int] = None) -> int:
        """US-158-012: Invalidate cache when quota is near exhaustion.

        Args:
            threshold: Custom quota threshold (uses instance default if not provided)

        Returns:
            Number of cache entries invalidated
        """
        if self._query_cache is None:
            return 0

        effective_threshold = threshold if threshold is not None else self._quota_exhaustion_cache_threshold
        return self._query_cache.invalidate_on_quota_exhaustion(
            self.quota_remaining,
            effective_threshold
        )

    def get_remaining_quota(self) -> int:
        """Get remaining quota units available across all keys.

        Returns:
            Number of quota units remaining before hitting limit
        """
        # Return remaining quota from current key
        return self.quota_remaining

    def get_quota_remaining(self) -> int:
        """Get remaining quota for active key (US-152-3).

        Returns:
            Number of quota units remaining for current key
        """
        return self.quota_remaining

    def get_least_used_key(self) -> Optional[int]:
        """Get the key index with the lowest current quota usage.

        This method selects the API key that has the least amount of quota used,
        making it ideal for distributing load evenly across keys.

        Returns:
            Key index with lowest quota usage, or None if no keys available
        """
        if not self._api_keys:
            return None

        # Check for daily quota reset before querying
        self._check_quota_reset()

        # Filter to available (non-exhausted) keys
        available_keys = [i for i in range(len(self._api_keys)) if i not in self._exhausted_keys]

        if not available_keys:
            return None

        # Find key with minimum quota usage
        min_quota = float('inf')
        least_used_key = None

        for key_idx in available_keys:
            quota_used = self._key_quota_used.get(key_idx, 0)
            if quota_used < min_quota:
                min_quota = quota_used
                least_used_key = key_idx

        return least_used_key

    def get_per_key_quota(self) -> Dict[int, Dict[str, Any]]:
        """Get quota information for all keys (US-152-3).

        Returns:
            Dictionary mapping key index to quota info (used, limit, remaining, percent)
        """
        # US-155-009: Check for daily quota reset before returning
        self._check_quota_reset()

        result = {}
        for i in range(len(self._api_keys)):
            quota_used = self._key_quota_used.get(i, 0)
            remaining = max(0, self._total_quota_limit - quota_used)
            percent_used = (quota_used / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0
            result[i] = {
                "quota_used": quota_used,
                "quota_limit": self._total_quota_limit,
                "quota_remaining": remaining,
                "percent_used": round(percent_used, 1),
                "is_exhausted": i in self._exhausted_keys,
                "health_status": self._key_health_status.get(i, "healthy"),
            }
        return result

    def _check_quota_reset(self) -> None:
        """Check if any exhausted keys should be reset (US-155-009).

        YouTube API quota resets at midnight UTC daily. This method checks
        if enough time has passed since a key was exhausted to allow retry.
        """
        import datetime

        now = datetime.datetime.utcnow()
        current_hour = now.hour

        # Check each exhausted key
        keys_to_reset = []
        for key_idx in list(self._exhausted_keys):
            if key_idx in self._key_exhausted_at:
                exhausted_time = datetime.datetime.fromtimestamp(self._key_exhausted_at[key_idx])
                # Reset if it's a new day (UTC midnight has passed)
                if exhausted_time.date() < now.date():
                    keys_to_reset.append(key_idx)

        # Reset expired keys
        for key_idx in keys_to_reset:
            self._exhausted_keys.discard(key_idx)
            self._key_quota_used[key_idx] = 0
            self._key_health_status[key_idx] = "healthy"
            self._key_exhausted_at.pop(key_idx, None)
            logger.info(f"API key #{key_idx + 1} quota reset (daily reset)")

    def get_per_key_health(self) -> Dict[int, Dict[str, Any]]:
        """Get health status for all keys (US-155-009).

        Returns:
            Dictionary mapping key index to health info (status, quota_used, quota_remaining, percent_used)
        """
        # Check for daily reset first
        self._check_quota_reset()

        result = {}
        for i in range(len(self._api_keys)):
            quota_used = self._key_quota_used.get(i, 0)
            remaining = max(0, self._total_quota_limit - quota_used)
            percent_used = (quota_used / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

            # Determine health status based on quota usage
            if i in self._exhausted_keys:
                status = "exhausted"
            elif percent_used >= 90:
                status = "exhausted"
            elif percent_used >= self.warn_at_percent:
                status = "quota_warning"
            else:
                status = "healthy"

            # Update stored status
            self._key_health_status[i] = status

            # Check for quota reset time
            exhausted_at = self._key_exhausted_at.get(i)
            reset_in_hours = None
            if exhausted_at:
                import datetime
                exhausted_time = datetime.datetime.fromtimestamp(exhausted_at)
                next_reset = exhausted_time.replace(hour=0, minute=0, second=0, microsecond=0)
                if next_reset <= exhausted_time:
                    next_reset += datetime.timedelta(days=1)
                reset_in_hours = max(0, (next_reset - datetime.datetime.utcnow()).total_seconds() / 3600)

            result[i] = {
                "health_status": status,
                "quota_used": quota_used,
                "quota_remaining": remaining,
                "percent_used": round(percent_used, 1),
                "is_exhausted": i in self._exhausted_keys,
                "exhausted_at": self._key_exhausted_at.get(i),
                "reset_in_hours": round(reset_in_hours, 1) if reset_in_hours else None,
                "error_count": self._key_errors.get(i, {}).get("total", 0),
                "error_403": self._key_errors.get(i, {}).get("403", 0),
                "error_429": self._key_errors.get(i, {}).get("429", 0),
                "error_other": self._key_errors.get(i, {}).get("other", 0),
            }
        return result

    def get_error_summary(self) -> Dict[str, Any]:
        """Get user-facing error summary (US-155-010).

        Returns:
            Dictionary with error summary including counts, most common errors,
            and user-friendly guidance for recovery.
        """
        error_categories = self._metrics.get_error_category_counts()
        total_errors = sum(error_categories.values())

        # Determine most common error
        most_common = max(error_categories.items(), key=lambda x: x[1]) if error_categories else (None, 0)

        # Generate user-facing guidance based on most common error
        guidance = self._get_user_facing_guidance(error_categories)

        return {
            "total_errors": total_errors,
            "error_counts": error_categories,
            "most_common_error": most_common[0] if most_common[0] else "none",
            "most_common_count": most_common[1],
            "recovery_guidance": guidance,
            "keys_exhausted": len(self._exhausted_keys),
            "total_keys": len(self._api_keys),
        }

    def _get_user_facing_guidance(self, error_categories: Dict[str, int]) -> str:
        """Generate user-facing guidance based on error categories."""
        if error_categories.get("quota_exceeded", 0) > 0:
            return "API quota exceeded. Quota resets at midnight PST. Consider: (1) Request quota increase, (2) Use yt-dlp fallback, (3) Add more API keys."
        elif error_categories.get("rate_limited", 0) > 0:
            return "API rate limited. Consider: (1) Wait 60-100s, (2) Reduce frequency, (3) Enable yt-dlp fallback."
        elif error_categories.get("invalid_key", 0) > 0:
            return "Invalid API key. Check Google Cloud Console, verify YouTube Data API v3 enabled."
        elif error_categories.get("permission_denied", 0) > 0:
            return "API access denied. Verify key permissions in Google Cloud Console."
        elif error_categories.get("network_error", 0) > 0:
            return "Network errors. Check connection, verify firewall/proxy, enable yt-dlp fallback."
        elif error_categories.get("timeout", 0) > 0:
            return "Request timeouts. Increase timeout in config.yaml, check network stability."
        elif error_categories.get("temporary_error", 0) > 0:
            return "Temporary server errors. Wait and retry, enable yt-dlp fallback."
        elif sum(error_categories.values()) > 0:
            return "Some errors occurred. Check logs for details."
        return "No errors detected. API operating normally."

    @property
    def quota_percent_used(self) -> float:
        """Get percentage of quota used for active key."""
        key_quota = self._key_quota_used[self._current_key_index]
        return (key_quota / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

    @property
    def active_key_index(self) -> int:
        """Get the current active key index."""
        return self._current_key_index

    @property
    def total_keys(self) -> int:
        """Get total number of API keys."""
        return len(self._api_keys)

    @property
    def rotation_strategy(self) -> str:
        """Get the current rotation strategy."""
        return self._rotation_strategy

    def get_least_used_key(self) -> Optional[int]:
        """US-156-004: Get the key index with the lowest current quota usage.

        This method selects the API key that has the lowest quota consumption,
        useful for determining which key has the most capacity remaining.

        Returns:
            Key index with lowest quota usage, or None if no keys available
        """
        if not self._api_keys:
            return None

        # Consider only non-exhausted keys
        available_keys = [i for i in range(len(self._api_keys)) if i not in self._exhausted_keys]

        if not available_keys:
            return None

        # First, prefer healthy keys
        healthy_keys = [i for i in available_keys if self._key_health_status.get(i, "healthy") == "healthy"]
        keys_to_consider = healthy_keys if healthy_keys else available_keys

        # Find key with minimum quota usage
        min_quota = float('inf')
        least_used_key = None

        for key_idx in keys_to_consider:
            quota_used = self._key_quota_used[key_idx]
            if quota_used < min_quota:
                min_quota = quota_used
                least_used_key = key_idx

        return least_used_key

    def get_key_usage_tracking(self) -> Dict[int, int]:
        """US-156-004: Get the current quota usage per API key.

        Returns:
            Dictionary mapping key index to quota used
        """
        return dict(self._key_quota_used)

    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available API key based on rotation_strategy.

        Returns:
            True if successfully rotated to next key, False if all keys exhausted
        """
        import random

        # Get available (non-exhausted) keys
        available_keys = [i for i in range(len(self._api_keys)) if i not in self._exhausted_keys]

        if not available_keys:
            # All keys exhausted
            logger.warning(
                f"All {len(self._api_keys)} API keys exhausted. "
                f"Will fallback to yt-dlp."
            )
            return False

        # Capture old key index for rotation event tracking (US-153-10)
        old_key_index = self._current_key_index

        # Select next key based on strategy
        if self._rotation_strategy == "sequential":
            # Move to next key in sequence (wrapping around)
            next_index = (self._current_key_index + 1) % len(self._api_keys)
            # Find next available key in sequence
            while next_index not in available_keys:
                next_index = (next_index + 1) % len(self._api_keys)
            self._current_key_index = next_index

        elif self._rotation_strategy == "random":
            # Select random available key
            self._current_key_index = random.choice(available_keys)

        elif self._rotation_strategy == "least_used":
            # US-155-009: Select key with lowest quota usage, preferring healthy keys
            min_quota = float('inf')
            best_key = None

            # First, prefer healthy keys
            healthy_keys = [i for i in available_keys if self._key_health_status.get(i, "healthy") == "healthy"]
            keys_to_consider = healthy_keys if healthy_keys else available_keys

            for key_idx in keys_to_consider:
                quota_used = self._key_quota_used[key_idx]
                if quota_used < min_quota:
                    min_quota = quota_used
                    best_key = key_idx

            # Log if we're using a quota_warning key
            if healthy_keys and best_key not in healthy_keys:
                logger.info(f"Least-used rotation: All healthy keys exhausted, using quota_warning key #{best_key + 1}")

            self._current_key_index = best_key

        elif self._rotation_strategy == "weighted":
            # US-156-004: Distribute load proportionally to remaining quota
            # Keys with more remaining quota have higher probability of selection
            # Weighted random selection: weight = remaining_quota ^ 2 for better distribution

            # First, prefer healthy keys (not quota_warning or exhausted)
            healthy_keys = [i for i in available_keys if self._key_health_status.get(i, "healthy") == "healthy"]
            keys_to_consider = healthy_keys if healthy_keys else available_keys

            # Calculate weights based on remaining quota
            weights = []
            key_indices = []
            total_weight = 0.0

            for key_idx in keys_to_consider:
                quota_used = self._key_quota_used[key_idx]
                remaining = self._total_quota_limit - quota_used
                # Use squared remaining quota as weight for better distribution
                # Higher remaining = significantly higher probability
                weight = remaining ** 2 if remaining > 0 else 0.0
                weights.append(weight)
                key_indices.append(key_idx)
                total_weight += weight

            if total_weight > 0 and key_indices:
                # Weighted random selection
                r = random.random() * total_weight
                cumulative = 0.0
                for i, weight in enumerate(weights):
                    cumulative += weight
                    if r <= cumulative:
                        self._current_key_index = key_indices[i]
                        break
                else:
                    # Fallback to last key if something goes wrong
                    self._current_key_index = key_indices[-1]
            else:
                # All keys have zero remaining quota, fall back to sequential
                self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
                while self._current_key_index not in available_keys:
                    self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)

        elif self._rotation_strategy == "smart":
            # US-154-6: Select key with highest remaining quota (most headroom)
            # US-155-009: Enhanced to prefer healthy keys over quota_warning keys
            max_remaining = -1
            best_key = None
            key_remaining_list = []

            # First, prefer healthy keys (not quota_warning or exhausted)
            healthy_keys = [i for i in available_keys if self._key_health_status.get(i, "healthy") == "healthy"]
            keys_to_consider = healthy_keys if healthy_keys else available_keys

            for key_idx in keys_to_consider:
                quota_used = self._key_quota_used[key_idx]
                remaining = self._total_quota_limit - quota_used
                key_remaining_list.append((key_idx, remaining))
                if remaining > max_remaining:
                    max_remaining = remaining
                    best_key = key_idx

            # Log if we're using a quota_warning key
            if healthy_keys and best_key not in healthy_keys:
                logger.info(f"Smart rotation: All healthy keys exhausted, using quota_warning key #{best_key + 1}")

            # Record utilization balance metric
            if key_remaining_list:
                remaining_values = [r for _, r in key_remaining_list]
                if len(remaining_values) > 1:
                    avg_remaining = sum(remaining_values) / len(remaining_values)
                    min_remaining = min(remaining_values)
                    balance_score = min_remaining / avg_remaining if avg_remaining > 0 else 0
                    self._metrics.record_key_utilization_balance(balance_score)

            # US-154-6: Check if remaining quota is critically low (< 10%)
            # If all available keys are low, trigger fallback warning
            low_quota_threshold = int(self._total_quota_limit * 0.1)  # 10%
            all_keys_low = all(
                self._total_quota_limit - self._key_quota_used[key_idx] < low_quota_threshold
                for key_idx in available_keys
            )
            if all_keys_low:
                logger.warning(
                    f"Smart rotation: All available keys have low quota (< {low_quota_threshold}). "
                    f"Remaining per key: {dict((k, self._total_quota_limit - self._key_quota_used[k]) for k in available_keys)}. "
                    f"Consider falling back to yt-dlp."
                )
                self._metrics.record_fallback(
                    endpoint="smart_rotation",
                    reason="all_keys_low_quota",
                    query=""
                )

            self._current_key_index = best_key

        elif self._rotation_strategy == "weighted":
            # US-156-004: Distribute load proportionally to remaining quota
            # Keys with more remaining quota get selected more often (weighted random selection)
            # First, prefer healthy keys
            healthy_keys = [i for i in available_keys if self._key_health_status.get(i, "healthy") == "healthy"]
            keys_to_consider = healthy_keys if healthy_keys else available_keys

            if not keys_to_consider:
                logger.warning(f"Weighted rotation: No available keys to select from")
                return False

            # Calculate weights based on remaining quota
            weights = []
            total_remaining = 0
            for key_idx in keys_to_consider:
                remaining = max(0, self._total_quota_limit - self._key_quota_used.get(key_idx, 0))
                # Use remaining quota as weight, but ensure minimum weight of 1
                weight = max(1, remaining)
                weights.append(weight)
                total_remaining += remaining

            # Weighted random selection
            if total_remaining > 0:
                # Use cumulative weights for selection
                rand_val = random.random() * total_remaining
                cumulative = 0
                selected_key = keys_to_consider[0]
                for idx, key_idx in enumerate(keys_to_consider):
                    cumulative += weights[idx]
                    if rand_val <= cumulative:
                        selected_key = key_idx
                        break
                self._current_key_index = selected_key
            else:
                # All keys exhausted, select first available
                self._current_key_index = keys_to_consider[0]

            # Log the weighted selection
            remaining_for_selected = max(0, self._total_quota_limit - self._key_quota_used.get(self._current_key_index, 0))
            logger.debug(
                f"Weighted rotation: Selected key #{self._current_key_index + 1} "
                f"with {remaining_for_selected} quota remaining "
                f"(weights: {dict(zip(keys_to_consider, weights))})"
            )

        # Log the rotation
        key_preview = self._api_keys[self._current_key_index][:8]
        logger.info(
            f"Rotating to API key #{self._current_key_index + 1} ({key_preview}...) "
            f"via {self._rotation_strategy} strategy ({len(self._exhausted_keys)} keys exhausted)"
        )
        # US-149-3: Reset predictive warning flag for new key
        self._warn_soon_warned[self._current_key_index] = False

        # US-153-10: Record key rotation event
        self._metrics.record_key_rotation(
            from_key_index=old_key_index,
            to_key_index=self._current_key_index,
            reason="quota_exceeded"
        )

        return True

    def _check_quota(self, cost: int) -> None:
        """Check if quota is available for operation.

        Args:
            cost: Quota cost of the operation

        Raises:
            QuotaExceededError: If quota is exhausted
        """
        current_key = self._current_key_index
        key_quota = self._key_quota_used[current_key]

        if key_quota + cost > self._total_quota_limit:
            # Mark current key as exhausted (US-155-009)
            # US-156-010: Add error code mapping for detailed error logging
            error_desc = get_error_description(402)  # 402 maps to quota exceeded
            error_category = get_error_category(402)
            self._exhausted_keys.add(current_key)
            self._key_health_status[current_key] = "exhausted"
            self._key_exhausted_at[current_key] = time.time()
            key_preview = self._api_keys[current_key][:8]
            logger.warning(
                f"API key #{current_key + 1} ({key_preview}...) quota exhausted "
                f"({key_quota}/{self._total_quota_limit})"
                f" - Error Code: 402 ({error_category.get('description', 'Quota exceeded')})"
            )

            # Try to rotate to next key
            if self._rotate_to_next_key():
                # Successfully rotated, check quota on new key
                new_key = self._current_key_index
                new_quota = self._key_quota_used[new_key]
                if new_quota + cost > self._total_quota_limit:
                    # New key also doesn't have enough quota
                    raise QuotaExceededError(
                        f"All API keys quota exhausted"
                    )
            else:
                # US-152-3: Handle per-key quota_exhausted_action
                if self._quota_exhausted_action == "pause":
                    # Stop the pipeline when all keys exhausted
                    # US-155-010: Record error category for metrics
                    self._metrics.record_error_category("quota_exceeded")
                    raise YouTubeAPIQuotaExceededError(
                        f"All YouTube API keys quota exhausted (action=pause).\n"
                        f"Suggested next steps: (1) Check Google Cloud Console (https://console.cloud.google.com/apis/dashboard) "
                        f"for quota usage, (2) Request quota increase"
                    )
                else:
                    # US-150-11: Enhanced quota exhausted error with actionable guidance
                    # US-155-010: Record error category for metrics
                    self._metrics.record_error_category("quota_exceeded")
                    raise YouTubeAPIQuotaExceededError(
                        f"All YouTube API keys quota exhausted.\n"
                        f"Suggested next steps: (1) Check Google Cloud Console (https://console.cloud.google.com/apis/dashboard) "
                        f"for quota usage, (2) Request quota increase, (3) Enable yt-dlp fallback in config.yaml"
                    )

        # Warn at threshold
        key_quota = self._key_quota_used[self._current_key_index]
        if not self._key_quota_warned[self._current_key_index]:
            percent_used = (key_quota / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0
            if percent_used >= self.warn_at_percent:
                key_preview = self._api_keys[self._current_key_index][:8]
                logger.warning(
                    f"YouTube API key #{self._current_key_index + 1} ({key_preview}...) "
                    f"quota at {percent_used:.1f}% ({key_quota}/{self._total_quota_limit})"
                )
                self._key_quota_warned[self._current_key_index] = True
                # US-155-007: Trigger webhook notification
                self._send_quota_webhook(percent_used, key_quota)

        # US-149-3: Check predictive warning
        self._check_predictive_warning()

        # US-150-7: Check for proactive fallback trigger
        self._check_proactive_fallback()

    def _add_quota(self, cost: int, endpoint: str = None) -> None:
        """Add to quota usage for current key.

        Args:
            cost: Quota cost to add
            endpoint: Optional endpoint type for per-endpoint tracking (US-153-10)
        """
        self._key_quota_used[self._current_key_index] += cost
        # US-146-12: Record quota usage for historical tracking
        total_quota = sum(self._key_quota_used.values())
        self._metrics.record_quota_usage(total_quota)
        # US-153-10: Record quota by endpoint type
        if endpoint:
            self._metrics.record_quota_usage_by_endpoint(endpoint, cost)
        # US-149-3: Track quota usage with timestamps for velocity calculation
        current_time = time.time()
        self._quota_usage_timestamps.append((current_time, total_quota))
        # US-148-7: Persist quota after each update
        self._save_quota()

    def predict_exhaustion_time(self) -> Optional[float]:
        """Predict when quota will be exhausted based on current usage velocity.

        Calculates the rate of quota consumption (units per second) using recent
        usage history and predicts when the quota limit will be reached.

        Returns:
            Estimated minutes until quota exhaustion, or None if prediction
            cannot be made (insufficient data or quota already exhausted)
        """
        # Get current quota usage - check exhaustion first
        current_quota = sum(self._key_quota_used.values())
        remaining_quota = self._total_quota_limit - current_quota

        if remaining_quota <= 0:
            # Already exhausted
            return 0.0

        if len(self._quota_usage_timestamps) < 2:
            # Not enough data to calculate velocity
            return None

        # Calculate velocity using recent entries (last 20 or available)
        recent_entries = list(self._quota_usage_timestamps)
        if len(recent_entries) < 2:
            return None

        # Use linear regression on recent entries for better accuracy
        timestamps = [e[0] for e in recent_entries]
        quotas = [e[1] for e in recent_entries]

        # Calculate slope (velocity) using least squares
        n = len(timestamps)
        if n < 2:
            return None

        sum_x = sum(timestamps)
        sum_y = sum(quotas)
        sum_xy = sum(timestamps[i] * quotas[i] for i in range(n))
        sum_xx = sum(t * t for t in timestamps)

        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # If slope is negative or zero, quota is not increasing
        if slope <= 0:
            return None

        # Calculate time until exhaustion
        seconds_until_exhaustion = remaining_quota / slope
        minutes_until_exhaustion = seconds_until_exhaustion / 60.0

        return minutes_until_exhaustion

    def _send_quota_webhook(self, percent_used: float, quota_used: int) -> None:
        """Send quota warning webhook notification.

        Args:
            percent_used: Percentage of quota used
            quota_used: Absolute quota units used
        """
        if not self._webhook_enabled or not self._webhook_urls:
            return

        # Calculate remaining quota and estimated time
        quota_remaining = max(0, self._total_quota_limit - quota_used)
        remaining_percent = 100.0 - percent_used

        # Predict exhaustion time
        minutes_until_exhaustion = self.predict_exhaustion_time()
        estimated_time_str = f"{minutes_until_exhaustion:.1f} minutes" if minutes_until_exhaustion else "unknown"

        # Build webhook payload
        payload = {
            "event": "quota_warning",
            "key_index": self._current_key_index,
            "key_preview": self._api_keys[self._current_key_index][:8] + "...",
            "quota_limit": self._total_quota_limit,
            "quota_used": quota_used,
            "quota_remaining": quota_remaining,
            "percent_used": round(percent_used, 1),
            "remaining_percent": round(remaining_percent, 1),
            "estimated_time_remaining": estimated_time_str,
            "warn_threshold_percent": self.warn_at_percent,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # Send to all webhook URLs
        for url in self._webhook_urls:
            self._deliver_webhook(url, payload)

    def _deliver_webhook(self, url: str, payload: Dict[str, Any]) -> None:
        """Deliver webhook payload with retry logic.

        Args:
            url: Webhook URL to notify
            payload: JSON payload to send
        """
        import urllib.request
        import urllib.error
        import json

        data = json.dumps(payload).encode('utf-8')

        for attempt in range(self._webhook_retry_count):
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'Matcher/YouTubeAPIClient',
                    },
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=self._webhook_timeout) as response:
                    if response.status == 200:
                        logger.info(f"Quota webhook delivered successfully to {url}")
                    else:
                        logger.warning(f"Quota webhook returned status {response.status} from {url}")
                return  # Success, exit retry loop
            except urllib.error.HTTPError as e:
                logger.warning(f"Quota webhook HTTP error {e.code} to {url}: {e.reason}")
            except urllib.error.URLError as e:
                logger.warning(f"Quota webhook URL error to {url}: {e.reason}")
            except Exception as e:
                logger.warning(f"Quota webhook delivery failed to {url}: {e}")

            # Wait before retry (exponential backoff)
            if attempt < self._webhook_retry_count - 1:
                import time as time_module
                time_module.sleep(2 ** attempt)

        # All retries failed - log but don't crash
        logger.error(f"Quota webhook failed after {self._webhook_retry_count} attempts to {url}")

    def _check_predictive_warning(self) -> None:
        """Check if quota is predicted to exhaust soon and log warning.

        Uses predict_exhaustion_time() to warn before quota exhaustion based on:
        - Remaining quota percentage (below warn_soon_threshold_percent)
        - Predicted time until exhaustion (within warn_soon_threshold_minutes)
        """
        current_key = self._current_key_index

        # Check if we've already warned for this key
        if self._warn_soon_warned.get(current_key, False):
            return

        # Get remaining quota percentage
        current_quota = self._key_quota_used.get(current_key, 0)
        remaining_percent = ((self._total_quota_limit - current_quota) / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

        # Only warn if below threshold
        if remaining_percent >= self.warn_soon_threshold_percent:
            return

        # Predict exhaustion time
        minutes_left = self.predict_exhaustion_time()

        if minutes_left is None:
            return

        # Warn if predicted to exhaust within threshold
        if minutes_left <= self.warn_soon_threshold_minutes:
            logger.warning(
                f"YouTube API key #{current_key + 1} quota predicted to exhaust "
                f"in {minutes_left:.1f} minutes ({remaining_percent:.1f}% remaining). "
                f"Consider switching to yt-dlp fallback."
            )
            self._warn_soon_warned[current_key] = True

    def _check_proactive_fallback(self) -> None:
        """Check and log proactive fallback status.

        US-150-7: Logs when quota falls below the proactive fallback threshold.
        US-152-3: Now emits warning at quota_fallback_threshold_percent for each key.
        The actual fallback decision is made by callers using should_proactive_fallback().
        """
        current_key = self._current_key_index
        current_quota = self._key_quota_used.get(current_key, 0)
        remaining_percent = ((self._total_quota_limit - current_quota) / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

        # US-152-3: Warn at quota_fallback_threshold_percent for each key
        if remaining_percent < self._proactive_fallback_threshold_percent:
            if not self._key_fallback_threshold_warned.get(current_key, False):
                key_preview = self._api_keys[current_key][:8]
                logger.warning(
                    f"YouTube API key #{current_key + 1} ({key_preview}...) quota at {remaining_percent:.1f}% "
                    f"(< {self._proactive_fallback_threshold_percent}% threshold). "
                    f"Falling back to yt-dlp recommended."
                )
                self._key_fallback_threshold_warned[current_key] = True

    def should_proactive_fallback(self, estimated_cost: int = 100) -> bool:
        """Check if we should proactively fallback to yt-dlp based on quota prediction.

        US-150-7: Determines whether to switch to yt-dlp before quota exhaustion
        based on predicted remaining quota. This prevents hitting hard quota limits
        mid-operation by falling back proactively.

        US-155-003: Now uses configurable prediction threshold (default 30 minutes)
        and adaptive threshold based on time of day.

        Args:
            estimated_cost: Estimated quota cost for the upcoming operation

        Returns:
            True if proactive fallback to yt-dlp is recommended
        """
        current_key = self._current_key_index

        # Get current quota for active key
        current_quota = self._key_quota_used.get(current_key, 0)
        remaining_quota = self._total_quota_limit - current_quota
        remaining_percent = (remaining_quota / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

        # Calculate adaptive threshold based on time of day (US-155-003)
        prediction_threshold = self._get_adaptive_prediction_threshold()

        # Check if remaining quota is below static percentage threshold
        if remaining_percent >= self._proactive_fallback_threshold_percent:
            return False  # Still have enough quota

        # US-155-003: Check for abnormal usage rate
        if self._quota_abnormal_rate_warning_enabled:
            self._check_abnormal_usage_rate()

        # Predict exhaustion time
        minutes_left = self.predict_exhaustion_time()

        # If we can't predict, make decision based on remaining percent only
        if minutes_left is None:
            # Already below threshold but can't predict - fallback to be safe
            if remaining_percent < self._proactive_fallback_threshold_percent:
                logger.info(
                    f"Proactive fallback triggered: remaining quota {remaining_percent:.1f}% "
                    f"below threshold {self._proactive_fallback_threshold_percent}% "
                    f"(unable to predict exhaustion time)"
                )
                return True
            return False

        # US-155-003: Check if quota will exhaust within the prediction threshold (default 30 minutes)
        if minutes_left <= prediction_threshold:
            logger.info(
                f"Proactive fallback triggered: quota predicted to exhaust "
                f"in {minutes_left:.1f} minutes (threshold: {prediction_threshold} min, "
                f"{remaining_percent:.1f}% remaining)"
            )
            # Record prediction for accuracy metrics
            self._record_prediction_accuracy(minutes_left)
            return True

        # Also check if the estimated cost would exceed remaining quota
        if estimated_cost > remaining_quota:
            logger.info(
                f"Proactive fallback triggered: estimated cost {estimated_cost} "
                f"exceeds remaining quota {remaining_quota} ({remaining_percent:.1f}% remaining)"
            )
            return True

        return False

    def _get_adaptive_prediction_threshold(self) -> float:
        """Get the adaptive prediction threshold based on time of day.

        US-155-003: During peak hours, use a lower threshold (trigger fallback earlier)
        to account for higher usage rates and potential rate limiting.

        Returns:
            Prediction threshold in minutes (adjusted for peak hours if enabled)
        """
        if not self._quota_fallback_adaptive_enabled:
            return float(self._quota_fallback_prediction_minutes)

        # Get current hour
        current_hour = datetime.now().hour

        # Check if current time is within peak hours
        peak_start = self._quota_fallback_peak_start_hour
        peak_end = self._quota_fallback_peak_end_hour

        # Handle overnight peak hours (e.g., 22-6)
        is_peak_hours = False
        if peak_start > peak_end:
            # Peak hours span midnight (e.g., 22-6)
            is_peak_hours = current_hour >= peak_start or current_hour < peak_end
        else:
            # Peak hours within same day (e.g., 9-21)
            is_peak_hours = peak_start <= current_hour < peak_end

        if is_peak_hours:
            # During peak hours, trigger fallback earlier
            threshold = self._quota_fallback_prediction_minutes / self._quota_fallback_peak_multiplier
            logger.debug(
                f"Peak hours detected ({peak_start}:00-{peak_end}:00), "
                f"using adjusted threshold: {threshold:.1f} minutes"
            )
            return threshold

        return float(self._quota_fallback_prediction_minutes)

    def _check_abnormal_usage_rate(self) -> None:
        """Check if current quota usage rate is abnormal and log warning if so.

        US-155-003: Compares current usage rate to historical average and logs
        a warning if the current rate significantly exceeds the average.
        """
        if len(self._quota_usage_timestamps) < 5:
            return  # Not enough data

        try:
            # Calculate current rate (last few entries)
            recent = list(self._quota_usage_timestamps)
            if len(recent) < 3:
                return

            # Current rate: quota change over time in last 3 entries
            time_delta = recent[-1][0] - recent[-3][0]
            quota_delta = recent[-1][1] - recent[-3][1]

            if time_delta <= 0:
                return

            current_rate_per_minute = (quota_delta / time_delta) * 60

            # Calculate average rate from all historical data
            if len(recent) < 2:
                return

            total_time = recent[-1][0] - recent[0][0]
            total_quota = recent[-1][1] - recent[0][1]

            if total_time <= 0:
                return

            avg_rate_per_minute = (total_quota / total_time) * 60

            if avg_rate_per_minute <= 0:
                return

            # Check if current rate exceeds threshold
            rate_ratio = current_rate_per_minute / avg_rate_per_minute

            if rate_ratio > self._quota_abnormal_rate_threshold:
                logger.warning(
                    f"Abnormal quota usage rate detected: current rate {current_rate_per_minute:.1f} quota/min "
                    f"is {rate_ratio:.1f}x the average ({avg_rate_per_minute:.1f} quota/min). "
                    f"This may indicate a rate limit issue or high API usage."
                )
                # Record in metrics
                self._metrics.record_abnormal_rate_warning(rate_ratio)

        except (ZeroDivisionError, IndexError):
            pass  # Skip if calculation fails

    def _record_prediction_accuracy(self, predicted_minutes: float) -> None:
        """Record quota prediction for accuracy tracking.

        US-155-003: Tracks predictions to calculate accuracy metrics.

        Args:
            predicted_minutes: Predicted minutes until quota exhaustion
        """
        # Store prediction with current quota level for later accuracy calculation
        current_quota = sum(self._key_quota_used.values())
        self._quota_predictions.append({
            'timestamp': time.time(),
            'predicted_minutes': predicted_minutes,
            'quota_used': current_quota,
            'quota_limit': self._total_quota_limit
        })

    def get_quota_prediction_metrics(self) -> Dict[str, Any]:
        """Get metrics about quota prediction accuracy.

        US-155-003: Returns prediction accuracy metrics based on historical predictions.

        Returns:
            Dict with prediction accuracy metrics
        """
        if not self._quota_predictions:
            return {
                "prediction_count": 0,
                "average_predicted_minutes": 0.0,
                "has_predictions": False
            }

        predictions = list(self._quota_predictions)
        predicted_minutes_values = [p['predicted_minutes'] for p in predictions]

        return {
            "prediction_count": len(predictions),
            "average_predicted_minutes": sum(predicted_minutes_values) / len(predicted_minutes_values) if predicted_minutes_values else 0.0,
            "min_predicted_minutes": min(predicted_minutes_values) if predicted_minutes_values else 0.0,
            "max_predicted_minutes": max(predicted_minutes_values) if predicted_minutes_values else 0.0,
            "has_predictions": True,
            "adaptive_threshold_enabled": self._quota_fallback_adaptive_enabled,
            "prediction_threshold_minutes": self._quota_fallback_prediction_minutes,
            "peak_hours": f"{self._quota_fallback_peak_start_hour}:00-{self._quota_fallback_peak_end_hour}:00",
            "peak_multiplier": self._quota_fallback_peak_multiplier,
            "abnormal_rate_warning_enabled": self._quota_abnormal_rate_warning_enabled,
            "abnormal_rate_threshold": self._quota_abnormal_rate_threshold
        }

    def get_quota_status(self) -> Dict[str, Any]:
        """Get current quota status with prediction info.

        US-150-7: Returns comprehensive quota status including remaining
        quota, percentage used, and prediction info.

        Returns:
            Dict with quota status details
        """
        current_key = self._current_key_index
        current_quota = self._key_quota_used.get(current_key, 0)
        remaining_quota = self._total_quota_limit - current_quota
        remaining_percent = (remaining_quota / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

        # Get prediction
        minutes_left = self.predict_exhaustion_time()

        # Get per-key quota breakdown (US-155-011)
        per_key_quota = []
        for i in range(len(self._api_keys)):
            key_quota_used = self._key_quota_used.get(i, 0)
            key_remaining = max(0, self._total_quota_limit - key_quota_used)
            key_percent = (key_remaining / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0
            per_key_quota.append({
                'key_index': i,
                'quota_used': key_quota_used,
                'quota_limit': self._total_quota_limit,
                'quota_remaining': key_remaining,
                'quota_percent_remaining': round(key_percent, 1),
                'is_exhausted': i in self._exhausted_keys,
            })

        return {
            "current_key": current_key,
            "quota_used": current_quota,
            "quota_limit": self._total_quota_limit,
            "quota_remaining": remaining_quota,
            "quota_percent_remaining": round(remaining_percent, 2),
            "warn_at_percent": self.warn_at_percent,
            "proactive_fallback_threshold_percent": self._proactive_fallback_threshold_percent,
            "predict_minutes_until_exhaustion": round(minutes_left, 1) if minutes_left is not None else None,
            "should_proactive_fallback": self.should_proactive_fallback(),
            "per_key_quota": per_key_quota,
        }

    def estimate_quota_for_pipeline(
        self,
        keyword_count: int,
        max_results_per_search: int = 50,
        estimated_video_metadata_calls: int = 0,
        estimated_caption_fetches: int = 0,
    ) -> Dict[str, Any]:
        """Estimate quota needed for a full pipeline run.

        US-154-11: Pre-flight quota check before starting pipeline.

        Args:
            keyword_count: Number of keywords to search
            max_results_per_search: Maximum results per search query
            estimated_video_metadata_calls: Estimated number of video metadata API calls
            estimated_caption_fetches: Estimated number of caption fetch attempts

        Returns:
            Dict with quota estimates and recommendations
        """
        # Quota costs per operation
        search_quota = keyword_count * QUOTA_COST_SEARCH  # 100 units per search
        metadata_quota = estimated_video_metadata_calls * QUOTA_COST_VIDEOS  # 1 unit per video metadata
        caption_quota = estimated_caption_fetches * QUOTA_COST_CAPTIONS  # 50 units per caption fetch

        total_estimated = search_quota + metadata_quota + caption_quota

        # Get current remaining quota
        remaining = self.get_remaining_quota()
        remaining_percent = (remaining / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

        # Determine status
        status = "sufficient"
        if remaining < total_estimated * 0.5:
            status = "critically_low"
        elif remaining < total_estimated:
            status = "insufficient"
        elif remaining_percent < 20:
            status = "low"

        return {
            "estimated_search_quota": search_quota,
            "estimated_metadata_quota": metadata_quota,
            "estimated_caption_quota": caption_quota,
            "total_estimated_quota": total_estimated,
            "current_quota_remaining": remaining,
            "current_quota_limit": self._total_quota_limit,
            "remaining_percent": round(remaining_percent, 2),
            "status": status,
            "recommendation": self._get_quota_recommendation(status, remaining, total_estimated),
        }

    def _get_quota_recommendation(self, status: str, remaining: int, estimated: int) -> str:
        """Get recommendation based on quota status."""
        if status == "sufficient":
            return "Proceed with YouTube API - sufficient quota available"
        elif status == "low":
            return "Warning: Quota running low. Consider using yt-dlp fallback for captions."
        elif status == "insufficient":
            return f"Warning: Estimated quota ({estimated}) exceeds remaining ({remaining}). Consider --no-youtube-api"
        else:  # critically_low
            return "Critical: Insufficient quota. Recommend using --no-youtube-api to skip API entirely"

    def _get_quota_file_path(self) -> str:
        """Get the path to the quota persistence file.

        Returns:
            Path to youtube_api_quota.json in ~/.matcher/
        """
        matcher_dir = os.path.expanduser("~/.matcher")
        os.makedirs(matcher_dir, exist_ok=True)
        return os.path.join(matcher_dir, "youtube_api_quota.json")

    def _load_quota(self) -> None:
        """Load persisted quota usage from file.

        Handles corrupted quota files gracefully by starting fresh.
        Also resets if the number of API keys has changed.
        """
        if not os.path.exists(self._quota_file_path):
            logger.debug(f"Quota file not found, starting fresh: {self._quota_file_path}")
            return

        try:
            with open(self._quota_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            if not isinstance(data, dict):
                logger.warning(f"Invalid quota file format, starting fresh")
                return

            # Check if number of keys has changed - if so, reset quota
            key_quota = data.get("key_quota_used", {})
            saved_key_count = len(key_quota)
            if saved_key_count != len(self._api_keys):
                logger.info(
                    f"API key count changed ({saved_key_count} -> {len(self._api_keys)}), "
                    f"resetting quota"
                )
                return

            # Load per-key quota usage
            for key_index, quota in key_quota.items():
                idx = int(key_index)
                if idx < len(self._api_keys):
                    self._key_quota_used[idx] = int(quota)

            # Load last reset date
            self._last_reset_date = data.get("last_reset_date")

            # Load exhausted keys
            exhausted = data.get("exhausted_keys", [])
            self._exhausted_keys = set(int(k) for k in exhausted if k.isdigit())

            logger.info(
                f"Loaded quota from persistence: {self._key_quota_used}, "
                f"last reset: {self._last_reset_date}"
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted quota file (JSON error), starting fresh: {e}")
        except Exception as e:
            logger.warning(f"Failed to load quota file, starting fresh: {e}")

    def _save_quota(self) -> None:
        """Save current quota usage to file for persistence across sessions."""
        try:
            data = {
                "key_quota_used": {str(k): v for k, v in self._key_quota_used.items()},
                "exhausted_keys": list(self._exhausted_keys),
                "last_reset_date": self._get_utc_date(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._quota_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save quota to file: {e}")

    def _get_utc_date(self) -> str:
        """Get current UTC date as YYYY-MM-DD string.

        Returns:
            Current date in UTC timezone.
        """
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _parse_date_parameter(self, date_value: str) -> str:
        """Convert relative date or ISO string to YouTube API format.

        Args:
            date_value: ISO 8601 date string (e.g., '2020-01-01T00:00:00Z') or
                       relative date (e.g., '7days', '30days', '90days', '1year', '5years')

        Returns:
            ISO 8601 formatted date string for YouTube API, or empty string if invalid.

        Raises:
            ValueError: If date format is invalid (for validation purposes).
        """
        if not date_value:
            return ""

        # US-155-004: Validate ISO 8601 date format
        import re
        # Match ISO 8601 format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ
        iso_pattern = r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z?)?$'

        # Check if it's an ISO 8601 date (starts with 20xx year)
        if date_value.startswith("20"):
            # Validate the format
            if not re.match(iso_pattern, date_value):
                logger.warning(f"Invalid ISO 8601 date format: {date_value}, treating as empty")
                return ""
            return date_value

        # Map relative dates to timedelta
        relative_map = {
            "today": timedelta(0),
            "7days": timedelta(days=7),
            "30days": timedelta(days=30),
            "90days": timedelta(days=90),
            "1year": timedelta(days=365),
            "5years": timedelta(days=365 * 5),
            "last_7_days": timedelta(days=7),
            "last_30_days": timedelta(days=30),
            "last_90_days": timedelta(days=90),
            "last_1year": timedelta(days=365),
            "last_5years": timedelta(days=365 * 5),
            "last_year": timedelta(days=365),
        }

        # Normalize the input
        normalized = date_value.lower().strip()

        if normalized in relative_map:
            delta = relative_map[normalized]
            target_date = datetime.now(timezone.utc) - delta
            return target_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.warning(f"Unrecognized date format: {date_value}, treating as empty")
        return ""

    def _check_daily_reset(self) -> None:
        """Check if it's a new day and reset quota if needed.

        Automatically resets quota at midnight UTC each day.
        Also invalidates query cache when quota resets (US-149-9).
        """
        current_date = self._get_utc_date()

        # Force reset via CLI flag
        if self._force_reset:
            logger.info("Forcing quota reset via CLI flag")
            self.reset_quota()
            # US-149-9: Invalidate cache on forced reset
            self.invalidate_query_cache()
            self._force_reset = False
            return

        # Check if date has changed
        if self._last_reset_date is None or self._last_reset_date != current_date:
            if self._last_reset_date is not None:
                days_elapsed = (datetime.strptime(current_date, "%Y-%m-%d") -
                               datetime.strptime(self._last_reset_date, "%Y-%m-%d")).days
                logger.info(
                    f"New day detected (last: {self._last_reset_date}, current: {current_date}, "
                    f"days_elapsed: {days_elapsed}) - resetting quota"
                )
            else:
                logger.info(f"First run or no reset date - resetting quota")

            self.reset_quota()
            # US-149-9: Invalidate query cache on quota reset
            self.invalidate_query_cache()
            self._last_reset_date = current_date
            self._save_quota()

    def force_quota_reset(self) -> None:
        """Force quota reset on next check (called from CLI).

        This is used by the --reset-youtube-quota flag.
        """
        self._force_reset = True
        self._check_daily_reset()

    def _check_circuit_breaker(self, endpoint: str = None) -> None:
        """Check if circuit breaker is open and raise if so.

        Args:
            endpoint: The endpoint type ("search", "videos", "captions", "channels")

        Raises:
            APIError: If circuit breaker is open
        """
        # US-152-4: Use per-endpoint circuit breaker
        if endpoint:
            # Use the new per-endpoint circuit breaker
            self._endpoint_circuit_breaker.check_and_wait(endpoint)
            return

        # Fallback to old simple circuit breaker
        if not self._circuit_breaker_open:
            return

        if time.time() >= self._circuit_breaker_open_until:
            # Circuit breaker reset time passed, try again
            logger.info("Circuit breaker reset - attempting API call")
            self._circuit_breaker_open = False
            return

        raise YouTubeAPIRateLimitedError(
            f"Circuit breaker open - API paused for {self._circuit_breaker_pause}s",
            retry_after=float(self._circuit_breaker_pause),
        )

    def _record_call_result(self, success: bool, endpoint: str = None) -> None:
        """Record the result of an API call for circuit breaker.

        US-157-009: Track unnecessary backoffs when predictive backoff was applied but request succeeded.

        Args:
            success: Whether the call succeeded
            endpoint: The endpoint type ("search", "videos", "captions", "channels")
        """
        # US-157-009: Track unnecessary backoff when request succeeds after predictive backoff
        if success and self._last_backoff_applied:
            # Predictive backoff was applied but request succeeded - unnecessary backoff
            self._record_unnecessary_backoff()
            self._last_backoff_applied = False  # Reset after recording

        # US-152-4: Use per-endpoint circuit breaker when endpoint is provided
        if endpoint:
            if success:
                self._endpoint_circuit_breaker.record_success(endpoint)
            else:
                self._endpoint_circuit_breaker.record_failure(endpoint)
            return

        # Fallback to old simple circuit breaker
        self._recent_calls.append(success)

        # Check if we need to open the circuit
        if len(self._recent_calls) >= self._circuit_breaker_window:
            failures = sum(1 for s in self._recent_calls if not s)
            failure_rate = failures / len(self._recent_calls)

            if failure_rate > self._circuit_breaker_threshold:
                self._circuit_breaker_open = True
                self._circuit_breaker_open_until = time.time() + self._circuit_breaker_pause
                logger.warning(
                    f"Circuit breaker OPEN: {failure_rate:.1%} failure rate "
                    f"({failures}/{len(self._recent_calls)}), pausing for {self._circuit_breaker_pause}s"
                )

    def _record_key_error(self, status_code: int) -> None:
        """Record an error for the current key (US-150-3).

        Args:
            status_code: HTTP status code (403, 429, etc.)
        """
        key_idx = self._current_key_index
        if status_code == 403:
            self._key_errors[key_idx]["403"] += 1
        elif status_code == 429:
            self._key_errors[key_idx]["429"] += 1
        else:
            self._key_errors[key_idx]["other"] += 1
        self._key_errors[key_idx]["total"] += 1
        # US-152-7: Also record in metrics for errors_by_type
        self._metrics.record_error_by_status(status_code)

    def _record_key_success(self) -> None:
        """Record a success for the current key (US-150-3)."""
        self._key_successes[self._current_key_index] += 1

    # US-153-9: Rate limit prediction methods

    def _apply_predictive_backoff(self, base_delay: float, keyword: Optional[str] = None) -> float:
        """Apply predictive backoff based on time-of-day patterns.

        This method uses the RateLimitPredictor to estimate the likelihood of
        rate limiting and applies additional backoff when the likelihood is high.

        Args:
            base_delay: Base delay in seconds (e.g., from exponential backoff)
            keyword: Optional keyword for keyword-specific prediction

        Returns:
            Modified delay with predictive backoff applied
        """
        if not self._predictor:
            return base_delay

        # Record this attempt
        self._predictor.record_attempt(keyword=keyword)

        # Get prediction with confidence
        prediction = self._predictor.get_prediction_with_confidence(keyword=keyword)
        likelihood = prediction["likelihood"]
        confidence = prediction["confidence"]
        confidence_level = prediction["confidence_level"]

        # US-153-9: Log prediction decisions with confidence level
        logger.info(
            f"Rate limit prediction: likelihood={likelihood:.2f}, confidence={confidence} "
            f"({confidence_level}), backoff_multiplier={self._backoff_multiplier}"
        )

        # Apply backoff multiplier if likelihood is high (threshold 0.6)
        if likelihood > 0.6:
            # Track predicted rate limit
            self._predicted_rate_limits += 1
            self._last_backoff_applied = True

            # Calculate additional delay based on likelihood and confidence
            # Higher confidence = more aggressive backoff
            confidence_boost = {
                "high": 1.0,
                "medium": 0.7,
                "low": 0.4,
            }.get(confidence_level, 0.5)

            extra_delay = base_delay * (self._backoff_multiplier - 1.0) * likelihood * confidence_boost
            total_delay = base_delay + extra_delay

            logger.info(
                f"Predictive backoff applied: base={base_delay:.2f}s, extra={extra_delay:.2f}s, "
                f"total={total_delay:.2f}s (likelihood={likelihood:.2f}, confidence={confidence})"
            )

            return total_delay

        # US-157-9: Reset the backoff flag when not applied
        self._last_backoff_applied = False
        return base_delay

    def _record_rate_limit_event(self, trigger_category: str = "429", keyword: Optional[str] = None) -> None:
        """Record a rate limit event for prediction tracking.

        Args:
            trigger_category: Category of trigger (e.g., '429', '403')
            keyword: Keyword associated with this event
        """
        if not self._predictor:
            return

        self._predictor.record_rate_limit_event(
            trigger_category=trigger_category,
            keyword=keyword,
        )
        self._actual_rate_limits += 1

        # US-157-9: Track prediction accuracy
        # If we predicted rate limit and it occurred, that's a true positive
        # If we predicted but it didn't occur, that's a false positive (unnecessary backoff)
        if self._last_backoff_applied:
            # We predicted and it happened - correct prediction
            self._predictor.record_actual_result(
                predicted_rate_limit=True,
                actual_rate_limit=True
            )
        else:
            # We didn't apply backoff but got rate limit - false negative
            self._predictor.record_actual_result(
                predicted_rate_limit=False,
                actual_rate_limit=True
            )
        # Reset the flag
        self._last_backoff_applied = False

        logger.info(
            f"Rate limit event recorded: category={trigger_category}, "
            f"predicted={self._predicted_rate_limits}, actual={self._actual_rate_limits}"
        )

    def _record_unnecessary_backoff(self) -> None:
        """Record when predictive backoff was applied but no rate limit occurred.

        US-157-009: Track unnecessary backoffs for metrics.

        This should be called when a request completes successfully (no rate limit)
        after predictive backoff was applied.
        """
        self._unnecessary_backoffs += 1

        # Also record this in the predictor for accuracy tracking
        if self._predictor:
            self._predictor.record_actual_result(
                predicted_rate_limit=True,
                actual_rate_limit=False  # Backoff predicted but didn't occur
            )

        logger.info(
            f"Unnecessary backoff recorded: total_unnecessary={self._unnecessary_backoffs}"
        )

    def predictive_rate_limit_status(self) -> Dict[str, Any]:
        """Get current predictive rate limit risk level.

        US-157-009: Add endpoint to check current risk level.

        Returns:
            Dict with current risk assessment including:
            - risk_level: "low", "medium", "high", or "critical"
            - likelihood: predicted probability of rate limit (0.0-1.0)
            - confidence: prediction confidence level
            - time_window: current time window
            - recommended_action: action to take based on risk
        """
        if not self._predictor or not self._rate_limit_prediction_enabled:
            return {
                "risk_level": "unknown",
                "prediction_enabled": False,
                "message": "Rate limit prediction is not enabled",
            }

        prediction = self._predictor.get_prediction_with_confidence()
        likelihood = prediction["likelihood"]
        confidence = prediction["confidence"]
        time_window_stats = self._predictor.get_current_time_window_stats()
        time_window = time_window_stats.get("time_window", "unknown")

        # Determine risk level based on likelihood
        if likelihood >= 0.8:
            risk_level = "critical"
            recommended_action = "pause_requests"
        elif likelihood >= 0.6:
            risk_level = "high"
            recommended_action = "aggressive_backoff"
        elif likelihood >= 0.4:
            risk_level = "medium"
            recommended_action = "moderate_backoff"
        elif likelihood >= 0.2:
            risk_level = "low"
            recommended_action = "normal_requests"
        else:
            risk_level = "minimal"
            recommended_action = "normal_requests"

        # Adjust based on confidence
        if confidence == "low" and likelihood > 0.5:
            # Low confidence with moderate likelihood - be more cautious
            if risk_level in ("medium", "low"):
                risk_level = "medium"
                recommended_action = "moderate_backoff"

        return {
            "risk_level": risk_level,
            "likelihood": likelihood,
            "confidence": confidence,
            "confidence_level": prediction.get("confidence_level", confidence),
            "time_window": time_window,
            "hour": time_window_stats.get("hour"),
            "recommended_action": recommended_action,
            "prediction_enabled": True,
            "should_increase_budget": prediction.get("should_increase_budget", False),
            "budget_recommendation": prediction.get("budget_recommendation", "normal"),
        }

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get rate limit prediction statistics.

        US-157-009: Includes prediction_accuracy and unnecessary_backoffs metrics.

        Returns:
            Dict with prediction metrics including predicted vs actual rate limits
        """
        stats = {
            "prediction_enabled": self._rate_limit_prediction_enabled,
            "predicted_rate_limits": self._predicted_rate_limits,
            "actual_rate_limits": self._actual_rate_limits,
            "unnecessary_backoffs": self._unnecessary_backoffs,
        }

        if self._predictor:
            stats["predictor"] = self._predictor.get_prediction_with_confidence()
            stats["sliding_window"] = self._predictor.get_sliding_window_stats()
            stats["time_window_prediction"] = self._predictor.get_time_window_prediction()
            # US-157-009: Add prediction accuracy metrics
            stats["prediction_accuracy"] = self._predictor.get_prediction_metrics()

        return stats

    def predictive_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit risk level for proactive decision making.

        US-157-009: Endpoint to check current risk level before making API calls.

        Returns:
            Dict with:
                - risk_level: "low", "medium", "high", or "critical"
                - likelihood: predicted rate limit probability (0.0-1.0)
                - confidence: prediction confidence level
                - recommendation: "proceed", "backoff", or "wait"
                - current_time_window: time window info
        """
        if not self._predictor:
            return {
                "risk_level": "unknown",
                "prediction_enabled": False,
                "message": "Rate limit prediction is not enabled",
            }

        # Get current prediction
        prediction = self._predictor.get_prediction_with_confidence()
        time_window_info = self._predictor.get_time_window_prediction()

        likelihood = prediction["likelihood"]
        confidence = prediction["confidence"]
        confidence_level = prediction["confidence_level"]

        # Determine risk level based on likelihood and confidence
        if likelihood > 0.8:
            risk_level = "critical"
        elif likelihood > 0.6:
            risk_level = "high"
        elif likelihood > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Determine recommendation
        if risk_level == "critical":
            recommendation = "wait"
        elif risk_level == "high":
            recommendation = "backoff"
        elif risk_level == "medium":
            recommendation = "backoff" if confidence == "high" else "proceed"
        else:
            recommendation = "proceed"

        return {
            "risk_level": risk_level,
            "likelihood": likelihood,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "current_time_window": time_window_info.get("time_window", "unknown"),
            "hour": time_window_info.get("hour"),
            "is_weekend": time_window_info.get("is_weekend"),
            "prediction_enabled": True,
            "unnecessary_backoffs": self._unnecessary_backoffs,
            "predicted_rate_limits": self._predicted_rate_limits,
            "actual_rate_limits": self._actual_rate_limits,
        }

    def get_key_health(self, key_index: int) -> Dict[str, Any]:
        """Get health metrics for a specific key (US-150-3).

        Args:
            key_index: Index of the key to get health for

        Returns:
            Dict with error rates and health status
        """
        errors = self._key_errors.get(key_index, {"403": 0, "429": 0, "other": 0, "total": 0})
        successes = self._key_successes.get(key_index, 0)
        total = errors["total"] + successes

        error_rate = (errors["total"] / total * 100) if total > 0 else 0.0

        return {
            "errors_403": errors["403"],
            "errors_429": errors["429"],
            "errors_other": errors["other"],
            "total_errors": errors["total"],
            "successes": successes,
            "total_requests": total,
            "error_rate_percent": round(error_rate, 2),
            "is_healthy": error_rate < 50.0,  # Healthy if <50% error rate
        }

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, expires_at = self._cache[key]
            if time.time() < expires_at:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Set cache value with TTL."""
        self._cache[key] = (value, time.time() + self.cache_ttl)

    # =========================================================================
    # US-157-003: Metadata enrichment caching
    # =========================================================================

    def _get_metadata_cache_key(self, video_id: str, preferred_language: Optional[str] = None) -> str:
        """Generate cache key for video metadata.

        Args:
            video_id: YouTube video ID
            preferred_language: Optional language code

        Returns:
            Cache key string
        """
        lang = preferred_language or "default"
        return f"metadata:{lang}:{video_id}"

    def _get_cached_video_details(
        self,
        video_ids: List[str],
        preferred_language: Optional[str] = None,
    ) -> Tuple[Dict[str, VideoDetails], Dict[str, VideoDetails], List[str]]:
        """Get cached video details if not expired.

        Args:
            video_ids: List of YouTube video IDs
            preferred_language: Optional language code for cache key

        Returns:
            Tuple of (cached_results, cached_dict, uncached_ids)
        """
        cached_results: Dict[str, VideoDetails] = {}
        uncached_ids: List[str] = []

        for video_id in video_ids:
            cache_key = self._get_metadata_cache_key(video_id, preferred_language)
            if cache_key in self._metadata_enrichment_cache:
                value, expires_at = self._metadata_enrichment_cache[cache_key]
                if time.time() < expires_at:
                    cached_results[video_id] = value
                else:
                    # TTL expired - remove from cache
                    del self._metadata_enrichment_cache[cache_key]
                    uncached_ids.append(video_id)
            else:
                uncached_ids.append(video_id)

        return cached_results, cached_results, uncached_ids

    def _cache_video_details(
        self,
        video_details: Dict[str, VideoDetails],
        preferred_language: Optional[str] = None,
    ) -> None:
        """Cache video details with configurable TTL.

        Args:
            video_details: Dict mapping video_id to VideoDetails
            preferred_language: Optional language code for cache key
        """
        for video_id, details in video_details.items():
            cache_key = self._get_metadata_cache_key(video_id, preferred_language)
            self._metadata_enrichment_cache[cache_key] = (
                details,
                time.time() + self._metadata_enrichment_cache_ttl
            )

    def _parse_date_string(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO 8601 format (US-155-4).

        Supports:
        - ISO 8601: "2024-01-01T00:00:00Z"
        - Relative: "last_30_days", "last_year", "last_2_years", "last_week", "last_month"

        Args:
            date_str: Date string to parse

        Returns:
            ISO 8601 formatted date string or None if invalid
        """
        if not date_str:
            return None

        # Check if already ISO 8601 format
        try:
            # Try parsing as ISO 8601
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        except (ValueError, AttributeError):
            pass

        # Parse relative dates
        date_str_lower = date_str.lower().strip()

        now = datetime.now(timezone.utc)

        if date_str_lower == "last_week":
            delta = timedelta(days=7)
        elif date_str_lower == "last_month":
            delta = timedelta(days=30)
        elif date_str_lower == "last_year":
            delta = timedelta(days=365)
        elif date_str_lower.startswith("last_") and date_str_lower.endswith("_days"):
            # Extract number from "last_N_days"
            try:
                days = int(date_str_lower[5:-5])  # Remove "last_" and "_days"
                delta = timedelta(days=days)
            except ValueError:
                logger.warning(f"US-155-4: Invalid relative date format: {date_str}")
                return None
        elif date_str_lower.startswith("last_") and date_str_lower.endswith("_years"):
            # Extract number from "last_N_years"
            try:
                years = int(date_str_lower[5:-6])  # Remove "last_" and "_years"
                delta = timedelta(days=years * 365)
            except ValueError:
                logger.warning(f"US-155-4: Invalid relative date format: {date_str}")
                return None
        else:
            logger.warning(f"US-155-4: Unknown date format: {date_str}")
            return None

        result_date = now - delta
        return result_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # US-156-002: Response structure validation
    def validate_response_structure(
        self,
        data: Dict[str, Any],
        endpoint: str,
    ) -> None:
        """Validate that API response has required fields before processing.

        US-156-002: Validates response JSON has required 'items' array field.
        This prevents processing malformed responses that could cause
        downstream issues or silent failures.

        Args:
            data: The API response data to validate
            endpoint: The API endpoint name (search, videos, captions)

        Raises:
            YouTubeAPIError: If response structure is invalid (missing 'items' field)
        """
        if not isinstance(data, dict):
            raise YouTubeAPIError(
                f"Invalid response from {endpoint} endpoint: response is not a valid JSON object",
                error_type=YouTubeAPIError.TYPE_NETWORK_ERROR,
                endpoint=endpoint,
            )

        if "items" not in data:
            raise YouTubeAPIError(
                f"Invalid response from {endpoint} endpoint: missing required 'items' field. "
                f"Response keys: {list(data.keys())}",
                error_type=YouTubeAPIError.TYPE_NETWORK_ERROR,
                endpoint=endpoint,
            )

        if not isinstance(data["items"], list):
            raise YouTubeAPIError(
                f"Invalid response from {endpoint} endpoint: 'items' field is not an array. "
                f"Expected list, got {type(data['items']).__name__}",
                error_type=YouTubeAPIError.TYPE_NETWORK_ERROR,
                endpoint=endpoint,
            )

        logger.debug(f"Response structure validation passed for {endpoint} endpoint: {len(data['items'])} items")

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        quota_cost: int,
    ) -> Dict[str, Any]:
        """Make API request with quota tracking, retry logic, and circuit breaker.

        Args:
            endpoint: API endpoint (e.g., "search", "videos")
            params: Query parameters
            quota_cost: Quota cost for this operation

        Returns:
            JSON response from API

        Raises:
            QuotaExceededError: If quota is exhausted
            InvalidCredentialsError: If API key is invalid (403)
            RateLimitError: If rate limited (429)
            APIError: On other API errors
        """
        # US-149-4: Check retry budget before making request
        if self._retry_budget.budget_exhausted():
            logger.warning(
                f"YouTube API retry budget exhausted, falling back to yt-dlp. "
                f"Stats: {self._retry_budget.get_stats()}"
            )
            raise YouTubeAPIQuotaExceededError(
                f"YouTube API retry budget exhausted, falling back to yt-dlp",
                endpoint=endpoint,
            )

        # Check circuit breaker before making request (US-152-4: pass endpoint for per-endpoint tracking)
        self._check_circuit_breaker(endpoint)
        self._check_quota(quota_cost)

        # US-152-8: Apply rate limiting before making request
        wait_time = self._rate_limiter.acquire(1)
        if wait_time > 0:
            logger.debug(f"Rate limiter: waited {wait_time:.3f}s for {endpoint} request")

        url = f"{YOUTUBE_API_BASE}/{endpoint}"
        params["key"] = self.current_api_key
        logger.debug(f"Using API key #{self._current_key_index + 1} for {endpoint}")

        # Check cache
        cache_key = f"{endpoint}:{urlencode(sorted(params.items()))}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {endpoint}")
            self._record_call_result(True, endpoint)
            return cached

        # US-152-12: Log API call details (INFO level for summary, DEBUG for details)
        # Log endpoint, parameters (excluding API key for security), and quota cost
        safe_params = {k: v for k, v in params.items() if k != 'key'}
        logger.info(
            f"YouTube API Request: endpoint={endpoint}, params={safe_params}, quota_cost={quota_cost}, "
            f"key_index={self._current_key_index + 1}"
        )
        # US-152-12: DEBUG level for detailed request info including masked key
        logger.debug(
            f"YouTube API Request DEBUG: url={url}, params_with_key={params}, "
            f"timeout={self.timeout}, max_retries={self.max_retries}"
        )

        # Retry logic with exponential backoff (using retry budget)
        last_error = None
        for attempt in range(self.max_retries):
            # US-149-4: Check retry budget before each attempt
            if self._retry_budget.budget_exhausted():
                logger.warning(
                    f"YouTube API retry budget exhausted during retries, falling back to yt-dlp. "
                    f"Stats: {self._retry_budget.get_stats()}"
                )
                raise YouTubeAPIQuotaExceededError(
                    f"YouTube API retry budget exhausted, falling back to yt-dlp",
                    endpoint=endpoint,
                )

            # US-149-4: Record attempt for retry budget
            self._retry_budget.record_attempt()

            try:
                # US-153-10: Track latency for the API call
                start_time = time.time()
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
                latency_ms = (time.time() - start_time) * 1000
                # Record latency for this endpoint
                self._metrics.record_latency(endpoint, latency_ms)
                # US-155-008: Record latency for adaptive rate limiting
                latency_seconds = latency_ms / 1000.0
                self._rate_limiter.record_latency(latency_seconds)

                # Handle specific error codes
                if response.status_code == 403:
                    error_data = response.json()
                    # US-158-008: Parse error response for detailed error information
                    parsed_error = parse_youtube_api_error_response(error_data)
                    error_msg = parsed_error.get("message", "")
                    error_reason = parsed_error.get("reason")
                    error_domain = parsed_error.get("domain")
                    error_code = parsed_error.get("code", 403)

                    # US-158-008: Get detailed error reason info for better handling
                    if error_reason:
                        reason_info = get_error_reason_info(error_reason)
                        logger.debug(
                            f"US-158-008: YouTube API error reason detected: "
                            f"reason={error_reason}, domain={error_domain}, "
                            f"error_type={reason_info.get('error_type')}, "
                            f"retryable={reason_info.get('retryable')}"
                        )
                        # Record error reason for metrics
                        self._metrics.record_error_category(reason_info.get("error_type", "unknown"))

                    # US-150-3: Record per-key error for health tracking
                    self._record_key_error(403)
                    if "quotaexceeded" in error_msg.lower() or "quota exceeded" in error_msg.lower():
                        self._add_quota(quota_cost, endpoint)  # Count the failed request too
                        self._record_call_result(False, endpoint)
                        # US-149-4: Record failure for retry budget
                        self._retry_budget.record_failure()
                        # Mark current key as exhausted (US-155-009)
                        self._exhausted_keys.add(self._current_key_index)
                        self._key_health_status[self._current_key_index] = "exhausted"
                        self._key_exhausted_at[self._current_key_index] = time.time()
                        # US-150-11: Enhanced quota exceeded error logging with Google Cloud Console link
                        # US-158-008: Include error reason and domain in log
                        logger.error(
                            f"API Quota Exceeded Error:\n"
                            f"  - Endpoint: {endpoint}\n"
                            f"  - Params: {params}\n"
                            f"  - Error: {error_msg}\n"
                            f"  - Error Code: {error_code}\n"
                            f"  - Error Reason: {error_reason or 'N/A'}\n"
                            f"  - Error Domain: {error_domain or 'N/A'}\n"
                            f"  - Current Key: #{self._current_key_index + 1}\n"
                            f"  - Action: Check Google Cloud Console (https://console.cloud.google.com/apis/dashboard) "
                            f"for quota usage. Quota resets at midnight PST."
                        )
                        # Try to rotate to next key
                        if self._rotate_to_next_key():
                            logger.info(f"Rotated to key #{self._current_key_index + 1}, retrying request")
                            # Retry with new key - don't raise, let loop continue
                            continue
                        raise YouTubeAPIQuotaExceededError(
                            f"All YouTube API keys quota exceeded: {error_msg}\n"
                            f"Suggested next steps: (1) Check Google Cloud Console for quota usage, "
                            f"(2) Request quota increase, (3) Enable yt-dlp fallback in config.yaml",
                            endpoint=endpoint,
                        )
                    # Invalid credentials - mark key as exhausted for this session (US-155-009)
                    # US-150-11: Enhanced 403 error logging with actionable guidance
                    # US-156-010: Add error code mapping for detailed error logging
                    # US-158-008: Include error reason and domain
                    error_desc = get_error_description(403)
                    error_category = get_error_category(403)
                    self._exhausted_keys.add(self._current_key_index)
                    self._key_health_status[self._current_key_index] = "exhausted"
                    self._key_exhausted_at[self._current_key_index] = time.time()
                    self._record_call_result(False, endpoint)
                    logger.error(
                        f"API 403 Forbidden Error:\n"
                        f"  - Endpoint: {endpoint}\n"
                        f"  - Params: {params}\n"
                        f"  - Error: {error_msg}\n"
                        f"  - Error Code: {error_code} ({error_category.get('description', 'Rate limit or invalid key')})\n"
                        f"  - Error Reason: {error_reason or 'N/A'}\n"
                        f"  - Error Domain: {error_domain or 'N/A'}\n"
                        f"  - Current Key: #{self._current_key_index + 1}\n"
                        f"  - Action: Check API key validity in Google Cloud Console. "
                        f"Ensure YouTube is enabled and key Data API v3 has correct restrictions."
                    )
                    # Try rotating to next key for invalid credentials too
                    if self._rotate_to_next_key():
                        logger.info(f"Rotated to key #{self._current_key_index + 1} after invalid credentials")
                        continue
                    # US-152-9: Raise YouTubeAPIPermissionDeniedError for 403 (not quota exceeded)
                    # US-155-010: Record error category for metrics
                    self._metrics.record_error_category("permission_denied")
                    raise YouTubeAPIPermissionDeniedError(
                        f"API access denied (HTTP 403): {error_msg}\n"
                        f"Suggested fixes: (1) Verify API key permissions in Google Cloud Console, "
                        f"(2) Enable YouTube Data API v3, (3) Check for API restrictions",
                        endpoint=endpoint,
                    )

                if response.status_code == 404:
                    self._record_call_result(False, endpoint)
                    # US-149-4: Record failure for retry budget
                    self._retry_budget.record_failure()
                    logger.error(
                        f"API Error: endpoint={endpoint}, params={params}, "
                        f"status_code=404"
                    )
                    raise YouTubeAPIError(
                        f"Resource not found: {endpoint}",
                        error_type=YouTubeAPIError.TYPE_PERMISSION_DENIED,
                        endpoint=endpoint,
                    )

                if response.status_code == 429:
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    retry_after_value = float(retry_after) if retry_after else None
                    # US-150-3: Record per-key error for health tracking
                    self._record_key_error(429)
                    self._record_call_result(False, endpoint)
                    # US-149-4: Record failure for retry budget
                    self._retry_budget.record_failure()
                    # US-155-010: Record error category for metrics
                    self._metrics.record_error_category("rate_limited")
                    # US-150-11: Enhanced 429 error logging with actionable guidance
                    # US-156-010: Add error code mapping for detailed error logging
                    error_desc = get_error_description(403)  # 403 maps to rate limit in our mapping
                    error_category = get_error_category(403)
                    logger.error(
                        f"API 429 Rate Limited Error:\n"
                        f"  - Endpoint: {endpoint}\n"
                        f"  - Params: {params}\n"
                        f"  - Retry-After: {retry_after_value}s\n"
                        f"  - Current Key: #{self._current_key_index + 1}\n"
                        f"  - Error Code: 403 ({error_category.get('description', 'Rate limit exceeded')})\n"
                        f"  - Action: Wait before retrying. Consider reducing request frequency "
                        f"or enabling yt-dlp fallback in config.yaml."
                    )
                    raise YouTubeAPIRateLimitedError(
                        f"Rate limited by YouTube API (HTTP 429): wait {retry_after_value}s before retry\n"
                        f"Suggested next steps: (1) Wait 60-100 seconds, "
                        f"(2) Reduce request frequency, (3) Enable yt-dlp fallback",
                        endpoint=endpoint,
                        retry_after=retry_after_value,
                    )

                if response.status_code >= 500:
                    # US-149-7: Use YouTubeAPITemporaryError for 5xx errors
                    error_msg = f"Server error: {response.status_code}"
                    # US-149-7: Reduce retries for 5xx errors - fail fast after fewer attempts
                    temp_error_retries = min(2, self.max_retries)
                    if attempt >= temp_error_retries:
                        logger.warning(
                            f"Too many 5xx errors ({temp_error_retries} retries), falling back to yt-dlp"
                        )
                        raise YouTubeAPITemporaryError(
                            error_msg,
                            status_code=response.status_code,
                            endpoint=endpoint,
                        )
                    # US-149-4: Record failure for retry budget
                    self._retry_budget.record_failure()
                    # Use retry budget's backoff calculation
                    wait_time = self._retry_budget.get_backoff_time(attempt, self.retry_delay)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries}: {error_msg}, "
                        f"endpoint={endpoint}, waiting {wait_time}s"
                    )
                    # US-149-4: Record backoff time in retry budget
                    self._retry_budget.record_backoff(wait_time)
                    # Check budget after backoff
                    if self._retry_budget.budget_exhausted():
                        logger.warning(
                            f"YouTube API retry budget exhausted after backoff, falling back to yt-dlp. "
                            f"Stats: {self._retry_budget.get_stats()}"
                        )
                        raise YouTubeAPIQuotaError(
                            f"YouTube API retry budget exhausted, falling back to yt-dlp",
                            endpoint=endpoint,
                        )
                    continue

                response.raise_for_status()
                data = response.json()

                # US-152-12: Log quota usage after successful call with running total
                # Get current quota usage for logging
                current_quota = self._key_quota_used[self._current_key_index]
                quota_limit = self._total_quota_limit
                quota_percent = (current_quota / quota_limit * 100) if quota_limit > 0 else 0
                logger.info(
                    f"YouTube API Response SUCCESS: endpoint={endpoint}, quota_cost={quota_cost}, "
                    f"quota_used={current_quota}/{quota_limit} ({quota_percent:.1f}%)"
                )

                # US-152-12: Check for request ID in response headers (YouTube API doesn't typically include this, but some Google APIs do)
                request_id = response.headers.get("X-Request-Id") or response.headers.get("X-GUploader-UploadID")
                if request_id:
                    logger.debug(f"YouTube API Request ID: {request_id}")

                # US-152-12: DEBUG level for response details
                logger.debug(
                    f"YouTube API Response DEBUG: endpoint={endpoint}, status={response.status_code}, "
                    f"response_keys={list(data.keys())}, request_id={request_id}"
                )

                self._add_quota(quota_cost, endpoint)
                self._set_cache(cache_key, data)
                self._record_call_result(True, endpoint)
                # US-146-12: Record successful API call
                self._metrics.record_api_call(endpoint, success=True)
                # US-149-4: Record success for retry budget
                self._retry_budget.record_success()
                # US-150-3: Record key success for health tracking
                self._record_key_success()
                # US-157-9: Track prediction accuracy - if backoff was applied but no rate limit occurred
                if self._last_backoff_applied:
                    self._unnecessary_backoffs += 1
                    if self._predictor:
                        self._predictor.record_actual_result(
                            predicted_rate_limit=True,
                            actual_rate_limit=False
                        )
                    logger.info(
                        f"Unnecessary backoff detected: predicted rate limit but got success. "
                        f"Total unnecessary: {self._unnecessary_backoffs}"
                    )
                    self._last_backoff_applied = False
                return data

            except (YouTubeAPIQuotaExceededError, YouTubeAPIInvalidKeyError, YouTubeAPIRateLimitedError, YouTubeAPITemporaryError, YouTubeAPIQuotaError, YouTubeAPIError):
                # Re-raise YouTube API errors without wrapping
                raise
            except requests.exceptions.Timeout:
                # US-149-4: Record failure for retry budget
                self._retry_budget.record_failure()
                last_error = "Request timeout"
                # US-155-010: Record error category for metrics
                self._metrics.record_error_category("timeout")
                # Use retry budget's backoff calculation
                wait_time = self._retry_budget.get_backoff_time(attempt, self.retry_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries}: {last_error}, "
                    f"endpoint={endpoint}, waiting {wait_time}s"
                )
                # US-149-4: Record backoff time in retry budget
                self._retry_budget.record_backoff(wait_time)
            except requests.exceptions.RequestException as e:
                # US-149-4: Record failure for retry budget
                self._retry_budget.record_failure()
                last_error = str(e)
                # US-155-010: Record error category for metrics (network error)
                self._metrics.record_error_category("network_error")
                # Use retry budget's backoff calculation
                wait_time = self._retry_budget.get_backoff_time(attempt, self.retry_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries}: {last_error}, "
                    f"endpoint={endpoint}, waiting {wait_time}s"
                )
                # US-149-4: Record backoff time in retry budget
                self._retry_budget.record_backoff(wait_time)

        self._record_call_result(False, endpoint)
        # US-149-4: Record failure for retry budget when retries exhausted
        self._retry_budget.record_failure()
        # US-146-12: Record failed API call
        self._metrics.record_api_call(endpoint, success=False)
        # US-149-4: Check budget after all retries exhausted
        if self._retry_budget.budget_exhausted():
            logger.warning(
                f"YouTube API retry budget exhausted after all retries, falling back to yt-dlp. "
                f"Stats: {self._retry_budget.get_stats()}"
            )
            raise YouTubeAPIQuotaError(
                f"YouTube API retry budget exhausted, falling back to yt-dlp",
                endpoint=endpoint,
            )
        logger.error(
            f"API Error: endpoint={endpoint}, params={params}, "
            f"status_code=0, message=Failed after {self.max_retries} retries"
        )
        raise YouTubeAPINetworkError(
            f"Failed after {self.max_retries} retries: {last_error}",
            endpoint=endpoint,
        )

    def search_videos(
        self,
        query: str,
        max_results: int = 50,
        video_type: str = "video",
        max_total_results: int = 10000,
        published_after: str = "",
        published_before: str = "",
        video_category_id: str = "",
        order: str = "relevance",
        video_duration: str = "",
        region_code: str = "",
        safe_search: str = "",
    ) -> List[VideoSearchResult]:
        """Search for videos using YouTube Data API with pagination.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (API returns max 50 per call)
            video_type: Type of results to return (default: "video")
            max_total_results: Maximum total results to allow (default: 10000, YouTube API limit)
            published_after: Filter videos published after this date
                - ISO 8601 format: "2024-01-01T00:00:00Z"
                - Relative: "last_30_days", "last_year", "last_2_years"
            published_before: Filter videos published before this date
                - ISO 8601 format: "2024-12-31T23:59:59Z"
                - Relative: "last_month"
            video_category_id: Filter by YouTube video category ID
                - Common IDs: 1=Film/Animation, 2=Autos, 10=Music, 15=Pets/Animals,
                  17=Sports, 20=Gaming, 22=People/Blogs, 23=Comedy, 24=Entertainment,
                  25=News/Politics, 26=Howto/Style, 27=Education, 28=Science/Technology
            order: Search results ordering (default: "relevance")
                - "relevance": Most relevant results
                - "date": Most recently published
                - "viewCount": Highest view count
                - "rating": Highest rating
                - "videoCount": Channel with most videos
                Invalid values will fall back to "relevance"
            video_duration: Filter videos by duration (default: client setting)
                - "": Use client default (self._video_duration)
                - "any": No duration filter
                - "short": Videos less than 4 minutes
                - "medium": Videos between 4 and 20 minutes
                - "long": Videos longer than 20 minutes
                Invalid values will fall back to client default
            region_code: Filter videos by region (default: client setting)
                - "": Use client default (self._region_code)
                - ISO 3166-1 alpha-2: US, GB, DE, JP, etc.
                - Empty string uses YouTube default (no region filter)
                Invalid values will fall back to client default
            safe_search: Filter explicit content (default: client setting)
                - "": Use client default (self._safe_search)
                - "none": No content filtering
                - "moderate": Some explicit content filtered (default)
                - "strict": Most explicit content filtered
                Invalid values will fall back to client default

        Returns:
            List of VideoSearchResult objects

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        # US-158-002: Validate order parameter with fallback to client default or relevance
        valid_orders = ("relevance", "date", "viewCount", "rating", "videoCount")
        if order not in valid_orders:
            logger.warning(
                f"US-158-002: Invalid order '{order}' in search_videos, "
                f"falling back to client default '{self._order_by}'"
            )
            order = self._order_by
        else:
            logger.debug(f"US-158-002: Using order '{order}' for query '{query}'")

        # US-158-003: Validate video_duration parameter with fallback to client default
        valid_durations = ("any", "short", "medium", "long")
        if video_duration and video_duration not in valid_durations:
            logger.warning(
                f"US-158-003: Invalid video_duration '{video_duration}' in search_videos, "
                f"falling back to client default '{self._video_duration}'"
            )
            video_duration = ""
        elif video_duration:
            logger.debug(f"US-158-003: Using video_duration '{video_duration}' for query '{query}'")

        # US-158-004: Validate region_code parameter with fallback to client default
        effective_region_code = region_code if region_code else self._region_code
        if effective_region_code and len(effective_region_code) != 2:
            logger.warning(
                f"US-158-004: Invalid region_code '{region_code}' in search_videos, "
                f"falling back to client default '{self._region_code}'"
            )
            effective_region_code = self._region_code
        elif effective_region_code:
            logger.debug(f"US-158-004: Using region_code '{effective_region_code}' for query '{query}'")

        # US-158-005: Validate safe_search parameter with fallback to client default
        effective_safe_search = safe_search if safe_search else self._safe_search
        valid_safe_search = ("none", "moderate", "strict")
        if effective_safe_search not in valid_safe_search:
            logger.warning(
                f"US-158-005: Invalid safe_search '{safe_search}' in search_videos, "
                f"falling back to client default '{self._safe_search}'"
            )
            effective_safe_search = self._safe_search
        else:
            logger.debug(f"US-158-005: Using safe_search '{effective_safe_search}' for query '{query}'")

        # US-156-005: Sanitize and check for duplicate queries
        if self._deduplicate_searches_enabled and query:
            # Check if this normalized query was recently searched
            normalized = self.query_normalize(query)
            if normalized in self._recent_queries:
                logger.info(f"US-156-005: Skipping recent query (already searched): {query}")
                return []
            # Sanitize the query
            query = self.sanitize_query(query)
            if not query:
                logger.warning(f"US-156-005: Query sanitized to empty: {query}")
                return []
            # Track this query
            self._recent_queries.add(normalized)

        # US-155-4: Parse relative date strings
        date_range_info = {}
        parsed_published_after = self._parse_date_parameter(published_after) if published_after else None
        parsed_published_before = self._parse_date_parameter(published_before) if published_before else None

        if parsed_published_after or parsed_published_before:
            date_range_info = {
                "published_after": parsed_published_after,
                "published_before": parsed_published_before,
            }
            logger.info(f"US-155-4: Date range filter: after={parsed_published_after}, before={parsed_published_before}")

        # US-155-005: Validate video category ID
        category_info = {}
        if video_category_id:
            # Validate that it's a numeric string
            if not video_category_id.isdigit():
                logger.warning(f"US-155-005: Invalid video_category_id '{video_category_id}': must be numeric. Ignoring filter.")
                video_category_id = ""
            else:
                category_info = {"video_category_id": video_category_id}
                logger.info(f"US-155-005: Video category filter: {video_category_id}")

        # US-158-002: Validate search ordering parameter
        valid_orders = {"relevance", "date", "viewCount", "rating", "videoCount"}
        if order not in valid_orders:
            logger.warning(f"US-158-002: Invalid order '{order}': must be one of {valid_orders}. Falling back to 'relevance'.")
            order = "relevance"
        else:
            logger.info(f"US-158-002: Search ordering: {order}")

        # US-154-12: Mock mode - return mock search results
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock search results for query: {query}")
            return self._get_mock_search_results(query, max_results, video_category_id)

        # US-156-005: Sanitize query before use
        original_query = query
        query = self.sanitize_query(query)
        if query != original_query:
            logger.debug(f"US-156-005: Sanitized query from '{original_query}' to '{query}'")

        # US-156-005: Check if this query was recently executed
        if self._deduplicate_searches and self.is_query_recent(query):
            logger.info(
                f"US-156-005: Skipping duplicate query '{query}' - recently executed in this session"
            )
            return []  # Return empty to avoid duplicate API call

        # US-155-008: In-session request deduplication
        # Check if we've already made this exact request in this session
        dedupe_hash = self._get_deduplication_hash(
            query, max_results, published_after, published_before, video_category_id
        )
        if dedupe_hash in self._deduplication_cache:
            cached_results = self._deduplication_cache[dedupe_hash]
            self._metrics.deduplicated_requests += 1
            logger.info(
                f"US-155-008: Deduplicated search request for '{query}' "
                f"(cached {len(cached_results)} results, total deduplicated: {self._metrics.deduplicated_requests})"
            )
            return cached_results

        # Cap max_results at max_total_results
        max_results = min(max_results, max_total_results)
        # US-149-9: Check SQLite cache first
        if self._query_cache is not None:
            cached = self._query_cache.get(query, max_results, video_type)
            if cached is not None:
                self._metrics.cache_hits += 1
                # Convert cached dicts back to VideoSearchResult objects
                results = [
                    VideoSearchResult(
                        video_id=v['video_id'],
                        title=v['title'],
                        channel_id=v['channel_id'],
                        channel_title=v['channel_title'],
                        published_at=v['published_at'],
                        description=v.get('description', ''),
                        thumbnail_url=v.get('thumbnail_url', ''),
                        view_count=v.get('view_count', 0),
                        subscriber_count=v.get('subscriber_count', 0),
                        total_views=v.get('total_views', 0),
                        channel_created_date=v.get('channel_created_date', ''),
                    )
                    for v in cached
                ]
                logger.info(f"YouTube API cache HIT for '{query}': {len(results)} results")
                return results
            self._metrics.cache_misses += 1

        # US-156-011: Check for cached page token to resume interrupted pagination
        cached_token_info = self.get_cached_page_token(query)
        results = []
        page_token = None
        initial_results_count = 0

        if cached_token_info:
            # Resume from cached page token
            page_token, initial_results_count = cached_token_info
            logger.info(
                f"US-156-011: Resuming pagination for '{query}' "
                f"from page token with {initial_results_count} results already fetched"
            )
            # Track that we're resuming (for metrics)
            self._metrics.search_queries_with_pagination += 1

        remaining = max_results - initial_results_count
        page_num = 0
        total_pages = (max_results + 49) // 50  # Estimate total pages needed
        seen_page_tokens: set = set()  # US-154-10: Track seen page tokens to detect duplicates
        pagination_start_time = time.time()  # US-154-10: Track pagination time
        pagination_timeout = self._pagination_timeout  # US-154-10: Use instance timeout
        last_progress_update = time.time()  # For progress bar updates

        # US-154-10: Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False

        # US-154-10: Create progress bar for pagination
        if use_progress_bar:
            pbar = tqdm(total=total_pages, desc=f"Searching: {query[:30]}", unit="page", leave=False)
        else:
            pbar = None

        while remaining > 0:
            # US-154-10: Check pagination timeout to prevent infinite loops
            if time.time() - pagination_start_time > pagination_timeout:
                logger.warning(
                    f"YouTube API pagination timeout for '{query}': "
                    f"returning {len(results)} results after {page_num} pages. "
                    f"Timeout: {pagination_timeout}s exceeded."
                )
                self._metrics.search_page_timeouts += 1
                # US-156-011: Cache page token for potential resume
                if page_token:
                    self._cache_page_token(query, page_token, len(results))
                break

            # US-156-011: Check max_pages_per_query limit
            # Only check after first page (page_num > 0) to allow at least 1 page
            if page_num > 0 and page_num >= self._max_pages_per_query:
                logger.info(
                    f"YouTube API max pages reached for '{query}': "
                    f"returning {len(results)} results after {page_num} pages. "
                    f"Limit: {self._max_pages_per_query} pages."
                )
                # Cache page token for potential resume
                if page_token:
                    self._cache_page_token(query, page_token, len(results))
                break

            # US-157-011: Calculate results for this page (using configurable results_per_page)
            page_size = min(remaining, self._results_per_page)
            page_num += 1

            params = {
                "part": "snippet",
                "q": query,
                "type": video_type,
                "maxResults": page_size,
                "order": order,  # US-158-002: Use configured ordering
            }

            # US-155-4: Add date range filters to API request
            if parsed_published_after:
                params["publishedAfter"] = parsed_published_after
            if parsed_published_before:
                params["publishedBefore"] = parsed_published_before

            # US-155-005: Add video category filter to API request
            if video_category_id:
                params["videoCategoryId"] = video_category_id

            # US-158-003: Add video duration filter to API request
            # Use method parameter if provided, otherwise fall back to client default
            effective_duration = video_duration if video_duration else self._video_duration
            if effective_duration and effective_duration != "any":
                params["videoDuration"] = effective_duration
                logger.debug(f"US-158-003: Applying videoDuration filter: {effective_duration}")

            # US-158-004: Add region code filter to API request
            # Use method parameter if provided, otherwise fall back to client default
            if effective_region_code:
                params["regionCode"] = effective_region_code
                logger.debug(f"US-158-004: Applying regionCode filter: {effective_region_code}")

            # US-158-005: Add safe search filter to API request
            # Use method parameter if provided, otherwise fall back to client default
            if effective_safe_search and effective_safe_search != "moderate":
                # Only add to request if not default "moderate" (YouTube defaults to moderate)
                params["safeSearch"] = effective_safe_search
                logger.debug(f"US-158-005: Applying safeSearch filter: {effective_safe_search}")
            elif effective_safe_search:
                logger.debug(f"US-158-005: Using default safeSearch level: {effective_safe_search}")

            # Add page token if we have one
            if page_token:
                params["pageToken"] = page_token

            # US-150-9: Progress indicator for paginated searches
            total_results = len(results) + initial_results_count
            logger.info(f"YouTube API search '{query}': Fetching page {page_num}/{total_pages} ({total_results} results so far)")

            # US-150-9: Handle quota limits gracefully during pagination
            try:
                data = self._make_request("search", params, QUOTA_COST_SEARCH)
            except (YouTubeAPIQuotaExceededError, QuotaExceededError) as e:
                logger.warning(
                    f"YouTube API quota exhausted during pagination for '{query}': "
                    f"returning {len(results)} results collected so far. "
                    f"Quota error: {e}"
                )
                # Return partial results instead of failing completely
                break

            # US-156-002: Validate response structure before processing
            try:
                self.validate_response_structure(data, "search")
            except YouTubeAPIError as e:
                logger.error(
                    f"YouTube API search '{query}': Invalid response structure on page {page_num}, "
                    f"stopping pagination. Error: {e}"
                )
                # US-156-003: Invalidate cache on invalid response (stale data indicator)
                if self._auto_invalidate_on_error and self._query_cache is not None:
                    cache_key = self._query_cache._make_key(query, max_results, video_type)
                    self._query_cache.invalidate(cache_key)
                    logger.info(f"US-156-003: Cache invalidated for query '{query}' due to invalid response structure")
                break

            # US-154-10: Handle empty result pages
            page_results = []
            if not data.get("items"):
                logger.warning(
                    f"YouTube API search '{query}': Empty page {page_num} received, stopping pagination."
                )
                # US-156-003: Invalidate cache on empty results (stale data indicator)
                if self._auto_invalidate_on_error and self._query_cache is not None:
                    cache_key = self._query_cache._make_key(query, max_results, video_type)
                    self._query_cache.invalidate(cache_key)
                    logger.info(f"US-156-003: Cache invalidated for query '{query}' due to empty results")
                break

            for item in data.get("items", []):
                if item.get("id", {}).get("kind") != "youtube#video":
                    continue

                snippet = item.get("snippet", {})
                thumbnails = snippet.get("thumbnails", {})

                # Get best available thumbnail
                thumbnail_url = ""
                for size in ("high", "medium", "default"):
                    if size in thumbnails:
                        thumbnail_url = thumbnails[size].get("url", "")
                        break

                page_results.append(VideoSearchResult(
                    video_id=item["id"]["videoId"],
                    title=snippet.get("title", ""),
                    channel_id=snippet.get("channelId", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    description=snippet.get("description", ""),
                    thumbnail_url=thumbnail_url,
                    search_date_range=date_range_info if date_range_info else None,
                    video_category_id=video_category_id if video_category_id else None,
                ))

            results.extend(page_results)
            remaining -= len(page_results)

            # US-154-10: Update progress bar
            if pbar:
                pbar.update(1)

            # Check for next page token
            page_token = data.get("nextPageToken")

            # US-156-011: Track pagination state for each page
            # Get results count from this page
            page_results_count = len(page_results)
            if page_token:
                self.track_pagination_state(query, page_token, page_results_count)

            # US-154-10: Detect and handle duplicate pageToken
            if page_token:
                if page_token in seen_page_tokens:
                    logger.warning(
                        f"YouTube API search '{query}': Duplicate pageToken detected, stopping pagination. "
                        f"Token: {page_token[:20]}..."
                    )
                    self._metrics.duplicate_page_tokens += 1
                    break
                seen_page_tokens.add(page_token)

            if not page_token:
                # No more pages available
                break

        # US-154-10: Close progress bar
        if pbar:
            pbar.close()

        # US-154-10: Track pagination metrics
        if page_num > 1:
            self._metrics.search_queries_with_pagination += 1
        self._metrics.search_pages_total += page_num

        # US-156-011: Include initial results count if resuming
        total_results = len(results) + initial_results_count
        logger.info(f"YouTube API search '{query}': {total_results} results in {page_num} pages (key #{self._current_key_index + 1}, quota: {self._key_quota_used[self._current_key_index]}/{self._total_quota_limit})")

        # US-155-004: Handle empty results gracefully when date range is too narrow
        if not results and (parsed_published_after or parsed_published_before):
            logger.warning(
                f"YouTube API search '{query}': No results found with date range filter. "
                f"published_after={parsed_published_after}, published_before={parsed_published_before}. "
                f"Try broadening the date range or removing filters."
            )

        # US-156-003: Auto-invalidate cache on consecutive empty/stale results
        if self._auto_invalidate_on_error and self._query_cache is not None:
            cache_key = self._query_cache._make_key(query, max_results, video_type)
            if not results:
                # Track consecutive empty results
                self._consecutive_empty_results[cache_key] = self._consecutive_empty_results.get(cache_key, 0) + 1
                if self._consecutive_empty_results[cache_key] >= self._auto_invalidate_threshold:
                    # Invalidate the cache entry
                    self._query_cache.invalidate(cache_key)
                    logger.info(f"US-156-003: Auto-invalidated cache for query '{query}' after {self._consecutive_empty_results[cache_key]} consecutive empty results")
                    self._consecutive_empty_results[cache_key] = 0
            else:
                # Reset counter on successful results
                self._consecutive_empty_results[cache_key] = 0

        # US-149-9: Cache the results for future use
        if self._query_cache is not None and results:
            # Convert results to dicts for caching
            results_dicts = [
                {
                    'video_id': r.video_id,
                    'title': r.title,
                    'channel_id': r.channel_id,
                    'channel_title': r.channel_title,
                    'published_at': r.published_at,
                    'description': r.description,
                    'thumbnail_url': r.thumbnail_url,
                    'view_count': r.view_count,
                    'subscriber_count': r.subscriber_count,
                    'total_views': r.total_views,
                    'channel_created_date': r.channel_created_date,
                }
                for r in results
            ]
            # US-158-012: Pass TTL in seconds if configured
            ttl_seconds = self._cache_ttl_seconds if self._cache_ttl_seconds > 0 else None
            self._query_cache.set(query, max_results, video_type, results_dicts, ttl_seconds=ttl_seconds)

        # US-155-008: Cache results for in-session deduplication
        # US-156-011: Include initial results if resuming
        if initial_results_count > 0:
            # Note: We can't return the initial results here since they're not stored
            # The caller is responsible for tracking total results when resuming
            logger.info(
                f"US-156-011: Resume complete for '{query}': "
                f"{len(results)} new results fetched (started from {initial_results_count} previous results)"
            )

        # Store before returning (don't affect quota - this is just caching our own results)
        self._deduplication_cache[dedupe_hash] = results
        logger.debug(f"US-155-008: Cached {len(results)} results for dedupe hash {dedupe_hash[:8]}...")

        # US-156-005: Mark query as executed for session-level deduplication
        self.mark_query_executed(query)

        # US-156-011: Clear page token cache after successful completion
        if initial_results_count > 0:
            self.clear_page_token_cache(query)

        # US-157-011: Track pagination efficiency metrics
        self._metrics.pagination_results_requested += max_results
        self._metrics.pagination_results_returned += len(results)

        return results

    async def async_search_videos(
        self,
        query: str,
        max_results: int = 50,
        video_type: str = "video",
        max_total_results: int = 10000,
        published_after: str = "",
        published_before: str = "",
        video_category_id: str = "",
        order: str = "relevance",
        video_duration: str = "",
    ) -> List[VideoSearchResult]:
        """Search for videos using YouTube Data API (async version).

        Provides the same functionality as search_videos but with async/await
        support for better concurrency. Uses semaphore for rate limiting.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (API returns max 50 per call)
            video_type: Type of results to return (default: "video")
            max_total_results: Maximum total results to allow (default: 10000, YouTube API limit)
            published_after: Filter videos published after this date (ISO 8601 or relative)
            published_before: Filter videos published before this date (ISO 8601 or relative)
            video_category_id: Filter by YouTube video category ID
            order: Search results ordering (default: "relevance")
                - "relevance": Most relevant results
                - "date": Most recently published
                - "viewCount": Highest view count
                - "rating": Highest rating
                - "videoCount": Channel with most videos

        Returns:
            List of VideoSearchResult objects

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        # US-154-12: Mock mode - return mock search results
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock search results for query: {query}")
            return self._get_mock_search_results(query, max_results, video_category_id)

        # US-158-002: Validate search ordering parameter
        valid_orders = {"relevance", "date", "viewCount", "rating", "videoCount"}
        if order not in valid_orders:
            logger.warning(f"US-158-002: Invalid order '{order}': must be one of {valid_orders}. Falling back to 'relevance'.")
            order = "relevance"
        else:
            logger.info(f"US-158-002: Search ordering: {order}")

        # US-156-005: Sanitize query before use
        original_query = query
        query = self.sanitize_query(query)
        if query != original_query:
            logger.debug(f"US-156-005: Sanitized query from '{original_query}' to '{query}'")

        # Check cache
        if self._query_cache is not None:
            cached = self._query_cache.get(query, max_results, video_type)
            if cached is not None:
                self._metrics.cache_hits += 1
                results = [
                    VideoSearchResult(
                        video_id=v['video_id'],
                        title=v['title'],
                        channel_id=v['channel_id'],
                        channel_title=v['channel_title'],
                        published_at=v['published_at'],
                        description=v.get('description', ''),
                        thumbnail_url=v.get('thumbnail_url', ''),
                        view_count=v.get('view_count', 0),
                        subscriber_count=v.get('subscriber_count', 0),
                        total_views=v.get('total_views', 0),
                        channel_created_date=v.get('channel_created_date', ''),
                    )
                    for v in cached
                ]
                logger.info(f"YouTube API cache HIT for '{query}': {len(results)} results")
                return results
            self._metrics.cache_misses += 1

        # Get semaphore for concurrency limiting
        semaphore = self._get_semaphore()

        results: List[VideoSearchResult] = []
        page_token = None
        remaining = max_results

        while remaining > 0:
            # US-157-011: Build request parameters with configurable results_per_page
            params: Dict[str, Any] = {
                "part": "snippet",
                "q": query,
                "type": video_type,
                "maxResults": min(self._results_per_page, remaining),
                "order": order,  # US-158-002: Use configured ordering
            }

            if page_token:
                params["pageToken"] = page_token

            if published_after:
                parsed = self._parse_date_parameter(published_after)
                if parsed:
                    params["publishedAfter"] = parsed

            if published_before:
                parsed = self._parse_date_parameter(published_before)
                if parsed:
                    params["publishedBefore"] = parsed

            if video_category_id:
                params["videoCategoryId"] = video_category_id

            # US-158-003: Add video duration filter to API request
            effective_duration = video_duration if video_duration else self._video_duration
            if effective_duration and effective_duration != "any":
                params["videoDuration"] = effective_duration
                logger.debug(f"US-158-003: Applying videoDuration filter: {effective_duration}")

            # US-158-004: Add region code filter to API request
            if effective_region_code:
                params["regionCode"] = effective_region_code
                logger.debug(f"US-158-004: Applying regionCode filter: {effective_region_code}")

            try:
                # Make async request
                data = await self._make_async_request(
                    "search",
                    params,
                    QUOTA_COST_SEARCH,
                    semaphore,
                )

                # US-156-002: Validate response structure before processing
                self.validate_response_structure(data, "search")

                items = data.get("items", [])
                if not items:
                    break

                for item in items:
                    if item.get("id", {}).get("kind") == "youtube#video":
                        snippet = item.get("snippet", {})
                        results.append(VideoSearchResult(
                            video_id=item["id"]["videoId"],
                            title=snippet.get("title", ""),
                            channel_id=snippet.get("channelId", ""),
                            channel_title=snippet.get("channelTitle", ""),
                            published_at=snippet.get("publishedAt", ""),
                            description=snippet.get("description", ""),
                            thumbnail_url=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                        ))

                # Check for next page
                page_token = data.get("nextPageToken")
                if not page_token:
                    break

                remaining = max_results - len(results)

            except YouTubeAPIError as e:
                logger.warning(f"Async search failed for '{query}': {e}")
                break

        # Cache results
        if results and self._query_cache is not None:
            cache_data = [
                {
                    'video_id': r.video_id,
                    'title': r.title,
                    'channel_id': r.channel_id,
                    'channel_title': r.channel_title,
                    'published_at': r.published_at,
                    'description': r.description,
                    'thumbnail_url': r.thumbnail_url,
                }
                for r in results
            ]
            # US-158-012: Pass TTL in seconds if configured
            ttl_seconds = self._cache_ttl_seconds if self._cache_ttl_seconds > 0 else None
            self._query_cache.set(query, max_results, video_type, cache_data, ttl_seconds=ttl_seconds)

        logger.info(f"Async search '{query}': {len(results)} results")

        # US-157-011: Track pagination efficiency metrics
        self._metrics.pagination_results_requested += max_results
        self._metrics.pagination_results_returned += len(results)

        return results

    # US-155-5: Helper method for parallel chunk fetching
    def _fetch_video_details_chunk(
        self,
        video_ids: List[str],
        part: str = "contentDetails,statistics,topicDetails",
        preferred_language: Optional[str] = None,
    ) -> Tuple[Dict[str, VideoDetails], List[str]]:
        """Fetch video details for a single chunk (used by parallel execution).

        Args:
            video_ids: List of YouTube video IDs (up to 50 per API call)
            part: Comma-separated list of parts to request
            preferred_language: Optional ISO 639-1 language code for localized metadata

        Returns:
            Tuple of (results dict, list of failed video IDs)
        """
        results: Dict[str, VideoDetails] = {}
        failed_video_ids: List[str] = []

        params: Dict[str, Any] = {
            "part": part,
            "id": ",".join(video_ids),
        }

        # US-157-003: Add preferred language for localized metadata
        if preferred_language:
            params["preferredLanguage"] = preferred_language

        try:
            data = self._make_request("videos", params, QUOTA_COST_VIDEOS)

            # US-156-002: Validate response structure before processing
            self.validate_response_structure(data, "videos")

            for item in data.get("items", []):
                try:
                    content_details = item.get("contentDetails", {})
                    statistics = item.get("statistics", {})
                    topic_details = item.get("topicDetails", {})

                    # Parse ISO 8601 duration to seconds
                    duration_str = content_details.get("duration", "PT0S")
                    duration_seconds = self._parse_duration(duration_str)

                    # Extract engagement metrics from statistics
                    view_count = int(statistics.get("viewCount", 0)) if statistics.get("viewCount") else 0
                    like_count = int(statistics.get("likeCount", 0)) if statistics.get("likeCount") else 0
                    comment_count = int(statistics.get("commentCount", 0)) if statistics.get("commentCount") else 0

                    # US-158-010: Calculate quality score based on engagement metrics
                    # Formula: viewCount * 0.7 + likeCount * 0.2 + commentCount * 0.1
                    quality_score = 0.0
                    if self._quality_boost_enabled:
                        quality_score = self.calculate_video_quality_score(
                            view_count, like_count, comment_count
                        )

                    # US-158-007: Extract channel info from snippet
                    snippet = item.get("snippet", {})
                    channel_id = snippet.get("channelId", "")
                    channel_title = snippet.get("channelTitle", "")

                    video_details = VideoDetails(
                        video_id=item["id"],
                        duration=duration_str,
                        duration_seconds=duration_seconds,
                        tags=content_details.get("tags", []),
                        category_id=content_details.get("categoryId", ""),
                        topic_details={
                            "topic_categories": topic_details.get("topicCategories", []),
                            "relevant_topic_ids": topic_details.get("relevantTopicIds", []),
                        },
                        topic_categories=topic_details.get("topicCategories", []),
                        caption_available=content_details.get("caption", "false") == "true",
                        dimension=content_details.get("dimension", ""),
                        definition=content_details.get("definition", ""),
                        view_count=view_count,
                        like_count=like_count,
                        comment_count=comment_count,
                        quality_score=quality_score,
                        channel_id=channel_id,
                        channel_title=channel_title,
                    )
                    results[item["id"]] = video_details
                except Exception as e:
                    # Gracefully handle individual video parsing failures
                    logger.warning(f"Failed to parse video details for {item.get('id', 'unknown')}: {e}")
                    failed_video_ids.append(item.get("id", "unknown"))

        except Exception as e:
            # Gracefully handle batch failure - mark all videos in batch as failed
            logger.warning(f"Batch request failed for videos {video_ids}: {e}")
            failed_video_ids.extend(video_ids)

        return results, failed_video_ids

    def get_video_details(
        self,
        video_ids: List[str],
        part: str = "contentDetails,statistics,topicDetails",
        preferred_language: Optional[str] = None,
    ) -> Tuple[Dict[str, VideoDetails], List[str]]:
        """Get detailed metadata for videos with partial failure handling.

        Args:
            video_ids: List of YouTube video IDs (up to 50 per API call)
            part: Comma-separated list of parts to request
            preferred_language: Optional ISO 639-1 language code for localized metadata

        Returns:
            Tuple of (Dict mapping video_id to VideoDetails objects, List of failed video IDs)

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        if not video_ids:
            return {}, []

        # US-154-12: Mock mode - return mock video details
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock video details for {len(video_ids)} videos")
            mock_data = self._get_mock_video_details(video_ids)
            results = {}
            for video_id, data in mock_data.items():
                # US-158-010: Calculate quality score based on engagement metrics
                view_count = int(data.get("statistics", {}).get("viewCount", 0))
                like_count = int(data.get("statistics", {}).get("likeCount", 0))
                comment_count = int(data.get("statistics", {}).get("commentCount", 0))
                quality_score = 0.0
                if self._quality_boost_enabled:
                    quality_score = self.calculate_video_quality_score(
                        view_count, like_count, comment_count
                    )

                # US-158-007: Extract channel info from snippet
                channel_id = data.get("snippet", {}).get("channelId", "")
                channel_title = data.get("snippet", {}).get("channelTitle", "")

                results[video_id] = VideoDetails(
                    video_id=video_id,
                    duration=data.get("contentDetails", {}).get("duration", "PT0S"),
                    duration_seconds=self._parse_duration(data.get("contentDetails", {}).get("duration", "PT0S")),
                    tags=data.get("contentDetails", {}).get("tags", []),
                    category_id=data.get("snippet", {}).get("categoryId", ""),
                    view_count=view_count,
                    like_count=like_count,
                    comment_count=comment_count,
                    quality_score=quality_score,
                    channel_id=channel_id,
                    channel_title=channel_title,
                )

            # US-158-007: Enrich mock data with channel subscriber counts
            if results:
                channel_ids = list(set(v.channel_id for v in results.values() if v.channel_id))
                if channel_ids:
                    channel_metadata = self.get_channel_metadata(channel_ids)
                    for video in results.values():
                        if video.channel_id and video.channel_id in channel_metadata:
                            channel = channel_metadata[video.channel_id]
                            video.subscriber_count = channel.get("subscriber_count", 0)

            # US-156-007: Mock mode returns no failures
            return results, []

        # US-153-5: Track batch operation timing
        batch_start_time = time.perf_counter()

        # API allows up to 50 video IDs per request
        results: Dict[str, VideoDetails] = {}
        total_api_calls = 0
        failed_video_ids: List[str] = []

        # US-155-005: Split into chunks based on configurable video_details_chunk_size
        chunk_size = self._video_details_chunk_size
        chunks = [video_ids[i:i + chunk_size] for i in range(0, len(video_ids), chunk_size)]
        num_chunks = len(chunks)

        # Determine if we should use parallel execution
        # Use parallel if: enabled, more than 1 chunk, and enough videos to benefit
        use_parallel = (
            self._parallel_video_details_enabled and
            num_chunks > 1 and
            len(video_ids) >= chunk_size
        )

        if use_parallel:
            # US-155-005: Parallel execution using ThreadPoolExecutor
            # Track sequential time for comparison
            sequential_start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=min(num_chunks, self._video_details_max_workers)) as executor:
                # Submit all chunk requests
                future_to_chunk = {
                    executor.submit(self._fetch_video_details_chunk, chunk, part, preferred_language): chunk
                    for chunk in chunks
                }

                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        chunk_results, chunk_failed = future.result()
                        results.update(chunk_results)
                        failed_video_ids.extend(chunk_failed)
                        total_api_calls += 1
                    except Exception as e:
                        # Gracefully handle chunk failure - mark all videos in chunk as failed
                        logger.warning(f"Chunk request failed for videos {chunk}: {e}")
                        failed_video_ids.extend(chunk)
                        total_api_calls += 1

            # Calculate time saved vs sequential
            sequential_elapsed = time.perf_counter() - sequential_start_time
            parallel_elapsed = time.perf_counter() - batch_start_time
            time_saved_ms = (sequential_elapsed - parallel_elapsed) * 1000

            # Update metrics
            self._metrics.parallel_batch_operations += 1
            self._metrics.parallel_batch_time_saved_ms += max(0, time_saved_ms)

            logger.debug(
                f"Parallel batch: {len(video_ids)} videos in {num_chunks} chunks, "
                f"parallel: {parallel_elapsed:.3f}s, sequential est: {sequential_elapsed:.3f}s, "
                f"saved: {time_saved_ms:.1f}ms"
            )
        else:
            # US-155-5: Sequential execution (fallback)
            self._metrics.sequential_batch_operations += 1

            for chunk in chunks:
                try:
                    chunk_results, chunk_failed = self._fetch_video_details_chunk(chunk, part, preferred_language)
                    results.update(chunk_results)
                    failed_video_ids.extend(chunk_failed)
                    total_api_calls += 1
                except Exception as e:
                    # Gracefully handle chunk failure
                    logger.warning(f"Chunk request failed for videos {chunk}: {e}")
                    failed_video_ids.extend(chunk)
                    total_api_calls += 1

        # US-152-6: Log batch efficiency savings
        individual_calls = len(video_ids)
        batch_calls = total_api_calls
        quota_saved = individual_calls - batch_calls

        # US-153-5: Log batch operation timing metrics
        batch_elapsed = time.perf_counter() - batch_start_time
        videos_per_second = len(video_ids) / batch_elapsed if batch_elapsed > 0 else 0

        if len(video_ids) > 1:
            logger.info(
                f"Batch video details: {len(video_ids)} videos fetched in {batch_calls} API call(s) "
                f"in {batch_elapsed:.2f}s ({videos_per_second:.1f} videos/sec) "
                f"(saved {quota_saved} quota units vs individual calls). "
                f"Failed: {len(failed_video_ids)}"
            )

        # US-158-007: Enrich video details with channel subscriber counts
        # Extract channel IDs from video results and fetch metadata
        if results:
            channel_ids = list(set(v.channel_id for v in results.values() if v.channel_id))
            if channel_ids:
                channel_metadata = self.get_channel_metadata(channel_ids)
                for video in results.values():
                    if video.channel_id and video.channel_id in channel_metadata:
                        channel = channel_metadata[video.channel_id]
                        video.subscriber_count = channel.get("subscriber_count", 0)

        # US-156-007: Return both results and failed_video_ids
        return results, failed_video_ids

    # =========================================================================
    # US-156-007: Batch operation retry logic for partial failures
    # =========================================================================

    def get_video_details_batch(
        self,
        video_ids: List[str],
        part: str = "contentDetails,statistics,topicDetails",
        max_retries: int = 2,
        preferred_language: Optional[str] = None,
    ) -> Tuple[Dict[str, VideoDetails], List[str]]:
        """Get detailed metadata for videos with partial failure handling and retry.

        This method wraps get_video_details() to provide:
        - Continues on individual video failures (returns partial results)
        - Returns failed_video_ids list for downstream handling
        - Logs warning when partial failures exceed threshold
        - Retries only failed video IDs in batch
        - Caches results with configurable TTL (default 24 hours)

        Args:
            video_ids: List of YouTube video IDs (up to 50 per API call)
            part: Comma-separated list of parts to request
            max_retries: Maximum retry attempts for failed video IDs
            preferred_language: Optional ISO 639-1 language code for localized metadata

        Returns:
            Tuple of (results dict mapping video_id to VideoDetails, list of failed video IDs)

        Raises:
            YouTubeAPIError: On API errors (after retries exhausted)
        """
        if not video_ids:
            return {}, []

        # US-157-003: Check cache first
        cache_results, _, uncached_ids = self._get_cached_video_details(
            video_ids, preferred_language
        )

        if not uncached_ids:
            # All videos were cached
            logger.debug(f"US-157-003: All {len(video_ids)} videos found in metadata cache")
            return cache_results, []

        # US-156-007: Initial fetch using existing get_video_details
        # This already handles chunk-level failures gracefully
        results, _ = self.get_video_details(uncached_ids, part, preferred_language)

        # US-157-003: Cache the newly fetched results
        if results:
            self._cache_video_details(results, preferred_language)

        # Merge cached results with newly fetched results
        all_results = {**cache_results, **results}

        # Identify failed video IDs
        successful_ids = set(all_results.keys())
        failed_video_ids = [vid for vid in video_ids if vid not in successful_ids]

        # Calculate failure percentage
        failure_percent = (len(failed_video_ids) / len(video_ids) * 100) if video_ids else 0

        # US-156-007: Log warning when partial failures exceed threshold
        if failure_percent > self._max_partial_failure_percent:
            logger.warning(
                f"US-156-007: Partial failure threshold exceeded: {failure_percent:.1f}% "
                f"({len(failed_video_ids)}/{len(video_ids)} videos failed) "
                f"exceeds threshold of {self._max_partial_failure_percent}%"
            )

        # US-156-007: Retry only failed video IDs
        retry_count = 0
        while failed_video_ids and retry_count < max_retries:
            retry_count += 1
            logger.info(
                f"US-156-007: Retrying {len(failed_video_ids)} failed videos "
                f"(attempt {retry_count}/{max_retries})"
            )

            # Retry fetch for failed videos
            retry_results, _ = self.get_video_details(failed_video_ids, part, preferred_language)

            # US-157-003: Cache retry results
            if retry_results:
                self._cache_video_details(retry_results, preferred_language)

            # Update results and recalculate failed
            results.update(retry_results)
            all_results = {**cache_results, **results}
            successful_ids = set(all_results.keys())
            failed_video_ids = [vid for vid in video_ids if vid not in successful_ids]

        # Final failure percentage after retries
        final_failure_percent = (len(failed_video_ids) / len(video_ids) * 100) if video_ids else 0

        # Log final status
        if failed_video_ids:
            logger.warning(
                f"US-157-003: Batch completed with {len(failed_video_ids)} failed videos "
                f"({final_failure_percent:.1f}% failure rate) after {retry_count} retry attempt(s)"
            )
        else:
            logger.info(
                f"US-157-003: Batch completed successfully: {len(all_results)}/{len(video_ids)} "
                f"videos fetched ({retry_count} retry attempt(s) made)"
            )

        return all_results, failed_video_ids

    # =========================================================================
    # US-149-8: Async batch enrichment methods
    # =========================================================================

    def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session for async requests.

        Returns:
            aiohttp.ClientSession instance
        """
        if not hasattr(self, '_async_session') or self._async_session is None or self._async_session.closed:
            # Create or get an event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop - create a new one for the session
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(loop=loop)
            self._async_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._async_session

    def _get_semaphore(self, max_concurrent: Optional[int] = None) -> asyncio.Semaphore:
        """Get or create a semaphore for concurrency limiting.

        Args:
            max_concurrent: Maximum concurrent requests (default: 10)

        Returns:
            asyncio.Semaphore instance
        """
        if max_concurrent is None:
            max_concurrent = self._max_concurrent_requests
        if self._async_semaphore is None or self._async_semaphore._value != max_concurrent:
            self._async_semaphore = asyncio.Semaphore(max_concurrent)
        return self._async_semaphore

    async def _make_async_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        quota_cost: int,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Make async API request with quota tracking and concurrency limiting.

        Args:
            endpoint: API endpoint (e.g., "search", "videos")
            params: Query parameters
            quota_cost: Quota cost for this operation
            semaphore: Semaphore for concurrency limiting

        Returns:
            JSON response from API

        Raises:
            YouTubeAPIQuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        async with semaphore:
            # Check quota before making request
            try:
                self._check_quota(quota_cost)
            except Exception as e:
                raise YouTubeAPIQuotaExceededError(
                    f"Quota exceeded: {e}",
                    endpoint=endpoint,
                ) from e

            # US-152-12: Log API call details for async requests
            safe_params = {k: v for k, v in params.items() if k != 'key'}
            logger.info(
                f"YouTube API Request (async): endpoint={endpoint}, params={safe_params}, "
                f"quota_cost={quota_cost}, key_index={self._current_key_index + 1}"
            )
            logger.debug(
                f"YouTube API Request DEBUG (async): url={url}, params_with_key={params}"
            )

            url = f"{YOUTUBE_API_BASE}/{endpoint}"
            params["key"] = self.current_api_key

            # Check cache
            cache_key = f"{endpoint}:{urlencode(sorted(params.items()))}"
            cached = self._get_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for async {endpoint}")
                self._record_call_result(True, endpoint)
                return cached

            session = self._get_async_session()

            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # US-152-12: Log quota usage after successful async call
                        current_quota = self._key_quota_used[self._current_key_index]
                        quota_limit = self._total_quota_limit
                        quota_percent = (current_quota / quota_limit * 100) if quota_limit > 0 else 0
                        logger.info(
                            f"YouTube API Response SUCCESS (async): endpoint={endpoint}, "
                            f"quota_cost={quota_cost}, quota_used={current_quota}/{quota_limit} ({quota_percent:.1f}%)"
                        )
                        logger.debug(
                            f"YouTube API Response DEBUG (async): endpoint={endpoint}, "
                            f"status={response.status}, response_keys={list(data.keys())}"
                        )

                        self._add_quota(quota_cost, endpoint)
                        self._record_call_result(True, endpoint)
                        # US-150-3: Record key success for health tracking
                        self._record_key_success()
                        self._set_cache(cache_key, data)
                        return data
                    elif response.status == 403:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", "Unknown error")
                        # US-150-3: Record per-key error for health tracking
                        self._record_key_error(403)
                        if "quotaExceeded" in error_msg or "exceeded" in error_msg.lower():
                            self._record_call_result(False, endpoint)
                            self._retry_budget.record_failure()
                            # US-155-009: Mark key as exhausted with health status and timestamp
                            self._exhausted_keys.add(self._current_key_index)
                            self._key_health_status[self._current_key_index] = "exhausted"
                            self._key_exhausted_at[self._current_key_index] = time.time()
                            if self._rotate_to_next_key():
                                logger.info(f"Rotated to key #{self._current_key_index + 1}, retrying request")
                                return await self._make_async_request(endpoint, params, quota_cost, semaphore)
                            raise YouTubeAPIQuotaExceededError(
                                f"All YouTube API keys quota exceeded",
                                endpoint=endpoint,
                            )
                        # US-155-010: Record error category for metrics
                        self._metrics.record_error_category("invalid_key")
                        raise YouTubeAPIInvalidKeyError(
                            f"API key invalid or insufficient permissions: {error_msg}",
                            endpoint=endpoint,
                        )
                    elif response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        retry_after_value = float(retry_after) if retry_after else None
                        # US-150-3: Record per-key error for health tracking
                        self._record_key_error(429)
                        self._record_call_result(False, endpoint)
                        self._retry_budget.record_failure()
                        # US-155-010: Record error category for metrics
                        self._metrics.record_error_category("rate_limited")
                        raise YouTubeAPIRateLimitedError(
                            f"Rate limited by YouTube API: {response.status}",
                            endpoint=endpoint,
                            retry_after=retry_after_value,
                        )
                    else:
                        raise YouTubeAPIError(
                            f"API request failed with status {response.status}",
                            endpoint=endpoint,
                        )
            except aiohttp.ClientError as e:
                self._record_call_result(False, endpoint)
                # US-155-010: Record error category for metrics
                self._metrics.record_error_category("network_error")
                raise YouTubeAPINetworkError(
                    f"Network error: {e}",
                    endpoint=endpoint,
                ) from e

    async def async_search_videos(
        self,
        query: str,
        max_results: int = 50,
        video_type: str = "video",
        max_concurrent: Optional[int] = None,
        max_total_results: int = 10000,
        enrich_with_channel_data: bool = False,
        order: str = "relevance",
    ) -> List[VideoSearchResult]:
        """Search for videos using YouTube Data API with async pagination.

        Uses aiohttp for concurrent requests with semaphore-based concurrency limiting.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (API returns max 50 per call)
            video_type: Type of results to return (default: "video")
            max_concurrent: Max concurrent requests (default: 10)
            max_total_results: Maximum total results to allow (default: 10000, YouTube API limit)
            order: Search results ordering (default: "relevance")
                - "relevance": Most relevant results
                - "date": Most recently published
                - "viewCount": Highest view count
                - "rating": Highest rating
                - "videoCount": Channel with most videos

        Returns:
            List of VideoSearchResult objects

        Raises:
            YouTubeAPIQuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        # US-158-002: Validate search ordering parameter
        valid_orders = {"relevance", "date", "viewCount", "rating", "videoCount"}
        if order not in valid_orders:
            logger.warning(f"US-158-002: Invalid order '{order}': must be one of {valid_orders}. Falling back to 'relevance'.")
            order = "relevance"
        else:
            logger.info(f"US-158-002: Search ordering: {order}")

        # Cap max_results at max_total_results
        max_results = min(max_results, max_total_results)

        results = []
        semaphore = self._get_semaphore(max_concurrent)

        # For search, we typically only need 1 request since results come in one page
        # But we structure it to be parallel-ready for future multi-query scenarios

        async def fetch_page(page_token: Optional[str] = None) -> Dict[str, Any]:
            # US-157-011: Use configurable results_per_page
            params = {
                "part": "snippet",
                "q": query,
                "type": video_type,
                "maxResults": min(self._results_per_page, max_results),
                "order": order,  # US-158-002: Use configured ordering
            }
            if page_token:
                params["pageToken"] = page_token

            return await self._make_async_request("search", params, QUOTA_COST_SEARCH, semaphore)

        # Fetch first page
        data = await fetch_page()

        for item in data.get("items", []):
            if item.get("id", {}).get("kind") != "youtube#video":
                continue

            snippet = item.get("snippet", {})
            thumbnails = snippet.get("thumbnails", {})

            thumbnail_url = ""
            for size in ("high", "medium", "default"):
                if size in thumbnails:
                    thumbnail_url = thumbnails[size].get("url", "")
                    break

            results.append(VideoSearchResult(
                video_id=item["id"]["videoId"],
                title=snippet.get("title", ""),
                channel_id=snippet.get("channelId", ""),
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
                description=snippet.get("description", ""),
                thumbnail_url=thumbnail_url,
            ))

        # Handle pagination if needed
        remaining = max_results - len(results)
        page_token = data.get("nextPageToken")

        while remaining > 0 and page_token:
            data = await fetch_page(page_token)

            for item in data.get("items", []):
                if item.get("id", {}).get("kind") != "youtube#video":
                    continue

                snippet = item.get("snippet", {})
                thumbnails = snippet.get("thumbnails", {})

                thumbnail_url = ""
                for size in ("high", "medium", "default"):
                    if size in thumbnails:
                        thumbnail_url = thumbnails[size].get("url", "")
                        break

                results.append(VideoSearchResult(
                    video_id=item["id"]["videoId"],
                    title=snippet.get("title", ""),
                    channel_id=snippet.get("channelId", ""),
                    channel_title=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    description=snippet.get("description", ""),
                    thumbnail_url=thumbnail_url,
                ))

            remaining = max_results - len(results)
            page_token = data.get("nextPageToken")
            if not page_token:
                break

        # US-157-008: Optionally enrich results with channel metadata
        if enrich_with_channel_data and results:
            channel_ids = list(set(r.channel_id for r in results if r.channel_id))
            if channel_ids:
                channel_metadata = await self.async_get_channel_metadata(channel_ids)
                current_time = datetime.now()

                for video in results:
                    if video.channel_id and video.channel_id in channel_metadata:
                        channel = channel_metadata[video.channel_id]
                        video.subscriber_count = channel.get("subscriber_count", 0)
                        video.total_views = channel.get("view_count", 0)

                        # Calculate channel age for quality scoring
                        channel_age_days = None
                        published_at = channel.get("published_at", "")
                        if published_at:
                            try:
                                published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                                channel_age_days = (current_time - published_date).days
                            except (ValueError, TypeError):
                                pass

                        # Store channel age for quality calculation in matching stage
                        video.channel_created_date = published_at

        logger.info(f"YouTube API async search '{query}': {len(results)} results (key #{self._current_key_index + 1})")

        # US-157-011: Track pagination efficiency metrics
        self._metrics.pagination_results_requested += max_results
        self._metrics.pagination_results_returned += len(results)

        return results

    async def async_get_video_details(
        self,
        video_ids: List[str],
        part: str = "contentDetails,statistics,topicDetails",
        max_concurrent: Optional[int] = None,
    ) -> List[VideoDetails]:
        """Get detailed metadata for videos using async parallel requests.

        Uses aiohttp for concurrent batch requests with semaphore-based concurrency limiting.

        Args:
            video_ids: List of YouTube video IDs
            part: Comma-separated list of parts to request
            max_concurrent: Max concurrent requests (default: 5)

        Returns:
            List of VideoDetails objects

        Raises:
            YouTubeAPIQuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        if not video_ids:
            return []

        # US-153-5: Track batch operation timing
        batch_start_time = time.perf_counter()

        semaphore = self._get_semaphore(max_concurrent)

        # Split video IDs into batches of 50 (API limit)
        batches = [video_ids[i:i + 50] for i in range(0, len(video_ids), 50)]

        async def fetch_batch(batch: List[str]) -> List[VideoDetails]:
            params = {
                "part": part,
                "id": ",".join(batch),
            }

            data = await self._make_async_request("videos", params, QUOTA_COST_VIDEOS, semaphore)

            results = []
            for item in data.get("items", []):
                content_details = item.get("contentDetails", {})
                statistics = item.get("statistics", {})
                topic_details = item.get("topicDetails", {})

                duration_str = content_details.get("duration", "PT0S")
                duration_seconds = self._parse_duration(duration_str)

                view_count = int(statistics.get("viewCount", 0)) if statistics.get("viewCount") else 0
                like_count = int(statistics.get("likeCount", 0)) if statistics.get("likeCount") else 0
                comment_count = int(statistics.get("commentCount", 0)) if statistics.get("commentCount") else 0

                # US-158-007: Extract channel info from snippet
                snippet = item.get("snippet", {})
                channel_id = snippet.get("channelId", "")
                channel_title = snippet.get("channelTitle", "")

                results.append(VideoDetails(
                    video_id=item["id"],
                    duration=duration_str,
                    duration_seconds=duration_seconds,
                    tags=content_details.get("tags", []),
                    category_id=content_details.get("categoryId", ""),
                    topic_details={
                        "topic_categories": topic_details.get("topicCategories", []),
                        "relevant_topic_ids": topic_details.get("relevantTopicIds", []),
                    },
                    # US-150-6: Populate dedicated topic_categories field
                    topic_categories=topic_details.get("topicCategories", []),
                    caption_available=content_details.get("caption", "false") == "true",
                    dimension=content_details.get("dimension", ""),
                    definition=content_details.get("definition", ""),
                    view_count=view_count,
                    like_count=like_count,
                    comment_count=comment_count,
                    channel_id=channel_id,
                    channel_title=channel_title,
                ))

            return results

        # Execute all batches concurrently with semaphore limiting
        tasks = [fetch_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results, filtering out exceptions
        all_results = []
        failed_count = 0
        for result in batch_results:
            if isinstance(result, Exception):
                logger.warning(f"Batch failed with error: {result}")
                failed_count += 1
                continue
            all_results.extend(result)

        # US-153-5: Log batch operation timing metrics
        batch_elapsed = time.perf_counter() - batch_start_time
        videos_per_second = len(video_ids) / batch_elapsed if batch_elapsed > 0 else 0
        logger.info(
            f"YouTube API async video details: {len(all_results)}/{len(video_ids)} videos "
            f"in {len(batches)} batch(es) in {batch_elapsed:.2f}s "
            f"({videos_per_second:.1f} videos/sec, key #{self._current_key_index + 1})"
        )

        # US-158-007: Enrich video details with channel subscriber counts
        if all_results:
            channel_ids = list(set(v.channel_id for v in all_results if v.channel_id))
            if channel_ids:
                channel_metadata = self.get_channel_metadata(channel_ids)
                for video in all_results:
                    if video.channel_id and video.channel_id in channel_metadata:
                        channel = channel_metadata[video.channel_id]
                        video.subscriber_count = channel.get("subscriber_count", 0)

        return all_results

    async def async_check_captions_available(
        self,
        video_id: str,
        max_concurrent: Optional[int] = None,
    ) -> List[CaptionInfo]:
        """Check available captions for a video using async API request.

        Note: This requires the captions.fetch scope which may need
        additional API key configuration. Falls back to checking
        video details for caption availability.

        Args:
            video_id: YouTube video ID
            max_concurrent: Max concurrent requests (default: 5)

        Returns:
            List of CaptionInfo objects (may be empty)

        Raises:
            YouTubeAPIQuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        semaphore = self._get_semaphore(max_concurrent)

        # First try to get from video details (cheaper)
        params = {
            "part": "contentDetails",
            "id": video_id,
        }

        try:
            data = await self._make_async_request(
                "videos", params, QUOTA_COST_VIDEOS, semaphore
            )
            item = data.get("items", [{}])[0]
            content_details = item.get("contentDetails", {})

            # Check if video has captions
            if content_details.get("caption") != "true":
                return []

            # Try to get actual caption tracks (more expensive, requires special scope)
            # This will fail gracefully if scope isn't available
            try:
                caption_params = {
                    "part": "snippet",
                    "videoId": video_id,
                }
                caption_data = await self._make_async_request(
                    "captions", caption_params, QUOTA_COST_CAPTIONS, semaphore
                )

                captions = []
                for cap_item in caption_data.get("items", []):
                    snippet = cap_item.get("snippet", {})
                    captions.append(CaptionInfo(
                        video_id=snippet.get("videoId", video_id),
                        language=snippet.get("language", "en"),
                        track_id=cap_item.get("id", ""),
                        name=snippet.get("name", ""),
                        is_auto_generated=snippet.get("trackKind") == "ASR",
                    ))
                return captions
            except YouTubeAPIError:
                # Fallback: return CaptionInfo indicating captions exist but can't list
                return [CaptionInfo(
                    video_id=video_id,
                    language=self._preferred_caption_language,
                    track_id="",
                    name="",
                    is_auto_generated=False,
                )]
        except YouTubeAPIError:
            # If we can't get details, assume no captions
            return []

    def check_captions_available(self, video_id: str) -> List[CaptionInfo]:
        """Check available captions for a video using captions.list API.

        Note: This requires the captions.fetch scope which may need
        additional API key configuration. Falls back to checking
        video details for caption availability.

        Args:
            video_id: YouTube video ID

        Returns:
            List of CaptionInfo objects (may be empty)

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        # US-154-12: Mock mode - return mock caption info
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock caption info for video: {video_id}")
            from .youtube_api_fixtures import create_mock_caption_list_response
            mock_data = create_mock_caption_list_response(video_id=video_id)
            captions = []
            for item in mock_data.get("items", []):
                snippet = item.get("snippet", {})
                captions.append(CaptionInfo(
                    video_id=snippet.get("videoId", video_id),
                    language=snippet.get("language", "en"),
                    track_id=item.get("id", ""),
                    name=snippet.get("name", ""),
                    is_auto_generated=snippet.get("trackKind") == "ASR",
                ))
            return captions

        # First try to get from video details (cheaper)
        params = {
            "part": "contentDetails",
            "id": video_id,
        }

        try:
            data = self._make_request("videos", params, QUOTA_COST_VIDEOS)
            item = data.get("items", [{}])[0]
            content_details = item.get("contentDetails", {})

            # Check if video has captions
            if content_details.get("caption") != "true":
                return []

            # Try to get actual caption tracks
            # This may fail if scope is not configured
            return self._get_caption_tracks(video_id)

        except APIError as e:
            logger.warning(f"Could not check captions for {video_id}: {e}")
            return []

    def _get_caption_tracks(self, video_id: str) -> List[CaptionInfo]:
        """Get caption track details using captions.list API.

        Args:
            video_id: YouTube video ID

        Returns:
            List of CaptionInfo objects
        """
        params = {
            "part": "snippet",
            "videoId": video_id,
        }

        try:
            data = self._make_request("captions", params, QUOTA_COST_CAPTIONS)

            # US-156-002: Validate response structure before processing
            self.validate_response_structure(data, "captions")

            captions = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                captions.append(CaptionInfo(
                    language=snippet.get("language", ""),
                    track_id=snippet.get("trackId", ""),
                    is_auto_generated=snippet.get("trackKind", "") == "ASR",
                ))

            return captions

        except APIError:
            # Caption API may not be available for all API keys
            # Return a generic result if caption is enabled
            return []

    def fetch_caption_content(
        self,
        video_id: str,
        language: str = "en",
        prefer_manual: bool = True
    ) -> Optional[str]:
        """Fetch full caption content via YouTube Data API.

        US-154-2: Uses captions.download API to fetch actual caption text
        instead of relying on yt-dlp subprocess. This can be faster when
        API quota is available but yt-dlp is blocked/rate-limited.

        US-155-3: Implements fallback chain: preferred_lang -> English -> auto-generated -> any

        Note: This requires the captions.fetch scope which may need
        additional API key configuration. The API key must have access to
        the captions API.

        Args:
            video_id: YouTube video ID
            language: Preferred language code (ISO 639-1)
            prefer_manual: If True, prefer manually uploaded captions over auto-generated

        Returns:
            Raw caption content in SRT format, or None if unavailable

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        # US-154-12: Mock mode - return mock caption content
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock caption content for video: {video_id}")
            mock_data = self._get_mock_captions(video_id)
            if mock_data and mock_data.get("segments"):
                # Convert segments to SRT format
                return self._segments_to_srt(mock_data["segments"])
            return None

        # First get available caption tracks
        caption_tracks = self._get_caption_tracks(video_id)

        if not caption_tracks:
            logger.debug(f"No caption tracks available for {video_id} via API")
            return None

        # US-155-3: Build fallback chain based on config
        # Priority: preferred_language -> English -> auto-generated -> any
        fallback_languages = [language]  # Start with requested language
        if self._caption_language_fallback:
            if language != "en":
                fallback_languages.append("en")  # Add English as fallback

        # Try each language in the fallback chain
        selected_track = None
        selected_language = None

        for lang in fallback_languages:
            # Filter by language
            language_matches = [c for c in caption_tracks if c.language == lang]

            if not language_matches:
                continue

            # Select track: prefer manual over auto-generated if requested
            if prefer_manual:
                manual_tracks = [c for c in language_matches if not c.is_auto_generated]
                if manual_tracks:
                    selected_track = manual_tracks[0]
                    selected_language = lang
                    break
                else:
                    # Auto-generated is better than nothing
                    selected_track = language_matches[0]
                    selected_language = lang
                    break
            else:
                selected_track = language_matches[0]
                selected_language = lang
                break

        # If still no track, try any available (last resort)
        if not selected_track and self._caption_language_fallback:
            # Try auto-generated if we haven't already
            auto_tracks = [c for c in caption_tracks if c.is_auto_generated]
            if auto_tracks:
                selected_track = auto_tracks[0]
                selected_language = auto_tracks[0].language
            else:
                # Any caption is better than none
                selected_track = caption_tracks[0]
                selected_language = caption_tracks[0].language

        if not selected_track:
            logger.debug(f"No caption track selected for video {video_id}")
            return None

        # US-155-3: Log which language was selected for each video
        logger.debug(f"Selected caption language '{selected_language}' for video {video_id}" +
                    (f" (auto-generated)" if selected_track.is_auto_generated else " (manual)"))

        # US-155-3: Track caption language distribution for metrics
        if self._caption_language_metrics and selected_language:
            self._caption_language_counts[selected_language] = self._caption_language_counts.get(selected_language, 0) + 1
            # Also record in metrics for export
            self._metrics.record_caption_language(selected_language)

        # Fetch caption content using captions.download
        try:
            # The captions.download endpoint requires OAuth or API key with proper scope
            # Using API key method for simplicity
            caption_content = self._download_caption(
                video_id,
                selected_track.track_id,
                selected_track.language
            )
            return caption_content
        except APIError as e:
            logger.warning(f"Failed to fetch caption content for {video_id}: {e}")
            return None

    def fetch_captions_batch(
        self,
        video_ids: List[str],
        language: str = "en",
        prefer_manual: bool = True,
        batch_size: Optional[int] = None
    ) -> Dict[str, Optional[str]]:
        """Fetch captions for multiple videos in batches.

        US-158-006: Batch caption fetching for multiple videos in single call.
        Uses captions.list API to check availability, then captions.download
        to fetch actual content. Processes videos in configurable batch sizes
        to avoid overwhelming the API.

        Args:
            video_ids: List of YouTube video IDs
            language: Preferred language code (ISO 639-1)
            prefer_manual: If True, prefer manually uploaded captions over auto-generated
            batch_size: Number of videos to process per batch (default: use instance _caption_batch_size)

        Returns:
            Dict mapping video_id to caption content (SRT format), or None if unavailable.
            Partial results are returned even if some videos fail.

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        # Use instance default if not specified
        if batch_size is None:
            batch_size = self._caption_batch_size
        if not video_ids:
            return {}

        # US-158-006: Mock mode - return mock caption content for all videos
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock caption content for {len(video_ids)} videos")
            results = {}
            for video_id in video_ids:
                mock_data = self._get_mock_captions(video_id)
                if mock_data and mock_data.get("segments"):
                    results[video_id] = self._segments_to_srt(mock_data["segments"])
                else:
                    results[video_id] = None
            return results

        results: Dict[str, Optional[str]] = {}

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]
            logger.debug(f"Fetching captions for batch {i // batch_size + 1}: {len(batch)} videos")

            for video_id in batch:
                try:
                    caption_content = self.fetch_caption_content(
                        video_id=video_id,
                        language=language,
                        prefer_manual=prefer_manual
                    )
                    results[video_id] = caption_content
                except Exception as e:
                    # Handle partial failures gracefully - log and continue
                    logger.warning(f"Failed to fetch caption for {video_id} in batch: {e}")
                    results[video_id] = None

            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(video_ids):
                import time
                time.sleep(0.1)

        # Log summary
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch caption fetch complete: {successful}/{len(video_ids)} successful")

        return results

    def _download_caption(
        self,
        video_id: str,
        track_id: str,
        language: str
    ) -> Optional[str]:
        """Download caption content using captions.download API.

        Args:
            video_id: YouTube video ID
            track_id: Caption track ID from captions.list
            language: Language code

        Returns:
            Caption content in SRT format

        Raises:
            APIError: On API errors
        """
        # Build the download URL
        # captions.download requires POST request with the track ID
        base_url = "https://www.googleapis.com/youtube/v3/captions/download"

        params = {
            "id": track_id,
            "tfmt": "srt",  # Request SRT format
        }

        try:
            # Make the request - captions.download uses API key auth
            # Note: This endpoint may require OAuth2 for some API keys
            response = self._session.get(
                base_url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=self.timeout
            )

            if response.status_code == 401:
                # Auth required - need OAuth2, can't use API key alone
                logger.debug(f"Caption download requires OAuth2 for {video_id}")
                raise APIError("Caption download requires OAuth2 authentication", status_code=401)

            if response.status_code == 403:
                # Permission denied - API key doesn't have captions scope
                logger.debug(f"Caption download not permitted for {video_id} - missing scope")
                raise APIError("API key missing captions scope", status_code=403)

            if response.status_code == 404:
                logger.debug(f"Caption track not found for {video_id}")
                return None

            if response.status_code != 200:
                raise APIError(
                    f"Failed to download caption: {response.status_code}",
                    status_code=response.status_code
                )

            # Track quota usage
            self._add_quota(QUOTA_COST_CAPTIONS, "captions")

            return response.text

        except requests.exceptions.RequestException as e:
            raise APIError(f"Network error downloading caption: {e}")

    def _segments_to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert caption segments to SRT format.

        Args:
            segments: List of caption segment dicts with start_time, end_time, text

        Returns:
            SRT formatted string
        """
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_srt_time(seg.get("start_time", 0))
            end = self._format_srt_time(seg.get("end_time", 0))
            text = seg.get("text", "")
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _parse_duration(self, iso_duration: str) -> int:
        """Parse ISO 8601 duration to seconds.

        Args:
            iso_duration: ISO 8601 duration string (e.g., "PT1H2M10S")

        Returns:
            Duration in seconds
        """
        import re

        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        match = re.match(pattern, iso_duration)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def _calculate_adaptive_ttl(self, channel_id: str) -> float:
        """US-154-9: Calculate adaptive TTL based on channel update frequency.

        Analyzes the channel's update history to determine how often it uploads
        new content, then sets an appropriate TTL.

        Args:
            channel_id: YouTube channel ID

        Returns:
            TTL in seconds (based on base TTL multiplied by activity factor)
        """
        history = self._channel_update_history.get(channel_id, [])

        if len(history) < 2:
            # Not enough history - use default TTL
            return self._channel_cache_ttl

        # Calculate average days between updates
        sorted_history = sorted(history)
        intervals = [
            sorted_history[i + 1] - sorted_history[i]
            for i in range(len(sorted_history) - 1)
        ]
        avg_interval_days = sum(intervals) / len(intervals) / 86400  # Convert seconds to days

        # Determine activity level and apply multiplier
        if avg_interval_days <= 1.5:
            activity = 'daily'
        elif avg_interval_days <= 8:
            activity = 'weekly'
        elif avg_interval_days <= 35:
            activity = 'monthly'
        else:
            activity = 'rare'

        multiplier = self._channel_ttl_multipliers.get(activity, 1.0)
        return self._channel_cache_ttl * multiplier

    def _should_invalidate_channel(self, channel_id: str, new_metadata: Dict[str, Any]) -> bool:
        """US-154-9: Check if channel cache should be invalidated due to significant changes.

        Invalidates cache if:
        - Subscriber count changed significantly (>10% or >1000, whichever is larger)
        - Video count increased (new videos uploaded)

        Args:
            channel_id: YouTube channel ID
            new_metadata: New metadata from API

        Returns:
            True if cache should be invalidated
        """
        if channel_id not in self._channel_cache_metadata:
            return True  # No previous data - not cached yet

        old_metadata = self._channel_cache_metadata[channel_id]
        old_subscribers = old_metadata.get('subscriber_count', 0)
        old_videos = old_metadata.get('video_count', 0)

        new_subscribers = new_metadata.get('subscriber_count', 0)
        new_videos = new_metadata.get('video_count', 0)

        # Check for subscriber count change (>10% or >1000, whichever is larger)
        if old_subscribers > 0:
            change_threshold = max(1000, old_subscribers * 0.10)
            if abs(new_subscribers - old_subscribers) > change_threshold:
                logger.debug(
                    f"Channel {channel_id}: subscriber change detected "
                    f"({old_subscribers} -> {new_subscribers}), invalidating cache"
                )
                return True

        # Check for new video uploads
        if new_videos > old_videos:
            logger.debug(
                f"Channel {channel_id}: new videos detected "
                f"({old_videos} -> {new_videos}), invalidating cache"
            )
            return True

        return False

    def _track_channel_update(self, channel_id: str, metadata: Dict[str, Any]) -> None:
        """US-154-9: Track channel update for adaptive TTL calculation.

        Args:
            channel_id: YouTube channel ID
            metadata: Channel metadata from API
        """
        import time
        current_time = time.time()

        # Update cache metadata
        self._channel_cache_metadata[channel_id] = metadata.copy()

        # Track update timestamp for frequency calculation
        if channel_id not in self._channel_update_history:
            self._channel_update_history[channel_id] = []

        self._channel_update_history[channel_id].append(current_time)

        # Keep only last 10 updates for frequency calculation
        if len(self._channel_update_history[channel_id]) > 10:
            self._channel_update_history[channel_id] = self._channel_update_history[channel_id][-10:]

    def get_channel_cache_stats(self) -> Dict[str, Any]:
        """US-154-9: Get per-channel cache statistics.

        Returns:
            Dict with cache hits/misses per channel and overall stats
        """
        total_hits = sum(self._channel_cache_hits.values())
        total_misses = sum(self._channel_cache_misses.values())
        total = total_hits + total_misses

        return {
            'per_channel_hits': dict(self._channel_cache_hits),
            'per_channel_misses': dict(self._channel_cache_misses),
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': total_hits / total if total > 0 else 0.0,
            'channels_tracked': len(self._channel_cache_metadata),
        }

    def get_per_channel_circuit_breaker_stats(self, channel_id: str) -> Optional[Dict]:
        """US-155-010: Get circuit breaker stats for a specific channel.

        Args:
            channel_id: The YouTube channel ID

        Returns:
            Dict with channel circuit breaker stats or None if not tracked
        """
        if not self._per_channel_tracking_enabled:
            return None
        return self._per_channel_circuit_breaker.get_channel_stats(channel_id)

    def get_per_channel_circuit_breaker_metrics(self) -> Dict[str, Any]:
        """US-155-010: Get overall per-channel circuit breaker metrics.

        Returns:
            Dict with aggregated per-channel metrics
        """
        if not self._per_channel_tracking_enabled:
            return {"enabled": False}
        return self._per_channel_circuit_breaker.get_metrics()

    def get_channel_metadata(self, channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for channels with 24-hour caching.

        Args:
            channel_ids: List of YouTube channel IDs

        Returns:
            Dict mapping channel_id to metadata dict

        Raises:
            QuotaExceededError: If quota is exhausted
            YouTubeAPIError: On API errors
        """
        if not channel_ids:
            return {}

        # US-154-12: Mock mode - return mock channel metadata
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock channel metadata for {len(channel_ids)} channels")
            return self._get_mock_channel_metadata(channel_ids)

        # Check channel cache first (with intelligent invalidation)
        cached_results = {}
        uncached_ids = []

        for cid in channel_ids:
            if cid in self._channel_cache:
                value, expires_at = self._channel_cache[cid]
                # US-154-9: Check if cache is still valid by time
                if time.time() < expires_at:
                    cached_results[cid] = value
                    # Track cache hit
                    self._channel_cache_hits[cid] = self._channel_cache_hits.get(cid, 0) + 1
                else:
                    # TTL expired - remove and fetch fresh
                    del self._channel_cache[cid]
                    if cid in self._channel_cache_metadata:
                        del self._channel_cache_metadata[cid]
                    uncached_ids.append(cid)
                    self._channel_cache_misses[cid] = self._channel_cache_misses.get(cid, 0) + 1
            else:
                uncached_ids.append(cid)
                self._channel_cache_misses[cid] = self._channel_cache_misses.get(cid, 0) + 1

        # If all cached, return immediately
        if not uncached_ids:
            logger.debug(f"Channel metadata cache hit: {len(channel_ids)} channels")
            return cached_results

        # Fetch uncached channels
        # API allows up to 50 channel IDs per request
        result = dict(cached_results)
        requested_channels = set(uncached_ids)  # Track what we requested
        found_channels = set()  # Track what the API returned

        # US-155-010: Per-channel circuit breaker - check each channel before making requests
        channels_to_fetch = []
        for cid in uncached_ids:
            if self._per_channel_tracking_enabled:
                # Check circuit breaker and wait if needed
                self._per_channel_circuit_breaker.check_and_wait(cid)
            channels_to_fetch.append(cid)

        for i in range(0, len(channels_to_fetch), 50):
            batch = channels_to_fetch[i:i + 50]

            # Check quota before making request
            self._check_quota(QUOTA_COST_CHANNELS)

            params = {
                # US-153-6: Added status part for potential verified badge info
                "part": "statistics,snippet,status",
                "id": ",".join(batch),
            }

            url = f"{YOUTUBE_API_BASE}/channels"
            params["key"] = self.api_key

            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 403:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", "")
                    if "quotaexceeded" in error_msg.lower():
                        self._add_quota(QUOTA_COST_CHANNELS, "channels")
                        raise YouTubeAPIQuotaExceededError(
                            f"YouTube API quota exceeded: {error_msg}",
                            endpoint="channels"
                        )
                    # Determine if it's invalid key or permission denied
                    if "key" in error_msg.lower() or "invalid" in error_msg.lower():
                        raise YouTubeAPIInvalidKeyError(
                            f"API key invalid or lacks permissions: {error_msg}",
                            endpoint="channels"
                        )
                    raise YouTubeAPIPermissionDeniedError(
                        f"API permission denied: {error_msg}",
                        endpoint="channels"
                    )

                response.raise_for_status()
                data = response.json()

                self._add_quota(QUOTA_COST_CHANNELS, "channels")

                for item in data.get("items", []):
                    channel_id = item["id"]
                    found_channels.add(channel_id)
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    status = item.get("status", {})

                    # US-153-6: Note - verified badge is not directly available via Data API
                    # The status part provides: longUploads, madeForKids, isLinked, uploadStatus
                    # Verified status would need to be scraped from YouTube frontend or use alternative methods
                    metadata = {
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "subscriber_count": int(stats.get("subscriberCount", 0)),
                        "video_count": int(stats.get("videoCount", 0)),
                        "view_count": int(stats.get("viewCount", 0)),
                        "published_at": snippet.get("publishedAt", ""),
                        # US-153-6: Channel status info (limited - verified badge not available via API)
                        "is_linked": status.get("isLinked", False),
                        "made_for_kids": status.get("madeForKids", False),
                    }

                    result[channel_id] = metadata

                    # US-154-9: Intelligent cache invalidation - check for subscriber/video count changes
                    should_invalidate = self._should_invalidate_channel(channel_id, metadata)

                    # Calculate adaptive TTL based on channel update frequency
                    adaptive_ttl = self._calculate_adaptive_ttl(channel_id)

                    if should_invalidate:
                        # Update the cache metadata and track the update
                        self._track_channel_update(channel_id, metadata)

                    # Cache with adaptive TTL
                    self._channel_cache[channel_id] = (
                        metadata,
                        time.time() + adaptive_ttl
                    )

                # US-154-9: Handle channel not found errors gracefully
                # Channels that were requested but not found in response
                not_found_channels = requested_channels - found_channels
                if not_found_channels:
                    for channel_id in not_found_channels:
                        # Don't cache - mark as not found but don't cache error
                        # Mark as a miss for metrics
                        self._channel_cache_misses[channel_id] = self._channel_cache_misses.get(channel_id, 0) + 1
                        logger.debug(f"Channel {channel_id} not found in API response")
                        # US-155-010: Mark channel as having no videos
                        if self._per_channel_tracking_enabled:
                            self._per_channel_circuit_breaker.mark_no_videos(channel_id)

                # US-155-010: Record successful API call for each channel in batch
                if self._per_channel_tracking_enabled:
                    for channel_id in batch:
                        self._per_channel_circuit_breaker.record_success(channel_id)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Channel metadata request failed: {e}")
                # US-155-010: Record failure for each channel in batch
                if self._per_channel_tracking_enabled:
                    for channel_id in batch:
                        self._per_channel_circuit_breaker.record_failure(channel_id)
                continue

        logger.info(
            f"Channel metadata: {len(result)} channels fetched "
            f"({len(cached_results)} cached, {len(found_channels)} found, {len(requested_channels) - len(found_channels)} not found)"
        )
        return result

    def calculate_channel_quality_score(
        self,
        subscriber_count: int,
        video_count: int,
        total_views: int,
        channel_age_days: Optional[int] = None,
    ) -> float:
        """Calculate channel quality score based on engagement metrics.

        US-157-008: Channel quality scoring based on subscriber count and upload frequency.

        Args:
            subscriber_count: Number of subscribers
            video_count: Number of uploaded videos
            total_views: Total view count across all videos
            channel_age_days: Age of channel in days (calculated from published_at if not provided)

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Subscriber count scoring (logarithmic scale)
        # 0-1K: 0.1-0.3, 1K-10K: 0.3-0.5, 10K-100K: 0.5-0.7, 100K-1M: 0.7-0.85, 1M+: 0.85-1.0
        if subscriber_count <= 0:
            subscriber_score = 0.0
        elif subscriber_count < 1000:
            subscriber_score = 0.1 + (subscriber_count / 1000) * 0.2
        elif subscriber_count < 10000:
            subscriber_score = 0.3 + (subscriber_count / 10000) * 0.2
        elif subscriber_count < 100000:
            subscriber_score = 0.5 + (subscriber_count / 100000) * 0.2
        elif subscriber_count < 1000000:
            subscriber_score = 0.7 + (subscriber_count / 1000000) * 0.15
        else:
            subscriber_score = min(0.85 + (subscriber_count / 10000000) * 0.15, 1.0)

        # Upload frequency scoring (videos per month)
        upload_frequency_score = 0.5  # Default moderate frequency
        if channel_age_days and video_count and channel_age_days > 0:
            months_active = channel_age_days / 30.0
            videos_per_month = video_count / months_active
            # 0-1/month: 0.2, 1-4/month: 0.2-0.5, 4-10/month: 0.5-0.75, 10+/month: 0.75-1.0
            if videos_per_month < 1:
                upload_frequency_score = 0.2
            elif videos_per_month < 4:
                upload_frequency_score = 0.2 + (videos_per_month - 1) / 3 * 0.3
            elif videos_per_month < 10:
                upload_frequency_score = 0.5 + (videos_per_month - 4) / 6 * 0.25
            else:
                upload_frequency_score = min(0.75 + (videos_per_month - 10) / 40 * 0.25, 1.0)

        # Engagement ratio (views per subscriber) - indicates content quality
        engagement_ratio = 0.0
        if subscriber_count > 0:
            engagement_ratio = total_views / subscriber_count
        # Scale: 0-100: 0.1-0.3, 100-500: 0.3-0.6, 500-2000: 0.6-0.8, 2000+: 0.8-1.0
        if engagement_ratio < 100:
            engagement_score = 0.1 + (engagement_ratio / 100) * 0.2
        elif engagement_ratio < 500:
            engagement_score = 0.3 + (engagement_ratio - 100) / 400 * 0.3
        elif engagement_ratio < 2000:
            engagement_score = 0.6 + (engagement_ratio - 500) / 1500 * 0.2
        else:
            engagement_score = min(0.8 + (engagement_ratio - 2000) / 8000 * 0.2, 1.0)

        # Weighted average: subscribers (40%), frequency (25%), engagement (35%)
        final_score = (subscriber_score * 0.4) + (upload_frequency_score * 0.25) + (engagement_score * 0.35)

        return round(final_score, 3)

    def calculate_video_quality_score(
        self,
        view_count: int,
        like_count: int,
        comment_count: int,
    ) -> float:
        """Calculate video quality score based on engagement metrics.

        US-158-010: Video quality signals integration for result ranking.

        Formula: viewCount * view_weight + likeCount * like_weight + commentCount * comment_weight
        Default weights: view_count * 0.7 + like_count * 0.2 + comment_count * 0.1

        Handles zero engagement gracefully by returning 0.0.

        Args:
            view_count: Number of video views
            like_count: Number of video likes
            comment_count: Number of video comments

        Returns:
            Quality score as a float (higher = better quality)
        """
        # Handle zero engagement gracefully
        if view_count == 0 and like_count == 0 and comment_count == 0:
            return 0.0

        # Calculate weighted quality score
        quality_score = (
            view_count * self._quality_view_count_weight +
            like_count * self._quality_like_count_weight +
            comment_count * self._quality_comment_count_weight
        )

        return round(quality_score, 2)

    def get_channel_quality_for_results(
        self,
        video_results: List["VideoSearchResult"],
    ) -> Dict[str, float]:
        """Enrich video search results with channel quality scores.

        US-157-008: Integrate channel data into video search results for better match ranking.

        Args:
            video_results: List of VideoSearchResult objects from search

        Returns:
            Dict mapping video_id to channel quality score
        """
        if not video_results:
            return {}

        # Extract unique channel IDs
        channel_ids = list(set(r.channel_id for r in video_results if r.channel_id))

        if not channel_ids:
            return {}

        # Fetch channel metadata
        channel_metadata = self.get_channel_metadata(channel_ids)

        # Calculate quality scores for each video
        quality_scores: Dict[str, float] = {}
        current_time = datetime.now()

        for video in video_results:
            if not video.channel_id or video.channel_id not in channel_metadata:
                quality_scores[video.video_id] = 0.3  # Default for unknown channels
                continue

            channel = channel_metadata[video.channel_id]
            subscriber_count = channel.get("subscriber_count", 0)
            video_count = channel.get("video_count", 0)
            total_views = channel.get("view_count", 0)

            # Calculate channel age
            channel_age_days = None
            published_at = channel.get("published_at", "")
            if published_at:
                try:
                    published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    channel_age_days = (current_time - published_date).days
                except (ValueError, TypeError):
                    pass

            quality_scores[video.video_id] = self.calculate_channel_quality_score(
                subscriber_count=subscriber_count,
                video_count=video_count,
                total_views=total_views,
                channel_age_days=channel_age_days,
            )

        return quality_scores

    async def async_get_channel_metadata(
        self,
        channel_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Get metadata for channels with 24-hour caching (async version).

        US-157-008: Async version of channel metadata fetching.

        Args:
            channel_ids: List of YouTube channel IDs

        Returns:
            Dict mapping channel_id to metadata dict
        """
        if not channel_ids:
            return {}

        # US-154-12: Mock mode - return mock channel metadata
        if self._mock_mode:
            logger.debug(f"Mock mode: returning mock channel metadata for {len(channel_ids)} channels")
            return self._get_mock_channel_metadata(channel_ids)

        # Check channel cache first
        cached_results = {}
        uncached_ids = []

        for cid in channel_ids:
            if cid in self._channel_cache:
                value, expires_at = self._channel_cache[cid]
                if time.time() < expires_at:
                    cached_results[cid] = value
                    self._channel_cache_hits[cid] = self._channel_cache_hits.get(cid, 0) + 1
                else:
                    del self._channel_cache[cid]
                    if cid in self._channel_cache_metadata:
                        del self._channel_cache_metadata[cid]
                    uncached_ids.append(cid)
                    self._channel_cache_misses[cid] = self._channel_cache_misses.get(cid, 0) + 1
            else:
                uncached_ids.append(cid)
                self._channel_cache_misses[cid] = self._channel_cache_misses.get(cid, 0) + 1

        if not uncached_ids:
            logger.debug(f"Channel metadata cache hit: {len(channel_ids)} channels")
            return cached_results

        result = dict(cached_results)
        requested_channels = set(uncached_ids)
        found_channels = set()

        # Get semaphore for concurrency limiting
        semaphore = self._get_semaphore()

        for i in range(0, len(uncached_ids), 50):
            batch = uncached_ids[i:i + 50]

            # Check quota
            self._check_quota(QUOTA_COST_CHANNELS)

            params = {
                "part": "statistics,snippet,status",
                "id": ",".join(batch),
            }

            url = f"{YOUTUBE_API_BASE}/channels"
            params["key"] = self.api_key

            try:
                async with semaphore:
                    async with self._session.get(
                        url,
                        params=params,
                        timeout=self.timeout,
                    ) as response:

                        if response.status == 403:
                            error_data = await response.json()
                            error_msg = error_data.get("error", {}).get("message", "")
                            if "quotaexceeded" in error_msg.lower():
                                self._add_quota(QUOTA_COST_CHANNELS, "channels")
                                raise YouTubeAPIQuotaExceededError(
                                    f"YouTube API quota exceeded: {error_msg}",
                                    endpoint="channels"
                                )
                            if "key" in error_msg.lower() or "invalid" in error_msg.lower():
                                raise YouTubeAPIInvalidKeyError(
                                    f"API key invalid or lacks permissions: {error_msg}",
                                    endpoint="channels"
                                )
                            raise YouTubeAPIPermissionDeniedError(
                                f"API permission denied: {error_msg}",
                                endpoint="channels"
                            )

                        response.raise_for_status()
                        data = await response.json()

                self._add_quota(QUOTA_COST_CHANNELS, "channels")

                for item in data.get("items", []):
                    channel_id = item["id"]
                    found_channels.add(channel_id)
                    snippet = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    status = item.get("status", {})

                    metadata = {
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "subscriber_count": int(stats.get("subscriberCount", 0)),
                        "video_count": int(stats.get("videoCount", 0)),
                        "view_count": int(stats.get("viewCount", 0)),
                        "published_at": snippet.get("publishedAt", ""),
                        "is_linked": status.get("isLinked", False),
                        "made_for_kids": status.get("madeForKids", False),
                    }

                    result[channel_id] = metadata

                    # Calculate adaptive TTL and cache
                    adaptive_ttl = self._calculate_adaptive_ttl(channel_id)
                    should_invalidate = self._should_invalidate_channel(channel_id, metadata)

                    if should_invalidate:
                        self._track_channel_update(channel_id, metadata)

                    self._channel_cache[channel_id] = (
                        metadata,
                        time.time() + adaptive_ttl
                    )

                not_found_channels = requested_channels - found_channels
                if not_found_channels:
                    for channel_id in not_found_channels:
                        self._channel_cache_misses[channel_id] = self._channel_cache_misses.get(channel_id, 0) + 1
                        logger.debug(f"Channel {channel_id} not found in API response")

            except aiohttp.ClientError as e:
                logger.warning(f"Async channel metadata request failed: {e}")
                continue

        logger.info(
            f"Async channel metadata: {len(result)} channels fetched "
            f"({len(cached_results)} cached, {len(found_channels)} found)"
        )
        return result

    def reset_quota(self) -> None:
        """Reset quota tracking (for new day)."""
        # Reset per-key quota tracking
        self._key_quota_used = {i: 0 for i in range(len(self._api_keys))}
        self._key_quota_warned = {i: False for i in range(len(self._api_keys))}
        # US-149-3: Reset predictive warning flags
        self._warn_soon_warned = {i: False for i in range(len(self._api_keys))}
        # US-152-3: Reset fallback threshold warning flags
        self._key_fallback_threshold_warned = {i: False for i in range(len(self._api_keys))}
        self._quota_usage_timestamps.clear()
        # US-155-003: Reset quota prediction tracking
        self._quota_predictions.clear()
        # Reset exhausted keys and rotation
        self._exhausted_keys.clear()
        self._current_key_index = 0
        self._cache.clear()
        self._channel_cache.clear()
        # US-154-9: Also clear channel cache metadata and metrics
        self._channel_cache_metadata.clear()
        self._channel_cache_hits.clear()
        self._channel_cache_misses.clear()
        self._channel_update_history.clear()
        # US-155-010: Reset per-channel circuit breaker
        if self._per_channel_tracking_enabled:
            self._per_channel_circuit_breaker.reset()
        # US-149-9: Invalidate SQLite query cache on quota reset (new day = fresh results)
        if self._query_cache is not None:
            invalidated = self._query_cache.invalidate_all()
            logger.info(f"YouTube API query cache invalidated: {invalidated} entries cleared")
        # US-146-12: Reset interval metrics but keep session totals
        self._metrics.reset()
        logger.info(f"YouTube API quota and channel cache reset ({len(self._api_keys)} keys)")

    def record_fallback(self, endpoint: str, reason: str, query: str = "") -> None:
        """Record a fallback event where API failed and yt-dlp was used.

        Args:
            endpoint: API endpoint that failed (search, videos, channels, captions)
            reason: Reason for fallback (quota_exceeded, error, circuit_breaker)
            query: Optional search query or video ID
        """
        self._metrics.record_fallback(endpoint, reason, query)
        # US-152-12: Log fallback with reason and timestamp
        timestamp = datetime.now().isoformat()
        logger.info(f"YouTube API fallback: {endpoint} -> yt-dlp (reason={reason}, timestamp={timestamp})")

    def get_api_metrics(self) -> Dict[str, Any]:
        """Get API metrics for export.

        Returns:
            Dictionary with API metrics including call counts, errors, fallbacks,
            and quota aggregates.
        """
        # US-152-7: End session before export
        self._metrics.end_session()

        metrics = self._metrics.to_dict()
        # US-150-3: Include key health metrics
        metrics["key_health"] = self.get_key_health_metrics()
        # US-150-10: Include quota_used and quota_remaining for standard metrics export
        metrics["quota_used"] = self.quota_used
        metrics["quota_remaining"] = self.quota_remaining
        metrics["quota_limit"] = self._total_quota_limit
        # US-152-7: Add quota_per_key to quota_aggregates
        metrics["quota_per_key"] = self._get_quota_per_key()
        # US-153-10: Include circuit breaker state per endpoint
        metrics["circuit_breaker_state"] = self._endpoint_circuit_breaker.get_all_stats()
        # US-155-010: Include per-channel circuit breaker metrics
        if self._per_channel_tracking_enabled:
            metrics["per_channel_circuit_breaker"] = self._per_channel_circuit_breaker.get_metrics()
            metrics["channel_query_distribution"] = self._per_channel_circuit_breaker.get_channel_query_distribution()
        return metrics

    def _get_quota_per_key(self) -> Dict[str, Any]:
        """Get quota usage per API key (US-152-7).

        Returns:
            Dictionary mapping key index to quota usage.
        """
        quota_per_key = {}
        for key_index, quota_used in self._key_quota_used.items():
            masked_key = f"...{self._api_keys[key_index][-4:]}" if len(self._api_keys[key_index]) >= 4 else "****"
            quota_per_key[str(key_index)] = {
                "key_masked": masked_key,
                "quota_used": quota_used,
                "quota_limit": self._total_quota_limit,
                "quota_percent": round(quota_used / self._total_quota_limit * 100, 1) if self._total_quota_limit > 0 else 0,
            }
        return quota_per_key

    def get_key_health_metrics(self) -> Dict[str, Any]:
        """Get per-key health metrics for export (US-150-3).

        Returns:
            Dictionary with per-key health data including quota usage,
            error rates, and health status.
        """
        if not self._api_keys:
            return {"error": "No API keys configured"}

        total_quota = self._total_quota_limit * len(self._api_keys)
        key_health = []

        for i, key in enumerate(self._api_keys):
            # Mask the key for security (show only last 4 chars)
            masked_key = f"...{key[-4:]}" if len(key) >= 4 else "****"

            quota_used = self._key_quota_used.get(i, 0)
            quota_percent = (quota_used / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0

            # Get error counts for this key
            error_counts = self._key_errors.get(i, {"403": 0, "429": 0, "other": 0, "total": 0})
            total_errors = error_counts.get("total", 0)

            # Determine health status
            is_exhausted = i in self._exhausted_keys
            is_warned = self._key_quota_warned.get(i, False)
            # US-152-3: Include fallback threshold warning status
            fallback_threshold_warned = self._key_fallback_threshold_warned.get(i, False)

            if is_exhausted:
                health_status = "exhausted"
            elif quota_percent >= 90:
                health_status = "critical"
            elif quota_percent >= self.warn_at_percent:
                health_status = "warning"
            else:
                health_status = "healthy"

            # Calculate error rate for this key (approximate based on error counts)
            key_calls = self._key_successes.get(i, 0) + total_errors
            error_rate = (total_errors / key_calls * 100) if key_calls > 0 else 0

            key_health.append({
                "key_index": i,
                "key_masked": masked_key,
                "quota_used": quota_used,
                "quota_limit": self._total_quota_limit,
                "quota_percent": round(quota_percent, 1),
                "errors_403": error_counts.get("403", 0),
                "errors_429": error_counts.get("429", 0),
                "errors_other": error_counts.get("other", 0),
                "total_errors": total_errors,
                "total_calls": key_calls,
                "error_rate": round(error_rate, 2),
                "health_status": health_status,
                "is_exhausted": is_exhausted,
                "is_current": i == self._current_key_index,
                "is_warned": is_warned,
                # US-152-3: Include fallback threshold warning status
                "fallback_threshold_warned": fallback_threshold_warned,
            })

        return {
            "total_keys": len(self._api_keys),
            "total_quota_available": total_quota,
            "total_quota_used": sum(self._key_quota_used.values()),
            "current_key_index": self._current_key_index,
            "exhausted_keys_count": len(self._exhausted_keys),
            "keys": key_health,
        }

    def get_usage_summary(self) -> str:
        """Get a human-readable usage summary for logging.

        Returns:
            String with API usage summary for dashboard logging.
        """
        metrics = self._metrics
        total_calls = metrics.get_total_calls()
        total_errors = metrics.get_total_errors()
        fallbacks = metrics.fallback_to_ytdlp

        # Multi-key status (always show for multi-key scenario)
        key_status = []
        for i, quota in self._key_quota_used.items():
            exhausted = "EXHAUSTED" if i in self._exhausted_keys else ""
            key_status.append(f"#{i+1}:{quota}{exhausted}")

        if total_calls == 0:
            return f"YouTube API: No calls made ({', '.join(key_status)})"

        error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0

        return (
            f"YouTube API: {total_calls} calls "
            f"({metrics.search_calls} search, {metrics.videos_calls} videos, "
            f"{metrics.channels_calls} channels, {metrics.captions_calls} captions), "
            f"{total_errors} errors ({error_rate:.1f}% error rate), "
            f"{fallbacks} fallbacks to yt-dlp, "
            f"keys: [{', '.join(key_status)}]"
        )

    def get_key_status(self) -> Dict[str, Any]:
        """Get status of all API keys.

        Returns:
            Dictionary with status of each key including quota usage and error rates (US-150-3).
        """
        key_status = {}
        for i in range(len(self._api_keys)):
            quota = self._key_quota_used.get(i, 0)
            # US-150-3: Include error rate in key status
            health = self.get_key_health(i)
            key_status[f"key_{i+1}"] = {
                "quota_used": quota,
                "quota_limit": self._total_quota_limit,
                "percent_used": round((quota / self._total_quota_limit * 100) if self._total_quota_limit > 0 else 0, 1),
                "exhausted": i in self._exhausted_keys,
                # Error tracking (US-150-3)
                "errors_403": health["errors_403"],
                "errors_429": health["errors_429"],
                "errors_other": health["errors_other"],
                "total_errors": health["total_errors"],
                "total_requests": health["total_requests"],
                "error_rate_percent": health["error_rate_percent"],
                "is_healthy": health["is_healthy"],
            }
        return {
            "active_key": self._current_key_index + 1,
            "total_keys": len(self._api_keys),
            "keys": key_status,
        }

    def health_check(self) -> tuple[bool, str, dict]:
        """Validate API key with minimal quota cost (health check).

        Uses the channels.list endpoint with a known invalid channel ID
        to validate the API key without consuming meaningful quota.

        Results are cached for the session (5 minutes) to avoid repeated API calls.

        Returns:
            Tuple of (is_valid: bool, error_message: str, quota_info: dict)
            - is_valid: True if API key is valid and accessible
            - error_message: Error message if not valid, empty string if valid
            - quota_info: Dict with quota_used, quota_limit, percent_used
        """
        # US-150-8: Check session cache first
        cached = get_cached_validation(self.current_api_key)
        if cached is not None:
            is_valid, error_msg, quota_info = cached
            logger.debug(f"Using cached validation result for API key: {is_valid}")
            return (is_valid, error_msg, quota_info)

        quota_info = {
            "quota_used": self.quota_used,
            "quota_limit": self._total_quota_limit,
            "percent_used": round(self.quota_percent_used, 2),
            "keys_available": self.total_keys - len(self._exhausted_keys),
            "keys_total": self.total_keys,
            "keys_exhausted_count": len(self._exhausted_keys),
            "rotation_strategy": self._rotation_strategy,
        }

        try:
            # Use channels.list with a placeholder ID - this costs 1 quota unit
            # and validates the API key without returning real data
            # We use 'UCinvalid' which is a valid format but won't match any channel
            params = {
                "part": "snippet",
                "id": "UCinvalid_channel_id_12345",  # Invalid ID format
            }

            # Make request directly without going through _make_request
            # to avoid quota tracking for health check
            url = f"{YOUTUBE_API_BASE}/channels"
            params["key"] = self.current_api_key

            response = self._session.get(
                url,
                params=params,
                timeout=self.timeout,
            )

            # Check for specific error conditions
            if response.status_code == 403:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "")
                if "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    quota_info["quota_exceeded"] = True
                    result = (False, f"Quota exceeded: {error_msg}", quota_info)
                else:
                    # Invalid key
                    result = (False, f"Invalid API key: {error_msg}", quota_info)
                set_cached_validation(self.current_api_key, result[0], result[1], result[2])
                return result

            if response.status_code == 401:
                result = (False, "Authentication failed - API key may be invalid", quota_info)
                set_cached_validation(self.current_api_key, result[0], result[1], result[2])
                return result

            if response.status_code == 200:
                # Success - API key is valid
                # Even though we got 200, we didn't actually consume quota
                # since we're not calling _make_request
                result = (True, "", quota_info)
                set_cached_validation(self.current_api_key, result[0], result[1], result[2])
                return result

            # Other status codes
            result = (False, f"Unexpected response: HTTP {response.status_code}", quota_info)
            set_cached_validation(self.current_api_key, result[0], result[1], result[2])
            return result

        except requests.exceptions.Timeout:
            result = (False, "Request timeout during health check", quota_info)
            set_cached_validation(self.current_api_key, result[0], result[1], result[2])
            return result
        except requests.exceptions.RequestException as e:
            result = (False, f"Network error during health check: {str(e)}", quota_info)
            set_cached_validation(self.current_api_key, result[0], result[1], result[2])
            return result
        except Exception as e:
            result = (False, f"Health check failed: {str(e)}", quota_info)
            set_cached_validation(self.current_api_key, result[0], result[1], result[2])
            return result

    def close(self) -> None:
        """Close the HTTP session and save quota before cleanup."""
        # Save quota before closing
        self._save_quota()
        self._session.close()
        # US-149-8: Clean up async resources
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        # Close async session if it exists
        if hasattr(self, '_async_session') and self._async_session and not self._async_session.closed:
            asyncio.run(self._async_session.close())

    async def close_async(self) -> None:
        """Close the async HTTP session."""
        if hasattr(self, '_async_session') and self._async_session and not self._async_session.closed:
            await self._async_session.close()
            self._async_session = None

    def __enter__(self) -> "YouTubeAPIClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> "YouTubeAPIClient":
        """Async context manager entry.

        Returns:
            Self for use in async with statements.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.

        Closes the async HTTP session and cleans up resources.
        """
        await self.close_async()
