"""Batch-level retry queue for rate-limited videos.

When rate limiting affects a batch, failed videos are collected and retried
together after a delay. This provides better recovery than individual retries
by allowing YouTube's rate limit window to pass before attempting the batch.

Key concepts:
- Failed videos are queued with their keyword and tier context
- After batch completes, the queue is processed with configurable delay
- Maximum 2 retry passes per download session (configurable)
- Queue is cleared on session start or when all retries complete

Architecture (Single Responsibility Principle):
- RetryQueue: Pure data structure (add/get/clear items, track state)
- RetryQueueProcessor: Execution logic (delay, circuit breaker wait, cookie cooldown)

For processing, use RetryQueueProcessor which wraps a RetryQueue instance.
RetryQueue still exposes processing methods for backward compatibility, but
they delegate to an internal processor instance.

Circuit breaker coordination:
- When a circuit breaker is linked, the retry queue checks its state before processing
- If the circuit breaker is tripped, waits for it to recover before starting retries
- This prevents wasting retry attempts during active rate limit periods
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, TYPE_CHECKING

from .retry_stats import RetryQueueStats
from .errors import (
    ClassifiedDownloadError,
    StructuredDownloadError,
)
from .error_classification import classify_error_category

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreaker
    from .cookie_rotator import CookieRotator
    from .rate_limit_budget import RateLimitBudget
    from .retry_processor import RetryQueueProcessor

logger = logging.getLogger(__name__)


# =============================================================================
# Download Retry Budget (US-129-002)
# =============================================================================

@dataclass
class VideoRetryState:
    """Tracks retry budget state for a single video."""
    video_id: str
    attempts: int = 0
    total_backoff_seconds: float = 0.0
    # US-136-011: Per-error-category retry tracking
    # Tracks attempts per error category: {"rate_limit": 2, "network_error": 1, ...}
    attempts_by_category: Dict[str, int] = field(default_factory=dict)
    # US-144-003: Per-category successful retry tracking for backoff reduction
    # Tracks successful retries per category: {"rate_limit": 1, "network": 0, ...}
    successful_retries_by_category: Dict[str, int] = field(default_factory=dict)
    # US-144-003: Current backoff per category (used for reduction on success)
    current_backoff_by_category: Dict[str, float] = field(default_factory=dict)


class DownloadRetryBudget:
    """Per-video retry budget tracker (US-129-002).

    Tracks remaining retry attempts and backoff time per video ID to prevent
    infinite download loops on persistent failures.

    Usage:
        budget = DownloadRetryBudget(config)
        budget.record_attempt(video_id, backoff_seconds=10.0)

        if budget.is_exhausted(video_id):
            logger.warning(f"Skipping {video_id} - retry budget exhausted")
            return

        if download_succeeded:
            budget.reset(video_id)
    """

    def __init__(self, config: Optional['DownloadRetryBudgetConfig'] = None):
        """Initialize retry budget tracker.

        Args:
            config: DownloadRetryBudgetConfig instance. Uses defaults if None.
        """
        from ..config.sections.download import DownloadRetryBudgetConfig
        self._config = config or DownloadRetryBudgetConfig()
        self._video_states: Dict[str, VideoRetryState] = {}

    @property
    def enabled(self) -> bool:
        """Check if retry budget tracking is enabled."""
        return self._config.enabled

    def record_attempt(self, video_id: str, backoff_seconds: float = None, error_category: str = "unknown") -> None:
        """Record a retry attempt for a video.

        US-144-003: When backoff_seconds is not provided, calculates category-aware
        exponential backoff based on the error category.

        Args:
            video_id: YouTube video ID
            backoff_seconds: Backoff time used for this attempt. If None, calculates
                category-aware exponential backoff.
            error_category: Error category for per-category tracking (e.g., "rate_limit", "network_error", "server_error")
        """
        if not self.enabled:
            return

        # US-144-003: Calculate category-aware backoff if not provided
        if backoff_seconds is None:
            backoff_seconds = self.calculate_category_backoff(video_id, error_category)

        if video_id not in self._video_states:
            self._video_states[video_id] = VideoRetryState(video_id=video_id)

        state = self._video_states[video_id]
        state.attempts += 1
        state.total_backoff_seconds += backoff_seconds

        # US-136-011: Track attempts by error category
        if not state.attempts_by_category:
            state.attempts_by_category = {}
        state.attempts_by_category[error_category] = state.attempts_by_category.get(error_category, 0) + 1

        # US-144-003: Track current backoff per category for potential reduction
        if not state.current_backoff_by_category:
            state.current_backoff_by_category = {}
        state.current_backoff_by_category[error_category] = backoff_seconds

        logger.debug(
            f"Retry budget: recorded attempt {state.attempts}/{self._config.max_attempts} "
            f"for {video_id} (category={error_category}, backoff: {state.total_backoff_seconds:.1f}s/"
            f"{self._config.max_backoff_time_seconds}s)"
        )

    def is_exhausted(self, video_id: str) -> bool:
        """Check if retry budget is exhausted for a video.

        Returns True if either:
        - attempts >= max_attempts
        - total_backoff_seconds >= max_backoff_time_seconds

        Args:
            video_id: YouTube video ID

        Returns:
            True if budget exhausted, False otherwise
        """
        if not self.enabled:
            return False

        if video_id not in self._video_states:
            return False

        state = self._video_states[video_id]

        # Check both limits
        attempts_exhausted = (
            self._config.max_attempts > 0 and
            state.attempts >= self._config.max_attempts
        )
        backoff_exhausted = (
            self._config.max_backoff_time_seconds > 0 and
            state.total_backoff_seconds >= self._config.max_backoff_time_seconds
        )

        if attempts_exhausted or backoff_exhausted:
            logger.warning(
                f"Retry budget EXHAUSTED for {video_id}: "
                f"attempts={state.attempts}/{self._config.max_attempts}, "
                f"backoff={state.total_backoff_seconds:.1f}s/{self._config.max_backoff_time_seconds}s"
            )
            return True

        return False

    def get_remaining_attempts(self, video_id: str) -> int:
        """Get remaining retry attempts for a video.

        Args:
            video_id: YouTube video ID

        Returns:
            Remaining attempts, or -1 if unlimited/not tracking
        """
        if not self.enabled:
            return -1

        if video_id not in self._video_states:
            # Return max_attempts if 0 (unlimited), otherwise return the config value
            return -1 if self._config.max_attempts == 0 else self._config.max_attempts

        state = self._video_states[video_id]
        if self._config.max_attempts <= 0:
            return -1  # Unlimited

        return max(0, self._config.max_attempts - state.attempts)

    def get_attempts_by_category(self, video_id: str) -> Dict[str, int]:
        """Get retry attempts broken down by error category.

        US-136-011: Per-error-category budget tracking.

        Args:
            video_id: YouTube video ID

        Returns:
            Dict mapping error category to attempt count, e.g., {"rate_limit": 2, "network_error": 1}
        """
        if not self.enabled:
            return {}

        if video_id not in self._video_states:
            return {}

        return dict(self._video_states[video_id].attempts_by_category)

    def calculate_category_backoff(self, video_id: str, error_category: str = "unknown") -> float:
        """US-144-003: Calculate exponential backoff time based on error category.

        Uses category-specific base times and exponential growth:
        - rate_limit: 30s base (YouTube rate limiting)
        - network: 10s base (connectivity issues)
        - format: 5s base (format-specific errors)
        - server: 15s base (5xx errors)
        - bot_detection: 30s base (403/bot detection)
        - timeout: 15s base (timeout errors)
        - geo_blocked: 60s base (needs VPN)

        Each retry doubles the backoff (exponential_base^attempts).

        Args:
            video_id: YouTube video ID
            error_category: Error category for backoff calculation

        Returns:
            Backoff time in seconds for this retry attempt
        """
        if not self.enabled:
            return 0.0

        category_config = self._config.category_backoff

        # Get the base and max backoff for this category
        base_times = {
            "rate_limit": category_config.rate_limit_base,
            "network": category_config.network_base,
            "network_error": category_config.network_base,
            "format": category_config.format_base,
            "format_error": category_config.format_base,
            "server": category_config.server_base,
            "server_error": category_config.server_base,
            "bot_detection": category_config.bot_detection_base,
            "bot_detection_error": category_config.bot_detection_base,
            "timeout": category_config.timeout_base,
            "timeout_error": category_config.timeout_base,
            "geo_blocked": category_config.geo_blocked_base,
            "geo_blocked_error": category_config.geo_blocked_base,
        }

        max_backoffs = {
            "rate_limit": category_config.rate_limit_max,
            "network": category_config.network_max,
            "network_error": category_config.network_max,
            "format": category_config.format_max,
            "format_error": category_config.format_max,
            "server": category_config.server_max,
            "server_error": category_config.server_max,
            "bot_detection": category_config.bot_detection_max,
            "bot_detection_error": category_config.bot_detection_max,
            "timeout": category_config.timeout_max,
            "timeout_error": category_config.timeout_max,
            "geo_blocked": category_config.geo_blocked_max,
            "geo_blocked_error": category_config.geo_blocked_max,
        }

        # Get base and max for category, default to network if unknown
        base = base_times.get(error_category, category_config.network_base)
        max_backoff = max_backoffs.get(error_category, category_config.network_max)

        # Get attempt count for this category
        if video_id in self._video_states:
            state = self._video_states[video_id]
            category_attempts = state.attempts_by_category.get(error_category, 0)
            successful_retries = state.successful_retries_by_category.get(error_category, 0)
        else:
            category_attempts = 0
            successful_retries = 0

        # Calculate exponential backoff: base * (exponential_base ^ category_attempts)
        # But apply reduction factor if we've had successful retries
        effective_attempts = category_attempts - (successful_retries if self._config.reduce_backoff_on_success else 0)
        effective_attempts = max(0, effective_attempts)

        backoff = base * (category_config.exponential_base ** effective_attempts)

        # Cap at max backoff
        backoff = min(backoff, max_backoff)

        # Add jitter (±10%) to avoid thundering herd
        import random
        jitter = backoff * 0.1 * (2 * random.random() - 1)
        backoff = backoff + jitter

        logger.debug(
            f"Category-aware backoff for {video_id} (category={error_category}): "
            f"{backoff:.1f}s (base={base}s, attempts={category_attempts}, "
            f"successful_retries={successful_retries}, max={max_backoff}s)"
        )

        return backoff

    def record_successful_retry(self, video_id: str, error_category: str = "unknown") -> None:
        """US-144-003: Record a successful retry for a category to reduce future backoff.

        When enabled, successful retries reduce the backoff for subsequent retries
        in the same category by the success_backoff_reduction_factor.

        Args:
            video_id: YouTube video ID
            error_category: Error category that was successfully retried
        """
        if not self.enabled or not self._config.reduce_backoff_on_success:
            return

        if video_id not in self._video_states:
            return

        state = self._video_states[video_id]

        if not state.successful_retries_by_category:
            state.successful_retries_by_category = {}

        state.successful_retries_by_category[error_category] = (
            state.successful_retries_by_category.get(error_category, 0) + 1
        )

        logger.debug(
            f"Recorded successful retry for {video_id} (category={error_category}), "
            f"total successful: {state.successful_retries_by_category[error_category]}"
        )

    def reset(self, video_id: str) -> None:
        """Reset budget for a video after successful download.

        Args:
            video_id: YouTube video ID
        """
        if not self.enabled:
            return

        if video_id in self._video_states:
            del self._video_states[video_id]
            logger.debug(f"Retry budget: reset for {video_id}")

    def reset_all(self) -> None:
        """Reset all video budgets."""
        self._video_states.clear()
        logger.debug("Retry budget: reset all")

    def to_checkpoint_dict(self) -> dict:
        """Serialize budget state to dictionary for checkpoint persistence.

        Returns:
            Dict that can be saved to checkpoint JSON.
        """
        return {
            'video_states': {
                video_id: {
                    'attempts': state.attempts,
                    'total_backoff_seconds': state.total_backoff_seconds,
                    'attempts_by_category': state.attempts_by_category,
                    'successful_retries_by_category': state.successful_retries_by_category,
                }
                for video_id, state in self._video_states.items()
            }
        }

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore budget state from checkpoint dictionary.

        Args:
            data: Dict from checkpoint JSON.
        """
        if not data:
            return

        self._video_states.clear()
        for video_id, state_data in data.get('video_states', {}).items():
            self._video_states[video_id] = VideoRetryState(
                video_id=video_id,
                attempts=state_data.get('attempts', 0),
                total_backoff_seconds=state_data.get('total_backoff_seconds', 0.0),
            )
            # Restore category-specific tracking
            attempts_by_cat = state_data.get('attempts_by_category', {})
            if attempts_by_cat:
                self._video_states[video_id].attempts_by_category = attempts_by_cat
            successful_retries = state_data.get('successful_retries_by_category', {})
            if successful_retries:
                self._video_states[video_id].successful_retries_by_category = successful_retries

        if self._video_states:
            logger.debug(f"Retry budget: restored {len(self._video_states)} video states from checkpoint")


# =============================================================================
# Keyword Category Extractor (US-123-012)
# =============================================================================

def keyword_category_extractor(keyword: str) -> str:
    """Extract category from keyword for retry strategy selection.

    Categories help group similar keywords that likely have similar retry patterns.
    This is used by cross-keyword retry learning to recommend strategies.

    Args:
        keyword: The search keyword to categorize

    Returns:
        Category name (e.g., 'tech', 'news', 'tutorial', 'general')
    """
    kw_lower = keyword.lower()

    # Check for common keyword patterns (order matters - more specific first)
    if any(term in kw_lower for term in ['footage', 'stock', 'b-roll', 'broll']):
        return 'stock_footage'
    elif any(term in kw_lower for term in ['documentary', 'history', 'explained']):
        return 'documentary'
    elif any(term in kw_lower for term in ['game', 'gaming', 'playthrough', 'gameplay']):
        return 'gaming'
    elif any(term in kw_lower for term in ['tour', 'walk through', 'travel']):
        return 'travel'
    elif any(term in kw_lower for term in ['aerial', 'drone', '4k', 'timelapse']):
        return 'cinematic'
    elif any(term in kw_lower for term in ['interview', 'speech', 'talk']):
        return 'interview'
    elif any(term in kw_lower for term in ['python', 'javascript', 'java', 'code', 'programming',
                                            'tutorial', 'how to', 'learn', 'course', 'coding']):
        return 'tech'
    elif any(term in kw_lower for term in ['news', 'breaking', 'report', 'update']):
        return 'news'
    elif any(term in kw_lower for term in ['music', 'song', 'album', 'artist', 'band']):
        return 'music'
    elif any(term in kw_lower for term in ['sports', 'match', 'football', 'soccer']):
        return 'sports'
    elif any(term in kw_lower for term in ['recipe', 'cooking', 'food', 'bake', 'chef']):
        return 'cooking'
    elif any(term in kw_lower for term in ['fitness', 'workout', 'exercise', 'gym', 'training']):
        return 'fitness'
    elif any(term in kw_lower for term in ['education', 'learn', 'lesson', 'class', 'school']):
        return 'education'
    elif any(term in kw_lower for term in ['movie', 'film', 'trailer', 'review', 'show']):
        return 'entertainment'
    elif any(term in kw_lower for term in ['science', 'experiment', 'research', 'discovery']):
        return 'science'
    elif any(term in kw_lower for term in ['nature', 'wildlife', 'animal', 'planet']):
        return 'nature'
    else:
        return 'general'


# =============================================================================
# Retry Strategy Recommendation
# =============================================================================

@dataclass
class RetryStrategyRecommendation:
    """Recommended retry strategy based on keyword category history."""
    recommended_delay_multiplier: float = 1.0  # Multiplier for base delay
    recommended_max_attempts: int = 2  # Additional retry attempts
    confidence: float = 0.0  # Confidence in recommendation (0.0-1.0)
    source: str = 'default'  # 'default', 'keyword', 'category', 'cross_category'
    category: str = 'general'  # The category this recommendation is based on


@dataclass
class BatchRetryConfig:
    """Configuration for batch-level retry queue.

    When rate limiting affects multiple videos in a batch, collect them
    and retry the entire batch after a delay. This is more effective than
    individual retries because it allows the rate limit window to pass.

    Example with defaults:
      - Video fails due to rate limit → added to retry queue
      - After batch completes, wait 120s
      - Retry all queued videos together (pass 1)
      - If still failing, wait and retry again (pass 2)
      - After 2 passes, give up on remaining failures

    Circuit breaker coordination:
      When respect_circuit_breaker is enabled (default), the batch retry queue
      will check the circuit breaker state before processing. If the circuit
      breaker is tripped, the retry queue will wait for it to recover before
      retrying. This prevents retries from being wasted during active rate limits.
    """
    # Enable/disable batch retry queue
    enabled: bool = True

    # Delay before processing retry queue (seconds)
    # Should be long enough for rate limit window to pass
    delay_seconds: float = 120.0

    # Maximum retry passes per download session
    # After this many batch retries, give up on remaining failures
    max_passes: int = 2

    # Respect circuit breaker state when processing retries
    # If True, wait for circuit breaker to recover before retrying
    # If False, retry immediately after delay_seconds regardless of circuit breaker
    respect_circuit_breaker: bool = True

    # Wait for cookie cooldown before processing retries
    # If True, check if any cookies are in cooldown and extend delay if needed
    # If False, proceed with retry even if cookies are in cooldown
    wait_for_cookie_cooldown: bool = True

    # Maximum combined wait time (seconds) when both circuit breaker and cookie
    # cooldown are blocking simultaneously. If exceeded, force-process the retry
    # queue with the current best-available cookie method instead of waiting
    # for both to clear. Prevents deadlock when CB and cooldown overlap.
    max_combined_wait_seconds: float = 300.0

    # Maximum retries per individual video across pipeline restarts (US-51-010).
    # Videos exceeding this count are permanently skipped.
    # Unlike max_passes (batch-level), this tracks per-video retry_count.
    max_retries_per_video: int = 3

    # Jitter factor for randomizing delay durations (0.0 to 1.0)
    # Delay is computed as: base_delay * (1 + random.uniform(-jitter, +jitter))
    # Default 0.2 means ±20% randomization to prevent thundering herd
    jitter_factor: float = 0.2

    # US-114-012: Progressive retry delay with exponential backoff
    # Initial delay for first retry pass (seconds)
    initial_delay_seconds: float = 1.0
    # Maximum delay cap - delays will not exceed this value (seconds)
    max_delay_seconds: float = 60.0
    # Multiplier for exponential backoff (delay * multiplier^pass)
    backoff_multiplier: float = 2.0

    # US-114-002: Priority weighting for smart retry queue prioritization.
    # Controls how segment_value_score is calculated:
    # - duration: Weight for segment duration tier (longer = higher priority)
    # - confidence: Weight for match confidence score (higher = higher priority)
    # - retry_count: Weight for retry count (lower = higher priority, ensures eventual retry)
    # Formula: score = (duration_score * duration_weight) + (confidence * confidence_weight) + ((max_retries - retry_count) * retry_weight)
    retry_priority_weighting: Dict[str, float] = None

    # US-143-011: Deadline-aware retry ordering configuration
    # Controls deadline urgency scoring in priority calculation:
    # - enabled: Whether deadline affects priority (if False, deadlines are ignored)
    # - urgency_threshold_seconds: How close to deadline before urgency kicks in (default 300s = 5 min)
    # - max_urgency_score: Maximum urgency score added when at or past deadline (default 1.0)
    deadline_aware: bool = True
    deadline_urgency_threshold_seconds: float = 300.0  # 5 minutes
    deadline_max_urgency_score: float = 1.0

    def __post_init__(self):
        """Set default priority weighting if not provided."""
        if self.retry_priority_weighting is None:
            self.retry_priority_weighting = {
                "duration": 0.5,
                "confidence": 0.3,
                "retry_count": 0.2,
            }
        # Handle case where it loads as a dict from YAML/config
        elif isinstance(self.retry_priority_weighting, dict):
            self.retry_priority_weighting = {
                "duration": self.retry_priority_weighting.get("duration", 0.5),
                "confidence": self.retry_priority_weighting.get("confidence", 0.3),
                "retry_count": self.retry_priority_weighting.get("retry_count", 0.2),
            }


@dataclass
class RetryItem:
    """A single video waiting in the retry queue."""
    video_id: str
    keyword: str
    tier: str
    error_message: str
    retry_count: int = 0
    added_at: float = field(default_factory=time.time)
    severity: str = 'medium'  # low, medium, high - determines delay multiplier
    error_category: str = 'video_specific'  # 'network', 'bot_detection', 'timeout', or 'video_specific'
    escalation_tier: int = 1  # US-49-010: Last-used escalation tier (1-4) so retry starts at this tier or higher
    # US-114-002: Segment value fields for smart prioritization
    duration_tier: str = ""  # Duration tier of the segment (short, medium, long, longer)
    match_confidence: float = 0.0  # Match confidence score (0.0-1.0) for priority calculation
    # US-143-011: Deadline-aware retry ordering
    # Unix timestamp when this retry must complete (None = no deadline)
    deadline: Optional[float] = None


class RetryQueue:
    """Batch-level retry queue for rate-limited videos.

    Collects videos that fail due to rate limiting and retries them
    together after a delay. This provides better recovery than
    individual retries by:

    1. Batching failures together for efficient retry
    2. Applying a delay that lets rate limit windows pass
    3. Limiting total retry passes to avoid infinite loops

    Usage:
        queue = RetryQueue(config)

        # During download batch
        if rate_limit_error:
            queue.add(video_id, keyword, tier, error_message)

        # After batch completes
        if queue.has_pending():
            queue.process_retry_pass(download_func)

    Attributes:
        config: BatchRetryConfig with delay and pass limits
        items: Dict mapping video_id to RetryItem
        current_pass: Current retry pass number (1-indexed)
        stats: Statistics for reporting
    """

    def __init__(self, config: Optional[BatchRetryConfig] = None):
        """Initialize retry queue with configuration.

        Args:
            config: BatchRetryConfig. If None, uses defaults.
        """
        self.config = config or BatchRetryConfig()
        self.items: Dict[str, RetryItem] = {}
        self.current_pass: int = 0
        self._completed_ids: Set[str] = set()  # Successfully retried
        self._failed_ids: Set[str] = set()  # Permanently failed after max retries
        self._total_added: int = 0
        self._total_retried: int = 0

        # Processor handles execution logic (delay, CB wait, cookie cooldown)
        # Lazily created to avoid circular imports
        self._processor: Optional['RetryQueueProcessor'] = None

        # Initialize stats tracker with config values
        self._stats = RetryQueueStats(
            enabled=self.config.enabled,
            max_passes=self.config.max_passes,
            delay_seconds=self.config.delay_seconds,
            respect_circuit_breaker=self.config.respect_circuit_breaker,
            wait_for_cookie_cooldown=self.config.wait_for_cookie_cooldown,
            max_combined_wait_seconds=self.config.max_combined_wait_seconds,
        )

        # =====================================================================
        # Cross-keyword retry learning (US-123-012)
        # =====================================================================
        # Track retry success by keyword and category for strategy recommendations
        self._keyword_retry_history: Dict[str, Dict] = {}  # keyword -> {attempts, successes, failures}
        self._category_retry_history: Dict[str, Dict] = {}  # category -> {attempts, successes, failures}

        # Import config for cross-keyword learning settings
        try:
            from ..config import get_config
            cfg = get_config()
            self._cross_keyword_config = getattr(cfg, 'cross_keyword_learning', None)
        except Exception:
            self._cross_keyword_config = None

    @property
    def processor(self) -> 'RetryQueueProcessor':
        """Get the processor instance, creating it lazily if needed."""
        if self._processor is None:
            from .retry_processor import RetryQueueProcessor
            self._processor = RetryQueueProcessor(self)
        return self._processor

    # =========================================================================
    # Backward compatibility properties - delegate to processor
    # =========================================================================

    @property
    def _circuit_breaker(self) -> Optional['CircuitBreaker']:
        """Get circuit breaker from processor (backward compat)."""
        return self.processor._circuit_breaker

    @_circuit_breaker.setter
    def _circuit_breaker(self, value: Optional['CircuitBreaker']) -> None:
        """Set circuit breaker on processor (backward compat)."""
        self.processor._circuit_breaker = value

    @property
    def _cookie_rotator(self) -> Optional['CookieRotator']:
        """Get cookie rotator from processor (backward compat)."""
        return self.processor._cookie_rotator

    @_cookie_rotator.setter
    def _cookie_rotator(self, value: Optional['CookieRotator']) -> None:
        """Set cookie rotator on processor (backward compat)."""
        self.processor._cookie_rotator = value

    @property
    def _circuit_breaker_wait_time(self) -> float:
        """Get CB wait time from processor (backward compat)."""
        return self.processor._circuit_breaker_wait_time

    @_circuit_breaker_wait_time.setter
    def _circuit_breaker_wait_time(self, value: float) -> None:
        """Set CB wait time on processor (backward compat)."""
        self.processor._circuit_breaker_wait_time = value

    @property
    def _cookie_cooldown_wait_time(self) -> float:
        """Get cookie cooldown wait time from processor (backward compat)."""
        return self.processor._cookie_cooldown_wait_time

    @_cookie_cooldown_wait_time.setter
    def _cookie_cooldown_wait_time(self, value: float) -> None:
        """Set cookie cooldown wait time on processor (backward compat)."""
        self.processor._cookie_cooldown_wait_time = value

    @property
    def _budget_state(self) -> Optional[Dict]:
        """Get budget state from processor (backward compat)."""
        return self.processor._budget_state

    @_budget_state.setter
    def _budget_state(self, value: Optional[Dict]) -> None:
        """Set budget state on processor (backward compat)."""
        self.processor._budget_state = value

    @property
    def _forced_retry(self) -> bool:
        """Get forced retry flag from processor (backward compat)."""
        return self.processor._forced_retry

    @_forced_retry.setter
    def _forced_retry(self, value: bool) -> None:
        """Set forced retry flag on processor (backward compat)."""
        self.processor._forced_retry = value

    @property
    def _last_jitter_applied(self) -> float:
        """Get last jitter from processor (backward compat)."""
        return self.processor._last_jitter_applied

    @_last_jitter_applied.setter
    def _last_jitter_applied(self, value: float) -> None:
        """Set last jitter on processor (backward compat)."""
        self.processor._last_jitter_applied = value

    # =========================================================================
    # Processing methods - delegate to processor
    # =========================================================================

    def set_circuit_breaker(self, circuit_breaker: 'CircuitBreaker') -> None:
        """Link a circuit breaker to coordinate retry timing.

        When a circuit breaker is linked and respect_circuit_breaker is enabled,
        the retry queue will check the circuit breaker state before processing.
        If tripped, it waits for the circuit breaker to recover before retrying.

        Args:
            circuit_breaker: CircuitBreaker instance to coordinate with.
        """
        self.processor.set_circuit_breaker(circuit_breaker)
        logger.debug("Retry queue: linked to circuit breaker")

    def set_cookie_rotator(self, cookie_rotator: 'CookieRotator') -> None:
        """Link a cookie rotator to coordinate retry timing with cookie cooldowns.

        When a cookie rotator is linked and wait_for_cookie_cooldown is enabled,
        the retry queue will check if any cookies are in cooldown before processing.
        If all cookies are in cooldown, it waits for the shortest cooldown to expire
        before retrying.

        Args:
            cookie_rotator: CookieRotator instance to coordinate with.
        """
        self.processor.set_cookie_rotator(cookie_rotator)
        logger.debug("Retry queue: linked to cookie rotator")

    def set_budget_state(self, budget_summary: Dict) -> None:
        """Store a snapshot of the rate limit budget state.

        Called before retry processing so the retry queue knows the
        remaining budget when deciding whether to retry items.

        Args:
            budget_summary: Dict from RateLimitBudget.get_summary()
        """
        self.processor.set_budget_state(budget_summary)

    def get_budget_state(self) -> Optional[Dict]:
        """Get the stored budget state snapshot.

        Returns:
            Budget summary dict, or None if not set.
        """
        return self.processor.get_budget_state()

    def _apply_jitter(self, delay: float) -> float:
        """Apply random jitter to a delay value. Delegates to processor."""
        return self.processor._apply_jitter(delay)

    def _get_cookie_cooldown_remaining(self) -> float:
        """Get remaining cookie cooldown time. Delegates to processor."""
        return self.processor._get_cookie_cooldown_remaining()

    def _get_cb_remaining(self) -> float:
        """Get remaining circuit breaker pause time. Delegates to processor."""
        return self.processor._get_cb_remaining()

    def _wait_for_cookie_cooldown(self) -> float:
        """Wait for cookie cooldown to expire. Delegates to processor."""
        return self.processor._wait_for_cookie_cooldown()

    def _wait_for_circuit_breaker(self) -> float:
        """Wait for circuit breaker to recover. Delegates to processor."""
        return self.processor._wait_for_circuit_breaker()

    def _wait_combined(self) -> float:
        """Wait for both circuit breaker and cookie cooldown. Delegates to processor."""
        return self.processor._wait_combined()

    @property
    def is_enabled(self) -> bool:
        """Check if batch retry is enabled."""
        return self.config.enabled

    @property
    def can_retry(self) -> bool:
        """Check if more retry passes are allowed."""
        return self.current_pass < self.config.max_passes

    def has_pending(self) -> bool:
        """Check if there are items waiting to be retried."""
        return len(self.items) > 0 and self.can_retry

    def add(
        self,
        video_id: str,
        keyword: str,
        tier: str,
        error_message: str,
        error_category: str = 'video_specific',
        escalation_tier: int = 1,
        duration_tier: str = "",
        match_confidence: float = 0.0,
        deadline: Optional[float] = None,
    ) -> bool:
        """Add a failed video to the retry queue.

        Args:
            video_id: YouTube video ID
            keyword: Search keyword that found this video
            tier: Duration tier (short, medium, long, longer)
            error_message: Error message from the failure
            error_category: 'network', 'bot_detection', 'timeout', or 'video_specific'.
                Network errors (DNS, no connectivity) affect all
                segments and should not be retried. Other errors
                (403, unavailable) may succeed on retry with escalation.
            escalation_tier: The escalation tier (1-4) that was in effect when
                the video failed. Retry will start at this tier or higher,
                avoiding wasted attempts at already-failed lower tiers.
            duration_tier: US-114-002 - Duration tier for smart prioritization (short, medium, long, longer)
            match_confidence: US-114-002 - Match confidence score (0.0-1.0) for priority calculation
            deadline: US-143-011 - Optional deadline timestamp (Unix epoch) for deadline-aware retry ordering.
                Videos with approaching deadlines get higher priority. None = no deadline.

        Returns:
            True if added to queue, False if disabled or already in queue.
        """
        if not self.config.enabled:
            return False

        # Import here to avoid circular import (core.py imports retry_queue.py)
        from .core import classify_error_severity

        # Check if already in queue
        if video_id in self.items:
            # Update error message, severity, category, and escalation tier but don't re-add
            self.items[video_id].error_message = error_message
            self.items[video_id].severity = classify_error_severity(error_message)
            self.items[video_id].error_category = error_category
            # Keep the higher escalation tier (don't regress)
            self.items[video_id].escalation_tier = max(
                self.items[video_id].escalation_tier, escalation_tier
            )
            logger.debug(f"Retry queue: {video_id} already queued, updated error")
            return False

        # Check if already completed or permanently failed
        if video_id in self._completed_ids or video_id in self._failed_ids:
            logger.debug(f"Retry queue: {video_id} already processed, skipping")
            return False

        # Classify error severity for adaptive delay scaling
        severity = classify_error_severity(error_message)

        self.items[video_id] = RetryItem(
            video_id=video_id,
            keyword=keyword,
            tier=tier,
            error_message=error_message,
            retry_count=0,
            severity=severity,
            error_category=error_category,
            escalation_tier=escalation_tier,
            duration_tier=duration_tier,
            match_confidence=match_confidence,
            deadline=deadline,
        )
        self._total_added += 1
        self._stats.record_failure(video_id, error_message)

        # US-93-006: Include structured error info for better debugging
        classified = classify_error_category(error_message)
        structured = StructuredDownloadError(classified, video_id=video_id, keyword=keyword)

        logger.debug(
            f"Retry queue: added {video_id} ({keyword}/{tier}) severity={severity} "
            f"category={error_category} error_code={structured.error_code.value} "
            f"message={structured.user_message} escalation_tier={escalation_tier} "
            f"- queue size now {len(self.items)}"
        )
        return True

    def get_pending_items(self) -> List[RetryItem]:
        """Get all items waiting to be retried.

        Returns:
            List of RetryItem objects in the queue.
        """
        return list(self.items.values())

    def get_items_with_budget(self) -> List[RetryItem]:
        """Get items that still have remaining retry budget.

        Filters out videos that have exhausted their retry budget
        (either max_attempts or max_backoff_time_seconds exceeded).

        Returns:
            List of RetryItem objects that have budget remaining.
        """
        if not self.processor._retry_budget:
            return list(self.items.values())

        budget = self.processor._retry_budget
        result = []
        skipped = []

        for item in self.items.values():
            if budget.is_exhausted(item.video_id):
                skipped.append(item.video_id)
                # Move to permanently failed
                self._failed_ids.add(item.video_id)
            else:
                result.append(item)

        if skipped:
            logger.warning(
                f"Retry queue: skipping {len(skipped)} videos with exhausted retry budget: {skipped[:5]}..."
            )

        return result

    def get_retryable_items(self) -> List[RetryItem]:
        """Get items that are worth retrying (excludes network errors).

        Network errors (DNS failure, no connectivity) affect all
        segments and won't resolve by retrying individual items. Only
        video-specific errors (403, removed) may succeed with escalation.

        Returns:
            List of RetryItem objects with error_category != 'network'.
        """
        return [
            item for item in self.items.values()
            if item.error_category != 'network'
        ]

    # =========================================================================
    # US-114-002: Smart retry queue prioritization
    # =========================================================================

    def calculate_segment_value_score(self, item: RetryItem) -> float:
        """Calculate segment value score for smart prioritization.

        Higher scores indicate higher priority (should be retried first).
        Score = (duration_score * duration_weight) + (confidence * confidence_weight) +
                ((max_retries - retry_count) * retry_weight_normalized) + deadline_urgency

        US-143-011: Deadline-aware scoring adds urgency boost when deadline is approaching.

        Args:
            item: RetryItem to calculate score for

        Returns:
            Priority score (higher = more important to retry first)
        """
        weighting = self.config.retry_priority_weighting
        duration_weight = weighting.get("duration", 0.5)
        confidence_weight = weighting.get("confidence", 0.3)
        retry_count_weight = weighting.get("retry_count", 0.2)

        # Duration score: map tier to numeric (longer = higher score)
        duration_scores = {
            "": 0.5,  # Unknown/default
            "short": 0.25,
            "medium": 0.5,
            "long": 0.75,
            "longer": 1.0,
        }
        duration_score = duration_scores.get(item.duration_tier.lower() if item.duration_tier else "", 0.5)

        # Confidence score: use directly (0.0-1.0)
        confidence_score = item.match_confidence if item.match_confidence else 0.5

        # Retry count score: lower retries = higher priority (so they eventually get retried)
        # Normalize: max_retries - retry_count, then normalize to 0-1 range
        max_retries = self.config.max_retries_per_video
        retry_score = (max_retries - item.retry_count) / max_retries if max_retries > 0 else 0.5

        # US-143-011: Deadline urgency scoring
        # Items with approaching deadlines get boosted priority
        deadline_urgency = 0.0
        if self.config.deadline_aware and item.deadline is not None:
            current_time = time.time()
            time_until_deadline = item.deadline - current_time

            if time_until_deadline <= 0:
                # Deadline passed - maximum urgency
                deadline_urgency = self.config.deadline_max_urgency_score
            elif time_until_deadline <= self.config.deadline_urgency_threshold_seconds:
                # Deadline approaching - scale urgency linearly from 0 to max
                urgency_ratio = 1.0 - (time_until_deadline / self.config.deadline_urgency_threshold_seconds)
                deadline_urgency = urgency_ratio * self.config.deadline_max_urgency_score
            # else: deadline far away - no urgency boost

        # Calculate weighted score (deadline urgency is additive boost)
        score = (
            (duration_score * duration_weight) +
            (confidence_score * confidence_weight) +
            (retry_score * retry_count_weight) +
            deadline_urgency
        )

        return score

    def get_prioritized_items(self) -> List[RetryItem]:
        """Get items sorted by priority score (highest first).

        US-114-002: Smart prioritization that balances:
        - Higher value segments (longer duration, higher confidence) get priority
        - Lower retry count segments get some priority to ensure eventual retry

        Returns:
            List of RetryItem objects sorted by priority score (highest first).
        """
        if not self.items:
            return []

        # Calculate scores for all items
        items_with_scores = [
            (item, self.calculate_segment_value_score(item))
            for item in self.items.values()
        ]

        # Sort by score descending (highest priority first)
        items_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return items in priority order
        return [item for item, score in items_with_scores]

    def mark_success(self, video_id: str) -> None:
        """Mark a video as successfully retried.

        Removes from queue and adds to completed set.

        Args:
            video_id: YouTube video ID that succeeded.
        """
        if video_id in self.items:
            del self.items[video_id]
            self._completed_ids.add(video_id)
            self._total_retried += 1
            self._stats.clear_failure(video_id)
            logger.debug(f"Retry queue: {video_id} succeeded, removed from queue")

    def mark_failed(self, video_id: str) -> None:
        """Mark a video as failed in the current retry pass.

        Increments retry count. If max passes reached after this pass,
        the video will be moved to permanently failed.

        Args:
            video_id: YouTube video ID that failed.
        """
        if video_id in self.items:
            self.items[video_id].retry_count += 1
            logger.debug(
                f"Retry queue: {video_id} failed again "
                f"(attempt {self.items[video_id].retry_count})"
            )

    @property
    def forced_retry(self) -> bool:
        """True if the last retry pass was forced due to combined wait timeout."""
        return self._forced_retry

    def start_retry_pass(self) -> int:
        """Start a new retry pass. Delegates to processor for execution logic.

        Increments the pass counter, checks circuit breaker state, checks cookie
        cooldowns, applies the delay, and logs the start. Should be called before
        processing items in the queue.

        When both circuit breaker AND cookie cooldown are active simultaneously,
        uses a combined wait strategy: wait for min(cb, cooldown) + buffer instead
        of waiting for both sequentially. If the combined wait would exceed
        max_combined_wait_seconds, forces retry with best-available cookie method.

        The circuit breaker and cookie cooldown wait times are additional to the
        retry delay (not subtracted from it).

        Returns:
            The new pass number (1-indexed).
        """
        return self.processor.start_retry_pass()

    def finish_retry_pass(self) -> None:
        """Finish the current retry pass.

        Moves videos that have exhausted all retry passes or exceeded
        max_retries_per_video to permanently failed. Should be called
        after processing all items in a pass.
        """
        # US-51-010: Check per-video retry limit
        per_video_exceeded = []
        for video_id, item in list(self.items.items()):
            if item.retry_count >= self.config.max_retries_per_video:
                # US-93-006: Include structured error info for better debugging
                classified = classify_error_category(item.error_message)
                structured = StructuredDownloadError(classified, video_id=video_id, keyword=item.keyword)
                per_video_exceeded.append((video_id, structured))
                self._failed_ids.add(video_id)
                del self.items[video_id]

        if per_video_exceeded:
            # Show sample error with code and suggestions
            sample = per_video_exceeded[0][1]
            logger.warning(
                f"Batch retry: {len(per_video_exceeded)} video(s) permanently "
                f"skipped after exceeding max_retries_per_video="
                f"{self.config.max_retries_per_video}. "
                f"Sample error: {sample.error_code.value} - {sample.user_message}. "
                f"Suggestions: {sample.suggestions[0] if sample.suggestions else 'None'}"
            )

        # Check if we've exhausted all passes
        if self.current_pass >= self.config.max_passes:
            # Move remaining items to permanently failed
            remaining = list(self.items.keys())
            for video_id in remaining:
                self._failed_ids.add(video_id)
                del self.items[video_id]

            if remaining:
                logger.warning(
                    f"Batch retry: {len(remaining)} videos failed after "
                    f"{self.config.max_passes} retry passes"
                )

    def clear(self) -> None:
        """Clear all state for a new session."""
        self.items.clear()
        self._completed_ids.clear()
        self._failed_ids.clear()
        self.current_pass = 0
        self._total_added = 0
        self._total_retried = 0
        self.processor.clear()  # Clear processor state (wait times, forced_retry)
        self._stats.reset()
        logger.debug("Retry queue: cleared for new session")

    def get_stats(self) -> dict:
        """Get retry queue statistics for reporting.

        Delegates to RetryQueueStats.get_summary() after syncing current state.

        Returns:
            Dict with stats including:
            - enabled: Whether batch retry is enabled
            - pending: Number of items currently queued
            - completed: Number of successfully retried videos
            - failed: Number of permanently failed videos
            - current_pass: Current retry pass number
            - max_passes: Maximum allowed passes
            - total_added: Total videos added to queue this session
            - total_retried: Total successful retries this session
            - respect_circuit_breaker: Whether circuit breaker is respected
            - circuit_breaker_wait_time: Total time spent waiting for circuit breaker
            - wait_for_cookie_cooldown: Whether cookie cooldown is respected
            - cookie_cooldown_wait_time: Total time spent waiting for cookie cooldown
        """
        self._sync_stats()
        return self._stats.get_summary()

    def get_failure_reasons(self) -> Dict[str, str]:
        """Get mapping of failed video IDs to their error messages.

        Delegates to RetryQueueStats.get_failure_reasons().

        Returns:
            Dict mapping video_id to the last error message received.
        """
        self._sync_stats()
        return self._stats.get_failure_reasons()

    def get_retry_metrics(self) -> Dict:
        """Calculate retry-specific metrics for analysis.

        Delegates to RetryQueueStats.get_retry_metrics().

        Returns:
            Dict with metrics including success_rate, retry_efficiency, etc.
        """
        self._sync_stats()
        return self._stats.get_retry_metrics()

    # =========================================================================
    # Cross-keyword retry learning (US-123-012)
    # =========================================================================

    def record_retry_attempt(self, keyword: str, success: bool) -> None:
        """Record a retry attempt for keyword category learning.

        Tracks retry success/failure by keyword and category to enable
        adaptive retry strategy recommendations.

        Args:
            keyword: The keyword associated with the retry
            success: Whether the retry was successful
        """
        if not self._cross_keyword_config or not getattr(self._cross_keyword_config, 'enabled', False):
            return

        # Get category for this keyword
        category = keyword_category_extractor(keyword)

        # Update keyword-level history
        if keyword not in self._keyword_retry_history:
            self._keyword_retry_history[keyword] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'category': category
            }
        self._keyword_retry_history[keyword]['attempts'] += 1
        if success:
            self._keyword_retry_history[keyword]['successes'] += 1
        else:
            self._keyword_retry_history[keyword]['failures'] += 1

        # Update category-level history
        if category not in self._category_retry_history:
            self._category_retry_history[category] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0
            }
        self._category_retry_history[category]['attempts'] += 1
        if success:
            self._category_retry_history[category]['successes'] += 1
        else:
            self._category_retry_history[category]['failures'] += 1

        logger.debug(
            f"Cross-keyword learning: recorded {keyword} ({category}) - "
            f"success={success}, category_history={self._category_retry_history[category]}"
        )

    def get_recommended_retry_strategy(self, keyword: str) -> RetryStrategyRecommendation:
        """Get recommended retry strategy based on keyword category history.

        Analyzes retry history for the keyword and its category to recommend
        optimal retry parameters. Can learn from similar keywords in the same
        category when keyword-specific data is insufficient.

        Args:
            keyword: The keyword to get recommendations for

        Returns:
            RetryStrategyRecommendation with delay multiplier, max attempts, and confidence
        """
        # Default recommendation
        default_recommendation = RetryStrategyRecommendation(
            recommended_delay_multiplier=1.0,
            recommended_max_attempts=2,
            confidence=0.0,
            source='default',
            category='general'
        )

        if not self._cross_keyword_config or not getattr(self._cross_keyword_config, 'enabled', False):
            return default_recommendation

        # Get config parameters
        min_attempts = getattr(self._cross_keyword_config, 'min_attempts_for_recommendation', 3)
        confidence_threshold = getattr(self._cross_keyword_config, 'confidence_threshold', 0.6)
        enable_category_learning = getattr(self._cross_keyword_config, 'enable_category_learning', True)

        # Get category for this keyword
        category = keyword_category_extractor(keyword)

        # Check keyword-specific history first
        keyword_history = self._keyword_retry_history.get(keyword, {})
        keyword_attempts = keyword_history.get('attempts', 0)

        if keyword_attempts >= min_attempts:
            # Have enough keyword-specific data
            successes = keyword_history.get('successes', 0)
            failures = keyword_history.get('failures', 0)
            success_rate = successes / keyword_attempts if keyword_attempts > 0 else 0.0

            # Calculate strategy based on success rate
            if success_rate >= 0.7:
                # High success rate - aggressive retries
                multiplier = 0.8
                max_attempts = 3
            elif success_rate >= 0.4:
                # Medium success rate - normal retries
                multiplier = 1.0
                max_attempts = 2
            else:
                # Low success rate - more conservative with longer delays
                multiplier = 1.5
                max_attempts = 3

            confidence = min(1.0, keyword_attempts / 10.0)  # Max confidence at 10 attempts

            return RetryStrategyRecommendation(
                recommended_delay_multiplier=multiplier,
                recommended_max_attempts=max_attempts,
                confidence=confidence,
                source='keyword',
                category=category
            )

        # Not enough keyword data - try category-based learning
        if enable_category_learning and category in self._category_retry_history:
            cat_history = self._category_retry_history[category]
            cat_attempts = cat_history.get('attempts', 0)

            if cat_attempts >= min_attempts * 2:  # Need more category data
                successes = cat_history.get('successes', 0)
                failures = cat_history.get('failures', 0)
                success_rate = successes / cat_attempts if cat_attempts > 0 else 0.0

                if success_rate >= 0.7:
                    multiplier = 0.8
                    max_attempts = 3
                elif success_rate >= 0.4:
                    multiplier = 1.0
                    max_attempts = 2
                else:
                    multiplier = 1.5
                    max_attempts = 3

                # Category-based confidence is lower - use scaled threshold
                # Category learning needs at least min_attempts * 2 to be useful
                category_confidence_threshold = confidence_threshold * 0.7
                confidence = min(category_confidence_threshold, cat_attempts / 20.0)

                if confidence >= category_confidence_threshold:
                    return RetryStrategyRecommendation(
                        recommended_delay_multiplier=multiplier,
                        recommended_max_attempts=max_attempts,
                        confidence=confidence,
                        source='category',
                        category=category
                    )

        # Fall back to cross-category learning if enabled
        if enable_category_learning:
            # Find similar categories with more data
            similar_category = self._find_similar_category_with_history(category)
            if similar_category:
                cat_history = self._category_retry_history.get(similar_category, {})
                cat_attempts = cat_history.get('attempts', 0)

                if cat_attempts >= min_attempts * 3:
                    successes = cat_history.get('successes', 0)
                    success_rate = successes / cat_attempts if cat_attempts > 0 else 0.0

                    if success_rate >= 0.7:
                        multiplier = 0.8
                        max_attempts = 3
                    elif success_rate >= 0.4:
                        multiplier = 1.0
                        max_attempts = 2
                    else:
                        multiplier = 1.5
                        max_attempts = 3

                    # Lower confidence for cross-category
                    confidence = min(confidence_threshold * 0.7, cat_attempts / 30.0)

                    if confidence >= confidence_threshold * 0.5:
                        return RetryStrategyRecommendation(
                            recommended_delay_multiplier=multiplier,
                            recommended_max_attempts=max_attempts,
                            confidence=confidence,
                            source='cross_category',
                            category=similar_category
                        )

        return default_recommendation

    def _find_similar_category_with_history(self, category: str) -> Optional[str]:
        """Find a similar category with sufficient retry history.

        Uses simple category similarity heuristics to find categories
        that might have similar retry patterns.

        Args:
            category: The category to find similar matches for

        Returns:
            Category name with history, or None if not found
        """
        # Define category similarity groups
        category_groups = {
            'tech': ['tutorial', 'education', 'general'],
            'news': ['general', 'documentary'],
            'tutorial': ['tech', 'education', 'general'],
            'music': ['entertainment', 'general'],
            'gaming': ['entertainment', 'general'],
            'sports': ['general', 'entertainment'],
            'cooking': ['general', 'lifestyle'],
            'fitness': ['general', 'lifestyle'],
            'education': ['tutorial', 'tech', 'general'],
            'entertainment': ['general', 'music', 'gaming'],
            'science': ['documentary', 'nature', 'general'],
            'nature': ['documentary', 'science', 'general'],
            'stock_footage': ['cinematic', 'general'],
            'cinematic': ['stock_footage', 'documentary', 'general'],
            'documentary': ['cinematic', 'news', 'science', 'nature', 'general'],
            'travel': ['documentary', 'cinematic', 'general'],
            'interview': ['documentary', 'news', 'general'],
            'general': []  # General is fallback, not used for cross-learning
        }

        # Get similar categories
        similar = category_groups.get(category, [])

        # Find one with sufficient history
        min_attempts = getattr(self._cross_keyword_config, 'min_attempts_for_recommendation', 3)
        for sim_cat in similar:
            if sim_cat in self._category_retry_history:
                cat_history = self._category_retry_history[sim_cat]
                if cat_history.get('attempts', 0) >= min_attempts * 2:
                    return sim_cat

        return None

    def get_category_stats(self) -> Dict[str, Dict]:
        """Get retry statistics by category.

        Returns:
            Dict mapping category name to retry statistics
        """
        return copy.deepcopy(self._category_retry_history)

    def get_keyword_stats(self) -> Dict[str, Dict]:
        """Get retry statistics by keyword.

        Returns:
            Dict mapping keyword to retry statistics
        """
        return copy.deepcopy(self._keyword_retry_history)

    def _sync_stats(self) -> None:
        """Sync internal state to RetryQueueStats for accurate reporting."""
        self._stats.pending = len(self.items)
        self._stats.completed_ids = self._completed_ids.copy()
        self._stats.failed_ids = self._failed_ids.copy()
        self._stats.current_pass = self.current_pass
        self._stats.total_added = self._total_added
        self._stats.total_retried = self._total_retried
        self._stats.circuit_breaker_wait_time = self._circuit_breaker_wait_time
        self._stats.cookie_cooldown_wait_time = self._cookie_cooldown_wait_time
        self._stats.forced_retry = self._forced_retry
        self._stats.budget_state = self._budget_state

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence.

        Returns:
            Dict that can be saved to checkpoint JSON.
        """
        checkpoint = {
            'items': [
                {
                    'video_id': item.video_id,
                    'keyword': item.keyword,
                    'tier': item.tier,
                    'error_message': item.error_message,
                    'failure_reason': item.error_message,  # US-51-010: alias
                    'retry_count': item.retry_count,
                    'error_category': item.error_category,
                    'escalation_tier': item.escalation_tier,
                    'last_tier_attempted': item.escalation_tier,  # US-51-010: alias
                    'timestamp': item.added_at,  # US-51-010: when video was first queued
                    # US-114-002: Segment value fields for smart prioritization
                    'duration_tier': item.duration_tier,
                    'match_confidence': item.match_confidence,
                    # US-143-011: Deadline-aware retry ordering
                    'deadline': item.deadline,
                }
                for item in self.items.values()
            ],
            'current_pass': self.current_pass,
            'completed_ids': list(self._completed_ids),
            'failed_ids': list(self._failed_ids),
            'total_added': self._total_added,
            'total_retried': self._total_retried,
        }
        # Merge processor checkpoint data
        checkpoint.update(self.processor.to_checkpoint_dict())
        return checkpoint

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore state from checkpoint dictionary.

        Args:
            data: Dict from checkpoint JSON.
        """
        if not data:
            return

        # Restore items
        self.items.clear()
        for item_data in data.get('items', []):
            video_id = item_data.get('video_id')
            if video_id:
                # US-51-010: Skip videos that have exceeded max_retries_per_video
                retry_count = item_data.get('retry_count', 0)
                if retry_count >= self.config.max_retries_per_video:
                    self._failed_ids.add(video_id)
                    logger.info(
                        f"Retry queue: permanently skipping {video_id} "
                        f"(retry_count={retry_count} >= max_retries_per_video="
                        f"{self.config.max_retries_per_video})"
                    )
                    continue
                self.items[video_id] = RetryItem(
                    video_id=video_id,
                    keyword=item_data.get('keyword', ''),
                    tier=item_data.get('tier', 'short'),
                    error_message=item_data.get('error_message', ''),
                    retry_count=retry_count,
                    added_at=item_data.get('timestamp', time.time()),
                    error_category=item_data.get('error_category', 'video_specific'),
                    escalation_tier=item_data.get('escalation_tier', 1),
                    # US-114-002: Segment value fields for smart prioritization
                    duration_tier=item_data.get('duration_tier', ''),
                    match_confidence=item_data.get('match_confidence', 0.0),
                    # US-143-011: Deadline-aware retry ordering
                    deadline=item_data.get('deadline'),
                )

        # Restore state (merge failed_ids to preserve US-51-010 max_retries_per_video skips)
        self.current_pass = data.get('current_pass', 0)
        self._completed_ids = set(data.get('completed_ids', []))
        self._failed_ids.update(data.get('failed_ids', []))
        self._total_added = data.get('total_added', 0)
        self._total_retried = data.get('total_retried', 0)

        # Restore processor state
        self.processor.from_checkpoint_dict(data)

        if self.items:
            logger.debug(
                f"Retry queue: restored {len(self.items)} items from checkpoint "
                f"(pass {self.current_pass}/{self.config.max_passes})"
            )
