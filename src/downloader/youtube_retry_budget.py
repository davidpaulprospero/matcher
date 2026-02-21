"""YouTube API retry budget management.

Provides retry budget tracking for YouTube API calls similar to CaptionRetryBudget.
Tracks total attempts and backoff time to prevent infinite retry loops on transient errors.

Includes:
- YouTubeAPIRetryBudget: Per-batch retry tracking with auto-scaling
- YouTubeAPIRetryBudgetConfig: Configuration dataclass
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# History file for quota prediction data
QUOTA_HISTORY_FILE = os.path.expanduser("~/.matcher_youtube_api_quota_history.json")

# Time-of-day periods for prediction
TIME_PERIODS = {
    "morning": (6, 12),      # 6 AM - 12 PM
    "afternoon": (12, 18),   # 12 PM - 6 PM
    "evening": (18, 22),     # 6 PM - 10 PM
    "overnight": (22, 6),    # 10 PM - 6 AM
}


@dataclass
class YouTubeAPIRetryBudgetConfig:
    """Configuration for YouTubeAPIRetryBudget (US-149-004).

    Controls when YouTube API calls should stop due to resource exhaustion.

    Configure in config.yaml under youtube_api.retry_budget.
    """
    # Enable/disable retry budget tracking
    enabled: bool = True

    # Maximum total API call attempts across entire batch
    # Set to 0 for unlimited attempts
    max_attempts: int = 100

    # Maximum cumulative backoff time (seconds) before exhaustion
    # Set to 0 for unlimited backoff
    max_backoff_time_seconds: float = 300.0

    # Automatic scaling settings
    # When enabled, max_attempts scales up based on batch size
    auto_scale: bool = True

    # Attempts per video multiplier for auto-scaling
    # 2.0 = 1 attempt + 1 retry per video average
    # Only scales UP when batch > max_attempts / attempts_per_video
    attempts_per_video: float = 2.0


class YouTubeAPIRetryBudget:
    """Tracks retry resources used across all YouTube API operations (US-149-004).

    Tracks total resources used across the entire batch:
    - Total API call attempts (success + failures)
    - Total failures
    - Total backoff time spent waiting

    When any limit is exceeded, budget_exhausted() returns True, signaling the
    client to skip remaining operations and proceed with fallback (yt-dlp).

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access.

    Attributes:
        attempts: Total API call attempts.
        failures: Total failures.
        successes: Total successes.
        backoff_time_spent: Cumulative backoff delay (seconds).
        videos_skipped: Count of videos skipped due to budget exhaustion.
        max_attempts: Maximum allowed attempts (0 = unlimited).
        max_backoff_time: Maximum allowed backoff time (0 = unlimited).
    """

    def __init__(
        self,
        max_attempts: int = 100,
        max_backoff_time: float = 300.0,
        auto_scale: bool = True,
        attempts_per_video: float = 2.0,
        budget_warning_threshold: float = 0.8,
    ):
        """Initialize YouTubeAPIRetryBudget.

        Args:
            max_attempts: Maximum total attempts (0 = unlimited)
            max_backoff_time: Maximum cumulative backoff time in seconds (0 = unlimited)
            auto_scale: Whether to auto-scale based on batch size
            attempts_per_video: Attempts per video for auto-scaling
            budget_warning_threshold: Threshold for warning (0.0-1.0)
        """
        # Usage counters
        self.attempts: int = 0
        self.failures: int = 0
        self.successes: int = 0
        self.backoff_time_spent: float = 0.0
        self.videos_skipped: int = 0

        # Batch size tracking
        self.batch_size: Optional[int] = None

        # Budget limits
        self.max_attempts: int = max_attempts
        self.original_max_attempts: int = max_attempts
        self.max_backoff_time: float = max_backoff_time

        # Auto-scaling settings
        self.auto_scale: bool = auto_scale
        self.attempts_per_video: float = attempts_per_video

        # Budget warning threshold
        self.budget_warning_threshold: float = budget_warning_threshold

        # Track if warning has been logged
        self._warned_budget_threshold: bool = False

        # Thread-safety lock
        self._lock: threading.RLock = threading.RLock()

        # Per-video attempt tracking
        self.attempts_per_video_id: Dict[str, int] = {}

    @classmethod
    def from_config(cls, config: Optional[YouTubeAPIRetryBudgetConfig]) -> "YouTubeAPIRetryBudget":
        """Create a YouTubeAPIRetryBudget from config.

        Args:
            config: YouTubeAPIRetryBudgetConfig instance or None for defaults.

        Returns:
            YouTubeAPIRetryBudget with limits set from config.
        """
        if config is None:
            return cls()

        # Handle both dict and dataclass config
        if isinstance(config, dict):
            max_attempts = int(config.get('max_attempts', 100))
            max_backoff_time = float(config.get('max_backoff_time_seconds', 300.0))
            auto_scale = bool(config.get('auto_scale', True))
            attempts_per_video = float(config.get('attempts_per_video', 2.0))
            budget_warning_threshold = float(config.get('budget_warning_threshold', 0.8))
        else:
            max_attempts = int(getattr(config, 'max_attempts', 100))
            max_backoff_time = float(getattr(config, 'max_backoff_time_seconds', 300.0))
            auto_scale = bool(getattr(config, 'auto_scale', True))
            attempts_per_video = float(getattr(config, 'attempts_per_video', 2.0))
            budget_warning_threshold = float(getattr(config, 'budget_warning_threshold', 0.8))

        return cls(
            max_attempts=max_attempts,
            max_backoff_time=max_backoff_time,
            auto_scale=auto_scale,
            attempts_per_video=attempts_per_video,
            budget_warning_threshold=budget_warning_threshold,
        )

    def set_batch_size(self, batch_size: int) -> None:
        """Set batch size and auto-scale max_attempts if enabled.

        Args:
            batch_size: Number of videos/items in the batch
        """
        with self._lock:
            self.batch_size = batch_size
            if self.auto_scale and batch_size > 0:
                # Calculate scaled max attempts
                scaled_max = max(self.original_max_attempts, int(batch_size * self.attempts_per_video))
                if scaled_max != self.max_attempts:
                    logger.info(
                        f"YouTube API retry budget: auto-scaling max_attempts from "
                        f"{self.max_attempts} to {scaled_max} (batch_size={batch_size}, "
                        f"attempts_per_video={self.attempts_per_video})"
                    )
                    self.max_attempts = scaled_max

            # US-155-012: Log budget status at INFO when batch is configured
            self._log_budget_status("INFO")

    def _log_budget_status(self, level: str = "INFO") -> None:
        """Log current retry budget status.

        Args:
            level: Log level to use ("INFO" or "DEBUG")
        """
        if self.max_attempts > 0:
            remaining = self.max_attempts - self.attempts
            utilization = (self.attempts / self.max_attempts * 100) if self.max_attempts > 0 else 0
            msg = (
                f"YouTube API retry budget: {self.attempts}/{self.max_attempts} attempts used "
                f"({utilization:.1f}%), {remaining} remaining, "
                f"successes={self.successes}, failures={self.failures}"
            )
        else:
            msg = (
                f"YouTube API retry budget: unlimited, "
                f"attempts={self.attempts}, successes={self.successes}, failures={self.failures}"
            )

        if self.backoff_time_spent > 0:
            if self.max_backoff_time > 0:
                backoff_util = (self.backoff_time_spent / self.max_backoff_time * 100)
                msg += f", backoff={self.backoff_time_spent:.1f}s/{self.max_backoff_time:.1f}s ({backoff_util:.1f}%)"
            else:
                msg += f", backoff={self.backoff_time_spent:.1f}s"

        if self.videos_skipped > 0:
            msg += f", videos_skipped={self.videos_skipped}"

        if level == "INFO":
            logger.info(msg)
        elif level == "DEBUG":
            logger.debug(msg)
        else:
            logger.info(msg)

    def get_budget_status(self) -> Dict[str, any]:
        """Get current budget utilization status.

        Returns:
            Dict with attempts remaining, utilization percentage, etc.
        """
        with self._lock:
            if self.max_attempts > 0:
                remaining = self.max_attempts - self.attempts
                utilization = (self.attempts / self.max_attempts * 100) if self.max_attempts > 0 else 0
            else:
                remaining = -1  # Unlimited
                utilization = 0.0

            if self.max_backoff_time > 0:
                backoff_remaining = self.max_backoff_time - self.backoff_time_spent
                backoff_utilization = (self.backoff_time_spent / self.max_backoff_time * 100) if self.max_backoff_time > 0 else 0
            else:
                backoff_remaining = -1  # Unlimited
                backoff_utilization = 0.0

            return {
                "attempts_used": self.attempts,
                "attempts_remaining": remaining,
                "attempts_max": self.max_attempts,
                "attempts_utilization_percent": utilization,
                "successes": self.successes,
                "failures": self.failures,
                "backoff_time_spent": self.backoff_time_spent,
                "backoff_time_remaining": backoff_remaining,
                "backoff_time_max": self.max_backoff_time,
                "backoff_utilization_percent": backoff_utilization,
                "videos_skipped": self.videos_skipped,
                "budget_exhausted": self.budget_exhausted(),
                "batch_size": self.batch_size,
            }

    def record_attempt(self, video_id: str = "") -> None:
        """Record an API call attempt.

        Args:
            video_id: Optional video ID for per-video tracking
        """
        with self._lock:
            self.attempts += 1
            if video_id:
                self.attempts_per_video_id[video_id] = self.attempts_per_video_id.get(video_id, 0) + 1
            self._check_budget_warning()

    def record_success(self, video_id: str = "") -> None:
        """Record a successful API call.

        Args:
            video_id: Optional video ID for per-video tracking
        """
        with self._lock:
            self.successes += 1

    def record_failure(self, video_id: str = "") -> None:
        """Record a failed API call.

        Args:
            video_id: Optional video ID for per-video tracking
        """
        with self._lock:
            self.failures += 1

    def record_backoff(self, seconds: float) -> None:
        """Record time spent in backoff delay.

        Args:
            seconds: Number of seconds waited
        """
        with self._lock:
            self.backoff_time_spent += seconds
            self._check_budget_warning()

    def get_backoff_time(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff time.

        Args:
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap

        Returns:
            Calculated delay in seconds
        """
        delay = base_delay * (2 ** attempt)
        return min(delay, max_delay)

    def wait_with_backoff(self, attempt: int, video_id: str = "") -> float:
        """Wait with exponential backoff and record the time spent.

        Args:
            attempt: Current attempt number (0-indexed)
            video_id: Optional video ID for logging

        Returns:
            Actual time spent waiting
        """
        delay = self.get_backoff_time(attempt)
        if delay > 0:
            logger.debug(f"YouTube API backoff: waiting {delay:.1f}s (attempt {attempt + 1})")
            time.sleep(delay)
            self.record_backoff(delay)
        return delay

    def _check_budget_warning(self) -> None:
        """Check if budget threshold warning should be emitted."""
        if self._warned_budget_threshold:
            return

        if self.max_attempts > 0:
            ratio = self.attempts / self.max_attempts
            if ratio >= self.budget_warning_threshold:
                logger.warning(
                    f"YouTube API retry budget warning: {self.attempts}/{self.max_attempts} "
                    f"({ratio:.0%}) attempts used"
                )
                self._warned_budget_threshold = True
                return

        if self.max_backoff_time > 0:
            ratio = self.backoff_time_spent / self.max_backoff_time
            if ratio >= self.budget_warning_threshold:
                logger.warning(
                    f"YouTube API retry budget warning: {self.backoff_time_spent:.1f}/{self.max_backoff_time:.1f}s "
                    f"({ratio:.0%}) backoff time used"
                )
                self._warned_budget_threshold = True

    def budget_exhausted(self) -> bool:
        """Check if the retry budget is exhausted.

        Returns True when:
        - max_attempts > 0 and attempts >= max_attempts, OR
        - max_backoff_time > 0 and backoff_time_spent >= max_backoff_time

        Returns:
            True if budget is exhausted, False otherwise
        """
        with self._lock:
            if self.max_attempts > 0 and self.attempts >= self.max_attempts:
                logger.warning(
                    f"YouTube API retry budget EXHAUSTED: {self.attempts}/{self.max_attempts} attempts used"
                )
                return True

            if self.max_backoff_time > 0 and self.backoff_time_spent >= self.max_backoff_time:
                logger.warning(
                    f"YouTube API retry budget EXHAUSTED: {self.backoff_time_spent:.1f}s/{self.max_backoff_time:.1f}s "
                    f"backoff time used"
                )
                return True

            return False

    def skip_video(self) -> None:
        """Record a video skip due to budget exhaustion."""
        with self._lock:
            self.videos_skipped += 1

    def get_stats(self) -> Dict[str, any]:
        """Get current budget statistics.

        Returns:
            Dict with attempts, failures, successes, backoff_time_spent, etc.
        """
        with self._lock:
            return {
                "attempts": self.attempts,
                "failures": self.failures,
                "successes": self.successes,
                "backoff_time_spent": self.backoff_time_spent,
                "videos_skipped": self.videos_skipped,
                "max_attempts": self.max_attempts,
                "max_backoff_time": self.max_backoff_time,
                "batch_size": self.batch_size,
                "budget_exhausted": self.budget_exhausted(),
            }

    def reset(self) -> None:
        """Reset all counters and warnings for reuse."""
        with self._lock:
            self.attempts = 0
            self.failures = 0
            self.successes = 0
            self.backoff_time_spent = 0.0
            self.videos_skipped = 0
            self.attempts_per_video_id.clear()
            self._warned_budget_threshold = False


class QuotaPredictor:
    """Predicts and allocates YouTube API quota based on project size (US-153-003).

    Estimates required quota before pipeline starts based on:
    - Voiceover segment count
    - Keyword count
    - Estimated videos per keyword

    Provides:
    - Pre-flight quota check that warns if estimated usage exceeds available quota
    - Dynamic allocation strategy (balanced, search_first, caption_first)
    - Per-operation type quota tracking metrics

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access.
    """

    def __init__(
        self,
        quota_limit: int = 10000,
        allocation_strategy: str = "balanced",
        estimated_quota_per_search: int = 100,
        estimated_quota_per_caption: int = 50,
        estimated_quota_per_metadata: int = 1,
        enable_pre_flight_check: bool = True,
        enable_operation_metrics: bool = True,
    ):
        """Initialize QuotaPredictor.

        Args:
            quota_limit: Total available quota (default: 10,000 for free tier)
            allocation_strategy: Strategy for quota allocation (balanced, search_first, caption_first)
            estimated_quota_per_search: Estimated quota cost per search operation
            estimated_quota_per_caption: Estimated quota cost per caption operation
            estimated_quota_per_metadata: Estimated quota cost per metadata operation
            enable_pre_flight_check: Whether to warn if estimated usage exceeds quota
            enable_operation_metrics: Whether to track per-operation metrics
        """
        self.quota_limit = quota_limit
        self.allocation_strategy = allocation_strategy
        self.estimated_quota_per_search = estimated_quota_per_search
        self.estimated_quota_per_caption = estimated_quota_per_caption
        self.estimated_quota_per_metadata = estimated_quota_per_metadata
        self.enable_pre_flight_check = enable_pre_flight_check
        self.enable_operation_metrics = enable_operation_metrics

        # Project size estimates (set before pipeline starts)
        self.segment_count: int = 0
        self.keyword_count: int = 0
        self.estimated_videos_per_keyword: int = 50  # Default: 50 videos per keyword

        # Allocated quota per operation type
        self.search_quota: int = 0
        self.caption_quota: int = 0
        self.metadata_quota: int = 0
        self.estimated_total_quota: int = 0

        # Actual usage tracking (if enable_operation_metrics is True)
        self._search_used: int = 0
        self._caption_used: int = 0
        self._metadata_used: int = 0

        # Thread-safety lock
        self._lock = threading.RLock()

        # Pre-flight warning flag
        self._pre_flight_warning_logged: bool = False

        # Historical data for prediction (US-156-009)
        self._history: List[Dict] = []
        self._history_file: str = QUOTA_HISTORY_FILE
        self._last_prediction: Optional[Dict] = None
        self._prediction_confidence: str = "low"  # low, medium, high
        self._weighted_avg_window: int = 7  # Days to consider for weighted average

        # US-157-002: Quota prediction accuracy tracking
        self._prediction_records: List[Dict] = []  # Stores (predicted, actual) pairs
        self._total_predictions: int = 0
        self._accurate_predictions: int = 0  # Within 20% of actual
        self._prediction_error_sum: float = 0.0  # Sum of percentage errors

    @classmethod
    def from_config(cls, config_dict: dict) -> "QuotaPredictor":
        """Create a QuotaPredictor from config dict.

        Args:
            config_dict: Configuration dictionary from download.youtube_api

        Returns:
            QuotaPredictor with settings from config
        """
        if config_dict is None:
            return cls()

        return cls(
            quota_limit=int(config_dict.get('quota_limit', 10000)),
            allocation_strategy=str(config_dict.get('quota_allocation_strategy', 'balanced')),
            estimated_quota_per_search=int(config_dict.get('estimated_quota_per_search', 100)),
            estimated_quota_per_caption=int(config_dict.get('estimated_quota_per_caption', 50)),
            estimated_quota_per_metadata=int(config_dict.get('estimated_quota_per_metadata', 1)),
            enable_pre_flight_check=bool(config_dict.get('enable_pre_flight_check', True)),
            enable_operation_metrics=bool(config_dict.get('enable_operation_metrics', True)),
        )

    def set_project_size(
        self,
        segment_count: int,
        keyword_count: int,
        estimated_videos_per_keyword: int = 50,
    ) -> None:
        """Set project size estimates for quota prediction.

        Args:
            segment_count: Number of voiceover segments
            keyword_count: Number of search keywords
            estimated_videos_per_keyword: Estimated videos to fetch per keyword
        """
        with self._lock:
            self.segment_count = segment_count
            self.keyword_count = keyword_count
            self.estimated_videos_per_keyword = estimated_videos_per_keyword

    def estimate_quota(self) -> int:
        """Estimate total quota needed for the project.

        Returns:
            Estimated total quota needed
        """
        with self._lock:
            # Estimate searches needed
            # Each keyword typically needs multiple searches for different topics
            estimated_searches = self.keyword_count * 2  # 2 searches per keyword average

            # Estimate caption operations
            # Assume we need captions for ~50% of found videos
            estimated_videos = self.keyword_count * self.estimated_videos_per_keyword
            estimated_caption_calls = int(estimated_videos * 0.5)

            # Estimate metadata operations
            # Each video needs at least one metadata call
            estimated_metadata_calls = estimated_videos

            # Calculate total
            total = (
                estimated_searches * self.estimated_quota_per_search +
                estimated_caption_calls * self.estimated_quota_per_caption +
                estimated_metadata_calls * self.estimated_quota_per_metadata
            )

            self.estimated_total_quota = total
            return total

    def allocate_quota(self) -> Dict[str, int]:
        """Allocate quota across operation types based on strategy.

        Returns:
            Dict with allocated quota per operation type
        """
        with self._lock:
            if self.estimated_total_quota == 0:
                self.estimate_quota()

            available = self.quota_limit

            if self.allocation_strategy == "search_first":
                # Prioritize search operations
                search_needed = self.keyword_count * 2 * self.estimated_quota_per_search
                search_allocation = min(search_needed, available)

                remaining = available - search_allocation
                # Split remaining between captions and metadata
                caption_metadata_ratio = 0.7  # More to captions
                caption_allocation = int(remaining * caption_metadata_ratio)
                metadata_allocation = remaining - caption_allocation

                self.search_quota = search_allocation
                self.caption_quota = caption_allocation
                self.metadata_quota = metadata_allocation

            elif self.allocation_strategy == "caption_first":
                # Prioritize caption operations
                estimated_videos = self.keyword_count * self.estimated_videos_per_keyword
                caption_needed = int(estimated_videos * 0.5) * self.estimated_quota_per_caption
                caption_allocation = min(caption_needed, available)

                remaining = available - caption_allocation
                # Split remaining between search and metadata
                search_metadata_ratio = 0.6  # More to search
                search_allocation = int(remaining * search_metadata_ratio)
                metadata_allocation = remaining - search_allocation

                self.search_quota = search_allocation
                self.caption_quota = caption_allocation
                self.metadata_quota = metadata_allocation

            else:  # "balanced"
                # Equal distribution based on estimated needs
                search_need = self.keyword_count * 2 * self.estimated_quota_per_search
                estimated_videos = self.keyword_count * self.estimated_videos_per_keyword
                caption_need = int(estimated_videos * 0.5) * self.estimated_quota_per_caption
                metadata_need = estimated_videos * self.estimated_quota_per_metadata

                total_need = search_need + caption_need + metadata_need
                if total_need > available:
                    # Scale down proportionally
                    scale = available / total_need
                    search_need = int(search_need * scale)
                    caption_need = int(caption_need * scale)
                    metadata_need = int(metadata_need * scale)

                self.search_quota = search_need
                self.caption_quota = caption_need
                self.metadata_quota = metadata_need

            return {
                "search": self.search_quota,
                "caption": self.caption_quota,
                "metadata": self.metadata_quota,
                "total": self.search_quota + self.caption_quota + self.metadata_quota,
            }

    def pre_flight_check(self) -> Dict[str, any]:
        """Run pre-flight quota check and warn if estimated usage exceeds available.

        Returns:
            Dict with check results including warnings
        """
        with self._lock:
            estimated = self.estimate_quota()
            allocated = self.allocate_quota()

            results = {
                "estimated_quota": estimated,
                "available_quota": self.quota_limit,
                "allocation": allocated,
                "strategy": self.allocation_strategy,
                "warnings": [],
            }

            if self.enable_pre_flight_check:
                if estimated > self.quota_limit:
                    results["warnings"].append(
                        f"Estimated quota ({estimated}) exceeds available quota ({self.quota_limit}). "
                        f"Pipeline may need to fall back to yt-dlp for some operations."
                    )
                    logger.warning(results["warnings"][-1])

                # Check if allocation leaves critical operations underfunded
                if self.search_quota < self.keyword_count * self.estimated_quota_per_search:
                    results["warnings"].append(
                        f"Search operations may be underfunded. Consider switching to 'search_first' strategy."
                    )
                    logger.warning(results["warnings"][-1])

            self._pre_flight_warning_logged = len(results["warnings"]) > 0
            return results

    def record_search(self, count: int = 1) -> None:
        """Record search API quota usage.

        Args:
            count: Number of search operations
        """
        if self.enable_operation_metrics:
            with self._lock:
                self._search_used += count * self.estimated_quota_per_search

    def record_caption(self, count: int = 1) -> None:
        """Record caption API quota usage.

        Args:
            count: Number of caption operations
        """
        if self.enable_operation_metrics:
            with self._lock:
                self._caption_used += count * self.estimated_quota_per_caption

    def record_metadata(self, count: int = 1) -> None:
        """Record metadata API quota usage.

        Args:
            count: Number of metadata operations
        """
        if self.enable_operation_metrics:
            with self._lock:
                self._metadata_used += count * self.estimated_quota_per_metadata

    def get_usage_metrics(self) -> Dict[str, any]:
        """Get current quota usage metrics by operation type.

        Returns:
            Dict with usage metrics
        """
        with self._lock:
            total_used = self._search_used + self._caption_used + self._metadata_used

            return {
                "search_used": self._search_used,
                "caption_used": self._caption_used,
                "metadata_used": self._metadata_used,
                "total_used": total_used,
                "available": self.quota_limit,
                "remaining": self.quota_limit - total_used,
                "usage_percent": (total_used / self.quota_limit * 100) if self.quota_limit > 0 else 0,
                "search_percent": (self._search_used / self.search_quota * 100) if self.search_quota > 0 else 0,
                "caption_percent": (self._caption_used / self.caption_quota * 100) if self.caption_quota > 0 else 0,
                "metadata_percent": (self._metadata_used / self.metadata_quota * 100) if self.metadata_quota > 0 else 0,
            }

    def get_allocation(self) -> Dict[str, int]:
        """Get current quota allocation.

        Returns:
            Dict with allocated quota per operation type
        """
        with self._lock:
            return {
                "search": self.search_quota,
                "caption": self.caption_quota,
                "metadata": self.metadata_quota,
            }

    def load_history(self) -> List[Dict]:
        """Load historical usage patterns from file (US-156-009).

        Loads quota usage history from ~/.matcher_youtube_api_quota_history.json
        to enable prediction based on past usage patterns.

        Returns:
            List of historical usage records
        """
        with self._lock:
            self._history = []

            if not os.path.exists(self._history_file):
                logger.debug(f"Quota history file not found: {self._history_file}")
                return self._history

            try:
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both old format (list) and new format (dict with 'history')
                    if isinstance(data, list):
                        self._history = data
                    elif isinstance(data, dict):
                        self._history = data.get('history', [])
                    logger.info(f"Loaded {len(self._history)} historical quota records")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load quota history: {e}")

            return self._history

    def _get_time_period(self, hour: int) -> str:
        """Determine time period from hour of day.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Time period name: morning, afternoon, evening, overnight
        """
        for period, (start, end) in TIME_PERIODS.items():
            if start > end:
                # Overnight period wraps around midnight
                if hour >= start or hour < end:
                    return period
            else:
                if start <= hour < end:
                    return period
        return "overnight"

    def _get_time_multiplier(self, hour: int) -> float:
        """Get quota usage multiplier based on time of day.

        Peak usage times (evening) have higher multipliers as API tends
        to be more constrained. Off-peak (overnight) has lower multipliers.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Multiplier for quota prediction
        """
        period = self._get_time_period(hour)
        multipliers = {
            "morning": 1.0,     # Baseline
            "afternoon": 1.1,    # Slightly higher
            "evening": 1.3,      # Peak usage - API more constrained
            "overnight": 0.7,    # Lower usage
        }
        return multipliers.get(period, 1.0)

    def predict_daily_usage(self) -> Dict[str, any]:
        """Predict daily quota usage using weighted moving average (US-156-009).

        Uses historical data with exponential weighting - more recent days
        have higher weight. Also factors in time of day for current predictions.

        Returns:
            Dict with predicted usage, confidence, and factors
        """
        with self._lock:
            # Load history if not already loaded
            if not self._history:
                self.load_history()

            # Calculate weighted moving average
            if len(self._history) == 0:
                self._last_prediction = {
                    "predicted_daily_usage": self.estimate_quota(),
                    "time_factor": 1.0,
                    "based_on_days": 0,
                    "message": "No historical data - using estimate",
                }
                self._prediction_confidence = "low"
                return self._last_prediction

            # Sort by date descending
            sorted_history = sorted(
                self._history,
                key=lambda x: x.get('date', ''),
                reverse=True
            )

            # Take last N days for weighted average
            window = min(len(sorted_history), self._weighted_avg_window)
            recent_history = sorted_history[:window]

            # Calculate weighted average (exponential decay - more recent = higher weight)
            total_weight = 0.0
            weighted_sum = 0.0

            for i, record in enumerate(recent_history):
                # Weight: most recent = window, oldest = 1
                weight = window - i
                usage = record.get('total_usage', 0)
                weighted_sum += usage * weight
                total_weight += weight

            predicted_usage = int(weighted_sum / total_weight) if total_weight > 0 else 0

            # Apply time-of-day factor
            current_hour = datetime.now().hour
            time_multiplier = self._get_time_multiplier(current_hour)
            adjusted_prediction = int(predicted_usage * time_multiplier)

            # Determine confidence based on data quality
            if window >= 7 and len(set(r.get('total_usage', 0) for r in recent_history)) > 1:
                self._prediction_confidence = "high"
            elif window >= 3:
                self._prediction_confidence = "medium"
            else:
                self._prediction_confidence = "low"

            self._last_prediction = {
                "predicted_daily_usage": adjusted_prediction,
                "base_prediction": predicted_usage,
                "time_of_day": current_hour,
                "time_period": self._get_time_period(current_hour),
                "time_factor": time_multiplier,
                "based_on_days": window,
                "confidence": self._prediction_confidence,
            }

            return self._last_prediction

    def get_prediction_confidence(self) -> str:
        """Get prediction confidence level (US-156-009).

        Returns:
            Confidence level: 'low', 'medium', or 'high'
        """
        with self._lock:
            if not self._last_prediction:
                # Run prediction to get confidence
                self.predict_daily_usage()
            return self._prediction_confidence

    def save_usage_record(self, total_usage: int) -> None:
        """Save current usage as a historical record (US-156-009).

        Call this at the end of a pipeline run to record quota usage
        for future predictions.

        Args:
            total_usage: Total quota used in this session
        """
        with self._lock:
            record = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "total_usage": total_usage,
                "search_used": self._search_used,
                "caption_used": self._caption_used,
                "metadata_used": self._metadata_used,
                "hour": datetime.now().hour,
                "time_period": self._get_time_period(datetime.now().hour),
            }

            # Load existing history
            if not self._history:
                self.load_history()

            # Add new record
            self._history.append(record)

            # Keep only last 30 days
            if len(self._history) > 30:
                self._history = sorted(
                    self._history,
                    key=lambda x: x.get('date', ''),
                    reverse=True
                )[:30]

            # Save to file
            try:
                os.makedirs(os.path.dirname(self._history_file), exist_ok=True)
                with open(self._history_file, 'w', encoding='utf-8') as f:
                    json.dump({"history": self._history}, f, indent=2)
                logger.debug(f"Saved quota usage record to {self._history_file}")
            except IOError as e:
                logger.warning(f"Failed to save quota history: {e}")

    def get_prediction_metrics(self) -> Dict[str, any]:
        """Get prediction-related metrics for monitoring (US-156-009).

        Returns:
            Dict with prediction metrics
        """
        with self._lock:
            if not self._last_prediction:
                self.predict_daily_usage()

            return {
                "prediction": self._last_prediction,
                "confidence": self._prediction_confidence,
                "history_count": len(self._history),
                "history_file": self._history_file,
                "weighted_avg_window": self._weighted_avg_window,
            }

    def record_actual_usage(self, actual_usage: int) -> None:
        """Record actual quota usage to calculate prediction accuracy (US-157-002).

        Call this after pipeline completes with the actual quota used.
        It compares against the last prediction to calculate accuracy.

        Args:
            actual_usage: Actual quota units consumed
        """
        with self._lock:
            if not self._last_prediction:
                logger.debug("No prediction to compare against for accuracy tracking")
                return

            predicted_usage = self._last_prediction.get('predicted_daily_usage', 0)

            # Calculate error percentage
            if predicted_usage > 0:
                error_percent = abs(actual_usage - predicted_usage) / predicted_usage * 100
            else:
                error_percent = 0.0

            # Record the prediction/actual pair
            record = {
                "timestamp": datetime.now().isoformat(),
                "predicted": predicted_usage,
                "actual": actual_usage,
                "error_percent": round(error_percent, 2),
                "is_accurate": error_percent <= 20.0,  # Within 20% is considered accurate
            }
            self._prediction_records.append(record)

            # Update aggregate metrics
            self._total_predictions += 1
            if error_percent <= 20.0:
                self._accurate_predictions += 1
            self._prediction_error_sum += error_percent

            # Keep only last 30 records
            if len(self._prediction_records) > 30:
                self._prediction_records = self._prediction_records[-30:]

            logger.info(
                f"Quota prediction accuracy recorded: predicted={predicted_usage}, "
                f"actual={actual_usage}, error={error_percent:.1f}%"
            )

    def get_prediction_accuracy_metrics(self) -> Dict[str, any]:
        """Get metrics about quota prediction accuracy (US-157-002).

        Returns:
            Dict with prediction accuracy metrics including accuracy percentage
            and error statistics
        """
        with self._lock:
            if self._total_predictions == 0:
                return {
                    "total_predictions": 0,
                    "accuracy_percent": 0.0,
                    "average_error_percent": 0.0,
                    "has_data": False,
                    "message": "No prediction accuracy data yet - call record_actual_usage() after pipeline"
                }

            accuracy_percent = (self._accurate_predictions / self._total_predictions) * 100
            avg_error = self._prediction_error_sum / self._total_predictions

            return {
                "total_predictions": self._total_predictions,
                "accurate_predictions": self._accurate_predictions,
                "accuracy_percent": round(accuracy_percent, 2),
                "average_error_percent": round(avg_error, 2),
                "min_error_percent": min(r['error_percent'] for r in self._prediction_records) if self._prediction_records else 0.0,
                "max_error_percent": max(r['error_percent'] for r in self._prediction_records) if self._prediction_records else 0.0,
                "recent_records": self._prediction_records[-5:] if self._prediction_records else [],
                "has_data": True,
            }
