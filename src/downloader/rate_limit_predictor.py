"""
Predictive rate limit detection using historical patterns.

This module provides predictive capabilities for rate limit events based on
historical data from escalation_manager.py and rate_limit_budget.py.

Features:
- Track historical rate limit events by hour-of-day and day-of-week
- Hour-of-day granularity (24 bins) for fine-grained predictions
- Weekend vs weekday separate tracking and aggregated patterns
- Time-window based prediction: morning vs evening vs overnight sessions
- predict_rate_limit_likelihood() method returns probability (0.0-1.0)
- Automatic budget allocation increase when likelihood > 0.6
- Configurable prediction sensitivity
- Sliding window (24 hours) for recent activity analysis
- Keyword-specific prediction weights based on past success rate
- Prediction confidence scoring (high/medium/low)

US-109-008: Add predictive rate limit detection using historical patterns
US-143-004: Weekend-aware rate limit prediction with hour-of-day granularity
US-144-007: Enhance rate limit prediction with sliding window analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TimeWindow(Enum):
    """Time windows for rate limit pattern analysis.

    Different time windows have different rate limit likelihoods:
    - MORNING (6-12): Moderate usage, moderate rate limits
    - AFTERNOON (12-18): Peak usage, higher rate limits
    - EVENING (18-22): Highest usage, highest rate limits
    - OVERNIGHT (22-6): Lowest usage, lowest rate limits
    """
    MORNING = "morning"    # 6:00 - 12:00
    AFTERNOON = "afternoon"  # 12:00 - 18:00
    EVENING = "evening"     # 18:00 - 22:00
    OVERNIGHT = "overnight"  # 22:00 - 6:00

    @classmethod
    def from_hour(cls, hour: int) -> "TimeWindow":
        """Determine time window from hour of day.

        Args:
            hour: Hour of day (0-23)

        Returns:
            TimeWindow corresponding to the hour
        """
        if 6 <= hour < 12:
            return cls.MORNING
        elif 12 <= hour < 18:
            return cls.AFTERNOON
        elif 18 <= hour < 22:
            return cls.EVENING
        else:
            return cls.OVERNIGHT


class DayOfWeek(Enum):
    """Day of week for rate limit pattern analysis."""
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"

    @classmethod
    def from_datetime(cls, dt: datetime) -> "DayOfWeek":
        """Get day of week from datetime.

        Args:
            dt: datetime object

        Returns:
            DayOfWeek enum
        """
        days = [
            cls.MONDAY, cls.TUESDAY, cls.WEDNESDAY,
            cls.THURSDAY, cls.FRIDAY, cls.SATURDAY, cls.SUNDAY
        ]
        return days[dt.weekday()]

    @classmethod
    def is_weekend(cls, day: "DayOfWeek") -> bool:
        """Check if day is weekend.

        Args:
            day: DayOfWeek to check

        Returns:
            True if Saturday or Sunday
        """
        return day in (cls.SATURDAY, cls.SUNDAY)


@dataclass
class RateLimitEvent:
    """A single rate limit event with timestamp and metadata."""

    timestamp: float  # Unix timestamp
    trigger_category: str = "unknown"  # e.g., '429', '403', 'ip_blocked'
    tier: str = "tier1"  # Which tier triggered the rate limit
    keyword: Optional[str] = None  # Keyword that triggered the event
    resolved: bool = True  # Whether the rate limit was resolved

    @property
    def datetime(self) -> datetime:
        """Get datetime from timestamp."""
        return datetime.fromtimestamp(self.timestamp)

    @property
    def time_window(self) -> TimeWindow:
        """Get time window for this event."""
        return TimeWindow.from_hour(self.datetime.hour)

    @property
    def day_of_week(self) -> DayOfWeek:
        """Get day of week for this event."""
        return DayOfWeek.from_datetime(self.datetime)


@dataclass
class HistoricalPattern:
    """Statistical pattern for a specific time window and day combination."""

    time_window: TimeWindow
    day_of_week: DayOfWeek
    total_attempts: int = 0
    rate_limit_count: int = 0

    @property
    def rate_limit_rate(self) -> float:
        """Calculate rate limit rate for this pattern.

        Returns:
            Rate as probability (0.0-1.0), or 0.0 if no data
        """
        if self.total_attempts == 0:
            return 0.0
        return self.rate_limit_count / self.total_attempts

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "time_window": self.time_window.value,
            "day_of_week": self.day_of_week.value,
            "total_attempts": self.total_attempts,
            "rate_limit_count": self.rate_limit_count,
            "rate_limit_rate": round(self.rate_limit_rate, 4),
        }


@dataclass
class HourlyPattern:
    """Statistical pattern for a specific hour of day (0-23).

    US-143-004: Added for hour-of-day granularity.
    """

    hour: int  # 0-23
    is_weekend: bool  # True for Saturday/Sunday, False for weekday
    total_attempts: int = 0
    rate_limit_count: int = 0

    @property
    def rate_limit_rate(self) -> float:
        """Calculate rate limit rate for this hour pattern.

        Returns:
            Rate as probability (0.0-1.0), or 0.0 if no data
        """
        if self.total_attempts == 0:
            return 0.0
        return self.rate_limit_count / self.total_attempts

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "hour": self.hour,
            "is_weekend": self.is_weekend,
            "total_attempts": self.total_attempts,
            "rate_limit_count": self.rate_limit_count,
            "rate_limit_rate": round(self.rate_limit_rate, 4),
        }


@dataclass
class WeekendAggregatedPattern:
    """Aggregated weekend vs weekday patterns.

    US-143-004: Added for weekend-aware rate limit prediction.
    """

    is_weekend: bool  # True for Saturday/Sunday, False for weekday
    time_window: TimeWindow
    total_attempts: int = 0
    rate_limit_count: int = 0

    @property
    def rate_limit_rate(self) -> float:
        """Calculate rate limit rate for this aggregated pattern.

        Returns:
            Rate as probability (0.0-1.0), or 0.0 if no data
        """
        if self.total_attempts == 0:
            return 0.0
        return self.rate_limit_count / self.total_attempts

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "is_weekend": self.is_weekend,
            "time_window": self.time_window.value,
            "total_attempts": self.total_attempts,
            "rate_limit_count": self.rate_limit_count,
            "rate_limit_rate": round(self.rate_limit_rate, 4),
        }


class PredictionConfidence(Enum):
    """Prediction confidence levels based on data quality.

    US-144-007: Added for confidence-based budget allocation.
    """
    HIGH = "high"      # Sufficient recent data (>= 20 events in 24h window)
    MEDIUM = "medium"  # Some recent data (>= 10 events in 24h window)
    LOW = "low"        # Insufficient recent data (< 10 events in 24h window)


@dataclass
class KeywordPattern:
    """Pattern tracking for a specific keyword.

    US-144-007: Added for keyword-specific prediction weights.
    """

    keyword: str
    total_attempts: int = 0
    rate_limit_count: int = 0
    last_event_timestamp: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this keyword.

        Returns:
            Success rate as probability (0.0-1.0), 1.0 if no attempts
        """
        if self.total_attempts == 0:
            return 1.0
        return 1.0 - (self.rate_limit_count / self.total_attempts)

    @property
    def rate_limit_rate(self) -> float:
        """Calculate rate limit rate for this keyword.

        Returns:
            Rate as probability (0.0-1.0), or 0.0 if no attempts
        """
        if self.total_attempts == 0:
            return 0.0
        return self.rate_limit_count / self.total_attempts

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "keyword": self.keyword,
            "total_attempts": self.total_attempts,
            "rate_limit_count": self.rate_limit_count,
            "success_rate": round(self.success_rate, 4),
            "rate_limit_rate": round(self.rate_limit_rate, 4),
        }


@dataclass
class SlidingWindowStats:
    """Sliding window statistics for recent activity.

    US-144-007: Added for 24-hour sliding window analysis.
    """

    window_hours: int = 24
    recent_attempts: int = 0
    recent_rate_limits: int = 0
    window_start_timestamp: Optional[float] = None

    @property
    def rate_limit_rate(self) -> float:
        """Calculate rate limit rate in the sliding window.

        Returns:
            Rate as probability (0.0-1.0), or 0.0 if no attempts
        """
        if self.recent_attempts == 0:
            return 0.0
        return self.recent_rate_limits / self.recent_attempts

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "window_hours": self.window_hours,
            "recent_attempts": self.recent_attempts,
            "recent_rate_limits": self.recent_rate_limits,
            "rate_limit_rate": round(self.rate_limit_rate, 4),
        }


class RateLimitPredictor:
    """Predictive rate limit detection using historical patterns.

    This class analyzes historical rate limit events to predict the likelihood
    of rate limits occurring based on current time (hour-of-day and day-of-week).

    Features (US-144-007):
    - Sliding window analysis (last 24 hours) for recent activity
    - Keyword-specific prediction weights based on past success rate
    - Prediction confidence scoring (high/medium/low)
    - Adaptive budget allocation based on prediction confidence

    Usage:
        predictor = RateLimitPredictor()
        predictor.record_attempt(timestamp=ts, keyword="python tutorial")  # Record attempt
        predictor.record_rate_limit_event(timestamp=ts, trigger_category='429', keyword="python tutorial")  # Record event

        likelihood = predictor.predict_rate_limit_likelihood()  # Get prediction
        confidence = predictor.get_prediction_confidence()  # Get confidence level

        if likelihood > 0.6:
            predictor.increase_budget_allocation(budget)
    """

    # Threshold for automatic budget increase
    BUDGET_INCREASE_THRESHOLD = 0.6
    # Multiplier for budget increase when likelihood is high
    BUDGET_INCREASE_MULTIPLIER = 1.5
    # Minimum events needed for reliable prediction
    MIN_EVENTS_FOR_RELIABLE_PREDICTION = 5

    # US-143-004: Minimum events for hourly pattern reliability
    MIN_HOURLY_EVENTS_FOR_RELIABLE_PREDICTION = 10
    # Default prediction sensitivity (higher = more responsive to patterns)
    DEFAULT_SENSITIVITY = 0.5

    # US-144-007: Sliding window configuration
    SLIDING_WINDOW_HOURS = 24
    # Minimum events for HIGH confidence
    HIGH_CONFIDENCE_MIN_EVENTS = 20
    # Minimum events for MEDIUM confidence
    MEDIUM_CONFIDENCE_MIN_EVENTS = 10
    # Keyword weight influence factor
    KEYWORD_WEIGHT_INFLUENCE = 0.3

    def __init__(self, max_history_days: int = 30, sensitivity: float = DEFAULT_SENSITIVITY):
        """Initialize the predictor.

        Args:
            max_history_days: Maximum days to keep in history (default 30)
            sensitivity: Prediction sensitivity (0.0-1.0), higher = more responsive (default 0.5)
        """
        self._events: List[RateLimitEvent] = []
        self._attempts: List[Tuple[float, TimeWindow, DayOfWeek, int, bool]] = []  # (timestamp, time_window, day_of_week, hour, is_weekend)
        self._patterns: Dict[Tuple[TimeWindow, DayOfWeek], HistoricalPattern] = {}

        # US-143-004: Hourly patterns (24 hours × 2 for weekend/weekday)
        self._hourly_patterns: Dict[Tuple[int, bool], HourlyPattern] = {}

        # US-143-004: Weekend aggregated patterns (4 time windows × 2 for weekend/weekday)
        self._weekend_patterns: Dict[Tuple[bool, TimeWindow], WeekendAggregatedPattern] = {}

        # US-144-007: Sliding window stats for recent activity
        self._sliding_window: SlidingWindowStats = SlidingWindowStats(window_hours=self.SLIDING_WINDOW_HOURS)

        # US-144-007: Keyword-specific patterns
        self._keyword_patterns: Dict[str, KeywordPattern] = {}

        self._max_history_days = max_history_days
        self._sensitivity = max(0.0, min(1.0, sensitivity))  # Clamp to 0-1

        # Initialize all pattern combinations
        for tw in TimeWindow:
            for dow in DayOfWeek:
                self._patterns[(tw, dow)] = HistoricalPattern(time_window=tw, day_of_week=dow)

        # Initialize hourly patterns
        for hour in range(24):
            for is_weekend in [False, True]:
                self._hourly_patterns[(hour, is_weekend)] = HourlyPattern(
                    hour=hour, is_weekend=is_weekend
                )

        # Initialize weekend aggregated patterns
        for is_weekend in [False, True]:
            for tw in TimeWindow:
                self._weekend_patterns[(is_weekend, tw)] = WeekendAggregatedPattern(
                    is_weekend=is_weekend, time_window=tw
                )

        logger.debug(f"RateLimitPredictor initialized with {max_history_days} day history, sensitivity={sensitivity}")

    def record_attempt(self, timestamp: Optional[float] = None,
                       time_window: Optional[TimeWindow] = None,
                       day_of_week: Optional[DayOfWeek] = None,
                       keyword: Optional[str] = None) -> None:
        """Record a download attempt for pattern tracking.

        Args:
            timestamp: Unix timestamp of attempt (default: now)
            time_window: TimeWindow of attempt (derived from timestamp if not provided)
            day_of_week: DayOfWeek of attempt (derived from timestamp if not provided)
            keyword: Keyword associated with this attempt
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        dt = datetime.fromtimestamp(timestamp)

        if time_window is None:
            time_window = TimeWindow.from_hour(dt.hour)
        if day_of_week is None:
            day_of_week = DayOfWeek.from_datetime(dt)

        is_weekend = DayOfWeek.is_weekend(day_of_week)

        self._attempts.append((timestamp, time_window, day_of_week, dt.hour, is_weekend))

        # Update pattern statistics
        pattern = self._patterns[(time_window, day_of_week)]
        pattern.total_attempts += 1

        # US-143-004: Update hourly pattern
        hourly_pattern = self._hourly_patterns[(dt.hour, is_weekend)]
        hourly_pattern.total_attempts += 1

        # US-143-004: Update weekend aggregated pattern
        weekend_pattern = self._weekend_patterns[(is_weekend, time_window)]
        weekend_pattern.total_attempts += 1

        # US-144-007: Update sliding window
        self._update_sliding_window(timestamp)

        # US-144-007: Update keyword pattern
        if keyword:
            self._update_keyword_pattern(keyword, timestamp, is_rate_limit=False)

        # Prune old attempts
        self._prune_old_data()

    def record_rate_limit_event(self, timestamp: Optional[float] = None,
                                trigger_category: str = "unknown",
                                tier: str = "tier1",
                                keyword: Optional[str] = None) -> None:
        """Record a rate limit event for pattern tracking.

        Args:
            timestamp: Unix timestamp of event (default: now)
            trigger_category: Category of trigger (e.g., '429', '403')
            tier: Which tier triggered the rate limit
            keyword: Keyword that triggered the event
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        event = RateLimitEvent(
            timestamp=timestamp,
            trigger_category=trigger_category,
            tier=tier,
            keyword=keyword,
        )

        self._events.append(event)

        # Update pattern statistics
        pattern = self._patterns[(event.time_window, event.day_of_week)]
        pattern.rate_limit_count += 1

        # US-143-004: Update hourly pattern
        dt = event.datetime
        is_weekend = DayOfWeek.is_weekend(event.day_of_week)
        hourly_pattern = self._hourly_patterns[(dt.hour, is_weekend)]
        hourly_pattern.rate_limit_count += 1

        # US-143-004: Update weekend aggregated pattern
        weekend_pattern = self._weekend_patterns[(is_weekend, event.time_window)]
        weekend_pattern.rate_limit_count += 1

        # US-144-007: Update sliding window for rate limits
        self._update_sliding_window(timestamp, is_rate_limit=True)

        # US-144-007: Update keyword pattern for rate limit
        if keyword:
            self._update_keyword_pattern(keyword, timestamp, is_rate_limit=True)

        # Prune old data
        self._prune_old_data()

        logger.debug(
            f"Rate limit event recorded: {trigger_category} at {event.time_window.value} "
            f"on {event.day_of_week.value}"
        )

    def _prune_old_data(self) -> None:
        """Remove data older than max_history_days."""
        cutoff = (datetime.now() - timedelta(days=self._max_history_days)).timestamp()

        # Prune events
        self._events = [e for e in self._events if e.timestamp > cutoff]

        # Prune attempts (now includes hour and is_weekend)
        self._attempts = [(t, tw, dow, h, w) for t, tw, dow, h, w in self._attempts if t > cutoff]

        # US-144-007: Update sliding window after pruning
        self._recalculate_sliding_window()

    def _update_sliding_window(self, timestamp: float, is_rate_limit: bool = False) -> None:
        """Update sliding window stats with new event.

        US-144-007: Added for 24-hour sliding window analysis.

        Args:
            timestamp: Unix timestamp of the event
            is_rate_limit: Whether this is a rate limit event
        """
        # Initialize window start if not set
        if self._sliding_window.window_start_timestamp is None:
            self._sliding_window.window_start_timestamp = timestamp

        # Check if timestamp is within the sliding window
        window_cutoff = datetime.now() - timedelta(hours=self.SLIDING_WINDOW_HOURS)
        window_start = self._sliding_window.window_start_timestamp

        # Reset window if too much time has passed
        if timestamp < window_start or timestamp < window_cutoff.timestamp():
            self._sliding_window = SlidingWindowStats(
                window_hours=self.SLIDING_WINDOW_HOURS,
                window_start_timestamp=timestamp
            )

        # Update counts
        self._sliding_window.recent_attempts += 1
        if is_rate_limit:
            self._sliding_window.recent_rate_limits += 1

    def _recalculate_sliding_window(self) -> None:
        """Recalculate sliding window stats from scratch.

        US-144-007: Added for accurate window recalculation after pruning.
        """
        window_cutoff = (datetime.now() - timedelta(hours=self.SLIDING_WINDOW_HOURS)).timestamp()

        recent_attempts = 0
        recent_rate_limits = 0

        # Count recent attempts
        for t, _, _, _, _ in self._attempts:
            if t > window_cutoff:
                recent_attempts += 1

        # Count recent rate limit events
        for event in self._events:
            if event.timestamp > window_cutoff:
                recent_rate_limits += 1

        # Update sliding window
        self._sliding_window.recent_attempts = recent_attempts
        self._sliding_window.recent_rate_limits = recent_rate_limits

    def _update_keyword_pattern(self, keyword: str, timestamp: float, is_rate_limit: bool) -> None:
        """Update keyword-specific pattern.

        US-144-007: Added for keyword-specific prediction weights.

        Args:
            keyword: The keyword to update
            timestamp: Unix timestamp of the event
            is_rate_limit: Whether this is a rate limit event
        """
        if keyword not in self._keyword_patterns:
            self._keyword_patterns[keyword] = KeywordPattern(keyword=keyword)

        pattern = self._keyword_patterns[keyword]
        pattern.total_attempts += 1
        if is_rate_limit:
            pattern.rate_limit_count += 1
        pattern.last_event_timestamp = timestamp

    def get_keyword_success_rate(self, keyword: str) -> float:
        """Get success rate for a specific keyword.

        US-144-007: Added for keyword-specific prediction weights.

        Args:
            keyword: The keyword to look up

        Returns:
            Success rate (0.0-1.0), 1.0 if no data
        """
        if keyword not in self._keyword_patterns:
            return 1.0  # Default to full success if no history
        return self._keyword_patterns[keyword].success_rate

    def get_keyword_prediction_weight(self, keyword: str) -> float:
        """Get prediction weight for a keyword based on its past success rate.

        US-144-007: Added for keyword-specific prediction weights.

        Args:
            keyword: The keyword to look up

        Returns:
            Weight factor (0.5-1.5), lower for historically problematic keywords
        """
        success_rate = self.get_keyword_success_rate(keyword)
        # Lower success rate = higher weight (more cautious)
        # Success rate 1.0 -> weight 0.5 (very safe)
        # Success rate 0.0 -> weight 1.5 (very risky)
        return 1.5 - (success_rate * 1.0)

    def get_prediction_confidence(self, timestamp: Optional[float] = None) -> PredictionConfidence:
        """Get prediction confidence level based on recent data.

        US-144-007: Added for confidence-based budget allocation.

        Args:
            timestamp: Unix timestamp to check (default: now)

        Returns:
            PredictionConfidence level (HIGH, MEDIUM, or LOW)
        """
        # Recalculate to ensure accuracy
        self._recalculate_sliding_window()

        recent_events = self._sliding_window.recent_attempts

        if recent_events >= self.HIGH_CONFIDENCE_MIN_EVENTS:
            return PredictionConfidence.HIGH
        elif recent_events >= self.MEDIUM_CONFIDENCE_MIN_EVENTS:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW

    def get_sliding_window_stats(self) -> Dict:
        """Get sliding window statistics.

        US-144-007: Added for 24-hour sliding window analysis.

        Returns:
            Dict with sliding window stats
        """
        return self._sliding_window.to_dict()

    def get_keyword_stats(self) -> List[Dict]:
        """Get statistics for all keyword patterns.

        US-144-007: Added for keyword-specific prediction.

        Returns:
            List of keyword pattern statistics
        """
        return [pattern.to_dict() for pattern in self._keyword_patterns.values()]

    def predict_rate_limit_likelihood(self, timestamp: Optional[float] = None,
                                      keyword: Optional[str] = None) -> float:
        """Predict the likelihood of rate limit occurring.

        This method analyzes historical patterns using hour-of-day granularity
        and day-of-week to predict rate limit likelihood. Uses sensitivity config
        to weight between hourly patterns and time window patterns.

        US-144-007: Enhanced with keyword-specific weights and sliding window.

        Args:
            timestamp: Unix timestamp to predict for (default: now)
            keyword: Optional keyword for keyword-specific weighting

        Returns:
            Probability of rate limit (0.0-1.0)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        dt = datetime.fromtimestamp(timestamp)
        time_window = TimeWindow.from_hour(dt.hour)
        day_of_week = DayOfWeek.from_datetime(dt)
        is_weekend = DayOfWeek.is_weekend(day_of_week)

        # US-143-004: Use hourly pattern with sensitivity
        base_likelihood = self._predict_with_hour_granularity(dt.hour, is_weekend, time_window, day_of_week)

        # US-144-007: Apply keyword-specific weight if provided
        if keyword:
            keyword_weight = self.get_keyword_prediction_weight(keyword)
            # Adjust likelihood based on keyword history
            adjusted_likelihood = base_likelihood * keyword_weight
            return min(1.0, max(0.0, adjusted_likelihood))

        # US-144-007: Also factor in sliding window recent activity
        self._recalculate_sliding_window()
        if self._sliding_window.recent_attempts > 0:
            # Blend with sliding window rate
            window_rate = self._sliding_window.rate_limit_rate
            # Use confidence to determine blend weight
            confidence = self.get_prediction_confidence(timestamp)
            if confidence == PredictionConfidence.HIGH:
                window_weight = 0.4
            elif confidence == PredictionConfidence.MEDIUM:
                window_weight = 0.2
            else:
                window_weight = 0.1

            blended = (base_likelihood * (1 - window_weight)) + (window_rate * window_weight)
            return min(1.0, max(0.0, blended))

        return base_likelihood

    def _predict_with_hour_granularity(self, hour: int, is_weekend: bool,
                                       time_window: TimeWindow,
                                       day_of_week: DayOfWeek) -> float:
        """Predict likelihood using hour-of-day granularity.

        US-143-004: New method for fine-grained predictions.

        Args:
            hour: Hour of day (0-23)
            is_weekend: Whether it's a weekend
            time_window: Time window enum
            day_of_week: Day of week enum

        Returns:
            Probability of rate limit (0.0-1.0)
        """
        # Get hourly pattern
        hourly_pattern = self._hourly_patterns[(hour, is_weekend)]

        # Get time window pattern
        pattern = self._patterns[(time_window, day_of_week)]

        # Get weekend aggregated pattern
        weekend_pattern = self._weekend_patterns[(is_weekend, time_window)]

        # Check if we have enough hourly data
        hourly_data_sufficient = hourly_pattern.total_attempts >= self.MIN_HOURLY_EVENTS_FOR_RELIABLE_PREDICTION

        if hourly_data_sufficient:
            # Use weighted combination based on sensitivity
            # Higher sensitivity = more weight on hourly pattern
            hourly_weight = self._sensitivity
            window_weight = (1 - self._sensitivity) * 0.5
            weekend_weight = (1 - self._sensitivity) * 0.5

            hourly_rate = hourly_pattern.rate_limit_rate
            window_rate = pattern.rate_limit_rate
            weekend_rate = weekend_pattern.rate_limit_rate

            # Blend the rates
            combined_rate = (
                hourly_rate * hourly_weight +
                window_rate * window_weight +
                weekend_rate * weekend_weight
            )

            return min(1.0, max(0.0, combined_rate))

        # If not enough hourly data, fall back to time window with weekend factor
        if pattern.total_attempts >= self.MIN_EVENTS_FOR_RELIABLE_PREDICTION:
            base_likelihood = pattern.rate_limit_rate
            # Apply weekend boost if weekend
            if is_weekend:
                base_likelihood *= 1.3
            return min(1.0, max(0.0, base_likelihood))

        # Not enough data - use time factors estimation
        return self._estimate_from_time_factors(time_window, day_of_week)

    def _estimate_from_time_factors(self, time_window: TimeWindow,
                                     day_of_week: DayOfWeek) -> float:
        """Estimate likelihood when insufficient historical data.

        Uses known YouTube rate limit patterns:
        - Evening hours have highest rate limits
        - Weekend days have more usage
        - Morning hours are moderate

        Args:
            time_window: Current time window
            day_of_week: Current day of week

        Returns:
            Estimated probability (0.0-1.0)
        """
        # Base rates by time window (empirical YouTube patterns)
        time_window_rates = {
            TimeWindow.MORNING: 0.15,
            TimeWindow.AFTERNOON: 0.25,
            TimeWindow.EVENING: 0.40,
            TimeWindow.OVERNIGHT: 0.10,
        }

        # Day of week multipliers
        if DayOfWeek.is_weekend(day_of_week):
            day_multiplier = 1.3  # Higher on weekends
        else:
            day_multiplier = 1.0

        base_rate = time_window_rates.get(time_window, 0.2)
        estimated = base_rate * day_multiplier

        return min(1.0, max(0.0, estimated))

    def get_time_window_prediction(self, timestamp: Optional[float] = None) -> Dict:
        """Get detailed prediction for current time window.

        US-143-004: Enhanced with hour-of-day granularity info.

        Args:
            timestamp: Unix timestamp (default: now)

        Returns:
            Dict with prediction details
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        dt = datetime.fromtimestamp(timestamp)
        time_window = TimeWindow.from_hour(dt.hour)
        day_of_week = DayOfWeek.from_datetime(dt)
        is_weekend = DayOfWeek.is_weekend(day_of_week)

        pattern = self._patterns[(time_window, day_of_week)]
        hourly_pattern = self._hourly_patterns[(dt.hour, is_weekend)]
        weekend_pattern = self._weekend_patterns[(is_weekend, time_window)]

        likelihood = self.predict_rate_limit_likelihood(timestamp)

        # US-143-004: Check if hourly data is sufficient
        hourly_data_sufficient = hourly_pattern.total_attempts >= self.MIN_HOURLY_EVENTS_FOR_RELIABLE_PREDICTION

        return {
            "time_window": time_window.value,
            "day_of_week": day_of_week.value,
            "hour": dt.hour,
            "is_weekend": is_weekend,
            "likelihood": round(likelihood, 4),
            "total_attempts": pattern.total_attempts,
            "rate_limit_count": pattern.rate_limit_count,
            "historical_rate": round(pattern.rate_limit_rate, 4),
            "hourly_data_sufficient": hourly_data_sufficient,
            "hourly_attempts": hourly_pattern.total_attempts,
            "hourly_rate_limit_count": hourly_pattern.rate_limit_count,
            "hourly_rate": round(hourly_pattern.rate_limit_rate, 4),
            "weekend_attempts": weekend_pattern.total_attempts,
            "weekend_rate_limit_count": weekend_pattern.rate_limit_count,
            "weekend_rate": round(weekend_pattern.rate_limit_rate, 4),
            "data_sufficient": pattern.total_attempts >= self.MIN_EVENTS_FOR_RELIABLE_PREDICTION,
            "should_increase_budget": likelihood > self.BUDGET_INCREASE_THRESHOLD,
            "sensitivity": self._sensitivity,
        }

    def get_hourly_prediction(self, timestamp: Optional[float] = None) -> Dict:
        """Get detailed hourly prediction.

        US-143-004: New method for hour-of-day granularity predictions.

        Args:
            timestamp: Unix timestamp (default: now)

        Returns:
            Dict with hourly prediction details
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        dt = datetime.fromtimestamp(timestamp)
        time_window = TimeWindow.from_hour(dt.hour)
        day_of_week = DayOfWeek.from_datetime(dt)
        is_weekend = DayOfWeek.is_weekend(day_of_week)

        hourly_pattern = self._hourly_patterns[(dt.hour, is_weekend)]
        likelihood = self.predict_rate_limit_likelihood(timestamp)

        return {
            "hour": dt.hour,
            "is_weekend": is_weekend,
            "time_window": time_window.value,
            "day_of_week": day_of_week.value,
            "likelihood": round(likelihood, 4),
            "total_attempts": hourly_pattern.total_attempts,
            "rate_limit_count": hourly_pattern.rate_limit_count,
            "historical_rate": round(hourly_pattern.rate_limit_rate, 4),
            "data_sufficient": hourly_pattern.total_attempts >= self.MIN_HOURLY_EVENTS_FOR_RELIABLE_PREDICTION,
        }

    def get_weekend_stats(self) -> Dict:
        """Get weekend vs weekday aggregated statistics.

        US-143-004: New method for weekend-aware stats.

        Returns:
            Dict with weekend/weekday aggregated stats
        """
        weekday_total = sum(
            p.total_attempts for (is_weekend, _), p in self._weekend_patterns.items() if not is_weekend
        )
        weekday_limits = sum(
            p.rate_limit_count for (is_weekend, _), p in self._weekend_patterns.items() if not is_weekend
        )
        weekend_total = sum(
            p.total_attempts for (is_weekend, _), p in self._weekend_patterns.items() if is_weekend
        )
        weekend_limits = sum(
            p.rate_limit_count for (is_weekend, _), p in self._weekend_patterns.items() if is_weekend
        )

        return {
            "weekday": {
                "total_attempts": weekday_total,
                "rate_limit_count": weekday_limits,
                "rate": round(weekday_limits / weekday_total, 4) if weekday_total > 0 else 0.0,
            },
            "weekend": {
                "total_attempts": weekend_total,
                "rate_limit_count": weekend_limits,
                "rate": round(weekend_limits / weekend_total, 4) if weekend_total > 0 else 0.0,
            },
            "sensitivity": self._sensitivity,
        }

    def increase_budget_allocation(self, budget: "RateLimitBudgetWrapper", timestamp: Optional[float] = None,
                                   keyword: Optional[str] = None) -> None:
        """Increase budget allocation when rate limit likelihood is high.

        This method should be called when predict_rate_limit_likelihood() returns
        a value > 0.6 (BUDGET_INCREASE_THRESHOLD).

        US-144-007: Enhanced with confidence-based adaptive allocation.

        Args:
            budget: Budget object to increase (must have max_* attributes)
            timestamp: Unix timestamp to predict for (default: now)
            keyword: Optional keyword for keyword-specific weighting
        """
        likelihood = self.predict_rate_limit_likelihood(timestamp, keyword)
        confidence = self.get_prediction_confidence(timestamp)

        if likelihood <= self.BUDGET_INCREASE_THRESHOLD:
            logger.debug(
                f"Budget increase skipped: likelihood {likelihood:.2f} <= "
                f"threshold {self.BUDGET_INCREASE_THRESHOLD}"
            )
            return

        # US-144-007: Apply multiplier based on likelihood AND confidence
        # Higher confidence = more aggressive budget increase
        base_multiplier = self.BUDGET_INCREASE_MULTIPLIER

        if confidence == PredictionConfidence.HIGH:
            # Most aggressive: fully trust the prediction
            increase_factor = base_multiplier * 1.2
        elif confidence == PredictionConfidence.MEDIUM:
            # Moderate: trust the prediction somewhat
            increase_factor = base_multiplier
        else:
            # Low confidence: be more conservative
            increase_factor = base_multiplier * 0.8

        # Also factor in keyword-specific risk
        if keyword:
            keyword_weight = self.get_keyword_prediction_weight(keyword)
            increase_factor *= keyword_weight

        # Increase rotations
        if hasattr(budget, 'max_rotations') and budget.max_rotations > 0:
            old_max = budget.max_rotations
            budget.max_rotations = int(budget.max_rotations * increase_factor)
            logger.info(
                f"Budget increased: max_rotations {old_max} → {budget.max_rotations} "
                f"(likelihood: {likelihood:.2f}, confidence: {confidence.value})"
            )

        # Increase VPN switches
        if hasattr(budget, 'max_vpn_switches') and budget.max_vpn_switches > 0:
            old_max = budget.max_vpn_switches
            budget.max_vpn_switches = int(budget.max_vpn_switches * increase_factor)
            logger.info(
                f"Budget increased: max_vpn_switches {old_max} → {budget.max_vpn_switches} "
                f"(likelihood: {likelihood:.2f}, confidence: {confidence.value})"
            )

        # Increase backoff time
        if hasattr(budget, 'max_backoff_time') and budget.max_backoff_time > 0:
            old_max = budget.max_backoff_time
            budget.max_backoff_time = budget.max_backoff_time * increase_factor
            logger.info(
                f"Budget increased: max_backoff_time {old_max:.0f}s → {budget.max_backoff_time:.0f}s "
                f"(likelihood: {likelihood:.2f}, confidence: {confidence.value})"
            )

    def get_pattern_stats(self) -> List[Dict]:
        """Get statistics for all time window patterns.

        Returns:
            List of pattern statistics
        """
        return [pattern.to_dict() for pattern in self._patterns.values()]

    def get_hourly_pattern_stats(self) -> List[Dict]:
        """Get statistics for all hourly patterns.

        US-143-004: New method for hourly pattern stats.

        Returns:
            List of hourly pattern statistics
        """
        return [pattern.to_dict() for pattern in self._hourly_patterns.values()]

    def get_weekend_pattern_stats(self) -> List[Dict]:
        """Get statistics for weekend aggregated patterns.

        US-143-004: New method for weekend pattern stats.

        Returns:
            List of weekend pattern statistics
        """
        return [pattern.to_dict() for pattern in self._weekend_patterns.values()]

    def get_upcoming_window_likelihood(self, hours_ahead: int = 2) -> Dict:
        """Predict likelihood for upcoming time window.

        Useful for proactive budget allocation before starting a download session.

        Args:
            hours_ahead: Hours ahead to predict (default: 2)

        Returns:
            Dict with prediction for upcoming window
        """
        future_time = datetime.now() + timedelta(hours=hours_ahead)
        likelihood = self.predict_rate_limit_likelihood(future_time.timestamp())

        # US-143-004: Also get is_weekend for future time
        future_dow = DayOfWeek.from_datetime(future_time)
        is_weekend = DayOfWeek.is_weekend(future_dow)

        return {
            "hours_ahead": hours_ahead,
            "future_time_window": TimeWindow.from_hour(future_time.hour).value,
            "future_day_of_week": future_dow.value,
            "future_is_weekend": is_weekend,
            "future_hour": future_time.hour,
            "predicted_likelihood": round(likelihood, 4),
            "recommendation": "increase_budget" if likelihood > self.BUDGET_INCREASE_THRESHOLD else "normal",
        }

    def to_dict(self) -> Dict:
        """Serialize predictor state for checkpoint.

        US-144-007: Includes sliding window and keyword patterns.

        Returns:
            Dict with predictor state
        """
        # Recalculate sliding window before serializing
        self._recalculate_sliding_window()

        return {
            "patterns": self.get_pattern_stats(),
            "hourly_patterns": self.get_hourly_pattern_stats(),
            "weekend_patterns": self.get_weekend_pattern_stats(),
            "sliding_window": self._sliding_window.to_dict(),
            "keyword_patterns": self.get_keyword_stats(),
            "total_events": len(self._events),
            "total_attempts": len(self._attempts),
            "sensitivity": self._sensitivity,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict], max_history_days: int = 30) -> "RateLimitPredictor":
        """Create predictor from checkpoint data.

        US-144-007: Handles sliding window and keyword patterns.

        Args:
            data: Dict with predictor state
            max_history_days: Max history days for new instance

        Returns:
            RateLimitPredictor instance
        """
        sensitivity = data.get("sensitivity", cls.DEFAULT_SENSITIVITY) if data else cls.DEFAULT_SENSITIVITY
        predictor = cls(max_history_days=max_history_days, sensitivity=sensitivity)

        if not data or "patterns" not in data:
            return predictor

        # Restore patterns
        for pattern_data in data.get("patterns", []):
            tw = TimeWindow(pattern_data["time_window"])
            dow = DayOfWeek(pattern_data["day_of_week"])
            predictor._patterns[(tw, dow)] = HistoricalPattern(
                time_window=tw,
                day_of_week=dow,
                total_attempts=pattern_data["total_attempts"],
                rate_limit_count=pattern_data["rate_limit_count"],
            )

        # US-143-004: Restore hourly patterns
        hourly_data = data.get("hourly_patterns", [])
        for pattern_data in hourly_data:
            hour = pattern_data["hour"]
            is_weekend = pattern_data["is_weekend"]
            predictor._hourly_patterns[(hour, is_weekend)] = HourlyPattern(
                hour=hour,
                is_weekend=is_weekend,
                total_attempts=pattern_data["total_attempts"],
                rate_limit_count=pattern_data["rate_limit_count"],
            )

        # US-143-004: Restore weekend patterns
        weekend_data = data.get("weekend_patterns", [])
        for pattern_data in weekend_data:
            is_weekend = pattern_data["is_weekend"]
            tw = TimeWindow(pattern_data["time_window"])
            predictor._weekend_patterns[(is_weekend, tw)] = WeekendAggregatedPattern(
                is_weekend=is_weekend,
                time_window=tw,
                total_attempts=pattern_data["total_attempts"],
                rate_limit_count=pattern_data["rate_limit_count"],
            )

        # US-144-007: Restore sliding window
        sliding_data = data.get("sliding_window", {})
        if sliding_data:
            predictor._sliding_window = SlidingWindowStats(
                window_hours=sliding_data.get("window_hours", cls.SLIDING_WINDOW_HOURS),
                recent_attempts=sliding_data.get("recent_attempts", 0),
                recent_rate_limits=sliding_data.get("recent_rate_limits", 0),
            )

        # US-144-007: Restore keyword patterns
        keyword_data = data.get("keyword_patterns", [])
        for pattern_data in keyword_data:
            keyword = pattern_data["keyword"]
            predictor._keyword_patterns[keyword] = KeywordPattern(
                keyword=keyword,
                total_attempts=pattern_data.get("total_attempts", 0),
                rate_limit_count=pattern_data.get("rate_limit_count", 0),
            )

        return predictor

    def clear(self) -> None:
        """Clear all historical data."""
        self._events.clear()
        self._attempts.clear()

        # Reset patterns
        for tw in TimeWindow:
            for dow in DayOfWeek:
                self._patterns[(tw, dow)] = HistoricalPattern(time_window=tw, day_of_week=dow)

        # US-143-004: Reset hourly patterns
        for hour in range(24):
            for is_weekend in [False, True]:
                self._hourly_patterns[(hour, is_weekend)] = HourlyPattern(
                    hour=hour, is_weekend=is_weekend
                )

        # US-143-004: Reset weekend patterns
        for is_weekend in [False, True]:
            for tw in TimeWindow:
                self._weekend_patterns[(is_weekend, tw)] = WeekendAggregatedPattern(
                    is_weekend=is_weekend, time_window=tw
                )

        # US-144-007: Reset sliding window
        self._sliding_window = SlidingWindowStats(window_hours=self.SLIDING_WINDOW_HOURS)

        # US-144-007: Reset keyword patterns
        self._keyword_patterns.clear()

        logger.debug("RateLimitPredictor cleared")

    def get_prediction_with_confidence(self, timestamp: Optional[float] = None,
                                       keyword: Optional[str] = None) -> Dict:
        """Get prediction with full confidence details.

        US-144-007: Added for comprehensive prediction with confidence scoring.

        Args:
            timestamp: Unix timestamp to predict for (default: now)
            keyword: Optional keyword for keyword-specific weighting

        Returns:
            Dict with prediction and confidence details
        """
        likelihood = self.predict_rate_limit_likelihood(timestamp, keyword)
        confidence = self.get_prediction_confidence(timestamp)
        sliding_stats = self.get_sliding_window_stats()

        keyword_weight = None
        keyword_stats = None
        if keyword:
            keyword_weight = self.get_keyword_prediction_weight(keyword)
            if keyword in self._keyword_patterns:
                keyword_stats = self._keyword_patterns[keyword].to_dict()

        return {
            "likelihood": round(likelihood, 4),
            "confidence": confidence.value,
            "confidence_level": confidence.name,
            "sliding_window": sliding_stats,
            "keyword_weight": round(keyword_weight, 4) if keyword_weight else None,
            "keyword_stats": keyword_stats,
            "should_increase_budget": likelihood > self.BUDGET_INCREASE_THRESHOLD,
            "budget_recommendation": self._get_budget_recommendation(confidence, likelihood),
        }

    def _get_budget_recommendation(self, confidence: PredictionConfidence, likelihood: float) -> str:
        """Get budget recommendation based on confidence and likelihood.

        US-144-007: Added for adaptive budget allocation guidance.

        Args:
            confidence: The prediction confidence level
            likelihood: The predicted likelihood

        Returns:
            Recommendation string
        """
        if likelihood <= self.BUDGET_INCREASE_THRESHOLD:
            return "normal"

        if confidence == PredictionConfidence.HIGH:
            if likelihood > 0.8:
                return "aggressive_increase"
            return "moderate_increase"
        elif confidence == PredictionConfidence.MEDIUM:
            return "cautious_increase"
        else:
            return "minimal_increase"


class RateLimitBudgetWrapper:
    """Wrapper for budget objects to allow budget increase.

    This is a simple wrapper that provides attribute access for
    the increase_budget_allocation method.
    """

    def __init__(self, max_rotations: int = 10, max_vpn_switches: int = 3,
                 max_backoff_time: float = 300.0):
        """Initialize budget wrapper.

        Args:
            max_rotations: Maximum cookie rotations
            max_vpn_switches: Maximum VPN switches
            max_backoff_time: Maximum backoff time in seconds
        """
        self.max_rotations = max_rotations
        self.max_vpn_switches = max_vpn_switches
        self.max_backoff_time = max_backoff_time
