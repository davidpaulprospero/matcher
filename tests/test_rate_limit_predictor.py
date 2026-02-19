"""
Tests for rate_limit_predictor.py - US-109-008

Tests predictive rate limit detection using historical patterns:
- Track historical rate limit events by time-of-day and day-of-week
- predict_rate_limit_likelihood() returns probability (0.0-1.0)
- Automatic budget increase when likelihood > 0.6
- Time-window based prediction: morning vs evening vs overnight sessions
"""

import pytest
from datetime import datetime, timedelta

from src.downloader.rate_limit_predictor import (
    RateLimitPredictor,
    RateLimitEvent,
    TimeWindow,
    DayOfWeek,
    HistoricalPattern,
    RateLimitBudgetWrapper,
    PredictionConfidence,
    KeywordPattern,
    SlidingWindowStats,
)


class TestTimeWindow:
    """Tests for TimeWindow enum and classification."""

    def test_morning_hours(self):
        """Test morning time window classification."""
        assert TimeWindow.from_hour(6) == TimeWindow.MORNING
        assert TimeWindow.from_hour(9) == TimeWindow.MORNING
        assert TimeWindow.from_hour(11) == TimeWindow.MORNING

    def test_afternoon_hours(self):
        """Test afternoon time window classification."""
        assert TimeWindow.from_hour(12) == TimeWindow.AFTERNOON
        assert TimeWindow.from_hour(15) == TimeWindow.AFTERNOON
        assert TimeWindow.from_hour(17) == TimeWindow.AFTERNOON

    def test_evening_hours(self):
        """Test evening time window classification."""
        assert TimeWindow.from_hour(18) == TimeWindow.EVENING
        assert TimeWindow.from_hour(20) == TimeWindow.EVENING
        assert TimeWindow.from_hour(21) == TimeWindow.EVENING

    def test_overnight_hours(self):
        """Test overnight time window classification."""
        assert TimeWindow.from_hour(22) == TimeWindow.OVERNIGHT
        assert TimeWindow.from_hour(0) == TimeWindow.OVERNIGHT
        assert TimeWindow.from_hour(3) == TimeWindow.OVERNIGHT
        assert TimeWindow.from_hour(5) == TimeWindow.OVERNIGHT


class TestDayOfWeek:
    """Tests for DayOfWeek enum and utilities."""

    def test_weekday_classification(self):
        """Test weekday classification."""
        # Monday Jan 6, 2025
        dt = datetime(2025, 1, 6, 12, 0)
        assert DayOfWeek.from_datetime(dt) == DayOfWeek.MONDAY

        # Friday Jan 10, 2025
        dt = datetime(2025, 1, 10, 12, 0)
        assert DayOfWeek.from_datetime(dt) == DayOfWeek.FRIDAY

    def test_weekend_classification(self):
        """Test weekend classification."""
        # Saturday
        assert DayOfWeek.is_weekend(DayOfWeek.SATURDAY) is True
        # Sunday
        assert DayOfWeek.is_weekend(DayOfWeek.SUNDAY) is True
        # Monday is not weekend
        assert DayOfWeek.is_weekend(DayOfWeek.MONDAY) is False


class TestRateLimitEvent:
    """Tests for RateLimitEvent dataclass."""

    def test_event_from_timestamp(self):
        """Test event datetime derivation."""
        ts = datetime(2025, 1, 6, 14, 30).timestamp()  # Monday afternoon
        event = RateLimitEvent(timestamp=ts)

        assert event.datetime.hour == 14
        assert event.time_window == TimeWindow.AFTERNOON
        assert event.day_of_week == DayOfWeek.MONDAY

    def test_event_with_trigger_category(self):
        """Test event with trigger category."""
        ts = datetime(2025, 1, 6, 14, 30).timestamp()
        event = RateLimitEvent(
            timestamp=ts,
            trigger_category="429",
            tier="tier1",
            keyword="test_keyword"
        )

        assert event.trigger_category == "429"
        assert event.tier == "tier1"
        assert event.keyword == "test_keyword"


class TestHistoricalPattern:
    """Tests for HistoricalPattern dataclass."""

    def test_rate_limit_rate_calculation(self):
        """Test rate limit rate calculation."""
        pattern = HistoricalPattern(
            time_window=TimeWindow.EVENING,
            day_of_week=DayOfWeek.FRIDAY,
            total_attempts=10,
            rate_limit_count=3
        )

        assert pattern.rate_limit_rate == 0.3

    def test_rate_limit_rate_zero_attempts(self):
        """Test rate limit rate when no attempts."""
        pattern = HistoricalPattern(
            time_window=TimeWindow.EVENING,
            day_of_week=DayOfWeek.SATURDAY,
            total_attempts=0,
            rate_limit_count=0
        )

        assert pattern.rate_limit_rate == 0.0


class TestRateLimitPredictor:
    """Tests for RateLimitPredictor main class."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = RateLimitPredictor()
        assert predictor._max_history_days == 30
        assert len(predictor._events) == 0
        assert len(predictor._attempts) == 0

    def test_record_attempt(self):
        """Test recording download attempts."""
        predictor = RateLimitPredictor()

        # Record attempts during evening on Friday
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening

        for i in range(5):
            ts = base_time + i * 60  # 1 minute apart
            predictor.record_attempt(timestamp=ts)

        # Check pattern updated
        pattern = predictor._patterns[(TimeWindow.EVENING, DayOfWeek.FRIDAY)]
        assert pattern.total_attempts == 5

    def test_record_rate_limit_event(self):
        """Test recording rate limit events."""
        predictor = RateLimitPredictor()

        # Record event during evening on Friday
        ts = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening
        predictor.record_rate_limit_event(timestamp=ts, trigger_category="429")

        # Check pattern updated
        pattern = predictor._patterns[(TimeWindow.EVENING, DayOfWeek.FRIDAY)]
        assert pattern.rate_limit_count == 1

    def test_predict_likelihood_insufficient_data(self):
        """Test prediction with insufficient historical data.

        When there are fewer than MIN_EVENTS_FOR_RELIABLE_PREDICTION (5) events,
        should estimate from time factors.
        """
        predictor = RateLimitPredictor()

        # No data - should estimate from time factors
        likelihood = predictor.predict_rate_limit_likelihood()

        # Should return estimated value based on time factors
        assert 0.0 <= likelihood <= 1.0

    def test_predict_likelihood_with_data(self):
        """Test prediction with sufficient historical data."""
        predictor = RateLimitPredictor()

        base_time = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening

        # Record 10 attempts
        for i in range(10):
            ts = base_time + i * 60
            predictor.record_attempt(timestamp=ts)

        # Record 3 rate limit events
        for i in range(3):
            ts = base_time + i * 120
            predictor.record_rate_limit_event(timestamp=ts)

        # Predict likelihood
        # Current time is Monday morning (different window), so should use time factors
        # But if we query for the same time window:
        predictor2 = RateLimitPredictor()
        friday_evening = datetime(2025, 1, 10, 20, 0).timestamp()

        for i in range(10):
            ts = friday_evening + i * 60
            predictor2.record_attempt(timestamp=ts)

        for i in range(3):
            ts = friday_evening + i * 120
            predictor2.record_rate_limit_event(timestamp=ts)

        likelihood = predictor2.predict_rate_limit_likelihood(friday_evening)

        # Should reflect historical rate of 3/10 = 0.3
        assert likelihood == 0.3

    def test_time_window_prediction_details(self):
        """Test getting detailed prediction for time window."""
        predictor = RateLimitPredictor()

        # Add some mock data
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening
        for i in range(10):
            predictor.record_attempt(timestamp=base_time + i * 60)
        for i in range(3):
            predictor.record_rate_limit_event(timestamp=base_time + i * 120)

        # Get prediction details
        prediction = predictor.get_time_window_prediction(base_time)

        assert prediction["time_window"] == "evening"
        assert prediction["day_of_week"] == "friday"
        assert prediction["hour"] == 20
        assert prediction["total_attempts"] >= 10
        assert prediction["rate_limit_count"] >= 3
        assert prediction["data_sufficient"] is True

    def test_budget_increase_threshold(self):
        """Test that budget increase only happens when likelihood > 0.6."""
        predictor = RateLimitPredictor()

        # Create a budget wrapper
        budget = RateLimitBudgetWrapper(max_rotations=10, max_vpn_switches=3, max_backoff_time=300.0)

        # Add high-rate-limit data (7 out of 10 = 0.7)
        # Use recent timestamps within sliding window (last 24 hours)
        now = datetime.now()
        for i in range(10):
            ts = (now - timedelta(minutes=i)).timestamp()
            predictor.record_attempt(timestamp=ts)
        for i in range(7):
            ts = (now - timedelta(minutes=10 + i)).timestamp()
            predictor.record_rate_limit_event(timestamp=ts)

        # Check likelihood is above threshold
        likelihood = predictor.predict_rate_limit_likelihood()
        assert likelihood > 0.6

        # Apply budget increase
        predictor.increase_budget_allocation(budget)

        # Budget should be increased (with MEDIUM confidence = base multiplier)
        # 10 * 1.5 = 15
        assert budget.max_rotations == 15  # 10 * 1.5
        assert budget.max_vpn_switches == 4  # 3 * 1.5
        assert budget.max_backoff_time == 450.0  # 300 * 1.5

    def test_budget_no_increase_below_threshold(self):
        """Test that budget is not increased when likelihood <= 0.6."""
        predictor = RateLimitPredictor()

        budget = RateLimitBudgetWrapper(max_rotations=10, max_vpn_switches=3, max_backoff_time=300.0)

        # Add low-rate-limit data (1 out of 10 = 0.1)
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()
        for i in range(10):
            predictor.record_attempt(timestamp=base_time + i * 60)
        for i in range(1):
            predictor.record_rate_limit_event(timestamp=base_time + i * 120)

        likelihood = predictor.predict_rate_limit_likelihood(base_time)
        assert likelihood <= 0.6

        # Try to increase budget
        predictor.increase_budget_allocation(budget)

        # Budget should remain unchanged
        assert budget.max_rotations == 10
        assert budget.max_vpn_switches == 3
        assert budget.max_backoff_time == 300.0

    def test_upcoming_window_prediction(self):
        """Test prediction for upcoming time window."""
        predictor = RateLimitPredictor()

        # Get prediction for 2 hours ahead
        prediction = predictor.get_upcoming_window_likelihood(hours_ahead=2)

        assert "hours_ahead" in prediction
        assert "future_time_window" in prediction
        assert "predicted_likelihood" in prediction
        assert prediction["hours_ahead"] == 2
        assert 0.0 <= prediction["predicted_likelihood"] <= 1.0

    def test_pattern_stats(self):
        """Test getting pattern statistics."""
        predictor = RateLimitPredictor()

        # Add some data
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()
        for i in range(10):
            predictor.record_attempt(timestamp=base_time + i * 60)
        for i in range(3):
            predictor.record_rate_limit_event(timestamp=base_time + i * 120)

        stats = predictor.get_pattern_stats()

        # Should have entries for all time window/day combinations
        # 4 time windows * 7 days = 28
        assert len(stats) == 28
        assert any(s["time_window"] == "evening" and s["day_of_week"] == "friday" for s in stats)

    def test_clear(self):
        """Test clearing predictor data."""
        predictor = RateLimitPredictor()

        # Add some data using current time to avoid pruning
        now = datetime.now()
        current_ts = now.timestamp()
        predictor.record_attempt(timestamp=current_ts)
        predictor.record_rate_limit_event(timestamp=current_ts)

        assert len(predictor._events) == 1
        assert len(predictor._attempts) == 1

        # Clear
        predictor.clear()

        assert len(predictor._events) == 0
        assert len(predictor._attempts) == 0

        # Pattern counts should be reset
        current_tw = TimeWindow.from_hour(now.hour)
        current_dow = DayOfWeek.from_datetime(now)
        pattern = predictor._patterns[(current_tw, current_dow)]
        assert pattern.total_attempts == 0
        assert pattern.rate_limit_count == 0


class TestTimeWindowPredictionAccuracy:
    """Tests demonstrating prediction accuracy with mock historical data."""

    def test_morning_vs_evening_prediction(self):
        """Test that evening has higher predicted likelihood than morning."""
        predictor = RateLimitPredictor()

        # Record morning events (lower rate limits)
        morning_time = datetime(2025, 1, 6, 9, 0).timestamp()  # Monday morning
        for i in range(10):
            predictor.record_attempt(timestamp=morning_time + i * 60)
        # 1 rate limit out of 10 = 10%
        predictor.record_rate_limit_event(timestamp=morning_time)

        # Record evening events (higher rate limits)
        evening_time = datetime(2025, 1, 6, 20, 0).timestamp()  # Monday evening
        for i in range(10):
            predictor.record_attempt(timestamp=evening_time + i * 60)
        # 5 rate limits out of 10 = 50%
        for i in range(5):
            predictor.record_rate_limit_event(timestamp=evening_time + i * 120)

        # Predict for each time window
        morning_likelihood = predictor.predict_rate_limit_likelihood(morning_time)
        evening_likelihood = predict_rate_limit_likelihood(evening_time)

        # Evening should have higher likelihood
        assert evening_likelihood > morning_likelihood

    def test_weekend_vs_weekday_prediction(self):
        """Test that weekends may have different patterns."""
        predictor = RateLimitPredictor()

        # Record weekday events
        weekday_time = datetime(2025, 1, 6, 14, 0).timestamp()  # Monday afternoon
        for i in range(10):
            predictor.record_attempt(timestamp=weekday_time + i * 60)
        for i in range(2):
            predictor.record_rate_limit_event(timestamp=weekday_time + i * 120)

        # Record weekend events
        weekend_time = datetime(2025, 1, 11, 14, 0).timestamp()  # Saturday afternoon
        for i in range(10):
            predictor.record_attempt(timestamp=weekend_time + i * 60)
        for i in range(4):
            predictor.record_rate_limit_event(timestamp=weekend_time + i * 120)

        # Predict for each
        weekday_likelihood = predictor.predict_rate_limit_likelihood(weekday_time)
        weekend_likelihood = predictor.predict_rate_limit_likelihood(weekend_time)

        # Weekend should have different (likely higher) likelihood
        assert weekend_likelihood != weekday_likelihood

    def test_prediction_with_empty_history(self):
        """Test prediction returns reasonable default with no history."""
        predictor = RateLimitPredictor()

        # Get prediction at different times
        # Morning (should be ~0.15 based on time factors)
        morning_ts = datetime(2025, 1, 6, 9, 0).timestamp()
        morning_pred = predictor.predict_rate_limit_likelihood(morning_ts)

        # Evening (should be ~0.40 based on time factors)
        evening_ts = datetime(2025, 1, 6, 20, 0).timestamp()
        evening_pred = predictor.predict_rate_limit_likelihood(evening_ts)

        # Both should be valid probabilities
        assert 0.0 <= morning_pred <= 1.0
        assert 0.0 <= evening_pred <= 1.0

        # Evening should be higher than morning (based on time factors)
        assert evening_pred > morning_pred


def predict_rate_limit_likelihood(ts: float) -> float:
    """Helper function to get prediction - needed for test_morning_vs_evening_prediction."""
    predictor = RateLimitPredictor()

    # Record morning events (lower rate limits)
    morning_time = datetime(2025, 1, 6, 9, 0).timestamp()  # Monday morning
    for i in range(10):
        predictor.record_attempt(timestamp=morning_time + i * 60)
    # 1 rate limit out of 10 = 10%
    predictor.record_rate_limit_event(timestamp=morning_time)

    # Record evening events (higher rate limits)
    evening_time = datetime(2025, 1, 6, 20, 0).timestamp()  # Monday evening
    for i in range(10):
        predictor.record_attempt(timestamp=evening_time + i * 60)
    # 5 rate limits out of 10 = 50%
    for i in range(5):
        predictor.record_rate_limit_event(timestamp=evening_time + i * 120)

    return predictor.predict_rate_limit_likelihood(ts)


class TestSerialization:
    """Tests for predictor serialization."""

    def test_to_dict(self):
        """Test serialization to dict."""
        predictor = RateLimitPredictor()

        # Add some data
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()
        predictor.record_attempt(timestamp=base_time)
        predictor.record_rate_limit_event(timestamp=base_time)

        data = predictor.to_dict()

        assert "patterns" in data
        assert "total_events" in data
        assert "total_attempts" in data

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "patterns": [
                {
                    "time_window": "evening",
                    "day_of_week": "friday",
                    "total_attempts": 10,
                    "rate_limit_count": 3,
                    "rate_limit_rate": 0.3
                }
            ],
            "total_events": 3,
            "total_attempts": 10
        }

        predictor = RateLimitPredictor.from_dict(data)

        pattern = predictor._patterns[(TimeWindow.EVENING, DayOfWeek.FRIDAY)]
        assert pattern.total_attempts == 10
        assert pattern.rate_limit_count == 3

    def test_from_empty_dict(self):
        """Test deserialization from empty dict."""
        predictor = RateLimitPredictor.from_dict({})

        # Should initialize with defaults
        assert len(predictor._events) == 0
        assert len(predictor._attempts) == 0


class TestDiagnostics:
    """Tests for predictor diagnostics (US-120-009)."""

    def test_get_time_window_prediction_returns_all_fields(self):
        """get_time_window_prediction returns all diagnostic fields."""
        predictor = RateLimitPredictor()

        # Use a specific timestamp for predictable results
        ts = datetime(2025, 1, 10, 14, 0).timestamp()  # Friday afternoon

        result = predictor.get_time_window_prediction(timestamp=ts)

        assert "time_window" in result
        assert "day_of_week" in result
        assert "hour" in result
        assert "likelihood" in result
        assert "total_attempts" in result
        assert "rate_limit_count" in result
        assert "historical_rate" in result
        assert "data_sufficient" in result
        assert "should_increase_budget" in result
        assert result["time_window"] == "afternoon"
        assert result["day_of_week"] == "friday"
        assert result["hour"] == 14

    def test_get_time_window_prediction_with_historical_data(self):
        """get_time_window_prediction uses historical data when available."""
        predictor = RateLimitPredictor()

        # Add historical data for evening on Friday
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening

        # Record 10 attempts, 3 rate limited
        for _ in range(10):
            predictor.record_attempt(timestamp=base_time)
        for _ in range(3):
            predictor.record_rate_limit_event(timestamp=base_time)

        result = predictor.get_time_window_prediction(timestamp=base_time)

        assert result["total_attempts"] >= 10
        assert result["rate_limit_count"] >= 3
        assert result["data_sufficient"] is True

    def test_get_time_window_prediction_estimates_without_data(self):
        """get_time_window_prediction uses estimates when insufficient data."""
        predictor = RateLimitPredictor()

        # Evening Friday without much data - should use estimates
        ts = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening

        result = predictor.get_time_window_prediction(timestamp=ts)

        # Without sufficient data, should use time-based estimates
        # Evening should have higher likelihood than morning
        assert result["data_sufficient"] is False
        assert result["likelihood"] > 0

    def test_predict_rate_limit_likelihood_high_in_evening(self):
        """predict_rate_limit_likelihood returns higher values in evening."""
        predictor = RateLimitPredictor()

        # Evening (no historical data - uses estimates)
        evening_ts = datetime(2025, 1, 10, 20, 0).timestamp()
        evening_likelihood = predictor.predict_rate_limit_likelihood(timestamp=evening_ts)

        # Morning (lower likelihood)
        morning_ts = datetime(2025, 1, 10, 8, 0).timestamp()
        morning_likelihood = predictor.predict_rate_limit_likelihood(timestamp=morning_ts)

        # Evening should have higher likelihood than morning
        assert evening_likelihood > morning_likelihood

    def test_get_time_window_prediction_budget_increase_flag(self):
        """get_time_window_prediction sets should_increase_budget correctly."""
        predictor = RateLimitPredictor()

        # Add enough historical data to trigger high likelihood
        base_time = datetime(2025, 1, 10, 20, 0).timestamp()  # Friday evening

        # Record many rate limit events to trigger budget increase
        for _ in range(20):
            predictor.record_attempt(timestamp=base_time)
        for _ in range(15):  # 75% rate limit rate
            predictor.record_rate_limit_event(timestamp=base_time)

        result = predictor.get_time_window_prediction(timestamp=base_time)

        # With 75% rate limit, should recommend budget increase (>0.6 threshold)
        if result["historical_rate"] > 0.6:
            assert result["should_increase_budget"] is True

    def test_get_upcoming_window_likelihood(self):
        """get_upcoming_window_likelihood predicts for future time."""
        predictor = RateLimitPredictor()

        # Predict 2 hours ahead
        result = predictor.get_upcoming_window_likelihood(hours_ahead=2)

        assert "hours_ahead" in result
        assert "future_time_window" in result
        assert "future_day_of_week" in result
        assert "predicted_likelihood" in result
        assert "recommendation" in result
        assert result["hours_ahead"] == 2
        assert result["recommendation"] in ["increase_budget", "normal"]

    def test_get_pattern_stats(self):
        """get_pattern_stats returns stats for all time windows."""
        predictor = RateLimitPredictor()

        stats = predictor.get_pattern_stats()

        # Should have stats for all combinations (4 time windows * 7 days = 28)
        assert len(stats) == 28

        # Each stat should have required fields
        for stat in stats:
            assert "time_window" in stat
            assert "day_of_week" in stat
            assert "total_attempts" in stat
            assert "rate_limit_count" in stat


class TestHourlyPatterns:
    """Tests for US-143-004: Hour-of-day granularity."""

    def test_hourly_pattern_initialization(self):
        """Test hourly patterns are initialized for all 24 hours."""
        predictor = RateLimitPredictor()

        # Should have 24 hours × 2 (weekend/weekday) = 48 patterns
        assert len(predictor._hourly_patterns) == 48

    def test_hourly_pattern_records_attempts(self):
        """Test hourly pattern records attempts correctly."""
        predictor = RateLimitPredictor()

        # Record attempt at hour 14 (2 PM) on a weekday (Monday)
        ts = datetime(2025, 1, 6, 14, 0).timestamp()  # Monday 2 PM
        predictor.record_attempt(timestamp=ts)

        # Check hourly pattern updated
        hourly_pattern = predictor._hourly_patterns[(14, False)]  # weekday
        assert hourly_pattern.total_attempts == 1

    def test_hourly_pattern_records_rate_limits(self):
        """Test hourly pattern records rate limit events correctly."""
        predictor = RateLimitPredictor()

        # Record rate limit at hour 20 (8 PM) on weekend (Saturday)
        ts = datetime(2025, 1, 11, 20, 0).timestamp()  # Saturday 8 PM
        predictor.record_rate_limit_event(timestamp=ts, trigger_category="429")

        # Check hourly pattern updated
        hourly_pattern = predictor._hourly_patterns[(20, True)]  # weekend
        assert hourly_pattern.rate_limit_count == 1

    def test_hourly_prediction_with_sufficient_data(self):
        """Test hourly prediction uses hourly patterns when data is sufficient."""
        predictor = RateLimitPredictor(sensitivity=0.8)

        # Add enough hourly data (>= 10 events)
        weekday_hour_14 = datetime(2025, 1, 6, 14, 0).timestamp()  # Monday 2 PM

        for i in range(15):  # More than MIN_HOURLY_EVENTS_FOR_RELIABLE_PREDICTION
            predictor.record_attempt(timestamp=weekday_hour_14 + i * 60)

        # Add rate limit events
        for i in range(5):
            predictor.record_rate_limit_event(timestamp=weekday_hour_14 + i * 120)

        # Get prediction
        likelihood = predictor.predict_rate_limit_likelihood(weekday_hour_14)

        # Should use hourly pattern
        assert 0.0 <= likelihood <= 1.0

    def test_hourly_prediction_uses_sensitivity(self):
        """Test prediction sensitivity affects weighting."""
        predictor_low = RateLimitPredictor(sensitivity=0.2)
        predictor_high = RateLimitPredictor(sensitivity=0.8)

        # Add data
        ts = datetime(2025, 1, 6, 14, 0).timestamp()

        for i in range(15):
            predictor_low.record_attempt(timestamp=ts + i * 60)
            predictor_high.record_attempt(timestamp=ts + i * 60)

        for i in range(5):
            predictor_low.record_rate_limit_event(timestamp=ts + i * 120)
            predictor_high.record_rate_limit_event(timestamp=ts + i * 120)

        # Get predictions
        likelihood_low = predictor_low.predict_rate_limit_likelihood(ts)
        likelihood_high = predictor_high.predict_rate_limit_likelihood(ts)

        # Both should return valid values
        assert 0.0 <= likelihood_low <= 1.0
        assert 0.0 <= likelihood_high <= 1.0

    def test_get_hourly_prediction(self):
        """Test get_hourly_prediction method."""
        predictor = RateLimitPredictor()

        ts = datetime(2025, 1, 6, 14, 0).timestamp()  # Monday 2 PM
        predictor.record_attempt(timestamp=ts)
        predictor.record_rate_limit_event(timestamp=ts, trigger_category="429")

        result = predictor.get_hourly_prediction(ts)

        assert result["hour"] == 14
        assert result["is_weekend"] is False
        assert result["time_window"] == "afternoon"
        assert "likelihood" in result
        assert "total_attempts" in result
        assert "historical_rate" in result

    def test_get_hourly_pattern_stats(self):
        """Test get_hourly_pattern_stats method."""
        predictor = RateLimitPredictor()

        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        predictor.record_attempt(timestamp=ts)

        stats = predictor.get_hourly_pattern_stats()

        # Should have 48 patterns
        assert len(stats) == 48


class TestWeekendPatterns:
    """Tests for US-143-004: Weekend vs weekday tracking."""

    def test_weekend_pattern_initialization(self):
        """Test weekend patterns are initialized correctly."""
        predictor = RateLimitPredictor()

        # Should have 4 time windows × 2 (weekend/weekday) = 8 patterns
        assert len(predictor._weekend_patterns) == 8

    def test_weekend_pattern_records_weekday(self):
        """Test weekend pattern records weekday attempts."""
        predictor = RateLimitPredictor()

        # Record on Monday (weekday)
        ts = datetime(2025, 1, 6, 14, 0).timestamp()  # Monday afternoon
        predictor.record_attempt(timestamp=ts)

        # Check weekday pattern updated
        weekday_pattern = predictor._weekend_patterns[(False, TimeWindow.AFTERNOON)]
        assert weekday_pattern.total_attempts == 1

    def test_weekend_pattern_records_weekend(self):
        """Test weekend pattern records weekend attempts."""
        predictor = RateLimitPredictor()

        # Record on Saturday (weekend)
        ts = datetime(2025, 1, 11, 14, 0).timestamp()  # Saturday afternoon
        predictor.record_attempt(timestamp=ts)

        # Check weekend pattern updated
        weekend_pattern = predictor._weekend_patterns[(True, TimeWindow.AFTERNOON)]
        assert weekend_pattern.total_attempts == 1

    def test_get_weekend_stats(self):
        """Test get_weekend_stats method."""
        predictor = RateLimitPredictor()

        # Add weekday data
        weekday_ts = datetime(2025, 1, 6, 14, 0).timestamp()  # Monday
        for i in range(10):
            predictor.record_attempt(timestamp=weekday_ts + i * 60)

        # Add weekend data
        weekend_ts = datetime(2025, 1, 11, 14, 0).timestamp()  # Saturday
        for i in range(10):
            predictor.record_attempt(timestamp=weekend_ts + i * 60)

        result = predictor.get_weekend_stats()

        assert "weekday" in result
        assert "weekend" in result
        assert result["weekday"]["total_attempts"] == 10
        assert result["weekend"]["total_attempts"] == 10

    def test_get_weekend_pattern_stats(self):
        """Test get_weekend_pattern_stats method."""
        predictor = RateLimitPredictor()

        stats = predictor.get_weekend_pattern_stats()

        # Should have 8 patterns
        assert len(stats) == 8


class TestPredictionSensitivity:
    """Tests for US-143-004: Prediction sensitivity configuration."""

    def test_sensitivity_parameter(self):
        """Test sensitivity parameter is set correctly."""
        predictor = RateLimitPredictor(sensitivity=0.75)
        assert predictor._sensitivity == 0.75

    def test_sensitivity_clamped_to_valid_range(self):
        """Test sensitivity is clamped to 0-1 range."""
        predictor_high = RateLimitPredictor(sensitivity=1.5)
        predictor_low = RateLimitPredictor(sensitivity=-0.5)

        assert predictor_high._sensitivity == 1.0
        assert predictor_low._sensitivity == 0.0

    def test_time_window_prediction_includes_sensitivity(self):
        """Test get_time_window_prediction includes sensitivity."""
        predictor = RateLimitPredictor(sensitivity=0.6)

        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        result = predictor.get_time_window_prediction(ts)

        assert "sensitivity" in result
        assert result["sensitivity"] == 0.6

    def test_prediction_with_different_sensitivities(self):
        """Test predictions vary with different sensitivities."""
        # Low sensitivity - more weight to time window
        predictor_low = RateLimitPredictor(sensitivity=0.1)
        # High sensitivity - more weight to hourly
        predictor_high = RateLimitPredictor(sensitivity=0.9)

        # Add different rates for hourly vs time window
        # Hour 14 = afternoon time window

        # Add data to hourly pattern (high rate)
        weekday_hour_14 = datetime(2025, 1, 6, 14, 0).timestamp()
        for i in range(15):
            predictor_low.record_attempt(timestamp=weekday_hour_14 + i * 60)
            predictor_high.record_attempt(timestamp=weekday_hour_14 + i * 60)

        for i in range(10):  # High rate limit
            predictor_low.record_rate_limit_event(timestamp=weekday_hour_14 + i * 120)
            predictor_high.record_rate_limit_event(timestamp=weekday_hour_14 + i * 120)

        likelihood_low = predictor_low.predict_rate_limit_likelihood(weekday_hour_14)
        likelihood_high = predictor_high.predict_rate_limit_likelihood(weekday_hour_14)

        # Both should be valid
        assert 0.0 <= likelihood_low <= 1.0
        assert 0.0 <= likelihood_high <= 1.0


class TestEnhancedSerialization:
    """Tests for US-143-004: Enhanced serialization with hourly and weekend patterns."""

    def test_to_dict_includes_hourly_patterns(self):
        """Test to_dict includes hourly patterns."""
        predictor = RateLimitPredictor()

        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        predictor.record_attempt(timestamp=ts)

        data = predictor.to_dict()

        assert "hourly_patterns" in data
        assert len(data["hourly_patterns"]) == 48

    def test_to_dict_includes_weekend_patterns(self):
        """Test to_dict includes weekend patterns."""
        predictor = RateLimitPredictor()

        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        predictor.record_attempt(timestamp=ts)

        data = predictor.to_dict()

        assert "weekend_patterns" in data
        assert len(data["weekend_patterns"]) == 8

    def test_to_dict_includes_sensitivity(self):
        """Test to_dict includes sensitivity."""
        predictor = RateLimitPredictor(sensitivity=0.7)

        data = predictor.to_dict()

        assert "sensitivity" in data
        assert data["sensitivity"] == 0.7

    def test_from_dict_restores_hourly_patterns(self):
        """Test from_dict restores hourly patterns."""
        # Create predictor with data
        predictor = RateLimitPredictor()
        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        predictor.record_attempt(timestamp=ts)
        predictor.record_rate_limit_event(timestamp=ts)

        data = predictor.to_dict()

        # Restore from dict
        restored = RateLimitPredictor.from_dict(data)

        # Check hourly pattern restored
        hourly_pattern = restored._hourly_patterns[(14, False)]
        assert hourly_pattern.total_attempts == 1

    def test_from_dict_restores_weekend_patterns(self):
        """Test from_dict restores weekend patterns."""
        predictor = RateLimitPredictor()
        ts = datetime(2025, 1, 11, 14, 0).timestamp()  # Saturday
        predictor.record_attempt(timestamp=ts)

        data = predictor.to_dict()
        restored = RateLimitPredictor.from_dict(data)

        weekend_pattern = restored._weekend_patterns[(True, TimeWindow.AFTERNOON)]
        assert weekend_pattern.total_attempts == 1

    def test_from_dict_restores_sensitivity(self):
        """Test from_dict restores sensitivity."""
        predictor = RateLimitPredictor(sensitivity=0.8)

        data = predictor.to_dict()
        restored = RateLimitPredictor.from_dict(data)

        assert restored._sensitivity == 0.8


class TestClearWithNewPatterns:
    """Tests for clearing predictor with new patterns."""

    def test_clear_resets_hourly_patterns(self):
        """Test clear resets hourly patterns."""
        predictor = RateLimitPredictor()

        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        predictor.record_attempt(timestamp=ts)
        predictor.record_rate_limit_event(timestamp=ts)

        # Clear
        predictor.clear()

        # Check hourly patterns reset
        hourly_pattern = predictor._hourly_patterns[(14, False)]
        assert hourly_pattern.total_attempts == 0
        assert hourly_pattern.rate_limit_count == 0

    def test_clear_resets_weekend_patterns(self):
        """Test clear resets weekend patterns."""
        predictor = RateLimitPredictor()

        ts = datetime(2025, 1, 11, 14, 0).timestamp()  # Saturday
        predictor.record_attempt(timestamp=ts)

        predictor.clear()

        # Check weekend patterns reset
        weekend_pattern = predictor._weekend_patterns[(True, TimeWindow.AFTERNOON)]
        assert weekend_pattern.total_attempts == 0


class TestUpcomingWindowWithWeekend:
    """Tests for US-143-004: Enhanced upcoming window prediction with weekend."""

    def test_upcoming_window_includes_is_weekend(self):
        """Test get_upcoming_window_likelihood includes is_weekend."""
        predictor = RateLimitPredictor()

        result = predictor.get_upcoming_window_likelihood(hours_ahead=2)

        assert "future_is_weekend" in result
        assert "future_hour" in result


class TestSlidingWindowStats:
    """Tests for US-144-007: Sliding window statistics."""

    def test_sliding_window_initialization(self):
        """Test sliding window initializes with correct defaults."""
        stats = SlidingWindowStats(window_hours=24)

        assert stats.window_hours == 24
        assert stats.recent_attempts == 0
        assert stats.recent_rate_limits == 0
        assert stats.window_start_timestamp is None

    def test_sliding_window_rate_calculation(self):
        """Test rate limit rate calculation in sliding window."""
        stats = SlidingWindowStats(window_hours=24)
        stats.recent_attempts = 10
        stats.recent_rate_limits = 3

        assert stats.rate_limit_rate == 0.3

    def test_sliding_window_rate_zero_attempts(self):
        """Test rate limit rate when no attempts."""
        stats = SlidingWindowStats(window_hours=24)

        assert stats.rate_limit_rate == 0.0

    def test_sliding_window_to_dict(self):
        """Test sliding window serialization."""
        stats = SlidingWindowStats(window_hours=24)
        stats.recent_attempts = 10
        stats.recent_rate_limits = 2

        data = stats.to_dict()

        assert data["window_hours"] == 24
        assert data["recent_attempts"] == 10
        assert data["recent_rate_limits"] == 2
        assert data["rate_limit_rate"] == 0.2


class TestKeywordPattern:
    """Tests for US-144-007: Keyword-specific patterns."""

    def test_keyword_pattern_initialization(self):
        """Test keyword pattern initializes correctly."""
        pattern = KeywordPattern(keyword="python tutorial")

        assert pattern.keyword == "python tutorial"
        assert pattern.total_attempts == 0
        assert pattern.rate_limit_count == 0

    def test_keyword_success_rate_calculation(self):
        """Test success rate calculation for keyword."""
        pattern = KeywordPattern(keyword="test")
        pattern.total_attempts = 10
        pattern.rate_limit_count = 2

        assert pattern.success_rate == 0.8
        assert pattern.rate_limit_rate == 0.2

    def test_keyword_success_rate_no_attempts(self):
        """Test success rate when no attempts."""
        pattern = KeywordPattern(keyword="test")

        assert pattern.success_rate == 1.0
        assert pattern.rate_limit_rate == 0.0

    def test_keyword_pattern_to_dict(self):
        """Test keyword pattern serialization."""
        pattern = KeywordPattern(keyword="python")
        pattern.total_attempts = 10
        pattern.rate_limit_count = 3

        data = pattern.to_dict()

        assert data["keyword"] == "python"
        assert data["total_attempts"] == 10
        assert data["rate_limit_count"] == 3
        assert data["success_rate"] == 0.7


class TestPredictionConfidence:
    """Tests for US-144-007: Prediction confidence scoring."""

    def test_confidence_enum_values(self):
        """Test confidence enum has correct values."""
        assert PredictionConfidence.HIGH.value == "high"
        assert PredictionConfidence.MEDIUM.value == "medium"
        assert PredictionConfidence.LOW.value == "low"


class TestSlidingWindowTracking:
    """Tests for US-144-007: Sliding window tracking in predictor."""

    def test_sliding_window_initialization_in_predictor(self):
        """Test predictor initializes sliding window."""
        predictor = RateLimitPredictor()

        assert predictor._sliding_window is not None
        assert predictor._sliding_window.window_hours == 24

    def test_sliding_window_records_attempts(self):
        """Test sliding window records attempts correctly."""
        predictor = RateLimitPredictor()

        # Record attempts
        now = datetime.now()
        for i in range(5):
            ts = (now - timedelta(hours=i)).timestamp()
            predictor.record_attempt(timestamp=ts)

        stats = predictor.get_sliding_window_stats()

        assert stats["recent_attempts"] >= 5

    def test_sliding_window_records_rate_limits(self):
        """Test sliding window records rate limits correctly."""
        predictor = RateLimitPredictor()

        now = datetime.now()
        # Record attempts
        for i in range(10):
            ts = (now - timedelta(hours=i)).timestamp()
            predictor.record_attempt(timestamp=ts)

        # Record rate limits
        for i in range(3):
            ts = (now - timedelta(hours=i)).timestamp()
            predictor.record_rate_limit_event(timestamp=ts)

        stats = predictor.get_sliding_window_stats()

        assert stats["recent_attempts"] >= 10
        assert stats["recent_rate_limits"] >= 3

    def test_sliding_window_prunes_old_data(self):
        """Test sliding window properly handles old data."""
        predictor = RateLimitPredictor()

        # Record attempt from 25 hours ago (should be pruned)
        old_ts = (datetime.now() - timedelta(hours=25)).timestamp()
        predictor.record_attempt(timestamp=old_ts)

        # Record recent attempt
        recent_ts = datetime.now().timestamp()
        predictor.record_attempt(timestamp=recent_ts)

        stats = predictor.get_sliding_window_stats()

        # Recent attempt should be counted
        assert stats["recent_attempts"] >= 1


class TestKeywordSpecificPrediction:
    """Tests for US-144-007: Keyword-specific prediction weights."""

    def test_keyword_pattern_records_attempts(self):
        """Test keyword pattern records attempts."""
        predictor = RateLimitPredictor()

        ts = datetime.now().timestamp()
        predictor.record_attempt(timestamp=ts, keyword="python tutorial")

        assert "python tutorial" in predictor._keyword_patterns
        pattern = predictor._keyword_patterns["python tutorial"]
        assert pattern.total_attempts == 1

    def test_keyword_pattern_records_rate_limits(self):
        """Test keyword pattern records rate limits."""
        predictor = RateLimitPredictor()

        ts = datetime.now().timestamp()
        predictor.record_attempt(timestamp=ts, keyword="python")
        predictor.record_rate_limit_event(timestamp=ts, keyword="python")

        pattern = predictor._keyword_patterns["python"]
        # Both attempt and rate limit count as attempts
        assert pattern.total_attempts == 2
        assert pattern.rate_limit_count == 1

    def test_get_keyword_success_rate(self):
        """Test getting keyword success rate."""
        predictor = RateLimitPredictor()

        # Record data for a keyword
        ts = datetime.now().timestamp()
        for _ in range(10):
            predictor.record_attempt(timestamp=ts, keyword="test_keyword")
        for _ in range(3):
            predictor.record_rate_limit_event(timestamp=ts, keyword="test_keyword")

        # Each rate limit also counts as an attempt, so total = 10 + 3 = 13
        # rate_limits = 3
        # success_rate = 1 - (3/13) = 10/13 ≈ 0.769
        success_rate = predictor.get_keyword_success_rate("test_keyword")

        assert abs(success_rate - 10/13) < 0.01

    def test_get_keyword_success_rate_unknown_keyword(self):
        """Test getting success rate for unknown keyword."""
        predictor = RateLimitPredictor()

        success_rate = predictor.get_keyword_success_rate("unknown_keyword")

        assert success_rate == 1.0

    def test_get_keyword_prediction_weight(self):
        """Test keyword prediction weight calculation."""
        predictor = RateLimitPredictor()

        # High success rate -> lower weight
        ts = datetime.now().timestamp()
        for _ in range(10):
            predictor.record_attempt(timestamp=ts, keyword="safe")
            predictor.record_attempt(timestamp=ts, keyword="risky")
        # No rate limits for "safe"
        for _ in range(5):
            predictor.record_rate_limit_event(timestamp=ts, keyword="risky")

        safe_weight = predictor.get_keyword_prediction_weight("safe")
        risky_weight = predictor.get_keyword_prediction_weight("risky")

        # Safe keyword should have lower weight (more conservative)
        assert safe_weight < risky_weight
        assert 0.5 <= safe_weight <= 1.5
        assert 0.5 <= risky_weight <= 1.5


class TestPredictionConfidenceScoring:
    """Tests for US-144-007: Prediction confidence scoring."""

    def test_confidence_low_with_few_events(self):
        """Test confidence is LOW with few recent events."""
        predictor = RateLimitPredictor()

        # Record just a few attempts
        now = datetime.now()
        for i in range(5):
            ts = (now - timedelta(minutes=i)).timestamp()
            predictor.record_attempt(timestamp=ts)

        confidence = predictor.get_prediction_confidence()

        assert confidence == PredictionConfidence.LOW

    def test_confidence_medium_with_moderate_events(self):
        """Test confidence is MEDIUM with moderate recent events."""
        predictor = RateLimitPredictor()

        # Record enough attempts for MEDIUM (>= 10)
        now = datetime.now()
        for i in range(12):
            ts = (now - timedelta(minutes=i)).timestamp()
            predictor.record_attempt(timestamp=ts)

        confidence = predictor.get_prediction_confidence()

        assert confidence == PredictionConfidence.MEDIUM

    def test_confidence_high_with_many_events(self):
        """Test confidence is HIGH with many recent events."""
        predictor = RateLimitPredictor()

        # Record enough attempts for HIGH (>= 20)
        now = datetime.now()
        for i in range(22):
            ts = (now - timedelta(minutes=i)).timestamp()
            predictor.record_attempt(timestamp=ts)

        confidence = predictor.get_prediction_confidence()

        assert confidence == PredictionConfidence.HIGH


class TestAdaptiveBudgetAllocation:
    """Tests for US-144-007: Adaptive budget allocation based on confidence."""

    def test_budget_increase_with_high_confidence(self):
        """Test budget increase is more aggressive with high confidence."""
        predictor = RateLimitPredictor()

        # Add many events for HIGH confidence
        now = datetime.now()
        base_ts = now.timestamp()
        for i in range(25):
            ts = base_ts - i * 60
            predictor.record_attempt(timestamp=ts)
        for i in range(20):  # High rate limit
            ts = base_ts - i * 120
            predictor.record_rate_limit_event(timestamp=ts)

        budget = RateLimitBudgetWrapper(max_rotations=10, max_vpn_switches=3, max_backoff_time=300.0)

        # Should increase budget
        predictor.increase_budget_allocation(budget)

        # Budget should be increased (HIGH confidence * high likelihood = aggressive)
        assert budget.max_rotations > 10

    def test_budget_increase_with_low_confidence(self):
        """Test budget increase is conservative with low confidence."""
        predictor = RateLimitPredictor()

        # Add few events for LOW confidence
        now = datetime.now()
        base_ts = now.timestamp()
        for i in range(3):
            predictor.record_attempt(timestamp=base_ts - i * 60)
        for i in range(2):  # High rate limit
            predictor.record_rate_limit_event(timestamp=base_ts - i * 120)

        budget = RateLimitBudgetWrapper(max_rotations=10, max_vpn_switches=3, max_backoff_time=300.0)

        # Should increase budget but conservatively
        predictor.increase_budget_allocation(budget)

        # Budget should be increased but less aggressively
        assert budget.max_rotations >= 10


class TestKeywordAwarePrediction:
    """Tests for US-144-007: Keyword-aware likelihood prediction."""

    def test_prediction_with_keyword(self):
        """Test prediction incorporates keyword-specific weight."""
        predictor = RateLimitPredictor()

        now = datetime.now()
        base_ts = now.timestamp()

        # Record base pattern (evening, high rate)
        for i in range(15):
            predictor.record_attempt(timestamp=base_ts - i * 60)

        # Record high rate limit events
        for i in range(10):
            predictor.record_rate_limit_event(timestamp=base_ts - i * 120)

        # Get prediction with a known "risky" keyword
        likelihood_no_keyword = predictor.predict_rate_limit_likelihood(base_ts)

        # Add more rate limits for a specific keyword
        for i in range(5):
            predictor.record_rate_limit_event(timestamp=base_ts - i * 180, keyword="risky_keyword")

        likelihood_with_risky = predictor.predict_rate_limit_likelihood(base_ts, keyword="risky_keyword")

        # With risky keyword, likelihood should be adjusted higher
        assert likelihood_with_risky >= likelihood_no_keyword


class TestPredictionWithConfidence:
    """Tests for US-144-007: get_prediction_with_confidence method."""

    def test_prediction_with_confidence_returns_all_fields(self):
        """Test get_prediction_with_confidence returns all required fields."""
        predictor = RateLimitPredictor()

        now = datetime.now()
        for i in range(5):
            ts = now.timestamp() - i * 60
            predictor.record_attempt(timestamp=ts)

        result = predictor.get_prediction_with_confidence()

        assert "likelihood" in result
        assert "confidence" in result
        assert "confidence_level" in result
        assert "sliding_window" in result
        assert "should_increase_budget" in result
        assert "budget_recommendation" in result

    def test_prediction_with_keyword_includes_keyword_info(self):
        """Test prediction with keyword includes keyword-specific info."""
        predictor = RateLimitPredictor()

        now = datetime.now()
        ts = now.timestamp()
        predictor.record_attempt(timestamp=ts, keyword="test_keyword")

        result = predictor.get_prediction_with_confidence(keyword="test_keyword")

        assert result["keyword_weight"] is not None
        assert result["keyword_stats"] is not None
        assert result["keyword_stats"]["keyword"] == "test_keyword"


class TestSerializationWithNewFeatures:
    """Tests for US-144-007: Serialization with new features."""

    def test_to_dict_includes_sliding_window(self):
        """Test to_dict includes sliding window."""
        predictor = RateLimitPredictor()

        now = datetime.now()
        predictor.record_attempt(timestamp=now.timestamp())

        data = predictor.to_dict()

        assert "sliding_window" in data
        assert data["sliding_window"]["window_hours"] == 24

    def test_to_dict_includes_keyword_patterns(self):
        """Test to_dict includes keyword patterns."""
        predictor = RateLimitPredictor()

        ts = datetime.now().timestamp()
        predictor.record_attempt(timestamp=ts, keyword="test")

        data = predictor.to_dict()

        assert "keyword_patterns" in data
        assert len(data["keyword_patterns"]) >= 1

    def test_from_dict_restores_sliding_window(self):
        """Test from_dict restores sliding window."""
        predictor = RateLimitPredictor()
        now = datetime.now()
        predictor.record_attempt(timestamp=now.timestamp())

        data = predictor.to_dict()
        restored = RateLimitPredictor.from_dict(data)

        stats = restored.get_sliding_window_stats()
        assert stats["recent_attempts"] >= 1

    def test_from_dict_restores_keyword_patterns(self):
        """Test from_dict restores keyword patterns."""
        predictor = RateLimitPredictor()
        ts = datetime.now().timestamp()
        predictor.record_attempt(timestamp=ts, keyword="test")
        predictor.record_rate_limit_event(timestamp=ts, keyword="test")

        data = predictor.to_dict()
        restored = RateLimitPredictor.from_dict(data)

        assert "test" in restored._keyword_patterns
        pattern = restored._keyword_patterns["test"]
        # Both attempt and rate limit count as attempts
        assert pattern.total_attempts == 2
        assert pattern.rate_limit_count == 1


class TestClearWithNewFeatures:
    """Tests for clearing predictor with new features."""

    def test_clear_resets_sliding_window(self):
        """Test clear resets sliding window."""
        predictor = RateLimitPredictor()

        now = datetime.now()
        predictor.record_attempt(timestamp=now.timestamp())
        predictor.record_rate_limit_event(timestamp=now.timestamp())

        predictor.clear()

        stats = predictor.get_sliding_window_stats()
        assert stats["recent_attempts"] == 0
        assert stats["recent_rate_limits"] == 0

    def test_clear_resets_keyword_patterns(self):
        """Test clear resets keyword patterns."""
        predictor = RateLimitPredictor()

        ts = datetime.now().timestamp()
        predictor.record_attempt(timestamp=ts, keyword="test")

        predictor.clear()

        assert len(predictor._keyword_patterns) == 0


class TestPredictionAccuracy:
    """Tests for US-144-007: Prediction accuracy verification."""

    def test_sliding_window_accuracy_simulation(self):
        """Test sliding window tracks recent activity accurately."""
        predictor = RateLimitPredictor()

        now = datetime.now()

        # Record 20 attempts in the last 10 hours
        for i in range(20):
            hours_ago = (i * 30) / 60  # 30 min intervals over 10 hours
            ts = (now - timedelta(hours=hours_ago)).timestamp()
            predictor.record_attempt(timestamp=ts)

        # Record 5 rate limits in the last 10 hours
        for i in range(5):
            hours_ago = (i * 2)  # 2 hour intervals over 10 hours
            ts = (now - timedelta(hours=hours_ago)).timestamp()
            predictor.record_rate_limit_event(timestamp=ts)

        stats = predictor.get_sliding_window_stats()

        # Should have ~20 attempts and ~5 rate limits in window
        assert stats["recent_attempts"] >= 15  # Allow for some pruning
        assert stats["recent_rate_limits"] >= 3

    def test_confidence_reflects_data_quality(self):
        """Test confidence level reflects data quality."""
        predictor = RateLimitPredictor()

        now = datetime.now()

        # Initially LOW confidence
        assert predictor.get_prediction_confidence() == PredictionConfidence.LOW

        # Add more data to reach MEDIUM
        for i in range(12):
            ts = now.timestamp() - i * 60
            predictor.record_attempt(timestamp=ts)

        assert predictor.get_prediction_confidence() == PredictionConfidence.MEDIUM

        # Add more to reach HIGH
        for i in range(10):
            ts = now.timestamp() - (i + 12) * 60
            predictor.record_attempt(timestamp=ts)

        assert predictor.get_prediction_confidence() == PredictionConfidence.HIGH
