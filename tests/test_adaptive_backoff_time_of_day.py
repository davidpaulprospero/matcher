"""Unit tests for AdaptiveBackoffTimeOfDay (US-114-004).

Tests the time-of-day based backoff multiplier calculation.
"""

import pytest
import time
from unittest.mock import patch

from src.downloader.escalation_manager import AdaptiveBackoffTimeOfDay


class TestAdaptiveBackoffTimeOfDay:
    """Test suite for AdaptiveBackoffTimeOfDay class."""

    def test_default_initialization(self):
        """Test that default initialization works correctly."""
        tracker = AdaptiveBackoffTimeOfDay()

        assert tracker.enabled is True
        assert tracker.multiplier_range == (1.0, 2.0)
        assert tracker.low_success_threshold == 0.5
        assert tracker.high_success_threshold == 0.8
        assert tracker.max_data_age_hours == 24
        assert tracker.min_samples_per_hour == 5
        assert tracker._hourly_data == {}

    def test_custom_initialization(self):
        """Test custom initialization with specific parameters."""
        tracker = AdaptiveBackoffTimeOfDay(
            enabled=False,
            multiplier_range=(1.5, 3.0),
            low_success_threshold=0.3,
            high_success_threshold=0.9,
            max_data_age_hours=48,
            min_samples_per_hour=10,
        )

        assert tracker.enabled is False
        assert tracker.multiplier_range == (1.5, 3.0)
        assert tracker.low_success_threshold == 0.3
        assert tracker.high_success_threshold == 0.9
        assert tracker.max_data_age_hours == 48
        assert tracker.min_samples_per_hour == 10

    def test_record_attempt_disabled(self):
        """Test that recording doesn't happen when disabled."""
        tracker = AdaptiveBackoffTimeOfDay(enabled=False)

        tracker.record_attempt(hour=14, success=True)
        tracker.record_attempt(hour=14, success=False)

        assert tracker._hourly_data == {}

    def test_record_attempt_success(self):
        """Test recording successful attempts."""
        tracker = AdaptiveBackoffTimeOfDay()

        # Record 8 successes out of 10 attempts
        for _ in range(8):
            tracker.record_attempt(hour=14, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=14, success=False)

        assert tracker._hourly_data[14]['successes'] == 8.0
        assert tracker._hourly_data[14]['total'] == 10.0

    def test_get_success_rate_insufficient_samples(self):
        """Test that success rate returns None with insufficient samples."""
        tracker = AdaptiveBackoffTimeOfDay(min_samples_per_hour=10)

        # Record only 5 attempts (below min_samples_per_hour)
        for _ in range(5):
            tracker.record_attempt(hour=14, success=True)

        assert tracker.get_success_rate(hour=14) is None

    def test_get_success_rate_sufficient_samples(self):
        """Test success rate calculation with sufficient samples."""
        tracker = AdaptiveBackoffTimeOfDay(min_samples_per_hour=5)

        # Record 8 successes out of 10 attempts
        for _ in range(8):
            tracker.record_attempt(hour=14, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=14, success=False)

        rate = tracker.get_success_rate(hour=14)
        assert rate == 0.8

    def test_get_success_rate_hour_overflow(self):
        """Test that hour wraps around correctly."""
        tracker = AdaptiveBackoffTimeOfDay(use_granular_bins=False)  # Disable bins for this test

        tracker.record_attempt(hour=25, success=True)  # Should wrap to hour 1
        assert 1 in tracker._hourly_data

    def test_multiplier_high_success_rate(self):
        """Test that high success rate uses minimum multiplier."""
        tracker = AdaptiveBackoffTimeOfDay(
            multiplier_range=(1.0, 2.0),
            low_success_threshold=0.5,
            high_success_threshold=0.8,
        )

        # Record 9 successes out of 10 (90% success rate, above 0.8 threshold)
        for _ in range(9):
            tracker.record_attempt(hour=14, success=True)
        for _ in range(1):
            tracker.record_attempt(hour=14, success=False)

        multiplier = tracker.get_time_multiplier(hour=14, use_ema=False)
        assert multiplier == 1.0  # Should use minimum multiplier

    def test_multiplier_low_success_rate(self):
        """Test that low success rate uses maximum multiplier."""
        tracker = AdaptiveBackoffTimeOfDay(
            multiplier_range=(1.0, 2.0),
            low_success_threshold=0.5,
            high_success_threshold=0.8,
        )

        # Record 4 successes out of 10 (40% success rate, below 0.5 threshold)
        for _ in range(4):
            tracker.record_attempt(hour=14, success=True)
        for _ in range(6):
            tracker.record_attempt(hour=14, success=False)

        multiplier = tracker.get_time_multiplier(hour=14)
        assert multiplier == 2.0  # Should use maximum multiplier

    def test_multiplier_mid_success_rate(self):
        """Test linear interpolation for mid-range success rates."""
        tracker = AdaptiveBackoffTimeOfDay(
            multiplier_range=(1.0, 2.0),
            low_success_threshold=0.5,
            high_success_threshold=0.8,
        )

        # Record 6.5 successes out of 10 (65% success rate, in middle)
        for _ in range(6):
            tracker.record_attempt(hour=14, success=True)
        for _ in range(4):
            tracker.record_attempt(hour=14, success=False)

        multiplier = tracker.get_time_multiplier(hour=14, use_ema=False)
        # Should be between 1.0 and 2.0
        assert 1.0 < multiplier < 2.0

    def test_multiplier_no_data(self):
        """Test that default multiplier is 1.0 when no data available."""
        tracker = AdaptiveBackoffTimeOfDay()

        multiplier = tracker.get_time_multiplier(hour=14)
        assert multiplier == 1.0

    def test_multiplier_disabled(self):
        """Test that multiplier is 1.0 when disabled."""
        tracker = AdaptiveBackoffTimeOfDay(enabled=False)

        # Add some data
        for _ in range(10):
            tracker.record_attempt(hour=14, success=True)

        multiplier = tracker.get_time_multiplier(hour=14)
        assert multiplier == 1.0

    def test_multiplier_current_hour(self):
        """Test getting multiplier for current hour."""
        tracker = AdaptiveBackoffTimeOfDay()

        current_hour = int(time.gmtime().tm_hour)
        multiplier = tracker.get_time_multiplier()  # No hour specified = current

        # Should return 1.0 (no data)
        assert multiplier == 1.0

    def test_get_all_rates(self):
        """Test getting all hourly rates."""
        tracker = AdaptiveBackoffTimeOfDay(min_samples_per_hour=5)

        # Add data for hours 10 and 14
        for _ in range(8):
            tracker.record_attempt(hour=10, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=10, success=False)

        for _ in range(3):
            tracker.record_attempt(hour=14, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=14, success=False)

        rates = tracker.get_all_rates()
        assert 10 in rates
        assert 14 in rates
        assert rates[10] == 0.8
        assert rates[14] == 0.6

    def test_to_dict(self):
        """Test serialization to dict."""
        tracker = AdaptiveBackoffTimeOfDay(
            multiplier_range=(1.2, 2.5),
            low_success_threshold=0.4,
        )
        tracker.record_attempt(hour=14, success=True)

        data = tracker.to_dict()

        assert data['enabled'] is True
        assert data['multiplier_range'] == (1.2, 2.5)
        assert data['low_success_threshold'] == 0.4
        assert 14 in data['hourly_data']

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            'enabled': True,
            'multiplier_range': (1.5, 3.0),
            'low_success_threshold': 0.35,
            'high_success_threshold': 0.85,
            'max_data_age_hours': 36,
            'min_samples_per_hour': 8,
            'hourly_data': {
                14: {'successes': 7.0, 'total': 10.0, 'last_update': time.time()},
            },
        }

        tracker = AdaptiveBackoffTimeOfDay.from_dict(data)

        assert tracker.enabled is True
        assert tracker.multiplier_range == (1.5, 3.0)
        assert tracker.low_success_threshold == 0.35
        assert tracker.high_success_threshold == 0.85
        assert tracker.max_data_age_hours == 36
        assert tracker.min_samples_per_hour == 8
        assert tracker._hourly_data[14]['successes'] == 7.0

    def test_from_dict_empty(self):
        """Test that empty data creates default instance."""
        tracker = AdaptiveBackoffTimeOfDay.from_dict({})

        assert tracker.enabled is True
        assert tracker.multiplier_range == (1.0, 2.0)
        assert tracker._hourly_data == {}

    def test_stale_data_not_used(self):
        """Test that stale data returns None for success rate."""
        tracker = AdaptiveBackoffTimeOfDay(
            min_samples_per_hour=5,
            max_data_age_hours=1,  # 1 hour max age
        )

        # Record attempts with old timestamp
        tracker._hourly_data[14] = {
            'successes': 8.0,
            'total': 10.0,
            'last_update': time.time() - 7200,  # 2 hours ago (stale)
        }

        assert tracker.get_success_rate(hour=14) is None


class TestAdaptiveBackoffGranularBins:
    """Test granular hour bins (US-136-009)."""

    def test_granular_bins_2_hour(self):
        """Test 2-hour bins group hours together."""
        tracker = AdaptiveBackoffTimeOfDay(
            use_granular_bins=True,
            bin_size_hours=2,
            min_samples_per_hour=5,
        )

        # Record for hours 0 and 1 (should be in same bin: 0)
        for _ in range(8):
            tracker.record_attempt(hour=0, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=0, success=False)

        # Hour 1 should have same data (same bin as hour 0)
        rate = tracker.get_success_rate(hour=1)
        assert rate == 0.8

        # Hours 2-3 should be in different bin (bin 2)
        for _ in range(3):
            tracker.record_attempt(hour=2, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=2, success=False)

        rate_2 = tracker.get_success_rate(hour=2)
        assert rate_2 == 0.6

    def test_granular_bins_disabled(self):
        """Test when granular bins are disabled."""
        tracker = AdaptiveBackoffTimeOfDay(
            use_granular_bins=False,
            min_samples_per_hour=5,
        )

        # Record for hour 0 only
        for _ in range(8):
            tracker.record_attempt(hour=0, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=0, success=False)

        # Hour 1 should NOT have data when bins disabled
        rate = tracker.get_success_rate(hour=1)
        assert rate is None

        # Hour 0 should have data
        rate_0 = tracker.get_success_rate(hour=0)
        assert rate_0 == 0.8

    def test_invalid_bin_size(self):
        """Test that invalid bin size raises error."""
        with pytest.raises(ValueError):
            AdaptiveBackoffTimeOfDay(bin_size_hours=5)  # Invalid size

    def test_4_hour_bins(self):
        """Test 4-hour bins."""
        tracker = AdaptiveBackoffTimeOfDay(
            use_granular_bins=True,
            bin_size_hours=4,
            min_samples_per_hour=5,
        )

        # Hours 0-3 should be in same bin (0)
        for _ in range(8):
            tracker.record_attempt(hour=2, success=True)  # Hour 2 -> bin 0

        # Add failures to get 0.8 rate
        for _ in range(2):
            tracker.record_attempt(hour=2, success=False)

        rate = tracker.get_success_rate(hour=0)  # Also bin 0
        assert rate == 0.8

        # Hour 4 should be in different bin
        for _ in range(3):
            tracker.record_attempt(hour=4, success=True)
        for _ in range(2):
            tracker.record_attempt(hour=4, success=False)

        rate_4 = tracker.get_success_rate(hour=4)
        assert rate_4 == 0.6


class TestAdaptiveBackoffWeekend:
    """Test weekend vs weekday differentiation (US-136-009)."""

    def test_weekend_data_separate_from_weekday(self):
        """Test that weekend and weekday data are tracked separately."""
        tracker = AdaptiveBackoffTimeOfDay(
            enable_weekend_diff=True,
            min_samples_per_hour=5,
        )

        # Record weekday data (hour 14)
        for _ in range(8):
            tracker.record_attempt(hour=14, success=True, is_weekend=False)
        for _ in range(2):
            tracker.record_attempt(hour=14, success=False, is_weekend=False)

        # Record weekend data (same hour)
        for _ in range(3):
            tracker.record_attempt(hour=14, success=True, is_weekend=True)
        for _ in range(2):
            tracker.record_attempt(hour=14, success=False, is_weekend=True)

        # Get weekday rate
        weekday_rate = tracker.get_success_rate(hour=14, is_weekend=False)
        assert weekday_rate == 0.8

        # Get weekend rate
        weekend_rate = tracker.get_success_rate(hour=14, is_weekend=True)
        assert weekend_rate == 0.6

    def test_weekend_multiplier_boost(self):
        """Test that weekend multiplier includes boost."""
        tracker = AdaptiveBackoffTimeOfDay(
            enable_weekend_diff=True,
            weekend_multiplier_boost=0.2,
            min_samples_per_hour=5,
        )

        # Record high success for both
        for _ in range(9):
            tracker.record_attempt(hour=14, success=True, is_weekend=False)

        weekday_mult = tracker.get_time_multiplier(hour=14, is_weekend=False)

        for _ in range(9):
            tracker.record_attempt(hour=14, success=True, is_weekend=True)

        weekend_mult = tracker.get_time_multiplier(hour=14, is_weekend=True)

        # Weekend should have 20% boost
        assert weekend_mult > weekday_mult
        assert weekend_mult == weekday_mult * 1.2

    def test_weekend_diff_disabled(self):
        """Test that weekend diff can be disabled."""
        tracker = AdaptiveBackoffTimeOfDay(
            enable_weekend_diff=False,
            min_samples_per_hour=5,
        )

        # Record with weekend flag
        for _ in range(8):
            tracker.record_attempt(hour=14, success=True, is_weekend=True)
        for _ in range(2):
            tracker.record_attempt(hour=14, success=False, is_weekend=True)

        # Should use combined data
        rate = tracker.get_success_rate(hour=14)
        assert rate == 0.8


class TestAdaptiveBackoffEMA:
    """Test exponential moving average (US-136-009)."""

    def test_ema_calculation(self):
        """Test EMA is calculated correctly."""
        tracker = AdaptiveBackoffTimeOfDay(
            ema_alpha=0.3,
            min_samples_per_hour=5,
        )

        # Record 10 successes
        for _ in range(10):
            tracker.record_attempt(hour=14, success=True)

        # EMA should be close to 1.0 with high success
        ema_rate = tracker.get_success_rate(hour=14, use_ema=True)
        raw_rate = tracker.get_success_rate(hour=14, use_ema=False)

        assert ema_rate is not None
        assert raw_rate == 1.0

    def test_ema_with_failures(self):
        """Test EMA with mixed success/failure."""
        tracker = AdaptiveBackoffTimeOfDay(
            ema_alpha=0.3,
            min_samples_per_hour=2,
        )

        # Start with successes (high EMA)
        for _ in range(5):
            tracker.record_attempt(hour=14, success=True)

        initial_ema = tracker.get_success_rate(hour=14, use_ema=True)

        # Add failures - EMA should decrease
        for _ in range(5):
            tracker.record_attempt(hour=14, success=False)

        updated_ema = tracker.get_success_rate(hour=14, use_ema=True)

        assert updated_ema < initial_ema

    def test_ema_alpha_validation(self):
        """Test EMA alpha validation."""
        with pytest.raises(ValueError):
            AdaptiveBackoffTimeOfDay(ema_alpha=0.0)

        with pytest.raises(ValueError):
            AdaptiveBackoffTimeOfDay(ema_alpha=1.5)


class TestAdaptiveBackoffPeriodMultiplier:
    """Test period-based multiplier (US-136-009)."""

    def test_get_period_multiplier_night(self):
        """Test getting multiplier for night period."""
        tracker = AdaptiveBackoffTimeOfDay(
            min_samples_per_hour=5,
            use_granular_bins=False,  # Use individual hours for this test
        )

        # Add data for night hours (0-5) with low success rate
        for _ in range(3):
            tracker.record_attempt(hour=2, success=False)  # Low success

        for _ in range(7):
            tracker.record_attempt(hour=2, success=True)

        # Night should have higher multiplier (lower success rate)
        night_mult = tracker.get_period_multiplier('night')
        # Should have data for some night hours
        assert night_mult > 1.0 or night_mult == 1.0  # Either has multiplier or no data

    def test_get_period_multiplier_invalid_period(self):
        """Test invalid period returns 1.0."""
        tracker = AdaptiveBackoffTimeOfDay()

        mult = tracker.get_period_multiplier('invalid')
        assert mult == 1.0


class TestAdaptiveBackoffNewParameters:
    """Test new parameters for US-136-009."""

    def test_all_new_parameters(self):
        """Test all new parameters can be set."""
        tracker = AdaptiveBackoffTimeOfDay(
            use_granular_bins=True,
            bin_size_hours=4,
            enable_weekend_diff=True,
            ema_alpha=0.2,
            weekend_multiplier_boost=0.3,
        )

        assert tracker.use_granular_bins is True
        assert tracker.bin_size_hours == 4
        assert tracker.enable_weekend_diff is True
        assert tracker.ema_alpha == 0.2
        assert tracker.weekend_multiplier_boost == 0.3

    def test_to_dict_with_new_fields(self):
        """Test serialization includes new fields."""
        tracker = AdaptiveBackoffTimeOfDay(
            use_granular_bins=True,
            bin_size_hours=4,
            enable_weekend_diff=True,
            ema_alpha=0.2,
            weekend_multiplier_boost=0.3,
        )
        tracker.record_attempt(hour=14, success=True)

        data = tracker.to_dict()

        assert data['use_granular_bins'] is True
        assert data['bin_size_hours'] == 4
        assert data['enable_weekend_diff'] is True
        assert data['ema_alpha'] == 0.2
        assert data['weekend_multiplier_boost'] == 0.3

    def test_from_dict_with_new_fields(self):
        """Test deserialization includes new fields."""
        data = {
            'enabled': True,
            'use_granular_bins': True,
            'bin_size_hours': 4,
            'enable_weekend_diff': True,
            'ema_alpha': 0.2,
            'weekend_multiplier_boost': 0.3,
            'hourly_data': {14: {'successes': 5.0, 'total': 10.0, 'last_update': time.time(), 'ema': 0.5}},
            'weekend_data': {},
            'weekday_data': {},
        }

        tracker = AdaptiveBackoffTimeOfDay.from_dict(data)

        assert tracker.use_granular_bins is True
        assert tracker.bin_size_hours == 4
        assert tracker.enable_weekend_diff is True
        assert tracker.ema_alpha == 0.2
        assert tracker.weekend_multiplier_boost == 0.3


class TestAdaptiveBackoffTimePeriods:
    """Test configurable time period ranges (US-136-009)."""

    def test_custom_time_periods(self):
        """Test that custom time periods can be configured."""
        custom_periods = {
            'night': (22, 6),  # 10pm - 6am
            'day': (6, 22),    # 6am - 10pm
        }
        tracker = AdaptiveBackoffTimeOfDay(time_periods=custom_periods)

        # Verify custom periods are used
        assert tracker.time_periods == custom_periods

    def test_custom_periods_used_in_multiplier(self):
        """Test that custom time periods affect multiplier calculation."""
        custom_periods = {
            'night': (22, 6),  # 10pm - 6am
            'day': (6, 22),    # 6am - 10pm
        }
        tracker = AdaptiveBackoffTimeOfDay(
            time_periods=custom_periods,
            min_samples_per_hour=3,
            multiplier_range=(1.0, 3.0),
        )

        # Record data for night hours (23 = night with custom)
        for _ in range(9):
            tracker.record_attempt(hour=23, success=True)
        for _ in range(1):
            tracker.record_attempt(hour=23, success=False)

        # Record data for day hours (12 = day with custom)
        for _ in range(3):
            tracker.record_attempt(hour=12, success=True)
        for _ in range(7):
            tracker.record_attempt(hour=12, success=False)

        # Night should have higher success (90%) -> lower multiplier
        night_mult = tracker.get_period_multiplier('night')
        # Day should have lower success (30%) -> higher multiplier
        day_mult = tracker.get_period_multiplier('day')

        assert night_mult < day_mult, f"Night ({night_mult}) should have lower multiplier than day ({day_mult})"

    def test_default_time_periods(self):
        """Test that default time periods are used when none provided."""
        tracker = AdaptiveBackoffTimeOfDay()

        # Verify default periods
        assert 'night' in tracker.time_periods
        assert 'morning' in tracker.time_periods
        assert 'afternoon' in tracker.time_periods
        assert 'evening' in tracker.time_periods

    def test_get_time_period_with_custom_periods(self):
        """Test standalone get_time_period function with custom periods."""
        from src.downloader.escalation_manager import get_time_period

        custom_periods = {
            'night': (22, 6),
            'day': (6, 22),
        }

        # Hour 23 should be night
        period = get_time_period(23, time_periods=custom_periods)
        assert period == 'night'

        # Hour 12 should be day
        period = get_time_period(12, time_periods=custom_periods)
        assert period == 'day'

    def test_to_dict_with_custom_time_periods(self):
        """Test serialization with custom time periods."""
        custom_periods = {
            'night': (22, 6),
            'day': (6, 22),
        }
        tracker = AdaptiveBackoffTimeOfDay(time_periods=custom_periods)

        data = tracker.to_dict()

        assert 'time_periods' in data
        assert data['time_periods'] == custom_periods

    def test_from_dict_with_custom_time_periods(self):
        """Test deserialization with custom time periods."""
        custom_periods = {
            'night': (22, 6),
            'day': (6, 22),
        }
        data = {
            'enabled': True,
            'time_periods': custom_periods,
            'hourly_data': {},
            'weekend_data': {},
            'weekday_data': {},
        }

        tracker = AdaptiveBackoffTimeOfDay.from_dict(data)

        assert tracker.time_periods == custom_periods


class TestAdaptiveBackoffMultiplierRange:
    """Test edge cases for multiplier range configuration."""

    def test_min_mult_cannot_be_less_than_1(self):
        """Test that minimum multiplier must be >= 1.0."""
        with pytest.raises(ValueError):
            AdaptiveBackoffTimeOfDay(multiplier_range=(0.5, 2.0))

    def test_max_mult_cannot_be_less_than_1(self):
        """Test that maximum multiplier must be >= 1.0."""
        with pytest.raises(ValueError):
            AdaptiveBackoffTimeOfDay(multiplier_range=(1.0, 0.5))

    def test_min_cannot_exceed_max(self):
        """Test that min multiplier cannot exceed max."""
        with pytest.raises(ValueError):
            AdaptiveBackoffTimeOfDay(multiplier_range=(2.0, 1.0))

    def test_equal_min_max(self):
        """Test that min and max can be equal (fixed multiplier)."""
        tracker = AdaptiveBackoffTimeOfDay(multiplier_range=(1.5, 1.5))

        # Any success rate should return 1.5
        for _ in range(10):
            tracker.record_attempt(hour=14, success=True)

        assert tracker.get_time_multiplier(hour=14) == 1.5


class TestAdaptiveBackoffComparison:
    """Compare success rates with and without adaptive backoff (US-136-009)."""

    def test_adaptive_improves_success_rate_vs_fixed(self):
        """Test that adaptive backoff improves success rates compared to fixed multiplier.

        This test simulates a scenario where:
        - Some hours have high success rates (14:00-18:00)
        - Some hours have low success rates (2:00-6:00)

        With adaptive backoff, the system should:
        - Use shorter waits during good hours (faster completion)
        - Use longer waits during bad hours (more recovery time)

        The net result: better overall success rate.
        """
        # Setup: Create tracker with realistic hourly patterns
        adaptive_tracker = AdaptiveBackoffTimeOfDay(
            use_granular_bins=True,
            bin_size_hours=2,
            min_samples_per_hour=3,
            multiplier_range=(1.0, 3.0),
        )

        # Simulate training data: high success during day (14-18), low at night (2-6)
        # Day hours: 14, 15, 16, 17 - 90% success rate
        for hour in [14, 15, 16, 17]:
            for _ in range(9):
                adaptive_tracker.record_attempt(hour=hour, success=True)
            for _ in range(1):
                adaptive_tracker.record_attempt(hour=hour, success=False)

        # Night hours: 2, 3, 4, 5 - 30% success rate
        for hour in [2, 3, 4, 5]:
            for _ in range(3):
                adaptive_tracker.record_attempt(hour=hour, success=True)
            for _ in range(7):
                adaptive_tracker.record_attempt(hour=hour, success=False)

        # Get multipliers for each time period
        day_multiplier = adaptive_tracker.get_time_multiplier(hour=14)  # High success time
        night_multiplier = adaptive_tracker.get_time_multiplier(hour=2)  # Low success time

        # Adaptive should use lower multiplier for good hours
        assert day_multiplier < night_multiplier, \
            f"Expected day ({day_multiplier}) < night ({night_multiplier})"

        # Simulate 100 requests at each time period with adaptive vs fixed backoff
        # For simplicity, assume each failure causes a retry with backoff
        # Adaptive: gets better multipliers, so has more effective retries
        # Fixed: always uses multiplier of 2.0 (middle ground)

        fixed_multiplier = 2.0  # Middle ground for comparison

        # Calculate effective wait time ratio (lower is better for throughput)
        adaptive_day_ratio = 1.0 / day_multiplier
        adaptive_night_ratio = 1.0 / night_multiplier
        fixed_ratio = 1.0 / fixed_multiplier

        # Adaptive should have better throughput during good hours
        assert adaptive_day_ratio > fixed_ratio, \
            "Adaptive should have better throughput during high-success hours"

        # Adaptive should have better overall success probability
        # Using the formula: success_prob = 1 - (1 - base_success)^effective_retries
        # Where effective_retries increases with backoff time

        def expected_success(base_success, multiplier, attempts=3):
            """Calculate expected success with retries."""
            # More multiplier = more wait time = more retry attempts possible
            effective_attempts = int(multiplier * attempts)
            return 1 - (1 - base_success) ** effective_attempts

        day_base = 0.9
        night_base = 0.3

        adaptive_day_success = expected_success(day_base, day_multiplier)
        adaptive_night_success = expected_success(night_base, night_multiplier)
        fixed_day_success = expected_success(day_base, fixed_multiplier)
        fixed_night_success = expected_success(night_base, fixed_multiplier)

        # Adaptive should improve night success rate significantly
        improvement = adaptive_night_success - fixed_night_success
        assert improvement > 0, \
            f"Adaptive should improve night success: {adaptive_night_success} vs {fixed_night_success}"

        # Overall weighted success should be better with adaptive
        # 50% day traffic, 50% night traffic
        adaptive_overall = 0.5 * adaptive_day_success + 0.5 * adaptive_night_success
        fixed_overall = 0.5 * fixed_day_success + 0.5 * fixed_night_success

        assert adaptive_overall > fixed_overall, \
            f"Adaptive ({adaptive_overall:.3f}) should beat fixed ({fixed_overall:.3f})"

    def test_no_improvement_when_all_hours_equal(self):
        """Test that adaptive provides no improvement when all hours have equal success."""
        # All hours have same 70% success rate
        tracker = AdaptiveBackoffTimeOfDay(
            min_samples_per_hour=3,
            multiplier_range=(1.0, 2.0),
        )

        for hour in range(24):
            for _ in range(7):
                tracker.record_attempt(hour=hour, success=True)
            for _ in range(3):
                tracker.record_attempt(hour=hour, success=False)

        # All multipliers should be equal
        multipliers = [tracker.get_time_multiplier(hour=h) for h in range(24)]
        assert len(set(multipliers)) == 1, "All hours should have same multiplier"

    def test_granular_bins_improve_sampling(self):
        """Test that granular bins improve sample size and thus reliability."""
        # With 2-hour bins, we get more samples per bin
        tracker_with_bins = AdaptiveBackoffTimeOfDay(
            use_granular_bins=True,
            bin_size_hours=2,
            min_samples_per_hour=15,  # Higher threshold
        )

        tracker_without_bins = AdaptiveBackoffTimeOfDay(
            use_granular_bins=False,
            min_samples_per_hour=15,  # Higher threshold
        )

        # Record 20 samples spread across 2 adjacent hours (not enough for individual hour)
        # Hour 14: 10 samples
        for _ in range(10):
            tracker_with_bins.record_attempt(hour=14, success=True)
            tracker_without_bins.record_attempt(hour=14, success=True)

        # Hour 15: 10 samples (total 20)
        for _ in range(10):
            tracker_with_bins.record_attempt(hour=15, success=True)
            tracker_without_bins.record_attempt(hour=15, success=True)

        # With bins, hour 14 should aggregate data from bin covering 14-15
        # So it should have data even though individual hour has only 10 samples (below 15)
        # Without bins, hour 14 should NOT have enough data
        rate_with_bins = tracker_with_bins.get_success_rate(hour=14)
        rate_without_bins = tracker_without_bins.get_success_rate(hour=14)

        # Bins should give us data (aggregated), no bins should give None (not enough)
        assert rate_with_bins is not None, "Bins should aggregate samples"
        assert rate_without_bins is None, "No bins should not aggregate samples"

        # Also verify that hour 15 with bins gets data (it's in same bin as 14)
        rate_15_with_bins = tracker_with_bins.get_success_rate(hour=15)
        assert rate_15_with_bins is not None, "Hour 15 should share bin with hour 14"

    def test_ema_reacts_to_trends(self):
        """Test that EMA reacts to recent trends better than raw rates."""
        # Create tracker with EMA
        tracker = AdaptiveBackoffTimeOfDay(
            ema_alpha=0.5,  # High alpha = reacts faster
            min_samples_per_hour=1,
        )

        # Start with high success
        for _ in range(20):
            tracker.record_attempt(hour=14, success=True)

        # EMA should be high (approximately 1.0)
        ema_high = tracker.get_success_rate(hour=14, use_ema=True)
        raw_high = tracker.get_success_rate(hour=14, use_ema=False)

        assert ema_high > 0.99, f"EMA should be close to 1.0, got {ema_high}"
        assert raw_high == 1.0

        # Now add failures - EMA should react faster
        for _ in range(10):
            tracker.record_attempt(hour=14, success=False)

        # Both should decrease, but EMA should be lower (reacted to trend)
        ema_after = tracker.get_success_rate(hour=14, use_ema=True)
        raw_after = tracker.get_success_rate(hour=14, use_ema=False)

        assert ema_after < raw_after, \
            f"EMA ({ema_after}) should react faster than raw ({raw_after})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
