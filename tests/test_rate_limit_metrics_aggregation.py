"""Tests for rate limit metrics aggregation on resume (US-006).

Tests cross-session metrics aggregation to verify:
- from_checkpoint() increments session_count
- Metrics accumulate correctly across save/load cycles
- Summary shows 'Total across N sessions' when session_count > 1
- reset() clears session_count along with other metrics
"""

import pytest
from unittest.mock import patch

from src.downloader.rate_limit_metrics import RateLimitMetrics


class TestSessionCount:
    """Test session_count field behavior."""

    @pytest.mark.fast
    def test_default_session_count_is_one(self):
        """Test that new metrics start with session_count=1."""
        metrics = RateLimitMetrics()
        assert metrics.session_count == 1

    @pytest.mark.fast
    def test_session_count_can_be_set(self):
        """Test session_count can be set on creation."""
        metrics = RateLimitMetrics(session_count=5)
        assert metrics.session_count == 5

    @pytest.mark.fast
    def test_session_count_in_to_dict(self):
        """Test session_count is included in serialization."""
        metrics = RateLimitMetrics(session_count=3)
        data = metrics.to_dict()
        assert 'session_count' in data
        assert data['session_count'] == 3


class TestFromCheckpoint:
    """Test from_checkpoint() classmethod."""

    @pytest.mark.fast
    def test_from_checkpoint_increments_session_count(self):
        """Test that from_checkpoint() increments session_count."""
        data = {
            'total_downloads': 50,
            'session_count': 1
        }
        metrics = RateLimitMetrics.from_checkpoint(data)
        assert metrics.session_count == 2

    @pytest.mark.fast
    def test_from_checkpoint_empty_data(self):
        """Test from_checkpoint with empty data returns default metrics."""
        metrics = RateLimitMetrics.from_checkpoint({})
        assert metrics.session_count == 1  # Default, not incremented
        assert metrics.total_downloads == 0

    @pytest.mark.fast
    def test_from_checkpoint_none_data(self):
        """Test from_checkpoint with None returns default metrics."""
        metrics = RateLimitMetrics.from_checkpoint(None)
        assert metrics.session_count == 1
        assert metrics.total_downloads == 0

    @pytest.mark.fast
    def test_from_checkpoint_preserves_all_metrics(self):
        """Test from_checkpoint preserves all metric values."""
        data = {
            'total_downloads': 100,
            'successful_downloads': 90,
            'failed_downloads': 10,
            'retry_attempts': 25,
            'rate_limit_events': 8,
            'backoff_attempts': 5,
            'time_spent_backing_off': 75.0,
            'cookie_rotations': 2,
            'vpn_switches': 1,
            'session_count': 2
        }
        metrics = RateLimitMetrics.from_checkpoint(data)

        # Verify all values preserved
        assert metrics.total_downloads == 100
        assert metrics.successful_downloads == 90
        assert metrics.failed_downloads == 10
        assert metrics.retry_attempts == 25
        assert metrics.rate_limit_events == 8
        assert metrics.backoff_attempts == 5
        assert metrics.time_spent_backing_off == 75.0
        assert metrics.cookie_rotations == 2
        assert metrics.vpn_switches == 1
        # Session count incremented
        assert metrics.session_count == 3

    @pytest.mark.fast
    def test_from_checkpoint_logs_restoration(self):
        """Test from_checkpoint logs the restoration."""
        data = {
            'total_downloads': 50,
            'rate_limit_events': 5,
            'session_count': 1
        }
        with patch('src.downloader.rate_limit_metrics.logger') as mock_logger:
            metrics = RateLimitMetrics.from_checkpoint(data)
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert 'session 2' in log_message
            assert '50 downloads' in log_message
            assert '5 rate limits' in log_message

    @pytest.mark.fast
    def test_from_checkpoint_vs_from_dict(self):
        """Test that from_checkpoint increments while from_dict does not."""
        data = {'session_count': 3, 'total_downloads': 100}

        metrics_from_dict = RateLimitMetrics.from_dict(data)
        assert metrics_from_dict.session_count == 3  # Not incremented

        metrics_from_checkpoint = RateLimitMetrics.from_checkpoint(data)
        assert metrics_from_checkpoint.session_count == 4  # Incremented


class TestFromDictSessionCount:
    """Test from_dict() handles session_count."""

    @pytest.mark.fast
    def test_from_dict_preserves_session_count(self):
        """Test from_dict preserves session_count without incrementing."""
        data = {'session_count': 5, 'total_downloads': 100}
        metrics = RateLimitMetrics.from_dict(data)
        assert metrics.session_count == 5

    @pytest.mark.fast
    def test_from_dict_missing_session_count_defaults_to_one(self):
        """Test from_dict defaults session_count to 1 when missing."""
        data = {'total_downloads': 100}
        metrics = RateLimitMetrics.from_dict(data)
        assert metrics.session_count == 1


class TestSummaryCrossSessions:
    """Test summary() shows cross-session info."""

    @pytest.mark.fast
    def test_summary_single_session_no_indicator(self):
        """Test summary does not show session indicator for single session."""
        metrics = RateLimitMetrics(
            total_downloads=50,
            successful_downloads=45,
            session_count=1
        )
        summary = metrics.summary()
        assert 'sessions:' not in summary.lower()

    @pytest.mark.fast
    def test_summary_multiple_sessions_shows_indicator(self):
        """Test summary shows 'Total across N sessions' when session_count > 1."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            session_count=3
        )
        summary = metrics.summary()
        assert 'Total across 3 sessions:' in summary

    @pytest.mark.fast
    def test_summary_two_sessions(self):
        """Test summary for exactly 2 sessions."""
        metrics = RateLimitMetrics(
            total_downloads=75,
            session_count=2
        )
        summary = metrics.summary()
        assert 'Total across 2 sessions:' in summary


class TestClearSessionCount:
    """Test clear() resets session_count."""

    @pytest.mark.fast
    def test_clear_resets_session_count_to_one(self):
        """Test clear() resets session_count to 1."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            session_count=5
        )
        metrics.clear()
        assert metrics.session_count == 1

    @pytest.mark.fast
    def test_clear_resets_all_including_session_count(self):
        """Test clear() resets everything including session_count."""
        metrics = RateLimitMetrics(
            total_downloads=100,
            successful_downloads=90,
            rate_limit_events=10,
            session_count=5
        )
        metrics.clear()

        assert metrics.total_downloads == 0
        assert metrics.successful_downloads == 0
        assert metrics.rate_limit_events == 0
        assert metrics.session_count == 1


class TestMultipleSaveLoadCycles:
    """Test cumulative metrics across multiple save/load cycles."""

    @pytest.mark.fast
    def test_three_session_roundtrip(self):
        """Test metrics accumulate correctly across 3 sessions."""
        # Session 1
        metrics1 = RateLimitMetrics()
        metrics1.record_download_attempt()
        metrics1.record_download_attempt()
        metrics1.record_download_success()
        metrics1.record_download_success()
        metrics1.record_rate_limit_event()
        assert metrics1.session_count == 1
        assert metrics1.total_downloads == 2

        # Save session 1
        checkpoint1 = metrics1.to_dict()

        # Session 2: Resume from checkpoint
        metrics2 = RateLimitMetrics.from_checkpoint(checkpoint1)
        assert metrics2.session_count == 2
        assert metrics2.total_downloads == 2  # Preserved from session 1

        # Add more in session 2
        metrics2.record_download_attempt()
        metrics2.record_download_success()
        metrics2.record_rate_limit_event()
        assert metrics2.total_downloads == 3
        assert metrics2.rate_limit_events == 2

        # Save session 2
        checkpoint2 = metrics2.to_dict()

        # Session 3: Resume from checkpoint
        metrics3 = RateLimitMetrics.from_checkpoint(checkpoint2)
        assert metrics3.session_count == 3
        assert metrics3.total_downloads == 3
        assert metrics3.rate_limit_events == 2

        # Add more in session 3
        metrics3.record_download_attempt()
        metrics3.record_download_failure()
        assert metrics3.total_downloads == 4
        assert metrics3.failed_downloads == 1

        # Verify summary shows cross-session indicator
        summary = metrics3.summary()
        assert 'Total across 3 sessions:' in summary

    @pytest.mark.fast
    def test_cumulative_backoff_time(self):
        """Test backoff time accumulates across sessions."""
        # Session 1
        metrics1 = RateLimitMetrics()
        metrics1.record_backoff(10.0)
        metrics1.record_backoff(20.0)
        checkpoint1 = metrics1.to_dict()

        # Session 2
        metrics2 = RateLimitMetrics.from_checkpoint(checkpoint1)
        metrics2.record_backoff(15.0)

        assert metrics2.time_spent_backing_off == 45.0  # 10 + 20 + 15
        assert metrics2.backoff_attempts == 3
        assert metrics2.session_count == 2

    @pytest.mark.fast
    def test_cumulative_retries_by_type(self):
        """Test retries_by_error_type accumulates across sessions."""
        # Session 1: 3 timeout, 2 transient
        metrics1 = RateLimitMetrics()
        metrics1.record_retry('timeout')
        metrics1.record_retry('timeout')
        metrics1.record_retry('timeout')
        metrics1.record_retry('transient')
        metrics1.record_retry('transient')
        checkpoint1 = metrics1.to_dict()

        # Session 2: 1 timeout, 3 network
        metrics2 = RateLimitMetrics.from_checkpoint(checkpoint1)
        metrics2.record_retry('timeout')
        metrics2.record_retry('network')
        metrics2.record_retry('network')
        metrics2.record_retry('network')

        assert metrics2.retries_by_error_type == {
            'timeout': 4,  # 3 + 1
            'transient': 2,
            'network': 3
        }
        assert metrics2.retry_attempts == 9

    @pytest.mark.fast
    def test_cumulative_escalations(self):
        """Test cookie rotations and VPN switches accumulate."""
        # Session 1
        metrics1 = RateLimitMetrics()
        metrics1.record_cookie_rotation()
        metrics1.record_cookie_rotation()
        metrics1.record_vpn_switch()
        checkpoint1 = metrics1.to_dict()

        # Session 2
        metrics2 = RateLimitMetrics.from_checkpoint(checkpoint1)
        metrics2.record_cookie_rotation()
        metrics2.record_vpn_switch()
        metrics2.record_vpn_switch()

        assert metrics2.cookie_rotations == 3
        assert metrics2.vpn_switches == 3

    @pytest.mark.fast
    def test_speed_metrics_accumulation(self):
        """Test speed samples accumulate across sessions."""
        # Session 1: 2 samples, avg 5.0
        metrics1 = RateLimitMetrics()
        metrics1.record_speed_sample(4.0)
        metrics1.record_speed_sample(6.0)
        metrics1.record_timeout_extension()
        checkpoint1 = metrics1.to_dict()

        assert metrics1.speed_samples == 2
        assert metrics1.avg_speed_mbps == 5.0  # (4+6)/2

        # Session 2: Additional samples
        metrics2 = RateLimitMetrics.from_checkpoint(checkpoint1)
        # Note: speed recording uses running average, so new samples update the average
        metrics2.record_speed_sample(8.0)
        metrics2.record_timeout_extension()

        assert metrics2.speed_samples == 3  # 2 + 1
        assert metrics2.timeout_extensions == 2  # 1 + 1
        # Running average: 5.0 + (8.0 - 5.0) / 3 = 6.0
        assert abs(metrics2.avg_speed_mbps - 6.0) < 0.01


class TestBackwardCompatibility:
    """Test backward compatibility with checkpoints missing session_count."""

    @pytest.mark.fast
    def test_from_checkpoint_missing_session_count(self):
        """Test from_checkpoint handles missing session_count (old checkpoint)."""
        # Simulate old checkpoint without session_count
        old_checkpoint = {
            'total_downloads': 50,
            'rate_limit_events': 5
            # No session_count field
        }
        metrics = RateLimitMetrics.from_checkpoint(old_checkpoint)

        # Should default to 1 and then increment to 2
        assert metrics.session_count == 2
        assert metrics.total_downloads == 50

    @pytest.mark.fast
    def test_from_dict_missing_session_count(self):
        """Test from_dict handles missing session_count."""
        old_checkpoint = {
            'total_downloads': 50
        }
        metrics = RateLimitMetrics.from_dict(old_checkpoint)
        assert metrics.session_count == 1


class TestConfigRecommendationsWithSessions:
    """Test config recommendations work with cross-session data."""

    @pytest.mark.fast
    def test_recommendations_based_on_cumulative_data(self):
        """Test recommendations use cumulative data across sessions."""
        # Simulate high rate limiting accumulated over 3 sessions
        metrics = RateLimitMetrics(
            total_downloads=100,
            rate_limit_events=15,  # 15% > 10% threshold
            cookie_rotations=0,
            session_count=3
        )
        recommendations = metrics.get_config_recommendations()

        # Should still give recommendations based on cumulative data
        assert len(recommendations) >= 1
        assert any('rate limiting' in r.lower() for r in recommendations)
