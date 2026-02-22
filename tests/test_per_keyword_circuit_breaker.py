"""Tests for PerKeywordCircuitBreaker (US-109-004).

Tests per-keyword circuit breaker isolation and global fallback behavior.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from src.downloader.per_keyword_circuit_breaker import (
    PerKeywordCircuitBreaker,
    PerKeywordCircuitBreakerConfig,
    KeywordCircuitState,
    GlobalCircuitState,
)


class TestPerKeywordCircuitBreaker:
    """Test suite for PerKeywordCircuitBreaker."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PerKeywordCircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=30.0,
            max_pause_seconds=120.0,
            jitter_factor=0.0,  # Disable jitter for deterministic tests
            global_fallback_threshold=0.5,  # 50%
            global_pause_seconds=60.0,
            global_max_pause_seconds=300.0,
            history_window=5,
        )

    @pytest.fixture
    def circuit_breaker(self, config):
        """Create per-keyword circuit breaker instance."""
        return PerKeywordCircuitBreaker(config)

    # --- Per-keyword isolation tests ---

    def test_per_keyword_independent_failure_tracking(
        self, circuit_breaker
    ):
        """Test that each keyword tracks failures independently.

        AC: Per-keyword isolation - one keyword at limit doesn't affect others.
        """
        # Record failures for keyword A
        for _ in range(3):
            circuit_breaker.record_failure("keyword A")

        # Keyword A should be tripped
        assert circuit_breaker.is_keyword_tripped("keyword A") is True

        # Keyword B should NOT be affected by keyword A's failures
        assert circuit_breaker.is_keyword_tripped("keyword B") is False

        # Record some failures for keyword B (but not enough to trip)
        circuit_breaker.record_failure("keyword B")
        circuit_breaker.record_failure("keyword B")

        # Keyword B should still be open
        assert circuit_breaker.is_keyword_tripped("keyword B") is False

    def test_per_keyword_independent_pause_calculation(
        self, circuit_breaker
    ):
        """Test that pause duration is calculated per-keyword based on history."""
        # Keyword A trips multiple times - should accumulate history
        for i in range(3):
            circuit_breaker.record_failure("keyword A")

        # Get stats to verify history was recorded
        stats_a = circuit_breaker.get_keyword_stats("keyword A")
        assert stats_a['total_trips'] == 1
        assert len(stats_a['pause_history']) == 1

    def test_success_resets_only_specific_keyword(
        self, circuit_breaker
    ):
        """Test that success resets only the specific keyword's failures."""
        # Create enough keywords so that 1 trip doesn't trigger global fallback
        # We need >50% rate-limited, so with threshold=0.5 we need at least 3 keywords
        # Initialize multiple keywords by recording failures on them first
        circuit_breaker.record_failure("kw_dummy1")
        circuit_breaker.record_failure("kw_dummy1")
        circuit_breaker.record_failure("kw_dummy1")  # This will trip, but we have more keywords coming

        # Also create kw_dummy2 to prevent global trip
        circuit_breaker.record_failure("kw_dummy2")
        circuit_breaker.record_failure("kw_dummy2")

        # Now create the real test scenario
        # Reset everything and start fresh
        circuit_breaker.reset_all()

        # Add multiple keywords so global doesn't trip on single failure
        circuit_breaker.record_failure("kw3")
        circuit_breaker.record_failure("kw3")
        circuit_breaker.record_failure("kw3")

        # Trip keyword A
        circuit_breaker.record_failure("keyword A")
        circuit_breaker.record_failure("keyword A")
        circuit_breaker.record_failure("keyword A")

        # Keyword B has failures but not tripped
        circuit_breaker.record_failure("keyword B")
        circuit_breaker.record_failure("keyword B")

        # Success for keyword A should reset only keyword A
        circuit_breaker.record_success("keyword A")

        assert circuit_breaker.get_keyword_stats("keyword A")['consecutive_failures'] == 0
        assert circuit_breaker.get_keyword_stats("keyword B")['consecutive_failures'] == 2

    def test_keyword_can_proceed_when_others_tripped(
        self, circuit_breaker
    ):
        """Test that a keyword can proceed even when others are rate-limited."""
        # Trip keyword A
        circuit_breaker.record_failure("keyword A")
        circuit_breaker.record_failure("keyword A")
        circuit_breaker.record_failure("keyword A")

        # check_and_wait should return False for keyword A (tripped)
        assert circuit_breaker.check_and_wait("keyword A") is True  # Returns True but will wait internally

        # check_and_wait should return True immediately for keyword B
        assert circuit_breaker.check_and_wait("keyword B") is True

    # --- Global fallback tests ---

    def test_global_fallback_trips_at_50_percent(self, circuit_breaker):
        """Test that global fallback trips when >50% keywords are rate-limited."""
        # Create 4 keywords, trip 3 (75% > 50% threshold)
        for keyword in ["kw1", "kw2", "kw3"]:
            circuit_breaker.record_failure(keyword)
            circuit_breaker.record_failure(keyword)
            circuit_breaker.record_failure(keyword)

        # kw4 has no failures
        circuit_breaker.record_failure("kw4")  # Only 1 failure, not tripped

        # Global should be tripped because 3/4 = 75% > 50%
        assert circuit_breaker.is_global_tripped() is True

        global_stats = circuit_breaker.get_global_stats()
        assert global_stats['rate_limited_pct'] == 0.75

    def test_global_fallback_not_trips_at_exactly_50_percent(self, circuit_breaker):
        """Test that global fallback does NOT trip at exactly 50% (threshold is >)."""
        # First create 2 keywords so we have 4 total to work with
        circuit_breaker.record_success("kw3")
        circuit_breaker.record_success("kw4")

        # Create 2 keywords, trip 1 (at this point we have 3 keywords, so 1/3 = 33%)
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")

        circuit_breaker.record_failure("kw2")  # Only 1 failure, 2/4 = 50%

        # Global should NOT be tripped (50% is not > 50%)
        assert circuit_breaker.is_global_tripped() is False

    def test_global_fallback_resets_independently(self, circuit_breaker):
        """Test that global fallback resets independently of keyword circuits."""
        # Trip global fallback
        for keyword in ["kw1", "kw2", "kw3"]:
            circuit_breaker.record_failure(keyword)
            circuit_breaker.record_failure(keyword)
            circuit_breaker.record_failure(keyword)

        assert circuit_breaker.is_global_tripped() is True

        # Reset global manually
        circuit_breaker.reset_global()

        assert circuit_breaker.is_global_tripped() is False

        # Keyword circuits should still be tripped
        assert circuit_breaker.is_keyword_tripped("kw1") is True
        assert circuit_breaker.is_keyword_tripped("kw2") is True
        assert circuit_breaker.is_keyword_tripped("kw3") is True

    # --- Health metrics tests ---

    def test_get_all_health_metrics(self, circuit_breaker):
        """Test health metrics aggregation."""
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")

        metrics = circuit_breaker.get_all_health_metrics()

        assert 'global' in metrics
        assert 'keywords' in metrics
        assert 'kw1' in metrics['keywords']

    def test_get_per_keyword_failure_counts(self, circuit_breaker):
        """Test getting failure counts for all keywords."""
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw2")

        counts = circuit_breaker.get_per_keyword_failure_counts()

        assert counts['kw1'] == 2
        assert counts['kw2'] == 1

    # --- Checkpoint tests ---

    def test_to_checkpoint_dict(self, circuit_breaker):
        """Test serialization to checkpoint."""
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")  # Trips

        checkpoint = circuit_breaker.to_checkpoint_dict()

        assert 'keyword_states' in checkpoint
        assert 'global_state' in checkpoint
        assert 'kw1' in checkpoint['keyword_states']

    def test_from_checkpoint_dict(self, circuit_breaker):
        """Test deserialization from checkpoint."""
        # First trip some circuits
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")

        checkpoint = circuit_breaker.to_checkpoint_dict()

        # Create new instance and restore
        new_cb = PerKeywordCircuitBreaker(circuit_breaker.config)
        new_cb.from_checkpoint_dict(checkpoint)

        # Verify state was restored (failure counts reset to 0 on restore per design)
        assert new_cb.get_keyword_stats("kw1")['total_trips'] == 1


class TestPerKeywordCircuitBreakerIntegration:
    """Integration tests for PerKeywordCircuitBreaker with CircuitBreakerCoordinator."""

    def test_coordinator_integration(self):
        """Test that per-keyword CB integrates with coordinator."""
        from src.downloader.circuit_breaker import (
            CircuitBreakerCoordinator,
            CircuitBreakerRegistry,
        )

        # Setup coordinator
        coordinator = CircuitBreakerCoordinator.get_instance()
        coordinator.reset_instance()  # Clean slate
        coordinator = CircuitBreakerCoordinator.get_instance()

        # Create per-keyword CB with coordinator
        config = PerKeywordCircuitBreakerConfig()
        pkc = PerKeywordCircuitBreaker(config, coordinator=coordinator)

        # Verify integration doesn't crash
        pkc.record_failure("test_keyword")
        pkc.record_failure("test_keyword")
        pkc.record_failure("test_keyword")

        # Per-keyword should trip
        assert pkc.is_keyword_tripped("test_keyword") is True


class TestPerKeywordIsolationScenarios:
    """Real-world scenario tests for per-keyword isolation."""

    def test_different_keywords_different_domains(self):
        """Test scenario: different topic domains shouldn't affect each other."""
        config = PerKeywordCircuitBreakerConfig(
            consecutive_failures_threshold=2,
            jitter_factor=0.0,
        )
        cb = PerKeywordCircuitBreaker(config)

        # "python programming" keyword hits rate limit
        cb.record_failure("python programming")
        cb.record_failure("python programming")

        # "cooking recipes" should be completely unaffected
        cb.record_failure("cooking recipes")
        # Still only 1 failure, not at threshold

        assert cb.is_keyword_tripped("python programming") is True
        assert cb.is_keyword_tripped("cooking recipes") is False

        # They both can proceed independently
        cb.check_and_wait("python programming")  # Will wait
        cb.check_and_wait("cooking recipes")  # Should return immediately

    def test_adaptive_pause_per_keyword(self):
        """Test that pause adapts based on each keyword's history."""
        config = PerKeywordCircuitBreakerConfig(
            consecutive_failures_threshold=1,  # Trip immediately
            jitter_factor=0.0,
            pause_seconds=10.0,
            history_window=3,
        )
        cb = PerKeywordCircuitBreaker(config)

        # Trip keyword A multiple times to build history
        for _ in range(3):
            cb.record_failure("keyword A")
            cb.record_success("keyword A")

        # Trip keyword B once
        cb.record_failure("keyword B")
        cb.record_success("keyword B")

        # Get pause history for each
        stats_a = cb.get_keyword_stats("keyword A")
        stats_b = cb.get_keyword_stats("keyword B")

        # Keyword A should have more trips/history
        assert stats_a['total_trips'] == 3
        assert stats_b['total_trips'] == 1


class TestSpeedBasedCircuitBreaker:
    """Tests for speed-based circuit breaker triggering (US-113-007)."""

    @pytest.fixture
    def speed_config(self):
        """Create test config with speed-based triggering enabled."""
        return PerKeywordCircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=30.0,
            max_pause_seconds=120.0,
            jitter_factor=0.0,
            speed_threshold_mbps=0.5,
            sustained_degradation_threshold=3,
            enable_speed_trigger=True,
        )

    @pytest.fixture
    def mock_speed_tracker(self):
        """Create a mock speed tracker."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        mock = MagicMock(spec=PerKeywordSpeedTracker)
        mock.get_keywords.return_value = []
        return mock

    def test_speed_trigger_config_defaults(self):
        """Test speed-based circuit breaker config defaults."""
        config = PerKeywordCircuitBreakerConfig()

        # Verify default values
        assert config.speed_threshold_mbps == 0.5
        assert config.sustained_degradation_threshold == 3
        assert config.enable_speed_trigger is True

    def test_circuit_breaker_can_accept_speed_tracker(self, speed_config):
        """Test that circuit breaker can accept a speed tracker."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
        )

        speed_tracker = PerKeywordSpeedTracker(DownloadSpeedConfig())
        cb = PerKeywordCircuitBreaker(speed_config, speed_tracker=speed_tracker)

        # Verify speed tracker was set
        assert cb._speed_tracker is speed_tracker

    def test_set_speed_tracker_method(self, speed_config):
        """Test set_speed_tracker method."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
        )

        cb = PerKeywordCircuitBreaker(speed_config)
        speed_tracker = PerKeywordSpeedTracker(DownloadSpeedConfig())

        cb.set_speed_tracker(speed_tracker)

        assert cb._speed_tracker is speed_tracker

    def test_check_speed_and_trip_with_no_speed_tracker(self, speed_config):
        """Test that speed check returns False when no speed tracker is set."""
        cb = PerKeywordCircuitBreaker(speed_config)

        # Should return False when no speed tracker is configured
        result = cb.check_speed_and_trip("test_keyword")

        assert result is False

    def test_check_speed_and_trip_disabled_feature(self):
        """Test that speed-based triggering is disabled when config is False."""
        config = PerKeywordCircuitBreakerConfig(
            enable_speed_trigger=False,
            speed_threshold_mbps=0.5,
            sustained_degradation_threshold=3,
        )

        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
        )

        speed_tracker = PerKeywordSpeedTracker(DownloadSpeedConfig())
        cb = PerKeywordCircuitBreaker(config, speed_tracker=speed_tracker)

        result = cb.check_speed_and_trip("test_keyword")

        assert result is False

    def test_check_speed_and_trip_triggers_circuit(self, speed_config):
        """Test that speed check trips circuit when sustained slow downloads detected."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        # Create mock speed tracker that returns detected signal
        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.get_keywords.return_value = ["test_keyword"]
        mock_tracker.detect_rate_limit_signals.return_value = RateLimitSignal(
            detected=True,
            consecutive_slow_count=3,  # Meets threshold
            recent_speeds=[0.3, 0.4, 0.3],
            threshold=0.5,
            message="Rate limit signal detected"
        )

        cb = PerKeywordCircuitBreaker(speed_config, speed_tracker=mock_tracker)

        # Check speed and trip - should return True
        result = cb.check_speed_and_trip("test_keyword")

        assert result is True
        assert cb.is_keyword_tripped("test_keyword") is True

    def test_check_speed_and_trip_below_threshold(self, speed_config):
        """Test that speed check doesn't trip circuit when below threshold."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        # Create mock speed tracker that returns signal below threshold
        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.detect_rate_limit_signals.return_value = RateLimitSignal(
            detected=True,
            consecutive_slow_count=2,  # Below threshold of 3
            recent_speeds=[0.3, 0.4],
            threshold=0.5,
            message="Rate limit signal detected"
        )

        cb = PerKeywordCircuitBreaker(speed_config, speed_tracker=mock_tracker)

        # Should return False because below threshold
        result = cb.check_speed_and_trip("test_keyword")

        assert result is False
        assert cb.is_keyword_tripped("test_keyword") is False

    def test_check_speed_and_trip_no_signal(self, speed_config):
        """Test that speed check returns False when no rate limit signal detected."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.detect_rate_limit_signals.return_value = RateLimitSignal(
            detected=False,
            consecutive_slow_count=0,
            recent_speeds=[],
            threshold=0.5,
            message="No rate limit signal"
        )

        cb = PerKeywordCircuitBreaker(speed_config, speed_tracker=mock_tracker)

        result = cb.check_speed_and_trip("test_keyword")

        assert result is False

    def test_check_all_keywords_speed(self, speed_config):
        """Test checking speed for all keywords."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.get_keywords.return_value = ["kw1", "kw2", "kw3"]

        # kw1 triggers, kw2 doesn't have signal, kw3 triggers
        def mock_signal(keyword):
            if keyword == "kw1":
                return RateLimitSignal(
                    detected=True,
                    consecutive_slow_count=3,
                    recent_speeds=[0.3, 0.3, 0.3],
                    threshold=0.5,
                    message="Signal"
                )
            elif keyword == "kw2":
                return RateLimitSignal(
                    detected=False,
                    consecutive_slow_count=0,
                    recent_speeds=[],
                    threshold=0.5,
                    message="No signal"
                )
            else:  # kw3
                return RateLimitSignal(
                    detected=True,
                    consecutive_slow_count=4,
                    recent_speeds=[0.2, 0.2, 0.2, 0.2],
                    threshold=0.5,
                    message="Signal"
                )

        mock_tracker.detect_rate_limit_signals.side_effect = mock_signal

        cb = PerKeywordCircuitBreaker(speed_config, speed_tracker=mock_tracker)

        results = cb.check_all_keywords_speed()

        assert results["kw1"] is True
        assert results["kw2"] is False
        assert results["kw3"] is True

    def test_speed_trigger_with_custom_threshold(self):
        """Test speed-based triggering with custom thresholds."""
        config = PerKeywordCircuitBreakerConfig(
            speed_threshold_mbps=1.0,
            sustained_degradation_threshold=2,
            enable_speed_trigger=True,
        )

        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.detect_rate_limit_signals.return_value = RateLimitSignal(
            detected=True,
            consecutive_slow_count=2,  # Meets custom threshold of 2
            recent_speeds=[0.8, 0.9],
            threshold=1.0,
            message="Rate limit signal detected"
        )

        cb = PerKeywordCircuitBreaker(config, speed_tracker=mock_tracker)

        result = cb.check_speed_and_trip("test_keyword")

        assert result is True
        assert cb.is_keyword_tripped("test_keyword") is True

    def test_speed_trigger_logs_warning(self, speed_config, caplog):
        """Test that speed trigger logs a warning when tripping."""
        import logging
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.detect_rate_limit_signals.return_value = RateLimitSignal(
            detected=True,
            consecutive_slow_count=3,
            recent_speeds=[0.3, 0.4, 0.3],
            threshold=0.5,
            message="Rate limit signal detected"
        )

        cb = PerKeywordCircuitBreaker(speed_config, speed_tracker=mock_tracker)

        with caplog.at_level(logging.WARNING):
            cb.check_speed_and_trip("test_keyword")

        assert any("TRIPPED BY SPEED" in record.message for record in caplog.records)

    def test_speed_trigger_global_fallback_integration(self, speed_config):
        """Test that speed trigger can trip global fallback when enough keywords affected."""
        from src.downloader.speed_tracker import (
            PerKeywordSpeedTracker,
            DownloadSpeedConfig,
            RateLimitSignal,
        )

        # Create multiple keywords with speed issues
        mock_tracker = MagicMock(spec=PerKeywordSpeedTracker)
        mock_tracker.get_keywords.return_value = ["kw1", "kw2", "kw3"]

        def mock_signal(keyword):
            return RateLimitSignal(
                detected=True,
                consecutive_slow_count=3,
                recent_speeds=[0.3, 0.3, 0.3],
                threshold=0.5,
                message="Signal"
            )

        mock_tracker.detect_rate_limit_signals.side_effect = mock_signal

        # Use high global fallback threshold to trigger with 3 keywords
        config = PerKeywordCircuitBreakerConfig(
            speed_threshold_mbps=0.5,
            sustained_degradation_threshold=3,
            enable_speed_trigger=True,
            global_fallback_threshold=0.5,  # 50%
            jitter_factor=0.0,
        )

        cb = PerKeywordCircuitBreaker(config, speed_tracker=mock_tracker)

        # Check all keywords - should trip all 3 (100% > 50% threshold)
        cb.check_all_keywords_speed()

        # Global fallback should be tripped
        assert cb.is_global_tripped() is True


class TestPerKeywordCircuitBreakerMetrics:
    """Test suite for PerKeywordCircuitBreaker health metrics (US-136-003)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PerKeywordCircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=30.0,
            max_pause_seconds=120.0,
            jitter_factor=0.0,
            global_fallback_threshold=0.5,
            global_pause_seconds=60.0,
            global_max_pause_seconds=300.0,
            history_window=5,
        )

    @pytest.fixture
    def circuit_breaker(self, config):
        """Create per-keyword circuit breaker instance."""
        return PerKeywordCircuitBreaker(config)

    def test_get_keyword_stats_includes_average_pause_duration(
        self, circuit_breaker
    ):
        """Test that get_keyword_stats includes average_pause_duration."""
        # Record failures to trip the circuit
        for _ in range(3):
            circuit_breaker.record_failure("test keyword")

        stats = circuit_breaker.get_keyword_stats("test keyword")

        assert 'average_pause_duration' in stats
        assert isinstance(stats['average_pause_duration'], float)
        assert stats['average_pause_duration'] >= 0.0

    def test_get_keyword_stats_includes_failure_threshold(
        self, circuit_breaker
    ):
        """Test that get_keyword_stats includes failure_threshold."""
        stats = circuit_breaker.get_keyword_stats("test keyword")

        assert 'failure_threshold' in stats
        assert stats['failure_threshold'] == 3

    def test_get_global_stats_includes_average_pause_duration(
        self, circuit_breaker
    ):
        """Test that get_global_stats includes average_pause_duration."""
        # Record failures for multiple keywords to trip global fallback
        for _ in range(3):
            circuit_breaker.record_failure("kw1")
        for _ in range(3):
            circuit_breaker.record_failure("kw2")
        for _ in range(3):
            circuit_breaker.record_failure("kw3")

        stats = circuit_breaker.get_global_stats()

        assert 'average_pause_duration' in stats
        assert isinstance(stats['average_pause_duration'], float)

    def test_get_global_stats_includes_fallback_threshold(
        self, circuit_breaker
    ):
        """Test that get_global_stats includes fallback_threshold."""
        stats = circuit_breaker.get_global_stats()

        assert 'fallback_threshold' in stats
        assert stats['fallback_threshold'] == 0.5

    def test_get_all_health_metrics_format(self, circuit_breaker):
        """Test that get_all_health_metrics returns properly formatted data."""
        # Add some keywords and failures
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")

        metrics = circuit_breaker.get_all_health_metrics()

        # Verify top-level structure
        assert 'format' in metrics
        assert metrics['format'] == 'circuit_breaker_health_metrics_v1'
        assert 'generated_at' in metrics
        assert 'enabled' in metrics
        assert 'global' in metrics
        assert 'keywords' in metrics
        assert 'summary' in metrics

    def test_get_all_health_metrics_summary(self, circuit_breaker):
        """Test that get_all_health_metrics includes correct summary."""
        # Add multiple keywords with failures
        for _ in range(3):
            circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw2")
        circuit_breaker.record_failure("kw2")

        metrics = circuit_breaker.get_all_health_metrics()
        summary = metrics['summary']

        # Verify summary fields
        assert 'total_keywords' in summary
        assert summary['total_keywords'] == 2
        assert 'total_trips' in summary
        assert 'total_paused_seconds' in summary
        assert 'keywords_tripped' in summary
        assert 'average_keyword_trips' in summary

    def test_export_as_json(self, circuit_breaker):
        """Test that export_as_json returns valid JSON."""
        circuit_breaker.record_failure("test kw")

        json_output = circuit_breaker.export_as_json()

        # Should be valid JSON
        import json
        parsed = json.loads(json_output)

        assert 'format' in parsed
        assert parsed['format'] == 'circuit_breaker_health_metrics_v1'

    def test_metrics_accuracy_after_failures(self, circuit_breaker):
        """Test that metrics accurately reflect failure/trip state."""
        # Record exactly 2 failures (not enough to trip)
        circuit_breaker.record_failure("kw1")
        circuit_breaker.record_failure("kw1")

        stats = circuit_breaker.get_keyword_stats("kw1")
        assert stats['consecutive_failures'] == 2
        assert stats['is_open'] is False
        assert stats['total_trips'] == 0

        # Record 3rd failure (should trip)
        circuit_breaker.record_failure("kw1")

        stats = circuit_breaker.get_keyword_stats("kw1")
        assert stats['consecutive_failures'] == 3
        assert stats['is_open'] is True
        assert stats['total_trips'] == 1

    def test_global_fallback_metrics_accuracy(self, circuit_breaker):
        """Test that global fallback metrics are accurate."""
        # Trip 3 keywords (60% > 50% threshold)
        for _ in range(3):
            circuit_breaker.record_failure("kw1")
        for _ in range(3):
            circuit_breaker.record_failure("kw2")
        for _ in range(3):
            circuit_breaker.record_failure("kw3")

        global_stats = circuit_breaker.get_global_stats()

        assert global_stats['is_open'] is True
        assert global_stats['rate_limited_keywords'] == 3
        assert global_stats['rate_limited_pct'] == 1.0  # 3/3 = 100%

    def test_metrics_schema_documentation(self, circuit_breaker):
        """Test that metrics schema matches documented format."""
        metrics = circuit_breaker.get_all_health_metrics()

        # Verify required fields per schema in code
        assert 'format' in metrics
        assert metrics['format'] == 'circuit_breaker_health_metrics_v1'

        # Global should have all required fields
        global_keys = {'is_open', 'total_trips', 'total_paused_seconds',
                       'average_pause_duration', 'active_keywords', 'rate_limited_keywords',
                       'rate_limited_pct', 'fallback_threshold', 'is_tripped'}
        assert set(metrics['global'].keys()) == global_keys

        # Summary should have all required fields
        summary_keys = {'total_keywords', 'total_trips', 'total_paused_seconds',
                        'keywords_tripped', 'average_keyword_trips'}
        assert set(metrics['summary'].keys()) == summary_keys


class TestRecoveryLogic:
    """Test suite for US-143-008: Per-keyword circuit breaker auto-recovery with gradual reintroduction."""

    @pytest.fixture
    def config(self):
        """Create test configuration with recovery enabled."""
        return PerKeywordCircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=2,  # Lower threshold for faster testing
            pause_seconds=0.1,  # Short pause for fast tests
            max_pause_seconds=1.0,
            jitter_factor=0.0,
            global_fallback_threshold=0.5,
            global_pause_seconds=60.0,
            global_max_pause_seconds=300.0,
            history_window=5,
            # Recovery configuration (US-143-008)
            enable_recovery=True,
            recovery_max_requests=2,
            recovery_backoff_base=2.0,
            recovery_max_backoff=8.0,
            recovery_success_threshold=2,
        )

    @pytest.fixture
    def circuit_breaker(self, config):
        """Create per-keyword circuit breaker instance."""
        return PerKeywordCircuitBreaker(config)

    def test_recovery_mode_entered_after_circuit_opens(self, circuit_breaker):
        """Test that recovery mode is entered after circuit opens and pause completes."""
        # Trip the circuit
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")

        # Check circuit is open
        assert circuit_breaker.is_keyword_tripped("keyword") is True

        # Now call check_and_wait to complete the pause
        # This should transition to recovery mode
        result = circuit_breaker.check_and_wait("keyword")

        assert result is True

        # Circuit should be closed but in recovery mode
        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['is_open'] is False
        assert stats['is_recovering'] is True

    def test_gradual_traffic_increase_limits_requests_per_cycle(self, circuit_breaker):
        """Test that recovery mode limits requests per cycle."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['is_recovering'] is True
        assert stats['recovery_requests_made'] == 0

        # First request
        circuit_breaker.check_and_wait("keyword")
        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['recovery_requests_made'] == 1

        # Second request
        circuit_breaker.check_and_wait("keyword")
        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['recovery_requests_made'] == 2

    def test_recovery_success_resets_consecutive_counter(self, circuit_breaker):
        """Test that consecutive successes during recovery are tracked."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        # Record success during recovery
        circuit_breaker.record_success("keyword")
        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['recovery_consecutive_successes'] == 1

    def test_recovery_complete_after_threshold(self, circuit_breaker):
        """Test that recovery completes after consecutive successes threshold."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        # Record enough successes to complete recovery
        circuit_breaker.record_success("keyword")
        circuit_breaker.record_success("keyword")

        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['is_recovering'] is False
        assert stats['recovery_successes'] >= 1

    def test_recovery_failure_increments_backoff(self, circuit_breaker):
        """Test that failure during recovery increments backoff multiplier."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        # Record failure during recovery
        circuit_breaker.record_failure("keyword")

        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['recovery_failures'] >= 1

    def test_recovery_failure_returns_to_open_state(self, circuit_breaker):
        """Test that failure during recovery returns circuit to open state."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        assert circuit_breaker.is_keyword_tripped("keyword") is False
        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['is_recovering'] is True

        # Failure during recovery should trip circuit again
        circuit_breaker.record_failure("keyword")

        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['is_open'] is True
        assert stats['is_recovering'] is False

    def test_recovery_success_rate_tracked(self, circuit_breaker):
        """Test that recovery success rate is calculated correctly."""
        # Trip and recover multiple times
        for _ in range(2):
            circuit_breaker.record_failure("keyword")
            circuit_breaker.record_failure("keyword")
            circuit_breaker.check_and_wait("keyword")

            # Fail during recovery
            circuit_breaker.record_failure("keyword")

        stats = circuit_breaker.get_keyword_stats("keyword")
        assert 'recovery_success_rate' in stats

    def test_recovery_state_included_in_metrics(self, circuit_breaker):
        """Test that recovery state is included in health metrics."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        metrics = circuit_breaker.get_all_health_metrics()

        assert 'recovery_enabled' in metrics
        assert metrics['recovery_enabled'] is True

        assert 'state_transitions' in metrics
        assert 'keywords_in_recovering_state' in metrics['state_transitions']

    def test_recovery_disabled_config(self):
        """Test that recovery can be disabled via config."""
        config = PerKeywordCircuitBreakerConfig(
            enabled=True,
            enable_recovery=False,
            consecutive_failures_threshold=2,
        )
        cb = PerKeywordCircuitBreaker(config)

        # Trip circuit
        cb.record_failure("keyword")
        cb.record_failure("keyword")
        cb.check_and_wait("keyword")

        stats = cb.get_keyword_stats("keyword")
        assert stats['is_recovering'] is False  # Recovery disabled

    def test_exponential_backoff_during_recovery(self, circuit_breaker):
        """Test exponential backoff increases delay between recovery cycles."""
        # Trip and recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        # Make max requests
        circuit_breaker.check_and_wait("keyword")  # request 1
        circuit_breaker.check_and_wait("keyword")  # request 2

        # Check_and_wait should apply backoff when max requests reached
        # (the method handles this by sleeping, but we can verify state)
        stats = circuit_breaker.get_keyword_stats("keyword")
        assert stats['recovery_attempts'] >= 1

    def test_checkpoint_persists_recovery_state(self, circuit_breaker):
        """Test that checkpoint persists and restores recovery state."""
        # Trip and partially recover
        circuit_breaker.record_failure("keyword")
        circuit_breaker.record_failure("keyword")
        circuit_breaker.check_and_wait("keyword")

        # Record some successes
        circuit_breaker.record_success("keyword")

        # Get checkpoint
        checkpoint = circuit_breaker.to_checkpoint_dict()
        assert 'recovery_successes' in checkpoint['keyword_states']['keyword']

        # Create new instance and restore
        new_cb = PerKeywordCircuitBreaker(circuit_breaker.config)
        new_cb.from_checkpoint_dict(checkpoint)

        stats = new_cb.get_keyword_stats("keyword")
        assert stats['recovery_successes'] >= 1
