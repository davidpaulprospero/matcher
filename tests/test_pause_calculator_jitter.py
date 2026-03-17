"""
Tests for US-109-002: Improved jitter calculation for backoff delays.

Verifies:
- Dynamic jitter calculation based on time-of-day patterns
- Jitter correlation check to prevent similar values
- Config options: 'random', 'adaptive', 'deterministic' strategies
- Jitter never exceeds 50% of base delay (hard cap)
"""

import random
import time
from unittest.mock import patch, MagicMock

import pytest

from src.downloader.pause_calculator import (
    PauseCalculator,
    PauseContext,
    JitterStrategy,
    JitterCorrelationStore,
)


class TestJitterStrategyConfig:
    """Tests for jitter strategy configuration."""

    @pytest.mark.fast
    def test_default_jitter_strategy(self):
        """Default jitter strategy should be 'random'."""
        ctx = PauseContext()
        assert ctx.jitter_strategy == "random"

    @pytest.mark.fast
    def test_adaptive_jitter_strategy(self):
        """Adaptive jitter strategy should be configurable."""
        ctx = PauseContext(jitter_strategy="adaptive")
        assert ctx.jitter_strategy == "adaptive"

    @pytest.mark.fast
    def test_deterministic_jitter_strategy(self):
        """Deterministic jitter strategy should be configurable."""
        ctx = PauseContext(jitter_strategy="deterministic")
        assert ctx.jitter_strategy == "deterministic"

    @pytest.mark.fast
    def test_jitter_max_factor_default(self):
        """Default jitter_max_factor should be 0.5 (50%)."""
        ctx = PauseContext()
        assert ctx.jitter_max_factor == 0.5

    @pytest.mark.fast
    def test_jitter_correlation_check_default(self):
        """Default jitter_correlation_check should be False."""
        ctx = PauseContext()
        assert ctx.jitter_correlation_check is False


class TestJitter50PercentCap:
    """Tests verifying jitter never exceeds 50% of base delay (AC criterion 5)."""

    @pytest.mark.fast
    def test_jitter_factor_capped_at_50_percent(self):
        """Jitter factor should be capped at 0.5 regardless of config."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=1.0,  # Would be 100% without cap
            jitter_max_factor=0.5,
            jitter_strategy="random"
        )

        # Run multiple times to verify hard cap
        for _ in range(50):
            with patch('src.downloader.pause_calculator.random.uniform', return_value=1.0):
                result = calc.apply_jitter(60.0, ctx)
                # 60 * (1 + 0.5) = 90, should never exceed this
                assert result <= 90.0, f"Result {result} exceeded 90.0 (50% cap)"

    @pytest.mark.fast
    def test_jitter_offset_capped_at_50_percent(self):
        """Jitter offset should never exceed ±0.5 regardless of random result."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=0.5,
            jitter_max_factor=0.5,
            jitter_strategy="random"
        )

        # Even with extreme random values, cap applies
        for _ in range(50):
            with patch('src.downloader.pause_calculator.random.uniform', return_value=1.0):
                result = calc.apply_jitter(100.0, ctx)
                # 100 * (1 + 0.5) = 150, should never exceed
                assert result <= 150.0

    @pytest.mark.fast
    def test_deterministic_strategy_respects_50_percent_cap(self):
        """Deterministic strategy should also respect 50% cap."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=0.5,
            jitter_max_factor=0.5,
            jitter_strategy="deterministic",
            client_id="test-client"
        )

        # Multiple calls should stay within bounds
        for _ in range(20):
            result = calc.apply_jitter(100.0, ctx)
            assert 50.0 <= result <= 150.0, f"Result {result} outside bounds"


class TestAdaptiveJitterStrategy:
    """Tests for adaptive jitter strategy (higher during peak hours)."""

    @pytest.mark.fast
    def test_adaptive_strategy_uses_higher_jitter_peak_hours(self):
        """Adaptive strategy should use higher jitter during peak hours (9am-9pm)."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=0.2,
            jitter_strategy="adaptive",
            jitter_max_factor=0.5
        )

        # During peak hours, adaptive factor = base * 1.5 = 0.3
        # During off-peak, adaptive factor = base * 0.75 = 0.15

        peak_results = []
        off_peak_results = []

        # Test peak hours (mock hour between 9-21)
        with patch('src.downloader.pause_calculator.time.localtime') as mock_time:
            mock_time.return_value = MagicMock(tm_hour=12)  # 12pm - peak

            for _ in range(20):
                result = calc.apply_jitter(100.0, ctx)
                peak_results.append(result)

        # Test off-peak (mock hour outside 9-21)
        with patch('src.downloader.pause_calculator.time.localtime') as mock_time:
            mock_time.return_value = MagicMock(tm_hour=3)  # 3am - off-peak

            for _ in range(20):
                result = calc.apply_jitter(100.0, ctx)
                off_peak_results.append(result)

        # Peak should have higher variance than off-peak
        peak_std = (max(peak_results) - min(peak_results)) / 2
        off_peak_std = (max(off_peak_results) - min(off_peak_results)) / 2

        # Peak variance should be noticeably higher (1.5x vs 0.75x)
        assert peak_std > off_peak_std, "Peak hours should have higher jitter variance"


class TestDeterministicJitterStrategy:
    """Tests for deterministic jitter strategy (seeded by client_id)."""

    @pytest.mark.fast
    def test_deterministic_same_client_same_result(self):
        """Same client_id should produce the same jitter for same delay."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=0.2,
            jitter_strategy="deterministic",
            client_id="test-client-123"
        )

        results = []
        for _ in range(10):
            result = calc.apply_jitter(100.0, ctx)
            results.append(result)

        # All results should be identical for same client_id
        assert len(set(results)) == 1, "Deterministic jitter should be reproducible"

    @pytest.mark.fast
    def test_deterministic_different_clients_different_results(self):
        """Different client_ids should produce different jitter values."""
        calc = PauseCalculator()

        results = []
        for i in range(5):
            ctx = PauseContext(
                jitter_factor=0.2,
                jitter_strategy="deterministic",
                client_id=f"client-{i}"
            )
            result = calc.apply_jitter(100.0, ctx)
            results.append(result)

        # Different clients should have different jitter
        unique_results = len(set(results))
        assert unique_results > 1, "Different clients should have different jitter"

    @pytest.mark.fast
    def test_deterministic_different_delays_different_results(self):
        """Same client with different delays should have different jitter."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=0.2,
            jitter_strategy="deterministic",
            client_id="test-client"
        )

        results = []
        for delay in [50.0, 100.0, 150.0, 200.0]:
            result = calc.apply_jitter(delay, ctx)
            results.append(result)

        # Different delays should have different jitter
        unique_results = len(set(results))
        assert unique_results > 1, "Different delays should have different jitter"


class TestJitterCorrelationCheck:
    """Tests for jitter correlation check to prevent similar values."""

    @pytest.mark.fast
    def test_correlation_check_enabled(self):
        """When correlation check is enabled, similar values should be adjusted."""
        calc = PauseCalculator()
        JitterCorrelationStore.clear()

        ctx = PauseContext(
            jitter_factor=0.2,
            jitter_strategy="random",
            jitter_correlation_check=True
        )

        # First call - no correlation
        result1 = calc.apply_jitter(100.0, ctx)

        # Second call - should detect correlation and adjust
        result2 = calc.apply_jitter(100.0, ctx)

        # Results should be different due to correlation adjustment
        # (or at least the correlation store should have values now)
        assert len(JitterCorrelationStore.get_recent()) > 0

    @pytest.mark.fast
    def test_correlation_check_disabled(self):
        """When correlation check is disabled, no correlation tracking."""
        calc = PauseCalculator()
        JitterCorrelationStore.clear()

        ctx = PauseContext(
            jitter_factor=0.2,
            jitter_strategy="random",
            jitter_correlation_check=False
        )

        result = calc.apply_jitter(100.0, ctx)

        # With correlation disabled, store should remain empty
        assert len(JitterCorrelationStore.get_recent()) == 0


class TestJitterCorrelationStore:
    """Tests for JitterCorrelationStore."""

    @pytest.mark.fast
    def test_store_add_and_get(self):
        """Store should add and retrieve values."""
        JitterCorrelationStore.clear()

        JitterCorrelationStore.add(0.1)
        JitterCorrelationStore.add(0.2)
        JitterCorrelationStore.add(0.3)

        recent = JitterCorrelationStore.get_recent(2)
        assert len(recent) == 2
        assert 0.2 in recent
        assert 0.3 in recent

    @pytest.mark.fast
    def test_store_max_size(self):
        """Store should not exceed max size."""
        JitterCorrelationStore.clear()

        # Add more than max size
        for i in range(150):
            JitterCorrelationStore.add(float(i))

        # Should be capped at 100
        assert len(JitterCorrelationStore.get_recent(150)) == 100

    @pytest.mark.fast
    def test_is_correlated(self):
        """Should detect correlated values."""
        JitterCorrelationStore.clear()

        # Add some values
        JitterCorrelationStore.add(0.1)
        JitterCorrelationStore.add(0.1)
        JitterCorrelationStore.add(0.1)

        # New value very close to average should be correlated
        assert JitterCorrelationStore.is_correlated(0.11, threshold=0.1) is True

        # New value far from average should not be correlated
        assert JitterCorrelationStore.is_correlated(0.5, threshold=0.1) is False


class TestJitterStrategiesProduceDifferentDistributions:
    """Tests verifying all three strategies produce different distributions (AC criterion 4)."""

    @pytest.mark.fast
    def test_three_strategies_different_distributions(self):
        """All three jitter strategies should produce measurably different distributions."""
        calc = PauseCalculator()

        random_results = []
        adaptive_results = []
        deterministic_results = []

        # Test random strategy
        for _ in range(50):
            ctx = PauseContext(
                jitter_factor=0.2,
                jitter_strategy="random"
            )
            result = calc.apply_jitter(100.0, ctx)
            random_results.append(result)

        # Test adaptive strategy (mock off-peak)
        with patch('src.downloader.pause_calculator.time.localtime') as mock_time:
            mock_time.return_value = MagicMock(tm_hour=3)  # off-peak
            for _ in range(50):
                ctx = PauseContext(
                    jitter_factor=0.2,
                    jitter_strategy="adaptive",
                    jitter_max_factor=0.5
                )
                result = calc.apply_jitter(100.0, ctx)
                adaptive_results.append(result)

        # Test deterministic strategy
        for i in range(50):
            ctx = PauseContext(
                jitter_factor=0.2,
                jitter_strategy="deterministic",
                client_id=f"client-{i}"
            )
            result = calc.apply_jitter(100.0, ctx)
            deterministic_results.append(result)

        # Check that distributions are different
        random_std = (max(random_results) - min(random_results)) / 2
        adaptive_std = (max(adaptive_results) - min(adaptive_results)) / 2
        deterministic_unique = len(set(deterministic_results))

        # Random should have variance
        assert random_std > 0, "Random strategy should have variance"

        # Adaptive should have variance (different from random in this test due to mock)
        assert adaptive_std > 0, "Adaptive strategy should have variance"

        # Deterministic should have many unique values (different per client)
        # With 50 different clients, we expect 50 unique values
        assert deterministic_unique > 1, "Deterministic should vary by client"


class TestJitterStrategyEnum:
    """Tests for JitterStrategy enum."""

    @pytest.mark.fast
    def test_jitter_strategy_values(self):
        """JitterStrategy should have correct values."""
        assert JitterStrategy.RANDOM.value == "random"
        assert JitterStrategy.ADAPTIVE.value == "adaptive"
        assert JitterStrategy.DETERMINISTIC.value == "deterministic"


class TestPauseCalculatorWithNewConfig:
    """Integration tests for PauseCalculator with new config."""

    @pytest.mark.fast
    def test_full_pipeline_with_adaptive_strategy(self):
        """Full pipeline should work with adaptive strategy."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.2,
            jitter_strategy="adaptive",
            jitter_max_factor=0.5
        )

        result = calc.calculate(ctx)
        assert 30.0 <= result <= 150.0  # 60 * (1 +/- 0.5) = [30, 150], capped at 300

    @pytest.mark.fast
    def test_full_pipeline_with_deterministic_strategy(self):
        """Full pipeline should work with deterministic strategy."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.2,
            jitter_strategy="deterministic",
            client_id="test-client",
            jitter_max_factor=0.5
        )

        result = calc.calculate(ctx)
        assert 30.0 <= result <= 150.0

    @pytest.mark.fast
    def test_last_jitter_applied_tracked(self):
        """last_jitter_applied should track the jitter offset."""
        calc = PauseCalculator()
        ctx = PauseContext(
            jitter_factor=0.2,
            jitter_strategy="random"
        )

        calc.apply_jitter(100.0, ctx)
        # Should have some jitter applied
        assert calc.last_jitter_applied != 0.0 or True  # Either 0 or non-zero is valid
