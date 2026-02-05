"""Unit tests for per-healer timeout and max_attempts configuration (US-64-004)."""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import Healer, HealerResult, HealerAction
from src.agents.strategy import HealingStrategy
from src.agents.orchestrator import HealingOrchestrator


class SlowHealer(Healer):
    """Test healer that sleeps for a configurable duration."""

    name = "slow-healer"
    error_patterns = ["slow"]

    def __init__(self, config, project_dir, sleep_duration: float = 5.0):
        super().__init__(config, project_dir)
        self.sleep_duration = sleep_duration
        self.fix_called = False

    def fix(self, error, state, stage_name):
        self.fix_called = True
        time.sleep(self.sleep_duration)
        return HealerResult.fixed("Fixed after sleeping")


class FastHealer(Healer):
    """Test healer that completes quickly."""

    name = "fast-healer"
    error_patterns = ["fast"]

    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)
        self.fix_called = False

    def fix(self, error, state, stage_name):
        self.fix_called = True
        return HealerResult.fixed("Fixed instantly")


class TestHealingStrategyTimeoutConfig:
    """Test HealingStrategy per-healer timeout configuration."""

    @pytest.mark.fast
    def test_healer_timeouts_has_defaults(self):
        """Test that healer_timeouts has sensible defaults."""
        strategy = HealingStrategy()

        # Should have defaults for common healers
        assert "api-healer" in strategy.healer_timeouts
        assert "disk-healer" in strategy.healer_timeouts
        assert "download-healer" in strategy.healer_timeouts

    @pytest.mark.fast
    def test_api_healer_has_longer_timeout(self):
        """Test that API healer has longer timeout than disk healer."""
        strategy = HealingStrategy()

        api_timeout = strategy.healer_timeouts.get("api-healer", 0)
        disk_timeout = strategy.healer_timeouts.get("disk-healer", 0)

        # API healer should have longer timeout (rate limit waits)
        assert api_timeout > disk_timeout

    @pytest.mark.fast
    def test_disk_healer_has_short_timeout(self):
        """Test that disk healer has short timeout (should fail fast)."""
        strategy = HealingStrategy()

        disk_timeout = strategy.healer_timeouts.get("disk-healer", 0)

        # Disk healer should be quick (< 20s)
        assert disk_timeout <= 20.0

    @pytest.mark.fast
    def test_default_healer_timeout_exists(self):
        """Test that DEFAULT_HEALER_TIMEOUT is defined."""
        strategy = HealingStrategy()

        assert hasattr(strategy, 'DEFAULT_HEALER_TIMEOUT')
        assert strategy.DEFAULT_HEALER_TIMEOUT > 0

    @pytest.mark.fast
    def test_custom_healer_timeouts(self):
        """Test that custom healer_timeouts can be provided."""
        custom_timeouts = {
            "my-healer": 60.0,
            "fast-healer": 5.0,
        }
        strategy = HealingStrategy(healer_timeouts=custom_timeouts)

        assert strategy.healer_timeouts.get("my-healer") == 60.0
        assert strategy.healer_timeouts.get("fast-healer") == 5.0


class TestHealingStrategyMaxAttemptsConfig:
    """Test HealingStrategy per-healer max_attempts configuration."""

    @pytest.mark.fast
    def test_healer_max_attempts_has_defaults(self):
        """Test that healer_max_attempts has sensible defaults."""
        strategy = HealingStrategy()

        # Should have defaults for common healers
        assert "api-healer" in strategy.healer_max_attempts
        assert "disk-healer" in strategy.healer_max_attempts

    @pytest.mark.fast
    def test_api_healer_has_more_attempts(self):
        """Test that API healer has more attempts than disk healer."""
        strategy = HealingStrategy()

        api_attempts = strategy.healer_max_attempts.get("api-healer", 0)
        disk_attempts = strategy.healer_max_attempts.get("disk-healer", 0)

        # API healer should have more attempts (rate limit retries)
        assert api_attempts > disk_attempts

    @pytest.mark.fast
    def test_disk_healer_has_single_attempt(self):
        """Test that disk healer has single attempt (needs user intervention)."""
        strategy = HealingStrategy()

        disk_attempts = strategy.healer_max_attempts.get("disk-healer", 0)

        # Disk healer should fail fast
        assert disk_attempts == 1

    @pytest.mark.fast
    def test_custom_healer_max_attempts(self):
        """Test that custom healer_max_attempts can be provided."""
        custom_attempts = {
            "my-healer": 10,
            "one-shot-healer": 1,
        }
        strategy = HealingStrategy(healer_max_attempts=custom_attempts)

        assert strategy.healer_max_attempts.get("my-healer") == 10
        assert strategy.healer_max_attempts.get("one-shot-healer") == 1


class NoopHealer(Healer):
    """Healer that does nothing, for injection purposes."""

    name = "noop-healer"
    error_patterns = []

    def fix(self, error, state, stage_name):
        return HealerResult.failed("Noop")


class TestOrchestratorHealerTimeout:
    """Test HealingOrchestrator respects per-healer timeout."""

    def _create_orchestrator(self, strategy: HealingStrategy) -> HealingOrchestrator:
        """Create orchestrator with mocked healers to avoid real healer init."""
        config = MagicMock()
        config.healing = None

        # Pass NoopHealer as the only healer to avoid initializing real healers
        orchestrator = HealingOrchestrator(
            config, "/tmp/project", strategy=strategy, healers=[NoopHealer]
        )
        return orchestrator

    @pytest.mark.fast
    def test_get_healer_timeout_from_strategy(self):
        """Test _get_healer_timeout returns value from strategy."""
        strategy = HealingStrategy(healer_timeouts={"api-healer": 120.0})
        orchestrator = self._create_orchestrator(strategy)

        timeout = orchestrator._get_healer_timeout("api-healer")
        assert timeout == 120.0

    @pytest.mark.fast
    def test_get_healer_timeout_default_for_unknown(self):
        """Test _get_healer_timeout returns default for unknown healers."""
        strategy = HealingStrategy()
        orchestrator = self._create_orchestrator(strategy)

        # Unknown healer should get default timeout
        timeout = orchestrator._get_healer_timeout("unknown-healer")
        assert timeout == strategy.DEFAULT_HEALER_TIMEOUT

    @pytest.mark.fast
    def test_get_healer_max_attempts_from_strategy(self):
        """Test _get_healer_max_attempts returns value from strategy."""
        strategy = HealingStrategy(healer_max_attempts={"api-healer": 5})
        orchestrator = self._create_orchestrator(strategy)

        attempts = orchestrator._get_healer_max_attempts("api-healer")
        assert attempts == 5

    @pytest.mark.fast
    def test_get_healer_max_attempts_default_for_unknown(self):
        """Test _get_healer_max_attempts returns global default for unknown healers."""
        strategy = HealingStrategy(max_attempts_per_stage=3)
        orchestrator = self._create_orchestrator(strategy)

        # Unknown healer should get global max_attempts_per_stage
        attempts = orchestrator._get_healer_max_attempts("unknown-healer")
        assert attempts == strategy.max_attempts_per_stage

    @pytest.mark.fast
    def test_try_healer_respects_timeout(self):
        """Test that _try_healer times out slow healers."""
        # Set a very short timeout (0.1 seconds)
        strategy = HealingStrategy(
            healer_timeouts={"slow-healer": 0.1},
        )
        orchestrator = self._create_orchestrator(strategy)

        # Create a slow healer that takes 5 seconds
        config = MagicMock()
        slow_healer = SlowHealer(config, "/tmp/project", sleep_duration=5.0)
        state = MagicMock()
        error = Exception("slow error")

        start = time.time()
        result = orchestrator._try_healer(slow_healer, error, state, "TEST")
        elapsed = time.time() - start

        # Should have timed out, not waited full 5 seconds
        assert not result.success
        assert "timed out" in result.message.lower()
        assert elapsed < 1.0  # Should be close to 0.1s timeout

    @pytest.mark.fast
    def test_try_healer_completes_fast_healer(self):
        """Test that _try_healer allows fast healers to complete."""
        # Set reasonable timeout
        strategy = HealingStrategy(
            healer_timeouts={"fast-healer": 10.0},
        )
        orchestrator = self._create_orchestrator(strategy)

        config = MagicMock()
        fast_healer = FastHealer(config, "/tmp/project")
        state = MagicMock()
        error = Exception("fast error")

        result = orchestrator._try_healer(fast_healer, error, state, "TEST")

        assert result.success
        assert fast_healer.fix_called

    @pytest.mark.fast
    def test_try_healer_uses_default_timeout_for_unconfigured(self):
        """Test that _try_healer uses default timeout for unconfigured healers."""
        # Don't configure timeout for fast-healer
        strategy = HealingStrategy()
        strategy.healer_timeouts = {}  # Clear all healer timeouts
        orchestrator = self._create_orchestrator(strategy)

        config = MagicMock()
        fast_healer = FastHealer(config, "/tmp/project")
        state = MagicMock()
        error = Exception("fast error")

        # Should use DEFAULT_HEALER_TIMEOUT and complete successfully
        result = orchestrator._try_healer(fast_healer, error, state, "TEST")

        assert result.success
        assert fast_healer.fix_called

    @pytest.mark.fast
    def test_try_healer_records_timeout_in_metrics(self):
        """Test that healer timeout is recorded in metrics."""
        strategy = HealingStrategy(
            healer_timeouts={"slow-healer": 0.1},
        )
        orchestrator = self._create_orchestrator(strategy)

        config = MagicMock()
        slow_healer = SlowHealer(config, "/tmp/project", sleep_duration=5.0)
        state = MagicMock()
        error = Exception("slow error")

        orchestrator._try_healer(slow_healer, error, state, "TEST")

        # Check timeout was recorded in errors
        assert any("timed out" in e for e in orchestrator.metrics.errors_encountered)

        # Check failed heal was recorded
        assert orchestrator.metrics.failed_heals == 1


class TestStrategyPresets:
    """Test that strategy presets include healer timeout/attempts config."""

    @pytest.mark.fast
    def test_aggressive_strategy_has_timeouts(self):
        """Test aggressive strategy has healer_timeouts."""
        strategy = HealingStrategy.aggressive()
        assert hasattr(strategy, 'healer_timeouts')
        assert len(strategy.healer_timeouts) > 0

    @pytest.mark.fast
    def test_conservative_strategy_has_timeouts(self):
        """Test conservative strategy has healer_timeouts."""
        strategy = HealingStrategy.conservative()
        assert hasattr(strategy, 'healer_timeouts')
        assert len(strategy.healer_timeouts) > 0

    @pytest.mark.fast
    def test_minimal_strategy_has_timeouts(self):
        """Test minimal strategy has healer_timeouts."""
        strategy = HealingStrategy.minimal()
        assert hasattr(strategy, 'healer_timeouts')
        assert len(strategy.healer_timeouts) > 0

    @pytest.mark.fast
    def test_interactive_strategy_has_timeouts(self):
        """Test interactive strategy has healer_timeouts."""
        strategy = HealingStrategy.interactive()
        assert hasattr(strategy, 'healer_timeouts')
        assert len(strategy.healer_timeouts) > 0
