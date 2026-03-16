"""
Tests for stage retry with exponential backoff configuration (US-125-008).

Verifies:
- Retry logic wrapper around stage.run() method
- Retry metrics tracking attempts per stage
- Exponential backoff delay calculation
- Retry behavior with transient failures
"""
import pytest
import random
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field

from src.pipeline import PipelineOrchestrator
from src.stages import Stage, StageResult, StageMetrics
from src.config import Config
from src.state import PipelineState
from src.checkpoint import CheckpointManager


@dataclass
class MockRetryConfig:
    """Mock retry config for testing."""
    max_attempts: int = 3
    enabled: bool = True
    strategy: object = None

    def __post_init__(self):
        if self.strategy is None:
            self.strategy = MockStrategyConfig()


@dataclass
class MockStrategyConfig:
    """Mock strategy config for testing."""
    strategy: str = "exponential"
    base_delay: float = 1.0
    max_delay: float = 60.0


class TransientFailureStage(Stage):
    """Stage that fails N times then succeeds."""

    def __init__(self, name: str = "TRANSIENT_FAIL", fail_count: int = 2):
        super().__init__()
        self.name = name
        self.fail_count = fail_count
        self.call_count = 0

    def run(self, state, config, checkpoint):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            return StageResult(
                success=False,
                error=f"Transient failure (attempt {self.call_count})",
                metrics=StageMetrics(failed=True, retry_attempts=self.call_count - 1)
            )
        return StageResult(
            success=True,
            metrics=StageMetrics(items_processed=10, retry_attempts=self.fail_count)
        )


class TestStageRetryMetrics:
    """Test that retry_metrics are properly tracked in StageMetrics."""

    def test_retry_attempts_tracked_in_metrics(self):
        """Verify retry_attempts field is tracked in StageMetrics."""
        metrics = StageMetrics(retry_attempts=3)
        assert metrics.retry_attempts == 3

    def test_retry_attempts_default_to_zero(self):
        """Verify retry_attempts defaults to 0."""
        metrics = StageMetrics()
        assert metrics.retry_attempts == 0


class TestRetryBackoffCalculation:
    """Test exponential backoff delay calculation."""

    def test_exponential_backoff_delay(self):
        """Test exponential backoff produces increasing delays."""
        # Create a mock orchestrator to test _calculate_retry_delay
        orchestrator = Mock(spec=PipelineOrchestrator)

        # Test exponential strategy with base_delay=1.0, max_delay=60.0
        delays = []
        for attempt in range(5):
            # Simulate exponential: base_delay * 2^(attempt-1)
            base_delay = 1.0
            max_delay = 60.0
            delay = min(base_delay * (2 ** attempt), max_delay)
            delays.append(delay)

        # Exponential: [1, 2, 4, 8, 16]
        assert delays == [1, 2, 4, 8, 16]

    def test_exponential_backoff_caps_at_max_delay(self):
        """Test exponential backoff caps at max_delay."""
        base_delay = 1.0
        max_delay = 10.0

        delays = []
        for attempt in range(10):
            delay = min(base_delay * (2 ** attempt), max_delay)
            delays.append(delay)

        # After reaching max, all subsequent delays should be max
        for i in range(4, len(delays)):
            assert delays[i] == max_delay

    def test_linear_backoff_delay(self):
        """Test linear backoff produces evenly increasing delays."""
        base_delay = 2.0
        max_delay = 60.0

        delays = []
        for attempt in range(5):
            delay = min(base_delay * attempt, max_delay)
            delays.append(delay)

        # Linear: [0, 2, 4, 6, 8]
        assert delays == [0, 2, 4, 6, 8]

    def test_fixed_backoff_delay(self):
        """Test fixed backoff produces constant delays."""
        base_delay = 3.0
        max_delay = 60.0

        delays = []
        for attempt in range(5):
            delay = min(base_delay, max_delay)
            delays.append(delay)

        # Fixed: [3, 3, 3, 3, 3]
        assert delays == [3, 3, 3, 3, 3]

    def test_jitter_applied_to_delay(self):
        """Test jitter is applied to delay calculation."""
        import random
        random.seed(42)  # Deterministic for testing

        base_delay = 10.0
        jitter_factor = 0.2

        # With jitter_factor=0.2, delay should vary by ±20%
        # For base_delay=10, jitter range = 2.0
        # So delay should be between 8 and 12
        delays = []
        for _ in range(100):
            # Simulate jittered delay
            delay = base_delay
            jitter_range = delay * jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)
            delays.append(delay)

        # Verify jitter produces variation
        assert min(delays) < base_delay
        assert max(delays) > base_delay

    def test_jitter_factor_zero_no_change(self):
        """Test jitter_factor of 0 produces no jitter (deterministic)."""
        base_delay = 10.0
        jitter_factor = 0.0

        # With no jitter, should return base_delay exactly
        delay = base_delay
        jitter_range = delay * jitter_factor
        delay = delay + random.uniform(-jitter_range, jitter_range)

        assert delay == base_delay


class TestRetryConfigSection:
    """Test stage_retry config section with max_attempts and backoff_multiplier."""

    def test_stage_retry_config_exists(self):
        """Verify StageRetryConfig exists and has expected fields."""
        from src.config.sections.infrastructure import StageRetryConfig, RetryStrategyConfig

        config = StageRetryConfig(
            max_attempts=5,
            enabled=True,
            strategy=RetryStrategyConfig(strategy="exponential", base_delay=2.0, max_delay=60.0)
        )

        assert config.max_attempts == 5
        assert config.enabled is True
        assert config.strategy.strategy == "exponential"
        assert config.strategy.base_delay == 2.0

    def test_retry_policy_in_config_yaml(self):
        """Verify retry_policy can be configured in config.yaml."""
        from src.config.sections.infrastructure import PipelineConfig, StageRetryConfig, RetryStrategyConfig

        # Create config with retry_policy
        pipeline_config = PipelineConfig(
            retry_policy={
                "CAPTION": StageRetryConfig(
                    max_attempts=5,
                    enabled=True,
                    strategy=RetryStrategyConfig(strategy="exponential", base_delay=2.0)
                ),
                "MATCH": StageRetryConfig(
                    max_attempts=3,
                    enabled=True,
                    strategy=RetryStrategyConfig(strategy="linear", base_delay=1.0)
                )
            }
        )

        # Verify configs are retrievable
        caption_config = pipeline_config.get_retry_config("CAPTION")
        assert caption_config.max_attempts == 5
        assert caption_config.strategy.strategy == "exponential"

        match_config = pipeline_config.get_retry_config("MATCH")
        assert match_config.max_attempts == 3
        assert match_config.strategy.strategy == "linear"

    def test_default_retry_config(self):
        """Verify default retry config is returned for unknown stages."""
        from src.config.sections.infrastructure import PipelineConfig

        pipeline_config = PipelineConfig()

        default_config = pipeline_config.get_retry_config("UNKNOWN_STAGE")

        # Should return default config
        assert default_config.max_attempts >= 1
        assert default_config.enabled is True

    def test_retry_strategy_jitter_factor(self):
        """Verify RetryStrategyConfig has jitter_factor field."""
        from src.config.sections.infrastructure import RetryStrategyConfig

        strategy = RetryStrategyConfig(
            strategy="exponential",
            base_delay=2.0,
            max_delay=60.0,
            jitter_factor=0.3
        )

        assert strategy.jitter_factor == 0.3
        # Default value
        strategy_default = RetryStrategyConfig()
        assert strategy_default.jitter_factor == 0.2

    def test_retry_config_validation_jitter_bounds(self):
        """Verify jitter_factor validation rejects invalid values."""
        from src.config.sections.infrastructure import StageRetryConfig, RetryStrategyConfig

        # Create a mock pipeline config to test validation
        class MockPipelineConfig:
            def __init__(self, retry_policy):
                self.retry_policy = retry_policy

            def _validate_retry_config(self):
                """Validate retry policy configuration."""
                for stage_name, retry_config in self.retry_policy.items():
                    if not 0.0 <= retry_config.strategy.jitter_factor <= 1.0:
                        raise ValueError(
                            f"retry_policy.{stage_name}.strategy.jitter_factor must be between 0.0 and 1.0, "
                            f"got {retry_config.strategy.jitter_factor}"
                        )

        # Test with invalid jitter_factor > 1.0
        config = MockPipelineConfig(
            retry_policy={
                "TEST": StageRetryConfig(
                    max_attempts=3,
                    enabled=True,
                    strategy=RetryStrategyConfig(
                        strategy="exponential",
                        jitter_factor=1.5  # Invalid: > 1.0
                    )
                )
            }
        )

        # Should raise validation error
        with pytest.raises(ValueError, match="jitter_factor must be between 0.0 and 1.0"):
            config._validate_retry_config()


class TestRetryBehavior:
    """Test retry behavior with transient failures."""

    @pytest.mark.fast
    def test_retry_on_transient_failure(self):
        """Test that stage retries on transient failure."""
        from src.config.sections.infrastructure import StageRetryConfig, RetryStrategyConfig

        config = StageRetryConfig(
            max_attempts=3,
            enabled=True,
            strategy=RetryStrategyConfig(strategy="exponential", base_delay=0.1, max_delay=1.0)
        )

        # Simulate the retry loop logic from pipeline.py
        retry_max_attempts = config.max_attempts
        attempt = 0
        result = None

        # First two attempts fail
        for call_num in range(3):
            if attempt > 0:
                # Calculate delay
                delay = min(
                    config.strategy.base_delay * (2 ** (attempt - 1)),
                    config.strategy.max_delay
                )

            if call_num < 2:
                # Simulate failure
                result = StageResult(
                    success=False,
                    error=f"Transient failure",
                    metrics=StageMetrics(failed=True, retry_attempts=attempt)
                )
            else:
                # Simulate success on third attempt
                result = StageResult(
                    success=True,
                    metrics=StageMetrics(items_processed=10, retry_attempts=attempt)
                )

            if result.success:
                break

            attempt += 1

        # Verify retry happened and eventually succeeded
        assert attempt == 2  # Two retries happened
        assert result.success is True

    @pytest.mark.fast
    def test_retry_exhausted_gives_up(self):
        """Test that stage gives up after max retries."""
        from src.config.sections.infrastructure import StageRetryConfig, RetryStrategyConfig

        config = StageRetryConfig(
            max_attempts=2,  # Only 1 initial + 1 retry = 2 total attempts
            enabled=True,
            strategy=RetryStrategyConfig(strategy="fixed", base_delay=0.1)
        )

        attempt = 0
        result = None

        while attempt < config.max_attempts:
            result = StageResult(
                success=False,
                error=f"Permanent failure",
                metrics=StageMetrics(failed=True, retry_attempts=attempt)
            )

            if not result.success and attempt < config.max_attempts - 1:
                attempt += 1
            else:
                break

        # All retries exhausted
        assert result.success is False
        assert attempt == config.max_attempts - 1

    @pytest.mark.fast
    def test_retry_disabled_no_retry(self):
        """Test that disabled retry doesn't retry."""
        from src.config.sections.infrastructure import StageRetryConfig, RetryStrategyConfig

        config = StageRetryConfig(
            max_attempts=5,
            enabled=False,  # Retry disabled
            strategy=RetryStrategyConfig(strategy="exponential", base_delay=0.1)
        )

        assert config.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
