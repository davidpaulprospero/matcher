"""
Integration tests for CaptionStage retry budget edge cases (US-38-005).

Complements test_caption_stage_large_batch.py with focus on:
- Budget fallback creation when config is missing
- Budget scaling for 175+ video batches
- VPN rotation trigger on rate limit exhaustion
- Early termination on low success rate

All tests use mocked CaptionFetcher to avoid network calls.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.caption.enums import CaptionErrorCategory
from src.caption.retry_budget import (
    CaptionRetryBudget,
    CaptionRetryBudgetConfig,
)
from src.stages.caption_stage import CaptionStage


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class MockConfig:
    """Mock config for testing CaptionStage."""
    download: Any = None


@dataclass
class MockDownloadConfig:
    """Mock download config with caption_first settings."""
    caption_first: Any = None
    impersonation: Any = None
    extractor_args: Any = None


@dataclass
class MockCaptionFirstConfig:
    """Mock caption_first config for budget tests."""
    enabled: bool = True
    preferred_language: str = "en"
    prefer_human_captions: bool = True
    timeout: int = 30
    max_parallel_fetches: int = 4
    min_coverage_threshold: float = 0.5
    pre_check_availability: bool = False
    skip_live_streams: bool = False
    fallback_to_transcription: bool = True
    checkpoint_save_interval: int = 10
    use_global_coordinator: bool = False
    circuit_breaker: Optional[Dict] = None
    retry_budget: Optional[Dict] = None
    adaptive_format_order: bool = False


@dataclass
class MockCheckpointManager:
    """Mock checkpoint manager for testing."""
    stage_data: Dict[str, Any] = field(default_factory=dict)

    def get_stage_data(self, stage_name: str) -> Optional[Dict]:
        return self.stage_data.get(stage_name)

    def should_skip_stage(self, stage_name: str) -> bool:
        return False

    def save_intermediate(self, stage_name: str, data: Dict) -> None:
        self.stage_data[stage_name] = data


@dataclass
class MockPipelineState:
    """Mock pipeline state for testing."""
    video_ids: List[str] = field(default_factory=list)
    video_search_results: List = field(default_factory=list)
    caption_results: Dict = field(default_factory=dict)


@dataclass
class MockVideoSearchResult:
    """Mock video search result with duration."""
    video_id: str
    duration: float = 120.0


@pytest.fixture
def batch_175_video_ids():
    """Generate 175 mock video IDs for budget scaling tests."""
    return [f"vid{i:04d}xxxx" for i in range(175)]


@pytest.fixture
def batch_200_video_ids():
    """Generate 200 mock video IDs for large batch tests."""
    return [f"vid{i:04d}xxxx" for i in range(200)]


@pytest.fixture
def mock_video_search_results_175(batch_175_video_ids):
    """Generate 175 mock video search results with durations."""
    return [
        MockVideoSearchResult(video_id=vid, duration=120.0 + (i % 60))
        for i, vid in enumerate(batch_175_video_ids)
    ]


def make_mock_config(
    retry_budget_config: Optional[Dict] = None,
    circuit_breaker_config: Optional[Dict] = None,
) -> MockConfig:
    """Create mock config with optional retry budget and circuit breaker."""
    caption_first = MockCaptionFirstConfig(
        retry_budget=retry_budget_config,
        circuit_breaker=circuit_breaker_config,
    )
    download_config = MockDownloadConfig(caption_first=caption_first)
    return MockConfig(download=download_config)


def make_mock_config_without_retry_budget() -> MockConfig:
    """Create mock config with no retry_budget setting (tests fallback)."""
    caption_first = MockCaptionFirstConfig(
        retry_budget=None,  # Explicitly None - no config
    )
    download_config = MockDownloadConfig(caption_first=caption_first)
    return MockConfig(download=download_config)


# =============================================================================
# Integration Tests: Retry Budget Fallback
# =============================================================================


@pytest.mark.integration
class TestRetryBudgetFallback:
    """Integration tests for retry budget fallback when config missing."""

    def test_budget_created_with_defaults_when_config_missing(self):
        """Test retry_budget fallback is created when config missing."""
        # Create budget with None config (simulates missing config)
        budget = CaptionRetryBudget.from_config(None)

        # Should have default values
        assert budget is not None
        assert budget.max_attempts == 100  # Default
        assert budget.max_backoff_time == 300.0  # Default
        assert budget.auto_scale is True  # Default
        assert budget.attempts_per_video == 1.5  # Default

    def test_budget_created_from_empty_dict(self):
        """Test retry_budget created from empty dict config."""
        budget = CaptionRetryBudget.from_config({})

        assert budget is not None
        assert budget.max_attempts == 100
        assert budget.auto_scale is True

    def test_budget_created_from_partial_dict(self):
        """Test retry_budget created from partial dict config."""
        config = {'max_attempts': 200}  # Only one field set
        budget = CaptionRetryBudget.from_config(config)

        assert budget.max_attempts == 200
        assert budget.auto_scale is True  # Falls back to default
        assert budget.attempts_per_video == 1.5  # Falls back to default

    def test_budget_created_from_dataclass_config(self):
        """Test retry_budget created from dataclass config."""
        config = CaptionRetryBudgetConfig(
            max_attempts=150,
            auto_scale=False,
            attempts_per_video=2.0,
        )
        budget = CaptionRetryBudget.from_config(config)

        assert budget.max_attempts == 150
        assert budget.auto_scale is False
        assert budget.attempts_per_video == 2.0


# =============================================================================
# Integration Tests: Budget Scaling for 175+ Video Batch
# =============================================================================


@pytest.mark.integration
class TestBudgetScalingLargeBatch:
    """Integration tests for budget scaling with 175+ video batches."""

    def test_budget_scales_for_175_video_batch(self):
        """Test budget scales correctly for 175 video batch."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=100,
            auto_scale=True,
            attempts_per_video=1.5,
        ))

        # Initial budget is 100
        assert budget.max_attempts == 100

        # Scale to 175 videos: 175 * 1.5 = 262.5 -> 263
        new_max = budget.scale_to_batch_size(175)

        assert new_max == 263
        assert budget.max_attempts == 263

    def test_budget_scales_for_176_to_250_video_batches(self):
        """Test budget scales correctly for various batch sizes above 175."""
        test_cases = [
            (176, 264),   # 176 * 1.5 = 264
            (180, 270),   # 180 * 1.5 = 270
            (200, 300),   # 200 * 1.5 = 300
            (250, 375),   # 250 * 1.5 = 375
        ]

        for batch_size, expected_max in test_cases:
            budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
                max_attempts=100,
                auto_scale=True,
                attempts_per_video=1.5,
            ))
            new_max = budget.scale_to_batch_size(batch_size)
            assert new_max == expected_max, (
                f"Batch {batch_size}: expected {expected_max}, got {new_max}"
            )

    def test_budget_does_not_exhaust_at_video_101_for_175_batch(self):
        """Test budget doesn't exhaust at video 101 for 175 video batch."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            max_attempts=100,  # Default
            auto_scale=True,
            attempts_per_video=1.5,
        ))

        # Scale to 175 videos
        budget.scale_to_batch_size(175)
        assert budget.max_attempts == 263

        # Record 101 attempts - should NOT exhaust
        for i in range(101):
            budget.record_attempt(f"video_{i}")

        assert not budget.budget_exhausted(), (
            "Budget should NOT exhaust at 101 attempts for 175-video batch"
        )

        # Continue to 175 - still not exhausted (each video gets 1 attempt)
        for i in range(74):  # 101 + 74 = 175
            budget.record_attempt(f"video_{101 + i}")

        assert not budget.budget_exhausted(), (
            "Budget should NOT exhaust at 175 attempts (1 per video)"
        )

    def test_budget_only_scales_up_never_down(self):
        """Test budget only scales up, never reduces below default."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            max_attempts=100,
            auto_scale=True,
            attempts_per_video=1.5,
        ))

        # Small batch: 50 * 1.5 = 75, but default is 100
        new_max = budget.scale_to_batch_size(50)

        assert new_max == 100, "Budget should not scale down below default"
        assert budget.max_attempts == 100


# =============================================================================
# Integration Tests: VPN Rotation on Rate Limit Exhaustion
# =============================================================================


@pytest.mark.integration
class TestVPNRotationOnRateLimitExhaustion:
    """Integration tests for VPN rotation trigger on rate limit exhaustion."""

    def test_vpn_rotation_triggered_when_rate_limit_over_50_percent(self):
        """Test VPN rotation triggers when rate_limit_pct > 50%."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=20,  # Low limit to test exhaustion
            auto_scale=False,
            trigger_vpn_rotation_on_rate_limit=True,
            max_vpn_resets_per_session=2,
        ))

        # Record 20 attempts with >50% rate limit errors
        # 12 rate limits (60%), 8 other errors
        for i in range(12):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", CaptionErrorCategory.RATE_LIMIT)

        for i in range(8):
            budget.record_attempt(f"video_{12 + i}")
            budget.record_failure(f"video_{12 + i}", CaptionErrorCategory.NETWORK)

        # Budget should be exhausted
        assert budget.budget_exhausted()

        # Rate limit percentage should be > 50%
        rate_limit_pct = budget.get_rate_limit_error_percentage()
        assert rate_limit_pct == 60.0, f"Expected 60%, got {rate_limit_pct}%"

        # VPN rotation should be triggered
        assert budget.should_trigger_vpn_rotation()

    def test_vpn_rotation_not_triggered_when_rate_limit_under_50_percent(self):
        """Test VPN rotation NOT triggered when rate_limit_pct < 50%."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=20,
            auto_scale=False,
            trigger_vpn_rotation_on_rate_limit=True,
        ))

        # Record 20 attempts with <50% rate limit errors
        # 8 rate limits (40%), 12 other errors
        for i in range(8):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", CaptionErrorCategory.RATE_LIMIT)

        for i in range(12):
            budget.record_attempt(f"video_{8 + i}")
            budget.record_failure(f"video_{8 + i}", CaptionErrorCategory.NETWORK)

        # Budget exhausted
        assert budget.budget_exhausted()

        # Rate limit percentage should be < 50%
        rate_limit_pct = budget.get_rate_limit_error_percentage()
        assert rate_limit_pct == 40.0

        # VPN rotation should NOT be triggered
        assert not budget.should_trigger_vpn_rotation()

    def test_vpn_rotation_not_triggered_when_budget_not_exhausted(self):
        """Test VPN rotation NOT triggered when budget not exhausted."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=100,  # High limit
            auto_scale=False,
            trigger_vpn_rotation_on_rate_limit=True,
        ))

        # Record some rate limit errors but not enough to exhaust budget
        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", CaptionErrorCategory.RATE_LIMIT)

        # Budget NOT exhausted
        assert not budget.budget_exhausted()

        # VPN rotation should NOT trigger (budget not exhausted)
        assert not budget.should_trigger_vpn_rotation()

    def test_vpn_rotation_respects_max_resets_per_session(self):
        """Test VPN rotation stops after max_vpn_resets_per_session reached."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=10,
            auto_scale=False,
            trigger_vpn_rotation_on_rate_limit=True,
            max_vpn_resets_per_session=2,
        ))

        # First round: exhaust budget with rate limits
        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", CaptionErrorCategory.RATE_LIMIT)

        assert budget.should_trigger_vpn_rotation()
        budget.record_vpn_reset()  # First reset

        # Reset budget for second round
        budget.reset(preserve_vpn_count=True)

        # Second round: exhaust again
        for i in range(10):
            budget.record_attempt(f"video2_{i}")
            budget.record_failure(f"video2_{i}", CaptionErrorCategory.RATE_LIMIT)

        assert budget.should_trigger_vpn_rotation()
        budget.record_vpn_reset()  # Second reset

        # Reset for third round
        budget.reset(preserve_vpn_count=True)

        # Third round: exhaust again
        for i in range(10):
            budget.record_attempt(f"video3_{i}")
            budget.record_failure(f"video3_{i}", CaptionErrorCategory.RATE_LIMIT)

        # VPN rotation should NOT trigger (max resets reached)
        assert not budget.should_trigger_vpn_rotation()
        assert budget.vpn_resets_used == 2
        assert not budget.can_vpn_reset()

    def test_vpn_reset_resets_counters_but_preserves_vpn_count(self):
        """Test budget.reset() resets counters but preserves VPN reset count."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=10,
            trigger_vpn_rotation_on_rate_limit=True,
            max_vpn_resets_per_session=2,
        ))

        # Record some attempts
        for i in range(10):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", CaptionErrorCategory.RATE_LIMIT)

        budget.record_vpn_reset()
        assert budget.vpn_resets_used == 1

        # Reset with preserve_vpn_count=True (default)
        budget.reset()

        # Counters should be reset
        assert budget.attempts == 0
        assert budget.failures == 0
        assert budget.successes == 0

        # VPN count should be preserved
        assert budget.vpn_resets_used == 1


# =============================================================================
# Integration Tests: Early Termination on Low Success Rate
# =============================================================================


@pytest.mark.integration
class TestEarlyTerminationLowSuccessRate:
    """Integration tests for early termination when success_rate < 30%."""

    def test_early_termination_at_25_percent_success_rate(self):
        """Test early termination when success rate is 25% (< 30%)."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,  # High limit to not trigger attempt exhaustion
            auto_scale=False,
            min_success_rate=0.3,  # 30% threshold
            min_sample_for_early_termination=20,
        ))

        # Simulate 20 videos: 5 success, 15 failures (25% success rate)
        for i in range(5):
            budget.record_success(f"video_success_{i}")

        for i in range(15):
            budget.record_failure(f"video_fail_{i}")

        # Should trigger early termination
        terminated = budget.check_and_terminate_early()
        assert terminated, "Should trigger early termination at 25% success rate"
        assert budget.is_early_terminated()
        assert budget.early_termination_reason is not None
        assert "25.0%" in budget.early_termination_reason

    def test_no_early_termination_at_35_percent_success_rate(self):
        """Test NO early termination when success rate is 35% (> 30%)."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,
            auto_scale=False,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,
        ))

        # Simulate 20 videos: 7 success, 13 failures (35% success rate)
        for i in range(7):
            budget.record_success(f"video_success_{i}")

        for i in range(13):
            budget.record_failure(f"video_fail_{i}")

        # Should NOT trigger early termination
        terminated = budget.check_and_terminate_early()
        assert not terminated, "Should NOT trigger early termination at 35% success rate"
        assert not budget.is_early_terminated()

    def test_early_termination_waits_for_minimum_sample(self):
        """Test early termination waits for min_sample_for_early_termination."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,
            auto_scale=False,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,  # Need 20 videos
        ))

        # Simulate 15 videos: 1 success, 14 failures (very low rate)
        budget.record_success("video_0")
        for i in range(14):
            budget.record_failure(f"video_{i + 1}")

        # Should NOT trigger (only 15 samples, need 20)
        terminated = budget.check_and_terminate_early()
        assert not terminated, "Should NOT terminate with only 15 samples"
        assert not budget.is_early_terminated()

        # Add 5 more failures to reach 20 samples
        for i in range(5):
            budget.record_failure(f"video_{15 + i}")

        # Now should trigger (20 samples, 5% success rate)
        terminated = budget.check_and_terminate_early()
        assert terminated, "Should terminate after 20 samples with 5% success rate"

    def test_early_termination_only_triggers_once(self):
        """Test check_and_terminate_early only returns True once."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,
        ))

        # Trigger early termination
        for i in range(5):
            budget.record_success(f"video_success_{i}")
        for i in range(15):
            budget.record_failure(f"video_fail_{i}")

        # First call should return True
        first_call = budget.check_and_terminate_early()
        assert first_call is True

        # Second call should return False (already terminated)
        second_call = budget.check_and_terminate_early()
        assert second_call is False

        # But is_early_terminated should still be True
        assert budget.is_early_terminated()

    def test_success_rate_calculation_with_zero_processed(self):
        """Test success rate is 1.0 when no videos processed."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,
        ))

        # No videos processed
        assert budget.get_success_rate() == 1.0  # Default when no data

    def test_success_rate_boundary_at_30_percent(self):
        """Test success rate boundary at exactly 30%."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,
        ))

        # Exactly 30%: 6 success, 14 failures
        for i in range(6):
            budget.record_success(f"video_success_{i}")
        for i in range(14):
            budget.record_failure(f"video_fail_{i}")

        success_rate = budget.get_success_rate()
        assert success_rate == 0.3  # Exactly 30%

        # At exactly 30%, should_terminate_early returns False (need < threshold)
        assert not budget.should_terminate_early()


# =============================================================================
# Integration Tests: Combined Scenarios
# =============================================================================


@pytest.mark.integration
class TestCombinedBudgetScenarios:
    """Integration tests combining multiple budget features."""

    def test_scaled_budget_with_early_termination(self):
        """Test scaled budget combined with early termination."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=100,
            auto_scale=True,
            attempts_per_video=1.5,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,
        ))

        # Scale to 175 videos
        budget.scale_to_batch_size(175)
        assert budget.max_attempts == 263

        # Process 20 videos with low success rate (15% success)
        for i in range(3):
            budget.record_success(f"video_success_{i}")
        for i in range(17):
            budget.record_failure(f"video_fail_{i}")

        # Early termination should trigger before budget exhaustion
        assert budget.check_and_terminate_early()
        assert not budget.budget_exhausted()  # Only 20 attempts, budget is 263

    def test_budget_exhaustion_before_early_termination_threshold(self):
        """Test budget exhausts before reaching early termination sample size."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=15,  # Very low limit
            auto_scale=False,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,  # Need 20 samples
        ))

        # Process 15 videos (all failures)
        for i in range(15):
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}")

        # Budget exhausted
        assert budget.budget_exhausted()

        # Early termination NOT triggered (only 15 samples, need 20)
        assert not budget.is_early_terminated()

    def test_vpn_rotation_after_scaled_budget_exhaustion(self):
        """Test VPN rotation triggers after scaled budget exhaustion."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=50,
            auto_scale=True,
            attempts_per_video=1.0,  # 1:1 scaling
            trigger_vpn_rotation_on_rate_limit=True,
            max_vpn_resets_per_session=2,
        ))

        # Scale to 60 videos (60 * 1.0 = 60 attempts)
        budget.scale_to_batch_size(60)
        assert budget.max_attempts == 60

        # Exhaust budget with >50% rate limit errors
        for i in range(40):  # 67% rate limits
            budget.record_attempt(f"video_{i}")
            budget.record_failure(f"video_{i}", CaptionErrorCategory.RATE_LIMIT)

        for i in range(20):  # 33% other errors
            budget.record_attempt(f"video_{40 + i}")
            budget.record_failure(f"video_{40 + i}", CaptionErrorCategory.NETWORK)

        # Budget exhausted with high rate limit percentage
        assert budget.budget_exhausted()
        assert budget.get_rate_limit_error_percentage() > 50

        # VPN rotation should trigger
        assert budget.should_trigger_vpn_rotation()


# =============================================================================
# Integration Tests: Budget Validation Warnings (US-38-010)
# =============================================================================


@pytest.mark.integration
class TestBudgetValidationWarnings:
    """Integration tests for _validate_budget_for_batch warnings."""

    def test_warning_when_max_attempts_less_than_batch_size(self, caplog):
        """Test WARNING logged when max_attempts < batch_size."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        # Create budget with max_attempts=50, batch_size=100
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=50,
            auto_scale=False,  # Disable auto-scale
        ))

        # Call validation with batch larger than max_attempts
        stage._validate_budget_for_batch(100, budget)

        # Check that WARNING was logged
        assert any(
            "max_attempts=50" in record.message and "batch_size=100" in record.message
            for record in caplog.records
        ), "Expected WARNING about max_attempts < batch_size"

    def test_warning_when_auto_scale_false_and_large_batch(self, caplog):
        """Test WARNING logged when auto_scale=False and batch_size > 100."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        # Create budget with auto_scale=False and high max_attempts
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=200,  # Enough for batch, but auto_scale is False
            auto_scale=False,
        ))

        # Call validation with batch > 100
        stage._validate_budget_for_batch(150, budget)

        # Check that WARNING was logged about large batch without auto_scale
        assert any(
            "auto_scale" in record.message.lower() and "150" in record.message
            for record in caplog.records
        ), "Expected WARNING about large batch without auto_scale"

    def test_no_warning_when_auto_scale_enabled(self, caplog):
        """Test NO warning when auto_scale=True even with large batch."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        # Create budget with auto_scale=True
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=50,  # Small, but auto_scale will fix it
            auto_scale=True,
        ))

        # Call validation
        stage._validate_budget_for_batch(200, budget)

        # WARNING about "without auto_scale" should NOT appear
        # (but there may be a warning about max_attempts < batch_size before scaling)
        assert not any(
            "without auto_scale" in record.message
            for record in caplog.records
        ), "Should NOT warn about auto_scale when it's enabled"

    def test_no_warning_when_budget_is_sufficient(self, caplog):
        """Test NO warning when max_attempts >= batch_size and auto_scale=True."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        # Create budget with sufficient max_attempts
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=200,
            auto_scale=True,
        ))

        # Call validation with batch smaller than max_attempts
        stage._validate_budget_for_batch(50, budget)

        # No warnings should be logged
        assert len(caplog.records) == 0, (
            f"Expected no warnings, got: {[r.message for r in caplog.records]}"
        )

    def test_no_warning_when_budget_is_none(self, caplog):
        """Test NO warning when retry_budget is None."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        # Call validation with None budget
        stage._validate_budget_for_batch(100, None)

        # No warnings should be logged
        assert len(caplog.records) == 0

    def test_no_warning_when_batch_size_is_zero(self, caplog):
        """Test NO warning when batch_size is 0."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=50,
            auto_scale=False,
        ))

        # Call validation with batch_size=0
        stage._validate_budget_for_batch(0, budget)

        # No warnings should be logged
        assert len(caplog.records) == 0

    def test_both_warnings_when_both_conditions_met(self, caplog):
        """Test both warnings logged when both conditions are met."""
        import logging
        caplog.set_level(logging.WARNING)

        stage = CaptionStage()

        # Create budget with max_attempts=50, auto_scale=False
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=50,
            auto_scale=False,
        ))

        # batch_size=150: > max_attempts AND > 100
        stage._validate_budget_for_batch(150, budget)

        # Should have two warnings
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_messages) == 2, (
            f"Expected 2 warnings, got {len(warning_messages)}: {warning_messages}"
        )
