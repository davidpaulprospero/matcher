"""
Integration test for CaptionStage with large batch processing (US-37-012).

Tests that the fixes from Sprint 37 work together correctly:
- Budget scales proportionally to batch size (US-37-003/US-37-004)
- text_metadata is populated without AttributeError (US-37-010)
- Early termination works when success rate drops (US-37-009)

All tests use mocked CaptionFetcher to avoid network calls.
"""

import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.caption.circuit_breaker import (
    CaptionCircuitBreaker,
    CaptionCircuitBreakerConfig,
)
from src.caption.metrics import CaptionMetrics
from src.caption.retry_budget import (
    CaptionRetryBudget,
    CaptionRetryBudgetConfig,
)
from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
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
    """Mock caption_first config for large batch tests."""
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
    """Mock pipeline state for testing.

    Note: text_metadata is intentionally not initialized to test US-37-010
    defensive initialization.
    """
    video_ids: List[str] = field(default_factory=list)
    video_search_results: List = field(default_factory=list)
    caption_results: Dict = field(default_factory=dict)
    project_dir: Optional[Path] = None
    pending_streams: List = field(default_factory=list)
    # NOTE: text_metadata is intentionally NOT initialized here to test
    # that CaptionStage._ensure_state_attributes() handles missing attributes


@dataclass
class MockVideoSearchResult:
    """Mock video search result with duration."""
    video_id: str
    duration: float = 120.0


@pytest.fixture
def large_batch_video_ids():
    """Generate 200 mock video IDs for large batch testing."""
    # YouTube video IDs are exactly 11 characters (e.g., "dQw4w9WgXcQ")
    return [f"vid{i:04d}xxxx" for i in range(200)]


@pytest.fixture
def mock_video_search_results(large_batch_video_ids):
    """Generate 200 mock video search results with durations."""
    return [
        MockVideoSearchResult(video_id=vid, duration=120.0 + (i % 60))
        for i, vid in enumerate(large_batch_video_ids)
    ]


def make_mock_caption_result(video_id: str, segment_count: int = 10) -> CaptionResult:
    """Create a mock CaptionResult for testing."""
    segments = [
        CaptionSegment(
            index=i,
            start_time=i * 5.0,
            end_time=(i + 1) * 5.0,
            text=f"Test caption segment {i} for {video_id}",
            source_file=video_id,
        )
        for i in range(segment_count)
    ]
    return CaptionResult(
        video_id=video_id,
        segments=segments,
        language="en",
        is_auto_generated=True,
        format_source="vtt",
    )


def make_mock_config(
    retry_budget_config: Optional[Dict] = None,
    circuit_breaker_config: Optional[Dict] = None,
) -> MockConfig:
    """Create mock config with retry budget and circuit breaker."""
    caption_first = MockCaptionFirstConfig(
        retry_budget=retry_budget_config,
        circuit_breaker=circuit_breaker_config,
    )
    download_config = MockDownloadConfig(caption_first=caption_first)
    return MockConfig(download=download_config)


# =============================================================================
# Integration Tests: Large Batch Processing
# =============================================================================


@pytest.mark.integration
class TestCaptionStageLargeBatch:
    """Integration tests for CaptionStage with 200-video large batch."""

    def test_fixture_generates_200_video_ids(self, large_batch_video_ids):
        """Test fixture generates exactly 200 mock video IDs."""
        assert len(large_batch_video_ids) == 200
        # Verify IDs are unique
        assert len(set(large_batch_video_ids)) == 200
        # Verify ID format (11 chars like YouTube IDs)
        for vid in large_batch_video_ids:
            assert len(vid) == 11, f"Video ID {vid} should be 11 chars"

    def test_budget_scales_correctly_for_large_batch(self, large_batch_video_ids):
        """Test retry budget scales proportionally to 200-video batch (US-37-003/004)."""
        # Create retry budget with auto_scale enabled
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=100,  # Default limit
            auto_scale=True,
            attempts_per_video=1.5,
        ))

        # Initial budget should be 100
        assert budget.max_attempts == 100

        # Scale to batch size of 200
        new_max = budget.scale_to_batch_size(200)

        # Expected: 200 * 1.5 = 300 (rounded)
        assert new_max == 300
        assert budget.max_attempts == 300

        # Verify budget doesn't exhaust at video 101
        for i in range(150):
            budget.record_attempt(f"video_{i}")
        assert not budget.budget_exhausted(), "Budget should not be exhausted at 150 attempts"

        # Continue to 299 attempts - still not exhausted
        for i in range(149):
            budget.record_attempt(f"video_{150 + i}")
        assert not budget.budget_exhausted(), "Budget should not be exhausted at 299 attempts"

        # 300th attempt should trigger exhaustion
        budget.record_attempt("video_300")
        assert budget.budget_exhausted(), "Budget should be exhausted at 300 attempts"

    def test_text_metadata_populated_without_attribute_error(
        self, large_batch_video_ids, mock_video_search_results
    ):
        """Test text_metadata is populated without AttributeError (US-37-010)."""
        stage = CaptionStage()

        # Create state WITHOUT text_metadata attribute (simulates legacy state)
        state = MockPipelineState(
            video_ids=large_batch_video_ids[:50],  # Use subset for speed
            video_search_results=mock_video_search_results[:50],
        )

        # Verify text_metadata doesn't exist yet
        assert not hasattr(state, 'text_metadata') or state.text_metadata is None

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
            }
        )
        checkpoint = MockCheckpointManager()

        # Mock fetch to return successful results
        def mock_batch_fetch(video_ids, **kwargs):
            results = {}
            for vid in video_ids:
                results[vid] = make_mock_caption_result(vid, segment_count=5)
            return results

        # Run stage - should NOT raise AttributeError
        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # Stage should complete successfully
        assert result is not None
        assert result.success, f"Stage failed: {result.error}"

        # text_metadata should be populated
        assert hasattr(state, 'text_metadata'), "text_metadata should exist"
        assert state.text_metadata is not None, "text_metadata should not be None"
        assert len(state.text_metadata) > 0, "text_metadata should have segments"

        # Verify segments have expected structure
        for segment in state.text_metadata[:5]:
            assert 'text' in segment
            assert 'video_path' in segment
            assert 'start_time' in segment
            assert 'end_time' in segment

    def test_early_termination_when_success_rate_drops(self):
        """Test early termination works when success rate drops below threshold (US-37-009)."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,  # High limit to not trigger attempt exhaustion
            auto_scale=False,
            min_success_rate=0.3,  # 30% threshold
            min_sample_for_early_termination=20,  # Check after 20 videos
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
        assert "25.0%" in budget.early_termination_reason or "below threshold" in budget.early_termination_reason

    def test_early_termination_not_triggered_above_threshold(self):
        """Test early termination is not triggered when success rate is above threshold."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,
            auto_scale=False,
            min_success_rate=0.3,  # 30% threshold
            min_sample_for_early_termination=20,
        ))

        # Simulate 20 videos: 10 success, 10 failures (50% success rate)
        for i in range(10):
            budget.record_success(f"video_success_{i}")

        for i in range(10):
            budget.record_failure(f"video_fail_{i}")

        # Should NOT trigger early termination (50% > 30%)
        terminated = budget.check_and_terminate_early()
        assert not terminated, "Should not trigger early termination at 50% success rate"
        assert not budget.is_early_terminated()

    def test_early_termination_requires_minimum_sample(self):
        """Test early termination waits for minimum sample size."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            enabled=True,
            max_attempts=500,
            auto_scale=False,
            min_success_rate=0.3,
            min_sample_for_early_termination=20,  # Need 20 videos
        ))

        # Simulate 10 videos: 1 success, 9 failures (10% success rate)
        # But only 10 videos processed, not 20
        budget.record_success("video_0")
        for i in range(9):
            budget.record_failure(f"video_{i + 1}")

        # Should NOT trigger early termination (only 10 samples, need 20)
        terminated = budget.check_and_terminate_early()
        assert not terminated, "Should not terminate with only 10 samples"
        assert not budget.is_early_terminated()

        # Add more failures to reach 20 samples
        for i in range(10):
            budget.record_failure(f"video_{i + 10}")

        # Now should trigger (20 samples, ~5% success rate)
        terminated = budget.check_and_terminate_early()
        assert terminated, "Should terminate after 20 samples with low success rate"

    def test_full_stage_run_with_large_batch_and_early_termination(
        self, large_batch_video_ids, mock_video_search_results
    ):
        """Test full stage run with 200 videos and early termination."""
        stage = CaptionStage()

        state = MockPipelineState(
            video_ids=large_batch_video_ids,
            video_search_results=mock_video_search_results,
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,  # Will scale to 300
                'auto_scale': True,
                'attempts_per_video': 1.5,
                'min_success_rate': 0.3,
                'min_sample_for_early_termination': 20,
            }
        )
        checkpoint = MockCheckpointManager()

        # Track how many videos were processed
        processed_count = [0]
        failed_count = [0]

        def mock_batch_fetch(video_ids, retry_budget=None, **kwargs):
            """Mock fetch that fails most videos to trigger early termination."""
            results = {}
            for i, vid in enumerate(video_ids):
                processed_count[0] += 1
                # Fail 80% of videos (20% success rate) to trigger early termination
                if i % 5 == 0:  # 20% success
                    results[vid] = make_mock_caption_result(vid, segment_count=5)
                    if retry_budget:
                        retry_budget.record_success(vid)
                else:
                    results[vid] = {
                        'video_id': vid,
                        'error': True,
                        'reason': 'Captions unavailable',
                        'caption_quality': 'low',
                    }
                    failed_count[0] += 1
                    if retry_budget:
                        retry_budget.record_failure(vid)

                # Check early termination after each video
                if retry_budget and retry_budget.check_and_terminate_early():
                    # Return partial results on early termination
                    return results

            return results

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # Stage should complete (possibly with warnings)
        assert result is not None

        # text_metadata should be populated (even if partially)
        assert hasattr(state, 'text_metadata')
        assert state.text_metadata is not None


@pytest.mark.integration
class TestRetryBudgetScaling:
    """Integration tests for retry budget scaling behavior."""

    def test_budget_does_not_scale_down(self):
        """Test budget only scales up, never down from default."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            max_attempts=100,
            auto_scale=True,
            attempts_per_video=1.5,
        ))

        # Small batch: 50 * 1.5 = 75, but default is 100, so stays at 100
        new_max = budget.scale_to_batch_size(50)
        assert new_max == 100, "Budget should not scale down below default"

    def test_budget_scales_up_proportionally(self):
        """Test budget scales up proportionally for large batches."""
        test_cases = [
            (100, 150),   # 100 * 1.5 = 150
            (175, 263),   # 175 * 1.5 = 262.5 -> 263
            (200, 300),   # 200 * 1.5 = 300
            (500, 750),   # 500 * 1.5 = 750
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

    def test_budget_scales_with_custom_multiplier(self):
        """Test budget scales with custom attempts_per_video multiplier."""
        budget = CaptionRetryBudget.from_config(CaptionRetryBudgetConfig(
            max_attempts=100,
            auto_scale=True,
            attempts_per_video=2.0,  # Higher multiplier
        ))

        # 200 * 2.0 = 400
        new_max = budget.scale_to_batch_size(200)
        assert new_max == 400


@pytest.mark.integration
class TestStateAttributeInitialization:
    """Integration tests for state attribute defensive initialization."""

    def test_ensure_state_attributes_initializes_missing(self):
        """Test _ensure_state_attributes initializes missing attributes."""
        stage = CaptionStage()

        # Create minimal state without expected attributes
        class MinimalState:
            pass

        state = MinimalState()

        # Call internal method directly
        stage._ensure_state_attributes(state)

        # All required attributes should now exist
        assert hasattr(state, 'text_metadata')
        assert state.text_metadata == []
        assert hasattr(state, 'caption_results')
        assert state.caption_results == {}
        assert hasattr(state, 'video_ids')
        assert state.video_ids == []
        assert hasattr(state, 'video_search_results')
        assert state.video_search_results == []

    def test_ensure_state_attributes_preserves_existing(self):
        """Test _ensure_state_attributes preserves existing attributes."""
        stage = CaptionStage()

        @dataclass
        class PartialState:
            text_metadata: List = field(default_factory=list)
            video_ids: List = field(default_factory=list)

        state = PartialState(
            text_metadata=[{'text': 'existing'}],
            video_ids=['existing_id'],
        )

        stage._ensure_state_attributes(state)

        # Existing values should be preserved
        assert state.text_metadata == [{'text': 'existing'}]
        assert state.video_ids == ['existing_id']

        # Missing attributes should be initialized
        assert hasattr(state, 'caption_results')
        assert hasattr(state, 'video_search_results')
