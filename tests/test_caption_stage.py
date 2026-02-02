"""Integration tests for CaptionStage checkpoint restore with budget scaling (US-43-004).

Tests the end-to-end flow from checkpoint restore through retry budget scaling.
Ensures that when a checkpoint is restored with a batch_size that requires more
attempts than the default max_attempts, the budget is scaled up correctly.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.caption.retry_budget import CaptionRetryBudget
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
    text_metadata: List = field(default_factory=list)


@dataclass
class MockVideoSearchResult:
    """Mock video search result with duration."""
    video_id: str
    duration: float = 120.0


def make_mock_config(retry_budget_config: Optional[Dict] = None) -> MockConfig:
    """Create mock config with optional retry budget settings."""
    caption_first = MockCaptionFirstConfig(
        retry_budget=retry_budget_config,
    )
    download_config = MockDownloadConfig(caption_first=caption_first)
    return MockConfig(download=download_config)


# =============================================================================
# US-43-004: Integration Test for Caption Stage Checkpoint Restore with Scaling
# =============================================================================


class TestCaptionStageCheckpointRestoreScaling:
    """Integration tests for CaptionStage checkpoint restore with budget scaling.

    US-43-004: Verifies end-to-end flow from checkpoint restore through scaling.
    """

    @pytest.mark.integration
    def test_caption_stage_scales_budget_on_checkpoint_restore(self):
        """Test CaptionStage scales budget after checkpoint restore.

        AC1: Create test in tests/test_caption_stage.py
        AC2: Mock checkpoint with batch_size=175, max_attempts=100, attempts=0
        AC3: Call caption stage run() with 175 video IDs
        AC4: Verify retry_budget.max_attempts >= 350 (175 * 2.0)

        This is a Tier 2 integration test verifying the end-to-end flow:
        1. Checkpoint is restored with batch_size=175, max_attempts=100
        2. CaptionStage.run() is called with 175 video IDs
        3. retry_budget.ensure_scaled() is called
        4. max_attempts is scaled up to 175 * 2.0 = 350
        """
        # AC2: Create 175 video IDs
        video_ids = [f"vid{i:04d}xxxx" for i in range(175)]

        # Create state with 175 videos
        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        # AC2: Config with retry budget settings (attempts_per_video=2.0)
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,  # Will be restored from checkpoint as 100
                'auto_scale': True,
                'attempts_per_video': 2.0,  # 175 * 2.0 = 350 required
            }
        )

        # AC2: Mock checkpoint with batch_size=175, max_attempts=100, attempts=0
        # This simulates the scenario where:
        # - Checkpoint was saved with batch_size=175 (from previous run)
        # - max_attempts was restored to config default (100) on resume
        # - No attempts have been used yet (attempts=0)
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},  # Empty - no prior results
                    'retry_budget': {
                        'attempts': 0,          # No attempts used
                        'failures': 0,
                        'successes': 0,
                        'backoff_time_spent': 0.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,    # Config default (too low!)
                        'max_backoff_time': 300.0,
                        'batch_size': 175,      # Was set from previous run
                    }
                }
            }
        )

        stage = CaptionStage()

        # Track the final budget state when fetch is called
        captured_budget = {}

        def mock_batch_fetch(video_ids, **kwargs):
            """Mock fetch that captures the budget state."""
            retry_budget = kwargs.get('retry_budget')
            if retry_budget:
                # AC4: Capture the budget's max_attempts at fetch time
                captured_budget['max_attempts'] = retry_budget.max_attempts
                captured_budget['batch_size'] = retry_budget.batch_size
                captured_budget['attempts'] = retry_budget.attempts
            return {vid: {} for vid in video_ids}

        # AC3: Call caption stage run() with 175 video IDs
        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch

            stage.run(state, config, checkpoint)

        # AC4: Verify retry_budget.max_attempts >= 350 (175 * 2.0)
        assert 'max_attempts' in captured_budget, (
            "retry_budget should have been passed to fetch_captions_batch"
        )
        assert captured_budget['max_attempts'] >= 350, (
            f"Expected max_attempts >= 350 (175 * 2.0), got {captured_budget['max_attempts']}"
        )

        # Also verify batch_size is correct
        assert captured_budget['batch_size'] == 175, (
            f"Expected batch_size=175, got {captured_budget['batch_size']}"
        )

        # Verify attempts were preserved (should still be 0)
        assert captured_budget['attempts'] == 0, (
            f"Expected attempts=0 (no prior work), got {captured_budget['attempts']}"
        )

    @pytest.mark.integration
    def test_caption_stage_preserves_attempts_after_checkpoint_restore_scaling(self):
        """Test that attempts counter is preserved when scaling after checkpoint restore.

        This tests the scenario where a checkpoint had some attempts used, and the
        budget needs to scale up. The attempts counter should be preserved.

        Note: Since 25 videos are already in checkpoint, only 150 will be fetched.
        Scaling is based on ids_to_fetch (150), not total batch (175).
        150 * 2.0 = 300 max_attempts.
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(175)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        # Checkpoint with some attempts already used
        # 25 videos already processed -> only 150 will be fetched -> scaled to 300
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {
                        f'vid{i:04d}xxxx': {'video_id': f'vid{i:04d}xxxx', 'segments': []}
                        for i in range(25)  # 25 videos already processed
                    },
                    'retry_budget': {
                        'attempts': 30,         # 30 attempts used
                        'failures': 5,          # 5 failures
                        'successes': 25,        # 25 successes
                        'backoff_time_spent': 10.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,    # Config default (needs scaling)
                        'max_backoff_time': 300.0,
                        'batch_size': 175,      # Original batch size
                    }
                }
            }
        )

        stage = CaptionStage()
        captured_budget = {}

        def mock_batch_fetch(video_ids, **kwargs):
            retry_budget = kwargs.get('retry_budget')
            if retry_budget:
                captured_budget['max_attempts'] = retry_budget.max_attempts
                captured_budget['batch_size'] = retry_budget.batch_size
                captured_budget['attempts'] = retry_budget.attempts
                captured_budget['failures'] = retry_budget.failures
                captured_budget['successes'] = retry_budget.successes
            return {vid: {} for vid in video_ids}

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch

            stage.run(state, config, checkpoint)

        # Verify budget was scaled for 150 remaining videos: 150 * 2.0 = 300
        assert captured_budget['max_attempts'] >= 300, (
            f"Expected max_attempts >= 300 (150 * 2.0), got {captured_budget['max_attempts']}"
        )

        # Verify attempts counter was preserved (not reset)
        assert captured_budget['attempts'] == 30, (
            f"Expected attempts=30 (preserved), got {captured_budget['attempts']}"
        )
        assert captured_budget['failures'] == 5, (
            f"Expected failures=5 (preserved), got {captured_budget['failures']}"
        )
        assert captured_budget['successes'] == 25, (
            f"Expected successes=25 (preserved), got {captured_budget['successes']}"
        )

    @pytest.mark.integration
    def test_caption_stage_no_scale_when_max_attempts_already_sufficient(self):
        """Test that budget is not scaled when max_attempts is already sufficient.

        If max_attempts from checkpoint is already >= required, no scaling should occur.
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(50)]  # Small batch

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,  # Already sufficient for 50 * 2.0 = 100
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 0,
                        'failures': 0,
                        'successes': 0,
                        'backoff_time_spent': 0.0,
                        'videos_skipped': 0,
                        'max_attempts': 200,  # Already sufficient
                        'max_backoff_time': 300.0,
                        'batch_size': 50,
                    }
                }
            }
        )

        stage = CaptionStage()
        captured_budget = {}

        def mock_batch_fetch(video_ids, **kwargs):
            retry_budget = kwargs.get('retry_budget')
            if retry_budget:
                captured_budget['max_attempts'] = retry_budget.max_attempts
                captured_budget['batch_size'] = retry_budget.batch_size
            return {vid: {} for vid in video_ids}

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch

            stage.run(state, config, checkpoint)

        # max_attempts should remain at 200 (no scaling needed)
        assert captured_budget['max_attempts'] == 200, (
            f"Expected max_attempts=200 (unchanged), got {captured_budget['max_attempts']}"
        )


# =============================================================================
# US-43-010: Fail-fast Check for Budget Exhaustion Before Batch Start
# =============================================================================


class TestCaptionStageFailFastBudgetExhaustion:
    """Test fail-fast detection when budget is already exhausted at batch start.

    US-43-010: If checkpoint restores budget with 100/100 attempts already used,
    the batch should immediately detect and report this rather than trying to
    process and skip all videos.
    """

    @pytest.mark.integration
    def test_fail_fast_when_budget_exhausted_at_start(self):
        """Test that CaptionStage returns early when budget is already exhausted.

        AC1: Add check in CaptionStage.run() after ensure_scaled() call
        AC2: If budget_exhausted() is already True, log ERROR and skip batch processing
        AC3: Log: '[US-43-010] Budget already exhausted at batch start - check checkpoint restore'
        AC4: Return early with appropriate warnings about skipped videos
        """
        # Create 50 video IDs that would be fetched
        video_ids = [f"vid{i:04d}xxxx" for i in range(50)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        # Config with retry budget enabled
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        # AC1/AC2: Checkpoint with budget ALREADY exhausted (100/100 attempts used)
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},  # No prior results - all attempts failed
                    'retry_budget': {
                        'attempts': 100,        # ALL attempts used!
                        'failures': 100,        # All failed
                        'successes': 0,
                        'backoff_time_spent': 50.0,
                        'videos_skipped': 0,    # No videos skipped yet
                        'max_attempts': 100,    # Limit reached
                        'max_backoff_time': 300.0,
                        'batch_size': 100,
                    }
                }
            }
        )

        stage = CaptionStage()

        # Track whether fetch was called (it should NOT be called)
        fetch_called = {'value': False}

        def mock_batch_fetch(video_ids, **kwargs):
            fetch_called['value'] = True
            return {vid: {} for vid in video_ids}

        # AC3/AC4: Run the stage and check for early exit
        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch

            result = stage.run(state, config, checkpoint)

        # Verify the stage returned early (OK with warnings, not fail)
        assert result.success is True, "Stage should return success (graceful early exit)"

        # Verify fetch was NOT called (fail-fast should prevent batch processing)
        assert fetch_called['value'] is False, (
            "fetch_captions_batch should NOT be called when budget is exhausted at start"
        )

        # AC4: Verify appropriate warnings are present
        assert len(result.warnings) > 0, "Should have warnings about skipped videos"
        warning_text = ' '.join(result.warnings)
        assert 'exhausted' in warning_text.lower() or 'skipped' in warning_text.lower(), (
            f"Warnings should mention exhaustion or skipped videos: {result.warnings}"
        )

        # Verify result data indicates budget exhaustion
        assert result.data.get('budget_exhausted_at_start') is True, (
            "Result should indicate budget_exhausted_at_start=True"
        )
        assert result.data.get('skipped_due_to_budget') == 50, (
            f"Expected 50 videos skipped, got {result.data.get('skipped_due_to_budget')}"
        )

        # Verify all videos are marked as skipped in caption_results
        caption_results = result.data.get('caption_results', {})
        for video_id in video_ids:
            assert video_id in caption_results, f"Video {video_id} should be in results"
            assert caption_results[video_id].get('skipped') is True
            assert caption_results[video_id].get('reason') == 'budget_exhausted_at_start'

    @pytest.mark.integration
    def test_no_fail_fast_when_budget_has_remaining_attempts(self):
        """Test that CaptionStage proceeds normally when budget has remaining attempts.

        Even if some attempts have been used, if budget is not exhausted,
        processing should proceed normally.
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(50)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,  # High limit
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        # Checkpoint with some attempts used but NOT exhausted
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 50,         # 50/200 used - plenty remaining
                        'failures': 10,
                        'successes': 40,
                        'backoff_time_spent': 10.0,
                        'videos_skipped': 0,
                        'max_attempts': 200,
                        'max_backoff_time': 300.0,
                        'batch_size': 50,
                    }
                }
            }
        )

        stage = CaptionStage()
        fetch_called = {'value': False}

        def mock_batch_fetch(video_ids, **kwargs):
            fetch_called['value'] = True
            return {vid: {'video_id': vid, 'segments': []} for vid in video_ids}

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch

            result = stage.run(state, config, checkpoint)

        # fetch SHOULD be called when budget is not exhausted
        assert fetch_called['value'] is True, (
            "fetch_captions_batch should be called when budget has remaining attempts"
        )

        # Should not have budget_exhausted_at_start flag
        assert result.data.get('budget_exhausted_at_start') is not True, (
            "Should not have budget_exhausted_at_start when budget has remaining"
        )

    @pytest.mark.integration
    def test_fail_fast_when_backoff_time_exhausted(self):
        """Test fail-fast when backoff time budget is exhausted.

        Budget can be exhausted via attempts OR backoff time. Test the backoff time case.
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(50)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 500,  # High attempt limit
                'auto_scale': True,
                'attempts_per_video': 2.0,
                'max_backoff_time_seconds': 100.0,  # Low backoff time limit
            }
        )

        # Checkpoint with backoff time exhausted
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 50,         # Not many attempts
                        'failures': 50,
                        'successes': 0,
                        'backoff_time_spent': 100.0,  # Backoff time exhausted!
                        'videos_skipped': 0,
                        'max_attempts': 500,
                        'max_backoff_time': 100.0,    # Same as spent
                        'batch_size': 50,
                    }
                }
            }
        )

        stage = CaptionStage()
        fetch_called = {'value': False}

        def mock_batch_fetch(video_ids, **kwargs):
            fetch_called['value'] = True
            return {vid: {} for vid in video_ids}

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch

            result = stage.run(state, config, checkpoint)

        # Fetch should NOT be called - budget exhausted via backoff time
        assert fetch_called['value'] is False, (
            "fetch_captions_batch should NOT be called when backoff time exhausted"
        )

        # Verify budget exhausted flag
        assert result.data.get('budget_exhausted_at_start') is True
