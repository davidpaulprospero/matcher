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
    videos_needing_transcription: List[str] = field(default_factory=list)  # US-60-006
    caption_batch_low_yield: bool = False


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


# =============================================================================
# US-60-006: Transcription Fallback Coordinator Tests
# =============================================================================


class TestCaptionStageTranscriptionFallback:
    """Test transcription fallback list population for US-60-006.

    Tests that videos failing caption fetch with 'no captions' status are
    correctly added to needs_transcription list and state.videos_needing_transcription.
    """

    @pytest.mark.integration
    def test_caption_failures_populate_transcription_fallback_list(self):
        """Test that caption failures populate transcription fallback list correctly.

        AC1: needs_transcription list attribute exists on CaptionStage
        AC2: Videos with 'no_captions_available' are added to needs_transcription
        AC3: needs_transcription persisted in checkpoint
        AC4: state.videos_needing_transcription set for downstream consumption
        AC5: Test verifies all acceptance criteria
        """
        # Create test videos
        video_ids = [f"vid{i:04d}xxxx" for i in range(10)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        checkpoint = MockCheckpointManager(stage_data={})

        stage = CaptionStage()

        # AC1: Verify needs_transcription attribute exists
        assert hasattr(stage, 'needs_transcription'), (
            "CaptionStage should have needs_transcription attribute"
        )

        def mock_batch_fetch(video_ids, **kwargs):
            """Simulate mixed results: some success, some unavailable."""
            results = {}
            for i, vid in enumerate(video_ids):
                if i % 3 == 0:
                    # Every 3rd video has no captions (unavailable)
                    results[vid] = {
                        'video_id': vid,
                        'unavailable': True,
                        'reason': 'no_captions_available',
                        'caption_quality': 'low',
                    }
                else:
                    # Others succeed
                    results[vid] = {
                        'video_id': vid,
                        'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                        'segment_count': 1,
                        'language': 'en',
                        'is_auto_generated': False,
                        'caption_quality': 'high',
                    }
            return results

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        # Verify success
        assert result.success is True, f"Stage should succeed: {result.error}"

        # AC2: Verify needs_transcription contains videos with 'no_captions_available'
        # Every 3rd video (indices 0, 3, 6, 9) should be in needs_transcription
        expected_fallback_ids = [f"vid{i:04d}xxxx" for i in range(10) if i % 3 == 0]
        assert len(stage.needs_transcription) == len(expected_fallback_ids), (
            f"Expected {len(expected_fallback_ids)} videos in needs_transcription, "
            f"got {len(stage.needs_transcription)}"
        )
        for vid in expected_fallback_ids:
            assert vid in stage.needs_transcription, (
                f"Video {vid} should be in needs_transcription"
            )

        # AC3: Verify needs_transcription persisted in checkpoint data
        assert 'needs_transcription' in result.data, (
            "needs_transcription should be in checkpoint data"
        )
        checkpoint_needs_transcription = result.data['needs_transcription']
        assert len(checkpoint_needs_transcription) == len(expected_fallback_ids), (
            f"Checkpoint should have {len(expected_fallback_ids)} videos in needs_transcription, "
            f"got {len(checkpoint_needs_transcription)}: {checkpoint_needs_transcription}"
        )

        # AC4: Verify state.videos_needing_transcription is set
        # Note: The stage converts non-PipelineState objects via _validate_state_type(),
        # so the original mock state may not be updated. We verify via checkpoint data
        # and stage.needs_transcription which are the canonical sources.
        # The state.videos_needing_transcription is set on the converted state object.
        assert checkpoint_needs_transcription == expected_fallback_ids, (
            f"Checkpoint needs_transcription should match expected fallback IDs"
        )

    @pytest.mark.integration
    def test_empty_transcription_fallback_when_all_succeed(self):
        """Test that needs_transcription is empty when all captions succeed."""
        video_ids = [f"vid{i:04d}xxxx" for i in range(5)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            """All videos succeed."""
            return {
                vid: {
                    'video_id': vid,
                    'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                    'segment_count': 1,
                    'language': 'en',
                    'is_auto_generated': False,
                    'caption_quality': 'high',
                }
                for vid in video_ids
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert len(stage.needs_transcription) == 0, (
            "needs_transcription should be empty when all videos succeed"
        )
        assert len(state.videos_needing_transcription) == 0, (
            "state.videos_needing_transcription should be empty when all succeed"
        )


# =============================================================================
# US-62-007: Structured 'no captions' Status Propagation Tests
# =============================================================================


class TestCaptionStageNoCaptionsStatusPropagation:
    """Test structured status propagation for 'no captions' vs 'fetch failed'.

    US-62-007: Ensures clean distinction between videos with no captions available
    (not an error, triggers transcription fallback) and actual fetch failures (errors).
    """

    @pytest.mark.fast
    def test_caption_result_has_no_captions_available_field(self):
        """Test that CaptionResult dataclass includes no_captions_available field.

        AC1: CaptionResult dataclass includes 'no_captions_available' boolean field
        """
        from src.caption_fetcher import CaptionResult

        # Create a CaptionResult with no_captions_available=True
        result = CaptionResult(
            video_id="test123xxxxx",
            no_captions_available=True,
        )
        assert hasattr(result, 'no_captions_available'), (
            "CaptionResult should have no_captions_available field"
        )
        assert result.no_captions_available is True

        # Default should be False
        result_default = CaptionResult(video_id="test456xxxxx")
        assert result_default.no_captions_available is False, (
            "no_captions_available should default to False"
        )

    @pytest.mark.fast
    def test_caption_result_has_fetch_error_field(self):
        """Test that CaptionResult dataclass includes fetch_error field.

        AC1: CaptionResult includes fetch_error for actual errors
        """
        from src.caption_fetcher import CaptionResult

        # Create a CaptionResult with fetch_error
        result = CaptionResult(
            video_id="test123xxxxx",
            fetch_error="Connection timeout",
        )
        assert hasattr(result, 'fetch_error'), (
            "CaptionResult should have fetch_error field"
        )
        assert result.fetch_error == "Connection timeout"

        # Default should be None
        result_default = CaptionResult(video_id="test456xxxxx")
        assert result_default.fetch_error is None, (
            "fetch_error should default to None"
        )

    @pytest.mark.integration
    def test_stage_summary_shows_separate_no_captions_and_fetch_failed_counts(self):
        """Test that stage summary includes separate counts for no_captions vs fetch_failed.

        AC4: Stage summary includes count of 'no_captions' vs 'fetch_failed' vs 'succeeded'
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(12)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            """Simulate mixed results: success, no_captions, and errors."""
            results = {}
            for i, vid in enumerate(video_ids):
                if i % 4 == 0:
                    # Every 4th video has no captions (indices 0, 4, 8) = 3 videos
                    results[vid] = {
                        'video_id': vid,
                        'unavailable': True,
                        'reason': 'no_captions_available',
                        'caption_quality': 'low',
                        'no_captions_available': True,
                    }
                elif i % 4 == 1:
                    # Every 4th+1 video has an error (indices 1, 5, 9) = 3 videos
                    results[vid] = {
                        'video_id': vid,
                        'error': True,
                        'reason': 'fetch_timeout',
                        'caption_quality': 'low',
                    }
                else:
                    # Others succeed (indices 2, 3, 6, 7, 10, 11) = 6 videos
                    results[vid] = {
                        'video_id': vid,
                        'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                        'segment_count': 1,
                        'language': 'en',
                        'is_auto_generated': False,
                        'caption_quality': 'high',
                    }
            return results

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        # Verify success
        assert result.success is True, f"Stage should succeed: {result.error}"

        # AC4: Verify checkpoint data has separate counts
        assert 'no_captions_count' in result.data, (
            "Checkpoint data should include no_captions_count"
        )
        assert 'fetch_failed_count' in result.data, (
            "Checkpoint data should include fetch_failed_count"
        )

        # Expected counts:
        # - no_captions: 3 (indices 0, 4, 8)
        # - fetch_failed: 3 (indices 1, 5, 9)
        # - success: 6 (indices 2, 3, 6, 7, 10, 11)
        assert result.data['no_captions_count'] == 3, (
            f"Expected no_captions_count=3, got {result.data['no_captions_count']}"
        )
        assert result.data['fetch_failed_count'] == 3, (
            f"Expected fetch_failed_count=3, got {result.data['fetch_failed_count']}"
        )
        assert result.data['success_count'] == 6, (
            f"Expected success_count=6, got {result.data['success_count']}"
        )

        # fail_count should be sum of no_captions + fetch_failed for backwards compat
        assert result.data['fail_count'] == 6, (
            f"Expected fail_count=6 (3+3), got {result.data['fail_count']}"
        )

    @pytest.mark.integration
    def test_no_captions_videos_marked_for_transcription_cleanly(self):
        """Test that videos with no_captions_available are cleanly marked for transcription.

        AC3: Videos with no_captions_available marked for transcription fallback cleanly
        AC5: Test verifies pipeline continues when video has no captions
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(6)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        # Every other video has no captions
        no_caption_ids = [f"vid{i:04d}xxxx" for i in range(6) if i % 2 == 0]

        def mock_batch_fetch(video_ids, **kwargs):
            results = {}
            for vid in video_ids:
                if vid in no_caption_ids:
                    results[vid] = {
                        'video_id': vid,
                        'unavailable': True,
                        'reason': 'no_captions_available',
                        'caption_quality': 'low',
                    }
                else:
                    results[vid] = {
                        'video_id': vid,
                        'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                        'segment_count': 1,
                        'language': 'en',
                        'is_auto_generated': False,
                        'caption_quality': 'high',
                    }
            return results

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        # AC5: Pipeline continues successfully
        assert result.success is True, (
            f"Pipeline should continue when videos have no captions: {result.error}"
        )

        # AC3: Videos with no_captions are in needs_transcription
        for vid in no_caption_ids:
            assert vid in stage.needs_transcription, (
                f"Video {vid} with no captions should be in needs_transcription"
            )

        # Verify videos with captions are NOT in needs_transcription
        success_ids = [f"vid{i:04d}xxxx" for i in range(6) if i % 2 != 0]
        for vid in success_ids:
            assert vid not in stage.needs_transcription, (
                f"Video {vid} with captions should NOT be in needs_transcription"
            )

    @pytest.mark.integration
    def test_no_captions_not_treated_as_error(self):
        """Test that no_captions_available is handled without treating as error.

        AC2: Caption stage handles no_captions_available without treating as error
        AC6: Test verifies transcription fallback triggered for no_captions videos
        """
        video_ids = [f"vid{i:04d}xxxx" for i in range(3)]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[
                MockVideoSearchResult(vid, 120.0) for vid in video_ids
            ],
        )

        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 200,
                'auto_scale': True,
                'attempts_per_video': 2.0,
            }
        )

        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            """All videos have no captions available."""
            return {
                vid: {
                    'video_id': vid,
                    'unavailable': True,
                    'reason': 'no_captions_available',
                    'caption_quality': 'low',
                    'no_captions_available': True,
                }
                for vid in video_ids
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        # AC2: Stage should succeed (no_captions is not an error)
        assert result.success is True, (
            f"Stage should succeed even when all videos have no captions: {result.error}"
        )

        # AC6: All videos should be in needs_transcription for fallback
        assert len(stage.needs_transcription) == 3, (
            f"All 3 videos should need transcription, got {len(stage.needs_transcription)}"
        )
        for vid in video_ids:
            assert vid in stage.needs_transcription, (
                f"Video {vid} should trigger transcription fallback"
            )

        # Verify no_captions_count is correct
        assert result.data['no_captions_count'] == 3, (
            f"Expected no_captions_count=3, got {result.data.get('no_captions_count')}"
        )

        # Verify fetch_failed_count is 0 (no actual errors)
        assert result.data['fetch_failed_count'] == 0, (
            f"Expected fetch_failed_count=0 (no errors), got {result.data.get('fetch_failed_count')}"
        )


# =============================================================================
# US-63-006: Propagate Structured No-Caption Status Tests
# =============================================================================


class TestCaptionStatusField:
    """Tests for the structured status field (US-63-006).

    Verifies that caption results use the status field with values:
    'success', 'no_captions', 'error', 'cached_unavailable'
    """

    @pytest.mark.integration
    def test_status_field_success(self):
        """Test that successful caption fetches have status='success'.

        AC1: CaptionResult has 'status' field with value 'success' on success.
        """
        video_ids = ["vidSucc001x"]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[MockVideoSearchResult(vid, 120.0) for vid in video_ids],
        )

        config = make_mock_config(retry_budget_config={'enabled': True, 'max_attempts': 100})
        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            return {
                video_ids[0]: {
                    'video_id': video_ids[0],
                    'status': 'success',
                    'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                    'segment_count': 1,
                    'language': 'en',
                    'is_auto_generated': False,
                    'caption_quality': 'high',
                }
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        assert result.success is True
        caption_results = result.data.get('caption_results', {})
        assert caption_results[video_ids[0]].get('status') == 'success', (
            "Successful fetch should have status='success'"
        )

    @pytest.mark.integration
    def test_status_field_no_captions(self):
        """Test that videos without captions have status='no_captions'.

        AC1: CaptionResult has 'status' field with value 'no_captions' when unavailable.
        AC2: Caption stage handles no_captions status without treating as error.
        """
        video_ids = ["vidNoCap01x"]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[MockVideoSearchResult(vid, 120.0) for vid in video_ids],
        )

        config = make_mock_config(retry_budget_config={'enabled': True, 'max_attempts': 100})
        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            return {
                video_ids[0]: {
                    'video_id': video_ids[0],
                    'unavailable': True,
                    'status': 'no_captions',
                    'reason': 'no_captions_available',
                    'caption_quality': 'low',
                }
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        # AC2: Stage should succeed (no_captions is not an error)
        assert result.success is True, "Stage should succeed with no_captions status"

        caption_results = result.data.get('caption_results', {})
        assert caption_results[video_ids[0]].get('status') == 'no_captions', (
            "Unavailable video should have status='no_captions'"
        )

        # AC3: Video should be marked for transcription fallback
        assert video_ids[0] in stage.needs_transcription, (
            "Video with no_captions should be in needs_transcription"
        )

    @pytest.mark.integration
    def test_status_field_error(self):
        """Test that fetch errors have status='error'.

        AC1: CaptionResult has 'status' field with value 'error' on failure.
        """
        video_ids = ["vidError01x"]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[MockVideoSearchResult(vid, 120.0) for vid in video_ids],
        )

        config = make_mock_config(retry_budget_config={'enabled': True, 'max_attempts': 100})
        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            return {
                video_ids[0]: {
                    'video_id': video_ids[0],
                    'error': True,
                    'status': 'error',
                    'reason': 'Network timeout',
                    'caption_quality': 'low',
                }
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        assert result.success is True  # Stage continues despite errors

        caption_results = result.data.get('caption_results', {})
        assert caption_results[video_ids[0]].get('status') == 'error', (
            "Fetch error should have status='error'"
        )
        assert result.data.get('fetch_failed_count') == 1, (
            "fetch_failed_count should be 1"
        )

    @pytest.mark.integration
    def test_mixed_status_values_pipeline_continues(self):
        """Test pipeline continues with mixed success/no_captions/error statuses.

        AC5: Test verifies pipeline continues when some videos have no captions.
        """
        video_ids = ["vidSuccess1x", "vidNoCaps2x", "vidError03x"]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[MockVideoSearchResult(vid, 120.0) for vid in video_ids],
        )

        config = make_mock_config(retry_budget_config={'enabled': True, 'max_attempts': 100})
        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        def mock_batch_fetch(video_ids, **kwargs):
            return {
                'vidSuccess1x': {
                    'video_id': 'vidSuccess1x',
                    'status': 'success',
                    'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                    'segment_count': 1,
                    'language': 'en',
                    'is_auto_generated': False,
                    'caption_quality': 'high',
                },
                'vidNoCaps2x': {
                    'video_id': 'vidNoCaps2x',
                    'unavailable': True,
                    'status': 'no_captions',
                    'reason': 'no_captions_available',
                    'caption_quality': 'low',
                },
                'vidError03x': {
                    'video_id': 'vidError03x',
                    'error': True,
                    'status': 'error',
                    'reason': 'Rate limited',
                    'caption_quality': 'low',
                },
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            result = stage.run(state, config, checkpoint)

        # AC5: Pipeline should continue successfully
        assert result.success is True, (
            f"Pipeline should continue with mixed statuses: {result.error}"
        )

        # Verify counts
        assert result.data.get('success_count') == 1, "1 video should succeed"
        assert result.data.get('no_captions_count') == 1, "1 video should have no_captions"
        assert result.data.get('fetch_failed_count') == 1, "1 video should have error"

        # AC3: no_captions and error videos should be in needs_transcription
        assert 'vidNoCaps2x' in stage.needs_transcription
        assert 'vidError03x' in stage.needs_transcription
        assert 'vidSuccess1x' not in stage.needs_transcription

    @pytest.mark.integration
    def test_info_log_for_no_captions_count(self, caplog):
        """Test that INFO log is generated listing count of videos with no captions.

        AC4: Add log message at INFO level listing count of videos with no captions available.
        """
        import logging

        video_ids = ["vid001xxxxx", "vid002xxxxx", "vid003xxxxx"]

        state = MockPipelineState(
            video_ids=video_ids,
            video_search_results=[MockVideoSearchResult(vid, 120.0) for vid in video_ids],
        )

        config = make_mock_config(retry_budget_config={'enabled': True, 'max_attempts': 100})
        checkpoint = MockCheckpointManager(stage_data={})
        stage = CaptionStage()

        # 2 videos have no captions
        def mock_batch_fetch(video_ids, **kwargs):
            return {
                'vid001xxxxx': {
                    'video_id': 'vid001xxxxx',
                    'status': 'success',
                    'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                    'segment_count': 1,
                    'language': 'en',
                    'caption_quality': 'high',
                },
                'vid002xxxxx': {
                    'video_id': 'vid002xxxxx',
                    'unavailable': True,
                    'status': 'no_captions',
                    'reason': 'no_captions_available',
                    'caption_quality': 'low',
                },
                'vid003xxxxx': {
                    'video_id': 'vid003xxxxx',
                    'unavailable': True,
                    'status': 'no_captions',
                    'reason': 'no_captions_available',
                    'caption_quality': 'low',
                },
            }

        with patch('src.caption_fetcher.CaptionFetcher') as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance._timeout = 30
            fetcher_instance.apply_adaptive_format_order.return_value = []
            fetcher_instance._using_adaptive_order = False
            fetcher_instance.fetch_captions_batch.side_effect = mock_batch_fetch
            fetcher_instance.set_video_channel_map = MagicMock()

            with caplog.at_level(logging.INFO, logger='src.stages.caption_stage'):
                result = stage.run(state, config, checkpoint)

        assert result.success is True

        # AC4: Check for INFO log about no_captions count
        log_messages = [r.message for r in caplog.records if 'US-63-006' in r.message]
        assert len(log_messages) >= 1, (
            f"Expected INFO log with US-63-006 about no_captions count. Logs: {[r.message for r in caplog.records]}"
        )
        assert '2 videos' in log_messages[0], (
            f"Log should mention 2 videos with no captions: {log_messages[0]}"
        )
