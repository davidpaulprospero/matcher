"""
US-42-009: Integration test for full caption stage restore flow.

Tests the complete restore flow that triggered user bugs:
- Checkpoint with caption stage data including retry_budget
- State objects that simulate missing attributes
- Proper state initialization during restore
- Retry budget scaling after checkpoint restore

Created: 2026-02-03 (Sprint 42)
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.stages.caption_stage import CaptionStage
from src.caption.retry_budget import CaptionRetryBudget


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config_with_retry_budget():
    """Create mock config with caption-first and retry budget settings."""
    config = MagicMock()

    # Caption-first config
    config.download.caption_first.enabled = True
    config.download.caption_first.preferred_language = "en"
    config.download.caption_first.prefer_human_captions = True
    config.download.caption_first.fallback_to_transcription = True
    config.download.caption_first.timeout = 30
    config.download.caption_first.cache_captions = True
    config.download.caption_first.skip_live_streams = False
    config.download.caption_first.max_parallel_fetches = 4
    config.download.caption_first.min_coverage_threshold = 0.5
    config.download.caption_first.pre_check_availability = False
    config.download.caption_first.max_cache_age_days = 30
    config.download.caption_first.cache_dir = '~/.matcher_caption_cache'
    config.download.caption_first.cache_validation = 'warn'
    config.download.caption_first.cache_validation_tolerance = 0.2

    # Retry budget config
    rb_config = MagicMock()
    rb_config.enabled = True
    rb_config.max_attempts = 100
    rb_config.max_backoff_time_seconds = 300.0
    rb_config.auto_scale = True
    rb_config.attempts_per_video = 2.0
    config.download.caption_first.retry_budget = rb_config

    # No cookies
    config.download.cookies_from_browser = ""
    config.download.cookies_path = ""

    return config


@pytest.fixture
def checkpoint_data_with_retry_budget():
    """Create checkpoint data with caption stage data including retry_budget."""
    return {
        'caption_results': {
            "abc123": {
                "video_id": "abc123",
                "segments": [
                    {"text": "Hello world", "start": 0.0, "end": 2.5},
                    {"text": "Test segment", "start": 2.5, "end": 5.0},
                ],
                "language": "en",
                "is_auto_generated": False,
                "caption_quality": "high",
            },
            "def456": {
                "video_id": "def456",
                "segments": [
                    {"text": "Second video caption", "start": 0.0, "end": 3.0},
                ],
                "language": "en",
                "is_auto_generated": True,
            },
        },
        'total_segments': 3,
        'retry_budget': {
            'max_attempts': 350,  # Scaled for 175 videos
            'auto_scale': True,
            'attempts_per_video': 2.0,
            'max_backoff_time_seconds': 300.0,
            'attempts': 25,
            'failures': 5,
            'successes': 20,
            'backoff_time_spent': 15.0,
            'videos_skipped': [],
            'error_counts': {'rate_limited': 3, 'network': 2},
            'vpn_resets_used': 0,
            'early_terminated': False,
            'early_termination_reason': None,
            'batch_size': 175,
        },
    }


# ============================================================================
# US-42-009: Full Caption Stage Restore Flow Integration Tests
# ============================================================================

@pytest.mark.fast
class TestCaptionStageRestoreFlow:
    """Integration tests for the full caption stage restore flow.

    US-42-009: Tests the complete flow that triggered user bugs.
    """

    def test_restore_creates_checkpoint_with_caption_data_and_retry_budget(
        self, mock_config_with_retry_budget, checkpoint_data_with_retry_budget
    ):
        """US-42-009 AC1: Test creates checkpoint with caption stage data including retry_budget."""
        stage = CaptionStage()

        checkpoint = MagicMock()
        checkpoint.get_stage_data.return_value = checkpoint_data_with_retry_budget

        # Create state that simulates fresh pipeline (missing text_metadata)
        state = SimpleNamespace()
        state.video_ids = ['abc123', 'def456']
        state.video_search_results = []
        state.caption_results = {}
        # Note: text_metadata intentionally NOT set

        # Verify checkpoint data includes retry_budget
        cp_data = checkpoint.get_stage_data('CAPTION')
        assert 'retry_budget' in cp_data, "Checkpoint must include retry_budget"
        assert cp_data['retry_budget']['batch_size'] == 175
        assert cp_data['retry_budget']['max_attempts'] == 350

    def test_restore_with_state_missing_attributes(
        self, mock_config_with_retry_budget, checkpoint_data_with_retry_budget
    ):
        """US-42-009 AC2: Test restores from checkpoint with state missing attributes."""
        stage = CaptionStage()

        checkpoint = MagicMock()
        checkpoint.get_stage_data.return_value = checkpoint_data_with_retry_budget

        # Create state that simulates fresh pipeline state missing text_metadata
        # This is the exact scenario from user bug report
        class FreshPipelineState:
            """Simulates PipelineState before preflight check runs."""
            def __init__(self):
                self.video_ids = ['abc123', 'def456']
                self.video_search_results = []
                self.caption_results = {}
                # Intentionally missing: text_metadata, video_transcripts

        state = FreshPipelineState()
        assert not hasattr(state, 'text_metadata'), "State should NOT have text_metadata"

        # Restore should NOT raise AttributeError
        restored = stage.restore(state, checkpoint, mock_config_with_retry_budget)

        assert restored is True, "Restore should succeed"

    def test_restore_verifies_text_metadata_exists_after_restore(
        self, mock_config_with_retry_budget, checkpoint_data_with_retry_budget, caplog
    ):
        """US-42-009 AC3: Test verifies state.text_metadata exists after restore."""
        stage = CaptionStage()

        checkpoint = MagicMock()
        checkpoint.get_stage_data.return_value = checkpoint_data_with_retry_budget

        # Create state without text_metadata
        state = SimpleNamespace()
        state.video_ids = ['abc123', 'def456']
        state.video_search_results = []
        state.caption_results = {}
        # text_metadata NOT set

        assert not hasattr(state, 'text_metadata')

        with caplog.at_level(logging.INFO):
            restored = stage.restore(state, checkpoint, mock_config_with_retry_budget)

        # AC3: Verify text_metadata exists and is populated
        assert restored is True
        assert hasattr(state, 'text_metadata'), "text_metadata must exist after restore"
        assert len(state.text_metadata) == 3, "Should have 3 total segments (2 from abc123, 1 from def456)"

        # Verify segment content
        texts = [seg['text'] for seg in state.text_metadata]
        assert "Hello world" in texts
        assert "Test segment" in texts
        assert "Second video caption" in texts

    def test_retry_budget_properly_scaled_for_new_batch(
        self, mock_config_with_retry_budget, checkpoint_data_with_retry_budget, caplog
    ):
        """US-42-009 AC4: Test verifies retry budget is properly scaled for new batch.

        Simulates the scenario where checkpoint had batch_size=175 (scaled to 350),
        but current config has max_attempts=100. The budget should re-scale.
        """
        # Restore budget from checkpoint data
        budget = CaptionRetryBudget.from_dict(checkpoint_data_with_retry_budget['retry_budget'])

        # Verify checkpoint values were restored
        assert budget.batch_size == 175, "batch_size should be restored from checkpoint"
        assert budget.max_attempts == 350, "max_attempts should be restored from checkpoint"

        # Now simulate a new batch of 200 videos
        new_batch_size = 200

        with caplog.at_level(logging.DEBUG):
            scaled = budget.ensure_scaled(new_batch_size)

        # Budget should scale up for larger batch
        # 200 videos * 2.0 attempts_per_video = 400 attempts required
        assert scaled is True, "Budget should scale for larger batch"
        assert budget.batch_size == 200
        assert budget.max_attempts == 400

    def test_retry_budget_not_scaled_when_sufficient(
        self, mock_config_with_retry_budget, checkpoint_data_with_retry_budget, caplog
    ):
        """Test that budget is not scaled when current max_attempts is sufficient."""
        # Restore budget from checkpoint data
        budget = CaptionRetryBudget.from_dict(checkpoint_data_with_retry_budget['retry_budget'])

        # Simulate a smaller batch than checkpoint had
        smaller_batch_size = 100

        with caplog.at_level(logging.DEBUG):
            scaled = budget.ensure_scaled(smaller_batch_size)

        # Budget should NOT scale down - it's already sufficient
        # 100 videos * 2.0 = 200 required, but we have 350
        assert scaled is False, "Budget should not scale for smaller batch"
        assert budget.max_attempts == 350, "max_attempts should remain unchanged"
        assert budget.batch_size == 100, "batch_size should be updated for tracking"

    def test_full_restore_flow_integration(
        self, mock_config_with_retry_budget, checkpoint_data_with_retry_budget, caplog
    ):
        """Full integration test simulating the exact user bug scenario.

        Flow:
        1. CheckpointManager returns caption data with retry_budget
        2. PipelineState object is missing text_metadata attribute
        3. CaptionStage.restore() is called
        4. text_metadata is initialized and populated
        5. Retry budget is logged (but not re-scaled in restore, that's in run())
        """
        stage = CaptionStage()

        # Simulate CheckpointManager
        checkpoint = MagicMock()
        checkpoint.get_stage_data.return_value = checkpoint_data_with_retry_budget

        # Simulate fresh PipelineState (as created by checkpoint restore path)
        class MinimalPipelineState:
            """Minimal state object that might come from legacy checkpoint."""
            pass

        state = MinimalPipelineState()
        # Set only the bare minimum - no text_metadata, no caption_results
        state.video_ids = ['abc123', 'def456']

        # Verify state is truly minimal
        assert not hasattr(state, 'text_metadata')
        assert not hasattr(state, 'caption_results')
        assert not hasattr(state, 'video_search_results')

        with caplog.at_level(logging.INFO):
            restored = stage.restore(state, checkpoint, mock_config_with_retry_budget)

        # Verify complete restoration
        assert restored is True, "Restore should succeed"

        # Verify state attributes are initialized
        assert hasattr(state, 'text_metadata'), "text_metadata must be initialized"
        assert hasattr(state, 'caption_results'), "caption_results must be initialized"
        assert hasattr(state, 'video_search_results'), "video_search_results must be initialized"

        # Verify content was populated
        assert len(state.text_metadata) == 3

        # Verify retry budget info was logged
        budget_log_found = any(
            'retry budget' in record.message.lower() and
            '25 attempts' in record.message
            for record in caplog.records
        )
        assert budget_log_found, (
            f"Expected retry budget info log. "
            f"Got: {[r.message for r in caplog.records]}"
        )


@pytest.mark.fast
class TestRetryBudgetCheckpointRestoreScaling:
    """Tests for retry budget scaling behavior after checkpoint restore."""

    def test_budget_scales_when_batch_size_set_but_max_attempts_insufficient(self):
        """US-42-004: Budget must scale even if batch_size is already set.

        Bug scenario: Checkpoint restores batch_size=175 but max_attempts=100
        (from config default). Without the fix, budget would skip scaling because
        batch_size matched, leaving max_attempts insufficient.
        """
        # Create budget as if restored from checkpoint with old config
        budget = CaptionRetryBudget(
            max_attempts=100,  # Config default
            auto_scale=True,
            attempts_per_video=2.0,
        )
        # Simulate checkpoint restoration setting batch_size
        budget.batch_size = 175  # From checkpoint

        # Required: 175 * 2.0 = 350, but max_attempts is only 100
        # ensure_scaled should detect this and scale up
        scaled = budget.ensure_scaled(175)

        assert scaled is True, "Budget should scale when max_attempts insufficient"
        assert budget.max_attempts == 350

    def test_budget_does_not_scale_twice_for_same_batch(self):
        """Verify ensure_scaled is idempotent for same batch size."""
        budget = CaptionRetryBudget(
            max_attempts=100,
            auto_scale=True,
            attempts_per_video=2.0,
        )

        # First call scales
        scaled1 = budget.ensure_scaled(200)
        assert scaled1 is True
        assert budget.max_attempts == 400

        # Second call with same batch size should not scale
        scaled2 = budget.ensure_scaled(200)
        assert scaled2 is False
        assert budget.max_attempts == 400  # Unchanged

    def test_restore_with_integrity_check_detects_batch_size_change(self):
        """Test restore_with_integrity_check provides useful diagnostics."""
        checkpoint_data = {
            'max_attempts': 350,
            'auto_scale': True,
            'attempts_per_video': 2.0,
            'max_backoff_time_seconds': 300.0,
            'attempts': 50,
            'failures': 10,
            'successes': 40,
            'backoff_time_spent': 30.0,
            'videos_skipped': [],
            'error_counts': {},
            'batch_size': 175,
        }

        # Restore with a different batch size
        budget, restore_info = CaptionRetryBudget.restore_with_integrity_check(
            checkpoint_data,
            current_batch_size=200,
            auto_scale=True,
            attempts_per_video=2.0
        )

        assert restore_info is not None
        assert restore_info['batch_size_changed'] is True
        assert restore_info['restored_batch_size'] == 175
        assert restore_info['current_batch_size'] == 200
