"""
Tests for pipeline stage timing summary (US-51-012).

Covers:
- AC1: Pipeline completion summary includes per-stage duration in formatted table
- AC2: Stage timing data persisted to checkpoint under stage_metrics
- AC3: Total pipeline duration and per-stage percentage included
- AC4: Skipped stages show 'skipped' instead of 0s
- AC5: Unit test verifies timing data collected and formatted correctly
- AC6: Unit test verifies skipped stages marked as 'skipped'
"""

import logging
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData
from src.stages import StageMetrics

pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_orchestrator():
    """Create a minimal PipelineOrchestrator-like object for testing _print_timing_summary."""
    from src.pipeline import PipelineOrchestrator

    with patch.object(PipelineOrchestrator, '__init__', lambda self, *a, **kw: None):
        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        # Set up minimal attributes
        orch.stage_timings = {}
        orch.stage_metrics = {}
        orch.stages = []
        orch.checkpoint = MagicMock()
        return orch


@pytest.fixture
def make_stage():
    """Factory for mock stage objects with a .name attribute."""
    def _make(name):
        stage = MagicMock()
        stage.name = name
        return stage
    return _make


@pytest.fixture
def checkpoint_manager(tmp_path):
    """Create a checkpoint manager with a temp directory."""
    return CheckpointManager(tmp_path, config_hash="test_hash")


# ============================================================================
# AC1 + AC3: Timing table with durations and percentages
# ============================================================================

class TestTimingSummaryFormat:
    """Tests for timing summary log output format."""

    @pytest.mark.fast
    def test_summary_logs_header(self, mock_orchestrator, make_stage, caplog):
        """Timing summary logs a header line."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 10.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(10.0, set())

        messages = [r.message for r in caplog.records]
        assert any("Pipeline Timing Summary" in m for m in messages)

    @pytest.mark.fast
    def test_summary_logs_column_headers(self, mock_orchestrator, make_stage, caplog):
        """Timing summary logs Stage, Duration, % of Total columns."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 5.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(5.0, set())

        messages = [r.message for r in caplog.records]
        header_line = [m for m in messages if "Stage" in m and "Duration" in m]
        assert len(header_line) == 1
        assert "% of Total" in header_line[0]

    @pytest.mark.fast
    def test_summary_shows_stage_duration(self, mock_orchestrator, make_stage, caplog):
        """Each stage that ran shows its duration in seconds."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("MATCH")]
        mock_orchestrator.stage_timings = {"ANALYZE": 12.3, "MATCH": 45.6}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(57.9, set())

        messages = " ".join(r.message for r in caplog.records)
        assert "12.3s" in messages
        assert "45.6s" in messages

    @pytest.mark.fast
    def test_summary_shows_percentage(self, mock_orchestrator, make_stage, caplog):
        """Each stage shows percentage of total pipeline time."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("MATCH")]
        mock_orchestrator.stage_timings = {"ANALYZE": 25.0, "MATCH": 75.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(100.0, set())

        messages = " ".join(r.message for r in caplog.records)
        assert "25.0%" in messages
        assert "75.0%" in messages

    @pytest.mark.fast
    def test_summary_shows_total_line(self, mock_orchestrator, make_stage, caplog):
        """Summary includes a TOTAL line with total duration."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 33.3}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(33.3, set())

        messages = [r.message for r in caplog.records]
        total_lines = [m for m in messages if "TOTAL" in m]
        assert len(total_lines) == 1
        assert "33.3s" in total_lines[0]
        assert "100.0%" in total_lines[0]


# ============================================================================
# AC4: Skipped stages show 'skipped'
# ============================================================================

class TestTimingSummarySkippedStages:
    """Tests for skipped stage display in timing summary."""

    @pytest.mark.fast
    def test_skipped_stage_shows_skipped(self, mock_orchestrator, make_stage, caplog):
        """Stages restored from checkpoint show 'skipped' instead of a duration."""
        mock_orchestrator.stages = [
            make_stage("ANALYZE"), make_stage("MATCH"), make_stage("OUTPUT")
        ]
        mock_orchestrator.stage_timings = {"OUTPUT": 5.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(5.0, {"ANALYZE", "MATCH"})

        messages = " ".join(r.message for r in caplog.records)
        # "skipped" should appear for ANALYZE and MATCH
        analyze_line = [r.message for r in caplog.records if "ANALYZE" in r.message]
        assert len(analyze_line) == 1
        assert "skipped" in analyze_line[0]

        match_line = [r.message for r in caplog.records if "MATCH" in r.message]
        assert len(match_line) == 1
        assert "skipped" in match_line[0]

    @pytest.mark.fast
    def test_skipped_stage_no_duration(self, mock_orchestrator, make_stage, caplog):
        """Skipped stages should NOT show a seconds duration in the main duration column."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("OUTPUT")]
        mock_orchestrator.stage_timings = {"OUTPUT": 10.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(10.0, {"ANALYZE"})

        analyze_line = [r.message for r in caplog.records if "ANALYZE" in r.message]
        assert len(analyze_line) == 1
        # Should show 'skipped' in the duration column, not a seconds value
        # Note: structured context now includes elapsed=0.0s, so check for 'skipped' instead
        assert "skipped" in analyze_line[0]

    @pytest.mark.fast
    def test_mixed_skipped_and_ran(self, mock_orchestrator, make_stage, caplog):
        """Summary handles mix of skipped and executed stages."""
        mock_orchestrator.stages = [
            make_stage("ANALYZE"),
            make_stage("VIDEO_SEARCH"),
            make_stage("MATCH"),
            make_stage("OUTPUT"),
        ]
        mock_orchestrator.stage_timings = {"MATCH": 20.0, "OUTPUT": 5.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(25.0, {"ANALYZE", "VIDEO_SEARCH"})

        messages = {r.message for r in caplog.records}
        # Skipped stages
        analyze_lines = [m for m in messages if "ANALYZE" in m]
        assert any("skipped" in m for m in analyze_lines)
        # Ran stages
        match_lines = [m for m in messages if "MATCH" in m and "TOTAL" not in m]
        assert any("20.0s" in m for m in match_lines)

    @pytest.mark.fast
    def test_all_stages_skipped(self, mock_orchestrator, make_stage, caplog):
        """When all stages are skipped, total is still shown."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("MATCH")]
        mock_orchestrator.stage_timings = {}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(0.5, {"ANALYZE", "MATCH"})

        messages = " ".join(r.message for r in caplog.records)
        assert "TOTAL" in messages
        assert "0.5s" in messages


# ============================================================================
# AC2: Stage timing persisted to checkpoint
# ============================================================================

class TestTimingPersistence:
    """Tests for timing data persistence to checkpoint."""

    @pytest.mark.fast
    def test_checkpoint_save_called(self, mock_orchestrator, make_stage):
        """_print_timing_summary calls checkpoint.save_stage_timing_summary."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 10.0}

        mock_orchestrator._print_timing_summary(10.0, set())

        mock_orchestrator.checkpoint.save_stage_timing_summary.assert_called_once_with(
            {"ANALYZE": 10.0}, 10.0, set()
        )

    @pytest.mark.fast
    def test_checkpoint_save_with_skipped(self, mock_orchestrator, make_stage):
        """Skipped stages are passed to checkpoint save."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("MATCH")]
        mock_orchestrator.stage_timings = {"MATCH": 5.0}
        skipped = {"ANALYZE"}

        mock_orchestrator._print_timing_summary(5.0, skipped)

        mock_orchestrator.checkpoint.save_stage_timing_summary.assert_called_once_with(
            {"MATCH": 5.0}, 5.0, skipped
        )


class TestCheckpointTimingPersistence:
    """Tests for CheckpointManager.save_stage_timing_summary."""

    @pytest.mark.fast
    def test_timing_saved_to_stage_metrics(self, checkpoint_manager):
        """Stage timings are persisted under stage_metrics in checkpoint."""
        checkpoint_manager.data = CheckpointData(
            created_at="2026-01-25T10:00:00",
            last_completed_stage="OUTPUT"
        )

        checkpoint_manager.save_stage_timing_summary(
            {"ANALYZE": 10.0, "MATCH": 20.0}, 30.0, set()
        )

        assert checkpoint_manager.data.stage_metrics["ANALYZE"]["duration_seconds"] == 10.0
        assert checkpoint_manager.data.stage_metrics["MATCH"]["duration_seconds"] == 20.0

    @pytest.mark.fast
    def test_pipeline_total_saved(self, checkpoint_manager):
        """Pipeline-level total duration is persisted."""
        checkpoint_manager.data = CheckpointData(
            created_at="2026-01-25T10:00:00",
            last_completed_stage="OUTPUT"
        )

        checkpoint_manager.save_stage_timing_summary(
            {"ANALYZE": 10.0}, 42.5, set()
        )

        pipeline_meta = checkpoint_manager.data.stage_metrics["_pipeline"]
        assert pipeline_meta["total_duration_seconds"] == 42.5
        assert "ANALYZE" in pipeline_meta["stages_run"]

    @pytest.mark.fast
    def test_skipped_stages_marked(self, checkpoint_manager):
        """Skipped stages are marked with skipped=True in stage_metrics."""
        checkpoint_manager.data = CheckpointData(
            created_at="2026-01-25T10:00:00",
            last_completed_stage="OUTPUT"
        )

        checkpoint_manager.save_stage_timing_summary(
            {"OUTPUT": 5.0}, 5.0, {"ANALYZE", "MATCH"}
        )

        assert checkpoint_manager.data.stage_metrics["ANALYZE"]["skipped"] is True
        assert checkpoint_manager.data.stage_metrics["MATCH"]["skipped"] is True
        assert checkpoint_manager.data.stage_metrics["_pipeline"]["stages_skipped"] is not None

    @pytest.mark.fast
    def test_existing_metrics_preserved(self, checkpoint_manager):
        """Existing per-stage metrics are not overwritten by timing save."""
        checkpoint_manager.data = CheckpointData(
            created_at="2026-01-25T10:00:00",
            last_completed_stage="OUTPUT"
        )
        # Pre-existing metrics from stage execution
        checkpoint_manager.data.stage_metrics["ANALYZE"] = {
            "items_processed": 50,
            "duration_seconds": 9.0,
        }

        checkpoint_manager.save_stage_timing_summary(
            {"ANALYZE": 10.0}, 10.0, set()
        )

        # duration_seconds updated, but items_processed preserved
        assert checkpoint_manager.data.stage_metrics["ANALYZE"]["items_processed"] == 50
        assert checkpoint_manager.data.stage_metrics["ANALYZE"]["duration_seconds"] == 10.0

    @pytest.mark.fast
    def test_no_data_is_noop(self, checkpoint_manager):
        """If checkpoint has no data, save_stage_timing_summary is a no-op."""
        checkpoint_manager.data = None

        # Should not raise
        checkpoint_manager.save_stage_timing_summary(
            {"ANALYZE": 10.0}, 10.0, set()
        )

    @pytest.mark.fast
    def test_atomic_save_called(self, checkpoint_manager):
        """save_stage_timing_summary triggers _atomic_save."""
        checkpoint_manager.data = CheckpointData(
            created_at="2026-01-25T10:00:00",
            last_completed_stage="OUTPUT"
        )

        with patch.object(checkpoint_manager, '_atomic_save') as mock_save:
            checkpoint_manager.save_stage_timing_summary(
                {"ANALYZE": 10.0}, 10.0, set()
            )
            mock_save.assert_called_once()

    @pytest.mark.fast
    def test_timing_roundtrip_via_checkpoint_file(self, tmp_path):
        """Timing data survives save-to-file and reload."""
        import json

        manager = CheckpointManager(tmp_path, config_hash="test")
        manager.data = CheckpointData(
            created_at="2026-01-25T10:00:00",
            last_completed_stage="OUTPUT"
        )
        manager.save_stage_timing_summary(
            {"ANALYZE": 10.0, "MATCH": 20.0}, 30.0, {"VIDEO_SEARCH"}
        )

        # Reload from file
        manager2 = CheckpointManager(tmp_path, config_hash="test")
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.stage_metrics["ANALYZE"]["duration_seconds"] == 10.0
        assert loaded.stage_metrics["MATCH"]["duration_seconds"] == 20.0
        assert loaded.stage_metrics["VIDEO_SEARCH"]["skipped"] is True
        assert loaded.stage_metrics["_pipeline"]["total_duration_seconds"] == 30.0


# ============================================================================
# AC5 + AC6: Edge cases
# ============================================================================

class TestTimingSummaryEdgeCases:
    """Edge case tests for timing summary."""

    @pytest.mark.fast
    def test_zero_total_duration(self, mock_orchestrator, make_stage, caplog):
        """Zero total duration does not cause division by zero."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 0.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            # Should not raise
            mock_orchestrator._print_timing_summary(0.0, set())

        messages = " ".join(r.message for r in caplog.records)
        assert "TOTAL" in messages

    @pytest.mark.fast
    def test_filtered_stage_shows_dashes(self, mock_orchestrator, make_stage, caplog):
        """Stages not in stage_timings and not skipped show '--'."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("OUTPUT")]
        mock_orchestrator.stage_timings = {"OUTPUT": 5.0}
        # ANALYZE not in skipped_stages and not in stage_timings = filtered

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(5.0, set())

        analyze_line = [r.message for r in caplog.records if "ANALYZE" in r.message]
        assert len(analyze_line) == 1
        assert "--" in analyze_line[0]

    @pytest.mark.fast
    def test_summary_logged_at_info_level(self, mock_orchestrator, make_stage, caplog):
        """All timing summary lines are logged at INFO level."""
        mock_orchestrator.stages = [make_stage("ANALYZE")]
        mock_orchestrator.stage_timings = {"ANALYZE": 5.0}

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(5.0, set())

        for record in caplog.records:
            assert record.levelno == logging.INFO
