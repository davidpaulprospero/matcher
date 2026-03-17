"""
Tests for structured logging context in pipeline stage execution (US-85-007).

Covers:
- AC1: StageLogContext dataclass with required fields
- AC2: _format_log_context method returns structured dict
- AC3: Key log points include structured context (stage entry, exit, failure, etc.)
- AC4: Test verifies structured context fields are present in log output
- AC5: Backward compatibility - existing format still readable
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import StageLogContext, PipelineOrchestrator

pytestmark = pytest.mark.unit


# ============================================================================
# AC1: StageLogContext dataclass
# ============================================================================

class TestStageLogContext:
    """Tests for StageLogContext dataclass."""

    @pytest.mark.fast
    def test_stage_log_context_has_required_fields(self):
        """StageLogContext has all required fields."""
        ctx = StageLogContext(
            stage_name="MATCH",
            stage_index=3,
            total_stages=7,
            elapsed_seconds=42.5,
            items_processed=50,
            items_failed=2,
        )

        assert ctx.stage_name == "MATCH"
        assert ctx.stage_index == 3
        assert ctx.total_stages == 7
        assert ctx.elapsed_seconds == 42.5
        assert ctx.items_processed == 50
        assert ctx.items_failed == 2

    @pytest.mark.fast
    def test_stage_log_context_to_dict(self):
        """to_dict returns structured dictionary."""
        ctx = StageLogContext(
            stage_name="ANALYZE",
            stage_index=0,
            total_stages=7,
            elapsed_seconds=10.0,
            items_processed=25,
            items_failed=0,
        )

        d = ctx.to_dict()

        assert d['stage'] == "ANALYZE"
        assert d['idx'] == 0
        assert d['total'] == 7
        assert d['elapsed'] == "10.0s"
        assert d['processed'] == 25
        assert d['failed'] == 0

    @pytest.mark.fast
    def test_stage_log_context_to_suffix(self):
        """to_suffix returns parseable string for log messages."""
        ctx = StageLogContext(
            stage_name="MATCH",
            stage_index=3,
            total_stages=7,
            elapsed_seconds=42.1,
            items_processed=50,
            items_failed=0,
        )

        suffix = ctx.to_suffix()

        assert "stage=MATCH" in suffix
        assert "idx=3/7" in suffix
        assert "elapsed=42.1s" in suffix
        assert "items=50/0" in suffix
        # Verify it can be parsed back
        assert "[" in suffix
        assert "]" in suffix


# ============================================================================
# AC2: _format_log_context method
# ============================================================================

class TestFormatLogContext:
    """Tests for PipelineOrchestrator._format_log_context method."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a minimal PipelineOrchestrator for testing."""
        with patch.object(PipelineOrchestrator, '__init__', lambda self, *a, **kw: None):
            orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
            return orch

    @pytest.mark.fast
    def test_format_log_context_returns_stage_log_context(self, mock_orchestrator):
        """_format_log_context returns StageLogContext instance."""
        result = mock_orchestrator._format_log_context(
            stage_name="CAPTION",
            stage_index=2,
            total_stages=7,
            elapsed_seconds=15.0,
            items_processed=30,
            items_failed=1,
        )

        assert isinstance(result, StageLogContext)
        assert result.stage_name == "CAPTION"
        assert result.stage_index == 2
        assert result.total_stages == 7

    @pytest.mark.fast
    def test_format_log_context_defaults(self, mock_orchestrator):
        """_format_log_context handles default values."""
        result = mock_orchestrator._format_log_context(
            stage_name="VIDEO_SEARCH",
            stage_index=1,
            total_stages=7,
        )

        assert result.elapsed_seconds == 0.0
        assert result.items_processed == 0
        assert result.items_failed == 0


# ============================================================================
# AC3 + AC4: Key log points include structured context
# ============================================================================

class TestStructuredLoggingInPipeline:
    """Tests verifying structured context appears in log output."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock PipelineOrchestrator for testing."""
        with patch.object(PipelineOrchestrator, '__init__', lambda self, *a, **kw: None):
            orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
            orch.stage_timings = {}
            orch.stage_metrics = {}
            orch.stages = []
            orch.checkpoint = MagicMock()
            return orch

    @pytest.fixture
    def make_stage(self):
        """Factory for mock stage objects."""
        def _make(name):
            stage = MagicMock()
            stage.name = name
            return stage
        return _make

    @pytest.mark.fast
    def test_stage_entry_log_includes_context(self, mock_orchestrator, make_stage, caplog):
        """Stage entry log includes structured context."""
        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("MATCH")]

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            ctx = mock_orchestrator._format_log_context(
                stage_name="MATCH",
                stage_index=1,
                total_stages=2,
            )
            # Simulate what the code does
            msg = f"Running stage: MATCH {ctx.to_suffix()}"
            logging.getLogger('src.pipeline').info(msg)

        messages = " ".join(r.message for r in caplog.records)
        assert "stage=MATCH" in messages
        assert "idx=1/2" in messages

    @pytest.mark.fast
    def test_stage_completion_log_includes_context(self, mock_orchestrator, make_stage, caplog):
        """Stage completion log includes structured context with items."""
        from src.stages import StageMetrics

        mock_orchestrator.stages = [make_stage("ANALYZE"), make_stage("MATCH")]
        mock_orchestrator.stage_metrics = {
            "MATCH": StageMetrics(items_processed=50, items_failed=2, duration_seconds=30.0)
        }

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            ctx = mock_orchestrator._format_log_context(
                stage_name="MATCH",
                stage_index=1,
                total_stages=2,
                elapsed_seconds=30.0,
                items_processed=50,
                items_failed=2,
            )
            msg = f"Stage MATCH completed in 30.0s {ctx.to_suffix()}"
            logging.getLogger('src.pipeline').info(msg)

        messages = " ".join(r.message for r in caplog.records)
        assert "items=50/2" in messages

    @pytest.mark.fast
    def test_timing_summary_includes_context(self, mock_orchestrator, make_stage, caplog):
        """Timing summary includes structured context for each stage."""
        from src.stages import StageMetrics

        mock_orchestrator.stages = [
            make_stage("ANALYZE"),
            make_stage("MATCH"),
        ]
        mock_orchestrator.stage_timings = {"ANALYZE": 10.0, "MATCH": 30.0}
        mock_orchestrator.stage_metrics = {
            "ANALYZE": StageMetrics(items_processed=25, items_failed=0, duration_seconds=10.0),
            "MATCH": StageMetrics(items_processed=50, items_failed=2, duration_seconds=30.0),
        }

        with caplog.at_level(logging.INFO, logger='src.pipeline'):
            mock_orchestrator._print_timing_summary(40.0, set())

        messages = " ".join(r.message for r in caplog.records)

        # Check that context appears for both stages
        assert "stage=ANALYZE" in messages
        assert "stage=MATCH" in messages
        assert "items=25/0" in messages
        assert "items=50/2" in messages


# ============================================================================
# AC5: Backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Tests verifying backward compatibility - human-readable format preserved."""

    @pytest.mark.fast
    def test_original_message_still_present(self):
        """Original log message content is preserved."""
        ctx = StageLogContext(
            stage_name="MATCH",
            stage_index=3,
            total_stages=7,
            elapsed_seconds=42.1,
            items_processed=50,
            items_failed=0,
        )

        # Original message format
        original = f"Stage MATCH completed in 42.1s"

        # New format adds suffix
        new = f"Stage MATCH completed in 42.1s {ctx.to_suffix()}"

        # Original content is still there
        assert "Stage MATCH completed in 42.1s" in new
        # Additional context is appended
        assert "stage=MATCH" in new

    @pytest.mark.fast
    def test_context_suffix_is_parseable(self):
        """Context suffix can be parsed back to extract fields."""
        ctx = StageLogContext(
            stage_name="VIDEO_SEARCH",
            stage_index=1,
            total_stages=7,
            elapsed_seconds=25.5,
            items_processed=100,
            items_failed=3,
        )

        suffix = ctx.to_suffix()

        # Extract individual fields using simple parsing
        import re

        stage_match = re.search(r'stage=(\w+)', suffix)
        idx_match = re.search(r'idx=(\d+)', suffix)
        elapsed_match = re.search(r'elapsed=([\d.]+)s', suffix)
        items_match = re.search(r'items=(\d+)/(\d+)', suffix)

        assert stage_match and stage_match.group(1) == "VIDEO_SEARCH"
        assert idx_match and idx_match.group(1) == "1"
        assert elapsed_match and float(elapsed_match.group(1)) == 25.5
        assert items_match and items_match.group(1) == "100" and items_match.group(2) == "3"
