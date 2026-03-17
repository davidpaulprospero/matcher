"""
Tests for pipeline data quality gate between MATCH and OUTPUT stages.

US-81-005: Verifies that warnings are emitted when match coverage
is below configurable threshold, and not emitted when above.
"""

import logging
import pytest
from unittest.mock import MagicMock, Mock

from src.pipeline import PipelineOrchestrator
from src.state import PipelineState, VoiceoverSegment, Match


@pytest.fixture
def mock_config():
    """Minimal config for quality gate testing."""
    config = MagicMock()
    config.pipeline.quality_gates.min_match_coverage = 0.5
    config.cache.cache_dir = ".cache"
    # Prevent freeze() from failing
    config.freeze = MagicMock()
    return config


@pytest.fixture
def orchestrator(mock_config, tmp_path):
    """Create a PipelineOrchestrator with no stages (for unit testing the gate method)."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = mock_config
    orch.project_dir = tmp_path
    orch.state = PipelineState()
    return orch


def _make_segments(n: int):
    """Create n VoiceoverSegment objects."""
    return [
        VoiceoverSegment(index=i, start=float(i), end=float(i + 1), text=f"seg {i}")
        for i in range(n)
    ]


def _make_matches(n: int):
    """Create n minimal Match objects."""
    return [
        Match(
            segment_index=i,
            video_file=f"vid_{i}.mp4",
            confidence=0.8,
            video_start=0.0,
            video_end=5.0,
        )
        for i in range(n)
    ]


# ============================================================================
# Tests
# ============================================================================


class TestMatchCoverageGate:
    """Test _check_match_coverage_gate method."""

    @pytest.mark.fast
    def test_warning_emitted_below_threshold(self, orchestrator, caplog):
        """Warning logged when match coverage < threshold."""
        orchestrator.state.voiceover_segments = _make_segments(10)
        orchestrator.state.matches = _make_matches(3)  # 30% < 50%

        with caplog.at_level(logging.WARNING):
            orchestrator._check_match_coverage_gate("MATCH")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("3/10" in m and "30%" in m for m in warning_msgs), (
            f"Expected warning about 3/10 segments, got: {warning_msgs}"
        )
        assert any("--force-rematch" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_no_warning_above_threshold(self, orchestrator, caplog):
        """No warning logged when match coverage >= threshold."""
        orchestrator.state.voiceover_segments = _make_segments(10)
        orchestrator.state.matches = _make_matches(8)  # 80% > 50%

        with caplog.at_level(logging.WARNING):
            orchestrator._check_match_coverage_gate("MATCH")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Quality gate" in m for m in warning_msgs), (
            f"Unexpected warning: {warning_msgs}"
        )

    @pytest.mark.fast
    def test_exact_threshold_no_warning(self, orchestrator, caplog):
        """No warning when coverage is exactly at threshold (50%)."""
        orchestrator.state.voiceover_segments = _make_segments(10)
        orchestrator.state.matches = _make_matches(5)  # 50% == 50%

        with caplog.at_level(logging.WARNING):
            orchestrator._check_match_coverage_gate("ITERATIVE_MATCH")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Quality gate" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_custom_threshold(self, orchestrator, caplog):
        """Custom threshold from config is respected."""
        orchestrator.config.pipeline.quality_gates.min_match_coverage = 0.9
        orchestrator.state.voiceover_segments = _make_segments(10)
        orchestrator.state.matches = _make_matches(8)  # 80% < 90%

        with caplog.at_level(logging.WARNING):
            orchestrator._check_match_coverage_gate("MATCH")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("8/10" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_zero_segments_no_crash(self, orchestrator, caplog):
        """No crash or warning with zero voiceover segments."""
        orchestrator.state.voiceover_segments = []
        orchestrator.state.matches = []

        with caplog.at_level(logging.WARNING):
            orchestrator._check_match_coverage_gate("MATCH")

        # Should silently return — no warnings
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Quality gate" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_no_matches_zero_percent(self, orchestrator, caplog):
        """Zero matches triggers warning."""
        orchestrator.state.voiceover_segments = _make_segments(5)
        orchestrator.state.matches = []  # 0%

        with caplog.at_level(logging.WARNING):
            orchestrator._check_match_coverage_gate("ITERATIVE_MATCH")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("0/5" in m and "0%" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_info_logged_above_threshold(self, orchestrator, caplog):
        """Info message logged when above threshold."""
        orchestrator.state.voiceover_segments = _make_segments(10)
        orchestrator.state.matches = _make_matches(7)

        with caplog.at_level(logging.INFO):
            orchestrator._check_match_coverage_gate("MATCH")

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("7/10" in m and "above" in m for m in info_msgs)
