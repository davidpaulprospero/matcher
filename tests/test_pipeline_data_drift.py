"""
Tests for cross-stage data drift detection in pipeline orchestrator.

US-81-011: Verifies that warnings are emitted when output counts drop
below expected ratios compared to source counts, and that drift detection
is non-blocking (pipeline continues regardless).
"""

import logging
import pytest
from unittest.mock import MagicMock

from src.pipeline import PipelineOrchestrator
from src.state import PipelineState, VoiceoverSegment, Match


@pytest.fixture
def mock_config():
    """Minimal config for drift testing."""
    config = MagicMock()
    config.cache.cache_dir = ".cache"
    config.freeze = MagicMock()
    return config


@pytest.fixture
def orchestrator(mock_config, tmp_path):
    """Create a PipelineOrchestrator with no stages (for unit testing)."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = mock_config
    orch.project_dir = tmp_path
    orch.state = PipelineState()
    return orch


def _make_segments(n: int):
    return [
        VoiceoverSegment(index=i, start=float(i), end=float(i + 1), text=f"seg {i}")
        for i in range(n)
    ]


def _make_matches(n: int):
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


class TestDataDriftDetection:
    """Test _check_data_drift method."""

    @pytest.mark.fast
    def test_drift_warning_caption_results_below_threshold(self, orchestrator, caplog):
        """Warning emitted when caption_results has 5 items but video_ids has 20 (25% < 80%)."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(20)]
        orchestrator.state.caption_results = {f"vid_{i}": {"text": "hello"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "Data drift" in m and "caption_results" in m
            and "5" in m and "20" in m and "25" in m
            for m in warning_msgs
        ), f"Expected drift warning about caption_results 5/20, got: {warning_msgs}"

    @pytest.mark.fast
    def test_no_drift_warning_above_threshold(self, orchestrator, caplog):
        """No warning when caption_results count is above 80% of video_ids."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(10)]
        orchestrator.state.caption_results = {f"vid_{i}": {"text": "hello"} for i in range(9)}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Data drift" in m for m in warning_msgs), (
            f"Unexpected drift warning: {warning_msgs}"
        )

    @pytest.mark.fast
    def test_drift_warning_text_metadata_below_threshold(self, orchestrator, caplog):
        """Warning when text_metadata << video_ids after MATCH stage."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(20)]
        orchestrator.state.text_metadata = [{"video_id": f"vid_{i}"} for i in range(5)]

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("MATCH")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Data drift" in m and "text_metadata" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_drift_warning_matches_below_threshold(self, orchestrator, caplog):
        """Warning when matches << voiceover_segments after OUTPUT stage."""
        orchestrator.state.voiceover_segments = _make_segments(20)
        orchestrator.state.matches = _make_matches(5)  # 25% < 50%

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("OUTPUT")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Data drift" in m and "matches" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_no_warning_for_unrelated_stage(self, orchestrator, caplog):
        """No drift checks fire for stages with no rules (e.g., ANALYZE)."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(20)]
        orchestrator.state.caption_results = {f"vid_{0}": {"text": "hello"}}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("ANALYZE")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Data drift" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_zero_source_skips_check(self, orchestrator, caplog):
        """No crash or warning when source field is empty (can't compute ratio)."""
        orchestrator.state.video_ids = []
        orchestrator.state.caption_results = {}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Data drift" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_exact_threshold_no_warning(self, orchestrator, caplog):
        """No warning when ratio is exactly at threshold (80%)."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(10)]
        orchestrator.state.caption_results = {f"vid_{i}": {"text": "hi"} for i in range(8)}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Data drift" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_drift_rules_are_class_constant(self):
        """DRIFT_RULES is defined as a class-level constant."""
        assert hasattr(PipelineOrchestrator, 'DRIFT_RULES')
        rules = PipelineOrchestrator.DRIFT_RULES
        assert isinstance(rules, list)
        assert len(rules) >= 3
        # Each rule is a 4-tuple: (trigger_stage, source_field, target_field, min_ratio)
        for rule in rules:
            assert len(rule) == 4
            trigger, source, target, ratio = rule
            assert isinstance(trigger, str)
            assert isinstance(source, str)
            assert isinstance(target, str)
            assert isinstance(ratio, float)
            assert 0.0 < ratio <= 1.0

    @pytest.mark.fast
    def test_drift_warning_format(self, orchestrator, caplog):
        """Warning message follows exact expected format."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(20)]
        orchestrator.state.caption_results = {f"vid_{i}": {"text": "hi"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        # Check the format matches AC: 'Data drift: {target_field} has {actual} items but {source_field} has {expected} (ratio: {ratio:.1%}, threshold: {min_ratio:.0%})'
        assert any(
            m == "Data drift: caption_results has 5 items but video_ids has 20 (ratio: 25.0%, threshold: 80%)"
            for m in warning_msgs
        ), f"Got: {warning_msgs}"
