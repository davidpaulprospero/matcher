"""
Tests for cross-stage data drift detection in pipeline orchestrator.

US-81-011: Verifies that warnings are emitted when output counts drop
below expected ratios compared to source counts, and that drift detection
is non-blocking (pipeline continues regardless).

US-88-006: Tests for configurable drift rules, different threshold types,
severity levels, and drift history tracking.
"""

import logging
import pytest
from unittest.mock import MagicMock

from src.pipeline import PipelineOrchestrator
from src.state import PipelineState, VoiceoverSegment, Match
from src.config.sections.infrastructure import DriftRuleConfig, DriftRulesConfig


@pytest.fixture
def mock_config():
    """Minimal config for drift testing."""
    config = MagicMock()
    config.cache.cache_dir = ".cache"
    config.freeze = MagicMock()
    # US-88-006: Add drift_rules to config
    config.pipeline = MagicMock()
    config.pipeline.drift_rules = DriftRulesConfig()
    return config


@pytest.fixture
def orchestrator(mock_config, tmp_path):
    """Create a PipelineOrchestrator with no stages (for unit testing)."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = mock_config
    orch.project_dir = tmp_path
    orch.state = PipelineState()
    # US-88-006: Initialize drift config and history
    orch._drift_config = mock_config.pipeline.drift_rules
    orch.drift_history = []
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
        """Warning message includes expected fields."""
        orchestrator.state.video_ids = [f"vid_{i}" for i in range(20)]
        orchestrator.state.caption_results = {f"vid_{i}": {"text": "hi"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orchestrator._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        # New format includes stage name, threshold type, and remediation
        assert any(
            "Data drift" in m and "caption_results" in m and "5 items" in m and "20" in m and "25" in m
            for m in warning_msgs
        ), f"Got: {warning_msgs}"


class TestConfigurableDriftRules:
    """Tests for US-88-006: Configurable drift rules."""

    @pytest.mark.fast
    def test_absolute_count_threshold_type(self, mock_config, tmp_path, caplog):
        """Absolute count threshold triggers when actual < threshold."""
        # Set up config with absolute_count threshold
        rules_config = DriftRulesConfig()
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="absolute_count",
                threshold=15,  # Need at least 15
                severity="warning",
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        # 20 source, 10 target - below 15 absolute threshold
        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(10)}

        with caplog.at_level(logging.WARNING):
            orch._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("absolute_count" in m for m in warning_msgs), f"Got: {warning_msgs}"

    @pytest.mark.fast
    def test_percentage_threshold_type(self, mock_config, tmp_path, caplog):
        """Percentage threshold triggers when ratio < threshold%."""
        rules_config = DriftRulesConfig()
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="percentage",
                threshold=50,  # Need at least 50%
                severity="warning",
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        # 20 source, 5 target = 25% - below 50% threshold
        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orch._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Data drift" in m for m in warning_msgs), f"Got: {warning_msgs}"

    @pytest.mark.fast
    def test_error_severity_logs_error(self, mock_config, tmp_path, caplog):
        """Error severity logs at ERROR level, not WARNING."""
        rules_config = DriftRulesConfig()
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="error",  # Error severity
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        # Trigger drift
        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(5)}

        with caplog.at_level(logging.ERROR):
            orch._check_data_drift("CAPTION")

        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("Data drift" in m for m in error_msgs), f"Got: {error_msgs}"

    @pytest.mark.fast
    def test_drift_history_tracked(self, mock_config, tmp_path):
        """Drift events are tracked in drift_history."""
        rules_config = DriftRulesConfig()
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        # Trigger drift
        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(5)}

        orch._check_data_drift("CAPTION")

        # Check history
        assert len(orch.drift_history) == 1
        event = orch.drift_history[0]
        assert event['stage'] == 'CAPTION'
        assert event['source_field'] == 'video_ids'
        assert event['target_field'] == 'caption_results'
        assert event['expected'] == 20
        assert event['actual'] == 5
        assert event['severity'] == 'warning'

    @pytest.mark.fast
    def test_drift_history_respects_max_history(self, mock_config, tmp_path):
        """Drift history respects max_history limit."""
        rules_config = DriftRulesConfig()
        rules_config.max_history = 3
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        # Trigger multiple drift events
        for i in range(5):
            orch.state.video_ids = [f"vid_{j}" for j in range(20)]
            orch.state.caption_results = {f"vid_{j}": {"text": f"hi{j}"} for j in range(5)}
            orch._check_data_drift("CAPTION")

        # History should be capped at max_history
        assert len(orch.drift_history) == 3

    @pytest.mark.fast
    def test_remediation_message_in_warning(self, mock_config, tmp_path, caplog):
        """Warning includes remediation message."""
        rules_config = DriftRulesConfig()
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
                remediation="Custom: Check your API quota and retry settings.",
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orch._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Custom: Check your API quota" in m for m in warning_msgs), f"Got: {warning_msgs}"

    @pytest.mark.fast
    def test_disabled_drift_detection_no_warnings(self, mock_config, tmp_path, caplog):
        """No warnings when drift detection is globally disabled."""
        rules_config = DriftRulesConfig()
        rules_config.enabled = False
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orch._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Data drift" in m for m in warning_msgs)

    @pytest.mark.fast
    def test_disabled_rule_no_warning(self, mock_config, tmp_path, caplog):
        """No warnings for disabled rules."""
        rules_config = DriftRulesConfig()
        rules_config.rules = [
            DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
                enabled=False,  # Disabled
            )
        ]
        mock_config.pipeline.drift_rules = rules_config

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = mock_config
        orch.project_dir = tmp_path
        orch.state = PipelineState()
        orch._drift_config = rules_config
        orch.drift_history = []

        orch.state.video_ids = [f"vid_{i}" for i in range(20)]
        orch.state.caption_results = {f"vid_{i}": {"text": f"hi{i}"} for i in range(5)}

        with caplog.at_level(logging.WARNING):
            orch._check_data_drift("CAPTION")

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Data drift" in m for m in warning_msgs)
