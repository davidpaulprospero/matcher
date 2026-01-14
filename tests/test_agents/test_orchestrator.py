"""
Tests for HealingOrchestrator - coordinated self-healing pipeline execution.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import time

from src.agents.orchestrator import (
    HealingOrchestrator,
    PreflightIssue,
    EscalationRequest,
    create_orchestrated_pipeline,
)
from src.agents.strategy import HealingStrategy, HealingMode, HealingMetrics
from src.agents.base import Healer, HealerResult, HealerAction


class TestPreflightIssue:
    """Tests for PreflightIssue dataclass."""

    def test_create_issue(self):
        """Test creating a preflight issue."""
        issue = PreflightIssue(
            category="disk",
            severity="warning",
            message="Low disk space",
            auto_fixable=True,
            healer="disk-healer"
        )

        assert issue.category == "disk"
        assert issue.severity == "warning"
        assert issue.auto_fixable is True

    def test_issue_defaults(self):
        """Test default values."""
        issue = PreflightIssue(
            category="test",
            severity="info",
            message="Test message"
        )

        assert issue.auto_fixable is False
        assert issue.healer is None


class TestEscalationRequest:
    """Tests for EscalationRequest dataclass."""

    def test_create_request(self):
        """Test creating an escalation request."""
        request = EscalationRequest(
            stage_name="MATCH",
            error="Rate limit exceeded",
            options=["retry", "skip", "abort"],
            recommendation="retry"
        )

        assert request.stage_name == "MATCH"
        assert "retry" in request.options
        assert request.recommendation == "retry"


class TestHealingOrchestrator:
    """Tests for HealingOrchestrator."""

    def test_initialization(self, mock_config, project_dir):
        """Test orchestrator initializes correctly."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        assert orchestrator.config == mock_config
        assert orchestrator.project_dir == project_dir
        assert orchestrator.strategy is not None
        assert len(orchestrator.healers) > 0

    def test_initialization_with_strategy(self, mock_config, project_dir):
        """Test orchestrator uses provided strategy."""
        strategy = HealingStrategy.aggressive()
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        assert orchestrator.strategy.mode == HealingMode.AGGRESSIVE

    def test_healer_priority(self, mock_config, project_dir):
        """Test healers are ordered by strategy priority."""
        strategy = HealingStrategy()
        strategy.healer_priority = ["checkpoint-healer", "otio-healer"]

        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        healer_names = [h.name for h in orchestrator.healers]
        # Checkpoint should come before otio if both present
        if "checkpoint-healer" in healer_names and "otio-healer" in healer_names:
            assert healer_names.index("checkpoint-healer") < healer_names.index("otio-healer")

    def test_skip_healers(self, mock_config, project_dir):
        """Test healers in skip_healers are excluded."""
        strategy = HealingStrategy(skip_healers={"disk-healer"})

        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        healer_names = [h.name for h in orchestrator.healers]
        assert "disk-healer" not in healer_names


class TestPreflightChecks:
    """Tests for preflight check functionality."""

    def test_run_preflight_disabled(self, mock_config, project_dir):
        """Test preflight returns empty when disabled."""
        strategy = HealingStrategy(run_preflight=False)
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        state = Mock()
        issues = orchestrator.run_preflight(state)

        assert issues == []

    @patch('shutil.disk_usage')
    def test_check_disk_space_low(self, mock_usage, mock_config, project_dir):
        """Test disk space check detects low space."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Less than 1GB free
        mock_usage.return_value = Mock(free=500 * 1024**2)

        issues = orchestrator._check_disk_space()

        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert "disk" in issues[0].category

    @patch('shutil.disk_usage')
    def test_check_disk_space_warning(self, mock_usage, mock_config, project_dir):
        """Test disk space check warns for moderate space."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Between 1GB and 5GB
        mock_usage.return_value = Mock(free=3 * 1024**3)

        issues = orchestrator._check_disk_space()

        assert len(issues) == 1
        assert issues[0].severity == "warning"

    @patch('os.environ.get')
    def test_check_api_keys(self, mock_env, mock_config, project_dir):
        """Test API key check detects missing keys."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        mock_env.return_value = None  # No keys set

        issues = orchestrator._check_api_keys()

        # Should have at least one issue for Gemini key
        assert len(issues) >= 1
        assert any("GEMINI" in i.message for i in issues)

    def test_check_paths_missing_project(self, mock_config, tmp_path):
        """Test path check detects missing project dir."""
        nonexistent = tmp_path / "nonexistent"
        orchestrator = HealingOrchestrator(mock_config, nonexistent)

        issues = orchestrator._check_paths()

        assert len(issues) >= 1
        assert any(i.severity == "critical" for i in issues)

    def test_check_matches_missing_media(self, mock_config, project_dir, mock_matches):
        """Test match check detects missing media files."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Create state with matches pointing to nonexistent files
        state = Mock()
        match = Mock()
        match.video_path = "/nonexistent/video.mp4"
        state.matches = [match]

        issues = orchestrator._check_matches(state)

        assert len(issues) == 1
        assert "media" in issues[0].category

    def test_fix_preflight_issues(self, mock_config, project_dir):
        """Test fixing preflight issues."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        issues = [
            PreflightIssue(
                category="test",
                severity="warning",
                message="Test issue",
                auto_fixable=False
            )
        ]

        state = Mock()
        fixed, remaining = orchestrator.fix_preflight_issues(issues, state)

        assert fixed == 0
        assert remaining == 1


class TestConfigSnapshots:
    """Tests for config snapshot and rollback."""

    def test_snapshot_config(self, mock_config, project_dir):
        """Test creating config snapshot."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        snapshot = orchestrator.snapshot_config("TEST_STAGE")

        assert snapshot.stage_name == "TEST_STAGE"
        assert snapshot.timestamp > 0
        assert len(snapshot.config_values) > 0

    def test_snapshot_excludes_protected(self, mock_config, project_dir):
        """Test snapshot excludes protected keys."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        snapshot = orchestrator.snapshot_config("TEST")

        # Should not contain api_key or similar
        for key in snapshot.config_values:
            assert "api_key" not in key.lower()
            assert "password" not in key.lower()

    def test_rollback_config(self, mock_config, project_dir):
        """Test rolling back config."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Create initial snapshot
        mock_config.output.gap_mode = "scale"
        orchestrator.snapshot_config("INITIAL")

        # Modify config
        mock_config.output.gap_mode = "none"

        # Create second snapshot
        orchestrator.snapshot_config("MODIFIED")

        # Rollback to initial
        success = orchestrator.rollback_config("INITIAL")

        assert success
        assert orchestrator.metrics.rollbacks_performed == 1

    def test_rollback_disabled(self, mock_config, project_dir):
        """Test rollback when disabled in strategy."""
        strategy = HealingStrategy(enable_rollback=False)
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        orchestrator.snapshot_config("TEST")
        success = orchestrator.rollback_config()

        assert not success

    def test_rollback_no_snapshots(self, mock_config, project_dir):
        """Test rollback with no snapshots available."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        success = orchestrator.rollback_config()

        assert not success


class TestHealerCoordination:
    """Tests for healer selection and coordination."""

    def test_select_healers(self, mock_config, project_dir):
        """Test selecting healers for an error."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        error = Exception("OTIO timeline error")
        healers = orchestrator.select_healers(error, "OUTPUT")

        # Should find at least OTIO healer
        assert len(healers) > 0
        healer_names = [h.name for h in healers]
        assert "otio-healer" in healer_names

    def test_select_healers_aggressive_mode(self, mock_config, project_dir):
        """Test aggressive mode returns all applicable healers."""
        strategy = HealingStrategy.aggressive()
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        error = Exception("rate limit exceeded")
        healers = orchestrator.select_healers(error, "MATCH")

        # Aggressive mode returns all applicable
        assert len(healers) >= 1

    def test_select_healers_minimal_mode(self, mock_config, project_dir):
        """Test minimal mode returns only first healer."""
        strategy = HealingStrategy.minimal()
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        error = Exception("rate limit exceeded")
        healers = orchestrator.select_healers(error, "MATCH")

        # Minimal mode returns at most 1
        assert len(healers) <= 1

    def test_coordinate_heal_success(self, mock_config, project_dir, mock_state):
        """Test coordinated healing on success."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        error = Exception("rate limit")
        with patch.object(orchestrator.healers[0], 'can_handle', return_value=True):
            with patch.object(orchestrator.healers[0], 'fix',
                            return_value=HealerResult.fixed("Fixed")):
                result = orchestrator.coordinate_heal(error, mock_state, "TEST")

        assert result.success

    def test_coordinate_heal_always_escalate(self, mock_config, project_dir, mock_state):
        """Test immediate escalation for always_escalate patterns."""
        strategy = HealingStrategy()
        strategy.always_escalate = ["permission denied"]
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        error = Exception("Permission denied")
        result = orchestrator.coordinate_heal(error, mock_state, "TEST")

        # Should fail with escalation
        assert not result.success
        assert orchestrator.metrics.user_escalations > 0

    def test_notify_healers_on_config_change(self, mock_config, project_dir):
        """Test healer notification on config changes."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Create a result with config modification
        result = HealerResult.config_changed("Changed config")

        # Should not raise
        orchestrator._notify_healers("test-healer", result)


class TestMetricsAndReporting:
    """Tests for metrics and reporting."""

    def test_get_metrics(self, mock_config, project_dir):
        """Test getting metrics."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        metrics = orchestrator.get_metrics()

        assert isinstance(metrics, HealingMetrics)
        assert metrics.total_heals == 0

    def test_reset(self, mock_config, project_dir):
        """Test resetting orchestrator state."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Modify state
        orchestrator.metrics.total_heals = 10
        orchestrator.config_snapshots.append(Mock())

        orchestrator.reset()

        assert orchestrator.metrics.total_heals == 0
        assert len(orchestrator.config_snapshots) == 0

    def test_print_report(self, mock_config, project_dir, capsys):
        """Test print_report outputs correctly."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        orchestrator.print_report()

        captured = capsys.readouterr()
        assert "HEALING ORCHESTRATOR REPORT" in captured.out
        assert "Strategy:" in captured.out


class TestEscalation:
    """Tests for user escalation."""

    def test_escalate_to_user_with_callback(self, mock_config, project_dir):
        """Test escalation with callback."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        def callback(request: EscalationRequest) -> str:
            return "retry"

        orchestrator.escalation_callback = callback

        result = orchestrator._escalate_to_user(Exception("test"), "STAGE")

        assert result.success
        assert result.action == HealerAction.RETRY

    def test_escalate_to_user_skip(self, mock_config, project_dir):
        """Test escalation with skip decision."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        orchestrator.escalation_callback = lambda r: "skip"

        result = orchestrator._escalate_to_user(Exception("test"), "STAGE")

        assert result.success
        assert result.action == HealerAction.SKIP

    def test_escalate_to_user_abort(self, mock_config, project_dir):
        """Test escalation with abort decision."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        orchestrator.escalation_callback = lambda r: "abort"

        result = orchestrator._escalate_to_user(Exception("test"), "STAGE")

        assert not result.success

    def test_escalate_to_user_rollback(self, mock_config, project_dir):
        """Test escalation with rollback decision."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Create snapshot to rollback to
        orchestrator.snapshot_config("TEST")

        orchestrator.escalation_callback = lambda r: "rollback"

        result = orchestrator._escalate_to_user(Exception("test"), "STAGE")

        # Rollback should succeed
        assert result.success


class TestFactoryFunction:
    """Tests for create_orchestrated_pipeline factory."""

    def test_create_orchestrated_pipeline(self, mock_config, project_dir):
        """Test factory function creates all components."""
        with patch('src.pipeline.create_default_pipeline') as mock_create:
            mock_pipeline = Mock()
            mock_pipeline.state = Mock()
            mock_create.return_value = mock_pipeline

            pipeline, orchestrator, runner = create_orchestrated_pipeline(
                mock_config,
                project_dir
            )

            assert pipeline == mock_pipeline
            assert isinstance(orchestrator, HealingOrchestrator)
            assert runner.orchestrator == orchestrator

    def test_create_orchestrated_pipeline_with_strategy(self, mock_config, project_dir):
        """Test factory function uses provided strategy."""
        with patch('src.pipeline.create_default_pipeline') as mock_create:
            mock_create.return_value = Mock()
            strategy = HealingStrategy.aggressive()

            pipeline, orchestrator, runner = create_orchestrated_pipeline(
                mock_config,
                project_dir,
                strategy=strategy
            )

            assert orchestrator.strategy.mode == HealingMode.AGGRESSIVE
