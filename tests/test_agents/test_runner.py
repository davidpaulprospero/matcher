"""
Tests for ResilientRunner - self-healing pipeline executor.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import time

from src.agents.runner import (
    ResilientRunner,
    create_resilient_pipeline,
    create_orchestrated_pipeline,
)
from src.agents.base import Healer, HealerResult, HealerAction
from src.agents.strategy import HealingStrategy, HealingMode
from src.agents.orchestrator import HealingOrchestrator


class TestResilientRunnerInit:
    """Tests for ResilientRunner initialization."""

    def test_initialization(self, mock_config, project_dir):
        """Test runner initializes correctly."""
        runner = ResilientRunner(mock_config, project_dir)

        assert runner.config == mock_config
        assert runner.project_dir == project_dir
        assert len(runner.healers) > 0
        assert runner.orchestrator is None

    def test_initialization_with_orchestrator(self, mock_config, project_dir):
        """Test runner uses orchestrator healers when provided."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        assert runner.orchestrator == orchestrator
        assert runner.healers == orchestrator.healers

    def test_initialization_with_custom_healers(self, mock_config, project_dir):
        """Test runner with custom healer list."""
        class CustomHealer(Healer):
            name = "custom-healer"
            description = "Test healer"
            error_patterns = ["custom"]

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Custom fix")

        runner = ResilientRunner(mock_config, project_dir, healers=[CustomHealer])

        assert len(runner.healers) == 1
        assert runner.healers[0].name == "custom-healer"

    def test_max_attempts_from_orchestrator(self, mock_config, project_dir):
        """Test MAX_HEAL_ATTEMPTS comes from orchestrator strategy."""
        strategy = HealingStrategy(max_attempts_per_stage=5)
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        assert runner.MAX_HEAL_ATTEMPTS == 5


class TestRunPipeline:
    """Tests for run_pipeline method."""

    def test_run_pipeline_success(self, mock_config, project_dir, mock_stage):
        """Test successful pipeline run."""
        runner = ResilientRunner(mock_config, project_dir)

        # Create mock pipeline
        pipeline = Mock()
        pipeline.stages = [mock_stage]
        pipeline.state = Mock()
        pipeline.state.stage_timings = {}
        pipeline.checkpoint = Mock()
        pipeline.config = mock_config
        pipeline.resume_mode = False
        pipeline.stage_timings = {}

        # Mock stage behavior
        from src.stages import StageResult
        mock_stage.can_skip.return_value = False
        mock_stage.validate_inputs.return_value = None
        mock_stage.run.return_value = StageResult.ok({"result": True})

        result = runner.run_pipeline(pipeline, resume=False)

        assert result is True

    def test_run_pipeline_with_preflight(self, mock_config, project_dir, mock_stage):
        """Test pipeline runs preflight when orchestrator present."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        pipeline = Mock()
        pipeline.stages = []
        pipeline.state = Mock()
        pipeline.state.stage_timings = {}
        pipeline.state.matches = None
        pipeline.checkpoint = Mock()

        with patch.object(orchestrator, 'run_preflight', return_value=[]) as mock_preflight:
            runner.run_pipeline(pipeline, resume=False)

        mock_preflight.assert_called_once()

    def test_run_pipeline_preflight_critical_blocks(self, mock_config, project_dir):
        """Test pipeline stops on unfixable critical preflight issues."""
        from src.agents.orchestrator import PreflightIssue

        orchestrator = HealingOrchestrator(mock_config, project_dir)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        pipeline = Mock()
        pipeline.stages = []
        pipeline.state = Mock()
        pipeline.state.matches = None

        critical_issue = PreflightIssue(
            category="disk",
            severity="critical",
            message="Critical error",
            auto_fixable=False
        )

        with patch.object(orchestrator, 'run_preflight', return_value=[critical_issue]):
            with patch.object(orchestrator, 'fix_preflight_issues', return_value=(0, 1)):
                result = runner.run_pipeline(pipeline, resume=False)

        assert result is False


class TestRunStage:
    """Tests for run_stage method."""

    def test_run_stage_success(self, mock_config, project_dir, mock_stage, mock_state):
        """Test successful stage execution."""
        runner = ResilientRunner(mock_config, project_dir)

        from src.stages import StageResult
        mock_stage.run.return_value = StageResult.ok({"done": True})

        result = runner.run_stage(
            mock_stage,
            mock_state,
            mock_config,
            Mock()  # checkpoint
        )

        assert result.success

    def test_run_stage_failure_with_heal(self, mock_config, project_dir, mock_stage, mock_state):
        """Test stage failure triggers healing."""
        runner = ResilientRunner(mock_config, project_dir)

        from src.stages import StageResult

        # First call fails, second succeeds
        mock_stage.run.side_effect = [
            StageResult.fail("test error"),
            StageResult.ok({})
        ]

        # Mock healer
        with patch.object(runner, '_try_heal', return_value=True) as mock_heal:
            with patch('time.sleep'):  # Skip delay
                result = runner.run_stage(
                    mock_stage,
                    mock_state,
                    mock_config,
                    Mock()
                )

        mock_heal.assert_called()
        assert result.success

    def test_run_stage_exception_triggers_heal(self, mock_config, project_dir, mock_stage, mock_state):
        """Test stage exception triggers healing."""
        runner = ResilientRunner(mock_config, project_dir)

        from src.stages import StageResult

        # First call raises, second succeeds
        mock_stage.run.side_effect = [
            Exception("test exception"),
            StageResult.ok({})
        ]

        with patch.object(runner, '_try_heal', return_value=True):
            with patch('time.sleep'):
                result = runner.run_stage(
                    mock_stage,
                    mock_state,
                    mock_config,
                    Mock()
                )

        assert result.success

    def test_run_stage_max_attempts(self, mock_config, project_dir, mock_stage, mock_state):
        """Test stage fails after max attempts."""
        runner = ResilientRunner(mock_config, project_dir)
        runner.MAX_HEAL_ATTEMPTS = 2

        from src.stages import StageResult
        mock_stage.run.return_value = StageResult.fail("always fails")

        with patch.object(runner, '_try_heal', return_value=True):
            with patch('time.sleep'):
                result = runner.run_stage(
                    mock_stage,
                    mock_state,
                    mock_config,
                    Mock()
                )

        assert not result.success
        assert "failed after" in result.error.lower()


class TestTryHeal:
    """Tests for _try_heal method."""

    def test_try_heal_with_orchestrator(self, mock_config, project_dir, mock_state):
        """Test _try_heal uses orchestrator when available."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        with patch.object(orchestrator, 'coordinate_heal',
                         return_value=HealerResult.fixed("Fixed")) as mock_coord:
            result = runner._try_heal(Exception("test"), mock_state, "TEST")

        mock_coord.assert_called_once()
        assert result is True

    def test_try_heal_direct_healer(self, mock_config, project_dir, mock_state):
        """Test _try_heal uses direct healers without orchestrator."""
        runner = ResilientRunner(mock_config, project_dir)

        # Mock healer that can handle the error
        mock_healer = Mock()
        mock_healer.can_handle.return_value = True
        mock_healer.fix.return_value = HealerResult.fixed("Fixed", action=HealerAction.RETRY)
        mock_healer.name = "mock-healer"

        runner.healers = [mock_healer]

        result = runner._try_heal(Exception("test"), mock_state, "TEST")

        mock_healer.fix.assert_called_once()
        assert result is True
        assert runner.total_heals == 1

    def test_try_heal_no_applicable_healer(self, mock_config, project_dir, mock_state):
        """Test _try_heal returns False when no healer can handle."""
        runner = ResilientRunner(mock_config, project_dir)

        # Mock healer that can't handle
        mock_healer = Mock()
        mock_healer.can_handle.return_value = False

        runner.healers = [mock_healer]

        result = runner._try_heal(Exception("unusual error"), mock_state, "TEST")

        assert result is False

    def test_try_heal_action_skip(self, mock_config, project_dir, mock_state):
        """Test _try_heal returns False for SKIP action."""
        runner = ResilientRunner(mock_config, project_dir)

        mock_healer = Mock()
        mock_healer.can_handle.return_value = True
        mock_healer.fix.return_value = HealerResult.fixed("Skipping", action=HealerAction.SKIP)
        mock_healer.name = "mock-healer"

        runner.healers = [mock_healer]

        result = runner._try_heal(Exception("test"), mock_state, "TEST")

        assert result is False  # SKIP means don't retry

    def test_try_heal_action_abort(self, mock_config, project_dir, mock_state):
        """Test _try_heal returns False for ABORT action."""
        runner = ResilientRunner(mock_config, project_dir)

        mock_healer = Mock()
        mock_healer.can_handle.return_value = True
        mock_healer.fix.return_value = HealerResult.failed("Cannot fix")
        mock_healer.name = "mock-healer"

        runner.healers = [mock_healer]

        result = runner._try_heal(Exception("test"), mock_state, "TEST")

        assert result is False

    def test_try_heal_records_history(self, mock_config, project_dir, mock_state):
        """Test _try_heal records heal attempts."""
        runner = ResilientRunner(mock_config, project_dir)

        mock_healer = Mock()
        mock_healer.can_handle.return_value = True
        mock_healer.fix.return_value = HealerResult.fixed("Fixed", action=HealerAction.RETRY)
        mock_healer.name = "test-healer"

        runner.healers = [mock_healer]

        runner._try_heal(Exception("test error"), mock_state, "TEST_STAGE")

        assert len(runner.heal_history) == 1
        assert runner.heal_history[0]["stage"] == "TEST_STAGE"
        assert runner.heal_history[0]["healer"] == "test-healer"


class TestSummary:
    """Tests for summary methods."""

    def test_get_summary(self, mock_config, project_dir):
        """Test get_summary returns correct data."""
        runner = ResilientRunner(mock_config, project_dir)

        runner.total_heals = 5
        runner.stage_attempts = {"TEST": 2}
        runner.heal_history.append({"stage": "TEST", "success": True})

        summary = runner.get_summary()

        assert summary["total_heals"] == 5
        assert summary["stage_attempts"]["TEST"] == 2
        assert len(summary["heal_history"]) == 1

    def test_get_summary_with_orchestrator(self, mock_config, project_dir):
        """Test get_summary includes orchestrator metrics."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        summary = runner.get_summary()

        assert "orchestrator_metrics" in summary

    def test_print_summary_no_heals(self, mock_config, project_dir, capsys):
        """Test print_summary with no healing needed."""
        runner = ResilientRunner(mock_config, project_dir)

        runner.print_summary()

        captured = capsys.readouterr()
        assert "No healing was needed" in captured.out

    def test_print_summary_with_heals(self, mock_config, project_dir, capsys):
        """Test print_summary with healing history."""
        runner = ResilientRunner(mock_config, project_dir)

        runner.heal_history.append({
            "stage": "TEST",
            "healer": "test-healer",
            "success": True,
            "message": "Fixed the issue"
        })
        runner.total_heals = 1

        runner.print_summary()

        captured = capsys.readouterr()
        assert "Healing Summary" in captured.out
        assert "TEST" in captured.out


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_resilient_pipeline(self, mock_config, project_dir):
        """Test create_resilient_pipeline factory."""
        with patch('src.pipeline.create_default_pipeline') as mock_create:
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline

            pipeline, runner = create_resilient_pipeline(mock_config, project_dir)

            assert pipeline == mock_pipeline
            assert isinstance(runner, ResilientRunner)

    def test_create_resilient_pipeline_with_strategy(self, mock_config, project_dir):
        """Test create_resilient_pipeline with strategy creates orchestrator."""
        with patch('src.pipeline.create_default_pipeline') as mock_create:
            mock_create.return_value = Mock()
            strategy = HealingStrategy.aggressive()

            pipeline, runner = create_resilient_pipeline(
                mock_config,
                project_dir,
                strategy=strategy
            )

            assert runner.orchestrator is not None
            assert runner.orchestrator.strategy.mode == HealingMode.AGGRESSIVE

    def test_create_orchestrated_pipeline(self, mock_config, project_dir):
        """Test create_orchestrated_pipeline factory."""
        with patch('src.pipeline.create_default_pipeline') as mock_create:
            mock_create.return_value = Mock()

            pipeline, orchestrator, runner = create_orchestrated_pipeline(
                mock_config,
                project_dir
            )

            assert isinstance(orchestrator, HealingOrchestrator)
            assert isinstance(runner, ResilientRunner)
            assert runner.orchestrator == orchestrator


class TestResetHealers:
    """Tests for healer reset functionality."""

    def test_reset_healers(self, mock_config, project_dir):
        """Test _reset_healers calls reset_backoff on healers."""
        runner = ResilientRunner(mock_config, project_dir)

        mock_healer = Mock()
        mock_healer.reset_backoff = Mock()

        runner.healers = [mock_healer]

        runner._reset_healers()

        mock_healer.reset_backoff.assert_called_once()

    def test_reset_healers_missing_method(self, mock_config, project_dir):
        """Test _reset_healers handles healers without reset_backoff."""
        runner = ResilientRunner(mock_config, project_dir)

        mock_healer = Mock(spec=[])  # No reset_backoff

        runner.healers = [mock_healer]

        # Should not raise
        runner._reset_healers()
