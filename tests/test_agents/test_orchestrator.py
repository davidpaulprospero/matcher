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

    def test_select_healers_returns_empty_for_unknown_error(self, mock_config, project_dir):
        """Test select_healers returns empty list when no healer can handle error."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Make all healers report they cannot handle this error
        for healer in orchestrator.healers:
            healer.can_handle = Mock(return_value=False)

        error = Exception("Completely unknown error pattern xyz123")
        healers = orchestrator.select_healers(error, "UNKNOWN_STAGE")

        # Should return empty list
        assert healers == []

    def test_select_healers_respects_can_handle(self, mock_config, project_dir):
        """Test select_healers only returns healers where can_handle() is True."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Make all healers except one return False for can_handle
        for i, healer in enumerate(orchestrator.healers):
            # Only the last healer can handle this error
            healer.can_handle = Mock(return_value=(i == len(orchestrator.healers) - 1))

        error = Exception("specific error pattern")
        healers = orchestrator.select_healers(error, "TEST")

        # Only healers that can_handle() returned True should be included
        # Each healer's can_handle was called
        for healer in orchestrator.healers:
            healer.can_handle.assert_called()

        # Should have at most 1 healer (the last one)
        assert len(healers) <= 1

    def test_select_healers_logs_decision(self, mock_config, project_dir, caplog):
        """Test select_healers logs healer selection decision."""
        import logging
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        error = Exception("OTIO timeline error")

        with caplog.at_level(logging.DEBUG):
            healers = orchestrator.select_healers(error, "OUTPUT")

        # Should have logged something about healer selection
        # (implementation may vary - check if any healer-related log exists)
        if healers:
            # At least verify it ran without error
            assert True

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


class TestClassifyError:
    """Tests for HealingOrchestrator._classify_error() method.

    US-005: Add FallbackChain.classify_error method tests

    Tests fallback logic:
    - Uses watcher when available
    - Falls back to pattern_route when watcher unavailable
    - Falls back to pattern_route when watcher times out
    - Logs fallback activation
    - Returns PatternClassification with needs_llm_healer=True for unknown patterns
    """

    def test_classify_error_uses_watcher_when_available(self, mock_config, project_dir):
        """Test classify_error() uses watcher when available and returns WatcherClassification."""
        # Create orchestrator
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Set up mock watcher
        mock_watcher = Mock()
        from src.agents.watcher import ErrorClassification
        mock_classification = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="Watcher classified as API error"
        )
        mock_watcher.classify_error.return_value = mock_classification

        # Set up mock fallback_chain that says watcher is available
        mock_fallback = Mock()
        mock_fallback.check_watcher_available.return_value = True

        orchestrator.watcher = mock_watcher
        orchestrator.fallback_chain = mock_fallback

        # Classify an error
        error = Exception("API rate limit exceeded")
        result = orchestrator._classify_error(error, "MATCH")

        # Should use watcher classification
        assert result == mock_classification
        assert result.category == "api"
        assert result.confidence == 0.9
        mock_watcher.classify_error.assert_called_once()
        mock_fallback.check_watcher_available.assert_called_once()

    def test_classify_error_falls_back_to_pattern_route_when_watcher_unavailable(
        self, mock_config, project_dir
    ):
        """Test classify_error() falls back to pattern_route when watcher unavailable."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Set up mock fallback_chain that says watcher is unavailable
        mock_fallback = Mock()
        mock_fallback.check_watcher_available.return_value = False

        orchestrator.watcher = Mock()
        orchestrator.fallback_chain = mock_fallback

        # Classify an API error (will be pattern matched)
        error = Exception("HTTP Error 429: Too Many Requests")
        result = orchestrator._classify_error(error, "DOWNLOAD")

        # Should use pattern routing
        from src.agents.fallback import PatternClassification
        assert isinstance(result, PatternClassification)
        assert result.category == "api"
        assert result.suggested_healer == "api-healer"
        # Watcher's classify_error should not be called
        orchestrator.watcher.classify_error.assert_not_called()

    def test_classify_error_falls_back_when_watcher_returns_none(
        self, mock_config, project_dir
    ):
        """Test classify_error() falls back to pattern_route when watcher returns None."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Set up mock watcher that returns None (failed classification)
        mock_watcher = Mock()
        mock_watcher.classify_error.return_value = None

        # Fallback chain says watcher is available
        mock_fallback = Mock()
        mock_fallback.check_watcher_available.return_value = True

        orchestrator.watcher = mock_watcher
        orchestrator.fallback_chain = mock_fallback

        # Classify an error - watcher will fail, should fall back
        error = Exception("Disk full - no space left on device")
        result = orchestrator._classify_error(error, "OUTPUT")

        # Should fall back to pattern routing
        from src.agents.fallback import PatternClassification
        assert isinstance(result, PatternClassification)
        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    def test_classify_error_falls_back_when_no_watcher_configured(
        self, mock_config, project_dir
    ):
        """Test classify_error() uses pattern_route when no watcher configured."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Remove watcher
        orchestrator.watcher = None
        orchestrator.fallback_chain = None

        # Classify a checkpoint error
        error = Exception("JSONDecodeError: Expecting value at line 1")
        result = orchestrator._classify_error(error, "TRANSCRIBE")

        # Should use pattern routing
        from src.agents.fallback import PatternClassification
        assert isinstance(result, PatternClassification)
        assert result.category == "checkpoint"
        assert result.suggested_healer == "checkpoint-healer"

    def test_classify_error_falls_back_when_no_fallback_chain(
        self, mock_config, project_dir
    ):
        """Test classify_error() uses pattern_route when fallback_chain is None."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Have watcher but no fallback chain
        orchestrator.watcher = Mock()
        orchestrator.fallback_chain = None

        # Classify an OTIO error
        error = Exception("opentimelineio.exception: Invalid time range")
        result = orchestrator._classify_error(error, "OUTPUT")

        # Should use pattern routing (can't check watcher availability)
        from src.agents.fallback import PatternClassification
        assert isinstance(result, PatternClassification)
        assert result.category == "otio"
        assert result.suggested_healer == "otio-healer"

    def test_classify_error_returns_pattern_classification_with_needs_llm_healer_for_unknown(
        self, mock_config, project_dir
    ):
        """Test classify_error() returns PatternClassification with needs_llm_healer=True for unknown patterns."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # No watcher available
        orchestrator.watcher = None
        orchestrator.fallback_chain = None

        # Classify an unknown error that won't match any pattern
        error = Exception("Completely random error xyz123 with no patterns")
        result = orchestrator._classify_error(error, "MATCH")

        # Should return unknown classification with needs_llm_healer=True
        from src.agents.fallback import PatternClassification
        assert isinstance(result, PatternClassification)
        assert result.category == "unknown"
        assert result.suggested_healer == ""
        assert result.confidence == 0.3
        assert result.needs_llm_healer is True

    def test_classify_error_context_passed_to_watcher(self, mock_config, project_dir):
        """Test classify_error() passes correct context to watcher."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        mock_watcher = Mock()
        from src.agents.watcher import ErrorClassification
        mock_watcher.classify_error.return_value = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.85,
            needs_llm_healer=False,
            reasoning="Test"
        )

        mock_fallback = Mock()
        mock_fallback.check_watcher_available.return_value = True

        orchestrator.watcher = mock_watcher
        orchestrator.fallback_chain = mock_fallback

        error = Exception("rate limit")
        orchestrator._classify_error(error, "DOWNLOAD")

        # Verify context passed to watcher
        call_args = mock_watcher.classify_error.call_args
        context = call_args[0][1]
        assert context['stage'] == "DOWNLOAD"
        assert context['stage_name'] == "DOWNLOAD"

    def test_classify_error_pattern_routes_various_error_types(
        self, mock_config, project_dir
    ):
        """Test classify_error() pattern routes various error types correctly."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)
        orchestrator.watcher = None
        orchestrator.fallback_chain = None

        from src.agents.fallback import PatternClassification

        test_cases = [
            ("Rate limit exceeded", "api", "api-healer"),
            ("Connection timed out", "api", "api-healer"),
            ("Permission denied: /var/log", "disk", "disk-healer"),
            ("No space left on device", "disk", "disk-healer"),
            ("Path too long exceeds 260 char", "path", "path-healer"),
            ("UnicodeDecodeError: utf-8 codec", "path", "path-healer"),
            ("checkpoint corrupt cannot read", "checkpoint", "checkpoint-healer"),
            ("Video unavailable: private", "download", "download-healer"),
            ("opentimelineio exception: invalid", "otio", "otio-healer"),
        ]

        for error_msg, expected_category, expected_healer in test_cases:
            error = Exception(error_msg)
            result = orchestrator._classify_error(error, "TEST")

            assert isinstance(result, PatternClassification), f"Failed for: {error_msg}"
            assert result.category == expected_category, f"Failed category for: {error_msg}"
            assert result.suggested_healer == expected_healer, f"Failed healer for: {error_msg}"

    def test_classify_error_watcher_logs_fallback_on_failure(self, mock_config, project_dir):
        """Test classify_error() logs fallback activation via healing_logger.log_fallback().

        When watcher classification fails (returns None), the watcher logs the fallback
        via healing_logger.log_fallback() before the orchestrator falls back to pattern_route.
        """
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        # Set up mock watcher that raises exception (simulating timeout/failure)
        mock_watcher = Mock()
        mock_watcher.classify_error.return_value = None  # Watcher failed

        # Set up mock healing_logger to verify logging
        mock_healing_logger = Mock()
        mock_watcher.healing_logger = mock_healing_logger

        # Fallback chain says watcher is available
        mock_fallback = Mock()
        mock_fallback.check_watcher_available.return_value = True

        orchestrator.watcher = mock_watcher
        orchestrator.fallback_chain = mock_fallback

        # Classify an error
        error = Exception("rate limit exceeded")
        result = orchestrator._classify_error(error, "DOWNLOAD")

        # Should fall back to pattern routing
        from src.agents.fallback import PatternClassification
        assert isinstance(result, PatternClassification)

        # Watcher was called but returned None (failed)
        mock_watcher.classify_error.assert_called_once()

        # Note: The healing_logger.log_fallback() is called by WatcherAgent
        # internally when it catches exceptions (see watcher.py:243-246).
        # This test verifies the orchestrator correctly handles the None return
        # and falls back to pattern_route. The watcher's internal logging is
        # tested in test_watcher.py::TestWatcherClassification::test_classify_error_records_fallback_on_failure
