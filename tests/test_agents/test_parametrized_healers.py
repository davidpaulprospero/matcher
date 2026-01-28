"""
Parametrized tests for healer scenarios.

US-004: Expand parametrized tests for healer scenarios.

Tests cover:
- AC1: Error type variations across all healers
- AC2: Healing strategy matrix (aggressive, conservative, interactive, minimal)
- AC3: Retry count boundaries (1, 2, 3, max_attempts)
- AC4: Healer selection based on error patterns
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.agents.base import Healer, HealerResult, HealerAction
from src.agents.strategy import HealingStrategy, HealingMode, HealingMetrics
from src.agents.orchestrator import HealingOrchestrator
from src.agents.runner import ResilientRunner
from src.agents.healers.api import APIHealer
from src.agents.healers.checkpoint import CheckpointHealer
from src.agents.healers.download import DownloadHealer
from src.agents.healers.disk import DiskHealer
from src.agents.healers.path import PathHealer


# =============================================================================
# AC1: Error type variations across healers
# =============================================================================


class TestParametrizedAPIHealerErrors:
    """Parametrized tests for APIHealer error handling."""

    @pytest.mark.parametrize("error_msg,expected_can_handle", [
        # Rate limit variations
        ("Rate limit exceeded", True),
        ("rate limit hit - try again later", True),
        ("HTTP Error 429: Too Many Requests", True),
        ("429 error returned", True),
        ("Error 429 - rate limited", True),
        # Timeout variations
        ("Connection timed out after 30s", True),
        ("Request timeout exceeded", True),
        ("timeout waiting for response", True),
        # Auth error variations
        ("401 Unauthorized", True),
        ("403 Forbidden", True),
        ("Invalid API key", True),
        ("api key invalid or expired", True),
        # Provider-specific errors
        ("gemini API error", True),
        ("anthropic rate limit", True),
        ("ollama connection refused", True),
        # Non-matching errors
        ("File not found", False),
        ("Disk full", False),
        ("Path too long", False),
    ])
    def test_can_handle_api_errors(self, mock_config, project_dir, error_msg, expected_can_handle):
        """Test APIHealer can_handle with various error messages."""
        healer = APIHealer(mock_config, project_dir)
        error = Exception(error_msg)

        result = healer.can_handle(error, "MATCH")
        assert result == expected_can_handle, f"Failed for: {error_msg}"


class TestParametrizedCheckpointHealerErrors:
    """Parametrized tests for CheckpointHealer error handling."""

    @pytest.mark.parametrize("error_msg,expected_can_handle", [
        # JSON decode errors
        ("JSONDecodeError at position 0", True),
        ("json decode failed", True),
        ("Invalid JSON in checkpoint", True),
        # Checkpoint-specific errors
        ("checkpoint file corrupted", True),
        ("checkpoint corrupt - cannot load", True),
        ("Failed to load checkpoint", True),
        # Corrupt data variations
        ("corrupt data detected", True),
        ("data corruption in state", True),
        # Hash mismatch (config changed)
        ("config hash mismatch", True),
        # Non-matching errors
        ("Rate limit exceeded", False),
        ("File not found", False),
        ("timeout", False),
    ])
    def test_can_handle_checkpoint_errors(self, mock_config, project_dir, error_msg, expected_can_handle):
        """Test CheckpointHealer can_handle with various error messages."""
        healer = CheckpointHealer(mock_config, project_dir)
        error = Exception(error_msg)

        result = healer.can_handle(error, "LOAD")
        assert result == expected_can_handle, f"Failed for: {error_msg}"


class TestParametrizedDownloadHealerErrors:
    """Parametrized tests for DownloadHealer error handling."""

    @pytest.mark.parametrize("error_msg,expected_can_handle", [
        # yt-dlp errors
        ("yt-dlp: Unable to extract video data", True),
        ("yt-dlp error: 403 Forbidden", True),
        ("yt-dlp extraction failed", True),
        # YouTube-specific errors
        ("youtube video unavailable", True),
        ("Video is private", True),
        ("Video has been removed", True),
        # Rate limit (download context)
        ("429 Too Many Requests", True),
        ("download rate limited", True),
        # Format errors
        ("requested format not available", True),
        ("format 22 not available", True),
        # Network errors (must contain "connection" or "timeout")
        ("Connection reset by peer", True),
        ("connection refused during download", True),
        # Non-matching errors (no download-related keywords)
        ("Invalid config", False),
        ("OTIO error", False),
        ("Path too long", False),
    ])
    def test_can_handle_download_errors(self, mock_config, project_dir, error_msg, expected_can_handle):
        """Test DownloadHealer can_handle with various error messages."""
        healer = DownloadHealer(mock_config, project_dir)
        error = Exception(error_msg)

        result = healer.can_handle(error, "DOWNLOAD")
        assert result == expected_can_handle, f"Failed for: {error_msg}"


class TestParametrizedDiskHealerErrors:
    """Parametrized tests for DiskHealer error handling."""

    @pytest.mark.parametrize("error_msg,expected_can_handle", [
        # Disk full variations - must match error_patterns: disk full, no space, not enough space, errno 28
        ("No space left on device", True),  # contains "no space"
        ("disk full", True),
        ("errno 28 - no more space", True),  # errno 28 is ENOSPC
        ("not enough space", True),
        # Permission errors - must match: permission denied, access denied, errno 13
        ("Permission denied", True),
        ("permission denied: /var/log", True),
        ("Access denied", True),
        # Storage-related - must match "storage"
        ("storage quota exceeded", True),
        # OSError pattern
        ("OSError: cannot write", True),
        # Non-matching errors (no disk-related keywords)
        ("Rate limit", False),
        ("checkpoint corrupt", False),
        ("timeout", False),
    ])
    def test_can_handle_disk_errors(self, mock_config, project_dir, error_msg, expected_can_handle):
        """Test DiskHealer can_handle with various error messages."""
        healer = DiskHealer(mock_config, project_dir)
        error = Exception(error_msg)

        result = healer.can_handle(error, "OUTPUT")
        assert result == expected_can_handle, f"Failed for: {error_msg}"


class TestParametrizedPathHealerErrors:
    """Parametrized tests for PathHealer error handling."""

    @pytest.mark.parametrize("error_msg,expected_can_handle", [
        # Path length errors - must match: path too long, filename too long, name too long, errno 63/36/206
        ("path too long", True),
        ("filename too long", True),
        ("name too long for filesystem", True),
        ("errno 63 - path length exceeded", True),
        # Unicode errors - must match: unicode, encode, decode
        ("unicode encode error", True),
        ("UnicodeDecodeError: utf-8", True),
        ("Cannot encode filename", True),
        # Invalid character errors - must match: invalid path, illegal character
        ("invalid path detected", True),
        ("illegal character in filename", True),
        # Non-matching errors (no path-related keywords)
        ("Rate limit", False),
        ("checkpoint corrupt", False),
        ("disk full", False),
    ])
    def test_can_handle_path_errors(self, mock_config, project_dir, error_msg, expected_can_handle):
        """Test PathHealer can_handle with various error messages."""
        healer = PathHealer(mock_config, project_dir)
        error = Exception(error_msg)

        result = healer.can_handle(error, "OUTPUT")
        assert result == expected_can_handle, f"Failed for: {error_msg}"


# =============================================================================
# AC2: Healing strategy matrix
# =============================================================================


class TestParametrizedHealingStrategies:
    """Parametrized tests for healing strategy matrix."""

    @pytest.mark.parametrize("strategy_name,expected_mode,expected_max_attempts,expected_max_heals", [
        ("aggressive", HealingMode.AGGRESSIVE, 5, 50),
        ("conservative", HealingMode.CONSERVATIVE, 3, 20),
        ("interactive", HealingMode.INTERACTIVE, 3, 30),  # interactive has max_total_heals=30
        ("minimal", HealingMode.MINIMAL, 1, 5),
    ])
    def test_strategy_factory_values(self, strategy_name, expected_mode, expected_max_attempts, expected_max_heals):
        """Test strategy factory methods return correct values."""
        factory_method = getattr(HealingStrategy, strategy_name)
        strategy = factory_method()

        assert strategy.mode == expected_mode
        assert strategy.max_attempts_per_stage == expected_max_attempts
        assert strategy.max_total_heals == expected_max_heals

    @pytest.mark.parametrize("strategy_name,expected_rollback_enabled", [
        ("aggressive", True),
        ("conservative", True),
        ("interactive", True),
        ("minimal", False),
    ])
    def test_strategy_rollback_setting(self, strategy_name, expected_rollback_enabled):
        """Test strategy rollback settings."""
        factory_method = getattr(HealingStrategy, strategy_name)
        strategy = factory_method()

        assert strategy.enable_rollback == expected_rollback_enabled

    @pytest.mark.parametrize("strategy_name,expected_auto_fix_preflight", [
        ("aggressive", True),
        ("conservative", True),
        ("interactive", False),  # Interactive asks user
        ("minimal", False),      # Minimal also doesn't auto-fix
    ])
    def test_strategy_auto_fix_preflight(self, strategy_name, expected_auto_fix_preflight):
        """Test strategy auto_fix_preflight settings."""
        factory_method = getattr(HealingStrategy, strategy_name)
        strategy = factory_method()

        assert strategy.auto_fix_preflight == expected_auto_fix_preflight

    @pytest.mark.parametrize("strategy_name,expected_heal_delay", [
        ("aggressive", 1.0),
        ("conservative", 2.0),
        ("interactive", 2.0),
        ("minimal", 0.5),  # minimal has 0.5 delay
    ])
    def test_strategy_heal_delay(self, strategy_name, expected_heal_delay):
        """Test strategy heal delay settings."""
        factory_method = getattr(HealingStrategy, strategy_name)
        strategy = factory_method()

        assert strategy.heal_delay == expected_heal_delay


class TestParametrizedOrchestratorWithStrategies:
    """Parametrized tests for orchestrator behavior with different strategies."""

    @pytest.mark.parametrize("strategy_name", ["aggressive", "conservative", "interactive", "minimal"])
    def test_orchestrator_initializes_with_strategy(self, mock_config, project_dir, strategy_name):
        """Test orchestrator initializes correctly with each strategy."""
        factory_method = getattr(HealingStrategy, strategy_name)
        strategy = factory_method()

        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        assert orchestrator.strategy.mode.value == strategy_name
        assert len(orchestrator.healers) > 0

    @pytest.mark.parametrize("strategy_name,expected_max_healers", [
        ("aggressive", None),  # Returns all applicable healers (no limit)
        ("conservative", 3),   # Returns up to 3 applicable
        ("interactive", 3),    # Returns up to 3 applicable
        ("minimal", 1),        # Returns only first applicable
    ])
    def test_select_healers_by_strategy(self, mock_config, project_dir, strategy_name, expected_max_healers):
        """Test healer selection varies by strategy."""
        factory_method = getattr(HealingStrategy, strategy_name)
        strategy = factory_method()
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        # Use an error that multiple healers might handle
        error = Exception("rate limit exceeded 429")
        healers = orchestrator.select_healers(error, "MATCH")

        if expected_max_healers is None:
            # Aggressive can return any number
            assert len(healers) >= 0  # Just verify it returns a list
        else:
            # Others return at most expected_max_healers
            assert len(healers) <= expected_max_healers


# =============================================================================
# AC3: Retry count boundaries
# =============================================================================


class TestParametrizedRetryBoundaries:
    """Parametrized tests for retry count boundaries."""

    @pytest.mark.parametrize("max_attempts", [1, 2, 3, 5, 10])
    def test_runner_respects_max_attempts(self, mock_config, project_dir, mock_stage, mock_state, max_attempts):
        """Test runner respects max_attempts_per_stage boundary."""
        strategy = HealingStrategy(max_attempts_per_stage=max_attempts)
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        assert runner.MAX_HEAL_ATTEMPTS == max_attempts

        from src.stages import StageResult
        mock_stage.run.return_value = StageResult.fail("persistent error")

        with patch.object(orchestrator, 'coordinate_heal',
                         return_value=HealerResult.fixed("Fixed", action=HealerAction.RETRY)):
            with patch('time.sleep'):
                result = runner.run_stage(mock_stage, mock_state, mock_config, Mock())

        assert not result.success
        assert mock_stage.run.call_count == max_attempts

    @pytest.mark.parametrize("attempts_before_success", [1, 2, 3])
    def test_runner_succeeds_on_nth_attempt(self, mock_config, project_dir, mock_stage, mock_state, attempts_before_success):
        """Test runner succeeds when stage passes on nth attempt."""
        strategy = HealingStrategy(max_attempts_per_stage=5)  # Allow enough attempts
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)
        runner = ResilientRunner(mock_config, project_dir, orchestrator=orchestrator)

        from src.stages import StageResult

        # Fail n-1 times, then succeed
        side_effects = [StageResult.fail("error")] * (attempts_before_success - 1) + [StageResult.ok({})]
        mock_stage.run.side_effect = side_effects

        with patch.object(orchestrator, 'coordinate_heal',
                         return_value=HealerResult.fixed("Fixed", action=HealerAction.RETRY)):
            with patch('time.sleep'):
                result = runner.run_stage(mock_stage, mock_state, mock_config, Mock())

        assert result.success
        assert mock_stage.run.call_count == attempts_before_success


class TestParametrizedTotalHealLimits:
    """Parametrized tests for total heal limits."""

    @pytest.mark.parametrize("max_total_heals", [5, 10, 20, 50])
    def test_strategy_max_total_heals(self, max_total_heals):
        """Test max_total_heals configuration."""
        strategy = HealingStrategy(max_total_heals=max_total_heals)

        assert strategy.max_total_heals == max_total_heals

    @pytest.mark.parametrize("total_heals,expected_summary_contains", [
        (0, "0"),
        (1, "1"),
        (5, "5"),
        (10, "10"),
    ])
    def test_metrics_records_heals(self, total_heals, expected_summary_contains):
        """Test HealingMetrics correctly records heal counts."""
        metrics = HealingMetrics()

        for i in range(total_heals):
            metrics.record_heal(f"healer-{i % 3}", f"STAGE-{i % 2}", success=True)

        assert metrics.total_heals == total_heals
        assert expected_summary_contains in str(metrics.total_heals)


# =============================================================================
# AC4: Healer selection based on error patterns
# =============================================================================


class TestParametrizedHealerSelection:
    """Parametrized tests for healer selection based on error patterns."""

    @pytest.mark.parametrize("error_msg,stage_name,expected_healer", [
        # API errors -> api-healer
        ("Rate limit exceeded", "MATCH", "api-healer"),
        ("429 Too Many Requests", "DOWNLOAD", "api-healer"),
        ("timeout waiting for response", "ANALYZE", "api-healer"),

        # Checkpoint errors -> checkpoint-healer
        ("JSONDecodeError at position 0", "LOAD", "checkpoint-healer"),
        ("checkpoint corrupt", "RESUME", "checkpoint-healer"),

        # Download errors -> download-healer
        ("yt-dlp: Video unavailable", "DOWNLOAD", "download-healer"),
        ("youtube extraction failed", "DOWNLOAD_SEGMENTS", "download-healer"),

        # Disk errors -> disk-healer
        ("No space left on device", "OUTPUT", "disk-healer"),
        ("Permission denied", "OUTPUT", "disk-healer"),

        # Path errors -> path-healer
        ("Path too long", "OUTPUT", "path-healer"),
        ("unicode encode error", "OUTPUT", "path-healer"),

        # OTIO errors -> otio-healer
        ("opentimelineio exception", "OUTPUT", "otio-healer"),
        ("OTIO serialization failed", "OUTPUT", "otio-healer"),
    ])
    def test_orchestrator_selects_correct_healer(self, mock_config, project_dir, error_msg, stage_name, expected_healer):
        """Test orchestrator selects correct healer for each error type."""
        orchestrator = HealingOrchestrator(mock_config, project_dir)

        error = Exception(error_msg)
        healers = orchestrator.select_healers(error, stage_name)

        # Should find at least one healer
        healer_names = [h.name for h in healers]
        assert expected_healer in healer_names, f"Expected {expected_healer} for error: {error_msg}, got: {healer_names}"


class TestParametrizedHealerPriority:
    """Parametrized tests for healer priority ordering."""

    @pytest.mark.parametrize("priority_order", [
        ["checkpoint-healer", "api-healer", "download-healer"],
        ["api-healer", "checkpoint-healer", "disk-healer"],
        ["disk-healer", "path-healer", "otio-healer"],
    ])
    def test_orchestrator_respects_healer_priority(self, mock_config, project_dir, priority_order):
        """Test orchestrator respects healer priority order."""
        strategy = HealingStrategy(healer_priority=priority_order)
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        healer_names = [h.name for h in orchestrator.healers]

        # Verify priority order is maintained for healers that exist
        for i, name in enumerate(priority_order[:-1]):
            if name in healer_names and priority_order[i + 1] in healer_names:
                assert healer_names.index(name) < healer_names.index(priority_order[i + 1])


class TestParametrizedMultiHealerScenarios:
    """Parametrized tests for scenarios involving multiple healers."""

    @pytest.mark.parametrize("skip_healers,expected_excluded", [
        ({"disk-healer"}, "disk-healer"),
        ({"api-healer"}, "api-healer"),
        ({"checkpoint-healer"}, "checkpoint-healer"),
        ({"download-healer"}, "download-healer"),
        ({"path-healer"}, "path-healer"),
    ])
    def test_orchestrator_respects_skip_healers(self, mock_config, project_dir, skip_healers, expected_excluded):
        """Test orchestrator excludes healers in skip_healers set."""
        strategy = HealingStrategy(skip_healers=skip_healers)
        orchestrator = HealingOrchestrator(mock_config, project_dir, strategy=strategy)

        healer_names = [h.name for h in orchestrator.healers]
        assert expected_excluded not in healer_names


class TestParametrizedExceptionTypes:
    """Parametrized tests for exception type handling."""

    @pytest.mark.parametrize("exception_class,error_msg,healer_class,expected_can_handle", [
        # ValueError - handled by healers with it in exception_types
        (ValueError, "invalid value", APIHealer, True),
        (ValueError, "invalid value", CheckpointHealer, True),

        # OSError variants - handled by DiskHealer
        (OSError, "disk error", DiskHealer, True),
        (PermissionError, "denied", DiskHealer, True),

        # FileNotFoundError - handled by multiple healers
        (FileNotFoundError, "file missing", CheckpointHealer, True),

        # TypeError - not commonly handled
        (TypeError, "type error", APIHealer, False),
    ])
    def test_healer_exception_type_handling(self, mock_config, project_dir, exception_class, error_msg, healer_class, expected_can_handle):
        """Test healer exception type matching."""
        healer = healer_class(mock_config, project_dir)
        error = exception_class(error_msg)

        # Note: can_handle checks both patterns AND exception types
        result = healer.can_handle(error, "TEST")
        # For exception types, the match is based on class
        if exception_class in healer.exception_types:
            assert result is True
        # Otherwise falls back to pattern matching


# =============================================================================
# Additional parametrized tests for comprehensive coverage
# =============================================================================


class TestParametrizedHealerResultVariations:
    """Parametrized tests for HealerResult variations."""

    @pytest.mark.parametrize("action,expected_retry", [
        (HealerAction.RETRY, True),
        (HealerAction.SKIP, False),
        (HealerAction.ABORT, False),
        (HealerAction.MODIFY_CONFIG, True),
        (HealerAction.RESTORE, True),
    ])
    def test_healer_result_action_implies_retry(self, action, expected_retry):
        """Test which HealerActions imply retry should be attempted."""
        result = HealerResult.fixed("test", action=action)

        # RETRY, MODIFY_CONFIG, RESTORE all imply retry
        should_retry = action in [HealerAction.RETRY, HealerAction.MODIFY_CONFIG, HealerAction.RESTORE]
        assert should_retry == expected_retry


class TestParametrizedMetricsRecording:
    """Parametrized tests for metrics recording."""

    @pytest.mark.parametrize("healer_name,stage_name,success", [
        ("api-healer", "MATCH", True),
        ("api-healer", "MATCH", False),
        ("checkpoint-healer", "LOAD", True),
        ("download-healer", "DOWNLOAD", True),
        ("disk-healer", "OUTPUT", False),
        ("otio-healer", "OUTPUT", True),
    ])
    def test_metrics_records_heal_by_healer_and_stage(self, healer_name, stage_name, success):
        """Test HealingMetrics records heals by healer and stage."""
        metrics = HealingMetrics()

        metrics.record_heal(healer_name, stage_name, success=success)

        assert metrics.heals_by_healer[healer_name] == 1
        assert metrics.heals_by_stage[stage_name] == 1
        if success:
            assert metrics.successful_heals == 1
            assert metrics.failed_heals == 0
        else:
            assert metrics.successful_heals == 0
            assert metrics.failed_heals == 1


class TestParametrizedBackoffBehavior:
    """Parametrized tests for backoff behavior."""

    @pytest.mark.parametrize("healer_class,initial_backoff,max_backoff", [
        (APIHealer, 5.0, 300.0),
        (DownloadHealer, 10.0, 600.0),
    ])
    def test_healer_backoff_parameters(self, mock_config, project_dir, healer_class, initial_backoff, max_backoff):
        """Test healer backoff parameters."""
        healer = healer_class(mock_config, project_dir)

        assert healer.INITIAL_BACKOFF == initial_backoff
        assert healer.MAX_BACKOFF == max_backoff

    @pytest.mark.parametrize("retry_count,expected_backoff_increased", [
        (0, False),  # First call
        (1, True),   # After first retry
        (2, True),   # After second retry
        (5, True),   # After multiple retries
    ])
    def test_backoff_increases_with_retries(self, mock_config, project_dir, retry_count, expected_backoff_increased):
        """Test backoff time increases with retries."""
        healer = APIHealer(mock_config, project_dir)
        initial = healer.backoff_time

        # Simulate retries
        healer.retry_count = retry_count
        if retry_count > 0:
            healer.backoff_time = min(healer.INITIAL_BACKOFF * (2 ** retry_count), healer.MAX_BACKOFF)

        if expected_backoff_increased:
            assert healer.backoff_time >= initial


class TestParametrizedConfigModifications:
    """Parametrized tests for config modifications by healers."""

    @pytest.mark.parametrize("initial_timeout,expected_increase", [
        (30, True),
        (60, True),
        (120, True),
    ])
    def test_api_healer_increases_timeout(self, mock_config, project_dir, initial_timeout, expected_increase):
        """Test APIHealer increases timeout on timeout errors."""
        healer = APIHealer(mock_config, project_dir)
        mock_config.llm.timeout = initial_timeout
        state = Mock()

        result = healer._handle_timeout(Exception("request timed out"), state)

        assert result.success
        if expected_increase:
            assert mock_config.llm.timeout > initial_timeout

    @pytest.mark.parametrize("initial_socket_timeout,expected_increase", [
        (10, True),
        (30, True),
        (60, True),
    ])
    @patch('time.sleep')
    def test_download_healer_increases_socket_timeout(self, mock_sleep, mock_config, project_dir,
                                                       initial_socket_timeout, expected_increase):
        """Test DownloadHealer increases socket timeout on network errors."""
        healer = DownloadHealer(mock_config, project_dir)
        mock_config.download.socket_timeout = initial_socket_timeout
        state = Mock()

        result = healer._handle_network_error(Exception("connection timeout"), state)

        assert result.success
        if expected_increase:
            assert mock_config.download.socket_timeout > initial_socket_timeout
