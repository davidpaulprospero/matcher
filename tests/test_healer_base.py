"""Unit tests for Healer base class and HealerResult."""

import logging
import pytest
from unittest.mock import MagicMock, patch


class TestHealerAction:
    """Test HealerAction enum values."""

    @pytest.mark.fast
    def test_retry_value(self):
        """Test RETRY action value."""
        from src.agents.base import HealerAction
        assert HealerAction.RETRY.value == "retry"

    @pytest.mark.fast
    def test_skip_value(self):
        """Test SKIP action value."""
        from src.agents.base import HealerAction
        assert HealerAction.SKIP.value == "skip"

    @pytest.mark.fast
    def test_modify_config_value(self):
        """Test MODIFY_CONFIG action value."""
        from src.agents.base import HealerAction
        assert HealerAction.MODIFY_CONFIG.value == "modify"

    @pytest.mark.fast
    def test_restore_value(self):
        """Test RESTORE action value."""
        from src.agents.base import HealerAction
        assert HealerAction.RESTORE.value == "restore"

    @pytest.mark.fast
    def test_abort_value(self):
        """Test ABORT action value."""
        from src.agents.base import HealerAction
        assert HealerAction.ABORT.value == "abort"

    @pytest.mark.fast
    def test_enum_has_five_values(self):
        """Test that HealerAction has exactly 5 values."""
        from src.agents.base import HealerAction
        assert len(HealerAction) == 5


class TestHealerResultFixed:
    """Test HealerResult.fixed() factory method."""

    @pytest.mark.fast
    def test_fixed_returns_success_true(self):
        """Test that fixed() returns result with success=True."""
        from src.agents.base import HealerResult
        result = HealerResult.fixed("Test message")
        assert result.success is True

    @pytest.mark.fast
    def test_fixed_default_action_is_retry(self):
        """Test that fixed() defaults to RETRY action."""
        from src.agents.base import HealerResult, HealerAction
        result = HealerResult.fixed("Test message")
        assert result.action == HealerAction.RETRY

    @pytest.mark.fast
    def test_fixed_with_custom_action(self):
        """Test that fixed() accepts custom action."""
        from src.agents.base import HealerResult, HealerAction
        result = HealerResult.fixed("Test message", action=HealerAction.SKIP)
        assert result.action == HealerAction.SKIP

    @pytest.mark.fast
    def test_fixed_stores_message(self):
        """Test that fixed() stores the message."""
        from src.agents.base import HealerResult
        result = HealerResult.fixed("Test message")
        assert result.message == "Test message"

    @pytest.mark.fast
    def test_fixed_modified_config_false_by_default(self):
        """Test that fixed() has modified_config=False by default."""
        from src.agents.base import HealerResult
        result = HealerResult.fixed("Test message")
        assert result.modified_config is False

    @pytest.mark.fast
    def test_fixed_accepts_extra_details(self):
        """Test that fixed() accepts and stores extra details."""
        from src.agents.base import HealerResult
        result = HealerResult.fixed("Test", key1="value1", key2=42)
        assert result.details["key1"] == "value1"
        assert result.details["key2"] == 42


class TestHealerResultFailed:
    """Test HealerResult.failed() factory method."""

    @pytest.mark.fast
    def test_failed_returns_success_false(self):
        """Test that failed() returns result with success=False."""
        from src.agents.base import HealerResult
        result = HealerResult.failed("Error message")
        assert result.success is False

    @pytest.mark.fast
    def test_failed_action_is_abort(self):
        """Test that failed() has action=ABORT."""
        from src.agents.base import HealerResult, HealerAction
        result = HealerResult.failed("Error message")
        assert result.action == HealerAction.ABORT

    @pytest.mark.fast
    def test_failed_stores_message(self):
        """Test that failed() stores the message."""
        from src.agents.base import HealerResult
        result = HealerResult.failed("Error message")
        assert result.message == "Error message"

    @pytest.mark.fast
    def test_failed_accepts_extra_details(self):
        """Test that failed() accepts and stores extra details."""
        from src.agents.base import HealerResult
        result = HealerResult.failed("Error", error_code=500)
        assert result.details["error_code"] == 500


class TestHealerResultConfigChanged:
    """Test HealerResult.config_changed() factory method."""

    @pytest.mark.fast
    def test_config_changed_returns_success_true(self):
        """Test that config_changed() returns result with success=True."""
        from src.agents.base import HealerResult
        result = HealerResult.config_changed("Config updated")
        assert result.success is True

    @pytest.mark.fast
    def test_config_changed_action_is_modify_config(self):
        """Test that config_changed() has action=MODIFY_CONFIG."""
        from src.agents.base import HealerResult, HealerAction
        result = HealerResult.config_changed("Config updated")
        assert result.action == HealerAction.MODIFY_CONFIG

    @pytest.mark.fast
    def test_config_changed_sets_modified_config_true(self):
        """Test that config_changed() sets modified_config=True."""
        from src.agents.base import HealerResult
        result = HealerResult.config_changed("Config updated")
        assert result.modified_config is True

    @pytest.mark.fast
    def test_config_changed_stores_message(self):
        """Test that config_changed() stores the message."""
        from src.agents.base import HealerResult
        result = HealerResult.config_changed("Timeout increased")
        assert result.message == "Timeout increased"

    @pytest.mark.fast
    def test_config_changed_accepts_extra_details(self):
        """Test that config_changed() accepts and stores extra details."""
        from src.agents.base import HealerResult
        result = HealerResult.config_changed("Updated", old_value=30, new_value=60)
        assert result.details["old_value"] == 30
        assert result.details["new_value"] == 60


class TestHealerCanHandle:
    """Test Healer.can_handle() method."""

    @pytest.mark.fast
    def test_can_handle_matches_error_pattern(self):
        """Test can_handle() matches error patterns in message."""
        from src.agents.base import Healer, HealerResult, HealerAction

        # Create a concrete healer for testing
        class TestHealer(Healer):
            name = "test-healer"
            error_patterns = ["rate limit", "429"]

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        # Should match
        error = Exception("Rate limit exceeded")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("HTTP 429 Too Many Requests")
        assert healer.can_handle(error, "TEST") is True

    @pytest.mark.fast
    def test_can_handle_case_insensitive(self):
        """Test can_handle() is case insensitive."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"
            error_patterns = ["timeout"]

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        # Mixed case should match
        error = Exception("Request TIMEOUT occurred")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("TimeOut error")
        assert healer.can_handle(error, "TEST") is True

    @pytest.mark.fast
    def test_can_handle_no_match(self):
        """Test can_handle() returns False when no pattern matches."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"
            error_patterns = ["timeout", "rate limit"]

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        # Should not match
        error = Exception("File not found")
        assert healer.can_handle(error, "TEST") is False

    @pytest.mark.fast
    def test_can_handle_matches_exception_type(self):
        """Test can_handle() matches exception types."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"
            error_patterns = []
            exception_types = [ValueError, TimeoutError]

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        # Should match exception type
        error = ValueError("Invalid value")
        assert healer.can_handle(error, "TEST") is True

        error = TimeoutError("Timed out")
        assert healer.can_handle(error, "TEST") is True

        # Should not match
        error = Exception("Generic error")
        assert healer.can_handle(error, "TEST") is False

    @pytest.mark.fast
    def test_can_handle_exception_type_inheritance(self):
        """Test can_handle() matches exception subclasses."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"
            error_patterns = []
            exception_types = [OSError]  # Parent class

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        # FileNotFoundError is subclass of OSError
        error = FileNotFoundError("File not found")
        assert healer.can_handle(error, "TEST") is True

        # PermissionError is subclass of OSError
        error = PermissionError("Permission denied")
        assert healer.can_handle(error, "TEST") is True

    @pytest.mark.fast
    def test_can_handle_pattern_or_exception(self):
        """Test can_handle() matches if either pattern or exception matches."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"
            error_patterns = ["timeout"]
            exception_types = [ValueError]

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        # Match via pattern
        error = Exception("Connection timeout")
        assert healer.can_handle(error, "TEST") is True

        # Match via type
        error = ValueError("Invalid")
        assert healer.can_handle(error, "TEST") is True

        # No match
        error = TypeError("Type error")
        assert healer.can_handle(error, "TEST") is False


class TestHealerLogging:
    """Test Healer logging methods."""

    @pytest.mark.fast
    def test_log_attempt(self, caplog):
        """Test log_attempt() logs to INFO level."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        with caplog.at_level(logging.INFO):
            healer.log_attempt("Attempting to fix error")

        assert "[test-healer]" in caplog.text
        assert "Attempting to fix error" in caplog.text

    @pytest.mark.fast
    def test_log_success(self, caplog):
        """Test log_success() logs to INFO level with checkmark."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        with caplog.at_level(logging.INFO):
            healer.log_success("Error fixed successfully")

        assert "[test-healer]" in caplog.text
        assert "✓" in caplog.text
        assert "Error fixed successfully" in caplog.text

    @pytest.mark.fast
    def test_log_failure(self, caplog):
        """Test log_failure() logs to WARNING level with x mark."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/tmp/project")

        with caplog.at_level(logging.WARNING):
            healer.log_failure("Could not fix error")

        assert "[test-healer]" in caplog.text
        assert "✗" in caplog.text
        assert "Could not fix error" in caplog.text


class TestHealerInit:
    """Test Healer initialization."""

    @pytest.mark.fast
    def test_healer_stores_config(self):
        """Test that Healer stores config in self.config."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        config.some_setting = "value"
        healer = TestHealer(config, "/tmp/project")

        assert healer.config is config
        assert healer.config.some_setting == "value"

    @pytest.mark.fast
    def test_healer_stores_project_dir(self):
        """Test that Healer stores project_dir."""
        from src.agents.base import Healer, HealerResult

        class TestHealer(Healer):
            name = "test-healer"

            def fix(self, error, state, stage_name):
                return HealerResult.fixed("Fixed")

        config = MagicMock()
        healer = TestHealer(config, "/path/to/project")

        assert healer.project_dir == "/path/to/project"


class TestHealerResultDefaults:
    """Test HealerResult default values."""

    @pytest.mark.fast
    def test_details_default_empty_dict(self):
        """Test that details defaults to empty dict."""
        from src.agents.base import HealerResult, HealerAction
        result = HealerResult(
            success=True,
            action=HealerAction.RETRY,
            message="Test"
        )
        assert result.details == {}
        assert isinstance(result.details, dict)

    @pytest.mark.fast
    def test_modified_config_default_false(self):
        """Test that modified_config defaults to False."""
        from src.agents.base import HealerResult, HealerAction
        result = HealerResult(
            success=True,
            action=HealerAction.RETRY,
            message="Test"
        )
        assert result.modified_config is False
