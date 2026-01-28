"""
Tests for base Healer class and HealerResult.
"""

import pytest
from unittest.mock import Mock

from src.agents.base import Healer, HealerResult, HealerAction


class TestHealerResult:
    """Tests for HealerResult dataclass."""

    @pytest.mark.fast
    def test_fixed_creates_success_result(self):
        """Test HealerResult.fixed() creates successful result."""
        result = HealerResult.fixed("Fixed the issue")

        assert result.success is True
        assert result.action == HealerAction.RETRY
        assert result.message == "Fixed the issue"
        assert result.modified_config is False

    @pytest.mark.fast
    def test_fixed_with_custom_action(self):
        """Test HealerResult.fixed() with custom action."""
        result = HealerResult.fixed("Skipping", action=HealerAction.SKIP)

        assert result.success is True
        assert result.action == HealerAction.SKIP

    @pytest.mark.fast
    def test_fixed_with_details(self):
        """Test HealerResult.fixed() with extra details."""
        result = HealerResult.fixed(
            "Fixed",
            fixed_count=5,
            files_resolved=["a.mp4", "b.mp4"]
        )

        assert result.details["fixed_count"] == 5
        assert result.details["files_resolved"] == ["a.mp4", "b.mp4"]

    @pytest.mark.fast
    def test_failed_creates_failure_result(self):
        """Test HealerResult.failed() creates failure result."""
        result = HealerResult.failed("Could not fix")

        assert result.success is False
        assert result.action == HealerAction.ABORT
        assert result.message == "Could not fix"

    @pytest.mark.fast
    def test_failed_with_details(self):
        """Test HealerResult.failed() with extra details."""
        result = HealerResult.failed("Error", error_code=500)

        assert result.details["error_code"] == 500

    @pytest.mark.fast
    def test_config_changed_creates_modified_result(self):
        """Test HealerResult.config_changed() creates config-modified result."""
        result = HealerResult.config_changed("Changed gap_mode")

        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert result.modified_config is True


class TestHealerAction:
    """Tests for HealerAction enum."""

    @pytest.mark.fast
    def test_action_values(self):
        """Test all action values exist."""
        assert HealerAction.RETRY.value == "retry"
        assert HealerAction.SKIP.value == "skip"
        assert HealerAction.MODIFY_CONFIG.value == "modify"
        assert HealerAction.RESTORE.value == "restore"
        assert HealerAction.ABORT.value == "abort"


class ConcreteHealer(Healer):
    """Concrete implementation for testing."""

    name = "test-healer"
    description = "Test healer"
    error_patterns = ["test error", "sample"]
    exception_types = [ValueError]

    def fix(self, error, state, stage_name):
        return HealerResult.fixed("Test fix applied")


class TestHealer:
    """Tests for Healer base class."""

    @pytest.mark.fast
    def test_healer_initialization(self, mock_config, project_dir):
        """Test healer initializes correctly."""
        healer = ConcreteHealer(mock_config, project_dir)

        assert healer.config == mock_config
        assert healer.project_dir == project_dir
        assert healer.name == "test-healer"

    @pytest.mark.fast
    def test_can_handle_by_pattern(self, mock_config, project_dir):
        """Test can_handle matches error patterns."""
        healer = ConcreteHealer(mock_config, project_dir)

        # Should match pattern
        assert healer.can_handle(Exception("This is a test error"), "STAGE")
        assert healer.can_handle(Exception("sample error here"), "STAGE")

        # Should not match
        assert not healer.can_handle(Exception("unrelated"), "STAGE")

    @pytest.mark.fast
    def test_can_handle_by_exception_type(self, mock_config, project_dir):
        """Test can_handle matches exception types."""
        healer = ConcreteHealer(mock_config, project_dir)

        # Should match exception type
        assert healer.can_handle(ValueError("any message"), "STAGE")

        # Should not match different type
        assert not healer.can_handle(TypeError("any message"), "STAGE")

    @pytest.mark.fast
    def test_can_handle_case_insensitive(self, mock_config, project_dir):
        """Test pattern matching is case-insensitive."""
        healer = ConcreteHealer(mock_config, project_dir)

        assert healer.can_handle(Exception("TEST ERROR"), "STAGE")
        assert healer.can_handle(Exception("Test Error"), "STAGE")
        assert healer.can_handle(Exception("SAMPLE"), "STAGE")

    @pytest.mark.fast
    def test_fix_returns_result(self, mock_config, project_dir, mock_state):
        """Test fix method returns HealerResult."""
        healer = ConcreteHealer(mock_config, project_dir)
        error = Exception("test error")

        result = healer.fix(error, mock_state, "STAGE")

        assert isinstance(result, HealerResult)
        assert result.success is True

    @pytest.mark.fast
    def test_log_methods(self, mock_config, project_dir, caplog):
        """Test logging methods."""
        import logging
        caplog.set_level(logging.INFO)

        healer = ConcreteHealer(mock_config, project_dir)

        healer.log_attempt("Trying something")
        healer.log_success("It worked")
        healer.log_failure("It failed")

        assert "[test-healer] Trying something" in caplog.text
        assert "[test-healer] ✓ It worked" in caplog.text
        assert "[test-healer] ✗ It failed" in caplog.text
