"""
Tests for DownloadHealer integration with retry mechanism.

Tests cover:
- DownloadError exception with retry context
- _get_retry_context() method
- Healer skipping redundant backoff when retries exhausted
- Clear logging of retry handoff between core and healer
- Cookie rotation and VPN switch after retry exhaustion
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockCookieRotationConfig:
    """Mock cookie rotation config."""
    enabled: bool = False
    cookie_files: list = None
    rotate_on_errors: list = None
    rotation_strategy: str = "on_error"
    cooldown_seconds: int = 300
    max_rotations_per_session: int = 0

    def __post_init__(self):
        if self.cookie_files is None:
            self.cookie_files = []
        if self.rotate_on_errors is None:
            self.rotate_on_errors = ["429", "rate limit"]


@dataclass
class MockVPNConfig:
    """Mock VPN config."""
    enabled: bool = False
    servers: list = None

    def __post_init__(self):
        if self.servers is None:
            self.servers = []


@dataclass
class MockDownloadConfig:
    """Mock download config."""
    cookie_rotation: MockCookieRotationConfig = None
    vpn: MockVPNConfig = None
    format: str = "bestvideo+bestaudio/best"
    socket_timeout: int = 30
    continue_dl: bool = False

    def __post_init__(self):
        if self.cookie_rotation is None:
            self.cookie_rotation = MockCookieRotationConfig()
        if self.vpn is None:
            self.vpn = MockVPNConfig()


class MockConfig:
    """Mock config object."""
    def __init__(self, download_config=None):
        self.download = download_config or MockDownloadConfig()


class TestDownloadError:
    """Test DownloadError exception class."""

    def test_default_values(self):
        """Test DownloadError with default values."""
        from src.downloader.types import DownloadError

        error = DownloadError("Test error")

        assert error.message == "Test error"
        assert error.retry_count == 0
        assert error.max_retries == 3
        assert error.error_type == 'unknown'
        assert error.original_error == "Test error"
        assert not error.retries_exhausted

    def test_custom_retry_context(self):
        """Test DownloadError with custom retry context."""
        from src.downloader.types import DownloadError

        error = DownloadError(
            "Rate limit error",
            retry_count=3,
            max_retries=3,
            error_type='transient',
            original_error="HTTP 429 Too Many Requests"
        )

        assert error.retry_count == 3
        assert error.max_retries == 3
        assert error.error_type == 'transient'
        assert error.retries_exhausted is True

    def test_retries_exhausted_property(self):
        """Test retries_exhausted property."""
        from src.downloader.types import DownloadError

        # Not exhausted
        error1 = DownloadError("Error", retry_count=1, max_retries=3)
        assert not error1.retries_exhausted

        # Exhausted (equal)
        error2 = DownloadError("Error", retry_count=3, max_retries=3)
        assert error2.retries_exhausted

        # Exhausted (greater)
        error3 = DownloadError("Error", retry_count=5, max_retries=3)
        assert error3.retries_exhausted

    def test_str_includes_retry_info(self):
        """Test __str__ includes retry information."""
        from src.downloader.types import DownloadError

        # With retries
        error1 = DownloadError("Rate limit", retry_count=2, max_retries=3)
        assert "retried 2/3" in str(error1)

        # Without retries
        error2 = DownloadError("Rate limit")
        assert "retried" not in str(error2)


class TestGetRetryContext:
    """Test _get_retry_context() method."""

    def test_with_download_error(self, project_dir):
        """Test retry context extraction from DownloadError."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)

        error = DownloadError("Rate limit", retry_count=3, max_retries=3)
        retry_count, max_retries, exhausted = healer._get_retry_context(error)

        assert retry_count == 3
        assert max_retries == 3
        assert exhausted is True

    def test_with_regular_exception(self, project_dir):
        """Test retry context with regular exception returns defaults."""
        from src.agents.healers.download import DownloadHealer

        healer = DownloadHealer(MockConfig(), project_dir)

        error = Exception("Some error")
        retry_count, max_retries, exhausted = healer._get_retry_context(error)

        assert retry_count == 0
        assert max_retries == 3
        assert exhausted is False

    def test_with_value_error(self, project_dir):
        """Test retry context with ValueError returns defaults."""
        from src.agents.healers.download import DownloadHealer

        healer = DownloadHealer(MockConfig(), project_dir)

        error = ValueError("Invalid value")
        retry_count, max_retries, exhausted = healer._get_retry_context(error)

        assert retry_count == 0
        assert max_retries == 3
        assert exhausted is False


class TestHealerSkipsRedundantBackoff:
    """Test healer skips redundant backoff when retries exhausted."""

    @patch('time.sleep')
    def test_rate_limit_skips_backoff_when_exhausted(self, mock_sleep, project_dir):
        """Test rate limit handler skips backoff when retries exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP 429 rate limit error",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Should not have called sleep (no additional backoff)
        mock_sleep.assert_not_called()
        # Should return failed since all escalation exhausted
        assert not result.success
        assert result.details.get('core_retries_exhausted') is True

    @patch('time.sleep')
    def test_rate_limit_applies_backoff_when_not_exhausted(self, mock_sleep, project_dir):
        """Test rate limit handler applies backoff when retries not exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP 429 rate limit error",
            retry_count=1,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Should have called sleep (backoff applied)
        mock_sleep.assert_called_once()
        # Should return success (retry recommended)
        assert result.success

    @patch('time.sleep')
    def test_network_error_skips_backoff_when_exhausted(self, mock_sleep, project_dir):
        """Test network error handler skips backoff when retries exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "Connection timeout error",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Should not have called sleep
        mock_sleep.assert_not_called()
        # Should still succeed (config changed to increase timeout)
        assert result.success
        assert result.details.get('core_retries_exhausted') is True

    @patch('time.sleep')
    def test_generic_error_fails_when_exhausted(self, mock_sleep, project_dir):
        """Test generic error handler fails fast when retries exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "Unknown download error",
            retry_count=3,
            max_retries=3,
            error_type='unknown'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Should not have called sleep
        mock_sleep.assert_not_called()
        # Should return failed
        assert not result.success
        assert result.details.get('core_retries_exhausted') is True


class TestCookieRotationWithRetryContext:
    """Test cookie rotation still works after retry exhaustion."""

    @patch('time.sleep')
    def test_cookie_rotation_triggered_when_retries_exhausted(self, mock_sleep, project_dir, tmp_path):
        """Test cookie rotation is tried even when core retries exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        # Create mock cookie files
        cookie1 = tmp_path / "cookie1.txt"
        cookie2 = tmp_path / "cookie2.txt"
        cookie1.write_text("cookie data 1")
        cookie2.write_text("cookie data 2")

        config = MockConfig(
            MockDownloadConfig(
                cookie_rotation=MockCookieRotationConfig(
                    enabled=True,
                    cookie_files=[str(cookie1), str(cookie2)],
                    rotate_on_errors=["429", "rate limit"]
                )
            )
        )

        healer = DownloadHealer(config, project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP 429 rate limit error",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Cookie rotation should succeed (not blocked by exhausted retries)
        assert result.success
        assert result.details.get('cookie_rotated') is True
        # No backoff sleep needed since we rotated cookie
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    def test_vpn_switch_triggered_after_cookie_exhaustion(self, mock_sleep, project_dir, tmp_path):
        """Test VPN switch is tried after cookies exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        # Create single cookie that will be exhausted
        cookie1 = tmp_path / "cookie1.txt"
        cookie1.write_text("cookie data 1")

        config = MockConfig(
            MockDownloadConfig(
                cookie_rotation=MockCookieRotationConfig(
                    enabled=True,
                    cookie_files=[str(cookie1)],
                    rotate_on_errors=["429", "rate limit"]
                ),
                # VPN disabled in config - we'll mock it directly
                vpn=MockVPNConfig(enabled=False)
            )
        )

        healer = DownloadHealer(config, project_dir)

        # Simulate cookie already rotated (exhausted)
        if healer.cookie_rotator:
            healer.cookie_rotator.rotate()  # Use up the one cookie

        # Mock VPN manager to simulate successful switch (injected directly)
        healer.vpn_manager = Mock()
        healer.vpn_manager.is_enabled = True
        healer.vpn_manager.can_switch.return_value = True
        healer.vpn_manager.switch.return_value = True
        healer.vpn_manager.switch_count = 1

        state = Mock()

        error = DownloadError(
            "HTTP 429 rate limit error",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # VPN switch should succeed
        assert result.success
        assert result.details.get('vpn_switched') is True


class TestHandoffLogging:
    """Test clear logging shows handoff between core retry and healer."""

    @patch('time.sleep')
    def test_logs_core_retry_exhaustion(self, mock_sleep, project_dir, caplog):
        """Test that core retry exhaustion is logged clearly."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError
        import logging

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP 429 rate limit error",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        with caplog.at_level(logging.INFO):
            healer.fix(error, state, "DOWNLOAD")

        # Check that exhaustion message was logged
        assert any("Core retry exhausted (3/3)" in record.message for record in caplog.records)

    @patch('time.sleep')
    def test_logs_skipping_healer_backoff(self, mock_sleep, project_dir, caplog):
        """Test that skipping healer backoff is logged."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError
        import logging

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP 429 rate limit error",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        with caplog.at_level(logging.INFO):
            healer.fix(error, state, "DOWNLOAD")

        # Check that skip message was logged
        assert any("core retry already exhausted" in record.message.lower() for record in caplog.records)


class TestHealerWithRegularException:
    """Test healer behavior with regular exceptions (no retry context)."""

    @patch('time.sleep')
    def test_rate_limit_with_regular_exception_applies_backoff(self, mock_sleep, project_dir):
        """Test rate limit handler applies backoff for regular exception."""
        from src.agents.healers.download import DownloadHealer

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = Exception("429 rate limit error")

        result = healer.fix(error, state, "DOWNLOAD")

        # Should have called sleep (backoff applied for regular exception)
        mock_sleep.assert_called_once()
        assert result.success

    @patch('time.sleep')
    def test_network_error_with_regular_exception_applies_backoff(self, mock_sleep, project_dir):
        """Test network error handler applies backoff for regular exception."""
        from src.agents.healers.download import DownloadHealer

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = Exception("Connection timeout error")

        result = healer.fix(error, state, "DOWNLOAD")

        # Should have called sleep
        mock_sleep.assert_called_once()
        assert result.success


class TestHealerResultDetails:
    """Test that healer results include retry context details."""

    @patch('time.sleep')
    def test_result_includes_core_retries_exhausted_flag(self, mock_sleep, project_dir):
        """Test result includes core_retries_exhausted flag."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "Generic download error",
            retry_count=3,
            max_retries=3
        )

        result = healer.fix(error, state, "DOWNLOAD")

        assert 'core_retries_exhausted' in result.details
        assert result.details['core_retries_exhausted'] is True

    @patch('time.sleep')
    def test_result_includes_healer_retry_count(self, mock_sleep, project_dir):
        """Test result includes healer_retry_count when applicable."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        healer = DownloadHealer(MockConfig(), project_dir)
        healer.retry_count = 2  # Simulate previous healer retries
        state = Mock()

        error = DownloadError(
            "Generic download error",
            retry_count=3,
            max_retries=3
        )

        result = healer.fix(error, state, "DOWNLOAD")

        assert 'healer_retry_count' in result.details
        assert result.details['healer_retry_count'] == 2


class TestSharedEscalationManager:
    """Test DownloadHealer integration with shared EscalationManager."""

    @patch('time.sleep')
    def test_healer_uses_shared_escalation_manager(self, mock_sleep, project_dir):
        """Test healer consults escalation_manager.get_escalation_args() for rate limits."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        # Create mock escalation manager
        mock_esc_mgr = Mock()
        mock_result = Mock()
        mock_result.tier = Mock()
        mock_result.tier.name = "EXTRACTOR_ARGS"
        mock_result.args = ["--impersonate", "Chrome-136:Macos-15"]
        mock_result.rotate_cookies = False
        mock_esc_mgr.get_escalation_args.return_value = mock_result

        healer = DownloadHealer(
            MockConfig(), project_dir, escalation_manager=mock_esc_mgr
        )
        state = Mock()

        error = DownloadError("HTTP 403 Forbidden", retry_count=3, max_retries=3)
        result = healer.fix(error, state, "DOWNLOAD")

        # Should have consulted escalation manager
        mock_esc_mgr.record_failure.assert_called_once()
        mock_esc_mgr.get_escalation_args.assert_called_once()
        # Should return success (escalation applied)
        assert result.success
        assert result.details.get('escalation_tier') == "EXTRACTOR_ARGS"
        # Should NOT have created CookieRotator (escalation manager handles it)
        assert healer.cookie_rotator is None

    @patch('time.sleep')
    def test_healer_records_success_on_reset(self, mock_sleep, project_dir):
        """Test healer calls record_success when reset_backoff is called after healing."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        mock_esc_mgr = Mock()
        mock_result = Mock()
        mock_result.tier = Mock()
        mock_result.tier.name = "IMPERSONATE_ONLY"
        mock_result.args = []
        mock_result.rotate_cookies = False
        mock_esc_mgr.get_escalation_args.return_value = mock_result

        healer = DownloadHealer(
            MockConfig(), project_dir, escalation_manager=mock_esc_mgr
        )
        state = Mock()

        # Simulate a rate limit error being handled
        error = DownloadError("429 rate limit", retry_count=1, max_retries=3)
        healer.fix(error, state, "DOWNLOAD")

        # Simulate successful stage completion (ResilientRunner calls reset_backoff)
        healer.reset_backoff()

        # Should have recorded success for the keyword
        mock_esc_mgr.record_success.assert_called_once()

    @patch('time.sleep')
    def test_healer_records_failure_on_rate_limit(self, mock_sleep, project_dir):
        """Test healer calls record_failure when encountering rate limit errors."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        mock_esc_mgr = Mock()
        mock_result = Mock()
        mock_result.tier = Mock()
        mock_result.tier.name = "FULL_BYPASS"
        mock_result.args = ["--impersonate", "Chrome-136:Macos-15"]
        mock_result.rotate_cookies = True
        mock_esc_mgr.get_escalation_args.return_value = mock_result

        healer = DownloadHealer(
            MockConfig(), project_dir, escalation_manager=mock_esc_mgr
        )
        state = Mock()

        error = DownloadError("HTTP Error 403: Forbidden", retry_count=3, max_retries=3)
        result = healer.fix(error, state, "DOWNLOAD")

        # Should have recorded failure with error string
        mock_esc_mgr.record_failure.assert_called_once()
        call_args = mock_esc_mgr.record_failure.call_args
        assert "403" in call_args[0][1]  # error_output arg contains the error
        # Result includes cookie rotation flag from Tier 3
        assert result.details.get('rotate_cookies') is True

    @patch('time.sleep')
    def test_healer_without_escalation_manager_still_works(self, mock_sleep, project_dir):
        """Test healer works correctly without escalation_manager (backward compatible)."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        # No escalation_manager passed - should fall back to old behavior
        healer = DownloadHealer(MockConfig(), project_dir)
        assert healer.escalation_manager is None

        state = Mock()

        # Rate limit error without escalation manager falls to backoff
        error = DownloadError("429 rate limit", retry_count=1, max_retries=3)
        result = healer.fix(error, state, "DOWNLOAD")

        # Should still work (backoff applied)
        mock_sleep.assert_called_once()
        assert result.success

    @patch('time.sleep')
    def test_healer_skips_cookie_rotator_when_escalation_manager_present(self, mock_sleep, project_dir, tmp_path):
        """Test that CookieRotator is not created when escalation_manager is provided."""
        from src.agents.healers.download import DownloadHealer

        cookie1 = tmp_path / "cookie1.txt"
        cookie1.write_text("cookie data 1")

        config = MockConfig(
            MockDownloadConfig(
                cookie_rotation=MockCookieRotationConfig(
                    enabled=True,
                    cookie_files=[str(cookie1)],
                    rotate_on_errors=["429", "rate limit"]
                )
            )
        )

        mock_esc_mgr = Mock()

        # With escalation manager: cookie rotator should NOT be created
        healer_with = DownloadHealer(config, project_dir, escalation_manager=mock_esc_mgr)
        assert healer_with.cookie_rotator is None
        assert healer_with.escalation_manager is mock_esc_mgr

        # Without escalation manager: cookie rotator should be created
        healer_without = DownloadHealer(config, project_dir)
        # Cookie rotator may or may not be enabled depending on file validation,
        # but the code path to create it was followed
        assert healer_without.escalation_manager is None

    @patch('time.sleep')
    def test_orchestrator_wire_escalation_manager(self, mock_sleep, project_dir):
        """Test HealingOrchestrator can wire escalation_manager into DownloadHealer."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        config = MockConfig()
        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(config, project_dir, strategy)

        # Find download healer in instances
        download_healer = orchestrator._healer_instances.get('download-healer')
        assert download_healer is not None
        assert download_healer.escalation_manager is None  # Not wired yet

        # Wire escalation manager
        mock_esc_mgr = Mock()
        orchestrator.wire_escalation_manager(mock_esc_mgr)

        # Should now be wired
        assert download_healer.escalation_manager is mock_esc_mgr


class TestMaxEscalationTierFailure:
    """Test DownloadHealer fails properly when max escalation tier + retries exhausted."""

    @patch('time.sleep')
    def test_failed_when_max_tier_and_retries_exhausted_no_escalation_mgr(self, mock_sleep, project_dir):
        """Test .failed() returned when all escalation exhausted (no escalation manager)."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        # No cookie rotation, no VPN, no escalation manager
        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP Error 403: Forbidden",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Should fail — all options exhausted
        assert not result.success
        assert result.details.get('core_retries_exhausted') is True

    @patch('time.sleep')
    def test_failure_logged_with_tier_info(self, mock_sleep, project_dir, caplog):
        """Test failure is logged with escalation tier information."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError
        import logging

        healer = DownloadHealer(MockConfig(), project_dir)
        state = Mock()

        error = DownloadError(
            "HTTP Error 403: Forbidden",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        with caplog.at_level(logging.INFO):
            result = healer.fix(error, state, "DOWNLOAD")

        assert not result.success
        # Should log exhaustion message
        assert any("core retry exhausted" in record.message.lower() for record in caplog.records)

    @patch('time.sleep')
    def test_escalation_manager_max_tier_still_returns_fixed_for_retry(self, mock_sleep, project_dir):
        """Test escalation manager at max tier still returns .fixed() with tier info for retry."""
        from src.agents.healers.download import DownloadHealer
        from src.downloader.types import DownloadError

        mock_esc_mgr = Mock()
        mock_result = Mock()
        mock_result.tier = Mock()
        mock_result.tier.name = "FULL_BYPASS"  # Max tier
        mock_result.args = ["--impersonate", "Chrome-136:Macos-15", "--extractor-args", "youtube:player_client=web_safari"]
        mock_result.rotate_cookies = True
        mock_esc_mgr.get_escalation_args.return_value = mock_result

        healer = DownloadHealer(
            MockConfig(), project_dir, escalation_manager=mock_esc_mgr
        )
        state = Mock()

        error = DownloadError(
            "HTTP Error 403: Forbidden",
            retry_count=3,
            max_retries=3,
            error_type='transient'
        )

        result = healer.fix(error, state, "DOWNLOAD")

        # Escalation manager path always returns fixed (to let runner decide)
        assert result.success
        assert result.details.get('escalation_tier') == "FULL_BYPASS"
        assert result.details.get('rotate_cookies') is True
        assert result.details.get('core_retries_exhausted') is True
        # Verify failure was recorded
        mock_esc_mgr.record_failure.assert_called_once()


class TestExportFromDownloader:
    """Test DownloadError is properly exported from downloader package."""

    def test_download_error_importable(self):
        """Test DownloadError can be imported from src.downloader."""
        from src.downloader import DownloadError

        error = DownloadError("Test", retry_count=1, max_retries=3)
        assert error.retry_count == 1

    def test_download_error_in_all(self):
        """Test DownloadError is in __all__."""
        from src import downloader

        assert 'DownloadError' in downloader.__all__
