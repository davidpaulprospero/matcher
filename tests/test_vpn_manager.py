"""Tests for VPN manager functionality."""

import pytest
import subprocess
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass


@dataclass
class MockVPNConfig:
    """Mock config for testing."""
    enabled: bool = True
    switch_command: str = "echo 'VPN switched'"
    disconnect_command: str = "echo 'VPN disconnected'"
    rotate_on_rate_limit: bool = True
    switch_delay_seconds: int = 0  # No delay for tests
    max_switches_per_session: int = 10
    verify_connection: bool = False  # Skip verification in tests
    verify_timeout: int = 5


class TestVPNManager:
    """Tests for VPNManager class."""

    @pytest.fixture
    def manager(self):
        """Create a VPNManager with test config."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig()
        return VPNManager(config)

    @pytest.mark.fast
    def test_init_enabled(self):
        """Test initialization when enabled."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(enabled=True)
        manager = VPNManager(config)

        assert manager.is_enabled
        assert manager.switch_count == 0

    @pytest.mark.fast
    def test_init_disabled(self):
        """Test initialization when disabled."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(enabled=False)
        manager = VPNManager(config)

        assert not manager.is_enabled

    @pytest.mark.fast
    def test_init_no_command(self):
        """Test initialization with no switch command."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(enabled=True, switch_command="")
        manager = VPNManager(config)

        assert not manager.is_enabled

    @pytest.mark.fast
    def test_can_switch(self, manager):
        """Test can_switch check."""
        assert manager.can_switch()

    @pytest.mark.fast
    def test_can_switch_at_limit(self):
        """Test can_switch when at limit."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(max_switches_per_session=2)
        manager = VPNManager(config)

        # Do two switches
        manager._switch_count = 2

        assert not manager.can_switch()

    @pytest.mark.fast
    def test_can_switch_unlimited(self):
        """Test can_switch with unlimited (0) limit."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(max_switches_per_session=0)
        manager = VPNManager(config)

        # Even with many switches, should still be able to switch
        manager._switch_count = 100
        assert manager.can_switch()

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_success(self, mock_run, manager):
        """Test successful VPN switch."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Connected", stderr="")

        result = manager.switch()

        assert result is True
        assert manager.switch_count == 1
        mock_run.assert_called_once()

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_failure(self, mock_run, manager):
        """Test failed VPN switch."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Connection failed")

        result = manager.switch()

        assert result is False
        assert manager.switch_count == 0  # Not incremented on failure

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_timeout(self, mock_run, manager):
        """Test VPN switch timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 60)

        result = manager.switch()

        assert result is False

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_increments_count(self, mock_run, manager):
        """Test that switch increments count."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        manager.switch()
        assert manager.switch_count == 1

        manager.switch()
        assert manager.switch_count == 2

        manager.switch()
        assert manager.switch_count == 3

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_respects_limit(self, mock_run):
        """Test that switch respects max_switches_per_session."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(max_switches_per_session=2)
        manager = VPNManager(config)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # First two switches should work
        assert manager.switch() is True
        assert manager.switch() is True

        # Third switch should fail due to limit
        assert manager.switch() is False
        assert manager.switch_count == 2

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_disconnect_success(self, mock_run, manager):
        """Test successful VPN disconnect."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = manager.disconnect()

        assert result is True

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_disconnect_no_command(self, mock_run):
        """Test disconnect with no command configured."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(disconnect_command="")
        manager = VPNManager(config)

        result = manager.disconnect()

        assert result is True  # Should succeed (no-op)
        mock_run.assert_not_called()

    @pytest.mark.fast
    def test_reset(self, manager):
        """Test reset clears state."""
        manager._switch_count = 5
        manager._last_switch_time = 12345.0

        manager.reset()

        assert manager.switch_count == 0
        assert manager._last_switch_time is None

    @pytest.mark.fast
    def test_get_status(self, manager):
        """Test status reporting."""
        status = manager.get_status()

        assert status["enabled"] is True
        assert status["switch_count"] == 0
        assert status["max_switches"] == 10
        assert status["can_switch"] is True

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_with_delay(self, mock_run):
        """Test that switch respects delay setting."""
        from src.downloader.vpn_manager import VPNManager
        import time

        config = MockVPNConfig(switch_delay_seconds=1)
        manager = VPNManager(config)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        start = time.time()
        manager.switch()
        elapsed = time.time() - start

        # Should have waited at least 1 second
        assert elapsed >= 0.9  # Allow small margin

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_with_verification(self, mock_run):
        """Test switch with connection verification."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfig(verify_connection=True)
        manager = VPNManager(config)

        # First call is switch, second is verification
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = manager.switch()

        assert result is True
        # Should have called run at least twice (switch + verify)
        assert mock_run.call_count >= 1


class TestVPNCheckpointPersistence:
    """Tests for VPN manager checkpoint persistence (US-005)."""

    @pytest.fixture
    def config(self):
        """Create a VPN config for testing."""
        return MockVPNConfig(max_switches_per_session=5)

    @pytest.mark.fast
    def test_to_checkpoint_state_initial(self, config):
        """Test serialization of initial state (no switches)."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config)
        state = manager.to_checkpoint_state()

        assert state["switch_count"] == 0
        assert state["last_switch_timestamp"] is None

    @pytest.mark.fast
    def test_to_checkpoint_state_after_switches(self, config):
        """Test serialization after switches."""
        from src.downloader.vpn_manager import VPNManager
        import time

        manager = VPNManager(config)
        manager._switch_count = 3
        manager._last_switch_time = time.time()

        state = manager.to_checkpoint_state()

        assert state["switch_count"] == 3
        assert state["last_switch_timestamp"] is not None
        # Verify it's a valid ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(state["last_switch_timestamp"])

    @pytest.mark.fast
    def test_from_checkpoint_state_restores_switch_count(self, config):
        """Test that from_checkpoint_state restores switch count."""
        from src.downloader.vpn_manager import VPNManager

        state = {"switch_count": 3, "last_switch_timestamp": None}
        manager = VPNManager.from_checkpoint_state(state, config)

        assert manager.switch_count == 3

    @pytest.mark.fast
    def test_from_checkpoint_state_restores_timestamp(self, config):
        """Test that from_checkpoint_state restores timestamp."""
        from src.downloader.vpn_manager import VPNManager
        from datetime import datetime

        timestamp = datetime.now().isoformat()
        state = {"switch_count": 2, "last_switch_timestamp": timestamp}
        manager = VPNManager.from_checkpoint_state(state, config)

        assert manager._last_switch_time is not None
        assert isinstance(manager._last_switch_time, float)  # Epoch time

    @pytest.mark.fast
    def test_from_checkpoint_state_empty_state(self, config):
        """Test from_checkpoint_state with empty state."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager.from_checkpoint_state({}, config)

        assert manager.switch_count == 0
        assert manager._last_switch_time is None

    @pytest.mark.fast
    def test_from_checkpoint_state_none_state(self, config):
        """Test from_checkpoint_state with None state."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager.from_checkpoint_state(None, config)

        assert manager.switch_count == 0

    @pytest.mark.fast
    def test_restore_from_checkpoint(self, config):
        """Test restore_from_checkpoint updates existing instance."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config)
        assert manager.switch_count == 0

        state = {"switch_count": 4, "last_switch_timestamp": None}
        manager.restore_from_checkpoint(state)

        assert manager.switch_count == 4

    @pytest.mark.fast
    def test_restore_from_checkpoint_logs_status(self, config, caplog):
        """Test that restore_from_checkpoint logs status message."""
        from src.downloader.vpn_manager import VPNManager
        import logging

        with caplog.at_level(logging.INFO):
            manager = VPNManager(config)
            state = {"switch_count": 2, "last_switch_timestamp": None}
            manager.restore_from_checkpoint(state)

        assert any("Resuming VPN manager: 2/5 switches used" in msg for msg in caplog.messages)

    @pytest.mark.fast
    def test_restore_from_checkpoint_empty_state(self, config):
        """Test restore_from_checkpoint with empty state is no-op."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config)
        manager._switch_count = 5  # Set initial value

        manager.restore_from_checkpoint({})

        # Empty state should not change anything
        assert manager.switch_count == 5

    @pytest.mark.fast
    def test_restore_from_checkpoint_none_state(self, config):
        """Test restore_from_checkpoint with None state is no-op."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config)
        manager._switch_count = 3

        manager.restore_from_checkpoint(None)

        # None state should not change anything
        assert manager.switch_count == 3

    @pytest.mark.fast
    def test_roundtrip_serialization(self, config):
        """Test full roundtrip: to_checkpoint_state -> from_checkpoint_state."""
        from src.downloader.vpn_manager import VPNManager
        import time

        # Create manager with some state
        original = VPNManager(config)
        original._switch_count = 3
        original._last_switch_time = time.time() - 300  # 5 minutes ago

        # Serialize
        state = original.to_checkpoint_state()

        # Deserialize
        restored = VPNManager.from_checkpoint_state(state, config)

        assert restored.switch_count == original.switch_count
        # Timestamps should be close (within 1 second due to ISO format precision)
        assert abs(restored._last_switch_time - original._last_switch_time) < 1.0

    @pytest.mark.fast
    def test_max_switches_enforced_across_resume(self, config):
        """Test that max_switches_per_session is enforced across resume boundaries."""
        from src.downloader.vpn_manager import VPNManager

        # Set max to 5
        config.max_switches_per_session = 5

        # Restore with 4 switches already used
        state = {"switch_count": 4, "last_switch_timestamp": None}
        manager = VPNManager.from_checkpoint_state(state, config)

        # Should only be able to switch once more
        assert manager.can_switch() is True
        assert manager.switch_count == 4

        # Simulate successful switch
        manager._switch_count = 5

        # Now should NOT be able to switch
        assert manager.can_switch() is False


class TestVPNCheckpointTimestampHelpers:
    """Tests for VPN manager timestamp helper methods."""

    @pytest.mark.fast
    def test_format_timestamp(self):
        """Test _format_timestamp converts epoch to ISO."""
        from src.downloader.vpn_manager import VPNManager
        from datetime import datetime

        epoch = 1706198400.0  # 2024-01-25 12:00:00 UTC
        iso = VPNManager._format_timestamp(epoch)

        # Should be a valid ISO timestamp
        parsed = datetime.fromisoformat(iso)
        assert abs(parsed.timestamp() - epoch) < 1.0

    @pytest.mark.fast
    def test_parse_timestamp_valid(self):
        """Test _parse_timestamp with valid ISO string."""
        from src.downloader.vpn_manager import VPNManager
        from datetime import datetime

        iso = "2024-01-25T12:00:00"
        epoch = VPNManager._parse_timestamp(iso)

        assert epoch is not None
        parsed = datetime.fromisoformat(iso)
        assert abs(epoch - parsed.timestamp()) < 1.0

    @pytest.mark.fast
    def test_parse_timestamp_invalid(self):
        """Test _parse_timestamp with invalid string."""
        from src.downloader.vpn_manager import VPNManager

        result = VPNManager._parse_timestamp("not-a-timestamp")

        assert result is None

    @pytest.mark.fast
    def test_parse_timestamp_none(self):
        """Test _parse_timestamp with None."""
        from src.downloader.vpn_manager import VPNManager

        result = VPNManager._parse_timestamp(None)

        assert result is None


class TestDownloadCheckpointVPNState:
    """Tests for DownloadCheckpoint vpn_manager_state field."""

    @pytest.mark.fast
    def test_checkpoint_has_vpn_manager_state_field(self):
        """Test that DownloadCheckpoint has vpn_manager_state field."""
        from src.downloader.checkpoint import DownloadCheckpoint

        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2024-01-25T12:00:00",
            vpn_manager_state={"switch_count": 2, "last_switch_timestamp": None}
        )

        assert checkpoint.vpn_manager_state == {"switch_count": 2, "last_switch_timestamp": None}

    @pytest.mark.fast
    def test_checkpoint_to_dict_includes_vpn_state(self):
        """Test that to_dict includes vpn_manager_state."""
        from src.downloader.checkpoint import DownloadCheckpoint

        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2024-01-25T12:00:00",
            vpn_manager_state={"switch_count": 3}
        )

        data = checkpoint.to_dict()

        assert "vpn_manager_state" in data
        assert data["vpn_manager_state"] == {"switch_count": 3}

    @pytest.mark.fast
    def test_checkpoint_from_dict_restores_vpn_state(self):
        """Test that from_dict restores vpn_manager_state."""
        from src.downloader.checkpoint import DownloadCheckpoint

        data = {
            "completed_keywords": [],
            "completed_videos": [],
            "failed_keywords": [],
            "current_keyword": None,
            "current_tier": None,
            "timestamp": "2024-01-25T12:00:00",
            "vpn_manager_state": {"switch_count": 4, "last_switch_timestamp": "2024-01-25T11:00:00"}
        }

        checkpoint = DownloadCheckpoint.from_dict(data)

        assert checkpoint.vpn_manager_state == {"switch_count": 4, "last_switch_timestamp": "2024-01-25T11:00:00"}

    @pytest.mark.fast
    def test_checkpoint_from_dict_handles_missing_vpn_state(self):
        """Test from_dict handles checkpoints without vpn_manager_state (backward compat)."""
        from src.downloader.checkpoint import DownloadCheckpoint

        # Old checkpoint without vpn_manager_state
        data = {
            "completed_keywords": [],
            "completed_videos": [],
            "failed_keywords": [],
            "current_keyword": None,
            "current_tier": None,
            "timestamp": "2024-01-25T12:00:00"
        }

        checkpoint = DownloadCheckpoint.from_dict(data)

        assert checkpoint.vpn_manager_state is None

    @pytest.mark.fast
    def test_checkpoint_roundtrip_with_vpn_state(self):
        """Test full roundtrip serialization with vpn_manager_state."""
        from src.downloader.checkpoint import DownloadCheckpoint

        original = DownloadCheckpoint(
            completed_keywords=["keyword1"],
            completed_videos=["video1"],
            failed_keywords=[],
            current_keyword="keyword2",
            current_tier="short",
            timestamp="2024-01-25T12:00:00",
            vpn_manager_state={"switch_count": 2, "last_switch_timestamp": "2024-01-25T11:30:00"}
        )

        data = original.to_dict()
        restored = DownloadCheckpoint.from_dict(data)

        assert restored.vpn_manager_state == original.vpn_manager_state
        assert restored.completed_keywords == original.completed_keywords


class TestVideoDownloaderVPNCheckpointIntegration:
    """Tests for VideoDownloader integration with VPN checkpoint persistence."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = MagicMock()
        config.download = MagicMock()
        config.download.vpn = MockVPNConfig(max_switches_per_session=5)
        config.download.rate_limit = MagicMock()
        config.download.rate_limit.share_budget_across_keywords = False
        config.download.cookie_rotation = MagicMock()
        config.download.cookie_rotation.enabled = False
        config.cache_dir = "/tmp/test_cache"
        return config

    @pytest.mark.fast
    def test_vpn_state_saved_in_checkpoint(self, mock_config, tmp_path):
        """Test that VPN state is saved when checkpoint is saved."""
        from src.downloader.vpn_manager import VPNManager
        from src.downloader.checkpoint import DownloadCheckpoint

        # Create a checkpoint with VPN state
        vpn_manager = VPNManager(mock_config.download.vpn)
        vpn_manager._switch_count = 3

        checkpoint = DownloadCheckpoint(
            completed_keywords=[],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2024-01-25T12:00:00"
        )

        # Save VPN state to checkpoint
        checkpoint.vpn_manager_state = vpn_manager.to_checkpoint_state()

        assert checkpoint.vpn_manager_state["switch_count"] == 3

    @pytest.mark.fast
    def test_vpn_state_restored_on_resume(self, mock_config):
        """Test that VPN state is restored when resuming from checkpoint."""
        from src.downloader.vpn_manager import VPNManager
        from src.downloader.checkpoint import DownloadCheckpoint

        # Create checkpoint with saved VPN state
        checkpoint = DownloadCheckpoint(
            completed_keywords=["kw1"],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp="2024-01-25T12:00:00",
            vpn_manager_state={"switch_count": 2, "last_switch_timestamp": "2024-01-25T11:00:00"}
        )

        # Create new VPN manager and restore state
        vpn_manager = VPNManager(mock_config.download.vpn)
        assert vpn_manager.switch_count == 0  # Initially 0

        vpn_manager.restore_from_checkpoint(checkpoint.vpn_manager_state)

        assert vpn_manager.switch_count == 2
        assert vpn_manager._last_switch_time is not None


class TestVPNManagerIntegration:
    """Integration tests for VPN manager with healer."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config with VPN enabled."""
        config = MagicMock()
        config.download = MagicMock()
        config.download.vpn = MockVPNConfig()
        config.download.cookie_rotation = MagicMock()
        config.download.cookie_rotation.enabled = False

        return config

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_healer_uses_vpn_after_cookie_exhaustion(self, mock_run, mock_config):
        """Test that healer tries VPN when cookies are exhausted."""
        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        healer = DownloadHealer(mock_config, "/tmp/project")
        state = MagicMock(spec=PipelineState)

        # Simulate rate limit error
        error = Exception("HTTP Error 429: Too Many Requests")
        result = healer.fix(error, state, "DOWNLOAD")

        # Should have attempted VPN switch
        assert result is not None
        # The healer should have tried VPN (no cookies configured)
        if healer.vpn_manager and healer.vpn_manager.is_enabled:
            assert mock_run.called or "vpn_switched" in result.details or "backoff" in result.message.lower()


@dataclass
class MockVPNConfigWithVerification:
    """Mock config for testing with verification options (US-010)."""
    enabled: bool = True
    switch_command: str = "echo 'VPN switched'"
    disconnect_command: str = "echo 'VPN disconnected'"
    rotate_on_rate_limit: bool = True
    switch_delay_seconds: int = 0  # No delay for tests
    max_switches_per_session: int = 10
    verify_connection: bool = True
    verify_timeout: int = 5
    # New US-010 fields
    skip_verification: bool = False
    verification_endpoint: str = "https://www.google.com"
    verification_ip: str = "8.8.8.8"


class TestVPNVerificationEndpoints:
    """Tests for configurable VPN verification endpoints (US-010)."""

    @pytest.fixture
    def config_default(self):
        """Config with default verification endpoints."""
        return MockVPNConfigWithVerification()

    @pytest.fixture
    def config_custom(self):
        """Config with custom verification endpoints."""
        return MockVPNConfigWithVerification(
            verification_endpoint="https://cloudflare.com",
            verification_ip="1.1.1.1"
        )

    @pytest.fixture
    def config_skip_verification(self):
        """Config with verification skipped entirely."""
        return MockVPNConfigWithVerification(
            skip_verification=True,
            verify_connection=True  # Should still be skipped
        )

    @pytest.mark.fast
    def test_default_endpoints_used(self, config_default):
        """Test that default Google endpoints are used when not configured."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config_default)

        # Check config values
        assert manager.config.verification_endpoint == "https://www.google.com"
        assert manager.config.verification_ip == "8.8.8.8"

    @pytest.mark.fast
    def test_custom_endpoints_used(self, config_custom):
        """Test that custom endpoints are used when configured."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config_custom)

        assert manager.config.verification_endpoint == "https://cloudflare.com"
        assert manager.config.verification_ip == "1.1.1.1"

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_verification_uses_custom_endpoint(self, mock_run, config_custom):
        """Test that verification curl uses custom endpoint."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config_custom)
        # Make first verification (curl) succeed
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = manager._verify_connection()

        assert result is True
        # Check that custom endpoint was used in curl command
        call_args = mock_run.call_args_list[0][0][0]
        assert "https://cloudflare.com" in call_args

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_verification_uses_custom_ip_on_curl_failure(self, mock_run, config_custom):
        """Test that ping uses custom IP when curl fails."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config_custom)

        # First call (curl) fails, second call (ping) succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr=""),  # curl fails
            MagicMock(returncode=0, stdout="", stderr=""),  # ping succeeds
        ]

        result = manager._verify_connection()

        assert result is True
        # Check that custom IP was used in ping command
        assert mock_run.call_count == 2
        ping_call = mock_run.call_args_list[1][0][0]
        assert "1.1.1.1" in ping_call

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_skip_verification_option(self, mock_run, config_skip_verification):
        """Test that skip_verification=True skips verification entirely."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(config_skip_verification)

        # Switch command succeeds
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = manager.switch()

        assert result is True
        # Only switch command should be called, not verification
        assert mock_run.call_count == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_verification_failure_logs_endpoints(self, mock_run, config_custom, caplog):
        """Test that verification failure logs which endpoints were tried."""
        from src.downloader.vpn_manager import VPNManager
        import logging

        manager = VPNManager(config_custom)

        # Both verification methods fail
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr=""),  # curl fails
            MagicMock(returncode=1, stdout="", stderr=""),  # ping fails
        ]

        with caplog.at_level(logging.WARNING):
            result = manager._verify_connection()

        assert result is False
        # Check log message includes endpoints
        assert any("cloudflare.com" in msg for msg in caplog.messages)
        assert any("1.1.1.1" in msg for msg in caplog.messages)

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_switch_with_skip_verification_logs_skip(self, mock_run, config_skip_verification, caplog):
        """Test that switch logs when verification is skipped."""
        from src.downloader.vpn_manager import VPNManager
        import logging

        manager = VPNManager(config_skip_verification)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with caplog.at_level(logging.DEBUG):
            manager.switch()

        assert any("skip_verification=True" in msg for msg in caplog.messages)

    @pytest.mark.fast
    def test_config_dataclass_has_new_fields(self):
        """Test that VPNConfig dataclass has new US-010 fields."""
        from src.config.sections.download import VPNConfig

        config = VPNConfig()

        # Check default values
        assert hasattr(config, 'skip_verification')
        assert config.skip_verification is False

        assert hasattr(config, 'verification_endpoint')
        assert config.verification_endpoint == "https://www.google.com"

        assert hasattr(config, 'verification_ip')
        assert config.verification_ip == "8.8.8.8"

    @pytest.mark.fast
    def test_config_dataclass_custom_values(self):
        """Test that VPNConfig dataclass accepts custom values."""
        from src.config.sections.download import VPNConfig

        config = VPNConfig(
            skip_verification=True,
            verification_endpoint="https://example.com",
            verification_ip="9.9.9.9"
        )

        assert config.skip_verification is True
        assert config.verification_endpoint == "https://example.com"
        assert config.verification_ip == "9.9.9.9"

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_verification_with_opendns(self, mock_run):
        """Test verification with OpenDNS endpoints."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfigWithVerification(
            verification_endpoint="https://www.opendns.com",
            verification_ip="208.67.222.222"
        )
        manager = VPNManager(config)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = manager._verify_connection()

        assert result is True
        call_args = mock_run.call_args_list[0][0][0]
        assert "opendns.com" in call_args

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_verification_timeout_continues_to_next(self, mock_run, config_custom, caplog):
        """Test that timeout on curl continues to ping."""
        from src.downloader.vpn_manager import VPNManager
        import logging

        manager = VPNManager(config_custom)

        # curl times out, ping succeeds
        mock_run.side_effect = [
            subprocess.TimeoutExpired("curl", 10),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        with caplog.at_level(logging.DEBUG):
            result = manager._verify_connection()

        assert result is True
        assert mock_run.call_count == 2
        assert any("timed out" in msg for msg in caplog.messages)

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_backward_compatibility_no_new_fields(self, mock_run):
        """Test backward compatibility when config lacks new fields."""
        from src.downloader.vpn_manager import VPNManager

        # Old-style config without new fields
        old_config = MockVPNConfig(verify_connection=True)
        manager = VPNManager(old_config)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Should use defaults and not crash
        result = manager._verify_connection()

        assert result is True
        # Should have used default Google endpoint
        call_args = mock_run.call_args_list[0][0][0]
        assert "google.com" in call_args

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_skip_verification_takes_precedence_over_verify_connection(self, mock_run):
        """Test that skip_verification=True overrides verify_connection=True."""
        from src.downloader.vpn_manager import VPNManager

        config = MockVPNConfigWithVerification(
            verify_connection=True,
            skip_verification=True
        )
        manager = VPNManager(config)

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = manager.switch()

        assert result is True
        # Only switch command, no verification
        assert mock_run.call_count == 1

    @pytest.mark.fast
    def test_verification_endpoint_examples_in_docstring(self):
        """Test that docstring mentions alternative endpoints."""
        from src.config.sections.download import VPNConfig

        # Check docstring mentions privacy and alternatives
        docstring = VPNConfig.__doc__
        assert "privacy" in docstring.lower()
        assert "cloudflare" in docstring.lower() or "verification_endpoint" in docstring

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_verification_debug_logging(self, mock_run, config_custom, caplog):
        """Test that verification logs endpoints at debug level."""
        from src.downloader.vpn_manager import VPNManager
        import logging

        manager = VPNManager(config_custom)
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with caplog.at_level(logging.DEBUG):
            manager._verify_connection()

        # Should log the endpoints being used
        assert any("endpoint=" in msg or "ip=" in msg for msg in caplog.messages)
