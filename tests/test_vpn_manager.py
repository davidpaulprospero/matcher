"""Comprehensive tests for vpn_manager.py - VPN CLI integration for IP rotation.

Tests cover:
- Happy path (normal usage)
- Edge cases (empty, null, boundary values)
- Error cases (invalid input, failures, exceptions)
- Branch coverage for all conditional paths
"""

import subprocess
import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.downloader.vpn_manager import (
    VPNProvider,
    VPNStatus,
    VPNConnection,
    VPNProviderConfig,
    VPNManager,
    VPNRotationHelper,
    PROVIDER_CONFIGS,
    detect_vpn,
    nordvpn_connect,
    mullvad_set_location,
    protonvpn_random,
)


# ============================================================================
# VPNProvider and VPNStatus Enum Tests
# ============================================================================

class TestEnums:
    """Tests for enum types."""

    def test_vpn_provider_values(self):
        """All provider values should be strings."""
        assert VPNProvider.NORDVPN.value == "nordvpn"
        assert VPNProvider.MULLVAD.value == "mullvad"
        assert VPNProvider.PROTONVPN.value == "protonvpn"
        assert VPNProvider.WIREGUARD.value == "wireguard"
        assert VPNProvider.OPENVPN.value == "openvpn"
        assert VPNProvider.UNKNOWN.value == "unknown"

    def test_vpn_status_values(self):
        """All status values should be strings."""
        assert VPNStatus.CONNECTED.value == "connected"
        assert VPNStatus.DISCONNECTED.value == "disconnected"
        assert VPNStatus.CONNECTING.value == "connecting"
        assert VPNStatus.ERROR.value == "error"
        assert VPNStatus.UNKNOWN.value == "unknown"


# ============================================================================
# VPNConnection Tests
# ============================================================================

class TestVPNConnection:
    """Tests for VPNConnection dataclass."""

    # --- Happy Path Tests ---

    @pytest.mark.parametrize("status,expected", [
        (VPNStatus.CONNECTED, True),
        (VPNStatus.DISCONNECTED, False),
        (VPNStatus.CONNECTING, False),
        (VPNStatus.ERROR, False),
        (VPNStatus.UNKNOWN, False),
    ])
    def test_is_connected_all_statuses(self, status, expected):
        """Should correctly report connected status."""
        conn = VPNConnection(provider=VPNProvider.NORDVPN, status=status)
        assert conn.is_connected is expected

    def test_duration_seconds_connected(self):
        """Should calculate connection duration."""
        conn = VPNConnection(
            provider=VPNProvider.NORDVPN,
            status=VPNStatus.CONNECTED,
            connected_at=time.time() - 60,  # 60 seconds ago
        )
        assert conn.duration_seconds >= 59  # Allow for test timing

    def test_duration_seconds_not_connected(self):
        """Should return 0 when not connected."""
        conn = VPNConnection(
            provider=VPNProvider.NORDVPN,
            status=VPNStatus.DISCONNECTED,
        )
        assert conn.duration_seconds == 0.0

    def test_duration_seconds_no_timestamp(self):
        """Should return 0 when no connected_at timestamp."""
        conn = VPNConnection(
            provider=VPNProvider.NORDVPN,
            status=VPNStatus.CONNECTED,
            connected_at=0.0,
        )
        assert conn.duration_seconds == 0.0

    # --- Edge Cases ---

    def test_connection_all_fields(self):
        """Should store all connection fields."""
        conn = VPNConnection(
            provider=VPNProvider.NORDVPN,
            status=VPNStatus.CONNECTED,
            server="us1234.nordvpn.com",
            country="United States",
            city="New York",
            ip_address="192.168.1.1",
            connected_at=time.time(),
            protocol="NordLynx",
        )
        assert conn.server == "us1234.nordvpn.com"
        assert conn.country == "United States"
        assert conn.city == "New York"
        assert conn.ip_address == "192.168.1.1"
        assert conn.protocol == "NordLynx"

    def test_connection_defaults(self):
        """Should have sensible defaults."""
        conn = VPNConnection(provider=VPNProvider.UNKNOWN, status=VPNStatus.UNKNOWN)
        assert conn.server == ""
        assert conn.country == ""
        assert conn.city == ""
        assert conn.ip_address == ""
        assert conn.connected_at == 0.0
        assert conn.protocol == ""


# ============================================================================
# VPNProviderConfig Tests
# ============================================================================

class TestVPNProviderConfig:
    """Tests for VPNProviderConfig dataclass."""

    def test_config_defaults(self):
        """Should have sensible defaults."""
        config = VPNProviderConfig(
            provider=VPNProvider.NORDVPN,
            cli_command="nordvpn",
        )
        assert config.connect_args == []
        assert config.disconnect_args == []
        assert config.status_args == []
        assert config.rotate_args == []
        assert config.preferred_countries == ["US", "UK", "CA", "DE", "NL"]
        assert config.socks5_port == 0

    def test_config_custom_values(self):
        """Should accept custom values."""
        config = VPNProviderConfig(
            provider=VPNProvider.NORDVPN,
            cli_command="nordvpn",
            connect_args=["connect"],
            socks5_port=1080,
        )
        assert config.connect_args == ["connect"]
        assert config.socks5_port == 1080


# ============================================================================
# Provider Configs Tests
# ============================================================================

class TestProviderConfigs:
    """Tests for pre-configured provider settings."""

    def test_nordvpn_config(self):
        """NordVPN config should be valid."""
        config = PROVIDER_CONFIGS[VPNProvider.NORDVPN]
        assert config.cli_command == "nordvpn"
        assert "connect" in config.connect_args
        assert "disconnect" in config.disconnect_args
        assert "status" in config.status_args
        assert config.socks5_port == 1080

    def test_mullvad_config(self):
        """Mullvad config should be valid."""
        config = PROVIDER_CONFIGS[VPNProvider.MULLVAD]
        assert config.cli_command == "mullvad"
        assert "connect" in config.connect_args
        assert "disconnect" in config.disconnect_args
        assert config.socks5_port == 1080

    def test_protonvpn_config(self):
        """ProtonVPN config should be valid."""
        config = PROVIDER_CONFIGS[VPNProvider.PROTONVPN]
        assert config.cli_command == "protonvpn-cli"
        # Either fastest or random should be in connect/rotate args
        all_args = config.connect_args + config.rotate_args
        assert any("--fastest" in arg or "--random" in arg for arg in all_args)

    def test_wireguard_config(self):
        """WireGuard config should be valid."""
        config = PROVIDER_CONFIGS[VPNProvider.WIREGUARD]
        assert config.cli_command == "wg-quick"
        assert "up" in config.connect_args
        assert "down" in config.disconnect_args

    def test_all_providers_have_configs(self):
        """All major providers should have configs."""
        expected = [VPNProvider.NORDVPN, VPNProvider.MULLVAD, VPNProvider.PROTONVPN, VPNProvider.WIREGUARD]
        for provider in expected:
            assert provider in PROVIDER_CONFIGS


# ============================================================================
# VPNManager Tests
# ============================================================================

class TestVPNManager:
    """Tests for VPNManager class."""

    # --- Initialization Tests ---

    def test_init_no_config(self):
        """Should initialize without config."""
        with patch.object(VPNManager, '_detect_provider'):
            manager = VPNManager()
            assert manager.is_available is False  # No provider detected in test

    def test_init_with_preferred_provider(self):
        """Should use preferred provider if CLI available."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            assert manager.detected_provider == VPNProvider.NORDVPN

    def test_init_with_custom_countries(self):
        """Should accept custom preferred countries."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(
                preferred_provider=VPNProvider.NORDVPN,
                preferred_countries=["JP", "KR", "SG"],
            )
            assert manager._preferred_countries == ["JP", "KR", "SG"]

    # --- Provider Detection Tests ---

    def test_detect_provider_nordvpn(self):
        """Should detect NordVPN when available."""
        def mock_which(cmd):
            if cmd == "nordvpn":
                return "/usr/bin/nordvpn"
            return None

        with patch('shutil.which', side_effect=mock_which):
            manager = VPNManager()
            assert manager.detected_provider == VPNProvider.NORDVPN

    def test_detect_provider_mullvad(self):
        """Should detect Mullvad when NordVPN not available."""
        def mock_which(cmd):
            if cmd == "mullvad":
                return "/usr/bin/mullvad"
            return None

        with patch('shutil.which', side_effect=mock_which):
            manager = VPNManager()
            assert manager.detected_provider == VPNProvider.MULLVAD

    def test_detect_provider_protonvpn(self):
        """Should detect ProtonVPN."""
        def mock_which(cmd):
            if cmd == "protonvpn-cli":
                return "/usr/bin/protonvpn-cli"
            return None

        with patch('shutil.which', side_effect=mock_which):
            manager = VPNManager()
            assert manager.detected_provider == VPNProvider.PROTONVPN

    def test_detect_provider_wireguard(self):
        """Should detect WireGuard."""
        def mock_which(cmd):
            if cmd == "wg-quick":
                return "/usr/bin/wg-quick"
            return None

        with patch('shutil.which', side_effect=mock_which):
            manager = VPNManager()
            assert manager.detected_provider == VPNProvider.WIREGUARD

    def test_detect_provider_none(self):
        """Should return None when no VPN detected."""
        with patch('shutil.which', return_value=None):
            manager = VPNManager()
            assert manager.detected_provider is None
            assert manager.is_available is False

    def test_detect_provider_priority(self):
        """Should prefer NordVPN over other providers."""
        with patch('shutil.which', return_value='/usr/bin/some-vpn'):
            manager = VPNManager()
            # If all providers return a path, should pick nordvpn first
            assert manager.detected_provider == VPNProvider.NORDVPN

    # --- Connection Tests ---

    def test_connect_success(self):
        """Should connect successfully."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Connected", stderr="")

                result = manager.connect(country="US")
                assert result is True

    def test_connect_with_server(self):
        """Should connect to specific server."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Connected", stderr="")

                result = manager.connect(server="us1234")
                assert result is True
                # Server should be in the command
                args = mock_run.call_args[0][0]
                assert "us1234" in args

    def test_connect_uses_preferred_country(self):
        """Should use first preferred country if no country specified."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(
                preferred_provider=VPNProvider.NORDVPN,
                preferred_countries=["DE", "FR"],
            )

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Connected", stderr="")

                manager.connect()  # No country specified
                args = mock_run.call_args[0][0]
                assert "DE" in args  # First preferred country

    def test_connect_failure(self):
        """Should handle connection failure."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=1, stdout="", stderr="Connection failed")

                result = manager.connect(country="US")
                assert result is False

    def test_connect_timeout(self):
        """Should handle connection timeout."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(cmd="nordvpn", timeout=60)

                result = manager.connect(country="US")
                assert result is False

    def test_connect_exception(self):
        """Should handle unexpected exceptions."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = Exception("Unexpected error")

                result = manager.connect(country="US")
                assert result is False

    def test_connect_no_provider(self):
        """Should fail gracefully without provider."""
        with patch('shutil.which', return_value=None):
            manager = VPNManager()

            result = manager.connect(country="US")
            assert result is False

    # --- Disconnection Tests ---

    def test_disconnect_success(self):
        """Should disconnect successfully."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Disconnected", stderr="")

                result = manager.disconnect()
                assert result is True

    def test_disconnect_failure(self):
        """Should handle disconnect failure."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error")

                result = manager.disconnect()
                assert result is False

    def test_disconnect_exception(self):
        """Should handle unexpected exceptions."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = Exception("Unexpected error")

                result = manager.disconnect()
                assert result is False

    def test_disconnect_no_provider(self):
        """Should fail gracefully without provider."""
        with patch('shutil.which', return_value=None):
            manager = VPNManager()

            result = manager.disconnect()
            assert result is False

    # --- Rotation Tests ---

    def test_rotate_success(self):
        """Should rotate to new server."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            manager._cooldown_seconds = 0  # Disable cooldown for test

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Connected to new server", stderr="")

                result = manager.rotate()
                assert result is True
                assert manager._rotation_count == 1

    def test_rotate_cycles_countries(self):
        """Should cycle through preferred countries."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(
                preferred_provider=VPNProvider.NORDVPN,
                preferred_countries=["US", "UK", "DE"],
            )
            manager._cooldown_seconds = 0

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Connected", stderr="")

                # Rotate multiple times
                countries_used = []
                for _ in range(6):
                    manager.rotate()
                    args = mock_run.call_args[0][0]
                    # Find which country was used (lowercase in command)
                    for c in ["us", "uk", "de"]:
                        if c in args:
                            countries_used.append(c)
                            break

                # Should have cycled through countries
                assert len(set(countries_used)) >= 2

    def test_rotate_respects_cooldown(self):
        """Should respect rotation cooldown."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            manager._cooldown_seconds = 60
            manager._last_rotation = time.time()  # Just rotated

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                with patch('time.sleep') as mock_sleep:
                    manager.rotate()
                    # Should have called sleep for cooldown
                    mock_sleep.assert_called()

    def test_rotate_fallback_to_reconnect(self):
        """Should fallback to disconnect/connect if rotate fails."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            manager._cooldown_seconds = 0
            # Clear rotate args to force fallback
            manager._provider_config.rotate_args = []

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                with patch('time.sleep'):
                    result = manager.rotate()
                    # Should have called disconnect then connect
                    assert mock_run.call_count >= 2

    def test_rotate_no_provider(self):
        """Should fail gracefully without provider."""
        with patch('shutil.which', return_value=None):
            manager = VPNManager()

            result = manager.rotate()
            assert result is False

    # --- Status Tests ---

    def test_get_status_connected(self):
        """Should parse connected status."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="Status: Connected\nCurrent server: us1234.nordvpn.com\nCountry: United States",
                    stderr=""
                )

                status = manager.get_status()
                assert status.is_connected is True
                assert "us1234" in status.server.lower()

    def test_get_status_disconnected(self):
        """Should parse disconnected status."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="Status: Disconnected",
                    stderr=""
                )

                status = manager.get_status()
                assert status.is_connected is False

    @pytest.mark.parametrize("output,expected_connected", [
        ("Status: Connected", True),
        ("Status: Disconnected", False),
        ("Connection: Active", True),
        ("Connection: Inactive", False),
        ("Tunnel: running", True),
        ("Tunnel: stopped", False),
        ("Not connected", False),
        ("connected to server", True),
        ("Disconnected from server", False),
    ])
    def test_parse_status_patterns(self, output, expected_connected):
        """Should parse various status output patterns."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            connection = manager._parse_status(output)
            assert connection.is_connected is expected_connected

    def test_get_status_error(self):
        """Should handle status check error."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = Exception("Status error")

                status = manager.get_status()
                assert status.status == VPNStatus.ERROR

    def test_get_status_no_provider(self):
        """Should return unknown without provider."""
        with patch('shutil.which', return_value=None):
            manager = VPNManager()

            status = manager.get_status()
            assert status.provider == VPNProvider.UNKNOWN
            assert status.status == VPNStatus.UNKNOWN

    def test_get_status_command_failure(self):
        """Should handle status command failure."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error")

                status = manager.get_status()
                assert status.is_connected is False

    # --- Parse Status Tests for Mullvad ---

    def test_parse_status_mullvad(self):
        """Should parse Mullvad status output."""
        with patch('shutil.which', return_value='/usr/bin/mullvad'):
            manager = VPNManager(preferred_provider=VPNProvider.MULLVAD)

            output = "Connected to se-got-wg-001"
            connection = manager._parse_status(output)
            assert connection.is_connected is True
            assert "se-got-wg-001" in connection.server

    # --- SOCKS5 Proxy Tests ---

    def test_socks5_proxy(self):
        """Should return SOCKS5 proxy URL when available."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            assert manager.socks5_proxy == "socks5://127.0.0.1:1080"

    def test_socks5_proxy_none(self):
        """Should return None when SOCKS5 not available."""
        with patch('shutil.which', return_value=None):
            manager = VPNManager()
            assert manager.socks5_proxy is None

    def test_socks5_proxy_no_port(self):
        """Should return None when provider has no SOCKS5 port."""
        with patch('shutil.which', return_value='/usr/bin/protonvpn-cli'):
            manager = VPNManager(preferred_provider=VPNProvider.PROTONVPN)
            # ProtonVPN config has socks5_port=0
            assert manager.socks5_proxy is None

    # --- Properties Tests ---

    def test_is_connected_property(self):
        """is_connected should check current status."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(manager, 'get_status') as mock_status:
                mock_status.return_value = VPNConnection(
                    provider=VPNProvider.NORDVPN,
                    status=VPNStatus.CONNECTED,
                )
                assert manager.is_connected is True

                mock_status.return_value = VPNConnection(
                    provider=VPNProvider.NORDVPN,
                    status=VPNStatus.DISCONNECTED,
                )
                assert manager.is_connected is False

    def test_current_server(self):
        """Should return current server from connection."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            manager._connection = VPNConnection(
                provider=VPNProvider.NORDVPN,
                status=VPNStatus.CONNECTED,
                server="us1234.nordvpn.com",
            )
            assert manager.current_server == "us1234.nordvpn.com"

    def test_current_server_empty(self):
        """Should return empty string without connection."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            assert manager.current_server == ""

    # --- Statistics Tests ---

    def test_get_stats(self):
        """Should return manager statistics."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(manager, 'get_status') as mock_status:
                mock_status.return_value = VPNConnection(
                    provider=VPNProvider.NORDVPN,
                    status=VPNStatus.CONNECTED,
                    server="us1234.nordvpn.com"
                )

                stats = manager.get_stats()
                assert stats["provider"] == "nordvpn"
                assert stats["available"] is True
                assert "socks5_proxy" in stats
                assert stats["connected"] is True

    # --- IP Check Tests ---

    def test_get_current_ip_success(self):
        """Should get current IP address."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('httpx.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = "1.2.3.4\n"
                mock_get.return_value = mock_response

                ip = manager.get_current_ip()
                assert ip == "1.2.3.4"

    def test_get_current_ip_fallback(self):
        """Should try multiple IP services."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            call_count = 0

            def mock_get(url, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise Exception("Service error")
                response = Mock()
                response.status_code = 200
                response.text = "5.6.7.8"
                return response

            with patch('httpx.get', side_effect=mock_get):
                ip = manager.get_current_ip()
                assert ip == "5.6.7.8"
                assert call_count == 3

    def test_get_current_ip_all_fail(self):
        """Should return None when all IP services fail."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch('httpx.get', side_effect=Exception("All failed")):
                ip = manager.get_current_ip()
                assert ip is None

    # --- Config Loading Tests ---

    def test_load_provider_from_config(self):
        """Should load provider from config."""
        mock_config = Mock()
        mock_config.download.fallback.vpn = {
            'enabled': True,
            'provider': 'nordvpn',
        }

        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(mock_config)
            assert manager.detected_provider == VPNProvider.NORDVPN

    def test_load_countries_from_config(self):
        """Should load preferred countries from config."""
        mock_config = Mock()
        mock_config.download.fallback.vpn = {
            'enabled': True,
            'provider': 'nordvpn',
            'preferred_countries': ['DE', 'FR', 'NL'],
        }

        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(mock_config)
            assert manager._preferred_countries == ['DE', 'FR', 'NL']

    def test_load_cooldown_from_config(self):
        """Should load rotation cooldown from config."""
        mock_config = Mock()
        mock_config.download.fallback.vpn = {
            'enabled': True,
            'provider': 'nordvpn',
            'rotation_cooldown': 120.0,
        }

        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(mock_config)
            assert manager._cooldown_seconds == 120.0

    def test_disabled_in_config(self):
        """Should respect disabled flag."""
        mock_config = Mock()
        mock_config.download.fallback.vpn = {
            'enabled': False,
        }

        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(mock_config)
            assert manager._enabled is False

    def test_load_config_object_style(self):
        """Should load from config object (not dict)."""
        mock_vpn = Mock()
        mock_vpn.enabled = True
        mock_vpn.provider = "mullvad"
        mock_vpn.preferred_countries = ["SE", "FI"]
        mock_vpn.rotation_cooldown = 45.0

        mock_config = Mock()
        mock_config.download.fallback.vpn = mock_vpn

        with patch('shutil.which', return_value='/usr/bin/mullvad'):
            manager = VPNManager(mock_config)
            assert manager._preferred_countries == ["SE", "FI"]

    def test_load_config_unknown_provider(self):
        """Should handle unknown provider in config."""
        mock_config = Mock()
        mock_config.download.fallback.vpn = {
            'enabled': True,
            'provider': 'unknown_vpn',
        }

        with patch('shutil.which', return_value=None):
            manager = VPNManager(mock_config)
            # Should not crash, but provider won't be set

    def test_load_config_no_fallback(self):
        """Should handle missing fallback config."""
        mock_config = Mock()
        mock_config.download.fallback = None

        with patch('shutil.which', return_value=None):
            manager = VPNManager(mock_config)
            # Should not crash

    def test_load_config_exception(self):
        """Should handle config loading errors gracefully."""
        mock_config = Mock()
        mock_config.download.fallback.vpn = Mock(
            side_effect=Exception("Config error")
        )

        # Should not crash
        with patch('shutil.which', return_value=None):
            manager = VPNManager(mock_config)


# ============================================================================
# VPNRotationHelper Tests
# ============================================================================

class TestVPNRotationHelper:
    """Tests for VPNRotationHelper class."""

    # --- Initialization Tests ---

    def test_init(self):
        """Should initialize with manager."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            helper = VPNRotationHelper(vpn_manager)

            assert helper.max_rotations_per_hour == 10
            assert helper.min_rotation_interval == 60.0

    def test_init_custom_limits(self):
        """Should accept custom limits."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            helper = VPNRotationHelper(
                vpn_manager,
                max_rotations_per_hour=5,
                min_rotation_interval=120.0,
            )

            assert helper.max_rotations_per_hour == 5
            assert helper.min_rotation_interval == 120.0

    # --- should_rotate Tests ---

    def test_should_rotate_yes(self):
        """Should allow rotation when conditions met."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            helper = VPNRotationHelper(vpn_manager, min_rotation_interval=0)
            assert helper.should_rotate() is True

    def test_should_rotate_no_interval(self):
        """Should not allow rotation within interval."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            helper = VPNRotationHelper(vpn_manager, min_rotation_interval=300)
            helper._last_rotation = time.time()  # Just rotated

            assert helper.should_rotate() is False

    def test_should_rotate_no_vpn_available(self):
        """Should not allow rotation when VPN unavailable."""
        with patch('shutil.which', return_value=None):
            vpn_manager = VPNManager()

            helper = VPNRotationHelper(vpn_manager)
            assert helper.should_rotate() is False

    def test_should_rotate_no_hourly_limit(self):
        """Should not allow rotation when hourly limit reached."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            helper = VPNRotationHelper(
                vpn_manager,
                max_rotations_per_hour=3,
                min_rotation_interval=0
            )

            # Simulate 3 rotations in the last hour
            now = time.time()
            helper._rotation_times = [now - 100, now - 200, now - 300]

            assert helper.should_rotate() is False

    def test_should_rotate_hourly_limit_expired(self):
        """Should allow rotation when old rotations expired."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            helper = VPNRotationHelper(
                vpn_manager,
                max_rotations_per_hour=3,
                min_rotation_interval=0,
            )

            # Simulate 3 rotations over an hour ago
            now = time.time()
            helper._rotation_times = [now - 4000, now - 4100, now - 4200]

            assert helper.should_rotate() is True

    # --- rotate_on_rate_limit Tests ---

    def test_rotate_on_rate_limit_success(self):
        """Should rotate and track stats on rate limit."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'rotate', return_value=True):
                helper = VPNRotationHelper(vpn_manager, min_rotation_interval=0)

                result = helper.rotate_on_rate_limit()
                assert result is True
                assert len(helper._rotation_times) == 1

    def test_rotate_on_rate_limit_tracks_rate_limits(self):
        """Should track rate limits since last rotation."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'rotate', return_value=True):
                helper = VPNRotationHelper(vpn_manager, min_rotation_interval=0)

                # Multiple rate limits before rotation
                helper._rate_limits_since_rotation = 5
                helper.rotate_on_rate_limit()

                # Counter should reset on successful rotation
                assert helper._rate_limits_since_rotation == 0

    def test_rotate_on_rate_limit_failure(self):
        """Should handle rotation failure."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'rotate', return_value=False):
                helper = VPNRotationHelper(vpn_manager, min_rotation_interval=0)

                result = helper.rotate_on_rate_limit()
                assert result is False

    def test_rotate_on_rate_limit_blocked(self):
        """Should return False when rotation blocked."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            helper = VPNRotationHelper(vpn_manager, min_rotation_interval=300)
            helper._last_rotation = time.time()  # Just rotated

            result = helper.rotate_on_rate_limit()
            assert result is False

    def test_rotate_on_rate_limit_cleans_old_times(self):
        """Should clean up old rotation times."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'rotate', return_value=True):
                helper = VPNRotationHelper(vpn_manager, min_rotation_interval=0)

                # Add old rotation times
                now = time.time()
                helper._rotation_times = [now - 4000, now - 3700]  # Over an hour ago

                helper.rotate_on_rate_limit()

                # Old times should be cleaned, only new one remains
                assert len(helper._rotation_times) == 1

    # --- Statistics Tests ---

    def test_get_stats(self):
        """Should return rotation statistics."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'get_status') as mock_status:
                mock_status.return_value = VPNConnection(
                    provider=VPNProvider.NORDVPN,
                    status=VPNStatus.CONNECTED,
                )

                helper = VPNRotationHelper(vpn_manager)
                stats = helper.get_stats()

                assert "rotations_this_hour" in stats
                assert "max_rotations_per_hour" in stats
                assert "rate_limits_since_rotation" in stats
                assert "vpn_stats" in stats

    def test_get_stats_last_rotation_ago(self):
        """Should calculate time since last rotation."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'get_status') as mock_status:
                mock_status.return_value = VPNConnection(
                    provider=VPNProvider.NORDVPN,
                    status=VPNStatus.CONNECTED,
                )

                helper = VPNRotationHelper(vpn_manager)
                helper._last_rotation = time.time() - 60  # 1 minute ago

                stats = helper.get_stats()
                assert stats["last_rotation_ago"] >= 59

    def test_get_stats_no_last_rotation(self):
        """Should handle no previous rotation."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            with patch.object(vpn_manager, 'get_status') as mock_status:
                mock_status.return_value = VPNConnection(
                    provider=VPNProvider.NORDVPN,
                    status=VPNStatus.CONNECTED,
                )

                helper = VPNRotationHelper(vpn_manager)
                stats = helper.get_stats()
                assert stats["last_rotation_ago"] is None

    # --- report_success Tests ---

    def test_report_success(self):
        """report_success should not raise."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            vpn_manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            helper = VPNRotationHelper(vpn_manager)

            # Should not raise
            helper.report_success()


# ============================================================================
# Helper Functions Tests
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_detect_vpn_nordvpn(self):
        """Should detect NordVPN."""
        def mock_which(cmd):
            return "/usr/bin/nordvpn" if cmd == "nordvpn" else None

        with patch('shutil.which', side_effect=mock_which):
            provider = detect_vpn()
            assert provider == VPNProvider.NORDVPN

    def test_detect_vpn_mullvad(self):
        """Should detect Mullvad."""
        def mock_which(cmd):
            return "/usr/bin/mullvad" if cmd == "mullvad" else None

        with patch('shutil.which', side_effect=mock_which):
            provider = detect_vpn()
            assert provider == VPNProvider.MULLVAD

    def test_detect_vpn_none(self):
        """Should return None when no VPN found."""
        with patch('shutil.which', return_value=None):
            provider = detect_vpn()
            assert provider is None

    def test_nordvpn_connect(self):
        """Should call NordVPN connect."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = nordvpn_connect("US")
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "nordvpn" in args
            assert "connect" in args
            assert "US" in args

    def test_nordvpn_connect_with_group(self):
        """Should call NordVPN connect with group."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = nordvpn_connect("US", group="P2P")
            assert result is True
            args = mock_run.call_args[0][0]
            assert "--group" in args
            assert "P2P" in args

    def test_nordvpn_connect_failure(self):
        """Should handle NordVPN connect failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)

            result = nordvpn_connect("US")
            assert result is False

    def test_nordvpn_connect_exception(self):
        """Should handle exception."""
        with patch('subprocess.run', side_effect=Exception("Error")):
            result = nordvpn_connect("US")
            assert result is False

    def test_mullvad_set_location(self):
        """Should call Mullvad set location."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = mullvad_set_location("us", "nyc")
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "mullvad" in args
            assert "us" in args
            assert "nyc" in args

    def test_mullvad_set_location_country_only(self):
        """Should call Mullvad with country only."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = mullvad_set_location("de")
            assert result is True
            args = mock_run.call_args[0][0]
            assert "de" in args

    def test_mullvad_set_location_failure(self):
        """Should handle Mullvad failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)

            result = mullvad_set_location("us")
            assert result is False

    def test_mullvad_set_location_exception(self):
        """Should handle exception."""
        with patch('subprocess.run', side_effect=Exception("Error")):
            result = mullvad_set_location("us")
            assert result is False

    def test_protonvpn_random(self):
        """Should call ProtonVPN random connect."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            result = protonvpn_random()
            assert result is True
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert "protonvpn-cli" in args
            assert "--random" in args

    def test_protonvpn_random_failure(self):
        """Should handle ProtonVPN failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1)

            result = protonvpn_random()
            assert result is False

    def test_protonvpn_random_exception(self):
        """Should handle exception."""
        with patch('subprocess.run', side_effect=Exception("Error")):
            result = protonvpn_random()
            assert result is False


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================

class TestEdgeCasesAndBoundaries:
    """Tests for edge cases and boundary conditions."""

    def test_rotation_count_overflow(self):
        """Should handle many rotations without overflow."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)
            manager._cooldown_seconds = 0

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

                for _ in range(1000):
                    manager.rotate()

                assert manager._rotation_count == 1000

    def test_empty_preferred_countries(self):
        """Should handle empty preferred countries list."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(
                preferred_provider=VPNProvider.NORDVPN,
                preferred_countries=[],
            )
            manager._cooldown_seconds = 0

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

                # Should not crash
                manager.rotate()

    def test_long_status_output(self):
        """Should handle very long status output."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            long_output = "Status: Connected\n" + "extra line\n" * 1000
            connection = manager._parse_status(long_output)
            assert connection.is_connected is True

    def test_unicode_in_server_name(self):
        """Should handle unicode in server names."""
        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            manager = VPNManager(preferred_provider=VPNProvider.NORDVPN)

            output = "Status: Connected\nCurrent server: münchen-de-001.nordvpn.com"
            connection = manager._parse_status(output)
            assert connection.is_connected is True
