"""Tests for MullvadVPN class."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestMullvadVPNConnect:
    """Tests for MullvadVPN.connect() method."""

    def test_mullvad_connect_calls_cli(self):
        """Test that connect() calls `mullvad connect` CLI command."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = vpn.connect()

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args == ["mullvad", "connect"]


class TestMullvadVPNDisconnect:
    """Tests for MullvadVPN.disconnect() method."""

    def test_mullvad_disconnect_calls_cli(self):
        """Test that disconnect() calls `mullvad disconnect` CLI command."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = vpn.disconnect()

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args == ["mullvad", "disconnect"]


class TestMullvadVPNVerifyConnection:
    """Tests for MullvadVPN.verify_connection() method."""

    def test_mullvad_verify_connection_parses_json(self):
        """Test that verify_connection() parses am.i.mullvad.net JSON response."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        mock_response = {
            "ip": "185.65.134.100",
            "country": "Sweden",
            "city": "Gothenburg",
            "mullvad_exit_ip": True,
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_response),
                stderr=""
            )
            result = vpn.verify_connection()

            assert result is True
            # Verify curl was called with am.i.mullvad.net
            call_args = mock_run.call_args[0][0]
            assert "curl" in call_args
            assert "https://am.i.mullvad.net/json" in call_args
            # Verify IP was stored
            assert vpn._last_verified_ip == "185.65.134.100"

    def test_mullvad_verify_connection_not_connected(self):
        """Test that verify_connection() returns False when not on Mullvad."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        mock_response = {
            "ip": "8.8.8.8",
            "country": "United States",
            "city": "Mountain View",
            "mullvad_exit_ip": False,
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_response),
                stderr=""
            )
            result = vpn.verify_connection()

            assert result is False

    def test_mullvad_verify_connection_fallback_on_unreachable(self):
        """Test fallback to ping verification when am.i.mullvad.net unreachable."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            # First call (curl) fails
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Connection refused")

            with patch.object(vpn, "_fallback_verification", return_value=True) as mock_fallback:
                result = vpn.verify_connection()

                assert result is True
                mock_fallback.assert_called_once()


class TestMullvadVPNRotateServer:
    """Tests for MullvadVPN.rotate_server() method."""

    def test_mullvad_rotate_server_changes_location(self):
        """Test that rotate_server() uses `mullvad relay set location` for rotation."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.max_switches_per_session = 10  # Required by can_switch()

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = vpn.rotate_server(country="de")

            assert result is True
            # Should have called mullvad relay set location
            calls = mock_run.call_args_list
            assert len(calls) >= 2  # relay set + reconnect
            # First call should be relay set location
            first_call = calls[0][0][0]
            assert first_call == ["mullvad", "relay", "set", "location", "de"]

    def test_mullvad_rotate_server_resets_circuit_breaker(self):
        """Test that rotate_server() resets circuit breaker on successful rotation."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.max_switches_per_session = 10

        vpn = MullvadVPN(config)

        # Create circuit breaker with some failure state
        cb_config = CircuitBreakerConfig(enabled=True, consecutive_failures_threshold=3)
        circuit_breaker = CircuitBreaker(cb_config)

        # Trip the circuit breaker
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.is_open is True
        assert circuit_breaker.state.consecutive_failures == 3

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = vpn.rotate_server(country="de", circuit_breaker=circuit_breaker)

            assert result is True
            # Circuit breaker should be reset after VPN rotation
            assert circuit_breaker.is_open is False
            assert circuit_breaker.state.consecutive_failures == 0

    def test_mullvad_rotate_server_no_circuit_breaker_reset_on_failure(self):
        """Test that circuit breaker is NOT reset when rotation fails."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.max_switches_per_session = 10

        vpn = MullvadVPN(config)

        # Create circuit breaker with some failure state
        cb_config = CircuitBreakerConfig(enabled=True, consecutive_failures_threshold=3)
        circuit_breaker = CircuitBreaker(cb_config)

        # Trip the circuit breaker
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        assert circuit_breaker.is_open is True

        with patch("subprocess.run") as mock_run:
            # Simulate rotation failure
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")
            result = vpn.rotate_server(country="de", circuit_breaker=circuit_breaker)

            assert result is False
            # Circuit breaker should still be tripped (not reset on failure)
            assert circuit_breaker.is_open is True
            assert circuit_breaker.state.consecutive_failures == 3

    def test_mullvad_rotate_server_without_circuit_breaker(self):
        """Test that rotate_server() works without circuit_breaker parameter."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.max_switches_per_session = 10

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            # Should work without circuit_breaker parameter (backwards compatible)
            result = vpn.rotate_server(country="de")

            assert result is True


class TestMullvadVPNGetStatus:
    """Tests for MullvadVPN.get_status() method."""

    def test_mullvad_get_status_parses_connected(self):
        """Test get_status() parses connected status output."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Connected to se-got-wg-001 in Gothenburg, Sweden",
                stderr=""
            )
            status = vpn.get_status()

            assert status["connected"] is True
            assert status["city"] == "Gothenburg"
            assert status["country"] == "Sweden"

    def test_mullvad_get_status_parses_disconnected(self):
        """Test get_status() parses disconnected status output."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0

        vpn = MullvadVPN(config)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Disconnected",
                stderr=""
            )
            status = vpn.get_status()

            assert status["connected"] is False
