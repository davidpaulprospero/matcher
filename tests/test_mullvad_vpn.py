"""Tests for MullvadVPN class."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestMullvadVPNIsAvailable:
    """Tests for MullvadVPN.is_available() static method (US-35-009)."""

    def test_is_available_returns_true_when_cli_found(self):
        """Test is_available() returns True when mullvad CLI runs successfully."""
        from src.downloader.mullvad_vpn import MullvadVPN

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="mullvad 2023.5",
                stderr=""
            )
            result = MullvadVPN.is_available()

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args == ["mullvad", "--version"]

    def test_is_available_returns_false_when_cli_not_found(self):
        """Test is_available() returns False when mullvad CLI is not installed."""
        from src.downloader.mullvad_vpn import MullvadVPN

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("mullvad not found")
            result = MullvadVPN.is_available()

            assert result is False

    def test_is_available_returns_false_on_nonzero_exit(self):
        """Test is_available() returns False when mullvad returns error."""
        from src.downloader.mullvad_vpn import MullvadVPN

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="unknown command"
            )
            result = MullvadVPN.is_available()

            assert result is False

    def test_is_available_returns_false_on_timeout(self):
        """Test is_available() returns False when mullvad times out."""
        from src.downloader.mullvad_vpn import MullvadVPN

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("mullvad", 10)
            result = MullvadVPN.is_available()

            assert result is False

    def test_is_available_is_static_method(self):
        """Test that is_available() is a static method callable without instance."""
        from src.downloader.mullvad_vpn import MullvadVPN

        # Should be callable as a static method (without creating an instance)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="mullvad 2023.5",
                stderr=""
            )
            # Call as class method, not instance method
            result = MullvadVPN.is_available()
            assert isinstance(result, bool)


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
        config.max_rotations_per_session = 10  # Required by MullvadVPN

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
        config.max_rotations_per_session = 10  # Required by MullvadVPN

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
        config.max_rotations_per_session = 10  # Required by MullvadVPN

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
        config.max_rotations_per_session = 10  # Required by MullvadVPN

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


class TestMullvadVPNPreferredCountries:
    """Tests for MullvadVPN._pick_next_country() with preferred_countries config (US-35-010)."""

    def test_pick_next_country_uses_config_preferred_countries(self):
        """Test that _pick_next_country uses config.preferred_countries when set."""
        from src.downloader.mullvad_vpn import MULLVAD_COUNTRIES, MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        # Set custom preferred countries
        config.preferred_countries = ['fr', 'it', 'es']

        vpn = MullvadVPN(config)

        # Pick countries multiple times to verify they come from config
        picked_countries = set()
        for _ in range(20):
            vpn._used_countries = []  # Reset to get fresh picks
            country = vpn._pick_next_country()
            picked_countries.add(country)

        # All picked countries should be from config.preferred_countries
        assert picked_countries.issubset({'fr', 'it', 'es'})
        # Should NOT include countries from hardcoded list that aren't in config
        assert not picked_countries.intersection({'us', 'gb', 'de', 'nl', 'se', 'ch', 'ca', 'au', 'jp', 'sg'} - {'fr', 'it', 'es'})

    def test_pick_next_country_falls_back_when_config_empty(self):
        """Test that _pick_next_country falls back to MULLVAD_COUNTRIES when config.preferred_countries is empty."""
        from src.downloader.mullvad_vpn import MULLVAD_COUNTRIES, MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        # Empty list should fall back to default
        config.preferred_countries = []

        vpn = MullvadVPN(config)

        picked_countries = set()
        for _ in range(30):
            vpn._used_countries = []
            country = vpn._pick_next_country()
            picked_countries.add(country)

        # All picked countries should be from hardcoded MULLVAD_COUNTRIES
        assert picked_countries.issubset(set(MULLVAD_COUNTRIES))

    def test_pick_next_country_falls_back_when_config_none(self):
        """Test that _pick_next_country falls back to MULLVAD_COUNTRIES when config.preferred_countries is None."""
        from src.downloader.mullvad_vpn import MULLVAD_COUNTRIES, MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        # None should fall back to default
        config.preferred_countries = None

        vpn = MullvadVPN(config)

        picked_countries = set()
        for _ in range(30):
            vpn._used_countries = []
            country = vpn._pick_next_country()
            picked_countries.add(country)

        # All picked countries should be from hardcoded MULLVAD_COUNTRIES
        assert picked_countries.issubset(set(MULLVAD_COUNTRIES))

    def test_pick_next_country_falls_back_when_config_missing(self):
        """Test that _pick_next_country falls back when preferred_countries attribute is missing."""
        from src.downloader.mullvad_vpn import MULLVAD_COUNTRIES, MullvadVPN

        # Use spec to ensure preferred_countries attribute doesn't exist
        config = MagicMock(spec=['enabled', 'max_vpn_switches', 'switch_delay_seconds', 'min_switch_interval', 'switch_command'])
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.switch_command = ""  # Required by VPNManager.is_enabled
        # Don't set preferred_countries at all - MagicMock with spec won't have it

        vpn = MullvadVPN(config)

        picked_countries = set()
        for _ in range(30):
            vpn._used_countries = []
            country = vpn._pick_next_country()
            picked_countries.add(country)

        # All picked countries should be from hardcoded MULLVAD_COUNTRIES
        assert picked_countries.issubset(set(MULLVAD_COUNTRIES))


class TestMullvadVPNMaxRotations:
    """Tests for MullvadVPN max_rotations_per_session limit (US-35-011)."""

    def test_rotate_server_stops_at_max_rotations(self):
        """Test that rotate_server() returns False when max_rotations_per_session is reached."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10  # Base class limit (high)
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 2  # Mullvad-specific limit (low)
        config.preferred_countries = ['us', 'de']

        vpn = MullvadVPN(config)

        # Simulate having already used max rotations
        vpn._switch_count = 2

        # rotate_server should return False since max_rotations reached
        result = vpn.rotate_server()
        assert result is False

    def test_rotate_server_allows_under_max_rotations(self):
        """Test that rotate_server() allows rotation when under max_rotations limit."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 5
        config.max_switches_per_session = 10  # Required by can_switch()
        config.preferred_countries = ['us', 'de']

        vpn = MullvadVPN(config)

        # Simulate having used some but not all rotations
        vpn._switch_count = 3

        # Mock the subprocess calls to succeed
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            result = vpn.rotate_server()

        # Should succeed and increment switch_count
        assert result is True
        assert vpn._switch_count == 4

    def test_max_rotations_defaults_to_5(self):
        """Test that _max_rotations defaults to 5 if config attribute is missing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        # Config without max_rotations_per_session attribute
        config = MagicMock(spec=['enabled', 'max_vpn_switches', 'switch_delay_seconds', 'min_switch_interval', 'switch_command'])
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.switch_command = ""

        vpn = MullvadVPN(config)

        # Should default to 5
        assert vpn._max_rotations == 5

    def test_max_rotations_uses_config_value(self):
        """Test that _max_rotations uses config.max_rotations_per_session when set."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.max_rotations_per_session = 3

        vpn = MullvadVPN(config)

        # Should use config value
        assert vpn._max_rotations == 3
