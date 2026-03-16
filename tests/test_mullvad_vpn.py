"""Tests for MullvadVPN class."""

import json
import subprocess
from typing import List
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
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0
        config.min_server_success_rate = 0.7

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
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0
        config.min_server_success_rate = 0.7

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
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0
        config.min_server_success_rate = 0.7

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
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

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


class TestMullvadVPNRotationCooldown:
    """Tests for MullvadVPN rotation_delay_seconds cooldown (US-35-012)."""

    def test_rotate_server_enforces_cooldown(self):
        """Test that rotate_server() returns False when cooldown is active."""
        import time
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.max_switches_per_session = 10
        config.rotation_delay_seconds = 10  # 10 second cooldown
        config.preferred_countries = ['us', 'de']

        vpn = MullvadVPN(config)

        # Simulate a recent rotation (1 second ago)
        vpn._last_rotation_time = time.time() - 1

        # Attempt rotation - should fail due to cooldown
        result = vpn.rotate_server()
        assert result is False

    def test_rotate_server_allows_after_cooldown_expires(self):
        """Test that rotate_server() allows rotation after cooldown expires."""
        import time
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.max_switches_per_session = 10
        config.rotation_delay_seconds = 5  # 5 second cooldown
        config.preferred_countries = ['us', 'de']
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

        vpn = MullvadVPN(config)

        # Simulate an old rotation (10 seconds ago - beyond 5s cooldown)
        vpn._last_rotation_time = time.time() - 10

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            result = vpn.rotate_server()

        # Should succeed since cooldown has expired
        assert result is True

    def test_rotate_server_updates_last_rotation_time(self):
        """Test that successful rotation updates _last_rotation_time."""
        import time
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.max_switches_per_session = 10
        config.rotation_delay_seconds = 5
        config.preferred_countries = ['us', 'de']
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

        vpn = MullvadVPN(config)

        # Initially 0
        assert vpn._last_rotation_time == 0.0

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            before = time.time()
            result = vpn.rotate_server()
            after = time.time()

        assert result is True
        # _last_rotation_time should be updated to around now
        assert before <= vpn._last_rotation_time <= after

    def test_rotation_delay_defaults_to_5(self):
        """Test that _rotation_delay defaults to 5 if config attribute is missing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        # Config without rotation_delay_seconds attribute
        config = MagicMock(spec=['enabled', 'max_vpn_switches', 'switch_delay_seconds', 'min_switch_interval', 'switch_command'])
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.switch_command = ""

        vpn = MullvadVPN(config)

        # Should default to 5
        assert vpn._rotation_delay == 5.0

    def test_rotation_delay_uses_config_value(self):
        """Test that _rotation_delay uses config.rotation_delay_seconds when set."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.rotation_delay_seconds = 15

        vpn = MullvadVPN(config)

        # Should use config value
        assert vpn._rotation_delay == 15.0

    def test_first_rotation_always_allowed(self):
        """Test that first rotation is always allowed (no prior rotation)."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.max_switches_per_session = 10
        config.rotation_delay_seconds = 60  # Long cooldown
        config.preferred_countries = ['us', 'de']
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

        vpn = MullvadVPN(config)

        # _last_rotation_time starts at 0
        assert vpn._last_rotation_time == 0.0

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            result = vpn.rotate_server()

        # First rotation should succeed despite long cooldown setting
        assert result is True


class TestMullvadVPNExponentialBackoff:
    """Tests for MullvadVPN exponential backoff on rotation delays (US-67-007)."""

    def _make_vpn(self, initial_delay=5.0, max_delay=60.0, rotation_delay=0):
        """Create a MullvadVPN with backoff config for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.max_switches_per_session = 10
        config.rotation_delay_seconds = rotation_delay
        config.preferred_countries = ['us', 'de', 'gb']
        config.initial_rotation_delay_seconds = initial_delay
        config.max_rotation_delay_seconds = max_delay

        return MullvadVPN(config)

    def test_backoff_progression_5_10_20_40_60(self):
        """Test backoff delays: 5s, 10s, 20s, 40s, 60s (capped at max_rotation_delay)."""
        vpn = self._make_vpn(initial_delay=5.0, max_delay=60.0)

        expected = [5.0, 10.0, 20.0, 40.0, 60.0]
        actual = []
        for i in range(5):
            vpn._backoff_count = i
            actual.append(vpn._compute_backoff_delay())

        assert actual == expected

    def test_backoff_cap_prevents_exceeding_max(self):
        """Test that backoff never exceeds max_rotation_delay_seconds."""
        vpn = self._make_vpn(initial_delay=5.0, max_delay=60.0)

        # At very high backoff count, delay should still be capped
        vpn._backoff_count = 100
        assert vpn._compute_backoff_delay() == 60.0

    def test_backoff_resets_after_success(self):
        """Test that reset_backoff() resets the backoff counter to 0."""
        vpn = self._make_vpn()

        # Simulate several rotations
        vpn._backoff_count = 4

        # 5 * 2^4 = 80, capped at 60
        assert vpn._compute_backoff_delay() == 60.0

        # Reset backoff
        vpn.reset_backoff()

        # Counter should be 0, delay back to base
        assert vpn._backoff_count == 0
        assert vpn._compute_backoff_delay() == 5.0

    def test_backoff_reset_is_noop_when_already_zero(self):
        """Test that reset_backoff() is safe to call when backoff is already 0."""
        vpn = self._make_vpn()

        assert vpn._backoff_count == 0
        vpn.reset_backoff()  # Should not raise
        assert vpn._backoff_count == 0

    def test_rotate_server_increments_backoff(self):
        """Test that each rotation increments the backoff counter."""
        vpn = self._make_vpn(initial_delay=0.01, max_delay=1.0)

        assert vpn._backoff_count == 0

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            vpn.rotate_server(country='us')

        assert vpn._backoff_count == 1

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
            vpn.rotate_server(country='de')

        assert vpn._backoff_count == 2

    def test_rotate_server_applies_backoff_delay(self):
        """Test that rotate_server() sleeps for the computed backoff delay."""
        vpn = self._make_vpn(initial_delay=5.0, max_delay=60.0)

        with patch('subprocess.run') as mock_run, \
             patch('time.sleep') as mock_sleep:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

            # First rotation: delay = 5 * 2^0 = 5s
            vpn.rotate_server(country='us')

            # time.sleep should have been called with 5.0 for backoff
            sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
            assert 5.0 in sleep_calls

    def test_backoff_uses_config_values(self):
        """Test that backoff reads initial and max delay from config."""
        vpn = self._make_vpn(initial_delay=10.0, max_delay=120.0)

        assert vpn._initial_rotation_delay == 10.0
        assert vpn._max_rotation_delay == 120.0

        # First delay should be 10s
        vpn._backoff_count = 0
        assert vpn._compute_backoff_delay() == 10.0

        # Second delay should be 20s
        vpn._backoff_count = 1
        assert vpn._compute_backoff_delay() == 20.0

        # Capped at 120s
        vpn._backoff_count = 5  # 10 * 32 = 320 -> capped at 120
        assert vpn._compute_backoff_delay() == 120.0

    def test_backoff_defaults_when_config_missing(self):
        """Test that backoff defaults to 5s initial and 60s max when config is missing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock(spec=[
            'enabled', 'max_vpn_switches', 'switch_delay_seconds',
            'min_switch_interval', 'switch_command'
        ])
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.switch_command = ""

        vpn = MullvadVPN(config)

        assert vpn._initial_rotation_delay == 5.0
        assert vpn._max_rotation_delay == 60.0


class TestServerSuccessRecord:
    """Tests for ServerSuccessRecord dataclass (US-109-012)."""

    def test_success_rate_defaults_to_05_for_untested(self):
        """Test that untested servers have 0.5 default success rate."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        assert record.success_rate == 0.5

    def test_success_rate_calculates_correctly(self):
        """Test success rate calculation."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        record.attempts = 10
        record.successes = 7
        assert record.success_rate == 0.7

    def test_record_success_increments_both(self):
        """Test that record_success increments attempts and successes."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        record.record_success()

        assert record.attempts == 1
        assert record.successes == 1

    def test_record_failure_increments_attempts_only(self):
        """Test that record_failure increments attempts but not successes."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        record.record_failure()

        assert record.attempts == 1
        assert record.successes == 0

    def test_should_swap_returns_true_at_threshold(self):
        """Test that should_swap returns True at threshold (3 failures by default)."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        # 0 successes, 3 attempts = 3 consecutive failures
        record.attempts = 3
        record.successes = 0

        assert record.should_swap(3) is True

    def test_should_swap_returns_false_below_threshold(self):
        """Test that should_swap returns False below threshold."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        record.attempts = 2
        record.successes = 0

        assert record.should_swap(3) is False

    def test_should_swap_returns_false_for_untested(self):
        """Test that should_swap returns False for untested servers."""
        from src.downloader.mullvad_vpn import ServerSuccessRecord

        record = ServerSuccessRecord("us")
        # 0 attempts
        assert record.should_swap(3) is False


class TestMullvadVPNGeographicProximity:
    """Tests for MullvadVPN geographic proximity scoring (US-109-012)."""

    def _make_vpn(self, preferred_countries=None):
        """Create a MullvadVPN for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.preferred_countries = preferred_countries or ['us', 'de', 'gb', 'jp', 'au']
        config.min_server_success_rate = 0.7
        config.swap_failure_threshold = 3
        config.rotation_delay_seconds = 0
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

        return MullvadVPN(config)

    def test_geographic_proximity_boost_same_region(self):
        """Test that same region servers get +0.3 boost."""
        vpn = self._make_vpn()

        # Simulate successful download in US (NA region)
        vpn._last_successful_region = "na"

        # US and CA are both NA region
        scored = vpn._score_countries_for_selection(["us", "de", "jp"])

        # US should have highest score due to region proximity
        scores = dict(scored)
        assert scores["us"] > scores["de"]
        assert scores["us"] > scores["jp"]

    def test_geographic_proximity_no_boost_different_region(self):
        """Test that different region servers don't get proximity boost."""
        vpn = self._make_vpn()

        # Last success was in EU region
        vpn._last_successful_region = "eu"

        scored = vpn._score_countries_for_selection(["us", "de"])
        scores = dict(scored)

        # de (EU) should have higher score than us (NA)
        assert scores["de"] > scores["us"]

    def test_success_rate_boost_above_threshold(self):
        """Test that servers with >70% success rate get +0.5 boost."""
        vpn = self._make_vpn()

        # Pre-populate server history with high success rate
        from src.downloader.mullvad_vpn import ServerSuccessRecord
        vpn._server_history["us"] = ServerSuccessRecord("us")
        vpn._server_history["us"].attempts = 10
        vpn._server_history["us"].successes = 8  # 80% success rate

        vpn._server_history["de"] = ServerSuccessRecord("de")
        vpn._server_history["de"].attempts = 5
        vpn._server_history["de"].successes = 1  # 20% success rate

        scored = vpn._score_countries_for_selection(["us", "de"])
        scores = dict(scored)

        # US should score higher due to success rate
        assert scores["us"] > scores["de"]
        assert scores["us"] >= 0.5  # Base + success rate boost

    def test_success_rate_penalty_below_30_percent(self):
        """Test that servers with <30% success rate get -0.3 penalty."""
        vpn = self._make_vpn()

        # Pre-populate with poor success rate
        from src.downloader.mullvad_vpn import ServerSuccessRecord
        vpn._server_history["us"] = ServerSuccessRecord("us")
        vpn._server_history["us"].attempts = 10
        vpn._server_history["us"].successes = 2  # 20% success rate

        scored = vpn._score_countries_for_selection(["us"])
        scores = dict(scored)

        # Should have penalty
        assert scores["us"] < 0  # Base 0 - penalty

    def test_combined_scoring_multiple_factors(self):
        """Test combined scoring with both geographic and success rate factors."""
        vpn = self._make_vpn()

        # Last success in APAC region
        vpn._last_successful_region = "apac"

        # Pre-populate server history
        from src.downloader.mullvad_vpn import ServerSuccessRecord
        # JP: APAC + 80% success = 0.3 + 0.5 = 0.8
        vpn._server_history["jp"] = ServerSuccessRecord("jp")
        vpn._server_history["jp"].attempts = 10
        vpn._server_history["jp"].successes = 8

        # DE: EU + 80% success = 0 (different region) + 0.5 = 0.5
        vpn._server_history["de"] = ServerSuccessRecord("de")
        vpn._server_history["de"].attempts = 10
        vpn._server_history["de"].successes = 8

        scored = vpn._score_countries_for_selection(["jp", "de"])
        scores = dict(scored)

        # JP should have higher score due to region proximity
        assert scores["jp"] > scores["de"]


class TestMullvadVPNServerHistoryTracking:
    """Tests for MullvadVPN server history tracking (US-109-012)."""

    def _make_vpn(self):
        """Create a MullvadVPN for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.preferred_countries = ['us', 'de', 'gb']
        config.min_server_success_rate = 0.7
        config.swap_failure_threshold = 3
        config.rotation_delay_seconds = 0

        return MullvadVPN(config)

    def test_record_server_success_creates_record(self):
        """Test that record_server_success creates a new record if none exists."""
        vpn = self._make_vpn()

        vpn.record_server_success("us")

        assert "us" in vpn._server_history
        assert vpn._server_history["us"].attempts == 1
        assert vpn._server_history["us"].successes == 1

    def test_record_server_success_updates_existing(self):
        """Test that record_server_success updates existing record."""
        vpn = self._make_vpn()

        vpn.record_server_success("us")
        vpn.record_server_success("us")

        assert vpn._server_history["us"].attempts == 2
        assert vpn._server_history["us"].successes == 2

    def test_record_server_failure_updates_record(self):
        """Test that record_server_failure updates failure count."""
        vpn = self._make_vpn()

        vpn.record_server_failure("us")
        vpn.record_server_failure("us")

        assert vpn._server_history["us"].attempts == 2
        assert vpn._server_history["us"].successes == 0

    def test_record_server_success_updates_region(self):
        """Test that record_server_success updates last successful region."""
        vpn = self._make_vpn()

        vpn._current_country = "us"
        vpn.record_server_success("us")

        assert vpn._last_successful_region == "na"  # US is in NA region

    def test_get_server_stats_returns_history(self):
        """Test that get_server_stats returns server history."""
        vpn = self._make_vpn()

        vpn.record_server_success("us")
        vpn.record_server_failure("de")

        stats = vpn.get_server_stats()

        assert "us" in stats["server_history"]
        assert "de" in stats["server_history"]
        assert stats["server_history"]["us"]["success_rate"] == 1.0
        assert stats["server_history"]["de"]["success_rate"] == 0.0


class TestMullvadVPNSwapRecommendation:
    """Tests for MullvadVPN server swap recommendation (US-109-012)."""

    def _make_vpn(self):
        """Create a MullvadVPN for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.preferred_countries = ['us', 'de', 'gb', 'jp', 'au']
        config.min_server_success_rate = 0.7
        config.swap_failure_threshold = 3
        config.rotation_delay_seconds = 0
        # Required for latency-based selection (US-114-008)
        config.prefer_low_latency = False
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

        return MullvadVPN(config)

    def test_swap_recommendation_none_when_no_record(self):
        """Test that swap recommendation returns None when no record exists."""
        vpn = self._make_vpn()

        assert vpn.get_swap_recommendation() is None

    def test_swap_recommendation_returns_country_at_threshold(self):
        """Test that swap recommendation returns alternative at failure threshold."""
        vpn = self._make_vpn()

        # Pre-populate with failing server
        from src.downloader.mullvad_vpn import ServerSuccessRecord
        vpn._current_country = "us"
        vpn._server_history["us"] = ServerSuccessRecord("us")
        vpn._server_history["us"].attempts = 3
        vpn._server_history["us"].successes = 0
        vpn._current_server_record = vpn._server_history["us"]

        recommendation = vpn.get_swap_recommendation()

        assert recommendation is not None
        assert recommendation != "us"  # Should recommend different country

    def test_swap_recommendation_none_below_threshold(self):
        """Test that swap recommendation returns None below failure threshold."""
        vpn = self._make_vpn()

        from src.downloader.mullvad_vpn import ServerSuccessRecord
        vpn._current_country = "us"
        vpn._server_history["us"] = ServerSuccessRecord("us")
        vpn._server_history["us"].attempts = 2
        vpn._server_history["us"].successes = 0
        vpn._current_server_record = vpn._server_history["us"]

        assert vpn.get_swap_recommendation() is None

    def test_swap_recommendation_avoids_same_region(self):
        """Test that swap recommendation prefers different region."""
        vpn = self._make_vpn()

        # Fail US (NA region)
        from src.downloader.mullvad_vpn import ServerSuccessRecord
        vpn._current_country = "us"
        vpn._server_history["us"] = ServerSuccessRecord("us")
        vpn._server_history["us"].attempts = 5
        vpn._server_history["us"].successes = 0
        vpn._current_server_record = vpn._server_history["us"]

        recommendation = vpn.get_swap_recommendation()

        # Recommendation should not be in NA region (us, ca)
        assert recommendation in ["de", "gb", "jp", "au", "sg", "se", "nl", "ch"]


class TestMullvadVPNDegradation:
    """Tests for VPN graceful degradation on consecutive failures (US-113-012)."""

    def _make_vpn(self, max_consecutive_vpn_failures=5):
        """Create a MullvadVPN with degradation config for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.max_vpn_switches = 10
        config.switch_delay_seconds = 0
        config.min_switch_interval = 0
        config.verify_connection = False
        config.skip_verification = True
        config.max_rotations_per_session = 10
        config.rotation_delay_seconds = 0
        config.preferred_countries = ['us', 'de']
        config.max_consecutive_vpn_failures = max_consecutive_vpn_failures

        return MullvadVPN(config)

    def test_consecutive_failures_counter_starts_at_zero(self):
        """Test that consecutive failures counter starts at 0."""
        vpn = self._make_vpn()
        assert vpn._consecutive_vpn_failures == 0
        assert vpn._is_degraded is False
        assert vpn._degradation_override is False

    def test_record_vpn_failure_increments_counter(self):
        """Test that record_vpn_failure increments the consecutive failures counter."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=5)

        vpn.record_vpn_failure()
        assert vpn._consecutive_vpn_failures == 1

        vpn.record_vpn_failure()
        assert vpn._consecutive_vpn_failures == 2

    def test_record_vpn_success_resets_counter(self):
        """Test that record_vpn_success resets the consecutive failures counter."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=5)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()
        assert vpn._consecutive_vpn_failures == 2

        vpn.record_vpn_success()
        assert vpn._consecutive_vpn_failures == 0

    def test_degradation_triggers_at_threshold(self):
        """Test that degradation triggers when consecutive failures exceed threshold."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=3)

        # Fail 3 times (threshold)
        vpn.record_vpn_failure()
        vpn.record_vpn_failure()
        assert vpn._is_degraded is False

        vpn.record_vpn_failure()  # 3rd failure = threshold reached
        assert vpn._is_degraded is True

    def test_degradation_status_returns_correct_info(self):
        """Test that get_degradation_status returns correct information."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=3)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()

        status = vpn.get_degradation_status()

        assert status['is_degraded'] is False
        assert status['consecutive_failures'] == 2
        assert status['max_consecutive_vpn_failures'] == 3
        assert status['degradation_override'] is False
        assert status['can_use_vpn'] is True

    def test_can_use_vpn_returns_false_when_degraded(self):
        """Test that can_use_vpn returns False when degraded."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=2)

        assert vpn.can_use_vpn() is True

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()  # Triggers degradation

        assert vpn.can_use_vpn() is False
        assert vpn.is_degraded() is True

    def test_can_use_vpn_returns_false_when_max_rotations_reached(self):
        """Test that can_use_vpn returns False when max rotations reached."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=10)

        # Simulate max rotations reached
        vpn._switch_count = vpn._max_rotations

        assert vpn.can_use_vpn() is False

    def test_can_use_vpn_returns_false_when_disabled(self):
        """Test that can_use_vpn returns False when VPN is disabled."""
        config = MagicMock()
        config.enabled = False
        config.max_consecutive_vpn_failures = 5

        from src.downloader.mullvad_vpn import MullvadVPN
        vpn = MullvadVPN(config)

        assert vpn.can_use_vpn() is False

    def test_degradation_override_allows_vpn_when_degraded(self):
        """Test that degradation override allows VPN even when degraded."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=2)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()  # Triggers degradation
        assert vpn.can_use_vpn() is False

        # Override degradation
        vpn.enable_vpn(override_degradation=True)
        assert vpn.can_use_vpn() is True

        # Status shows override is enabled
        status = vpn.get_degradation_status()
        assert status['degradation_override'] is True
        assert status['is_degraded'] is False

    def test_disable_vpn_degradation_override_clears_override(self):
        """Test that disable_vpn_degradation_override clears the override flag."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=2)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()  # Triggers degradation
        vpn.enable_vpn(override_degradation=True)

        assert vpn.can_use_vpn() is True

        # Disable override - should clear override flag
        # Note: degraded state was cleared when override was enabled,
        # so VPN is still usable after disabling override
        vpn.disable_vpn_degradation_override()
        # After disabling override, VPN is still usable because degraded state was cleared
        assert vpn.can_use_vpn() is True

    def test_disable_vpn_degradation_override_keeps_degraded_if_still_degraded(self):
        """Test that if VPN is degraded and override not enabled, disabling override keeps it degraded."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=2)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()  # Triggers degradation

        assert vpn.can_use_vpn() is False
        assert vpn._is_degraded is True

        # Disable override when already degraded (override not enabled)
        vpn.disable_vpn_degradation_override()
        assert vpn.can_use_vpn() is False

    def test_max_consecutive_vpn_failures_zero_disables_degradation(self):
        """Test that max_consecutive_vpn_failures=0 disables degradation."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=0)

        # Even many failures should not trigger degradation
        for _ in range(10):
            vpn.record_vpn_failure()

        assert vpn._is_degraded is False
        assert vpn.can_use_vpn() is True

    def test_record_vpn_success_clears_degraded_state(self):
        """Test that record_vpn_success clears degraded state."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=2)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()  # Triggers degradation
        assert vpn._is_degraded is True

        vpn.record_vpn_success()  # Should clear degraded state
        assert vpn._is_degraded is False

    def test_reset_degradation_resets_all_state(self):
        """Test that reset_degradation resets all degradation state."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=2)

        vpn.record_vpn_failure()
        vpn.record_vpn_failure()  # Triggers degradation
        vpn.enable_vpn(override_degradation=True)

        vpn.reset_degradation()

        assert vpn._consecutive_vpn_failures == 0
        assert vpn._is_degraded is False
        assert vpn._degradation_override is False
        assert vpn.can_use_vpn() is True

    def test_degradation_with_max_failures_one(self):
        """Test degradation with max_consecutive_vpn_failures=1."""
        vpn = self._make_vpn(max_consecutive_vpn_failures=1)

        # First failure should trigger degradation
        vpn.record_vpn_failure()
        assert vpn._is_degraded is True
        assert vpn.can_use_vpn() is False


class TestMullvadVPNLatencySelection:
    """Tests for latency-based server selection (US-114-008)."""

    def _make_vpn(
        self,
        prefer_low_latency: bool = True,
        max_latency_ms: int = 200,
        latency_measurement_timeout: float = 3.0,
    ) -> 'MullvadVPN':
        """Create a MullvadVPN instance with latency config."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.preferred_countries = ['us', 'gb', 'de', 'nl']
        config.rotation_strategy = 'random'
        config.max_rotations_per_session = 5
        config.switch_delay_seconds = 0
        config.verification_timeout = 10
        config.initial_rotation_delay_seconds = 1.0
        config.max_rotation_delay_seconds = 5.0
        config.rotation_delay_seconds = 1
        config.max_consecutive_vpn_failures = 5
        config.prefer_low_latency = prefer_low_latency
        config.max_latency_ms = max_latency_ms
        config.latency_measurement_timeout = latency_measurement_timeout

        return MullvadVPN(config)

    def test_latency_config_loaded_correctly(self):
        """Test that latency config options are loaded from config."""
        vpn = self._make_vpn(
            prefer_low_latency=True,
            max_latency_ms=150,
            latency_measurement_timeout=5.0
        )

        assert vpn._prefer_low_latency is True
        assert vpn._max_latency_ms == 150
        assert vpn._latency_timeout == 5.0

    def test_latency_disabled_when_prefer_low_latency_false(self):
        """Test that latency measurement is disabled when prefer_low_latency=False."""
        vpn = self._make_vpn(prefer_low_latency=False)

        assert vpn._prefer_low_latency is False

    def test_measure_latency_returns_cached_value(self):
        """Test that latency measurement uses cached values."""
        vpn = self._make_vpn()

        # Pre-populate cache
        vpn._server_latency['us'] = 45.0

        latency = vpn.measure_latency('us')

        assert latency == 45.0

    def test_measure_latency_returns_none_when_disabled(self):
        """Test that measure_latency returns None when latency is disabled."""
        vpn = self._make_vpn(prefer_low_latency=False)

        latency = vpn.measure_latency('us')

        assert latency is None

    @patch('src.downloader.mullvad_vpn.MullvadVPN._ping_latency')
    def test_measure_latency_calls_ping(self, mock_ping):
        """Test that measure_latency calls _ping_latency for unknown countries."""
        mock_ping.return_value = 50.0

        vpn = self._make_vpn()
        vpn._server_latency = {}  # Clear cache

        latency = vpn.measure_latency('us')

        assert latency == 50.0
        mock_ping.assert_called_once()

    def test_ping_latency_parses_windows_output(self):
        """Test that _ping_latency parses Windows ping output."""
        import platform
        from unittest.mock import patch

        vpn = self._make_vpn()

        with patch('platform.system') as mock_system:
            mock_system.return_value = 'Windows'

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='Reply from 8.8.8.8: bytes=32 time=45ms TTL=117',
                    stderr=''
                )

                latency = vpn._ping_latency('8.8.8.8')
                assert latency == 45.0

    def test_ping_latency_parses_linux_output(self):
        """Test that _ping_latency parses Linux ping output."""
        import platform
        from unittest.mock import patch

        vpn = self._make_vpn()

        with patch('platform.system') as mock_system:
            mock_system.return_value = 'Linux'

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='64 bytes from 8.8.8.8: icmp_seq=1 ttl=117 time=32.5 ms',
                    stderr=''
                )

                latency = vpn._ping_latency('8.8.8.8')
                assert latency == 32.5

    def test_ping_latency_returns_none_on_failure(self):
        """Test that _ping_latency returns None on ping failure."""
        vpn = self._make_vpn()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout='',
                stderr='Request timed out'
            )

            latency = vpn._ping_latency('8.8.8.8')
            assert latency is None

    def test_ping_latency_returns_none_on_timeout(self):
        """Test that _ping_latency returns None on timeout."""
        vpn = self._make_vpn()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('ping', 5)

            latency = vpn._ping_latency('8.8.8.8')
            assert latency is None

    def test_score_countries_includes_latency_boost(self):
        """Test that _score_countries_for_selection includes latency boost."""
        vpn = self._make_vpn(prefer_low_latency=True, max_latency_ms=200)

        # Pre-populate latency cache with low latency
        vpn._server_latency['us'] = 30.0
        vpn._server_latency['de'] = 150.0

        scored = vpn._score_countries_for_selection(['us', 'de'])

        # us should score higher due to lower latency
        scored_dict = {c: s for c, s in scored}
        assert scored_dict['us'] > scored_dict['de']

    def test_score_countries_applies_high_latency_penalty(self):
        """Test that high latency above threshold applies penalty."""
        vpn = self._make_vpn(prefer_low_latency=True, max_latency_ms=100)

        # Pre-populate latency cache with high latency
        vpn._server_latency['us'] = 150.0

        scored = vpn._score_countries_for_selection(['us'])

        # Should have negative score due to high latency penalty
        assert scored[0][1] < 0

    def test_score_countries_no_penalty_for_unknown_latency(self):
        """Test that unknown latency doesn't apply penalty."""
        vpn = self._make_vpn(prefer_low_latency=True, max_latency_ms=100)

        # No latency data
        scored = vpn._score_countries_for_selection(['us'])

        # Should have 0 score (no penalty for unknown)
        assert scored[0][1] == 0.0

    def test_get_server_stats_includes_latency(self):
        """Test that get_server_stats includes latency information."""
        vpn = self._make_vpn()

        vpn._server_latency['us'] = 45.0
        vpn._server_latency['de'] = 120.0

        stats = vpn.get_server_stats()

        assert 'server_latency' in stats
        assert stats['server_latency']['us'] == 45.0
        assert stats['server_latency']['de'] == 120.0
        assert stats['prefer_low_latency'] is True
        assert stats['max_latency_ms'] == 200

    def test_get_ping_targets_returns_list(self):
        """Test that _get_ping_targets returns a list of targets."""
        vpn = self._make_vpn()

        targets = vpn._get_ping_targets('us')

        assert isinstance(targets, list)
        assert len(targets) > 0

    def test_max_latency_zero_disables_threshold(self):
        """Test that max_latency_ms=0 disables the threshold."""
        vpn = self._make_vpn(prefer_low_latency=True, max_latency_ms=0)

        assert vpn._max_latency_ms == 0

        # Even high latency should not be penalized
        vpn._server_latency['us'] = 500.0
        scored = vpn._score_countries_for_selection(['us'])

        # Should have 0 score (no penalty when threshold is disabled)
        assert scored[0][1] == 0.0


class TestMullvadVPNServerSwitchMetrics:
    """Tests for server switch metrics (US-143-006)."""

    def _make_vpn(self) -> 'MullvadVPN':
        """Create a MullvadVPN instance for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.preferred_countries = ['us', 'gb', 'de', 'nl']
        config.rotation_strategy = 'random'
        config.max_rotations_per_session = 5
        config.switch_delay_seconds = 0
        config.verification_timeout = 10
        config.initial_rotation_delay_seconds = 1.0
        config.max_rotation_delay_seconds = 5.0
        config.rotation_delay_seconds = 1
        config.max_consecutive_vpn_failures = 5
        config.prefer_low_latency = True
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0

        return MullvadVPN(config)

    def test_record_server_switch_event_creates_event(self):
        """Test that record_server_switch_event creates a switch event."""
        vpn = self._make_vpn()

        vpn.record_server_switch_event(from_country="us", to_country="de", reason="rate_limit")

        events = vpn.get_switch_events()
        assert len(events) == 1
        assert events[0]["from_country"] == "us"
        assert events[0]["to_country"] == "de"
        assert events[0]["reason"] == "rate_limit"

    def test_record_server_switch_event_initial_connection(self):
        """Test that initial connection (from=None) is recorded correctly."""
        vpn = self._make_vpn()

        vpn.record_server_switch_event(from_country=None, to_country="us", reason="initial")

        events = vpn.get_switch_events()
        assert len(events) == 1
        assert events[0]["from_country"] is None
        assert events[0]["to_country"] == "us"
        assert events[0]["reason"] == "initial"

    def test_get_switch_metrics_returns_summary(self):
        """Test that get_switch_metrics returns summary statistics."""
        vpn = self._make_vpn()

        # Record some switch events
        vpn.record_server_switch_event(from_country=None, to_country="us", reason="initial")
        vpn.record_server_switch_event(from_country="us", to_country="de", reason="rate_limit")
        vpn.record_server_switch_event(from_country="de", to_country="gb", reason="rate_limit")

        metrics = vpn.get_switch_metrics()

        assert metrics["total_switches"] == 0  # _switch_count not incremented by record_server_switch_event
        assert metrics["events_count"] == 3
        assert metrics["switches_by_reason"]["initial"] == 1
        assert metrics["switches_by_reason"]["rate_limit"] == 2

    def test_switch_events_limited_to_100(self):
        """Test that switch events are limited to last 100."""
        vpn = self._make_vpn()

        # Record more than 100 events
        for i in range(150):
            vpn.record_server_switch_event(
                from_country=f"country_{i % 5}",
                to_country=f"country_{(i + 1) % 5}",
                reason="test"
            )

        events = vpn.get_switch_events()
        assert len(events) == 100

    def test_get_server_stats_includes_switch_metrics(self):
        """Test that get_server_stats includes switch metrics."""
        vpn = self._make_vpn()

        vpn.record_server_switch_event(from_country=None, to_country="us", reason="initial")

        stats = vpn.get_server_stats()

        assert "switch_events" in stats
        assert "switch_metrics" in stats
        assert stats["switch_metrics"]["events_count"] == 1


class TestMullvadVPNServerSelectionIntegration:
    """Integration tests for latency-based server selection (US-143-006)."""

    def _make_vpn(
        self,
        prefer_low_latency: bool = True,
        preferred_countries: List[str] = None,
    ) -> 'MullvadVPN':
        """Create a MullvadVPN instance for testing."""
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.preferred_countries = preferred_countries or ['us', 'gb', 'de', 'nl', 'jp']
        config.rotation_strategy = 'random'
        config.max_rotations_per_session = 10
        config.switch_delay_seconds = 0
        config.verification_timeout = 10
        config.initial_rotation_delay_seconds = 1.0
        config.max_rotation_delay_seconds = 5.0
        config.rotation_delay_seconds = 1
        config.max_consecutive_vpn_failures = 5
        config.prefer_low_latency = prefer_low_latency
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0
        config.min_server_success_rate = 0.7
        config.swap_failure_threshold = 3

        return MullvadVPN(config)

    def test_server_selection_prefers_low_latency(self):
        """Test that server selection considers latency in scoring."""
        vpn = self._make_vpn(prefer_low_latency=True)

        # Simulate latency measurements
        vpn._server_latency['us'] = 30.0   # Low latency
        vpn._server_latency['de'] = 150.0   # High latency
        vpn._server_latency['jp'] = 200.0   # Very high latency

        # Score countries
        scored = vpn._score_countries_for_selection(['us', 'de', 'jp'])
        scores = {c: s for c, s in scored}

        # US should score highest due to lowest latency
        assert scores['us'] > scores['de']
        assert scores['us'] > scores['jp']

    def test_server_selection_considers_success_rate(self):
        """Test that server selection considers historical success rate."""
        vpn = self._make_vpn()

        from src.downloader.mullvad_vpn import ServerSuccessRecord

        # US has low latency but poor success rate
        vpn._server_latency['us'] = 30.0
        vpn._server_history['us'] = ServerSuccessRecord("us")
        vpn._server_history['us'].attempts = 10
        vpn._server_history['us'].successes = 1  # 10% success rate

        # DE has higher latency but perfect success rate
        vpn._server_latency['de'] = 150.0
        vpn._server_history['de'] = ServerSuccessRecord("de")
        vpn._server_history['de'].attempts = 10
        vpn._server_history['de'].successes = 10  # 100% success rate

        # Both countries should be in consideration
        vpn._used_countries = []
        scored = vpn._score_countries_for_selection(['us', 'de'])
        scores = {c: s for c, s in scored}

        # DE should score higher due to better success rate despite higher latency
        # (latency penalty for 150ms at 200ms threshold is -0.2, but success rate boost is +0.5)
        # US: +0.4 (low latency boost) - 0.3 (poor success rate) = +0.1
        # DE: -0.2 (high latency) + 0.5 (good success rate) = +0.3
        assert scores['de'] > scores['us']

    def test_server_selection_combines_all_factors(self):
        """Test that server selection combines latency, success rate, and geography."""
        vpn = self._make_vpn()

        from src.downloader.mullvad_vpn import ServerSuccessRecord

        # Last successful was in NA region
        vpn._last_successful_region = "na"

        # US: Same region (NA) + good latency + good success = best choice
        vpn._server_latency['us'] = 30.0
        vpn._server_history['us'] = ServerSuccessRecord("us")
        vpn._server_history['us'].attempts = 10
        vpn._server_history['us'].successes = 8  # 80%

        # DE: Different region (EU) + lower latency + good success
        vpn._server_latency['de'] = 100.0
        vpn._server_history['de'] = ServerSuccessRecord("de")
        vpn._server_history['de'].attempts = 10
        vpn._server_history['de'].successes = 8  # 80%

        # JP: Different region (APAC) + medium latency + good success
        vpn._server_latency['jp'] = 180.0
        vpn._server_history['jp'] = ServerSuccessRecord("jp")
        vpn._server_history['jp'].attempts = 10
        vpn._server_history['jp'].successes = 8  # 80%

        scored = vpn._score_countries_for_selection(['us', 'de', 'jp'])
        scores = {c: s for c, s in scored}

        # US should be highest due to same region + low latency + good success
        assert scores['us'] > scores['de']
        assert scores['us'] > scores['jp']

    def test_latency_disabled_prefers_geography_and_success(self):
        """Test that when latency is disabled, geography and success rate dominate."""
        vpn = self._make_vpn(prefer_low_latency=False)

        from src.downloader.mullvad_vpn import ServerSuccessRecord

        # Last successful was in EU region
        vpn._last_successful_region = "eu"

        # US: Different region + excellent success
        vpn._server_history['us'] = ServerSuccessRecord("us")
        vpn._server_history['us'].attempts = 10
        vpn._server_history['us'].successes = 10  # 100%

        # DE: Same region (EU) + good success
        vpn._server_history['de'] = ServerSuccessRecord("de")
        vpn._server_history['de'].attempts = 10
        vpn._server_history['de'].successes = 8  # 80%

        scored = vpn._score_countries_for_selection(['us', 'de'])
        scores = {c: s for c, s in scored}

        # DE should be highest due to same region despite lower success rate
        # DE: +0.3 (region) + 0.5 (success) = 0.8
        # US: 0.0 (region) + 0.5 (success) = 0.5
        assert scores['de'] > scores['us']

    def test_rotation_with_latency_tracking(self):
        """Test full rotation cycle with latency tracking."""
        # Create VPN with full config
        from src.downloader.mullvad_vpn import MullvadVPN

        config = MagicMock()
        config.enabled = True
        config.preferred_countries = ['us', 'gb', 'de', 'nl']
        config.rotation_strategy = 'random'
        config.max_rotations_per_session = 10
        config.max_switches_per_session = 10
        config.switch_delay_seconds = 0
        config.verification_timeout = 10
        config.skip_verification = True
        config.verify_connection = False
        config.initial_rotation_delay_seconds = 0.01
        config.max_rotation_delay_seconds = 1.0
        config.rotation_delay_seconds = 0
        config.max_consecutive_vpn_failures = 5
        config.prefer_low_latency = True
        config.max_latency_ms = 200
        config.latency_measurement_timeout = 3.0
        config.min_server_success_rate = 0.7
        config.swap_failure_threshold = 3

        vpn = MullvadVPN(config)

        # Simulate successful rotations with latency tracking
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')

            # First rotation to US
            vpn._used_countries = []
            vpn._server_latency['us'] = 30.0
            result1 = vpn.rotate_server(country='us')
            assert result1 is True

            # Check switch event recorded
            events = vpn.get_switch_events()
            assert len(events) == 1
            assert events[0]['to_country'] == 'us'

            # Second rotation to DE
            vpn._server_latency['de'] = 100.0
            result2 = vpn.rotate_server(country='de')
            assert result2 is True

            events = vpn.get_switch_events()
            assert len(events) == 2
            assert events[1]['to_country'] == 'de'
            assert events[1]['from_country'] == 'us'
