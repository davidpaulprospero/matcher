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
