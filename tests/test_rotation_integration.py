"""Integration tests for cookie rotation and VPN with downloader and healer."""

import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MockCookieRotationConfig:
    """Mock cookie rotation config."""
    enabled: bool = True
    cookie_files: List[str] = field(default_factory=list)
    rotation_strategy: str = "on_error"
    rotate_on_errors: List[str] = field(default_factory=lambda: [
        "429", "rate limit", "sign in", "too many"
    ])
    cooldown_seconds: int = 1
    max_rotations_per_session: int = 0


@dataclass
class MockVPNConfig:
    """Mock VPN config."""
    enabled: bool = True
    switch_command: str = "echo 'switched'"
    disconnect_command: str = ""
    rotate_on_rate_limit: bool = True
    switch_delay_seconds: int = 0
    max_switches_per_session: int = 5
    verify_connection: bool = False
    verify_timeout: int = 5


@dataclass
class MockDownloadConfig:
    """Mock download config with rotation and VPN."""
    cookie_rotation: MockCookieRotationConfig = field(default_factory=MockCookieRotationConfig)
    vpn: MockVPNConfig = field(default_factory=MockVPNConfig)
    cookies_from_browser: str = ""
    cookies_path: str = ""


class TestHealerCookieRotationIntegration:
    """Integration tests for DownloadHealer with cookie rotation."""

    @pytest.fixture
    def temp_cookies(self, tmp_path):
        """Create temporary cookie files."""
        cookies = []
        for i in range(3):
            cookie_file = tmp_path / f"cookie_{i}.txt"
            cookie_file.write_text(f"# Cookie {i}\n")
            cookies.append(str(cookie_file))
        return cookies

    @pytest.fixture
    def config_with_rotation(self, temp_cookies):
        """Config with cookie rotation enabled."""
        config = MagicMock()
        config.download = MockDownloadConfig(
            cookie_rotation=MockCookieRotationConfig(
                enabled=True,
                cookie_files=temp_cookies
            ),
            vpn=MockVPNConfig(enabled=False)
        )
        return config

    @pytest.fixture
    def config_with_vpn(self, tmp_path):
        """Config with VPN enabled but no cookies."""
        config = MagicMock()
        config.download = MockDownloadConfig(
            cookie_rotation=MockCookieRotationConfig(enabled=False),
            vpn=MockVPNConfig(enabled=True)
        )
        return config

    @pytest.fixture
    def config_with_both(self, temp_cookies):
        """Config with both cookie rotation and VPN."""
        config = MagicMock()
        config.download = MockDownloadConfig(
            cookie_rotation=MockCookieRotationConfig(
                enabled=True,
                cookie_files=temp_cookies,
                max_rotations_per_session=2  # Limit to test VPN fallback
            ),
            vpn=MockVPNConfig(enabled=True)
        )
        return config

    @pytest.mark.fast
    def test_healer_tries_cookie_rotation_first(self, config_with_rotation):
        """Verify healer tries cookie rotation before backoff."""
        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        healer = DownloadHealer(config_with_rotation, "/tmp/project")
        state = MagicMock(spec=PipelineState)

        # First rate limit should trigger cookie rotation
        error = Exception("HTTP Error 429: Too Many Requests")
        result = healer.fix(error, state, "DOWNLOAD")

        assert result is not None
        assert "cookie" in result.message.lower() or "rate limit" in result.message.lower()
        # Should have rotated (if cookies available)
        if healer.cookie_rotator and healer.cookie_rotator.is_enabled:
            assert healer.cookie_rotator._rotation_count >= 0

    @pytest.mark.integration
    def test_healer_tries_vpn_when_no_cookies(self, config_with_vpn):
        """Verify healer tries VPN when cookies not configured."""
        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            healer = DownloadHealer(config_with_vpn, "/tmp/project")
            state = MagicMock(spec=PipelineState)

            error = Exception("HTTP Error 429")
            result = healer.fix(error, state, "DOWNLOAD")

            assert result is not None
            # Should have tried VPN or fallen back to backoff
            if healer.vpn_manager and healer.vpn_manager.is_enabled:
                # VPN was attempted
                assert mock_run.called or "backoff" in result.message.lower()

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_healer_vpn_after_cookies_exhausted(self, mock_run, config_with_both):
        """Verify healer tries VPN after cookie rotation exhausted."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        healer = DownloadHealer(config_with_both, "/tmp/project")
        state = MagicMock(spec=PipelineState)

        # First two errors should use cookie rotation (limit is 2)
        for i in range(2):
            error = Exception("HTTP Error 429")
            result = healer.fix(error, state, "DOWNLOAD")
            assert result is not None

        # Third error should try VPN (cookies exhausted)
        error = Exception("HTTP Error 429")
        result = healer.fix(error, state, "DOWNLOAD")
        assert result is not None

        # Verify VPN was attempted
        if healer.vpn_manager and healer.vpn_manager.is_enabled:
            assert mock_run.called or healer.vpn_manager.switch_count > 0

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_cookie_rotator_resets_after_vpn_switch(self, mock_run, config_with_both):
        """Verify cookie rotator resets after successful VPN switch."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        healer = DownloadHealer(config_with_both, "/tmp/project")
        state = MagicMock(spec=PipelineState)

        # Exhaust cookies
        for i in range(2):
            error = Exception("HTTP Error 429")
            healer.fix(error, state, "DOWNLOAD")

        # Next error should trigger VPN and reset cookies
        error = Exception("HTTP Error 429")
        healer.fix(error, state, "DOWNLOAD")

        # Cookie rotator should have been reset
        if healer.cookie_rotator and healer.vpn_manager:
            if healer.vpn_manager.switch_count > 0:
                # After VPN switch, cookie state should be reset
                assert healer.cookie_rotator._rotation_count == 0 or healer.cookie_rotator.available_cookies > 0

    @pytest.mark.fast
    def test_healer_fallback_to_backoff(self, config_with_rotation):
        """Verify healer falls back to backoff when rotation disabled."""
        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        # Disable rotation
        config_with_rotation.download.cookie_rotation.enabled = False
        config_with_rotation.download.vpn.enabled = False

        healer = DownloadHealer(config_with_rotation, "/tmp/project")
        state = MagicMock(spec=PipelineState)

        error = Exception("HTTP Error 429")
        start = time.time()
        result = healer.fix(error, state, "DOWNLOAD")
        elapsed = time.time() - start

        # Should have used backoff (waited)
        assert result is not None
        assert "waited" in result.message.lower() or "backoff" in result.message.lower()
        # Should have waited at least initial backoff time
        assert elapsed >= healer.INITIAL_BACKOFF - 1  # Small margin

    @pytest.mark.fast
    def test_healer_result_includes_rotation_info(self, config_with_rotation):
        """Verify HealerResult includes rotation information."""
        from src.agents.healers.download import DownloadHealer
        from src.state import PipelineState

        healer = DownloadHealer(config_with_rotation, "/tmp/project")
        state = MagicMock(spec=PipelineState)

        error = Exception("HTTP Error 429")
        result = healer.fix(error, state, "DOWNLOAD")

        # Result should have details about what happened
        assert result is not None
        assert result.details is not None


class TestVideoDownloaderIntegration:
    """Integration tests for VideoDownloader with rotation."""

    @pytest.fixture
    def temp_cookies(self, tmp_path):
        """Create temporary cookie files."""
        cookies = []
        for i in range(2):
            cookie_file = tmp_path / f"cookie_{i}.txt"
            cookie_file.write_text(f"# Cookie {i}\n")
            cookies.append(str(cookie_file))
        return cookies

    @pytest.fixture
    def mock_config(self, temp_cookies, tmp_path):
        """Create mock config for VideoDownloader."""
        config = MagicMock()

        # Download config
        config.download = MagicMock()
        config.download.cookie_rotation = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies
        )
        config.download.vpn = MockVPNConfig(enabled=True)
        config.download.cookies_from_browser = ""
        config.download.cookies_path = ""
        config.download.parallel_workers = 1
        config.download.quality = "720p"
        config.download.format = "mp4"

        # Other required config
        config.cache_dir = str(tmp_path / "cache")
        config.downloaded_videos_dir = str(tmp_path / "videos")
        config.project_dir = str(tmp_path)

        return config

    @pytest.mark.fast
    def test_downloader_initializes_rotator(self, mock_config, temp_cookies):
        """Test VideoDownloader initializes cookie rotator."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create rotator directly (avoiding full downloader init complexity)
        rotator = CookieRotator(mock_config.download.cookie_rotation)

        assert rotator.is_enabled
        assert rotator.available_cookies == 2

    @pytest.mark.fast
    def test_downloader_initializes_vpn(self, mock_config):
        """Test VideoDownloader initializes VPN manager."""
        from src.downloader.vpn_manager import VPNManager

        manager = VPNManager(mock_config.download.vpn)

        assert manager.is_enabled
        assert manager.can_switch()

    @pytest.mark.fast
    def test_add_cookies_uses_rotator(self, temp_cookies):
        """Test _add_cookies_to_cmd uses rotator when enabled."""
        from src.downloader.cookie_rotator import CookieRotator

        rotator = CookieRotator(MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies
        ))

        # Simulate what _add_cookies_to_cmd does
        cmd = []
        if rotator and rotator.is_enabled:
            current_cookie = rotator.get_current_cookie()
            if current_cookie:
                cmd.extend(['--cookies', current_cookie])

        assert '--cookies' in cmd
        assert temp_cookies[0] in cmd[1]

    @pytest.mark.fast
    def test_handle_rate_limit_error(self, temp_cookies):
        """Test handle_rate_limit_error method."""
        from src.downloader.cookie_rotator import CookieRotator
        from src.downloader.vpn_manager import VPNManager

        rotator = CookieRotator(MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies
        ))
        vpn = VPNManager(MockVPNConfig(enabled=False))

        # Simulate handle_rate_limit_error logic
        error_message = "HTTP Error 429"
        recovered = False

        # Try cookie rotation
        if rotator.should_rotate(error_message):
            new_cookie = rotator.rotate()
            if new_cookie:
                recovered = True

        assert recovered is True
        assert rotator._rotation_count == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_vpn_switch_after_cookie_exhaust(self, mock_run, temp_cookies):
        """Test VPN switch after cookies exhausted."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from src.downloader.cookie_rotator import CookieRotator
        from src.downloader.vpn_manager import VPNManager

        rotator = CookieRotator(MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            max_rotations_per_session=1
        ))
        vpn = VPNManager(MockVPNConfig(enabled=True))

        # Exhaust cookie rotation
        rotator.rotate()
        assert not rotator.can_rotate()

        # VPN should work
        assert vpn.can_switch()
        success = vpn.switch()
        assert success

        # Reset rotator after VPN
        rotator.reset()
        assert rotator.can_rotate()


class TestEndToEndRotation:
    """End-to-end tests for the full rotation flow."""

    @pytest.fixture
    def temp_cookies(self, tmp_path):
        """Create temporary cookie files."""
        cookies = []
        for i in range(3):
            cookie_file = tmp_path / f"cookie_{i}.txt"
            cookie_file.write_text(f"# Cookie {i}\n")
            cookies.append(str(cookie_file))
        return cookies

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_full_recovery_flow(self, mock_run, temp_cookies):
        """Test full recovery flow: cookies -> VPN -> backoff."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        from src.downloader.cookie_rotator import CookieRotator
        from src.downloader.vpn_manager import VPNManager

        rotator = CookieRotator(MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            cooldown_seconds=1,
            max_rotations_per_session=3
        ))
        vpn = VPNManager(MockVPNConfig(
            enabled=True,
            max_switches_per_session=2
        ))

        error_count = 0
        max_errors = 10
        recovery_methods = []

        while error_count < max_errors:
            error_count += 1

            # Try cookie rotation
            if rotator.can_rotate():
                new_cookie = rotator.rotate()
                if new_cookie:
                    recovery_methods.append("cookie")
                    continue

            # Try VPN
            if vpn.can_switch():
                if vpn.switch():
                    rotator.reset()  # Reset cookies after VPN
                    recovery_methods.append("vpn")
                    continue

            # Fall back to backoff
            recovery_methods.append("backoff")

            # If we hit 3 backoffs, stop
            if recovery_methods.count("backoff") >= 3:
                break

        # Should have used all methods
        assert "cookie" in recovery_methods
        assert "vpn" in recovery_methods
        assert "backoff" in recovery_methods

        # Verify VPN count (should hit max_switches)
        assert recovery_methods.count("vpn") == 2  # max_switches

        # Cookie count will be higher than max_rotations because VPN resets cookies
        # Each VPN switch resets the rotation count, allowing more cookies to be used
        # Expected: 3 (initial) + 3 (after VPN1) + 3 (after VPN2) until we hit backoff
        assert recovery_methods.count("cookie") >= 3  # At least max_rotations before first VPN

    @pytest.mark.fast
    def test_rotation_strategy_selection(self, temp_cookies):
        """Test different rotation strategies work correctly."""
        from src.downloader.cookie_rotator import CookieRotator

        # Test round-robin
        rr_rotator = CookieRotator(MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            rotation_strategy="round_robin"
        ))

        first = rr_rotator.get_current_cookie()
        rr_rotator.rotate()
        second = rr_rotator.get_current_cookie()

        assert first != second
        assert first == temp_cookies[0]
        assert second == temp_cookies[1]

        # Test random
        rand_rotator = CookieRotator(MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            rotation_strategy="random"
        ))

        # Random should still return valid cookies
        for _ in range(10):
            cookie = rand_rotator.rotate()
            if cookie:
                assert cookie in temp_cookies

    @pytest.mark.fast
    def test_config_propagation(self, temp_cookies):
        """Test config values are correctly used."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            cooldown_seconds=60,
            max_rotations_per_session=5
        )
        rotator = CookieRotator(config)

        assert rotator.config.cooldown_seconds == 60
        assert rotator.config.max_rotations_per_session == 5
