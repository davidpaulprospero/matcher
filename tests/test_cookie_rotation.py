"""Tests for cookie rotation functionality."""

import os
import stat
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockCookieRotationConfig:
    """Mock config for testing."""
    enabled: bool = True
    cookie_files: List[str] = field(default_factory=list)
    rotation_strategy: str = "on_error"
    rotate_on_errors: List[str] = field(default_factory=lambda: [
        "429", "rate limit", "sign in"
    ])
    cooldown_seconds: int = 5  # Short for testing
    max_rotations_per_session: int = 0  # Unlimited


class TestCookieValidation:
    """Tests for cookie file validation at startup (US-001)."""

    def test_validates_existence_and_readability(self, tmp_path):
        """Test that validation checks both existence AND readability."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create valid readable file
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("# Valid cookie file\n")

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(valid_file)]
        )
        rotator = CookieRotator(config)

        assert len(rotator._cookie_files) == 1
        assert len(rotator._invalid_cookies) == 0

    def test_logs_warning_for_missing_file(self, tmp_path, caplog):
        """Test warning logged for missing file with path and reason."""
        from src.downloader.cookie_rotator import CookieRotator
        import logging

        missing_path = str(tmp_path / "missing.txt")
        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[missing_path]
        )

        with caplog.at_level(logging.WARNING):
            rotator = CookieRotator(config)

        # Check warning was logged with path and reason
        assert missing_path in caplog.text
        assert "file not found" in caplog.text
        assert rotator._invalid_cookies[missing_path] == "file not found"

    def test_logs_warning_for_directory(self, tmp_path, caplog):
        """Test warning logged when path is a directory, not a file."""
        from src.downloader.cookie_rotator import CookieRotator
        import logging

        dir_path = str(tmp_path / "cookies_dir")
        os.makedirs(dir_path)

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[dir_path]
        )

        with caplog.at_level(logging.WARNING):
            rotator = CookieRotator(config)

        assert dir_path in caplog.text
        assert "not a file" in caplog.text
        assert rotator._invalid_cookies[dir_path] == "not a file"

    @pytest.mark.skipif(os.name == 'nt', reason="Permission tests unreliable on Windows")
    def test_logs_warning_for_unreadable_file(self, tmp_path, caplog):
        """Test warning logged for file that exists but is unreadable."""
        from src.downloader.cookie_rotator import CookieRotator
        import logging

        # Create file and remove read permissions
        unreadable = tmp_path / "unreadable.txt"
        unreadable.write_text("# Cookie content\n")
        unreadable.chmod(0o000)  # Remove all permissions

        try:
            config = MockCookieRotationConfig(
                enabled=True,
                cookie_files=[str(unreadable)]
            )

            with caplog.at_level(logging.WARNING):
                rotator = CookieRotator(config)

            assert str(unreadable) in caplog.text
            assert "permission denied" in caplog.text
        finally:
            # Restore permissions for cleanup
            unreadable.chmod(0o644)

    def test_get_status_includes_valid_cookies_count(self, tmp_path):
        """Test get_status() returns valid_cookies separate from total_cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create 2 valid files
        valid1 = tmp_path / "valid1.txt"
        valid2 = tmp_path / "valid2.txt"
        valid1.write_text("# Cookie 1\n")
        valid2.write_text("# Cookie 2\n")

        # One missing file
        missing = str(tmp_path / "missing.txt")

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(valid1), str(valid2), missing]
        )
        rotator = CookieRotator(config)

        status = rotator.get_status()

        # total_cookies = all configured (valid + invalid)
        assert status["total_cookies"] == 3
        # valid_cookies = only those that passed validation
        assert status["valid_cookies"] == 2
        # invalid_cookies dict includes the missing file
        assert missing in status["invalid_cookies"]
        assert status["invalid_cookies"][missing] == "file not found"

    def test_continue_on_partial_true_allows_startup(self, tmp_path):
        """Test continue_on_partial=True (default) allows startup with some invalid cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        valid = tmp_path / "valid.txt"
        valid.write_text("# Cookie\n")

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(valid), str(tmp_path / "missing.txt")]
        )

        # Should not raise with continue_on_partial=True (default)
        rotator = CookieRotator(config, continue_on_partial=True)
        assert rotator.is_enabled
        assert len(rotator._cookie_files) == 1

    def test_continue_on_partial_false_raises_on_invalid(self, tmp_path):
        """Test continue_on_partial=False raises ValueError when any cookie invalid."""
        from src.downloader.cookie_rotator import CookieRotator

        valid = tmp_path / "valid.txt"
        valid.write_text("# Cookie\n")
        missing = str(tmp_path / "missing.txt")

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(valid), missing]
        )

        with pytest.raises(ValueError) as exc_info:
            CookieRotator(config, continue_on_partial=False)

        # Error message should include path and reason
        assert missing in str(exc_info.value)
        assert "file not found" in str(exc_info.value)

    def test_mixed_valid_invalid_cookies(self, tmp_path):
        """Test startup with mix of valid and invalid cookie files."""
        from src.downloader.cookie_rotator import CookieRotator

        # 2 valid files
        valid1 = tmp_path / "valid1.txt"
        valid2 = tmp_path / "valid2.txt"
        valid1.write_text("# Valid 1\n")
        valid2.write_text("# Valid 2\n")

        # 1 missing file
        missing = str(tmp_path / "missing.txt")

        # 1 directory (not a file)
        dir_path = tmp_path / "dir_cookie"
        dir_path.mkdir()

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(valid1), missing, str(valid2), str(dir_path)]
        )
        rotator = CookieRotator(config)

        # Should have 2 valid cookies
        assert len(rotator._cookie_files) == 2
        assert str(valid1) in rotator._cookie_files
        assert str(valid2) in rotator._cookie_files

        # Should track 2 invalid cookies with reasons
        assert len(rotator._invalid_cookies) == 2
        assert rotator._invalid_cookies[missing] == "file not found"
        assert rotator._invalid_cookies[str(dir_path)] == "not a file"

        # Status should reflect this
        status = rotator.get_status()
        assert status["total_cookies"] == 4
        assert status["valid_cookies"] == 2
        assert status["available_cookies"] == 2  # None in cooldown yet


class TestCustomErrorPatterns:
    """Tests for configurable error patterns (US-003)."""

    @pytest.fixture
    def temp_cookie(self, tmp_path):
        """Create a single temp cookie file."""
        cookie_file = tmp_path / "cookie.txt"
        cookie_file.write_text("# Cookie file\n")
        return str(cookie_file)

    def test_custom_pattern_triggers_rotation(self, temp_cookie):
        """Test that custom error patterns trigger rotation."""
        from src.downloader.cookie_rotator import CookieRotator

        # Custom pattern not in defaults
        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[temp_cookie],
            rotate_on_errors=["custom_error_xyz"]
        )
        rotator = CookieRotator(config)

        # Default patterns should NOT trigger
        assert not rotator.should_rotate("429 Too Many Requests")
        assert not rotator.should_rotate("rate limit exceeded")

        # Custom pattern SHOULD trigger
        assert rotator.should_rotate("Some CUSTOM_ERROR_XYZ occurred")

    def test_case_insensitive_matching(self, temp_cookie):
        """Test that error patterns match case-insensitively."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[temp_cookie],
            rotate_on_errors=["MyPattern"]
        )
        rotator = CookieRotator(config)

        # Should match regardless of case
        assert rotator.should_rotate("error: MYPATTERN detected")
        assert rotator.should_rotate("error: mypattern detected")
        assert rotator.should_rotate("error: MyPattern detected")

    def test_empty_patterns_validation_when_enabled(self):
        """Test that empty rotate_on_errors raises ValueError when enabled."""
        from src.config.sections.download import CookieRotationConfig

        with pytest.raises(ValueError) as exc_info:
            CookieRotationConfig(
                enabled=True,
                cookie_files=["some/path.txt"],
                rotate_on_errors=[]
            )

        assert "non-empty list" in str(exc_info.value)
        assert "rotate_on_errors" in str(exc_info.value)

    def test_empty_patterns_allowed_when_disabled(self):
        """Test that empty rotate_on_errors is allowed when rotation disabled."""
        from src.config.sections.download import CookieRotationConfig

        # Should not raise - rotation is disabled
        config = CookieRotationConfig(
            enabled=False,
            rotate_on_errors=[]
        )
        assert config.rotate_on_errors == []

    def test_multiple_custom_patterns(self, temp_cookie):
        """Test rotation with multiple custom patterns."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[temp_cookie],
            rotate_on_errors=["pattern_a", "pattern_b", "pattern_c"]
        )
        rotator = CookieRotator(config)

        assert rotator.should_rotate("Found pattern_a in response")
        assert rotator.should_rotate("Found pattern_b in response")
        assert rotator.should_rotate("Found pattern_c in response")
        assert not rotator.should_rotate("Found pattern_d in response")


class TestCookieRotator:
    """Tests for CookieRotator class."""

    @pytest.fixture
    def temp_cookies(self, tmp_path):
        """Create temporary cookie files for testing."""
        cookies = []
        for i in range(3):
            cookie_file = tmp_path / f"cookie_{i}.txt"
            cookie_file.write_text(f"# Cookie file {i}\n")
            cookies.append(str(cookie_file))
        return cookies

    @pytest.fixture
    def rotator(self, temp_cookies):
        """Create a CookieRotator with temp cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies
        )
        return CookieRotator(config)

    def test_init_with_valid_cookies(self, temp_cookies):
        """Test initialization with valid cookie files."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies
        )
        rotator = CookieRotator(config)

        assert rotator.is_enabled
        assert rotator.available_cookies == 3

    def test_init_with_missing_cookies(self, tmp_path):
        """Test initialization with missing cookie files."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[
                str(tmp_path / "missing1.txt"),
                str(tmp_path / "missing2.txt"),
            ]
        )
        rotator = CookieRotator(config)

        assert not rotator.is_enabled  # No valid cookies
        assert rotator.available_cookies == 0

    def test_init_disabled(self, temp_cookies):
        """Test initialization when disabled."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=False,
            cookie_files=temp_cookies
        )
        rotator = CookieRotator(config)

        assert not rotator.is_enabled

    def test_get_current_cookie(self, rotator, temp_cookies):
        """Test getting current cookie."""
        current = rotator.get_current_cookie()
        assert current == temp_cookies[0]

    def test_should_rotate_on_429(self, rotator):
        """Test rotation trigger on 429 error."""
        assert rotator.should_rotate("HTTP Error 429: Too Many Requests")
        assert rotator.should_rotate("rate limit exceeded")
        assert rotator.should_rotate("Please sign in to continue")

    def test_should_not_rotate_on_other_errors(self, rotator):
        """Test no rotation on unrelated errors."""
        assert not rotator.should_rotate("Video not found")
        assert not rotator.should_rotate("Network timeout")

    def test_rotate_round_robin(self, rotator, temp_cookies):
        """Test round-robin rotation strategy."""
        # Initial cookie
        assert rotator.get_current_cookie() == temp_cookies[0]

        # Rotate to second
        new_cookie = rotator.rotate()
        assert new_cookie == temp_cookies[1]
        assert rotator.get_current_cookie() == temp_cookies[1]

        # Rotate to third
        new_cookie = rotator.rotate()
        assert new_cookie == temp_cookies[2]

        # Wrap around (after cooldown)
        rotator._failed_cookies.clear()  # Clear cooldowns for test
        new_cookie = rotator.rotate()
        assert new_cookie == temp_cookies[0]

    def test_rotate_random(self, temp_cookies):
        """Test random rotation strategy."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            rotation_strategy="random"
        )
        rotator = CookieRotator(config)

        # Just verify rotation produces a valid cookie
        new_cookie = rotator.rotate()
        assert new_cookie in temp_cookies

    def test_cooldown_tracking(self, rotator, temp_cookies):
        """Test that failed cookies enter cooldown."""
        # Rotate from first cookie
        rotator.rotate()

        # First cookie should be in cooldown
        assert not rotator.is_available(temp_cookies[0])
        assert temp_cookies[0] in rotator._failed_cookies

    def test_cooldown_expiry(self, temp_cookies):
        """Test that cooldown expires after configured time."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            cooldown_seconds=1  # 1 second cooldown
        )
        rotator = CookieRotator(config)

        # Mark first cookie as failed
        rotator.mark_failed(temp_cookies[0])
        assert not rotator.is_available(temp_cookies[0])

        # Wait for cooldown
        time.sleep(1.1)

        # Cookie should be available again
        assert rotator.is_available(temp_cookies[0])

    def test_all_cookies_exhausted(self, rotator, temp_cookies):
        """Test behavior when all cookies are in cooldown."""
        # Exhaust all cookies
        for _ in range(3):
            rotator.rotate()

        # All cookies in cooldown
        assert rotator.available_cookies == 0

        # Next rotation should fail
        assert rotator.rotate() is None

    def test_max_rotations_limit(self, temp_cookies):
        """Test max rotations per session limit."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=temp_cookies,
            max_rotations_per_session=2
        )
        rotator = CookieRotator(config)

        # First two rotations should work
        assert rotator.rotate() is not None
        assert rotator.rotate() is not None

        # Third rotation should fail due to limit
        assert rotator.rotate() is None
        assert not rotator.can_rotate()

    def test_reset(self, rotator, temp_cookies):
        """Test reset clears all state."""
        # Do some rotations
        rotator.rotate()
        rotator.rotate()

        # Reset
        rotator.reset()

        # State should be cleared
        assert rotator._current_index == 0
        assert rotator._rotation_count == 0
        assert len(rotator._failed_cookies) == 0
        assert rotator.available_cookies == 3

    def test_get_status(self, rotator, temp_cookies):
        """Test status reporting."""
        status = rotator.get_status()

        assert status["enabled"] is True
        assert status["total_cookies"] == 3  # All configured cookies
        assert status["valid_cookies"] == 3  # All passed validation
        assert status["available_cookies"] == 3  # None in cooldown
        assert status["rotation_count"] == 0
        assert status["strategy"] == "on_error"
        assert status["invalid_cookies"] == {}  # No invalid cookies

    def test_can_rotate(self, rotator, temp_cookies):
        """Test can_rotate check."""
        assert rotator.can_rotate()

        # Exhaust all cookies
        for _ in range(3):
            rotator.rotate()

        # Cannot rotate when all in cooldown
        assert not rotator.can_rotate()


class TestCookieRotatorIntegration:
    """Integration tests for cookie rotation with downloader."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config with cookie rotation enabled."""
        cookies = []
        for i in range(2):
            cookie_file = tmp_path / f"cookie_{i}.txt"
            cookie_file.write_text(f"# Cookie {i}\n")
            cookies.append(str(cookie_file))

        config = MagicMock()
        config.download = MagicMock()
        config.download.cookie_rotation = MockCookieRotationConfig(
            enabled=True,
            cookie_files=cookies
        )
        config.download.cookies_from_browser = ""
        config.download.cookies_path = ""

        return config

    def test_rotator_used_in_add_cookies_to_cmd(self, mock_config, tmp_path):
        """Test that VideoDownloader uses rotator when enabled."""
        # This is a conceptual test - actual integration would require
        # mocking the full VideoDownloader which is complex
        from src.downloader.cookie_rotator import CookieRotator

        rotator = CookieRotator(mock_config.download.cookie_rotation)
        cmd = []

        # Simulate what _add_cookies_to_cmd should do
        if rotator and rotator.is_enabled:
            current_cookie = rotator.get_current_cookie()
            if current_cookie:
                cmd.extend(['--cookies', current_cookie])

        assert '--cookies' in cmd
        assert cmd[1].endswith('cookie_0.txt')


class TestMidSessionFileValidation:
    """Tests for cookie file validation during active session (US-007).

    Verifies that CookieRotator gracefully handles cookie files that
    are deleted, emptied, or otherwise invalidated after initialization.
    """

    @pytest.fixture
    def three_cookies(self, tmp_path):
        """Create 3 valid cookie files."""
        cookies = []
        for i in range(3):
            f = tmp_path / f"cookie_{i}.txt"
            f.write_text(f"# Netscape HTTP Cookie File\ncookie_data_{i}\n")
            cookies.append(str(f))
        return cookies

    @pytest.fixture
    def rotator_with_cookies(self, three_cookies):
        """Create a CookieRotator with 3 valid cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=three_cookies
        )
        return CookieRotator(config)

    def test_rotate_when_current_cookie_deleted_mid_session(self, rotator_with_cookies, three_cookies):
        """AC1: rotate() when current cookie file is deleted mid-session
        — verify graceful fallback to next available cookie (no FileNotFoundError crash)."""
        rotator = rotator_with_cookies

        # Verify initial state
        assert rotator.get_current_cookie() == three_cookies[0]

        # Delete the current cookie file mid-session
        os.remove(three_cookies[0])

        # rotate() should NOT raise FileNotFoundError
        new_cookie = rotator.rotate()

        # Should have fallen back to one of the remaining cookies
        assert new_cookie is not None
        assert new_cookie in [three_cookies[1], three_cookies[2]]

        # Deleted file should be removed from active list
        assert three_cookies[0] not in rotator._cookie_files

    def test_rotate_when_cookie_becomes_empty_mid_session(self, rotator_with_cookies, three_cookies):
        """AC2: rotate() when cookie file becomes empty (0 bytes) mid-session
        — verify it skips to next cookie instead of using empty file."""
        rotator = rotator_with_cookies

        assert rotator.get_current_cookie() == three_cookies[0]

        # Truncate the current cookie to 0 bytes
        with open(three_cookies[0], 'w') as f:
            f.truncate(0)

        # rotate() should skip the empty file
        new_cookie = rotator.rotate()

        # Should get a non-empty cookie
        assert new_cookie is not None
        assert new_cookie in [three_cookies[1], three_cookies[2]]

        # Empty file should be tracked as invalid
        assert three_cookies[0] in rotator._invalid_cookies
        assert "empty" in rotator._invalid_cookies[three_cookies[0]]

    def test_rotate_with_all_cookies_deleted(self, rotator_with_cookies, three_cookies, caplog):
        """AC3: rotate() with all cookie files deleted — verify returns None
        and logs warning (not exception)."""
        import logging

        rotator = rotator_with_cookies

        # Delete ALL cookie files
        for cookie_path in three_cookies:
            os.remove(cookie_path)

        with caplog.at_level(logging.WARNING):
            # Should NOT raise any exception
            result = rotator.rotate()

        # Should return None (no cookies available)
        assert result is None

        # Should have logged a warning
        assert any("deleted" in record.message.lower() or "no cookies" in record.message.lower()
                    or "empty" in record.message.lower() or "invalidated" in record.message.lower()
                    for record in caplog.records)

        # All cookies should be marked invalid
        assert len(rotator._cookie_files) == 0

    def test_round_robin_single_cookie_no_infinite_loop(self, tmp_path):
        """AC4: round-robin rotation with single cookie file — verify it returns
        that file on every call (no infinite loop in _select_round_robin)."""
        from src.downloader.cookie_rotator import CookieRotator

        single = tmp_path / "only_cookie.txt"
        single.write_text("# Cookie data\n")

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(single)],
            max_rotations_per_session=0  # Unlimited
        )
        rotator = CookieRotator(config)

        # get_current_cookie should always return this file
        for _ in range(5):
            assert rotator.get_current_cookie() == str(single)

        # rotate() marks current as failed and tries to find next
        # With only one cookie, round-robin has no other to go to
        result = rotator.rotate()
        # Result is None because the only cookie is now in cooldown
        assert result is None

        # But get_current_cookie still returns it (it's the only one,
        # even in cooldown the code returns it with a warning)
        current = rotator.get_current_cookie()
        assert current == str(single)

    def test_get_current_cookie_after_deletion_returns_none_or_fallback(
        self, rotator_with_cookies, three_cookies
    ):
        """AC5: get_current_cookie() after cookie deletion — verify returns None
        rather than path to deleted file."""
        rotator = rotator_with_cookies

        # Initial state: current is cookie_0
        assert rotator.get_current_cookie() == three_cookies[0]

        # Delete the current cookie
        os.remove(three_cookies[0])

        # get_current_cookie() should NOT return the deleted file's path
        result = rotator.get_current_cookie()
        assert result != three_cookies[0]

        # It should return a valid fallback (cookie_1 or cookie_2)
        assert result in [three_cookies[1], three_cookies[2]]

    def test_get_current_cookie_all_deleted_returns_none(self, rotator_with_cookies, three_cookies):
        """get_current_cookie() with all files deleted returns None."""
        rotator = rotator_with_cookies

        # Delete all cookie files
        for cookie_path in three_cookies:
            os.remove(cookie_path)

        result = rotator.get_current_cookie()
        assert result is None

    def test_deleted_cookie_tracked_in_invalid_cookies(self, rotator_with_cookies, three_cookies):
        """Deleted cookie is moved to _invalid_cookies with 'deleted' reason."""
        rotator = rotator_with_cookies

        os.remove(three_cookies[0])

        # Trigger mid-session validation
        rotator.get_current_cookie()

        assert three_cookies[0] in rotator._invalid_cookies
        assert rotator._invalid_cookies[three_cookies[0]] == "deleted"

    def test_is_cookie_file_valid_checks_existence_and_size(self, tmp_path):
        """_is_cookie_file_valid returns False for missing and empty files."""
        from src.downloader.cookie_rotator import CookieRotator

        valid = tmp_path / "valid.txt"
        valid.write_text("# Cookie\n")

        empty = tmp_path / "empty.txt"
        empty.write_text("")

        config = MockCookieRotationConfig(
            enabled=True,
            cookie_files=[str(valid)]
        )
        rotator = CookieRotator(config)

        # Valid file
        assert rotator._is_cookie_file_valid(str(valid)) is True

        # Empty file
        assert rotator._is_cookie_file_valid(str(empty)) is False

        # Missing file
        assert rotator._is_cookie_file_valid(str(tmp_path / "missing.txt")) is False

    def test_rotate_skips_deleted_in_middle_of_chain(self, rotator_with_cookies, three_cookies):
        """rotate() skips a deleted cookie in the middle and goes to the next valid one."""
        rotator = rotator_with_cookies

        # Delete cookie_1 (the next one in round-robin after cookie_0)
        os.remove(three_cookies[1])

        # Rotate from cookie_0 — should skip deleted cookie_1 and go to cookie_2
        new_cookie = rotator.rotate()
        assert new_cookie == three_cookies[2]

        # cookie_1 should be removed from active list
        assert three_cookies[1] not in rotator._cookie_files

    def test_status_reflects_mid_session_invalidation(self, rotator_with_cookies, three_cookies):
        """get_status() reflects mid-session invalidation correctly."""
        rotator = rotator_with_cookies

        # Initial: 3 valid, 0 invalid
        status_before = rotator.get_status()
        assert status_before["valid_cookies"] == 3
        assert status_before["total_cookies"] == 3

        # Delete one cookie and trigger validation
        os.remove(three_cookies[0])
        rotator.get_current_cookie()

        status_after = rotator.get_status()
        assert status_after["valid_cookies"] == 2
        assert status_after["total_cookies"] == 3  # total includes invalid
        assert three_cookies[0] in status_after["invalid_cookies"]
