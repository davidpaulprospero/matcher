"""Tests for cookie cooldown dict cleanup on cookie removal.

Tests US-58-007 acceptance criteria:
1. _remove_invalid_cookie removes entry from _failed_cookies dict
2. _remove_invalid_cookie removes entry from _cooldown_until dict (tracked via _failed_cookies)
3. After mark_failed then remove, _failed_cookies no longer contains that path
4. Repeated fail+remove cycles do not grow _failed_cookies unboundedly
5. Existing cookie rotator tests continue to pass
"""

import pytest
import time
from pathlib import Path
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
    cooldown_seconds: int = 5
    max_rotations_per_session: int = 0


class TestRemoveInvalidCookieCleansUpFailedDict:
    """Test _remove_invalid_cookie cleans up _failed_cookies entries."""

    @pytest.mark.fast
    def test_remove_clears_failed_cookies_entry(self, tmp_path):
        """After mark_failed then _remove_invalid_cookie, _failed_cookies no longer contains that path."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookie1.txt"
        cookie2 = tmp_path / "cookie2.txt"
        cookie1.write_text("# cookie 1\n")
        cookie2.write_text("# cookie 2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)]
        )
        rotator = CookieRotator(config)

        # Mark cookie1 as failed (enters cooldown)
        rotator.mark_failed(str(cookie1))
        assert str(cookie1) in rotator._failed_cookies

        # Remove cookie1 as invalid
        rotator._remove_invalid_cookie(str(cookie1), "test removal")

        # _failed_cookies should no longer contain cookie1
        assert str(cookie1) not in rotator._failed_cookies

    @pytest.mark.fast
    def test_remove_clears_cooldown_tracking(self, tmp_path):
        """Removing a cookie clears its cooldown timestamp from _failed_cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookie1.txt"
        cookie2 = tmp_path / "cookie2.txt"
        cookie1.write_text("# cookie 1\n")
        cookie2.write_text("# cookie 2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)]
        )
        rotator = CookieRotator(config)

        # Mark cookie1 failed with a specific timestamp
        rotator._failed_cookies[str(cookie1)] = time.time()
        assert str(cookie1) in rotator._failed_cookies

        # Remove it
        rotator._remove_invalid_cookie(str(cookie1), "deleted")

        # Cooldown timestamp should be gone
        assert str(cookie1) not in rotator._failed_cookies

    @pytest.mark.fast
    def test_remove_without_failed_entry_is_safe(self, tmp_path):
        """Removing a cookie that was never marked failed doesn't raise."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookie1.txt"
        cookie2 = tmp_path / "cookie2.txt"
        cookie1.write_text("# cookie 1\n")
        cookie2.write_text("# cookie 2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)]
        )
        rotator = CookieRotator(config)

        # cookie1 was never marked failed
        assert str(cookie1) not in rotator._failed_cookies

        # Remove should not raise
        rotator._remove_invalid_cookie(str(cookie1), "test")

        # Still not in _failed_cookies
        assert str(cookie1) not in rotator._failed_cookies

    @pytest.mark.fast
    def test_repeated_fail_remove_cycles_no_growth(self, tmp_path):
        """Repeated fail+remove cycles do not grow _failed_cookies unboundedly."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create a pool of cookie files
        cookies = []
        for i in range(5):
            cf = tmp_path / f"cookie_{i}.txt"
            cf.write_text(f"# cookie {i}\n")
            cookies.append(str(cf))

        config = MockCookieRotationConfig(cookie_files=cookies.copy())
        rotator = CookieRotator(config)

        # Simulate 100 cycles of fail + remove
        for cycle in range(100):
            if not rotator._cookie_files:
                break

            target = rotator._cookie_files[0]
            # Mark failed
            rotator.mark_failed(target)
            # Remove
            rotator._remove_invalid_cookie(target, f"cycle {cycle}")

            # After removal, _failed_cookies should NOT contain the removed cookie
            assert target not in rotator._failed_cookies
            # _failed_cookies should never be larger than the active cookie list
            assert len(rotator._failed_cookies) <= len(rotator._cookie_files)

        # After all cookies removed, _failed_cookies should be empty
        assert len(rotator._failed_cookies) == 0

    @pytest.mark.fast
    def test_other_cookies_cooldowns_preserved(self, tmp_path):
        """Removing one cookie doesn't affect other cookies' cooldown entries."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookie1.txt"
        cookie2 = tmp_path / "cookie2.txt"
        cookie3 = tmp_path / "cookie3.txt"
        cookie1.write_text("# cookie 1\n")
        cookie2.write_text("# cookie 2\n")
        cookie3.write_text("# cookie 3\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2), str(cookie3)]
        )
        rotator = CookieRotator(config)

        # Mark cookies 1 and 2 as failed
        rotator.mark_failed(str(cookie1))
        rotator.mark_failed(str(cookie2))

        # Remove only cookie1
        rotator._remove_invalid_cookie(str(cookie1), "test")

        # cookie1 should be gone from _failed_cookies
        assert str(cookie1) not in rotator._failed_cookies
        # cookie2's cooldown should be preserved
        assert str(cookie2) in rotator._failed_cookies

    @pytest.mark.fast
    def test_mid_session_invalidation_cleans_cooldown(self, tmp_path):
        """Integration: mid-session file deletion cleans up cooldown via get_current_cookie."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookie1.txt"
        cookie2 = tmp_path / "cookie2.txt"
        cookie1.write_text("# cookie 1\n")
        cookie2.write_text("# cookie 2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)]
        )
        rotator = CookieRotator(config)

        # Mark cookie1 as failed (cooldown)
        rotator.mark_failed(str(cookie1))
        assert str(cookie1) in rotator._failed_cookies

        # Simulate file deletion
        cookie1.unlink()

        # Access current cookie — triggers mid-session validation
        # which calls _remove_invalid_cookie for deleted files
        rotator._current_index = 0  # Point to cookie1
        result = rotator.get_current_cookie()

        # cookie1 should be cleaned up from _failed_cookies
        assert str(cookie1) not in rotator._failed_cookies
        # Should have switched to cookie2
        assert result == str(cookie2)
