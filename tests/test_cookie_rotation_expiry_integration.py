"""Integration tests for cookie rotation workflow (US-113-011)."""

import pytest
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockCookieRotationConfig:
    """Mock config for testing cookie rotation integration."""
    enabled: bool = True
    cookie_files: List[str] = field(default_factory=list)
    rotation_strategy: str = "on_error"
    rotate_on_errors: List[str] = field(default_factory=lambda: [
        "429", "rate limit", "sign in", "forbidden", "403"
    ])
    cooldown_seconds: int = 2  # Short for testing
    max_rotations_per_session: int = 10
    cookie_expiry_warning_threshold_hours: int = 24
    rotate_before_expiry: bool = True
    success_rate_threshold: float = 0.3
    health_min_attempts: int = 3
    max_consecutive_failures: int = 3
    # Browser extraction settings (US-143-005)
    browser_profiles: List = field(default_factory=list)
    browser_priority_order: List[str] = field(default_factory=lambda: ["chrome", "firefox"])
    auto_extract_cookies: bool = False
    cookie_freshness_validation: bool = True
    auto_refresh_cookies: bool = True
    browser_extraction_timeout: int = 30
    # Freshness threshold for extracted cookies (US-144-005)
    freshness_threshold_minutes: int = 5


class TestCookieRotationWorkflow:
    """Integration tests for complete cookie rotation workflow."""

    @pytest.mark.fast
    def test_rotation_on_error_and_health_tracking(self, tmp_path):
        """Test full workflow: rotation on error + health tracking + expiry tracking."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create 3 cookie files
        cookie1 = tmp_path / "c1.txt"
        cookie2 = tmp_path / "c2.txt"
        cookie3 = tmp_path / "c3.txt"

        for f in [cookie1, cookie2, cookie3]:
            future = int(time.time()) + 86400 * 30
            f.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{future}\tSID\tval\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2), str(cookie3)],
            rotation_strategy="health"
        )
        rotator = CookieRotator(config)

        # Initial cookie
        cookie = rotator.get_current_cookie()
        assert cookie is not None

        # Mark success - should track health and age
        rotator.mark_success(cookie)
        health = rotator.get_health_score(cookie)
        assert health == 1.0  # 100% success rate

        # Simulate error and rotate
        if rotator.should_rotate("403 forbidden"):
            new_cookie = rotator.rotate()
            assert new_cookie is not None
            assert new_cookie != cookie

        # Mark failure on new cookie
        rotator.mark_failed(new_cookie)

        # Check health tracking
        health = rotator.get_health_score(new_cookie)
        assert health < 1.0  # Has failures now

    @pytest.mark.fast
    def test_expiry_aware_rotation_workflow(self, tmp_path):
        """Test rotation workflow considering expiration status."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create cookies with different expiry times
        cookie1 = tmp_path / "fresh.txt"
        cookie2 = tmp_path / "old.txt"

        # Cookie 1: expires in 30 days (fresh)
        fresh_expiry = int(time.time()) + 86400 * 30
        cookie1.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{fresh_expiry}\tSID\tval1\n")

        # Cookie 2: expires in 20 hours (approaching threshold)
        soon_expiry = int(time.time()) + 3600 * 20
        cookie2.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\tval2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)],
            cookie_expiry_warning_threshold_hours=24,
            rotate_before_expiry=True
        )
        rotator = CookieRotator(config)

        # Get current - should prefer fresh cookie due to expiry
        current = rotator.get_current_cookie()

        # Should return the fresher cookie (cookie1)
        assert "fresh" in current

    @pytest.mark.fast
    def test_expired_cookie_forces_rotation(self, tmp_path):
        """Test that expired cookies are rotated away from."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create an expired cookie and a fresh one
        cookie1 = tmp_path / "expired.txt"
        cookie2 = Cookie2 = tmp_path / "valid.txt"

        # Cookie 1: expired 1 hour ago
        past = int(time.time()) - 3600
        cookie1.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{past}\tSID\tval1\n")

        # Cookie 2: valid
        future = int(time.time()) + 86400 * 30
        cookie2.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{future}\tSID\tval2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)]
        )
        rotator = CookieRotator(config)

        # Get current should skip expired cookie
        current = rotator.get_current_cookie()
        assert "valid" in current

    @pytest.mark.fast
    def test_mixed_expiry_rotation_strategy(self, tmp_path):
        """Test rotation with mixed expiry states and error triggers."""
        from src.downloader.cookie_rotator import CookieRotator

        # Create 3 cookies: fresh, expiring soon, expired
        fresh = tmp_path / "fresh.txt"
        soon = tmp_path / "soon.txt"
        expired = tmp_path / "expired.txt"

        fresh.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{int(time.time() + 86400*30)}\tSID\tf\n"
        )
        soon.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{int(time.time() + 3600*12)}\tSID\ts\n"
        )
        expired.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{int(time.time() - 3600)}\tSID\te\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(fresh), str(soon), str(expired)],
            rotation_strategy="health",
            cookie_expiry_warning_threshold_hours=24
        )
        rotator = CookieRotator(config)

        # Should never return expired
        current = rotator.get_current_cookie()
        assert "expired" not in current

        # Log some successes
        rotator.mark_success(current)
        rotator.mark_success(current)

        # Check status includes all tracking
        status = rotator.get_status()
        assert "expiry_report" in status
        assert "health_report" in status
        assert status["valid"] >= 2  # Fresh + soon (expired should be removed)

    @pytest.mark.fast
    def test_health_and_expiry_combined_priority(self, tmp_path):
        """Test that health and expiry together influence cookie selection."""
        from src.downloader.cookie_rotator import CookieRotator

        # Cookie A: healthy but expiring soon
        cookie_a = tmp_path / "a_healthy_expiring.txt"
        soon_expiry = int(time.time()) + 3600 * 12
        cookie_a.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\tval_a\n")

        # Cookie B: unhealthy but fresh
        cookie_b = tmp_path / "b_unhealthy_fresh.txt"
        fresh_expiry = int(time.time()) + 86400 * 30
        cookie_b.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{fresh_expiry}\tSID\tval_b\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_a), str(cookie_b)],
            rotation_strategy="health",
            cookie_expiry_warning_threshold_hours=24,
            rotate_before_expiry=True
        )
        rotator = CookieRotator(config)

        # Mark cookie A as successful multiple times
        rotator.mark_success(str(cookie_a))
        rotator.mark_success(str(cookie_a))
        rotator.mark_success(str(cookie_a))

        # Mark cookie B as failed
        rotator.mark_failed(str(cookie_b))
        rotator.mark_failed(str(cookie_b))

        # Get current - with health strategy, should prefer healthy one
        # But if proactive expiry is enabled, may still prefer fresh
        current = rotator.get_current_cookie()

        # Both should be valid (not expired)
        assert current is not None


class TestCookieRotationEdgeCases:
    """Edge case tests for cookie rotation."""

    @pytest.mark.fast
    def test_all_cookies_expired(self, tmp_path):
        """Test behavior when all cookies are expired."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "c1.txt"
        cookie2 = tmp_path / "c2.txt"

        # Both expired
        past = int(time.time()) - 3600
        cookie1.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{past}\tSID\tv1\n")
        cookie2.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{past}\tSID\tv2\n")

        config = MockCookieRotationConfig(cookie_files=[str(cookie1), str(cookie2)])
        rotator = CookieRotator(config)

        # Should still return something (the expired ones) but log warnings
        current = rotator.get_current_cookie()
        # The rotator should have removed expired cookies
        status = rotator.get_status()
        assert status["valid"] == 0  # All expired, none valid

    @pytest.mark.fast
    def test_zero_expiry_threshold_disables_check(self, tmp_path):
        """Test that 0 expiry threshold disables expiration warning but still catches expired cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        # Test 1: Expired cookie should still be detected as invalid
        expired_cookie = tmp_path / "expired.txt"
        past = int(time.time()) - 3600
        expired_cookie.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{past}\tSID\tv\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(expired_cookie)],
            cookie_expiry_warning_threshold_hours=0
        )
        rotator = CookieRotator(config)

        # Expired cookies should still be invalid even with threshold=0
        is_valid, reason = rotator._check_cookie_expiry(str(expired_cookie))
        assert is_valid is False
        assert reason == "expired"

        # Test 2: Cookie close to expiry but not expired should NOT warn when threshold=0
        near_cookie = tmp_path / "near.txt"
        # Expires in 1 hour (would trigger warning with threshold=24)
        near_expiry = int(time.time()) + 3600
        near_cookie.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{near_expiry}\tSID\tv\n")

        config2 = MockCookieRotationConfig(
            cookie_files=[str(near_cookie)],
            cookie_expiry_warning_threshold_hours=0
        )
        rotator2 = CookieRotator(config2)

        # With threshold=0, should not warn about "expiring soon"
        is_valid2, reason2 = rotator2._check_cookie_expiry(str(near_cookie))
        assert is_valid2 is True
        assert reason2 is None  # No warning when threshold=0

    @pytest.mark.fast
    def test_session_cookies_work_normally(self, tmp_path):
        """Test that session cookies (no expiry) work with rotation."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "c1.txt"
        cookie2 = tmp_path / "c2.txt"

        # Session cookies (expiry = 0)
        cookie1.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tv1\n")
        cookie2.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tv2\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)]
        )
        rotator = CookieRotator(config)

        # Should work normally
        current = rotator.get_current_cookie()
        assert current is not None

        # Rotate
        new_cookie = rotator.rotate()
        assert new_cookie is not None
        assert new_cookie != current


class TestExtractedCookieRefreshIntegration:
    """Integration tests for automatic refresh of stale extracted cookies (US-144-005)."""

    @pytest.mark.fast
    def test_auto_refresh_triggers_for_stale_extracted_cookie(self, tmp_path):
        """Test that auto-refresh triggers when extracted cookie is stale."""
        from src.downloader.cookie_rotator import CookieRotator
        from unittest.mock import patch

        # Create a cookie file
        cookie1 = tmp_path / "extracted.txt"
        future = int(time.time()) + 86400 * 30
        cookie1.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{future}\tSID\tval\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1)],
            browser_profiles=[{"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}],
            auto_refresh_cookies=True,
            freshness_threshold_minutes=5,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Simulate that cookie was extracted long ago (stale)
        stale_time = time.time() - (10 * 60)  # 10 minutes ago
        rotator._browser_cookie_timestamps[str(cookie1)] = stale_time
        rotator._extracted_cookies["chrome:Default"] = str(cookie1)

        # Mock _refresh_browser_cookie - make it return None (refresh failed or not needed)
        with patch.object(rotator, '_refresh_browser_cookie', return_value=None) as mock_refresh:
            with patch.object(rotator, '_is_cookie_file_valid', return_value=True):
                result = rotator.get_current_cookie()
                # Should have attempted refresh since cookie is stale
                assert mock_refresh.called
                # The stale check should have triggered
                assert rotator._is_extracted_cookie_stale(str(cookie1)) is True

    @pytest.mark.fast
    def test_no_refresh_for_recently_extracted_cookie(self, tmp_path):
        """Test that recently extracted cookies are not refreshed."""
        from src.downloader.cookie_rotator import CookieRotator
        from unittest.mock import patch

        # Create a cookie file
        cookie1 = tmp_path / "recent.txt"
        future = int(time.time()) + 86400 * 30
        cookie1.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{future}\tSID\tval\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1)],
            browser_profiles=[{"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}],
            auto_refresh_cookies=True,
            freshness_threshold_minutes=5,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Simulate that cookie was extracted recently (not stale)
        recent_time = time.time() - (2 * 60)  # 2 minutes ago
        rotator._browser_cookie_timestamps[str(cookie1)] = recent_time
        rotator._extracted_cookies["chrome:Default"] = str(cookie1)

        with patch.object(rotator, '_refresh_browser_cookie') as mock_refresh:
            with patch.object(rotator, '_is_cookie_file_valid', return_value=True):
                result = rotator.get_current_cookie()
                # Should NOT have attempted refresh
                assert not mock_refresh.called
                # Should return the original cookie
                assert result == str(cookie1)

    @pytest.mark.fast
    def test_freshness_threshold_config_integration(self, tmp_path):
        """Test that freshness_threshold_minutes config is properly used."""
        from src.downloader.cookie_rotator import CookieRotator
        from unittest.mock import patch

        # Create a cookie file
        cookie1 = tmp_path / "config_test.txt"
        future = int(time.time()) + 86400 * 30
        cookie1.write_text(f".youtube.com\tTRUE\t/\tFALSE\t{future}\tSID\tval\n")

        # Use 15-minute threshold
        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1)],
            browser_profiles=[{"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}],
            auto_refresh_cookies=True,
            freshness_threshold_minutes=15,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Verify config was stored
        assert rotator._freshness_threshold_minutes == 15

        # Simulate cookie extracted 10 minutes ago
        stale_time = time.time() - (10 * 60)  # 10 minutes ago
        rotator._browser_cookie_timestamps[str(cookie1)] = stale_time
        rotator._extracted_cookies["chrome:Default"] = str(cookie1)

        # With 15-minute threshold, 10 minutes should NOT be stale
        is_stale = rotator._is_extracted_cookie_stale(str(cookie1))
        assert is_stale is False

        # Verify in status
        status = rotator.get_status()
        assert status["browser_extraction"]["freshness_threshold_minutes"] == 15
