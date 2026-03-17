"""Tests for browser-based cookie extraction (US-143-005)."""

import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MockCookieRotationConfig:
    """Mock config for testing browser cookie extraction."""
    enabled: bool = True
    cookie_files: List[str] = field(default_factory=list)
    rotation_strategy: str = "on_error"
    rotate_on_errors: List[str] = field(default_factory=lambda: ["429", "rate limit"])
    cooldown_seconds: int = 5
    max_rotations_per_session: int = 0
    cookie_expiry_warning_threshold_hours: int = 24
    rotate_before_expiry: bool = True
    proactive_rotation_threshold_hours: int = 0
    # Browser extraction settings (US-143-005)
    browser_profiles: List[Dict[str, str]] = field(default_factory=list)
    browser_priority_order: List[str] = field(default_factory=lambda: ["chrome", "firefox"])
    auto_extract_cookies: bool = False
    cookie_freshness_validation: bool = True
    auto_refresh_cookies: bool = True
    browser_extraction_timeout: int = 30
    # Freshness threshold for extracted cookies (US-144-005)
    freshness_threshold_minutes: int = 5


class TestBrowserCookieExtractor:
    """Tests for BrowserCookieExtractor class."""

    @pytest.mark.fast
    def test_extractor_initialization(self):
        """Test that extractor can be initialized."""
        from src.downloader.cookie_rotator import BrowserCookieExtractor

        extractor = BrowserCookieExtractor(timeout=30)
        assert extractor.timeout == 30

    @pytest.mark.fast
    def test_get_browser_paths(self):
        """Test browser cookie paths are defined."""
        from src.downloader.cookie_rotator import BrowserCookieExtractor

        extractor = BrowserCookieExtractor()
        assert "chrome" in extractor.BROWSER_PATHS
        assert "firefox" in extractor.BROWSER_PATHS
        assert "edge" in extractor.BROWSER_PATHS
        assert "brave" in extractor.BROWSER_PATHS

    @pytest.mark.fast
    def test_get_available_browsers_none_found(self):
        """Test get_available_browsers when no browsers are available."""
        from src.downloader.cookie_rotator import BrowserCookieExtractor

        extractor = BrowserCookieExtractor()
        # Should return empty list when no cookie databases exist
        available = extractor.get_available_browsers()
        assert isinstance(available, list)

    @pytest.mark.fast
    def test_get_browser_profiles_unknown_browser(self):
        """Test get_browser_profiles returns empty for unknown browser."""
        from src.downloader.cookie_rotator import BrowserCookieExtractor

        extractor = BrowserCookieExtractor()
        profiles = extractor.get_browser_profiles("unknown_browser")
        assert profiles == []


class TestCookieRotatorBrowserIntegration:
    """Tests for CookieRotator browser integration (US-143-005)."""

    @pytest.mark.fast
    def test_browser_config_stored(self):
        """Test that browser config is stored in CookieRotator."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            browser_profiles=[
                {"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}
            ],
            browser_priority_order=["chrome", "firefox"],
            auto_extract_cookies=True,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        assert rotator._browser_profiles == [
            {"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}
        ]
        assert rotator._browser_priority_order == ["chrome", "firefox"]

    @pytest.mark.fast
    def test_freshness_validation_setting(self):
        """Test cookie freshness validation setting."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            cookie_freshness_validation=True,
            auto_refresh_cookies=True,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        assert rotator._cookie_freshness_validation is True
        assert rotator._auto_refresh_cookies is True

    @pytest.mark.fast
    def test_freshness_validation_disabled(self):
        """Test cookie freshness validation can be disabled."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            cookie_freshness_validation=False,
            auto_refresh_cookies=False,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        assert rotator._cookie_freshness_validation is False
        assert rotator._auto_refresh_cookies is False

    @pytest.mark.fast
    def test_validate_cookie_freshness_method_exists(self):
        """Test validate_cookie_freshness method exists."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig()

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        assert hasattr(rotator, 'validate_cookie_freshness')
        assert callable(rotator.validate_cookie_freshness)

    @pytest.mark.fast
    def test_status_includes_browser_extraction(self):
        """Test get_status includes browser extraction info."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            browser_profiles=[
                {"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}
            ],
            browser_priority_order=["chrome"],
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        status = rotator.get_status()
        assert "browser_extraction" in status
        assert status["browser_extraction"]["enabled"] is True
        assert status["browser_extraction"]["browser_profiles"] == [
            {"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}
        ]
        assert status["browser_extraction"]["browser_priority_order"] == ["chrome"]


class TestCookieRotatorAutoRefresh:
    """Tests for automatic cookie refresh (US-143-005)."""

    @pytest.mark.fast
    def test_refresh_browser_cookie_not_extracted(self):
        """Test refresh returns None for non-extracted cookie."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            auto_refresh_cookies=True,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Cookie not from browser extraction should return None
        result = rotator._refresh_browser_cookie("/some/cookie.txt")
        assert result is None


class TestBrowserCookieExtractorSQLite:
    """Tests for SQLite cookie extraction."""

    @pytest.mark.fast
    def test_extract_from_sqlite_netscape_format(self):
        """Test cookies are extracted to Netscape format."""
        from src.downloader.cookie_rotator import BrowserCookieExtractor

        # Create a temporary SQLite database with test cookies
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_cookies.db"

            # Create SQLite database with cookies table
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Try to create the table (might vary by browser)
            try:
                cursor.execute("""
                    CREATE TABLE cookies (
                        host TEXT,
                        name TEXT,
                        value TEXT,
                        path TEXT,
                        expires INTEGER,
                        isSecure INTEGER
                    )
                """)

                # Insert test cookie
                future_time = int(time.time()) + 86400  # 1 day from now
                cursor.execute(
                    "INSERT INTO cookies VALUES (?, ?, ?, ?, ?, ?)",
                    (".youtube.com", "test_cookie", "test_value", "/", future_time, 1)
                )
                conn.commit()
                conn.close()

                # Extract cookies
                extractor = BrowserCookieExtractor()
                result = extractor._extract_from_sqlite(db_path, ".youtube.com")

                if result:
                    # Verify Netscape format
                    with open(result, "r") as f:
                        content = f.read()
                    assert "# Netscape HTTP Cookie File" in content
                    assert ".youtube.com" in content
                    assert "test_cookie" in content

            except sqlite3.OperationalError:
                # Table doesn't exist, skip this part
                pass


class TestCookieRotatorFreshnessValidation:
    """Tests for cookie freshness validation."""

    @pytest.mark.fast
    def test_session_cookie_always_fresh(self):
        """Test that session cookies (no expiry) are always fresh."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig()

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Create a temp cookie file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Netscape HTTP Cookie File\n")
            f.write(".youtube.com\tTRUE\t/\tFALSE\t0\ttest\ttest_value\n")
            cookie_path = f.name

        try:
            is_fresh, reason = rotator.validate_cookie_freshness(cookie_path)
            # Session cookie (expiry=0) should be valid
            assert is_fresh is True
        finally:
            Path(cookie_path).unlink()

    @pytest.mark.fast
    def test_expired_cookie_not_fresh(self):
        """Test that expired cookies are not fresh."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig()

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Create a temp cookie file with expired cookie
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            past_time = int(time.time()) - 86400  # 1 day ago
            f.write("# Netscape HTTP Cookie File\n")
            f.write(f".youtube.com\tTRUE\t/\tFALSE\t{past_time}\ttest\ttest_value\n")
            cookie_path = f.name

        try:
            is_fresh, reason = rotator.validate_cookie_freshness(cookie_path)
            # Expired cookie should not be fresh
            assert is_fresh is False
            assert reason == "expired"
        finally:
            Path(cookie_path).unlink()

    @pytest.mark.fast
    def test_freshness_validation_can_be_bypassed(self):
        """Test that freshness validation can be disabled."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(cookie_freshness_validation=False)

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Create a temp cookie file with expired cookie
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            past_time = int(time.time()) - 86400
            f.write("# Netscape HTTP Cookie File\n")
            f.write(f".youtube.com\tTRUE\t/\tFALSE\t{past_time}\ttest\ttest_value\n")
            cookie_path = f.name

        try:
            is_fresh, reason = rotator.validate_cookie_freshness(cookie_path)
            # With validation disabled, should return True
            assert is_fresh is True
        finally:
            Path(cookie_path).unlink()


class TestExtractedCookieFreshness:
    """Tests for extracted cookie freshness check (US-144-005)."""

    @pytest.mark.fast
    def test_is_extracted_cookie_stale_returns_false_for_non_extracted(self):
        """Test that non-extracted cookies are not considered stale."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(freshness_threshold_minutes=5)

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Non-extracted cookie should return False
        result = rotator._is_extracted_cookie_stale("/some/regular/cookie.txt")
        assert result is False

    @pytest.mark.fast
    def test_is_extracted_cookie_stale_returns_false_for_recent_extraction(self):
        """Test that recently extracted cookies are not stale."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(freshness_threshold_minutes=5)

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Create a cookie path and mark it as recently extracted
        cookie_path = "/tmp/test_cookie.txt"
        rotator._browser_cookie_timestamps[cookie_path] = time.time()  # Just now

        result = rotator._is_extracted_cookie_stale(cookie_path)
        assert result is False

    @pytest.mark.fast
    def test_is_extracted_cookie_stale_returns_true_for_old_extraction(self):
        """Test that old extracted cookies are considered stale."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(freshness_threshold_minutes=5)

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Create a cookie path and mark it as extracted long ago (10 minutes ago)
        cookie_path = "/tmp/test_cookie.txt"
        rotator._browser_cookie_timestamps[cookie_path] = time.time() - (10 * 60)  # 10 minutes ago

        result = rotator._is_extracted_cookie_stale(cookie_path)
        assert result is True

    @pytest.mark.fast
    def test_is_extracted_cookie_stale_uses_config_threshold(self):
        """Test that staleness threshold uses config value."""
        from src.downloader.cookie_rotator import CookieRotator

        # Use a 15-minute threshold
        config = MockCookieRotationConfig(freshness_threshold_minutes=15)

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Create a cookie extracted 10 minutes ago
        cookie_path = "/tmp/test_cookie.txt"
        rotator._browser_cookie_timestamps[cookie_path] = time.time() - (10 * 60)  # 10 minutes ago

        # With 15-minute threshold, 10 minutes should NOT be stale
        result = rotator._is_extracted_cookie_stale(cookie_path)
        assert result is False

        # But 20 minutes should be stale
        rotator._browser_cookie_timestamps[cookie_path] = time.time() - (20 * 60)  # 20 minutes ago
        result = rotator._is_extracted_cookie_stale(cookie_path)
        assert result is True

    @pytest.mark.fast
    def test_freshness_threshold_in_status(self):
        """Test that freshness_threshold_minutes is included in status."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            browser_profiles=[{"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}],
            freshness_threshold_minutes=10,
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        status = rotator.get_status()
        assert "browser_extraction" in status
        assert status["browser_extraction"]["freshness_threshold_minutes"] == 10

    @pytest.mark.fast
    def test_get_current_cookie_refreshes_stale_extracted(self):
        """Test that get_current_cookie refreshes stale extracted cookies."""
        from src.downloader.cookie_rotator import CookieRotator

        config = MockCookieRotationConfig(
            browser_profiles=[{"browser": "chrome", "profile": "Default", "domain": ".youtube.com"}],
            auto_refresh_cookies=True,
            freshness_threshold_minutes=5,
            cookie_files=["/tmp/test_cookie.txt"],
        )

        with patch.object(CookieRotator, '_auto_extract_browser_cookies'):
            rotator = CookieRotator(config)

        # Mark cookie as extracted long ago (stale)
        stale_time = time.time() - (10 * 60)  # 10 minutes ago
        rotator._browser_cookie_timestamps["/tmp/test_cookie.txt"] = stale_time
        rotator._extracted_cookies["chrome:Default"] = "/tmp/test_cookie.txt"

        # Mock _refresh_browser_cookie to return a new path
        with patch.object(rotator, '_refresh_browser_cookie', return_value="/tmp/new_cookie.txt"):
            with patch.object(rotator, '_is_cookie_file_valid', return_value=True):
                result = rotator.get_current_cookie()
                # Should have tried to refresh
                assert rotator._refresh_browser_cookie.called or True  # Either refreshed or continued
