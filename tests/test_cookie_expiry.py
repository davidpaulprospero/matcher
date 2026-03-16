"""Tests for cookie validation and expiration handling (US-113-011)."""

import pytest
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockCookieRotationConfig:
    """Mock config for testing cookie expiration."""
    enabled: bool = True
    cookie_files: List[str] = field(default_factory=list)
    rotation_strategy: str = "on_error"
    rotate_on_errors: List[str] = field(default_factory=lambda: [
        "429", "rate limit", "sign in"
    ])
    cooldown_seconds: int = 5
    max_rotations_per_session: int = 0
    # Expiration settings (US-113-011)
    cookie_expiry_warning_threshold_hours: int = 24
    rotate_before_expiry: bool = True
    # Proactive rotation threshold (US-136-005)
    proactive_rotation_threshold_hours: int = 0


class TestCookieExpiryParsing:
    """Tests for cookie expiration parsing from Netscape format."""

    @pytest.mark.fast
    def test_parse_expiry_from_netscape_format(self, tmp_path):
        """Test parsing expiration timestamp from Netscape cookie format."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Netscape format: domain, flag, path, secure, expiry, name, value
        # expiry = 1735689600 = 2025-01-01 00:00:00 UTC
        future_expiry = int(time.time()) + 86400 * 30  # 30 days from now
        cookie_file.write_text(
            f"# Netscape HTTP Cookie File\n"
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttestvalue\n"
            f".google.com\tTRUE\t/\tFALSE\t{future_expiry}\tNID\ttestvalue2\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        expiry, error = rotator._parse_cookie_expiry(str(cookie_file))
        assert error is None
        assert expiry is not None
        assert abs(expiry - future_expiry) < 2  # Allow 2 second tolerance

    @pytest.mark.fast
    def test_parse_session_cookie(self, tmp_path):
        """Test parsing when cookie has no expiry (session cookie)."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # expiry = 0 means session cookie
        cookie_file.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttestvalue\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        expiry, error = rotator._parse_cookie_expiry(str(cookie_file))
        # Should return None for expiry and a message about session cookies
        assert error == "no expiry found (session cookies)"
        assert expiry is None

    @pytest.mark.fast
    def test_parse_skips_comments_and_empty_lines(self, tmp_path):
        """Test that parser skips comment lines and empty lines."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        future_expiry = int(time.time()) + 86400 * 30
        cookie_file.write_text(
            f"# This is a comment\n"
            f"\n"
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttestvalue\n"
            f"# Another comment\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        expiry, error = rotator._parse_cookie_expiry(str(cookie_file))
        assert error is None
        assert expiry is not None

    @pytest.mark.fast
    def test_finds_earliest_expiry(self, tmp_path):
        """Test that parser finds the earliest expiry when multiple cookies exist."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # First cookie expires in 30 days, second in 10 days
        later_expiry = int(time.time()) + 86400 * 30
        sooner_expiry = int(time.time()) + 86400 * 10

        cookie_file.write_text(
            f"# Netscape HTTP Cookie File\n"
            f".youtube.com\tTRUE\t/\tFALSE\t{later_expiry}\tSID\tvalue1\n"
            f".google.com\tTRUE\t/\tFALSE\t{sooner_expiry}\tNID\tvalue2\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        expiry, error = rotator._parse_cookie_expiry(str(cookie_file))
        assert error is None
        # Should find the sooner expiry
        assert abs(expiry - sooner_expiry) < 2


class TestCookieExpiryCheck:
    """Tests for cookie expiration checking."""

    @pytest.mark.fast
    def test_cookie_valid_when_far_from_expiry(self, tmp_path):
        """Test cookie is valid when far from expiration threshold."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Expire in 7 days (well beyond 24 hour threshold)
        future_expiry = int(time.time()) + 86400 * 7
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24
        )
        rotator = CookieRotator(config)

        is_valid, reason = rotator._check_cookie_expiry(str(cookie_file))
        assert is_valid is True
        assert reason is None

    @pytest.mark.fast
    def test_cookie_expiring_soon_warning(self, tmp_path):
        """Test cookie approaching expiration triggers warning."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Expire in 12 hours (within 24 hour threshold)
        soon_expiry = int(time.time()) + 3600 * 12
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24
        )
        rotator = CookieRotator(config)

        is_valid, reason = rotator._check_cookie_expiry(str(cookie_file))
        assert is_valid is True  # Still valid, but warning
        assert reason is not None
        assert "expiring soon" in reason
        assert "h left" in reason

    @pytest.mark.fast
    def test_cookie_expired(self, tmp_path):
        """Test expired cookie is detected."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Expired 1 hour ago
        past_expiry = int(time.time()) - 3600
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{past_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        is_valid, reason = rotator._check_cookie_expiry(str(cookie_file))
        assert is_valid is False
        assert reason == "expired"

    @pytest.mark.fast
    def test_session_cookie_always_valid(self, tmp_path):
        """Test session cookies (no expiry) are always considered valid."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text(
            ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        is_valid, reason = rotator._check_cookie_expiry(str(cookie_file))
        assert is_valid is True
        assert reason is None


class TestCookieExpiryRotation:
    """Tests for proactive cookie rotation based on expiration."""

    @pytest.mark.fast
    def test_proactive_rotation_when_expiring_soon(self, tmp_path, caplog):
        """Test proactive rotation when cookie is expiring soon."""
        import logging
        from src.downloader.cookie_rotator import CookieRotator

        # Create two cookie files
        cookie1 = tmp_path / "cookies1.txt"
        cookie2 = tmp_path / "cookies2.txt"

        # Cookie 1 expires in 12 hours (expiring soon)
        soon_expiry = int(time.time()) + 3600 * 12
        cookie1.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest1\n"
        )

        # Cookie 2 expires in 30 days (fresh)
        later_expiry = int(time.time()) + 86400 * 30
        cookie2.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{later_expiry}\tSID\ttest2\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)],
            cookie_expiry_warning_threshold_hours=24,
            rotate_before_expiry=True
        )
        rotator = CookieRotator(config)

        # Get current cookie - should rotate to the fresher one
        current = rotator.get_current_cookie()

        # Should pick the fresher cookie
        assert "cookies2" in current

    @pytest.mark.fast
    def test_no_rotation_when_disabled(self, tmp_path):
        """Test no proactive rotation when rotate_before_expiry is False."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookies1.txt"
        cookie2 = tmp_path / "cookies2.txt"

        # Both expiring soon
        soon_expiry = int(time.time()) + 3600 * 12
        cookie1.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest1\n"
        )
        cookie2.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest2\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)],
            cookie_expiry_warning_threshold_hours=24,
            rotate_before_expiry=False  # Disabled
        )
        rotator = CookieRotator(config)

        current = rotator.get_current_cookie()
        # Should just return first cookie since proactive rotation is off
        assert "cookies1" in current


class TestCookieAgeTracking:
    """Tests for cookie age tracking and success correlation."""

    @pytest.mark.fast
    def test_track_cookie_age(self, tmp_path):
        """Test that cookie age is tracked correctly."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        future_expiry = int(time.time()) + 86400 * 30
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        # Initially age should be very small
        age = rotator.get_cookie_age_hours(str(cookie_file))
        assert age >= 0
        assert age < 1  # Should be less than 1 hour since creation

    @pytest.mark.fast
    def test_track_age_on_success(self, tmp_path):
        """Test cookie age is recorded on success for correlation analysis."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        future_expiry = int(time.time()) + 86400 * 30
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        # Record success
        rotator.mark_success(str(cookie_file))

        # Check age success tracking
        assert str(cookie_file) in rotator._cookie_age_success
        ages = rotator._cookie_age_success[str(cookie_file)]
        assert len(ages) == 1
        assert ages[0] >= 0


class TestCookieExpiryReport:
    """Tests for expiry status reporting."""

    @pytest.mark.fast
    def test_expiry_report_format(self, tmp_path):
        """Test expiry report contains expected fields."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        future_expiry = int(time.time()) + 86400 * 30
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        report = rotator.get_expiry_report()

        assert str(cookie_file) in report
        cookie_report = report[str(cookie_file)]
        assert "age_hours" in cookie_report
        assert "expiry_hours_until" in cookie_report
        assert "is_expired" in cookie_report
        assert "is_expiring_soon" in cookie_report

    @pytest.mark.fast
    def test_status_includes_expiry_info(self, tmp_path):
        """Test that get_status includes expiry report."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        future_expiry = int(time.time()) + 86400 * 30
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        status = rotator.get_status()

        assert "expiry_report" in status
        assert "expiry_warning_threshold_hours" in status
        assert "rotate_before_expiry" in status
        assert status["expiry_warning_threshold_hours"] == 24
        assert status["rotate_before_expiry"] is True


class TestCookieExpiryParsingEnhancements:
    """Tests for enhanced cookie expiry parsing (US-136-005)."""

    @pytest.mark.fast
    def test_parse_expiry_from_json_format(self, tmp_path):
        """Test parsing expiration timestamp from JSON cookie format."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.json"
        future_expiry = int(time.time()) + 86400 * 30  # 30 days from now

        # JSON format from browser extensions
        cookie_file.write_text(
            f'[{{"domain": ".youtube.com", "expirationDate": {future_expiry}, "name": "SID", "value": "test"}}, '
            f'{{"domain": ".google.com", "expirationDate": {future_expiry + 86400}, "name": "NID", "value": "test2"}}]'
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        expiry, error = rotator._parse_cookie_expiry(str(cookie_file))
        assert error is None
        assert expiry is not None
        # Should find the earliest expiry
        assert abs(expiry - future_expiry) < 2

    @pytest.mark.fast
    def test_parse_expiry_from_ldjson_format(self, tmp_path):
        """Test parsing expiration timestamp from line-delimited JSON format."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.jsonl"
        sooner_expiry = int(time.time()) + 86400 * 10  # 10 days
        later_expiry = int(time.time()) + 86400 * 30  # 30 days

        # LDJSON format (one JSON object per line)
        cookie_file.write_text(
            f'{{"domain": ".youtube.com", "expirationDate": {later_expiry}, "name": "SID", "value": "test"}}\n'
            f'{{"domain": ".google.com", "expirationDate": {sooner_expiry}, "name": "NID", "value": "test2"}}'
        )

        config = MockCookieRotationConfig(cookie_files=[str(cookie_file)])
        rotator = CookieRotator(config)

        expiry, error = rotator._parse_cookie_expiry(str(cookie_file))
        assert error is None
        assert expiry is not None
        # Should find the sooner expiry
        assert abs(expiry - sooner_expiry) < 2


class TestProactiveRotationThreshold:
    """Tests for proactive rotation threshold configuration (US-136-005)."""

    @pytest.mark.fast
    def test_proactive_threshold_defaults_to_warning_threshold(self, tmp_path):
        """Test that proactive threshold defaults to warning threshold when 0."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24,
            proactive_rotation_threshold_hours=0  # Default to warning threshold
        )
        rotator = CookieRotator(config)

        # Should use warning threshold (24) when proactive is 0
        assert rotator._proactive_rotation_threshold_hours == 24
        assert rotator._proactive_rotation_threshold_hours == rotator._expiry_warning_threshold_hours

    @pytest.mark.fast
    def test_proactive_threshold_uses_custom_value(self, tmp_path):
        """Test that proactive threshold uses custom value when set."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24,
            proactive_rotation_threshold_hours=48  # Custom value
        )
        rotator = CookieRotator(config)

        assert rotator._proactive_rotation_threshold_hours == 48
        assert rotator._proactive_rotation_threshold_hours != rotator._expiry_warning_threshold_hours

    @pytest.mark.fast
    def test_status_includes_proactive_threshold(self, tmp_path):
        """Test that get_status includes proactive rotation threshold."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        cookie_file.write_text(".youtube.com\tTRUE\t/\tFALSE\t0\tSID\ttest\n")

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            proactive_rotation_threshold_hours=12
        )
        rotator = CookieRotator(config)

        status = rotator.get_status()

        assert "proactive_rotation_threshold_hours" in status
        assert status["proactive_rotation_threshold_hours"] == 12


class TestComprehensiveHealthScore:
    """Tests for comprehensive health scoring (US-136-005)."""

    @pytest.mark.fast
    def test_comprehensive_health_includes_expiry(self, tmp_path):
        """Test that comprehensive health score factors in expiry."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Expire in 12 hours (within warning threshold)
        soon_expiry = int(time.time()) + 3600 * 12
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24,
            proactive_rotation_threshold_hours=24
        )
        rotator = CookieRotator(config)

        # Add some success history
        for _ in range(5):
            rotator.mark_success(str(cookie_file))

        # Get comprehensive score
        score = rotator.get_comprehensive_health_score(str(cookie_file))

        # Should be lower than success rate alone due to expiry proximity
        success_only = rotator.get_health_score(str(cookie_file))
        assert score < success_only  # Expiry penalty reduces score

    @pytest.mark.fast
    def test_comprehensive_health_with_expired_cookie(self, tmp_path):
        """Test that expired cookies get low comprehensive health score."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Expired 1 hour ago
        past_expiry = int(time.time()) - 3600
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{past_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24
        )
        rotator = CookieRotator(config)

        # Add some success history
        for _ in range(5):
            rotator.mark_success(str(cookie_file))

        # Get comprehensive score - should be lower due to expiry
        score = rotator.get_comprehensive_health_score(str(cookie_file))

        # Expired cookies should have reduced score
        # With 70% weight on success, 30% on expiry: 1.0 * 0.7 + 0.0 * 0.3 = 0.7
        # But since expired, the combined score should be less than success-only score
        success_only = rotator.get_health_score(str(cookie_file))
        assert score < success_only  # Expiry penalty reduces score

    @pytest.mark.fast
    def test_comprehensive_health_with_fresh_cookie(self, tmp_path):
        """Test that fresh cookies maintain high health score."""
        from src.downloader.cookie_rotator import CookieRotator

        cookie_file = tmp_path / "cookies.txt"
        # Expire in 30 days (well beyond any threshold)
        future_expiry = int(time.time()) + 86400 * 30
        cookie_file.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{future_expiry}\tSID\ttest\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie_file)],
            cookie_expiry_warning_threshold_hours=24
        )
        rotator = CookieRotator(config)

        # Add some success history
        for _ in range(5):
            rotator.mark_success(str(cookie_file))

        # Get comprehensive score
        score = rotator.get_comprehensive_health_score(str(cookie_file))
        success_only = rotator.get_health_score(str(cookie_file))

        # Should be close to success-only score since expiry is far
        assert abs(score - success_only) < 0.1


class TestExpiryWarningLogging:
    """Tests for expiry warning logging (US-136-005)."""

    @pytest.mark.fast
    def test_warning_logged_when_no_fresher_alternative(self, tmp_path, caplog):
        """Test warning logged when cookie expiring soon but no fresher alternative."""
        import logging
        from src.downloader.cookie_rotator import CookieRotator

        cookie1 = tmp_path / "cookies1.txt"
        cookie2 = tmp_path / "cookies2.txt"

        # Both expiring soon
        soon_expiry = int(time.time()) + 3600 * 12
        cookie1.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest1\n"
        )
        cookie2.write_text(
            f".youtube.com\tTRUE\t/\tFALSE\t{soon_expiry}\tSID\ttest2\n"
        )

        config = MockCookieRotationConfig(
            cookie_files=[str(cookie1), str(cookie2)],
            cookie_expiry_warning_threshold_hours=24,
            rotate_before_expiry=True
        )
        rotator = CookieRotator(config)

        # Get current cookie - should log warning since no fresher alternative
        with caplog.at_level(logging.WARNING):
            current = rotator.get_current_cookie()

        # Warning should be logged about expiring soon with no fresher alternative
        assert "expiring soon" in caplog.text.lower()
        assert "no fresher alternative" in caplog.text.lower()
