"""
Smoke tests for conftest factory fixtures (US-55-010).

Verifies that each download-segments factory fixture is importable,
callable, and produces correctly-typed instances with both defaults
and custom overrides.
"""

import pytest


# ── download_config_factory ─────────────────────────────────────────────────

class TestDownloadConfigFactory:
    """Smoke tests for download_config_factory fixture."""

    def test_default_values(self, download_config_factory):
        cfg = download_config_factory()
        assert cfg.cookies_from_browser == "chrome"
        assert cfg.cookiefile is None
        assert cfg.socket_timeout == 30
        assert cfg.max_retries == 3
        assert cfg.escalation.enabled is True

    def test_override_values(self, download_config_factory):
        cfg = download_config_factory(
            cookies_from_browser="firefox",
            cookiefile="/tmp/cookies.txt",
            socket_timeout=60,
            max_retries=5,
            escalation={'enabled': False, 'max_tier': 2},
        )
        assert cfg.cookies_from_browser == "firefox"
        assert cfg.cookiefile == "/tmp/cookies.txt"
        assert cfg.socket_timeout == 60
        assert cfg.max_retries == 5
        assert cfg.escalation.enabled is False
        assert cfg.escalation.max_tier == 2

    def test_extra_fields(self, download_config_factory):
        cfg = download_config_factory(custom_field="value")
        assert cfg.custom_field == "value"


# ── escalation_result_factory ───────────────────────────────────────────────

class TestEscalationResultFactory:
    """Smoke tests for escalation_result_factory fixture."""

    def test_default_values(self, escalation_result_factory):
        from src.downloader.types import EscalationTier

        result = escalation_result_factory()
        assert result.args == ['--impersonate', 'Chrome-131:Windows-11']
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert result.rotate_cookies is False
        assert result.rotate_vpn is False

    def test_custom_tier_and_args(self, escalation_result_factory):
        from src.downloader.types import EscalationTier

        result = escalation_result_factory(
            args=['--impersonate', 'Safari-18:Macos-15', '--extractor-args', 'youtube:player_client=web_safari'],
            tier=2,
            rotate_cookies=True,
        )
        assert '--extractor-args' in result.args
        assert result.tier == EscalationTier.EXTRACTOR_ARGS
        assert result.rotate_cookies is True

    def test_vpn_tier(self, escalation_result_factory):
        from src.downloader.types import EscalationTier

        result = escalation_result_factory(tier=4, rotate_vpn=True)
        assert result.tier == EscalationTier.VPN_ROTATION
        assert result.rotate_vpn is True


# ── segment_download_stats_factory ──────────────────────────────────────────

class TestSegmentDownloadStatsFactory:
    """Smoke tests for segment_download_stats_factory fixture."""

    def test_default_values(self, segment_download_stats_factory):
        from src.stages.download_segments import SegmentDownloadStats

        stats = segment_download_stats_factory()
        assert isinstance(stats, SegmentDownloadStats)
        assert stats.succeeded == 0
        assert stats.failed == 0
        assert stats.cached == 0
        assert stats.total == 0
        assert stats.error_categories == {}

    def test_custom_counts(self, segment_download_stats_factory):
        stats = segment_download_stats_factory(
            succeeded=5,
            failed=2,
            cached=3,
            attempted=7,
            total=10,
        )
        assert stats.succeeded == 5
        assert stats.failed == 2
        assert stats.cached == 3
        assert stats.attempted == 7
        assert stats.total == 10

    def test_error_categories(self, segment_download_stats_factory):
        stats = segment_download_stats_factory(
            error_categories={'network': 3, 'bot_detection': 1},
        )
        assert stats.error_categories == {'network': 3, 'bot_detection': 1}

    def test_durations_and_bytes(self, segment_download_stats_factory):
        stats = segment_download_stats_factory(
            segment_durations=[1.5, 2.3, 0.8],
            total_bytes=1024000,
        )
        assert stats.segment_durations == [1.5, 2.3, 0.8]
        assert stats.total_bytes == 1024000
