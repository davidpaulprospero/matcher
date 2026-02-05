"""
Tests for CaptionHealer - intelligent caption format discovery and fallback.

US-64-002: CaptionHealer for intelligent caption format discovery
US-64-003: Negative caching to prevent repeated failed lookups
"""

import pytest
import time
import tempfile
from unittest.mock import MagicMock, patch

from src.agents.healers.caption import CaptionHealer, SubtitleInfo, NoCaptionsStatus
from src.agents.base import HealerAction, HealerResult
from src.caption.exceptions import (
    CaptionFormatUnavailableError,
    CaptionUnavailableError,
    CaptionFetchError,
)


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = MagicMock()
    config.download = MagicMock()
    config.download.caption_first = MagicMock()
    config.download.caption_first.enable_auto_subs = False
    # US-64-003: Set negative cache TTL config values
    config.download.caption_first.negative_cache_ttl_seconds = 3600
    config.download.caption_first.negative_cache_ttl_hours = 1.0
    return config


@pytest.fixture
def healer(mock_config):
    """Create a CaptionHealer instance."""
    return CaptionHealer(mock_config, "/test/project")


@pytest.fixture
def mock_state():
    """Create a mock pipeline state."""
    return MagicMock()


class TestCaptionHealerCanHandle:
    """Tests for can_handle() method."""

    def test_handles_caption_format_unavailable_error(self, healer):
        """Should handle CaptionFormatUnavailableError by type."""
        error = CaptionFormatUnavailableError("abc123", "json3")
        assert healer.can_handle(error, "CAPTION") is True

    def test_handles_caption_unavailable_error(self, healer):
        """Should handle CaptionUnavailableError by type."""
        error = CaptionUnavailableError("abc123", "no captions")
        assert healer.can_handle(error, "CAPTION") is True

    def test_handles_caption_fetch_error(self, healer):
        """Should handle CaptionFetchError by type."""
        error = CaptionFetchError("abc123", "network error")
        assert healer.can_handle(error, "CAPTION") is True

    def test_handles_no_captions_pattern(self, healer):
        """Should handle errors with 'no captions' in message."""
        error = Exception("There are no subtitles for this video")
        assert healer.can_handle(error, "CAPTION") is True

    def test_handles_format_unavailable_pattern(self, healer):
        """Should handle format unavailable errors."""
        error = Exception("Requested format 'json3' is not available")
        assert healer.can_handle(error, "CAPTION") is True

    def test_handles_subtitle_pattern(self, healer):
        """Should handle errors mentioning subtitles."""
        error = Exception("Unable to download video subtitles for abc123")
        assert healer.can_handle(error, "CAPTION") is True

    def test_does_not_handle_unrelated_error(self, healer):
        """Should not handle unrelated errors."""
        error = Exception("Disk full, cannot write file")
        assert healer.can_handle(error, "DOWNLOAD") is False


class TestCaptionHealerFormatDiscovery:
    """Tests for format discovery using --list-subs."""

    def test_parse_list_subs_manual_captions(self, healer):
        """Should parse manual subtitle formats correctly."""
        stdout = """
Available subtitles for dQw4w9WgXcQ:
Language  formats
en        json3, srv1, srv2, srv3, ttml, vtt
es        json3, srv1, srv2, srv3, ttml, vtt   Spanish
"""
        result = healer._parse_list_subs_output(stdout, "")

        assert len(result) > 0
        en_json3 = [s for s in result if s.language == "en" and s.format == "json3"]
        assert len(en_json3) == 1
        assert en_json3[0].is_auto_generated is False

    def test_parse_list_subs_auto_captions(self, healer):
        """Should parse auto-generated captions correctly."""
        stdout = """
Available automatic captions for dQw4w9WgXcQ:
Language  formats
en        json3, srv1, srv2, srv3, ttml, vtt
"""
        result = healer._parse_list_subs_output(stdout, "")

        assert len(result) > 0
        en_json3 = [s for s in result if s.language == "en" and s.format == "json3"]
        assert len(en_json3) == 1
        assert en_json3[0].is_auto_generated is True

    def test_parse_list_subs_mixed(self, healer):
        """Should correctly distinguish manual and auto captions."""
        stdout = """
Available subtitles for dQw4w9WgXcQ:
Language  formats
en        json3, srv1, srv2, srv3, ttml, vtt

Available automatic captions for dQw4w9WgXcQ:
Language  formats
de        json3, srv1, srv2, srv3, ttml, vtt
"""
        result = healer._parse_list_subs_output(stdout, "")

        manual = [s for s in result if not s.is_auto_generated]
        auto = [s for s in result if s.is_auto_generated]

        assert len(manual) > 0
        assert len(auto) > 0
        assert all(s.language == "en" for s in manual if s.format == "json3")
        assert all(s.language == "de" for s in auto if s.format == "json3")


class TestCaptionHealerFormatSelection:
    """Tests for format selection logic."""

    def test_prefers_json3_format(self, healer):
        """Should prefer json3 format when available."""
        available = [
            SubtitleInfo("en", "vtt", False),
            SubtitleInfo("en", "json3", False),
            SubtitleInfo("en", "srv1", False),
        ]
        best = healer._select_best_format(available)
        assert best.format == "json3"

    def test_prefers_manual_over_auto(self, healer):
        """Should prefer manual captions over auto-generated."""
        available = [
            SubtitleInfo("en", "json3", True),  # auto
            SubtitleInfo("en", "srv1", False),  # manual
        ]
        best = healer._select_best_format(available)
        assert best.is_auto_generated is False

    def test_prefers_english_variants(self, healer):
        """Should prefer English language variants."""
        available = [
            SubtitleInfo("de", "json3", False),
            SubtitleInfo("en-US", "json3", False),
            SubtitleInfo("fr", "json3", False),
        ]
        best = healer._select_best_format(available)
        assert best.language == "en-US"

    def test_format_preference_order(self, healer):
        """Should follow format preference order: json3 > srv3 > srv2 > srv1 > vtt > ttml."""
        available = [
            SubtitleInfo("en", "ttml", False),
            SubtitleInfo("en", "srv2", False),
            SubtitleInfo("en", "vtt", False),
        ]
        best = healer._select_best_format(available)
        assert best.format == "srv2"


class TestCaptionHealerFix:
    """Tests for fix() method."""

    @patch.object(CaptionHealer, '_discover_formats')
    def test_fix_with_available_formats(self, mock_discover, healer, mock_state):
        """Should return config_changed when formats are discovered."""
        mock_discover.return_value = [
            SubtitleInfo("en", "json3", False),
            SubtitleInfo("en", "srv1", False),
        ]
        error = CaptionFormatUnavailableError("abc123", "json3")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert result.details.get("discovered_format") == "json3"

    @patch.object(CaptionHealer, '_discover_formats')
    def test_fix_enables_auto_subs_when_no_manual(self, mock_discover, healer, mock_state):
        """Should enable auto-subs when no manual captions available."""
        mock_discover.return_value = []  # No captions found
        error = CaptionUnavailableError("abc123")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert result.details.get("auto_subs_enabled") is True

    @patch.object(CaptionHealer, '_discover_formats')
    def test_fix_skips_video_when_no_captions(self, mock_discover, healer, mock_state):
        """Should skip video when no captions even with auto-sub enabled."""
        mock_discover.return_value = []
        healer._auto_caption_enabled = True  # Already tried auto-subs
        error = CaptionUnavailableError("abc123")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is True
        assert result.action == HealerAction.SKIP
        assert "abc123" in healer._no_captions_videos

    def test_fix_fails_without_video_id(self, healer, mock_state):
        """Should fail when video ID cannot be extracted."""
        error = Exception("Some generic error without video ID")

        result = healer.fix(error, mock_state, "CAPTION")

        assert result.success is False


class TestCaptionHealerVideoIdExtraction:
    """Tests for video ID extraction from errors."""

    def test_extract_from_error_attribute(self, healer):
        """Should extract video_id from exception attribute."""
        error = CaptionFetchError("dQw4w9WgXcQ", "test")
        video_id = healer._extract_video_id(str(error), error)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_from_url_in_message(self, healer):
        """Should extract video_id from YouTube URL in message."""
        error = Exception("Failed for https://youtube.com/watch?v=dQw4w9WgXcQ")
        video_id = healer._extract_video_id(str(error), error)
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_from_quoted_id(self, healer):
        """Should extract video_id from quoted string."""
        error = Exception("Video 'dQw4w9WgXcQ' has no captions")
        video_id = healer._extract_video_id(str(error), error)
        assert video_id == "dQw4w9WgXcQ"


class TestCaptionHealerCaching:
    """Tests for format discovery caching."""

    def test_caches_discovered_formats(self, healer, mock_state):
        """Should cache discovered formats for reuse."""
        # Manually populate the cache to test get_discovered_formats
        test_formats = [SubtitleInfo("en", "json3", False)]
        healer._format_cache["abc123"] = test_formats

        # Check cache retrieval
        cached = healer.get_discovered_formats("abc123")
        assert len(cached) == 1
        assert cached[0].format == "json3"

    @patch.object(CaptionHealer, '_discover_formats')
    def test_fix_populates_cache(self, mock_discover, healer, mock_state):
        """Fix should use discovered formats (caching happens in _discover_formats)."""
        mock_discover.return_value = [SubtitleInfo("en", "json3", False)]
        error = CaptionFormatUnavailableError("abc123", "json3")

        # Call fix
        result = healer.fix(error, mock_state, "CAPTION")

        # Verify _discover_formats was called
        mock_discover.assert_called_once_with("abc123")
        # Verify result uses discovered format
        assert result.details.get("discovered_format") == "json3"

    def test_reset_discovery_cache(self, healer):
        """Should clear cache when reset_discovery_cache() called."""
        healer._format_cache["abc123"] = [SubtitleInfo("en", "json3", False)]
        healer._no_captions_videos.add("xyz789")
        healer._auto_caption_enabled = True

        healer.reset_discovery_cache()

        assert len(healer._format_cache) == 0
        assert len(healer._no_captions_videos) == 0
        assert healer._auto_caption_enabled is False


class TestCaptionHealerRegistry:
    """Tests for healer registry integration."""

    def test_healer_in_registry(self):
        """Should be registered in HEALER_REGISTRY."""
        from src.agents.healers import HEALER_REGISTRY, CaptionHealer as RegisteredHealer

        # HEALER_REGISTRY contains classes, not instances
        assert RegisteredHealer in HEALER_REGISTRY

    def test_healer_in_exports(self):
        """Should be exported from healers package."""
        from src.agents.healers import CaptionHealer
        assert CaptionHealer is not None


class TestPatternRouting:
    """Tests for pattern routing integration."""

    def test_caption_patterns_in_routing(self):
        """Caption patterns should be in PATTERN_ROUTING."""
        from src.agents.fallback import PATTERN_ROUTING

        # Check that at least one caption pattern exists
        caption_patterns = [
            pattern for pattern, (category, _) in PATTERN_ROUTING.items()
            if category == "caption"
        ]
        assert len(caption_patterns) > 0

    def test_pattern_route_caption_error(self):
        """pattern_route should route caption errors correctly."""
        from src.agents.fallback import pattern_route

        result = pattern_route("caption unavailable for video abc123")
        assert result.category == "caption"
        assert result.suggested_healer == "caption-healer"

    def test_pattern_route_format_error(self):
        """pattern_route should route format unavailable errors."""
        from src.agents.fallback import pattern_route

        result = pattern_route("format unavailable: json3 not found")
        assert result.category == "caption"

    def test_pattern_route_no_subtitles(self):
        """pattern_route should route 'no subtitles' errors."""
        from src.agents.fallback import pattern_route

        result = pattern_route("there are no subtitles for this video")
        assert result.category == "caption"


# US-64-003: Tests for negative cache integration

class TestNoCaptionsStatus:
    """Tests for NoCaptionsStatus dataclass (US-64-003)."""

    def test_create_no_captions_status(self):
        """Should create NoCaptionsStatus with default values."""
        status = NoCaptionsStatus(video_id="abc123")
        assert status.video_id == "abc123"
        assert status.language == "en"
        assert status.reason == "no_captions_available"
        assert status.cached is False
        assert status.ttl_seconds == 3600

    def test_to_dict_conversion(self):
        """Should convert to dictionary correctly."""
        status = NoCaptionsStatus(
            video_id="abc123",
            language="es",
            reason="negative_cache_hit",
            cached=True,
            cached_at=1000.0,
            ttl_seconds=7200,
            auto_subs_attempted=True,
        )
        d = status.to_dict()
        assert d["video_id"] == "abc123"
        assert d["language"] == "es"
        assert d["cached"] is True
        assert d["ttl_seconds"] == 7200

    def test_from_dict_conversion(self):
        """Should create from dictionary correctly."""
        d = {
            "video_id": "xyz789",
            "language": "de",
            "reason": "test",
            "cached": True,
            "cached_at": 2000.0,
            "ttl_seconds": 1800,
            "auto_subs_attempted": False,
            "discovered_formats": ["json3"],
        }
        status = NoCaptionsStatus.from_dict(d)
        assert status.video_id == "xyz789"
        assert status.language == "de"
        assert status.ttl_seconds == 1800
        assert status.discovered_formats == ["json3"]

    def test_is_expired_fresh(self):
        """Fresh status should not be expired."""
        status = NoCaptionsStatus(
            video_id="abc123",
            cached=True,
            cached_at=time.time(),
            ttl_seconds=3600,
        )
        assert status.is_expired is False

    def test_is_expired_old(self):
        """Old status should be expired."""
        status = NoCaptionsStatus(
            video_id="abc123",
            cached=True,
            cached_at=time.time() - 7200,  # 2 hours ago
            ttl_seconds=3600,  # 1 hour TTL
        )
        assert status.is_expired is True

    def test_is_expired_not_cached(self):
        """Non-cached status should not be expired."""
        status = NoCaptionsStatus(video_id="abc123", cached=False)
        assert status.is_expired is False


@pytest.mark.fast
class TestCaptionHealerNegativeCache:
    """Tests for CaptionHealer negative cache integration (US-64-003)."""

    def _make_cache(self, ttl_seconds=3600, validation_mode='warn'):
        """Create a CaptionCache with temp directory."""
        from src.config.sections.download import CaptionFirstConfig
        from src.caption.cache import CaptionCache

        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_seconds=ttl_seconds,
            cache_validation=validation_mode,
            max_cache_age_days=30,
        )
        return CaptionCache(config)

    def _make_healer_with_cache(self, cache=None, ttl_seconds=3600):
        """Create a CaptionHealer with optional negative cache."""
        config = MagicMock()
        config.download = MagicMock()
        config.download.caption_first = MagicMock()
        config.download.caption_first.negative_cache_ttl_seconds = ttl_seconds
        config.download.caption_first.negative_cache_ttl_hours = ttl_seconds / 3600
        config.download.caption_first.enable_auto_subs = False
        return CaptionHealer(config, "/test/project", caption_cache=cache)

    def test_negative_cache_prevents_repeated_failures(self):
        """AC: Negative cache check before attempting caption fetch prevents repeated failures.

        When a video is known to have no captions, fix() should return SKIP immediately
        without making any subprocess calls to yt-dlp.
        """
        cache = self._make_cache()
        healer = self._make_healer_with_cache(cache)

        # Store video as unavailable in cache
        cache.store_unavailable("no_caps_vid1", "en")

        # Create an error for this video
        error = CaptionUnavailableError("no_caps_vid1", "test error")
        mock_state = MagicMock()

        # Mock _discover_formats to verify it's NOT called
        with patch.object(healer, '_discover_formats') as mock_discover:
            result = healer.fix(error, mock_state, "CAPTION")

            # Should skip immediately due to negative cache hit
            assert result.success is True
            assert result.action == HealerAction.SKIP

            # Should NOT have called _discover_formats (no yt-dlp subprocess)
            mock_discover.assert_not_called()

            # Result should indicate cache hit
            assert result.details.get("cached") is True
            assert "no_captions_status" in result.details

    def test_negative_cache_stores_on_no_captions(self):
        """AC: When no captions found, store result in negative cache."""
        cache = self._make_cache()
        healer = self._make_healer_with_cache(cache)

        # Enable auto-subs (so we go past the first attempt)
        healer._auto_caption_enabled = True

        error = CaptionUnavailableError("store_test_vid", "test")
        mock_state = MagicMock()

        # Mock _discover_formats to return empty (no captions)
        with patch.object(healer, '_discover_formats', return_value=[]):
            result = healer.fix(error, mock_state, "CAPTION")

        # Should skip
        assert result.success is True
        assert result.action == HealerAction.SKIP

        # Should have stored in negative cache
        assert cache.is_caption_unavailable("store_test_vid", "en") is True

        # Result should include no_captions_status
        assert result.details.get("no_captions_status") is not None
        status_dict = result.details["no_captions_status"]
        assert status_dict["video_id"] == "store_test_vid"
        assert status_dict["cached"] is True

    def test_set_caption_cache_after_construction(self):
        """set_caption_cache() should allow setting cache after construction."""
        healer = self._make_healer_with_cache(cache=None)
        assert healer._caption_cache is None

        cache = self._make_cache()
        healer.set_caption_cache(cache)

        assert healer._caption_cache is cache

    def test_get_no_captions_status_from_cache(self):
        """get_no_captions_status() should return status from negative cache."""
        cache = self._make_cache()
        healer = self._make_healer_with_cache(cache)

        # Initially no status
        assert healer.get_no_captions_status("unknown_vid") is None

        # Store in cache
        cache.store_unavailable("known_vid", "en")

        # Now should return status
        status = healer.get_no_captions_status("known_vid", "en")
        assert status is not None
        assert status.video_id == "known_vid"
        assert status.cached is True

    def test_get_no_captions_status_from_session(self):
        """get_no_captions_status() should return status from session memory."""
        healer = self._make_healer_with_cache(cache=None)

        # Add to in-memory tracking
        healer._no_captions_videos.add("session_vid")

        status = healer.get_no_captions_status("session_vid")
        assert status is not None
        assert status.video_id == "session_vid"
        assert status.cached is False
        assert status.reason == "session_cache_hit"

    def test_negative_cache_ttl_from_config(self):
        """TTL should be read from config (US-64-003)."""
        cache = self._make_cache(ttl_seconds=7200)  # 2 hours
        healer = self._make_healer_with_cache(cache, ttl_seconds=7200)

        assert healer._negative_cache_ttl_seconds == 7200

    def test_format_cache_age_helper(self):
        """_format_cache_age() should format age correctly."""
        healer = self._make_healer_with_cache(cache=None)

        # Test various ages
        assert healer._format_cache_age(None) == "unknown"
        assert "s" in healer._format_cache_age(time.time() - 30)  # 30 seconds
        assert "m" in healer._format_cache_age(time.time() - 300)  # 5 minutes
        assert "h" in healer._format_cache_age(time.time() - 7200)  # 2 hours

    def test_healer_without_cache_works_normally(self):
        """Healer without cache should work without errors."""
        healer = self._make_healer_with_cache(cache=None)

        # Store video in session memory
        healer._no_captions_videos.add("session_vid")
        healer._auto_caption_enabled = True

        error = CaptionUnavailableError("session_vid", "test")
        mock_state = MagicMock()

        result = healer.fix(error, mock_state, "CAPTION")

        # Should skip (from session memory)
        assert result.success is True
        assert result.action == HealerAction.SKIP


@pytest.mark.fast
class TestCaptionHealerNegativeCacheTTL:
    """Tests for TTL expiration allowing retry (US-64-003)."""

    def _make_cache_with_short_ttl(self, ttl_seconds=1, validation_mode='strict'):
        """Create a CaptionCache with short TTL for testing expiration."""
        from src.config.sections.download import CaptionFirstConfig
        from src.caption.cache import CaptionCache

        config = CaptionFirstConfig(
            cache_dir=tempfile.mkdtemp(),
            negative_cache_ttl_seconds=ttl_seconds,
            cache_validation=validation_mode,
            max_cache_age_days=30,
        )
        return CaptionCache(config)

    def test_ttl_expiration_allows_retry(self):
        """AC: TTL expiration should allow retry after configured interval.

        When the negative cache entry expires, the healer should attempt
        format discovery again instead of returning cached status.
        """
        # Use 1 second TTL and strict mode
        cache = self._make_cache_with_short_ttl(ttl_seconds=1, validation_mode='strict')

        config = MagicMock()
        config.download = MagicMock()
        config.download.caption_first = MagicMock()
        config.download.caption_first.negative_cache_ttl_seconds = 1
        config.download.caption_first.enable_auto_subs = False
        healer = CaptionHealer(config, "/test/project", caption_cache=cache)

        # Store in cache
        cache.store_unavailable("ttl_test_vid", "en")

        # Verify it's cached
        assert cache.is_caption_unavailable("ttl_test_vid", "en") is True

        # Wait for TTL to expire
        time.sleep(1.5)

        # Now cache should return False (expired in strict mode)
        assert cache.is_caption_unavailable("ttl_test_vid", "en") is False

        # fix() should attempt discovery again
        error = CaptionUnavailableError("ttl_test_vid", "test")
        mock_state = MagicMock()

        with patch.object(healer, '_discover_formats', return_value=[]) as mock_discover:
            # Enable auto-subs to get to the final skip
            healer._auto_caption_enabled = True
            result = healer.fix(error, mock_state, "CAPTION")

            # Should have called _discover_formats (TTL expired)
            mock_discover.assert_called_once()

    def test_fresh_cache_prevents_retry(self):
        """Fresh cache entry should prevent retry."""
        cache = self._make_cache_with_short_ttl(ttl_seconds=3600)  # 1 hour

        config = MagicMock()
        config.download = MagicMock()
        config.download.caption_first = MagicMock()
        config.download.caption_first.negative_cache_ttl_seconds = 3600
        config.download.caption_first.enable_auto_subs = False
        healer = CaptionHealer(config, "/test/project", caption_cache=cache)

        # Store in cache
        cache.store_unavailable("fresh_test_vid", "en")

        error = CaptionUnavailableError("fresh_test_vid", "test")
        mock_state = MagicMock()

        with patch.object(healer, '_discover_formats') as mock_discover:
            result = healer.fix(error, mock_state, "CAPTION")

            # Should NOT call _discover_formats (cache hit)
            mock_discover.assert_not_called()

            # Should skip
            assert result.action == HealerAction.SKIP


@pytest.mark.fast
class TestNoCaptionsStatusExports:
    """Tests for NoCaptionsStatus exports (US-64-003)."""

    def test_no_captions_status_exported(self):
        """NoCaptionsStatus should be exported from healers package."""
        from src.agents.healers import NoCaptionsStatus
        assert NoCaptionsStatus is not None

    def test_subtitle_info_exported(self):
        """SubtitleInfo should be exported from healers package."""
        from src.agents.healers import SubtitleInfo
        assert SubtitleInfo is not None
