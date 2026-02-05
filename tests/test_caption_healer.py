"""
Tests for CaptionHealer - intelligent caption format discovery and fallback.

US-64-002: CaptionHealer for intelligent caption format discovery
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.healers.caption import CaptionHealer, SubtitleInfo
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
