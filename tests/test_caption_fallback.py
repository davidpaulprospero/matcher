"""Tests for caption fallback system.

Tests multi-tier caption extraction:
- Tier 2: youtube-transcript-api
- Tier 3: Direct Innertube/timedtext
- Tier 4: Invidious API
- Tier 5: Piped API
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.downloader.caption_fallback import (
    CaptionFallbackChain,
    TranscriptAPIFetcher,
    InnertubeDirectFetcher,
    InvidiousCaptionFetcher,
    PipedCaptionFetcher,
    CaptionTier,
    CaptionSegment,
    CaptionResult,
    fetch_captions,
)


# Known video with captions for integration tests
TEST_VIDEO_ID = "dQw4w9WgXcQ"


class TestCaptionSegment:
    """Tests for CaptionSegment dataclass."""

    def test_end_property(self):
        """Test end time calculation."""
        seg = CaptionSegment("Hello", 10.0, 2.5)
        assert seg.end == 12.5

    def test_to_dict(self):
        """Test serialization."""
        seg = CaptionSegment("Hello world", 5.0, 3.0)
        d = seg.to_dict()
        assert d == {"text": "Hello world", "start": 5.0, "duration": 3.0}

    def test_from_dict(self):
        """Test deserialization."""
        d = {"text": "Test", "start": 1.0, "duration": 2.0}
        seg = CaptionSegment.from_dict(d)
        assert seg.text == "Test"
        assert seg.start == 1.0
        assert seg.duration == 2.0


class TestCaptionResult:
    """Tests for CaptionResult dataclass."""

    def test_text_property(self):
        """Test full text extraction."""
        result = CaptionResult(
            success=True,
            tier_used=CaptionTier.TRANSCRIPT_API,
            segments=[
                CaptionSegment("Hello", 0.0, 1.0),
                CaptionSegment("world", 1.0, 1.0),
            ],
        )
        assert result.text == "Hello world"

    def test_duration_property(self):
        """Test duration calculation."""
        result = CaptionResult(
            success=True,
            tier_used=CaptionTier.TRANSCRIPT_API,
            segments=[
                CaptionSegment("First", 0.0, 2.0),
                CaptionSegment("Last", 5.0, 3.0),
            ],
        )
        assert result.duration == 8.0  # 5.0 + 3.0

    def test_duration_empty(self):
        """Test duration with no segments."""
        result = CaptionResult(
            success=False,
            tier_used=CaptionTier.TRANSCRIPT_API,
            segments=[],
        )
        assert result.duration == 0.0


class TestTranscriptAPIFetcher:
    """Tests for youtube-transcript-api wrapper."""

    def test_api_not_installed(self):
        """Test handling when youtube-transcript-api is not installed."""
        fetcher = TranscriptAPIFetcher()
        fetcher._api = False  # Simulate not installed

        result = fetcher.fetch("test_video")

        assert not result.success
        assert "not installed" in result.error

    @patch("src.downloader.caption_fallback.TranscriptAPIFetcher.api")
    def test_fetch_manual_transcript(self, mock_api):
        """Test fetching manual transcript."""
        # Mock transcript list
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0},
        ]

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.return_value = (
            mock_transcript
        )

        mock_api.list_transcripts.return_value = mock_transcript_list

        fetcher = TranscriptAPIFetcher(["en"])
        result = fetcher.fetch("test_video")

        assert result.success
        assert result.tier_used == CaptionTier.TRANSCRIPT_API
        assert len(result.segments) == 2
        assert result.is_auto_generated is False

    @patch("src.downloader.caption_fallback.TranscriptAPIFetcher.api")
    def test_fetch_auto_transcript(self, mock_api):
        """Test falling back to auto-generated transcript."""
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [
            {"text": "Auto text", "start": 0.0, "duration": 2.0}
        ]

        mock_transcript_list = MagicMock()
        mock_transcript_list.find_manually_created_transcript.side_effect = Exception(
            "Not found"
        )
        mock_transcript_list.find_generated_transcript.return_value = mock_transcript

        mock_api.list_transcripts.return_value = mock_transcript_list

        fetcher = TranscriptAPIFetcher(["en"])
        result = fetcher.fetch("test_video")

        assert result.success
        assert result.is_auto_generated is True


class TestInnertubeDirectFetcher:
    """Tests for direct Innertube/timedtext extraction."""

    def test_extract_caption_url_from_tracks(self):
        """Test extraction from captionTracks JSON."""
        fetcher = InnertubeDirectFetcher()
        html = '''
        {"captionTracks":[{"baseUrl":"https://www.youtube.com/api/timedtext?v=abc","languageCode":"en"}]}
        '''
        url, lang = fetcher._extract_caption_url(html)
        assert url == "https://www.youtube.com/api/timedtext?v=abc"
        assert lang == "en"

    def test_extract_caption_url_prefers_english(self):
        """Test that English captions are preferred."""
        fetcher = InnertubeDirectFetcher()
        html = '''
        {"captionTracks":[
            {"baseUrl":"https://example.com/es","languageCode":"es"},
            {"baseUrl":"https://example.com/en","languageCode":"en"}
        ]}
        '''
        url, lang = fetcher._extract_caption_url(html)
        assert "en" in url
        assert lang == "en"

    def test_parse_srv3_format(self):
        """Test parsing srv3 XML format."""
        fetcher = InnertubeDirectFetcher()
        content = '''
        <text start="0.5" dur="2.0">Hello world</text>
        <text start="3.0" dur="1.5">Test text</text>
        '''
        segments = fetcher._parse_captions(content, "srv3")
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[0].start == 0.5
        assert segments[0].duration == 2.0

    def test_parse_json3_format(self):
        """Test parsing json3 format."""
        fetcher = InnertubeDirectFetcher()
        content = '{"events":[{"segs":[{"utf8":"Test"}],"tStartMs":1000,"dDurationMs":2000}]}'
        segments = fetcher._parse_captions(content, "json3")
        assert len(segments) == 1
        assert segments[0].text == "Test"
        assert segments[0].start == 1.0
        assert segments[0].duration == 2.0

    def test_vtt_to_seconds(self):
        """Test VTT timestamp conversion."""
        assert InnertubeDirectFetcher._vtt_to_seconds("00:01:30.500") == 90.5
        assert InnertubeDirectFetcher._vtt_to_seconds("01:00:00.000") == 3600.0
        assert InnertubeDirectFetcher._vtt_to_seconds("00:00:05,123") == 5.123


class TestInvidiousCaptionFetcher:
    """Tests for Invidious API caption fetching."""

    def test_instance_rotation(self):
        """Test that instances are rotated."""
        fetcher = InvidiousCaptionFetcher(
            instances=["https://a.com", "https://b.com", "https://c.com"]
        )

        # Simulate failures
        instances_tried = []
        for _ in range(3):
            inst = fetcher._get_next_instance()
            if inst:
                instances_tried.append(inst)

        # Should have rotated through instances
        assert len(set(instances_tried)) == 3

    def test_cooldown_respects_timeout(self):
        """Test that failed instances are skipped during cooldown."""
        import time

        fetcher = InvidiousCaptionFetcher(
            instances=["https://a.com", "https://b.com"],
            cooldown_seconds=1.0,
        )

        # Mark first instance as failed
        fetcher.failed_instances["https://a.com"] = time.time()

        # Should skip failed instance
        inst = fetcher._get_next_instance()
        assert inst == "https://b.com"

    def test_parse_xml_captions(self):
        """Test parsing XML captions from Invidious."""
        fetcher = InvidiousCaptionFetcher()
        content = '<text start="1.0" dur="2.0">Caption text</text>'
        segments = fetcher._parse_captions(content)
        assert len(segments) == 1
        assert segments[0].text == "Caption text"


class TestPipedCaptionFetcher:
    """Tests for Piped API caption fetching."""

    def test_instance_rotation(self):
        """Test instance rotation."""
        fetcher = PipedCaptionFetcher(
            instances=["https://a.com", "https://b.com"]
        )

        instances = list(fetcher._rotate_instances())
        assert len(instances) == 2
        assert "https://a.com" in instances
        assert "https://b.com" in instances

    def test_parse_vtt(self):
        """Test VTT parsing."""
        fetcher = PipedCaptionFetcher()
        content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:04.000 --> 00:00:06.000
Test text
"""
        segments = fetcher._parse_vtt(content)
        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[1].text == "Test text"


class TestCaptionFallbackChain:
    """Tests for the full fallback chain."""

    def test_initialization(self):
        """Test chain initialization."""
        chain = CaptionFallbackChain()
        assert chain.transcript_api is not None
        assert chain.innertube_direct is not None
        assert chain.invidious is not None
        assert chain.piped is not None

    def test_to_srt_format(self):
        """Test SRT conversion."""
        chain = CaptionFallbackChain()
        segments = [
            CaptionSegment("Hello world", 0.0, 2.5),
            CaptionSegment("Testing captions", 2.5, 3.0),
        ]

        srt = chain.to_srt(segments)

        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "Hello world" in srt
        assert "2\n" in srt
        assert "Testing captions" in srt

    def test_to_vtt_format(self):
        """Test WebVTT conversion."""
        chain = CaptionFallbackChain()
        segments = [
            CaptionSegment("Hello world", 0.0, 2.5),
        ]

        vtt = chain.to_vtt(segments)

        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:02.500" in vtt
        assert "Hello world" in vtt

    def test_to_transcript_list(self):
        """Test conversion to transcript list format."""
        chain = CaptionFallbackChain()
        segments = [
            CaptionSegment("Test", 1.0, 2.0),
        ]

        result = chain.to_transcript_list(segments)

        assert result == [{"text": "Test", "start": 1.0, "duration": 2.0}]

    def test_stats_tracking(self):
        """Test that tier statistics are tracked."""
        chain = CaptionFallbackChain()

        # Mock a successful fetch
        chain.tier_stats[CaptionTier.TRANSCRIPT_API]["success"] = 5
        chain.tier_stats[CaptionTier.INVIDIOUS]["failure"] = 2

        stats = chain.get_stats()

        assert stats["TRANSCRIPT_API"]["success"] == 5
        assert stats["INVIDIOUS"]["failure"] == 2

    def test_reset_stats(self):
        """Test stats reset."""
        chain = CaptionFallbackChain()
        chain.tier_stats[CaptionTier.TRANSCRIPT_API]["success"] = 10

        chain.reset_stats()

        assert chain.tier_stats[CaptionTier.TRANSCRIPT_API]["success"] == 0

    def test_skip_tiers(self):
        """Test skipping specific tiers."""
        chain = CaptionFallbackChain()

        # Mock all fetchers to track calls
        chain.transcript_api = MagicMock()
        chain.transcript_api.fetch.return_value = CaptionResult(
            success=False, tier_used=CaptionTier.TRANSCRIPT_API
        )

        chain.innertube_direct = MagicMock()
        chain.innertube_direct.fetch.return_value = CaptionResult(
            success=True,
            tier_used=CaptionTier.INNERTUBE_DIRECT,
            segments=[CaptionSegment("test", 0, 1)],
        )

        # Skip TRANSCRIPT_API tier
        result = chain.fetch("test", skip_tiers=[CaptionTier.TRANSCRIPT_API])

        # Should not have called transcript_api
        chain.transcript_api.fetch.assert_not_called()
        # Should have called innertube_direct
        chain.innertube_direct.fetch.assert_called_once()

    def test_seconds_to_srt(self):
        """Test SRT timestamp formatting."""
        assert CaptionFallbackChain._seconds_to_srt(0) == "00:00:00,000"
        assert CaptionFallbackChain._seconds_to_srt(65.5) == "00:01:05,500"
        assert CaptionFallbackChain._seconds_to_srt(3661.123) == "01:01:01,123"

    def test_seconds_to_vtt(self):
        """Test VTT timestamp formatting."""
        assert CaptionFallbackChain._seconds_to_vtt(0) == "00:00:00.000"
        assert CaptionFallbackChain._seconds_to_vtt(65.5) == "00:01:05.500"


class TestConvenienceFunction:
    """Tests for fetch_captions convenience function."""

    @patch.object(CaptionFallbackChain, "fetch")
    def test_fetch_captions(self, mock_fetch):
        """Test convenience function creates chain and calls fetch."""
        mock_fetch.return_value = CaptionResult(
            success=True,
            tier_used=CaptionTier.TRANSCRIPT_API,
            segments=[CaptionSegment("test", 0, 1)],
        )

        result = fetch_captions("test_video", ["en"])

        assert result.success
        mock_fetch.assert_called_once()


# Integration tests (require network access)
@pytest.mark.integration
class TestCaptionFallbackIntegration:
    """Integration tests for caption fallback (require network)."""

    def test_fetch_real_captions(self):
        """Test fetching real captions from YouTube."""
        chain = CaptionFallbackChain()
        result = chain.fetch(TEST_VIDEO_ID)

        # Should succeed via at least one tier
        assert result.success
        assert len(result.segments) > 0
        assert result.tier_used in [
            CaptionTier.TRANSCRIPT_API,
            CaptionTier.INNERTUBE_DIRECT,
            CaptionTier.INVIDIOUS,
            CaptionTier.PIPED,
        ]

    def test_transcript_api_real(self):
        """Test youtube-transcript-api with real video."""
        fetcher = TranscriptAPIFetcher(["en"])
        result = fetcher.fetch(TEST_VIDEO_ID)

        # Should succeed if youtube-transcript-api is installed
        if result.success:
            assert len(result.segments) > 0
            assert all(isinstance(s, CaptionSegment) for s in result.segments)


# =============================================================================
# Proxy Integration Tests
# =============================================================================


class TestInnertubeProxyIntegration:
    """Tests for InnertubeDirectFetcher proxy support."""

    def test_accepts_rate_limit_handler(self):
        """Should accept rate_limit_handler parameter."""
        handler = Mock(has_proxy=False)
        fetcher = InnertubeDirectFetcher(rate_limit_handler=handler)
        assert fetcher.handler is handler

    def test_creates_client_with_proxy(self):
        """Should create httpx client with proxy from handler."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        with patch('src.downloader.caption_fallback.create_httpx_client') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            fetcher = InnertubeDirectFetcher(rate_limit_handler=handler)
            _ = fetcher.client  # Trigger lazy initialization

            mock_create.assert_called_once_with(
                handler=handler,
                timeout=30.0,
                follow_redirects=True,
            )

    def test_calls_on_rate_limit_on_429(self):
        """Should call handler.on_rate_limit when 429 received."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = None

        with patch('src.downloader.caption_fallback.create_httpx_client') as mock_create:
            mock_client = Mock()
            mock_response = Mock(status_code=429, text="Rate limited")
            mock_client.get.return_value = mock_response
            mock_create.return_value = mock_client

            fetcher = InnertubeDirectFetcher(rate_limit_handler=handler)
            result = fetcher.fetch("test_video")

            assert not result.success
            assert "429" in result.error
            handler.on_rate_limit.assert_called_once_with("innertube_page")

    def test_calls_on_success_on_200(self):
        """Should call handler.on_success on successful page fetch."""
        handler = Mock(has_proxy=False)

        with patch('src.downloader.caption_fallback.create_httpx_client') as mock_create:
            mock_client = Mock()
            # Return valid page but no captions
            mock_response = Mock(status_code=200, text='<html>no captions</html>')
            mock_client.get.return_value = mock_response
            mock_create.return_value = mock_client

            fetcher = InnertubeDirectFetcher(rate_limit_handler=handler)
            fetcher.fetch("test_video")

            handler.on_success.assert_called_once()


class TestInvidiousProxyIntegration:
    """Tests for InvidiousCaptionFetcher proxy support."""

    def test_accepts_rate_limit_handler(self):
        """Should accept rate_limit_handler parameter."""
        handler = Mock(has_proxy=False)
        fetcher = InvidiousCaptionFetcher(rate_limit_handler=handler)
        assert fetcher.handler is handler

    def test_stores_proxy_from_handler(self):
        """Should get proxy from handler on init."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "http://proxy.example.com:8080"

        with patch('src.downloader.caption_fallback.get_proxy_for_httpx') as mock_get:
            mock_get.return_value = "http://proxy.example.com:8080"
            fetcher = InvidiousCaptionFetcher(rate_limit_handler=handler)
            assert fetcher._proxy == "http://proxy.example.com:8080"

    def test_passes_proxy_to_httpx_get(self):
        """Should pass proxy to httpx.get calls."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "http://proxy.example.com:8080"

        with patch('src.downloader.caption_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.return_value = "http://proxy.example.com:8080"
            with patch('src.downloader.caption_fallback.httpx.get') as mock_get:
                mock_response = Mock(status_code=404, text="Not found")
                mock_get.return_value = mock_response

                fetcher = InvidiousCaptionFetcher(
                    instances=["https://test.invidious.io"],
                    rate_limit_handler=handler,
                )
                fetcher.fetch("test_video")

                # Should have called httpx.get with proxy parameter
                mock_get.assert_called()
                call_kwargs = mock_get.call_args.kwargs
                assert call_kwargs.get("proxy") == "http://proxy.example.com:8080"

    def test_rotates_proxy_on_429(self):
        """Should refresh proxy after 429 rate limit."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.side_effect = ["proxy1", "proxy2"]

        with patch('src.downloader.caption_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.side_effect = ["proxy1", "proxy2"]
            with patch('src.downloader.caption_fallback.httpx.get') as mock_get:
                mock_get.return_value = Mock(status_code=429, text="Rate limited")

                fetcher = InvidiousCaptionFetcher(
                    instances=["https://a.invidious.io", "https://b.invidious.io"],
                    rate_limit_handler=handler,
                )
                fetcher.fetch("test_video")

                # Should have called on_rate_limit and refreshed proxy
                handler.on_rate_limit.assert_called()


class TestPipedProxyIntegration:
    """Tests for PipedCaptionFetcher proxy support."""

    def test_accepts_rate_limit_handler(self):
        """Should accept rate_limit_handler parameter."""
        handler = Mock(has_proxy=False)
        fetcher = PipedCaptionFetcher(rate_limit_handler=handler)
        assert fetcher.handler is handler

    def test_passes_proxy_to_httpx_get(self):
        """Should pass proxy to httpx.get calls."""
        handler = Mock(has_proxy=True)

        with patch('src.downloader.caption_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.return_value = "socks5://127.0.0.1:1080"
            with patch('src.downloader.caption_fallback.httpx.get') as mock_get:
                mock_response = Mock(status_code=404, text="Not found")
                mock_get.return_value = mock_response

                fetcher = PipedCaptionFetcher(
                    instances=["https://test.piped.io"],
                    rate_limit_handler=handler,
                )
                fetcher.fetch("test_video")

                mock_get.assert_called()
                call_kwargs = mock_get.call_args.kwargs
                assert call_kwargs.get("proxy") == "socks5://127.0.0.1:1080"


class TestCaptionFallbackChainProxyIntegration:
    """Tests for CaptionFallbackChain proxy support."""

    def test_initializes_rate_limit_handler(self):
        """Should initialize rate limit handler."""
        with patch('src.downloader.caption_fallback.get_rate_limit_handler') as mock_get:
            with patch('src.downloader.caption_fallback.get_proxy_for_httpx') as mock_proxy:
                mock_handler = Mock(has_proxy=False)
                mock_handler.get_proxy.return_value = None
                mock_get.return_value = mock_handler
                mock_proxy.return_value = None

                chain = CaptionFallbackChain()

                mock_get.assert_called_once()
                assert chain.rate_limit_handler is mock_handler

    def test_passes_handler_to_fetchers(self):
        """Should pass rate limit handler to all fetchers."""
        with patch('src.downloader.caption_fallback.get_rate_limit_handler') as mock_get:
            with patch('src.downloader.caption_fallback.get_proxy_for_httpx') as mock_proxy:
                mock_handler = Mock(has_proxy=False)
                mock_handler.get_proxy.return_value = None
                mock_get.return_value = mock_handler
                mock_proxy.return_value = None

                chain = CaptionFallbackChain()

                # Check that fetchers received the handler
                assert chain.innertube_direct.handler is mock_handler
                assert chain.invidious.handler is mock_handler
                assert chain.piped.handler is mock_handler
