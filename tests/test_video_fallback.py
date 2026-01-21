"""Tests for video download fallback system.

Tests multi-tier video downloading:
- Tier 2: Invidious API
- Tier 3: Piped API
- Tier 4: Cobalt API (optional)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.downloader.video_fallback import (
    VideoFallbackChain,
    InvidiousStreamFetcher,
    PipedStreamFetcher,
    CobaltFetcher,
    VideoTier,
    StreamInfo,
    VideoResult,
    download_video,
)


class TestStreamInfo:
    """Tests for StreamInfo dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        stream = StreamInfo(
            url="https://example.com/video.mp4",
            quality="720p",
            format="mp4",
            filesize=100 * 1024 * 1024,  # 100MB
        )
        assert "720p" in str(stream)
        assert "mp4" in str(stream)
        assert "100MB" in str(stream)

    def test_str_no_filesize(self):
        """Test string without filesize."""
        stream = StreamInfo(
            url="https://example.com/video.mp4",
            quality="1080p",
            format="webm",
        )
        assert "1080p" in str(stream)
        assert "webm" in str(stream)


class TestInvidiousStreamFetcher:
    """Tests for Invidious stream fetcher."""

    def test_initialization(self):
        """Test default initialization."""
        fetcher = InvidiousStreamFetcher()
        assert len(fetcher.instances) > 0
        assert fetcher.timeout == 30.0

    def test_custom_instances(self):
        """Test custom instance list."""
        instances = ["https://custom1.com", "https://custom2.com"]
        fetcher = InvidiousStreamFetcher(instances=instances)
        assert fetcher.instances == instances

    def test_sort_by_quality_prefers_combined(self):
        """Test that combined streams (with audio) are preferred."""
        fetcher = InvidiousStreamFetcher()
        streams = [
            StreamInfo("url1", "720p", "mp4", has_audio=False),
            StreamInfo("url2", "720p", "mp4", has_audio=True),
            StreamInfo("url3", "1080p", "mp4", has_audio=False),
        ]

        sorted_streams = fetcher._sort_by_quality(streams, "720p")

        # Combined stream should be first
        assert sorted_streams[0].has_audio is True

    def test_sort_by_quality_prefers_matching(self):
        """Test quality matching."""
        fetcher = InvidiousStreamFetcher()
        streams = [
            StreamInfo("url1", "1080p", "mp4", has_audio=True),
            StreamInfo("url2", "720p", "mp4", has_audio=True),
            StreamInfo("url3", "480p", "mp4", has_audio=True),
        ]

        sorted_streams = fetcher._sort_by_quality(streams, "720p")

        # 720p should be first
        assert sorted_streams[0].quality == "720p"

    def test_parse_int_safe(self):
        """Test safe integer parsing."""
        assert InvidiousStreamFetcher._parse_int(123) == 123
        assert InvidiousStreamFetcher._parse_int("456") == 456
        assert InvidiousStreamFetcher._parse_int(None) is None
        assert InvidiousStreamFetcher._parse_int("invalid") is None


class TestPipedStreamFetcher:
    """Tests for Piped stream fetcher."""

    def test_initialization(self):
        """Test default initialization."""
        fetcher = PipedStreamFetcher()
        assert len(fetcher.instances) > 0
        assert fetcher.timeout == 30.0

    def test_instance_rotation(self):
        """Test instance rotation."""
        fetcher = PipedStreamFetcher(
            instances=["https://a.com", "https://b.com", "https://c.com"]
        )

        # Collect rotated instances
        rotated = list(fetcher._rotate_instances())
        assert len(rotated) == 3

        # Index should have advanced
        assert fetcher.current_index == 1

    def test_sort_streams(self):
        """Test stream sorting."""
        fetcher = PipedStreamFetcher()
        streams = [
            StreamInfo("url1", "1080p", "mp4"),
            StreamInfo("url2", "720p audio", "m4a"),  # Audio stream
            StreamInfo("url3", "720p", "webm"),
        ]

        sorted_streams = fetcher._sort_streams(streams, "720p")

        # Audio streams should be last
        assert "audio" not in sorted_streams[0].quality.lower()


class TestCobaltFetcher:
    """Tests for Cobalt API fetcher."""

    def test_disabled_when_no_url(self):
        """Test returns None when no URL configured."""
        fetcher = CobaltFetcher(base_url=None)
        result = fetcher.get_download_url("test_video")
        assert result is None

    @patch("httpx.post")
    def test_successful_fetch(self, mock_post):
        """Test successful Cobalt API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "stream",
            "url": "https://example.com/download.mp4",
        }
        mock_post.return_value = mock_response

        fetcher = CobaltFetcher(base_url="http://localhost:9000")
        result = fetcher.get_download_url("test_video", "720")

        assert result is not None
        assert result.url == "https://example.com/download.mp4"
        assert result.quality == "720p"


class TestVideoFallbackChain:
    """Tests for the full video fallback chain."""

    def test_initialization(self):
        """Test chain initialization."""
        chain = VideoFallbackChain()
        assert chain.invidious is not None
        assert chain.piped is not None
        assert chain.cobalt is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        chain = VideoFallbackChain(
            preferred_quality="1080p",
            cobalt_url="http://localhost:9000",
        )
        assert chain.preferred_quality == "1080p"
        assert chain.cobalt.base_url == "http://localhost:9000"

    def test_stats_tracking(self):
        """Test tier statistics tracking."""
        chain = VideoFallbackChain()

        # Set some stats
        chain.tier_stats[VideoTier.INVIDIOUS]["success"] = 3
        chain.tier_stats[VideoTier.PIPED]["failure"] = 1

        stats = chain.get_stats()

        assert stats["INVIDIOUS"]["success"] == 3
        assert stats["PIPED"]["failure"] == 1

    def test_reset_stats(self):
        """Test stats reset."""
        chain = VideoFallbackChain()
        chain.tier_stats[VideoTier.INVIDIOUS]["success"] = 10

        chain.reset_stats()

        assert chain.tier_stats[VideoTier.INVIDIOUS]["success"] == 0

    @patch.object(InvidiousStreamFetcher, "get_streams")
    def test_download_with_invidious(self, mock_get_streams):
        """Test download using Invidious."""
        mock_get_streams.return_value = [
            StreamInfo(
                url="https://example.com/video.mp4",
                quality="720p",
                format="mp4",
                has_audio=True,
            )
        ]

        chain = VideoFallbackChain()

        # Mock the download method to avoid actual download
        with patch.object(chain, "_download_stream") as mock_download:
            mock_download.return_value = VideoResult(
                success=True,
                tier_used=VideoTier.INVIDIOUS,
                file_path=Path("/tmp/test.mp4"),
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                result = chain.download("test_video", Path(tmpdir))

            assert result.success
            assert result.tier_used == VideoTier.INVIDIOUS

    @patch.object(InvidiousStreamFetcher, "get_streams")
    @patch.object(PipedStreamFetcher, "get_streams")
    def test_fallback_to_piped(self, mock_piped_streams, mock_invidious_streams):
        """Test fallback from Invidious to Piped."""
        # Invidious fails
        mock_invidious_streams.return_value = []

        # Piped succeeds
        mock_piped_streams.return_value = [
            StreamInfo(
                url="https://piped.example.com/video.mp4",
                quality="720p",
                format="mp4",
                has_audio=True,
            )
        ]

        chain = VideoFallbackChain()

        with patch.object(chain, "_download_stream") as mock_download:
            mock_download.return_value = VideoResult(
                success=True,
                tier_used=VideoTier.PIPED,
                file_path=Path("/tmp/test.mp4"),
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                result = chain.download("test_video", Path(tmpdir))

            assert result.success
            assert result.tier_used == VideoTier.PIPED

    def test_skip_tiers(self):
        """Test skipping specific tiers."""
        chain = VideoFallbackChain()

        # Mock fetchers
        chain.invidious = MagicMock()
        chain.piped = MagicMock()
        chain.piped.get_streams.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            result = chain.download(
                "test_video",
                Path(tmpdir),
                skip_tiers=[VideoTier.INVIDIOUS],
            )

        # Invidious should not be called
        chain.invidious.get_streams.assert_not_called()
        # Piped should be called
        chain.piped.get_streams.assert_called_once()


class TestConvenienceFunction:
    """Tests for download_video convenience function."""

    @patch.object(VideoFallbackChain, "download")
    def test_download_video(self, mock_download):
        """Test convenience function."""
        mock_download.return_value = VideoResult(
            success=True,
            tier_used=VideoTier.INVIDIOUS,
            file_path=Path("/tmp/test.mp4"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_video("test_video", tmpdir, "720p")

        assert result.success
        mock_download.assert_called_once()


# Integration tests (require network access)
@pytest.mark.integration
class TestVideoFallbackIntegration:
    """Integration tests for video fallback (require network)."""

    def test_get_streams_invidious(self):
        """Test fetching streams from Invidious."""
        fetcher = InvidiousStreamFetcher()
        # Use a known video
        streams = fetcher.get_streams("dQw4w9WgXcQ", "720p")

        # May or may not succeed depending on instance availability
        # Just ensure no exceptions
        assert isinstance(streams, list)

    def test_get_streams_piped(self):
        """Test fetching streams from Piped."""
        fetcher = PipedStreamFetcher()
        streams = fetcher.get_streams("dQw4w9WgXcQ", "720p")

        # May or may not succeed depending on instance availability
        assert isinstance(streams, list)


# =============================================================================
# Proxy Integration Tests
# =============================================================================


class TestInvidiousStreamFetcherProxyIntegration:
    """Tests for InvidiousStreamFetcher proxy support."""

    def test_accepts_rate_limit_handler(self):
        """Should accept rate_limit_handler parameter."""
        handler = Mock(has_proxy=False)
        fetcher = InvidiousStreamFetcher(rate_limit_handler=handler)
        assert fetcher.handler is handler

    def test_stores_proxy_from_handler(self):
        """Should get proxy from handler on init."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "http://proxy.example.com:8080"

        with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_get:
            mock_get.return_value = "http://proxy.example.com:8080"
            fetcher = InvidiousStreamFetcher(rate_limit_handler=handler)
            assert fetcher._proxy == "http://proxy.example.com:8080"

    def test_passes_proxy_to_httpx_get(self):
        """Should pass proxy to httpx.get calls."""
        handler = Mock(has_proxy=True)

        with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.return_value = "socks5://127.0.0.1:1080"
            with patch('src.downloader.video_fallback.httpx.get') as mock_get:
                mock_response = Mock(status_code=404, text="Not found")
                mock_get.return_value = mock_response

                fetcher = InvidiousStreamFetcher(
                    instances=["https://test.invidious.io"],
                    rate_limit_handler=handler,
                )
                fetcher.get_streams("test_video")

                mock_get.assert_called()
                call_kwargs = mock_get.call_args.kwargs
                assert call_kwargs.get("proxy") == "socks5://127.0.0.1:1080"

    def test_calls_on_rate_limit_on_429(self):
        """Should call handler.on_rate_limit when 429 received."""
        handler = Mock(has_proxy=True)

        with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.return_value = None
            with patch('src.downloader.video_fallback.httpx.get') as mock_get:
                mock_get.return_value = Mock(status_code=429, text="Rate limited")

                fetcher = InvidiousStreamFetcher(
                    instances=["https://test.invidious.io"],
                    rate_limit_handler=handler,
                )
                fetcher.get_streams("test_video")

                handler.on_rate_limit.assert_called()


class TestPipedStreamFetcherProxyIntegration:
    """Tests for PipedStreamFetcher proxy support."""

    def test_accepts_rate_limit_handler(self):
        """Should accept rate_limit_handler parameter."""
        handler = Mock(has_proxy=False)
        fetcher = PipedStreamFetcher(rate_limit_handler=handler)
        assert fetcher.handler is handler

    def test_passes_proxy_to_httpx_get(self):
        """Should pass proxy to httpx.get calls."""
        handler = Mock(has_proxy=True)

        with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.return_value = "http://proxy:8080"
            with patch('src.downloader.video_fallback.httpx.get') as mock_get:
                mock_response = Mock(status_code=404, text="Not found")
                mock_get.return_value = mock_response

                fetcher = PipedStreamFetcher(
                    instances=["https://test.piped.io"],
                    rate_limit_handler=handler,
                )
                fetcher.get_streams("test_video")

                mock_get.assert_called()
                call_kwargs = mock_get.call_args.kwargs
                assert call_kwargs.get("proxy") == "http://proxy:8080"


class TestCobaltFetcherProxyIntegration:
    """Tests for CobaltFetcher proxy support."""

    def test_accepts_rate_limit_handler(self):
        """Should accept rate_limit_handler parameter."""
        handler = Mock(has_proxy=False)
        fetcher = CobaltFetcher(rate_limit_handler=handler)
        assert fetcher.handler is handler

    def test_passes_proxy_to_httpx_post(self):
        """Should pass proxy to httpx.post calls."""
        handler = Mock(has_proxy=True)

        with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_get_proxy:
            mock_get_proxy.return_value = "socks5://127.0.0.1:1080"
            with patch('src.downloader.video_fallback.httpx.post') as mock_post:
                mock_response = Mock(status_code=404, text="Not found")
                mock_post.return_value = mock_response

                fetcher = CobaltFetcher(
                    base_url="https://cobalt.example.com",
                    rate_limit_handler=handler,
                )
                fetcher.get_download_url("test_video")

                mock_post.assert_called()
                call_kwargs = mock_post.call_args.kwargs
                assert call_kwargs.get("proxy") == "socks5://127.0.0.1:1080"


class TestVideoFallbackChainProxyIntegration:
    """Tests for VideoFallbackChain proxy support."""

    def test_initializes_rate_limit_handler(self):
        """Should initialize rate limit handler."""
        with patch('src.downloader.video_fallback.get_rate_limit_handler') as mock_get:
            with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_proxy:
                mock_handler = Mock(has_proxy=False)
                mock_handler.get_proxy.return_value = None
                mock_get.return_value = mock_handler
                mock_proxy.return_value = None

                chain = VideoFallbackChain()

                mock_get.assert_called_once()
                assert chain.rate_limit_handler is mock_handler

    def test_passes_handler_to_fetchers(self):
        """Should pass rate limit handler to all fetchers."""
        with patch('src.downloader.video_fallback.get_rate_limit_handler') as mock_get:
            with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_proxy:
                mock_handler = Mock(has_proxy=False)
                mock_handler.get_proxy.return_value = None
                mock_get.return_value = mock_handler
                mock_proxy.return_value = None

                chain = VideoFallbackChain()

                # Check that fetchers received the handler
                assert chain.invidious.handler is mock_handler
                assert chain.piped.handler is mock_handler
                assert chain.cobalt.handler is mock_handler

    def test_download_stream_uses_proxy(self):
        """Should pass proxy to stream download."""
        with patch('src.downloader.video_fallback.get_rate_limit_handler') as mock_get:
            with patch('src.downloader.video_fallback.get_proxy_for_httpx') as mock_proxy:
                mock_handler = Mock(has_proxy=True)
                mock_handler.get_proxy.return_value = "socks5://127.0.0.1:1080"
                mock_get.return_value = mock_handler
                mock_proxy.return_value = "socks5://127.0.0.1:1080"

                with patch('src.downloader.video_fallback.httpx.stream') as mock_stream:
                    # Setup mock context manager
                    mock_response = Mock()
                    mock_response.status_code = 429
                    mock_context = MagicMock()
                    mock_context.__enter__.return_value = mock_response
                    mock_stream.return_value = mock_context

                    chain = VideoFallbackChain()
                    stream = StreamInfo(url="http://test.com/video.mp4", quality="720p", format="mp4")

                    with tempfile.TemporaryDirectory() as tmpdir:
                        chain._download_stream(stream, "test_video", Path(tmpdir), VideoTier.INVIDIOUS)

                    mock_stream.assert_called()
                    call_kwargs = mock_stream.call_args.kwargs
                    assert call_kwargs.get("proxy") == "socks5://127.0.0.1:1080"
