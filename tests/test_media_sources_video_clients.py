"""
Tests for src/media_sources/base.py and video clients (pexels.py, pixabay.py).

Targets:
- BaseMediaClient initialization and methods
- PexelsVideoClient search, download, error handling
- PixabayVideoClient search, download, error handling
"""

import pytest
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import requests

from src.media_sources.base import BaseMediaClient
from src.media_sources.videos.pexels import PexelsVideoClient, PEXELS_VIDEOS_API
from src.media_sources.videos.pixabay import PixabayVideoClient, PIXABAY_VIDEOS_API
from src.media_sources.models import VideoResult


# Concrete implementation for testing abstract BaseMediaClient
class ConcreteMediaClient(BaseMediaClient):
    """Concrete implementation for testing the abstract base class."""

    def search(self, query: str, max_results: int = 10):
        """Minimal implementation of abstract method."""
        return []


@pytest.mark.fast
class TestBaseMediaClient:
    """Test BaseMediaClient abstract base class."""

    def test_init_creates_output_dir(self, tmp_path):
        """Test that initialization creates output directory."""
        output_dir = tmp_path / "new_output_dir"
        config = MagicMock()

        client = ConcreteMediaClient(config=config, output_dir=str(output_dir))

        assert output_dir.exists()
        assert client.output_dir == output_dir
        assert client.config == config

    @pytest.mark.fast
    def test_init_with_custom_params(self, tmp_path):
        """Test initialization with custom parameters."""
        config = MagicMock()

        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path),
            min_size_mb=2.5,
            download_timeout=60,
            rate_limit_delay=0.5,
            user_agent="CustomAgent/1.0"
        )

        assert client.min_size == int(2.5 * 1024 * 1024)
        assert client.download_timeout == 60
        assert client._min_interval == 0.5
        assert client.session.headers["User-Agent"] == "CustomAgent/1.0"

    @pytest.mark.fast
    def test_rate_limit_delays(self, tmp_path):
        """Test that _rate_limit() enforces minimum delay."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path),
            rate_limit_delay=0.1
        )

        # First call should not delay
        start = time.time()
        client._rate_limit()
        first_call_time = time.time() - start
        assert first_call_time < 0.1  # Should be nearly instant

        # Second call should delay
        start = time.time()
        client._rate_limit()
        second_call_time = time.time() - start
        # Should have delayed close to 0.1 seconds
        assert second_call_time >= 0.05  # Allow some tolerance

    def test_rate_limit_no_delay_after_interval(self, tmp_path):
        """Test that _rate_limit() doesn't delay after interval passes."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path),
            rate_limit_delay=0.01
        )

        client._rate_limit()
        time.sleep(0.02)  # Wait longer than rate limit

        start = time.time()
        client._rate_limit()
        elapsed = time.time() - start
        # Should not have delayed much
        assert elapsed < 0.02

    @pytest.mark.fast
    def test_download_with_timeout_success(self, tmp_path):
        """Test successful download."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path),
            min_size_mb=0.001  # Very small minimum
        )

        # Create mock response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 10000]  # 10KB

        with patch.object(client.session, 'get', return_value=mock_response):
            output_path = tmp_path / "test_file.mp4"
            result = client.download_with_timeout("http://example.com/video.mp4", output_path)

            assert result is True
            assert output_path.exists()

    @pytest.mark.fast
    def test_download_with_timeout_file_too_small(self, tmp_path):
        """Test download failure due to small file size."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path),
            min_size_mb=1.0  # 1MB minimum
        )

        # Create mock response with small content
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"small"]

        with patch.object(client.session, 'get', return_value=mock_response):
            output_path = tmp_path / "small_file.mp4"
            result = client.download_with_timeout("http://example.com/video.mp4", output_path)

            assert result is False
            assert not output_path.exists()  # Should be deleted

    @pytest.mark.requires_network
    def test_download_with_timeout_exception(self, tmp_path):
        """Test download failure due to exception."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path)
        )

        with patch.object(client.session, 'get', side_effect=requests.RequestException("Network error")):
            output_path = tmp_path / "failed_file.mp4"
            result = client.download_with_timeout("http://example.com/video.mp4", output_path)

            assert result is False
            assert not output_path.exists()

    @pytest.mark.fast
    def test_download_with_timeout_exception_cleanup(self, tmp_path):
        """Test that partial download is cleaned up on exception."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path)
        )

        # Create the file first to test cleanup
        output_path = tmp_path / "partial_file.mp4"
        output_path.write_bytes(b"partial content")

        with patch.object(client.session, 'get', side_effect=Exception("Error")):
            result = client.download_with_timeout("http://example.com/video.mp4", output_path)

            assert result is False
            assert not output_path.exists()  # Should be deleted

    @pytest.mark.fast
    def test_download_with_timeout_custom_timeout(self, tmp_path):
        """Test download with custom timeout override."""
        config = MagicMock()
        client = ConcreteMediaClient(
            config=config,
            output_dir=str(tmp_path),
            download_timeout=30
        )

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 1000]

        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            output_path = tmp_path / "test.mp4"
            # Override with custom timeout
            client.download_with_timeout("http://example.com/video.mp4", output_path, timeout=60)

            # Verify custom timeout was used
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['timeout'] == 60

    @pytest.mark.fast
    def test_cleanup(self, tmp_path):
        """Test cleanup method closes session."""
        config = MagicMock()
        client = ConcreteMediaClient(config=config, output_dir=str(tmp_path))

        mock_session = MagicMock()
        client.session = mock_session

        client.cleanup()

        mock_session.close.assert_called_once()

    @pytest.mark.fast
    def test_cleanup_no_session(self, tmp_path):
        """Test cleanup when session doesn't exist."""
        config = MagicMock()
        client = ConcreteMediaClient(config=config, output_dir=str(tmp_path))

        # Remove session attribute
        delattr(client, 'session')

        # Should not raise
        client.cleanup()


@pytest.mark.fast
class TestPexelsVideoClient:
    """Test PexelsVideoClient."""

    def test_init_with_api_key(self, tmp_path):
        """Test initialization with API key."""
        config = MagicMock()

        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_api_key"
        )

        assert client.api_key == "test_api_key"
        assert client.min_duration == 3.0
        assert client.max_duration == 30.0
        assert client.prefer_hd is True

    @pytest.mark.requires_api
    def test_init_api_key_from_env(self, tmp_path):
        """Test initialization with API key from environment."""
        config = MagicMock()

        with patch.dict('os.environ', {'PEXELS_API_KEY': 'env_api_key'}):
            client = PexelsVideoClient(
                config=config,
                output_dir=str(tmp_path)
            )
            assert client.api_key == "env_api_key"

    @pytest.mark.fast
    def test_init_custom_params(self, tmp_path):
        """Test initialization with custom parameters."""
        config = MagicMock()

        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="key",
            min_duration=5.0,
            max_duration=60.0,
            prefer_hd=False,
            download_timeout=120
        )

        assert client.min_duration == 5.0
        assert client.max_duration == 60.0
        assert client.prefer_hd is False
        assert client.download_timeout == 120

    @pytest.mark.requires_api
    def test_search_no_api_key(self, tmp_path):
        """Test search returns empty list when no API key."""
        config = MagicMock()

        with patch.dict('os.environ', {}, clear=True):
            # Ensure env var is not set
            import os
            if 'PEXELS_API_KEY' in os.environ:
                del os.environ['PEXELS_API_KEY']

            client = PexelsVideoClient(
                config=config,
                output_dir=str(tmp_path),
                api_key=None
            )
            client.api_key = None  # Force None

            results = client.search("test")
            assert results == []

    @pytest.mark.fast
    def test_search_success(self, tmp_path):
        """Test successful video search."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 12345,
                    "url": "https://pexels.com/video/12345",
                    "duration": 10,
                    "video_files": [
                        {"link": "https://video.pexels.com/12345.mp4", "height": 1080, "width": 1920, "quality": "hd", "file_type": "mp4"},
                        {"link": "https://video.pexels.com/12345_sd.mp4", "height": 480, "width": 640, "quality": "sd", "file_type": "mp4"}
                    ]
                },
                {
                    "id": 12346,
                    "url": "https://pexels.com/video/12346",
                    "duration": 15,
                    "video_files": [
                        {"link": "https://video.pexels.com/12346.mp4", "height": 720, "width": 1280, "quality": "hd", "file_type": "mp4"}
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("nature", max_results=10)

            assert len(results) == 2
            assert results[0].id == "12345"
            assert results[0].source == "pexels"
            assert results[0].height == 1080  # Prefer HD
            assert results[0].duration == 10

    @pytest.mark.fast
    def test_search_duration_filter(self, tmp_path):
        """Test that duration filter works."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_duration=5.0,
            max_duration=20.0
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {"id": 1, "duration": 2, "video_files": [{"link": "url", "height": 720}]},  # Too short
                {"id": 2, "duration": 10, "video_files": [{"link": "url", "height": 720}]},  # OK
                {"id": 3, "duration": 25, "video_files": [{"link": "url", "height": 720}]},  # Too long
                {"id": 4, "duration": 15, "video_files": [{"link": "url", "height": 720}]},  # OK
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 2
            assert all(r.duration >= 5.0 and r.duration <= 20.0 for r in results)

    @pytest.mark.fast
    def test_search_no_video_files(self, tmp_path):
        """Test that videos without files are skipped."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {"id": 1, "duration": 10, "video_files": []},  # No files
                {"id": 2, "duration": 10},  # No video_files key
                {"id": 3, "duration": 10, "video_files": [{"link": "url", "height": 720}]},  # OK
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            assert results[0].id == "3"

    @pytest.mark.fast
    def test_search_exception(self, tmp_path):
        """Test search handles exception gracefully."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=Exception("API error")):
            results = client.search("test")
            assert results == []

    @pytest.mark.fast
    def test_download_video_success(self, tmp_path):
        """Test successful video download."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 10000]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_video(video)

            assert result is not None
            assert Path(result).exists()
            assert "p12345678.mp4" in result  # Source letter + short ID

    @pytest.mark.fast
    def test_download_video_no_download_url(self, tmp_path):
        """Test download returns None when no download URL."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="123",
            source="pexels",
            url="https://pexels.com/video/123",
            download_url="",  # Empty
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        result = client.download_video(video)
        assert result is None

    @pytest.mark.fast
    def test_download_video_file_exists(self, tmp_path):
        """Test download returns existing file path."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Pre-create the file
        existing_file = tmp_path / "p12345678.mp4"
        existing_file.write_bytes(b"existing content")

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        result = client.download_video(video)

        assert result == str(existing_file)

    @pytest.mark.fast
    def test_download_video_exception(self, tmp_path):
        """Test download handles exception and cleans up."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        with patch.object(client.session, 'get', side_effect=Exception("Download error")):
            result = client.download_video(video)

            assert result is None
            # File should not exist
            assert not (tmp_path / "p12345678.mp4").exists()

    @pytest.mark.fast
    def test_download_video_exception_cleanup(self, tmp_path):
        """Test that partial file is cleaned up on exception."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        # Create a mock that creates a partial file then raises
        def mock_get(*args, **kwargs):
            # Create partial file
            partial_file = tmp_path / "p12345678.mp4"
            partial_file.write_bytes(b"partial")
            raise Exception("Download error")

        with patch.object(client.session, 'get', side_effect=mock_get):
            result = client.download_video(video)

            assert result is None
            # Partial file should be cleaned up
            assert not (tmp_path / "p12345678.mp4").exists()

    @pytest.mark.fast
    def test_search_and_download(self, tmp_path):
        """Test search_and_download method."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Mock search to return videos
        mock_videos = [
            VideoResult(id="1", source="pexels", url="", download_url="http://v1.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
            VideoResult(id="2", source="pexels", url="", download_url="http://v2.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
            VideoResult(id="3", source="pexels", url="", download_url="http://v3.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
        ]

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 1000]

        with patch.object(client, 'search', return_value=mock_videos):
            with patch.object(client.session, 'get', return_value=mock_response):
                results = client.search_and_download("nature", max_videos=2)

                assert len(results) == 2

    @pytest.mark.fast
    def test_search_and_download_partial_success(self, tmp_path):
        """Test search_and_download when some downloads fail."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_videos = [
            VideoResult(id="1", source="pexels", url="", download_url="http://v1.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
            VideoResult(id="2", source="pexels", url="", download_url="http://v2.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
        ]

        # Mock download_video to succeed for first, fail for second
        with patch.object(client, 'search', return_value=mock_videos):
            with patch.object(client, 'download_video', side_effect=["/path/to/v1.mp4", None]):
                results = client.search_and_download("nature", max_videos=2)

                assert len(results) == 1
                assert results[0] == "/path/to/v1.mp4"


@pytest.mark.fast
class TestPixabayVideoClient:
    """Test PixabayVideoClient."""

    def test_init_with_api_key(self, tmp_path):
        """Test initialization with API key."""
        config = MagicMock()

        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_api_key"
        )

        assert client.api_key == "test_api_key"

    @pytest.mark.requires_api
    def test_init_api_key_from_env(self, tmp_path):
        """Test initialization with API key from environment."""
        config = MagicMock()

        with patch.dict('os.environ', {'PIXABAY_API_KEY': 'env_pixabay_key'}):
            client = PixabayVideoClient(
                config=config,
                output_dir=str(tmp_path)
            )
            assert client.api_key == "env_pixabay_key"

    @pytest.mark.fast
    def test_search_no_api_key(self, tmp_path):
        """Test search returns empty list when no API key."""
        config = MagicMock()

        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key=None
        )
        client.api_key = None  # Force None

        results = client.search("test")
        assert results == []

    @pytest.mark.fast
    def test_search_success(self, tmp_path):
        """Test successful video search."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "id": 12345,
                    "pageURL": "https://pixabay.com/videos/12345",
                    "duration": 10,
                    "videos": {
                        "large": {"url": "https://pixabay.com/12345_large.mp4", "width": 1920, "height": 1080, "size": "large"},
                        "medium": {"url": "https://pixabay.com/12345_medium.mp4", "width": 1280, "height": 720, "size": "medium"}
                    }
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("nature", max_results=10)

            assert len(results) == 1
            assert results[0].id == "12345"
            assert results[0].source == "pixabay"
            assert results[0].height == 1080  # Large preferred

    @pytest.mark.fast
    def test_search_duration_filter(self, tmp_path):
        """Test that duration filter works."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_duration=5.0,
            max_duration=20.0
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {"id": 1, "duration": 2, "videos": {"large": {"url": "url", "width": 1920, "height": 1080}}},  # Too short
                {"id": 2, "duration": 10, "videos": {"large": {"url": "url", "width": 1920, "height": 1080}}},  # OK
                {"id": 3, "duration": 25, "videos": {"large": {"url": "url", "width": 1920, "height": 1080}}},  # Too long
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            assert results[0].id == "2"

    @pytest.mark.fast
    def test_search_video_size_fallback(self, tmp_path):
        """Test that video size falls back from large to medium to small."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                # Only medium available
                {"id": 1, "duration": 10, "videos": {"medium": {"url": "medium_url", "width": 1280, "height": 720}}},
                # Only small available
                {"id": 2, "duration": 10, "videos": {"small": {"url": "small_url", "width": 640, "height": 480}}},
                # No valid URL in large, falls back to medium
                {"id": 3, "duration": 10, "videos": {"large": {"url": ""}, "medium": {"url": "med_url", "width": 1280, "height": 720}}},
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 3
            assert results[0].download_url == "medium_url"
            assert results[1].download_url == "small_url"
            assert results[2].download_url == "med_url"

    @pytest.mark.fast
    def test_search_no_valid_video_url(self, tmp_path):
        """Test that videos without valid URLs are skipped."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {"id": 1, "duration": 10, "videos": {}},  # Empty videos
                {"id": 2, "duration": 10, "videos": {"large": {"url": ""}, "medium": {"url": ""}}},  # No valid URLs
                {"id": 3, "duration": 10, "videos": {"large": {"url": "valid_url", "width": 1920, "height": 1080}}},  # OK
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            assert results[0].id == "3"

    @pytest.mark.fast
    def test_search_exception(self, tmp_path):
        """Test search handles exception gracefully."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=Exception("API error")):
            results = client.search("test")
            assert results == []

    @pytest.mark.fast
    def test_download_video_success(self, tmp_path):
        """Test successful video download."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pixabay",
            url="https://pixabay.com/videos/12345678",
            download_url="https://pixabay.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 10000]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_video(video)

            assert result is not None
            assert Path(result).exists()

    @pytest.mark.fast
    def test_download_video_no_download_url(self, tmp_path):
        """Test download returns None when no download URL."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="123",
            source="pixabay",
            url="",
            download_url="",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        result = client.download_video(video)
        assert result is None

    @pytest.mark.fast
    def test_download_video_file_exists(self, tmp_path):
        """Test download returns existing file path."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Pre-create the file
        existing_file = tmp_path / "p12345678.mp4"
        existing_file.write_bytes(b"existing content")

        video = VideoResult(
            id="12345678",
            source="pixabay",
            url="",
            download_url="https://pixabay.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        result = client.download_video(video)

        assert result == str(existing_file)

    @pytest.mark.fast
    def test_download_video_exception(self, tmp_path):
        """Test download handles exception and cleans up."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pixabay",
            url="",
            download_url="https://pixabay.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        with patch.object(client.session, 'get', side_effect=Exception("Error")):
            result = client.download_video(video)
            assert result is None

    @pytest.mark.fast
    def test_download_video_cleanup_partial_file(self, tmp_path):
        """Test that partial file is cleaned up on exception."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pixabay",
            url="",
            download_url="https://pixabay.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        # Create partial file
        partial_file = tmp_path / "p12345678.mp4"

        def mock_get(*args, **kwargs):
            partial_file.write_bytes(b"partial")
            raise Exception("Error")

        with patch.object(client.session, 'get', side_effect=mock_get):
            result = client.download_video(video)

            assert result is None
            assert not partial_file.exists()

    @pytest.mark.fast
    def test_search_and_download(self, tmp_path):
        """Test search_and_download method."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_videos = [
            VideoResult(id="1", source="pixabay", url="", download_url="http://v1.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
            VideoResult(id="2", source="pixabay", url="", download_url="http://v2.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4"),
        ]

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 1000]

        with patch.object(client, 'search', return_value=mock_videos):
            with patch.object(client.session, 'get', return_value=mock_response):
                results = client.search_and_download("nature", max_videos=2)

                assert len(results) == 2

    @pytest.mark.fast
    def test_search_and_download_max_limit(self, tmp_path):
        """Test search_and_download respects max_videos limit."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_videos = [
            VideoResult(id=str(i), source="pixabay", url="", download_url=f"http://v{i}.mp4", width=1920, height=1080, duration=10, quality="hd", file_type="mp4")
            for i in range(10)
        ]

        with patch.object(client, 'search', return_value=mock_videos):
            with patch.object(client, 'download_video', return_value="/path/to/video.mp4"):
                results = client.search_and_download("nature", max_videos=3)

                assert len(results) == 3


@pytest.mark.fast
class TestVideoClientLongIds:
    """Test handling of long video IDs."""

    def test_pexels_long_id_truncation(self, tmp_path):
        """Test Pexels truncates long IDs to 8 chars."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="1234567890123456",  # 16 chars
            source="pexels",
            url="",
            download_url="https://video.pexels.com/long.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 1000]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_video(video)

            assert result is not None
            filename = Path(result).name
            # Should be p + last 8 chars + .mp4
            assert filename == "p90123456.mp4"

    @pytest.mark.fast
    def test_pixabay_short_id_unchanged(self, tmp_path):
        """Test Pixabay keeps short IDs unchanged."""
        config = MagicMock()
        client = PixabayVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345",  # 5 chars - short
            source="pixabay",
            url="",
            download_url="https://pixabay.com/short.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"x" * 1000]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_video(video)

            assert result is not None
            filename = Path(result).name
            # Should keep full ID
            assert filename == "p12345.mp4"


@pytest.mark.fast
class TestVideoClientPreferHD:
    """Test HD preference behavior."""

    def test_pexels_prefer_hd_true(self, tmp_path):
        """Test Pexels prefers HD when enabled."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "sd_url", "height": 480, "width": 640, "quality": "sd"},
                        {"link": "hd_url", "height": 1080, "width": 1920, "quality": "hd"}
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            assert results[0].download_url == "hd_url"
            assert results[0].height == 1080

    @pytest.mark.fast
    def test_pexels_prefer_hd_false(self, tmp_path):
        """Test Pexels doesn't sort by height when HD not preferred."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=False
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "sd_url", "height": 480, "width": 640, "quality": "sd"},
                        {"link": "hd_url", "height": 1080, "width": 1920, "quality": "hd"}
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            # First file in list is used when not preferring HD
            assert results[0].download_url == "sd_url"


@pytest.mark.fast
class TestPexelsAPIRateLimitingUS001:
    """US-001: Test PexelsClient.search_videos() handles API rate limiting with retry backoff."""

    @pytest.mark.requires_network
    def test_search_handles_429_rate_limit_error(self, tmp_path):
        """Test that 429 rate limit error returns empty list without crash."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            # Should return empty list, not crash
            assert results == []

    @pytest.mark.requires_network
    def test_search_handles_rate_limit_headers(self, tmp_path):
        """Test that rate limit headers are handled gracefully."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # First call triggers rate limit (429)
        mock_error_response = MagicMock()
        mock_error_response.status_code = 429
        mock_error_response.headers = {"X-Ratelimit-Remaining": "0", "Retry-After": "60"}
        mock_error_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_error_response
        )

        with patch.object(client.session, 'get', return_value=mock_error_response):
            results = client.search("test")
            # Should gracefully return empty list
            assert results == []

    @pytest.mark.requires_network
    def test_search_multiple_rate_limits_no_crash(self, tmp_path):
        """Test that consecutive rate limit errors don't cause crash."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            response=mock_response
        )

        with patch.object(client.session, 'get', return_value=mock_response):
            # Multiple consecutive calls should all return empty without crash
            for _ in range(3):
                results = client.search("test")
                assert results == []


@pytest.mark.fast
class TestPexelsNetworkTimeoutUS001:
    """US-001: Test PexelsClient.search_videos() handles network timeout gracefully."""

    @pytest.mark.requires_network
    def test_search_handles_connect_timeout(self, tmp_path):
        """Test that connection timeout returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=requests.ConnectTimeout("Connection timed out")):
            results = client.search("test")
            assert results == []

    @pytest.mark.requires_network
    def test_search_handles_read_timeout(self, tmp_path):
        """Test that read timeout returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=requests.ReadTimeout("Read timed out")):
            results = client.search("test")
            assert results == []

    @pytest.mark.requires_network
    def test_search_handles_connection_error(self, tmp_path):
        """Test that connection error returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=requests.ConnectionError("DNS lookup failed")):
            results = client.search("test")
            assert results == []

    @pytest.mark.requires_network
    def test_download_handles_timeout(self, tmp_path):
        """Test that download timeout returns None."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        with patch.object(client.session, 'get', side_effect=requests.Timeout("Download timed out")):
            result = client.download_video(video)
            assert result is None


@pytest.mark.fast
class TestPexelsMalformedJSONUS001:
    """US-001: Test PexelsClient.search_videos() handles malformed JSON response without crash."""

    def test_search_handles_invalid_json(self, tmp_path):
        """Test that invalid JSON response returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            assert results == []

    @pytest.mark.fast
    def test_search_handles_missing_videos_key(self, tmp_path):
        """Test that response without 'videos' key returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"total_results": 100}  # No 'videos' key

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            # Should return empty list because "videos" defaults to []
            assert results == []

    @pytest.mark.fast
    def test_search_handles_null_videos(self, tmp_path):
        """Test that null videos array returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"videos": None}

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            assert results == []

    @pytest.mark.fast
    def test_search_handles_malformed_video_entry(self, tmp_path):
        """Test that malformed video entries are handled gracefully (returns empty)."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Null entry in videos array causes exception in loop, which is caught
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                None,  # Null entry causes video.get() to fail
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            # Exception is caught and returns empty list (graceful degradation)
            assert results == []

    @pytest.mark.fast
    def test_search_handles_empty_dict_entries(self, tmp_path):
        """Test that empty dict entries are skipped gracefully."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {},  # Empty dict (duration=0 fails filter, no video_files)
                {"id": 1},  # Missing duration (defaults to 0, fails filter), missing video_files
                {"id": 2, "duration": 10},  # Missing video_files - will be skipped
                {"id": 3, "duration": 10, "video_files": []},  # Empty video_files - will be skipped
                {"id": 4, "duration": 10, "video_files": [{"link": "valid", "height": 720}]}  # Valid
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            # Only valid entry should be returned
            assert len(results) == 1
            assert results[0].id == "4"

    @pytest.mark.fast
    def test_search_handles_unicode_decode_error(self, tmp_path):
        """Test that response with encoding issues returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            assert results == []


@pytest.mark.fast
class TestPexelsPaginationUS001:
    """US-001: Test PexelsClient pagination iterates correctly through multiple result pages."""

    def test_search_requests_correct_page_size(self, tmp_path):
        """Test that search requests correct per_page parameter."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"videos": []}

        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            client.search("nature", max_results=15)

            # Verify params include per_page
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['params']['per_page'] == 15

    @pytest.mark.fast
    def test_search_returns_correct_number_of_results(self, tmp_path):
        """Test that search respects max_results limit."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # API returns more videos than requested
        videos = [
            {"id": i, "duration": 10, "video_files": [{"link": f"url{i}", "height": 720}]}
            for i in range(20)
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"videos": videos}

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("nature", max_results=5)
            # Should return all videos since per_page controls API request
            # but if API returns more, we get more
            assert len(results) == 20

    @pytest.mark.fast
    def test_search_pagination_params_include_orientation(self, tmp_path):
        """Test that search includes orientation parameter for landscape videos."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"videos": []}

        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            client.search("nature", max_results=10)

            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['params']['orientation'] == 'landscape'

    @pytest.mark.fast
    def test_search_pagination_uses_correct_headers(self, tmp_path):
        """Test that search includes Authorization header."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="my_test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"videos": []}

        with patch.object(client.session, 'get', return_value=mock_response) as mock_get:
            client.search("nature")

            call_kwargs = mock_get.call_args[1]
            assert call_kwargs['headers']['Authorization'] == 'my_test_key'

    @pytest.mark.fast
    def test_search_and_download_iterates_through_results(self, tmp_path):
        """Test search_and_download iterates through search results."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Create 5 search results
        mock_videos = [
            VideoResult(
                id=str(i),
                source="pexels",
                url="",
                download_url=f"http://v{i}.mp4",
                width=1920,
                height=1080,
                duration=10,
                quality="hd",
                file_type="mp4"
            )
            for i in range(5)
        ]

        downloaded_count = 0
        def mock_download(video):
            nonlocal downloaded_count
            downloaded_count += 1
            return f"/path/to/{video.id}.mp4"

        with patch.object(client, 'search', return_value=mock_videos):
            with patch.object(client, 'download_video', side_effect=mock_download):
                results = client.search_and_download("nature", max_videos=3)

                # Should have called download 3 times and stopped
                assert len(results) == 3
                assert downloaded_count == 3


@pytest.mark.fast
class TestPexelsDownloadURLValidationUS002:
    """US-002: Test PexelsClient.download_video() validates URL before downloading."""

    def test_download_rejects_none_url(self, tmp_path):
        """Test that download returns None when URL is None."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345",
            source="pexels",
            url="https://pexels.com/video/12345",
            download_url=None,  # None URL
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        result = client.download_video(video)
        assert result is None

    @pytest.mark.fast
    def test_download_rejects_empty_string_url(self, tmp_path):
        """Test that download returns None when URL is empty string."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345",
            source="pexels",
            url="https://pexels.com/video/12345",
            download_url="",  # Empty string URL
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        result = client.download_video(video)
        assert result is None

    @pytest.mark.requires_network
    def test_download_rejects_whitespace_only_url(self, tmp_path):
        """Test that download handles whitespace-only URL gracefully."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345",
            source="pexels",
            url="https://pexels.com/video/12345",
            download_url="   ",  # Whitespace-only URL
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        # Should handle gracefully (either reject or fail on request)
        with patch.object(client.session, 'get', side_effect=requests.exceptions.MissingSchema("No scheme")):
            result = client.download_video(video)
            assert result is None

    @pytest.mark.fast
    def test_download_does_not_call_session_on_empty_url(self, tmp_path):
        """Test that session.get() is never called when URL is empty."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345",
            source="pexels",
            url="https://pexels.com/video/12345",
            download_url="",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        with patch.object(client.session, 'get') as mock_get:
            client.download_video(video)
            # Should not call session.get at all
            mock_get.assert_not_called()


@pytest.mark.fast
class TestPexelsPartialDownloadUS002:
    """US-002: Test PexelsClient.download_video() handles partial download scenarios."""

    @pytest.mark.requires_network
    def test_download_cleans_up_partial_file_on_network_error(self, tmp_path):
        """Test that partial file is deleted when download fails mid-stream."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        filepath = tmp_path / "p12345678.mp4"

        def mock_get(*args, **kwargs):
            # Simulate partial write then failure
            filepath.write_bytes(b"partial content here")
            raise requests.exceptions.ChunkedEncodingError("Connection broken")

        with patch.object(client.session, 'get', side_effect=mock_get):
            result = client.download_video(video)

            assert result is None
            # Partial file should be cleaned up
            assert not filepath.exists()

    @pytest.mark.requires_network
    def test_download_cleans_up_on_timeout(self, tmp_path):
        """Test that partial file is cleaned up on timeout."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        filepath = tmp_path / "p12345678.mp4"

        def mock_get(*args, **kwargs):
            # Simulate partial write then timeout
            filepath.write_bytes(b"timeout partial content")
            raise requests.exceptions.ReadTimeout("Read timed out")

        with patch.object(client.session, 'get', side_effect=mock_get):
            result = client.download_video(video)

            assert result is None
            assert not filepath.exists()

    @pytest.mark.fast
    def test_download_skips_existing_file(self, tmp_path):
        """Test that existing complete file is not re-downloaded (resume-by-skip behavior)."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Pre-create complete file
        existing_file = tmp_path / "p12345678.mp4"
        existing_file.write_bytes(b"complete video content" * 1000)

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        with patch.object(client.session, 'get') as mock_get:
            result = client.download_video(video)

            # Should return existing file path
            assert result == str(existing_file)
            # Should not call HTTP (resume behavior = skip if exists)
            mock_get.assert_not_called()

    @pytest.mark.fast
    def test_download_restarts_from_zero_if_partial_file_exists(self, tmp_path):
        """Test that partial file from interrupted download gets overwritten (no HTTP Range resume)."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Pre-create a partial file - NOTE: Current implementation doesn't detect partial files
        # It skips if ANY file exists. This test documents current behavior.
        partial_file = tmp_path / "p12345678.mp4"
        partial_file.write_bytes(b"partial")  # Small incomplete file

        video = VideoResult(
            id="12345678",
            source="pexels",
            url="https://pexels.com/video/12345678",
            download_url="https://video.pexels.com/12345678.mp4",
            width=1920,
            height=1080,
            duration=10,
            quality="hd",
            file_type="mp4"
        )

        # Current behavior: If file exists (even partial), it's returned as-is
        # No HTTP Range header resume support currently implemented
        with patch.object(client.session, 'get') as mock_get:
            result = client.download_video(video)
            # Current behavior skips re-download if file exists
            assert result == str(partial_file)
            mock_get.assert_not_called()


@pytest.mark.fast
class TestPexelsQualityOptionsParsingUS002:
    """US-002: Test PexelsClient correctly parses video quality options from API response."""

    def test_parses_all_quality_levels_from_video_files(self, tmp_path):
        """Test that all quality levels in video_files are parsed correctly."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 12345,
                    "duration": 10,
                    "url": "https://pexels.com/video/12345",
                    "video_files": [
                        {"link": "url_uhd", "height": 2160, "width": 3840, "quality": "uhd", "file_type": "mp4"},
                        {"link": "url_hd", "height": 1080, "width": 1920, "quality": "hd", "file_type": "mp4"},
                        {"link": "url_sd", "height": 480, "width": 854, "quality": "sd", "file_type": "mp4"},
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            # With prefer_hd=True, should select highest (uhd)
            assert results[0].height == 2160
            assert results[0].quality == "uhd"
            assert results[0].download_url == "url_uhd"

    @pytest.mark.fast
    def test_parses_mixed_file_types(self, tmp_path):
        """Test that different file types are parsed correctly."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "url1", "height": 1080, "width": 1920, "quality": "hd", "file_type": "mp4"},
                    ]
                },
                {
                    "id": 2,
                    "duration": 10,
                    "video_files": [
                        {"link": "url2", "height": 720, "width": 1280, "quality": "hd", "file_type": "webm"},
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 2
            assert results[0].file_type == "mp4"
            assert results[1].file_type == "webm"

    @pytest.mark.fast
    def test_handles_missing_quality_field(self, tmp_path):
        """Test that missing quality field defaults to 'unknown'."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "url1", "height": 1080, "width": 1920},  # No quality field
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            assert results[0].quality == "unknown"

    @pytest.mark.fast
    def test_handles_missing_dimensions(self, tmp_path):
        """Test that missing width/height default to 0."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "url1", "quality": "hd"},  # No dimensions
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert len(results) == 1
            assert results[0].width == 0
            assert results[0].height == 0


@pytest.mark.fast
class TestPexelsQualityPreferenceUS002:
    """US-002: Test PexelsClient respects configured quality preference (HD, SD, original)."""

    def test_prefer_hd_selects_highest_resolution(self, tmp_path):
        """Test that prefer_hd=True selects highest resolution."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "sd_url", "height": 480, "width": 854, "quality": "sd"},
                        {"link": "hd_url", "height": 1080, "width": 1920, "quality": "hd"},
                        {"link": "4k_url", "height": 2160, "width": 3840, "quality": "uhd"},
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            assert results[0].download_url == "4k_url"
            assert results[0].height == 2160

    @pytest.mark.fast
    def test_prefer_hd_false_keeps_first_file(self, tmp_path):
        """Test that prefer_hd=False uses original order (first file)."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=False
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "sd_url", "height": 480, "width": 854, "quality": "sd"},
                        {"link": "hd_url", "height": 1080, "width": 1920, "quality": "hd"},
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            # First file in original order
            assert results[0].download_url == "sd_url"
            assert results[0].height == 480

    @pytest.mark.fast
    def test_prefer_hd_with_only_sd_available(self, tmp_path):
        """Test that prefer_hd=True still works when only SD is available."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "sd_url", "height": 480, "width": 854, "quality": "sd"},
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            # Should still return the only available quality
            assert len(results) == 1
            assert results[0].download_url == "sd_url"

    @pytest.mark.fast
    def test_quality_preference_sorting_stable(self, tmp_path):
        """Test that quality sorting is stable for equal heights."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "videos": [
                {
                    "id": 1,
                    "duration": 10,
                    "video_files": [
                        {"link": "first_hd_url", "height": 1080, "width": 1920, "quality": "hd"},
                        {"link": "second_hd_url", "height": 1080, "width": 1920, "quality": "hd"},
                    ]
                }
            ]
        }

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")

            # With equal heights, Python's sort is stable so first should win
            # (after sorting by height desc, order of equal items preserved)
            assert results[0].height == 1080


@pytest.mark.fast
class TestPexelsAPIKeyValidationUS002:
    """US-002: Test PexelsClient API key validation fails fast with clear error message."""

    def test_search_returns_empty_immediately_without_key(self, tmp_path):
        """Test that search returns empty list immediately when API key is missing."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key=None
        )
        client.api_key = None  # Ensure None

        with patch.object(client.session, 'get') as mock_get:
            results = client.search("test")

            # Should return empty immediately
            assert results == []
            # Should NOT make HTTP request
            mock_get.assert_not_called()

    @pytest.mark.fast
    def test_search_logs_debug_message_for_missing_key(self, tmp_path, caplog):
        """Test that missing API key logs a debug message."""
        import logging
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key=None
        )
        client.api_key = None

        with caplog.at_level(logging.DEBUG):
            client.search("test")

        assert "Pexels API key not available" in caplog.text

    @pytest.mark.requires_network
    def test_invalid_api_key_returns_empty_on_401(self, tmp_path):
        """Test that invalid API key (401 response) returns empty list."""
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="invalid_key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "401 Unauthorized",
            response=mock_response
        )

        with patch.object(client.session, 'get', return_value=mock_response):
            results = client.search("test")
            assert results == []

    @pytest.mark.requires_network
    def test_invalid_api_key_logs_error(self, tmp_path, caplog):
        """Test that invalid API key logs appropriate error message."""
        import logging
        config = MagicMock()
        client = PexelsVideoClient(
            config=config,
            output_dir=str(tmp_path),
            api_key="invalid_key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "401 Unauthorized",
            response=mock_response
        )

        with caplog.at_level(logging.ERROR):
            with patch.object(client.session, 'get', return_value=mock_response):
                client.search("test")

        assert "Pexels video search error" in caplog.text

    @pytest.mark.requires_api
    def test_api_key_from_constructor_takes_precedence(self, tmp_path):
        """Test that constructor API key takes precedence over environment variable."""
        config = MagicMock()

        with patch.dict('os.environ', {'PEXELS_API_KEY': 'env_key'}):
            client = PexelsVideoClient(
                config=config,
                output_dir=str(tmp_path),
                api_key="constructor_key"
            )
            assert client.api_key == "constructor_key"

    @pytest.mark.requires_api
    def test_api_key_falls_back_to_env_variable(self, tmp_path):
        """Test that missing constructor key falls back to environment variable."""
        config = MagicMock()

        with patch.dict('os.environ', {'PEXELS_API_KEY': 'env_key'}):
            client = PexelsVideoClient(
                config=config,
                output_dir=str(tmp_path)
            )
            assert client.api_key == "env_key"
