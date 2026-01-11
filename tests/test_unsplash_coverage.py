"""
Test coverage for src/media_sources/images/unsplash.py

Target: Cover all 26 missed lines to achieve 100% coverage.

Covers:
- No API key scenario
- No download_url in result
- Existing file checks (too small / size OK)
- HEAD pre-check failures
- Content-length too small
- Downloaded file too small
- Entity metadata vs regular metadata
- Download exceptions
"""

import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock, PropertyMock


class TestUnsplashSearchNoApiKey:
    """Test search behavior without API key."""

    def test_search_no_api_key_returns_empty(self, tmp_path):
        """Test that search returns empty list when no API key."""
        from src.media_sources.images.unsplash import UnsplashImageClient

        mock_config = Mock()

        with patch.dict('os.environ', {}, clear=True):
            client = UnsplashImageClient(
                config=mock_config,
                output_dir=str(tmp_path),
                api_key=None
            )

        result = client.search("test query", max_results=10)
        assert result == []


class TestUnsplashSearchResults:
    """Test search result processing."""

    def test_search_skips_photos_without_download_url(self, tmp_path):
        """Test that photos without download URL are skipped."""
        from src.media_sources.images.unsplash import UnsplashImageClient

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        # Mock API response with a photo missing download URL
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": "photo1",
                    "urls": {},  # No urls -> no download_url
                    "width": 1920,
                    "height": 1080
                },
                {
                    "id": "photo2",
                    "urls": {"full": "https://example.com/photo2.jpg"},
                    "width": 1920,
                    "height": 1080,
                    "links": {"html": "https://unsplash.com/photos/photo2"},
                    "user": {"name": "Photographer"},
                    "description": "Test description"
                }
            ]
        }

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'get', return_value=mock_response):
                result = client.search("test", max_results=5)

        # Only photo2 should be in results (photo1 has no download_url)
        assert len(result) == 1
        assert result[0].id == "unsplash_photo2"


class TestUnsplashDownloadImage:
    """Test image download scenarios."""

    def test_existing_file_returns_path(self, tmp_path):
        """Test that existing file with adequate size returns path."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.001  # 1KB minimum
        )

        # Create existing file larger than min_size
        existing_file = tmp_path / "ash_id12.jpg"
        existing_file.write_bytes(b"x" * 2000)  # 2KB

        image = ImageResult(
            id="unsplash_id12",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Test",
            description="Test",
            tags=[]
        )

        result = client.download_image(image, check_size=True)
        assert result == str(existing_file)

    def test_existing_file_too_small_returns_none(self, tmp_path):
        """Test that existing file below min_size returns None."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB minimum
        )

        # Create existing file smaller than min_size
        existing_file = tmp_path / "ash_id12.jpg"
        existing_file.write_bytes(b"x" * 100)  # 100 bytes

        image = ImageResult(
            id="unsplash_id12",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Test",
            description="Test",
            tags=[]
        )

        result = client.download_image(image, check_size=True)
        assert result is None

    def test_head_precheck_fails_continues_download(self, tmp_path):
        """Test that download continues even if HEAD request fails."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001  # Very small minimum
        )

        image = ImageResult(
            id="unsplash_testid1",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Test",
            description="Test",
            tags=[]
        )

        # Mock HEAD to fail
        mock_get_response = Mock()
        mock_get_response.headers = {'content-length': '10000'}
        mock_get_response.iter_content.return_value = [b"x" * 10000]

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'head', side_effect=Exception("Timeout")):
                with patch.object(client.session, 'get', return_value=mock_get_response):
                    result = client.download_image(image, check_size=True)

        # Should still succeed despite HEAD failure
        assert result is not None

    def test_content_length_precheck_too_small(self, tmp_path):
        """Test that image is rejected if HEAD content-length is too small."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB minimum
        )

        image = ImageResult(
            id="unsplash_small1",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Test",
            description="Test",
            tags=[]
        )

        # Mock HEAD to return small content-length
        mock_head_response = Mock()
        mock_head_response.headers = {'content-length': '1000'}  # 1KB < 1MB

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'head', return_value=mock_head_response):
                result = client.download_image(image, check_size=True)

        assert result is None

    def test_downloaded_file_too_small_deleted(self, tmp_path):
        """Test that downloaded file below min_size is deleted."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB minimum
        )

        image = ImageResult(
            id="unsplash_toosmall",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Test",
            description="Test",
            tags=[]
        )

        # Mock HEAD to return 0 (unknown size)
        mock_head_response = Mock()
        mock_head_response.headers = {'content-length': '0'}

        # Mock GET to return small file
        mock_get_response = Mock()
        mock_get_response.headers = {'content-length': '0'}
        mock_get_response.iter_content.return_value = [b"x" * 100]  # 100 bytes

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'head', return_value=mock_head_response):
                with patch.object(client.session, 'get', return_value=mock_get_response):
                    result = client.download_image(image, check_size=True)

        assert result is None
        # Temp file should be deleted
        temp_file = tmp_path / "oosmall.tmp"
        assert not temp_file.exists()

    def test_entity_metadata_saved(self, tmp_path):
        """Test that entity metadata is saved to .entity.json."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001
        )

        image = ImageResult(
            id="unsplash_entity1",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="John Doe",
            description="Test photo",
            tags=["tag1"]
        )

        # Mock successful download
        mock_get_response = Mock()
        mock_get_response.headers = {'content-length': '10000'}
        mock_get_response.iter_content.return_value = [b"x" * 10000]

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'head', side_effect=Exception()):
                with patch.object(client.session, 'get', return_value=mock_get_response):
                    result = client.download_image(
                        image,
                        check_size=True,
                        entity_name="Test Entity",
                        entity_type="GPE"
                    )

        assert result is not None
        # Check that .entity.json was created
        meta_file = Path(result).with_suffix('.entity.json')
        assert meta_file.exists()

        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        assert metadata['entity_name'] == 'Test Entity'
        assert metadata['entity_type'] == 'GPE'

    def test_regular_metadata_saved_when_no_entity(self, tmp_path):
        """Test that regular metadata is saved to .meta.json when no entity."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001
        )

        image = ImageResult(
            id="unsplash_noent1",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Jane Doe",
            description="No entity photo",
            tags=["tag2"]
        )

        mock_get_response = Mock()
        mock_get_response.headers = {'content-length': '10000'}
        mock_get_response.iter_content.return_value = [b"x" * 10000]

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'head', side_effect=Exception()):
                with patch.object(client.session, 'get', return_value=mock_get_response):
                    result = client.download_image(
                        image,
                        check_size=True,
                        entity_name="",  # No entity
                        entity_type=""
                    )

        assert result is not None
        # Check that .meta.json was created (not .entity.json)
        meta_file = Path(result).with_suffix('.meta.json')
        assert meta_file.exists()
        entity_file = Path(result).with_suffix('.entity.json')
        assert not entity_file.exists()

    def test_download_exception_adds_to_failed(self, tmp_path):
        """Test that download exception adds to failed_downloads list."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001
        )

        image = ImageResult(
            id="unsplash_fail1",
            source="unsplash",
            url="https://unsplash.com/photo",
            download_url="https://example.com/photo.jpg",
            width=1920,
            height=1080,
            photographer="Test",
            description="Test",
            tags=[]
        )

        with patch.object(client, '_rate_limit'):
            with patch.object(client.session, 'head', side_effect=Exception()):
                with patch.object(client.session, 'get', side_effect=Exception("Download failed")):
                    result = client.download_image(image, check_size=True)

        assert result is None
        assert "unsplash_fail1" in client.failed_downloads


class TestUnsplashSearchAndDownload:
    """Test search_and_download method."""

    def test_search_and_download_limits_results(self, tmp_path):
        """Test that search_and_download respects max_images limit."""
        from src.media_sources.images.unsplash import UnsplashImageClient
        from src.media_sources.models import ImageResult

        mock_config = Mock()
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001
        )

        # Create mock search results
        mock_results = [
            ImageResult(
                id=f"unsplash_img{i}",
                source="unsplash",
                url=f"https://unsplash.com/photo{i}",
                download_url=f"https://example.com/photo{i}.jpg",
                width=1920,
                height=1080,
                photographer="Test",
                description="Test",
                tags=[]
            )
            for i in range(5)
        ]

        with patch.object(client, 'search', return_value=mock_results):
            with patch.object(client, 'download_image', return_value=str(tmp_path / "img.jpg")):
                result = client.search_and_download("test query", max_images=2)

        # Should only have 2 results
        assert len(result) == 2
