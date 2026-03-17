"""
Tests for Google/Bing image search client (google_bing.py).

Targets uncovered lines:
- Lines 23-148: GoogleBingImageClient initialization and _init_client()
- Lines 145-194: _search_with_timeout() with threading
- Lines 196-201: _reinitialize_client()
- Lines 202-408: search_and_download() main workflow
- Lines 410-467: _download_single_image() with requests
- Lines 469-491: _cleanup_empty_folders()

Current coverage: 5.93%
Target coverage: 75%+

Created: 2026-01-10 (Session 13 - Critical gap coverage, Phase 1)
"""

import json
import random
import string
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# GoogleBingImageClient tests


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for testing"""
    output_dir = tmp_path / "test_images"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_imagedl_client():
    """Mock imagedl ImageClient"""
    mock_client = Mock()
    mock_client.search = Mock(return_value=[
        {
            'candidate_urls': ['https://example.com/image1.jpg'],
            'title': 'Test Image 1'
        },
        {
            'candidate_urls': ['https://example.com/image2.jpg'],
            'title': 'Test Image 2'
        }
    ])
    return mock_client


# ============================================================================
# Test GoogleBingImageClient Initialization
# ============================================================================

class TestGoogleBingImageClientInit:
    """Test client initialization and configuration"""

    @pytest.mark.fast
    def test_init_creates_output_directory(self, temp_output_dir):
        """Test that initialization creates output directory"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))

            assert client.output_dir == temp_output_dir
            assert temp_output_dir.exists()

    @pytest.mark.fast
    def test_init_sets_default_configuration(self, temp_output_dir):
        """Test default configuration values"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))

            assert client.source == "GoogleImageClient"
            assert client.min_size == int(1.0 * 1024 * 1024)  # 1MB default
            assert client.download_timeout == 10
            assert client.max_search_time == 300
            assert client.max_results_to_check == 500
            assert client.search_until_found is True

    @pytest.mark.fast
    def test_init_accepts_custom_configuration(self, temp_output_dir):
        """Test custom configuration parameters"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(
                output_dir=str(temp_output_dir),
                source="BingImageClient",
                min_size_mb=2.5,
                download_timeout=30,
                max_search_time=600,
                max_results_to_check=1000,
                search_until_found=False
            )

            assert client.source == "BingImageClient"
            assert client.min_size == int(2.5 * 1024 * 1024)
            assert client.download_timeout == 30
            assert client.max_search_time == 600
            assert client.max_results_to_check == 1000
            assert client.search_until_found is False

    @pytest.mark.fast
    def test_behavior_without_client(self, temp_output_dir, mock_imagedl_client):
        """Test behavior when client is None (imagedl not available)"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        # Create client with mock, then set client to None to simulate missing imagedl
        mock_spec = Mock()
        mock_spec.origin = '/path/to/site-packages/imagedl/__init__.py'
        mock_spec.loader.exec_module = Mock()

        mock_module = Mock()
        mock_module.imagedl = Mock()
        mock_module.imagedl.ImageClient = Mock(return_value=mock_imagedl_client)

        with patch('importlib.util.find_spec', return_value=mock_spec), \
             patch('importlib.util.module_from_spec', return_value=mock_module):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            # Simulate imagedl not being available
            client.client = None

            # search_and_download should return empty list when client is None
            images = client.search_and_download("test query", max_images=5)
            assert images == []

    @pytest.mark.fast
    def test_search_and_download_no_client(self, temp_output_dir, mock_imagedl_client):
        """Test search_and_download gracefully handles missing client"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_spec = Mock()
        mock_spec.origin = '/path/to/site-packages/imagedl/__init__.py'
        mock_spec.loader.exec_module = Mock()

        mock_module = Mock()
        mock_module.imagedl = Mock()
        mock_module.imagedl.ImageClient = Mock(return_value=mock_imagedl_client)

        with patch('importlib.util.find_spec', return_value=mock_spec), \
             patch('importlib.util.module_from_spec', return_value=mock_module):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            # Force client to None
            client.client = None

            # Should handle gracefully and return empty list
            images = client.search_and_download("test query", max_images=5, entity_name="Test")
            assert isinstance(images, list)
            assert images == []

    @pytest.mark.fast
    def test_init_client_success(self, temp_output_dir, mock_imagedl_client):
        """Test successful imagedl client initialization"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_spec = Mock()
        mock_spec.origin = '/path/to/site-packages/imagedl/__init__.py'
        mock_spec.loader.exec_module = Mock()

        mock_module = Mock()
        mock_module.imagedl = Mock()
        mock_module.imagedl.ImageClient = Mock(return_value=mock_imagedl_client)

        with patch('importlib.util.find_spec', return_value=mock_spec), \
             patch('importlib.util.module_from_spec', return_value=mock_module):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))

            assert client.client is not None


# ============================================================================
# Test _search_with_timeout()
# ============================================================================

class TestSearchWithTimeout:
    """Test threaded search with timeout handling"""

    @pytest.mark.fast
    def test_search_with_timeout_success(self, temp_output_dir, mock_imagedl_client):
        """Test successful search with timeout"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            results = client._search_with_timeout("test keyword", search_limit=10, timeout=5)

            assert len(results) == 2
            assert results[0]['title'] == 'Test Image 1'
            mock_imagedl_client.search.assert_called_once_with(
                keyword="test keyword",
                search_limits_overrides=10
            )

    @pytest.mark.fast
    def test_search_with_timeout_no_client(self, temp_output_dir):
        """Test search when client is None"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = None

            results = client._search_with_timeout("test", search_limit=10, timeout=5)

            assert results == []

    @pytest.mark.fast
    def test_search_with_timeout_error(self, temp_output_dir, mock_imagedl_client):
        """Test search error handling"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_imagedl_client.search.side_effect = Exception("API error")

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            results = client._search_with_timeout("test", search_limit=10, timeout=5)

            assert results == []

    @pytest.mark.slow
    def test_search_with_timeout_hangs(self, temp_output_dir, mock_imagedl_client):
        """Test timeout when search hangs"""
        from src.media_sources.images.google_bing import GoogleBingImageClient
        import time

        # Mock search that hangs
        def slow_search(*args, **kwargs):
            time.sleep(10)  # Longer than timeout
            return []

        mock_imagedl_client.search = slow_search

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            # Use very short timeout
            results = client._search_with_timeout("test", search_limit=10, timeout=0.5)

            assert results == []


# ============================================================================
# Test _reinitialize_client()
# ============================================================================

class TestReinitializeClient:
    """Test client reinitialization"""

    @pytest.mark.fast
    def test_reinitialize_client(self, temp_output_dir):
        """Test client can be reinitialized"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client') as mock_init:
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            mock_init.reset_mock()

            client._reinitialize_client()

            assert client.client is None
            mock_init.assert_called_once()


# ============================================================================
# Test _download_single_image()
# ============================================================================

class TestDownloadSingleImage:
    """Test single image download with requests"""

    @pytest.mark.requires_network
    def test_download_single_image_success_jpg(self, temp_output_dir):
        """Test successful JPG image download"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content = lambda chunk_size: [b'fake_image_data']
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            output_folder = temp_output_dir / "test_folder"
            output_folder.mkdir()

            result = client._download_single_image(
                url="https://example.com/test.jpg",
                output_folder=output_folder,
                index=1,
                timeout=10
            )

            assert result is not None
            assert result.exists()
            assert result.suffix == '.jpg'
            assert 'img_' in result.name

    @pytest.mark.requires_network
    def test_download_single_image_png(self, temp_output_dir):
        """Test PNG image download"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/png'}
        mock_response.iter_content = lambda chunk_size: [b'fake_png_data']
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            output_folder = temp_output_dir / "test_folder"
            output_folder.mkdir()

            result = client._download_single_image(
                url="https://example.com/test.png",
                output_folder=output_folder,
                index=1
            )

            assert result.suffix == '.png'

    @pytest.mark.requires_network
    def test_download_single_image_webp(self, temp_output_dir):
        """Test WebP image download"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/webp'}
        mock_response.iter_content = lambda chunk_size: [b'fake_webp_data']
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            output_folder = temp_output_dir / "test_folder"
            output_folder.mkdir()

            result = client._download_single_image(
                url="https://example.com/test.webp",
                output_folder=output_folder,
                index=1
            )

            assert result.suffix == '.webp'

    @pytest.mark.requires_network
    def test_download_single_image_timeout(self, temp_output_dir):
        """Test download timeout handling"""
        from src.media_sources.images.google_bing import GoogleBingImageClient
        import requests

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', side_effect=requests.Timeout("Timeout")):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            output_folder = temp_output_dir / "test_folder"
            output_folder.mkdir()

            result = client._download_single_image(
                url="https://example.com/test.jpg",
                output_folder=output_folder,
                index=1,
                timeout=1
            )

            assert result is None

    @pytest.mark.requires_network
    def test_download_single_image_invalid_content_type(self, temp_output_dir):
        """Test rejection of non-image content"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            output_folder = temp_output_dir / "test_folder"
            output_folder.mkdir()

            result = client._download_single_image(
                url="https://example.com/notimage.html",
                output_folder=output_folder,
                index=1
            )

            assert result is None

    @pytest.mark.requires_network
    def test_download_single_image_random_naming(self, temp_output_dir):
        """Test that filenames are randomized to avoid sequence detection"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content = lambda chunk_size: [b'data']
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            output_folder = temp_output_dir / "test_folder"
            output_folder.mkdir()

            # Download multiple images
            results = []
            for i in range(5):
                result = client._download_single_image(
                    url=f"https://example.com/test{i}.jpg",
                    output_folder=output_folder,
                    index=i
                )
                if result:
                    results.append(result.name)

            # Check that names are randomized (not sequential)
            assert len(set(results)) == len(results)  # All unique
            assert all('img_' in name for name in results)


# ============================================================================
# Test _cleanup_empty_folders()
# ============================================================================

class TestCleanupEmptyFolders:
    """Test cleanup of empty or image-less folders"""

    @pytest.mark.fast
    def test_cleanup_empty_folder(self, temp_output_dir):
        """Test cleanup of completely empty folder"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))

            empty_folder = temp_output_dir / "empty"
            empty_folder.mkdir()

            client._cleanup_empty_folders([empty_folder])

            assert not empty_folder.exists()

    @pytest.mark.fast
    def test_cleanup_folder_with_small_images(self, temp_output_dir):
        """Test cleanup of folder with only small images"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir), min_size_mb=1.0)

            folder = temp_output_dir / "small_images"
            folder.mkdir()

            # Create small image (< 1MB)
            small_image = folder / "small.jpg"
            small_image.write_bytes(b'x' * 500000)  # 500KB

            client._cleanup_empty_folders([folder])

            # Folder should be deleted
            assert not folder.exists()

    @pytest.mark.fast
    def test_cleanup_preserves_folder_with_large_images(self, temp_output_dir):
        """Test that folders with large images are preserved"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir), min_size_mb=1.0)

            folder = temp_output_dir / "large_images"
            folder.mkdir()

            # Create large image (>= 1MB)
            large_image = folder / "large.jpg"
            large_image.write_bytes(b'x' * (1024 * 1024 + 1))  # 1MB + 1 byte

            client._cleanup_empty_folders([folder])

            # Folder should NOT be deleted
            assert folder.exists()
            assert large_image.exists()

    @pytest.mark.fast
    def test_cleanup_nonexistent_folder(self, temp_output_dir):
        """Test cleanup of non-existent folder (no error)"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))

            nonexistent = temp_output_dir / "doesnotexist"

            # Should not raise error
            client._cleanup_empty_folders([nonexistent])


# ============================================================================
# Test search_and_download() Main Workflow
# ============================================================================

class TestSearchAndDownload:
    """Test main search and download workflow"""

    @pytest.mark.fast
    def test_search_and_download_no_client(self, temp_output_dir):
        """Test when imagedl client is not available"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = None

            results = client.search_and_download(query="test", max_images=5)

            assert results == []

    @pytest.mark.requires_network
    def test_search_and_download_success(self, temp_output_dir, mock_imagedl_client):
        """Test successful search and download"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        # Mock successful search
        mock_imagedl_client.search.return_value = [
            {'candidate_urls': ['https://example.com/img1.jpg']},
            {'candidate_urls': ['https://example.com/img2.jpg']}
        ]

        # Mock successful downloads
        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content = lambda chunk_size: [b'x' * (1024 * 1024 + 1)]  # > 1MB
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            results = client.search_and_download(
                query="test query",
                max_images=2,
                entity_name="TestEntity",
                entity_type="PERSON"
            )

            assert len(results) == 2
            # Check entity metadata was saved
            for result_path in results:
                meta_path = Path(result_path).with_suffix('.entity.json')
                assert meta_path.exists()

                with open(meta_path) as f:
                    meta = json.load(f)
                assert meta['entity_name'] == "TestEntity"
                assert meta['entity_type'] == "PERSON"

    @pytest.mark.fast
    def test_search_and_download_no_results(self, temp_output_dir, mock_imagedl_client):
        """Test when search returns no results"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_imagedl_client.search.return_value = []

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            results = client.search_and_download(query="no results", max_images=5)

            assert results == []

    def test_search_and_download_max_search_time(self, temp_output_dir, mock_imagedl_client):
        """Test max search time limit"""
        from src.media_sources.images.google_bing import GoogleBingImageClient
        import time

        # Mock search that returns results slowly
        call_count = [0]
        def slow_search(*args, **kwargs):
            call_count[0] += 1
            time.sleep(2)  # Simulate slow search
            return [{'candidate_urls': [f'https://example.com/img{call_count[0]}.jpg']}]

        mock_imagedl_client.search = slow_search

        with patch.object(GoogleBingImageClient, '_init_client'):
            client = GoogleBingImageClient(
                output_dir=str(temp_output_dir),
                max_search_time=3  # 3 second limit
            )
            client.client = mock_imagedl_client

            # This should timeout before finding all images
            results = client.search_and_download(query="test", max_images=10)

            # Should stop early due to timeout
            assert len(results) < 10

    @pytest.mark.requires_network
    def test_search_and_download_filters_small_images(self, temp_output_dir, mock_imagedl_client):
        """Test that small images are filtered out"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_imagedl_client.search.return_value = [
            {'candidate_urls': ['https://example.com/img1.jpg']},
            {'candidate_urls': ['https://example.com/img2.jpg']},
            {'candidate_urls': ['https://example.com/img3.jpg']}
        ]

        # Mock downloads: first is large, second is small, third is large
        responses = []
        for size_mb in [1.5, 0.5, 2.0]:
            resp = Mock()
            resp.headers = {'content-type': 'image/jpeg'}
            resp.iter_content = lambda chunk_size, s=size_mb: [b'x' * int(s * 1024 * 1024)]
            resp.raise_for_status = Mock()
            responses.append(resp)

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', side_effect=responses):

            client = GoogleBingImageClient(
                output_dir=str(temp_output_dir),
                min_size_mb=1.0
            )
            client.client = mock_imagedl_client

            results = client.search_and_download(query="test", max_images=3)

            # Should only get 2 large images (1st and 3rd)
            assert len(results) == 2

    @pytest.mark.requires_network
    def test_search_and_download_skips_bracketed_paths(self, temp_output_dir, mock_imagedl_client):
        """Test that paths with brackets are skipped (image sequence notation)"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        mock_imagedl_client.search.return_value = [
            {'candidate_urls': ['https://example.com/img1.jpg']}
        ]

        # Mock download that would create bracketed path
        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content = lambda chunk_size: [b'x' * (1024 * 1024 + 1)]
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response), \
             patch.object(GoogleBingImageClient, '_download_single_image') as mock_dl:

            # Mock download returns path with brackets
            bracketed_path = temp_output_dir / "test[001].jpg"
            bracketed_path.write_bytes(b'x' * (1024 * 1024 + 1))
            mock_dl.return_value = bracketed_path

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            results = client.search_and_download(query="test", max_images=1)

            # Should be empty (bracketed path filtered out)
            assert results == []

    @pytest.mark.requires_network
    def test_search_and_download_query_variations(self, temp_output_dir, mock_imagedl_client):
        """Test search_until_found tries query variations"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        search_calls = []
        def track_searches(keyword, **kwargs):
            search_calls.append(keyword)
            # First search returns nothing, second returns result
            if len(search_calls) == 1:
                return []
            return [{'candidate_urls': ['https://example.com/img1.jpg']}]

        mock_imagedl_client.search = track_searches

        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content = lambda chunk_size: [b'x' * (1024 * 1024 + 1)]
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(
                output_dir=str(temp_output_dir),
                search_until_found=True
            )
            client.client = mock_imagedl_client

            results = client.search_and_download(
                query="original query",
                max_images=1,
                entity_name="Entity"
            )

            # Should have tried multiple query variations
            assert len(search_calls) >= 2
            assert search_calls[0] == "original query"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestGoogleBingEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.requires_network
    def test_consecutive_search_failures_trigger_reinit(self, temp_output_dir, mock_imagedl_client):
        """Test that consecutive failures trigger client reinitialization"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        # First 2 searches fail, 3rd succeeds
        call_count = [0]
        def failing_searches(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return []  # No results = failure
            return [{'candidate_urls': ['https://example.com/img.jpg']}]

        mock_imagedl_client.search = failing_searches

        mock_response = Mock()
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.iter_content = lambda chunk_size: [b'x' * (1024 * 1024 + 1)]
        mock_response.raise_for_status = Mock()

        with patch.object(GoogleBingImageClient, '_init_client') as mock_init, \
             patch('requests.get', return_value=mock_response):

            client = GoogleBingImageClient(
                output_dir=str(temp_output_dir),
                search_until_found=True
            )
            client.client = mock_imagedl_client

            # Track reinitialization
            reinit_count = 0
            original_reinit = client._reinitialize_client
            def track_reinit():
                nonlocal reinit_count
                reinit_count += 1
                # Don't actually reinit (keep mock_imagedl_client)
            client._reinitialize_client = track_reinit

            client.search_and_download(query="test", max_images=1)

            # After 2 consecutive failures, should reinitialize
            assert reinit_count >= 1

    @pytest.mark.fast
    def test_skips_duplicate_urls(self, temp_output_dir, mock_imagedl_client):
        """Test that duplicate URLs are skipped"""
        from src.media_sources.images.google_bing import GoogleBingImageClient

        # Return same URL multiple times
        mock_imagedl_client.search.return_value = [
            {'candidate_urls': ['https://example.com/same.jpg']},
            {'candidate_urls': ['https://example.com/same.jpg']},
            {'candidate_urls': ['https://example.com/same.jpg']}
        ]

        download_count = [0]
        def track_downloads(*args, **kwargs):
            download_count[0] += 1
            # Create unique file for each call
            folder = kwargs['output_folder']
            filepath = folder / f"img_{download_count[0]}.jpg"
            filepath.write_bytes(b'x' * (1024 * 1024 + 1))
            return filepath

        with patch.object(GoogleBingImageClient, '_init_client'), \
             patch.object(GoogleBingImageClient, '_download_single_image', side_effect=track_downloads):

            client = GoogleBingImageClient(output_dir=str(temp_output_dir))
            client.client = mock_imagedl_client

            results = client.search_and_download(query="test", max_images=3)

            # Should only download once (duplicate URLs skipped)
            assert download_count[0] == 1
            assert len(results) == 1
