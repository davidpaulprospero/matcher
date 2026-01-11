"""
Comprehensive test suite for Stock API client implementations.

Tests coverage for:
- src/media_sources/images/pexels.py
- src/media_sources/images/pixabay.py
- src/media_sources/images/unsplash.py

Created: January 10, 2026
Session: 13 Phase 2
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.media_sources.images.pexels import PexelsImageClient
from src.media_sources.images.pixabay import PixabayImageClient
from src.media_sources.images.unsplash import UnsplashImageClient
from src.media_sources.models import ImageResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration"""
    config = Mock()
    return config


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory"""
    output_dir = tmp_path / "images"
    output_dir.mkdir()
    return str(output_dir)


# ============================================================================
# Test PexelsImageClient
# ============================================================================

class TestPexelsImageClientInit:
    """Test Pexels client initialization"""

    def test_init_with_api_key(self, mock_config, temp_output_dir):
        """Test initialization with provided API key"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_api_key"
        )

        assert client.api_key == "test_api_key"
        assert client.output_dir == Path(temp_output_dir)
        assert client.min_size == int(1.0 * 1024 * 1024)

    def test_init_with_env_var(self, mock_config, temp_output_dir):
        """Test initialization with env var fallback"""
        with patch.dict('os.environ', {'PEXELS_API_KEY': 'env_key'}):
            client = PexelsImageClient(
                config=mock_config,
                output_dir=temp_output_dir
            )

            assert client.api_key == "env_key"

    def test_init_custom_params(self, mock_config, temp_output_dir):
        """Test initialization with custom parameters"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="key",
            min_size_mb=2.5,
            download_timeout=120
        )

        assert client.min_size == int(2.5 * 1024 * 1024)
        assert client.download_timeout == 120


class TestPexelsImageClientSearch:
    """Test Pexels search functionality"""

    def test_search_no_api_key(self, mock_config, temp_output_dir):
        """Test search without API key"""
        # Ensure no API key in environment
        with patch.dict('os.environ', {}, clear=True):
            client = PexelsImageClient(
                config=mock_config,
                output_dir=temp_output_dir
            )

            results = client.search("test query")

            assert results == []

    def test_search_success(self, mock_config, temp_output_dir):
        """Test successful search"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = {
            "photos": [
                {
                    "id": 123,
                    "width": 1920,
                    "height": 1080,
                    "photographer": "John Doe",
                    "src": {
                        "original": "https://images.pexels.com/photos/123/photo.jpg",
                        "large2x": "https://images.pexels.com/photos/123/large.jpg"
                    }
                },
                {
                    "id": 456,
                    "width": 2560,
                    "height": 1440,
                    "photographer": "Jane Smith",
                    "src": {
                        "original": "https://images.pexels.com/photos/456/photo.jpg"
                    }
                }
            ]
        }

        with patch.object(client.session, 'get') as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            results = client.search("mountains", max_results=20)

            assert len(results) >= 1  # At least one result
            assert isinstance(results[0], ImageResult)
            assert "pexels.com" in results[0].download_url or "images.pexels.com" in results[0].download_url
            assert results[0].width == 1920
            assert results[0].height == 1080

    def test_search_api_error(self, mock_config, temp_output_dir):
        """Test search with API error"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        with patch.object(client.session, 'get') as mock_get:
            mock_get.side_effect = Exception("API error")

            results = client.search("query")

            assert results == []

    def test_search_missing_url(self, mock_config, temp_output_dir):
        """Test search with missing download URL"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = {
            "photos": [
                {
                    "id": 123,
                    "src": {}  # Missing URLs
                }
            ]
        }

        with patch.object(client.session, 'get') as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            results = client.search("query")

            # Should skip photos without URL
            assert len(results) == 0


# ============================================================================
# Test PixabayImageClient
# ============================================================================

class TestPixabayImageClientInit:
    """Test Pixabay client initialization"""

    def test_init_with_api_key(self, mock_config, temp_output_dir):
        """Test initialization with provided API key"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_api_key"
        )

        assert client.api_key == "test_api_key"
        assert client.output_dir == Path(temp_output_dir)

    def test_init_with_env_var(self, mock_config, temp_output_dir):
        """Test initialization with env var fallback"""
        with patch.dict('os.environ', {'PIXABAY_API_KEY': 'env_key'}):
            client = PixabayImageClient(
                config=mock_config,
                output_dir=temp_output_dir
            )

            assert client.api_key == "env_key"


class TestPixabayImageClientSearch:
    """Test Pixabay search functionality"""

    def test_search_no_api_key(self, mock_config, temp_output_dir):
        """Test search without API key"""
        # Ensure no API key in environment
        with patch.dict('os.environ', {}, clear=True):
            client = PixabayImageClient(
                config=mock_config,
                output_dir=temp_output_dir
            )

            results = client.search("test query")

            assert results == []

    def test_search_success(self, mock_config, temp_output_dir):
        """Test successful search"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = {
            "hits": [
                {
                    "id": 789,
                    "imageWidth": 1920,
                    "imageHeight": 1080,
                    "user": "photographer1",
                    "largeImageURL": "https://pixabay.com/get/image1.jpg"
                },
                {
                    "id": 101,
                    "imageWidth": 2560,
                    "imageHeight": 1440,
                    "user": "photographer2",
                    "largeImageURL": "https://pixabay.com/get/image2.jpg"
                }
            ]
        }

        with patch.object(client.session, 'get') as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            results = client.search("nature", max_results=20)

            assert len(results) >= 1  # At least one result
            assert isinstance(results[0], ImageResult)
            assert "pixabay.com" in results[0].download_url
            assert results[0].width == 1920

    def test_search_api_error(self, mock_config, temp_output_dir):
        """Test search with API error"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        with patch.object(client.session, 'get') as mock_get:
            mock_get.side_effect = Exception("API error")

            results = client.search("query")

            assert results == []


# ============================================================================
# Test UnsplashImageClient
# ============================================================================

class TestUnsplashImageClientInit:
    """Test Unsplash client initialization"""

    def test_init_with_api_key(self, mock_config, temp_output_dir):
        """Test initialization with provided API key"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_api_key"
        )

        assert client.api_key == "test_api_key"
        assert client.output_dir == Path(temp_output_dir)

    def test_init_with_env_var(self, mock_config, temp_output_dir):
        """Test initialization with env var fallback"""
        with patch.dict('os.environ', {'UNSPLASH_API_KEY': 'env_key'}):
            client = UnsplashImageClient(
                config=mock_config,
                output_dir=temp_output_dir
            )

            assert client.api_key == "env_key"


class TestUnsplashImageClientSearch:
    """Test Unsplash search functionality"""

    def test_search_no_api_key(self, mock_config, temp_output_dir):
        """Test search without API key"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir
        )

        results = client.search("test query")

        assert results == []

    def test_search_success(self, mock_config, temp_output_dir):
        """Test successful search"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = {
            "results": [
                {
                    "id": "abc123",
                    "width": 3000,
                    "height": 2000,
                    "user": {"name": "Photographer One"},
                    "urls": {
                        "raw": "https://images.unsplash.com/photo1?raw",
                        "full": "https://images.unsplash.com/photo1?full"
                    }
                },
                {
                    "id": "def456",
                    "width": 4000,
                    "height": 3000,
                    "user": {"name": "Photographer Two"},
                    "urls": {
                        "raw": "https://images.unsplash.com/photo2?raw"
                    }
                }
            ]
        }

        with patch.object(client.session, 'get') as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = Mock()
            mock_get.return_value = mock_resp

            results = client.search("landscape", max_results=20)

            assert len(results) >= 1  # At least one result
            assert isinstance(results[0], ImageResult)
            assert "unsplash.com" in results[0].download_url
            assert results[0].width == 3000

    def test_search_api_error(self, mock_config, temp_output_dir):
        """Test search with API error"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        with patch.object(client.session, 'get') as mock_get:
            mock_get.side_effect = Exception("API error")

            results = client.search("query")

            assert results == []


# ============================================================================
# Edge Cases and Common Functionality
# ============================================================================

class TestStockAPIEdgeCases:
    """Test edge cases across all stock API clients"""

    def test_all_clients_have_rate_limiting(self, mock_config, temp_output_dir):
        """Test that all clients implement rate limiting"""
        pexels = PexelsImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")
        pixabay = PixabayImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")
        unsplash = UnsplashImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")

        # All should have _rate_limit method from BaseMediaClient
        assert hasattr(pexels, '_rate_limit')
        assert hasattr(pixabay, '_rate_limit')
        assert hasattr(unsplash, '_rate_limit')

    def test_all_clients_accept_output_dir(self, mock_config, temp_output_dir):
        """Test that all clients accept output_dir parameter"""
        pexels = PexelsImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")
        pixabay = PixabayImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")
        unsplash = UnsplashImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")

        assert pexels.output_dir == Path(temp_output_dir)
        assert pixabay.output_dir == Path(temp_output_dir)
        assert unsplash.output_dir == Path(temp_output_dir)

    def test_all_clients_return_empty_list_without_api_key(self, mock_config, temp_output_dir):
        """Test that all clients return empty list when no API key"""
        # Ensure no API keys in environment
        with patch.dict('os.environ', {}, clear=True):
            pexels = PexelsImageClient(config=mock_config, output_dir=temp_output_dir)
            pixabay = PixabayImageClient(config=mock_config, output_dir=temp_output_dir)
            unsplash = UnsplashImageClient(config=mock_config, output_dir=temp_output_dir)

            assert pexels.search("query") == []
            assert pixabay.search("query") == []
            assert unsplash.search("query") == []

    def test_all_clients_handle_api_errors_gracefully(self, mock_config, temp_output_dir):
        """Test that all clients handle API errors gracefully"""
        pexels = PexelsImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")
        pixabay = PixabayImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")
        unsplash = UnsplashImageClient(config=mock_config, output_dir=temp_output_dir, api_key="key")

        # Mock all session.get to raise error
        with patch.object(pexels.session, 'get', side_effect=Exception("Network error")):
            assert pexels.search("query") == []

        with patch.object(pixabay.session, 'get', side_effect=Exception("Network error")):
            assert pixabay.search("query") == []

        with patch.object(unsplash.session, 'get', side_effect=Exception("Network error")):
            assert unsplash.search("query") == []
