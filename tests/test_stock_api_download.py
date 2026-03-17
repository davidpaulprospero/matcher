"""
Additional test suite for Stock API download functionality.

Tests coverage for download methods in:
- src/media_sources/images/pexels.py
- src/media_sources/images/pixabay.py
- src/media_sources/images/unsplash.py

Created: January 10, 2026
Session: 13 Phase 2 - Download Tests
"""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

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
# Test Pexels Download
# ============================================================================

class TestPexelsImageDownload:
    """Test Pexels image download functionality"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample ImageResult"""
        return ImageResult(
            source="pexels",
            id=12345678,
            url="https://www.pexels.com/photo/12345678/",
            download_url="https://images.pexels.com/photos/12345678/photo.jpg",
            photographer="Test Photographer",
            width=1920,
            height=1080,
            description="Test image",
            tags=["test", "image"]
        )

    @pytest.mark.fast
    def test_download_image_success(self, mock_config, temp_output_dir, sample_image):
        """Test successful image download"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.headers = {'content-length': '5000000'}  # 5MB
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(5000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        assert result is not None
        assert Path(result).exists()
        assert Path(result).stat().st_size == 5000000
        assert len(client.downloaded_files) == 1

    @pytest.mark.fast
    def test_download_image_skips_existing(self, mock_config, temp_output_dir, sample_image):
        """Test download skips existing files"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        # Create existing file
        existing_path = Path(temp_output_dir) / "12345678.jpg"
        existing_path.write_bytes(b'x' * 5000000)  # 5MB

        result = client.download_image(sample_image, check_size=False)

        assert result == str(existing_path)
        # No HTTP request should be made
        assert len(client.downloaded_files) == 0

    @pytest.mark.fast
    def test_download_image_too_small_precheck(self, mock_config, temp_output_dir, sample_image):
        """Test download skips images that are too small (HEAD request)"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=5.0
        )

        # Mock HEAD response with small size
        mock_head = Mock()
        mock_head.headers = {'content-length': '1000000'}  # 1MB

        with patch.object(client.session, 'head', return_value=mock_head):
            result = client.download_image(sample_image, check_size=True)

        assert result is None

    @pytest.mark.fast
    def test_download_image_too_small_content_length(self, mock_config, temp_output_dir, sample_image):
        """Test download skips based on content-length header"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=5.0
        )

        # Mock response with small content-length
        mock_response = Mock()
        mock_response.headers = {'content-length': '2000000'}  # 2MB

        with patch.object(client.session, 'head', side_effect=Exception("No HEAD")):
            with patch.object(client.session, 'get', return_value=mock_response):
                result = client.download_image(sample_image, check_size=True)

        assert result is None

    @pytest.mark.fast
    def test_download_image_too_small_actual_size(self, mock_config, temp_output_dir, sample_image):
        """Test download checks actual downloaded size"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=5.0
        )

        # Mock response with no content-length but small actual size
        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(1000)]  # 1MB

        with patch.object(client.session, 'head', side_effect=Exception("No HEAD")):
            with patch.object(client.session, 'get', return_value=mock_response):
                result = client.download_image(sample_image, check_size=True)

        assert result is None

    @pytest.mark.fast
    def test_download_image_with_entity_metadata(self, mock_config, temp_output_dir, sample_image):
        """Test download saves entity metadata"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {'content-length': '5000000'}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(5000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(
                sample_image,
                entity_name="Paris",
                entity_type="Location",
                check_size=False
            )

        # Check entity metadata file exists
        meta_path = Path(result).with_suffix('.entity.json')
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata['entity_name'] == "Paris"
        assert metadata['entity_type'] == "Location"

    @pytest.mark.fast
    def test_download_image_network_error(self, mock_config, temp_output_dir, sample_image):
        """Test download handles network errors"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=Exception("Network error")):
            result = client.download_image(sample_image, check_size=False)

        assert result is None
        assert sample_image.id in client.failed_downloads

    @pytest.mark.fast
    def test_download_image_http_error(self, mock_config, temp_output_dir, sample_image):
        """Test download handles HTTP errors"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        assert result is None
        assert sample_image.id in client.failed_downloads

    @pytest.mark.fast
    def test_download_image_creates_metadata_file(self, mock_config, temp_output_dir, sample_image):
        """Test download creates metadata JSON file"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {'content-length': '5000000'}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(5000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        # Check metadata file
        meta_path = Path(result).with_suffix('.meta.json')
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata['source'] == 'pexels'
        assert metadata['photographer'] == 'Test Photographer'
        assert metadata['is_stock_image'] is True


# ============================================================================
# Test Pixabay Download
# ============================================================================

class TestPixabayImageDownload:
    """Test Pixabay image download functionality"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample ImageResult"""
        return ImageResult(
            source="pixabay",
            id=98765432,
            url="https://pixabay.com/photos/98765432/",
            download_url="https://pixabay.com/get/98765432.jpg",
            photographer="Test User",
            width=1920,
            height=1080,
            description="Pixabay test",
            tags=["nature", "landscape"]
        )

    @pytest.mark.fast
    def test_download_image_success(self, mock_config, temp_output_dir, sample_image):
        """Test successful Pixabay image download"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {'content-length': '6000000'}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(6000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        assert result is not None
        assert Path(result).exists()
        assert len(client.downloaded_files) == 1

    @pytest.mark.fast
    def test_download_tracks_failed_downloads(self, mock_config, temp_output_dir, sample_image):
        """Test Pixabay tracks failed downloads"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=Exception("Error")):
            result = client.download_image(sample_image)

        assert result is None
        assert sample_image.id in client.failed_downloads


# ============================================================================
# Test Unsplash Download
# ============================================================================

class TestUnsplashImageDownload:
    """Test Unsplash image download functionality"""

    @pytest.fixture
    def sample_image(self):
        """Create a sample ImageResult"""
        return ImageResult(
            source="unsplash",
            id="abc123xyz",
            url="https://unsplash.com/photos/abc123xyz",
            download_url="https://images.unsplash.com/photo-abc123xyz",
            photographer="Unsplash User",
            width=3840,
            height=2160,
            description="Unsplash photo",
            tags=["mountains", "snow"]
        )

    @pytest.mark.fast
    def test_download_image_success(self, mock_config, temp_output_dir, sample_image):
        """Test successful Unsplash image download"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {'content-length': '8000000'}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(8000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        assert result is not None
        assert Path(result).exists()

    @pytest.mark.fast
    def test_download_short_filename_generation(self, mock_config, temp_output_dir, sample_image):
        """Test Unsplash generates short filenames from ID"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(5000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        # Check filename is based on ID (last 8 chars for long IDs)
        filename = Path(result).name
        # ID is truncated to last 8 chars if > 8
        expected_id = sample_image.id[-8:] if len(sample_image.id) > 8 else sample_image.id
        assert expected_id in filename


# ============================================================================
# Test Search and Download Integration
# ============================================================================

class TestSearchAndDownload:
    """Test integrated search_and_download functionality"""

    @pytest.mark.fast
    def test_pexels_search_and_download(self, mock_config, temp_output_dir):
        """Test Pexels search_and_download integration"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        # Mock search results
        mock_search = Mock(return_value=[
            ImageResult(
                source="pexels",
                id=111,
                url="https://pexels.com/111",
                download_url="https://images.pexels.com/111.jpg",
                photographer="User1",
                width=1920,
                height=1080,
                description="Test",
                tags=["nature"]
            )
        ])

        # Mock download
        mock_download = Mock(return_value=str(Path(temp_output_dir) / "111.jpg"))

        with patch.object(client, 'search', mock_search):
            with patch.object(client, 'download_image', mock_download):
                results = client.search_and_download("nature", max_images=1)

        assert len(results) == 1
        assert results[0] == str(Path(temp_output_dir) / "111.jpg")
        mock_search.assert_called_once()
        mock_download.assert_called_once()

    @pytest.mark.fast
    def test_search_and_download_with_entity(self, mock_config, temp_output_dir):
        """Test search_and_download with entity metadata"""
        client = PexelsImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_search = Mock(return_value=[
            ImageResult(
                source="pexels",
                id=222,
                url="https://pexels.com/222",
                download_url="https://images.pexels.com/222.jpg",
                photographer="User2",
                width=1920,
                height=1080,
                description="Paris",
                tags=["city"]
            )
        ])

        mock_download = Mock(return_value=str(Path(temp_output_dir) / "222.jpg"))

        with patch.object(client, 'search', mock_search):
            with patch.object(client, 'download_image', mock_download):
                results = client.search_and_download(
                    "Paris",
                    max_images=1,
                    entity_name="Paris",
                    entity_type="Location"
                )

        # Check entity metadata was passed to download
        download_call = mock_download.call_args
        assert download_call.kwargs.get('entity_name') == "Paris"
        assert download_call.kwargs.get('entity_type') == "Location"


class TestPixabayAdditionalDownload:
    """Additional Pixabay download tests"""

    @pytest.fixture
    def sample_image(self):
        return ImageResult(
            source="pixabay",
            id=55555,
            url="https://pixabay.com/55555",
            download_url="https://pixabay.com/get/55555.jpg",
            photographer="User",
            width=1920,
            height=1080,
            description="Test",
            tags=["test"]
        )

    @pytest.mark.fast
    def test_download_with_size_check(self, mock_config, temp_output_dir, sample_image):
        """Test download with size checking"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=3.0
        )

        mock_response = Mock()
        mock_response.headers = {'content-length': '4000000'}  # 4MB
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(4000)]

        with patch.object(client.session, 'head', side_effect=Exception("No HEAD")):
            with patch.object(client.session, 'get', return_value=mock_response):
                result = client.download_image(sample_image, check_size=True)

        assert result is not None

    @pytest.mark.fast
    def test_download_creates_metadata(self, mock_config, temp_output_dir, sample_image):
        """Test Pixabay creates metadata file"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(5000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        meta_path = Path(result).with_suffix('.meta.json')
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata['source'] == 'pixabay'

    @pytest.mark.fast
    def test_download_existing_file_too_small(self, mock_config, temp_output_dir, sample_image):
        """Test skips existing file that's too small"""
        client = PixabayImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=5.0
        )

        # Create small existing file
        existing_path = Path(temp_output_dir) / "55555.jpg"
        existing_path.write_bytes(b'x' * 1000000)  # 1MB

        result = client.download_image(sample_image, check_size=True)

        # Should return None (too small)
        assert result is None


class TestUnsplashAdditionalDownload:
    """Additional Unsplash download tests"""

    @pytest.fixture
    def sample_image(self):
        return ImageResult(
            source="unsplash",
            id="test123",
            url="https://unsplash.com/test123",
            download_url="https://images.unsplash.com/test123",
            photographer="User",
            width=1920,
            height=1080,
            description="Test",
            tags=["test"]
        )

    @pytest.mark.fast
    def test_download_with_size_check(self, mock_config, temp_output_dir, sample_image):
        """Test Unsplash download with size checking"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=2.0
        )

        mock_response = Mock()
        mock_response.headers = {'content-length': '3000000'}  # 3MB
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(3000)]

        with patch.object(client.session, 'head', side_effect=Exception("No HEAD")):
            with patch.object(client.session, 'get', return_value=mock_response):
                result = client.download_image(sample_image, check_size=True)

        assert result is not None

    @pytest.mark.fast
    def test_download_creates_metadata(self, mock_config, temp_output_dir, sample_image):
        """Test Unsplash creates metadata file"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key"
        )

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [b'x' * 1000 for _ in range(5000)]

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.download_image(sample_image, check_size=False)

        meta_path = Path(result).with_suffix('.meta.json')
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata['source'] == 'unsplash'

    @pytest.mark.fast
    def test_download_existing_file_too_small(self, mock_config, temp_output_dir, sample_image):
        """Test skips existing file that's too small"""
        client = UnsplashImageClient(
            config=mock_config,
            output_dir=temp_output_dir,
            api_key="test_key",
            min_size_mb=5.0
        )

        # Create small existing file
        existing_path = Path(temp_output_dir) / "test123.jpg"
        existing_path.write_bytes(b'x' * 1000000)  # 1MB

        result = client.download_image(sample_image, check_size=True)

        # Should return None (too small)
        assert result is None
