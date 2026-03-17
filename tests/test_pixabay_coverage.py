"""
Test coverage for src/media_sources/images/pixabay.py

Target: Cover key missed lines to improve coverage.

Covers:
- Line 103: continue when no download_url
- Line 164: return existing file path when size OK
- Lines 175-176: pre-check HEAD request shows file too small
- Lines 187-188: content-length header shows file too small
- Lines 201-203: file too small after download
- Lines 224-226: entity metadata saving
- Lines 266-283: search_and_download method
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock
import json


def create_mock_config():
    """Create a mock config object."""
    config = MagicMock()
    return config


def create_mock_image_result(download_url="http://example.com/img.jpg", image_id="12345"):
    """Create a mock ImageResult object."""
    from src.media_sources.models import ImageResult
    return ImageResult(
        id=f"pixabay_{image_id}",
        source="pixabay",
        url="http://pixabay.com/photo/12345",
        download_url=download_url,
        width=1920,
        height=1080,
        photographer="TestUser",
        description="Test image",
        tags=["test", "pixabay"],
        estimated_size=1000000
    )


class TestPixabayImageClientSearch:
    """Test search method."""

    @pytest.mark.fast
    def test_search_no_api_key(self, tmp_path):
        """Test search returns empty when no API key."""
        from src.media_sources.images.pixabay import PixabayImageClient

        with patch.dict('os.environ', {}, clear=True):
            client = PixabayImageClient(
                config=create_mock_config(),
                output_dir=str(tmp_path),
                api_key=None
            )

        result = client.search("test query")
        assert result == []

    @pytest.mark.fast
    def test_search_success(self, tmp_path):
        """Test successful search returns results."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "id": 12345,
                    "largeImageURL": "http://pixabay.com/image.jpg",
                    "pageURL": "http://pixabay.com/photo/12345",
                    "imageWidth": 1920,
                    "imageHeight": 1080,
                    "user": "TestUser",
                    "tags": "nature, landscape"
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.search("nature")

        assert len(result) == 1
        assert result[0].source == "pixabay"

    @pytest.mark.fast
    def test_search_no_download_url(self, tmp_path):
        """Test line 103: skip hits without download_url."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [
                {
                    "id": 12345,
                    "largeImageURL": "",  # Empty URL - line 103
                    "pageURL": "http://pixabay.com/photo/12345",
                    "imageWidth": 1920,
                    "imageHeight": 1080,
                    "user": "TestUser",
                    "tags": "test"
                },
                {
                    "id": 12346,
                    "largeImageURL": "http://pixabay.com/image.jpg",  # Valid URL
                    "pageURL": "http://pixabay.com/photo/12346",
                    "imageWidth": 1920,
                    "imageHeight": 1080,
                    "user": "TestUser",
                    "tags": "test"
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.session, 'get', return_value=mock_response):
            result = client.search("test")

        # Should only return the one with valid URL
        assert len(result) == 1
        assert result[0].id == "pixabay_12346"

    @pytest.mark.fast
    def test_search_exception(self, tmp_path):
        """Test search handles exceptions."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client.session, 'get', side_effect=Exception("API error")):
            result = client.search("test")

        assert result == []


class TestPixabayImageClientDownload:
    """Test download_image method."""

    @pytest.mark.fast
    def test_download_existing_file_valid_size(self, tmp_path):
        """Test line 164: return path when existing file has valid size."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.001  # 1KB min
        )

        # Create existing file with valid size
        image = create_mock_image_result()
        short_id = str(image.id)[-8:]
        filename = f"{short_id}.jpg"
        existing_path = tmp_path / filename
        existing_path.write_bytes(b"x" * 2000)  # 2KB > 1KB min

        result = client.download_image(image, check_size=True)

        # Line 164: should return existing path
        assert result == str(existing_path)

    @pytest.mark.fast
    def test_download_existing_file_too_small(self, tmp_path):
        """Test existing file too small returns None."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB min
        )

        # Create existing file with small size
        image = create_mock_image_result()
        short_id = str(image.id)[-8:]
        filename = f"{short_id}.jpg"
        existing_path = tmp_path / filename
        existing_path.write_bytes(b"x" * 100)  # 100 bytes < 1MB

        result = client.download_image(image, check_size=True)

        assert result is None

    @pytest.mark.fast
    def test_download_head_request_too_small(self, tmp_path):
        """Test lines 175-176: HEAD request shows file too small."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB min
        )

        image = create_mock_image_result()

        # Mock HEAD response with small content-length
        mock_head_response = MagicMock()
        mock_head_response.headers = {'content-length': '1000'}  # 1KB < 1MB

        with patch.object(client.session, 'head', return_value=mock_head_response):
            result = client.download_image(image, check_size=True)

        # Lines 175-176: should return None
        assert result is None

    @pytest.mark.fast
    def test_download_content_length_too_small(self, tmp_path):
        """Test lines 187-188: GET content-length shows file too small."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB min
        )

        image = create_mock_image_result()

        # Mock HEAD to pass
        mock_head_response = MagicMock()
        mock_head_response.headers = {'content-length': '2000000'}  # 2MB - passes HEAD

        # Mock GET with small content-length in response
        mock_get_response = MagicMock()
        mock_get_response.headers = {'content-length': '1000'}  # 1KB < 1MB
        mock_get_response.raise_for_status = MagicMock()

        with patch.object(client.session, 'head', return_value=mock_head_response):
            with patch.object(client.session, 'get', return_value=mock_get_response):
                result = client.download_image(image, check_size=True)

        # Lines 187-188: should return None
        assert result is None

    @pytest.mark.fast
    def test_download_actual_size_too_small(self, tmp_path):
        """Test lines 201-203: downloaded file too small after download."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB min
        )

        image = create_mock_image_result()

        # Mock HEAD to pass
        mock_head_response = MagicMock()
        mock_head_response.headers = {}  # No content-length, so HEAD check passes

        # Mock GET with no content-length, but small actual data
        mock_get_response = MagicMock()
        mock_get_response.headers = {}  # No content-length
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.iter_content.return_value = [b"x" * 100]  # Only 100 bytes

        with patch.object(client.session, 'head', return_value=mock_head_response):
            with patch.object(client.session, 'get', return_value=mock_get_response):
                result = client.download_image(image, check_size=True)

        # Lines 201-203: should return None and delete temp file
        assert result is None

    @pytest.mark.fast
    def test_download_with_entity_metadata(self, tmp_path):
        """Test lines 224-226: entity metadata saved."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001  # Very small min for test
        )

        image = create_mock_image_result()

        # Mock successful download
        mock_head_response = MagicMock()
        mock_head_response.headers = {}

        mock_get_response = MagicMock()
        mock_get_response.headers = {'content-length': '2000'}
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.iter_content.return_value = [b"x" * 2000]

        with patch.object(client.session, 'head', return_value=mock_head_response):
            with patch.object(client.session, 'get', return_value=mock_get_response):
                result = client.download_image(
                    image,
                    check_size=True,
                    entity_name="Test Entity",  # Line 224
                    entity_type="PERSON"
                )

        assert result is not None
        # Check entity metadata file exists (line 226)
        meta_path = Path(result).with_suffix('.entity.json')
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)
        assert meta['entity_name'] == "Test Entity"
        assert meta['entity_type'] == "PERSON"

    @pytest.mark.fast
    def test_download_without_entity_metadata(self, tmp_path):
        """Test lines 227-228: regular metadata saved when no entity."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001
        )

        image = create_mock_image_result()

        mock_head_response = MagicMock()
        mock_head_response.headers = {}

        mock_get_response = MagicMock()
        mock_get_response.headers = {'content-length': '2000'}
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.iter_content.return_value = [b"x" * 2000]

        with patch.object(client.session, 'head', return_value=mock_head_response):
            with patch.object(client.session, 'get', return_value=mock_get_response):
                result = client.download_image(
                    image,
                    check_size=True,
                    entity_name="",  # No entity
                    entity_type=""
                )

        assert result is not None
        # Check regular metadata file exists (line 228)
        meta_path = Path(result).with_suffix('.meta.json')
        assert meta_path.exists()

    @pytest.mark.fast
    def test_download_exception(self, tmp_path):
        """Test download handles exceptions."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        image = create_mock_image_result()

        # Mock HEAD to raise exception
        with patch.object(client.session, 'head', side_effect=Exception("Network error")):
            with patch.object(client.session, 'get', side_effect=Exception("Network error")):
                result = client.download_image(image)

        assert result is None
        assert image.id in client.failed_downloads

    @pytest.mark.fast
    def test_download_no_check_size(self, tmp_path):
        """Test download without size checking."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=1.0  # 1MB min, but we'll skip check
        )

        image = create_mock_image_result()

        mock_get_response = MagicMock()
        mock_get_response.headers = {}
        mock_get_response.raise_for_status = MagicMock()
        mock_get_response.iter_content.return_value = [b"x" * 100]  # Small file

        with patch.object(client.session, 'get', return_value=mock_get_response):
            result = client.download_image(image, check_size=False)  # No size check

        # Should succeed even with small file
        assert result is not None


class TestPixabaySearchAndDownload:
    """Test search_and_download method (lines 266-283)."""

    @pytest.mark.fast
    def test_search_and_download_success(self, tmp_path):
        """Test search_and_download returns downloaded paths."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_size_mb=0.0001
        )

        # Mock search
        mock_results = [
            create_mock_image_result(image_id="111"),
            create_mock_image_result(image_id="222"),
            create_mock_image_result(image_id="333"),
        ]

        with patch.object(client, 'search', return_value=mock_results):
            # Mock download
            download_count = [0]

            def mock_download(image, check_size=True, entity_name="", entity_type=""):
                download_count[0] += 1
                return f"/path/to/image{download_count[0]}.jpg"

            with patch.object(client, 'download_image', side_effect=mock_download):
                result = client.search_and_download("test", max_images=2)

        # Should download up to max_images
        assert len(result) == 2

    @pytest.mark.fast
    def test_search_and_download_with_entity(self, tmp_path):
        """Test search_and_download passes entity info."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_results = [create_mock_image_result()]
        entity_params = []

        with patch.object(client, 'search', return_value=mock_results):
            def mock_download(image, check_size=True, entity_name="", entity_type=""):
                entity_params.append((entity_name, entity_type))
                return "/path/to/image.jpg"

            with patch.object(client, 'download_image', side_effect=mock_download):
                result = client.search_and_download(
                    "test",
                    max_images=1,
                    entity_name="Test Entity",
                    entity_type="PERSON"
                )

        assert len(entity_params) == 1
        assert entity_params[0] == ("Test Entity", "PERSON")

    @pytest.mark.fast
    def test_search_and_download_handles_failed_downloads(self, tmp_path):
        """Test search_and_download handles failed downloads."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        mock_results = [
            create_mock_image_result(image_id="111"),
            create_mock_image_result(image_id="222"),
            create_mock_image_result(image_id="333"),
        ]

        with patch.object(client, 'search', return_value=mock_results):
            # First download fails, second succeeds
            download_count = [0]

            def mock_download(image, check_size=True, entity_name="", entity_type=""):
                download_count[0] += 1
                if download_count[0] == 1:
                    return None  # First fails
                return f"/path/to/image{download_count[0]}.jpg"

            with patch.object(client, 'download_image', side_effect=mock_download):
                result = client.search_and_download("test", max_images=2)

        # Should get 2 successful downloads (skipping first failure)
        assert len(result) == 2

    @pytest.mark.fast
    def test_search_and_download_empty_search(self, tmp_path):
        """Test search_and_download with no search results."""
        from src.media_sources.images.pixabay import PixabayImageClient

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(tmp_path),
            api_key="test_key"
        )

        with patch.object(client, 'search', return_value=[]):
            result = client.search_and_download("test", max_images=5)

        assert result == []


class TestPixabayClientInit:
    """Test client initialization."""

    @pytest.mark.requires_api
    def test_init_with_env_api_key(self, tmp_path):
        """Test initialization with env var API key."""
        from src.media_sources.images.pixabay import PixabayImageClient

        with patch.dict('os.environ', {'PIXABAY_API_KEY': 'env_key'}):
            client = PixabayImageClient(
                config=create_mock_config(),
                output_dir=str(tmp_path),
                api_key=None  # Use env var
            )

        assert client.api_key == 'env_key'

    @pytest.mark.requires_api
    def test_init_with_explicit_api_key(self, tmp_path):
        """Test initialization with explicit API key."""
        from src.media_sources.images.pixabay import PixabayImageClient

        with patch.dict('os.environ', {'PIXABAY_API_KEY': 'env_key'}):
            client = PixabayImageClient(
                config=create_mock_config(),
                output_dir=str(tmp_path),
                api_key="explicit_key"  # Override env var
            )

        assert client.api_key == "explicit_key"

    @pytest.mark.fast
    def test_init_creates_output_dir(self, tmp_path):
        """Test initialization creates output directory."""
        from src.media_sources.images.pixabay import PixabayImageClient

        output_dir = tmp_path / "new_output"

        client = PixabayImageClient(
            config=create_mock_config(),
            output_dir=str(output_dir),
            api_key="test_key"
        )

        assert output_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
