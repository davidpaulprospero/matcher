"""
Test coverage for src/media_sources/images/orchestrator.py

Target: Cover all 54 missed lines to achieve 100% coverage.

Covers:
- No image sources available (early return)
- Local cache hit (skip search)
- Global entity cache hit (copy to project)
- Empty/duplicate entity name handling
- Google search error with alive_progress
- Bing fallback logic
- Stock API fallback chain (Pexels → Pixabay → Unsplash)
- Path validation (brackets, non-existent paths)
- Global cache registration
"""

import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock


class TestDownloadEntityImagesNoSources:
    """Test behavior when no image sources are available."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    def test_no_sources_available_returns_empty(self, mock_google, tmp_path, caplog):
        """Test that no available sources returns empty dict with warning."""
        from src.media_sources.images.orchestrator import download_entity_images

        # Mock Google client with no client (unavailable)
        mock_google_instance = MagicMock()
        mock_google_instance.client = None
        mock_google.return_value = mock_google_instance

        result = download_entity_images(
            entities=[{'text': 'Test', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=False,
            use_stock_apis=False  # No stock APIs
        )

        assert result == {}
        assert "No image sources available" in caplog.text


class TestDownloadEntityImagesEntityFiltering:
    """Test entity filtering scenarios."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    def test_empty_entity_name_skipped(self, mock_google, tmp_path):
        """Test that entities with empty text are skipped."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_google_instance = MagicMock()
        mock_google_instance.client = Mock()
        mock_google.return_value = mock_google_instance

        result = download_entity_images(
            entities=[
                {'text': '', 'type': 'GPE'},  # Empty - skip
                {'text': None, 'type': 'PERSON'},  # None - skip
            ],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=False,
            use_stock_apis=False
        )

        assert result == {}
        mock_google_instance.search_and_download.assert_not_called()

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_duplicate_entity_processed_once(self, mock_time, mock_build, mock_google, tmp_path):
        """Test that duplicate entity names are processed only once."""
        from src.media_sources.images.orchestrator import download_entity_images

        # Create actual image file so path validation passes
        img_path = tmp_path / "img.jpg"
        img_path.write_bytes(b"fake image")

        mock_google_instance = MagicMock()
        mock_google_instance.client = Mock()
        mock_google_instance.search_and_download.return_value = [str(img_path)]
        mock_google.return_value = mock_google_instance

        mock_build.return_value = "paris query"

        result = download_entity_images(
            entities=[
                {'text': 'Paris', 'type': 'GPE'},
                {'text': 'Paris', 'type': 'GPE'},  # Duplicate - skip
            ],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=False,
            use_stock_apis=False
        )

        # build_entity_query should only be called once
        assert mock_build.call_count == 1
        assert 'Paris' in result


class TestDownloadEntityImagesLocalCache:
    """Test local cache hit scenarios."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    def test_local_cache_hit_skips_search(self, mock_check, mock_google, tmp_path):
        """Test that local cache hit skips image search."""
        from src.media_sources.images.orchestrator import download_entity_images

        # Return enough images from local cache
        mock_check.return_value = ['/local/img1.jpg', '/local/img2.jpg', '/local/img3.jpg']

        mock_google_instance = MagicMock()
        mock_google_instance.client = Mock()
        mock_google.return_value = mock_google_instance

        result = download_entity_images(
            entities=[{'text': 'CachedEntity', 'type': 'GPE', 'context': 'test'}],
            output_dir=str(tmp_path),
            images_per_entity=3,
            use_google=True,
            use_bing=False,
            use_stock_apis=False,
            skip_local_cache=False  # Use local cache
        )

        assert 'CachedEntity' in result
        assert result['CachedEntity'].query == "(local cache)"
        # Google should NOT be called
        mock_google_instance.search_and_download.assert_not_called()


class TestDownloadEntityImagesGlobalCache:
    """Test global entity cache hit scenarios."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    def test_global_cache_hit_uses_cached_images(self, mock_check, mock_google, tmp_path):
        """Test that global cache hit copies images to project."""
        from src.media_sources.images.orchestrator import download_entity_images

        # No local cache
        mock_check.return_value = []

        mock_google_instance = MagicMock()
        mock_google_instance.client = Mock()
        mock_google.return_value = mock_google_instance

        # Mock global entity cache
        mock_cached_entity = Mock()
        mock_cached_entity.query = "cached query"

        mock_entity_cache = Mock()
        mock_entity_cache.find_entity.return_value = mock_cached_entity
        mock_entity_cache.get_images_for_project.return_value = ['/cached/img1.jpg', '/cached/img2.jpg']

        result = download_entity_images(
            entities=[{'text': 'GlobalCachedEntity', 'type': 'GPE', 'context': 'test'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=False,
            use_stock_apis=False,
            entity_cache=mock_entity_cache
        )

        assert 'GlobalCachedEntity' in result
        assert result['GlobalCachedEntity'].query == "cached query"
        # Google should NOT be called
        mock_google_instance.search_and_download.assert_not_called()


class TestDownloadEntityImagesGoogleErrors:
    """Test Google search error handling."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_alive_progress_error_reinitializes(self, mock_time, mock_build, mock_check, mock_google, tmp_path):
        """Test that alive_progress error triggers client reinitialization."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        mock_google_instance = MagicMock()
        mock_google_instance.client = Mock()
        # First call raises alive_progress error
        mock_google_instance.search_and_download.side_effect = Exception("alive_progress nested error")
        mock_google.return_value = mock_google_instance

        result = download_entity_images(
            entities=[{'text': 'TestEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=False,
            use_stock_apis=False
        )

        # Should have called _reinitialize_client
        mock_google_instance._reinitialize_client.assert_called()


class TestDownloadEntityImagesBingFallback:
    """Test Bing fallback behavior."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_bing_used_when_google_finds_nothing(self, mock_time, mock_build, mock_check, mock_google, tmp_path):
        """Test Bing is used when Google finds 0 images."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Create two different mock instances for Google and Bing
        google_instance = MagicMock()
        google_instance.client = Mock()
        google_instance.search_and_download.return_value = []  # Google finds nothing

        bing_instance = MagicMock()
        bing_instance.client = Mock()
        bing_instance.search_and_download.return_value = []

        # Return different instances for Google and Bing
        mock_google.side_effect = [google_instance, bing_instance]

        result = download_entity_images(
            entities=[{'text': 'TestEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=True,  # Enable Bing
            use_stock_apis=False
        )

        # Bing client should be created and reinitialized
        assert mock_google.call_count == 2  # Google + Bing

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_bing_skipped_when_google_finds_partial(self, mock_time, mock_build, mock_check, mock_google, tmp_path):
        """Test Bing is skipped when Google finds some images (partial)."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Create mock for existing image file
        img_path = tmp_path / "img1.jpg"
        img_path.write_bytes(b"fake image")

        google_instance = MagicMock()
        google_instance.client = Mock()
        google_instance.search_and_download.return_value = [str(img_path)]  # 1 image (partial)

        bing_instance = MagicMock()
        bing_instance.client = Mock()

        mock_google.side_effect = [google_instance, bing_instance]

        result = download_entity_images(
            entities=[{'text': 'TestEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            images_per_entity=3,
            use_google=True,
            use_bing=True,
            use_stock_apis=False
        )

        # Key assertion: Bing's search_and_download should NOT be called
        # when Google finds partial results (>0 but <images_per_entity)
        bing_instance.search_and_download.assert_not_called()
        assert 'TestEntity' in result


class TestDownloadEntityImagesStockFallback:
    """Test stock API fallback chain."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.PexelsImageClient')
    @patch('src.media_sources.images.orchestrator.PixabayImageClient')
    @patch('src.media_sources.images.orchestrator.UnsplashImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_stock_api_fallback_chain(self, mock_time, mock_build, mock_check,
                                       mock_unsplash, mock_pixabay, mock_pexels, mock_google, tmp_path):
        """Test Pexels → Pixabay → Unsplash fallback chain."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Create test image files
        img1 = tmp_path / "pexels_img.jpg"
        img1.write_bytes(b"fake")
        img2 = tmp_path / "pixabay_img.jpg"
        img2.write_bytes(b"fake")

        # Google finds nothing
        google_instance = MagicMock()
        google_instance.client = None  # Not available
        mock_google.return_value = google_instance

        # Pexels returns 1 image
        pexels_instance = MagicMock()
        pexels_instance.api_key = "pexels_key"
        pexels_instance.search_and_download.return_value = [str(img1)]
        mock_pexels.return_value = pexels_instance

        # Pixabay returns 1 more
        pixabay_instance = MagicMock()
        pixabay_instance.api_key = "pixabay_key"
        pixabay_instance.search_and_download.return_value = [str(img2)]
        mock_pixabay.return_value = pixabay_instance

        # Unsplash not needed (already have 2)
        unsplash_instance = MagicMock()
        unsplash_instance.api_key = None  # Not available
        mock_unsplash.return_value = unsplash_instance

        mock_config = Mock()

        result = download_entity_images(
            entities=[{'text': 'StockTest', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            images_per_entity=2,
            use_google=False,
            use_stock_apis=True,
            config=mock_config
        )

        assert 'StockTest' in result
        assert len(result['StockTest'].images) == 2


class TestDownloadEntityImagesPathValidation:
    """Test path validation logic."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_bracket_paths_skipped(self, mock_time, mock_build, mock_check, mock_google, tmp_path, caplog):
        """Test that paths with brackets are skipped."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        google_instance = MagicMock()
        google_instance.client = Mock()
        # Return path with brackets (image sequence notation)
        google_instance.search_and_download.return_value = ['/path/img[001-100].jpg']
        mock_google.return_value = google_instance

        result = download_entity_images(
            entities=[{'text': 'BracketTest', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_stock_apis=False
        )

        # Entity should not be in results (all paths invalid)
        assert 'BracketTest' not in result
        assert "Skipping invalid path with brackets" in caplog.text

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_nonexistent_paths_skipped(self, mock_time, mock_build, mock_check, mock_google, tmp_path, caplog):
        """Test that non-existent paths are skipped."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        google_instance = MagicMock()
        google_instance.client = Mock()
        # Return path that doesn't exist
        google_instance.search_and_download.return_value = ['/nonexistent/path/img.jpg']
        mock_google.return_value = google_instance

        result = download_entity_images(
            entities=[{'text': 'NonexistentTest', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_stock_apis=False
        )

        # Entity should not be in results
        assert 'NonexistentTest' not in result
        assert "Skipping non-existent path" in caplog.text


class TestDownloadEntityImagesGlobalCacheRegistration:
    """Test global cache registration."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_downloaded_images_registered_in_global_cache(self, mock_time, mock_build, mock_check, mock_google, tmp_path):
        """Test that downloaded images are added to global entity cache."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Create actual image file
        img_path = tmp_path / "downloaded.jpg"
        img_path.write_bytes(b"fake image data")

        google_instance = MagicMock()
        google_instance.client = Mock()
        google_instance.search_and_download.return_value = [str(img_path)]
        mock_google.return_value = google_instance

        mock_entity_cache = Mock()
        mock_entity_cache.find_entity.return_value = None  # Not in cache

        result = download_entity_images(
            entities=[{'text': 'CacheRegTest', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_stock_apis=False,
            entity_cache=mock_entity_cache,
            source_project="TestProject"
        )

        # Should have registered with entity cache
        mock_entity_cache.add_entity.assert_called_once()
        call_kwargs = mock_entity_cache.add_entity.call_args.kwargs
        assert call_kwargs['entity_name'] == 'CacheRegTest'
        assert call_kwargs['entity_type'] == 'GPE'
        assert call_kwargs['source_project'] == 'TestProject'


class TestDownloadEntityImagesEmptyQuery:
    """Test empty query handling (line 207)."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    def test_empty_query_skips_entity(self, mock_build, mock_check, mock_google, tmp_path):
        """Test line 207: Entity is skipped when query is empty."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = ""  # Empty query

        google_instance = MagicMock()
        google_instance.client = Mock()
        mock_google.return_value = google_instance

        result = download_entity_images(
            entities=[{'text': 'EmptyQueryEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_stock_apis=False
        )

        # Should be skipped - no search attempted
        assert 'EmptyQueryEntity' not in result
        google_instance.search_and_download.assert_not_called()


class TestDownloadEntityImagesGoogleGenericError:
    """Test Google search generic error handling (line 231)."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_generic_google_error_logged(self, mock_time, mock_build, mock_check, mock_google, tmp_path, caplog):
        """Test line 231: Generic Google search error is logged."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        google_instance = MagicMock()
        google_instance.client = Mock()
        # Raise a non-alive_progress error
        google_instance.search_and_download.side_effect = Exception("Network timeout")
        mock_google.return_value = google_instance

        result = download_entity_images(
            entities=[{'text': 'ErrorEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_stock_apis=False
        )

        # Should log the generic error (not reinitialize)
        assert "Google search error" in caplog.text or "Network timeout" in caplog.text
        # _reinitialize_client should NOT be called for generic errors
        google_instance._reinitialize_client.assert_not_called()


class TestDownloadEntityImagesBingException:
    """Test Bing search exception handling (lines 257-260)."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_bing_exception_logged(self, mock_time, mock_build, mock_check, mock_google, tmp_path, caplog):
        """Test lines 257-258: Bing search exception is logged."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Google finds nothing
        google_instance = MagicMock()
        google_instance.client = Mock()
        google_instance.search_and_download.return_value = []

        # Bing raises exception
        bing_instance = MagicMock()
        bing_instance.client = Mock()
        bing_instance.search_and_download.side_effect = Exception("Bing API error")

        mock_google.side_effect = [google_instance, bing_instance]

        result = download_entity_images(
            entities=[{'text': 'BingErrorEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=True,
            use_stock_apis=False
        )

        assert "Bing search failed" in caplog.text

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_bing_client_unavailable(self, mock_time, mock_build, mock_check, mock_google, tmp_path, caplog):
        """Test lines 259-260: Bing client unavailable is logged."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Google finds nothing
        google_instance = MagicMock()
        google_instance.client = Mock()
        google_instance.search_and_download.return_value = []

        # Bing starts with client (for initial check line 144) but becomes None after reinit
        bing_instance = MagicMock()
        bing_instance.client = Mock()  # Initially available

        # After _reinitialize_client is called, client becomes None
        def clear_client():
            bing_instance.client = None

        bing_instance._reinitialize_client.side_effect = clear_client

        mock_google.side_effect = [google_instance, bing_instance]

        result = download_entity_images(
            entities=[{'text': 'BingUnavailEntity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            use_google=True,
            use_bing=True,
            use_stock_apis=False
        )

        # Line 260 should be executed
        assert "Bing client unavailable" in caplog.text


class TestDownloadEntityImagesUnsplashFallback:
    """Test Unsplash fallback (lines 291-298)."""

    @patch('src.media_sources.images.orchestrator.GoogleBingImageClient')
    @patch('src.media_sources.images.orchestrator.PexelsImageClient')
    @patch('src.media_sources.images.orchestrator.PixabayImageClient')
    @patch('src.media_sources.images.orchestrator.UnsplashImageClient')
    @patch('src.media_sources.images.orchestrator.check_local_entity_images')
    @patch('src.media_sources.images.orchestrator.build_entity_query')
    @patch('src.media_sources.images.orchestrator.time')
    def test_unsplash_used_when_others_insufficient(self, mock_time, mock_build, mock_check,
                                                     mock_unsplash, mock_pixabay, mock_pexels,
                                                     mock_google, tmp_path):
        """Test lines 291-298: Unsplash is used when Pexels+Pixabay insufficient."""
        from src.media_sources.images.orchestrator import download_entity_images

        mock_check.return_value = []
        mock_build.return_value = "test query"

        # Create test image files
        img1 = tmp_path / "pexels.jpg"
        img1.write_bytes(b"fake")
        img2 = tmp_path / "pixabay.jpg"
        img2.write_bytes(b"fake")
        img3 = tmp_path / "unsplash.jpg"
        img3.write_bytes(b"fake")

        # Google not available
        google_instance = MagicMock()
        google_instance.client = None
        mock_google.return_value = google_instance

        # Pexels returns 1 image
        pexels_instance = MagicMock()
        pexels_instance.api_key = "pexels_key"
        pexels_instance.search_and_download.return_value = [str(img1)]
        mock_pexels.return_value = pexels_instance

        # Pixabay returns 1 image
        pixabay_instance = MagicMock()
        pixabay_instance.api_key = "pixabay_key"
        pixabay_instance.search_and_download.return_value = [str(img2)]
        mock_pixabay.return_value = pixabay_instance

        # Unsplash returns 1 image (needed to reach 3)
        unsplash_instance = MagicMock()
        unsplash_instance.api_key = "unsplash_key"
        unsplash_instance.search_and_download.return_value = [str(img3)]
        mock_unsplash.return_value = unsplash_instance

        mock_config = Mock()

        result = download_entity_images(
            entities=[{'text': 'UnsplashTest', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            images_per_entity=3,  # Need 3 images
            use_google=False,
            use_stock_apis=True,
            config=mock_config
        )

        # Unsplash should be called
        unsplash_instance.search_and_download.assert_called_once()
        assert 'UnsplashTest' in result
        assert len(result['UnsplashTest'].images) == 3
