"""
Unit tests for media_sources package.

Tests the refactored entity image/video download clients extracted from entity_images.py.
Created Jan 7, 2026.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json

# Import models
from src.media_sources.models import (
    ImageResult,
    EntityImageResult,
    VideoResult,
    EntityVideoResult
)

# Import base class
from src.media_sources.base import BaseMediaClient

# Import utils
from src.media_sources.utils import (
    build_entity_query,
    check_local_entity_images,
    map_entities_to_segments,
    restore_entity_images_from_disk
)


class TestModels(unittest.TestCase):
    """Test data models"""

    def test_image_result_creation(self):
        """Test ImageResult dataclass creation"""
        img = ImageResult(
            id="test123",
            source="pexels",
            url="https://example.com",
            download_url="https://example.com/image.jpg",
            width=1920,
            height=1080,
            photographer="Test User",
            description="Test image",
            tags=["test", "image"]
        )
        self.assertEqual(img.id, "test123")
        self.assertEqual(img.source, "pexels")
        self.assertEqual(img.width, 1920)

    def test_entity_image_result_creation(self):
        """Test EntityImageResult with default fields"""
        result = EntityImageResult(
            entity_name="Paris",
            entity_type="GPE",
            context="capital of France",
            query="Paris capital France"
        )
        self.assertEqual(result.entity_name, "Paris")
        self.assertEqual(result.images, [])
        self.assertEqual(result.segment_indices, [])

    def test_video_result_creation(self):
        """Test VideoResult dataclass"""
        video = VideoResult(
            id="vid123",
            source="pexels",
            url="https://example.com",
            download_url="https://example.com/video.mp4",
            width=1920,
            height=1080,
            duration=15.5,
            quality="hd",
            file_type="mp4"
        )
        self.assertEqual(video.duration, 15.5)
        self.assertEqual(video.quality, "hd")

    def test_entity_video_result_creation(self):
        """Test EntityVideoResult with default fields"""
        result = EntityVideoResult(
            entity_name="Ocean",
            entity_type="MISC",
            context="marine environment",
            query="ocean waves"
        )
        self.assertEqual(result.entity_name, "Ocean")
        self.assertEqual(result.videos, [])


class TestBaseMediaClient(unittest.TestCase):
    """Test BaseMediaClient abstract class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        """Clean up test files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test BaseMediaClient initialization"""
        # Create concrete subclass for testing
        class TestClient(BaseMediaClient):
            def search(self, query, max_results):
                return []

        client = TestClient(
            config=self.config,
            output_dir=self.temp_dir,
            min_size_mb=1.0,
            download_timeout=30
        )

        self.assertEqual(client.min_size, 1024 * 1024)  # 1MB in bytes
        self.assertEqual(client.download_timeout, 30)
        self.assertTrue(Path(self.temp_dir).exists())

    def test_rate_limiting(self):
        """Test rate limiting enforcement"""
        class TestClient(BaseMediaClient):
            def search(self, query, max_results):
                return []

        client = TestClient(
            config=self.config,
            output_dir=self.temp_dir,
            rate_limit_delay=0.1
        )

        import time
        start = time.time()
        client._rate_limit()
        client._rate_limit()
        elapsed = time.time() - start

        # Should have delayed at least 0.1 seconds
        self.assertGreaterEqual(elapsed, 0.1)


class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def test_build_entity_query_basic(self):
        """Test basic query building"""
        entity = {
            'text': 'Paris',
            'type': 'GPE',
            'context': 'capital of France'
        }
        query = build_entity_query(entity, topic="travel documentary")

        self.assertIn('Paris', query)
        self.assertTrue(len(query) <= 60)

    def test_build_entity_query_with_context(self):
        """Test query building with context"""
        entity = {
            'text': 'Einstein',
            'type': 'PERSON',
            'context': 'physicist who developed relativity'
        }
        query = build_entity_query(entity)

        self.assertIn('Einstein', query)

    def test_build_entity_query_empty(self):
        """Test query building with empty entity"""
        entity = {'text': '', 'type': 'PERSON'}
        query = build_entity_query(entity)

        self.assertEqual(query, "")

    def test_build_entity_query_length_limit(self):
        """Test query length limiting"""
        entity = {
            'text': 'Test',
            'type': 'MISC',
            'context': 'a very long context with many words that should be trimmed'
        }
        topic = "documentary about various topics and subjects"
        query = build_entity_query(entity, topic)

        # Should be trimmed to reasonable length
        self.assertLessEqual(len(query), 60)

    def test_check_local_entity_images_no_dir(self):
        """Test checking local images when directory doesn't exist"""
        images = check_local_entity_images("/nonexistent/path", "Paris", "GPE")
        self.assertEqual(images, [])

    def test_check_local_entity_images_with_metadata(self):
        """Test finding local images with metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image and metadata
            img_path = Path(temp_dir) / "12345.jpg"
            meta_path = Path(temp_dir) / "12345.entity.json"

            img_path.write_bytes(b"fake image data")
            meta_path.write_text(json.dumps({
                'entity_name': 'Paris',
                'entity_type': 'GPE',
                'query': 'Paris France'
            }))

            # Find images
            images = check_local_entity_images(temp_dir, "Paris", "GPE")

            self.assertEqual(len(images), 1)
            self.assertTrue(images[0].endswith("12345.jpg"))

    def test_check_local_entity_images_case_insensitive(self):
        """Test case-insensitive entity name matching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            meta_path = Path(temp_dir) / "test.entity.json"
            img_path = Path(temp_dir) / "test.jpg"

            meta_path.write_text(json.dumps({'entity_name': 'PARIS'}))
            img_path.write_bytes(b"test")

            # Should find with lowercase query
            images = check_local_entity_images(temp_dir, "paris")
            self.assertEqual(len(images), 1)

    def test_map_entities_to_segments(self):
        """Test entity-to-segment mapping"""
        entities = [
            {'text': 'Paris', 'type': 'GPE'},
            {'text': 'Einstein', 'type': 'PERSON'}
        ]

        segments = [
            {'text': 'Paris is the capital of France'},
            {'text': 'Einstein developed relativity'},
            {'text': 'Paris has the Eiffel Tower'}
        ]

        mapping = map_entities_to_segments(entities, segments)

        self.assertIn('Paris', mapping)
        self.assertIn('Einstein', mapping)
        self.assertEqual(mapping['Paris'], [0, 2])  # Appears in segments 0 and 2
        self.assertEqual(mapping['Einstein'], [1])  # Appears in segment 1

    def test_map_entities_to_segments_no_matches(self):
        """Test mapping when entity doesn't appear"""
        entities = [{'text': 'Tokyo', 'type': 'GPE'}]
        segments = [{'text': 'This is about Paris'}]

        mapping = map_entities_to_segments(entities, segments)

        self.assertEqual(mapping['Tokyo'], [])

    def test_restore_entity_images_from_disk_no_dir(self):
        """Test restore when directory doesn't exist"""
        results = restore_entity_images_from_disk("/nonexistent/path")
        self.assertEqual(results, {})

    def test_restore_entity_images_from_disk_with_files(self):
        """Test restoring entity images from metadata files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            img1 = Path(temp_dir) / "img1.jpg"
            meta1 = Path(temp_dir) / "img1.entity.json"
            img2 = Path(temp_dir) / "img2.jpg"
            meta2 = Path(temp_dir) / "img2.entity.json"

            img1.write_bytes(b"image1")
            img2.write_bytes(b"image2")

            meta1.write_text(json.dumps({
                'entity_name': 'Paris',
                'entity_type': 'GPE',
                'context': 'capital',
                'query': 'Paris France'
            }))

            meta2.write_text(json.dumps({
                'entity_name': 'Paris',
                'entity_type': 'GPE',
                'context': 'capital',
                'query': 'Paris France'
            }))

            # Restore
            results = restore_entity_images_from_disk(temp_dir)

            self.assertIn('Paris', results)
            self.assertEqual(len(results['Paris'].images), 2)
            self.assertEqual(results['Paris'].entity_type, 'GPE')

    def test_restore_entity_images_with_segment_mapping(self):
        """Test restore with segment mapping"""
        with tempfile.TemporaryDirectory() as temp_dir:
            img = Path(temp_dir) / "test.jpg"
            meta = Path(temp_dir) / "test.entity.json"

            img.write_bytes(b"test")
            meta.write_text(json.dumps({
                'entity_name': 'Paris',
                'entity_type': 'GPE',
                'query': 'Paris'
            }))

            segments = [{'text': 'Paris is beautiful'}]

            results = restore_entity_images_from_disk(temp_dir, segments)

            self.assertEqual(results['Paris'].segment_indices, [0])


class TestImageClients(unittest.TestCase):
    """Test image client modules (import checks)"""

    def test_import_pexels_image_client(self):
        """Test importing PexelsImageClient"""
        from src.media_sources.images import PexelsImageClient
        self.assertIsNotNone(PexelsImageClient)

    def test_import_pixabay_image_client(self):
        """Test importing PixabayImageClient"""
        from src.media_sources.images import PixabayImageClient
        self.assertIsNotNone(PixabayImageClient)

    def test_import_unsplash_image_client(self):
        """Test importing UnsplashImageClient"""
        from src.media_sources.images import UnsplashImageClient
        self.assertIsNotNone(UnsplashImageClient)

    def test_import_google_bing_client(self):
        """Test importing GoogleBingImageClient"""
        from src.media_sources.images import GoogleBingImageClient
        self.assertIsNotNone(GoogleBingImageClient)


class TestVideoClients(unittest.TestCase):
    """Test video client modules (import checks)"""

    def test_import_pexels_video_client(self):
        """Test importing PexelsVideoClient"""
        from src.media_sources.videos import PexelsVideoClient
        self.assertIsNotNone(PexelsVideoClient)

    def test_import_pixabay_video_client(self):
        """Test importing PixabayVideoClient"""
        from src.media_sources.videos import PixabayVideoClient
        self.assertIsNotNone(PixabayVideoClient)


class TestOrchestrators(unittest.TestCase):
    """Test orchestration functions (import checks)"""

    def test_import_download_entity_images(self):
        """Test importing download_entity_images"""
        from src.media_sources.images import download_entity_images
        self.assertIsNotNone(download_entity_images)

    def test_import_download_entity_videos(self):
        """Test importing download_entity_videos"""
        from src.media_sources.videos import download_entity_videos
        self.assertIsNotNone(download_entity_videos)


class TestPackageAPI(unittest.TestCase):
    """Test public package API"""

    def test_import_from_main_package(self):
        """Test importing from main media_sources package"""
        from src.media_sources import (
            ImageResult,
            EntityImageResult,
            VideoResult,
            EntityVideoResult,
            BaseMediaClient,
            build_entity_query,
            check_local_entity_images,
            download_entity_images,
            download_entity_videos
        )

        # Verify all imports successful
        self.assertIsNotNone(ImageResult)
        self.assertIsNotNone(EntityImageResult)
        self.assertIsNotNone(VideoResult)
        self.assertIsNotNone(EntityVideoResult)
        self.assertIsNotNone(BaseMediaClient)
        self.assertIsNotNone(build_entity_query)
        self.assertIsNotNone(check_local_entity_images)
        self.assertIsNotNone(download_entity_images)
        self.assertIsNotNone(download_entity_videos)


if __name__ == '__main__':
    unittest.main()
