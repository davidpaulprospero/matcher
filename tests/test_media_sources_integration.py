import pytest
"""
Integration tests for media_sources package with mocked API responses.

Tests the full flow of searching and downloading from mocked Pexels, Pixabay, and Unsplash APIs.
Created Jan 7, 2026.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json

from src.media_sources.images import PexelsImageClient, PixabayImageClient, UnsplashImageClient
from src.media_sources.videos import PexelsVideoClient, PixabayVideoClient
from src.media_sources.images import download_entity_images
from src.media_sources.videos import download_entity_videos


class TestPexelsImageIntegration(unittest.TestCase):
    """Integration tests for Pexels image search and download"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        """Clean up test files"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.base.requests.Session')
    @pytest.mark.fast
    def test_pexels_search_and_download(self, mock_session_class):
        """Test Pexels search and download with mocked API"""
        # Mock API response
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Mock search response
        search_response = MagicMock()
        search_response.json.return_value = {
            'photos': [
                {
                    'id': 12345,
                    'url': 'https://pexels.com/photo/12345',
                    'photographer': 'Test User',
                    'alt': 'Beautiful landscape',
                    'width': 1920,
                    'height': 1080,
                    'src': {
                        'original': 'https://images.pexels.com/12345.jpg'
                    }
                }
            ]
        }
        search_response.raise_for_status = Mock()

        # Mock download response
        download_response = MagicMock()
        download_response.headers = {'content-length': '2000000'}  # 2MB
        download_response.iter_content.return_value = [b'x' * 2000000]
        download_response.raise_for_status = Mock()

        # Set up session mock to return different responses
        # Need 3 responses: search (test), search_and_download (search), download
        mock_session.get.side_effect = [search_response, search_response, download_response]
        mock_session.head.return_value.headers = {'content-length': '2000000'}

        # Create client and test
        client = PexelsImageClient(
            config=self.config,
            output_dir=self.temp_dir,
            api_key="test_key"
        )

        # Search
        results = client.search("test query", max_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "pexels")

        # Download (calls search again internally)
        downloaded = client.search_and_download("test", max_images=1)
        self.assertEqual(len(downloaded), 1)


class TestPixabayImageIntegration(unittest.TestCase):
    """Integration tests for Pixabay image search"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.base.requests.Session')
    @pytest.mark.fast
    def test_pixabay_search(self, mock_session_class):
        """Test Pixabay search with mocked API"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'hits': [
                {
                    'id': 67890,
                    'pageURL': 'https://pixabay.com/67890',
                    'user': 'Test Photographer',
                    'tags': 'nature, landscape',
                    'imageWidth': 1920,
                    'imageHeight': 1080,
                    'largeImageURL': 'https://pixabay.com/67890_large.jpg'
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        client = PixabayImageClient(
            config=self.config,
            output_dir=self.temp_dir,
            api_key="test_key"
        )

        results = client.search("landscape", max_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "pixabay")
        self.assertIn("nature", results[0].tags)


class TestUnsplashImageIntegration(unittest.TestCase):
    """Integration tests for Unsplash image search"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.base.requests.Session')
    @pytest.mark.fast
    def test_unsplash_search(self, mock_session_class):
        """Test Unsplash search with mocked API"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'results': [
                {
                    'id': 'abc123',
                    'description': 'Mountain view',
                    'width': 1920,
                    'height': 1080,
                    'urls': {
                        'full': 'https://unsplash.com/abc123/full.jpg',
                        'regular': 'https://unsplash.com/abc123/regular.jpg'
                    },
                    'user': {
                        'name': 'John Doe'
                    },
                    'links': {
                        'html': 'https://unsplash.com/photos/abc123'
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        client = UnsplashImageClient(
            config=self.config,
            output_dir=self.temp_dir,
            api_key="test_key"
        )

        results = client.search("mountains", max_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "unsplash")
        self.assertEqual(results[0].photographer, "John Doe")


class TestPexelsVideoIntegration(unittest.TestCase):
    """Integration tests for Pexels video search"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.base.requests.Session')
    @pytest.mark.fast
    def test_pexels_video_search(self, mock_session_class):
        """Test Pexels video search with mocked API"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'videos': [
                {
                    'id': 111222,
                    'url': 'https://pexels.com/video/111222',
                    'duration': 10.5,
                    'video_files': [
                        {
                            'link': 'https://videos.pexels.com/111222.mp4',
                            'quality': 'hd',
                            'width': 1920,
                            'height': 1080,
                            'file_type': 'video/mp4'
                        }
                    ]
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        client = PexelsVideoClient(
            config=self.config,
            output_dir=self.temp_dir,
            api_key="test_key",
            min_duration=5.0,
            max_duration=30.0
        )

        results = client.search("ocean", max_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "pexels")
        self.assertEqual(results[0].duration, 10.5)


class TestPixabayVideoIntegration(unittest.TestCase):
    """Integration tests for Pixabay video search"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.base.requests.Session')
    @pytest.mark.fast
    def test_pixabay_video_search(self, mock_session_class):
        """Test Pixabay video search with mocked API"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'hits': [
                {
                    'id': 333444,
                    'pageURL': 'https://pixabay.com/videos/333444',
                    'duration': 15,
                    'videos': {
                        'large': {
                            'url': 'https://pixabay.com/videos/333444_large.mp4',
                            'width': 1920,
                            'height': 1080,
                            'size': 'large'
                        }
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        client = PixabayVideoClient(
            config=self.config,
            output_dir=self.temp_dir,
            api_key="test_key"
        )

        results = client.search("forest", max_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "pixabay")


class TestDownloadEntityImagesIntegration(unittest.TestCase):
    """Integration tests for download_entity_images orchestration"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.images.orchestrator.PexelsImageClient')
    @pytest.mark.fast
    def test_download_with_stock_apis_only(self, mock_pexels_class):
        """Test download using only stock APIs (no Google/Bing)"""
        # Mock Pexels client
        mock_client = MagicMock()
        mock_client.api_key = "test_key"
        mock_client.search_and_download.return_value = [
            str(Path(self.temp_dir) / "img1.jpg"),
            str(Path(self.temp_dir) / "img2.jpg")
        ]
        mock_pexels_class.return_value = mock_client

        # Create test image files
        (Path(self.temp_dir) / "img1.jpg").write_bytes(b"test1")
        (Path(self.temp_dir) / "img2.jpg").write_bytes(b"test2")

        entities = [
            {'text': 'Paris', 'type': 'GPE', 'context': 'capital of France'}
        ]

        with patch('src.media_sources.images.orchestrator.PixabayImageClient'), \
             patch('src.media_sources.images.orchestrator.UnsplashImageClient'):

            results = download_entity_images(
                entities=entities,
                output_dir=self.temp_dir,
                topic="travel",
                images_per_entity=2,
                use_google=False,
                use_bing=False,
                use_stock_apis=True,
                pexels_key="test_key",
                config=self.config
            )

        self.assertIn('Paris', results)
        self.assertEqual(len(results['Paris'].images), 2)


class TestDownloadEntityVideosIntegration(unittest.TestCase):
    """Integration tests for download_entity_videos orchestration"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Mock()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @pytest.mark.fast
    def test_download_entity_videos(self, mock_pixabay_class, mock_pexels_class):
        """Test entity video download orchestration"""
        # Mock Pexels client
        mock_pexels = MagicMock()
        mock_pexels.api_key = "pexels_key"
        mock_pexels.search_and_download.return_value = [
            str(Path(self.temp_dir) / "sv" / "video1.mp4")
        ]
        mock_pexels_class.return_value = mock_pexels

        # Mock Pixabay client
        mock_pixabay = MagicMock()
        mock_pixabay.api_key = None
        mock_pixabay_class.return_value = mock_pixabay

        # Create test video file
        (Path(self.temp_dir) / "sv").mkdir(parents=True)
        (Path(self.temp_dir) / "sv" / "video1.mp4").write_bytes(b"test video")

        entities = [
            {'text': 'Ocean', 'type': 'MISC', 'context': 'marine environment'}
        ]

        results = download_entity_videos(
            entities=entities,
            output_dir=self.temp_dir,
            topic="nature",
            videos_per_entity=1,
            pexels_key="pexels_key",
            config=self.config
        )

        self.assertIn('Ocean', results)
        self.assertEqual(len(results['Ocean'].videos), 1)


if __name__ == '__main__':
    unittest.main()
