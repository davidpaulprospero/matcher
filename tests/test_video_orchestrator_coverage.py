"""
Test coverage for src/media_sources/videos/orchestrator.py

Target: Cover all 12 missed lines to achieve 100% coverage.

Covers:
- No API keys scenario (early return)
- Empty entity name (skip processing)
- Duplicate entity (skip already-processed)
- Empty query (skip entity)
- Pexels fallback to Pixabay
- No videos found logging
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock


class TestDownloadEntityVideosNoApiKeys:
    """Test behavior when no API keys are available."""

    def test_no_api_keys_returns_empty_dict(self, tmp_path):
        """Test that no API keys returns empty dict with warning."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        with patch.dict('os.environ', {}, clear=True):
            result = download_entity_videos(
                entities=[{'text': 'Test Entity', 'type': 'GPE', 'context': 'test'}],
                output_dir=str(tmp_path),
                pexels_key=None,
                pixabay_key=None
            )

        assert result == {}

    def test_no_api_keys_logs_warning(self, tmp_path, caplog):
        """Test that missing API keys logs a warning."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        with patch.dict('os.environ', {}, clear=True):
            download_entity_videos(
                entities=[{'text': 'Test', 'type': 'GPE'}],
                output_dir=str(tmp_path),
                pexels_key=None,
                pixabay_key=None
            )

        assert "No video API keys available" in caplog.text


class TestDownloadEntityVideosEntityFiltering:
    """Test entity filtering scenarios."""

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    def test_empty_entity_name_skipped(self, mock_pixabay, mock_pexels, tmp_path):
        """Test that entities with empty text are skipped."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "test_key"
        mock_pexels.return_value = mock_pexels_instance

        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = None
        mock_pixabay.return_value = mock_pixabay_instance

        result = download_entity_videos(
            entities=[
                {'text': '', 'type': 'GPE', 'context': 'empty'},  # Should skip
                {'text': None, 'type': 'PERSON'},  # Should skip (falsy)
            ],
            output_dir=str(tmp_path),
            pexels_key="test_key"
        )

        assert result == {}
        # No searches should have been made for empty entities
        mock_pexels_instance.search_and_download.assert_not_called()

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @patch('src.media_sources.videos.orchestrator.build_entity_query')
    def test_duplicate_entity_skipped(self, mock_build, mock_pixabay, mock_pexels, tmp_path):
        """Test that duplicate entity names are processed only once."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "test_key"
        mock_pexels_instance.search_and_download.return_value = ['/path/to/video.mp4']
        mock_pexels.return_value = mock_pexels_instance

        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = None
        mock_pixabay.return_value = mock_pixabay_instance

        mock_build.return_value = "test query"

        result = download_entity_videos(
            entities=[
                {'text': 'New York', 'type': 'GPE'},
                {'text': 'New York', 'type': 'GPE'},  # Duplicate - should skip
            ],
            output_dir=str(tmp_path),
            pexels_key="test_key"
        )

        # Should only have one result entry
        assert len(result) == 1
        assert 'New York' in result
        # build_entity_query should only be called once
        assert mock_build.call_count == 1

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @patch('src.media_sources.videos.orchestrator.build_entity_query')
    def test_empty_query_skipped(self, mock_build, mock_pixabay, mock_pexels, tmp_path):
        """Test that entities resulting in empty queries are skipped."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "test_key"
        mock_pexels.return_value = mock_pexels_instance

        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = None
        mock_pixabay.return_value = mock_pixabay_instance

        # Return empty query for entity
        mock_build.return_value = ""

        result = download_entity_videos(
            entities=[{'text': 'Test', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            pexels_key="test_key"
        )

        assert result == {}
        mock_pexels_instance.search_and_download.assert_not_called()


class TestDownloadEntityVideosApiFallback:
    """Test API fallback behavior."""

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @patch('src.media_sources.videos.orchestrator.build_entity_query')
    def test_pixabay_fills_remaining_slots(self, mock_build, mock_pixabay, mock_pexels, tmp_path):
        """Test that Pixabay fills remaining slots when Pexels returns partial."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        # Pexels returns 1 video
        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "pexels_key"
        mock_pexels_instance.search_and_download.return_value = ['/path/pexels1.mp4']
        mock_pexels.return_value = mock_pexels_instance

        # Pixabay returns 1 more video
        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = "pixabay_key"
        mock_pixabay_instance.search_and_download.return_value = ['/path/pixabay1.mp4']
        mock_pixabay.return_value = mock_pixabay_instance

        mock_build.return_value = "test query"

        result = download_entity_videos(
            entities=[{'text': 'Test Entity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            videos_per_entity=2,
            pexels_key="pexels_key",
            pixabay_key="pixabay_key"
        )

        assert 'Test Entity' in result
        assert len(result['Test Entity'].videos) == 2
        assert '/path/pexels1.mp4' in result['Test Entity'].videos
        assert '/path/pixabay1.mp4' in result['Test Entity'].videos

        # Pixabay should be called with remaining=1
        mock_pixabay_instance.search_and_download.assert_called_once()
        call_kwargs = mock_pixabay_instance.search_and_download.call_args
        assert call_kwargs.kwargs.get('max_videos') == 1 or call_kwargs[1].get('max_videos') == 1

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @patch('src.media_sources.videos.orchestrator.build_entity_query')
    def test_pexels_only_when_pixabay_not_needed(self, mock_build, mock_pixabay, mock_pexels, tmp_path):
        """Test that Pixabay is not called when Pexels provides enough videos."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        # Pexels returns 2 videos (enough for videos_per_entity=2)
        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "pexels_key"
        mock_pexels_instance.search_and_download.return_value = ['/p1.mp4', '/p2.mp4']
        mock_pexels.return_value = mock_pexels_instance

        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = "pixabay_key"
        mock_pixabay.return_value = mock_pixabay_instance

        mock_build.return_value = "test query"

        result = download_entity_videos(
            entities=[{'text': 'Test', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            videos_per_entity=2,
            pexels_key="pexels_key",
            pixabay_key="pixabay_key"
        )

        assert len(result['Test'].videos) == 2
        # Pixabay should NOT be called since Pexels returned enough
        mock_pixabay_instance.search_and_download.assert_not_called()


class TestDownloadEntityVideosNoVideosFound:
    """Test behavior when no videos are found."""

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @patch('src.media_sources.videos.orchestrator.build_entity_query')
    def test_no_videos_found_not_in_result(self, mock_build, mock_pixabay, mock_pexels, tmp_path):
        """Test that no videos found means entity not in result."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "pexels_key"
        mock_pexels_instance.search_and_download.return_value = []
        mock_pexels.return_value = mock_pexels_instance

        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = "pixabay_key"
        mock_pixabay_instance.search_and_download.return_value = []
        mock_pixabay.return_value = mock_pixabay_instance

        mock_build.return_value = "test query"

        result = download_entity_videos(
            entities=[{'text': 'Obscure Entity', 'type': 'GPE'}],
            output_dir=str(tmp_path),
            pexels_key="pexels_key",
            pixabay_key="pixabay_key"
        )

        # No result entry for this entity when no videos found
        assert 'Obscure Entity' not in result
        assert result == {}


class TestDownloadEntityVideosSuccessPath:
    """Test successful video download scenarios."""

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    @patch('src.media_sources.videos.orchestrator.build_entity_query')
    def test_successful_download_creates_result(self, mock_build, mock_pixabay, mock_pexels, tmp_path):
        """Test successful download creates proper EntityVideoResult."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        mock_pexels_instance = MagicMock()
        mock_pexels_instance.api_key = "pexels_key"
        mock_pexels_instance.search_and_download.return_value = ['/video1.mp4', '/video2.mp4']
        mock_pexels.return_value = mock_pexels_instance

        mock_pixabay_instance = MagicMock()
        mock_pixabay_instance.api_key = None
        mock_pixabay.return_value = mock_pixabay_instance

        mock_build.return_value = "paris travel"

        result = download_entity_videos(
            entities=[{
                'text': 'Paris',
                'type': 'GPE',
                'context': 'capital of France'
            }],
            output_dir=str(tmp_path),
            topic="European Travel",
            pexels_key="pexels_key"
        )

        assert 'Paris' in result
        entity_result = result['Paris']
        assert entity_result.entity_name == 'Paris'
        assert entity_result.entity_type == 'GPE'
        assert entity_result.context == 'capital of France'
        assert entity_result.query == "paris travel"
        assert len(entity_result.videos) == 2


class TestDownloadEntityVideosDummyConfig:
    """Test that DummyConfig is created when no config provided."""

    @patch('src.media_sources.videos.orchestrator.PexelsVideoClient')
    @patch('src.media_sources.videos.orchestrator.PixabayVideoClient')
    def test_dummy_config_created_when_none(self, mock_pixabay, mock_pexels, tmp_path):
        """Test DummyConfig is created when config=None."""
        from src.media_sources.videos.orchestrator import download_entity_videos

        # Both clients should receive a config object
        mock_pexels.return_value = MagicMock(api_key=None)
        mock_pixabay.return_value = MagicMock(api_key=None)

        download_entity_videos(
            entities=[],
            output_dir=str(tmp_path),
            config=None  # No config provided
        )

        # Check that clients were called with some config
        pexels_call_kwargs = mock_pexels.call_args.kwargs
        assert 'config' in pexels_call_kwargs
        assert pexels_call_kwargs['config'] is not None
