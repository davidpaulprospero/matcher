"""
Tests for parallel download worker count configuration.

Verifies that:
- parallel_workers field exists in DownloadConfig dataclass
- Config default value is 4 workers
- VideoDownloader.download_all() uses config value
- Setting parallel_workers=8 results in 8 workers being used

US-002: Add parallel download worker count configuration
Created: 2026-01-25 (Sprint 3)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.download import DownloadConfig
from src.downloader.core import VideoDownloader


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================================
# Test DownloadConfig parallel_workers field
# ============================================================================

class TestDownloadConfigParallelWorkers:
    """Test parallel_workers field in DownloadConfig dataclass"""

    @pytest.mark.fast
    def test_parallel_workers_field_exists(self):
        """parallel_workers field should exist on DownloadConfig"""
        config = DownloadConfig()
        assert hasattr(config, 'parallel_workers')

    @pytest.mark.fast
    def test_parallel_workers_default_value(self):
        """Default parallel_workers should be 4"""
        config = DownloadConfig()
        assert config.parallel_workers == 4

    @pytest.mark.fast
    def test_parallel_workers_can_be_set_to_8(self):
        """parallel_workers should accept value of 8"""
        config = DownloadConfig(parallel_workers=8)
        assert config.parallel_workers == 8

    @pytest.mark.fast
    def test_parallel_workers_can_be_set_to_1(self):
        """parallel_workers should accept value of 1 (sequential)"""
        config = DownloadConfig(parallel_workers=1)
        assert config.parallel_workers == 1

    @pytest.mark.fast
    def test_parallel_workers_type_is_int(self):
        """parallel_workers should be an integer"""
        config = DownloadConfig()
        assert isinstance(config.parallel_workers, int)


# ============================================================================
# Test VideoDownloader uses config value
# ============================================================================

class TestVideoDownloaderParallelWorkers:
    """Test VideoDownloader.download_all() uses parallel_workers from config"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with download settings"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_download_all_uses_config_parallel_workers(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path
    ):
        """download_all should use config.download.parallel_workers when max_concurrent is None"""
        # Setup
        mock_config.download.parallel_workers = 6
        downloader = VideoDownloader(config=mock_config)

        # Mock checkpoint to avoid file operations
        downloader.checkpoint = MagicMock()
        downloader.checkpoint.completed_keywords = []
        downloader.checkpoint.failed_keywords = []

        # Mock download_for_keyword to avoid actual downloads
        downloader.download_for_keyword = MagicMock(return_value=[])

        # Create output dir
        output_dir = tmp_path / "videos"
        output_dir.mkdir()

        # Call download_all with no explicit max_concurrent
        with patch.object(downloader, '_load_checkpoint', return_value=None):
            with patch.object(downloader, '_save_checkpoint'):
                with patch('builtins.print'):  # Suppress progress output
                    with patch('src.downloader.orchestrator.logger') as mock_logger:
                        downloader.download_all(
                            keywords=["test"],
                            output_dir=output_dir,
                            resume=False
                        )

                        # Verify parallel workers was logged with config value (6)
                        log_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Parallel workers: 6" in c for c in log_calls), \
                            f"Expected 'Parallel workers: 6' in logs but got: {log_calls}"

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_download_all_explicit_max_concurrent_overrides_config(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path
    ):
        """download_all should use explicit max_concurrent when provided"""
        # Setup
        mock_config.download.parallel_workers = 4  # Config says 4
        downloader = VideoDownloader(config=mock_config)

        # Mock checkpoint
        downloader.checkpoint = MagicMock()
        downloader.checkpoint.completed_keywords = []
        downloader.checkpoint.failed_keywords = []

        # Mock download_for_keyword
        downloader.download_for_keyword = MagicMock(return_value=[])

        # Create output dir
        output_dir = tmp_path / "videos"
        output_dir.mkdir()

        # Call with explicit max_concurrent=8
        with patch.object(downloader, '_load_checkpoint', return_value=None):
            with patch.object(downloader, '_save_checkpoint'):
                with patch('builtins.print'):
                    with patch('src.downloader.orchestrator.logger') as mock_logger:
                        downloader.download_all(
                            keywords=["test"],
                            output_dir=output_dir,
                            max_concurrent=8,  # Explicit override
                            resume=False
                        )

                        # Verify parallel workers was logged with explicit value (8)
                        log_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Parallel workers: 8" in c for c in log_calls), \
                            f"Expected 'Parallel workers: 8' in logs but got: {log_calls}"

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_download_all_falls_back_to_default_if_no_config(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path
    ):
        """download_all should fall back to 4 workers if parallel_workers missing from config"""
        # Setup config WITHOUT parallel_workers attribute
        del mock_config.download.parallel_workers
        downloader = VideoDownloader(config=mock_config)

        # Mock checkpoint
        downloader.checkpoint = MagicMock()
        downloader.checkpoint.completed_keywords = []
        downloader.checkpoint.failed_keywords = []

        # Mock download_for_keyword
        downloader.download_for_keyword = MagicMock(return_value=[])

        # Create output dir
        output_dir = tmp_path / "videos"
        output_dir.mkdir()

        # Call without explicit max_concurrent
        with patch.object(downloader, '_load_checkpoint', return_value=None):
            with patch.object(downloader, '_save_checkpoint'):
                with patch('builtins.print'):
                    with patch('src.downloader.orchestrator.logger') as mock_logger:
                        downloader.download_all(
                            keywords=["test"],
                            output_dir=output_dir,
                            resume=False
                        )

                        # Verify fallback to default (4)
                        log_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Parallel workers: 4" in c for c in log_calls), \
                            f"Expected 'Parallel workers: 4' in logs but got: {log_calls}"


# ============================================================================
# Test parallel_workers=8 configuration scenario
# ============================================================================

class TestParallelWorkers8Configuration:
    """Test that setting parallel_workers=8 in config results in 8 workers"""

    @pytest.mark.fast
    def test_config_with_parallel_workers_8(self):
        """DownloadConfig with parallel_workers=8 should use 8 workers"""
        config = DownloadConfig(parallel_workers=8)
        assert config.parallel_workers == 8

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_downloader_with_8_workers_in_config(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, tmp_path
    ):
        """VideoDownloader should use 8 workers when config.download.parallel_workers=8"""
        # Create mock config with parallel_workers=8
        mock_config = MagicMock()
        mock_config.download = MagicMock()
        mock_config.download.parallel_workers = 8
        mock_config.download.cookies_path = ""
        mock_config.download.cookies_from_browser = ""
        mock_config.download.title_blacklist = []
        mock_config.download.llm_title_filter = MagicMock(enabled=False)
        mock_config.download.audio_first = MagicMock(enabled=False)
        mock_config.download.speech_screening = MagicMock(enabled=False)
        mock_config.download.zero_download_remix = MagicMock(enabled=False)
        mock_config.cache_dir = "/tmp/cache"
        mock_config.downloaded_videos_dir = "/tmp/videos"
        mock_config.duration_tiers = {}

        downloader = VideoDownloader(config=mock_config)

        # Mock checkpoint
        downloader.checkpoint = MagicMock()
        downloader.checkpoint.completed_keywords = []
        downloader.checkpoint.failed_keywords = []

        # Mock download_for_keyword
        downloader.download_for_keyword = MagicMock(return_value=[])

        # Create output dir
        output_dir = tmp_path / "videos"
        output_dir.mkdir()

        # Call download_all
        with patch.object(downloader, '_load_checkpoint', return_value=None):
            with patch.object(downloader, '_save_checkpoint'):
                with patch('builtins.print'):
                    with patch('src.downloader.orchestrator.logger') as mock_logger:
                        downloader.download_all(
                            keywords=["test"],
                            output_dir=output_dir,
                            resume=False
                        )

                        # Verify 8 workers used
                        log_calls = [str(c) for c in mock_logger.info.call_args_list]
                        assert any("Parallel workers: 8" in c for c in log_calls), \
                            f"Expected 'Parallel workers: 8' in logs but got: {log_calls}"
