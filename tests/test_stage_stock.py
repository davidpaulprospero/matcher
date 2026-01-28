"""
Test Suite for StockVideoStage

Tests the StockVideoStage class which handles:
- Generic stock footage downloads from Pexels/Pixabay
- Keyword-based search
- Video deduplication
- Failed keyword tracking
- Checkpoint operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.stages.stock import StockVideoStage
from src.state import PipelineState, DownloadedVideo


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config(tmp_path):
    """Create mock config with stock download settings"""
    config = MagicMock()
    config.pipeline.skip_download = False

    # Enhanced config for stock footage
    enhanced = MagicMock()
    enhanced.enabled = True
    enhanced.enable_pexels = True
    enhanced.enable_pixabay = True
    enhanced.stock_per_keyword = 3
    config.enhanced = enhanced

    config.downloaded_videos_dir = str(tmp_path / "downloads")
    return config


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.project_dir = tmp_path
    return checkpoint


@pytest.fixture
def mock_keywords():
    """Create mock keywords"""
    return ['ocean', 'mountains', 'sunset', 'forest', 'cityscape']


# ============================================================================
# Test Stage Initialization
# ============================================================================

class TestStockVideoStageInit:
    """Test stage initialization"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name"""
        stage = StockVideoStage()
        assert stage.name == "STOCK"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = StockVideoStage()
        assert "stock footage" in stage.description.lower()

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered in stage registry"""
        from src.stages import get_stage
        stage_class = get_stage("STOCK")
        assert stage_class is StockVideoStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    @pytest.mark.fast
    def test_validate_no_hard_requirements(self, mock_config):
        """Test validation has no hard requirements (optional stage)"""
        stage = StockVideoStage()
        state = PipelineState()
        # Empty state should still pass validation
        result = stage.validate_inputs(state, mock_config)
        assert result is None


# ============================================================================
# Test Skip Conditions
# ============================================================================

class TestSkipConditions:
    """Test various skip conditions"""

    @pytest.mark.fast
    def test_skip_when_download_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when pipeline.skip_download is True"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_config.pipeline.skip_download = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'skip_download_enabled'

    @pytest.mark.fast
    def test_skip_when_enhanced_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when enhanced features disabled"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_config.enhanced.enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'enhanced_disabled'

    @pytest.mark.fast
    def test_skip_when_no_enhanced_config(self, mock_config, mock_checkpoint):
        """Test skipping when enhanced config not present"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_config.enhanced = None

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'enhanced_disabled'

    @pytest.mark.fast
    def test_skip_when_no_stock_sources_enabled(self, mock_config, mock_checkpoint):
        """Test skipping when neither Pexels nor Pixabay enabled"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_config.enhanced.enable_pexels = False
        mock_config.enhanced.enable_pixabay = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_stock_sources_enabled'

    @pytest.mark.fast
    def test_skip_when_no_keywords(self, mock_config, mock_checkpoint):
        """Test skipping when no keywords available"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_keywords'


# ============================================================================
# Test Keyword Limiting
# ============================================================================

class TestKeywordLimiting:
    """Test keyword limiting for API rate limits"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_keyword_limit_15(self, mock_pixabay, mock_pexels, mock_config,
                             mock_checkpoint):
        """Test keywords are limited to 15 for API rate limits"""
        stage = StockVideoStage()
        state = PipelineState()

        # Create 20 keywords
        state.keywords = [f'keyword{i}' for i in range(20)]
        state.downloaded_videos = []

        mock_pexels.return_value = ([], {})
        mock_pixabay.return_value = ([], {})

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should only use first 15 keywords
        assert mock_pexels.called
        pexels_keywords = mock_pexels.call_args[0][0]
        assert len(pexels_keywords) == 15


# ============================================================================
# Test Pexels Download
# ============================================================================

class TestPexelsDownload:
    """Test Pexels download functionality"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_pexels_only(self, mock_pixabay, mock_pexels, mock_config,
                        mock_checkpoint, mock_keywords):
        """Test download from Pexels only"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_config.enhanced.enable_pexels = True
        mock_config.enhanced.enable_pixabay = False

        mock_pexels.return_value = (['/path/video1.mp4', '/path/video2.mp4'], {'ocean': 2})

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert mock_pexels.called
        assert not mock_pixabay.called
        assert result.data['stock_count'] == 2
        assert result.data['sources']['pexels'] == 2

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_pexels_failed_keyword_tracking(self, mock_pixabay, mock_pexels,
                                           mock_config, mock_checkpoint, mock_keywords):
        """Test tracking of failed keywords"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []
        state.failed_keywords = []

        mock_config.enhanced.enable_pexels = True
        mock_config.enhanced.enable_pixabay = False

        # Return 0 videos for 'ocean' keyword
        mock_pexels.return_value = (['/path/video1.mp4'], {'ocean': 0, 'mountains': 1})

        result = stage.run(state, mock_config, mock_checkpoint)

        # 'ocean' should be added to failed_keywords
        assert 'ocean' in state.failed_keywords
        assert 'mountains' not in state.failed_keywords


# ============================================================================
# Test Pixabay Download
# ============================================================================

class TestPixabayDownload:
    """Test Pixabay download functionality"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_pixabay_only(self, mock_pixabay, mock_pexels, mock_config,
                         mock_checkpoint, mock_keywords):
        """Test download from Pixabay only"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_config.enhanced.enable_pexels = False
        mock_config.enhanced.enable_pixabay = True

        mock_pixabay.return_value = (['/path/video3.mp4'], {'sunset': 1})

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert not mock_pexels.called
        assert mock_pixabay.called
        assert result.data['stock_count'] == 1
        assert result.data['sources']['pixabay'] == 1


# ============================================================================
# Test Combined Download
# ============================================================================

class TestCombinedDownload:
    """Test downloading from both sources"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_both_sources(self, mock_pixabay, mock_pexels, mock_config,
                         mock_checkpoint, mock_keywords):
        """Test download from both Pexels and Pixabay"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_pexels.return_value = (['/path/pexels1.mp4', '/path/pexels2.mp4'], {'ocean': 2})
        mock_pixabay.return_value = (['/path/pixabay1.mp4'], {'sunset': 1})

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert mock_pexels.called
        assert mock_pixabay.called
        assert result.data['stock_count'] == 3
        assert result.data['sources']['pexels'] == 2
        assert result.data['sources']['pixabay'] == 1


# ============================================================================
# Test Video Deduplication
# ============================================================================

class TestVideoDeduplication:
    """Test deduplication of stock videos"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_no_duplicate_videos(self, mock_pixabay, mock_pexels, mock_config,
                                mock_checkpoint, mock_keywords):
        """Test that videos already in downloaded_videos are not added again"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords

        # Pre-existing video
        existing_path = '/path/video1.mp4'
        state.downloaded_videos = [
            DownloadedVideo(
                file=existing_path,
                source='stock',
                keyword='existing',
                url='',
                title='video1',
                duration=10.0
            )
        ]

        mock_config.enhanced.enable_pexels = True
        mock_config.enhanced.enable_pixabay = False

        # Pexels returns same video + new video
        mock_pexels.return_value = ([existing_path, '/path/video2.mp4'], {'ocean': 2})

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should only add video2, not video1 again
        assert len(state.downloaded_videos) == 2
        assert state.downloaded_videos[0].file == existing_path  # Original
        assert state.downloaded_videos[1].file == '/path/video2.mp4'  # New


# ============================================================================
# Test DownloadedVideo Creation
# ============================================================================

class TestDownloadedVideoCreation:
    """Test creation of DownloadedVideo objects"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_downloaded_video_attributes(self, mock_pixabay, mock_pexels,
                                        mock_config, mock_checkpoint, mock_keywords):
        """Test DownloadedVideo objects have correct attributes"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_config.enhanced.enable_pexels = True
        mock_config.enhanced.enable_pixabay = False

        mock_pexels.return_value = (['/path/ocean_waves.mp4'], {'ocean': 1})

        result = stage.run(state, mock_config, mock_checkpoint)

        # Check DownloadedVideo attributes
        assert len(state.downloaded_videos) == 1
        video = state.downloaded_videos[0]
        assert video.file == '/path/ocean_waves.mp4'
        assert video.source == 'stock'
        assert video.keyword == 'stock_footage'
        assert video.url == ''
        assert video.title == 'ocean_waves'
        assert video.duration == 0.0


# ============================================================================
# Test Stock Directory Creation
# ============================================================================

class TestStockDirectory:
    """Test stock directory creation"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_stock_directory_created(self, mock_pixabay, mock_pexels,
                                    mock_config, mock_checkpoint, mock_keywords, tmp_path):
        """Test stock directory is created"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_config.downloaded_videos_dir = str(tmp_path / "downloads")
        mock_config.enhanced.enable_pexels = True
        mock_config.enhanced.enable_pixabay = False

        mock_pexels.return_value = ([], {})

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify stock directory was created
        stock_dir = tmp_path / "downloads" / "stock"
        assert stock_dir.exists()
        assert stock_dir.is_dir()


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestStageExecution:
    """Test full stage execution"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_run_success(self, mock_pixabay, mock_pexels, mock_config,
                        mock_checkpoint, mock_keywords):
        """Test successful full execution"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_pexels.return_value = (['/path/video1.mp4'], {'ocean': 1})
        mock_pixabay.return_value = (['/path/video2.mp4'], {'sunset': 1})

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['stock_count'] == 2
        assert 'stock_paths' in result.data
        assert len(result.data['stock_paths']) == 2

    @patch.object(StockVideoStage, '_download_pexels')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_pexels, mock_config,
                                   mock_checkpoint, mock_keywords):
        """Test exception handling in main run"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_pexels.side_effect = Exception("Download failed")

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "failed" in result.error.lower()


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestCheckpointOperations:
    """Test checkpoint save/restore"""

    @pytest.mark.fast
    def test_can_skip_with_checkpoint(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint exists"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_checkpoint.should_skip_stage.assert_called_with("STOCK")

    @pytest.mark.fast
    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_success(self, mock_checkpoint, tmp_path):
        """Test successful restore from checkpoint"""
        stage = StockVideoStage()
        state = PipelineState()
        state.downloaded_videos = []

        # Create video files
        video1 = tmp_path / "stock1.mp4"
        video2 = tmp_path / "stock2.mp4"
        video1.touch()
        video2.touch()

        # Mock checkpoint data
        checkpoint_data = {
            'stock_count': 2,
            'stock_paths': [str(video1), str(video2)],
            'sources': {'pexels': 2, 'pixabay': 0}
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_videos) == 2
        assert all(v.source == 'stock' for v in state.downloaded_videos)

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore fails when no checkpoint data"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_missing_video_files(self, mock_checkpoint):
        """Test restore with missing video files"""
        stage = StockVideoStage()
        state = PipelineState()
        state.downloaded_videos = []

        # Mock checkpoint data with non-existent files
        checkpoint_data = {
            'stock_count': 2,
            'stock_paths': ['/nonexistent/video1.mp4', '/nonexistent/video2.mp4']
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        result = stage.restore(state, mock_checkpoint)

        # Should return False since no valid videos found
        assert result is False
        assert len(state.downloaded_videos) == 0

    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_checkpoint):
        """Test restore handles exceptions gracefully"""
        stage = StockVideoStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.side_effect = Exception("Restore failed")

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Helper Methods
# ============================================================================

class TestHelperMethods:
    """Test internal helper methods"""

    @patch('src.pexels.download_pexels_footage')
    @pytest.mark.fast
    def test_download_pexels_success(self, mock_download):
        """Test _download_pexels with successful download"""
        stage = StockVideoStage()
        mock_config = MagicMock()

        mock_download.return_value = (['/path/video1.mp4'], {'ocean': 1})

        paths, counts = stage._download_pexels(['ocean'], '/output', 3, mock_config)

        assert len(paths) == 1
        assert counts == {'ocean': 1}
        assert mock_download.called

    @patch('src.pexels.download_pexels_footage')
    @pytest.mark.fast
    def test_download_pexels_import_error(self, mock_download):
        """Test _download_pexels with ImportError"""
        stage = StockVideoStage()
        mock_config = MagicMock()

        mock_download.side_effect = ImportError("Module not found")

        paths, counts = stage._download_pexels(['ocean'], '/output', 3, mock_config)

        # Should return empty results on ImportError
        assert paths == []
        assert counts == {}

    @patch('src.pexels.download_pexels_footage')
    @pytest.mark.fast
    def test_download_pexels_exception(self, mock_download):
        """Test _download_pexels with general exception"""
        stage = StockVideoStage()
        mock_config = MagicMock()

        mock_download.side_effect = Exception("Download error")

        paths, counts = stage._download_pexels(['ocean'], '/output', 3, mock_config)

        # Should return empty results on exception
        assert paths == []
        assert counts == {}

    @patch('src.pixabay.download_pixabay_footage')
    @pytest.mark.fast
    def test_download_pixabay_success(self, mock_download):
        """Test _download_pixabay with successful download"""
        stage = StockVideoStage()
        mock_config = MagicMock()

        mock_download.return_value = (['/path/video2.mp4'], {'sunset': 1})

        paths, counts = stage._download_pixabay(['sunset'], '/output', 3, mock_config)

        assert len(paths) == 1
        assert counts == {'sunset': 1}
        assert mock_download.called

    @patch('src.pixabay.download_pixabay_footage')
    @pytest.mark.fast
    def test_download_pixabay_import_error(self, mock_download):
        """Test _download_pixabay with ImportError"""
        stage = StockVideoStage()
        mock_config = MagicMock()

        mock_download.side_effect = ImportError("Module not found")

        paths, counts = stage._download_pixabay(['sunset'], '/output', 3, mock_config)

        # Should return empty results on ImportError
        assert paths == []
        assert counts == {}


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_default_stock_per_keyword(self, mock_pixabay, mock_pexels,
                                       mock_config, mock_checkpoint, mock_keywords):
        """Test default stock_per_keyword value"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        # Remove stock_per_keyword from config
        delattr(mock_config.enhanced, 'stock_per_keyword')
        mock_pexels.return_value = ([], {})
        mock_pixabay.return_value = ([], {})

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should default to 3
        pexels_per_keyword = mock_pexels.call_args[0][2]
        assert pexels_per_keyword == 3

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_empty_download_results(self, mock_pixabay, mock_pexels,
                                   mock_config, mock_checkpoint, mock_keywords):
        """Test handling when both sources return no videos"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = mock_keywords
        state.downloaded_videos = []

        mock_pexels.return_value = ([], {})
        mock_pixabay.return_value = ([], {})

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['stock_count'] == 0
        assert len(state.downloaded_videos) == 0

    @patch.object(StockVideoStage, '_download_pexels')
    @patch.object(StockVideoStage, '_download_pixabay')
    @pytest.mark.fast
    def test_no_duplicate_failed_keywords(self, mock_pixabay, mock_pexels,
                                         mock_config, mock_checkpoint):
        """Test failed keywords are not duplicated"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['ocean', 'mountains']
        state.downloaded_videos = []
        state.failed_keywords = ['ocean']  # Already failed once

        mock_config.enhanced.enable_pexels = True
        mock_config.enhanced.enable_pixabay = True

        # Both sources fail for 'ocean'
        mock_pexels.return_value = ([], {'ocean': 0, 'mountains': 1})
        mock_pixabay.return_value = ([], {'ocean': 0, 'mountains': 1})

        result = stage.run(state, mock_config, mock_checkpoint)

        # 'ocean' should still only appear once in failed_keywords
        assert state.failed_keywords.count('ocean') == 1
