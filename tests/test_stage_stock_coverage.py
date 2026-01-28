"""
Additional coverage tests for src/stages/stock.py

Tests cover edge cases:
- Skip download config
- Enhanced features disabled
- No stock sources enabled
- No keywords
- Pexels download with failed keywords
- Pixabay download with failed keywords
- Exception handling
- Restore from checkpoint
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.stock import StockVideoStage
from src.state import PipelineState


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config"""
    config = MagicMock()
    config.pipeline.skip_download = False
    config.enhanced = MagicMock()
    config.enhanced.enabled = True
    config.enhanced.enable_pexels = True
    config.enhanced.enable_pixabay = True
    config.enhanced.stock_per_keyword = 3
    config.downloaded_videos_dir = "/tmp/videos"
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    return checkpoint


# ============================================================================
# Test Skip Conditions
# ============================================================================

class TestStockStageSkip:
    """Test skip conditions"""

    @pytest.mark.fast
    def test_skip_download_enabled(self, mock_config, mock_checkpoint):
        """Test skipping when skip_download=true"""
        stage = StockVideoStage()
        state = PipelineState()

        mock_config.pipeline.skip_download = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'skip_download_enabled'

    @pytest.mark.fast
    def test_enhanced_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when enhanced features disabled"""
        stage = StockVideoStage()
        state = PipelineState()

        mock_config.enhanced.enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'enhanced_disabled'

    @pytest.mark.fast
    def test_no_enhanced_config(self, mock_config, mock_checkpoint):
        """Test skipping when no enhanced config"""
        stage = StockVideoStage()
        state = PipelineState()

        mock_config.enhanced = None

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True

    @pytest.mark.fast
    def test_no_stock_sources(self, mock_config, mock_checkpoint):
        """Test skipping when no stock sources enabled"""
        stage = StockVideoStage()
        state = PipelineState()

        mock_config.enhanced.enable_pexels = False
        mock_config.enhanced.enable_pixabay = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_stock_sources_enabled'

    @pytest.mark.fast
    def test_no_keywords(self, mock_config, mock_checkpoint):
        """Test skipping when no keywords"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_keywords'


# ============================================================================
# Test Validate Inputs
# ============================================================================

class TestStockValidateInputs:
    """Test validate_inputs method"""

    @pytest.mark.fast
    def test_validate_returns_none(self, mock_config):
        """Test validation always returns None (optional stage)"""
        stage = StockVideoStage()
        state = PipelineState()

        error = stage.validate_inputs(state, mock_config)

        assert error is None


# ============================================================================
# Test Pexels Download
# ============================================================================

class TestPexelsDownload:
    """Test _download_pexels method"""

    @pytest.mark.fast
    def test_download_pexels_success(self, mock_config):
        """Test successful Pexels download"""
        stage = StockVideoStage()

        # Patch where it's imported inside the method
        with patch.dict('sys.modules', {'src.pexels': MagicMock()}):
            import sys
            mock_module = sys.modules['src.pexels']
            mock_module.download_pexels_footage = Mock(return_value=(
                ['video1.mp4', 'video2.mp4'],
                {'beach': 2, 'ocean': 1}
            ))

            paths, counts = stage._download_pexels(
                keywords=['beach', 'ocean'],
                output_dir='/tmp/stock',
                per_keyword=3,
                config=mock_config
            )

            assert paths == ['video1.mp4', 'video2.mp4']
            assert counts == {'beach': 2, 'ocean': 1}

    @pytest.mark.fast
    def test_download_pexels_import_error(self, mock_config, caplog):
        """Test Pexels download when module not available"""
        import logging

        stage = StockVideoStage()

        with patch.dict('sys.modules', {'src.pexels': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with caplog.at_level(logging.WARNING):
                    paths, counts = stage._download_pexels(
                        keywords=['beach'],
                        output_dir='/tmp/stock',
                        per_keyword=3,
                        config=mock_config
                    )

        assert paths == []
        assert counts == {}

    @pytest.mark.fast
    def test_download_pexels_exception(self, mock_config, caplog):
        """Test Pexels download exception handling"""
        import logging

        stage = StockVideoStage()

        # Patch where it's imported inside the method
        with patch.dict('sys.modules', {'src.pexels': MagicMock()}):
            import sys
            mock_module = sys.modules['src.pexels']
            mock_module.download_pexels_footage = Mock(side_effect=Exception("API error"))

            with caplog.at_level(logging.ERROR):
                paths, counts = stage._download_pexels(
                    keywords=['beach'],
                    output_dir='/tmp/stock',
                    per_keyword=3,
                    config=mock_config
                )

        assert paths == []
        assert counts == {}
        assert "Pexels error" in caplog.text or "API error" in caplog.text


# ============================================================================
# Test Pixabay Download
# ============================================================================

class TestPixabayDownload:
    """Test _download_pixabay method"""

    @pytest.mark.fast
    def test_download_pixabay_success(self, mock_config):
        """Test successful Pixabay download"""
        stage = StockVideoStage()

        # Patch where it's imported inside the method
        with patch.dict('sys.modules', {'src.pixabay': MagicMock()}):
            import sys
            mock_module = sys.modules['src.pixabay']
            mock_module.download_pixabay_footage = Mock(return_value=(
                ['pixabay1.mp4', 'pixabay2.mp4'],
                {'mountain': 2}
            ))

            paths, counts = stage._download_pixabay(
                keywords=['mountain'],
                output_dir='/tmp/stock',
                per_keyword=3,
                config=mock_config
            )

            assert paths == ['pixabay1.mp4', 'pixabay2.mp4']
            assert counts == {'mountain': 2}

    @pytest.mark.fast
    def test_download_pixabay_import_error(self, mock_config, caplog):
        """Test Pixabay download when module not available"""
        import logging

        stage = StockVideoStage()

        with patch.dict('sys.modules', {'src.pixabay': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with caplog.at_level(logging.WARNING):
                    paths, counts = stage._download_pixabay(
                        keywords=['mountain'],
                        output_dir='/tmp/stock',
                        per_keyword=3,
                        config=mock_config
                    )

        assert paths == []
        assert counts == {}

    @pytest.mark.fast
    def test_download_pixabay_exception(self, mock_config, caplog):
        """Test Pixabay download exception handling"""
        import logging

        stage = StockVideoStage()

        # Patch where it's imported inside the method
        with patch.dict('sys.modules', {'src.pixabay': MagicMock()}):
            import sys
            mock_module = sys.modules['src.pixabay']
            mock_module.download_pixabay_footage = Mock(side_effect=Exception("API error"))

            with caplog.at_level(logging.ERROR):
                paths, counts = stage._download_pixabay(
                    keywords=['mountain'],
                    output_dir='/tmp/stock',
                    per_keyword=3,
                    config=mock_config
                )

        assert paths == []
        assert counts == {}


# ============================================================================
# Test Full Run
# ============================================================================

class TestStockStageRun:
    """Test full run method"""

    @pytest.mark.fast
    def test_run_success_both_sources(self, mock_config, mock_checkpoint, tmp_path):
        """Test successful run with both Pexels and Pixabay"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach', 'ocean', 'mountain']
        state.downloaded_videos = []
        state.failed_keywords = []

        mock_config.downloaded_videos_dir = str(tmp_path)

        with patch.object(stage, '_download_pexels') as mock_pexels:
            with patch.object(stage, '_download_pixabay') as mock_pixabay:
                mock_pexels.return_value = (
                    [str(tmp_path / 'pexels1.mp4')],
                    {'beach': 1, 'ocean': 0}  # ocean failed
                )
                mock_pixabay.return_value = (
                    [str(tmp_path / 'pixabay1.mp4')],
                    {'mountain': 1, 'ocean': 1}
                )

                result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['stock_count'] == 2
        assert 'ocean' in state.failed_keywords  # Failed in Pexels
        assert len(state.downloaded_videos) == 2

    @pytest.mark.fast
    def test_run_pexels_only(self, mock_config, mock_checkpoint, tmp_path):
        """Test run with only Pexels enabled"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach']
        state.downloaded_videos = []
        state.failed_keywords = []

        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.enhanced.enable_pixabay = False

        with patch.object(stage, '_download_pexels') as mock_pexels:
            mock_pexels.return_value = ([str(tmp_path / 'pexels1.mp4')], {'beach': 1})

            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['sources']['pexels'] == 1
        assert result.data['sources']['pixabay'] == 0

    @pytest.mark.fast
    def test_run_pixabay_only(self, mock_config, mock_checkpoint, tmp_path):
        """Test run with only Pixabay enabled"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach']
        state.downloaded_videos = []
        state.failed_keywords = []

        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.enhanced.enable_pexels = False

        with patch.object(stage, '_download_pixabay') as mock_pixabay:
            mock_pixabay.return_value = ([str(tmp_path / 'pixabay1.mp4')], {'beach': 1})

            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['sources']['pexels'] == 0
        assert result.data['sources']['pixabay'] == 1

    @pytest.mark.fast
    def test_run_exception(self, mock_config, mock_checkpoint, tmp_path):
        """Test run exception handling"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach']
        state.downloaded_videos = []

        mock_config.downloaded_videos_dir = str(tmp_path)

        with patch.object(stage, '_download_pexels', side_effect=Exception("Fatal error")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "Fatal error" in result.error

    @pytest.mark.fast
    def test_run_deduplicates_videos(self, mock_config, mock_checkpoint, tmp_path):
        """Test that duplicate videos are not added"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach']
        state.failed_keywords = []

        # Pre-populate with existing video
        from src.state import DownloadedVideo
        existing = DownloadedVideo(file=str(tmp_path / 'existing.mp4'), source='stock')
        state.downloaded_videos = [existing]

        mock_config.downloaded_videos_dir = str(tmp_path)

        with patch.object(stage, '_download_pexels') as mock_pexels:
            # Return same path as existing
            mock_pexels.return_value = ([str(tmp_path / 'existing.mp4')], {'beach': 1})
            mock_config.enhanced.enable_pixabay = False

            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # Should still only have 1 video (no duplicate)
        assert len(state.downloaded_videos) == 1

    @pytest.mark.fast
    def test_run_limits_keywords(self, mock_config, mock_checkpoint, tmp_path):
        """Test that keywords are limited to 15"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = [f'keyword{i}' for i in range(20)]  # 20 keywords
        state.downloaded_videos = []
        state.failed_keywords = []

        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.enhanced.enable_pixabay = False

        with patch.object(stage, '_download_pexels') as mock_pexels:
            mock_pexels.return_value = ([], {})

            stage.run(state, mock_config, mock_checkpoint)

        # Should only have passed 15 keywords
        call_args = mock_pexels.call_args[0]
        assert len(call_args[0]) == 15

    @pytest.mark.fast
    def test_run_tracks_failed_keywords_from_pixabay(self, mock_config, mock_checkpoint, tmp_path):
        """Test line 141: Failed keywords are tracked when pixabay returns 0 results"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach', 'mountain', 'desert']
        state.downloaded_videos = []
        state.failed_keywords = []

        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.enhanced.enable_pexels = False  # Disable pexels

        with patch.object(stage, '_download_pixabay') as mock_pixabay:
            # Return 0 results for 'mountain' and 'desert', 1 for 'beach'
            mock_pixabay.return_value = (
                [str(tmp_path / 'beach.mp4')],
                {'beach': 1, 'mountain': 0, 'desert': 0}
            )

            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # mountain and desert should be in failed_keywords
        assert 'mountain' in state.failed_keywords
        assert 'desert' in state.failed_keywords
        assert 'beach' not in state.failed_keywords

    @pytest.mark.fast
    def test_run_does_not_duplicate_failed_keywords(self, mock_config, mock_checkpoint, tmp_path):
        """Test line 141: Already failed keywords are not duplicated"""
        stage = StockVideoStage()
        state = PipelineState()
        state.keywords = ['beach', 'mountain']
        state.downloaded_videos = []
        state.failed_keywords = ['mountain']  # Already failed

        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.enhanced.enable_pexels = False

        with patch.object(stage, '_download_pixabay') as mock_pixabay:
            # Return 0 for mountain again
            mock_pixabay.return_value = ([], {'beach': 0, 'mountain': 0})

            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # mountain should appear only once
        assert state.failed_keywords.count('mountain') == 1
        # beach should be added
        assert 'beach' in state.failed_keywords


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestStockCheckpoint:
    """Test checkpoint operations"""

    @pytest.mark.fast
    def test_can_skip_false(self, mock_checkpoint):
        """Test can_skip returns False when not in checkpoint"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_can_skip_true(self, mock_checkpoint):
        """Test can_skip returns True when in checkpoint"""
        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint, caplog):
        """Test restore returns False when no data"""
        import logging

        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_with_existing_files(self, mock_checkpoint, tmp_path):
        """Test restore with existing stock files"""
        stage = StockVideoStage()
        state = PipelineState()
        state.downloaded_videos = []

        # Create test files
        stock_file = tmp_path / 'stock1.mp4'
        stock_file.write_bytes(b'fake video')

        mock_checkpoint.get_stage_data.return_value = {
            'stock_paths': [str(stock_file), str(tmp_path / 'missing.mp4')]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].source == 'stock'

    @pytest.mark.fast
    def test_restore_no_existing_files(self, mock_checkpoint, tmp_path, caplog):
        """Test restore returns False when no files exist"""
        import logging

        stage = StockVideoStage()
        state = PipelineState()
        state.downloaded_videos = []

        mock_checkpoint.get_stage_data.return_value = {
            'stock_paths': [str(tmp_path / 'missing1.mp4'), str(tmp_path / 'missing2.mp4')]
        }

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, mock_checkpoint)

        assert result is False
        assert "No stock videos found" in caplog.text

    @pytest.mark.fast
    def test_restore_exception(self, mock_checkpoint, caplog):
        """Test restore handles exceptions"""
        import logging

        stage = StockVideoStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.side_effect = Exception("Restore error")

        with caplog.at_level(logging.ERROR):
            result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_deduplicates_videos(self, mock_checkpoint, tmp_path):
        """Test restore doesn't add duplicate videos"""
        stage = StockVideoStage()
        state = PipelineState()

        # Pre-populate with existing video
        from src.state import DownloadedVideo
        stock_file = tmp_path / 'stock1.mp4'
        stock_file.write_bytes(b'fake video')

        existing = DownloadedVideo(file=str(stock_file), source='stock')
        state.downloaded_videos = [existing]

        mock_checkpoint.get_stage_data.return_value = {
            'stock_paths': [str(stock_file)]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        # Should still only have 1 video (no duplicate)
        assert len(state.downloaded_videos) == 1
