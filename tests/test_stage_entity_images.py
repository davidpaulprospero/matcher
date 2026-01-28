"""
Test Suite for EntityImagesStage

Tests the EntityImagesStage class which handles:
- Entity image downloads from Google, Bing, Pexels, Pixabay
- Entity filtering by type
- Local and global entity cache integration
- Entity-to-segment mapping
- Output directory management
- Checkpoint operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os

from src.stages.entity_images import EntityImagesStage
from src.state import PipelineState, VoiceoverSegment
from src.media_sources.models import EntityImageResult


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with image search settings"""
    config = MagicMock()
    config.pipeline.skip_image_search = False
    config.image_search.enabled = True
    config.image_search.entity_types = ['PERSON', 'ORG', 'LOC']
    config.image_search.max_entities = 0  # No limit
    config.image_search.images_per_entity = 3
    config.image_search.min_size_mb = 0.1
    config.image_search.use_google = True
    config.image_search.use_bing = False
    config.image_search.use_stock_apis = False
    config.image_search.root_dir = ""  # Project-relative mode
    config.image_search.folder_name = "images"
    config.image_search.download_timeout = 10
    config.image_search.max_search_time = 300
    config.image_search.max_results_to_check = 500
    config.image_search.search_until_found = True

    # Entity cache config
    cache_config = MagicMock()
    cache_config.enabled = False
    config.image_search.entity_cache = cache_config

    return config


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.project_dir = tmp_path
    checkpoint.refresh_entities = False
    return checkpoint


@pytest.fixture
def mock_entities():
    """Create mock extracted entities"""
    return [
        {'text': 'Albert Einstein', 'type': 'PERSON', 'context': 'physicist'},
        {'text': 'CERN', 'type': 'ORG', 'context': 'research organization'},
        {'text': 'Geneva', 'type': 'LOC', 'context': 'Swiss city'},
    ]


@pytest.fixture
def mock_voiceover_segments():
    """Create mock voiceover segments"""
    return [
        VoiceoverSegment(index=0, start=0.0, end=5.0, text="Albert Einstein was a physicist"),
        VoiceoverSegment(index=1, start=5.0, end=10.0, text="CERN is located in Geneva"),
        VoiceoverSegment(index=2, start=10.0, end=15.0, text="Einstein's work on relativity"),
    ]


@pytest.fixture
def mock_entity_results():
    """Create mock entity image results"""
    return {
        'Albert Einstein': EntityImageResult(
            entity_name='Albert Einstein',
            entity_type='PERSON',
            context='physicist',
            query='Albert Einstein physicist',
            images=['/path/einstein1.jpg', '/path/einstein2.jpg'],
            segment_indices=[0, 2]
        ),
        'CERN': EntityImageResult(
            entity_name='CERN',
            entity_type='ORG',
            context='research organization',
            query='CERN research organization',
            images=['/path/cern1.jpg'],
            segment_indices=[1]
        )
    }


# ============================================================================
# Test Stage Initialization
# ============================================================================

class TestEntityImagesStageInit:
    """Test stage initialization"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name"""
        stage = EntityImagesStage()
        assert stage.name == "ENTITY_IMAGES"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = EntityImagesStage()
        assert "entity images" in stage.description.lower()

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered in stage registry"""
        from src.stages import get_stage
        stage_class = get_stage("ENTITY_IMAGES")
        assert stage_class is EntityImagesStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    @pytest.mark.fast
    def test_validate_no_hard_requirements(self, mock_config):
        """Test validation has no hard requirements (optional stage)"""
        stage = EntityImagesStage()
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
    def test_skip_via_pipeline_config(self, mock_config, mock_checkpoint):
        """Test skipping when pipeline.skip_image_search is True"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_config.pipeline.skip_image_search = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'skip_pipeline_config'

    @pytest.mark.fast
    def test_skip_when_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when image_search.enabled is False"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_config.image_search.enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'disabled'

    @pytest.mark.fast
    def test_skip_when_no_entities(self, mock_config, mock_checkpoint):
        """Test skipping when no entities extracted"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_entities'

    @pytest.mark.fast
    def test_skip_when_no_matching_types(self, mock_config, mock_checkpoint):
        """Test skipping when no entities match configured types"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = [
            {'text': 'Something', 'type': 'DATE', 'context': 'temporal'},
        ]
        mock_config.image_search.entity_types = ['PERSON', 'ORG']

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_matching_types'


# ============================================================================
# Test Entity Filtering
# ============================================================================

class TestEntityFiltering:
    """Test entity filtering logic"""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_filter_by_entity_type(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_voiceover_segments):
        """Test filtering entities by configured types"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = [
            {'text': 'Einstein', 'type': 'PERSON', 'context': ''},
            {'text': '1905', 'type': 'DATE', 'context': ''},
            {'text': 'CERN', 'type': 'ORG', 'context': ''},
        ]
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_config.image_search.entity_types = ['PERSON', 'ORG']  # Exclude DATE
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should only search for PERSON and ORG, not DATE
        assert mock_download.called
        call_args = mock_download.call_args
        entities_searched = call_args[1]['entities']
        assert len(entities_searched) == 2
        assert all(e['type'] in ['PERSON', 'ORG'] for e in entities_searched)

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_max_entities_limit(self, mock_map, mock_download, mock_config,
                               mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test limiting number of entities processed"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities  # 3 entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_config.image_search.max_entities = 2  # Limit to 2
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should only process first 2 entities
        call_args = mock_download.call_args
        entities_searched = call_args[1]['entities']
        assert len(entities_searched) == 2


# ============================================================================
# Test Image Download
# ============================================================================

class TestImageDownload:
    """Test image download integration"""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_successful_download(self, mock_map, mock_download, mock_config,
                                 mock_checkpoint, mock_entities, mock_voiceover_segments,
                                 mock_entity_results):
        """Test successful image download"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_download.return_value = mock_entity_results
        mock_map.return_value = {
            'Albert Einstein': [0, 2],
            'CERN': [1]
        }

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.entity_images) == 2
        assert 'Albert Einstein' in state.entity_images
        assert 'CERN' in state.entity_images
        assert result.data['entity_count'] == 2
        assert result.data['total_images'] == 3

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_no_images_downloaded(self, mock_map, mock_download, mock_config,
                                  mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test handling when no images are downloaded"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_download.return_value = {}  # No results
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert state.entity_images == {}
        assert result.data['entity_count'] == 0

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.requires_api
    def test_download_with_api_keys(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test download passes API keys from environment"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_download.return_value = {}
        mock_map.return_value = {}

        with patch.dict(os.environ, {'PEXELS_API_KEY': 'test_pexels', 'PIXABAY_API_KEY': 'test_pixabay'}):
            result = stage.run(state, mock_config, mock_checkpoint)

        # Verify API keys were passed
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['pexels_key'] == 'test_pexels'
        assert call_kwargs['pixabay_key'] == 'test_pixabay'


# ============================================================================
# Test Entity-to-Segment Mapping
# ============================================================================

class TestEntitySegmentMapping:
    """Test entity-to-segment mapping"""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_segment_indices_mapped(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test entity results updated with segment indices"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        # Create results without segment_indices
        results = {
            'Albert Einstein': EntityImageResult(
                entity_name='Albert Einstein',
                entity_type='PERSON',
                context='',
                query='Albert Einstein',
                images=['/path/image.jpg'],
                segment_indices=[]  # Empty initially
            )
        }
        mock_download.return_value = results
        mock_map.return_value = {'Albert Einstein': [0, 2]}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Segment indices should be updated
        assert state.entity_images['Albert Einstein'].segment_indices == [0, 2]


# ============================================================================
# Test Output Directory Management
# ============================================================================

class TestOutputDirectory:
    """Test output directory determination"""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_project_relative_mode(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test project-relative output directory"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_config.image_search.root_dir = ""  # Project-relative
        mock_config.image_search.folder_name = "images"
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify output_dir passed to download function
        call_kwargs = mock_download.call_args[1]
        output_dir = Path(call_kwargs['output_dir'])
        assert output_dir == mock_checkpoint.project_dir / "images"

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_short_path_mode(self, mock_map, mock_download, mock_config,
                            mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test short path output directory"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_config.image_search.root_dir = "E:/i"  # Short path
        mock_checkpoint.project_dir = Path("C:/Long/Path/To/Project__2026-01-09")
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify output_dir uses short path
        call_kwargs = mock_download.call_args[1]
        output_dir = Path(call_kwargs['output_dir'])
        # Path may have backslashes on Windows
        output_str = str(output_dir).replace('\\', '/')
        assert output_str.startswith("E:/i")
        # Project name should be truncated to 15 chars
        assert len(output_dir.name) <= 15


# ============================================================================
# Test Entity Cache
# ============================================================================

class TestEntityCache:
    """Test global entity cache integration"""

    @patch('src.entity_cache.EntityCache')
    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_cache_enabled(self, mock_map, mock_download, mock_cache_class,
                          mock_config, mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test entity cache initialization when enabled"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        # Enable cache
        mock_config.image_search.entity_cache.enabled = True

        # Mock cache instance
        mock_cache = Mock()
        mock_cache.get_stats.return_value = {'total_entities': 50}
        mock_cache_class.return_value = mock_cache

        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Cache should be initialized
        assert mock_cache_class.called
        # Cache should be passed to download function
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['entity_cache'] == mock_cache

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_cache_disabled(self, mock_map, mock_download, mock_config,
                           mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test no cache when disabled"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_config.image_search.entity_cache.enabled = False
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Cache should be None
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['entity_cache'] is None

    @patch('src.entity_cache.EntityCache')
    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_cache_initialization_failure(self, mock_map, mock_download, mock_cache_class,
                                         mock_config, mock_checkpoint, mock_entities,
                                         mock_voiceover_segments):
        """Test handling cache initialization failure gracefully"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_config.image_search.entity_cache.enabled = True
        mock_cache_class.side_effect = Exception("Cache init failed")
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should continue without cache
        assert result.success is True
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['entity_cache'] is None


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestStageExecution:
    """Test full stage execution"""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_run_success(self, mock_map, mock_download, mock_config,
                        mock_checkpoint, mock_entities, mock_voiceover_segments,
                        mock_entity_results):
        """Test successful full execution"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_download.return_value = mock_entity_results
        mock_map.return_value = {
            'Albert Einstein': [0, 2],
            'CERN': [1]
        }

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.entity_images) == 2
        assert result.data['entity_count'] == 2
        assert result.data['total_images'] == 3
        assert 'entities' in result.data

    @pytest.mark.fast
    def test_run_import_error(self, mock_config, mock_checkpoint, mock_entities):
        """Test handling import errors"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities

        # Patch the import statement inside run()
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "import" in result.error.lower()

    @patch('src.media_sources.download_entity_images')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test exception handling in main run"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_download.side_effect = Exception("Download failed")

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "download failed" in result.error.lower()


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestCheckpointOperations:
    """Test checkpoint save/restore"""

    @pytest.mark.fast
    def test_can_skip_with_checkpoint(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint exists"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_checkpoint.should_skip_stage.assert_called_with("ENTITY_IMAGES")

    @pytest.mark.fast
    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    @patch('src.media_sources.restore_entity_images_from_disk')
    @pytest.mark.fast
    def test_restore_success(self, mock_restore, mock_checkpoint, mock_voiceover_segments):
        """Test successful restore from checkpoint"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments

        # Mock checkpoint data
        checkpoint_data = {
            'entity_count': 2,
            'total_images': 3
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        # Mock config
        config = MagicMock()
        config.image_search.root_dir = ""
        config.image_search.folder_name = "images"
        mock_checkpoint._config = config

        # Create images directory
        images_dir = mock_checkpoint.project_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Mock restore function
        restored_results = {
            'Einstein': EntityImageResult(
                entity_name='Einstein',
                entity_type='PERSON',
                context='',
                query='',
                images=['/path/image.jpg'],
                segment_indices=[0]
            )
        }
        mock_restore.return_value = restored_results

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.entity_images) == 1
        assert 'Einstein' in state.entity_images

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore fails when no checkpoint data"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_no_config(self, mock_checkpoint):
        """Test restore fails when no config available"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {'entity_count': 1}
        mock_checkpoint._config = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @patch('src.media_sources.restore_entity_images_from_disk')
    @pytest.mark.fast
    def test_restore_no_images_dir(self, mock_restore, mock_checkpoint):
        """Test restore fails when images directory doesn't exist"""
        stage = EntityImagesStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {'entity_count': 1}

        config = MagicMock()
        config.image_search.root_dir = ""
        config.image_search.folder_name = "images"
        mock_checkpoint._config = config

        # Don't create images directory

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @patch('src.media_sources.restore_entity_images_from_disk')
    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_restore, mock_checkpoint, mock_voiceover_segments):
        """Test restore handles exceptions gracefully"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments

        mock_checkpoint.get_stage_data.return_value = {'entity_count': 1}
        config = MagicMock()
        config.image_search.root_dir = ""
        config.image_search.folder_name = "images"
        mock_checkpoint._config = config

        images_dir = mock_checkpoint.project_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        mock_restore.side_effect = Exception("Restore failed")

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_empty_topic_context(self, mock_map, mock_download, mock_config,
                                 mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test handling empty topic context"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = ""  # Empty

        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should pass empty string as topic
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['topic'] == ""

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_refresh_entities_flag(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test skip_local_cache flag when refresh_entities is True"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_checkpoint.refresh_entities = True
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should pass skip_local_cache=True
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['skip_local_cache'] is True

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_project_name_truncation(self, mock_map, mock_download, mock_config,
                                     mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test project name is truncated to 15 chars in short path mode"""
        stage = EntityImagesStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_config.image_search.root_dir = "E:/i"
        mock_checkpoint.project_dir = Path("C:/VeryLongProjectNameThatExceeds15Characters__2026-01-09")
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify project name truncated
        call_kwargs = mock_download.call_args[1]
        output_dir = Path(call_kwargs['output_dir'])
        assert len(output_dir.name) == 15

    @patch('src.media_sources.download_entity_images')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_many_entities_display_limit(self, mock_map, mock_download, mock_config,
                                         mock_checkpoint, mock_voiceover_segments):
        """Test display limit for entity results (shows first 5)"""
        stage = EntityImagesStage()
        state = PipelineState()

        # Create 10 entities
        state.extracted_entities = [
            {'text': f'Entity{i}', 'type': 'PERSON', 'context': ''}
            for i in range(10)
        ]
        state.voiceover_segments = mock_voiceover_segments

        # Create 10 results
        results = {
            f'Entity{i}': EntityImageResult(
                entity_name=f'Entity{i}',
                entity_type='PERSON',
                context='',
                query=f'Entity{i}',
                images=[f'/path/image{i}.jpg'],
                segment_indices=[]
            )
            for i in range(10)
        }
        mock_download.return_value = results
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should succeed with all 10 entities
        assert result.success is True
        assert result.data['entity_count'] == 10
