"""
Test Suite for EntityVideosStage

Tests the EntityVideosStage class which handles:
- Entity stock video downloads from Pexels/Pixabay
- Entity filtering by type
- Entity-to-segment mapping
- Output directory management
- Checkpoint operations
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import os

from src.stages.entity_videos import EntityVideosStage
from src.state import PipelineState, VoiceoverSegment
from src.media_sources.models import EntityVideoResult


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with video search settings"""
    config = MagicMock()
    config.pipeline.skip_image_search = False
    config.image_search.enabled = True
    config.image_search.use_stock_apis = True
    config.image_search.entity_types = ['PERSON', 'ORG', 'LOC']
    config.image_search.max_entities = 0  # No limit
    config.image_search.videos_per_entity = 3
    config.image_search.root_dir = ""  # Project-relative mode
    config.image_search.folder_name = "images"
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
def mock_entity_video_results():
    """Create mock entity video results"""
    return {
        'Albert Einstein': EntityVideoResult(
            entity_name='Albert Einstein',
            entity_type='PERSON',
            context='physicist',
            query='Albert Einstein physicist',
            videos=['/path/einstein1.mp4', '/path/einstein2.mp4'],
            segment_indices=[0, 2]
        ),
        'CERN': EntityVideoResult(
            entity_name='CERN',
            entity_type='ORG',
            context='research organization',
            query='CERN research organization',
            videos=['/path/cern1.mp4'],
            segment_indices=[1]
        )
    }


# ============================================================================
# Test Stage Initialization
# ============================================================================

class TestEntityVideosStageInit:
    """Test stage initialization"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name"""
        stage = EntityVideosStage()
        assert stage.name == "ENTITY_VIDEOS"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = EntityVideosStage()
        assert "stock videos" in stage.description.lower()

    @pytest.mark.fast
    def test_stage_not_registered(self):
        """Test stage is NOT in the registry (optional stage, no @register_stage)"""
        from src.stages import get_stage
        stage_class = get_stage("ENTITY_VIDEOS")
        # Entity stages are optional and not auto-registered
        assert stage_class is None


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    @pytest.mark.fast
    def test_validate_no_hard_requirements(self, mock_config):
        """Test validation has no hard requirements (optional stage)"""
        stage = EntityVideosStage()
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
        stage = EntityVideosStage()
        state = PipelineState()
        mock_config.pipeline.skip_image_search = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'skip_pipeline_config'

    @pytest.mark.fast
    def test_skip_when_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when image_search.enabled is False"""
        stage = EntityVideosStage()
        state = PipelineState()
        mock_config.image_search.enabled = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'disabled'

    @pytest.mark.fast
    def test_skip_when_stock_apis_disabled(self, mock_config, mock_checkpoint):
        """Test skipping when use_stock_apis is False"""
        stage = EntityVideosStage()
        state = PipelineState()
        mock_config.image_search.use_stock_apis = False

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'stock_apis_disabled'

    @pytest.mark.fast
    def test_skip_when_no_entities(self, mock_config, mock_checkpoint):
        """Test skipping when no entities extracted"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'no_entities'

    @pytest.mark.fast
    def test_skip_when_no_matching_types(self, mock_config, mock_checkpoint):
        """Test skipping when no entities match configured types"""
        stage = EntityVideosStage()
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

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_filter_by_entity_type(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_voiceover_segments):
        """Test filtering entities by configured types"""
        stage = EntityVideosStage()
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

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_max_entities_limit(self, mock_map, mock_download, mock_config,
                               mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test limiting number of entities processed"""
        stage = EntityVideosStage()
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
# Test Video Download
# ============================================================================

class TestVideoDownload:
    """Test video download integration"""

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_successful_download(self, mock_map, mock_download, mock_config,
                                 mock_checkpoint, mock_entities, mock_voiceover_segments,
                                 mock_entity_video_results):
        """Test successful video download"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_download.return_value = mock_entity_video_results
        mock_map.return_value = {
            'Albert Einstein': [0, 2],
            'CERN': [1]
        }

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.entity_videos) == 2
        assert 'Albert Einstein' in state.entity_videos
        assert 'CERN' in state.entity_videos
        assert result.data['video_count'] == 2
        assert result.data['total_videos'] == 3

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_no_videos_downloaded(self, mock_map, mock_download, mock_config,
                                  mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test handling when no videos are downloaded"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_download.return_value = {}  # No results
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert state.entity_videos == {}
        assert result.data['video_count'] == 0

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.requires_api
    def test_download_with_api_keys(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test download passes API keys from environment"""
        stage = EntityVideosStage()
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

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_download_with_duration_params(self, mock_map, mock_download, mock_config,
                                          mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test download passes min/max duration parameters"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify duration parameters
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['min_duration'] == 3.0
        assert call_kwargs['max_duration'] == 30.0

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_videos_per_entity_config(self, mock_map, mock_download, mock_config,
                                     mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test videos_per_entity config parameter"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_config.image_search.videos_per_entity = 5
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Verify videos_per_entity was passed
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['videos_per_entity'] == 5


# ============================================================================
# Test Entity-to-Segment Mapping
# ============================================================================

class TestEntitySegmentMapping:
    """Test entity-to-segment mapping"""

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_segment_indices_mapped(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test entity results updated with segment indices"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        # Create results without segment_indices
        results = {
            'Albert Einstein': EntityVideoResult(
                entity_name='Albert Einstein',
                entity_type='PERSON',
                context='',
                query='Albert Einstein',
                videos=['/path/video.mp4'],
                segment_indices=[]  # Empty initially
            )
        }
        mock_download.return_value = results
        mock_map.return_value = {'Albert Einstein': [0, 2]}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Segment indices should be updated
        assert state.entity_videos['Albert Einstein'].segment_indices == [0, 2]


# ============================================================================
# Test Output Directory Management
# ============================================================================

class TestOutputDirectory:
    """Test output directory determination"""

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_project_relative_mode(self, mock_map, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test project-relative output directory"""
        stage = EntityVideosStage()
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

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_short_path_mode(self, mock_map, mock_download, mock_config,
                            mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test short path output directory"""
        stage = EntityVideosStage()
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
# Test Stage Execution
# ============================================================================

class TestStageExecution:
    """Test full stage execution"""

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_run_success(self, mock_map, mock_download, mock_config,
                        mock_checkpoint, mock_entities, mock_voiceover_segments,
                        mock_entity_video_results):
        """Test successful full execution"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments
        state.topic_context = "physics"

        mock_download.return_value = mock_entity_video_results
        mock_map.return_value = {
            'Albert Einstein': [0, 2],
            'CERN': [1]
        }

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.entity_videos) == 2
        assert result.data['video_count'] == 2
        assert result.data['total_videos'] == 3
        assert 'entities' in result.data

    @pytest.mark.fast
    def test_run_import_error(self, mock_config, mock_checkpoint, mock_entities):
        """Test handling import errors"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities

        # Patch the import statement inside run()
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "import" in result.error.lower()

    @patch('src.media_sources.download_entity_videos')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_download, mock_config,
                                   mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test exception handling in main run"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        mock_download.side_effect = Exception("Download failed")

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
        stage = EntityVideosStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_checkpoint.should_skip_stage.assert_called_with("ENTITY_VIDEOS")

    @pytest.mark.fast
    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = EntityVideosStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_success(self, mock_checkpoint, tmp_path):
        """Test successful restore from checkpoint"""
        stage = EntityVideosStage()
        state = PipelineState()

        # Create video files
        video1 = tmp_path / "einstein.mp4"
        video2 = tmp_path / "cern.mp4"
        video1.touch()
        video2.touch()

        # Mock checkpoint data
        checkpoint_data = {
            'video_count': 2,
            'entities': {
                'Einstein': {
                    'entity_type': 'PERSON',
                    'context': 'physicist',
                    'query': 'Einstein',
                    'videos': [str(video1)],
                    'segment_indices': [0]
                },
                'CERN': {
                    'entity_type': 'ORG',
                    'context': 'research',
                    'query': 'CERN',
                    'videos': [str(video2)],
                    'segment_indices': [1]
                }
            }
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        # Mock config
        config = MagicMock()
        mock_checkpoint._config = config

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.entity_videos) == 2
        assert 'Einstein' in state.entity_videos
        assert 'CERN' in state.entity_videos

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore fails when no checkpoint data"""
        stage = EntityVideosStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_no_config(self, mock_checkpoint):
        """Test restore fails when no config available"""
        stage = EntityVideosStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {'video_count': 1}
        mock_checkpoint._config = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_missing_video_files(self, mock_checkpoint):
        """Test restore skips entities with missing video files"""
        stage = EntityVideosStage()
        state = PipelineState()

        # Mock checkpoint data with non-existent files
        checkpoint_data = {
            'video_count': 1,
            'entities': {
                'Einstein': {
                    'entity_type': 'PERSON',
                    'context': '',
                    'query': 'Einstein',
                    'videos': ['/nonexistent/video.mp4'],
                    'segment_indices': []
                }
            }
        }
        mock_checkpoint.get_stage_data.return_value = checkpoint_data

        config = MagicMock()
        mock_checkpoint._config = config

        result = stage.restore(state, mock_checkpoint)

        # Should return False since no valid videos found
        assert result is False
        assert len(state.entity_videos) == 0

    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_checkpoint):
        """Test restore handles exceptions gracefully"""
        stage = EntityVideosStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.side_effect = Exception("Restore failed")

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_empty_topic_context(self, mock_map, mock_download, mock_config,
                                 mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test handling empty topic context"""
        stage = EntityVideosStage()
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

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_default_videos_per_entity(self, mock_map, mock_download, mock_config,
                                      mock_checkpoint, mock_entities, mock_voiceover_segments):
        """Test default videos_per_entity value when not configured"""
        stage = EntityVideosStage()
        state = PipelineState()
        state.extracted_entities = mock_entities
        state.voiceover_segments = mock_voiceover_segments

        # Remove videos_per_entity from config
        delattr(mock_config.image_search, 'videos_per_entity')
        mock_download.return_value = {}
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should default to 3
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['videos_per_entity'] == 3

    @patch('src.media_sources.download_entity_videos')
    @patch('src.media_sources.map_entities_to_segments')
    @pytest.mark.fast
    def test_many_entities_display_limit(self, mock_map, mock_download, mock_config,
                                         mock_checkpoint, mock_voiceover_segments):
        """Test display limit for entity results (shows first 5)"""
        stage = EntityVideosStage()
        state = PipelineState()

        # Create 10 entities
        state.extracted_entities = [
            {'text': f'Entity{i}', 'type': 'PERSON', 'context': ''}
            for i in range(10)
        ]
        state.voiceover_segments = mock_voiceover_segments

        # Create 10 results
        results = {
            f'Entity{i}': EntityVideoResult(
                entity_name=f'Entity{i}',
                entity_type='PERSON',
                context='',
                query=f'Entity{i}',
                videos=[f'/path/video{i}.mp4'],
                segment_indices=[]
            )
            for i in range(10)
        }
        mock_download.return_value = results
        mock_map.return_value = {}

        result = stage.run(state, mock_config, mock_checkpoint)

        # Should succeed with all 10 entities
        assert result.success is True
        assert result.data['video_count'] == 10
