"""
Tests for src/stages/scene_detection.py to improve coverage.

Focuses on:
- Silent video text_metadata creation
- Vision API integration
- Embedding computation for silent videos
- Scene metadata merging edge cases
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.scene_detection import SceneDetectionStage
from src.state import PipelineState, DownloadedVideo


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockSceneData:
    """Mock VideoSceneData"""
    scene_count: int
    scenes: list
    has_speech: bool = False


@dataclass
class MockScene:
    """Mock Scene"""
    scene_index: int
    start_time: float
    end_time: float
    is_broll: bool = False
    face_score: float = 0.5


@dataclass
class MockTranscriptSegment:
    """Mock transcript segment with scene attributes"""
    start_time: float
    end_time: float
    text: str
    is_broll: bool = False
    face_score: float = 0.5
    scene_index: int = 0


# ============================================================================
# Test SceneDetectionStage Basic Methods
# ============================================================================

class TestSceneDetectionStageBasics:
    """Test basic SceneDetectionStage methods"""

    def test_stage_name(self):
        """Test stage name is correct"""
        stage = SceneDetectionStage()
        assert stage.name == "SCENE_DETECTION"

    def test_stage_description(self):
        """Test stage description"""
        stage = SceneDetectionStage()
        assert "scene" in stage.description.lower()

    def test_validate_inputs_no_videos(self):
        """Test validate_inputs with no videos"""
        stage = SceneDetectionStage()
        state = PipelineState()
        config = Mock()

        result = stage.validate_inputs(state, config)

        assert result is not None
        assert "No videos" in result

    def test_validate_inputs_with_videos(self):
        """Test validate_inputs with videos present"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="test.mp4")]
        config = Mock()

        result = stage.validate_inputs(state, config)

        assert result is None  # No error

    def test_validate_inputs_with_audio_only(self):
        """Test validate_inputs with audio downloads only"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_audio = [Mock(file="test.mp3")]
        config = Mock()

        result = stage.validate_inputs(state, config)

        assert result is None  # Audio downloads count

    def test_can_skip_checks_checkpoint(self):
        """Test can_skip checks checkpoint"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, checkpoint)

        assert result is True
        checkpoint.should_skip_stage.assert_called_with("SCENE_DETECTION")


# ============================================================================
# Test _get_video_files
# ============================================================================

class TestGetVideoFiles:
    """Test _get_video_files method"""

    def test_get_video_files_empty(self):
        """Test with no downloaded videos"""
        stage = SceneDetectionStage()
        state = PipelineState()

        result = stage._get_video_files(state)

        assert result == []

    def test_get_video_files_with_downloads(self):
        """Test with downloaded videos"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="/path/to/video1.mp4"),
            DownloadedVideo(file="/path/to/video2.mp4")
        ]

        result = stage._get_video_files(state)

        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)


# ============================================================================
# Test _merge_scene_data_to_transcripts (lines 193-324)
# ============================================================================

class TestMergeSceneDataToTranscripts:
    """Test _merge_scene_data_to_transcripts method"""

    @pytest.fixture
    def stage(self):
        return SceneDetectionStage()

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = False
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        return config

    def test_merge_no_transcripts(self, stage, mock_config):
        """Test merge with no transcripts"""
        state = PipelineState()
        state.transcripts = {}

        # Should not raise
        stage._merge_scene_data_to_transcripts(state, {}, mock_config)

    def test_merge_with_transcripts_no_scene_data(self, stage, mock_config):
        """Test merge with transcripts but no matching scene data"""
        state = PipelineState()
        state.transcripts = {
            "/path/to/video.mp4": [
                MockTranscriptSegment(start_time=0.0, end_time=5.0, text="Test")
            ]
        }
        state.embeddings = []
        state.text_metadata = []

        stage._merge_scene_data_to_transcripts(state, {}, mock_config)

        # Should complete without error

    def test_merge_with_matching_scene_data(self, stage, mock_config):
        """Test merge with matching scene data"""
        state = PipelineState()

        segment = MockTranscriptSegment(start_time=0.0, end_time=5.0, text="Test")
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = []
        state.text_metadata = []

        scene_data = {
            "video": MockSceneData(
                scene_count=1,
                scenes=[MockScene(scene_index=0, start_time=0.0, end_time=10.0, is_broll=True, face_score=0.1)]
            )
        }

        stage._merge_scene_data_to_transcripts(state, scene_data, mock_config)

        # Segment should be updated
        assert segment.is_broll is True
        assert segment.face_score == 0.1

    def test_merge_updates_text_metadata(self, stage, mock_config):
        """Test merge updates text_metadata when embeddings exist"""
        import numpy as np

        state = PipelineState()
        state.transcripts = {
            "/path/to/video.mp4": [
                MockTranscriptSegment(start_time=0.0, end_time=5.0, text="Test", is_broll=True, face_score=0.2)
            ]
        }
        state.embeddings = np.array([[0.1, 0.2, 0.3]])  # Non-empty embeddings
        state.text_metadata = [
            {'video_path': '/path/to/video.mp4', 'start_time': 0.0, 'text': 'Test'}
        ]

        scene_data = {
            "video": MockSceneData(
                scene_count=1,
                scenes=[MockScene(scene_index=0, start_time=0.0, end_time=10.0, is_broll=True, face_score=0.2)]
            )
        }

        stage._merge_scene_data_to_transcripts(state, scene_data, mock_config)

        # text_metadata should be updated
        assert state.text_metadata[0].get('is_broll') is True


# ============================================================================
# Test _create_text_metadata_for_silent_videos (lines 326-437)
# ============================================================================

class TestCreateTextMetadataForSilentVideos:
    """Test _create_text_metadata_for_silent_videos method"""

    @pytest.fixture
    def stage(self):
        return SceneDetectionStage()

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = False
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        return config

    @pytest.fixture
    def mock_config_with_vision(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = True
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        return config

    def test_no_silent_videos(self, stage, mock_config):
        """Test with no silent videos"""
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        state.transcripts = {"/path/to/video.mp4": [Mock()]}  # Has transcript
        state.text_metadata = []

        scene_data = {
            "video": MockSceneData(scene_count=1, scenes=[])
        }

        result = stage._create_text_metadata_for_silent_videos(state, scene_data, mock_config)

        assert result == 0

    def test_creates_metadata_for_silent_video(self, stage, mock_config):
        """Test creates text_metadata for silent video"""
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}  # No transcripts
        state.text_metadata = []
        state.embeddings = []

        scene_data = {
            "silent": MockSceneData(
                scene_count=2,
                scenes=[
                    MockScene(scene_index=0, start_time=0.0, end_time=5.0, is_broll=True, face_score=0.1),
                    MockScene(scene_index=1, start_time=5.0, end_time=10.0, is_broll=False, face_score=0.7)
                ]
            )
        }

        result = stage._create_text_metadata_for_silent_videos(state, scene_data, mock_config)

        assert result == 1  # 1 silent video processed
        assert len(state.text_metadata) == 2  # 2 scenes
        assert state.text_metadata[0]['is_broll'] is True
        assert state.text_metadata[1]['is_broll'] is False

    @patch('src.vision.VisionProcessor')
    def test_with_vision_api_enabled(self, mock_vision_class, stage, mock_config_with_vision):
        """Test with vision API enabled"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.return_value = "A beautiful sunset scene"
        mock_vision.get_stats.return_value = {'api_calls': 1, 'estimated_cost': 0.001}
        mock_vision_class.return_value = mock_vision

        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []

        scene_data = {
            "silent": MockSceneData(
                scene_count=1,
                scenes=[MockScene(scene_index=0, start_time=0.0, end_time=5.0, is_broll=True)]
            )
        }

        result = stage._create_text_metadata_for_silent_videos(state, scene_data, mock_config_with_vision)

        assert result == 1
        assert state.text_metadata[0]['text'] == "A beautiful sunset scene"

    @patch('src.vision.VisionProcessor')
    def test_vision_api_not_available(self, mock_vision_class, stage, mock_config_with_vision):
        """Test when vision API is enabled but not available"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = False
        mock_vision_class.return_value = mock_vision

        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []

        scene_data = {
            "silent": MockSceneData(
                scene_count=1,
                scenes=[MockScene(scene_index=0, start_time=0.0, end_time=5.0)]
            )
        }

        result = stage._create_text_metadata_for_silent_videos(state, scene_data, mock_config_with_vision)

        # Should use placeholder text
        assert result == 1
        assert "[Silent video:" in state.text_metadata[0]['text']


# ============================================================================
# Test _compute_embeddings_for_silent_videos (lines 439-485)
# ============================================================================

class TestComputeEmbeddingsForSilentVideos:
    """Test _compute_embeddings_for_silent_videos method"""

    @pytest.fixture
    def stage(self):
        return SceneDetectionStage()

    def test_empty_entries(self, stage):
        """Test with empty entries"""
        state = PipelineState()

        # Should not raise
        stage._compute_embeddings_for_silent_videos(state, [])

    def test_no_existing_embeddings(self, stage):
        """Test when no existing embeddings"""
        state = PipelineState()
        state.embeddings = []

        entries = [{'text': 'Test entry'}]

        # Should not raise, but should warn
        stage._compute_embeddings_for_silent_videos(state, entries)

    @patch('sentence_transformers.SentenceTransformer')
    def test_computes_embeddings_success(self, mock_st_class, stage):
        """Test successful embedding computation"""
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])
        mock_st_class.return_value = mock_model

        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])  # Existing embeddings
        state.embedding_index = None

        entries = [{'text': 'Silent video description'}]

        stage._compute_embeddings_for_silent_videos(state, entries)

        assert state.embeddings.shape[0] == 2  # Original + new

    @patch('sentence_transformers.SentenceTransformer')
    def test_rebuilds_faiss_index(self, mock_st_class, stage):
        """Test FAISS index is rebuilt"""
        import numpy as np
        import sys

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.5, 0.6, 0.7]])
        mock_st_class.return_value = mock_model

        # Create mock faiss module
        mock_faiss = Mock()
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()  # Existing index

        entries = [{'text': 'Silent video description'}]

        # Patch faiss in sys.modules
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            stage._compute_embeddings_for_silent_videos(state, entries)

        mock_faiss.IndexFlatL2.assert_called_once()
        mock_index.add.assert_called_once()


# ============================================================================
# Test restore method (lines 145-168)
# ============================================================================

class TestSceneDetectionRestore:
    """Test restore method"""

    def test_restore_no_data(self):
        """Test restore with no checkpoint data"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, checkpoint)

        assert result is False

    def test_restore_skipped_stage(self):
        """Test restore when stage was skipped"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {'skipped': True}

        result = stage.restore(state, checkpoint)

        assert result is False

    def test_restore_with_data(self):
        """Test restore with valid checkpoint data"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'video_count': 5,
            'scene_count': 20,
            'broll_count': 8
        }

        result = stage.restore(state, checkpoint)

        assert result is True


# ============================================================================
# Test run method edge cases (lines 45-135)
# ============================================================================

class TestSceneDetectionRun:
    """Test run method edge cases"""

    def test_run_skip_by_config(self):
        """Test run skips when config says to"""
        stage = SceneDetectionStage()
        state = PipelineState()

        config = Mock()
        config.pipeline = Mock()
        config.pipeline.skip_scene_detection = True

        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True

    @patch('src.scene_detection.SceneDetector')
    def test_run_no_video_files(self, mock_detector_class):
        """Test run with no video files"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = []

        config = Mock()
        config.pipeline = Mock()
        config.pipeline.skip_scene_detection = False

        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert result.data.get('scene_count') == 0

    @patch('src.scene_detection.SceneDetector')
    def test_run_scene_detection_error(self, mock_detector_class):
        """Test run handles scene detection errors"""
        mock_detector = Mock()
        mock_detector.process_video.side_effect = Exception("Detection error")
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]

        config = Mock()
        config.pipeline = Mock()
        config.pipeline.skip_scene_detection = False

        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        # Should complete with warnings
        assert result.success is True
        assert len(result.warnings) > 0


# ============================================================================
# Test direct scene data updates for silent videos (lines 289-310)
# ============================================================================

class TestDirectSceneDataUpdates:
    """Test direct scene data updates for videos without transcripts"""

    @pytest.fixture
    def stage(self):
        return SceneDetectionStage()

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.vision = Mock()
        config.vision.enabled = False
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        return config

    def test_direct_update_from_scene_data(self, stage, mock_config):
        """Test text_metadata updated directly from scene data when no transcript match"""
        import numpy as np

        state = PipelineState()
        # Need non-empty transcripts to pass early return check, but for a different video
        state.transcripts = {'/path/to/other.mp4': []}
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [
            {
                'video_path': '/path/to/silent.mp4',
                'start_time': 2.5,
                'end_time': 7.5,
                'text': 'Silent video placeholder'
            }
        ]

        scene_data = {
            "silent": MockSceneData(
                scene_count=1,
                scenes=[MockScene(scene_index=0, start_time=0.0, end_time=10.0, is_broll=True, face_score=0.15)]
            )
        }

        stage._merge_scene_data_to_transcripts(state, scene_data, mock_config)

        # Should be updated directly from scene data
        assert state.text_metadata[0].get('is_broll') is True
        assert state.text_metadata[0].get('face_score') == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
