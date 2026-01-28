"""
Test Suite for SceneDetectionStage

Tests the SceneDetectionStage class which handles:
- Scene boundary detection with PySceneDetect
- Face detection per scene (B-roll classification)
- Silent video text_metadata creation
- Vision API integration for scene descriptions
- Scene metadata merging into transcripts and text_metadata
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from dataclasses import dataclass

from src.stages.scene_detection import SceneDetectionStage
from src.state import PipelineState, DownloadedVideo


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with required attributes"""
    config = MagicMock()
    config.pipeline.skip_scene_detection = False
    config.scene_detection.detect_faces_per_scene = True
    config.scene_detection.broll_face_threshold = 0.3
    config.vision.enabled = False
    config.cache.cache_dir = ".cache"
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.checkpoint_path = Path("checkpoint.json")
    return checkpoint


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create temporary project directory"""
    return tmp_path


@pytest.fixture
def mock_scene_info():
    """Create mock SceneInfo dataclass"""
    @dataclass
    class MockSceneInfo:
        scene_index: int
        start_frame: int
        end_frame: int
        start_time: float
        end_time: float
        duration: float
        face_score: float = 0.5
        is_broll: bool = False

    return MockSceneInfo


@pytest.fixture
def mock_video_scene_data(mock_scene_info):
    """Create mock VideoSceneData"""
    @dataclass
    class MockVideoSceneData:
        video_path: str
        video_name: str
        framerate: float
        total_frames: int
        total_duration: float
        scene_count: int
        scenes: list
        has_speech: bool = True
        otio_path: str = None

    return MockVideoSceneData


# ============================================================================
# Test Stage Initialization
# ============================================================================

class TestSceneDetectionStageInit:
    """Test stage initialization"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name"""
        stage = SceneDetectionStage()
        assert stage.name == "SCENE_DETECTION"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = SceneDetectionStage()
        assert "scene" in stage.description.lower() or "Scene" in stage.description

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage
        stage_class = get_stage("SCENE_DETECTION")
        assert stage_class is SceneDetectionStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    @pytest.mark.fast
    def test_validate_no_videos(self, mock_config):
        """Test validation fails when no videos"""
        stage = SceneDetectionStage()
        state = PipelineState()

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "video" in error.lower()

    @pytest.mark.fast
    def test_validate_with_downloaded_videos(self, mock_config):
        """Test validation succeeds with downloaded videos"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video.mp4", url="url", source="download")]

        error = stage.validate_inputs(state, mock_config)

        assert error is None

    @pytest.mark.fast
    def test_validate_with_downloaded_audio(self, mock_config):
        """Test validation succeeds with downloaded audio"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_audio = [Mock(file="audio.mp3")]

        error = stage.validate_inputs(state, mock_config)

        assert error is None


# ============================================================================
# Test Get Video Files
# ============================================================================

class TestGetVideoFiles:
    """Test _get_video_files method"""

    @pytest.mark.fast
    def test_get_files_from_downloaded_videos(self):
        """Test getting video files from downloaded_videos"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download"),
            DownloadedVideo(file="video2.mp4", url="url2", source="download")
        ]

        files = stage._get_video_files(state)

        assert len(files) == 2
        assert all(isinstance(f, Path) for f in files)
        assert str(files[0]) == "video1.mp4"

    @pytest.mark.fast
    def test_get_files_empty_state(self):
        """Test getting files from empty state"""
        stage = SceneDetectionStage()
        state = PipelineState()

        files = stage._get_video_files(state)

        assert len(files) == 0

    @pytest.mark.fast
    def test_get_files_handles_missing_file_attribute(self):
        """Test handling videos without file attribute"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [Mock(spec=[])]  # No 'file' attribute

        files = stage._get_video_files(state)

        assert len(files) == 0


# ============================================================================
# Test Scene Detection Processing
# ============================================================================

class TestSceneDetectionProcessing:
    """Test scene detection processing"""

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_process_single_video_success(self, mock_detector_class, mock_config,
                                         mock_checkpoint, mock_scene_info,
                                         mock_video_scene_data):
        """Test successful scene detection for single video"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]

        # Mock scene data
        scene1 = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.8, is_broll=False)
        scene2 = mock_scene_info(1, 100, 200, 5.0, 10.0, 5.0, face_score=0.1, is_broll=True)

        scene_data = mock_video_scene_data(
            video_path="video1.mp4",
            video_name="video1",
            framerate=30.0,
            total_frames=200,
            total_duration=10.0,
            scene_count=2,
            scenes=[scene1, scene2],
            has_speech=True
        )

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.scene_data) == 1
        assert "video1" in state.scene_data
        assert result.data['scene_count'] == 2
        assert result.data['broll_count'] == 1

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_process_multiple_videos(self, mock_detector_class, mock_config,
                                    mock_checkpoint, mock_scene_info,
                                    mock_video_scene_data):
        """Test processing multiple videos"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="video1.mp4", url="url1", source="download"),
            DownloadedVideo(file="video2.mp4", url="url2", source="download")
        ]

        # Mock scene data for both videos
        scene1 = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.8)
        scene2 = mock_scene_info(0, 0, 150, 0.0, 7.5, 7.5, face_score=0.2, is_broll=True)

        scene_data1 = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene1])
        scene_data2 = mock_video_scene_data("video2.mp4", "video2", 30.0, 150, 7.5, 1, [scene2])

        mock_detector = Mock()
        mock_detector.process_video.side_effect = [scene_data1, scene_data2]
        mock_detector_class.return_value = mock_detector

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.scene_data) == 2
        assert result.data['video_count'] == 2
        assert result.data['scene_count'] == 2
        assert result.data['broll_count'] == 1

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_process_video_detection_failure(self, mock_detector_class, mock_config,
                                            mock_checkpoint):
        """Test handling when scene detection fails for a video"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]

        mock_detector = Mock()
        mock_detector.process_video.return_value = None  # Detection failed
        mock_detector_class.return_value = mock_detector

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True  # Stage succeeds even if individual video fails
        assert len(result.warnings) > 0
        assert "failed" in result.warnings[0].lower()

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_process_video_exception_handling(self, mock_detector_class, mock_config,
                                             mock_checkpoint):
        """Test exception handling during video processing"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]

        mock_detector = Mock()
        mock_detector.process_video.side_effect = Exception("Processing error")
        mock_detector_class.return_value = mock_detector

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True  # Stage continues after error
        assert len(result.warnings) > 0


# ============================================================================
# Test Merge Scene Data to Transcripts
# ============================================================================

class TestMergeSceneDataToTranscripts:
    """Test merging scene metadata into transcripts"""

    @pytest.mark.fast
    def test_merge_no_transcripts(self, mock_config, mock_video_scene_data, mock_scene_info):
        """Test merge when no transcripts exist"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.transcripts = {}

        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1,
                                          [mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0)])

        # Should not raise
        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

    @pytest.mark.fast
    def test_merge_with_transcripts(self, mock_config, mock_video_scene_data, mock_scene_info):
        """Test merging scene metadata into transcript segments"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Mock transcript segment
        segment = Mock()
        segment.start_time = 1.0
        segment.end_time = 3.0
        segment.text = "Hello world"

        state.transcripts = {"video1.mp4": [segment]}

        # Mock scene data that overlaps with segment
        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.8, is_broll=False)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

        # Segment should now have scene metadata
        assert hasattr(segment, 'is_broll')
        assert segment.is_broll is False
        assert segment.face_score == 0.8
        assert segment.scene_index == 0

    @pytest.mark.fast
    def test_merge_broll_segment(self, mock_config, mock_video_scene_data, mock_scene_info):
        """Test merging B-roll scene metadata"""
        stage = SceneDetectionStage()
        state = PipelineState()

        segment = Mock()
        segment.start_time = 1.0
        segment.end_time = 3.0

        state.transcripts = {"video1.mp4": [segment]}

        # B-roll scene (low face_score)
        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.1, is_broll=True)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

        assert segment.is_broll is True
        assert segment.face_score == 0.1

    @pytest.mark.fast
    def test_merge_updates_text_metadata(self, mock_config, mock_video_scene_data, mock_scene_info):
        """Test merging updates text_metadata when embeddings exist"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Mock transcript
        segment = Mock()
        segment.start_time = 1.0
        segment.end_time = 3.0
        segment.is_broll = True
        segment.face_score = 0.2
        segment.scene_index = 0

        state.transcripts = {"video1.mp4": [segment]}

        # Mock text_metadata and embeddings
        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [
            {'video_path': 'video1.mp4', 'text': 'test', 'start_time': 1.0, 'end_time': 3.0}
        ]

        # Mock scene data
        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.2, is_broll=True)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

        # text_metadata should be updated
        assert state.text_metadata[0].get('is_broll') is True
        assert state.text_metadata[0].get('face_score') == 0.2


# ============================================================================
# Test Silent Video Handling
# ============================================================================

class TestSilentVideoHandling:
    """Test text_metadata creation for silent videos"""

    @pytest.mark.fast
    def test_create_text_metadata_no_silent_videos(self, mock_config):
        """Test when there are no silent videos"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]
        state.transcripts = {"video1.mp4": [Mock()]}  # Has transcript = not silent

        count = stage._create_text_metadata_for_silent_videos(state, {}, mock_config)

        assert count == 0

    @pytest.mark.fast
    def test_create_text_metadata_for_silent_video(self, mock_config, mock_video_scene_data,
                                                   mock_scene_info):
        """Test creating text_metadata for silent video"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]
        state.transcripts = {}  # No transcript = silent
        state.text_metadata = []

        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        # Mock scene data
        scene1 = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.8, is_broll=False)
        scene2 = mock_scene_info(1, 100, 200, 5.0, 10.0, 5.0, face_score=0.1, is_broll=True)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 200, 10.0, 2, [scene1, scene2])

        with patch.object(stage, '_compute_embeddings_for_silent_videos'):
            count = stage._create_text_metadata_for_silent_videos(
                state,
                {"video1": scene_data},
                mock_config
            )

        assert count == 1
        assert len(state.text_metadata) == 2  # One entry per scene
        assert state.text_metadata[0]['video_path'] == "video1.mp4"
        assert state.text_metadata[0]['is_broll'] is False
        assert state.text_metadata[1]['is_broll'] is True
        assert "[Silent video: video1]" in state.text_metadata[0]['text']

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_create_text_metadata_with_vision_api(self, mock_vision_class, mock_config,
                                                  mock_video_scene_data, mock_scene_info):
        """Test creating text_metadata with Vision API descriptions"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]
        state.transcripts = {}
        state.text_metadata = []

        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        # Enable vision API
        mock_config.vision.enabled = True

        # Mock vision processor
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.return_value = "A person walking on the beach at sunset"
        mock_vision.get_stats.return_value = {'estimated_cost': 0.001}
        mock_vision_class.return_value = mock_vision

        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.2, is_broll=True)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        with patch.object(stage, '_compute_embeddings_for_silent_videos'):
            count = stage._create_text_metadata_for_silent_videos(
                state,
                {"video1": scene_data},
                mock_config
            )

        assert count == 1
        assert len(state.text_metadata) == 1
        assert "beach at sunset" in state.text_metadata[0]['text']
        assert "[Silent video" not in state.text_metadata[0]['text']  # Should use vision description


# ============================================================================
# Test Embedding Computation for Silent Videos
# ============================================================================

class TestSilentVideoEmbeddings:
    """Test embedding computation for silent videos"""

    @patch('sentence_transformers.SentenceTransformer')
    @patch('faiss.IndexFlatL2')
    @pytest.mark.fast
    def test_compute_embeddings_for_silent_videos(self, mock_index_class, mock_model_class):
        """Test computing embeddings for silent video entries"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Existing embeddings
        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        state.embedding_index = Mock()

        # New entries
        new_entries = [
            {'text': 'Scene 1', 'video_path': 'video.mp4', 'start_time': 0, 'end_time': 5},
            {'text': 'Scene 2', 'video_path': 'video.mp4', 'start_time': 5, 'end_time': 10}
        ]

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
        mock_model_class.return_value = mock_model

        # Mock FAISS index
        mock_index = Mock()
        mock_index_class.return_value = mock_index

        stage._compute_embeddings_for_silent_videos(state, new_entries)

        # Should have 4 embeddings now (2 original + 2 new)
        assert state.embeddings.shape[0] == 4
        assert mock_index.add.called

    @pytest.mark.fast
    def test_compute_embeddings_no_existing_embeddings(self):
        """Test handling when no existing embeddings"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.embeddings = None

        new_entries = [{'text': 'Scene 1'}]

        # Should handle gracefully
        stage._compute_embeddings_for_silent_videos(state, new_entries)

        # Should remain None
        assert state.embeddings is None

    @pytest.mark.fast
    def test_compute_embeddings_empty_entries(self):
        """Test with no new entries"""
        stage = SceneDetectionStage()
        state = PipelineState()

        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        # Should not modify embeddings
        original_shape = state.embeddings.shape
        stage._compute_embeddings_for_silent_videos(state, [])

        assert state.embeddings.shape == original_shape


# ============================================================================
# Test Skip Scene Detection
# ============================================================================

class TestSkipSceneDetection:
    """Test scene detection skip behavior"""

    @pytest.mark.fast
    def test_skip_when_configured(self, mock_config, mock_checkpoint):
        """Test skipping scene detection when config says so"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_config.pipeline.skip_scene_detection = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'skip_pipeline_config'

    @pytest.mark.fast
    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_checkpoint.should_skip_stage.return_value = False

        assert stage.can_skip(state, mock_checkpoint) is False

    @pytest.mark.fast
    def test_can_skip_with_checkpoint(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint exists"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_checkpoint.should_skip_stage.return_value = True

        assert stage.can_skip(state, mock_checkpoint) is True


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestSceneDetectionStageExecution:
    """Test full stage execution"""

    @pytest.mark.fast
    def test_run_no_videos(self, mock_config, mock_checkpoint):
        """Test running with no videos"""
        stage = SceneDetectionStage()
        state = PipelineState()

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['scene_count'] == 0
        assert len(result.warnings) > 0

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_success_full_pipeline(self, mock_detector_class, mock_config,
                                      mock_checkpoint, mock_scene_info,
                                      mock_video_scene_data):
        """Test successful full pipeline execution"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]

        # Mock transcript
        segment = Mock()
        segment.start_time = 1.0
        segment.end_time = 3.0
        state.transcripts = {"video1.mp4": [segment]}

        # Mock embeddings
        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [
            {'video_path': 'video1.mp4', 'start_time': 1.0, 'end_time': 3.0, 'text': 'test'}
        ]

        # Mock scene data
        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.1, is_broll=True)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.scene_data) == 1
        assert state.text_metadata[0].get('is_broll') is True

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_detector_class, mock_config, mock_checkpoint):
        """Test exception handling in main run method"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]

        # Mock detector to raise exception
        mock_detector_class.side_effect = Exception("Detector initialization failed")

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "failed" in result.error.lower()


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestSceneDetectionCheckpoint:
    """Test checkpoint operations"""

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_skipped_stage(self, mock_checkpoint):
        """Test restore when stage was skipped"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = {'skipped': True}

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_restore_success(self, mock_detector_class, mock_checkpoint):
        """Test successful restore from checkpoint"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = {
            'video_count': 2,
            'scene_count': 10,
            'broll_count': 3
        }

        # Mock config - restore requires config to create SceneDetector
        mock_config = Mock()

        # Configure the mock_detector_class to return instance with scene_index
        mock_instance = Mock()
        mock_instance.scene_index = {'video1.mp4': Mock(scenes=[])}
        mock_detector_class.return_value = mock_instance

        result = stage.restore(state, mock_checkpoint, config=mock_config)

        assert result is True

    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_checkpoint):
        """Test restore handles exceptions"""
        stage = SceneDetectionStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.side_effect = Exception("Checkpoint error")

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestSceneDetectionEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.fast
    def test_segment_without_dict_attribute(self, mock_config, mock_video_scene_data,
                                           mock_scene_info):
        """Test handling segment without __dict__ attribute"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Create segment without __dict__ (like a dict)
        segment = {'start_time': 1.0, 'end_time': 3.0, 'text': 'test'}
        state.transcripts = {"video1.mp4": [segment]}

        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.8)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        # Should not raise
        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

    @pytest.mark.fast
    def test_text_metadata_non_dict_entries(self, mock_config, mock_video_scene_data,
                                           mock_scene_info):
        """Test handling non-dict entries in text_metadata"""
        stage = SceneDetectionStage()
        state = PipelineState()

        segment = Mock()
        segment.start_time = 1.0
        segment.end_time = 3.0
        segment.is_broll = True
        segment.face_score = 0.2

        state.transcripts = {"video1.mp4": [segment]}

        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [
            "not a dict",  # Should be skipped
            {'video_path': 'video1.mp4', 'start_time': 1.0, 'end_time': 3.0}
        ]

        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.2, is_broll=True)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        # Should handle gracefully
        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

        # Second entry should be updated
        assert state.text_metadata[1].get('is_broll') is True

    @pytest.mark.fast
    def test_video_name_mismatch(self, mock_config, mock_video_scene_data, mock_scene_info):
        """Test handling when video names don't match"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Use a real object instead of Mock to test hasattr properly
        from types import SimpleNamespace
        segment = SimpleNamespace(start_time=1.0, end_time=3.0, text="test")

        state.transcripts = {"video1.mp4": [segment]}

        # Scene data for different video
        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0)
        scene_data = mock_video_scene_data("video2.mp4", "video2", 30.0, 100, 5.0, 1, [scene])

        # Should not crash
        stage._merge_scene_data_to_transcripts(state, {"video2": scene_data}, mock_config)

        # Segment should not have scene metadata (no matching video name)
        assert not hasattr(segment, 'is_broll')

    @pytest.mark.fast
    def test_segment_outside_scene_bounds(self, mock_config, mock_video_scene_data,
                                         mock_scene_info):
        """Test segment that falls outside all scene boundaries"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Use a real object instead of Mock to test hasattr properly
        from types import SimpleNamespace
        segment = SimpleNamespace(start_time=10.0, end_time=12.0, text="test")

        state.transcripts = {"video1.mp4": [segment]}

        # Scene only covers 0-5 seconds
        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        stage._merge_scene_data_to_transcripts(state, {"video1": scene_data}, mock_config)

        # Segment should not have scene metadata (no matching scene found)
        assert not hasattr(segment, 'is_broll')

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_api_initialization_failure(self, mock_vision_class, mock_config,
                                              mock_video_scene_data, mock_scene_info):
        """Test handling Vision API initialization failure"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="video1.mp4", url="url", source="download")]
        state.transcripts = {}
        state.text_metadata = []

        import numpy as np
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        mock_config.vision.enabled = True
        mock_vision_class.side_effect = Exception("Vision init failed")

        scene = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0)
        scene_data = mock_video_scene_data("video1.mp4", "video1", 30.0, 100, 5.0, 1, [scene])

        with patch.object(stage, '_compute_embeddings_for_silent_videos'):
            count = stage._create_text_metadata_for_silent_videos(
                state,
                {"video1": scene_data},
                mock_config
            )

        # Should fall back to placeholder text
        assert count == 1
        assert "[Silent video:" in state.text_metadata[0]['text']
