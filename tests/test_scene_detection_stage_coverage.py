"""
Comprehensive tests for src/stages/scene_detection.py to cover missed lines.

Targets specific uncovered code paths:
- Scene detection initialization and errors
- B-roll detection with face detection
- Vision API integration for silent videos
- Progress reporting and caching
- Edge cases in transcript/metadata merging
- Embedding computation for silent videos
- FAISS index rebuilding
"""

import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from dataclasses import dataclass
from types import SimpleNamespace

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
    """Mock Scene with all required attributes"""
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


def create_mock_config(skip_scene_detection=False, vision_enabled=False):
    """Create mock config with all required attributes"""
    config = Mock()
    config.pipeline = Mock()
    config.pipeline.skip_scene_detection = skip_scene_detection
    config.vision = Mock()
    config.vision.enabled = vision_enabled
    config.cache = Mock()
    config.cache.cache_dir = ".cache"
    config.scene_detection = Mock()
    config.scene_detection.detect_faces_per_scene = True
    config.scene_detection.broll_face_threshold = 0.3
    return config


# ============================================================================
# Test Scene Detection Run - Comprehensive Coverage (lines 45-135)
# ============================================================================

class TestSceneDetectionRunCoverage:
    """Test run method for comprehensive coverage"""

    @pytest.mark.fast
    def test_run_skip_scene_detection_prints_message(self, capsys):
        """Test skip message is printed (line 56-58)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        config = create_mock_config(skip_scene_detection=True)
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        captured = capsys.readouterr()
        assert "Skipping scene detection" in captured.out
        assert result.success is True
        assert result.data['skipped'] is True
        assert result.data['reason'] == 'skip_pipeline_config'

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_no_video_files_warning(self, mock_detector_class, capsys):
        """Test warning when no video files (lines 65-68)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = []
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        captured = capsys.readouterr()
        assert "No video files found" in captured.out
        assert result.success is True
        assert result.data['scene_count'] == 0
        assert "No video files" in result.warnings[0]

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_scene_detection_returns_none(self, mock_detector_class, capsys):
        """Test handling when process_video returns None (lines 98-99)"""
        mock_detector = Mock()
        mock_detector.process_video.return_value = None
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert "failed" in result.warnings[0].lower()
        assert len(state.scene_data) == 0

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_scene_detection_exception_per_video(self, mock_detector_class, capsys):
        """Test exception handling per video (lines 101-103)"""
        mock_detector = Mock()
        mock_detector.process_video.side_effect = RuntimeError("Detection failed")
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is True
        assert "Error processing" in result.warnings[0]

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_prints_progress_per_video(self, mock_detector_class, capsys):
        """Test progress printing per video (lines 82-84, 97)"""
        scene = MockScene(scene_index=0, start_time=0, end_time=5, is_broll=True, face_score=0.1)
        scene_data = MockSceneData(scene_count=2, scenes=[scene, scene])

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/test_video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        captured = capsys.readouterr()
        assert "[1/1]" in captured.out
        assert "test_video" in captured.out
        assert "2 scenes" in captured.out
        assert "2 B-roll" in captured.out

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_summary_statistics(self, mock_detector_class, capsys):
        """Test summary statistics output (lines 112-115)"""
        scene1 = MockScene(scene_index=0, start_time=0, end_time=5, is_broll=True, face_score=0.1)
        scene2 = MockScene(scene_index=1, start_time=5, end_time=10, is_broll=False, face_score=0.8)
        scene_data = MockSceneData(scene_count=2, scenes=[scene1, scene2])

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        captured = capsys.readouterr()
        assert "Scene detection complete" in captured.out
        assert "1 videos processed" in captured.out
        assert "2 total scenes" in captured.out
        assert "1 B-roll scenes" in captured.out
        assert "50.0%" in captured.out

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_zero_scenes_percentage(self, mock_detector_class, capsys):
        """Test 0 scenes case for percentage calculation (line 115)"""
        scene_data = MockSceneData(scene_count=0, scenes=[])

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        captured = capsys.readouterr()
        assert "0.0%" in captured.out  # Should handle division by zero

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_checkpoint_data_structure(self, mock_detector_class):
        """Test checkpoint data is correctly structured (lines 117-129)"""
        scene1 = MockScene(scene_index=0, start_time=0, end_time=5, is_broll=True, face_score=0.1)
        scene2 = MockScene(scene_index=1, start_time=5, end_time=10, is_broll=False, face_score=0.8)
        scene_data = MockSceneData(scene_count=2, scenes=[scene1, scene2], has_speech=True)

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.data['video_count'] == 1
        assert result.data['scene_count'] == 2
        assert result.data['broll_count'] == 1
        assert 'scene_data' in result.data
        assert 'video' in result.data['scene_data']
        assert result.data['scene_data']['video']['has_speech'] is True

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_run_main_exception_handler(self, mock_detector_class):
        """Test main try/except block (lines 133-135)"""
        mock_detector_class.side_effect = Exception("Critical failure")

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        assert result.success is False
        assert "Critical failure" in result.error


# ============================================================================
# Test Restore Method Coverage (lines 145-168)
# ============================================================================

class TestRestoreMethodCoverage:
    """Test restore method for comprehensive coverage"""

    @pytest.mark.fast
    def test_restore_no_stage_data(self):
        """Test restore with no checkpoint data (line 153-154)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_skipped_stage(self):
        """Test restore when stage was skipped (line 153-154)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {'skipped': True}

        result = stage.restore(state, checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_with_valid_data(self):
        """Test restore with valid data (lines 163-164)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            'video_count': 5,
            'scene_count': 20,
            'broll_count': 8
        }

        # Mock config and SceneDetector
        mock_config = Mock()
        with patch('src.scene_detection.SceneDetector') as mock_detector:
            mock_instance = Mock()
            mock_instance.scene_index = {'video1.mp4': Mock(scenes=[])}
            mock_detector.return_value = mock_instance

            result = stage.restore(state, checkpoint, config=mock_config)

        assert result is True

    @pytest.mark.fast
    def test_restore_exception_handling(self):
        """Test restore exception handling (lines 166-168)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.get_stage_data.side_effect = Exception("Checkpoint corrupt")

        result = stage.restore(state, checkpoint)

        assert result is False


# ============================================================================
# Test Merge Scene Data Coverage (lines 193-324)
# ============================================================================

class TestMergeSceneDataCoverage:
    """Test _merge_scene_data_to_transcripts for comprehensive coverage"""

    @pytest.mark.fast
    def test_merge_no_transcripts_early_return(self):
        """Test early return when transcripts empty (lines 212-214)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.transcripts = {}
        config = create_mock_config()

        # Should return early without error
        stage._merge_scene_data_to_transcripts(state, {}, config)

    @pytest.mark.fast
    def test_merge_video_not_in_scene_data(self):
        """Test when video has no scene data (lines 220-221)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        segment = MockTranscriptSegment(start_time=0, end_time=5, text="test")
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = []
        state.text_metadata = []
        config = create_mock_config()

        # No matching scene data for "video"
        stage._merge_scene_data_to_transcripts(state, {"other_video": Mock()}, config)

        # Segment should not have scene metadata added
        assert segment.is_broll is False  # Default value unchanged

    @pytest.mark.fast
    def test_merge_segment_matches_scene(self):
        """Test segment matching scene by midpoint (lines 227-237)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Segment from 2.0 to 4.0 (midpoint 3.0)
        segment = SimpleNamespace(start_time=2.0, end_time=4.0, text="test")
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = []
        state.text_metadata = []
        config = create_mock_config()

        # Scene from 0.0 to 10.0 contains midpoint 3.0
        scene = MockScene(scene_index=1, start_time=0.0, end_time=10.0, is_broll=True, face_score=0.15)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        stage._merge_scene_data_to_transcripts(state, {"video": scene_data}, config)

        assert hasattr(segment, 'is_broll')
        assert segment.is_broll is True
        assert segment.face_score == 0.15
        assert segment.scene_index == 1

    @pytest.mark.fast
    def test_merge_skips_text_metadata_update_no_embeddings(self):
        """Test text_metadata update skipped when no embeddings (lines 254, 323-324)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        segment = MockTranscriptSegment(start_time=0, end_time=5, text="test")
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = []  # Empty embeddings
        state.text_metadata = [{'video_path': '/path/to/video.mp4', 'start_time': 0}]
        config = create_mock_config()

        scene = MockScene(scene_index=0, start_time=0, end_time=10, is_broll=True)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        stage._merge_scene_data_to_transcripts(state, {"video": scene_data}, config)

        # text_metadata should NOT be updated (embeddings empty)
        assert 'is_broll' not in state.text_metadata[0]

    @pytest.mark.fast
    def test_merge_non_dict_text_metadata_entries(self):
        """Test handling non-dict entries in text_metadata (lines 260-261)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        segment = MockTranscriptSegment(start_time=0, end_time=5, text="test", is_broll=True, face_score=0.2)
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        # Mix of dict and non-dict entries
        state.text_metadata = [
            "string entry",
            123,
            None,
            {'video_path': '/path/to/video.mp4', 'start_time': 0}
        ]
        config = create_mock_config()

        scene = MockScene(scene_index=0, start_time=0, end_time=10, is_broll=True, face_score=0.2)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        # Should not crash
        stage._merge_scene_data_to_transcripts(state, {"video": scene_data}, config)

        # Only the dict entry should be updated
        assert state.text_metadata[3].get('is_broll') is True

    @pytest.mark.fast
    def test_merge_text_metadata_missing_video_path(self):
        """Test handling text_metadata entries without video_path (lines 263-265)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        segment = MockTranscriptSegment(start_time=0, end_time=5, text="test", is_broll=True)
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [
            {'text': 'no video_path'},  # Missing video_path
            {'video_path': None, 'start_time': 0},  # None video_path
            {'video_path': '/path/to/video.mp4', 'start_time': 0}
        ]
        config = create_mock_config()

        scene = MockScene(scene_index=0, start_time=0, end_time=10, is_broll=True)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        # Should not crash
        stage._merge_scene_data_to_transcripts(state, {"video": scene_data}, config)

        # Only the valid entry should be updated
        assert 'is_broll' not in state.text_metadata[0]
        assert 'is_broll' not in state.text_metadata[1]
        assert state.text_metadata[2].get('is_broll') is True

    @pytest.mark.fast
    def test_merge_direct_update_for_silent_video(self):
        """Test direct update from scene data for silent videos (lines 289-310)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Transcript for different video
        state.transcripts = {"/path/to/other.mp4": []}
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        # text_metadata for silent video (not in transcripts)
        state.text_metadata = [
            {'video_path': '/path/to/silent.mp4', 'start_time': 2.5, 'end_time': 7.5}
        ]
        config = create_mock_config()

        # Scene data for silent video
        scene = MockScene(scene_index=0, start_time=0.0, end_time=10.0, is_broll=True, face_score=0.15)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        stage._merge_scene_data_to_transcripts(state, {"silent": scene_data}, config)

        # Should be updated directly from scene data
        assert state.text_metadata[0].get('is_broll') is True
        assert state.text_metadata[0].get('face_score') == 0.15
        assert state.text_metadata[0].get('scene_index') == 0

    @pytest.mark.fast
    def test_merge_no_scene_data_for_silent_video(self):
        """Test when silent video has no scene data (line 309-310)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        state.transcripts = {"/path/to/other.mp4": []}
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [
            {'video_path': '/path/to/unknown.mp4', 'start_time': 0}
        ]
        config = create_mock_config()

        # No matching scene data for "unknown"
        stage._merge_scene_data_to_transcripts(state, {"different": Mock()}, config)

        # Should not have is_broll set
        assert 'is_broll' not in state.text_metadata[0]

    @pytest.mark.fast
    def test_merge_broll_data_lost_warning(self):
        """Test B-roll data lost warning (lines 321-322)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Segment with is_broll=True that won't match text_metadata time
        segment = MockTranscriptSegment(start_time=0, end_time=5, text="test", is_broll=True, face_score=0.1)
        state.transcripts = {"/path/to/video.mp4": [segment]}
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        # text_metadata with different time (won't match)
        state.text_metadata = [
            {'video_path': '/path/to/video.mp4', 'start_time': 100.0, 'is_broll': False}
        ]
        config = create_mock_config()

        scene = MockScene(scene_index=0, start_time=0.0, end_time=10.0, is_broll=True, face_score=0.1)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        # This tests the case where broll_found > 0 but final count is 0
        # The warning should be logged
        stage._merge_scene_data_to_transcripts(state, {"video": scene_data}, config)


# ============================================================================
# Test Create Text Metadata for Silent Videos (lines 326-437)
# ============================================================================

class TestCreateTextMetadataForSilentVideosCoverage:
    """Test _create_text_metadata_for_silent_videos for comprehensive coverage"""

    @pytest.mark.fast
    def test_no_silent_videos_returns_zero(self):
        """Test returns 0 when no silent videos (lines 358-359)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]
        state.transcripts = {"/path/to/video.mp4": [Mock()]}  # Has transcript
        state.text_metadata = []
        config = create_mock_config()

        result = stage._create_text_metadata_for_silent_videos(state, {"video": Mock()}, config)

        assert result == 0

    @pytest.mark.fast
    def test_video_without_file_attribute(self):
        """Test handling videos without file attribute (lines 348-349)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Video without 'file' attribute
        video_mock = Mock(spec=[])
        state.downloaded_videos = [video_mock]
        state.transcripts = {}
        state.text_metadata = []
        config = create_mock_config()

        result = stage._create_text_metadata_for_silent_videos(state, {}, config)

        assert result == 0

    @pytest.mark.fast
    def test_creates_entries_for_each_scene(self):
        """Test creates one entry per scene (lines 387-419)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config()

        scene1 = MockScene(scene_index=0, start_time=0, end_time=5, is_broll=True, face_score=0.1)
        scene2 = MockScene(scene_index=1, start_time=5, end_time=10, is_broll=False, face_score=0.8)
        scene3 = MockScene(scene_index=2, start_time=10, end_time=15, is_broll=True, face_score=0.2)
        scene_data = MockSceneData(scene_count=3, scenes=[scene1, scene2, scene3])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        assert len(state.text_metadata) == 3
        assert state.text_metadata[0]['is_broll'] is True
        assert state.text_metadata[1]['is_broll'] is False
        assert state.text_metadata[2]['is_broll'] is True
        assert state.text_metadata[0]['scene_index'] == 0
        assert state.text_metadata[1]['scene_index'] == 1
        assert "[Silent video: silent]" in state.text_metadata[0]['text']

    @pytest.mark.fast
    def test_text_metadata_initially_none(self):
        """Test when text_metadata is None (lines 423-424)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = None  # Initially None
        state.embeddings = []
        config = create_mock_config()

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        assert state.text_metadata is not None
        assert len(state.text_metadata) == 1

    @pytest.mark.fast
    def test_no_scene_data_for_video(self):
        """Test when video has no scene data (lines 383-385)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        config = create_mock_config()

        # Scene data for different video
        result = stage._create_text_metadata_for_silent_videos(state, {"other": Mock()}, config)

        assert result == 0  # Video not in scene_data

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_api_available(self, mock_vision_class):
        """Test vision API available and used (lines 365-373)"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.return_value = "A scenic mountain view"
        mock_vision.get_stats.return_value = {'estimated_cost': 0.002}
        mock_vision_class.return_value = mock_vision

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config(vision_enabled=True)

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        assert state.text_metadata[0]['text'] == "A scenic mountain view"
        mock_vision.describe_scene.assert_called_once()

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_api_not_available(self, mock_vision_class):
        """Test vision API not available fallback (lines 369-371)"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = False
        mock_vision_class.return_value = mock_vision

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config(vision_enabled=True)

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        assert "[Silent video:" in state.text_metadata[0]['text']

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_api_init_exception(self, mock_vision_class):
        """Test vision API initialization exception (lines 374-376)"""
        mock_vision_class.side_effect = Exception("API key invalid")

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config(vision_enabled=True)

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        assert "[Silent video:" in state.text_metadata[0]['text']

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_api_describe_scene_exception(self, mock_vision_class):
        """Test vision API describe_scene exception (lines 404-405)"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.side_effect = Exception("API rate limit")
        mock_vision_class.return_value = mock_vision

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config(vision_enabled=True)

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        # Should fall back to placeholder
        assert "[Silent video:" in state.text_metadata[0]['text']

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_api_returns_none(self, mock_vision_class):
        """Test vision API returns None (line 402)"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.return_value = None  # Returns None
        mock_vision_class.return_value = mock_vision

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config(vision_enabled=True)

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        assert result == 1
        # Should use placeholder when vision returns None
        assert "[Silent video:" in state.text_metadata[0]['text']

    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_vision_stats_logging(self, mock_vision_class):
        """Test vision API stats logging (lines 430-432)"""
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.return_value = "Description"
        mock_vision.get_stats.return_value = {'estimated_cost': 0.005}
        mock_vision_class.return_value = mock_vision

        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
        state.transcripts = {}
        state.text_metadata = []
        state.embeddings = []
        config = create_mock_config(vision_enabled=True)

        scene = MockScene(scene_index=0, start_time=0, end_time=5)
        scene_data = MockSceneData(scene_count=1, scenes=[scene])

        result = stage._create_text_metadata_for_silent_videos(state, {"silent": scene_data}, config)

        mock_vision.get_stats.assert_called_once()


# ============================================================================
# Test Compute Embeddings for Silent Videos (lines 439-485)
# ============================================================================

class TestComputeEmbeddingsForSilentVideosCoverage:
    """Test _compute_embeddings_for_silent_videos for comprehensive coverage"""

    @pytest.mark.fast
    def test_empty_entries_early_return(self):
        """Test early return with empty entries (lines 450-451)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2]])

        original_shape = state.embeddings.shape
        stage._compute_embeddings_for_silent_videos(state, [])

        assert state.embeddings.shape == original_shape

    @pytest.mark.fast
    def test_no_existing_embeddings_warning(self):
        """Test warning when no existing embeddings (lines 458-460)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.embeddings = []  # Empty

        entries = [{'text': 'Test scene'}]
        stage._compute_embeddings_for_silent_videos(state, entries)

        # Should return early without modifying embeddings
        assert len(state.embeddings) == 0

    @pytest.mark.fast
    def test_none_embeddings_warning(self):
        """Test warning when embeddings is None (lines 458-460)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.embeddings = None

        entries = [{'text': 'Test scene'}]
        stage._compute_embeddings_for_silent_videos(state, entries)

        assert state.embeddings is None

    @pytest.mark.fast
    def test_computes_and_appends_embeddings(self):
        """Test embeddings are computed and appended (lines 463-472)"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])

        with patch.dict('sys.modules', {'sentence_transformers': Mock(SentenceTransformer=Mock(return_value=mock_model))}):
            stage = SceneDetectionStage()
            state = PipelineState()
            state.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            state.embedding_index = None

            entries = [{'text': 'New scene description'}]
            stage._compute_embeddings_for_silent_videos(state, entries)

            assert state.embeddings.shape[0] == 3
            mock_model.encode.assert_called_once_with(
                ['New scene description'],
                show_progress_bar=False,
                convert_to_numpy=True
            )

    @pytest.mark.fast
    def test_rebuilds_faiss_index(self):
        """Test FAISS index is rebuilt (lines 475-480)"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])

        mock_index = Mock()
        mock_faiss = Mock(IndexFlatL2=Mock(return_value=mock_index))

        with patch.dict('sys.modules', {
            'sentence_transformers': Mock(SentenceTransformer=Mock(return_value=mock_model)),
            'faiss': mock_faiss
        }):
            stage = SceneDetectionStage()
            state = PipelineState()
            state.embeddings = np.array([[0.1, 0.2, 0.3]])
            state.embedding_index = Mock()  # Has existing index

            entries = [{'text': 'New scene'}]
            stage._compute_embeddings_for_silent_videos(state, entries)

            mock_faiss.IndexFlatL2.assert_called_once_with(3)  # dimension
            mock_index.add.assert_called_once()
            assert state.embedding_index == mock_index

    @pytest.mark.fast
    def test_no_faiss_index_no_rebuild(self):
        """Test no FAISS rebuild when index is None (line 475)"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])

        with patch.dict('sys.modules', {'sentence_transformers': Mock(SentenceTransformer=Mock(return_value=mock_model))}):
            stage = SceneDetectionStage()
            state = PipelineState()
            state.embeddings = np.array([[0.1, 0.2, 0.3]])
            state.embedding_index = None  # No existing index

            entries = [{'text': 'New scene'}]
            stage._compute_embeddings_for_silent_videos(state, entries)

            # Embeddings should be updated but no FAISS operations
            assert state.embeddings.shape[0] == 2
            assert state.embedding_index is None

    @pytest.mark.fast
    def test_exception_handling(self):
        """Test exception handling (lines 484-485)"""
        mock_st_module = Mock()
        mock_st_module.SentenceTransformer.side_effect = Exception("Model load failed")

        with patch.dict('sys.modules', {'sentence_transformers': mock_st_module}):
            stage = SceneDetectionStage()
            state = PipelineState()
            state.embeddings = np.array([[0.1, 0.2, 0.3]])

            entries = [{'text': 'New scene'}]

            # Should not raise, should handle gracefully
            stage._compute_embeddings_for_silent_videos(state, entries)

            # Embeddings should be unchanged
            assert state.embeddings.shape[0] == 1


# ============================================================================
# Test Get Video Files Coverage (lines 182-191)
# ============================================================================

class TestGetVideoFilesCoverage:
    """Test _get_video_files for comprehensive coverage"""

    @pytest.mark.fast
    def test_empty_downloaded_videos(self):
        """Test with empty downloaded_videos list"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = []

        result = stage._get_video_files(state)

        assert result == []

    @pytest.mark.fast
    def test_downloaded_video_with_file(self):
        """Test with DownloadedVideo having file attribute"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [
            DownloadedVideo(file="/path/to/video1.mp4"),
            DownloadedVideo(file="/path/to/video2.mp4")
        ]

        result = stage._get_video_files(state)

        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)

    @pytest.mark.fast
    def test_downloaded_video_without_file(self):
        """Test with mock that has no file attribute (line 188)"""
        stage = SceneDetectionStage()
        state = PipelineState()

        # Mock without file attribute
        mock_video = Mock(spec=[])  # No 'file' in spec
        state.downloaded_videos = [mock_video]

        result = stage._get_video_files(state)

        assert result == []


# ============================================================================
# Test Validate Inputs Coverage (lines 170-178)
# ============================================================================

class TestValidateInputsCoverage:
    """Test validate_inputs for comprehensive coverage"""

    @pytest.mark.fast
    def test_no_videos_no_audio(self):
        """Test validation with no videos and no audio (lines 176-177)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = []
        config = create_mock_config()

        error = stage.validate_inputs(state, config)

        assert error is not None
        assert "No videos" in error

    @pytest.mark.fast
    def test_has_downloaded_videos(self):
        """Test validation with downloaded_videos (line 178)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="test.mp4")]
        state.downloaded_audio = []
        config = create_mock_config()

        error = stage.validate_inputs(state, config)

        assert error is None

    @pytest.mark.fast
    def test_has_downloaded_audio(self):
        """Test validation with downloaded_audio only (line 176)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = [Mock(file="audio.mp3")]
        config = create_mock_config()

        error = stage.validate_inputs(state, config)

        assert error is None


# ============================================================================
# Test Can Skip Coverage (lines 137-143)
# ============================================================================

class TestCanSkipCoverage:
    """Test can_skip for comprehensive coverage"""

    @pytest.mark.fast
    def test_can_skip_true(self):
        """Test can_skip returns True (line 143)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, checkpoint)

        assert result is True
        checkpoint.should_skip_stage.assert_called_with("SCENE_DETECTION")

    @pytest.mark.fast
    def test_can_skip_false(self):
        """Test can_skip returns False (line 143)"""
        stage = SceneDetectionStage()
        state = PipelineState()
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, checkpoint)

        assert result is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestSceneDetectionStageIntegration:
    """Integration tests for SceneDetectionStage"""

    @patch('src.scene_detection.SceneDetector')
    @pytest.mark.fast
    def test_full_pipeline_with_transcripts_and_embeddings(self, mock_detector_class):
        """Test complete pipeline with transcripts, embeddings, and text_metadata"""
        # Setup mock scene detector
        scene1 = MockScene(scene_index=0, start_time=0, end_time=5, is_broll=False, face_score=0.8)
        scene2 = MockScene(scene_index=1, start_time=5, end_time=10, is_broll=True, face_score=0.1)
        scene_data = MockSceneData(scene_count=2, scenes=[scene1, scene2], has_speech=True)

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        # Setup state
        stage = SceneDetectionStage()
        state = PipelineState()
        state.downloaded_videos = [DownloadedVideo(file="/path/to/video.mp4")]

        # Transcript segments
        seg1 = SimpleNamespace(start_time=1.0, end_time=3.0, text="Hello")
        seg2 = SimpleNamespace(start_time=6.0, end_time=8.0, text="World")
        state.transcripts = {"/path/to/video.mp4": [seg1, seg2]}

        # Embeddings and text_metadata
        state.embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        state.text_metadata = [
            {'video_path': '/path/to/video.mp4', 'start_time': 1.0, 'text': 'Hello'},
            {'video_path': '/path/to/video.mp4', 'start_time': 6.0, 'text': 'World'}
        ]

        config = create_mock_config()
        checkpoint = Mock()

        result = stage.run(state, config, checkpoint)

        # Verify result
        assert result.success is True
        assert result.data['scene_count'] == 2
        assert result.data['broll_count'] == 1

        # Verify transcript segments updated
        assert seg1.is_broll is False
        assert seg1.face_score == 0.8
        assert seg2.is_broll is True
        assert seg2.face_score == 0.1

        # Verify text_metadata updated
        assert state.text_metadata[0].get('is_broll') is False
        assert state.text_metadata[1].get('is_broll') is True

    @patch('src.scene_detection.SceneDetector')
    @patch('src.vision.VisionProcessor')
    @pytest.mark.fast
    def test_full_pipeline_with_silent_video_and_vision(
        self, mock_vision_class, mock_detector_class
    ):
        """Test complete pipeline with silent video using vision API

        Note: The run() method only calls _merge_scene_data_to_transcripts when
        state.transcripts is truthy. To trigger silent video processing, we need
        at least one transcript entry (even if for a different video).
        """
        # Setup mock scene detector
        scene = MockScene(scene_index=0, start_time=0, end_time=10, is_broll=True, face_score=0.1)
        scene_data = MockSceneData(scene_count=1, scenes=[scene], has_speech=False)

        mock_detector = Mock()
        mock_detector.process_video.return_value = scene_data
        mock_detector_class.return_value = mock_detector

        # Setup mock vision processor
        mock_vision = Mock()
        mock_vision.is_available.return_value = True
        mock_vision.describe_scene.return_value = "A beautiful sunset over the ocean"
        mock_vision.get_stats.return_value = {'estimated_cost': 0.001}
        mock_vision_class.return_value = mock_vision

        # Setup mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])

        with patch.dict('sys.modules', {'sentence_transformers': Mock(SentenceTransformer=Mock(return_value=mock_model))}):
            # Setup state with a video that has no transcripts but another video has transcripts
            stage = SceneDetectionStage()
            state = PipelineState()
            state.downloaded_videos = [DownloadedVideo(file="/path/to/silent.mp4")]
            # Need at least one transcript entry to trigger _merge_scene_data_to_transcripts
            state.transcripts = {"/path/to/other.mp4": []}
            state.embeddings = np.array([[0.1, 0.2, 0.3]])
            state.text_metadata = []
            state.embedding_index = None

            config = create_mock_config(vision_enabled=True)
            checkpoint = Mock()

            result = stage.run(state, config, checkpoint)

            # Verify result
            assert result.success is True
            assert result.data['scene_count'] == 1
            assert result.data['broll_count'] == 1

            # Verify text_metadata created with vision description
            assert len(state.text_metadata) == 1
            assert state.text_metadata[0]['text'] == "A beautiful sunset over the ocean"
            assert state.text_metadata[0]['is_broll'] is True

            # Verify embeddings updated
            assert state.embeddings.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
