"""
Comprehensive tests for face detection module.

Covers:
- FaceDetector initialization and singleton pattern
- Backend availability checking (MediaPipe, OpenCV)
- Video-level face detection
- Scene-level face detection (time ranges)
- Frame sampling and scoring
- Caching (memory and disk)
- Error handling
- B-roll classification
- Face preference adjustments

Created: 2026-01-09 (Phase 3.3)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import shutil
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_detection import (
    FaceDetector,
    apply_face_preference,
    apply_broll_preference,
    is_broll_scene
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_video_capture():
    """Mock cv2.VideoCapture"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0  # FPS
    return mock_cap


@pytest.fixture
def mock_mediapipe():
    """Mock MediaPipe face detection"""
    mock_mp = MagicMock()
    mock_detection = MagicMock()
    mock_mp.solutions.face_detection.FaceDetection.return_value = mock_detection
    return mock_mp, mock_detection


@pytest.fixture(autouse=True)
def reset_detector():
    """Reset FaceDetector singleton between tests"""
    FaceDetector._instance = None
    FaceDetector._cache = {}
    FaceDetector._scene_cache = {}
    FaceDetector._mediapipe_available = None
    FaceDetector._opencv_available = None
    FaceDetector._mp_face_detection = None
    yield


# ============================================================================
# Test FaceDetector Initialization
# ============================================================================

class TestFaceDetectorInit:
    """Test FaceDetector initialization"""

    def test_singleton_pattern(self):
        """Test FaceDetector uses singleton pattern"""
        detector1 = FaceDetector.get_instance()
        detector2 = FaceDetector.get_instance()

        assert detector1 is detector2

    @patch('src.face_detection.logger')
    def test_init_checks_backends(self, mock_logger):
        """Test initialization checks available backends"""
        with patch('builtins.__import__', side_effect=ImportError):
            detector = FaceDetector()

            # Should have tried to import mediapipe
            assert FaceDetector._mediapipe_available is not None

    def test_mediapipe_available(self):
        """Test MediaPipe availability detection"""
        mock_mp = MagicMock()
        mock_mp.solutions.face_detection.FaceDetection.return_value = MagicMock()

        with patch.dict('sys.modules', {'mediapipe': mock_mp}):
            FaceDetector._mediapipe_available = None
            detector = FaceDetector()

            assert FaceDetector._mediapipe_available is True

    def test_mediapipe_unavailable(self):
        """Test MediaPipe unavailable fallback"""
        with patch('builtins.__import__', side_effect=ImportError):
            FaceDetector._mediapipe_available = None
            FaceDetector._opencv_available = None
            detector = FaceDetector()

            assert FaceDetector._mediapipe_available is False


# ============================================================================
# Test Backend Availability
# ============================================================================

class TestBackendAvailability:
    """Test face detection backend availability"""

    def test_is_available_with_mediapipe(self):
        """Test is_available returns True with MediaPipe"""
        FaceDetector._mediapipe_available = True
        FaceDetector._opencv_available = False

        detector = FaceDetector()
        assert detector.is_available() is True

    def test_is_available_with_opencv(self):
        """Test is_available returns True with OpenCV"""
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = True

        detector = FaceDetector()
        assert detector.is_available() is True

    def test_is_available_with_both(self):
        """Test is_available returns True with both backends"""
        FaceDetector._mediapipe_available = True
        FaceDetector._opencv_available = True

        detector = FaceDetector()
        assert detector.is_available() is True

    def test_is_available_with_neither(self):
        """Test is_available returns False with no backends"""
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = False

        detector = FaceDetector()
        assert detector.is_available() is False


# ============================================================================
# Test Video-Level Face Detection
# ============================================================================

class TestVideoLevelDetection:
    """Test video-level face detection"""

    def test_get_face_score_no_backend(self):
        """Test get_face_score returns neutral when no backend"""
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = False

        detector = FaceDetector()
        score = detector.get_face_score("test.mp4")

        assert score == 0.5  # Neutral

    def test_get_face_score_from_memory_cache(self):
        """Test get_face_score uses memory cache"""
        FaceDetector._mediapipe_available = True
        FaceDetector._cache = {"test.mp4": 0.8}

        detector = FaceDetector()
        score = detector.get_face_score("test.mp4")

        assert score == 0.8

    def test_get_face_score_from_disk_cache(self, temp_dir):
        """Test get_face_score uses disk cache"""
        cache_file = temp_dir / ".face_cache.json"
        cache_data = {"test.mp4": 0.7}

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector.get_face_score("test.mp4", cache_dir=str(temp_dir))

        assert score == 0.7
        assert FaceDetector._cache["test.mp4"] == 0.7

    @patch('src.face_detection.FaceDetector._detect_faces_mediapipe')
    def test_get_face_score_computes_and_caches(self, mock_detect, temp_dir):
        """Test get_face_score computes and caches result"""
        mock_detect.return_value = 0.6

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector.get_face_score("test.mp4", cache_dir=str(temp_dir))

        assert score == 0.6
        assert FaceDetector._cache["test.mp4"] == 0.6

        # Check disk cache
        cache_file = temp_dir / ".face_cache.json"
        assert cache_file.exists()
        with open(cache_file, 'r') as f:
            data = json.load(f)
        assert data["test.mp4"] == 0.6


# ============================================================================
# Test Scene-Level Face Detection
# ============================================================================

class TestSceneLevelDetection:
    """Test scene-level face detection (time ranges)"""

    def test_get_scene_face_score_no_backend(self):
        """Test scene detection returns neutral when no backend"""
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = False

        detector = FaceDetector()
        score = detector.get_scene_face_score("test.mp4", 0.0, 5.0)

        assert score == 0.5

    def test_get_scene_face_score_from_memory_cache(self):
        """Test scene detection uses memory cache"""
        FaceDetector._mediapipe_available = True
        FaceDetector._scene_cache = {"test.mp4": {"0.0-5.0": 0.9}}

        detector = FaceDetector()
        score = detector.get_scene_face_score("test.mp4", 0.0, 5.0, scene_index=0)

        assert score == 0.9

    def test_get_scene_face_score_from_disk_cache(self, temp_dir):
        """Test scene detection uses disk cache"""
        cache_file = temp_dir / ".segment_face_cache.json"
        cache_data = {"test.mp4:0.0-5.0": 0.75}

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector.get_scene_face_score("test.mp4", 0.0, 5.0, cache_dir=str(temp_dir))

        assert score == 0.75

    @patch('src.face_detection.FaceDetector._detect_faces_in_range_mediapipe')
    def test_get_scene_face_score_computes_and_caches(self, mock_detect, temp_dir):
        """Test scene detection computes and caches result"""
        mock_detect.return_value = 0.4

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector.get_scene_face_score(
            "test.mp4", 10.0, 15.0,
            scene_index=1,
            sample_frames=3,
            cache_dir=str(temp_dir)
        )

        assert score == 0.4
        assert FaceDetector._scene_cache["test.mp4"]["10.0-15.0"] == 0.4

        # Check disk cache
        cache_file = temp_dir / ".segment_face_cache.json"
        assert cache_file.exists()

    def test_scene_cache_key_format(self):
        """Test scene cache key uses time range format"""
        FaceDetector._mediapipe_available = True
        FaceDetector._scene_cache = {"video.mp4": {"12.3-45.6": 0.5}}

        detector = FaceDetector()
        # Should match with rounded times
        score = detector.get_scene_face_score("video.mp4", 12.34, 45.67)

        assert score == 0.5


# ============================================================================
# Test MediaPipe Detection
# ============================================================================

class TestMediaPipeDetection:
    """Test MediaPipe face detection backend"""

    @patch('cv2.VideoCapture')
    def test_mediapipe_detection_with_faces(self, mock_cv2):
        """Test MediaPipe detects faces"""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # Total frames
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.return_value = mock_cap

        # Mock MediaPipe detection results
        mock_results = MagicMock()
        mock_results.detections = [MagicMock()]  # Has detections

        mock_detector = MagicMock()
        mock_detector.process.return_value = mock_results
        FaceDetector._mp_face_detection = mock_detector

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()

        # Should detect faces in all frames (5/5 = 1.0)
        with patch('cv2.cvtColor'):
            score = detector._detect_faces_mediapipe("test.mp4", sample_frames=5)

        assert score == 1.0

    @patch('cv2.VideoCapture')
    def test_mediapipe_detection_no_faces(self, mock_cv2):
        """Test MediaPipe with no faces"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.return_value = mock_cap

        # No detections
        mock_results = MagicMock()
        mock_results.detections = None

        mock_detector = MagicMock()
        mock_detector.process.return_value = mock_results
        FaceDetector._mp_face_detection = mock_detector

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()

        with patch('cv2.cvtColor'):
            score = detector._detect_faces_mediapipe("test.mp4", sample_frames=5)

        assert score == 0.0

    @patch('cv2.VideoCapture')
    def test_mediapipe_detection_error_handling(self, mock_cv2):
        """Test MediaPipe handles errors gracefully"""
        mock_cv2.side_effect = Exception("Video error")

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()

        score = detector._detect_faces_mediapipe("test.mp4")

        assert score == 0.5  # Neutral on error


# ============================================================================
# Test OpenCV Detection
# ============================================================================

class TestOpenCVDetection:
    """Test OpenCV face detection backend"""

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    def test_opencv_detection_with_faces(self, mock_cascade_class, mock_cv2):
        """Test OpenCV detects faces"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.return_value = mock_cap

        # Mock cascade detection
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(0, 0, 50, 50)]  # Found face
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        with patch('cv2.cvtColor'):
            score = detector._detect_faces_opencv("test.mp4", sample_frames=3)

        assert score == 1.0

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    def test_opencv_detection_no_faces(self, mock_cascade_class, mock_cv2):
        """Test OpenCV with no faces"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.return_value = mock_cap

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = []  # No faces
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        with patch('cv2.cvtColor'):
            score = detector._detect_faces_opencv("test.mp4", sample_frames=3)

        assert score == 0.0


# ============================================================================
# Test Face Preference
# ============================================================================

class TestFacePreference:
    """Test face preference adjustments"""

    def test_face_preference_neutral(self):
        """Test neutral preference doesn't adjust scores"""
        seg1 = Mock(source_file="video1.mp4", start_time=0.0, end_time=5.0, index=0)
        candidates = [(seg1, 0.8)]

        result = apply_face_preference(candidates, "neutral")

        assert result == candidates
        assert result[0][1] == 0.8

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_face_preference_no_backend(self, mock_get_instance):
        """Test face preference when no backend available"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = False
        mock_get_instance.return_value = mock_detector

        seg1 = Mock(source_file="video1.mp4")
        candidates = [(seg1, 0.8)]

        result = apply_face_preference(candidates, "more")

        assert result == candidates

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_face_preference_more_faces(self, mock_get_instance):
        """Test preference for more faces boosts score"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        mock_detector.get_scene_face_score.return_value = 0.8  # Has faces
        mock_get_instance.return_value = mock_detector

        seg1 = Mock(source_file="video1.mp4", start_time=0.0, end_time=5.0, index=0)
        candidates = [(seg1, 0.7)]

        result = apply_face_preference(candidates, "more")

        # Should boost by face_score * 0.1 = 0.8 * 0.1 = 0.08
        assert result[0][1] == pytest.approx(0.78, abs=0.01)

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_face_preference_no_faces(self, mock_get_instance):
        """Test preference for no faces penalizes face videos"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        mock_detector.get_scene_face_score.return_value = 0.8  # Has faces
        mock_get_instance.return_value = mock_detector

        seg1 = Mock(source_file="video1.mp4", start_time=0.0, end_time=5.0, index=0)
        candidates = [(seg1, 0.7)]

        result = apply_face_preference(candidates, "none")

        # Should penalize by -face_score * 0.15 = -0.8 * 0.15 = -0.12
        assert result[0][1] == pytest.approx(0.58, abs=0.01)


# ============================================================================
# Test B-roll Preference
# ============================================================================

class TestBRollPreference:
    """Test B-roll preference adjustments"""

    def test_broll_preference_no_topics(self):
        """Test B-roll preference with no topics"""
        seg1 = Mock(source_file="video1.mp4")
        candidates = [(seg1, 0.8)]

        result = apply_broll_preference(candidates, [], {}, {})

        assert result == candidates

    def test_broll_preference_topic_match_with_broll(self):
        """Test B-roll boost when topic matches and no faces"""
        seg1 = Mock(source_file="video1.mp4", scene_index=0)
        candidates = [(seg1, 0.7)]

        vo_topics = ["travel"]
        video_topics = {"video1.mp4": ["travel destinations"]}
        scene_face_scores = {"video1.mp4": {0: 0.1}}  # B-roll (< 0.3)

        result = apply_broll_preference(
            candidates, vo_topics, video_topics, scene_face_scores,
            broll_boost=0.1
        )

        # Should boost by 0.1
        assert result[0][1] == pytest.approx(0.8, abs=0.01)

    def test_broll_preference_no_boost_without_topic_match(self):
        """Test no boost when topic doesn't match"""
        seg1 = Mock(source_file="video1.mp4", scene_index=0)
        candidates = [(seg1, 0.7)]

        vo_topics = ["cooking"]
        video_topics = {"video1.mp4": ["travel"]}
        scene_face_scores = {"video1.mp4": {0: 0.1}}

        result = apply_broll_preference(
            candidates, vo_topics, video_topics, scene_face_scores
        )

        # No boost
        assert result[0][1] == 0.7

    def test_broll_preference_no_boost_with_faces(self):
        """Test no boost when faces present"""
        seg1 = Mock(source_file="video1.mp4", scene_index=0)
        candidates = [(seg1, 0.7)]

        vo_topics = ["travel"]
        video_topics = {"video1.mp4": ["travel"]}
        scene_face_scores = {"video1.mp4": {0: 0.8}}  # Has faces

        result = apply_broll_preference(
            candidates, vo_topics, video_topics, scene_face_scores
        )

        # No boost (faces present)
        assert result[0][1] == 0.7


# ============================================================================
# Test B-roll Classification
# ============================================================================

class TestBRollClassification:
    """Test B-roll classification"""

    def test_is_broll_scene_below_threshold(self):
        """Test scene is B-roll when below threshold"""
        assert is_broll_scene(0.2, threshold=0.3) is True

    def test_is_broll_scene_above_threshold(self):
        """Test scene is not B-roll when above threshold"""
        assert is_broll_scene(0.5, threshold=0.3) is False

    def test_is_broll_scene_at_threshold(self):
        """Test scene at exact threshold"""
        assert is_broll_scene(0.3, threshold=0.3) is False

    def test_is_broll_scene_custom_threshold(self):
        """Test B-roll with custom threshold"""
        assert is_broll_scene(0.4, threshold=0.5) is True
        assert is_broll_scene(0.6, threshold=0.5) is False


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    @patch('cv2.VideoCapture')
    def test_video_cannot_open(self, mock_cv2):
        """Test handling when video cannot be opened"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()

        score = detector._detect_faces_mediapipe("bad.mp4")

        assert score == 0.5

    @patch('cv2.VideoCapture')
    def test_video_zero_frames(self, mock_cv2):
        """Test handling video with zero frames"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0  # Zero frames
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()

        score = detector._detect_faces_mediapipe("empty.mp4")

        assert score == 0.5

    def test_cache_corruption_handled(self, temp_dir):
        """Test handling corrupted cache files"""
        cache_file = temp_dir / ".face_cache.json"

        # Write corrupted JSON
        with open(cache_file, 'w') as f:
            f.write("{ corrupted json")

        FaceDetector._mediapipe_available = True

        # Should not crash, should compute new score
        with patch('src.face_detection.FaceDetector._detect_faces_mediapipe', return_value=0.5):
            detector = FaceDetector()
            score = detector.get_face_score("test.mp4", cache_dir=str(temp_dir))

            assert score == 0.5

    @patch('cv2.VideoCapture')
    def test_scene_detection_invalid_time_range(self, mock_cv2):
        """Test scene detection with invalid time range"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()

        # End time before start time
        with patch('src.face_detection.FaceDetector._mp_face_detection'):
            score = detector._detect_faces_in_range_mediapipe("test.mp4", 10.0, 5.0)

        assert score == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
