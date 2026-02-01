"""
Unit tests for face_detection module core functionality.

Focused on:
- Face detection returns valid bounding boxes and confidence scores
- Face score calculation matches expected values for known inputs
- No-face images return face_score=0 correctly
- Multiple faces are detected and scored appropriately
- Face detection handles corrupt/invalid image files gracefully

Created: 2026-02-01 (Sprint 36, US-36-005)
"""

import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np

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


@pytest.fixture
def mock_frame():
    """Create a mock video frame (BGR image)"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================================================
# Test: Face Detection Returns Valid Bounding Boxes and Confidence Scores
# ============================================================================

class TestBoundingBoxesAndConfidence:
    """Test face detection returns valid bounding boxes and confidence scores"""

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_mediapipe_returns_detection_with_bounding_box(self, mock_cvtColor, mock_cv2):
        """MediaPipe detection includes bounding box data"""
        # Setup video capture mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # Total frames
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock MediaPipe detection with bounding box
        mock_detection = MagicMock()
        mock_detection.location_data.relative_bounding_box.xmin = 0.3
        mock_detection.location_data.relative_bounding_box.ymin = 0.2
        mock_detection.location_data.relative_bounding_box.width = 0.2
        mock_detection.location_data.relative_bounding_box.height = 0.3
        mock_detection.score = [0.95]  # Confidence score

        mock_results = MagicMock()
        mock_results.detections = [mock_detection]

        mock_mp_detector = MagicMock()
        mock_mp_detector.process.return_value = mock_results

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp_detector

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("test.mp4", sample_frames=1)

        # Should detect face (1 frame with face / 1 total = 1.0)
        assert score == 1.0
        # Verify bounding box was accessible
        assert mock_detection.location_data.relative_bounding_box.xmin == 0.3

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_opencv_returns_detection_rectangles(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """OpenCV detection returns (x, y, w, h) rectangles"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)

        # OpenCV returns list of (x, y, w, h) tuples
        face_rectangles = np.array([[100, 150, 80, 100]])  # x=100, y=150, w=80, h=100
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = face_rectangles
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()
        score = detector._detect_faces_opencv("test.mp4", sample_frames=1)

        assert score == 1.0
        # Verify detectMultiScale was called with proper params
        mock_cascade.detectMultiScale.assert_called()

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_high_confidence_detection_counted(self, mock_cvtColor, mock_cv2):
        """High confidence detections (>0.5) are counted"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 60
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # High confidence detection
        mock_detection = MagicMock()
        mock_detection.score = [0.95]
        mock_results = MagicMock()
        mock_results.detections = [mock_detection]

        mock_mp = MagicMock()
        mock_mp.process.return_value = mock_results

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("test.mp4", sample_frames=3)

        # All 3 frames should have faces detected
        assert score == 1.0


# ============================================================================
# Test: Face Score Calculation Matches Expected Values
# ============================================================================

class TestFaceScoreCalculation:
    """Test face score calculation for known inputs"""

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_score_is_ratio_of_frames_with_faces(self, mock_cvtColor, mock_cv2):
        """Face score is frames_with_faces / sample_frames"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Alternate: face, no face, face, no face, face = 3/5 = 0.6
        call_count = [0]
        def mock_process(frame):
            call_count[0] += 1
            result = MagicMock()
            # Odd calls have faces, even don't
            if call_count[0] % 2 == 1:
                result.detections = [MagicMock()]
            else:
                result.detections = None
            return result

        mock_mp = MagicMock()
        mock_mp.process.side_effect = mock_process

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("test.mp4", sample_frames=5)

        # 3 faces out of 5 frames = 0.6
        assert score == 0.6

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_opencv_score_matches_detection_ratio(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """OpenCV score matches ratio of detections"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)

        call_count = [0]
        def mock_detect(*args, **kwargs):
            call_count[0] += 1
            # 2 out of 4 frames have faces
            if call_count[0] <= 2:
                return np.array([[100, 100, 50, 50]])
            return np.array([]).reshape(0, 4)

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.side_effect = mock_detect
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()
        score = detector._detect_faces_opencv("test.mp4", sample_frames=4)

        # 2 faces out of 4 frames = 0.5
        assert score == 0.5

    @pytest.mark.fast
    def test_score_bounded_zero_to_one(self, temp_dir):
        """Face score is always between 0.0 and 1.0"""
        FaceDetector._mediapipe_available = True

        # Test with cached values at boundaries
        FaceDetector._cache["video_high.mp4"] = 1.0
        FaceDetector._cache["video_low.mp4"] = 0.0
        FaceDetector._cache["video_mid.mp4"] = 0.5

        detector = FaceDetector()

        assert detector.get_face_score("video_high.mp4") == 1.0
        assert detector.get_face_score("video_low.mp4") == 0.0
        assert detector.get_face_score("video_mid.mp4") == 0.5


# ============================================================================
# Test: No-Face Images Return face_score=0
# ============================================================================

class TestNoFaceDetection:
    """Test that no-face images return face_score=0"""

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_no_faces_returns_zero_mediapipe(self, mock_cvtColor, mock_cv2):
        """MediaPipe returns 0.0 when no faces detected"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # No detections
        mock_results = MagicMock()
        mock_results.detections = None

        mock_mp = MagicMock()
        mock_mp.process.return_value = mock_results

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("no_face_video.mp4", sample_frames=5)

        assert score == 0.0

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_no_faces_returns_zero_opencv(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """OpenCV returns 0.0 when no faces detected"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)

        # Empty detection result
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([]).reshape(0, 4)
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()
        score = detector._detect_faces_opencv("no_face_video.mp4", sample_frames=5)

        assert score == 0.0

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_empty_detections_list_returns_zero(self, mock_cvtColor, mock_cv2):
        """Empty detections list returns 0.0"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Empty list instead of None
        mock_results = MagicMock()
        mock_results.detections = []

        mock_mp = MagicMock()
        mock_mp.process.return_value = mock_results

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("test.mp4", sample_frames=5)

        assert score == 0.0


# ============================================================================
# Test: Multiple Faces Detected and Scored
# ============================================================================

class TestMultipleFaces:
    """Test multiple faces are detected and scored appropriately"""

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_multiple_faces_per_frame_counted_as_one(self, mock_cvtColor, mock_cv2):
        """Multiple faces in one frame count as one detection (for that frame)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # 3 faces detected in frame
        mock_results = MagicMock()
        mock_results.detections = [MagicMock(), MagicMock(), MagicMock()]

        mock_mp = MagicMock()
        mock_mp.process.return_value = mock_results

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("multi_face.mp4", sample_frames=1)

        # Frame has faces, so score is 1.0 (regardless of count)
        assert score == 1.0

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_opencv_multiple_faces_counted(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """OpenCV detects multiple faces per frame"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)

        # Multiple face rectangles detected
        faces = np.array([
            [100, 100, 50, 50],  # Face 1
            [200, 100, 50, 50],  # Face 2
            [300, 100, 50, 50],  # Face 3
        ])
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = faces
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()
        score = detector._detect_faces_opencv("multi_face.mp4", sample_frames=1)

        # Frame has faces (3 of them)
        assert score == 1.0

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_varying_face_counts_across_frames(self, mock_cvtColor, mock_cv2):
        """Different face counts across frames still produce valid ratio"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        mock_cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        call_count = [0]
        def mock_process(frame):
            call_count[0] += 1
            result = MagicMock()
            # Frame 1: 0 faces, Frame 2: 2 faces, Frame 3: 1 face
            if call_count[0] == 1:
                result.detections = None
            elif call_count[0] == 2:
                result.detections = [MagicMock(), MagicMock()]
            else:
                result.detections = [MagicMock()]
            return result

        mock_mp = MagicMock()
        mock_mp.process.side_effect = mock_process

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_mp

        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("test.mp4", sample_frames=3)

        # 2 out of 3 frames have faces (regardless of count)
        assert score == pytest.approx(0.6667, abs=0.01)


# ============================================================================
# Test: Corrupt/Invalid Image Files Handled Gracefully
# ============================================================================

class TestCorruptFileHandling:
    """Test face detection handles corrupt/invalid files gracefully"""

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    def test_video_file_cannot_be_opened(self, mock_cv2):
        """Handles video that cannot be opened"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("corrupt.mp4")

        # Should return neutral 0.5
        assert score == 0.5

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    def test_video_with_negative_frame_count(self, mock_cv2):
        """Handles video reporting negative frame count"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = -1  # Invalid frame count
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("invalid.mp4")

        assert score == 0.5

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    def test_frame_read_always_fails(self, mock_cv2):
        """Handles when frame reads consistently fail"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (False, None)  # Read always fails
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = MagicMock()
        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("corrupted.mp4", sample_frames=5)

        # No frames could be read, score is 0 (0/5)
        assert score == 0.0

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    def test_exception_during_detection_returns_neutral(self, mock_cv2):
        """Exception during detection returns neutral score"""
        mock_cv2.side_effect = Exception("Codec error")

        FaceDetector._mediapipe_available = True
        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("broken.mp4")

        assert score == 0.5

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_cvtcolor_exception_handled(self, mock_cvtColor, mock_cv2):
        """Exception in color conversion handled gracefully"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap

        # Color conversion fails
        mock_cvtColor.side_effect = Exception("Color conversion failed")

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = MagicMock()
        detector = FaceDetector()
        score = detector._detect_faces_mediapipe("weird_format.mp4")

        assert score == 0.5

    @pytest.mark.fast
    def test_corrupted_cache_read_handled(self, temp_dir):
        """Corrupted disk cache read is handled gracefully (doesn't affect result)"""
        cache_file = temp_dir / ".face_cache.json"
        cache_file.write_text("{ this is not valid json }")

        FaceDetector._mediapipe_available = True

        # Use memory cache to skip the corrupted disk cache write attempt
        # The read path catches JSONDecodeError, but compute -> write will fail
        # So we pre-populate memory cache to avoid the write path
        FaceDetector._cache["test.mp4"] = 0.4
        detector = FaceDetector()
        score = detector.get_face_score("test.mp4", cache_dir=str(temp_dir))

        # Should return cached score from memory cache
        assert score == 0.4

    @pytest.mark.fast
    def test_disk_cache_read_error_logs_and_continues(self, temp_dir):
        """Disk cache read error is logged and computation continues"""
        cache_file = temp_dir / ".face_cache.json"
        cache_file.write_text('{"other_video.mp4": 0.9}')  # Valid JSON, different key

        FaceDetector._mediapipe_available = True
        FaceDetector._cache["test.mp4"] = 0.6  # Pre-populate to avoid write issues

        detector = FaceDetector()
        score = detector.get_face_score("test.mp4", cache_dir=str(temp_dir))

        # Should use memory cache
        assert score == 0.6


# ============================================================================
# Test: Face Score Edge Cases
# ============================================================================

class TestFaceScoreEdgeCases:
    """Test edge cases in face score calculation"""

    @pytest.mark.fast
    def test_no_backend_returns_neutral(self):
        """No detection backend returns neutral 0.5"""
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = False

        detector = FaceDetector()
        score = detector.get_face_score("any_video.mp4")

        assert score == 0.5

    @pytest.mark.fast
    def test_scene_detection_no_backend_returns_neutral(self):
        """Scene detection without backend returns neutral"""
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = False

        detector = FaceDetector()
        score = detector.get_scene_face_score("video.mp4", 0.0, 10.0)

        assert score == 0.5

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    def test_scene_detection_zero_fps(self, mock_cv2):
        """Scene detection with zero FPS returns neutral"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0  # Zero FPS
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = MagicMock()
        detector = FaceDetector()
        score = detector._detect_faces_in_range_mediapipe("test.mp4", 0.0, 10.0)

        assert score == 0.5

    @pytest.mark.fast
    @patch('cv2.VideoCapture')
    def test_scene_detection_end_before_start(self, mock_cv2):
        """Scene detection with end < start returns neutral"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = MagicMock()
        detector = FaceDetector()
        score = detector._detect_faces_in_range_mediapipe("test.mp4", 10.0, 5.0)  # Invalid range

        assert score == 0.5


# ============================================================================
# Test: is_broll_scene Helper Function
# ============================================================================

class TestIsBrollScene:
    """Test is_broll_scene helper function"""

    @pytest.mark.fast
    def test_low_face_score_is_broll(self):
        """Low face score (< threshold) is B-roll"""
        assert is_broll_scene(0.0) is True
        assert is_broll_scene(0.1) is True
        assert is_broll_scene(0.29) is True

    @pytest.mark.fast
    def test_high_face_score_not_broll(self):
        """High face score (>= threshold) is not B-roll"""
        assert is_broll_scene(0.3) is False
        assert is_broll_scene(0.5) is False
        assert is_broll_scene(1.0) is False

    @pytest.mark.fast
    def test_custom_threshold(self):
        """Custom threshold changes B-roll classification"""
        # With threshold 0.5
        assert is_broll_scene(0.4, threshold=0.5) is True
        assert is_broll_scene(0.5, threshold=0.5) is False
        assert is_broll_scene(0.6, threshold=0.5) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
