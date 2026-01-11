"""
Extended tests for face detection module to cover missed lines.

Focuses on:
- Lines 139: OpenCV fallback when MediaPipe unavailable (get_face_score)
- Lines 211-212, 220: Disk cache exception handling + OpenCV fallback for scene detection
- Lines 235-236, 240-241: Disk cache read/write for scene detection
- Lines 272, 304, 308-309, 324: Frame read failure handling
- Lines 345-347, 421: OpenCV detection exception and video open failure
- Lines 433-434, 448, 460: OpenCV range detection edge cases
- Lines 529, 547: Fallback and adjustment logic in apply_face_preference

Created: 2026-01-11
"""

import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
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


# ============================================================================
# Test OpenCV Fallback Path (Line 139)
# ============================================================================

class TestOpenCVFallbackPath:
    """Test OpenCV fallback when MediaPipe is unavailable"""

    @patch('src.face_detection.FaceDetector._detect_faces_opencv')
    def test_get_face_score_uses_opencv_when_mediapipe_unavailable(self, mock_opencv_detect, temp_dir):
        """Test get_face_score uses OpenCV fallback (line 139)"""
        # Setup: MediaPipe unavailable, OpenCV available
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = True
        mock_opencv_detect.return_value = 0.75

        detector = FaceDetector()
        score = detector.get_face_score("test.mp4", cache_dir=str(temp_dir))

        # Verify OpenCV was called
        mock_opencv_detect.assert_called_once_with("test.mp4")
        assert score == 0.75

    @patch('src.face_detection.FaceDetector._detect_faces_in_range_opencv')
    def test_get_scene_face_score_uses_opencv_when_mediapipe_unavailable(self, mock_opencv_range, temp_dir):
        """Test get_scene_face_score uses OpenCV fallback (line 220)"""
        # Setup: MediaPipe unavailable, OpenCV available
        FaceDetector._mediapipe_available = False
        FaceDetector._opencv_available = True
        mock_opencv_range.return_value = 0.33

        detector = FaceDetector()
        score = detector.get_scene_face_score(
            "test.mp4", 5.0, 10.0,
            scene_index=1,
            cache_dir=str(temp_dir)
        )

        # Verify OpenCV range detection was called
        mock_opencv_range.assert_called_once_with("test.mp4", 5.0, 10.0, 3)
        assert score == 0.33


# ============================================================================
# Test Disk Cache Exception Handling (Lines 211-212, 240-241)
# ============================================================================

class TestDiskCacheExceptionHandling:
    """Test disk cache exception handling paths"""

    def test_scene_disk_cache_read_exception_handled(self, temp_dir):
        """Test corrupted scene disk cache is handled gracefully (lines 211-212)"""
        cache_file = temp_dir / ".segment_face_cache.json"

        # Write corrupted JSON to disk cache
        with open(cache_file, 'w') as f:
            f.write("{ corrupted: not valid json }")

        FaceDetector._mediapipe_available = True

        with patch('src.face_detection.FaceDetector._detect_faces_in_range_mediapipe', return_value=0.65):
            detector = FaceDetector()
            score = detector.get_scene_face_score(
                "test.mp4", 0.0, 5.0,
                cache_dir=str(temp_dir)
            )

        # Should compute new score despite cache corruption
        assert score == 0.65

    def test_scene_disk_cache_write_exception_handled(self, temp_dir):
        """Test disk cache write exception is handled (lines 240-241)"""
        FaceDetector._mediapipe_available = True

        # Make cache_dir read-only to cause write exception
        with patch('src.face_detection.FaceDetector._detect_faces_in_range_mediapipe', return_value=0.5):
            with patch('builtins.open', side_effect=[IOError("Permission denied")]):
                detector = FaceDetector()
                # Should not raise, should return computed score
                score = detector.get_scene_face_score(
                    "test.mp4", 0.0, 5.0,
                    cache_dir=str(temp_dir)
                )

        assert score == 0.5

    def test_scene_disk_cache_read_for_existing_cache(self, temp_dir):
        """Test reading existing disk cache and loading into memory (lines 235-236)"""
        cache_file = temp_dir / ".segment_face_cache.json"

        # Pre-populate existing cache
        existing_cache = {"other_video.mp4:0.0-5.0": 0.8}
        with open(cache_file, 'w') as f:
            json.dump(existing_cache, f)

        FaceDetector._mediapipe_available = True

        with patch('src.face_detection.FaceDetector._detect_faces_in_range_mediapipe', return_value=0.4):
            detector = FaceDetector()
            score = detector.get_scene_face_score(
                "new_video.mp4", 0.0, 10.0,
                cache_dir=str(temp_dir)
            )

        # Check that existing cache was preserved and new entry added
        with open(cache_file, 'r') as f:
            data = json.load(f)

        assert "other_video.mp4:0.0-5.0" in data
        assert "new_video.mp4:0.0-10.0" in data
        assert data["new_video.mp4:0.0-10.0"] == 0.4


# ============================================================================
# Test Frame Read Failure Handling (Lines 272, 324)
# ============================================================================

class TestFrameReadFailure:
    """Test frame read failure handling in detection methods"""

    @patch('cv2.VideoCapture')
    @patch('cv2.cvtColor')
    def test_mediapipe_detection_frame_read_failure_partial(self, mock_cvtColor, mock_cv2):
        """Test MediaPipe handles partial frame read failures (line 272)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # Total frames

        # First 2 reads succeed, rest fail
        read_results = [
            (True, MagicMock()),  # Frame 1 - success
            (True, MagicMock()),  # Frame 2 - success
            (False, None),        # Frame 3 - failure
            (False, None),        # Frame 4 - failure
            (False, None),        # Frame 5 - failure
        ]
        mock_cap.read.side_effect = read_results
        mock_cv2.return_value = mock_cap

        mock_cvtColor.return_value = MagicMock()

        # Mock MediaPipe detection - 1 face in first frame, 0 in second
        mock_detection = MagicMock()
        mock_results_with_face = MagicMock()
        mock_results_with_face.detections = [MagicMock()]
        mock_results_no_face = MagicMock()
        mock_results_no_face.detections = None
        mock_detection.process.side_effect = [mock_results_with_face, mock_results_no_face]

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = mock_detection
        detector = FaceDetector()

        score = detector._detect_faces_mediapipe("test.mp4", sample_frames=5)

        # 1 frame with faces out of 5 sampled (but only 2 read successfully)
        # score = 1/5 = 0.2
        assert score == 0.2

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_opencv_detection_frame_read_failure_partial(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """Test OpenCV handles partial frame read failures (line 324)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100

        # First read succeeds, second fails
        read_results = [
            (True, MagicMock()),   # Frame 1 - success
            (False, None),         # Frame 2 - failure
            (False, None),         # Frame 3 - failure
        ]
        mock_cap.read.side_effect = read_results
        mock_cv2.return_value = mock_cap

        mock_cvtColor.return_value = MagicMock()

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = [(0, 0, 50, 50)]  # Face found
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_opencv("test.mp4", sample_frames=3)

        # 1 frame with faces out of 3 sampled
        assert score == pytest.approx(1/3, abs=0.01)


# ============================================================================
# Test OpenCV Video Open/Zero Frames (Lines 304, 308-309)
# ============================================================================

class TestOpenCVVideoEdgeCases:
    """Test OpenCV video edge cases"""

    @patch('cv2.VideoCapture')
    def test_opencv_detection_video_cannot_open(self, mock_cv2):
        """Test OpenCV returns neutral when video cannot open (line 304)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_opencv("bad_video.mp4")

        assert score == 0.5  # Neutral

    @patch('cv2.VideoCapture')
    def test_opencv_detection_zero_frames(self, mock_cv2):
        """Test OpenCV returns neutral when video has zero frames (lines 308-309)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0  # Zero frames
        mock_cv2.return_value = mock_cap

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_opencv("empty_video.mp4")

        assert score == 0.5
        assert mock_cap.release.called

    @patch('cv2.VideoCapture')
    def test_opencv_detection_negative_frames(self, mock_cv2):
        """Test OpenCV returns neutral when video has negative frame count"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = -1  # Negative frames
        mock_cv2.return_value = mock_cap

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_opencv("corrupt_video.mp4")

        assert score == 0.5
        assert mock_cap.release.called


# ============================================================================
# Test OpenCV Detection Exception (Lines 345-347)
# ============================================================================

class TestOpenCVDetectionException:
    """Test OpenCV detection exception handling"""

    @patch('cv2.VideoCapture')
    def test_opencv_detection_exception_in_processing(self, mock_cv2):
        """Test OpenCV handles exceptions during processing (lines 345-347)"""
        mock_cv2.side_effect = Exception("Unexpected cv2 error")

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_opencv("test.mp4")

        assert score == 0.5  # Neutral on error

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    def test_opencv_detection_cascade_exception(self, mock_cascade_class, mock_cv2):
        """Test OpenCV handles cascade classifier exception"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.read.return_value = (True, MagicMock())
        mock_cv2.return_value = mock_cap

        # Cascade classifier throws exception
        mock_cascade_class.side_effect = Exception("Cascade load failed")

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_opencv("test.mp4")

        assert score == 0.5


# ============================================================================
# Test OpenCV Range Detection Edge Cases (Lines 421, 433-434, 448, 460)
# ============================================================================

class TestOpenCVRangeDetectionEdgeCases:
    """Test OpenCV range detection edge cases"""

    @patch('cv2.VideoCapture')
    def test_opencv_range_detection_video_cannot_open(self, mock_cv2):
        """Test OpenCV range detection when video cannot open (line 421)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_in_range_opencv("test.mp4", 0.0, 10.0)

        assert score == 0.5

    @patch('cv2.VideoCapture')
    def test_opencv_range_detection_negative_frame_range(self, mock_cv2):
        """Test OpenCV range detection with negative frame range (lines 433-434)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # FPS
        mock_cv2.return_value = mock_cap

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        # End time before start time causes negative frame range
        score = detector._detect_faces_in_range_opencv("test.mp4", 10.0, 5.0)

        assert score == 0.5
        assert mock_cap.release.called

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_opencv_range_detection_frame_read_failure(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """Test OpenCV range detection with frame read failure (line 448)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)  # All reads fail
        mock_cv2.return_value = mock_cap

        mock_cascade = MagicMock()
        mock_cascade_class.return_value = mock_cascade

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_in_range_opencv("test.mp4", 0.0, 10.0)

        # No samples checked, returns neutral
        assert score == 0.5

    @patch('cv2.VideoCapture')
    @patch('cv2.CascadeClassifier')
    @patch('cv2.cvtColor')
    def test_opencv_range_detection_no_samples_checked(self, mock_cvtColor, mock_cascade_class, mock_cv2):
        """Test OpenCV range detection returns 0.5 when no samples checked (line 460)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        # All frame reads fail
        mock_cap.read.return_value = (False, None)
        mock_cv2.return_value = mock_cap

        mock_cascade_class.return_value = MagicMock()

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_in_range_opencv("test.mp4", 0.0, 5.0, sample_frames=3)

        assert score == 0.5  # samples_checked == 0

    @patch('cv2.VideoCapture')
    def test_opencv_range_detection_exception(self, mock_cv2):
        """Test OpenCV range detection exception handling (lines 464-466)"""
        mock_cv2.side_effect = Exception("Video processing error")

        FaceDetector._opencv_available = True
        detector = FaceDetector()

        score = detector._detect_faces_in_range_opencv("test.mp4", 0.0, 10.0)

        assert score == 0.5


# ============================================================================
# Test apply_face_preference Edge Cases (Lines 529, 547)
# ============================================================================

class TestApplyFacePreferenceEdgeCases:
    """Test apply_face_preference edge cases"""

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_apply_face_preference_fallback_to_video_level(self, mock_get_instance):
        """Test fallback to video-level detection when no time range (line 529)"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        mock_detector.get_face_score.return_value = 0.6
        mock_get_instance.return_value = mock_detector

        # Segment with no time range (end_time == start_time)
        seg = Mock()
        seg.source_file = "video1.mp4"
        seg.start_time = 5.0
        seg.end_time = 5.0  # Same as start - no time range
        seg.index = 0
        candidates = [(seg, 0.7)]

        result = apply_face_preference(candidates, "more")

        # Should call get_face_score (video-level) instead of get_scene_face_score
        mock_detector.get_face_score.assert_called_once_with("video1.mp4", None)

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_apply_face_preference_unknown_preference(self, mock_get_instance):
        """Test unknown preference defaults to zero adjustment (line 547)"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        mock_detector.get_scene_face_score.return_value = 0.5
        mock_get_instance.return_value = mock_detector

        seg = Mock()
        seg.source_file = "video1.mp4"
        seg.start_time = 0.0
        seg.end_time = 5.0
        seg.index = 0
        candidates = [(seg, 0.7)]

        # Use an unknown preference
        result = apply_face_preference(candidates, "unknown_preference")

        # Score should remain unchanged (adjustment = 0)
        assert result[0][1] == 0.7

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_apply_face_preference_score_clamping_min(self, mock_get_instance):
        """Test score is clamped to minimum 0.0"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        mock_detector.get_scene_face_score.return_value = 1.0  # High face score
        mock_get_instance.return_value = mock_detector

        seg = Mock()
        seg.source_file = "video1.mp4"
        seg.start_time = 0.0
        seg.end_time = 5.0
        seg.index = 0
        # Low initial score that could go negative with penalty
        candidates = [(seg, 0.1)]

        result = apply_face_preference(candidates, "none")

        # Penalty = -1.0 * 0.15 = -0.15, but score clamped to 0.0
        assert result[0][1] == 0.0

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_apply_face_preference_score_clamping_max(self, mock_get_instance):
        """Test score is clamped to maximum 1.0"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        mock_detector.get_scene_face_score.return_value = 1.0  # High face score
        mock_get_instance.return_value = mock_detector

        seg = Mock()
        seg.source_file = "video1.mp4"
        seg.start_time = 0.0
        seg.end_time = 5.0
        seg.index = 0
        # High initial score
        candidates = [(seg, 0.98)]

        result = apply_face_preference(candidates, "more")

        # Boost = 1.0 * 0.1 = 0.1, total = 1.08 clamped to 1.0
        assert result[0][1] == 1.0

    @patch('src.face_detection.FaceDetector.get_instance')
    def test_apply_face_preference_resorting(self, mock_get_instance):
        """Test candidates are re-sorted by adjusted score"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        # First segment has faces (high face score), second doesn't
        mock_detector.get_scene_face_score.side_effect = [0.9, 0.1]
        mock_get_instance.return_value = mock_detector

        seg1 = Mock()
        seg1.source_file = "video1.mp4"
        seg1.start_time = 0.0
        seg1.end_time = 5.0
        seg1.index = 0

        seg2 = Mock()
        seg2.source_file = "video2.mp4"
        seg2.start_time = 0.0
        seg2.end_time = 5.0
        seg2.index = 0

        # seg1 initially higher score
        candidates = [(seg1, 0.8), (seg2, 0.7)]

        result = apply_face_preference(candidates, "none")

        # After penalty for faces, seg2 should be higher
        # seg1: 0.8 - (0.9 * 0.15) = 0.665
        # seg2: 0.7 - (0.1 * 0.15) = 0.685
        assert result[0][0] is seg2
        assert result[1][0] is seg1


# ============================================================================
# Test MediaPipe Initialization Fallback Paths (Lines 67-68, 70-75)
# ============================================================================

class TestMediaPipeInitializationFallback:
    """Test MediaPipe initialization fallback paths"""

    def test_mediapipe_no_solutions_face_detection(self):
        """Test MediaPipe without face_detection in solutions (lines 67-68)"""
        mock_mp = MagicMock()
        # Has solutions but no face_detection
        mock_mp.solutions = MagicMock()
        del mock_mp.solutions.face_detection

        with patch.dict('sys.modules', {'mediapipe': mock_mp}):
            FaceDetector._mediapipe_available = None
            detector = FaceDetector()

            assert FaceDetector._mediapipe_available is False

    def test_mediapipe_general_exception(self):
        """Test MediaPipe general exception during init (lines 73-75)"""
        mock_mp = MagicMock()
        mock_mp.solutions.face_detection.FaceDetection.side_effect = RuntimeError("GPU not available")

        with patch.dict('sys.modules', {'mediapipe': mock_mp}):
            FaceDetector._mediapipe_available = None
            detector = FaceDetector()

            assert FaceDetector._mediapipe_available is False


# ============================================================================
# Test OpenCV Initialization Fallback Paths (Lines 86-93)
# ============================================================================

class TestOpenCVInitializationFallback:
    """Test OpenCV initialization fallback paths"""

    def test_opencv_cascade_file_not_exists(self):
        """Test OpenCV cascade file doesn't exist (lines 86-87)"""
        mock_cv2 = MagicMock()
        mock_cv2.data.haarcascades = "/nonexistent/path/"

        with patch.dict('sys.modules', {'cv2': mock_cv2}):
            with patch.object(Path, 'exists', return_value=False):
                FaceDetector._opencv_available = None
                FaceDetector._mediapipe_available = False
                detector = FaceDetector()

                assert FaceDetector._opencv_available is False

    def test_opencv_general_exception(self):
        """Test OpenCV general exception during init (lines 91-93)"""
        mock_cv2 = MagicMock()
        type(mock_cv2.data).haarcascades = PropertyMock(side_effect=RuntimeError("cv2 data error"))

        with patch.dict('sys.modules', {'cv2': mock_cv2}):
            FaceDetector._opencv_available = None
            FaceDetector._mediapipe_available = False
            detector = FaceDetector()

            assert FaceDetector._opencv_available is False


# ============================================================================
# Test Scene Detection Memory Cache Initialization
# ============================================================================

class TestSceneCacheInitialization:
    """Test scene cache initialization paths"""

    @patch('src.face_detection.FaceDetector._detect_faces_in_range_mediapipe')
    def test_scene_cache_creates_new_video_entry(self, mock_detect):
        """Test scene cache creates new video entry when not in cache"""
        mock_detect.return_value = 0.4

        FaceDetector._mediapipe_available = True
        FaceDetector._scene_cache = {}  # Empty cache

        detector = FaceDetector()
        score = detector.get_scene_face_score("new_video.mp4", 0.0, 10.0)

        # Should create entry for new video
        assert "new_video.mp4" in FaceDetector._scene_cache
        assert "0.0-10.0" in FaceDetector._scene_cache["new_video.mp4"]
        assert FaceDetector._scene_cache["new_video.mp4"]["0.0-10.0"] == 0.4

    def test_scene_cache_disk_read_updates_memory_cache(self, temp_dir):
        """Test disk cache read updates memory cache correctly (lines 207-209)"""
        cache_file = temp_dir / ".segment_face_cache.json"
        cache_data = {"video.mp4:5.0-15.0": 0.25}

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

        FaceDetector._mediapipe_available = True
        FaceDetector._scene_cache = {}  # Empty memory cache

        detector = FaceDetector()
        score = detector.get_scene_face_score(
            "video.mp4", 5.0, 15.0,
            cache_dir=str(temp_dir)
        )

        # Memory cache should be updated
        assert "video.mp4" in FaceDetector._scene_cache
        assert FaceDetector._scene_cache["video.mp4"]["5.0-15.0"] == 0.25
        assert score == 0.25


# ============================================================================
# Test MediaPipe Range Detection All Reads Fail
# ============================================================================

class TestMediaPipeRangeDetectionAllReadsFail:
    """Test MediaPipe range detection when all reads fail"""

    @patch('cv2.VideoCapture')
    def test_mediapipe_range_all_reads_fail(self, mock_cv2):
        """Test MediaPipe range detection returns 0.5 when all reads fail (line 398-399)"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)  # All reads fail
        mock_cv2.return_value = mock_cap

        FaceDetector._mediapipe_available = True
        FaceDetector._mp_face_detection = MagicMock()
        detector = FaceDetector()

        score = detector._detect_faces_in_range_mediapipe("test.mp4", 0.0, 10.0, sample_frames=3)

        assert score == 0.5  # samples_checked == 0


# ============================================================================
# Test Face Statistics Tracking
# ============================================================================

class TestFaceStatisticsTracking:
    """Test face statistics tracking in apply_face_preference"""

    @patch('src.face_detection.FaceDetector.get_instance')
    @patch('src.face_detection.logger')
    def test_face_statistics_tracking(self, mock_logger, mock_get_instance):
        """Test segments_with_faces and segments_without_faces tracking"""
        mock_detector = Mock()
        mock_detector.is_available.return_value = True
        # Three segments: with faces, without faces, with faces
        mock_detector.get_scene_face_score.side_effect = [0.8, 0.1, 0.5]
        mock_get_instance.return_value = mock_detector

        segments = []
        for i in range(3):
            seg = Mock()
            seg.source_file = f"video{i}.mp4"
            seg.start_time = 0.0
            seg.end_time = 5.0
            seg.index = 0
            segments.append((seg, 0.5))

        result = apply_face_preference(segments, "more")

        # Should have logged debug info about face statistics
        assert len(result) == 3


# ============================================================================
# Test B-roll Preference Edge Cases
# ============================================================================

class TestBrollPreferenceEdgeCases:
    """Test B-roll preference edge cases"""

    def test_broll_preference_missing_scene_index(self):
        """Test B-roll preference when segment has no scene_index attribute"""
        seg = Mock(spec=['source_file'])  # No scene_index
        seg.source_file = "video1.mp4"
        candidates = [(seg, 0.7)]

        vo_topics = ["travel"]
        video_topics = {"video1.mp4": ["travel destinations"]}
        scene_face_scores = {"video1.mp4": {0: 0.1}}

        result = apply_broll_preference(
            candidates, vo_topics, video_topics, scene_face_scores
        )

        # Should handle missing scene_index gracefully (defaults to 0)
        assert len(result) == 1

    def test_broll_preference_video_not_in_face_scores(self):
        """Test B-roll preference when video not in face scores dict"""
        seg = Mock()
        seg.source_file = "unknown_video.mp4"
        seg.scene_index = 0
        candidates = [(seg, 0.7)]

        vo_topics = ["travel"]
        video_topics = {"unknown_video.mp4": ["travel"]}
        scene_face_scores = {}  # Video not in face scores

        result = apply_broll_preference(
            candidates, vo_topics, video_topics, scene_face_scores
        )

        # Should use default face_score of 0.5, not B-roll
        assert result[0][1] == 0.7  # No boost

    def test_broll_preference_scene_not_in_face_scores(self):
        """Test B-roll preference when scene not in face scores for video"""
        seg = Mock()
        seg.source_file = "video1.mp4"
        seg.scene_index = 5  # Scene 5 not in dict
        candidates = [(seg, 0.7)]

        vo_topics = ["travel"]
        video_topics = {"video1.mp4": ["travel"]}
        scene_face_scores = {"video1.mp4": {0: 0.1}}  # Only scene 0

        result = apply_broll_preference(
            candidates, vo_topics, video_topics, scene_face_scores
        )

        # Should use default of 0.5, not B-roll
        assert result[0][1] == 0.7  # No boost

    def test_broll_preference_custom_threshold(self):
        """Test B-roll preference with custom threshold"""
        seg = Mock()
        seg.source_file = "video1.mp4"
        seg.scene_index = 0
        candidates = [(seg, 0.7)]

        vo_topics = ["travel"]
        video_topics = {"video1.mp4": ["travel"]}
        # Face score 0.4 - above default 0.3 but below 0.5
        scene_face_scores = {"video1.mp4": {0: 0.4}}

        # With default threshold 0.3, not B-roll
        result1 = apply_broll_preference(
            candidates.copy(), vo_topics, video_topics, scene_face_scores,
            broll_threshold=0.3
        )
        assert result1[0][1] == 0.7  # No boost

        # With higher threshold 0.5, is B-roll
        result2 = apply_broll_preference(
            candidates.copy(), vo_topics, video_topics, scene_face_scores,
            broll_threshold=0.5
        )
        assert result2[0][1] == pytest.approx(0.8, abs=0.01)  # Boosted


# ============================================================================
# Test is_broll_scene Edge Cases
# ============================================================================

class TestIsBrollSceneEdgeCases:
    """Test is_broll_scene edge cases"""

    def test_is_broll_scene_zero_face_score(self):
        """Test B-roll classification with zero face score"""
        assert is_broll_scene(0.0, threshold=0.3) is True

    def test_is_broll_scene_one_face_score(self):
        """Test B-roll classification with maximum face score"""
        assert is_broll_scene(1.0, threshold=0.3) is False

    def test_is_broll_scene_zero_threshold(self):
        """Test B-roll classification with zero threshold"""
        assert is_broll_scene(0.0, threshold=0.0) is False
        assert is_broll_scene(0.01, threshold=0.0) is False

    def test_is_broll_scene_one_threshold(self):
        """Test B-roll classification with threshold of 1.0"""
        assert is_broll_scene(0.9, threshold=1.0) is True
        assert is_broll_scene(1.0, threshold=1.0) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
