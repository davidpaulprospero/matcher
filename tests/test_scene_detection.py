"""
Tests for src/scene_detection.py module.

Covers:
- sanitize_path_for_url() path normalization
- SceneInfo and VideoSceneData dataclasses (serialization round-trips)
- SceneCache (BaseCache subclass) cache hit/miss and serialization
- SceneDetector._detect_scenes_in_video() with mocked cv2/scenedetect
- SceneDetector.process_video() end-to-end with mocks
- Batch processing (process_all_videos) with mocked execution
"""

import json
import os
import time
import tempfile
import shutil
import hashlib
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# sanitize_path_for_url tests
# ---------------------------------------------------------------------------

class TestSanitizePathForUrl:
    """Tests for the sanitize_path_for_url() utility function."""

    def setup_method(self):
        from src.scene_detection import sanitize_path_for_url
        self.sanitize = sanitize_path_for_url

    @pytest.mark.fast
    def test_windows_backslashes_converted(self):
        """Backslashes are converted to forward slashes."""
        result = self.sanitize(r"E:\Videos\project\clip.mp4")
        assert result == "E:/Videos/project/clip.mp4"

    @pytest.mark.fast
    def test_extended_length_prefix_removed_backslash(self):
        r"""Windows \\?\ extended-length prefix is stripped."""
        result = self.sanitize(r"\\?\E:\Videos\clip.mp4")
        assert result == "E:/Videos/clip.mp4"

    @pytest.mark.fast
    def test_extended_length_prefix_removed_forward(self):
        """Windows //?/ extended-length prefix is stripped."""
        result = self.sanitize("//?/E:/Videos/clip.mp4")
        assert result == "E:/Videos/clip.mp4"

    @pytest.mark.fast
    def test_path_with_spaces(self):
        """Spaces in paths are preserved (not URL-encoded)."""
        result = self.sanitize(r"E:\My Videos\my clip.mp4")
        assert result == "E:/My Videos/my clip.mp4"

    @pytest.mark.fast
    def test_path_with_special_characters(self):
        """Special characters (parentheses, brackets, etc.) preserved."""
        result = self.sanitize(r"E:\Videos\clip (1) [final].mp4")
        assert result == "E:/Videos/clip (1) [final].mp4"

    @pytest.mark.fast
    def test_unix_path_unchanged(self):
        """Unix-style paths pass through without modification."""
        result = self.sanitize("/home/user/videos/clip.mp4")
        assert result == "/home/user/videos/clip.mp4"

    @pytest.mark.fast
    def test_pathlib_path_object(self):
        """Path objects are converted to string first."""
        result = self.sanitize(Path("E:/Videos/clip.mp4"))
        assert result == "E:/Videos/clip.mp4"

    @pytest.mark.fast
    def test_empty_string(self):
        """Empty string returns empty string."""
        result = self.sanitize("")
        assert result == ""

    @pytest.mark.fast
    def test_unicode_characters(self):
        """Unicode characters in path are preserved."""
        result = self.sanitize(r"E:\Vidéos\Üntertitel\clip.mp4")
        assert result == "E:/Vidéos/Üntertitel/clip.mp4"


# ---------------------------------------------------------------------------
# SceneInfo dataclass tests
# ---------------------------------------------------------------------------

class TestSceneInfo:
    """Tests for SceneInfo dataclass serialization."""

    @pytest.mark.fast
    def test_to_dict_basic(self):
        from src.scene_detection import SceneInfo
        scene = SceneInfo(
            scene_index=0,
            start_frame=0,
            end_frame=300,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
        )
        d = scene.to_dict()
        assert d['scene_index'] == 0
        assert d['start_frame'] == 0
        assert d['end_frame'] == 300
        assert d['start_time'] == 0.0
        assert d['end_time'] == 10.0
        assert d['duration'] == 10.0
        assert d['face_score'] == 0.5  # default
        assert d['is_broll'] is False  # default

    @pytest.mark.fast
    def test_to_dict_with_broll(self):
        from src.scene_detection import SceneInfo
        scene = SceneInfo(
            scene_index=2,
            start_frame=600,
            end_frame=900,
            start_time=20.0,
            end_time=30.0,
            duration=10.0,
            face_score=0.1,
            is_broll=True,
        )
        d = scene.to_dict()
        assert d['face_score'] == 0.1
        assert d['is_broll'] is True

    @pytest.mark.fast
    def test_to_dict_json_serializable(self):
        """to_dict() output must be JSON serializable."""
        from src.scene_detection import SceneInfo
        scene = SceneInfo(
            scene_index=0, start_frame=0, end_frame=100,
            start_time=0.0, end_time=3.33, duration=3.33,
        )
        # Should not raise
        json_str = json.dumps(scene.to_dict())
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# VideoSceneData dataclass tests
# ---------------------------------------------------------------------------

class TestVideoSceneData:
    """Tests for VideoSceneData serialization round-trip."""

    def _make_scene_data(self):
        from src.scene_detection import SceneInfo, VideoSceneData
        scenes = [
            SceneInfo(scene_index=0, start_frame=0, end_frame=300,
                      start_time=0.0, end_time=10.0, duration=10.0,
                      face_score=0.8, is_broll=False),
            SceneInfo(scene_index=1, start_frame=300, end_frame=600,
                      start_time=10.0, end_time=20.0, duration=10.0,
                      face_score=0.1, is_broll=True),
        ]
        return VideoSceneData(
            video_path="E:/Videos/test.mp4",
            video_name="test",
            framerate=30.0,
            total_frames=600,
            total_duration=20.0,
            scene_count=2,
            scenes=scenes,
            otio_path="E:/output/test.otio",
            has_speech=True,
            speech_ratio=0.75,
            audio_cut_points=[5.0, 15.0],
            file_hash="abc123",
        )

    @pytest.mark.fast
    def test_to_dict_round_trip(self):
        """to_dict -> from_dict produces equivalent object."""
        from src.scene_detection import VideoSceneData
        original = self._make_scene_data()
        d = original.to_dict()
        restored = VideoSceneData.from_dict(d)

        assert restored.video_path == original.video_path
        assert restored.video_name == original.video_name
        assert restored.framerate == original.framerate
        assert restored.total_frames == original.total_frames
        assert restored.total_duration == original.total_duration
        assert restored.scene_count == original.scene_count
        assert restored.otio_path == original.otio_path
        assert restored.has_speech == original.has_speech
        assert restored.speech_ratio == original.speech_ratio
        assert restored.audio_cut_points == original.audio_cut_points
        assert restored.file_hash == original.file_hash
        assert len(restored.scenes) == len(original.scenes)

    @pytest.mark.fast
    def test_from_dict_missing_optional_fields(self):
        """from_dict handles missing optional fields with defaults."""
        from src.scene_detection import VideoSceneData
        minimal = {
            'video_path': 'test.mp4',
            'video_name': 'test',
            'framerate': 30.0,
            'total_frames': 300,
            'total_duration': 10.0,
            'scene_count': 0,
            'scenes': [],
        }
        data = VideoSceneData.from_dict(minimal)
        assert data.otio_path is None
        assert data.has_speech is False
        assert data.speech_ratio == 0.0
        assert data.audio_cut_points == []
        assert data.file_hash == ''

    @pytest.mark.fast
    def test_from_dict_scenes_without_face_fields(self):
        """Old cache format without face_score/is_broll gets defaults."""
        from src.scene_detection import VideoSceneData
        d = {
            'video_path': 'test.mp4',
            'video_name': 'test',
            'framerate': 30.0,
            'total_frames': 300,
            'total_duration': 10.0,
            'scene_count': 1,
            'scenes': [{
                'scene_index': 0,
                'start_frame': 0,
                'end_frame': 300,
                'start_time': 0.0,
                'end_time': 10.0,
                'duration': 10.0,
                # No face_score or is_broll
            }],
        }
        data = VideoSceneData.from_dict(d)
        assert data.scenes[0].face_score == 0.5
        assert data.scenes[0].is_broll is False

    @pytest.mark.fast
    def test_post_init_defaults_audio_cut_points(self):
        """__post_init__ converts None audio_cut_points to empty list."""
        from src.scene_detection import VideoSceneData
        data = VideoSceneData(
            video_path="test.mp4",
            video_name="test",
            framerate=30.0,
            total_frames=300,
            total_duration=10.0,
            scene_count=0,
            scenes=[],
        )
        assert data.audio_cut_points == []

    @pytest.mark.fast
    def test_to_dict_json_serializable(self):
        """Full to_dict() output is JSON serializable."""
        data = self._make_scene_data()
        json_str = json.dumps(data.to_dict())
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# SceneCache tests
# ---------------------------------------------------------------------------

class TestSceneCache:
    """Tests for SceneCache (BaseCache subclass)."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        from src.scene_detection import SceneCache
        self.cache = SceneCache(
            cache_dir=Path(self.tmp_dir) / "scene_cache",
            index_name="scene_index.json",
        )

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _sample_scene_dict(self):
        return {
            'video_path': 'E:/Videos/test.mp4',
            'video_name': 'test',
            'framerate': 30.0,
            'total_frames': 600,
            'total_duration': 20.0,
            'scene_count': 1,
            'scenes': [{
                'scene_index': 0,
                'start_frame': 0,
                'end_frame': 600,
                'start_time': 0.0,
                'end_time': 20.0,
                'duration': 20.0,
                'face_score': 0.5,
                'is_broll': False,
            }],
            'otio_path': None,
            'has_speech': False,
            'speech_ratio': 0.0,
            'audio_cut_points': [],
            'file_hash': 'hash123',
        }

    @pytest.mark.fast
    def test_set_and_get(self):
        """Basic set/get round trip."""
        data = self._sample_scene_dict()
        self.cache.set("test_video", data)
        entry = self.cache.get("test_video")
        assert entry is not None
        assert entry.data == data

    @pytest.mark.fast
    def test_get_miss(self):
        """Getting a nonexistent key returns None."""
        entry = self.cache.get("nonexistent")
        assert entry is None

    @pytest.mark.fast
    def test_get_all_returns_entries(self):
        """get_all returns all cached entries."""
        self.cache.set("video_a", self._sample_scene_dict())
        self.cache.set("video_b", self._sample_scene_dict())
        all_entries = self.cache.get_all()
        assert len(all_entries) == 2
        assert "video_a" in all_entries
        assert "video_b" in all_entries

    @pytest.mark.fast
    def test_serialization_persistence(self):
        """Data persists across cache instances (disk persistence)."""
        from src.scene_detection import SceneCache
        data = self._sample_scene_dict()
        self.cache.set("persisted_video", data)

        # Create new cache instance pointing to same dir
        cache2 = SceneCache(
            cache_dir=Path(self.tmp_dir) / "scene_cache",
            index_name="scene_index.json",
        )
        entry = cache2.get("persisted_video")
        assert entry is not None
        assert entry.data == data

    @pytest.mark.fast
    def test_delete_entry(self):
        """Deleting an entry removes it."""
        self.cache.set("to_delete", self._sample_scene_dict())
        assert self.cache.get("to_delete") is not None
        self.cache.delete("to_delete")
        assert self.cache.get("to_delete") is None

    @pytest.mark.fast
    def test_overwrite_entry(self):
        """Setting same key overwrites previous value."""
        data1 = self._sample_scene_dict()
        data1['scene_count'] = 1
        self.cache.set("video", data1)

        data2 = self._sample_scene_dict()
        data2['scene_count'] = 5
        self.cache.set("video", data2)

        entry = self.cache.get("video")
        assert entry.data['scene_count'] == 5

    @pytest.mark.fast
    def test_stats_tracking(self):
        """Cache tracks hit/miss statistics."""
        self.cache.set("hit_video", self._sample_scene_dict())
        self.cache.get("hit_video")  # hit
        self.cache.get("miss_video")  # miss

        stats = self.cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1


# ---------------------------------------------------------------------------
# SceneDetector tests (with mocked dependencies)
# ---------------------------------------------------------------------------

def _make_mock_config(tmp_path):
    """Create a mock config object for SceneDetector."""
    scene_config = MagicMock()
    scene_config.preset = 'fast'
    scene_config.downscale_factor = None
    scene_config.frame_skip = None
    scene_config.threshold = None
    scene_config.min_scene_len = None
    scene_config.use_gpu = False
    scene_config.force_gpu = False
    scene_config.audio_analysis = False
    scene_config.detect_faces_per_scene = False

    cache_config = MagicMock()
    cache_config.cache_dir = str(tmp_path / ".cache")

    matching_config = MagicMock()
    matching_config.broll_face_threshold = 0.3

    config = MagicMock()
    config.scene_detection = scene_config
    config.cache = cache_config
    config.matching = matching_config
    config.output_dir = str(tmp_path / "output")

    return config


def _create_dummy_video(tmp_path, name="test_video.mp4", size=1024):
    """Create a small dummy file to act as a video."""
    video_path = tmp_path / name
    video_path.write_bytes(b'\x00' * size)
    return video_path


class TestSceneDetectorInit:
    """Tests for SceneDetector initialization."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_init_no_gpu(self, mock_cv2, tmp_path):
        """SceneDetector initializes with GPU disabled."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        assert detector.threshold == 30.0  # fast preset
        assert detector.downscale == 6  # fast preset
        assert detector.frame_skip == 3  # fast preset

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_init_preset_balanced(self, mock_cv2, tmp_path):
        """SceneDetector uses balanced preset settings."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector, PRESETS
        config = _make_mock_config(tmp_path)
        config.scene_detection.preset = 'balanced'
        detector = SceneDetector(config)

        assert detector.threshold == PRESETS['balanced']['threshold']
        assert detector.downscale == PRESETS['balanced']['downscale']

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_init_config_override(self, mock_cv2, tmp_path):
        """Config values override preset defaults."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        config.scene_detection.threshold = 40.0
        config.scene_detection.downscale_factor = 8
        detector = SceneDetector(config)

        assert detector.threshold == 40.0
        assert detector.downscale == 8


class TestDetectScenesInVideo:
    """Tests for SceneDetector._detect_scenes_in_video() with mocked scenedetect."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    @patch('src.scene_detection.open_video')
    @patch('src.scene_detection.SceneManager')
    @patch('src.scene_detection.ContentDetector')
    def test_detects_scenes(self, mock_cd, mock_sm_class, mock_open_video, mock_cv2, tmp_path):
        """Scene detection returns scene list, framerate, and total frames."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        # Mock the video object
        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        mock_video.duration.get_frames.return_value = 900
        mock_video.frame_size = (1920, 1080)
        mock_open_video.return_value = mock_video

        # Mock scene manager
        mock_sm = MagicMock()
        mock_sm_class.return_value = mock_sm

        # Create mock scene boundaries
        scene_start = MagicMock()
        scene_start.get_frames.return_value = 0
        scene_start.get_seconds.return_value = 0.0
        scene_mid = MagicMock()
        scene_mid.get_frames.return_value = 450
        scene_mid.get_seconds.return_value = 15.0
        scene_end = MagicMock()
        scene_end.get_frames.return_value = 900
        scene_end.get_seconds.return_value = 30.0

        mock_sm.get_scene_list.return_value = [
            (scene_start, scene_mid),
            (scene_mid, scene_end),
        ]

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        video_path = _create_dummy_video(tmp_path)
        scene_list, framerate, total_frames = detector._detect_scenes_in_video(video_path)

        assert framerate == 30.0
        assert total_frames == 900
        assert len(scene_list) == 2
        mock_sm.detect_scenes.assert_called_once()

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    @patch('src.scene_detection.open_video')
    @patch('src.scene_detection.SceneManager')
    @patch('src.scene_detection.ContentDetector')
    def test_no_scenes_detected(self, mock_cd, mock_sm_class, mock_open_video, mock_cv2, tmp_path):
        """When no scene cuts found, returns empty scene list."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        mock_video = MagicMock()
        mock_video.frame_rate = 24.0
        mock_video.duration.get_frames.return_value = 240
        mock_video.frame_size = (1280, 720)
        mock_open_video.return_value = mock_video

        mock_sm = MagicMock()
        mock_sm_class.return_value = mock_sm
        mock_sm.get_scene_list.return_value = []

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        video_path = _create_dummy_video(tmp_path)
        scene_list, framerate, total_frames = detector._detect_scenes_in_video(video_path)

        assert framerate == 24.0
        assert total_frames == 240
        assert scene_list == []


class TestSceneListToInfo:
    """Tests for SceneDetector._scene_list_to_info()."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_empty_scene_list_creates_single_scene(self, mock_cv2, tmp_path):
        """Empty scene list means the whole video is one scene."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        scenes = detector._scene_list_to_info([], framerate=30.0, total_frames=900)
        assert len(scenes) == 1
        assert scenes[0].scene_index == 0
        assert scenes[0].start_frame == 0
        assert scenes[0].end_frame == 900
        assert scenes[0].duration == 30.0  # 900 frames / 30 fps

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_multiple_scenes(self, mock_cv2, tmp_path):
        """Scene list with multiple entries creates correct SceneInfo objects."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        # Create mock scene tuples
        s1_start = MagicMock()
        s1_start.get_frames.return_value = 0
        s1_start.get_seconds.return_value = 0.0
        s1_end = MagicMock()
        s1_end.get_frames.return_value = 300
        s1_end.get_seconds.return_value = 10.0

        s2_start = MagicMock()
        s2_start.get_frames.return_value = 300
        s2_start.get_seconds.return_value = 10.0
        s2_end = MagicMock()
        s2_end.get_frames.return_value = 600
        s2_end.get_seconds.return_value = 20.0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        scenes = detector._scene_list_to_info(
            [(s1_start, s1_end), (s2_start, s2_end)],
            framerate=30.0, total_frames=600,
        )
        assert len(scenes) == 2
        assert scenes[0].scene_index == 0
        assert scenes[0].start_time == 0.0
        assert scenes[0].end_time == 10.0
        assert scenes[0].duration == 10.0
        assert scenes[1].scene_index == 1
        assert scenes[1].start_time == 10.0
        assert scenes[1].end_time == 20.0

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_zero_framerate_handled(self, mock_cv2, tmp_path):
        """Zero framerate doesn't cause division error."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        scenes = detector._scene_list_to_info([], framerate=0, total_frames=0)
        assert len(scenes) == 1
        assert scenes[0].duration == 0


# ---------------------------------------------------------------------------
# SceneDetector.process_video() end-to-end tests
# ---------------------------------------------------------------------------

class TestProcessVideo:
    """Tests for SceneDetector.process_video() with full mock chain."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    @patch('src.scene_detection.open_video')
    @patch('src.scene_detection.SceneManager')
    @patch('src.scene_detection.ContentDetector')
    @patch('src.scene_detection.otio')
    def test_process_video_returns_scene_data(
        self, mock_otio, mock_cd, mock_sm_class, mock_open_video, mock_cv2, tmp_path
    ):
        """process_video returns VideoSceneData with detected scenes."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        # Mock video
        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        mock_video.duration.get_frames.return_value = 600
        mock_video.frame_size = (1920, 1080)
        mock_open_video.return_value = mock_video

        # Mock scene manager - no scenes (single scene video)
        mock_sm = MagicMock()
        mock_sm_class.return_value = mock_sm
        mock_sm.get_scene_list.return_value = []

        # Mock OTIO timeline creation
        mock_timeline = MagicMock()
        mock_otio.schema.Timeline.return_value = mock_timeline
        mock_otio.schema.Track.return_value = MagicMock()
        mock_otio.schema.TrackKind.Video = "Video"
        mock_otio.schema.TrackKind.Audio = "Audio"
        mock_otio.opentime.from_timecode.side_effect = ValueError("no timecode")
        mock_otio.opentime.RationalTime.return_value = MagicMock()
        mock_otio.opentime.TimeRange.return_value = MagicMock()
        mock_otio.schema.ExternalReference.return_value = MagicMock()
        mock_otio.schema.Clip.return_value = MagicMock()
        mock_timeline.tracks = MagicMock()

        from src.scene_detection import SceneDetector, VideoSceneData
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        video_path = _create_dummy_video(tmp_path, "my_video.mp4")
        result = detector.process_video(video_path)

        assert result is not None
        assert isinstance(result, VideoSceneData)
        assert result.video_name == "my_video"
        assert result.framerate == 30.0
        assert result.scene_count == 1  # single scene (no cuts)
        assert len(result.scenes) == 1
        assert result.file_hash != ""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    @patch('src.scene_detection.open_video')
    @patch('src.scene_detection.SceneManager')
    @patch('src.scene_detection.ContentDetector')
    @patch('src.scene_detection.otio')
    def test_process_video_caches_result(
        self, mock_otio, mock_cd, mock_sm_class, mock_open_video, mock_cv2, tmp_path
    ):
        """Second call for same unchanged file returns cached result."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        mock_video.duration.get_frames.return_value = 300
        mock_video.frame_size = (1280, 720)
        mock_open_video.return_value = mock_video

        mock_sm = MagicMock()
        mock_sm_class.return_value = mock_sm
        mock_sm.get_scene_list.return_value = []

        mock_timeline = MagicMock()
        mock_otio.schema.Timeline.return_value = mock_timeline
        mock_otio.schema.Track.return_value = MagicMock()
        mock_otio.schema.TrackKind.Video = "Video"
        mock_otio.schema.TrackKind.Audio = "Audio"
        mock_otio.opentime.from_timecode.side_effect = ValueError("no timecode")
        mock_otio.opentime.RationalTime.return_value = MagicMock()
        mock_otio.opentime.TimeRange.return_value = MagicMock()
        mock_otio.schema.ExternalReference.return_value = MagicMock()
        mock_otio.schema.Clip.return_value = MagicMock()
        mock_timeline.tracks = MagicMock()

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        video_path = _create_dummy_video(tmp_path, "cached_video.mp4")

        # First call - processes
        result1 = detector.process_video(video_path)
        assert result1 is not None

        # Reset mock to verify it's NOT called again
        mock_open_video.reset_mock()

        # Second call - should use cache
        result2 = detector.process_video(video_path)
        assert result2 is not None
        assert result2.video_name == result1.video_name
        mock_open_video.assert_not_called()

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    @patch('src.scene_detection.open_video')
    @patch('src.scene_detection.SceneManager')
    @patch('src.scene_detection.ContentDetector')
    def test_process_video_error_returns_none(
        self, mock_cd, mock_sm_class, mock_open_video, mock_cv2, tmp_path
    ):
        """process_video returns None on error."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        mock_open_video.side_effect = RuntimeError("corrupted video")

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        video_path = _create_dummy_video(tmp_path, "broken.mp4")
        result = detector.process_video(video_path)
        assert result is None


# ---------------------------------------------------------------------------
# Batch processing tests
# ---------------------------------------------------------------------------

class TestProcessAllVideos:
    """Tests for SceneDetector.process_all_videos() batch logic."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_no_videos_in_dir(self, mock_cv2, tmp_path):
        """Empty directory returns empty results."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        video_dir = tmp_path / "empty_videos"
        video_dir.mkdir()
        results = detector.process_all_videos(video_dir)
        assert results == {}

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_processes_multiple_videos(self, mock_cv2, tmp_path):
        """process_all_videos calls process_video for each found video."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector, VideoSceneData, SceneInfo
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        # Create dummy video files
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "clip_a.mp4").write_bytes(b'\x00' * 100)
        (video_dir / "clip_b.mkv").write_bytes(b'\x00' * 100)
        (video_dir / "not_a_video.txt").write_bytes(b'text')

        # Mock process_video to return fake data
        def mock_process(video_path):
            return VideoSceneData(
                video_path=str(video_path),
                video_name=video_path.stem,
                framerate=30.0,
                total_frames=300,
                total_duration=10.0,
                scene_count=1,
                scenes=[SceneInfo(0, 0, 300, 0.0, 10.0, 10.0)],
                file_hash="fake",
            )

        detector.process_video = mock_process

        results = detector.process_all_videos(video_dir)
        # Should find .mp4 and .mkv but not .txt
        assert len(results) == 2
        assert "clip_a" in results
        assert "clip_b" in results

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_skips_non_video_extensions(self, mock_cv2, tmp_path):
        """Only files with VIDEO_EXTENSIONS are processed."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)
        detector.process_video = MagicMock(return_value=None)

        video_dir = tmp_path / "mixed"
        video_dir.mkdir()
        (video_dir / "notes.txt").write_bytes(b'text')
        (video_dir / "image.png").write_bytes(b'\x89PNG')
        (video_dir / "data.json").write_bytes(b'{}')

        results = detector.process_all_videos(video_dir)
        assert results == {}
        detector.process_video.assert_not_called()


# ---------------------------------------------------------------------------
# get_scene_clips tests
# ---------------------------------------------------------------------------

class TestGetSceneClips:
    """Tests for SceneDetector.get_scene_clips()."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_clips_from_scene_index(self, mock_cv2, tmp_path):
        """get_scene_clips converts scene index to matchable clip list."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector, VideoSceneData, SceneInfo
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        # Inject scene index directly
        detector.scene_index = {
            "test_video": VideoSceneData(
                video_path="E:/Videos/test_video.mp4",
                video_name="test_video",
                framerate=30.0,
                total_frames=600,
                total_duration=20.0,
                scene_count=2,
                scenes=[
                    SceneInfo(0, 0, 300, 0.0, 10.0, 10.0),
                    SceneInfo(1, 300, 600, 10.0, 20.0, 10.0),
                ],
            )
        }

        clips = detector.get_scene_clips()
        assert len(clips) == 2
        assert clips[0]['clip_id'] == "test_video_scene_000"
        assert clips[0]['is_scene_clip'] is True
        assert clips[0]['start_time'] == 0.0
        assert clips[1]['clip_id'] == "test_video_scene_001"
        assert clips[1]['start_time'] == 10.0

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_empty_scene_index(self, mock_cv2, tmp_path):
        """Empty scene index returns empty clips list."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        clips = detector.get_scene_clips()
        assert clips == []


# ---------------------------------------------------------------------------
# enrich_video_metadata tests
# ---------------------------------------------------------------------------

class TestEnrichVideoMetadata:
    """Tests for SceneDetector.enrich_video_metadata()."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_enriches_known_video(self, mock_cv2, tmp_path):
        """Known videos get scene_boundaries and scene_count added."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector, VideoSceneData, SceneInfo
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        detector.scene_index = {
            "my_video": VideoSceneData(
                video_path="my_video.mp4",
                video_name="my_video",
                framerate=30.0,
                total_frames=300,
                total_duration=10.0,
                scene_count=1,
                scenes=[SceneInfo(0, 0, 300, 0.0, 10.0, 10.0)],
            )
        }

        metadata = {"my_video": {"title": "My Video"}}
        enriched = detector.enrich_video_metadata(metadata)
        assert enriched["my_video"]["scene_count"] == 1
        assert len(enriched["my_video"]["scene_boundaries"]) == 1

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_unknown_video_gets_empty_boundaries(self, mock_cv2, tmp_path):
        """Unknown videos get empty scene_boundaries and scene_count=0."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        metadata = {"unknown_video": {"title": "Unknown"}}
        enriched = detector.enrich_video_metadata(metadata)
        assert enriched["unknown_video"]["scene_count"] == 0
        assert enriched["unknown_video"]["scene_boundaries"] == []


# ---------------------------------------------------------------------------
# get_stats tests
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for SceneDetector.get_stats()."""

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_empty_stats(self, mock_cv2, tmp_path):
        """Empty scene index returns minimal stats."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        stats = detector.get_stats()
        assert stats == {'videos': 0, 'scenes': 0}

    @pytest.mark.fast
    @patch('src.scene_detection.cv2')
    def test_stats_with_data(self, mock_cv2, tmp_path):
        """Stats include correct counts and averages."""
        mock_cv2.ocl = MagicMock()
        mock_cv2.ocl.haveOpenCL.return_value = False
        mock_cv2.cuda = MagicMock()
        mock_cv2.cuda.getCudaEnabledDeviceCount.return_value = 0

        from src.scene_detection import SceneDetector, VideoSceneData, SceneInfo
        config = _make_mock_config(tmp_path)
        detector = SceneDetector(config)

        detector.scene_index = {
            "vid_a": VideoSceneData(
                video_path="a.mp4", video_name="vid_a",
                framerate=30.0, total_frames=600,
                total_duration=20.0, scene_count=2,
                scenes=[
                    SceneInfo(0, 0, 300, 0.0, 10.0, 10.0),
                    SceneInfo(1, 300, 600, 10.0, 20.0, 10.0),
                ],
                has_speech=True, speech_ratio=0.8,
                audio_cut_points=[5.0, 15.0],
            ),
            "vid_b": VideoSceneData(
                video_path="b.mp4", video_name="vid_b",
                framerate=30.0, total_frames=300,
                total_duration=10.0, scene_count=1,
                scenes=[SceneInfo(0, 0, 300, 0.0, 10.0, 10.0)],
                has_speech=False, speech_ratio=0.0,
                audio_cut_points=[],
            ),
        }

        stats = detector.get_stats()
        assert stats['videos_processed'] == 2
        assert stats['total_scenes'] == 3
        assert stats['avg_scenes_per_video'] == 1.5
        assert stats['videos_with_speech'] == 1
        assert stats['total_audio_cut_points'] == 2


# ---------------------------------------------------------------------------
# get_file_hash tests
# ---------------------------------------------------------------------------

class TestGetFileHash:
    """Tests for get_file_hash() utility."""

    @pytest.mark.fast
    def test_hash_deterministic(self, tmp_path):
        """Same file produces same hash."""
        from src.scene_detection import get_file_hash
        f = tmp_path / "test.mp4"
        f.write_bytes(b'\x00' * 256)

        h1 = get_file_hash(f)
        h2 = get_file_hash(f)
        assert h1 == h2
        assert len(h1) == 32  # MD5 hex digest

    @pytest.mark.fast
    def test_different_content_different_hash(self, tmp_path):
        """Different file sizes produce different hashes."""
        from src.scene_detection import get_file_hash
        f1 = tmp_path / "a.mp4"
        f1.write_bytes(b'\x00' * 100)
        f2 = tmp_path / "b.mp4"
        f2.write_bytes(b'\x00' * 200)

        assert get_file_hash(f1) != get_file_hash(f2)


# ---------------------------------------------------------------------------
# PRESETS constant tests
# ---------------------------------------------------------------------------

class TestPresets:
    """Tests for PRESETS configuration constant."""

    @pytest.mark.fast
    def test_all_presets_have_required_keys(self):
        from src.scene_detection import PRESETS
        required = {'downscale', 'frame_skip', 'threshold', 'min_scene_length'}
        for name, preset in PRESETS.items():
            assert set(preset.keys()) == required, f"Preset '{name}' missing keys"

    @pytest.mark.fast
    def test_preset_names(self):
        from src.scene_detection import PRESETS
        assert set(PRESETS.keys()) == {'fast', 'balanced', 'accurate'}

    @pytest.mark.fast
    def test_accurate_has_lowest_threshold(self):
        from src.scene_detection import PRESETS
        assert PRESETS['accurate']['threshold'] < PRESETS['balanced']['threshold']
        assert PRESETS['balanced']['threshold'] < PRESETS['fast']['threshold']


# ---------------------------------------------------------------------------
# VIDEO_EXTENSIONS constant tests
# ---------------------------------------------------------------------------

class TestVideoExtensions:
    """Tests for VIDEO_EXTENSIONS constant."""

    @pytest.mark.fast
    def test_common_formats_included(self):
        from src.scene_detection import VIDEO_EXTENSIONS
        for ext in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
            assert ext in VIDEO_EXTENSIONS
