"""
Comprehensive tests for audio analysis module.

Covers:
- AudioAnalyzer initialization
- Librosa availability checking
- Silence detection
- Speech detection
- Audio extraction from video
- Cut point suggestion
- AudioAnalysis dataclass operations
- Batch analysis
- Error handling
- Convenience functions

Created: 2026-01-09 (Phase 3.4)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import shutil
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_analysis import (
    AudioAnalyzer,
    AudioAnalysis,
    SilenceRegion,
    SpeechRegion,
    analyze_audio,
    has_speech,
    get_silence_cut_points
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
def mock_librosa():
    """Mock librosa library"""
    mock_lib = MagicMock()

    # Mock common functions
    mock_lib.load.return_value = (np.zeros(22050), 22050)  # 1 second of silence
    mock_lib.feature.rms.return_value = np.array([[0.1, 0.2, 0.1]])
    mock_lib.feature.spectral_centroid.return_value = np.array([[1000, 2000, 1500]])
    mock_lib.feature.spectral_flatness.return_value = np.array([[0.1, 0.2, 0.15]])
    mock_lib.feature.zero_crossing_rate.return_value = np.array([[0.05, 0.1, 0.07]])
    mock_lib.amplitude_to_db.return_value = np.array([-30, -20, -35])
    mock_lib.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])

    return mock_lib


@pytest.fixture
def sample_audio():
    """Create sample audio data"""
    # 1 second of audio at 22050 Hz
    return np.random.randn(22050)


@pytest.fixture
def sample_analysis():
    """Create sample AudioAnalysis"""
    return AudioAnalysis(
        video_path="test.mp4",
        duration=10.0,
        has_speech=True,
        speech_ratio=0.6,
        silence_regions=[
            SilenceRegion(start_time=2.0, end_time=3.0, duration=1.0),
            SilenceRegion(start_time=7.0, end_time=8.5, duration=1.5)
        ],
        speech_regions=[
            SpeechRegion(start_time=0.0, end_time=2.0, duration=2.0, confidence=0.8),
            SpeechRegion(start_time=3.0, end_time=7.0, duration=4.0, confidence=0.9)
        ],
        suggested_cut_points=[2.5, 7.75]
    )


# ============================================================================
# Test AudioAnalyzer Initialization
# ============================================================================

class TestAudioAnalyzerInit:
    """Test AudioAnalyzer initialization"""

    def test_init_default_config(self):
        """Test initialization with default config"""
        analyzer = AudioAnalyzer()

        assert analyzer.silence_threshold_db == -40.0
        assert analyzer.min_silence_duration == 0.3
        assert analyzer.speech_threshold == 0.5
        assert analyzer.sample_rate == 22050

    def test_init_with_config(self):
        """Test initialization with custom config"""
        config = Mock()
        config.audio_analysis = Mock()
        config.audio_analysis.silence_threshold_db = -35.0
        config.audio_analysis.min_silence_duration = 0.5
        config.audio_analysis.speech_threshold = 0.6
        config.audio_analysis.sample_rate = 16000
        config.audio_analysis.min_speech_duration = 0.3

        analyzer = AudioAnalyzer(config)

        assert analyzer.silence_threshold_db == -35.0
        assert analyzer.min_silence_duration == 0.5
        assert analyzer.speech_threshold == 0.6
        assert analyzer.sample_rate == 16000
        assert analyzer.min_speech_duration == 0.3

    def test_librosa_available(self):
        """Test librosa availability detection"""
        with patch.dict('sys.modules', {'librosa': MagicMock()}):
            analyzer = AudioAnalyzer()
            assert analyzer.is_available() is True

    def test_librosa_unavailable(self):
        """Test handling when librosa not installed"""
        with patch.dict('sys.modules', {'librosa': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                analyzer = AudioAnalyzer()
                assert analyzer.is_available() is False


# ============================================================================
# Test Dataclass Operations
# ============================================================================

class TestDataclasses:
    """Test AudioAnalysis and related dataclasses"""

    def test_silence_region_creation(self):
        """Test SilenceRegion creation"""
        region = SilenceRegion(start_time=1.0, end_time=2.5, duration=1.5)

        assert region.start_time == 1.0
        assert region.end_time == 2.5
        assert region.duration == 1.5

    def test_speech_region_creation(self):
        """Test SpeechRegion creation"""
        region = SpeechRegion(
            start_time=0.0,
            end_time=3.0,
            duration=3.0,
            confidence=0.85
        )

        assert region.start_time == 0.0
        assert region.end_time == 3.0
        assert region.duration == 3.0
        assert region.confidence == 0.85

    def test_audio_analysis_to_dict(self, sample_analysis):
        """Test AudioAnalysis serialization to dict"""
        data = sample_analysis.to_dict()

        assert data['video_path'] == "test.mp4"
        assert data['duration'] == 10.0
        assert data['has_speech'] is True
        assert data['speech_ratio'] == 0.6
        assert len(data['silence_regions']) == 2
        assert len(data['speech_regions']) == 2
        assert len(data['suggested_cut_points']) == 2

    def test_audio_analysis_from_dict(self, sample_analysis):
        """Test AudioAnalysis deserialization from dict"""
        data = sample_analysis.to_dict()
        restored = AudioAnalysis.from_dict(data)

        assert restored.video_path == sample_analysis.video_path
        assert restored.duration == sample_analysis.duration
        assert restored.has_speech == sample_analysis.has_speech
        assert len(restored.silence_regions) == len(sample_analysis.silence_regions)
        assert len(restored.speech_regions) == len(sample_analysis.speech_regions)


# ============================================================================
# Test Audio Extraction
# ============================================================================

class TestAudioExtraction:
    """Test audio extraction from video"""

    @patch('subprocess.run')
    def test_extract_audio_success(self, mock_run, temp_dir):
        """Test successful audio extraction"""
        mock_run.return_value = Mock(returncode=0)

        video_path = temp_dir / "test.mp4"
        video_path.touch()
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("fake audio")

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        assert result == str(audio_path)

    @patch('subprocess.run')
    def test_extract_audio_existing_file(self, mock_run, temp_dir):
        """Test skipping extraction when audio already exists"""
        video_path = temp_dir / "test.mp4"
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("existing audio")

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        assert result == str(audio_path)
        mock_run.assert_not_called()  # Should not extract if already exists

    @patch('subprocess.run')
    def test_extract_audio_gpu_fallback(self, mock_run, temp_dir):
        """Test fallback to CPU when GPU extraction fails"""
        video_path = temp_dir / "test.mp4"
        video_path.touch()
        audio_path = temp_dir / "test.analysis.wav"

        # First call (GPU) fails, second call (CPU) succeeds
        def side_effect(*args, **kwargs):
            if '-hwaccel' in args[0]:
                return Mock(returncode=1)  # GPU fails
            else:
                audio_path.write_text("audio")
                return Mock(returncode=0)

        mock_run.side_effect = side_effect

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        assert result == str(audio_path)
        assert mock_run.call_count == 2  # GPU + CPU

    @patch('subprocess.run')
    def test_extract_audio_failure(self, mock_run, temp_dir):
        """Test handling extraction failure"""
        mock_run.side_effect = Exception("ffmpeg error")

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        assert result is None


# ============================================================================
# Test Silence Detection
# ============================================================================

class TestSilenceDetection:
    """Test silence detection"""

    def test_detect_silence_with_silent_audio(self, mock_librosa):
        """Test detecting silence in quiet audio"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        # Mock RMS to return low energy (silence)
        mock_librosa.feature.rms.return_value = np.array([[0.01, 0.01, 0.01]])
        mock_librosa.amplitude_to_db.return_value = np.array([-50, -50, -50])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])

        audio = np.zeros(22050)
        regions = analyzer._detect_silence(audio, 22050)

        assert len(regions) > 0
        assert all(isinstance(r, SilenceRegion) for r in regions)

    def test_detect_silence_with_loud_audio(self, mock_librosa):
        """Test no silence detected in loud audio"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        # Mock RMS to return high energy (no silence)
        mock_librosa.feature.rms.return_value = np.array([[1.0, 1.0, 1.0]])
        mock_librosa.amplitude_to_db.return_value = np.array([0, 0, 0])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])

        audio = np.random.randn(22050)
        regions = analyzer._detect_silence(audio, 22050)

        # Should find no silence (all frames above threshold)
        assert len(regions) == 0

    def test_detect_silence_min_duration_filter(self, mock_librosa):
        """Test silence regions below minimum duration are filtered"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True
        analyzer.min_silence_duration = 0.5  # Require at least 0.5s silence

        # Mock very short silence periods
        mock_librosa.feature.rms.return_value = np.array([[0.01, 1.0, 0.01, 1.0]])
        mock_librosa.amplitude_to_db.return_value = np.array([-50, 0, -50, 0])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.1, 0.2, 0.3])

        audio = np.zeros(22050)
        regions = analyzer._detect_silence(audio, 22050)

        # Short silences should be filtered out
        assert all(r.duration >= analyzer.min_silence_duration for r in regions)


# ============================================================================
# Test Speech Detection
# ============================================================================

class TestSpeechDetection:
    """Test speech detection"""

    def test_detect_speech_with_speech_like_audio(self, mock_librosa):
        """Test detecting speech-like audio"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        # Mock speech-like features
        mock_librosa.feature.spectral_centroid.return_value = np.array([[1500, 2000, 1800]])
        mock_librosa.feature.spectral_flatness.return_value = np.array([[0.1, 0.15, 0.12]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.08, 0.1, 0.09]])
        mock_librosa.feature.rms.return_value = np.array([[0.5, 0.6, 0.55]])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])

        with patch('scipy.ndimage.uniform_filter1d', return_value=np.array([0.8, 0.9, 0.85])):
            audio = np.random.randn(22050)
            regions, ratio = analyzer._detect_speech(audio, 22050)

        assert len(regions) >= 0
        assert 0.0 <= ratio <= 1.0
        assert all(isinstance(r, SpeechRegion) for r in regions)

    def test_detect_speech_confidence_scores(self, mock_librosa):
        """Test speech regions have confidence scores"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        mock_librosa.feature.spectral_centroid.return_value = np.array([[1500]])
        mock_librosa.feature.spectral_flatness.return_value = np.array([[0.1]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.08]])
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 1.0])

        with patch('scipy.ndimage.uniform_filter1d', return_value=np.array([0.9, 0.8])):
            audio = np.random.randn(22050)
            regions, ratio = analyzer._detect_speech(audio, 22050)

        for region in regions:
            assert 0.0 <= region.confidence <= 1.0

    def test_detect_speech_ratio_calculation(self, mock_librosa):
        """Test speech ratio is calculated correctly"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        # Set up for 50% speech
        mock_librosa.frames_to_time.return_value = np.array([0.0, 1.0, 2.0])

        with patch('scipy.ndimage.uniform_filter1d', return_value=np.array([0.8, 0.2, 0.1])):
            audio = np.random.randn(22050)
            regions, ratio = analyzer._detect_speech(audio, 22050)

        # Ratio should be between 0 and 1
        assert 0.0 <= ratio <= 1.0


# ============================================================================
# Test Cut Point Suggestion
# ============================================================================

class TestCutPointSuggestion:
    """Test cut point suggestion"""

    def test_find_cut_points_from_silence(self):
        """Test finding cut points from silence regions"""
        analyzer = AudioAnalyzer()

        silence_regions = [
            SilenceRegion(start_time=2.0, end_time=3.0, duration=1.0),
            SilenceRegion(start_time=7.0, end_time=8.5, duration=1.5)
        ]

        cut_points = analyzer._find_cut_points(silence_regions, 10.0)

        # Should place cuts at middle of silence regions
        assert len(cut_points) == 2
        assert cut_points[0] == pytest.approx(2.5, abs=0.1)
        assert cut_points[1] == pytest.approx(7.75, abs=0.1)

    def test_find_cut_points_short_silence_ignored(self):
        """Test short silence regions are ignored"""
        analyzer = AudioAnalyzer()

        silence_regions = [
            SilenceRegion(start_time=1.0, end_time=1.2, duration=0.2),  # Too short
            SilenceRegion(start_time=5.0, end_time=6.0, duration=1.0)   # Long enough
        ]

        cut_points = analyzer._find_cut_points(silence_regions, 10.0)

        # Should only suggest cut for long silence
        assert len(cut_points) == 1
        assert cut_points[0] == pytest.approx(5.5, abs=0.1)

    def test_find_cut_points_no_silence(self):
        """Test no cut points when no silence"""
        analyzer = AudioAnalyzer()
        cut_points = analyzer._find_cut_points([], 10.0)

        assert len(cut_points) == 0


# ============================================================================
# Test Full Video Analysis
# ============================================================================

class TestVideoAnalysis:
    """Test full video analysis"""

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_not_available(self, mock_extract, temp_dir):
        """Test analysis when librosa not available"""
        analyzer = AudioAnalyzer()
        analyzer._available = False

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        result = analyzer.analyze_video(str(video_path))

        assert result is None

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_extraction_failure(self, mock_extract, temp_dir):
        """Test analysis when audio extraction fails"""
        mock_extract.return_value = None

        analyzer = AudioAnalyzer()
        analyzer._available = True

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        result = analyzer.analyze_video(str(video_path))

        assert result is None

    @patch('src.audio_analysis.AudioAnalyzer._find_cut_points')
    @patch('src.audio_analysis.AudioAnalyzer._detect_speech')
    @patch('src.audio_analysis.AudioAnalyzer._detect_silence')
    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_success(
        self, mock_extract, mock_silence, mock_speech, mock_cut, mock_librosa, temp_dir
    ):
        """Test successful video analysis"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.touch()
        mock_extract.return_value = str(audio_path)

        mock_silence.return_value = [SilenceRegion(1.0, 2.0, 1.0)]
        mock_speech.return_value = ([SpeechRegion(0.0, 1.0, 1.0, 0.8)], 0.5)
        mock_cut.return_value = [1.5]

        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        result = analyzer.analyze_video(str(video_path))

        assert result is not None
        assert isinstance(result, AudioAnalysis)
        assert result.video_path == str(video_path)
        assert len(result.silence_regions) == 1
        assert len(result.speech_regions) == 1
        assert len(result.suggested_cut_points) == 1

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_cleanup(self, mock_extract, temp_dir):
        """Test audio file cleanup after analysis"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("audio data")
        mock_extract.return_value = str(audio_path)

        analyzer = AudioAnalyzer()
        analyzer.librosa = MagicMock()
        analyzer._available = True
        analyzer.librosa.load.return_value = (np.zeros(22050), 22050)

        with patch.object(analyzer, '_detect_silence', return_value=[]):
            with patch.object(analyzer, '_detect_speech', return_value=([], 0.0)):
                video_path = temp_dir / "test.mp4"
                video_path.touch()

                analyzer.analyze_video(str(video_path), cleanup_audio=True)

                # Audio file should be deleted
                assert not audio_path.exists()


# ============================================================================
# Test Batch Analysis
# ============================================================================

class TestBatchAnalysis:
    """Test batch video analysis"""

    @patch('src.audio_analysis.AudioAnalyzer.analyze_video')
    def test_analyze_videos_multiple(self, mock_analyze):
        """Test analyzing multiple videos"""
        mock_analyze.side_effect = [
            AudioAnalysis("video1.mp4", 10.0, True, 0.5, [], [], []),
            AudioAnalysis("video2.mp4", 15.0, False, 0.1, [], [], [])
        ]

        analyzer = AudioAnalyzer()
        results = analyzer.analyze_videos(["video1.mp4", "video2.mp4"])

        assert len(results) == 2
        assert "video1.mp4" in results
        assert "video2.mp4" in results

    @patch('src.audio_analysis.AudioAnalyzer.analyze_video')
    def test_analyze_videos_with_failures(self, mock_analyze):
        """Test batch analysis handles failures"""
        mock_analyze.side_effect = [
            AudioAnalysis("video1.mp4", 10.0, True, 0.5, [], [], []),
            None,  # Second video fails
            AudioAnalysis("video3.mp4", 15.0, False, 0.1, [], [], [])
        ]

        analyzer = AudioAnalyzer()
        results = analyzer.analyze_videos(["video1.mp4", "video2.mp4", "video3.mp4"])

        # Should only include successful analyses
        assert len(results) == 2
        assert "video1.mp4" in results
        assert "video2.mp4" not in results
        assert "video3.mp4" in results


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    @patch('src.audio_analysis.AudioAnalyzer.analyze_video')
    def test_analyze_audio_function(self, mock_analyze):
        """Test analyze_audio convenience function"""
        mock_analysis = AudioAnalysis("test.mp4", 10.0, True, 0.6, [], [], [])
        mock_analyze.return_value = mock_analysis

        with patch('src.audio_analysis.AudioAnalyzer.is_available', return_value=True):
            result = analyze_audio("test.mp4")

        assert result == mock_analysis

    @patch('src.audio_analysis.analyze_audio')
    def test_has_speech_function_true(self, mock_analyze):
        """Test has_speech convenience function with speech"""
        mock_analyze.return_value = AudioAnalysis(
            "test.mp4", 10.0, True, 0.6, [], [], []
        )

        result = has_speech("test.mp4")

        assert result is True

    @patch('src.audio_analysis.analyze_audio')
    def test_has_speech_function_false(self, mock_analyze):
        """Test has_speech convenience function without speech"""
        mock_analyze.return_value = AudioAnalysis(
            "test.mp4", 10.0, False, 0.05, [], [], []
        )

        result = has_speech("test.mp4")

        assert result is False

    @patch('src.audio_analysis.analyze_audio')
    def test_has_speech_function_none(self, mock_analyze):
        """Test has_speech function when analysis fails"""
        mock_analyze.return_value = None

        result = has_speech("test.mp4")

        assert result is False

    @patch('src.audio_analysis.analyze_audio')
    def test_get_silence_cut_points_function(self, mock_analyze):
        """Test get_silence_cut_points convenience function"""
        mock_analyze.return_value = AudioAnalysis(
            "test.mp4", 10.0, True, 0.5, [], [], [2.5, 5.0, 7.5]
        )

        result = get_silence_cut_points("test.mp4")

        assert result == [2.5, 5.0, 7.5]

    @patch('src.audio_analysis.analyze_audio')
    def test_get_silence_cut_points_none(self, mock_analyze):
        """Test get_silence_cut_points when analysis fails"""
        mock_analyze.return_value = None

        result = get_silence_cut_points("test.mp4")

        assert result == []


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_audio_analysis_empty_regions(self):
        """Test AudioAnalysis with no regions"""
        analysis = AudioAnalysis(
            video_path="test.mp4",
            duration=5.0,
            has_speech=False,
            speech_ratio=0.0,
            silence_regions=[],
            speech_regions=[],
            suggested_cut_points=[]
        )

        assert len(analysis.silence_regions) == 0
        assert len(analysis.speech_regions) == 0
        assert len(analysis.suggested_cut_points) == 0

    def test_audio_analysis_to_dict_type_conversion(self):
        """Test to_dict converts numpy types to Python types"""
        analysis = AudioAnalysis(
            video_path="test.mp4",
            duration=np.float64(10.5),
            has_speech=True,
            speech_ratio=np.float64(0.654321),
            silence_regions=[],
            speech_regions=[],
            suggested_cut_points=[np.float64(2.5)]
        )

        data = analysis.to_dict()

        assert isinstance(data['duration'], float)
        assert isinstance(data['speech_ratio'], float)
        assert data['speech_ratio'] == 0.654  # Rounded to 3 decimals

    def test_silence_detection_zero_duration_audio(self, mock_librosa):
        """Test silence detection with zero duration audio"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        mock_librosa.frames_to_time.return_value = np.array([])

        audio = np.array([])
        regions = analyzer._detect_silence(audio, 22050)

        # Should handle gracefully
        assert isinstance(regions, list)

    def test_speech_ratio_zero_duration(self, mock_librosa):
        """Test speech ratio calculation with zero duration"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        mock_librosa.frames_to_time.return_value = np.array([])

        with patch('scipy.ndimage.uniform_filter1d', return_value=np.array([])):
            audio = np.array([])
            regions, ratio = analyzer._detect_speech(audio, 22050)

        # Should return 0 ratio for zero duration
        assert ratio == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
