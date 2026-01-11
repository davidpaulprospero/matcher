"""
Additional tests for audio_analysis.py to cover missed lines.

Covers:
- Line 152: GPU extraction success path (file created on first attempt)
- Line 170: Both GPU and CPU extraction fail (no file created)
- Line 231: Silence region append when duration meets threshold
- Lines 307-308: scipy uniform_filter1d exception handling
- Lines 440-442: Exception during audio analysis
- Lines 449-450: File unlink exception during cleanup
- Line 490: analyze_audio when librosa not available

Created: 2026-01-11 (Coverage expansion)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import tempfile
import shutil
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
    mock_lib.load.return_value = (np.zeros(22050), 22050)
    mock_lib.feature.rms.return_value = np.array([[0.1, 0.2, 0.1]])
    mock_lib.feature.spectral_centroid.return_value = np.array([[1000, 2000, 1500]])
    mock_lib.feature.spectral_flatness.return_value = np.array([[0.1, 0.2, 0.15]])
    mock_lib.feature.zero_crossing_rate.return_value = np.array([[0.05, 0.1, 0.07]])
    mock_lib.amplitude_to_db.return_value = np.array([-30, -20, -35])
    mock_lib.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])
    return mock_lib


# ============================================================================
# Test GPU Extraction Success (Line 152)
# ============================================================================

class TestGPUExtractionSuccess:
    """Test cases where GPU extraction succeeds immediately (line 152)"""

    @patch('subprocess.run')
    def test_extract_audio_gpu_success_first_attempt(self, mock_run, temp_dir):
        """Test GPU extraction succeeds and creates file on first attempt"""
        video_path = temp_dir / "test.mp4"
        video_path.touch()
        audio_path = temp_dir / "test.analysis.wav"

        # GPU extraction succeeds - file is created
        def side_effect(*args, **kwargs):
            # Create the audio file as if ffmpeg succeeded
            audio_path.write_text("audio data")
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        # Should return the path and only call subprocess once (GPU succeeded)
        assert result == str(audio_path)
        assert mock_run.call_count == 1
        # Verify GPU command was used (has -hwaccel cuda)
        cmd_used = mock_run.call_args[0][0]
        assert '-hwaccel' in cmd_used


# ============================================================================
# Test Both GPU and CPU Extraction Fail (Line 170)
# ============================================================================

class TestExtractionBothFail:
    """Test cases where both GPU and CPU extraction fail (line 170)"""

    @patch('subprocess.run')
    def test_extract_audio_both_gpu_and_cpu_fail(self, mock_run, temp_dir):
        """Test when both GPU and CPU extraction fail to create file"""
        video_path = temp_dir / "test.mp4"
        video_path.touch()
        audio_path = temp_dir / "test.analysis.wav"

        # Both attempts succeed (returncode 0) but no file is created
        mock_run.return_value = Mock(returncode=0)

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        # Should return None since no file was created
        assert result is None
        # Should have tried both GPU and CPU
        assert mock_run.call_count == 2

    @patch('subprocess.run')
    def test_extract_audio_empty_file_created(self, mock_run, temp_dir):
        """Test when ffmpeg creates empty file (0 bytes)"""
        video_path = temp_dir / "test.mp4"
        video_path.touch()
        audio_path = temp_dir / "test.analysis.wav"

        # Create empty file (size 0) on GPU attempt
        def side_effect(*args, **kwargs):
            audio_path.touch()  # Creates empty file
            return Mock(returncode=0)

        mock_run.side_effect = side_effect

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        # Should return None since file has 0 size
        assert result is None


# ============================================================================
# Test Silence Region Append (Line 231)
# ============================================================================

class TestSilenceRegionAppend:
    """Test silence region append when duration meets threshold (line 231)"""

    def test_detect_silence_appends_region_when_meets_duration(self, mock_librosa):
        """Test that silence region is appended when duration >= min_silence_duration"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True
        analyzer.min_silence_duration = 0.3  # Set threshold

        # Mock a sequence: loud -> silent -> loud (silence duration = 0.5s > 0.3s threshold)
        # Times: 0.0, 0.5, 1.0 (hop = 0.5s)
        # RMS dB: 0, -50, 0 (middle frame is silent)
        mock_librosa.feature.rms.return_value = np.array([[1.0, 0.001, 1.0]])
        mock_librosa.amplitude_to_db.return_value = np.array([0, -50, 0])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])

        audio = np.zeros(22050)
        regions = analyzer._detect_silence(audio, 22050)

        # Should detect one silence region from 0.5 to 1.0 (0.5s duration >= 0.3s threshold)
        assert len(regions) == 1
        assert regions[0].start_time == 0.5
        assert regions[0].duration >= analyzer.min_silence_duration

    def test_detect_silence_multiple_regions_appended(self, mock_librosa):
        """Test multiple silence regions are appended"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True
        analyzer.min_silence_duration = 0.2

        # Pattern: loud, silent, loud, silent, loud
        # Times: 0.0, 0.25, 0.5, 0.75, 1.0
        mock_librosa.feature.rms.return_value = np.array([[1.0, 0.001, 1.0, 0.001, 1.0]])
        mock_librosa.amplitude_to_db.return_value = np.array([0, -50, 0, -50, 0])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        audio = np.zeros(22050)
        regions = analyzer._detect_silence(audio, 22050)

        # Should detect 2 silence regions
        assert len(regions) == 2
        for region in regions:
            assert isinstance(region, SilenceRegion)


# ============================================================================
# Test scipy Filter Exception (Lines 307-308)
# ============================================================================

class TestScipyFilterException:
    """Test scipy uniform_filter1d exception handling (lines 307-308)"""

    def test_detect_speech_handles_scipy_exception(self, mock_librosa):
        """Test that scipy filter exception is caught and handled"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        # Set up normal speech detection mocks
        mock_librosa.feature.spectral_centroid.return_value = np.array([[1500, 2000]])
        mock_librosa.feature.spectral_flatness.return_value = np.array([[0.1, 0.15]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.08, 0.1]])
        mock_librosa.feature.rms.return_value = np.array([[0.5, 0.6]])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 1.0])

        # Make scipy.ndimage.uniform_filter1d raise an exception
        with patch('scipy.ndimage.uniform_filter1d', side_effect=RuntimeError("Filter error")):
            audio = np.random.randn(22050)
            regions, ratio = analyzer._detect_speech(audio, 22050)

        # Should still return valid results (exception was caught)
        assert isinstance(regions, list)
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_detect_speech_handles_import_error_in_filter(self, mock_librosa):
        """Test handling when scipy.ndimage raises ImportError"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True

        mock_librosa.feature.spectral_centroid.return_value = np.array([[1500]])
        mock_librosa.feature.spectral_flatness.return_value = np.array([[0.1]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.08]])
        mock_librosa.feature.rms.return_value = np.array([[0.5]])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 1.0])

        # Make the filter raise a different exception type
        with patch('scipy.ndimage.uniform_filter1d', side_effect=ValueError("Bad value")):
            audio = np.random.randn(22050)
            regions, ratio = analyzer._detect_speech(audio, 22050)

        # Should still succeed
        assert isinstance(regions, list)


# ============================================================================
# Test Analysis Exception (Lines 440-442)
# ============================================================================

class TestAnalysisException:
    """Test exception handling during audio analysis (lines 440-442)"""

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_librosa_load_exception(self, mock_extract, temp_dir):
        """Test handling when librosa.load raises exception"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("fake audio")
        mock_extract.return_value = str(audio_path)

        analyzer = AudioAnalyzer()
        analyzer.librosa = MagicMock()
        analyzer._available = True
        # Make librosa.load raise an exception
        analyzer.librosa.load.side_effect = RuntimeError("Failed to load audio file")

        video_path = temp_dir / "test.mp4"
        video_path.touch()

        result = analyzer.analyze_video(str(video_path))

        # Should return None and handle exception gracefully
        assert result is None

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_detect_silence_exception(self, mock_extract, temp_dir):
        """Test handling when _detect_silence raises exception"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("fake audio")
        mock_extract.return_value = str(audio_path)

        analyzer = AudioAnalyzer()
        analyzer.librosa = MagicMock()
        analyzer._available = True
        analyzer.librosa.load.return_value = (np.zeros(22050), 22050)

        with patch.object(analyzer, '_detect_silence', side_effect=RuntimeError("Silence detection failed")):
            video_path = temp_dir / "test.mp4"
            video_path.touch()

            result = analyzer.analyze_video(str(video_path))

        # Should return None
        assert result is None

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_detect_speech_exception(self, mock_extract, temp_dir):
        """Test handling when _detect_speech raises exception"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("fake audio")
        mock_extract.return_value = str(audio_path)

        analyzer = AudioAnalyzer()
        analyzer.librosa = MagicMock()
        analyzer._available = True
        analyzer.librosa.load.return_value = (np.zeros(22050), 22050)

        with patch.object(analyzer, '_detect_silence', return_value=[]):
            with patch.object(analyzer, '_detect_speech', side_effect=RuntimeError("Speech detection failed")):
                video_path = temp_dir / "test.mp4"
                video_path.touch()

                result = analyzer.analyze_video(str(video_path))

        # Should return None
        assert result is None


# ============================================================================
# Test Cleanup Exception (Lines 449-450)
# ============================================================================

class TestCleanupException:
    """Test file unlink exception during cleanup (lines 449-450)"""

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_cleanup_unlink_exception(self, mock_extract, temp_dir):
        """Test that cleanup continues even if unlink fails"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("fake audio")
        mock_extract.return_value = str(audio_path)

        analyzer = AudioAnalyzer()
        analyzer.librosa = MagicMock()
        analyzer._available = True
        analyzer.librosa.load.return_value = (np.zeros(22050), 22050)

        with patch.object(analyzer, '_detect_silence', return_value=[]):
            with patch.object(analyzer, '_detect_speech', return_value=([], 0.0)):
                # Mock Path.unlink to raise PermissionError
                with patch.object(Path, 'unlink', side_effect=PermissionError("Access denied")):
                    video_path = temp_dir / "test.mp4"
                    video_path.touch()

                    # Should still succeed - cleanup exception is swallowed
                    result = analyzer.analyze_video(str(video_path), cleanup_audio=True)

        # Analysis should succeed even if cleanup fails
        assert result is not None
        assert isinstance(result, AudioAnalysis)

    @patch('src.audio_analysis.AudioAnalyzer._extract_audio')
    def test_analyze_video_cleanup_exception_after_analysis_failure(self, mock_extract, temp_dir):
        """Test cleanup exception handling when analysis also fails"""
        audio_path = temp_dir / "test.analysis.wav"
        audio_path.write_text("fake audio")
        mock_extract.return_value = str(audio_path)

        analyzer = AudioAnalyzer()
        analyzer.librosa = MagicMock()
        analyzer._available = True
        # Make analysis fail
        analyzer.librosa.load.side_effect = RuntimeError("Load failed")

        # Mock Path.unlink to also raise an exception
        with patch.object(Path, 'unlink', side_effect=OSError("Cannot delete")):
            video_path = temp_dir / "test.mp4"
            video_path.touch()

            # Should return None (analysis failed) and not crash on cleanup
            result = analyzer.analyze_video(str(video_path), cleanup_audio=True)

        assert result is None


# ============================================================================
# Test analyze_audio Not Available (Line 490)
# ============================================================================

class TestAnalyzeAudioNotAvailable:
    """Test analyze_audio convenience function when librosa not available (line 490)"""

    def test_analyze_audio_returns_none_when_unavailable(self):
        """Test analyze_audio returns None when librosa is not installed"""
        with patch('src.audio_analysis.AudioAnalyzer.is_available', return_value=False):
            with patch('src.audio_analysis.AudioAnalyzer.__init__', return_value=None) as mock_init:
                # Set up mock analyzer
                mock_init.return_value = None
                with patch('src.audio_analysis.AudioAnalyzer') as MockAnalyzer:
                    instance = MockAnalyzer.return_value
                    instance.is_available.return_value = False

                    result = analyze_audio("test.mp4")

                    assert result is None

    def test_analyze_audio_librosa_import_failure(self):
        """Test analyze_audio when librosa cannot be imported"""
        # Create analyzer that will report unavailable
        with patch.dict('sys.modules', {'librosa': None}):
            with patch('builtins.__import__', side_effect=ImportError("No librosa")):
                analyzer = AudioAnalyzer()

                # is_available should be False
                assert analyzer.is_available() is False


# ============================================================================
# Additional Edge Cases for Complete Coverage
# ============================================================================

class TestAdditionalEdgeCases:
    """Additional edge cases for complete coverage"""

    def test_detect_silence_ends_in_silence_meeting_threshold(self, mock_librosa):
        """Test silence at end that meets minimum duration threshold"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True
        analyzer.min_silence_duration = 0.3

        # Pattern: loud at start, then silence until end (0.5s of silence)
        mock_librosa.feature.rms.return_value = np.array([[1.0, 0.001, 0.001]])
        mock_librosa.amplitude_to_db.return_value = np.array([0, -50, -50])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.5, 1.0])

        audio = np.zeros(22050)
        regions = analyzer._detect_silence(audio, 22050)

        # Should detect silence region that extends to the end
        assert len(regions) == 1
        assert regions[0].end_time == 1.0

    def test_detect_speech_ends_in_speech_meeting_threshold(self, mock_librosa):
        """Test speech at end that meets minimum duration threshold"""
        analyzer = AudioAnalyzer()
        analyzer.librosa = mock_librosa
        analyzer._available = True
        analyzer.speech_threshold = 0.4
        analyzer.min_speech_duration = 0.2

        # Set up features that indicate speech in the middle and end
        mock_librosa.feature.spectral_centroid.return_value = np.array([[500, 1500, 2000, 1800]])
        mock_librosa.feature.spectral_flatness.return_value = np.array([[0.5, 0.1, 0.1, 0.1]])
        mock_librosa.feature.zero_crossing_rate.return_value = np.array([[0.01, 0.08, 0.1, 0.09]])
        mock_librosa.feature.rms.return_value = np.array([[0.001, 0.5, 0.6, 0.55]])
        mock_librosa.frames_to_time.return_value = np.array([0.0, 0.3, 0.6, 1.0])

        with patch('scipy.ndimage.uniform_filter1d', side_effect=lambda x, size: x):
            audio = np.random.randn(22050)
            regions, ratio = analyzer._detect_speech(audio, 22050)

        # Should have speech regions and valid ratio
        assert isinstance(regions, list)
        assert 0.0 <= ratio <= 1.0

    @patch('subprocess.run')
    def test_extract_audio_cpu_fallback_creates_file(self, mock_run, temp_dir):
        """Test CPU fallback successfully creates file when GPU fails"""
        video_path = temp_dir / "test.mp4"
        video_path.touch()
        audio_path = temp_dir / "test.analysis.wav"

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # GPU fails - no file created
                return Mock(returncode=1)
            else:
                # CPU succeeds - create file
                audio_path.write_text("audio data from cpu")
                return Mock(returncode=0)

        mock_run.side_effect = side_effect

        analyzer = AudioAnalyzer()
        result = analyzer._extract_audio(str(video_path))

        # Should succeed with CPU fallback
        assert result == str(audio_path)
        assert mock_run.call_count == 2

    def test_audio_analysis_from_dict_missing_optional_fields(self):
        """Test from_dict handles missing optional fields"""
        data = {
            'video_path': 'test.mp4',
            'duration': 10.0,
            'has_speech': False,
            'speech_ratio': 0.0,
            # Missing silence_regions, speech_regions, suggested_cut_points
        }

        analysis = AudioAnalysis.from_dict(data)

        assert analysis.video_path == 'test.mp4'
        assert analysis.silence_regions == []
        assert analysis.speech_regions == []
        assert analysis.suggested_cut_points == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
