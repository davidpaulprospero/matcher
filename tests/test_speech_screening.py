"""
Comprehensive tests for speech screening module.

Covers:
- SpeechScreener initialization
- Audio clip download (first N seconds)
- Whisper VAD-based speech detection
- Speech duration calculation
- Fallback strategies (accept/reject on error)
- Batch video screening
- Temporary file cleanup

Created: 2026-01-09 (Phase 7.2)
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import tempfile
import sys
from pathlib import Path
from contextlib import contextmanager

from src.downloader.speech_screening import SpeechScreener


# ============================================================================
# Test Helpers
# ============================================================================

@contextmanager
def mock_transcription_module(return_value):
    """Context manager to mock src.transcription module with transcribe_voiceover_audio"""
    mock_transcription = MagicMock()
    mock_transcribe = MagicMock(return_value=return_value)
    mock_transcription.transcribe_voiceover_audio = mock_transcribe

    with patch.dict(sys.modules, {'src.transcription': mock_transcription}):
        yield mock_transcribe


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with speech screening settings"""
    config = Mock()
    config.download = Mock()
    config.download.speech_screening = Mock()
    config.download.speech_screening.screening_duration = 5.0
    config.download.speech_screening.min_speech_duration = 0.5
    config.download.speech_screening.whisper_model = "base"
    config.download.speech_screening.timeout_per_video = 30
    config.download.speech_screening.fallback_on_error = "accept"
    config.download.speech_screening.reject_with_speech = True
    config.download.audio_first = Mock()
    config.download.audio_first.audio_quality = 5
    config.download.ffmpeg_location = ""
    return config


@pytest.fixture
def screener(mock_config):
    """Create SpeechScreener instance"""
    return SpeechScreener(
        config=mock_config,
        cookies_args=['--cookies', 'cookies.txt']
    )


@pytest.fixture
def sample_video():
    """Create sample video metadata"""
    return {
        'id': 'abc123',
        'webpage_url': 'https://youtube.com/watch?v=abc123',
        'title': 'Sample Beach Video',
        'duration': 120
    }


@pytest.fixture
def sample_videos():
    """Create list of sample videos"""
    return [
        {
            'id': 'vid1',
            'webpage_url': 'https://youtube.com/watch?v=vid1',
            'title': 'Silent Beach Footage',
            'duration': 60
        },
        {
            'id': 'vid2',
            'webpage_url': 'https://youtube.com/watch?v=vid2',
            'title': 'Travel Vlog with Commentary',
            'duration': 90
        },
        {
            'id': 'vid3',
            'webpage_url': 'https://youtube.com/watch?v=vid3',
            'title': 'Nature Sounds Only',
            'duration': 45
        }
    ]


# ============================================================================
# Test Initialization
# ============================================================================

class TestSpeechScreenerInit:
    """Test SpeechScreener initialization"""

    def test_init_with_config(self, mock_config):
        """Test initialization with config"""
        screener = SpeechScreener(
            config=mock_config,
            cookies_args=['--cookies', 'test.txt']
        )

        assert screener.config == mock_config
        assert screener.download_config == mock_config.download
        assert screener.cookies_args == ['--cookies', 'test.txt']

    def test_init_with_empty_cookies(self, mock_config):
        """Test initialization with empty cookies"""
        screener = SpeechScreener(config=mock_config, cookies_args=[])

        assert screener.cookies_args == []


# ============================================================================
# Test Audio Clip Download
# ============================================================================

class TestAudioClipDownload:
    """Test audio clip download for speech screening"""

    @patch('subprocess.run')
    def test_download_audio_clip_success(self, mock_run, screener):
        """Test successful audio clip download"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            mock_run.return_value = Mock(returncode=0)

            result = screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
                duration=5.0
            )

            assert result == audio_file
            mock_run.assert_called_once()
            # Verify yt-dlp command structure
            cmd = mock_run.call_args[0][0]
            assert 'yt-dlp' in cmd
            assert '--download-sections' in cmd
            assert '*0-5.0' in cmd
            assert '-x' in cmd
            assert '--audio-format' in cmd
            assert 'mp3' in cmd

    @patch('subprocess.run')
    def test_download_audio_clip_timeout(self, mock_run, screener):
        """Test audio download timeout handling"""
        import subprocess

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30)

            result = screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path
            )

            assert result is None

    @patch('subprocess.run')
    def test_download_audio_clip_error(self, mock_run, screener):
        """Test audio download error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_run.return_value = Mock(returncode=1)

            result = screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path
            )

            assert result is None

    @patch('subprocess.run')
    def test_download_audio_clip_with_ffmpeg_location(self, mock_run, screener):
        """Test audio download with custom FFmpeg location"""
        screener.download_config.ffmpeg_location = "/usr/local/bin/ffmpeg"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")
            mock_run.return_value = Mock(returncode=0)

            result = screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path
            )

            # Verify FFmpeg location is in command
            cmd = mock_run.call_args[0][0]
            assert '--ffmpeg-location' in cmd
            assert '/usr/local/bin/ffmpeg' in cmd

    @patch('subprocess.run')
    def test_download_audio_clip_custom_duration(self, mock_run, screener):
        """Test audio download with custom duration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")
            mock_run.return_value = Mock(returncode=0)

            result = screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
                duration=10.0
            )

            # Verify custom duration in command
            cmd = mock_run.call_args[0][0]
            assert '*0-10.0' in cmd


# ============================================================================
# Test Speech Detection
# ============================================================================

class TestSpeechDetection:
    """Test Whisper VAD-based speech detection"""

    def test_screen_video_with_speech(self, screener, sample_video):
        """Test screening video with speech detected"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([
                    {'start': 0.0, 'end': 2.5, 'text': 'Hello world'},
                    {'start': 2.5, 'end': 4.0, 'text': 'This is a test'}
                ]) as mock_transcribe:
                    has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

                    assert has_speech is True
                    assert duration == 4.0  # 2.5 + 1.5 seconds
                    mock_transcribe.assert_called_once()

    def test_screen_video_no_speech(self, screener, sample_video):
        """Test screening video with no speech"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([]):
                    has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

                    assert has_speech is False
                    assert duration == 0.0

    def test_screen_video_below_threshold(self, screener, sample_video):
        """Test screening video with speech below minimum threshold"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([{'start': 0.0, 'end': 0.3, 'text': 'Hi'}]):
                    has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

                    assert has_speech is False  # Below 0.5s threshold
                    assert duration == 0.3

    @patch('src.downloader.speech_screening.SpeechScreener.download_audio_clip')
    def test_screen_video_download_failed_accept(self, mock_download, screener, sample_video):
        """Test screening when download fails with accept fallback"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_download.return_value = None  # Download failed

            # Config has fallback_on_error = "accept"
            has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

            assert has_speech is False  # Accept = has_speech False
            assert duration == 0.0

    @patch('src.downloader.speech_screening.SpeechScreener.download_audio_clip')
    def test_screen_video_download_failed_reject(self, mock_download, screener, sample_video):
        """Test screening when download fails with reject fallback"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_download.return_value = None

            # Change fallback to reject
            screener.download_config.speech_screening.fallback_on_error = "reject"

            has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

            assert has_speech is True  # Reject = has_speech True
            assert duration == 0.0

    def test_screen_video_transcription_error_accept(self, screener, sample_video):
        """Test screening when transcription fails with accept fallback"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                # Create mock that raises an exception
                mock_transcription = MagicMock()
                mock_transcribe = MagicMock(side_effect=Exception("Whisper error"))
                mock_transcription._transcribe_with_shared_model = mock_transcribe

                with patch.dict(sys.modules, {'src.transcription': mock_transcription}):
                    has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

                    assert has_speech is False  # Accept fallback
                    assert duration == 0.0
                    # Verify cleanup happened
                    assert not audio_file.exists()

    def test_screen_video_transcription_error_reject(self, screener, sample_video):
        """Test screening when transcription fails with reject fallback"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                # Create mock that raises an exception
                mock_transcription = MagicMock()
                mock_transcribe = MagicMock(side_effect=Exception("Whisper error"))
                mock_transcription.transcribe_voiceover_audio = mock_transcribe

                # Change fallback
                screener.download_config.speech_screening.fallback_on_error = "reject"

                with patch.dict(sys.modules, {'src.transcription': mock_transcription}):
                    has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

                    assert has_speech is True  # Reject fallback
                    assert duration == 0.0


# ============================================================================
# Test Batch Video Screening
# ============================================================================

class TestBatchScreening:
    """Test batch video screening"""

    @patch('src.downloader.speech_screening.SpeechScreener.screen_video_for_speech')
    def test_screen_approved_videos_all_pass(self, mock_screen, screener, sample_videos):
        """Test screening multiple videos - all pass (no speech)"""
        # Mock all videos as having no speech
        mock_screen.return_value = (False, 0.0)

        result = screener.screen_approved_videos(sample_videos, keyword="beach")

        assert len(result) == 3  # All 3 passed
        assert mock_screen.call_count == 3

    @patch('src.downloader.speech_screening.SpeechScreener.screen_video_for_speech')
    def test_screen_approved_videos_some_rejected(self, mock_screen, screener, sample_videos):
        """Test screening multiple videos - some with speech"""
        # First video: no speech, second: speech, third: no speech
        mock_screen.side_effect = [
            (False, 0.0),
            (True, 2.5),
            (False, 0.0)
        ]

        result = screener.screen_approved_videos(sample_videos, keyword="beach")

        assert len(result) == 2  # Videos 1 and 3 passed
        assert result[0]['id'] == 'vid1'
        assert result[1]['id'] == 'vid3'

    @patch('src.downloader.speech_screening.SpeechScreener.screen_video_for_speech')
    def test_screen_approved_videos_reject_disabled(self, mock_screen, screener, sample_videos):
        """Test screening with reject_with_speech disabled (logging only)"""
        # Disable rejection
        screener.download_config.speech_screening.reject_with_speech = False

        # All videos have speech
        mock_screen.return_value = (True, 2.0)

        result = screener.screen_approved_videos(sample_videos, keyword="beach")

        # All 3 pass despite having speech (logging only mode)
        assert len(result) == 3

    @patch('src.downloader.speech_screening.SpeechScreener.screen_video_for_speech')
    def test_screen_approved_videos_empty_list(self, mock_screen, screener):
        """Test screening with empty video list"""
        result = screener.screen_approved_videos([], keyword="beach")

        assert len(result) == 0
        mock_screen.assert_not_called()

    @patch('src.downloader.speech_screening.SpeechScreener.screen_video_for_speech')
    def test_screen_approved_videos_temp_dir_cleanup(self, mock_screen, screener, sample_videos):
        """Test that temporary directory is cleaned up"""
        import os

        mock_screen.return_value = (False, 0.0)

        # Track temp dir creation
        original_tempdir = tempfile.TemporaryDirectory
        created_dirs = []

        def track_tempdir(*args, **kwargs):
            td = original_tempdir(*args, **kwargs)
            created_dirs.append(td.name)
            return td

        with patch('tempfile.TemporaryDirectory', side_effect=track_tempdir):
            result = screener.screen_approved_videos(sample_videos, keyword="beach footage")

        # Verify temp dir was created and cleaned up
        assert len(created_dirs) == 1
        assert not os.path.exists(created_dirs[0])  # Should be deleted


# ============================================================================
# Test Configuration Handling
# ============================================================================

class TestConfigurationHandling:
    """Test configuration edge cases"""

    def test_no_speech_config(self, mock_config):
        """Test when speech_screening config is missing"""
        mock_config.download.speech_screening = None
        screener = SpeechScreener(config=mock_config, cookies_args=[])

        # Should use defaults
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Don't actually call screen_video_for_speech since it needs mocking
            # Just verify initialization worked
            assert screener.download_config is not None

    def test_custom_whisper_model(self, mock_config, sample_video):
        """Test with custom Whisper model"""
        mock_config.download.speech_screening.whisper_model = "small"
        screener = SpeechScreener(config=mock_config, cookies_args=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([]) as mock_transcribe:
                    screener.screen_video_for_speech(sample_video, temp_path)

                    # Verify custom model was used
                    call_args = mock_transcribe.call_args
                    assert call_args[1]['model_name'] == 'small'

    def test_custom_thresholds(self, mock_config, sample_video):
        """Test with custom duration thresholds"""
        mock_config.download.speech_screening.min_speech_duration = 2.0
        screener = SpeechScreener(config=mock_config, cookies_args=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([{'start': 0.0, 'end': 1.5, 'text': 'Short'}]):
                    has_speech, duration = screener.screen_video_for_speech(sample_video, temp_path)

                    assert has_speech is False  # Below custom 2.0s threshold
                    assert duration == 1.5


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_video_without_id(self, screener):
        """Test screening video without ID"""
        video = {
            'webpage_url': 'https://youtube.com/watch?v=test',
            'title': 'No ID Video'
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / ".mp3"  # Empty ID
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([]):
                    has_speech, duration = screener.screen_video_for_speech(video, temp_path)

                    # Should handle missing ID gracefully
                    assert has_speech is False

    def test_video_with_very_long_title(self, screener):
        """Test video with very long title (truncation)"""
        video = {
            'id': 'abc123',
            'title': 'A' * 200,  # Very long title
            'webpage_url': 'https://youtube.com/watch?v=abc123'
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch.object(screener, 'download_audio_clip', return_value=audio_file):
                with mock_transcription_module([]):
                    # Should truncate title in logs without error
                    has_speech, duration = screener.screen_video_for_speech(video, temp_path)

                    assert has_speech is False

    @patch('src.downloader.speech_screening.SpeechScreener.screen_video_for_speech')
    def test_special_characters_in_keyword(self, mock_screen, screener):
        """Test keyword with special characters (temp dir naming)"""
        videos = [{'id': 'vid1', 'title': 'Test', 'webpage_url': 'https://youtube.com/watch?v=vid1'}]
        mock_screen.return_value = (False, 0.0)

        # Keyword with special chars that need sanitization
        result = screener.screen_approved_videos(videos, keyword="beach/ocean<>waves?*")

        # Should handle special chars in temp dir name
        assert len(result) == 1
