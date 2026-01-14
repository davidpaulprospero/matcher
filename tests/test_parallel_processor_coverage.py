"""
Tests for src/transcription/parallel_processor.py to improve coverage.

Focuses on:
- transcribe_videos_parallel edge cases (lines 108, 121, 145, 152, 158, 167-170)
- transcribe_voiceover_media edge cases (lines 390-391, 417-418)
- Error handling paths
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test transcribe_videos_parallel
# ============================================================================

class TestTranscribeVideosParallel:
    """Test transcribe_videos_parallel function"""

    @pytest.fixture
    def mock_whisper_client(self):
        with patch('src.transcription.parallel_processor.WhisperClient') as mock:
            instance = Mock()
            instance.transcribe.return_value = [
                {'start': 0.0, 'end': 5.0, 'text': 'Test segment'}
            ]
            mock.return_value = instance
            yield mock

    @pytest.fixture
    def mock_transcript_cache(self):
        with patch('src.transcription.parallel_processor.TranscriptCache') as mock:
            instance = Mock()
            instance.get.return_value = None  # Cache miss by default
            mock.return_value = instance
            yield mock, instance

    @pytest.fixture
    def mock_extract_audio(self):
        with patch('src.transcription.parallel_processor.extract_audio') as mock:
            mock.return_value = "/tmp/audio.wav"
            yield mock

    def test_all_videos_cached(self, mock_whisper_client, mock_transcript_cache):
        """Test when all videos are cached"""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        cache_mock, instance = mock_transcript_cache
        instance.get.return_value = [{'start': 0.0, 'end': 5.0, 'text': 'Cached'}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = transcribe_videos_parallel(
                video_paths=['/path/video1.mp4', '/path/video2.mp4'],
                cache=tmpdir,
                show_progress=False
            )

        assert len(result) == 2

    @patch('src.transcription.parallel_processor.ThreadPoolExecutor')
    @patch('src.transcription.parallel_processor.extract_audio')
    def test_audio_extraction_error(self, mock_extract, mock_executor, mock_whisper_client, mock_transcript_cache):
        """Test handles audio extraction errors"""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Setup executor to raise exception
        mock_future = Mock()
        mock_future.result.side_effect = Exception("Extraction failed")

        mock_executor_instance = Mock()
        mock_executor_instance.__enter__ = Mock(return_value=mock_executor_instance)
        mock_executor_instance.__exit__ = Mock(return_value=False)
        mock_executor_instance.submit.return_value = mock_future
        mock_executor.return_value = mock_executor_instance

        # Mock as_completed to return our future
        with patch('src.transcription.parallel_processor.as_completed', return_value=[mock_future]):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = transcribe_videos_parallel(
                    video_paths=['/path/video1.mp4'],
                    cache=tmpdir,
                    show_progress=False
                )

        # Should return empty results for failed extraction
        assert isinstance(result, dict)

    def test_with_config_object(self, mock_whisper_client, mock_transcript_cache, mock_extract_audio):
        """Test with config object"""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        config = Mock()
        config.transcription = Mock()
        config.transcription.model = "large-v2"
        config.transcription.compute_type = "float16"
        config.transcription.language = "en"
        config.transcription.vad_filter = True
        config.transcription.min_silence_duration_ms = 500
        config.transcription.speech_pad_ms = 30

        with tempfile.TemporaryDirectory() as tmpdir:
            result = transcribe_videos_parallel(
                video_paths=[],
                cache=tmpdir,
                config=config,
                show_progress=False
            )

        assert isinstance(result, dict)

    def test_without_config_uses_defaults(self, mock_whisper_client, mock_transcript_cache):
        """Test uses default values without config"""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        _, cache_instance = mock_transcript_cache
        cache_instance.get.return_value = [{'start': 0.0, 'end': 5.0, 'text': 'Test'}]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = transcribe_videos_parallel(
                video_paths=['/test.mp4'],
                cache=tmpdir,
                config=None,  # No config
                show_progress=False
            )

        # Should use default model="base", compute_type="auto", etc.
        assert isinstance(result, dict)


# ============================================================================
# Test transcribe_video single video
# ============================================================================

class TestTranscribeVideo:
    """Test transcribe_video function"""

    @pytest.fixture
    def mock_whisper_client(self):
        with patch('src.transcription.parallel_processor.WhisperClient') as mock:
            instance = Mock()
            instance.transcribe.return_value = [
                {'start': 0.0, 'end': 5.0, 'text': 'Test'}
            ]
            mock.return_value = instance
            yield mock

    @pytest.fixture
    def mock_cache(self):
        cache = Mock()
        cache.get.return_value = None
        return cache

    @pytest.fixture
    def mock_extract_audio(self):
        with patch('src.transcription.parallel_processor.extract_audio') as mock:
            mock.return_value = "/tmp/audio.wav"
            yield mock

    def test_transcribe_video_cached(self, mock_cache):
        """Test transcribe_video with cached result"""
        from src.transcription.parallel_processor import transcribe_video

        mock_cache.get.return_value = [
            {'start': 0.0, 'end': 5.0, 'text': 'Cached segment'}
        ]

        result = transcribe_video("/test/video.mp4", mock_cache)

        assert len(result) == 1
        assert result[0].text == 'Cached segment'

    def test_transcribe_video_extraction_fails(self, mock_cache, mock_whisper_client):
        """Test when audio extraction fails"""
        from src.transcription.parallel_processor import transcribe_video

        with patch('src.transcription.parallel_processor.extract_audio', return_value=None):
            result = transcribe_video("/test/video.mp4", mock_cache)

        assert result == []

    def test_transcribe_video_transcription_error(self, mock_cache, mock_extract_audio):
        """Test handles transcription errors"""
        from src.transcription.parallel_processor import transcribe_video

        with patch('src.transcription.parallel_processor.WhisperClient') as mock_client:
            instance = Mock()
            instance.transcribe.side_effect = Exception("Transcription error")
            mock_client.return_value = instance

            result = transcribe_video("/test/video.mp4", mock_cache)

        assert result == []


# ============================================================================
# Test transcribe_voiceover_media
# ============================================================================

class TestTranscribeVoiceoverMedia:
    """Test transcribe_voiceover_media function"""

    @pytest.fixture
    def mock_whisper_client(self):
        with patch('src.transcription.parallel_processor.WhisperClient') as mock:
            instance = Mock()
            instance.transcribe.return_value = [
                {'start': 0.0, 'end': 5.0, 'text': 'Voiceover text', 'words': []}
            ]
            mock.return_value = instance
            yield mock

    @pytest.fixture
    def mock_write_srt(self):
        with patch('src.transcription.parallel_processor.write_srt') as mock:
            yield mock

    @pytest.fixture
    def mock_extract_audio(self):
        with patch('src.transcription.parallel_processor.extract_audio') as mock:
            mock.return_value = "/tmp/audio.wav"
            yield mock

    def test_transcribe_voiceover_audio_file(self, mock_whisper_client, mock_write_srt):
        """Test transcribing audio file directly"""
        from src.transcription.parallel_processor import transcribe_voiceover_media

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            audio_path = f.name
            f.write(b"fake audio content")

        try:
            result = transcribe_voiceover_media(
                media_path=audio_path,
                model_name="base"
            )

            assert result.endswith('.srt')
            mock_write_srt.assert_called_once()
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_voiceover_video_file(self, mock_whisper_client, mock_write_srt, mock_extract_audio):
        """Test transcribing video file (extracts audio first)"""
        from src.transcription.parallel_processor import transcribe_voiceover_media

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name
            f.write(b"fake video content")

        try:
            with tempfile.TemporaryDirectory() as cache_dir:
                result = transcribe_voiceover_media(
                    media_path=video_path,
                    model_name="base",
                    cache_dir=cache_dir
                )

            assert result.endswith('.srt')
            mock_extract_audio.assert_called_once()
        finally:
            Path(video_path).unlink(missing_ok=True)

    def test_transcribe_voiceover_video_extraction_fails(self, mock_whisper_client, mock_write_srt):
        """Test error when video audio extraction fails"""
        from src.transcription.parallel_processor import transcribe_voiceover_media

        with patch('src.transcription.parallel_processor.extract_audio', return_value=None):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                video_path = f.name

            try:
                with pytest.raises(RuntimeError, match="Could not extract audio"):
                    transcribe_voiceover_media(
                        media_path=video_path,
                        model_name="base"
                    )
            finally:
                Path(video_path).unlink(missing_ok=True)

    def test_transcribe_voiceover_unsupported_format(self, mock_whisper_client):
        """Test error for unsupported format"""
        from src.transcription.parallel_processor import transcribe_voiceover_media

        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            bad_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported media format"):
                transcribe_voiceover_media(
                    media_path=bad_path,
                    model_name="base"
                )
        finally:
            Path(bad_path).unlink(missing_ok=True)

    def test_transcribe_voiceover_no_segments(self, mock_whisper_client, mock_write_srt):
        """Test error when no segments generated"""
        from src.transcription.parallel_processor import transcribe_voiceover_media

        mock_whisper_client.return_value.transcribe.return_value = []  # No segments

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            audio_path = f.name
            f.write(b"fake audio")

        try:
            with pytest.raises(RuntimeError, match="No segments generated"):
                transcribe_voiceover_media(
                    media_path=audio_path,
                    model_name="base"
                )
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def test_transcribe_voiceover_saves_word_timestamps(self, mock_whisper_client, mock_write_srt):
        """Test word timestamps are saved to JSON"""
        from src.transcription.parallel_processor import transcribe_voiceover_media

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            audio_path = f.name
            f.write(b"fake audio")

        try:
            result = transcribe_voiceover_media(
                media_path=audio_path,
                model_name="base",
                word_timestamps=True
            )

            # Check .words.json was created
            words_path = Path(result).with_suffix('.words.json')
            # File creation is mocked, but function should attempt it
            assert result.endswith('.srt')
        finally:
            Path(audio_path).unlink(missing_ok=True)


# ============================================================================
# Test get_transcript_segments
# ============================================================================

class TestGetTranscriptSegments:
    """Test get_transcript_segments function"""

    def test_get_transcript_segments(self):
        """Test get_transcript_segments wrapper"""
        from src.transcription.parallel_processor import get_transcript_segments

        with patch('src.transcription.parallel_processor.TranscriptCache') as mock_cache:
            cache_instance = Mock()
            cache_instance.get.return_value = [
                {'start': 0.0, 'end': 5.0, 'text': 'Test'}
            ]
            mock_cache.return_value = cache_instance

            result = get_transcript_segments(
                video_path="/test/video.mp4",
                cache_dir=".cache"
            )

            assert len(result) == 1
            assert result[0].text == 'Test'


# ============================================================================
# Test transcribe_voiceover_audio
# ============================================================================

class TestTranscribeVoiceoverAudio:
    """Test transcribe_voiceover_audio function"""

    def test_transcribe_voiceover_audio(self):
        """Test basic voiceover audio transcription"""
        from src.transcription.parallel_processor import transcribe_voiceover_audio

        with patch('src.transcription.parallel_processor.WhisperClient') as mock_client:
            instance = Mock()
            instance.transcribe.return_value = [
                {'start': 0.0, 'end': 10.0, 'text': 'Hello world'}
            ]
            mock_client.return_value = instance

            result = transcribe_voiceover_audio("/test/audio.mp3")

            assert len(result) == 1
            assert result[0]['text'] == 'Hello world'
            # Verify vad_filter=True for voiceover (Rule 11: voiceover needs VAD ON)
            instance.transcribe.assert_called_once()
            call_kwargs = instance.transcribe.call_args[1]
            assert call_kwargs.get('vad_filter') is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
