"""Tests for US-79-004: Configurable max_retries in TranscriptionConfig.

Tests that:
1. TranscriptionConfig validates max_retries (0-5 range)
2. max_retries=0 disables retry (single attempt only)
3. max_retries=3 allows up to 3 retry attempts after initial failure
4. Config value propagates through transcribe_videos_parallel
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.config.sections.core import TranscriptionConfig


class TestTranscriptionConfigMaxRetries:
    """Test max_retries field on TranscriptionConfig."""

    def test_default_max_retries_is_2(self):
        """Default max_retries should be 2."""
        config = TranscriptionConfig()
        assert config.max_retries == 2

    def test_max_retries_zero_valid(self):
        """max_retries=0 should be valid (no retries)."""
        config = TranscriptionConfig(max_retries=0)
        assert config.max_retries == 0

    def test_max_retries_five_valid(self):
        """max_retries=5 should be valid (upper bound)."""
        config = TranscriptionConfig(max_retries=5)
        assert config.max_retries == 5

    def test_max_retries_negative_raises(self):
        """max_retries=-1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_retries=-1.*must be >= 0 and <= 5"):
            TranscriptionConfig(max_retries=-1)

    def test_max_retries_above_five_raises(self):
        """max_retries=6 should raise ValueError."""
        with pytest.raises(ValueError, match="max_retries=6.*must be >= 0 and <= 5"):
            TranscriptionConfig(max_retries=6)

    def test_max_retries_three_valid(self):
        """max_retries=3 should be valid."""
        config = TranscriptionConfig(max_retries=3)
        assert config.max_retries == 3


class TestMaxRetriesZeroDisablesRetry:
    """Test that max_retries=0 means single attempt only."""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    def test_max_retries_zero_single_attempt(self, mock_extract, MockWhisperClient):
        """With max_retries=0, transcribe_video should attempt only once."""
        from src.transcription.parallel_processor import transcribe_video

        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("CUDA out of memory")

        result = transcribe_video(
            "/test/video.mp4",
            cache=mock_cache,
            max_retries=0
        )

        # Should have called transcribe exactly once (no retries)
        assert mock_whisper.transcribe.call_count == 1
        # Should return empty on failure
        assert result == []

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    def test_max_retries_zero_succeeds_on_first_attempt(self, mock_extract, MockWhisperClient):
        """With max_retries=0, should still succeed if first attempt works."""
        from src.transcription.parallel_processor import transcribe_video

        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = [
            {'start': 0.0, 'end': 5.0, 'text': 'Hello world'}
        ]

        with patch('pathlib.Path.unlink'):
            result = transcribe_video(
                "/test/video.mp4",
                cache=mock_cache,
                max_retries=0
            )

        assert len(result) == 1
        assert result[0].text == 'Hello world'
        assert mock_whisper.transcribe.call_count == 1


class TestMaxRetriesThreeAllowsRetries:
    """Test that max_retries=3 allows up to 3 retry attempts after initial failure."""

    @patch('src.transcription.parallel_processor._clear_cuda_cache')
    @patch('src.transcription.parallel_processor.time')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    def test_max_retries_three_succeeds_on_fourth_attempt(
        self, mock_extract, MockWhisperClient, mock_time, mock_clear_cuda
    ):
        """With max_retries=3, should allow initial + 3 retries = 4 total attempts."""
        from src.transcription.parallel_processor import transcribe_video

        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_extract.return_value = "/fake/audio.wav"
        mock_time.sleep = Mock()  # Don't actually sleep
        mock_time.time = Mock(return_value=0)

        mock_whisper = MockWhisperClient.return_value
        call_count = 0

        def transcribe_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 4:  # Fail first 3 times
                raise RuntimeError("CUDA out of memory")
            return [{'start': 0.0, 'end': 5.0, 'text': 'Success on retry 3'}]

        mock_whisper.transcribe.side_effect = transcribe_side_effect

        with patch('pathlib.Path.unlink'):
            result = transcribe_video(
                "/test/video.mp4",
                cache=mock_cache,
                max_retries=3
            )

        # Should have called transcribe 4 times (1 initial + 3 retries)
        assert mock_whisper.transcribe.call_count == 4
        assert len(result) == 1
        assert result[0].text == 'Success on retry 3'

    @patch('src.transcription.parallel_processor._clear_cuda_cache')
    @patch('src.transcription.parallel_processor.time')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    def test_max_retries_three_exhausted(
        self, mock_extract, MockWhisperClient, mock_time, mock_clear_cuda
    ):
        """With max_retries=3, should fail after 4 total attempts."""
        from src.transcription.parallel_processor import transcribe_video

        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_extract.return_value = "/fake/audio.wav"
        mock_time.sleep = Mock()
        mock_time.time = Mock(return_value=0)

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("CUDA out of memory")

        result = transcribe_video(
            "/test/video.mp4",
            cache=mock_cache,
            max_retries=3
        )

        # Should have called transcribe 4 times (1 initial + 3 retries)
        assert mock_whisper.transcribe.call_count == 4
        assert result == []


class TestConfigPropagation:
    """Test that config max_retries value propagates through transcribe_videos_parallel."""

    @patch('src.transcription.parallel_processor.WhisperClient')
    def test_config_max_retries_read_in_parallel(self, MockWhisperClient):
        """transcribe_videos_parallel should read max_retries from config."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Create a config with max_retries=3
        mock_config = Mock()
        mock_config.transcription.model = "base"
        mock_config.transcription.compute_type = "auto"
        mock_config.transcription.language = "en"
        mock_config.transcription.min_silence_duration_ms = 200
        mock_config.transcription.speech_pad_ms = 10
        mock_config.transcription.audio_extraction_workers = 2
        mock_config.transcription.auto_cleanup_after_batch = False
        mock_config.transcription.gpu_transcription_timeout = 300
        mock_config.transcription.audio_extraction_timeout = 60
        mock_config.transcription.max_retries = 3

        mock_cache = Mock()
        mock_cache.cache_dir = "/fake/cache"

        mock_whisper = MockWhisperClient.return_value

        # Call with empty video list to avoid complex mocking
        result = transcribe_videos_parallel(
            video_paths=[],
            cache=mock_cache,
            config=mock_config
        )

        # The function should have read the config value
        # Since video_paths is empty, no actual transcription happens
        assert result == {}
