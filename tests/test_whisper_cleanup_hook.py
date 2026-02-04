"""Tests for US-60-011: WhisperModel cleanup hook after transcription batch."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace


class TestAutoCleanupAfterBatch:
    """Test automatic cleanup of WhisperModel after batch transcription."""

    def test_cleanup_called_when_enabled(self):
        """Verify cleanup is called at end of transcribe_videos_parallel when enabled."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Mock config with auto_cleanup_after_batch=True
        config = SimpleNamespace(
            transcription=SimpleNamespace(
                model='base',
                compute_type='auto',
                language='en',
                min_silence_duration_ms=200,
                speech_pad_ms=10,
                audio_extraction_workers=2,
                auto_cleanup_after_batch=True
            )
        )

        # Mock the WhisperClient and TranscriptCache
        mock_whisper_client = MagicMock()
        mock_transcript_cache = MagicMock()
        mock_transcript_cache.get.return_value = [{'start': 0.0, 'end': 1.0, 'text': 'test'}]

        with patch('src.transcription.parallel_processor.WhisperClient', return_value=mock_whisper_client), \
             patch('src.transcription.parallel_processor.TranscriptCache', return_value=mock_transcript_cache):

            # Call with single video that's cached (quick path)
            results = transcribe_videos_parallel(
                video_paths=['/path/to/video.mp4'],
                cache='/tmp/cache',
                config=config,
                show_progress=False
            )

            # Verify cleanup was called
            mock_whisper_client.cleanup.assert_called_once()

    def test_cleanup_not_called_when_disabled(self):
        """Verify cleanup is NOT called when auto_cleanup_after_batch=False."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Mock config with auto_cleanup_after_batch=False
        config = SimpleNamespace(
            transcription=SimpleNamespace(
                model='base',
                compute_type='auto',
                language='en',
                min_silence_duration_ms=200,
                speech_pad_ms=10,
                audio_extraction_workers=2,
                auto_cleanup_after_batch=False
            )
        )

        # Mock the WhisperClient and TranscriptCache
        mock_whisper_client = MagicMock()
        mock_transcript_cache = MagicMock()
        mock_transcript_cache.get.return_value = [{'start': 0.0, 'end': 1.0, 'text': 'test'}]

        with patch('src.transcription.parallel_processor.WhisperClient', return_value=mock_whisper_client), \
             patch('src.transcription.parallel_processor.TranscriptCache', return_value=mock_transcript_cache):

            # Call with single video that's cached (quick path)
            results = transcribe_videos_parallel(
                video_paths=['/path/to/video.mp4'],
                cache='/tmp/cache',
                config=config,
                show_progress=False
            )

            # Verify cleanup was NOT called
            mock_whisper_client.cleanup.assert_not_called()

    def test_cleanup_called_even_on_exception(self):
        """Verify cleanup is called even when batch processing raises an exception."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Mock config with auto_cleanup_after_batch=True
        config = SimpleNamespace(
            transcription=SimpleNamespace(
                model='base',
                compute_type='auto',
                language='en',
                min_silence_duration_ms=200,
                speech_pad_ms=10,
                audio_extraction_workers=2,
                auto_cleanup_after_batch=True
            )
        )

        # Mock the WhisperClient and TranscriptCache
        mock_whisper_client = MagicMock()
        mock_transcript_cache = MagicMock()
        # Raise exception when checking cache to trigger error in main processing
        mock_transcript_cache.get.side_effect = RuntimeError("Simulated cache failure")

        with patch('src.transcription.parallel_processor.WhisperClient', return_value=mock_whisper_client), \
             patch('src.transcription.parallel_processor.TranscriptCache', return_value=mock_transcript_cache):

            # Call should raise the exception
            with pytest.raises(RuntimeError, match="Simulated cache failure"):
                transcribe_videos_parallel(
                    video_paths=['/path/to/video.mp4'],
                    cache='/tmp/cache',
                    config=config,
                    show_progress=False
                )

            # Verify cleanup was STILL called (finally block)
            mock_whisper_client.cleanup.assert_called_once()

    def test_cleanup_default_enabled_without_config(self):
        """Verify cleanup is enabled by default when no config provided."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Mock the WhisperClient and TranscriptCache
        mock_whisper_client = MagicMock()
        mock_transcript_cache = MagicMock()
        mock_transcript_cache.get.return_value = [{'start': 0.0, 'end': 1.0, 'text': 'test'}]

        with patch('src.transcription.parallel_processor.WhisperClient', return_value=mock_whisper_client), \
             patch('src.transcription.parallel_processor.TranscriptCache', return_value=mock_transcript_cache):

            # Call WITHOUT config - should use default (auto_cleanup=True)
            results = transcribe_videos_parallel(
                video_paths=['/path/to/video.mp4'],
                cache='/tmp/cache',
                config=None,  # No config
                show_progress=False
            )

            # Verify cleanup was called (default enabled)
            mock_whisper_client.cleanup.assert_called_once()


class TestCleanupLogsGpuMemory:
    """Test that cleanup logs GPU memory reclaimed."""

    def test_cleanup_logs_memory_reclaimed(self, caplog):
        """Verify cleanup logs GPU memory before/after and delta."""
        import logging
        from src.transcription.whisper_client import WhisperClient, _gpu_lock, cleanup_model

        # Set up logging
        caplog.set_level(logging.INFO, logger='src.transcription.whisper_client')

        # Create a mock shared model
        mock_model = MagicMock()

        with patch('src.transcription.whisper_client._shared_model', mock_model), \
             patch('src.transcription.whisper_client._model_config', {'model': 'base'}), \
             patch('src.transcription.whisper_client._get_gpu_memory_mb') as mock_mem:
            # Simulate memory reduction: 1000MB before -> 200MB after = 800MB freed
            mock_mem.side_effect = [
                (1000.0, 1200.0),  # Before cleanup
                (200.0, 300.0),    # After cleanup
            ]

            # Call cleanup
            client = WhisperClient()
            client.cleanup()

            # Check that memory logging occurred
            assert "GPU memory before cleanup" in caplog.text
            assert "GPU memory after cleanup" in caplog.text
            assert "GPU memory freed by cleanup" in caplog.text


class TestTranscriptionConfigCleanupOption:
    """Test auto_cleanup_after_batch config option in TranscriptionConfig."""

    def test_config_default_value(self):
        """Verify auto_cleanup_after_batch defaults to True."""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig()
        assert config.auto_cleanup_after_batch is True

    def test_config_can_be_set_false(self):
        """Verify auto_cleanup_after_batch can be set to False."""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig(auto_cleanup_after_batch=False)
        assert config.auto_cleanup_after_batch is False
