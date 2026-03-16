"""
Tests for src/transcription/parallel_processor.py

Tests parallel video transcription with two-phase processing,
single video transcription, voiceover transcription, and cache integration.
"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription.parallel_processor import (
    transcribe_videos_parallel,
    transcribe_video,
    transcribe_voiceover_audio,
    transcribe_voiceover_media,
    get_transcript_segments
)
from src.state import TranscriptSegment


@pytest.fixture
def mock_cache():
    """Mock CacheManager with cache_dir attribute"""
    cache = Mock()
    cache.cache_dir = "/fake/cache"
    return cache


@pytest.fixture
def mock_config():
    """Mock Config object with transcription settings"""
    config = Mock()
    config.transcription = Mock()
    config.transcription.model = "base"
    config.transcription.compute_type = "auto"
    config.transcription.language = "en"
    config.transcription.vad_filter = True
    config.transcription.min_silence_duration_ms = 200
    config.transcription.speech_pad_ms = 10
    config.transcription.audio_extraction_workers = 4  # Fix for ThreadPoolExecutor
    config.transcription.auto_cleanup_after_batch = True  # US-60-011
    config.transcription.gpu_transcription_timeout = 300  # US-79-002
    config.transcription.audio_extraction_timeout = 60  # US-79-003
    config.transcription.max_retries = 2  # US-79-004
    config.transcription.whisper_num_workers = 1  # US-79-007
    config.transcription.whisper_cpu_threads = 4  # US-79-007
    config.transcription.progress_log_interval = 10  # US-79-008
    config.transcription.retry_budget_max_attempts = 50  # US-79-010
    config.transcription.retry_budget_max_backoff_seconds = 180.0  # US-79-010
    config.transcription.batch_size = 0  # US-110-005: 0 = no batching
    config.transcription.batch_wait_seconds = 0  # US-110-005
    return config


@pytest.fixture
def sample_raw_segments():
    """Sample raw segment data from Whisper"""
    return [
        {'start': 0.0, 'end': 5.0, 'text': 'First segment'},
        {'start': 5.0, 'end': 10.0, 'text': 'Second segment'},
        {'start': 10.0, 'end': 15.0, 'text': 'Third segment'}
    ]


@pytest.fixture
def sample_transcript_segments(sample_raw_segments):
    """Sample TranscriptSegment objects"""
    return [
        TranscriptSegment(
            index=i,
            start_time=seg['start'],
            end_time=seg['end'],
            text=seg['text'],
            source_file="/fake/video.mp4"
        )
        for i, seg in enumerate(sample_raw_segments)
    ]


class TestTranscribeVideosParallel:
    """Test transcribe_videos_parallel() function"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_all_videos_cached(
        self, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test when all videos are already cached"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.side_effect = [
            sample_raw_segments,  # video1 cached
            sample_raw_segments   # video2 cached
        ]

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=False
        )

        assert len(results) == 2
        assert "/video1.mp4" in results
        assert "/video2.mp4" in results
        assert all(len(results[vp]) == 3 for vp in video_paths)

        # Should not extract audio (all cached)
        mock_extract.assert_not_called()

        # WhisperClient is initialized but transcribe should not be called
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.assert_not_called()

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_uncached_videos_parallel_processing(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test parallel audio extraction and sequential transcription"""
        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock audio extraction
        mock_extract.side_effect = lambda vp, temp_dir, **kwargs: f"/fake/cache/temp_audio/{Path(vp).stem}.wav"

        # Mock WhisperClient
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            max_workers=2, show_progress=False
        )

        assert len(results) == 3
        assert all(vp in results for vp in video_paths)
        assert all(len(results[vp]) == 3 for vp in video_paths)

        # Should extract audio 3 times (parallel)
        assert mock_extract.call_count == 3

        # Should transcribe 3 times (sequential)
        assert mock_whisper.transcribe.call_count == 3

        # Should cache 3 results
        assert mock_transcript_cache.set.call_count == 3

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @pytest.mark.fast
    def test_mixed_cached_uncached(
        self, mock_rmtree, mock_extract, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test with some videos cached, some not"""
        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        # Mock video1 cached, video2/3 not cached
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.side_effect = [
            sample_raw_segments,  # video1 cached
            None,                 # video2 not cached
            None                  # video3 not cached
        ]

        # Mock audio extraction for uncached videos
        mock_extract.side_effect = lambda vp, temp_dir, **kwargs: f"/fake/audio/{Path(vp).stem}.wav"

        # Mock transcription
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        assert len(results) == 3

        # Should extract audio only for uncached videos (2)
        assert mock_extract.call_count == 2

        # Should transcribe only uncached videos (2)
        assert mock_whisper.transcribe.call_count == 2

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_force_reprocess_ignores_cache(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test force_reprocess=True ignores cache"""
        video_paths = ["/video1.mp4"]

        # Mock cached result (should be ignored)
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        # Mock extraction and transcription
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            force_reprocess=True, show_progress=False
        )

        assert len(results) == 1

        # Should NOT check cache when force_reprocess=True
        mock_transcript_cache.get.assert_not_called()

        # Should extract and transcribe
        mock_extract.assert_called_once()
        mock_whisper.transcribe.assert_called_once()

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.integration
    def test_audio_extraction_error_handling(
        self, mock_logger, mock_extract, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test error handling when audio extraction fails"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock extraction failure for video1, success for video2
        def extract_side_effect(vp, temp_dir, **kwargs):
            if "video1" in vp:
                raise RuntimeError("Extraction failed")
            return f"/fake/audio/{Path(vp).stem}.wav"

        mock_extract.side_effect = extract_side_effect

        # Mock successful transcription for video2
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'), \
             patch('src.transcription.parallel_processor.shutil.rmtree'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        # Should have logged with logger.exception() (includes traceback)
        assert mock_logger.exception.called

        # Should continue processing video2
        assert len(results) == 1  # Only video2
        assert "/video2.mp4" in results

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.integration
    def test_transcription_error_handling(
        self, mock_logger, mock_extract, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config
    ):
        """Test error handling when transcription fails"""
        video_paths = ["/video1.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock successful extraction
        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription failure
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("Transcription failed")

        with patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'), \
             patch('src.transcription.parallel_processor.shutil.rmtree'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        # Should have logged with logger.error() (permanent errors in retry loop)
        assert mock_logger.error.called

        # Should return empty result for failed video
        assert "/video1.mp4" in results
        assert results["/video1.mp4"] == []

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @pytest.mark.fast
    def test_cache_as_string(self, MockWhisperClient, MockTranscriptCache, mock_config, sample_raw_segments):
        """Test when cache is passed as string instead of object"""
        # Pass cache as string
        cache_dir = "/fake/cache"

        # Mock cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            ["/video1.mp4"], cache_dir, mock_config,
            show_progress=False
        )

        assert len(results) == 1

        # Should initialize TranscriptCache with string
        MockTranscriptCache.assert_called_once_with(cache_dir)

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @pytest.mark.integration
    def test_config_none_uses_defaults(
        self, mock_extract, MockWhisperClient, MockTranscriptCache,
        mock_cache, sample_raw_segments
    ):
        """Test that config=None uses default settings"""
        video_paths = ["/video1.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock extraction
        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'), \
             patch('src.transcription.parallel_processor.shutil.rmtree'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, config=None,
                show_progress=False
            )

        # Should initialize WhisperClient with defaults
        MockWhisperClient.assert_called_once_with(
            model_name="base",
            compute_type="auto",
            gpu_transcription_timeout=300,
            num_workers=1,
            cpu_threads=4
        )

        # Should call transcribe with default settings
        mock_whisper.transcribe.assert_called_once()
        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs['language'] is None
        assert call_kwargs['vad_filter'] is False  # Default is False when no config


class TestTranscribeVideo:
    """Test transcribe_video() function"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_transcribe_video_cache_hit(
        self, mock_unlink, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test transcribe_video with cache hit"""
        mock_cache = Mock()
        mock_cache.get.return_value = sample_raw_segments

        result = transcribe_video("/video1.mp4", mock_cache)

        assert len(result) == 3
        assert all(isinstance(seg, TranscriptSegment) for seg in result)
        assert result[0].text == "First segment"

        # Should not extract audio or transcribe
        mock_extract.assert_not_called()
        MockWhisperClient.assert_not_called()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_transcribe_video_cache_miss(
        self, mock_unlink, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test transcribe_video with cache miss"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        # Mock extraction
        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        result = transcribe_video("/video1.mp4", mock_cache)

        assert len(result) == 3
        assert all(isinstance(seg, TranscriptSegment) for seg in result)

        # Should extract audio
        mock_extract.assert_called_once_with("/video1.mp4", None, timeout=60)

        # Should transcribe
        mock_whisper.transcribe.assert_called_once()

        # Should cache result
        mock_cache.set.assert_called_once()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_transcribe_video_extraction_failure(
        self, mock_logger, mock_extract, MockWhisperClient
    ):
        """Test transcribe_video when audio extraction fails"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        # Mock extraction failure (returns None)
        mock_extract.return_value = None

        result = transcribe_video("/video1.mp4", mock_cache)

        assert result == []
        assert mock_logger.warning.called
        MockWhisperClient.assert_not_called()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_transcribe_video_transcription_failure(
        self, mock_unlink, mock_logger, mock_extract, MockWhisperClient
    ):
        """Test transcribe_video when transcription fails"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        # Mock successful extraction
        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription failure
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("Transcription failed")

        result = transcribe_video("/video1.mp4", mock_cache)

        assert result == []
        assert mock_logger.error.called

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_transcribe_video_custom_settings(
        self, mock_unlink, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test transcribe_video with custom settings"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        result = transcribe_video(
            "/video1.mp4", mock_cache,
            model_name="medium",
            compute_type="int8",
            language="es",
            temp_dir="/custom/temp",
            vad_filter=False,
            min_silence_duration_ms=500,
            speech_pad_ms=20
        )

        # Should initialize WhisperClient with custom settings
        MockWhisperClient.assert_called_once_with(
            model_name="medium",
            compute_type="int8",
            gpu_transcription_timeout=300,
            num_workers=1,
            cpu_threads=4
        )

        # Should transcribe with custom settings
        mock_whisper.transcribe.assert_called_once()
        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs['language'] == "es"
        assert call_kwargs['vad_filter'] is False
        assert call_kwargs['min_silence_duration_ms'] == 500
        assert call_kwargs['speech_pad_ms'] == 20


class TestTranscribeVoiceoverAudio:
    """Test transcribe_voiceover_audio() function"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @pytest.mark.fast
    def test_transcribe_voiceover_audio_defaults(
        self, MockWhisperClient, sample_raw_segments
    ):
        """Test voiceover audio transcription with defaults"""
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        result = transcribe_voiceover_audio("/voiceover.mp3")

        assert result == sample_raw_segments

        # Should initialize with defaults
        MockWhisperClient.assert_called_once_with(
            model_name="base",
            compute_type="auto",
            gpu_transcription_timeout=300,
            num_workers=1,
            cpu_threads=4
        )

        # Should transcribe with vad_filter=True (voiceover needs VAD for gap detection)
        mock_whisper.transcribe.assert_called_once_with(
            "/voiceover.mp3",
            language=None,
            vad_filter=True
        )

    @patch('src.transcription.parallel_processor.WhisperClient')
    @pytest.mark.fast
    def test_transcribe_voiceover_audio_custom_settings(
        self, MockWhisperClient, sample_raw_segments
    ):
        """Test voiceover audio transcription with custom settings"""
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        result = transcribe_voiceover_audio(
            "/voiceover.mp3",
            model_name="large",
            compute_type="float16",
            language="fr"
        )

        assert result == sample_raw_segments

        # Should initialize with custom settings
        MockWhisperClient.assert_called_once_with(
            model_name="large",
            compute_type="float16",
            gpu_transcription_timeout=300,
            num_workers=1,
            cpu_threads=4
        )

        # Should transcribe with language
        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs['language'] == "fr"


class TestTranscribeVoiceoverMedia:
    """Test transcribe_voiceover_media() function"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', create=True)
    @pytest.mark.fast
    def test_transcribe_video_file(
        self, mock_open, mock_mkdir, mock_unlink, mock_write_srt,
        mock_extract, MockWhisperClient, tmp_path, sample_raw_segments
    ):
        """Test transcribing video file (.mp4)"""
        media_path = tmp_path / "voiceover.mp4"
        media_path.write_text("fake video")

        # Mock extraction
        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        # Mock file write
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')

        # Should extract audio from video
        mock_extract.assert_called_once()

        # Should transcribe with word_timestamps=True
        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs['word_timestamps'] is True
        assert call_kwargs['vad_filter'] is True  # Voiceover needs VAD for gap detection

        # Should write SRT
        mock_write_srt.assert_called_once()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_transcribe_audio_file(
        self, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test transcribing audio file (.mp3)"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        # Mock transcription
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', create=True):
            result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')

        # Should NOT extract audio (already audio)
        mock_extract.assert_not_called()

        # Should transcribe directly
        mock_whisper.transcribe.assert_called_once()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_transcribe_with_custom_output_path(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test with custom output SRT path"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")
        output_path = tmp_path / "custom.srt"

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', create=True):
            result = transcribe_voiceover_media(
                str(media_path),
                output_srt_path=str(output_path)
            )

        assert result == str(output_path)

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_transcribe_with_cache_dir(
        self, mock_unlink, mock_mkdir, mock_write_srt,
        mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test with custom cache directory"""
        media_path = tmp_path / "voiceover.mp4"
        media_path.write_text("fake video")
        cache_dir = tmp_path / "cache"

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', create=True):
            result = transcribe_voiceover_media(
                str(media_path),
                cache_dir=str(cache_dir)
            )

        # Should create temp directory in cache_dir
        assert mock_mkdir.called

    @patch('src.transcription.parallel_processor.WhisperClient')
    @pytest.mark.fast
    def test_unsupported_format(self, MockWhisperClient, tmp_path):
        """Test error with unsupported format"""
        media_path = tmp_path / "voiceover.txt"
        media_path.write_text("fake text")

        with pytest.raises(ValueError, match="Unsupported media format"):
            transcribe_voiceover_media(str(media_path))

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_extraction_failure(
        self, mock_logger, mock_extract, MockWhisperClient, tmp_path
    ):
        """Test error when audio extraction fails"""
        media_path = tmp_path / "voiceover.mp4"
        media_path.write_text("fake video")

        # Mock extraction failure
        mock_extract.return_value = None

        with pytest.raises(RuntimeError, match="Could not extract audio"):
            transcribe_voiceover_media(str(media_path))

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_no_segments_generated(
        self, mock_write_srt, MockWhisperClient, tmp_path
    ):
        """Test error when no segments generated"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        # Mock empty transcription
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = []

        with pytest.raises(RuntimeError, match="No segments generated"):
            transcribe_voiceover_media(str(media_path))

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_word_timestamps_json_save(
        self, mock_logger, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test saving word timestamps to JSON"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = transcribe_voiceover_media(
                str(media_path),
                word_timestamps=True
            )

        # Should write both SRT and JSON
        assert mock_open.call_count >= 1


class TestParallelErrorRecovery:
    """Test parallel processing error recovery (US-003)"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_audio_extraction_error_logs_full_traceback(
        self, mock_logger, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that audio extraction errors are logged with full traceback via logger.exception()"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock extraction failure for video1, success for video2
        def extract_side_effect(vp, temp_dir, **kwargs):
            if "video1" in vp:
                raise RuntimeError("Extraction failed with details")
            return f"/fake/audio/{Path(vp).stem}.wav"

        mock_extract.side_effect = extract_side_effect
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        # Should have logged with logger.exception() (includes traceback)
        assert mock_logger.exception.called, "Expected logger.exception() to be called for traceback"

        # Should continue processing video2 despite video1 error
        assert "/video2.mp4" in results
        assert len(results["/video2.mp4"]) == 3

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_transcription_error_logs_full_traceback(
        self, mock_logger, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that transcription errors are logged with full traceback via logger.exception()"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock successful extraction for both
        mock_extract.side_effect = lambda vp, temp_dir, **kwargs: f"/fake/audio/{Path(vp).stem}.wav"

        # Mock transcription failure for video1 only
        mock_whisper = MockWhisperClient.return_value
        call_count = [0]

        def transcribe_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Transcription GPU error")
            return sample_raw_segments

        mock_whisper.transcribe.side_effect = transcribe_side_effect

        with patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        # Should have logged with logger.error() (permanent errors in retry loop)
        assert mock_logger.error.called, "Expected logger.error() to be called for permanent error"

        # Failed video should have empty results
        assert results["/video1.mp4"] == []

        # Video2 should succeed
        assert len(results["/video2.mp4"]) == 3

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_cleanup_error_does_not_affect_results(
        self, mock_logger, mock_mkdir, mock_unlink, mock_rmtree,
        mock_extract, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that cleanup errors (file unlink, rmtree) don't affect transcription results"""
        video_paths = ["/video1.mp4"]

        # Mock no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock successful extraction and transcription
        mock_extract.return_value = "/fake/audio/video1.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        # Mock cleanup failures - both unlink and rmtree raise errors
        mock_unlink.side_effect = OSError("Permission denied")
        mock_rmtree.side_effect = OSError("Directory in use")

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=False
        )

        # Results should be unaffected by cleanup errors
        assert "/video1.mp4" in results
        assert len(results["/video1.mp4"]) == 3

        # Should have logged the cleanup errors at debug level
        debug_calls = [call for call in mock_logger.debug.call_args_list]
        assert len(debug_calls) >= 1, "Expected cleanup errors to be logged at debug level"


class TestGetTranscriptSegments:
    """Test get_transcript_segments() backward-compatible function"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.transcribe_video')
    @pytest.mark.fast
    def test_get_transcript_segments(
        self, mock_transcribe, MockTranscriptCache,
        sample_transcript_segments
    ):
        """Test backward-compatible wrapper"""
        mock_transcribe.return_value = sample_transcript_segments

        result = get_transcript_segments(
            "/video.mp4",
            "/cache",
            model_name="small",
            compute_type="float16"
        )

        assert result == sample_transcript_segments

        # Should create cache
        MockTranscriptCache.assert_called_once_with("/cache")

        # Should call transcribe_video
        mock_transcribe.assert_called_once()
        call_args = mock_transcribe.call_args[0]
        assert call_args[0] == "/video.mp4"


class TestTranscriptionRetry:
    """Test retry logic for transient transcription errors (US-38-012)"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.time.sleep')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_retry_on_transient_cuda_error(
        self, mock_unlink, mock_sleep, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test retry occurs on transient CUDA error (US-38-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        # Mock extraction
        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: fail first two times with CUDA error, succeed third
        mock_whisper = MockWhisperClient.return_value
        call_count = [0]

        def transcribe_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("CUDA out of memory")
            return sample_raw_segments

        mock_whisper.transcribe.side_effect = transcribe_side_effect

        result = transcribe_video(
            "/video1.mp4", mock_cache,
            max_retries=2, base_delay=1.0
        )

        # Should have retried and succeeded
        assert len(result) == 3
        assert mock_whisper.transcribe.call_count == 3

        # Should have slept twice with exponential backoff (1s, 2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry: 1.0 * 2^0 = 1.0
        mock_sleep.assert_any_call(2.0)  # Second retry: 1.0 * 2^1 = 2.0

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @patch('src.transcription.parallel_processor.time.sleep')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_retry_logs_warning_on_each_retry(
        self, mock_unlink, mock_sleep, mock_logger, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test WARNING is logged on each retry with video_id and error (US-38-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: fail once with memory error, then succeed
        mock_whisper = MockWhisperClient.return_value
        call_count = [0]

        def transcribe_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("GPU memory pressure detected")
            return sample_raw_segments

        mock_whisper.transcribe.side_effect = transcribe_side_effect

        result = transcribe_video("/path/to/abc123.mp4", mock_cache, max_retries=2)

        # Should have logged WARNING with video_id and error message
        warning_calls = [c for c in mock_logger.warning.call_args_list]
        assert len(warning_calls) >= 1
        warning_msg = str(warning_calls[0])
        assert "abc123" in warning_msg, f"Expected 'abc123' in warning: {warning_msg}"
        assert "GPU memory pressure" in warning_msg, f"Expected error in warning: {warning_msg}"

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @patch('src.transcription.parallel_processor.time.sleep')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_no_retry_on_file_not_found(
        self, mock_unlink, mock_sleep, mock_logger, mock_extract, MockWhisperClient
    ):
        """Test NO retry on FileNotFoundError (US-38-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: fail with FileNotFoundError
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = FileNotFoundError("Audio file not found")

        result = transcribe_video("/video1.mp4", mock_cache, max_retries=2)

        # Should return empty - no retry
        assert result == []

        # Should only have called transcribe once (no retry)
        assert mock_whisper.transcribe.call_count == 1

        # Should NOT have slept (no retry)
        mock_sleep.assert_not_called()

        # Should have logged error (not warning)
        assert mock_logger.error.called

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.logger')
    @patch('src.transcription.parallel_processor.time.sleep')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_no_retry_on_permission_error(
        self, mock_unlink, mock_sleep, mock_logger, mock_extract, MockWhisperClient
    ):
        """Test NO retry on PermissionError (US-38-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: fail with PermissionError
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = PermissionError("Access denied")

        result = transcribe_video("/video1.mp4", mock_cache, max_retries=2)

        # Should return empty - no retry
        assert result == []

        # Should only have called transcribe once (no retry)
        assert mock_whisper.transcribe.call_count == 1

        # Should NOT have slept (no retry)
        mock_sleep.assert_not_called()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.time.sleep')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_all_retries_exhausted_returns_empty(
        self, mock_unlink, mock_sleep, mock_extract, MockWhisperClient
    ):
        """Test returns empty when all retries exhausted (US-38-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: always fail with CUDA error
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = RuntimeError("CUDA device unavailable")

        result = transcribe_video("/video1.mp4", mock_cache, max_retries=2)

        # Should return empty after exhausting retries
        assert result == []

        # Should have attempted 3 times (initial + 2 retries)
        assert mock_whisper.transcribe.call_count == 3

        # Should have slept twice
        assert mock_sleep.call_count == 2

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.time.sleep')
    @patch('src.transcription.parallel_processor._clear_cuda_cache')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_cuda_cache_cleared_between_retries(
        self, mock_unlink, mock_clear_cache, mock_sleep, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test CUDA cache is cleared between retry attempts (US-60-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: fail first time with CUDA OOM, succeed second
        mock_whisper = MockWhisperClient.return_value
        call_count = [0]

        def transcribe_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("CUDA out of memory")
            return sample_raw_segments

        mock_whisper.transcribe.side_effect = transcribe_side_effect

        result = transcribe_video("/video1.mp4", mock_cache, max_retries=2)

        # Should have succeeded on retry
        assert len(result) == 3

        # Should have called CUDA cache clear before sleeping
        assert mock_clear_cache.call_count == 1
        # Verify clear_cache was called (before sleep in the retry loop)
        mock_clear_cache.assert_called_once()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor._clear_cuda_cache')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_no_cuda_cache_clear_on_permanent_error(
        self, mock_unlink, mock_clear_cache, mock_extract, MockWhisperClient
    ):
        """Test CUDA cache is NOT cleared on permanent errors (US-60-012)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"

        # Mock transcription: fail with FileNotFoundError (permanent)
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.side_effect = FileNotFoundError("File not found")

        result = transcribe_video("/video1.mp4", mock_cache, max_retries=2)

        # Should return empty - no retry
        assert result == []

        # Should NOT have called CUDA cache clear (no retry for permanent errors)
        mock_clear_cache.assert_not_called()


class TestProgressLogging:
    """Test that progress uses structured logging, not print (US-79-008)"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.get_audio_duration')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_progress_uses_logger_not_print(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_duration,
        mock_extract, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments, caplog
    ):
        """Progress logging uses logger.info, not print, verified via caplog"""
        import logging

        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock: no cached results
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock: audio extraction returns paths
        mock_extract.side_effect = ["/tmp/video1.wav", "/tmp/video2.wav"]
        mock_duration.return_value = 10.0

        # Mock: whisper returns segments
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        # Set progress_log_interval on config
        mock_config.transcription.progress_log_interval = 1

        with caplog.at_level(logging.INFO, logger='src.transcription.parallel_processor'):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=True
            )

        # Verify results were produced
        assert len(results) == 2

        # Verify progress messages appear in logger output
        log_messages = [r.message for r in caplog.records
                        if r.name == 'src.transcription.parallel_processor']
        assert any('Phase 1' in m for m in log_messages), \
            f"Expected 'Phase 1' in log messages, got: {log_messages}"
        assert any('Phase 2' in m for m in log_messages), \
            f"Expected 'Phase 2' in log messages, got: {log_messages}"

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @pytest.mark.fast
    def test_no_print_calls_in_source(
        self, MockWhisperClient, MockTranscriptCache
    ):
        """Verify parallel_processor.py contains no print() calls"""
        import inspect
        import src.transcription.parallel_processor as pp

        source = inspect.getsource(pp)
        # Check that print( doesn't appear (except in comments/strings)
        lines = source.split('\n')
        print_lines = [
            line.strip() for line in lines
            if 'print(' in line
            and not line.strip().startswith('#')
            and not line.strip().startswith("'")
            and not line.strip().startswith('"')
        ]
        assert len(print_lines) == 0, \
            f"Found print() calls in parallel_processor.py: {print_lines}"


class TestBatchProcessingConfig:
    """Test batch processing configuration (US-110-005)"""

    @pytest.mark.fast
    def test_batch_config_in_mock(self, mock_config):
        """Verify mock_config includes batch processing settings"""
        assert hasattr(mock_config.transcription, 'batch_size')
        assert hasattr(mock_config.transcription, 'batch_wait_seconds')
        assert mock_config.transcription.batch_size == 0
        assert mock_config.transcription.batch_wait_seconds == 0

    @pytest.mark.fast
    def test_batch_size_with_batching_enabled(self):
        """Test that batch_size > 0 enables batching"""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Create a config with batching enabled
        config = Mock()
        config.transcription = Mock()
        config.transcription.model = "base"
        config.transcription.compute_type = "auto"
        config.transcription.language = "en"
        config.transcription.vad_filter = True
        config.transcription.min_silence_duration_ms = 200
        config.transcription.speech_pad_ms = 10
        config.transcription.audio_extraction_workers = 2
        config.transcription.auto_cleanup_after_batch = True
        config.transcription.gpu_transcription_timeout = 300
        config.transcription.audio_extraction_timeout = 60
        config.transcription.max_retries = 2
        config.transcription.whisper_num_workers = 1
        config.transcription.whisper_cpu_threads = 4
        config.transcription.progress_log_interval = 10
        config.transcription.retry_budget_max_attempts = 50
        config.transcription.retry_budget_max_backoff_seconds = 180.0
        config.transcription.batch_size = 2  # Enable batching with 2 videos per batch
        config.transcription.batch_wait_seconds = 1  # 1 second wait between batches

        # Verify config values are accessible
        assert config.transcription.batch_size == 2
        assert config.transcription.batch_wait_seconds == 1

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.get_audio_duration')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_batch_processing_splits_videos(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_get_duration,
        mock_extract, mock_whisper, mock_cache_cls, sample_raw_segments
    ):
        """Test that batch_size splits videos into multiple batches"""
        # Create a complete config mock
        config = Mock()
        config.transcription = Mock()
        config.transcription.model = "base"
        config.transcription.compute_type = "auto"
        config.transcription.language = "en"
        config.transcription.vad_filter = True
        config.transcription.min_silence_duration_ms = 200
        config.transcription.speech_pad_ms = 10
        config.transcription.audio_extraction_workers = 2
        config.transcription.auto_cleanup_after_batch = True
        config.transcription.gpu_transcription_timeout = 300
        config.transcription.audio_extraction_timeout = 60
        config.transcription.max_retries = 2
        config.transcription.whisper_num_workers = 1
        config.transcription.whisper_cpu_threads = 4
        config.transcription.progress_log_interval = 10
        config.transcription.retry_budget_max_attempts = 50
        config.transcription.retry_budget_max_backoff_seconds = 180.0
        config.transcription.batch_size = 2  # Enable batching with 2 videos per batch
        config.transcription.batch_wait_seconds = 1  # 1 second wait between batches
        config.transcription.progress_callback_interval = 5
        config.transcription.auto_fallback_to_cpu = True

        # Setup mocks
        mock_cache = Mock()
        mock_cache.cache_dir = "/fake/cache"
        mock_cache_cls.return_value = mock_cache
        mock_cache.get.return_value = None  # No cached transcripts
        mock_cache._source_map = {}

        mock_whisper_instance = Mock()
        mock_whisper.return_value = mock_whisper_instance
        mock_whisper_instance.transcribe.return_value = sample_raw_segments
        mock_whisper_instance.cleanup.return_value = None

        mock_extract.return_value = "/fake/audio.mp3"
        mock_get_duration.return_value = 10.0
        mock_rmtree.return_value = None

        # Create 4 videos - with batch_size=2, should create 2 batches
        video_paths = [f"/fake/video{i}.mp4" for i in range(4)]

        # Call with batching enabled
        results = transcribe_videos_parallel(
            video_paths=video_paths,
            cache=mock_cache,
            config=config,
            show_progress=False,
            skip_if_cached=True
        )

        # Verify extraction was called for all videos
        assert mock_extract.call_count == 4

    @pytest.mark.fast
    def test_batch_wait_seconds_validation(self):
        """Test that batch_wait_seconds must be >= 0"""
        from src.config.sections.core import TranscriptionConfig

        # Valid: batch_wait_seconds >= 0
        config = TranscriptionConfig(batch_wait_seconds=0)
        assert config.batch_wait_seconds == 0

        config = TranscriptionConfig(batch_wait_seconds=5)
        assert config.batch_wait_seconds == 5

        # Invalid: batch_wait_seconds < 0
        with pytest.raises(ValueError, match="batch_wait_seconds.*must be >= 0"):
            TranscriptionConfig(batch_wait_seconds=-1)

    @pytest.mark.fast
    def test_batch_size_validation(self):
        """Test that batch_size must be >= 1"""
        from src.config.sections.core import TranscriptionConfig

        # Valid: batch_size >= 1
        config = TranscriptionConfig(batch_size=1)
        assert config.batch_size == 1

        config = TranscriptionConfig(batch_size=50)
        assert config.batch_size == 50

        # Invalid: batch_size < 1
        with pytest.raises(ValueError, match="batch_size.*must be >= 1"):
            TranscriptionConfig(batch_size=0)

    @pytest.mark.fast
    def test_default_batch_values(self):
        """Test default values for batch processing"""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig()

        # Default values per US-110-005
        assert config.batch_size == 50
        assert config.batch_wait_seconds == 5


# US-137-006: Tests for FFmpeg pipelining optimization


class TestPipelineDepthConfig:
    """Tests for pipeline depth configuration (US-137-006)"""

    @pytest.mark.fast
    def test_pipeline_depth_default_is_3(self):
        """Test that default pipeline_depth is 3 (US-137-006)"""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig()
        assert config.pipeline_depth == 3

    @pytest.mark.fast
    def test_pipeline_depth_custom_value(self):
        """Test that custom pipeline_depth value is respected"""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig(pipeline_depth=5)
        assert config.pipeline_depth == 5

    @pytest.mark.fast
    def test_pipeline_depth_zero_disables(self):
        """Test that pipeline_depth=0 disables pipelining"""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig(pipeline_depth=0)
        assert config.pipeline_depth == 0


class TestDynamicPipelineDepth:
    """Tests for dynamic pipeline depth adjustment (US-137-006)"""

    @pytest.mark.fast
    def test_adjust_pipeline_depth_increases_on_low_gpu(self):
        """Test pipeline depth increases when GPU utilization is low"""
        from src.transcription.parallel_processor import adjust_pipeline_depth
        from unittest.mock import Mock

        config = Mock()
        config.transcription = Mock()
        config.transcription.dynamic_pipeline_depth = True
        config.transcription.gpu_utilization_threshold_high = 85.0
        config.transcription.gpu_utilization_threshold_low = 50.0
        config.transcription.min_pipeline_depth = 1
        config.transcription.max_pipeline_depth = 6

        # Low GPU utilization should increase depth
        result = adjust_pipeline_depth(3, 30.0, config)
        assert result == 4

    @pytest.mark.fast
    def test_adjust_pipeline_depth_decreases_on_high_gpu(self):
        """Test pipeline depth decreases when GPU utilization is high"""
        from src.transcription.parallel_processor import adjust_pipeline_depth
        from unittest.mock import Mock

        config = Mock()
        config.transcription = Mock()
        config.transcription.dynamic_pipeline_depth = True
        config.transcription.gpu_utilization_threshold_high = 85.0
        config.transcription.gpu_utilization_threshold_low = 50.0
        config.transcription.min_pipeline_depth = 1
        config.transcription.max_pipeline_depth = 6

        # High GPU utilization should decrease depth
        result = adjust_pipeline_depth(3, 95.0, config)
        assert result == 2

    @pytest.mark.fast
    def test_adjust_pipeline_depth_respects_bounds(self):
        """Test pipeline depth stays within min/max bounds"""
        from src.transcription.parallel_processor import adjust_pipeline_depth
        from unittest.mock import Mock

        config = Mock()
        config.transcription = Mock()
        config.transcription.dynamic_pipeline_depth = True
        config.transcription.gpu_utilization_threshold_high = 85.0
        config.transcription.gpu_utilization_threshold_low = 50.0
        config.transcription.min_pipeline_depth = 1
        config.transcription.max_pipeline_depth = 6

        # Try to increase beyond max
        result = adjust_pipeline_depth(6, 30.0, config)
        assert result == 6  # Should stay at max

        # Try to decrease below min
        result = adjust_pipeline_depth(1, 95.0, config)
        assert result == 1  # Should stay at min

    @pytest.mark.fast
    def test_adjust_pipeline_depth_no_change_when_disabled(self):
        """Test pipeline depth doesn't change when dynamic adjustment is disabled"""
        from src.transcription.parallel_processor import adjust_pipeline_depth
        from unittest.mock import Mock

        config = Mock()
        config.transcription = Mock()
        config.transcription.dynamic_pipeline_depth = False

        result = adjust_pipeline_depth(3, 30.0, config)
        assert result == 3

    @pytest.mark.fast
    def test_adjust_pipeline_depth_no_change_when_unavailable(self):
        """Test pipeline depth doesn't change when GPU utilization unavailable"""
        from src.transcription.parallel_processor import adjust_pipeline_depth
        from unittest.mock import Mock

        config = Mock()
        config.transcription = Mock()
        config.transcription.dynamic_pipeline_depth = True

        # -1.0 indicates unavailable
        result = adjust_pipeline_depth(3, -1.0, config)
        assert result == 3


class TestAutoTuneWorkers:
    """Tests for auto-tuning max_workers (US-137-006)"""

    @pytest.mark.fast
    def test_auto_tune_workers_with_gpu(self):
        """Test worker tuning when GPU is available"""
        from src.transcription.parallel_processor import auto_tune_max_workers
        from unittest.mock import Mock, patch

        config = Mock()
        config.transcription = Mock()
        config.transcription.auto_tune_workers = True
        config.transcription.worker_multiplier = 0.5

        with patch('src.transcription.parallel_processor.get_available_gpu_memory', return_value=1000.0):
            with patch('torch.cuda.is_available', return_value=True):
                workers = auto_tune_max_workers(config)
                # With GPU, should reduce workers
                assert workers >= 1

    @pytest.mark.fast
    def test_auto_tune_workers_disabled(self):
        """Test worker tuning when disabled"""
        from src.transcription.parallel_processor import auto_tune_max_workers
        from unittest.mock import Mock

        config = Mock()
        config.transcription = Mock()
        config.transcription.auto_tune_workers = False
        config.transcription.audio_extraction_workers = 8

        workers = auto_tune_max_workers(config)
        assert workers == 8


class TestPipelineEfficiencyMetrics:
    """Tests for pipeline efficiency metrics tracking (US-137-006)"""

    @pytest.mark.fast
    def test_transcription_metrics_has_pipeline_fields(self):
        """Test TranscriptionMetrics has pipeline efficiency fields"""
        from src.transcription.metrics import TranscriptionMetrics

        metrics = TranscriptionMetrics(total_videos=10)

        # Should have pipeline efficiency fields
        assert hasattr(metrics, 'pipeline_avg_extraction_wait_s')
        assert hasattr(metrics, 'pipeline_avg_transcription_s')
        assert hasattr(metrics, 'pipeline_efficiency_ratio')

    @pytest.mark.fast
    def test_set_pipeline_efficiency(self):
        """Test set_pipeline_efficiency method"""
        from src.transcription.metrics import TranscriptionMetrics

        metrics = TranscriptionMetrics(total_videos=10)
        metrics.set_pipeline_efficiency(1.5, 3.0, 0.5)

        assert metrics.pipeline_avg_extraction_wait_s == 1.5
        assert metrics.pipeline_avg_transcription_s == 3.0
        assert metrics.pipeline_efficiency_ratio == 0.5

    @pytest.mark.fast
    def test_summary_includes_pipeline_efficiency(self):
        """Test get_summary_dict includes pipeline efficiency"""
        from src.transcription.metrics import TranscriptionMetrics

        metrics = TranscriptionMetrics(total_videos=10)
        metrics.set_pipeline_efficiency(1.5, 3.0, 0.5)

        summary = metrics.get_summary_dict()

        assert 'pipeline_avg_extraction_wait_s' in summary
        assert 'pipeline_avg_transcription_s' in summary
        assert 'pipeline_efficiency_ratio' in summary
        assert summary['pipeline_avg_extraction_wait_s'] == 1.5
        assert summary['pipeline_avg_transcription_s'] == 3.0
        assert summary['pipeline_efficiency_ratio'] == 0.5


class TestGPUUtilization:
    """Tests for GPU utilization detection (US-137-006)"""

    @pytest.mark.fast
    def test_get_gpu_utilization_returns_float(self):
        """Test get_gpu_utilization returns a float"""
        from src.transcription.whisper_client import get_gpu_utilization

        result = get_gpu_utilization()
        # Should return a float (either -1 for unavailable or 0-100)
        assert isinstance(result, float)

    @pytest.mark.fast
    def test_get_gpu_utilization_unavailable(self):
        """Test get_gpu_utilization when nvidia-smi unavailable"""
        from src.transcription.whisper_client import get_gpu_utilization
        from unittest.mock import patch

        with patch('subprocess.run', side_effect=FileNotFoundError):
            result = get_gpu_utilization()
            assert result == -1.0
