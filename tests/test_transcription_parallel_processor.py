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
        mock_extract.side_effect = lambda vp, temp_dir: f"/fake/cache/temp_audio/{Path(vp).stem}.wav"

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
        mock_extract.side_effect = lambda vp, temp_dir: f"/fake/audio/{Path(vp).stem}.wav"

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
        def extract_side_effect(vp, temp_dir):
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

        # Should have logged with logger.exception() (includes traceback)
        assert mock_logger.exception.called

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
            compute_type="auto"
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
        mock_extract.assert_called_once_with("/video1.mp4", None)

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
            compute_type="int8"
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
            compute_type="auto"
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
            compute_type="float16"
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
        def extract_side_effect(vp, temp_dir):
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
        mock_extract.side_effect = lambda vp, temp_dir: f"/fake/audio/{Path(vp).stem}.wav"

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

        # Should have logged with logger.exception() (includes traceback)
        assert mock_logger.exception.called, "Expected logger.exception() to be called for traceback"

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
