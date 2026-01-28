"""
Extended tests for src/transcription/parallel_processor.py

Focuses on uncovered code paths:
- Progress printing with show_progress=True
- Parallel processing edge cases (multiple workers, ETA calculation)
- Cleanup error handling (audio file deletion failures)
- Worker pool management edge cases
- Cache integration edge cases
- Word timestamps JSON saving (success and failure)
"""

import sys
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, call, mock_open
from pathlib import Path
import time
import tempfile
import os

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
    return config


@pytest.fixture
def sample_raw_segments():
    """Sample raw segment data from Whisper"""
    return [
        {'start': 0.0, 'end': 5.0, 'text': 'First segment'},
        {'start': 5.0, 'end': 10.0, 'text': 'Second segment'},
        {'start': 10.0, 'end': 15.0, 'text': 'Third segment'}
    ]


class TestProgressPrinting:
    """Test show_progress=True code paths"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_all_cached_with_progress(
        self, mock_print, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test progress printing when all videos are cached"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.side_effect = [
            sample_raw_segments,
            sample_raw_segments
        ]

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        assert len(results) == 2

        # Should print cache status
        mock_print.assert_any_call(
            "  Video index: 2 cached, 0 new",
            flush=True
        )

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_uncached_with_progress_phase1(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test Phase 1 progress printing during audio extraction"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        # Verify Phase 1 message was printed
        phase1_calls = [c for c in mock_print.call_args_list
                        if "Phase 1" in str(c)]
        assert len(phase1_calls) >= 1

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_uncached_with_progress_phase2(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test Phase 2 progress printing during transcription"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.side_effect = lambda vp, _: f"/fake/audio/{Path(vp).stem}.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        # Verify Phase 2 message was printed
        phase2_calls = [c for c in mock_print.call_args_list
                        if "Phase 2" in str(c)]
        assert len(phase2_calls) >= 1

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_progress_with_eta_calculation(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test ETA calculation path (i > 0 branch)"""
        # Need at least 2 videos to trigger ETA calculation on second video
        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.side_effect = lambda vp, _: f"/fake/audio/{Path(vp).stem}.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        assert len(results) == 3
        # ETA calculation should occur for videos after the first


class TestParallelProcessingEdgeCases:
    """Test edge cases in parallel processing"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_audio_extraction_returns_none(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test when extract_audio returns None (no exception, just None)"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # First video returns None, second returns valid path
        mock_extract.side_effect = [None, "/fake/audio/video2.wav"]

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=False
        )

        # Only video2 should be in results (video1 extraction failed)
        assert len(results) == 1
        assert "/video2.mp4" in results

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_extraction_progress_at_interval(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test progress printing at 10-video intervals"""
        # Create 11 videos to trigger the modulo 10 == 0 check
        video_paths = [f"/video{i}.mp4" for i in range(11)]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.side_effect = lambda vp, _: f"/fake/audio/{Path(vp).stem}.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            max_workers=4,
            show_progress=True
        )

        # Look for extraction progress message at interval
        extracted_calls = [c for c in mock_print.call_args_list
                          if "Extracted" in str(c) and "/11" in str(c)]
        assert len(extracted_calls) >= 1

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_single_worker(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test with single worker (max_workers=1)"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.side_effect = lambda vp, _: f"/fake/audio/{Path(vp).stem}.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            max_workers=1,
            show_progress=False
        )

        assert len(results) == 2

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_high_worker_count(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test with more workers than videos"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.side_effect = lambda vp, _: f"/fake/audio/{Path(vp).stem}.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            max_workers=10,  # More workers than videos
            show_progress=False
        )

        assert len(results) == 2


class TestCleanupErrorHandling:
    """Test error handling in cleanup operations"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_audio_file_unlink_failure(
        self, mock_mkdir, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test when audio file deletion fails (line 204-205)"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink', side_effect=PermissionError("Cannot delete")):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        # Should complete successfully despite unlink failure
        assert len(results) == 1
        assert "/video1.mp4" in results

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.integration
    def test_temp_dir_rmtree_failure(
        self, mock_mkdir, mock_unlink, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test when temp directory cleanup fails (line 217-218)"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('src.transcription.parallel_processor.shutil.rmtree',
                   side_effect=OSError("Cannot remove")):
            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                show_progress=False
            )

        # Should complete successfully despite rmtree failure
        assert len(results) == 1

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @pytest.mark.fast
    def test_transcribe_video_unlink_failure(
        self, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test transcribe_video when audio file deletion fails (line 289-290)"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink', side_effect=FileNotFoundError("Not found")):
            result = transcribe_video("/video1.mp4", mock_cache)

        # Should complete successfully despite unlink failure
        assert len(result) == 3


class TestVoiceoverMediaEdgeCases:
    """Test edge cases in transcribe_voiceover_media"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_video_extraction_without_cache_dir(
        self, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test video extraction uses parent dir when cache_dir is None (line 370-371)"""
        media_path = tmp_path / "voiceover.mp4"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink'):
            with patch('builtins.open', mock_open()):
                result = transcribe_voiceover_media(
                    str(media_path),
                    cache_dir=None  # No cache_dir, should use parent
                )

        # Verify extraction was called with media's parent as temp_dir
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args[0]
        assert str(tmp_path) in call_args[1]

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_video_audio_cleanup_failure(
        self, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test video cleanup failure is handled (line 390-391)"""
        media_path = tmp_path / "voiceover.mp4"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('pathlib.Path.unlink', side_effect=OSError("Cannot delete")):
            with patch('builtins.open', mock_open()):
                result = transcribe_voiceover_media(str(media_path))

        # Should complete despite cleanup failure
        assert result.endswith('.srt')

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_word_timestamps_json_save_success(
        self, mock_logger, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test successful word timestamps JSON save (line 414-416)"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        # Use a real temp file for JSON write
        with patch('builtins.open', mock_open()) as m:
            result = transcribe_voiceover_media(
                str(media_path),
                word_timestamps=True
            )

        # Verify JSON was written
        assert m.called
        # Check that json.dump was used (via write call)
        write_calls = m().write.call_args_list
        assert len(write_calls) > 0

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('src.transcription.parallel_processor.logger')
    @pytest.mark.fast
    def test_word_timestamps_json_save_failure(
        self, mock_logger, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test word timestamps JSON save failure is logged (line 417-418)"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        # Make file open fail for JSON write
        def selective_open(path, *args, **kwargs):
            if '.words.json' in str(path):
                raise PermissionError("Cannot write JSON")
            return mock_open()()

        with patch('builtins.open', side_effect=selective_open):
            result = transcribe_voiceover_media(
                str(media_path),
                word_timestamps=True
            )

        # Should complete and log warning
        assert result.endswith('.srt')
        mock_logger.warning.assert_called()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_word_timestamps_disabled(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test when word_timestamps=False, no JSON is saved"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()) as m:
            result = transcribe_voiceover_media(
                str(media_path),
                word_timestamps=False
            )

        # Should not attempt JSON write
        assert result.endswith('.srt')


class TestConfigEdgeCases:
    """Test config handling edge cases"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_config_with_missing_attributes(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, sample_raw_segments
    ):
        """Test config object with missing transcription attributes uses defaults"""
        video_paths = ["/video1.mp4"]

        # Config with partial attributes
        config = Mock()
        config.transcription = Mock(spec=[])  # Empty spec, no attributes

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, config,
            show_progress=False
        )

        # Should use defaults via getattr
        MockWhisperClient.assert_called_once_with(
            model_name="base",  # default
            compute_type="auto"  # default
        )


class TestEmptyInputs:
    """Test handling of empty inputs"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @pytest.mark.fast
    def test_empty_video_list(
        self, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config
    ):
        """Test with empty video list"""
        results = transcribe_videos_parallel(
            [], mock_cache, mock_config,
            show_progress=False
        )

        assert results == {}

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_all_extractions_fail(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config
    ):
        """Test when all audio extractions fail"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # All extractions return None
        mock_extract.return_value = None

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=False
        )

        # Should return empty results (no videos to transcribe)
        assert results == {}


class TestTranscribeVideoVadSettings:
    """Test VAD filter settings in transcribe_video"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_vad_filter_enabled(
        self, mock_unlink, mock_extract, MockWhisperClient,
        sample_raw_segments
    ):
        """Test transcribe_video with VAD filter enabled"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        result = transcribe_video(
            "/video1.mp4",
            mock_cache,
            vad_filter=True,
            min_silence_duration_ms=300,
            speech_pad_ms=15
        )

        # Verify VAD settings were passed
        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs['vad_filter'] is True
        assert call_kwargs['min_silence_duration_ms'] == 300
        assert call_kwargs['speech_pad_ms'] == 15


class TestCompletionStatusPrinting:
    """Test completion status printing"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_phase1_completion_message(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test Phase 1 completion message is printed"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        # Check for Phase 1 complete message
        phase1_complete = [c for c in mock_print.call_args_list
                          if "Phase 1 complete" in str(c)]
        assert len(phase1_complete) >= 1

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_phase2_completion_message(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test Phase 2 completion message is printed"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        # Check for Phase 2 complete message
        phase2_complete = [c for c in mock_print.call_args_list
                          if "Phase 2 complete" in str(c)]
        assert len(phase2_complete) >= 1

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_newline_after_progress(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test newline is printed after progress (line 208)"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        # Check for empty print() call (newline)
        newline_calls = [c for c in mock_print.call_args_list if c == call()]
        assert len(newline_calls) >= 1


class TestVoiceoverMediaExtensions:
    """Test different file extension handling"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_wav_audio_file(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .wav audio file processing"""
        media_path = tmp_path / "voiceover.wav"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_m4a_audio_file(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .m4a audio file processing"""
        media_path = tmp_path / "voiceover.m4a"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_mkv_video_file(
        self, mock_mkdir, mock_unlink, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .mkv video file processing"""
        media_path = tmp_path / "voiceover.mkv"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                cache_dir=str(tmp_path)
            )

        assert result.endswith('.srt')
        mock_extract.assert_called_once()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_webm_video_file(
        self, mock_mkdir, mock_unlink, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .webm video file processing"""
        media_path = tmp_path / "voiceover.webm"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                cache_dir=str(tmp_path)
            )

        assert result.endswith('.srt')


class TestTranscriptionWithLongVideoNames:
    """Test handling of long video names (truncation)"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_long_video_name_truncation(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that long video names are truncated in progress display"""
        # Create video path with very long name (>40 chars)
        long_name = "this_is_a_very_long_video_name_that_exceeds_forty_characters.mp4"
        video_paths = [f"/{long_name}"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.side_effect = lambda vp, _: f"/fake/audio/{Path(vp).stem}.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        assert len(results) == 1


class TestAdditionalAudioExtensions:
    """Test additional audio file extensions"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_ogg_audio_file(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .ogg audio file processing"""
        media_path = tmp_path / "voiceover.ogg"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_flac_audio_file(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .flac audio file processing"""
        media_path = tmp_path / "voiceover.flac"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_aac_audio_file(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .aac audio file processing"""
        media_path = tmp_path / "voiceover.aac"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(str(media_path))

        assert result.endswith('.srt')


class TestAdditionalVideoExtensions:
    """Test additional video file extensions"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_avi_video_file(
        self, mock_mkdir, mock_unlink, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .avi video file processing"""
        media_path = tmp_path / "voiceover.avi"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                cache_dir=str(tmp_path)
            )

        assert result.endswith('.srt')
        mock_extract.assert_called_once()

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_mov_video_file(
        self, mock_mkdir, mock_unlink, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .mov video file processing"""
        media_path = tmp_path / "voiceover.mov"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                cache_dir=str(tmp_path)
            )

        assert result.endswith('.srt')

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.write_srt')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @pytest.mark.fast
    def test_mxf_video_file(
        self, mock_mkdir, mock_unlink, mock_write_srt, mock_extract, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test .mxf video file processing"""
        media_path = tmp_path / "voiceover.mxf"
        media_path.write_text("fake video")

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                cache_dir=str(tmp_path)
            )

        assert result.endswith('.srt')


class TestFirstVideoEtaCalculation:
    """Test ETA calculation edge cases"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('builtins.print')
    @pytest.mark.fast
    def test_eta_zero_for_first_video(
        self, mock_print, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test ETA is 0 for first video (i == 0 branch)"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=True
        )

        # First video should show ETA: 0s
        eta_calls = [c for c in mock_print.call_args_list
                     if "ETA: 0s" in str(c)]
        assert len(eta_calls) >= 1


class TestVoiceoverMediaLanguageSettings:
    """Test language settings in voiceover media transcription"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_voiceover_with_language(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test voiceover transcription with specified language"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                language="es"
            )

        # Verify language was passed
        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs['language'] == "es"

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.write_srt')
    @pytest.mark.fast
    def test_voiceover_with_custom_model(
        self, mock_write_srt, MockWhisperClient,
        tmp_path, sample_raw_segments
    ):
        """Test voiceover transcription with custom model"""
        media_path = tmp_path / "voiceover.mp3"
        media_path.write_text("fake audio")

        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        with patch('builtins.open', mock_open()):
            result = transcribe_voiceover_media(
                str(media_path),
                model_name="large-v3",
                compute_type="float16"
            )

        # Verify model settings were passed
        MockWhisperClient.assert_called_once_with(
            model_name="large-v3",
            compute_type="float16"
        )


class TestPathObjectInput:
    """Test handling of Path objects vs string paths"""

    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('pathlib.Path.unlink')
    @pytest.mark.fast
    def test_transcribe_video_with_path_object(
        self, mock_unlink, mock_extract, MockWhisperClient,
        sample_raw_segments, tmp_path
    ):
        """Test transcribe_video accepts Path object input"""
        mock_cache = Mock()
        mock_cache.get.return_value = None

        mock_extract.return_value = str(tmp_path / "audio.wav")
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        # Pass Path object instead of string
        video_path = tmp_path / "video1.mp4"
        result = transcribe_video(video_path, mock_cache)

        assert len(result) == 3
        # Video path should be converted to string
        mock_cache.get.assert_called_once_with(str(video_path))
