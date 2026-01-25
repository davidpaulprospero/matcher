"""
Tests for skip_if_cached parameter in transcribe_videos_parallel()

Tests the skip_if_cached behavior:
- When skip_if_cached=True (default), skip videos already in cache
- When skip_if_cached=False, reprocess all videos ignoring cache
- Logging count of skipped videos
"""

import sys
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription.parallel_processor import transcribe_videos_parallel
from src.state import TranscriptSegment

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


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


class TestSkipIfCachedParameter:
    """Test skip_if_cached parameter exists and has correct default"""

    def test_skip_if_cached_parameter_exists(self):
        """Test that skip_if_cached parameter exists in function signature"""
        import inspect
        sig = inspect.signature(transcribe_videos_parallel)
        params = list(sig.parameters.keys())
        assert 'skip_if_cached' in params

    def test_skip_if_cached_default_true(self):
        """Test that skip_if_cached defaults to True"""
        import inspect
        sig = inspect.signature(transcribe_videos_parallel)
        default = sig.parameters['skip_if_cached'].default
        assert default is True

    def test_skip_if_cached_is_bool_type(self):
        """Test that skip_if_cached default is boolean"""
        import inspect
        sig = inspect.signature(transcribe_videos_parallel)
        default = sig.parameters['skip_if_cached'].default
        assert isinstance(default, bool)


class TestSkipIfCachedTrue:
    """Test behavior when skip_if_cached=True (default)"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    def test_cached_videos_skipped(
        self, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that cached videos are skipped when skip_if_cached=True"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock all videos cached
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.side_effect = [
            sample_raw_segments,  # video1 cached
            sample_raw_segments   # video2 cached
        ]

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,  # Explicit True
            show_progress=False
        )

        # Should return cached results
        assert len(results) == 2
        assert all(len(results[vp]) == 3 for vp in video_paths)

        # Should not transcribe (all cached)
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.assert_not_called()

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    def test_default_skips_cached(
        self, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that default behavior (no skip_if_cached arg) skips cached videos"""
        video_paths = ["/video1.mp4"]

        # Mock video cached
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        # Call without skip_if_cached argument (uses default True)
        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            show_progress=False
        )

        # Should check cache
        mock_transcript_cache.get.assert_called()

        # Should not transcribe
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.assert_not_called()

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_uncached_videos_processed_when_skip_true(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that uncached videos are processed even when skip_if_cached=True"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        # Mock video1 cached, video2 not cached
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.side_effect = [
            sample_raw_segments,  # video1 cached
            None                  # video2 not cached
        ]

        # Mock extraction and transcription for uncached video
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,
            show_progress=False
        )

        assert len(results) == 2

        # Should only transcribe uncached video
        assert mock_whisper.transcribe.call_count == 1


class TestSkipIfCachedFalse:
    """Test behavior when skip_if_cached=False"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_cached_videos_reprocessed(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that cached videos are reprocessed when skip_if_cached=False"""
        video_paths = ["/video1.mp4"]

        # Mock video cached (should be ignored)
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        # Mock extraction and transcription
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=False,  # Explicit False - reprocess all
            show_progress=False
        )

        assert len(results) == 1

        # Should NOT check cache when skip_if_cached=False
        mock_transcript_cache.get.assert_not_called()

        # Should extract and transcribe
        mock_extract.assert_called_once()
        mock_whisper.transcribe.assert_called_once()

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_all_videos_reprocessed(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that all videos are reprocessed when skip_if_cached=False"""
        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        # Mock extraction and transcription
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=False,
            show_progress=False
        )

        assert len(results) == 3

        # Should extract and transcribe all 3 videos
        assert mock_extract.call_count == 3
        assert mock_whisper.transcribe.call_count == 3


class TestSkipIfCachedWithForceReprocess:
    """Test interaction between skip_if_cached and force_reprocess"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_force_reprocess_overrides_skip_if_cached(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that force_reprocess=True overrides skip_if_cached=True"""
        video_paths = ["/video1.mp4"]

        # Mock extraction and transcription
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,   # Would skip
            force_reprocess=True,  # But force overrides
            show_progress=False
        )

        assert len(results) == 1

        # Should NOT check cache (force_reprocess takes precedence)
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.assert_not_called()

        # Should transcribe
        mock_whisper.transcribe.assert_called_once()

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_both_false_still_reprocesses(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that skip_if_cached=False still works when force_reprocess=False"""
        video_paths = ["/video1.mp4"]

        # Mock extraction and transcription
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=False,
            force_reprocess=False,
            show_progress=False
        )

        assert len(results) == 1

        # Should NOT check cache (skip_if_cached=False)
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.assert_not_called()

        # Should transcribe
        mock_whisper.transcribe.assert_called_once()


class TestSkipIfCachedLogging:
    """Test logging when videos are skipped"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.logger')
    def test_logs_skipped_count(
        self, mock_logger, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that skipped video count is logged when skip_if_cached=True"""
        video_paths = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

        # Mock all videos cached
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,
            show_progress=False
        )

        # Should log skipped count
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        skipped_log = [log for log in log_calls if 'Skipped' in log and 'cached' in log]
        assert len(skipped_log) >= 1
        assert '3' in skipped_log[0]  # 3 videos skipped

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.logger')
    def test_no_log_when_no_cached_videos(
        self, mock_logger, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that no skipped log when no videos are cached"""
        video_paths = ["/video1.mp4"]

        # Mock no videos cached
        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = None

        # Mock extraction and transcription
        with patch('src.transcription.parallel_processor.extract_audio') as mock_extract, \
             patch('pathlib.Path.unlink'), patch('pathlib.Path.mkdir'), \
             patch('src.transcription.parallel_processor.shutil.rmtree'):
            mock_extract.return_value = "/fake/audio.wav"
            mock_whisper = MockWhisperClient.return_value
            mock_whisper.transcribe.return_value = sample_raw_segments

            results = transcribe_videos_parallel(
                video_paths, mock_cache, mock_config,
                skip_if_cached=True,
                show_progress=False
            )

        # Should NOT log "Skipped" message (no cached videos)
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list if mock_logger.info.call_args_list]
        skipped_logs = [log for log in log_calls if 'Skipped' in log and 'cached' in log]
        assert len(skipped_logs) == 0

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    @patch('src.transcription.parallel_processor.logger')
    def test_no_skip_log_when_skip_false(
        self, mock_logger, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that no skipped log when skip_if_cached=False"""
        video_paths = ["/video1.mp4"]

        # Mock extraction and transcription
        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=False,
            show_progress=False
        )

        # Should NOT log "Skipped" message (skip_if_cached=False)
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list if mock_logger.info.call_args_list]
        skipped_logs = [log for log in log_calls if 'Skipped' in log and 'cached' in log]
        assert len(skipped_logs) == 0


class TestSkipIfCachedReturnsCorrectResults:
    """Test that correct results are returned in various scenarios"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    def test_returns_transcript_segments(
        self, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that cached results are converted to TranscriptSegment objects"""
        video_paths = ["/video1.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,
            show_progress=False
        )

        assert len(results) == 1
        segments = results["/video1.mp4"]
        assert len(segments) == 3
        assert all(isinstance(seg, TranscriptSegment) for seg in segments)
        assert segments[0].text == "First segment"
        assert segments[0].source_file == "/video1.mp4"

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    @patch('src.transcription.parallel_processor.extract_audio')
    @patch('src.transcription.parallel_processor.shutil.rmtree')
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.mkdir')
    def test_mixed_cached_and_fresh_results(
        self, mock_mkdir, mock_unlink, mock_rmtree, mock_extract,
        MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test mixed cached and freshly transcribed results"""
        video_paths = ["/cached.mp4", "/fresh.mp4"]

        # Different segments for fresh transcription
        fresh_segments = [
            {'start': 0.0, 'end': 3.0, 'text': 'Fresh content'}
        ]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.side_effect = [
            sample_raw_segments,  # cached.mp4 in cache
            None                  # fresh.mp4 not in cache
        ]

        mock_extract.return_value = "/fake/audio.wav"
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.return_value = fresh_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,
            show_progress=False
        )

        assert len(results) == 2

        # Cached result should have 3 segments
        assert len(results["/cached.mp4"]) == 3
        assert results["/cached.mp4"][0].text == "First segment"

        # Fresh result should have 1 segment
        assert len(results["/fresh.mp4"]) == 1
        assert results["/fresh.mp4"][0].text == "Fresh content"


class TestSkipIfCachedEdgeCases:
    """Test edge cases for skip_if_cached"""

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    def test_empty_video_list(
        self, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config
    ):
        """Test with empty video list"""
        results = transcribe_videos_parallel(
            [], mock_cache, mock_config,
            skip_if_cached=True,
            show_progress=False
        )

        assert results == {}

    @patch('src.transcription.parallel_processor.TranscriptCache')
    @patch('src.transcription.parallel_processor.WhisperClient')
    def test_all_cached_returns_immediately(
        self, MockWhisperClient, MockTranscriptCache,
        mock_cache, mock_config, sample_raw_segments
    ):
        """Test that function returns early when all videos cached"""
        video_paths = ["/video1.mp4", "/video2.mp4"]

        mock_transcript_cache = MockTranscriptCache.return_value
        mock_transcript_cache.get.return_value = sample_raw_segments

        results = transcribe_videos_parallel(
            video_paths, mock_cache, mock_config,
            skip_if_cached=True,
            show_progress=False
        )

        assert len(results) == 2

        # WhisperClient should be initialized but never used
        mock_whisper = MockWhisperClient.return_value
        mock_whisper.transcribe.assert_not_called()
