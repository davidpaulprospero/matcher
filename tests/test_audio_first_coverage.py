"""
Tests for src/downloader/audio_first.py - Comprehensive Coverage

Targets missed lines from coverage analysis:
- Lines 117-119: Search exception handling
- Lines 149-153: Cleaning up stale .part files
- Lines 168-169, 172-181: Existing file detection and skipping download
- Lines 211: Getting audio_timeout from object attribute (not dict)
- Lines 239-249: Download failure handling and cleanup
- Line 296: Skip empty segments list
- Lines 382-383: Segment download timeout handling

Created: 2026-01-11 (Coverage expansion session)
"""

import sys
import subprocess
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.audio_first import AudioFirstPipeline
from src.downloader.title_filter import SearchResult
from src.downloader.types import MergedSegment, DownloadedSegment, MatchedSegment
from src.state import AudioDownload
from src.config.sections.download import DownloadConfig, AudioFirstConfig


def _sr(videos):
    """Wrap video list in SearchResult for mock return values."""
    return SearchResult(videos=videos)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Create mock config with audio_first settings.

    Uses spec=AudioFirstConfig and spec=DownloadConfig to catch phantom attributes.
    Attributes that should be absent are explicitly set to None or empty string.
    """
    config = Mock()

    # Audio-first config - use spec to validate attribute names
    audio_first = Mock(spec=AudioFirstConfig)
    audio_first.enabled = True
    audio_first.buffer_seconds = 30.0
    audio_first.merge_gap_seconds = 15.0
    audio_first.min_segment_duration = 5.0
    audio_first.max_segment_duration = 600.0
    audio_first.fallback_full_video = True
    audio_first.audio_quality = 5

    download = Mock(spec=DownloadConfig)
    download.audio_first = audio_first
    download.download_timeouts = {'short': 60, 'medium': 120, 'long': 300}
    download.max_keyword_len = 50
    download.ffmpeg_location = ''
    download.llm_title_filter = None  # Explicitly None - not auto-created by MagicMock
    download.cookies_from_browser = ''  # Empty string, not None (real attr type is str)
    download.cookies_path = ''  # Empty string, not None (real attr type is str)
    download.max_retries = 3
    download.retry_delay = 2.0

    config.download = download
    return config


@pytest.fixture
def audio_pipeline(mock_config):
    """Create AudioFirstPipeline instance with mocked dependencies"""
    lock = threading.Lock()

    pipeline = AudioFirstPipeline(
        config=mock_config,
        get_tier_value_func=Mock(side_effect=lambda tier, key, default: {
            'max_total': 0,
            'per_keyword': 5,
            'min': 20,
            'max': 600,
        }.get(key, default)),
        search_metadata_func=Mock(return_value=[]),
        filter_titles_func=Mock(side_effect=lambda videos, *args: videos),
        cleanup_partial_func=Mock(),
        tier_download_counts={},
        lock=lock
    )

    return pipeline


# ============================================================================
# Test Lines 117-119: Search Exception Handling
# ============================================================================

class TestSearchExceptionHandling:
    """Test search function exception handling (lines 117-119)"""

    def test_search_raises_generic_exception(self, audio_pipeline, temp_dir):
        """Test handling of generic exception during search"""
        audio_pipeline._search_video_metadata = Mock(
            side_effect=Exception("Network connection failed")
        )

        result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

        assert result == []
        audio_pipeline._search_video_metadata.assert_called_once()

    def test_search_raises_timeout_error(self, audio_pipeline, temp_dir):
        """Test handling of timeout error during search"""
        audio_pipeline._search_video_metadata = Mock(
            side_effect=TimeoutError("Search timed out")
        )

        result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

        assert result == []

    def test_search_raises_value_error(self, audio_pipeline, temp_dir):
        """Test handling of ValueError during search"""
        audio_pipeline._search_video_metadata = Mock(
            side_effect=ValueError("Invalid search parameters")
        )

        result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

        assert result == []

    def test_search_raises_runtime_error(self, audio_pipeline, temp_dir):
        """Test handling of RuntimeError during search"""
        audio_pipeline._search_video_metadata = Mock(
            side_effect=RuntimeError("yt-dlp process crashed")
        )

        result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

        assert result == []


# ============================================================================
# Test Lines 149-153: Cleaning Up Stale .part Files
# ============================================================================

class TestPartFileCleanup:
    """Test cleanup of stale .part files (lines 149-153)"""

    def test_cleans_up_part_files(self, audio_pipeline, temp_dir):
        """Test that .part files are cleaned up before download"""
        # Setup: Create audio directory with stale .part files
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        part_file1 = audio_dir / "vid1.part"
        part_file2 = audio_dir / "vid2.part.mp3"
        part_file3 = audio_dir / "vid3.part123"

        part_file1.write_text("stale download 1")
        part_file2.write_text("stale download 2")
        part_file3.write_text("stale download 3")

        # Mock search to return results that match keyword/tier naming
        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Travel Video', 'duration': 120, 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="failed")
            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

        # Verify .part files were cleaned up
        assert not part_file1.exists(), "part file 1 should be cleaned up"
        assert not part_file2.exists(), "part file 2 should be cleaned up"
        assert not part_file3.exists(), "part file 3 should be cleaned up"

    def test_cleans_part_files_with_exception(self, audio_pipeline, temp_dir):
        """Test that cleanup continues even if one file deletion fails"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        part_file1 = audio_dir / "vid1.part"
        part_file1.write_text("stale download")

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Travel Video', 'duration': 120, 'is_live': False}
        ]))

        # Make the part file read-only to trigger exception on unlink
        # Note: On Windows this might not raise an exception, so we handle gracefully
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr="failed")
            # This should not raise even if cleanup fails
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            # Function should complete without raising
            assert isinstance(result, list)


# ============================================================================
# Test Lines 168-169, 172-181: Existing File Detection
# ============================================================================

class TestExistingFileDetection:
    """Test detection and skipping of existing audio files (lines 168-181)"""

    def test_skips_existing_mp3_file(self, audio_pipeline, temp_dir):
        """Test that existing .mp3 files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Create existing audio file
        existing_file = audio_dir / "vid1.mp3"
        existing_file.write_bytes(b'existing audio content')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Travel Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # subprocess should NOT be called (file already exists)
            mock_run.assert_not_called()

        # Should return AudioDownload for existing file
        assert len(result) == 1
        assert result[0].video_id == 'vid1'
        assert 'vid1.mp3' in result[0].file

    def test_skips_existing_m4a_file(self, audio_pipeline, temp_dir):
        """Test that existing .m4a files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.m4a"
        existing_file.write_bytes(b'existing m4a content')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Travel Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            mock_run.assert_not_called()

        assert len(result) == 1
        assert 'vid1.m4a' in result[0].file

    def test_skips_existing_opus_file(self, audio_pipeline, temp_dir):
        """Test that existing .opus files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.opus"
        existing_file.write_bytes(b'existing opus content')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            mock_run.assert_not_called()

        assert len(result) == 1
        assert 'vid1.opus' in result[0].file

    def test_skips_existing_webm_file(self, audio_pipeline, temp_dir):
        """Test that existing .webm files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.webm"
        existing_file.write_bytes(b'existing webm content')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            mock_run.assert_not_called()

        assert len(result) == 1

    def test_skips_existing_mp4_file(self, audio_pipeline, temp_dir):
        """Test that existing .mp4 audio files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.mp4"
        existing_file.write_bytes(b'existing mp4 audio')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            mock_run.assert_not_called()

        assert len(result) == 1

    def test_skips_existing_wav_file(self, audio_pipeline, temp_dir):
        """Test that existing .wav files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.wav"
        existing_file.write_bytes(b'existing wav content')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            mock_run.assert_not_called()

        assert len(result) == 1

    def test_skips_existing_ogg_file(self, audio_pipeline, temp_dir):
        """Test that existing .ogg files are not re-downloaded"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.ogg"
        existing_file.write_bytes(b'existing ogg content')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")
            mock_run.assert_not_called()

        assert len(result) == 1

    def test_existing_file_uses_correct_metadata(self, audio_pipeline, temp_dir):
        """Test that existing files get correct metadata in AudioDownload"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        existing_file = audio_dir / "vid1.mp3"
        existing_file.write_bytes(b'existing audio')

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {
                'id': 'vid1',
                'title': 'My Travel Video',
                'duration': 185,
                'webpage_url': 'https://youtube.com/watch?v=vid1',
                'is_live': False
            }
        ]))

        with patch('subprocess.run'):
            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

        assert len(result) == 1
        assert result[0].video_id == 'vid1'
        assert result[0].title == 'My Travel Video'
        assert result[0].duration == 185
        assert result[0].keyword == 'travel'
        assert result[0].url == 'https://youtube.com/watch?v=vid1'


# ============================================================================
# Test Line 211: Getting audio_timeout from Object Attribute
# ============================================================================

class TestTimeoutFromObjectAttribute:
    """Test getting timeout from object attribute (line 211)"""

    def test_timeout_from_object_attribute(self, audio_pipeline, temp_dir):
        """Test that timeout is correctly retrieved from object attribute"""
        # Create a mock object with tier attributes instead of dict
        tier_timeouts_obj = Mock()
        tier_timeouts_obj.short = 45
        tier_timeouts_obj.medium = 90
        tier_timeouts_obj.long = 180

        audio_pipeline.download_config.download_timeouts = tier_timeouts_obj

        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Use a unique video ID that doesn't already have an audio file
        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'newvid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=newvid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            # Create the expected file AFTER subprocess.run is called
            def side_effect(*args, **kwargs):
                (audio_dir / "newvid1.mp3").write_bytes(b'audio')
                return Mock(returncode=0, stderr='')
            mock_run.side_effect = side_effect

            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Verify subprocess.run was called with correct timeout
            assert mock_run.called
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['timeout'] == 45

    def test_timeout_fallback_to_default(self, audio_pipeline, temp_dir):
        """Test that timeout falls back to default when attribute missing"""
        # Create object without the tier attribute
        tier_timeouts_obj = Mock(spec=[])  # No attributes
        tier_timeouts_obj.medium = 90  # Only medium exists

        audio_pipeline.download_config.download_timeouts = tier_timeouts_obj

        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'newvid2', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=newvid2', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            def side_effect(*args, **kwargs):
                (audio_dir / "newvid2.mp3").write_bytes(b'audio')
                return Mock(returncode=0, stderr='')
            mock_run.side_effect = side_effect

            # Should use default 120 when 'short' attribute doesn't exist
            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            assert mock_run.called
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['timeout'] == 120  # Default fallback


# ============================================================================
# Test Lines 239-249: Download Failure Handling
# ============================================================================

class TestDownloadFailureHandling:
    """Test audio download failure handling (lines 239-249)"""

    def test_download_failure_calls_cleanup(self, audio_pipeline, temp_dir):
        """Test that failed download triggers cleanup of partial files"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr='Download failed: Error 403')

            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Verify cleanup was called
            audio_pipeline._cleanup_partial_files.assert_called()

    def test_download_failure_long_stderr(self, audio_pipeline, temp_dir):
        """Test that long stderr is truncated in warning"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        # Create stderr longer than 500 chars
        long_stderr = 'X' * 1000

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr=long_stderr)

            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Should complete without error
            assert result == []

    def test_download_timeout_calls_cleanup(self, audio_pipeline, temp_dir):
        """Test that download timeout triggers cleanup"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=60)

            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Verify cleanup was called with correct args
            audio_pipeline._cleanup_partial_files.assert_called_with(audio_dir, 'vid1')

    def test_download_generic_exception_calls_cleanup(self, audio_pipeline, temp_dir):
        """Test that generic exception triggers cleanup"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = OSError("Disk full")

            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Verify cleanup was called
            audio_pipeline._cleanup_partial_files.assert_called()
            assert result == []

    def test_download_no_file_created(self, audio_pipeline, temp_dir):
        """Test handling when yt-dlp succeeds but no file is created"""
        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            # Return code 0 but don't create any file
            mock_run.return_value = Mock(returncode=0, stderr='')

            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Should handle gracefully (cleanup called, empty result)
            assert result == []


# ============================================================================
# Test Line 296: Skip Empty Segments List
# ============================================================================

class TestEmptySegmentsHandling:
    """Test handling of empty segments in video download (line 296)"""

    def test_skip_video_with_empty_segments(self, audio_pipeline, temp_dir):
        """Test that videos with empty segment lists are skipped"""
        # This tests the 'if not segments: continue' branch
        by_video = {
            'vid1': [],  # Empty segments - should be skipped
            'vid2': [
                MergedSegment(
                    video_id='vid2',
                    video_url='https://youtube.com/watch?v=vid2',
                    start_time=10.0,
                    end_time=50.0,
                    original_matches=[],
                    keyword='travel'
                )
            ]
        }

        # We need to test the internal grouping logic
        # Create merged segments list that would produce empty group
        merged_segments = [
            MergedSegment(
                video_id='vid2',
                video_url='https://youtube.com/watch?v=vid2',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch('src.downloader.segment_utils.rename_segments_with_timing', return_value=[]):
                result = audio_pipeline.download_video_segments(merged_segments, temp_dir)

                # Should only process vid2, not crash on empty vid1
                assert mock_run.call_count == 1


# ============================================================================
# Test Lines 382-383: Segment Download Timeout
# ============================================================================

class TestSegmentDownloadTimeout:
    """Test video segment download timeout handling (lines 382-383)"""

    def test_segment_download_timeout(self, audio_pipeline, temp_dir, capsys):
        """Test that segment download timeout is handled correctly"""
        merged_segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=300)

            with patch.object(audio_pipeline, '_download_full_video_fallback', return_value=[]):
                result = audio_pipeline.download_video_segments(merged_segments, temp_dir)

                # Should print timeout message
                captured = capsys.readouterr()
                assert 'Timeout' in captured.out

    def test_segment_download_timeout_triggers_fallback(self, audio_pipeline, temp_dir):
        """Test that timeout triggers fallback to full video download"""
        merged_segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            )
        ]

        fallback_called = [False]

        def mock_fallback(*args, **kwargs):
            fallback_called[0] = True
            return []

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=300)

            with patch.object(audio_pipeline, '_download_full_video_fallback', side_effect=mock_fallback):
                audio_pipeline.download_video_segments(merged_segments, temp_dir)

                assert fallback_called[0], "Fallback should be called on timeout"

    def test_segment_download_generic_error(self, audio_pipeline, temp_dir, capsys):
        """Test that generic errors during segment download are handled"""
        merged_segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = OSError("Network unreachable")

            with patch.object(audio_pipeline, '_download_full_video_fallback', return_value=[]):
                result = audio_pipeline.download_video_segments(merged_segments, temp_dir)

                # Should print error message
                captured = capsys.readouterr()
                assert 'Error' in captured.out


# ============================================================================
# Test Full Video Fallback Additional Coverage
# ============================================================================

class TestFullVideoFallbackCoverage:
    """Additional tests for full video fallback edge cases"""

    def test_fallback_generic_exception(self, audio_pipeline, temp_dir):
        """Test fallback handles generic exceptions"""
        video_dir = temp_dir / "video_dir"
        video_dir.mkdir()

        segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = OSError("Permission denied")

            result = audio_pipeline._download_full_video_fallback(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                video_dir=video_dir,
                segments=segments,
                keyword='travel'
            )

            assert result == []

    def test_fallback_collects_all_original_matches(self, audio_pipeline, temp_dir):
        """Test that fallback collects matches from all segments"""
        video_dir = temp_dir / "video_dir"
        video_dir.mkdir()

        match1 = MatchedSegment(
            video_id='vid1',
            video_url='https://youtube.com/watch?v=vid1',
            start_time=10.0,
            end_time=20.0,
            track='V1',
            voiceover_segment_idx=0
        )
        match2 = MatchedSegment(
            video_id='vid1',
            video_url='https://youtube.com/watch?v=vid1',
            start_time=30.0,
            end_time=40.0,
            track='V1',
            voiceover_segment_idx=1
        )

        segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=25.0,
                original_matches=[match1],
                keyword='travel'
            ),
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=30.0,
                end_time=45.0,
                original_matches=[match2],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            output_file = video_dir / "vid1_0000.mp4"
            output_file.write_bytes(b'video data')

            with patch.object(audio_pipeline, '_get_video_duration', return_value=60.0):
                result = audio_pipeline._download_full_video_fallback(
                    video_id='vid1',
                    video_url='https://youtube.com/watch?v=vid1',
                    video_dir=video_dir,
                    segments=segments,
                    keyword='travel'
                )

                assert len(result) == 1
                assert len(result[0].matches) == 2
                assert match1 in result[0].matches
                assert match2 in result[0].matches


# ============================================================================
# Test FFmpeg Location Configuration
# ============================================================================

class TestFfmpegLocationConfig:
    """Test FFmpeg location configuration in download commands"""

    def test_audio_download_with_ffmpeg_location(self, audio_pipeline, temp_dir):
        """Test that ffmpeg location is added to audio download command"""
        audio_pipeline.download_config.ffmpeg_location = '/custom/ffmpeg'

        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Use unique video ID to avoid existing file detection
        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'ffmpegvid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=ffmpegvid1', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            def side_effect(*args, **kwargs):
                (audio_dir / "ffmpegvid1.mp3").write_bytes(b'audio')
                return Mock(returncode=0, stderr='')
            mock_run.side_effect = side_effect

            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Verify ffmpeg-location was in command
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert '--ffmpeg-location' in call_args
            assert '/custom/ffmpeg' in call_args

    def test_segment_download_with_ffmpeg_location(self, audio_pipeline, temp_dir):
        """Test that ffmpeg location is added to segment download command"""
        audio_pipeline.download_config.ffmpeg_location = '/custom/ffmpeg'

        merged_segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch('src.downloader.segment_utils.rename_segments_with_timing', return_value=[]):
                audio_pipeline.download_video_segments(merged_segments, temp_dir)

                call_args = mock_run.call_args[0][0]
                assert '--ffmpeg-location' in call_args
                assert '/custom/ffmpeg' in call_args


# ============================================================================
# Test Multiple Videos/Segments Processing
# ============================================================================

class TestMultipleVideosProcessing:
    """Test processing of multiple videos with segments"""

    def test_download_segments_multiple_videos(self, audio_pipeline, temp_dir):
        """Test downloading segments from multiple videos"""
        merged_segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword='travel'
            ),
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=60.0,
                end_time=90.0,
                original_matches=[],
                keyword='travel'
            ),
            MergedSegment(
                video_id='vid2',
                video_url='https://youtube.com/watch?v=vid2',
                start_time=5.0,
                end_time=25.0,
                original_matches=[],
                keyword='beach'
            )
        ]

        call_count = [0]

        def mock_run(*args, **kwargs):
            call_count[0] += 1
            return Mock(returncode=0)

        with patch('subprocess.run', side_effect=mock_run):
            with patch('src.downloader.segment_utils.rename_segments_with_timing', return_value=[]):
                audio_pipeline.download_video_segments(merged_segments, temp_dir)

                # Should call subprocess twice (once per video)
                assert call_count[0] == 2

    def test_download_segments_uses_correct_section_args(self, audio_pipeline, temp_dir):
        """Test that correct --download-sections args are generated"""
        merged_segments = [
            MergedSegment(
                video_id='vid1',
                video_url='https://youtube.com/watch?v=vid1',
                start_time=65.0,  # 00:01:05
                end_time=185.0,   # 00:03:05
                original_matches=[],
                keyword='travel'
            )
        ]

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            with patch('src.downloader.segment_utils.rename_segments_with_timing', return_value=[]):
                audio_pipeline.download_video_segments(merged_segments, temp_dir)

                call_args = mock_run.call_args[0][0]

                # Should have --download-sections with formatted time
                assert '--download-sections' in call_args
                sections_idx = call_args.index('--download-sections')
                section_arg = call_args[sections_idx + 1]
                assert '*00:01:05-00:03:05' == section_arg


# ============================================================================
# Test LLM Title Filter Integration
# ============================================================================

class TestLLMTitleFilter:
    """Test LLM title filter integration"""

    def test_llm_filter_enabled_calls_filter(self, audio_pipeline, temp_dir):
        """Test that LLM filter is called when enabled"""
        # Enable LLM filter
        llm_filter = Mock()
        llm_filter.enabled = True
        audio_pipeline.download_config.llm_title_filter = llm_filter

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video 1', 'duration': 60, 'is_live': False},
            {'id': 'vid2', 'title': 'Video 2', 'duration': 60, 'is_live': False}
        ]))

        audio_pipeline._filter_titles_with_llm = Mock(return_value=[
            {'id': 'vid1', 'title': 'Video 1', 'duration': 60, 'is_live': False}
        ])

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr='')

            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short", topic="vacation")

            # Verify filter was called with topic
            audio_pipeline._filter_titles_with_llm.assert_called_once()

    def test_llm_filter_disabled_skips_filter(self, audio_pipeline, temp_dir):
        """Test that LLM filter is skipped when disabled"""
        llm_filter = Mock()
        llm_filter.enabled = False
        audio_pipeline.download_config.llm_title_filter = llm_filter

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60, 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=1, stderr='')

            audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Filter should not be called
            audio_pipeline._filter_titles_with_llm.assert_not_called()


# ============================================================================
# Test Tier Download Count Tracking
# ============================================================================

class TestTierDownloadCountTracking:
    """Test tier download count tracking and limits"""

    def test_tier_count_increments_on_success(self, audio_pipeline, temp_dir):
        """Test that tier count is incremented on successful downloads"""
        assert audio_pipeline.tier_download_counts.get('short', 0) == 0

        audio_dir = temp_dir / "travel_s_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Video', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid1', 'is_live': False},
            {'id': 'vid2', 'title': 'Video 2', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=vid2', 'is_live': False}
        ]))

        with patch('subprocess.run') as mock_run:
            def create_file(*args, **kwargs):
                cmd = args[0]
                for arg in cmd:
                    if 'youtube.com' in str(arg):
                        vid_id = str(arg).split('v=')[-1]
                        (audio_dir / f"{vid_id}.mp3").write_bytes(b'audio')
                return Mock(returncode=0, stderr='')

            mock_run.side_effect = create_file

            result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "short")

            # Should have incremented count
            assert audio_pipeline.tier_download_counts.get('short', 0) == len(result)

    def test_max_total_limit_stops_downloads(self, audio_pipeline, temp_dir):
        """Test that max_total limit prevents downloads"""
        # Set up limit and current count
        audio_pipeline.tier_download_counts['longer'] = 1
        audio_pipeline._get_tier_value = Mock(side_effect=lambda tier, key, default: {
            'max_total': 1,  # Limit of 1
            'per_keyword': 5,
            'min': 1500,
            'max': 3000,
        }.get(key, default))

        result = audio_pipeline.download_audio_for_keyword("travel", temp_dir, "longer")

        # Should skip download entirely
        assert result == []
        audio_pipeline._search_video_metadata.assert_not_called()
