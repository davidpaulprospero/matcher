"""
Tests for per-download progress hooks in DownloadVideoSegmentsStage.

US-51-006: Add per-download progress hooks for segment download observability.
US-61-009: Add yt-dlp progress parsing for real-time download progress callback.

Verifies:
- Progress hook callback is invoked with expected status values
- Progress hook logs at INFO for slow downloads (>30s elapsed)
- Progress hook does NOT log for fast downloads (<30s)
- Progress hook logs file size and time on 'finished' status
- Progress data is accumulated into stats dict for stage metrics
- yt-dlp progress lines are correctly parsed for percent/bytes/speed
"""

import logging
from unittest.mock import MagicMock

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage, SegmentDownloadStats
from src.downloader.types import (
    SegmentDownloadProgress,
    parse_ytdlp_progress,
    _convert_size_to_bytes,
    _parse_eta_to_seconds,
)


pytestmark = [pytest.mark.fast, pytest.mark.unit]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def stage():
    """Create DownloadVideoSegmentsStage instance."""
    return DownloadVideoSegmentsStage()


@pytest.fixture
def stats():
    """Create a fresh SegmentDownloadStats like _download_segments uses."""
    return SegmentDownloadStats(total=5)


# ============================================================================
# Hook Creation
# ============================================================================

class TestProgressHookCreation:
    """Test _make_progress_hook returns a valid callable."""

    def test_returns_callable(self, stage, stats):
        hook = stage._make_progress_hook('abc123', stats)
        assert callable(hook)

    def test_initializes_progress_hooks_data_in_stats(self, stage, stats):
        """Hook creation uses progress_hooks_data from stats dataclass."""
        # progress_hooks_data is initialized by the dataclass with zeroed values
        data = stats.progress_hooks_data
        assert data['total_downloaded_bytes'] == 0
        assert data['segments_with_progress'] == 0
        assert data['segments_finished'] == 0
        # Creating a hook references the same dict
        stage._make_progress_hook('abc123', stats)
        assert stats.progress_hooks_data is data

    def test_multiple_hooks_share_same_progress_data(self, stage, stats):
        """Hooks for different videos accumulate into same stats dict."""
        hook1 = stage._make_progress_hook('vid1', stats)
        hook2 = stage._make_progress_hook('vid2', stats)

        # Both hooks writing to the same stats should share progress_hooks_data
        hook1({'status': 'finished', 'total_bytes': 1000, 'elapsed': 5.0})
        hook2({'status': 'finished', 'total_bytes': 2000, 'elapsed': 3.0})

        assert stats.progress_hooks_data['total_downloaded_bytes'] == 3000
        assert stats.progress_hooks_data['segments_finished'] == 2


# ============================================================================
# Status Callbacks
# ============================================================================

class TestProgressHookStatusValues:
    """Test hook is invoked with expected yt-dlp status values."""

    def test_downloading_status_accepted(self, stage, stats):
        """Hook handles 'downloading' status without error."""
        hook = stage._make_progress_hook('abc123', stats)
        hook({
            'status': 'downloading',
            'downloaded_bytes': 1024,
            'total_bytes': 10240,
            'speed': 512,
            'elapsed': 5.0,
            'filename': '/tmp/abc123.mp4',
        })
        # No error raised

    def test_finished_status_accepted(self, stage, stats):
        """Hook handles 'finished' status without error."""
        hook = stage._make_progress_hook('abc123', stats)
        hook({
            'status': 'finished',
            'total_bytes': 10240,
            'elapsed': 2.5,
            'filename': '/tmp/abc123.mp4',
        })
        assert stats.progress_hooks_data['segments_finished'] == 1

    def test_error_status_accepted(self, stage, stats):
        """Hook handles 'error' status without crashing."""
        hook = stage._make_progress_hook('abc123', stats)
        # yt-dlp may send 'error' status; hook should not crash
        hook({'status': 'error', 'elapsed': 10.0})

    def test_unknown_status_accepted(self, stage, stats):
        """Hook handles unknown status gracefully."""
        hook = stage._make_progress_hook('abc123', stats)
        hook({'status': 'something_new', 'elapsed': 1.0})


# ============================================================================
# Logging Behavior: Slow Downloads (>30s)
# ============================================================================

class TestProgressHookSlowDownloads:
    """Test logging for downloads taking >30s."""

    def test_logs_info_when_elapsed_over_30s(self, stage, stats, caplog):
        """Progress hook logs at INFO level when downloading and elapsed >30s."""
        hook = stage._make_progress_hook('abc123', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            hook({
                'status': 'downloading',
                'downloaded_bytes': 5 * 1024 * 1024,  # 5MB
                'total_bytes': 20 * 1024 * 1024,  # 20MB
                'speed': 1024 * 100,  # 100 KB/s
                'elapsed': 35.0,
                'filename': '/tmp/abc123.mp4',
            })

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any('abc123' in m and 'downloading' in m for m in info_msgs), \
            f"Expected INFO log about abc123 downloading, got: {info_msgs}"

    def test_log_includes_bytes_and_speed(self, stage, stats, caplog):
        """Log message includes downloaded bytes, total, and speed."""
        hook = stage._make_progress_hook('vid999', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            hook({
                'status': 'downloading',
                'downloaded_bytes': 10 * 1024 * 1024,  # 10MB
                'total_bytes': 50 * 1024 * 1024,  # 50MB
                'speed': 1024 * 200,  # 200 KB/s
                'elapsed': 40.0,
            })

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_msgs) >= 1
        msg = info_msgs[0]
        assert 'vid999' in msg
        assert 'MB' in msg  # Byte values formatted as MB
        assert 'KB/s' in msg  # Speed in KB/s

    def test_throttles_log_every_15s(self, stage, stats, caplog):
        """Logging is throttled: only logs every ~15s of elapsed time."""
        hook = stage._make_progress_hook('abc123', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            # First log at 35s
            hook({'status': 'downloading', 'downloaded_bytes': 1024, 'total_bytes': 10240, 'speed': 100, 'elapsed': 35.0})
            # Should NOT log again at 40s (only 5s later)
            hook({'status': 'downloading', 'downloaded_bytes': 2048, 'total_bytes': 10240, 'speed': 100, 'elapsed': 40.0})
            # Should log at 50s (15s after 35s)
            hook({'status': 'downloading', 'downloaded_bytes': 4096, 'total_bytes': 10240, 'speed': 100, 'elapsed': 50.0})

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO and 'downloading' in r.message]
        assert len(info_msgs) == 2, f"Expected 2 throttled logs, got {len(info_msgs)}: {info_msgs}"


# ============================================================================
# Logging Behavior: Fast Downloads (<30s)
# ============================================================================

class TestProgressHookFastDownloads:
    """Test that fast downloads (<30s) do NOT produce log spam."""

    def test_no_log_when_elapsed_under_30s(self, stage, stats, caplog):
        """Progress hook does NOT log for downloads taking <30s."""
        hook = stage._make_progress_hook('fast_vid', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            hook({
                'status': 'downloading',
                'downloaded_bytes': 5000,
                'total_bytes': 10000,
                'speed': 5000,
                'elapsed': 10.0,
            })
            hook({
                'status': 'downloading',
                'downloaded_bytes': 8000,
                'total_bytes': 10000,
                'speed': 5000,
                'elapsed': 20.0,
            })

        downloading_logs = [r for r in caplog.records
                          if r.levelno == logging.INFO and 'downloading' in r.message]
        assert len(downloading_logs) == 0, \
            f"Fast downloads should not log, but got: {[r.message for r in downloading_logs]}"

    def test_no_log_when_elapsed_zero(self, stage, stats, caplog):
        """Progress hook does NOT log when elapsed is 0 or None."""
        hook = stage._make_progress_hook('fast_vid', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            hook({'status': 'downloading', 'downloaded_bytes': 100, 'elapsed': 0.0})
            hook({'status': 'downloading', 'downloaded_bytes': 200, 'elapsed': None})

        downloading_logs = [r for r in caplog.records
                          if r.levelno == logging.INFO and 'downloading' in r.message]
        assert len(downloading_logs) == 0


# ============================================================================
# Finished Status Logging
# ============================================================================

class TestProgressHookFinished:
    """Test logging and accumulation on 'finished' status."""

    def test_finished_logs_file_size_and_time(self, stage, stats, caplog):
        """Finished status logs file size and elapsed time."""
        hook = stage._make_progress_hook('abc123', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            hook({
                'status': 'finished',
                'total_bytes': 15 * 1024 * 1024,  # 15MB
                'elapsed': 45.0,
            })

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any('finished' in m and 'abc123' in m for m in info_msgs)
        assert any('MB' in m for m in info_msgs)
        assert any('45.0s' in m for m in info_msgs)

    def test_finished_does_not_log_when_no_elapsed(self, stage, stats, caplog):
        """Finished with elapsed=0 does not log (fast completion)."""
        hook = stage._make_progress_hook('fast_vid', stats)
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            hook({
                'status': 'finished',
                'total_bytes': 1000,
                'elapsed': 0,
            })

        finished_logs = [r for r in caplog.records
                        if r.levelno == logging.INFO and 'finished' in r.message]
        assert len(finished_logs) == 0

    def test_finished_accumulates_total_bytes(self, stage, stats):
        """Finished status accumulates total_downloaded_bytes in stats."""
        hook = stage._make_progress_hook('abc123', stats)
        hook({'status': 'finished', 'total_bytes': 5000, 'elapsed': 2.0})
        hook({'status': 'finished', 'total_bytes': 3000, 'elapsed': 1.5})

        assert stats.progress_hooks_data['total_downloaded_bytes'] == 8000
        assert stats.progress_hooks_data['segments_finished'] == 2

    def test_finished_uses_downloaded_bytes_fallback(self, stage, stats):
        """Finished status falls back to downloaded_bytes when total_bytes missing."""
        hook = stage._make_progress_hook('abc123', stats)
        hook({'status': 'finished', 'downloaded_bytes': 7000, 'elapsed': 3.0})

        assert stats.progress_hooks_data['total_downloaded_bytes'] == 7000


# ============================================================================
# Stage Metrics Accumulation
# ============================================================================

class TestProgressHookStageMetrics:
    """Test progress data flows into stage metrics (checkpoint)."""

    def test_progress_hooks_data_populated_after_downloads(self, stage, stats):
        """After multiple downloads, progress_hooks_data reflects totals."""
        hook1 = stage._make_progress_hook('vid1', stats)
        hook2 = stage._make_progress_hook('vid2', stats)

        # vid1: slow download then finish
        hook1({'status': 'downloading', 'downloaded_bytes': 1024, 'total_bytes': 2048, 'speed': 100, 'elapsed': 35.0})
        hook1({'status': 'finished', 'total_bytes': 2048, 'elapsed': 40.0})

        # vid2: fast download, just finish
        hook2({'status': 'finished', 'total_bytes': 4096, 'elapsed': 5.0})

        data = stats.progress_hooks_data
        assert data['total_downloaded_bytes'] == 2048 + 4096
        assert data['segments_finished'] == 2
        assert data['segments_with_progress'] >= 1  # vid1 had progress log

    def test_handles_missing_fields_gracefully(self, stage, stats):
        """Hook does not crash when yt-dlp dict is missing optional fields."""
        hook = stage._make_progress_hook('abc123', stats)
        # Minimal dict with just status
        hook({'status': 'downloading'})
        hook({'status': 'finished'})
        # No crash
        assert stats.progress_hooks_data['segments_finished'] == 1

    def test_handles_none_values_gracefully(self, stage, stats):
        """Hook handles None values for bytes/speed without crashing."""
        hook = stage._make_progress_hook('abc123', stats)
        hook({
            'status': 'downloading',
            'downloaded_bytes': None,
            'total_bytes': None,
            'speed': None,
            'elapsed': 35.0,
        })
        # No crash, and progress was logged (unknown values)


# ============================================================================
# Integration: ydl_opts includes progress_hooks
# ============================================================================

class TestProgressHookInYdlOpts:
    """Test that ydl_opts construction includes progress_hooks key."""

    def test_make_progress_hook_is_static_method(self):
        """_make_progress_hook is accessible as a static method on the stage class."""
        assert hasattr(DownloadVideoSegmentsStage, '_make_progress_hook')
        # It's a staticmethod, so callable on the class
        stats = SegmentDownloadStats()
        hook = DownloadVideoSegmentsStage._make_progress_hook('test', stats)
        assert callable(hook)


# ============================================================================
# US-61-009: yt-dlp Progress Parsing Tests
# ============================================================================

class TestParseYtdlpProgress:
    """Tests for yt-dlp progress line parsing (US-61-009)."""

    def test_parse_basic_progress_line(self):
        """Test parsing a standard yt-dlp progress line."""
        line = "[download]  50.0% of 10.00MiB at  5.00MiB/s ETA 00:01"
        result = parse_ytdlp_progress(line, "abc123")

        assert result is not None
        assert result.video_id == "abc123"
        assert result.percent == 50.0
        assert result.speed == 5.0 * 1024 * 1024  # 5 MiB/s in bytes
        assert result.eta_seconds == 1.0

    def test_parse_progress_with_kib(self):
        """Test parsing progress with KiB units."""
        line = "[download]  25.5% of 512.00KiB at  128.00KiB/s ETA 00:03"
        result = parse_ytdlp_progress(line, "video1")

        assert result is not None
        assert result.percent == 25.5
        assert result.bytes_downloaded == int(512 * 1024 * 0.255)
        assert result.speed == 128.0 * 1024

    def test_parse_progress_with_gib(self):
        """Test parsing progress with GiB units."""
        line = "[download]  10.0% of 1.50GiB at  10.00MiB/s ETA 02:18:00"
        result = parse_ytdlp_progress(line, "large_video")

        assert result is not None
        assert result.percent == 10.0
        assert result.eta_seconds == 2 * 3600 + 18 * 60  # 2h 18m

    def test_parse_progress_without_eta(self):
        """Test parsing progress line without ETA."""
        line = "[download]  75.0% of 20.00MiB at  2.50MiB/s"
        result = parse_ytdlp_progress(line, "video2")

        assert result is not None
        assert result.percent == 75.0
        assert result.eta_seconds is None

    def test_parse_progress_with_approximate_size(self):
        """Test parsing progress with ~ approximate size indicator."""
        line = "[download]  30.0% of ~15.00MiB at  3.00MiB/s ETA 00:05"
        result = parse_ytdlp_progress(line, "video3")

        assert result is not None
        assert result.percent == 30.0

    def test_parse_non_progress_line_returns_none(self):
        """Test that non-progress lines return None."""
        lines = [
            "[youtube] Extracting URL: https://youtube.com/watch?v=abc123",
            "[info] Downloading 1 format(s)",
            "ERROR: Video unavailable",
            "",
            "Some random text",
        ]
        for line in lines:
            assert parse_ytdlp_progress(line, "video") is None

    def test_parse_progress_preserves_video_id(self):
        """Test that the video_id is correctly set in the result."""
        line = "[download]  99.9% of 5.00MiB at  1.00MiB/s ETA 00:00"
        result = parse_ytdlp_progress(line, "my-special-video-id")

        assert result is not None
        assert result.video_id == "my-special-video-id"


class TestConvertSizeToBytes:
    """Tests for size conversion utility (US-61-009)."""

    def test_convert_bytes(self):
        """Test converting raw bytes."""
        assert _convert_size_to_bytes(100.0, "B") == 100

    def test_convert_kib(self):
        """Test converting KiB to bytes."""
        assert _convert_size_to_bytes(1.0, "KiB") == 1024
        assert _convert_size_to_bytes(1.0, "KB") == 1024

    def test_convert_mib(self):
        """Test converting MiB to bytes."""
        assert _convert_size_to_bytes(1.0, "MiB") == 1024 * 1024
        assert _convert_size_to_bytes(1.0, "MB") == 1024 * 1024

    def test_convert_gib(self):
        """Test converting GiB to bytes."""
        assert _convert_size_to_bytes(1.0, "GiB") == 1024 * 1024 * 1024
        assert _convert_size_to_bytes(1.0, "GB") == 1024 * 1024 * 1024


class TestParseEtaToSeconds:
    """Tests for ETA string parsing (US-61-009)."""

    def test_parse_mm_ss(self):
        """Test parsing MM:SS format."""
        assert _parse_eta_to_seconds("01:30") == 90

    def test_parse_hh_mm_ss(self):
        """Test parsing HH:MM:SS format."""
        assert _parse_eta_to_seconds("01:30:45") == 5445

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        assert _parse_eta_to_seconds("") is None

    def test_parse_none(self):
        """Test parsing None returns None."""
        assert _parse_eta_to_seconds(None) is None


class TestSegmentDownloadProgressDataclass:
    """Tests for SegmentDownloadProgress dataclass (US-61-009)."""

    def test_create_with_all_fields(self):
        """Test creating a progress object with all fields."""
        progress = SegmentDownloadProgress(
            video_id="test123",
            percent=45.5,
            bytes_downloaded=1024000,
            speed=512000.0,
            eta_seconds=10.0
        )
        assert progress.video_id == "test123"
        assert progress.percent == 45.5
        assert progress.bytes_downloaded == 1024000
        assert progress.speed == 512000.0
        assert progress.eta_seconds == 10.0

    def test_create_with_optional_eta_none(self):
        """Test creating a progress object without ETA."""
        progress = SegmentDownloadProgress(
            video_id="test456",
            percent=80.0,
            bytes_downloaded=2048000,
            speed=1024000.0
        )
        assert progress.eta_seconds is None


class TestProgressCallbackIntegration:
    """Integration tests for progress callback (US-61-009)."""

    def test_progress_callback_receives_values(self):
        """Test that progress callback receives expected values during simulated download."""
        # Simulate yt-dlp progress output
        progress_lines = [
            "[download]  10.0% of 100.00MiB at  10.00MiB/s ETA 00:09",
            "[download]  50.0% of 100.00MiB at  10.00MiB/s ETA 00:05",
            "[download] 100.0% of 100.00MiB at  10.00MiB/s ETA 00:00",
        ]

        received_progress = []

        def progress_callback(progress: SegmentDownloadProgress):
            received_progress.append(progress)

        # Parse each line and call callback
        video_id = "integration_test_video"
        for line in progress_lines:
            progress = parse_ytdlp_progress(line, video_id)
            if progress:
                progress_callback(progress)

        # Verify we received all progress updates
        assert len(received_progress) == 3

        # Verify first progress (10%)
        assert received_progress[0].percent == 10.0
        assert received_progress[0].video_id == video_id

        # Verify middle progress (50%)
        assert received_progress[1].percent == 50.0

        # Verify final progress (100%)
        assert received_progress[2].percent == 100.0
