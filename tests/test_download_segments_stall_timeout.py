"""Tests for US-52-002: Stall detection wrapper for yt-dlp Python API download calls.

Tests cover:
  - ydl.download() wrapped in concurrent.futures timeout mechanism
  - Timeout value configurable via segment_stall_timeout (default 120s)
  - TimeoutError raised with 'stall_timeout' message on stall
  - Timeout errors classified as 'timeout' category (not 'network')
  - Timeout errors feed into existing retry logic (retry queue)
  - Successful downloads within timeout are not interrupted
"""

import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import (
    DownloadVideoSegmentsStage,
    classify_error_category,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_downloader(segment_stall_timeout=120, socket_timeout=30):
    """Create a mock downloader with required attributes for stall detection tests."""
    mock_dl = MagicMock()
    mock_dl.escalation_manager = None
    mock_dl.circuit_breaker = None
    mock_dl.cookie_rotator = None
    mock_dl.retry_queue = MagicMock()
    mock_dl.retry_queue.has_pending.return_value = False
    mock_dl.impersonation_manager = None

    mock_dl.download_config = MagicMock()
    mock_dl.download_config.bot_detection_tier_floor_threshold = 5
    mock_dl.download_config.bot_detection_abort_threshold = 0  # Disable abort
    mock_dl.download_config.network_failure_threshold = 3
    mock_dl.download_config.segment_socket_timeout = socket_timeout
    mock_dl.download_config.segment_max_resolution = 1080
    mock_dl.download_config.segment_format = 'best[height<={segment_max_resolution}]'
    mock_dl.download_config.segment_stall_timeout = segment_stall_timeout
    mock_dl.download_config.cookies_from_browser = ''
    mock_dl.download_config.cookies_path = ''
    mock_dl.download_config.cookie_rotation = None
    mock_dl.download_config.ffmpeg_location = ''
    return mock_dl


def _make_segments(count=1):
    """Generate a list of fake segments."""
    return [
        {'video_id': f'vid_{i}', 'start': 10.0, 'end': 20.0}
        for i in range(count)
    ]


def _make_blocking_download(block_seconds=3):
    """Create a blocking download function that auto-unblocks after block_seconds.

    Uses an event so the download thread unblocks quickly after the timeout
    fires, allowing ThreadPoolExecutor.shutdown(wait=True) to complete fast.
    """
    event = threading.Event()

    def _download(urls):
        event.wait(timeout=block_seconds)

    return _download, event


def _patch_yt_dlp(side_effect):
    """Return patch context for yt_dlp.YoutubeDL with download side_effect."""
    mock_yt_dlp_cls = patch('yt_dlp.YoutubeDL')
    return mock_yt_dlp_cls, side_effect


def _setup_ydl_mock(mock_yt_dlp_cls, side_effect):
    """Configure mock yt_dlp.YoutubeDL context manager with download side_effect."""
    mock_ydl = MagicMock()
    mock_ydl.download.side_effect = side_effect
    mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
    mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)
    return mock_ydl


# ---------------------------------------------------------------------------
# Test: Timeout triggers after configured duration
# ---------------------------------------------------------------------------

class TestStallTimeoutTriggers:
    """Verify ydl.download() timeout triggers when download stalls."""

    @pytest.mark.fast
    def test_timeout_triggers_on_blocking_download(self):
        """ydl.download() that blocks should be killed after
        segment_stall_timeout seconds, raising a TimeoutError."""
        stall_timeout = 1
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        segments = _make_segments(1)
        _download, event = _make_blocking_download(block_seconds=3)

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _download)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_stall_timeout"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        event.set()

        # Should have failed (timeout)
        assert stats.failed == 1
        assert stats.succeeded == 0
        assert len(downloaded) == 0

    @pytest.mark.fast
    def test_timeout_error_message_contains_stall_info(self):
        """The error logged on stall should reference 'stall' so it can
        be identified as a stall timeout."""
        stall_timeout = 1
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        segments = _make_segments(1)
        _download, event = _make_blocking_download(block_seconds=3)
        error_messages = []

        with patch('yt_dlp.YoutubeDL') as mock_cls, \
             patch('src.stages.download_segments.logger') as mock_logger:
            _setup_ydl_mock(mock_cls, _download)

            stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_stall_msg"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

            for call_args in mock_logger.warning.call_args_list:
                msg = str(call_args[0][0]) if call_args[0] else ''
                if 'stall' in msg.lower():
                    error_messages.append(msg)

        event.set()

        assert len(error_messages) > 0, (
            "Expected at least one stall-related warning log"
        )
        combined = ' '.join(error_messages).lower()
        assert 'stall' in combined

    @pytest.mark.fast
    def test_timeout_respects_configured_value(self):
        """Shorter timeout should produce faster failure detection."""
        stall_timeout = 1
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        segments = _make_segments(1)
        # Block for 2s — just above the 1s timeout
        _download, event = _make_blocking_download(block_seconds=2)

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _download)

            start_time = time.time()
            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_stall_config"),
                buffer_seconds=5.0,
                progress_callback=None,
            )
            elapsed = time.time() - start_time

        event.set()

        assert stats.failed == 1
        # Timeout at ~1s, thread finishes at ~2s, total < 4s
        assert elapsed < 4, f"Expected < 4s, got {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Test: Successful downloads are not interrupted
# ---------------------------------------------------------------------------

class TestSuccessfulDownloadNotInterrupted:
    """Verify that downloads completing within timeout are not killed."""

    @pytest.mark.fast
    def test_fast_download_succeeds_with_stall_timeout_enabled(self, tmp_path):
        """A download that completes quickly should succeed even when
        stall_timeout is configured."""
        stall_timeout = 10
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        segments = _make_segments(1)
        output_dir = tmp_path / "segments"
        output_dir.mkdir()

        def _fast_download(urls):
            video_id = segments[0]['video_id']
            start = max(0, segments[0]['start'] - 5.0)
            end = segments[0]['end'] + 5.0
            output_file = output_dir / f"{video_id}_{int(start)}_{int(end)}.mp4"
            output_file.write_bytes(b'fake video data')

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _fast_download)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        assert stats.succeeded == 1
        assert stats.failed == 0
        assert len(downloaded) == 1

    @pytest.mark.fast
    def test_download_with_moderate_duration_succeeds(self, tmp_path):
        """A download that takes a few seconds but finishes before timeout
        should succeed."""
        stall_timeout = 10
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        segments = _make_segments(1)
        output_dir = tmp_path / "segments"
        output_dir.mkdir()

        def _moderate_download(urls):
            time.sleep(0.5)  # Takes 0.5s, well within 10s timeout
            video_id = segments[0]['video_id']
            start = max(0, segments[0]['start'] - 5.0)
            end = segments[0]['end'] + 5.0
            output_file = output_dir / f"{video_id}_{int(start)}_{int(end)}.mp4"
            output_file.write_bytes(b'fake video data')

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _moderate_download)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        assert stats.succeeded == 1
        assert stats.failed == 0
        assert len(downloaded) == 1


# ---------------------------------------------------------------------------
# Test: Error classification for stall timeout
# ---------------------------------------------------------------------------

class TestStallTimeoutErrorClassification:
    """Verify stall timeout errors are classified correctly."""

    @pytest.mark.fast
    def test_stall_timeout_classified_as_timeout_category(self):
        """TimeoutError from stall detection should be classified as 'timeout',
        NOT as 'network' failure."""
        error_msg = (
            "ydl.download() stalled for 120.5s "
            "(segment_stall_timeout=120s)"
        )
        category = classify_error_category(error_msg)
        assert category == 'timeout', (
            f"Expected 'timeout' category, got '{category}'"
        )

    @pytest.mark.fast
    def test_stall_timeout_not_classified_as_network(self):
        """Stall timeout should NOT be classified as a network error."""
        error_msg = (
            "ydl.download() stalled for 300.0s "
            "(segment_stall_timeout=300s)"
        )
        category = classify_error_category(error_msg)
        assert category != 'network'

    @pytest.mark.fast
    def test_stall_timeout_not_classified_as_bot_detection(self):
        """Stall timeout should NOT be classified as bot detection."""
        error_msg = (
            "ydl.download() stalled for 60.0s "
            "(segment_stall_timeout=60s)"
        )
        category = classify_error_category(error_msg)
        assert category != 'bot_detection'


# ---------------------------------------------------------------------------
# Test: Timeout errors feed into retry queue
# ---------------------------------------------------------------------------

class TestStallTimeoutRetryIntegration:
    """Verify stall timeout errors are added to the retry queue."""

    @pytest.mark.fast
    def test_stall_timeout_added_to_retry_queue(self):
        """When ydl.download() stalls, the failed video should be added
        to the retry queue for batch retry."""
        stall_timeout = 1
        stage = DownloadVideoSegmentsStage()
        mock_dl = _make_mock_downloader(segment_stall_timeout=stall_timeout)
        stage.downloader = mock_dl

        segments = _make_segments(1)
        _download, event = _make_blocking_download(block_seconds=2)

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _download)

            stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_stall_retry"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        event.set()

        # Retry queue should have been called
        assert mock_dl.retry_queue.add.call_count == 1

        # Verify the retry queue entry has the timeout error category
        add_kwargs = mock_dl.retry_queue.add.call_args
        assert add_kwargs is not None
        if add_kwargs.kwargs:
            err_cat = add_kwargs.kwargs.get('error_category', '')
        else:
            err_cat = add_kwargs[1].get('error_category', '') if len(add_kwargs) > 1 else ''
        assert err_cat == 'timeout', (
            f"Expected error_category='timeout', got '{err_cat}'"
        )

    @pytest.mark.fast
    def test_stall_timeout_does_not_trigger_network_failure_counter(self):
        """Stall timeout should NOT increment the consecutive network failure
        counter, which would cause premature stage abort."""
        stall_timeout = 1
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        # 3 segments all stall. If counted as network failures,
        # NETWORK_FAILURE_THRESHOLD=3 would abort before attempting all.
        segments = _make_segments(3)

        def _blocking_download(urls):
            # Block just past the 1s timeout then release
            time.sleep(1.5)

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _blocking_download)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_stall_no_network_abort"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # All 3 should be attempted (not aborted as network failures)
        assert stats.attempted == 3, (
            f"Expected 3 attempted, got {stats.attempted} — "
            f"stall timeouts may be incorrectly counted as network failures"
        )
        assert stats.failed == 3

    @pytest.mark.fast
    def test_stall_timeout_recorded_in_error_categories(self):
        """Stall timeout should be recorded in the error_categories stats dict."""
        stall_timeout = 1
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=stall_timeout)

        segments = _make_segments(1)
        _download, event = _make_blocking_download(block_seconds=2)

        with patch('yt_dlp.YoutubeDL') as mock_cls:
            _setup_ydl_mock(mock_cls, _download)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_stall_categories"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        event.set()

        assert 'timeout' in stats.error_categories, (
            f"Expected 'timeout' in error_categories, got {stats.error_categories}"
        )
        assert stats.error_categories['timeout'] == 1


# ---------------------------------------------------------------------------
# Test: Disabled stall timeout (segment_stall_timeout=0)
# ---------------------------------------------------------------------------

class TestStallTimeoutDisabled:
    """Verify behavior when stall timeout is disabled (set to 0)."""

    @pytest.mark.fast
    def test_no_timeout_wrapper_when_disabled(self, tmp_path):
        """When segment_stall_timeout=0, ydl.download() should be called
        directly without the ThreadPoolExecutor wrapper."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(segment_stall_timeout=0)

        segments = _make_segments(1)
        output_dir = tmp_path / "segments"
        output_dir.mkdir()

        def _fast_download(urls):
            video_id = segments[0]['video_id']
            start = max(0, segments[0]['start'] - 5.0)
            end = segments[0]['end'] + 5.0
            output_file = output_dir / f"{video_id}_{int(start)}_{int(end)}.mp4"
            output_file.write_bytes(b'fake video data')

        with patch('yt_dlp.YoutubeDL') as mock_cls, \
             patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            _setup_ydl_mock(mock_cls, _fast_download)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # ThreadPoolExecutor should NOT have been used
        mock_executor.assert_not_called()
        assert stats.succeeded == 1


# ---------------------------------------------------------------------------
# Test: Config field defaults
# ---------------------------------------------------------------------------

class TestStallTimeoutConfigDefaults:
    """Verify the segment_stall_timeout config field exists and has correct default."""

    @pytest.mark.fast
    def test_download_config_has_segment_stall_timeout(self):
        """DownloadConfig dataclass should have segment_stall_timeout field
        with default value of 120 seconds."""
        from src.config.sections.download import DownloadConfig
        config = DownloadConfig()
        assert hasattr(config, 'segment_stall_timeout')
        assert config.segment_stall_timeout == 120

    @pytest.mark.fast
    def test_segment_stall_timeout_configurable(self):
        """segment_stall_timeout should be configurable via constructor."""
        from src.config.sections.download import DownloadConfig
        config = DownloadConfig(segment_stall_timeout=300)
        assert config.segment_stall_timeout == 300
