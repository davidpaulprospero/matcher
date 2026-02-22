"""Tests for US-49-008: bot-detection abort threshold in download_segments stage.

Tests cover:
  - Stage aborts after threshold consecutive bot-detection errors
  - Checkpoint is saved before aborting
  - Abort does not fire when below threshold
  - Abort respects configurable threshold from DownloadConfig
  - Counter resets on successful download (no false abort)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.stages.download_segments import (
    DownloadVideoSegmentsStage,
    BOT_DETECTION_ABORT_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_downloader(bot_abort_threshold=10, bot_floor_threshold=5):
    """Create a mock downloader with required attributes."""
    mock_dl = MagicMock()
    mock_dl.escalation_manager = None
    mock_dl.circuit_breaker = None
    mock_dl.cookie_rotator = None
    mock_dl.retry_queue = MagicMock()
    mock_dl.retry_queue.has_pending.return_value = False
    mock_dl.impersonation_manager = None

    mock_dl.download_config = MagicMock()
    mock_dl.download_config.bot_detection_tier_floor_threshold = bot_floor_threshold
    mock_dl.download_config.bot_detection_abort_threshold = bot_abort_threshold
    mock_dl.download_config.network_failure_threshold = 3
    mock_dl.download_config.segment_socket_timeout = 30
    mock_dl.download_config.segment_max_resolution = 1080
    mock_dl.download_config.segment_format = 'best[height<={segment_max_resolution}]'
    mock_dl.download_config.segment_stall_timeout = 0  # Disable stall detection
    mock_dl.download_config.cookies_from_browser = ''
    mock_dl.download_config.cookies_path = ''
    mock_dl.download_config.cookie_rotation = None
    mock_dl.download_config.ffmpeg_location = ''
    return mock_dl


def _make_segments(count):
    """Generate a list of fake segments."""
    return [
        {'video_id': f'vid_{i}', 'start': 10.0, 'end': 20.0}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Test: Stage aborts after threshold bot-detection errors
# ---------------------------------------------------------------------------


class TestBotDetectionAbort:
    """Verify stage aborts when bot-detection errors exceed the threshold."""

    @pytest.mark.fast
    def test_aborts_after_threshold_bot_errors(self):
        """Stage should abort after N consecutive bot-detection errors,
        skipping remaining segments."""
        threshold = 5
        total_segments = 10  # More than threshold
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        segments = _make_segments(total_segments)

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            # Every download raises a 403 Forbidden error
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_abort"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # Should have attempted exactly `threshold` downloads then aborted
        assert stats.attempted == threshold
        assert stats.failed == threshold
        assert stats.succeeded == 0
        assert len(downloaded) == 0

    @pytest.mark.fast
    def test_does_not_abort_below_threshold(self):
        """Stage should complete all segments when bot errors are below threshold."""
        threshold = 20  # High threshold, won't trigger
        total_segments = 5
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        segments = _make_segments(total_segments)

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            # Every download raises a 403 error but below threshold
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_no_abort"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # All segments attempted because threshold was not hit
        assert stats.attempted == total_segments
        assert stats.failed == total_segments

    @pytest.mark.fast
    def test_abort_disabled_when_threshold_zero(self):
        """When bot_detection_abort_threshold=0, bot-detection abort is disabled."""
        total_segments = 15
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=0)

        segments = _make_segments(total_segments)

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_disabled"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # All segments attempted — abort was disabled
        assert stats.attempted == total_segments

    @pytest.mark.fast
    def test_counter_resets_on_success_no_false_abort(self):
        """A successful download resets the bot-detection counter,
        preventing false aborts after intermittent errors."""
        threshold = 3
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        # 2 bot errors, 1 success (resets counter), 2 more bot errors = no abort
        segments = _make_segments(5)

        call_count = [0]

        def _side_effect(urls):
            call_count[0] += 1
            idx = call_count[0]
            if idx <= 2:
                raise Exception("ERROR: [youtube] vid: HTTP Error 403: Forbidden")
            elif idx == 3:
                # Simulate success by writing a file
                output_path = Path("/tmp/test_bot_reset") / f"vid_2_5_25.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b'fake video data')
                return None
            else:
                raise Exception("ERROR: [youtube] vid: HTTP Error 403: Forbidden")

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = _side_effect
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_reset"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # All 5 segments attempted (counter reset at index 3 prevents abort)
        assert stats.attempted == 5
        # 4 failures + 1 success (or file-not-found depends on path)
        assert stats.failed >= 2  # At minimum the first 2 and last 2

    @pytest.mark.fast
    def test_default_constant_value(self):
        """BOT_DETECTION_ABORT_THRESHOLD module constant defaults to 10."""
        assert BOT_DETECTION_ABORT_THRESHOLD == 10


# ---------------------------------------------------------------------------
# Test: Checkpoint saved before abort
# ---------------------------------------------------------------------------


class TestBotDetectionAbortCheckpoint:
    """Verify checkpoint is saved before stage aborts on bot-detection."""

    @pytest.mark.fast
    def test_checkpoint_saved_before_abort(self):
        """progress_callback should be called right before the abort break."""
        threshold = 3
        total_segments = 10
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        segments = _make_segments(total_segments)
        progress_calls = []

        def mock_progress(current, total, downloaded):
            progress_calls.append((current, total, len(downloaded)))

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: Sign in to confirm you're not a bot"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_checkpoint"),
                buffer_seconds=5.0,
                progress_callback=mock_progress,
            )

        assert stats.attempted == threshold

        # progress_callback should have been called for each attempt + the final
        # abort checkpoint call. The last call should be at idx == threshold.
        assert len(progress_calls) > 0
        # The final progress call should be the abort checkpoint at index = threshold
        last_call = progress_calls[-1]
        assert last_call[0] == threshold  # current index
        assert last_call[1] == total_segments  # total

    @pytest.mark.fast
    def test_checkpoint_not_called_when_no_callback(self):
        """When progress_callback is None, abort still works without error."""
        threshold = 2
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        segments = _make_segments(5)

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            # Should not crash even with progress_callback=None
            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_no_cb"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        assert stats.attempted == threshold


# ---------------------------------------------------------------------------
# Test: Bot-detection abort logged as 'bot_detection_abort' category
# ---------------------------------------------------------------------------


class TestBotDetectionAbortCategory:
    """US-52-007: Verify bot-detection abort is logged as a stage-level failure
    with 'bot_detection_abort' category in error_categories stats."""

    @pytest.mark.fast
    def test_abort_records_bot_detection_abort_category(self):
        """When the stage aborts due to bot-detection threshold, the stats
        error_categories dict should contain 'bot_detection_abort'."""
        threshold = 3
        total_segments = 10
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        segments = _make_segments(total_segments)

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats, _ = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_abort_cat"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        assert stats.attempted == threshold
        assert 'bot_detection_abort' in stats.error_categories, (
            f"Expected 'bot_detection_abort' in error_categories, "
            f"got {stats.error_categories}"
        )
        assert stats.error_categories['bot_detection_abort'] == 1

    @pytest.mark.fast
    def test_abort_message_includes_count_and_cookie_guidance(self):
        """The abort log message should include the count of bot detections
        and recommend checking cookie configuration."""
        threshold = 3
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(bot_abort_threshold=threshold)

        segments = _make_segments(10)
        error_messages = []

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls, \
             patch('src.stages.download_segments.logger') as mock_logger:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_bot_abort_msg"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

            for call_args in mock_logger.error.call_args_list:
                msg = str(call_args[0][0]) if call_args[0] else ''
                error_messages.append(msg)

        combined = ' '.join(error_messages).lower()
        # Verify count is mentioned
        assert str(threshold) in ' '.join(error_messages), (
            f"Expected bot count '{threshold}' in error messages"
        )
        # Verify cookie guidance is included
        assert 'cookie' in combined, (
            "Expected 'cookie' guidance in abort error messages"
        )


# ---------------------------------------------------------------------------
# Test: Mixed bot-detection + success resets counter (no false abort)
# ---------------------------------------------------------------------------


class TestBotDetectionMixedSuccessReset:
    """US-52-007: Verify that successful downloads between bot-detection errors
    reset the counter, preventing false aborts."""

    @pytest.mark.fast
    def test_mixed_bot_and_success_resets_counter(self):
        """Pattern: 2 bot errors, 1 success, 2 bot errors → no abort
        (threshold=3). The success at position 3 resets the counter."""
        import tempfile, shutil
        threshold = 3
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_abort_threshold=threshold,
            bot_floor_threshold=50,  # High so tier floor doesn't interfere
        )

        segments = _make_segments(5)

        tmp_dir = Path(tempfile.mkdtemp(prefix="test_mixed_bot_"))
        try:
            call_count = [0]

            def _side_effect(urls):
                call_count[0] += 1
                idx = call_count[0]
                if idx <= 2:
                    raise Exception("ERROR: [youtube] vid: HTTP Error 403: Forbidden")
                elif idx == 3:
                    # Success: write the file for vid_2 (start=10-5=5, end=20+5=25)
                    output_path = tmp_dir / "vid_2_5_25.mp4"
                    output_path.write_bytes(b'fake video data')
                    return None
                else:
                    raise Exception("ERROR: [youtube] vid: HTTP Error 403: Forbidden")

            with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
                mock_ydl = MagicMock()
                mock_ydl.download.side_effect = _side_effect
                mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
                mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

                downloaded, stats, _ = stage._download_segments(
                    segments=segments,
                    output_dir=tmp_dir,
                    buffer_seconds=5.0,
                    progress_callback=None,
                )

            # All 5 segments should have been attempted (no abort at threshold=3
            # because the success at position 3 reset the counter)
            assert stats.attempted == 5, (
                f"Expected all 5 segments attempted, got {stats.attempted} — "
                f"counter was not reset on success"
            )
            assert stats.succeeded >= 1
            # bot_detection_abort should NOT be in error_categories
            assert 'bot_detection_abort' not in stats.error_categories, (
                "bot_detection_abort should not appear when abort didn't fire"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
