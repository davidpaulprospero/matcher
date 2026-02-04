"""Tests for US-50-006: global bot-detection counter with tier floor propagation.

Tests cover:
  - After N consecutive bot-detection errors across different video IDs,
    the tier floor is set to VPN_ROTATION tier
  - A successful download resets the consecutive bot-detection counter to zero
  - Bot-detection errors do not reset the network failure counter
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage
from src.downloader.types import EscalationTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_downloader(bot_floor_threshold=5, bot_abort_threshold=50):
    """Create a mock downloader with escalation manager and required attributes.

    bot_abort_threshold is set high by default so tier-floor tests are not
    cut short by the abort logic.
    """
    mock_dl = MagicMock()
    mock_dl.circuit_breaker = None
    mock_dl.cookie_rotator = None
    mock_dl.retry_queue = MagicMock()
    mock_dl.retry_queue.has_pending.return_value = False
    mock_dl.impersonation_manager = None

    # Real-ish escalation manager mock that tracks set_tier_floor calls.
    # get_escalation_args returns None so _build_ydl_opts skips escalation
    # and the tier comparison at line 615 doesn't crash on MagicMock > int.
    esc_mgr = MagicMock()
    esc_mgr.set_tier_floor = MagicMock()
    esc_mgr.clear_tier_floor = MagicMock()
    esc_mgr.record_failure = MagicMock()
    esc_mgr.record_success = MagicMock()
    esc_mgr.get_escalation_args = MagicMock(return_value=None)
    mock_dl.escalation_manager = esc_mgr

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
    """Generate a list of fake segments with distinct video IDs."""
    return [
        {'video_id': f'vid_{i}', 'start': 10.0, 'end': 20.0}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# AC4: After N consecutive bot-detection errors across different video IDs,
#      the tier floor is set to VPN_ROTATION tier
# ---------------------------------------------------------------------------


class TestTierFloorPropagation:
    """Verify tier floor is set after consecutive bot-detection errors."""

    @pytest.mark.fast
    def test_tier_floor_set_after_threshold_bot_errors(self):
        """After bot_detection_tier_floor_threshold consecutive bot errors
        across different video IDs, set_tier_floor(VPN_ROTATION) is called."""
        floor_threshold = 3
        segment_count = 5  # More than threshold, less than abort
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_floor_threshold=floor_threshold,
            bot_abort_threshold=50,
        )

        segments = _make_segments(segment_count)
        esc_mgr = stage.downloader.escalation_manager

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_tier_floor"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # set_tier_floor should have been called with VPN_ROTATION
        esc_mgr.set_tier_floor.assert_called_with(EscalationTier.VPN_ROTATION)

    @pytest.mark.fast
    def test_tier_floor_not_set_below_threshold(self):
        """Tier floor should NOT be set when bot errors are below threshold."""
        floor_threshold = 10
        segment_count = 5  # Below threshold
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_floor_threshold=floor_threshold,
            bot_abort_threshold=50,
        )

        segments = _make_segments(segment_count)
        esc_mgr = stage.downloader.escalation_manager

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_no_tier_floor"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # set_tier_floor should NOT have been called
        esc_mgr.set_tier_floor.assert_not_called()

    @pytest.mark.fast
    def test_tier_floor_uses_different_video_ids(self):
        """Counter increments across different video IDs (stage-level, not per-video)."""
        floor_threshold = 3
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_floor_threshold=floor_threshold,
            bot_abort_threshold=50,
        )

        # Each segment has a UNIQUE video_id
        segments = [
            {'video_id': 'alpha', 'start': 10.0, 'end': 20.0},
            {'video_id': 'bravo', 'start': 10.0, 'end': 20.0},
            {'video_id': 'charlie', 'start': 10.0, 'end': 20.0},
            {'video_id': 'delta', 'start': 10.0, 'end': 20.0},
        ]
        esc_mgr = stage.downloader.escalation_manager

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = Exception(
                "ERROR: [youtube] vid: Sign in to confirm you're not a bot"
            )
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_tier_floor_ids"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # Tier floor set at the 3rd error (threshold=3) even though
        # each error came from a different video_id
        esc_mgr.set_tier_floor.assert_called_with(EscalationTier.VPN_ROTATION)


# ---------------------------------------------------------------------------
# AC5: Successful download resets the consecutive bot-detection counter
# ---------------------------------------------------------------------------


class TestBotCounterResetOnSuccess:
    """Verify that a successful download resets the bot-detection counter."""

    @pytest.mark.fast
    def test_success_resets_counter_and_clears_tier_floor(self):
        """A successful download should reset consecutive_bot_detections to 0
        and clear the tier floor so subsequent downloads don't start elevated."""
        import tempfile, shutil
        floor_threshold = 3
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_floor_threshold=floor_threshold,
            bot_abort_threshold=50,
        )

        # 3 bot errors (triggers tier floor) → 1 success (resets) → 3 more
        # bot errors (should trigger tier floor again, proving it was cleared)
        segments = _make_segments(7)
        esc_mgr = stage.downloader.escalation_manager

        # Use a real temp dir to avoid stale cached files from previous runs
        tmp_dir = Path(tempfile.mkdtemp(prefix="test_counter_reset_"))
        try:
            call_count = [0]

            def _side_effect(urls):
                call_count[0] += 1
                idx = call_count[0]
                if idx <= 3:
                    raise Exception("ERROR: [youtube] vid: HTTP Error 403: Forbidden")
                elif idx == 4:
                    # Simulate success: write the file that the download loop
                    # expects for vid_3 (start=10-5=5, end=20+5=25)
                    output_path = tmp_dir / "vid_3_5_25.mp4"
                    output_path.write_bytes(b'fake video data')
                    return None
                else:
                    raise Exception("ERROR: [youtube] vid: HTTP Error 403: Forbidden")

            with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
                mock_ydl = MagicMock()
                mock_ydl.download.side_effect = _side_effect
                mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
                mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

                downloaded, stats = stage._download_segments(
                    segments=segments,
                    output_dir=tmp_dir,
                    buffer_seconds=5.0,
                    progress_callback=None,
                )

            # All 7 segments should have been attempted (no abort at threshold 50)
            assert stats.attempted == 7

            # clear_tier_floor should have been called once on the success (idx=4)
            esc_mgr.clear_tier_floor.assert_called_once()

            # set_tier_floor should have been called TWICE:
            #   1st time: after errors 1-3 (first batch hits threshold)
            #   2nd time: after errors 5-7 (second batch hits threshold after reset)
            assert esc_mgr.set_tier_floor.call_count == 2
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# AC6: Bot-detection errors do not reset the network failure counter
# ---------------------------------------------------------------------------


class TestBotErrorsDoNotResetNetworkCounter:
    """Verify that bot-detection errors are counted separately from network
    failures, and bot errors do NOT reset the network failure counter."""

    @pytest.mark.fast
    def test_bot_errors_preserve_network_failure_count(self):
        """Network failure counter should NOT be reset by intervening bot
        errors. Network errors → bot error → more network errors should
        still trigger the network failure abort."""
        from src.stages.download_segments import NETWORK_FAILURE_THRESHOLD

        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_floor_threshold=50,   # High so tier floor doesn't interfere
            bot_abort_threshold=50,   # High so bot abort doesn't interfere
        )

        # We need enough segments to reach the network failure threshold.
        # Pattern: (NETWORK_FAILURE_THRESHOLD - 1) network errors, then 1
        # bot error, then 1 more network error. If bot errors incorrectly
        # reset the counter, the final network error won't trigger abort.
        net_before = NETWORK_FAILURE_THRESHOLD - 1
        total_segments = net_before + 2  # +1 bot error + 1 network error

        segments = _make_segments(total_segments)

        call_count = [0]

        def _side_effect(urls):
            call_count[0] += 1
            idx = call_count[0]
            if idx <= net_before:
                # Network failure errors
                raise Exception(
                    "ERROR: unable to download webpage: "
                    "<urlopen error [Errno 11001] getaddrinfo failed>"
                )
            elif idx == net_before + 1:
                # Bot-detection error (should NOT reset network counter)
                raise Exception(
                    "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
                )
            else:
                # One more network failure — should trigger abort
                raise Exception(
                    "ERROR: unable to download webpage: "
                    "<urlopen error [Errno 11001] getaddrinfo failed>"
                )

        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_ydl.download.side_effect = _side_effect
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_net_bot_isolation"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # The bot error at position (net_before + 1) should NOT have reset
        # the network counter. So when we hit the next network error, the
        # counter should be at NETWORK_FAILURE_THRESHOLD and trigger abort.
        #
        # If bot errors incorrectly reset the counter, all segments would
        # be attempted (no abort).
        #
        # Expected: net_before network errors + 1 bot error + 1 network
        # error = net_before + 2 attempted, then abort.
        assert stats.attempted == net_before + 2
        assert stats.failed == net_before + 2
        assert stats.succeeded == 0
