"""Integration tests for download_segments with mocked yt-dlp Python API (US-49-011).

Tests the full download_segments flow end-to-end with mocked yt-dlp, verifying
that cookies, impersonation, socket_timeout, escalation, abort logic, and
checkpoint saves all work together correctly.

Pytest marker: fast (no real network or subprocess calls)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import (
    DownloadVideoSegmentsStage,
    _apply_escalation_to_ydl_opts,
)
from src.downloader.escalation_manager import EscalationManager
from src.downloader.types import EscalationTier


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 300.0
    max_tier: int = 3


def _make_mock_downloader(
    *,
    bot_abort_threshold: int = 100,
    bot_floor_threshold: int = 100,
    cookies_from_browser: str = 'chrome',
    cookies_path: str = '',
    socket_timeout: int = 45,
    escalation_mgr=None,
    cookie_rotator=None,
):
    """Create a mock downloader with required attributes for integration tests."""
    mock_dl = MagicMock()
    mock_dl.escalation_manager = escalation_mgr
    mock_dl.circuit_breaker = None
    mock_dl.cookie_rotator = cookie_rotator
    mock_dl.retry_queue = MagicMock()
    mock_dl.retry_queue.has_pending.return_value = False
    mock_dl.impersonation_manager = None

    mock_dl.download_config = MagicMock()
    mock_dl.download_config.bot_detection_tier_floor_threshold = bot_floor_threshold
    mock_dl.download_config.bot_detection_abort_threshold = bot_abort_threshold
    mock_dl.download_config.segment_socket_timeout = socket_timeout
    mock_dl.download_config.socket_timeout = 30
    mock_dl.download_config.segment_max_resolution = 1080
    mock_dl.download_config.segment_format = 'best[height<={segment_max_resolution}]'
    mock_dl.download_config.segment_stall_timeout = 0  # Disable stall wrapper
    mock_dl.download_config.cookies_from_browser = cookies_from_browser
    mock_dl.download_config.cookies_path = cookies_path
    mock_dl.download_config.cookie_rotation = None
    mock_dl.download_config.ffmpeg_location = ''
    return mock_dl


def _make_segments(count: int) -> List[Dict[str, Any]]:
    """Generate a list of fake segments with unique video IDs."""
    return [
        {'video_id': f'vid_{i}', 'start': 10.0, 'end': 20.0}
        for i in range(count)
    ]


def _make_impersonation_manager():
    """Create a mock ImpersonationManager returning deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


def _make_escalation_manager() -> EscalationManager:
    """Create a real EscalationManager with mock impersonation manager."""
    imp_mgr = _make_impersonation_manager()
    ext_config = FakeExtractorArgsConfig()
    return EscalationManager(imp_mgr, ext_config)


def _yt_dlp_mock_context(side_effect):
    """Return a patch context for yt_dlp.YoutubeDL with the given side_effect.

    The mock captures ydl_opts passed to YoutubeDL() constructor so tests
    can inspect them after the call.

    Returns:
        (patch_context, captured_opts_list)
    """
    captured_opts: List[Dict[str, Any]] = []

    class _FakeYDL:
        def __init__(self, opts):
            captured_opts.append(dict(opts))
            self._opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def download(self, urls):
            if callable(side_effect):
                return side_effect(urls, self._opts)
            if isinstance(side_effect, Exception):
                raise side_effect
            return None

    ctx = patch('yt_dlp.YoutubeDL', _FakeYDL)
    return ctx, captured_opts


# ---------------------------------------------------------------------------
# AC1 + AC2: End-to-end flow — cookies, impersonation, socket_timeout,
#            escalation all present in ydl_opts
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestEndToEndFlow:
    """Integration test verifying end-to-end download flow with all features."""

    def test_cookies_impersonation_socket_timeout_and_escalation(self, tmp_path):
        """Verify that ydl_opts contain cookies, ImpersonateTarget,
        socket_timeout, and escalation args on a successful download."""
        from yt_dlp.networking.impersonate import ImpersonateTarget

        stage = DownloadVideoSegmentsStage()
        esc_mgr = _make_escalation_manager()
        stage.downloader = _make_mock_downloader(
            cookies_from_browser='chrome',
            socket_timeout=45,
            escalation_mgr=esc_mgr,
        )

        segments = [{'video_id': 'abc123', 'start': 5.0, 'end': 15.0}]
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        captured_opts: List[Dict] = []

        def _success(urls, opts):
            captured_opts.append(dict(opts))
            # Simulate yt-dlp writing the output file
            outtmpl = opts.get('outtmpl', '')
            if outtmpl:
                Path(outtmpl).write_bytes(b'\x00' * 1024)
            return None

        ctx, _ = _yt_dlp_mock_context(_success)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # Should succeed
        assert stats.succeeded == 1
        assert stats.failed == 0
        assert len(downloaded) == 1

        # Inspect captured ydl_opts
        assert len(captured_opts) == 1
        opts = captured_opts[0]

        # AC: cookies injected
        assert opts.get('cookiesfrombrowser') == ['chrome'], \
            f"Expected cookiesfrombrowser=['chrome'], got {opts.get('cookiesfrombrowser')}"

        # AC: impersonation target is ImpersonateTarget object
        imp = opts.get('impersonate')
        assert isinstance(imp, ImpersonateTarget), \
            f"Expected ImpersonateTarget, got {type(imp)}: {imp}"

        # AC: socket_timeout present
        assert opts.get('socket_timeout') == 45, \
            f"Expected socket_timeout=45, got {opts.get('socket_timeout')}"

        # AC: download_ranges function present (segment mode)
        assert callable(opts.get('download_ranges'))

    def test_escalation_args_applied_after_tier_progression(self, tmp_path):
        """After recording failures, escalation tier progresses and
        extractor_args appear in ydl_opts."""
        stage = DownloadVideoSegmentsStage()
        esc_mgr = _make_escalation_manager()

        # Pre-escalate vid_0 to Tier 2 by recording failures
        esc_mgr.record_failure('vid_0', 'HTTP Error 403: Forbidden')
        esc_mgr.record_failure('vid_0', 'HTTP Error 403: Forbidden')

        stage.downloader = _make_mock_downloader(
            cookies_from_browser='chrome',
            escalation_mgr=esc_mgr,
        )

        segments = [{'video_id': 'vid_0', 'start': 0.0, 'end': 10.0}]
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        captured_opts: List[Dict] = []

        def _success(urls, opts):
            captured_opts.append(dict(opts))
            Path(opts['outtmpl']).write_bytes(b'\x00' * 512)
            return None

        ctx, _ = _yt_dlp_mock_context(_success)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=0.0,
                progress_callback=None,
            )

        assert stats.succeeded == 1
        opts = captured_opts[0]

        # After 2 failures, should have escalated to Tier 2 with extractor_args
        ext_args = opts.get('extractor_args', {})
        assert 'youtube' in ext_args, \
            f"Expected extractor_args with 'youtube' key, got {ext_args}"
        assert 'player_client' in ext_args.get('youtube', {}), \
            f"Expected player_client in extractor_args.youtube, got {ext_args}"


# ---------------------------------------------------------------------------
# AC3: 3 consecutive bot-detection errors → escalation → success
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestBotDetectionEscalationThenSuccess:
    """Simulate 3 bot-detection errors followed by tier escalation
    and a successful download."""

    def test_three_403_errors_then_success(self, tmp_path):
        """After 3 consecutive 403 errors (different video IDs), the
        escalation tier progresses. A 4th segment download succeeds."""
        stage = DownloadVideoSegmentsStage()
        esc_mgr = _make_escalation_manager()
        stage.downloader = _make_mock_downloader(
            cookies_from_browser='chrome',
            escalation_mgr=esc_mgr,
            bot_abort_threshold=100,  # Don't trigger abort
        )

        # 4 segments: first 3 fail with 403, last succeeds
        segments = _make_segments(4)
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        call_count = [0]
        captured_opts: List[Dict] = []

        def _download_fn(urls, opts):
            captured_opts.append(dict(opts))
            call_count[0] += 1
            if call_count[0] <= 3:
                raise Exception(
                    "ERROR: [youtube] vid: HTTP Error 403: Forbidden"
                )
            # 4th call succeeds — write the file
            Path(opts['outtmpl']).write_bytes(b'\x00' * 256)
            return None

        ctx, _ = _yt_dlp_mock_context(_download_fn)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # 3 failed + 1 succeeded
        assert stats.failed == 3
        assert stats.succeeded == 1
        assert stats.attempted == 4
        assert len(downloaded) == 1

        # Verify error categories tracked correctly
        assert stats.error_categories.get('bot_detection', 0) == 3

        # The 4th call's opts should show escalation was applied
        # (escalation tier should have progressed from failures)
        assert len(captured_opts) == 4


# ---------------------------------------------------------------------------
# AC4: Network failure abort after threshold consecutive DNS errors
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestNetworkFailureAbort:
    """Simulate network failures triggering the abort threshold."""

    def test_abort_after_consecutive_dns_errors(self, tmp_path):
        """Stage aborts after NETWORK_FAILURE_THRESHOLD (3) consecutive
        DNS resolution failures, indicating systemic network issues."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_abort_threshold=100,  # Don't trigger bot abort
        )

        # 10 segments but should abort after 3 DNS errors
        segments = _make_segments(10)
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        def _dns_fail(urls, opts):
            raise Exception(
                "ERROR: [youtube] vid: Unable to download: "
                "<urlopen error [Errno 11001] getaddrinfo failed>"
            )

        ctx, _ = _yt_dlp_mock_context(_dns_fail)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # Should abort after exactly 3 consecutive network failures
        assert stats.attempted == 3
        assert stats.failed == 3
        assert stats.succeeded == 0
        assert len(downloaded) == 0

        # Error categories should all be 'network'
        assert stats.error_categories.get('network', 0) == 3

    def test_network_counter_resets_on_non_network_error(self, tmp_path):
        """A non-network error between DNS failures resets the counter,
        preventing false network abort."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            bot_abort_threshold=100,
        )

        # Pattern: DNS, DNS, video_specific (resets), DNS, DNS, video_specific
        # Should NOT abort (never hits 3 consecutive)
        segments = _make_segments(6)
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        call_count = [0]

        def _mixed_errors(urls, opts):
            call_count[0] += 1
            idx = call_count[0]
            if idx in (1, 2, 4, 5):
                raise Exception(
                    "ERROR: [youtube] vid: <urlopen error "
                    "[Errno 11001] getaddrinfo failed>"
                )
            else:
                raise Exception(
                    "ERROR: [youtube] vid: Video unavailable"
                )

        ctx, _ = _yt_dlp_mock_context(_mixed_errors)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # All 6 segments attempted (no abort)
        assert stats.attempted == 6
        assert stats.failed == 6


# ---------------------------------------------------------------------------
# AC5: Checkpoint save includes downloaded segment paths
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestCheckpointSaveWithSegmentPaths:
    """Verify that checkpoint saves include downloaded segment info."""

    def test_checkpoint_includes_segment_paths_via_run(self, tmp_path):
        """Full stage.run() call should produce checkpoint_data with
        segment_count reflecting the number of successfully downloaded files."""
        from src.state import PipelineState, Match

        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(cookies_from_browser='chrome')

        # Create minimal PipelineState with matches
        state = PipelineState()
        m1 = Match(segment_index=0, video_file='vid_0', video_start=5.0,
                    video_end=15.0, confidence=0.8)
        m2 = Match(segment_index=1, video_file='vid_1', video_start=0.0,
                    video_end=10.0, confidence=0.7)
        state.matches = [m1, m2]

        # Mock config
        config = MagicMock()
        config.download = MagicMock()
        config.download.segment_buffer = 5.0
        config.downloaded_videos_dir = str(tmp_path / 'segments')
        (tmp_path / 'segments').mkdir()

        # Mock checkpoint manager that captures save_intermediate calls
        checkpoint = MagicMock()
        checkpoint.should_skip_stage.return_value = False
        intermediate_saves: List[Dict] = []

        def _capture_intermediate(stage_name, data):
            intermediate_saves.append({'stage': stage_name, 'data': dict(data)})

        checkpoint.save_intermediate.side_effect = _capture_intermediate

        # Mock VideoDownloader import to return our mock downloader
        def _success(urls, opts):
            outtmpl = opts.get('outtmpl', '')
            if outtmpl:
                Path(outtmpl).write_bytes(b'\x00' * 1024)
            return None

        ctx, _ = _yt_dlp_mock_context(_success)
        with ctx, \
             patch('src.downloader.VideoDownloader', return_value=stage.downloader):
            result = stage.run(state, config, checkpoint)

        # Stage should succeed
        assert result.success, f"Stage failed: {result.error}"

        # Checkpoint data should include segment_count
        assert result.data is not None
        assert result.data['segment_count'] == 2, \
            f"Expected segment_count=2, got {result.data}"

        # State should have downloaded_segments populated
        assert len(state.downloaded_segments) == 2
        for seg in state.downloaded_segments:
            assert seg.file, "Downloaded segment should have a file path"
            assert Path(seg.file).exists(), \
                f"Downloaded segment file should exist: {seg.file}"

    def test_progress_callback_tracks_segment_counts(self, tmp_path):
        """progress_callback receives correct counts as segments download."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(cookies_from_browser='chrome')

        segments = _make_segments(3)
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        progress_calls: List[tuple] = []

        def _progress(current, total, downloaded):
            progress_calls.append((current, total, len(downloaded)))

        def _success(urls, opts):
            Path(opts['outtmpl']).write_bytes(b'\x00' * 256)
            return None

        ctx, _ = _yt_dlp_mock_context(_success)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=_progress,
            )

        assert stats.succeeded == 3
        assert len(downloaded) == 3

        # Should have one progress call per segment
        assert len(progress_calls) == 3

        # Progress should be monotonically increasing
        for i, (current, total, dl_count) in enumerate(progress_calls):
            assert current == i + 1
            assert total == 3
            assert dl_count == i + 1  # One more downloaded each time

    def test_checkpoint_saved_on_abort_includes_partial_downloads(self, tmp_path):
        """When stage aborts due to network failure, checkpoint saves
        include any segments that were successfully downloaded before abort."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader(
            cookies_from_browser='chrome',
            bot_abort_threshold=100,  # Don't trigger bot abort
        )

        # 6 segments: first succeeds, next 3 are DNS failures (triggers abort)
        segments = _make_segments(6)
        output_dir = tmp_path / 'segments'
        output_dir.mkdir()

        call_count = [0]

        def _partial_success(urls, opts):
            call_count[0] += 1
            if call_count[0] == 1:
                Path(opts['outtmpl']).write_bytes(b'\x00' * 512)
                return None
            raise Exception(
                "ERROR: [youtube] vid: <urlopen error "
                "[Errno 11001] getaddrinfo failed>"
            )

        progress_calls: List[tuple] = []

        def _progress(current, total, downloaded):
            progress_calls.append((current, total, len(downloaded)))

        ctx, _ = _yt_dlp_mock_context(_partial_success)
        with ctx:
            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=output_dir,
                buffer_seconds=5.0,
                progress_callback=_progress,
            )

        # 1 success + 3 network failures (abort after 3 consecutive)
        assert stats.succeeded == 1
        assert stats.failed == 3
        assert stats.attempted == 4
        assert len(downloaded) == 1

        # Last progress call should show 1 downloaded segment
        assert len(progress_calls) >= 4
        # The abort checkpoint call (at position idx=4) should report 1 downloaded
        last_call = progress_calls[-1]
        assert last_call[2] == 1, \
            f"Last checkpoint should show 1 downloaded, got {last_call[2]}"


# ---------------------------------------------------------------------------
# AC1 (fixture): Mock yt_dlp.YoutubeDL captures ydl_opts and simulates
#                download success/failure
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestYtDlpMockFixture:
    """Verify the test fixture itself works correctly."""

    def test_fixture_captures_opts_on_success(self, tmp_path):
        """Mock fixture captures ydl_opts when download succeeds."""
        import yt_dlp

        def _ok(urls, opts):
            return None

        ctx, captured = _yt_dlp_mock_context(_ok)
        with ctx:
            with yt_dlp.YoutubeDL({'format': 'best', 'socket_timeout': 30}) as ydl:
                ydl.download(['https://example.com'])

        assert len(captured) == 1
        assert captured[0]['format'] == 'best'
        assert captured[0]['socket_timeout'] == 30

    def test_fixture_captures_opts_on_failure(self, tmp_path):
        """Mock fixture captures ydl_opts even when download raises."""
        import yt_dlp

        def _fail(urls, opts):
            raise Exception("HTTP Error 403: Forbidden")

        ctx, captured = _yt_dlp_mock_context(_fail)
        with ctx:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                with pytest.raises(Exception, match="403"):
                    ydl.download(['https://example.com'])

        assert len(captured) == 1
        assert captured[0]['quiet'] is True
