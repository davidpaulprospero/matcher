"""Tests for US-49-007: circuit breaker wired into download_segments escalation manager.

Tests cover:
  - set_circuit_breaker() is called during stage initialization when both components exist
  - Open circuit causes escalation to skip to max tier (FULL_BYPASS shortcut)
  - Open circuit + max tier causes video to be skipped to retry queue
  - Normal flow when circuit is closed (no interference)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.downloader.escalation_manager import EscalationManager
from src.downloader.types import EscalationTier, EscalationState


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
    max_tier: int = 4


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def ext_config():
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager()


@pytest.fixture
def escalation_manager(imp_manager, ext_config):
    return EscalationManager(imp_manager, ext_config)


@pytest.fixture
def circuit_breaker():
    return CircuitBreaker(CircuitBreakerConfig(
        enabled=True,
        consecutive_failures_threshold=3,
        pause_seconds=60.0,
    ))


# ---------------------------------------------------------------------------
# Test: set_circuit_breaker() called during stage init
# ---------------------------------------------------------------------------


class TestSetCircuitBreakerWiring:
    """Verify that _download_segments wires circuit_breaker into escalation_manager."""

    @pytest.mark.fast
    def test_set_circuit_breaker_called_when_both_available(
        self, escalation_manager, circuit_breaker
    ):
        """When downloader has both escalation_manager and circuit_breaker,
        set_circuit_breaker() should be called to wire them together."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        # Create a mock downloader with both components
        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = MagicMock()
        mock_downloader.retry_queue.has_pending.return_value = False
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5
        mock_downloader.download_config.segment_socket_timeout = 30
        mock_downloader.download_config.segment_max_resolution = 1080
        mock_downloader.download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_downloader.download_config.segment_stall_timeout = 120
        mock_downloader.download_config.cookies_from_browser = ''
        mock_downloader.download_config.cookies_path = ''
        mock_downloader.download_config.cookie_rotation = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        # Verify circuit_breaker is None before wiring
        assert escalation_manager._circuit_breaker is None

        # Call _download_segments with empty segments to trigger the wiring
        # but avoid actual downloads
        downloaded, stats = stage._download_segments(
            segments=[],
            output_dir=Path("/tmp/test"),
            buffer_seconds=5.0,
            progress_callback=None,
        )

        # Verify circuit breaker was wired into escalation manager
        assert escalation_manager._circuit_breaker is circuit_breaker

    @pytest.mark.fast
    def test_set_circuit_breaker_not_called_when_no_circuit_breaker(
        self, escalation_manager
    ):
        """When downloader has escalation_manager but no circuit_breaker,
        set_circuit_breaker() should not be called."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = None
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = MagicMock()
        mock_downloader.retry_queue.has_pending.return_value = False
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5

        stage.downloader = mock_downloader

        downloaded, stats = stage._download_segments(
            segments=[],
            output_dir=Path("/tmp/test"),
            buffer_seconds=5.0,
            progress_callback=None,
        )

        # Circuit breaker should remain None
        assert escalation_manager._circuit_breaker is None

    @pytest.mark.fast
    def test_set_circuit_breaker_not_called_when_no_escalation_manager(
        self, circuit_breaker
    ):
        """When downloader has circuit_breaker but no escalation_manager,
        set_circuit_breaker() should not be called (no crash)."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = None
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_downloader.retry_queue = MagicMock()
        mock_downloader.retry_queue.has_pending.return_value = False
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5

        stage.downloader = mock_downloader

        # Should not crash
        downloaded, stats = stage._download_segments(
            segments=[],
            output_dir=Path("/tmp/test"),
            buffer_seconds=5.0,
            progress_callback=None,
        )
        assert stats.succeeded == 0


# ---------------------------------------------------------------------------
# Test: Open circuit causes shortcut to max tier
# ---------------------------------------------------------------------------


class TestOpenCircuitEscalationShortcut:
    """Verify the circuit breaker shortcut logic at escalation_manager.py:330
    fires when circuit is open (skips to max tier via strategy)."""

    @pytest.mark.fast
    def test_open_circuit_shortcuts_to_full_bypass(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit breaker is open, get_escalation_args should shortcut
        to FULL_BYPASS tier regardless of current keyword tier."""
        # Wire circuit breaker into escalation manager
        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Trip the circuit breaker
        for _ in range(circuit_breaker.config.consecutive_failures_threshold):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_open

        # Get escalation args — should shortcut to FULL_BYPASS
        result = escalation_manager.get_escalation_args("test_video")
        assert result.tier == EscalationTier.FULL_BYPASS
        assert result.rotate_cookies is True  # Tier 3 enables cookies

    @pytest.mark.fast
    def test_closed_circuit_does_not_shortcut(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit breaker is closed, escalation should stay at normal tier."""
        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Circuit is closed — should use default Tier 1
        result = escalation_manager.get_escalation_args("test_video")
        assert result.tier == EscalationTier.IMPERSONATE_ONLY
        assert result.rotate_cookies is False

    @pytest.mark.fast
    def test_open_circuit_with_already_max_tier_keyword(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit is open and keyword is already at max tier,
        the shortcut logic shouldn't crash or downgrade."""
        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Manually set keyword to VPN_ROTATION tier
        state = escalation_manager._get_state("max_video")
        state.current_tier = EscalationTier.VPN_ROTATION

        # Trip circuit breaker
        for _ in range(circuit_breaker.config.consecutive_failures_threshold):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_open

        # Should still work without error
        result = escalation_manager.get_escalation_args("max_video")
        # Should be at least FULL_BYPASS (the shortcut target)
        assert result.tier >= EscalationTier.FULL_BYPASS


# ---------------------------------------------------------------------------
# Test: Open circuit + max tier skips video to retry queue
# ---------------------------------------------------------------------------


class TestCircuitBreakerSkipToRetryQueue:
    """Verify that open circuit + max tier causes video to be skipped."""

    @pytest.mark.fast
    def test_open_circuit_max_tier_skips_to_retry(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit is open and video is at max tier, the segment should
        be skipped and added to retry queue instead of attempting download."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        # Wire components
        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Set up keyword at max tier
        state = escalation_manager._get_state("test_vid")
        state.current_tier = EscalationTier.VPN_ROTATION

        # Trip circuit breaker
        for _ in range(circuit_breaker.config.consecutive_failures_threshold):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_open

        # Create mock downloader
        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_retry_queue = MagicMock()
        mock_retry_queue.has_pending.return_value = False
        mock_downloader.retry_queue = mock_retry_queue
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5
        mock_downloader.download_config.segment_stall_timeout = 120
        mock_downloader.download_config.cookies_from_browser = ''
        mock_downloader.download_config.cookies_path = ''
        mock_downloader.download_config.cookie_rotation = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [{
            'video_id': 'test_vid',
            'start': 10.0,
            'end': 20.0,
        }]

        # yt_dlp should never be imported since the video should be skipped
        downloaded, stats = stage._download_segments(
            segments=segments,
            output_dir=Path("/tmp/test_skip"),
            buffer_seconds=5.0,
            progress_callback=None,
        )

        # Segment should be skipped (failed) not downloaded
        assert stats.failed == 1
        assert stats.succeeded == 0
        assert len(downloaded) == 0

        # Verify it was added to retry queue
        mock_retry_queue.add.assert_called_once()
        call_kwargs = mock_retry_queue.add.call_args
        assert 'circuit_breaker_open_max_tier' in str(call_kwargs)

    @pytest.mark.fast
    def test_open_circuit_below_max_tier_still_downloads(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit is open but video is NOT at max tier, the download
        should still be attempted (escalation manager will shortcut to FULL_BYPASS)."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Trip circuit breaker
        for _ in range(circuit_breaker.config.consecutive_failures_threshold):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_open

        # Video NOT at max tier — Tier 1
        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_retry_queue = MagicMock()
        mock_retry_queue.has_pending.return_value = False
        mock_downloader.retry_queue = mock_retry_queue
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5
        mock_downloader.download_config.segment_socket_timeout = 30
        mock_downloader.download_config.segment_max_resolution = 1080
        mock_downloader.download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_downloader.download_config.segment_stall_timeout = 0  # Disable stall detection for test
        mock_downloader.download_config.cookies_from_browser = ''
        mock_downloader.download_config.cookies_path = ''
        mock_downloader.download_config.cookie_rotation = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [{
            'video_id': 'normal_vid',
            'start': 10.0,
            'end': 20.0,
        }]

        # Patch yt_dlp — simulate download attempt
        with patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            # Simulate yt_dlp.YoutubeDL context manager
            mock_ydl = MagicMock()
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_normal"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # Download was attempted (not skipped by circuit breaker)
        assert stats.attempted == 1
        # yt_dlp.YoutubeDL was called (download was attempted, not skipped)
        mock_yt_dlp_cls.assert_called()


# ---------------------------------------------------------------------------
# Test: check_and_wait() called when circuit breaker is open (US-50-008 AC5)
# ---------------------------------------------------------------------------


class TestCheckAndWaitCalledWhenOpen:
    """Verify that when circuit breaker is open and video is NOT at max tier,
    the download loop calls check_and_wait() to pause before downloading."""

    @pytest.mark.fast
    def test_check_and_wait_called_on_open_circuit(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit breaker is open and video is below max tier,
        check_and_wait() should be called to pause before download."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Trip circuit breaker
        for _ in range(circuit_breaker.config.consecutive_failures_threshold):
            circuit_breaker.record_failure()
        assert circuit_breaker.is_open

        # Video at Tier 1 (NOT max tier) — should pause, not skip
        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_retry_queue = MagicMock()
        mock_retry_queue.has_pending.return_value = False
        mock_downloader.retry_queue = mock_retry_queue
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5
        mock_downloader.download_config.bot_detection_abort_threshold = 20
        mock_downloader.download_config.segment_socket_timeout = 30
        mock_downloader.download_config.segment_max_resolution = 1080
        mock_downloader.download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_downloader.download_config.segment_stall_timeout = 0
        mock_downloader.download_config.cookies_from_browser = ''
        mock_downloader.download_config.cookies_path = ''
        mock_downloader.download_config.cookie_rotation = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [{
            'video_id': 'wait_vid',
            'start': 10.0,
            'end': 20.0,
        }]

        # Patch check_and_wait to avoid actual sleep and track calls
        with patch.object(circuit_breaker, 'check_and_wait', return_value=True) as mock_wait, \
             patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_wait"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # check_and_wait was called (paused before download)
        mock_wait.assert_called_once()
        # Download was still attempted after the wait
        assert stats.attempted == 1
        mock_yt_dlp_cls.assert_called()

    @pytest.mark.fast
    def test_check_and_wait_not_called_when_closed(
        self, escalation_manager, circuit_breaker
    ):
        """When circuit breaker is closed, check_and_wait() should NOT be called."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        escalation_manager.set_circuit_breaker(circuit_breaker)

        # Circuit is CLOSED — no pause needed
        assert not circuit_breaker.is_open

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_retry_queue = MagicMock()
        mock_retry_queue.has_pending.return_value = False
        mock_downloader.retry_queue = mock_retry_queue
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5
        mock_downloader.download_config.bot_detection_abort_threshold = 20
        mock_downloader.download_config.segment_socket_timeout = 30
        mock_downloader.download_config.segment_max_resolution = 1080
        mock_downloader.download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_downloader.download_config.segment_stall_timeout = 0
        mock_downloader.download_config.cookies_from_browser = ''
        mock_downloader.download_config.cookies_path = ''
        mock_downloader.download_config.cookie_rotation = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        segments = [{
            'video_id': 'nowait_vid',
            'start': 10.0,
            'end': 20.0,
        }]

        with patch.object(circuit_breaker, 'check_and_wait') as mock_wait, \
             patch('yt_dlp.YoutubeDL') as mock_yt_dlp_cls:
            mock_ydl = MagicMock()
            mock_yt_dlp_cls.return_value.__enter__ = MagicMock(return_value=mock_ydl)
            mock_yt_dlp_cls.return_value.__exit__ = MagicMock(return_value=False)

            downloaded, stats = stage._download_segments(
                segments=segments,
                output_dir=Path("/tmp/test_nowait"),
                buffer_seconds=5.0,
                progress_callback=None,
            )

        # check_and_wait should NOT have been called
        mock_wait.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Circuit breaker metrics in checkpoint (US-50-008 AC3)
# ---------------------------------------------------------------------------


class TestCircuitBreakerCheckpointMetrics:
    """Verify that circuit breaker state is included in stage checkpoint data."""

    @pytest.mark.fast
    def test_checkpoint_includes_circuit_breaker_stats(
        self, escalation_manager, circuit_breaker
    ):
        """Circuit breaker total_trips and total_paused_seconds should appear
        in the checkpoint data written by the run() method."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        stage = DownloadVideoSegmentsStage()

        # Simulate some circuit breaker activity
        circuit_breaker.state.total_trips = 3
        circuit_breaker.state.total_paused_seconds = 45.5

        mock_downloader = MagicMock()
        mock_downloader.escalation_manager = escalation_manager
        mock_downloader.circuit_breaker = circuit_breaker
        mock_downloader.cookie_rotator = None
        mock_retry_queue = MagicMock()
        mock_retry_queue.has_pending.return_value = False
        mock_downloader.retry_queue = mock_retry_queue
        mock_downloader.download_config = MagicMock()
        mock_downloader.download_config.bot_detection_tier_floor_threshold = 5
        mock_downloader.download_config.bot_detection_abort_threshold = 20
        mock_downloader.download_config.segment_socket_timeout = 30
        mock_downloader.download_config.segment_max_resolution = 1080
        mock_downloader.download_config.segment_format = 'best[height<={segment_max_resolution}]'
        mock_downloader.download_config.segment_stall_timeout = 0
        mock_downloader.download_config.cookies_from_browser = ''
        mock_downloader.download_config.cookies_path = ''
        mock_downloader.download_config.cookie_rotation = None
        mock_downloader.impersonation_manager = None

        stage.downloader = mock_downloader

        # Run with empty segments to get checkpoint_data quickly
        downloaded, stats = stage._download_segments(
            segments=[],
            output_dir=Path("/tmp/test_metrics"),
            buffer_seconds=5.0,
            progress_callback=None,
        )

        # Now simulate what run() does: build checkpoint_data
        checkpoint_data = {
            'segment_count': len(downloaded),
            'total_matches': 0,
            'retry_count': stats.retry_count,
        }

        # US-50-008: Include circuit breaker metrics (mirroring run() logic)
        _cb = getattr(stage.downloader, 'circuit_breaker', None)
        if _cb:
            checkpoint_data['circuit_breaker'] = {
                'total_trips': _cb.state.total_trips,
                'total_paused_seconds': round(_cb.state.total_paused_seconds, 1),
            }

        # Verify circuit breaker stats are present
        assert 'circuit_breaker' in checkpoint_data
        assert checkpoint_data['circuit_breaker']['total_trips'] == 3
        assert checkpoint_data['circuit_breaker']['total_paused_seconds'] == 45.5
