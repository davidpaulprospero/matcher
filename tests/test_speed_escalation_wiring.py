"""Integration tests for AudioFirstPipeline speed escalation wiring (Sprint 14 US-002).

Tests that AudioFirstPipeline._check_speed_escalation() is correctly wired:
- Calls escalation_manager.record_slow_speed() when speed_tracker detects signal
- Is called after download_audio_for_keyword() completes
- Is called after each video in download_video_segments()
- Handles speed_tracker=None gracefully (no AttributeError)
- Handles escalation_manager=None gracefully (no crash)

Pytest marker: fast
"""

import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from src.downloader.audio_first import AudioFirstPipeline
from src.downloader.escalation_manager import EscalationManager
from src.downloader.speed_tracker import RateLimitSignal
from src.downloader.types import EscalationTier, MergedSegment, DownloadedSegment


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
    cooldown_seconds: float = 0.0
    max_tier: int = 3


def _make_mock_config():
    """Create a mock Config with audio_first settings."""
    config = Mock()

    audio_first = Mock()
    audio_first.enabled = True
    audio_first.audio_quality = 5
    audio_first.fallback_full_video = False  # Disable fallback for simpler tests

    download = Mock()
    download.audio_first = audio_first
    download.download_timeouts = {'short': 60, 'medium': 120, 'long': 300}
    download.max_keyword_len = 50
    download.max_retries = 3
    download.retry_delay = 0.01
    download.ffmpeg_location = ''
    download.checkpoint_interval = 10
    download.llm_title_filter = None

    config.download = download
    return config


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = ["--impersonate", "Chrome-136:Macos-15"]
    mgr.target_count = 5
    return mgr


def _make_search_results(videos):
    """Wrap video dicts in a SearchResult-like object."""
    result = Mock()
    result.videos = videos
    return result


def _make_rate_limit_signal(detected=True):
    """Create a RateLimitSignal with given detection state."""
    return RateLimitSignal(
        detected=detected,
        consecutive_slow_count=3 if detected else 0,
        recent_speeds=[0.05, 0.03, 0.04] if detected else [5.0, 6.0],
        threshold=0.1,
        message="Rate limit detected" if detected else "Normal speed",
    )


def _make_speed_tracker(detected=True, avg_speed=0.04):
    """Create a mock speed tracker that returns a rate limit signal."""
    tracker = MagicMock()
    tracker.config = Mock()
    tracker.config.enabled = True
    tracker.detect_rate_limit_signals.return_value = _make_rate_limit_signal(detected)
    tracker.get_average_speed_mbps.return_value = avg_speed
    return tracker


def _make_pipeline(
    escalation_manager=None,
    cookie_rotator=None,
    impersonation_manager=None,
    speed_tracker=None,
    config=None,
):
    """Create an AudioFirstPipeline with mock deps."""
    if config is None:
        config = _make_mock_config()

    def get_tier_value(tier, key, default):
        values = {
            'max_total': 0,
            'per_keyword': 2,
            'min': 20,
            'max': 600,
        }
        return values.get(key, default)

    pipeline = AudioFirstPipeline(
        config=config,
        get_tier_value_func=get_tier_value,
        search_metadata_func=Mock(return_value=_make_search_results([])),
        filter_titles_func=Mock(side_effect=lambda vids, *a: vids),
        cleanup_partial_func=Mock(),
        tier_download_counts={},
        lock=threading.Lock(),
        cookie_rotator=cookie_rotator,
        impersonation_manager=impersonation_manager,
        escalation_manager=escalation_manager,
        speed_tracker=speed_tracker,
    )
    return pipeline


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def imp_mgr():
    return _make_impersonation_manager()


@pytest.fixture
def esc_mgr(imp_mgr):
    return EscalationManager(imp_mgr, FakeExtractorArgsConfig())


# ---------------------------------------------------------------------------
# AC 1: _check_speed_escalation() calls escalation_manager.record_slow_speed()
#        when speed_tracker detects rate limit signal
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestCheckSpeedEscalationCallsRecordSlowSpeed:
    """_check_speed_escalation() triggers record_slow_speed on signal detection."""

    def test_calls_record_slow_speed_when_signal_detected(self, esc_mgr):
        """When speed_tracker.detect_rate_limit_signals().detected is True,
        escalation_manager.record_slow_speed(keyword, speed_mbps) is called."""
        speed_tracker = _make_speed_tracker(detected=True, avg_speed=0.04)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )

        with patch.object(esc_mgr, 'record_slow_speed') as mock_record:
            pipeline._check_speed_escalation("travel")

        mock_record.assert_called_once_with("travel", speed_mbps=0.04)

    @pytest.mark.fast
    def test_no_call_when_signal_not_detected(self, esc_mgr):
        """When detect_rate_limit_signals().detected is False,
        record_slow_speed is NOT called."""
        speed_tracker = _make_speed_tracker(detected=False)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )

        with patch.object(esc_mgr, 'record_slow_speed') as mock_record:
            pipeline._check_speed_escalation("travel")

        mock_record.assert_not_called()

    @pytest.mark.fast
    def test_passes_correct_keyword_and_speed(self, esc_mgr):
        """Verifies keyword and average speed are forwarded correctly."""
        speed_tracker = _make_speed_tracker(detected=True, avg_speed=1.23)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )

        with patch.object(esc_mgr, 'record_slow_speed') as mock_record:
            pipeline._check_speed_escalation("nature documentary")

        mock_record.assert_called_once_with("nature documentary", speed_mbps=1.23)

    @pytest.mark.fast
    def test_speed_tracker_detect_invoked(self, esc_mgr):
        """detect_rate_limit_signals() is invoked exactly once per call."""
        speed_tracker = _make_speed_tracker(detected=True)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )

        pipeline._check_speed_escalation("kw")
        speed_tracker.detect_rate_limit_signals.assert_called_once()


# ---------------------------------------------------------------------------
# AC 2: _check_speed_escalation() is called after download_audio_for_keyword()
#        — verify invocation count
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSpeedEscalationCalledAfterAudioDownload:
    """_check_speed_escalation() is invoked after download_audio_for_keyword()."""

    @pytest.mark.integration
    def test_called_once_after_audio_download(self, temp_dir, esc_mgr):
        """_check_speed_escalation is called exactly once after
        download_audio_for_keyword() completes, regardless of video count."""
        speed_tracker = _make_speed_tracker(detected=False)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
            {'id': 'vid2', 'title': 'Test2', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid2'},
        ]))

        def mock_run(cmd, **kwargs):
            cmd_str = ' '.join(str(c) for c in cmd)
            audio_dir = temp_dir / "test_s_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            for vid in ['vid1', 'vid2']:
                if vid in cmd_str:
                    (audio_dir / f"{vid}.mp3").write_bytes(b'audio')
            return Mock(returncode=0, stderr='')

        with patch('subprocess.run', side_effect=mock_run), \
             patch.object(pipeline, '_check_speed_escalation', wraps=pipeline._check_speed_escalation) as mock_check:
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        # Called exactly once at end of method (not per video)
        mock_check.assert_called_once_with("test")

    @pytest.mark.integration
    def test_called_even_when_no_downloads_succeed(self, temp_dir, esc_mgr):
        """_check_speed_escalation is called even when all downloads fail."""
        speed_tracker = _make_speed_tracker(detected=False)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
        ]))

        def mock_run(cmd, **kwargs):
            return Mock(returncode=1, stderr='Connection refused')

        with patch('subprocess.run', side_effect=mock_run), \
             patch.object(pipeline, '_check_speed_escalation', wraps=pipeline._check_speed_escalation) as mock_check:
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        mock_check.assert_called_once_with("test")

    @pytest.mark.integration
    def test_called_with_correct_keyword(self, temp_dir, esc_mgr):
        """_check_speed_escalation receives the keyword passed to download_audio_for_keyword."""
        speed_tracker = _make_speed_tracker(detected=False)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )
        # No search results means quick return but _check_speed_escalation still called
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'v1', 'title': 'T', 'duration': 60,
             'webpage_url': 'https://youtube.com/watch?v=v1'},
        ]))

        def mock_run(cmd, **kwargs):
            audio_dir = temp_dir / "deepoce_s_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / "v1.mp3").write_bytes(b'audio')
            return Mock(returncode=0, stderr='')

        with patch('subprocess.run', side_effect=mock_run), \
             patch.object(pipeline, '_check_speed_escalation', wraps=pipeline._check_speed_escalation) as mock_check:
            pipeline.download_audio_for_keyword("deep ocean", temp_dir, "short")

        mock_check.assert_called_once_with("deep ocean")


# ---------------------------------------------------------------------------
# AC 3: _check_speed_escalation() is called after segment download in
#        download_video_segments() — verify invocation with correct keyword
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSpeedEscalationCalledAfterSegmentDownload:
    """_check_speed_escalation() is invoked per video in download_video_segments()."""

    @pytest.mark.integration
    def test_called_per_video_in_segments(self, temp_dir, esc_mgr):
        """_check_speed_escalation is called once per video_id group."""
        speed_tracker = _make_speed_tracker(detected=False)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            ),
            MergedSegment(
                video_id="vid2",
                video_url="https://youtube.com/watch?v=vid2",
                start_time=5.0,
                end_time=30.0,
                original_matches=[],
                keyword="nature",
            ),
        ]

        def mock_run(cmd, **kwargs):
            return Mock(returncode=0)

        with patch('subprocess.run', side_effect=mock_run), \
             patch('src.downloader.segment_utils.rename_segments_with_timing',
                   return_value=[None]), \
             patch.object(pipeline, '_check_speed_escalation', wraps=pipeline._check_speed_escalation) as mock_check:
            pipeline.download_video_segments(segments, temp_dir)

        # Called once per video_id (2 videos)
        assert mock_check.call_count == 2

    @pytest.mark.integration
    def test_called_with_correct_keyword_per_video(self, temp_dir, esc_mgr):
        """Each call uses the keyword from the corresponding segment group."""
        speed_tracker = _make_speed_tracker(detected=False)
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
        )

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            ),
            MergedSegment(
                video_id="vid2",
                video_url="https://youtube.com/watch?v=vid2",
                start_time=5.0,
                end_time=30.0,
                original_matches=[],
                keyword="nature",
            ),
        ]

        def mock_run(cmd, **kwargs):
            return Mock(returncode=0)

        call_keywords = []

        original_check = pipeline._check_speed_escalation

        def tracking_check(keyword):
            call_keywords.append(keyword)
            return original_check(keyword)

        with patch('subprocess.run', side_effect=mock_run), \
             patch('src.downloader.segment_utils.rename_segments_with_timing',
                   return_value=[None]), \
             patch.object(pipeline, '_check_speed_escalation', side_effect=tracking_check):
            pipeline.download_video_segments(segments, temp_dir)

        assert "travel" in call_keywords
        assert "nature" in call_keywords

    @pytest.mark.integration
    def test_called_even_on_download_failure(self, temp_dir, esc_mgr):
        """_check_speed_escalation is called after segment download fails."""
        speed_tracker = _make_speed_tracker(detected=False)
        config = _make_mock_config()
        config.download.max_retries = 1  # Fail fast
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=speed_tracker,
            config=config,
        )

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            ),
        ]

        def mock_run(cmd, **kwargs):
            return Mock(returncode=1, stderr='Non-retryable error')

        with patch('subprocess.run', side_effect=mock_run), \
             patch.object(pipeline, '_check_speed_escalation', wraps=pipeline._check_speed_escalation) as mock_check:
            pipeline.download_video_segments(segments, temp_dir)

        mock_check.assert_called_once_with("travel")


# ---------------------------------------------------------------------------
# AC 4: _check_speed_escalation() handles speed_tracker=None gracefully
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSpeedEscalationWithNullSpeedTracker:
    """_check_speed_escalation() handles speed_tracker=None without errors."""

    def test_no_error_when_speed_tracker_none(self, esc_mgr):
        """No AttributeError when speed_tracker is None."""
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=None,
        )

        # Should return silently, no exception
        pipeline._check_speed_escalation("test_keyword")

    @pytest.mark.fast
    def test_record_slow_speed_not_called_when_tracker_none(self, esc_mgr):
        """record_slow_speed is never called when speed_tracker is None."""
        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=None,
        )

        with patch.object(esc_mgr, 'record_slow_speed') as mock_record:
            pipeline._check_speed_escalation("test_keyword")

        mock_record.assert_not_called()

    @pytest.mark.fast
    def test_no_error_when_speed_tracker_disabled(self, esc_mgr):
        """No error when speed_tracker exists but config.enabled is False."""
        tracker = MagicMock()
        tracker.config = Mock()
        tracker.config.enabled = False

        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            speed_tracker=tracker,
        )

        # Should return silently
        pipeline._check_speed_escalation("test_keyword")
        # detect_rate_limit_signals should NOT be called
        tracker.detect_rate_limit_signals.assert_not_called()


# ---------------------------------------------------------------------------
# AC 5: _check_speed_escalation() handles escalation_manager=None gracefully
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSpeedEscalationWithNullEscalationManager:
    """_check_speed_escalation() handles escalation_manager=None without crash."""

    def test_no_error_when_escalation_manager_none(self):
        """No crash when escalation_manager is None, even with active signal."""
        speed_tracker = _make_speed_tracker(detected=True, avg_speed=0.05)
        pipeline = _make_pipeline(
            escalation_manager=None,
            speed_tracker=speed_tracker,
        )

        # Should return silently, no AttributeError
        pipeline._check_speed_escalation("test_keyword")

    @pytest.mark.fast
    def test_detect_called_but_record_skipped_when_no_manager(self):
        """With speed_tracker active but no escalation_manager, detect may or
        may not be called but no record_slow_speed happens."""
        speed_tracker = _make_speed_tracker(detected=True, avg_speed=0.05)
        pipeline = _make_pipeline(
            escalation_manager=None,
            speed_tracker=speed_tracker,
        )

        pipeline._check_speed_escalation("test_keyword")

        # The method checks escalation_manager before calling detect,
        # so detect_rate_limit_signals should NOT be called
        speed_tracker.detect_rate_limit_signals.assert_not_called()

    @pytest.mark.fast
    def test_both_none_no_crash(self):
        """No error when both speed_tracker and escalation_manager are None."""
        pipeline = _make_pipeline(
            escalation_manager=None,
            speed_tracker=None,
        )

        # Should return silently
        pipeline._check_speed_escalation("any_keyword")

    @pytest.mark.integration
    def test_integration_audio_download_with_none_managers(self, temp_dir):
        """download_audio_for_keyword() works when both managers are None."""
        pipeline = _make_pipeline(
            escalation_manager=None,
            speed_tracker=None,
        )
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
        ]))

        def mock_run(cmd, **kwargs):
            audio_dir = temp_dir / "test_s_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / "vid1.mp3").write_bytes(b'audio')
            return Mock(returncode=0, stderr='')

        with patch('subprocess.run', side_effect=mock_run):
            result = pipeline.download_audio_for_keyword("test", temp_dir, "short")

        # Should complete without errors
        assert len(result) == 1
