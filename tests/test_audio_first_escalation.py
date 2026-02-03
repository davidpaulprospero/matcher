"""Integration tests for AudioFirstPipeline escalation wiring (Sprint 14 US-001).

Tests that AudioFirstPipeline correctly integrates with EscalationManager:
- download_audio_for_keyword() passes escalation args to yt-dlp
- download_audio_for_keyword() handles rotate_cookies flag
- download_video_segments() applies escalation args
- _download_full_video_fallback() escalates independently
- Retry loop triggers tier progression on consecutive 403s

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
from src.downloader.escalation_manager import (
    EscalationManager,
    EscalationResult,
    is_escalation_trigger,
)
from src.downloader.types import EscalationTier, MergedSegment, DownloadedSegment
from src.state import AudioDownload


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
    cooldown_seconds: float = 0.0  # No cooldown for fast tests
    max_tier: int = 3


def _make_mock_config():
    """Create a mock Config with audio_first settings."""
    config = Mock()

    audio_first = Mock()
    audio_first.enabled = True
    audio_first.audio_quality = 5
    audio_first.fallback_full_video = True

    download = Mock()
    download.audio_first = audio_first
    download.download_timeouts = {'short': 60, 'medium': 120, 'long': 300}
    download.max_keyword_len = 50
    download.max_retries = 3
    download.retry_delay = 0.01  # Fast retries for tests
    download.ffmpeg_location = ''
    download.checkpoint_interval = 10
    download.llm_title_filter = None
    download.stall_timeout = 60

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


def _make_mock_popen(returncode=0, stderr_text=''):
    """Create a mock subprocess.Popen that completes immediately.

    Used for segment/fallback download tests that use Popen + stall detection.
    """
    proc = Mock()
    proc.pid = 12345
    proc.returncode = returncode
    proc.poll = Mock(return_value=0)  # Process finished immediately
    proc.wait = Mock()
    proc.kill = Mock()
    proc.stdout = Mock()
    proc.stdout.readline = Mock(return_value='')
    proc.stdout.closed = False
    proc.stdout.close = Mock()
    proc.stderr = Mock()
    proc.stderr.readline = Mock(return_value='')
    proc.stderr.closed = False
    proc.stderr.close = Mock()
    return proc


def _make_pipeline(
    escalation_manager=None,
    cookie_rotator=None,
    impersonation_manager=None,
    speed_tracker=None,
    config=None,
):
    """Create an AudioFirstPipeline with mock deps and optional escalation."""
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
# AC 1: download_audio_for_keyword() passes escalation args to yt-dlp
#        — verify args include --impersonate when Tier 1
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestAudioDownloadEscalationArgs:
    """download_audio_for_keyword() integrates with EscalationManager."""

    @pytest.mark.integration
    def test_tier1_impersonate_args_in_ytdlp_command(self, temp_dir, esc_mgr):
        """At Tier 1, yt-dlp command includes --impersonate from escalation."""
        pipeline = _make_pipeline(escalation_manager=esc_mgr)
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
        ]))

        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            # Create fake audio file
            audio_dir = temp_dir / "test_s_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / "vid1.mp3").write_bytes(b'audio')
            result = Mock(returncode=0, stderr='')
            return result

        with patch('subprocess.run', side_effect=mock_run):
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        assert len(captured_cmds) >= 1, "Expected at least one subprocess call"
        cmd = captured_cmds[0]
        assert "--impersonate" in cmd, (
            f"Expected --impersonate in command at Tier 1, got: {cmd}"
        )

    @pytest.mark.integration
    def test_tier1_no_extractor_args(self, temp_dir, esc_mgr):
        """At Tier 1, yt-dlp command does NOT include --extractor-args."""
        pipeline = _make_pipeline(escalation_manager=esc_mgr)
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
        ]))

        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            audio_dir = temp_dir / "test_s_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / "vid1.mp3").write_bytes(b'audio')
            return Mock(returncode=0, stderr='')

        with patch('subprocess.run', side_effect=mock_run):
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]
        assert "--extractor-args" not in cmd, (
            f"Tier 1 should NOT have --extractor-args, got: {cmd}"
        )


# ---------------------------------------------------------------------------
# AC 2: download_audio_for_keyword() handles rotate_cookies=True
#        by calling cookie_rotator.rotate() before retry
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestAudioDownloadCookieRotation:
    """download_audio_for_keyword() rotates cookies on Tier 3 escalation."""

    @pytest.mark.integration
    def test_rotate_cookies_called_on_tier3(self, temp_dir, imp_mgr):
        """When escalation reaches Tier 3 (rotate_cookies=True),
        cookie_rotator.rotate() is called before the download."""
        ext_config = FakeExtractorArgsConfig()
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Pre-escalate vid1 to Tier 3
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        esc_mgr.record_failure("vid1", "HTTP Error 403")  # -> Tier 2
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        esc_mgr.record_failure("vid1", "HTTP Error 403")  # -> Tier 3
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.FULL_BYPASS

        cookie_rotator = MagicMock()
        cookie_rotator.is_enabled = True
        cookie_rotator.available_cookies = 3
        cookie_rotator.get_current_cookie.return_value = "/tmp/cookies.txt"
        cookie_rotator.should_rotate.return_value = False

        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            cookie_rotator=cookie_rotator,
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
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        # Cookie rotation should have been triggered due to Tier 3
        cookie_rotator.rotate.assert_called()

    @pytest.mark.integration
    def test_no_cookie_rotation_at_tier1(self, temp_dir, esc_mgr):
        """At Tier 1, cookie_rotator.rotate() is NOT called proactively."""
        cookie_rotator = MagicMock()
        cookie_rotator.is_enabled = True
        cookie_rotator.available_cookies = 3
        cookie_rotator.get_current_cookie.return_value = "/tmp/cookies.txt"
        cookie_rotator.should_rotate.return_value = False

        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            cookie_rotator=cookie_rotator,
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
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        # rotate() should NOT be called at Tier 1 (proactive rotation is Tier 3 only)
        cookie_rotator.rotate.assert_not_called()


# ---------------------------------------------------------------------------
# AC 3: download_video_segments() applies escalation args to commands
#        — verify --extractor-args added when Tier 2+
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestSegmentDownloadEscalationArgs:
    """download_video_segments() applies escalation args per tier."""

    @pytest.mark.integration
    def test_tier2_extractor_args_in_segment_command(self, temp_dir, imp_mgr):
        """At Tier 2, segment download includes --extractor-args."""
        ext_config = FakeExtractorArgsConfig()
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Pre-escalate vid1 to Tier 2
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.EXTRACTOR_ARGS

        pipeline = _make_pipeline(escalation_manager=esc_mgr)

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            )
        ]

        captured_cmds = []

        def mock_popen_factory(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return _make_mock_popen(returncode=0)

        with patch('subprocess.Popen', side_effect=mock_popen_factory), \
             patch('src.downloader.segment_utils.rename_segments_with_timing',
                   return_value=[None]):
            pipeline.download_video_segments(segments, temp_dir)

        assert len(captured_cmds) >= 1, "Expected at least one subprocess call"
        cmd = captured_cmds[0]
        assert "--extractor-args" in cmd, (
            f"Tier 2 should include --extractor-args, got: {cmd}"
        )
        assert "--impersonate" in cmd, (
            f"Tier 2 should also include --impersonate, got: {cmd}"
        )

    @pytest.mark.integration
    def test_segment_records_success_on_completion(self, temp_dir, esc_mgr):
        """After successful segment download, escalation_manager.record_success() is called."""
        pipeline = _make_pipeline(escalation_manager=esc_mgr)

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            )
        ]

        seg_dir = temp_dir / "travel_segments"

        def mock_popen_factory(cmd, **kwargs):
            # Create segment file on successful download (simulating yt-dlp)
            seg_dir.mkdir(parents=True, exist_ok=True)
            (seg_dir / "vid1_0010.mp4").write_bytes(b'segment')
            return _make_mock_popen(returncode=0)

        with patch('subprocess.Popen', side_effect=mock_popen_factory), \
             patch('src.downloader.segment_utils.rename_segments_with_timing',
                   return_value=[str(seg_dir / "vid1_0010.mp4")]):
            pipeline.download_video_segments(segments, temp_dir)

        # Verify success was recorded
        metrics = esc_mgr.get_metrics()
        assert metrics["total_successes"] >= 1


# ---------------------------------------------------------------------------
# AC 4: _download_full_video_fallback() escalates independently from
#        segment download — verify tier state preserved across transition
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestFallbackEscalationIndependence:
    """_download_full_video_fallback() uses escalation independently."""

    @pytest.mark.integration
    def test_fallback_preserves_tier_from_segment_escalation(self, temp_dir, imp_mgr):
        """After segment download escalates vid1 to Tier 2, fallback still
        uses Tier 2 args (tier state preserved across fallback transition)."""
        ext_config = FakeExtractorArgsConfig()
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Pre-escalate vid1 to Tier 2 (simulating failures during segment download)
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.EXTRACTOR_ARGS

        pipeline = _make_pipeline(escalation_manager=esc_mgr)

        captured_cmds = []
        video_dir = temp_dir / "video_dir"
        video_dir.mkdir()

        def mock_popen_factory(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            # Create the output file so fallback considers it successful
            output_file = video_dir / "vid1_0000.mp4"
            output_file.write_bytes(b'full video')
            return _make_mock_popen(returncode=0)

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[{'match_id': 1}],
                keyword="travel",
            )
        ]

        with patch('subprocess.Popen', side_effect=mock_popen_factory), \
             patch.object(pipeline, '_get_video_duration', return_value=300.0):
            result = pipeline._download_full_video_fallback(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                video_dir=video_dir,
                segments=segments,
                keyword="travel",
                timeout=60,
            )

        assert len(result) == 1
        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]
        # Fallback should use Tier 2 args (preserved from segment escalation)
        assert "--extractor-args" in cmd, (
            f"Fallback should use Tier 2 --extractor-args, got: {cmd}"
        )
        assert "--impersonate" in cmd

    @pytest.mark.integration
    def test_fallback_records_failure_on_403(self, temp_dir, imp_mgr):
        """Fallback records 403 failure with escalation manager."""
        ext_config = FakeExtractorArgsConfig()
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        pipeline = _make_pipeline(escalation_manager=esc_mgr)

        video_dir = temp_dir / "video_dir"
        video_dir.mkdir()

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            )
        ]

        def mock_popen_factory(cmd, **kwargs):
            return _make_mock_popen(returncode=1)

        with patch('subprocess.Popen', side_effect=mock_popen_factory), \
             patch.object(pipeline, '_wait_for_process_with_progress',
                        return_value=('', "ERROR: HTTP Error 403: Forbidden", None)):
            pipeline._download_full_video_fallback(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                video_dir=video_dir,
                segments=segments,
                keyword="travel",
                timeout=60,
            )

        metrics = esc_mgr.get_metrics()
        assert metrics["total_403s"] >= 1, "Fallback should record 403 failure"

    @pytest.mark.integration
    def test_fallback_escalation_independent_from_other_keywords(
        self, temp_dir, imp_mgr
    ):
        """Fallback for vid1 doesn't affect vid2's escalation tier."""
        ext_config = FakeExtractorArgsConfig()
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Escalate vid1 to Tier 2
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        esc_mgr.record_failure("vid1", "HTTP Error 403")

        # vid2 stays at Tier 1
        assert esc_mgr.get_escalation_args("vid2").tier == EscalationTier.IMPERSONATE_ONLY

        pipeline = _make_pipeline(escalation_manager=esc_mgr)
        video_dir = temp_dir / "video_dir"
        video_dir.mkdir()

        segments = [
            MergedSegment(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                start_time=10.0,
                end_time=50.0,
                original_matches=[],
                keyword="travel",
            )
        ]

        def mock_popen_factory(cmd, **kwargs):
            (video_dir / "vid1_0000.mp4").write_bytes(b'data')
            return _make_mock_popen(returncode=0)

        with patch('subprocess.Popen', side_effect=mock_popen_factory), \
             patch.object(pipeline, '_get_video_duration', return_value=120.0):
            pipeline._download_full_video_fallback(
                video_id="vid1",
                video_url="https://youtube.com/watch?v=vid1",
                video_dir=video_dir,
                segments=segments,
                keyword="travel",
            )

        # vid2 should still be at Tier 1
        assert esc_mgr.get_escalation_args("vid2").tier == EscalationTier.IMPERSONATE_ONLY


# ---------------------------------------------------------------------------
# AC 5: Retry loop with 2 consecutive 403 errors triggers escalation
#        from Tier 1 to Tier 2 within download_audio_for_keyword()
# ---------------------------------------------------------------------------

@pytest.mark.fast
class TestRetryLoopEscalation:
    """Consecutive 403s in download_audio_for_keyword() trigger tier escalation."""

    @pytest.mark.integration
    def test_two_403s_escalate_tier1_to_tier2(self, temp_dir, imp_mgr):
        """Two consecutive 403 errors during audio download trigger
        escalation from Tier 1 to Tier 2.

        The retry loop in download_audio_for_keyword() retries on cookie
        rotation. We configure cookie_rotator to trigger on 403 so the
        same video_id gets two 403 failures, reaching the threshold."""
        ext_config = FakeExtractorArgsConfig(escalation_threshold=2)
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Verify starting at Tier 1
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.IMPERSONATE_ONLY

        # Cookie rotator that triggers on 403 (allows retry loop to continue)
        cookie_rotator = MagicMock()
        cookie_rotator.is_enabled = True
        cookie_rotator.available_cookies = 3
        cookie_rotator.get_current_cookie.return_value = "/tmp/cookies.txt"
        cookie_rotator.should_rotate.return_value = True  # Triggers retry on 403
        cookie_rotator.rotate.return_value = "/tmp/cookies2.txt"

        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            cookie_rotator=cookie_rotator,
        )
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
        ]))

        def mock_run(cmd, **kwargs):
            # All calls return 403 to trigger escalation recording
            return Mock(
                returncode=1,
                stderr="ERROR: [youtube] vid1: HTTP Error 403: Forbidden",
            )

        with patch('subprocess.run', side_effect=mock_run):
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        # After 2+ 403 failures for vid1 (via cookie rotation retries),
        # escalation should have reached Tier 2
        result = esc_mgr.get_escalation_args("vid1")
        assert result.tier == EscalationTier.EXTRACTOR_ARGS, (
            f"Expected Tier 2 after consecutive 403s, got {result.tier}"
        )

    @pytest.mark.integration
    def test_escalation_reflected_in_subsequent_download_commands(
        self, temp_dir, imp_mgr
    ):
        """After vid1 escalates, vid2's download still starts at its own tier.

        Uses cookie rotation to enable retry loop so vid1 accumulates
        multiple 403 failures within download_audio_for_keyword()."""
        ext_config = FakeExtractorArgsConfig(escalation_threshold=2)
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Cookie rotator triggers on 403 so the retry loop retries vid1
        cookie_rotator = MagicMock()
        cookie_rotator.is_enabled = True
        cookie_rotator.available_cookies = 3
        cookie_rotator.get_current_cookie.return_value = "/tmp/cookies.txt"
        cookie_rotator.should_rotate.return_value = True
        cookie_rotator.rotate.return_value = "/tmp/cookies2.txt"

        pipeline = _make_pipeline(
            escalation_manager=esc_mgr,
            cookie_rotator=cookie_rotator,
        )
        pipeline._search_video_metadata = Mock(return_value=_make_search_results([
            {'id': 'vid1', 'title': 'Test', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid1'},
            {'id': 'vid2', 'title': 'Test2', 'duration': 120,
             'webpage_url': 'https://youtube.com/watch?v=vid2'},
        ]))

        captured_cmds_by_vid = {'vid1': [], 'vid2': []}

        def mock_run(cmd, **kwargs):
            cmd_str = ' '.join(cmd)
            for vid in ['vid1', 'vid2']:
                if vid in cmd_str:
                    captured_cmds_by_vid[vid].append(list(cmd))

            # vid1 always fails with 403, vid2 succeeds
            if 'vid1' in cmd_str:
                return Mock(
                    returncode=1,
                    stderr="ERROR: HTTP Error 403: Forbidden",
                )
            else:
                audio_dir = temp_dir / "test_s_audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                (audio_dir / "vid2.mp3").write_bytes(b'audio')
                return Mock(returncode=0, stderr='')

        with patch('subprocess.run', side_effect=mock_run):
            pipeline.download_audio_for_keyword("test", temp_dir, "short")

        # vid1 escalated to Tier 2 (2+ 403s via retries), vid2 stays at Tier 1
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.EXTRACTOR_ARGS
        assert esc_mgr.get_escalation_args("vid2").tier == EscalationTier.IMPERSONATE_ONLY

        # vid2's command should NOT have --extractor-args (Tier 1)
        if captured_cmds_by_vid['vid2']:
            vid2_cmd = captured_cmds_by_vid['vid2'][0]
            assert "--extractor-args" not in vid2_cmd

    @pytest.mark.fast
    def test_success_after_403_resets_counter_but_keeps_tier(
        self, temp_dir, imp_mgr
    ):
        """A success after escalation resets the 403 counter but keeps Tier 2."""
        ext_config = FakeExtractorArgsConfig(escalation_threshold=2)
        esc_mgr = EscalationManager(imp_mgr, ext_config)

        # Escalate vid1 to Tier 2
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.EXTRACTOR_ARGS

        # Record success (simulating successful download at Tier 2)
        esc_mgr.record_success("vid1")

        # Tier stays at 2 (sticky)
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.EXTRACTOR_ARGS

        # Single new failure doesn't re-escalate (counter was reset)
        esc_mgr.record_failure("vid1", "HTTP Error 403")
        assert esc_mgr.get_escalation_args("vid1").tier == EscalationTier.EXTRACTOR_ARGS
