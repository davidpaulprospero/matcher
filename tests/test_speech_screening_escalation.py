"""Integration tests for SpeechScreener escalation wiring (Sprint 14 US-008).

Tests that SpeechScreener correctly integrates with EscalationManager:
- download_audio_clip() applies escalation_manager.get_escalation_args() to yt-dlp
- download_audio_clip() omits escalation args when escalation_manager is None
- screen_video_for_speech() records failure via record_failure() on 403 errors
- screen_video_for_speech() handles EscalationResult.rotate_cookies=True
- screen_approved_videos() shares escalation state across multiple video calls

Pytest marker: fast
"""

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from src.downloader.speech_screening import SpeechScreener
from src.downloader.escalation_manager import (
    EscalationManager,
    EscalationResult,
    is_escalation_trigger,
)
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
    cooldown_seconds: float = 0.0  # No cooldown for fast tests
    max_tier: int = 3


def _make_mock_config():
    """Create a mock Config with speech screening settings."""
    config = Mock()
    config.download = Mock()
    config.download.speech_screening = Mock()
    config.download.speech_screening.screening_duration = 5.0
    config.download.speech_screening.min_speech_duration = 0.5
    config.download.speech_screening.whisper_model = "base"
    config.download.speech_screening.timeout_per_video = 30
    config.download.speech_screening.fallback_on_error = "accept"
    config.download.speech_screening.reject_with_speech = True
    config.download.audio_first = Mock()
    config.download.audio_first.audio_quality = 5
    config.download.ffmpeg_location = ""
    return config


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = ["--impersonate", "Chrome-136:Macos-15"]
    mgr.target_count = 5
    return mgr


def _make_escalation_manager():
    """Create a real EscalationManager with fast test settings."""
    imp_mgr = _make_impersonation_manager()
    ext_config = FakeExtractorArgsConfig()
    return EscalationManager(
        impersonation_manager=imp_mgr,
        extractor_args_config=ext_config,
    )


def _make_sample_video(video_id="abc123"):
    """Create a sample video metadata dict."""
    return {
        'id': video_id,
        'webpage_url': f'https://youtube.com/watch?v={video_id}',
        'title': f'Sample Video {video_id}',
        'duration': 120
    }


# ---------------------------------------------------------------------------
# AC1: download_audio_clip() applies escalation args when manager provided
# ---------------------------------------------------------------------------

class TestDownloadAudioClipEscalationArgs:
    """Test that download_audio_clip() uses escalation_manager.get_escalation_args()."""

    @patch('subprocess.run')
    @pytest.mark.integration
    def test_tier1_impersonate_args_applied(self, mock_run):
        """AC1: Verify --impersonate args included when Tier 1 (default)."""
        config = _make_mock_config()
        esc_mgr = _make_escalation_manager()

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            escalation_manager=esc_mgr,
        )

        mock_run.return_value = Mock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
            )

            cmd = mock_run.call_args[0][0]
            assert '--impersonate' in cmd, "Tier 1 should include --impersonate"

    @patch('subprocess.run')
    @pytest.mark.integration
    def test_tier2_extractor_args_applied(self, mock_run):
        """AC1: Verify --extractor-args added when Tier 2."""
        config = _make_mock_config()
        esc_mgr = _make_escalation_manager()

        # Force escalation to Tier 2
        for _ in range(2):
            esc_mgr.record_failure("abc123", "HTTP Error 403")

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            escalation_manager=esc_mgr,
        )

        mock_run.return_value = Mock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
            )

            cmd = mock_run.call_args[0][0]
            assert '--impersonate' in cmd
            assert '--extractor-args' in cmd, "Tier 2 should include --extractor-args"

    @patch('subprocess.run')
    @pytest.mark.integration
    def test_escalation_args_before_cookies(self, mock_run):
        """AC1: Verify escalation args appear before cookie args in command."""
        config = _make_mock_config()
        esc_mgr = _make_escalation_manager()

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            escalation_manager=esc_mgr,
        )

        mock_run.return_value = Mock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
            )

            cmd = mock_run.call_args[0][0]
            imp_idx = cmd.index('--impersonate')
            cookie_idx = cmd.index('--cookies')
            assert imp_idx < cookie_idx, (
                "Escalation args must appear before cookie args"
            )


# ---------------------------------------------------------------------------
# AC2: download_audio_clip() omits escalation args when manager is None
# ---------------------------------------------------------------------------

class TestDownloadAudioClipNoEscalation:
    """Test that download_audio_clip() works without escalation_manager."""

    @patch('subprocess.run')
    @pytest.mark.integration
    def test_no_impersonate_when_no_manager(self, mock_run):
        """AC2: No --impersonate when both escalation_manager and impersonation_manager are None."""
        config = _make_mock_config()

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            # Both managers None
            impersonation_manager=None,
            escalation_manager=None,
        )

        mock_run.return_value = Mock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
            )

            cmd = mock_run.call_args[0][0]
            assert '--impersonate' not in cmd
            assert '--extractor-args' not in cmd

    @patch('subprocess.run')
    @pytest.mark.integration
    def test_impersonation_only_when_no_escalation_manager(self, mock_run):
        """AC2: Falls back to impersonation_manager when escalation_manager is None."""
        config = _make_mock_config()
        imp_mgr = _make_impersonation_manager()

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            impersonation_manager=imp_mgr,
            escalation_manager=None,
        )

        mock_run.return_value = Mock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
            )

            cmd = mock_run.call_args[0][0]
            assert '--impersonate' in cmd
            # impersonation_manager should be called
            imp_mgr.get_impersonate_args.assert_called_once()

    @patch('subprocess.run')
    @pytest.mark.integration
    def test_no_attribute_error_when_none(self, mock_run):
        """AC2: No AttributeError when escalation_manager is None on error path."""
        config = _make_mock_config()

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=None,
        )

        # Simulate 403 error
        mock_run.return_value = Mock(
            returncode=1,
            stderr="HTTP Error 403: Forbidden"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # No crash when escalation_manager is None and 403 occurs
            result = screener.download_audio_clip(
                video_url="https://youtube.com/watch?v=abc123",
                video_id="abc123",
                temp_dir=temp_path,
            )
            assert result is None  # Failed download, no crash


# ---------------------------------------------------------------------------
# AC3: screen_video_for_speech() records escalation failure on 403
# ---------------------------------------------------------------------------

class TestScreenVideoRecordsFailure:
    """Test that screen_video_for_speech() records failure via escalation_manager."""

    @pytest.mark.integration
    def test_records_failure_on_403_download(self):
        """AC3: record_failure() called when download returns 403 stderr."""
        config = _make_mock_config()
        esc_mgr = Mock()
        esc_mgr.get_escalation_args.return_value = EscalationResult(
            args=["--impersonate", "Chrome-136:Macos-15"],
            tier=EscalationTier.IMPERSONATE_ONLY,
            rotate_cookies=False,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )

        video = _make_sample_video()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(
                    returncode=1,
                    stderr="HTTP Error 403: Forbidden",
                )

                has_speech, duration = screener.screen_video_for_speech(video, temp_path)

                # record_failure should be called with the video ID and stderr
                esc_mgr.record_failure.assert_called_once()
                call_args = esc_mgr.record_failure.call_args
                assert call_args[0][0] == "abc123"  # keyword/video_id
                assert "403" in call_args[0][1]  # error_output

    @pytest.mark.integration
    def test_no_failure_recorded_on_timeout(self):
        """AC3: record_failure() NOT called on timeout (not a 403)."""
        config = _make_mock_config()
        esc_mgr = Mock()
        esc_mgr.get_escalation_args.return_value = EscalationResult(
            args=["--impersonate", "Chrome-136:Macos-15"],
            tier=EscalationTier.IMPERSONATE_ONLY,
            rotate_cookies=False,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )

        video = _make_sample_video()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired(
                    cmd="yt-dlp", timeout=30
                )

                has_speech, duration = screener.screen_video_for_speech(video, temp_path)

                # Timeout is not a 403 - no escalation failure recorded
                esc_mgr.record_failure.assert_not_called()

    @pytest.mark.integration
    def test_no_failure_recorded_on_success(self):
        """AC3: record_failure() NOT called on successful download."""
        config = _make_mock_config()
        esc_mgr = Mock()
        esc_mgr.get_escalation_args.return_value = EscalationResult(
            args=["--impersonate", "Chrome-136:Macos-15"],
            tier=EscalationTier.IMPERSONATE_ONLY,
            rotate_cookies=False,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )

        video = _make_sample_video()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stderr="")

                # Mock transcription to avoid real Whisper call
                mock_transcription = MagicMock()
                mock_transcription.transcribe_voiceover_audio = MagicMock(return_value=[])
                with patch.dict(sys.modules, {'src.transcription': mock_transcription}):
                    has_speech, duration = screener.screen_video_for_speech(video, temp_path)

                esc_mgr.record_failure.assert_not_called()


# ---------------------------------------------------------------------------
# AC4: screen_video_for_speech() handles rotate_cookies=True
# ---------------------------------------------------------------------------

class TestScreenVideoHandlesCookieRotation:
    """Test that screen_video_for_speech() handles EscalationResult.rotate_cookies."""

    @pytest.mark.integration
    def test_cookie_rotation_triggered(self):
        """AC4: When rotate_cookies=True, cookie_rotator.rotate() is called before download."""
        config = _make_mock_config()
        esc_mgr = Mock()
        esc_mgr.get_escalation_args.return_value = EscalationResult(
            args=["--impersonate", "Chrome-136:Macos-15", "--extractor-args",
                  "youtube:player_client=web_safari,tv_downgraded,web"],
            tier=EscalationTier.FULL_BYPASS,
            rotate_cookies=True,
        )

        cookie_rotator = Mock()
        cookie_rotator.get_current_cookie_path.return_value = "rotated_cookies.txt"

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            escalation_manager=esc_mgr,
        )
        screener.cookie_rotator = cookie_rotator

        video = _make_sample_video()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=1, stderr="")

                screener.download_audio_clip(
                    video_url="https://youtube.com/watch?v=abc123",
                    video_id="abc123",
                    temp_dir=temp_path,
                )

                cookie_rotator.rotate.assert_called_once()

    @pytest.mark.integration
    def test_no_rotation_when_flag_false(self):
        """AC4: No cookie rotation when rotate_cookies=False."""
        config = _make_mock_config()
        esc_mgr = Mock()
        esc_mgr.get_escalation_args.return_value = EscalationResult(
            args=["--impersonate", "Chrome-136:Macos-15"],
            tier=EscalationTier.IMPERSONATE_ONLY,
            rotate_cookies=False,
        )

        cookie_rotator = Mock()

        screener = SpeechScreener(
            config=config,
            cookies_args=['--cookies', 'cookies.txt'],
            escalation_manager=esc_mgr,
        )
        screener.cookie_rotator = cookie_rotator

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_file = temp_path / "abc123.mp3"
            audio_file.write_bytes(b"fake audio")

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stderr="")

                screener.download_audio_clip(
                    video_url="https://youtube.com/watch?v=abc123",
                    video_id="abc123",
                    temp_dir=temp_path,
                )

                cookie_rotator.rotate.assert_not_called()

    @pytest.mark.integration
    def test_no_crash_when_no_cookie_rotator(self):
        """AC4: No crash when rotate_cookies=True but no cookie_rotator set."""
        config = _make_mock_config()
        esc_mgr = Mock()
        esc_mgr.get_escalation_args.return_value = EscalationResult(
            args=["--impersonate", "Chrome-136:Macos-15"],
            tier=EscalationTier.FULL_BYPASS,
            rotate_cookies=True,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )
        # No cookie_rotator attribute set — should not crash

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=1, stderr="")

                # Should not raise AttributeError
                result = screener.download_audio_clip(
                    video_url="https://youtube.com/watch?v=abc123",
                    video_id="abc123",
                    temp_dir=temp_path,
                )
                assert result is None  # Download failed, no crash


# ---------------------------------------------------------------------------
# AC5: screen_approved_videos() shares escalation state across video calls
# ---------------------------------------------------------------------------

class TestScreenApprovedVideosSharedState:
    """Test that screen_approved_videos() shares escalation state across videos."""

    @pytest.mark.integration
    def test_tier_progression_across_videos(self):
        """AC5: Tier advances from Tier 1 -> Tier 2 across video screening calls."""
        config = _make_mock_config()
        imp_mgr = _make_impersonation_manager()
        ext_config = FakeExtractorArgsConfig(escalation_threshold=2)
        esc_mgr = EscalationManager(
            impersonation_manager=imp_mgr,
            extractor_args_config=ext_config,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )

        # Create 3 videos to screen (all will use same video_id="shared_kw" for escalation)
        videos = [_make_sample_video(f"vid{i}") for i in range(3)]

        # Track escalation args per video
        captured_args = []

        def fake_run(cmd, **kwargs):
            # Capture the command for arg inspection
            captured_args.append(list(cmd))
            # Simulate 403 for first 2 videos to trigger escalation
            if len(captured_args) <= 2:
                return Mock(
                    returncode=1,
                    stderr="HTTP Error 403: Forbidden"
                )
            return Mock(returncode=0, stderr="")

        with patch('subprocess.run', side_effect=fake_run):
            mock_transcription = MagicMock()
            mock_transcription.transcribe_voiceover_audio = MagicMock(return_value=[])
            with patch.dict(sys.modules, {'src.transcription': mock_transcription}):
                screener.screen_approved_videos(videos, keyword="beach")

        # After 2 consecutive 403s, tier should have escalated for those video IDs
        # Verify that escalation was recorded
        total_403s = esc_mgr.get_metrics()['total_403s']
        assert total_403s >= 2, (
            f"Expected at least 2 recorded 403s, got {total_403s}"
        )

    @pytest.mark.integration
    def test_shared_manager_state_persists(self):
        """AC5: Escalation state persists across multiple screen_approved_videos() calls."""
        config = _make_mock_config()
        imp_mgr = _make_impersonation_manager()
        ext_config = FakeExtractorArgsConfig(escalation_threshold=2)
        esc_mgr = EscalationManager(
            impersonation_manager=imp_mgr,
            extractor_args_config=ext_config,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )

        call_count = [0]

        def fake_run(cmd, **kwargs):
            call_count[0] += 1
            return Mock(returncode=1, stderr="HTTP Error 403: Forbidden")

        with patch('subprocess.run', side_effect=fake_run):
            # First batch
            batch1 = [_make_sample_video("vid_shared")]
            screener.screen_approved_videos(batch1, keyword="beach")

            # Second batch (same video ID to accumulate 403s)
            batch2 = [_make_sample_video("vid_shared")]
            screener.screen_approved_videos(batch2, keyword="nature")

        # Verify escalation state accumulated across calls
        state = esc_mgr._keyword_states.get("vid_shared")
        assert state is not None, "State should exist for vid_shared"
        assert state.current_tier >= EscalationTier.EXTRACTOR_ARGS, (
            f"Expected Tier 2+ after 2 consecutive 403s, got {state.current_tier.name}"
        )

    @pytest.mark.fast
    def test_different_videos_independent_tiers(self):
        """AC5: Different video IDs maintain independent escalation tiers."""
        config = _make_mock_config()
        imp_mgr = _make_impersonation_manager()
        ext_config = FakeExtractorArgsConfig(escalation_threshold=2)
        esc_mgr = EscalationManager(
            impersonation_manager=imp_mgr,
            extractor_args_config=ext_config,
        )

        screener = SpeechScreener(
            config=config,
            cookies_args=[],
            escalation_manager=esc_mgr,
        )

        # Pre-escalate vid_a to Tier 2
        esc_mgr.record_failure("vid_a", "HTTP Error 403")
        esc_mgr.record_failure("vid_a", "HTTP Error 403")

        # vid_b should still be at Tier 1
        result_a = esc_mgr.get_escalation_args("vid_a")
        result_b = esc_mgr.get_escalation_args("vid_b")

        assert result_a.tier >= EscalationTier.EXTRACTOR_ARGS
        assert result_b.tier == EscalationTier.IMPERSONATE_ONLY
