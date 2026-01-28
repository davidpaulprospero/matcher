"""
Speech screening using Whisper VAD.

Migrated from VideoDownloader speech screening methods (lines 848-1030).
Screens videos for speech content to identify B-roll footage (silent videos).
"""

from __future__ import annotations

import subprocess
import tempfile
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .escalation_manager import is_escalation_trigger

if TYPE_CHECKING:
    from ..config import Config
    from .impersonation import ImpersonationManager
    from .escalation_manager import EscalationManager

logger = logging.getLogger(__name__)


class SpeechScreener:
    """Screens videos for speech content using Whisper VAD.

    Migrated from VideoDownloader speech screening methods.
    """

    def __init__(
        self,
        config: 'Config',
        cookies_args: List[str],
        impersonation_manager: Optional['ImpersonationManager'] = None,
        escalation_manager: Optional['EscalationManager'] = None,
    ):
        """
        Initialize SpeechScreener.

        Args:
            config: Config object with download.speech_screening settings
            cookies_args: Cookie arguments for yt-dlp (from utils.get_cookies_args)
            impersonation_manager: Optional ImpersonationManager for TLS fingerprint bypass
            escalation_manager: Optional EscalationManager for 3-tier bypass orchestration
        """
        self.config = config
        self.download_config = config.download
        self.cookies_args = cookies_args
        self.impersonation_manager = impersonation_manager
        self.escalation_manager = escalation_manager

    def download_audio_clip(
        self,
        video_url: str,
        video_id: str,
        temp_dir: Path,
        duration: float = 5.0
    ) -> Optional[Path]:
        """
        Download first N seconds of audio for speech screening.

        Migrated from downloader.py lines 848-904.

        Reuses the same yt-dlp audio pattern from download_audio_for_keyword().

        Args:
            video_url: YouTube video URL
            video_id: Video ID for naming
            temp_dir: Temporary directory for audio files
            duration: Seconds from start to download

        Returns:
            Path to downloaded audio file, or None on failure
        """
        audio_config = getattr(self.download_config, 'audio_first', None)
        audio_quality = getattr(audio_config, 'audio_quality', 5) if audio_config else 5

        cmd = [
            'yt-dlp',
            '--ignore-config',
            video_url,
            '--download-sections', f'*0-{duration}',  # Only first N seconds
            '-x',  # Extract audio
            '--audio-format', 'mp3',
            '--audio-quality', str(audio_quality),
            '-o', str(temp_dir / f'{video_id}.%(ext)s'),
            '--no-playlist',
            '--no-warnings',
            '--quiet',
        ]

        # Add ffmpeg location if configured
        ffmpeg_loc = getattr(self.download_config, 'ffmpeg_location', '')
        if ffmpeg_loc:
            cmd.extend(['--ffmpeg-location', ffmpeg_loc])

        # Add escalation/impersonation args before cookies for correct argument ordering
        if self.escalation_manager:
            esc_result = self.escalation_manager.get_escalation_args(video_id)
            if esc_result.args:
                cmd.extend(esc_result.args)
            # Tier 3: trigger cookie rotation proactively
            if esc_result.rotate_cookies:
                cookie_rotator = getattr(self, 'cookie_rotator', None)
                if cookie_rotator:
                    cookie_rotator.rotate()
        elif self.impersonation_manager:
            imp_args = self.impersonation_manager.get_impersonate_args()
            if imp_args:
                cmd.extend(imp_args)

        # Add cookies
        cmd.extend(self.cookies_args)

        try:
            speech_config = getattr(self.download_config, 'speech_screening', None)
            timeout = getattr(speech_config, 'timeout_per_video', 30) if speech_config else 30

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                matches = list(temp_dir.glob(f"{video_id}.*"))
                return matches[0] if matches else None
            else:
                # Record 403/bot errors with escalation manager
                stderr = result.stderr or ''
                if self.escalation_manager and is_escalation_trigger(stderr):
                    self.escalation_manager.record_failure(video_id, stderr)
        except subprocess.TimeoutExpired:
            logger.debug(f"[SPEECH SCREEN] {video_id}: download timeout")
        except Exception as e:
            logger.debug(f"[SPEECH SCREEN] {video_id}: download error: {e}")
        return None

    def screen_video_for_speech(
        self,
        video: Dict[str, Any],
        temp_dir: Path
    ) -> Tuple[bool, float]:
        """
        Download first N seconds and check for speech using Whisper VAD.

        Migrated from downloader.py lines 906-974.

        Args:
            video: Video metadata dict with 'id', 'webpage_url', etc.
            temp_dir: Temporary directory for audio files

        Returns:
            Tuple[bool, float]: (has_speech, speech_duration_seconds)
        """
        video_id = video.get('id', '')
        video_url = video.get('webpage_url', f"https://www.youtube.com/watch?v={video_id}")

        speech_config = getattr(self.download_config, 'speech_screening', None)
        screening_duration = getattr(speech_config, 'screening_duration', 5.0) if speech_config else 5.0
        min_speech_duration = getattr(speech_config, 'min_speech_duration', 0.5) if speech_config else 0.5
        whisper_model = getattr(speech_config, 'whisper_model', 'base') if speech_config else 'base'
        fallback = getattr(speech_config, 'fallback_on_error', 'accept') if speech_config else 'accept'

        # Download first N seconds of audio
        video_title = video.get('title', 'Unknown')[:50]
        logger.info(f"[SPEECH SCREEN] {video_id} - '{video_title}': downloading first {screening_duration}s audio...")
        audio_path = self.download_audio_clip(video_url, video_id, temp_dir, screening_duration)

        if not audio_path or not audio_path.exists():
            logger.warning(f"[SPEECH SCREEN] {video_id}: audio download failed, fallback={fallback}")
            return (fallback == 'reject', 0.0)  # has_speech=True if fallback is reject

        try:
            # Import transcription module
            from src.transcription import transcribe_voiceover_audio

            logger.info(f"[SPEECH SCREEN] {video_id}: transcribing with Whisper ({whisper_model})...")
            # Transcribe with VAD to detect speech
            segments = transcribe_voiceover_audio(
                str(audio_path),
                model_name=whisper_model,
                compute_type="auto",  # Let faster-whisper auto-detect best type
                vad_filter=True,
                language=None  # Auto-detect
            )

            # Calculate total speech duration (segments are dicts with 'start', 'end', 'text')
            speech_duration = sum(seg['end'] - seg['start'] for seg in segments) if segments else 0.0
            has_speech = speech_duration >= min_speech_duration

            # Log detected text if speech found
            if has_speech and segments:
                text_preview = " ".join(seg.get('text', '') for seg in segments[:3])[:100]
                logger.info(f"[SPEECH SCREEN] {video_id}: detected speech ({speech_duration:.1f}s): '{text_preview}...'")
            else:
                logger.info(f"[SPEECH SCREEN] {video_id}: no significant speech ({speech_duration:.1f}s < {min_speech_duration}s threshold)")

            return (has_speech, speech_duration)

        except Exception as e:
            logger.warning(f"[SPEECH SCREEN] {video_id}: transcription error: {e}, fallback={fallback}")
            return (fallback == 'reject', 0.0)
        finally:
            # Clean up audio file
            try:
                if audio_path and audio_path.exists():
                    audio_path.unlink()
            except Exception:
                pass

    def screen_approved_videos(
        self,
        approved_videos: List[Dict[str, Any]],
        keyword: str
    ) -> List[Dict[str, Any]]:
        """
        Screen approved videos for speech, filter out those with talking.

        Migrated from downloader.py lines 976-1030.

        Args:
            approved_videos: Videos that passed LLM title filter
            keyword: Current search keyword (for temp dir naming)

        Returns:
            List of videos that passed speech screening (no speech detected)
        """
        speech_config = getattr(self.download_config, 'speech_screening', None)
        reject_with_speech = getattr(speech_config, 'reject_with_speech', True) if speech_config else True

        passed = []
        rejected = []

        logger.info(f"[SPEECH SCREEN] Starting screening for keyword '{keyword}': {len(approved_videos)} videos to check")

        # Create temp directory for audio clips
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)[:20]
        with tempfile.TemporaryDirectory(prefix=f"speech_screen_{safe_keyword}_") as temp_dir:
            temp_path = Path(temp_dir)

            for idx, video in enumerate(approved_videos, 1):
                video_id = video.get('id', 'unknown')
                logger.info(f"[SPEECH SCREEN] Progress: {idx}/{len(approved_videos)} - checking {video_id}")
                has_speech, speech_duration = self.screen_video_for_speech(video, temp_path)

                if has_speech:
                    if reject_with_speech:
                        logger.info(f"[SPEECH SCREEN] {video_id}: ❌ REJECTED - Speech detected ({speech_duration:.1f}s)")
                        rejected.append(video)
                    else:
                        logger.info(f"[SPEECH SCREEN] {video_id}: ⚠️  PASS (logging only) - Speech detected ({speech_duration:.1f}s)")
                        passed.append(video)
                else:
                    logger.info(f"[SPEECH SCREEN] {video_id}: ✅ PASSED - No speech ({speech_duration:.1f}s)")
                    passed.append(video)

        # Summary logging
        pass_rate = (len(passed) / len(approved_videos) * 100) if approved_videos else 0
        logger.info(f"[SPEECH SCREEN] ═══════════════════════════════════════")
        logger.info(f"[SPEECH SCREEN] Keyword: '{keyword}'")
        logger.info(f"[SPEECH SCREEN] Total screened: {len(approved_videos)}")
        logger.info(f"[SPEECH SCREEN] ✅ Passed (B-roll): {len(passed)} ({pass_rate:.1f}%)")
        logger.info(f"[SPEECH SCREEN] ❌ Rejected (speech): {len(rejected)} ({100-pass_rate:.1f}%)")
        logger.info(f"[SPEECH SCREEN] ═══════════════════════════════════════")

        return passed
