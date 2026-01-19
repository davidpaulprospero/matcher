"""
Caption Healer - YouTube caption fetching error recovery.

Handles:
- Caption fetch failures (yt-dlp subtitle errors)
- No captions available (fallback to audio transcription)
- Language unavailable (try alternate languages)
- Caption parse errors (malformed SRT/VTT)
- Rate limiting during caption fetch
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Set

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class CaptionHealer(Healer):
    """
    Heals caption fetching errors.

    Recovery strategies:
    1. No captions: Enable audio fallback mode
    2. Language unavailable: Try alternate language codes
    3. Parse error: Skip corrupted caption, try next
    4. Rate limit: Exponential backoff (only for CAPTION stage)
    5. Fetch timeout: Retry with longer timeout

    Note: This healer is stage-aware - it primarily handles CAPTION stage errors.
    For DOWNLOAD stage errors, DownloadHealer takes precedence.
    """

    name = "caption-healer"
    description = "Fix caption fetching errors"

    # Caption-specific patterns (avoid overlap with DownloadHealer)
    error_patterns = [
        "caption",
        "subtitle",
        "subtitles",
        "srt",
        "vtt",
        "no captions",
        "caption fetch",
        "write-sub",
        "write-auto-sub",
        "sub-lang",
        "parse caption",
        "caption parse",
        "captionfetcher",
        "caption_fetcher",
        "po token",           # PO Token required for subtitles
        "po_token",
        "subtitles require",  # "subtitles require a PO Token"
        "missing subtitles",  # "missing subtitles languages"
    ]

    # Stages this healer handles (stage-aware filtering)
    handled_stages = ["CAPTION"]

    # Language fallback order
    LANGUAGE_FALLBACKS = [
        ["en", "en-US", "en-GB", "en-AU"],  # English variants
        ["en-orig", "en-auto"],              # Auto-generated
        ["es", "es-ES", "es-MX"],            # Spanish variants
        ["fr", "de", "it", "pt"],            # Other European
    ]

    # Backoff configuration
    INITIAL_BACKOFF = 5.0
    MAX_BACKOFF = 120.0
    BACKOFF_MULTIPLIER = 2.0

    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0
        self.skipped_videos: Set[str] = set()
        self.language_attempt_index = 0

    def can_handle(self, error: Exception, stage_name: str) -> bool:
        """
        Check if this healer can handle the given error.

        Stage-aware: Primarily handles CAPTION stage errors.
        For other stages, only handles if error message is caption-specific.
        """
        # Always handle errors from CAPTION stage
        if stage_name.upper() in self.handled_stages:
            # Check error patterns for CAPTION stage
            error_str = str(error).lower()
            for pattern in self.error_patterns:
                if pattern.lower() in error_str:
                    return True
            # Also handle generic errors in CAPTION stage
            return True

        # For other stages, only handle if explicitly caption-related
        error_str = str(error).lower()
        caption_specific = [
            "caption", "subtitle", "srt", "vtt",
            "no captions", "write-sub", "sub-lang",
        ]
        return any(p in error_str for p in caption_specific)

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix caption-related errors."""
        error_str = str(error).lower()

        # PO Token required (YouTube restriction as of late 2024)
        if any(p in error_str for p in ["po token", "po_token", "subtitles require"]):
            return self._handle_po_token_error(error, state)

        # No captions available
        if any(p in error_str for p in ["no captions", "no subtitles", "caption not found"]):
            return self._handle_no_captions(error, state)

        # Language not available
        if any(p in error_str for p in ["language", "sub-lang", "lang not"]):
            return self._handle_language_error(error, state)

        # Parse error
        if any(p in error_str for p in ["parse", "malformed", "invalid srt", "invalid vtt"]):
            return self._handle_parse_error(error, state)

        # Rate limiting
        if any(p in error_str for p in ["429", "rate limit", "too many"]):
            return self._handle_rate_limit(error, state)

        # Timeout/network
        if any(p in error_str for p in ["timeout", "connection", "network"]):
            return self._handle_timeout(error, state)

        # Generic caption error - enable fallback
        return self._handle_generic_error(error, state)

    def _handle_po_token_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle PO Token requirement by switching to tv_embedded player.

        YouTube (as of late 2024) requires a PO Token for subtitle access on
        some player clients (web, mweb). The tv_embedded client bypasses this.
        """
        self.log_attempt("PO Token required for subtitles, switching to tv_embedded player...")

        # Get bypass config and ensure tv_embedded is enabled
        download_config = getattr(self.config, 'download', None)
        if download_config:
            bypass_config = getattr(download_config, 'rate_limit_bypass', None)
            if bypass_config:
                # Enable tv_embedded for subtitles
                if hasattr(bypass_config, 'subtitle_always_tv_embedded'):
                    bypass_config.subtitle_always_tv_embedded = True
                elif isinstance(bypass_config, dict):
                    bypass_config['subtitle_always_tv_embedded'] = True

                # Also escalate to tier 3 (tv_embedded) if not already there
                current_tier = getattr(bypass_config, '_current_tier', 1)
                if isinstance(bypass_config, dict):
                    current_tier = bypass_config.get('_current_tier', 1)

                if current_tier < 3:
                    if hasattr(bypass_config, '_current_tier'):
                        bypass_config._current_tier = 3
                    elif isinstance(bypass_config, dict):
                        bypass_config['_current_tier'] = 3

                self.log_success("Switched to tv_embedded player (bypasses PO Token requirement)")
                return HealerResult.config_changed(
                    "Enabled tv_embedded player for subtitles (bypasses PO Token)",
                    player_client="tv_embedded",
                    po_token_bypass=True
                )

        # Fallback: just enable audio fallback
        self.log_attempt("Could not configure tv_embedded, enabling audio fallback")
        return self._handle_no_captions(error, state)

    def _handle_no_captions(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle videos with no captions by enabling fallback."""
        self.log_attempt("No captions available, enabling audio fallback...")

        # Get caption config
        caption_config = self._get_caption_config()
        if caption_config:
            # Ensure fallback is enabled
            if hasattr(caption_config, 'fallback_to_audio'):
                caption_config.fallback_to_audio = True
            elif isinstance(caption_config, dict):
                caption_config['fallback_to_audio'] = True

        # Extract video ID if possible
        video_id = self._extract_video_id(str(error))
        if video_id:
            # Mark video for audio fallback
            if hasattr(state, 'videos_need_audio'):
                if video_id not in state.videos_need_audio:
                    state.videos_need_audio.append(video_id)
            self.skipped_videos.add(video_id)

            self.log_success(f"Video {video_id} marked for audio fallback")
            return HealerResult.fixed(
                f"No captions for {video_id}, using audio fallback",
                action=HealerAction.RETRY,
                video_id=video_id,
                fallback_enabled=True
            )

        self.log_success("Audio fallback enabled for videos without captions")
        return HealerResult.fixed(
            "Audio fallback enabled for uncaptioned videos",
            action=HealerAction.RETRY,
            fallback_enabled=True
        )

    def _handle_language_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle language not available by trying alternates."""
        self.log_attempt("Caption language not available, trying alternates...")

        caption_config = self._get_caption_config()
        if not caption_config:
            return HealerResult.failed("No caption config available")

        # Get current languages
        current_langs = getattr(caption_config, 'languages', None)
        if isinstance(caption_config, dict):
            current_langs = caption_config.get('languages', [])

        # Try next language group
        self.language_attempt_index += 1
        if self.language_attempt_index >= len(self.LANGUAGE_FALLBACKS):
            # All languages exhausted - fall back to audio
            self.log_attempt("All caption languages exhausted, enabling audio fallback")
            return self._handle_no_captions(error, state)

        new_langs = self.LANGUAGE_FALLBACKS[self.language_attempt_index]

        if hasattr(caption_config, 'languages'):
            caption_config.languages = new_langs
        elif isinstance(caption_config, dict):
            caption_config['languages'] = new_langs

        self.log_success(f"Trying alternate languages: {new_langs}")
        return HealerResult.config_changed(
            f"Switched to alternate languages: {new_langs}",
            old_languages=current_langs,
            new_languages=new_langs
        )

    def _handle_parse_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle caption parsing errors."""
        self.log_attempt("Caption parse error, attempting recovery...")

        video_id = self._extract_video_id(str(error))

        # Try to identify and remove corrupted caption file
        if video_id and hasattr(self.config, 'cache'):
            caption_dir = Path(self.config.cache.cache_dir) / "captions"
            if caption_dir.exists():
                # Look for caption files for this video
                corrupted_files = list(caption_dir.glob(f"{video_id}.*"))
                for f in corrupted_files:
                    try:
                        f.unlink()
                        self.log_success(f"Removed corrupted caption: {f.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove {f}: {e}")

        # Mark for re-fetch or audio fallback
        if video_id:
            if hasattr(state, 'videos_need_audio'):
                if video_id not in state.videos_need_audio:
                    state.videos_need_audio.append(video_id)

            return HealerResult.fixed(
                f"Removed corrupted caption for {video_id}, will retry or use audio",
                action=HealerAction.RETRY,
                video_id=video_id,
                corrupted_removed=True
            )

        return HealerResult.fixed(
            "Caption parse error, retrying",
            action=HealerAction.RETRY
        )

    def _handle_rate_limit(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle rate limiting during caption fetch."""
        self.log_attempt(f"Rate limited, waiting {self.backoff_time:.0f}s...")

        time.sleep(self.backoff_time)

        old_backoff = self.backoff_time
        self.backoff_time = min(self.backoff_time * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
        self.retry_count += 1

        self.log_success(f"Waited {old_backoff:.0f}s, resuming caption fetch")

        return HealerResult.fixed(
            f"Caption rate limit: waited {old_backoff:.0f}s",
            action=HealerAction.RETRY,
            backoff_seconds=old_backoff,
            retry_count=self.retry_count
        )

    def _handle_timeout(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle timeout errors during caption fetch."""
        self.log_attempt("Caption fetch timeout, increasing timeout...")

        caption_config = self._get_caption_config()
        if caption_config:
            current_timeout = getattr(caption_config, 'fetch_timeout', 30)
            if isinstance(caption_config, dict):
                current_timeout = caption_config.get('fetch_timeout', 30)

            new_timeout = min(current_timeout * 2, 120)

            if hasattr(caption_config, 'fetch_timeout'):
                caption_config.fetch_timeout = new_timeout
            elif isinstance(caption_config, dict):
                caption_config['fetch_timeout'] = new_timeout

            self.log_success(f"Increased caption timeout: {current_timeout}s -> {new_timeout}s")
            return HealerResult.config_changed(
                f"Increased caption timeout to {new_timeout}s",
                old_timeout=current_timeout,
                new_timeout=new_timeout
            )

        # No config to modify - just retry with backoff
        time.sleep(self.INITIAL_BACKOFF)
        return HealerResult.fixed(
            "Caption timeout, retrying",
            action=HealerAction.RETRY
        )

    def _handle_generic_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle generic caption errors by enabling fallback."""
        self.log_attempt("Generic caption error, enabling audio fallback...")

        self.retry_count += 1

        if self.retry_count > 3:
            # Too many retries - just skip to audio fallback
            caption_config = self._get_caption_config()
            if caption_config:
                if hasattr(caption_config, 'fallback_to_audio'):
                    caption_config.fallback_to_audio = True
                elif isinstance(caption_config, dict):
                    caption_config['fallback_to_audio'] = True

            self.log_success("Enabled audio fallback after multiple caption failures")
            return HealerResult.config_changed(
                "Enabled audio fallback after caption failures",
                retry_count=self.retry_count,
                fallback_enabled=True
            )

        time.sleep(self.INITIAL_BACKOFF)
        return HealerResult.fixed(
            f"Caption error, retry attempt {self.retry_count}",
            action=HealerAction.RETRY,
            retry_count=self.retry_count
        )

    def _get_caption_config(self):
        """Get caption_first config section."""
        download_config = getattr(self.config, 'download', None)
        if download_config:
            return getattr(download_config, 'caption_first', None)
        return None

    def _extract_video_id(self, error_str: str) -> str | None:
        """Extract video ID from error message."""
        import re

        patterns = [
            r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:\?|&|$|/)',
            r'Video ID: ([a-zA-Z0-9_-]{11})',
            r"'([a-zA-Z0-9_-]{11})'",
            r'"([a-zA-Z0-9_-]{11})"',
            r'([a-zA-Z0-9_-]{11})\.(?:srt|vtt|ass)',
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str)
            if match:
                return match.group(1)

        return None

    def reset(self):
        """Reset healer state."""
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0
        self.language_attempt_index = 0

    def get_skipped_videos(self) -> Set[str]:
        """Get set of videos that were skipped (no captions)."""
        return self.skipped_videos.copy()
