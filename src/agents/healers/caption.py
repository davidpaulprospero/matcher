"""
Caption Healer - Intelligent caption format discovery and fallback.

Handles:
- Caption format unavailable errors (e.g., json3 not available)
- No captions available (fallback to auto-generated)
- yt-dlp subtitle extraction errors
- Format discovery using --list-subs

This healer addresses the ~40 second delay caused by format fallback loops
by proactively discovering available formats before attempting download.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class SubtitleInfo:
    """Information about an available subtitle track."""
    language: str
    format: str
    is_auto_generated: bool
    name: Optional[str] = None


class CaptionHealer(Healer):
    """
    Heals caption-related errors.

    Recovery strategies:
    1. Format discovery: Use --list-subs to find available formats before retry
    2. Auto-caption fallback: Enable --write-auto-sub when manual captions unavailable
    3. Language fallback: Try alternative languages (en, en-US, a]auto] variants)
    4. Format negotiation: Suggest working format based on available options

    This healer reduces wasted time by proactively discovering what formats
    are available rather than trying each format sequentially.
    """

    name = "caption-healer"
    description = "Fix caption/subtitle fetch errors"

    error_patterns = [
        "no captions",
        "caption",
        "subtitle",
        "no subtitles",
        "format unavailable",
        "json3",
        "srv1",
        "srv2",
        "srv3",
        "vtt",
        "ttml",
        "there are no subtitles",
        "requested format is not available",
        "unable to download video subtitles",
        "--write-auto-sub",
    ]

    # Preferred format order (json3 is best for parsing, then srv variants, then vtt)
    FORMAT_PREFERENCE = ["json3", "srv3", "srv2", "srv1", "vtt", "ttml"]

    def __init__(self, config: 'Config', project_dir):
        super().__init__(config, project_dir)
        # Cache discovered formats to avoid repeated --list-subs calls
        self._format_cache: Dict[str, List[SubtitleInfo]] = {}
        # Track videos with no captions at all
        self._no_captions_videos: Set[str] = set()
        # Track auto-caption enabled state
        self._auto_caption_enabled: bool = False

    def can_handle(self, error: Exception, stage_name: str) -> bool:
        """
        Check if this healer can handle caption-related errors.

        Handles:
        - CaptionFormatUnavailableError: Specific format not available
        - CaptionUnavailableError: No captions at all
        - CaptionFetchError: General fetch failures that might be format-related
        - Generic errors with caption/subtitle keywords
        """
        # Check by exception type
        error_type = type(error).__name__
        if error_type in (
            "CaptionFormatUnavailableError",
            "CaptionUnavailableError",
            "CaptionFetchError",
        ):
            return True

        # Check error message patterns
        error_str = str(error).lower()
        for pattern in self.error_patterns:
            if pattern.lower() in error_str:
                return True

        return False

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """
        Attempt to fix caption-related errors.

        Strategy:
        1. Extract video ID from error
        2. Run --list-subs to discover available formats
        3. If formats available: suggest best format and retry
        4. If no manual captions: enable auto-caption and retry
        5. If no captions at all: skip video
        """
        error_str = str(error)
        video_id = self._extract_video_id(error_str, error)

        if not video_id:
            self.log_attempt("Could not extract video ID from error")
            return HealerResult.failed(
                "Could not determine video ID for format discovery"
            )

        # Check if we already know this video has no captions
        if video_id in self._no_captions_videos:
            return HealerResult.fixed(
                f"Video {video_id} known to have no captions, skipping",
                action=HealerAction.SKIP,
                video_id=video_id
            )

        # Discover available formats
        self.log_attempt(f"Discovering subtitle formats for {video_id}...")
        available = self._discover_formats(video_id)

        if not available:
            # No subtitles at all - try enabling auto-captions first
            if not self._auto_caption_enabled:
                self.log_attempt("No subtitles found, enabling auto-caption fallback...")
                self._auto_caption_enabled = True
                return self._enable_auto_captions()

            # Auto-captions already enabled but still no luck
            self._no_captions_videos.add(video_id)
            self.log_attempt(f"No captions available for {video_id} even with auto-sub")
            return HealerResult.fixed(
                f"No captions available for {video_id}, skipping",
                action=HealerAction.SKIP,
                video_id=video_id,
                no_captions=True
            )

        # Check for auto-generated captions
        auto_subs = [s for s in available if s.is_auto_generated]
        manual_subs = [s for s in available if not s.is_auto_generated]

        # Determine best available format
        best_format = self._select_best_format(available)

        if manual_subs:
            self.log_success(
                f"Found {len(manual_subs)} manual subtitle(s), "
                f"best format: {best_format.format}"
            )
        elif auto_subs:
            self.log_success(
                f"Found {len(auto_subs)} auto-generated subtitle(s), "
                f"best format: {best_format.format}"
            )
            # Ensure auto-caption mode is enabled
            if not self._auto_caption_enabled:
                self._auto_caption_enabled = True
                self._update_config_for_auto_subs()

        return HealerResult.config_changed(
            f"Discovered format '{best_format.format}' for {video_id}",
            video_id=video_id,
            discovered_format=best_format.format,
            language=best_format.language,
            is_auto_generated=best_format.is_auto_generated,
            available_formats=[s.format for s in available],
            manual_count=len(manual_subs),
            auto_count=len(auto_subs),
        )

    def _extract_video_id(self, error_str: str, error: Exception) -> Optional[str]:
        """Extract video ID from error message or exception attributes."""
        # Check exception attributes first
        if hasattr(error, 'video_id'):
            return error.video_id

        # Try common patterns in error message
        patterns = [
            r'video[_\s]?(?:id)?[:\s]+([a-zA-Z0-9_-]{11})',
            r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:[&?/]|$)',
            r"'([a-zA-Z0-9_-]{11})'",
            r'"([a-zA-Z0-9_-]{11})"',
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _discover_formats(self, video_id: str) -> List[SubtitleInfo]:
        """
        Discover available subtitle formats using yt-dlp --list-subs.

        Args:
            video_id: YouTube video ID

        Returns:
            List of SubtitleInfo for available subtitles
        """
        # Check cache first
        if video_id in self._format_cache:
            return self._format_cache[video_id]

        try:
            cmd = [
                "yt-dlp",
                "--list-subs",
                "--skip-download",
                f"https://www.youtube.com/watch?v={video_id}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30,
            )

            available = self._parse_list_subs_output(result.stdout, result.stderr)
            self._format_cache[video_id] = available
            return available

        except subprocess.TimeoutExpired:
            self.log_attempt(f"Timeout discovering formats for {video_id}")
            return []
        except Exception as e:
            self.log_attempt(f"Error discovering formats: {e}")
            return []

    def _parse_list_subs_output(
        self,
        stdout: str,
        stderr: str
    ) -> List[SubtitleInfo]:
        """
        Parse yt-dlp --list-subs output to extract available formats.

        Output format example:
        Available subtitles for VIDEO_ID:
        Language  formats
        en        json3, srv1, srv2, srv3, ttml, vtt

        Available automatic captions for VIDEO_ID:
        Language  formats
        af        json3, srv1, srv2, srv3, ttml, vtt
        en        json3, srv1, srv2, srv3, ttml, vtt
        ...
        """
        combined = stdout + "\n" + stderr
        available: List[SubtitleInfo] = []
        is_auto_section = False

        for line in combined.split('\n'):
            line = line.strip()

            # Detect section headers
            if 'automatic captions' in line.lower():
                is_auto_section = True
                continue
            elif 'available subtitles' in line.lower():
                is_auto_section = False
                continue
            elif line.startswith('Language') or not line:
                continue

            # Parse language/format lines: "en   json3, srv1, srv2, srv3, ttml, vtt"
            # or "en-US  json3, srv1, srv2, srv3, ttml, vtt  English (United States)"
            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) >= 2:
                language = parts[0]
                # Skip header-like entries
                if language.lower() in ('language', 'formats'):
                    continue

                rest = parts[1]
                # Extract formats (comma-separated list before any description)
                format_match = re.match(r'^([\w, ]+)', rest)
                if format_match:
                    formats_str = format_match.group(1)
                    formats = [f.strip() for f in formats_str.split(',') if f.strip()]

                    # Extract optional name (after formats)
                    name = None
                    name_match = re.search(r'\s+(\w[\w\s()]+)$', rest)
                    if name_match:
                        name = name_match.group(1).strip()

                    for fmt in formats:
                        # Skip if not a known format
                        if fmt not in self.FORMAT_PREFERENCE + ['json']:
                            continue
                        available.append(SubtitleInfo(
                            language=language,
                            format=fmt,
                            is_auto_generated=is_auto_section,
                            name=name,
                        ))

        return available

    def _select_best_format(self, available: List[SubtitleInfo]) -> SubtitleInfo:
        """
        Select the best subtitle format based on preference order.

        Prefers:
        1. Manual captions over auto-generated
        2. English variants (en, en-US, en-GB) over other languages
        3. json3 > srv3 > srv2 > srv1 > vtt > ttml
        """
        if not available:
            raise ValueError("No available formats to select from")

        # Separate by type
        manual = [s for s in available if not s.is_auto_generated]
        auto = [s for s in available if s.is_auto_generated]

        # Prefer manual if available
        candidates = manual if manual else auto

        # Prefer English variants
        english_candidates = [
            s for s in candidates
            if s.language.lower().startswith('en')
        ]
        if english_candidates:
            candidates = english_candidates

        # Sort by format preference
        def format_rank(sub: SubtitleInfo) -> int:
            try:
                return self.FORMAT_PREFERENCE.index(sub.format)
            except ValueError:
                return len(self.FORMAT_PREFERENCE)  # Unknown formats last

        candidates.sort(key=format_rank)
        return candidates[0]

    def _enable_auto_captions(self) -> HealerResult:
        """Enable auto-caption fallback in config."""
        download_config = getattr(self.config, 'download', None)
        if download_config:
            caption_config = getattr(download_config, 'caption_first', None)
            if caption_config:
                # Enable auto-sub fallback
                if hasattr(caption_config, 'enable_auto_subs'):
                    caption_config.enable_auto_subs = True
                elif isinstance(caption_config, dict):
                    caption_config['enable_auto_subs'] = True

        self.log_success("Enabled auto-caption fallback (--write-auto-sub)")
        return HealerResult.config_changed(
            "Enabled auto-caption fallback for videos without manual captions",
            auto_subs_enabled=True
        )

    def _update_config_for_auto_subs(self):
        """Update config to prefer auto-subs when no manual available."""
        download_config = getattr(self.config, 'download', None)
        if download_config:
            caption_config = getattr(download_config, 'caption_first', None)
            if caption_config:
                if hasattr(caption_config, 'prefer_auto_subs_if_no_manual'):
                    caption_config.prefer_auto_subs_if_no_manual = True
                elif isinstance(caption_config, dict):
                    caption_config['prefer_auto_subs_if_no_manual'] = True

    def get_discovered_formats(self, video_id: str) -> List[SubtitleInfo]:
        """Get cached format discovery results for a video."""
        return self._format_cache.get(video_id, [])

    def get_no_caption_videos(self) -> Set[str]:
        """Get set of video IDs known to have no captions."""
        return self._no_captions_videos.copy()

    def reset_discovery_cache(self):
        """Clear format discovery cache."""
        self._format_cache.clear()
        self._no_captions_videos.clear()
        self._auto_caption_enabled = False
