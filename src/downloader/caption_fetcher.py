"""
Caption fetcher for YouTube videos using yt-dlp.

Fetches subtitles/captions without downloading video or audio,
enabling transcript-first matching pipeline.

Uses CaptionCache for unified caching behavior with index-based tracking.
"""

from __future__ import annotations

import subprocess
import logging
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from . import utils

if TYPE_CHECKING:
    from ..config import Config
    from ..cache import CaptionCache

logger = logging.getLogger(__name__)


@dataclass
class CaptionInfo:
    """Information about available captions for a video"""
    video_id: str
    has_manual: bool = False
    has_auto: bool = False
    manual_languages: List[str] = field(default_factory=list)
    auto_languages: List[str] = field(default_factory=list)

    @property
    def has_captions(self) -> bool:
        return self.has_manual or self.has_auto

    def best_language(self, preferred: List[str] = None) -> Optional[str]:
        """Get best available language, preferring manual captions"""
        preferred = preferred or ["en", "en-US", "en-GB"]

        # Check manual captions first
        for lang in preferred:
            if lang in self.manual_languages:
                return lang

        # Fall back to auto captions
        for lang in preferred:
            # Auto captions often have "-auto" or "-orig" suffix
            for auto_lang in self.auto_languages:
                if auto_lang.startswith(lang):
                    return auto_lang

        # Return first available
        if self.manual_languages:
            return self.manual_languages[0]
        if self.auto_languages:
            return self.auto_languages[0]

        return None


@dataclass
class CaptionResult:
    """Result of caption fetch operation"""
    video_id: str
    file: str  # Path to downloaded caption file
    language: str
    is_auto_generated: bool
    format: str = "srt"  # srt, vtt, etc.


class CaptionFetcher:
    """
    Fetches YouTube captions/subtitles using yt-dlp.

    Supports both manual and auto-generated captions,
    with preference for manual when available.

    Uses CaptionCache for indexed caching with proper TTL and validation.
    """

    def __init__(self, config: 'Config', cache_dir: Optional[Path] = None):
        """
        Initialize CaptionFetcher.

        Args:
            config: Config object for cookie settings
            cache_dir: Directory to store downloaded captions
        """
        self.config = config
        self.cache_dir = cache_dir or Path(".cache/captions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize indexed cache for caption metadata
        from ..cache import CaptionCache
        self._cache = CaptionCache(
            cache_dir=self.cache_dir,
            ttl_hours=0,  # No expiration for captions
            auto_save=True
        )

        # Migrate any existing caption files to the indexed cache
        self._cache.migrate_from_files()

    def check_caption_availability(
        self,
        video_ids: List[str],
        timeout: int = 30
    ) -> Dict[str, CaptionInfo]:
        """
        Check caption availability for multiple videos.

        Uses yt-dlp --dump-json to get subtitle info without downloading.

        Args:
            video_ids: List of YouTube video IDs
            timeout: Timeout per video in seconds

        Returns:
            Dict mapping video_id to CaptionInfo
        """
        results = {}

        for video_id in video_ids:
            url = f"https://www.youtube.com/watch?v={video_id}"

            # Use tv_embedded client to bypass PO Token requirement for subtitles
            cmd = [
                'yt-dlp',
                url,
                '--sleep-interval', '5',
                '--dump-json',
                '--skip-download',
                '--no-warnings',
                '--no-playlist',
                '--extractor-args', 'youtube:player_client=tv_embedded',
            ]

            # Add base args (JS runtime for challenge solving) and cookies
            cmd.extend(utils.get_ytdlp_base_args())
            cmd.extend(utils.get_cookies_args(self.config))

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode == 0 and result.stdout:
                    info = json.loads(result.stdout)
                    results[video_id] = self._parse_subtitle_info(video_id, info)
                else:
                    logger.debug(f"No caption info for {video_id}: {result.stderr[:200] if result.stderr else 'no output'}")
                    results[video_id] = CaptionInfo(video_id=video_id)

            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout checking captions for {video_id}")
                results[video_id] = CaptionInfo(video_id=video_id)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for {video_id}: {e}")
                results[video_id] = CaptionInfo(video_id=video_id)
            except Exception as e:
                logger.warning(f"Error checking captions for {video_id}: {e}")
                results[video_id] = CaptionInfo(video_id=video_id)

        return results

    def _parse_subtitle_info(self, video_id: str, info: dict) -> CaptionInfo:
        """Parse yt-dlp JSON output to extract subtitle information"""
        caption_info = CaptionInfo(video_id=video_id)

        # Manual subtitles
        subtitles = info.get('subtitles', {})
        if subtitles:
            caption_info.has_manual = True
            caption_info.manual_languages = list(subtitles.keys())

        # Auto-generated subtitles
        auto_captions = info.get('automatic_captions', {})
        if auto_captions:
            caption_info.has_auto = True
            caption_info.auto_languages = list(auto_captions.keys())

        return caption_info

    def fetch_captions(
        self,
        video_id: str,
        languages: List[str] = None,
        prefer_manual: bool = True,
        timeout: int = 60
    ) -> Optional[CaptionResult]:
        """
        Fetch captions for a single video.

        Args:
            video_id: YouTube video ID
            languages: Preferred languages (default: ["en", "en-US"])
            prefer_manual: Prefer manual captions over auto-generated
            timeout: Download timeout in seconds

        Returns:
            CaptionResult if successful, None otherwise
        """
        languages = languages or ["en", "en-US", "en-GB"]
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Check for cached caption
        cached = self._check_cache(video_id, languages)
        if cached:
            logger.debug(f"Using cached caption for {video_id}")
            return cached

        # Build language string for yt-dlp
        lang_str = ",".join(languages)

        # Build yt-dlp command
        # Use tv_embedded client to bypass PO Token requirement for subtitles
        # (YouTube requires PO Token for web/mweb clients as of late 2024)
        cmd = [
            'yt-dlp',
            url,
            '--sleep-interval', '5',
            '--skip-download',
            '--no-playlist',
            '--no-warnings',
            '--extractor-args', 'youtube:player_client=tv_embedded',
        ]

        # Add subtitle options based on preference
        if prefer_manual:
            # Try manual first, then auto
            cmd.extend([
                '--write-sub',
                '--write-auto-sub',
            ])
        else:
            # Try auto first (faster, usually available)
            cmd.extend([
                '--write-auto-sub',
                '--write-sub',
            ])

        cmd.extend([
            '--sub-lang', lang_str,
            '--sub-format', 'srt/vtt/best',
            '--convert-subs', 'srt',  # Convert to SRT format
            '-o', str(self.cache_dir / f'{video_id}.%(ext)s'),
        ])

        # Add base args (JS runtime for challenge solving) and cookies
        cmd.extend(utils.get_ytdlp_base_args())
        cmd.extend(utils.get_cookies_args(self.config))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                logger.debug(f"yt-dlp caption fetch failed for {video_id}: {result.stderr[:200] if result.stderr else 'unknown error'}")
                return None

            # Find the downloaded caption file
            caption_file = self._find_caption_file(video_id, languages)
            if caption_file:
                is_auto = '.auto.' in caption_file.name or '-auto' in caption_file.name
                lang = self._extract_language(caption_file.name, video_id)

                logger.debug(f"Fetched {'auto' if is_auto else 'manual'} caption for {video_id}: {caption_file.name}")

                # Add to indexed cache
                self._cache.set_caption(
                    video_id=video_id,
                    file=str(caption_file),
                    language=lang,
                    is_auto_generated=is_auto,
                    format="srt"
                )

                return CaptionResult(
                    video_id=video_id,
                    file=str(caption_file),
                    language=lang,
                    is_auto_generated=is_auto,
                    format="srt"
                )
            else:
                logger.debug(f"No caption file found for {video_id}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout fetching captions for {video_id}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching captions for {video_id}: {e}")
            return None

    def _check_cache(self, video_id: str, languages: List[str]) -> Optional[CaptionResult]:
        """Check if caption is already cached using indexed cache"""
        # First check the indexed cache
        cached_entry = self._cache.find_caption_by_language(video_id, languages)
        if cached_entry:
            return CaptionResult(
                video_id=video_id,
                file=cached_entry.file,
                language=cached_entry.language,
                is_auto_generated=cached_entry.is_auto_generated,
                format=cached_entry.format
            )

        # Fallback: check for files not yet indexed (backward compatibility)
        for lang in languages:
            patterns = [
                f"{video_id}.{lang}.srt",
                f"{video_id}.{lang}-*.srt",
            ]

            for pattern in patterns:
                matches = list(self.cache_dir.glob(pattern))
                if matches:
                    caption_file = matches[0]
                    is_auto = '.auto.' in caption_file.name or '-auto' in caption_file.name

                    # Add to index for future lookups
                    self._cache.set_caption(
                        video_id=video_id,
                        file=str(caption_file),
                        language=lang,
                        is_auto_generated=is_auto,
                        format="srt"
                    )

                    return CaptionResult(
                        video_id=video_id,
                        file=str(caption_file),
                        language=lang,
                        is_auto_generated=is_auto,
                        format="srt"
                    )

        return None

    def _find_caption_file(self, video_id: str, languages: List[str]) -> Optional[Path]:
        """Find the downloaded caption file"""
        # yt-dlp saves as: video_id.lang.srt or video_id.lang-auto.srt

        # First try preferred languages in order
        for lang in languages:
            # Manual caption patterns
            patterns = [
                f"{video_id}.{lang}.srt",
                f"{video_id}.{lang}-orig.srt",
            ]
            for pattern in patterns:
                matches = list(self.cache_dir.glob(pattern))
                if matches:
                    return matches[0]

        # Then try auto-generated
        for lang in languages:
            patterns = [
                f"{video_id}.{lang}-auto.srt",
                f"{video_id}.{lang}*.srt",
            ]
            for pattern in patterns:
                matches = list(self.cache_dir.glob(pattern))
                if matches:
                    return matches[0]

        # Finally, any caption for this video
        all_captions = list(self.cache_dir.glob(f"{video_id}.*.srt"))
        if all_captions:
            return all_captions[0]

        return None

    def _extract_language(self, filename: str, video_id: str) -> str:
        """Extract language code from caption filename"""
        # Pattern: video_id.lang.srt or video_id.lang-auto.srt
        base = filename.replace(video_id + ".", "").replace(".srt", "")
        # Remove -auto, -orig suffixes
        lang = base.split("-")[0] if "-" in base else base
        return lang or "en"

    def parse_caption_file(self, caption_path: str) -> List[dict]:
        """
        Parse a caption file (SRT or VTT) into transcript segments.

        Returns segments in the same format as Whisper transcription:
        [{"index": 0, "start": 0.0, "end": 2.5, "text": "...", "source_file": "..."}]

        Args:
            caption_path: Path to caption file

        Returns:
            List of transcript segment dicts
        """
        path = Path(caption_path)
        if not path.exists():
            logger.error(f"Caption file not found: {caption_path}")
            return []

        # Determine format
        suffix = path.suffix.lower()

        try:
            content = self._read_caption_file(path)
            if not content:
                return []

            if suffix == ".vtt":
                return self._parse_vtt(content, caption_path)
            else:
                # Default to SRT parsing
                return self._parse_srt(content, caption_path)

        except Exception as e:
            logger.error(f"Error parsing caption file {caption_path}: {e}")
            return []

    def _read_caption_file(self, path: Path) -> Optional[str]:
        """Read caption file with multiple encoding attempts"""
        encodings = ['utf-8', 'utf-16', 'utf-16-le', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                if content and ('-->' in content or '\n' in content):
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue

        # Last resort: binary read
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
                return raw.decode('utf-16', errors='ignore')
            return raw.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Could not read caption file: {e}")
            return None

    def _parse_srt(self, content: str, source_file: str) -> List[dict]:
        """Parse SRT format caption content"""
        segments = []

        # Split by double newline (segment separator)
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue

            # Find timestamp line (contains -->)
            timestamp_idx = None
            for i, line in enumerate(lines):
                if '-->' in line:
                    timestamp_idx = i
                    break

            if timestamp_idx is None:
                continue

            # Parse timestamp
            timestamp_line = lines[timestamp_idx]
            times = timestamp_line.split('-->')
            if len(times) != 2:
                continue

            try:
                start = self._parse_timestamp(times[0].strip())
                end = self._parse_timestamp(times[1].strip().split()[0])  # Handle position info
            except ValueError:
                continue

            # Text is everything after timestamp line
            text_lines = lines[timestamp_idx + 1:]
            text = ' '.join(line.strip() for line in text_lines if line.strip())

            # Clean up HTML tags and special characters
            text = self._clean_caption_text(text)

            if text:
                segments.append({
                    'index': len(segments),
                    'start': start,
                    'end': end,
                    'text': text,
                    'source_file': source_file
                })

        return segments

    def _parse_vtt(self, content: str, source_file: str) -> List[dict]:
        """Parse VTT format caption content"""
        segments = []

        # Remove WEBVTT header and metadata
        lines = content.split('\n')
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 0:
                content_start = i + 1
                break

        # Join back and parse like SRT
        remaining = '\n'.join(lines[content_start:])
        return self._parse_srt(remaining, source_file)

    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert SRT/VTT timestamp to seconds"""
        timestamp = timestamp.strip().replace(',', '.')

        parts = timestamp.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            return float(timestamp)

    def _clean_caption_text(self, text: str) -> str:
        """Clean up caption text - remove HTML tags, normalize whitespace"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove speaker labels like "[Speaker 1]:" or "(narrator):"
        text = re.sub(r'^\s*[\[\(][^\]\)]+[\]\)]\s*:?\s*', '', text)

        # Remove music/sound effect indicators: [Music], (applause), etc.
        text = re.sub(r'[\[\(][^\]\)]*(?:music|applause|laughter|silence|inaudible)[^\]\)]*[\]\)]', '', text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def fetch_batch(
        self,
        video_ids: List[str],
        languages: List[str] = None,
        prefer_manual: bool = True,
        progress_callback=None
    ) -> Tuple[List[CaptionResult], List[str]]:
        """
        Fetch captions for multiple videos.

        Args:
            video_ids: List of YouTube video IDs
            languages: Preferred languages
            prefer_manual: Prefer manual captions
            progress_callback: Optional callback(current, total, results)

        Returns:
            Tuple of (successful CaptionResults, video_ids that failed)
        """
        results = []
        failed = []

        total = len(video_ids)
        for i, video_id in enumerate(video_ids):
            result = self.fetch_captions(
                video_id,
                languages=languages,
                prefer_manual=prefer_manual
            )

            if result:
                results.append(result)
            else:
                failed.append(video_id)

            if progress_callback:
                progress_callback(i + 1, total, results)

        return results, failed

    # === Cache Management Methods ===

    @property
    def cache(self) -> 'CaptionCache':
        """Get the caption cache instance"""
        return self._cache

    def update_segment_count(self, video_id: str, segment_count: int, duration_covered: float = 0.0) -> None:
        """
        Update cached caption entry with segment count after parsing.

        Args:
            video_id: YouTube video ID
            segment_count: Number of parsed segments
            duration_covered: Total duration covered by captions
        """
        entry = self._cache.get_caption_info(video_id)
        if entry:
            self._cache.set_caption(
                video_id=video_id,
                file=entry.file,
                language=entry.language,
                is_auto_generated=entry.is_auto_generated,
                format=entry.format,
                segment_count=segment_count,
                duration_covered=duration_covered
            )

    def get_cache_statistics(self) -> Dict[str, any]:
        """
        Get caption cache statistics.

        Returns:
            Dict with cache metrics
        """
        return self._cache.get_statistics()

    def has_cached_caption(self, video_id: str) -> bool:
        """
        Check if a video has a valid cached caption.

        Args:
            video_id: YouTube video ID

        Returns:
            True if valid caption exists in cache
        """
        return self._cache.has_valid_caption(video_id)

    def cleanup_cache(self) -> int:
        """
        Clean up cache: remove expired entries and orphaned files.

        Returns:
            Number of items cleaned up
        """
        expired = self._cache.cleanup_expired()
        orphaned = self._cache.cleanup_orphaned_files()
        return expired + orphaned
