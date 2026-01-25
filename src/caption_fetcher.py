"""
YouTube caption fetcher module.

Fetches captions/subtitles from YouTube videos using yt-dlp.
This is the foundation for caption-first mode, enabling transcript retrieval
without downloading video content.

Usage:
    fetcher = CaptionFetcher()
    result = fetcher.fetch_captions("dQw4w9WgXcQ")
    for segment in result.segments:
        print(f"{segment.start_time:.2f} -> {segment.end_time:.2f}: {segment.text}")

    # With caching:
    cache = CaptionCache(config.download.caption_first)
    cached = cache.get("dQw4w9WgXcQ", "en")
    if cached:
        print(f"Cache hit: {len(cached.segments)} segments")
    else:
        result = fetcher.fetch_captions("dQw4w9WgXcQ")
        cache.store(result)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .cache import BaseCache, CacheEntry

if TYPE_CHECKING:
    from .config import Config
    from .config.sections.download import CaptionFirstConfig

logger = logging.getLogger(__name__)


class CaptionError(Exception):
    """Base exception for caption-related errors."""
    pass


class CaptionUnavailableError(CaptionError):
    """Raised when captions are not available for a video.

    This indicates the video genuinely has no captions (auto or manual),
    as opposed to a temporary fetch failure.
    """
    def __init__(self, video_id: str, reason: str = ""):
        self.video_id = video_id
        self.reason = reason
        message = f"No captions available for video {video_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class CaptionFetchError(CaptionError):
    """Raised when caption fetch fails due to a temporary/network error.

    This indicates a potentially retryable failure, not that captions
    don't exist.
    """
    def __init__(self, video_id: str, reason: str = ""):
        self.video_id = video_id
        self.reason = reason
        message = f"Failed to fetch captions for video {video_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


@dataclass
class CaptionSegment:
    """A single caption segment with timing information.

    Compatible with TranscriptSegment for downstream matching.
    """
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""  # Video ID or path

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'start': self.start_time,
            'end': self.end_time,
            'text': self.text,
            'source_file': self.source_file,
        }


@dataclass
class CaptionResult:
    """Result of a caption fetch operation."""
    video_id: str
    segments: List[CaptionSegment] = field(default_factory=list)
    language: str = ""  # ISO 639-1 code (e.g., 'en')
    is_auto_generated: bool = False
    format_source: str = ""  # 'vtt', 'srv3', 'json3', etc.

    @property
    def text(self) -> str:
        """Get full caption text concatenated."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration(self) -> float:
        """Get total duration covered by captions."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end_time - self.segments[0].start_time

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
            'segments': [seg.to_dict() for seg in self.segments],
            'language': self.language,
            'is_auto_generated': self.is_auto_generated,
            'format_source': self.format_source,
        }


@dataclass
class AvailableLanguage:
    """Represents an available caption language for a video."""
    code: str  # ISO 639-1 code (e.g., 'en', 'es', 'fr')
    name: str  # Human-readable name (e.g., 'English', 'Spanish')
    is_auto_generated: bool  # True if auto-generated captions


class CaptionFetcher:
    """Fetches YouTube captions using yt-dlp.

    Supports multiple caption formats (VTT, SRT, JSON3) and auto-converts
    them to a common segment format.

    Example:
        fetcher = CaptionFetcher()
        result = fetcher.fetch_captions("dQw4w9WgXcQ")
        print(f"Found {len(result.segments)} segments, auto={result.is_auto_generated}")

        # List available languages first
        languages = fetcher.list_available_languages("dQw4w9WgXcQ")
        for lang in languages:
            print(f"{lang.code}: {lang.name} (auto={lang.is_auto_generated})")
    """

    def __init__(self, config: Optional['Config'] = None):
        """Initialize the caption fetcher.

        Args:
            config: Optional config for cookies and other settings.
        """
        self.config = config
        self._timeout = 60  # seconds

    def list_available_languages(self, video_id: str) -> List[AvailableLanguage]:
        """List available caption languages for a YouTube video.

        Uses yt-dlp to query subtitle metadata without downloading.
        Returns both manual and auto-generated caption languages.

        Args:
            video_id: YouTube video ID (11 characters).

        Returns:
            List of AvailableLanguage objects, sorted with manual captions first,
            then auto-generated. Within each group, sorted by language code.

        Raises:
            CaptionFetchError: If unable to query video metadata.

        Example:
            languages = fetcher.list_available_languages("dQw4w9WgXcQ")
            # [AvailableLanguage(code='en', name='English', is_auto_generated=False),
            #  AvailableLanguage(code='en', name='English (auto-generated)', is_auto_generated=True)]
        """
        if not self._is_valid_video_id(video_id):
            raise CaptionFetchError(video_id, f"Invalid video ID format: {video_id}")

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        cmd = [
            'yt-dlp',
            video_url,
            '--skip-download',
            '--list-subs',
            '--no-playlist',
            '--no-warnings',
        ]

        # Add cookies if configured
        cmd.extend(self._get_cookies_args())

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout
            )

            # Parse the output to extract available languages
            return self._parse_list_subs_output(result.stdout, result.stderr)

        except subprocess.TimeoutExpired:
            raise CaptionFetchError(video_id, f"Timeout after {self._timeout}s")
        except Exception as e:
            raise CaptionFetchError(video_id, str(e))

    def _parse_list_subs_output(
        self,
        stdout: str,
        stderr: str
    ) -> List[AvailableLanguage]:
        """Parse yt-dlp --list-subs output to extract available languages.

        The output format looks like:
            [info] Available subtitles for VIDEO_ID:
            Language  Name                 Formats
            en        English              vtt, ttml, srv3, srv2, srv1, json3
            es        Spanish              vtt, ttml, srv3, srv2, srv1, json3

            [info] Available automatic captions for VIDEO_ID:
            Language  Name                              Formats
            en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3

        Args:
            stdout: Standard output from yt-dlp.
            stderr: Standard error from yt-dlp.

        Returns:
            List of AvailableLanguage objects.
        """
        languages = []
        combined_output = stdout + "\n" + stderr

        # Track which section we're in
        in_manual_section = False
        in_auto_section = False

        # Pattern to match language lines (after header)
        # Format: "en        English              vtt, ttml, ..."
        lang_line_pattern = re.compile(
            r'^([a-z]{2,3}(?:-[A-Za-z]{2,4})?)\s+(.+?)\s+(?:vtt|ttml|srv|json)',
            re.IGNORECASE
        )

        for line in combined_output.split('\n'):
            line = line.strip()

            # Detect section headers
            if 'Available subtitles' in line:
                in_manual_section = True
                in_auto_section = False
                continue
            elif 'Available automatic captions' in line:
                in_manual_section = False
                in_auto_section = True
                continue
            elif line.startswith('[info]') or line.startswith('Language'):
                # Skip info lines and headers
                continue
            elif not line:
                # Empty line might end a section
                continue

            # Try to parse language line
            match = lang_line_pattern.match(line)
            if match:
                lang_code = match.group(1).lower()
                lang_name = match.group(2).strip()

                # Determine if auto-generated
                is_auto = in_auto_section or '(auto' in lang_name.lower()

                # Clean up the name
                if '(auto-generated)' in lang_name:
                    display_name = lang_name
                elif is_auto:
                    display_name = f"{lang_name} (auto-generated)"
                else:
                    display_name = lang_name

                languages.append(AvailableLanguage(
                    code=lang_code,
                    name=display_name,
                    is_auto_generated=is_auto
                ))

        # Sort: manual captions first, then auto-generated, alphabetically within each
        languages.sort(key=lambda x: (x.is_auto_generated, x.code))

        logger.debug(f"Found {len(languages)} available caption languages")
        return languages

    def select_best_language(
        self,
        available: List[AvailableLanguage],
        preferred: Optional[str] = None,
        fallback_to_english: bool = True,
        prefer_manual: bool = True
    ) -> Optional[AvailableLanguage]:
        """Select the best language from available options using fallback chain.

        Implements the fallback chain:
        1. Preferred language (manual if prefer_manual, else any)
        2. English 'en' (manual if prefer_manual, else any)
        3. Any available language (manual if prefer_manual, else any)

        Args:
            available: List of available languages from list_available_languages().
            preferred: Preferred language code (e.g., 'en', 'es'). If None, uses
                       config.download.caption_first.preferred_language or
                       config.transcription.language.
            fallback_to_english: If True, fall back to English if preferred unavailable.
            prefer_manual: If True, prefer manual captions over auto-generated.

        Returns:
            Selected AvailableLanguage, or None if no languages available.

        Example:
            languages = fetcher.list_available_languages("dQw4w9WgXcQ")
            best = fetcher.select_best_language(languages, preferred='es')
            if best:
                result = fetcher.fetch_captions("dQw4w9WgXcQ", language=best.code)
        """
        if not available:
            logger.warning("No caption languages available to select from")
            return None

        # Determine preferred language from args or config
        if preferred is None:
            preferred = self._get_preferred_language_from_config()

        logger.debug(f"Selecting caption language: preferred={preferred}, "
                    f"fallback_english={fallback_to_english}, prefer_manual={prefer_manual}")

        def find_language(code: str, manual_only: bool = False) -> Optional[AvailableLanguage]:
            """Find a language by code, optionally filtering to manual only."""
            for lang in available:
                if lang.code.lower() == code.lower():
                    if manual_only and lang.is_auto_generated:
                        continue
                    return lang
            return None

        # Step 1: Try preferred language
        if preferred:
            # Try manual first if preferred
            if prefer_manual:
                result = find_language(preferred, manual_only=True)
                if result:
                    logger.info(f"Selected preferred language: {result.code} (manual)")
                    return result

            # Try auto if manual not found
            result = find_language(preferred, manual_only=False)
            if result:
                logger.info(f"Selected preferred language: {result.code} "
                           f"({'auto' if result.is_auto_generated else 'manual'})")
                return result

            logger.debug(f"Preferred language '{preferred}' not available")

        # Step 2: Fall back to English
        if fallback_to_english and (preferred is None or preferred.lower() != 'en'):
            if prefer_manual:
                result = find_language('en', manual_only=True)
                if result:
                    logger.info("Falling back to English (manual)")
                    return result

            result = find_language('en', manual_only=False)
            if result:
                logger.info(f"Falling back to English "
                           f"({'auto' if result.is_auto_generated else 'manual'})")
                return result

            logger.debug("English not available")

        # Step 3: Fall back to any available language
        if prefer_manual:
            manual_langs = [l for l in available if not l.is_auto_generated]
            if manual_langs:
                result = manual_langs[0]  # Already sorted by code
                logger.info(f"Falling back to any available: {result.code} (manual)")
                return result

        if available:
            result = available[0]  # Already sorted: manual first, then auto
            logger.info(f"Falling back to any available: {result.code} "
                       f"({'auto' if result.is_auto_generated else 'manual'})")
            return result

        logger.warning("No suitable caption language found")
        return None

    def _get_preferred_language_from_config(self) -> str:
        """Get preferred language from config.

        Checks in order:
        1. config.download.caption_first.preferred_language
        2. config.transcription.language
        3. Default to 'en'

        Returns:
            Language code (ISO 639-1).
        """
        if not self.config:
            return 'en'

        try:
            # First try caption_first.preferred_language
            caption_first = getattr(self.config.download, 'caption_first', None)
            if caption_first:
                preferred = getattr(caption_first, 'preferred_language', None)
                if preferred:
                    return preferred

            # Then try transcription.language
            transcription = getattr(self.config, 'transcription', None)
            if transcription:
                lang = getattr(transcription, 'language', None)
                if lang:
                    return lang
        except AttributeError:
            pass

        return 'en'

    def fetch_captions_auto_language(
        self,
        video_id: str,
        preferred_language: Optional[str] = None
    ) -> CaptionResult:
        """Fetch captions with automatic language selection.

        Combines list_available_languages and fetch_captions with intelligent
        language selection based on config and availability.

        Args:
            video_id: YouTube video ID.
            preferred_language: Optional override for preferred language.

        Returns:
            CaptionResult with captions in the best available language.

        Raises:
            CaptionUnavailableError: If no captions available in any language.
            CaptionFetchError: If fetch fails due to network/temporary error.
        """
        # List available languages
        available = self.list_available_languages(video_id)

        if not available:
            raise CaptionUnavailableError(
                video_id,
                "No captions available in any language"
            )

        # Select best language
        prefer_manual = True
        if self.config:
            caption_first = getattr(self.config.download, 'caption_first', None)
            if caption_first:
                prefer_manual = getattr(caption_first, 'prefer_human_captions', True)

        selected = self.select_best_language(
            available,
            preferred=preferred_language,
            fallback_to_english=True,
            prefer_manual=prefer_manual
        )

        if not selected:
            raise CaptionUnavailableError(
                video_id,
                "No suitable language found despite available captions"
            )

        # Fetch captions in selected language
        return self.fetch_captions(
            video_id,
            language=selected.code,
            prefer_manual=not selected.is_auto_generated
        )

    def fetch_captions(
        self,
        video_id: str,
        language: str = "en",
        prefer_manual: bool = True
    ) -> CaptionResult:
        """Fetch captions for a YouTube video.

        Args:
            video_id: YouTube video ID (11 characters).
            language: Preferred language code (ISO 639-1).
            prefer_manual: If True, prefer manually uploaded captions over auto-generated.

        Returns:
            CaptionResult with parsed segments and metadata.

        Raises:
            CaptionUnavailableError: If no captions exist for the video.
            CaptionFetchError: If fetch fails due to network/temporary error.
        """
        # Validate video ID format
        if not self._is_valid_video_id(video_id):
            raise CaptionFetchError(video_id, f"Invalid video ID format: {video_id}")

        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Create temp directory for subtitle files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Determine which subtitle to fetch
            # Try manual captions first if preferred, then auto-generated
            formats_to_try = []
            if prefer_manual:
                formats_to_try.append((language, False))  # Manual
                formats_to_try.append((language, True))   # Auto
            else:
                formats_to_try.append((language, True))   # Auto
                formats_to_try.append((language, False))  # Manual

            # Also try English as fallback if not already preferred
            if language != "en":
                formats_to_try.append(("en", False))
                formats_to_try.append(("en", True))

            result = None
            last_error = None

            for lang, auto in formats_to_try:
                try:
                    result = self._fetch_subtitle(
                        video_url, video_id, temp_path, lang, auto
                    )
                    if result and result.segments:
                        return result
                except CaptionUnavailableError:
                    continue  # Try next format
                except CaptionFetchError as e:
                    last_error = e
                    continue

            # No captions found in any format
            if last_error:
                raise last_error
            raise CaptionUnavailableError(
                video_id,
                f"No captions available in {language} or en"
            )

    def _fetch_subtitle(
        self,
        video_url: str,
        video_id: str,
        temp_dir: Path,
        language: str,
        auto_generated: bool
    ) -> Optional[CaptionResult]:
        """Fetch a specific subtitle track.

        Args:
            video_url: Full YouTube URL.
            video_id: Video ID for result metadata.
            temp_dir: Temporary directory for downloaded files.
            language: Language code.
            auto_generated: Whether to fetch auto-generated captions.

        Returns:
            CaptionResult if successful, None if no captions for this format.
        """
        # Build yt-dlp command
        sub_flag = '--write-auto-subs' if auto_generated else '--write-subs'
        output_template = str(temp_dir / '%(id)s.%(ext)s')

        cmd = [
            'yt-dlp',
            video_url,
            '--skip-download',  # Don't download video
            sub_flag,
            '--sub-lang', language,
            '--sub-format', 'json3/srv3/vtt/srt/best',  # Prefer structured formats
            '--convert-subs', 'vtt',  # Convert to VTT for parsing
            '-o', output_template,
            '--no-playlist',
            '--no-warnings',
        ]

        # Add cookies if configured
        cmd.extend(self._get_cookies_args())

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout
            )

            # Check for errors indicating no captions
            if result.returncode != 0:
                stderr_lower = result.stderr.lower()
                if any(phrase in stderr_lower for phrase in [
                    'no subtitles',
                    'no automatic captions',
                    'subtitles are disabled',
                    'video unavailable',
                    'private video',
                ]):
                    raise CaptionUnavailableError(video_id, result.stderr[:200])
                else:
                    raise CaptionFetchError(video_id, result.stderr[:200])

            # Find downloaded subtitle file
            sub_files = list(temp_dir.glob(f"{video_id}*.vtt"))
            if not sub_files:
                # Try other extensions
                sub_files = list(temp_dir.glob(f"{video_id}*"))
                sub_files = [f for f in sub_files if f.suffix in ['.vtt', '.srt', '.json3', '.srv3']]

            if not sub_files:
                return None  # No subtitle file created

            sub_file = sub_files[0]
            logger.debug(f"Found subtitle file: {sub_file}")

            # Parse the subtitle file
            segments = self._parse_subtitle_file(sub_file, video_id)

            if not segments:
                return None

            # Determine format from filename
            format_source = sub_file.suffix.lstrip('.')

            return CaptionResult(
                video_id=video_id,
                segments=segments,
                language=language,
                is_auto_generated=auto_generated,
                format_source=format_source
            )

        except subprocess.TimeoutExpired:
            raise CaptionFetchError(video_id, f"Timeout after {self._timeout}s")
        except (CaptionUnavailableError, CaptionFetchError):
            raise
        except Exception as e:
            raise CaptionFetchError(video_id, str(e))

    def _parse_subtitle_file(
        self,
        file_path: Path,
        video_id: str
    ) -> List[CaptionSegment]:
        """Parse a subtitle file into segments.

        Supports VTT, SRT, and JSON3/SRV3 formats.

        Args:
            file_path: Path to subtitle file.
            video_id: Video ID for segment source_file field.

        Returns:
            List of CaptionSegment objects.
        """
        suffix = file_path.suffix.lower()

        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='utf-8', errors='replace')

        if suffix == '.vtt':
            return self._parse_vtt(content, video_id)
        elif suffix == '.srt':
            return self._parse_srt(content, video_id)
        elif suffix in ['.json3', '.srv3', '.json']:
            return self._parse_json3(content, video_id)
        else:
            logger.warning(f"Unknown subtitle format: {suffix}, trying VTT parser")
            return self._parse_vtt(content, video_id)

    def _parse_vtt(self, content: str, video_id: str) -> List[CaptionSegment]:
        """Parse VTT (WebVTT) format captions.

        VTT format:
            WEBVTT

            00:00:01.000 --> 00:00:04.000
            Hello, world!

            00:00:05.000 --> 00:00:08.000
            This is a test.
        """
        segments = []
        lines = content.split('\n')

        # Skip header
        i = 0
        while i < len(lines) and not '-->' in lines[i]:
            i += 1

        # VTT timestamp pattern: HH:MM:SS.mmm or MM:SS.mmm
        timestamp_pattern = re.compile(
            r'(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})'
        )

        current_index = 0
        while i < len(lines):
            line = lines[i].strip()

            match = timestamp_pattern.search(line)
            if match:
                # Parse timestamps
                start_time = self._parse_timestamp(match.group(0).split('-->')[0].strip())
                end_time = self._parse_timestamp(match.group(0).split('-->')[1].strip())

                # Collect text lines until empty line or next timestamp
                i += 1
                text_lines = []
                while i < len(lines):
                    text_line = lines[i].strip()
                    if not text_line:
                        i += 1
                        break
                    if timestamp_pattern.search(text_line):
                        break
                    # Skip VTT style tags
                    text_line = re.sub(r'<[^>]+>', '', text_line)
                    if text_line:
                        text_lines.append(text_line)
                    i += 1

                text = ' '.join(text_lines).strip()
                if text:
                    segments.append(CaptionSegment(
                        index=current_index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                        source_file=video_id
                    ))
                    current_index += 1
            else:
                i += 1

        return segments

    def _parse_srt(self, content: str, video_id: str) -> List[CaptionSegment]:
        """Parse SRT (SubRip) format captions.

        SRT format:
            1
            00:00:01,000 --> 00:00:04,000
            Hello, world!

            2
            00:00:05,000 --> 00:00:08,000
            This is a test.
        """
        segments = []
        blocks = re.split(r'\n\s*\n', content.strip())

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 2:
                continue

            # Find timestamp line
            timestamp_line = None
            text_start_idx = 0
            for idx, line in enumerate(lines):
                if '-->' in line:
                    timestamp_line = line
                    text_start_idx = idx + 1
                    break

            if not timestamp_line:
                continue

            # Parse timestamps
            parts = timestamp_line.split('-->')
            if len(parts) != 2:
                continue

            start_time = self._parse_timestamp(parts[0].strip())
            end_time = self._parse_timestamp(parts[1].strip())

            # Get text
            text = ' '.join(lines[text_start_idx:]).strip()
            text = re.sub(r'<[^>]+>', '', text)  # Remove tags

            if text and start_time is not None and end_time is not None:
                segments.append(CaptionSegment(
                    index=len(segments),
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    source_file=video_id
                ))

        return segments

    def _parse_json3(self, content: str, video_id: str) -> List[CaptionSegment]:
        """Parse JSON3/SRV3 format captions from YouTube.

        JSON3 format has 'events' array with 'segs' containing text segments.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON3 caption format")
            return []

        segments = []
        events = data.get('events', [])

        for event in events:
            if 'segs' not in event:
                continue

            start_ms = event.get('tStartMs', 0)
            duration_ms = event.get('dDurationMs', 0)

            # Combine all segs text
            text_parts = []
            for seg in event['segs']:
                if 'utf8' in seg:
                    text_parts.append(seg['utf8'])

            text = ''.join(text_parts).strip()
            text = text.replace('\n', ' ')

            if text:
                segments.append(CaptionSegment(
                    index=len(segments),
                    start_time=start_ms / 1000.0,
                    end_time=(start_ms + duration_ms) / 1000.0,
                    text=text,
                    source_file=video_id
                ))

        return segments

    def _parse_timestamp(self, ts: str) -> Optional[float]:
        """Parse a timestamp string to seconds.

        Supports formats:
            HH:MM:SS,mmm (SRT)
            HH:MM:SS.mmm (VTT)
            MM:SS.mmm (VTT short)
        """
        ts = ts.strip()

        # Replace comma with period for SRT format
        ts = ts.replace(',', '.')

        # Pattern for HH:MM:SS.mmm or MM:SS.mmm
        match = re.match(r'^(?:(\d+):)?(\d+):(\d+)\.(\d+)$', ts)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            millis = int(match.group(4).ljust(3, '0')[:3])  # Ensure 3 digits

            return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

        # Try simpler pattern without milliseconds
        match = re.match(r'^(?:(\d+):)?(\d+):(\d+)$', ts)
        if match:
            hours = int(match.group(1)) if match.group(1) else 0
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            return hours * 3600 + minutes * 60 + seconds

        logger.warning(f"Could not parse timestamp: {ts}")
        return None

    def _get_cookies_args(self) -> List[str]:
        """Get yt-dlp cookie arguments from config."""
        if not self.config:
            return []

        try:
            download_config = self.config.download

            # Prefer browser cookies
            cookies_browser = getattr(download_config, 'cookies_from_browser', '')
            if cookies_browser:
                return ['--cookies-from-browser', cookies_browser]

            # Fall back to cookies file
            cookies_path = getattr(download_config, 'cookies_path', '')
            if cookies_path and Path(cookies_path).exists():
                return ['--cookies', cookies_path]
        except AttributeError:
            pass

        return []

    def _is_valid_video_id(self, video_id: str) -> bool:
        """Validate YouTube video ID format.

        YouTube IDs are 11 characters: [A-Za-z0-9_-]
        """
        if not video_id or len(video_id) != 11:
            return False
        return bool(re.match(r'^[A-Za-z0-9_-]{11}$', video_id))


@dataclass
class CachedCaption:
    """Cached caption data for cross-project reuse.

    Stores the full caption result along with fetch metadata.
    """
    video_id: str
    language: str
    segments: List[Dict[str, Any]]  # CaptionSegment.to_dict() format
    is_auto_generated: bool
    format_source: str
    fetch_timestamp: float  # Unix timestamp when fetched
    duration: float = 0.0  # Total caption duration

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'video_id': self.video_id,
            'language': self.language,
            'segments': self.segments,
            'is_auto_generated': self.is_auto_generated,
            'format_source': self.format_source,
            'fetch_timestamp': self.fetch_timestamp,
            'duration': self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CachedCaption':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_caption_result(self) -> CaptionResult:
        """Convert cached data back to CaptionResult."""
        segments = [
            CaptionSegment(
                index=seg.get('index', i),
                start_time=seg.get('start', seg.get('start_time', 0.0)),
                end_time=seg.get('end', seg.get('end_time', 0.0)),
                text=seg.get('text', ''),
                source_file=seg.get('source_file', self.video_id),
            )
            for i, seg in enumerate(self.segments)
        ]
        return CaptionResult(
            video_id=self.video_id,
            segments=segments,
            language=self.language,
            is_auto_generated=self.is_auto_generated,
            format_source=self.format_source,
        )


class CaptionCache(BaseCache):
    """Cache for YouTube captions, enabling cross-project reuse.

    Caches fetched captions by video_id + language code to avoid
    re-fetching the same captions across different projects.

    Features:
    - JSON index for fast lookups
    - Age-based expiration (configurable days)
    - Cache key: video_id_language (e.g., 'dQw4w9WgXcQ_en')
    - Stores caption text, timing info, is_auto_generated, fetch_timestamp

    Usage:
        cache = CaptionCache(config.download.caption_first)

        # Check cache before fetching
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        if cached:
            result = cached.to_caption_result()
        else:
            result = fetcher.fetch_captions("dQw4w9WgXcQ")
            cache.store(result)

    Example with CaptionFetcher:
        fetcher = CaptionFetcher(config)
        cache = CaptionCache(config.download.caption_first)

        def fetch_with_cache(video_id: str, language: str = "en"):
            cached = cache.get_caption(video_id, language)
            if cached:
                return cached.to_caption_result()
            result = fetcher.fetch_captions(video_id, language=language)
            cache.store(result)
            return result
    """

    def __init__(self, config: Optional['CaptionFirstConfig'] = None):
        """Initialize the caption cache.

        Args:
            config: CaptionFirstConfig with cache settings. If None, uses defaults.
        """
        # Get config values with defaults
        if config:
            cache_dir = getattr(config, 'cache_dir', '~/.matcher_caption_cache')
            max_age_days = getattr(config, 'max_cache_age_days', 30)
            self.enabled = getattr(config, 'cache_captions', True)
        else:
            cache_dir = '~/.matcher_caption_cache'
            max_age_days = 30
            self.enabled = True

        # Expand ~ in cache_dir
        cache_dir = Path(os.path.expanduser(cache_dir))

        # Convert max_age_days to TTL seconds (0 = no expiration)
        ttl_seconds = max_age_days * 24 * 3600 if max_age_days > 0 else 0

        # Initialize BaseCache
        super().__init__(
            cache_dir=cache_dir,
            index_name="caption_cache_index.json",
            ttl_seconds=ttl_seconds,
            auto_save=True
        )

        self.max_age_days = max_age_days

        logger.debug(f"CaptionCache initialized: dir={cache_dir}, "
                    f"ttl={max_age_days} days, enabled={self.enabled}")

    def _make_cache_key(self, video_id: str, language: str) -> str:
        """Create cache key from video_id and language.

        Args:
            video_id: YouTube video ID (11 characters).
            language: ISO 639-1 language code (e.g., 'en').

        Returns:
            Cache key in format 'video_id_language' (e.g., 'dQw4w9WgXcQ_en').
        """
        return f"{video_id}_{language}"

    def _serialize_entry(self, entry: CacheEntry) -> Dict[str, Any]:
        """Serialize CachedCaption to dict."""
        return {
            'data': entry.data,  # CachedCaption.to_dict()
            'cached_at': entry.cached_at,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry:
        """Deserialize dict to CachedCaption entry."""
        return CacheEntry(
            data=data.get('data', {}),
            cached_at=data.get('cached_at', 0.0),
            key='',
            metadata=data.get('metadata', {})
        )

    def get_caption(self, video_id: str, language: str) -> Optional[CachedCaption]:
        """Get cached caption for a video and language.

        Args:
            video_id: YouTube video ID.
            language: ISO 639-1 language code.

        Returns:
            CachedCaption if found and valid, None otherwise.
        """
        if not self.enabled:
            return None

        key = self._make_cache_key(video_id, language)
        entry = self.get(key)

        if entry is None:
            logger.debug(f"Caption cache miss: {key}")
            return None

        try:
            cached = CachedCaption.from_dict(entry.data)
            logger.debug(f"Caption cache hit: {key} "
                        f"({len(cached.segments)} segments, "
                        f"auto={cached.is_auto_generated})")
            return cached
        except Exception as e:
            logger.warning(f"Failed to deserialize cached caption {key}: {e}")
            self.delete(key)
            return None

    def store(self, result: CaptionResult) -> bool:
        """Store a CaptionResult in the cache.

        Args:
            result: CaptionResult to cache.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.enabled:
            return False

        if not result.segments:
            logger.debug(f"Not caching empty caption result for {result.video_id}")
            return False

        key = self._make_cache_key(result.video_id, result.language)

        cached = CachedCaption(
            video_id=result.video_id,
            language=result.language,
            segments=[seg.to_dict() for seg in result.segments],
            is_auto_generated=result.is_auto_generated,
            format_source=result.format_source,
            fetch_timestamp=time.time(),
            duration=result.duration,
        )

        self.set(key, cached.to_dict())

        logger.info(f"Cached captions: {key} "
                   f"({len(result.segments)} segments, "
                   f"duration={result.duration:.1f}s, "
                   f"auto={result.is_auto_generated})")
        return True

    def get_or_fetch(
        self,
        fetcher: 'CaptionFetcher',
        video_id: str,
        language: str = "en",
        prefer_manual: bool = True
    ) -> CaptionResult:
        """Get from cache or fetch and cache.

        Convenience method that combines cache lookup and fetching.

        Args:
            fetcher: CaptionFetcher instance to use for fetching.
            video_id: YouTube video ID.
            language: Preferred language code.
            prefer_manual: Prefer manual captions over auto-generated.

        Returns:
            CaptionResult from cache or freshly fetched.

        Raises:
            CaptionUnavailableError: If no captions exist.
            CaptionFetchError: If fetch fails due to network/temporary error.
        """
        # Check cache first
        cached = self.get_caption(video_id, language)
        if cached:
            return cached.to_caption_result()

        # Fetch and cache
        result = fetcher.fetch_captions(video_id, language=language, prefer_manual=prefer_manual)
        self.store(result)
        return result

    def invalidate(self, video_id: str, language: Optional[str] = None) -> int:
        """Invalidate cached captions for a video.

        Args:
            video_id: YouTube video ID.
            language: If provided, only invalidate for this language.
                     If None, invalidate all languages for this video.

        Returns:
            Number of cache entries invalidated.
        """
        if language:
            key = self._make_cache_key(video_id, language)
            if self.delete(key):
                logger.debug(f"Invalidated caption cache: {key}")
                return 1
            return 0

        # Invalidate all languages for this video
        invalidated = 0
        for key in list(self.index.keys()):
            if key.startswith(f"{video_id}_"):
                self.delete(key)
                invalidated += 1

        if invalidated:
            logger.debug(f"Invalidated {invalidated} caption cache entries for {video_id}")

        return invalidated

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache metrics.
        """
        base_stats = super().get_stats()

        # Add caption-specific stats
        total_segments = 0
        auto_generated_count = 0
        manual_count = 0

        for entry in self.get_all().values():
            try:
                cached = CachedCaption.from_dict(entry.data)
                total_segments += len(cached.segments)
                if cached.is_auto_generated:
                    auto_generated_count += 1
                else:
                    manual_count += 1
            except Exception:
                pass

        base_stats.update({
            'total_segments': total_segments,
            'auto_generated_entries': auto_generated_count,
            'manual_entries': manual_count,
            'max_age_days': self.max_age_days,
            'enabled': self.enabled,
        })

        return base_stats
