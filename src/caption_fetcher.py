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
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .config import Config

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


class CaptionFetcher:
    """Fetches YouTube captions using yt-dlp.

    Supports multiple caption formats (VTT, SRT, JSON3) and auto-converts
    them to a common segment format.

    Example:
        fetcher = CaptionFetcher()
        result = fetcher.fetch_captions("dQw4w9WgXcQ")
        print(f"Found {len(result.segments)} segments, auto={result.is_auto_generated}")
    """

    def __init__(self, config: Optional['Config'] = None):
        """Initialize the caption fetcher.

        Args:
            config: Optional config for cookies and other settings.
        """
        self.config = config
        self._timeout = 60  # seconds

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
