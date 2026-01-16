"""
Utility functions for video downloading and caption handling.

Migrated from downloader.py lines 122-166, 2431-2456.
Extended with caption utilities for caption-first mode.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# Characters that cause issues in DaVinci Resolve
# Note: Spaces are OK! Only these specific chars cause crashes.
_UNSAFE_FILENAME_CHARS = re.compile(r'[%&$#]')


def sanitize_filename_for_nle(filepath: Path) -> Path:
    """
    Sanitize filename to remove characters that cause issues in DaVinci Resolve.

    Migrated from downloader.py lines 126-165.

    Based on testing: Spaces are OK, but % & $ # cause crashes.

    Args:
        filepath: Path to file

    Returns:
        New path (renamed if necessary), or original path if already safe
    """
    filename = filepath.name
    stem = filepath.stem
    suffix = filepath.suffix

    # Check if sanitization is needed
    if not _UNSAFE_FILENAME_CHARS.search(stem):
        return filepath

    # Create safe filename
    safe_stem = _UNSAFE_FILENAME_CHARS.sub('_', stem)
    safe_stem = re.sub(r'_+', '_', safe_stem)  # Remove multiple underscores

    new_path = filepath.parent / f"{safe_stem}{suffix}"

    # Handle collision
    counter = 1
    while new_path.exists() and new_path != filepath:
        new_path = filepath.parent / f"{safe_stem}_{counter}{suffix}"
        counter += 1

    # Rename the file
    try:
        filepath.rename(new_path)
        logger.info(f"Sanitized filename: {filename} → {new_path.name}")
        return new_path
    except Exception as e:
        logger.warning(f"Could not sanitize filename {filename}: {e}")
        return filepath


def format_time(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS for yt-dlp.

    Migrated from downloader.py lines 2431-2436.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_ytdlp_base_args() -> List[str]:
    """
    Get base yt-dlp arguments required for all downloads.

    Returns arguments needed for YouTube's JavaScript challenge solving.
    Without these, downloads fail with "Signature solving failed" errors.

    Returns:
        List of base arguments for yt-dlp
    """
    return ['--js-runtimes', 'node']


def get_cookies_args(config: 'Config') -> List[str]:
    """
    Get yt-dlp cookie arguments (browser or file).

    Migrated from downloader.py lines 2438-2456.

    Prefers cookies_from_browser, falls back to cookies_path.
    Returns list of arguments to extend yt-dlp command.

    Args:
        config: Config object with download.cookies_from_browser or download.cookies_path

    Returns:
        List of cookie arguments for yt-dlp (empty if no cookies configured)
    """
    download_config = config.download

    # Prefer browser cookies
    cookies_browser = getattr(download_config, 'cookies_from_browser', '')
    if cookies_browser:
        logger.debug(f"Using cookies from browser: {cookies_browser}")
        return ['--cookies-from-browser', cookies_browser]

    # Fall back to cookies file
    cookies_path = getattr(download_config, 'cookies_path', '')
    if cookies_path and Path(cookies_path).exists():
        logger.debug(f"Using cookies file: {cookies_path}")
        return ['--cookies', cookies_path]

    return []


# YouTube video ID regex pattern (11 characters, alphanumeric + _ -)
_VIDEO_ID_PATTERN = re.compile(r'[a-zA-Z0-9_-]{11}')

# YouTube URL patterns for video ID extraction
_YOUTUBE_URL_PATTERNS = [
    re.compile(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'),
    re.compile(r'youtube\.com/embed/([a-zA-Z0-9_-]{11})'),
    re.compile(r'youtube\.com/v/([a-zA-Z0-9_-]{11})'),
]


def extract_video_id(source: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL, filename, or raw ID.

    Handles:
    - Full YouTube URLs (youtube.com/watch?v=..., youtu.be/...)
    - Embed URLs (youtube.com/embed/...)
    - Filenames containing video ID (video_dQw4w9WgXcQ.mp4)
    - Raw 11-character video IDs

    Args:
        source: URL, filename, or video ID string

    Returns:
        11-character video ID, or None if not found

    Examples:
        >>> extract_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("video_dQw4w9WgXcQ.en.srt")
        'dQw4w9WgXcQ'
        >>> extract_video_id("dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if not source:
        return None

    # Try URL patterns first
    for pattern in _YOUTUBE_URL_PATTERNS:
        match = pattern.search(source)
        if match:
            return match.group(1)

    # Try to find video ID in filename or raw string
    # Look for 11-char alphanumeric sequence
    match = _VIDEO_ID_PATTERN.search(source)
    if match:
        return match.group(0)

    return None


def detect_caption_format(filepath: Path) -> Optional[str]:
    """
    Detect caption file format from extension or content.

    Args:
        filepath: Path to caption file

    Returns:
        Format string ('srt', 'vtt', 'ass', 'json3'), or None if unknown
    """
    suffix = filepath.suffix.lower()

    format_map = {
        '.srt': 'srt',
        '.vtt': 'vtt',
        '.ass': 'ass',
        '.ssa': 'ass',
        '.json3': 'json3',
        '.json': 'json3',
        '.ttml': 'ttml',
        '.dfxp': 'ttml',
    }

    if suffix in format_map:
        return format_map[suffix]

    # Try to detect from content if file exists
    if filepath.exists():
        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')[:200]
            if content.startswith('WEBVTT'):
                return 'vtt'
            if re.match(r'^\d+\s*\n\d{2}:\d{2}:', content):
                return 'srt'
            if '[Script Info]' in content:
                return 'ass'
        except Exception:
            pass

    return None


def get_caption_language(filepath: Path, video_id: Optional[str] = None) -> str:
    """
    Extract language code from caption filename.

    Expected formats:
    - {video_id}.{lang}.srt (e.g., dQw4w9WgXcQ.en.srt)
    - {video_id}.{lang}-auto.srt (e.g., dQw4w9WgXcQ.en-auto.srt)

    Args:
        filepath: Path to caption file
        video_id: Optional video ID to strip from filename

    Returns:
        Language code (e.g., 'en', 'es', 'en-US'), defaults to 'en' if not found
    """
    stem = filepath.stem

    # Remove video ID prefix if provided
    if video_id and stem.startswith(video_id):
        stem = stem[len(video_id):]
        if stem.startswith('.'):
            stem = stem[1:]

    # Remove format suffix
    stem = stem.replace('-auto', '').replace('.auto', '')

    # Extract language code
    parts = stem.split('.')
    for part in parts:
        # Language codes are typically 2-5 chars (en, en-US, zh-Hans)
        if 2 <= len(part) <= 10 and re.match(r'^[a-zA-Z]{2}(-[a-zA-Z]+)?$', part):
            return part

    return 'en'


def is_auto_generated_caption(filepath: Path) -> bool:
    """
    Check if caption file is auto-generated (vs manual).

    Auto-generated captions typically have:
    - '-auto' or '.auto' in filename
    - 'auto' in the language code (e.g., en-auto)

    Args:
        filepath: Path to caption file

    Returns:
        True if auto-generated, False if manual
    """
    name_lower = filepath.name.lower()
    return '-auto' in name_lower or '.auto' in name_lower


def caption_file_priority(filepath: Path) -> Tuple[int, str]:
    """
    Get sort priority for caption files (lower = better).

    Priority order:
    1. Manual captions (priority 0)
    2. Auto-generated captions (priority 1)
    3. Preferred languages (en, en-US, en-GB) get sub-priority 0
    4. Other languages get sub-priority 1

    Args:
        filepath: Path to caption file

    Returns:
        Tuple of (priority, language) for sorting
    """
    is_auto = is_auto_generated_caption(filepath)
    lang = get_caption_language(filepath)

    priority = 1 if is_auto else 0
    lang_priority = 0 if lang.lower().startswith('en') else 1

    return (priority, lang_priority, lang)
