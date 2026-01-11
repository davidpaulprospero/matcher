"""
Utility functions for video downloading.

Migrated from downloader.py lines 122-166, 2431-2456.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

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
