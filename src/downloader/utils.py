"""
Utility functions for video downloading.

Migrated from downloader.py lines 122-166, 2431-2456.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
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


def calculate_file_hash(
    file_path: Path,
    algorithm: str = 'sha256',
    chunk_size: int = 8192
) -> Optional[str]:
    """
    Calculate hash of a file using specified algorithm.

    US-93-002: Download segment hash verification

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (default: sha256)
        chunk_size: Size of chunks to read (default: 8KB)

    Returns:
        Hex digest of the file hash, or None if file doesn't exist
    """
    if not file_path.exists():
        return None

    try:
        hasher = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.warning(f"Failed to calculate {algorithm} hash for {file_path}: {e}")
        return None


def verify_download_hash(
    file_path: Path,
    expected_hash: Optional[str] = None,
    algorithm: str = 'sha256'
) -> Tuple[bool, Optional[str]]:
    """
    Verify file integrity after download using hash verification.

    US-93-002: Download segment hash verification

    Args:
        file_path: Path to the downloaded file
        expected_hash: Expected hash value (if provided)
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Tuple of (verification_passed, actual_hash)
        - If expected_hash is None/missing: returns (True, actual_hash) - no verification needed
        - If expected_hash provided but file doesn't exist: returns (False, None)
        - If verification succeeds: returns (True, actual_hash)
        - If verification fails: returns (False, actual_hash)
    """
    # Handle missing expected hash gracefully - skip verification
    if not expected_hash:
        logger.debug(f"Hash verification skipped for {file_path.name}: no expected hash provided")
        actual_hash = calculate_file_hash(file_path, algorithm)
        return True, actual_hash

    # File doesn't exist - verification fails
    if not file_path.exists():
        logger.warning(f"Hash verification failed for {file_path.name}: file not found")
        return False, None

    # Calculate actual hash
    actual_hash = calculate_file_hash(file_path, algorithm)
    if actual_hash is None:
        logger.warning(f"Hash calculation failed for {file_path.name}")
        return False, None

    # Compare hashes
    passed = actual_hash.lower() == expected_hash.lower()

    if passed:
        logger.info(f"Hash verification passed for {file_path.name}: {actual_hash[:16]}...")
    else:
        logger.warning(
            f"Hash verification FAILED for {file_path.name}: "
            f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        )

    return passed, actual_hash
