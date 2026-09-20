"""
Utility functions for video downloading.

Migrated from downloader.py lines 122-166, 2431-2456.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# On Windows, prevent subprocess from spawning visible console windows
# On Linux/macOS, start a new process group so we can kill the entire tree (yt-dlp + ffmpeg)
# via os.killpg — fixes the orchestrator "kill" not propagating to ffmpeg children,
# which caused the BEQ9C9Z2lx0/id-YPfTveTM hung downloads.
SUBPROCESS_FLAGS: dict = (
    {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32'
    else {'start_new_session': True}
)


# Cached yt-dlp executable resolution. The repo embeds Python+yt-dlp in
# tools/python/Scripts (see activate-tools.ps1) so pipeline runs don't need a
# system-wide yt-dlp install. Without this resolver, subprocess.run with a
# bare 'yt-dlp' fails with [WinError 2] when tools/python/Scripts isn't on
# PATH (e.g. when launched outside an activated shell).
_YTDLP_EXECUTABLE: Optional[str] = None


def get_ytdlp_executable() -> str:
    """Resolve the yt-dlp executable path with PATH + repo-local fallback.

    Search order:
      1. ``shutil.which('yt-dlp')`` — normal PATH lookup
      2. ``<cwd>/tools/python/Scripts/yt-dlp{,.exe}`` and ``<cwd>/tools/yt-dlp{,.exe}``
      3. ``<repo>/tools/python/Scripts/yt-dlp{,.exe}`` and ``<repo>/tools/yt-dlp{,.exe}``
         where ``<repo>`` is the directory three levels above this file
         (``src/downloader/utils.py`` → repo root).

    The result is cached so repeated subprocess invocations don't re-scan disk.

    Returns:
        Absolute path to the yt-dlp executable when found, or the bare string
        ``'yt-dlp'`` as a last resort (subprocess will then rely on PATH and
        likely fail with ``[WinError 2]`` if PATH is unset).
    """
    global _YTDLP_EXECUTABLE
    if _YTDLP_EXECUTABLE is not None:
        return _YTDLP_EXECUTABLE

    found = shutil.which('yt-dlp')
    if found:
        _YTDLP_EXECUTABLE = found
        logger.debug("yt-dlp resolved via PATH: %s", found)
        return found

    suffixes: tuple = ('yt-dlp.exe', 'yt-dlp') if sys.platform == 'win32' else ('yt-dlp',)
    subdirs: tuple = ('tools/python/Scripts', 'tools')

    roots: list = [Path.cwd()]
    try:
        roots.append(Path(__file__).resolve().parent.parent.parent)
    except Exception:  # noqa: BLE001 — defensive; fall back to cwd-only
        pass

    for root in roots:
        for sub in subdirs:
            for suf in suffixes:
                cand = root / sub / suf
                if cand.is_file():
                    _YTDLP_EXECUTABLE = str(cand)
                    logger.info("yt-dlp resolved from repo-local: %s", _YTDLP_EXECUTABLE)
                    return _YTDLP_EXECUTABLE

    logger.warning(
        "yt-dlp executable not found in PATH or repo tools/. "
        "Subprocess calls will likely fail with [WinError 2]; "
        "either install yt-dlp system-wide or run via activate-tools.ps1."
    )
    _YTDLP_EXECUTABLE = 'yt-dlp'
    return _YTDLP_EXECUTABLE

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


def estimate_video_size_mb(
    duration_seconds: int,
    quality: str = "1080",
    size_estimates: Optional[Dict[str, float]] = None
) -> float:
    """
    Estimate video file size in MB based on duration and quality.

    US-93-005: Adaptive download timeout

    Uses bitrate assumptions to estimate file size:
    - 4K (2160p): ~25 MB/min
    - 2K (1440p): ~15 MB/min
    - 1080p: ~5 MB/min
    - 720p: ~2.5 MB/min
    - 480p: ~1.5 MB/min

    Args:
        duration_seconds: Video duration in seconds
        quality: Video resolution (e.g., "1080", "720", "2160")
        size_estimates: Optional dict of MB/min by resolution

    Returns:
        Estimated file size in MB
    """
    if duration_seconds <= 0:
        return 0.0

    # Default size estimates (MB per minute)
    if size_estimates is None:
        size_estimates = {
            '2160': 25.0,
            '1440': 15.0,
            '1080': 5.0,
            '720': 2.5,
            '480': 1.5,
            'default': 3.0
        }

    # Extract resolution number from quality string
    resolution = quality.upper()
    for res in ['2160', '1440', '1080', '720', '480']:
        if res in resolution:
            mb_per_min = size_estimates.get(res, size_estimates['default'])
            break
    else:
        mb_per_min = size_estimates['default']

    # Calculate size: MB/min * minutes
    duration_minutes = duration_seconds / 60.0
    return mb_per_min * duration_minutes


def calculate_adaptive_timeout(
    estimated_size_mb: float,
    base_timeout: int,
    multiplier: float,
    max_timeout: Optional[int] = None
) -> int:
    """
    Calculate adaptive timeout based on estimated file size.

    US-93-005: Adaptive download timeout

    Formula: base_timeout + (size_mb * multiplier)

    Args:
        estimated_size_mb: Estimated file size in MB
        base_timeout: Base timeout in seconds (minimum)
        multiplier: Seconds to add per MB of file size
        max_timeout: Optional maximum timeout cap

    Returns:
        Calculated timeout in seconds

    Example:
        >>> calculate_adaptive_timeout(100, 30, 0.5)
        80  # 30 + (100 * 0.5) = 80
        >>> calculate_adaptive_timeout(500, 30, 0.5, max_timeout=600)
        600  # Capped at max_timeout
    """
    if estimated_size_mb <= 0:
        return base_timeout

    calculated = int(base_timeout + (estimated_size_mb * multiplier))

    if max_timeout is not None and calculated > max_timeout:
        return max_timeout

    return calculated


def resolve_js_runtime(download_config: Optional[Any]) -> Optional[List[str]]:
    """Build ``--js-runtimes deno[:PATH]`` argv for yt-dlp subprocess calls.

    Tries (in order):
    1. ``download_config.js_runtime_path`` if set and the file exists.
    2. ``shutil.which('deno')`` / ``node`` / ``bun`` PATH lookup.

    Returns the argv fragment as a list (e.g. ``['--js-runtimes', 'deno']``)
    or ``None`` when no runtime can be resolved — caller can skip the flag.
    """
    import shutil as _shutil
    explicit = getattr(download_config, 'js_runtime_path', '') if download_config else ''
    if explicit and Path(explicit).is_file():
        return ['--js-runtimes', f'deno:{explicit}']
    for runtime in ('deno', 'node', 'bun'):
        found = _shutil.which(runtime)
        if found:
            return ['--js-runtimes', runtime]
    return None


def append_external_tool_args(
    cmd: List[str],
    download_config: Optional[Any],
) -> List[str]:
    """Append ``--ffmpeg-location`` and ``--js-runtimes`` flags to a yt-dlp CLI command.

    Used by every CLI subprocess call site so ffmpeg merging and the EJS solver
    work consistently without depending on a shell PATH.
    """
    ffmpeg_loc = getattr(download_config, 'ffmpeg_location', '') if download_config else ''
    if ffmpeg_loc:
        cmd.extend(['--ffmpeg-location', ffmpeg_loc])
    js_args = resolve_js_runtime(download_config)
    if js_args:
        cmd.extend(js_args)
    return cmd


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


def check_partial_file(
    file_path: Path,
    partial_extension: str = ".part"
) -> bool:
    """
    Check if there's a partial download file for the given target file.

    US-93-008: Implement download resume from partial

    Checks for:
    - File with .part extension (yt-dlp default)
    - File with .partial extension

    Args:
        file_path: The target file path (without partial extension)
        partial_extension: Extension used for partial downloads

    Returns:
        True if a partial file exists, False otherwise
    """
    # Check for .part file
    partial_file = file_path.with_suffix(file_path.suffix + partial_extension)
    if partial_file.exists():
        return True

    # Also check for .partial extension variant
    partial_file_alt = file_path.with_suffix(file_path.suffix + ".partial")
    if partial_file_alt.exists():
        return True

    return False


def get_partial_file_info(
    file_path: Path,
    partial_extension: str = ".part"
) -> Optional[Dict[str, Any]]:
    """
    Get information about a partial download file.

    US-93-008: Implement download resume from partial

    Args:
        file_path: The target file path (without partial extension)
        partial_extension: Extension used for partial downloads

    Returns:
        Dict with keys: path, size_bytes, exists, or None if no partial file
    """
    # Check for .part file
    partial_file = file_path.with_suffix(file_path.suffix + partial_extension)
    if partial_file.exists():
        try:
            size = partial_file.stat().st_size
            return {
                "path": partial_file,
                "size_bytes": size,
                "exists": True,
            }
        except OSError:
            pass

    # Check for .partial extension variant
    partial_file_alt = file_path.with_suffix(file_path.suffix + ".partial")
    if partial_file_alt.exists():
        try:
            size = partial_file_alt.stat().st_size
            return {
                "path": partial_file_alt,
                "size_bytes": size,
                "exists": True,
            }
        except OSError:
            pass

    return None


def cleanup_partial_files(
    file_path: Path,
    partial_extension: str = ".part",
    min_size_bytes: int = 0,
    logger_instance: Optional[logging.Logger] = None
) -> int:
    """
    Clean up partial download files.

    US-93-008: Implement download resume from partial

    Removes partial files that are smaller than min_size_bytes (if specified).

    Args:
        file_path: The target file path (without partial extension)
        partial_extension: Extension used for partial downloads
        min_size_bytes: Only clean up files larger than this (0 = clean all)
        logger_instance: Optional logger for logging cleanup actions

    Returns:
        Number of files cleaned up
    """
    cleaned = 0

    # Log function
    def log(msg: str):
        if logger_instance:
            logger_instance.info(msg)
        else:
            logging.info(msg)

    # Check for .part file
    partial_file = file_path.with_suffix(file_path.suffix + partial_extension)
    if partial_file.exists():
        try:
            size = partial_file.stat().st_size
            if min_size_bytes == 0 or size >= min_size_bytes:
                partial_file.unlink()
                log(f"Cleaned up partial file: {partial_file.name} ({size} bytes)")
                cleaned += 1
            else:
                log(f"Skipped small partial file: {partial_file.name} ({size} bytes < {min_size_bytes})")
        except OSError as e:
            log(f"Failed to clean up partial file {partial_file.name}: {e}")

    # Check for .partial extension variant
    partial_file_alt = file_path.with_suffix(file_path.suffix + ".partial")
    if partial_file_alt.exists():
        try:
            size = partial_file_alt.stat().st_size
            if min_size_bytes == 0 or size >= min_size_bytes:
                partial_file_alt.unlink()
                log(f"Cleaned up partial file: {partial_file_alt.name} ({size} bytes)")
                cleaned += 1
        except OSError as e:
            log(f"Failed to clean up partial file {partial_file_alt.name}: {e}")

    return cleaned


def get_resume_offset(
    file_path: Path,
    partial_extension: str = ".part",
    min_partial_size: int = 1024
) -> Optional[int]:
    """
    Get the byte offset to resume from for a partial download.

    US-93-008: Implement download resume from partial

    Args:
        file_path: The target file path (without partial extension)
        partial_extension: Extension used for partial downloads
        min_partial_size: Minimum size in bytes to consider for resume

    Returns:
        Byte offset to resume from, or None if no valid partial file
    """
    partial_info = get_partial_file_info(file_path, partial_extension)
    if partial_info and partial_info["size_bytes"] >= min_partial_size:
        return partial_info["size_bytes"]
    return None
