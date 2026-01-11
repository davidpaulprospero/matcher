"""
Utility functions for OTIO timeline generation.

Pure functions extracted from otio_builder.py for easier testing and reuse.
Includes path handling, URL formatting, type conversion, and media utilities.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import opentimelineio as otio

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy types to Python native types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# Path and URL Utilities
# ============================================================

def _to_windows_path(path: str) -> str:
    """
    Convert path to Windows format with backslashes.

    DaVinci Resolve requires Windows-style paths: E:\\folder\\file.mp4
    Forward slashes cause import issues.
    """
    abs_path = str(Path(path).resolve())
    # Ensure backslashes (Windows format)
    return abs_path.replace('/', '\\')


def format_path_url(file_path: str) -> str:
    """Format file path for DaVinci Resolve XML - use standard path with forward slashes."""
    path = str(Path(file_path).resolve()).replace('\\', '/')
    # Return plain path - DaVinci prefers standard paths over file:// URLs
    return path


def sanitize_path_for_url(path: str) -> str:
    r"""
    Sanitize a file path for use as a URL in OTIO.

    Handles:
    - Windows extended-length paths (\\?\C:\...)
    - Backslashes to forward slashes
    - Proper file:// URL format

    Note: Does NOT URL-encode. DaVinci Resolve doesn't want encoded paths.
    Problematic characters (%, &, $, #) should be sanitized at download time.
    """
    # Convert to string if Path object
    path = str(path)

    # Remove Windows extended-length path prefix in various forms
    # Check multiple patterns to be safe
    prefixes_to_remove = [
        '\\\\?\\',    # Standard form: \\?\
        '\\\\.\\',    # Device form: \\.\
        '//?/',       # Forward slash form
        '//.//',      # Device forward slash
        '\\?\\',      # Single backslash form (shouldn't happen but just in case)
    ]

    for prefix in prefixes_to_remove:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    # Also check if it starts with ?\  or ?/ after any conversions
    if path.startswith('?\\') or path.startswith('?/'):
        path = path[2:]

    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')

    # Remove any double slashes (except at start for UNC paths - but we don't want UNC)
    while '//' in path:
        path = path.replace('//', '/')

    # For absolute Windows paths (C:/...), ensure proper format
    # Don't add file:// prefix - let OTIO/NLE handle it

    return path


def encode_path_for_xml_url(path: str) -> str:
    """
    Format a file path for use in XML pathurl elements.

    DaVinci Resolve expects standard Windows paths (E:/folder/file.mp4),
    NOT file:// URL format (file://localhost/E:/...) which causes hangs.

    Args:
        path: File path (can be Windows or Unix style)

    Returns:
        Clean file path for DaVinci Resolve (forward slashes)
    """
    # First sanitize the path (remove extended-length prefix, convert slashes)
    path = sanitize_path_for_url(path)

    # Return plain path with forward slashes - no file:// prefix
    # DaVinci Resolve handles this format natively
    return path


def escape_xml(text: str) -> str:
    """Escape special XML characters in text."""
    return (str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;'))


# ============================================================
# Type Conversion Utilities
# ============================================================

def _to_python_type(value):
    """Convert numpy types to native Python types for OTIO compatibility."""
    if value is None:
        return None

    # Check for numpy types
    type_name = type(value).__name__
    module_name = type(value).__module__

    # Handle numpy scalar types
    if module_name == 'numpy' or 'numpy' in str(type(value)):
        # numpy float types
        if 'float' in type_name.lower():
            return float(value)
        # numpy int types
        elif 'int' in type_name.lower():
            return int(value)
        # numpy bool
        elif 'bool' in type_name.lower():
            return bool(value)
        # numpy string types
        elif 'str' in type_name.lower():
            return str(value)
        # numpy array - convert to list
        elif hasattr(value, 'tolist'):
            return value.tolist()

    # Handle lists recursively
    if isinstance(value, list):
        return [_to_python_type(v) for v in value]

    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: _to_python_type(v) for k, v in value.items()}

    return value


def _sanitize_metadata(metadata: dict) -> dict:
    """Convert all metadata values to OTIO-compatible Python types."""
    return {k: _to_python_type(v) for k, v in metadata.items()}


# ============================================================
# Media Utilities
# ============================================================

def _get_media_duration(media_path: str) -> Optional[float]:
    """
    Get actual duration of audio/video file using ffprobe.

    Returns duration in seconds, or None if ffprobe fails.
    Used to determine actual voiceover file length for timeline alignment.
    """
    if not media_path:
        return None

    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(media_path)
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.debug(f"ffprobe duration for {Path(media_path).name}: {duration:.2f}s")
            return duration
        else:
            logger.warning(f"ffprobe returned no duration for {media_path}: rc={result.returncode}, stderr={result.stderr[:100] if result.stderr else 'none'}")
    except FileNotFoundError:
        logger.warning("ffprobe not found in PATH - cannot determine voiceover duration")
    except Exception as e:
        logger.warning(f"Could not get duration for {media_path}: {e}")

    return None


def get_segment_file_offset(file_path: str) -> float:
    """
    Extract the start time offset from a segment filename.

    Audio-first mode downloads video segments with filenames like:
        {video_id}_{start_seconds:04d}.mp4

    For these files, the start_time in the original video is embedded
    in the filename. This function extracts it so the OTIO builder
    can calculate the correct clip offset.

    Args:
        file_path: Path to the video/segment file

    Returns:
        The segment offset in seconds (0.0 for non-segment files)

    Examples:
        "abc123_0045.mp4" -> 45.0 (segment starts at 45s in original)
        "abc123_0120.mp4" -> 120.0 (segment starts at 2min in original)
        "regular_video.mp4" -> 0.0 (not a segment file)
    """
    filename = Path(file_path).stem  # Get filename without extension

    # Pattern: video_id (11 chars) followed by _ and 4-digit start time
    # Example: abc12345678_0045
    match = re.match(r'^[a-zA-Z0-9_-]{11}_(\d{4})$', filename)
    if match:
        return float(match.group(1))

    # Also try pattern with longer IDs (some video IDs vary)
    match = re.match(r'^.+_(\d{4})$', filename)
    if match:
        # Verify this looks like a segment (4-digit suffix)
        return float(match.group(1))

    return 0.0


def is_segment_file(file_path: str) -> bool:
    """Check if a file is a downloaded segment (audio-first mode)."""
    return get_segment_file_offset(file_path) > 0.0 or file_path.endswith('_0000.mp4')


# ============================================================
# Formatting Utilities
# ============================================================

def frames_to_tc(frames: int, fps: float = 30.0) -> str:
    """Convert frame count to timecode string HH:MM:SS:FF"""
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frame = int(frames % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"


def get_confidence_color(confidence: float) -> str:
    """Get clip color name based on confidence tier"""
    if confidence >= 0.8:
        return "GREEN"
    elif confidence >= 0.6:
        return "CYAN"
    elif confidence >= 0.4:
        return "YELLOW"
    elif confidence >= 0.2:
        return "ORANGE"
    else:
        return "RED"


# ============================================================
# OTIO Clip Creation
# ============================================================

def create_clip_with_timewarp(
    name: str,
    source_path: str,
    source_start: float,
    source_duration: float,
    target_duration: float,
    frame_rate: float = 30.0,
    metadata: Optional[Dict] = None,
    media_duration: float = None  # Total duration of the source media file
) -> otio.schema.Clip:
    """
    Create a clip with speed adjustment to match target (voiceover) duration.

    Uses LinearTimeWarp to stretch/compress source footage to fit target duration.
    This ensures clips align with voiceover without manual speed adjustment in NLE.

    Args:
        name: Clip name
        source_path: Path to source video/audio
        source_start: Start time in source (seconds)
        source_duration: Original duration in source (seconds)
        target_duration: Desired duration on timeline (seconds)
        frame_rate: Frame rate
        metadata: Optional metadata dict
        media_duration: Total duration of the source media file (for available_range)

    Returns:
        OTIO Clip with LinearTimeWarp applied to match target_duration
    """
    rate = frame_rate

    # Create absolute Windows path with backslashes for DaVinci Resolve
    abs_path = _to_windows_path(source_path)

    # Make media reference name unique by including parent folder
    # This prevents DaVinci Resolve from confusing clips with same filename in different folders
    folder_name = Path(source_path).parent.name
    filename = Path(source_path).name
    unique_media_name = f"{folder_name}_{filename}"

    # Determine available_range for the media file
    # If we don't know the media duration, estimate from source_start + source_duration
    if media_duration is None:
        # Estimate: assume media is at least as long as what we're using
        estimated_duration = source_start + source_duration + 10  # Add buffer
        media_duration = estimated_duration

    available_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, rate),
        duration=otio.opentime.RationalTime(round(media_duration * rate), rate)
    )

    # Create media reference with proper format for DaVinci Resolve
    # Note: name must be set as attribute, not constructor param
    media_ref = otio.schema.ExternalReference(
        target_url=abs_path,
        available_range=available_range
    )
    media_ref.name = unique_media_name  # Unique name includes folder

    # IMPORTANT: Use round() to avoid floating-point precision drift
    # This prevents timing deviation over many clips
    start_frames = round(source_start * rate)

    # Use TARGET duration for source_range.duration to ensure correct timeline duration
    # DaVinci Resolve may not properly interpret LinearTimeWarp, so we set the
    # timeline duration directly. The LinearTimeWarp effect and metadata indicate
    # the speed adjustment needed to fit source_duration into target_duration.
    target_duration_frames = round(target_duration * rate)

    # Ensure we have at least 1 frame
    if target_duration_frames < 1:
        target_duration_frames = 1

    source_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(start_frames, rate),
        duration=otio.opentime.RationalTime(target_duration_frames, rate)
    )

    # Create clip
    clip = otio.schema.Clip(
        name=name,
        media_reference=media_ref,
        source_range=source_range
    )

    # Apply LinearTimeWarp to match target duration
    # time_scalar = source_duration / target_duration
    # - time_scalar < 1: slow down (stretch footage to fill longer duration)
    # - time_scalar > 1: speed up (compress footage to fit shorter duration)
    # - time_scalar = 1: no change (source and target match)
    if target_duration > 0 and source_duration > 0:
        time_scalar = source_duration / target_duration

        # Only apply time warp if there's a meaningful speed change (>1% difference)
        if abs(time_scalar - 1.0) > 0.01:
            time_warp = otio.schema.LinearTimeWarp(time_scalar=time_scalar)
            clip.effects.append(time_warp)

    # Add Resolve_OTIO metadata (required for DaVinci import)
    clip.metadata['Resolve_OTIO'] = {}

    # Calculate and store speed info in metadata for reference
    if metadata is None:
        metadata = {}

    if target_duration > 0 and source_duration > 0:
        time_scalar = source_duration / target_duration
        metadata['time_scalar'] = time_scalar
        metadata['speed_percent'] = time_scalar * 100
        metadata['target_duration'] = target_duration
        metadata['source_duration'] = source_duration
        metadata['suggested_speed'] = f"{time_scalar * 100:.0f}%"

    # Add metadata (convert numpy types to Python native types)
    metadata = _sanitize_metadata(metadata)
    for key, value in metadata.items():
        clip.metadata[key] = value

    return clip
