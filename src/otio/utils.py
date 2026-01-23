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
from typing import Dict, List, Optional

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
    Convert path to absolute path with forward slashes.

    DaVinci Resolve imports better with forward slashes: E:/folder/file.mp4
    Backslashes can cause hangs during OTIO import.
    """
    abs_path = str(Path(path).resolve())
    # Use forward slashes (works better with DaVinci)
    return abs_path.replace('\\', '/')


def format_path_url(file_path: str) -> str:
    """Format file path for DaVinci Resolve XML - plain Windows path with forward slashes."""
    path = str(Path(file_path).resolve()).replace('\\', '/')
    # DaVinci Resolve on Windows expects plain paths, not file:// URLs
    # E:\path\file.mp4 -> E:/path/file.mp4
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

    DaVinci Resolve on Windows requires file:/// prefix for absolute paths
    (e.g., file:///E:/folder/file.mp4). Without this prefix, DaVinci reports
    "files not found" when importing XML.

    Args:
        path: File path (can be Windows or Unix style)

    Returns:
        File URL with file:/// prefix for DaVinci Resolve compatibility
    """
    # First sanitize the path (remove extended-length prefix, convert slashes)
    path = sanitize_path_for_url(path)

    # Add file:/// prefix for DaVinci Resolve compatibility on Windows
    # Absolute paths need this prefix for DaVinci to locate files
    if path and len(path) > 1 and path[1] == ':':
        # Windows absolute path (e.g., E:/folder/file.mp4)
        return f'file:///{path}'
    elif path and path.startswith('/'):
        # Unix absolute path
        return f'file://{path}'

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

    If an SRT file is provided, automatically looks for an accompanying audio file
    (MP3, WAV, M4A, MP4, AAC) with the same base name.
    """
    if not media_path:
        return None

    path = Path(media_path)

    # If SRT file, look for accompanying audio file
    if path.suffix.lower() == '.srt':
        audio_extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.aac', '.flac', '.ogg']
        for ext in audio_extensions:
            audio_path = path.with_suffix(ext)
            if audio_path.exists():
                logger.info(f"SRT file provided, using accompanying audio: {audio_path.name}")
                media_path = str(audio_path)
                break
        else:
            # No accompanying audio found - return None so fallback is used
            logger.warning(f"SRT file provided but no accompanying audio found ({path.stem}.[mp3|wav|m4a|...])")
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
# OTIO Track Optimization
# ============================================================

def optimize_track_gaps(track: otio.schema.Track) -> otio.schema.Track:
    """
    Optimize gaps in an OTIO track for better DaVinci Resolve compatibility.

    Performs two optimizations:
    1. Merges consecutive gaps into single gaps
    2. Removes trailing gaps (they serve no purpose)

    This prevents potential performance issues with DaVinci Resolve
    when importing tracks with many small gaps.

    Args:
        track: OTIO Track to optimize

    Returns:
        The same track with optimized gap structure
    """
    try:
        # Use list() to safely get items - avoids OTIO internal iteration issues
        items = list(track)
        if not items:
            return track
    except (AttributeError, RuntimeError) as e:
        # Track may be in inconsistent state - skip optimization
        import logging
        logging.getLogger(__name__).warning(f"Skipping track optimization due to error: {e}")
        return track

    frame_rate = 30.0  # Default, will be detected from first item

    # Detect frame rate from first item with a source_range
    for item in items:
        if hasattr(item, 'source_range') and item.source_range:
            frame_rate = item.source_range.duration.rate
            break

    # Merge consecutive gaps
    merged_children = []
    current_gap_frames = 0.0

    for item in items:
        is_gap = isinstance(item, otio.schema.Gap)

        if is_gap:
            # Accumulate gap duration
            if item.source_range:
                current_gap_frames += item.source_range.duration.value
        else:
            # Flush accumulated gap as single gap
            if current_gap_frames > 0:
                gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, frame_rate),
                        duration=otio.opentime.RationalTime(current_gap_frames, frame_rate)
                    )
                )
                merged_children.append(gap)
                current_gap_frames = 0
            merged_children.append(item)

    # Don't add trailing gap - they serve no purpose
    # (If current_gap_frames > 0 at this point, it's a trailing gap)

    # Update track children
    # Clear existing children and add merged ones
    try:
        track.clear()  # Use clear() instead of while loop
    except (AttributeError, RuntimeError):
        # Fallback: remove items one by one
        while True:
            try:
                if not list(track):
                    break
                del track[0]
            except (IndexError, RuntimeError):
                break

    for child in merged_children:
        track.append(child)

    return track


def optimize_timeline_gaps(timeline: otio.schema.Timeline) -> otio.schema.Timeline:
    """
    Optimize gaps in all tracks of a timeline.

    Args:
        timeline: OTIO Timeline to optimize

    Returns:
        The same timeline with optimized gap structure in all tracks
    """
    for track in timeline.tracks:
        if isinstance(track, otio.schema.Track):
            optimize_track_gaps(track)

    return timeline


def count_timeline_items(timeline: otio.schema.Timeline) -> int:
    """
    Count total items (clips + gaps) in a timeline.

    DaVinci Resolve processes each item when importing OTIO.
    Large item counts (10k+) cause crashes/hangs.

    Args:
        timeline: OTIO Timeline to count

    Returns:
        Total number of clips and gaps across all tracks
    """
    total = 0
    for track in timeline.tracks:
        if isinstance(track, otio.schema.Track):
            total += len(list(track))  # Use list() to safely iterate
    return total


# ============================================================
# Media Path Normalization
# ============================================================

def build_canonical_media_map(media_paths: List[str]) -> Dict[str, str]:
    """
    Build a map of media paths to their canonical (deduplicated) versions.

    DaVinci Resolve hangs when importing OTIO files that reference the same
    video file from multiple different paths (e.g., stock/ and broll/ copies).
    This function identifies duplicate files and maps them to a single canonical path.

    Identification is based on filename + file size (fast, reliable for video files).
    When duplicates exist, preference order is:
    1. stock/ folder (original downloads)
    2. Shorter path (simpler reference)

    Args:
        media_paths: List of all media file paths in the timeline

    Returns:
        Dict mapping each input path to its canonical path.
        Paths without duplicates map to themselves.

    Example:
        Input: ['E:/v/proj/stock/video.mp4', 'E:/v/proj/broll/pexels/video.mp4']
        Output: {
            'E:/v/proj/stock/video.mp4': 'E:/v/proj/stock/video.mp4',
            'E:/v/proj/broll/pexels/video.mp4': 'E:/v/proj/stock/video.mp4'
        }
    """
    import os

    # Build map: (filename, size) -> list of paths
    file_key_to_paths: Dict[tuple, List[str]] = {}

    for path in media_paths:
        if not path or not os.path.exists(path):
            continue

        try:
            filename = os.path.basename(path)
            size = os.path.getsize(path)
            key = (filename, size)

            if key not in file_key_to_paths:
                file_key_to_paths[key] = []
            file_key_to_paths[key].append(path)
        except (OSError, IOError):
            # Skip files we can't access
            continue

    # Build canonical map
    canonical_map: Dict[str, str] = {}

    for key, paths in file_key_to_paths.items():
        if len(paths) == 1:
            # No duplicates - maps to itself
            canonical_map[paths[0]] = paths[0]
        else:
            # Multiple paths for same file - pick canonical
            # Preference: stock/ > shorter path
            canonical = None

            for p in paths:
                if '/stock/' in p:
                    canonical = p
                    break

            if not canonical:
                # No stock/ path - use shortest
                canonical = min(paths, key=len)

            # Map all paths to canonical
            for p in paths:
                canonical_map[p] = canonical

            if len(paths) > 1:
                logger.debug(f"Normalized {len(paths)} duplicate paths to: {canonical}")

    return canonical_map


class MediaPathNormalizer:
    """
    Context manager for normalizing media paths in OTIO generation.

    Collects all media paths during timeline creation, then normalizes
    them to canonical paths to prevent DaVinci Resolve import hangs.

    Usage:
        normalizer = MediaPathNormalizer()

        # During clip creation, register paths:
        normalizer.register(path1)
        normalizer.register(path2)

        # After all paths collected, build map:
        normalizer.build_map()

        # Get canonical path for any registered path:
        canonical = normalizer.get_canonical(path1)
    """

    def __init__(self):
        self._paths: List[str] = []
        self._canonical_map: Optional[Dict[str, str]] = None
        self._duplicates_found = 0

    def register(self, path: str) -> None:
        """Register a media path for normalization."""
        if path:
            self._paths.append(path)

    def build_map(self) -> None:
        """Build the canonical path map from all registered paths."""
        self._canonical_map = build_canonical_media_map(self._paths)

        # Count duplicates for logging
        unique_canonicals = set(self._canonical_map.values())
        self._duplicates_found = len(self._paths) - len(unique_canonicals)

        if self._duplicates_found > 0:
            logger.info(f"Media path normalization: {self._duplicates_found} duplicate paths normalized")

    def get_canonical(self, path: str) -> str:
        """Get the canonical path for a registered path."""
        if not self._canonical_map:
            return path
        return self._canonical_map.get(path, path)

    @property
    def duplicates_found(self) -> int:
        """Number of duplicate paths that were normalized."""
        return self._duplicates_found


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

    # =========================================================================
    # SEGMENT FILE TIMESTAMP ADJUSTMENT
    # =========================================================================
    # Segment files (from audio-first mode) have filenames like: video_id_NNNN.mp4
    # where NNNN is the start second in the original video. The segment's local
    # timeline starts at 0, so we need to adjust source_start to segment-local time.
    #
    # Example: file "NOD5Kt49s4E_0006.mp4" is a segment starting at 6s in the original.
    # If source_start=117s (original video time), we need to convert to segment-local:
    #   adjusted_start = 117 - 6 = 111s
    #
    # This handles cases where segment_lookup is empty (e.g., cross-project file reuse)
    # and the timestamp adjustment wasn't done earlier in the pipeline.
    #
    # Heuristic: Only adjust if source_start looks like it's still in original video
    # coordinates (> 60s). This avoids double-adjusting timestamps that were already
    # converted to segment-local time by resolve_video_segment().
    segment_offset = get_segment_file_offset(source_path)
    if segment_offset > 0 or is_segment_file(source_path):
        # This is a segment file - check if source_start needs adjustment
        # Only adjust if:
        # 1. source_start is >= segment_offset (sanity check)
        # 2. source_start is > 60s (heuristic: likely still in original video coordinates)
        #    OR source_start > segment_offset + 30 (clearly beyond expected segment-local range)
        needs_adjustment = (
            source_start >= segment_offset and
            (source_start > 60 or source_start > segment_offset + 30)
        )

        if needs_adjustment:
            original_source_start = source_start
            source_start = source_start - segment_offset

            # Log adjustment for debugging
            logger.debug(
                f"Adjusted source_start for segment file: {original_source_start:.1f}s -> {source_start:.1f}s "
                f"(offset: {segment_offset}s, file: {Path(source_path).name})"
            )

    # Create absolute path with forward slashes for DaVinci Resolve
    # Use file:/// URL format for better DaVinci compatibility
    abs_path = _to_windows_path(source_path)
    if abs_path and len(abs_path) > 1 and abs_path[1] == ':':
        abs_path = f'file:///{abs_path}'

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

    # OTIO TIMING MODEL:
    # - source_range.duration determines timeline duration (how long clip plays)
    # - LinearTimeWarp.time_scalar affects playback speed of those frames
    #
    # Since source_range.duration is already set to target_duration (line 393),
    # the clip will play for exactly target_duration on the timeline.
    # NO LinearTimeWarp is needed - the trim alone achieves correct timing.
    #
    # Adding LinearTimeWarp would COMPOUND with the trim:
    #   timeline_duration = source_range.duration / time_scalar
    #                     = target_duration / (source_duration / target_duration)
    #                     = target_duration² / source_duration  <-- WRONG!
    #
    # We intentionally do NOT apply LinearTimeWarp here.

    # Add Resolve_OTIO metadata (required for DaVinci import)
    clip.metadata['Resolve_OTIO'] = {}

    # Store timing info in metadata for reference
    # Note: time_scalar is always 1.0 since we use trim approach (no speed change)
    if metadata is None:
        metadata = {}

    # Detect if this is an audio-only clip (from audio track)
    # Audio tracks have 'from_track' metadata set by track builders
    is_audio_clip = metadata.get('from_track') is not None

    # For audio clips, add explicit stream mapping to tell DaVinci to use audio stream
    if is_audio_clip:
        # DaVinci Resolve audio stream metadata
        # This tells DaVinci which audio channels to use from the video file
        clip.metadata['Resolve_OTIO']['Audio Channels'] = {
            'ChannelCount': 2,  # Stereo
            'ChannelFormat': 'Stereo',
            'StreamIndex': 0  # Use first audio stream from video file
        }

    if target_duration > 0 and source_duration > 0:
        metadata['time_scalar'] = 1.0  # Always normal speed (trim approach)
        metadata['speed_percent'] = 100.0
        metadata['target_duration'] = target_duration
        metadata['source_duration'] = source_duration
        metadata['original_duration'] = source_duration  # For reference
        metadata['suggested_speed'] = "100%"

    # Add metadata (convert numpy types to Python native types)
    metadata = _sanitize_metadata(metadata)
    for key, value in metadata.items():
        clip.metadata[key] = value

    return clip
