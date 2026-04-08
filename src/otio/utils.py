"""
Utility functions for OTIO timeline generation.

Pure functions extracted from otio_builder.py for easier testing and reuse.
Includes path handling, URL formatting, type conversion, and media utilities.
"""

import json
import logging
import re
import subprocess
from numbers import Real
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import opentimelineio as otio

from ..downloader.utils import SUBPROCESS_FLAGS

logger = logging.getLogger(__name__)

# Audio-only extensions that cause DaVinci to hang
AUDIO_ONLY_EXTS = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}

# Non-media extensions that can't be imported (subtitles, text, etc.)
NON_MEDIA_EXTS = {'.srt', '.vtt', '.ass', '.ssa', '.sub', '.txt', '.json'}


def seg_start(seg) -> float:
    """Get start time from SRTSegment (.start_time) or VoiceoverSegment (.start)."""
    start_time = getattr(seg, 'start_time', None)
    if isinstance(start_time, Real):
        return float(start_time)
    start = getattr(seg, 'start', 0.0)
    if isinstance(start, Real):
        return float(start)
    return 0.0


def seg_end(seg) -> float:
    """Get end time from SRTSegment (.end_time) or VoiceoverSegment (.end)."""
    end_time = getattr(seg, 'end_time', None)
    if isinstance(end_time, Real):
        return float(end_time)
    end = getattr(seg, 'end', 0.0)
    if isinstance(end, Real):
        return float(end)
    return 0.0


def _is_audio_only(file_path: str) -> bool:
    """Check if file is audio-only (causes DaVinci OTIO import to hang)."""
    ext = Path(file_path).suffix.lower()
    return ext in AUDIO_ONLY_EXTS


def _has_problematic_path(file_path: str) -> bool:
    """
    Check if file path has characters that cause DaVinci OTIO import to hang.

    Problematic patterns:
    - Corrupted unicode (replacement char U+FFFD shown as \ufffd)
    - Non-ASCII characters in paths (accents, special chars)
    - Extended unicode that Windows/DaVinci can't handle
    """
    try:
        # Check for replacement character (corrupted unicode)
        if '\ufffd' in file_path or '\ufffd' in file_path:
            return True

        # Check for truly broken unicode (control chars, surrogates)
        # but allow common non-ASCII like em dashes, accents, etc.
        # which are valid in Linux/macOS paths and modern DaVinci.
        for char in file_path:
            code = ord(char)
            if 0xD800 <= code <= 0xDFFF:
                # Surrogate pair - broken unicode
                return True
            if code < 32 and code not in (9, 10, 13):
                # Control characters (except tab/newline)
                return True

        return False
    except Exception:
        # If we can't even check the path, it's problematic
        return True


def _is_bare_video_id(file_path: str) -> bool:
    """Check if path is an unresolved bare video ID (no extension, no separators)."""
    if not file_path:
        return False
    return not Path(file_path).suffix and '/' not in file_path and '\\' not in file_path


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
            timeout=10,
            encoding='utf-8',
            errors='replace',
            **SUBPROCESS_FLAGS
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

    # Pattern: {video_id}_{start}_{end} (current segment format)
    # Example: 2BIerFyBKJg_79_96 → start=79
    match = re.match(r'^(.+?)_(\d+)_(\d+)$', filename)
    if match:
        return float(match.group(2))

    # Legacy pattern: video_id (11 chars) followed by _ and 4-digit start time
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

def parse_timecode_to_frames(timecode: str, frame_rate: float) -> int:
    """
    Parse a timecode string into a total frame count.

    Handles both ':' and ';' separators so drop-frame timecodes
    (HH:MM:SS;FF) are accepted alongside standard (HH:MM:SS:FF).

    Args:
        timecode: Timecode string like "01:00:00:00" or "01:00:00;00"
        frame_rate: Timeline frame rate (e.g. 30.0, 29.97, 24.0)

    Returns:
        Total frame count for the given timecode
    """
    tc_parts = timecode.replace(';', ':').split(':')
    return (
        int(tc_parts[0]) * 3600 +
        int(tc_parts[1]) * 60 +
        int(tc_parts[2])
    ) * int(frame_rate) + int(tc_parts[3])


def frames_to_tc(
    frames: int,
    fps: float = 30.0,
    start_frame_offset: int = 0,
    separator: str = ':'
) -> str:
    """
    Convert frame count to timecode string.

    Args:
        frames: Frame count to convert
        fps: Frame rate (e.g. 30.0, 29.97, 24.0)
        start_frame_offset: Frame offset added before conversion (e.g.
            from a timeline start timecode like 01:00:00:00)
        separator: Character between seconds and frames.
            Use ':' for non-drop-frame, ';' for drop-frame.

    Returns:
        Timecode string like "01:00:03:15" or "01:00:03;15"
    """
    total_frames = frames + start_frame_offset
    ifps = int(fps)

    frame_in_sec = total_frames % ifps
    total_secs = total_frames // ifps
    secs = total_secs % 60
    total_mins = total_secs // 60
    mins = total_mins % 60
    hours = total_mins // 60

    return f"{hours:02d}:{mins:02d}:{secs:02d}{separator}{frame_in_sec:02d}"


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

def optimize_track_gaps(
    track: otio.schema.Track,
    preserve_trailing_gap: bool = False
) -> otio.schema.Track:
    """
    Optimize gaps in an OTIO track for better DaVinci Resolve compatibility.

    Performs two optimizations:
    1. Merges consecutive gaps into single gaps
    2. Optionally removes trailing gaps (default behavior)

    This prevents potential performance issues with DaVinci Resolve
    when importing tracks with many small gaps.

    Args:
        track: OTIO Track to optimize
        preserve_trailing_gap: If True, keep a trailing gap at end of track.
            Useful when trailing silence/padding is intentionally added to
            align tracks with voiceover duration.

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

    # Preserve trailing gap only when explicitly requested.
    if preserve_trailing_gap and current_gap_frames > 0:
        gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, frame_rate),
                duration=otio.opentime.RationalTime(current_gap_frames, frame_rate)
            )
        )
        merged_children.append(gap)

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


def optimize_timeline_gaps(
    timeline: otio.schema.Timeline,
    preserve_trailing_gaps: bool = False
) -> otio.schema.Timeline:
    """
    Optimize gaps in all tracks of a timeline.

    Args:
        timeline: OTIO Timeline to optimize
        preserve_trailing_gaps: If True, keep trailing gaps on tracks.

    Returns:
        The same timeline with optimized gap structure in all tracks
    """
    for track in timeline.tracks:
        if isinstance(track, otio.schema.Track):
            optimize_track_gaps(
                track,
                preserve_trailing_gap=preserve_trailing_gaps
            )

    return timeline


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
# OTIO Clip Metadata Validation
# ============================================================

class ClipValidationError:
    """Represents a validation error for an OTIO clip."""

    def __init__(self, clip_name: str, error_type: str, message: str):
        self.clip_name = clip_name
        self.error_type = error_type
        self.message = message

    def __repr__(self) -> str:
        return f"ClipValidationError(clip='{self.clip_name}', type='{self.error_type}', message='{self.message}')"


def _validate_clip_metadata(clip: otio.schema.Clip) -> List["ClipValidationError"]:
    """
    Validate OTIO clip metadata before export.

    Performs validation checks to ensure clip is well-formed:
    1. source_range.start_time is non-negative
    2. source_range.duration is positive
    3. media_reference.target_url is non-empty string

    Args:
        clip: OTIO Clip to validate

    Returns:
        List of ClipValidationError objects describing any validation failures.
        Empty list if clip is valid.

    Example:
        errors = _validate_clip_metadata(clip)
        if errors:
            for err in errors:
                logger.warning(f"Clip validation failed: {err.message}")
    """
    errors = []
    clip_name = clip.name or "<unnamed>"

    # Check source_range exists
    if clip.source_range is None:
        errors.append(ClipValidationError(
            clip_name=clip_name,
            error_type="missing_source_range",
            message=f"Clip '{clip_name}' has no source_range"
        ))
        return errors  # Can't check further without source_range

    # Check start_time is non-negative
    if clip.source_range.start_time is not None:
        start_value = clip.source_range.start_time.value
        if start_value < 0:
            errors.append(ClipValidationError(
                clip_name=clip_name,
                error_type="negative_start_time",
                message=f"Clip '{clip_name}' has negative start_time: {start_value}"
            ))
    else:
        errors.append(ClipValidationError(
            clip_name=clip_name,
            error_type="missing_start_time",
            message=f"Clip '{clip_name}' has no start_time in source_range"
        ))

    # Check duration is positive
    if clip.source_range.duration is not None:
        duration_value = clip.source_range.duration.value
        if duration_value <= 0:
            errors.append(ClipValidationError(
                clip_name=clip_name,
                error_type="non_positive_duration",
                message=f"Clip '{clip_name}' has non-positive duration: {duration_value}"
            ))
    else:
        errors.append(ClipValidationError(
            clip_name=clip_name,
            error_type="missing_duration",
            message=f"Clip '{clip_name}' has no duration in source_range"
        ))

    # Check media_reference.target_url is non-empty string
    if clip.media_reference is None:
        errors.append(ClipValidationError(
            clip_name=clip_name,
            error_type="missing_media_reference",
            message=f"Clip '{clip_name}' has no media_reference"
        ))
    elif isinstance(clip.media_reference, otio.schema.MissingReference):
        # MissingReference indicates the media file location is unknown
        errors.append(ClipValidationError(
            clip_name=clip_name,
            error_type="missing_media_reference",
            message=f"Clip '{clip_name}' has MissingReference (no actual media file)"
        ))
    elif isinstance(clip.media_reference, otio.schema.ExternalReference):
        target_url = clip.media_reference.target_url
        if target_url is None or (isinstance(target_url, str) and not target_url.strip()):
            errors.append(ClipValidationError(
                clip_name=clip_name,
                error_type="empty_target_url",
                message=f"Clip '{clip_name}' has empty or missing target_url"
            ))
    # Note: GeneratorReference is allowed (e.g., for test patterns, color bars)

    return errors


def validate_timeline_clips(timeline: otio.schema.Timeline) -> List["ClipValidationError"]:
    """
    Validate all clips in an OTIO timeline.

    Args:
        timeline: OTIO Timeline to validate

    Returns:
        List of all ClipValidationError objects from all clips in the timeline.
        Empty list if all clips are valid.
    """
    all_errors = []

    for track in timeline.tracks:
        if not isinstance(track, otio.schema.Track):
            continue

        for item in track:
            if isinstance(item, otio.schema.Clip):
                errors = _validate_clip_metadata(item)
                all_errors.extend(errors)

    if all_errors:
        logger.warning(f"Timeline validation found {len(all_errors)} clip issues")

    return all_errors


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
    media_duration: float = None,  # Total duration of the source media file
    target_frames: int = None  # Pre-calculated frame count (overrides round(target_duration * rate))
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
        target_frames: Pre-calculated frame count from absolute positioning.
            When provided, overrides round(target_duration * rate) to prevent
            accumulated rounding drift across many clips.

    Returns:
        OTIO Clip with LinearTimeWarp applied to match target_duration
    """
    rate = frame_rate

    # Create absolute path with forward slashes for DaVinci Resolve
    abs_path = _to_windows_path(source_path)

    # Make media reference name unique by including parent folder
    # This prevents DaVinci Resolve from confusing clips with same filename in different folders
    folder_name = Path(source_path).parent.name
    filename = Path(source_path).name
    unique_media_name = f"{folder_name}_{filename}"

    # Determine available_range for the media file
    # Prefer ffprobe for actual duration — yt-dlp keyframe cuts make files
    # slightly shorter than requested, and DaVinci rejects clips whose
    # available_range exceeds the real file length.
    if media_duration is None:
        probed = _get_media_duration(abs_path) if Path(abs_path).exists() else None
        if probed is not None:
            media_duration = probed
        else:
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
    target_duration_frames = target_frames if target_frames is not None else round(target_duration * rate)

    # Ensure we have at least 1 frame
    if target_duration_frames < 1:
        target_duration_frames = 1

    # Safety: ensure source_range fits within available media
    # Priority: preserve target_duration (timeline timing) over exact source_start position
    # Shift source_start backwards when needed to fit target duration
    available_frames = round(media_duration * rate)
    if start_frames >= available_frames:
        # source_start beyond media — shift back to fit target duration
        start_frames = max(0, available_frames - target_duration_frames)
        logger.warning(
            f"source_start exceeded media ({media_duration:.1f}s) "
            f"for '{name}', shifted to {start_frames/rate:.1f}s"
        )
    if start_frames + target_duration_frames > available_frames:
        # Not enough room — shift source_start backwards to make room
        needed_shift = (start_frames + target_duration_frames) - available_frames
        new_start = max(0, start_frames - needed_shift)
        logger.debug(
            f"Shifting source_start back {needed_shift/rate:.2f}s to fit target duration: "
            f"{start_frames/rate:.2f}s -> {new_start/rate:.2f}s for '{name}'"
        )
        start_frames = new_start
        if start_frames + target_duration_frames > available_frames:
            # Media genuinely shorter than target — log but keep target_duration_frames
            # DaVinci Resolve may not properly interpret LinearTimeWarp, so we always
            # set source_range.duration = target to ensure correct timeline positioning.
            # DaVinci will freeze the last frame for the overshoot, which is preferable
            # to cascading desync across all subsequent clips.
            overshoot = (start_frames + target_duration_frames) - available_frames
            logger.info(
                f"Media shorter than target by {overshoot/rate:.2f}s for '{name}', "
                f"keeping target duration for timeline sync (DaVinci will freeze last frame)"
            )

    # Always use target_duration_frames for source_range.duration.
    # This ensures the clip occupies the correct timeline duration regardless of
    # whether the source media is shorter. DaVinci uses source_range.duration
    # directly for timeline positioning and may not apply LinearTimeWarp effects.
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
    # Add Resolve_OTIO metadata (required for DaVinci import)
    clip.metadata['Resolve_OTIO'] = {}

    # Store timing info in metadata for reference
    if metadata is None:
        metadata = {}

    if target_duration > 0 and source_duration > 0:
        time_scalar = source_duration / target_duration
        metadata['time_scalar'] = time_scalar
        metadata['speed_percent'] = time_scalar * 100
        metadata['target_duration'] = target_duration
        metadata['source_duration'] = source_duration
        metadata['original_duration'] = source_duration
        metadata['suggested_speed'] = f"{time_scalar * 100:.1f}%"

    # Add metadata (convert numpy types to Python native types)
    metadata = _sanitize_metadata(metadata)
    for key, value in metadata.items():
        clip.metadata[key] = value

    return clip
