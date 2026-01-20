"""DaVinci Resolve marker import for editorial feedback.

Parses marker exports from DaVinci Resolve (CSV or EDL) to extract feedback on video clips.
Supports both explicit naming (REJECT:, GOOD:) and color-based conventions.

DaVinci Resolve CSV Format (Timeline > Export > Markers > CSV):
#,Name,Start TC,End TC,Duration,Notes,Color
1,REJECT: trainer,01:00:00:00,01:00:05:00,00:00:05:00,Cesar Millan clip,Red
2,GOOD: emotional,01:05:00:00,01:05:10:00,00:00:10:00,Keep this,Green

DaVinci Resolve EDL Format (Timeline > Export > Markers > EDL):
001  001      V     C        01:00:00:00 01:00:00:01 01:00:00:00 01:00:00:01
 |C:ResolveColorRed |M:REJECT: trainer |D:1

Marker Color Conventions:
- Red: Reject - don't use again (add to rejection database)
- Orange: Warning - review needed (flag for manual review)
- Yellow: Replace - find alternative (re-match segment)
- Green: Approved - use more like this (boost channel score)
- Blue: Note - informational only (log only)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .rejections import RejectedVideo

logger = logging.getLogger(__name__)


class MarkerAction(Enum):
    """Action to take based on marker."""
    REJECT = "reject"       # Add to rejection database
    APPROVE = "approve"     # Boost channel score
    REPLACE = "replace"     # Flag for re-matching
    WARNING = "warning"     # Flag for review
    NOTE = "note"           # Log only


# Action priority for conflict resolution (lower number = higher priority)
ACTION_PRIORITY = {
    MarkerAction.REJECT: 1,   # Highest - if rejected, don't use
    MarkerAction.REPLACE: 2,  # Need to find alternative
    MarkerAction.WARNING: 3,  # Needs review
    MarkerAction.APPROVE: 4,  # Positive feedback
    MarkerAction.NOTE: 5,     # Lowest - informational only
}


# Marker color to action mapping
COLOR_ACTIONS = {
    'red': MarkerAction.REJECT,
    'orange': MarkerAction.WARNING,
    'yellow': MarkerAction.REPLACE,
    'green': MarkerAction.APPROVE,
    'blue': MarkerAction.NOTE,
    # Additional color variations
    'pink': MarkerAction.REJECT,
    'purple': MarkerAction.WARNING,
    'cyan': MarkerAction.NOTE,
    'cream': MarkerAction.NOTE,
    'white': MarkerAction.NOTE,
    'black': MarkerAction.NOTE,
}

# DaVinci Resolve EDL color names (e.g., ResolveColorRed -> red)
EDL_COLOR_MAP = {
    'resolvecolorred': 'red',
    'resolvecolororange': 'orange',
    'resolvecoloryellow': 'yellow',
    'resolvecolorgreen': 'green',
    'resolvecolorblue': 'blue',
    'resolvecolorpink': 'pink',
    'resolvecolorpurple': 'purple',
    'resolvecolorcyan': 'cyan',
    'resolvecolorcream': 'cream',
    'resolvecolorwhite': 'white',
    'resolvecolorblack': 'black',
}

# Name prefix to action mapping (overrides color) - case insensitive
PREFIX_ACTIONS = {
    'reject:': MarkerAction.REJECT,
    'bad:': MarkerAction.REJECT,
    'remove:': MarkerAction.REJECT,
    'delete:': MarkerAction.REJECT,
    'good:': MarkerAction.APPROVE,
    'keep:': MarkerAction.APPROVE,
    'approved:': MarkerAction.APPROVE,
    'love:': MarkerAction.APPROVE,
    'replace:': MarkerAction.REPLACE,
    'alt:': MarkerAction.REPLACE,
    'swap:': MarkerAction.REPLACE,
    'warning:': MarkerAction.WARNING,
    'review:': MarkerAction.WARNING,
    'check:': MarkerAction.WARNING,
    'note:': MarkerAction.NOTE,
    'info:': MarkerAction.NOTE,
    'comment:': MarkerAction.NOTE,
}

# Required CSV columns (at least one timecode column required)
REQUIRED_COLUMNS = {'Name', 'Color'}
TIMECODE_COLUMNS = {'Start TC', 'start tc', 'In', 'Timecode', 'TC'}


@dataclass
class MarkerFeedback:
    """Feedback extracted from a DaVinci marker."""

    action: MarkerAction
    video_id: str
    channel_id: str
    channel_name: str
    title: str
    reason: str
    notes: str
    timecode: float
    timecode_str: str
    color: str
    project: str
    timestamp: str


@dataclass
class ImportResult:
    """Result of importing markers from a CSV file."""

    rejections: List[RejectedVideo]
    approvals: List[MarkerFeedback]
    replacements: List[MarkerFeedback]
    warnings: List[MarkerFeedback]
    notes: List[MarkerFeedback]
    unmapped: List[Dict]  # Markers that couldn't be mapped to videos
    import_warnings: List[str] = field(default_factory=list)  # Warnings during import

    @property
    def total_markers(self) -> int:
        return (
            len(self.rejections) +
            len(self.approvals) +
            len(self.replacements) +
            len(self.warnings) +
            len(self.notes) +
            len(self.unmapped)
        )

    def summary(self) -> str:
        """Generate summary string."""
        parts = []
        if self.rejections:
            parts.append(f"{len(self.rejections)} rejections")
        if self.approvals:
            parts.append(f"{len(self.approvals)} approvals")
        if self.replacements:
            parts.append(f"{len(self.replacements)} replacements")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        if self.notes:
            parts.append(f"{len(self.notes)} notes")
        if self.unmapped:
            parts.append(f"{len(self.unmapped)} unmapped")
        return ", ".join(parts) if parts else "no markers"


class CSVValidationError(Exception):
    """Raised when CSV validation fails."""
    pass


def validate_csv_structure(csv_path: Path) -> Tuple[str, Set[str]]:
    """Validate CSV has required columns and detect delimiter.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (delimiter, set of column names)

    Raises:
        CSVValidationError: If CSV is malformed or missing required columns
    """
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            # Check if file is empty
            content = f.read()
            if not content.strip():
                raise CSVValidationError(f"CSV file is empty: {csv_path}")

            f.seek(0)

            # Detect delimiter
            sample = content[:2048]
            delimiter = '\t' if '\t' in sample else ','

            # Read header
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                raise CSVValidationError(f"CSV file has no header row: {csv_path}")

            # Normalize header names (strip whitespace)
            header = [col.strip() for col in header]
            header_set = set(header)
            header_lower = {col.lower() for col in header}

            # Check for required columns (case-insensitive)
            missing = []
            for req in REQUIRED_COLUMNS:
                if req not in header_set and req.lower() not in header_lower:
                    missing.append(req)

            # Check for at least one timecode column
            has_timecode = any(
                tc in header_set or tc.lower() in header_lower
                for tc in TIMECODE_COLUMNS
            )
            if not has_timecode:
                missing.append("Start TC (or In/Timecode)")

            if missing:
                raise CSVValidationError(
                    f"CSV missing required columns: {', '.join(missing)}. "
                    f"Found columns: {', '.join(header)}"
                )

            return delimiter, header_set

    except UnicodeDecodeError as e:
        raise CSVValidationError(f"CSV encoding error (try UTF-8): {e}")
    except csv.Error as e:
        raise CSVValidationError(f"CSV parsing error: {e}")


def parse_edl_markers(edl_path: Path, fps: float = 24.0) -> List[Dict]:
    """Parse DaVinci Resolve EDL marker export.

    EDL Format:
    001  001      V     C        01:03:12:06 01:03:12:07 01:03:12:06 01:03:12:07
     |C:ResolveColorYellow |M:Marker 1 |D:1

    Args:
        edl_path: Path to EDL file
        fps: Frames per second for timecode parsing

    Returns:
        List of marker dicts with keys: name, color, timecode_str, notes
    """
    markers = []

    # Regex patterns for EDL marker lines
    # Event line: captures record-in timecode (4th timecode)
    event_pattern = re.compile(
        r'^\d+\s+\d+\s+[VAB]+\s+\w+\s+'  # Event num, reel, track, transition
        r'(\d{2}:\d{2}:\d{2}:\d{2})\s+'   # Source in
        r'(\d{2}:\d{2}:\d{2}:\d{2})\s+'   # Source out
        r'(\d{2}:\d{2}:\d{2}:\d{2})\s+'   # Record in (this is what we want)
        r'(\d{2}:\d{2}:\d{2}:\d{2})'      # Record out
    )

    # Marker info line: |C:ResolveColor<Color> |M:<Name> |D:<Duration>
    marker_info_pattern = re.compile(
        r'\|C:(\w+)\s*\|M:([^|]*?)(?:\s*\|D:\d+)?$'
    )

    try:
        with open(edl_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        current_timecode = None
        current_extra_text = ""

        for line in lines:
            line = line.rstrip()

            # Skip empty lines and header
            if not line or line.startswith('TITLE:') or line.startswith('FCM:'):
                continue

            # Check for event line (starts with number)
            event_match = event_pattern.match(line)
            if event_match:
                # Record-in timecode is the 3rd captured group
                current_timecode = event_match.group(3)
                current_extra_text = ""
                continue

            # Check for marker info line (contains |C: and |M:)
            if '|C:' in line and '|M:' in line:
                # Extract any text before the |C: tag (extra notes)
                pre_text = line.split('|C:')[0].strip()
                if pre_text:
                    current_extra_text = pre_text

                marker_match = marker_info_pattern.search(line)
                if marker_match and current_timecode:
                    edl_color = marker_match.group(1).lower()
                    marker_name = marker_match.group(2).strip()

                    # Convert EDL color to standard color
                    color = EDL_COLOR_MAP.get(edl_color, edl_color.replace('resolvecolor', ''))

                    # Combine marker name with any extra text as notes
                    notes = current_extra_text if current_extra_text else ""

                    markers.append({
                        'name': marker_name,
                        'color': color,
                        'timecode_str': current_timecode,
                        'notes': notes,
                    })

                    current_timecode = None
                    current_extra_text = ""

        logger.info(f"Parsed {len(markers)} markers from EDL: {edl_path.name}")
        return markers

    except Exception as e:
        logger.error(f"Error parsing EDL file: {e}")
        raise


def detect_marker_file_type(file_path: Path) -> str:
    """Detect if file is CSV or EDL format.

    Args:
        file_path: Path to marker file

    Returns:
        'csv' or 'edl'
    """
    suffix = file_path.suffix.lower()
    if suffix == '.edl':
        return 'edl'
    elif suffix == '.csv':
        return 'csv'

    # Check content for EDL markers
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            first_lines = f.read(500)
        if 'TITLE:' in first_lines and 'FCM:' in first_lines:
            return 'edl'
        if '|C:ResolveColor' in first_lines:
            return 'edl'
    except Exception:
        pass

    return 'csv'  # Default to CSV


def parse_timecode(tc: str, fps: float = 24.0) -> float:
    """Parse timecode string to seconds.

    Args:
        tc: Timecode string (HH:MM:SS:FF or HH:MM:SS;FF)
        fps: Frames per second (default 24)

    Returns:
        Time in seconds
    """
    if not tc:
        return 0.0

    tc = tc.strip()

    # Handle both : and ; as frame separator
    tc = tc.replace(';', ':')
    parts = tc.split(':')

    try:
        if len(parts) == 4:
            hours, minutes, seconds, frames = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds + frames / fps
        elif len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # MM:SS format
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            logger.warning(f"Unexpected timecode format: {tc}")
            return 0.0
    except ValueError as e:
        logger.warning(f"Failed to parse timecode '{tc}': {e}")
        return 0.0


def determine_action(name: str, color: str) -> Tuple[MarkerAction, str, Optional[str]]:
    """Determine action from marker name and color.

    Args:
        name: Marker name/text
        color: Marker color

    Returns:
        Tuple of (action, reason/description, warning if unknown color)
    """
    name_lower = name.lower().strip()
    warning = None

    # Check for explicit prefix (overrides color) - case insensitive
    for prefix, action in PREFIX_ACTIONS.items():
        if name_lower.startswith(prefix):
            reason = name[len(prefix):].strip()
            return action, reason, None

    # Fall back to color-based action
    color_lower = color.lower().strip()
    action = COLOR_ACTIONS.get(color_lower)

    if action is None:
        # Unknown color - warn and default to NOTE
        warning = f"Unknown marker color '{color}', treating as NOTE"
        action = MarkerAction.NOTE

    return action, name, warning


def resolve_marker_conflicts(
    markers_by_video: Dict[str, List['MarkerFeedback']],
) -> Tuple[Dict[str, 'MarkerFeedback'], List[str]]:
    """Resolve conflicts when multiple markers are on the same video.

    Priority order: REJECT > REPLACE > WARNING > APPROVE > NOTE

    Args:
        markers_by_video: Dict mapping video_id to list of MarkerFeedback objects

    Returns:
        Tuple of (resolved markers dict, list of conflict warnings)
    """
    resolved = {}
    conflict_warnings = []

    for video_id, markers in markers_by_video.items():
        if len(markers) == 1:
            # No conflict
            resolved[video_id] = markers[0]
            continue

        # Sort by priority (lowest number = highest priority)
        sorted_markers = sorted(markers, key=lambda m: ACTION_PRIORITY[m.action])
        winner = sorted_markers[0]
        resolved[video_id] = winner

        # Check for actual conflicts (different action types)
        action_types = set(m.action for m in markers)
        if len(action_types) > 1:
            # Build conflict warning
            conflict_details = []
            for m in markers:
                conflict_details.append(f"{m.action.value.upper()}: {m.reason} ({m.color})")

            warning = (
                f"Conflicting markers on video '{video_id}' at {markers[0].timecode_str}:\n"
                f"   " + "\n   ".join(conflict_details) + "\n"
                f"   → Using {winner.action.value.upper()} (highest priority)"
            )
            conflict_warnings.append(warning)
            logger.warning(warning)
        else:
            # Same action type, just note the duplicates
            logger.debug(
                f"Multiple {winner.action.value} markers on video '{video_id}', "
                f"using first: {winner.reason}"
            )

    return resolved, conflict_warnings


def detect_fps_from_segments(project_dir: Path) -> float:
    """Detect frame rate from timeline_segments.json.

    Args:
        project_dir: Project directory

    Returns:
        Frame rate (default 24.0 if not found)
    """
    output_dirs = sorted(
        (project_dir / 'output').glob('*'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ) if (project_dir / 'output').exists() else []

    for output_dir in output_dirs:
        # Try timeline_segments.json first (actual filename)
        for filename in ['timeline_segments.json', 'segments.json']:
            segments_file = output_dir / filename
            if segments_file.exists():
                try:
                    with open(segments_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    fps = data.get('frame_rate', 24.0)
                    if fps:
                        logger.debug(f"Detected frame rate from {filename}: {fps}")
                        return float(fps)
                except Exception as e:
                    logger.debug(f"Error reading frame rate: {e}")

    return 24.0


def check_segments_freshness(csv_path: Path, project_dir: Path) -> Optional[str]:
    """Check if timeline_segments.json is older than the CSV file.

    Args:
        csv_path: Path to marker CSV
        project_dir: Project directory

    Returns:
        Warning message if stale, None otherwise
    """
    output_dirs = sorted(
        (project_dir / 'output').glob('*'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ) if (project_dir / 'output').exists() else []

    if not output_dirs:
        return "No output folder found - timeline_segments.json may be missing"

    # Try both filenames
    segments_file = None
    for filename in ['timeline_segments.json', 'segments.json']:
        candidate = output_dirs[0] / filename
        if candidate.exists():
            segments_file = candidate
            break

    if not segments_file:
        return f"timeline_segments.json not found in {output_dirs[0].name}"

    csv_mtime = csv_path.stat().st_mtime
    segments_mtime = segments_file.stat().st_mtime

    if segments_mtime < csv_mtime:
        csv_time = datetime.fromtimestamp(csv_mtime).strftime('%Y-%m-%d %H:%M')
        seg_time = datetime.fromtimestamp(segments_mtime).strftime('%Y-%m-%d %H:%M')
        return (
            f"timeline_segments.json ({seg_time}) is older than markers.csv ({csv_time}). "
            f"Timeline may have changed - timecode mapping could be inaccurate. "
            f"Consider re-running the pipeline before importing markers."
        )

    return None


def load_video_metadata_from_checkpoint(project_dir: Path) -> Dict[str, Dict]:
    """Load video metadata from checkpoint.json.

    Builds a lookup from video_id and filename to video info.

    Args:
        project_dir: Project directory

    Returns:
        Dict mapping video_id/filename to video info
    """
    lookup: Dict[str, Dict] = {}
    checkpoint_file = project_dir / 'checkpoint.json'

    if not checkpoint_file.exists():
        return lookup

    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        video_candidates = data.get('video_metadata', {}).get('video_candidates', [])

        for v in video_candidates:
            video_id = v.get('video_id', '')
            if video_id:
                info = {
                    'id': video_id,
                    'channel_id': v.get('channel_id', ''),
                    'channel': v.get('channel', ''),
                    'title': v.get('title', ''),
                }
                # Index by video_id
                lookup[video_id] = info
                # Also index by title (lowercase) for fallback
                title = v.get('title', '')
                if title:
                    lookup[title.lower()] = info

        logger.debug(f"Loaded {len(video_candidates)} video metadata entries from checkpoint")

    except Exception as e:
        logger.debug(f"Error loading video metadata from checkpoint: {e}")

    return lookup


def load_segments_data(project_dir: Path) -> Tuple[List[Dict], Dict[str, Dict], float, float]:
    """Load segments and build lookup indices.

    Args:
        project_dir: Project directory

    Returns:
        Tuple of (segments list with computed times, video lookup dict, frame_rate, timeline_start_offset)
        timeline_start_offset is the time offset from timeline_start_tc (e.g., 3600.0 for 01:00:00:00)
    """
    segments = []
    video_lookup: Dict[str, Dict] = {}
    frame_rate = 30.0
    timeline_start_offset = 0.0  # Will be extracted from timeline_start_tc

    # Load video metadata from checkpoint
    video_lookup = load_video_metadata_from_checkpoint(project_dir)

    output_dirs = sorted(
        (project_dir / 'output').glob('*'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ) if (project_dir / 'output').exists() else []

    for output_dir in output_dirs:
        # Try both filenames
        segments_file = None
        for filename in ['timeline_segments.json', 'segments.json']:
            candidate = output_dir / filename
            if candidate.exists():
                segments_file = candidate
                break

        if segments_file:
            try:
                with open(segments_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                frame_rate = data.get('frame_rate', 30.0)
                raw_segments = data.get('segments', [])

                # Extract timeline start offset (DaVinci typically starts at 01:00:00:00)
                timeline_start_tc = data.get('timeline_start_tc', '00:00:00:00')
                timeline_start_offset = parse_timecode(timeline_start_tc, frame_rate)
                logger.debug(f"Timeline start offset: {timeline_start_tc} = {timeline_start_offset}s")

                # Convert timeline_segments.json format to normalized format
                for seg in raw_segments:
                    # Calculate start/end time from frames
                    start_frame = seg.get('start_frame', 0)
                    end_frame = seg.get('end_frame', 0)
                    start_time = start_frame / frame_rate
                    end_time = end_frame / frame_rate

                    # Get video info from v1_clip
                    v1_clip = seg.get('v1_clip', {})
                    filename = v1_clip.get('file', '')

                    # Extract video_id from filename (format: VIDEO_ID_SCENE.mp4 or VIDEO_ID.mp3)
                    video_id = ''
                    if filename:
                        # Remove extension and scene suffix
                        base = filename.rsplit('.', 1)[0]  # Remove .mp4/.mp3
                        # Handle VIDEO_ID_0067 format
                        if '_' in base and base.rsplit('_', 1)[1].isdigit():
                            video_id = base.rsplit('_', 1)[0]
                        else:
                            video_id = base

                    # Look up full video info
                    video_info = video_lookup.get(video_id, {})

                    normalized_seg = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'match': {
                            'video_id': video_id,
                            'channel_id': video_info.get('channel_id', ''),
                            'channel': video_info.get('channel', ''),
                            'video_title': video_info.get('title', filename),
                        },
                        'voiceover_text': seg.get('voiceover_text', ''),
                    }
                    segments.append(normalized_seg)

                    # Also add to lookup by title
                    title = video_info.get('title', '')
                    if title and title.lower() not in video_lookup:
                        video_lookup[title.lower()] = {
                            'id': video_id,
                            'channel_id': video_info.get('channel_id', ''),
                            'channel': video_info.get('channel', ''),
                            'title': title,
                        }

                logger.debug(f"Loaded {len(segments)} segments from {segments_file.name}")
                break

            except Exception as e:
                logger.debug(f"Error loading segments: {e}")

    return segments, video_lookup, frame_rate, timeline_start_offset


def find_video_at_timecode(
    timecode: float,
    segments: List[Dict],
    snap_threshold: float = 2.0,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Find which video is at the given timecode in the timeline.

    Args:
        timecode: Timeline position in seconds
        segments: List of segment dictionaries
        snap_threshold: Seconds to snap to nearest segment if between segments

    Returns:
        Tuple of (video info dict, warning message if snapped)
    """
    if not segments:
        return None, None

    # First try exact match
    for seg in segments:
        seg_start = seg.get('start_time', 0)
        seg_end = seg.get('end_time', 0)

        if seg_start <= timecode < seg_end:
            match = seg.get('match', {})
            if match:
                return {
                    'id': match.get('video_id', ''),
                    'channel_id': match.get('channel_id', ''),
                    'channel': match.get('channel', ''),
                    'title': match.get('video_title', ''),
                }, None

    # Try snapping to nearest segment
    closest_seg = None
    closest_dist = float('inf')

    for seg in segments:
        seg_start = seg.get('start_time', 0)
        seg_end = seg.get('end_time', 0)

        # Distance to segment
        if timecode < seg_start:
            dist = seg_start - timecode
        elif timecode >= seg_end:
            dist = timecode - seg_end
        else:
            dist = 0

        if dist < closest_dist:
            closest_dist = dist
            closest_seg = seg

    if closest_seg and closest_dist <= snap_threshold:
        match = closest_seg.get('match', {})
        if match:
            warning = f"Marker at {timecode:.2f}s snapped to nearest segment ({closest_dist:.2f}s away)"
            return {
                'id': match.get('video_id', ''),
                'channel_id': match.get('channel_id', ''),
                'channel': match.get('channel', ''),
                'title': match.get('video_title', ''),
            }, warning

    return None, None


def import_davinci_markers(
    csv_path: Path,
    project_dir: Optional[Path] = None,
    fps: float = 24.0,
) -> List[RejectedVideo]:
    """Parse DaVinci Resolve marker export CSV (legacy compatibility).

    This function maintains backward compatibility by returning only rejections.
    For full feedback import, use import_davinci_markers_full().

    Args:
        csv_path: Path to exported CSV file
        project_dir: Project directory for finding video info
        fps: Frames per second for timecode parsing

    Returns:
        List of RejectedVideo objects for rejection markers
    """
    result = import_davinci_markers_full(csv_path, project_dir, fps)
    return result.rejections


def import_davinci_markers_full(
    marker_path: Path,
    project_dir: Optional[Path] = None,
    fps: Optional[float] = None,
) -> ImportResult:
    """Parse DaVinci Resolve marker export (CSV or EDL) with full feedback support.

    Handles all marker colors and name prefixes:
    - Red/REJECT: Add to rejection database
    - Green/GOOD: Boost channel score
    - Yellow/REPLACE: Flag for re-matching
    - Orange/WARNING: Flag for review
    - Blue/NOTE: Log only

    Args:
        marker_path: Path to exported marker file (CSV or EDL)
        project_dir: Project directory for finding video info
        fps: Frames per second for timecode parsing (auto-detected if None)

    Returns:
        ImportResult with categorized feedback
    """
    marker_path = Path(marker_path)
    if not marker_path.exists():
        raise FileNotFoundError(f"Marker file not found: {marker_path}")

    # Initialize result lists
    rejections = []
    approvals = []
    replacements = []
    warnings = []
    notes = []
    unmapped = []
    import_warnings = []
    markers_by_video: Dict[str, List[MarkerFeedback]] = {}  # For conflict resolution

    project_name = project_dir.name if project_dir else ""
    timestamp = datetime.now().isoformat()

    # Detect file type
    file_type = detect_marker_file_type(marker_path)
    logger.info(f"Detected marker file type: {file_type.upper()}")

    # Auto-detect fps from project if not provided
    if fps is None and project_dir:
        fps = detect_fps_from_segments(project_dir)
    elif fps is None:
        fps = 24.0

    logger.info(f"Using frame rate: {fps} fps")

    # Check if segments.json is fresh
    if project_dir:
        staleness_warning = check_segments_freshness(marker_path, project_dir)
        if staleness_warning:
            logger.warning(staleness_warning)
            import_warnings.append(staleness_warning)

    # Load segments data
    segments = []
    video_lookup: Dict[str, Dict] = {}
    timeline_start_offset = 0.0
    if project_dir:
        segments, video_lookup, detected_fps, timeline_start_offset = load_segments_data(project_dir)
        # Use detected fps if not explicitly provided
        if fps is None or fps == 24.0:
            fps = detected_fps
            logger.info(f"Using detected frame rate: {fps} fps")
        if timeline_start_offset > 0:
            logger.info(f"Timeline start offset: {timeline_start_offset}s (will be subtracted from marker timecodes)")

    # Track timecodes for duplicate detection
    timecode_counts: Counter = Counter()

    # Parse markers based on file type
    if file_type == 'edl':
        # Parse EDL format
        raw_markers = parse_edl_markers(marker_path, fps)
        marker_rows = [
            {
                'name': m['name'],
                'color': m['color'],
                'start_tc': m['timecode_str'],
                'notes': m.get('notes', ''),
            }
            for m in raw_markers
        ]
    else:
        # Parse CSV format
        try:
            delimiter, header_set = validate_csv_structure(marker_path)
        except CSVValidationError as e:
            logger.error(str(e))
            raise

        marker_rows = []
        with open(marker_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Get marker data (case-insensitive column access)
                name = (row.get('Name') or row.get('name') or '').strip()
                color = (row.get('Color') or row.get('color') or '').strip()
                notes_text = (
                    row.get('Notes') or row.get('notes') or
                    row.get('Comment') or row.get('comment') or ''
                ).strip()

                # Find timecode column
                start_tc = None
                for tc_col in ['Start TC', 'start tc', 'In', 'Timecode', 'TC']:
                    if tc_col in row:
                        start_tc = row[tc_col]
                        break
                start_tc = start_tc or ''

                marker_rows.append({
                    'name': name,
                    'color': color,
                    'start_tc': start_tc,
                    'notes': notes_text,
                })

    try:
        row_count = 0
        for row in marker_rows:
            row_count += 1

            name = row['name']
            color = row['color']
            start_tc = row['start_tc']
            notes_text = row.get('notes', '')

            if not name and not color:
                continue  # Skip empty rows

            # Parse timecode and apply timeline offset
            timecode_raw = parse_timecode(start_tc, fps)
            timecode = timecode_raw - timeline_start_offset

            # Handle negative timecodes (marker before timeline start)
            if timecode < 0:
                logger.debug(f"Marker at {start_tc} is before timeline start, adjusting to 0")
                timecode = 0.0

            # Check for duplicate timecodes
            tc_key = f"{timecode:.2f}"
            timecode_counts[tc_key] += 1
            if timecode_counts[tc_key] > 1:
                dup_warning = f"Duplicate marker at timecode {start_tc} (occurrence #{timecode_counts[tc_key]})"
                logger.warning(dup_warning)
                import_warnings.append(dup_warning)

            # Determine action and reason
            action, reason, color_warning = determine_action(name, color)
            if color_warning:
                logger.warning(color_warning)
                import_warnings.append(color_warning)

            # Try to find video at timecode
            video_info = None
            snap_warning = None
            if segments:
                video_info, snap_warning = find_video_at_timecode(timecode, segments)
                if snap_warning:
                    logger.info(snap_warning)
                    import_warnings.append(snap_warning)

            # Fallback: try title lookup from notes
            if not video_info and notes_text and video_lookup:
                notes_lower = notes_text.lower()
                for title, info in video_lookup.items():
                    if isinstance(title, str) and title in notes_lower:
                        video_info = info
                        logger.info(f"Matched marker via notes text to: {info.get('title', title)[:40]}")
                        break

            if not video_info or not video_info.get('id'):
                # Can't map to video - store for logging
                unmapped.append({
                    'name': name,
                    'color': color,
                    'timecode': start_tc,
                    'timecode_seconds': timecode,
                    'notes': notes_text,
                    'action': action.value,
                    'reason': reason,
                })
                logger.warning(
                    f"Unmapped marker at {start_tc} ({action.value}: {reason[:30]}...) - "
                    f"could not find video at this timecode"
                )
                continue

            # Create feedback object
            feedback = MarkerFeedback(
                action=action,
                video_id=video_info['id'],
                channel_id=video_info.get('channel_id', ''),
                channel_name=video_info.get('channel', ''),
                title=video_info.get('title', ''),
                reason=reason,
                notes=notes_text,
                timecode=timecode,
                timecode_str=start_tc,
                color=color,
                project=project_name,
                timestamp=timestamp,
            )

            # Collect markers for conflict resolution (group by video_id)
            video_id = video_info['id']
            if video_id not in markers_by_video:
                markers_by_video[video_id] = []
            markers_by_video[video_id].append(feedback)

        # Resolve conflicts (multiple markers on same video)
        resolved_markers, conflict_warnings = resolve_marker_conflicts(markers_by_video)
        import_warnings.extend(conflict_warnings)

        # Categorize resolved markers by action
        for video_id, feedback in resolved_markers.items():
            action = feedback.action

            if action == MarkerAction.REJECT:
                rejection = RejectedVideo(
                    video_id=feedback.video_id,
                    channel_id=feedback.channel_id,
                    channel_name=feedback.channel_name,
                    title=feedback.title,
                    rejection_reason=feedback.reason,
                    rejection_source='davinci_marker',
                    project=project_name,
                )
                rejections.append(rejection)
                logger.info(f"Rejection: {feedback.title[:40]}... ({feedback.reason})")

            elif action == MarkerAction.APPROVE:
                approvals.append(feedback)
                logger.info(f"Approval: {feedback.title[:40]}... ({feedback.reason})")

            elif action == MarkerAction.REPLACE:
                replacements.append(feedback)
                logger.info(f"Replace: {feedback.title[:40]}... ({feedback.reason})")

            elif action == MarkerAction.WARNING:
                warnings.append(feedback)
                logger.info(f"Warning: {feedback.title[:40]}... ({feedback.reason})")

            else:  # NOTE
                notes.append(feedback)
                logger.debug(f"Note: {feedback.title[:40]}... ({feedback.reason})")

        # Log summary
        if row_count == 0:
            import_warnings.append("CSV file contained no data rows")

        result = ImportResult(
            rejections=rejections,
            approvals=approvals,
            replacements=replacements,
            warnings=warnings,
            notes=notes,
            unmapped=unmapped,
            import_warnings=import_warnings,
        )

        logger.info(f"Imported markers from {marker_path.name}: {result.summary()}")

        if import_warnings:
            logger.warning(f"{len(import_warnings)} warnings during import")

        return result

    except Exception as e:
        logger.error(f"Error parsing marker file: {e}")
        raise


def apply_approvals_to_database(
    approvals: List[MarkerFeedback],
    rejection_db,
) -> int:
    """Apply approval markers to boost channel scores.

    Args:
        approvals: List of approval feedback
        rejection_db: RejectionDatabase instance

    Returns:
        Number of acceptances recorded
    """
    count = 0
    for approval in approvals:
        if approval.video_id and approval.channel_id:
            rejection_db.record_acceptance(
                video_id=approval.video_id,
                channel_id=approval.channel_id,
                channel_name=approval.channel_name,
            )
            count += 1

    if count > 0:
        logger.info(f"Recorded {count} video acceptances for channel scoring")

    return count


def export_rejections_to_csv(
    rejections: List[RejectedVideo],
    output_path: Path,
) -> None:
    """Export rejections to CSV format (for sharing/backup).

    Args:
        rejections: List of RejectedVideo objects
        output_path: Path to write CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'video_id',
            'channel_name',
            'title',
            'rejection_reason',
            'rejection_source',
            'project',
            'timestamp',
        ])

        for r in rejections:
            writer.writerow([
                r.video_id,
                r.channel_name,
                r.title,
                r.rejection_reason,
                r.rejection_source,
                r.project,
                r.timestamp,
            ])

    logger.info(f"Exported {len(rejections)} rejections to {output_path}")


def generate_feedback_report(result: ImportResult, output_path: Path) -> None:
    """Generate a human-readable feedback report.

    Args:
        result: ImportResult from marker import
        output_path: Path to write report
    """
    lines = [
        "# DaVinci Marker Feedback Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- Total markers processed: {result.total_markers}",
        f"- Rejections: {len(result.rejections)}",
        f"- Approvals: {len(result.approvals)}",
        f"- Replacements: {len(result.replacements)}",
        f"- Warnings: {len(result.warnings)}",
        f"- Notes: {len(result.notes)}",
        f"- Unmapped: {len(result.unmapped)}",
        "",
    ]

    # Add import warnings
    if result.import_warnings:
        lines.extend([
            "## Import Warnings",
            "",
        ])
        for warning in result.import_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if result.rejections:
        lines.extend([
            "## Rejections (Red Markers)",
            "",
            "| Video | Channel | Reason |",
            "|-------|---------|--------|",
        ])
        for r in result.rejections:
            title = r.title[:30] + "..." if len(r.title) > 30 else r.title
            lines.append(f"| {title} | {r.channel_name} | {r.rejection_reason} |")
        lines.append("")

    if result.approvals:
        lines.extend([
            "## Approvals (Green Markers)",
            "",
            "| Video | Channel | Notes |",
            "|-------|---------|-------|",
        ])
        for a in result.approvals:
            title = a.title[:30] + "..." if len(a.title) > 30 else a.title
            lines.append(f"| {title} | {a.channel_name} | {a.reason} |")
        lines.append("")

    if result.replacements:
        lines.extend([
            "## Replacements Needed (Yellow Markers)",
            "",
            "| Timecode | Video | Reason |",
            "|----------|-------|--------|",
        ])
        for r in result.replacements:
            title = r.title[:30] + "..." if len(r.title) > 30 else r.title
            lines.append(f"| {r.timecode_str} | {title} | {r.reason} |")
        lines.append("")

    if result.warnings:
        lines.extend([
            "## Warnings (Orange Markers)",
            "",
            "| Timecode | Video | Notes |",
            "|----------|-------|-------|",
        ])
        for w in result.warnings:
            title = w.title[:30] + "..." if len(w.title) > 30 else w.title
            lines.append(f"| {w.timecode_str} | {title} | {w.reason} |")
        lines.append("")

    if result.unmapped:
        lines.extend([
            "## Unmapped Markers",
            "",
            "These markers could not be mapped to videos in the timeline.",
            "Check if the timeline was modified after the pipeline ran.",
            "",
            "| Timecode | Seconds | Name | Color | Action |",
            "|----------|---------|------|-------|--------|",
        ])
        for u in result.unmapped:
            name = u['name'][:25] + "..." if len(u['name']) > 25 else u['name']
            lines.append(
                f"| {u['timecode']} | {u.get('timecode_seconds', 0):.2f}s | "
                f"{name} | {u['color']} | {u['action']} |"
            )
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')

    logger.info(f"Generated feedback report: {output_path}")
