"""
Video Chapter Detector

Extracts chapter markers from video metadata (e.g., YouTube chapters).
Handles parsing, normalization, and validation of chapter data.
"""

import logging
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoChapter:
    """
    Represents a chapter extracted from video metadata.

    Attributes:
        title: Chapter title (normalized)
        start_time: Start time in seconds
        end_time: End time in seconds (or None if last chapter)
        duration: Duration in seconds (calculated)
    """
    title: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    def __post_init__(self):
        """Calculate duration if end_time is provided."""
        if self.end_time is not None and self.duration is None:
            self.duration = max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoChapter":
        """Create from dictionary."""
        return cls(
            title=data.get('title', ''),
            start_time=float(data.get('start_time', 0)),
            end_time=float(data['end_time']) if data.get('end_time') is not None else None,
            duration=float(data['duration']) if data.get('duration') is not None else None,
        )


def extract_chapters_from_metadata(
    metadata: Dict[str, Any],
    video_duration: Optional[float] = None
) -> List[VideoChapter]:
    """
    Extract chapter markers from video metadata.

    Supports multiple metadata formats:
    - YouTube 'chapters' field (list of dicts with start_time, title)
    - YouTube 'description' with timestamp markers
    - Generic 'chapters' list

    Args:
        metadata: Video metadata dictionary (e.g., from yt-dlp)
        video_duration: Optional total video duration for calculating last chapter end

    Returns:
        List of VideoChapter objects, empty if no chapters found
    """
    if not metadata:
        return []

    # Try structured chapters first
    chapters = _extract_structured_chapters(metadata, video_duration)
    if chapters:
        return chapters

    # Try parsing from description
    description = metadata.get('description', '')
    if description:
        chapters = _extract_chapters_from_description(description, video_duration)
        if chapters:
            return chapters

    return []


def _extract_structured_chapters(
    metadata: Dict[str, Any],
    video_duration: Optional[float] = None
) -> List[VideoChapter]:
    """Extract chapters from structured metadata fields."""
    raw_chapters = metadata.get('chapters')

    if not raw_chapters:
        return []

    if not isinstance(raw_chapters, list):
        logger.debug(f"Chapters field is not a list: {type(raw_chapters)}")
        return []

    chapters = []

    for i, ch in enumerate(raw_chapters):
        if not isinstance(ch, dict):
            logger.debug(f"Skipping non-dict chapter at index {i}")
            continue

        # Extract title
        title = _extract_chapter_title(ch)
        if not title:
            continue

        # Extract start time
        start_time = _extract_timestamp(ch, 'start_time', 'start')
        if start_time is None:
            continue

        # Extract end time (or calculate from next chapter)
        end_time = _extract_timestamp(ch, 'end_time', 'end')

        if end_time is None and i + 1 < len(raw_chapters):
            next_ch = raw_chapters[i + 1]
            if isinstance(next_ch, dict):
                end_time = _extract_timestamp(next_ch, 'start_time', 'start')

        if end_time is None and video_duration is not None:
            # Last chapter ends at video end
            if i == len(raw_chapters) - 1:
                end_time = video_duration

        # Normalize title
        normalized_title = _normalize_title(title)

        chapters.append(VideoChapter(
            title=normalized_title,
            start_time=start_time,
            end_time=end_time,
        ))

    return chapters


def _extract_chapters_from_description(
    description: str,
    video_duration: Optional[float] = None
) -> List[VideoChapter]:
    """
    Extract chapters from video description with timestamp markers.

    Common formats:
    - 0:00 Introduction
    - 00:00 - Title
    - 1:23:45 Chapter Name
    - [0:00] Title
    """
    if not description:
        return []

    # Pattern for timestamps: HH:MM:SS, MM:SS, or H:MM:SS
    # Supports:
    # - Standard: 0:00, 00:00, 1:23:45
    # - With decimals: 0:00.5, 1:23:45.678
    # - With brackets: [0:00], (0:00)
    # - With LIVE prefix: LIVE 0:00
    # - With leading zeros: 00:00:00
    timestamp_pattern = r'(?:^|\s|\[|(?:LIVE|live|Live)\s+)(\d{1,2}(?::\d{2}){1,2}(?:\.\d+)?)(?:\]|\)|\s*[-–—]?\s*)'

    lines = description.split('\n')
    raw_chapters = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(timestamp_pattern, line)
        if match:
            timestamp_str = match.group(1)
            start_time = _parse_timestamp_string(timestamp_str)
            if start_time is None:
                continue

            # Extract title (everything after the timestamp pattern)
            title_part = line[match.end():].strip()
            title_part = re.sub(r'^[-–—]\s*', '', title_part)  # Remove leading dashes

            if title_part:
                raw_chapters.append({
                    'title': title_part,
                    'start_time': start_time,
                })

    if not raw_chapters:
        return []

    # Calculate end times
    chapters = []
    for i, ch in enumerate(raw_chapters):
        end_time = None
        if i + 1 < len(raw_chapters):
            end_time = raw_chapters[i + 1]['start_time']
        elif video_duration is not None:
            end_time = video_duration

        chapters.append(VideoChapter(
            title=_normalize_title(ch['title']),
            start_time=ch['start_time'],
            end_time=end_time,
        ))

    return chapters


def _extract_chapter_title(chapter_data: Dict[str, Any]) -> Optional[str]:
    """Extract title from chapter data, handling various field names."""
    for field in ('title', 'name', 'label', 'chapter_title'):
        value = chapter_data.get(field)
        if value and isinstance(value, str):
            return value.strip()
    return None


def _extract_timestamp(
    data: Dict[str, Any],
    *field_names: str
) -> Optional[float]:
    """Extract timestamp value from dict, trying multiple field names."""
    for field in field_names:
        value = data.get(field)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                if isinstance(value, str):
                    parsed = _parse_timestamp_string(value)
                    if parsed is not None:
                        return parsed
    return None


def _parse_timestamp_string(timestamp: str) -> Optional[float]:
    """
    Parse a timestamp string to seconds.

    Supports:
    - HH:MM:SS, HH:MM:SS.sss, HH:MM:SS.s
    - MM:SS, MM:SS.sss, MM:SS.s
    - H:MM:SS
    - Formats with leading zeros: 00:00:00
    - Compact formats: 1h2m3s, 2m3s
    - Decimal seconds: 0:00.5, 1:23:45.678

    Returns:
        Timestamp in seconds, or None if parsing fails
    """
    if not timestamp:
        return None

    timestamp = timestamp.strip()

    # Try parsing compact formats (1h2m3s, 2m3s, 1h3s)
    compact_result = _parse_compact_timestamp(timestamp)
    if compact_result is not None:
        return compact_result

    # Remove any non-standard characters but keep colons, decimals
    # Handle cases like "LIVE 0:00" or "(0:00)" or "Chapter 0:00"
    timestamp = re.sub(r'^(LIVE|Live|live)\s+', '', timestamp)
    timestamp = re.sub(r'^\(?', '', timestamp)
    timestamp = re.sub(r'\)?$', '', timestamp)
    timestamp = timestamp.strip()

    # Handle milliseconds by truncating to reasonable precision
    # YouTube chapters don't need millisecond precision
    timestamp = re.sub(r'(\.\d{3})\d+', r'\1', timestamp)

    parts = timestamp.split(':')

    try:
        if len(parts) == 2:
            # MM:SS or MM:SS.s
            minutes, seconds = int(parts[0]), float(parts[1])
            if minutes < 0 or seconds < 0:
                return None
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS or HH:MM:SS.s
            hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0:
                return None
            return hours * 3600 + minutes * 60 + seconds
        else:
            return None
    except (ValueError, IndexError):
        return None


def _parse_compact_timestamp(timestamp: str) -> Optional[float]:
    """
    Parse compact timestamp format like 1h2m3s, 2m3s, 1h3s.

    Args:
        timestamp: String like "1h2m3s", "2m3s", "1h", "3s"

    Returns:
        Timestamp in seconds, or None if not a valid compact format
    """
    if not timestamp:
        return None

    # Must contain at least one of h, m, s
    if not any(c in timestamp.lower() for c in 'hms'):
        return None

    # Extract numeric values with their units
    hours = 0
    minutes = 0
    seconds = 0

    # Match patterns like "1h", "2m", "3s", "1h2m", "2m3s", "1h2m3s"
    hour_match = re.search(r'(\d+)\s*h', timestamp.lower())
    minute_match = re.search(r'(\d+)\s*m', timestamp.lower())
    second_match = re.search(r'(\d+)\s*s', timestamp.lower())

    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))
    if second_match:
        seconds = int(second_match.group(1))

    # If no valid numbers found, return None
    if hours == 0 and minutes == 0 and seconds == 0:
        return None

    return hours * 3600 + minutes * 60 + seconds


def _normalize_title(title: str) -> str:
    """
    Normalize chapter title.

    - Strips whitespace
    - Collapses multiple spaces
    - Removes leading numbers/bullets
    - Handles common prefixes
    """
    if not title:
        return ""

    # Strip and collapse spaces
    title = ' '.join(title.split())

    # Remove leading numbered bullets (e.g., "1.", "1)", "#1", etc.)
    title = re.sub(r'^[\d]+[.):\s]+\s*', '', title)

    # Remove leading symbols
    title = re.sub(r'^[•\-–—\*#]+\s*', '', title)

    # Remove surrounding quotes if present
    if len(title) >= 2:
        if (title[0] == '"' and title[-1] == '"') or \
           (title[0] == "'" and title[-1] == "'"):
            title = title[1:-1]

    return title.strip()
