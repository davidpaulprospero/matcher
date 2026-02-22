"""
Caption format parsers.

This module provides parsers for various caption formats:
- VTT (WebVTT)
- SRT (SubRip)
- JSON3/SRV3 (YouTube native)

All parsers support segment-level error recovery, allowing partial
results when some segments fail to parse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .models import CaptionSegment, ParseResult

logger = logging.getLogger(__name__)


def parse_timestamp(ts: str) -> Optional[float]:
    """Parse a timestamp string to seconds.

    This is the single canonical timestamp parsing function for all caption
    modules. All other timestamp parsers should delegate to this function.

    Supports formats:
        HH:MM:SS,mmm (SRT)
        HH:MM:SS.mmm (VTT)
        MM:SS.mmm (VTT short)
        HH:MM:SS (no milliseconds)
        MM:SS (no milliseconds, short)
        Bare float string (e.g., "123.456")

    Args:
        ts: Timestamp string to parse. None or empty returns None.

    Returns:
        Time in seconds, or None if parsing fails.
    """
    if not ts:
        return None

    ts = ts.strip()

    if not ts:
        return None

    # Replace comma with period for SRT format
    ts = ts.replace(',', '.')

    # Try bare float (e.g., "123.456" or "90")
    try:
        val = float(ts)
        if val < 0:
            logger.warning(f"Negative timestamp value: {ts}")
            return None
        return val
    except ValueError:
        pass

    # Pattern for HH:MM:SS.mmm or MM:SS.mmm (with optional milliseconds)
    match = re.match(r'^(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?$', ts)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int(match.group(4).ljust(3, '0')[:3]) if match.group(4) else 0

        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    logger.warning(f"Could not parse timestamp: {ts}")
    return None


def normalize_segments(
    segments: list[CaptionSegment],
    video_duration: Optional[float] = None,
    min_segment_duration: float = 0.1,
) -> list[CaptionSegment]:
    """Normalize caption segment timestamps for consistency (US-100-008).

    This function ensures all segments have valid, consistent timestamps:
    1. Non-negative start times
    2. start_time <= end_time (no backwards segments)
    3. Minimum segment duration enforced
    4. Optional: Cap end times to video duration

    Args:
        segments: List of CaptionSegment to normalize.
        video_duration: Optional video duration to cap segment end times.
        min_segment_duration: Minimum allowed segment duration in seconds.
            Default 0.1s. Segments shorter than this are extended.

    Returns:
        List of normalized CaptionSegment (may be same objects if no changes needed).

    Example:
        >>> segments = [
        ...     CaptionSegment(0, 5.0, 3.0, "Backwards", "vid"),  # end < start
        ...     CaptionSegment(1, -1.0, 2.0, "Negative", "vid"),   # negative start
        ...     CaptionSegment(2, 1.0, 1.05, "Short", "vid"),     # < min_duration
        ... ]
        >>> normalized = normalize_segments(segments, video_duration=10.0)
        >>> [(s.start_time, s.end_time) for s in normalized]
        [(0.0, 3.0), (0.0, 2.0), (1.0, 1.1)]
    """
    if not segments:
        return segments

    normalized = []
    for seg in segments:
        start = seg.start_time
        end = seg.end_time

        # Fix negative start times
        if start < 0:
            logger.debug(f"Segment {seg.index}: negative start {start}, clamping to 0.0")
            start = 0.0

        # Fix backwards segments (end < start)
        if end < start:
            logger.debug(f"Segment {seg.index}: end {end} < start {start}, swapping")
            start, end = end, start  # Swap to make end >= start

        # Fix too-short segments
        duration = end - start
        if duration < min_segment_duration:
            logger.debug(
                f"Segment {seg.index}: duration {duration:.3f}s < {min_segment_duration}s, "
                f"extending end"
            )
            end = start + min_segment_duration

        # Cap to video duration if provided
        if video_duration and video_duration > 0 and end > video_duration:
            # Only cap if end exceeds by more than epsilon (avoid floating-point noise)
            if end - video_duration > 0.01:
                logger.debug(
                    f"Segment {seg.index}: end {end:.2f}s > video {video_duration:.2f}s, capping"
                )
            end = video_duration

        # Create new segment if changes were made
        if (start != seg.start_time or end != seg.end_time):
            normalized.append(CaptionSegment(
                index=seg.index,
                start_time=start,
                end_time=end,
                text=seg.text,
                source_file=seg.source_file,
            ))
        else:
            normalized.append(seg)

    return normalized


def parse_vtt(content: str, video_id: str) -> ParseResult:
    """Parse VTT (WebVTT) format captions with segment-level error recovery.

    VTT format:
        WEBVTT

        00:00:01.000 --> 00:00:04.000
        Hello, world!

        00:00:05.000 --> 00:00:08.000
        This is a test.

    Error Recovery (US-001 Sprint 7):
        Individual segment parsing errors are caught and logged, allowing
        the parser to continue with remaining segments. Skipped segments
        are recorded with their index and error reason.

    Args:
        content: VTT file content as string.
        video_id: Video ID for source_file field.

    Returns:
        ParseResult with segments and any skipped segment info.
    """
    segments = []
    skipped_segments = []
    lines = content.split('\n')

    # Skip header
    i = 0
    while i < len(lines) and '-->' not in lines[i]:
        i += 1

    # VTT timestamp pattern: HH:MM:SS.mmm or MM:SS.mmm
    timestamp_pattern = re.compile(
        r'(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}:)?(\d{2}):(\d{2})[.,](\d{3})'
    )

    current_index = 0
    segment_count = 0  # Track total segments attempted
    while i < len(lines):
        line = lines[i].strip()

        match = timestamp_pattern.search(line)
        if match:
            segment_count += 1
            try:
                # Parse timestamps
                start_time = parse_timestamp(match.group(0).split('-->')[0].strip())
                end_time = parse_timestamp(match.group(0).split('-->')[1].strip())

                if start_time is None or end_time is None:
                    raise ValueError(f"Invalid timestamp at line {i + 1}")

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
                # Empty text is not an error, just skip silently
            except Exception as e:
                reason = f"Parse error: {str(e)}"
                skipped_segments.append((segment_count - 1, reason))
                logger.debug(f"[{video_id}] Skipped VTT segment {segment_count - 1}: {reason}")
                # Advance to next timestamp or empty line to continue parsing
                while i < len(lines) and lines[i].strip() and not timestamp_pattern.search(lines[i]):
                    i += 1
        else:
            i += 1

    if skipped_segments:
        logger.info(
            f"[{video_id}] VTT parse: {len(segments)} segments parsed, "
            f"{len(skipped_segments)} skipped"
        )

    return ParseResult(
        segments=segments,
        skipped_segments=skipped_segments,
        total_attempted=segment_count
    )


def parse_srt(content: str, video_id: str) -> ParseResult:
    """Parse SRT (SubRip) format captions with segment-level error recovery.

    SRT format:
        1
        00:00:01,000 --> 00:00:04,000
        Hello, world!

        2
        00:00:05,000 --> 00:00:08,000
        This is a test.

    Error Recovery (US-001 Sprint 7):
        Individual block parsing errors are caught and logged, allowing
        the parser to continue with remaining blocks. Skipped blocks
        are recorded with their index and error reason.

    Args:
        content: SRT file content as string.
        video_id: Video ID for source_file field.

    Returns:
        ParseResult with segments and any skipped segment info.
    """
    segments = []
    skipped_segments = []
    blocks = re.split(r'\n\s*\n', content.strip())

    for block_idx, block in enumerate(blocks):
        try:
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
                raise ValueError(f"Invalid timestamp format: {timestamp_line}")

            start_time = parse_timestamp(parts[0].strip())
            end_time = parse_timestamp(parts[1].strip())

            if start_time is None or end_time is None:
                raise ValueError(f"Could not parse timestamps: {timestamp_line}")

            # Get text
            text = ' '.join(lines[text_start_idx:]).strip()
            text = re.sub(r'<[^>]+>', '', text)  # Remove tags

            if text:
                segments.append(CaptionSegment(
                    index=len(segments),
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    source_file=video_id
                ))
            # Empty text after timestamp is not an error, just skip
        except Exception as e:
            reason = f"Parse error: {str(e)}"
            skipped_segments.append((block_idx, reason))
            logger.debug(f"[{video_id}] Skipped SRT block {block_idx}: {reason}")
            continue

    if skipped_segments:
        logger.info(
            f"[{video_id}] SRT parse: {len(segments)} segments parsed, "
            f"{len(skipped_segments)} skipped"
        )

    return ParseResult(
        segments=segments,
        skipped_segments=skipped_segments,
        total_attempted=len(blocks)
    )


def parse_json3(content: str, video_id: str) -> ParseResult:
    """Parse JSON3/SRV3 format captions from YouTube with error recovery.

    JSON3 format has 'events' array with 'segs' containing text segments.

    Error Recovery (US-001 Sprint 7):
        Events with missing required fields (tStartMs, segs) are skipped
        and logged. Other events continue to be processed.

    Args:
        content: JSON3 file content as string.
        video_id: Video ID for source_file field.

    Returns:
        ParseResult with segments and any skipped event info.
    """
    skipped_segments = []

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"[{video_id}] Failed to parse JSON3 caption format: {e}")
        return ParseResult(
            segments=[],
            skipped_segments=[(0, f"JSON decode error: {str(e)}")],
            total_attempted=1
        )

    segments = []
    events = data.get('events', [])

    for event_idx, event in enumerate(events):
        try:
            # Check for required fields per acceptance criteria
            if 'segs' not in event:
                skipped_segments.append((event_idx, "Missing 'segs' field"))
                logger.debug(f"[{video_id}] Skipped JSON3 event {event_idx}: Missing 'segs' field")
                continue

            if 'tStartMs' not in event:
                skipped_segments.append((event_idx, "Missing 'tStartMs' field"))
                logger.debug(f"[{video_id}] Skipped JSON3 event {event_idx}: Missing 'tStartMs' field")
                continue

            start_ms = event['tStartMs']
            duration_ms = event.get('dDurationMs', 0)

            # Validate numeric types
            if not isinstance(start_ms, (int, float)):
                raise ValueError(f"tStartMs is not numeric: {type(start_ms)}")

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
            # Empty text events are not errors, just skip
        except Exception as e:
            reason = f"Parse error: {str(e)}"
            skipped_segments.append((event_idx, reason))
            logger.debug(f"[{video_id}] Skipped JSON3 event {event_idx}: {reason}")
            continue

    if skipped_segments:
        logger.info(
            f"[{video_id}] JSON3 parse: {len(segments)} segments parsed, "
            f"{len(skipped_segments)} skipped"
        )

    return ParseResult(
        segments=segments,
        skipped_segments=skipped_segments,
        total_attempted=len(events) if events else 1
    )


def parse_caption_content(content: str, format_suffix: str, video_id: str) -> ParseResult:
    """Parse caption content based on format suffix.

    Args:
        content: Caption file content as string.
        format_suffix: File format suffix (e.g., '.vtt', '.srt', '.json3').
        video_id: Video ID for source_file field.

    Returns:
        ParseResult with parsed segments.
    """
    suffix = format_suffix.lower()

    if suffix in ['.vtt', '.webvtt']:
        return parse_vtt(content, video_id)
    elif suffix in ['.srt']:
        return parse_srt(content, video_id)
    elif suffix in ['.json3', '.srv3', '.json']:
        return parse_json3(content, video_id)
    else:
        logger.warning(f"Unknown subtitle format: {suffix}, trying VTT parser")
        return parse_vtt(content, video_id)


def detect_caption_format(content: str) -> str:
    """Detect caption format from content signature (US-100-012).

    Analyzes the content to determine the caption format without relying
    on file extension. This is useful when yt-dlp returns a different
    format than requested or when detecting from raw content.

    Args:
        content: Caption file content as string.

    Returns:
        Detected format: 'json3', 'vtt', 'srt', 'xml', or 'unknown'.
    """
    if not content:
        return "unknown"

    # Strip leading whitespace for detection
    content_stripped = content.strip()

    # JSON3/SRV3 detection: starts with '{' and contains 'events' or 'timedtext'
    if content_stripped.startswith('{'):
        try:
            import json
            data = json.loads(content_stripped)
            if 'events' in data or 'timedtext' in data:
                return "json3"
            # Check if it's a YouTube-specific format
            if 'player_response' in data:
                # Might be a player response with captions
                return "unknown"
        except json.JSONDecodeError:
            pass

    # VTT detection: starts with 'WEBVTT'
    if content_stripped.startswith('WEBVTT'):
        return "vtt"

    # TTML/XML detection: starts with '<?xml' or '<tt' or '<tsp:span'
    if content_stripped.startswith('<?xml') or content_stripped.startswith('<tt') or '<tt ' in content_stripped[:100]:
        return "xml"

    # SRT detection: starts with a number (segment index)
    # SRT segments start with a numeric index like "1", "2", etc.
    if content_stripped and content_stripped[0].isdigit():
        # Check for typical SRT pattern: number followed by timestamp
        lines = content_stripped.split('\n')
        if lines:
            first_line = lines[0].strip()
            if first_line.isdigit() or (first_line and first_line[0].isdigit()):
                # Likely SRT format
                return "srt"

    return "unknown"


def auto_select_parser(content: str, preferred_formats: list[str] = None) -> tuple[str, str]:
    """Automatically select the best parser based on content and preferences (US-100-012).

    This function:
    1. Detects the format from content signature
    2. Falls back to preferred formats if detection is uncertain
    3. Returns (format_detected, parser_name) tuple

    Args:
        content: Caption file content as string.
        preferred_formats: List of preferred formats in priority order.
            Defaults to ["json3", "vtt", "srt"].

    Returns:
        Tuple of (detected_format, parser_name):
        - detected_format: The format detected from content or inferred from preferences
        - parser_name: Name of the parser to use ('parse_json3', 'parse_vtt', 'parse_srt')
    """
    if preferred_formats is None:
        preferred_formats = ["json3", "vtt", "srt"]

    # First, try to detect from content signature
    detected = detect_caption_format(content)

    if detected != "unknown":
        parser_map = {
            "json3": "parse_json3",
            "vtt": "parse_vtt",
            "srt": "parse_srt",
            "xml": "parse_vtt",  # TTML/XML falls back to VTT parser
        }
        parser = parser_map.get(detected, "parse_vtt")
        logger.debug(f"Format auto-detected: {detected}, using parser: {parser}")
        return detected, parser

    # Fall back to preferred format order
    if preferred_formats:
        # Use the first available preferred format
        primary = preferred_formats[0]
        parser_map = {
            "json3": "parse_json3",
            "vtt": "parse_vtt",
            "srt": "parse_srt",
            "srv3": "parse_json3",
        }
        parser = parser_map.get(primary, "parse_vtt")
        logger.debug(
            f"Format detection uncertain, using preferred format: {primary}, "
            f"parser: {parser}"
        )
        return primary, parser

    # Ultimate fallback to VTT
    logger.debug("No format detected or preferred, defaulting to VTT parser")
    return "vtt", "parse_vtt"
