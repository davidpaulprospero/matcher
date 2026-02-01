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

    Supports formats:
        HH:MM:SS,mmm (SRT)
        HH:MM:SS.mmm (VTT)
        MM:SS.mmm (VTT short)
        HH:MM:SS (no milliseconds)

    Args:
        ts: Timestamp string to parse.

    Returns:
        Time in seconds, or None if parsing fails.
    """
    ts = ts.strip()

    # Replace comma with period for SRT format
    ts = ts.replace(',', '.')

    # Pattern for HH:MM:SS.mmm or MM:SS.mmm
    match = re.match(r'^(?:(\d+):)?(\d+):(\d+)\.(\d+)$', ts)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int(match.group(4).ljust(3, '0')[:3])  # Ensure 3 digits

        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0

    # Try simpler pattern without milliseconds
    match = re.match(r'^(?:(\d+):)?(\d+):(\d+)$', ts)
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    logger.warning(f"Could not parse timestamp: {ts}")
    return None


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
