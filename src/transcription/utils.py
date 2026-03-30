"""
Utility functions for transcription.

Provides helpers for:
- Audio extraction from videos
- Subtitle file generation (SRT format)
- Helper functions for transcription processing
"""

import subprocess
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from ..downloader.utils import SUBPROCESS_FLAGS

logger = logging.getLogger(__name__)


def is_trimmed_voiceover(path: str) -> bool:
    """Check if a voiceover path refers to a trimmed variant."""
    return '_trimmed' in Path(path).stem


def extract_audio(video_path: str, output_dir: str = None, timeout: int = 60) -> Optional[str]:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Directory for output audio file (default: same as video)
        timeout: Maximum time in seconds for extraction (default: 60).
                 If exceeded, FFmpeg subprocess is killed and TimeoutError is raised.

    Returns:
        Path to extracted audio file, or None if extraction fails

    Raises:
        TimeoutError: If audio extraction exceeds the timeout limit
    """
    video_path = Path(video_path)

    # Use hash of full path to avoid collisions with similar filenames
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    audio_filename = f"{video_path.stem[:80]}_{path_hash}.wav"

    if output_dir:
        audio_path = Path(output_dir) / audio_filename
    else:
        audio_path = video_path.parent / audio_filename

    # Skip if already extracted
    if audio_path.exists():
        return str(audio_path)

    try:
        cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', str(audio_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            **SUBPROCESS_FLAGS
        )

        if result.returncode == 0 and audio_path.exists():
            return str(audio_path)
        else:
            logger.debug(f"FFmpeg error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Audio extraction timed out after {timeout}s: {video_path}")
        # Clean up partial file if it exists
        if audio_path.exists():
            try:
                audio_path.unlink()
            except OSError:
                pass
        raise TimeoutError(f"Audio extraction timed out after {timeout} seconds: {video_path}")

    except Exception as e:
        logger.debug(f"Audio extraction error: {e}")
        return None


def normalize_segments_contiguous(
    segments: List[dict],
    *,
    start_at_zero: bool = True
) -> List[dict]:
    """
    Normalize segment timing to be strictly contiguous (no gaps/overlaps).

    Preserves each segment's duration while chaining segment starts to the
    previous segment's end. Useful for voiceover timelines where pauses should
    not produce timeline holes.

    Args:
        segments: List of segment dicts with 'start'/'end' timing fields.
        start_at_zero: When True, first segment starts at 0.0.

    Returns:
        New list of segment dicts with normalized contiguous timing.
    """
    if not segments:
        return []

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    normalized = []
    cursor = 0.0 if start_at_zero else None

    for seg in segments:
        start = _to_float(seg.get('start', 0.0), 0.0)
        end = _to_float(seg.get('end', start), start)
        if end < start:
            end = start
        duration = max(0.0, end - start)

        if cursor is None:
            new_start = max(0.0, start)
        else:
            new_start = cursor
        new_end = new_start + duration

        new_seg = dict(seg)
        new_seg['start'] = new_start
        new_seg['end'] = new_end
        normalized.append(new_seg)

        cursor = new_end

    # Keep original ordering/indexing semantics; only timing is normalized.
    return normalized


def anchor_segment_starts_to_words(segments: List[dict]) -> List[dict]:
    """Anchor each segment's start/end to its word-level timestamps.

    Whisper sometimes sets segment-level start times slightly before the first
    word's actual onset.  This ensures segment boundaries match real speech
    onset/offset for maximum accuracy.

    When a segment has no ``words`` list the original timing is kept as-is.

    Args:
        segments: List of segment dicts, optionally containing a ``words`` list
                  where each word dict has ``start`` and ``end`` float keys.

    Returns:
        New list of segment dicts with start/end anchored to word timing.
    """
    if not segments:
        return []

    anchored = []
    for seg in segments:
        new_seg = dict(seg)
        words = seg.get('words', [])
        if words:
            first_start = words[0].get('start')
            if first_start is not None:
                new_seg['start'] = float(first_start)
            last_end = words[-1].get('end')
            if last_end is not None:
                new_seg['end'] = float(last_end)
        anchored.append(new_seg)

    return anchored


def normalize_segments_start_anchored(
    segments: List[dict],
    *,
    audio_duration: float = None
) -> List[dict]:
    """Make segments contiguous while preserving original start times.

    Unlike :func:`normalize_segments_contiguous` which chains segments by
    *duration* (destroying start-time accuracy over long files), this function
    preserves each segment's original start time and makes timing contiguous by
    setting ``end[i] = start[i+1]``.

    The last segment keeps its original end, or extends to *audio_duration* if
    provided.

    Use this for voiceover SRT generation where start-time accuracy is critical
    and gaps should not exist (for clean OTIO export).

    Args:
        segments: List of segment dicts with ``start``/``end`` timing fields.
        audio_duration: Optional total audio duration — used to extend the last
                        segment so the timeline covers the full file.

    Returns:
        New list of segment dicts with start-anchored contiguous timing.
    """
    if not segments:
        return []

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    normalized = []
    for i, seg in enumerate(segments):
        new_seg = dict(seg)
        start = _to_float(seg.get('start', 0.0))
        end = _to_float(seg.get('end', start))

        new_seg['start'] = start

        if i < len(segments) - 1:
            next_start = _to_float(segments[i + 1].get('start', end))
            new_seg['end'] = next_start
        else:
            # Last segment: extend to audio duration if available and longer
            if audio_duration is not None and audio_duration > end:
                new_seg['end'] = audio_duration
            else:
                new_seg['end'] = end

        normalized.append(new_seg)

    return normalized


def compress_segment_gaps(
    segments: List[dict],
    *,
    target_duration: float = None
) -> List[dict]:
    """
    Compress gaps between segments while preserving total timeline duration.

    Unlike normalize_segments_contiguous which reduces total duration, this function
    stretches segments proportionally to fill the gaps, maintaining the original
    total timeline duration.

    Args:
        segments: List of segment dicts with 'start'/'end' timing fields.
        target_duration: Optional target duration. If not provided, uses the
            last segment's end time from original timing.

    Returns:
        New list of segment dicts with gaps compressed but total duration preserved.
    """
    if not segments:
        return []

    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # Calculate original timing
    original_starts = []
    original_ends = []
    durations = []

    for seg in segments:
        start = _to_float(seg.get('start', 0.0), 0.0)
        end = _to_float(seg.get('end', start), start)
        duration = max(0.0, end - start)
        original_starts.append(start)
        original_ends.append(end)
        durations.append(duration)

    # Calculate total gap time
    total_gap = 0.0
    for i in range(len(segments) - 1):
        gap = original_starts[i + 1] - original_ends[i]
        if gap > 0:
            total_gap += gap

    if total_gap == 0:
        # No gaps to compress
        return segments

    # Calculate target duration
    if target_duration is None:
        target_duration = original_ends[-1]

    # Calculate stretch factor
    total_duration = sum(durations)
    # New duration after removing gaps = total_duration (segments only)
    # We want to stretch to fill target_duration
    stretch_factor = target_duration / total_duration

    # Build new segments with gaps compressed
    normalized = []
    cursor = 0.0

    for i, seg in enumerate(segments):
        duration = durations[i]
        # Stretch the duration
        new_duration = duration * stretch_factor
        new_start = cursor
        new_end = cursor + new_duration

        new_seg = dict(seg)
        new_seg['start'] = new_start
        new_seg['end'] = new_end
        normalized.append(new_seg)

        cursor = new_end

    return normalized


def remap_segments_to_original_time(
    segments: List[dict],
    speech_regions_ms: List[Tuple[int, int]],
    crossfade_ms: int = 50,
) -> List[dict]:
    """Remap segment timestamps from trimmed-audio time to original-audio time.

    When silence removal concatenates speech regions (with optional crossfade),
    Whisper transcribes against the shorter trimmed audio. This function maps
    those timestamps back to the original timeline using the speech region
    boundaries as anchors.

    Args:
        segments: List of segment dicts with 'start'/'end' in seconds
            (from Whisper, in trimmed-audio time).
        speech_regions_ms: Merged (start_ms, end_ms) pairs in original timeline.
        crossfade_ms: Crossfade applied between regions during concatenation.

    Returns:
        New list of segment dicts with remapped 'start'/'end' times.
    """
    if not segments or not speech_regions_ms:
        return segments

    # Build lookup table: for each region, compute its start position in
    # the trimmed audio and its length.
    # Region k starts at sum(L_0..L_{k-1}) - k * crossfade in trimmed time.
    regions = []  # (trimmed_start_ms, trimmed_end_ms, orig_start_ms, orig_end_ms)
    trimmed_cursor_ms = 0

    for i, (orig_start, orig_end) in enumerate(speech_regions_ms):
        region_len = orig_end - orig_start
        trimmed_start = trimmed_cursor_ms
        trimmed_end = trimmed_start + region_len
        regions.append((trimmed_start, trimmed_end, orig_start, orig_end))
        # Next region starts after this one, minus crossfade overlap
        # (first region has no preceding crossfade)
        trimmed_cursor_ms = trimmed_end - crossfade_ms if i < len(speech_regions_ms) - 1 else trimmed_end

    def _remap_time(t_sec: float) -> float:
        """Map a single timestamp from trimmed time to original time."""
        t_ms = t_sec * 1000.0

        for trimmed_start, trimmed_end, orig_start, orig_end in regions:
            if t_ms <= trimmed_end:
                # Clamp to region start
                offset = max(0.0, t_ms - trimmed_start)
                return (orig_start + offset) / 1000.0

        # Past all regions — clamp to end of last region
        if regions:
            return regions[-1][3] / 1000.0
        return t_sec

    remapped = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg['start'] = _remap_time(seg.get('start', 0.0))
        new_seg['end'] = _remap_time(seg.get('end', 0.0))
        remapped.append(new_seg)

    return remapped


def write_srt(
    segments: List[dict],
    srt_path: str,
    *,
    force_contiguous_timing: bool = False
):
    """
    Write segments to SRT subtitle format.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        srt_path: Path for output SRT file
        force_contiguous_timing: When True, rewrite timings back-to-back with
            no inter-segment gaps/overlaps while preserving each segment duration.
    """
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    output_segments = (
        normalize_segments_contiguous(segments)
        if force_contiguous_timing else segments
    )

    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(output_segments, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()

            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def extract_video_id(filename: str) -> Optional[str]:
    """
    Extract YouTube video ID from filename.

    Handles formats:
    - video_id.mp4
    - video_id.mp3
    - video_id_0045.mp4 (segments)

    YouTube IDs are 11 characters: [A-Za-z0-9_-]

    Args:
        filename: Video filename

    Returns:
        Video ID string, or None if not found
    """
    import re
    stem = Path(filename).stem

    # Pattern 1: Segment file like "abc12345678_0045" -> extract "abc12345678"
    segment_match = re.match(r'^([A-Za-z0-9_-]{11})_\d+$', stem)
    if segment_match:
        return segment_match.group(1)

    # Pattern 2: Regular file with 11-char ID at start
    if len(stem) >= 11:
        potential_id = stem[:11]
        if re.match(r'^[A-Za-z0-9_-]{11}$', potential_id):
            return potential_id

    # Pattern 3: Try to find 11-char sequence anywhere (less reliable)
    match = re.search(r'[A-Za-z0-9_-]{11}', stem)
    if match:
        return match.group(0)

    return None


def format_timestamp_srt(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_audio_duration(audio_path: str, timeout: int = 10) -> Optional[float]:
    """
    Get audio file duration using ffprobe.

    Args:
        audio_path: Path to audio file
        timeout: Maximum time in seconds for ffprobe (default: 10)

    Returns:
        Duration in seconds, or None if unable to determine
    """
    import json as json_module

    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(audio_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            **SUBPROCESS_FLAGS
        )

        if result.returncode == 0:
            data = json_module.loads(result.stdout)
            duration_str = data.get('format', {}).get('duration')
            if duration_str:
                return float(duration_str)

    except (subprocess.TimeoutExpired, json_module.JSONDecodeError, ValueError, KeyError) as e:
        logger.debug(f"Could not get audio duration for {audio_path}: {e}")

    return None


def merge_segments_by_speaker(
    segments: List[dict],
    max_gap_seconds: float = 0.5,
    min_duration_seconds: float = 1.0
) -> List[dict]:
    """Merge segments with same speaker label (US-124-011).

    Looks for speaker labels in segment data and merges consecutive segments
    from the same speaker if the gap between them is small enough.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys.
                  Optional: 'speaker' key for speaker labels.
        max_gap_seconds: Maximum gap between segments to consider merging.
        min_duration_seconds: Minimum duration for merged segment.

    Returns:
        List of merged segment dicts.
    """
    if not segments:
        return []

    # Check if any segments have speaker labels
    has_speakers = any(seg.get('speaker') for seg in segments)
    if not has_speakers:
        return segments

    merged = []
    current = None

    for seg in segments:
        speaker = seg.get('speaker')
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        text = seg.get('text', '')

        if current is None:
            # Start first segment
            current = {
                'start': seg_start,
                'end': seg_end,
                'text': text,
                'speaker': speaker
            }
            continue

        # Check if we can merge with current segment
        gap = seg_start - current['end']
        same_speaker = speaker == current.get('speaker')

        if same_speaker and gap <= max_gap_seconds:
            # Merge: extend end time and append text
            current['end'] = seg_end
            current['text'] = current['text'] + ' ' + text
        else:
            # Can't merge - check if current meets minimum duration
            duration = current['end'] - current['start']
            if duration >= min_duration_seconds:
                merged.append(current)
            else:
                # Keep current but merge with next if possible
                current['end'] = seg_end
                current['text'] = current['text'] + ' ' + text
                current['speaker'] = speaker
                continue

            # Start new segment
            current = {
                'start': seg_start,
                'end': seg_end,
                'text': text,
                'speaker': speaker
            }

    # Don't forget the last segment
    if current is not None:
        duration = current['end'] - current['start']
        if duration >= min_duration_seconds:
            merged.append(current)
        elif merged:
            # Merge small tail into previous
            merged[-1]['text'] = merged[-1]['text'] + ' ' + current['text']
            merged[-1]['end'] = current['end']

    return merged


def detect_speaker_changes(
    segments: List[dict],
    vad_segments: List[dict] = None,
    min_gap_seconds: float = 1.5,
    confidence_threshold: float = 0.7
) -> List[dict]:
    """Detect speaker changes based on VAD segments and segment patterns (US-137-008).

    Analyzes gaps between segments and VAD boundaries to identify likely speaker
    transitions. Returns a list of speaker change points with confidence scores.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys.
        vad_segments: Optional list of VAD segment dicts with 'start', 'end' keys.
                     If not provided, uses gaps between transcription segments.
        min_gap_seconds: Minimum gap to consider as potential speaker change.
                        Longer gaps suggest different speaker.
        confidence_threshold: Minimum confidence to include a speaker change.

    Returns:
        List of dicts with 'timestamp', 'confidence', 'reason' keys representing
        detected speaker changes. Timestamps are where speaker changes occur.
    """
    if not segments or len(segments) < 2:
        return []

    speaker_changes = []

    # Strategy 1: Use VAD segments if available
    if vad_segments and len(vad_segments) >= 2:
        speaker_changes = _detect_from_vad_segments(
            vad_segments, min_gap_seconds, confidence_threshold
        )
    else:
        # Strategy 2: Fall back to analyzing gaps between transcription segments
        speaker_changes = _detect_from_segment_gaps(
            segments, min_gap_seconds, confidence_threshold
        )

    return speaker_changes


def _detect_from_vad_segments(
    vad_segments: List[dict],
    min_gap_seconds: float,
    confidence_threshold: float
) -> List[dict]:
    """Detect speaker changes from VAD segments.

    Analyzes the pattern of VAD segments - long gaps between speech segments
    often indicate speaker changes.
    """
    speaker_changes = []

    for i in range(1, len(vad_segments)):
        prev_end = vad_segments[i - 1].get('end', 0)
        curr_start = vad_segments[i].get('start', 0)
        gap = curr_start - prev_end

        if gap >= min_gap_seconds:
            # Calculate confidence based on gap duration
            # Longer gaps = higher confidence of speaker change
            confidence = min(1.0, gap / 5.0)  # 5+ second gap = 100% confidence

            if confidence >= confidence_threshold:
                speaker_changes.append({
                    'timestamp': curr_start,
                    'confidence': confidence,
                    'reason': 'long_vad_gap',
                    'gap_duration': gap
                })

    return speaker_changes


def _detect_from_segment_gaps(
    segments: List[dict],
    min_gap_seconds: float,
    confidence_threshold: float
) -> List[dict]:
    """Detect speaker changes from transcription segment gaps.

    Analyzes gaps between transcription segments to identify speaker transitions.
    """
    speaker_changes = []

    for i in range(1, len(segments)):
        prev_end = segments[i - 1].get('end', 0)
        curr_start = segments[i].get('start', 0)
        gap = curr_start - prev_end

        # Also consider segment duration - very short segments at boundaries
        # might indicate interruption/overlap
        prev_duration = prev_end - segments[i - 1].get('start', 0)
        curr_duration = segments[i].get('end', 0) - curr_start

        if gap >= min_gap_seconds:
            # Base confidence on gap duration
            confidence = min(1.0, gap / 5.0)

            # Boost confidence if surrounding segments are substantial
            # (not just short fragments that might be artifacts)
            if prev_duration > 2.0 and curr_duration > 2.0:
                confidence = min(1.0, confidence + 0.1)

            # Reduce confidence for very short segments (might be artifacts)
            if prev_duration < 1.0 or curr_duration < 1.0:
                confidence = confidence * 0.7

            if confidence >= confidence_threshold:
                speaker_changes.append({
                    'timestamp': curr_start,
                    'confidence': confidence,
                    'reason': 'segment_gap',
                    'gap_duration': gap
                })

    return speaker_changes


def merge_segments_by_detected_speakers(
    segments: List[dict],
    speaker_changes: List[dict],
    max_gap_seconds: float = 0.5,
    min_duration_seconds: float = 1.0
) -> List[dict]:
    """Merge segments by detected speaker changes (US-137-008).

    Uses speaker change timestamps to group segments into speaker turns,
    merging segments within the same speaker's turn.

    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys.
        speaker_changes: List of speaker change dicts from detect_speaker_changes().
        max_gap_seconds: Maximum gap within same speaker turn.
        min_duration_seconds: Minimum duration for merged segment.

    Returns:
        List of merged segment dicts.
    """
    if not segments:
        return []

    # Extract speaker change timestamps
    change_timestamps = {sc['timestamp'] for sc in speaker_changes}

    merged = []
    current = None

    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        text = seg.get('text', '')

        if current is None:
            current = {
                'start': seg_start,
                'end': seg_end,
                'text': text,
                'speaker_turn': 0
            }
            continue

        # Check if this segment starts at a detected speaker change
        is_speaker_change = seg_start in change_timestamps

        # Check gap to current segment
        gap = seg_start - current['end']
        same_turn = gap <= max_gap_seconds and not is_speaker_change

        if same_turn:
            # Merge into current speaker turn
            current['end'] = seg_end
            current['text'] = current['text'] + ' ' + text
        else:
            # End current turn, start new one
            duration = current['end'] - current['start']
            if duration >= min_duration_seconds:
                merged.append(current)
            else:
                # Merge small segment into previous
                if merged:
                    merged[-1]['text'] = merged[-1]['text'] + ' ' + current['text']
                    merged[-1]['end'] = current['end']

            current = {
                'start': seg_start,
                'end': seg_end,
                'text': text,
                'speaker_turn': len(merged)
            }

    # Handle last segment
    if current is not None:
        duration = current['end'] - current['start']
        if duration >= min_duration_seconds:
            merged.append(current)
        elif merged:
            merged[-1]['text'] = merged[-1]['text'] + ' ' + current['text']
            merged[-1]['end'] = current['end']

    return merged


def split_segment_at_punctuation(
    segment: dict,
    punctuation: str = ".!?",
    abbreviations: List[str] = None,
    min_after_text: int = 2,
    min_duration: float = 0.5
) -> List[dict]:
    """Split a segment at natural language boundaries (punctuation) (US-124-011).

    Splits a segment at sentence-ending punctuation while avoiding
    splitting at abbreviations like "Mr.", "Dr.", "e.g.", etc.

    Args:
        segment: Segment dict with 'start', 'end', 'text' keys.
        punctuation: Punctuation marks that trigger splits.
        abbreviations: List of abbreviations to ignore (case-insensitive).
        min_after_text: Minimum text length after punctuation to create new segment.
        min_duration: Minimum duration for resulting segments.

    Returns:
        List of split segment dicts (usually 1-2 segments).
    """
    import re

    if abbreviations is None:
        abbreviations = [
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
            "vs", "etc", "eg", "ie", "al",
            "us", "usa", "uk", "eu", "un", "nato",
        ]

    text = segment.get('text', '')
    start = segment.get('start', 0)
    end = segment.get('end', 0)
    duration = end - start

    if duration < min_duration * 2:  # Too short to split
        return [segment]

    # Build pattern that matches punctuation but excludes abbreviations
    # Pattern: punctuation followed by space and word characters
    # But NOT if preceded by abbreviation pattern

    # Create abbreviation pattern (case-insensitive)
    abbrev_pattern = r'\b(' + '|'.join(re.escape(a) for a in abbreviations) + r')\.'
    abbrev_regex = re.compile(abbrev_pattern, re.IGNORECASE)

    # Find all punctuation marks that aren't part of abbreviations
    splits = []
    last_end = 0

    for i, char in enumerate(text):
        if char in punctuation:
            # Check if this punctuation is part of an abbreviation
            # Look at text before this punctuation
            before = text[max(0, i-5):i].lower().strip()

            # Check if followed by space and text
            after = text[i+1:min(len(text), i+1+10)].lstrip()

            if len(after) < min_after_text:
                continue  # Not enough text after punctuation

            # Check if preceded by abbreviation
            if abbrev_regex.search(before + '.'):
                continue  # Skip abbreviation

            # Valid split point
            split_text = text[last_end:i+1].strip()
            if split_text:
                # Calculate proportional time for this portion
                portion = (i + 1) / len(text)
                split_end = start + duration * portion

                splits.append({
                    'start': start if not splits else splits[-1]['end'],
                    'end': split_end,
                    'text': split_text
                })
                last_end = i + 1

    # Add remaining text
    if last_end < len(text):
        remaining_text = text[last_end:].strip()
        if remaining_text:
            remaining_start = splits[-1]['end'] if splits else start
            splits.append({
                'start': remaining_start,
                'end': end,
                'text': remaining_text
            })

    # Filter by minimum duration
    result = []
    for seg in splits:
        seg_duration = seg['end'] - seg['start']
        if seg_duration >= min_duration:
            result.append(seg)

    # If no valid splits, return original
    if not result:
        return [segment]

    return result


def split_segment_word_aware(
    segment: dict,
    punctuation: str = ".!?",
    abbreviations: List[str] = None,
    min_after_text: int = 2,
    min_duration: float = 0.5
) -> List[dict]:
    """Split a segment at punctuation using word-level timestamps when available.

    When the segment contains a ``words`` list with per-word ``start``/``end``
    timing, uses those timestamps for precise split boundaries instead of the
    character-proportion estimation used by :func:`split_segment_at_punctuation`.

    Falls back to :func:`split_segment_at_punctuation` when word data is absent.

    Args:
        segment: Segment dict with ``start``, ``end``, ``text``, and optionally
                 ``words`` (list of dicts with ``word``, ``start``, ``end``).
        punctuation: Characters that trigger a split.
        abbreviations: Abbreviation stems to skip (case-insensitive).
        min_after_text: Minimum remaining text length to create a new segment.
        min_duration: Minimum duration for resulting segments.

    Returns:
        List of segment dicts with word-level accurate start times.
    """
    import re

    words = segment.get('words', [])
    text = segment.get('text', '')
    start = float(segment.get('start', 0))
    end = float(segment.get('end', 0))

    if not text or end - start < min_duration * 2:
        return [segment]

    if abbreviations is None:
        abbreviations = [
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
            "vs", "etc", "eg", "ie", "al",
            "us", "usa", "uk", "eu", "un", "nato",
        ]

    # Fall back to proportional split when no word timestamps
    if not words:
        return split_segment_at_punctuation(
            segment, punctuation, abbreviations, min_after_text, min_duration
        )

    abbrev_set = {a.lower() for a in abbreviations}
    splits: List[dict] = []
    current_words: List[dict] = []

    for wi, word_info in enumerate(words):
        word_text = word_info.get('word', '').strip()
        current_words.append(word_info)

        if not word_text or word_text[-1] not in punctuation:
            continue

        # Skip abbreviations
        bare = re.sub(r'[.!?]+$', '', word_text).lower()
        if bare in abbrev_set:
            continue

        # Ensure enough text remains after this split
        remaining_words = words[wi + 1:]
        remaining_text = ' '.join(w.get('word', '').strip() for w in remaining_words)
        if len(remaining_text.strip()) < min_after_text:
            continue

        # Build sub-segment with word-level timing
        seg_start = float(current_words[0].get('start', start))
        seg_end = float(current_words[-1].get('end', end))
        seg_text = ' '.join(w.get('word', '').strip() for w in current_words)

        new_seg = {
            'start': seg_start,
            'end': seg_end,
            'text': seg_text,
            'words': list(current_words),
        }
        if 'segment_confidence' in segment:
            new_seg['segment_confidence'] = segment['segment_confidence']

        splits.append(new_seg)
        current_words = []

    # Remaining words become the final sub-segment
    if current_words:
        seg_start = float(current_words[0].get('start', start))
        seg_end = float(current_words[-1].get('end', end))
        seg_text = ' '.join(w.get('word', '').strip() for w in current_words)

        new_seg = {
            'start': seg_start,
            'end': seg_end,
            'text': seg_text,
            'words': list(current_words),
        }
        if 'segment_confidence' in segment:
            new_seg['segment_confidence'] = segment['segment_confidence']

        splits.append(new_seg)

    if not splits:
        return [segment]

    # Filter by minimum duration
    result = [s for s in splits if float(s['end']) - float(s['start']) >= min_duration]
    return result if result else [segment]


def post_process_segments(
    segments: List[dict],
    config: 'SegmentPostProcessingConfig' = None,
    vad_segments: List[dict] = None
) -> List[dict]:
    """Apply post-processing to transcription segments (US-124-011, US-137-008).

    Combines segment merging and splitting operations based on config.
    Supports speaker-aware merging using VAD segment analysis (US-137-008).

    Args:
        segments: List of segment dicts from Whisper transcription.
        config: SegmentPostProcessingConfig with processing options.
               If None, uses default config.
        vad_segments: Optional list of VAD segment dicts with 'start', 'end' keys.
                     Used for speaker-aware merging when enable_speaker_aware_merging is True.

    Returns:
        List of processed segment dicts.
    """
    from src.config.sections.core import SegmentPostProcessingConfig

    if not segments:
        return []

    if config is None:
        config = SegmentPostProcessingConfig()

    if not config.enabled:
        return segments

    result = segments

    # Step 1a: Merge segments with same speaker (legacy method with speaker labels)
    if config.merge_same_speaker and not config.enable_speaker_aware_merging:
        result = merge_segments_by_speaker(
            result,
            max_gap_seconds=config.merge_max_gap_seconds,
            min_duration_seconds=config.merge_min_duration
        )

    # Step 1b: Speaker-aware merging using VAD segment analysis (US-137-008)
    if config.enable_speaker_aware_merging:
        # Detect speaker changes from VAD segments or segment gaps
        speaker_changes = detect_speaker_changes(
            result,
            vad_segments=vad_segments,
            min_gap_seconds=config.merge_max_gap_seconds,
            confidence_threshold=config.speaker_change_confidence_threshold
        )

        # Merge segments by detected speakers
        result = merge_segments_by_detected_speakers(
            result,
            speaker_changes,
            max_gap_seconds=config.merge_max_gap_seconds,
            min_duration_seconds=config.merge_min_duration
        )
    elif config.merge_same_speaker:
        # Legacy path: merge by speaker labels if available
        result = merge_segments_by_speaker(
            result,
            max_gap_seconds=config.merge_max_gap_seconds,
            min_duration_seconds=config.merge_min_duration
        )

    # Step 2: Split at natural language boundaries (word-aware when possible)
    if config.split_at_punctuation:
        processed = []
        for seg in result:
            splits = split_segment_word_aware(
                seg,
                punctuation=config.split_punctuation,
                abbreviations=config.abbreviations,
                min_after_text=config.split_min_after_text,
                min_duration=config.split_min_duration
            )
            processed.extend(splits)
        result = processed

    # Preserve any additional segment metadata (confidence, words, etc.)
    # by copying from original segments where possible
    if result and segments:
        for i, seg in enumerate(result):
            if 'words' not in seg and i < len(segments):
                # Try to preserve word-level data
                orig = segments[i % len(segments)]
                if 'words' in orig:
                    seg['words'] = orig.get('words', [])
                if 'segment_confidence' in orig:
                    seg['segment_confidence'] = orig.get('segment_confidence')

    return result
