"""
Listicle Structure Detection in Voiceover Narration

Detects list-style structure in voiceover segments by identifying:
- Ordinal markers: 'first', 'second', 'third', 'finally', 'lastly'
- Numbered markers: '#1', 'number one', 'step 1', 'item 1'
- Transition markers: 'next up', 'moving on to', "let's talk about"

Returns ListicleGroup objects representing each detected list item.
"""

import logging
import re
from typing import List, Optional, Tuple, Any

from .models import ListicleGroup

logger = logging.getLogger(__name__)


# Ordinal words mapped to their sequence position (for ordering)
ORDINAL_WORDS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
    'firstly': 1, 'secondly': 2, 'thirdly': 3,
    'finally': -1,  # Terminal marker
    'lastly': -1,   # Terminal marker
    'last': -1,     # Terminal marker
}

# Patterns for ordinal markers at segment boundaries
# Match at start of text (with optional leading punctuation/whitespace)
ORDINAL_PATTERN = re.compile(
    r'^\s*(?:and\s+)?(?:the\s+)?(' +
    '|'.join(re.escape(w) for w in ORDINAL_WORDS) +
    r')\b',
    re.IGNORECASE
)

# Mid-segment ordinal pattern (no ^ anchor, matches anywhere)
ORDINAL_MID_PATTERN = re.compile(
    r'(?:and\s+)?(?:the\s+)?(' +
    '|'.join(re.escape(w) for w in ORDINAL_WORDS) +
    r')\b',
    re.IGNORECASE
)

# Patterns for numbered markers: #1, number 1, step 1, item 1, no. 1, etc.
NUMBERED_PATTERNS = [
    re.compile(r'^\s*#\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'^\s*(?:number|num\.?)\s+(\w+)\b', re.IGNORECASE),
    re.compile(r'^\s*(?:step|item|point|reason|tip|thing|way)\s+(\w+)\b', re.IGNORECASE),
    re.compile(r'^\s*(?:no\.?|n°)\s*(\d+)\b', re.IGNORECASE),
]

# Mid-segment numbered patterns (no ^ anchor)
NUMBERED_MID_PATTERNS = [
    re.compile(r'#\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'(?:number|num\.?)\s+(\w+)\b', re.IGNORECASE),
    re.compile(r'(?:step|item|point|reason|tip|thing|way)\s+(\w+)\b', re.IGNORECASE),
    re.compile(r'(?:no\.?|n°)\s*(\d+)\b', re.IGNORECASE),
]

# Number words for "number one", "step two", etc.
NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}

# Transition marker patterns
TRANSITION_PATTERNS = [
    re.compile(r'^\s*next\s+up\b', re.IGNORECASE),
    re.compile(r'^\s*moving\s+on\s+to\b', re.IGNORECASE),
    re.compile(r"^\s*let'?s\s+talk\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*let'?s\s+move\s+on\s+to\b", re.IGNORECASE),
    re.compile(r'^\s*now\s+(?:for|let\'?s\s+look\s+at)\b', re.IGNORECASE),
    re.compile(r'^\s*another\s+(?:thing|reason|way|tip|point)\b', re.IGNORECASE),
    re.compile(r'^\s*on\s+to\s+(?:the\s+)?(?:next|our\s+next)\b', re.IGNORECASE),
]

# Mid-segment transition patterns (no ^ anchor)
TRANSITION_MID_PATTERNS = [
    re.compile(r'next\s+up\b', re.IGNORECASE),
    re.compile(r'moving\s+on\s+to\b', re.IGNORECASE),
    re.compile(r"let'?s\s+talk\s+about\b", re.IGNORECASE),
    re.compile(r"let'?s\s+move\s+on\s+to\b", re.IGNORECASE),
    re.compile(r'now\s+(?:for|let\'?s\s+look\s+at)\b', re.IGNORECASE),
    re.compile(r'another\s+(?:thing|reason|way|tip|point)\b', re.IGNORECASE),
    re.compile(r'on\s+to\s+(?:the\s+)?(?:next|our\s+next)\b', re.IGNORECASE),
]

# Word-form numbers for header detection (e.g., "five reasons", "seven tips")
HEADER_NUMBER_WORDS = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
}

# List header nouns that follow the count
_HEADER_NOUNS = (
    r'(?:reasons?|ways?|tips?|things?|steps?|points?|facts?|'
    r'places?|ideas?|mistakes?|secrets?|signs?|tricks?|methods?|'
    r'rules?|lessons?|examples?|benefits?|attractions?|destinations?)'
)

# Header patterns: "top 10 reasons", "5 ways", "seven tips", "the 3 best things"
HEADER_PATTERNS = [
    # "top N <noun>" — numeric
    re.compile(r'\btop\s+(\d+)\s+' + _HEADER_NOUNS, re.IGNORECASE),
    # "top N <noun>" — word-form
    re.compile(
        r'\btop\s+(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+' + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
    # "N <noun>" at start — numeric (e.g., "5 reasons", "10 tips")
    re.compile(r'^\s*(\d+)\s+' + _HEADER_NOUNS, re.IGNORECASE),
    # "N <noun>" at start — word-form (e.g., "five reasons", "seven tips")
    re.compile(
        r'^\s*(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+' + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
    # "the N best/worst/most <noun>" — numeric
    re.compile(r'\bthe\s+(\d+)\s+(?:best|worst|most\s+\w+)\s+' + _HEADER_NOUNS, re.IGNORECASE),
    # "the N best/worst/most <noun>" — word-form
    re.compile(
        r'\bthe\s+(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+(?:best|worst|most\s+\w+)\s+'
        + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
]


def _extract_topic_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract simple topic keywords from segment text."""
    # Remove common stop words and short words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
        'too', 'very', 'just', 'also', 'now', 'then', 'here', 'there',
        'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'it', 'its', 'of', 'in', 'to',
        'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'up',
        'down', 'out', 'off', 'over', 'under', 'again', 'further',
        'we', 'you', 'he', 'she', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'our', 'their', 'i',
    }
    words = re.findall(r'[a-zA-Z]+', text.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    # Return unique keywords preserving order
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:max_keywords]


def _detect_ordinal(text: str, scan_full_text: bool = False) -> Optional[Tuple[str, str, int]]:
    """
    Detect ordinal marker in text.

    Args:
        text: Text to search.
        scan_full_text: If True, search anywhere in text (not just start).

    Returns (marker_type, label, char_offset) or None.
    marker_type is 'ordinal'. char_offset is the character position of the match.
    """
    match = ORDINAL_PATTERN.match(text)
    if match:
        word = match.group(1).lower()
        return ('ordinal', word, match.start(1))
    if scan_full_text:
        match = ORDINAL_MID_PATTERN.search(text)
        if match and match.start() > 0:  # Only mid-segment (not start)
            word = match.group(1).lower()
            return ('ordinal', word, match.start())
    return None


def _detect_numbered(text: str, scan_full_text: bool = False) -> Optional[Tuple[str, str, int]]:
    """
    Detect numbered marker in text.

    Args:
        text: Text to search.
        scan_full_text: If True, search anywhere in text (not just start).

    Returns (marker_type, label, char_offset) or None.
    marker_type is 'numbered'. char_offset is the character position of the match.
    """
    for pattern in NUMBERED_PATTERNS:
        match = pattern.match(text)
        if match:
            full_match = match.group(0).strip()
            return ('numbered', full_match, match.start())
    if scan_full_text:
        for pattern in NUMBERED_MID_PATTERNS:
            match = pattern.search(text)
            if match and match.start() > 0:  # Only mid-segment
                full_match = match.group(0).strip()
                return ('numbered', full_match, match.start())
    return None


def _detect_transition(text: str, scan_full_text: bool = False) -> Optional[Tuple[str, str, int]]:
    """
    Detect transition marker in text.

    Args:
        text: Text to search.
        scan_full_text: If True, search anywhere in text (not just start).

    Returns (marker_type, label, char_offset) or None.
    marker_type is 'transition'. char_offset is the character position of the match.
    """
    for pattern in TRANSITION_PATTERNS:
        match = pattern.match(text)
        if match:
            return ('transition', match.group(0).strip(), match.start())
    if scan_full_text:
        for pattern in TRANSITION_MID_PATTERNS:
            match = pattern.search(text)
            if match and match.start() > 0:  # Only mid-segment
                return ('transition', match.group(0).strip(), match.start())
    return None


def _get_segment_text(segment: Any) -> str:
    """Extract text from a segment, handling both objects and dicts."""
    if isinstance(segment, dict):
        return segment.get('text', '')
    return getattr(segment, 'text', '')


def _get_segment_index(segment: Any, position: int) -> int:
    """Extract index from a segment, falling back to position."""
    if isinstance(segment, dict):
        return segment.get('index', position)
    return getattr(segment, 'index', position)


def _parse_header_count(value: str) -> Optional[int]:
    """Parse a count from a header match — handles both digits and word-form."""
    value_lower = value.lower().strip()
    if value_lower.isdigit():
        return int(value_lower)
    return HEADER_NUMBER_WORDS.get(value_lower)


def detect_list_header(text: str) -> Optional[int]:
    """
    Detect a list header pattern in text and return the expected count.

    Recognizes patterns like:
    - "top 10 reasons" → 10
    - "5 ways to improve" → 5
    - "seven tips for success" → 7
    - "the 3 best places" → 3

    Returns the expected count or None if no header detected.
    """
    for pattern in HEADER_PATTERNS:
        match = pattern.search(text)
        if match:
            return _parse_header_count(match.group(1))
    return None


def _extract_marker_position(marker_type: str, label: str) -> Optional[int]:
    """
    Extract a numeric position from a marker label.

    For ordinal markers, uses ORDINAL_WORDS mapping (e.g., 'first' → 1).
    For numbered markers, parses digits or number words from the label.
    Terminal markers (finally, lastly, last) return -1.
    Transition markers return None (no position inferable).

    Returns:
        Positive int for positioned markers, -1 for terminal markers,
        None if no position can be inferred.
    """
    if marker_type == 'ordinal':
        return ORDINAL_WORDS.get(label.lower())

    if marker_type == 'numbered':
        # Try to find a digit in the label (e.g., '#2', 'Step 3')
        digit_match = re.search(r'(\d+)', label)
        if digit_match:
            return int(digit_match.group(1))
        # Try number words (e.g., 'Step two', 'Number one')
        for word, num in NUMBER_WORDS.items():
            if word in label.lower():
                return num

    # Transition markers have no inherent position
    return None


def _normalize_marker_sequence(groups: List['ListicleGroup']) -> List['ListicleGroup']:
    """
    Normalize mixed marker numbering to a consistent 0-based sequence.

    When markers come from mixed types (ordinals, numbered, transitions),
    their group_ids may not reflect logical order. This function:
    1. Extracts numeric positions from labels where possible
    2. Sorts groups by inferred position (positioned markers first)
    3. Appends unpositioned markers (transitions) in original order
    4. Reassigns group_id to a clean 0-based sequence

    Terminal markers (finally, lastly) are always placed last.
    Groups where no position can be inferred keep their relative order.
    """
    if not groups:
        return groups

    positioned: List[Tuple[int, int, ListicleGroup]] = []  # (position, original_idx, group)
    terminal: List[Tuple[int, ListicleGroup]] = []          # (original_idx, group)
    unpositioned: List[Tuple[int, ListicleGroup]] = []      # (original_idx, group)

    for idx, group in enumerate(groups):
        pos = _extract_marker_position(group.marker_type, group.item_label)
        if pos is None:
            unpositioned.append((idx, group))
        elif pos == -1:
            terminal.append((idx, group))
        else:
            positioned.append((pos, idx, group))

    # If no positioned markers exist, nothing to normalize
    if not positioned:
        return groups

    # Sort positioned markers by their inferred numeric position,
    # breaking ties by original detection order
    positioned.sort(key=lambda x: (x[0], x[1]))

    # Build final ordered list:
    # 1. Positioned markers in sorted order
    # 2. Unpositioned markers in original order (interleaved by original position)
    # 3. Terminal markers at the end
    result: List[ListicleGroup] = []

    # Merge positioned and unpositioned by original index to maintain
    # relative ordering when both types are present
    pos_iter = iter(positioned)
    unpos_iter = iter(unpositioned)

    current_pos = next(pos_iter, None)
    current_unpos = next(unpos_iter, None)

    while current_pos is not None or current_unpos is not None:
        if current_pos is not None and current_unpos is not None:
            # Positioned markers go in their sorted position;
            # unpositioned markers fill gaps based on original order
            # Strategy: place all positioned first, then unpositioned
            result.append(current_pos[2])
            current_pos = next(pos_iter, None)
        elif current_pos is not None:
            result.append(current_pos[2])
            current_pos = next(pos_iter, None)
        else:
            result.append(current_unpos[1])
            current_unpos = next(unpos_iter, None)

    # Append remaining unpositioned
    while current_unpos is not None:
        result.append(current_unpos[1])
        current_unpos = next(unpos_iter, None)

    # Terminal markers always go last, in original order
    terminal.sort(key=lambda x: x[0])
    for _, group in terminal:
        result.append(group)

    # Reassign group_id to clean 0-based sequence
    for new_id, group in enumerate(result):
        group.group_id = new_id

    return result


def _scan_markers(
    segments: List[Any],
    max_chars_offset: int,
    case_insensitive_numbers: bool = False,
) -> List[Tuple[int, str, str, bool, int]]:
    """
    Scan segments for listicle markers.

    Args:
        segments: List of VoiceoverSegment objects (or dicts with 'text' field).
        max_chars_offset: Maximum character offset for mid-segment detection.
        case_insensitive_numbers: If True, also match number words
            case-insensitively in mid-segment text (relaxed mode).

    Returns:
        List of (position, marker_type, label, is_mid_segment, char_offset) tuples.
    """
    markers: List[Tuple[int, str, str, bool, int]] = []

    for i, segment in enumerate(segments):
        text = _get_segment_text(segment)
        if not text:
            continue

        # First pass: start-of-text markers (priority)
        result = _detect_ordinal(text)
        if result is None:
            result = _detect_numbered(text)
        if result is None:
            result = _detect_transition(text)

        if result is not None:
            marker_type, label, char_offset = result
            markers.append((i, marker_type, label, False, char_offset))
            continue

        # Second pass: mid-segment markers within max_chars_offset
        scan_text = text[:max_chars_offset] if max_chars_offset > 0 else ''
        if not scan_text:
            continue

        result = _detect_ordinal(scan_text, scan_full_text=True)
        if result is None:
            result = _detect_numbered(scan_text, scan_full_text=True)
        if result is None:
            result = _detect_transition(scan_text, scan_full_text=True)

        if result is not None:
            marker_type, label, char_offset = result
            markers.append((i, marker_type, label, True, char_offset))

    return markers


def _build_groups_from_markers(
    markers: List[Tuple[int, str, str, bool, int]],
    segments: List[Any],
    expected_count: Optional[int],
) -> List[ListicleGroup]:
    """Build ListicleGroup objects from detected markers."""
    groups: List[ListicleGroup] = []

    for idx, (pos, marker_type, label, is_mid, char_off) in enumerate(markers):
        # Determine end of this group: next marker start - 1, or end of segments
        if idx + 1 < len(markers):
            next_pos = markers[idx + 1][0]
            next_is_mid = markers[idx + 1][3]
            # If next marker is mid-segment, this group includes that segment too
            # (the text before the marker belongs to this group)
            if next_is_mid and next_pos > pos:
                end_pos = next_pos
            elif next_pos > pos:
                end_pos = next_pos - 1
            else:
                # Same segment (consecutive mid-segment markers) or adjacent
                end_pos = pos
        else:
            end_pos = len(segments) - 1

        # Collect topic keywords from all segments in this group
        all_text = ' '.join(
            _get_segment_text(segments[j])
            for j in range(pos, min(end_pos + 1, len(segments)))
        )
        topic_keywords = _extract_topic_keywords(all_text)

        group = ListicleGroup(
            group_id=idx,
            item_label=label,
            marker_type=marker_type,
            start_segment_idx=_get_segment_index(segments[pos], pos),
            end_segment_idx=_get_segment_index(segments[end_pos], end_pos),
            topic_keywords=topic_keywords,
            expected_count=expected_count,
        )
        groups.append(group)

    return groups


def _auto_correct_markers(
    segments: List[Any],
    existing_markers: List[Tuple[int, str, str, bool, int]],
    expected_count: int,
    initial_max_chars_offset: int,
) -> List[Tuple[int, str, str, bool, int]]:
    """
    Re-scan segments with relaxed detection thresholds to find missed markers.

    Only runs once (no recursion). Relaxes by:
    - Increasing max_chars_offset to 150 (from default 50)
    - Scanning segments that didn't already have a marker

    Args:
        segments: Original segment list.
        existing_markers: Markers from the initial scan.
        expected_count: Expected number of items from list header.
        initial_max_chars_offset: The max_chars_offset used in the initial scan.

    Returns:
        Merged marker list (existing + newly found), sorted by position.
        Will not exceed expected_count + 1 total markers.
    """
    existing_positions = {m[0] for m in existing_markers}
    relaxed_offset = max(150, initial_max_chars_offset * 3)

    new_markers: List[Tuple[int, str, str, bool, int]] = []

    for i, segment in enumerate(segments):
        if i in existing_positions:
            continue  # Already has a marker

        text = _get_segment_text(segment)
        if not text:
            continue

        # Relaxed scan: search further into the text
        scan_text = text[:relaxed_offset] if relaxed_offset > 0 else ''
        if not scan_text:
            continue

        # Try full-text scan for ordinals, numbered, transitions
        result = _detect_ordinal(scan_text, scan_full_text=True)
        if result is None:
            result = _detect_numbered(scan_text, scan_full_text=True)
        if result is None:
            result = _detect_transition(scan_text, scan_full_text=True)

        if result is not None:
            marker_type, label, char_offset = result
            # For relaxed scan, markers at start (offset 0) are also accepted
            # since the initial pass may have been blocked by max_chars_offset
            is_mid = char_offset > 0
            new_markers.append((i, marker_type, label, is_mid, char_offset))

    if not new_markers:
        return existing_markers

    # Merge existing and new markers, sorted by segment position
    merged = list(existing_markers) + new_markers
    merged.sort(key=lambda m: (m[0], m[4]))  # Sort by position, then char offset

    # Deduplicate: keep only first marker per segment position
    seen_positions: set = set()
    deduped: List[Tuple[int, str, str, bool, int]] = []
    for m in merged:
        if m[0] not in seen_positions:
            seen_positions.add(m[0])
            deduped.append(m)

    # Cap at expected_count + 1 to prevent over-detection
    if len(deduped) > expected_count + 1:
        deduped = deduped[:expected_count + 1]

    return deduped


def detect_listicle_groups(
    segments: List[Any],
    max_chars_offset: int = 50,
) -> List[ListicleGroup]:
    """
    Detect listicle (list-style) structure in voiceover segments.

    Scans segment text for ordinal markers, numbered markers, and transition
    markers to identify list items. Each detected marker starts a new group
    that extends until the next marker or end of segments.

    Performs two passes:
    1. Start-of-text markers (existing behavior)
    2. Mid-segment markers within the first max_chars_offset characters

    When a list header with expected_count is found and the detected marker
    count differs by more than 1, an auto-correction pass re-scans with
    relaxed thresholds (increased max_chars_offset). Auto-correction runs
    only once (no recursion) and logs before/after marker counts.

    Mid-segment markers split the segment: text before the marker belongs to
    the previous group.

    Args:
        segments: List of VoiceoverSegment objects (or dicts with 'text' field)
        max_chars_offset: Maximum character offset for mid-segment detection.
            Markers beyond this offset are ignored. Default 50.

    Returns:
        List[ListicleGroup] representing detected list items.
        Empty list if no listicle structure detected.
        Requires at least 2 markers to confirm listicle structure.
    """
    if not segments:
        return []

    # Initial marker scan
    markers = _scan_markers(segments, max_chars_offset)

    # Scan all segments for a list header to get expected_count
    expected_count: Optional[int] = None
    for segment in segments:
        text = _get_segment_text(segment)
        if text:
            count = detect_list_header(text)
            if count is not None:
                expected_count = count
                break  # Use the first header found

    # Auto-correction: if expected_count known and gap > 1, re-scan with relaxed thresholds
    if expected_count is not None and len(markers) >= 2:
        initial_count = len(markers)
        if abs(initial_count - expected_count) > 1 and initial_count < expected_count:
            logger.info(
                "Auto-correction: expected %d items but detected %d, "
                "re-scanning with relaxed thresholds",
                expected_count, initial_count,
            )
            markers = _auto_correct_markers(
                segments, markers, expected_count, max_chars_offset,
            )
            corrected_count = len(markers)
            logger.info(
                "Auto-correction complete: markers %d -> %d (expected %d)",
                initial_count, corrected_count, expected_count,
            )
            if abs(corrected_count - expected_count) > 1:
                logger.warning(
                    "Auto-correction could not close gap: expected %d, "
                    "detected %d after correction (was %d before)",
                    expected_count, corrected_count, initial_count,
                )

    # Need at least 2 markers to confirm listicle structure
    if len(markers) < 2:
        return []

    # Build groups from markers
    groups = _build_groups_from_markers(markers, segments, expected_count)

    # Normalize mixed marker numbering to consistent sequence
    groups = _normalize_marker_sequence(groups)

    # Log warning if detected count differs from expected by more than 1
    if expected_count is not None and abs(len(groups) - expected_count) > 1:
        logger.warning(
            "Listicle header expected %d items but detected %d markers (diff=%d)",
            expected_count, len(groups), abs(len(groups) - expected_count),
        )

    return groups
