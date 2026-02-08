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

    # (position, marker_type, label, is_mid_segment, char_offset)
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

    # Need at least 2 markers to confirm listicle structure
    if len(markers) < 2:
        return []

    # Scan all segments for a list header to get expected_count
    expected_count: Optional[int] = None
    for segment in segments:
        text = _get_segment_text(segment)
        if text:
            count = detect_list_header(text)
            if count is not None:
                expected_count = count
                break  # Use the first header found

    # Build groups from markers
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

    # Log warning if detected count differs from expected by more than 1
    if expected_count is not None and abs(len(groups) - expected_count) > 1:
        logger.warning(
            "Listicle header expected %d items but detected %d markers (diff=%d)",
            expected_count, len(groups), abs(len(groups) - expected_count),
        )

    return groups
