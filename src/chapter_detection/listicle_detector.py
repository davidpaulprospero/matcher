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
from typing import Any, Dict, List, Optional, Tuple

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


# Number format types for normalization
class NumberFormat:
    ORDINAL = 'ordinal'      # first, second, third
    HASH_NUMBERED = 'hash'   # #1, #2, #3
    WORD_NUMBERED = 'word'   # one, two, three
    DIGIT_NUMBERED = 'digit' # 1, 2, 3
    TRANSITION = 'transition' # next up, moving on to


def _get_ordinal_suffix(n: int) -> str:
    """Get the ordinal suffix for a number (st, nd, rd, th)."""
    if 10 <= n % 100 <= 20:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


def _detect_number_format(marker_type: str, label: str) -> Optional[str]:
    """
    Detect the numbering format of a marker.

    Args:
        marker_type: Type of marker ('ordinal', 'numbered', 'transition')
        label: The marker label text

    Returns:
        One of: 'ordinal', 'hash', 'word', 'digit', 'transition', or None
    """
    if marker_type == 'transition':
        return NumberFormat.TRANSITION

    if marker_type == 'ordinal':
        return NumberFormat.ORDINAL

    if marker_type == 'numbered':
        # Check for hash format (#1, #2)
        if label.lower().startswith('#'):
            return NumberFormat.HASH_NUMBERED
        # Check for digit format (Step 1, Item 2)
        if re.search(r'\d', label):
            return NumberFormat.DIGIT_NUMBERED
        # Check for word format (number one, step two)
        for word in NUMBER_WORDS:
            if word in label.lower():
                return NumberFormat.WORD_NUMBERED

    return None


def detect_inconsistent_numbering(
    groups: List['ListicleGroup'],
) -> bool:
    """
    Detect if listicle groups have inconsistent numbering formats.

    A listicle is considered inconsistent if it mixes different numbering
    format types (e.g., ordinals 'first, second' with '#3', or ordinals
    with word-numbered 'one, two, three').

    Args:
        groups: List of ListicleGroup objects

    Returns:
        True if numbering is inconsistent, False if consistent or indeterminate
    """
    if len(groups) < 2:
        return False

    detected_formats: List[str] = []
    positioned_count = 0

    for group in groups:
        fmt = _detect_number_format(group.marker_type, group.item_label)
        if fmt and fmt != NumberFormat.TRANSITION:
            detected_formats.append(fmt)
            positioned_count += 1

    # Need at least 2 positioned markers to detect inconsistency
    if positioned_count < 2:
        return False

    # Check if we have more than one format type
    unique_formats = set(detected_formats)
    if len(unique_formats) > 1:
        return True

    return False


def normalize_numbering_format(
    groups: List['ListicleGroup'],
) -> List['ListicleGroup']:
    """
    Normalize numbering format within listicle groups to be consistent.

    When groups have mixed numbering (e.g., 'first, #3, third'), this
    normalizes all markers to use a consistent ordinal format (1st, 2nd, 3rd)
    based on the group's position in the sequence.

    Also detects inconsistent numbering and sets the inconsistent_numbering
    flag on groups when detected (for later penalty application).

    Args:
        groups: List of ListicleGroup objects (modified in place and returned)

    Returns:
        List of ListicleGroup with normalized numbering
    """
    if len(groups) < 2:
        return groups

    # Check for inconsistency
    is_inconsistent = detect_inconsistent_numbering(groups)

    if is_inconsistent:
        # Determine the most common format or default to ordinal
        format_counts: Dict[str, int] = {}
        for group in groups:
            fmt = _detect_number_format(group.marker_type, group.item_label)
            if fmt and fmt != NumberFormat.TRANSITION:
                format_counts[fmt] = format_counts.get(fmt, 0) + 1

        # Choose target format: ordinal has priority if present, else most common
        target_format = NumberFormat.ORDINAL
        if NumberFormat.ORDINAL not in format_counts and format_counts:
            target_format = max(format_counts, key=format_counts.get)

        logger.info(
            "Listicle numbering inconsistent (formats: %s), normalizing to %s",
            format_counts,
            target_format,
        )

        # Mark all groups as having inconsistent numbering (for penalty application)
        for group in groups:
            group.inconsistent_numbering = True

        # Normalize all markers to ordinal format (1st, 2nd, 3rd...)
        for idx, group in enumerate(groups):
            # Convert to ordinal: 1st, 2nd, 3rd, etc.
            ordinal_suffix = _get_ordinal_suffix(idx + 1)
            new_label = f"{idx + 1}{ordinal_suffix}"
            group.item_label = new_label
            group.marker_type = 'ordinal'

    return groups


# Transition marker patterns
TRANSITION_PATTERNS = [
    re.compile(r'^\s*next\s+up\b', re.IGNORECASE),
    re.compile(r'^\s*moving\s+on\s+to\b', re.IGNORECASE),
    re.compile(r"^\s*let'?s\s+talk\s+about\b", re.IGNORECASE),
    re.compile(r"^\s*let'?s\s+move\s+on\s+to\b", re.IGNORECASE),
    re.compile(r'^\s*now\s+(?:for|let\'?s\s+look\s+at)\b', re.IGNORECASE),
    re.compile(r'^\s*another\s+(?:thing|reason|way|tip|point)\b', re.IGNORECASE),
    re.compile(r'^\s*on\s+to\s+(?:the\s+)?(?:next|our\s+next)\b', re.IGNORECASE),
    # US-122-009: New transition patterns
    re.compile(r'^\s*in\s+this\s+(?:episode|video|part|section)\b', re.IGNORECASE),
    re.compile(r'^\s*coming\s+up\s+next\b', re.IGNORECASE),
    re.compile(r'^\s*up\s+next\b', re.IGNORECASE),
    # Note: "here is/are/comes" patterns removed - too aggressive, match header patterns
    re.compile(r'^\s*stay\s+tuned\s+for\b', re.IGNORECASE),
    re.compile(r"^\s*don'?t\s+(?:go\s+away|leave)\b", re.IGNORECASE),
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
    # US-122-009: New transition patterns (mid-segment)
    re.compile(r'\bin\s+this\s+(?:episode|video|part|section)\b', re.IGNORECASE),
    re.compile(r'\bcoming\s+up\s+next\b', re.IGNORECASE),
    re.compile(r'\bup\s+next\b', re.IGNORECASE),
    # Note: "here is/are/comes" patterns removed - too aggressive
    re.compile(r'\bstay\s+tuned\s+for\b', re.IGNORECASE),
    re.compile(r"\bdon't\s+(?:go\s+away|leave)\b", re.IGNORECASE),
]

# Word-form numbers for header detection (e.g., "five reasons", "seven tips")
# Includes English and common non-English number words
HEADER_NUMBER_WORDS = {
    # English
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20,
    # Extended English
    'thirteen': 13, 'fourteen': 14, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100,
    # Spanish
    'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
    'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
    'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
    'dieciséis': 16, 'diecisiete': 17, 'dieciocho': 18, 'diecinueve': 19,
    'veinte': 20, 'treinta': 30, 'cuarenta': 40, 'cincuenta': 50,
    'sesenta': 60, 'setenta': 70, 'ochenta': 80, 'noventa': 90,
    'cien': 100,
    # French
    'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5,
    'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10,
    'onze': 11, 'douze': 12, 'treize': 13, 'quatorze': 14, 'quinze': 15,
    'seize': 16, 'dix-sept': 17, 'dix-huit': 18, 'dix-neuf': 19,
    'vingt': 20, 'trente': 30, 'quarante': 40, 'cinquante': 50,
    'soixante': 60, 'soixante-dix': 70, 'quatre-vingts': 80,
    'quatre-vingt-dix': 90, 'cent': 100,
    # German
    'zwei': 2, 'drei': 3, 'vier': 4, 'fünf': 5,
    'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
    'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
    'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19,
    'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50,
    'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90,
    'hundert': 100,
    # Portuguese (Brazilian/European)
    'um': 1, 'dois': 2, 'duas': 2, 'três': 3, 'quatro': 4, 'cinco': 5,
    'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10,
    'onze': 11, 'doze': 12, 'treze': 13, 'catorze': 14, 'quinze': 15,
    'dezasseis': 16, 'dezessete': 17, 'dezoito': 18, 'dezanove': 19,
    'vinte': 20, 'trinta': 30, 'quarenta': 40, 'cinquenta': 50,
    'sessenta': 60, 'setenta': 70, 'oitenta': 80, 'noventa': 90,
    'cem': 100,
    # Italian
    'uno': 1, 'due': 2, 'tre': 3, 'quattro': 4, 'cinque': 5,
    'sei': 6, 'sette': 7, 'otto': 8, 'nove': 9, 'dieci': 10,
    'undici': 11, 'dodici': 12, 'tredici': 13, 'quattordici': 14, 'quindici': 15,
    'sedici': 16, 'diciassette': 17, 'diciotto': 18, 'diciannove': 19,
    'venti': 20, 'trenta': 30, 'quaranta': 40, 'cinquanta': 50,
    'sessanta': 70, 'settanta': 70, 'ottanta': 80, 'novanta': 90,
    'cento': 100,
    # Japanese kanji numbers (used in context like "5つの理由" = "5 reasons")
    # Only include kanji that are unambiguous when followed by common counters
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}

# List header nouns that follow the count
_HEADER_NOUNS = (
    r'(?:reasons?|ways?|tips?|things?|steps?|points?|facts?|'
    r'places?|ideas?|mistakes?|secrets?|signs?|tricks?|methods?|'
    r'rules?|lessons?|examples?|benefits?|attractions?|destinations?|'
    r'trails?|hotels?|restaurants?|products?|gadgets?|apps?|tools?|'
    r'spots?|cities?|countries?)'
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
    # "the N best/worst/most <adj> <noun>" — numeric with adjective (e.g., "the 10 best hiking trails")
    re.compile(r'\bthe\s+(\d+)\s+(?:best|worst|most\s+\w+)\s+\w+\s+' + _HEADER_NOUNS, re.IGNORECASE),
    # "the N best/worst/most <noun>" — word-form
    re.compile(
        r'\bthe\s+(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+(?:best|worst|most\s+\w+)\s+'
        + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
    # "the N best/worst/most <adj> <noun>" — word-form with adjective
    re.compile(
        r'\bthe\s+(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+(?:best|worst|most\s+\w+)\s+\w+\s+'
        + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
    # "N best/worst/most <noun>" without "the" — numeric (e.g., "10 best tips")
    re.compile(r'\b(\d+)\s+(?:best|worst|most\s+\w+)\s+' + _HEADER_NOUNS, re.IGNORECASE),
    # "N best/worst/most <adj> <noun>" without "the" — numeric (e.g., "10 best hiking trails")
    re.compile(r'\b(\d+)\s+(?:best|worst|most\s+\w+)\s+\w+\s+' + _HEADER_NOUNS, re.IGNORECASE),
    # "N best/worst/most <noun>" without "the" — word-form (e.g., "five best tips")
    re.compile(
        r'\b(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+(?:best|worst|most\s+\w+)\s+'
        + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
    # "N best/worst/most <adj> <noun>" without "the" — word-form with adjective
    re.compile(
        r'\b(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+(?:best|worst|most\s+\w+)\s+\w+\s+'
        + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
    # Range pattern: "15-20 ways to...", "five to ten tips"
    # Captures the first number from the range
    re.compile(r'(\d+)\s*-\s*\d+\s+' + _HEADER_NOUNS, re.IGNORECASE),
    re.compile(
        r'(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+(?:to|-)\s+'
        + r'(' + '|'.join(HEADER_NUMBER_WORDS) + r')\s+' + _HEADER_NOUNS,
        re.IGNORECASE,
    ),
]


def _extract_simple_keywords(
    text: str,
    max_keywords: int = 5,
    min_length_threshold: int = 20,
) -> List[str]:
    """Extract simple topic keywords from segment text using rule-based approach.

    Args:
        text: Text to extract keywords from.
        max_keywords: Maximum number of keywords to return.
        min_length_threshold: Minimum text length required for extraction.
            Shorter texts return empty list gracefully.
    """
    # Handle very short segments gracefully - return empty for insufficient text
    if not text or len(text.strip()) < min_length_threshold:
        logger.debug(
            "Text too short for keyword extraction (length=%d < threshold=%d)",
            len(text) if text else 0,
            min_length_threshold,
        )
        return []

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

    # Deduplicate keywords case-insensitively (e.g., 'Python' vs 'python' same)
    # Preserve order of first occurrence (case of first occurrence kept)
    seen_lower = set()
    unique = []
    for w in keywords:
        w_lower = w.lower()
        if w_lower not in seen_lower:
            seen_lower.add(w_lower)
            unique.append(w)

    # Validate minimum quality threshold: require at least some meaningful content
    # If after filtering we have very few, it's low quality
    if len(unique) < 2 and len(words) > 3:
        logger.debug(
            "Keyword quality too low: only %d keywords from %d words",
            len(unique),
            len(words),
        )
        return unique[:max_keywords] if unique else []

    return unique[:max_keywords]


def _extract_keywords_with_llm(text: str, max_keywords: int = 5) -> List[str]:
    """Extract topic keywords using LLM for better quality.

    Called when simple extraction yields insufficient keywords.
    """
    try:
        from src.llm_client import LLMRequest, ResponseFormat, create_client

        prompt = f"""Extract {max_keywords} distinct topic keywords from the following text.
Return ONLY a JSON array of strings, like ["keyword1", "keyword2", "keyword3"].
Focus on meaningful nouns and topics, not common words.

Text: {text[:1000]}"""

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            max_tokens=200,
            cache_key_prefix="listicle_topic_keywords"
        )

        client = create_client()
        response = client.generate(request)

        if response.parsed_data and isinstance(response.parsed_data, list):
            # Filter to only valid string keywords
            keywords = [str(k).lower().strip() for k in response.parsed_data if k]
            return keywords[:max_keywords]

    except Exception as e:
        logger.warning("LLM keyword extraction failed: %s", e)

    return []


def _enhance_keywords_with_embeddings(
    text: str,
    base_keywords: List[str],
    max_keywords: int = 5,
    similarity_threshold: float = 0.6,
    embedding_provider: Any = None,
) -> List[str]:
    """Enhance extracted keywords using semantic embeddings.

    Uses embedding similarity to find related terms beyond simple word matching.
    This helps discover semantically related keywords that wouldn't be caught
    by rule-based extraction.

    Args:
        text: Original text for context.
        base_keywords: Keywords extracted via simple extraction.
        max_keywords: Maximum total keywords to return.
        similarity_threshold: Minimum similarity to include boosted keywords.
        embedding_provider: Provider for embedding computation.

    Returns:
        Enhanced keyword list with semantically similar terms.
    """
    if not base_keywords or not embedding_provider:
        return base_keywords

    try:
        from ..transcription.embeddings import cosine_similarity
    except ImportError:
        logger.debug("Embeddings module not available for semantic enhancement")
        return base_keywords

    # If we already have enough keywords, skip enhancement
    if len(base_keywords) >= max_keywords:
        return base_keywords

    # Common related terms to search for (domain-agnostic base)
    # In practice, this could be expanded with domain-specific term lists
    candidate_terms = [
        # Technology
        'software', 'hardware', 'digital', 'computer', 'mobile', 'app', 'application',
        'website', 'online', 'internet', 'cloud', 'data', 'ai', 'machine learning',
        # Travel
        'travel', 'trip', 'destination', 'vacation', 'tourist', 'attraction', 'landmark',
        'city', 'country', 'visit', 'guide', 'tour', 'hotel', 'restaurant',
        # Food
        'food', 'recipe', 'cook', 'kitchen', 'restaurant', 'dish', 'meal', 'ingredient',
        'taste', 'delicious', 'cooking', 'baking', 'chef',
        # Business
        'business', 'company', 'startup', 'entrepreneur', 'market', 'industry', 'strategy',
        'growth', 'sales', 'marketing', 'customer', 'product', 'service',
        # Science
        'research', 'study', 'experiment', 'discovery', 'science', 'technology', 'innovation',
        # Health
        'health', 'fitness', 'exercise', 'workout', 'diet', 'nutrition', 'wellness',
        'medical', 'healthcare', 'doctor', 'treatment',
    ]

    # Filter candidates to exclude already-known keywords
    known_lower = {kw.lower() for kw in base_keywords}
    candidates = [t for t in candidate_terms if t.lower() not in known_lower]

    if not candidates:
        return base_keywords

    try:
        # Embed base keywords combined text
        base_text = " ".join(base_keywords)
        base_emb = embedding_provider.get_embedding(base_text.lower())

        # Embed each candidate and compute similarity
        enhanced = list(base_keywords)
        scored_candidates = []

        for candidate in candidates:
            try:
                cand_emb = embedding_provider.get_embedding(candidate)
                similarity = cosine_similarity(base_emb, cand_emb)

                if similarity >= similarity_threshold:
                    scored_candidates.append((candidate, similarity))
            except Exception:
                continue

        # Sort by similarity and add top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        for candidate, _ in scored_candidates:
            if len(enhanced) >= max_keywords:
                break
            enhanced.append(candidate)

        if len(enhanced) > len(base_keywords):
            logger.debug(
                "Embedding enhancement added %d related keywords (threshold=%.2f)",
                len(enhanced) - len(base_keywords),
                similarity_threshold,
            )

        return enhanced[:max_keywords]

    except Exception as e:
        logger.debug("Embedding keyword enhancement failed: %s", e)
        return base_keywords


def _extract_topic_keywords(
    text: str,
    max_keywords: int = 5,
    use_llm: bool = False,
    min_keywords_for_llm: int = 3,
    use_embedding: bool = False,
    embedding_similarity_threshold: float = 0.6,
    embedding_provider: Any = None,
) -> List[str]:
    """Extract topic keywords from segment text.

    Uses simple rule-based extraction first. If use_llm is True and simple
    extraction yields fewer than min_keywords_for_llm keywords, falls back
    to LLM-based extraction for better quality. If use_embedding is True,
    attempts to enhance keywords with semantic embedding similarity.

    Args:
        text: Text to extract keywords from.
        max_keywords: Maximum number of keywords to return.
        use_llm: Whether to use LLM fallback when simple extraction is insufficient.
        min_keywords_for_llm: Minimum keywords needed before LLM fallback triggers.
        use_embedding: Whether to use embedding-based enhancement (US-135-003).
        embedding_similarity_threshold: Minimum similarity for embedding-boosted keywords.
        embedding_provider: Provider for embedding computation.
    """
    # Simple extraction first
    keywords = _extract_simple_keywords(text, max_keywords)

    # If we have enough keywords or LLM is disabled, return simple results
    if not use_llm or len(keywords) >= min_keywords_for_llm:
        if use_llm and len(keywords) >= min_keywords_for_llm:
            logger.debug(
                "Using simple keyword extraction (got %d >= %d min keywords)",
                len(keywords),
                min_keywords_for_llm,
            )

        # Apply embedding enhancement if enabled (US-135-003)
        if use_embedding and embedding_provider and len(keywords) < max_keywords:
            keywords = _enhance_keywords_with_embeddings(
                text,
                keywords,
                max_keywords,
                embedding_similarity_threshold,
                embedding_provider,
            )

        return keywords

    # LLM fallback: simple extraction yielded insufficient keywords
    logger.debug(
        "Simple extraction returned %d keywords (< %d min), attempting LLM fallback",
        len(keywords),
        min_keywords_for_llm,
    )
    llm_keywords = _extract_keywords_with_llm(text, max_keywords)

    # If LLM succeeded, return those keywords
    if llm_keywords:
        logger.info(
            "LLM keyword extraction succeeded: got %d keywords",
            len(llm_keywords),
        )
        # Also try embedding enhancement on LLM results
        if use_embedding and embedding_provider:
            llm_keywords = _enhance_keywords_with_embeddings(
                text,
                llm_keywords,
                max_keywords,
                embedding_similarity_threshold,
                embedding_provider,
            )
        return llm_keywords

    # LLM failed, try embedding enhancement on simple extraction results
    if use_embedding and embedding_provider:
        keywords = _enhance_keywords_with_embeddings(
            text,
            keywords,
            max_keywords,
            embedding_similarity_threshold,
            embedding_provider,
        )

    # LLM failed, return whatever we got from simple extraction
    logger.debug("LLM keyword extraction failed, falling back to simple extraction")
    return keywords


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
    - "15-20 ways to..." → 15 (uses first number)
    - "five to ten tips" → 5 (uses first number)

    Returns the expected count or None if no header detected.
    """
    for pattern in HEADER_PATTERNS:
        match = pattern.search(text)
        if match:
            # Handle range patterns with two capture groups (e.g., "five to ten")
            if match.lastindex == 2:
                # Use first group for ranges
                return _parse_header_count(match.group(1))
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
    # 1. Positioned markers in sorted order (all positioned first)
    # 2. Unpositioned markers in original order (all unpositioned after)
    # 3. Terminal markers at the end
    result: List[ListicleGroup] = []

    # Add all positioned markers first (in sorted order)
    for _, _, group in positioned:
        result.append(group)

    # Add unpositioned markers in original order
    for _, group in unpositioned:
        result.append(group)

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
    use_llm: bool = False,
    min_keywords_for_llm: int = 3,
    use_embedding: bool = False,
    embedding_similarity_threshold: float = 0.6,
    embedding_provider: Any = None,
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
        topic_keywords = _extract_topic_keywords(
            all_text,
            use_llm=use_llm,
            min_keywords_for_llm=min_keywords_for_llm,
            use_embedding=use_embedding,
            embedding_similarity_threshold=embedding_similarity_threshold,
            embedding_provider=embedding_provider,
        )

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
    listicle_topic_config: Any = None,
    embedding_provider: Any = None,
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
        listicle_topic_config: Optional ListicleTopicConfig for keyword extraction.
            When provided and use_llm_topic_extraction is True, uses LLM fallback
            when simple keyword extraction yields insufficient results.

    Returns:
        List[ListicleGroup] representing detected list items.
        Empty list if no listicle structure detected.
        Requires at least 2 markers to confirm listicle structure.
    """
    # Extract config settings for keyword extraction
    use_llm = False
    min_keywords_for_llm = 3
    use_embedding = False
    embedding_similarity_threshold = 0.6
    auto_correction = True  # Default to True for backward compatibility
    if listicle_topic_config is not None:
        use_llm = getattr(listicle_topic_config, 'use_llm_topic_extraction', False)
        min_keywords_for_llm = getattr(listicle_topic_config, 'min_keywords_for_simple', 3)
        use_embedding = getattr(listicle_topic_config, 'use_embedding_topic_extraction', False)
        embedding_similarity_threshold = getattr(
            listicle_topic_config, 'embedding_similarity_threshold', 0.6
        )
        auto_correction = getattr(listicle_topic_config, 'auto_correction', True)
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
    groups = _build_groups_from_markers(
        markers, segments, expected_count,
        use_llm=use_llm,
        min_keywords_for_llm=min_keywords_for_llm,
        use_embedding=use_embedding,
        embedding_similarity_threshold=embedding_similarity_threshold,
        embedding_provider=embedding_provider,
    )

    # Normalize mixed marker numbering to consistent sequence
    groups = _normalize_marker_sequence(groups)

    # Auto-correct inconsistent numbering (e.g., 'first, #3, third' -> '1st, 2nd, 3rd')
    # This normalizes numbering format when multiple formats are mixed
    # Controlled by auto_correction config option (US-140-004)
    if auto_correction:
        groups = normalize_numbering_format(groups)

    # Log warning if detected count differs from expected by more than 1 (use original expected_count)
    original_expected = expected_count
    if expected_count is not None and abs(len(groups) - expected_count) > 1:
        logger.warning(
            "Listicle header expected %d items but detected %d markers (diff=%d)",
            expected_count, len(groups), abs(len(groups) - expected_count),
        )

    # Cap expected_count when header count significantly exceeds actual segments (graceful handling)
    # Only cap when diff is large (> 3) to preserve expected_count for minor mismatches
    if expected_count is not None and len(groups) < expected_count:
        if abs(len(groups) - expected_count) > 3:
            logger.info(
                "Capping expected_count from %d to %d (only %d segments detected)",
                expected_count, len(groups), len(groups),
            )
            # Update expected_count on all groups to reflect actual count
            expected_count = len(groups)
            for g in groups:
                g.expected_count = expected_count

    return groups
