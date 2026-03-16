"""
Keyword validation and filtering.

Filters out abstract concepts and ensures keywords are visual/filmable.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Abstract pattern regexes (centralized - used in prompts too)
ABSTRACT_PATTERNS = [
    r'\b(the\s+)?quiet\s+confession',
    r'\b(the\s+)?death\s+of\b',
    r'\bpricing\s+out\b',
    r'\bthe\s+truth\s+about\b',
    r'\bthe\s+problem\s+with\b',
    r'\bwhat\s+happened\s+to\b',
    r'\bthe\s+rise\s+and\s+fall\b',
    r'\bchanged\s+everything\b',
    r'\bthe\s+secret\b',
    r'\bthe\s+real\s+reason\b',
    r'\bwhale\s+economy\b',
    r'\bthe\s+end\s+of\b',
    r'\bthe\s+future\s+of\b',
    r'\bthe\s+cost\s+of\b',
    r'\bthe\s+price\s+of\b',
    r'\ba\s+new\s+era\b',
    r'\bthe\s+untold\s+story\b',
    r'\bhidden\s+truth\b',
    r'\bbehind\s+the\s+scenes\b(?!\s+(footage|video|tour))',  # Allow "behind the scenes footage"
]

# Words that strongly indicate visual/filmable content
VISUAL_INDICATORS = [
    'hotel', 'casino', 'street', 'building', 'aerial', 'drone',
    'walkthrough', 'tour', 'footage', '4k', 'timelapse', 'night',
    'day', 'crowd', 'people', 'exterior', 'interior', 'lobby',
    'pool', 'restaurant', 'bar', 'show', 'performance', 'sign',
    'neon', 'lights', 'skyline', 'view', 'entrance', 'parking',
    'strip', 'boulevard', 'avenue', 'plaza', 'resort', 'tower',
    'fountain', 'buffet', 'slot', 'table', 'game', 'room',
    'suite', 'penthouse', 'rooftop', 'desert', 'highway',
]


def validate_visual_keywords(keywords: List[str], max_words: int = 8) -> List[str]:
    """
    Filter out abstract/narrative keywords that won't find B-roll footage.

    Returns only keywords that describe filmable, searchable content.

    Args:
        keywords: List of keyword strings to validate
        max_words: Maximum number of words allowed per keyword (default: 8)

    Returns:
        List of validated keywords (abstract/narrative phrases removed)
    """
    validated = []
    filtered_keywords = []  # Track what was filtered for debugging

    for kw in keywords:
        kw_lower = kw.lower()

        # Strip " footage" suffix from abstract phrases
        # e.g., "the quiet confession footage" → reject entirely
        # This prevents bad keywords like those in your error log
        if ' footage' in kw_lower:
            # Check if the part before " footage" is abstract
            base_kw = kw_lower.replace(' footage', '').strip()
            is_abstract_with_footage = False
            for pattern in ABSTRACT_PATTERNS:
                if re.search(pattern, base_kw):
                    filtered_keywords.append(f"'{kw}' (abstract phrase with footage suffix)")
                    is_abstract_with_footage = True
                    break
            if is_abstract_with_footage:
                continue

        # Check if it matches abstract patterns
        is_abstract = False
        for pattern in ABSTRACT_PATTERNS:
            if re.search(pattern, kw_lower):
                filtered_keywords.append(f"'{kw}' (abstract pattern)")
                is_abstract = True
                break

        if is_abstract:
            continue

        # Check if it's too long (likely a script phrase)
        word_count = len(kw.split())
        if word_count > max_words:
            filtered_keywords.append(f"'{kw}' (too long: {word_count} words)")
            continue

        # Check for visual indicators or proper nouns (locations/names)
        has_visual = any(ind in kw_lower for ind in VISUAL_INDICATORS)
        words = kw.split()
        has_proper_noun = any(w[0].isupper() for w in words if len(w) > 2)

        # Accept if has visual indicator, proper noun, or is short/specific
        if has_visual or has_proper_noun or word_count <= 3:
            validated.append(kw)
        else:
            # Log but still accept - might be valid
            logger.debug(f"Keyword without visual indicator (kept): '{kw}'")
            validated.append(kw)

    if filtered_keywords:
        logger.info(f"Filtered {len(filtered_keywords)} abstract/narrative keywords: {filtered_keywords}")

    return validated


def is_visual_keyword(keyword: str, max_words: int = 8) -> bool:
    """
    Check if keyword represents filmable content.

    Args:
        keyword: Keyword string to check
        max_words: Maximum number of words allowed (default: 8)

    Returns:
        True if keyword is visual/filmable, False if abstract
    """
    kw_lower = keyword.lower()

    # Check against abstract patterns
    for pattern in ABSTRACT_PATTERNS:
        if re.search(pattern, kw_lower):
            return False

    # Check length (too long = likely narrative)
    word_count = len(keyword.split())
    if word_count > max_words:
        return False

    # Check for visual indicators or proper nouns
    has_visual = any(ind in kw_lower for ind in VISUAL_INDICATORS)
    words = keyword.split()
    has_proper_noun = any(w[0].isupper() for w in words if len(w) > 2)

    # Accept if has visual indicator, proper noun, or is short/specific
    return has_visual or has_proper_noun or word_count <= 3
