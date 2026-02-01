"""
File I/O utilities for keyword extraction.

Convenience functions for extracting keywords from SRT files and matching keywords.
"""

import re
import logging
from typing import List, Tuple
from collections import Counter

from .models import KeywordResult
from .core import LLMKeywordExtractor

logger = logging.getLogger(__name__)


def extract_keywords_from_srt(
    srt_path: str,
    config
) -> KeywordResult:
    """
    Convenience function to extract keywords from SRT file.

    Args:
        srt_path: Path to SRT file
        config: Pipeline config

    Returns:
        KeywordResult with extracted keywords
    """
    import srt

    # Parse SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))

    # Convert to segments format
    segments = [{'text': sub.content} for sub in subtitles]

    # Extract keywords
    extractor = LLMKeywordExtractor(config)
    return extractor.extract_keywords(segments)


def extract_keyword_per_segment_from_srt(
    srt_path: str,
    config,
    topic: str = ""
) -> List[str]:
    """
    Extract ONE keyword per SRT segment for precise B-roll matching.

    Args:
        srt_path: Path to SRT file
        config: Pipeline config
        topic: Documentary topic for context (auto-detected if not provided)

    Returns:
        List of keywords (one per segment, in order)
    """
    import srt

    # Parse SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))

    # Convert to segments format
    segments = [{'text': sub.content} for sub in subtitles]

    # Auto-detect topic from first few segments if not provided
    if not topic:
        first_text = ' '.join([s['text'] for s in segments[:5]])
        # Simple topic detection: use most common nouns
        words = re.findall(r'\b[A-Z][a-z]+\b', first_text)
        if words:
            common = Counter(words).most_common(2)
            topic = ' '.join([w for w, _ in common])

    # Extract per-segment keywords
    extractor = LLMKeywordExtractor(config)
    return extractor.extract_keyword_per_segment(segments, topic=topic)


def find_keyword_matches(
    voiceover_keywords: List[str],
    video_keywords: List[str],
    visual_keywords: List[str] = None,
    keyword_boost: float = 0.05,
    visual_boost: float = 0.03,
    max_boost: float = 0.2
) -> Tuple[float, bool, bool]:
    """
    Find keyword overlap between voiceover and video.
    Returns (boost_score, is_keyword_match, is_visual_match)

    Args:
        voiceover_keywords: Keywords from voiceover
        video_keywords: Keywords from video transcript
        visual_keywords: Keywords from vision analysis
        keyword_boost: Boost per matching text keyword (default from config.matching.keyword_boost)
        visual_boost: Boost per matching visual keyword
        max_boost: Maximum total boost cap

    Returns:
        Tuple of (boost_score, is_keyword_match, is_visual_match)
    """
    vo_set = set(k.lower() for k in voiceover_keywords)
    vid_set = set(k.lower() for k in video_keywords)
    vis_set = set(k.lower() for k in (visual_keywords or []))

    # Text keyword match
    text_overlap = len(vo_set & vid_set)
    is_keyword_match = text_overlap > 0

    # Visual keyword match
    visual_overlap = len(vo_set & vis_set)
    is_visual_match = visual_overlap > 0

    # Calculate boost score
    boost = 0.0
    if is_keyword_match:
        boost += keyword_boost * text_overlap
    if is_visual_match:
        boost += visual_boost * visual_overlap

    return min(boost, max_boost), is_keyword_match, is_visual_match
