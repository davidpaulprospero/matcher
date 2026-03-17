"""
Keyword extraction package - LLM-based visual keyword extraction.

Provides:
- LLMKeywordExtractor - Main extraction class
- extract_keywords_from_srt - Extract from SRT files
- extract_keyword_per_segment_from_srt - Extract per-segment from SRT
- find_keyword_matches - Match keywords between voiceover and video
- Models: KeywordResult, PrioritizedKeyword

Example:
    from src.keyword_extractor import LLMKeywordExtractor

    extractor = LLMKeywordExtractor(config)
    result = extractor.extract_keywords(voiceover_segments)
    print(result.keywords)  # ['mountain climbing', 'alpine gear', ...]
"""

# Import core class
from .core import LLMKeywordExtractor

# Import models
from .models import KeywordResult, PrioritizedKeyword

# Import utility functions
from .utils import (
    extract_keywords_from_srt,
    extract_keyword_per_segment_from_srt,
    find_keyword_matches
)

# Import specialized functions (for advanced use)
from .validator import validate_visual_keywords, is_visual_keyword
from .segment_processor import (
    extract_keyword_per_segment,
    extract_keywords_grouped,
    extract_simple_keyword
)
from .entity_extractor import extract_entities
from .topic_detector import detect_topic
from .prioritizer import build_prioritized_keywords

# Public API
__all__ = [
    # Main class
    'LLMKeywordExtractor',

    # Models
    'KeywordResult',
    'PrioritizedKeyword',

    # High-level utility functions
    'extract_keywords_from_srt',
    'extract_keyword_per_segment_from_srt',
    'find_keyword_matches',

    # Advanced functions (for specialized use)
    'validate_visual_keywords',
    'is_visual_keyword',
    'extract_keyword_per_segment',
    'extract_keywords_grouped',
    'extract_simple_keyword',
    'extract_entities',
    'detect_topic',
    'build_prioritized_keywords',
]
