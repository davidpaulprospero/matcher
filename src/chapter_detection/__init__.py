"""
Chapter Detection Package

Multi-pass chapter detection system for voiceover content.
Addresses incorrect boundaries, missing chapters, and granularity issues.
"""

from .models import ChapterCandidate, ChapterConfidence
from .detector import EnhancedChapterDetector
from .bridge import build_unified_chapters, build_segment_chapter_map, compute_relevance_matrix

__all__ = [
    'ChapterCandidate',
    'ChapterConfidence',
    'EnhancedChapterDetector',
    'build_unified_chapters',
    'build_segment_chapter_map',
    'compute_relevance_matrix',
]
