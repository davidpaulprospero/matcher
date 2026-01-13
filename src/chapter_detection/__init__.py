"""
Chapter Detection Package

Multi-pass chapter detection system for voiceover content.
Addresses incorrect boundaries, missing chapters, and granularity issues.
"""

from .models import ChapterCandidate, ChapterConfidence
from .detector import EnhancedChapterDetector

__all__ = [
    'ChapterCandidate',
    'ChapterConfidence',
    'EnhancedChapterDetector',
]
