"""
Chapter detection module for listicle and multi-topic videos.

Uses LLM to detect chapter/list structure and extract per-chapter keywords.
"""

from .core import ChapterDetector, detect_chapters
from .models import Chapter, ChapterDetectionResult

__all__ = [
    'ChapterDetector',
    'detect_chapters',
    'Chapter',
    'ChapterDetectionResult',
]
