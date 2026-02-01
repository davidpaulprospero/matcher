"""
Chapter Detector Module

Extracts chapter markers from video metadata (e.g., YouTube chapters).
Distinct from chapter_detection which uses LLM for transcript analysis.
"""

from .detector import VideoChapter, extract_chapters_from_metadata

__all__ = ['VideoChapter', 'extract_chapters_from_metadata']
