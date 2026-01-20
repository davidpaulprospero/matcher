"""
Data models for chapter detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chapter:
    """A detected chapter/list item in the voiceover."""

    name: str  # Original name from ASR
    corrected_name: str  # ASR-corrected name (e.g., "Keatsahut" -> "Pizza Hut")
    rank: Optional[int]  # Position in list (15, 14, 13...) or None for non-listicle
    start_segment: int  # First segment index
    end_segment: int  # Last segment index (inclusive)
    keywords: List[str] = field(default_factory=list)  # Per-chapter search keywords
    description: str = ""  # Brief description of chapter content

    @property
    def segment_range(self) -> range:
        """Get segment indices as a range."""
        return range(self.start_segment, self.end_segment + 1)

    def contains_segment(self, segment_idx: int) -> bool:
        """Check if segment index is within this chapter."""
        return self.start_segment <= segment_idx <= self.end_segment

    def __repr__(self) -> str:
        rank_str = f"#{self.rank} " if self.rank is not None else ""
        return f"Chapter({rank_str}{self.corrected_name}, segments {self.start_segment}-{self.end_segment})"


@dataclass
class ChapterDetectionResult:
    """Result of chapter detection."""

    is_listicle: bool
    list_type: Optional[str]  # "countdown", "top_n", "ranked", "unordered", None
    total_items: int
    chapters: List[Chapter]
    intro_end_segment: Optional[int] = None  # Where intro ends (before first list item)
    outro_start_segment: Optional[int] = None  # Where outro starts (after last list item)

    @property
    def has_chapters(self) -> bool:
        """Check if any chapters were detected."""
        return len(self.chapters) > 0

    def get_chapter_for_segment(self, segment_idx: int) -> Optional[Chapter]:
        """Get the chapter containing a given segment index."""
        for chapter in self.chapters:
            if chapter.contains_segment(segment_idx):
                return chapter
        return None

    def __repr__(self) -> str:
        if self.is_listicle:
            return f"ChapterDetectionResult(listicle={self.list_type}, {len(self.chapters)} items)"
        elif self.has_chapters:
            return f"ChapterDetectionResult(chapters={len(self.chapters)})"
        else:
            return "ChapterDetectionResult(no chapters)"
