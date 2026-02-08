"""
Chapter Detection Models

Data structures for enhanced chapter detection with confidence scoring.
Backward compatible with existing LocationChapter format.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple


@dataclass
class ChapterConfidence:
    """
    Confidence breakdown for a detected chapter.

    Provides granular confidence scores for different aspects
    of chapter detection quality.
    """
    overall: float = 0.8           # Final combined score (0.0-1.0)
    boundary_confidence: float = 0.8  # How clear are the boundaries?
    content_coherence: float = 0.8    # How coherent is chapter content?
    title_match: float = 0.8          # How well does title match content?
    validation_score: float = 0.8     # Score from validation pass
    detection_agreement: float = 1.0  # Did multiple strategies agree?
    gap_score: float = 0.0            # Semantic gap at boundaries (higher = better)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChapterConfidence":
        return cls(
            overall=data.get('overall', 0.8),
            boundary_confidence=data.get('boundary_confidence', 0.8),
            content_coherence=data.get('content_coherence', 0.8),
            title_match=data.get('title_match', 0.8),
            validation_score=data.get('validation_score', 0.8),
            detection_agreement=data.get('detection_agreement', 1.0),
            gap_score=data.get('gap_score', 0.0),
        )


@dataclass
class ChapterCandidate:
    """
    Enhanced chapter with confidence scoring and detection metadata.

    Backward compatible with LocationChapter - existing fields match,
    new fields have defaults for old checkpoint compatibility.
    """
    # Core fields (match LocationChapter for compatibility)
    chapter_id: int = 0
    start_segment_idx: int = 0
    end_segment_idx: int = 0
    title: str = ""
    topics: List[str] = field(default_factory=list)

    # Location-specific fields (from LocationChapter)
    location_name: str = ""
    location_type: str = "city"  # city, country, landmark, region, natural_feature
    visual_keywords: List[str] = field(default_factory=list)
    context_keywords: List[str] = field(default_factory=list)
    location_data: Optional[Dict] = None  # Resolved GeoLocation as dict

    # NEW: Enhanced detection fields (with defaults for backward compat)
    confidence: float = 0.8              # Overall confidence score
    confidence_details: Optional[Dict] = None  # ChapterConfidence as dict
    detection_strategy: str = "topic"    # How it was detected
    boundary_reasoning: str = ""         # Why boundary was placed here

    def __post_init__(self):
        """Ensure topics list exists"""
        if self.topics is None:
            self.topics = []

    @property
    def segment_range(self) -> Tuple[int, int]:
        """Get segment index range as tuple"""
        return (self.start_segment_idx, self.end_segment_idx)

    @property
    def segment_count(self) -> int:
        """Number of segments in this chapter"""
        return self.end_segment_idx - self.start_segment_idx + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for checkpoint serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChapterCandidate":
        """
        Create from dict with backward-compatible defaults.

        Handles both old LocationChapter format and new enhanced format.
        """
        return cls(
            chapter_id=data.get('chapter_id', 0),
            start_segment_idx=data.get('start_segment_idx', 0),
            end_segment_idx=data.get('end_segment_idx', 0),
            title=data.get('title', ''),
            topics=data.get('topics', []),
            location_name=data.get('location_name', ''),
            location_type=data.get('location_type', 'city'),
            visual_keywords=data.get('visual_keywords', []),
            context_keywords=data.get('context_keywords', []),
            location_data=data.get('location_data'),
            # New fields with defaults for old checkpoints
            confidence=data.get('confidence', 0.8),
            confidence_details=data.get('confidence_details'),
            detection_strategy=data.get('detection_strategy', 'legacy'),
            boundary_reasoning=data.get('boundary_reasoning', ''),
        )

    @classmethod
    def from_location_chapter(cls, lc: Any) -> "ChapterCandidate":
        """
        Create from existing LocationChapter object.

        Used for backward compatibility with existing code.
        """
        if isinstance(lc, dict):
            return cls.from_dict(lc)

        return cls(
            chapter_id=getattr(lc, 'chapter_id', 0),
            start_segment_idx=getattr(lc, 'start_segment_idx', 0),
            end_segment_idx=getattr(lc, 'end_segment_idx', 0),
            title=getattr(lc, 'title', ''),
            topics=getattr(lc, 'topics', []),
            location_name=getattr(lc, 'location_name', ''),
            location_type=getattr(lc, 'location_type', 'city'),
            visual_keywords=getattr(lc, 'visual_keywords', []),
            context_keywords=getattr(lc, 'context_keywords', []),
            location_data=getattr(lc, 'location_data', None),
            confidence=getattr(lc, 'confidence', 0.8),
            detection_strategy='legacy',
        )


@dataclass
class ValidationResult:
    """Result from the validation pass."""
    chapter_id: int
    boundary_correct: bool = True
    suggested_start: Optional[int] = None
    suggested_end: Optional[int] = None
    title_accurate: bool = True
    suggested_title: Optional[str] = None
    content_coherent: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        return cls(
            chapter_id=data.get('chapter_id', 0),
            boundary_correct=data.get('boundary_correct', True),
            suggested_start=data.get('suggested_start'),
            suggested_end=data.get('suggested_end'),
            title_accurate=data.get('title_accurate', True),
            suggested_title=data.get('suggested_title'),
            content_coherent=data.get('content_coherent', True),
            notes=data.get('notes', ''),
        )


@dataclass
class MissedChapter:
    """A chapter that validation detected but initial detection missed."""
    start_segment_idx: int
    end_segment_idx: int
    suggested_title: str
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ListicleGroup:
    """
    A detected listicle item within a voiceover narration.

    Represents a single item in a list-style narration (e.g., "Top 10 reasons...",
    "Step 1... Step 2..."). Groups consecutive segments that belong to the same
    listicle item.
    """
    group_id: int = 0                      # Sequential ID within the listicle
    item_label: str = ""                   # Detected label (e.g., "first", "#3", "step 2")
    start_segment_idx: int = 0
    end_segment_idx: int = 0
    topic_keywords: List[str] = field(default_factory=list)  # Key topics in this item

    @property
    def segment_count(self) -> int:
        return self.end_segment_idx - self.start_segment_idx + 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ListicleGroup":
        return cls(
            group_id=data.get('group_id', 0),
            item_label=data.get('item_label', ''),
            start_segment_idx=data.get('start_segment_idx', 0),
            end_segment_idx=data.get('end_segment_idx', 0),
            topic_keywords=data.get('topic_keywords', []),
        )


@dataclass
class DetectionResult:
    """Complete result from chapter detection pipeline."""
    chapters: List[ChapterCandidate]
    content_type: str = "general"  # travel, educational, documentary, narrative, general
    total_segments: int = 0
    detection_passes_run: List[str] = field(default_factory=list)
    fallback_used: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chapters': [c.to_dict() for c in self.chapters],
            'content_type': self.content_type,
            'total_segments': self.total_segments,
            'detection_passes_run': self.detection_passes_run,
            'fallback_used': self.fallback_used,
            'error_message': self.error_message,
        }
