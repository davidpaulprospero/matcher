"""
Candidate Filter - Extract filtering logic from TieredMatcher.

US-33-006: Refactor tiered_matcher.py - extract candidate filtering.
This module provides composition-based filtering for video candidates.

Filtering operations:
- Face preference filtering (more/none/neutral)
- Location-based geographic filtering
- Smart reuse prevention filtering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import SRTSegment, ReuseTracker
    from .location_matching import LocationMatcher

logger = logging.getLogger(__name__)


@dataclass
class CandidateFilterConfig:
    """Configuration for candidate filtering."""
    face_preference: str = "neutral"  # "more", "none", or "neutral"
    location_matching_enabled: bool = False
    location_matching_config: Optional[dict] = None


@dataclass
class FilterResult:
    """Result from filtering operation."""
    candidates: List[Tuple['SRTSegment', float]]
    face_filter_applied: bool = False
    location_filter_applied: bool = False
    location_reason: str = ""
    reuse_filter_applied: bool = False
    original_count: int = 0
    filtered_count: int = 0


class CandidateFilter:
    """
    Filters video candidates based on face preference, location, and reuse.

    Uses composition pattern - initialized with config and optional matchers.
    Delegates location filtering to LocationMatcher if available.

    Example usage:
        filter = CandidateFilter(config)
        result = filter.apply_all_filters(
            vo_segment, candidates, segment_idx, reuse_tracker, cache_dir
        )
        filtered_candidates = result.candidates
    """

    def __init__(
        self,
        config: Optional['Config'] = None,
        location_matcher: Optional['LocationMatcher'] = None
    ):
        """
        Initialize CandidateFilter.

        Args:
            config: Configuration object (uses get_config() if None)
            location_matcher: Optional LocationMatcher for geographic filtering
        """
        from ..config import get_config
        self.config = config or get_config()
        self.location_matcher = location_matcher

        # Extract filter-related config
        mc = self.config.matching
        self.face_preference = getattr(mc, 'face_preference', 'neutral')
        self.location_matching_config = getattr(mc, 'location_matching', None)
        self.location_matching_enabled = False

        if self.location_matching_config:
            if isinstance(self.location_matching_config, dict):
                self.location_matching_enabled = self.location_matching_config.get('enabled', False)
            else:
                self.location_matching_enabled = getattr(self.location_matching_config, 'enabled', False)

    def apply_face_preference(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        cache_dir: Optional[str] = None
    ) -> Tuple[List[Tuple['SRTSegment', float]], bool]:
        """
        Apply face preference filtering to candidates.

        Separates candidates into project (has face detection) and cached (uses cached scores).

        Args:
            candidates: List of (segment, similarity) tuples
            cache_dir: Optional cache directory for face detection

        Returns:
            Tuple of (filtered_candidates, was_applied)
        """
        if self.face_preference == 'neutral':
            return candidates, False

        from ..face_detection import apply_face_preference as detect_face_preference

        current_project_candidates = []
        global_cache_candidates = []

        for seg, sim in candidates:
            source = getattr(seg, 'source', None)
            if source == 'global_cache':
                global_cache_candidates.append((seg, sim))
            else:
                current_project_candidates.append((seg, sim))

        # Apply face detection to current project videos
        if current_project_candidates:
            logger.debug(
                f"Face preference '{self.face_preference}' - "
                f"{len(current_project_candidates)} project, {len(global_cache_candidates)} cached"
            )
            current_project_candidates = detect_face_preference(
                current_project_candidates, self.face_preference, cache_dir
            )

        # Apply cached face scores to global cache candidates
        if global_cache_candidates:
            adjusted_cache = []
            for seg, score in global_cache_candidates:
                cached_face_score = getattr(seg, 'face_score', None)
                if cached_face_score is not None:
                    if self.face_preference == "more":
                        boost = cached_face_score * 0.3
                        adjusted_cache.append((seg, min(1.0, score + boost)))
                    elif self.face_preference == "none":
                        boost = (1.0 - cached_face_score) * 0.3
                        adjusted_cache.append((seg, min(1.0, score + boost)))
                    else:
                        adjusted_cache.append((seg, score))
                else:
                    adjusted_cache.append((seg, score))
            global_cache_candidates = adjusted_cache

        # Merge and sort
        result = current_project_candidates + global_cache_candidates
        result.sort(key=lambda x: x[1], reverse=True)

        return result, True

    def apply_location_filter(
        self,
        vo_segment: 'SRTSegment',
        candidates: List[Tuple['SRTSegment', float]],
        segment_idx: int = 0
    ) -> Tuple[List[Tuple['SRTSegment', float]], bool, str]:
        """
        Apply location-based filtering using LocationMatcher.

        Args:
            vo_segment: Voiceover segment
            candidates: List of (segment, similarity) tuples
            segment_idx: Index of voiceover segment for chapter lookup

        Returns:
            Tuple of (filtered_candidates, was_applied, reason)
        """
        if not self.location_matcher:
            return candidates, False, ""

        return self.location_matcher.apply_location_filter(
            vo_segment, candidates, segment_idx,
            self.location_matching_enabled, self.location_matching_config
        )

    def apply_reuse_filter(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        reuse_tracker: 'ReuseTracker'
    ) -> List[Tuple['SRTSegment', float]]:
        """
        Apply smart reuse filtering to prevent clip overuse.

        Args:
            candidates: List of (segment, similarity) tuples
            reuse_tracker: ReuseTracker instance

        Returns:
            Filtered candidates with adjusted similarities
        """
        valid_candidates = []
        for seg, sim in candidates:
            if reuse_tracker.can_use(seg):
                adjusted_sim = reuse_tracker.adjust_confidence(seg, sim)
                valid_candidates.append((seg, adjusted_sim))

        # Fallback: if all filtered, use top 5 with penalty
        if not valid_candidates:
            valid_candidates = [(seg, sim * 0.5) for seg, sim in candidates[:5]]

        return valid_candidates

    def apply_all_filters(
        self,
        vo_segment: 'SRTSegment',
        candidates: List[Tuple['SRTSegment', float]],
        segment_idx: int,
        reuse_tracker: 'ReuseTracker',
        cache_dir: Optional[str] = None
    ) -> FilterResult:
        """
        Apply all filters in sequence: face -> location -> reuse.

        Args:
            vo_segment: Voiceover segment
            candidates: List of (segment, similarity) tuples
            segment_idx: Index of voiceover segment
            reuse_tracker: ReuseTracker instance
            cache_dir: Optional cache directory for face detection

        Returns:
            FilterResult with filtered candidates and metadata
        """
        original_count = len(candidates)

        # 1. Face preference filter
        candidates, face_applied = self.apply_face_preference(candidates, cache_dir)

        # 2. Location filter
        candidates, location_applied, location_reason = self.apply_location_filter(
            vo_segment, candidates, segment_idx
        )

        # 3. Reuse filter
        candidates = self.apply_reuse_filter(candidates, reuse_tracker)
        reuse_applied = True  # Always applied

        return FilterResult(
            candidates=candidates,
            face_filter_applied=face_applied,
            location_filter_applied=location_applied,
            location_reason=location_reason,
            reuse_filter_applied=reuse_applied,
            original_count=original_count,
            filtered_count=len(candidates)
        )


def filter_by_context_relevance(
    candidates: List[Tuple['SRTSegment', float]],
    vo_segment: 'SRTSegment',
    threshold: float,
    video_metadata: Optional[Dict[str, Any]] = None
) -> List[Tuple['SRTSegment', float]]:
    """
    Filter candidates by context relevance using title/description/tags overlap.

    US-141-009: Pre-filters candidates before expensive embedding computation.
    Uses simple keyword overlap to quickly filter candidates that are clearly irrelevant.

    Args:
        candidates: List of (video_segment, similarity) tuples
        vo_segment: Voiceover segment to match against
        threshold: Minimum overlap score to keep candidate (0.0-1.0)
        video_metadata: Optional dict mapping source_file -> metadata dict
                       with title, description, tags keys

    Returns:
        Filtered list of candidates that meet the relevance threshold
    """
    if not candidates or threshold <= 0.0:
        return candidates

    if threshold >= 1.0:
        return []

    video_metadata = video_metadata or {}

    # If no video metadata available, return all candidates (can't filter without data)
    if not video_metadata:
        return candidates

    # Get voiceover text and keywords
    vo_text = (vo_segment.text or "").lower()
    vo_keywords = set(k.lower() for k in (vo_segment.keywords or []))
    vo_topics = set(t.lower() for t in (vo_segment.topics or []))

    # Tokenize voiceover text into words
    import re
    vo_words = set(re.findall(r'\b\w+\b', vo_text))

    filtered = []

    for seg, sim in candidates:
        # Get video metadata
        meta = video_metadata.get(seg.source_file, {})
        if isinstance(meta, dict):
            title = (meta.get('title') or "").lower()
            description = (meta.get('description') or "").lower()
            tags = [t.lower() for t in (meta.get('tags') or [])]
        else:
            title = ""
            description = ""
            tags = []

        # Compute overlap score
        score = compute_context_overlap_score(
            vo_text, vo_words, vo_keywords, vo_topics,
            title, description, tags
        )

        if score >= threshold:
            filtered.append((seg, sim))

    return filtered


def compute_context_overlap_score(
    vo_text: str,
    vo_words: set,
    vo_keywords: set,
    vo_topics: set,
    title: str,
    description: str,
    tags: List[str]
) -> float:
    """
    Compute overlap score between voiceover and video metadata.

    Scores based on:
    - Title word overlap (highest weight - 40%)
    - Description word overlap (medium weight - 30%)
    - Tag/keyword overlap (highest weight - 30%)

    Args:
        vo_text: Voiceover segment text
        vo_words: Set of words from voiceover text
        vo_keywords: Set of keywords from voiceover
        vo_topics: Set of topics from voiceover
        title: Video title
        description: Video description
        tags: List of video tags

    Returns:
        Overlap score between 0.0 and 1.0
    """
    import re

    if not vo_text:
        return 0.0

    # Title overlap (40%)
    title_words = set(re.findall(r'\b\w+\b', title.lower())) if title else set()
    title_overlap = len(title_words & vo_words) / max(len(title_words), 1) if title_words else 0.0

    # Description overlap (30%)
    desc_words = set(re.findall(r'\b\w+\b', description.lower())) if description else set()
    desc_overlap = len(desc_words & vo_words) / max(len(desc_words), 1) if desc_words else 0.0

    # Tag/keyword overlap (30%)
    tag_set = set(t.lower() for t in tags)
    keyword_overlap = len(tag_set & vo_keywords) / max(len(tag_set), 1) if tag_set else 0.0

    # Also check title/topics overlap
    title_topic_overlap = len(title_words & vo_topics) / max(len(title_words), 1) if title_words and vo_topics else 0.0
    tag_topic_overlap = len(tag_set & vo_topics) / max(len(tag_set), 1) if tag_set and vo_topics else 0.0

    # Combine with weights
    score = (
        0.4 * title_overlap +
        0.3 * desc_overlap +
        0.3 * keyword_overlap +
        0.2 * (title_topic_overlap + tag_topic_overlap) / 2.0
    )

    # Cap at 1.0
    return min(score, 1.0)


__all__ = ['CandidateFilter', 'CandidateFilterConfig', 'FilterResult', 'filter_by_context_relevance', 'compute_context_overlap_score']
