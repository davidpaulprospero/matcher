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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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


__all__ = ['CandidateFilter', 'CandidateFilterConfig', 'FilterResult']
