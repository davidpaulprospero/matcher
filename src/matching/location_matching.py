"""
Location-aware matching and geographic filtering.

Migrated from TieredMatcher in matching.py (lines 691-832).
Handles geographic filtering based on location chapters and video locations.
"""

from typing import List, Optional, Dict, Tuple
import logging

from ..utils import SRTSegment
from ..location_service import GeoLocation, LocationService
from ..topic_extraction import LocationChapter

logger = logging.getLogger(__name__)


class LocationMatcher:
    """
    Handles geographic matching and filtering.

    Migrated from TieredMatcher location-related methods.
    """

    def __init__(self, location_service: Optional[LocationService] = None):
        """
        Initialize LocationMatcher.

        Args:
            location_service: LocationService instance for geographic comparisons
        """
        self.location_service = location_service
        self.location_chapters: Dict[int, LocationChapter] = {}
        self.video_locations: Dict[str, GeoLocation] = {}

    def set_location_chapters(self, location_chapters: List[LocationChapter]):
        """
        Set location chapters for location-aware matching.

        Migrated from TieredMatcher.set_location_chapters (lines 691-702).

        Args:
            location_chapters: List of LocationChapter objects
        """
        self.location_chapters = {}
        for lc in location_chapters:
            # Handle both dict (from checkpoint) and object forms
            if isinstance(lc, dict):
                start_idx = lc.get('start_segment_idx', 0)
                end_idx = lc.get('end_segment_idx', 0)
            else:
                start_idx = lc.start_segment_idx
                end_idx = lc.end_segment_idx
            for idx in range(start_idx, end_idx + 1):
                self.location_chapters[idx] = lc
        logger.info(f"Set {len(location_chapters)} location chapters covering {len(self.location_chapters)} segments")

    def set_video_locations(self, video_locations: Dict[str, GeoLocation]):
        """
        Set video location data for location-aware matching.

        Migrated from TieredMatcher.set_video_locations (lines 704-712).

        Args:
            video_locations: Dict mapping video paths to GeoLocation objects
        """
        self.video_locations = video_locations
        logger.info(f"Set locations for {len(video_locations)} videos")

    def _get_location_chapter(self, segment_idx: int) -> Optional[LocationChapter]:
        """
        Get LocationChapter for a segment index.

        Migrated from TieredMatcher._get_location_chapter (lines 714-716).
        """
        return self.location_chapters.get(segment_idx)

    def _get_video_location(self, video_path: str) -> Optional[GeoLocation]:
        """
        Get GeoLocation for a video path.

        Migrated from TieredMatcher._get_video_location (lines 718-720).
        """
        return self.video_locations.get(video_path)

    def apply_location_filter(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        segment_idx: int,
        location_matching_enabled: bool,
        location_matching_config
    ) -> Tuple[List[Tuple[SRTSegment, float]], bool, str]:
        """
        Apply location-based filtering to candidates.

        Migrated from TieredMatcher._apply_location_filter (lines 722-832).

        This implements:
        1. Hard filter: Remove candidates from wrong geographic region
        2. Soft fallback: If all filtered, apply penalties instead

        Args:
            vo_segment: Voiceover segment being matched
            candidates: List of (video_segment, similarity) tuples
            segment_idx: Index of voiceover segment (for location chapter lookup)
            location_matching_enabled: Whether location matching is enabled
            location_matching_config: Config with hard_filter_level, geographic_penalty, hierarchy_bonus

        Returns:
            Tuple of:
            - Filtered/adjusted candidates list
            - Boolean indicating if location filter was applied
            - Reason string describing what happened
        """
        if not location_matching_enabled or not self.location_service:
            return candidates, False, ""

        # Get location chapter for this segment
        location_chapter = self._get_location_chapter(segment_idx)
        if not location_chapter:
            return candidates, False, ""

        # Handle both dict (from checkpoint) and object forms
        if isinstance(location_chapter, dict):
            location_data = location_chapter.get('location_data')
        else:
            location_data = location_chapter.location_data

        if not location_data:
            return candidates, False, ""

        # Get location_name for logging
        if isinstance(location_chapter, dict):
            location_name = location_chapter.get('location_name', 'Unknown')
        else:
            location_name = getattr(location_chapter, 'location_name', 'Unknown')

        # Get the chapter's resolved location
        chapter_location = GeoLocation.from_dict(location_data)

        # Get config values
        lm_config = location_matching_config
        if isinstance(lm_config, dict):
            hard_filter_level = lm_config.get('hard_filter_level', 'country')
            geographic_penalty = lm_config.get('geographic_penalty', 0.4)
            hierarchy_bonus = lm_config.get('hierarchy_bonus', 0.15)
        else:
            hard_filter_level = getattr(lm_config, 'hard_filter_level', 'country')
            geographic_penalty = getattr(lm_config, 'geographic_penalty', 0.4)
            hierarchy_bonus = getattr(lm_config, 'hierarchy_bonus', 0.15)

        # Step 1: Try hard filtering
        filtered_candidates = []
        for seg, sim in candidates:
            video_location = self._get_video_location(seg.source_file)

            if video_location:
                # Check if locations match based on filter level (strictest to loosest)
                if hard_filter_level == "city":
                    matches = self.location_service.same_city(chapter_location, video_location)
                elif hard_filter_level == "state":
                    matches = self.location_service.same_region(chapter_location, video_location)
                elif hard_filter_level == "country":
                    matches = self.location_service.same_country(chapter_location, video_location)
                elif hard_filter_level == "continent":
                    matches = self.location_service.same_continent(chapter_location, video_location)
                else:
                    matches = True  # Unknown filter level, allow all

                if matches:
                    # Apply hierarchy bonus if video is in a parent/child region
                    if self.location_service.is_parent_region(chapter_location, video_location):
                        sim = min(1.0, sim + hierarchy_bonus)
                    elif self.location_service.is_parent_region(video_location, chapter_location):
                        sim = min(1.0, sim + hierarchy_bonus * 0.5)

                    filtered_candidates.append((seg, sim))
            else:
                # No location data for video - include with small penalty
                filtered_candidates.append((seg, sim - 0.05))

        # Step 2: Check if hard filter removed all candidates
        if filtered_candidates:
            # Sort by similarity
            filtered_candidates.sort(key=lambda x: -x[1])
            reason = f"location filter: {location_name} ({chapter_location.country_code})"
            logger.info(f"Location filter: kept {len(filtered_candidates)}/{len(candidates)} candidates for '{location_name}' ({chapter_location.country_name})")
            return filtered_candidates, True, reason

        # Step 3: Fallback to soft penalty mode
        logger.warning(
            f"Location filter removed all candidates for '{location_name}', "
            f"falling back to soft penalty mode"
        )

        penalized_candidates = []
        for seg, sim in candidates:
            video_location = self._get_video_location(seg.source_file)

            penalty = 0.0
            if video_location:
                if hard_filter_level == "city" and not self.location_service.same_city(chapter_location, video_location):
                    penalty = geographic_penalty
                elif hard_filter_level == "state" and not self.location_service.same_region(chapter_location, video_location):
                    penalty = geographic_penalty * 0.9
                elif hard_filter_level == "country" and not self.location_service.same_country(chapter_location, video_location):
                    penalty = geographic_penalty * 0.7
                elif hard_filter_level == "continent" and not self.location_service.same_continent(chapter_location, video_location):
                    penalty = geographic_penalty * 0.5

            adjusted_sim = max(0.0, sim - penalty)
            penalized_candidates.append((seg, adjusted_sim))

        penalized_candidates.sort(key=lambda x: -x[1])
        reason = f"location soft penalty: {location_name} (fallback)"
        return penalized_candidates, True, reason
