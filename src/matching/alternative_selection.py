"""
Alternative and secondary match selection.

Extracted from tiered_matcher.py for single responsibility: selecting alternative
video segments for tracks V2-V6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from ..utils import SRTSegment, SceneInfo, AlternativeMatch


@dataclass
class AlternativeSelectionConfig:
    """Configuration for alternative selection."""
    num_alternatives: int = 2  # Number of alternatives for V2-V3
    num_secondary: int = 3  # Number of secondary matches for V4-V6


class AlternativeSelector:
    """
    Selects alternative video segments for multi-track output.

    Provides candidates for V2-V3 (alternatives) and V4-V6 (secondary matches)
    using diversity-based selection to maximize variety.
    """

    def __init__(self, config: AlternativeSelectionConfig):
        """Initialize with configuration."""
        self.config = config

    @classmethod
    def from_output_config(cls, output_config) -> 'AlternativeSelector':
        """Create AlternativeSelector from output config section."""
        config = AlternativeSelectionConfig(
            num_alternatives=getattr(output_config, 'num_alternatives', 2),
            num_secondary=3  # Fixed for V4-V6
        )
        return cls(config)

    def get_alternatives(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        scenes: Optional[Dict[str, List['SceneInfo']]],
        primary_match: Optional['SRTSegment'] = None,
        get_scene_fn=None
    ) -> List['AlternativeMatch']:
        """
        Get alternative matches for V2-V3.

        Prefers different video sources from primary match.

        Args:
            candidates: List of (video_segment, similarity) tuples
            scenes: Dict mapping video path to scene list
            primary_match: The primary selected segment (to exclude its source)
            get_scene_fn: Function to get scene for a segment

        Returns:
            List of AlternativeMatch objects for V2-V3
        """
        from ..utils import AlternativeMatch

        alternatives = []
        used_sources: Set[str] = set()

        if primary_match and primary_match.source_file:
            used_sources.add(primary_match.source_file)

        # First pass: prefer different sources
        for seg, sim in candidates:
            if len(alternatives) >= self.config.num_alternatives:
                break

            if seg.source_file not in used_sources:
                scene = get_scene_fn(seg, scenes) if get_scene_fn else None
                alternatives.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim,
                    reasoning=f"Alternative (different source: {Path(seg.source_file).stem})"
                ))
                used_sources.add(seg.source_file)

        # Second pass: fill remaining slots
        if len(alternatives) < self.config.num_alternatives:
            for seg, sim in candidates:
                if len(alternatives) >= self.config.num_alternatives:
                    break

                if any(alt.video_segment.source_file == seg.source_file and
                       alt.video_segment.start_time == seg.start_time for alt in alternatives):
                    continue

                scene = get_scene_fn(seg, scenes) if get_scene_fn else None
                alternatives.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.9,
                    reasoning="Alternative (fallback)"
                ))

        return alternatives

    def get_secondary_matches(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        scenes: Optional[Dict[str, List['SceneInfo']]],
        excluded_video_files: Set[str],
        primary_segment: Optional['SRTSegment'] = None,
        alt_segments: Optional[List['SRTSegment']] = None,
        get_scene_fn=None
    ) -> List['AlternativeMatch']:
        """
        Get secondary matches for V4-V6.

        Three-pass approach:
        1. Different video files from V1-V3, different from each other
        2. Different video files from V1-V3, allow same source within V4-V6
        3. Allow same video file as V1-V3 but different segment

        Args:
            candidates: List of (video_segment, similarity) tuples
            scenes: Dict mapping video path to scene list
            excluded_video_files: Video files used by V1-V3
            primary_segment: Primary match segment
            alt_segments: Alternative segments (V2-V3)
            get_scene_fn: Function to get scene for a segment

        Returns:
            List of AlternativeMatch objects for V4-V6
        """
        from ..utils import AlternativeMatch

        secondary = []
        used_sources: Set[str] = set()
        num_secondary = self.config.num_secondary

        # Collect exact segments used by V1-V3
        used_segments: Set[Tuple[str, float]] = set()
        if primary_segment:
            used_segments.add((primary_segment.source_file, primary_segment.start_time))
        if alt_segments:
            for seg in alt_segments:
                if seg:
                    used_segments.add((seg.source_file, seg.start_time))

        # First pass: Different video files, different from each other
        for seg, sim in candidates:
            if len(secondary) >= num_secondary:
                break

            if seg.source_file in excluded_video_files:
                continue

            if seg.source_file in used_sources:
                continue

            scene = get_scene_fn(seg, scenes) if get_scene_fn else None
            position = len(secondary)
            label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"

            secondary.append(AlternativeMatch(
                video_segment=seg,
                video_scene=scene,
                confidence=sim,
                reasoning=f"{label} (source: {Path(seg.source_file).stem})"
            ))
            used_sources.add(seg.source_file)

        # Second pass: Different video files, allow same source within V4-V6
        if len(secondary) < num_secondary:
            for seg, sim in candidates:
                if len(secondary) >= num_secondary:
                    break

                if seg.source_file in excluded_video_files:
                    continue

                if any(s.video_segment.source_file == seg.source_file and
                       s.video_segment.start_time == seg.start_time for s in secondary):
                    continue

                scene = get_scene_fn(seg, scenes) if get_scene_fn else None
                position = len(secondary)
                label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"

                secondary.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.95,
                    reasoning=f"{label} (same source ok)"
                ))

        # Third pass: Allow same video file but different segment
        if len(secondary) < num_secondary:
            for seg, sim in candidates:
                if len(secondary) >= num_secondary:
                    break

                seg_key = (seg.source_file, seg.start_time)
                if seg_key in used_segments:
                    continue

                if any(s.video_segment.source_file == seg.source_file and
                       s.video_segment.start_time == seg.start_time for s in secondary):
                    continue

                scene = get_scene_fn(seg, scenes) if get_scene_fn else None
                position = len(secondary)
                label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"

                secondary.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.85,
                    reasoning=f"{label} (fallback - different segment)"
                ))

        return secondary
