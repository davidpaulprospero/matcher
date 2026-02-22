"""
Alternative and secondary match selection.

Extracted from tiered_matcher.py for single responsibility: selecting alternative
video segments for tracks V2-V6.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from ..utils import SRTSegment, SceneInfo, AlternativeMatch

logger = logging.getLogger(__name__)


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


# =============================================================================
# INTER-TRACK EMBEDDING DIVERSITY (US-84-009)
# =============================================================================


def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two vectors.

    Returns 0.0 for identical vectors, up to 2.0 for opposite vectors.
    Returns 1.0 if either vector is zero-length.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    # Clamp to handle floating point errors
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


@dataclass
class InterTrackDiversityResult:
    """Per-segment inter-track embedding diversity measurement."""
    segment_index: int
    pairwise_distances: Dict[str, float]  # e.g. {"V1-V2": 0.3, "V1-V3": 0.5, "V2-V3": 0.4}
    avg_distance: float  # Mean of pairwise distances
    is_low_diversity: bool  # True when avg_distance < threshold


@dataclass
class TrackEmbeddingDiversityReport:
    """Aggregate inter-track embedding diversity across all segments."""
    segment_results: List[InterTrackDiversityResult]
    track_diversity_score: float  # Mean pairwise distance across all segments
    low_diversity_segments_count: int  # Number of segments flagged as low diversity
    total_segments: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for checkpoint/quality report."""
        return {
            'track_diversity_score': float(self.track_diversity_score),
            'low_diversity_segments_count': self.low_diversity_segments_count,
            'total_segments': self.total_segments,
        }


def compute_inter_track_embedding_diversity(
    results: List[Any],
    get_embedding: Callable[[str, float], Optional[List[float]]],
    min_distance_threshold: float = 0.15,
) -> TrackEmbeddingDiversityReport:
    """Compute pairwise embedding cosine distance between V1, V2, V3 per segment.

    After alternative selection, measures how semantically different the chosen
    alternatives really are. Low diversity means V1/V2/V3 are similar despite
    being from different sources (e.g., all cityscape variants).

    Args:
        results: List of MatchResult objects (post alternative selection).
        get_embedding: Callback that returns embedding vector for a segment,
            given (source_file, start_time). Returns None if embedding unavailable.
        min_distance_threshold: Minimum avg pairwise distance for acceptable diversity.

    Returns:
        TrackEmbeddingDiversityReport with per-segment and aggregate metrics.
    """
    segment_results: List[InterTrackDiversityResult] = []
    total_distances: List[float] = []
    low_count = 0

    for idx, result in enumerate(results):
        # Extract V1, V2, V3 segments
        track_segments: Dict[str, Any] = {}

        # V1 - primary match
        if hasattr(result, 'primary_match') and result.primary_match:
            v1_seg = getattr(result.primary_match, 'video_segment', None)
            if v1_seg:
                track_segments['V1'] = v1_seg

        # V2, V3 - alternatives
        alts = getattr(result, 'alternatives', []) or []
        for i, track_name in enumerate(['V2', 'V3']):
            if i < len(alts) and alts[i]:
                seg = getattr(alts[i], 'video_segment', None)
                if seg:
                    track_segments[track_name] = seg

        # Need at least 2 tracks for pairwise comparison
        if len(track_segments) < 2:
            continue

        # Get embeddings for each track's segment
        track_embeddings: Dict[str, List[float]] = {}
        for track_name, seg in track_segments.items():
            src = getattr(seg, 'source_file', '')
            start = getattr(seg, 'start_time', 0.0)
            emb = get_embedding(src, start)
            if emb is not None:
                track_embeddings[track_name] = emb

        # Need at least 2 embeddings for pairwise comparison
        if len(track_embeddings) < 2:
            continue

        # Compute pairwise distances
        pairwise: Dict[str, float] = {}
        track_names = sorted(track_embeddings.keys())
        for i in range(len(track_names)):
            for j in range(i + 1, len(track_names)):
                pair_key = f"{track_names[i]}-{track_names[j]}"
                dist = cosine_distance(
                    track_embeddings[track_names[i]],
                    track_embeddings[track_names[j]],
                )
                pairwise[pair_key] = dist

        if not pairwise:
            continue

        avg_dist = sum(pairwise.values()) / len(pairwise)
        is_low = avg_dist < min_distance_threshold
        if is_low:
            low_count += 1

        total_distances.append(avg_dist)
        segment_results.append(InterTrackDiversityResult(
            segment_index=idx,
            pairwise_distances=pairwise,
            avg_distance=avg_dist,
            is_low_diversity=is_low,
        ))

    overall_score = sum(total_distances) / len(total_distances) if total_distances else 0.0

    return TrackEmbeddingDiversityReport(
        segment_results=segment_results,
        track_diversity_score=overall_score,
        low_diversity_segments_count=low_count,
        total_segments=len(results),
    )


def log_inter_track_diversity(
    report: TrackEmbeddingDiversityReport,
    min_distance_threshold: float = 0.15,
) -> None:
    """Log inter-track embedding diversity results.

    Logs aggregate metrics at INFO level, warnings when diversity is low.
    """
    logger.info("=== Inter-Track Embedding Diversity ===")
    logger.info(f"  Track diversity score: {report.track_diversity_score:.3f}")
    logger.info(f"  Low diversity segments: {report.low_diversity_segments_count}/{report.total_segments}")

    if report.track_diversity_score < min_distance_threshold:
        logger.warning(
            "LOW TRACK DIVERSITY: avg_embedding_distance(V1,V2,V3) = %.3f < %.3f threshold -- "
            "alternatives may look visually similar despite different sources",
            report.track_diversity_score,
            min_distance_threshold,
        )

    logger.info("=======================================")
