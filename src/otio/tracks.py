"""
Track building strategies for OTIO timeline generation.

Uses strategy pattern to eliminate the massive if/elif branching in create_timeline().
Each track type (V1-V11) has its own builder class with focused responsibility.

This breaks up the 749-line create_timeline() function into manageable, testable components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import opentimelineio as otio

from .utils import (
    create_clip_with_timewarp,
    get_confidence_color,
    get_segment_file_offset,
    _is_audio_only,
    _has_problematic_path,
    seg_start,
    seg_end,
)


if TYPE_CHECKING:
    from ..config import Config
    from ..utils import MatchResult

logger = logging.getLogger(__name__)


class TrackBuilder(ABC):
    """
    Abstract base class for track building strategies.

    Each track type (V1-V11) implements this interface to build its video/audio tracks.
    """

    # Class constant: track names (avoids recreating list on each _get_track_name call)
    TRACK_NAMES = (
        "Primary Video",
        "Alternative Video 1",
        "Alternative Video 2",
        "Secondary Diversity 1",
        "Secondary Diversity 2",
        "Secondary Diversity 3",
        "Embedding-Diversity Strategy",
        "B-roll Only",
        "Entity Images (Google)",
        "Stock Videos (Pexels/Pixabay, Generic)",
        "Entity Videos (Pexels/Pixabay)",
    )

    def __init__(
        self,
        matches: List['MatchResult'],
        config: 'Config',
        frame_rate: float,
        **kwargs  # For additional data like entity_images, segment_lookup, etc.
    ):
        self.matches = matches
        self.config = config
        self.frame_rate = frame_rate
        self.kwargs = kwargs

        # Extract common parameters from kwargs
        self.segment_lookup = kwargs.get('segment_lookup', {})
        self.resolve_video_segment = kwargs.get('resolve_video_segment')

    @abstractmethod
    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """
        Build video and audio tracks for this strategy.

        Args:
            track_idx: Track index (0=V1, 1=V2, etc.)

        Returns:
            Tuple of (video_track, audio_track)
        """
        pass

    def _get_track_name(self, track_idx: int) -> str:
        """Get track display name based on index (uses class constant)."""
        if track_idx < len(self.TRACK_NAMES):
            return self.TRACK_NAMES[track_idx]
        return f"Track {track_idx+1}"

    def _create_clip(
        self,
        match,
        segment_id: str,
        clip_label: str,
        target_duration: float,
        metadata: Optional[Dict] = None
    ) -> otio.schema.Clip:
        """
        Create a video clip with timewarp from a match.

        Args:
            match: Match object with video_segment
            segment_id: Segment ID (e.g., "S000")
            clip_label: Label for clip name (e.g., "ALT1", "SECONDARY")
            target_duration: Target duration in seconds
            metadata: Optional metadata dict

        Returns:
            OTIO clip with speed adjustment
        """
        vid_seg = match.video_segment
        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time

        # Resolve audio file to video segment (audio-first mode)
        if self.resolve_video_segment:
            resolved_source, adjusted_start = self.resolve_video_segment(vid_seg.source_file, source_start)
        else:
            resolved_source, adjusted_start = vid_seg.source_file, source_start

        # Legacy segment file offset support
        segment_offset = get_segment_file_offset(resolved_source)
        if segment_offset > 0 and resolved_source == vid_seg.source_file:
            adjusted_start = max(0, source_start - segment_offset)

        source_file = resolved_source
        source_start = adjusted_start

        # Skip audio-only files - they cause DaVinci to hang
        if _is_audio_only(source_file):
            return None

        # Skip files with problematic unicode in path - they cause DaVinci to hang
        if _has_problematic_path(source_file):
            logger.warning(f"Skipping clip with problematic path (unicode issues): {source_file}")
            return None

        # Build clip name - use just the stem (filename without extension)
        # to match the actual file name for DaVinci Resolve media linking
        clip_stem = Path(source_file).stem
        clip_name = f"[{segment_id}] {clip_label}: {clip_stem}"

        # Create clip with timewarp
        clip = create_clip_with_timewarp(
            name=clip_name,
            source_path=source_file,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=self.frame_rate,
            metadata=metadata or {}
        )

        return clip

    def _create_gap(self, duration_frames: int) -> otio.schema.Gap:
        """Create a gap clip."""
        return otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, self.frame_rate),
                duration=otio.opentime.RationalTime(duration_frames, self.frame_rate)
            )
        )

    def _add_gap_if_needed(
        self,
        video_track: otio.schema.Track,
        audio_track: otio.schema.Track,
        match_idx: int,
        vo_seg,
        first_segment_start: float,
        timeline_frames: int
    ) -> int:
        """
        Add gap clips to tracks if there's silence between voiceover segments.

        Returns:
            Updated timeline_frames position
        """
        # Check for gap before this segment (silence in voiceover)
        expected_start_frames = round((seg_start(vo_seg) - first_segment_start) * self.frame_rate)

        if expected_start_frames > timeline_frames:
            # There's a gap - insert silence/gap clips
            gap_frames = expected_start_frames - timeline_frames

            logger.debug(
                f"Segment {match_idx}: Inserting {gap_frames/self.frame_rate:.2f}s gap "
                f"(vo gap from {timeline_frames/self.frame_rate:.2f}s to {expected_start_frames/self.frame_rate:.2f}s)"
            )

            gap = self._create_gap(gap_frames)
            video_track.append(gap)
            audio_track.append(copy.deepcopy(gap))

            return expected_start_frames

        return timeline_frames


# ============================================================
# Track Strategy Factory
# ============================================================

def get_track_builder(
    track_idx: int,
    matches: List['MatchResult'],
    config: 'Config',
    frame_rate: float,
    **kwargs
) -> TrackBuilder:
    """
    Factory function to get appropriate track builder for track index.

    Args:
        track_idx: Track index (0=V1, 1=V2, ..., 10=V11)
        matches: List of match results
        config: Pipeline configuration
        frame_rate: Timeline frame rate
        **kwargs: Additional data (entity_images, entity_videos, segment_lookup, etc.)

    Returns:
        Appropriate TrackBuilder instance

    Raises:
        ValueError: If track_idx is invalid
    """
    # Map track indices to builder classes
    # NOTE: This is a stub - actual builders will be implemented in phases
    builders = {
        0: PrimaryTrackBuilder,       # V1
        1: AlternativeTrackBuilder,   # V2
        2: AlternativeTrackBuilder,   # V3
        3: DiversityTrackBuilder,     # V4
        4: DiversityTrackBuilder,     # V5
        5: DiversityTrackBuilder,     # V6
        6: EmbeddingDiversityTrackBuilder,  # V7
        7: BRollTrackBuilder,         # V8
        8: EntityImageTrackBuilder,   # V9
        9: GenericStockTrackBuilder,  # V10
        10: EntityVideoTrackBuilder,  # V11
    }

    builder_class = builders.get(track_idx)
    if not builder_class:
        raise ValueError(f"No builder for track index {track_idx}")

    return builder_class(matches, config, frame_rate, **kwargs)


# ============================================================
# Strategy Implementations (Stubs for now - will be filled in phases)
# ============================================================

class PrimaryTrackBuilder(TrackBuilder):
    """V1 - Primary video track (best matches, enabled by default)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """Build V1 (primary video) and A1 (primary audio) tracks."""
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = True
        audio_track.enabled = True

        # Process each match
        for match_idx, match_result in enumerate(self.matches):
            match = match_result.primary_match
            vo_seg = match.voiceover_segment
            vid_seg = match.video_segment

            # Target duration = voiceover segment duration
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            duration_frames = round(target_duration * self.frame_rate)

            # Source duration = video segment duration
            source_duration = vid_seg.end_time - vid_seg.start_time
            source_start = vid_seg.start_time

            # Resolve audio file to video segment (audio-first mode)
            if self.resolve_video_segment:
                resolved_source, adjusted_start = self.resolve_video_segment(vid_seg.source_file, source_start)
            else:
                resolved_source, adjusted_start = vid_seg.source_file, source_start

            # Legacy segment file offset support
            segment_offset = get_segment_file_offset(resolved_source)
            if segment_offset > 0 and resolved_source == vid_seg.source_file:
                adjusted_start = max(0, source_start - segment_offset)

            source_file = resolved_source
            source_start = adjusted_start

            # Determine clip color based on confidence
            clip_color = get_confidence_color(match.confidence)

            # Build metadata
            segment_id = f"S{match_idx:03d}"
            metadata = {
                'segment_index': match_idx,
                'segment_id': segment_id,
                'confidence': match.confidence,
                'reasoning': match.reasoning,
                'voiceover_text': vo_seg.text,
                'video_text': vid_seg.text,
                'is_keyword_match': match.is_keyword_match,
                'is_visual_match': match.is_visual_match,
                'embedding_similarity': match.embedding_similarity,
                'reuse_count': match.clip_reuse_count,
                'original_duration': source_duration,
                'target_duration': target_duration
            }

            # Create primary video clip (V1)
            clip_folder = Path(source_file).parent.name
            clip_stem = Path(source_file).stem
            v1_clip = create_clip_with_timewarp(
                name=f"[{segment_id}] {clip_folder}_{clip_stem} [{vid_seg.start_time:.1f}s]",
                source_path=source_file,
                source_start=source_start,
                source_duration=source_duration,
                target_duration=target_duration,
                frame_rate=self.frame_rate,
                metadata=metadata
            )
            v1_clip.metadata['clip_color'] = clip_color
            video_track.append(v1_clip)

            # Create primary audio clip (A1)
            a1_clip = create_clip_with_timewarp(
                name=f"[{segment_id}] Audio: {clip_folder}_{clip_stem}",
                source_path=source_file,
                source_start=source_start,
                source_duration=source_duration,
                target_duration=target_duration,
                frame_rate=self.frame_rate,
                metadata={'from_track': 'V1'}
            )
            audio_track.append(a1_clip)

        return video_track, audio_track


class AlternativeTrackBuilder(TrackBuilder):
    """V2-V3 - Alternative video tracks (2nd/3rd best matches, disabled by default)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """Build alternative tracks (V2-V3, A2-A3)."""
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        # Determine which alternative index (0 for V2, 1 for V3)
        alt_idx = track_idx - 1
        num_alternatives = self.config.output.num_alternatives if self.config.output.include_alternatives else 0

        # Process each match
        for match_idx, match_result in enumerate(self.matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            duration_frames = round(target_duration * self.frame_rate)

            if alt_idx < len(match_result.alternatives):
                alt = match_result.alternatives[alt_idx]
                segment_id = f"S{match_idx:03d}"

                # Create alternative clip
                metadata = {
                    'confidence': alt.confidence,
                    'reasoning': alt.reasoning,
                    'original_duration': alt.video_segment.end_time - alt.video_segment.start_time,
                    'target_duration': target_duration
                }

                alt_v_clip = self._create_clip(
                    alt,
                    segment_id,
                    f"ALT{alt_idx+1}",
                    target_duration,
                    metadata
                )
                # Handle audio-only files (returns None)
                if alt_v_clip is None:
                    video_track.append(self._create_gap(duration_frames))
                    audio_track.append(self._create_gap(duration_frames))
                    continue

                alt_v_clip.metadata['clip_color'] = get_confidence_color(alt.confidence)
                video_track.append(alt_v_clip)

                # Create alternative audio clip
                alt_a_clip = self._create_clip(
                    alt,
                    segment_id,
                    f"Audio ALT{alt_idx+1}",
                    target_duration,
                    {'from_track': f'V{track_idx+1}'}
                )
                if alt_a_clip:
                    audio_track.append(alt_a_clip)
                else:
                    audio_track.append(self._create_gap(duration_frames))
            else:
                # No alternative available - add gap
                video_track.append(self._create_gap(duration_frames))
                audio_track.append(self._create_gap(duration_frames))

        return video_track, audio_track


class DiversityTrackBuilder(TrackBuilder):
    """V4-V6 - Diversity-scored tracks with strict source filtering."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """Build secondary diversity tracks (V4-V6, A4-A6)."""
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        # Determine which secondary index (0 for V4, 1 for V5, 2 for V6)
        # V4=track_idx 3, V5=track_idx 4, V6=track_idx 5
        num_alternatives = self.config.output.num_alternatives if self.config.output.include_alternatives else 0
        secondary_base_idx = 1 + num_alternatives  # Track index where secondary tracks start
        sec_idx = track_idx - secondary_base_idx

        secondary_names = ["Secondary Primary", "Secondary Alt 1", "Secondary Alt 2"]
        secondary_colors = ["PURPLE", "BLUE", "TEAL"]

        # Process each match
        for match_idx, match_result in enumerate(self.matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            duration_frames = round(target_duration * self.frame_rate)

            if sec_idx < len(match_result.secondary_matches):
                sec_match = match_result.secondary_matches[sec_idx]
                segment_id = f"S{match_idx:03d}"

                # Create secondary clip
                metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': sec_match.confidence,
                    'reasoning': sec_match.reasoning,
                    'original_duration': sec_match.video_segment.end_time - sec_match.video_segment.start_time,
                    'target_duration': target_duration,
                    'is_secondary': True
                }

                sec_label = secondary_names[sec_idx] if sec_idx < len(secondary_names) else f"Secondary {sec_idx}"
                sec_v_clip = self._create_clip(
                    sec_match,
                    segment_id,
                    sec_label,
                    target_duration,
                    metadata
                )
                # Handle audio-only files (returns None)
                if sec_v_clip is None:
                    video_track.append(self._create_gap(duration_frames))
                    audio_track.append(self._create_gap(duration_frames))
                    continue

                sec_v_clip.metadata['clip_color'] = secondary_colors[sec_idx] if sec_idx < len(secondary_colors) else "GRAY"
                video_track.append(sec_v_clip)

                # Create secondary audio clip
                sec_a_clip = self._create_clip(
                    sec_match,
                    segment_id,
                    f"Audio {sec_label}",
                    target_duration,
                    {'from_track': f'V{track_idx+1}'}
                )
                if sec_a_clip:
                    audio_track.append(sec_a_clip)
                else:
                    audio_track.append(self._create_gap(duration_frames))
            else:
                # No secondary match available - add gap
                video_track.append(self._create_gap(duration_frames))
                audio_track.append(self._create_gap(duration_frames))

        return video_track, audio_track


class EmbeddingDiversityTrackBuilder(TrackBuilder):
    """V7 - Embedding-based diversity strategy."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """Build embedding diversity strategy track (V7, A7)."""
        return self._build_strategy_track(track_idx, "embedding_diversity")

    def _build_strategy_track(self, track_idx: int, strategy: str) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """Generic strategy track builder."""
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        strategy_colors = {
            "embedding_diversity": "PINK",
            "broll_only": "TEAL"
        }

        # Process each match
        for match_idx, match_result in enumerate(self.matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            duration_frames = round(target_duration * self.frame_rate)

            # Find strategy match for this strategy
            strat_match = None
            if match_result.strategy_matches:
                for sm in match_result.strategy_matches:
                    if sm.strategy == strategy:
                        strat_match = sm
                        break

            if strat_match:
                segment_id = f"S{match_idx:03d}"

                # Create strategy clip
                metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': strat_match.confidence,
                    'reasoning': strat_match.reasoning,
                    'strategy': strat_match.strategy,
                    'original_duration': strat_match.video_segment.end_time - strat_match.video_segment.start_time,
                    'target_duration': target_duration
                }

                strat_v_clip = self._create_clip(
                    strat_match,
                    segment_id,
                    strategy.upper(),
                    target_duration,
                    metadata
                )
                # Handle audio-only files (returns None)
                if strat_v_clip is None:
                    video_track.append(self._create_gap(duration_frames))
                    audio_track.append(self._create_gap(duration_frames))
                    continue

                strat_v_clip.metadata['clip_color'] = strategy_colors.get(strategy, "GRAY")
                video_track.append(strat_v_clip)

                # Create strategy audio clip
                strat_a_clip = self._create_clip(
                    strat_match,
                    segment_id,
                    f"Audio {strategy.upper()}",
                    target_duration,
                    {'from_track': f'V{track_idx+1}', 'strategy': strategy}
                )
                if strat_a_clip:
                    audio_track.append(strat_a_clip)
                else:
                    audio_track.append(self._create_gap(duration_frames))
            else:
                # No strategy match available - add gap
                video_track.append(self._create_gap(duration_frames))
                audio_track.append(self._create_gap(duration_frames))

        return video_track, audio_track


class BRollTrackBuilder(TrackBuilder):
    """V8 - B-roll only track (no faces detected)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        """Build B-roll strategy track (V8, A8)."""
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        # Process each match
        for match_idx, match_result in enumerate(self.matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            duration_frames = round(target_duration * self.frame_rate)

            # Find B-roll strategy match
            strat_match = None
            if match_result.strategy_matches:
                for sm in match_result.strategy_matches:
                    if sm.strategy == "broll_only":
                        strat_match = sm
                        break

            if strat_match:
                segment_id = f"S{match_idx:03d}"

                # Create B-roll clip
                metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': strat_match.confidence,
                    'reasoning': strat_match.reasoning,
                    'strategy': strat_match.strategy,
                    'original_duration': strat_match.video_segment.end_time - strat_match.video_segment.start_time,
                    'target_duration': target_duration
                }

                broll_v_clip = self._create_clip(
                    strat_match,
                    segment_id,
                    "BROLL_ONLY",
                    target_duration,
                    metadata
                )
                # Handle audio-only files (returns None)
                if broll_v_clip is None:
                    video_track.append(self._create_gap(duration_frames))
                    audio_track.append(self._create_gap(duration_frames))
                    continue

                broll_v_clip.metadata['clip_color'] = "TEAL"
                video_track.append(broll_v_clip)

                # Create B-roll audio clip
                broll_a_clip = self._create_clip(
                    strat_match,
                    segment_id,
                    "Audio BROLL_ONLY",
                    target_duration,
                    {'from_track': f'V{track_idx+1}', 'strategy': 'broll_only'}
                )
                if broll_a_clip:
                    audio_track.append(broll_a_clip)
                else:
                    audio_track.append(self._create_gap(duration_frames))
            else:
                # No B-roll match available - add gap
                video_track.append(self._create_gap(duration_frames))
                audio_track.append(self._create_gap(duration_frames))

        return video_track, audio_track


class EntityImageTrackBuilder(TrackBuilder):
    """V9 - Entity images track (Google/Bing stills)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # Delegate to unified entity builder
        from .entities import _add_entity_images_to_track

        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        # Get entity images from kwargs
        entity_images = self.kwargs.get('entity_images')
        time_scale_factor = self.kwargs.get('time_scale_factor', 1.0)
        if entity_images:
            _add_entity_images_to_track(video_track, entity_images, self.matches, self.frame_rate, self.config, time_scale_factor)

        return video_track, audio_track


class GenericStockTrackBuilder(TrackBuilder):
    """V10 - Generic stock videos track (Pexels/Pixabay)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        from .entities import _add_stock_videos_to_track

        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        stock_videos = self.kwargs.get('stock_videos')
        time_scale_factor = self.kwargs.get('time_scale_factor', 1.0)
        if stock_videos:
            _add_stock_videos_to_track(video_track, stock_videos, self.matches, self.frame_rate, self.config, time_scale_factor)

        return video_track, audio_track


class EntityVideoTrackBuilder(TrackBuilder):
    """V11 - Entity videos track (Pexels/Pixabay)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # Delegate to unified entity builder
        from .entities import _add_entity_videos_to_track

        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        # Get entity videos from kwargs
        entity_videos = self.kwargs.get('entity_videos')
        time_scale_factor = self.kwargs.get('time_scale_factor', 1.0)
        if entity_videos:
            _add_entity_videos_to_track(video_track, entity_videos, self.matches, self.frame_rate, self.config, time_scale_factor)

        return video_track, audio_track


# ============================================================
# Clip Budget Tracker
# ============================================================

class ClipBudgetTracker:
    """
    Estimates total clip count before timeline assembly and auto-disables
    tracks when the count would exceed DaVinci Resolve's safe limits.

    DaVinci Resolve OTIO import hangs when total clips exceed ~3130.
    This tracker runs *before* track building to proactively reduce
    track count rather than just logging warnings after the fact.

    Formula: clips = segments * (1 + num_alternatives + num_secondary + len(strategy_tracks) + entity_tracks)
    """

    # Import thresholds from timeline module to stay in sync
    WARNING_THRESHOLD = 2500
    ERROR_THRESHOLD = 3000

    def __init__(
        self,
        match_count: int,
        num_alternatives: int,
        num_secondary: int,
        strategy_track_names: list,
        has_entity_images: bool = False,
        has_stock_videos: bool = False,
        has_entity_videos: bool = False,
    ):
        self.match_count = match_count
        self.num_alternatives = num_alternatives
        self.num_secondary = num_secondary
        self.strategy_track_names = list(strategy_track_names)
        self.has_entity_images = has_entity_images
        self.has_stock_videos = has_stock_videos
        self.has_entity_videos = has_entity_videos
        self._actions_taken: list = []

    def estimate_clips(self) -> int:
        """
        Estimate total clip count based on match count and enabled tracks.

        Each segment produces one clip per enabled track.
        Entity/stock tracks (V9, V10, V11) contribute roughly 1 clip per segment when populated.
        """
        track_count = 1  # V1 primary always present
        track_count += self.num_alternatives  # V2-V3
        track_count += self.num_secondary  # V4-V6
        track_count += len(self.strategy_track_names)  # V7-V8
        if self.has_entity_images:
            track_count += 1  # V9
        if self.has_stock_videos:
            track_count += 1  # V10
        if self.has_entity_videos:
            track_count += 1  # V11

        return self.match_count * track_count

    def check_and_adjust(self, config: 'Config') -> int:
        """
        Check estimated clip count against thresholds and auto-adjust config.

        Modifies config.output in-place when thresholds are exceeded:
        - WARNING (>=2500): Logs recommendation to disable V4-V8
        - ERROR (>=3000): Auto-disables strategy tracks (V7-V8)

        Args:
            config: Pipeline configuration (may be modified in-place)

        Returns:
            Final estimated clip count after adjustments
        """
        estimated = self.estimate_clips()

        if estimated >= self.ERROR_THRESHOLD:
            # Auto-disable strategy tracks (V7-V8) to bring count down
            logger.warning(
                f"ClipBudgetTracker: Estimated {estimated} clips exceeds error threshold "
                f"({self.ERROR_THRESHOLD}). Auto-disabling strategy tracks (V7-V8) to reduce clip count."
            )
            self._actions_taken.append(
                f"Auto-disabled strategy tracks (V7-V8): {estimated} clips >= {self.ERROR_THRESHOLD} threshold"
            )

            # Modify config to disable strategy tracks
            config.output.include_strategy_tracks = False
            self.strategy_track_names = []

            # Recalculate after adjustment
            estimated = self.estimate_clips()
            logger.info(f"ClipBudgetTracker: After disabling strategy tracks, estimated {estimated} clips")

        elif estimated >= self.WARNING_THRESHOLD:
            logger.warning(
                f"ClipBudgetTracker: Estimated {estimated} clips approaching DaVinci limit. "
                f"Consider disabling V4-V8 tracks (secondary/strategy) to reduce clip count."
            )
            self._actions_taken.append(
                f"Warning: {estimated} clips >= {self.WARNING_THRESHOLD} threshold"
            )

        return estimated

    @property
    def actions_taken(self) -> list:
        """Return list of actions taken by the tracker."""
        return list(self._actions_taken)
