"""
Track building strategies for OTIO timeline generation.

Uses strategy pattern to eliminate the massive if/elif branching in create_timeline().
Each track type (V1-V10) has its own builder class with focused responsibility.

This breaks up the 749-line create_timeline() function into manageable, testable components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import opentimelineio as otio

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import MatchResult

logger = logging.getLogger(__name__)


class TrackBuilder(ABC):
    """
    Abstract base class for track building strategies.

    Each track type (V1-V10) implements this interface to build its video/audio tracks.
    """

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
        """Get track display name based on index."""
        track_names = [
            "Primary Video",
            "Alternative Video 1",
            "Alternative Video 2",
            "Secondary Diversity 1",
            "Secondary Diversity 2",
            "Secondary Diversity 3",
            "Embedding-Diversity Strategy",
            "B-roll Only",
            "Entity Images (Google)",
            "Stock Videos (Pexels/Pixabay)",
        ]
        return track_names[track_idx] if track_idx < len(track_names) else f"Track {track_idx+1}"


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
        track_idx: Track index (0=V1, 1=V2, ..., 9=V10)
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
        9: EntityVideoTrackBuilder,   # V10
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
        # TODO: Extract from create_timeline() lines ~600-680
        # For now, return empty tracks as placeholder
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = True
        audio_track.enabled = True
        logger.warning(f"PrimaryTrackBuilder.build() not fully implemented yet")
        return video_track, audio_track


class AlternativeTrackBuilder(TrackBuilder):
    """V2-V3 - Alternative video tracks (2nd/3rd best matches, disabled by default)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # TODO: Extract from create_timeline() lines ~680-760
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False
        logger.warning(f"AlternativeTrackBuilder.build() not fully implemented yet")
        return video_track, audio_track


class DiversityTrackBuilder(TrackBuilder):
    """V4-V6 - Diversity-scored tracks with strict source filtering."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # TODO: Extract from create_timeline() lines ~760-920
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False
        logger.warning(f"DiversityTrackBuilder.build() not fully implemented yet")
        return video_track, audio_track


class EmbeddingDiversityTrackBuilder(TrackBuilder):
    """V7 - Embedding-based diversity strategy."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # TODO: Extract from create_timeline() lines ~920-1050
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False
        logger.warning(f"EmbeddingDiversityTrackBuilder.build() not fully implemented yet")
        return video_track, audio_track


class BRollTrackBuilder(TrackBuilder):
    """V8 - B-roll only track (no faces detected)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # TODO: Extract from create_timeline() lines ~1050-1150
        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False
        logger.warning(f"BRollTrackBuilder.build() not fully implemented yet")
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
        if entity_images:
            _add_entity_images_to_track(video_track, entity_images, self.matches, self.frame_rate, self.config)

        return video_track, audio_track


class EntityVideoTrackBuilder(TrackBuilder):
    """V10 - Stock videos track (Pexels/Pixabay)."""

    def build(self, track_idx: int) -> Tuple[otio.schema.Track, otio.schema.Track]:
        # Delegate to unified entity builder
        from .entities import _add_entity_videos_to_track

        video_track = otio.schema.Track(name=self._get_track_name(track_idx), kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name=f"{self._get_track_name(track_idx)} Audio", kind=otio.schema.TrackKind.Audio)
        video_track.enabled = False
        audio_track.enabled = False

        # Get entity videos from kwargs
        entity_videos = self.kwargs.get('entity_videos')
        if entity_videos:
            _add_entity_videos_to_track(video_track, entity_videos, self.matches, self.frame_rate, self.config)

        return video_track, audio_track


# ============================================================
# Notes for Future Implementation
# ============================================================

# TODO Phase 3 Day 7-8: Implement simple strategies
# - PrimaryTrackBuilder: Extract lines 600-680 from create_timeline()
# - AlternativeTrackBuilder: Extract lines 680-760
# - BRollTrackBuilder: Extract lines 1050-1150

# TODO Phase 3 Day 9-10: Implement complex strategies
# - DiversityTrackBuilder: Extract lines 760-920 (most complex - source filtering)
# - EmbeddingDiversityTrackBuilder: Extract lines 920-1050

# Current status: Base infrastructure complete, V9/V10 functional via entity builders
# Next: Extract actual track building logic from create_timeline()
