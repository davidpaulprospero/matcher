"""
Compilation Timeline Builder

Generates OTIO timeline with N alternative tracks for compilation videos.
All tracks are enabled and contain unique clips placed back-to-back.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import opentimelineio as otio

from ..state import DownloadedClip

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CompilationTimelineBuilder:
    """
    Build simplified N-track timeline for compilation videos.

    Unlike the main pipeline's 10-track layout with alternatives and strategies,
    this creates N equivalent tracks, each containing unique clips.
    """

    def __init__(self, frame_rate: float = 30.0):
        self.frame_rate = frame_rate

    def create_timeline(
        self,
        tracks: List[List[DownloadedClip]],
        name: str = "Compilation"
    ) -> otio.schema.Timeline:
        """
        Create OTIO timeline from arranged tracks.

        Args:
            tracks: List of N lists of clips (one per track)
            name: Timeline name

        Returns:
            OTIO Timeline object
        """
        # Create timeline with DaVinci Resolve metadata
        timeline = otio.schema.Timeline(name=name)
        timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}

        # Set global start time (required for DaVinci)
        timeline.global_start_time = otio.opentime.RationalTime(0, self.frame_rate)

        # Create video and audio tracks for each arranged track
        for i, track_clips in enumerate(tracks):
            track_num = i + 1

            # Create video track
            video_track = otio.schema.Track(
                name=f"V{track_num} - Alternative {track_num}",
                kind=otio.schema.TrackKind.Video
            )
            video_track.enabled = True  # All alternatives enabled

            # Create matching audio track
            audio_track = otio.schema.Track(
                name=f"A{track_num} - Audio",
                kind=otio.schema.TrackKind.Audio
            )
            audio_track.enabled = True

            # Add clips sequentially (no gaps)
            for clip_data in track_clips:
                clip = self._create_clip(clip_data)
                video_track.append(clip)
                audio_track.append(copy.deepcopy(clip))

            timeline.tracks.append(video_track)
            timeline.tracks.append(audio_track)

        return timeline

    def _create_clip(self, clip_data: DownloadedClip) -> otio.schema.Clip:
        """
        Create OTIO clip from DownloadedClip.

        Args:
            clip_data: Downloaded clip data

        Returns:
            OTIO Clip object
        """
        # Format file path for OTIO (forward slashes, file:// prefix for DaVinci)
        file_path = self._format_path(clip_data.file)

        # Create media reference
        media_ref = otio.schema.ExternalReference(
            target_url=file_path,
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, self.frame_rate),
                duration=otio.opentime.RationalTime(
                    int(clip_data.actual_duration * self.frame_rate),
                    self.frame_rate
                )
            )
        )

        # Create clip
        clip = otio.schema.Clip(
            name=clip_data.title[:50] if clip_data.title else clip_data.video_id,
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, self.frame_rate),
                duration=otio.opentime.RationalTime(
                    int(clip_data.actual_duration * self.frame_rate),
                    self.frame_rate
                )
            )
        )

        # Add metadata
        clip.metadata['video_id'] = clip_data.video_id
        clip.metadata['keyword'] = clip_data.keyword

        return clip

    def _format_path(self, file_path: str) -> str:
        """
        Format file path for OTIO/DaVinci compatibility.

        - Use forward slashes
        - Add file:// prefix for absolute paths
        """
        path = Path(file_path)

        # Make absolute if relative
        if not path.is_absolute():
            path = path.resolve()

        # Convert to file:// URL using pathlib's as_uri()
        # Produces file:///E:/... on Windows, file:///home/... on Linux
        if not str(path).startswith('file://'):
            path_str = path.as_uri()
        else:
            path_str = str(path)

        return path_str

    def get_timeline_stats(self, timeline: otio.schema.Timeline) -> dict:
        """Get statistics about the timeline."""
        video_tracks = [
            t for t in timeline.tracks
            if t.kind == otio.schema.TrackKind.Video
        ]

        track_stats = []
        for track in video_tracks:
            clips = [item for item in track if isinstance(item, otio.schema.Clip)]
            duration = sum(
                c.source_range.duration.to_seconds()
                for c in clips
            )
            track_stats.append({
                'name': track.name,
                'clips': len(clips),
                'duration': duration,
            })

        return {
            'name': timeline.name,
            'num_tracks': len(video_tracks),
            'tracks': track_stats,
            'total_clips': sum(t['clips'] for t in track_stats),
        }
