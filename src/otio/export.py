"""
Timeline export functions for various formats.

Migrated from otio_builder.py - provides OTIO, EDL export functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import opentimelineio as otio

from .reporting import print_timeline_statistics

if TYPE_CHECKING:
    from ..utils import MatchResult

logger = logging.getLogger(__name__)


def save_timeline(timeline: otio.schema.Timeline, output_path: str):
    """Save timeline to OTIO file"""
    otio.adapters.write_to_file(timeline, output_path)
    logger.info(f"Saved timeline to {output_path}")


def save_timeline_split(timeline: otio.schema.Timeline, output_path: str, num_parts: int = 3, clips_per_file: int = 10) -> List[str]:
    """
    Save timeline as:
    - Track-specific files (V1, V2, V3, etc.)
    - Full timeline with all tracks

    Args:
        timeline: The full OTIO timeline
        output_path: Base output path
        num_parts: Ignored (kept for backwards compatibility)
        clips_per_file: Ignored (kept for backwards compatibility)

    Returns:
        List of paths to generated OTIO files
    """
    base_path = Path(output_path).with_suffix('')
    generated_paths = []

    # Get all tracks
    video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

    def safe_name(name):
        """Create filesystem-safe name"""
        safe = name.replace(' ', '_').replace('-', '_').replace('/', '_')
        return ''.join(c for c in safe if c.isalnum() or c == '_')[:15]

    # =========================================================================
    # 1. TRACK-SPECIFIC FILES (one per video track)
    # =========================================================================
    # Get frame rate from original timeline if available
    frame_rate = 30.0  # Default
    if hasattr(timeline, 'global_start_time') and timeline.global_start_time:
        frame_rate = timeline.global_start_time.rate

    for i, track in enumerate(video_tracks):
        track_timeline = otio.schema.Timeline(name=track.name)
        # Add Resolve_OTIO metadata (required for DaVinci import)
        track_timeline.metadata['Resolve_OTIO'] = {
            'Resolve OTIO Meta Version': '1.0'
        }
        # CRITICAL: Set valid global_start_time to prevent DaVinci Resolve hang
        track_timeline.global_start_time = otio.opentime.RationalTime(
            int(3600 * frame_rate),  # 1 hour in frames
            frame_rate
        )

        cloned_track = track.clone()
        cloned_track.enabled = True  # Enable track in standalone file
        track_timeline.tracks.append(cloned_track)

        clip_count = sum(1 for item in track if isinstance(item, otio.schema.Clip))
        track_path = f"{base_path}_V{i+1}_{safe_name(track.name)}.otio"
        otio.adapters.write_to_file(track_timeline, track_path)
        generated_paths.append(track_path)
        logger.info(f"Saved V{i+1}: {track_path} ({clip_count} clips)")

    # Voiceover track
    for track in audio_tracks:
        if 'voiceover' in track.name.lower() or 'a8' in track.name.lower():
            vo_timeline = otio.schema.Timeline(name="Voiceover")
            # Add Resolve_OTIO metadata (required for DaVinci import)
            vo_timeline.metadata['Resolve_OTIO'] = {
                'Resolve OTIO Meta Version': '1.0'
            }
            # CRITICAL: Set valid global_start_time to prevent DaVinci Resolve hang
            vo_timeline.global_start_time = otio.opentime.RationalTime(
                int(3600 * frame_rate),  # 1 hour in frames
                frame_rate
            )

            cloned_vo = track.clone()
            cloned_vo.enabled = True  # Enable track in standalone file
            vo_timeline.tracks.append(cloned_vo)

            vo_path = f"{base_path}_A8_voiceover.otio"
            otio.adapters.write_to_file(vo_timeline, vo_path)
            generated_paths.append(vo_path)
            logger.info(f"Saved A8 voiceover: {vo_path}")
            break

    # =========================================================================
    # 2. FULL TIMELINE (all tracks)
    # =========================================================================
    full_path = f"{base_path}_FULL.otio"
    otio.adapters.write_to_file(timeline, full_path)
    generated_paths.append(full_path)
    logger.info(f"Saved FULL timeline: {full_path}")

    logger.info(f"Generated {len(generated_paths)} OTIO files total")

    # Print timeline statistics checklist
    print_timeline_statistics(timeline)

    return generated_paths


def save_timeline_as_edl(matches: List['MatchResult'], output_path: str, frame_rate: float = 30.0,
                         timeline_start_tc: str = "01:00:00:00", entities: List[dict] = None):
    """
    Save markers as EDL for DaVinci Resolve TIMELINE markers.

    These markers are placed at timeline positions, not on clips.
    Import into DaVinci: File > Import > Timeline (select EDL, check "Import markers")

    Marker colors:
    - Green/Cyan/Yellow/Orange/Red: Confidence tiers
    - Pink: Entity markers (TEXT OVERLAY needed)

    Args:
        matches: List of MatchResult from matching stage
        output_path: Path to EDL file
        frame_rate: Timeline frame rate
        timeline_start_tc: Timeline start timecode (default 01:00:00:00)
        entities: Optional list of entity dictionaries with 'name' and 'position_sec' keys

    Returns:
        Path to the generated EDL file
    """
    from .utils import get_confidence_color

    # Parse timeline start timecode to frame offset
    tc_parts = timeline_start_tc.split(':')
    start_frame_offset = (
        int(tc_parts[0]) * 3600 +
        int(tc_parts[1]) * 60 +
        int(tc_parts[2])
    ) * int(frame_rate) + int(tc_parts[3])

    def frames_to_tc(frames: int) -> str:
        """Convert frame count to timecode string."""
        total_frames = frames + start_frame_offset
        fps = int(frame_rate)

        frame_in_sec = total_frames % fps
        total_secs = total_frames // fps
        secs = total_secs % 60
        total_mins = total_secs // 60
        mins = total_mins % 60
        hours = total_mins // 60

        return f"{hours:02d}:{mins:02d}:{secs:02d}:{frame_in_sec:02d}"

    # DaVinci Resolve EDL color mapping
    color_to_edl = {
        "GREEN": "Mint",
        "CYAN": "Cyan",
        "YELLOW": "Yellow",
        "ORANGE": "Orange",
        "RED": "Red",
        "PINK": "Pink"
    }

    edl_lines = []
    edl_lines.append("TITLE: Matched Footage Markers")
    edl_lines.append(f"FCM: NON-DROP FRAME")
    edl_lines.append("")

    current_frame = 0
    marker_num = 1

    # Add segment markers
    for match_idx, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment

        # Calculate segment duration
        target_duration = vo_seg.end_time - vo_seg.start_time
        duration_frames = int(target_duration * frame_rate)

        # Marker at segment start
        marker_tc = frames_to_tc(current_frame)

        # Get confidence color
        clip_color = get_confidence_color(match.confidence)
        edl_color = color_to_edl.get(clip_color, "Blue")

        # Marker name (truncate voiceover text to 40 chars)
        marker_name = vo_seg.text[:40].strip()
        if len(vo_seg.text) > 40:
            marker_name += "..."

        # EDL marker entry format (matches DaVinci export exactly)
        edl_lines.append(f"{marker_num:03d}  BL       V     C        {marker_tc} {marker_tc} {marker_tc} {marker_tc}")
        edl_lines.append(f"* FROM CLIP NAME: {marker_name}")
        edl_lines.append(f"|C:ResolveColorBlue |M:{marker_name} |D:1")
        edl_lines.append("")

        marker_num += 1
        current_frame += duration_frames

    # Add entity markers if provided
    if entities:
        for entity in entities:
            entity_name = entity.get('name', 'Unknown')
            position_sec = entity.get('position_sec', 0.0)
            entity_frames = int(position_sec * frame_rate)
            entity_tc = frames_to_tc(entity_frames)

            edl_lines.append(f"{marker_num:03d}  BL       V     C        {entity_tc} {entity_tc} {entity_tc} {entity_tc}")
            edl_lines.append(f"* FROM CLIP NAME: Entity: {entity_name}")
            edl_lines.append(f"|C:ResolveColorPink |M:Entity: {entity_name} |D:1")
            edl_lines.append("")

            marker_num += 1

    # Write to file
    edl_path = Path(output_path).with_suffix('.edl')
    with open(edl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(edl_lines))

    logger.info(f"Generated EDL with {marker_num - 1} markers: {edl_path}")
    print(f"  ✓ Saved EDL: {edl_path} ({marker_num - 1} markers)")

    return str(edl_path)
