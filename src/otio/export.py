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

# DaVinci Resolve EDL color mapping
# Maps internal confidence color names to DaVinci Resolve EDL color names.
# Covers all colors returned by get_confidence_color() plus entity markers.
EDL_COLOR_MAP = {
    "GREEN": "Mint",
    "CYAN": "Cyan",
    "YELLOW": "Yellow",
    "ORANGE": "Orange",
    "RED": "Red",
    "PINK": "Pink",
    "BLUE": "Blue",
    "PURPLE": "Purple",
}


def save_timeline(timeline: otio.schema.Timeline, output_path: str):
    """Save timeline to OTIO file"""
    otio.adapters.write_to_file(timeline, output_path)
    logger.info(f"Saved timeline to {output_path}")


def _split_timeline_by_segments(timeline: otio.schema.Timeline, max_segments: int,
                                 frame_rate: float = 30.0) -> List[otio.schema.Timeline]:
    """
    Split a timeline into multiple timelines based on segment count.

    Segments are determined by counting clips/gaps on the first video track (V1).
    Each track is split at the same boundaries to maintain sync.

    Args:
        timeline: The timeline to split
        max_segments: Maximum number of segments per output timeline
        frame_rate: Frame rate for the output timelines

    Returns:
        List of split timelines (may be single element if no split needed)
    """
    # Get all tracks
    video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

    if not video_tracks:
        # No video tracks, return original timeline
        return [timeline]

    # Use first video track (V1) to determine segment boundaries
    v1_track = video_tracks[0]
    segment_count = len(list(v1_track))

    if segment_count <= max_segments:
        # No split needed
        return [timeline]

    # Calculate number of parts needed
    import math
    num_parts = math.ceil(segment_count / max_segments)

    # Create split timelines
    split_timelines = []

    for part_idx in range(num_parts):
        start_segment = part_idx * max_segments
        end_segment = min((part_idx + 1) * max_segments, segment_count)

        # Create new timeline for this part
        part_timeline = otio.schema.Timeline(name=f"{timeline.name} (Part {part_idx + 1})")

        # Copy metadata (update in place since metadata is read-only attribute)
        if hasattr(timeline, 'metadata') and timeline.metadata:
            part_timeline.metadata.update(dict(timeline.metadata))

        # Add Resolve_OTIO metadata if not present
        if 'Resolve_OTIO' not in part_timeline.metadata:
            part_timeline.metadata['Resolve_OTIO'] = {
                'Resolve OTIO Meta Version': '1.0'
            }

        # Set global_start_time
        part_timeline.global_start_time = otio.opentime.RationalTime(
            int(3600 * frame_rate),  # 1 hour in frames
            frame_rate
        )

        # Split each video track
        for track in video_tracks:
            items = list(track)
            new_track = otio.schema.Track(
                name=track.name,
                kind=otio.schema.TrackKind.Video
            )
            new_track.enabled = track.enabled

            # Copy segment range for this part
            for item in items[start_segment:end_segment]:
                new_track.append(item.clone())

            part_timeline.tracks.append(new_track)

        # Split each audio track
        for track in audio_tracks:
            items = list(track)
            new_track = otio.schema.Track(
                name=track.name,
                kind=otio.schema.TrackKind.Audio
            )
            new_track.enabled = track.enabled

            # Copy segment range for this part
            for item in items[start_segment:end_segment]:
                new_track.append(item.clone())

            part_timeline.tracks.append(new_track)

        split_timelines.append(part_timeline)
        logger.debug(f"Created timeline part {part_idx + 1} with segments {start_segment + 1}-{end_segment}")

    return split_timelines


def save_timeline_split(timeline: otio.schema.Timeline, output_path: str, num_parts: int = 3,
                        clips_per_file: int = 10, max_segments_per_file: int = None) -> List[str]:
    """
    Save timeline as:
    - Track-specific files (V1, V2, V3, etc.)
    - Full timeline with all tracks
    - Optionally split full timeline by segment count

    Args:
        timeline: The full OTIO timeline
        output_path: Base output path
        num_parts: Ignored (kept for backwards compatibility)
        clips_per_file: Ignored (kept for backwards compatibility)
        max_segments_per_file: If specified, split FULL timeline into multiple parts
                               where each part contains at most this many segments.
                               Generates files like timeline_FULL_part1.otio, timeline_FULL_part2.otio

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
    # 2. FULL TIMELINE (all tracks) - optionally split by segment count
    # =========================================================================
    if max_segments_per_file is not None and max_segments_per_file > 0:
        # Split timeline by segment count
        split_timelines = _split_timeline_by_segments(timeline, max_segments_per_file, frame_rate)

        if len(split_timelines) == 1:
            # Only one part needed, save as regular FULL
            full_path = f"{base_path}_FULL.otio"
            otio.adapters.write_to_file(timeline, full_path)
            generated_paths.append(full_path)
            logger.info(f"Saved FULL timeline: {full_path}")
        else:
            # Multiple parts
            for part_num, part_timeline in enumerate(split_timelines, start=1):
                part_path = f"{base_path}_FULL_part{part_num}.otio"
                otio.adapters.write_to_file(part_timeline, part_path)
                generated_paths.append(part_path)
                logger.info(f"Saved FULL timeline part {part_num}/{len(split_timelines)}: {part_path}")
    else:
        # No splitting requested, save as single FULL file
        full_path = f"{base_path}_FULL.otio"
        otio.adapters.write_to_file(timeline, full_path)
        generated_paths.append(full_path)
        logger.info(f"Saved FULL timeline: {full_path}")

    logger.info(f"Generated {len(generated_paths)} OTIO files total")

    # Print timeline statistics checklist
    print_timeline_statistics(timeline)

    return generated_paths


def _generate_reel_name(file_path: str, max_length: int = 32) -> str:
    """
    Generate a reel name from a video file path.

    Reel names are used in CMX3600 EDL format to identify source media.
    Format: folder_filename (sanitized, truncated to max_length).

    Args:
        file_path: Path to the video file
        max_length: Maximum length for reel name (default 32 for CMX3600)

    Returns:
        Sanitized reel name, alphanumeric + underscore only, max 32 chars
    """
    if not file_path:
        return "BL"  # Black/default

    path = Path(file_path)
    folder_name = path.parent.name
    file_stem = path.stem

    # Combine folder and filename
    combined = f"{folder_name}_{file_stem}"

    # Sanitize to alphanumeric + underscore only
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in combined)

    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')

    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Strip trailing underscore after truncation
    sanitized = sanitized.rstrip('_')

    # Ensure we have something
    if not sanitized:
        return "CLIP"

    return sanitized.upper()


def save_timeline_as_edl(matches: List['MatchResult'], output_path: str, frame_rate: float = 30.0,
                         timeline_start_tc: str = "01:00:00:00", entities: List[dict] = None,
                         drop_frame: bool = False, include_reel_names: bool = False):
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
        drop_frame: If True, use drop-frame timecode (semicolons) for 29.97/59.94fps
        include_reel_names: If True, generate reel names from video file paths

    Returns:
        Path to the generated EDL file
    """
    from .utils import get_confidence_color, parse_timecode_to_frames, frames_to_tc as _frames_to_tc

    # Determine timecode separator based on drop_frame mode
    # Drop-frame uses semicolon between seconds and frames (HH:MM:SS;FF)
    # Non-drop-frame uses colon (HH:MM:SS:FF)
    tc_separator = ';' if drop_frame else ':'

    # Parse timeline start timecode to frame offset
    start_frame_offset = parse_timecode_to_frames(timeline_start_tc, frame_rate)

    def frames_to_tc(frames: int) -> str:
        """Convert frame count to timecode string."""
        return _frames_to_tc(frames, fps=frame_rate, start_frame_offset=start_frame_offset, separator=tc_separator)

    edl_lines = []
    edl_lines.append("TITLE: Matched Footage Markers")
    # FCM (Frame Count Mode) line: DROP FRAME or NON-DROP FRAME
    edl_lines.append(f"FCM: {'DROP FRAME' if drop_frame else 'NON-DROP FRAME'}")
    edl_lines.append("")

    current_frame = 0
    marker_num = 1

    # Add segment markers
    for match_idx, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment

        # Calculate segment duration
        target_duration = vo_seg.end - vo_seg.start
        duration_frames = int(target_duration * frame_rate)

        # Marker at segment start
        marker_tc = frames_to_tc(current_frame)

        # Get confidence color
        clip_color = get_confidence_color(match.confidence)
        edl_color = EDL_COLOR_MAP.get(clip_color, "White")

        # Marker name (truncate voiceover text to 40 chars)
        marker_name = vo_seg.text[:40].strip()
        if len(vo_seg.text) > 40:
            marker_name += "..."

        # Get reel name from video source file if include_reel_names is True
        if include_reel_names and hasattr(match, 'video_segment') and hasattr(match.video_segment, 'source_file'):
            reel_name = _generate_reel_name(match.video_segment.source_file)
            # Pad reel name to 8 characters for CMX3600 format
            reel_name_padded = reel_name[:8].ljust(8)
        else:
            reel_name_padded = "BL      "

        # EDL marker entry format (matches DaVinci export exactly)
        edl_lines.append(f"{marker_num:03d}  {reel_name_padded} V     C        {marker_tc} {marker_tc} {marker_tc} {marker_tc}")
        edl_lines.append(f"* FROM CLIP NAME: {marker_name}")
        edl_lines.append(f"|C:ResolveColor{edl_color} |M:{marker_name} |D:1")
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
            entity_edl_color = EDL_COLOR_MAP.get("PINK", "White")
            edl_lines.append(f"|C:ResolveColor{entity_edl_color} |M:Entity: {entity_name} |D:1")
            edl_lines.append("")

            marker_num += 1

    # Write to file
    edl_path = Path(output_path).with_suffix('.edl')
    with open(edl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(edl_lines))

    logger.info(f"Generated EDL with {marker_num - 1} markers: {edl_path}")
    print(f"  ✓ Saved EDL: {edl_path} ({marker_num - 1} markers)")

    return str(edl_path)
