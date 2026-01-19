"""
Timeline export functions for various formats.

Migrated from otio_builder.py - provides OTIO, EDL export functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import opentimelineio as otio

from .reporting import print_timeline_statistics
from .utils import count_timeline_items

if TYPE_CHECKING:
    from ..config.sections.output import OutputConfig
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


def save_timeline_split_with_config(
    timeline: otio.schema.Timeline,
    output_path: str,
    config: Optional['OutputConfig'] = None
) -> Dict[str, str]:
    """
    Save timeline with automatic splitting for large timelines.

    When total items exceed config.auto_split_threshold, splits the timeline
    into config.split_parts parts to prevent DaVinci Resolve crashes.

    Args:
        timeline: The full OTIO timeline
        output_path: Base output path (e.g., 'output/timeline')
        config: OutputConfig with splitting settings

    Returns:
        Dict mapping file type to path:
        {
            'V1_Primary': 'output/timeline_V1_Primary.otio',
            'FULL': 'output/timeline_FULL.otio',
            'PART1': 'output/timeline_PART1.otio',
            ...
        }
    """
    # Use defaults if no config
    auto_split_threshold = getattr(config, 'auto_split_threshold', 3000) if config else 3000
    split_parts = getattr(config, 'split_parts', 4) if config else 4
    always_generate_full = getattr(config, 'always_generate_full', True) if config else True

    base_path = Path(output_path).with_suffix('')
    result_paths: Dict[str, str] = {}

    # Get frame rate from timeline
    frame_rate = 30.0
    if hasattr(timeline, 'global_start_time') and timeline.global_start_time:
        frame_rate = timeline.global_start_time.rate

    # =========================================================================
    # 1. TRACK-SPECIFIC FILES (existing behavior - one per video track)
    # =========================================================================
    video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

    def safe_name(name):
        """Create filesystem-safe name"""
        safe = name.replace(' ', '_').replace('-', '_').replace('/', '_')
        return ''.join(c for c in safe if c.isalnum() or c == '_')[:15]

    for i, track in enumerate(video_tracks):
        track_timeline = otio.schema.Timeline(name=track.name)
        track_timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}
        track_timeline.global_start_time = otio.opentime.RationalTime(
            int(3600 * frame_rate), frame_rate
        )

        cloned_track = track.clone()
        cloned_track.enabled = True
        track_timeline.tracks.append(cloned_track)

        clip_count = sum(1 for item in track if isinstance(item, otio.schema.Clip))
        track_path = f"{base_path}_V{i+1}_{safe_name(track.name)}.otio"
        otio.adapters.write_to_file(track_timeline, track_path)
        result_paths[f"V{i+1}_{safe_name(track.name)}"] = track_path
        logger.info(f"Saved V{i+1}: {track_path} ({clip_count} clips)")

    # Voiceover track
    for track in audio_tracks:
        if 'voiceover' in track.name.lower() or 'a8' in track.name.lower():
            vo_timeline = otio.schema.Timeline(name="Voiceover")
            vo_timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}
            vo_timeline.global_start_time = otio.opentime.RationalTime(
                int(3600 * frame_rate), frame_rate
            )

            cloned_vo = track.clone()
            cloned_vo.enabled = True
            vo_timeline.tracks.append(cloned_vo)

            vo_path = f"{base_path}_A8_voiceover.otio"
            otio.adapters.write_to_file(vo_timeline, vo_path)
            result_paths["A8_voiceover"] = vo_path
            logger.info(f"Saved A8 voiceover: {vo_path}")
            break

    # =========================================================================
    # 2. FULL TIMELINE (all tracks) - optional
    # =========================================================================
    if always_generate_full:
        full_path = f"{base_path}_FULL.otio"
        otio.adapters.write_to_file(timeline, full_path)
        result_paths["FULL"] = full_path
        logger.info(f"Saved FULL timeline: {full_path}")

    # =========================================================================
    # 3. SPLIT INTO PARTS (if threshold exceeded)
    # =========================================================================
    total_items = count_timeline_items(timeline)
    logger.info(f"Timeline has {total_items} total items across all tracks")

    if auto_split_threshold > 0 and total_items > auto_split_threshold:
        logger.info(f"Splitting timeline into {split_parts} parts (threshold: {auto_split_threshold})")

        part_paths = split_timeline_by_time_ranges(
            timeline=timeline,
            num_parts=split_parts,
            output_dir=base_path.parent,
            base_name=base_path.name
        )

        for i, path in enumerate(part_paths, 1):
            result_paths[f"PART{i}"] = path

        print(f"  ✓ Split into {len(part_paths)} parts for DaVinci Resolve compatibility")
    else:
        logger.info(f"Timeline below split threshold ({total_items} < {auto_split_threshold}), no splitting needed")

    # Print timeline statistics
    print_timeline_statistics(timeline)

    logger.info(f"Generated {len(result_paths)} OTIO files total")
    return result_paths


def split_timeline_by_time_ranges(
    timeline: otio.schema.Timeline,
    num_parts: int,
    output_dir: Path,
    base_name: str
) -> List[str]:
    """
    Split a timeline into N parts by time ranges.

    Each part contains ALL tracks (V1-V10, A1-A8) but only clips/gaps
    within its time range. This preserves track structure so parts
    can be imported independently into DaVinci Resolve.

    Args:
        timeline: Full OTIO timeline to split
        num_parts: Number of parts to create
        output_dir: Directory to save parts
        base_name: Base filename (e.g., "timeline")

    Returns:
        List of paths to generated PART files
    """
    # Get timeline duration
    duration = timeline.duration()
    if not duration or duration.value <= 0:
        logger.warning("Timeline has no duration, cannot split")
        return []

    total_frames = duration.value
    frame_rate = duration.rate

    # Calculate time boundaries for each part
    frames_per_part = total_frames / num_parts
    boundaries = []
    for i in range(num_parts):
        start_frame = i * frames_per_part
        end_frame = (i + 1) * frames_per_part
        boundaries.append((start_frame, end_frame))

    logger.info(f"Splitting timeline ({total_frames:.0f} frames @ {frame_rate}fps) into {num_parts} parts")

    # Create each part
    output_paths = []
    for part_idx, (start_frame, end_frame) in enumerate(boundaries, 1):
        part_timeline = _create_timeline_subset(
            source=timeline,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_rate=frame_rate,
            part_name=f"{timeline.name or 'Timeline'}_Part{part_idx}"
        )

        # Save part
        part_path = output_dir / f"{base_name}_PART{part_idx}.otio"
        otio.adapters.write_to_file(part_timeline, str(part_path))
        output_paths.append(str(part_path))

        part_items = count_timeline_items(part_timeline)
        start_sec = start_frame / frame_rate
        end_sec = end_frame / frame_rate
        logger.info(f"Part {part_idx}: {part_items} items, {start_sec:.1f}s - {end_sec:.1f}s -> {part_path.name}")

    return output_paths


def _create_timeline_subset(
    source: otio.schema.Timeline,
    start_frame: float,
    end_frame: float,
    frame_rate: float,
    part_name: str
) -> otio.schema.Timeline:
    """
    Create a new timeline containing only items within the time range.

    Items are included if they overlap with [start_frame, end_frame).
    Leading gaps are added to maintain proper timing within each part.

    Args:
        source: Source timeline to extract from
        start_frame: Start frame of the range (inclusive)
        end_frame: End frame of the range (exclusive)
        frame_rate: Frame rate for time calculations
        part_name: Name for the new timeline

    Returns:
        New OTIO Timeline with items in the specified range
    """
    # Create new timeline with same metadata structure
    subset = otio.schema.Timeline(name=part_name)
    subset.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}

    # Set global start time (1 hour offset for DaVinci)
    subset.global_start_time = otio.opentime.RationalTime(
        int(3600 * frame_rate), frame_rate
    )

    # Process each track
    for source_track in source.tracks:
        if not isinstance(source_track, otio.schema.Track):
            continue

        # Create matching track
        new_track = otio.schema.Track(
            name=source_track.name,
            kind=source_track.kind
        )
        new_track.enabled = source_track.enabled

        # Track position as we iterate through source items
        current_pos = 0.0
        first_item_in_range = True

        for item in source_track:
            item_duration = item.duration().value if item.duration() else 0
            item_start = current_pos
            item_end = current_pos + item_duration

            # Check if item overlaps with our range
            if item_end > start_frame and item_start < end_frame:
                # Item is at least partially in range

                # Add leading gap if this is the first item and it doesn't start at range start
                if first_item_in_range and item_start > start_frame:
                    gap_duration = item_start - start_frame
                    leading_gap = otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, frame_rate),
                            duration=otio.opentime.RationalTime(gap_duration, frame_rate)
                        )
                    )
                    new_track.append(leading_gap)

                first_item_in_range = False

                # Clone the item (includes all metadata)
                cloned = item.clone()
                new_track.append(cloned)

            current_pos = item_end

        subset.tracks.append(new_track)

    return subset


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
