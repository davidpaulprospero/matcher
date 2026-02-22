"""
DaVinci Resolve XML generation for timeline import.

Generates FCP7 XML format compatible with DaVinci Resolve, including
media bins and timeline sequences.

Migrated from otio_builder.py - complex XML generation logic.
"""

from __future__ import annotations

import logging
import os
import uuid as uuid_module
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .utils import escape_xml, format_path_url, parse_timecode_to_frames, seg_start, seg_end, _get_media_duration
from .timeline import _validate_entity_images

if TYPE_CHECKING:
    from ..utils import MatchResult

logger = logging.getLogger(__name__)

# Standard NLE frame rates
STANDARD_NLE_RATES = {23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0}

# NTSC rates that require ntsc=TRUE in FCP7 XML and drop-frame timecode
NTSC_RATES = {23.976, 29.97, 59.94}


def _is_ntsc_rate(frame_rate: float) -> bool:
    """
    Check if a frame rate is an NTSC rate (23.976, 29.97, or 59.94 fps).

    NTSC rates use a 1000/1001 ratio (e.g., 30000/1001 = 29.97).
    These require <ntsc>TRUE</ntsc> in FCP7 XML and may use drop-frame
    timecode in EDL exports.

    Args:
        frame_rate: The frame rate to check

    Returns:
        True if the frame rate is an NTSC rate (within 0.01 tolerance)
    """
    return any(abs(frame_rate - ntsc) < 0.01 for ntsc in NTSC_RATES)


def _validate_frame_rate(frame_rate: float) -> Tuple[int, bool]:
    """
    Validate and convert frame rate for XML timebase element.

    NLE software expects integer timebases. This function:
    1. Logs a warning if the rate is non-standard
    2. Rounds to nearest integer for XML compatibility
    3. Detects NTSC rates for the <ntsc> XML element

    Args:
        frame_rate: The frame rate to validate (e.g., 29.97, 30.0)

    Returns:
        Tuple of (integer timebase, is_ntsc flag) for XML generation
    """
    # Check if it's a standard NLE rate (within 0.01 tolerance)
    is_standard = any(abs(frame_rate - std) < 0.01 for std in STANDARD_NLE_RATES)

    if not is_standard:
        logger.warning(
            f"Non-standard frame rate {frame_rate:.3f} fps. "
            f"Standard NLE rates: {sorted(STANDARD_NLE_RATES)}. "
            f"Rounding to {round(frame_rate)} for XML timebase."
        )

    is_ntsc = _is_ntsc_rate(frame_rate)

    # Round to nearest integer for XML timebase
    return round(frame_rate), is_ntsc


def _build_segment_lookup(downloaded_segments: Optional[List]) -> Dict:
    """Build video_id -> segment info lookup for audio-first mode resolution."""
    segment_lookup = {}
    if downloaded_segments:
        for seg in downloaded_segments:
            video_id = seg.video_id
            if video_id not in segment_lookup:
                segment_lookup[video_id] = []
            segment_lookup[video_id].append({
                'file': seg.file,
                'start': seg.original_start,
                'end': seg.original_end
            })
    return segment_lookup


def _resolve_video_segment(
    source_file: str,
    source_start: float,
    segment_lookup: Dict
) -> Tuple[str, float]:
    """
    Resolve audio file path to video segment path for audio-first mode.

    Args:
        source_file: Original source file (may be audio .mp3)
        source_start: Start time in the original source
        segment_lookup: Dict mapping video_id to list of segment info

    Returns:
        Tuple of (resolved_path, adjusted_start_time)
    """
    if not segment_lookup:
        return source_file, source_start

    # Extract video_id from the source file path
    stem = Path(source_file).stem
    video_id = stem

    # Check if we have segment(s) for this video
    if video_id not in segment_lookup:
        return source_file, source_start

    # Find the segment that contains this time
    segments = segment_lookup[video_id]
    for seg_info in segments:
        if seg_info['start'] <= source_start <= seg_info['end']:
            adjusted_start = source_start - seg_info['start']
            return seg_info['file'], adjusted_start

    # Fallback: find the nearest segment (closest start/end to source_start)
    if segments:
        best_seg = None
        best_distance = float('inf')
        for seg_info in segments:
            # Distance from source_start to the segment's time range
            if source_start < seg_info['start']:
                dist = seg_info['start'] - source_start
            elif source_start > seg_info['end']:
                dist = source_start - seg_info['end']
            else:
                dist = 0  # Should have been caught above
            if dist < best_distance:
                best_distance = dist
                best_seg = seg_info
        # Allow up to 60s gap between source_start and nearest segment
        if best_seg and best_distance <= 60:
            adjusted_start = max(0, source_start - best_seg['start'])
            # Clamp to segment duration
            seg_duration = best_seg['end'] - best_seg['start']
            adjusted_start = min(adjusted_start, max(0, seg_duration - 0.1))
            return best_seg['file'], adjusted_start

    return source_file, source_start


def _is_unresolved_path(resolved_path: str, original_source: str) -> bool:
    """Check if segment resolution failed (bare video ID returned unchanged)."""
    if resolved_path != original_source:
        return False  # Resolution succeeded
    # Bare video ID: no path separators, no file extension
    ext = Path(resolved_path).suffix
    has_sep = '/' in resolved_path or '\\' in resolved_path
    return not ext and not has_sep


_ffprobe_cache: Dict[str, float] = {}


def _get_segment_file_duration(
    resolved_path: str,
    segment_lookup: Dict,
    fallback_duration: float = 60.0
) -> float:
    """
    Get the physical duration of a resolved segment file in seconds.

    Uses ffprobe for actual file duration (cached), falling back to
    segment_lookup range, then fallback_duration. ffprobe is essential
    because yt-dlp cuts on keyframes, making files slightly shorter
    than the requested range — DaVinci Resolve rejects clips whose
    declared duration exceeds the actual file frames.
    """
    # Try ffprobe first (cached) — gives exact file duration
    if resolved_path in _ffprobe_cache:
        return _ffprobe_cache[resolved_path]

    if Path(resolved_path).exists():
        probed = _get_media_duration(resolved_path)
        if probed is not None:
            _ffprobe_cache[resolved_path] = probed
            return probed

    # Fall back to segment lookup range
    for video_id, segments in segment_lookup.items():
        for seg_info in segments:
            if seg_info['file'] == resolved_path:
                duration = seg_info['end'] - seg_info['start']
                _ffprobe_cache[resolved_path] = duration
                return duration
    return fallback_duration


def generate_resolve_xml_with_bins(
    matches: List['MatchResult'],
    output_path: str,
    voiceover_path: str = None,
    frame_rate: float = 30.0,
    entity_images: Dict = None,
    entity_videos: Dict = None,
    config = None,
    num_parts: int = 2,
    downloaded_segments: Optional[List] = None,
    timeline_start_tc: str = '01:00:00:00'
) -> List[str]:
    """
    Generate DaVinci Resolve compatible FCP7 XML with media bin AND timeline.

    Creates XML files that contain both:
    1. A bin with all media files
    2. A timeline sequence that references those clips

    This ensures media is properly linked when imported.

    Args:
        matches: List of match results
        output_path: Output XML path
        voiceover_path: Path to voiceover audio
        frame_rate: Timeline frame rate
        entity_images: Dict of entity name -> list of image paths
        entity_videos: Dict of entity name -> list of video paths
        config: Config object
        num_parts: Number of XML files to split into (default 2)

    Returns:
        List of paths to generated XML files
    """
    base_path = Path(output_path).with_suffix('')
    fps_int, is_ntsc = _validate_frame_rate(frame_rate)
    ntsc_str = 'TRUE' if is_ntsc else 'FALSE'

    # Build segment lookup for audio-first mode resolution
    segment_lookup = _build_segment_lookup(downloaded_segments)
    if segment_lookup:
        logger.info(f"XML export: Built segment lookup for {len(segment_lookup)} video IDs")

    # Collect ALL unique files
    all_files = {}
    file_counter = 1
    # Track original path -> resolved path mapping for timeline references
    path_resolution_map = {}

    def add_file(path: str, duration_seconds: float = 0, source_start: float = 0) -> dict:
        nonlocal file_counter
        # Resolve audio to video segment if in audio-first mode
        resolved_path, _ = _resolve_video_segment(path, source_start, segment_lookup)
        path_resolution_map[path] = resolved_path

        if resolved_path not in all_files:
            # Use physical segment file duration (from segment lookup) if available.
            # This ensures <duration> matches the actual file on disk.
            seg_dur = _get_segment_file_duration(resolved_path, segment_lookup, fallback_duration=0)
            if seg_dur > 0:
                dur_frames = int(seg_dur * frame_rate)
            elif duration_seconds > 0:
                dur_frames = int(duration_seconds * frame_rate)
            else:
                dur_frames = int(60 * frame_rate)
            all_files[resolved_path] = {
                'file_id': f"file-{file_counter}",
                'uuid': str(uuid_module.uuid4()),
                'duration_frames': dur_frames
            }
            file_counter += 1
        return all_files[resolved_path]

    # Collect files from all tracks
    # NOTE: duration must be segment duration (end - start), NOT absolute end_time.
    # After segment resolution, files are trimmed segments (e.g., 20s), not full videos.
    for match_result in matches:
        vid_seg = match_result.primary_match.video_segment
        dur = (vid_seg.end_time - vid_seg.start_time) if vid_seg.end_time > vid_seg.start_time else 60.0
        add_file(vid_seg.source_file, dur, vid_seg.start_time)

        for alt in match_result.alternatives:
            s = alt.video_segment
            dur = (s.end_time - s.start_time) if s.end_time > s.start_time else 60.0
            add_file(s.source_file, dur, s.start_time)

        for sec in getattr(match_result, 'secondary_matches', []):
            s = sec.video_segment
            dur = (s.end_time - s.start_time) if s.end_time > s.start_time else 60.0
            add_file(s.source_file, dur, s.start_time)

        for strat in match_result.strategy_matches:
            s = strat.video_segment
            dur = (s.end_time - s.start_time) if s.end_time > s.start_time else 60.0
            add_file(s.source_file, dur, s.start_time)

    if entity_images:
        # Validate entity images first
        validated_entity_images = _validate_entity_images(entity_images)
        for entity, result in validated_entity_images.items():
            # Handle both EntityImageResult objects and plain lists
            if hasattr(result, 'images'):
                images = result.images  # EntityImageResult dataclass
            elif isinstance(result, list):
                images = result
            else:
                continue
            for img_path in images:
                add_file(str(img_path), 5.0)

    if entity_videos:
        for entity, result in entity_videos.items():
            # Handle both EntityVideoResult objects and plain lists
            if hasattr(result, 'videos'):
                videos = result.videos  # EntityVideoResult dataclass
            elif isinstance(result, list):
                videos = result
            else:
                continue
            for vid_path in videos:
                add_file(str(vid_path), 30.0)

    if voiceover_path:
        vo_duration = sum(
            seg_end(m.primary_match.voiceover_segment) - seg_start(m.primary_match.voiceover_segment)
            for m in matches
        )
        add_file(voiceover_path, vo_duration)

    # Calculate total timeline duration
    total_frames = 0
    for m in matches:
        vo_seg = m.primary_match.voiceover_segment
        target_duration = seg_end(vo_seg) - seg_start(vo_seg)
        total_frames += int(target_duration * frame_rate)

    # Filter out bare video IDs that failed segment resolution (only when
    # segment_lookup is populated, meaning we have downloaded segments)
    if segment_lookup:
        unresolved_count = 0
        filtered_files = {}
        for fp, fi in all_files.items():
            ext = Path(fp).suffix
            has_sep = '/' in fp or '\\' in fp
            if not ext and not has_sep:
                unresolved_count += 1
            else:
                filtered_files[fp] = fi
        if unresolved_count:
            logger.info(f"Filtered {unresolved_count} unresolved video IDs from media bins")
        all_files = filtered_files

    # Generate complete XML with bin AND timeline
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '',
        '    <!-- Project with all media and timeline -->',
        '    <project>',
        '        <name>Matched Footage Project</name>',
        '        <children>',
        '',
        '            <!-- Media Bin -->',
        '            <bin>',
        '                <name>Footage</name>',
        '                <children>',
    ]

    # Add all files to bin with proper structure
    for file_path, file_info in all_files.items():
        # Make clip name unique by including parent folder
        folder_name = Path(file_path).parent.name
        base_name = Path(file_path).name
        unique_name = escape_xml(f"{folder_name}_{base_name}")

        path_url = format_path_url(file_path)
        file_ext = Path(file_path).suffix.lower()
        duration_frames = file_info['duration_frames']

        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf', '.m4v'}
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        audio_exts = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}

        is_video = file_ext in video_exts
        is_image = file_ext in image_exts
        is_audio = file_ext in audio_exts

        # Extensionless paths are video IDs (caption-first mode) - treat as video
        if not file_ext:
            is_video = True

        # File definition must be at clip level (direct child of <clip>), not nested in media
        xml_lines.extend([
            f'                    <clip id="masterclip-{file_info["file_id"]}">',
            f'                        <uuid>{file_info["uuid"]}</uuid>',
            f'                        <name>{unique_name}</name>',
            f'                        <duration>{duration_frames}</duration>',
            '                        <rate>',
            f'                            <timebase>{fps_int}</timebase>',
            f'                            <ntsc>{ntsc_str}</ntsc>',
            '                        </rate>',
            # File definition at clip level - this is where DaVinci looks for it
            f'                        <file id="{file_info["file_id"]}">',
            f'                            <name>{unique_name}</name>',
            f'                            <pathurl>{path_url}</pathurl>',
            '                            <rate>',
            f'                                <timebase>{fps_int}</timebase>',
            f'                                <ntsc>{ntsc_str}</ntsc>',
            '                            </rate>',
            f'                            <duration>{duration_frames}</duration>',
            '                            <timecode>',
            '                                <rate>',
            f'                                    <timebase>{fps_int}</timebase>',
            f'                                    <ntsc>{ntsc_str}</ntsc>',
            '                                </rate>',
            '                                <string>00:00:00:00</string>',
            '                                <frame>0</frame>',
            '                            </timecode>',
            '                        </file>',
            '                        <media>',
        ])

        # Add video section for all files that might be used in video tracks
        # In audio-first mode, audio files may be used in video tracks before resolution
        # DaVinci needs the video section to exist even for audio-only files
        if is_video or is_image or is_audio:
            xml_lines.extend([
                '                            <video>',
                '                                <track>',
                '                                    <clipitem>',
                f'                                        <name>{unique_name}</name>',
                f'                                        <duration>{duration_frames}</duration>',
                '                                        <start>0</start>',
                f'                                        <end>{duration_frames}</end>',
                '                                        <in>0</in>',
                f'                                        <out>{duration_frames}</out>',
                # Reference to file defined at clip level
                f'                                        <file id="{file_info["file_id"]}"/>',
                '                                    </clipitem>',
                '                                </track>',
                '                            </video>',
            ])

        if is_video or is_audio:
            xml_lines.extend([
                '                            <audio>',
                '                                <track>',
                '                                    <clipitem>',
                f'                                        <name>{unique_name}</name>',
                # Reference to file defined at clip level
                f'                                        <file id="{file_info["file_id"]}"/>',
                '                                    </clipitem>',
                '                                </track>',
                '                            </audio>',
            ])

        xml_lines.extend([
            '                        </media>',
            '                    </clip>',
        ])

    xml_lines.extend([
        '                </children>',
        '            </bin>',
        '',
        '            <!-- Timeline Sequence -->',
        '            <sequence>',
        '                <name>V1 - Primary Edit</name>',
        f'                <duration>{total_frames}</duration>',
        '                <rate>',
        f'                    <timebase>{fps_int}</timebase>',
        f'                    <ntsc>{ntsc_str}</ntsc>',
        '                </rate>',
        '                <timecode>',
        '                    <rate>',
        f'                        <timebase>{fps_int}</timebase>',
        f'                        <ntsc>{ntsc_str}</ntsc>',
        '                    </rate>',
        f'                    <string>{timeline_start_tc}</string>',
        f'                    <frame>{parse_timecode_to_frames(timeline_start_tc, frame_rate)}</frame>',
        '                </timecode>',
        '                <media>',
        '                    <video>',
        '                        <track>',
    ])

    # Add clips to V1 timeline track
    timeline_pos = 0
    for match_idx, match_result in enumerate(matches):
        vo_seg = match_result.primary_match.voiceover_segment
        vid_seg = match_result.primary_match.video_segment

        target_duration = seg_end(vo_seg) - seg_start(vo_seg)
        target_frames = int(target_duration * frame_rate)

        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time
        source_frames = int(source_duration * frame_rate)

        # Resolve audio to video segment path and adjust start time
        resolved_path, adjusted_start = _resolve_video_segment(
            vid_seg.source_file, source_start, segment_lookup
        )
        source_start_frames = int(adjusted_start * frame_rate)

        # Clamp in/out to physical segment file duration
        seg_dur_secs = _get_segment_file_duration(resolved_path, segment_lookup)
        seg_dur_frames = int(seg_dur_secs * frame_rate)
        source_start_frames = min(source_start_frames, max(0, seg_dur_frames - 1))
        clamped_out = min(source_start_frames + source_frames, seg_dur_frames)

        file_info = all_files.get(resolved_path, {})
        file_id = file_info.get('file_id', '')
        # Ensure duration >= out point so DaVinci doesn't reject the clip
        file_dur_frames = max(file_info.get('duration_frames', seg_dur_frames), clamped_out)

        # Make clip name unique by including segment ID and parent folder
        segment_id = f"S{match_idx:03d}"
        folder_name = Path(resolved_path).parent.name
        base_name = Path(resolved_path).stem
        unique_name = escape_xml(f"[{segment_id}] {folder_name}_{base_name}")

        xml_lines.extend([
            '                            <clipitem>',
            f'                                <name>{unique_name}</name>',
            f'                                <duration>{file_dur_frames}</duration>',
            f'                                <start>{timeline_pos}</start>',
            f'                                <end>{timeline_pos + target_frames}</end>',
            f'                                <in>{source_start_frames}</in>',
            f'                                <out>{clamped_out}</out>',
            f'                                <file id="{file_id}"/>',
        ])

        # Add chapter title as comment if available on the video segment
        chapter_title = getattr(vid_seg, 'chapter_title', '')
        if chapter_title:
            xml_lines.append(f'                                <comment>{escape_xml(chapter_title)}</comment>')

        # Add speed adjustment if needed
        if source_frames != target_frames and target_frames > 0:
            speed = (source_frames / target_frames) * 100
            xml_lines.extend([
                '                                <filter>',
                '                                    <effect>',
                '                                        <name>Time Remap</name>',
                '                                        <effectid>timeremap</effectid>',
                '                                        <parameter>',
                '                                            <parameterid>speed</parameterid>',
                f'                                            <value>{speed:.2f}</value>',
                '                                        </parameter>',
                '                                    </effect>',
                '                                </filter>',
            ])

        xml_lines.append('                            </clipitem>')
        timeline_pos += target_frames

    xml_lines.extend([
        '                        </track>',
    ])

    # Add V2-V8 tracks (disabled by default)
    # V2-V3: Alternatives, V4-V6: Secondary, V7-V8: Strategy
    track_configs = []

    # V2-V3: alternatives
    for alt_idx in range(2):
        track_configs.append({
            'name': f'V{alt_idx + 2} - Alternative {alt_idx + 1}',
            'get_match': lambda mr, idx=alt_idx: mr.alternatives[idx] if idx < len(mr.alternatives) else None,
        })

    # V4-V6: secondary_matches
    for sec_idx in range(3):
        track_configs.append({
            'name': f'V{sec_idx + 4} - Secondary {sec_idx + 1}',
            'get_match': lambda mr, idx=sec_idx: (
                mr.secondary_matches[idx]
                if hasattr(mr, 'secondary_matches') and mr.secondary_matches and idx < len(mr.secondary_matches)
                else None
            ),
        })

    # V7-V8: strategy_matches
    for strat_idx in range(2):
        strat_labels = ['Embedding-Diversity', 'B-roll Only']
        track_configs.append({
            'name': f'V{strat_idx + 7} - {strat_labels[strat_idx]}',
            'get_match': lambda mr, idx=strat_idx: (
                mr.strategy_matches[idx]
                if hasattr(mr, 'strategy_matches') and mr.strategy_matches and idx < len(mr.strategy_matches)
                else None
            ),
        })

    for track_config in track_configs:
        xml_lines.append('                        <track>')
        # Disabled track
        xml_lines.append('                            <enabled>FALSE</enabled>')

        alt_timeline_pos = 0
        for match_idx, match_result in enumerate(matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            target_frames = int(target_duration * frame_rate)

            alt_match = track_config['get_match'](match_result)
            if alt_match is not None:
                alt_seg = alt_match.video_segment
                alt_source_duration = alt_seg.end_time - alt_seg.start_time
                alt_source_start = alt_seg.start_time
                alt_source_frames = int(alt_source_duration * frame_rate)

                resolved_path, adjusted_start = _resolve_video_segment(
                    alt_seg.source_file, alt_source_start, segment_lookup
                )

                # Skip unresolved clips (bare video IDs with no downloaded segment)
                if not _is_unresolved_path(resolved_path, alt_seg.source_file):
                    alt_start_frames = int(adjusted_start * frame_rate)

                    # Clamp in/out to physical segment file duration
                    seg_dur_secs = _get_segment_file_duration(resolved_path, segment_lookup)
                    seg_dur_frames = int(seg_dur_secs * frame_rate)
                    alt_start_frames = min(alt_start_frames, max(0, seg_dur_frames - 1))
                    alt_out_frames = min(alt_start_frames + alt_source_frames, seg_dur_frames)

                    file_info = all_files.get(resolved_path, {})
                    file_id = file_info.get('file_id', '')
                    file_dur_frames = max(file_info.get('duration_frames', seg_dur_frames), alt_out_frames)

                    segment_id = f"S{match_idx:03d}"
                    folder_name = Path(resolved_path).parent.name
                    base_name = Path(resolved_path).stem
                    unique_name = escape_xml(f"[{segment_id}] {folder_name}_{base_name}")

                    xml_lines.extend([
                        '                            <clipitem>',
                        f'                                <name>{unique_name}</name>',
                        f'                                <duration>{file_dur_frames}</duration>',
                        f'                                <start>{alt_timeline_pos}</start>',
                        f'                                <end>{alt_timeline_pos + target_frames}</end>',
                        f'                                <in>{alt_start_frames}</in>',
                        f'                                <out>{alt_out_frames}</out>',
                        f'                                <file id="{file_id}"/>',
                    ])

                    # Add chapter title as comment if available on the alt video segment
                    alt_chapter_title = getattr(alt_seg, 'chapter_title', '')
                    if alt_chapter_title:
                        xml_lines.append(f'                                <comment>{escape_xml(alt_chapter_title)}</comment>')

                    xml_lines.append('                            </clipitem>')

            alt_timeline_pos += target_frames

        xml_lines.append('                        </track>')

    xml_lines.extend([
        '                    </video>',
    ])

    # Add voiceover audio track
    if voiceover_path:
        vo_file_info = all_files.get(voiceover_path, {})
        vo_file_id = vo_file_info.get('file_id', '')

        xml_lines.extend([
            '                    <audio>',
            '                        <track>',
            '                            <clipitem>',
            '                                <name>Voiceover</name>',
            f'                                <duration>{total_frames}</duration>',
            '                                <start>0</start>',
            f'                                <end>{total_frames}</end>',
            '                                <in>0</in>',
            f'                                <out>{total_frames}</out>',
            f'                                <file id="{vo_file_id}"/>',
            '                            </clipitem>',
            '                        </track>',
            '                    </audio>',
        ])

    xml_lines.extend([
        '                </media>',
        '            </sequence>',
        '',
        '        </children>',
        '    </project>',
        '</xmeml>',
    ])

    # Write single XML file (project with bin + timeline)
    xml_path = Path(f"{base_path}_project.xml")
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))

    logger.info(f"Saved DaVinci Resolve project XML with {len(all_files)} media files and timeline to {xml_path}")

    # Also generate media-only XMLs for manual import (split into parts)
    generated_paths = [str(xml_path)]

    if num_parts > 1:
        # Group files by actual filename to detect conflicts
        # DaVinci Resolve hangs if same filename appears multiple times in one XML
        filename_to_paths = {}
        for file_path in all_files.keys():
            filename = Path(file_path).name
            if filename not in filename_to_paths:
                filename_to_paths[filename] = []
            filename_to_paths[filename].append(file_path)

        # Separate unique files from conflicting ones
        unique_files = {}  # Files with unique filenames - safe to group
        conflicting_files = []  # Files sharing a filename - must be in separate XMLs

        for filename, paths in filename_to_paths.items():
            if len(paths) == 1:
                # Unique filename - safe
                unique_files[paths[0]] = all_files[paths[0]]
            else:
                # Same filename in multiple folders - conflict!
                logger.warning(f"Filename conflict: '{filename}' appears in {len(paths)} folders")
                for path in paths:
                    conflicting_files.append((path, all_files[path]))

        logger.info(f"Media files: {len(unique_files)} unique, {len(conflicting_files)} with filename conflicts")

        # Split unique files into parts
        file_items = list(unique_files.items())
        total_files = len(file_items)
        files_per_part = max(1, (total_files + num_parts - 1) // num_parts)

        part_idx = 0

        # Generate parts for unique files
        for part_start in range(0, total_files, files_per_part):
            part_end = min(part_start + files_per_part, total_files)
            files_subset = dict(file_items[part_start:part_end])

            if not files_subset:
                continue

            part_idx += 1
            _write_media_xml_part(
                files_subset,
                part_idx,
                f"{base_path}_media_part{part_idx}.xml",
                fps_int,
                generated_paths,
                logger,
                timeline_start_tc=timeline_start_tc,
                frame_rate=frame_rate,
                is_ntsc=is_ntsc
            )

        # Generate separate XMLs for conflicting files (one file per XML)
        for conflict_path, conflict_info in conflicting_files:
            part_idx += 1
            folder_name = Path(conflict_path).parent.name
            # Sanitize folder name for filename
            safe_folder = ''.join(c for c in folder_name if c.isalnum() or c in '_-')[:20]
            _write_media_xml_part(
                {conflict_path: conflict_info},
                part_idx,
                f"{base_path}_media_conflict_{part_idx}_{safe_folder}.xml",
                fps_int,
                generated_paths,
                logger,
                bin_name_override=f"Media - {folder_name}",
                timeline_start_tc=timeline_start_tc,
                frame_rate=frame_rate,
                is_ntsc=is_ntsc
            )

    # Log file sizes for all generated XML files
    total_size = 0
    for path in generated_paths:
        try:
            size = os.path.getsize(path)
            total_size += size
            size_kb = size / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
            logger.debug(f"[OUTPUT] XML file: {Path(path).name} ({size_str})")
        except OSError:
            pass

    total_size_kb = total_size / 1024
    total_size_str = f"{total_size_kb:.1f} KB" if total_size_kb < 1024 else f"{total_size_kb / 1024:.2f} MB"
    logger.info(f"[OUTPUT] XML export complete: {len(generated_paths)} files, {total_size_str} total")

    return generated_paths


def _write_media_xml_part(
    files_subset: dict,
    part_idx: int,
    output_path: str,
    fps_int: int,
    generated_paths: list,
    logger,
    bin_name_override: str = None,
    timeline_start_tc: str = '01:00:00:00',
    frame_rate: float = 30.0,
    is_ntsc: bool = False
):
    """Write a single media XML part file.

    DaVinci Resolve requires a specific structure for media bin imports:
    - <bin> directly under <xmeml> (no <project> wrapper)
    - <file> definition inside <clipitem>, not at clip level
    - Empty <sequence> sibling to trigger proper import
    """
    bin_name = bin_name_override or f"Media Part {part_idx}"
    ntsc_str = 'TRUE' if is_ntsc else 'FALSE'

    part_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '    <bin>',
        f'        <name>{bin_name}</name>',
        '        <children>',
    ]

    for file_path, file_info in files_subset.items():
        # Make clip name unique by including parent folder
        folder_name = Path(file_path).parent.name
        base_name = Path(file_path).name
        unique_name = escape_xml(f"{folder_name}_{base_name}")

        path_url = format_path_url(file_path)
        file_ext = Path(file_path).suffix.lower()
        duration_frames = file_info['duration_frames']

        video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf', '.m4v'}
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        audio_exts = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}
        is_video = file_ext in video_exts
        is_image = file_ext in image_exts
        is_audio_only = file_ext in audio_exts and file_ext not in video_exts

        # Extensionless paths are video IDs (caption-first mode) - treat as video
        if not file_ext:
            is_video = True

        clip_num = file_info["file_id"].replace("file-", "")

        # Skip audio-only files - DaVinci XML import doesn't handle them well
        # These are intermediates from audio-first mode, not needed for video editing
        if is_audio_only:
            continue

        # DaVinci structure: clip > rate > media > video/audio > track > clipitem > file
        part_lines.extend([
            f'            <clip id="clip-{clip_num}">',
            f'                <uuid>{file_info["uuid"]}</uuid>',
            f'                <name>{unique_name}</name>',
            '                <rate>',
            f'                    <timebase>{fps_int}</timebase>',
            f'                    <ntsc>{ntsc_str}</ntsc>',
            '                </rate>',
            '                <media>',
        ])

        # Add video track - file definition goes INSIDE clipitem
        # Note: Audio-only files should NOT have a video track
        if is_video or is_image:
            part_lines.extend([
                '                    <video>',
                '                        <track>',
                f'                            <clipitem id="clipitem-{clip_num}">',
                f'                                <name>{unique_name}</name>',
                f'                                <file id="{file_info["file_id"]}">',
                f'                                    <name>{unique_name}</name>',
                f'                                    <pathurl>{path_url}</pathurl>',
                '                                    <rate>',
                f'                                        <timebase>{fps_int}</timebase>',
                f'                                        <ntsc>{ntsc_str}</ntsc>',
                '                                    </rate>',
                f'                                    <duration>{duration_frames}</duration>',
                '                                    <timecode>',
                '                                        <rate>',
                f'                                            <timebase>{fps_int}</timebase>',
                f'                                            <ntsc>{ntsc_str}</ntsc>',
                '                                        </rate>',
                '                                        <string>00:00:00:00</string>',
                '                                        <frame>0</frame>',
                '                                    </timecode>',
                '                                </file>',
                '                            </clipitem>',
                '                        </track>',
                '                    </video>',
            ])

        # Add audio section
        # For video files: reference file by id (file already defined in video section)
        # For audio-only files: define file here (no video section)
        if is_video:
            part_lines.extend([
                '                    <audio>',
                '                        <track>',
                f'                            <clipitem id="clipitem-{clip_num}-audio">',
                f'                                <name>{unique_name}</name>',
                f'                                <file id="{file_info["file_id"]}"/>',
                '                            </clipitem>',
                '                        </track>',
                '                    </audio>',
            ])
        elif is_audio_only:
            # Audio-only files need full file definition in audio section
            part_lines.extend([
                '                    <audio>',
                '                        <track>',
                f'                            <clipitem id="clipitem-{clip_num}">',
                f'                                <name>{unique_name}</name>',
                f'                                <file id="{file_info["file_id"]}">',
                f'                                    <name>{unique_name}</name>',
                f'                                    <pathurl>{path_url}</pathurl>',
                '                                    <rate>',
                f'                                        <timebase>{fps_int}</timebase>',
                f'                                        <ntsc>{ntsc_str}</ntsc>',
                '                                    </rate>',
                f'                                    <duration>{duration_frames}</duration>',
                '                                    <timecode>',
                '                                        <rate>',
                f'                                            <timebase>{fps_int}</timebase>',
                f'                                            <ntsc>{ntsc_str}</ntsc>',
                '                                        </rate>',
                '                                        <string>00:00:00:00</string>',
                '                                        <frame>0</frame>',
                '                                    </timecode>',
                '                                </file>',
                '                            </clipitem>',
                '                        </track>',
                '                    </audio>',
            ])

        part_lines.extend([
            '                </media>',
            '            </clip>',
        ])

    part_lines.extend([
        '        </children>',
        '    </bin>',
        '',
        '    <!-- Empty sequence required for DaVinci Resolve to import bins properly -->',
        f'    <sequence id="media-seq-{part_idx}">',
        f'        <name>{bin_name} - Import Helper</name>',
        '        <rate>',
        f'            <timebase>{fps_int}</timebase>',
        f'            <ntsc>{ntsc_str}</ntsc>',
        '        </rate>',
        '        <duration>1</duration>',
        '        <timecode>',
        '            <rate>',
        f'                <timebase>{fps_int}</timebase>',
        f'                <ntsc>{ntsc_str}</ntsc>',
        '            </rate>',
        f'            <string>{timeline_start_tc}</string>',
        f'            <frame>{parse_timecode_to_frames(timeline_start_tc, frame_rate)}</frame>',
        f'            <displayformat>{"DF" if is_ntsc else "NDF"}</displayformat>',
        '        </timecode>',
        '        <media>',
        '            <video>',
        '                <track/>',
        '            </video>',
        '            <audio>',
        '                <track/>',
        '            </audio>',
        '        </media>',
        '    </sequence>',
        '</xmeml>',
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(part_lines))

    generated_paths.append(str(output_path))
    logger.info(f"Saved media XML: {output_path} ({len(files_subset)} files)")


def _add_sequence_alt_tracks(
    xml_lines: list,
    matches: List['MatchResult'],
    frame_rate: float,
    fps_int: int,
    ntsc_str: str,
    is_ntsc: bool,
    segment_lookup: Dict,
    file_ids: Dict,
    width: int,
    height: int,
):
    """Add V2-V8 disabled tracks to sequence XML."""
    track_configs = []

    # V2-V3: alternatives
    for alt_idx in range(2):
        track_configs.append({
            'name': f'V{alt_idx + 2} - Alternative {alt_idx + 1}',
            'get_match': lambda mr, idx=alt_idx: mr.alternatives[idx] if idx < len(mr.alternatives) else None,
        })

    # V4-V6: secondary_matches
    for sec_idx in range(3):
        track_configs.append({
            'name': f'V{sec_idx + 4} - Secondary {sec_idx + 1}',
            'get_match': lambda mr, idx=sec_idx: (
                mr.secondary_matches[idx]
                if hasattr(mr, 'secondary_matches') and mr.secondary_matches and idx < len(mr.secondary_matches)
                else None
            ),
        })

    # V7-V8: strategy_matches
    for strat_idx in range(2):
        strat_labels = ['Embedding-Diversity', 'B-roll Only']
        track_configs.append({
            'name': f'V{strat_idx + 7} - {strat_labels[strat_idx]}',
            'get_match': lambda mr, idx=strat_idx: (
                mr.strategy_matches[idx]
                if hasattr(mr, 'strategy_matches') and mr.strategy_matches and idx < len(mr.strategy_matches)
                else None
            ),
        })

    for track_config in track_configs:
        xml_lines.append('                <track>')
        xml_lines.append('                    <enabled>FALSE</enabled>')

        alt_timeline_pos = 0
        for match_idx, match_result in enumerate(matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = seg_end(vo_seg) - seg_start(vo_seg)
            target_frames = int(target_duration * frame_rate)

            alt_match = track_config['get_match'](match_result)
            if alt_match is not None:
                alt_seg = alt_match.video_segment
                alt_source_start = alt_seg.start_time
                alt_source_duration = alt_seg.end_time - alt_seg.start_time
                alt_source_frames = int(alt_source_duration * frame_rate)

                resolved_path, adjusted_start = _resolve_video_segment(
                    alt_seg.source_file, alt_source_start, segment_lookup
                )

                # Skip unresolved clips (bare video IDs with no downloaded segment)
                if _is_unresolved_path(resolved_path, alt_seg.source_file):
                    alt_timeline_pos += target_frames
                    continue

                in_frames = int(adjusted_start * frame_rate)
                out_frames = in_frames + alt_source_frames

                # Get or create file ID
                if resolved_path not in file_ids:
                    file_ids[resolved_path] = max(file_ids.values(), default=0) + 1
                current_file_id = file_ids[resolved_path]

                clip_name = escape_xml(Path(resolved_path).name)
                path_url = format_path_url(resolved_path)
                # file_duration must match the physical segment file on disk.
                # Clamp in/out to not exceed the segment file.
                seg_dur_secs = _get_segment_file_duration(resolved_path, segment_lookup)
                seg_dur_frames = int(seg_dur_secs * frame_rate)
                file_duration = seg_dur_frames
                in_frames = min(in_frames, max(0, seg_dur_frames - 1))
                out_frames = min(out_frames, seg_dur_frames)

                segment_id = f"S{match_idx:03d}"

                xml_lines.extend([
                    f'                    <clipitem id="{clip_name} {match_idx} {track_config["name"]}">',
                    f'                        <name>{clip_name}</name>',
                    f'                        <duration>{file_duration}</duration>',
                    '                        <rate>',
                    f'                            <timebase>{fps_int}</timebase>',
                    f'                            <ntsc>{ntsc_str}</ntsc>',
                    '                        </rate>',
                    f'                        <start>{alt_timeline_pos}</start>',
                    f'                        <end>{alt_timeline_pos + target_frames}</end>',
                    '                        <enabled>TRUE</enabled>',
                    f'                        <in>{in_frames}</in>',
                    f'                        <out>{out_frames}</out>',
                    f'                        <file id="{clip_name} {current_file_id}">',
                    f'                            <duration>{file_duration}</duration>',
                    '                            <rate>',
                    f'                                <timebase>{fps_int}</timebase>',
                    f'                                <ntsc>{ntsc_str}</ntsc>',
                    '                            </rate>',
                    f'                            <name>{clip_name}</name>',
                    f'                            <pathurl>{path_url}</pathurl>',
                    '                            <timecode>',
                    '                                <string>00:00:00:00</string>',
                    f'                                <displayformat>{"DF" if is_ntsc else "NDF"}</displayformat>',
                    '                                <rate>',
                    f'                                    <timebase>{fps_int}</timebase>',
                    f'                                    <ntsc>{ntsc_str}</ntsc>',
                    '                                </rate>',
                    '                            </timecode>',
                    '                            <media>',
                    '                                <video>',
                    f'                                    <duration>{file_duration}</duration>',
                    '                                    <samplecharacteristics>',
                    f'                                        <width>{width}</width>',
                    f'                                        <height>{height}</height>',
                    '                                    </samplecharacteristics>',
                    '                                </video>',
                    '                                <audio>',
                    '                                    <channelcount>2</channelcount>',
                    '                                </audio>',
                    '                            </media>',
                    '                        </file>',
                    '                        <compositemode>normal</compositemode>',
                ])

                # Add chapter title as comment if available on the alt video segment
                alt_chapter_title = getattr(alt_seg, 'chapter_title', '')
                if alt_chapter_title:
                    xml_lines.append(f'                        <comment>{escape_xml(alt_chapter_title)}</comment>')

                xml_lines.append('                    </clipitem>')

            alt_timeline_pos += target_frames

        xml_lines.append('                </track>')


def generate_davinci_sequence_xml(
    matches: List['MatchResult'],
    output_path: str,
    voiceover_path: str = None,
    frame_rate: float = 30.0,
    downloaded_segments: Optional[List] = None,
    width: int = 1920,
    height: int = 1080,
    timeline_start_tc: str = '01:00:00:00'
) -> str:
    """
    Generate DaVinci Resolve compatible sequence XML (V1 track only).

    This format matches what DaVinci Resolve exports and imports reliably.
    Uses xmeml version 5 with file definitions inside clipitems.

    Args:
        matches: List of match results
        output_path: Output XML path
        voiceover_path: Path to voiceover audio (unused, for API compat)
        frame_rate: Timeline frame rate
        downloaded_segments: Video segments for audio-first resolution
        width: Video width
        height: Video height

    Returns:
        Path to generated XML file
    """
    fps_int, is_ntsc = _validate_frame_rate(frame_rate)
    ntsc_str = 'TRUE' if is_ntsc else 'FALSE'

    # Build segment lookup for audio-first mode resolution
    segment_lookup = _build_segment_lookup(downloaded_segments)

    # Calculate total duration
    total_frames = 0
    for m in matches:
        vo_seg = m.primary_match.voiceover_segment
        total_frames += int((seg_end(vo_seg) - seg_start(vo_seg)) * frame_rate)

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="5">',
        '    <sequence>',
        '        <name>V1 - Primary Edit</name>',
        f'        <duration>{total_frames}</duration>',
        '        <rate>',
        f'            <timebase>{fps_int}</timebase>',
        f'            <ntsc>{ntsc_str}</ntsc>',
        '        </rate>',
        '        <in>-1</in>',
        '        <out>-1</out>',
        '        <timecode>',
        f'            <string>{timeline_start_tc}</string>',
        f'            <frame>{parse_timecode_to_frames(timeline_start_tc, frame_rate)}</frame>',
        f'            <displayformat>{"DF" if is_ntsc else "NDF"}</displayformat>',
        '            <rate>',
        f'                <timebase>{fps_int}</timebase>',
        f'                <ntsc>{ntsc_str}</ntsc>',
        '            </rate>',
        '        </timecode>',
        '        <media>',
        '            <video>',
        '                <track>',
    ]

    # Track file IDs and clip data for audio track
    file_ids = {}
    file_counter = 0
    clip_data = []  # Store clip info for audio track

    timeline_pos = 0
    for match_idx, match_result in enumerate(matches):
        vo_seg = match_result.primary_match.voiceover_segment
        vid_seg = match_result.primary_match.video_segment

        target_duration = seg_end(vo_seg) - seg_start(vo_seg)
        target_frames = int(target_duration * frame_rate)

        source_start = vid_seg.start_time
        source_duration = vid_seg.end_time - vid_seg.start_time
        source_frames = int(source_duration * frame_rate)

        # Resolve audio to video segment path
        resolved_path, adjusted_start = _resolve_video_segment(
            vid_seg.source_file, source_start, segment_lookup
        )
        in_frames = int(adjusted_start * frame_rate)
        out_frames = in_frames + source_frames

        # Get or create file ID
        if resolved_path not in file_ids:
            file_counter += 1
            file_ids[resolved_path] = file_counter

        current_file_id = file_ids[resolved_path]
        # Include file extension in name (DaVinci expects this)
        clip_name = escape_xml(Path(resolved_path).name)
        path_url = format_path_url(resolved_path)

        # file_duration must match the physical segment file on disk.
        # Clamp in/out to not exceed the segment file.
        seg_dur_secs = _get_segment_file_duration(resolved_path, segment_lookup)
        seg_dur_frames = int(seg_dur_secs * frame_rate)
        file_duration = seg_dur_frames
        in_frames = min(in_frames, max(0, seg_dur_frames - 1))
        out_frames = min(out_frames, seg_dur_frames)

        # Store clip data for audio track
        clip_data.append({
            'clip_name': clip_name,
            'file_id': current_file_id,
            'file_duration': file_duration,
            'start': timeline_pos,
            'end': timeline_pos + target_frames,
            'in_frames': in_frames,
            'out_frames': out_frames,
            'match_idx': match_idx,
        })

        xml_lines.extend([
            f'                    <clipitem id="{clip_name} {match_idx}">',
            f'                        <name>{clip_name}</name>',
            f'                        <duration>{file_duration}</duration>',
            '                        <rate>',
            f'                            <timebase>{fps_int}</timebase>',
            f'                            <ntsc>{ntsc_str}</ntsc>',
            '                        </rate>',
            f'                        <start>{timeline_pos}</start>',
            f'                        <end>{timeline_pos + target_frames}</end>',
            '                        <enabled>TRUE</enabled>',
            f'                        <in>{in_frames}</in>',
            f'                        <out>{out_frames}</out>',
            f'                        <file id="{clip_name} {current_file_id}">',
            f'                            <duration>{file_duration}</duration>',
            '                            <rate>',
            f'                                <timebase>{fps_int}</timebase>',
            f'                                <ntsc>{ntsc_str}</ntsc>',
            '                            </rate>',
            f'                            <name>{clip_name}</name>',
            f'                            <pathurl>{path_url}</pathurl>',
            '                            <timecode>',
            '                                <string>00:00:00:00</string>',
            f'                                <displayformat>{"DF" if is_ntsc else "NDF"}</displayformat>',
            '                                <rate>',
            f'                                    <timebase>{fps_int}</timebase>',
            f'                                    <ntsc>{ntsc_str}</ntsc>',
            '                                </rate>',
            '                            </timecode>',
            '                            <media>',
            '                                <video>',
            f'                                    <duration>{file_duration}</duration>',
            '                                    <samplecharacteristics>',
            f'                                        <width>{width}</width>',
            f'                                        <height>{height}</height>',
            '                                    </samplecharacteristics>',
            '                                </video>',
            '                                <audio>',
            '                                    <channelcount>2</channelcount>',
            '                                </audio>',
            '                            </media>',
            '                        </file>',
            '                        <compositemode>normal</compositemode>',
        ])

        # Add chapter title as comment if available on the video segment
        chapter_title = getattr(vid_seg, 'chapter_title', '')
        if chapter_title:
            xml_lines.append(f'                        <comment>{escape_xml(chapter_title)}</comment>')

        xml_lines.append('                    </clipitem>')

        timeline_pos += target_frames

    # Close V1 video track
    xml_lines.extend([
        '                </track>',
    ])

    # Add V2-V8 tracks (disabled by default)
    _add_sequence_alt_tracks(
        xml_lines, matches, frame_rate, fps_int, ntsc_str, is_ntsc,
        segment_lookup, file_ids, width, height
    )

    xml_lines.extend([
        '            </video>',
    ])

    # Add audio track (references same files as video)
    xml_lines.extend([
        '            <audio>',
        '                <track>',
    ])

    for clip in clip_data:
        xml_lines.extend([
            f'                    <clipitem id="{clip["clip_name"]} {clip["match_idx"]} audio">',
            f'                        <name>{clip["clip_name"]}</name>',
            f'                        <duration>{clip["file_duration"]}</duration>',
            '                        <rate>',
            f'                            <timebase>{fps_int}</timebase>',
            f'                            <ntsc>{ntsc_str}</ntsc>',
            '                        </rate>',
            f'                        <start>{clip["start"]}</start>',
            f'                        <end>{clip["end"]}</end>',
            '                        <enabled>TRUE</enabled>',
            f'                        <in>{clip["in_frames"]}</in>',
            f'                        <out>{clip["out_frames"]}</out>',
            f'                        <file id="{clip["clip_name"]} {clip["file_id"]}"/>',
            '                    </clipitem>',
        ])

    xml_lines.extend([
        '                </track>',
        '            </audio>',
        '        </media>',
        '    </sequence>',
        '</xmeml>',
    ])

    # Write the file
    xml_path = Path(output_path).with_suffix('.xml')
    if not str(xml_path).endswith('_sequence.xml'):
        xml_path = Path(str(output_path).replace('.xml', '') + '_sequence.xml')

    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))

    # Get file size
    try:
        file_size = os.path.getsize(xml_path)
        file_size_kb = file_size / 1024
        file_size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb / 1024:.2f} MB"
    except OSError:
        file_size_str = "unknown size"

    logger.info(
        f"[OUTPUT] XML (DaVinci sequence) file generated: {xml_path} "
        f"({len(matches)} clips, {file_size_str})"
    )
    return str(xml_path)
