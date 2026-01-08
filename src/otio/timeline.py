"""
Timeline orchestration for OTIO generation.

Migrated from otio_builder.py create_timeline() function.
This module handles the core timeline assembly logic.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import opentimelineio as otio

from .utils import (
    _to_windows_path,
    _get_media_duration,
    get_segment_file_offset,
    get_confidence_color,
    create_clip_with_timewarp,
)
from .entities import _add_entity_images_to_track, _add_entity_videos_to_track

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import MatchResult

logger = logging.getLogger(__name__)


def _validate_entity_images(entity_images: Dict) -> Dict:
    """
    Validate entity images and filter out invalid entries.

    Returns a dict with only valid entities that have downloadable images.
    """
    validated = {}
    for entity_name, entity_result in entity_images.items():
        # Check if entity_result has images attribute and it's not empty
        if hasattr(entity_result, 'images') and entity_result.images:
            # Filter out invalid image paths
            valid_images = [
                img for img in entity_result.images
                if hasattr(img, 'file') and img.file and Path(img.file).exists()
            ]
            if valid_images:
                # Create a copy of entity_result with only valid images
                import copy
                validated_result = copy.copy(entity_result)
                validated_result.images = valid_images
                validated[entity_name] = validated_result

    return validated


def create_timeline(
    matches: List['MatchResult'],
    config: 'Config',
    voiceover_path: Optional[str] = None,
    frame_rate: float = 30.0,
    entity_images: Optional[Dict] = None,
    entity_videos: Optional[Dict] = None,
    downloaded_segments: Optional[List] = None
) -> otio.schema.Timeline:
    """
    Create OTIO timeline from matches.

    Track structure:
    - V1: Primary video (speed-adjusted) - enabled
    - V2: Alternative 1 (speed-adjusted) - disabled
    - V3: Alternative 2 (speed-adjusted) - disabled
    - V4: Secondary Primary (different video files from V1-V3) - disabled
    - V5: Secondary Alt 1 (different video files) - disabled
    - V6: Secondary Alt 2 (different video files) - disabled
    - V7: Embedding-Diversity strategy - disabled
    - V8: B-roll Only - disabled
    - V9: Entity Images (Google stills) - disabled
    - V10: Stock Videos (Pexels/Pixabay) - disabled
    - A1-A8: Corresponding audio tracks
    - A9: Voiceover - enabled

    Args:
        matches: List of match results from matching stage
        config: Pipeline configuration
        voiceover_path: Path to voiceover file
        frame_rate: Timeline frame rate
        entity_images: Entity images from EntityImagesStage
        entity_videos: Stock videos from EntityVideosStage
        downloaded_segments: Optional list of DownloadedSegment from audio-first mode.
            When provided, video segment files are used instead of audio files.

    Returns:
        OTIO Timeline with all tracks populated
    """

    # Build lookup for audio-first mode: video_id -> segment info
    # This maps audio file video IDs to their downloaded video segment files
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

    def resolve_video_segment(source_file: str, source_start: float) -> Tuple[str, float]:
        """
        Resolve audio file path to video segment path for audio-first mode.

        Args:
            source_file: Original source file (may be audio .mp3)
            source_start: Start time in the original source

        Returns:
            Tuple of (resolved_path, adjusted_start_time)
            - If video segment found: (segment_file, time_relative_to_segment)
            - Otherwise: (original_source_file, original_source_start)
        """
        if not segment_lookup:
            return source_file, source_start

        # Extract video_id from the source file path
        # Audio files are like: /path/to/video_id.mp3 or /path/to/folder/video_id.mp3
        stem = Path(source_file).stem
        video_id = stem

        # Check if we have segment(s) for this video
        if video_id not in segment_lookup:
            return source_file, source_start

        # Find the segment that contains this time
        segments = segment_lookup[video_id]
        for seg_info in segments:
            # Check if source_start falls within this segment's range
            if seg_info['start'] <= source_start <= seg_info['end']:
                # Calculate the offset within the segment file
                adjusted_start = source_start - seg_info['start']
                return seg_info['file'], adjusted_start

        # If no segment contains this exact time, use the first segment
        # and let the clip reference the original time (fallback)
        if segments:
            seg_info = segments[0]
            # Check if it's reasonably close
            if source_start >= seg_info['start'] and source_start <= seg_info['end'] + 60:
                adjusted_start = max(0, source_start - seg_info['start'])
                return seg_info['file'], adjusted_start

        return source_file, source_start

    timeline = otio.schema.Timeline(name="Matched Footage")

    # Set tracks stack name to empty (DaVinci format)
    timeline.tracks.name = ""

    # Add Resolve_OTIO metadata (required for DaVinci import)
    timeline.metadata['Resolve_OTIO'] = {
        'Resolve OTIO Meta Version': '1.0'
    }

    # CRITICAL: Set global_start_time to valid RationalTime (not empty string!)
    # DaVinci Resolve hangs indefinitely if this is "" or invalid
    # Using 01:00:00:00 timecode start (86400 frames at 24fps, scaled to frame_rate)
    timeline.global_start_time = otio.opentime.RationalTime(
        int(3600 * frame_rate),  # 1 hour in frames
        frame_rate
    )

    rate = frame_rate

    # Determine number of alternative tracks (V2-V3)
    num_alternatives = config.output.num_alternatives if config.output.include_alternatives else 0

    # Secondary tracks (V4-V6) - always 3
    num_secondary = 3

    # Strategy tracks (V7+)
    strategy_names = []
    if config.output.include_strategy_tracks:
        strategy_names = list(config.output.strategy_tracks)

    strategy_display_names = {
        "embedding_diversity": "Embedding-Diversity",
        "broll_only": "B-roll Only"
    }

    # Create video tracks
    video_tracks = []

    # V1: Primary
    track = otio.schema.Track(name="V1 - Primary", kind=otio.schema.TrackKind.Video)
    track.metadata['Resolve_OTIO'] = {'Locked': False}
    video_tracks.append(track)

    # V2-V3: Alternatives
    for i in range(num_alternatives):
        track = otio.schema.Track(name=f"V{i+2} - Alternative {i+1}", kind=otio.schema.TrackKind.Video)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        video_tracks.append(track)

    # V4-V6: Secondary matches (different video files from V1-V3)
    secondary_names = ["Secondary Primary", "Secondary Alt 1", "Secondary Alt 2"]
    for i in range(num_secondary):
        track_num = 1 + num_alternatives + i + 1  # V4, V5, V6
        track = otio.schema.Track(name=f"V{track_num} - {secondary_names[i]}", kind=otio.schema.TrackKind.Video)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        video_tracks.append(track)

    # V7-V8: Strategy tracks
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + num_secondary + i + 1  # V7, V8
        track = otio.schema.Track(name=f"V{track_num} - {display_name}", kind=otio.schema.TrackKind.Video)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        video_tracks.append(track)

    # V9: Entity Images track (Google Images)
    image_track = otio.schema.Track(name="V9 - Entity Images", kind=otio.schema.TrackKind.Video)
    image_track.enabled = False  # Disabled by default, user enables as needed
    image_track.metadata['Resolve_OTIO'] = {'Locked': False}

    # V10: Stock Videos track (Pexels/Pixabay)
    stock_video_track = otio.schema.Track(name="V10 - Stock Videos", kind=otio.schema.TrackKind.Video)
    stock_video_track.enabled = False  # Disabled by default
    stock_video_track.metadata['Resolve_OTIO'] = {'Locked': False}

    # Create audio tracks for video audio
    audio_tracks = []

    # A1: Primary audio
    track = otio.schema.Track(name="A1 - Video Audio", kind=otio.schema.TrackKind.Audio)
    track.metadata['Resolve_OTIO'] = {'Locked': False}
    audio_tracks.append(track)

    # A2-A3: Alternative audio
    for i in range(num_alternatives):
        track = otio.schema.Track(name=f"A{i+2} - Alt {i+1} Audio", kind=otio.schema.TrackKind.Audio)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        audio_tracks.append(track)

    # A4-A6: Secondary audio tracks
    for i in range(num_secondary):
        track_num = 1 + num_alternatives + i + 1  # A4, A5, A6
        track = otio.schema.Track(name=f"A{track_num} - {secondary_names[i]} Audio", kind=otio.schema.TrackKind.Audio)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        audio_tracks.append(track)

    # A7-A8: Strategy audio tracks
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + num_secondary + i + 1  # A7, A8
        track = otio.schema.Track(name=f"A{track_num} - {display_name} Audio", kind=otio.schema.TrackKind.Audio)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        audio_tracks.append(track)

    # Create voiceover track
    voiceover_track_num = 1 + num_alternatives + num_secondary + len(strategy_names) + 1
    voiceover_track = otio.schema.Track(
        name=f"A{voiceover_track_num} - Voiceover",
        kind=otio.schema.TrackKind.Audio
    )
    voiceover_track.metadata['Resolve_OTIO'] = {'Locked': False}

    # Track timeline position in FRAMES (integer) to avoid floating-point drift
    timeline_frames = 0

    # Get actual voiceover duration for proper timeline alignment
    actual_vo_duration = _get_media_duration(voiceover_path) if voiceover_path else None
    if actual_vo_duration:
        logger.info(f"Voiceover file duration: {actual_vo_duration:.2f}s")
        print(f"  ✓ Voiceover duration detected: {actual_vo_duration:.2f}s ({actual_vo_duration/60:.1f} min)")
    elif matches:
        # Fallback: use last segment end time + buffer for trailing content
        last_segment = matches[-1].primary_match.voiceover_segment
        fallback_duration = last_segment.end_time + 30.0  # Add 30s buffer for trailing
        logger.warning(f"ffprobe unavailable, using fallback duration: {fallback_duration:.2f}s (last segment + 30s buffer)")
        print(f"  ⚠ Using fallback VO duration: {fallback_duration:.2f}s (ffprobe unavailable)")
        actual_vo_duration = fallback_duration

    # Get the first segment's start time as timeline reference
    first_segment_start = matches[0].primary_match.voiceover_segment.start_time if matches else 0.0

    # Add leading gap if first segment doesn't start at 0
    # This aligns video clips with the actual voiceover playback timing
    if matches and first_segment_start > 0.1:  # More than 100ms of leading silence
        leading_frames = round(first_segment_start * rate)
        logger.info(f"Adding {first_segment_start:.1f}s leading gap to align with voiceover start")

        leading_gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(leading_frames, rate)
            )
        )

        # Add leading gap to all video tracks
        for track in video_tracks:
            track.append(copy.deepcopy(leading_gap))
        # Add leading gap to all audio tracks
        for track in audio_tracks:
            track.append(copy.deepcopy(leading_gap))

        # Update timeline position
        timeline_frames += leading_frames

    # Process each match
    for match_idx, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        vid_seg = match.video_segment

        # Check for gap before this segment (silence in voiceover)
        # Expected position = where this segment should start relative to first segment
        expected_start_frames = round((vo_seg.start_time - first_segment_start) * frame_rate)

        if expected_start_frames > timeline_frames:
            # There's a gap - insert silence/gap clips on all tracks
            gap_frames = expected_start_frames - timeline_frames
            gap_duration = otio.opentime.RationalTime(gap_frames, rate)

            logger.debug(f"Segment {match_idx}: Inserting {gap_frames/rate:.2f}s gap before (vo gap from {timeline_frames/rate:.2f}s to {expected_start_frames/rate:.2f}s)")

            # Add gap to all video tracks
            for track in video_tracks:
                track.append(otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                ))

            # Add gap to all audio tracks
            for track in audio_tracks:
                track.append(otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                ))

            timeline_frames = expected_start_frames

        # Target duration = voiceover segment duration
        target_duration = vo_seg.end_time - vo_seg.start_time
        duration_frames = round(target_duration * frame_rate)

        # Source duration = video segment duration
        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time

        # Resolve audio file to video segment (audio-first mode)
        # This maps .mp3 audio files to downloaded .mp4 video segments
        resolved_source, adjusted_start = resolve_video_segment(vid_seg.source_file, source_start)

        # Also check for segment file offset from filename (legacy support)
        segment_offset = get_segment_file_offset(resolved_source)
        if segment_offset > 0 and resolved_source == vid_seg.source_file:
            # Only apply filename-based offset if we didn't already resolve
            adjusted_start = max(0, source_start - segment_offset)
            logger.debug(f"Segment file offset: {segment_offset}s, adjusted start: {adjusted_start}s")

        # Use the resolved source file and adjusted start time
        source_file_for_clip = resolved_source
        source_start = adjusted_start

        # Determine clip color based on confidence
        clip_color = get_confidence_color(match.confidence)

        # Build metadata - include segment_index for post-edit analysis tracing
        segment_id = f"S{match_idx:03d}"  # S000, S001, S002, ...
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

        # Create primary video clip (V1) - prefix with segment ID for tracing
        clip_folder = Path(source_file_for_clip).parent.name
        clip_stem = Path(source_file_for_clip).stem
        v1_clip = create_clip_with_timewarp(
            name=f"[{segment_id}] {clip_folder}_{clip_stem} [{vid_seg.start_time:.1f}s]",
            source_path=source_file_for_clip,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata=metadata
        )

        # Set clip color
        v1_clip.metadata['clip_color'] = clip_color

        video_tracks[0].append(v1_clip)

        # Create primary audio clip (A1) - same source, same timing
        a1_clip = create_clip_with_timewarp(
            name=f"[{segment_id}] Audio: {clip_folder}_{clip_stem}",
            source_path=source_file_for_clip,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata={'from_track': 'V1'}
        )
        audio_tracks[0].append(a1_clip)

        # Process alternatives (V2-V3, A2-A3)
        for alt_idx in range(num_alternatives):
            if alt_idx < len(match_result.alternatives):
                alt = match_result.alternatives[alt_idx]
                alt_seg = alt.video_segment

                alt_source_duration = alt_seg.end_time - alt_seg.start_time
                alt_source_start = alt_seg.start_time

                # Resolve audio file to video segment (audio-first mode)
                alt_resolved_source, alt_adjusted_start = resolve_video_segment(alt_seg.source_file, alt_source_start)

                # Legacy segment file offset support
                alt_segment_offset = get_segment_file_offset(alt_resolved_source)
                if alt_segment_offset > 0 and alt_resolved_source == alt_seg.source_file:
                    alt_adjusted_start = max(0, alt_source_start - alt_segment_offset)

                alt_source_file = alt_resolved_source
                alt_source_start = alt_adjusted_start

                alt_metadata = {
                    'confidence': alt.confidence,
                    'reasoning': alt.reasoning,
                    'original_duration': alt_source_duration,
                    'target_duration': target_duration
                }

                # Alternative video clip - include segment ID for tracing
                alt_folder = Path(alt_source_file).parent.name
                alt_stem = Path(alt_source_file).stem
                alt_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] ALT{alt_idx+1}: {alt_folder}_{alt_stem}",
                    source_path=alt_source_file,
                    source_start=alt_source_start,
                    source_duration=alt_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=alt_metadata
                )

                # Set clip color for alternatives too
                alt_v_clip.metadata['clip_color'] = get_confidence_color(alt.confidence)

                video_tracks[alt_idx + 1].append(alt_v_clip)

                # Alternative audio clip
                alt_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio ALT{alt_idx+1}: {alt_folder}_{alt_stem}",
                    source_path=alt_source_file,
                    source_start=alt_source_start,
                    source_duration=alt_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata={'from_track': f'V{alt_idx+2}'}
                )
                audio_tracks[alt_idx + 1].append(alt_a_clip)
            else:
                # No alternative available - add gap (use rounded frames)
                gap_duration = otio.opentime.RationalTime(duration_frames, rate)

                v_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                video_tracks[alt_idx + 1].append(v_gap)

                a_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                audio_tracks[alt_idx + 1].append(a_gap)

        # Process secondary tracks (V4-V6) - different video files from V1-V3
        secondary_base_idx = 1 + num_alternatives  # Index where secondary tracks start

        for sec_idx in range(num_secondary):
            track_idx = secondary_base_idx + sec_idx

            if sec_idx < len(match_result.secondary_matches):
                sec_match = match_result.secondary_matches[sec_idx]
                sec_seg = sec_match.video_segment
                sec_source_duration = sec_seg.end_time - sec_seg.start_time
                sec_source_start = sec_seg.start_time

                # Resolve audio file to video segment (audio-first mode)
                sec_resolved_source, sec_adjusted_start = resolve_video_segment(sec_seg.source_file, sec_source_start)

                # Legacy segment file offset support
                sec_segment_offset = get_segment_file_offset(sec_resolved_source)
                if sec_segment_offset > 0 and sec_resolved_source == sec_seg.source_file:
                    sec_adjusted_start = max(0, sec_source_start - sec_segment_offset)

                sec_source_file = sec_resolved_source
                sec_source_start = sec_adjusted_start

                sec_metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': sec_match.confidence,
                    'reasoning': sec_match.reasoning,
                    'original_duration': sec_source_duration,
                    'target_duration': target_duration,
                    'is_secondary': True
                }

                # Secondary video clip - include segment ID for tracing
                sec_label = secondary_names[sec_idx] if sec_idx < len(secondary_names) else f"Secondary {sec_idx}"
                sec_folder = Path(sec_source_file).parent.name
                sec_stem = Path(sec_source_file).stem
                sec_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] {sec_label}: {sec_folder}_{sec_stem}",
                    source_path=sec_source_file,
                    source_start=sec_source_start,
                    source_duration=sec_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=sec_metadata
                )

                # Color for secondary tracks
                secondary_colors = ["PURPLE", "BLUE", "TEAL"]
                sec_v_clip.metadata['clip_color'] = secondary_colors[sec_idx] if sec_idx < len(secondary_colors) else "GRAY"

                video_tracks[track_idx].append(sec_v_clip)

                # Secondary audio clip
                sec_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio {sec_label}: {sec_folder}_{sec_stem}",
                    source_path=sec_source_file,
                    source_start=sec_source_start,
                    source_duration=sec_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata={'from_track': f'V{track_idx+1}'}
                )
                audio_tracks[track_idx].append(sec_a_clip)
            else:
                # No secondary match available - add gap
                gap_duration = otio.opentime.RationalTime(duration_frames, rate)

                v_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                video_tracks[track_idx].append(v_gap)

                a_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                audio_tracks[track_idx].append(a_gap)

        # Process strategy tracks (V7-V8: embedding_diversity, broll_only)
        strategy_base_idx = 1 + num_alternatives + num_secondary  # Index where strategy tracks start

        for strat_idx, strategy in enumerate(strategy_names):
            track_idx = strategy_base_idx + strat_idx

            # Find strategy match for this strategy
            strat_match = None
            if match_result.strategy_matches:
                for sm in match_result.strategy_matches:
                    if sm.strategy == strategy:
                        strat_match = sm
                        break

            if strat_match:
                strat_seg = strat_match.video_segment
                strat_source_duration = strat_seg.end_time - strat_seg.start_time
                strat_source_start = strat_seg.start_time

                # Resolve audio file to video segment (audio-first mode)
                strat_resolved_source, strat_adjusted_start = resolve_video_segment(strat_seg.source_file, strat_source_start)

                # Legacy segment file offset support
                strat_segment_offset = get_segment_file_offset(strat_resolved_source)
                if strat_segment_offset > 0 and strat_resolved_source == strat_seg.source_file:
                    strat_adjusted_start = max(0, strat_source_start - strat_segment_offset)

                strat_source_file = strat_resolved_source
                strat_source_start = strat_adjusted_start

                strat_metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': strat_match.confidence,
                    'reasoning': strat_match.reasoning,
                    'strategy': strat_match.strategy,
                    'original_duration': strat_source_duration,
                    'target_duration': target_duration
                }

                # Strategy video clip - include segment ID for tracing
                strat_folder = Path(strat_source_file).parent.name
                strat_stem = Path(strat_source_file).stem
                strat_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] {strategy.upper()}: {strat_folder}_{strat_stem}",
                    source_path=strat_source_file,
                    source_start=strat_source_start,
                    source_duration=strat_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=strat_metadata
                )

                # Color based on strategy
                strategy_colors = {
                    "embedding_diversity": "PINK",
                    "broll_only": "TEAL"
                }
                strat_v_clip.metadata['clip_color'] = strategy_colors.get(strategy, "GRAY")

                video_tracks[track_idx].append(strat_v_clip)

                # Strategy audio clip
                strat_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio {strategy.upper()}: {strat_folder}_{strat_stem}",
                    source_path=strat_source_file,
                    source_start=strat_source_start,
                    source_duration=strat_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata={'from_track': f'V{track_idx+1}', 'strategy': strategy}
                )
                audio_tracks[track_idx].append(strat_a_clip)
            else:
                # No strategy match available - add gap
                gap_duration = otio.opentime.RationalTime(duration_frames, rate)

                v_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                video_tracks[track_idx].append(v_gap)

                a_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                audio_tracks[track_idx].append(a_gap)

        # Update timeline position using integer frames to avoid drift
        timeline_frames += duration_frames

    # Add trailing gap to match actual voiceover duration
    # This ensures video tracks extend to cover trailing audio (music, silence, outro)
    if actual_vo_duration and matches:
        accumulated_duration = timeline_frames / rate  # Current timeline in seconds

        if actual_vo_duration > accumulated_duration + 0.1:  # More than 100ms trailing
            trailing_seconds = actual_vo_duration - accumulated_duration
            trailing_frames = round(trailing_seconds * rate)
            logger.info(f"Adding {trailing_seconds:.1f}s trailing gap to match voiceover end")
            print(f"  ✓ Adding {trailing_seconds:.1f}s trailing gap (VO: {actual_vo_duration:.1f}s, timeline: {accumulated_duration:.1f}s)")

            trailing_gap = otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, rate),
                    duration=otio.opentime.RationalTime(trailing_frames, rate)
                )
            )

            # Add trailing gap to all video tracks
            for track in video_tracks:
                track.append(copy.deepcopy(trailing_gap))
            # Add trailing gap to all audio tracks
            for track in audio_tracks:
                track.append(copy.deepcopy(trailing_gap))

            # Update timeline position to include trailing content
            timeline_frames += trailing_frames

    # Add voiceover track
    if voiceover_path and matches:
        # Create absolute path for voiceover (Windows format for DaVinci)
        abs_vo_path = _to_windows_path(voiceover_path)
        vo_folder = Path(voiceover_path).parent.name
        vo_filename = Path(voiceover_path).name
        vo_unique_name = f"{vo_folder}_{vo_filename}"

        # Use actual voiceover file duration if available, otherwise use accumulated frames
        if actual_vo_duration:
            vo_total_frames = round(actual_vo_duration * rate)
            logger.info(f"Voiceover clip: using actual duration {actual_vo_duration:.2f}s ({vo_total_frames} frames)")
        else:
            vo_total_frames = timeline_frames
            logger.warning("Could not determine voiceover duration, using accumulated segment total")

        # Create proper ExternalReference with available_range
        vo_available_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, rate),
            duration=otio.opentime.RationalTime(vo_total_frames, rate)
        )

        vo_ref = otio.schema.ExternalReference(
            target_url=abs_vo_path,
            available_range=vo_available_range
        )
        vo_ref.name = vo_unique_name  # Unique name includes folder

        # Voiceover clip starts at 0 and uses actual file duration
        # Video/audio tracks have leading gap added to align with VO playback
        vo_clip = otio.schema.Clip(
            name="Voiceover",
            media_reference=vo_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(vo_total_frames, rate)
            )
        )
        vo_clip.metadata['Resolve_OTIO'] = {}  # Required for DaVinci import
        voiceover_track.append(vo_clip)

    # Add all tracks to timeline
    for track in video_tracks:
        timeline.tracks.append(track)

    for track in audio_tracks:
        timeline.tracks.append(track)

    # Only add voiceover track if it has content
    if len(voiceover_track) > 0:
        timeline.tracks.append(voiceover_track)

    # Populate image track if entity_images provided
    if entity_images:
        # Log what we received
        print(f"  [V9] Entity images received: {len(entity_images)} entities")
        for ename, eresult in entity_images.items():
            img_count = len(getattr(eresult, 'images', []))
            print(f"    • {ename}: {img_count} images")

        # Validate and filter entity images before using
        validated_entity_images = _validate_entity_images(entity_images)
        if validated_entity_images:
            total_images = sum(len(e.images) for e in validated_entity_images.values())
            print(f"  [V9] After validation: {len(validated_entity_images)} entities, {total_images} images")
            logger.info(f"Entity images: {len(validated_entity_images)} entities with valid images")
            _add_entity_images_to_track(
                image_track=image_track,
                entity_images=validated_entity_images,
                matches=matches,
                frame_rate=rate,
                config=config
            )
        else:
            logger.warning("No valid entity images after validation")

    # Always add V9 Entity Images track (even if empty, for manual use)
    timeline.tracks.append(image_track)

    # Populate stock video track if entity_videos provided
    if entity_videos:
        # Log what we received
        print(f"  [V10] Stock videos received: {len(entity_videos)} entities")
        for ename, eresult in entity_videos.items():
            vid_count = len(getattr(eresult, 'videos', []))
            print(f"    • {ename}: {vid_count} videos")

        _add_entity_videos_to_track(
            video_track=stock_video_track,
            entity_videos=entity_videos,
            matches=matches,
            frame_rate=rate,
            config=config
        )

    # Always add V10 Stock Videos track (even if empty, for manual use)
    timeline.tracks.append(stock_video_track)

    return timeline
