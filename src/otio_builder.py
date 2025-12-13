"""
OTIO Timeline builder with:
- Speed-adjusted video clips to match voiceover duration
- Parallel audio tracks (A1-A3) matching video tracks (V1-V3)
- LinearTimeWarp for speed adjustment
- Alternative matches as disabled tracks
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict

import opentimelineio as otio

from .config import Config
from .utils import SRTSegment, MatchResult, AlternativeMatch

logger = logging.getLogger(__name__)


def create_clip_with_timewarp(
    name: str,
    source_path: str,
    source_start: float,
    source_duration: float,
    target_duration: float,
    frame_rate: float = 30.0,
    metadata: Optional[Dict] = None
) -> otio.schema.Clip:
    """
    Create a clip with duration matching the target (voiceover) duration.
    
    The clip's source_range is set to target_duration so it aligns perfectly
    on the timeline. Speed adjustment can be done manually in the NLE.
    
    Args:
        name: Clip name
        source_path: Path to source video/audio
        source_start: Start time in source (seconds)
        source_duration: Original duration in source (seconds) - stored in metadata
        target_duration: Desired duration on timeline (seconds) - used for source_range
        frame_rate: Frame rate
        metadata: Optional metadata dict
    
    Returns:
        OTIO Clip with duration matching target_duration
    """
    rate = frame_rate
    
    # Create media reference
    media_ref = otio.schema.ExternalReference(target_url=source_path)
    
    # IMPORTANT: Use round() to avoid floating-point precision drift
    # This prevents timing deviation over many clips
    start_frames = round(source_start * rate)
    duration_frames = round(target_duration * rate)
    
    # Source range uses TARGET duration so clips align with voiceover
    # The clip will be trimmed to fit - user can adjust speed manually
    source_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(start_frames, rate),
        duration=otio.opentime.RationalTime(duration_frames, rate)
    )
    
    # Create clip
    clip = otio.schema.Clip(
        name=name,
        media_reference=media_ref,
        source_range=source_range
    )
    
    # Calculate and store speed info in metadata for reference
    if metadata is None:
        metadata = {}
    
    if target_duration > 0 and source_duration > 0:
        time_scalar = source_duration / target_duration
        metadata['time_scalar'] = time_scalar
        metadata['speed_percent'] = time_scalar * 100
        metadata['target_duration'] = target_duration
        metadata['source_duration'] = source_duration
        metadata['suggested_speed'] = f"{time_scalar * 100:.0f}%"
    
    # Add metadata
    for key, value in metadata.items():
        clip.metadata[key] = value
    
    return clip


def create_timeline(
    matches: List[MatchResult],
    config: Config,
    voiceover_path: Optional[str] = None,
    frame_rate: float = 30.0
) -> otio.schema.Timeline:
    """
    Create OTIO timeline from matches.
    
    Track structure:
    - V1: Primary video (speed-adjusted) - enabled
    - V2: Alternative 1 (speed-adjusted) - disabled
    - V3: Alternative 2 (speed-adjusted) - disabled
    - V4: Visual-First strategy - disabled
    - V5: Different-Source strategy - disabled
    - V6: Keyword-Only strategy - disabled
    - V7: Embedding-Diversity strategy - disabled
    - A1-A7: Corresponding audio tracks
    - A8: Voiceover - enabled
    """
    
    timeline = otio.schema.Timeline(name="Matched Footage")
    rate = frame_rate
    
    # Determine number of alternative tracks
    num_alternatives = config.output.num_alternatives if config.output.include_alternatives else 0
    
    # Strategy tracks
    strategy_names = config.output.strategy_tracks if config.output.include_strategy_tracks else []
    strategy_display_names = {
        "visual_first": "Visual-First",
        "different_source": "Different-Source",
        "keyword_only": "Keyword-Only",
        "embedding_diversity": "Embedding-Diversity",
        "source_rotation": "Source-Rotation"
    }
    
    # Create video tracks
    video_tracks = []
    
    # V1: Primary
    track = otio.schema.Track(name="V1 - Primary", kind=otio.schema.TrackKind.Video)
    video_tracks.append(track)
    
    # V2-V3: Alternatives
    for i in range(num_alternatives):
        track = otio.schema.Track(name=f"V{i+2} - Alternative {i+1}", kind=otio.schema.TrackKind.Video)
        track.enabled = False
        video_tracks.append(track)
    
    # V4-V7: Strategy tracks
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + i + 1
        track = otio.schema.Track(name=f"V{track_num} - {display_name}", kind=otio.schema.TrackKind.Video)
        track.enabled = False
        video_tracks.append(track)
    
    # Create audio tracks for video audio
    audio_tracks = []
    
    # A1: Primary audio
    track = otio.schema.Track(name="A1 - Video Audio", kind=otio.schema.TrackKind.Audio)
    audio_tracks.append(track)
    
    # A2-A3: Alternative audio
    for i in range(num_alternatives):
        track = otio.schema.Track(name=f"A{i+2} - Alt {i+1} Audio", kind=otio.schema.TrackKind.Audio)
        track.enabled = False
        audio_tracks.append(track)
    
    # A4-A7: Strategy audio tracks
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + i + 1
        track = otio.schema.Track(name=f"A{track_num} - {display_name} Audio", kind=otio.schema.TrackKind.Audio)
        track.enabled = False
        audio_tracks.append(track)
    
    # Create voiceover track
    voiceover_track_num = 1 + num_alternatives + len(strategy_names) + 1
    voiceover_track = otio.schema.Track(
        name=f"A{voiceover_track_num} - Voiceover",
        kind=otio.schema.TrackKind.Audio
    )
    
    # Track timeline position in FRAMES (integer) to avoid floating-point drift
    timeline_frames = 0
    timeline_markers = []
    
    # Process each match
    for match_idx, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        vid_seg = match.video_segment
        
        # Target duration = voiceover segment duration
        target_duration = vo_seg.end_time - vo_seg.start_time
        duration_frames = round(target_duration * frame_rate)
        
        # Source duration = video segment duration
        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time
        
        # Determine clip color based on confidence
        clip_color = get_confidence_color(match.confidence)
        
        # Build metadata
        metadata = {
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
        v1_clip = create_clip_with_timewarp(
            name=f"{Path(vid_seg.source_file).stem} [{vid_seg.start_time:.1f}s]",
            source_path=vid_seg.source_file,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata=metadata
        )
        
        # Set clip color
        v1_clip.metadata['clip_color'] = clip_color
        
        # Add markers to primary clip
        add_markers_to_clip(v1_clip, match_result, match, rate, target_duration)
        
        video_tracks[0].append(v1_clip)
        
        # Create timeline-level marker for this segment (using frame position)
        timeline_position_seconds = timeline_frames / frame_rate
        timeline_marker = create_timeline_marker(
            match_idx + 1,
            match_result,
            match,
            timeline_position_seconds,
            target_duration,
            rate
        )
        timeline_markers.append(timeline_marker)
        
        # Create primary audio clip (A1) - same source, same timing
        a1_clip = create_clip_with_timewarp(
            name=f"Audio: {Path(vid_seg.source_file).stem}",
            source_path=vid_seg.source_file,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata={'from_track': 'V1'}
        )
        audio_tracks[0].append(a1_clip)
        
        # Process alternatives
        for alt_idx in range(num_alternatives):
            if alt_idx < len(match_result.alternatives):
                alt = match_result.alternatives[alt_idx]
                alt_seg = alt.video_segment
                
                alt_source_duration = alt_seg.end_time - alt_seg.start_time
                alt_source_start = alt_seg.start_time
                
                alt_metadata = {
                    'confidence': alt.confidence,
                    'reasoning': alt.reasoning,
                    'original_duration': alt_source_duration,
                    'target_duration': target_duration
                }
                
                # Alternative video clip
                alt_v_clip = create_clip_with_timewarp(
                    name=f"ALT{alt_idx+1}: {Path(alt_seg.source_file).stem}",
                    source_path=alt_seg.source_file,
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
                    name=f"Audio ALT{alt_idx+1}: {Path(alt_seg.source_file).stem}",
                    source_path=alt_seg.source_file,
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
        
        # Process strategy tracks (V4-V7)
        strategy_base_idx = 1 + num_alternatives  # Index where strategy tracks start
        
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
                
                strat_metadata = {
                    'confidence': strat_match.confidence,
                    'reasoning': strat_match.reasoning,
                    'strategy': strat_match.strategy,
                    'original_duration': strat_source_duration,
                    'target_duration': target_duration
                }
                
                # Strategy video clip
                strat_v_clip = create_clip_with_timewarp(
                    name=f"{strategy.upper()}: {Path(strat_seg.source_file).stem}",
                    source_path=strat_seg.source_file,
                    source_start=strat_source_start,
                    source_duration=strat_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=strat_metadata
                )
                
                # Color based on strategy
                strategy_colors = {
                    "visual_first": "PURPLE",
                    "different_source": "BLUE",
                    "keyword_only": "TEAL",
                    "embedding_diversity": "PINK"
                }
                strat_v_clip.metadata['clip_color'] = strategy_colors.get(strategy, "GRAY")
                
                video_tracks[track_idx].append(strat_v_clip)
                
                # Strategy audio clip
                strat_a_clip = create_clip_with_timewarp(
                    name=f"Audio {strategy.upper()}: {Path(strat_seg.source_file).stem}",
                    source_path=strat_seg.source_file,
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
    
    # Add voiceover track
    if voiceover_path and matches:
        vo_ref = otio.schema.ExternalReference(target_url=voiceover_path)
        
        # Total duration should match total frames accumulated
        total_frames = timeline_frames
        
        vo_clip = otio.schema.Clip(
            name="Voiceover",
            media_reference=vo_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(total_frames, rate)
            )
        )
        voiceover_track.append(vo_clip)
    
    # Add all tracks to timeline
    for track in video_tracks:
        timeline.tracks.append(track)
    
    for track in audio_tracks:
        timeline.tracks.append(track)
    
    timeline.tracks.append(voiceover_track)
    
    # Add timeline-level markers to the primary video track
    for marker in timeline_markers:
        video_tracks[0].markers.append(marker)
    
    return timeline


def get_confidence_color(confidence: float) -> str:
    """Get clip color name based on confidence tier"""
    if confidence >= 0.8:
        return "GREEN"
    elif confidence >= 0.6:
        return "CYAN"
    elif confidence >= 0.4:
        return "YELLOW"
    elif confidence >= 0.2:
        return "ORANGE"
    else:
        return "RED"


def create_timeline_marker(
    segment_num: int,
    match_result: MatchResult,
    match,
    position: float,
    duration: float,
    rate: float
) -> otio.schema.Marker:
    """Create a timeline-level marker for a segment"""
    
    confidence = match.confidence
    
    # Determine color
    if confidence >= 0.8:
        color = otio.schema.MarkerColor.GREEN
        tier = "HIGH"
    elif confidence >= 0.6:
        color = otio.schema.MarkerColor.CYAN
        tier = "GOOD"
    elif confidence >= 0.4:
        color = otio.schema.MarkerColor.YELLOW
        tier = "MED"
    elif confidence >= 0.2:
        color = otio.schema.MarkerColor.ORANGE
        tier = "LOW"
    else:
        color = otio.schema.MarkerColor.RED
        tier = "GAP"
    
    # Override to RED if it's a gap
    if match_result.has_gap:
        color = otio.schema.MarkerColor.RED
        tier = "GAP"
    
    # Build marker name - NO EMOJI for DaVinci compatibility
    vo_text = match.voiceover_segment.text[:35] + "..." if len(match.voiceover_segment.text) > 35 else match.voiceover_segment.text
    vo_text = vo_text.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
    
    marker_name = f"{tier} {confidence:.0%} - {vo_text}"
    
    # Use rounded frames to avoid drift
    start_frames = round(position * rate)
    duration_frames = round(duration * rate)
    
    marker = otio.schema.Marker(
        name=marker_name,
        marked_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start_frames, rate),
            duration=otio.opentime.RationalTime(duration_frames, rate)
        ),
        color=color
    )
    
    # Add metadata
    marker.metadata['segment_num'] = segment_num
    marker.metadata['confidence'] = confidence
    marker.metadata['voiceover_text'] = match.voiceover_segment.text
    marker.metadata['video_file'] = Path(match.video_segment.source_file).name
    marker.metadata['reasoning'] = match.reasoning
    
    return marker


def add_markers_to_clip(
    clip: otio.schema.Clip,
    match_result: MatchResult,
    match,
    rate: float,
    duration: float
):
    """Add markers to indicate match quality and properties"""
    
    # 1 frame duration for markers
    marker_duration = otio.opentime.RationalTime(1, rate)
    
    # Confidence tier marker (FIRST - most important)
    confidence = match.confidence
    
    if confidence >= 0.8:
        tier_name = f"HIGH {confidence:.0%}"
        tier_color = otio.schema.MarkerColor.GREEN
    elif confidence >= 0.6:
        tier_name = f"GOOD {confidence:.0%}"
        tier_color = otio.schema.MarkerColor.CYAN
    elif confidence >= 0.4:
        tier_name = f"MED {confidence:.0%}"
        tier_color = otio.schema.MarkerColor.YELLOW
    elif confidence >= 0.2:
        tier_name = f"LOW {confidence:.0%}"
        tier_color = otio.schema.MarkerColor.ORANGE
    else:
        tier_name = f"GAP {confidence:.0%}"
        tier_color = otio.schema.MarkerColor.RED
    
    # Override to RED if it's a gap
    if match_result.has_gap:
        tier_name = f"NEEDS REVIEW {confidence:.0%}"
        tier_color = otio.schema.MarkerColor.RED
    
    # Add confidence marker at start of clip
    conf_marker = otio.schema.Marker(
        name=tier_name,
        marked_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, rate),
            duration=marker_duration
        ),
        color=tier_color
    )
    clip.markers.append(conf_marker)
    
    # Keyword match marker (offset slightly)
    if match.is_keyword_match:
        marker = otio.schema.Marker(
            name="Keyword Match",
            marked_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(2, rate),  # 2 frames offset
                duration=marker_duration
            ),
            color=otio.schema.MarkerColor.GREEN
        )
        clip.markers.append(marker)
    
    # Visual match marker
    if match.is_visual_match:
        marker = otio.schema.Marker(
            name="Visual Match",
            marked_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(3, rate),  # 3 frames offset
                duration=marker_duration
            ),
            color=otio.schema.MarkerColor.BLUE
        )
        clip.markers.append(marker)
    
    # Reused clip marker
    if match.clip_reuse_count > 0:
        marker = otio.schema.Marker(
            name=f"Reused ({match.clip_reuse_count}x)",
            marked_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(4, rate),  # 4 frames offset
                duration=marker_duration
            ),
            color=otio.schema.MarkerColor.YELLOW
        )
        clip.markers.append(marker)
    
    # Suggested speed marker - tells user what speed to apply
    time_scalar = clip.metadata.get('time_scalar', 1.0)
    if time_scalar and abs(time_scalar - 1.0) > 0.05:
        speed_pct = time_scalar * 100
        if time_scalar > 1:
            marker_name = f"Set {speed_pct:.0f}% speed"
            color = otio.schema.MarkerColor.CYAN
        else:
            marker_name = f"Set {speed_pct:.0f}% speed"
            color = otio.schema.MarkerColor.MAGENTA
        
        marker = otio.schema.Marker(
            name=marker_name,
            marked_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(5, rate),  # 5 frames offset
                duration=marker_duration
            ),
            color=color
        )
        clip.markers.append(marker)


def save_timeline(timeline: otio.schema.Timeline, output_path: str):
    """Save timeline to OTIO file"""
    otio.adapters.write_to_file(timeline, output_path)
    logger.info(f"Saved timeline to {output_path}")


def save_timeline_as_edl(matches: List[MatchResult], output_path: str, frame_rate: float = 30.0, 
                         timeline_start_tc: str = "01:00:00:00"):
    """
    Save markers as EDL for DaVinci Resolve TIMELINE markers.
    
    These markers are placed at timeline positions, not on clips.
    Import into DaVinci: File > Import > Timeline (select EDL, check "Import markers")
    
    Format matches exact DaVinci export:
    
    TITLE: matched_timeline
    FCM: NON-DROP FRAME
    
    001  001      V     C        01:00:00:00 01:00:00:01 01:00:00:00 01:00:00:01  
     |C:ResolveColorGreen |M:Marker Name |D:1
    """
    edl_path = Path(output_path).with_suffix('.edl')
    
    # Parse timeline start timecode to frames
    start_parts = timeline_start_tc.split(':')
    start_frames = (int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + 
                    int(start_parts[2])) * int(frame_rate) + int(start_parts[3])
    
    lines = [
        "TITLE: matched_timeline",
        "FCM: NON-DROP FRAME",
        ""  # Empty line after header
    ]
    
    # Track position in frames from timeline start
    timeline_frames = start_frames
    
    for i, match_result in enumerate(matches, start=1):  # Start at 001
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        vid_seg = match.video_segment
        
        # Calculate target duration (voiceover duration) in frames
        target_duration = vo_seg.end_time - vo_seg.start_time
        source_duration = vid_seg.end_time - vid_seg.start_time
        duration_frames = round(target_duration * frame_rate)
        
        # Convert current timeline position to timecode
        tc_start = frames_to_tc(timeline_frames, frame_rate)
        tc_end = frames_to_tc(timeline_frames + 1, frame_rate)  # 1 frame ahead
        
        # Confidence tier and color
        conf = match.confidence
        if conf >= 0.8:
            tier = "HIGH"
            color = "ResolveColorGreen"
        elif conf >= 0.6:
            tier = "GOOD"
            color = "ResolveColorCyan"
        elif conf >= 0.4:
            tier = "MED"
            color = "ResolveColorYellow"
        elif conf >= 0.2:
            tier = "LOW"
            color = "ResolveColorOrange"
        else:
            tier = "GAP"
            color = "ResolveColorRed"
        
        if match_result.has_gap:
            tier = "GAP"
            color = "ResolveColorRed"
        
        # Speed percentage
        speed = (source_duration / target_duration * 100) if target_duration > 0 else 100
        
        # Marker text - clean it up for EDL
        vo_text = vo_seg.text[:35].replace('|', '-').replace('\n', ' ').replace('\r', ' ')
        marker_name = f"[{tier}] {conf:.0%} {speed:.0f}% - {vo_text}"
        
        # EDL marker format for DaVinci timeline markers
        # All 4 timecodes should be the timeline position (source=record for markers)
        lines.append(f"{i:03d}  001      V     C        {tc_start} {tc_end} {tc_start} {tc_end}  ")
        lines.append(f" |C:{color} |M:{marker_name} |D:1")
        lines.append("")  # Empty line between markers
        
        # Move to next segment position on timeline
        timeline_frames += duration_frames
    
    # Write EDL file
    with open(edl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved EDL with {len(matches)} timeline markers to {edl_path}")
    logger.info(f"  Timeline starts at {timeline_start_tc}")
    return str(edl_path)


def save_timeline_as_resolve_xml(matches: List[MatchResult], output_path: str, 
                                   voiceover_path: str = None, frame_rate: float = 30.0):
    """
    Save timeline as DaVinci Resolve compatible XML with explicit durations.
    This format handles speed changes better than OTIO.
    """
    xml_path = Path(output_path).with_suffix('.xml')
    
    # Calculate total duration from voiceover segments
    total_frames = 0
    for m in matches:
        vo_seg = m.primary_match.voiceover_segment
        target_duration = vo_seg.end_time - vo_seg.start_time
        total_frames += int(target_duration * frame_rate)
    
    fps_int = int(frame_rate)
    
    # Build XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="5">',
        '  <sequence>',
        '    <name>Matched Footage</name>',
        f'    <duration>{total_frames}</duration>',
        '    <rate>',
        f'      <timebase>{fps_int}</timebase>',
        '      <ntsc>FALSE</ntsc>',
        '    </rate>',
        '    <media>',
        '      <video>',
    ]
    
    # V1 - Primary track
    xml_lines.append('        <track>')
    timeline_pos = 0
    
    for match_result in matches:
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        vid_seg = match.video_segment
        
        target_duration = vo_seg.end_time - vo_seg.start_time
        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time
        
        target_frames = int(target_duration * frame_rate)
        source_frames = int(source_duration * frame_rate)
        source_start_frames = int(source_start * frame_rate)
        
        # Speed factor (100 = normal, 200 = 2x fast)
        speed = (source_duration / target_duration) * 100 if target_duration > 0 else 100
        
        clip_name = Path(vid_seg.source_file).stem
        
        xml_lines.extend([
            '          <clipitem>',
            f'            <name>{clip_name}</name>',
            f'            <duration>{target_frames}</duration>',
            f'            <start>{timeline_pos}</start>',
            f'            <end>{timeline_pos + target_frames}</end>',
            f'            <in>{source_start_frames}</in>',
            f'            <out>{source_start_frames + source_frames}</out>',
            '            <file>',
            f'              <pathurl>file://{vid_seg.source_file}</pathurl>',
            '            </file>',
        ])
        
        # Add speed filter if not 100%
        if abs(speed - 100) > 1:
            xml_lines.extend([
                '            <filter>',
                '              <effect>',
                '                <name>Time Remap</name>',
                '                <effectid>timeremap</effectid>',
                '                <parameter>',
                '                  <parameterid>speed</parameterid>',
                f'                  <value>{speed:.2f}</value>',
                '                </parameter>',
                '              </effect>',
                '            </filter>',
            ])
        
        # Add marker for confidence
        conf = match.confidence
        if conf >= 0.8:
            marker_color = 'green'
        elif conf >= 0.6:
            marker_color = 'cyan'
        elif conf >= 0.4:
            marker_color = 'yellow'
        else:
            marker_color = 'red'
        
        vo_text = vo_seg.text[:40].replace('<', '').replace('>', '').replace('&', '')
        xml_lines.extend([
            '            <marker>',
            '              <in>0</in>',
            f'              <name>{conf:.0%} - {vo_text}</name>',
            f'              <comment>{marker_color}</comment>',
            '            </marker>',
        ])
        
        xml_lines.append('          </clipitem>')
        timeline_pos += target_frames
    
    xml_lines.append('        </track>')
    
    # V2, V3 - Alternative tracks (similar structure)
    for alt_idx in range(2):
        xml_lines.append('        <track>')
        timeline_pos = 0
        
        for match_result in matches:
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = vo_seg.end_time - vo_seg.start_time
            target_frames = int(target_duration * frame_rate)
            
            if alt_idx < len(match_result.alternatives):
                alt = match_result.alternatives[alt_idx]
                alt_seg = alt.video_segment
                
                source_duration = alt_seg.end_time - alt_seg.start_time
                source_start = alt_seg.start_time
                source_frames = int(source_duration * frame_rate)
                source_start_frames = int(source_start * frame_rate)
                speed = (source_duration / target_duration) * 100 if target_duration > 0 else 100
                
                clip_name = Path(alt_seg.source_file).stem
                
                xml_lines.extend([
                    '          <clipitem>',
                    f'            <name>ALT{alt_idx+1}: {clip_name}</name>',
                    f'            <duration>{target_frames}</duration>',
                    f'            <start>{timeline_pos}</start>',
                    f'            <end>{timeline_pos + target_frames}</end>',
                    f'            <in>{source_start_frames}</in>',
                    f'            <out>{source_start_frames + source_frames}</out>',
                    '            <file>',
                    f'              <pathurl>file://{alt_seg.source_file}</pathurl>',
                    '            </file>',
                ])
                
                if abs(speed - 100) > 1:
                    xml_lines.extend([
                        '            <filter>',
                        '              <effect>',
                        '                <name>Time Remap</name>',
                        '                <effectid>timeremap</effectid>',
                        '                <parameter>',
                        '                  <parameterid>speed</parameterid>',
                        f'                  <value>{speed:.2f}</value>',
                        '                </parameter>',
                        '              </effect>',
                        '            </filter>',
                    ])
                
                xml_lines.append('          </clipitem>')
            
            timeline_pos += target_frames
        
        xml_lines.append('        </track>')
    
    xml_lines.extend([
        '      </video>',
        '      <audio>',
        '        <track>',
    ])
    
    # A4 - Voiceover track
    if voiceover_path:
        xml_lines.extend([
            '          <clipitem>',
            '            <name>Voiceover</name>',
            f'            <duration>{total_frames}</duration>',
            '            <start>0</start>',
            f'            <end>{total_frames}</end>',
            '            <in>0</in>',
            f'            <out>{total_frames}</out>',
            '            <file>',
            f'              <pathurl>file://{voiceover_path}</pathurl>',
            '            </file>',
            '          </clipitem>',
        ])
    
    xml_lines.extend([
        '        </track>',
        '      </audio>',
        '    </media>',
        '  </sequence>',
        '</xmeml>',
    ])
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(xml_lines))
    
    logger.info(f"Saved DaVinci XML to {xml_path}")
    return str(xml_path)


def frames_to_tc(frames: int, fps: float = 30.0) -> str:
    """Convert frame count to timecode string HH:MM:SS:FF"""
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frame = int(frames % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"


def generate_match_report(matches: List[MatchResult], output_path: str, config=None):
    """Generate a detailed match report"""
    
    # Count by confidence tier
    high = sum(1 for m in matches if m.primary_match.confidence >= 0.8)
    good = sum(1 for m in matches if 0.6 <= m.primary_match.confidence < 0.8)
    med = sum(1 for m in matches if 0.4 <= m.primary_match.confidence < 0.6)
    low = sum(1 for m in matches if 0.2 <= m.primary_match.confidence < 0.4)
    gap = sum(1 for m in matches if m.primary_match.confidence < 0.2 or m.has_gap)
    
    # Strategy track info
    strategy_names = []
    if config and hasattr(config, 'output') and config.output.include_strategy_tracks:
        strategy_names = config.output.strategy_tracks
    
    strategy_display = {
        "visual_first": "Visual-First (prioritizes scene descriptions)",
        "different_source": "Different-Source (forces different video file)",
        "keyword_only": "Keyword-Only (matches on keywords/entities)",
        "embedding_diversity": "Embedding-Diversity (maximally different from V1)"
    }
    
    report_lines = [
        "# Voiceover-to-Footage Match Report",
        "",
        f"Total segments: {len(matches)}",
        "",
        "## Confidence Tiers",
        f"- [HIGH] (>=80%): {high}",
        f"- [GOOD] (60-80%): {good}",
        f"- [MED] (40-60%): {med}",
        f"- [LOW] (20-40%): {low}",
        f"- [GAP] (<20%): {gap}",
        "",
        f"Gaps (needs review): {sum(1 for m in matches if m.has_gap)}",
        f"Keyword matches: {sum(1 for m in matches if m.primary_match.is_keyword_match)}",
        f"Visual matches: {sum(1 for m in matches if m.primary_match.is_visual_match)}",
        "",
        "## Track Structure",
        "- V1: Primary video (enabled)",
        "- V2: Alternative 1 (disabled)",
        "- V3: Alternative 2 (disabled)",
    ]
    
    # Add strategy tracks to structure
    for i, strat in enumerate(strategy_names):
        track_num = 4 + i
        display = strategy_display.get(strat, strat)
        report_lines.append(f"- V{track_num}: {display} (disabled)")
    
    report_lines.extend([
        "- A1-A3: Audio from V1-V3",
    ])
    
    if strategy_names:
        report_lines.append(f"- A4-A{3+len(strategy_names)}: Audio from strategy tracks")
    
    report_lines.extend([
        f"- A{4+len(strategy_names)}: Voiceover (enabled)",
        "",
        "## Strategy Track Stats",
    ])
    
    # Strategy statistics
    for strat in strategy_names:
        count = sum(1 for m in matches if m.strategy_matches and any(sm.strategy == strat for sm in m.strategy_matches))
        report_lines.append(f"- {strat}: {count}/{len(matches)} segments matched")
    
    report_lines.extend([
        "",
        "## Matches",
        ""
    ])
    
    for i, match_result in enumerate(matches, 1):
        match = match_result.primary_match
        vo = match.voiceover_segment
        vid = match.video_segment
        
        # Calculate speed
        vo_dur = vo.end_time - vo.start_time
        vid_dur = vid.end_time - vid.start_time
        speed = (vid_dur / vo_dur * 100) if vo_dur > 0 else 100
        
        # Determine tier
        conf = match.confidence
        if conf >= 0.8:
            tier = "[HIGH]"
        elif conf >= 0.6:
            tier = "[GOOD]"
        elif conf >= 0.4:
            tier = "[MED]"
        elif conf >= 0.2:
            tier = "[LOW]"
        else:
            tier = "[GAP]"
        
        if match_result.has_gap:
            tier = "[GAP]"
        
        status = "[!]" if match_result.has_gap else "[OK]"
        kw = " [KW]" if match.is_keyword_match else ""
        vis = " [VIS]" if match.is_visual_match else ""
        
        # Speed info - tells user what to set in DaVinci
        speed_note = ""
        if speed > 105:
            speed_note = f" >> Set to {speed:.0f}%"
        elif speed < 95:
            speed_note = f" << Set to {speed:.0f}%"
        
        report_lines.extend([
            f"### {i}. {status} {tier}{kw}{vis}",
            f"**Voiceover**: \"{vo.text}\" ({vo_dur:.1f}s)",
            f"**Video**: {Path(vid.source_file).name} @ {vid.start_time:.1f}s ({vid_dur:.1f}s)",
            f"**Suggested Speed**: {speed:.0f}%{speed_note}",
            f"**Confidence**: {match.confidence:.2f}",
            f"**Reasoning**: {match.reasoning}",
            ""
        ])
        
        if match_result.alternatives:
            report_lines.append("**Alternatives (V2-V3)**:")
            for j, alt in enumerate(match_result.alternatives, 1):
                alt_dur = alt.video_segment.end_time - alt.video_segment.start_time
                alt_speed = (alt_dur / vo_dur * 100) if vo_dur > 0 else 100
                report_lines.append(
                    f"  V{j+1}. {Path(alt.video_segment.source_file).name} "
                    f"({alt.confidence:.2f}, {alt_speed:.0f}% speed)"
                )
            report_lines.append("")
        
        # Strategy matches
        if match_result.strategy_matches:
            report_lines.append("**Strategy Tracks (V4+)**:")
            for sm in match_result.strategy_matches:
                sm_dur = sm.video_segment.end_time - sm.video_segment.start_time
                sm_speed = (sm_dur / vo_dur * 100) if vo_dur > 0 else 100
                strat_idx = strategy_names.index(sm.strategy) if sm.strategy in strategy_names else 0
                report_lines.append(
                    f"  V{4+strat_idx}. [{sm.strategy}] {Path(sm.video_segment.source_file).name} "
                    f"({sm.confidence:.2f}, {sm_speed:.0f}% speed) - {sm.reasoning}"
                )
            report_lines.append("")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved match report to {output_path}")
