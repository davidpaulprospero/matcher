"""
OTIO Timeline builder with:
- Speed-adjusted video clips to match voiceover duration
- Parallel audio tracks (A1-A3) matching video tracks (V1-V3)
- LinearTimeWarp for speed adjustment
- Alternative matches as disabled tracks
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import opentimelineio as otio

from .config import Config
from .utils import SRTSegment, MatchResult, AlternativeMatch
from .embeddings import cosine_similarity

logger = logging.getLogger(__name__)


def _to_windows_path(path: str) -> str:
    """
    Convert path to Windows format with backslashes.
    
    DaVinci Resolve requires Windows-style paths: E:\\folder\\file.mp4
    Forward slashes cause import issues.
    """
    abs_path = str(Path(path).resolve())
    # Ensure backslashes (Windows format)
    return abs_path.replace('/', '\\')


def escape_xml(text: str) -> str:
    """Escape special XML characters in text."""
    return (str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;'))


def format_path_url(file_path: str) -> str:
    """Format file path for DaVinci Resolve XML - use standard path with forward slashes."""
    path = str(Path(file_path).resolve()).replace('\\', '/')
    # Return plain path - DaVinci prefers standard paths over file:// URLs
    return path


def _to_python_type(value):
    """Convert numpy types to native Python types for OTIO compatibility."""
    if value is None:
        return None
    
    # Check for numpy types
    type_name = type(value).__name__
    module_name = type(value).__module__
    
    # Handle numpy scalar types
    if module_name == 'numpy' or 'numpy' in str(type(value)):
        # numpy float types
        if 'float' in type_name.lower():
            return float(value)
        # numpy int types
        elif 'int' in type_name.lower():
            return int(value)
        # numpy bool
        elif 'bool' in type_name.lower():
            return bool(value)
        # numpy string types
        elif 'str' in type_name.lower():
            return str(value)
        # numpy array - convert to list
        elif hasattr(value, 'tolist'):
            return value.tolist()
    
    # Handle lists recursively
    if isinstance(value, list):
        return [_to_python_type(v) for v in value]
    
    # Handle dicts recursively
    if isinstance(value, dict):
        return {k: _to_python_type(v) for k, v in value.items()}
    
    return value


def _sanitize_metadata(metadata: dict) -> dict:
    """Convert all metadata values to OTIO-compatible Python types."""
    return {k: _to_python_type(v) for k, v in metadata.items()}


def sanitize_path_for_url(path: str) -> str:
    r"""
    Sanitize a file path for use as a URL in OTIO.
    
    Handles:
    - Windows extended-length paths (\\?\C:\...)
    - Backslashes to forward slashes
    - Proper file:// URL format
    
    Note: Does NOT URL-encode. DaVinci Resolve doesn't want encoded paths.
    Problematic characters (%, &, $, #) should be sanitized at download time.
    """
    # Convert to string if Path object
    path = str(path)
    
    # Remove Windows extended-length path prefix in various forms
    # Check multiple patterns to be safe
    prefixes_to_remove = [
        '\\\\?\\',    # Standard form: \\?\
        '\\\\.\\',    # Device form: \\.\
        '//?/',       # Forward slash form
        '//.//',      # Device forward slash
        '\\?\\',      # Single backslash form (shouldn't happen but just in case)
    ]
    
    for prefix in prefixes_to_remove:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    
    # Also check if it starts with ?\  or ?/ after any conversions
    if path.startswith('?\\') or path.startswith('?/'):
        path = path[2:]
    
    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')
    
    # Remove any double slashes (except at start for UNC paths - but we don't want UNC)
    while '//' in path:
        path = path.replace('//', '/')
    
    # For absolute Windows paths (C:/...), ensure proper format
    # Don't add file:// prefix - let OTIO/NLE handle it
    
    return path


def encode_path_for_xml_url(path: str) -> str:
    """
    Format a file path for use in XML pathurl elements.
    
    DaVinci Resolve expects standard Windows paths (E:/folder/file.mp4),
    NOT file:// URL format (file://localhost/E:/...) which causes hangs.
    
    Args:
        path: File path (can be Windows or Unix style)
        
    Returns:
        Clean file path for DaVinci Resolve (forward slashes)
    """
    # First sanitize the path (remove extended-length prefix, convert slashes)
    path = sanitize_path_for_url(path)
    
    # Return plain path with forward slashes - no file:// prefix
    # DaVinci Resolve handles this format natively
    return path


def create_clip_with_timewarp(
    name: str,
    source_path: str,
    source_start: float,
    source_duration: float,
    target_duration: float,
    frame_rate: float = 30.0,
    metadata: Optional[Dict] = None,
    media_duration: float = None  # Total duration of the source media file
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
        media_duration: Total duration of the source media file (for available_range)
    
    Returns:
        OTIO Clip with duration matching target_duration
    """
    rate = frame_rate
    
    # Create absolute Windows path with backslashes for DaVinci Resolve
    abs_path = _to_windows_path(source_path)
    
    # Make media reference name unique by including parent folder
    # This prevents DaVinci Resolve from confusing clips with same filename in different folders
    folder_name = Path(source_path).parent.name
    filename = Path(source_path).name
    unique_media_name = f"{folder_name}_{filename}"
    
    # Determine available_range for the media file
    # If we don't know the media duration, estimate from source_start + source_duration
    if media_duration is None:
        # Estimate: assume media is at least as long as what we're using
        estimated_duration = source_start + source_duration + 10  # Add buffer
        media_duration = estimated_duration
    
    available_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, rate),
        duration=otio.opentime.RationalTime(round(media_duration * rate), rate)
    )
    
    # Create media reference with proper format for DaVinci Resolve
    # Note: name must be set as attribute, not constructor param
    media_ref = otio.schema.ExternalReference(
        target_url=abs_path,
        available_range=available_range
    )
    media_ref.name = unique_media_name  # Unique name includes folder
    
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
    
    # Add Resolve_OTIO metadata (required for DaVinci import)
    clip.metadata['Resolve_OTIO'] = {}
    
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
    
    # Add metadata (convert numpy types to Python native types)
    metadata = _sanitize_metadata(metadata)
    for key, value in metadata.items():
        clip.metadata[key] = value
    
    return clip


def create_timeline(
    matches: List[MatchResult],
    config: Config,
    voiceover_path: Optional[str] = None,
    frame_rate: float = 30.0,
    entity_images: Optional[Dict] = None,
    entity_videos: Optional[Dict] = None
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
    - V9: Entity Images (Google stills) - disabled
    - V10: Stock Videos (Pexels/Pixabay) - disabled
    - A1-A7: Corresponding audio tracks
    - A8: Voiceover - enabled
    """
    
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
    
    # Strategy tracks (V7 only - embedding_diversity)
    strategy_names = []
    if config.output.include_strategy_tracks:
        # Only include embedding_diversity for V7
        if "embedding_diversity" in config.output.strategy_tracks:
            strategy_names = ["embedding_diversity"]
    
    strategy_display_names = {
        "embedding_diversity": "Embedding-Diversity"
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
    
    # V7: Strategy tracks (embedding_diversity only)
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + num_secondary + i + 1  # V7
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
    
    # A7: Strategy audio tracks (embedding_diversity only)
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + num_secondary + i + 1  # A7
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
        clip_folder = Path(vid_seg.source_file).parent.name
        clip_stem = Path(vid_seg.source_file).stem
        v1_clip = create_clip_with_timewarp(
            name=f"[{segment_id}] {clip_folder}_{clip_stem} [{vid_seg.start_time:.1f}s]",
            source_path=vid_seg.source_file,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata=metadata
        )
        
        # Set clip color
        v1_clip.metadata['clip_color'] = clip_color
        
        # Note: Markers removed - not used in DaVinci workflow
        
        video_tracks[0].append(v1_clip)
        
        # Create primary audio clip (A1) - same source, same timing
        a1_clip = create_clip_with_timewarp(
            name=f"[{segment_id}] Audio: {clip_folder}_{clip_stem}",
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
                
                # Alternative video clip - include segment ID for tracing
                alt_folder = Path(alt_seg.source_file).parent.name
                alt_stem = Path(alt_seg.source_file).stem
                alt_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] ALT{alt_idx+1}: {alt_folder}_{alt_stem}",
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
                    name=f"[{segment_id}] Audio ALT{alt_idx+1}: {alt_folder}_{alt_stem}",
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
        
        # Process secondary tracks (V4-V6) - different video files from V1-V3
        secondary_base_idx = 1 + num_alternatives  # Index where secondary tracks start
        
        for sec_idx in range(num_secondary):
            track_idx = secondary_base_idx + sec_idx
            
            if sec_idx < len(match_result.secondary_matches):
                sec_match = match_result.secondary_matches[sec_idx]
                sec_seg = sec_match.video_segment
                sec_source_duration = sec_seg.end_time - sec_seg.start_time
                sec_source_start = sec_seg.start_time
                
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
                sec_folder = Path(sec_seg.source_file).parent.name
                sec_stem = Path(sec_seg.source_file).stem
                sec_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] {sec_label}: {sec_folder}_{sec_stem}",
                    source_path=sec_seg.source_file,
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
                    source_path=sec_seg.source_file,
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
        
        # Process strategy tracks (V7 - embedding_diversity only)
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
                strat_folder = Path(strat_seg.source_file).parent.name
                strat_stem = Path(strat_seg.source_file).stem
                strat_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] {strategy.upper()}: {strat_folder}_{strat_stem}",
                    source_path=strat_seg.source_file,
                    source_start=strat_source_start,
                    source_duration=strat_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=strat_metadata
                )
                
                # Color based on strategy
                strategy_colors = {
                    "embedding_diversity": "PINK"
                }
                strat_v_clip.metadata['clip_color'] = strategy_colors.get(strategy, "GRAY")
                
                video_tracks[track_idx].append(strat_v_clip)
                
                # Strategy audio clip
                strat_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio {strategy.upper()}: {strat_folder}_{strat_stem}",
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
        # Create absolute path for voiceover (Windows format for DaVinci)
        abs_vo_path = _to_windows_path(voiceover_path)
        vo_folder = Path(voiceover_path).parent.name
        vo_filename = Path(voiceover_path).name
        vo_unique_name = f"{vo_folder}_{vo_filename}"
        
        # Total duration should match total frames accumulated
        total_frames = timeline_frames
        total_duration_seconds = total_frames / rate
        
        # Create proper ExternalReference with available_range
        vo_available_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, rate),
            duration=otio.opentime.RationalTime(total_frames, rate)
        )
        
        vo_ref = otio.schema.ExternalReference(
            target_url=abs_vo_path,
            available_range=vo_available_range
        )
        vo_ref.name = vo_unique_name  # Unique name includes folder
        
        vo_clip = otio.schema.Clip(
            name="Voiceover",
            media_reference=vo_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(total_frames, rate)
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
        _add_entity_images_to_track(
            image_track=image_track,
            entity_images=entity_images,
            matches=matches,
            frame_rate=rate
        )

    # Always add V9 Entity Images track (even if empty, for manual use)
    timeline.tracks.append(image_track)

    # Populate stock video track if entity_videos provided
    if entity_videos:
        _add_entity_videos_to_track(
            video_track=stock_video_track,
            entity_videos=entity_videos,
            matches=matches,
            frame_rate=rate
        )

    # Always add V10 Stock Videos track (even if empty, for manual use)
    timeline.tracks.append(stock_video_track)
    
    # Note: Timeline markers removed - not used in DaVinci workflow
    
    return timeline


def _find_best_entity_match(
    vo_text: str,
    entity_dict: Dict,
    last_matched_entity: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Find the best matching entity for a voiceover segment.

    Strategy:
    1. Exact match: Entity name appears in voiceover text
    2. Semantic match: Use embedding similarity between voiceover and entity query
    3. Sticky: Use last matched entity if no match found

    Args:
        vo_text: Voiceover segment text (lowercase)
        entity_dict: Dict of entity_name -> EntityResult
        last_matched_entity: Previous segment's matched entity name

    Returns:
        (entity_name, match_type) where match_type is 'exact', 'semantic', or 'sticky'
    """
    # 1. Try exact match first
    for entity_name, entity_result in entity_dict.items():
        if entity_name.lower() in vo_text:
            assets = getattr(entity_result, 'images', None) or getattr(entity_result, 'videos', None)
            if assets:
                return entity_name, 'exact'

    # 2. Try semantic matching using entity query similarity
    best_entity = None
    best_score = 0.0

    try:
        # Simple word overlap scoring as semantic proxy
        # (Full embedding similarity would require pre-computed embeddings)
        vo_words = set(vo_text.split())

        for entity_name, entity_result in entity_dict.items():
            assets = getattr(entity_result, 'images', None) or getattr(entity_result, 'videos', None)
            if not assets:
                continue

            # Get query text for matching
            query = getattr(entity_result, 'query', entity_name)
            query_words = set(query.lower().split())

            # Also include entity type in matching
            entity_type = getattr(entity_result, 'entity_type', '')
            if entity_type:
                query_words.update(entity_type.lower().split())

            # Calculate word overlap score
            common_words = vo_words & query_words
            if common_words:
                # Jaccard-like similarity
                score = len(common_words) / (len(vo_words | query_words) + 1)
                if score > best_score:
                    best_score = score
                    best_entity = entity_name

        # Require minimum semantic score threshold
        if best_entity and best_score >= 0.05:
            return best_entity, 'semantic'
    except Exception as e:
        logger.debug(f"Semantic matching failed: {e}")

    # 3. Fall back to sticky entity
    if last_matched_entity and last_matched_entity in entity_dict:
        entity_result = entity_dict[last_matched_entity]
        assets = getattr(entity_result, 'images', None) or getattr(entity_result, 'videos', None)
        if assets:
            return last_matched_entity, 'sticky'

    return None, 'none'


def _add_entity_images_to_track(
    image_track: otio.schema.Track,
    entity_images: Dict,
    matches: List[MatchResult],
    frame_rate: float
):
    """
    Add entity images to V9 track at segment positions.

    Features:
    - Sticky entity: Last matched entity persists to fill subsequent segments
    - Semantic matching: Uses word overlap to find relevant entities
    - No duplicate sources: Same image won't appear twice in one segment

    ALL images for an entity are placed as separate clips within
    the segment, divided equally by duration.

    Example: 10 second segment with 5 images = 5 clips of 2 seconds each

    Format matches DaVinci Resolve's OTIO export:
    - media_references dict with DEFAULT_MEDIA key
    - available_range = 1 frame (still image)
    - source_range = display duration
    - active_media_reference_key = "DEFAULT_MEDIA"
    """
    rate = frame_rate

    # Build segment timing map: segment_index -> (start_frame, duration_frames, duration_sec)
    segment_timing = {}
    current_frame = 0

    for i, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        target_duration = vo_seg.end_time - vo_seg.start_time
        duration_frames = round(target_duration * frame_rate)

        segment_timing[i] = (current_frame, duration_frames, target_duration)
        current_frame += duration_frames

    # Track sticky entity across segments
    last_matched_entity = None

    # Process each segment
    for seg_idx, (start_frame, duration_frames, duration_sec) in segment_timing.items():
        match = matches[seg_idx].primary_match
        vo_text = match.voiceover_segment.text.lower()

        # Find best matching entity (exact -> semantic -> sticky)
        entity_name, match_type = _find_best_entity_match(
            vo_text, entity_images, last_matched_entity
        )

        if entity_name:
            entity_result = entity_images[entity_name]

            # Update sticky entity for next segments
            last_matched_entity = entity_name

            # Get all images for this entity
            all_images = entity_result.images
            num_images = len(all_images)

            if num_images > 0:
                # Track used sources within this segment to prevent duplicates
                used_sources = set()
                unique_images = []
                for img_path in all_images:
                    # Use filename as source identifier
                    source_id = Path(img_path).name
                    if source_id not in used_sources:
                        used_sources.add(source_id)
                        unique_images.append(img_path)

                # Use deduplicated images
                all_images = unique_images
                num_images = len(all_images)

                # Divide segment duration equally among all images
                frames_per_image = max(1, duration_frames // num_images)
                remaining_frames = duration_frames - (frames_per_image * num_images)

                # Create a clip for each image
                for img_idx, image_path in enumerate(all_images):
                    # Calculate this image's duration (distribute remaining frames to last clips)
                    clip_frames = frames_per_image
                    if img_idx >= num_images - remaining_frames:
                        clip_frames += 1

                    # Get folder and filename for unique reference name
                    image_path_obj = Path(image_path)
                    image_folder = image_path_obj.parent.name
                    image_filename = image_path_obj.name
                    # Include segment ID for post-edit analysis tracing
                    segment_id = f"[S{seg_idx:03d}]"
                    image_unique_name = f"{segment_id} {image_folder}_{image_filename}"

                    # Convert to Windows path format with backslashes for Resolve
                    image_path_resolved = _to_windows_path(image_path)

                    # Create external reference for still image
                    # CRITICAL: available_range = 1 frame signals to Resolve this is a still image
                    # Resolve then auto-holds (repeats) this single frame for the full source_range
                    # DO NOT set to null - that causes "media offline" because Resolve can't
                    # determine the media's temporal extent
                    image_ref = otio.schema.ExternalReference(
                        target_url=image_path_resolved,
                        available_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(1, rate)  # 1 frame = still image
                        )
                    )
                    image_ref.name = image_unique_name

                    # Create clip with source_range = display duration
                    # source_range.start_time MUST be 0 for stills (cannot advance frames that don't exist)
                    # source_range.duration = how long to display the still (Resolve auto-holds it)
                    image_clip = otio.schema.Clip(
                        name=image_unique_name,
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),  # Always 0 for stills
                            duration=otio.opentime.RationalTime(clip_frames, rate)
                        )
                    )

                    # Set media references using the Resolve format
                    # This uses the newer OTIO format with media_references dict
                    image_clip.media_reference = image_ref

                    # Add Resolve_OTIO metadata (required for DaVinci import)
                    image_clip.metadata['Resolve_OTIO'] = {}

                    # Add metadata
                    image_clip.metadata['entity_name'] = entity_name
                    image_clip.metadata['entity_type'] = entity_result.entity_type
                    image_clip.metadata['query'] = entity_result.query
                    image_clip.metadata['image_path'] = image_path
                    image_clip.metadata['segment_index'] = seg_idx
                    image_clip.metadata['image_index'] = img_idx
                    image_clip.metadata['total_images'] = num_images
                    image_clip.metadata['is_still_image'] = True
                    image_clip.metadata['match_type'] = match_type  # Track how entity was matched

                    # Add Resolve-specific metadata
                    image_clip.metadata['Resolve_OTIO'] = {}

                    image_track.append(image_clip)

                # Successfully added images, continue to next segment
                continue

        # No entity matched even with fallbacks - add gap to maintain sync
        gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(duration_frames, rate)
            )
        )
        image_track.append(gap)


def _add_entity_videos_to_track(
    video_track: otio.schema.Track,
    entity_videos: Dict,
    matches: List[MatchResult],
    frame_rate: float
):
    """
    Add stock videos to V10 track at segment positions.

    Features:
    - Sticky entity: Last matched entity persists to fill subsequent segments
    - Semantic matching: Uses word overlap to find relevant entities
    - No duplicate sources: Same video source won't appear twice in one segment

    ALL videos for an entity are placed as separate clips within
    the segment, divided equally by duration.

    Unlike images, videos have actual duration. We use the video's
    natural duration but may need to trim/adjust to fit the segment.
    """
    rate = frame_rate

    # Build segment timing map: segment_index -> (start_frame, duration_frames, duration_sec)
    segment_timing = {}
    current_frame = 0

    for i, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        target_duration = vo_seg.end_time - vo_seg.start_time
        duration_frames = round(target_duration * frame_rate)

        segment_timing[i] = (current_frame, duration_frames, target_duration)
        current_frame += duration_frames

    # Track sticky entity across segments
    last_matched_entity = None

    # Process each segment
    for seg_idx, (start_frame, duration_frames, duration_sec) in segment_timing.items():
        match = matches[seg_idx].primary_match
        vo_text = match.voiceover_segment.text.lower()

        # Find best matching entity (exact -> semantic -> sticky)
        entity_name, match_type = _find_best_entity_match(
            vo_text, entity_videos, last_matched_entity
        )

        if entity_name:
            entity_result = entity_videos[entity_name]

            # Update sticky entity for next segments
            last_matched_entity = entity_name

            # Get all videos for this entity
            all_videos = entity_result.videos
            num_videos = len(all_videos)

            if num_videos > 0:
                # Track used sources within this segment to prevent duplicates
                used_sources = set()
                unique_videos = []
                for vid_path in all_videos:
                    # Use filename as source identifier
                    source_id = Path(vid_path).name
                    if source_id not in used_sources:
                        used_sources.add(source_id)
                        unique_videos.append(vid_path)

                # Use deduplicated videos
                all_videos = unique_videos
                num_videos = len(all_videos)

                # Divide segment duration equally among all videos
                frames_per_video = max(1, duration_frames // num_videos)
                remaining_frames = duration_frames - (frames_per_video * num_videos)

                # Create a clip for each video
                for vid_idx, video_path in enumerate(all_videos):
                    # Calculate this video's display duration
                    clip_frames = frames_per_video
                    if vid_idx >= num_videos - remaining_frames:
                        clip_frames += 1

                    clip_duration_sec = clip_frames / rate

                    # Get folder and filename for unique reference name
                    video_path_obj = Path(video_path)
                    video_folder = video_path_obj.parent.name
                    video_filename = video_path_obj.name
                    # Include segment ID for post-edit analysis tracing
                    segment_id = f"[S{seg_idx:03d}]"
                    video_unique_name = f"{segment_id} {video_folder}_{video_filename}"

                    # Convert to Windows path format with backslashes for Resolve
                    video_path_resolved = _to_windows_path(video_path)

                    # Get actual video duration using ffprobe if available
                    actual_duration_frames = _get_video_duration_frames(video_path, rate)
                    if actual_duration_frames is None:
                        # Fallback: assume video is long enough
                        actual_duration_frames = clip_frames * 2

                    # Create external reference for video
                    # Unlike images, videos have actual duration (available_range)
                    video_ref = otio.schema.ExternalReference(
                        target_url=video_path_resolved,
                        available_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(actual_duration_frames, rate)
                        )
                    )
                    video_ref.name = video_unique_name

                    # Create clip - use portion of video that fits segment
                    # Start from beginning, play for clip_frames duration
                    video_clip = otio.schema.Clip(
                        name=video_unique_name,
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(clip_frames, rate)
                        )
                    )

                    video_clip.media_reference = video_ref

                    # Add Resolve_OTIO metadata (required for DaVinci import)
                    video_clip.metadata['Resolve_OTIO'] = {}

                    # Add metadata
                    video_clip.metadata['entity_name'] = entity_name
                    video_clip.metadata['entity_type'] = entity_result.entity_type
                    video_clip.metadata['query'] = entity_result.query
                    video_clip.metadata['video_path'] = video_path
                    video_clip.metadata['segment_index'] = seg_idx
                    video_clip.metadata['video_index'] = vid_idx
                    video_clip.metadata['total_videos'] = num_videos
                    video_clip.metadata['source'] = 'stock_video'
                    video_clip.metadata['match_type'] = match_type  # Track how entity was matched

                    # Add Resolve-specific metadata
                    video_clip.metadata['Resolve_OTIO'] = {}

                    video_track.append(video_clip)

                # Successfully added videos, continue to next segment
                continue

        # No entity matched even with fallbacks - add gap to maintain sync
        gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(duration_frames, rate)
            )
        )
        video_track.append(gap)


def _get_video_duration_frames(video_path: str, frame_rate: float) -> Optional[int]:
    """
    Get video duration in frames using ffprobe.
    
    Returns None if ffprobe fails or is not available.
    """
    import subprocess
    
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and result.stdout.strip():
            duration_sec = float(result.stdout.strip())
            return int(duration_sec * frame_rate)
    except Exception:
        pass
    
    return None


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


def print_timeline_statistics(timeline: otio.schema.Timeline):
    """
    Print a comprehensive checklist of timeline statistics.

    Shows:
    - Track counts (video/audio)
    - Clips vs gaps per track
    - Entity match types (exact/semantic/sticky)
    - Total duration
    - Segment coverage
    """
    print("\n" + "=" * 60)
    print("  OTIO TIMELINE STATISTICS")
    print("=" * 60)

    video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
    audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

    # Calculate total duration
    total_duration = 0.0
    if video_tracks and video_tracks[0]:
        for item in video_tracks[0]:
            if hasattr(item, 'source_range') and item.source_range:
                total_duration += item.source_range.duration.to_seconds()

    print(f"\n  Timeline: {timeline.name}")
    print(f"  Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Video Tracks: {len(video_tracks)}")
    print(f"  Audio Tracks: {len(audio_tracks)}")

    print("\n  " + "-" * 56)
    print("  TRACK BREAKDOWN")
    print("  " + "-" * 56)
    print(f"  {'Track':<25} {'Clips':>8} {'Gaps':>8} {'Coverage':>10}")
    print("  " + "-" * 56)

    # Track statistics
    for track in video_tracks:
        clips = 0
        gaps = 0
        clip_duration = 0.0
        gap_duration = 0.0

        for item in track:
            if isinstance(item, otio.schema.Clip):
                clips += 1
                if item.source_range:
                    clip_duration += item.source_range.duration.to_seconds()
            elif isinstance(item, otio.schema.Gap):
                gaps += 1
                if item.source_range:
                    gap_duration += item.source_range.duration.to_seconds()

        track_total = clip_duration + gap_duration
        coverage = (clip_duration / track_total * 100) if track_total > 0 else 0

        track_name = track.name[:25] if track.name else "Unnamed"
        status = "✓" if coverage > 0 else "○"
        print(f"  {status} {track_name:<23} {clips:>8} {gaps:>8} {coverage:>9.1f}%")

    # Entity matching statistics (V9/V10)
    print("\n  " + "-" * 56)
    print("  ENTITY MATCHING (V9 Images / V10 Stock Videos)")
    print("  " + "-" * 56)

    match_types = {'exact': 0, 'semantic': 0, 'sticky': 0, 'unknown': 0}
    entity_clips = 0

    for track in video_tracks:
        track_name = track.name.lower() if track.name else ""
        if 'v9' in track_name or 'v10' in track_name or 'image' in track_name or 'stock' in track_name:
            for item in track:
                if isinstance(item, otio.schema.Clip):
                    entity_clips += 1
                    if hasattr(item, 'metadata') and 'match_type' in item.metadata:
                        mt = item.metadata['match_type']
                        if mt in match_types:
                            match_types[mt] += 1
                        else:
                            match_types['unknown'] += 1
                    else:
                        match_types['unknown'] += 1

    if entity_clips > 0:
        print(f"  Total entity clips: {entity_clips}")
        print(f"    ✓ Exact matches:    {match_types['exact']:>4} ({match_types['exact']/entity_clips*100:.1f}%)")
        print(f"    ~ Semantic matches: {match_types['semantic']:>4} ({match_types['semantic']/entity_clips*100:.1f}%)")
        print(f"    → Sticky (carried): {match_types['sticky']:>4} ({match_types['sticky']/entity_clips*100:.1f}%)")
        if match_types['unknown'] > 0:
            print(f"    ? Unknown:          {match_types['unknown']:>4}")
    else:
        print("  No entity clips found (V9/V10 empty or not provided)")

    # Segment IDs check
    print("\n  " + "-" * 56)
    print("  SEGMENT ID COVERAGE")
    print("  " + "-" * 56)

    segment_ids = set()
    clips_with_ids = 0
    clips_without_ids = 0

    for track in video_tracks:
        for item in track:
            if isinstance(item, otio.schema.Clip) and item.name:
                if '[S' in item.name and ']' in item.name:
                    clips_with_ids += 1
                    # Extract segment number
                    import re
                    match = re.search(r'\[S(\d+)\]', item.name)
                    if match:
                        segment_ids.add(int(match.group(1)))
                else:
                    clips_without_ids += 1

    total_clips = clips_with_ids + clips_without_ids
    if total_clips > 0:
        print(f"  Clips with segment IDs:    {clips_with_ids:>4} ({clips_with_ids/total_clips*100:.1f}%)")
        print(f"  Clips without segment IDs: {clips_without_ids:>4}")
        print(f"  Unique segments:           {len(segment_ids):>4}")
        if segment_ids:
            print(f"  Segment range:             S{min(segment_ids):03d} - S{max(segment_ids):03d}")

    # Summary checklist
    print("\n  " + "-" * 56)
    print("  QUALITY CHECKLIST")
    print("  " + "-" * 56)

    v1_clips = 0
    for track in video_tracks:
        if track.name and 'V1' in track.name:
            v1_clips = sum(1 for item in track if isinstance(item, otio.schema.Clip))
            break

    checks = [
        ("V1 Primary track populated", v1_clips > 0),
        ("Segment IDs present", clips_with_ids > 0),
        ("Multiple video tracks", len(video_tracks) >= 3),
        ("Audio track present", len(audio_tracks) > 0),
    ]

    # Check for entity tracks
    has_v9 = any('v9' in (t.name or '').lower() or 'image' in (t.name or '').lower() for t in video_tracks)
    has_v10 = any('v10' in (t.name or '').lower() or 'stock' in (t.name or '').lower() for t in video_tracks)
    checks.append(("V9 Entity Images track", has_v9))
    checks.append(("V10 Stock Videos track", has_v10))

    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {check_name}")

    print("\n" + "=" * 60 + "\n")


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



def save_timeline_as_edl(matches: List[MatchResult], output_path: str, frame_rate: float = 30.0, 
                         timeline_start_tc: str = "01:00:00:00", entities: List[dict] = None):
    """
    Save markers as EDL for DaVinci Resolve TIMELINE markers.
    
    These markers are placed at timeline positions, not on clips.
    Import into DaVinci: File > Import > Timeline (select EDL, check "Import markers")
    
    Marker colors:
    - Green/Cyan/Yellow/Orange/Red: Confidence tiers
    - Pink: Entity markers (TEXT OVERLAY needed)
    
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
    marker_num = 1
    entity_markers_added = 0
    
    for i, match_result in enumerate(matches):
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
        lines.append(f"{marker_num:03d}  001      V     C        {tc_start} {tc_end} {tc_start} {tc_end}  ")
        lines.append(f" |C:{color} |M:{marker_name} |D:1")
        lines.append("")  # Empty line between markers
        marker_num += 1
        
        # ─────────────────────────────────────────────────────────────────────
        # ENTITY MARKERS - Pink markers for text overlays
        # ─────────────────────────────────────────────────────────────────────
        segment_entities = getattr(vo_seg, 'entities', []) or []
        
        # Also check if entities passed from extracted_entities match this segment's text
        if entities and not segment_entities:
            # Find entities that appear in this segment's text
            for entity in entities:
                entity_text = entity.get('text', '')
                if entity_text and entity_text.lower() in vo_seg.text.lower():
                    segment_entities.append(entity)
        
        if segment_entities:
            # Create offset timecode for entity markers (1 frame after main marker)
            entity_tc = frames_to_tc(timeline_frames + 2, frame_rate)
            entity_tc_end = frames_to_tc(timeline_frames + 3, frame_rate)
            
            # Group and add entity markers
            type_icons = {
                'PERSON': 'NAME',
                'GPE': 'LOCATION', 
                'ORG': 'ORG',
                'DATE': 'DATE',
                'EVENT': 'EVENT',
                'NUMBER': 'NUMBER'
            }
            
            for entity in segment_entities[:3]:  # Max 3 entities per segment
                etype = entity.get('type', 'ENTITY')
                ename = entity.get('text', '')[:25]
                context = entity.get('context', '')[:20]
                
                if ename:
                    icon = type_icons.get(etype, 'TEXT')
                    
                    # Format: TEXT: TYPE - Name (context)
                    if context:
                        entity_marker_name = f"TEXT: {icon} - {ename} ({context})"
                    else:
                        entity_marker_name = f"TEXT: {icon} - {ename}"
                    
                    # Clean for EDL
                    entity_marker_name = entity_marker_name.replace('|', '-').replace('\n', ' ')
                    
                    lines.append(f"{marker_num:03d}  001      V     C        {entity_tc} {entity_tc_end} {entity_tc} {entity_tc_end}  ")
                    lines.append(f" |C:ResolveColorPink |M:{entity_marker_name} |D:1")
                    lines.append("")
                    marker_num += 1
                    entity_markers_added += 1
        
        # Move to next segment position on timeline
        timeline_frames += duration_frames
    
    # Write EDL file
    with open(edl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved EDL with {len(matches)} timeline markers + {entity_markers_added} entity markers to {edl_path}")
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

    for match_idx, match_result in enumerate(matches):
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

        # Include segment ID in clip name for post-edit analysis tracing
        segment_id = f"S{match_idx:03d}"
        clip_name = f"[{segment_id}] {Path(vid_seg.source_file).stem}"

        xml_lines.extend([
            '          <clipitem>',
            f'            <name>{clip_name}</name>',
            f'            <duration>{target_frames}</duration>',
            f'            <start>{timeline_pos}</start>',
            f'            <end>{timeline_pos + target_frames}</end>',
            f'            <in>{source_start_frames}</in>',
            f'            <out>{source_start_frames + source_frames}</out>',
            '            <file>',
            f'              <pathurl>{encode_path_for_xml_url(vid_seg.source_file)}</pathurl>',
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

        for match_idx, match_result in enumerate(matches):
            vo_seg = match_result.primary_match.voiceover_segment
            target_duration = vo_seg.end_time - vo_seg.start_time
            target_frames = int(target_duration * frame_rate)

            # Segment ID for tracing
            segment_id = f"S{match_idx:03d}"

            if alt_idx < len(match_result.alternatives):
                alt = match_result.alternatives[alt_idx]
                alt_seg = alt.video_segment

                source_duration = alt_seg.end_time - alt_seg.start_time
                source_start = alt_seg.start_time
                source_frames = int(source_duration * frame_rate)
                source_start_frames = int(source_start * frame_rate)
                speed = (source_duration / target_duration) * 100 if target_duration > 0 else 100

                clip_name = f"[{segment_id}] ALT{alt_idx+1}: {Path(alt_seg.source_file).stem}"

                xml_lines.extend([
                    '          <clipitem>',
                    f'            <name>{clip_name}</name>',
                    f'            <duration>{target_frames}</duration>',
                    f'            <start>{timeline_pos}</start>',
                    f'            <end>{timeline_pos + target_frames}</end>',
                    f'            <in>{source_start_frames}</in>',
                    f'            <out>{source_start_frames + source_frames}</out>',
                    '            <file>',
                    f'              <pathurl>{encode_path_for_xml_url(alt_seg.source_file)}</pathurl>',
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
            f'              <pathurl>{encode_path_for_xml_url(voiceover_path)}</pathurl>',
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




def generate_resolve_xml_with_bins(
    matches: List[MatchResult],
    output_path: str,
    voiceover_path: str = None,
    frame_rate: float = 30.0,
    entity_images: Dict = None,
    entity_videos: Dict = None,
    config = None,
    num_parts: int = 2
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
    import uuid as uuid_module
    
    base_path = Path(output_path).with_suffix('')
    fps_int = int(frame_rate)
    
    # Collect ALL unique files
    all_files = {}
    file_counter = 1
    
    def add_file(path: str, duration_seconds: float = 0) -> dict:
        nonlocal file_counter
        if path not in all_files:
            dur_frames = int(duration_seconds * frame_rate) if duration_seconds > 0 else int(60 * frame_rate)
            all_files[path] = {
                'file_id': f"file-{file_counter}",
                'uuid': str(uuid_module.uuid4()),
                'duration_frames': dur_frames
            }
            file_counter += 1
        return all_files[path]
    
    # Collect files from all tracks
    for match_result in matches:
        vid_seg = match_result.primary_match.video_segment
        dur = vid_seg.end_time if vid_seg.end_time > 0 else 60.0
        add_file(vid_seg.source_file, dur)
        
        for alt in match_result.alternatives:
            dur = alt.video_segment.end_time if alt.video_segment.end_time > 0 else 60.0
            add_file(alt.video_segment.source_file, dur)
        
        for sec in getattr(match_result, 'secondary_matches', []):
            dur = sec.video_segment.end_time if sec.video_segment.end_time > 0 else 60.0
            add_file(sec.video_segment.source_file, dur)
        
        for strat in match_result.strategy_matches:
            dur = strat.video_segment.end_time if strat.video_segment.end_time > 0 else 60.0
            add_file(strat.video_segment.source_file, dur)
    
    if entity_images:
        for entity, result in entity_images.items():
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
            m.primary_match.voiceover_segment.end_time - m.primary_match.voiceover_segment.start_time
            for m in matches
        )
        add_file(voiceover_path, vo_duration)
    
    # Calculate total timeline duration
    total_frames = 0
    for m in matches:
        vo_seg = m.primary_match.voiceover_segment
        target_duration = vo_seg.end_time - vo_seg.start_time
        total_frames += int(target_duration * frame_rate)
    
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
        
        xml_lines.extend([
            f'                    <clip id="masterclip-{file_info["file_id"]}">',
            f'                        <uuid>{file_info["uuid"]}</uuid>',
            f'                        <name>{unique_name}</name>',
            '                        <rate>',
            f'                            <timebase>{fps_int}</timebase>',
            '                            <ntsc>FALSE</ntsc>',
            '                        </rate>',
            f'                        <duration>{duration_frames}</duration>',
            '                        <media>',
        ])
        
        if is_video or is_image:
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
                f'                                        <file id="{file_info["file_id"]}">',
                f'                                            <name>{unique_name}</name>',
                f'                                            <pathurl>{path_url}</pathurl>',
                '                                            <rate>',
                f'                                                <timebase>{fps_int}</timebase>',
                '                                                <ntsc>FALSE</ntsc>',
                '                                            </rate>',
                f'                                            <duration>{duration_frames}</duration>',
                '                                            <timecode>',
                '                                                <rate>',
                f'                                                    <timebase>{fps_int}</timebase>',
                '                                                    <ntsc>FALSE</ntsc>',
                '                                                </rate>',
                '                                                <string>00:00:00:00</string>',
                '                                                <frame>0</frame>',
                '                                            </timecode>',
                '                                        </file>',
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
            ])
            if not (is_video or is_image):
                # Audio-only - need full file definition
                xml_lines.extend([
                    f'                                        <file id="{file_info["file_id"]}">',
                    f'                                            <name>{unique_name}</name>',
                    f'                                            <pathurl>{path_url}</pathurl>',
                    '                                            <rate>',
                    f'                                                <timebase>{fps_int}</timebase>',
                    '                                                <ntsc>FALSE</ntsc>',
                    '                                            </rate>',
                    f'                                            <duration>{duration_frames}</duration>',
                    '                                        </file>',
                ])
            else:
                xml_lines.append(f'                                        <file id="{file_info["file_id"]}"/>')
            xml_lines.extend([
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
        '                    <ntsc>FALSE</ntsc>',
        '                </rate>',
        '                <timecode>',
        '                    <rate>',
        f'                        <timebase>{fps_int}</timebase>',
        '                        <ntsc>FALSE</ntsc>',
        '                    </rate>',
        '                    <string>01:00:00:00</string>',
        f'                    <frame>{fps_int * 3600}</frame>',
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

        target_duration = vo_seg.end_time - vo_seg.start_time
        target_frames = int(target_duration * frame_rate)

        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time
        source_frames = int(source_duration * frame_rate)
        source_start_frames = int(source_start * frame_rate)

        file_info = all_files.get(vid_seg.source_file, {})
        file_id = file_info.get('file_id', '')

        # Make clip name unique by including segment ID and parent folder
        segment_id = f"S{match_idx:03d}"
        folder_name = Path(vid_seg.source_file).parent.name
        base_name = Path(vid_seg.source_file).stem
        unique_name = escape_xml(f"[{segment_id}] {folder_name}_{base_name}")

        xml_lines.extend([
            '                            <clipitem>',
            f'                                <name>{unique_name}</name>',
            f'                                <duration>{target_frames}</duration>',
            f'                                <start>{timeline_pos}</start>',
            f'                                <end>{timeline_pos + target_frames}</end>',
            f'                                <in>{source_start_frames}</in>',
            f'                                <out>{source_start_frames + source_frames}</out>',
            f'                                <file id="{file_id}"/>',
        ])
        
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
                logger
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
                bin_name_override=f"Media - {folder_name}"
            )
    
    return generated_paths


def _write_media_xml_part(
    files_subset: dict,
    part_idx: int,
    output_path: str,
    fps_int: int,
    generated_paths: list,
    logger,
    bin_name_override: str = None
):
    """Write a single media XML part file."""
    bin_name = bin_name_override or f"Media Part {part_idx}"
    
    part_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="4">',
        '    <bin>',
        f'        <n>{bin_name}</n>',
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
        audio_exts = {'.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg'}
        is_video = file_ext in video_exts
        is_audio = file_ext in audio_exts
        
        clip_num = file_info["file_id"].replace("file-", "")
        
        part_lines.extend([
            f'            <clip id="clip-{clip_num}">',
            f'                <uuid>{file_info["uuid"]}</uuid>',
            f'                <n>{unique_name}</n>',
            '                <rate>',
            f'                    <timebase>{fps_int}</timebase>',
            '                    <ntsc>FALSE</ntsc>',
            '                </rate>',
            '                <media>',
            '                    <video>',
            '                        <track>',
            f'                            <clipitem id="clipitem-{clip_num}">',
            f'                                <n>{unique_name}</n>',
            f'                                <file id="{file_info["file_id"]}">',
            f'                                    <n>{unique_name}</n>',
            f'                                    <pathurl>{path_url}</pathurl>',
            '                                    <rate>',
            f'                                        <timebase>{fps_int}</timebase>',
            '                                        <ntsc>FALSE</ntsc>',
            '                                    </rate>',
            f'                                    <duration>{duration_frames}</duration>',
            '                                    <timecode>',
            '                                        <rate>',
            f'                                            <timebase>{fps_int}</timebase>',
            '                                            <ntsc>FALSE</ntsc>',
            '                                        </rate>',
            '                                        <string>00:00:00:00</string>',
            '                                        <frame>0</frame>',
            '                                    </timecode>',
            '                                </file>',
            '                            </clipitem>',
            '                        </track>',
            '                    </video>',
        ])
        
        # Add audio section for video and audio files
        if is_video or is_audio:
            part_lines.extend([
                '                    <audio>',
                '                        <track>',
                f'                            <clipitem id="clipitem-{clip_num}-audio">',
                f'                                <n>{unique_name}</n>',
                f'                                <file id="{file_info["file_id"]}"/>',
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
        f'        <n>{bin_name} - Import Helper</n>',
        '        <rate>',
        f'            <timebase>{fps_int}</timebase>',
        '            <ntsc>FALSE</ntsc>',
        '        </rate>',
        '        <duration>1</duration>',
        '        <timecode>',
        '            <rate>',
        f'                <timebase>{fps_int}</timebase>',
        '                <ntsc>FALSE</ntsc>',
        '            </rate>',
        '            <string>01:00:00:00</string>',
        '            <frame>108000</frame>',
        '            <displayformat>NDF</displayformat>',
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
