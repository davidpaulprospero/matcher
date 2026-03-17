#!/usr/bin/env python3
"""
Final Edit Analyzer v1.0

Analyzes exported OTIO from DaVinci Resolve to learn user's clip selections.
Tracks which clips were chosen for each SRT segment and caches globally
for improving future matching.

Usage:
    python analyze_final_edit.py <exported.otio>
    
Or drag-drop OTIO onto final_selections.bat
"""

import argparse
import json
import sys
import re
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

# Try to import opentimelineio
try:
    import opentimelineio as otio
except ImportError:
    print("ERROR: opentimelineio not installed. Run: pip install opentimelineio")
    sys.exit(1)

from script_utils import print_ok, print_warn, print_error, print_info, print_header, set_verbosity


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ClipSelection:
    """Represents a user's clip selection for a segment"""
    segment_index: int
    segment_start_time: float
    segment_end_time: float
    segment_text: str
    
    selected_video: str  # Filename
    selected_video_id: str  # YouTube ID extracted from filename
    selected_track: str  # e.g., "V1", "V3"
    selected_track_num: int  # e.g., 1, 3
    
    source_in: float  # Source timecode in
    source_out: float  # Source timecode out
    
    original_confidence: float  # From clip metadata if available
    
    # Alternatives that were available but not selected
    alternatives: List[Dict] = field(default_factory=list)


@dataclass 
class SelectionReport:
    """Complete report of selections from an edit"""
    project_name: str
    otio_path: str
    analysis_date: str
    
    total_segments: int
    total_selections: int
    
    selections: List[ClipSelection] = field(default_factory=list)
    
    # Aggregated stats
    videos_used: Dict[str, int] = field(default_factory=dict)  # video_id -> count
    tracks_used: Dict[str, int] = field(default_factory=dict)  # track -> count
    avg_confidence: float = 0.0


# =============================================================================
# YOUTUBE ID EXTRACTION
# =============================================================================

def extract_youtube_id(filename: str) -> str:
    """
    Extract YouTube video ID from filename.
    
    Common patterns:
    - Video_Title-xYz12345678.mp4
    - Video_Title_xYz12345678.mp4
    - xYz12345678.mp4
    """
    # Remove extension
    name = Path(filename).stem
    
    # YouTube IDs are 11 characters: [a-zA-Z0-9_-]
    # Pattern 1: ID at end after dash
    match = re.search(r'-([a-zA-Z0-9_-]{11})$', name)
    if match:
        return match.group(1)
    
    # Pattern 2: ID at end after underscore
    match = re.search(r'_([a-zA-Z0-9_-]{11})$', name)
    if match:
        return match.group(1)
    
    # Pattern 3: Just the ID (11 chars)
    if len(name) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', name):
        return name
    
    # Pattern 4: ID somewhere in the name
    match = re.search(r'[_-]([a-zA-Z0-9_-]{11})[_-]?', name)
    if match:
        return match.group(1)
    
    # Fallback: use whole filename as ID
    return name


# =============================================================================
# SRT PARSING
# =============================================================================

def parse_srt(srt_path: str) -> List[Dict]:
    """Parse SRT file to get segment timings and text"""
    segments = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        try:
            with open(srt_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except:
            return []
    
    # Split by double newline (segment separator)
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Line 1: index
            # Line 2: timecode
            # Line 3+: text
            try:
                index = int(lines[0])
                timecode = lines[1]
                text = ' '.join(lines[2:])
                
                # Parse timecode: 00:00:01,000 --> 00:00:05,000
                match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
                    timecode
                )
                if match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
                    start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
                    end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000
                    
                    segments.append({
                        'index': index,
                        'start_time': start,
                        'end_time': end,
                        'text': text.strip()
                    })
            except (ValueError, IndexError):
                continue
    
    return segments


def find_srt_file(otio_path: str) -> Optional[str]:
    """Try to find the SRT file associated with this project"""
    otio_dir = Path(otio_path).parent
    
    # Look for SRT files in the project directory and parent
    search_dirs = [otio_dir, otio_dir.parent]
    
    for search_dir in search_dirs:
        # Common SRT names
        for pattern in ['*.srt', 'voiceover*.srt', 'transcript*.srt']:
            srt_files = list(search_dir.glob(pattern))
            if srt_files:
                # Prefer voiceover.srt if multiple
                for srt in srt_files:
                    if 'voiceover' in srt.name.lower():
                        return str(srt)
                return str(srt_files[0])
    
    return None


# =============================================================================
# OTIO ANALYSIS
# =============================================================================

def analyze_otio(otio_path: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Analyze OTIO file to extract clip placements.
    
    Returns:
        - List of clips with their track, timing, and metadata
        - Timeline metadata
    """
    timeline = otio.adapters.read_from_file(otio_path)
    
    clips_by_position = defaultdict(list)  # start_time -> [clips]
    timeline_meta = {
        'name': timeline.name,
        'duration': 0,
        'tracks': []
    }
    
    # Get frame rate
    if hasattr(timeline, 'global_start_time') and timeline.global_start_time:
        frame_rate = timeline.global_start_time.rate
    else:
        frame_rate = 30.0  # Default
    
    # Process all video tracks
    for track in timeline.tracks:
        if track.kind != otio.schema.TrackKind.Video:
            continue
        
        track_name = track.name or "Unknown"
        track_enabled = track.enabled if hasattr(track, 'enabled') else True
        
        # Extract track number from name (e.g., "V1 - Primary" -> 1)
        track_num_match = re.search(r'V(\d+)', track_name)
        track_num = int(track_num_match.group(1)) if track_num_match else 0
        
        timeline_meta['tracks'].append({
            'name': track_name,
            'number': track_num,
            'enabled': track_enabled
        })
        
        # Track position as we iterate through clips
        current_position = 0.0
        
        for item in track:
            if isinstance(item, otio.schema.Gap):
                # Skip gaps but track position
                if item.source_range:
                    gap_duration = item.source_range.duration.to_seconds()
                    current_position += gap_duration
                continue
            
            if isinstance(item, otio.schema.Clip):
                clip_enabled = item.enabled if hasattr(item, 'enabled') else True
                
                # Get clip timing
                if item.source_range:
                    clip_start = current_position
                    clip_duration = item.source_range.duration.to_seconds()
                    source_in = item.source_range.start_time.to_seconds()
                    source_out = source_in + clip_duration
                else:
                    clip_start = current_position
                    clip_duration = 0
                    source_in = 0
                    source_out = 0
                
                # Get media path
                media_path = ""
                if item.media_reference:
                    if hasattr(item.media_reference, 'target_url'):
                        media_path = item.media_reference.target_url
                
                # Get metadata
                confidence = 0.0
                if 'confidence' in item.metadata:
                    confidence = float(item.metadata.get('confidence', 0))
                
                clip_info = {
                    'name': item.name,
                    'track_name': track_name,
                    'track_num': track_num,
                    'track_enabled': track_enabled,
                    'clip_enabled': clip_enabled,
                    'timeline_start': clip_start,
                    'timeline_end': clip_start + clip_duration,
                    'duration': clip_duration,
                    'source_in': source_in,
                    'source_out': source_out,
                    'media_path': media_path,
                    'confidence': confidence,
                    'metadata': dict(item.metadata) if item.metadata else {}
                }
                
                # Group by timeline start position (rounded to handle float precision)
                position_key = round(clip_start, 2)
                clips_by_position[position_key].append(clip_info)
                
                current_position += clip_duration
    
    # Convert to list sorted by position
    all_clips = []
    for pos in sorted(clips_by_position.keys()):
        all_clips.append({
            'position': pos,
            'clips': clips_by_position[pos]
        })
    
    return all_clips, timeline_meta


def get_selected_clip(clips: List[Dict]) -> Optional[Dict]:
    """
    From a list of clips at the same position, find the "selected" one.
    
    Selection rule: Highest track number among enabled tracks/clips.
    """
    # Filter to enabled clips on enabled tracks
    enabled_clips = [
        c for c in clips 
        if c.get('track_enabled', True) and c.get('clip_enabled', True)
    ]
    
    if not enabled_clips:
        # Fallback to any clip if none enabled
        enabled_clips = clips
    
    if not enabled_clips:
        return None
    
    # Sort by track number (highest first)
    enabled_clips.sort(key=lambda c: c.get('track_num', 0), reverse=True)
    
    return enabled_clips[0]


# =============================================================================
# SEGMENT MATCHING
# =============================================================================

def match_clips_to_segments(
    clip_positions: List[Dict],
    srt_segments: List[Dict],
    markers: List[Dict] = None
) -> List[ClipSelection]:
    """
    Match clips to SRT segments by start time.
    
    Tries multiple methods:
    1. Exact start time match (within tolerance)
    2. Marker metadata if available
    3. Sequence order as fallback
    """
    selections = []
    tolerance = 0.1  # 100ms tolerance for time matching
    
    # Build segment lookup by start time
    segment_by_time = {}
    for seg in srt_segments:
        key = round(seg['start_time'], 1)
        segment_by_time[key] = seg
    
    # Also build by index
    segment_by_index = {seg['index']: seg for seg in srt_segments}
    
    for pos_data in clip_positions:
        position = pos_data['position']
        clips = pos_data['clips']
        
        # Get selected clip
        selected = get_selected_clip(clips)
        if not selected:
            continue
        
        # Try to match to segment
        matched_segment = None
        
        # Method 1: Exact time match
        for time_key in segment_by_time:
            if abs(time_key - position) <= tolerance:
                matched_segment = segment_by_time[time_key]
                break
        
        # Method 2: Check clip metadata for segment info
        if not matched_segment and selected.get('metadata'):
            meta = selected['metadata']
            if 'segment_num' in meta:
                seg_idx = meta['segment_num']
                if seg_idx in segment_by_index:
                    matched_segment = segment_by_index[seg_idx]
            elif 'voiceover_text' in meta:
                # Match by text
                vo_text = meta['voiceover_text'][:50]
                for seg in srt_segments:
                    if seg['text'][:50] == vo_text:
                        matched_segment = seg
                        break
        
        # Method 3: Sequence order (fallback)
        if not matched_segment and len(selections) < len(srt_segments):
            matched_segment = srt_segments[len(selections)]
        
        if not matched_segment:
            continue
        
        # Extract video ID
        filename = Path(selected.get('media_path', selected.get('name', ''))).name
        video_id = extract_youtube_id(filename)
        
        # Build alternatives list
        alternatives = []
        for clip in clips:
            if clip != selected:
                alternatives.append({
                    'video': Path(clip.get('media_path', clip.get('name', ''))).name,
                    'video_id': extract_youtube_id(Path(clip.get('media_path', clip.get('name', ''))).name),
                    'track': clip.get('track_name', ''),
                    'confidence': clip.get('confidence', 0)
                })
        
        selection = ClipSelection(
            segment_index=matched_segment.get('index', len(selections) + 1),
            segment_start_time=matched_segment.get('start_time', position),
            segment_end_time=matched_segment.get('end_time', position + selected.get('duration', 0)),
            segment_text=matched_segment.get('text', ''),
            selected_video=filename,
            selected_video_id=video_id,
            selected_track=selected.get('track_name', ''),
            selected_track_num=selected.get('track_num', 0),
            source_in=selected.get('source_in', 0),
            source_out=selected.get('source_out', 0),
            original_confidence=selected.get('confidence', 0),
            alternatives=alternatives
        )
        
        selections.append(selection)
    
    return selections


# =============================================================================
# GLOBAL CACHE
# =============================================================================

def get_global_cache_path() -> Path:
    """Get path to global cache file"""
    # Look for .global_cache in installation directory
    script_dir = Path(__file__).parent
    cache_path = script_dir / '.global_cache' / 'selection_history.json'
    
    # Create directory if needed
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    return cache_path


def load_global_cache() -> Dict:
    """Load global selection cache"""
    cache_path = get_global_cache_path()
    
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    # Default structure
    return {
        'version': '1.0',
        'last_updated': None,
        'videos': {},  # video_id -> selection stats
        'topic_preferences': {},  # topic -> preferred videos
        'keyword_video_map': {},  # keyword -> [video_ids]
        'selection_history': []  # Recent selections for analysis
    }


def save_global_cache(cache: Dict):
    """Save global selection cache"""
    cache_path = get_global_cache_path()
    cache['last_updated'] = datetime.now().isoformat()
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)
    
    print_ok(f"Cache updated: {cache_path}")


def update_global_cache(cache: Dict, report: SelectionReport, topic: str = None, keywords: List[str] = None):
    """
    Update global cache with new selections.
    
    Tracks:
    - Video selection counts and history
    - Topic preferences
    - Keyword-video associations
    """
    # Extract topic/keywords from project name if not provided
    if not topic:
        topic = report.project_name.split('_')[0] if '_' in report.project_name else report.project_name
    
    if not keywords:
        keywords = []
    
    for selection in report.selections:
        video_id = selection.selected_video_id
        
        # Update video stats
        if video_id not in cache['videos']:
            cache['videos'][video_id] = {
                'filename': selection.selected_video,
                'selected_count': 0,
                'rejected_count': 0,
                'avg_confidence': 0.0,
                'topics': [],
                'keywords': [],
                'first_selected': None,
                'last_selected': None,
                'selection_history': []
            }
        
        video_entry = cache['videos'][video_id]
        video_entry['selected_count'] += 1
        video_entry['last_selected'] = report.analysis_date
        if not video_entry['first_selected']:
            video_entry['first_selected'] = report.analysis_date
        
        # Update running average confidence
        old_count = video_entry['selected_count'] - 1
        old_avg = video_entry['avg_confidence']
        new_conf = selection.original_confidence
        video_entry['avg_confidence'] = (old_avg * old_count + new_conf) / video_entry['selected_count']
        
        # Add topic if not present
        if topic and topic not in video_entry['topics']:
            video_entry['topics'].append(topic)
        
        # Add keywords
        for kw in keywords:
            if kw and kw not in video_entry['keywords']:
                video_entry['keywords'].append(kw)
        
        # Add to selection history (keep last 20)
        video_entry['selection_history'].append({
            'project': report.project_name,
            'date': report.analysis_date,
            'segment_text': selection.segment_text[:100],
            'track': selection.selected_track,
            'confidence': selection.original_confidence
        })
        video_entry['selection_history'] = video_entry['selection_history'][-20:]
        
        # Track rejected alternatives
        for alt in selection.alternatives:
            alt_id = alt.get('video_id', '')
            if alt_id and alt_id in cache['videos']:
                cache['videos'][alt_id]['rejected_count'] += 1
        
        # Update topic preferences
        if topic:
            if topic not in cache['topic_preferences']:
                cache['topic_preferences'][topic] = {
                    'preferred_videos': [],
                    'video_scores': {}
                }
            
            topic_prefs = cache['topic_preferences'][topic]
            if video_id not in topic_prefs['video_scores']:
                topic_prefs['video_scores'][video_id] = 0
            topic_prefs['video_scores'][video_id] += 1
            
            # Update preferred list (sorted by score)
            scores = topic_prefs['video_scores']
            topic_prefs['preferred_videos'] = sorted(
                scores.keys(), 
                key=lambda x: scores[x], 
                reverse=True
            )[:50]  # Keep top 50
        
        # Update keyword-video map
        for kw in keywords:
            if kw:
                if kw not in cache['keyword_video_map']:
                    cache['keyword_video_map'][kw] = []
                if video_id not in cache['keyword_video_map'][kw]:
                    cache['keyword_video_map'][kw].append(video_id)
    
    # Add to global selection history
    cache['selection_history'].append({
        'project': report.project_name,
        'date': report.analysis_date,
        'total_selections': report.total_selections,
        'avg_confidence': report.avg_confidence,
        'top_videos': list(report.videos_used.keys())[:5]
    })
    cache['selection_history'] = cache['selection_history'][-100:]  # Keep last 100


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_markdown_report(report: SelectionReport, output_path: Path):
    """Generate detailed markdown report"""
    lines = [
        f"# Final Edit Selection Report",
        f"",
        f"**Project:** {report.project_name}",
        f"**Date:** {report.analysis_date}",
        f"**OTIO:** `{Path(report.otio_path).name}`",
        f"",
        f"## Summary",
        f"",
        f"- **Total Segments:** {report.total_segments}",
        f"- **Total Selections:** {report.total_selections}",
        f"- **Average Confidence:** {report.avg_confidence:.1%}",
        f"",
        f"### Videos Used",
        f"",
    ]
    
    # Sort videos by usage count
    sorted_videos = sorted(report.videos_used.items(), key=lambda x: x[1], reverse=True)
    for video_id, count in sorted_videos[:20]:
        lines.append(f"- `{video_id}`: {count} segments")
    
    lines.extend([
        f"",
        f"### Tracks Used",
        f"",
    ])
    
    for track, count in sorted(report.tracks_used.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {track}: {count} selections")
    
    lines.extend([
        f"",
        f"## Selections Detail",
        f"",
    ])
    
    for sel in report.selections[:50]:  # Limit to first 50 for readability
        lines.extend([
            f"### Segment {sel.segment_index}",
            f"",
            f"**Text:** {sel.segment_text[:150]}{'...' if len(sel.segment_text) > 150 else ''}",
            f"",
            f"- **Selected:** `{sel.selected_video}` (ID: `{sel.selected_video_id}`)",
            f"- **Track:** {sel.selected_track}",
            f"- **Source:** {sel.source_in:.2f}s - {sel.source_out:.2f}s",
            f"- **Confidence:** {sel.original_confidence:.1%}",
            f"",
        ])
        
        if sel.alternatives:
            lines.append(f"**Alternatives not selected:**")
            for alt in sel.alternatives[:5]:
                lines.append(f"- `{alt['video']}` on {alt['track']} ({alt['confidence']:.1%})")
            lines.append("")
    
    if len(report.selections) > 50:
        lines.append(f"*... and {len(report.selections) - 50} more selections*")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_plaintext_report(report: SelectionReport, output_path: Path):
    """Generate minimal plaintext report for LLMs"""
    lines = [
        f"PROJECT: {report.project_name}",
        f"DATE: {report.analysis_date}",
        f"SEGMENTS: {report.total_segments}",
        f"SELECTIONS: {report.total_selections}",
        f"AVG_CONF: {report.avg_confidence:.0%}",
        f"",
        f"VIDEOS_USED:",
    ]
    
    for video_id, count in sorted(report.videos_used.items(), key=lambda x: x[1], reverse=True)[:20]:
        lines.append(f"  {video_id}={count}")
    
    lines.extend([
        f"",
        f"TRACKS:",
    ])
    
    for track, count in sorted(report.tracks_used.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {track}={count}")
    
    lines.extend([
        f"",
        f"SELECTIONS:",
    ])
    
    for sel in report.selections:
        alt_count = len(sel.alternatives)
        lines.append(
            f"  seg={sel.segment_index} vid={sel.selected_video_id} "
            f"track={sel.selected_track_num} conf={sel.original_confidence:.0%} "
            f"alts={alt_count}"
        )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyzes exported OTIO from DaVinci Resolve to learn user clip selections.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python analyze_final_edit.py export.otio
    python analyze_final_edit.py "E:/Edit Job/project/export.otio"
'''
    )
    parser.add_argument('otio_file', help='Path to exported OTIO file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    otio_path = args.otio_file

    if not os.path.exists(otio_path):
        print_error(f"File not found: {otio_path}", exit_code=1)

    print_header("FINAL EDIT ANALYZER")
    print_info(f"OTIO: {Path(otio_path).name}")

    # Analyze OTIO
    print_info("Analyzing OTIO structure...")
    try:
        clip_positions, timeline_meta = analyze_otio(otio_path)
        print_ok(f"Found {len(clip_positions)} clip positions")
        print_ok(f"Found {len(timeline_meta['tracks'])} video tracks")
    except Exception as e:
        print_error(f"Failed to analyze OTIO: {e}", exit_code=1)

    # Find and parse SRT
    print_info("Looking for SRT file...")
    srt_path = find_srt_file(otio_path)
    srt_segments = []

    if srt_path:
        print_ok(f"Found: {Path(srt_path).name}")
        srt_segments = parse_srt(srt_path)
        print_ok(f"Parsed {len(srt_segments)} segments")
    else:
        print_warn("No SRT found, using clip metadata for segment info")

    # Match clips to segments
    print_info("Matching clips to segments...")
    selections = match_clips_to_segments(clip_positions, srt_segments)
    print_ok(f"Matched {len(selections)} selections")
    
    # Build report
    project_name = Path(otio_path).stem
    # Try to get project name from parent folder
    parent_name = Path(otio_path).parent.name
    if parent_name and parent_name != project_name:
        project_name = parent_name
    
    report = SelectionReport(
        project_name=project_name,
        otio_path=otio_path,
        analysis_date=datetime.now().isoformat(),
        total_segments=len(srt_segments) if srt_segments else len(selections),
        total_selections=len(selections),
        selections=selections
    )
    
    # Aggregate stats
    for sel in selections:
        # Count video usage
        if sel.selected_video_id not in report.videos_used:
            report.videos_used[sel.selected_video_id] = 0
        report.videos_used[sel.selected_video_id] += 1
        
        # Count track usage
        if sel.selected_track not in report.tracks_used:
            report.tracks_used[sel.selected_track] = 0
        report.tracks_used[sel.selected_track] += 1
    
    # Calculate average confidence
    if selections:
        confidences = [s.original_confidence for s in selections if s.original_confidence > 0]
        if confidences:
            report.avg_confidence = sum(confidences) / len(confidences)
    
    # Update global cache
    print_info("Updating global cache...")
    cache = load_global_cache()
    
    # Try to extract topic/keywords from project metadata or name
    topic = None
    keywords = []
    
    # Check for config.json or similar in project folder
    project_dir = Path(otio_path).parent
    config_files = list(project_dir.glob('config*.json')) + list(project_dir.glob('config*.yaml'))
    if config_files:
        try:
            import yaml
# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

            with open(config_files[0], 'r') as f:
                if config_files[0].suffix == '.yaml':
                    proj_config = yaml.safe_load(f)
                else:
                    proj_config = json.load(f)
            topic = proj_config.get('topic', proj_config.get('topic_context', None))
            keywords = proj_config.get('keywords', [])
        except:
            pass
    
    update_global_cache(cache, report, topic=topic, keywords=keywords)
    save_global_cache(cache)
    
    # Generate reports
    print_info("Generating reports...")

    # Find or create logs directory
    logs_dir = project_dir / 'logs'
    if not logs_dir.exists():
        logs_dir = project_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    md_path = logs_dir / f'selection_report_{timestamp}.md'
    txt_path = logs_dir / f'selection_report_{timestamp}.txt'

    generate_markdown_report(report, md_path)
    print_ok(f"Markdown: {md_path.name}")

    generate_plaintext_report(report, txt_path)
    print_ok(f"Plaintext: {txt_path.name}")

    # Print summary
    print_header("SUMMARY")
    print_info(f"Segments: {report.total_segments}")
    print_info(f"Selections: {report.total_selections}")
    print_info(f"Avg Confidence: {report.avg_confidence:.1%}")
    print_info(f"Unique Videos: {len(report.videos_used)}")
    print_info("Top Videos:")
    for video_id, count in sorted(report.videos_used.items(), key=lambda x: x[1], reverse=True)[:5]:
        print_info(f"  {video_id}: {count} uses")
    print_info("Track Distribution:")
    for track, count in sorted(report.tracks_used.items(), key=lambda x: x[1], reverse=True):
        pct = count / report.total_selections * 100 if report.total_selections > 0 else 0
        print_info(f"  {track}: {count} ({pct:.0f}%)")
    print_info(f"Cache: {get_global_cache_path()}")
    print_info(f"Reports: {logs_dir}")


if __name__ == '__main__':
    main()
