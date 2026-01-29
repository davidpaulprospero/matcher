"""
Timeline statistics and reporting functions.

Migrated from otio_builder.py - provides segment mapping and statistics.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List

import opentimelineio as otio

from .utils import NumpyEncoder

if TYPE_CHECKING:
    from ..utils import MatchResult

logger = logging.getLogger(__name__)


def _calculate_track_coverage(
    matches: List['MatchResult'],
    frame_rate: float
) -> dict:
    """
    Calculate track coverage statistics from match results.

    Args:
        matches: List of MatchResult from matching stage
        frame_rate: Timeline frame rate for duration calculations

    Returns:
        Dict mapping track names (V1-V10) to coverage statistics
    """
    total_segments = len(matches)
    if total_segments == 0:
        return {}

    # Calculate total timeline duration
    total_duration = 0.0
    for match_result in matches:
        vo_seg = match_result.primary_match.voiceover_segment
        total_duration += vo_seg.end - vo_seg.start

    # Initialize track statistics
    track_stats = {}
    track_names = [
        ("V1", "Primary"),
        ("V2", "Alternative 1"),
        ("V3", "Alternative 2"),
        ("V4", "Secondary 1"),
        ("V5", "Secondary 2"),
        ("V6", "Secondary 3"),
        ("V7", "Strategy: Embedding-Diversity"),
        ("V8", "Strategy: B-roll Only"),
        ("V9", "Entity Images"),
        ("V10", "Stock Videos"),
    ]

    for track_id, track_desc in track_names:
        track_stats[track_id] = {
            "description": track_desc,
            "clip_count": 0,
            "gap_count": 0,
            "clip_duration_sec": 0.0,
            "gap_duration_sec": 0.0,
            "coverage_percent": 0.0
        }

    # Count clips and gaps per track
    for match_idx, match_result in enumerate(matches):
        vo_seg = match_result.primary_match.voiceover_segment
        segment_duration = vo_seg.end - vo_seg.start

        # V1 - Primary track
        if match_result.has_gap:
            track_stats["V1"]["gap_count"] += 1
            track_stats["V1"]["gap_duration_sec"] += segment_duration
        else:
            track_stats["V1"]["clip_count"] += 1
            track_stats["V1"]["clip_duration_sec"] += segment_duration

        # V2-V3 - Alternatives
        for alt_idx, alt in enumerate(match_result.alternatives[:2]):
            track_id = f"V{alt_idx + 2}"
            track_stats[track_id]["clip_count"] += 1
            track_stats[track_id]["clip_duration_sec"] += segment_duration

        # Count gaps for unfilled V2-V3 slots
        for gap_idx in range(len(match_result.alternatives[:2]), 2):
            track_id = f"V{gap_idx + 2}"
            track_stats[track_id]["gap_count"] += 1
            track_stats[track_id]["gap_duration_sec"] += segment_duration

        # V4-V6 - Secondary matches
        for sec_idx, sec in enumerate(match_result.secondary_matches[:3]):
            track_id = f"V{sec_idx + 4}"
            track_stats[track_id]["clip_count"] += 1
            track_stats[track_id]["clip_duration_sec"] += segment_duration

        # Count gaps for unfilled V4-V6 slots
        for gap_idx in range(len(match_result.secondary_matches[:3]), 3):
            track_id = f"V{gap_idx + 4}"
            track_stats[track_id]["gap_count"] += 1
            track_stats[track_id]["gap_duration_sec"] += segment_duration

        # V7-V8 - Strategy matches (check by strategy name)
        v7_filled = False
        v8_filled = False
        for strategy_match in match_result.strategy_matches:
            strategy = getattr(strategy_match, 'strategy', '').lower()
            if 'diversity' in strategy or 'embedding' in strategy:
                if not v7_filled:
                    track_stats["V7"]["clip_count"] += 1
                    track_stats["V7"]["clip_duration_sec"] += segment_duration
                    v7_filled = True
            elif 'broll' in strategy or 'b-roll' in strategy or 'silent' in strategy:
                if not v8_filled:
                    track_stats["V8"]["clip_count"] += 1
                    track_stats["V8"]["clip_duration_sec"] += segment_duration
                    v8_filled = True

        # Count gaps for unfilled V7-V8
        if not v7_filled:
            track_stats["V7"]["gap_count"] += 1
            track_stats["V7"]["gap_duration_sec"] += segment_duration
        if not v8_filled:
            track_stats["V8"]["gap_count"] += 1
            track_stats["V8"]["gap_duration_sec"] += segment_duration

        # V9-V10 are typically filled by entity matching stage, not in MatchResult
        # They'll show as gaps unless explicitly tracked
        track_stats["V9"]["gap_count"] += 1
        track_stats["V9"]["gap_duration_sec"] += segment_duration
        track_stats["V10"]["gap_count"] += 1
        track_stats["V10"]["gap_duration_sec"] += segment_duration

    # Calculate coverage percentages
    for track_id, stats in track_stats.items():
        if total_duration > 0:
            stats["coverage_percent"] = round(
                (stats["clip_duration_sec"] / total_duration) * 100, 1
            )
        stats["clip_duration_sec"] = round(stats["clip_duration_sec"], 3)
        stats["gap_duration_sec"] = round(stats["gap_duration_sec"], 3)

    return track_stats


def generate_segment_map(
    matches: List['MatchResult'],
    output_path: str,
    frame_rate: float = 30.0,
    source_srt: str = "",
    timeline_start_tc: str = "01:00:00:00"
) -> str:
    """
    Generate a segment map JSON file for post-edit analysis.

    This file maps timeline positions to segment IDs, enabling accurate
    comparison between the original generated timeline and an edited export
    from DaVinci Resolve (which strips custom metadata).

    Args:
        matches: List of MatchResult from matching stage
        output_path: Base output path (will append _segments.json)
        frame_rate: Timeline frame rate
        source_srt: Path to source SRT file (for reference)
        timeline_start_tc: Timeline start timecode (default 01:00:00:00)

    Returns:
        Path to the generated segment map JSON file
    """
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

    segments = []

    for match_idx, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        vid_seg = match.video_segment

        # Calculate segment position using ABSOLUTE voiceover timestamps
        # This prevents drift from accumulating rounding errors
        start_frame = round(vo_seg.start * frame_rate)
        end_frame = round(vo_seg.end * frame_rate)
        target_duration = vo_seg.end - vo_seg.start

        # Extract clip filename
        clip_file = Path(vid_seg.source_file).name

        # Build segment entry
        segment_entry = {
            "id": f"S{match_idx:03d}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_tc": frames_to_tc(start_frame),
            "end_tc": frames_to_tc(end_frame),
            "voiceover_text": vo_seg.text,
            "duration_sec": round(target_duration, 3),
            "v1_clip": {
                "file": clip_file,
                "confidence": round(match.confidence, 3),
                "source_start": round(vid_seg.start_time, 3),
                "source_end": round(vid_seg.end_time, 3)
            }
        }

        # Add alternatives (V2-V3)
        if match_result.alternatives:
            segment_entry["alternatives"] = []
            for alt_idx, alt in enumerate(match_result.alternatives):
                alt_file = Path(alt.video_segment.source_file).name
                segment_entry["alternatives"].append({
                    "track": f"V{alt_idx + 2}",
                    "file": alt_file,
                    "confidence": round(alt.confidence, 3)
                })

        # Add secondary matches (V4-V6)
        if match_result.secondary_matches:
            segment_entry["secondary"] = []
            for sec_idx, sec in enumerate(match_result.secondary_matches):
                sec_file = Path(sec.video_segment.source_file).name
                segment_entry["secondary"].append({
                    "track": f"V{sec_idx + 4}",
                    "file": sec_file,
                    "confidence": round(sec.confidence, 3)
                })

        segments.append(segment_entry)

    # Calculate total frames from the last segment's end frame
    total_frames = segments[-1]["end_frame"] if segments else 0

    # Calculate track coverage statistics
    track_coverage = _calculate_track_coverage(matches, frame_rate)

    # Build output structure
    segment_map = {
        "generated_at": datetime.now().isoformat(),
        "source_srt": source_srt,
        "frame_rate": frame_rate,
        "timeline_start_tc": timeline_start_tc,
        "total_segments": len(segments),
        "total_frames": total_frames,
        "total_duration_sec": round(total_frames / frame_rate, 3),
        "track_coverage": track_coverage,
        "segments": segments
    }

    # Write to file
    base_path = Path(output_path).with_suffix('')
    json_path = f"{base_path}_segments.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(segment_map, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"Generated segment map: {json_path} ({len(segments)} segments)")

    return json_path


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

    print("\n" + "=" * 60)
