#!/usr/bin/env python3
"""
fix-otio-timings skill

Repairs OTIO timeline clip timings that are out of sync with their reference voiceover SRT.

Usage:
    python skill.py "E:\Edit Job\Marcos\1\output\20260521_022003\timeline_V1_V1___Primary.otio"
    python skill.py "timeline.otio" --srt "voiceover_trimmed.srt" --output "fixed.otio"
"""

import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import opentimelineio as otio

logging.basicConfig(level=logging.INFO, format='[fix-otio-timings] %(message)s')
logger = logging.getLogger(__name__)

DRIFT_ERROR = 0.5  # seconds — anything beyond this is an error
DRIFT_WARN = 0.033  # seconds — sub-frame at 30fps


def parse_srt(srt_path: Path) -> List[dict]:
    """Parse SRT file, return list of {start, end, text} in seconds."""
    content = srt_path.read_text(encoding='utf-8-sig')
    blocks = re.split(r'\n\n+', content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        ts_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[1]
        )
        if not ts_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, ts_match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = ' '.join(lines[2:]).strip()
        segments.append({'start': start, 'end': end, 'text': text})
    return segments


def extract_segment_index(clip: otio.schema.Clip) -> Optional[int]:
    """Extract segment index from clip name or metadata.

    Looks for [S###] pattern in clip.name, e.g. clip.name = "Title [S002]"
    Returns None if no segment index found.
    """
    if clip.name:
        match = re.search(r'\[S(\d+)\]', clip.name)
        if match:
            return int(match.group(1))
    # Also check metadata if available
    if hasattr(clip, 'metadata') and clip.metadata:
        seg = clip.metadata.get('segment_index') or clip.metadata.get('segment')
        if seg is not None:
            try:
                return int(seg)
            except (ValueError, TypeError):
                pass
    return None


def get_track_clips(track: otio.schema.Track) -> List[otio.schema.Clip]:
    """Extract all clips from a track."""
    return [c for c in track if isinstance(c, otio.schema.Clip)]


def build_segment_index_map(clips: List[otio.schema.Clip]) -> dict:
    """Build a map: segment_index -> clip for all clips that have a segment index.

    Only includes clips with a valid [S###] pattern in their name.
    Clips without a segment index are ignored (they won't have SRT matches).
    """
    seg_map = {}
    for clip in clips:
        seg_idx = extract_segment_index(clip)
        if seg_idx is not None:
            seg_map[seg_idx] = clip
    return seg_map


def compute_clip_timeline_positions(clips: List[otio.schema.Clip]) -> dict:
    """Return dict of segment_index -> (timeline_position_sec, clip_duration_sec).

    Only includes clips that have a segment index.
    """
    positions = {}
    pos = 0.0
    fps = clips[0].source_range.duration.rate if clips else 30.0
    for clip in clips:
        seg_idx = extract_segment_index(clip)
        if seg_idx is not None:
            dur = clip.source_range.duration.value / fps
            positions[seg_idx] = (pos, dur)
        # Advance position regardless - gaps are valid
        if clip.source_range.duration.value > 0:
            pos += clip.source_range.duration.value / fps
    return positions


def check_track_alignment(
    track: otio.schema.Track,
    srt_segments: List[dict],
    fps: float = 30.0,
    track_name: str = "Unknown"
) -> Tuple[int, float, List[dict], int]:
    """Check all clips in a track vs SRT segments matched by segment_index.

    Returns (mismatch_count, max_drift, mismatches, clips_checked).
    Only checks clips that have a matching SRT segment (by segment_index).
    """
    mismatches = []
    max_drift = 0.0
    mismatch_count = 0
    clips_checked = 0

    clips = get_track_clips(track)
    clip_positions = compute_clip_timeline_positions(clips)

    # Build SRT index map for O(1) lookup
    srt_map = {i: seg for i, seg in enumerate(srt_segments)}  # 0-indexed

    for seg_idx, (tl_pos, clip_dur) in clip_positions.items():
        if seg_idx not in srt_map:
            # No matching SRT segment for this clip - skip (gap is valid)
            continue

        clips_checked += 1
        srt_seg = srt_map[seg_idx]
        expected_start = srt_seg['start']
        srt_dur = srt_seg['end'] - srt_seg['start']

        drift = tl_pos - expected_start
        dur_drift = clip_dur - srt_dur
        max_drift = max(max_drift, abs(drift), abs(dur_drift))

        if abs(drift) > DRIFT_ERROR or abs(dur_drift) > DRIFT_ERROR:
            mismatches.append({
                'segment_index': seg_idx,
                'clip_timeline': tl_pos,
                'srt_start': expected_start,
                'drift': drift,
                'clip_dur': clip_dur,
                'srt_dur': srt_dur,
                'dur_drift': dur_drift,
            })
            mismatch_count += 1

    return mismatch_count, max_drift, mismatches, clips_checked


def get_all_video_tracks(timeline: otio.schema.Timeline) -> List[Tuple[str, otio.schema.Track]]:
    """Get all video tracks (V1-V6) that exist in the timeline."""
    video_tracks = []
    track_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    found = set()
    for track in timeline.tracks:
        for name in track_names:
            if track.name and track.name.startswith(name) and name not in found:
                video_tracks.append((name, track))
                found.add(name)
                break
    return video_tracks


def get_all_audio_tracks(timeline: otio.schema.Timeline) -> List[Tuple[str, otio.schema.Track]]:
    """Get all audio tracks (A1-A8) that exist in the timeline."""
    audio_tracks = []
    track_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    found = set()
    for track in timeline.tracks:
        for name in track_names:
            if track.name and track.name.startswith(name) and name not in found:
                audio_tracks.append((name, track))
                found.add(name)
                break
    return audio_tracks


def fix_track_from_srt(
    track: otio.schema.Track,
    srt_segments: List[dict],
    fps: float = 30.0,
    track_name: str = "Unknown"
) -> Tuple[int, int]:
    """Fix clips in a track that have matching SRT segments (by segment_index).

    Returns (clips_fixed, clips_skipped) - clips_skipped includes gaps and non-matched clips.
    """
    clips = get_track_clips(track)
    srt_map = {i: seg for i, seg in enumerate(srt_segments)}  # 0-indexed

    clips_fixed = 0
    clips_skipped = 0

    for clip in clips:
        seg_idx = extract_segment_index(clip)
        if seg_idx is None:
            clips_skipped += 1
            continue
        if seg_idx not in srt_map:
            # Gap - no matching SRT segment, skip
            clips_skipped += 1
            continue

        srt_seg = srt_map[seg_idx]
        srt_dur = srt_seg['end'] - srt_seg['start']

        # Update clip duration to match SRT
        new_duration = otio.opentime.RationalTime(srt_dur * fps, fps)
        clip.source_range = otio.opentime.TimeRange(
            start_time=clip.source_range.start_time,
            duration=new_duration
        )
        clips_fixed += 1

    return clips_fixed, clips_skipped


def check_all_tracks(
    timeline: otio.schema.Timeline,
    srt_segments: List[dict],
    fps: float = 30.0
) -> dict:
    """Check alignment across all video and audio tracks.

    Returns a dict with per-track results: {track_name: {'checked': N, 'drifted': M, 'max_drift': X}}
    """
    results = {}

    # Check video tracks (V1-V6)
    for track_name, track in get_all_video_tracks(timeline):
        mismatch_count, max_drift, mismatches, clips_checked = check_track_alignment(
            track, srt_segments, fps, track_name
        )
        results[track_name] = {
            'checked': clips_checked,
            'drifted': mismatch_count,
            'max_drift': max_drift,
            'mismatches': mismatches[:5] if mismatches else [],
        }

    # Check audio tracks (A1-A8)
    for track_name, track in get_all_audio_tracks(timeline):
        mismatch_count, max_drift, mismatches, clips_checked = check_track_alignment(
            track, srt_segments, fps, track_name
        )
        results[track_name] = {
            'checked': clips_checked,
            'drifted': mismatch_count,
            'max_drift': max_drift,
            'mismatches': mismatches[:5] if mismatches else [],
        }

    return results


def fix_all_tracks(
    timeline: otio.schema.Timeline,
    srt_segments: List[dict],
    fps: float = 30.0
) -> dict:
    """Fix clips across all video and audio tracks.

    Returns a dict with per-track fix results: {track_name: {'fixed': N, 'skipped': M}}
    """
    results = {}

    # Fix video tracks (V1-V6)
    for track_name, track in get_all_video_tracks(timeline):
        clips_fixed, clips_skipped = fix_track_from_srt(track, srt_segments, fps, track_name)
        results[track_name] = {'fixed': clips_fixed, 'skipped': clips_skipped}

    # Fix audio tracks (A1-A8)
    for track_name, track in get_all_audio_tracks(timeline):
        clips_fixed, clips_skipped = fix_track_from_srt(track, srt_segments, fps, track_name)
        results[track_name] = {'fixed': clips_fixed, 'skipped': clips_skipped}

    return results


def fix_timeline_from_srt(
    timeline: otio.schema.Timeline,
    srt_segments: List[dict],
    fps: float = 30.0
) -> dict:
    """Rebuild all video and audio tracks with corrected clip positions from SRT.

    Only fixes clips that have a matching SRT segment (by segment_index).
    Skips clips without segment index or without matching SRT segment (gaps).

    Returns dict with per-track results: {track_name: {'fixed': N, 'skipped': M}}
    """
    return fix_all_tracks(timeline, srt_segments, fps)


def fix_timeline_from_checkpoint(
    timeline: otio.schema.Timeline,
    checkpoint_path: Path,
    srt_path: Optional[Path] = None,
    fps: float = 30.0
) -> dict:
    """Fix timeline using checkpoint's clean trimmed_segments.

    The checkpoint stores trimmed_segments captured from raw Whisper output (not SRT).
    This is the preferred fix when checkpoint has clean data.

    Returns dict with per-track results: {track_name: {'fixed': N, 'skipped': M}}
    """
    # Load checkpoint
    with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
        cp = json.load(f)

    trimmed_segs = cp.get('analyze', {}).get('trimmed_segments', [])
    if not trimmed_segs:
        raise ValueError(f"Checkpoint has no trimmed_segments: {checkpoint_path}")

    # Convert to seconds
    srt_segments = [
        {'start': s['start'], 'end': s['end'], 'text': s.get('text', '')}
        for s in trimmed_segs
    ]

    # If SRT path provided, verify it matches checkpoint trimmed_segments
    if srt_path and srt_path.exists():
        srt_parsed = parse_srt(srt_path)
        if len(srt_parsed) == len(srt_segments):
            # Check if SRT matches checkpoint within truncation tolerance
            match = all(
                abs(srt_parsed[i]['start'] - s['start']) < 0.002 and
                abs(srt_parsed[i]['end'] - s['end']) < 0.002
                for i, s in enumerate(srt_segments)
            )
            if match:
                logger.info("SRT matches checkpoint trimmed_segments (within truncation tolerance)")
            else:
                logger.warning("SRT differs from checkpoint trimmed_segments — using checkpoint data")

    return fix_all_tracks(timeline, srt_segments, fps)


def detect_timing_system(otio_path: Path) -> Tuple[str, float]:
    """Detect if timeline uses original or trimmed timing based on duration."""
    timeline = otio.adapters.read_from_file(str(otio_path))
    total_dur = timeline.duration().value / timeline.duration().rate

    # Heuristics: original timing ~685s, trimmed ~556s
    if 650 < total_dur < 720:
        return 'ORIGINAL', total_dur
    elif 500 < total_dur < 600:
        return 'TRIMMED', total_dur
    else:
        return 'UNKNOWN', total_dur


def find_reference_srt(otio_path: Path, explicit_srt: Optional[Path] = None) -> Optional[Path]:
    """Find reference SRT for the timeline."""
    if explicit_srt and explicit_srt.exists():
        return explicit_srt

    # OTIO path: .../output/<timestamp>/timeline_V1_*.otio
    # Project path: .../ (parent of output/)
    otio_dir = otio_path.parent
    # output/ dir is directly under project, so go up two levels
    project_dir = otio_dir.parent.parent if otio_dir.parent.name == 'output' else otio_dir.parent

    candidates = [
        project_dir / 'voiceover' / 'voiceover_trimmed.srt',
        project_dir / 'voiceover' / 'voiceover.srt',
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def find_project_dir(otio_path: Path) -> Path:
    """Find project directory by scanning up from output/ until checkpoint.json is found."""
    # OTIO is typically at: <project>/output/<timestamp>/timeline_*.otio
    # Scan up to find the project root
    search = otio_path.parent
    for _ in range(6):  # safety limit
        if (search / 'checkpoint.json').exists() or (search / 'checkpoint.backup.json').exists():
            return search
        parent = search.parent
        if parent == search:  # reached filesystem root
            break
        search = parent
    # fallback: assume two levels up from output dir
    return otio_path.parent.parent


def find_checkpoint(project_dir: Path) -> Optional[Path]:
    """Find checkpoint.json for project."""
    for name in ('checkpoint.json', 'checkpoint.backup.json'):
        cp = project_dir / name
        if cp.exists():
            return cp
    return None


def main():
    parser = argparse.ArgumentParser(description='Fix OTIO timeline clip timings')
    parser.add_argument('otio', help='Path to OTIO timeline file')
    parser.add_argument('--srt', type=Path, help='Reference SRT file (default: auto-detect)')
    parser.add_argument('--output', '-o', type=Path, help='Output path (default: overwrite original)')
    parser.add_argument('--fps', type=float, default=30.0, help='Frame rate (default: 30.0)')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='Prefer checkpoint trimmed_segments over SRT (recommended)')
    parser.add_argument('--dry-run', action='store_true', help='Check only, do not modify')
    args = parser.parse_args()

    otio_path = Path(args.otio)
    if not otio_path.exists():
        logger.error(f"OTIO file not found: {otio_path}")
        sys.exit(1)

    # Detect timing system
    timing_sys, total_dur = detect_timing_system(otio_path)
    logger.info(f"Detected timing: {timing_sys} ({total_dur:.1f}s)")

    # Find reference SRT
    srt_path = find_reference_srt(otio_path, args.srt)
    if not srt_path:
        logger.warning("No reference SRT found — will use checkpoint only")
    else:
        logger.info(f"Reference SRT: {srt_path}")

    # Load timeline
    timeline = otio.adapters.read_from_file(str(otio_path))
    fps = args.fps

    # List all available tracks
    video_tracks = get_all_video_tracks(timeline)
    audio_tracks = get_all_audio_tracks(timeline)
    all_tracks = video_tracks + audio_tracks

    if not all_tracks:
        logger.error("No video or audio tracks found in timeline")
        sys.exit(1)

    for track_name, _ in all_tracks:
        clip_count = len(get_track_clips(_))
        logger.info(f"  {track_name}: {clip_count} clips")

    # Load reference segments
    srt_segments = []
    if srt_path:
        srt_segments = parse_srt(srt_path)
        logger.info(f"SRT segments: {len(srt_segments)}")

    # Check all tracks alignment
    if srt_segments:
        all_results = check_all_tracks(timeline, srt_segments, fps)
        total_checked = sum(r['checked'] for r in all_results.values())
        total_drifted = sum(r['drifted'] for r in all_results.values())
        logger.info(f"All tracks checked: {total_checked} clips | Drifted: {total_drifted}")

        for track_name, result in all_results.items():
            if result['checked'] > 0:
                logger.info(
                    f"  {track_name}: checked={result['checked']} "
                    f"drifted={result['drifted']} max_drift={result['max_drift']:.3f}s"
                )
                for m in result['mismatches']:
                    logger.warning(
                        f"    Segment [S{m['segment_index']}]: "
                        f"tl={m['clip_timeline']:.3f}s expected={m['srt_start']:.3f}s "
                        f"drift={m['drift']:+.3f}s dur={m['clip_dur']:.3f}s "
                        f"expected={m['srt_dur']:.3f}s"
                    )

        if total_drifted == 0:
            logger.info("All tracks OK — no fix needed")
            return

    # Try checkpoint fix
    project_dir = find_project_dir(otio_path)
    checkpoint_path = find_checkpoint(project_dir)

    fix_results = {}
    if args.use_checkpoint or (not srt_segments and checkpoint_path):
        if not checkpoint_path:
            logger.error("No checkpoint found — cannot use checkpoint fix")
            sys.exit(1)

        logger.info(f"Using checkpoint: {checkpoint_path}")
        try:
            fix_results = fix_timeline_from_checkpoint(timeline, checkpoint_path, srt_path, fps)
        except Exception as e:
            logger.error(f"Checkpoint fix failed: {e}")
            sys.exit(1)
    elif srt_segments:
        if args.dry_run:
            logger.info("Dry run — not applying fix")
            return
        logger.info("Applying fix from SRT...")
        fix_results = fix_all_tracks(timeline, srt_segments, fps)
    else:
        logger.error("No reference SRT and no checkpoint — cannot fix")
        sys.exit(1)

    # Report fix results per track
    total_fixed = sum(r['fixed'] for r in fix_results.values())
    total_skipped = sum(r['skipped'] for r in fix_results.values())
    logger.info(f"Fix applied: {total_fixed} clips fixed, {total_skipped} skipped (gaps/non-matched)")

    for track_name, result in fix_results.items():
        if result['fixed'] > 0 or result['skipped'] > 0:
            logger.info(f"  {track_name}: fixed={result['fixed']} skipped={result['skipped']}")

    # Validate fix by re-checking all tracks
    if srt_segments:
        all_results = check_all_tracks(timeline, srt_segments, fps)
        total_drifted = sum(r['drifted'] for r in all_results.values())
        logger.info(f"Re-checked: {total_drifted} clips still drifted")
        for track_name, result in all_results.items():
            if result['drifted'] > 0:
                logger.warning(f"  {track_name}: {result['drifted']} clips still out of alignment")

    # Save
    output_path = args.output or otio_path
    if args.dry_run:
        logger.info(f"Dry run — would save to: {output_path}")
        return

    otio.adapters.write_to_file(timeline, str(output_path))
    logger.info(f"Fixed timeline saved: {output_path}")


if __name__ == '__main__':
    main()
