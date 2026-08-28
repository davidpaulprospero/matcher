#!/usr/bin/env python3
"""
diagnose-otio-mismatch skill

Investigates OTIO timeline timing issues by checking:
1. Timeline vs SRT structure (clip counts, gap counts, durations)
2. Checkpoint vs actual SRT on disk (trimmed_segments timing)
3. Leading gap investigation
4. SRT source resolution on --output-only
5. Clip duration vs SRT duration for each clip
6. Timeline total duration vs SRT total duration

Run with: python skill.py "path/to/timeline.otio" --project "path/to/project"

JSON mode:
  python skill.py timeline.otio --project . --json
  python skill.py timeline.otio --project . --json --output-json result.json
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Detect if terminal supports UTF-8, fallback to ASCII-safe mode
def _supports_unicode() -> bool:
    try:
        import os
        # Explicit UTF-8 env var always wins
        encoding = os.environ.get('PYTHONIOENCODING', '').lower()
        if encoding in ('utf-8', 'utf8'):
            return True
        # On Windows, always return False to force ASCII-safe output
        # sys.stdout.encoding is unreliable on Windows consoles
        if sys.platform == 'win32':
            return False
        # Try encoding a checkmark to the actual stdout encoding
        '\u2713'.encode(sys.stdout.encoding or 'ascii')
        return True
    except (LookupError, UnicodeEncodeError, OSError):
        return False

_UNICODE = _supports_unicode()

# ANSI color codes for colored output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

if _UNICODE:
    CHECK = f"{Colors.GREEN}{Colors.BOLD}\u2713{Colors.RESET}"
    WARN = f"{Colors.YELLOW}{Colors.BOLD}\u26a0{Colors.RESET}"
    FAIL = f"{Colors.RED}{Colors.BOLD}\u2717{Colors.RESET}"
    INFO = f"{Colors.BLUE}{Colors.BOLD}\u2139{Colors.RESET}"
    PASS = f"{Colors.GREEN}{Colors.BOLD}\u2713{Colors.RESET}"
else:
    CHECK = f"{Colors.GREEN}[OK]{Colors.RESET}"
    WARN = f"{Colors.YELLOW}[WARN]{Colors.RESET}"
    FAIL = f"{Colors.RED}[FAIL]{Colors.RESET}"
    INFO = f"{Colors.BLUE}[INFO]{Colors.RESET}"
    PASS = f"{Colors.GREEN}[PASS]{Colors.RESET}"


def parse_srt(srt_path: Path, verbose: bool = False) -> list:
    """Parse SRT file, return list of {start, end, text} in seconds."""
    try:
        content = srt_path.read_text(encoding='utf-8-sig')
    except Exception as e:
        if verbose:
            print(f"{WARN} Failed to read SRT file: {e}")
        return []

    blocks = re.split(r'\n\n+', content.strip())
    segments = []
    for idx, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            if verbose:
                print(f"{WARN} Skipping malformed block {idx + 1}: not enough lines")
            continue
        try:
            ts_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
                lines[1]
            )
            if not ts_match:
                if verbose:
                    print(f"{WARN} Skipping block {idx + 1}: invalid timestamp format '{lines[1]}'")
                continue
            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, ts_match.groups())
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            text = ' '.join(lines[2:]).strip()
            segments.append({'start': start, 'end': end, 'text': text})
        except (ValueError, IndexError) as e:
            if verbose:
                print(f"{WARN} Skipping block {idx + 1} due to parse error: {e}")
            continue
    return segments


def load_checkpoint(project_path: Path) -> dict:
    """Load checkpoint, handling gzip compression."""
    cp_path = project_path / 'checkpoint.json'
    if not cp_path.exists():
        return {}
    try:
        data = open(cp_path, 'rb').read()
        return json.loads(gzip.decompress(data))
    except Exception:
        try:
            return json.loads(data.decode('latin1'))
        except Exception:
            return {}


def load_timeline(timeline_path: Path, verbose: bool = False) -> tuple:
    """Load timeline with opentimelineio, return (timeline, v1_track)."""
    try:
        import opentimelineio as otio
    except ImportError:
        if verbose:
            print(f"{FAIL} opentimelineio is not installed. Run: pip install opentimelineio")
        return None, None

    try:
        tl = otio.adapters.read_from_file(str(timeline_path))
    except Exception as e:
        if verbose:
            print(f"{FAIL} Failed to read OTIO file: {e}")
        return None, None

    v1 = None
    v1_candidates = []
    for track in tl.tracks:
        name = getattr(track, 'name', '') or ''
        if 'V1' in name or 'V1 - Primary' in name:
            v1_candidates.append(track)
        if name == 'V1 - Primary':
            v1 = track
            break

    if v1 is None and v1_candidates:
        if verbose:
            print(f"{INFO} V1 track not found by exact name, using first candidate")
        v1 = v1_candidates[0]

    if verbose:
        if v1:
            print(f"{CHECK} Found V1 track: {v1.name}")
        else:
            print(f"{WARN} No V1 track found. Available tracks: {[getattr(t, 'name', '') for t in tl.tracks]}")

    return tl, v1


def check_timeline_vs_srt(v1, srt_segments, project_path: Optional[Path] = None, json_mode: bool = False):
    """Check Area 1: Timeline vs SRT Structure."""
    if not json_mode:
        print()
        print("=== Area 1: Timeline vs SRT Structure ===")

    result = {
        'clips': 0,
        'gaps': 0,
        'srt_segments': len(srt_segments),
        'max_pos_drift': 0.0,
    }

    if v1 is None:
        result['status'] = 'fail'
        result['error'] = 'Could not load timeline or find V1 track'
        if not json_mode:
            print("[WARN] Could not load timeline or find V1 track")
        return result

    children = list(v1)
    clips = [c for c in children if str(type(c).__name__) == 'Clip']
    gaps = [g for g in children if str(type(g).__name__) == 'Gap']

    result['clips'] = len(clips)
    result['gaps'] = len(gaps)

    if not json_mode:
        print(f"V1 items: {len(children)} total, {len(clips)} clips, {len(gaps)} gaps")
        print(f"SRT segments: {len(srt_segments)}")

    # --- Sub-check 1a: Clip count vs SRT segment count ---
    clip_count_match = len(clips) == len(srt_segments)
    result['clip_count_match'] = clip_count_match

    if len(clips) == len(srt_segments):
        if not json_mode:
            print(f"[OK] Clip count matches SRT segment count")
    else:
        if not json_mode:
            print(f"[WARN] MISMATCH: {len(clips)} clips vs {len(srt_segments)} SRT segments")
            print("  -> Look at: src/otio/timeline.py lines ~1014-1112 (gap insertion logic)")
            print("  -> Also check: src/otio/timeline.py lines ~1117-1125 (duration_frames calculation)")

    # --- Sub-check 1b: SRT vs Timeline total duration ---
    srt_total_dur = 0.0
    timeline_total_dur = 0.0
    dur_diff = 0.0

    if srt_segments:
        srt_total_dur = sum(seg['end'] - seg['start'] for seg in srt_segments)
        timeline_total_dur = sum(c.duration().to_seconds() for c in clips) + sum(g.duration().to_seconds() for g in gaps)
        dur_diff = abs(timeline_total_dur - srt_total_dur)
        result['srt_total_dur'] = srt_total_dur
        result['timeline_total_dur'] = timeline_total_dur
        result['dur_diff'] = dur_diff

        if not json_mode:
            print(f"\n[SRT vs Timeline Duration]")
            print(f"  SRT total:        {srt_total_dur:.3f}s (sum of {len(srt_segments)} segment durations)")
            print(f"  Timeline total:   {timeline_total_dur:.3f}s (clips + gaps)")
            print(f"  Difference:       {dur_diff:.3f}s")
            if dur_diff < 0.1:
                print(f"  [OK] Duration matches SRT total")
            elif dur_diff < 1.0:
                print(f"  [WARN] Minor duration mismatch ({dur_diff:.3f}s) — may indicate rounding drift")
            else:
                print(f"  [WARN] SEVERE mismatch ({dur_diff:.3f}s) — check for missing clips or extra gaps")

    # --- Sub-check 1c: Checkpoint trimmed_segments count vs timeline clip count ---
    checkpoint_trimmed_count = 0
    if project_path:
        ck = load_checkpoint(project_path)
        if ck:
            trim_dicts = ck.get('analyze', {}).get('trimmed_segments', [])
            checkpoint_trimmed_count = len(trim_dicts)
            result['checkpoint_trimmed_count'] = checkpoint_trimmed_count

            if not json_mode:
                print(f"\n[Checkpoint vs Timeline Clip Count]")
                print(f"  checkpoint.trimmed_segments: {len(trim_dicts)}")
                print(f"  timeline V1 clips:            {len(clips)}")
                if len(trim_dicts) == len(clips):
                    print(f"  [OK] Segment count matches timeline clip count")
                elif len(trim_dicts) > len(clips):
                    diff = len(trim_dicts) - len(clips)
                    print(f"  [WARN] Checkpoint has {diff} MORE segments ({len(trim_dicts)}) than timeline clips ({len(clips)})")
                    print(f"  -> {diff} unmatched segments — may indicate clips dropped during OTIO export")
                else:
                    diff = len(clips) - len(trim_dicts)
                    print(f"  [WARN] Checkpoint has {diff} FEWER segments ({len(trim_dicts)}) than timeline clips ({len(clips)})")
                    print(f"  -> {diff} extra clips in timeline — may indicate duplicate clip insertion")
        else:
            if not json_mode:
                print(f"\n[Checkpoint vs Timeline Clip Count]")
                print("  [WARN] Could not load checkpoint for cross-validation")

    # --- Sub-check 1d: Gap Location Analysis ---
    if len(gaps) > 0:
        gap_info = []
        if not json_mode:
            print(f"\n[Gap Location Analysis]")
            print(f"  V1 has {len(gaps)} gap(s) — reporting surrounding SRT segments...")

        # Build SRT position map
        srt_positions = []
        running = 0.0
        for seg in srt_segments:
            srt_positions.append({'start': running, 'end': running + (seg['end'] - seg['start']), 'text': seg['text']})
            running = srt_positions[-1]['end']

        for i, gap in enumerate(gaps):
            gap_idx = list(v1).index(gap)
            gap_start = sum(c.duration().to_seconds() for c in children[:gap_idx])
            gap_dur = gap.duration().to_seconds()
            gap_end = gap_start + gap_dur

            # Find surrounding SRT segments by position
            before_idx = None
            after_idx = None
            for j, pos in enumerate(srt_positions):
                if pos['end'] <= gap_start + 0.001:
                    before_idx = j
                if pos['start'] >= gap_end - 0.001 and after_idx is None:
                    after_idx = j
                    break

            before_seg = srt_segments[before_idx] if before_idx is not None else None
            after_seg = srt_segments[after_idx] if after_idx is not None else None

            gap_entry = {
                'index': i,
                'duration': gap_dur,
                'start': gap_start,
                'end': gap_end,
                'before_srt_index': before_idx,
                'after_srt_index': after_idx,
            }
            gap_info.append(gap_entry)

            if not json_mode:
                print(f"  Gap {i+1}: {gap_dur:.3f}s at timeline {gap_start:.3f}s -> {gap_end:.3f}s")
                if before_seg:
                    txt = before_seg['text']
                    print(f"    Before: SRT seg {before_idx+1} ends at {before_seg['end']:.3f}s")
                    print(f"            \"{txt[:60]}...\"" if len(txt) > 60 else f"            \"{txt}\"")
                else:
                    print(f"    Before: (gap at timeline start)")
                if after_seg:
                    txt = after_seg['text']
                    print(f"    After:  SRT seg {after_idx+1} starts at {after_seg['start']:.3f}s")
                    print(f"            \"{txt[:60]}...\"" if len(txt) > 60 else f"            \"{txt}\"")
                if gap_dur > 2.0:
                    print(f"    [WARN] Gap > 2s — possible silence incorrectly preserved as gap")
            if gap_dur > 2.0:
                gap_entry['large_gap'] = True

        result['gaps_info'] = gap_info
        if not json_mode:
            print("  -> Look at: src/otio/timeline.py lines ~1014-1112")
            print("  -> timeline_frames advances by source_range.duration (media dur), not SRT segment dur")
            print("  -> This causes drift -> spurious gaps")
    else:
        if not json_mode:
            print(f"[OK] No gaps (expected for trimmed voiceover)")

    # Check clip positions vs SRT
    max_pos_drift = 0.0
    for i, clip in enumerate(clips[:10]):
        p = 0.0
        for c in list(v1)[:i]:
            p += c.duration().to_seconds()

        if i < len(srt_segments):
            srt = srt_segments[i]
            drift = abs(p - srt['start'])
            max_pos_drift = max(max_pos_drift, drift)

    result['max_pos_drift'] = max_pos_drift

    if not json_mode:
        if max_pos_drift < 0.033:
            print(f"\n[OK] Clip positions match SRT (max drift {max_pos_drift:.3f}s < 0.033s threshold)")
        elif max_pos_drift < 0.5:
            print(f"\n[WARN] Minor position drift: {max_pos_drift:.3f}s")
        else:
            print(f"\n[WARN] SEVERE position drift: {max_pos_drift:.3f}s (> 0.5s)")
            print("  -> Look at: src/otio/timeline.py lines ~1117-1125")
            print("  -> duration_frames = max(1, expected_end_frames - timeline_frames)")
            print("  -> timeline_frames advances by media duration, causing accumulated drift")

    # Determine status
    has_severe_issues = max_pos_drift >= 0.5 or len(gaps) > 0 or dur_diff >= 1.0 or not clip_count_match
    has_minor_issues = max_pos_drift >= 0.033 or dur_diff >= 0.1

    if has_severe_issues:
        result['status'] = 'fail'
    elif has_minor_issues:
        result['status'] = 'warn'
    else:
        result['status'] = 'pass'

    return result


def check_checkpoint_vs_srt(project_path: Path, srt_segments: list, json_mode: bool = False) -> dict:
    """Check Area 2: Checkpoint vs Actual SRT."""
    if not json_mode:
        print()
        print("=== Area 2: Checkpoint vs Actual SRT ===")

    result = {}

    ck = load_checkpoint(project_path)
    if not ck:
        result['status'] = 'fail'
        result['error'] = 'Could not load checkpoint'
        if not json_mode:
            print("[WARN] Could not load checkpoint")
        return result

    analyze = ck.get('analyze', {})
    trim_dicts = analyze.get('trimmed_segments', [])
    segs = analyze.get('segments', [])

    result['cp_segments_count'] = len(segs)
    result['cp_trimmed_count'] = len(trim_dicts)

    if not json_mode:
        print(f"Checkpoint analyze.segments: {len(segs)}")
        print(f"Checkpoint trimmed_segments: {len(trim_dicts)}")

    cp_first = None
    srt_first = None

    # --- Sub-check 2a: Per-segment confidence flags (if available) ---
    if trim_dicts:
        confidence_warnings = []
        for i, seg in enumerate(trim_dicts):
            conf = seg.get('confidence') or seg.get('score') or seg.get('match_confidence')
            if conf is not None and isinstance(conf, (int, float)) and conf < 0.6:
                confidence_warnings.append({'index': i, 'confidence': conf})
            if i < len(srt_segments):
                srt_dur = srt_segments[i]['end'] - srt_segments[i]['start']
                cp_dur = seg.get('duration', seg.get('end', 0) - seg.get('start', 0))
                if cp_dur > 0 and abs(cp_dur - srt_dur) > 1.0:
                    confidence_warnings.append({'index': i, 'confidence': conf, 'dur_mismatch': abs(cp_dur - srt_dur)})

        cp_first = trim_dicts[0].get('start', 'N/A')
        srt_first = srt_segments[0]['start'] if srt_segments else 'N/A'
        result['cp_first_trimmed'] = cp_first
        result['srt_first'] = srt_first

        if not json_mode:
            print(f"Checkpoint trimmed_segments[0].start: {cp_first}s")
            print(f"Actual voiceover_trimmed.srt[0].start: {srt_first}s")

        if abs(cp_first - srt_first) < 0.01:
            result['status'] = 'pass'
            if not json_mode:
                print("[OK] Checkpoint matches actual SRT")
        else:
            result['status'] = 'fail'
            result['mismatch'] = True
            result['mismatch_amount'] = abs(cp_first - srt_first)
            if not json_mode:
                print(f"[WARN] MISMATCH: checkpoint uses {cp_first}s, SRT has {srt_first}s")
                print("  -> This causes wrong clip positions in timeline")
                print("  -> Look at: src/stages/analyze.py lines ~261-280")
                print("  -> The restore logic checks: is_trimmed_voiceover(state.voiceover_path)")
                print("  -> If state.voiceover_path is empty or doesn't contain '_trimmed',")
                print("     the trimmed resync is skipped and wrong timings are used")

        result['confidence_warnings'] = confidence_warnings
        if confidence_warnings:
            if not json_mode:
                print(f"\n[Confidence Correlation]")
                print(f"  [WARN] Found {len(confidence_warnings)} segments with low confidence or large duration mismatch:")
                for w in confidence_warnings[:10]:
                    idx = w['index']
                    conf = w.get('confidence')
                    dur_mis = w.get('dur_mismatch')
                    seg_text = srt_segments[idx]['text'][:50] if idx < len(srt_segments) else 'N/A'
                    if conf is not None:
                        print(f"    Seg {idx+1}: confidence={conf:.3f} < 0.6 — \"{seg_text}...\"")
                    if dur_mis:
                        print(f"    Seg {idx+1}: duration mismatch={dur_mis:.3f}s — \"{seg_text}...\"")
                if len(confidence_warnings) > 10:
                    print(f"    ... and {len(confidence_warnings) - 10} more")
                print("  -> Low confidence segments may indicate timing issues")
                print("  -> Look at: src/matching/scoring.py for confidence threshold logic")
        else:
            if not json_mode:
                print(f"\n[Confidence Correlation]")
                print(f"  [OK] No low-confidence segments detected in checkpoint")
    else:
        result['status'] = 'warn'
        result['warning'] = 'No trimmed_segments in checkpoint'
        if not json_mode:
            print("[WARN] No trimmed_segments in checkpoint")

    if segs:
        cp_segs_first = segs[0].get('start', 'N/A')
        result['cp_segs_first'] = cp_segs_first
        if not json_mode:
            print(f"Checkpoint analyze.segments[0].start: {cp_segs_first}s")

    return result


def check_leading_gap(v1, srt_segments, json_mode: bool = False):
    """Check Area 3: Leading Gap Investigation."""
    if not json_mode:
        print()
        print("=== Area 3: Leading Gap Investigation ===")

    result = {}

    if v1 is None or not srt_segments:
        result['status'] = 'fail'
        result['error'] = 'Cannot check leading gap - missing V1 or SRT'
        if not json_mode:
            print("[WARN] Cannot check leading gap")
        return result

    children = list(v1)
    first_item = children[0] if children else None

    if str(type(first_item).__name__) == 'Gap':
        leading_gap = first_item.duration().to_seconds()
        result['leading_gap'] = leading_gap
        if not json_mode:
            print(f"Timeline has leading gap: {leading_gap:.3f}s")
    else:
        leading_gap = 0.0
        result['leading_gap'] = 0.0
        if not json_mode:
            print("Timeline has no leading gap (first item is a clip)")

    srt_first_start = srt_segments[0]['start']
    result['srt_first_start'] = srt_first_start
    if not json_mode:
        print(f"First SRT segment starts at: {srt_first_start:.3f}s")

    if leading_gap > 0.1 and srt_first_start < 0.1:
        result['extra_gap_inserted'] = True
        result['status'] = 'fail'
        if not json_mode:
            print(f"[WARN] Leading gap {leading_gap:.3f}s but SRT starts at 0.0 — extra gap inserted")
            print("  -> Look at: src/otio/timeline.py lines ~938-963")
            print("  -> leading_frames = round(adjusted_first_segment_start * rate)")
            print("  -> adjusted_first_segment_start comes from matches[0], not SRT on disk")
    elif abs(leading_gap - srt_first_start) < 0.1:
        result['status'] = 'pass'
        if not json_mode:
            print("[OK] Leading gap matches SRT first segment start")
    else:
        result['status'] = 'warn'
        result['mismatch'] = True
        if not json_mode:
            print(f"[WARN] Leading gap ({leading_gap:.3f}s) doesn't match SRT ({srt_first_start:.3f}s)")

    return result


def check_srt_source(project_path: Path, json_mode: bool = False) -> dict:
    """Check Area 4: SRT Source Resolution."""
    if not json_mode:
        print()
        print("=== Area 4: SRT Source Resolution ===")

    result = {}

    voiceover_dir = project_path / 'voiceover'
    if not voiceover_dir.exists():
        result['status'] = 'fail'
        result['error'] = f'Voiceover directory not found: {voiceover_dir}'
        if not json_mode:
            print(f"[WARN] Voiceover directory not found: {voiceover_dir}")
        return result

    trimmed_audio = None
    for f in voiceover_dir.iterdir():
        if f.suffix.lower() in ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg') and '_trimmed' in f.stem:
            trimmed_audio = f
            break

    trimmed_srt = None
    if trimmed_audio:
        result['trimmed_audio'] = str(trimmed_audio)
        if not json_mode:
            print(f"Trimmed audio found: {trimmed_audio.name}")

        trimmed_srt = trimmed_audio.with_suffix('.srt')
        result['trimmed_srt'] = str(trimmed_srt)
        if trimmed_srt.exists():
            if not json_mode:
                print(f"[OK] Companion SRT exists: {trimmed_srt.name}")
        else:
            result['status'] = 'fail'
            result['companion_srt_missing'] = True
            if not json_mode:
                print(f"[WARN] Companion SRT NOT found: {trimmed_srt.name}")
                print("  -> _resync_trimmed_segments() will fail to find the SRT")
                print("  -> Look at: src/stages/analyze.py lines ~317-320")
    else:
        result['status'] = 'fail'
        result['trimmed_audio'] = None
        result['trimmed_srt'] = None
        if not json_mode:
            print("[WARN] No trimmed audio found in voiceover/ directory")

    original_audio = None
    for f in voiceover_dir.iterdir():
        if f.suffix.lower() in ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg') and '_trimmed' not in f.stem:
            original_audio = f
            break

    if original_audio:
        result['original_audio'] = str(original_audio)
        if not json_mode:
            print(f"Original audio found: {original_audio.name}")

    # Set status if not already set
    if 'status' not in result:
        result['status'] = 'pass'

    return result


def check_clip_duration_vs_srt(v1, srt_segments, json_mode: bool = False):
    """Check Area 5: Clip Duration vs SRT Duration."""
    if not json_mode:
        print()
        print("=== Area 5: Clip Duration vs SRT Duration ===")

    result = {
        'total_clips': 0,
        'mismatch_count': 0,
        'mismatches': [],
    }

    if v1 is None:
        result['status'] = 'fail'
        result['error'] = 'Could not load timeline or find V1 track'
        if not json_mode:
            print("[WARN] Could not load timeline or find V1 track")
        return result

    children = list(v1)
    clips = [c for c in children if str(type(c).__name__) == 'Clip']

    if not clips:
        result['status'] = 'warn'
        result['warning'] = 'No clips found in V1 track'
        if not json_mode:
            print("[WARN] No clips found in V1 track")
        return result

    result['total_clips'] = len(clips)

    if not json_mode:
        print(f"Comparing {len(clips)} clip durations against {len(srt_segments)} SRT segment durations")
        print(f"Threshold for reporting: 0.1s difference")

    mismatches = []
    for i, clip in enumerate(clips):
        if i >= len(srt_segments):
            break

        clip_dur = clip.duration().to_seconds()
        srt_seg = srt_segments[i]
        srt_dur = srt_seg['end'] - srt_seg['start']
        diff = abs(clip_dur - srt_dur)

        if diff > 0.1:
            mismatches.append({
                'index': i,
                'clip_duration': clip_dur,
                'srt_duration': srt_dur,
                'difference': diff
            })

    result['mismatch_count'] = len(mismatches)
    result['mismatches'] = mismatches

    if not mismatches:
        result['status'] = 'pass'
        if not json_mode:
            print(f"[OK] All clip durations match SRT segment durations (within 0.1s threshold)")
    else:
        result['status'] = 'fail'
        if not json_mode:
            print(f"[WARN] Found {len(mismatches)} clips with duration mismatch > 0.1s:")
            for m in mismatches[:5]:
                print(f"  Clip {m['index']}: clip={m['clip_duration']:.3f}s, SRT={m['srt_duration']:.3f}s, diff={m['difference']:.3f}s")
            if len(mismatches) > 5:
                print(f"  ... and {len(mismatches) - 5} more mismatches")
            print("  -> Look at: src/otio/timeline.py lines ~1117-1125")
            print("  -> duration_frames = max(1, expected_end_frames - timeline_frames)")
            print("  -> timeline_frames advances by source_range.duration (media dur), not SRT segment dur")
            print("  -> If downloaded clip duration differs from SRT segment duration, drift accumulates")

    return result


def check_timeline_total_vs_srt_total(v1, srt_segments, json_mode: bool = False):
    """Check Area 6: Timeline Total Duration vs SRT Total Duration."""
    if not json_mode:
        print()
        print("=== Area 6: Timeline Total Duration vs SRT Total Duration ===")

    result = {}

    if v1 is None:
        result['status'] = 'fail'
        result['error'] = 'Could not load timeline or find V1 track'
        if not json_mode:
            print("[WARN] Could not load timeline or find V1 track")
        return result

    srt_total = sum(seg['end'] - seg['start'] for seg in srt_segments)
    result['srt_total'] = srt_total
    if not json_mode:
        print(f"SRT total duration: {srt_total:.3f}s ({len(srt_segments)} segments)")

    timeline_total = sum(c.duration().to_seconds() for c in list(v1))
    result['timeline_total'] = timeline_total
    if not json_mode:
        print(f"V1 track total duration: {timeline_total:.3f}s")

    diff = abs(timeline_total - srt_total)
    result['difference'] = diff

    if diff < 0.1:
        result['status'] = 'pass'
        if not json_mode:
            print(f"[OK] Timeline total matches SRT total (diff {diff:.3f}s < 0.1s threshold)")
    else:
        result['status'] = 'fail'
        result['timeline_longer'] = timeline_total > srt_total
        if not json_mode:
            print(f"[WARN] Duration mismatch: timeline={timeline_total:.3f}s, SRT={srt_total:.3f}s, diff={diff:.3f}s")
            if timeline_total > srt_total:
                print("  -> Timeline is LONGER than SRT — clips have extra padding or media duration exceeds SRT")
            else:
                print("  -> Timeline is SHORTER than SRT — clips are trimmed or gaps are missing")
            print("  -> Look at: src/otio/timeline.py lines ~1117-1125")
            print("  -> duration_frames = max(1, expected_end_frames - timeline_frames)")
            print("  -> If source_range.duration (media) differs from SRT segment duration, total drifts")
            print("  -> Also check: src/otio/timeline.py lines ~1014-1112 (gap insertion logic)")

    return result


def _compute_summary(results: dict) -> dict:
    """Compute summary from results."""
    issue_count = 0
    areas_with_status = ['timeline_vs_srt', 'checkpoint_vs_srt', 'leading_gap', 'srt_source', 'clip_duration', 'total_duration']
    for area in areas_with_status:
        data = results.get(area, {})
        if isinstance(data, dict) and 'status' in data:
            if data['status'] in ('fail', 'warn'):
                issue_count += 1

    return {
        'has_issues': issue_count > 0,
        'issue_count': issue_count
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose OTIO timeline timing issues')
    parser.add_argument('timeline', help='Path to timeline.otio file')
    parser.add_argument('--project', '-p', help='Path to project directory (for checkpoint and SRT)')
    parser.add_argument('--srt', '-s', help='Path to reference SRT file')
    parser.add_argument('--json', action='store_true', help='Output structured JSON instead of human-readable text')
    parser.add_argument('--output-json', type=str, help='Path to write JSON output file')
    args = parser.parse_args()

    json_mode = args.json

    timeline_path = Path(args.timeline)
    if not timeline_path.exists():
        if json_mode:
            print(json.dumps({'error': f'Timeline not found: {timeline_path}'}))
        else:
            print(f"Timeline not found: {timeline_path}")
        return 1

    project_path = Path(args.project) if args.project else timeline_path.parent.parent

    if not json_mode:
        print(f"[diagnose-otio-mismatch] Analyzing: {timeline_path}")
        print(f"[diagnose-otio-mismatch] Project: {project_path}")

    srt_path = None
    if args.srt:
        srt_path = Path(args.srt)
    else:
        vo_dir = project_path / 'voiceover'
        if vo_dir.exists():
            for f in vo_dir.iterdir():
                if f.suffix.lower() == '.srt' and '_trimmed' in f.stem:
                    srt_path = f
                    break

    if not srt_path or not srt_path.exists():
        if json_mode:
            print(json.dumps({'error': 'SRT file not found. Provide --srt or ensure voiceover/voiceover_trimmed.srt exists'}))
        else:
            print(f"[WARN] SRT file not found")
            print("  -> Provide --srt or ensure voiceover/voiceover_trimmed.srt exists")
        return 1
    else:
        if not json_mode:
            print(f"[diagnose-otio-mismatch] Reference SRT: {srt_path}")

    srt_segments = parse_srt(srt_path) if srt_path and srt_path.exists() else []
    if not json_mode:
        print(f"[diagnose-otio-mismatch] SRT segments: {len(srt_segments)}")

    _, v1 = load_timeline(timeline_path, verbose=not json_mode)

    results = {}

    if srt_segments and v1 is not None:
        results['timeline_vs_srt'] = check_timeline_vs_srt(v1, srt_segments, project_path, json_mode=json_mode)
        results['leading_gap'] = check_leading_gap(v1, srt_segments, json_mode=json_mode)
        results['clip_duration'] = check_clip_duration_vs_srt(v1, srt_segments, json_mode=json_mode)
        results['total_duration'] = check_timeline_total_vs_srt_total(v1, srt_segments, json_mode=json_mode)
    else:
        if json_mode:
            results['timeline_vs_srt'] = {'status': 'fail', 'error': 'Cannot compare — missing SRT or V1 track'}
        else:
            print()
            print("=== Area 1: Timeline vs SRT Structure ===")
            print("[WARN] Cannot compare — missing SRT or V1 track")

    if project_path.exists():
        results['checkpoint_vs_srt'] = check_checkpoint_vs_srt(project_path, srt_segments, json_mode=json_mode)
        results['srt_source'] = check_srt_source(project_path, json_mode=json_mode)
    else:
        if json_mode:
            results['checkpoint_vs_srt'] = {'status': 'fail', 'error': 'Project path does not exist'}
            results['srt_source'] = {'status': 'fail', 'error': 'Project path does not exist'}
        else:
            print()
            print("=== Area 2-4: Checkpoint/SRT Source ===")
            print("[WARN] Project path not provided or doesn't exist")

    # Compute summary
    results['summary'] = _compute_summary(results)

    # Text mode summary
    if not json_mode:
        print()
        print("=== DIAGNOSIS ===")

        has_issue = False
        if results.get('checkpoint_vs_srt', {}).get('cp_first_trimmed') is not None:
            cp = results['checkpoint_vs_srt']['cp_first_trimmed']
            srt = results['checkpoint_vs_srt']['srt_first']
            if cp and srt and abs(cp - srt) > 0.01:
                print("[FAIL] Checkpoint uses different segment timing than SRT on disk")
                print("  -> Problem: src/stages/analyze.py line ~261")
                print("  -> Fix: always call _resync_trimmed_segments() for trimmed voiceover")
                has_issue = True

        if results.get('timeline_vs_srt', {}).get('gaps', 0) > 0:
            print(f"[WARN] Timeline has {results['timeline_vs_srt']['gaps']} gaps in V1")
            print("  -> Problem: src/otio/timeline.py lines ~1014-1112")
            print("  -> timeline_frames advances by source_range.duration, not SRT segment duration")
            print("  -> This causes timeline_frames to drift, triggering spurious gaps")
            has_issue = True

        if results.get('timeline_vs_srt', {}).get('max_pos_drift', 0) > 0.5:
            print(f"[WARN] Severe position drift: {results['timeline_vs_srt']['max_pos_drift']:.3f}s")
            print("  -> Problem: src/otio/timeline.py lines ~1117-1125")
            print("  -> duration_frames calculation causes accumulated drift")
            has_issue = True

        if results.get('clip_duration', {}).get('mismatch_count', 0) > 0:
            mc = results['clip_duration']['mismatch_count']
            print(f"[WARN] {mc} clips have duration mismatch > 0.1s vs SRT")
            print("  -> Problem: src/otio/timeline.py lines ~1117-1125")
            print("  -> timeline_frames advances by media duration, not SRT segment duration")
            has_issue = True

        if results.get('total_duration', {}).get('difference', 0) > 0.1:
            diff = results['total_duration']['difference']
            tl = results['total_duration']['timeline_total']
            srt = results['total_duration']['srt_total']
            print(f"[WARN] Timeline total ({tl:.3f}s) differs from SRT total ({srt:.3f}s) by {diff:.3f}s")
            print("  -> Problem: src/otio/timeline.py lines ~1117-1125")
            print("  -> duration_frames calculation causes accumulated drift")
            has_issue = True

        if not has_issue:
            print("[OK] No obvious issues detected")

        print()
        print("Key investigation points:")
        print("  1. src/stages/analyze.py line ~261 — is trimmed resync being called?")
        print("  2. src/stages/analyze.py lines ~239-260 — is voiceover_path reconstructed correctly?")
        print("  3. src/otio/timeline.py lines ~1117-1125 — duration_frames calculation")
        print("  4. src/otio/timeline.py lines ~1014-1112 — gap insertion logic")
        print()

    # Output JSON
    if json_mode:
        output = json.dumps(results, indent=2)
        if args.output_json:
            Path(args.output_json).write_text(output, encoding='utf-8')
        else:
            print(output)

    return 0


if __name__ == '__main__':
    exit(main())