#!/usr/bin/env python3
"""
Improve OTIO Timing

Takes a previous OTIO generation (via timeline_segments.json), re-transcribes
the voiceover audio with start-anchored word-level timing, maps the existing
clip assignments to the new segments, and rebuilds the OTIO + SRT.

Usage:
    python scripts/improve_otio_timing.py "E:\\Edit Job\\client\\project"
    python scripts/improve_otio_timing.py "E:\\Edit Job\\client\\project" --output-folder 20260124_052654
    python scripts/improve_otio_timing.py "E:\\Edit Job\\client\\project" --model large-v3
    python scripts/improve_otio_timing.py "E:\\Edit Job\\client\\project" --dry-run
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Standard script imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)

# Fix Windows console encoding
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from script_utils import print_ok, print_warn, print_error, print_info, print_header
from src.state import Match
from src.utils.project_metadata import get_card_title


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def parse_srt(path: str) -> List[dict]:
    """Parse SRT file into list of {start, end, text} dicts (seconds)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)"
    segments = []
    for m in re.finditer(pattern, content, re.DOTALL):
        segments.append({
            "idx": int(m.group(1)),
            "start": _ts_to_sec(m.group(2)),
            "end": _ts_to_sec(m.group(3)),
            "text": m.group(4).strip(),
        })
    return segments


def _ts_to_sec(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def fmt_time(s: float) -> str:
    m = int(s // 60)
    sec = s % 60
    return "%d:%06.3f" % (m, sec)


# ---------------------------------------------------------------------------
# Segment mapping: old clip assignments -> new SRT segments
# ---------------------------------------------------------------------------

def map_old_to_new_segments(
    old_segments: List[dict],
    new_srt_segments: List[dict],
) -> List[Optional[int]]:
    """Map each new SRT segment to the best-matching old segment index.

    Strategy:
    1. Build word timeline from old segments (word -> old segment index)
    2. For each new segment, find which old segment shares the most words
    3. If no word overlap found, fall back to temporal proximity

    Returns:
        List of old segment indices (one per new segment), or None for unmatched.
    """
    # Build word -> old_segment_index mapping with position tracking
    old_words_flat = []  # [(word, old_idx), ...]
    for oi, seg in enumerate(old_segments):
        words = seg.get("voiceover_text", seg.get("text", "")).lower().split()
        for w in words:
            old_words_flat.append((w, oi))

    mapping = []

    for ni, new_seg in enumerate(new_srt_segments):
        new_words = set(new_seg["text"].lower().split())

        # Score each old segment by word overlap
        scores = {}
        for ow, oi in old_words_flat:
            if ow in new_words:
                scores[oi] = scores.get(oi, 0) + 1

        if scores:
            best_oi = max(scores, key=scores.get)
            mapping.append(best_oi)
        else:
            # Fallback: nearest old segment by timeline position
            new_mid = (new_seg["start"] + new_seg["end"]) / 2
            best_oi = None
            best_dist = float("inf")
            for oi, seg in enumerate(old_segments):
                old_start = seg.get("start_frame", 0) / 30.0
                old_end = seg.get("end_frame", old_start * 30 + 30) / 30.0
                old_mid = (old_start + old_end) / 2
                dist = abs(old_mid - new_mid)
                if dist < best_dist:
                    best_dist = dist
                    best_oi = oi
            mapping.append(best_oi)

    return mapping


# ---------------------------------------------------------------------------
# Segment file discovery and resolution
# ---------------------------------------------------------------------------

def discover_segment_root(project_dir: Path, config=None) -> Optional[Path]:
    """Discover the segment files directory.

    Strategy chain:
    1. Scan existing OTIOs for V1 clip URLs → extract parent directory
    2. Fall back to config.downloaded_videos_dir if available
    """
    # Strategy 1: Extract from existing good OTIO (most reliable)
    segment_root = _discover_from_otio(project_dir)
    if segment_root:
        return segment_root

    # Strategy 2: Use config-resolved download directory
    if config:
        dvd = getattr(config, "downloaded_videos_dir", "")
        if dvd and Path(dvd).exists():
            # Verify it actually has segment files
            test_files = list(Path(dvd).glob("*_*_*.mp4"))[:1]
            if test_files:
                return Path(dvd)

    return None


def _discover_from_otio(project_dir: Path) -> Optional[Path]:
    """Extract segment root from an existing OTIO with populated V1 clips."""
    import opentimelineio as otio_lib

    output_dir = project_dir / "output"
    if not output_dir.exists():
        return None

    # Try each output folder (oldest first — more likely to have a good OTIO)
    candidates = sorted(
        (f for f in output_dir.iterdir() if f.is_dir()),
        key=lambda p: p.name,
    )

    for folder in candidates:
        otio_path = folder / "timeline_FULL.otio"
        if not otio_path.exists():
            continue
        try:
            tl = otio_lib.adapters.read_from_file(str(otio_path))
            for track in tl.tracks:
                if "V1" not in track.name:
                    continue
                for item in track:
                    if not isinstance(item, otio_lib.schema.Clip):
                        continue
                    mr = item.media_reference
                    if mr and hasattr(mr, "target_url") and mr.target_url:
                        url = mr.target_url
                        seg_dir = Path(url).parent
                        if seg_dir.exists():
                            return seg_dir
        except Exception:
            continue

    return None


def build_downloaded_segments(segment_dir: Path) -> List:
    """Build DownloadedSegment objects from segment files on disk.

    Parses filenames like `{video_id}_{start}_{end}.mp4` to reconstruct
    the segment metadata that create_timeline() needs for resolution.
    """
    from src.downloader.types import DownloadedSegment

    segments = []
    seg_pattern = re.compile(r'^(.+?)_(\d+)_(\d+)\.(mp4|webm|mkv)$')

    for f in segment_dir.iterdir():
        if not f.is_file():
            continue
        m = seg_pattern.match(f.name)
        if not m:
            continue
        video_id = m.group(1)
        start = float(m.group(2))
        end = float(m.group(3))
        segments.append(DownloadedSegment(
            file=str(f),
            video_id=video_id,
            original_start=start,
            original_end=end,
            file_duration=end - start,
            matches=[],
        ))

    return segments


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def find_latest_output_folder(project_dir: Path) -> Optional[Path]:
    output_dir = project_dir / "output"
    if not output_dir.exists():
        return None
    candidates = [
        f for f in output_dir.iterdir()
        if f.is_dir() and (f / "timeline_segments.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def find_voiceover_audio(project_dir: Path) -> Optional[Path]:
    """Find voiceover audio/video file in the project."""
    vo_dir = project_dir / "voiceover"
    search_dirs = [vo_dir, project_dir] if vo_dir.exists() else [project_dir]

    for d in search_dirs:
        for ext in ["*.mp3", "*.wav", "*.m4a", "*.mp4"]:
            candidates = list(d.glob(ext))
            # Prefer files named 'voiceover'
            for c in candidates:
                if "voiceover" in c.stem.lower():
                    return c
            if candidates:
                return candidates[0]
    return None


def improve_otio_timing(
    project_dir: Path,
    output_folder: Optional[str] = None,
    model_name: str = "large-v3",
    dry_run: bool = False,
) -> Optional[Path]:
    """Improve OTIO timing by re-transcribing with start-anchored word-level timing.

    Args:
        project_dir: Project directory path
        output_folder: Specific source output folder name
        model_name: Whisper model to use for re-transcription
        dry_run: If True, only re-transcribe and compare — don't rebuild OTIO

    Returns:
        Path to the new output directory, or None on failure.
    """
    print_header("OTIO Timing Improvement")
    print_info(f"Project: {project_dir}")

    # ── Step 1: Find source timeline_segments.json ──────────────────────
    if output_folder:
        source_dir = project_dir / "output" / output_folder
    else:
        source_dir = find_latest_output_folder(project_dir)

    if not source_dir or not source_dir.exists():
        print_error(f"No output folder found in {project_dir / 'output'}")
        return None

    segments_json_path = source_dir / "timeline_segments.json"
    if not segments_json_path.exists():
        print_error(f"timeline_segments.json not found in {source_dir}")
        return None

    with open(segments_json_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)

    old_segments = segments_data.get("segments", [])
    frame_rate = segments_data.get("frame_rate", 30.0)
    print_ok(f"Loaded {len(old_segments)} segments from {source_dir.name}")

    # ── Step 2: Load config + discover segment files (before transcription) ──
    from src.config import load_config as load_config_raw
    from src.cli.config_utils import make_paths_project_relative

    global_config_path = project_root / "config.yaml"
    config = load_config_raw(str(global_config_path), skip_final_validation=True)
    config = make_paths_project_relative(config, project_dir)
    config.otio_output_dir = str(project_dir / "output")

    segment_root = discover_segment_root(project_dir, config)
    downloaded_segments = []
    if segment_root:
        print_info(f"Segment root: {segment_root}")
        downloaded_segments = build_downloaded_segments(segment_root)
        print_ok(f"Indexed {len(downloaded_segments)} segment files")
    else:
        print_warn("No segment directory found — video clips will be unresolved")
        print_warn("Need at least one previous OTIO with populated V1 clips, "
                    "or a valid download.root_dir in config")

    # ── Step 3: Find and re-transcribe voiceover ────────────────────────
    vo_audio = find_voiceover_audio(project_dir)
    if not vo_audio:
        print_error("No voiceover audio file found")
        return None

    print_info(f"Voiceover: {vo_audio.name}")

    from src.transcription import transcribe_voiceover_media

    # Output new SRT alongside the audio
    new_srt_path = vo_audio.with_name("voiceover_improved.srt")

    print_info(f"Re-transcribing with {model_name} + start-anchored timing...")
    transcribe_voiceover_media(
        media_path=str(vo_audio),
        output_srt_path=str(new_srt_path),
        model_name=model_name,
        language="en",
        word_timestamps=True,
        vad_filter=True,
        force_contiguous_timing=True,
    )
    print_ok(f"Improved SRT: {new_srt_path.name}")

    # Parse the new SRT
    new_srt_segments = parse_srt(str(new_srt_path))
    print_info(f"New SRT: {len(new_srt_segments)} segments")

    # ── Step 4: Compare timing quality ──────────────────────────────────
    # Parse old SRT if it exists for comparison
    old_srt_path = vo_audio.with_suffix(".srt")
    if old_srt_path.exists() and old_srt_path != new_srt_path:
        old_srt_segs = parse_srt(str(old_srt_path))
        print()
        print_info("Timing comparison (contiguity check):")

        for label, segs in [("OLD", old_srt_segs), ("NEW", new_srt_segments)]:
            gaps = sum(
                1 for i in range(len(segs) - 1)
                if abs(segs[i + 1]["start"] - segs[i]["end"]) > 0.001
            )
            durs = [s["end"] - s["start"] for s in segs]
            short = sum(1 for d in durs if d < 1.0)
            print(
                f"  {label}: {len(segs)} segs, {gaps} gaps, "
                f"{short} under 1s, avg dur {sum(durs)/len(durs):.2f}s"
            )
        print()

    if dry_run:
        print_warn("Dry run — skipping OTIO rebuild")
        print_ok(f"Improved SRT saved to: {new_srt_path}")
        return None

    # ── Step 5: Map old clip assignments to new segments ────────────────
    print_info("Mapping clip assignments to new segments...")
    mapping = map_old_to_new_segments(old_segments, new_srt_segments)

    mapped = sum(1 for m in mapping if m is not None)
    print_ok(f"Mapped {mapped}/{len(new_srt_segments)} segments")

    # ── Step 6: Build match wrappers with NEW timing + OLD clips ────────
    from scripts.regenerate_otio import TimelineAwareMatchWrapper

    matches = []
    for ni, new_seg in enumerate(new_srt_segments):
        oi = mapping[ni]
        if oi is None or oi >= len(old_segments):
            # No clip assignment — create gap placeholder
            simple_match = Match(
                segment_index=ni, video_file="", video_start=0.0, video_end=0.0,
                confidence=0.0, strategy="gap", reason="unmatched", face_score=0.0,
            )
            wrapped = TimelineAwareMatchWrapper(simple_match, new_seg["start"], new_seg["end"])
            matches.append(wrapped)
            continue

        old_seg = old_segments[oi]
        v1_clip = old_seg.get("v1_clip", {})

        # Pass raw video ID + source times — create_timeline resolves via
        # downloaded_segments lookup (video_id -> segment file + adjusted offset)
        simple_match = Match(
            segment_index=ni,
            video_file=v1_clip.get("file", ""),
            video_start=v1_clip.get("source_start", 0.0),
            video_end=v1_clip.get("source_end", 0.0),
            confidence=v1_clip.get("confidence", 0.0),
            strategy=v1_clip.get("strategy", "restored"),
            reason=v1_clip.get("reason", ""),
            face_score=v1_clip.get("face_score", 0.5),
        )

        # Carry over alternatives and secondaries (raw video IDs)
        alternatives = []
        for alt in old_seg.get("alternatives", []):
            alternatives.append({
                "source_file": alt.get("file", ""),
                "start_time": alt.get("source_start", 0.0),
                "end_time": alt.get("source_end", 0.0),
                "confidence": alt.get("confidence", 0.5),
                "strategy": alt.get("strategy", ""),
            })

        secondary = []
        for sec in old_seg.get("secondary", []):
            secondary.append({
                "source_file": sec.get("file", ""),
                "start_time": sec.get("source_start", 0.0),
                "end_time": sec.get("source_end", 0.0),
                "confidence": sec.get("confidence", 0.5),
                "strategy": sec.get("strategy", ""),
            })

        # NEW timing + OLD clips
        wrapped = TimelineAwareMatchWrapper(
            simple_match, new_seg["start"], new_seg["end"],
            alternatives, secondary,
        )
        matches.append(wrapped)

    print_ok(f"Built {len(matches)} match wrappers")

    # ── Step 7: Create OTIO + exports ───────────────────────────────────
    from src.otio import (
        create_timeline,
        save_timeline_as_edl,
        generate_segment_map,
        generate_resolve_xml_with_bins,
    )
    from src.otio.export import save_timeline_split

    output_basename = get_card_title(project_dir) or "timeline"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_dir / "output" / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"Output: {output_dir}")

    # Create timeline with new timing + segment resolution
    timeline = create_timeline(
        matches=matches,
        config=config,
        voiceover_path=str(vo_audio),
        frame_rate=frame_rate,
        entity_images=None,
        entity_videos=None,
        downloaded_segments=downloaded_segments or None,
    )

    # Save OTIO (full + per-track split)
    otio_paths = save_timeline_split(
        timeline=timeline,
        output_path=str(output_dir / output_basename),
    )
    for path in otio_paths:
        print_ok(f"  + {Path(path).name}")

    # Segment map
    segment_map_path = generate_segment_map(
        matches=matches,
        output_path=str(output_dir / output_basename),
        frame_rate=frame_rate,
        source_srt=str(new_srt_path),
        timeline_start_tc=segments_data.get("timeline_start_tc", "01:00:00:00"),
    )
    print_ok(f"  + {Path(segment_map_path).name}")

    # EDL
    edl_path = save_timeline_as_edl(
        matches=matches,
        output_path=str(output_dir / f"{output_basename}.edl"),
        frame_rate=frame_rate,
    )
    if edl_path:
        print_ok(f"  + {Path(edl_path).name}")

    # XML
    xml_paths = generate_resolve_xml_with_bins(
        matches=matches,
        output_path=str(output_dir / output_basename),
        voiceover_path=str(vo_audio),
        frame_rate=frame_rate,
        config=config,
    )
    for path in (xml_paths if isinstance(xml_paths, list) else [xml_paths]):
        if path:
            print_ok(f"  + {Path(path).name}")

    # Copy improved SRT to output
    import shutil
    output_srt = output_dir / "voiceover_improved.srt"
    shutil.copy2(str(new_srt_path), str(output_srt))
    print_ok(f"  + voiceover_improved.srt")

    print()
    print_header("DONE")
    print_ok(f"Improved OTIO + SRT: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Improve OTIO timing with start-anchored re-transcription"
    )
    parser.add_argument("project", help="Project directory path")
    parser.add_argument(
        "--output-folder", default=None,
        help="Specific source output folder (default: latest)"
    )
    parser.add_argument(
        "--model", default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model (default: large-v3)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only re-transcribe and compare — don't rebuild OTIO"
    )

    args = parser.parse_args()
    project_dir = Path(args.project)

    if not project_dir.exists():
        print_error(f"Project directory not found: {project_dir}")
        sys.exit(1)

    result = improve_otio_timing(
        project_dir=project_dir,
        output_folder=args.output_folder,
        model_name=args.model,
        dry_run=args.dry_run,
    )

    if result is None and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
