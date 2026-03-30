#!/usr/bin/env python3
"""Fix OTIO timing for projects that used untrimmed voiceover timing.

Takes a project directory, ensures the trimmed voiceover + SRT exist,
remaps checkpoint segments to trimmed time, and regenerates the OTIO
with all tracks (V1-V12) aligned to the trimmed audio.

Usage:
    python scripts/fix_trimmed_otio.py "E:/Edit Job/Degold/channel/project_dir"
    python scripts/fix_trimmed_otio.py "/path/to/project" --dry-run
    python scripts/fix_trimmed_otio.py "/path/to/project" --skip-output
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Ensure stdout handles unicode on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_checkpoint(cp_path: Path) -> dict:
    """Load checkpoint (gzipped or plain JSON)."""
    with open(cp_path, "rb") as f:
        header = f.read(2)
    if header == b"\x1f\x8b":
        with gzip.open(cp_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)


def save_checkpoint(cp_path: Path, data: dict) -> None:
    """Save checkpoint as gzipped JSON."""
    with gzip.open(cp_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def get_audio_duration(path: str) -> Optional[float]:
    """Get audio duration via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def ensure_trimmed_voiceover(vo_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[dict]]:
    """Ensure voiceover_trimmed.mp3 and regions JSON exist.

    Returns (trimmed_mp3_path, regions_path, regions_data) or (None, None, None) on failure.
    """
    original = vo_dir / "voiceover.mp3"
    trimmed = vo_dir / "voiceover_trimmed.mp3"
    regions_path = vo_dir / "voiceover_trimmed_regions.json"

    if not original.exists():
        # Try wav
        original = vo_dir / "voiceover.wav"
        trimmed = vo_dir / "voiceover_trimmed.wav"
        regions_path = vo_dir / "voiceover_trimmed_regions.json"

    if not original.exists():
        logger.error("  No voiceover.mp3 or voiceover.wav found in %s", vo_dir)
        return None, None, None

    # Check if original is valid (not zero-filled)
    with open(original, "rb") as f:
        header = f.read(16)
    if not any(b != 0 for b in header):
        logger.error("  voiceover file is zero-filled (corrupted): %s", original)
        return None, None, None

    # If trimmed exists and is valid, load regions
    if trimmed.exists() and regions_path.exists():
        with open(trimmed, "rb") as f:
            header = f.read(4)
        if any(b != 0 for b in header):
            try:
                regions_data = json.loads(regions_path.read_text(encoding="utf-8"))
                logger.info("  Using existing trimmed voiceover: %s", trimmed.name)
                return trimmed, regions_path, regions_data
            except (json.JSONDecodeError, KeyError):
                logger.info("  Regions JSON corrupt, re-running silence removal")

    # Need to create trimmed version
    # Delete any zero-filled trimmed files
    for p in [trimmed, regions_path]:
        if p.exists():
            p.unlink()

    logger.info("  Running silence removal on %s...", original.name)
    try:
        from src.transcription.silence_removal import remove_voiceover_silence
        sr_result = remove_voiceover_silence(str(original))
        if not sr_result or not sr_result.was_trimmed:
            logger.warning("  Silence removal: no significant silence found, using original")
            return None, None, None
        logger.info("  Trimmed: %.1fs -> %.1fs (%.1f%% removed, %d regions)",
                     get_audio_duration(str(original)) or 0,
                     get_audio_duration(sr_result.trimmed_path) or 0,
                     100 * (1 - (get_audio_duration(sr_result.trimmed_path) or 1) /
                            (get_audio_duration(str(original)) or 1)),
                     len(sr_result.speech_regions_ms))
        regions_data = json.loads(regions_path.read_text(encoding="utf-8"))
        return Path(sr_result.trimmed_path), regions_path, regions_data
    except Exception as e:
        logger.error("  Silence removal failed: %s", e)
        return None, None, None


def build_time_mapper(speech_regions_ms: List, crossfade_ms: int):
    """Build a function that maps original-audio time to trimmed-audio time."""
    regions = []
    trimmed_cursor_ms = 0
    for i, (orig_start, orig_end) in enumerate(speech_regions_ms):
        region_len = orig_end - orig_start
        trimmed_start = trimmed_cursor_ms
        trimmed_end = trimmed_start + region_len
        regions.append((trimmed_start, trimmed_end, orig_start, orig_end))
        if i < len(speech_regions_ms) - 1:
            trimmed_cursor_ms = trimmed_end - crossfade_ms
        else:
            trimmed_cursor_ms = trimmed_end

    def map_to_trimmed(t_sec: float) -> float:
        t_ms = t_sec * 1000.0
        for trimmed_start, trimmed_end, orig_start, orig_end in regions:
            if t_ms <= orig_end:
                offset = max(0.0, t_ms - orig_start)
                return (trimmed_start + offset) / 1000.0
        if regions:
            return regions[-1][1] / 1000.0
        return t_sec

    return map_to_trimmed


def detect_timing_domain(segments: list, trimmed_duration: float, original_duration: float) -> str:
    """Detect whether checkpoint segments are in original or trimmed time."""
    if not segments:
        return "unknown"
    last_end = segments[-1].get("end", 0)
    diff_orig = abs(last_end - original_duration)
    diff_trimmed = abs(last_end - trimmed_duration)
    if diff_trimmed < 30:
        return "trimmed"
    if diff_orig < 30:
        return "original"
    # Heuristic: if last end is closer to trimmed duration
    return "trimmed" if diff_trimmed < diff_orig else "original"


def _scan_zero_filled_files(project_dir: Path) -> List[Tuple[str, int]]:
    """Scan stock videos, voiceover, and SRT files for zero-fill corruption."""
    bad = []
    scan_dirs = ["stock", "dvids", "voiceover"]
    scan_exts = {".mp4", ".mp3", ".wav", ".srt", ".json"}

    for subdir in scan_dirs:
        d = project_dir / subdir
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if Path(f).suffix.lower() not in scan_exts:
                    continue
                full = os.path.join(root, f)
                size = os.path.getsize(full)
                if size == 0:
                    continue
                with open(full, "rb") as fh:
                    header = fh.read(min(64, size))
                if not any(b != 0 for b in header):
                    bad.append((os.path.relpath(full, project_dir), size))
    return bad


def fix_trimmed_otio(project_dir: Path, dry_run: bool = False, skip_output: bool = False) -> bool:
    """Main fix function."""
    print("=" * 60)
    print("  Fix Trimmed OTIO Timing")
    print("=" * 60)
    print(f"  Project: {project_dir.name}")

    # --- Check prerequisites ---
    vo_dir = project_dir / "voiceover"
    if not vo_dir.exists():
        logger.error("  No voiceover/ directory found")
        return False

    cp_path = project_dir / "checkpoint.json"
    if not cp_path.exists():
        logger.error("  No checkpoint.json found")
        return False

    # --- Step 1: Ensure trimmed voiceover ---
    print("\n[1/4] Checking trimmed voiceover...")
    trimmed_path, regions_path, regions_data = ensure_trimmed_voiceover(vo_dir)
    if trimmed_path is None:
        logger.error("  Cannot proceed without trimmed voiceover")
        return False

    trimmed_duration = get_audio_duration(str(trimmed_path))
    original_path = vo_dir / "voiceover.mp3"
    if not original_path.exists():
        original_path = vo_dir / "voiceover.wav"
    original_duration = get_audio_duration(str(original_path)) if original_path.exists() else None

    print(f"  Trimmed: {trimmed_duration:.1f}s ({trimmed_duration / 60:.1f}min)")
    if original_duration:
        print(f"  Original: {original_duration:.1f}s ({original_duration / 60:.1f}min)")

    # --- Step 2: Load and analyze checkpoint ---
    print("\n[2/4] Analyzing checkpoint...")
    data = load_checkpoint(cp_path)
    segments = data.get("analyze", {}).get("segments", [])
    if not segments:
        logger.error("  No segments in checkpoint")
        return False

    domain = detect_timing_domain(
        segments, trimmed_duration or 0, original_duration or 0
    )
    last_end = segments[-1].get("end", 0)
    print(f"  Segments: {len(segments)}, last end: {last_end:.2f}s")
    print(f"  Timing domain: {domain}")

    if domain == "trimmed":
        print("  Segments already in trimmed time - no remapping needed")
    elif domain == "original":
        speech_regions = [tuple(r) for r in regions_data["speech_regions_ms"]]
        crossfade_ms = regions_data.get("crossfade_ms", 50)
        map_fn = build_time_mapper(speech_regions, crossfade_ms)

        print(f"  Remapping {len(segments)} segments from original -> trimmed time...")
        for seg in segments:
            seg["start"] = round(map_fn(seg["start"]), 3)
            seg["end"] = round(map_fn(seg["end"]), 3)
            seg["duration"] = round(seg["end"] - seg["start"], 3)

        new_last = segments[-1]["end"]
        print(f"  Remapped: last end {last_end:.2f}s -> {new_last:.2f}s")
    else:
        logger.warning("  Could not determine timing domain, proceeding anyway")

    # Detect last-segment bloat: Whisper sometimes extends the final segment
    # to the full file duration, creating an absurdly long segment for a
    # short sentence. Cap it to a reasonable duration.
    if len(segments) >= 2:
        last = segments[-1]
        prev = segments[-2]
        last_dur = last["end"] - last["start"]
        # Average of last 20 segments (excluding the final one)
        recent = segments[max(0, len(segments) - 21):-1]
        avg_dur = sum(s["end"] - s["start"] for s in recent) / len(recent) if recent else 5.0
        if last_dur > avg_dur * 10 and last_dur > 30:
            # Bloated — cap to a reasonable duration based on text length
            words = len(last.get("text", "").split())
            estimated_dur = max(words * 0.4, avg_dur)  # ~0.4s per word minimum
            new_end = round(last["start"] + estimated_dur, 3)
            print(f"  WARNING: Last segment bloated ({last_dur:.1f}s for "
                  f"{words} words). Capped: {last['end']:.1f}s -> {new_end:.1f}s")
            last["end"] = new_end
            last["duration"] = round(new_end - last["start"], 3)

    # Scan for zero-filled media files (corrupted USB transfers)
    zero_filled = _scan_zero_filled_files(project_dir)
    if zero_filled:
        print(f"\n  WARNING: {len(zero_filled)} zero-filled (corrupted) media files:")
        for rel, size in zero_filled[:10]:
            print(f"    {rel} ({size:,} bytes)")
        if len(zero_filled) > 10:
            print(f"    ... and {len(zero_filled) - 10} more")
        print("  These files will cause DaVinci to crash on import.")

    # --- Step 3: Fix voiceover path and write trimmed SRT ---
    print("\n[3/4] Updating checkpoint and SRT...")

    # Fix any Linux paths
    raw = json.dumps(data)
    # Common Linux prefix patterns
    for prefix in ["/home/hpmint/Desktop/matcher/projects/"]:
        if prefix in raw:
            # Find what comes after the prefix to build the replacement
            # e.g. /home/hpmint/.../channel/project -> E:/Edit Job/.../channel/project
            raw = raw.replace(
                prefix + "/".join(str(project_dir).replace("\\", "/").split("/")[-2:]),
                str(project_dir).replace("\\", "/"),
            )
    data = json.loads(raw)

    # Set voiceover path to trimmed
    data["voiceover_path"] = str(trimmed_path).replace("\\", "/")

    # Store trimmed segments for new checkpoint format
    data["analyze"]["trimmed_segments"] = [dict(s) for s in segments]

    # Write trimmed SRT
    trimmed_srt = trimmed_path.with_suffix(".srt")
    from src.transcription.utils import write_srt
    seg_dicts = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments]
    write_srt(seg_dicts, str(trimmed_srt))
    print(f"  Wrote {trimmed_srt.name}: {len(segments)} segments")

    if dry_run:
        print("\n  [DRY RUN] Would save checkpoint and regenerate OTIO")
        print(f"  Segments: {len(segments)}, last end: {segments[-1]['end']:.2f}s")
        print(f"  Voiceover: {trimmed_path.name} ({trimmed_duration:.1f}s)")
        return True

    # Save checkpoint
    save_checkpoint(cp_path, data)
    print(f"  Checkpoint saved")

    if skip_output:
        print("\n  [SKIP] Output regeneration skipped (--skip-output)")
        print("  Run: python main.py --output-only --project \"%s\" --non-interactive" % project_dir)
        return True

    # --- Step 4: Regenerate OTIO ---
    print("\n[4/4] Regenerating OTIO (--output-only)...")
    result = subprocess.run(
        [sys.executable, "main.py", "--output-only",
         "--project", str(project_dir), "--non-interactive"],
        cwd=str(project_root),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )

    # Parse output for key info
    for line in result.stdout.split("\n"):
        if any(k in line for k in ["Pipeline", "Voiceover duration", "trailing gap",
                                     "clips", "EDL", "completed", "failed"]):
            print(f"  {line.strip()}")

    if result.returncode != 0:
        logger.error("  Pipeline failed (exit code %d)", result.returncode)
        # Show last few error lines
        for line in result.stdout.split("\n")[-10:]:
            if line.strip():
                print(f"    {line.strip()}")
        return False

    # Find the new output folder
    output_dir = project_dir / "output"
    latest = max(
        (d for d in output_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        default=None,
    )
    if latest:
        print(f"\n  Output: {latest}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix OTIO timing to use trimmed voiceover")
    parser.add_argument("project_dir", help="Project directory path")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--skip-output", action="store_true",
                        help="Fix checkpoint/SRT only, skip OTIO regeneration")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not project_dir.exists():
        # Try glob for special characters
        import glob
        matches = glob.glob(args.project_dir)
        if matches:
            project_dir = Path(matches[0]).resolve()
        else:
            logger.error("Project directory not found: %s", args.project_dir)
            sys.exit(1)

    success = fix_trimmed_otio(project_dir, dry_run=args.dry_run, skip_output=args.skip_output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
