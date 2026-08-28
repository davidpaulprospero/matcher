#!/usr/bin/env python3
"""
distribute_ref_audio.py — Split an audio file (MP3/WAV/M4A) into N equal-length
segments and lay them across a NEW single-track audio OTIO. The source audio file
is read-only.

Usage:
    python scripts/distribute_ref_audio.py \\
        --mp3 "E:/Edit Job/Oscar/some/voiceover.mp3" \\
        --n 20 \\
        [--output "<new_otio>"] [--otio "<source_otio>"]

If --otio is passed, the new OTIO total length matches the source OTIO duration
(extra trailing silence if MP3 is shorter, no extension if MP3 is longer).
Without --otio, the new OTIO total length equals the MP3 length (segments placed
back-to-back, no gaps).

Default N is 20. Each segment is exactly (mp3_duration / N) seconds long.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("distribute-ref-audio")

RATE_DEFAULT = 30.0  # OTIO timeline rate (audio tracks ignore rate but a value is required)


def _to_windows_path(path: str) -> str:
    try:
        abs_path = str(Path(path).resolve())
    except OSError:
        abs_path = str(path)
    return abs_path.replace("\\", "/")


def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    return float(r.stdout.strip())


def split_mp3(mp3: Path, n: int, out_dir: Path) -> List[Path]:
    """Use ffmpeg segment muxer to split into N parts. Returns sorted paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{mp3.stem}_part_%03d.mp3"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp3),
        "-f", "segment",
        "-segment_time", str(_duration_for(mp3, n)),
        "-reset_timestamps", "1",
        "-c", "copy",
        str(pattern),
    ]
    logger.info("Running: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {r.stderr[-500:]}")
    files = sorted(out_dir.glob(f"{mp3.stem}_part_*.mp3"))
    if len(files) != n:
        # ffmpeg may emit N-1 parts if the last segment is below segment_time.
        # Pad with a copy of the last segment if needed (rare).
        logger.warning(
            f"Expected {n} segments, got {len(files)}; ffmpeg may have undercounted"
        )
    return files


def _duration_for(mp3: Path, n: int) -> float:
    return ffprobe_duration(mp3) / n


def build_timeline(
    segments: List[Path],
    mp3: Path,
    total_duration_sec: float,
    rate: float,
) -> otio.schema.Timeline:
    new_tl = otio.schema.Timeline(name=f"{mp3.stem}.ref_audio")
    new_tl.metadata["Resolve_OTIO"] = {"Resolve OTIO Meta Version": "1.0"}
    new_tl.global_start_time = RationalTime(0, rate)

    track = otio.schema.Track(name="A-Ref", kind=otio.schema.TrackKind.Audio)
    cursor_sec = 0.0
    for seg in segments:
        seg_dur = ffprobe_duration(seg)
        media_ref = otio.schema.ExternalReference(
            target_url=_to_windows_path(str(seg)),
            available_range=TimeRange(
                start_time=RationalTime(0, rate),
                duration=RationalTime(int(round(seg_dur * rate)), rate),
            ),
        )
        media_ref.name = seg.name
        clip = otio.schema.Clip(
            name=seg.name,
            media_reference=media_ref,
            source_range=TimeRange(
                start_time=RationalTime(0, rate),
                duration=RationalTime(int(round(seg_dur * rate)), rate),
            ),
        )
        clip.metadata["Resolve_OTIO"] = {}
        clip.metadata["is_audio_segment"] = True
        clip.metadata["source_mp3"] = str(mp3)
        clip.metadata["start_sec"] = cursor_sec
        track.append(clip)
        cursor_sec += seg_dur

    # Trailing silence if user asked to match a longer source OTIO.
    used_frames = int(round(cursor_sec * rate))
    target_frames = int(round(total_duration_sec * rate))
    if target_frames > used_frames:
        track.append(
            otio.schema.Gap(
                source_range=TimeRange(
                    start_time=RationalTime(0, rate),
                    duration=RationalTime(target_frames - used_frames, rate),
                )
            )
        )

    new_tl.tracks.append(track)
    return new_tl


def main() -> int:
    p = argparse.ArgumentParser(description="Distribute MP3 segments across a new audio OTIO.")
    p.add_argument("--mp3", required=True, help="Source audio file (mp3/wav/m4a).")
    p.add_argument("--otio", default=None, help="Optional source OTIO; if set, match its total duration.")
    p.add_argument("--output", default=None, help="Output OTIO path.")
    p.add_argument("--n", type=int, default=20, help="Number of segments (default 20).")
    p.add_argument("--frame-rate", type=float, default=RATE_DEFAULT, help="Timeline rate (default 30.0).")
    p.add_argument("--track-name", default="A-Ref", help="Audio track name (default A-Ref).")
    p.add_argument("--tmp-dir", default=None, help="Where to write segment files (default <output>_audio/).")
    p.add_argument("--keep-tmp", action="store_true", help="Keep segment files after build.")
    args = p.parse_args()

    mp3 = Path(args.mp3).resolve()
    if not mp3.exists():
        logger.error(f"MP3 not found: {mp3}")
        return 2

    if args.otio:
        src_otio = Path(args.otio).resolve()
        if not src_otio.exists():
            logger.error(f"Source OTIO not found: {src_otio}")
            return 2
    else:
        src_otio = None

    output_path = (
        Path(args.output).resolve()
        if args.output
        else mp3.with_name(mp3.stem + ".ref_audio.otio")
    )
    if output_path == mp3:
        logger.error("Refusing to overwrite source MP3 path; --output must differ")
        return 2

    tmp_dir = (
        Path(args.tmp_dir).resolve()
        if args.tmp_dir
        else output_path.with_name(output_path.stem + "_audio")
    )

    mp3_dur = ffprobe_duration(mp3)
    rate = float(args.frame_rate)
    target_dur = mp3_dur

    if src_otio:
        tl = otio.adapters.read_from_file(str(src_otio))
        src_dur = tl.duration().value / tl.duration().rate
        target_dur = src_dur
        logger.info(f"Source OTIO duration: {src_dur:.2f}s; matching that length")

    logger.info(f"MP3: {mp3}  |  Duration: {mp3_dur:.2f}s  |  N: {args.n}")
    logger.info(f"Output: {output_path}  |  Target total: {target_dur:.2f}s")

    segments = split_mp3(mp3, args.n, tmp_dir)
    if not segments:
        logger.error("No segments produced")
        return 3

    new_tl = build_timeline(segments, mp3, target_dur, rate)
    # Override track name if user asked
    if args.track_name != "A-Ref":
        new_tl.tracks[0].name = args.track_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(new_tl, str(output_path))

    summary = {
        "mp3": str(mp3),
        "mp3_duration_sec": mp3_dur,
        "n": args.n,
        "n_actual_segments": len(segments),
        "output_otio": str(output_path),
        "track_name": new_tl.tracks[0].name,
        "frame_rate": rate,
        "segments": [
            {
                "index": i,
                "path": str(p),
                "duration_sec": ffprobe_duration(p),
            }
            for i, p in enumerate(segments)
        ],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Wrote {output_path}")
    logger.info(f"Wrote {summary_path}")
    logger.info(
        f"Built A-Ref timeline: {len(segments)} segments, total {target_dur:.2f}s"
    )

    if not args.keep_tmp:
        logger.info(f"Segments in {tmp_dir} (use --keep-tmp to retain after future runs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())