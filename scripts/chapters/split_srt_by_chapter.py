#!/usr/bin/env python3
"""Split an SRT file into one SRT per chapter.

Reads an SRT and a chapter JSON file (output of `detect_chapters.py` or
`detect_chapters_llm.py`). For each chapter it writes a normalized SRT that
starts at 00:00:00,000 with indices renumbered from 1.

Usage:
    python scripts/chapters/split_srt_by_chapter.py VOICEOVER.srt chapters.json --output-dir chapters/
    python scripts/chapters/split_srt_by_chapter.py VOICEOVER.srt chapters.json      # sibling dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.chapters._srt_io import (  # noqa: E402
    load_segments,
    renumber_and_shift,
    slugify,
    write_srt,
)


def _load_chapters(chapters_json: Path):
    """Extract a chapter list from a chapter JSON.

    Accepts either the `unified_chapters`/`llm_chapters`/`chapters` formats
    produced by the other scripts in this package.
    """
    payload = json.loads(chapters_json.read_text(encoding="utf-8"))
    for key in ("unified_chapters", "llm_chapters", "chapters"):
        if key in payload and payload[key]:
            return payload[key], key
    return [], None


def _normalize_slices(segments, chapter):
    """Slice `segments` down to a single chapter's range and shift timestamps."""
    start = int(chapter.get("start_segment_idx", 0))
    end = int(chapter.get("end_segment_idx", start))
    start = max(0, min(start, len(segments) - 1))
    end = max(start, min(end, len(segments) - 1))
    chunk = [s for s in segments if start <= int(s["index"]) <= end]
    if not chunk:
        return []
    return renumber_and_shift(chunk, shift_seconds=float(chunk[0]["start"]))


def split_by_chapter(srt_path: Path, chapters_json: Path, output_dir: Path):
    """Split `srt_path` into per-chapter SRTs in `output_dir`. Returns list of paths."""
    segments = load_segments(srt_path)
    chapters, source_key = _load_chapters(chapters_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = srt_path.stem
    paths = []

    for i, ch in enumerate(chapters, 1):
        title = ch.get("title") or f"chapter-{i}"
        chunk = _normalize_slices(segments, ch)
        if not chunk:
            print(
                f"warning: chapter {i} ({title!r}) maps to no segments; skipping",
                file=sys.stderr,
            )
            continue

        filename = f"{stem}__ch{i:02d}_{slugify(title)}.srt"
        out = output_dir / filename
        write_srt(chunk, out)
        paths.append(out)

    print(
        f"split {srt_path.name} into {len(paths)} chapter SRT(s) "
        f"({source_key or 'no_chapters_found'}) at {output_dir}",
        file=sys.stderr,
    )
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("srt_path", type=Path)
    p.add_argument("chapters_json", type=Path)
    p.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Where to write per-chapter SRTs. Default: <srt_dir>/<srt_stem>_chapters/",
    )
    args = p.parse_args()

    output_dir = args.output_dir or (
        args.srt_path.parent / f"{args.srt_path.stem}_chapters"
    )
    split_by_chapter(args.srt_path, args.chapters_json, output_dir)


if __name__ == "__main__":
    main()
