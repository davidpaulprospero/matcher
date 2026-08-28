#!/usr/bin/env python3
"""Detect listicle structure in an SRT file.

Reads an SRT, runs the regex-driven listicle detector, and writes the
detected groups as JSON. Pure regex/keyword extraction — no LLM cost.

Usage:
    python scripts/chapters/detect_listicle.py VOICEOVER.srt --output VOICEOVER_listicle.json
    python scripts/chapters/detect_listicle.py VOICEOVER.srt       # JSON to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.chapter_detection.listicle_detector import detect_listicle_groups  # noqa: E402
from src.chapter_detection.models import ListicleGroup  # noqa: E402

from scripts.chapters._srt_io import load_segments  # noqa: E402


def detect(segments, *, max_chars_offset: int = 50):
    """Run listicle detection on segment dicts. Returns List[ListicleGroup]."""
    return detect_listicle_groups(segments, max_chars_offset=max_chars_offset)


def _build_payload(srt_path: Path, segments, groups) -> dict:
    expected = groups[0].expected_count if groups else None
    return {
        "voiceover_path": str(srt_path),
        "total_segments": len(segments),
        "expected_count": expected,
        "listicle_groups": [g.to_dict() for g in groups],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("srt_path", type=Path, help="Input SRT file")
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON path. Default: stdout.",
    )
    p.add_argument(
        "--max-chars-offset",
        type=int,
        default=50,
        help="Mid-segment marker search offset (chars). Default: 50.",
    )
    args = p.parse_args()

    segments = load_segments(args.srt_path)
    if not segments:
        print(f"warning: no segments in {args.srt_path}", file=sys.stderr)

    groups = detect(segments, max_chars_offset=args.max_chars_offset)
    payload = _build_payload(args.srt_path, segments, groups)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"wrote {len(groups)} listicle group(s) to {args.output}",
            file=sys.stderr,
        )
    else:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
