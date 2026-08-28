#!/usr/bin/env python3
"""Detect chapters in an SRT file using the LLM (no listicle seeding).

Thin standalone wrapper around `EnhancedChapterDetector`. Use this to
get raw LLM chapter output without any listicle intervention.

Usage:
    python scripts/chapters/detect_chapters_llm.py VOICEOVER.srt --output out.json
    python scripts/chapters/detect_chapters_llm.py VOICEOVER.srt --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.chapter_detection.detector import EnhancedChapterDetector  # noqa: E402

from scripts.chapters._srt_io import load_segments  # noqa: E402


def _load_config(config_path: Path):
    """Load a Config object without touching the module-level global.

    Skips final config validation: standalone scripts may run with partial
    configs (e.g., missing API keys for providers we don't use).
    """
    from src.config.base import Config
    return Config.from_yaml(str(config_path), skip_final_validation=True)


def detect_llm(segments, *, config_path: Path, content_type: str = "auto", overall_topic: str = None):
    config = _load_config(config_path)
    detector = EnhancedChapterDetector(config)
    result = detector.detect_chapters_full(
        segments=segments,
        content_type=content_type,
        overall_topic=overall_topic,
    )
    return result


def _build_payload(srt_path: Path, segments, result) -> dict:
    return {
        "voiceover_path": str(srt_path),
        "total_segments": len(segments),
        "content_type": result.content_type,
        "detection_passes_run": list(result.detection_passes_run),
        "llm_chapters": [ch.to_dict() for ch in result.chapters],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("srt_path", type=Path, help="Input SRT file")
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSON path. Default: stdout.",
    )
    p.add_argument(
        "--config", "-c", type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml. Default: ./config.yaml",
    )
    p.add_argument(
        "--content-type", default="auto",
        choices=["auto", "topic", "location", "narrative", "travel", "educational", "documentary"],
        help="Content type hint. Default: auto-detect.",
    )
    p.add_argument(
        "--overall-topic", default=None,
        help="Optional topic context for the LLM.",
    )
    args = p.parse_args()

    segments = load_segments(args.srt_path)
    if not segments:
        print(f"warning: no segments in {args.srt_path}", file=sys.stderr)

    result = detect_llm(
        segments,
        config_path=args.config,
        content_type=args.content_type,
        overall_topic=args.overall_topic,
    )
    payload = _build_payload(args.srt_path, segments, result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"wrote {len(result.chapters)} chapter(s) to {args.output}",
            file=sys.stderr,
        )
    else:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
