#!/usr/bin/env python3
"""Detect chapters in an SRT file using BOTH listicle structure and LLM.

This is the standalone centerpiece: it runs listicle detection first, then
seeds the LLM topic prompt with those boundaries. The LLM is constrained to
respect the seeds as a strong prior while still having final say.

If listicle detection finds no usable seeds, the script falls back to the
in-pipeline `EnhancedChapterDetector` so behavior on non-listicle scripts is
unchanged.

Output JSON contains:
    - listicle_groups          (raw listicle detection)
    - llm_chapters             (chapters returned by the LLM, seed-aware)
    - unified_chapters         (merged result; ready for downstream consumers)
    - seeded                   (bool — True when listicle seeds were used)

Usage:
    python scripts/chapters/detect_chapters.py VOICEOVER.srt --output out.json
    python scripts/chapters/detect_chapters.py VOICEOVER.srt              # stdout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.chapter_detection.listicle_detector import detect_listicle_groups  # noqa: E402
from src.chapter_detection.listicle_seed import (  # noqa: E402
    format_listicle_seed_block,
    select_seed_groups,
)
from src.chapter_detection.chunking import create_chunks  # noqa: E402
from src.chapter_detection.models import ChapterCandidate  # noqa: E402
from src.chapter_detection.bridge import build_unified_chapters  # noqa: E402
from src.chapter_detection.detector import EnhancedChapterDetector  # noqa: E402
from src.chapter_detection.prompts import format_topic_prompt  # noqa: E402

from scripts.chapters._srt_io import load_segments  # noqa: E402

logger = logging.getLogger("chapters.detect")


# ---------------------------------------------------------------------------
# LLM client construction (mirrors EnhancedChapterDetector logic; kept
# local so this script doesn't depend on the full detector config plumbing).
# ---------------------------------------------------------------------------


def _build_llm_client(config):
    """Best-effort LLM client. Returns None if none available."""
    from src.llm_client import create_client
    try:
        if hasattr(config, "llm") and hasattr(config.llm, "ollama_host"):
            host = config.llm.ollama_host
            model = getattr(config.llm, "model", "gemma3:4b")
            return create_client("ollama", model=model, host=host)
    except Exception as e:
        logger.debug("Ollama init failed: %s", e)

    try:
        api_key = getattr(config, "gemini_api_key", None) or _env_or_none(
            "GEMINI_API_KEY"
        ) or _env_or_none("GOOGLE_API_KEY")
        if api_key:
            model = getattr(config, "gemini_model", "gemini-2.5-flash")
            return create_client("gemini", api_key=api_key, model=model)
    except Exception as e:
        logger.debug("Gemini init failed: %s", e)

    try:
        api_key = _env_or_none("ANTHROPIC_API_KEY")
        if api_key:
            return create_client("anthropic", api_key=api_key)
    except Exception as e:
        logger.debug("Anthropic init failed: %s", e)

    return None


def _env_or_none(name):
    import os
    return os.environ.get(name)


def _load_config(config_path: Path):
    from src.config.base import Config
    # skip_final_validation=True: standalone scripts may run with partial
    # configs (e.g., dummy test configs); the production pipeline enforces
    # API keys elsewhere.
    return Config.from_yaml(str(config_path), skip_final_validation=True)


# ---------------------------------------------------------------------------
# Seeded LLM call
# ---------------------------------------------------------------------------


def _conf_to_float(value):
    """Map LLM-returned confidence string to float. Tolerates floats too."""
    if isinstance(value, (int, float)):
        return float(value)
    mapping = {"high": 0.9, "medium": 0.7, "low": 0.5}
    return mapping.get(str(value).lower().strip(), 0.7)


def _dict_to_candidate(ch):
    return ChapterCandidate(
        chapter_id=ch.get("chapter_id", 0),
        start_segment_idx=ch.get("start_segment_idx", 0),
        end_segment_idx=ch.get("end_segment_idx", 0),
        title=ch.get("title", ""),
        topics=list(ch.get("topics", [])),
        confidence=_conf_to_float(ch.get("confidence", "medium")),
        detection_strategy="topic_seeded",
    )


def _call_seeded_llm(segments, llm_client, seed_groups, *, max_chars=6000):
    """Run the topic LLM call across chunks with the listicle seed appended."""
    from src.llm_client import LLMRequest, ResponseFormat

    chunks = create_chunks(segments, max_chars=max_chars)
    seed_block = format_listicle_seed_block(seed_groups)
    all_chapters = []

    for chunk in chunks:
        prompt = format_topic_prompt(
            indexed_text=chunk.text,
            total_segments=len(segments),
            min_segments=3,
            max_chapters=max(len(seed_groups) + 2, 6),
        ) + "\n\n" + seed_block

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            max_tokens=500,
            timeout=900,
            cache_key_prefix="chapter_detection_topic_seeded",
        )
        try:
            response = llm_client.generate(request)
        except Exception as e:
            logger.warning("Seeded LLM call failed: %s", e)
            continue
        if not response.parsed_data or not isinstance(response.parsed_data, list):
            continue

        for ch in response.parsed_data:
            if not isinstance(ch, dict):
                continue
            ch["start_segment_idx"] = max(
                0, min(int(ch.get("start_segment_idx", 0)) + chunk.start_segment_idx,
                       len(segments) - 1)
            )
            ch["end_segment_idx"] = max(
                ch["start_segment_idx"],
                min(int(ch.get("end_segment_idx", len(segments) - 1)) + chunk.start_segment_idx,
                    len(segments) - 1),
            )
            all_chapters.append(ch)

    return all_chapters


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _fallback_llm_chapters(segments, config_path: Path, content_type: str):
    """Use EnhancedChapterDetector when no listicle seed is available."""
    config = _load_config(config_path)
    detector = EnhancedChapterDetector(config)
    result = detector.detect_chapters_full(
        segments=segments, content_type=content_type
    )
    return list(result.chapters)


def detect(srt_path: Path, *, config_path: Path,
           min_confidence: float = 0.7,
           require_count_match: bool = False,
           content_type: str = "auto"):
    """Run the full listicle-seeded chapter detection on an SRT file.

    Returns a dict payload suitable for JSON serialization.
    """
    segments = load_segments(srt_path)

    groups = detect_listicle_groups(segments)
    seed_groups = select_seed_groups(
        groups,
        min_confidence=min_confidence,
        require_count_match=require_count_match,
    )

    seeded = bool(seed_groups)
    llm_chapter_dicts: list = []

    if seeded:
        config = _load_config(config_path)
        llm_client = _build_llm_client(config)
        if llm_client is not None:
            llm_chapter_dicts = _call_seeded_llm(
                segments, llm_client, seed_groups
            )
            logger.info("Seeded LLM returned %d chapter(s)", len(llm_chapter_dicts))
        else:
            logger.warning("No LLM client available; will rely on listicle groups only")

    if not llm_chapter_dicts:
        # No seeding possible (no groups, no LLM, or LLM call failed/empty):
        # fall back to the in-pipeline detector for chapter guesses.
        try:
            fallback = _fallback_llm_chapters(
                segments, config_path, content_type=content_type
            )
        except Exception as e:
            logger.warning("Fallback detector failed: %s", e)
            fallback = []
        llm_chapter_dicts = [ch.to_dict() for ch in fallback]
        seeded = False  # nothing was actually seeded

    llm_chapters = [_dict_to_candidate(ch) for ch in llm_chapter_dicts]
    unified = build_unified_chapters(
        llm_chapters, groups, merge_strategy="highest_confidence"
    )

    return {
        "voiceover_path": str(srt_path),
        "total_segments": len(segments),
        "seeded": seeded,
        "seed_group_count": len(seed_groups),
        "listicle_groups": [g.to_dict() for g in groups],
        "llm_chapters": [ch.to_dict() for ch in llm_chapters],
        "unified_chapters": [ch.to_dict() for ch in unified],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("srt_path", type=Path)
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Output JSON path. Default: stdout.")
    p.add_argument("--config", "-c", type=Path, default=Path("config.yaml"))
    p.add_argument("--min-confidence", type=float, default=0.7,
                   help="Minimum listicle group confidence to seed. Default 0.7.")
    p.add_argument("--require-count-match", action="store_true",
                   help="Drop seeds when detected count disagrees with header.")
    p.add_argument("--content-type", default="auto")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    payload = detect(
        args.srt_path,
        config_path=args.config,
        min_confidence=args.min_confidence,
        require_count_match=args.require_count_match,
        content_type=args.content_type,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        unified = payload["unified_chapters"]
        print(
            f"wrote {len(unified)} chapter(s) "
            f"(seeded={payload['seeded']}, "
            f"listicle_groups={len(payload['listicle_groups'])}) to {args.output}",
            file=sys.stderr,
        )
    else:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
