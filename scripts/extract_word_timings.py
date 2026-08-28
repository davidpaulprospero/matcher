"""
Generate word-level timestamp sidecar for lower-thirds word-onset anchoring.

Re-transcribes a voiceover audio file with word_timestamps=True and writes a
<voiceover>.words.json sidecar that scripts/lower_thirds.py reads to anchor
each entity's start_time to the actual spoken-word onset instead of the SRT
segment start.

Usage:
    python scripts/extract_word_timings.py <project_or_srt> [options]

The SRT file is NOT modified: force_contiguous_timing=False preserves the
existing segment boundaries (only the sidecar is added). Re-running the
lower-thirds render then anchors start_time to the word within each segment
while keeping end_time at seg.end_time.

Output sidecar location: <srt_path>.words.json (sibling of the SRT file).
"""

import argparse
import logging
import sys
from pathlib import Path

# Repo root on sys.path so we can import src.* and scripts.* modules.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.lower_thirds import find_srt_path  # noqa: E402
from src.transcription import transcribe_voiceover_media  # noqa: E402

logger = logging.getLogger("extract_word_timings")


def _resolve_audio_for_srt(srt_path: Path) -> Path:
    """Find the matching audio file for an SRT. Prefer *_trimmed.mp3.

    Mirrors the SRT discovery logic so re-transcription targets the audio
    that the SRT was actually derived from.
    """
    candidates = []
    stem = srt_path.stem
    parent = srt_path.parent
    for ext in (".mp3", ".wav", ".m4a", ".aac"):
        for name in (f"{stem}{ext}", f"{stem}_trimmed{ext}"):
            p = parent / name
            if p.exists():
                candidates.append(p)
    # Prefer *_trimmed.mp3 over plain mp3 (matches SRT discovery preference).
    candidates.sort(key=lambda p: (0 if "_trimmed" in p.stem else 1, p.suffix != ".mp3"))
    if not candidates:
        raise FileNotFoundError(
            f"No audio file found matching {srt_path} "
            f"(looked for .mp3/.wav/.m4a/.aac siblings)"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate .words.json sidecar for lower-thirds word-onset anchoring."
    )
    parser.add_argument(
        "project_or_srt",
        help="Project directory (auto-discovers SRT) or path to a specific .srt file.",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="faster-whisper model name (default: large-v3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if the .words.json sidecar already exists.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering (default: VAD on, matches improve_otio_timing.py).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    target = Path(args.project_or_srt)
    if target.is_file() and target.suffix.lower() == ".srt":
        srt_path = target
        project_dir = target.parent
    else:
        project_dir = target
        srt_path = find_srt_path(project_dir)
        if srt_path is None:
            logger.error(f"No SRT found in {project_dir}")
            return 1

    sidecar_path = srt_path.with_suffix(".words.json")
    if sidecar_path.exists() and not args.force:
        logger.info(
            f"Sidecar already exists: {sidecar_path} "
            f"(use --force to regenerate)"
        )
        return 0

    try:
        audio_path = _resolve_audio_for_srt(srt_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"SRT:    {srt_path}")
    logger.info(f"Audio:  {audio_path}")
    logger.info(f"Sidecar: {sidecar_path}")
    logger.info(
        f"Re-transcribing with model={args.model}, "
        f"word_timestamps=True, force_contiguous_timing=False "
        f"(preserves SRT segment boundaries)..."
    )
    logger.info(
        "This is a one-time operation (~2-3 min for a 20MB audio on large-v3). "
        "Subsequent lower-thirds runs reuse the sidecar."
    )

    try:
        transcribe_voiceover_media(
            media_path=str(audio_path),
            output_srt_path=str(srt_path),  # re-writes SRT idempotently
            model_name=args.model,
            word_timestamps=True,
            vad_filter=not args.no_vad,
            force_contiguous_timing=False,  # CRITICAL: keep original seg.end_time
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 1

    if not sidecar_path.exists():
        logger.error(
            f"Transcription completed but sidecar was not written at {sidecar_path}"
        )
        return 1

    import json
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            word_data = json.load(f)
        seg_count = len(word_data)
        word_count = sum(len(s.get("words") or []) for s in word_data)
    except (OSError, json.JSONDecodeError):
        seg_count = word_count = "?"

    logger.info(f"Done. {seg_count} segments, {word_count} words written to {sidecar_path}")
    logger.info(
        "Now run: python scripts/lower_thirds.py <project> "
        "--template-by-type \"person=classic,date=boxed,info=modern\""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())