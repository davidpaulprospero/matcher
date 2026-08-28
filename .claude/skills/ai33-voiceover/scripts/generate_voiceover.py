#!/usr/bin/env python3
"""
Generate a voiceover MP3 (and matching SRT scaffold) via AI33 TTS, ready for:

    python main.py --voiceover <output_dir>/voiceover.srt

Inputs (pick ONE):
  --text        Inline script text
  --input-file  Path to a .txt / .md / .srt script
  --input-srt   Path to an existing SRT (re-renders audio for it)

Outputs (under --output-dir):
  voiceover.mp3      The MP3 produced by AI33
  voiceover.srt      SRT with the script as text + placeholder timings (the
                     pipeline will re-transcribe the MP3 to get real timings)
  voiceover.txt      Plain-text copy of the script

Optional:
  --voice-id       Override AI33_VOICE_ID
  --speed/stability/similarity/style/no-speaker-boost  Override voice_settings
  --with-transcript    Ask AI33 to also return a transcript URL
  --no-poll            Submit only, exit immediately (returns task_id)

Exit codes:
  0  success
  1  API failure
  2  bad input
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

THIS_DIR = Path(__file__).resolve().parent
CLIENT_PATH = THIS_DIR / "ai33_client.py"

# Add the skill's scripts dir to sys.path so `import ai33_client` works
sys.path.insert(0, str(THIS_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script-to-segment helpers


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_existing_srt(text: str) -> str:
    """If input looks like an SRT, return the concatenated dialogue text."""
    # SRT lines look like:  00:00:01,000 --> 00:00:04,000
    srt_ts = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*$")
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.isdigit() or srt_ts.match(line):
            continue
        out.append(line)
    return " ".join(out).strip()


def _chunk_sentences(text: str, max_chars: int = 240) -> List[str]:
    """Split text into sentence-ish chunks suitable for SRT cues."""
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        candidate = (buf + " " + s).strip() if buf else s.strip()
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # Hard-split very long sentences
            while len(s) > max_chars:
                chunks.append(s[:max_chars].strip())
                s = s[max_chars:].strip()
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def _format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h, rem = divmod(seconds, 3600)
    m, rem = divmod(rem, 60)
    s, ms = divmod(rem, 1)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(ms * 1000):03d}"


def _build_placeholder_srt(text: str) -> str:
    """
    Build an SRT where each cue is text only with placeholder timings
    (00:00:00,000 --> 00:00:00,000). The pipeline re-transcribes the MP3
    to recover real timings, so these are just informational.
    """
    cues = _chunk_sentences(text)
    lines: List[str] = []
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append("00:00:00,000 --> 00:00:00,000")
        lines.append(cue)
        lines.append("")
    if not lines:
        # Fall back to a single cue with the whole text
        lines = ["1", "00:00:00,000 --> 00:00:00,000", text.strip(), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate voiceover MP3 + SRT via AI33 TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Inline script text")
    src.add_argument("--input-file", help="Path to .txt/.md script")
    src.add_argument("--input-srt", help="Path to existing .srt (re-render audio)")

    parser.add_argument("--output-dir", required=True, help="Where to write voiceover.*")
    parser.add_argument("--filename", default="voiceover", help="Output stem (default: voiceover)")
    parser.add_argument("--voice-id", help="Override AI33_VOICE_ID env var")
    parser.add_argument("--speed", type=float)
    parser.add_argument("--stability", type=float)
    parser.add_argument("--similarity", type=float)
    parser.add_argument("--style", type=float)
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--format", default="mp3_44100_128")
    parser.add_argument("--with-transcript", action="store_true",
                        help="Ask AI33 to also produce a transcript")
    parser.add_argument("--no-poll", action="store_true",
                        help="Submit task only, do not wait for completion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build inputs and print plan, do not call AI33")
    parser.add_argument("--dotenv",
                        default=str(Path(__file__).resolve().parents[4] / ".env"),
                        help="Path to .env (default: repo root)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--keep-srt-text", action="store_true",
                        help="Do not strip SRT timestamps when re-rendering --input-srt")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv(args.dotenv, override=False)
    except ImportError:
        logger.warning("python-dotenv not installed; relying on already-set env vars")

    # ----- resolve script text
    if args.text is not None:
        raw_text = args.text
        input_was_srt = False
        source_label = "<inline --text>"
    elif args.input_srt:
        raw = _read_text(Path(args.input_srt))
        if args.keep_srt_text:
            raw_text = raw
        else:
            raw_text = _strip_existing_srt(raw)
        input_was_srt = True
        source_label = str(args.input_srt)
    else:
        raw = _read_text(Path(args.input_file))
        raw_text = _strip_existing_srt(raw) if Path(args.input_file).suffix.lower() == ".srt" else raw
        input_was_srt = Path(args.input_file).suffix.lower() == ".srt"
        source_label = str(args.input_file)

    raw_text = raw_text.strip()
    if not raw_text:
        logger.error("No text content extracted from %s", source_label)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.filename
    mp3_path = out_dir / f"{stem}.mp3"
    srt_path = out_dir / f"{stem}.srt"
    txt_path = out_dir / f"{stem}.txt"

    logger.info("Script: %d chars from %s", len(raw_text), source_label)
    logger.info("Outputs: %s / %s / %s", mp3_path, srt_path, txt_path)

    if args.dry_run:
        print(f"DRY RUN - would submit to AI33:")
        print(f"  text_chars:  {len(raw_text)}")
        print(f"  text_preview: {raw_text[:200]!r}{'...' if len(raw_text) > 200 else ''}")
        print(f"  srt_cues:    {len(_chunk_sentences(raw_text))}")
        print(f"  mp3_path:    {mp3_path}")
        print(f"  srt_path:    {srt_path}")
        print(f"  txt_path:    {txt_path}")
        print(f"  input_was_srt: {input_was_srt}")
        return 0

    # ----- always write txt + placeholder srt before the API call
    txt_path.write_text(raw_text + "\n", encoding="utf-8")
    srt_path.write_text(_build_placeholder_srt(raw_text) + "\n", encoding="utf-8")

    # ----- build voice_settings overrides
    overrides = {}
    if args.speed is not None:
        overrides["speed"] = args.speed
    if args.stability is not None:
        overrides["stability"] = args.stability
    if args.similarity is not None:
        overrides["similarity_boost"] = args.similarity
    if args.style is not None:
        overrides["style"] = args.style
    if args.no_speaker_boost:
        overrides["use_speaker_boost"] = False

    # ----- call AI33
    try:
        import ai33_client  # type: ignore

        client = ai33_client.AI33TextToSpeech(
            voice_id=args.voice_id,
            voice_settings=overrides or None,
        )
    except ai33_client.AI33Error as exc:
        logger.error("AI33 init failed: %s", exc)
        return 2

    try:
        result = client.generate_voiceover(
            text=raw_text,
            output_path=str(mp3_path),
            output_format=args.format,
            with_transcript=args.with_transcript or None,
            wait_for_completion=not args.no_poll,
        )
    except ai33_client.AI33Error as exc:
        logger.error("AI33 generation failed: %s", exc)
        return 1

    print(f"\nAI33 status:    {result.status}")
    print(f"Task ID:        {result.task_id}")
    print(f"Audio URL:      {result.audio_url}")
    print(f"Transcript URL: {result.transcript_url}")
    print(f"Duration:       {result.duration}")
    print(f"Local MP3:      {result.local_path}")
    print(f"SRT (placeholder timings, will be re-transcribed): {srt_path}")
    print(f"TXT:            {txt_path}")
    if result.local_path:
        print(f"\nNext: python main.py --voiceover \"{srt_path}\"")
    return 0 if result.local_path else 1


if __name__ == "__main__":
    raise SystemExit(main())