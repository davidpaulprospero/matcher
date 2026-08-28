#!/usr/bin/env python3
"""
shot_otio.py — Transcribe a video to SRT, then generate one AI still image per
segment using Google Gemini Flash (gemini-2.5-flash-image), and produce an OTIO
timeline with those images on the V12 track.

Usage:
    python scripts/shot_otio.py "<video_or_project>" [--size WxH] [--budget USD] [--output <path>"]

Examples:
    python scripts/shot_otio.py "E:\Edit Job\Stu\Shorts\1\Stu tiktok1 .mov"
    python scripts/shot_otio.py "E:\Edit Job\Stu\Shorts\1" --size 1408x768 --budget 3.0
    python scripts/shot_otio.py "E:\Edit Job\Stu\Shorts\1" --dry-run --verbose
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

from src.generated_images import build_generated_image_batches, GeneratedImageService
from src.generated_images.prompt_builder import GeneratedImagePromptBuilder
from src.config.sections.generated_images import GeneratedImagesConfig, IMAGEN_SUPPORTED_SIZES, GeneratedImageSizeConfig
from src.state import VoiceoverSegment, GeneratedImageResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RATE = 30.0
PROMPT_RECOMMENDED_MAX_WORDS = 75


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _seconds_to_tc(seconds: float, rate: float = RATE) -> str:
    """Convert seconds to HH:MM:SS:FF timecode string."""
    total_frames = int(seconds * rate)
    ff = total_frames % int(rate)
    ss = (total_frames // int(rate)) % 60
    mm = (total_frames // (int(rate) * 60)) % 60
    hh = total_frames // (int(rate) * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _aspect_label(w: int, h: int) -> str:
    """Return a human-readable aspect ratio label (e.g. '16:9 1408x768')."""
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    g = gcd(w, h)
    rw, rh = w // g, h // g
    return f"{rw}:{rh} {w}x{h}"


def _is_bare_prompt(prompt: str, raw_text: str) -> bool:
    """Return True if the prompt is just bare raw text with minimal framing."""
    stripped = prompt.strip()
    bare_marker = f"Scene: {raw_text.strip()}"
    return stripped == bare_marker or stripped == f"Scene: {raw_text.strip()}." or stripped == bare_marker.rstrip(".")


def _word_count(text: str) -> int:
    return len(text.split())


def _shot_id_from_batch_id(batch_id: str) -> str:
    """Convert internal batch_id (e.g. 'generated_000') to display form ('shot_000')."""
    if batch_id.startswith("generated_"):
        return "shot_" + batch_id.removeprefix("generated_")
    return batch_id


# ─────────────────────────────────────────────────────────────────────────────
# Video discovery
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}


def find_video(project_dir: Path) -> Optional[Path]:
    """Find the first video file in a project directory."""
    for ext in VIDEO_EXTENSIONS:
        matches = sorted(project_dir.glob(f"*{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Transcription (faster-whisper)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_video(video_path: Path) -> List[VoiceoverSegment]:
    """Transcribe a video file to VoiceoverSegments using faster-whisper."""
    from src.transcription.whisper_client import WhisperClient

    logger.info(f"Transcribing {video_path.name} ...")
    client = WhisperClient(model_name='medium', model_version='v3-turbo')
    segments_raw, info = client.transcribe(
        str(video_path),
        language=None,
        word_timestamps=False,
    )

    # segments may be dicts or objects — normalize to dict
    seg_dicts = []
    for i, seg in enumerate(segments_raw):
        if isinstance(seg, dict):
            text = seg.get('text', '').strip()
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
        else:
            text = getattr(seg, 'text', '').strip()
            start = getattr(seg, 'start', 0.0)
            end = getattr(seg, 'end', 0.0)

        if not text:
            continue
        seg_dicts.append(VoiceoverSegment(
            index=i,
            start=float(start),
            end=float(end),
            text=text,
        ))

    lang = getattr(info, 'language', 'unknown')
    dur = getattr(info, 'duration', 'unknown')
    logger.info(f"Transcribed {len(seg_dicts)} segments | lang={lang} | dur={dur}s")
    return seg_dicts


# ─────────────────────────────────────────────────────────────────────────────
# SRT parsing (fallback when video already has SRT)
# ─────────────────────────────────────────────────────────────────────────────

def parse_srt(srt_path: Path) -> List[VoiceoverSegment]:
    """Parse an SRT file into VoiceoverSegment objects."""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    segments: List[VoiceoverSegment] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
            timing = lines[1].strip()
            text = "\n".join(lines[2:])
        except (ValueError, IndexError):
            continue

        m = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            timing,
        )
        if not m:
            continue

        def parse_timestamp(ts: str) -> float:
            ts = ts.replace(",", ".")
            h, m, s = ts.split(":")
            sec, frac = s.split(".")
            return int(h) * 3600 + int(m) * 60 + float(sec) + float(frac) / 1000

        segments.append(VoiceoverSegment(
            index=index,
            start=parse_timestamp(m.group(1)),
            end=parse_timestamp(m.group(2)),
            text=text.strip(),
        ))

    logger.info(f"Parsed {len(segments)} SRT segments from {srt_path.name}")
    return segments


def find_srt(project_dir: Path) -> Optional[Path]:
    """Find an SRT file in the project directory."""
    for sub in [project_dir, project_dir / "voiceover", project_dir / "voiceover" / "srt"]:
        if sub.is_dir():
            srt_files = sorted(sub.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if srt_files:
                return srt_files[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt enhancement (Ollama — free, local)
# ─────────────────────────────────────────────────────────────────────────────

ENHANCE_SYSTEM_PROMPT = (
    "You are a visual scene description rewriter. Given voiceover narration, "
    "rewrite it as a detailed, vivid scene description for an AI image generator. "
    "Focus on what can be VISUALLY depicted — objects, setting, lighting, mood, "
    "colors, composition. Keep it under 50 words. Return ONLY the scene description, "
    "no preamble or explanation."
)

ENHANCE_USER_TEMPLATE = (
    'Given this voiceover narration: "{segment_text}"\n'
    "Rewrite it as a detailed, vivid scene description for an AI image generator.\n"
    "Focus on what can be VISUALLY depicted — objects, setting, lighting, mood.\n"
    "Keep it under 50 words. Return ONLY the scene description, no preamble."
)


def enhance_segment_text(segment_text: str, ollama_client) -> str:
    """
    Call Ollama to rewrite raw segment text into a vivid visual scene description.
    Returns the enhanced text, or original if the call fails.
    """
    user_prompt = ENHANCE_USER_TEMPLATE.format(segment_text=segment_text)
    try:
        from src.llm_client.base import LLMRequest
        request = LLMRequest(prompt=user_prompt, system_prompt=ENHANCE_SYSTEM_PROMPT)
        response = ollama_client.generate(request)
        result = response.text.strip() if (response and hasattr(response, 'text')) else segment_text
        if len(result) < 10:
            return segment_text
        return result
    except Exception as e:
        logger.warning(f"Prompt enhancement failed: {e}. Using original text.")
        return segment_text


def enhance_all_segments(
    segments: List[VoiceoverSegment],
) -> Dict[int, str]:
    """
    Call Ollama for each segment to generate enhanced scene descriptions.
    Returns a dict mapping segment index to enhanced prompt text.
    Uses OllamaClient from src.llm_client.providers.ollama.
    """
    from src.llm_client.providers.ollama import OllamaClient

    logger.info(f"Enhancing {len(segments)} segment prompts with Ollama (llama3.2)...")
    client = OllamaClient(model="llama3.2")

    enhanced: Dict[int, str] = {}
    for i, seg in enumerate(segments):
        enhanced_text = enhance_segment_text(seg.text, client)
        enhanced[seg.index] = enhanced_text
        logger.debug(f"Enhanced segment {i}: {enhanced_text[:80]}...")

    logger.info(f"Prompt enhancement complete for {len(enhanced)} segments")
    return enhanced


# ─────────────────────────────────────────────────────────────────────────────
# OTIO builder
# ─────────────────────────────────────────────────────────────────────────────

def _to_windows_path(path: str) -> str:
    if Path(path).is_absolute():
        return str(Path(path)).replace("\\", "/")
    return path.replace("\\", "/")


def build_v12_otio(results: List[GeneratedImageResult], global_start_frame: int = 0, prefix: str = "") -> otio.schema.Timeline:
    """Build an OTIO timeline containing V12 generated image clips (one per segment)."""
    timeline = otio.schema.Timeline(
        name="shot_images_timeline",
        metadata={"Resolve_OTIO": {"Resolve OTIO Meta Version": "1.0"}},
    )
    timeline.global_start_time = RationalTime(value=global_start_frame, rate=RATE)
    timeline.tracks.name = ""

    v12 = otio.schema.Track(name="V12 - Generated Images", kind=otio.schema.TrackKind.Video)
    v12.enabled = True
    v12.color = None
    v12.metadata["Resolve_OTIO"] = {"Locked": False}

    if not results:
        logger.warning("No image results — V12 track will be empty")
    else:
        sorted_results = sorted(results, key=lambda r: r.start_time)
        current_time = 0.0

        for result in sorted_results:
            file_path = result.file
            if not file_path or not Path(file_path).exists():
                logger.debug(f"Skipping {result.batch_id}: file not found")
                continue

            start = result.start_time
            end = result.end_time
            duration = max(0.001, end - start)

            gap_duration = start - current_time
            if gap_duration > 0.001:
                gap = otio.schema.Gap(
                    source_range=TimeRange(
                        start_time=RationalTime(0, RATE),
                        duration=RationalTime(int(gap_duration * RATE), RATE),
                    )
                )
                gap.enabled = True
                gap.color = None
                v12.append(gap)

            duration_frames = int(duration * RATE)

            if prefix:
                num = result.batch_id.removeprefix("shot_")
                safe_name = f"{prefix}_{num}.png"
                clip_name = f"IMG:{prefix}_{num}"
            else:
                safe_name = Path(file_path).name
                clip_name = f"IMG:{result.batch_id}"

            safe_path = _to_windows_path(file_path)
            media_ref = otio.schema.ExternalReference(
                target_url=safe_path,
                available_range=TimeRange(
                    duration=RationalTime(duration_frames, RATE),
                    start_time=RationalTime(0, RATE),
                ),
            )
            media_ref.name = safe_name

            clip = otio.schema.Clip(
                name=clip_name,
                media_reference=media_ref,
                source_range=TimeRange(
                    start_time=RationalTime(0, RATE),
                    duration=RationalTime(duration_frames, RATE),
                ),
            )
            clip.active_media_reference_key = "DEFAULT_MEDIA"
            clip.media_references = {"DEFAULT_MEDIA": media_ref}
            clip.enabled = True
            clip.color = None
            clip.effects.append(otio.schema.FreezeFrame())
            clip.metadata["Resolve_OTIO"] = {}
            clip.metadata["batch_id"] = result.batch_id
            clip.metadata["prompt_snippet"] = result.prompt[:200] if result.prompt else ""
            clip.metadata["width"] = result.width
            clip.metadata["height"] = result.height
            clip.metadata["segment_indices"] = result.segment_indices
            clip.metadata["start_time"] = result.start_time
            clip.metadata["end_time"] = result.end_time
            if prefix:
                clip.metadata["prefix"] = prefix

            v12.append(clip)
            current_time = end

    timeline.tracks.append(v12)
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_dry_run(
    batches,
    w: int,
    h: int,
    cost_per_image: float,
    verbose: bool,
    model: str,
    budget: float,
    enhance_prompts: bool,
    enhanced_prompts: Dict[int, str],
):
    """Print an enhanced dry-run summary to stdout."""
    aspect = _aspect_label(w, h)
    total_cost = cost_per_image * len(batches)
    total_duration_sec = sum(b.end_time - b.start_time for b in batches)
    total_frames = int(total_duration_sec * RATE)

    header_len = 62
    print(f"\n{'=' * header_len}")
    print(f"  DRY RUN  |  {len(batches)} images would be generated")
    print(f"  Model: {model}")
    print(f"  Resolution: {aspect}")
    print(f"  Estimated cost: ${total_cost:.3f} | Budget cap: ${budget:.2f}")
    print(f"  Total timeline: {total_duration_sec:.1f}s ({total_frames} frames @ {RATE:.0f}fps)")
    if enhance_prompts:
        print(f"  Prompt enhancement: ON ({len(enhanced_prompts)} segments enhanced)")
    else:
        print(f"  Prompt enhancement: OFF")
    print(f"{'=' * header_len}")

    for b in batches:
        shot_id = _shot_id_from_batch_id(b.batch_id)
        raw_text = b.text
        duration_sec = b.end_time - b.start_time
        duration_frames = int(duration_sec * RATE)
        start_tc = _seconds_to_tc(b.start_time)
        end_tc = _seconds_to_tc(b.end_time)
        words = _word_count(raw_text)

        # Build what the full prompt looks like to detect bare
        prompt_approx = f"Scene: {raw_text.strip()}"
        is_bare = (raw_text.strip() == prompt_approx.strip()) or (raw_text.strip() == prompt_approx.rstrip(".").strip())

        if enhanced_prompts:
            seg_idx = b.segment_start_index
            if seg_idx in enhanced_prompts:
                display_text = enhanced_prompts[seg_idx]
            else:
                display_text = raw_text
        else:
            display_text = raw_text

        print(f"\n[{shot_id}]  segment {b.segment_start_index}  |  ${cost_per_image:.3f}/image")
        print(f"  timing:  {start_tc} -> {end_tc}  |  {duration_sec:.2f}s  |  {duration_frames}f")
        print(f"  text ({words}w): {display_text[:120]}{'...' if len(display_text) > 120 else ''}")

        if verbose:
            word_status = "OK" if words <= PROMPT_RECOMMENDED_MAX_WORDS else f"above {PROMPT_RECOMMENDED_MAX_WORDS}-word recommended max"
            est_gen_sec = 3 + (words / 20)
            print(f"  word count: {words}  |  status: {word_status}")
            print(f"  est. generation time: ~{est_gen_sec:.0f}s/image")

        print(f"  prompt: Scene: {display_text[:300]}{'...' if len(display_text) > 300 else ''}")

        if is_bare and not enhance_prompts:
            print("  (prompt is bare — use --enhance-prompts for richer visual descriptions)")

    print(f"\n{'=' * header_len}")
    print(f"  Total estimate: ${total_cost:.3f}  |  {len(batches)} images  |  {total_frames} frames")
    print(f"{'=' * header_len}")

    if verbose:
        print(f"\n  Verbose notes:")
        print(f"  - Gemini Flash (gemini-2.5-flash-image) uses generate_content API")
        print(f"  - $0.039/image at 1024px; larger sizes may cost more")
        print(f"  - Recommended prompt length: <= {PROMPT_RECOMMENDED_MAX_WORDS} words")
        print(f"  - Batch IDs shown as 'shot_NNN' in this preview (internal IDs may differ)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video to SRT and generate one AI image per segment as V12 OTIO"
    )
    parser.add_argument("input", help="Video file path or project directory containing a video")
    parser.add_argument("--size", default="1408x768", help="Image size WxH (default: 1408x768)")
    parser.add_argument("--budget", type=float, default=2.0, help="Max spend in USD (default: 2.0)")
    parser.add_argument("--output", help="Output OTIO path (default: <project>/shot_images.otio)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show all segments and prompts without calling the API — zero billable calls. "
             "Outputs enhanced preview including timing, resolution, cost estimate, and "
             "bare-prompt warnings. Use --verbose for word count, length rating, and "
             "estimated generation time per image.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show additional detail in dry-run: segment word count, prompt length rating, "
             "and estimated generation time per image.",
    )
    parser.add_argument("--project-prefix", help="Prefix for renamed output files")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip transcription and use existing SRT if found")
    parser.add_argument(
        "--enhance-prompts",
        action="store_true",
        default=False,
        help="Use Gemini to rewrite each segment into a vivid visual scene description before generating images",
    )
    args = parser.parse_args()

    # ── Resolve input ────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if input_path.is_dir():
        project_dir = input_path
        video_path = find_video(project_dir)
        if not video_path:
            logger.error(f"No video file found in {project_dir}")
            sys.exit(1)
    elif input_path.is_file():
        video_path = input_path
        project_dir = video_path.parent
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)

    logger.info(f"Video: {video_path}")
    logger.info(f"Project: {project_dir}")

    # ── Transcribe or parse SRT ───────────────────────────────────────────────
    if args.skip_transcribe:
        srt_path = find_srt(project_dir)
        if not srt_path:
            logger.error("--skip-transcribe but no SRT found")
            sys.exit(1)
        segments = parse_srt(srt_path)
    else:
        srt_path = project_dir / "voiceover.srt"
        segments = transcribe_video(video_path)
        # Write SRT for reference
        from src.transcription.utils import write_srt
        seg_dicts = [
            {'id': i, 'start': s.start, 'end': s.end, 'text': s.text}
            for i, s in enumerate(segments)
        ]
        srt_path.parent.mkdir(parents=True, exist_ok=True)
        write_srt(seg_dicts, str(srt_path))
        logger.info(f"SRT written to {srt_path}")

    if not segments:
        logger.error("No segments — cannot generate images")
        sys.exit(1)

    # ── Resolve size ────────────────────────────────────────────────────────────
    try:
        w, h = map(int, args.size.lower().split("x"))
        size_tuple = (w, h)
    except ValueError:
        logger.error(f"Invalid size format '{args.size}' — use WxH like 1792x1024")
        sys.exit(1)

    if size_tuple not in IMAGEN_SUPPORTED_SIZES:
        supported = sorted(IMAGEN_SUPPORTED_SIZES)
        logger.error(f"Unsupported size {size_tuple}. Supported: {supported}")
        sys.exit(1)

    # ── Build batches: ONE segment per image ────────────────────────────────────
    config = GeneratedImagesConfig(
        enabled=True,
        image_size=GeneratedImageSizeConfig(width=w, height=h),
        quality="standard",
        budget_usd=args.budget,
        min_segments_per_image=1,
        max_segments_per_image=1,
        batch_target_segments=1,
    )

    batches = build_generated_image_batches(segments, config)
    logger.info(f"Built {len(batches)} batches ({len(segments)} segments, 1 image per segment)")

    if not batches:
        logger.error("No batches produced")
        sys.exit(1)

    # ── Prompt enhancement step ─────────────────────────────────────────────────
    enhanced_prompts: Dict[int, str] = {}
    if args.enhance_prompts:
        enhanced_prompts = enhance_all_segments(segments)

    # ── Dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        model = getattr(config, 'model', 'gemini-2.5-flash-image')
        from src.generated_images.service import IMAGEN_COST_PER_IMAGE, IMAGEN_COST_DEFAULT
        cost_per_image = IMAGEN_COST_PER_IMAGE.get(model, IMAGEN_COST_DEFAULT)
        _print_dry_run(
            batches, w, h, cost_per_image, args.verbose,
            model, args.budget, args.enhance_prompts, enhanced_prompts,
        )
        sys.exit(0)

    # ── Generate images ─────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_API_KEY not set. Set it in .env or as environment variable.")
        sys.exit(1)

    output_dir = str(project_dir / "shot_images")

    # Build text override dict for service.generate_for_batches if enhanced prompts exist
    texts_override: Optional[Dict[int, str]] = None
    if enhanced_prompts:
        texts_override = {}
        for batch in batches:
            seg_idx = batch.segment_start_index
            if seg_idx in enhanced_prompts:
                texts_override[seg_idx] = enhanced_prompts[seg_idx]

    service = GeneratedImageService(config, api_key=api_key)

    logger.info(f"Generating {len(batches)} images (1 per segment, budget=${args.budget})...")
    results = service.generate_for_batches(
        batches=batches,
        topic_context="",
        output_dir=output_dir,
        texts=texts_override,
    )

    if not results:
        logger.warning("No images generated — producing empty V12 timeline")
    else:
        total_cost = sum(r.cost_usd for r in results)
        logger.info(f"Generated {len(results)}/{len(batches)} images — ${total_cost:.3f} total")

        prefix = args.project_prefix
        if prefix:
            for r in results:
                old_path = Path(r.file)
                num = r.batch_id.removeprefix("shot_")
                new_name = f"{prefix}_{num}.png"
                new_path = old_path.parent / new_name
                if old_path.exists():
                    old_path.rename(new_path)
                r.file = str(new_path)
            logger.info(f"Files renamed with prefix '{prefix}_'")

    # ── Build and write OTIO ───────────────────────────────────────────────────
    timeline = build_v12_otio(results, prefix=args.project_prefix or "")
    timeline_duration_sec = timeline.duration().value / RATE
    logger.info(f"Timeline duration: {timeline_duration_sec:.2f}s")

    if args.output:
        output_otio = Path(args.output)
    else:
        output_otio = project_dir / "shot_images.otio"

    output_otio.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(timeline, str(output_otio))
    logger.info(f"Wrote: {output_otio}")

    # ── Summary JSON ───────────────────────────────────────────────────────────
    summary = {
        "video_file": str(video_path),
        "srt_file": str(srt_path),
        "images_generated": len(results),
        "segments": len(segments),
        "total_cost_usd": sum(r.cost_usd for r in results),
        "size": {"width": w, "height": h},
        "budget_usd": args.budget,
        "prompt_enhancement": args.enhance_prompts,
        "timeline_duration_sec": timeline_duration_sec,
        "images": [
            {
                "batch_id": r.batch_id,
                "file": r.file,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "segment_index": r.segment_indices[0] if r.segment_indices else -1,
                "prompt_snippet": r.prompt[:200] if r.prompt else "",
                "cost_usd": r.cost_usd,
            }
            for r in results
        ],
    }

    summary_path = project_dir / "shot_images" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary: {summary_path}")

    print("\n" + "=" * 60)
    print(f"  Generated {len(results)} images  |  ${sum(r.cost_usd for r in results):.3f}")
    print(f"  OTIO: {output_otio}")
    print(f"  Images: {output_dir}")
    if args.enhance_prompts:
        print(f"  Prompt enhancement: ON ({len(enhanced_prompts)} segments)")
    print("=" * 60)


if __name__ == "__main__":
    main()
