#!/usr/bin/env python3
"""
imagen_otio.py — Generate AI still images from voiceover using Imagen and
produce an OTIO timeline with those images on the V12 track.

Usage:
    python scripts/imagen_otio.py "<srt_or_project>" [--size WxH] [--quality Q] [--budget USD] [--output <path>"]

Examples:
    python scripts/imagen_otio.py "E:\Edit Job\Stu\my_project\voiceover\voiceover.srt"
    python scripts/imagen_otio.py "E:\Edit Job\Stu\my_project" --size 1792x1024 --budget 5.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load .env from project root so GEMINI_API_KEY is available
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

from src.generated_images import build_generated_image_batches, GeneratedImageService
from src.config.sections.generated_images import GeneratedImagesConfig, IMAGEN_SUPPORTED_SIZES, GeneratedImageSizeConfig
from src.state import VoiceoverSegment, GeneratedImageResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RATE = 30.0  # frames per second for timeline


# ─────────────────────────────────────────────────────────────────────────────
# SRT Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_srt(srt_path: Path) -> List[VoiceoverSegment]:
    """Parse an SRT file into VoiceoverSegment objects."""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    segments: List[VoiceoverSegment] = []
    import re

    # SRT block pattern: index, timing, text
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

        # Parse timing: "00:00:01,200 --> 00:00:04,500"
        m = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", timing)
        if not m:
            continue

        def parse_timestamp(ts: str) -> float:
            ts = ts.replace(",", ".")
            h, m, s = ts.split(":")
            sec, frac = s.split(".")
            return int(h) * 3600 + int(m) * 60 + float(sec) + float(frac) / 1000

        start = parse_timestamp(m.group(1))
        end = parse_timestamp(m.group(2))

        segments.append(VoiceoverSegment(
            index=index,
            start=start,
            end=end,
            text=text.strip(),
        ))

    logger.info(f"Parsed {len(segments)} SRT segments from {srt_path.name}")
    return segments


def find_srt(project_dir: Path) -> Optional[Path]:
    """Find an SRT file in the project directory."""
    search_dirs = [
        project_dir / "voiceover",
        project_dir / "voiceover" / "srt",
        project_dir / "voiceover" / "SRT",
    ]
    for search_dir in search_dirs:
        if search_dir.is_dir():
            srt_files = sorted(search_dir.glob("*.srt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if srt_files:
                return srt_files[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# OTIO Builder
# ─────────────────────────────────────────────────────────────────────────────

def _to_windows_path(path: str) -> str:
    """Forward-slash Windows path — DaVinci imports better without file:///."""
    if Path(path).is_absolute():
        return str(Path(path)).replace("\\", "/")
    return path.replace("\\", "/")


def build_v12_otio(results: List[GeneratedImageResult], global_start_frame: int = 0, prefix: str = "") -> otio.schema.Timeline:
    """Build an OTIO timeline containing V12 generated image clips.

    DaVinci-compatible structure:
    - timeline.tracks is a Stack with name="" (empty string — DaVinci convention)
    - V12 track is a direct child of timeline.tracks — no inner Stack wrapper
    - global_start_time = 0 (not 108000)
    - Clip uses clip.media_reference = media_ref (produces ExternalReference in JSON)
    - DaVinci-compatible metadata on all elements
    - File renaming: if prefix is set, files are renamed to {prefix}_{NNN}.png and
      all clip names/media refs use the prefixed name to avoid DaVinci media pool conflicts
    """

    timeline = otio.schema.Timeline(
        name="generated_images_timeline",
        metadata={"Resolve_OTIO": {"Resolve OTIO Meta Version": "1.0"}},
    )
    timeline.global_start_time = RationalTime(value=global_start_frame, rate=RATE)
    timeline.tracks.name = ""

    # V12 track — direct child of timeline.tracks, no inner Stack wrapper
    v12 = otio.schema.Track(name="V12 - Generated Images", kind=otio.schema.TrackKind.Video)
    v12.enabled = True
    v12.color = None
    v12.metadata["Resolve_OTIO"] = {"Locked": False}

    if not results:
        logger.warning("No image results — V12 track will be empty")
    else:
        sorted_results = sorted(results, key=lambda r: r.start_time)
        current_time = 0.0  # seconds

        for result in sorted_results:
            file_path = result.file
            if not file_path or not Path(file_path).exists():
                logger.debug(f"Skipping {result.batch_id}: file not found")
                continue

            start = result.start_time
            end = result.end_time
            duration = max(0.001, end - start)

            # Gap from current position to this clip's start
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

            # Determine display name (prefixed if project prefix given)
            if prefix:
                # Derive number from batch_id like "generated_042" -> "042"
                num = result.batch_id.removeprefix("generated_")
                safe_name = f"{prefix}_{num}.png"
                clip_name = f"IMG:{prefix}_{num}"
            else:
                safe_name = Path(file_path).name
                clip_name = f"IMG:{result.batch_id}"

            # Media reference — forward-slash Windows path (DaVinci-compatible)
            safe_path = _to_windows_path(file_path)
            media_ref = otio.schema.ExternalReference(
                target_url=safe_path,
                available_range=TimeRange(
                    duration=RationalTime(duration_frames, RATE),
                    start_time=RationalTime(0, RATE),
                ),
            )
            media_ref.name = safe_name

            # Clip — direct media_reference assignment (produces ExternalReference in JSON)
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

    # V12 is direct child of timeline.tracks — no inner Stack wrapper
    timeline.tracks.append(v12)
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate imagen OTIO from voiceover SRT")
    parser.add_argument("input", help="SRT file path or project directory")
    parser.add_argument("--size", default="1408x768", help="Image size WxH (default: 1408x768, cheapest 1K 16:9)")
    parser.add_argument("--quality", default="standard", choices=["standard", "fast", "ultra"], help="Quality tier (default: standard, $0.04/img — Fast is fixed-size and rejects image_size)")
    parser.add_argument("--budget", type=float, default=2.0, help="Max spend in USD (default: 2.0)")
    parser.add_argument("--output", help="Output OTIO path (default: <project>/generated_images.otio)")
    parser.add_argument("--dry-run", action="store_true", help="Show batches and prompts without calling Imagen or generating images")
    parser.add_argument("--project-prefix", help="Prefix for renamed output files (e.g. wcyzksaq, eqiecv9k). If omitted, files keep their default names.")
    args = parser.parse_args()

    # ── Resolve SRT and project ────────────────────────────────────────────────
    input_path = Path(args.input)
    if input_path.is_dir():
        project_dir = input_path
        srt_path = find_srt(project_dir)
        if not srt_path:
            logger.error(f"No SRT found in {project_dir}/voiceover/")
            sys.exit(1)
    elif input_path.is_file():
        srt_path = input_path
        project_dir = srt_path.parent
        # Walk up to find project root
        for parent in [srt_path.parent, srt_path.parent.parent]:
            if (parent / "checkpoint.json").exists() or (parent / "config.yaml").exists():
                project_dir = parent
                break
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)

    logger.info(f"SRT: {srt_path}")
    logger.info(f"Project: {project_dir}")

    # ── Parse SRT ───────────────────────────────────────────────────────────────
    segments = parse_srt(srt_path)
    if not segments:
        logger.error("No segments parsed from SRT")
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

    # ── Build batches ───────────────────────────────────────────────────────────
    config = GeneratedImagesConfig(
        enabled=True,
        image_size=GeneratedImageSizeConfig(width=w, height=h),
        quality=args.quality,
        budget_usd=args.budget,
        min_segments_per_image=7,
        max_segments_per_image=9,
        batch_target_segments=8,
    )

    batches = build_generated_image_batches(segments, config)
    logger.info(f"Built {len(batches)} batches from {len(segments)} segments")

    if not batches:
        logger.error("No batches produced — check segment count and batching config")
        sys.exit(1)

    # ── Dry run: show batches and prompts without billing ─────────────────────
    if args.dry_run:
        from src.generated_images import GeneratedImagePromptBuilder
        prompt_builder = GeneratedImagePromptBuilder(config)
        total_cost = 0.0
        model = getattr(config, 'model', 'imagen-4.0-generate-001')
        from src.generated_images.service import IMAGEN_COST_PER_IMAGE, IMAGEN_COST_DEFAULT
        cost_per_image = IMAGEN_COST_PER_IMAGE.get(model, IMAGEN_COST_DEFAULT)

        print("\n" + "=" * 60)
        print(f"  DRY RUN - {len(batches)} images would be generated")
        print(f"  Estimated cost: ${cost_per_image * len(batches):.3f} ({model})")
        print(f"  Size: {w}x{h} | Quality: {args.quality} | Budget cap: ${args.budget}")
        print("=" * 60)
        for b in batches:
            prompt = prompt_builder.build(b.text, topic_context="")
            cost = cost_per_image
            total_cost += cost
            print(f"\n[{b.batch_id}] segments {b.segment_start_index}-{b.segment_end_index} ({b.segment_count} segs) | ${cost:.3f}")
            print(f"  time: {b.start_time:.2f}s -> {b.end_time:.2f}s")
            print(f"  prompt: {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
        print("\n" + "=" * 60)
        print(f"  Total estimate: ${total_cost:.3f}")
        print("  Pass --dry-run to see this output; omit to generate.")
        print("=" * 60)
        sys.exit(0)

    # ── Generate images ─────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY not set. Set it in .env or as environment variable.\n"
            "  export GEMINI_API_KEY=your_key  # Linux/Mac\n"
            "  $env:GEMINI_API_KEY='your_key'   # PowerShell\n"
        )
        sys.exit(1)

    output_dir = str(project_dir / "generated_images")
    service = GeneratedImageService(config, api_key=api_key)

    logger.info(f"Generating {len(batches)} images (budget=${args.budget}, quality={args.quality})...")
    results = service.generate_for_batches(
        batches=batches,
        topic_context="",
        output_dir=output_dir,
    )

    if not results:
        logger.warning("No images generated — producing empty V12 timeline")
    else:
        total_cost = sum(r.cost_usd for r in results)
        logger.info(f"Generated {len(results)}/{len(batches)} images — ${total_cost:.3f} total")

        # Rename files on disk to prefixed names and update result.file paths
        prefix = args.project_prefix
        if prefix:
            for r in results:
                old_path = Path(r.file)
                num = r.batch_id.removeprefix("generated_")
                new_name = f"{prefix}_{num}.png"
                new_path = old_path.parent / new_name
                if old_path.exists():
                    old_path.rename(new_path)
                r.file = str(new_path)
            logger.info(f"Files renamed with prefix '{prefix}_'")

    # ── Build OTIO ──────────────────────────────────────────────────────────────
    timeline = build_v12_otio(results, prefix=args.project_prefix or "")
    timeline_duration_sec = timeline.duration().value / RATE
    logger.info(f"Timeline duration: {timeline_duration_sec:.2f}s")

    # ── Write OTIO ──────────────────────────────────────────────────────────────
    if args.output:
        output_otio = Path(args.output)
    else:
        output_otio = project_dir / "generated_images.otio"

    output_otio.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(timeline, str(output_otio))
    logger.info(f"Wrote: {output_otio}")

    # ── Write summary JSON ──────────────────────────────────────────────────────
    summary = {
        "srt_file": str(srt_path),
        "images_generated": len(results),
        "batches_created": len(batches),
        "total_cost_usd": sum(r.cost_usd for r in results),
        "size": {"width": w, "height": h},
        "quality": args.quality,
        "budget_usd": args.budget,
        "timeline_duration_sec": timeline_duration_sec,
        "images": [
            {
                "batch_id": r.batch_id,
                "file": r.file,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "segment_count": len(r.segment_indices),
                "cost_usd": r.cost_usd,
            }
            for r in results
        ],
    }

    summary_path = project_dir / "generated_images" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary: {summary_path}")

    # ── Report ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Generated {len(results)} images  |  ${sum(r.cost_usd for r in results):.3f}")
    print(f"  OTIO: {output_otio}")
    print(f"  Images: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()