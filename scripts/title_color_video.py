#!/usr/bin/env python3
"""
Generate a small FFmpeg video that alternates white/black backgrounds
at each title timestamp found in a project SRT.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import parse_srt_file, SRTSegment


DEFAULT_TITLE_PHRASES = [
    "Let's start with the most basic version",
    "The building isn't just standing",
    "That's the track",
    "But it's not just vehicles sitting in buildings",
    "The harder cases are the ones where something didn't just close and sit",
    "The invisible park isn't just behind walls",
    "Here is where it gets genuinely strange",
    "Which raises a question that sounds simple",
    "Here's where the story gets sharp",
    "Which brings us to the final layer",
    "Every data point in this video connects",
    "Here is the practical piece",
]


def find_srt_path(project_dir: Path) -> Path | None:
    """Find the appropriate SRT path for a project, preferring trimmed variant."""
    voiceover_dir = project_dir / "voiceover"
    candidates = []

    # Search voiceover/ subdirectory
    if voiceover_dir.exists():
        for f in voiceover_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".srt":
                candidates.append(f)

    # Also search project root
    for f in project_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".srt" and f not in candidates:
            candidates.append(f)

    if not candidates:
        return None

    # Prefer trimmed variant: drop regular if _trimmed also exists
    trimmed_stems = {f.stem.replace("_trimmed", "") for f in candidates if "_trimmed" in f.stem}
    if trimmed_stems:
        candidates = [f for f in candidates if "_trimmed" in f.stem or f.stem not in trimmed_stems]

    # If multiple remain, use first (deterministic enough)
    return candidates[0] if candidates else None


from rapidfuzz import fuzz


def phrase_score(text: str, phrase: str) -> float:
    """Return fuzzy match score between text and phrase.

    Uses partial_ratio to detect substring containment.
    token_set_ratio is NOT used because it gives false positives when
    a short segment shares many tokens with the phrase — even if those
    tokens are in a completely different context.
    """
    return fuzz.partial_ratio(text.lower(), phrase.lower())


def find_title_cut_points(
    segments: list[SRTSegment],
    phrases: list[str],
    threshold: int = 82,
    window: int = 3,
) -> list[float]:
    """Find one cut point per phrase.

    Strategy for each phrase:
    1. Exact substring match on individual segments
    2. Exact substring scan over consecutive multi-segment groups (up to window size)
    3. Fuzzy fallback only if neither exact strategy succeeds
    """
    cut_points = []
    used_starts = []  # for dedup within 5s

    for phrase in phrases:
        found_start = None

        # Step 1: exact match on single segment
        for seg in segments:
            if phrase.lower() in seg.text.lower():
                found_start = seg.start_time
                break

        # Step 2: exact multi-segment scan (spans across consecutive SRT segments)
        if found_start is None:
            for window_size in range(2, window + 1):
                for i in range(len(segments) - window_size + 1):
                    group_text = " ".join(segments[i + j].text for j in range(window_size))
                    if phrase.lower() in group_text.lower():
                        found_start = segments[i].start_time
                        break
                if found_start is not None:
                    break

        # Step 3: fuzzy fallback (last resort for phrasing variations)
        if found_start is None:
            best_score = 0
            best_start = None
            for i in range(len(segments)):
                # Gate: if the starting segment alone scores too low, the phrase
                # can't be well-represented here — skip to avoid false positives
                first_score = phrase_score(segments[i].text, phrase)
                if first_score < 60:
                    continue
                # Additional gate: skip if segment is much longer than phrase
                # (likely a continuation segment that happens to share substring)
                if len(segments[i].text) > len(phrase) * 1.5:
                    continue
                for window_size in range(1, window + 1):
                    if i + window_size > len(segments):
                        break
                    group_text = " ".join(segments[i + j].text for j in range(window_size))
                    score = phrase_score(group_text, phrase)
                    if score > best_score:
                        best_score = score
                        best_start = segments[i].start_time
            if best_score >= threshold and best_start is not None:
                found_start = best_start

        if found_start is not None:
            if not any(abs(found_start - s) < 5.0 for s in used_starts):
                cut_points.append(found_start)
                used_starts.append(found_start)

    return sorted(set(cut_points))


def build_color_segments(cut_points: list[float], total_duration: float) -> list[tuple[float, float, str]]:
    """Build (start, end, color) segments alternating black/white."""
    if not cut_points:
        return [(0.0, total_duration, "black")]

    # Prepend 0 if not present
    points = [0.0] + cut_points
    # Append total_duration if last point isn't at end
    if cut_points[-1] < total_duration - 0.5:
        points.append(total_duration)

    segments = []
    colors = ["black", "white"]
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        if end > start:
            segments.append((start, end, colors[i % 2]))
    return segments


def build_ffmpeg_cmd(segments: list[tuple[float, float, str]], output_path: Path) -> list[str]:
    """Build FFmpeg command for color-segment video."""
    filter_parts = []
    concat_inputs = []
    for i, (start, end, color) in enumerate(segments):
        duration = end - start
        filter_parts.append(
            f"color=c={color}:s=640x360:r=24:d={duration:.3f}[v{i}]"
        )
        concat_inputs.append(f"[v{i}]")

    filter_complex = "; ".join(filter_parts)
    filter_complex += f"; {''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[out]"

    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-an",
        str(output_path),
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Generate title-driven color-switch video")
    parser.add_argument("project", help="Project directory")
    parser.add_argument("--output", "-o", default="title_colors.mp4", help="Output MP4 path")
    parser.add_argument(
        "--phrases",
        help="Pipe-separated title phrases (overrides defaults)",
    )
    parser.add_argument(
        "--srt",
        help="Explicit SRT path (skip auto-discovery)",
    )
    parser.add_argument(
        "--threshold", "-t", type=int, default=82,
        help="Fuzzy match threshold 0-100 (default 82)",
    )
    args = parser.parse_args()

    project_dir = Path(args.project)
    if not project_dir.exists():
        print(f"ERROR: project directory not found: {project_dir}", file=sys.stderr)
        sys.exit(1)

    # Find SRT
    if args.srt:
        srt_path = Path(args.srt)
    else:
        srt_path = find_srt_path(project_dir)
        if not srt_path:
            print(f"ERROR: no SRT found in {project_dir} or voiceover/", file=sys.stderr)
            sys.exit(1)

    print(f"Using SRT: {srt_path}")

    # Parse SRT
    segments = parse_srt_file(str(srt_path))
    if not segments:
        print("ERROR: SRT parsed to empty segments", file=sys.stderr)
        sys.exit(1)

    total_duration = segments[-1].end_time
    print(f"SRT duration: {total_duration:.2f}s, {len(segments)} segments")

    # Title phrases
    phrases = DEFAULT_TITLE_PHRASES
    if args.phrases:
        phrases = args.phrases.split("|")

    # Find cut points
    cut_points = find_title_cut_points(segments, phrases, threshold=args.threshold)
    print(f"Found {len(cut_points)} title timestamps: {cut_points}")

    if not cut_points:
        print("WARNING: no title phrases matched — making solid black video", file=sys.stderr)

    # Build color segments
    color_segs = build_color_segments(cut_points, total_duration)
    print(f"Color segments: {len(color_segs)}")
    for i, (s, e, c) in enumerate(color_segs):
        print(f"  [{i}] {s:.3f}s - {e:.3f}s ({c})")

    # Build FFmpeg command
    output_path = Path(args.output)
    cmd = build_ffmpeg_cmd(color_segs, output_path)

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"FFmpeg failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    size_kb = output_path.stat().st_size // 1024
    print(f"\nOutput: {output_path} ({size_kb}KB)")


if __name__ == "__main__":
    main()