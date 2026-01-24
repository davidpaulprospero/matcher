#!/usr/bin/env python3
"""
Generate a video with hard color cuts at EDL marker positions.
Used for triggering scene detection in editors that don't support EDL import (e.g., CapCut).
"""

import re
import subprocess
import sys
from pathlib import Path


def parse_timecode(tc: str, fps: float = 30.0) -> float:
    """Convert HH:MM:SS:FF timecode to seconds."""
    match = re.match(r"(\d+):(\d+):(\d+):(\d+)", tc)
    if not match:
        raise ValueError(f"Invalid timecode: {tc}")
    h, m, s, f = map(int, match.groups())
    return h * 3600 + m * 60 + s + f / fps


def parse_edl(edl_path: Path, fps: float = 30.0) -> list[float]:
    """Extract marker timecodes from EDL file, return as seconds."""
    markers = []
    content = edl_path.read_text(encoding="utf-8")

    # Match timecode pattern in EDL events (first timecode is the marker position)
    # Format: 001  001      V     C        01:00:22:24 01:00:22:25 ...
    for line in content.splitlines():
        match = re.match(r"\d+\s+\d+\s+V\s+C\s+(\d+:\d+:\d+:\d+)", line.strip())
        if match:
            tc = match.group(1)
            markers.append(parse_timecode(tc, fps))

    return sorted(markers)


def generate_scene_video(edl_path: Path, output_path: Path, fps: float = 30.0):
    """Generate video with color changes at each marker."""
    markers = parse_edl(edl_path, fps)

    if not markers:
        print("No markers found in EDL!")
        sys.exit(1)

    print(f"Found {len(markers)} markers")

    # Timeline starts at 01:00:00:00 = 3600 seconds
    timeline_start = 3600.0
    timeline_end = markers[-1] + 1.0  # 1 second after last marker

    # Convert absolute timecodes to relative (from video start)
    relative_markers = [m - timeline_start for m in markers]
    total_duration = timeline_end - timeline_start

    print(f"Video duration: {total_duration:.2f}s ({total_duration/60:.1f} min)")
    print(f"First marker at: {relative_markers[0]:.2f}s")
    print(f"Last marker at: {relative_markers[-1]:.2f}s")

    # Build segment list: [(start, end, color), ...]
    segments = []
    colors = ["black", "white"]

    # First segment: 0 to first marker
    if relative_markers[0] > 0:
        segments.append((0, relative_markers[0], colors[0]))

    # Segments between markers
    for i, marker in enumerate(relative_markers):
        if i < len(relative_markers) - 1:
            next_marker = relative_markers[i + 1]
        else:
            next_marker = total_duration

        color = colors[(i + 1) % 2]  # Alternate starting from second color
        segments.append((marker, next_marker, color))

    print(f"\nSegments ({len(segments)} total):")
    for i, (start, end, color) in enumerate(segments[:5]):
        print(f"  {i+1}. {start:.2f}s - {end:.2f}s [{color}]")
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more")

    # Build FFmpeg filter complex
    # Create color sources for each segment, then concat
    filter_parts = []
    concat_inputs = []

    for i, (start, end, color) in enumerate(segments):
        duration = end - start
        # color source with specific duration
        filter_parts.append(
            f"color=c={color}:s=640x360:r={fps}:d={duration:.6f}[v{i}]"
        )
        concat_inputs.append(f"[v{i}]")

    # Concat all segments
    filter_complex = "; ".join(filter_parts)
    filter_complex += f"; {''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=0[out]"

    # Build FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        str(output_path)
    ]

    print(f"\nGenerating: {output_path}")
    print("Running FFmpeg...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr}")
        sys.exit(1)

    print(f"Done! Output: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python edl_to_scene_video.py <edl_file> [output_file]")
        print("       If output not specified, creates scene_trigger.mp4 in same folder")
        sys.exit(1)

    edl_path = Path(sys.argv[1])
    if not edl_path.exists():
        print(f"EDL file not found: {edl_path}")
        sys.exit(1)

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = edl_path.parent / "scene_trigger.mp4"

    generate_scene_video(edl_path, output_path)


if __name__ == "__main__":
    main()
