#!/usr/bin/env python3
"""Split SRT file for timeline parts.

Creates Part1 SRT (entries 1 to N) and Part2 SRT (entries N+1 onwards,
renumbered to start from 1 with timestamps adjusted relative to Part2 start).

Usage:
    python scripts/split_srt.py "e:/Edit Job/Marcos/1/voiceover/voiceover_trimmed.srt" 321.237 --output e:/Edit Job/Marcos/1/voiceover
"""

import argparse
import re
import os

def parse_srt_time(t: str) -> float:
    """Parse SRT time (HH:MM:SS,mmm) to seconds."""
    t = t.replace(',', '.')
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

def parse_srt(content: str):
    """Parse SRT content into list of (index, start, end, text)."""
    entries = []
    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            idx = int(lines[0])
            times = lines[1].split(' --> ')
            start = parse_srt_time(times[0].strip())
            end = parse_srt_time(times[1].strip())
            text = '\n'.join(lines[2:])
            entries.append((idx, start, end, text))
    return entries

def write_srt_entry(f, number, start, end, text):
    """Write a single SRT entry with renumbered timestamps."""
    f.write(f"{number}\n")
    f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
    f.write(f"{text}\n\n")

def split_srt(srt_path: str, split_at_seconds: float, output_dir: str = None):
    """Split SRT at given time, creating Part1 and Part2 files."""
    if output_dir is None:
        output_dir = os.path.dirname(srt_path)

    base = os.path.splitext(os.path.basename(srt_path))[0]

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = parse_srt(content)
    print(f"Loaded {len(entries)} SRT entries")
    print(f"Split at {split_at_seconds:.3f}s")

    # Find split point
    part1_entries = []
    part2_entries = []
    for idx, start, end, text in entries:
        if end <= split_at_seconds:
            part1_entries.append((idx, start, end, text))
        elif start >= split_at_seconds:
            part2_entries.append((idx, start, end, text))
        else:
            # Entry spans the split - include in part1
            part1_entries.append((idx, start, end, text))

    print(f"Part1: entries 1-{len(part1_entries)}")
    print(f"Part2: entries {part1_entries[-1][0] if part1_entries else 0 + 1}-{len(entries)} (renumbered)")

    # Write Part1 SRT (same numbering, only entries before split)
    part1_path = os.path.join(output_dir, f"{base}_Part1.srt")
    with open(part1_path, 'w', encoding='utf-8') as f:
        for i, (orig_idx, start, end, text) in enumerate(part1_entries, 1):
            write_srt_entry(f, i, start, end, text)
    print(f"Written: {part1_path}")

    # Write Part2 SRT (renumbered from 1, timestamps relative to split point)
    if part2_entries:
        part2_path = os.path.join(output_dir, f"{base}_Part2.srt")
        with open(part2_path, 'w', encoding='utf-8') as f:
            for i, (orig_idx, start, end, text) in enumerate(part2_entries, 1):
                adj_start = start - split_at_seconds
                adj_end = end - split_at_seconds
                write_srt_entry(f, i, adj_start, adj_end, text)
        print(f"Written: {part2_path}")

        # Verify first few Part2 entries
        print("Part2 first 3 entries:")
        for i, (orig_idx, start, end, text) in enumerate(part2_entries[:3], 1):
            adj_start = start - split_at_seconds
            print(f"  [{i}] orig start={start:.3f}s -> {adj_start:.3f}s, text={text[:40]}")
    else:
        print("Part2: no entries (split point at or after last entry)")

    return part1_path, part2_path if part2_entries else None


def main():
    parser = argparse.ArgumentParser(description="Split SRT at a given time for timeline parts")
    parser.add_argument("srt_path", help="Input SRT file")
    parser.add_argument("split_seconds", type=float, help="Split point in seconds")
    parser.add_argument("--output", "-o", help="Output directory")
    args = parser.parse_args()

    split_srt(args.srt_path, args.split_seconds, args.output)


if __name__ == "__main__":
    main()