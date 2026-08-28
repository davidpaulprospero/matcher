#!/usr/bin/env python3
"""Correctly split timeline_FULL.otio into two parts at a given time.

Handles gaps spanning the split point by:
- Part1: trim the gap portion before split
- Part2: start with trimmed gap (remaining duration after split)

Usage:
    python scripts/split_otio.py "e:/Edit Job/Marcos/1/output/20260522_202900/timeline_FULL23.otio" 322
    python scripts/split_otio.py "/path/to/timeline.otio" 322 --dry-run
"""

import argparse
import opentimelineio as otio
from opentimelineio import opentime
from opentimelineio.schema import TrackKind, Track, Timeline, Stack

def split_time_to_rational(split_seconds: float, rate: float = 30.0) -> opentime.RationalTime:
    return opentime.RationalTime(round(split_seconds * rate), rate)

from opentimelineio.schema import TrackKind, Track, Timeline, Stack, Clip, Gap

def clone_clip(clip):
    """Clone a clip."""
    return Clip(
        name=clip.name,
        source_range=clip.source_range,
        media_reference=clip.media_reference,
        effects=clip.effects[:] if clip.effects else [],
        markers=clip.markers[:] if clip.markers else [],
        metadata=dict(clip.metadata) if clip.metadata else {},
    )

def clone_gap(gap):
    """Clone a gap."""
    return Gap(
        name=gap.name,
        source_range=gap.source_range,
        effects=gap.effects[:] if gap.effects else [],
        markers=gap.markers[:] if gap.markers else [],
        metadata=dict(gap.metadata) if gap.metadata else {},
    )

def split_timeline(tl: Timeline, split_seconds: float, rate: float = 30.0):
    """Split timeline into two parts at split_seconds."""
    split_rt = split_time_to_rational(split_seconds, rate)

    tracks_p1 = []
    tracks_p2 = []

    for track in tl.tracks:
        track_p1 = Track(name=track.name, kind=TrackKind.Video)
        track_p2 = Track(name=track.name, kind=TrackKind.Video)

        cum_rt = opentime.RationalTime(0, rate)

        for item in track:
            if item is None:
                continue

            item_type = type(item).__name__
            sr = item.source_range
            dur_rt = sr.duration
            end_rt = cum_rt + dur_rt

            # Case 1: Item entirely before split
            if end_rt <= split_rt:
                new_item = clone_clip(item) if item_type == 'Clip' else clone_gap(item)
                track_p1.append(new_item)
                cum_rt = end_rt

            # Case 2: Item entirely after split
            elif cum_rt >= split_rt:
                new_item = clone_clip(item) if item_type == 'Clip' else clone_gap(item)
                track_p2.append(new_item)
                cum_rt = end_rt

            # Case 3: Item spans the split point
            else:
                # Split the item at split_rt
                pre_dur = (split_rt - cum_rt).value
                post_dur = dur_rt.value - pre_dur

                # PRE part: goes to Part1
                pre_range = otio.opentime.TimeRange(
                    start_time=sr.start_time,
                    duration=opentime.RationalTime(pre_dur, rate)
                )
                if item_type == 'Clip':
                    pre_item = Clip(
                        name=item.name,
                        source_range=pre_range,
                        media_reference=item.media_reference,
                        effects=item.effects[:] if item.effects else [],
                        markers=item.markers[:] if item.markers else [],
                        metadata=dict(item.metadata) if item.metadata else {},
                    )
                else:
                    pre_item = Gap(
                        name=item.name or "",
                        source_range=pre_range,
                        effects=item.effects[:] if item.effects else [],
                        markers=item.markers[:] if item.markers else [],
                        metadata=dict(item.metadata) if item.metadata else {},
                    )
                track_p1.append(pre_item)

                # POST part: goes to Part2 (adjust source_range start to beginning)
                post_range = otio.opentime.TimeRange(
                    start_time=sr.start_time + opentime.RationalTime(pre_dur, rate),
                    duration=opentime.RationalTime(post_dur, rate)
                )
                if item_type == 'Clip':
                    post_item = Clip(
                        name=item.name,
                        source_range=post_range,
                        media_reference=item.media_reference,
                        effects=item.effects[:] if item.effects else [],
                        markers=item.markers[:] if item.markers else [],
                        metadata=dict(item.metadata) if item.metadata else {},
                    )
                else:
                    post_item = Gap(
                        name=item.name or "",
                        source_range=post_range,
                        effects=item.effects[:] if item.effects else [],
                        markers=item.markers[:] if item.markers else [],
                        metadata=dict(item.metadata) if item.metadata else {},
                    )
                track_p2.append(post_item)
                cum_rt = end_rt

        tracks_p1.append(track_p1)
        tracks_p2.append(track_p2)

    # Build Part1 timeline
    tl_p1 = Timeline(
        name=f"{tl.name}_Part1",
        metadata=dict(tl.metadata) if tl.metadata else {},
        global_start_time=opentime.RationalTime(0, rate),
    )
    stack_p1 = Stack(name="tracks")
    for t in tracks_p1:
        stack_p1.append(t)
    tl_p1.tracks = stack_p1

    # Build Part2 timeline
    tl_p2 = Timeline(
        name=f"{tl.name}_Part2",
        metadata=dict(tl.metadata) if tl.metadata else {},
        global_start_time=opentime.RationalTime(0, rate),
    )
    stack_p2 = Stack(name="tracks")
    for t in tracks_p2:
        stack_p2.append(t)
    tl_p2.tracks = stack_p2

    return tl_p1, tl_p2


def main():
    parser = argparse.ArgumentParser(description="Split OTIO timeline at a given time")
    parser.add_argument("input_otio", help="Input OTIO file path")
    parser.add_argument("split_seconds", type=float, help="Split point in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    args = parser.parse_args()

    input_path = args.input_otio
    split_sec = args.split_seconds

    print(f"Loading: {input_path}")
    tl = otio.adapters.read_from_file(input_path)
    print(f"Timeline: {tl.name}, duration={tl.duration().value / 30:.3f}s, {len(tl.tracks)} tracks")

    print(f"\nSplitting at {split_sec}s ({int(split_sec * 30)} frames at 30fps)")
    tl_p1, tl_p2 = split_timeline(tl, split_sec)

    print(f"\nPart1: {tl_p1.name}, duration={tl_p1.duration().value / 30:.3f}s")
    print(f"Part2: {tl_p2.name}, duration={tl_p2.duration().value / 30:.3f}s")

    # Verify key tracks
    for track_p1, track_p2 in zip(tl_p1.tracks, tl_p2.tracks):
        p1_dur = track_p1.duration().value / 30 if track_p1.duration() else 0
        p2_dur = track_p2.duration().value / 30 if track_p2.duration() else 0
        print(f"  {track_p1.name}: Part1={p1_dur:.3f}s, Part2={p2_dur:.3f}s")

    if args.dry_run:
        print("\n[DRY RUN] Not writing files")
        return

    import os
    output_dir = os.path.dirname(input_path)

    out_p1 = os.path.join(output_dir, f"{tl.name}_fixed_Part1.otio")
    out_p2 = os.path.join(output_dir, f"{tl.name}_fixed_Part2.otio")

    print(f"\nWriting {out_p1}")
    otio.adapters.write_to_file(tl_p1, out_p1)
    print(f"Writing {out_p2}")
    otio.adapters.write_to_file(tl_p2, out_p2)
    print("Done")


if __name__ == "__main__":
    main()