"""Flatten an OTIO to multi-track output preserving ALL visible clip portions.

Unlike `flatten-otio-to-single-track.py` (which picks only the topmost clip per
moment), this version emits every enabled clip whose timeline range has any
visible portion. A clip is "visible" if no enabled clip on a higher track
covers it. When a higher clip only partially overlaps, the lower clip is split
into its visible sub-ranges.

Output is multi-track: one output track per source track that has visible
content, ordered with the highest source track on top. Gaps fill the empty
portions (where a higher clip occludes this track, or where no clip exists).

Usage:
    python scripts/flatten-otio-visible.py <input.otio> [output.otio]

Default output: <input>_visible.otio
"""
import sys
from pathlib import Path

import opentimelineio as otio


RATE = 30.0


def parse_visible(tl):
    """Compute per-track visible segments.

    Returns:
      - per_track_visible: dict[track_idx] -> list of (tl_start, tl_end, clip)
        where (tl_start, tl_end) is a visible sub-range of `clip` on that track.
      - track_order: list of source track_idx that have visible content.
      - total_dur: int (frame count) of the timeline.
    """
    all_clips = []
    track_order_candidates = []
    src_dur = 0
    for track_idx, track in enumerate(tl.tracks):
        if not track.enabled or track.kind == otio.schema.TrackKind.Audio:
            continue
        track_order_candidates.append(track_idx)
        tp = otio.opentime.RationalTime(0, RATE)
        for clip in track:
            dur = clip.duration()
            if not isinstance(clip, otio.schema.Gap):
                clip_enabled = getattr(clip, 'enabled', True)
                all_clips.append({
                    'track_idx': track_idx,
                    'clip': clip,
                    'is_gap': False,
                    'clip_enabled': clip_enabled,
                    'tl_start': tp.value,
                    'tl_end': (tp + dur).value,
                })
            tp = tp + dur
        src_dur = max(src_dur, tp.value)

    change_pts = {0, int(src_dur)}
    for c in all_clips:
        change_pts.add(int(c['tl_start']))
        change_pts.add(int(c['tl_end']))
    change_pts = sorted(p for p in change_pts if 0 <= p <= src_dur)

    # For each clip, compute its visible sub-ranges.
    per_track_segments = {}  # track_idx -> list of (start, end, clip)
    for c in all_clips:
        if not c['clip_enabled']:
            continue
        track_idx = c['track_idx']
        # Find enabled clips on higher tracks that cover any part of [c['tl_start'], c['tl_end']))
        # Then subtract their coverage from the clip's range.
        coverage = []  # list of (start, end) of higher-track clips covering this one
        for c2 in all_clips:
            if c2['track_idx'] <= track_idx:
                continue
            if not c2['clip_enabled']:
                continue
            # overlap?
            ov_start = max(c['tl_start'], c2['tl_start'])
            ov_end = min(c['tl_end'], c2['tl_end'])
            if ov_start < ov_end:
                coverage.append((ov_start, ov_end))

        # Subtract coverage from clip's range
        if not coverage:
            segs = [(c['tl_start'], c['tl_end'])]
        else:
            coverage.sort()
            merged = []
            for s, e in coverage:
                if merged and s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            segs = []
            cur = c['tl_start']
            for s, e in merged:
                if cur < s:
                    segs.append((cur, s))
                cur = max(cur, e)
            if cur < c['tl_end']:
                segs.append((cur, c['tl_end']))

        for s, e in segs:
            if e > s:
                per_track_segments.setdefault(track_idx, []).append((s, e, c['clip']))

    # Determine which tracks have any visible content (in display order)
    track_order = [t for t in track_order_candidates if t in per_track_segments]

    return per_track_segments, track_order, int(src_dur)


def _clip_identity(clip):
    mr = clip.media_reference
    target = str(mr.target_url) if mr and mr.target_url else ''
    sr = clip.source_range
    return (
        clip.name,
        target,
        (sr.start_time.value, sr.duration.value) if sr else None,
    )


def flatten(input_path, output_path):
    tl = otio.adapters.read_from_file(str(input_path))
    per_track_segments, track_order, src_dur = parse_visible(tl)

    total_visible_segments = sum(len(v) for v in per_track_segments.values())
    print(f"  Source duration: {src_dur} frames ({src_dur / RATE:.2f}s)")
    print(f"  Visible tracks: {len(track_order)} (highest -> lowest)")
    print(f"  Total visible segments: {total_visible_segments}")

    new_tl = otio.schema.Timeline(name=(tl.name or "flattened") + " (visible)")
    out_tracks = []
    for src_track_idx in track_order:
        # Source track name fallback
        src_track_name = None
        for tidx, t in enumerate(tl.tracks):
            if tidx == src_track_idx:
                src_track_name = t.name
                break
        # Use the source track name as-is (it's already like "V1 - Primary")
        out_track = otio.schema.Track(
            name=src_track_name or f"V{src_track_idx + 1} - Flattened",
            kind=otio.schema.TrackKind.Video,
        )
        # Build segments with gaps between them
        segs = sorted(per_track_segments[src_track_idx], key=lambda x: x[0])
        cursor = 0
        for s, e, clip in segs:
            if cursor < s:
                gap_dur = s - cursor
                out_track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, RATE),
                    duration=otio.opentime.RationalTime(gap_dur, RATE),
                )))
            seg_dur = e - s
            sr = clip.source_range
            src_start_val = sr.start_time.value if sr else 0
            clip_tl_start = _clip_timeline_start(clip, src_track_idx, tl)
            if clip_tl_start is None:
                # Fallback: just emit at offset 0 of source
                seg_src_start = src_start_val
            else:
                offset = s - clip_tl_start
                seg_src_start = src_start_val + offset
            new_clip = otio.schema.Clip(
                name=clip.name,
                media_reference=clip.media_reference,
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(seg_src_start, RATE),
                    duration=otio.opentime.RationalTime(seg_dur, RATE),
                ),
                metadata=dict(clip.metadata) if clip.metadata else {},
            )
            new_clip.enabled = True
            out_track.append(new_clip)
            cursor = e
        if cursor < src_dur:
            gap_dur = src_dur - cursor
            out_track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, RATE),
                duration=otio.opentime.RationalTime(gap_dur, RATE),
            )))
        new_tl.tracks.append(out_track)
        out_tracks.append(out_track)

    # Audio: preserve voiceover
    a_track = None
    audio_clones = 0
    for track in tl.tracks:
        if track.kind == otio.schema.TrackKind.Audio:
            if a_track is None:
                a_track = otio.schema.Track(name="A1 - Voiceover", kind=otio.schema.TrackKind.Audio)
            for child in track:
                if isinstance(child, otio.schema.Clip):
                    new_clip = otio.schema.Clip(
                        name=child.name,
                        media_reference=child.media_reference,
                        source_range=child.source_range,
                        metadata=dict(child.metadata) if child.metadata else {},
                    )
                    new_clip.enabled = getattr(child, 'enabled', True)
                    a_track.append(new_clip)
                    audio_clones += 1
    if a_track is not None:
        new_tl.tracks.append(a_track)

    # The track order in tl.tracks: index 0 = lowest track = visually BOTTOM.
    # We appended in track_order (lowest first). In OTIO, the FIRST track in
    # tl.tracks is the bottom, the LAST track is the top. So we want our
    # appended order reversed so the highest source track lands at the end
    # (top). Use clear/extend to avoid Stack reassignment.
    reversed_tracks = list(reversed(new_tl.tracks))
    new_tl.tracks.clear()
    new_tl.tracks.extend(reversed_tracks)

    otio.adapters.write_to_file(new_tl, str(output_path))
    print(f"  Wrote {len(track_order)} video tracks ({total_visible_segments} visible segments) "
          f"+ {audio_clones} audio clips -> {output_path}")
    return total_visible_segments, audio_clones


def _clip_timeline_start(clip, src_track_idx, tl):
    """Compute the timeline start frame of a clip on a specific source track."""
    cur = otio.opentime.RationalTime(0, RATE)
    for tidx, t in enumerate(tl.tracks):
        if tidx != src_track_idx:
            cur = otio.opentime.RationalTime(0, RATE)  # reset, we'll re-walk
            continue
        for child in t:
            if child is clip or (isinstance(child, otio.schema.Clip)
                                  and child.name == clip.name
                                  and child.source_range == clip.source_range):
                return cur.value
            cur = cur + child.duration()
        break
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_name(f"{input_path.stem}_visible.otio")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    flatten(input_path, output_path)


if __name__ == '__main__':
    main()
