"""Flatten an OTIO timeline to a single video track of top-most visible clips.

For each unique timeline moment, picks the highest-track-index enabled, non-gap,
non-audio clip covering that moment. Disabled topmost clips emit a Gap (visible
placeholder for transitions) instead of falling back to lower tracks. Iterates
at every clip boundary (start AND end) so the flat V1 has no overlapping clips,
no missing coverage at gaps, and total duration matches the source.

Usage:
    python scripts/flatten-otio-to-single-track.py <input.otio> [output.otio]

Default output: <input>_single.otio
"""
import sys
from pathlib import Path

import opentimelineio as otio


RATE = 30.0


def _timeline_duration(track):
    """Sum clip durations on a track (treats disabled tracks/clips as full length)."""
    end = 0
    tp = otio.opentime.RationalTime(0, RATE)
    for c in track:
        tp = tp + c.duration()
        end = tp.value
    return end


def parse_topmost(tl):
    """Return ordered list of (moment, content) entries covering the full timeline.

    Iterates at every change point (clip start AND clip end across all enabled
    non-audio tracks) so no moment is missed. Each entry:
      - type: 'clip' or 'gap'
      - moment: the timeline moment
      - tl_end: upper bound for the segment (full clip end or next change point)
      - tl_start (clips only): full timeline start of the underlying clip
      - clip / track (clips only)
    """
    all_clips = []
    src_dur = 0
    for track_idx, track in enumerate(tl.tracks):
        if not track.enabled:
            continue
        is_audio = track.kind == otio.schema.TrackKind.Audio
        tp = otio.opentime.RationalTime(0, RATE)
        for clip in track:
            dur = clip.duration()
            if not is_audio:
                all_clips.append({
                    'track_idx': track_idx,
                    'track': track.name,
                    'clip': clip,
                    'is_gap': isinstance(clip, otio.schema.Gap),
                    'clip_enabled': getattr(clip, 'enabled', True),
                    'tl_start': tp.value,
                    'tl_end': (tp + dur).value,
                })
            tp = tp + dur
        if not is_audio:
            src_dur = max(src_dur, tp.value)

    # Build change points: every clip's tl_start AND tl_end, plus 0 and src_dur
    change_pts = {0, int(src_dur)}
    for c in all_clips:
        change_pts.add(int(c['tl_start']))
        change_pts.add(int(c['tl_end']))
    change_pts = sorted(p for p in change_pts if 0 <= p <= src_dur)

    results = []
    for pt in change_pts:
        if pt >= src_dur:
            break
        covering = [c for c in all_clips
                    if c['tl_start'] <= pt < c['tl_end']
                    and not c['is_gap']]
        if not covering:
            # Find next change point to set gap end
            nxt = next((p for p in change_pts if p > pt), int(src_dur))
            results.append({'type': 'gap', 'moment': pt, 'tl_end': nxt})
            continue
        top = max(covering, key=lambda x: x['track_idx'])
        if not top['clip_enabled']:
            results.append({
                'type': 'gap',
                'moment': pt,
                'tl_end': top['tl_end'],
                'track': top['track'],
            })
            continue
        results.append({
            'type': 'clip',
            'moment': pt,
            'tl_start': top['tl_start'],
            'tl_end': top['tl_end'],
            'track': top['track'],
            'clip': top['clip'],
        })
    return results


def _clip_identity(clip):
    mr = clip.media_reference
    target = str(mr.target_url) if mr and mr.target_url else ''
    src_range = clip.source_range
    trim = clip.trimmed_range()
    return (
        clip.name,
        target,
        (src_range.start_time.value, src_range.duration.value) if src_range else None,
        (trim.start_time.value, trim.duration.value) if trim else None,
    )


def flatten(input_path, output_path):
    tl = otio.adapters.read_from_file(str(input_path))
    topmost = parse_topmost(tl)
    print(f"  Found {len(topmost)} top-most moments")

    segments = []
    for entry in topmost:
        moment = entry['moment']
        if entry.get('type') == 'gap':
            if segments and segments[-1].get('type') == 'gap':
                # Adjacent gap — extend prev gap's end to this gap's end
                if segments[-1]['end'] < entry['tl_end']:
                    segments[-1]['end'] = entry['tl_end']
                continue
            if segments and segments[-1].get('type') == 'clip':
                if segments[-1]['end'] > moment:
                    segments[-1]['end'] = moment
                if segments[-1]['end'] < moment:
                    segments.append({'type': 'gap', 'start': segments[-1]['end'], 'end': moment})
            segments.append({'type': 'gap', 'start': moment, 'end': entry['tl_end']})
            continue
        clip = entry['clip']
        if segments and segments[-1].get('type') == 'gap':
            if segments[-1]['end'] > moment:
                segments[-1]['end'] = moment
        if segments and segments[-1].get('type') == 'clip':
            if _clip_identity(segments[-1]['clip']) == _clip_identity(clip):
                if segments[-1]['end'] < entry['tl_end']:
                    segments[-1]['end'] = entry['tl_end']
                continue
            if segments[-1]['end'] > moment:
                segments[-1]['end'] = moment
            if segments[-1]['end'] < moment:
                segments.append({'type': 'gap', 'start': segments[-1]['end'], 'end': moment})
        segments.append({
            'type': 'clip',
            'start': moment,
            'end': entry['tl_end'],
            'tl_start': entry['tl_start'],
            'clip': clip,
        })

    segments = [s for s in segments if s['end'] > s['start']]

    clip_count = sum(1 for s in segments if s['type'] == 'clip')
    gap_count = sum(1 for s in segments if s['type'] == 'gap')
    print(f"  Segments: {clip_count} clips + {gap_count} gaps")

    new_tl = otio.schema.Timeline(name=tl.name or "flattened")
    v_track = otio.schema.Track(name="V1 - Flattened Topmost", kind=otio.schema.TrackKind.Video)
    a_track = otio.schema.Track(name="A1 - Voiceover", kind=otio.schema.TrackKind.Audio)

    for seg in segments:
        if seg['type'] == 'gap':
            dur = seg['end'] - seg['start']
            v_track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, RATE),
                duration=otio.opentime.RationalTime(dur, RATE),
            )))
            continue
        src_clip = seg['clip']
        seg_dur = seg['end'] - seg['start']
        src_start_val = src_clip.source_range.start_time.value
        offset = seg['start'] - seg['tl_start']
        seg_src_start = src_start_val + offset
        new_clip = otio.schema.Clip(
            name=src_clip.name,
            media_reference=src_clip.media_reference,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(seg_src_start, RATE),
                duration=otio.opentime.RationalTime(seg_dur, RATE),
            ),
            metadata=dict(src_clip.metadata) if src_clip.metadata else {},
        )
        new_clip.enabled = True
        v_track.append(new_clip)

    audio_clones = 0
    for track in tl.tracks:
        if track.kind == otio.schema.TrackKind.Audio:
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

    new_tl.tracks.append(v_track)
    if audio_clones > 0:
        new_tl.tracks.append(a_track)

    otio.adapters.write_to_file(new_tl, str(output_path))
    print(f"  Wrote {clip_count} video clips + {gap_count} gaps + {audio_clones} audio clips -> {output_path}")
    return clip_count, audio_clones


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_name(f"{input_path.stem}_single.otio")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    flatten(input_path, output_path)


if __name__ == '__main__':
    main()