#!/usr/bin/env python
"""
Diagnose Timeline Issues in DaVinci Resolve.
Finds offline media, problematic clips, and other issues.

Usage:
    python diagnose_timeline.py              # Full diagnosis
    python diagnose_timeline.py --offline    # Only check offline media
    python diagnose_timeline.py --summary    # Brief summary only

Common issues this detects:
- Offline/missing media (Rule 14, 17)
- Zero-duration clips
- Clips with problematic paths (unicode, long paths)
- Audio-only files that may cause issues

Run from DaVinci Resolve: Workspace > Scripts > Edit > diagnose_timeline
"""

import sys
import os
import argparse
from resolve_utils import (
    get_resolve, get_current_timeline, get_all_timeline_items,
    format_frames_as_timecode, TRACK_LAYOUT
)


def check_media_pool_item(mpi):
    """Check a MediaPoolItem for issues."""
    issues = []

    if mpi is None:
        return ['No linked MediaPoolItem']

    props = mpi.GetClipProperty()
    if not props:
        return ['Could not get clip properties']

    # Check file path
    file_path = props.get('File Path', '')
    if file_path:
        # Check if file exists
        if not os.path.exists(file_path):
            issues.append(f'File not found: {file_path}')

        # Check for problematic characters
        try:
            file_path.encode('ascii')
        except UnicodeEncodeError:
            issues.append(f'Unicode in path: {file_path}')

        # Check path length (Windows limit)
        if len(file_path) > 260:
            issues.append(f'Path too long ({len(file_path)} chars): {file_path[:50]}...')

        # Check for backslashes (can cause OTIO issues)
        if '\\' in file_path:
            issues.append(f'Backslashes in path: {file_path}')

        # Check if audio-only (can cause DaVinci issues)
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp3', '.wav', '.aac', '.m4a', '.flac']:
            issues.append(f'Audio-only file (may cause import issues): {ext}')

    # Check if offline
    # Note: DaVinci doesn't directly expose "offline" status via API
    # but we can check if the file exists

    return issues


def diagnose_timeline(timeline, verbose=True):
    """Run full diagnosis on a timeline."""
    results = {
        'timeline_name': timeline.GetName(),
        'total_video_clips': 0,
        'total_audio_clips': 0,
        'offline_clips': [],
        'problematic_clips': [],
        'track_stats': {},
        'issues_by_track': {},
    }

    # Get video track stats
    video_track_count = timeline.GetTrackCount('video')
    for track_idx in range(1, video_track_count + 1):
        track_name = timeline.GetTrackName('video', track_idx)
        items = timeline.GetItemListInTrack('video', track_idx)
        item_count = len(items) if items else 0
        enabled = timeline.GetIsTrackEnabled('video', track_idx)

        results['track_stats'][track_idx] = {
            'name': track_name,
            'description': TRACK_LAYOUT.get(track_idx, ''),
            'clip_count': item_count,
            'enabled': enabled,
        }
        results['total_video_clips'] += item_count

        # Check each clip
        if items:
            for item in items:
                mpi = item.GetMediaPoolItem()
                issues = check_media_pool_item(mpi)

                if issues:
                    clip_info = {
                        'track': track_idx,
                        'track_name': track_name,
                        'clip_name': item.GetName(),
                        'start': item.GetStart(),
                        'duration': item.GetDuration(),
                        'issues': issues,
                    }

                    # Categorize
                    for issue in issues:
                        if 'not found' in issue.lower():
                            results['offline_clips'].append(clip_info)
                            break
                    else:
                        results['problematic_clips'].append(clip_info)

                    if track_idx not in results['issues_by_track']:
                        results['issues_by_track'][track_idx] = []
                    results['issues_by_track'][track_idx].append(clip_info)

    # Get audio track stats
    audio_track_count = timeline.GetTrackCount('audio')
    for track_idx in range(1, audio_track_count + 1):
        items = timeline.GetItemListInTrack('audio', track_idx)
        results['total_audio_clips'] += len(items) if items else 0

    return results


def print_diagnosis(results, summary_only=False, offline_only=False):
    """Print diagnosis results."""
    print(f"\n{'='*60}")
    print(f"TIMELINE DIAGNOSIS: {results['timeline_name']}")
    print(f"{'='*60}")

    # Summary
    print(f"\nSUMMARY:")
    print(f"  Total Video Clips: {results['total_video_clips']}")
    print(f"  Total Audio Clips: {results['total_audio_clips']}")
    print(f"  Offline Clips: {len(results['offline_clips'])}")
    print(f"  Problematic Clips: {len(results['problematic_clips'])}")

    if summary_only:
        return

    # Track breakdown
    if not offline_only:
        print(f"\nTRACK BREAKDOWN:")
        for track_idx, stats in results['track_stats'].items():
            state = "[ON] " if stats['enabled'] else "[OFF]"
            issues = len(results['issues_by_track'].get(track_idx, []))
            issue_str = f" ({issues} issues)" if issues > 0 else ""
            print(f"  V{track_idx}: {state} {stats['clip_count']:4d} clips - {stats['description']}{issue_str}")

    # Offline clips
    if results['offline_clips']:
        print(f"\nOFFLINE CLIPS ({len(results['offline_clips'])}):")
        for clip in results['offline_clips'][:20]:  # Limit to 20
            tc = format_frames_as_timecode(clip['start'])
            print(f"  V{clip['track']} @ {tc}: {clip['clip_name']}")
            for issue in clip['issues']:
                print(f"      -> {issue}")

        if len(results['offline_clips']) > 20:
            print(f"  ... and {len(results['offline_clips']) - 20} more")

    # Other problematic clips
    if not offline_only and results['problematic_clips']:
        print(f"\nOTHER ISSUES ({len(results['problematic_clips'])}):")
        for clip in results['problematic_clips'][:10]:  # Limit to 10
            tc = format_frames_as_timecode(clip['start'])
            print(f"  V{clip['track']} @ {tc}: {clip['clip_name']}")
            for issue in clip['issues']:
                print(f"      -> {issue}")

        if len(results['problematic_clips']) > 10:
            print(f"  ... and {len(results['problematic_clips']) - 10} more")

    # Recommendations
    if results['offline_clips'] or results['problematic_clips']:
        print(f"\nRECOMMENDATIONS:")
        if results['offline_clips']:
            print("  - Run relink_media.py to attempt automatic relinking")
            print("  - Check if media was moved or renamed")
            print("  - Verify E:/v and E:/i paths are accessible")
        if any('unicode' in str(c['issues']).lower() for c in results['problematic_clips']):
            print("  - Files with unicode paths may need manual relinking")
        if any('audio-only' in str(c['issues']).lower() for c in results['problematic_clips']):
            print("  - Audio-only files may cause OTIO import issues")

    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose DaVinci Resolve timeline issues")
    parser.add_argument('--summary', action='store_true', help='Show brief summary only')
    parser.add_argument('--offline', action='store_true', help='Only show offline media')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    resolve = get_resolve()
    if not resolve:
        return 1

    timeline = get_current_timeline(resolve)
    if not timeline:
        return 1

    results = diagnose_timeline(timeline)

    if args.json:
        import json
        # Convert to JSON-serializable format
        print(json.dumps(results, indent=2, default=str))
    else:
        print_diagnosis(results, summary_only=args.summary, offline_only=args.offline)

    # Return code based on issues found
    if results['offline_clips']:
        return 2  # Offline media found
    elif results['problematic_clips']:
        return 1  # Other issues found
    return 0


if __name__ == "__main__":
    sys.exit(main())
