#!/usr/bin/env python
"""
Track Manager for DaVinci Resolve.
Enable/disable video tracks according to our V1-V10 layout.

Usage:
    python track_manager.py                    # Show current track status
    python track_manager.py --primary          # Enable V1-V3 only
    python track_manager.py --all              # Enable all tracks
    python track_manager.py --broll            # Enable V1 + V8 (B-roll)
    python track_manager.py --entities         # Enable V1 + V9-V10 (entity media)
    python track_manager.py --enable 1,2,3     # Enable specific tracks
    python track_manager.py --disable 4,5,6    # Disable specific tracks
    python track_manager.py --toggle 8         # Toggle track 8

Run from DaVinci Resolve: Workspace > Scripts > Edit > track_manager
"""

import sys
import argparse
from resolve_utils import get_resolve, get_current_timeline, TRACK_LAYOUT


def get_track_status(timeline):
    """Get enabled/disabled status of all video tracks."""
    track_count = timeline.GetTrackCount("video")
    status = {}

    for i in range(1, track_count + 1):
        enabled = timeline.GetIsTrackEnabled("video", i)
        name = timeline.GetTrackName("video", i)
        description = TRACK_LAYOUT.get(i, f"V{i}")
        status[i] = {
            'enabled': enabled,
            'name': name,
            'description': description
        }

    return status


def print_track_status(timeline):
    """Print current track status."""
    status = get_track_status(timeline)
    print(f"\nTimeline: {timeline.GetName()}")
    print("-" * 50)

    for track_idx, info in status.items():
        state = "[ON] " if info['enabled'] else "[OFF]"
        print(f"  V{track_idx}: {state} {info['description']}")
        if info['name'] != f"Video {track_idx}":
            print(f"         ({info['name']})")

    print("-" * 50)


def set_tracks(timeline, track_indices, enabled):
    """Enable or disable specific tracks."""
    for idx in track_indices:
        if 1 <= idx <= timeline.GetTrackCount("video"):
            success = timeline.SetTrackEnable("video", idx, enabled)
            state = "enabled" if enabled else "disabled"
            if success:
                print(f"  V{idx} {state}")
            else:
                print(f"  V{idx} failed to change")
        else:
            print(f"  V{idx} does not exist (max: {timeline.GetTrackCount('video')})")


def set_preset(timeline, preset_name):
    """Apply a preset track configuration."""
    track_count = timeline.GetTrackCount("video")

    presets = {
        'primary': [1, 2, 3],           # V1-V3 only
        'all': list(range(1, 11)),      # All tracks
        'broll': [1, 8],                # Primary + B-roll
        'entities': [1, 9, 10],         # Primary + entity media
        'diversity': [1, 4, 5, 6, 7],   # Primary + diversity alternatives
        'minimal': [1],                 # V1 only
    }

    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {', '.join(presets.keys())}")
        return

    enabled_tracks = presets[preset_name]

    print(f"\nApplying preset: {preset_name}")
    print("-" * 30)

    for i in range(1, min(track_count + 1, 11)):
        should_enable = i in enabled_tracks
        timeline.SetTrackEnable("video", i, should_enable)
        state = "[ON] " if should_enable else "[OFF]"
        desc = TRACK_LAYOUT.get(i, f"V{i}")
        print(f"  V{i}: {state} {desc}")


def toggle_track(timeline, track_idx):
    """Toggle a specific track."""
    if 1 <= track_idx <= timeline.GetTrackCount("video"):
        current = timeline.GetIsTrackEnabled("video", track_idx)
        new_state = not current
        timeline.SetTrackEnable("video", track_idx, new_state)
        state = "enabled" if new_state else "disabled"
        print(f"  V{track_idx} toggled to {state}")
    else:
        print(f"  V{track_idx} does not exist")


def main():
    parser = argparse.ArgumentParser(description="Manage DaVinci Resolve video tracks")
    parser.add_argument('--primary', action='store_true', help='Enable V1-V3 only')
    parser.add_argument('--all', action='store_true', help='Enable all tracks V1-V10')
    parser.add_argument('--broll', action='store_true', help='Enable V1 + V8 (B-roll)')
    parser.add_argument('--entities', action='store_true', help='Enable V1 + V9-V10 (entity media)')
    parser.add_argument('--diversity', action='store_true', help='Enable V1 + V4-V7 (diversity)')
    parser.add_argument('--minimal', action='store_true', help='Enable V1 only')
    parser.add_argument('--enable', type=str, help='Comma-separated track numbers to enable')
    parser.add_argument('--disable', type=str, help='Comma-separated track numbers to disable')
    parser.add_argument('--toggle', type=int, help='Toggle a specific track')

    args = parser.parse_args()

    resolve = get_resolve()
    if not resolve:
        return 1

    timeline = get_current_timeline(resolve)
    if not timeline:
        return 1

    # Determine action
    if args.primary:
        set_preset(timeline, 'primary')
    elif args.all:
        set_preset(timeline, 'all')
    elif args.broll:
        set_preset(timeline, 'broll')
    elif args.entities:
        set_preset(timeline, 'entities')
    elif args.diversity:
        set_preset(timeline, 'diversity')
    elif args.minimal:
        set_preset(timeline, 'minimal')
    elif args.enable:
        tracks = [int(t.strip()) for t in args.enable.split(',')]
        print("\nEnabling tracks:")
        set_tracks(timeline, tracks, True)
    elif args.disable:
        tracks = [int(t.strip()) for t in args.disable.split(',')]
        print("\nDisabling tracks:")
        set_tracks(timeline, tracks, False)
    elif args.toggle:
        print("\nToggling track:")
        toggle_track(timeline, args.toggle)
    else:
        # No args - just show status
        print_track_status(timeline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
