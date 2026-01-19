#!/usr/bin/env python
"""
Shared utilities for DaVinci Resolve scripts.
Must be run with DaVinci Resolve open.
"""

import sys
import os


def load_resolve_module():
    """Load the DaVinciResolveScript module from the default location."""
    if sys.platform.startswith("win"):
        expected_path = os.path.join(
            os.getenv('PROGRAMDATA', ''),
            "Blackmagic Design", "DaVinci Resolve", "Support",
            "Developer", "Scripting", "Modules"
        )
    elif sys.platform.startswith("darwin"):
        expected_path = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"
    else:
        expected_path = "/opt/resolve/Developer/Scripting/Modules/"

    if expected_path not in sys.path:
        sys.path.append(expected_path)

    try:
        import DaVinciResolveScript as bmd
        return bmd
    except ImportError as e:
        print(f"Error: Could not import DaVinciResolveScript from {expected_path}")
        print(f"Make sure DaVinci Resolve is running and scripting is enabled.")
        print(f"Details: {e}")
        return None


def get_resolve():
    """Get the Resolve application object."""
    bmd = load_resolve_module()
    if bmd is None:
        return None

    resolve = bmd.scriptapp("Resolve")
    if resolve is None:
        print("Error: Could not connect to DaVinci Resolve.")
        print("Make sure DaVinci Resolve is running.")
        return None

    return resolve


def get_current_project(resolve=None):
    """Get the current project."""
    if resolve is None:
        resolve = get_resolve()
    if resolve is None:
        return None

    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()

    if project is None:
        print("Error: No project is currently open.")
        return None

    return project


def get_current_timeline(resolve=None):
    """Get the current timeline."""
    project = get_current_project(resolve)
    if project is None:
        return None

    timeline = project.GetCurrentTimeline()
    if timeline is None:
        print("Error: No timeline is currently open.")
        return None

    return timeline


def print_timeline_info(timeline):
    """Print basic timeline information."""
    if timeline is None:
        return

    print(f"\nTimeline: {timeline.GetName()}")
    print(f"  Start Frame: {timeline.GetStartFrame()}")
    print(f"  End Frame: {timeline.GetEndFrame()}")
    print(f"  Video Tracks: {timeline.GetTrackCount('video')}")
    print(f"  Audio Tracks: {timeline.GetTrackCount('audio')}")


def get_all_timeline_items(timeline, track_type="video"):
    """Get all items from all tracks of a given type."""
    items = []
    track_count = timeline.GetTrackCount(track_type)

    for i in range(1, track_count + 1):
        track_items = timeline.GetItemListInTrack(track_type, i)
        if track_items:
            for item in track_items:
                items.append({
                    'item': item,
                    'track_type': track_type,
                    'track_index': i,
                    'track_name': timeline.GetTrackName(track_type, i)
                })

    return items


def format_frames_as_timecode(frames, fps=30):
    """Convert frame count to timecode string."""
    total_seconds = frames / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    remaining_frames = int(frames % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{remaining_frames:02d}"


# Track layout from CLAUDE.md
TRACK_LAYOUT = {
    1: "V1 - Primary video",
    2: "V2 - Alternative 1",
    3: "V3 - Alternative 2",
    4: "V4 - Secondary (diversity)",
    5: "V5 - Secondary (diversity)",
    6: "V6 - Secondary (diversity)",
    7: "V7 - Embedding-Diversity",
    8: "V8 - B-roll Only",
    9: "V9 - Entity Images",
    10: "V10 - Stock Videos",
}


if __name__ == "__main__":
    # Test connection
    resolve = get_resolve()
    if resolve:
        print(f"Connected to DaVinci Resolve {resolve.GetVersionString()}")
        project = get_current_project(resolve)
        if project:
            print(f"Current project: {project.GetName()}")
            timeline = get_current_timeline(resolve)
            if timeline:
                print_timeline_info(timeline)
