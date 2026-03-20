"""
Auto Lipsync Submitter

Fully automated lipsync submission from a project path.
Reduces manual intervention to just providing the project folder.

Usage:
    python auto_lipsync.py "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project..."

    # Or with options:
    python auto_lipsync.py "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project..." --title "Custom Title"
    python auto_lipsync.py "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project..." --channel DSR
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add Degold to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from channels import CHANNELS, get_channel

# Constants
FORM_URL = "https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c"
AVATAR_TRACKER = SCRIPT_DIR / "avatars" / "avatar_usage.json"
AVATAR_DIR = SCRIPT_DIR / "avatars"

# Channel name to code mapping (folder name -> channel code)
CHANNEL_FROM_FOLDER = {
    "DeepSeaReports": "DSR",
    "RennReports": "RRU",
    "JournalOfDrunkPeople": "JDRP",
}


def get_next_avatar(channel: str) -> str:
    """
    Get next avatar for channel, cycling through V1-V6 randomly but using all once before repeating.
    Creates tracker file if it doesn't exist.
    """
    tracker = {}

    # Load or create tracker
    if AVATAR_TRACKER.exists():
        with open(AVATAR_TRACKER, 'r') as f:
            tracker = json.load(f)

    # Get avatar list for channel
    channel_avatars = {
        "DSR": ["Harold_V1.jpg", "Harold_V2.jpg", "Harold_V3.jpg",
                "Harold_V4.jpg", "Harold_V5.jpg", "Harold_V6.jpg"],
        "RRU": ["RennActor.jpg"],
    }

    avatars = channel_avatars.get(channel, ["RennActor.jpg"])

    # Initialize channel in tracker if needed
    if channel not in tracker:
        tracker[channel] = {
            "avatars": avatars.copy(),
            "used_order": [],
            "current_index": 0,
        }
        # Shuffle initially
        import random
        random.shuffle(tracker[channel]["avatars"])

    data = tracker[channel]
    current_avatars = data["avatars"]
    current_index = data.get("current_index", 0)

    # If we've used all avatars, shuffle and reset
    if current_index >= len(current_avatars):
        import random
        random.shuffle(current_avatars)
        current_index = 0
        data["used_order"] = []

    # Get next avatar
    avatar_file = current_avatars[current_index]
    data["used_order"] = data.get("used_order", []) + [avatar_file]
    data["current_index"] = current_index + 1
    tracker[channel] = data

    # Save tracker
    AVATAR_TRACKER.parent.mkdir(parents=True, exist_ok=True)
    with open(AVATAR_TRACKER, 'w') as f:
        json.dump(tracker, f, indent=2)

    return str(AVATAR_DIR / channel / avatar_file)


def detect_channel_from_path(project_path: str) -> str:
    """
    Auto-detect channel code from project folder path.
    E.g., "DeepSeaReports" -> "DSR"
    Checks current folder and all parent folders.
    """
    path = Path(project_path)

    # Check current folder and all parents
    for folder in [path] + list(path.parents):
        folder_name = folder.name

        # Check against mapping
        for folder_key, channel_code in CHANNEL_FROM_FOLDER.items():
            if folder_key.lower() in folder_name.lower():
                print(f"[INFO] Detected channel: {channel_code} (from folder '{folder_name}')")
                return channel_code

        # Fallback: check if channel code is in folder name
        for channel_code in CHANNELS.keys():
            if channel_code.lower() in folder_name.lower():
                print(f"[INFO] Detected channel: {channel_code} (from folder name)")
                return channel_code

    raise ValueError(f"Could not detect channel from path: {project_path}")


def find_voiceover(project_path: str) -> str:
    """
    Find voiceover file in project folder.
    Looks in voiceover/ subfolder first, then root.
    """
    project = Path(project_path)

    # Common voiceover locations
    candidates = [
        project / "voiceover",
        project,
    ]

    extensions = [".mp3", ".wav", ".m4a", ".mp4"]

    for base in candidates:
        if not base.exists():
            continue

        # Look for voiceover files
        for ext in extensions:
            files = list(base.glob(f"*{ext}"))
            # Prefer files with "voiceover" or project name in name
            for f in sorted(files):
                name_lower = f.name.lower()
                if "voiceover" in name_lower or f.stem.lower() == project.name.lower()[:10]:
                    print(f"[INFO] Found voiceover: {f}")
                    return str(f)

            # Return first match if no preferred one
            if files:
                print(f"[INFO] Found voiceover: {files[0]}")
                return str(files[0])

    raise FileNotFoundError(f"No voiceover file found in {project_path}")


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    import re
    # Replace problematic chars with underscore
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Collapse multiple underscores/spaces
    name = re.sub(r'[_\s]+', '_', name).strip('_')
    # Limit length
    return name[:120]


def trim_audio_to_1min(audio_path: str, output_dir: str = None, title: str = None) -> str:
    """
    Trim audio to first 59 seconds.
    If title is provided, the output file is named after the title.
    Returns path to trimmed file.
    """
    audio_path_obj = Path(audio_path)

    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = audio_path_obj.parent

    if title:
        safe_title = _sanitize_filename(title)
        trimmed_path = output_path / f"{safe_title}{audio_path_obj.suffix}"
    else:
        trimmed_path = output_path / f"{audio_path_obj.stem}_1min{audio_path_obj.suffix}"

    # Skip if already exists
    if trimmed_path.exists():
        print(f"[INFO] Using existing trimmed audio: {trimmed_path}")
        return str(trimmed_path)

    print(f"[INFO] Trimming audio to 1 minute...")

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-t", "59", "-c", "copy", str(trimmed_path), "-y"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
        )
        if result.returncode == 0 and trimmed_path.exists():
            print(f"[OK] Created trimmed audio: {trimmed_path}")
            return str(trimmed_path)
        else:
            print(f"[WARN] Trim failed, using original: {result.stderr[:200] if result.stderr else ''}")
            return audio_path
    except FileNotFoundError:
        print(f"[ERROR] ffmpeg not found")
        raise
    except Exception as e:
        print(f"[ERROR] Trim error: {e}")
        raise


def get_video_title(project_path: str, custom_title: str = None) -> str:
    """
    Extract or generate video title from project.
    Prefers folder name over SRT content.
    """
    if custom_title:
        return custom_title

    project = Path(project_path)

    # Prefer folder name - clean it up
    name = project.name

    # Handle UUID-prefixed names like "3dWWwtJc-How_USS_Charlotte..."
    # Find where the meaningful title starts
    if '-' in name:
        parts = name.split('-')
        # If first part is short (like UUID), join the rest
        if len(parts[0]) <= 5:
            name = '-'.join(parts[1:])

    # Clean up
    name = name.replace('_', ' ').replace('-', ' ')
    name = name.strip()[:100]

    if len(name) > 10:
        return name

    # Fallback: Try SRT file (but not ideal - first caption is often just "March 4, 2026...")
    srt_files = list(project.glob("**/*.srt"))
    if srt_files:
        srt = srt_files[0]
        try:
            with open(srt, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                # Skip date-only first lines, get a better caption
                for i, line in enumerate(lines):
                    if '-->' in line and i + 1 < len(lines):
                        text = lines[i + 1].strip()
                        if text and len(text) > 15 and not text[0].isdigit():
                            return text[:100]
        except:
            pass

    return name or "Untitled"


def copy_to_project_dir(source_path: str, project_dir: str = None) -> str:
    """
    Copy file to project directory for Playwright MCP access.
    Playwright MCP can only access files within the project directory.
    """
    source = Path(source_path)
    project = Path(project_dir) if project_dir else SCRIPT_DIR

    # Use a temp filename
    dest = project / "Degold" / "temp_upload" / source.name

    # Create directory
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy if not already there
    if not dest.exists() or dest.stat().st_size != source.stat().st_size:
        shutil.copy2(source, dest)
        print(f"[INFO] Copied to temp location: {dest}")

    return str(dest)


def submit_via_api(
    video_title: str,
    channel_code: str,
    avatar_path: str,
    audio_path: str,
    drive_folder_id: str = None,
) -> bool:
    """
    Submit lipsync job via API (requests).
    """
    import requests

    # Get drive folder from channel if not provided
    if not drive_folder_id:
        channel = get_channel(channel_code)
        if channel:
            drive_folder_id = channel.drive_folder

    # Prepare files
    files = {
        'field-2': open(avatar_path, 'rb'),
        'field-3': open(audio_path, 'rb'),
    }

    data = {
        'field-0': video_title,
        'field-1': channel_code,
        'field-4': drive_folder_id,
    }

    try:
        response = requests.post(FORM_URL, data=data, files=files, timeout=60)
        print(f"[INFO] API response: {response.status_code}")

        if response.status_code in (200, 201, 202):
            print("[OK] Job submitted successfully!")
            return True
        else:
            print(f"[ERROR] Submission failed: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[ERROR] API error: {e}")
        return False
    finally:
        files['field-2'].close()
        files['field-3'].close()


def main():
    parser = argparse.ArgumentParser(description="Auto-submit lipsync job from project folder")
    parser.add_argument("project_path", help="Path to project folder")
    parser.add_argument("--title", "-t", help="Custom video title (auto-detected if omitted)")
    parser.add_argument("--channel", "-c", help="Channel code (auto-detected if omitted)")
    parser.add_argument("--no-trim", action="store_true", help="Don't trim audio to 1 minute")
    parser.add_argument("--use-api", action="store_true", help="Use API instead of Playwright")

    args = parser.parse_args()

    project_path = args.project_path
    project = Path(project_path)

    if not project.exists():
        print(f"[ERROR] Project not found: {project_path}")
        sys.exit(1)

    print(f"=" * 50)
    print(f"Auto Lipsync Submitter")
    print(f"=" * 50)
    print(f"Project: {project.name}")

    # 1. Detect channel
    channel_code = args.channel
    if not channel_code:
        channel_code = detect_channel_from_path(project_path)
    else:
        channel_code = channel_code.upper()

    if channel_code not in CHANNELS:
        print(f"[ERROR] Unknown channel: {channel_code}")
        sys.exit(1)

    channel_config = get_channel(channel_code)
    print(f"Channel: {channel_code} -> Drive: {channel_config.drive_folder}")

    # 2. Find voiceover
    voiceover_path = find_voiceover(project_path)
    print(f"Voiceover: {voiceover_path}")

    # 4. Get avatar (auto-rotated)
    avatar_path = get_next_avatar(channel_code)
    print(f"Avatar: {avatar_path}")

    # 5. Get title (before trim so we can name the file)
    video_title = get_video_title(project_path, args.title)
    print(f"Title: {video_title}")

    # 3. Trim to 1 minute if needed (named after title)
    if not args.no_trim:
        audio_path = trim_audio_to_1min(voiceover_path, title=video_title)
    else:
        audio_path = voiceover_path
    print(f"Audio: {audio_path}")

    # 6. Copy files to accessible location (for API or Playwright)
    temp_avatar = copy_to_project_dir(avatar_path)
    temp_audio = copy_to_project_dir(audio_path)

    # 7. Submit
    print(f"\n[INFO] Submitting job...")
    success = submit_via_api(
        video_title=video_title,
        channel_code=channel_code,
        avatar_path=temp_avatar,
        audio_path=temp_audio,
        drive_folder_id=channel_config.drive_folder,
    )

    if success:
        print(f"\n[OK] Done! Check Google Drive folder for output.")
    else:
        print(f"\n[ERROR] Submission failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
