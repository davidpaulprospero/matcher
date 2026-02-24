"""
AI Lipsync Form Automator

Submits avatar + audio segments to the n8n form at:
https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c

Form fields discovered via inspection:
- field-0: Video Title (text)
- field-1: Channel Code (select: JDRP, DSR, RRU)
- field-2: Avatar Image (file, accepts .jpg,.jpeg,.png,.webp)
- field-3: Audio Segments (file, accepts .mp3,.wav,.m4a, multiple)
- field-4: Google Drive Folder ID (text)

Usage:
    # Direct submission
    python lipsync_automator.py --title "EP42" --channel RRU --avatar face.png --drive-folder "abc123" audio.mp3

    # Using channels.py config
    python lipsync_automator.py --title "EP42" --channel RRU --avatar face.png audio.mp3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

FORM_URL = "https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c"

VALID_CHANNELS = ["JDRP", "DSR", "RRU"]

# Try to import channels config
try:
    from channels import CHANNELS, get_channel
    VALID_CHANNELS = list(CHANNELS.keys())
except ImportError:
    def get_channel(code: str) -> Optional[dict]:
        return None


def trim_audio_to_1min(audio_path: str) -> str:
    """
    Trim audio file to first 60 seconds using ffmpeg.
    Returns path to trimmed file.
    """
    import subprocess
    import tempfile

    audio_path_obj = Path(audio_path)
    suffix = audio_path_obj.suffix

    # Create temp file in same directory
    trimmed_path = audio_path_obj.parent / f"{audio_path_obj.stem}_1min{suffix}"

    # Skip if already trimmed or file is short enough
    if trimmed_path.exists():
        return str(trimmed_path)

    print(f"  Trimming audio to 1 minute...")

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-t", "60", "-c", "copy", str(trimmed_path), "-y"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0 and trimmed_path.exists():
            return str(trimmed_path)
        else:
            print(f"  [WARN] Trim failed, using original: {result.stderr[:200] if result.stderr else ''}")
            return audio_path
    except FileNotFoundError:
        print(f"  [WARN] ffmpeg not found, using original file")
        return audio_path
    except Exception as e:
        print(f"  [WARN] Trim error: {e}, using original file")
        return audio_path


def submit_lipsync_job(
    video_title: str,
    channel_code: str,
    avatar_path: str,
    audio_files: list[str],
    drive_folder_id: str,
    trim_to_1min: bool = True,
) -> tuple[int, str]:
    """
    Submit a lipsync job to the n8n form.

    Args:
        video_title: Title for the video
        channel_code: "JDRP", "DSR", or "RRU"
        avatar_path: Path to avatar image file
        audio_files: List of paths to audio segment files (in order)
        drive_folder_id: Google Drive folder ID for output
        trim_to_1min: If True, trim audio to first 60 seconds (default: True)

    Returns:
        Tuple of (status_code, response_body)
    """
    import requests

    # Validate channel code
    if channel_code not in VALID_CHANNELS:
        raise ValueError(f"Invalid channel code: {channel_code}. Must be one of {VALID_CHANNELS}")

    # Trim audio files to 1 minute if requested
    if trim_to_1min:
        audio_files = [trim_audio_to_1min(f) for f in audio_files]

    # Prepare file uploads
    files = {
        'field-2': open(avatar_path, 'rb'),  # Avatar Image
    }

    # Add all audio files to field-3 (supports multiple files)
    audio_file_handles = []
    for audio_path in audio_files:
        fh = open(audio_path, 'rb')
        audio_file_handles.append(fh)
        files['field-3'] = fh  # Keep reassigning - requests will handle multiple

    # Form data
    data = {
        'field-0': video_title,       # Video Title
        'field-1': channel_code,      # Channel Code
        'field-4': drive_folder_id,   # Google Drive Folder ID
    }

    try:
        response = requests.post(FORM_URL, data=data, files=files, timeout=60)
        return response.status_code, response.text
    finally:
        # Close all file handles
        for fh in [files['field-2']] + audio_file_handles:
            fh.close()


def main():
    parser = argparse.ArgumentParser(
        description="Submit AI Lipsync job to n8n form",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python lipsync_automator.py -t "EP42 - Test" -c JDRP -a avatar.png -d "abc123..." 01_hook.mp3 02_segment.mp3
  python lipsync_automator.py --title "My Video" --channel DSR --avatar face.png --drive-folder "xyz" audio1.mp3 audio2.mp3

Valid channels: {', '.join(VALID_CHANNELS)}
        """
    )

    parser.add_argument('-t', '--title', required=True, help="Video title")
    parser.add_argument('-c', '--channel', required=True, help=f"Channel code ({', '.join(VALID_CHANNELS)})")
    parser.add_argument('-a', '--avatar', required=True, help="Path to avatar image file")
    parser.add_argument('-d', '--drive-folder', default=None, help="Google Drive folder ID (auto-filled from channels.py if omitted)")
    parser.add_argument('--no-trim', action='store_true', help="Don't trim audio to first 60 seconds (default: trim to 1 min)")
    parser.add_argument('audio_files', nargs='+', help="Audio segment files (in order)")

    args = parser.parse_args()

    # Validate channel and auto-fill drive folder from config
    channel_code = args.channel.upper()
    if channel_code not in VALID_CHANNELS:
        print(f"[ERROR] Unknown channel: {channel_code}. Valid channels: {', '.join(VALID_CHANNELS)}")
        sys.exit(1)

    # Auto-fill drive folder from channel config if not provided
    drive_folder = args.drive_folder
    if not drive_folder:
        channel = get_channel(channel_code)
        if channel:
            drive_folder = channel.drive_folder
            print(f"[INFO] Using drive folder from channels.py: {drive_folder}")
        else:
            print("[ERROR] No --drive-folder provided and channel not in channels.py")
            sys.exit(1)

    # Verify avatar exists
    if not Path(args.avatar).exists():
        print(f"[ERROR] Avatar file not found: {args.avatar}")
        sys.exit(1)

    # Verify audio files exist
    missing = [f for f in args.audio_files if not Path(f).exists()]
    if missing:
        print(f"[ERROR] Audio files not found: {missing}")
        sys.exit(1)

    print(f"Submitting lipsync job:")
    print(f"  Title:     {args.title}")
    print(f"  Channel:   {channel_code}")
    print(f"  Avatar:    {args.avatar}")
    print(f"  Drive ID:  {drive_folder}")
    print(f"  Audio:     {args.audio_files}")

    status, body = submit_lipsync_job(
        video_title=args.title,
        channel_code=channel_code,
        avatar_path=args.avatar,
        audio_files=args.audio_files,
        drive_folder_id=drive_folder,
        trim_to_1min=not args.no_trim,
    )

    print(f"\nResponse status: {status}")
    print(f"Response body (first 500 chars): {body[:500]}")

    if status in (200, 201, 202):
        # Check if response indicates success or redirects to waiting page
        if 'form-waiting' in body.lower() or 'success' in body.lower():
            print("\n[OK] Job submitted successfully!")
        else:
            print("\n[OK] Request sent (check Drive for output)")
    else:
        print(f"\n[ERROR] Submission failed with status {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
