#!/usr/bin/env python3
"""
Auto WAN Facecam Generator

Fully automated facecam video generation from a project path.
Auto-detects channel, voiceover, avatar, and title — then generates
a talking-head lip-synced video via the Alibaba WAN API (DashScope).

Reduces manual intervention to just providing the project folder.

Usage:
    python scripts/auto_wan_facecam.py "E:\\Edit Job\\Degold\\DeepSeaReports\\3dWWwtJc-Project..."

    # With options:
    python scripts/auto_wan_facecam.py "E:\\Edit Job\\..." --title "Custom Title"
    python scripts/auto_wan_facecam.py "E:\\Edit Job\\..." --channel DSR
    python scripts/auto_wan_facecam.py "E:\\Edit Job\\..." --resolution 720P --max-duration 60
    python scripts/auto_wan_facecam.py "E:\\Edit Job\\..." --no-trim
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
from pathlib import Path

# Add project root and Degold to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "Degold"))
sys.path.insert(0, str(SCRIPT_DIR))

from channels import CHANNELS, get_channel
from wan_facecam import WanFacecamService, DEFAULT_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AVATAR_DIR = PROJECT_ROOT / "Degold" / "avatars"
STU_AVATAR_DIR = PROJECT_ROOT / "Stu" / "avatars"
AVATAR_TRACKER = AVATAR_DIR / "avatar_usage.json"

# Channel name to code mapping (folder name -> channel code)
CHANNEL_FROM_FOLDER = {
    "DeepSeaReports": "DSR",
    "RennReports": "RRU",
    "JournalOfDrunkPeople": "JDRP",
    "Stu": "STU",
}

# On Windows, prevent subprocess from spawning visible console windows
_SUBPROCESS_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
)


# ---------------------------------------------------------------------------
# Auto-detection helpers (patterned after Degold/auto_lipsync.py)
# ---------------------------------------------------------------------------
def detect_channel_from_path(project_path: str) -> str:
    """
    Auto-detect channel code from project folder path.
    E.g., "DeepSeaReports" -> "DSR"
    Checks current folder and all parent folders.
    """
    path = Path(project_path)

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
    Prefers voiceover_trimmed over voiceover.
    """
    project = Path(project_path)
    candidates = [project / "voiceover", project]
    extensions = [".mp3", ".wav", ".m4a", ".mp4"]

    for base in candidates:
        if not base.exists():
            continue

        for ext in extensions:
            files = list(base.glob(f"*{ext}"))
            # Prefer trimmed voiceover
            for f in sorted(files):
                if "voiceover_trimmed" in f.name.lower():
                    print(f"[INFO] Found trimmed voiceover: {f}")
                    return str(f)
            # Then any voiceover file
            for f in sorted(files):
                if "voiceover" in f.name.lower():
                    print(f"[INFO] Found voiceover: {f}")
                    return str(f)
            # Return first audio match
            if files:
                print(f"[INFO] Found audio: {files[0]}")
                return str(files[0])

    raise FileNotFoundError(f"No voiceover file found in {project_path}")


def get_next_avatar(channel: str) -> str:
    """
    Get next avatar for channel, cycling through available avatars randomly
    but using all once before repeating.
    Creates tracker file if it doesn't exist.
    """
    tracker = {}

    if AVATAR_TRACKER.exists():
        with open(AVATAR_TRACKER, "r") as f:
            tracker = json.load(f)

    # Channel-specific avatar lists
    # DSR: Only V1 works reliably (V2-V6 fail on Degold backend)
    channel_avatars = {
        "DSR": ["Harold_V1.jpg"],
        "RRU": ["RennActor.jpg"],
        "STU": ["Ethan.jpeg"],
    }

    avatars = channel_avatars.get(channel, ["RennActor.jpg"])

    # For single-avatar channels, just return it
    if len(avatars) == 1:
        avatar_file = avatars[0]
    else:
        # Avatar rotation (same pattern as auto_lipsync.py)
        key = f"{channel}_facecam"
        if key not in tracker:
            tracker[key] = {
                "avatars": avatars.copy(),
                "used_order": [],
                "current_index": 0,
            }
            random.shuffle(tracker[key]["avatars"])

        data = tracker[key]
        current_index = data.get("current_index", 0)

        if current_index >= len(data["avatars"]):
            random.shuffle(data["avatars"])
            current_index = 0
            data["used_order"] = []

        avatar_file = data["avatars"][current_index]
        data["used_order"] = data.get("used_order", []) + [avatar_file]
        data["current_index"] = current_index + 1

        AVATAR_TRACKER.parent.mkdir(parents=True, exist_ok=True)
        with open(AVATAR_TRACKER, "w") as f:
            json.dump(tracker, f, indent=2)

    # Resolve avatar path — STU uses its own directory, others use Degold
    if channel == "STU":
        avatar_paths = [
            STU_AVATAR_DIR / avatar_file,
        ]
    else:
        avatar_paths = [
            AVATAR_DIR / channel / avatar_file,
            AVATAR_DIR / avatar_file,
        ]

    for p in avatar_paths:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        f"Avatar not found for {channel}: {avatar_file} "
        f"(searched: {', '.join(str(p) for p in avatar_paths)})"
    )


def get_video_title(project_path: str, custom_title: str = None) -> str:
    """
    Extract or generate video title from project.
    Prefers folder name over SRT content.
    """
    if custom_title:
        return custom_title

    project = Path(project_path)
    name = project.name

    # Handle UUID-prefixed names like "3dWWwtJc-How_USS_Charlotte..."
    if "-" in name:
        parts = name.split("-")
        if len(parts[0]) <= 5:
            name = "-".join(parts[1:])

    name = name.replace("_", " ").replace("-", " ").strip()[:100]

    if len(name) > 10:
        return name

    # Fallback: SRT content
    srt_files = list(project.glob("**/*.srt"))
    if srt_files:
        try:
            with open(srt_files[0], "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "-->" in line and i + 1 < len(lines):
                        text = lines[i + 1].strip()
                        if text and len(text) > 15 and not text[0].isdigit():
                            return text[:100]
        except Exception:
            pass

    return name or "Untitled"


def check_existing_facecam(project_path: str) -> str | None:
    """Check if a facecam video already exists in the project."""
    project = Path(project_path)
    facecam_dir = project / "facecam"

    if not facecam_dir.exists():
        return None

    for ext in [".mp4", ".mkv", ".webm"]:
        videos = list(facecam_dir.glob(f"*{ext}"))
        if videos:
            latest = max(videos, key=lambda p: p.stat().st_mtime)
            return str(latest)

    return None


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"[_\s]+", "_", name).strip("_")
    return name[:120]


def trim_audio(audio_path: str, max_duration: int, output_dir: str = None) -> str:
    """
    Trim audio to max_duration seconds using ffmpeg.
    Returns path to trimmed file (or original if already short enough).
    """
    audio_path_obj = Path(audio_path)

    # Check current duration
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            **_SUBPROCESS_FLAGS,
        )
        current_duration = float(probe.stdout.strip())
        if current_duration <= max_duration:
            print(f"[INFO] Audio is {current_duration:.1f}s (within {max_duration}s limit)")
            return audio_path
    except Exception:
        pass  # Can't determine duration, trim anyway

    out_dir = Path(output_dir) if output_dir else audio_path_obj.parent
    trimmed_path = out_dir / f"{audio_path_obj.stem}_trimmed{audio_path_obj.suffix}"

    if trimmed_path.exists():
        print(f"[INFO] Using existing trimmed audio: {trimmed_path}")
        return str(trimmed_path)

    print(f"[INFO] Trimming audio to {max_duration}s...")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-t",
                str(max_duration),
                "-c",
                "copy",
                str(trimmed_path),
                "-y",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            **_SUBPROCESS_FLAGS,
        )
        if result.returncode == 0 and trimmed_path.exists():
            print(f"[OK] Created trimmed audio: {trimmed_path}")
            return str(trimmed_path)
        else:
            print(
                f"[WARN] Trim failed, using original: "
                f"{result.stderr[:200] if result.stderr else ''}"
            )
            return audio_path
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found")
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate facecam video from project folder"
    )
    parser.add_argument("project_path", help="Path to project folder")
    parser.add_argument(
        "--title", "-t", help="Custom video title (auto-detected if omitted)"
    )
    parser.add_argument(
        "--channel", "-c", help="Channel code (auto-detected if omitted)"
    )
    parser.add_argument("--image", help="Override avatar/presenter image path")
    parser.add_argument("--audio", help="Override voiceover audio path")
    parser.add_argument(
        "--resolution",
        default="480P",
        choices=["480P", "720P", "1080P"],
        help="Video resolution (default: 480P)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Max audio duration to process in seconds (default: 120)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=10,
        help="Seconds per video chunk, 5 or 10 (default: 10)",
    )
    parser.add_argument("--prompt", default=None, help="Custom generation prompt")
    parser.add_argument(
        "--region",
        default="international",
        choices=["international", "beijing"],
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Don't trim audio to --max-duration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if facecam already exists",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    project_path = args.project_path
    project = Path(project_path)

    if not project.exists():
        print(f"[ERROR] Project not found: {project_path}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("Auto WAN Facecam Generator")
    print(f"{'=' * 60}")
    print(f"Project: {project.name}")

    # Step 0: Check for existing facecam
    existing = check_existing_facecam(project_path)
    if existing and not args.force:
        size_mb = Path(existing).stat().st_size / (1024 * 1024)
        print(f"\n[EXISTS] Facecam already exists: {existing} ({size_mb:.1f} MB)")
        print("Use --force to regenerate.")
        sys.exit(0)

    # Step 1: Detect channel
    channel_code = args.channel
    if not channel_code:
        try:
            channel_code = detect_channel_from_path(project_path)
        except ValueError:
            channel_code = None
            print(
                "[WARN] Could not detect channel — avatar must be provided via --image"
            )
    else:
        channel_code = channel_code.upper()

    if channel_code:
        channel_config = get_channel(channel_code)
        if channel_config:
            print(
                f"Channel: {channel_code} ({channel_config.name or channel_code})"
            )
        else:
            print(f"[WARN] Unknown channel: {channel_code} — not in Degold/channels.py")

    # Step 2: Find voiceover
    if args.audio:
        audio_path = args.audio
    else:
        audio_path = find_voiceover(project_path)

    if not Path(audio_path).exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Audio: {audio_path}")

    # Step 2b: Trim audio if needed
    if not args.no_trim:
        audio_path = trim_audio(audio_path, args.max_duration)

    # Step 3: Get avatar
    if args.image:
        image_path = args.image
    elif channel_code:
        image_path = get_next_avatar(channel_code)
    else:
        print("[ERROR] No avatar: provide --image or ensure channel is detected")
        sys.exit(1)

    if not Path(image_path).exists():
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    print(f"Avatar: {image_path}")

    # Step 4: Get title
    video_title = get_video_title(project_path, args.title)
    print(f"Title: {video_title}")

    # Step 5: Check API key
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] DASHSCOPE_API_KEY environment variable not set")
        sys.exit(1)

    print(f"API Key: {api_key[:8]}...")

    # Step 6: Output directory
    output_dir = project / "facecam"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Max Duration: {args.max_duration}s")
    print(f"Chunk Duration: {args.chunk_duration}s")

    # Step 7: Generate
    print(f"\n[INFO] Starting WAN facecam generation...")
    print(f"[INFO] Estimated time: ~{(args.max_duration // args.chunk_duration) * 4} minutes")

    svc = WanFacecamService(
        api_key=api_key,
        resolution=args.resolution,
        chunk_duration=args.chunk_duration,
        region=args.region,
    )

    result = svc.generate(
        audio_path=audio_path,
        image_path=image_path,
        output_dir=str(output_dir),
        prompt=args.prompt or DEFAULT_PROMPT,
        max_duration=args.max_duration,
    )

    # Step 8: Rename output to title-based name
    if result.concatenated_path:
        final_name = f"{_sanitize_filename(video_title)}_facecam.mp4"
        final_path = output_dir / final_name

        if str(final_path) != result.concatenated_path:
            try:
                Path(result.concatenated_path).rename(final_path)
                result = result  # Keep reference
                concat_display = str(final_path)
            except OSError:
                concat_display = result.concatenated_path
        else:
            concat_display = result.concatenated_path
    else:
        concat_display = None

    # Step 9: Report
    print(f"\n{'=' * 60}")
    print("Facecam Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Title:            {video_title}")
    print(f"  Channel:          {channel_code or 'N/A'}")
    print(f"  Chunks generated: {result.chunks_generated}")
    print(f"  Chunks failed:    {result.chunks_failed}")
    print(f"  Total duration:   {result.total_duration:.1f}s")
    print(f"  Resolution:       {result.resolution}")
    print(f"  Models used:      {', '.join(result.models_used)}")

    if concat_display:
        size_mb = Path(concat_display).stat().st_size / (1024 * 1024)
        print(f"  Output:           {concat_display} ({size_mb:.1f} MB)")
    else:
        print("  Output:           No video generated")

    if result.errors:
        print("\n  Errors:")
        for err in result.errors:
            print(f"    - {err}")

    print(f"{'=' * 60}")

    sys.exit(0 if result.chunks_generated > 0 else 1)


if __name__ == "__main__":
    main()
