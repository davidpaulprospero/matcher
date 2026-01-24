#!/usr/bin/env python3
"""
Convert VP9 videos to H.264 for DaVinci Resolve compatibility.

Usage:
    python convert_vp9_to_h264.py "E:/v/ProjectName"
    python convert_vp9_to_h264.py "E:/v/ProjectName" --dry-run
    python convert_vp9_to_h264.py "E:/v/ProjectName" --keep-original
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


def get_video_codec(video_path: Path) -> str | None:
    """Get the video codec using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def convert_to_h264(video_path: Path, keep_original: bool = False) -> tuple[bool, str]:
    """Convert a video to H.264 codec."""
    output_path = video_path.with_suffix(".h264.mp4")

    try:
        # FFmpeg command for high-quality H.264 conversion
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",  # High quality (lower = better, 18-23 is good)
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout per video
        )

        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr[:200]}"

        # Replace original or keep both
        if not keep_original:
            video_path.unlink()
            output_path.rename(video_path)
            return True, f"Converted and replaced: {video_path.name}"
        else:
            return True, f"Converted (kept original): {output_path.name}"

    except subprocess.TimeoutExpired:
        if output_path.exists():
            output_path.unlink()
        return False, "Conversion timeout"
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        return False, str(e)


def find_vp9_videos(project_dir: Path) -> list[Path]:
    """Find all VP9 videos in the project directory."""
    vp9_videos = []
    video_extensions = {".mp4", ".webm", ".mkv"}

    all_videos = [
        f for f in project_dir.rglob("*")
        if f.suffix.lower() in video_extensions
    ]

    print(f"Scanning {len(all_videos)} video files...")

    for i, video in enumerate(all_videos, 1):
        if i % 100 == 0:
            print(f"  Scanned {i}/{len(all_videos)}...")

        codec = get_video_codec(video)
        if codec == "vp9":
            vp9_videos.append(video)

    return vp9_videos


def main():
    parser = argparse.ArgumentParser(description="Convert VP9 videos to H.264")
    parser.add_argument("project_dir", help="Project video directory (e.g., E:/v/ProjectName)")
    parser.add_argument("--dry-run", action="store_true", help="Only scan, don't convert")
    parser.add_argument("--keep-original", action="store_true", help="Keep original VP9 files")
    parser.add_argument("--workers", type=int, default=2, help="Parallel conversion workers (default: 2)")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        print(f"Error: Directory not found: {project_dir}")
        sys.exit(1)

    print(f"=== VP9 to H.264 Converter ===")
    print(f"Project: {project_dir}")
    print()

    # Find VP9 videos
    vp9_videos = find_vp9_videos(project_dir)

    if not vp9_videos:
        print("\n[OK] No VP9 videos found! All videos are already DaVinci-friendly.")
        return

    print(f"\nFound {len(vp9_videos)} VP9 videos:")
    for v in vp9_videos[:10]:
        print(f"  - {v.relative_to(project_dir)}")
    if len(vp9_videos) > 10:
        print(f"  ... and {len(vp9_videos) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No conversions performed.")
        return

    # Calculate total size
    total_size = sum(v.stat().st_size for v in vp9_videos)
    print(f"\nTotal size to convert: {total_size / (1024**3):.2f} GB")

    # Convert videos
    print(f"\nConverting with {args.workers} workers...")
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_to_h264, v, args.keep_original): v
            for v in vp9_videos
        }

        for future in as_completed(futures):
            video = futures[future]
            success, message = future.result()

            if success:
                success_count += 1
                print(f"  [OK] {message}")
            else:
                fail_count += 1
                print(f"  [FAIL] {video.name}: {message}")

    print(f"\n=== Complete ===")
    print(f"Converted: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
