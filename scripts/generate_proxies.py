#!/usr/bin/env python3
"""Generate Premiere-optimized proxy media for top-notch scrubbing.

Creates intra-frame (all-keyframe) proxies so Premiere can seek to any frame
instantly without decoding a GOP chain. Forces CFR to eliminate VFR scrubbing
jank from YouTube downloads.

Presets (benchmarked on RTX 3060 Ti):
  mjpeg   - MJPEG 540p (default). Fastest decode: 859 fps.
            Each frame is a JPEG. Simplest possible decode. ~5x source size.
  scrub   - H.264 All-Intra NVENC, VBR 2Mbps cap, 540p.
            482 fps decode. Smallest files. ~1.15x source size.
  compact - H.264 Short-GOP (15) NVENC, QP 34, 540p.
            ~0.6x source size. Max 15-frame decode on seek.
  prores  - ProRes Proxy 540p (CPU-only).
            465 fps decode. Industry standard. ~8x source size.

Usage:
  python scripts/generate_proxies.py "E:\\Edit Job\\Stu\\all_segments"
  python scripts/generate_proxies.py "E:\\Edit Job\\Stu" --preset scrub
  python scripts/generate_proxies.py "E:\\Edit Job\\Stu" --preset mjpeg --scale 640:360
  python scripts/generate_proxies.py "E:\\Edit Job\\Stu" --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Presets — tuned from RTX 3060 Ti benchmarks on Stu's segments
# ---------------------------------------------------------------------------

PRESETS = {
    "mjpeg": {
        "label": "MJPEG (859 fps decode)",
        "ext": ".mov",
        "video_args": lambda _bitrate: [
            "-c:v", "mjpeg",
            "-q:v", "12",           # JPEG quality (2=best, 31=worst)
            "-pix_fmt", "yuvj422p",
        ],
        "audio_args": ["-c:a", "aac", "-b:a", "96k", "-ac", "2"],
        "default_bitrate": "",
        "container_note": "MOV/AAC — each frame is a JPEG, fastest decode",
    },
    "scrub": {
        "label": "H.264 All-Intra NVENC (482 fps decode)",
        "ext": ".mp4",
        "video_args": lambda bitrate: [
            "-c:v", "h264_nvenc",
            "-g", "1",              # every frame is a keyframe
            "-preset", "p1",        # fastest NVENC preset
            "-rc", "vbr",
            "-b:v", bitrate,
            "-maxrate", _double_bitrate(bitrate),
            "-bufsize", _double_bitrate(bitrate),
            "-profile:v", "high",
        ],
        "audio_args": ["-c:a", "aac", "-b:a", "96k", "-ac", "2"],
        "default_bitrate": "2M",
        "container_note": "MP4/AAC — all I-frame, smallest proxies",
    },
    "compact": {
        "label": "H.264 Short-GOP NVENC",
        "ext": ".mp4",
        "video_args": lambda _bitrate: [
            "-c:v", "h264_nvenc",
            "-g", "15",             # keyframe every ~0.5s
            "-bf", "0",             # no B-frames (faster decode)
            "-preset", "p1",
            "-rc", "constqp",
            "-qp", "34",
            "-profile:v", "high",
        ],
        "audio_args": ["-c:a", "aac", "-b:a", "64k", "-ac", "1"],
        "default_bitrate": "",
        "container_note": "MP4/AAC — GOP-15, ~0.6x source size",
    },
    "prores": {
        "label": "ProRes Proxy (465 fps decode)",
        "ext": ".mov",
        "video_args": lambda _bitrate: [
            "-c:v", "prores_ks",
            "-profile:v", "0",      # Proxy profile
            "-vendor", "apl0",
            "-pix_fmt", "yuv422p10le",
        ],
        "audio_args": ["-c:a", "pcm_s16le"],
        "default_bitrate": "",
        "container_note": "MOV/PCM — intra, industry standard",
    },
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mxf", ".m4v", ".ts"}


def _double_bitrate(bitrate_str: str) -> str:
    """Double a bitrate string like '2M' -> '4M' or '1500k' -> '3000k'."""
    if not bitrate_str:
        return "4M"
    s = bitrate_str.strip()
    suffix = ""
    while s and not s[-1].isdigit():
        suffix = s[-1] + suffix
        s = s[:-1]
    try:
        return f"{int(s) * 2}{suffix}"
    except ValueError:
        return "4M"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_nvenc() -> bool:
    """Check if NVENC H.264 encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def find_video_files(source: Path, recursive: bool = False) -> list[Path]:
    """Find all video files in source directory."""
    if source.is_file():
        if source.suffix.lower() in VIDEO_EXTENSIONS:
            return [source]
        return []

    files = []
    pattern = "**/*" if recursive else "*"
    for p in source.glob(pattern):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            # Skip files already in a proxy directory
            if "_proxy" in p.parent.name.lower() or "proxy" == p.parent.name.lower():
                continue
            files.append(p)

    return sorted(files)


def build_proxy_path(source_file: Path, source_root: Path, proxy_root: Path,
                     ext: str) -> Path:
    """Build output proxy path preserving relative structure."""
    rel = source_file.relative_to(source_root)
    return proxy_root / rel.with_suffix(ext)


def encode_proxy(
    source: Path,
    output: Path,
    preset_name: str,
    scale: str,
    bitrate: str,
    fps_mode: str = "cfr",
) -> dict:
    """Encode a single proxy file. Returns result dict."""
    t0 = time.monotonic()
    preset = PRESETS[preset_name]

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-fps_mode", fps_mode,
        "-vf", f"scale={scale}:force_original_aspect_ratio=decrease:force_divisible_by=2",
    ]
    cmd.extend(preset["video_args"](bitrate))
    cmd.extend(preset["audio_args"])
    cmd.append(str(output))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            return {
                "source": str(source), "output": str(output),
                "ok": False, "error": result.stderr.strip()[:300],
                "elapsed": elapsed,
            }

        out_size = output.stat().st_size if output.exists() else 0
        return {
            "source": str(source), "output": str(output),
            "ok": True, "elapsed": elapsed, "size": out_size,
        }
    except subprocess.TimeoutExpired:
        return {
            "source": str(source), "output": str(output),
            "ok": False, "error": "Timeout (300s)", "elapsed": 300,
        }
    except Exception as e:
        return {
            "source": str(source), "output": str(output),
            "ok": False, "error": str(e)[:300], "elapsed": time.monotonic() - t0,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Premiere-optimized proxy media for scrubbing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", type=Path,
                        help="Directory (or file) containing video segments")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="mjpeg",
                        help="Encoding preset (default: mjpeg)")
    parser.add_argument("--scale", default="960:540",
                        help="Output resolution W:H (default: 960:540)")
    parser.add_argument("--bitrate", default=None,
                        help="Video bitrate cap for scrub preset (default: 2M)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel encode workers (default: 4)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Proxy output directory (default: <source>_proxy)")
    parser.add_argument("--recursive", action="store_true",
                        help="Scan subdirectories recursively")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode existing proxies")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without encoding")
    parser.add_argument("--fps-mode", default="cfr",
                        choices=["cfr", "passthrough"],
                        help="Frame rate mode (default: cfr to fix VFR issues)")

    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"Error: {source} does not exist")
        sys.exit(1)

    # Determine proxy output root
    default_cache = Path("F:/Premiere Cache")
    if args.output:
        proxy_root = args.output.resolve()
    elif default_cache.exists():
        proxy_root = default_cache / f"{source.name}_proxy"
    elif source.is_file():
        proxy_root = source.parent / f"{source.parent.name}_proxy"
    else:
        proxy_root = source.parent / f"{source.name}_proxy"

    preset_name = args.preset
    preset = PRESETS[preset_name]
    bitrate = args.bitrate or preset["default_bitrate"]

    # NVENC fallback to CPU
    if preset_name in ("scrub", "compact") and not detect_nvenc():
        print("Warning: NVENC not available, falling back to CPU libx264")
        if preset_name == "scrub":
            PRESETS["scrub"]["video_args"] = lambda br: [
                "-c:v", "libx264", "-g", "1", "-preset", "ultrafast",
                "-b:v", br, "-maxrate", _double_bitrate(br),
                "-bufsize", _double_bitrate(br), "-profile:v", "high",
            ]
        else:
            PRESETS["compact"]["video_args"] = lambda _br: [
                "-c:v", "libx264", "-g", "15", "-bf", "0",
                "-preset", "ultrafast", "-crf", "34", "-profile:v", "high",
            ]

    # Find source files
    source_root = source if source.is_dir() else source.parent
    files = find_video_files(source, recursive=args.recursive)

    if not files:
        print(f"No video files found in {source}")
        sys.exit(1)

    # Build work list (skip existing unless --overwrite)
    work = []
    skipped = 0
    for f in files:
        proxy_path = build_proxy_path(f, source_root, proxy_root, preset["ext"])
        if proxy_path.exists() and not args.overwrite:
            skipped += 1
            continue
        work.append((f, proxy_path))

    # Summary
    print(f"Proxy Generator — {preset['label']}")
    print(f"  Source:     {source}")
    print(f"  Output:     {proxy_root}")
    print(f"  Scale:      {args.scale}")
    if bitrate:
        print(f"  Bitrate:    {bitrate}")
    print(f"  FPS mode:   {args.fps_mode}")
    print(f"  Container:  {preset['container_note']}")
    print(f"  Files:      {len(work)} to encode, {skipped} already exist")
    print(f"  Workers:    {args.workers}")
    print()

    if args.dry_run:
        for src, dst in work[:20]:
            print(f"  {src.name} -> {dst.name}")
        if len(work) > 20:
            print(f"  ... and {len(work) - 20} more")
        print(f"\nDry run complete. {len(work)} files would be encoded.")
        return

    if not work:
        print("Nothing to do — all proxies already generated.")
        return

    # Encode
    proxy_root.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    done = 0
    failed = 0
    total_proxy_bytes = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                encode_proxy, src, dst, preset_name, args.scale, bitrate, args.fps_mode,
            ): (src, dst)
            for src, dst in work
        }

        for future in as_completed(futures):
            result = future.result()
            done += 1

            if result["ok"]:
                total_proxy_bytes += result.get("size", 0)
                elapsed_s = f"{result['elapsed']:.1f}s"
                pct = done * 100 // len(work)
                src_name = Path(result["source"]).name
                print(
                    f"\r  [{pct:3d}%] {done}/{len(work)}  "
                    f"{src_name[:50]:<50s}  {elapsed_s}",
                    end="", flush=True,
                )
            else:
                failed += 1
                src_name = Path(result["source"]).name
                print(f"\n  FAIL: {src_name} — {result['error']}")

    elapsed = time.monotonic() - t_start
    proxy_gb = total_proxy_bytes / (1024 ** 3)

    print(f"\n\nDone in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Encoded: {done - failed}/{len(work)}")
    if failed:
        print(f"  Failed:  {failed}")
    print(f"  Size:    {proxy_gb:.1f} GB")
    print(f"  Output:  {proxy_root}")

    if failed == 0:
        print(f"\nPremiere: Select all clips > Right-click > Proxy >")
        print(f"  Attach Proxies > point to:\n  {proxy_root}")


if __name__ == "__main__":
    main()
