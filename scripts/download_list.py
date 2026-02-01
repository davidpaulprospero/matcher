#!/usr/bin/env python3
"""
Standalone YouTube downloader with cookie rotation and H264 preference.

Features:
- Downloads videos one-by-one (not batch)
- Auto-discovers and rotates cookies from common locations
- Prefers H264 codec for NLE compatibility
- Resilient to failures (continues with next video)

Usage:
    python scripts/download_list.py videos.txt -o ./downloads
    python scripts/download_list.py videos.txt --cookies cookies1.txt cookies2.txt
    python scripts/download_list.py videos.txt --browser firefox
    python scripts/download_list.py videos.txt --no-auto-cookies  # Disable auto-discovery
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import json
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Common cookie file locations and patterns
COOKIE_SEARCH_PATTERNS = [
    # Current directory
    'cookies.txt',
    'cookies*.txt',
    'youtube_cookies*.txt',
    'yt_cookies*.txt',
    # Cookies subdirectory
    'cookies/cookies*.txt',
    'cookies/*.txt',
    # Home directory
    '~/cookies.txt',
    '~/cookies/*.txt',
    '~/.cookies/*.txt',
    # yt-dlp config locations
    '~/.config/yt-dlp/cookies*.txt',
    '~/AppData/Roaming/yt-dlp/cookies*.txt',  # Windows
]


def auto_discover_cookies() -> List[Path]:
    """
    Auto-discover cookie files from common locations.

    Returns list of valid cookie file paths, sorted by modification time (newest first).
    """
    found_files = set()

    for pattern in COOKIE_SEARCH_PATTERNS:
        # Expand ~ to home directory
        expanded = os.path.expanduser(pattern)

        # Use glob to find matching files
        matches = glob.glob(expanded)
        for match in matches:
            path = Path(match)
            if path.is_file() and path.stat().st_size > 0:
                found_files.add(path.resolve())

    # Sort by modification time (newest first)
    sorted_files = sorted(found_files, key=lambda p: p.stat().st_mtime, reverse=True)

    return sorted_files


def validate_cookie_file(path: Path) -> bool:
    """Check if a cookie file is valid (exists, readable, non-empty)."""
    if not path.exists():
        return False
    if not path.is_file():
        return False
    if path.stat().st_size == 0:
        return False

    # Check if readable
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline()
            # Netscape cookie format check
            if '# Netscape HTTP Cookie File' in first_line or '\t' in first_line:
                return True
            # JSON cookie format
            if first_line.strip().startswith('[') or first_line.strip().startswith('{'):
                return True
    except Exception:
        return False

    return True


@dataclass
class CookieRotator:
    """Round-robin cookie rotation with success/fail tracking."""

    cookie_files: List[Path] = field(default_factory=list)
    browser: Optional[str] = None
    auto_discover: bool = True
    _current_index: int = 0
    _last_success_index: int = -1

    def __post_init__(self):
        # Auto-discover cookies if enabled and none provided
        if self.auto_discover and not self.cookie_files and not self.browser:
            discovered = auto_discover_cookies()
            if discovered:
                logger.info(f"Auto-discovered {len(discovered)} cookie file(s)")
                self.cookie_files = discovered

        # Filter to only existing, valid files
        self.cookie_files = [f for f in self.cookie_files if validate_cookie_file(f)]
        self._build_methods()

    def _build_methods(self):
        """Build the rotation chain."""
        self.methods = []

        # Browser cookies first (if specified)
        if self.browser:
            self.methods.append(('browser', self.browser))

        # Cookie files
        for f in self.cookie_files:
            self.methods.append(('file', str(f)))

        # No-cookies fallback
        self.methods.append(('none', None))

        logger.info(f"Cookie chain: {len(self.methods)} methods")
        for i, (method_type, value) in enumerate(self.methods):
            logger.info(f"  [{i}] {method_type}: {value or '(no cookies)'}")

    @property
    def current(self) -> tuple:
        """Get current cookie method."""
        if not self.methods:
            return ('none', None)
        return self.methods[self._current_index % len(self.methods)]

    def add_to_cmd(self, cmd: List[str]) -> None:
        """Add cookie args to yt-dlp command."""
        method_type, value = self.current

        if method_type == 'browser':
            cmd.extend(['--cookies-from-browser', value])
            logger.debug(f"Using browser cookies: {value}")
        elif method_type == 'file':
            cmd.extend(['--cookies', value])
            logger.debug(f"Using cookie file: {value}")
        # 'none' = no cookies added

    def rotate(self) -> None:
        """Rotate to next method (round-robin)."""
        if len(self.methods) > 1:
            old_idx = self._current_index
            self._current_index = (self._current_index + 1) % len(self.methods)
            old_method = self.methods[old_idx]
            new_method = self.methods[self._current_index]
            logger.info(f"Rotated: {old_method[0]}:{old_method[1]} → {new_method[0]}:{new_method[1]}")

    def mark_success(self) -> None:
        """Mark current method as successful and rotate for next download."""
        self._last_success_index = self._current_index
        self.rotate()

    def mark_failure(self) -> None:
        """Mark current method as failed and rotate to try next."""
        self.rotate()

    def reset(self) -> None:
        """Reset to last successful method (or first if none succeeded)."""
        if self._last_success_index >= 0:
            self._current_index = self._last_success_index
        else:
            self._current_index = 0


def build_format_string(quality: str = 'best', prefer_h264: bool = True) -> str:
    """
    Build yt-dlp format selection string.

    Args:
        quality: 'best', '1080p', '720p', '480p', or 'audio'
        prefer_h264: If True, prefer H264/AVC1 codec (DaVinci-friendly)
    """
    if quality == 'audio':
        return 'bestaudio'

    if quality == 'best':
        if prefer_h264:
            # Prefer h264 (avc1) over vp9/av1 to avoid transcoding
            return 'bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]/bestvideo+bestaudio/best'
        return 'bestvideo+bestaudio/best'

    # Specific quality (720p, 1080p, etc.)
    height = quality.rstrip('p')
    if prefer_h264:
        return f'bestvideo[height<={height}][vcodec^=avc1]+bestaudio/bestvideo[height<={height}]+bestaudio/best[height<={height}]/best'
    return f'bestvideo[height<={height}]+bestaudio/best[height<={height}]/best'


def parse_video_list(input_file: Path) -> List[str]:
    """
    Parse video list file. Supports:
    - Full URLs: https://www.youtube.com/watch?v=VIDEO_ID
    - Short URLs: https://youtu.be/VIDEO_ID
    - Just video IDs: VIDEO_ID
    - Lines starting with # are ignored (comments)
    """
    videos = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract video ID from various URL formats
            if 'youtube.com/watch?v=' in line:
                vid_id = line.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in line:
                vid_id = line.split('youtu.be/')[1].split('?')[0]
            else:
                # Assume it's already a video ID
                vid_id = line

            videos.append(vid_id)

    return videos


def download_video(
    video_id: str,
    output_dir: Path,
    cookie_rotator: CookieRotator,
    quality: str = 'best',
    prefer_h264: bool = True,
    max_retries: int = 3
) -> bool:
    """
    Download a single video with retry and cookie rotation.

    Returns True if successful, False otherwise.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries} for {video_id}")

        cmd = [
            'yt-dlp',
            '--ignore-config',
            '-f', build_format_string(quality, prefer_h264),
            '--merge-output-format', 'mp4',
            '--no-playlist',
            '--write-info-json',
            '--restrict-filenames',
            '--no-overwrites',
            '--socket-timeout', '30',
            '--retries', '5',
            '--fragment-retries', '5',
            '--force-ipv4',
            '--http-chunk-size', '10M',
            '--skip-unavailable-fragments',
            '-o', str(output_dir / '%(title).100s_%(id)s.%(ext)s'),
            '--progress',
            '--newline',
            # Impersonation for TLS fingerprint spoofing (bypass detection)
            '--impersonate', 'Chrome-136:Macos-15',
        ]

        # Add cookie authentication
        cookie_rotator.add_to_cmd(cmd)

        cmd.append(url)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600  # 10 minute timeout per video
            )

            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {video_id}")
                cookie_rotator.mark_success()
                return True

            # Check for auth errors
            stderr = result.stderr.lower()
            if '403' in stderr or 'sign in' in stderr or 'private video' in stderr:
                logger.warning(f"Auth error, rotating cookies...")
                cookie_rotator.mark_failure()
            else:
                logger.warning(f"Download failed: {result.stderr[:200]}")
                cookie_rotator.mark_failure()

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout downloading {video_id}")
            cookie_rotator.mark_failure()
        except Exception as e:
            logger.error(f"Error: {e}")
            cookie_rotator.mark_failure()

    logger.error(f"✗ Failed after {max_retries} attempts: {video_id}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube videos with cookie rotation and H264 preference'
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Text file with video URLs/IDs (one per line)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('./downloads'),
        help='Output directory (default: ./downloads)'
    )
    parser.add_argument(
        '--cookies',
        type=Path,
        nargs='+',
        default=[],
        help='Cookie files to rotate through'
    )
    parser.add_argument(
        '--browser',
        type=str,
        choices=['firefox', 'chrome', 'edge', 'brave', 'opera', 'vivaldi'],
        help='Browser to extract cookies from'
    )
    parser.add_argument(
        '-q', '--quality',
        type=str,
        default='best',
        choices=['best', '1080p', '720p', '480p', '360p', 'audio'],
        help='Video quality (default: best)'
    )
    parser.add_argument(
        '--no-h264',
        action='store_true',
        help='Disable H264 preference (allow VP9/AV1)'
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Max retries per video (default: 3)'
    )
    parser.add_argument(
        '--no-auto-cookies',
        action='store_true',
        help='Disable auto-discovery of cookie files'
    )
    parser.add_argument(
        '--list-cookies',
        action='store_true',
        help='List discovered cookie files and exit'
    )

    args = parser.parse_args()

    # List cookies mode
    if args.list_cookies:
        logger.info("Searching for cookie files...")
        discovered = auto_discover_cookies()
        if discovered:
            logger.info(f"Found {len(discovered)} cookie file(s):")
            for i, f in enumerate(discovered, 1):
                mtime = f.stat().st_mtime
                from datetime import datetime
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                size = f.stat().st_size
                valid = "✓" if validate_cookie_file(f) else "✗"
                logger.info(f"  [{i}] {valid} {f} ({size:,} bytes, {mtime_str})")
        else:
            logger.info("No cookie files found in common locations")
        sys.exit(0)

    # Validate input file
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Parse video list
    videos = parse_video_list(args.input_file)
    if not videos:
        logger.error("No videos found in input file")
        sys.exit(1)

    logger.info(f"Found {len(videos)} videos to download")

    # Initialize cookie rotator
    rotator = CookieRotator(
        cookie_files=args.cookies,
        browser=args.browser,
        auto_discover=not args.no_auto_cookies
    )

    # Download one-by-one
    success_count = 0
    fail_count = 0

    for i, video_id in enumerate(videos, 1):
        logger.info(f"\n[{i}/{len(videos)}] Downloading: {video_id}")

        # Check if already downloaded
        existing = list(args.output.glob(f'*_{video_id}.*'))
        if existing:
            logger.info(f"Already exists: {existing[0].name}")
            success_count += 1
            continue

        success = download_video(
            video_id=video_id,
            output_dir=args.output,
            cookie_rotator=rotator,
            quality=args.quality,
            prefer_h264=not args.no_h264,
            max_retries=args.retries
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

        # Reset rotator for next video (start from last working method)
        rotator.reset()

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Complete: {success_count} succeeded, {fail_count} failed")
    logger.info(f"Output: {args.output.absolute()}")


if __name__ == '__main__':
    main()
