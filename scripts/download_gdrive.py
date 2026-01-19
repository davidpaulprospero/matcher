#!/usr/bin/env python3
"""CLI: Download from Google Drive without authentication (gdown with fallbacks)."""

import sys
import io
from pathlib import Path

# Fix Unicode on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gdrive.fallback_downloader import download_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_gdrive.py <google_drive_url> [output_path]")
        print("\nExample:")
        print("  python download_gdrive.py 'https://drive.google.com/open?id=1ABC...' voiceover.mp3")
        sys.exit(1)

    url = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "downloaded_file"

    print("=" * 60)
    print("Google Drive Downloader (No Auth)")
    print("=" * 60)
    print(f"URL: {url[:60]}...")
    print(f"Output: {output_path}")
    print("=" * 60)

    success = download_file(url, output_path, verbose=True)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
