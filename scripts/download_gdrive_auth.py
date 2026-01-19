#!/usr/bin/env python3
"""CLI: Download from Google Drive with OAuth authentication."""

import sys
import io
from pathlib import Path

# Fix Unicode on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gdrive.oauth_downloader import (
    get_drive_service,
    download_file,
    find_file_by_name,
)
from src.gdrive.extract import extract_file_id


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_gdrive_auth.py <file_id_or_url> [output_path]")
        print("\nExamples:")
        print("  # By file ID:")
        print("  python download_gdrive_auth.py '1ABC123xyz' voiceover.mp3")
        print("\n  # By Google Drive URL:")
        print("  python download_gdrive_auth.py 'https://drive.google.com/file/d/1ABC...' voiceover.mp3")
        print("\n  # By file name search:")
        print("  python download_gdrive_auth.py 'my_audio_file.mp3' voiceover.mp3")
        sys.exit(1)

    file_identifier = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "downloaded_file"

    print("=" * 70)
    print("Google Drive Downloader (OAuth Authentication)")
    print("=" * 70)

    # Get authenticated service
    service = get_drive_service()

    print("\n" + "=" * 70)
    print("Downloading file...")
    print("=" * 70)

    # Try as file ID first
    file_id = extract_file_id(file_identifier)

    if file_id:
        print(f"File ID: {file_id}")
        success = download_file(service, file_id, output_path)
    else:
        # Try as file name
        print(f"Searching for: {file_identifier}")
        files = find_file_by_name(service, file_identifier)
        if files:
            if len(files) == 1:
                file_id = files[0]['id']
                success = download_file(service, file_id, output_path)
            else:
                print(f"\nMultiple matches found. Using first match: {files[0]['name']}")
                file_id = files[0]['id']
                success = download_file(service, file_id, output_path)
        else:
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
