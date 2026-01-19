"""Skill: Download media from Google Drive (OAuth or anonymous)."""

import sys
from pathlib import Path

from .extract import extract_file_id
from .fallback_downloader import download_file as gdown_download
from .oauth_downloader import (
    download_file as oauth_download,
    find_file_by_name,
    get_drive_service,
)


def download_gdrive(
    url: str,
    output_path: str,
    use_oauth: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Download file from Google Drive.

    Args:
        url: Google Drive URL, file ID, or file name
        output_path: Where to save the file
        use_oauth: If True, use OAuth authentication (can access private files)
                   If False, use gdown (anonymous, public files only)
        verbose: Print progress

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)

    if use_oauth:
        if verbose:
            print("📌 Using OAuth authentication...")
        service = get_drive_service()

        # Try as file ID first
        file_id = extract_file_id(url)

        if file_id:
            if verbose:
                print(f"File ID: {file_id}")
            return oauth_download(service, file_id, str(output_path))
        else:
            # Try as file name
            if verbose:
                print(f"Searching for: {url}")
            files = find_file_by_name(service, url)
            if files:
                file_id = files[0]['id']
                return oauth_download(service, file_id, str(output_path))
            return False
    else:
        if verbose:
            print("📌 Using anonymous download (gdown)...")
        return gdown_download(url, str(output_path), verbose=verbose)


def print_help():
    """Print skill usage help."""
    print("""
Usage: /gdrive-download <url> <output> [--oauth] [--no-verbose]

Download media files from Google Drive to your project.

Arguments:
  <url>              Google Drive URL, file ID, or file name
  <output>           Output file path (relative to project dir)

Options:
  --oauth            Use OAuth authentication (for private files)
  --no-verbose       Suppress progress output

Examples:
  # Anonymous download (public files)
  /gdrive-download 'https://drive.google.com/open?id=1ABC...' voiceover.mp3

  # OAuth download (any file you have access to)
  /gdrive-download 'my_audio_file.mp3' voiceover.mp3 --oauth

  # By file ID
  /gdrive-download '1ABC123xyz' output.mp4
    """).strip()


def main():
    """CLI entry point for skill."""
    if len(sys.argv) < 3:
        print_help()
        sys.exit(1)

    url = sys.argv[1]
    output_path = sys.argv[2]
    use_oauth = '--oauth' in sys.argv
    verbose = '--no-verbose' not in sys.argv

    print("=" * 70)
    print("Google Drive Downloader")
    print("=" * 70)
    print(f"Method: {'OAuth (authenticated)' if use_oauth else 'Anonymous (gdown)'}")
    print(f"Input: {url[:60]}...")
    print(f"Output: {output_path}")
    print("=" * 70)

    success = download_gdrive(url, output_path, use_oauth=use_oauth, verbose=verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
