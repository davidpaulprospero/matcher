#!/usr/bin/env python3
"""
New Project Setup with Google Drive Voiceover Download

Creates a project folder, downloads voiceover from Google Drive links
in a Google Doc, combines them, and starts the pipeline.

Usage:
    python -m src.cli.newproject <project_name> <editing_for> <google_doc_url>

Example:
    python -m src.cli.newproject 67 Stu https://docs.google.com/document/d/1abc/edit
"""

import os
import re
import sys
import subprocess
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def get_month_name() -> str:
    """Get current month name (January, February, etc.)"""
    return datetime.now().strftime("%B")


def get_date_suffix() -> str:
    """Get date suffix for project folder (YYYY-MM-DD)"""
    return datetime.now().strftime("%Y-%m-%d")


def extract_doc_id(url: str) -> str | None:
    """Extract Google Doc ID from URL"""
    # Patterns:
    # https://docs.google.com/document/d/DOC_ID/edit
    # https://docs.google.com/document/d/DOC_ID/edit?usp=sharing
    pattern = r'docs\.google\.com/document/d/([a-zA-Z0-9_-]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_google_doc_content(doc_id: str) -> str | None:
    """Fetch Google Doc content as plain text"""
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

    print(f"  Fetching Google Doc content...")

    try:
        with urllib.request.urlopen(export_url, timeout=30) as response:
            content = response.read().decode('utf-8')
            return content
    except urllib.error.HTTPError as e:
        print(f"  ERROR: Failed to fetch doc (HTTP {e.code})")
        print(f"  Make sure the document is publicly accessible or shared with 'Anyone with the link'")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to fetch doc: {e}")
        return None


def extract_drive_links(text: str) -> list[str]:
    """Extract Google Drive file IDs from text"""
    # Pattern for Google Drive file links
    pattern = r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)'
    matches = re.findall(pattern, text)
    return matches


def download_drive_files(file_ids: list[str], output_dir: Path) -> list[Path]:
    """Download files from Google Drive using gdown"""
    try:
        import gdown
    except ImportError:
        print("  ERROR: gdown not installed. Run: pip install gdown")
        return []

    downloaded = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, file_id in enumerate(file_ids, 1):
        url = f'https://drive.google.com/uc?id={file_id}'
        output_path = output_dir / f"Pt{i}_download.mp3"

        print(f"  Downloading file {i}/{len(file_ids)} (ID: {file_id})...")

        try:
            # Skip if already exists
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"    File already exists, skipping download.")
                downloaded.append(output_path)
                continue

            gdown.download(url, str(output_path), quiet=False, fuzzy=True)

            # Verify download
            if output_path.exists() and output_path.stat().st_size > 0:
                downloaded.append(output_path)
            else:
                print(f"    Warning: File is empty.")
        except Exception as e:
            print(f"    Failed to download: {e}")

    return downloaded


def combine_audio_files(audio_files: list[Path], output_path: Path) -> bool:
    """Combine multiple audio files into one using pydub (in order provided)"""
    try:
        from pydub import AudioSegment
    except ImportError:
        print("  ERROR: pydub not installed. Run: pip install pydub")
        return False

    if not audio_files:
        print("  ERROR: No audio files to combine")
        return False

    print(f"\n  Downloads finished. Starting audio combination...")

    try:
        combined = AudioSegment.empty()

        for audio_file in audio_files:  # No sort - preserve download order
            print(f"    Adding {audio_file.name}...")
            segment = AudioSegment.from_file(str(audio_file))
            combined += segment

        # Export as MP3
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n  Exporting combined audio to {output_path.name}...")
        combined.export(str(output_path), format="mp3")

        duration_sec = len(combined) / 1000
        print(f"  Success! Audio combined ({duration_sec / 60:.1f} minutes)")
        return True
    except Exception as e:
        print(f"  Failed to export combined audio: {e}")
        return False


def run_setup_project(project_path: Path) -> bool:
    """Run setup_project.py to create project structure"""
    setup_script = PROJECT_ROOT / "setup_project.py"

    if not setup_script.exists():
        print(f"  ERROR: setup_project.py not found at {setup_script}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(setup_script), str(project_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ERROR: setup_project.py failed:")
            print(result.stderr)
            return False

        # Print setup output (filtered)
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"  {line}")

        return True
    except Exception as e:
        print(f"  ERROR running setup_project.py: {e}")
        return False


def run_pipeline(project_path: Path, voiceover_path: Path) -> bool:
    """Run the voiceover-matcher pipeline"""
    main_script = PROJECT_ROOT / "main.py"

    if not main_script.exists():
        print(f"  ERROR: main.py not found at {main_script}")
        return False

    print("\n" + "=" * 60)
    print("  STARTING PIPELINE")
    print("=" * 60)

    try:
        # Run pipeline with the project and voiceover
        result = subprocess.run(
            [
                sys.executable, str(main_script),
                "--project", str(project_path),
                "--voiceover", str(voiceover_path),
                "--save-keywords"
            ],
            cwd=str(PROJECT_ROOT)
        )

        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR running pipeline: {e}")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 4:
        print("\nUsage: python -m src.cli.newproject <project_name> <editing_for> <google_doc_url>")
        print("\nExample:")
        print('  python -m src.cli.newproject 67 Stu https://docs.google.com/document/d/1abc/edit')
        print('  python -m src.cli.newproject "Episode 5" Client https://docs.google.com/document/d/xyz/edit')
        sys.exit(1)

    project_name = sys.argv[1]
    editing_for = sys.argv[2]
    google_doc_url = sys.argv[3]

    print("\n" + "=" * 60)
    print("  NEW PROJECT SETUP")
    print("=" * 60)
    print(f"  Project:    {project_name}")
    print(f"  Client:     {editing_for}")
    print(f"  Doc URL:    {google_doc_url[:50]}...")

    # Validate Google Doc URL
    doc_id = extract_doc_id(google_doc_url)
    if not doc_id:
        print("\n  ERROR: Invalid Google Doc URL")
        print("  Expected format: https://docs.google.com/document/d/DOC_ID/edit")
        sys.exit(1)

    # Build project path
    base_path = Path(r"E:\Edit Job")
    month = get_month_name()
    date_suffix = get_date_suffix()
    project_folder_name = f"{project_name}__{date_suffix}"

    project_path = base_path / editing_for / month / project_folder_name

    print(f"\n  Project path: {project_path}")

    # Step 1: Fetch Google Doc content
    print("\n" + "-" * 60)
    print("  STEP 1: Fetch Google Doc")
    print("-" * 60)

    doc_content = fetch_google_doc_content(doc_id)
    if not doc_content:
        sys.exit(1)

    # Step 2: Extract Drive links
    print("\n" + "-" * 60)
    print("  STEP 2: Extract Drive Links")
    print("-" * 60)

    file_ids = extract_drive_links(doc_content)
    if not file_ids:
        print("  WARNING: No Google Drive links found in document")
        print("  Creating project without voiceover...")
        file_ids = []
    else:
        print(f"  Found {len(file_ids)} Google Drive links")

    # Step 3: Create project structure
    print("\n" + "-" * 60)
    print("  STEP 3: Create Project Structure")
    print("-" * 60)

    if not run_setup_project(project_path):
        sys.exit(1)

    # Step 4: Download audio files
    if file_ids:
        print("\n" + "-" * 60)
        print("  STEP 4: Download Audio Files")
        print("-" * 60)

        # Use temp dir for downloads, then combine
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            downloaded = download_drive_files(file_ids, temp_path)

            if not downloaded:
                print("  ERROR: No audio files downloaded successfully")
                sys.exit(1)

            # Step 5: Combine audio
            print("\n" + "-" * 60)
            print("  STEP 5: Combine Audio")
            print("-" * 60)

            voiceover_path = project_path / "voiceover" / "voiceover.mp3"
            if not combine_audio_files(downloaded, voiceover_path):
                sys.exit(1)

            print(f"\n  Voiceover saved to: {voiceover_path}")
    else:
        voiceover_path = None

    # Step 6: Start pipeline
    print("\n" + "-" * 60)
    print("  STEP 6: Start Pipeline")
    print("-" * 60)

    if voiceover_path and voiceover_path.exists():
        success = run_pipeline(project_path, voiceover_path)

        if success:
            print("\n" + "=" * 60)
            print("  PROJECT COMPLETE")
            print("=" * 60)
            print(f"  Project: {project_path}")
            print(f"  Output:  {project_path / 'output'}")
        else:
            print("\n  Pipeline failed. You can retry with:")
            print(f'  cd "{project_path}" && run.bat')
    else:
        print("\n  No voiceover to process. Add audio to:")
        print(f"  {project_path / 'voiceover'}")
        print("\n  Then run:")
        print(f'  cd "{project_path}" && run.bat')


if __name__ == "__main__":
    main()
