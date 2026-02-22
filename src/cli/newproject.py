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

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from script_utils import print_ok, print_warn, print_error, print_info, print_header


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

    print_info("Fetching Google Doc content...")

    try:
        with urllib.request.urlopen(export_url, timeout=30) as response:
            content = response.read().decode('utf-8')
            return content
    except urllib.error.HTTPError as e:
        print_error(f"Failed to fetch doc (HTTP {e.code})")
        print_error("Make sure the document is publicly accessible or shared with 'Anyone with the link'", exit_code=1)
        return None
    except Exception as e:
        print_error(f"Failed to fetch doc: {e}", exit_code=1)
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
        print_error("gdown not installed. Run: pip install gdown", exit_code=1)
        return []

    downloaded = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, file_id in enumerate(file_ids, 1):
        url = f'https://drive.google.com/uc?id={file_id}'
        output_path = output_dir / f"Pt{i}_download.mp3"

        print_info(f"Downloading file {i}/{len(file_ids)} (ID: {file_id})")

        try:
            # Skip if already exists
            if output_path.exists() and output_path.stat().st_size > 0:
                print_ok(f"File already exists, skipping download.")
                downloaded.append(output_path)
                continue

            gdown.download(url, str(output_path), quiet=False, fuzzy=True)

            # Verify download
            if output_path.exists() and output_path.stat().st_size > 0:
                downloaded.append(output_path)
            else:
                print_warn("File is empty.")
        except Exception as e:
            print_error(f"Failed to download: {e}", exit_code=1)

    return downloaded


def combine_audio_files(audio_files: list[Path], output_path: Path) -> bool:
    """Combine multiple audio files into one using pydub (in order provided)"""
    try:
        from pydub import AudioSegment
    except ImportError:
        print_error("pydub not installed. Run: pip install pydub", exit_code=1)
        return False

    if not audio_files:
        print_error("No audio files to combine", exit_code=1)
        return False

    print_info("Downloads finished. Starting audio combination...")

    try:
        combined = AudioSegment.empty()

        for audio_file in audio_files:  # No sort - preserve download order
            print_info(f"Adding {audio_file.name}...")
            segment = AudioSegment.from_file(str(audio_file))
            combined += segment

        # Export as MP3
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print_info(f"Exporting combined audio to {output_path.name}...")
        combined.export(str(output_path), format="mp3")

        duration_sec = len(combined) / 1000
        print_ok(f"Audio combined ({duration_sec / 60:.1f} minutes)")
        return True
    except Exception as e:
        print_error(f"Failed to export combined audio: {e}", exit_code=1)
        return False


def run_setup_project(project_path: Path) -> bool:
    """Run setup_project.py to create project structure"""
    setup_script = PROJECT_ROOT / "setup_project.py"

    if not setup_script.exists():
        print_error(f"setup_project.py not found at {setup_script}", exit_code=1)
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(setup_script), str(project_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            print_error("setup_project.py failed:")
            print_error(result.stderr, exit_code=1)
            return False

        # Print setup output (filtered)
        for line in result.stdout.split('\n'):
            if line.strip():
                print_info(line)

        return True
    except Exception as e:
        print_error(f"Error running setup_project.py: {e}", exit_code=1)
        return False


def run_pipeline(project_path: Path, voiceover_path: Path) -> bool:
    """Run the voiceover-matcher pipeline"""
    main_script = PROJECT_ROOT / "main.py"

    if not main_script.exists():
        print_error(f"main.py not found at {main_script}", exit_code=1)
        return False

    print_header("STARTING PIPELINE")

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
        print_error(f"Error running pipeline: {e}", exit_code=1)
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 4:
        print_info("Usage: python -m src.cli.newproject <project_name> <editing_for> <google_doc_url>")
        print_info("Example:")
        print_info('python -m src.cli.newproject 67 Stu https://docs.google.com/document/d/1abc/edit')
        print_info('python -m src.cli.newproject "Episode 5" Client https://docs.google.com/document/d/xyz/edit')
        sys.exit(1)

    project_name = sys.argv[1]
    editing_for = sys.argv[2]
    google_doc_url = sys.argv[3]

    print_header("NEW PROJECT SETUP")
    print_info(f"Project:    {project_name}")
    print_info(f"Client:     {editing_for}")
    print_info(f"Doc URL:    {google_doc_url[:50]}...")

    # Validate Google Doc URL
    doc_id = extract_doc_id(google_doc_url)
    if not doc_id:
        print_error("Invalid Google Doc URL", exit_code=1)
        print_error("Expected format: https://docs.google.com/document/d/DOC_ID/edit", exit_code=1)
        sys.exit(1)

    # Build project path
    base_path = Path(r"E:\Edit Job")
    month = get_month_name()
    date_suffix = get_date_suffix()
    project_folder_name = f"{project_name}__{date_suffix}"

    project_path = base_path / editing_for / month / project_folder_name

    print_info(f"Project path: {project_path}")

    # Step 1: Fetch Google Doc content
    print_header("STEP 1: Fetch Google Doc")

    doc_content = fetch_google_doc_content(doc_id)
    if not doc_content:
        sys.exit(1)

    # Step 2: Extract Drive links
    print_header("STEP 2: Extract Drive Links")

    file_ids = extract_drive_links(doc_content)
    if not file_ids:
        print_warn("No Google Drive links found in document")
        print_info("Creating project without voiceover...")
        file_ids = []
    else:
        print_ok(f"Found {len(file_ids)} Google Drive links")

    # Step 3: Create project structure
    print_header("STEP 3: Create Project Structure")

    if not run_setup_project(project_path):
        sys.exit(1)

    # Step 4: Download audio files
    if file_ids:
        print_header("STEP 4: Download Audio Files")

        # Use temp dir for downloads, then combine
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            downloaded = download_drive_files(file_ids, temp_path)

            if not downloaded:
                print_error("No audio files downloaded successfully", exit_code=1)
                sys.exit(1)

            # Step 5: Combine audio
            print_header("STEP 5: Combine Audio")

            voiceover_path = project_path / "voiceover" / "voiceover.mp3"
            if not combine_audio_files(downloaded, voiceover_path):
                sys.exit(1)

            print_ok(f"Voiceover saved to: {voiceover_path}")
    else:
        voiceover_path = None

    # Step 6: Start pipeline
    print_header("STEP 6: Start Pipeline")

    if voiceover_path and voiceover_path.exists():
        success = run_pipeline(project_path, voiceover_path)

        if success:
            print_header("PROJECT COMPLETE")
            print_ok(f"Project: {project_path}")
            print_ok(f"Output:  {project_path / 'output'}")
        else:
            print_warn("Pipeline failed. You can retry with:")
            print_info(f'cd "{project_path}" && run.bat')
    else:
        print_info("No voiceover to process. Add audio to:")
        print_info(f"{project_path / 'voiceover'}")
        print_info("Then run:")
        print_info(f'cd "{project_path}" && run.bat')


if __name__ == "__main__":
    main()
