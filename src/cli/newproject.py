#!/usr/bin/env python3
"""
New Project Setup with Google Drive Voiceover Download

Creates a project folder, downloads voiceover from:
- Google Drive links in a Google Doc, OR
- Trello card attachments (Script / VO / Description Drive folder)

Usage:
    python -m src.cli.newproject <project_name> <channel> <url>

Examples:
    # From Google Doc
    python -m src.cli.newproject "Episode 67" RennReports https://docs.google.com/document/d/1abc/edit

    # From Trello card
    python -m src.cli.newproject "Breaking News" RennReports https://trello.com/c/FUVH0Ah6
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
from typing import Optional

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from script_utils import print_ok, print_warn, print_error, print_info, print_header


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


def extract_trello_card_id(url: str) -> Optional[str]:
    """Extract Trello card ID from URL"""
    # Patterns:
    # https://trello.com/c/FUVH0Ah6
    # https://trello.com/c/FUVH0Ah6/6-breaking-news
    pattern = r'trello\.com/c/([a-zA-Z0-9]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_trello_card_info(card_id: str, channel: str = "RRU") -> tuple[Optional[str], Optional[str]]:
    """
    Get card name and voiceover Drive folder URL from Trello.

    Args:
        card_id: The Trello card ID
        channel: Channel code (e.g., "RRU", "DSR") to determine which credentials to use

    Returns:
        (card_name, drive_folder_url) or (None, None) on error
    """
    try:
        import requests
    except ImportError:
        print_error("requests not installed. Run: pip install requests", exit_code=1)
        return None, None

    # Map channel to account env file
    channel_account_map = {
        "RRU": "david.env",
        "DSR": "stuart.env",
        "JDRP": "david.env",  # Default to david for now
    }

    account_file = channel_account_map.get(channel.upper(), "david.env")

    # Try to load from Degold accounts
    degold_accounts = PROJECT_ROOT / "Degold" / "accounts" / account_file
    api_key = None
    token = None

    if degold_accounts.exists():
        from dotenv import load_dotenv
        load_dotenv(degold_accounts)
        api_key = os.getenv("TRELLO_API_KEY")
        token = os.getenv("TRELLO_TOKEN")

    if not api_key or not token:
        print_error(f"Trello credentials not found. Set up in Degold/accounts/{account_file}")
        return None, None

    print_info(f"Fetching Trello card {card_id}...")

    # Get card details
    card_resp = requests.get(
        f"https://api.trello.com/1/cards/{card_id}",
        params={"key": api_key, "token": token, "fields": "name"}
    )
    if card_resp.status_code != 200:
        print_error(f"Failed to fetch card (HTTP {card_resp.status_code})")
        return None, None

    card_name = card_resp.json().get("name")

    # Get attachments
    attach_resp = requests.get(
        f"https://api.trello.com/1/cards/{card_id}/attachments",
        params={"key": api_key, "token": token}
    )
    if attach_resp.status_code != 200:
        print_error(f"Failed to fetch attachments (HTTP {attach_resp.status_code})")
        return None, None

    attachments = attach_resp.json()

    # Find "Script / VO / Description" attachment
    drive_folder_url = None
    for a in attachments:
        if a.get("name") == "Script / VO / Description":
            drive_folder_url = a.get("url")
            break

    return card_name, drive_folder_url


def download_voiceover_from_drive_folder(drive_folder_url: str, output_dir: Path) -> Optional[Path]:
    """
    Download voiceover file from Google Drive folder.
    Uses Google Drive API to list files, then downloads individually.

    Returns:
        Path to downloaded voiceover file, or None on error
    """
    # Extract folder ID from URL
    folder_id = drive_folder_url.split("/folders/")[-1].split("?")[0]

    print_info(f"Downloading from Drive folder {folder_id}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get folder contents using Drive API
    try:
        import requests as req

        # Use the folder URL to get contents via Drive API
        # This endpoint lists files in a folder
        api_url = f"https://www.googleapis.com/drive/v3/files"
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "files(id,name,mimeType)"
        }

        # Note: This requires API key. For now, fall back to manual method.
        print_warn("Using gdown to download folder...")

    except Exception as e:
        print_warn(f"API error: {e}")

    # Try a different gdown approach - download files individually by pattern
    # Use fuzzy matching to find Voiceover file
    try:
        import subprocess

        # Try to download the entire folder using gdown --folder with --continue
        result = subprocess.run(
            [sys.executable, "-m", "gdown", "--folder",
             f"https://drive.google.com/drive/folders/{folder_id}",
             "-O", str(output_dir),
             "--no-check-certificate", "--continue"],
            capture_output=True,
            text=True,
            timeout=300
        )

    except Exception as e:
        print_warn(f"gdown error: {e}")

    # Check for downloaded files and their sizes
    voiceover_file = None
    max_size = 0

    for f in output_dir.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            name_lower = f.name.lower()

            # Look for voiceover file with actual content
            if "voiceover" in name_lower and f.suffix.lower() == ".mp3" and size > max_size:
                max_size = size
                voiceover_file = f

    if voiceover_file and max_size > 1000:
        final_path = output_dir / "voiceover.mp3"
        if voiceover_file != final_path:
            import shutil
            shutil.move(str(voiceover_file), str(final_path))
        print_ok(f"Voiceover: {final_path.name} ({max_size} bytes)")
        return final_path

    # Final fallback: show manual download instructions
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    if files:
        print_warn("Downloaded files:")
        for f in files:
            print_warn(f"  {f.name}: {f.stat().st_size} bytes")

    print_error("Auto-download failed. Please download manually:")
    print_info(f"  1. Go to: {drive_folder_url}")
    print_info(f"  2. Download 'Voiceover' .mp3 file")
    print_info(f"  3. Place in: {output_dir}/voiceover.mp3")

    return None


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
    setup_script = SCRIPTS_DIR / "setup_project.py"

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
        print_info("Usage: python -m src.cli.newproject <project_name> <channel> <url>")
        print_info("")
        print_info("Examples:")
        print_info('  # From Trello card (recommended)')
        print_info('  python -m src.cli.newproject "Breaking News" RennReports https://trello.com/c/FUVH0Ah6')
        print_info("")
        print_info('  # From Google Doc')
        print_info('  python -m src.cli.newproject "Episode 67" RennReports https://docs.google.com/document/d/1abc/edit')
        print_info("")
        print_info("Channels:")
        print_info("  RennReports (or RRU)")
        print_info("  DeepSeaReports (or DSR)")
        print_info("  JournalOfDrunkPeople (or JDRP)")
        sys.exit(1)

    project_name = sys.argv[1]
    channel = sys.argv[2]
    url = sys.argv[3]

    # Map channel code to folder name
    channel_map = {
        "RRU": "RennReports",
        "DSR": "DeepSeaReports",
        "JDRP": "JournalOfDrunkPeople",
    }

    # Check if input is a code or full name
    channel_folder = channel_map.get(channel.upper(), channel)

    print_header("NEW PROJECT SETUP")
    print_info(f"Project:    {project_name}")
    print_info(f"Channel:    {channel_folder}")
    print_info(f"URL:        {url[:50]}...")

    # Detect URL type and get voiceover
    is_trello = "trello.com" in url.lower()
    is_google_doc = "docs.google.com" in url.lower()

    if is_trello:
        # Handle Trello card URL
        card_id = extract_trello_card_id(url)
        if not card_id:
            print_error("Invalid Trello URL", exit_code=1)
            print_error("Expected format: https://trello.com/c/CARD_ID", exit_code=1)
            sys.exit(1)

        # Get card info and voiceover folder
        card_name, drive_folder_url = get_trello_card_info(card_id, channel)
        if not card_name:
            sys.exit(1)

        # Use card name as project name if not provided
        if not project_name or project_name == "_":
            project_name = card_name

        print_ok(f"Card: {card_name}")
        if drive_folder_url:
            print_ok(f"Voiceover folder: {drive_folder_url}")
        else:
            print_warn("No voiceover folder found in card attachments")

        voiceover_path = None
        file_ids = []

    elif is_google_doc:
        # Handle Google Doc URL (existing logic)
        doc_id = extract_doc_id(url)
        if not doc_id:
            print_error("Invalid Google Doc URL", exit_code=1)
            print_error("Expected format: https://docs.google.com/document/d/DOC_ID/edit", exit_code=1)
            sys.exit(1)

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

        drive_folder_url = None
        voiceover_path = None

    else:
        print_error("Unknown URL format. Expected Trello or Google Doc URL", exit_code=1)
        sys.exit(1)

    # Build project path: E:\Edit Job\Degold\[CHANNEL]\[PROJECT]__[DATE]
    base_path = Path(r"E:\Edit Job\Degold")
    date_suffix = get_date_suffix()
    project_folder_name = f"{project_name}__{date_suffix}"

    project_path = base_path / channel_folder / project_folder_name

    print_info(f"Project path: {project_path}")

    # Step 3: Create project structure
    print_header("STEP 3: Create Project Structure")

    if not run_setup_project(project_path):
        sys.exit(1)

    # Step 4: Download audio files
    # Case 1: Trello with Drive folder
    if is_trello and drive_folder_url:
        print_header("STEP 4: Download Voiceover from Drive")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voiceover_path = download_voiceover_from_drive_folder(drive_folder_url, temp_path)

            if not voiceover_path:
                print_error("Failed to download voiceover from Drive folder", exit_code=1)
                sys.exit(1)

            # Move to project voiceover folder
            final_voiceover = project_path / "voiceover" / "voiceover.mp3"
            import shutil
            shutil.move(str(voiceover_path), str(final_voiceover))
            voiceover_path = final_voiceover

            print_ok(f"Voiceover saved to: {voiceover_path}")

    # Case 2: Google Doc with multiple file IDs
    elif file_ids:
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
