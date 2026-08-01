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

import json
import os
import re
import sys
import subprocess
import tempfile
import urllib.request

from ..downloader.utils import SUBPROCESS_FLAGS
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


def build_gws_drive_context_for_channel(channel: str, account: str = None):
    """Build GWS Drive context for a channel or account."""
    # Add scripts to path for gws_drive imports
    scripts_path = PROJECT_ROOT / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    try:
        import gws_drive
        gws_is_available = gws_drive.gws_is_available
        GwsDriveContext = gws_drive.GwsDriveContext
        _context_from_env = gws_drive._context_from_env
    except ImportError:
        print_warn("GWS module not available")
        return None

    if not gws_is_available():
        print_warn("GWS not available")
        return None

    # Determine account name - prefer explicit account, then fall back to channel mapping
    account_name = None
    if account:
        # Map account name to env file
        account_map = {
            "david": "david", "David": "david",
            "stuart": "stuart", "Stuart": "stuart",
            "pamela": "pamela", "Pamela": "pamela",
        }
        account_name = account_map.get(account, account.lower())
    else:
        # Map channel to account
        channel_account_map = {
            "RRU": "david",
            "DSR": "stuart",
            "JDRP": "david",
            "STU": "david",
            "SAMPLES": "david",
            "DISNEY": "david",
            "CRUISE": "david",
        }
        account_name = channel_account_map.get(channel.upper(), channel.lower())
    # STU accounts live under Stu/, everything else under Degold/
    accounts_parent = "Stu" if channel.upper() == "STU" else "Degold"
    account_file = PROJECT_ROOT / accounts_parent / "accounts" / f"{account_name}.env"

    if not account_file.exists():
        return None

    # Parse env file
    env = {}
    with open(account_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    token = env.get("GOOGLE_WORKSPACE_CLI_TOKEN", "")
    # Token can be empty — gws will use keyring/gcloud auth as fallback
    return _context_from_env(token=token)


def get_date_suffix() -> str:
    """Get date suffix for project folder (YYYY-MM-DD)"""
    return datetime.now().strftime("%Y-%m-%d")


def truncate_title(name: str, max_len: int = 40) -> str:
    """Truncate project name at word boundary."""
    if len(name) <= max_len:
        return name
    truncated = name[:max_len].rsplit(" ", 1)[0].rstrip(" -_")
    return truncated or name[:max_len]


def _build_voiceover_filename(card_id: str | None, title: str | None, ext: str = ".mp3") -> str:
    """Build voiceover filename matching pattern: [cardid]-[first-few-words]-[title].ext"""
    if not card_id or not title:
        return f"voiceover{ext}"
    # Lowercase, keep only alphanumeric and spaces/hyphens, then convert spaces to hyphens
    slug = re.sub(r'[^a-z0-9\s-]', '', title.lower()).strip()
    slug = re.sub(r'[\s-]+', '-', slug)
    # Ensure at least 2 parts after card_id (need 3 total hyphen-separated parts)
    parts = slug.split("-")
    if len(parts) < 2:
        slug = f"{slug}-voiceover"
    # Truncate to keep filename reasonable
    slug = "-".join(slug.split("-")[:8])
    return f"{card_id.lower()}-{slug}{ext}"


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
        "JDRP": "david.env",
        "STU": "david.env",
        "SAMPLES": "david.env",
        "DISNEY": "david.env",
        "CRUISE": "david.env",
    }

    account_file = channel_account_map.get(channel.upper(), "david.env")

    # STU accounts live under Stu/, everything else under Degold/
    accounts_parent = "Stu" if channel.upper() == "STU" else "Degold"
    degold_accounts = PROJECT_ROOT / accounts_parent / "accounts" / account_file
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

    # Find Drive attachment: prefer folders, fall back to direct file links
    drive_url = None
    for a in attachments:
        url = a.get("url", "")
        name = a.get("name", "")
        # Prefer Google Drive folders
        if "drive.google.com" in url and "/folders/" in url:
            drive_url = url
            print_info(f"Found Drive folder: {name}")
            break
    # Fallback: direct Drive file link (e.g. /file/d/XXX/view)
    if not drive_url:
        for a in attachments:
            url = a.get("url", "")
            name = a.get("name", "")
            if "drive.google.com" in url and "/file/d/" in url:
                drive_url = url
                print_info(f"Found Drive file attachment: {name}")
                break
    # Fallback: direct Trello file attachment (audio uploaded to card)
    if not drive_url:
        audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.mp4'}
        for a in attachments:
            url = a.get("url", "")
            name = a.get("name", "")
            if any(name.lower().endswith(ext) for ext in audio_exts):
                drive_url = url
                print_info(f"Found Trello audio attachment: {name}")
                break

    return card_name, drive_url


def download_voiceover_from_drive_folder(drive_folder_url: str, output_dir: Path, channel: str = "RRU", account: str = None) -> Optional[Path]:
    """
    Download voiceover file from Google Drive folder using GWS.

    Args:
        drive_folder_url: URL to the Google Drive folder
        output_dir: Where to save the voiceover
        channel: Channel (RRU, DSR, etc)
        account: Account name (David, Stuart, Pamela) - if not provided, uses channel mapping

    Returns:
        Path to downloaded voiceover file, or None on error
    """
    # Add scripts to path for gws_drive imports
    scripts_path = PROJECT_ROOT / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    try:
        import gws_drive
    except ImportError:
        print_warn("GWS module not available")
        return None

    # Extract folder ID from URL
    folder_id = drive_folder_url.split("/folders/")[-1].split("?")[0]

    print_info(f"Downloading from Drive folder {folder_id} via GWS...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build GWS context for the channel
    context = build_gws_drive_context_for_channel(channel, account)

    if not context:
        print_error("GWS not available. Install: npm install -g @googleworkspace/cli", exit_code=1)
        return None

    try:
        # List files in the folder
        files = gws_drive.list_drive_folder_files(folder_id, context=context)
    except Exception as e:
        print_error(f"GWS list error: {e}")
        return None

    # Find voiceover file — prefer files named "voiceover", fall back to any audio
    audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    voiceover_file_id = None
    voiceover_name = None
    fallback_file_id = None
    fallback_name = None
    for f in files:
        name = f.get("name", "")
        mime = f.get("mimeType", "")
        is_audio = mime.startswith("audio/") or any(name.lower().endswith(ext) for ext in audio_exts)
        if not is_audio:
            continue
        if "voiceover" in name.lower():
            voiceover_file_id = f.get("id")
            voiceover_name = name
            break
        if fallback_file_id is None:
            fallback_file_id = f.get("id")
            fallback_name = name

    if not voiceover_file_id and fallback_file_id:
        voiceover_file_id = fallback_file_id
        voiceover_name = fallback_name
        print_info(f"No file named 'voiceover' found, using audio file: {voiceover_name}")

    if not voiceover_file_id:
        print_warn("No voiceover file found in Drive folder")
        return None

    # Download the file
    voiceover_path = output_dir / "voiceover.mp3"
    try:
        gws_drive.download_drive_file(voiceover_file_id, voiceover_path, context=context)
        print_ok(f"Voiceover: {voiceover_path.name}")
        return voiceover_path
    except Exception as e:
        print_warn(f"GWS download error: {e}")
        return None


def download_voiceover_from_drive_file(drive_file_url: str, output_dir: Path, channel: str = "RRU", account: str = None) -> Optional[Path]:
    """
    Download voiceover from a direct Google Drive file link.

    Args:
        drive_file_url: URL like https://drive.google.com/file/d/FILE_ID/view
        output_dir: Where to save the voiceover
        channel: Channel (RRU, DSR, etc)
        account: Account name

    Returns:
        Path to downloaded voiceover file, or None on error
    """
    scripts_path = PROJECT_ROOT / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    try:
        import gws_drive
    except ImportError:
        print_warn("GWS module not available")
        return None

    # Extract file ID from URL
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', drive_file_url)
    if not match:
        print_warn(f"Could not extract file ID from: {drive_file_url}")
        return None

    file_id = match.group(1)
    print_info(f"Downloading Drive file {file_id} via GWS...")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = build_gws_drive_context_for_channel(channel, account)
    if not context:
        print_error("GWS not available. Install: npm install -g @googleworkspace/cli", exit_code=1)
        return None

    voiceover_path = output_dir / "voiceover.mp3"
    try:
        gws_drive.download_drive_file(file_id, voiceover_path, context=context)
        print_ok(f"Voiceover: {voiceover_path.name}")
        return voiceover_path
    except Exception as e:
        print_warn(f"GWS download error: {e}")
        return None


def download_drive_files(file_ids: list[str], output_dir: Path, channel: str = "RRU", account: str = None) -> list[Path]:
    """Download files from Google Drive using GWS CLI."""
    scripts_path = PROJECT_ROOT / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    import gws_drive

    context = build_gws_drive_context_for_channel(channel, account)
    if not context:
        print_error("GWS not available. Install: npm install -g @googleworkspace/cli", exit_code=1)
        return []

    downloaded = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, file_id in enumerate(file_ids, 1):
        output_path = output_dir / f"Pt{i}_download.mp3"

        print_info(f"Downloading file {i}/{len(file_ids)} (ID: {file_id})")

        try:
            # Skip if already exists
            if output_path.exists() and output_path.stat().st_size > 0:
                print_ok(f"File already exists, skipping download.")
                downloaded.append(output_path)
                continue

            gws_drive.download_drive_file(file_id, output_path, context=context, timeout_seconds=600)

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
            errors='replace',
            **SUBPROCESS_FLAGS
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
            cwd=str(PROJECT_ROOT),
            **SUBPROCESS_FLAGS
        )

        return result.returncode == 0
    except Exception as e:
        print_error(f"Error running pipeline: {e}", exit_code=1)
        return False


def main():
    """Main entry point"""
    # Parse optional arguments
    account = None
    card_id = None
    positional_args = []

    no_pipeline = False

    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--account" and i + 2 < len(sys.argv):
            account = sys.argv[i + 2]
        elif arg == "--card-id" and i + 2 < len(sys.argv):
            card_id = sys.argv[i + 2]
        elif arg == "--no-pipeline":
            no_pipeline = True
        elif not arg.startswith("--"):
            positional_args.append(arg)

    if len(positional_args) < 3:
        print_info("Usage: python -m src.cli.newproject <project_name> <channel> <url> [--account NAME] [--card-id ID]")
        print_info("")
        print_info("Examples:")
        print_info('  # From Trello card (recommended)')
        print_info('  python -m src.cli.newproject "Breaking News" RRU https://trello.com/c/FUVH0Ah6 --account David')
        print_info("")
        print_info('  # From Google Doc')
        print_info('  python -m src.cli.newproject "Episode 67" RRU https://docs.google.com/document/d/1abc/edit')
        print_info("")
        print_info("Channels:")
        print_info("  RRU (RennReports)")
        print_info("  DSR (DeepSeaReports)")
        print_info("  JDRP (JournalOfDrunkPeople)")
        print_info("")
        print_info("Options:")
        print_info("  --account NAME   Account name (David, Stuart, Pamela)")
        print_info("  --card-id ID     Trello card ID for naming")
        print_info("  --no-pipeline    Create project and download VO only, do not start pipeline")
        sys.exit(1)

    project_name = positional_args[0]
    channel = positional_args[1]
    url = positional_args[2]

    # Use account if provided, otherwise get from environment
    account_name = account

    # Map channel code to folder name
    channel_map = {
        "RRU": "RennReports",
        "DSR": "DeepSeaReports",
        "JDRP": "JournalOfDrunkPeople",
        "STU": "Stu",
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

        # Get card info and voiceover Drive URL (folder or file)
        card_name, drive_url = get_trello_card_info(card_id, channel)
        if not card_name:
            sys.exit(1)

        # Determine if this is a folder, direct file link, or Trello attachment
        drive_folder_url = None
        drive_file_url = None
        trello_attachment_url = None
        if drive_url and "/folders/" in drive_url:
            drive_folder_url = drive_url
        elif drive_url and "/file/d/" in drive_url:
            drive_file_url = drive_url
        elif drive_url and "trello.com" in drive_url:
            trello_attachment_url = drive_url

        # Use card name as project name if not provided
        if not project_name or project_name == "_":
            project_name = card_name

        print_ok(f"Card: {card_name}")
        if drive_folder_url:
            print_ok(f"Voiceover folder: {drive_folder_url}")
        elif drive_file_url:
            print_ok(f"Voiceover file: {drive_file_url}")
        elif trello_attachment_url:
            print_ok(f"Voiceover Trello attachment: {trello_attachment_url}")
        else:
            print_warn("No voiceover attachment found on card")

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
        drive_file_url = None
        trello_attachment_url = None
        voiceover_path = None

    else:
        print_error("Unknown URL format. Expected Trello or Google Doc URL", exit_code=1)
        sys.exit(1)

    # Build project path: projects/{root}/[CHANNEL]/[PROJECT]__[DATE]
    # STU projects go directly under projects/Stu/ (no channel subfolder)
    _project_root = Path(__file__).resolve().parent.parent.parent
    if channel.upper() == "STU":
        base_path = _project_root / "projects" / "Stu"
    else:
        base_path = _project_root / "projects" / "Degold" / channel_folder
    date_suffix = get_date_suffix()
    # Sanitize project name for Windows filesystem (remove invalid chars)
    safe_name = re.sub(r'[<>:"/\\|?*]', '', project_name).strip()
    truncated_name = truncate_title(safe_name, 40)
    if card_id:
        project_folder_name = f"{card_id}-{truncated_name}__{date_suffix}"
    else:
        project_folder_name = f"{truncated_name}__{date_suffix}"

    project_path = base_path / project_folder_name

    print_info(f"Project path: {project_path}")

    # Step 3: Create project structure
    print_header("STEP 3: Create Project Structure")

    if not run_setup_project(project_path):
        sys.exit(1)

    # Save Trello card metadata for output file naming
    if is_trello and card_id:
        trello_card_path = project_path / "trello_card.json"
        trello_card_data = {
            "card_id": card_id,
            "name": card_name,
            "shortUrl": f"https://trello.com/c/{card_id}",
            "url": url,
        }
        trello_card_path.write_text(json.dumps(trello_card_data, indent=2), encoding="utf-8")
        print_ok(f"Saved card metadata to trello_card.json")

    # Step 4: Download audio files
    # Case 1: Trello with Drive folder
    if is_trello and drive_folder_url:
        print_header("STEP 4: Download Voiceover from Drive Folder")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voiceover_path = download_voiceover_from_drive_folder(drive_folder_url, temp_path, channel, account_name)

            if not voiceover_path:
                print_error("Failed to download voiceover from Drive folder", exit_code=1)
                sys.exit(1)

            # Move to project voiceover folder with proper naming
            vo_filename = _build_voiceover_filename(card_id, project_name, voiceover_path.suffix or ".mp3")
            final_voiceover = project_path / "voiceover" / vo_filename
            import shutil
            shutil.move(str(voiceover_path), str(final_voiceover))
            voiceover_path = final_voiceover

            print_ok(f"Voiceover saved to: {voiceover_path}")

    # Case 1b: Trello with direct Drive file link
    elif is_trello and drive_file_url:
        print_header("STEP 4: Download Voiceover from Drive File")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            voiceover_path = download_voiceover_from_drive_file(drive_file_url, temp_path, channel, account_name)

            if not voiceover_path:
                print_error("Failed to download voiceover from Drive file", exit_code=1)
                sys.exit(1)

            # Move to project voiceover folder with proper naming
            vo_filename = _build_voiceover_filename(card_id, project_name, voiceover_path.suffix or ".mp3")
            final_voiceover = project_path / "voiceover" / vo_filename
            import shutil
            shutil.move(str(voiceover_path), str(final_voiceover))
            voiceover_path = final_voiceover

            print_ok(f"Voiceover saved to: {voiceover_path}")

    # Case 1c: Trello with direct audio file attachment
    elif is_trello and trello_attachment_url:
        print_header("STEP 4: Download Voiceover from Trello Attachment")

        import requests as _requests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "voiceover.mp3"
            print_info(f"Downloading from Trello...")
            # Trello attachments require OAuth authentication
            trello_api_key = os.getenv("TRELLO_API_KEY", "")
            trello_token = os.getenv("TRELLO_TOKEN", "")
            auth_headers = {}
            if trello_api_key and trello_token:
                auth_headers["Authorization"] = f'OAuth oauth_consumer_key="{trello_api_key}", oauth_token="{trello_token}"'
            resp = _requests.get(trello_attachment_url, headers=auth_headers, stream=True, timeout=120)
            if resp.status_code != 200:
                print_error(f"Failed to download Trello attachment (HTTP {resp.status_code})", exit_code=1)
                sys.exit(1)
            with open(temp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if temp_path.stat().st_size == 0:
                print_error("Downloaded file is empty", exit_code=1)
                sys.exit(1)

            # Move to project voiceover folder with proper naming
            vo_filename = _build_voiceover_filename(card_id, project_name, ".mp3")
            final_voiceover = project_path / "voiceover" / vo_filename
            import shutil
            shutil.move(str(temp_path), str(final_voiceover))
            voiceover_path = final_voiceover

            print_ok(f"Voiceover saved to: {voiceover_path}")

    # Case 2: Google Doc with multiple file IDs
    elif file_ids:
        print_header("STEP 4: Download Audio Files")

        # Use temp dir for downloads, then combine
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            downloaded = download_drive_files(file_ids, temp_path, channel=channel, account=account_name)

            if not downloaded:
                print_error("No audio files downloaded successfully", exit_code=1)
                sys.exit(1)

            # Step 5: Combine audio
            print_header("STEP 5: Combine Audio")

            vo_filename = _build_voiceover_filename(card_id, project_name, ".mp3")
            voiceover_path = project_path / "voiceover" / vo_filename
            if not combine_audio_files(downloaded, voiceover_path):
                sys.exit(1)

            print_ok(f"Voiceover saved to: {voiceover_path}")
    else:
        voiceover_path = None

    # Step 6: Start pipeline
    if no_pipeline:
        print_header("PROJECT PREPARED (pipeline skipped)")
        print_ok(f"Project: {project_path}")
        if voiceover_path and voiceover_path.exists():
            print_ok(f"Voiceover: {voiceover_path}")
        print_info("Run pipeline manually when ready:")
        print_info(f'cd "{project_path}" && run.bat')
    elif voiceover_path and voiceover_path.exists():
        print_header("STEP 6: Start Pipeline")
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


def select_best_voiceover_file(voiceover_files: list) -> str | None:
    """
    Select the best voiceover file from a list of candidates.
    Used by pipeline_queue_state.py for launch selection.

    Args:
        voiceover_files: List of Path objects for voiceover files

    Returns:
        Path to best voiceover file, or None if no suitable file found
    """
    if not voiceover_files:
        return None

    # Priority: .mp3 > .wav > .m4a > others
    priority_exts = {'.mp3': 0, '.wav': 1, '.m4a': 2, '.aac': 3}

    best = None
    best_priority = 999

    for f in voiceover_files:
        ext = f.suffix.lower()
        priority = priority_exts.get(ext, 99)
        if priority < best_priority:
            best_priority = priority
            best = f

    return str(best) if best else None


if __name__ == "__main__":
    main()
