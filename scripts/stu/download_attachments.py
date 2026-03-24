"""Download Trello card attachments for a STU (NEW AMERICA) project.

Given a project directory, resolves the matching Trello card and downloads
all attachments (images, videos, clips) into {project}/attachments/.

Usage:
    python scripts/stu/download_attachments.py "E:\\Edit Job\\Stu\\SAMPLE-NEW-AMERICA" --card-id KxQ0SoGh
    python scripts/stu/download_attachments.py "E:\\Edit Job\\Stu\\KxQ0SoGh-title__2026-03-19"
    python scripts/stu/download_attachments.py "E:\\Edit Job\\Stu\\PROJECT" --dry-run
    python scripts/stu/download_attachments.py "E:\\Edit Job\\Stu\\PROJECT" --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

_script_path = os.path.abspath(__file__)
PROJECT_ROOT = Path(_script_path).parent.parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Degold"))
os.chdir(PROJECT_ROOT)

from script_utils import (
    print_ok, print_info, print_warn, print_error, print_header,
    set_verbosity,
)
from local_project_selector import (
    contains_card_id_token,
    extract_project_card_id,
)

DEFAULT_STATE_FILE = PROJECT_ROOT / "Stu" / "pipeline_queue_state.json"
DEFAULT_ACCOUNTS_DIR = PROJECT_ROOT / "Stu" / "accounts"

# Title-matching helpers (same as download_vo.py)
_PROJECT_TITLE_STOPWORDS = {
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with", "just", "now", "ago", "breaking",
}
_PROJECT_NAME_TITLE_MIN_SHARED_TOKENS = 4
_PROJECT_NAME_TITLE_MIN_JACCARD = 0.55


def _title_tokens(value: str) -> set[str]:
    """Tokenize free text for resilient project-title matching."""
    text = str(value or "").strip().lower()
    if not text:
        return set()
    text = re.sub(r"__\d{4}-\d{2}-\d{2}$", "", text)
    text = re.sub(r"^[a-z0-9]{8,24}-", "", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t for t in tokens if len(t) >= 3 and t not in _PROJECT_TITLE_STOPWORDS}


# ---------------------------------------------------------------------------
# Env / auth helpers
# ---------------------------------------------------------------------------

def parse_env_file(path: Path) -> dict[str, str]:
    """Parse .env file without loading global environment."""
    values: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def load_trello_credentials(accounts_dir: Path) -> dict[str, str]:
    """Load Trello API key and token from the first .env file."""
    if not accounts_dir.is_dir():
        print_error(f"Accounts directory not found: {accounts_dir}")
        sys.exit(1)
    for env_file in sorted(accounts_dir.glob("*.env")):
        if env_file.name == "example.env":
            continue
        env_data = parse_env_file(env_file)
        api_key = env_data.get("TRELLO_API_KEY", "")
        token = env_data.get("TRELLO_TOKEN", "")
        if api_key and token:
            return {"api_key": api_key, "token": token}
    print_error(f"No TRELLO_API_KEY/TRELLO_TOKEN found in {accounts_dir}/*.env")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Card ID resolution
# ---------------------------------------------------------------------------

def resolve_card_id(project_dir: Path, state_file: Path) -> str | None:
    """Resolve a project directory to a Trello card shortLink.

    Strategy:
    1. trello_card.json in project dir
    2. Card ID token in folder name (e.g. KxQ0SoGh-title__date)
    3. Title-token Jaccard matching against state file entries
    """
    # Strategy 1: trello_card.json
    stored_id = extract_project_card_id(project_dir)
    if stored_id:
        return stored_id

    # Strategy 2: folder name card ID prefix
    folder_name = project_dir.name
    # Load state to get known card IDs
    state = _load_state(state_file)
    pipelines = state.get("pipelines", {})

    for key in pipelines:
        card_raw = (pipelines[key].get("trello") or {}).get("card_raw") or {}
        short_link = card_raw.get("shortLink", key)
        match_keys = {key, short_link.lower()}
        if contains_card_id_token(folder_name, match_keys):
            return short_link

    # Strategy 3: title-token Jaccard matching
    project_tokens = _title_tokens(folder_name)
    if project_tokens:
        for key, entry in pipelines.items():
            title = str(entry.get("title") or "")
            card_tokens = _title_tokens(title)
            if not card_tokens:
                continue
            shared = len(project_tokens & card_tokens)
            if shared < _PROJECT_NAME_TITLE_MIN_SHARED_TOKENS:
                continue
            union = len(project_tokens | card_tokens)
            if union > 0 and (shared / union) >= _PROJECT_NAME_TITLE_MIN_JACCARD:
                card_raw = (entry.get("trello") or {}).get("card_raw") or {}
                return card_raw.get("shortLink", key)

    return None


def _load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {}
    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Attachment fetching and downloading
# ---------------------------------------------------------------------------

def fetch_attachments(card_id: str, creds: dict[str, str]) -> list[dict]:
    """Fetch attachment list from Trello API."""
    response = requests.get(
        f"https://api.trello.com/1/cards/{card_id}/attachments",
        params={"key": creds["api_key"], "token": creds["token"]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _trello_auth_headers(creds: dict[str, str]) -> dict[str, str]:
    """Build OAuth Authorization header for Trello attachment downloads."""
    return {
        "Authorization": (
            f'OAuth oauth_consumer_key="{creds["api_key"]}", '
            f'oauth_token="{creds["token"]}"'
        ),
    }


def _is_trello_hosted(url: str) -> bool:
    """Check if a URL is hosted on Trello (needs auth params)."""
    return "trello.com" in url or "trello-attachments" in url


_MIME_TO_EXT = {
    "video/quicktime": ".mov",
    "video/mp4": ".mp4",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp4": ".m4a",
    "application/pdf": ".pdf",
}


def _ensure_extension(name: str, mime_type: str) -> str:
    """Add a file extension based on MIME type if the filename has none."""
    if Path(name).suffix:
        return name
    ext = _MIME_TO_EXT.get(mime_type.lower(), "")
    if not ext and "/" in mime_type:
        # Fallback: use subtype (e.g. "image/png" -> ".png")
        subtype = mime_type.split("/", 1)[1].split(";")[0].strip()
        if re.match(r"^[a-z0-9]+$", subtype):
            ext = f".{subtype}"
    return f"{name}{ext}" if ext else name


def _human_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def download_attachment(
    att: dict,
    output_dir: Path,
    creds: dict[str, str],
    *,
    force: bool = False,
) -> dict:
    """Download a single attachment. Returns result dict."""
    raw_name = att.get("name", "unnamed")
    url = att.get("url", "")
    att_bytes = att.get("bytes")
    mime_type = att.get("mimeType", "")
    name = _ensure_extension(raw_name, mime_type)

    if not url:
        return {"name": name, "status": "skipped", "reason": "no_url"}

    # External URLs (Google Drive links, etc.) — log but don't download
    is_external = not _is_trello_hosted(url) and "trello" not in url.lower()
    if is_external:
        return {
            "name": name,
            "status": "external",
            "url": url,
            "reason": "external_url",
        }

    output_path = output_dir / name

    # Skip if already exists with matching size
    if not force and output_path.exists() and att_bytes:
        local_size = output_path.stat().st_size
        if local_size == att_bytes:
            return {
                "name": name,
                "status": "skipped",
                "reason": "exists",
                "size": att_bytes,
            }

    # Download — Trello-hosted files need OAuth header on api.trello.com
    if _is_trello_hosted(url):
        download_url = url.replace("https://trello.com/", "https://api.trello.com/", 1)
        headers = _trello_auth_headers(creds)
    else:
        download_url = url
        headers = {}
    try:
        resp = requests.get(download_url, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = output_path.stat().st_size
        return {
            "name": name,
            "status": "ok",
            "size": file_size,
            "mime": mime_type,
            "path": str(output_path),
        }
    except requests.RequestException as e:
        return {"name": name, "status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Trello card attachments for a STU (NEW AMERICA) project",
    )
    parser.add_argument("project", type=Path, help="Project directory path")
    parser.add_argument("--card-id", help="Explicit Trello card shortLink (overrides auto-detection)")
    parser.add_argument("--dry-run", action="store_true", help="Preview attachments only")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE,
                        help=f"Pipeline queue state file (default: {DEFAULT_STATE_FILE})")
    parser.add_argument("--accounts-dir", type=Path, default=DEFAULT_ACCOUNTS_DIR,
                        help=f"Account env files directory (default: {DEFAULT_ACCOUNTS_DIR})")
    args = parser.parse_args()

    set_verbosity(2)

    project_dir = Path(args.project).resolve()
    if not project_dir.is_dir():
        print_error(f"Project directory not found: {project_dir}")
        return 1

    # Resolve card ID
    card_id = args.card_id
    if not card_id:
        print_info("Resolving Trello card from project directory...")
        card_id = resolve_card_id(project_dir, args.state_file)

    if not card_id:
        print_error(f"Could not resolve Trello card for: {project_dir.name}")
        print_warn("Use --card-id to specify the card shortLink explicitly")
        # Show available cards from state file
        state = _load_state(args.state_file)
        pipelines = state.get("pipelines", {})
        if pipelines:
            print_info("Available cards:")
            for key, entry in pipelines.items():
                card_raw = (entry.get("trello") or {}).get("card_raw") or {}
                sl = card_raw.get("shortLink", key)
                title = str(entry.get("title") or "?")[:65]
                print(f"  {sl}: {title}")
        return 1

    print_ok(f"Card: {card_id}")

    # Load credentials and fetch attachments
    creds = load_trello_credentials(args.accounts_dir)
    print_info("Fetching attachments from Trello...")
    try:
        attachments = fetch_attachments(card_id, creds)
    except requests.RequestException as e:
        print_error(f"Failed to fetch attachments: {e}")
        return 1

    if not attachments:
        print_ok("No attachments found on this card")
        return 0

    print_info(f"Found {len(attachments)} attachment(s)")

    # Dry run: just list them
    if args.dry_run:
        print_header("ATTACHMENTS (DRY RUN)")
        for att in attachments:
            raw_name = att.get("name", "?")
            mime = att.get("mimeType", "?")
            display_name = _ensure_extension(raw_name, mime)
            size = att.get("bytes")
            url = att.get("url", "")
            size_str = _human_size(size) if size else "?"
            is_ext = not _is_trello_hosted(url) and "trello" not in url.lower()
            tag = " [external]" if is_ext else ""
            print(f"  {display_name} ({mime}, {size_str}){tag}")
        return 0

    # Download
    output_dir = project_dir / "attachments"
    print_header(f"DOWNLOADING TO {output_dir}")

    results = []
    for i, att in enumerate(attachments, 1):
        name = att.get("name", "?")
        print_info(f"  [{i}/{len(attachments)}] {name}")
        result = download_attachment(att, output_dir, creds, force=args.force)
        results.append(result)

        status = result["status"]
        if status == "ok":
            print_ok(f"    Downloaded ({_human_size(result['size'])})")
        elif status == "skipped":
            reason = result.get("reason", "")
            if reason == "exists":
                print_info(f"    Skipped (exists, {_human_size(result.get('size', 0))})")
            else:
                print_info(f"    Skipped ({reason})")
        elif status == "external":
            print_info(f"    External URL: {result.get('url', '')[:80]}")
        elif status == "error":
            print_error(f"    Failed: {result.get('error', '?')}")

    # Summary
    print_header("SUMMARY")
    downloaded = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    external = [r for r in results if r["status"] == "external"]
    errors = [r for r in results if r["status"] == "error"]

    total_size = sum(r.get("size", 0) for r in downloaded)
    print(f"  Downloaded: {len(downloaded)} ({_human_size(total_size)})")
    print(f"  Skipped:    {len(skipped)}")
    print(f"  External:   {len(external)}")
    print(f"  Failed:     {len(errors)}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
