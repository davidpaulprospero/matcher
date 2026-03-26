"""Download and combine voiceover audio from STU (NEW AMERICA) Trello card descriptions.

For each card with a VO Google Doc link, fetches the doc, extracts Google Drive
audio part links (Pt1, Pt2, ...), downloads them, and combines into voiceover.mp3.
Validates the combined output with ffprobe.

Usage:
    python scripts/stu/download_vo.py                    # All cards with VO docs
    python scripts/stu/download_vo.py --card-id KxQ0SoGh # Specific card
    python scripts/stu/download_vo.py --dry-run           # Preview only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

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
from gws_drive import GwsDriveContext, GwsDriveError, download_drive_file
from local_project_selector import (
    contains_card_id_token,
    extract_project_card_id,
    iter_local_project_roots,
    project_dir_quality_score,
)

DEFAULT_STATE_FILE = PROJECT_ROOT / "Stu" / "pipeline_queue_state.json"
DEFAULT_PROJECTS_ROOT = PROJECT_ROOT / "projects" / "Stu"
DEFAULT_ACCOUNTS_DIR = PROJECT_ROOT / "Stu" / "accounts"

# ---------------------------------------------------------------------------
# VO detection regexes (mirroring pipeline_queue_state.py:1413-1442)
# ---------------------------------------------------------------------------
_VO_LINE_RE = re.compile(
    r"^[^\n]*\bvo(?:ice\s*over)?\b[^\n]*"
    r"(https://docs\.google\.com/document/d/[a-zA-Z0-9_-]+(?:/[^\s<>\"]*)?)",
    re.IGNORECASE | re.MULTILINE,
)
_SCRIPT_LABEL_RE = re.compile(r"Script\s*\d+\s*-", re.IGNORECASE)
_ALL_DOCS_RE = re.compile(
    r"https://docs\.google\.com/document/d/[a-zA-Z0-9_-]+(?:/[^\s<>\"]*)?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Title-matching helpers (copied from pipeline_queue_state.py:261-290)
# ---------------------------------------------------------------------------
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


def _is_title_match(project_name: str, card_title_tokens: set[str]) -> bool:
    """Return True when project folder name strongly matches card title tokens."""
    if not card_title_tokens:
        return False
    project_tokens = _title_tokens(project_name)
    if not project_tokens:
        return False
    shared = len(project_tokens & card_title_tokens)
    if shared < _PROJECT_NAME_TITLE_MIN_SHARED_TOKENS:
        return False
    union = len(project_tokens | card_title_tokens)
    if union <= 0:
        return False
    return (shared / union) >= _PROJECT_NAME_TITLE_MIN_JACCARD


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


def load_gws_context(accounts_dir: Path) -> GwsDriveContext:
    """Load GWS Drive auth context from the first .env file with a token."""
    if not accounts_dir.is_dir():
        print_error(f"Accounts directory not found: {accounts_dir}")
        sys.exit(1)
    for env_file in sorted(accounts_dir.glob("*.env")):
        if env_file.name == "example.env":
            continue
        env_data = parse_env_file(env_file)
        token = env_data.get("GOOGLE_WORKSPACE_CLI_TOKEN", "")
        if token:
            return GwsDriveContext(token=token)
    print_error(f"No GOOGLE_WORKSPACE_CLI_TOKEN found in {accounts_dir}/*.env")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        print_error(f"State file not found: {state_file}")
        print_warn("Run /stu-queue first to sync state")
        sys.exit(1)
    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_vo_doc_url(entry: dict) -> str:
    """Extract the VO Google Doc URL from a pipeline entry's card description.

    Uses VO-labeled line detection (matching pipeline_queue_state.py logic):
    1. Lines containing 'VO' or 'voiceover' before a Google Docs URL
    2. Falls back to all Google Docs only when no Script N labeling convention detected
    3. Retains old candidates-based fallback if description field missing
    """
    card_raw = (entry.get("trello") or {}).get("card_raw") or {}
    description = str(card_raw.get("desc") or "")

    if description:
        # Strategy 1: VO-labeled lines only
        vo_urls = list(dict.fromkeys(_VO_LINE_RE.findall(description)))
        if vo_urls:
            return vo_urls[0]

        # If Script N convention exists but no VO lines → not ready
        if _SCRIPT_LABEL_RE.search(description):
            return ""

        # Strategy 2: no labeling convention — all docs (backward compat)
        all_urls = list(dict.fromkeys(_ALL_DOCS_RE.findall(description)))
        if all_urls:
            return all_urls[0]

    # Fallback: old candidates approach for state files missing card_raw.desc
    candidates = (entry.get("start_checks") or {}).get("raw_voiceover_candidates", []) or []
    for cand in reversed(candidates):
        if cand.get("is_google_doc"):
            raw_url = str(cand.get("url") or "")
            if "](http" in raw_url:
                raw_url = raw_url.split("](")[0]
            return raw_url
    return ""


def fetch_doc_text(doc_id: str) -> str:
    """Fetch Google Doc as plain text."""
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def extract_drive_file_ids(text: str) -> list[str]:
    """Extract Google Drive file IDs from text."""
    return re.findall(r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", text)


def extract_doc_id(url: str) -> str:
    """Extract Google Doc ID from URL."""
    m = re.search(r"/document/d/([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else ""


def download_drive_files(
    file_ids: list[str],
    output_dir: Path,
    gws_context: GwsDriveContext,
) -> list[Path]:
    """Download files from Google Drive using gws_drive."""
    downloaded = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, file_id in enumerate(file_ids, 1):
        output_path = output_dir / f"Pt{i}.mp3"
        print_info(f"  Downloading Pt{i}/{len(file_ids)} ({file_id[:15]}...)")
        try:
            download_drive_file(file_id, output_path, context=gws_context, timeout_seconds=300)
            if output_path.exists() and output_path.stat().st_size > 0:
                downloaded.append(output_path)
            else:
                print_warn(f"  Pt{i} empty or missing")
        except GwsDriveError as e:
            print_warn(f"  Pt{i} failed (gws): {e}")
        except Exception as e:
            print_warn(f"  Pt{i} failed: {e}")

    return downloaded


def combine_audio(parts: list[Path], output_path: Path) -> bool:
    """Combine audio parts using pydub."""
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for part in parts:
        combined += AudioSegment.from_file(str(part))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), format="mp3")
    return True


def validate_audio(path: Path) -> dict:
    """Validate audio file with ffprobe, return duration and size."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration,size",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        return {"valid": False, "error": result.stderr.strip()}

    parts = result.stdout.strip().split(",")
    if len(parts) >= 2:
        duration = float(parts[0])
        size = int(parts[1])
        return {"valid": True, "duration_sec": duration, "size_bytes": size}
    return {"valid": False, "error": "unexpected ffprobe output"}


def find_project_dir(card_id: str, card_title: str, projects_root: Path) -> Path | None:
    """Find existing project directory using card-ID matching with title fallback.

    Strategy (in priority order):
    1. Check trello_card.json for stored card_id
    2. Check directory name for card_id token
    3. Jaccard title-token matching
    """
    if not projects_root.exists():
        return None

    project_dirs = iter_local_project_roots(projects_root)
    card_id_lower = str(card_id or "").strip().lower()
    match_keys = {card_id_lower} if card_id_lower else set()

    # Pass 1: card_id match
    if match_keys:
        best_match: Path | None = None
        best_score = -1
        for pdir in project_dirs:
            stored_id = extract_project_card_id(pdir)
            matched = (stored_id and stored_id.lower() == card_id_lower) or \
                      contains_card_id_token(pdir.name, match_keys)
            if matched:
                score = project_dir_quality_score(pdir)
                if score > best_score:
                    best_match = pdir
                    best_score = score
        if best_match is not None:
            return best_match

    # Pass 2: title-token Jaccard fallback
    card_tokens = _title_tokens(card_title)
    if card_tokens:
        for pdir in project_dirs:
            if _is_title_match(pdir.name, card_tokens):
                return pdir

    return None


def collect_vo_targets(
    state: dict,
    card_ids: list[str] | None,
    projects_root: Path,
) -> list[dict]:
    """Collect cards that have VO Google Doc links."""
    pipelines = state.get("pipelines", {})
    targets = []

    if card_ids:
        ordered = card_ids
    else:
        queue = state.get("queue", {})
        ordered = (
            (queue.get("ready_card_ids") or [])
            + (queue.get("blocked_card_ids") or [])
            + (queue.get("pending_not_started_card_ids") or [])
        )

    seen = set()
    for cid in ordered:
        key = cid.lower()
        if key in seen:
            continue
        seen.add(key)

        entry = pipelines.get(key) or pipelines.get(cid) or {}
        if not entry:
            continue

        vo_url = extract_vo_doc_url(entry)
        if not vo_url:
            continue

        title = str(entry.get("title") or "Untitled")
        project_dir = find_project_dir(cid, title, projects_root)

        has_vo = False
        if project_dir:
            vo_path = project_dir / "voiceover" / "voiceover.mp3"
            has_vo = vo_path.exists() and vo_path.stat().st_size > 1000

        targets.append({
            "card_id": cid,
            "title": title,
            "vo_doc_url": vo_url,
            "project_dir": str(project_dir) if project_dir else "",
            "has_voiceover": has_vo,
        })

    return targets


def process_card(
    target: dict,
    gws_context: GwsDriveContext,
    projects_root: Path,
) -> dict:
    """Download, combine, and validate VO for one card."""
    card_id = target["card_id"]
    title = target["title"]
    vo_url = target["vo_doc_url"]
    project_dir = Path(target["project_dir"]) if target["project_dir"] else None

    doc_id = extract_doc_id(vo_url)
    if not doc_id:
        return {"card_id": card_id, "status": "error", "error": "bad_doc_url"}

    # 1. Fetch VO doc
    print_info(f"Fetching VO doc ({doc_id[:20]}...)")
    try:
        doc_text = fetch_doc_text(doc_id)
    except Exception as e:
        return {"card_id": card_id, "status": "error", "error": f"doc_fetch_failed: {e}"}

    # 2. Extract Drive links
    file_ids = extract_drive_file_ids(doc_text)
    if not file_ids:
        return {"card_id": card_id, "status": "error", "error": "no_drive_links_in_doc"}

    print_info(f"Found {len(file_ids)} audio parts")

    # 3. Determine output path
    if project_dir and project_dir.exists():
        output_path = project_dir / "voiceover" / "voiceover.mp3"
    else:
        # Create project dir with correct naming: {card_id}-{name}__{date}
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title).strip()
        if len(safe_title) > 80:
            safe_title = safe_title[:80].rstrip()
        date_suffix = datetime.now().strftime("%Y-%m-%d")
        folder_name = f"{card_id}-{safe_title}__{date_suffix}"
        project_dir = projects_root / folder_name

        setup_script = PROJECT_ROOT / "scripts" / "setup_project.py"
        if setup_script.exists():
            print_info(f"Creating project: {project_dir.name}")
            result = subprocess.run(
                [sys.executable, str(setup_script), str(project_dir),
                 "--install-dir", str(PROJECT_ROOT)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if result.returncode != 0:
                print_warn("setup_project failed, creating minimal structure")
                (project_dir / "voiceover").mkdir(parents=True, exist_ok=True)
        else:
            (project_dir / "voiceover").mkdir(parents=True, exist_ok=True)
        output_path = project_dir / "voiceover" / "voiceover.mp3"

    # 4. Download parts to temp dir and combine
    with tempfile.TemporaryDirectory() as tmpdir:
        parts = download_drive_files(file_ids, Path(tmpdir), gws_context)
        if not parts:
            return {"card_id": card_id, "status": "error", "error": "no_parts_downloaded"}

        print_info(f"Downloaded {len(parts)}/{len(file_ids)} parts, combining...")
        try:
            combine_audio(parts, output_path)
        except Exception as e:
            return {"card_id": card_id, "status": "error", "error": f"combine_failed: {e}"}

    # 5. Validate
    validation = validate_audio(output_path)
    if not validation.get("valid"):
        return {"card_id": card_id, "status": "error", "error": f"validation_failed: {validation.get('error')}"}

    duration_min = validation["duration_sec"] / 60
    size_mb = validation["size_bytes"] / (1024 * 1024)

    print_ok(f"voiceover.mp3: {duration_min:.1f} min, {size_mb:.1f} MB ({len(parts)} parts)")
    return {
        "card_id": card_id,
        "status": "ok",
        "output": str(output_path),
        "project_dir": str(project_dir),
        "parts": len(parts),
        "duration_min": round(duration_min, 1),
        "size_mb": round(size_mb, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and combine STU voiceovers from Trello card VO docs",
    )
    parser.add_argument("--card-id", action="append", help="Specific card ID(s)")
    parser.add_argument("--dry-run", action="store_true", help="Preview targets only")
    parser.add_argument("--force", action="store_true", help="Re-download even if voiceover exists")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE,
                        help=f"Pipeline queue state file (default: {DEFAULT_STATE_FILE})")
    parser.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS_ROOT,
                        help=f"Root directory for projects (default: {DEFAULT_PROJECTS_ROOT})")
    parser.add_argument("--accounts-dir", type=Path, default=DEFAULT_ACCOUNTS_DIR,
                        help=f"Account env files directory (default: {DEFAULT_ACCOUNTS_DIR})")
    args = parser.parse_args()

    set_verbosity(2)

    state = load_state(args.state_file)
    targets = collect_vo_targets(state, args.card_id, args.projects_root)

    if not targets:
        print_ok("No cards with VO Google Doc links found")
        return 0

    # Filter out cards that already have voiceover (unless --force)
    if not args.force:
        actionable = [t for t in targets if not t["has_voiceover"]]
        skipped = [t for t in targets if t["has_voiceover"]]
    else:
        actionable = targets
        skipped = []

    if args.dry_run or (args.json and not actionable):
        print_header("STU VO DOWNLOAD - DRY RUN")
        for t in actionable:
            print(f"  {t['card_id']}: {t['title'][:65]}")
            print(f"    VO doc: {t['vo_doc_url'][:70]}")
            print(f"    Project: {t['project_dir'] or 'will create'}")
        if skipped:
            print(f"\n  Skipping {len(skipped)} card(s) with existing voiceover (use --force)")
        if args.json:
            print(json.dumps({"actionable": actionable, "skipped": skipped}, indent=2))
        return 0

    # Load GWS auth (only needed for actual downloads)
    gws_context = load_gws_context(args.accounts_dir)

    print_header("STU VO DOWNLOAD")
    print_info(f"Cards to process: {len(actionable)}")
    if skipped:
        print_info(f"Skipping {len(skipped)} with existing voiceover")

    results = []
    for i, target in enumerate(actionable, 1):
        print_header(f"CARD {i}/{len(actionable)}: {target['title'][:60]}")
        result = process_card(target, gws_context, args.projects_root)
        results.append(result)

    # Summary
    print_header("SUMMARY")
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    for r in ok:
        print_ok(f"{r['card_id']}: {r['duration_min']} min, {r['parts']} parts -> {Path(r['output']).parent.parent.name}")
    for r in errors:
        print_error(f"{r['card_id']}: {r.get('error', '?')}")
    print(f"\n  OK: {len(ok)}  Failed: {len(errors)}  Skipped: {len(skipped)}")

    if args.json:
        print(json.dumps({"results": results, "skipped": [t["card_id"] for t in skipped]}, indent=2))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
