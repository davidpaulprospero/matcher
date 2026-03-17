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
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STU_STATE_FILE = PROJECT_ROOT / "Stu" / "pipeline_queue_state.json"
STU_PROJECTS_ROOT = Path(r"E:\Edit Job\Stu")


def print_ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def print_info(msg: str) -> None:
    print(f"  [..] {msg}")


def print_warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def print_error(msg: str) -> None:
    print(f"  [ERROR] {msg}")


def print_header(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def load_state() -> dict:
    if not STU_STATE_FILE.exists():
        print_error(f"State file not found: {STU_STATE_FILE}")
        print_info("Run /stu-queue first to sync state")
        sys.exit(1)
    with open(STU_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_vo_doc_url(entry: dict) -> str:
    """Extract the VO Google Doc URL from a pipeline entry's candidates."""
    candidates = (entry.get("start_checks") or {}).get("raw_voiceover_candidates", []) or []
    # The VO doc is typically the last Google Doc in the description
    for cand in reversed(candidates):
        if cand.get("is_google_doc"):
            raw_url = str(cand.get("url") or "")
            # Fix doubled Trello markdown URLs: url](url
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


def download_drive_files(file_ids: list[str], output_dir: Path) -> list[Path]:
    """Download files from Google Drive using gdown."""
    import gdown

    downloaded = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, file_id in enumerate(file_ids, 1):
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = output_dir / f"Pt{i}.mp3"
        print_info(f"  Downloading Pt{i}/{len(file_ids)} ({file_id[:15]}...)")
        try:
            gdown.download(url, str(output_path), quiet=True, fuzzy=True)
            if output_path.exists() and output_path.stat().st_size > 0:
                downloaded.append(output_path)
            else:
                print_warn(f"  Pt{i} empty or missing")
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


def find_project_dir(card_title: str) -> Path | None:
    """Find existing project directory for a card title."""
    if not STU_PROJECTS_ROOT.exists():
        return None
    # Match by first few words of title
    title_prefix = re.sub(r'[<>:"/\\|?*]', '', card_title)[:40].strip().lower()
    for child in STU_PROJECTS_ROOT.iterdir():
        if child.is_dir() and title_prefix[:20].lower() in child.name.lower():
            return child
    return None


def collect_vo_targets(state: dict, card_ids: list[str] | None) -> list[dict]:
    """Collect cards that have VO Google Doc links."""
    pipelines = state.get("pipelines", {})
    targets = []

    if card_ids:
        ordered = card_ids
    else:
        # All ready + blocked cards
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
        project_dir = find_project_dir(title)

        # Check if voiceover already exists
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


def process_card(target: dict) -> dict:
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
        # Create project dir with full structure via setup_project.py
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title).strip()
        from datetime import datetime
        date_suffix = datetime.now().strftime("%Y-%m-%d")
        project_dir = STU_PROJECTS_ROOT / f"{safe_title}__{date_suffix}"
        setup_script = PROJECT_ROOT / "scripts" / "setup_project.py"
        if setup_script.exists():
            print_info(f"Creating project: {project_dir.name}")
            result = subprocess.run(
                [sys.executable, str(setup_script), str(project_dir), str(PROJECT_ROOT)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if result.returncode != 0:
                print_warn(f"setup_project failed, creating minimal structure")
                (project_dir / "voiceover").mkdir(parents=True, exist_ok=True)
        else:
            (project_dir / "voiceover").mkdir(parents=True, exist_ok=True)
        output_path = project_dir / "voiceover" / "voiceover.mp3"

    # 4. Download parts to temp dir and combine
    with tempfile.TemporaryDirectory() as tmpdir:
        parts = download_drive_files(file_ids, Path(tmpdir))
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
    parser = argparse.ArgumentParser(description="Download and combine STU voiceovers from Trello card VO docs")
    parser.add_argument("--card-id", action="append", help="Specific card ID(s)")
    parser.add_argument("--dry-run", action="store_true", help="Preview targets only")
    parser.add_argument("--force", action="store_true", help="Re-download even if voiceover exists")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    state = load_state()
    targets = collect_vo_targets(state, args.card_id)

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

    if args.dry_run or args.json:
        print_header("STU VO DOWNLOAD - DRY RUN")
        for t in actionable:
            print(f"  {t['card_id']}: {t['title'][:65]}")
            print(f"    VO doc: {t['vo_doc_url'][:70]}")
            print(f"    Project: {t['project_dir'] or 'will create'}")
        if skipped:
            print(f"\n  Skipping {len(skipped)} card(s) with existing voiceover (use --force to re-download)")
        if args.json:
            print(json.dumps({"actionable": actionable, "skipped": skipped}, indent=2))
        return 0

    print_header("STU VO DOWNLOAD")
    print_info(f"Cards to process: {len(actionable)}")
    if skipped:
        print_info(f"Skipping {len(skipped)} with existing voiceover")

    results = []
    for i, target in enumerate(actionable, 1):
        print_header(f"CARD {i}/{len(actionable)}: {target['title'][:60]}")
        result = process_card(target)
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

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
