#!/usr/bin/env python3
"""
Download missing project files from Google Drive to local project directories.

Workflow:
  Linux machine runs pipeline → uploads output to a shared "Pipeline Sync" Drive folder
  This script pulls down missing files to the Windows edit machine.

Modes:
  sync   — Download missing files from Drive to local project dirs (default)
  scan   — Check local projects for missing media dirs referenced by OTIO files

Drive folder structure:
  Pipeline Sync/
    <project-folder-name>/
      output/...
      .cache/v/...
      .cache/i/...
      stock/...

Usage:
  # Scan local projects for missing media (no Drive needed)
  python scripts/drive_sync.py scan --state-file Stu/pipeline_queue_state.json

  # Sync all projects from a queue
  python scripts/drive_sync.py sync --state-file Stu/pipeline_queue_state.json \
      --accounts-dir Stu/accounts --drive-folder-id 1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq

  # Sync a specific project by card ID
  python scripts/drive_sync.py sync --state-file Stu/pipeline_queue_state.json \
      --accounts-dir Stu/accounts --drive-folder-id 1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq \
      --card-id KxQ0SoGh

  # Dry run (show what would be downloaded)
  python scripts/drive_sync.py sync --state-file Stu/pipeline_queue_state.json \
      --accounts-dir Stu/accounts --drive-folder-id 1j1wJV64WjIBFmzDBaRferbJd2k0H8tJq \
      --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Force unbuffered output for background/pipe usage
if not sys.stdout.isatty():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

# Ensure scripts/ is on sys.path for sibling imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gws_drive import (
    GwsDriveContext,
    GwsDriveError,
    create_drive_folder,
    download_drive_file,
    list_drive_folder_files,
    upload_drive_file,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"

# Directories to sync from Drive → local
SYNC_DIRS = {"output", ".cache", "stock"}
# Inside .cache, only sync these subdirectories
CACHE_ALLOW = {"v", "i"}


# ---------------------------------------------------------------------------
# Scan: detect missing media in local projects by parsing OTIO
# ---------------------------------------------------------------------------

def scan_otio_media_refs(otio_path: str) -> list[str]:
    """Extract all media reference target_urls from an OTIO file."""
    try:
        import opentimelineio as otio_mod
    except ImportError:
        return []

    try:
        tl = otio_mod.adapters.read_from_file(otio_path)
    except Exception:
        return []

    urls: list[str] = []
    for track in tl.tracks:
        for item in track:
            mr = getattr(item, "media_reference", None)
            if mr:
                url = getattr(mr, "target_url", None)
                if url:
                    urls.append(url)
    return urls


def extract_media_dirs_from_urls(urls: list[str]) -> dict[str, set[str]]:
    """Categorize OTIO media URLs by their directory relative to the project root.

    Returns {relative_dir: set_of_filenames}
    E.g., {".cache/v/VIDEO_ID": {"segment_0_30.mp4"}, "stock/sv": {"p123.mp4"}}
    """
    media_dirs: dict[str, set[str]] = {}

    for url in urls:
        # Normalize to forward slashes
        url = url.replace("\\", "/")

        # Extract relative path from project root
        # Linux paths: /home/hpmint/Desktop/matcher/projects/Stu/PROJECT_DIR/relative/path
        # Or: /home/hpmint/Desktop/matcher/.cache/i/PROJECT_SHORT/file
        rel = None

        # Pattern 1: .../projects/<channel>/<project_dir>/<rel_path>
        m = re.search(r"/projects/[^/]+/[^/]+/(.+)$", url)
        if m:
            rel = m.group(1)

        # Pattern 2: .../.cache/<subdir>/<project_short>/<file>
        # These are global cache refs that map to project-local .cache/<subdir>/
        if not rel:
            m = re.search(r"/\.cache/([^/]+)/[^/]+/(.+)$", url)
            if m:
                rel = f".cache/{m.group(1)}/{m.group(2)}"

        if not rel:
            continue

        # Split into directory and filename
        parts = rel.rsplit("/", 1)
        if len(parts) == 2:
            dir_part, file_part = parts
            media_dirs.setdefault(dir_part, set()).add(file_part)
        else:
            media_dirs.setdefault(".", set()).add(parts[0])

    return media_dirs


def scan_project(project_dir: Path) -> dict[str, Any]:
    """Scan a local project for missing media files referenced by OTIO.

    Returns scan result with missing dirs/files info.
    """
    result: dict[str, Any] = {
        "project_dir": str(project_dir),
        "has_otio": False,
        "total_media_refs": 0,
        "missing_dirs": [],
        "missing_files": 0,
        "present_files": 0,
        "media_summary": {},
    }

    # Find OTIO files — look for timeline_FULL.otio in latest output subdir
    output_dir = project_dir / "output"
    if not output_dir.is_dir():
        return result

    # Find latest output subdir
    subdirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not subdirs:
        return result

    latest = subdirs[-1]
    full_otio = latest / "timeline_FULL.otio"
    if not full_otio.exists():
        # Try any OTIO
        otio_files = list(latest.glob("*.otio"))
        if not otio_files:
            return result
        full_otio = otio_files[0]

    result["has_otio"] = True

    # Parse media references
    urls = scan_otio_media_refs(str(full_otio))
    result["total_media_refs"] = len(urls)

    if not urls:
        return result

    # Categorize by directory
    media_dirs = extract_media_dirs_from_urls(urls)

    # Check which dirs/files are missing locally
    missing_count = 0
    present_count = 0
    missing_dirs: list[dict[str, Any]] = []

    # Group by top-level dir for summary
    top_level_summary: dict[str, dict[str, int]] = {}

    for rel_dir, files in sorted(media_dirs.items()):
        top_parts = rel_dir.split("/")
        if top_parts[0] == ".cache" and len(top_parts) >= 2:
            top_key = f".cache/{top_parts[1]}"
        else:
            top_key = top_parts[0]

        if top_key not in top_level_summary:
            top_level_summary[top_key] = {"total": 0, "missing": 0, "present": 0}

        local_dir_path = project_dir / rel_dir.replace("/", os.sep)
        dir_exists = local_dir_path.is_dir()

        for fname in files:
            top_level_summary[top_key]["total"] += 1
            local_file = local_dir_path / fname
            if local_file.exists():
                present_count += 1
                top_level_summary[top_key]["present"] += 1
            else:
                missing_count += 1
                top_level_summary[top_key]["missing"] += 1

        if not dir_exists and files:
            missing_dirs.append({
                "dir": rel_dir,
                "file_count": len(files),
            })

    result["missing_dirs"] = missing_dirs
    result["missing_files"] = missing_count
    result["present_files"] = present_count
    result["media_summary"] = top_level_summary

    return result


def run_scan(state_file: str, card_ids: list[str]) -> int:
    """Scan local projects for missing media and print report."""
    projects = load_queue_projects(state_file)

    if card_ids:
        filter_ids = {cid.lower() for cid in card_ids}
        projects = {k: v for k, v in projects.items() if k in filter_ids or v["card_id"].lower() in filter_ids}

    print(f"Scanning {len(projects)} projects for missing media...\n")

    needs_sync: list[dict[str, Any]] = []

    for _key, proj in sorted(projects.items(), key=lambda kv: kv[1].get("title", "")):
        local_dir = find_local_project_dir(proj)
        if not local_dir:
            continue

        result = scan_project(local_dir)
        if not result["has_otio"]:
            continue

        card_id = proj["card_id"]
        title = proj["title"][:60]
        total = result["total_media_refs"]
        missing = result["missing_files"]
        present = result["present_files"]

        if missing > 0:
            status = "MISSING FILES"
            needs_sync.append({"card_id": card_id, "title": title, "result": result})
        else:
            status = "OK"

        print(f"  {card_id} | {status:14s} | {present}/{present+missing} files present | {title}")

        if result["media_summary"]:
            for dir_key, counts in sorted(result["media_summary"].items()):
                flag = " <<<" if counts["missing"] > 0 else ""
                print(f"    {dir_key:20s}: {counts['present']}/{counts['total']} present{flag}")

    # Summary
    print(f"\n{'='*60}")
    if needs_sync:
        print(f"PROJECTS NEEDING SYNC: {len(needs_sync)}")
        for item in needs_sync:
            missing = item["result"]["missing_files"]
            dirs = [d["dir"] for d in item["result"]["missing_dirs"]]
            dir_str = ", ".join(dirs[:3])
            if len(dirs) > 3:
                dir_str += f" (+{len(dirs)-3} more)"
            print(f"  {item['card_id']}: {missing} missing files in {dir_str}")
    else:
        print("All projects have their media files present.")

    return 0


# ---------------------------------------------------------------------------
# Sync: download missing files from Drive
# ---------------------------------------------------------------------------

def load_env(accounts_dir: str) -> None:
    """Load first .env file found in accounts directory."""
    if not load_dotenv:
        return
    accounts = Path(accounts_dir)
    if not accounts.is_dir():
        return
    for env_file in sorted(accounts.glob("*.env")):
        load_dotenv(str(env_file), override=True)
        break


def make_context() -> GwsDriveContext:
    """Build GwsDriveContext from environment."""
    return GwsDriveContext(
        token=os.getenv("GOOGLE_WORKSPACE_CLI_TOKEN", "").strip(),
        credentials_file=os.getenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "").strip(),
        impersonated_user=os.getenv("GOOGLE_WORKSPACE_CLI_IMPERSONATED_USER", "").strip(),
    )


def list_folder_recursive(
    folder_id: str,
    context: GwsDriveContext,
    *,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 8,
) -> list[dict[str, Any]]:
    """
    Recursively list files in a Drive folder, respecting SYNC_DIRS and CACHE_ALLOW filters.

    Returns list of dicts with keys: id, name, rel_path, size, mimeType
    """
    if depth > max_depth:
        return []

    items = list_drive_folder_files(folder_id, context=context)
    results: list[dict[str, Any]] = []

    for item in items:
        name = item.get("name", "")
        mime = item.get("mimeType", "")
        rel_path = f"{prefix}/{name}" if prefix else name

        if mime == GOOGLE_FOLDER_MIME:
            # At depth 1 (inside project folder), filter to SYNC_DIRS
            if depth == 1 and name not in SYNC_DIRS:
                continue
            # Inside .cache, only allow CACHE_ALLOW subdirs
            if depth == 2 and prefix.endswith(".cache") and name not in CACHE_ALLOW:
                continue

            sub_files = list_folder_recursive(
                item["id"], context, prefix=rel_path, depth=depth + 1, max_depth=max_depth,
            )
            results.extend(sub_files)
        else:
            results.append({
                "id": item.get("id", ""),
                "name": name,
                "rel_path": rel_path,
                "size": int(item.get("size", 0) or 0),
                "mimeType": mime,
            })

    return results


def load_queue_projects(state_file: str) -> dict[str, dict[str, Any]]:
    """Load project info from pipeline queue state file.

    Returns {card_id_lower: {card_id, title, local_dirs}}
    """
    with open(state_file, encoding="utf-8") as f:
        state = json.load(f)

    projects: dict[str, dict[str, Any]] = {}
    for key, entry in state.get("pipelines", {}).items():
        card_id = entry.get("card_id", "")
        title = entry.get("title", "")
        local_dirs = entry.get("project", {}).get("local_project_dirs", [])
        projects[key] = {
            "card_id": card_id,
            "title": title,
            "local_dirs": local_dirs,
            "pipeline_state": entry.get("pipeline_state", ""),
        }
    return projects


def match_drive_folder_to_project(
    folder_name: str,
    projects: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Match a Drive folder name to a queue project by card ID prefix or title similarity."""
    folder_lower = folder_name.lower()

    # Strategy 1: card ID prefix match
    for _key, proj in projects.items():
        card_id = proj["card_id"]
        if card_id and card_id.lower() in folder_lower:
            return proj

    # Strategy 2: numbered title prefix
    num_match = re.match(r"^(\d+)\.", folder_name)
    if num_match:
        num_prefix = num_match.group(1) + "."
        for _key, proj in projects.items():
            if proj["title"].startswith(num_prefix):
                return proj

    return None


def find_local_project_dir(proj: dict[str, Any]) -> Path | None:
    """Find the first existing local project directory for a project."""
    for d in proj.get("local_dirs", []):
        # Use glob for Windows special char safety
        matches = glob.glob(d)
        if matches:
            p = Path(matches[0])
            if p.is_dir():
                return p
        p = Path(d)
        if p.is_dir():
            return p
    return None


def sync_project(
    drive_folder_id: str,
    drive_folder_name: str,
    local_dir: Path,
    context: GwsDriveContext,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync a single project from Drive to local directory."""
    stats = {"total_drive": 0, "already_local": 0, "downloaded": 0, "skipped": 0, "errors": []}

    print(f"\n  Listing Drive files in '{drive_folder_name}'...")
    drive_files = list_folder_recursive(drive_folder_id, context, depth=1)
    stats["total_drive"] = len(drive_files)

    if not drive_files:
        print("  No syncable files found on Drive.")
        return stats

    print(f"  Found {len(drive_files)} files on Drive")

    for file_info in drive_files:
        rel_path = file_info["rel_path"]
        local_path = local_dir / rel_path.replace("/", os.sep)
        size_mb = file_info["size"] / (1024 * 1024) if file_info["size"] else 0

        if local_path.exists():
            local_size = local_path.stat().st_size
            drive_size = file_info["size"]
            if drive_size and local_size >= drive_size:
                stats["already_local"] += 1
                continue
            if not dry_run:
                print(f"  RE-DOWNLOAD (size mismatch): {rel_path} (local={local_size}, drive={drive_size})")

        if dry_run:
            print(f"  [DRY RUN] Would download: {rel_path} ({size_mb:.1f} MB)")
            stats["downloaded"] += 1
            continue

        print(f"  Downloading: {rel_path} ({size_mb:.1f} MB)")
        try:
            download_drive_file(
                file_info["id"],
                local_path,
                context=context,
                timeout_seconds=600,
            )
            stats["downloaded"] += 1
        except (GwsDriveError, Exception) as exc:
            error_msg = f"Failed to download {rel_path}: {exc}"
            print(f"  ERROR: {error_msg}")
            stats["errors"].append(error_msg)

    return stats


def verify_downloads(local_dir: Path) -> list[str]:
    """Check files for zero-fill corruption."""
    bad_files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(local_dir):
        for fname in filenames:
            full = Path(dirpath) / fname
            try:
                size = full.stat().st_size
                if size == 0:
                    continue
                with open(full, "rb") as fh:
                    header = fh.read(min(64, size))
                if not any(b != 0 for b in header):
                    bad_files.append(str(full.relative_to(local_dir)))
            except OSError:
                continue
    return bad_files


def run_sync(args: argparse.Namespace) -> int:
    """Run the sync operation."""
    load_env(args.accounts_dir)
    context = make_context()

    if not Path(args.state_file).exists():
        print(f"[ERROR] State file not found: {args.state_file}", file=sys.stderr)
        return 1
    projects = load_queue_projects(args.state_file)
    print(f"Loaded {len(projects)} projects from queue")

    if args.card_id:
        filter_ids = {cid.lower() for cid in args.card_id}
        projects = {k: v for k, v in projects.items() if k in filter_ids or v["card_id"].lower() in filter_ids}
        if not projects:
            print(f"[ERROR] No matching projects for card IDs: {args.card_id}", file=sys.stderr)
            return 1

    print(f"\nListing Drive folder {args.drive_folder_id}...")
    try:
        drive_top = list_drive_folder_files(args.drive_folder_id, context=context)
    except GwsDriveError as exc:
        print(f"[ERROR] Failed to list Drive folder: {exc}", file=sys.stderr)
        return 2

    drive_folders = [f for f in drive_top if f.get("mimeType") == GOOGLE_FOLDER_MIME]
    drive_files_top = [f for f in drive_top if f.get("mimeType") != GOOGLE_FOLDER_MIME]

    print(f"Found {len(drive_folders)} project folders on Drive")
    if drive_files_top:
        print(f"  (plus {len(drive_files_top)} top-level files, skipped)")

    if not drive_folders:
        print("\nNo project folders found on Drive. Nothing to sync.")
        return 0

    total_stats = {"synced": 0, "skipped": 0, "no_match": 0, "no_local_dir": 0, "errors": []}

    for drive_folder in sorted(drive_folders, key=lambda f: f.get("name", "")):
        folder_name = drive_folder.get("name", "")
        folder_id = drive_folder.get("id", "")

        proj = match_drive_folder_to_project(folder_name, projects)
        if not proj:
            print(f"\n[SKIP] '{folder_name}' — no matching queue project")
            total_stats["no_match"] += 1
            continue

        local_dir = find_local_project_dir(proj)
        if not local_dir:
            print(f"\n[SKIP] '{folder_name}' → {proj['card_id']} — no local project dir exists")
            total_stats["no_local_dir"] += 1
            continue

        print(f"\n{'='*60}")
        print(f"Project: {proj['title'][:70]}")
        print(f"Card ID: {proj['card_id']} | State: {proj['pipeline_state']}")
        print(f"Drive:   {folder_name}")
        print(f"Local:   {local_dir}")

        stats = sync_project(folder_id, folder_name, local_dir, context, dry_run=args.dry_run)

        if stats["downloaded"] > 0:
            total_stats["synced"] += 1
            if not args.dry_run and not args.skip_verify:
                bad = verify_downloads(local_dir)
                if bad:
                    print(f"\n  WARNING: {len(bad)} zero-filled files detected!")
                    for bf in bad[:5]:
                        print(f"    {bf}")
                    total_stats["errors"].extend(f"Zero-fill: {bf}" for bf in bad)
                else:
                    print(f"  Integrity check passed")
        else:
            if stats["already_local"] == stats["total_drive"]:
                print(f"  Already up to date ({stats['already_local']} files)")
            total_stats["skipped"] += 1

        if stats["errors"]:
            total_stats["errors"].extend(stats["errors"])

    print(f"\n{'='*60}")
    print("SYNC SUMMARY")
    print(f"  Projects synced:    {total_stats['synced']}")
    print(f"  Already up to date: {total_stats['skipped']}")
    print(f"  No queue match:     {total_stats['no_match']}")
    print(f"  No local dir:       {total_stats['no_local_dir']}")
    if total_stats["errors"]:
        print(f"  Errors:             {len(total_stats['errors'])}")
        for err in total_stats["errors"][:5]:
            print(f"    - {err}")

    return 1 if total_stats["errors"] else 0


# ---------------------------------------------------------------------------
# Upload: push local project files to Drive
# ---------------------------------------------------------------------------

# Directories to upload from each project
UPLOAD_DIRS = {"output", ".cache", "voiceover"}

# Global cache directory (entity images live outside project dirs)
GLOBAL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "i"


def find_global_cache_for_project(card_id: str) -> Path | None:
    """Find the global .cache/i/<card_id>-*/ directory for a project."""
    if not GLOBAL_CACHE_DIR.is_dir():
        return None
    for d in GLOBAL_CACHE_DIR.iterdir():
        if d.is_dir() and d.name.startswith(card_id):
            return d
    # Try case-insensitive
    card_lower = card_id.lower()
    for d in GLOBAL_CACHE_DIR.iterdir():
        if d.is_dir() and d.name.lower().startswith(card_lower):
            return d
    return None


def collect_upload_files(
    project_dir: Path,
    card_id: str,
) -> list[tuple[Path, str]]:
    """Collect local files to upload for a project.

    Returns list of (local_path, drive_relative_path) tuples.
    """
    files: list[tuple[Path, str]] = []

    # Walk UPLOAD_DIRS inside the project directory
    for sync_dir_name in UPLOAD_DIRS:
        local_dir = project_dir / sync_dir_name
        if not local_dir.is_dir():
            continue

        # Check if pre-packed v.tar exists to avoid uploading loose segments
        v_tar_local_path = project_dir / ".cache" / "v.tar"
        v_tar_exists = v_tar_local_path.is_file()

        for dirpath, _dirnames, filenames in os.walk(local_dir):
            dp = Path(dirpath)
            # Inside .cache, only sync CACHE_ALLOW subdirs
            rel_to_project = dp.relative_to(project_dir)
            parts = rel_to_project.parts
            if len(parts) >= 2 and parts[0] == ".cache" and parts[1] not in CACHE_ALLOW:
                continue

            for fname in filenames:
                # Skip loose .mp4 segment files if v.tar was created (upload .tar instead)
                if v_tar_exists and fname.endswith(".mp4"):
                    continue
                local_path = dp / fname
                drive_rel = str(rel_to_project / fname).replace(os.sep, "/")
                files.append((local_path, drive_rel))

            # Also add .tar files directly under .cache/ (e.g. .cache/v.tar)
            # Scan THIS directory's filenames for .tar, not the stale fname from the mp4 loop
            if rel_to_project.parts[0] == ".cache":
                for tar_fname in filenames:
                    if tar_fname.endswith(".tar"):
                        tar_local_path = dp / tar_fname
                        tar_drive_rel = str(rel_to_project / tar_fname).replace(os.sep, "/")
                        files.append((tar_local_path, tar_drive_rel))

    # Also collect global .cache/i/<card_id>-*/ files → .cache/i/ on Drive
    global_cache = find_global_cache_for_project(card_id)
    if global_cache and global_cache.is_dir():
        for dirpath, _dirnames, filenames in os.walk(global_cache):
            dp = Path(dirpath)
            rel_to_cache = dp.relative_to(global_cache)
            for fname in filenames:
                local_path = dp / fname
                drive_rel = ".cache/i/" + str(rel_to_cache / fname).replace(os.sep, "/")
                # Normalize ./ prefix
                drive_rel = drive_rel.replace("/./" , "/").replace("/./", "/")
                if drive_rel.endswith("/."):
                    drive_rel = drive_rel[:-2]
                files.append((local_path, drive_rel))

    return files


def ensure_drive_folder_path(
    parent_id: str,
    path_parts: list[str],
    context: GwsDriveContext,
    folder_cache: dict[str, str],
) -> str:
    """Create nested Drive folder structure, caching folder IDs.

    Returns the ID of the deepest folder.
    """
    current_id = parent_id
    for i, part in enumerate(path_parts):
        cache_key = "/".join(path_parts[: i + 1])
        if cache_key in folder_cache:
            current_id = folder_cache[cache_key]
            continue

        # Check if folder already exists
        items = list_drive_folder_files(current_id, context=context)
        existing = None
        for item in items:
            if (
                item.get("name") == part
                and item.get("mimeType") == GOOGLE_FOLDER_MIME
            ):
                existing = item
                break

        if existing:
            current_id = existing["id"]
        else:
            current_id = create_drive_folder(part, current_id, context=context)

        folder_cache[cache_key] = current_id

    return current_id


def upload_project(
    project_folder_id: str,
    local_files: list[tuple[Path, str]],
    context: GwsDriveContext,
    *,
    dry_run: bool = False,
    existing_drive_files: set[str] | None = None,
) -> dict[str, Any]:
    """Upload files for a single project to its Drive folder."""
    stats = {"total": len(local_files), "uploaded": 0, "skipped": 0, "errors": []}
    folder_cache: dict[str, str] = {}

    for local_path, drive_rel in local_files:
        # Skip if already on Drive
        if existing_drive_files and drive_rel in existing_drive_files:
            stats["skipped"] += 1
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024) if local_path.exists() else 0

        if dry_run:
            print(f"  [DRY RUN] Would upload: {drive_rel} ({size_mb:.1f} MB)")
            stats["uploaded"] += 1
            continue

        # Ensure parent folder structure exists on Drive
        rel_parts = drive_rel.split("/")
        filename = rel_parts[-1]
        folder_parts = rel_parts[:-1]

        try:
            if folder_parts:
                parent_id = ensure_drive_folder_path(
                    project_folder_id, folder_parts, context, folder_cache,
                )
            else:
                parent_id = project_folder_id

            print(f"  Uploading: {drive_rel} ({size_mb:.1f} MB)")
            upload_drive_file(local_path, parent_id, context=context, name=filename)
            stats["uploaded"] += 1
        except (GwsDriveError, Exception) as exc:
            error_msg = f"Failed to upload {drive_rel}: {exc}"
            print(f"  ERROR: {error_msg}")
            stats["errors"].append(error_msg)

    return stats


def run_upload(args: argparse.Namespace) -> int:
    """Run the upload operation — push local project files to Drive."""
    load_env(args.accounts_dir)
    context = make_context()

    if not Path(args.state_file).exists():
        print(f"[ERROR] State file not found: {args.state_file}", file=sys.stderr)
        return 1
    projects = load_queue_projects(args.state_file)
    print(f"Loaded {len(projects)} projects from queue")

    if args.card_id:
        filter_ids = {cid.lower() for cid in args.card_id}
        projects = {k: v for k, v in projects.items()
                    if k in filter_ids or v["card_id"].lower() in filter_ids}
        if not projects:
            print(f"[ERROR] No matching projects for card IDs: {args.card_id}", file=sys.stderr)
            return 1

    # Filter to completed projects only (unless --all)
    if not args.all:
        completed = {k: v for k, v in projects.items()
                     if v.get("pipeline_state") in ("completed", "done", "exported")}
        if completed:
            projects = completed
            print(f"Filtered to {len(projects)} completed projects (use --all to include all)")

    root_drive_folder_id = args.drive_folder_id

    # Derive master folder name from state file path (e.g., "stu" from "clients/stu/pipeline_queue_state.json")
    _sf_parts = Path(args.state_file).parts
    if len(_sf_parts) >= 2 and _sf_parts[0] == "clients":
        master_name = _sf_parts[1]
    elif _sf_parts:
        master_name = _sf_parts[0]
    else:
        master_name = "uploads"

    # Find or create master folder inside the Drive root
    print(f"\nListing Drive folder {root_drive_folder_id}...")
    try:
        root_items = list_drive_folder_files(root_drive_folder_id, context=context)
    except GwsDriveError as exc:
        print(f"[ERROR] Failed to list Drive folder: {exc}", file=sys.stderr)
        return 2

    master_folder_id = None
    for item in root_items:
        if item.get("mimeType") == GOOGLE_FOLDER_MIME and item.get("name") == master_name:
            master_folder_id = item["id"]
            break

    if master_folder_id:
        print(f"Using existing master folder '{master_name}'")
    else:
        if args.dry_run:
            print(f"[DRY RUN] Would create master folder '{master_name}'")
            master_folder_id = "DRY_RUN_MASTER"
        else:
            print(f"Creating master folder '{master_name}'")
            master_folder_id = create_drive_folder(master_name, root_drive_folder_id, context=context)

    drive_folder_id = master_folder_id

    # Check existing Drive project folders inside master
    print(f"Listing project folders in '{master_name}'...")
    try:
        if drive_folder_id == "DRY_RUN_MASTER":
            drive_top = []
        else:
            drive_top = list_drive_folder_files(drive_folder_id, context=context)
    except GwsDriveError as exc:
        print(f"[ERROR] Failed to list master folder: {exc}", file=sys.stderr)
        return 2

    existing_folders: dict[str, str] = {}  # card_id_lower -> folder_id
    for item in drive_top:
        if item.get("mimeType") == GOOGLE_FOLDER_MIME:
            name = item.get("name", "")
            # Try to extract card ID from folder name
            parts = name.split("-", 1)
            if parts:
                existing_folders[parts[0].lower()] = item["id"]

    total_stats = {"uploaded": 0, "skipped": 0, "no_local": 0, "errors": []}

    for _key, proj in sorted(projects.items(), key=lambda kv: kv[1].get("title", "")):
        card_id = proj["card_id"]
        title = proj["title"]

        local_dir = find_local_project_dir(proj)
        if not local_dir:
            print(f"\n[SKIP] {card_id} — no local project dir")
            total_stats["no_local"] += 1
            continue

        # Collect files to upload
        upload_files = collect_upload_files(local_dir, card_id)
        if not upload_files:
            print(f"\n[SKIP] {card_id} — no files to upload")
            total_stats["skipped"] += 1
            continue

        total_size_mb = sum(
            f.stat().st_size / (1024 * 1024)
            for f, _ in upload_files
            if f.exists()
        )

        print(f"\n{'=' * 60}")
        print(f"Project: {title[:70]}")
        print(f"Card ID: {card_id}")
        print(f"Local:   {local_dir}")
        print(f"Files:   {len(upload_files)} ({total_size_mb:.0f} MB)")

        # Find or create project folder on Drive
        folder_name = f"{card_id}-{title[:50]}"
        # Sanitize folder name
        folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name).strip()

        if card_id.lower() in existing_folders:
            project_folder_id = existing_folders[card_id.lower()]
            print(f"Drive:   Using existing folder")
        else:
            if args.dry_run:
                print(f"Drive:   [DRY RUN] Would create folder '{folder_name}'")
                project_folder_id = "DRY_RUN"
            else:
                print(f"Drive:   Creating folder '{folder_name}'")
                project_folder_id = create_drive_folder(
                    folder_name, drive_folder_id, context=context,
                )

        # Get existing files on Drive for this project (to skip already-uploaded)
        existing_on_drive: set[str] = set()
        if not args.dry_run and project_folder_id != "DRY_RUN":
            try:
                drive_files = list_folder_recursive(
                    project_folder_id, context, depth=1,
                )
                existing_on_drive = {f["rel_path"] for f in drive_files}
                if existing_on_drive:
                    print(f"  Already on Drive: {len(existing_on_drive)} files")
            except Exception:
                pass

        stats = upload_project(
            project_folder_id,
            upload_files,
            context,
            dry_run=args.dry_run,
            existing_drive_files=existing_on_drive,
        )

        if stats["uploaded"] > 0:
            total_stats["uploaded"] += 1
        else:
            total_stats["skipped"] += 1

        if stats["errors"]:
            total_stats["errors"].extend(stats["errors"])

        print(f"  Done: {stats['uploaded']} uploaded, {stats['skipped']} skipped")

    print(f"\n{'=' * 60}")
    print("UPLOAD SUMMARY")
    print(f"  Projects uploaded:  {total_stats['uploaded']}")
    print(f"  Already up to date: {total_stats['skipped']}")
    print(f"  No local dir:       {total_stats['no_local']}")
    if total_stats["errors"]:
        print(f"  Errors:             {len(total_stats['errors'])}")
        for err in total_stats["errors"][:5]:
            print(f"    - {err}")

    return 1 if total_stats["errors"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync pipeline output from Google Drive to local project dirs")
    subparsers = parser.add_subparsers(dest="command")

    # scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Scan local projects for missing media files")
    scan_parser.add_argument("--state-file", required=True, help="Pipeline queue state JSON file")
    scan_parser.add_argument("--card-id", action="append", default=[], help="Scan specific card ID(s) only")

    # sync subcommand
    sync_parser = subparsers.add_parser("sync", help="Download missing files from Drive")
    sync_parser.add_argument("--state-file", required=True, help="Pipeline queue state JSON file")
    sync_parser.add_argument("--accounts-dir", required=True, help="Directory with .env credentials")
    sync_parser.add_argument("--drive-folder-id", required=True, help="Parent Drive folder ID")
    sync_parser.add_argument("--card-id", action="append", default=[], help="Sync specific card ID(s) only")
    sync_parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    sync_parser.add_argument("--skip-verify", action="store_true", help="Skip integrity verification")

    # upload subcommand
    upload_parser = subparsers.add_parser("upload", help="Upload local project files to Drive")
    upload_parser.add_argument("--state-file", required=True, help="Pipeline queue state JSON file")
    upload_parser.add_argument("--accounts-dir", required=True, help="Directory with .env credentials")
    upload_parser.add_argument("--drive-folder-id", required=True, help="Parent Drive folder ID")
    upload_parser.add_argument("--card-id", action="append", default=[], help="Upload specific card ID(s) only")
    upload_parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    upload_parser.add_argument("--all", action="store_true", help="Include non-completed projects")

    args = parser.parse_args()

    if args.command == "scan":
        return run_scan(args.state_file, args.card_id or [])
    elif args.command == "sync":
        return run_sync(args)
    elif args.command == "upload":
        return run_upload(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
