"""Import OTIO timeline into Premiere Pro via pymiere bridge.

Handles: Clip.2→Clip.1 conversion, cross-platform path remapping,
special character sanitization, CEP panel recovery, dialog handling.

Usage:
    python scripts/premiere_import.py "E:\\Edit Job\\client\\project"
    python scripts/premiere_import.py  # auto-detect from most recent output
"""
import json
import glob
import os
import shutil
import sys
import time
import threading
import subprocess
from pathlib import Path

import requests

PANEL_URL = "http://127.0.0.1:3000"
CLEAN_OTIO_PATH = "C:/Users/daves/Documents/Adobe/Premiere Pro/26.0/timeline_import.otio"


# ---------------------------------------------------------------------------
# 1. Find OTIO file
# ---------------------------------------------------------------------------

def find_otio(project_path: str) -> tuple[str, str]:
    """Find the FULL OTIO file and return (otio_path, project_dir).

    Uses glob to handle special characters in paths.
    """
    if not project_path:
        raise FileNotFoundError("No project path provided")

    # Resolve the project directory (handle special chars)
    if os.path.isdir(project_path):
        project_dir = project_path
    else:
        # Try glob pattern
        matches = [d for d in glob.glob(project_path) if os.path.isdir(d)]
        if not matches:
            raise FileNotFoundError(f"Project directory not found: {project_path}")
        project_dir = matches[0]

    # Find OTIO in output subdirectories
    otio_files = glob.glob(os.path.join(project_dir, "output", "*", "*_FULL.otio"))
    if not otio_files:
        # Try direct OTIO files in project
        otio_files = glob.glob(os.path.join(project_dir, "output", "*.otio"))
    if not otio_files:
        otio_files = glob.glob(os.path.join(project_dir, "*.otio"))

    if not otio_files:
        raise FileNotFoundError(f"No OTIO files found in {project_dir}")

    # Use the most recent
    otio_files.sort(key=os.path.getmtime, reverse=True)
    return otio_files[0], project_dir


# ---------------------------------------------------------------------------
# 2. Convert Clip.2 → Clip.1
# ---------------------------------------------------------------------------

def downgrade_clip(item: dict) -> dict:
    """Convert Clip.2 -> Clip.1 by flattening media_references."""
    if item.get("OTIO_SCHEMA") == "Clip.2":
        item["OTIO_SCHEMA"] = "Clip.1"
        refs = item.pop("media_references", {})
        active_key = item.pop("active_media_reference_key", "DEFAULT_MEDIA")

        if refs and active_key in refs:
            item["media_reference"] = refs[active_key]
        elif refs:
            item["media_reference"] = next(iter(refs.values()))
        else:
            item["media_reference"] = {
                "OTIO_SCHEMA": "MissingReference.1",
                "metadata": {},
                "name": "",
                "available_range": None,
                "available_image_bounds": None,
            }
    return item


def clean_resolve_metadata(obj):
    """Remove DaVinci-specific metadata that confuses Premiere."""
    if isinstance(obj, dict):
        meta = obj.get("metadata", {})
        if isinstance(meta, dict):
            meta.pop("Resolve_OTIO", None)
        for key in ("children", "tracks", "effects", "markers"):
            if key in obj:
                val = obj[key]
                if isinstance(val, list):
                    for child in val:
                        clean_resolve_metadata(child)
                elif isinstance(val, dict):
                    clean_resolve_metadata(val)
        if "media_reference" in obj:
            clean_resolve_metadata(obj["media_reference"])
    return obj


# ---------------------------------------------------------------------------
# 3. Cross-platform path remapping
# ---------------------------------------------------------------------------

def _find_source_root(urls: list[str]) -> str:
    """Find the common project root from a list of target_url values.

    Looks for the parent of known subdirectories (.cache/, voiceover/, stock/).
    """
    known_markers = ["/.cache/", "/voiceover/", "/stock/", "/entity_images/"]

    for url in urls:
        for marker in known_markers:
            idx = url.find(marker)
            if idx > 0:
                return url[:idx]

    # Fallback: longest common prefix
    if urls:
        prefix = os.path.commonprefix(urls)
        # Trim to last directory separator
        last_sep = prefix.rfind("/")
        if last_sep > 0:
            return prefix[:last_sep]

    return ""


def remap_paths(data: dict, project_dir: str) -> tuple[int, list[str]]:
    """Remap all target_url values from source platform to Windows project dir.

    Returns (remapped_count, list_of_missing_files).
    """
    win_base = project_dir.replace(os.sep, "/")

    # Collect all target_url values
    all_urls = []
    def collect_urls(obj):
        if isinstance(obj, dict):
            url = obj.get("target_url", "")
            if url and not url.startswith("file:///") and "/" in url:
                all_urls.append(url)
            elif url and url.startswith("file:///"):
                # Already a file URL - extract path for checking
                all_urls.append(url[8:] if url.startswith("file:///") else url)
            for v in obj.values():
                collect_urls(v)
        elif isinstance(obj, list):
            for item in obj:
                collect_urls(item)
    collect_urls(data)

    if not all_urls:
        return 0, []

    # Determine source root
    source_root = _find_source_root(all_urls)

    # Check if remapping is needed
    needs_remap = False
    if source_root:
        # Source root is not a valid Windows path
        if not os.path.isdir(source_root) and not source_root[1:3] == ":/":
            needs_remap = True
        # Source root starts with / (Unix absolute)
        elif source_root.startswith("/"):
            needs_remap = True

    if not needs_remap:
        # Check if paths already resolve
        test_path = all_urls[0]
        if test_path.startswith("file:///"):
            test_path = test_path[8:]
        if os.path.exists(test_path):
            return 0, []  # Paths already valid
        needs_remap = True

    # Remap all URLs
    remapped = 0
    missing = []

    def remap_urls(obj):
        nonlocal remapped
        if isinstance(obj, dict):
            if "target_url" in obj and isinstance(obj["target_url"], str):
                url = obj["target_url"]
                # Strip file:/// prefix if present
                raw_path = url
                if raw_path.startswith("file:///"):
                    raw_path = raw_path[8:]

                # Extract relative path from source root
                rel_path = None
                if source_root and raw_path.startswith(source_root):
                    rel_path = raw_path[len(source_root):].lstrip("/")
                else:
                    # Try to find known directory markers
                    for marker in ["/.cache/", "/voiceover/", "/stock/", "/entity_images/"]:
                        idx = raw_path.find(marker)
                        if idx >= 0:
                            rel_path = raw_path[idx + 1:]  # skip leading /
                            break

                if rel_path is None:
                    # Last resort: just use filename
                    rel_path = os.path.basename(raw_path)

                new_url = "file:///" + win_base + "/" + rel_path

                # Verify file exists
                full_path = os.path.join(project_dir, rel_path.replace("/", os.sep))
                if not os.path.exists(full_path):
                    missing.append(rel_path)

                obj["target_url"] = new_url
                remapped += 1

            for v in obj.values():
                remap_urls(v)
        elif isinstance(obj, list):
            for item in obj:
                remap_urls(item)

    remap_urls(data)
    return remapped, list(set(missing))


# ---------------------------------------------------------------------------
# 4. Full OTIO conversion pipeline
# ---------------------------------------------------------------------------

def convert_otio(input_path: str, project_dir: str) -> tuple[str, dict]:
    """Convert OTIO for Premiere: Clip.2→Clip.1, path remap, clean path copy.

    Returns (clean_otio_path, stats_dict).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Downgrade clips
    tracks = data.get("tracks", {}).get("children", [])
    clip_count = 0
    for track in tracks:
        for item in track.get("children", []):
            if item.get("OTIO_SCHEMA", "").startswith("Clip"):
                downgrade_clip(item)
                clip_count += 1

    # Clean DaVinci metadata
    clean_resolve_metadata(data)

    # Remap paths
    remapped, missing = remap_paths(data, project_dir)

    # Write to clean ASCII path
    os.makedirs(os.path.dirname(CLEAN_OTIO_PATH), exist_ok=True)
    with open(CLEAN_OTIO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    stats = {
        "clips_converted": clip_count,
        "paths_remapped": remapped,
        "missing_files": missing,
        "output_path": CLEAN_OTIO_PATH,
    }
    return CLEAN_OTIO_PATH, stats


# ---------------------------------------------------------------------------
# 5. Premiere connection management
# ---------------------------------------------------------------------------

def check_premiere() -> str:
    """Check Premiere status. Returns: 'ready', 'blocked', 'not_running'."""
    try:
        requests.get(PANEL_URL, timeout=2)
    except Exception:
        return "not_running"
    try:
        r = requests.post(PANEL_URL, json={"to_eval": '"alive";'}, timeout=5)
        return "ready"
    except requests.exceptions.Timeout:
        return "blocked"
    except Exception:
        return "not_running"


def recover_stuck_panel():
    """Kill orphaned CEPHtmlEngine holding port 3000, let Premiere respawn it."""
    import psutil
    import subprocess as sp

    # Find which PID owns port 3000
    result = sp.run(["netstat", "-aon"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if ":3000" in line and "LISTENING" in line:
            parts = line.strip().split()
            pid = int(parts[-1])
            try:
                proc = psutil.Process(pid)
                if "CEPHtmlEngine" in proc.name():
                    proc.kill()
                    print(f"  Killed stuck CEPHtmlEngine (PID {pid})")
                    time.sleep(4)
                    return True
            except Exception:
                pass
    return False


def ensure_premiere_ready(max_wait: int = 120) -> bool:
    """Ensure Premiere is running and responsive. Attempts recovery if blocked."""
    status = check_premiere()

    if status == "ready":
        return True

    if status == "blocked":
        print("  Premiere blocked - attempting CEP panel recovery...")
        if recover_stuck_panel():
            # Wait for panel to come back
            for _ in range(30):
                time.sleep(2)
                if check_premiere() == "ready":
                    print("  Panel recovered!")
                    return True
        print("  Recovery failed - Premiere may need manual restart")
        return False

    if status == "not_running":
        print("  Premiere not running - launching...")
        premiere_exe = r"C:\Program Files\Adobe\Adobe Premiere Pro 2026\Adobe Premiere Pro.exe"
        if os.path.exists(premiere_exe):
            subprocess.Popen([premiere_exe])
            for i in range(max_wait // 2):
                time.sleep(2)
                if check_premiere() == "ready":
                    print(f"  Premiere ready after {(i+1)*2}s")
                    return True
        print("  Premiere did not start in time")
        return False

    return False


# ---------------------------------------------------------------------------
# 6. Import via QE API
# ---------------------------------------------------------------------------

def _dialog_watcher(stop_event: threading.Event):
    """Background thread to auto-dismiss Premiere dialogs during import."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    try:
        from premiere_dialog_handler import find_premiere_dialogs, handle_dialog
    except ImportError:
        return  # No dialog handler available

    while not stop_event.is_set():
        time.sleep(1)
        try:
            for d in find_premiere_dialogs():
                print(f"  Auto-handling dialog: {d.get('title', '?')}")
                handle_dialog(d)
        except Exception:
            pass


def import_otio(otio_path: str) -> dict:
    """Import OTIO into Premiere via QE API. Returns import result."""
    path_json = json.dumps(otio_path.replace("\\", "/"))

    # Start dialog watcher
    stop_event = threading.Event()
    watcher = threading.Thread(target=_dialog_watcher, args=(stop_event,), daemon=True)
    watcher.start()

    try:
        eval_code = (
            "app.enableQE(); "
            "try { "
            f"  var result = qe.project.importFiles([{path_json}]); "
            "  JSON.stringify({success: true, result: String(result)}); "
            "} catch(e) { "
            "  JSON.stringify({success: false, error: e.toString()}); "
            "}"
        )
        r = requests.post(PANEL_URL, json={"to_eval": eval_code}, timeout=120)
        return json.loads(r.text)
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Import timed out (120s) - check for dialogs"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        stop_event.set()


# ---------------------------------------------------------------------------
# 7. Verify import
# ---------------------------------------------------------------------------

def verify_import() -> dict:
    """Verify the imported timeline in Premiere."""
    verify_code = """
    var result = {sequences: [], media: {online: 0, offline: 0}};
    for (var i = 0; i < app.project.sequences.numSequences; i++) {
        var s = app.project.sequences[i];
        var vClips = 0, aClips = 0;
        var tracks = [];
        for (var t = 0; t < s.videoTracks.numTracks; t++) {
            var clips = s.videoTracks[t].clips.numItems;
            vClips += clips;
            if (clips > 0) tracks.push({name: s.videoTracks[t].name, clips: clips, muted: s.videoTracks[t].isMuted()});
        }
        for (var t = 0; t < s.audioTracks.numTracks; t++) {
            var clips = s.audioTracks[t].clips.numItems;
            aClips += clips;
            if (clips > 0) tracks.push({name: s.audioTracks[t].name, clips: clips, muted: s.audioTracks[t].isMuted()});
        }
        result.sequences.push({name: s.name, videoClips: vClips, audioClips: aClips, tracks: tracks});
    }
    var items = app.project.rootItem.children;
    for (var i = 0; i < items.numItems; i++) {
        if (items[i].type === 1) {
            if (items[i].isOffline()) result.media.offline++;
            else result.media.online++;
        }
    }
    JSON.stringify(result);
    """
    try:
        r = requests.post(PANEL_URL, json={"to_eval": verify_code}, timeout=15)
        return json.loads(r.text)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 8. Main orchestrator
# ---------------------------------------------------------------------------

def premiere_import(project_path: str) -> dict:
    """Full import pipeline: find → convert → connect → import → verify.

    Returns a result dict with all stats and verification.
    """
    result = {"success": False, "steps": {}}

    # Step 1: Find OTIO
    print("[1/5] Finding OTIO file...")
    otio_path, project_dir = find_otio(project_path)
    print(f"  Found: {otio_path}")
    result["steps"]["find"] = {"otio": otio_path, "project": project_dir}

    # Step 2: Convert
    print("[2/5] Converting OTIO (Clip.2→Clip.1, path remap, metadata cleanup)...")
    clean_path, conv_stats = convert_otio(otio_path, project_dir)
    print(f"  Converted {conv_stats['clips_converted']} clips, remapped {conv_stats['paths_remapped']} paths")
    if conv_stats["missing_files"]:
        print(f"  Missing files ({len(conv_stats['missing_files'])}): {conv_stats['missing_files'][:5]}")
    result["steps"]["convert"] = conv_stats

    # Step 3: Connect to Premiere
    print("[3/5] Connecting to Premiere Pro...")
    if not ensure_premiere_ready():
        result["error"] = "Could not connect to Premiere Pro"
        return result
    print("  Connected!")

    # Step 4: Import
    print("[4/5] Importing via QE API...")
    import_result = import_otio(clean_path)
    print(f"  Import: {import_result}")
    result["steps"]["import"] = import_result

    if not import_result.get("success"):
        result["error"] = f"Import failed: {import_result.get('error', 'unknown')}"
        return result

    # Step 5: Verify
    time.sleep(2)
    print("[5/5] Verifying import...")
    verify = verify_import()
    result["steps"]["verify"] = verify

    for seq in verify.get("sequences", []):
        print(f"  Sequence: {seq['name']}")
        print(f"    {seq['videoClips']} video clips, {seq['audioClips']} audio clips")
        for t in seq.get("tracks", []):
            muted = " (muted)" if t.get("muted") else ""
            print(f"    {t['name']}: {t['clips']} clips{muted}")

    media = verify.get("media", {})
    online = media.get("online", 0)
    offline = media.get("offline", 0)
    total = online + offline
    pct = (online / total * 100) if total > 0 else 0
    print(f"  Media: {online}/{total} online ({pct:.0f}%)")

    if offline > 0:
        print(f"  {offline} offline items (missing source files)")

    result["success"] = True
    return result


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else None
    if not project:
        print("Usage: python premiere_import.py <project_path>")
        sys.exit(1)

    result = premiere_import(project)
    if not result["success"]:
        print(f"\nFAILED: {result.get('error', 'unknown')}")
        sys.exit(1)
    else:
        print("\nImport complete!")
