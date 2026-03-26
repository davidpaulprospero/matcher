#!/usr/bin/env python3
"""
Lipsync Queue - Submit multiple lipsync jobs from Trello URLs

Usage:
    python queue_lipsync.py urls.txt
    python queue_lipsync.py https://trello.com/c/xxx/1 https://trello.com/c/yyy/2
    python queue_lipsync.py --download-only --card-id vp3PMq5T
"""

import argparse
import os
import re
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import json
import requests
import yaml

# Add project directories to path for imports
DEGOLD_DIR = Path(__file__).parent
PROJECT_ROOT = DEGOLD_DIR.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DEGOLD_DIR))

from dotenv import load_dotenv
from gws_drive import GwsDriveContext, GwsDriveError, download_drive_file, list_drive_folder_files
from channel_routing import (
    default_channel_for_account,
    load_board_channel_map as load_shared_board_channel_map,
    resolve_channels_for_board,
)
from local_project_selector import (
    contains_card_id_token,
    extract_project_card_id,
    iter_local_project_roots,
    project_dir_quality_score,
    sort_project_match_candidates,
)
import subprocess
import unicodedata

LIPSYNC_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
LIPSYNC_TITLE_STOPWORDS = {
    "breaking",
    "just",
    "now",
    "minute",
    "ago",
    "voiceover",
    "lipsync",
    "video",
}
PREFERRED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")
LOCAL_PROJECTS_ROOT = Path(__file__).resolve().parent.parent / "projects" / "Degold"
CHANNEL_DIR_ALIASES = {
    "DSR": ("DeepSeaReports", "DSR"),
    "RRU": ("RennReports", "RRU"),
    "JDRP": ("JournalOfDrunkPeople", "JDRP"),
}


def sanitize_filename(name: str) -> str:
    """Sanitize filename by removing problematic characters."""
    # Remove control characters and normalize unicode
    name = unicodedata.normalize('NFKC', name)
    # Replace problematic characters
    for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        name = name.replace(char, '_')
    # Truncate if too long
    if len(name) > 200:
        name = name[:200]
    return name


def load_pipeline_queue_state(state_path: Optional[Path] = None) -> dict[str, Any]:
    """Load local queue state used to map cards to prepared project directories."""
    resolved_path = state_path or (Path(__file__).parent / "pipeline_queue_state.json")
    if not resolved_path.exists():
        return {}

    try:
        with open(resolved_path, "r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    return payload


def get_pipeline_entry_for_card(card_id: str, state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return the queue-state pipeline entry for a card, matching case-insensitively."""
    normalized_card_id = str(card_id or "").strip().lower()
    if not normalized_card_id:
        return {}

    payload = state if isinstance(state, dict) else load_pipeline_queue_state()
    pipelines = payload.get("pipelines", {})
    if not isinstance(pipelines, dict):
        return {}

    for key, value in pipelines.items():
        if str(key or "").strip().lower() == normalized_card_id and isinstance(value, dict):
            return value
    return {}


def get_local_project_dirs_for_card(card_id: str, state: Optional[dict[str, Any]] = None) -> list[str]:
    """Return queue-backed local project directories for a Trello card."""
    entry = get_pipeline_entry_for_card(card_id, state=state)
    project_blob = entry.get("project") or {}
    if not isinstance(project_blob, dict):
        return []

    raw_dirs = project_blob.get("local_project_dirs") or []
    if not isinstance(raw_dirs, list):
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_dirs:
        path_str = str(raw_path or "").strip()
        if not path_str:
            continue
        path_key = path_str.lower()
        if path_key in seen:
            continue
        seen.add(path_key)
        deduped.append(path_str)
    return deduped


def resolve_channel_dirs(channel: str) -> list[Path]:
    """Resolve existing local project roots for a channel code."""
    channel_upper = str(channel or "").strip().upper()
    candidates = list(CHANNEL_DIR_ALIASES.get(channel_upper, ()))
    if channel_upper and channel_upper not in CHANNEL_DIR_ALIASES:
        candidates.append(channel_upper)

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        candidate_path = LOCAL_PROJECTS_ROOT / candidate
        if candidate_path.exists():
            resolved.append(candidate_path)
    return resolved


def path_matches_channel(path_value: str, channel: str) -> bool:
    """Return True when a project path is rooted under the expected channel directory."""
    path_obj = Path(str(path_value or "").strip())
    if not path_obj.exists():
        return False
    for channel_dir in resolve_channel_dirs(channel):
        try:
            path_obj.resolve().relative_to(channel_dir.resolve())
            return True
        except ValueError:
            continue
    return False


def find_local_project_dirs_for_channel(channel: str, card_id: str, card_title: str) -> list[str]:
    """Search the expected channel folder for matching local project directories."""
    matched: list[str] = []
    seen: set[str] = set()
    card_id_lower = str(card_id or "").strip().lower()

    for channel_dir in resolve_channel_dirs(channel):
        for project_dir in sorted(channel_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            name_lower = project_dir.name.lower()
            id_match = bool(card_id_lower and card_id_lower in name_lower)
            title_match = False
            if not id_match and card_title:
                title_match = is_high_confidence_title_match(score_title_match(project_dir.name, card_title))
            if not id_match and not title_match:
                continue
            if not any((project_dir / marker).exists() for marker in ("voiceover", "output", "logs", ".cache")):
                continue
            key = str(project_dir).lower()
            if key in seen:
                continue
            seen.add(key)
            matched.append(str(project_dir))
    return matched


def find_local_project_dirs_by_card_id(card_id: str) -> list[str]:
    """Search every local channel root for exact card-ID-backed project directories."""
    match_keys = {str(card_id or "").strip().lower()}
    match_keys.discard("")
    if not match_keys:
        return []

    matched: list[str] = []
    seen: set[str] = set()
    for root in iter_local_project_roots(LOCAL_PROJECTS_ROOT):
        try:
            project_dirs = sorted(root.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for project_dir in project_dirs:
            if not project_dir.is_dir():
                continue
            if not any((project_dir / marker).exists() for marker in ("voiceover", "output", "logs", ".cache", "trello_card.json")):
                continue
            metadata_card_id = extract_project_card_id(project_dir)
            if not (
                contains_card_id_token(project_dir.name, match_keys)
                or contains_card_id_token(metadata_card_id, match_keys)
            ):
                continue
            key = str(project_dir).lower()
            if key in seen:
                continue
            seen.add(key)
            matched.append(str(project_dir))
    return matched


def rank_local_project_dirs(
    project_dirs: list[str],
    *,
    expected_channel: str = "",
    card_id: str = "",
    card_title: str = "",
) -> list[str]:
    """Rank candidate project directories so richer canonical copies win over stale shells."""
    match_keys = {str(card_id or "").strip().lower()}
    match_keys.discard("")
    candidate_map: dict[str, dict[str, Any]] = {}

    for raw_path in project_dirs:
        path_str = str(raw_path or "").strip()
        if not path_str:
            continue
        project_dir = Path(path_str)
        if not project_dir.exists() or not project_dir.is_dir():
            continue

        metadata_card_id = extract_project_card_id(project_dir)
        id_match = bool(match_keys) and (
            contains_card_id_token(project_dir.name, match_keys)
            or contains_card_id_token(metadata_card_id, match_keys)
        )
        title_match = bool(
            card_title
            and is_high_confidence_title_match(score_title_match(project_dir.name, card_title))
        )
        candidate_map[path_str.lower()] = {
            "path": path_str,
            "channel_match": bool(expected_channel and path_matches_channel(path_str, expected_channel)),
            "id_match": id_match,
            "title_match": title_match,
            "quality_score": project_dir_quality_score(project_dir),
        }

    return [str(item.get("path") or "") for item in sort_project_match_candidates(list(candidate_map.values()))]


def find_preferred_project_audio(project_dir: Path) -> dict[str, str] | None:
    """Pick the best voiceover file from a prepared local project directory."""
    voiceover_dir = project_dir / "voiceover"
    if not voiceover_dir.exists() or not voiceover_dir.is_dir():
        return None

    candidates: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()

    def add_candidate(source: str, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return
        if path.suffix.lower() not in PREFERRED_AUDIO_EXTENSIONS:
            return
        path_key = str(path.resolve()).lower()
        if path_key in seen_paths:
            return
        seen_paths.add(path_key)
        candidates.append((source, path))

    for ext in PREFERRED_AUDIO_EXTENSIONS:
        add_candidate("local_project_nosilence", voiceover_dir / f"voiceover_nosilence{ext}")
    for ext in PREFERRED_AUDIO_EXTENSIONS:
        add_candidate("local_project_voiceover", voiceover_dir / f"voiceover{ext}")

    dynamic_candidates = sorted(
        [
            item for item in voiceover_dir.iterdir()
            if item.is_file() and item.suffix.lower() in PREFERRED_AUDIO_EXTENSIONS
        ],
        key=lambda item: (item.stat().st_mtime if item.exists() else 0.0, item.name.lower()),
        reverse=True,
    )
    for item in dynamic_candidates:
        stem_lower = item.stem.lower()
        if stem_lower.startswith("voiceover_nosilence"):
            add_candidate("local_project_nosilence", item)
    for item in dynamic_candidates:
        stem_lower = item.stem.lower()
        if stem_lower == "voiceover_nosilence":
            continue
        if stem_lower.startswith("voiceover"):
            add_candidate("local_project_voiceover", item)

    if not candidates:
        return None

    source, path = candidates[0]
    return {
        "audio_source": source,
        "audio_path": str(path),
        "project_dir": str(project_dir),
    }


def stage_local_audio_file(
    audio_path: str,
    card_id: str,
    staging_root: Optional[Path] = None,
) -> Optional[str]:
    """Copy project-local audio into the workspace so browser uploads can access it."""
    source_path = Path(audio_path)
    if not source_path.exists() or not source_path.is_file():
        return None

    target_root = staging_root or (Path(__file__).parent / "audio" / "local_project")
    destination_dir = Path(target_root) / sanitize_filename(str(card_id or "").strip() or "unknown")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (sanitize_filename(source_path.stem) + (source_path.suffix or ".mp3"))

    try:
        if source_path.resolve() == destination.resolve():
            return str(destination)
    except OSError:
        pass

    try:
        should_copy = True
        if destination.exists():
            source_stat = source_path.stat()
            destination_stat = destination.stat()
            should_copy = (
                destination_stat.st_size != source_stat.st_size
                or int(destination_stat.st_mtime) < int(source_stat.st_mtime)
            )
        if should_copy:
            shutil.copy2(source_path, destination)
        return str(destination.resolve())
    except OSError:
        return None


def resolve_local_project_audio(
    card_id: str,
    state: Optional[dict[str, Any]] = None,
    staging_root: Optional[Path] = None,
    expected_channel: str = "",
    card_title: str = "",
) -> dict[str, Any] | None:
    """Resolve preferred audio from queue-backed local project directories."""
    local_project_dirs = get_local_project_dirs_for_card(card_id, state=state)
    if expected_channel or card_id or card_title:
        discovered_dirs = []
        if expected_channel:
            discovered_dirs.extend(find_local_project_dirs_for_channel(expected_channel, card_id, card_title))
        discovered_dirs.extend(find_local_project_dirs_by_card_id(card_id))
        local_project_dirs = rank_local_project_dirs(
            local_project_dirs + discovered_dirs,
            expected_channel=expected_channel,
            card_id=card_id,
            card_title=card_title,
        )
    if not local_project_dirs:
        return None

    for project_dir_raw in local_project_dirs:
        project_dir = Path(project_dir_raw)
        if not project_dir.exists() or not project_dir.is_dir():
            continue

        selected_audio = find_preferred_project_audio(project_dir)
        if not selected_audio:
            continue

        original_audio_path = str(selected_audio.get("audio_path") or "").strip()
        staged_audio_path = stage_local_audio_file(
            original_audio_path,
            card_id,
            staging_root=staging_root,
        )
        return {
            "audio_source": selected_audio.get("audio_source", "local_project"),
            "audio_path": staged_audio_path or original_audio_path,
            "original_audio_path": original_audio_path,
            "project_dir": str(project_dir),
            "local_project_dirs": local_project_dirs,
            "staged_audio_path": staged_audio_path or "",
        }

    return None


def trim_audio_to_1min(audio_path: str) -> str:
    """Trim audio to first 60 seconds using ffmpeg."""
    audio_path_obj = Path(audio_path)
    suffix = audio_path_obj.suffix
    trimmed_path = audio_path_obj.parent / f"{audio_path_obj.stem}_1min{suffix}"

    # Skip if already trimmed
    if trimmed_path.exists():
        print(f"    [OK] Using existing trimmed file: {trimmed_path.name}")
        return str(trimmed_path)

    # Check if file is a valid audio file (not HTML/text)
    if audio_path_obj.exists():
        with open(audio_path_obj, 'rb') as f:
            header = f.read(10)
            # Check for MP3 ID3 or common audio headers
            if not (header[:3] == b'ID3' or header[:2] in [b'\xff\xfb', b'\xff\xf3', b'\xff\xf2', b'RIFF']):
                print(f"    [WARN] File is not valid audio (may be HTML), skipping trim")
                return audio_path

    print(f"    Trimming to 1 minute...")
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-t", "60", "-c", "copy", str(trimmed_path), "-y"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
        )
        if result.returncode == 0 and trimmed_path.exists():
            print(f"    [OK] Trimmed: {trimmed_path.name}")
            return str(trimmed_path)
        else:
            print(f"    [WARN] Trim failed, using original")
            return audio_path
    except FileNotFoundError:
        print(f"    [WARN] ffmpeg not found, using original")
        return audio_path
    except Exception as e:
        print(f"    [WARN] Trim error: {e}")
        return audio_path


def load_config(config_name: str) -> dict:
    """Load a YAML config file."""
    if config_name == "board_channel_map.yaml":
        return {"boards": load_shared_board_channel_map(PROJECT_ROOT)}
    config_path = Path(__file__).parent / config_name
    if not config_path.exists():
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def get_account_config(account_name: str) -> tuple[str, str, GwsDriveContext, str]:
    """Load Trello credentials for an account."""
    accounts_dir = Path(__file__).parent / "accounts"
    config_path = accounts_dir / f"{account_name}.env"

    if not config_path.exists():
        raise ValueError(f"Account '{account_name}' not found at {config_path}")

    # Ensure credentials switch correctly when trying multiple accounts.
    load_dotenv(config_path, override=True)

    api_key = os.getenv("TRELLO_API_KEY")
    token = os.getenv("TRELLO_TOKEN")

    if not api_key or not token:
        raise ValueError(f"Invalid config for account '{account_name}' - missing TRELLO_API_KEY or TRELLO_TOKEN")

    gws_context = GwsDriveContext(
        token=str(os.getenv("GOOGLE_WORKSPACE_CLI_TOKEN") or "").strip(),
        credentials_file=str(os.getenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE") or "").strip(),
        impersonated_user=str(os.getenv("GOOGLE_WORKSPACE_CLI_IMPERSONATED_USER") or "").strip(),
    )
    default_channel = default_channel_for_account(account_name, str(os.getenv("DEFAULT_CHANNEL") or ""))
    return api_key, token, gws_context, default_channel


def extract_card_id(url: str) -> Optional[str]:
    """Extract card ID from Trello URL."""
    # Format: https://trello.com/c/<card_id>/...
    match = re.search(r'trello\.com/c/([a-zA-Z0-9]+)', url)
    if match:
        return match.group(1)
    return None


def canonicalize_card_id(card_id: str) -> str:
    """
    Canonicalize Trello short card ID using known local queue state when possible.

    Useful when a user provides a different case variant (for example vp3PMq5T vs vP3PMq5T).
    """
    normalized = str(card_id or "").strip()
    if not normalized:
        return normalized
    state_path = Path(__file__).parent / "pipeline_queue_state.json"
    if not state_path.exists():
        return normalized
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        by_lower: dict[str, str] = {}
        queue_blob = payload.get("queue", {})
        if isinstance(queue_blob, dict):
            queue_fields = (
                "pending_not_started_card_ids",
                "ready_card_ids",
                "blocked_card_ids",
                "completed_card_ids",
                "not_required_card_ids",
                "not_queued_card_ids",
            )
            for field in queue_fields:
                values = queue_blob.get(field, [])
                if isinstance(values, list):
                    for value in values:
                        card_value = str(value or "").strip()
                        if card_value:
                            by_lower[card_value.lower()] = card_value

        pipelines = payload.get("pipelines", {})
        if isinstance(pipelines, dict):
            for key in pipelines.keys():
                card_value = str(key or "").strip()
                if card_value and card_value.lower() not in by_lower:
                    by_lower[card_value.lower()] = card_value
        return by_lower.get(normalized.lower(), normalized)
    except Exception:
        return normalized


def get_card(api_key: str, token: str, card_id: str) -> dict:
    """Get card details from Trello API."""
    response = requests.get(
        f"https://api.trello.com/1/cards/{card_id}",
        params={
            "key": api_key,
            "token": token,
            "fields": "name,idBoard,idMembers,shortUrl,labels",
        }
    )
    response.raise_for_status()
    return response.json()


def get_board(api_key: str, token: str, board_id: str) -> dict:
    """Get board details from Trello API."""
    response = requests.get(
        f"https://api.trello.com/1/boards/{board_id}",
        params={
            "key": api_key,
            "token": token,
            "fields": "name",
        }
    )
    response.raise_for_status()
    return response.json()


def get_members(api_key: str, token: str, member_ids: list[str]) -> list[dict]:
    """Get member details from Trello API."""
    if not member_ids:
        return []

    # Trello API accepts comma-separated member IDs
    response = requests.get(
        f"https://api.trello.com/1/members/{','.join(member_ids)}",
        params={
            "key": api_key,
            "token": token,
            "fields": "fullName,username",
        }
    )
    response.raise_for_status()
    members = response.json()
    # API returns dict if single member, list if multiple
    if isinstance(members, dict):
        return [members]
    return members


def get_card_attachments(api_key: str, token: str, card_id: str) -> list[dict]:
    """Get attachments from a Trello card."""
    response = requests.get(
        f"https://api.trello.com/1/cards/{card_id}/attachments",
        params={"key": api_key, "token": token}
    )
    response.raise_for_status()
    return response.json()


def extract_drive_folder_id(url: str) -> Optional[str]:
    """Extract folder ID from Google Drive URL."""
    # Format: https://drive.google.com/drive/folders/FOLDER_ID
    match = re.search(r'drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None


def file_matches_card_id(filename: str, card_id: str) -> bool:
    """Return True when filename clearly belongs to a Trello card ID."""
    card_id_norm = str(card_id or "").strip().lower()
    if not card_id_norm:
        return False

    filename_norm = str(filename or "").strip().lower()
    if not filename_norm:
        return False

    if filename_norm.startswith(card_id_norm):
        return True

    pattern = rf"(?<![a-z0-9]){re.escape(card_id_norm)}(?![a-z0-9])"
    return bool(re.search(pattern, filename_norm))


def normalize_text_for_match(text: str) -> str:
    """Normalize text for fuzzy title/file matching."""
    value = str(text or "").lower()
    value = value.replace("_", " ")
    value = re.sub(r"\.[a-z0-9]{2,5}$", "", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_significant_tokens(text: str) -> set[str]:
    """Extract meaningful tokens for overlap scoring."""
    tokens = set(re.findall(r"[a-z0-9]+", normalize_text_for_match(text)))
    return {token for token in tokens if len(token) >= 4 and token not in LIPSYNC_TITLE_STOPWORDS}


def score_title_match(filename: str, card_title: str) -> dict:
    """Score how likely a filename belongs to a card title."""
    title_norm = normalize_text_for_match(card_title)
    file_norm = normalize_text_for_match(Path(filename).stem)
    title_tokens = extract_significant_tokens(title_norm)
    file_tokens = extract_significant_tokens(file_norm)
    overlap = len(title_tokens & file_tokens)
    title_token_count = len(title_tokens)
    title_coverage = overlap / max(1, title_token_count)
    seq_ratio = SequenceMatcher(None, title_norm, file_norm).ratio() if title_norm and file_norm else 0.0
    return {
        "overlap": overlap,
        "title_token_count": title_token_count,
        "title_coverage": title_coverage,
        "seq_ratio": seq_ratio,
        "score": max(title_coverage, seq_ratio),
    }


def is_high_confidence_title_match(score: dict) -> bool:
    """Decide whether a title-based match is strong enough to trust."""
    title_token_count = int(score.get("title_token_count") or 0)
    overlap = int(score.get("overlap") or 0)
    title_coverage = float(score.get("title_coverage") or 0.0)
    seq_ratio = float(score.get("seq_ratio") or 0.0)
    overlap_threshold = max(3, min(7, int(round(title_token_count * 0.35))))
    return (
        overlap >= overlap_threshold
        and (title_coverage >= 0.42 or seq_ratio >= 0.58)
    )


def check_existing_lipsync_in_drive(
    drive_folder_id: str,
    card_title: str,
    card_id: str = "",
    gws_context: Optional[GwsDriveContext] = None,
) -> dict:
    """
    Check channel output Drive folder for already-generated lipsync videos.

    Matching strategy:
    1) Card ID token/prefix match in filename.
    2) High-confidence title similarity fallback.
    """
    folder_id = str(drive_folder_id or "").strip()
    if not folder_id:
        return {"status": "unknown", "error": "missing_drive_folder"}

    context = gws_context or GwsDriveContext()
    card_id_norm = str(card_id or "").strip().lower()
    has_card_id = bool(card_id_norm)

    try:
        files = list_drive_folder_files(folder_id, context=context)
    except GwsDriveError as exc:
        return {
            "status": "error",
            "error": f"gws_list_failed:{exc.code}",
            "detail": str(exc),
            "folder_id": folder_id,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "folder_id": folder_id,
        }

    found_files_card_id: list[dict] = []
    found_files_title: list[dict] = []
    scanned_files = 0

    for item in files:
        scanned_files += 1
        file_id = str(item.get("id") or "").strip()
        name = str(item.get("name") or "").strip()
        mime_type = str(item.get("mimeType") or "").strip().lower()
        if not file_id or not name:
            continue

        ext = Path(name).suffix.lower()
        is_video = ext in LIPSYNC_VIDEO_EXTENSIONS or mime_type.startswith("video/")
        if not is_video:
            continue

        entry = {
            "id": file_id,
            "name": name,
            "mimeType": mime_type,
        }

        if has_card_id and file_matches_card_id(name, card_id_norm):
            entry["match_reason"] = "card_id"
            found_files_card_id.append(entry)
            continue

        title_score = score_title_match(name, card_title)
        if is_high_confidence_title_match(title_score):
            entry["match_reason"] = "title_similarity"
            entry["match_score"] = round(float(title_score.get("score", 0.0)), 3)
            found_files_title.append(entry)

    if found_files_card_id:
        return {
            "status": "complete",
            "files": found_files_card_id[:5],
            "match_strategy": "card_id",
            "scanned_files": scanned_files,
            "folder_id": folder_id,
        }
    if found_files_title:
        files_sorted = sorted(
            found_files_title,
            key=lambda item: float(item.get("match_score", 0.0)),
            reverse=True,
        )
        return {
            "status": "complete",
            "files": files_sorted[:5],
            "match_strategy": "title_similarity_fallback" if has_card_id else "title_similarity",
            "scanned_files": scanned_files,
            "folder_id": folder_id,
        }

    return {
        "status": "not_found",
        "files": [],
        "match_strategy": "card_id" if has_card_id else "title_similarity",
        "scanned_files": scanned_files,
        "folder_id": folder_id,
    }


def get_audio_from_drive(
    folder_url: str,
    output_dir: str = "Degold/audio",
    gws_context: Optional[GwsDriveContext] = None,
) -> Optional[str]:
    """
    Download audio file from Google Drive folder.

    Primary: gws list/get.
    Fallback: gdown folder download.

    Returns path to the audio file (mp3), or None if not found.
    """
    folder_id = extract_drive_folder_id(folder_url)
    if not folder_id:
        return None

    # Create output directory with folder-specific subdir to avoid conflicts
    folder_output_dir = Path(output_dir) / folder_id
    folder_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"    Downloading from Drive folder: {folder_id}")
    context = gws_context or GwsDriveContext()

    try:
        files = list_drive_folder_files(folder_id, context=context)
        audio_candidates: list[tuple[int, str, str]] = []
        for item in files:
            file_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            mime_type = str(item.get("mimeType") or "").lower()
            if not file_id or not name:
                continue
            ext = Path(name).suffix.lower()
            if ext != ".mp3" and not mime_type.startswith("audio/"):
                continue
            raw_size = str(item.get("size") or "0").strip()
            try:
                size = int(raw_size)
            except ValueError:
                size = 0
            audio_candidates.append((size, file_id, name))

        if audio_candidates:
            audio_candidates.sort(key=lambda row: (row[0], row[2].lower()), reverse=True)
            _, selected_id, selected_name = audio_candidates[0]
            clean_name = sanitize_filename(Path(selected_name).stem) + (Path(selected_name).suffix or ".mp3")
            destination = folder_output_dir / clean_name
            download_drive_file(selected_id, destination, context=context)
            print(f"    [OK] Found audio via gws: {destination.name} ({destination.stat().st_size / 1024 / 1024:.1f}MB)")
            return str(destination)
        if files:
            print("    [WARN] gws listed files but no mp3/audio candidate was found")
    except GwsDriveError as exc:
        print(f"    [WARN] gws download unavailable ({exc.code}), falling back to gdown: {exc}")
    except Exception as exc:
        print(f"    [WARN] gws download failed unexpectedly, falling back to gdown: {exc}")

    try:
        # Use gdown.download_folder to get all files
        # Use a simpler output path to avoid Windows path issues
        temp_dir = Path(output_dir) / "temp_download"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            import gdown
            downloaded_files = gdown.download_folder(
                folder_url,
                output=str(temp_dir),
                quiet=True
            )
        except Exception as gdown_error:
            # Fallback: try to get file IDs and download individually
            print(f"    gdown.download_folder failed, trying fallback...")
            downloaded_files = []

            # Try to scrape file IDs from the folder page
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(folder_url, headers=headers)
            file_ids = re.findall(r'"([a-zA-Z0-9_-]{33})"', response.text)
            file_ids = [fid for fid in file_ids if fid != folder_id]

            for fid in file_ids:
                try:
                    output_path = temp_dir / f"audio_{fid[:8]}.mp3"
                    import gdown
                    gdown.download(f"https://drive.google.com/uc?id={fid}&export=download", str(output_path), quiet=True)
                    if output_path.exists() and output_path.stat().st_size > 10000:
                        downloaded_files.append(str(output_path))
                except:
                    continue

        if not downloaded_files:
            print(f"    [WARN] No files downloaded from folder")
            return None

        # Find the audio file (mp3) in the downloaded files
        audio_file = None
        audio_size = 0
        for f in downloaded_files:
            fpath = Path(f)
            if fpath.suffix.lower() == '.mp3':
                try:
                    size = fpath.stat().st_size
                    # Found an mp3 file - use the largest one (likely the voiceover)
                    if size > audio_size:
                        audio_size = size
                        audio_file = f
                except:
                    continue

        if audio_file:
            # Copy to a clean filename in the target directory
            src_path = Path(audio_file)
            clean_name = sanitize_filename(src_path.stem) + src_path.suffix
            dest_path = folder_output_dir / clean_name

            # Copy file to destination with clean name
            import shutil
            shutil.copy2(src_path, dest_path)

            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

            print(f"    [OK] Found audio: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f}MB)")
            return str(dest_path)
        else:
            print(f"    [WARN] No mp3 file found in folder")
            print(f"    Files: {[Path(f).name for f in downloaded_files]}")
            return None

    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def load_channel_config() -> dict:
    """Load channel configuration from channels.py."""
    try:
        from channels import CHANNELS
        return {
            ch.code: {
                'drive_folder': ch.drive_folder,
                'avatar_folder': ch.avatar_folder,
                'avatar_path': f"Degold/avatars/{ch.code.lower()}_avatar.jpg",
            }
            for ch in CHANNELS.values()
        }
    except Exception:
        return {}


def resolve_avatar_path(
    channel: str,
    channel_cfg: dict,
    auto_download: bool = False,
    gws_context: Optional[GwsDriveContext] = None,
) -> Optional[str]:
    """
    Resolve avatar image path for a channel.
    Priority:
    1) Explicit channel avatar_path in config
    2) Degold/avatars/<channel>_avatar.(jpg|jpeg|png|webp)
    3) First image inside Degold/avatars/<CHANNEL>/
    4) If auto_download=True, download from channel avatar_folder and pick first image
    """
    base_dir = Path(__file__).parent.parent  # repo root
    avatars_dir = Path(__file__).parent / "avatars"
    channel_upper = channel.upper()
    channel_lower = channel.lower()
    exts = (".jpg", ".jpeg", ".png", ".webp")

    def _exists(path_str: str) -> Optional[str]:
        p = Path(path_str)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if p.exists() and p.is_file() and p.suffix.lower() in exts:
            return str(p)
        return None

    def _first_image(folder: Path) -> Optional[str]:
        if not folder.exists():
            return None
        files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda p: p.name.lower()
        )
        if files:
            return str(files[0].resolve())
        return None

    # 1) Config-defined avatar path
    configured = channel_cfg.get("avatar_path")
    if configured:
        found = _exists(configured)
        if found:
            return found

    # 2) Standard filename patterns
    for ext in exts:
        candidate = avatars_dir / f"{channel_lower}_avatar{ext}"
        if candidate.exists():
            return str(candidate.resolve())

    # 3) Channel subfolder
    found_in_folder = _first_image(avatars_dir / channel_upper)
    if found_in_folder:
        return found_in_folder

    # 4) Auto-download from Drive folder
    if auto_download:
        avatar_folder_id = channel_cfg.get("avatar_folder")
        if avatar_folder_id:
            target_dir = avatars_dir / channel_upper
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                context = gws_context or GwsDriveContext()
                drive_files = list_drive_folder_files(str(avatar_folder_id), context=context)
                image_candidates: list[tuple[str, str]] = []
                for item in drive_files:
                    file_id = str(item.get("id") or "").strip()
                    name = str(item.get("name") or "").strip()
                    ext = Path(name).suffix.lower()
                    if file_id and ext in exts:
                        image_candidates.append((file_id, name))
                image_candidates.sort(key=lambda row: row[1].lower())
                for file_id, name in image_candidates:
                    out_name = sanitize_filename(Path(name).stem) + (Path(name).suffix or ".jpg")
                    destination = target_dir / out_name
                    download_drive_file(file_id, destination, context=context)
                    downloaded = _exists(str(destination))
                    if downloaded:
                        return downloaded
            except GwsDriveError as e:
                print(f"    [WARN] Avatar gws download unavailable for {channel} ({e.code}), trying gdown fallback: {e}")
                folder_url = f"https://drive.google.com/drive/folders/{avatar_folder_id}"
                try:
                    import gdown
                    gdown.download_folder(folder_url, output=str(target_dir), quiet=True)
                    downloaded = _first_image(target_dir)
                    if downloaded:
                        return downloaded
                except Exception as fallback_exc:
                    print(f"    [WARN] Avatar gdown fallback failed for {channel}: {fallback_exc}")
            except Exception as e:
                print(f"    [WARN] Avatar download failed for {channel}: {e}")

    return None


FORM_URL = "https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c"


def generate_mcp_commands(jobs: list[dict], channel_config: dict) -> list[dict]:
    """
    Generate Playwright MCP commands for submitting jobs.

    Preserve submitted jobs by opening a fresh tab per job instead of
    navigating an in-flight submission page back to the form.
    """
    commands = []

    for i, job in enumerate(jobs):
        job_index = i + 1
        title = job['title']
        channel = job['channel']
        audio_path = job.get('audio_path', '')

        # Get channel config
        cfg = channel_config.get(channel, {})
        drive_folder = cfg.get('drive_folder', '')

        # Clean title for display
        clean_title = title.replace('"', '\\"').replace('\n', ' ')
        avatar_path = cfg.get('avatar_path', 'Degold/avatars/face.png').replace(chr(92), '/')

        # Critical for Degold: each submitted job must keep its own page alive.
        commands.append({
            "tool": "browser_tabs",
            "params": {"action": "new"}
        })
        commands.append({
            "tool": "browser_navigate",
            "params": {"url": FORM_URL}
        })
        commands.append({
            "tool": "browser_wait_for",
            "params": {"text": "Video Title", "time": 10 if i == 0 else 5}
        })
        commands.append({
            "tool": "browser_fill_form",
            "params": {
                "fields": [
                    {"name": "Video Title", "type": "textbox", "value": clean_title},
                    {"name": "Channel Code", "type": "combobox", "value": channel},
                    {"name": "Drive Folder ID", "type": "textbox", "value": drive_folder}
                ]
            }
        })
        commands.append({
            "tool": "browser_run_code",
            "params": {
                "code": f"async (page) => {{ await page.setInputFiles('input[name=\"field-2\"]', '{avatar_path}'); return 'Avatar uploaded'; }}"
            }
        })

        # Upload audio if available
        if audio_path:
            audio_forward = audio_path.replace(chr(92), '/')
            commands.append({
                "tool": "browser_run_code",
                "params": {
                    "code": f"async (page) => {{ await page.setInputFiles('input[name=\"field-3\"]', '{audio_forward}'); return 'Audio uploaded'; }}"
                }
            })

        # Take screenshot
        commands.append({
            "tool": "browser_take_screenshot",
            "params": {"filename": f"lipsync-job-{job_index}.png", "type": "png"}
        })

        # Click submit
        commands.append({
            "tool": "browser_click",
            "params": {"selector": "button[type=\"submit\"]"}
        })

        # Leave the submitted tab alive and pause before opening the next one.
        commands.append({
            "tool": "browser_wait_for",
            "params": {"time": 2}
        })

    return commands


def generate_mcp_commands_v2(jobs: list[dict], channel_config: dict) -> list[dict]:
    """
    Generate commands for the current browser MCP tool schema (mcp__browser__*).

    Use direct input selectors for Degold form fields instead of higher-level form
    helpers so the generated workflow matches the Playwright plugin-safe sequence.
    """

    def _abs_posix(path_str: str) -> str:
        path_obj = Path(path_str)
        if not path_obj.is_absolute():
            path_obj = (Path(__file__).parent.parent / path_obj).resolve()
        return str(path_obj).replace("\\", "/")

    def _build_text_fill_actions(title_value: str, drive_folder_value: str) -> list[dict]:
        actions = [
            {"type": "clear", "selector": "input[type='text'][name='field-0']"},
        ]
        if title_value:
            actions.append({
                "type": "type",
                "selector": "input[type='text'][name='field-0']",
                "text": title_value,
            })

        actions.append({
            "type": "clear",
            "selector": "input[type='text'][name='field-4']",
        })
        if drive_folder_value:
            actions.append({
                "type": "type",
                "selector": "input[type='text'][name='field-4']",
                "text": drive_folder_value,
            })
        return actions

    commands = [
        {"tool": "enable", "params": {"client_id": "lipsync-queue"}},
    ]

    for i, job in enumerate(jobs):
        job_index = i + 1
        title = job.get('title', '')
        channel = job.get('channel', '')
        audio_path = job.get('audio_path', '')
        cfg = channel_config.get(channel, {})
        drive_folder = job.get('drive_folder') or cfg.get('drive_folder', '')
        avatar_source = job.get('avatar_path') or cfg.get('avatar_path', 'Degold/avatars/rru_avatar.jpg')
        avatar_path = _abs_posix(avatar_source)
        clean_title = title.replace('\n', ' ')

        # Critical for Degold: keep previously submitted tabs alive.
        # Open a brand-new tab per job instead of navigating away.
        commands.append({
            "tool": "browser_tabs",
            "params": {"action": "new", "url": FORM_URL, "activate": True}
        })
        commands.append({
            "tool": "browser_verify_text_visible",
            "params": {"text": "Video Title"}
        })

        commands.append({
            "tool": "browser_interact",
            "params": {
                "actions": _build_text_fill_actions(clean_title, drive_folder)
            }
        })
        commands.append({
            "tool": "browser_interact",
            "params": {
                "actions": [
                    {"type": "select_option", "selector": "select[name='field-1']", "value": channel}
                ]
            }
        })
        commands.append({
            "tool": "browser_interact",
            "params": {
                "actions": [
                    {
                        "type": "file_upload",
                        "selector": "input[type='file'][name='field-2']",
                        "files": [avatar_path],
                    }
                ]
            }
        })

        if audio_path:
            commands.append({
                "tool": "browser_interact",
                "params": {
                    "actions": [
                        {
                            "type": "file_upload",
                            "selector": "input[type='file'][name='field-3']",
                            "files": [_abs_posix(audio_path)],
                        }
                    ]
                }
            })

        commands.append({
            "tool": "browser_take_screenshot",
            "params": {"path": f"lipsync-job-{job_index}.png", "type": "png"}
        })
        commands.append({
            "tool": "browser_interact",
            "params": {"actions": [{"type": "click", "selector": "button[type='submit']"}]}
        })
        commands.append({"tool": "browser_snapshot", "params": {}})

    return commands


def parse_urls(urls_or_file: list[str]) -> list[str]:
    """Parse URLs from arguments or from a file."""
    urls = []
    for arg in urls_or_file:
        # Check if it's a file path
        if Path(arg).exists():
            with open(arg, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        urls.append(line)
        else:
            urls.append(arg)
    return urls


def main():
    parser = argparse.ArgumentParser(description="Submit multiple lipsync jobs from Trello URLs")
    parser.add_argument('urls', nargs='*', help="Trello URLs or path to file with URLs")
    parser.add_argument(
        '--card-id',
        action='append',
        default=[],
        help="Trello card short ID (can be repeated), e.g. --card-id vp3PMq5T"
    )
    parser.add_argument('--dry-run', action='store_true', help="Show what would be submitted")
    parser.add_argument('--verbose', '-v', action='store_true', help="Show detailed output")
    parser.add_argument('--download-audio', '-d', action='store_true', help="Download audio from Drive folders")
    parser.add_argument(
        '--download-only',
        action='store_true',
        help="Only resolve/download assets (audio/avatar), do not generate submission commands"
    )
    parser.add_argument(
        '--auto-assets',
        action='store_true',
        help="Auto-resolve avatar by channel and auto-resolve audio (local project first, Trello fallback)"
    )
    parser.add_argument(
        '--skip-existing-drive',
        action='store_true',
        help="Skip cards when a matching lipsync video already exists in the channel output Drive folder"
    )
    parser.add_argument('--submit', '-s', action='store_true', help="Generate browser MCP commands to submit jobs")
    parser.add_argument(
        '--run',
        '-r',
        action='store_true',
        help="Deprecated: attempted auto-run of MCP commands (not recommended for Degold form processing)"
    )
    parser.add_argument(
        '--emit-json',
        action='store_true',
        help="Print prepared jobs as JSON (for Claude/agent workflows)"
    )
    args = parser.parse_args()

    if args.download_only and (args.submit or args.run):
        print("[WARN] --download-only ignores --submit/--run")
        args.submit = False
        args.run = False

    # Load configs
    board_map = load_config('board_channel_map.yaml')
    member_map = load_config('member_account_map.yaml')
    channel_config = load_channel_config()
    pipeline_queue_state = load_pipeline_queue_state()

    # Parse URLs
    urls = parse_urls(args.urls) if args.urls else []
    for raw_card_id in (args.card_id or []):
        card_id = str(raw_card_id or "").strip()
        if not card_id:
            continue
        canonical_card_id = canonicalize_card_id(card_id)
        urls.append(f"https://trello.com/c/{canonical_card_id}")
    if not urls:
        parser.error("Provide at least one Trello URL/path or --card-id.")
    print(f"[INFO] Processing {len(urls)} URL(s)\n")

    # Track jobs
    jobs = []

    for i, trello_url in enumerate(urls, 1):
        card_id = extract_card_id(trello_url)
        if not card_id:
            print(f"[{i}/{len(urls)}] ERROR: Could not extract card ID from: {trello_url}")
            continue

        print(f"[{i}/{len(urls)}] {trello_url}")

        # Try each account until one works
        card_data = None
        used_account = None
        used_gws_context: GwsDriveContext | None = None
        used_default_channel = ""

        # Try all accounts that have valid credentials
        accounts_dir = Path(__file__).parent / "accounts"
        available_accounts = [f.stem for f in accounts_dir.glob("*.env") if f.name != 'example.env']

        for account_name in available_accounts:
            try:
                api_key, token, account_gws_context, account_default_channel = get_account_config(account_name)
                card_data = get_card(api_key, token, card_id)
                used_account = account_name
                used_gws_context = account_gws_context
                used_default_channel = account_default_channel
                break
            except Exception as e:
                if args.verbose:
                    print(f"  [WARN] Account '{account_name}' failed: {e}")
                continue

        if not card_data:
            print(f"  [ERROR] Could not fetch card with any account")
            continue

        # Get member info
        member_ids = card_data.get('idMembers', [])
        members = get_members(api_key, token, member_ids)

        # Match member to account
        account = None
        if members:
            for member in members:
                full_name = member.get('fullName', '')
                username = member.get('username', '')

                # Try full name first, then username
                account = member_map.get('members', {}).get(full_name) or \
                          member_map.get('members', {}).get(username)

                if account:
                    print(f"  Member: {full_name} -> Account: {account}")
                    break

        if not account and used_account:
            # Fall back to the account that fetched the card
            account = used_account
            print(f"  Using account: {account} (fallback)")

        if not account:
            print(f"  [ERROR] Could not determine account for card")
            continue

        try:
            _, _, selected_gws_context, selected_default_channel = get_account_config(account)
        except Exception:
            selected_gws_context = used_gws_context or GwsDriveContext()
            selected_default_channel = used_default_channel

        # Resolve project channel separately from the lipsync output channel.
        board_id = card_data.get('idBoard', '')
        board_payload = {}
        if board_id and board_map.get('boards'):
            board_payload = board_map['boards'].get(board_id, {}) or {}
        routing = resolve_channels_for_board(
            board_payload,
            selected_default_channel,
            card_data.get('labels') or [],
        )
        project_channel = routing["project_channel"]
        channel = routing["lipsync_channel"]

        if project_channel == channel:
            print(f"  Board: {board_id} -> Channel: {channel}")
        else:
            print(
                f"  Board: {board_id} -> Project Channel: {project_channel} | "
                f"Lipsync Channel: {channel}"
            )
        print(f"  Title: {card_data.get('name', 'N/A')}")

        cfg = channel_config.get(channel, {})
        drive_folder_id = cfg.get('drive_folder', '')
        local_project_dirs = get_local_project_dirs_for_card(card_id, state=pipeline_queue_state)
        discovered_local_project_dirs: list[str] = []
        if project_channel:
            discovered_local_project_dirs.extend(
                find_local_project_dirs_for_channel(
                    project_channel,
                    card_id,
                    card_data.get('name', ''),
                )
            )
        discovered_local_project_dirs.extend(find_local_project_dirs_by_card_id(card_id))
        local_project_dirs = rank_local_project_dirs(
            local_project_dirs + discovered_local_project_dirs,
            expected_channel=project_channel,
            card_id=card_id,
            card_title=card_data.get('name', ''),
        )
        if args.verbose and local_project_dirs:
            print(f"  Local project dirs: {', '.join(local_project_dirs)}")
        existing_drive_output = check_existing_lipsync_in_drive(
            drive_folder_id,
            card_data.get('name', ''),
            card_id=card_id,
            gws_context=selected_gws_context,
        )
        drive_status = str(existing_drive_output.get("status") or "unknown")
        if drive_status == "complete":
            existing_files = existing_drive_output.get("files", [])
            match_strategy = existing_drive_output.get("match_strategy", "unknown")
            first_name = str((existing_files[0] or {}).get("name") or "unknown") if existing_files else "unknown"
            print(f"  [OK] Existing lipsync found in Drive ({match_strategy}): {first_name}")
            if args.verbose and len(existing_files) > 1:
                for file_entry in existing_files[1:]:
                    print(f"    - {file_entry.get('name', 'unknown')}")
            if args.skip_existing_drive:
                print("  [SKIP] --skip-existing-drive enabled; skipping card")
                continue
        elif drive_status == "error":
            print(f"  [WARN] Drive existence check failed: {existing_drive_output.get('error', 'unknown_error')}")
            if args.verbose and existing_drive_output.get("detail"):
                print(f"    Detail: {existing_drive_output.get('detail')}")
        elif args.verbose:
            print("  [INFO] No existing lipsync output found in channel Drive folder")

        # Resolve avatar automatically (local first; optional Drive download).
        avatar_path = resolve_avatar_path(
            channel,
            cfg,
            auto_download=args.auto_assets,
            gws_context=selected_gws_context,
        )
        if avatar_path:
            print(f"  Avatar: {avatar_path}")
        else:
            print(f"  [WARN] No avatar found for channel {channel}")
            print(f"  [WARN] Expected one of: Degold/avatars/{channel.lower()}_avatar.(jpg|jpeg|png|webp)")

        # Get audio from Drive if requested
        audio_path = None
        audio_source = ""
        original_audio_path = ""
        drive_folder_url = None
        should_download_audio = args.download_audio or args.auto_assets or args.download_only
        if should_download_audio:
            local_audio = resolve_local_project_audio(
                card_id,
                state=pipeline_queue_state,
                expected_channel=project_channel,
                card_title=card_data.get('name', ''),
            )
            if local_audio:
                audio_source = str(local_audio.get("audio_source") or "local_project")
                original_audio_path = str(local_audio.get("original_audio_path") or "")
                audio_path = str(local_audio.get("audio_path") or "")
                local_project_dirs = local_audio.get("local_project_dirs") or local_project_dirs
                print(f"  Project audio: {original_audio_path} ({audio_source})")
                if audio_path and Path(audio_path).exists():
                    audio_path = trim_audio_to_1min(audio_path)
            else:
                if local_project_dirs:
                    print("  [INFO] No preferred project audio found; falling back to Trello attachments")
                print(f"  Fetching attachments...")
                attachments = get_card_attachments(api_key, token, card_id)

                # Find Drive folder URL
                for att in attachments:
                    attachment_url = att.get('url', '')
                    if 'drive.google.com' in attachment_url and '/folders/' in attachment_url:
                        drive_folder_url = attachment_url
                        break

                if drive_folder_url:
                    print(f"  Drive folder: {drive_folder_url}")
                    audio_path = get_audio_from_drive(
                        drive_folder_url,
                        gws_context=selected_gws_context,
                    )
                    if audio_path:
                        audio_source = "trello_drive_attachment"
                        original_audio_path = audio_path
                    # Trim audio to 1 minute
                    if audio_path and Path(audio_path).exists():
                        audio_path = trim_audio_to_1min(audio_path)
                else:
                    print(f"  [WARN] No Drive folder found in attachments")

        if args.auto_assets and not avatar_path:
            print(f"  [ERROR] Auto-assets enabled but no avatar resolved for channel {channel}; skipping card")
            continue
        if should_download_audio and not audio_path:
            print(f"  [ERROR] No downloadable/trimmed audio resolved for card; skipping card")
            continue

        job = {
            'url': trello_url,
            'card_id': card_id,
            'title': card_data.get('name', ''),
            'channel': channel,
            'lipsync_channel': channel,
            'project_channel': project_channel,
            'account': account,
            'board_id': board_id,
            'avatar_path': avatar_path,
            'audio_path': audio_path,
            'audio_source': audio_source,
            'original_audio_path': original_audio_path or audio_path,
            'drive_folder': drive_folder_id,
            'source_drive_folder_url': drive_folder_url,
            'existing_drive_output': existing_drive_output,
            'local_project_dirs': local_project_dirs,
        }
        jobs.append(job)

        if args.verbose:
            print(f"  [OK] Job prepared")

    print(f"\n[INFO] Prepared {len(jobs)} job(s)")

    if args.emit_json:
        print(json.dumps({"jobs": jobs}, indent=2, ensure_ascii=False))
        return

    if args.download_only:
        print("\n[DOWNLOAD-ONLY] Assets resolved:")
        for j in jobs:
            avatar_info = j.get('avatar_path') or 'NOT_RESOLVED'
            audio_info = j.get('audio_path') or 'NOT_DOWNLOADED'
            audio_source = j.get('audio_source') or 'unknown'
            existing_drive = j.get('existing_drive_output') or {}
            drive_info = existing_drive.get('status', 'unknown')
            drive_match = existing_drive.get('match_strategy', 'n/a')
            print(f"  - {j['title']} ({j['card_id']})")
            print(f"      Avatar: {avatar_info}")
            print(f"      Audio:  {audio_info} ({audio_source})")
            print(f"      Drive Existing: {drive_info} ({drive_match})")
        return

    if args.dry_run:
        print("\n[DRY-RUN] Jobs that would be submitted:")
        for j in jobs:
            avatar_info = j.get('avatar_path') or 'MISSING_AVATAR'
            audio_info = j.get('audio_path') or 'MISSING_AUDIO'
            audio_source = j.get('audio_source') or 'unknown'
            existing_drive = j.get('existing_drive_output') or {}
            drive_info = existing_drive.get('status', 'unknown')
            drive_match = existing_drive.get('match_strategy', 'n/a')
            routing_info = j['channel']
            if j.get('project_channel') and j.get('project_channel') != j.get('channel'):
                routing_info = f"{j['project_channel']} -> {j['channel']}"
            print(f"  - {j['title']} ({routing_info} via {j['account']})")
            print(f"      Avatar: {avatar_info}")
            print(f"      Audio:  {audio_info} ({audio_source})")
            print(f"      Drive Existing: {drive_info} ({drive_match})")
        return

    if not jobs:
        print("[ERROR] No jobs to submit")
        return

    # Show jobs
    print("\n[INFO] Jobs prepared:")
    for j in jobs:
        avatar_info = j.get('avatar_path') or 'MISSING_AVATAR'
        audio_info = ""
        if j.get('audio_path'):
            audio_source = j.get('audio_source') or 'unknown'
            audio_info = f" | Audio: {j.get('audio_path', 'N/A')} ({audio_source})"
        existing_drive = j.get('existing_drive_output') or {}
        drive_info = existing_drive.get('status', 'unknown')
        routing_info = j['channel']
        if j.get('project_channel') and j.get('project_channel') != j.get('channel'):
            routing_info = f"{j['project_channel']} -> {j['channel']}"
        print(
            f"  - {j['title']} -> {routing_info} | "
            f"Avatar: {avatar_info}{audio_info} | Drive Existing: {drive_info}"
        )

    # Generate and/or run MCP commands
    if args.submit or args.run:
        if not channel_config:
            print("[WARN] Could not load channels.py, using empty config")

        commands = generate_mcp_commands_v2(jobs, channel_config)

        print("\n" + "="*60)
        print("PLAYWRIGHT MCP COMMANDS")
        print("="*60)
        print("\nRun these commands with Browser MCP (`mcp__browser__*`), not `plugin:playwright:playwright`.\n")
        print("If Chrome launch errors mention launchPersistentContext or an existing browser session,")
        print("stop retrying the plugin flow and use the Browser MCP commands below.\n")

        for i, cmd in enumerate(commands, 1):
            tool = cmd['tool']
            params = cmd['params']
            print(f"# Step {i}: {tool}")
            print(f"mcp__browser__{tool}")
            print(json.dumps(params, indent=2, ensure_ascii=False))
            print()

        print("="*60)
        print("IMPORTANT: Keep browser open after final submission!")
        print("="*60)

    # Auto-run MCP commands if --run flag is set
    if args.run:
        print("\n[WARN] --run is deprecated and intentionally disabled.")
        print("[WARN] Use --submit and run the commands in Claude Playwright MCP.")
        print("[WARN] This avoids closing the browser session before Degold processing completes.")


if __name__ == "__main__":
    main()
