#!/usr/bin/env python3
"""
Pipeline queue state manager for Trello-driven projects.

This script maintains a single JSON source of truth that includes:
1) Full Trello card payloads for tracked projects
2) Per-project pipeline state
3) Start-readiness gates
4) Queues for pending/not-started and ready projects
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
import random
import re
from shutil import move, which
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import requests


_script_path = os.path.abspath(__file__)
PROJECT_ROOT = Path(_script_path).parent.parent.resolve()
SCRIPTS_DIR = Path(_script_path).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "clients" / "shared"))
os.chdir(PROJECT_ROOT)

from script_utils import print_error, print_header, print_info, print_ok, print_warn, set_verbosity
from script_utils import run_subprocess, popen_subprocess
from gws_drive import GwsDriveContext, GwsDriveError, list_drive_folder_files
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
    is_strong_project_candidate,
    sort_project_match_candidates,
)


STATE_SCHEMA_VERSION = "1.0.0"
DEFAULT_STATE_FILE = PROJECT_ROOT / "clients" / "degold" / "pipeline_queue_state.json"
DEFAULT_LIPSYNC_TRACKING_FILE = PROJECT_ROOT / "clients" / "degold" / "lipsync_tracking.json"
DEFAULT_BOARD_MAP_FILE = PROJECT_ROOT / "clients" / "degold" / "board_channel_map.yaml"
DEFAULT_ACCOUNTS_DIR = PROJECT_ROOT / "clients" / "degold" / "accounts"
DEFAULT_DISCORD_STATE_FILE = PROJECT_ROOT / "clients" / "degold" / "discord_pipeline_projects.json"
DEFAULT_LIPSYNC_NEXT_SYNC_TIMEOUT_SECONDS = 30.0
LOCAL_PROJECTS_ROOT = Path(PROJECT_ROOT / "projects" / "Degold")
ARCHIVED_PROJECTS_DIR_PREFIX = "_archived_pipeline_projects"

EDITING_LIST_NAMES = {"editing"}
REVIEW_LIST_NAMES = {"edit review"}
# Only include workflow states that still require active submission/compliance work.
# `ready_to_schedule` is treated as no further action required.
COMPLIANCE_WORKFLOW_STATES = {"editing", "review", "final_checks"}
READY_TO_SCHEDULE_LIST_NAMES = {"ready to schedule"}
FINAL_CHECK_LIST_NAMES = {
    "thumbnail/final checks",
    "thumbnail / final checks",
    "thumbnail final checks",
}
ARCHIVE_LIST_NAMES = {"archive"}
READY_TO_UPLOAD_ARCHIVE_LIST_NAMES = {"ready to upload"}
SCRIPT_VO_DESCRIPTION_LIST_NAMES = {
    "script/vo/description",
    "script / vo / description",
    "script vo description",
    "script/voiceover/description",
    "script / voiceover / description",
    "script voiceover description",
}
PIPELINE_READY_LIST_NAMES = {
    "scripts & vo's",
    "scripts & vos",
    "scripts/vo's",
    "scripts/vos",
    "scripts / vo's",
    "scripts / vos",
    "scripts and vo's",
    "scripts and vos",
}
NOT_STARTED_LIST_NAMES = {"not started", "not-started"}
PENDING_LIST_NAMES = {"pending", "todo", "to do", "backlog", "queue"}
COMPLETED_LIST_NAMES = {"done", "completed", "complete", "published", "uploaded", "scheduled"}

# Trello channel codes map to local folder names under E:\Edit Job\Degold.
CHANNEL_DIR_ALIASES = {
    "DSR": ("DeepSeaReports", "DSR"),
    "RRU": ("RennReports",),
    "STU": (".",),
}

COMPLETION_LOG_MARKERS = (
    "last_completed=output",
    "stage output completed",
    "[output] stage complete",
)
RUNNING_LOG_ACTIVITY_WINDOW_SECONDS = 10 * 60
OUTPUT_COMPLETION_EVIDENCE_FILES = (
    "timeline_FULL.otio",
    "timeline_project.xml",
    "timeline.edl",
    "timeline_sequence.xml",
)
OUTPUT_COMPLETION_MIN_MATCHES = 2

VOICEOVER_HINTS = (
    "voiceover",
    "voice over",
    "script/vo",
    "script vo",
    "raw vo",
    "raw voiceover",
)
VOICEOVER_SHORT_TOKEN_RE = re.compile(r"(?<![a-z0-9])vo(?![a-z0-9])", re.IGNORECASE)
SILENCE_REMOVED_HINTS = (
    "silence removed",
    "silence-removed",
    "silence_removed",
    "remove silence",
    "no silence",
)
LIKELY_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma")
LIKELY_VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv")
VOICEOVER_FILENAME_PATTERN = re.compile(r"^[a-z0-9]+-[a-z0-9][a-z0-9-]*-[a-z0-9][a-z0-9-]*\.[a-z0-9]+$")
PROJECT_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "just",
    "now",
    "ago",
    "breaking",
}
LEGACY_PROJECT_MARKER_DIRS = ("script", "vo", "description", "voiceover")
PROJECT_STRUCTURE_HINT_DIRS = ("voiceover", "output", "logs", ".cache", "script", "vo", "description")
PROJECT_TITLE_MATCH_SCAN_DIRS = ("voiceover", "output", "lipsync", "transcriptions")
PROJECT_TITLE_MATCH_MAX_FILES_PER_DIR = 24
PROJECT_NAME_TITLE_MIN_SHARED_TOKENS = 4
PROJECT_NAME_TITLE_MIN_JACCARD = 0.55

BLOCKER_ACTION_HINTS = {
    "trello_card_not_in_editing": "Move Trello card to Editing list",
    "missing_raw_voiceover_attachment": "Attach raw voiceover (non-silence-removed)",
    # Legacy lipsync blocker identifiers remain here for older state snapshots.
    "lipsync_not_submitted": "Submit lipsync",
    "lipsync_not_downloaded": "Download lipsync to local project folder",
}

SUBMISSION_WORKFLOW_HINTS = {
    "editing": "Finalize edit deliverables and move card to Edit Review when ready",
    "review": "Address review feedback and advance card to next delivery stage",
    "ready_to_schedule": "Submit/upload final assets and schedule publish",
    "final_checks": "Complete final checks (thumbnail/metadata) and publish",
}

DISCORD_PIPELINE_HEADER_RE = re.compile(
    r"script\s*/\s*voiceover\s*/\s*description\s*--\s*pipeline\s*complete",
    re.IGNORECASE,
)
DISCORD_VIDEO_LINE_RE = re.compile(
    r"^\s*(?:📹\s*)?video\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
DISCORD_FILES_LINE_RE = re.compile(
    r"^\s*(?:📁\s*)?files\s*:\s*<?(https?://drive\.google\.com/drive/folders/[^\s>]+)>?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
DRIVE_FOLDER_ID_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
GOOGLE_DOC_URL_RE = re.compile(
    r"https://docs\.google\.com/document/d/[a-zA-Z0-9_-]+(?:/[^\s<>\"]*)?",
    re.IGNORECASE,
)
DISCORD_RATE_LIMIT_HINTS = (
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "retry after",
)
DISCORD_TRANSIENT_ERROR_HINTS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "internal server error",
    "bad gateway",
    "service unavailable",
    "gateway timeout",
)
DISCORD_NON_RETRYABLE_ERROR_HINTS = (
    "unauthorized",
    "invalid token",
    "forbidden",
    "missing access",
    "missing permission",
)
DEFAULT_DISCORD_EXPORT_MAX_ATTEMPTS = 4
DEFAULT_DISCORD_EXPORT_BASE_BACKOFF_SECONDS = 2.0
DEFAULT_DISCORD_EXPORT_MAX_BACKOFF_SECONDS = 45.0
DEFAULT_DISCORD_EXPORT_TIMEOUT_SECONDS = 180
DEFAULT_DISCORD_CHANNEL_COOLDOWN_SECONDS = 1.5
DEFAULT_DISCORD_INCREMENTAL_LOOKBACK_MINUTES = 20
DEFAULT_PIPELINE_AUTOSTART_WAIT_SECONDS = 3.0
ACTIVE_PIPELINE_PROCESS_CACHE_SECONDS = 2.0
STATE_WRITE_RETRY_DELAYS_SECONDS = (0.15, 0.35, 0.75, 1.5, 3.0)

_psutil = None
_active_pipeline_process_cache: tuple[float, dict[str, list[int]]] | None = None


def now_iso() -> str:
    """Return UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_lower(value: str) -> str:
    """Normalize and lowercase a string for comparisons."""
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def title_tokens(value: str) -> set[str]:
    """Tokenize free text for resilient project-title matching."""
    text = str(value or "").strip().lower()
    if not text:
        return set()
    text = re.sub(r"__\d{4}-\d{2}-\d{2}$", "", text)
    text = re.sub(r"^[a-z0-9]{8,24}-", "", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in PROJECT_TITLE_STOPWORDS
    }


def is_title_match(project_name: str, card_title_tokens: set[str]) -> bool:
    """Return True when project folder name strongly matches card title tokens."""
    if not card_title_tokens:
        return False
    project_tokens = title_tokens(project_name)
    if not project_tokens:
        return False
    shared = len(project_tokens & card_title_tokens)
    if shared < PROJECT_NAME_TITLE_MIN_SHARED_TOKENS:
        return False
    union = len(project_tokens | card_title_tokens)
    if union <= 0:
        return False
    jaccard = shared / union
    return jaccard >= PROJECT_NAME_TITLE_MIN_JACCARD


def iter_project_title_match_texts(project_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield title-bearing strings from a project folder for resilient matching."""
    yield "directory_name", project_dir.name

    try:
        children = sorted(project_dir.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return

    for child in children:
        if child.is_file():
            yield "root_file", child.stem or child.name
            continue
        if not child.is_dir() or normalize_lower(child.name) not in PROJECT_TITLE_MATCH_SCAN_DIRS:
            continue
        try:
            nested_children = sorted(child.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        emitted = 0
        for nested in nested_children:
            if not nested.is_file():
                continue
            yield f"{normalize_lower(child.name)}_file", nested.stem or nested.name
            emitted += 1
            if emitted >= PROJECT_TITLE_MATCH_MAX_FILES_PER_DIR:
                break


def resolve_project_title_match(project_dir: Path, card_title_tokens: set[str]) -> dict[str, Any]:
    """Return best title match details for a project dir using folder and file names."""
    if not card_title_tokens:
        return {}

    seen_texts: set[str] = set()
    best_match: dict[str, Any] = {}
    best_score = (-1, -1.0)

    for source_kind, raw_text in iter_project_title_match_texts(project_dir):
        source_text = str(raw_text or "").strip()
        source_key = normalize_lower(source_text)
        if not source_key or source_key in seen_texts:
            continue
        seen_texts.add(source_key)

        project_tokens = title_tokens(source_text)
        if not project_tokens:
            continue

        shared = len(project_tokens & card_title_tokens)
        if shared < PROJECT_NAME_TITLE_MIN_SHARED_TOKENS:
            continue

        union = len(project_tokens | card_title_tokens)
        if union <= 0:
            continue

        jaccard = shared / union
        if jaccard < PROJECT_NAME_TITLE_MIN_JACCARD:
            continue

        score = (shared, jaccard)
        if score <= best_score:
            continue

        best_score = score
        best_match = {
            "matched": True,
            "source": source_kind,
            "text": source_text,
            "shared_tokens": shared,
            "jaccard": jaccard,
        }

    return best_match


def has_project_structure_hints(project_dir: Path) -> bool:
    """Detect whether a directory looks like a local matcher project."""
    if (project_dir / "trello_card.json").is_file():
        return True
    if has_legacy_script_vo_description_markers(project_dir):
        return True
    for marker in PROJECT_STRUCTURE_HINT_DIRS:
        if (project_dir / marker).exists():
            return True
    for ext in LIKELY_VIDEO_EXTENSIONS:
        for media_file in project_dir.glob(f"*{ext}"):
            if media_file.is_file() and "lipsync" in media_file.name.lower():
                return True
    return False


def has_legacy_script_vo_description_markers(project_dir: Path) -> bool:
    """Detect legacy Script/VO/Description project layouts."""
    try:
        children = {child.name.strip().lower() for child in project_dir.iterdir() if child.is_dir()}
    except OSError:
        return False
    return bool(children & set(LEGACY_PROJECT_MARKER_DIRS))


def console_safe_text(value: str) -> str:
    """Return text safe for current console encoding."""
    text = str(value or "")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def split_csv(value: str) -> list[str]:
    """Split comma-separated text into normalized tokens."""
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part and part.strip()]


def normalize_discord_channel_id(value: str) -> str:
    """Normalize Discord channel IDs from env vars (e.g. '<123...>' -> '123...')."""
    text = str(value or "").strip().strip("'\"")
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    if text.startswith("#"):
        text = text[1:].strip()
    if text.startswith("channel:"):
        text = text.split(":", 1)[1].strip()
    return text


def parse_iso_datetime(value: Any) -> datetime | None:
    """Parse ISO8601-like timestamp to aware datetime."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_member_ids(member_ids: Any) -> list[str]:
    """Normalize Trello member ID lists to stable, non-empty strings."""
    if not isinstance(member_ids, list):
        return []
    normalized: list[str] = []
    for value in member_ids:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def extract_drive_folder_id(url: str) -> str | None:
    """Extract Google Drive folder ID from URL."""
    match = DRIVE_FOLDER_ID_RE.search(str(url or ""))
    if not match:
        return None
    return match.group(1)


def extract_google_doc_urls(text: str) -> list[str]:
    """Extract Google Doc URLs from free text, preserving order."""
    if not text:
        return []
    return list(dict.fromkeys(GOOGLE_DOC_URL_RE.findall(text)))


def build_gws_drive_context(account: dict[str, Any] | None) -> GwsDriveContext:
    """Build per-account gws auth context used for Drive folder inspection."""
    payload = account if isinstance(account, dict) else {}
    return GwsDriveContext(
        token=str(payload.get("gws_token") or "").strip(),
        credentials_file=str(payload.get("gws_credentials_file") or "").strip(),
        impersonated_user=str(payload.get("gws_impersonated_user") or "").strip(),
    )


def has_silence_removed_hint(*values: Any) -> bool:
    """Return True when attachment/file metadata indicates silence-removed audio."""
    blob = " ".join(str(value or "") for value in values).lower()
    return any(hint in blob for hint in SILENCE_REMOVED_HINTS)


def has_voiceover_hint(*values: Any) -> bool:
    """Return True when attachment/file metadata looks voiceover-related."""
    blob = " ".join(str(value or "") for value in values).lower()
    if any(hint in blob for hint in VOICEOVER_HINTS):
        return True
    return VOICEOVER_SHORT_TOKEN_RE.search(blob) is not None


def is_drive_folder_attachment(item: dict[str, Any]) -> bool:
    """Return True when a Trello attachment points at a Google Drive folder."""
    url = str(item.get("url") or "")
    lower_url = url.lower()
    return "drive.google.com" in lower_url and "/folders/" in lower_url


def list_drive_folder_raw_voiceover_candidates(
    attachment: dict[str, Any],
    *,
    gws_context: GwsDriveContext | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Inspect a Drive folder attachment and return raw voiceover audio candidates."""
    folder_url = str(attachment.get("url") or "").strip()
    folder_id = extract_drive_folder_id(folder_url)
    if not folder_id:
        return [], "missing_folder_id"

    try:
        files = list_drive_folder_files(folder_id, context=gws_context or GwsDriveContext())
    except GwsDriveError as exc:
        return [], f"gws:{exc.code}"
    except Exception as exc:
        return [], f"unexpected:{exc}"

    candidates: list[dict[str, Any]] = []
    for item in files:
        file_name = str(item.get("name") or "").strip()
        file_id = str(item.get("id") or "").strip()
        mime_type = str(item.get("mimeType") or "").strip()
        if not file_name or not file_id:
            continue
        if has_silence_removed_hint(file_name, mime_type):
            continue
        if not is_audio_attachment(file_name, file_name, mime_type):
            continue

        candidates.append(
            {
                "id": file_id,
                "name": file_name,
                "url": f"https://drive.google.com/file/d/{file_id}/view",
                "mimeType": mime_type,
                "bytes": item.get("size"),
                "is_drive_folder": False,
                "is_drive_folder_item": True,
                "source": "drive_folder",
                "folder_id": folder_id,
                "folder_name": str(attachment.get("name") or "").strip(),
                "folder_url": folder_url,
            }
        )

    return candidates, None


def normalize_discord_title(value: str) -> str:
    """Normalize Discord video title for safer project naming."""
    text = str(value or "").strip()
    if not text:
        return "Untitled Discord Project"
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        text = text[1:-1].strip()
    return text or "Untitled Discord Project"


def channel_project_dirs(channel: str) -> list[Path]:
    """Resolve local channel directories for a Trello channel code."""
    channel_upper = (channel or "").upper()
    candidates: list[str] = list(CHANNEL_DIR_ALIASES.get(channel_upper, ()))

    if channel and channel_upper not in CHANNEL_DIR_ALIASES:
        candidates.append(channel)

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalize_lower(candidate)
        if key in seen:
            continue
        seen.add(key)
        path = LOCAL_PROJECTS_ROOT / candidate
        if path.exists():
            resolved.append(path)
    return resolved


def find_local_project_dirs_by_title(channel: str, project_title: str) -> list[str]:
    """Find existing local project directories whose names strongly match a project title."""
    token_set = title_tokens(project_title)
    if not token_set:
        return []

    matches: list[str] = []
    for channel_dir in channel_project_dirs(channel):
        for project_dir in sorted(channel_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            if not has_project_structure_hints(project_dir):
                continue
            if resolve_project_title_match(project_dir, token_set):
                matches.append(str(project_dir))

    return sorted(set(matches), key=lambda value: value.lower())


def resolve_local_project_dir_matches(
    channel: str,
    card_id: str,
    full_card_id: str | None = None,
    card_title: str = "",
    *,
    allow_cross_channel_card_ids: bool = True,
) -> list[dict[str, Any]]:
    """Find candidate local project directories, preferring richer/canonical copies."""
    match_keys = {str(card_id or "").strip().lower()}
    if full_card_id:
        match_keys.add(str(full_card_id).strip().lower())
    match_keys.discard("")
    card_title_token_set = title_tokens(card_title)

    candidate_map: dict[str, dict[str, Any]] = {}
    expected_roots = channel_project_dirs(channel)
    expected_root_keys = {str(path).strip().lower() for path in expected_roots}

    def register_candidate(project_dir: Path, *, allow_title_match: bool, require_id_match: bool) -> None:
        if not project_dir.is_dir():
            return
        if not has_project_structure_hints(project_dir):
            return

        metadata_card_id = extract_project_card_id(project_dir)
        id_match = bool(match_keys) and (
            contains_card_id_token(project_dir.name, match_keys)
            or contains_card_id_token(metadata_card_id, match_keys)
        )
        title_match_details = {}
        if allow_title_match and card_title_token_set:
            title_match_details = resolve_project_title_match(project_dir, card_title_token_set)
        title_match = bool(title_match_details)

        if require_id_match and not id_match:
            return
        if not id_match and not title_match:
            return

        path_str = str(project_dir)
        path_key = path_str.strip().lower()
        candidate = {
            "path": path_str,
            "channel_match": str(project_dir.parent).strip().lower() in expected_root_keys,
            "id_match": id_match,
            "title_match": title_match,
            "quality_score": project_dir_quality_score(project_dir),
        }
        if title_match_details:
            candidate["title_match_source"] = str(title_match_details.get("source") or "")
            candidate["title_match_text"] = str(title_match_details.get("text") or "")
        existing = candidate_map.get(path_key)
        if existing:
            existing["channel_match"] = bool(existing.get("channel_match")) or candidate["channel_match"]
            existing["id_match"] = bool(existing.get("id_match")) or candidate["id_match"]
            existing["title_match"] = bool(existing.get("title_match")) or candidate["title_match"]
            existing["quality_score"] = max(
                int(existing.get("quality_score") or 0),
                int(candidate.get("quality_score") or 0),
            )
            if title_match_details and not existing.get("title_match_source"):
                existing["title_match_source"] = candidate.get("title_match_source")
                existing["title_match_text"] = candidate.get("title_match_text")
            return
        candidate_map[path_key] = candidate

    for channel_dir in expected_roots:
        try:
            project_dirs = sorted(channel_dir.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for project_dir in project_dirs:
            register_candidate(project_dir, allow_title_match=True, require_id_match=False)

    if allow_cross_channel_card_ids and match_keys and not any(
        is_strong_project_candidate(candidate)
        for candidate in candidate_map.values()
    ):
        for root in iter_local_project_roots(LOCAL_PROJECTS_ROOT, ARCHIVED_PROJECTS_DIR_PREFIX):
            if str(root).strip().lower() in expected_root_keys:
                continue
            try:
                project_dirs = sorted(root.iterdir(), key=lambda item: item.name.lower())
            except OSError:
                continue
            for project_dir in project_dirs:
                register_candidate(project_dir, allow_title_match=False, require_id_match=True)

    return sort_project_match_candidates(list(candidate_map.values()))


def collect_pipeline_drive_folder_ids(entry: dict[str, Any]) -> set[str]:
    """Collect normalized Drive folder IDs from pipeline raw-voiceover candidates."""
    start_checks = entry.get("start_checks", {}) if isinstance(entry, dict) else {}
    raw_candidates = start_checks.get("raw_voiceover_candidates", []) if isinstance(start_checks, dict) else []
    folder_ids: set[str] = set()
    if not isinstance(raw_candidates, list):
        return folder_ids

    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        folder_id = str(candidate.get("folder_id") or "").strip().lower()
        if folder_id:
            folder_ids.add(folder_id)
        folder_url = str(candidate.get("folder_url") or "").strip()
        extracted = extract_drive_folder_id(folder_url)
        if extracted:
            folder_ids.add(extracted.lower())
    return folder_ids


def resolve_discord_candidate_trello_card(
    state: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, str]:
    """
    Resolve a Discord pipeline-complete candidate back to a Trello card when possible.

    Resolution order:
    1) Exact Drive folder ID match against Trello raw-voiceover candidates.
    2) Unique strong title match within the same channel.
    """
    payload = state if isinstance(state, dict) else {}
    pipelines = payload.get("pipelines", {}) if isinstance(payload.get("pipelines"), dict) else {}
    if not pipelines:
        return {}

    candidate_channel = str(candidate.get("channel") or "").strip().upper()
    candidate_folder_id = str(candidate.get("drive_folder_id") or "").strip().lower()
    candidate_title = str(candidate.get("video_title") or "")
    candidate_tokens = title_tokens(candidate_title)

    drive_matches: dict[str, dict[str, str]] = {}
    title_matches: list[tuple[int, float, str, dict[str, str]]] = []

    for entry in pipelines.values():
        if not isinstance(entry, dict):
            continue

        project = entry.get("project", {}) if isinstance(entry.get("project"), dict) else {}
        entry_channel = str(project.get("channel") or "").strip().upper()
        if candidate_channel and entry_channel and entry_channel != candidate_channel:
            continue

        card_id = str(entry.get("card_id") or "").strip()
        if not card_id:
            continue

        resolved = {
            "card_id": card_id,
            "card_url": resolve_card_url(entry, card_id),
            "match_strategy": "",
        }

        if candidate_folder_id and candidate_folder_id in collect_pipeline_drive_folder_ids(entry):
            resolved["match_strategy"] = "drive_folder_id"
            drive_matches[card_id.lower()] = resolved
            continue

        title = str(entry.get("title") or "").strip()
        entry_tokens = title_tokens(title)
        if not candidate_tokens or not entry_tokens:
            continue

        shared = len(candidate_tokens & entry_tokens)
        if shared < PROJECT_NAME_TITLE_MIN_SHARED_TOKENS:
            continue

        union = len(candidate_tokens | entry_tokens)
        jaccard = shared / max(1, union)
        if jaccard < PROJECT_NAME_TITLE_MIN_JACCARD:
            continue

        resolved["match_strategy"] = "title_match"
        title_matches.append((shared, jaccard, card_id.lower(), resolved))

    if len(drive_matches) == 1:
        return next(iter(drive_matches.values()))

    if title_matches:
        title_matches.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best = title_matches[0]
        if len(title_matches) == 1:
            return best[3]
        runner_up = title_matches[1]
        if best[0] > runner_up[0] or best[1] > runner_up[1]:
            return best[3]

    return {}


def list_bucket(list_name: str) -> str:
    """Map Trello list names to canonical workflow buckets."""
    value = normalize_lower(list_name)
    if value in SCRIPT_VO_DESCRIPTION_LIST_NAMES:
        return "script_vo_description"
    if value in PIPELINE_READY_LIST_NAMES:
        return "pipeline_ready"
    if (
        "script" in value
        and "description" in value
        and ("voiceover" in value or re.search(r"\bvo\b", value) is not None)
    ):
        return "script_vo_description"
    if (
        "script" in value
        and ("voiceover" in value or re.search(r"\bvo'?s?\b", value) is not None)
        and "description" not in value
    ):
        return "pipeline_ready"
    if value in EDITING_LIST_NAMES:
        return "editing"
    if value in REVIEW_LIST_NAMES or "review" in value:
        return "review"
    if value in READY_TO_SCHEDULE_LIST_NAMES:
        return "ready_to_schedule"
    if value in FINAL_CHECK_LIST_NAMES:
        return "final_checks"
    if value in ARCHIVE_LIST_NAMES or value in READY_TO_UPLOAD_ARCHIVE_LIST_NAMES:
        return "archived"
    if value in NOT_STARTED_LIST_NAMES:
        return "not_started"
    if value in PENDING_LIST_NAMES:
        return "pending"
    if value in COMPLETED_LIST_NAMES:
        return "completed"
    return "other"


def is_pipeline_required(workflow_state: str) -> bool:
    """Return True when pipeline run work is expected for this Trello workflow state."""
    if workflow_state == "script_vo_description":
        # Script/VO/Description cards should be project-prepared, but they do not require a pipeline run yet.
        return False
    return workflow_state in {"pending", "not_started", "pipeline_ready", "editing", "review"}


def derive_project_state(workflow_state: str) -> str:
    """Project-level status from Trello workflow."""
    if workflow_state in {"archived", "completed"}:
        return "project_completed"
    if workflow_state in {"ready_to_schedule", "final_checks"}:
        return "project_submission_flow"
    if workflow_state in {"editing", "review", "pending", "not_started", "script_vo_description", "pipeline_ready"}:
        return "project_active"
    return "project_unknown"


def should_archive_project_dir(list_name: str, workflow_state: str) -> bool:
    """Return True when a Trello workflow state is ready for local project archival."""
    normalized_list = normalize_lower(list_name)
    normalized_state = normalize_lower(workflow_state or list_bucket(list_name))
    if normalized_list in READY_TO_UPLOAD_ARCHIVE_LIST_NAMES:
        return True
    return normalized_state in {"ready_to_schedule", "archived", "completed"}


def resolve_archived_projects_dir() -> Path:
    """Pick the latest archive root under the local Degold projects root."""
    candidates: list[Path] = []
    prefix = ARCHIVED_PROJECTS_DIR_PREFIX.lower()
    if LOCAL_PROJECTS_ROOT.exists():
        for child in LOCAL_PROJECTS_ROOT.iterdir():
            if child.is_dir() and child.name.lower().startswith(prefix):
                candidates.append(child)
    if candidates:
        return sorted(candidates, key=lambda path: normalize_lower(path.name), reverse=True)[0]
    stamp = datetime.now().strftime("%Y-%m-%d")
    return LOCAL_PROJECTS_ROOT / f"{ARCHIVED_PROJECTS_DIR_PREFIX}_{stamp}"


def is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for Path.is_relative_to()."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def build_archived_project_destination(project_dir: Path, archive_root: Path) -> Path:
    """Build a collision-safe archive destination while preserving the source channel folder."""
    destination_parent = archive_root / (project_dir.parent.name or "projects")
    destination = destination_parent / project_dir.name
    if not destination.exists():
        return destination

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = destination_parent / f"{project_dir.name}__archived_{suffix}"
    counter = 2
    while candidate.exists():
        candidate = destination_parent / f"{project_dir.name}__archived_{suffix}_{counter}"
        counter += 1
    return candidate


def archive_completed_project_dirs(state: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Move local project dirs for Ready To Upload and later Trello cards into archive storage."""
    archive_root = resolve_archived_projects_dir()
    actions: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()
    pipelines = state.get("pipelines") or {}

    for entry in sorted(pipelines.values(), key=lambda item: str((item or {}).get("card_id") or "")):
        if not isinstance(entry, dict):
            continue

        trello_list = ((entry.get("trello") or {}).get("list") or {})
        list_name = str(trello_list.get("name") or "")
        workflow_state = str(entry.get("workflow_state") or trello_list.get("bucket") or "")
        if not should_archive_project_dir(list_name, workflow_state):
            continue

        project = entry.get("project") or {}
        local_dirs = (project.get("local_project_dirs", []) or [])
        if not local_dirs:
            trello = entry.get("trello") or {}
            card_raw = trello.get("card_raw") or {}
            recovered_matches = resolve_local_project_dir_matches(
                str(project.get("channel") or ""),
                str(entry.get("card_id") or ""),
                full_card_id=str(card_raw.get("id") or "").strip() or None,
                card_title=str(entry.get("title") or ""),
            )
            local_dirs = [
                str(candidate.get("path") or "").strip()
                for candidate in recovered_matches
                if str(candidate.get("path") or "").strip()
            ]
        for project_dir_raw in local_dirs:
            source = Path(str(project_dir_raw))
            source_key = str(source).strip().lower()
            if not source_key or source_key in seen_dirs:
                continue
            seen_dirs.add(source_key)

            action = {
                "card_id": str(entry.get("card_id") or ""),
                "title": str(entry.get("title") or ""),
                "trello_list": list_name,
                "workflow_state": workflow_state or list_bucket(list_name),
                "source": str(source),
                "status": "skipped",
            }

            if not source.exists() or not source.is_dir():
                action["status"] = "missing"
                actions.append(action)
                continue

            if is_relative_to(source, archive_root):
                action["status"] = "already_archived"
                action["destination"] = str(source)
                actions.append(action)
                continue

            destination = build_archived_project_destination(source, archive_root)
            action["destination"] = str(destination)

            if dry_run:
                action["status"] = "would_move"
                actions.append(action)
                continue

            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                move(str(source), str(destination))
                action["status"] = "moved"
            except OSError as exc:
                action["status"] = "error"
                action["error"] = str(exc)
            actions.append(action)

    summary = Counter(str(action.get("status") or "unknown") for action in actions)
    return {
        "archive_root": str(archive_root),
        "actions": actions,
        "summary": dict(summary),
    }


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


def load_accounts(accounts_dir: Path) -> list[dict[str, Any]]:
    """Load Trello credentials from Degold/accounts/*.env."""
    accounts: list[dict[str, Any]] = []
    if not accounts_dir.exists():
        return accounts

    for account_file in sorted(accounts_dir.glob("*.env")):
        if account_file.name.lower() == "example.env":
            continue
        env_data = parse_env_file(account_file)
        api_key = env_data.get("TRELLO_API_KEY")
        token = env_data.get("TRELLO_TOKEN")
        if not api_key or not token:
            continue
        account_name = account_file.stem
        configured_channel = default_channel_for_account(
            account_name,
            str(env_data.get("DEFAULT_CHANNEL") or ""),
        )
        raw_discord_channels = split_csv(
            str(env_data.get("DISCORD_CHANNEL_IDS") or env_data.get("DISCORD_CHANNELS") or "").strip()
        )
        discord_channels: list[str] = []
        seen_channels: set[str] = set()
        for raw_channel in raw_discord_channels:
            channel_id = normalize_discord_channel_id(raw_channel)
            if not channel_id or channel_id in seen_channels:
                continue
            seen_channels.add(channel_id)
            discord_channels.append(channel_id)
        accounts.append(
            {
                "name": account_name,
                "display_name": account_name.title(),
                "api_key": api_key,
                "token": token,
                "gws_token": str(env_data.get("GOOGLE_WORKSPACE_CLI_TOKEN") or "").strip(),
                "gws_credentials_file": str(env_data.get("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE") or "").strip(),
                "gws_impersonated_user": str(env_data.get("GOOGLE_WORKSPACE_CLI_IMPERSONATED_USER") or "").strip(),
                "default_channel": configured_channel,
                "discord_token": str(env_data.get("DISCORD_TOKEN") or "").strip(),
                "discord_channel_ids": discord_channels,
                "discord_exporter": str(
                    env_data.get("DISCORD_CHAT_EXPORTER")
                    or env_data.get("DISCORD_CHAT_EXPORTER_PATH")
                    or ""
                ).strip(),
                "board_id": str(env_data.get("TRELLO_BOARD_ID") or "").strip(),
            }
        )
    return accounts


def load_board_channel_map(path: Path) -> dict[str, dict[str, str]]:
    """Load board routing metadata from YAML file."""
    return load_shared_board_channel_map(PROJECT_ROOT, preferred_path=path)


def run_lipsync_refresh() -> None:
    """Refresh Degold/lipsync_tracking.json by running track_lipsync.py."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "track_lipsync.py")]
    result = run_subprocess(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"track_lipsync.py failed with exit code {result.returncode}: {result.stderr.strip()}")
    if result.stdout.strip():
        print_info("Refreshed lipsync tracking data")


def load_lipsync_tracking(path: Path) -> dict[str, dict[str, Any]]:
    """Load indexed lipsync tracking by card_id."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    indexed: dict[str, dict[str, Any]] = {}
    for card in payload.get("cards", []):
        card_id = str(card.get("card_id", "")).strip().lower()
        if card_id:
            indexed[card_id] = card
    return indexed


def check_live_lipsync_drive_status(channel: str, card_title: str, card_id: str) -> dict[str, Any]:
    """Query Google Drive directly for the current lipsync status."""
    try:
        from track_lipsync import check_drive_for_lipsync
    except Exception as exc:
        return {"status": "error", "error": f"drive_check_import_failed: {exc}"}

    try:
        result = check_drive_for_lipsync(channel, card_title, card_id)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    return result if isinstance(result, dict) else {"status": "unknown"}


def resolve_lipsync_submission_status(
    tracked_lipsync: dict[str, Any],
    *,
    channel: str,
    card_id: str,
    card_title: str,
    local_scan: dict[str, Any],
    probe_live_drive: bool = True,
) -> tuple[dict[str, Any], bool, bool]:
    """Resolve current lipsync status, probing Drive before concluding it is not submitted."""
    effective = dict(tracked_lipsync) if isinstance(tracked_lipsync, dict) else {}

    local_status = str(local_scan.get("status") or "").strip().lower()
    if local_status == "downloaded":
        effective["local_status"] = "downloaded"
        effective.setdefault("drive_status_source", "local_download")
        return effective, True, True

    drive_status = str(effective.get("drive_status") or "").strip().lower()
    if drive_status != "complete" and probe_live_drive:
        live_status = check_live_lipsync_drive_status(channel, card_title, card_id)
        live_drive_status = str(live_status.get("status") or "").strip()
        if live_drive_status:
            effective["drive_status"] = live_drive_status
        if "files" in live_status:
            effective["drive_files"] = live_status.get("files") or []
        if "match_strategy" in live_status:
            effective["drive_match_strategy"] = live_status.get("match_strategy")
        if "checked_folders" in live_status:
            effective["drive_checked_folders"] = live_status.get("checked_folders")
        if live_status.get("error"):
            effective["drive_error"] = live_status.get("error")
        effective["drive_status_source"] = "live_check"
    else:
        effective.setdefault("drive_status_source", "tracking")

    local_tracked_status = str(effective.get("local_status") or "").strip().lower()
    submitted = str(effective.get("drive_status") or "").strip().lower() == "complete"
    downloaded = local_tracked_status == "downloaded"
    return effective, submitted, downloaded


# Circuit breaker for external services
class CircuitBreaker:
    """Simple circuit breaker for external API calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self.last_failure_time is not None:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker is open, retry after timeout")
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as exc:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "open"
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and refusing calls."""
    pass


# Global circuit breaker for Trello API
_trello_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)


def trello_request_with_retry(
    method: Callable[..., requests.Response],
    endpoint: str,
    retries: int = 3,
    backoff: int = 2,
    **kwargs: Any
) -> dict[str, Any]:
    """Make Trello API call with exponential backoff for transient failures."""
    for attempt in range(retries):
        try:
            response = method(endpoint, **kwargs)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print_warn(f"Trello rate limited (429), waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            if attempt >= retries - 1:
                raise
            # Check for server errors (5xx)
            if hasattr(exc, 'response') and exc.response is not None:
                if exc.response.status_code >= 500:
                    sleep_time = backoff ** attempt
                    print_warn(f"Trello server error ({exc.response.status_code}), retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
            # Re-raise for client errors (4xx except 429) or network issues
            raise


def trello_get(endpoint: str, api_key: str, token: str, params: dict[str, Any]) -> Any:
    """Call Trello API and return JSON response with retry logic."""
    merged = dict(params)
    merged["key"] = api_key
    merged["token"] = token

    def _do_get() -> requests.Response:
        return requests.get(
            endpoint,
            params=merged,
            timeout=30,
        )

    try:
        return trello_request_with_retry(_do_get, endpoint)
    except Exception:
        # Fallback to direct call if circuit breaker fails
        response = requests.get(
            endpoint,
            params=merged,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def get_me(account: dict[str, str]) -> dict[str, Any]:
    """Get Trello member profile for current token."""
    return trello_get(
        "https://api.trello.com/1/members/me",
        account["api_key"],
        account["token"],
        {"fields": "id,fullName,username"},
    )


def get_boards(account: dict[str, str]) -> list[dict[str, Any]]:
    """Get all boards accessible to the account."""
    return trello_get(
        "https://api.trello.com/1/members/me/boards",
        account["api_key"],
        account["token"],
        {"fields": "id,name,closed,url"},
    )


def get_board_lists(account: dict[str, str], board_id: str) -> list[dict[str, Any]]:
    """Get lists in a board."""
    return trello_get(
        f"https://api.trello.com/1/boards/{board_id}/lists",
        account["api_key"],
        account["token"],
        {"fields": "id,name,closed,pos"},
    )


def get_board_cards(account: dict[str, str], board_id: str) -> list[dict[str, Any]]:
    """Get cards in a board with summary fields."""
    return trello_get(
        f"https://api.trello.com/1/boards/{board_id}/cards",
        account["api_key"],
        account["token"],
        {
            "fields": "id,shortLink,shortUrl,name,idList,idMembers,idBoard,closed,due,dateLastActivity,url,labels",
            "filter": "visible",
        },
    )


def get_card_detail(account: dict[str, str], card_id: str) -> dict[str, Any]:
    """Get full card payload for tracking."""
    return trello_get(
        f"https://api.trello.com/1/cards/{card_id}",
        account["api_key"],
        account["token"],
        {
            "fields": "all",
            "attachments": "true",
            "attachment_fields": "all",
            "members": "true",
            "member_fields": "all",
            "checklists": "all",
            "customFieldItems": "true",
            "pluginData": "true",
        },
    )


def is_audio_attachment(name: str, url: str, mime_type: str) -> bool:
    """Heuristic audio attachment detector."""
    blob = f"{name} {url} {mime_type}".lower()
    if any(blob.endswith(ext) or ext in blob for ext in LIKELY_AUDIO_EXTENSIONS):
        return True
    if str(mime_type).lower().startswith("audio/"):
        return True
    return False


def extract_raw_voiceover_candidates(
    attachments: list[dict[str, Any]],
    *,
    gws_context: GwsDriveContext | None = None,
) -> list[dict[str, Any]]:
    """Return attachments that look like raw (non-silence-removed) voiceovers."""
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    drive_folder_cache: dict[str, tuple[list[dict[str, Any]], str | None]] = {}

    def append_candidate(payload: dict[str, Any]) -> None:
        key = (
            str(payload.get("id") or "").strip().lower(),
            str(payload.get("url") or "").strip().lower(),
            str(payload.get("name") or "").strip().lower(),
        )
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append(payload)

    for item in attachments:
        name = str(item.get("name", ""))
        url = str(item.get("url", ""))
        mime_type = str(item.get("mimeType", ""))
        drive_folder_like = is_drive_folder_attachment(item)

        if has_silence_removed_hint(name, url, mime_type):
            continue

        has_vo_hint = has_voiceover_hint(name, url)
        audio_like = is_audio_attachment(name, url, mime_type)

        if not drive_folder_like and (has_vo_hint or audio_like):
            append_candidate(
                {
                    "id": item.get("id"),
                    "name": name,
                    "url": url,
                    "mimeType": mime_type,
                    "bytes": item.get("bytes"),
                    "is_drive_folder": False,
                    "source": "trello_attachment",
                }
            )
            continue

        if not drive_folder_like:
            continue

        folder_id = extract_drive_folder_id(url) or f"attachment:{item.get('id') or name}"
        folder_candidates, folder_error = drive_folder_cache.get(folder_id, (None, None))
        if folder_candidates is None:
            folder_candidates, folder_error = list_drive_folder_raw_voiceover_candidates(
                item,
                gws_context=gws_context,
            )
            drive_folder_cache[folder_id] = (folder_candidates, folder_error)

        for folder_candidate in folder_candidates:
            append_candidate(folder_candidate)

        # Preserve previous behavior for obvious VO folders when Drive inspection
        # cannot run, but prefer actual folder contents when available.
        if folder_candidates or not folder_error or not has_vo_hint:
            continue

        append_candidate(
            {
                "id": item.get("id"),
                "name": name,
                "url": url,
                "mimeType": mime_type,
                "bytes": item.get("bytes"),
                "is_drive_folder": True,
                "source": "trello_drive_folder_unverified",
                "drive_scan_error": folder_error,
            }
        )
    return candidates


def extract_description_google_doc_voiceover_candidates(description: str) -> list[dict[str, Any]]:
    """Extract Google Doc voiceover candidates from a Trello card description.

    Only Google Docs on lines labeled as voiceover (containing "VO" or "VOICE OVER"
    before the URL) are treated as VO candidates.  Lines that only reference a script
    (e.g. "Script 3 - [link]") are ignored.

    If no VO-labeled lines are found, falls back to treating all Google Docs as
    candidates for backward compatibility with boards that don't use this convention.
    """
    if not description:
        return []

    vo_line_re = re.compile(
        r"^[^\n]*\bvo(?:ice\s*over)?\b[^\n]*"
        r"(https://docs\.google\.com/document/d/[a-zA-Z0-9_-]+(?:/[^\s<>\"]*)?)",
        re.IGNORECASE | re.MULTILINE,
    )
    vo_urls: list[str] = list(dict.fromkeys(vo_line_re.findall(description)))

    # When VO-labeled lines exist, use only those.
    if vo_urls:
        candidates: list[dict[str, Any]] = []
        for index, doc_url in enumerate(vo_urls, start=1):
            candidates.append(
                {
                    "id": f"description-vo-doc-{index}",
                    "name": f"description_vo_google_doc_{index}",
                    "url": doc_url,
                    "mimeType": "application/vnd.google-apps.document",
                    "bytes": None,
                    "is_drive_folder": False,
                    "is_google_doc": True,
                    "source": "trello_description_google_doc",
                }
            )
        return candidates

    # Check whether the description uses the "Script N - " / "Script N - VO - "
    # convention at all.  If it does but no VO lines had links, there are no VOs.
    script_label_re = re.compile(r"Script\s*\d+\s*-", re.IGNORECASE)
    if script_label_re.search(description):
        return []

    # Fallback: no labeling convention detected — treat all docs as VO candidates
    # (backward compat for boards that don't use the Script/VO naming).
    candidates = []
    for index, doc_url in enumerate(extract_google_doc_urls(description), start=1):
        candidates.append(
            {
                "id": f"description-doc-{index}",
                "name": f"description_google_doc_{index}",
                "url": doc_url,
                "mimeType": "application/vnd.google-apps.document",
                "bytes": None,
                "is_drive_folder": False,
                "is_google_doc": True,
                "source": "trello_description_google_doc",
            }
        )
    return candidates


def extract_card_raw_voiceover_candidates(
    card_payload: dict[str, Any],
    *,
    workflow_state: str,
    gws_context: GwsDriveContext | None = None,
) -> list[dict[str, Any]]:
    """Collect raw-voiceover candidates from card attachments and supported description sources."""
    payload = card_payload if isinstance(card_payload, dict) else {}
    candidates = extract_raw_voiceover_candidates(
        payload.get("attachments") or [],
        gws_context=gws_context,
    )
    if candidates:
        return candidates
    if workflow_state not in {"script_vo_description", "pipeline_ready"}:
        return candidates
    return extract_description_google_doc_voiceover_candidates(str(payload.get("desc") or ""))


def collect_project_lipsync_video_files(project_dir: Path) -> list[str]:
    """Collect local lipsync video files for one project directory."""
    files: set[str] = set()
    for sub in ("output", "lipsync"):
        subdir = project_dir / sub
        if not subdir.exists() or not subdir.is_dir():
            continue
        for ext in LIKELY_VIDEO_EXTENSIONS:
            for media_file in subdir.glob(f"*{ext}"):
                if media_file.is_file():
                    files.add(str(media_file))
    # Some legacy/manual downloads place lipsync_video directly in project root.
    for ext in LIKELY_VIDEO_EXTENSIONS:
        for media_file in project_dir.glob(f"*{ext}"):
            if media_file.is_file() and "lipsync" in media_file.name.lower():
                files.add(str(media_file))
    return sorted(files)


def find_local_lipsync_download(
    channel: str,
    card_id: str,
    full_card_id: str | None = None,
    card_title: str = "",
) -> dict[str, Any]:
    """Check local Degold project directories for downloaded lipsync outputs."""
    matched_dirs: list[str] = []
    files: list[str] = []
    for candidate in resolve_local_project_dir_matches(
        channel,
        card_id,
        full_card_id=full_card_id,
        card_title=card_title,
    ):
        path_str = str(candidate.get("path") or "").strip()
        if not path_str:
            continue
        project_dir = Path(path_str)
        matched_dirs.append(path_str)
        files.extend(collect_project_lipsync_video_files(project_dir))

    status = "downloaded" if files else "not_downloaded"
    return {"status": status, "project_dirs": matched_dirs, "files": sorted(set(files))}


def _read_tail_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    """Read only the tail of a text file for fast marker scans."""
    size = path.stat().st_size
    offset = max(0, size - max_bytes)
    with open(path, "rb") as handle:
        if offset:
            handle.seek(offset)
        chunk = handle.read()
    return chunk.decode("utf-8", errors="replace").lower()


def normalize_path_for_compare(value: str) -> str:
    """Normalize filesystem paths for case-insensitive matching."""
    text = str(value or "").strip().strip("\"")
    if not text:
        return ""
    return os.path.normcase(os.path.normpath(text))


def _get_psutil_module():
    """Lazily import psutil for process inspection."""
    global _psutil
    if _psutil is False:
        return None
    if _psutil is None:
        try:
            import psutil

            _psutil = psutil
        except ImportError:
            _psutil = False
            return None
    return _psutil


def extract_project_arg_from_cmdline(cmdline: list[str]) -> str:
    """Extract the `--project` value from a main.py command line."""
    for index, part in enumerate(cmdline):
        text = str(part or "").strip()
        if text == "--project" and index + 1 < len(cmdline):
            return str(cmdline[index + 1] or "").strip()
        if text.startswith("--project="):
            return text.split("=", 1)[1].strip()
    return ""


def list_active_pipeline_processes() -> dict[str, list[int]] | None:
    """Return running `main.py --project ...` processes indexed by normalized project path."""
    global _active_pipeline_process_cache
    psutil_module = _get_psutil_module()
    if psutil_module is None:
        return None

    now_ts = time.time()
    if _active_pipeline_process_cache is not None:
        cached_at, cached_index = _active_pipeline_process_cache
        if now_ts - cached_at <= ACTIVE_PIPELINE_PROCESS_CACHE_SECONDS:
            return cached_index

    process_index: dict[str, list[int]] = {}
    try:
        for proc in psutil_module.process_iter(["pid", "cmdline"]):
            try:
                cmdline = [str(part or "") for part in (proc.info.get("cmdline") or [])]
            except (psutil_module.Error, OSError):
                continue
            if not cmdline:
                continue
            joined = " ".join(cmdline).lower()
            if "main.py" not in joined or "--project" not in joined:
                continue
            project_arg = normalize_path_for_compare(extract_project_arg_from_cmdline(cmdline))
            if not project_arg:
                continue
            process_index.setdefault(project_arg, []).append(int(proc.info.get("pid") or 0))
    except (psutil_module.Error, OSError):
        return None

    _active_pipeline_process_cache = (now_ts, process_index)
    return process_index


def detect_output_completion_evidence(project_dir: Path) -> str | None:
    """Return an output artifact path when output stage completion is detected."""
    output_dir = project_dir / "output"
    if not output_dir.exists() or not output_dir.is_dir():
        return None

    candidate_dirs: list[Path] = [output_dir]
    subdirs = [entry for entry in output_dir.iterdir() if entry.is_dir()]
    candidate_dirs.extend(sorted(subdirs, key=lambda item: item.stat().st_mtime, reverse=True))

    best: tuple[float, str] | None = None
    for candidate_dir in candidate_dirs:
        matched_files: list[Path] = []
        for filename in OUTPUT_COMPLETION_EVIDENCE_FILES:
            path = candidate_dir / filename
            if path.exists() and path.is_file():
                matched_files.append(path)
        if len(matched_files) < OUTPUT_COMPLETION_MIN_MATCHES:
            continue

        newest = max(matched_files, key=lambda item: item.stat().st_mtime)
        newest_mtime = newest.stat().st_mtime
        if best is None or newest_mtime > best[0]:
            best = (newest_mtime, str(newest))

    if best is None:
        return None
    return best[1]


def detect_local_pipeline_progress(project_dirs: list[str]) -> dict[str, Any]:
    """Detect whether local project(s) have started or completed the pipeline."""
    progress: dict[str, Any] = {
        "has_project_dir": bool(project_dirs),
        "has_run_logs": False,
        "completed_signal": False,
        "completion_log": None,
        "completion_source": None,
        "running_signal": False,
        "active_process_signal": False,
        "active_process_ids": [],
        "process_check_available": False,
        "recent_log_activity": False,
        "latest_log": None,
        "latest_log_updated_at": None,
        "latest_log_age_seconds": None,
        "output_completion_signal": False,
        "output_completion_evidence": None,
    }
    if not project_dirs:
        return progress

    process_index = list_active_pipeline_processes()
    if process_index is not None:
        active_process_ids: list[int] = []
        for project_dir_raw in project_dirs:
            normalized_path = normalize_path_for_compare(project_dir_raw)
            if not normalized_path:
                continue
            active_process_ids.extend(process_index.get(normalized_path, []))
        progress["process_check_available"] = True
        progress["active_process_ids"] = sorted({pid for pid in active_process_ids if pid})
        progress["active_process_signal"] = bool(progress["active_process_ids"])

    latest_log_path: Path | None = None
    latest_log_mtime: float | None = None
    best_output_completion: tuple[float, str] | None = None
    for project_dir_raw in project_dirs:
        project_dir = Path(project_dir_raw)
        logs_dir = project_dir / "logs"

        output_evidence = detect_output_completion_evidence(project_dir)
        if output_evidence:
            evidence_path = Path(output_evidence)
            evidence_mtime = evidence_path.stat().st_mtime
            if best_output_completion is None or evidence_mtime > best_output_completion[0]:
                best_output_completion = (evidence_mtime, str(evidence_path))

        if not logs_dir.exists():
            continue

        run_logs = sorted(
            [path for path in logs_dir.glob("run_*.log") if path.is_file()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not run_logs:
            continue

        progress["has_run_logs"] = True
        first_log = run_logs[0]
        first_log_mtime = first_log.stat().st_mtime
        if latest_log_mtime is None or first_log_mtime > latest_log_mtime:
            latest_log_mtime = first_log_mtime
            latest_log_path = first_log

    if latest_log_path is not None and latest_log_mtime is not None:
        latest_text = _read_tail_text(latest_log_path)
        if any(marker in latest_text for marker in COMPLETION_LOG_MARKERS):
            progress["completed_signal"] = True
            progress["completion_log"] = str(latest_log_path)
            progress["completion_source"] = "log_marker"

        now_ts = datetime.now(timezone.utc).timestamp()
        age_seconds = max(0, int(now_ts - latest_log_mtime))
        progress["latest_log"] = str(latest_log_path)
        progress["latest_log_updated_at"] = datetime.fromtimestamp(
            latest_log_mtime,
            tz=timezone.utc,
        ).isoformat()
        progress["latest_log_age_seconds"] = age_seconds
        progress["recent_log_activity"] = age_seconds <= RUNNING_LOG_ACTIVITY_WINDOW_SECONDS

    if best_output_completion is not None:
        progress["output_completion_signal"] = True
        progress["output_completion_evidence"] = best_output_completion[1]
        if not progress["completed_signal"]:
            progress["completed_signal"] = True
            progress["completion_log"] = best_output_completion[1]
            progress["completion_source"] = "output_artifact"

    # Completion evidence wins over recent log churn. Completed projects should not
    # continue to block the queue as "running" just because a log was touched.
    if progress["completed_signal"]:
        progress["running_signal"] = False
    elif progress["process_check_available"]:
        progress["running_signal"] = bool(progress["active_process_signal"])
    else:
        progress["running_signal"] = bool(
            progress["has_run_logs"]
            and progress["recent_log_activity"]
        )

    return progress


def build_start_checks(
    list_name: str,
    raw_voiceover_candidates: list[dict[str, Any]],
    lipsync_submitted: bool,
    lipsync_downloaded: bool,
) -> dict[str, Any]:
    """Build start-check gate payload with blocking gates and nonblocking statuses."""
    list_kind = list_bucket(list_name)
    is_editing = list_kind in {"editing", "pipeline_ready"}
    has_raw_voiceover = len(raw_voiceover_candidates) > 0

    blockers: list[str] = []
    nonblocking_statuses: list[str] = []
    if not is_editing:
        blockers.append("trello_card_not_in_editing")
    if not has_raw_voiceover:
        blockers.append("missing_raw_voiceover_attachment")
    if not lipsync_submitted:
        nonblocking_statuses.append("lipsync_not_submitted")
    if not lipsync_downloaded:
        nonblocking_statuses.append("lipsync_not_downloaded")

    ready_to_start = len(blockers) == 0

    return {
        "is_editing_list": is_editing,
        "has_raw_voiceover": has_raw_voiceover,
        "raw_voiceover_candidates": raw_voiceover_candidates,
        "lipsync_submitted": lipsync_submitted,
        "lipsync_downloaded": lipsync_downloaded,
        "ready_to_start": ready_to_start,
        "blockers": blockers,
        "nonblocking_statuses": nonblocking_statuses,
    }


def derive_pipeline_state(
    list_name: str,
    checks: dict[str, Any],
    local_progress: dict[str, Any] | None = None,
) -> str:
    """Derive pipeline-run state (separate from Trello workflow/submission state)."""
    local = local_progress or {}
    workflow_state = list_bucket(list_name)

    if local.get("completed_signal"):
        return "completed"

    if not is_pipeline_required(workflow_state):
        return "not_required"

    # Check readiness first - if all gates pass, it's ready regardless of prior runs
    if checks["ready_to_start"]:
        return "ready"

    # If not ready, determine if it's not_started (new project) or blocked (started but stuck)
    if local.get("has_project_dir") and not local.get("has_run_logs"):
        return "not_started"
    return "blocked"


def queue_sort_key(pipeline: dict[str, Any]) -> tuple[str, str, str]:
    """Sort by due date, last activity, card id."""
    trello_raw = pipeline.get("trello", {}).get("card_raw", {})
    due = str(trello_raw.get("due") or "9999-12-31T23:59:59Z")
    last_activity = str(trello_raw.get("dateLastActivity") or "9999-12-31T23:59:59Z")
    return (due, last_activity, str(pipeline.get("card_id")))


def load_previous_state(path: Path) -> dict[str, Any]:
    """Load existing state file if present."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def merge_history(
    previous: dict[str, Any] | None,
    new_state: str,
    new_blockers: list[str],
) -> list[dict[str, Any]]:
    """Preserve history and append transition events."""
    history: list[dict[str, Any]] = []
    if previous:
        history = list(previous.get("history", []))
    prev_state = previous.get("pipeline_state") if previous else None
    prev_blockers = previous.get("start_checks", {}).get("blockers", []) if previous else []

    if prev_state != new_state:
        history.append(
            {
                "ts": now_iso(),
                "event": "state_changed",
                "from": prev_state,
                "to": new_state,
            }
        )
    if sorted(prev_blockers) != sorted(new_blockers):
        history.append(
            {
                "ts": now_iso(),
                "event": "blockers_changed",
                "from": prev_blockers,
                "to": new_blockers,
            }
        )
    return history[-100:]


def collect_state(
    accounts: list[dict[str, str]],
    board_channel_map: dict[str, dict[str, str]],
    lipsync_index: dict[str, dict[str, Any]],
    previous_state: dict[str, Any],
    *,
    probe_live_lipsync: bool = True,
) -> dict[str, Any]:
    """Collect full state from Trello + local filesystem + lipsync tracking."""
    pipelines: dict[str, dict[str, Any]] = {}
    queue_buckets = {
        "pending_not_started_card_ids": [],
        "ready_card_ids": [],
        "blocked_card_ids": [],
        "completed_card_ids": [],
        "not_required_card_ids": [],
        "not_queued_card_ids": [],
    }
    running_card_ids: list[str] = []
    workflow_counts: Counter[str] = Counter()
    project_state_counts: Counter[str] = Counter()
    submission_queue_ids: list[str] = []
    excluded_external_assignee_cards: list[str] = []

    account_member_ids: dict[str, str] = {}
    for account in accounts:
        try:
            me = get_me(account)
        except Exception as exc:
            print_warn(f"Skipping account '{account['name']}' (cannot load profile): {exc}")
            continue
        me_id = str(me.get("id") or "").strip()
        if not me_id:
            print_warn(f"Skipping account '{account['name']}' (profile missing member id)")
            continue
        account_member_ids[account["name"]] = me_id

    allowed_member_ids = set(account_member_ids.values())

    for account in accounts:
        me_id = account_member_ids.get(account["name"])
        if not me_id:
            continue
        try:
            boards = get_boards(account)
        except Exception as exc:
            print_warn(f"Skipping account '{account['name']}' (cannot load boards): {exc}")
            continue

        account_board_id = str(account.get("board_id") or "").strip()
        for board in boards:
            if board.get("closed"):
                continue
            board_id = str(board.get("id"))
            if account_board_id and board_id != account_board_id:
                continue
            board_name = board.get("name", "")

            try:
                board_lists = get_board_lists(account, board_id)
                cards = get_board_cards(account, board_id)
            except Exception as exc:
                print_warn(f"Board '{board_name}' ({board_id}) skipped: {exc}")
                continue

            list_id_to_name = {lst.get("id"): lst.get("name", "") for lst in board_lists}

            for card in cards:
                if card.get("closed"):
                    continue
                member_ids = normalize_member_ids(card.get("idMembers") or [])
                # Only include cards where at least one member has a local account file.
                # This filters out cards assigned to people without accounts (e.g. Hamza, Liam).
                if member_ids and not (set(member_ids) & allowed_member_ids):
                    short_card_id = str(card.get("shortLink") or card.get("id") or "")
                    if short_card_id:
                        excluded_external_assignee_cards.append(short_card_id)
                    continue

                full_card_id = str(card.get("id"))
                short_card_id = str(card.get("shortLink") or "")
                if not short_card_id:
                    continue
                key = short_card_id.lower()

                # Preserve first-seen copy and merge account visibility if duplicated.
                if key in pipelines:
                    seen = pipelines[key].setdefault("seen_by_accounts", [])
                    if account["display_name"] not in seen:
                        seen.append(account["display_name"])
                    continue

                try:
                    card_detail = get_card_detail(account, full_card_id)
                except Exception as exc:
                    print_warn(f"Card '{short_card_id}' skipped (detail fetch failed): {exc}")
                    continue

                routing = resolve_channels_for_board(
                    board_channel_map.get(board_id, {}),
                    str(account.get("default_channel") or "RRU"),
                    (card_detail.get("labels") if isinstance(card_detail, dict) else None)
                    or (card.get("labels") if isinstance(card, dict) else None)
                    or [],
                )
                project_channel = routing["project_channel"]
                lipsync_channel = routing["lipsync_channel"]

                list_name = list_id_to_name.get(card_detail.get("idList"), "")
                workflow_state = list_bucket(list_name)
                project_state = derive_project_state(workflow_state)
                raw_voiceovers = extract_card_raw_voiceover_candidates(
                    card_detail,
                    workflow_state=workflow_state,
                    gws_context=build_gws_drive_context(account),
                )

                tracked = lipsync_index.get(key, {})
                tracked_lipsync = tracked.get("lipsync", {}) if isinstance(tracked, dict) else {}
                local_scan = find_local_lipsync_download(
                    project_channel,
                    short_card_id,
                    full_card_id,
                    card_title=str(card_detail.get("name", card.get("name", ""))),
                )

                effective_lipsync, lipsync_submitted, lipsync_downloaded = resolve_lipsync_submission_status(
                    tracked_lipsync,
                    channel=lipsync_channel,
                    card_id=short_card_id,
                    card_title=str(card_detail.get("name", card.get("name", ""))),
                    local_scan=local_scan,
                    probe_live_drive=probe_live_lipsync,
                )

                checks = build_start_checks(
                    list_name=list_name,
                    raw_voiceover_candidates=raw_voiceovers,
                    lipsync_submitted=lipsync_submitted,
                    lipsync_downloaded=lipsync_downloaded,
                )
                project_dirs = local_scan.get("project_dirs", [])
                local_progress = detect_local_pipeline_progress(project_dirs)
                pipeline_state = derive_pipeline_state(list_name, checks, local_progress)
                previous_pipeline = (previous_state.get("pipelines") or {}).get(key, {})
                history = merge_history(previous_pipeline, pipeline_state, checks["blockers"])
                needs_submission_workflow = bool(
                    local_progress.get("completed_signal")
                    and workflow_state in COMPLIANCE_WORKFLOW_STATES
                )
                if needs_submission_workflow:
                    submission_queue_ids.append(short_card_id)

                pipeline_entry = {
                    "card_id": short_card_id,
                    "title": card_detail.get("name", card.get("name", "")),
                    "pipeline_state": pipeline_state,
                    "pipeline_run_state": pipeline_state,
                    "pipeline_runtime_state": "running" if local_progress.get("running_signal") else "idle",
                    "workflow_state": workflow_state,
                    "project_state": project_state,
                    "needs_submission_workflow": needs_submission_workflow,
                    "seen_by_accounts": [account["display_name"]],
                    "project": {
                        "channel": project_channel,
                        "lipsync_channel": lipsync_channel,
                        "local_project_dirs": project_dirs,
                        "local_progress": local_progress,
                    },
                    "start_checks": checks,
                    "lipsync": {
                        "tracking_snapshot": effective_lipsync,
                        "local_scan": local_scan,
                    },
                    "trello": {
                        "account_used": account["display_name"],
                        "board": board,
                        "list": {
                            "id": card_detail.get("idList"),
                            "name": list_name,
                            "bucket": workflow_state,
                        },
                        "card_summary_raw": card,
                        "card_raw": card_detail,
                    },
                    "history": history,
                    "updated_at": now_iso(),
                }
                pipelines[key] = pipeline_entry
                workflow_counts[workflow_state] += 1
                project_state_counts[project_state] += 1

    sorted_pipelines = sorted(pipelines.values(), key=queue_sort_key)
    for pipeline in sorted_pipelines:
        state = pipeline["pipeline_state"]
        card_id = pipeline["card_id"]
        local_progress = ((pipeline.get("project") or {}).get("local_progress")) or {}
        if local_progress.get("running_signal"):
            running_card_ids.append(card_id)
        if state == "not_started":
            queue_buckets["pending_not_started_card_ids"].append(card_id)
        elif state == "ready":
            queue_buckets["ready_card_ids"].append(card_id)
        elif state == "blocked":
            queue_buckets["blocked_card_ids"].append(card_id)
        elif state == "completed":
            queue_buckets["completed_card_ids"].append(card_id)
        elif state == "not_required":
            queue_buckets["not_required_card_ids"].append(card_id)
        else:
            queue_buckets["not_queued_card_ids"].append(card_id)

    submission_cards = [
        build_submission_card_payload(card_id, get_pipeline(pipelines, card_id))
        for card_id in submission_queue_ids
    ]

    summary = {
        "total_pipelines": len(pipelines),
        "pending_not_started": len(queue_buckets["pending_not_started_card_ids"]),
        "ready": len(queue_buckets["ready_card_ids"]),
        "blocked": len(queue_buckets["blocked_card_ids"]),
        "completed": len(queue_buckets["completed_card_ids"]),
        "not_required": len(queue_buckets["not_required_card_ids"]),
        "not_queued": len(queue_buckets["not_queued_card_ids"]),
        "running": len(running_card_ids),
        "pipeline_actionable": (
            len(queue_buckets["pending_not_started_card_ids"])
            + len(queue_buckets["ready_card_ids"])
            + len(queue_buckets["blocked_card_ids"])
        ),
        "needs_submission_workflow": len(submission_queue_ids),
        "excluded_external_assignee_cards": len(set(excluded_external_assignee_cards)),
    }

    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "queue_policy": {
            "queued_states": ["not_started", "ready", "blocked"],
            "pipeline_required_workflow_states": ["pending", "not_started", "editing", "review"],
            "submission_workflow_states": sorted(COMPLIANCE_WORKFLOW_STATES),
            "start_requirements": [
                "raw_voiceover_attachment_present_and_not_silence_removed",
                "trello_card_in_editing_list",
            ],
            "nonblocking_status_requirements": [
                "lipsync_submitted",
                "lipsync_downloaded_to_local_project_dir",
            ],
        },
        "summary": summary,
        "workflow_summary": dict(workflow_counts),
        "project_state_summary": dict(project_state_counts),
        "queue": queue_buckets,
        "runtime_summary": {
            "is_any_pipeline_running": bool(running_card_ids),
            "running_count": len(running_card_ids),
            "running_card_ids": running_card_ids,
            "running_window_seconds": RUNNING_LOG_ACTIVITY_WINDOW_SECONDS,
        },
        "submission_queue": {
            "needs_submission_card_ids": submission_queue_ids,
            "needs_submission_cards": submission_cards,
        },
        "filters": {
            "allowed_member_ids": sorted(allowed_member_ids),
            "excluded_external_assignee_card_ids": sorted(
                {str(card_id) for card_id in excluded_external_assignee_cards}
            ),
        },
        "pipelines": pipelines,
    }


def write_state(path: Path, payload: dict[str, Any]) -> None:
    """Write state payload to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    fallback_text: str | None = None
    try:
        with open(temp_path, "w", encoding="utf-8", errors="replace") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        fallback_text = temp_path.read_text(encoding="utf-8", errors="replace")
        for attempt_index, delay_seconds in enumerate((0.0, *STATE_WRITE_RETRY_DELAYS_SECONDS), start=1):
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            try:
                os.replace(temp_path, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {5, 32}:
                    raise
                if attempt_index >= len(STATE_WRITE_RETRY_DELAYS_SECONDS) + 1:
                    with open(path, "w", encoding="utf-8", errors="replace") as handle:
                        handle.write(fallback_text or "")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def print_sync_summary(state_file: Path, state: dict[str, Any]) -> None:
    """Print human-readable sync summary."""
    summary = state["summary"]
    runtime_summary = state.get("runtime_summary", {}) or {}
    print_header("PIPELINE QUEUE STATE")
    print_ok(f"State file: {state_file}")
    print_ok(f"Total pipelines tracked: {summary['total_pipelines']}")
    print_ok(
        "Pipeline run queue: "
        f"not_started={summary['pending_not_started']}, ready={summary['ready']}, blocked={summary['blocked']}"
    )
    print_ok(
        "Pipeline run outcomes: "
        f"completed={summary['completed']}, not_required={summary.get('not_required', 0)}, other={summary['not_queued']}"
    )
    print_ok(
        "Excluded by assignee filter: "
        f"{summary.get('excluded_external_assignee_cards', 0)}"
    )
    print_ok(f"Actionable pipeline runs: {summary.get('pipeline_actionable', 0)}")
    print_ok(
        "Pipeline currently running: "
        f"{'yes' if runtime_summary.get('is_any_pipeline_running') else 'no'} "
        f"(count={runtime_summary.get('running_count', summary.get('running', 0))})"
    )
    print_ok(f"Cards needing submission/review workflow: {summary.get('needs_submission_workflow', 0)}")


def load_state_or_error(path: Path) -> dict[str, Any]:
    """Load state JSON or fail with helpful error."""
    if not path.exists():
        print_error(
            f"State file not found: {path}. Run sync first: python scripts/pipeline_queue_state.py sync",
            exit_code=1,
        )
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def sync_state(state_file: Path, refresh_lipsync: bool) -> dict[str, Any]:
    """Run full sync and persist state."""
    if refresh_lipsync:
        print_info("Refreshing lipsync tracking before queue sync")
        run_lipsync_refresh()

    accounts = load_accounts(DEFAULT_ACCOUNTS_DIR)
    if not accounts:
        print_error(f"No valid accounts found in {DEFAULT_ACCOUNTS_DIR}", exit_code=1)

    board_channel_map = load_board_channel_map(DEFAULT_BOARD_MAP_FILE)
    lipsync_index = load_lipsync_tracking(DEFAULT_LIPSYNC_TRACKING_FILE)
    previous = load_previous_state(state_file)

    # Regular queue sync should stay fast and rely on cached lipsync tracking.
    # Live Drive probing is reserved for explicit refresh flows.
    state = collect_state(
        accounts,
        board_channel_map,
        lipsync_index,
        previous,
        probe_live_lipsync=bool(refresh_lipsync),
    )
    write_state(state_file, state)
    return state


def sync_state_for_lipsync_next(
    state_file: Path,
    *,
    refresh_lipsync: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str, str | None]:
    """Best-effort sync for lipsync-next with cached-state fallback on slow refreshes."""
    refreshed_source = (
        "queue_with_live_lipsync_refresh"
        if refresh_lipsync
        else "queue_state_refreshed"
    )
    timeout_value = float(timeout_seconds or 0)
    if timeout_value <= 0:
        return sync_state(state_file, refresh_lipsync), refreshed_source, None

    cmd = [
        sys.executable,
        _script_path,
        "--state-file",
        str(state_file),
        "sync",
    ]
    if refresh_lipsync:
        cmd.append("--refresh-lipsync")

    try:
        result = run_subprocess(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_value,
        )
    except subprocess.TimeoutExpired:
        if not state_file.exists():
            print_error(
                f"Queue sync exceeded {timeout_value:g}s and no cached state file exists at {state_file}",
                exit_code=1,
            )
        warning = (
            f"Queue sync exceeded {timeout_value:g}s; using cached state file instead."
        )
        return load_state_or_error(state_file), "queue_state_file_after_sync_timeout", warning

    if result.returncode != 0:
        details = (result.stderr or "").strip() or (result.stdout or "").strip() or f"exit code {result.returncode}"
        if not state_file.exists():
            print_error(f"Queue sync failed and no cached state file exists: {details}", exit_code=1)
        warning = f"Queue sync failed ({details}); using cached state file instead."
        return load_state_or_error(state_file), "queue_state_file_after_sync_error", warning

    return load_state_or_error(state_file), refreshed_source, None


def command_sync(args: argparse.Namespace) -> int:
    """Handle sync subcommand."""
    state = sync_state(Path(args.state_file), args.refresh_lipsync)
    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0
    print_sync_summary(Path(args.state_file), state)
    return 0


def command_archive_completed(args: argparse.Namespace) -> int:
    """Handle archive-completed subcommand."""
    state_file = Path(args.state_file)
    if args.sync_first:
        sync_state(state_file, args.refresh_lipsync)

    state = load_state_or_error(state_file)
    result = archive_completed_project_dirs(state, dry_run=args.dry_run)
    summary = Counter(result.get("summary") or {})
    moved_count = int(summary.get("moved", 0))
    error_count = int(summary.get("error", 0))

    if moved_count > 0 and not args.dry_run and not args.no_sync_after:
        print_info("Refreshing queue state after archiving completed projects")
        sync_state(state_file, refresh_lipsync=False)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if error_count == 0 else 1

    print_header("ARCHIVE COMPLETED PROJECTS")
    print_ok(f"Archive root: {result.get('archive_root')}")
    print_ok(f"Moved: {moved_count}")
    if args.dry_run:
        print_ok(f"Would move: {int(summary.get('would_move', 0))}")
    if summary.get("already_archived"):
        print_ok(f"Already archived: {int(summary.get('already_archived', 0))}")
    if summary.get("missing"):
        print_warn(f"Missing source dirs: {int(summary.get('missing', 0))}")
    if error_count:
        print_warn(f"Move errors: {error_count}")

    material_actions = [
        action
        for action in result.get("actions", [])
        if str(action.get("status") or "") in {"moved", "would_move", "error"}
    ]
    if not material_actions:
        print_ok("No local project directories met the archive threshold")
        return 0 if error_count == 0 else 1

    for action in material_actions:
        status = str(action.get("status") or "unknown").upper()
        print(
            f"  - [{status}] {action.get('card_id')} "
            f"{truncate_text(action.get('title') or 'Untitled card', 90)}"
        )
        print(f"      from: {action.get('source')}")
        destination = action.get("destination")
        if destination:
            print(f"      to:   {destination}")
        if action.get("error"):
            print(f"      error: {action.get('error')}")

    return 0 if error_count == 0 else 1


def command_next(args: argparse.Namespace) -> int:
    """Handle next subcommand."""
    state_file = Path(args.state_file)
    if args.sync_first:
        sync_state(state_file, args.refresh_lipsync)

    state = load_state_or_error(state_file)
    queue = state.get("queue", {})
    pipelines = state.get("pipelines", {})
    running_ids = resolve_running_card_ids(state, pipelines)
    running_id_set = {card_id.lower() for card_id in running_ids}
    ready_ids = [
        card_id for card_id in (queue.get("ready_card_ids", []) or [])
        if str(card_id).lower() not in running_id_set
    ]
    requested_channels = normalize_channels(args.channel or [])
    if requested_channels:
        ready_ids = [
            card_id for card_id in ready_ids
            if get_pipeline_channel(pipelines, card_id) in requested_channels
        ]

    if not ready_ids:
        blocked_ids = queue.get("blocked_card_ids", [])
        not_started_ids = queue.get("pending_not_started_card_ids", [])
        completed_ids = queue.get("completed_card_ids", [])
        not_required_ids = queue.get("not_required_card_ids", [])
        submission_ids = (
            (state.get("submission_queue", {}) or {}).get("needs_submission_card_ids", []) or []
        )
        if args.json:
            payload = {
                "next_pipeline": None,
                "reason": "no_ready_pipeline",
                "blocked_preview": [
                    {
                        "card_id": card_id,
                        "title": (pipelines.get(card_id.lower(), {}) or {}).get("title"),
                        "blockers": (
                            (pipelines.get(card_id.lower(), {}) or {})
                            .get("start_checks", {})
                            .get("blockers", [])
                        ),
                    }
                    for card_id in blocked_ids[:5]
                ],
                "not_started_preview": [
                    {
                        "card_id": card_id,
                        "title": (pipelines.get(card_id.lower(), {}) or {}).get("title"),
                    }
                    for card_id in not_started_ids[:5]
                ],
                "submission_preview": [
                    build_submission_card_payload(card_id, pipelines.get(card_id.lower(), {}) or {})
                    for card_id in submission_ids[:5]
                ],
                "runtime_summary": {
                    "is_any_pipeline_running": bool(running_ids),
                    "running_count": len(running_ids),
                    "running_card_ids": running_ids[:10],
                    "running_window_seconds": RUNNING_LOG_ACTIVITY_WINDOW_SECONDS,
                },
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        print_warn("No pipeline is ready to start")
        print_ok(
            "Pipeline run summary: "
            f"not_started={len(not_started_ids)}, ready=0, blocked={len(blocked_ids)}, "
            f"completed={len(completed_ids)}, not_required={len(not_required_ids)}"
        )
        if running_ids:
            preview = ", ".join(running_ids[:5])
            suffix = " ..." if len(running_ids) > 5 else ""
            print_warn(f"Pipeline currently running: {preview}{suffix}")
        if not_started_ids:
            print_header("NOT-STARTED PIPELINE PREVIEW")
            for card_id in not_started_ids[:5]:
                entry = pipelines.get(card_id.lower(), {})
                print(f"  - {card_id}: {truncate_text(entry.get('title') or 'Untitled card')}")
        if blocked_ids:
            print_header("BLOCKED PIPELINE PREVIEW")
            for card_id in blocked_ids[:5]:
                entry = pipelines.get(card_id.lower(), {})
                blockers = entry.get("start_checks", {}).get("blockers", [])
                print(f"  - {card_id}: {', '.join(blockers) if blockers else 'unknown blockers'}")
        if submission_ids:
            print_header("CARDS NEEDING SUBMISSION OR REVIEW")
            for card_id in submission_ids[:8]:
                entry = pipelines.get(card_id.lower(), {})
                workflow_state = entry.get("workflow_state") or "other"
                print(
                    f"  - {card_id}: {truncate_text(entry.get('title') or 'Untitled card')} "
                    f"(workflow={workflow_state})"
                )
        return 0

    next_card_id = ready_ids[0]
    next_pipeline = pipelines.get(next_card_id.lower())
    if not next_pipeline:
        print_error(f"Ready pipeline '{next_card_id}' missing in pipeline map", exit_code=1)
    if args.json:
        print(json.dumps(next_pipeline, indent=2, ensure_ascii=False))
        return 0

    print_header("NEXT READY PIPELINE")
    if running_ids:
        preview = ", ".join(running_ids[:5])
        suffix = " ..." if len(running_ids) > 5 else ""
        print_warn(f"Pipeline currently running: {preview}{suffix}")
    project = next_pipeline.get("project", {}) if isinstance(next_pipeline.get("project"), dict) else {}
    print_ok(f"Card ID: {next_pipeline['card_id']}")
    print_ok(f"Title: {next_pipeline['title']}")
    print_ok(f"Channel: {project.get('channel')}")
    if project.get("lipsync_channel") and project.get("lipsync_channel") != project.get("channel"):
        print_ok(f"Lipsync Channel: {project.get('lipsync_channel')}")
    print_ok(f"Trello list: {next_pipeline.get('trello', {}).get('list', {}).get('name')}")
    print_ok(f"Local dirs: {next_pipeline.get('project', {}).get('local_project_dirs', [])}")
    return 0


def command_lipsync_next(args: argparse.Namespace) -> int:
    """Handle lipsync-next subcommand."""
    state_file = Path(args.state_file)
    source = "queue_state_file"
    sync_warning: str | None = None
    if args.sync_first:
        sync_timeout_seconds = float(getattr(args, "sync_timeout_seconds", 0) or 0)
        state, source, sync_warning = sync_state_for_lipsync_next(
            state_file,
            refresh_lipsync=args.refresh_lipsync,
            timeout_seconds=sync_timeout_seconds,
        )
    else:
        state = load_state_or_error(state_file)
    candidate = select_next_lipsync_submission_candidate(state)
    payload = {
        "candidate": candidate,
        "candidate_count": len(collect_lipsync_submission_candidates(state)),
        "source": source,
    }
    if sync_warning:
        payload["sync_warning"] = sync_warning

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if sync_warning:
        print_warn(sync_warning)
    if not candidate:
        print_ok("No queue-backed project currently needs a new lipsync generation.")
        print_info("Drive/local lipsync status was checked before making this determination.")
        return 0

    print_header("NEXT LIPSYNC SUBMISSION CANDIDATE")
    print_ok(f"Card ID: {candidate['card_id']}")
    print_ok(f"Title: {candidate['title']}")
    print_ok(f"Channel: {candidate['channel']}")
    if candidate.get("lipsync_channel") and candidate.get("lipsync_channel") != candidate.get("channel"):
        print_ok(f"Lipsync Channel: {candidate['lipsync_channel']}")
    print_ok(f"Account: {candidate['account']}")
    print_ok(f"Workflow: {candidate['workflow_state']}")
    print_ok(f"Pipeline: {candidate['pipeline_state']}")
    print_ok(f"Trello list: {candidate['trello_list']}")
    print_ok(f"Card URL: {candidate['card_url']}")
    print_ok(f"Drive status: {candidate['lipsync_tracking_drive_status']}")
    print_ok(f"Local video status: {candidate['lipsync_local_video_status']}")
    return 0


def resolve_card_url(entry: dict[str, Any], card_id: str) -> str:
    """Resolve the best Trello URL for a card."""
    trello = entry.get("trello", {}) if isinstance(entry, dict) else {}
    card_raw = trello.get("card_raw", {}) if isinstance(trello, dict) else {}
    card_summary = trello.get("card_summary_raw", {}) if isinstance(trello, dict) else {}

    for source in (card_raw, card_summary):
        if not isinstance(source, dict):
            continue
        for key in ("shortUrl", "url"):
            value = str(source.get(key, "")).strip()
            if value:
                return value

    return f"https://trello.com/c/{card_id}"


def resolve_card_accounts(entry: dict[str, Any]) -> list[str]:
    """Resolve account ownership/visibility labels for a pipeline card."""
    accounts: list[str] = []
    seen = entry.get("seen_by_accounts", []) if isinstance(entry, dict) else []
    if isinstance(seen, list):
        for value in seen:
            label = str(value or "").strip()
            if label and label not in accounts:
                accounts.append(label)

    if not accounts:
        trello = entry.get("trello", {}) if isinstance(entry, dict) else {}
        account_used = str((trello.get("account_used") if isinstance(trello, dict) else "") or "").strip()
        if account_used:
            accounts.append(account_used)

    return accounts


def submission_next_action(workflow_state: str) -> str:
    """Resolve a human-readable next action for submission/review workflow."""
    return SUBMISSION_WORKFLOW_HINTS.get(
        workflow_state,
        "Advance Trello workflow and complete submission deliverables",
    )


def build_submission_card_payload(card_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized submission/review payload row for reporting."""
    if not isinstance(entry, dict):
        entry = {}

    trello = entry.get("trello", {}) if isinstance(entry.get("trello"), dict) else {}
    list_payload = trello.get("list", {}) if isinstance(trello.get("list"), dict) else {}
    workflow_state = str(entry.get("workflow_state") or list_payload.get("bucket") or "other")
    list_name = str(list_payload.get("name") or "Unknown")
    accounts = resolve_card_accounts(entry)
    account_label = ", ".join(accounts) if accounts else "Unknown"

    return {
        "card_id": card_id,
        "title": str(entry.get("title") or "Untitled card"),
        "trello_list": list_name,
        "workflow_state": workflow_state,
        "next_action": submission_next_action(workflow_state),
        "account": account_label,
        "accounts": accounts,
        "channel": str((entry.get("project") or {}).get("channel") or "Unknown"),
        "card_url": resolve_card_url(entry, card_id),
    }


def collect_unprepared_targets(
    state: dict[str, Any],
    requested_card_ids: list[str] | None = None,
    requested_channels: Sequence[str] | None = None,
    limit: int = 0,
) -> list[dict[str, str]]:
    """Collect cards that need local project setup."""
    pipelines = state.get("pipelines", {}) or {}

    if requested_card_ids:
        ordered_ids = [str(card_id).strip() for card_id in requested_card_ids if str(card_id).strip()]
    else:
        ordered_ids = []
        for entry in sorted((pipelines or {}).values(), key=queue_sort_key):
            if not isinstance(entry, dict):
                continue
            workflow_state = str(entry.get("workflow_state") or "")
            pipeline_state = str(entry.get("pipeline_state") or entry.get("pipeline_run_state") or "")
            if workflow_state != "script_vo_description" and pipeline_state not in {"not_started", "ready", "blocked"}:
                continue
            card_id = str(entry.get("card_id") or "").strip()
            if card_id:
                ordered_ids.append(card_id)

    if requested_channels:
        channel_set = normalize_channels(requested_channels)
        ordered_ids = [
            card_id for card_id in ordered_ids
            if get_pipeline_channel(pipelines, card_id) in channel_set
        ]

    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for card_id in ordered_ids:
        key = card_id.lower()
        if key in seen:
            continue
        seen.add(key)

        entry = get_pipeline(pipelines, card_id)
        if not entry:
            continue

        local_dirs = ((entry.get("project") or {}).get("local_project_dirs", []) or [])

        # Check if project exists AND has any voiceover file (named or generic)
        needs_setup = True
        for local_dir in local_dirs:
            vo_dir = Path(local_dir) / "voiceover"
            if vo_dir.is_dir():
                for vo_file in vo_dir.iterdir():
                    if vo_file.is_file() and vo_file.suffix.lower() in LIKELY_AUDIO_EXTENSIONS:
                        needs_setup = False
                        break
            if not needs_setup:
                break

        if not needs_setup:
            continue

        title = str(entry.get("title") or "Untitled card")
        card_url = resolve_card_url(entry, card_id)

        # Prefer the VO Google Doc URL (contains Drive links to audio parts)
        vo_doc_url = ""
        candidates = (entry.get("start_checks") or {}).get("raw_voiceover_candidates", []) or []
        # Look for a candidate whose name/label hints at VO (not the script doc)
        for cand in reversed(candidates):
            if cand.get("is_google_doc"):
                raw_url = str(cand.get("url") or "")
                # Fix doubled Trello markdown URLs: url](url
                if "](http" in raw_url:
                    raw_url = raw_url.split("](")[0]
                vo_doc_url = raw_url
                break  # reversed: last doc is typically the VO

        targets.append(
            {
                "card_id": card_id,
                "card_url": card_url,
                "title": title,
                "channel": str((entry.get("project") or {}).get("channel") or ""),
                "account": str(((entry.get("trello") or {}).get("account_used") or "")),
                "vo_doc_url": vo_doc_url,
            }
        )
        if limit > 0 and len(targets) >= limit:
            break

    return targets


def run_newproject_for_card(
    project_name: str,
    channel: str,
    card_url: str,
    account_name: str = "",
    card_id: str = "",
    no_pipeline: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Invoke newproject flow for a Trello card URL."""
    cmd = [sys.executable, "-m", "src.cli.newproject", project_name, channel, card_url]
    account_value = str(account_name or "").strip()
    if account_value:
        cmd.extend(["--account", account_value])
    # Pass card ID for naming the project folder
    card_id_value = str(card_id or "").strip()
    if card_id_value:
        cmd.extend(["--card-id", card_id_value])
    if no_pipeline:
        cmd.append("--no-pipeline")
    env = dict(os.environ)
    if account_value:
        env["MATCHER_ACCOUNT_ENV"] = account_value
    return run_subprocess(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )


def run_newproject_for_discord(
    video_title: str,
    channel: str,
    drive_folder_url: str,
    account_name: str = "",
    card_id: str = "",
) -> subprocess.CompletedProcess[str]:
    """Invoke newproject flow for a Discord Pipeline Complete post."""
    cmd = [sys.executable, "-m", "src.cli.newproject"]
    account_value = str(account_name or "").strip()
    if account_value:
        cmd.extend(["--account", account_value])
    card_value = str(card_id or "").strip()
    if card_value:
        cmd.extend(["--card-id", card_value])
    cmd.extend([video_title, channel, drive_folder_url])
    env = dict(os.environ)
    if account_value:
        env["MATCHER_ACCOUNT_ENV"] = account_value
    return run_subprocess(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )


def parse_command_prefix(raw: str) -> list[str]:
    """Parse a command string into argv tokens."""
    value = str(raw or "").strip()
    if not value:
        return []
    if os.path.exists(value):
        return [value]
    try:
        import shlex

        return shlex.split(value, posix=False)
    except ValueError:
        return [value]


def resolve_discord_exporter_prefixes(preferred: str = "") -> list[list[str]]:
    """Resolve possible DiscordChatExporter command prefixes."""
    prefixes: list[list[str]] = []
    if preferred:
        parsed = parse_command_prefix(preferred)
        if parsed:
            prefixes.append(parsed)

    bundled = PROJECT_ROOT / "tools" / "DiscordChatExporter.Cli" / "DiscordChatExporter.Cli.exe"
    if bundled.exists():
        prefixes.append([str(bundled)])

    # Support globally installed dotnet tools even if PATH isn't configured.
    user_dotnet_tools_dir = Path.home() / ".dotnet" / "tools"
    for global_tool in (
        user_dotnet_tools_dir / "DiscordChatExporter.Cli.exe",
        user_dotnet_tools_dir / "DiscordChatExporter.Cli",
        user_dotnet_tools_dir / "discordchatexporter.cli.exe",
        user_dotnet_tools_dir / "discordchatexporter.cli",
    ):
        if global_tool.exists():
            prefixes.append([str(global_tool)])

    for command_name in (
        "DiscordChatExporter.Cli",
        "DiscordChatExporter.Cli.exe",
        "discordchatexporter.cli",
        "discordchatexporter",
    ):
        resolved = which(command_name)
        if resolved:
            prefixes.append([resolved])

    # `dotnet tool run` requires a local tool manifest in the repository.
    tool_manifest = PROJECT_ROOT / ".config" / "dotnet-tools.json"
    if which("dotnet") and tool_manifest.exists():
        prefixes.append(["dotnet", "tool", "run", "DiscordChatExporter.Cli"])

    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for prefix in prefixes:
        key = tuple(part.lower() for part in prefix if part)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(prefix)
    return unique


def build_discord_export_commands(
    prefix: list[str],
    token: str,
    channel_id: str,
    output_path: Path,
    *,
    after: str = "",
) -> list[list[str]]:
    """Build command variants for multiple DiscordChatExporter CLI versions."""
    base_commands = [
        # Current CLI (v2.47+): requires explicit --channel/-c option.
        prefix + [
            "export",
            "--channel",
            str(channel_id),
            "--token",
            token,
            "--output",
            str(output_path),
            "--format",
            "Json",
        ],
        prefix + [
            "export",
            "-c",
            str(channel_id),
            "-t",
            token,
            "-o",
            str(output_path),
            "-f",
            "Json",
        ],
        # Older variants retained for compatibility.
        prefix + [
            "export",
            str(channel_id),
            "--token",
            token,
            "--output",
            str(output_path),
            "--format",
            "Json",
        ],
        prefix + [
            "export",
            str(channel_id),
            "-t",
            token,
            "-o",
            str(output_path),
            "-f",
            "Json",
        ],
    ]
    legacy_exportchannel = prefix + [
        "exportchannel",
        "-t",
        token,
        "-c",
        str(channel_id),
        "-f",
        "Json",
        "-o",
        str(output_path),
    ]

    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add_command(parts: list[str]) -> None:
        key = tuple(parts)
        if key in seen:
            return
        seen.add(key)
        commands.append(parts)

    for base in base_commands:
        if after:
            add_command(base + ["--after", after])
        # Keep fallback without --after for exporter versions that don't support it.
        add_command(base)
    add_command(legacy_exportchannel)
    return commands


def extract_discord_export_failure_reason(result: subprocess.CompletedProcess[str]) -> str:
    """Extract concise failure reason from command output."""
    stderr_tail = "\n".join((result.stderr or "").splitlines()[-3:]).strip()
    stdout_tail = "\n".join((result.stdout or "").splitlines()[-3:]).strip()
    return stderr_tail or stdout_tail or f"exit={result.returncode}"


def is_non_retryable_discord_export_error(text: str) -> bool:
    """Return True when an export failure should not be retried."""
    value = normalize_lower(text)
    if not value:
        return False
    return any(hint in value for hint in DISCORD_NON_RETRYABLE_ERROR_HINTS)


def is_retryable_discord_export_error(text: str) -> bool:
    """Return True for rate-limit/transient failures worth retrying."""
    value = normalize_lower(text)
    if not value:
        return False
    if is_non_retryable_discord_export_error(value):
        return False
    if any(hint in value for hint in DISCORD_RATE_LIMIT_HINTS):
        return True
    if any(hint in value for hint in DISCORD_TRANSIENT_ERROR_HINTS):
        return True
    # Catch plain HTTP status outputs, e.g. "HTTP 502".
    return bool(re.search(r"\b5\d\d\b", value))


def compute_discord_backoff_seconds(
    attempt_number: int,
    *,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
) -> float:
    """Exponential backoff with jitter for Discord export retries."""
    safe_base = max(float(base_backoff_seconds), 0.1)
    safe_cap = max(float(max_backoff_seconds), safe_base)
    exp_delay = min(safe_base * (2 ** max(0, int(attempt_number) - 1)), safe_cap)
    jitter = random.uniform(0.0, min(safe_base, safe_cap) * 0.35)
    return min(exp_delay + jitter, safe_cap)


def run_discord_export(
    token: str,
    channel_id: str,
    output_path: Path,
    exporter: str = "",
    *,
    after: str = "",
    max_attempts: int = DEFAULT_DISCORD_EXPORT_MAX_ATTEMPTS,
    base_backoff_seconds: float = DEFAULT_DISCORD_EXPORT_BASE_BACKOFF_SECONDS,
    max_backoff_seconds: float = DEFAULT_DISCORD_EXPORT_MAX_BACKOFF_SECONDS,
    timeout_seconds: int = DEFAULT_DISCORD_EXPORT_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """
    Export one Discord channel to JSON.

    Returns:
        (success, message) where message is strategy info (success) or error summary (failure).
    """
    prefixes = resolve_discord_exporter_prefixes(exporter)
    if not prefixes:
        bundled_dir = PROJECT_ROOT / "tools" / "DiscordChatExporter.Cli"
        return False, (
            "DiscordChatExporter not found. Download the standalone release from "
            "https://github.com/Tyrrrz/DiscordChatExporter/releases and place "
            f"`DiscordChatExporter.Cli.exe` under `{bundled_dir}` "
            "(or set DISCORD_CHAT_EXPORTER / --exporter)."
        )

    retry_limit = max(1, int(max_attempts))
    timeout_limit = max(30, int(timeout_seconds))
    errors: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for prefix in prefixes:
        for command in build_discord_export_commands(prefix, token, channel_id, output_path, after=after):
            strategy = " ".join(prefix + [command[len(prefix)]]) if len(command) > len(prefix) else " ".join(prefix)
            for attempt in range(1, retry_limit + 1):
                if output_path.exists():
                    output_path.unlink()
                try:
                    result = run_subprocess(
                        command,
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                        timeout=timeout_limit,
                    )
                except subprocess.TimeoutExpired:
                    reason = f"timeout after {timeout_limit}s"
                    should_retry = attempt < retry_limit
                    if should_retry:
                        delay = compute_discord_backoff_seconds(
                            attempt,
                            base_backoff_seconds=base_backoff_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                        )
                        print_warn(
                            f"Discord export timeout for channel={channel_id} (attempt {attempt}/{retry_limit}). "
                            f"Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    errors.append(f"{strategy}: {reason}")
                    break
                except OSError as exc:
                    errors.append(f"{strategy}: {exc}")
                    break

                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    return True, strategy

                reason = extract_discord_export_failure_reason(result)
                failure_text = "\n".join(
                    part for part in (result.stderr or "", result.stdout or "", reason) if part
                )
                if attempt < retry_limit and is_retryable_discord_export_error(failure_text):
                    delay = compute_discord_backoff_seconds(
                        attempt,
                        base_backoff_seconds=base_backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                    print_warn(
                        f"Discord export retry for channel={channel_id} (attempt {attempt}/{retry_limit}): "
                        f"{truncate_text(reason, 140)}; sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                errors.append(f"{strategy}: {reason}")
                break

    return False, "; ".join(errors[-5:]) if errors else "unknown export error"


def load_discord_export_messages(path: Path) -> list[dict[str, Any]]:
    """Load message list from DiscordChatExporter JSON output."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        raw = payload.get("messages")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def collect_discord_message_text(message: dict[str, Any]) -> str:
    """Collect searchable text from a Discord export message payload."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content)

    embeds = message.get("embeds")
    if isinstance(embeds, list):
        for embed in embeds:
            if not isinstance(embed, dict):
                continue
            for key in ("title", "description"):
                value = embed.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value)
            fields = embed.get("fields")
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    for key in ("name", "value"):
                        value = field.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value)

    return "\n".join(parts)


def parse_discord_pipeline_message(
    message: dict[str, Any],
    *,
    account_name: str,
    channel_code: str,
    discord_channel_id: str,
) -> dict[str, Any] | None:
    """Parse one Discord message into a project candidate if it matches the pipeline-complete pattern."""
    text = collect_discord_message_text(message)
    if not text:
        return None

    # Normalize markdown labels from Discord exports (e.g., "**Video:**", "**Files:**").
    normalized_text = re.sub(r"\*\*", "", text)
    normalized_text = re.sub(r"__+", "", normalized_text)

    if not DISCORD_PIPELINE_HEADER_RE.search(normalized_text):
        return None

    video_match = DISCORD_VIDEO_LINE_RE.search(normalized_text)
    files_match = DISCORD_FILES_LINE_RE.search(normalized_text)
    if not video_match or not files_match:
        return None

    video_title = normalize_discord_title(video_match.group(1))
    files_url = files_match.group(1).strip().rstrip(").,")
    folder_id = extract_drive_folder_id(files_url)
    if not folder_id:
        return None

    message_id = str(message.get("id") or "").strip()
    message_timestamp = str(
        message.get("timestamp")
        or message.get("timestampParsed")
        or message.get("date")
        or ""
    ).strip()

    return {
        "source": "discord",
        "account": account_name,
        "channel": str(channel_code or "").upper() or "RRU",
        "discord_channel_id": str(discord_channel_id),
        "discord_message_id": message_id,
        "message_timestamp": message_timestamp,
        "video_title": video_title,
        "files_url": files_url,
        "drive_folder_id": folder_id,
    }


def discord_candidate_key(candidate: dict[str, Any]) -> str:
    """Build stable key for dedupe/tracking."""
    channel = str(candidate.get("channel") or "").upper() or "RRU"
    folder_id = str(candidate.get("drive_folder_id") or "").strip().lower()
    return f"{channel}:{folder_id}"


def load_discord_state(path: Path) -> dict[str, Any]:
    """Load or initialize Discord project tracking state."""
    if not path.exists():
        return {
            "schema_version": "1.0.0",
            "updated_at": now_iso(),
            "projects": {},
        }
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {
            "schema_version": "1.0.0",
            "updated_at": now_iso(),
            "projects": {},
        }
    projects = payload.get("projects")
    if not isinstance(projects, dict):
        payload["projects"] = {}
    return payload


def write_discord_state(path: Path, state: dict[str, Any]) -> None:
    """Persist Discord tracking state."""
    state["schema_version"] = "1.0.0"
    state["updated_at"] = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def summarize_discord_prepare_warnings(warnings: list[str]) -> dict[str, int]:
    """Classify Discord prepare warnings into blocking vs informational buckets."""
    summary = {
        "total": 0,
        "blocking": 0,
        "channel_conflicts": 0,
        "other": 0,
    }
    for warning in warnings or []:
        summary["total"] += 1
        value = normalize_lower(warning)
        if (
            "discord export failed" in value
            or "invalid discord export json" in value
            or "missing discord_token or discord_channel_ids" in value
        ):
            summary["blocking"] += 1
        elif "discord channel conflict" in value:
            summary["channel_conflicts"] += 1
        else:
            summary["other"] += 1
    return summary


def latest_discord_message_timestamp_for_channel(
    projects: dict[str, Any] | None,
    channel_code: str,
) -> datetime | None:
    """Return latest known Discord message timestamp for a channel from tracking state."""
    if not isinstance(projects, dict):
        return None
    target_channel = str(channel_code or "").upper()
    latest: datetime | None = None
    for payload in projects.values():
        if not isinstance(payload, dict):
            continue
        payload_channel = str(payload.get("channel") or "").upper()
        if target_channel and payload_channel and payload_channel != target_channel:
            continue
        timestamp = parse_iso_datetime(payload.get("message_timestamp") or payload.get("updated_at"))
        if timestamp and (latest is None or timestamp > latest):
            latest = timestamp
    return latest


def resolve_discord_export_after_timestamp(
    *,
    channel_code: str,
    lookback_cutoff: datetime | None,
    tracked_projects: dict[str, Any] | None,
    overlap_minutes: int = DEFAULT_DISCORD_INCREMENTAL_LOOKBACK_MINUTES,
) -> datetime | None:
    """
    Compute an export "after" timestamp.

    Preference:
    1) Most recent tracked message for this channel minus overlap window
    2) Explicit lookback cutoff
    3) No lower bound
    """
    chosen = lookback_cutoff
    latest_seen = latest_discord_message_timestamp_for_channel(tracked_projects, channel_code)
    if latest_seen:
        overlap = max(0, int(overlap_minutes))
        incremental = latest_seen - timedelta(minutes=overlap)
        if chosen is None or incremental > chosen:
            chosen = incremental
    return chosen


def format_discord_after_value(value: datetime | None) -> str:
    """Format timestamp for DiscordChatExporter --after argument."""
    if not value:
        return ""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_discord_pipeline_candidates(
    accounts: list[dict[str, Any]],
    *,
    account_filters: list[str] | None = None,
    channel_filters: list[str] | None = None,
    lookback_hours: int = 168,
    exporter: str = "",
    tracked_projects: dict[str, Any] | None = None,
    incremental_lookback_minutes: int = DEFAULT_DISCORD_INCREMENTAL_LOOKBACK_MINUTES,
    export_max_attempts: int = DEFAULT_DISCORD_EXPORT_MAX_ATTEMPTS,
    export_base_backoff_seconds: float = DEFAULT_DISCORD_EXPORT_BASE_BACKOFF_SECONDS,
    export_max_backoff_seconds: float = DEFAULT_DISCORD_EXPORT_MAX_BACKOFF_SECONDS,
    export_timeout_seconds: int = DEFAULT_DISCORD_EXPORT_TIMEOUT_SECONDS,
    channel_cooldown_seconds: float = DEFAULT_DISCORD_CHANNEL_COOLDOWN_SECONDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Scan configured Discord channels and collect pipeline-complete project candidates."""
    selected_accounts = accounts
    if account_filters:
        names = {str(value).strip().lower() for value in account_filters if str(value).strip()}
        selected_accounts = [acc for acc in accounts if str(acc.get("name", "")).lower() in names]

    selected_channels = {str(value).strip().upper() for value in (channel_filters or []) if str(value).strip()}
    cutoff: datetime | None = None
    if int(lookback_hours) > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(lookback_hours))

    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    # Resolve one owner per Discord channel ID to avoid duplicate exports and
    # cross-brand routing when multiple accounts point to the same channel.
    channel_owners: dict[str, dict[str, str]] = {}
    for account in selected_accounts:
        account_name = str(account.get("name") or "unknown")
        channel_code = str(account.get("default_channel") or "RRU").upper()
        if selected_channels and channel_code not in selected_channels:
            continue

        token = str(account.get("discord_token") or "").strip()
        channel_ids = [str(item).strip() for item in (account.get("discord_channel_ids") or []) if str(item).strip()]
        if not token or not channel_ids:
            warnings.append(f"Account '{account_name}' missing DISCORD_TOKEN or DISCORD_CHANNEL_IDS (skipped)")
            continue

        account_exporter = str(account.get("discord_exporter") or "").strip()
        exporter_command = exporter or account_exporter
        for discord_channel_id in channel_ids:
            existing = channel_owners.get(discord_channel_id)
            if existing:
                if str(existing.get("channel_code") or "") != channel_code:
                    warnings.append(
                        "Discord channel conflict: "
                        f"id={discord_channel_id} mapped to channel={existing['channel_code']} "
                        f"(account={existing['account_name']}) and channel={channel_code} "
                        f"(account={account_name}); using account={existing['account_name']}"
                    )
                continue
            channel_owners[discord_channel_id] = {
                "account_name": account_name,
                "channel_code": channel_code,
                "token": token,
                "exporter_command": exporter_command,
            }

    ordered_owners = sorted(
        channel_owners.items(),
        key=lambda item: (
            str(item[1].get("channel_code") or ""),
            str(item[1].get("account_name") or ""),
            str(item[0]),
        ),
    )
    total_channels = len(ordered_owners)
    cooldown_seconds = max(0.0, float(channel_cooldown_seconds))

    for owner_index, (discord_channel_id, owner) in enumerate(ordered_owners, start=1):
        account_name = owner["account_name"]
        channel_code = owner["channel_code"]
        token = owner["token"]
        exporter_command = owner["exporter_command"]
        after_timestamp = resolve_discord_export_after_timestamp(
            channel_code=channel_code,
            lookback_cutoff=cutoff,
            tracked_projects=tracked_projects,
            overlap_minutes=incremental_lookback_minutes,
        )
        after_value = format_discord_after_value(after_timestamp)
        if after_value:
            print_info(
                f"Discord export window for {account_name}:{discord_channel_id} "
                f"starts after {after_value}"
            )

        with tempfile.TemporaryDirectory(prefix="discord_export_") as temp_dir:
            export_path = Path(temp_dir) / f"{account_name}_{discord_channel_id}.json"
            ok, detail = run_discord_export(
                token=token,
                channel_id=discord_channel_id,
                output_path=export_path,
                exporter=exporter_command,
                after=after_value,
                max_attempts=export_max_attempts,
                base_backoff_seconds=export_base_backoff_seconds,
                max_backoff_seconds=export_max_backoff_seconds,
                timeout_seconds=export_timeout_seconds,
            )
            if not ok:
                warnings.append(
                    f"Discord export failed for account={account_name} channel={discord_channel_id}: {detail}"
                )
                continue
            print_info(f"Discord export succeeded for {account_name}:{discord_channel_id} via {detail}")

            try:
                messages = load_discord_export_messages(export_path)
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(
                    f"Invalid Discord export JSON for account={account_name} channel={discord_channel_id}: {exc}"
                )
                continue

            for message in messages:
                candidate = parse_discord_pipeline_message(
                    message,
                    account_name=account_name,
                    channel_code=channel_code,
                    discord_channel_id=discord_channel_id,
                )
                if not candidate:
                    continue
                timestamp = parse_iso_datetime(candidate.get("message_timestamp"))
                if cutoff and timestamp and timestamp < cutoff:
                    continue
                candidates.append(candidate)

        if cooldown_seconds > 0.0 and owner_index < total_channels:
            print_info(
                f"Cooldown {cooldown_seconds:.1f}s before next Discord export "
                f"({owner_index}/{total_channels})"
            )
            time.sleep(cooldown_seconds)

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = discord_candidate_key(candidate)
        existing = deduped.get(key)
        if not existing:
            deduped[key] = candidate
            continue
        current_ts = parse_iso_datetime(candidate.get("message_timestamp"))
        existing_ts = parse_iso_datetime(existing.get("message_timestamp"))
        if current_ts and (not existing_ts or current_ts >= existing_ts):
            deduped[key] = candidate

    sorted_candidates = sorted(
        deduped.values(),
        key=lambda item: (
            parse_iso_datetime(item.get("message_timestamp")) or datetime.min.replace(tzinfo=timezone.utc),
            str(item.get("video_title") or "").lower(),
        ),
    )
    return sorted_candidates, warnings


def extract_project_path_from_newproject_stdout(output: str) -> str:
    """Extract created project path from newproject output text."""
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if "Project:" not in line:
            continue
        _, _, tail = line.partition("Project:")
        candidate = tail.strip()
        if candidate:
            return candidate
    return ""


def command_discord_prepare(args: argparse.Namespace) -> int:
    """Scan Discord Pipeline Complete messages and create local projects."""
    accounts = load_accounts(DEFAULT_ACCOUNTS_DIR)
    if not accounts:
        print_error(f"No valid accounts found in {DEFAULT_ACCOUNTS_DIR}", exit_code=1)

    discord_state_file = Path(args.discord_state_file)
    tracker = load_discord_state(discord_state_file)
    projects = tracker.get("projects", {})
    if not isinstance(projects, dict):
        projects = {}
        tracker["projects"] = projects
    state_file = Path(args.state_file)
    trello_state: dict[str, Any] = {}
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            trello_state = payload

    discovered, warnings = collect_discord_pipeline_candidates(
        accounts=accounts,
        account_filters=args.account,
        channel_filters=args.channel,
        lookback_hours=args.lookback_hours,
        exporter=str(args.exporter or "").strip(),
        tracked_projects=projects,
        incremental_lookback_minutes=args.incremental_lookback_minutes,
        export_max_attempts=args.export_max_attempts,
        export_base_backoff_seconds=args.export_base_backoff_seconds,
        export_max_backoff_seconds=args.export_max_backoff_seconds,
        export_timeout_seconds=args.export_timeout_seconds,
        channel_cooldown_seconds=args.channel_cooldown_seconds,
    )
    warning_summary = summarize_discord_prepare_warnings(warnings)
    for warning in warnings:
        print_warn(warning)

    pending: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    skipped_existing_local: list[dict[str, Any]] = []
    skipped_unresolved_card: list[dict[str, Any]] = []
    for candidate in discovered:
        key = discord_candidate_key(candidate)
        if not args.rescan and key in projects:
            skipped_existing.append(candidate)
            continue
        card_match = resolve_discord_candidate_trello_card(trello_state, candidate)
        matched_card_id = str(card_match.get("card_id") or "").strip()
        existing_project_paths: list[str] = []
        if matched_card_id:
            existing_project_paths = [
                str(entry.get("path") or "")
                for entry in resolve_local_project_dir_matches(
                    str(candidate.get("channel") or ""),
                    matched_card_id,
                    card_title=str(candidate.get("video_title") or ""),
                )
                if str(entry.get("path") or "").strip()
            ]
        if not existing_project_paths:
            existing_project_paths = find_local_project_dirs_by_title(
                str(candidate.get("channel") or ""),
                str(candidate.get("video_title") or ""),
            )
        if existing_project_paths:
            skipped_existing_local.append(
                {
                    **candidate,
                    **card_match,
                    "existing_project_paths": existing_project_paths,
                }
            )
            continue
        if not matched_card_id:
            skipped_unresolved_card.append(
                {
                    **candidate,
                    **card_match,
                    "reason": "missing_trello_card_match",
                }
            )
            continue
        pending.append({**candidate, **card_match})

    if args.limit > 0:
        pending = pending[: int(args.limit)]

    if args.json:
        payload = {
            "discovered": discovered,
            "pending": pending,
            "skipped_existing": skipped_existing,
            "skipped_existing_local": skipped_existing_local,
            "skipped_unresolved_card": skipped_unresolved_card,
            "warnings": warnings,
            "warning_summary": warning_summary,
            "counts": {
                "discovered": len(discovered),
                "pending": len(pending),
                "skipped_existing": len(skipped_existing),
                "skipped_existing_local": len(skipped_existing_local),
                "skipped_unresolved_card": len(skipped_unresolved_card),
                "warnings": len(warnings),
                "blocking_warnings": warning_summary["blocking"],
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.dry_run:
            return 1 if warning_summary["blocking"] else 0

    if not pending:
        if not args.dry_run and skipped_existing_local:
            for candidate in skipped_existing_local:
                key = discord_candidate_key(candidate)
                existing_paths = [
                    str(path)
                    for path in (candidate.get("existing_project_paths") or [])
                    if str(path).strip()
                ]
                projects[key] = {
                    **candidate,
                    "status": "existing_local",
                    "updated_at": now_iso(),
                    "project_path": existing_paths[0] if existing_paths else "",
                }
            write_discord_state(discord_state_file, tracker)
        if warning_summary["blocking"]:
            print_warn("Discord prepare did not finish cleanly; export coverage is incomplete")
            return 1
        if skipped_existing_local:
            print_ok(f"Already present locally: {len(skipped_existing_local)}")
        if skipped_unresolved_card:
            print_warn(
                "Skipped without Trello card match: "
                f"{len(skipped_unresolved_card)}"
            )
        print_ok("No new Discord pipeline-complete projects to prepare")
        return 0

    print_header("DISCORD PROJECT PREPARE")
    print_ok(f"Discovered: {len(discovered)}")
    print_ok(f"Already tracked: {len(skipped_existing)}")
    print_ok(f"Already present locally: {len(skipped_existing_local)}")
    print_ok(f"Skipped without Trello card match: {len(skipped_unresolved_card)}")
    print_ok(f"Queued for creation: {len(pending)}")

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, candidate in enumerate(pending, start=1):
        project_title = normalize_discord_title(str(candidate.get("video_title") or "Untitled Discord Project"))
        channel = str(candidate.get("channel") or "RRU").upper()
        files_url = str(candidate.get("files_url") or "").strip()
        key = discord_candidate_key(candidate)
        card_match = resolve_discord_candidate_trello_card(trello_state, candidate)
        matched_card_id = str(card_match.get("card_id") or "").strip()

        print_header(f"DISCORD SETUP {index}/{len(pending)}")
        print_info(f"Account: {candidate.get('account')}")
        print_info(f"Channel: {channel}")
        print_info(f"Video: {truncate_text(project_title, 140)}")
        print_info(f"Files: {files_url}")
        if matched_card_id:
            print_info(
                f"Trello: {matched_card_id} "
                f"({str(card_match.get('match_strategy') or 'matched')})"
            )

        if args.dry_run:
            print_ok("Dry run: skipped execution")
            continue

        result = run_newproject_for_discord(
            project_title,
            channel,
            files_url,
            account_name=str(candidate.get("account") or ""),
            card_id=matched_card_id,
        )
        if result.returncode != 0:
            stderr_tail = "\n".join((result.stderr or "").splitlines()[-15:]).strip()
            stdout_tail = "\n".join((result.stdout or "").splitlines()[-15:]).strip()
            failure = {
                **candidate,
                **card_match,
                "key": key,
                "status": "failed",
                "returncode": result.returncode,
            }
            failures.append(failure)
            projects[key] = {
                **candidate,
                **card_match,
                "status": "failed",
                "updated_at": now_iso(),
                "returncode": result.returncode,
                "error": stderr_tail or stdout_tail or f"exit={result.returncode}",
            }
            print_warn(f"Project setup failed for Discord candidate {key} (exit={result.returncode})")
            if stdout_tail:
                print_info("newproject stdout (tail):")
                print(console_safe_text(stdout_tail))
            if stderr_tail:
                print_info("newproject stderr (tail):")
                print(console_safe_text(stderr_tail))
            continue

        project_path = extract_project_path_from_newproject_stdout(result.stdout or "")
        success = {
            **candidate,
            **card_match,
            "key": key,
            "status": "created",
            "project_path": project_path,
        }
        successes.append(success)
        projects[key] = {
            **candidate,
            **card_match,
            "status": "created",
            "updated_at": now_iso(),
            "project_path": project_path,
        }
        print_ok(f"Prepared Discord project: {project_title}")
        if project_path:
            print_ok(f"Project path: {project_path}")

    if not args.dry_run:
        for candidate in skipped_existing_local:
            key = discord_candidate_key(candidate)
            existing_paths = [str(path) for path in (candidate.get("existing_project_paths") or []) if str(path).strip()]
            projects[key] = {
                **candidate,
                "status": "existing_local",
                "updated_at": now_iso(),
                "project_path": existing_paths[0] if existing_paths else "",
            }
        write_discord_state(discord_state_file, tracker)
        if successes and not args.no_sync_after:
            print_info("Refreshing queue state after Discord project creation")
            try:
                sync_state(Path(args.state_file), refresh_lipsync=False)
            except Exception as exc:
                print_warn(f"Could not refresh queue state after Discord prepare: {exc}")

    print_header("DISCORD PREPARE SUMMARY")
    print_ok(f"Created: {len(successes)}")
    print_ok(f"Failed: {len(failures)}")
    print_ok(f"Skipped existing: {len(skipped_existing)}")
    print_ok(f"Skipped existing local: {len(skipped_existing_local)}")
    if warning_summary["blocking"]:
        print_warn(f"Blocking warnings: {warning_summary['blocking']}")
    return 0 if not failures and not warning_summary["blocking"] else 1


def find_voiceover_files_for_card(channel: str, card_id: str) -> list[Path]:
    """Locate voiceover files for a card in matching local project dirs."""
    card_key = str(card_id or "").lower()
    files: list[Path] = []

    for channel_dir in channel_project_dirs(channel):
        for project_dir in sorted(channel_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            if card_key not in project_dir.name.lower():
                continue
            voiceover_dir = project_dir / "voiceover"
            if not voiceover_dir.exists():
                continue
            for candidate in voiceover_dir.iterdir():
                if candidate.is_file() and candidate.suffix.lower() in LIKELY_AUDIO_EXTENSIONS:
                    files.append(candidate)

    return sorted(files, key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)


def is_valid_voiceover_filename_for_card(path: Path, card_id: str) -> bool:
    """Validate required naming format: [cardid]-[few-words]-[card-title].ext."""
    name = path.name.lower()
    prefix = f"{str(card_id).lower()}-"
    if not name.startswith(prefix):
        return False
    if not VOICEOVER_FILENAME_PATTERN.match(name):
        return False
    # Require at least 3 parts in stem: cardid + first-words + title.
    stem = path.stem.lower()
    return len(stem.split("-")) >= 3


def find_voiceover_files_for_project(project_dir: Path) -> list[Path]:
    """Locate voiceover files inside one prepared project directory."""
    voiceover_dir = project_dir / "voiceover"
    if not voiceover_dir.exists() or not voiceover_dir.is_dir():
        return []

    files: list[Path] = []
    for candidate in voiceover_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in LIKELY_AUDIO_EXTENSIONS:
            files.append(candidate)

    return sorted(files, key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)


def load_project_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load a project checkpoint payload, handling gzip-compressed files transparently."""
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        return None

    try:
        with open(checkpoint_path, "rb") as handle:
            header = handle.read(2)

        open_fn = gzip.open if header == b"\x1f\x8b" else open
        with open_fn(checkpoint_path, "rt", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile):
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _count_usable_caption_entries(caption_results: dict[str, Any]) -> int:
    """Count caption result entries that include at least one usable text segment."""
    usable_count = 0
    for result in caption_results.values():
        if not isinstance(result, dict):
            continue
        if result.get("unavailable") or result.get("error") or result.get("skipped"):
            continue

        segments = result.get("segments") or []
        if not isinstance(segments, list):
            continue

        has_text = any(
            isinstance(segment, dict) and str(segment.get("text") or "").strip()
            for segment in segments
        )
        if has_text:
            usable_count += 1

    return usable_count


def detect_checkpoint_fresh_fallback(checkpoint_path: Path) -> dict[str, Any] | None:
    """Return a fallback descriptor when a CAPTION resume would immediately fail at MATCH."""
    payload = load_project_checkpoint_payload(checkpoint_path)
    if not payload:
        return None

    last_stage = str(payload.get("last_completed_stage") or "").strip().upper()
    if last_stage != "CAPTION":
        return None

    caption_data = payload.get("caption") or {}
    if not isinstance(caption_data, dict):
        return None

    caption_results = caption_data.get("caption_results") or {}
    if not isinstance(caption_results, dict) or not caption_results:
        return None

    total_segments_raw = caption_data.get("total_segments", 0)
    try:
        total_segments = int(total_segments_raw or 0)
    except (TypeError, ValueError):
        total_segments = 0

    usable_caption_entries = _count_usable_caption_entries(caption_results)
    if total_segments > 0 or usable_caption_entries > 0:
        return None

    return {
        "reason": "caption_checkpoint_without_usable_text",
        "last_completed_stage": last_stage,
        "caption_result_count": len(caption_results),
        "usable_caption_entries": usable_caption_entries,
        "total_segments": total_segments,
    }


def select_fresh_launch_voiceover(
    project_dir: Path,
    card_id: str,
    *,
    allow_generic_fallback: bool = False,
) -> Path | None:
    """Pick the voiceover file for a fresh queue launch."""
    voiceover_files = find_voiceover_files_for_project(project_dir)
    valid_voiceovers = [
        path for path in voiceover_files
        if is_valid_voiceover_filename_for_card(path, card_id)
    ]
    if valid_voiceovers:
        return valid_voiceovers[0]

    if not allow_generic_fallback:
        return None

    from src.cli.newproject import select_best_voiceover_file

    return select_best_voiceover_file(voiceover_files)


def build_pipeline_launch_command(
    project_dir: Path,
    *,
    voiceover_path: Path | None = None,
    resume: bool = False,
) -> list[str]:
    """Build a non-interactive pipeline launch command for one prepared project."""
    project_dir = Path(project_dir).resolve()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--project",
        str(project_dir),
        "--non-interactive",
    ]
    if resume:
        command.append("--resume")
        return command
    if voiceover_path is None:
        raise ValueError("voiceover_path is required for fresh pipeline launches")
    command.extend(["--fresh", "--voiceover", str(Path(voiceover_path).resolve()), "--save-keywords"])
    return command


def resolve_next_ready_launch_plan(
    state: dict[str, Any],
    *,
    requested_card_ids: list[str] | None = None,
    requested_channels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve the next ready pipeline into a concrete launch plan."""
    pipelines = state.get("pipelines", {}) or {}
    queue = state.get("queue", {}) or {}
    running_ids = resolve_running_card_ids(state, pipelines)
    if running_ids:
        return {
            "status": "already_running",
            "running_card_ids": running_ids,
        }

    requested_ids = {
        str(card_id).strip().lower()
        for card_id in (requested_card_ids or [])
        if str(card_id).strip()
    }
    ready_ids = [str(card_id).strip() for card_id in (queue.get("ready_card_ids", []) or []) if str(card_id).strip()]
    if requested_ids:
        ready_ids = [card_id for card_id in ready_ids if card_id.lower() in requested_ids]
    if requested_channels:
        channel_set = normalize_channels(requested_channels)
        ready_ids = [
            card_id for card_id in ready_ids
            if get_pipeline_channel(pipelines, card_id) in channel_set
        ]
    if not ready_ids:
        payload: dict[str, Any] = {"status": "no_ready_pipeline"}
        if requested_ids:
            payload["requested_card_ids"] = sorted(requested_ids)
        return payload

    first_issue: dict[str, Any] | None = None
    for card_id in ready_ids:
        entry = get_pipeline(pipelines, card_id)
        if not entry:
            continue

        project_dirs = [
            Path(str(path))
            for path in (((entry.get("project") or {}).get("local_project_dirs", []) or []))
            if str(path).strip()
        ]
        existing_dirs = [project_dir for project_dir in project_dirs if project_dir.exists() and project_dir.is_dir()]
        if not existing_dirs:
            if first_issue is None:
                first_issue = {
                    "status": "missing_project_dir",
                    "card_id": card_id,
                    "title": str(entry.get("title") or "Untitled card"),
                }
            continue

        for project_dir in existing_dirs:
            checkpoint_path = project_dir / "checkpoint.json"
            checkpoint_fallback = detect_checkpoint_fresh_fallback(checkpoint_path)
            valid_voiceover = select_fresh_launch_voiceover(project_dir, card_id)
            fallback_voiceover = select_fresh_launch_voiceover(
                project_dir,
                card_id,
                allow_generic_fallback=True,
            )

            # Log launch decision inputs for debugging
            checkpoint_exists = checkpoint_path.exists()
            checkpoint_payload = load_project_checkpoint_payload(checkpoint_path) if checkpoint_exists else None
            checkpoint_stage = (checkpoint_payload or {}).get("last_completed_stage", "none")
            print_info(
                f"Launch decision for {card_id}: "
                f"checkpoint_exists={checkpoint_exists} "
                f"checkpoint_stage={checkpoint_stage} "
                f"checkpoint_fallback={checkpoint_fallback is not None} "
                f"valid_voiceover={valid_voiceover is not None} "
                f"fallback_voiceover={fallback_voiceover is not None}"
            )

            if checkpoint_exists and checkpoint_fallback is None:
                return {
                    "status": "launchable",
                    "card_id": card_id,
                    "title": str(entry.get("title") or "Untitled card"),
                    "project_dir": str(project_dir),
                    "channel": str((entry.get("project") or {}).get("channel") or ""),
                    "mode": "resume",
                    "voiceover": str(valid_voiceover) if valid_voiceover else "",
                    "command": build_pipeline_launch_command(project_dir, resume=True),
                }

            if checkpoint_fallback and fallback_voiceover:
                return {
                    "status": "launchable",
                    "card_id": card_id,
                    "title": str(entry.get("title") or "Untitled card"),
                    "project_dir": str(project_dir),
                    "channel": str((entry.get("project") or {}).get("channel") or ""),
                    "mode": "fresh",
                    "voiceover": str(fallback_voiceover),
                    "checkpoint_fallback": checkpoint_fallback,
                    "command": build_pipeline_launch_command(project_dir, voiceover_path=fallback_voiceover, resume=False),
                }

            if valid_voiceover:
                chosen = valid_voiceover
                return {
                    "status": "launchable",
                    "card_id": card_id,
                    "title": str(entry.get("title") or "Untitled card"),
                    "project_dir": str(project_dir),
                    "channel": str((entry.get("project") or {}).get("channel") or ""),
                    "mode": "fresh",
                    "voiceover": str(chosen),
                    "command": build_pipeline_launch_command(project_dir, voiceover_path=chosen, resume=False),
                }

            # Fall back to generic voiceover (e.g. voiceover.mp3 without card-id prefix)
            if fallback_voiceover:
                print_warn(
                    f"No card-prefixed voiceover for {card_id}, "
                    f"using generic fallback: {Path(str(fallback_voiceover)).name}"
                )
                return {
                    "status": "launchable",
                    "card_id": card_id,
                    "title": str(entry.get("title") or "Untitled card"),
                    "project_dir": str(project_dir),
                    "channel": str((entry.get("project") or {}).get("channel") or ""),
                    "mode": "fresh",
                    "voiceover": str(fallback_voiceover),
                    "command": build_pipeline_launch_command(project_dir, voiceover_path=fallback_voiceover, resume=False),
                }

        if first_issue is None:
            first_issue = {
                "status": "missing_voiceover",
                "card_id": card_id,
                "title": str(entry.get("title") or "Untitled card"),
                "project_dir": str(existing_dirs[0]),
            }

    return first_issue or {"status": "no_ready_pipeline"}


def launch_pipeline_detached(command: list[str]) -> subprocess.Popen[Any]:
    """Launch a pipeline process in the background and return the process handle."""
    creationflags = 0
    for attr in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS", "CREATE_NO_WINDOW"):
        creationflags |= int(getattr(subprocess, attr, 0) or 0)

    kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if creationflags:
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True

    return popen_subprocess(command, **kwargs)


def maybe_autostart_next_ready_pipeline(
    state_file: Path,
    *,
    requested_card_ids: list[str] | None = None,
    requested_channels: Sequence[str] | None = None,
    startup_wait_seconds: float = DEFAULT_PIPELINE_AUTOSTART_WAIT_SECONDS,
) -> dict[str, Any]:
    """Launch the next ready pipeline in the background, if possible."""
    state = load_state_or_error(state_file)
    plan = resolve_next_ready_launch_plan(
        state,
        requested_card_ids=requested_card_ids,
        requested_channels=requested_channels,
    )
    if plan.get("status") != "launchable":
        return plan

    # Log launch decision for debugging resume vs fresh issues
    mode = plan.get("mode", "unknown")
    card_id = plan.get("card_id", "?")
    command = plan.get("command", [])
    command_str = " ".join(str(x) for x in command)
    print_info(f"Launch plan: card={card_id} mode={mode}")
    print_info(f"Command: {command_str}")

    # Safety check: if mode is fresh but checkpoint exists, prefer resume
    if mode == "fresh":
        project_dir = Path(str(plan.get("project_dir") or ""))
        checkpoint_path = project_dir / "checkpoint.json"
        if checkpoint_path.exists():
            payload = load_project_checkpoint_payload(checkpoint_path)
            if payload and payload.get("last_completed_stage"):
                last_stage = payload["last_completed_stage"]
                print_warn(
                    f"Plan chose fresh but checkpoint exists with stage={last_stage}. "
                    f"Overriding to resume to preserve progress."
                )
                plan["mode"] = "resume"
                plan["command"] = build_pipeline_launch_command(project_dir, resume=True)
                command = plan["command"]

    process = launch_pipeline_detached(list(plan.get("command") or []))
    result = dict(plan)
    result["status"] = "launched"
    result["pid"] = int(getattr(process, "pid", 0) or 0)

    if startup_wait_seconds > 0:
        time.sleep(startup_wait_seconds)

    return result


def command_prepare(args: argparse.Namespace) -> int:
    """Prepare missing local projects by invoking src.cli.newproject."""
    state_file = Path(args.state_file)
    if args.sync_first:
        sync_state(state_file, args.refresh_lipsync)

    state = load_state_or_error(state_file)
    targets = collect_unprepared_targets(
        state=state,
        requested_card_ids=args.card_id,
        requested_channels=args.channel,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps({"targets": targets, "count": len(targets)}, indent=2, ensure_ascii=False))
        if args.dry_run:
            return 0

    if not targets and not args.run_ready:
        print_ok("No missing projects detected for actionable pipeline cards")
        return 0

    if targets:
        print_header("PREPARE MISSING PROJECTS")
        print_ok(f"Cards queued for setup: {len(targets)}")
    else:
        print_ok("No missing projects detected for actionable pipeline cards")

    failures: list[dict[str, str]] = []
    successes: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for index, target in enumerate(targets, start=1):
        card_id = target["card_id"]
        title = target["title"]
        card_url = target["card_url"]
        channel = target["channel"]

        vo_doc_url = str(target.get("vo_doc_url") or "").strip()
        # Prefer VO Google Doc URL (contains Drive links to audio parts)
        effective_url = vo_doc_url if vo_doc_url else card_url

        print_header(f"SETUP {index}/{len(targets)}")
        print_info(f"Card: {card_id} - {truncate_text(title, 120)}")
        print_info(f"URL: {effective_url}")
        if vo_doc_url:
            print_info("Source: VO Google Doc (Drive audio parts)")

        if args.dry_run:
            print_ok("Dry run: skipped execution")
            continue

        # Check if VO source exists before running newproject — if the card has
        # no VO candidates and the project dir already exists, skip rather than
        # re-running newproject and failing on missing VO every cycle.
        vo_doc_url_value = str(target.get("vo_doc_url") or "").strip()
        local_dirs = ((get_pipeline(state.get("pipelines", {}), card_id) or {})
                      .get("project", {}).get("local_project_dirs", []) or [])
        project_exists = any(Path(d).is_dir() for d in local_dirs)
        if project_exists and not vo_doc_url_value:
            skipped.append({"card_id": card_id, "reason": "no_voiceover_source"})
            print_info(
                f"Skipped {card_id}: project exists but no VO source "
                "(no Google Doc link in card description)"
            )
            continue

        result = run_newproject_for_card(
            project_name=title,
            channel=channel,
            card_url=effective_url,
            account_name=str(target.get("account") or ""),
            card_id=str(target.get("card_id") or ""),
        )
        if result.returncode != 0:
            stderr_tail = "\n".join((result.stderr or "").splitlines()[-15:]).strip()
            stdout_tail = "\n".join((result.stdout or "").splitlines()[-15:]).strip()
            failures.append({"card_id": card_id, "reason": "newproject_failed"})
            print_warn(f"Project setup failed for {card_id} (exit={result.returncode})")
            if stdout_tail:
                print_info("newproject stdout (tail):")
                print(console_safe_text(stdout_tail))
            if stderr_tail:
                print_info("newproject stderr (tail):")
                print(console_safe_text(stderr_tail))
            continue

        voiceover_files = find_voiceover_files_for_card(channel, card_id)
        named = [path for path in voiceover_files if is_valid_voiceover_filename_for_card(path, card_id)]
        if not named:
            # For re-runs (project has checkpoint), accept any existing voiceover
            # file without strict naming validation — the file was already used
            # in a prior pipeline run.
            is_rerun = False
            if voiceover_files:
                for vo_path in voiceover_files:
                    project_dir = vo_path.parent.parent  # voiceover/ -> project dir
                    if (project_dir / "checkpoint.json").exists():
                        is_rerun = True
                        break

            if is_rerun:
                chosen = voiceover_files[0]
                successes.append({"card_id": card_id, "voiceover": str(chosen)})
                print_ok(
                    f"Re-run detected for {card_id} (checkpoint exists), "
                    f"using existing voiceover: {chosen.name}"
                )
                continue

            failures.append({"card_id": card_id, "reason": "voiceover_filename_validation_failed"})
            print_warn(
                "Project created but no voiceover file matched required pattern "
                f"[cardid]-[first-few-words]-[card-title]: card={card_id}"
            )
            if voiceover_files:
                preview = ", ".join(path.name for path in voiceover_files[:5])
                print_info(f"Found voiceover files: {preview}")
            continue

        chosen = named[0]
        successes.append({"card_id": card_id, "voiceover": str(chosen)})
        print_ok(f"Prepared {card_id} with voiceover: {chosen.name}")

    if not args.dry_run:
        # Refresh local project paths and queue buckets after setup work.
        sync_state(state_file, refresh_lipsync=False)

    print_header("PREPARE SUMMARY")
    print_ok(f"Prepared successfully: {len(successes)}")
    if skipped:
        print_ok(f"Skipped (no VO source): {len(skipped)}")
        for entry in skipped:
            print(f"  - {entry['card_id']}: {entry['reason']}")
    print_ok(f"Failed: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"  - {failure['card_id']}: {failure['reason']}")

    exit_code = 0 if not failures else 1

    if args.run_ready and not args.dry_run:
        print_header("AUTO-START READY PIPELINE")
        launch_result = maybe_autostart_next_ready_pipeline(
            state_file,
            requested_card_ids=args.card_id,
            requested_channels=args.channel,
        )
        status = str(launch_result.get("status") or "")
        if status == "launched":
            print_ok(
                f"Started {launch_result.get('card_id')} "
                f"({launch_result.get('mode')}) pid={launch_result.get('pid')}"
            )
            print_info(f"Project: {launch_result.get('project_dir')}")
            if launch_result.get("voiceover"):
                print_info(f"Voiceover: {launch_result.get('voiceover')}")
            try:
                sync_state(state_file, refresh_lipsync=False)
            except Exception as exc:
                print_warn(f"Could not refresh queue state after auto-start: {exc}")
        elif status == "already_running":
            running_ids = launch_result.get("running_card_ids") or []
            preview = ", ".join(str(card_id) for card_id in running_ids[:5])
            suffix = " ..." if len(running_ids) > 5 else ""
            print_warn(f"Skipped auto-start because pipeline is already running: {preview}{suffix}")
        elif status == "no_ready_pipeline":
            print_ok("No ready pipeline to auto-start after preparation")
        else:
            print_warn(
                f"Could not auto-start ready pipeline for "
                f"{launch_result.get('card_id') or 'unknown card'}: {status}"
            )
            exit_code = 1

    return exit_code


def truncate_text(text: Any, max_len: int = 100) -> str:
    """Truncate text for compact CLI display."""
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def iter_preview(items: list[str], limit: int) -> list[str]:
    """Return a bounded preview list. limit<=0 means all."""
    if limit <= 0:
        return list(items)
    return list(items[:limit])


def get_pipeline(pipelines: dict[str, Any], card_id: str) -> dict[str, Any]:
    """Lookup pipeline entry by card id (case-insensitive)."""
    return (pipelines or {}).get(str(card_id).lower(), {}) or {}


def normalize_channels(values: Sequence[Any]) -> tuple[str, ...]:
    """Normalize channel codes to sorted unique uppercase strings."""
    return tuple(sorted({str(v).strip().upper() for v in values if str(v).strip()}))


def get_pipeline_channel(pipelines: dict[str, Any], card_id: str) -> str:
    """Extract the channel code for a card from pipelines (uppercase, empty if unknown)."""
    entry = get_pipeline(pipelines, card_id)
    return str((entry.get("project") or {}).get("channel") or "").strip().upper()


def collect_running_card_ids(pipelines: dict[str, Any]) -> list[str]:
    """Collect currently-running card IDs from per-project local progress."""
    running: list[tuple[int, str]] = []
    for entry in (pipelines or {}).values():
        pipeline_state = str(entry.get("pipeline_run_state") or entry.get("pipeline_state") or "").strip().lower()
        local_progress = ((entry.get("project") or {}).get("local_progress")) or {}
        if pipeline_state == "completed" or local_progress.get("completed_signal"):
            continue
        if not local_progress.get("running_signal"):
            continue
        card_id = str(entry.get("card_id") or "").strip()
        if not card_id:
            continue
        age_raw = local_progress.get("latest_log_age_seconds")
        age_seconds = int(age_raw) if isinstance(age_raw, (int, float)) else 999999999
        running.append((age_seconds, card_id))

    running.sort(key=lambda item: (item[0], item[1].lower()))
    return [card_id for _, card_id in running]


def resolve_running_card_ids(state: dict[str, Any], pipelines: dict[str, Any]) -> list[str]:
    """Get running card IDs from runtime summary or derive from pipeline entries."""
    runtime_summary = state.get("runtime_summary", {}) or {}
    runtime_ids = runtime_summary.get("running_card_ids", []) or []
    normalized: list[str] = []
    for raw_card_id in runtime_ids:
        card_id = str(raw_card_id).strip()
        if not card_id:
            continue
        entry = get_pipeline(pipelines, card_id)
        if not entry:
            continue
        pipeline_state = str(entry.get("pipeline_run_state") or entry.get("pipeline_state") or "").strip().lower()
        local_progress = ((entry.get("project") or {}).get("local_progress")) or {}
        if pipeline_state == "completed" or local_progress.get("completed_signal"):
            continue
        if not local_progress.get("running_signal"):
            continue
        normalized.append(card_id)
    if normalized:
        return normalized
    return collect_running_card_ids(pipelines)


def collect_lipsync_video_status_rows(
    pipelines: dict[str, Any],
    card_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect per-project local lipsync video status rows."""
    card_id_filter: set[str] | None
    if card_ids is None:
        card_id_filter = None
    else:
        card_id_filter = {str(card_id).strip().lower() for card_id in card_ids if str(card_id).strip()}
    rows: list[dict[str, Any]] = []
    for entry in sorted(
        (pipelines or {}).values(),
        key=lambda item: str(item.get("card_id") or "").lower(),
    ):
        card_id = str(entry.get("card_id") or "").strip()
        if not card_id:
            continue
        if card_id_filter is not None and card_id.lower() not in card_id_filter:
            continue

        channel = str(((entry.get("project") or {}).get("channel")) or "Unknown")
        account = ", ".join(resolve_card_accounts(entry)) or "Unknown"
        workflow_state = str(entry.get("workflow_state") or "other")
        checks = (entry.get("start_checks") or {})
        tracked_lipsync = ((entry.get("lipsync") or {}).get("tracking_snapshot")) or {}
        local_dirs = ((entry.get("project") or {}).get("local_project_dirs", []) or [])

        for project_dir_raw in sorted({str(path) for path in local_dirs}, key=lambda value: value.lower()):
            project_dir = Path(project_dir_raw)
            files = collect_project_lipsync_video_files(project_dir)
            local_video_status = "present" if files else "missing"
            rows.append(
                {
                    "card_id": card_id,
                    "channel": channel,
                    "account": account,
                    "workflow_state": workflow_state,
                    "project_name": project_dir.name,
                    "project_dir": str(project_dir),
                    "local_video_status": local_video_status,
                    "local_video_file_count": len(files),
                    "local_video_files": files,
                    "lipsync_submitted": bool(checks.get("lipsync_submitted", False)),
                    "tracking_drive_status": str(tracked_lipsync.get("drive_status") or "unknown"),
                    "tracking_local_status": str(tracked_lipsync.get("local_status") or "unknown"),
                }
            )

    rows.sort(
        key=lambda row: (
            row.get("local_video_status") != "missing",
            str(row.get("card_id") or "").lower(),
            str(row.get("project_name") or "").lower(),
        )
    )
    return rows


def summarize_lipsync_video_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize local lipsync video status counts."""
    total = len(rows)
    missing = sum(1 for row in rows if str(row.get("local_video_status")) == "missing")
    present = total - missing
    return {
        "total_local_projects": total,
        "with_lipsync_video": present,
        "missing_lipsync_video": missing,
    }


def format_due_date(value: Any) -> str:
    """Format Trello due timestamps into a compact date label."""
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def format_age_brief(age_seconds: Any) -> str:
    """Render an age value in compact seconds/minutes/hours/days."""
    if not isinstance(age_seconds, (int, float)):
        return ""
    seconds = max(0, int(age_seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def summarize_card_lipsync_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-project lipsync scan rows into a single project-level status."""
    if not rows:
        return {
            "local_project_count": 0,
            "local_video_status": "no_local_project",
            "local_video_file_count": 0,
            "local_projects_with_video": 0,
            "local_projects_missing_video": 0,
            "tracking_drive_status": "unknown",
            "tracking_local_status": "unknown",
            "project_names": [],
            "project_dirs": [],
        }

    present = sum(1 for row in rows if str(row.get("local_video_status") or "") == "present")
    missing = sum(1 for row in rows if str(row.get("local_video_status") or "") == "missing")
    if present and missing:
        status = "mixed"
    elif present:
        status = "present"
    else:
        status = "missing"

    drive_statuses = sorted({str(row.get("tracking_drive_status") or "unknown") for row in rows})
    local_statuses = sorted({str(row.get("tracking_local_status") or "unknown") for row in rows})

    return {
        "local_project_count": len(rows),
        "local_video_status": status,
        "local_video_file_count": sum(int(row.get("local_video_file_count") or 0) for row in rows),
        "local_projects_with_video": present,
        "local_projects_missing_video": missing,
        "tracking_drive_status": drive_statuses[0] if len(drive_statuses) == 1 else "mixed",
        "tracking_local_status": local_statuses[0] if len(local_statuses) == 1 else "mixed",
        "project_names": [str(row.get("project_name") or "") for row in rows if str(row.get("project_name") or "")],
        "project_dirs": [str(row.get("project_dir") or "") for row in rows if str(row.get("project_dir") or "")],
    }


def build_project_status_report(
    entry: dict[str, Any],
    card_lipsync_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a concise project-status payload for reporting and JSON output."""
    if not isinstance(entry, dict):
        entry = {}

    project = entry.get("project", {}) if isinstance(entry.get("project"), dict) else {}
    trello = entry.get("trello", {}) if isinstance(entry.get("trello"), dict) else {}
    list_payload = trello.get("list", {}) if isinstance(trello.get("list"), dict) else {}
    card_raw = trello.get("card_raw", {}) if isinstance(trello.get("card_raw"), dict) else {}
    checks = entry.get("start_checks", {}) if isinstance(entry.get("start_checks"), dict) else {}
    local_progress = project.get("local_progress", {}) if isinstance(project.get("local_progress"), dict) else {}
    local_dirs = [str(path) for path in (project.get("local_project_dirs", []) or []) if str(path).strip()]
    local_project_name = Path(local_dirs[0]).name if local_dirs else ""
    accounts = resolve_card_accounts(entry)
    account_label = ", ".join(accounts) if accounts else "Unknown"
    lipsync_summary = summarize_card_lipsync_rows(card_lipsync_rows or [])

    return {
        "card_id": str(entry.get("card_id") or ""),
        "title": str(entry.get("title") or "Untitled card"),
        "card_url": resolve_card_url(entry, str(entry.get("card_id") or "")),
        "account": account_label,
        "accounts": accounts,
        "channel": str(project.get("channel") or "Unknown"),
        "lipsync_channel": str(project.get("lipsync_channel") or project.get("channel") or "Unknown"),
        "trello_list": str(list_payload.get("name") or "Unknown"),
        "workflow_state": str(entry.get("workflow_state") or list_payload.get("bucket") or "other"),
        "pipeline_state": str(entry.get("pipeline_run_state") or entry.get("pipeline_state") or "unknown"),
        "runtime_state": str(entry.get("pipeline_runtime_state") or "idle"),
        "project_state": str(entry.get("project_state") or "project_unknown"),
        "needs_submission_workflow": bool(entry.get("needs_submission_workflow", False)),
        "next_action": submission_next_action(str(entry.get("workflow_state") or list_payload.get("bucket") or "other")),
        "due_at": str(card_raw.get("due") or ""),
        "due_date": format_due_date(card_raw.get("due")),
        "local_project_count": len(local_dirs),
        "local_project_dirs": local_dirs,
        "local_project_name": local_project_name,
        "raw_voiceover_present": bool(checks.get("has_raw_voiceover", False)),
        "raw_voiceover_candidate_count": len(checks.get("raw_voiceover_candidates", []) or []),
        "lipsync_submitted": bool(checks.get("lipsync_submitted", False)),
        "lipsync_downloaded": bool(checks.get("lipsync_downloaded", False)),
        "lipsync_nonblocking_statuses": list(checks.get("nonblocking_statuses", []) or []),
        "lipsync_local_video_status": lipsync_summary["local_video_status"],
        "lipsync_local_video_file_count": lipsync_summary["local_video_file_count"],
        "lipsync_local_project_count": lipsync_summary["local_project_count"],
        "lipsync_local_projects_with_video": lipsync_summary["local_projects_with_video"],
        "lipsync_local_projects_missing_video": lipsync_summary["local_projects_missing_video"],
        "lipsync_tracking_drive_status": lipsync_summary["tracking_drive_status"],
        "lipsync_tracking_local_status": lipsync_summary["tracking_local_status"],
        "latest_log_age_seconds": local_progress.get("latest_log_age_seconds"),
        "latest_log_age_brief": format_age_brief(local_progress.get("latest_log_age_seconds")),
        "recent_log_activity": bool(local_progress.get("recent_log_activity", False)),
        "completed_signal": bool(local_progress.get("completed_signal", False)),
        "completion_source": str(local_progress.get("completion_source") or ""),
        "output_completion_signal": bool(local_progress.get("output_completion_signal", False)),
    }


def collect_project_status_reports(
    pipelines: dict[str, Any],
    card_ids: list[str] | None = None,
    lipsync_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build ordered project-status report rows for queue/report output."""
    rows_by_card: dict[str, list[dict[str, Any]]] = {}
    for row in lipsync_rows or []:
        key = str(row.get("card_id") or "").strip().lower()
        if not key:
            continue
        rows_by_card.setdefault(key, []).append(row)

    ordered_entries: list[dict[str, Any]] = []
    if card_ids is None:
        ordered_entries = [
            entry
            for entry in sorted((pipelines or {}).values(), key=queue_sort_key)
            if isinstance(entry, dict)
        ]
    else:
        seen: set[str] = set()
        for card_id in card_ids:
            key = str(card_id).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            entry = get_pipeline(pipelines, card_id)
            if entry:
                ordered_entries.append(entry)

    return [
        build_project_status_report(
            entry,
            rows_by_card.get(str(entry.get("card_id") or "").strip().lower(), []),
        )
        for entry in ordered_entries
    ]


def format_channel_route(project_channel: Any, lipsync_channel: Any = "") -> str:
    """Format project/lipsync routing for human-readable status output."""
    project_text = str(project_channel or "").strip() or "Unknown"
    lipsync_text = str(lipsync_channel or "").strip()
    if lipsync_text and lipsync_text != project_text:
        return f"{project_text}->{lipsync_text}"
    return project_text


def summarize_project_lipsync_status(reports: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize project-level lipsync status across report rows."""
    counter: Counter[str] = Counter(
        str(report.get("lipsync_local_video_status") or "unknown")
        for report in reports
    )
    return {
        "total_projects": len(reports),
        "present": int(counter.get("present", 0)),
        "missing": int(counter.get("missing", 0)),
        "mixed": int(counter.get("mixed", 0)),
        "no_local_project": int(counter.get("no_local_project", 0)),
        "unknown": int(counter.get("unknown", 0)),
    }


def resolve_lipsync_scope_card_ids(
    queue: dict[str, Any],
    submission_ids: list[str],
    scope: str,
) -> list[str] | None:
    """Resolve which card IDs should be included in lipsync status reporting."""
    submission_ids_local = list(submission_ids or [])
    not_started_ids = list(queue.get("pending_not_started_card_ids", []) or [])

    if scope == "all":
        return None
    if scope == "submission":
        return submission_ids_local
    if scope == "not_started":
        return not_started_ids
    if scope == "submission_and_not_started":
        ordered: list[str] = []
        seen: set[str] = set()
        for card_id in submission_ids_local + not_started_ids:
            key = str(card_id).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(str(card_id))
        return ordered
    return submission_ids_local


def resolve_lipsync_submission_candidate_ids(state: dict[str, Any]) -> list[str]:
    """Resolve ordered card IDs that are eligible for lipsync submission suggestion."""
    queue = state.get("queue", {}) or {}
    submission_queue = state.get("submission_queue", {}) or {}
    pipelines = state.get("pipelines", {}) or {}

    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in (
        queue.get("ready_card_ids", []) or [],
        queue.get("pending_not_started_card_ids", []) or [],
        queue.get("blocked_card_ids", []) or [],
        submission_queue.get("needs_submission_card_ids", []) or [],
    ):
        for raw_card_id in bucket:
            card_id = str(raw_card_id or "").strip()
            key = card_id.lower()
            if not card_id or key in seen:
                continue
            seen.add(key)
            ordered.append(card_id)

    return sorted(
        ordered,
        key=lambda card_id: queue_sort_key(get_pipeline(pipelines, card_id)),
    )


def is_lipsync_submission_candidate(report: dict[str, Any]) -> bool:
    """Return True when a report still needs a new lipsync generation."""
    workflow_state = str(report.get("workflow_state") or "").strip().lower()
    drive_status = str(report.get("lipsync_tracking_drive_status") or "").strip().lower()
    local_tracking_status = str(report.get("lipsync_tracking_local_status") or "").strip().lower()
    local_video_status = str(report.get("lipsync_local_video_status") or "").strip().lower()

    return bool(
        report.get("raw_voiceover_present")
        and workflow_state in COMPLIANCE_WORKFLOW_STATES
        and not bool(report.get("lipsync_submitted"))
        and not bool(report.get("lipsync_downloaded"))
        and drive_status != "complete"
        and local_tracking_status != "downloaded"
        and local_video_status not in {"present", "mixed"}
    )


def collect_lipsync_submission_candidates(
    state: dict[str, Any],
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Collect queue-backed cards that still need a new lipsync generation."""
    pipelines = state.get("pipelines", {}) or {}
    candidate_ids = resolve_lipsync_submission_candidate_ids(state)
    all_lipsync_rows = collect_lipsync_video_status_rows(pipelines, None)
    reports = collect_project_status_reports(pipelines, candidate_ids, all_lipsync_rows)

    candidates: list[dict[str, Any]] = []
    for report in reports:
        if not is_lipsync_submission_candidate(report):
            continue
        entry = get_pipeline(pipelines, str(report.get("card_id") or ""))
        candidate = dict(report)
        candidate["card_url"] = resolve_card_url(entry, candidate["card_id"])
        candidates.append(candidate)
        if limit > 0 and len(candidates) >= limit:
            break
    return candidates


def select_next_lipsync_submission_candidate(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the next queue-backed card that still needs lipsync submission."""
    candidates = collect_lipsync_submission_candidates(state, limit=1)
    if not candidates:
        return None
    return candidates[0]


def print_queue_counts(queue: dict[str, Any]) -> None:
    """Print pipeline-run counts in table-like form."""
    rows = [
        ("To Run: Not Started", len(queue.get("pending_not_started_card_ids", []))),
        ("To Run: Ready", len(queue.get("ready_card_ids", []))),
        ("To Run: Blocked", len(queue.get("blocked_card_ids", []))),
        ("Run Completed", len(queue.get("completed_card_ids", []))),
        ("Run Not Required", len(queue.get("not_required_card_ids", []))),
        ("Other/Unknown", len(queue.get("not_queued_card_ids", []))),
    ]

    status_width = max(len(name) for name, _count in rows)
    print("\n  Pipeline Run Counts:")
    print(f"  {'Status':<{status_width}}  Count")
    for name, count in rows:
        print(f"  {name:<{status_width}}  {count:>5}")


def print_workflow_counts(workflow_summary: dict[str, int]) -> None:
    """Print Trello workflow-state counts."""
    if not workflow_summary:
        return
    rows = sorted(workflow_summary.items(), key=lambda item: (-int(item[1]), str(item[0])))
    key_width = max(len(name) for name, _count in rows)
    print("\n  Trello Workflow Counts:")
    print(f"  {'Workflow State':<{key_width}}  Count")
    for name, count in rows:
        print(f"  {name:<{key_width}}  {int(count):>5}")


def print_project_state_counts(project_summary: dict[str, int]) -> None:
    """Print project-state counts derived from Trello workflow."""
    if not project_summary:
        return
    rows = sorted(project_summary.items(), key=lambda item: (-int(item[1]), str(item[0])))
    key_width = max(len(name) for name, _count in rows)
    print("\n  Project State Counts:")
    print(f"  {'Project State':<{key_width}}  Count")
    for name, count in rows:
        print(f"  {name:<{key_width}}  {int(count):>5}")


def print_trello_card_status_table(pipelines: dict[str, Any], limit: int) -> None:
    """Print a Trello-card status table with workflow and pipeline states."""
    if not pipelines:
        return

    rows: list[tuple[str, str, str, str, str]] = []
    for entry in sorted(
        (pipelines or {}).values(),
        key=lambda item: str(item.get("card_id") or "").lower(),
    ):
        card_id = str(entry.get("card_id") or "")
        if not card_id:
            continue
        list_name = (((entry.get("trello") or {}).get("list") or {}).get("name")) or "Unknown"
        workflow_state = str(entry.get("workflow_state") or "other")
        pipeline_state = str(entry.get("pipeline_run_state", entry.get("pipeline_state") or "unknown"))
        runtime_state = str(entry.get("pipeline_runtime_state") or "idle")
        rows.append((card_id, list_name, workflow_state, pipeline_state, runtime_state))

    if not rows:
        return

    preview = rows if limit <= 0 else rows[:limit]
    card_w = max(len("Card"), max(len(row[0]) for row in preview))
    list_w = max(len("Trello List"), max(len(row[1]) for row in preview))
    wf_w = max(len("Workflow"), max(len(row[2]) for row in preview))
    run_w = max(len("Pipeline"), max(len(row[3]) for row in preview))
    rt_w = max(len("Runtime"), max(len(row[4]) for row in preview))

    print_header("TRELLO CARD STATUS TABLE")
    print(
        f"  {'Card':<{card_w}}  {'Trello List':<{list_w}}  {'Workflow':<{wf_w}}  "
        f"{'Pipeline':<{run_w}}  {'Runtime':<{rt_w}}"
    )
    for row in preview:
        print(
            f"  {row[0]:<{card_w}}  {row[1]:<{list_w}}  {row[2]:<{wf_w}}  "
            f"{row[3]:<{run_w}}  {row[4]:<{rt_w}}"
        )
    if limit > 0 and len(rows) > limit:
        print(f"  ... showing {limit} of {len(rows)}")


def print_lipsync_video_status_table(rows: list[dict[str, Any]], limit: int = 0) -> None:
    """Print per-project local lipsync video status table."""
    if not rows:
        print_ok("No local project directories found for lipsync status scan")
        return

    preview = rows if limit <= 0 else rows[:limit]

    card_w = max(len("Card"), max(len(str(row.get("card_id") or "")) for row in preview))
    ch_w = max(len("Channel"), max(len(str(row.get("channel") or "")) for row in preview))
    acct_w = max(len("Account"), max(len(str(row.get("account") or "")) for row in preview))
    proj_w = max(
        len("Project"),
        max(len(truncate_text(str(row.get("project_name") or ""), 48)) for row in preview),
    )
    status_w = max(len("Video"), max(len(str(row.get("local_video_status") or "")) for row in preview))

    print_header("LIPSYNC VIDEO STATUS TABLE")
    print(
        f"  {'Card':<{card_w}}  {'Channel':<{ch_w}}  {'Account':<{acct_w}}  "
        f"{'Project':<{proj_w}}  {'Video':<{status_w}}  {'Files':>5}  {'Submitted':<9}"
    )

    for row in preview:
        submitted = "yes" if bool(row.get("lipsync_submitted")) else "no"
        project_label = truncate_text(str(row.get("project_name") or ""), 48)
        print(
            f"  {str(row.get('card_id') or ''):<{card_w}}  {str(row.get('channel') or ''):<{ch_w}}  "
            f"{str(row.get('account') or ''):<{acct_w}}  {project_label:<{proj_w}}  "
            f"{str(row.get('local_video_status') or ''):<{status_w}}  "
            f"{int(row.get('local_video_file_count') or 0):>5}  {submitted:<9}"
        )

    if limit > 0 and len(rows) > limit:
        print(f"  ... showing {limit} of {len(rows)}")

    summary = summarize_lipsync_video_status(rows)
    print_ok(
        "Local project lipsync summary: "
        f"projects={summary['total_local_projects']}, "
        f"with_video={summary['with_lipsync_video']}, "
        f"missing_video={summary['missing_lipsync_video']}"
    )

    missing_rows = [row for row in rows if str(row.get("local_video_status")) == "missing"]
    if missing_rows:
        print_header("PROJECTS MISSING LOCAL LIPSYNC VIDEO")
        for row in missing_rows:
            print(
                f"  - {row.get('card_id')} ({row.get('channel')}) "
                f"{row.get('project_dir')}"
            )


def print_project_status_reports(
    header: str,
    reports: list[dict[str, Any]],
    limit: int,
    *,
    include_next_action: bool = False,
) -> None:
    """Print concise, high-signal project report lines for compact summaries."""
    if not reports:
        return

    print_header(header)
    preview = reports if limit <= 0 else reports[:limit]
    for index, report in enumerate(preview, start=1):
        channel_route = format_channel_route(report.get("channel"), report.get("lipsync_channel"))
        print(f"  {index}. {truncate_text(report.get('title') or 'Untitled card', 88)}")
        print(
            f"     card={report.get('card_id')}  list={report.get('trello_list')}  "
            f"channel={channel_route}  account={report.get('account')}"
        )

        project_bits = [
            f"pipeline={report.get('pipeline_state')}",
            f"runtime={report.get('runtime_state')}",
            f"project_dirs={int(report.get('local_project_count') or 0)}",
            f"raw_voiceover={'yes' if report.get('raw_voiceover_present') else 'no'}",
        ]
        if report.get("local_project_name"):
            project_bits.append(
                f"project={truncate_text(report.get('local_project_name') or '', 40)}"
            )
        if report.get("due_date"):
            project_bits.append(f"due={report.get('due_date')}")
        if report.get("latest_log_age_brief"):
            project_bits.append(f"log_age={report.get('latest_log_age_brief')}")
        if report.get("completed_signal"):
            completion_source = str(report.get("completion_source") or "signal")
            project_bits.append(f"completion={completion_source}")
        print(f"     project: {'  '.join(project_bits)}")

        lipsync_bits = [
            f"submitted={'yes' if report.get('lipsync_submitted') else 'no'}",
            f"downloaded={'yes' if report.get('lipsync_downloaded') else 'no'}",
            f"local_video={report.get('lipsync_local_video_status')}",
            f"files={int(report.get('lipsync_local_video_file_count') or 0)}",
            f"drive={report.get('lipsync_tracking_drive_status')}",
            f"local={report.get('lipsync_tracking_local_status')}",
        ]
        notes = report.get("lipsync_nonblocking_statuses") or []
        if notes:
            lipsync_bits.append(f"notes={','.join(str(note) for note in notes)}")
        print(f"     lipsync: {'  '.join(lipsync_bits)}")

        if include_next_action:
            print(f"     next: {report.get('next_action')}")

    if limit > 0 and len(reports) > limit:
        print(f"  ... showing {limit} of {len(reports)}")


def print_compact_lipsync_overview(
    scope: str,
    reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    limit: int,
) -> None:
    """Print compact lipsync summary with project-level attention items."""
    report_summary = summarize_project_lipsync_status(reports)
    row_summary = summarize_lipsync_video_status(rows)

    print_header(f"LIPSYNC STATUS ({scope})")
    print_ok(
        "Project-level status: "
        f"present={report_summary['present']}, "
        f"missing={report_summary['missing']}, "
        f"mixed={report_summary['mixed']}, "
        f"no_local_project={report_summary['no_local_project']}"
    )
    print_ok(
        "Local project scan: "
        f"projects={row_summary['total_local_projects']}, "
        f"with_video={row_summary['with_lipsync_video']}, "
        f"missing_video={row_summary['missing_lipsync_video']}"
    )

    attention = [
        report
        for report in reports
        if (
            str(report.get("lipsync_local_video_status") or "") != "present"
            or not report.get("lipsync_submitted")
            or not report.get("lipsync_downloaded")
        )
    ]
    if not attention:
        print_ok("No project-level lipsync issues in this scope")
        return

    print_header("LIPSYNC ATTENTION ITEMS")
    preview = attention if limit <= 0 else attention[:limit]
    for report in preview:
        channel_route = format_channel_route(report.get("channel"), report.get("lipsync_channel"))
        print(
            f"  - {report.get('card_id')}: {truncate_text(report.get('title') or 'Untitled card', 72)}"
        )
        print(
            f"    list={report.get('trello_list')}  channel={channel_route}  "
            f"project={truncate_text(report.get('local_project_name') or 'none', 40)}  "
            f"due={report.get('due_date') or 'none'}"
        )
        print(
            f"    submitted={'yes' if report.get('lipsync_submitted') else 'no'}  "
            f"downloaded={'yes' if report.get('lipsync_downloaded') else 'no'}  "
            f"local_video={report.get('lipsync_local_video_status')}  "
            f"drive={report.get('lipsync_tracking_drive_status')}  "
            f"local={report.get('lipsync_tracking_local_status')}"
        )

    if limit > 0 and len(attention) > limit:
        print(f"  ... showing {limit} of {len(attention)}")


def print_pipeline_lines(
    header: str,
    card_ids: list[str],
    pipelines: dict[str, Any],
    limit: int,
    show_urls: bool,
    include_checks: bool = False,
) -> None:
    """Print pipeline preview lines for a queue bucket."""
    if not card_ids:
        return

    print_header(header)
    for index, card_id in enumerate(iter_preview(card_ids, limit), start=1):
        entry = get_pipeline(pipelines, card_id)
        title = truncate_text(entry.get("title") or "Untitled card")
        list_name = (((entry.get("trello") or {}).get("list") or {}).get("name")) or "Unknown"
        project = entry.get("project") or {}
        channel = format_channel_route(project.get("channel"), project.get("lipsync_channel"))
        account = ", ".join(resolve_card_accounts(entry)) or "Unknown"

        print(f"  {index}. {title}")
        print(f"     card={card_id}  list={list_name}  channel={channel}  account={account}")
        local_progress = ((entry.get("project") or {}).get("local_progress")) or {}
        runtime_state = entry.get("pipeline_runtime_state")
        if not runtime_state:
            runtime_state = "running" if local_progress.get("running_signal") else "idle"
        print(
            "     states: "
            f"pipeline_run={entry.get('pipeline_run_state', entry.get('pipeline_state'))}  "
            f"runtime={runtime_state}  "
            f"workflow={entry.get('workflow_state', (((entry.get('trello') or {}).get('list') or {}).get('bucket')))}  "
            f"project={entry.get('project_state', 'project_unknown')}  "
            f"needs_submission={bool(entry.get('needs_submission_workflow', False))}"
        )
        if local_progress.get("latest_log_age_seconds") is not None:
            print(
                "     runtime_detail: "
                f"recent_log_activity={bool(local_progress.get('recent_log_activity'))}  "
                f"latest_log_age_seconds={int(local_progress.get('latest_log_age_seconds') or 0)}"
            )

        blockers = (entry.get("start_checks") or {}).get("blockers", []) or []
        if include_checks and blockers:
            print(f"     blockers: {', '.join(blockers)}")

        if include_checks:
            checks = entry.get("start_checks", {}) or {}
            nonblocking_statuses = checks.get("nonblocking_statuses", []) or []
            print(
                "     checks: "
                f"editing={checks.get('is_editing_list', False)}  "
                f"raw_voiceover={checks.get('has_raw_voiceover', False)}  "
                f"lipsync_submitted={checks.get('lipsync_submitted', False)}  "
                f"lipsync_downloaded={checks.get('lipsync_downloaded', False)}"
            )
            if nonblocking_statuses:
                print(f"     nonblocking: {', '.join(nonblocking_statuses)}")

        local_dirs = (entry.get("project") or {}).get("local_project_dirs", []) or []
        if local_dirs:
            preview_dirs = iter_preview(local_dirs, 2)
            suffix = " ..." if len(local_dirs) > len(preview_dirs) else ""
            print(f"     local_dirs: {', '.join(preview_dirs)}{suffix}")

        if show_urls:
            card_url = (((entry.get("trello") or {}).get("card_raw") or {}).get("shortUrl")) or ""
            if card_url:
                print(f"     trello: {card_url}")

    if limit > 0 and len(card_ids) > limit:
        print(f"  ... showing {limit} of {len(card_ids)}")


def print_blocker_rollup(blocked_ids: list[str], pipelines: dict[str, Any], limit: int) -> None:
    """Print blocker frequency and actionable next steps."""
    if not blocked_ids:
        return

    blocker_counts: Counter[str] = Counter()
    blocker_cards: dict[str, list[str]] = {}
    for card_id in blocked_ids:
        entry = get_pipeline(pipelines, card_id)
        blockers = (entry.get("start_checks", {}) or {}).get("blockers", []) or []
        for blocker in blockers:
            blocker_counts[blocker] += 1
            blocker_cards.setdefault(blocker, []).append(card_id)

    if not blocker_counts:
        return

    print_header("BLOCKER SUMMARY")
    for blocker, count in blocker_counts.most_common():
        hint = BLOCKER_ACTION_HINTS.get(blocker, "Resolve this blocker")
        cards = blocker_cards.get(blocker, [])
        preview_cards = iter_preview(cards, limit)
        suffix = " ..." if limit > 0 and len(cards) > len(preview_cards) else ""
        print(f"  - {blocker}: {count}")
        print(f"    action: {hint}")
        print(f"    cards: {', '.join(preview_cards)}{suffix}")

    print_header("SUGGESTED NEXT ACTIONS")
    for index, (blocker, count) in enumerate(blocker_counts.most_common(), start=1):
        hint = BLOCKER_ACTION_HINTS.get(blocker, "Resolve blocker")
        print(f"  {index}. {hint} ({count} pipeline(s))")


def print_submission_rollup(submission_ids: list[str], pipelines: dict[str, Any], limit: int) -> None:
    """Print submission/review workflow breakdown and suggested actions."""
    if not submission_ids:
        return

    state_counts: Counter[str] = Counter()
    state_cards: dict[str, list[str]] = {}
    for card_id in submission_ids:
        entry = get_pipeline(pipelines, card_id)
        workflow_state = str(entry.get("workflow_state") or "other")
        state_counts[workflow_state] += 1
        state_cards.setdefault(workflow_state, []).append(card_id)

    print_header("SUBMISSION/REVIEW WORKFLOW BREAKDOWN")
    for workflow_state, count in state_counts.most_common():
        hint = submission_next_action(workflow_state)
        cards = state_cards.get(workflow_state, [])
        preview_cards = iter_preview(cards, limit)
        suffix = " ..." if limit > 0 and len(cards) > len(preview_cards) else ""
        print(f"  - {workflow_state}: {count}")
        print(f"    next_action: {hint}")
        print(f"    cards: {', '.join(preview_cards)}{suffix}")


def print_completion_evidence_preview(completed_ids: list[str], pipelines: dict[str, Any], limit: int) -> None:
    """Print completion-evidence preview for completed pipeline runs."""
    if not completed_ids:
        return

    print_header("COMPLETED RUN EVIDENCE")
    for index, card_id in enumerate(iter_preview(completed_ids, limit), start=1):
        entry = get_pipeline(pipelines, card_id)
        title = truncate_text(entry.get("title") or "Untitled card")
        local_progress = ((entry.get("project") or {}).get("local_progress")) or {}
        completion_log = local_progress.get("completion_log")
        print(f"  {index}. {title}")
        print(f"     card={card_id}  completed_signal={bool(local_progress.get('completed_signal'))}")
        if completion_log:
            print(f"     completion_log: {completion_log}")

    if limit > 0 and len(completed_ids) > limit:
        print(f"  ... showing {limit} of {len(completed_ids)}")


def command_kill_pipeline(args: argparse.Namespace) -> int:
    """Kill running pipeline processes and signal autorun to restart.

    Finds main.py processes, optionally filtered by card ID, kills them,
    syncs queue state, and creates a force-cycle signal for autorun.
    """
    import signal as _signal

    psutil_module = _get_psutil_module()
    if psutil_module is None:
        print_error("psutil not available — cannot find pipeline processes")
        return 1

    print_header("KILL RUNNING PIPELINES")

    state = load_state_or_error(Path(args.state_file))
    pipelines = state.get("pipelines", {})

    # Build normalized_path -> card_id mapping
    path_to_card: dict[str, str] = {}
    for _cid, pdata in pipelines.items():
        proj = pdata.get("project", {})
        for d in proj.get("local_project_dirs", []):
            norm = normalize_path_for_compare(d)
            if norm:
                path_to_card[norm] = _cid

    # Get target card IDs (if specified)
    target_cards: list[str] = args.card_id or []

    # Find all running main.py processes
    process_index = list_active_pipeline_processes()
    if process_index is None:
        print_warn("Could not enumerate processes")
        return 1

    if not process_index:
        print_ok("No running pipeline processes found")
        # Still sync + force-cycle in case state is stale
        if not args.no_sync:
            _do_sync_and_force_cycle(args)
        return 0

    # Match processes to card IDs
    killed_pids: list[int] = []
    killed_cards: list[str] = []

    for norm_path, pids in process_index.items():
        matched_card = path_to_card.get(norm_path)

        # Skip if filtering by card and this doesn't match
        if target_cards and matched_card not in target_cards:
            continue

        for pid in pids:
            try:
                proc = psutil_module.Process(pid)
                proc_name = " ".join(proc.cmdline()[-3:]) if proc.cmdline() else str(pid)
                if args.dry_run:
                    print_info(f"Would kill PID {pid} (card={matched_card or '?'}) {proc_name}")
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil_module.TimeoutExpired:
                        proc.kill()
                    print_ok(f"Killed PID {pid} (card={matched_card or '?'})")
                killed_pids.append(pid)
                if matched_card and matched_card not in killed_cards:
                    killed_cards.append(matched_card)
            except (psutil_module.NoSuchProcess, psutil_module.AccessDenied) as e:
                print_warn(f"Could not kill PID {pid}: {e}")

    if not killed_pids:
        if target_cards:
            print_warn(f"No running pipelines found for cards: {', '.join(target_cards)}")
        else:
            print_ok("No running pipeline processes found")
    else:
        action = "Would kill" if args.dry_run else "Killed"
        print_ok(f"{action} {len(killed_pids)} process(es) for cards: {', '.join(killed_cards) or '?'}")

    # Sync queue state and signal autorun
    if not args.dry_run and not args.no_sync:
        _do_sync_and_force_cycle(args)

    return 0


def _do_sync_and_force_cycle(args: argparse.Namespace) -> None:
    """Sync queue state and create force-cycle signal for autorun."""
    print_info("Syncing queue state...")
    try:
        state = sync_state(Path(args.state_file), refresh_lipsync=False)
        running_count = state.get("runtime_summary", {}).get("running_count", 0)
        print_ok(f"Queue synced (running={running_count})")
    except Exception as e:
        print_warn(f"Sync failed: {e}")

    force_cycle_path = Path(args.state_file).parent / "degold_autorun.force_cycle"
    force_cycle_path.write_text("force", encoding="utf-8")
    print_ok(f"Force-cycle signal created: autorun will restart immediately")


def command_show(args: argparse.Namespace) -> int:
    """Handle show subcommand."""
    state = load_state_or_error(Path(args.state_file))
    queue = state.get("queue", {})
    pipelines = state.get("pipelines", {})
    ready_ids = queue.get("ready_card_ids", []) or []
    blocked_ids = queue.get("blocked_card_ids", []) or []
    pending_ids = queue.get("pending_not_started_card_ids", []) or []
    completed_ids = queue.get("completed_card_ids", []) or []
    not_required_ids = queue.get("not_required_card_ids", []) or []
    not_queued_ids = queue.get("not_queued_card_ids", []) or []
    running_ids = resolve_running_card_ids(state, pipelines)
    runtime_summary = state.get("runtime_summary", {}) or {}
    submission_queue = state.get("submission_queue", {}) or {}
    submission_cards = submission_queue.get("needs_submission_cards", []) or []
    submission_ids = submission_queue.get("needs_submission_card_ids", []) or []
    lipsync_scope = str(getattr(args, "lipsync_scope", "submission") or "submission")
    scope_card_ids = resolve_lipsync_scope_card_ids(queue, submission_ids, lipsync_scope)
    all_lipsync_rows = collect_lipsync_video_status_rows(pipelines, None)
    lipsync_rows = collect_lipsync_video_status_rows(pipelines, scope_card_ids)
    lipsync_summary = summarize_lipsync_video_status(lipsync_rows)
    running_reports = collect_project_status_reports(pipelines, running_ids, all_lipsync_rows)
    pending_reports = collect_project_status_reports(pipelines, pending_ids, all_lipsync_rows)
    ready_reports = collect_project_status_reports(pipelines, ready_ids, all_lipsync_rows)
    blocked_reports = collect_project_status_reports(pipelines, blocked_ids, all_lipsync_rows)
    completed_reports = collect_project_status_reports(pipelines, completed_ids, all_lipsync_rows)
    not_required_reports = collect_project_status_reports(pipelines, not_required_ids, all_lipsync_rows)
    submission_reports = collect_project_status_reports(pipelines, submission_ids, all_lipsync_rows)
    lipsync_scope_reports = collect_project_status_reports(pipelines, scope_card_ids, all_lipsync_rows)
    project_reports_payload = {
        "running": running_reports,
        "not_started": pending_reports,
        "ready": ready_reports,
        "blocked": blocked_reports,
        "completed": completed_reports,
        "not_required": not_required_reports,
        "submission": submission_reports,
        "lipsync_scope": lipsync_scope_reports,
        "lipsync_scope_summary": summarize_project_lipsync_status(lipsync_scope_reports),
    }

    if args.json:
        payload = {
            "summary": state.get("summary", {}),
            "queue": state.get("queue", {}),
            "runtime_summary": {
                "is_any_pipeline_running": bool(running_ids),
                "running_count": len(running_ids),
                "running_card_ids": running_ids,
                "running_window_seconds": runtime_summary.get(
                    "running_window_seconds",
                    RUNNING_LOG_ACTIVITY_WINDOW_SECONDS,
                ),
            },
            "submission_queue": submission_queue,
            "submission_preview": submission_cards[:10],
            "lipsync_video_status": {
                "scope": lipsync_scope,
                "scope_card_ids": scope_card_ids if scope_card_ids is not None else [],
                "summary": lipsync_summary,
                "rows": lipsync_rows,
            },
            "project_reports": project_reports_payload,
            "workflow_summary": state.get("workflow_summary", {}),
            "project_state_summary": state.get("project_state_summary", {}),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    workflow_summary = state.get("workflow_summary", {}) or {}
    project_state_summary = state.get("project_state_summary", {}) or {}
    if args.compact:
        print_header("PIPELINE + SUBMISSION/REVIEW SUMMARY")
        print_ok(f"Pipelines to run (not started): {len(queue.get('pending_not_started_card_ids', []))}")
        print_ok(f"Pipelines to run (ready): {len(queue.get('ready_card_ids', []))}")
        print_ok(f"Pipelines to run (blocked): {len(queue.get('blocked_card_ids', []))}")
        print_ok(f"Pipeline runs completed locally: {len(queue.get('completed_card_ids', []))}")
        print_ok(f"Pipeline not required by workflow: {len(queue.get('not_required_card_ids', []))}")
        print_ok(
            "Pipeline currently running: "
            f"{'yes' if running_ids else 'no'} (count={len(running_ids)})"
        )
        print_ok(f"Cards needing submission/review workflow: {len(submission_ids)}")
        print_ok(
            f"Lipsync local videos ({lipsync_scope}): "
            f"projects={lipsync_summary['total_local_projects']}, "
            f"with_video={lipsync_summary['with_lipsync_video']}, "
            f"missing_video={lipsync_summary['missing_lipsync_video']}"
        )
        if pending_reports:
            print_project_status_reports("NOT-STARTED PIPELINES", pending_reports, args.limit)
        if ready_reports:
            print_project_status_reports("READY PIPELINES", ready_reports, args.limit)
        if blocked_reports:
            print_project_status_reports("BLOCKED PIPELINES", blocked_reports, args.limit)
        if submission_reports:
            print_project_status_reports(
                "CARDS NEEDING SUBMISSION OR REVIEW",
                submission_reports,
                args.limit,
                include_next_action=True,
            )
        print_compact_lipsync_overview(
            lipsync_scope,
            lipsync_scope_reports,
            lipsync_rows,
            args.limit,
        )
        if not ready_reports and not submission_reports:
            print_trello_card_status_table(pipelines, args.limit)
        return 0

    print_header("PIPELINE AND TRELLO STATUS")
    print_ok(f"Generated at (UTC): {state.get('generated_at', 'unknown')}")
    print_ok(f"State file: {Path(args.state_file)}")
    print_ok(f"Total pipelines tracked: {state.get('summary', {}).get('total_pipelines', len(pipelines))}")
    print_ok(
        "Pipeline currently running: "
        f"{'yes' if running_ids else 'no'} (count={len(running_ids)})"
    )
    print_queue_counts(queue)
    print_workflow_counts(workflow_summary)
    print_project_state_counts(project_state_summary)
    print_trello_card_status_table(pipelines, args.limit)
    if scope_card_ids is None or scope_card_ids:
        print_lipsync_video_status_table(lipsync_rows, limit=0)
    else:
        print_ok(f"No cards in lipsync scope '{lipsync_scope}' to evaluate")

    actionable_run_ids = pending_ids + ready_ids + blocked_ids

    if actionable_run_ids:
        print_header("PIPELINES THAT MUST RUN")
    else:
        print_ok("No pipelines currently require a run")
        print_info(
            f"All tracked pipelines are either completed ({len(completed_ids)}) "
            f"or not required by workflow ({len(not_required_ids)})."
        )

    if running_ids:
        print_pipeline_lines(
            header="PIPELINES CURRENTLY RUNNING",
            card_ids=running_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
        )

    if pending_ids:
        print_pipeline_lines(
            header="RUN QUEUE: NOT STARTED",
            card_ids=pending_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
            include_checks=True,
        )

    if ready_ids:
        print_pipeline_lines(
            header="RUN QUEUE: READY",
            card_ids=ready_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
        )
    elif actionable_run_ids:
        print_warn("No ready pipelines found")

    if blocked_ids:
        print_pipeline_lines(
            header="RUN QUEUE: BLOCKED",
            card_ids=blocked_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
            include_checks=True,
        )
        print_blocker_rollup(blocked_ids, pipelines, args.limit)
    elif actionable_run_ids:
        print_ok("No blocked pipelines")

    if submission_ids:
        print_pipeline_lines(
            header="CARDS NEEDING SUBMISSION OR REVIEW",
            card_ids=submission_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
        )
        print_submission_rollup(submission_ids, pipelines, args.limit)
    else:
        print_ok("No cards currently need submission/review workflow actions")

    if completed_ids:
        print_completion_evidence_preview(completed_ids, pipelines, min(args.limit, 5) if args.limit > 0 else 5)

    if not_required_ids:
        print_pipeline_lines(
            header="PIPELINE NOT REQUIRED BY CURRENT TRELLO WORKFLOW",
            card_ids=not_required_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
        )

    if not_queued_ids:
        bucket_counts: Counter[str] = Counter()
        list_name_counts: Counter[str] = Counter()
        for card_id in not_queued_ids:
            entry = get_pipeline(pipelines, card_id)
            bucket = (((entry.get("trello") or {}).get("list") or {}).get("bucket")) or "other"
            list_name = (((entry.get("trello") or {}).get("list") or {}).get("name")) or "Unknown"
            bucket_counts[bucket] += 1
            list_name_counts[list_name] += 1
        print_header("NOT QUEUED BREAKDOWN")
        for bucket, count in bucket_counts.most_common():
            print_ok(f"{bucket}: {count}")
        print("  Trello lists:")
        for list_name, count in list_name_counts.most_common():
            print(f"    - {list_name}: {count}")
        print_pipeline_lines(
            header="NOT QUEUED PREVIEW",
            card_ids=not_queued_ids,
            pipelines=pipelines,
            limit=args.limit,
            show_urls=args.show_urls,
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Maintain JSON pipeline queue state from Trello + lipsync tracking",
    )
    parser.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help=f"Path to state JSON (default: {DEFAULT_STATE_FILE})",
    )
    parser.add_argument(
        "--accounts-dir",
        default=None,
        help="Override accounts directory (default: Degold/accounts)",
    )
    parser.add_argument(
        "--board-map-file",
        default=None,
        help="Override board channel map YAML (default: Degold/board_channel_map.yaml)",
    )
    parser.add_argument(
        "--projects-root",
        default=None,
        help="Override local projects root (default: projects/Degold)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--board",
        default=None,
        help=(
            "Board key from config/board_registry.yaml (e.g. stu, degold). "
            "Shorthand that auto-applies --state-file, --accounts-dir, "
            "--board-map-file, and --projects-root for the named board."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Refresh and write the state file")
    sync_parser.add_argument(
        "--refresh-lipsync",
        action="store_true",
        help="Refresh Degold/lipsync_tracking.json before syncing state",
    )
    sync_parser.add_argument("--json", action="store_true", help="Print full synced state JSON")
    sync_parser.set_defaults(func=command_sync)

    archive_parser = subparsers.add_parser(
        "archive-completed",
        help="Move local project dirs for Ready To Upload and later Trello cards into archive storage",
    )
    archive_parser.add_argument(
        "--sync-first",
        action="store_true",
        help="Sync queue state before archiving local project directories",
    )
    archive_parser.add_argument(
        "--refresh-lipsync",
        action="store_true",
        help="Used with --sync-first; refresh lipsync tracking first",
    )
    archive_parser.add_argument(
        "--no-sync-after",
        action="store_true",
        help="Skip queue-state sync after moving project directories",
    )
    archive_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview archive moves without changing the filesystem",
    )
    archive_parser.add_argument("--json", action="store_true", help="Print archive actions as JSON")
    archive_parser.set_defaults(func=command_archive_completed)

    next_parser = subparsers.add_parser("next", help="Get next pipeline that can start now")
    next_parser.add_argument(
        "--sync-first",
        action="store_true",
        help="Sync state before selecting next ready pipeline",
    )
    next_parser.add_argument(
        "--refresh-lipsync",
        action="store_true",
        help="Used with --sync-first; refresh lipsync tracking first",
    )
    next_parser.add_argument("--json", action="store_true", help="Output the pipeline as JSON")
    next_parser.add_argument(
        "--channel",
        action="append",
        help="Limit to pipelines from these channel codes, e.g. STU, RRU, DSR (repeatable)",
    )
    next_parser.set_defaults(func=command_next)

    lipsync_next_parser = subparsers.add_parser(
        "lipsync-next",
        help="Suggest the next queue-backed card that still needs a new lipsync generation",
    )
    lipsync_next_parser.add_argument(
        "--sync-first",
        action="store_true",
        help="Sync state before selecting the next lipsync submission candidate",
    )
    lipsync_next_parser.add_argument(
        "--refresh-lipsync",
        action="store_true",
        help="Used with --sync-first; refresh lipsync tracking first",
    )
    lipsync_next_parser.add_argument(
        "--sync-timeout-seconds",
        type=float,
        default=DEFAULT_LIPSYNC_NEXT_SYNC_TIMEOUT_SECONDS,
        help=(
            "When --sync-first is used, fall back to the cached state file if sync exceeds this many "
            f"seconds (<=0 waits indefinitely; default: {DEFAULT_LIPSYNC_NEXT_SYNC_TIMEOUT_SECONDS:g})"
        ),
    )
    lipsync_next_parser.add_argument("--json", action="store_true", help="Output the candidate as JSON")
    lipsync_next_parser.set_defaults(func=command_lipsync_next)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Create local projects for prep-only and actionable cards that have no prepared project directory",
    )
    prepare_parser.add_argument(
        "--sync-first",
        action="store_true",
        help="Sync queue state before preparing missing projects",
    )
    prepare_parser.add_argument(
        "--refresh-lipsync",
        action="store_true",
        help="Used with --sync-first; refresh lipsync tracking first",
    )
    prepare_parser.add_argument(
        "--card-id",
        action="append",
        help="Specific card ID to prepare (can be provided multiple times)",
    )
    prepare_parser.add_argument(
        "--channel",
        action="append",
        help="Limit preparation to these channel codes, e.g. STU, RRU, DSR (repeatable)",
    )
    prepare_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of cards to prepare (<=0 means all)",
    )
    prepare_parser.add_argument(
        "--run-ready",
        action="store_true",
        help="After preparation, start the next ready pipeline in the background",
    )
    prepare_parser.add_argument("--dry-run", action="store_true", help="Preview targets without creating projects")
    prepare_parser.add_argument("--json", action="store_true", help="Print preparation targets as JSON")
    prepare_parser.set_defaults(func=command_prepare)

    discord_prepare_parser = subparsers.add_parser(
        "discord-prepare",
        help="Scan Discord pipeline-complete messages and create local projects from Drive folders",
    )
    discord_prepare_parser.add_argument(
        "--lookback-hours",
        type=int,
        default=0,
        help="Only consider Discord messages from the last N hours (0 means no time filter; default: 0)",
    )
    discord_prepare_parser.add_argument(
        "--account",
        action="append",
        help="Limit Discord scan to account env names (can be provided multiple times)",
    )
    discord_prepare_parser.add_argument(
        "--channel",
        action="append",
        help="Limit Discord scan to channel codes (RRU/DSR)",
    )
    discord_prepare_parser.add_argument(
        "--exporter",
        default="",
        help="DiscordChatExporter command/path override",
    )
    discord_prepare_parser.add_argument(
        "--export-max-attempts",
        type=int,
        default=DEFAULT_DISCORD_EXPORT_MAX_ATTEMPTS,
        help=(
            "Max retries per Discord export command on rate-limit/transient failures "
            f"(default: {DEFAULT_DISCORD_EXPORT_MAX_ATTEMPTS})"
        ),
    )
    discord_prepare_parser.add_argument(
        "--export-base-backoff-seconds",
        type=float,
        default=DEFAULT_DISCORD_EXPORT_BASE_BACKOFF_SECONDS,
        help=(
            "Base backoff seconds for Discord export retries "
            f"(default: {DEFAULT_DISCORD_EXPORT_BASE_BACKOFF_SECONDS})"
        ),
    )
    discord_prepare_parser.add_argument(
        "--export-max-backoff-seconds",
        type=float,
        default=DEFAULT_DISCORD_EXPORT_MAX_BACKOFF_SECONDS,
        help=(
            "Max backoff cap seconds for Discord export retries "
            f"(default: {DEFAULT_DISCORD_EXPORT_MAX_BACKOFF_SECONDS})"
        ),
    )
    discord_prepare_parser.add_argument(
        "--export-timeout-seconds",
        type=int,
        default=DEFAULT_DISCORD_EXPORT_TIMEOUT_SECONDS,
        help=f"Timeout per Discord export command (default: {DEFAULT_DISCORD_EXPORT_TIMEOUT_SECONDS})",
    )
    discord_prepare_parser.add_argument(
        "--channel-cooldown-seconds",
        type=float,
        default=DEFAULT_DISCORD_CHANNEL_COOLDOWN_SECONDS,
        help=(
            "Delay between channel exports to reduce rate-limit bursts "
            f"(default: {DEFAULT_DISCORD_CHANNEL_COOLDOWN_SECONDS})"
        ),
    )
    discord_prepare_parser.add_argument(
        "--incremental-lookback-minutes",
        type=int,
        default=DEFAULT_DISCORD_INCREMENTAL_LOOKBACK_MINUTES,
        help=(
            "When existing Discord tracker state exists, export only after the latest tracked "
            "message minus this overlap window "
            f"(default: {DEFAULT_DISCORD_INCREMENTAL_LOOKBACK_MINUTES})"
        ),
    )
    discord_prepare_parser.add_argument(
        "--discord-state-file",
        default=str(DEFAULT_DISCORD_STATE_FILE),
        help=f"Path to Discord project tracking JSON (default: {DEFAULT_DISCORD_STATE_FILE})",
    )
    discord_prepare_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of Discord projects to create (<=0 means all pending)",
    )
    discord_prepare_parser.add_argument(
        "--rescan",
        action="store_true",
        help="Ignore tracking state and reprocess already-seen Drive folders",
    )
    discord_prepare_parser.add_argument(
        "--no-sync-after",
        action="store_true",
        help="Skip queue-state sync after creating Discord projects",
    )
    discord_prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview Discord candidates without creating projects",
    )
    discord_prepare_parser.add_argument("--json", action="store_true", help="Print Discord candidates as JSON")
    discord_prepare_parser.set_defaults(func=command_discord_prepare)

    kill_parser = subparsers.add_parser(
        "kill-pipeline",
        aliases=["kill"],
        help="Kill running pipeline processes, sync state, and signal autorun to restart",
    )
    kill_parser.add_argument(
        "--card-id",
        action="append",
        default=None,
        help="Kill only pipelines for these card IDs (repeatable). Kills all if omitted.",
    )
    kill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be killed without killing",
    )
    kill_parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip queue sync and force-cycle signal after killing",
    )
    kill_parser.set_defaults(func=command_kill_pipeline)

    show_parser = subparsers.add_parser(
        "show",
        aliases=["status"],
        help="Show queue summary from state file",
    )
    show_parser.add_argument("--json", action="store_true", help="Output queue buckets as JSON")
    show_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max cards to preview per section (<=0 means show all)",
    )
    show_parser.add_argument(
        "--lipsync-scope",
        choices=["submission", "not_started", "submission_and_not_started", "all"],
        default="submission",
        help=(
            "Scope for lipsync status table: submission (default), not_started, "
            "submission_and_not_started, or all tracked cards"
        ),
    )
    show_parser.add_argument("--show-urls", action="store_true", help="Include Trello card URLs in previews")
    show_parser.add_argument("--compact", action="store_true", help="Show only top-level queue counts")
    show_parser.set_defaults(func=command_show)

    return parser


def _resolve_board_shorthand(args: argparse.Namespace) -> None:
    """Expand ``--board KEY`` into the per-field overrides it represents.

    Reads ``config/board_registry.yaml`` and applies state_file, accounts_dir,
    board_map_file, and projects_root from the named board entry — but only for
    fields the user did *not* already set explicitly.
    """
    if not args.board:
        return

    import yaml

    registry_path = PROJECT_ROOT / "config" / "board_registry.yaml"
    if not registry_path.exists():
        raise SystemExit(f"--board requires config/board_registry.yaml (not found: {registry_path})")

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    boards = raw.get("boards") or {}
    entry = boards.get(args.board)
    if not isinstance(entry, dict):
        available = ", ".join(boards.keys()) if boards else "(none)"
        raise SystemExit(f"Unknown board '{args.board}'. Available: {available}")

    # Only override fields the user didn't pass explicitly
    if args.state_file == str(DEFAULT_STATE_FILE) and entry.get("state_file"):
        args.state_file = str(PROJECT_ROOT / entry["state_file"])
    if not args.accounts_dir and entry.get("accounts_dir"):
        args.accounts_dir = str(PROJECT_ROOT / entry["accounts_dir"])
    if not args.board_map_file and entry.get("board_map_file"):
        args.board_map_file = str(PROJECT_ROOT / entry["board_map_file"])
    if not args.projects_root and entry.get("projects_root"):
        args.projects_root = str(entry["projects_root"])


def _apply_root_overrides(args: argparse.Namespace) -> None:
    """Override module-level defaults from CLI args."""
    global DEFAULT_ACCOUNTS_DIR, DEFAULT_BOARD_MAP_FILE, LOCAL_PROJECTS_ROOT
    global DEFAULT_LIPSYNC_TRACKING_FILE, DEFAULT_DISCORD_STATE_FILE

    # Expand --board shorthand first so subsequent checks see the resolved values
    _resolve_board_shorthand(args)

    if args.accounts_dir:
        DEFAULT_ACCOUNTS_DIR = Path(args.accounts_dir)
    if args.board_map_file:
        DEFAULT_BOARD_MAP_FILE = Path(args.board_map_file)
    if args.projects_root:
        LOCAL_PROJECTS_ROOT = Path(args.projects_root)
    # Derive lipsync/discord state from same parent as state-file when overridden
    state_parent = Path(args.state_file).parent
    if args.accounts_dir or args.board_map_file or args.projects_root:
        DEFAULT_LIPSYNC_TRACKING_FILE = state_parent / "lipsync_tracking.json"
        DEFAULT_DISCORD_STATE_FILE = state_parent / "discord_pipeline_projects.json"


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    _apply_root_overrides(args)
    set_verbosity(2 if args.verbose else 1)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
