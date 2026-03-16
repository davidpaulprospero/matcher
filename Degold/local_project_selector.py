from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CHECKPOINT_FILENAMES = (
    "checkpoint.json",
    "checkpoint.backup.json",
    "checkpoint.backup.1.json",
)


def iter_local_project_roots(
    local_projects_root: Path,
    archived_prefix: str = "_archived_pipeline_projects",
) -> list[Path]:
    """Return first-level local project roots, excluding archive buckets by default."""
    if not local_projects_root.exists() or not local_projects_root.is_dir():
        return []

    roots: list[Path] = []
    archived_prefix_lower = str(archived_prefix or "").strip().lower()
    for child in sorted(local_projects_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        if archived_prefix_lower and child.name.lower().startswith(archived_prefix_lower):
            continue
        roots.append(child)
    return roots


def contains_card_id_token(value: str, match_keys: set[str]) -> bool:
    """Return True when text contains an exact-ish card ID token."""
    text = str(value or "").strip().lower()
    if not text:
        return False

    for match_key in match_keys:
        key = str(match_key or "").strip().lower()
        if not key:
            continue
        if text == key:
            return True
        if text.startswith(f"{key}-") or text.startswith(f"{key}_"):
            return True
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text):
            return True
    return False


def extract_project_card_id(project_dir: Path) -> str:
    """Read a stored Trello card ID from trello_card.json when available."""
    info_file = project_dir / "trello_card.json"
    if not info_file.is_file():
        return ""

    try:
        payload = json.loads(info_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return ""

    for field in ("card_id", "short_link", "shortLink"):
        value = str(payload.get(field) or "").strip()
        if value:
            return value

    for field in ("short_url", "shortUrl", "url"):
        value = str(payload.get(field) or "").strip()
        match = re.search(r"trello\.com/c/([A-Za-z0-9]+)", value)
        if match:
            return match.group(1)
    return ""


def directory_has_any_files(path: Path) -> bool:
    """Return True when a directory contains at least one file or non-empty subdirectory."""
    if not path.exists() or not path.is_dir():
        return False

    try:
        for child in path.iterdir():
            if child.is_file():
                return True
            if child.is_dir():
                try:
                    next(child.iterdir())
                    return True
                except (StopIteration, OSError):
                    continue
    except OSError:
        return False
    return False


def project_dir_quality_score(project_dir: Path) -> int:
    """Score how likely a directory is the active/canonical project copy."""
    score = 0

    if (project_dir / "trello_card.json").is_file():
        score += 8

    if any((project_dir / filename).is_file() for filename in CHECKPOINT_FILENAMES):
        score += 7

    if directory_has_any_files(project_dir / "voiceover"):
        score += 4

    if directory_has_any_files(project_dir / "logs"):
        score += 3

    if directory_has_any_files(project_dir / "output"):
        score += 5

    if directory_has_any_files(project_dir / "lipsync"):
        score += 2

    return score


def is_strong_project_candidate(candidate: dict[str, Any], quality_threshold: int = 10) -> bool:
    """Return True when a candidate has strong evidence it is the active project."""
    return bool(candidate.get("id_match")) or int(candidate.get("quality_score") or 0) >= quality_threshold


def sort_project_match_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort candidate project directories by strength, then stability."""
    return sorted(
        candidates,
        key=lambda item: (
            -int(item.get("quality_score") or 0),
            -int(bool(item.get("channel_match"))),
            -int(bool(item.get("id_match"))),
            -int(bool(item.get("title_match"))),
            str(item.get("path") or "").lower(),
        ),
    )
