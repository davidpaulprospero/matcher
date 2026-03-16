"""Shared Degold channel-routing helpers for Trello projects and lipsync jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CHANNEL_BY_ACCOUNT = {
    "david": "RRU",
    "stuart": "DSR",
    "pamela": "RRU",
}
BOARD_CHANNEL_MAP_FILENAME = "board_channel_map.yaml"
LABEL_CHANNEL_OVERRIDES = {
    "rru": "RRU",
    "rennreports": "RRU",
    "rennreportsus": "RRU",
    "dsr": "DSR",
    "deepseareports": "DSR",
    "jdrp": "JDRP",
    "journalofdrunkpeople": "JDRP",
}


def normalize_channel_code(value: Any, fallback: str = "RRU") -> str:
    """Return an uppercased channel code with a stable fallback."""
    text = str(value or "").strip().upper()
    if text:
        return text
    fallback_text = str(fallback or "RRU").strip().upper()
    return fallback_text or "RRU"


def default_channel_for_account(account_name: str, configured_channel: str = "") -> str:
    """Resolve the account's brand/default channel."""
    fallback = DEFAULT_CHANNEL_BY_ACCOUNT.get(str(account_name or "").strip().lower(), "RRU")
    return normalize_channel_code(configured_channel, fallback)


def normalize_label_key(value: Any) -> str:
    """Normalize a Trello label name for channel alias matching."""
    if isinstance(value, dict):
        value = value.get("name")
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def resolve_channel_override_from_labels(labels: Any) -> str:
    """Resolve a per-card channel override from Trello labels when unambiguous."""
    if isinstance(labels, (str, bytes)):
        values = [labels]
    elif isinstance(labels, (list, tuple, set)):
        values = list(labels)
    else:
        return ""

    matches: list[str] = []
    seen: set[str] = set()
    for value in values:
        channel = LABEL_CHANNEL_OVERRIDES.get(normalize_label_key(value), "")
        if not channel or channel in seen:
            continue
        seen.add(channel)
        matches.append(channel)

    return matches[0] if len(matches) == 1 else ""


def _normalize_board_map_entry(payload: Any) -> dict[str, str]:
    """Normalize one board routing entry from YAML."""
    if isinstance(payload, dict):
        return {
            key: str(value).strip().upper()
            for key, value in payload.items()
            if key in {"channel", "project_channel", "lipsync_channel"} and str(value).strip()
        }
    if isinstance(payload, str):
        value = str(payload).strip().upper()
        return {"channel": value} if value else {}
    return {}


def discover_board_channel_map_paths(
    project_root: str | Path,
    preferred_path: str | Path | None = None,
) -> list[Path]:
    """Return existing board-map files, merging Degold with any top-level overrides."""
    root = Path(project_root).resolve()
    discovered: list[Path] = []
    seen: set[Path] = set()

    def append(candidate: str | Path | None) -> None:
        if candidate is None:
            return
        path = Path(candidate).resolve()
        if path in seen or not path.exists() or not path.is_file():
            return
        seen.add(path)
        discovered.append(path)

    append(preferred_path)
    append(root / "Degold" / BOARD_CHANNEL_MAP_FILENAME)

    try:
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        children = []

    for child in children:
        if not child.is_dir():
            continue
        append(child / BOARD_CHANNEL_MAP_FILENAME)

    return discovered


def load_board_channel_map(
    project_root: str | Path,
    preferred_path: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    """Load merged board routing metadata from Degold plus any top-level overrides."""
    merged: dict[str, dict[str, str]] = {}

    for path in discover_board_channel_map_paths(project_root, preferred_path=preferred_path):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            data = yaml.safe_load(handle) or {}
        for board_id, payload in (data.get("boards") or {}).items():
            entry = _normalize_board_map_entry(payload)
            if entry:
                merged[str(board_id)] = entry

    return merged


def resolve_channels_for_board(
    board_payload: Any,
    account_default_channel: str,
    card_labels: Any = None,
) -> dict[str, str]:
    """
    Resolve per-card routing.

    `project_channel` is where the Trello project/local folder lives.
    `lipsync_channel` is which channel config to use for avatar + Drive output.
    Card-label overrides are treated as more specific than board/account defaults.
    """
    payload: dict[str, Any]
    if isinstance(board_payload, dict):
        payload = board_payload
    elif isinstance(board_payload, str):
        payload = {"channel": board_payload}
    else:
        payload = {}

    project_channel = normalize_channel_code(
        payload.get("project_channel") or payload.get("channel") or account_default_channel,
        account_default_channel,
    )
    lipsync_channel = normalize_channel_code(
        payload.get("lipsync_channel") or account_default_channel or project_channel,
        project_channel,
    )
    label_channel = resolve_channel_override_from_labels(card_labels)
    if label_channel:
        project_channel = label_channel
        lipsync_channel = label_channel
    return {
        "project_channel": project_channel,
        "lipsync_channel": lipsync_channel,
    }
