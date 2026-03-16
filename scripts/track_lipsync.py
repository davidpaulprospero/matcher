#!/usr/bin/env python3
"""
Track lipsync status for cards in editing + edit review.

Generates a JSON file tracking:
1. Cards in editing/edit review
2. Whether lipsync job was submitted
3. Status of lipsync in Google Drive
4. Whether files are downloaded locally
"""

import os
import sys
import json
import html
import re
from difflib import SequenceMatcher
from pathlib import Path
from datetime import datetime
from typing import Optional, Set

import requests

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Add Degold to path for channels import
sys.path.insert(0, str(PROJECT_ROOT / "Degold"))

from script_utils import print_header, print_info, print_ok, print_warn
from channel_routing import (
    default_channel_for_account,
    load_board_channel_map as load_shared_board_channel_map,
    resolve_channels_for_board,
)

LOCAL_PROJECTS_ROOT = Path(r"E:\Edit Job\Degold")
CHANNEL_DIR_ALIASES = {
    "DSR": ("DeepSeaReports", "DSR"),
    "RRU": ("RennReports", "RRU"),
}
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


def resolve_channel_dirs(base_path: Path, channel: str) -> list[Path]:
    """Resolve existing channel directories from channel code/folder aliases."""
    channel_upper = (channel or "").upper()
    candidates = list(CHANNEL_DIR_ALIASES.get(channel_upper, ()))
    if channel:
        candidates.append(channel)

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        candidate_path = base_path / candidate
        if candidate_path.exists():
            resolved.append(candidate_path)
    return resolved


def load_accounts():
    """Load Trello credentials from Degold accounts."""
    accounts = {}

    for account_file, account_name in [
        ("david.env", "David"),
        ("stuart.env", "Stuart"),
        ("pamela.env", "Pamela"),
    ]:
        account_path = PROJECT_ROOT / "Degold" / "accounts" / account_file
        if not account_path.exists():
            continue

        env_vars = {}
        with open(account_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip()

        api_key = env_vars.get("TRELLO_API_KEY")
        token = env_vars.get("TRELLO_TOKEN")

        if api_key and token:
            default_channel = default_channel_for_account(
                account_name,
                str(env_vars.get("DEFAULT_CHANNEL") or ""),
            )
            accounts[account_name] = {
                "api_key": api_key,
                "token": token,
                "default_channel": default_channel,
            }

    return accounts


def get_board_id_to_channel() -> dict[str, dict]:
    """Load board to channel mapping from yaml."""
    return load_shared_board_channel_map(PROJECT_ROOT)


def normalize_member_ids(member_ids: Optional[list[str]]) -> list[str]:
    """Normalize member IDs to non-empty strings."""
    if not isinstance(member_ids, list):
        return []
    normalized: list[str] = []
    for value in member_ids:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def resolve_allowed_member_ids(accounts: dict[str, dict]) -> set[str]:
    """Resolve Trello member IDs for configured accounts."""
    allowed_member_ids: set[str] = set()
    for account in accounts.values():
        try:
            me_resp = requests.get(
                "https://api.trello.com/1/members/me",
                params={"key": account["api_key"], "token": account["token"], "fields": "id"},
                timeout=20,
            )
        except Exception:
            continue
        if me_resp.status_code != 200:
            continue
        me_id = str((me_resp.json() or {}).get("id") or "").strip()
        if me_id:
            allowed_member_ids.add(me_id)
    return allowed_member_ids


def get_cards_in_lists(
    board_id: str,
    api_key: str,
    token: str,
    list_names: list[str],
    allowed_member_ids: Optional[Set[str]] = None,
) -> list[dict]:
    """Get cards in specific lists for the current user."""

    # Get current user ID
    me_resp = requests.get(
        "https://api.trello.com/1/members/me",
        params={"key": api_key, "token": token, "fields": "id"},
        timeout=20,
    )

    if me_resp.status_code != 200:
        return []

    my_id = str((me_resp.json() or {}).get("id") or "").strip()
    if not my_id:
        return []

    # Get list IDs
    lists_resp = requests.get(
        f"https://api.trello.com/1/boards/{board_id}/lists",
        params={"key": api_key, "token": token, "fields": "id,name"},
        timeout=20,
    )

    list_ids = {}
    if lists_resp.status_code == 200:
        for lst in lists_resp.json():
            for name in list_names:
                if lst.get("name", "").lower() == name.lower():
                    list_ids[lst.get("id")] = lst.get("name")

    if not list_ids:
        return []

    # Get cards
    resp = requests.get(
        f"https://api.trello.com/1/boards/{board_id}/cards",
        params={
            "key": api_key,
            "token": token,
            "fields": "id,name,shortUrl,idList,labels",
            "filter": "visible",
        },
        timeout=30,
    )

    if resp.status_code != 200:
        return []

    all_cards = resp.json()

    # Filter by list and membership
    cards = []
    for card in all_cards:
        card_list_id = card.get("idList")
        if card_list_id not in list_ids:
            continue

        # Check if assigned to me
        card_resp = requests.get(
            f"https://api.trello.com/1/cards/{card.get('id')}",
            params={"key": api_key, "token": token, "fields": "idMembers"},
            timeout=20,
        )
        if card_resp.status_code == 200:
            card_data = card_resp.json() or {}
            member_ids = normalize_member_ids(card_data.get("idMembers"))
            if my_id in member_ids:
                if (
                    allowed_member_ids is not None
                    and any(member_id not in allowed_member_ids for member_id in member_ids)
                ):
                    continue
                card["list_name"] = list_ids[card_list_id]
                cards.append(card)

    return cards


def get_user_boards(api_key: str, token: str) -> list[dict]:
    """Get all boards the user is a member of."""
    resp = requests.get(
        "https://api.trello.com/1/members/me/boards",
        params={"key": api_key, "token": token, "fields": "id,name"},
        timeout=20,
    )

    if resp.status_code != 200:
        return []

    return resp.json()


def get_existing_projects() -> set[str]:
    """Get all existing project card IDs from E:\\Edit Job\\Degold."""
    base_path = LOCAL_PROJECTS_ROOT
    if not base_path.exists():
        return set()

    projects = set()
    for channel_dir in base_path.iterdir():
        if not channel_dir.is_dir():
            continue
        for project_dir in channel_dir.iterdir():
            if not project_dir.is_dir():
                continue
            name = project_dir.name
            if "-" in name:
                card_id = name.split("-")[0]
                projects.add(card_id.lower())

    return projects


def extract_mp4_filenames(folder_html: str) -> list[str]:
    """Extract .mp4 filenames from Google Drive folder HTML."""
    # Pattern observed in Drive folder payload:
    # &quot;filename.mp4&quot;,null,true
    # Use a tempered pattern so we stop at the next &quot; and avoid swallowing JSON blobs.
    raw_names = re.findall(r"&quot;((?:(?!&quot;).)+?\.mp4)&quot;,null,true", folder_html, flags=re.IGNORECASE)
    names: list[str] = []
    for raw in raw_names:
        name = html.unescape(raw).strip()
        if name:
            names.append(name)
    return names


def file_matches_card_id(filename: str, card_id: str) -> bool:
    """Return True when filename clearly belongs to a card ID."""
    card_id_norm = (card_id or "").strip().lower()
    if not card_id_norm:
        return False

    filename_norm = (filename or "").strip().lower()
    if not filename_norm:
        return False

    # Prefer strict token match and also allow direct prefix conventions.
    if filename_norm.startswith(card_id_norm):
        return True

    pattern = rf"(?<![a-z0-9]){re.escape(card_id_norm)}(?![a-z0-9])"
    return bool(re.search(pattern, filename_norm))


def normalize_text_for_match(text: str) -> str:
    """Normalize text for fuzzy matching."""
    value = html.unescape(str(text or "")).lower()
    value = value.replace("_", " ")
    value = re.sub(r"\.[a-z0-9]{2,5}$", "", value)  # strip extension-like suffix
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_significant_tokens(text: str) -> set[str]:
    """Extract meaningful tokens for title/file overlap scoring."""
    tokens = set(re.findall(r"[a-z0-9]+", normalize_text_for_match(text)))
    return {tok for tok in tokens if len(tok) >= 4 and tok not in LIPSYNC_TITLE_STOPWORDS}


def score_title_match(filename: str, card_title: str) -> dict:
    """Score how likely a filename belongs to the given card title."""
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


def check_drive_for_lipsync(channel: str, card_title: str, card_id: str = None) -> dict:
    """Check Google Drive for lipsync output by scraping the folder page and subfolders."""
    try:
        from channels import get_channel
        import requests

        config = get_channel(channel)
        if not config:
            return {"status": "unknown", "error": f"Unknown channel: {channel}"}

        folder_id = config.drive_folder

        card_id_norm = (card_id or "").strip().lower()
        has_card_id = bool(card_id_norm)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }

        # Get all subfolder IDs from main folder
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        response = requests.get(folder_url, headers=headers, timeout=15)

        if response.status_code != 200:
            return {"status": "error", "error": f"HTTP {response.status_code}"}

        html = response.text
        folder_ids = list(set(re.findall(r'data-id="([a-zA-Z0-9_-]{33})"', html)))
        folder_ids.append(folder_id)  # Include main folder

        # Check each folder for matching mp4 files
        found_files_card_id = []
        found_files_title = []
        seen_files: set[tuple[str, str]] = set()
        checked_folders = 0

        for fid in folder_ids[:20]:  # Limit to 20 folders
            try:
                url = f"https://drive.google.com/drive/folders/{fid}"
                resp = requests.get(url, headers=headers, timeout=10)
                checked_folders += 1

                if resp.status_code == 200:
                    folder_html = resp.text
                    for mp4_file in extract_mp4_filenames(folder_html):
                        dedupe_key = (mp4_file.lower(), fid)
                        if dedupe_key in seen_files:
                            continue
                        seen_files.add(dedupe_key)

                        file_entry = {
                            "name": mp4_file[:100],
                            "type": "video",
                            "folder_id": fid,
                        }
                        if has_card_id and file_matches_card_id(mp4_file, card_id_norm):
                            file_entry["match_reason"] = "card_id"
                            found_files_card_id.append(file_entry)
                            continue

                        title_score = score_title_match(mp4_file, card_title)
                        if is_high_confidence_title_match(title_score):
                            file_entry["match_reason"] = "title_similarity"
                            file_entry["match_score"] = round(float(title_score.get("score", 0.0)), 3)
                            found_files_title.append(file_entry)
            except:
                continue

        found_files: list[dict] = []
        match_strategy = "card_id" if has_card_id else "title_similarity"
        if found_files_card_id:
            found_files = found_files_card_id
            match_strategy = "card_id"
        elif found_files_title:
            found_files = sorted(found_files_title, key=lambda item: float(item.get("match_score", 0.0)), reverse=True)
            match_strategy = "title_similarity_fallback" if has_card_id else "title_similarity"

        if found_files:
            return {
                "status": "complete",
                "files": found_files[:5],  # Limit to 5 files
                "checked_folders": checked_folders,
                "match_strategy": match_strategy,
            }

        return {
            "status": "not_found",
            "folder_id": folder_id,
            "checked_folders": checked_folders,
            "match_strategy": match_strategy,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_local_lipsync(channel: str, card_id: str, project_title: str) -> dict:
    """Check if lipsync files are downloaded locally."""
    base_path = LOCAL_PROJECTS_ROOT
    channel_dirs = resolve_channel_dirs(base_path, channel)
    if not channel_dirs:
        return {"status": "not_downloaded"}

    for channel_dir in channel_dirs:
        for match in sorted(channel_dir.iterdir()):
            if not match.is_dir():
                continue

            name_lower = match.name.lower()
            id_match = card_id.lower() in name_lower
            title_match = False
            if not id_match:
                title_score = score_title_match(match.name, project_title)
                title_match = is_high_confidence_title_match(title_score)
            if not id_match and not title_match:
                continue

            # Check output folder
            output_dir = match / "output"
            if output_dir.exists():
                videos = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.mov"))
                if videos:
                    return {
                        "status": "downloaded",
                        "path": str(output_dir),
                        "files": [v.name for v in videos],
                    }

            # Check for lipsync specific folder
            lipsync_dir = match / "lipsync"
            if lipsync_dir.exists():
                videos = list(lipsync_dir.glob("*.mp4")) + list(lipsync_dir.glob("*.mov"))
                if videos:
                    return {
                        "status": "downloaded",
                        "path": str(lipsync_dir),
                        "files": [v.name for v in videos],
                    }

            root_videos = [
                video
                for video in list(match.glob("*.mp4")) + list(match.glob("*.mov"))
                if "lipsync" in video.name.lower()
            ]
            if root_videos:
                return {
                    "status": "downloaded",
                    "path": str(match),
                    "files": [v.name for v in root_videos],
                }

    return {"status": "not_downloaded"}


def main():
    print_header("TRACKING LIPSYNC STATUS")

    # Load accounts
    accounts = load_accounts()
    if not accounts:
        print_warn("No Trello accounts configured")
        return

    allowed_member_ids = resolve_allowed_member_ids(accounts)
    if not allowed_member_ids:
        print_warn("Could not resolve configured Trello member IDs; strict assignee filter will exclude all cards")

    # Get board to channel mapping
    board_to_channel = get_board_id_to_channel()

    # Get existing projects
    existing_projects = get_existing_projects()

    # Collect all cards
    all_cards = []
    target_lists = ["editing", "edit review"]

    for account_name, account in accounts.items():
        boards = get_user_boards(account["api_key"], account["token"])

        for board in boards:
            board_id = board.get("id")

            cards = get_cards_in_lists(
                board_id,
                account["api_key"],
                account["token"],
                target_lists,
                allowed_member_ids=allowed_member_ids,
            )

            for card in cards:
                card_url = card.get("shortUrl", "")
                card_id = card_url.split("/")[-1] if card_url else ""
                card_name = card.get("name", "")
                card_labels = [
                    str(label.get("name") or "").strip()
                    for label in (card.get("labels") or [])
                    if isinstance(label, dict) and str(label.get("name") or "").strip()
                ]
                routing = resolve_channels_for_board(
                    board_to_channel.get(board_id, {}),
                    str(account.get("default_channel") or "RRU"),
                    card.get("labels") or [],
                )
                project_channel = routing["project_channel"]
                lipsync_channel = routing["lipsync_channel"]

                all_cards.append({
                    "card_id": card_id,
                    "card_url": card_url,
                    "title": card_name,
                    "list": card.get("list_name", "Unknown"),
                    "account": account_name,
                    "labels": card_labels,
                    "channel": project_channel,
                    "lipsync_channel": lipsync_channel,
                    "has_project": card_id.lower() in existing_projects,
                })

    # Build tracking data
    tracking = {
        "generated_at": datetime.now().isoformat(),
        "cards": []
    }

    for card in all_cards:
        card_id = card["card_id"]
        channel = card["channel"]
        lipsync_channel = card.get("lipsync_channel") or channel
        title = card["title"]

        # Check lipsync status
        drive_status = check_drive_for_lipsync(lipsync_channel, title, card_id)
        local_status = check_local_lipsync(channel, card_id, title)

        card_tracking = {
            **card,
            "lipsync": {
                "drive_status": drive_status.get("status", "unknown"),
                "drive_files": drive_status.get("files", []),
                "drive_match_strategy": drive_status.get("match_strategy"),
                "local_status": local_status.get("status", "unknown"),
                "local_files": local_status.get("files", []),
                "local_path": local_status.get("path"),
            }
        }

        tracking["cards"].append(card_tracking)

        # Print summary
        drive_indicator = "[OK]" if drive_status.get("status") == "complete" else "[--]"
        local_indicator = "[OK]" if local_status.get("status") == "downloaded" else "[--]"
        print(f"  [{card['account']}] Drive: {drive_indicator} | Local: {local_indicator} | {card['title'][:50]}")

    # Save JSON
    output_file = PROJECT_ROOT / "Degold" / "lipsync_tracking.json"
    with open(output_file, "w") as f:
        json.dump(tracking, f, indent=2)

    print_header(f"Tracked {len(tracking['cards'])} cards")
    print_info(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
