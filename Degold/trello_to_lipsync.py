#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trello to Lipsync Integration

Fetches cards from Trello and submits them to the Degold lipsync automator.

Supports multiple Trello accounts via the --account flag.
Account configs are stored in Degold/accounts/ directory.

Usage:
    python trello_to_lipsync.py --list "Ready To Upload" --dry-run
    python trello_to_lipsync.py --list "Editing" --account david
    python trello_to_lipsync.py --list "Ideas" --account john --limit 5
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add Degold directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
import requests


def get_account_config(account_name: str) -> tuple[str, str, str]:
    """
    Load Trello credentials for a specific account.

    Args:
        account_name: Name of the account (matches .env filename)

    Returns:
        Tuple of (api_key, token, default_board_id)
    """
    accounts_dir = Path(__file__).parent / "accounts"
    config_path = accounts_dir / f"{account_name}.env"

    if not config_path.exists():
        # List available accounts
        available = [f.stem for f in accounts_dir.glob("*.env") if f.name != "example.env"]
        print(f"[ERROR] Account '{account_name}' not found")
        print(f"Available accounts: {available if available else 'none'}")
        print(f"Create a new account config in accounts/{account_name}.env")
        sys.exit(1)

    load_dotenv(config_path)

    api_key = os.getenv("TRELLO_API_KEY")
    token = os.getenv("TRELLO_TOKEN")
    board_id = os.getenv("TRELLO_BOARD_ID")

    if not api_key or not token:
        print(f"[ERROR] Invalid config for account '{account_name}'")
        print("TRELLO_API_KEY and TRELLO_TOKEN are required")
        sys.exit(1)

    return api_key, token, board_id


def list_accounts() -> list[str]:
    """List all available accounts."""
    accounts_dir = Path(__file__).parent / "accounts"
    if not accounts_dir.exists():
        return []
    return [f.stem for f in accounts_dir.glob("*.env") if f.name != "example.env"]


def get_boards(api_key: str, token: str):
    """Get all accessible boards."""
    response = requests.get(
        "https://api.trello.com/1/members/me/boards",
        params={"key": api_key, "token": token, "fields": "name,id"}
    )
    response.raise_for_status()
    return response.json()


def get_lists(board_id: str, api_key: str, token: str):
    """Get all lists in a board."""
    response = requests.get(
        f"https://api.trello.com/1/boards/{board_id}/lists",
        params={"key": api_key, "token": token, "fields": "name,id"}
    )
    response.raise_for_status()
    return response.json()


def get_cards(
    api_key: str,
    token: str,
    board_id: str,
    list_name: Optional[str] = None,
    list_id: Optional[str] = None,
    assigned_to_me: bool = False,
    has_due: bool = False,
    due_before: Optional[str] = None,
    due_after: Optional[str] = None,
    limit: Optional[int] = None,
    labels: Optional[list[str]] = None,
) -> list[dict]:
    """
    Get cards from a board with optional filters.

    Args:
        board_id: Trello board ID
        list_name: Filter by list name
        list_id: Filter by list ID (overrides list_name)
        assigned_to_me: Only cards assigned to current user
        has_due: Only cards with due dates
        due_before: ISO date string - cards due before this date
        due_after: ISO date string - cards due after this date
        limit: Maximum number of cards to return
        labels: List of label names to filter by
    """
    # Get lists to find the target list ID
    lists = get_lists(board_id, api_key, token)

    target_list_id = list_id
    if list_name and not target_list_id:
        matching = [l for l in lists if l["name"].lower() == list_name.lower()]
        if matching:
            target_list_id = matching[0]["id"]
        else:
            print(f"[WARN] List '{list_name}' not found. Available lists:")
            for l in lists:
                print(f"  - {l['name']}")
            return []

    # Get cards - either from specific list or entire board
    if target_list_id:
        url = f"https://api.trello.com/1/lists/{target_list_id}/cards"
    else:
        url = f"https://api.trello.com/1/boards/{board_id}/cards"

    params = {
        "key": api_key,
        "token": token,
        "fields": "name,id,idList,due,labels,idMembers,shortUrl,desc",
    }

    if limit:
        params["limit"] = limit

    response = requests.get(url, params=params)
    response.raise_for_status()
    cards = response.json()

    # Get member info to check if assigned to me
    me_response = requests.get(
        "https://api.trello.com/1/members/me",
        params={"key": api_key, "token": token, "fields": "id"}
    )
    my_id = me_response.json()["id"]

    # Apply filters
    filtered = []
    for card in cards:
        # Filter: assigned to me
        if assigned_to_me and my_id not in card.get("idMembers", []):
            continue

        # Filter: has due date
        if has_due and not card.get("due"):
            continue

        # Filter: due before
        if due_before:
            if not card.get("due"):
                continue
            due_date = datetime.fromisoformat(card["due"].replace("Z", "+00:00"))
            before_date = datetime.fromisoformat(due_before.replace("Z", "+00:00"))
            if due_date > before_date:
                continue

        # Filter: due after
        if due_after:
            if not card.get("due"):
                continue
            due_date = datetime.fromisoformat(card["due"].replace("Z", "+00:00"))
            after_date = datetime.fromisoformat(due_after.replace("Z", "+00:00"))
            if due_date < after_date:
                continue

        # Filter: labels
        if labels:
            card_labels = [l["name"] for l in card.get("labels", [])]
            if not any(l in card_labels for l in labels):
                continue

        filtered.append(card)

    return filtered


def submit_lipsync(
    title: str,
    channel: str,
    avatar_path: str,
    audio_files: list[str],
    drive_folder: Optional[str] = None,
):
    """
    Submit a lipsync job using the lipsync_automator module.
    """
    from lipsync_automator import submit_lipsync_job, get_channel

    # Auto-fill drive folder from channel if not provided
    if not drive_folder:
        channel_config = get_channel(channel)
        if channel_config:
            drive_folder = channel_config.drive_folder

    if not drive_folder:
        print("[ERROR] No drive folder specified and not found in channels.py")
        return False

    # Verify files exist
    if not Path(avatar_path).exists():
        print(f"[ERROR] Avatar file not found: {avatar_path}")
        return False

    for audio in audio_files:
        if not Path(audio).exists():
            print(f"[ERROR] Audio file not found: {audio}")
            return False

    print(f"  Title:     {title}")
    print(f"  Channel:   {channel}")
    print(f"  Avatar:    {avatar_path}")
    print(f"  Drive:     {drive_folder}")
    print(f"  Audio:     {audio_files}")

    try:
        status, body = submit_lipsync_job(
            video_title=title,
            channel_code=channel,
            avatar_path=avatar_path,
            audio_files=audio_files,
            drive_folder_id=drive_folder,
        )

        if status in (200, 201, 202):
            print(f"  [OK] Status: {status}")
            return True
        else:
            print(f"  [ERROR] Status: {status}")
            return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def main():
    # Enable UTF-8 output on Windows
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Fetch cards from Trello and submit to lipsync automator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available boards
  python trello_to_lipsync.py --list-boards

  # List available lists on a board
  python trello_to_lipsync.py --board 699ddc7210f3d0fab35d2e5d --list-lists

  # Show cards in "Ready To Upload" list (dry run)
  python trello_to_lipsync.py --list "Ready To Upload" --dry-run

  # Submit cards from "Editing" list to lipsync
  python trello_to_lipsync.py --list "Editing" --channel RRU --avatar avatar.png audio.mp3

  # Use specific account
  python trello_to_lipsync.py --account david --list "Ready To Upload" --dry-run

  # Only cards assigned to you
  python trello_to_lipsync.py --list "Ready To Upload" --assigned-to-me

  # Cards due within 3 days
  python trello_to_lipsync.py --list "Scheduled" --due-within 3
        """
    )

    # Account selection
    parser.add_argument(
        "--account", "-A",
        default="david",
        help="Trello account name (matches file in accounts/)"
    )

    # Board selection
    parser.add_argument(
        "--board", "-b",
        default=None,
        help="Board ID (default: from account config)"
    )
    parser.add_argument(
        "--list-accounts",
        action="store_true",
        help="List all available accounts and exit"
    )
    parser.add_argument(
        "--list-boards",
        action="store_true",
        help="List all accessible boards and exit"
    )
    parser.add_argument(
        "--list-lists",
        action="store_true",
        help="List all lists on the board and exit"
    )

    # Card filtering
    parser.add_argument(
        "--list", "-l",
        help="Filter cards by list name"
    )
    parser.add_argument(
        "--list-id",
        help="Filter cards by list ID (overrides --list)"
    )
    parser.add_argument(
        "--assigned-to-me", "-m",
        action="store_true",
        help="Only cards assigned to you"
    )
    parser.add_argument(
        "--has-due",
        action="store_true",
        help="Only cards with due dates"
    )
    parser.add_argument(
        "--due-within",
        type=int,
        metavar="DAYS",
        help="Cards due within N days"
    )
    parser.add_argument(
        "--due-before",
        help="Cards due before ISO date (e.g., 2026-03-01)"
    )
    parser.add_argument(
        "--due-after",
        help="Cards due after ISO date (e.g., 2026-02-01)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Filter by label names"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Maximum number of cards to process"
    )

    # Lipsync submission (optional for listing modes)
    parser.add_argument(
        "--channel", "-c",
        default="RRU",
        choices=["RRU", "DSR", "JDRP"],
        help="Channel code (default: RRU)"
    )
    parser.add_argument(
        "--avatar", "-a",
        default=None,
        help="Path to avatar image file (required for submission)"
    )
    parser.add_argument(
        "--drive-folder",
        help="Google Drive folder ID (auto-filled from channels.py if omitted)"
    )
    parser.add_argument(
        "audio_files",
        nargs="*",
        help="Audio segment files to submit (if omitted, just lists cards)"
    )

    # Mode options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cards that would be submitted without actually submitting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Load account credentials
    api_key, token, default_board = get_account_config(args.account)
    print(f"[INFO] Using account: {args.account}")

    # Validate required args for submission modes
    needs_avatar = (
        not args.list_accounts and
        not args.list_boards and
        not args.list_lists and
        (args.audio_files or not args.dry_run)
    )

    if needs_avatar and not args.avatar:
        parser.error("--avatar/-a is required when submitting to lipsync or using audio files")

    # Use default board from account config if not specified
    board_id = args.board or default_board

    # List accounts mode
    if args.list_accounts:
        accounts = list_accounts()
        print("Available accounts:")
        if accounts:
            for a in accounts:
                print(f"  - {a}")
        else:
            print("  (none found)")
        print("\nCreate new account: Degold/accounts/<name>.env")
        return

    # List boards mode
    if args.list_boards:
        boards = get_boards(api_key, token)
        print("Available boards:")
        for b in boards:
            print(f"  {b['id']} - {b['name']}")
        return

    # List lists mode
    if args.list_lists:
        lists = get_lists(board_id, api_key, token)
        print("Available lists:")
        for l in lists:
            print(f"  {l['id']} - {l['name']}")
        return

    # Calculate due filters
    due_before = args.due_before
    due_after = args.due_after

    if args.due_within:
        from datetime import timedelta
        future = datetime.now() + timedelta(days=args.due_within)
        due_before = future.isoformat()

    # Get cards
    cards = get_cards(
        api_key=api_key,
        token=token,
        board_id=board_id,
        list_name=args.list,
        list_id=args.list_id,
        assigned_to_me=args.assigned_to_me,
        has_due=args.has_due,
        due_before=due_before,
        due_after=due_after,
        limit=args.limit,
        labels=args.labels,
    )

    if not cards:
        print("[INFO] No cards found matching criteria")
        return

    print(f"[INFO] Found {len(cards)} card(s)")

    # Show cards without submitting
    if args.dry_run or not args.audio_files:
        for i, card in enumerate(cards, 1):
            print(f"\n{i}. {card['name']}")
            print(f"   URL: {card.get('shortUrl', 'N/A')}")
            if card.get('due'):
                print(f"   Due: {card['due']}")
            if card.get('labels'):
                labels = ", ".join([l['name'] for l in card['labels']])
                print(f"   Labels: {labels}")
        print(f"\n[DRY-RUN] Would submit {len(cards)} card(s)")
        return

    # Submit to lipsync
    print(f"\n[INFO] Submitting {len(cards)} card(s) to lipsync...")

    success_count = 0
    for i, card in enumerate(cards, 1):
        print(f"\n[{i}/{len(cards)}] {card['name']}")

        if args.dry_run:
            continue

        # Submit card
        result = submit_lipsync(
            title=card["name"],
            channel=args.channel,
            avatar_path=args.avatar,
            audio_files=args.audio_files,
            drive_folder=args.drive_folder,
        )

        if result:
            success_count += 1

    print(f"\n[INFO] Submitted {success_count}/{len(cards)} cards successfully")


if __name__ == "__main__":
    main()
