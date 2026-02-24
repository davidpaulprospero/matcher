#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
History Tracker for Trello Cards

Tracks which cards have been submitted to lipsync and their status.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum


class Status(Enum):
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CardHistory:
    card_id: str
    card_name: str
    trello_url: str
    status: str
    submitted_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    account: Optional[str] = None


class HistoryTracker:
    def __init__(self, history_file: Optional[Path] = None):
        if history_file is None:
            history_file = Path(__file__).parent / "history.json"
        self.history_file = history_file
        self.history: dict[str, CardHistory] = {}
        self._load()

    def _load(self):
        """Load history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = {
                        k: CardHistory(**v) for k, v in data.items()
                    }
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[WARN] Could not load history: {e}")
                self.history = {}

    def _save(self):
        """Save history to file."""
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(
                {k: asdict(v) for k, v in self.history.items()},
                f,
                indent=2,
                ensure_ascii=False
            )

    def add(
        self,
        card_id: str,
        card_name: str,
        trello_url: str,
        account: str,
        status: Status = Status.SUBMITTED,
        error: Optional[str] = None,
    ):
        """Add a card to history."""
        self.history[card_id] = CardHistory(
            card_id=card_id,
            card_name=card_name,
            trello_url=trello_url,
            status=status.value,
            submitted_at=datetime.now().isoformat(),
            account=account,
            error=error,
        )
        self._save()

    def mark_completed(self, card_id: str):
        """Mark a card as completed."""
        if card_id in self.history:
            self.history[card_id].status = Status.COMPLETED.value
            self.history[card_id].completed_at = datetime.now().isoformat()
            self._save()

    def mark_failed(self, card_id: str, error: str):
        """Mark a card as failed."""
        if card_id in self.history:
            self.history[card_id].status = Status.FAILED.value
            self.history[card_id].error = error
            self._save()

    def is_processed(self, card_id: str) -> bool:
        """Check if a card has been processed."""
        return card_id in self.history

    def get(self, card_id: str) -> Optional[CardHistory]:
        """Get history entry for a card."""
        return self.history.get(card_id)

    def get_all(self, status: Optional[Status] = None) -> list[CardHistory]:
        """Get all history entries, optionally filtered by status."""
        entries = list(self.history.values())
        if status:
            entries = [e for e in entries if e.status == status.value]
        return sorted(entries, key=lambda x: x.submitted_at, reverse=True)

    def clear(self):
        """Clear all history."""
        self.history = {}
        self._save()

    def remove(self, card_id: str):
        """Remove a card from history."""
        if card_id in self.history:
            del self.history[card_id]
            self._save()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Track lipsync history for Trello cards")
    parser.add_argument("--history-file", type=Path, help="Path to history file")
    parser.add_argument("--list", "-l", action="store_true", help="List all history")
    parser.add_argument("--status", choices=["submitted", "completed", "failed"], help="Filter by status")
    parser.add_argument("--clear", action="store_true", help="Clear all history")
    parser.add_argument("--remove", metavar="CARD_ID", help="Remove card from history")

    args = parser.parse_args()

    tracker = HistoryTracker(args.history_file)

    if args.clear:
        tracker.clear()
        print("[OK] History cleared")
        return

    if args.remove:
        tracker.remove(args.remove)
        print(f"[OK] Removed {args.remove} from history")
        return

    if args.list:
        status = Status(args.status) if args.status else None
        entries = tracker.get_all(status)

        if not entries:
            print("[INFO] No history entries")
            return

        print(f"History ({len(entries)} entries):\n")
        for e in entries:
            status_icon = {
                "submitted": "[*]",
                "completed": "[OK]",
                "failed": "[X]",
            }.get(e.status, "[?]")

            print(f"{status_icon} {e.card_name}")
            print(f"    URL: {e.trello_url}")
            print(f"    Status: {e.status}")
            print(f"    Submitted: {e.submitted_at}")
            if e.completed_at:
                print(f"    Completed: {e.completed_at}")
            if e.error:
                print(f"    Error: {e.error}")
            print()


if __name__ == "__main__":
    main()
