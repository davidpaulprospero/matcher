"""
Persistent adaptive caption format ordering stats (US-67-006).

Saves per-format success/failure counts to .cache/caption_format_stats.json
so that adaptive format ordering survives across pipeline runs. Entries older
than 7 days are excluded from the ordering calculation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STATS_FILENAME = "caption_format_stats.json"
STALE_DAYS = 7


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_stale(last_updated: str, max_age_days: int = STALE_DAYS) -> bool:
    """Return True if *last_updated* ISO timestamp is older than *max_age_days*."""
    try:
        ts = datetime.fromisoformat(last_updated)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        return age.total_seconds() > max_age_days * 86400
    except (ValueError, TypeError):
        return True  # unparseable → treat as stale


class FormatStatsFile:
    """Read / write ``caption_format_stats.json`` in a project's .cache dir.

    File schema::

        {
          "vtt":   {"successes": 95, "failures": 2, "last_updated": "2026-02-05T12:00:00+00:00"},
          "json3": {"successes": 80, "failures": 5, "last_updated": "2026-02-05T12:00:00+00:00"},
          ...
        }
    """

    def __init__(self, project_dir: str | Path) -> None:
        self.path = Path(project_dir) / ".cache" / STATS_FILENAME

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save(self, format_stats: Dict[str, Dict[str, int]]) -> bool:
        """Persist *format_stats* to disk.

        Args:
            format_stats: ``{format_name: {successes: int, failures: int}}``
                          (``last_updated`` is set automatically if missing).

        Returns:
            True on success, False on error.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # Merge with existing data so we accumulate across runs
            existing = self._read_raw()
            now = _now_iso()

            for fmt, counts in format_stats.items():
                entry = existing.get(fmt, {"successes": 0, "failures": 0})
                entry["successes"] = counts.get("successes", 0)
                entry["failures"] = counts.get("failures", 0)
                entry["last_updated"] = now
                existing[fmt] = entry

            self.path.write_text(
                json.dumps(existing, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"Saved format stats to {self.path}: {list(existing.keys())}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save format stats: {e}")
            return False

    # ------------------------------------------------------------------
    # Load (with stale-entry filtering)
    # ------------------------------------------------------------------
    def load(self, max_age_days: int = STALE_DAYS) -> Dict[str, Dict[str, int]]:
        """Load format stats, excluding entries older than *max_age_days*.

        Returns:
            ``{format_name: {successes: int, failures: int, last_updated: str}}``
            Empty dict when file missing or on error.
        """
        raw = self._read_raw()
        if not raw:
            return {}

        fresh: Dict[str, Dict[str, int]] = {}
        for fmt, entry in raw.items():
            last_updated = entry.get("last_updated", "")
            if not _is_stale(last_updated, max_age_days):
                fresh[fmt] = entry

        if len(fresh) < len(raw):
            excluded = len(raw) - len(fresh)
            logger.debug(f"Excluded {excluded} stale format stats entries (>{max_age_days}d)")

        return fresh

    # ------------------------------------------------------------------
    # Derive preferred format order
    # ------------------------------------------------------------------
    def compute_order(
        self,
        default_formats: Optional[List[str]] = None,
        max_age_days: int = STALE_DAYS,
    ) -> List[str]:
        """Return format names sorted by success rate (desc), stale entries excluded.

        Formats not present in the stats file but in *default_formats* are
        appended at the end in their original order.
        """
        stats = self.load(max_age_days=max_age_days)
        if not stats:
            return list(default_formats) if default_formats else []

        def _rate(entry: Dict[str, int]) -> float:
            s = entry.get("successes", 0)
            f = entry.get("failures", 0)
            total = s + f
            return s / total if total > 0 else 0.0

        sorted_fmts = sorted(stats.keys(), key=lambda f: _rate(stats[f]), reverse=True)

        # Append any default formats not already in sorted list
        if default_formats:
            for fmt in default_formats:
                if fmt not in sorted_fmts:
                    sorted_fmts.append(fmt)

        return sorted_fmts

    # ------------------------------------------------------------------
    # Helpers for building stats from CaptionMetrics
    # ------------------------------------------------------------------
    @staticmethod
    def from_metrics_counts(
        success_counts: Dict[str, int],
        failure_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Build the stats dict from CaptionMetrics.format_success_counts.

        Since CaptionMetrics only tracks successes, failures default to 0
        unless *failure_counts* is provided.
        """
        result: Dict[str, Dict[str, int]] = {}
        all_fmts = set(success_counts.keys())
        if failure_counts:
            all_fmts |= set(failure_counts.keys())

        for fmt in all_fmts:
            result[fmt] = {
                "successes": success_counts.get(fmt, 0),
                "failures": failure_counts.get(fmt, 0) if failure_counts else 0,
            }
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _read_raw(self) -> Dict[str, dict]:
        """Read the JSON file. Returns empty dict on missing / corrupt file."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as e:
            logger.warning(f"Failed to read format stats file: {e}")
            return {}
