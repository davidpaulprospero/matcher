"""SRT read/write helpers for the standalone chapter scripts.

Uses `src/caption/parsers.py:parse_srt` for parsing (the project's canonical
parser) and a small SRT writer that mirrors `scripts/split_srt.py`.

Segments are exposed to scripts as plain dicts with keys:
    - `index` (int): 0-based sequential index
    - `text` (str): caption text
    - `start` (float): start time in seconds
    - `end` (float): end time in seconds

This shape matches what `src/chapter_detection/listicle_detector.detect_listicle_groups`
expects (it tolerates dicts with at least `text` and `index`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

import sys

# Ensure repo root is on sys.path so we can import src.* from this script
# when invoked as `python scripts/chapters/_srt_io.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.caption.parsers import parse_srt  # noqa: E402


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_time(t: str) -> float:
    """Parse `HH:MM:SS,mmm` to seconds. Comma-form SRT only."""
    t = t.replace(",", ".")
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(t)


def _format_time(seconds: float) -> str:
    """Format seconds as `HH:MM:SS,mmm`."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def load_segments(srt_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read an SRT file and return segments as dicts with `index`/`text`/`start`/`end`.

    Uses the project's canonical `parse_srt` (handles edge cases, multi-line text).
    """
    path = Path(srt_path)
    content = path.read_text(encoding="utf-8")
    parse_result = parse_srt(content, video_id=str(path))
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(parse_result.segments):
        out.append(
            {
                "index": i,
                "text": seg.text,
                "start": float(seg.start_time),
                "end": float(seg.end_time),
                "source_file": seg.source_file,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_entry(f, number: int, start: float, end: float, text: str) -> None:
    f.write(f"{number}\n")
    f.write(f"{_format_time(start)} --> {_format_time(end)}\n")
    f.write(f"{text.rstrip()}\n\n")


def write_srt(segments: List[Dict[str, Any]], srt_path: Union[str, Path]) -> Path:
    """Write segments to SRT. Indices are renumbered starting from 1."""
    out = Path(srt_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            _write_entry(
                f,
                number=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg.get("text", "")),
            )
    return out


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def renumber_and_shift(
    segments: List[Dict[str, Any]],
    *,
    shift_seconds: float = 0.0,
    renumber: bool = True,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    """Return a copy of `segments` with timestamps shifted by `shift_seconds`.

    Optional `renumber` rewrites `index` to be sequential starting from `start_index`.
    Negative `shift_seconds` is clamped to 0 per-entry to keep timestamps monotonic.
    """
    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        new_start = max(0.0, float(seg["start"]) - shift_seconds)
        new_end = max(new_start, float(seg["end"]) - shift_seconds)
        new_seg = {
            **seg,
            "start": new_start,
            "end": new_end,
        }
        if renumber:
            new_seg["index"] = start_index + i
        out.append(new_seg)
    return out


# ---------------------------------------------------------------------------
# Conversion back to plain string text
# ---------------------------------------------------------------------------


def concat_text(segments: List[Dict[str, Any]], sep: str = " ") -> str:
    return sep.join(str(s.get("text", "")).strip() for s in segments).strip()


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Make a filesystem-safe slug from arbitrary text."""
    slug = _SLUG_RE.sub("_", text.strip()).strip("_")
    return slug[:max_len] or "chapter"
