#!/usr/bin/env python3
"""
lower_thirds.py — Extract named entities (people, places, dates, info) from a
voiceover SRT using a local Ollama model, render broadcast-style lower-third
PNGs with Pillow, and emit a separate OTIO file containing a single video
track of those PNGs with FreezeFrame effects so DaVinci Resolve can import
them directly. No paid software, no FFmpeg compositing.

Usage:
    python scripts/lower_thirds.py "<project_or_srt>" [--ollama-model llama3.2]
    python scripts/lower_thirds.py "<project>" --dry-run
    python scripts/lower_thirds.py "<project>" --resolution 1920x1080 --font arial.ttf
    python scripts/lower_thirds.py "<project>" --template {classic|minimal|boxed|modern|corner|newsroom}
    python scripts/lower_thirds.py "<project>" --mode {image|video} --animation-style {slide|pop|fade}

Outputs:
    <project>/lowerthirds.otio          — single V1 track "Lower Thirds"
    <project>/lowerthirds/lowerthird_NNN.png  — transparent RGBA assets
    <project>/lowerthirds/entities.json — summary of detected entities
"""

import argparse
import gzip
import json
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange
from PIL import Image, ImageDraw, ImageFont

from src.utils import parse_srt_file, SRTSegment
from src.otio.utils import _to_windows_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RATE = 30.0

ANIMATION_STYLES = ("slide", "pop", "fade")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}
_VIDEO_EXTS = {".mov", ".mp4", ".mkv", ".webm", ".m4v"}

ENTITY_EXTRACT_SYSTEM_PROMPT = (
    "You extract notable entities from a voiceover transcript and classify each. "
    "Output ONLY a JSON array (top-level must be [...]) — no commentary, no markdown fences, "
    "no top-level object. If nothing is mentioned, return [].\n\n"
    "Each entry must have THREE fields:\n"
    "  - \"name\": the entity name as it appears (or its short form for graphics)\n"
    "  - \"entity_type\": one of \"person\", \"place\", \"date\", or \"info\"\n"
    "  - \"role\": an optional secondary descriptor (NOT a type — see examples)\n\n"
    "Examples:\n"
    '  {"name": "Alice Smith", "entity_type": "person", "role": "Mayor"}\n'
    '  {"name": "Paris", "entity_type": "place", "role": "France"}\n'
    '  {"name": "1946 — Trust founded", "entity_type": "date", "role": ""}\n'
    '  {"name": "99% solar", "entity_type": "info", "role": "2010 census"}\n\n'
    "Per-type rules:\n"
    "- \"person\": a real person the narrator names.\n"
    "- \"place\": a proper-noun location (city, landmark, building, country).\n"
    "- \"date\": a specific year or named historical date the narrator emphasizes. "
    "ALWAYS append a short contextual tag so the on-screen graphic is meaningful "
    "on its own. Format: \"YEAR — short event\". Examples: \"1946 — Trust founded\", "
    "\"1989 — groundbreaking\", \"1869 — Suez Canal opens\". Plain bare years like "
    "\"1946\" are NOT acceptable — the editor needs to know what happened. Limit to "
    "5 words after the em dash.\n"
    "- \"info\": a short, on-screen fact callout — a striking statistic, headline "
    "number, or surprising claim. name MUST be <= 8 words.\n\n"
    "IMPORTANT: \"entity_type\" must be exactly \"person\", \"place\", \"date\", or \"info\". "
    "Do not put the type in the \"role\" field.\n\n"
    "Only include items the narrator explicitly mentions. Do not infer or summarize. "
    "If nothing is mentioned, return [].\n\n"
    "The system will find where each entity appears in the transcript and set the "
    "timing — do NOT include any index/segment/timing fields in your output."
)

ENTITY_EXTRACT_USER_TEMPLATE = (
    'Transcript (one entry per line, "i: HH:MM:SS,mmm --> HH:MM:SS,mmm | text"):\n'
    "{srt_dump}\n\n"
    "Return only the JSON array."
)


class EntityType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    DATE = "date"
    INFO = "info"
    CHAPTER = "chapter"


_ENTITY_TYPE_ALIASES: Dict[str, EntityType] = {
    # person
    "people": EntityType.PERSON,
    "p": EntityType.PERSON,
    # place
    "location": EntityType.PLACE,
    "city": EntityType.PLACE,
    "country": EntityType.PLACE,
    "region": EntityType.PLACE,
    "landmark": EntityType.PLACE,
    "building": EntityType.PLACE,
    "l": EntityType.PLACE,
    # date
    "year": EntityType.DATE,
    "time": EntityType.DATE,
    "event": EntityType.DATE,
    "d": EntityType.DATE,
    # info
    "fact": EntityType.INFO,
    "stat": EntityType.INFO,
    "number": EntityType.INFO,
    "figure": EntityType.INFO,
    # chapter (section heading / topic boundary from src/chapter_detection/detector.py)
    "section": EntityType.CHAPTER,
    "part": EntityType.CHAPTER,
    "segment": EntityType.CHAPTER,
    "topic": EntityType.CHAPTER,
}


@dataclass
class EntitySpan:
    name: str
    role: str
    start_time: float
    end_time: float
    srt_indices: List[int] = field(default_factory=list)
    entity_type: EntityType = EntityType.PERSON


class Layout(str, Enum):
    BAR = "bar"
    UNDERLINE = "under"
    PILL = "pill"
    TOP_CORNER = "top"
    STACK = "stack"


@dataclass
class LowerThirdTemplate:
    width: int = 1920
    height: int = 1080
    bar_height: int = 180
    bar_color: Tuple[int, int, int, int] = (20, 20, 20, 235)
    accent_color: Tuple[int, int, int] = (255, 200, 40)
    accent_width: int = 8
    name_font_size: int = 64
    role_font_size: int = 36
    name_color: Tuple[int, int, int] = (255, 255, 255)
    role_color: Tuple[int, int, int] = (220, 220, 220)
    pad_x: int = 60
    pad_y: int = 30
    line_gap: int = 8
    y_offset_from_bottom: int = 80
    font_name: Optional[str] = None
    layout: Layout = Layout.BAR
    bg_color_2: Tuple[int, int, int, int] = (0, 0, 0, 0)
    shadow: bool = False
    pill_radius: int = 60
    pill_max_width: int = 1200
    frame_thickness: int = 4
    y_offset_from_top: int = 60


def _tpl(**overrides) -> LowerThirdTemplate:
    """Build a template inheriting from the classic default + applying overrides."""
    base = LowerThirdTemplate()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


TEMPLATE_REGISTRY: Dict[str, LowerThirdTemplate] = {
    "classic": _tpl(),  # byte-identical to today's default output
    "minimal": _tpl(
        bar_height=0,
        accent_width=0,
        bar_color=(0, 0, 0, 0),
        name_font_size=72,
        role_font_size=32,
        pad_y=20,
        line_gap=14,
        y_offset_from_bottom=140,
        layout=Layout.UNDERLINE,
    ),
    "boxed": _tpl(
        bar_height=220,
        bar_color=(20, 20, 20, 230),
        bg_color_2=(60, 60, 60, 230),
        accent_width=0,
        name_font_size=70,
        role_font_size=34,
        pad_x=70,
        pad_y=40,
        y_offset_from_bottom=70,
        shadow=True,
        pill_radius=24,
        layout=Layout.BAR,
    ),
    "modern": _tpl(
        bar_height=160,
        bar_color=(15, 15, 15, 220),
        accent_width=0,
        name_font_size=58,
        role_font_size=30,
        pad_x=50,
        pad_y=30,
        y_offset_from_bottom=80,
        pill_radius=80,
        pill_max_width=1100,
        layout=Layout.PILL,
    ),
    "corner": _tpl(
        bar_height=130,
        bar_color=(0, 0, 0, 210),
        accent_color=(220, 40, 40),
        accent_width=10,
        name_font_size=56,
        role_font_size=28,
        pad_x=40,
        pad_y=25,
        y_offset_from_top=60,
        y_offset_from_bottom=0,
        layout=Layout.TOP_CORNER,
    ),
    "newsroom": _tpl(
        bar_height=240,
        bar_color=(245, 245, 245, 235),
        accent_color=(180, 30, 30),
        accent_width=6,
        name_color=(15, 15, 15),
        role_color=(80, 80, 80),
        name_font_size=58,
        role_font_size=30,
        pad_x=50,
        pad_y=30,
        line_gap=6,
        y_offset_from_bottom=60,
        frame_thickness=4,
        layout=Layout.STACK,
    ),
}


# Track ordering + naming for the per-category OTIO output. The 3 important
# categories (Names / Dates / Key Points) come first; Places follows so the
# editor can hide/collapse places if a project has many of them. Chapters are
# optional — they appear last and only when the project checkpoint has
# `chapter_data.chapters` from src/chapter_detection/detector.py.
TRACK_ORDER: List[EntityType] = [
    EntityType.PERSON,  # V1 - Lower Thirds · Names
    EntityType.DATE,   # V2 - Lower Thirds · Dates
    EntityType.INFO,   # V3 - Lower Thirds · Key Points
    EntityType.PLACE,  # V4 - Lower Thirds · Places
    EntityType.CHAPTER,  # V5 - Lower Thirds · Chapters
]

TRACK_NAME_BY_TYPE: Dict[EntityType, str] = {
    EntityType.PERSON:  "V1 - Lower Thirds · Names",
    EntityType.DATE:    "V2 - Lower Thirds · Dates",
    EntityType.INFO:    "V3 - Lower Thirds · Key Points",
    EntityType.PLACE:   "V4 - Lower Thirds · Places",
    EntityType.CHAPTER: "V5 - Lower Thirds · Chapters",
}


def find_srt_path(project_dir: Path) -> Optional[Path]:
    """Find the appropriate SRT path for a project, preferring trimmed variant.

    Mirrors scripts/title_color_video.py:33 — search voiceover/ subdir, then
    project root; prefer *_trimmed.srt over voiceover.srt.
    """
    voiceover_dir = project_dir / "voiceover"
    candidates: List[Path] = []

    if voiceover_dir.exists():
        for f in voiceover_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".srt":
                candidates.append(f)

    if project_dir.exists():
        for f in project_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".srt" and f not in candidates:
                candidates.append(f)

    if not candidates:
        return None

    trimmed_stems = {f.stem.replace("_trimmed", "") for f in candidates if "_trimmed" in f.stem}
    if trimmed_stems:
        candidates = [
            f for f in candidates
            if "_trimmed" in f.stem or f.stem not in trimmed_stems
        ]

    return candidates[0] if candidates else None


def load_chapter_entities(
    project_dir: Path,
    segments: List[SRTSegment],
) -> List[EntitySpan]:
    """Load chapter title overlays from `<project>/checkpoint.json → chapter_data.chapters`.

    Each `ChapterCandidate` (from src/chapter_detection/models.py:44) becomes one
    `EntitySpan` with `entity_type=CHAPTER`. The chapter entity spans the first
    TWO segments of the chapter (≈10s of on-screen title — short enough to read
    without dominating the chapter). Timing is left intact for the downstream
    clamp/redistribute stages to trim further if needed.

    Returns an empty list when:
    - `checkpoint.json` is missing, not readable, or has no `chapter_data.chapters`
    - the chapters list is empty
    - any chapter lacks a `title` or a valid `start_segment_idx`
    """
    cp_path = project_dir / "checkpoint.json"
    if not cp_path.exists():
        return []

    try:
        with open(cp_path, "rb") as f:
            raw = f.read()
        # Checkpoint may be raw JSON or gzip-compressed (gzip magic 1f 8b).
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        cp = json.loads(raw.decode("utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, Exception) as exc:
        logger.warning(f"Could not read checkpoint at {cp_path}: {exc}")
        return []

    chapter_data = cp.get("chapter_data") or {}
    raw_chapters = chapter_data.get("chapters") or []
    if not raw_chapters:
        return []

    out: List[EntitySpan] = []
    for ch in raw_chapters:
        title = _sanitize_chapter_title((ch.get("title") or "").strip())
        start_idx = ch.get("start_segment_idx")
        end_idx = ch.get("end_segment_idx") or start_idx
        if not title or not isinstance(start_idx, int):
            continue
        if start_idx < 1 or start_idx > len(segments):
            continue
        # Pin to first two segments of the chapter for a title-card feel.
        end_seg_idx = min(end_idx, start_idx + 1)
        if end_seg_idx < start_idx:
            end_seg_idx = start_idx
        if end_seg_idx > len(segments):
            end_seg_idx = len(segments)
        try:
            start_seg = segments[start_idx - 1]
            end_seg = segments[end_seg_idx - 1]
        except IndexError:
            continue
        # Use the chapter's `location_name` as the role/tagline if present.
        role = (ch.get("location_name") or "").strip()
        topics = ch.get("topics") or []
        if not role and topics:
            role = ", ".join(str(t) for t in topics[:3])
        out.append(EntitySpan(
            name=title,
            role=role,
            start_time=start_seg.start_time,
            end_time=end_seg.end_time,
            srt_indices=list(range(start_idx, end_seg_idx + 1)),
            entity_type=EntityType.CHAPTER,
        ))
    if out:
        logger.info(f"Chapters: loaded {len(out)} from {cp_path.name}")
    return out


def build_ollama_prompt(segments: List[SRTSegment]) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for entity extraction."""
    lines: List[str] = []
    for seg in segments:
        start_tc = _seconds_to_srt_tc(seg.start_time)
        end_tc = _seconds_to_srt_tc(seg.end_time)
        text = seg.text.replace("\n", " ").strip()
        lines.append(f'{seg.index}: {start_tc} --> {end_tc} | {text}')
    srt_dump = "\n".join(lines)
    user_prompt = ENTITY_EXTRACT_USER_TEMPLATE.format(srt_dump=srt_dump)
    return ENTITY_EXTRACT_SYSTEM_PROMPT, user_prompt


# Patterns for Python pre-screening of entity candidates from SRT text.
# These run BEFORE Ollama so the LLM only classifies/filters — Python owns
# the timing and never trusts the model to invent names.
_DATE_RE = re.compile(r"\b(1[789]\d{2}|20\d{2})\b")
_DECADE_RE = re.compile(r"\b(\d{3})0s\b")  # "1990s"
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_CURRENCY_RE = re.compile(r"\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|thousand|m|b|k))?")
_NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:miles|mi|feet|ft|inches|in|kg|lbs?|pounds?|hours?|hrs?|minutes?|mins?|"
    r"seconds?|secs?|days?|years?|yrs?|months?|degrees?|°[FC]?|"
    r"times|fold|percent|°)\b",
    re.IGNORECASE,
)
# Big-number stats: comma-thousands with optional internal space
# ("2 ,000", "4 ,000", "10,000"), optionally followed by a noun.
# Voiceover SRTs often insert a space inside numbers.
_BIG_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:\s*,\s*\d{3})+(?:\s+\w+)?"
    r"|\b\d+\s+(?:robotaxi|robotaxies|vehicle|vehicles|company|companies"
    r"|city|cities|country|countries|year|years|month|months"
    r"|service|services|operator|operators|provider|providers)\b"
)
_PERSON_TITLES_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Dame|Senator|Sen|Rep(?:resentative)?|"
    r"President|Vice President|Governor|Mayor|Captain|Cpt|Lt|Col|Gen|"
    r"Hon(?:orable)?|Rev|Pastor|Father|Brother|Sister)\.?\s+"
    r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,3})",
    re.IGNORECASE,
)
# Place hints: prepositions + Capitalized proper nouns ("in Paris", "from Wales")
_PLACE_PREP_RE = re.compile(
    r"\b(?:in|at|from|to|of|near|across|through|via|toward|into|onto|over|around|"
    r"between)\s+"
    r"(?:the\s+)?"
    r"((?:[A-Z][a-zA-Z'\-]+"
    r"(?:\s+(?:of|the|de|la|le|du)\s+|\s+)[A-Z][a-zA-Z'\-]+|\s+"
    r"[A-Z][a-zA-Z'\-]+)*"
    r"[A-Z][a-zA-Z'\-]+)",
)
# Place suffix patterns (weak signal; combined with context)
_PLACE_SUFFIX_RE = re.compile(
    r"\b[A-Z][a-zA-Z'\-]+"
    r"(?:shire|shire|burgh|borough|burg|ville|town|city|haven|port|ford|"
    r"field|wood|land|stan|abad|polis|ington|ington|heim|heim|"
    r"stan|grad|sk|ø|øya|øy)\b"
)
# Multi-word proper-noun sequences anywhere in the text
_PROPER_NOUN_RE = re.compile(
    r"\b(?<![.!?]\s)([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,4})\b"
)
# Brands: capitalized name followed by a verb-like token.
# Catches "Uber began", "Waymo built", "Apollo GO launched".
_BRAND_VERB_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9.]+(?:\s+[A-Z][A-Za-z0-9.]+)?)\s+"
    r"(?:began|offers?|offered|announced?|built|operates?|partners?|founded|"
    r"launched|launches?|deployed|expanded|reported|raised|partnered|acquired|"
    r"merged|started|rolled\s+out|rolls\s+out|pushed|pushes?|rolled)\b"
)
# Sentence-start capitalized proper noun (1-3 words). Catches brands at
# the beginning of a segment when the verb comes AFTER the name ("Uber began"
# at SRT start, "Waymo built" at SRT start).
_SENTENCE_START_RE = re.compile(
    r"(?:^|[.!?]\s+|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?=\s|$|[,.])"
)
# Stopwords that look like proper nouns at sentence start but aren't.
_SENTENCE_START_STOPWORDS = {
    "now", "then", "the", "a", "an", "and", "but", "or", "so", "yet",
    "for", "of", "in", "on", "at", "to", "by", "with", "from",
    "as", "is", "was", "were", "are", "be", "been", "being",
    "this", "that", "these", "those", "it", "they", "we", "he", "she",
    "i", "you", "his", "her", "their", "our", "my", "your",
    "today", "yesterday", "tomorrow",
    "more", "less", "much", "many", "some", "any", "all", "most",
    "still", "also", "however", "although", "though", "because",
    "after", "before", "since", "until", "while", "when", "where",
    "five", "six", "seven", "eight", "nine", "ten",
}


@dataclass
class Candidate:
    """Python-extracted entity candidate from SRT text."""
    text: str
    srt_index: int
    hint: str  # 'person' | 'place' | 'date' | 'info'
    context: str  # surrounding ~80 chars for LLM context


def _proper_noun_at_sentence_start(text: str, m: re.Match) -> bool:
    """Heuristic: a proper noun match is likely a sentence-starter, not a name.

    Returns True when the match starts the text or follows a sentence terminator
    followed by whitespace — in which case we treat it as a generic capitalized
    word (drop it from candidates) unless it has stronger signals (titles,
    multi-word, place suffixes).
    """
    start = m.start()
    if start == 0:
        return True
    # Look at the preceding non-space char
    prefix = text[:start].rstrip()
    if not prefix:
        return True
    last_char = prefix[-1]
    return last_char in ".!?"


def extract_candidates(segments: List[SRTSegment]) -> List[Candidate]:
    """Python heuristic extraction of entity candidates from SRT text.

    Runs BEFORE Ollama. Each candidate carries its srt_index so the LLM cannot
    hallucinate timing — Python owns where each mention actually is.

    Patterns:
      - person: titles (Mr/Dr/Senator + name) + multi-word proper nouns that
        don't look like sentence-starters and aren't on a place suffix list
      - place: preposition + capitalized ("in Paris"); place suffix words
        (-shire/-burg/-ville/-land/-town); multi-word proper nouns after
        prepositions; standalone 2+ word proper nouns
      - date: 4-digit years (1800-2099), decades ("1990s")
      - info: percentages, currency, number+unit, fractions ("half of")

    Returns candidates sorted by (srt_index, hint) with duplicates deduped
    by (text.lower(), srt_index).
    """
    candidates: Dict[Tuple[str, int], Candidate] = {}

    def _add(text: str, idx: int, hint: str, ctx: str):
        text = text.strip().rstrip(",.;:!?")
        if not text or len(text) < 2:
            return
        key = text.lower()
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = Candidate(text=text, srt_index=idx, hint=hint, context=ctx)
        else:
            # Keep the EARLIEST occurrence so the first-mention entity
            # anchors as a news-hook in the opening minute.
            if idx < existing.srt_index:
                existing.srt_index = idx
                existing.context = ctx
            # First-detection-wins on hint: the more specific regexes
            # (_PLACE_PREP_RE, _DATE_RE) run before generic ones
            # (_SENTENCE_START_RE). Allowing the generic pass to "promote"
            # a specific place/date into a person brand would silently
            # break typing (e.g. "Zagreb" matched by SENTENCE_START in a
            # later SRT would demote the place hint to person).

    for seg in segments:
        text = seg.text
        if not text:
            continue

        # 1. Person with title: "Dr. Smith", "Senator Jones"
        for m in _PERSON_TITLES_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(1).strip(), seg.index, "person", ctx)

        # 2. Place after preposition: "in Paris", "from Wales"
        for m in _PLACE_PREP_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(1).strip(), seg.index, "place", ctx)

        # 3. Place suffix
        for m in _PLACE_SUFFIX_RE.finditer(text):
            if _proper_noun_at_sentence_start(text, m):
                continue
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(0), seg.index, "place", ctx)

        # 4. Year (date) and decade
        for m in _DATE_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(1), seg.index, "date", ctx)
        for m in _DECADE_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(1) + "0s", seg.index, "date", ctx)

        # 5. Percentages / currency / number+unit (info)
        for m in _PERCENT_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(0), seg.index, "info", ctx)
        for m in _CURRENCY_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(0), seg.index, "info", ctx)
        for m in _NUMBER_UNIT_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(0), seg.index, "info", ctx)
        for m in _BIG_NUMBER_RE.finditer(text):
            ctx = _context_window(text, m.start(), m.end())
            _add(m.group(0), seg.index, "info", ctx)

        # 6. Multi-word proper nouns (catch-all)
        for m in _PROPER_NOUN_RE.finditer(text):
            matched = m.group(1).strip()
            if _proper_noun_at_sentence_start(text, m):
                continue
            # Skip very common non-entity words
            if matched.lower() in {"united states", "new york", "los angeles"}:
                # Still add — likely place
                ctx = _context_window(text, m.start(), m.end())
                _add(matched, seg.index, "place", ctx)
                continue
            # Default to person when multi-word; place suffix overrides via _add
            ctx = _context_window(text, m.start(), m.end())
            _add(matched, seg.index, "person", ctx)

        # 7. Brands — capitalized name + verb, OR sentence-start capitalized.
        # Brands go into the person hint and live on V1 - Lower Thirds · Names.
        for m in _BRAND_VERB_RE.finditer(text):
            name = m.group(1).strip()
            first_word = name.split()[0].lower()
            if first_word in _SENTENCE_START_STOPWORDS:
                continue
            ctx = _context_window(text, m.start(), m.end())
            _add(name, seg.index, "person", ctx)

        for m in _SENTENCE_START_RE.finditer(text):
            name = m.group(1).strip()
            first_word = name.split()[0].lower()
            if first_word in _SENTENCE_START_STOPWORDS:
                continue
            if len(name) < 3:
                continue
            ctx = _context_window(text, m.start(), m.end())
            _add(name, seg.index, "person", ctx)

    return sorted(candidates.values(), key=lambda c: (c.srt_index, c.hint))


def _context_window(text: str, start: int, end: int, width: int = 60) -> str:
    """Return ~width chars of surrounding text, trimmed to word boundaries."""
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    snippet = text[lo:hi].replace("\n", " ")
    if lo > 0:
        snippet = "..." + snippet[snippet.find(" ") + 1:]
    if hi < len(text):
        # Trim trailing partial word
        last_space = snippet.rfind(" ")
        if last_space > 0 and last_space > len(snippet) - 15:
            snippet = snippet[:last_space] + "..."
    return snippet


CANDIDATE_SYSTEM_PROMPT = (
    "You filter and classify a list of candidate phrases extracted from a "
    "voiceover transcript. For each candidate, decide if it's a notable "
    "on-screen entity worth showing as a lower-third graphic. The system "
    "already knows WHERE each candidate appears in the transcript — your job "
    "is ONLY to include/exclude and pick the entity_type.\n\n"
    "Output ONLY a JSON array. No commentary, no markdown fences, no "
    "top-level object.\n\n"
    "Each entry:\n"
    '  {"idx": <int>, "include": true/false, "entity_type": "person"|"place"|"date"|"info", "role": "<optional>"}\n\n'
    '"idx" MUST match the candidate number shown in the prompt.\n\n'
    "Per-type rules:\n"
    '- "person": a real person the narrator names, OR a brand/company name. '
    'Brands (Uber, Waymo, Apollo GO, Baidu, etc.) go here too.\n'
    '- "place": a proper-noun location (city, landmark, building, country, region). '
    'Include any named place the narrator mentions — even common ones like "United States", '
    '"China", or "London". The viewer needs to see WHERE the story is happening.\n'
    '- "date": a specific year or named historical date the narrator emphasizes. '
    'For dates, set "role" to a short contextual tag (e.g. "Trust founded", '
    '"Groundbreaking"). Limit to 5 words.\n'
    '- "info": a short, on-screen fact callout — striking statistic, '
    'headline number, or surprising claim. Set "role" to source/unit if '
    'applicable.\n\n'
    "Be INCLUSIVE of real entities. The first minute of a news/video story is the news-hook — "
    "viewers tune in or leave based on what they see in the opening 30 seconds. Strongly prefer "
    "to include: years (2026, 1990), named places (Zagreb, Croatia), brand names (Uber, Waymo), "
    "and any number + unit (2 ,000 robotaxies, 4 ,000 vehicles). These are the things an editor "
    "wants on screen as the story opens.\n\n"
    "Skip ONLY:\n"
    "- obvious non-entities: 'the', 'a', 'and', or other pure function words\n"
    "- generic role-only phrases like 'Local', 'Governments', 'Instead' that don't name "
    "a specific thing\n\n"
    "If nothing is worth showing, return []."
)


def build_candidate_prompt(candidates: List[Candidate]) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the candidate-classification pass.

    The user_prompt lists each candidate by index, text, type hint, and
    surrounding context so the LLM can decide include/exclude + correct type.
    """
    if not candidates:
        return CANDIDATE_SYSTEM_PROMPT, "(no candidates — return [])"
    lines = ["Candidates:"]
    for i, c in enumerate(candidates, start=1):
        ctx = c.context.replace("\n", " ")
        lines.append(f'{i}. [srt {c.srt_index}, hint={c.hint}] "{c.text}"  :: {ctx}')
    user_prompt = "\n".join(lines) + "\n\nReturn ONLY the JSON array."
    return CANDIDATE_SYSTEM_PROMPT, user_prompt


def _sanitize_chapter_title(title: str) -> str:
    """Clean up LLM-generated chapter titles.

    Two failure modes observed in src/chapter_detection/detector.py output:
    1. Duplication: "second - second quarter ride" → first half repeats a
       token in the second half. Strip the first "X - " segment if the LHS
       is short (<=15 chars) and a word from it reappears in the RHS.
    2. Over-length: titles >40 chars get truncated at the last word boundary.
    """
    if not title:
        return ""
    parts = title.split(" - ")
    if len(parts) >= 2:
        first = parts[0].strip()
        rest = " - ".join(parts[1:]).strip()
        # If the first segment is short and any of its words reappears in
        # the rest, treat it as a duplicated prefix.
        if 0 < len(first) <= 15:
            first_words = {w.lower().strip(".,;:") for w in first.split()}
            rest_words = {w.lower().strip(".,;:") for w in rest.split()}
            if first_words & rest_words:
                title = rest
    # Cap at 40 chars on a word boundary so long titles stay readable.
    if len(title) > 40:
        words = title.split()
        out, total = [], 0
        for w in words:
            if total + len(w) + (1 if out else 0) > 40:
                break
            out.append(w)
            total += len(w) + (1 if out else 0)
        title = " ".join(out).rstrip(",;:")
    return title.strip()


def _normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace + strip punctuation, for fuzzy word match.

    Used by `_word_onset` to align entity names (often punctuated: "U.S.",
    "Apollo GO", "AI33") with raw faster-whisper tokens (no punctuation,
    case preserved).
    """
    if not text:
        return ""
    lowered = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def load_word_timings_by_idx(srt_path: Path, segments: List[SRTSegment]) -> Dict[int, List[dict]]:
    """Build {srt_index: [word_dicts]} from the .words.json sidecar.

    Sidecar is a JSON list of faster-whisper segments, each with
    `start`/`end`/`text`/`words: [{word, start, end, confidence}]`.
    Loaded by `subtitle_design.load_word_timestamps(srt_path)`.

    Mapping strategy: for each SRT segment, find words whose `start` falls
    inside [seg.start_time, seg.end_time] (time-window match). This is robust
    against faster-whisper re-segmentation producing a different number of
    segments than the original SRT.

    Returns empty dict if sidecar missing — callers should fall back to
    segment-level anchoring.
    """
    words_path = Path(srt_path).with_suffix(".words.json")
    if not words_path.exists():
        return {}
    try:
        from scripts.subtitle_design import load_word_timestamps  # local import; avoids cycle
        words_data = load_word_timestamps(Path(srt_path))
    except (ImportError, OSError, ValueError):
        logger.warning(
            f"Could not load word timestamps from {words_path}; "
            f"falling back to segment-level anchoring"
        )
        return {}
    if not words_data:
        return {}

    out: Dict[int, List[dict]] = {}
    for seg in segments:
        words: List[dict] = []
        for entry in words_data:
            seg_words = entry.get("words") or []
            for w in seg_words:
                try:
                    w_start = float(w.get("start", 0.0))
                except (TypeError, ValueError):
                    continue
                # +0.001s tolerance — faster-whisper word starts can sit a
                # hair after the SRT seg end on adjacent-segment boundaries.
                if seg.start_time - 0.001 <= w_start <= seg.end_time + 0.001:
                    words.append(w)
        # Stable order by start time so `_word_onset` walks left-to-right.
        words.sort(key=lambda w: float(w.get("start", 0.0)))
        out[seg.index] = words
    return out


def _word_onset(entity_name: str, seg_words: List[dict], seg_start: float) -> Optional[float]:
    """Find the start time of the first word matching entity_name.

    Returns None if no confident match — caller falls back to seg.start_time.

    Matching strategy:
      1. Flat-tokenize the segment's words (each word may contribute 1+
         tokens after punctuation stripping — e.g. "Baidu's" -> ["baidu", "s"])
         and walk a sliding window of len(target_tokens) flat tokens. Match
         exact equality.
      2. Fallback: case-insensitive match on the first token only (handles
         edge cases like "U.S." where the model doesn't emit "U" + "S" as
         separate words, OR contraction-bearing names where flat-token
         alignment differs from raw word alignment).
    """
    if not entity_name or not seg_words:
        return None
    target_tokens = _normalize_text(entity_name).split()
    if not target_tokens:
        return None

    # Build flat token list with each token's source word dict (for the start
    # time). One sidecar word may produce multiple tokens after punctuation
    # strip ("Baidu's" -> "baidu s" -> ["baidu", "s"]); all share the
    # parent word's start time.
    flat: List[Tuple[str, dict]] = []
    for w in seg_words:
        try:
            conf = float(w.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        if conf < 0.3:
            continue
        for tok in _normalize_text(w.get("word", "")).split():
            flat.append((tok, w))

    # Strategy 1: multi-token sliding window on flat token list
    n = len(target_tokens)
    for i in range(len(flat) - n + 1):
        window = [flat[j][0] for j in range(i, i + n)]
        if window == target_tokens:
            try:
                return float(flat[i][1].get("start", seg_start))
            except (TypeError, ValueError):
                return seg_start

    # Strategy 2: first-token-only match (handles punctuation-fragmented names)
    first = target_tokens[0]
    for tok, w in flat:
        if tok == first:
            try:
                return float(w.get("start", seg_start))
            except (TypeError, ValueError):
                return seg_start
    return None


def _normalize_entity_type(raw: Optional[str]) -> Optional[EntityType]:
    """Normalize a model's `entity_type` field to a canonical EntityType.

    Accepts canonical values ("person"/"place"/"date"/"info"), common aliases
    ("people", "location", "year", "fact", ...), and returns None when the
    value is missing or unparseable so the caller can decide the fallback.
    """
    if raw is None:
        return None
    cleaned = str(raw).strip().lower()
    if not cleaned:
        return None
    # Direct match first
    try:
        return EntityType(cleaned)
    except ValueError:
        pass
    # Alias match
    if cleaned in _ENTITY_TYPE_ALIASES:
        return _ENTITY_TYPE_ALIASES[cleaned]
    return None


def parse_template_by_type(spec: str) -> Dict[EntityType, str]:
    """Parse 'person=classic,date=boxed,info=modern' → {EntityType.PERSON: 'classic', ...}.

    - Whitespace around keys/values stripped
    - Unknown entity_type values log + skip (don't fail the whole run)
    - Unknown template names log + skip (fall back to --template default)
    - Empty / malformed entries log + skip
    - Empty input string returns {}
    """
    out: Dict[EntityType, str] = {}
    if not spec:
        return out
    for raw_kv in spec.split(","):
        kv = raw_kv.strip()
        if not kv or "=" not in kv:
            logger.error(f"Template-by-type: malformed entry {kv!r}; skipping")
            continue
        k, v = (s.strip() for s in kv.split("=", 1))
        norm = _normalize_entity_type(k)
        if norm is None:
            logger.error(
                f"Template-by-type: unknown entity_type {k!r}; "
                f"valid: {[t.value for t in EntityType]}"
            )
            continue
        if v not in TEMPLATE_REGISTRY:
            logger.error(
                f"Template-by-type: unknown template {v!r}; "
                f"valid: {sorted(TEMPLATE_REGISTRY.keys())}"
            )
            continue
        out[norm] = v
    return out


def _find_entity_segments(
    name: str,
    segments: List[SRTSegment],
    entity_type: EntityType,
) -> List[int]:
    """Find SRT indices where the entity name appears, using word-boundary matching.

    Strategy (first hit wins):
    1. Full-name word-boundary substring match (case-insensitive) — covers
       proper nouns and verbatim phrases.
    2. For DATE: try the first 4-digit number (e.g. "1991") as a fallback.
    3. For INFO: try the first 2 words, then the first word — covers
       paraphrased facts where the first noun still appears literally.
    4. Return [] if nothing matches — caller should drop the entity as a
       likely hallucination.

    Returns at most one index (the first verified segment) so the resulting
    lower-third has a sensible on-screen duration instead of spanning the
    entire transcript when the entity is mentioned multiple times.
    """
    needle = name.strip().lower()
    if not needle:
        return []

    # 1. Full word-boundary match
    pattern = r"\b" + re.escape(needle) + r"\b"
    for s in segments:
        if re.search(pattern, s.text.lower()):
            return [s.index]

    # 2. DATE — try the first 4-digit year
    if entity_type == EntityType.DATE:
        m = re.search(r"\b(\d{4})\b", name)
        if m:
            year = m.group(1)
            year_pat = r"\b" + re.escape(year) + r"\b"
            for s in segments:
                if re.search(year_pat, s.text):
                    return [s.index]

    # 3. INFO — try the first 2 words, then the first word (paraphrase-friendly)
    if entity_type == EntityType.INFO:
        words = name.split()
        for n in (2, 1):
            if len(words) >= n:
                first = " ".join(words[:n]).strip(".,;:!?\"'").lower()
                if not first:
                    continue
                pattern = r"\b" + re.escape(first) + r"\b"
                for s in segments:
                    if re.search(pattern, s.text.lower()):
                        return [s.index]

    return []


def _extract_json_payload(text: str):
    """Extract the top-level JSON array OR object from a noisy LLM response.

    Small models (e.g. Mistral 7B) sometimes return a flat object whose keys
    are entity "names" and whose values are short descriptions, e.g.
        {"Paris": "France capital", "1946": "Trust founded"}
    instead of a proper array. We accept both shapes here so the rest of the
    parser can normalise them uniformly.

    When the model returns MULTIPLE JSON arrays (one per candidate), all of
    them are concatenated into a single list — this happens when small models
    fail to follow the "single top-level array" instruction.
    """
    arrays: List[list] = []
    objects: List[dict] = []
    n = len(text)
    i = 0
    while i < n:
        # Skip whitespace
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        ch = text[i]
        if ch not in "[{":
            i += 1
            continue
        open_ch = ch
        close_ch = "]" if ch == "[" else "}"
        start = i
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(i, n):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            # No matching close — try salvage as object literals
            break
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            i = end + 1
            continue
        if isinstance(parsed, list):
            arrays.append(parsed)
        elif isinstance(parsed, dict):
            objects.append(parsed)
        i = end + 1

    if arrays:
        merged: List = []
        for a in arrays:
            for item in a:
                # Some small models wrap object literals as strings inside the
                # outer array, e.g. ["{...}", "{...}"]. Unwrap them.
                if (
                    isinstance(item, str)
                    and item.strip().startswith("{")
                    and item.strip().endswith("}")
                ):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        pass
                merged.append(item)
        return merged
    if objects:
        # Legacy: a single flat object — keys are names, values are roles.
        if len(objects) == 1:
            return objects[0]
        merged_d: Dict = {}
        for o in objects:
            merged_d.update(o)
        return merged_d
    return None


def _extract_truncated_array_objects(text: str) -> List[dict]:
    """Fallback: when the model emits a malformed/truncated array, scan for
    individual complete `{ ... }` object literals and parse each separately.

    This salvages a few real entities from a JSON response that was cut off
    mid-array by the model's max_tokens ceiling. We only accept objects whose
    JSON parses cleanly AND that have a string-typed `name` field, to avoid
    pulling random inner braces out of prose.
    """
    results: List[dict] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        # Try to find the matching close brace, respecting string boundaries.
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break  # no further complete objects; remaining text is truncated
        candidate = text[i : end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            i = end + 1
            continue
        # Accept either legacy (name) or new (idx) entry shape
        if isinstance(obj, dict) and (
            isinstance(obj.get("name"), str) or isinstance(obj.get("idx"), int)
        ):
            results.append(obj)
        i = end + 1
    return results


_FLAT_ARRAY_SCHEMA_KEYS: Tuple[str, ...] = (
    "idx", "include", "entity_type", "role", "name",
)


def _repair_flat_array_of_dicts(text: str) -> str:
    """Repair pattern: flat `[ "k": v, ..., "k": v, ... ]` missing all `{}`.

    Observed with llama3.2: the model emits a single `[...]` array where every
    candidate's keys/values are comma-separated inline, with no object wrappers
    at all:

        ["idx": 1, "include": true, "entity_type": "info", "role": " ",
         "idx": 2, "include": false, ...,
         "idx": 8, "include": true, ...]

    Expected (valid JSON):

        [{"idx": 1, "include": true, ...},
         {"idx": 2, "include": false, ...},
         ...,
         {"idx": 8, "include": true, ...}]

    The boundary between objects is `, "key":` where `key` is one of the schema
    keys. We insert `{` right after the opening `[` (before the leading `"`),
    and replace every boundary with `}, {"key":`. The closing `]` is preserved
    when present; otherwise we append `}]`.
    """
    if not text or text[0] != "[":
        return text
    if "}" in text:
        # Some braces are already present — let the caller fall back to the
        # single-missing-brace repair path, which handles those cases.
        return text

    # 1. Insert `{` between the opening `[` and the first `"` of the first key.
    text = re.sub(r'^\[\s*"', '[{"', text, count=1)

    # 2. Replace each `, "key":` boundary with `}, {"key":` so each candidate
    #    becomes a properly wrapped object.
    boundary_re = re.compile(
        r',\s*("' + "|".join(_FLAT_ARRAY_SCHEMA_KEYS) + r'"\s*:)'
    )
    text = boundary_re.sub(lambda m: '}, {' + m.group(1), text)

    # 3. Ensure the array closes with `}]`. If a bare `]` is at the end without
    #    a matching `}` (e.g. the last object never closed), insert `}` first.
    text = text.rstrip()
    if not text.endswith("]"):
        text = text + "}]"
    elif not text.endswith("}]"):
        text = text[:-1] + "}]"
    return text


def _repair_missing_open_brace(text: str) -> str:
    """Repair JSON where small models drop the `{` opener on array elements.

    Three patterns observed:
      1. Single object, no `{` opener:
           ["idx": 19, "include": true, "entity_type": "place", "role": ""}
         → [{"idx": 19, ...}]

      2. Multiple objects, each missing `{`, separated by `], [`:
           ["idx": 9, ...],
           ["idx": 10, ...}
         → [{"idx": 9, ...}, {"idx": 10, ...}]

      3. (Handled by _repair_flat_array_of_dicts) All candidates in ONE flat
         array, every `{}` missing — `["idx": 1, ..., "idx": 2, ..., ...]`.

    For patterns 1 & 2, we insert `{` after `[` and balance with `}` before
    the next `,` or `]`. If `]` is missing entirely, append it.
    """
    if not text or text[0] != "[":
        return text

    # Pattern 3 short-circuit: when no `}` is present anywhere, the model
    # emitted a flat array of dicts — handled by the more aggressive repair.
    if "}" not in text:
        return _repair_flat_array_of_dicts(text)

    n = len(text)
    out: List[str] = []
    i = 0
    n_inserted = 0
    while i < n:
        ch = text[i]
        # Look for `[` followed (after whitespace) by `"` instead of `{`.
        if ch == "[":
            # Check the next non-whitespace char
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] == '"':
                # Replace `[` with `[{` so the item becomes a proper object
                out.append("[{")
                n_inserted += 1
                i += 1
                continue
        out.append(ch)
        i += 1

    if n_inserted == 0:
        return text

    repaired = "".join(out)

    # For each `{` we inserted, ensure the object is closed before the next
    # `,` or `]`. Walk the string tracking depth.
    n = len(repaired)
    out2: List[str] = []
    i = 0
    in_str = False
    esc = False
    depth = 0
    while i < n:
        ch = repaired[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            out2.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = True
            out2.append(ch)
            i += 1
            continue
        if ch == "{":
            depth += 1
            out2.append(ch)
            i += 1
            continue
        if ch == "}":
            depth -= 1
            out2.append(ch)
            i += 1
            continue
        if ch in ",]" and depth == 0:
            # We're at the array boundary with no open object — close one
            # implicitly so the array contains valid JSON objects.
            out2.append("}")
            depth -= 1
            out2.append(ch)
            i += 1
            continue
        out2.append(ch)
        i += 1

    # If array close `]` is missing entirely, append it.
    repaired = "".join(out2)
    if "]" not in repaired[repaired.rfind("}"):]:
        repaired = repaired + "]"
    return repaired


def parse_ollama_entities(
    raw: str,
    segments: List[SRTSegment],
    words_by_idx: Optional[Dict[int, List[dict]]] = None,
) -> List[EntitySpan]:
    """Parse the Ollama response into a list of EntitySpan, deriving SRT timing by search.

    The LLM's job is to classify entities (name, role, entity_type). The parser
    independently finds where each name appears in the SRT via word-boundary
    substring search, so the LLM cannot hallucinate timings.

    When `words_by_idx` is provided (sourced from the .words.json sidecar),
    each entity's start_time is anchored to the actual spoken-word onset via
    `_word_onset` — start at the word, end at seg.end_time. Without it,
    falls back to whole-segment anchoring (start = seg.start_time).

    Tolerates: ```json fences, surrounding prose, missing keys, empty names,
    unknown entity_type (defaults to PERSON; dropped if model emits an
    unparseable value), and entities not found in the SRT (dropped as likely
    hallucinations). Any srt_indices field returned by the model is ignored.
    """
    if not raw or not raw.strip():
        return []

    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    parsed = _extract_json_payload(text)
    if parsed is None:
        # Fallback: the array may be truncated (model hit max_tokens). Salvage
        # any complete { ... } objects we can find, so the editor still gets
        # a partial result instead of a hard zero.
        salvaged = _extract_truncated_array_objects(text)
        if salvaged:
            logger.warning(
                f"Primary JSON parse failed; salvaged {len(salvaged)} object(s) "
                f"from truncated Ollama response"
            )
            # Re-use the array-handling branch below by pretending the model
            # returned a clean list.
            parsed = salvaged
        else:
            logger.warning(
                f"Could not find JSON array/object in Ollama response: {text[:200]}"
            )
            return []

    if isinstance(parsed, dict):
        # Top-level object: keys are entity names, values are short role/description.
        data: List[dict] = []
        for key, value in parsed.items():
            name = str(key).strip()
            if not name:
                continue
            if value is None:
                role = ""
            elif isinstance(value, str):
                role = value.strip()
            else:
                # Defensive: stringify non-string values (e.g. numbers) safely.
                role = str(value).strip()
            # Strip a leading em-dash / hyphen separator when the model embedded
            # the event tag into the value, e.g. "1946": " — Moon landing".
            for sep in ("\u2014 ", "- "):  # em-dash + space, hyphen + space
                if role.startswith(sep):
                    role = role[len(sep):].strip()
                    break
            data.append({"name": name, "role": role})
    elif isinstance(parsed, list):
        data = parsed
    else:
        logger.warning(
            f"Unexpected JSON root type {type(parsed).__name__} in Ollama response"
        )
        return []

    index_by_pos = {seg.index: i for i, seg in enumerate(segments)}
    entities: List[EntitySpan] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        # Guard against `"role": null` from the model — str(None) = "None"
        # which then renders as the literal word "None" on the lower-third.
        raw_role = item.get("role")
        if raw_role is None:
            role = ""
        elif isinstance(raw_role, str):
            role = raw_role.strip()
        else:
            role = str(raw_role).strip()

        # entity_type — accept "entity_type" or alias "type", normalize, fall back to PERSON
        raw_type = item.get("entity_type", item.get("type"))
        normalized = _normalize_entity_type(raw_type)
        if raw_type is not None and normalized is None:
            logger.warning(
                f"Dropping entity {name!r}: unparseable entity_type {raw_type!r}"
            )
            continue
        entity_type = normalized if normalized is not None else EntityType.PERSON

        # Safety net: some small models put the type into the `role` field instead
        # of `entity_type`. When the model left entity_type blank but the role
        # string parses as a type, promote role to entity_type and clear role.
        if normalized is None:
            role_as_type = _normalize_entity_type(role)
            if role_as_type is not None:
                logger.debug(
                    f"Promoting role={role!r} to entity_type={role_as_type.value} "
                    f"for entity {name!r} (model put type in role field)"
                )
                entity_type = role_as_type
                role = ""

        # Find the SRT segment(s) that actually mention this entity name.
        # LLMs are no longer trusted to pick srt_indices.
        indices = _find_entity_segments(name, segments, entity_type)
        if not indices:
            logger.warning(
                f"Dropping entity {name!r} (type={entity_type.value}): "
                f"not found in SRT — likely hallucination"
            )
            continue

        # Use the first verified segment for timing (one mention per entity keeps
        # the lower-third on screen for a sensible duration).
        pos = index_by_pos[indices[0]]
        seg = segments[pos]
        # Word-onset anchoring: when a .words.json sidecar is available, start
        # at the actual spoken-word onset within the segment. Falls back to
        # seg.start_time when no confident match.
        onset = _word_onset(name, (words_by_idx or {}).get(seg.index, []), seg.start_time)
        start = onset if onset is not None else seg.start_time
        end = seg.end_time

        entities.append(EntitySpan(
            name=name[:120],
            role=role[:120],
            start_time=start,
            end_time=end,
            srt_indices=indices,
            entity_type=entity_type,
        ))

    entities.sort(key=lambda e: e.start_time)
    return entities


def parse_candidate_response(
    raw: str,
    candidates: List[Candidate],
    segments: List[SRTSegment],
    words_by_idx: Optional[Dict[int, List[dict]]] = None,
) -> List[EntitySpan]:
    """Parse the Ollama response to a candidate-classification prompt.

    Each confirmed candidate is built into an EntitySpan using the
    Python-pre-computed srt_index — the LLM cannot influence timing.

    When `words_by_idx` is provided (sourced from the .words.json sidecar),
    each entity's start_time is anchored to the actual spoken-word onset via
    `_word_onset` — start at the word, end at seg.end_time. Without it,
    falls back to whole-segment anchoring (start = seg.start_time).

    Output schema expected from the model:
      [{"idx": 1, "include": true, "entity_type": "person", "role": "Mayor"},
       {"idx": 2, "include": false},
       ...]

    Tolerates:
      - ```json fences, surrounding prose
      - missing `include` (defaults to true when the field is absent)
      - missing `entity_type` (falls back to candidate.hint, then PERSON)
      - missing `role` (defaults to "")
      - unknown entity_type values (entity dropped with warning)
      - idx values that don't match (silently skipped)
    """
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    parsed = _extract_json_payload(text)
    if parsed is None:
        salvaged = _extract_truncated_array_objects(text)
        if salvaged:
            logger.warning(
                f"Primary JSON parse failed; salvaged {len(salvaged)} candidate(s)"
            )
            parsed = salvaged
        else:
            # Last-ditch salvage: small models sometimes emit
            #   ["idx": 19, "include": true, ...}
            # (missing the `{` opener on the first object inside the array).
            # Detect and repair the pattern, then retry the parser.
            repaired = _repair_missing_open_brace(text)
            if repaired and repaired != text:
                parsed = _extract_json_payload(repaired)
                if parsed is None:
                    salvaged = _extract_truncated_array_objects(repaired)
                    if salvaged:
                        logger.warning(
                            f"Repaired malformed JSON; salvaged {len(salvaged)} candidate(s)"
                        )
                        parsed = salvaged
            if parsed is None:
                logger.warning(
                    f"Could not find JSON array in candidate response: {text[:200]}"
                )
                return []
    if not isinstance(parsed, list):
        logger.warning(
            f"Unexpected JSON root type {type(parsed).__name__} in candidate response"
        )
        return []

    # Build index → candidate lookup (1-based, matches prompt numbering)
    by_idx: Dict[int, Candidate] = {i + 1: c for i, c in enumerate(candidates)}
    index_by_pos = {seg.index: i for i, seg in enumerate(segments)}
    entities: List[EntitySpan] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx_raw = item.get("idx")
        try:
            idx = int(idx_raw) if idx_raw is not None else -1
        except (TypeError, ValueError):
            continue
        candidate = by_idx.get(idx)
        if candidate is None:
            continue
        include = item.get("include", True)
        # Coerce truthy: True / "true" / "yes" / 1 / "include"
        if isinstance(include, str):
            include = include.strip().lower() in ("true", "yes", "1", "include", "y")
        if not include:
            continue

        # entity_type — accept "entity_type" or alias "type", normalize
        raw_type = item.get("entity_type", item.get("type"))
        normalized = _normalize_entity_type(raw_type)
        if raw_type is not None and normalized is None:
            logger.warning(
                f"Dropping candidate {candidate.text!r}: "
                f"unparseable entity_type {raw_type!r}"
            )
            continue
        if normalized is not None:
            entity_type = normalized
        else:
            # Fall back to the candidate's hint, normalized
            hint_norm = _normalize_entity_type(candidate.hint)
            entity_type = hint_norm if hint_norm is not None else EntityType.PERSON

        # Guard against `"role": null` from the model — str(None) = "None"
        # which then renders as the literal word "None" on the lower-third.
        raw_role = item.get("role")
        if raw_role is None:
            role = ""
        elif isinstance(raw_role, str):
            role = raw_role.strip()
        else:
            role = str(raw_role).strip()

        # Use Python's pre-computed srt_index for timing — never trust the model
        srt_index = candidate.srt_index
        pos = index_by_pos.get(srt_index)
        if pos is None:
            continue
        seg = segments[pos]
        # Word-onset anchoring: when a .words.json sidecar is available, start
        # at the actual spoken-word onset within the segment. Falls back to
        # seg.start_time when no confident match.
        onset = _word_onset(candidate.text, (words_by_idx or {}).get(seg.index, []), seg.start_time)
        start = onset if onset is not None else seg.start_time
        end = seg.end_time

        entities.append(EntitySpan(
            name=candidate.text[:120],
            role=role[:120],
            start_time=start,
            end_time=end,
            srt_indices=[srt_index],
            entity_type=entity_type,
        ))

    entities.sort(key=lambda e: e.start_time)
    return entities


def dedupe_entities(entities: List[EntitySpan]) -> List[EntitySpan]:
    """Drop later duplicates of the same name; keep the earliest occurrence.

    - Type-agnostic: collisions merge across all EntityType values (place + info
      of the same name collapse to one).
    - Case + whitespace insensitive.
    - Leading "the " / "The " stripped before comparison.
    - First occurrence (earliest start_time, then input order on ties) wins.
    - Output is sorted by start_time for a stable, timeline-ordered list.

    We deliberately do NOT merge:
    - Different proper nouns that share a token ("Peter Scott" / "Scott")
    - Synonyms ("UK" / "United Kingdom")
    - Substring overlaps ("Wales" / "New South Wales") — see
      `dedupe_substring_entities` for that pass (opt-in via pipeline order).
    Smart-merge is a separate feature; this pass is a strict name-key dedupe.
    """
    seen: Dict[str, EntitySpan] = {}
    for e in entities:
        key = e.name.strip().lower()
        if key.startswith("the "):
            key = key[4:].strip()
        if not key:
            continue
        if key in seen:
            # Only replace if this one starts strictly earlier; on ties keep
            # the first-seen one (preserves model output order).
            if e.start_time < seen[key].start_time:
                seen[key] = e
            continue
        seen[key] = e
    return sorted(seen.values(), key=lambda x: x.start_time)


def dedupe_substring_entities(entities: List[EntitySpan]) -> List[EntitySpan]:
    """Drop an entity whose normalized name is a strict substring of another.

    Applied AFTER `dedupe_entities` to catch cases where the model emitted
    both the short form ("United") and the long form ("United States"). The
    short form is dropped because it's visually noisy and gets superseded by
    the long form ~1s later — with word-onset anchoring it would also fire at
    the same word ("United" inside "United States") which is a duplicate by
    any reading.

    Trade-off (user-accepted): a person literally named "United" as a surname
    (e.g., "Mrs. United") would be dropped if "United States" exists anywhere
    in the entity list. Short-name guard (<=3 chars) keeps "AI" inside "AI33"
    from being mistakenly dropped.

    Case + leading-article insensitive. Whitespace normalized.
    """
    def _norm(name: str) -> str:
        if not name:
            return ""
        s = re.sub(r"\s+", " ", name.strip().lower())
        if s.startswith("the "):
            s = s[4:].strip()
        return s

    superset_names = {_norm(e.name) for e in entities}
    out: List[EntitySpan] = []
    dropped = 0
    for e in entities:
        e_norm = _norm(e.name)
        # Short-name guard: avoid masking "AI" inside "AI33", "UK" inside "UK Ltd".
        if len(e_norm) <= 3:
            out.append(e)
            continue
        if any(s != e_norm and e_norm in s for s in superset_names):
            dropped += 1
            continue
        out.append(e)
    if dropped:
        logger.info(
            f"Substring dedup: dropped {dropped} entity/entities "
            f"(superseded by longer name)"
        )
    return out


def extract_entities_via_ollama(
    srt_path: Path,
    model: str,
    host: str = "http://localhost:11434",
    use_pre_screening: bool = True,
    words_by_idx: Optional[Dict[int, List[dict]]] = None,
) -> Tuple[List[EntitySpan], List[SRTSegment]]:
    """Read SRT, extract entities via Python pre-screening + Ollama classification.

    Two-stage approach:
      1. Python regex/heuristics extract entity candidates from the SRT text,
         each carrying its srt_index. Timing is owned by Python — Ollama never
         invents segment indices.
      2. Ollama sees ONLY the candidate list (no full transcript) and decides
         which to include and what entity_type each is.

    When `words_by_idx` is provided (sourced from the .words.json sidecar via
    `load_word_timings_by_idx`), each entity's start_time is anchored to the
    actual spoken-word onset via `_word_onset` — start at the word, end at
    seg.end_time. Without it, falls back to whole-segment anchoring.

    Returns (entities, segments). Segments are returned alongside entities so
    callers (e.g. chapter loader) can reuse the parsed SRT without re-parsing.

    When use_pre_screening=False (legacy mode), the full SRT is sent to Ollama
    and entities are searched by string match. This is brittle on long SRTs
    where the model hallucinates names.
    """
    from src.llm_client.providers.ollama import OllamaClient, check_ollama_model_available
    from src.llm_client.base import LLMRequest, ResponseFormat

    available, err = check_ollama_model_available(model, host)
    if not available:
        logger.error(f"Ollama model '{model}' unavailable: {err}")
        return [], []

    segments = parse_srt_file(str(srt_path))
    if not segments:
        logger.error(f"SRT parsed to 0 segments: {srt_path}")
        return [], []

    if use_pre_screening:
        candidates = extract_candidates(segments)
        if not candidates:
            logger.warning(
                f"Pre-screening found 0 candidates in {len(segments)} segments — "
                "falling back to legacy full-SRT prompt"
            )
            return _legacy_extract(segments, model, host, words_by_idx=words_by_idx), segments
        logger.info(
            f"Pre-screening: extracted {len(candidates)} candidates from "
            f"{len(segments)} segments"
        )
        # Batch candidates to keep each LLM response small — small models
        # produce malformed JSON when asked to classify >~20 at once.
        BATCH_SIZE = 20
        all_entities: List[EntitySpan] = []
        for batch_idx, start in enumerate(range(0, len(candidates), BATCH_SIZE), start=1):
            batch = candidates[start:start + BATCH_SIZE]
            batch_entities = _classify_candidate_batch(
                batch, segments, model, host, batch_idx, words_by_idx=words_by_idx
            )
            all_entities.extend(batch_entities)
        logger.info(
            f"Pre-screening classification: {len(all_entities)} entities confirmed "
            f"across {((len(candidates) - 1) // BATCH_SIZE) + 1} batch(es)"
        )
        return all_entities, segments
    else:
        logger.info("Pre-screening disabled — sending full SRT to Ollama (legacy mode)")
        system_prompt, user_prompt = build_ollama_prompt(segments)

    combined_prompt = system_prompt + "\n\n" + user_prompt

    client = OllamaClient(model=model, host=host)
    request = LLMRequest(
        prompt=combined_prompt,
        max_tokens=2048,
        temperature=0.1,
        response_format=ResponseFormat.TEXT,
        cache_key_prefix=(
            "lowerthirds_candidates" if use_pre_screening else "lowerthirds_entities"
        ),
        timeout=300,
        # 16384 covers long SRTs (474+ segments → ~10k input tokens + 2k output).
        # llama3.2's default 4096 silently truncates and returns [].
        context_window=16384,
    )
    logger.info(f"Calling Ollama ({model}) for entity classification...")
    response = client.generate(request)
    logger.info(f"Ollama returned {len(response.text)} chars")

    if use_pre_screening:
        # Reached only when caller ignored the batched path (e.g., tests).
        return parse_candidate_response(response.text, candidates, segments), segments
    return parse_ollama_entities(response.text, segments), segments


def _classify_candidate_batch(
    batch: List[Candidate],
    segments: List[SRTSegment],
    model: str,
    host: str,
    batch_idx: int,
    words_by_idx: Optional[Dict[int, List[dict]]] = None,
) -> List[EntitySpan]:
    """Send one batch of candidates to Ollama and parse the response.

    Batches keep each request's output small, which dramatically improves
    JSON validity for small local models (llama3.2 2GB, mistral 7B).

    `words_by_idx` is forwarded to `parse_candidate_response` so each entity
    can be anchored to the spoken-word onset when a .words.json sidecar exists.
    """
    from src.llm_client.providers.ollama import OllamaClient, check_ollama_model_available
    from src.llm_client.base import LLMRequest, ResponseFormat

    available, err = check_ollama_model_available(model, host)
    if not available:
        logger.error(f"Ollama model '{model}' unavailable: {err}")
        return []

    system_prompt, user_prompt = build_candidate_prompt(batch)
    combined = system_prompt + "\n\n" + user_prompt

    client = OllamaClient(model=model, host=host)
    request = LLMRequest(
        prompt=combined,
        max_tokens=2048,
        temperature=0.1,
        response_format=ResponseFormat.TEXT,
        cache_key_prefix=f"lowerthirds_candidates_b{batch_idx}",
        timeout=300,
        context_window=16384,
    )
    logger.info(
        f"Classifying batch {batch_idx}: {len(batch)} candidates "
        f"(prompt {len(combined)} chars)"
    )
    response = client.generate(request)
    logger.info(f"Batch {batch_idx}: Ollama returned {len(response.text)} chars")

    # Re-number idx values: build_candidate_prompt uses 1-based numbering
    # relative to the batch, so parse_candidate_response's `by_idx` lookup
    # needs to match. We pass the batch as-is because the prompt numbers
    # from 1 within the batch — that's exactly how parse_candidate_response
    # expects them.
    return parse_candidate_response(response.text, batch, segments, words_by_idx=words_by_idx)


def _legacy_extract(
    segments: List[SRTSegment],
    model: str,
    host: str,
    words_by_idx: Optional[Dict[int, List[dict]]] = None,
) -> List[EntitySpan]:
    """Fallback: send full SRT to Ollama when pre-screening yields no candidates.

    `words_by_idx` is forwarded to `parse_ollama_entities` so word-onset
    anchoring still applies in legacy mode when a sidecar exists.
    """
    from src.llm_client.providers.ollama import OllamaClient, check_ollama_model_available
    from src.llm_client.base import LLMRequest, ResponseFormat

    available, err = check_ollama_model_available(model, host)
    if not available:
        return []

    system_prompt, user_prompt = build_ollama_prompt(segments)
    combined_prompt = system_prompt + "\n\n" + user_prompt

    client = OllamaClient(model=model, host=host)
    request = LLMRequest(
        prompt=combined_prompt,
        max_tokens=2048,
        temperature=0.1,
        response_format=ResponseFormat.TEXT,
        cache_key_prefix="lowerthirds_entities",
        timeout=300,
        context_window=16384,
    )
    response = client.generate(request)
    logger.info(f"Legacy Ollama returned {len(response.text)} chars")
    return parse_ollama_entities(response.text, segments, words_by_idx=words_by_idx)


def clamp_entity_durations(
    entities: List[EntitySpan],
    duration_min: float,
    word_anchored: bool = False,  # kept for back-compat with call sites; no longer changes behaviour
) -> List[EntitySpan]:
    """Extend short entities to >= duration_min.

    Every lower-third clip must hold on screen for at least duration_min so
    the viewer has time to read it. Capping rules:
      - never past the next entity's start (no visual stacking)
      - if a neighbour forces a sub-minimum window, keep natural end and log
        so the user can re-time those entities manually

    Previously this function was a no-op when `word_anchored=True`, leaving
    short SRT segments at sub-2s durations. The user's "minimum 2s" rule
    applies to all entities regardless of anchoring mode.
    """
    if not entities:
        return []

    # Sort by start_time so neighbours are predictable
    entities = sorted(entities, key=lambda x: x.start_time)

    # Epsilon guard: prevent zero/negative-duration entities that would
    # crash Pillow/FFmpeg rendering. 50ms is invisible on screen but
    # enough for the renderers.
    EPSILON = 0.05

    out: List[EntitySpan] = []
    extended = 0
    kept_short = 0
    for i, e in enumerate(entities):
        # Zero/negative guard runs first
        if e.end_time - e.start_time < EPSILON:
            out.append(EntitySpan(
                name=e.name,
                role=e.role,
                start_time=e.start_time,
                end_time=e.start_time + EPSILON,
                srt_indices=list(e.srt_indices),
                entity_type=e.entity_type,
            ))
            extended += 1
            continue

        if e.end_time - e.start_time >= duration_min:
            out.append(e)
            continue

        desired_end = e.start_time + duration_min
        next_start = entities[i + 1].start_time if i + 1 < len(entities) else None

        if next_start is not None and desired_end < next_start - 0.001:
            new_end = desired_end
            extended += 1
        elif next_start is not None and desired_end >= next_start - 0.001:
            # Neighbour is too close — can't honour the minimum without overlap.
            # Cap at next_start - 1ms so they don't visually stack.
            new_end = next_start - 0.001
            kept_short += 1
        else:
            # Tail entity — extend to the full minimum.
            new_end = desired_end
            extended += 1

        if new_end != e.end_time:
            e = EntitySpan(
                name=e.name,
                role=e.role,
                start_time=e.start_time,
                end_time=new_end,
                srt_indices=list(e.srt_indices),
                entity_type=e.entity_type,
            )
        out.append(e)

    if extended:
        logger.info(f"Clamp: extended {extended} entities to >= {duration_min}s")
    if kept_short:
        logger.info(
            f"Clamp: kept {kept_short} entities shorter than {duration_min}s "
            f"(neighbour too close; capped to avoid overlap)"
        )
    return out


def redistribute_stacked_entities(
    entities: List[EntitySpan],
    word_anchored: bool = False,
) -> List[EntitySpan]:
    """Split stacked entities (sharing srt_indices[0]) into sub-windows.

    When N entities share the same SRT index they would all start at the
    same time and visually stack in Resolve. Split the segment into N
    equal sub-windows so they appear sequentially within the original
    narration cue (timing still tied to the actual mention).

    - Stable: input order preserved within each group
    - No-op for singletons
    - Skips entities with empty srt_indices

    When `word_anchored=True`, each entity in a shared segment already has
    a distinct word-onset start (set by the parse functions). Sub-windowing
    would clobber that, so this function is a no-op in word-anchored mode.
    """
    if not entities:
        return []
    if word_anchored:
        return entities  # each entity already anchored to its word onset

    grouped: Dict[int, List[EntitySpan]] = {}
    for e in entities:
        if not e.srt_indices:
            continue
        grouped.setdefault(e.srt_indices[0], []).append(e)

    if not grouped:
        return entities

    replacements: Dict[int, EntitySpan] = {}
    redistributed = 0
    for fi, group in grouped.items():
        if len(group) <= 1:
            continue
        # Sort by (start_time, name) for stable, predictable slot assignment
        group.sort(key=lambda x: (x.start_time, x.name))
        seg_start = group[0].start_time
        seg_end = group[0].end_time
        seg_dur = seg_end - seg_start
        n = len(group)
        slot = seg_dur / n
        for i, e in enumerate(group):
            replacements[id(e)] = EntitySpan(
                name=e.name,
                role=e.role,
                start_time=seg_start + i * slot,
                end_time=seg_start + (i + 1) * slot,
                srt_indices=[fi],
                entity_type=e.entity_type,
            )
        redistributed += n

    if redistributed:
        logger.info(
            f"Redistribute: split {redistributed} stacked entities into "
            f"sub-windows within their SRT segment"
        )
    return [replacements.get(id(e), e) for e in entities]


def _load_font(template: LowerThirdTemplate, size: int) -> ImageFont.ImageFont:
    """Load a TTF font with safe fallback to PIL default."""
    candidates: List[Optional[str]] = []
    if template.font_name:
        candidates.append(template.font_name)
    candidates += ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]

    for name in candidates:
        if not name:
            continue
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue

    logger.warning("No TrueType font found — using PIL default bitmap font")
    return ImageFont.load_default()


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    """Truncate text with an ellipsis if it exceeds max_width."""
    if not text:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    ellipsis = "..."
    while text and draw.textlength(text + ellipsis, font=font) > max_width:
        text = text[:-1]
    return (text + ellipsis) if text else ellipsis


def _draw_shadow(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int],
                 radius: int, base_color: Tuple[int, int, int, int]) -> None:
    """Approximate a soft drop shadow by stacking offset semi-transparent rects."""
    if not base_color[3]:
        return
    base_alpha = base_color[3]
    offsets = [(8, 8, 0.30), (16, 16, 0.15), (24, 24, 0.08)]
    for dx, dy, factor in offsets:
        sx0, sy0, sx1, sy1 = box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy
        shadow_color = (0, 0, 0, int(base_alpha * factor))
        if radius > 0:
            draw.rounded_rectangle((sx0, sy0, sx1, sy1), radius=radius, fill=shadow_color)
        else:
            draw.rectangle((sx0, sy0, sx1, sy1), fill=shadow_color)


def render_lower_third_png(
    entity: EntitySpan,
    output_path: Path,
    template: LowerThirdTemplate,
) -> Path:
    """Render a single lower-third PNG and return the path."""
    img = Image.new("RGBA", (template.width, template.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    name_font = _load_font(template, template.name_font_size)
    role_font = _load_font(template, template.role_font_size)

    layout = template.layout

    if layout == Layout.UNDERLINE:
        name_text = _fit_text(draw, entity.name, name_font, template.width - 2 * template.pad_x)
        role_text = (
            _fit_text(draw, entity.role, role_font, template.width - 2 * template.pad_x)
            if entity.role else ""
        )
        total_h = template.name_font_size + (template.line_gap + template.role_font_size if role_text else 0)
        base_y = template.height - template.y_offset_from_bottom - total_h
        name_y = base_y
        draw.text((template.pad_x, name_y), name_text, font=name_font,
                  fill=template.name_color + (255,))
        if role_text:
            role_y = name_y + template.name_font_size + template.line_gap
            draw.text((template.pad_x, role_y), role_text, font=role_font,
                      fill=template.role_color + (255,))
        underline_y = (role_y + template.role_font_size + 8) if role_text else (name_y + template.name_font_size + 8)
        underline_x0 = template.pad_x
        underline_x1 = min(template.width - template.pad_x,
                           underline_x0 + max(60, int(draw.textlength(name_text, font=name_font) * 0.6)))
        draw.rectangle((underline_x0, underline_y, underline_x1, underline_y + 3),
                       fill=template.accent_color + (255,))

    elif layout == Layout.PILL:
        text_x_pad = template.pad_x + template.accent_width
        name_text = _fit_text(draw, entity.name, name_font,
                              template.pill_max_width - 2 * text_x_pad)
        role_text = (
            _fit_text(draw, entity.role, role_font,
                      template.pill_max_width - 2 * text_x_pad)
            if entity.role else ""
        )
        total_h = template.name_font_size + (template.line_gap + template.role_font_size if role_text else 0)
        pill_h = total_h + 2 * template.pad_y
        pill_w = template.pill_max_width
        pill_x0 = (template.width - pill_w) // 2
        pill_y0 = template.height - template.y_offset_from_bottom - pill_h
        pill_box = (pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h)
        if template.shadow:
            _draw_shadow(draw, pill_box, template.pill_radius, template.bar_color)
        draw.rounded_rectangle(pill_box, radius=template.pill_radius, fill=template.bar_color)
        text_x = pill_x0 + text_x_pad
        name_y = pill_y0 + template.pad_y
        draw.text((text_x, name_y), name_text, font=name_font,
                  fill=template.name_color + (255,))
        if role_text:
            role_y = name_y + template.name_font_size + template.line_gap
            draw.text((text_x, role_y), role_text, font=role_font,
                      fill=template.role_color + (255,))

    elif layout == Layout.TOP_CORNER:
        bar_y0 = template.y_offset_from_top
        bar_y1 = bar_y0 + template.bar_height
        bar_w = min(template.width - 2 * template.pad_x, 900)
        bar_box = (template.pad_x, bar_y0, template.pad_x + bar_w, bar_y1)
        if template.shadow:
            _draw_shadow(draw, bar_box, 0, template.bar_color)
        draw.rectangle(bar_box, fill=template.bar_color)
        if template.accent_width > 0:
            accent_box = (template.pad_x, bar_y0,
                          template.pad_x + template.accent_width, bar_y1)
            draw.rectangle(accent_box, fill=template.accent_color + (255,))
        text_x = template.pad_x + template.accent_width
        text_max_width = bar_w - template.accent_width - template.pad_x
        name_text = _fit_text(draw, entity.name, name_font, text_max_width)
        role_text = _fit_text(draw, entity.role, role_font, text_max_width) if entity.role else ""
        if role_text:
            name_y = bar_y0 + template.pad_y
            role_y = name_y + template.name_font_size + template.line_gap
            draw.text((text_x, name_y), name_text, font=name_font,
                      fill=template.name_color + (255,))
            draw.text((text_x, role_y), role_text, font=role_font,
                      fill=template.role_color + (255,))
        else:
            total_h = template.name_font_size
            name_y = bar_y0 + (template.bar_height - total_h) // 2
            draw.text((text_x, name_y), name_text, font=name_font,
                      fill=template.name_color + (255,))

    elif layout == Layout.STACK:
        org_text = ""
        if entity.role and ":" in entity.role:
            org_text = entity.role.split(":", 1)[1].strip()
            clean_role = entity.role.split(":", 1)[0].strip()
        else:
            clean_role = entity.role
        bar_y0 = template.height - template.y_offset_from_bottom - template.bar_height
        bar_y1 = bar_y0 + template.bar_height
        bar_box = (0, bar_y0, template.width, bar_y1)
        draw.rectangle(bar_box, fill=template.bar_color)
        if template.accent_width > 0:
            accent_box = (0, bar_y0, template.accent_width, bar_y1)
            draw.rectangle(accent_box, fill=template.accent_color + (255,))
        if template.frame_thickness > 0:
            ft = template.frame_thickness
            draw.rectangle((0, bar_y0, template.width - 1, bar_y1 - 1),
                           outline=template.accent_color + (255,), width=ft)
        text_x = template.pad_x + template.accent_width
        text_max_width = template.width - text_x - template.pad_x
        name_text = _fit_text(draw, entity.name, name_font, text_max_width)
        name_y = bar_y0 + template.pad_y
        draw.text((text_x, name_y), name_text, font=name_font,
                  fill=template.name_color + (255,))
        cursor_y = name_y + template.name_font_size + template.line_gap
        if clean_role:
            role_text = _fit_text(draw, clean_role, role_font, text_max_width)
            draw.text((text_x, cursor_y), role_text, font=role_font,
                      fill=template.role_color + (255,))
            cursor_y += template.role_font_size + 2
        if org_text:
            org_font = _load_font(template, max(20, template.role_font_size - 6))
            org_display = _fit_text(draw, org_text, org_font, text_max_width)
            draw.text((text_x, cursor_y), org_display, font=org_font,
                      fill=template.role_color + (200,))

    else:  # Layout.BAR (classic + boxed)
        bar_y0 = template.height - template.y_offset_from_bottom - template.bar_height
        bar_y1 = bar_y0 + template.bar_height
        bar_box = (0, bar_y0, template.width, bar_y1)
        if template.shadow:
            _draw_shadow(draw, bar_box, template.pill_radius, template.bar_color)
        draw.rectangle(bar_box, fill=template.bar_color)

        if template.accent_width > 0:
            accent_box = (0, bar_y0, template.accent_width, bar_y1)
            draw.rectangle(accent_box, fill=template.accent_color + (255,))

        text_x = template.pad_x + template.accent_width
        text_max_width = template.width - text_x - template.pad_x

        if entity.role:
            name_text = _fit_text(draw, entity.name, name_font, text_max_width)
            role_text = _fit_text(draw, entity.role, role_font, text_max_width)
            name_y = bar_y0 + template.pad_y
            role_y = name_y + template.name_font_size + template.line_gap
            draw.text((text_x, name_y), name_text, font=name_font, fill=template.name_color + (255,))
            draw.text((text_x, role_y), role_text, font=role_font, fill=template.role_color + (255,))
        else:
            name_text = _fit_text(draw, entity.name, name_font, text_max_width)
            total_h = template.name_font_size
            name_y = bar_y0 + (template.bar_height - total_h) // 2
            draw.text((text_x, name_y), name_text, font=name_font, fill=template.name_color + (255,))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), format="PNG", optimize=False)
    return output_path


def _build_track_for_group(
    triples: List[Tuple[EntitySpan, Path, str]],
    track_name: str,
    frame_rate: float,
) -> otio.schema.Track:
    """Build a single per-category track from (entity, media_path, template_name) triples.

    Clips are ordered by entity.start_time; gaps fill the spaces between
    non-contiguous entities so the track duration equals the last entity's end.
    Each clip's metadata carries `entity`, `template`, and `entity_type` for
    Resolve filtering.
    """
    track = otio.schema.Track(name=track_name, kind=otio.schema.TrackKind.Video)
    track.enabled = True
    track.color = None
    track.metadata["Resolve_OTIO"] = {"Locked": False}

    if not triples:
        return track

    sorted_triples = sorted(triples, key=lambda t: t[0].start_time)
    current_time = 0.0
    for idx, (entity, media_path, template_name) in enumerate(sorted_triples, start=1):
        start = entity.start_time
        end = entity.end_time
        duration = max(0.001, end - start)
        duration_frames = max(1, int(round(duration * frame_rate)))

        gap_duration = start - current_time
        if gap_duration > 0.001:
            gap_frames = max(1, int(round(gap_duration * frame_rate)))
            gap = otio.schema.Gap(
                source_range=TimeRange(
                    start_time=RationalTime(0, frame_rate),
                    duration=RationalTime(gap_frames, frame_rate),
                )
            )
            gap.enabled = True
            gap.color = None
            track.append(gap)

        safe_path = _to_windows_path(str(media_path))
        media_ref = otio.schema.ExternalReference(
            target_url=safe_path,
            available_range=TimeRange(
                start_time=RationalTime(0, frame_rate),
                duration=RationalTime(duration_frames, frame_rate),
            ),
        )
        media_ref.name = media_path.name

        clip_name = f"LT:{idx:03d}:{entity.name[:40]}"
        clip = otio.schema.Clip(
            name=clip_name,
            media_reference=media_ref,
            source_range=TimeRange(
                start_time=RationalTime(0, frame_rate),
                duration=RationalTime(duration_frames, frame_rate),
            ),
        )
        clip.active_media_reference_key = "DEFAULT_MEDIA"
        clip.media_references = {"DEFAULT_MEDIA": media_ref}
        clip.enabled = True
        clip.color = None

        ext = media_path.suffix.lower()
        if ext in _IMAGE_EXTS:
            file_kind = "image"
        elif ext in _VIDEO_EXTS:
            file_kind = "video"
        else:
            logger.warning(
                f"Unknown extension {ext!r} for {media_path.name}; "
                "treating as image with FreezeFrame"
            )
            file_kind = "image"
        # FreezeFrame is for stills. Video files (ProRes 4444 .mov with the
        # slide/pop/fade animation) must play as video — never freeze them.
        if file_kind == "image":
            clip.effects.append(otio.schema.FreezeFrame())

        clip.metadata["Resolve_OTIO"] = {}
        clip.metadata["entity"] = {
            "name": entity.name,
            "role": entity.role,
            "entity_type": entity.entity_type.value,
            "start_time": entity.start_time,
            "end_time": entity.end_time,
            "srt_indices": entity.srt_indices,
            "media_file": media_path.name,
            "media_kind": file_kind,
            "template": template_name,
        }

        track.append(clip)
        current_time = end

    return track


def build_lower_thirds_otio(
    entities: List[EntitySpan],
    media_paths: List[Path],
    template_names: List[str],
    output_path: Path,
    frame_rate: float = RATE,
) -> otio.schema.Timeline:
    """Build a multi-track standalone OTIO timeline (one track per category).

    Tracks emitted (only those with >= 1 entity, in TRACK_ORDER):
      V1 - Lower Thirds · Names       (EntityType.PERSON)
      V2 - Lower Thirds · Dates       (EntityType.DATE)
      V3 - Lower Thirds · Key Points  (EntityType.INFO)
      V4 - Lower Thirds · Places      (EntityType.PLACE)

    Each track has its own clip+gap timeline; no cross-track interaction.
    `template_names` is parallel to `entities` / `media_paths` and is recorded
    in each clip's metadata under `entity.template` for Resolve filtering.
    """
    if not (len(entities) == len(media_paths) == len(template_names)):
        raise ValueError(
            f"entities ({len(entities)}), media_paths ({len(media_paths)}), "
            f"template_names ({len(template_names)}) must all match"
        )

    timeline = otio.schema.Timeline(
        name="lowerthirds_timeline",
        metadata={"Resolve_OTIO": {"Resolve OTIO Meta Version": "1.0"}},
    )
    timeline.global_start_time = RationalTime(0, frame_rate)
    timeline.tracks.name = ""

    if not entities:
        logger.warning("No entities — OTIO will have no tracks")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(timeline, str(output_path))
        return timeline

    # Group (entity, media_path, template_name) by category
    triples_by_type: Dict[EntityType, List[Tuple[EntitySpan, Path, str]]] = {
        t: [] for t in TRACK_ORDER
    }
    for entity, media_path, template_name in zip(entities, media_paths, template_names):
        triples_by_type.setdefault(entity.entity_type, []).append(
            (entity, media_path, template_name)
        )

    written_track_names: List[str] = []
    for entity_type in TRACK_ORDER:
        group = triples_by_type.get(entity_type, [])
        if not group:
            continue
        track_name = TRACK_NAME_BY_TYPE[entity_type]
        track = _build_track_for_group(group, track_name, frame_rate)
        timeline.tracks.append(track)
        written_track_names.append(f"{track_name} ({len(group)})")

    if written_track_names:
        logger.info(f"Wrote {len(written_track_names)} tracks: " + ", ".join(written_track_names))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(timeline, str(output_path))
    return timeline


def _seconds_to_srt_tc(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm SRT timestamp."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# ─────────────────────────────────────────────────────────────────────────────
# HTML / animation mode (video output via Playwright + FFmpeg)
# ─────────────────────────────────────────────────────────────────────────────

def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _rgba_to_css(rgba: Tuple[int, int, int, int]) -> str:
    """Convert RGBA tuple to CSS rgba() string."""
    r, g, b, a = rgba
    return f"rgba({r}, {g}, {b}, {a / 255:.3f})"


def _font_family_css(template: LowerThirdTemplate) -> str:
    if template.font_name:
        return f'"{template.font_name}", Arial, sans-serif'
    return "Arial, Helvetica, sans-serif"


def build_html_template(
    entity: EntitySpan,
    template: LowerThirdTemplate,
    style: str,
    anim_duration_s: float,
) -> str:
    """Build a self-contained HTML page that renders a lower-third with CSS animation.

    The page is transparent (no body background). The DOM/CSS is dispatched on
    `template.layout` so each named template produces its distinct visual. Use
    the Web Animations API on the client to scrub frames.
    """
    if style not in ANIMATION_STYLES:
        raise ValueError(f"Unknown animation style {style!r}; valid: {ANIMATION_STYLES}")

    bar_bg = _rgba_to_css(template.bar_color)
    accent_color = "rgb({}, {}, {})".format(*template.accent_color)
    name_color = "rgb({}, {}, {})".format(*template.name_color)
    role_color = "rgb({}, {}, {})".format(*template.role_color)
    font_family = _font_family_css(template)
    name_html = f'<div class="name">{_html_escape(entity.name)}</div>'

    org_text = ""
    if template.layout == Layout.STACK and entity.role and ":" in entity.role:
        org_text = entity.role.split(":", 1)[1].strip()
        role_for_html = entity.role.split(":", 1)[0].strip()
    else:
        role_for_html = entity.role
    role_html = (
        f'<div class="role">{_html_escape(role_for_html)}</div>' if role_for_html else ""
    )
    org_html = (
        f'<div class="org">{_html_escape(org_text)}</div>' if org_text else ""
    )

    # ── per-layout CSS ─────────────────────────────────────────────────────
    if template.layout == Layout.UNDERLINE:
        bar_top = template.height - template.y_offset_from_bottom
        css = _css_underline(template, accent_color, name_color, role_color, font_family, bar_top)
        body = (
            f'<div class="text-wrap">{name_html}{role_html}</div>'
            f'<div class="underline"></div>'
        )
    elif template.layout == Layout.PILL:
        css = _css_pill(template, bar_bg, accent_color, name_color, role_color, font_family)
        body = (
            f'<div class="bar"></div>'
            f'<div class="text-wrap">{name_html}{role_html}</div>'
        )
    elif template.layout == Layout.TOP_CORNER:
        css = _css_top_corner(template, bar_bg, accent_color, name_color, role_color, font_family)
        body = (
            f'<div class="bar"></div>'
            f'<div class="accent"></div>'
            f'<div class="text-wrap">{name_html}{role_html}</div>'
        )
    elif template.layout == Layout.STACK:
        css = _css_stack(template, bar_bg, accent_color, name_color, role_color, font_family)
        body = (
            f'<div class="bar"></div>'
            f'<div class="accent"></div>'
            f'<div class="frame"></div>'
            f'<div class="text-wrap">{name_html}{role_html}{org_html}</div>'
        )
    else:  # BAR (classic + boxed)
        css = _css_bar(template, bar_bg, accent_color, name_color, role_color, font_family)
        body = (
            f'<div class="bar"></div>'
            f'<div class="accent"></div>'
            f'<div class="text-wrap">{name_html}{role_html}</div>'
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
{css}

  body[data-style="slide"] .bar,
  body[data-style="slide"] .accent,
  body[data-style="slide"] .underline {{ transform: translateX(-100%); animation: slide-in {anim_duration_s}s ease-out forwards; }}
  body[data-style="slide"] .text-wrap {{ opacity: 0; animation: fade-in {min(0.3, anim_duration_s * 0.5):.3f}s ease-out {anim_duration_s * 0.5:.3f}s forwards; }}

  body[data-style="pop"] .bar,
  body[data-style="pop"] .accent,
  body[data-style="pop"] .underline {{ transform: scale(0); transform-origin: left center; animation: pop-in {anim_duration_s}s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }}
  body[data-style="pop"] .text-wrap {{ opacity: 0; animation: fade-in {min(0.3, anim_duration_s * 0.5):.3f}s ease-out {anim_duration_s * 0.5:.3f}s forwards; }}

  body[data-style="fade"] .bar,
  body[data-style="fade"] .accent,
  body[data-style="fade"] .underline {{ opacity: 0; animation: fade-in {anim_duration_s}s ease-out forwards; }}
  body[data-style="fade"] .text-wrap {{ opacity: 0; animation: fade-in {min(0.4, max(0.05, anim_duration_s - 0.1)):.3f}s ease-out {min(0.1, anim_duration_s * 0.25):.3f}s forwards; }}

  body[data-style="slide"] .frame,
  body[data-style="fade"] .frame {{ opacity: 0; animation: fade-in {anim_duration_s}s ease-out forwards; }}

  @keyframes slide-in {{ to {{ transform: translateX(0); }} }}
  @keyframes pop-in {{ to {{ transform: scale(1); }} }}
  @keyframes fade-in {{ to {{ opacity: 1; }} }}
</style>
</head>
<body data-style="{style}">
{body}
</body>
</html>
"""


def _css_base(template: LowerThirdTemplate) -> str:
    return f"""  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{
    background: transparent;
    width: {template.width}px;
    height: {template.height}px;
    overflow: hidden;
  }}
  .text-wrap {{ color: white; font-family: {template.font_name or 'Arial, Helvetica, sans-serif'}; }}
  .name {{
    font-size: {template.name_font_size}px;
    font-weight: bold;
    color: rgb({template.name_color[0]}, {template.name_color[1]}, {template.name_color[2]});
    line-height: 1;
  }}
  .role {{
    font-size: {template.role_font_size}px;
    color: rgb({template.role_color[0]}, {template.role_color[1]}, {template.role_color[2]});
    line-height: 1;
  }}
  .org {{
    font-size: {max(20, template.role_font_size - 6)}px;
    color: rgb({template.role_color[0]}, {template.role_color[1]}, {template.role_color[2]});
    line-height: 1;
    opacity: 0.85;
    margin-top: 2px;
  }}
"""


def _css_bar(template, bar_bg, accent_color, name_color, role_color, font_family):
    bar_top = template.height - template.y_offset_from_bottom - template.bar_height
    text_x = template.pad_x + template.accent_width
    role_padding_top = template.pad_y + template.name_font_size + template.line_gap
    box_shadow = (
        f"  .bar {{ box-shadow: 0 8px 24px rgba(0,0,0,0.45); }}\n"
        if template.shadow else ""
    )
    border_radius = (
        f"  .bar {{ border-radius: {template.pill_radius}px; }}\n"
        if template.pill_radius > 0 else ""
    )
    return _css_base(template) + f"""  .bar {{
    position: absolute;
    left: 0;
    top: {bar_top}px;
    width: 100%;
    height: {template.bar_height}px;
    background: {bar_bg};
  }}
{box_shadow}{border_radius}  .accent {{
    position: absolute;
    left: 0;
    top: {bar_top}px;
    width: {template.accent_width}px;
    height: {template.bar_height}px;
    background: {accent_color};
  }}
  .text-wrap {{
    position: absolute;
    left: {text_x}px;
    top: {bar_top + template.pad_y}px;
    font-family: {font_family};
  }}
  .role {{ margin-top: {role_padding_top - (template.pad_y + template.name_font_size)}px; }}
"""


def _css_underline(template, accent_color, name_color, role_color, font_family, bar_top):
    name_y = bar_top - template.name_font_size - (template.line_gap + template.role_font_size if template.pad_y > 0 else 0) - 20
    return _css_base(template) + f"""  .text-wrap {{
    position: absolute;
    left: {template.pad_x}px;
    top: {name_y}px;
    font-family: {font_family};
  }}
  .role {{ margin-top: {template.line_gap}px; opacity: 0.85; }}
  .underline {{
    position: absolute;
    left: {template.pad_x}px;
    top: {name_y + template.name_font_size + 12}px;
    width: 200px;
    height: 3px;
    background: {accent_color};
  }}
"""


def _css_pill(template, bar_bg, accent_color, name_color, role_color, font_family):
    text_x_pad = template.pad_x + template.accent_width
    pill_w = template.pill_max_width
    pill_h = template.bar_height
    pill_x0 = (template.width - pill_w) // 2
    pill_y0 = template.height - template.y_offset_from_bottom - pill_h
    return _css_base(template) + f"""  .bar {{
    position: absolute;
    left: {pill_x0}px;
    top: {pill_y0}px;
    width: {pill_w}px;
    height: {pill_h}px;
    background: {bar_bg};
    border-radius: {template.pill_radius}px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
  }}
  .text-wrap {{
    position: absolute;
    left: {pill_x0 + text_x_pad}px;
    top: {pill_y0 + template.pad_y}px;
    font-family: {font_family};
  }}
  .role {{ margin-top: {template.line_gap}px; opacity: 0.85; }}
"""


def _css_top_corner(template, bar_bg, accent_color, name_color, role_color, font_family):
    bar_w = min(template.width - 2 * template.pad_x, 900)
    bar_y0 = template.y_offset_from_top
    return _css_base(template) + f"""  .bar {{
    position: absolute;
    left: {template.pad_x}px;
    top: {bar_y0}px;
    width: {bar_w}px;
    height: {template.bar_height}px;
    background: {bar_bg};
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
  }}
  .accent {{
    position: absolute;
    left: {template.pad_x}px;
    top: {bar_y0}px;
    width: {template.accent_width}px;
    height: {template.bar_height}px;
    background: {accent_color};
  }}
  .text-wrap {{
    position: absolute;
    left: {template.pad_x + template.accent_width}px;
    top: {bar_y0 + template.pad_y}px;
    font-family: {font_family};
  }}
  .role {{ margin-top: {template.line_gap}px; opacity: 0.9; }}
"""


def _css_stack(template, bar_bg, accent_color, name_color, role_color, font_family):
    bar_top = template.height - template.y_offset_from_bottom - template.bar_height
    text_x = template.pad_x + template.accent_width
    return _css_base(template) + f"""  .bar {{
    position: absolute;
    left: 0;
    top: {bar_top}px;
    width: 100%;
    height: {template.bar_height}px;
    background: {bar_bg};
  }}
  .accent {{
    position: absolute;
    left: 0;
    top: {bar_top}px;
    width: {template.accent_width}px;
    height: {template.bar_height}px;
    background: {accent_color};
  }}
  .frame {{
    position: absolute;
    left: 0;
    top: {bar_top}px;
    width: 100%;
    height: {template.bar_height}px;
    border: {template.frame_thickness}px solid {accent_color};
    box-sizing: border-box;
  }}
  .text-wrap {{
    position: absolute;
    left: {text_x}px;
    top: {bar_top + template.pad_y}px;
    font-family: {font_family};
  }}
  .role {{ margin-top: {template.line_gap}px; }}
"""


def capture_animation_frames(
    html_path: Path,
    output_dir: Path,
    frame_count: int,
    fps: int,
) -> List[Path]:
    """Open HTML in headless Chromium and capture `frame_count` transparent PNGs.

    Uses the Web Animations API to scrub each animation to the exact frame time
    so the resulting PNG sequence is deterministic.
    """
    from playwright.sync_api import sync_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.goto(f"file://{html_path.resolve()}")
        page.wait_for_load_state("domcontentloaded")

        # Pause all CSS animations so we can scrub them deterministically.
        page.evaluate("""() => {
            document.getAnimations().forEach(a => a.pause());
        }""")

        for frame_idx in range(frame_count):
            time_ms = (frame_idx / fps) * 1000.0
            page.evaluate(
                """(t) => {
                    document.getAnimations().forEach(a => { a.currentTime = t; });
                }""",
                time_ms,
            )
            png_path = output_dir / f"frame_{frame_idx:04d}.png"
            page.screenshot(path=str(png_path), omit_background=True)
            saved.append(png_path)

        browser.close()
    return saved


def build_lower_third_video(
    png_seq_dir: Path,
    output_path: Path,
    total_duration_s: float,
    fps: int,
    anim_duration_s: float,
) -> Path:
    """Combine a PNG sequence (the animation) into a ProRes 4444 .mov with alpha.

    Captured frames cover `anim_duration_s`; the FFmpeg `tpad` filter extends the
    sequence to `total_duration_s` by cloning the last frame (final state).
    """
    anim_frames = max(1, int(round(anim_duration_s * fps)))
    total_frames = max(anim_frames, int(round(total_duration_s * fps)))
    hold_seconds = max(0.0, (total_frames - anim_frames) / fps)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(png_seq_dir / "frame_%04d.png"),
    ]
    if hold_seconds > 0:
        cmd += ["-vf", f"tpad=stop_mode=clone:stop_duration={hold_seconds:.3f}"]
    cmd += [
        "-c:v", "prores_ks",
        "-profile:v", "4",
        "-pix_fmt", "yuva444p10le",
        "-an",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return output_path


def render_lower_third_video(
    entity: EntitySpan,
    output_path: Path,
    template: LowerThirdTemplate,
    style: str,
    anim_duration_s: float,
    fps: int = RATE,
) -> Path:
    """End-to-end: build HTML, capture animation frames, render ProRes 4444 .mov."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_duration_s = max(0.001, entity.end_time - entity.start_time)

    # Clamp animation to half the clip so the fade-in (delay = anim*0.5,
    # duration scales with anim) finishes well before the clip ends. Without
    # this, a 0.3s entity with default anim=0.4s produces a blank frame
    # because opacity-0 text never reaches opacity-1 inside 0.3s.
    if total_duration_s > 0.05:
        anim_duration_s = max(0.05, min(anim_duration_s, total_duration_s * 0.5))

    html = build_html_template(entity, template, style, anim_duration_s)
    html_path = output_path.parent / f"_tmp_{output_path.stem}.html"
    html_path.write_text(html, encoding="utf-8")

    frame_dir = output_path.parent / f"_frames_{output_path.stem}"
    try:
        # Use ceil + 1 so the last captured frame lands at or past
        # anim_duration_s; round() can otherwise stop at frame_count-1 well
        # before the animation finishes (e.g. anim=0.085s@30fps = round 2.55
        # = 2 frames; the last frame at 33ms is invisible before the fade-in
        # completes at 85ms).
        anim_frames = max(2, int(math.ceil(anim_duration_s * fps)) + 1)
        capture_animation_frames(html_path, frame_dir, anim_frames, fps)
        build_lower_third_video(
            frame_dir, output_path, total_duration_s, fps, anim_duration_s
        )
    finally:
        for p in frame_dir.glob("frame_*.png"):
            p.unlink(missing_ok=True)
        frame_dir.rmdir()
        html_path.unlink(missing_ok=True)

    return output_path


def probe_video_duration(path: Path) -> Optional[float]:
    """Use ffprobe to read a video's duration in seconds. Returns None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None



def _print_dry_run(
    entities: List[EntitySpan],
    template: LowerThirdTemplate,
    mode: str,
    style: Optional[str],
    anim_duration_s: Optional[float],
) -> None:
    """Print a human-readable summary without writing any files."""
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  DRY RUN  |  {len(entities)} lower-third(s) would be rendered")
    layout_name = getattr(template.layout, "value", str(template.layout))
    if mode == "video" and style:
        print(f"  Mode: video  |  Template: {layout_name}  |  Style: {style}  |  Animation: {anim_duration_s:.2f}s")
    else:
        print(f"  Mode: image (Pillow, no animation)")
    print(f"  Template: {template.width}x{template.height}, bar_h={template.bar_height}, "
          f"name={template.name_font_size}px, role={template.role_font_size}px")
    print(f"  Font: {template.font_name or 'auto-detect (arial -> fallback)'}")
    print(bar)
    for i, e in enumerate(entities, start=1):
        start_tc = _seconds_to_srt_tc(e.start_time)
        end_tc = _seconds_to_srt_tc(e.end_time)
        dur = e.end_time - e.start_time
        role_str = f" - {e.role}" if e.role else ""
        ext = "mov" if mode == "video" else "png"
        print(f"\n[{i:03d}]  LT:{e.name}{role_str}")
        print(f"  timing:    {start_tc} -> {end_tc}  |  {dur:.2f}s")
        print(f"  file:      lowerthird_{i:03d}.{ext}")
        print(f"  srt_idx:   {e.srt_indices}")
    print(f"\n{bar}")
    if entities:
        last = max(e.end_time for e in entities)
        print(f"  Total timeline: {last:.2f}s ({int(last * RATE)} frames @ {RATE:.0f}fps)")
    print(f"{bar}\n")


def audit_first_minute(
    entities: List[EntitySpan],
    segments: List[SRTSegment],
    *,
    opening_s: float = 60.0,
    print_report: bool = True,
) -> Dict[str, Any]:
    """Audit lower-third coverage for the opening minute of the SRT.

    The first minute of a news/documentary video is the news hook — the
    dense opener where the narrator drops the date, location, brands,
    and big stats. This function checks whether the extracted entities
    actually cover that opener, and flags three classes of gap:

      1. anchored_in_window       — entities already placed at t < opening_s
      2. first_occurrence_misplaced — entity was emitted, but anchored to
                                      a later SRT occurrence; the first
                                      mention in the window has no
                                      lower-third
      3. missing_news_hook        — news-hook phrases in the window that
                                    weren't extracted at all

    Plus a coverage_score (0.0–1.0) and a 1-line recommendation.

    Args:
        entities: Extracted entities (post-dedupe, post-clamp).
        segments: Parsed SRT segments (from parse_srt_file).
        opening_s: Length of the opener window in seconds. Default 60.
        print_report: When True (default), prints a human-readable report
                     to stdout. Set False to suppress output (e.g. from
                     tests).

    Returns:
        Dict with keys:
            opening_seconds      : float
            segments_in_window   : int
            anchored_in_window   : list[{name, entity_type, start_time,
                                          end_time, srt_indices}]
            first_occurrence_misplaced : list[{name, entity_type,
                                                first_srt_idx,
                                                first_srt_time,
                                                emitted_at_srt_idx,
                                                emitted_at_time}]
            missing_news_hook    : list[{phrase, hint, srt_idx, srt_time}]
            missing_stats        : list[{phrase, srt_idx, srt_time}]
            coverage_score       : float in [0.0, 1.0]
            recommendation       : str

    Run from CLI:
        python scripts/lower_thirds.py <project> --audit-first-minute
    """
    window_segs = [s for s in segments if s.start_time < opening_s]
    window_end = max((s.end_time for s in window_segs), default=0.0)

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    def _present_in_entities(phrase: str, hint: Optional[str] = None) -> bool:
        """True if any emitted entity matches phrase.

        Token-based contiguous-subsequence match (stripped of punctuation),
        which is more precise than raw substring matching: avoids "Zagreb"
        being suppressed by the chapter "Zagreb's Role...".

        When `hint` is given (e.g. "place"), only entities of the same
        entity_type are considered — so "Zagreb" (a missing place) is
        NOT considered covered by a chapter that happens to mention
        Zagreb in its title.
        """
        p_norm = _norm(phrase)
        if not p_norm:
            return False
        p_tokens = re.findall(r"\w+", p_norm)
        if not p_tokens:
            return False
        for e in entities:
            if hint is not None:
                e_type = e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)
                if e_type != hint:
                    continue
            n_tokens = re.findall(r"\w+", _norm(e.name))
            for i in range(len(n_tokens) - len(p_tokens) + 1):
                if n_tokens[i:i + len(p_tokens)] == p_tokens:
                    return True
        return False

    def _first_occurrence(name: str) -> Optional[SRTSegment]:
        """Find earliest window segment whose text contains `name`.

        1. Word-boundary match (handles 95% of cases correctly).
        2. Token-sequence fallback for multi-token names with internal
           punctuation (e.g. "pony.ai" matching "pony .ai" in SRT).

        Word boundaries are enforced on BOTH passes — without them,
        "America" would falsely match "American" in SRT text.
        """
        if not name or not name.strip():
            return None
        pat = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        for s in window_segs:
            if pat.search(s.text):
                return s
        # Token-sequence fallback: split the name on whitespace/.-_,
        # then require each token to appear in order in the same segment,
        # separated by up to 3 intervening words.
        tokens = [t for t in re.split(r"[\s.\-_,]+", name) if t]
        if len(tokens) >= 2:
            first = r"\b" + re.escape(tokens[0]) + r"\b"
            rest = r"\W+(?:\w+\W+){0,3}?\b".join(
                re.escape(t) for t in tokens[1:]
            )
            pattern = first + r"\W+(?:\w+\W+){0,3}?" + rest + r"\b"
            try:
                token_pat = re.compile(pattern, re.IGNORECASE)
            except re.error:
                return None
            for s in window_segs:
                if token_pat.search(s.text):
                    return s
        return None

    # 1. anchored_in_window
    anchored = []
    for e in entities:
        if e.start_time < opening_s:
            anchored.append({
                "name": e.name,
                "entity_type": e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type),
                "start_time": round(e.start_time, 3),
                "end_time": round(e.end_time, 3),
                "srt_indices": list(e.srt_indices),
            })
    anchored.sort(key=lambda x: x["start_time"])

    # 2. first_occurrence_misplaced
    misplaced = []
    for e in entities:
        if e.start_time < opening_s:
            continue  # already in window, not misplaced
        occ = _first_occurrence(e.name)
        if occ is None:
            continue
        first_idx = occ.index
        emitted_idx = e.srt_indices[0] if e.srt_indices else None
        if emitted_idx is not None and first_idx < emitted_idx:
            misplaced.append({
                "name": e.name,
                "entity_type": e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type),
                "first_srt_idx": first_idx,
                "first_srt_time": round(occ.start_time, 3),
                "emitted_at_srt_idx": emitted_idx,
                "emitted_at_time": round(e.start_time, 3),
            })

    # 3a. missing_news_hook — date + place + brand patterns from the window
    window_text = " ".join(s.text for s in window_segs)

    # Date: 4-digit years (1800-2099)
    year_re = re.compile(r"\b(1[789]\d{2}|20\d{2})\b")
    # Place: preposition + capitalized word(s)
    place_re = re.compile(
        r"\b(?:in|at|from|to|across|into|near|around)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    )
    # Brand: words containing a dot (pony.ai, uber.com) or title-case 1-2 words
    # immediately after a verb like "began", "offers", "announced"
    brand_re = re.compile(
        r"\b(?:began|offers?|announced?|built|operates?|partners?|founded)\s+"
        r"([A-Z][A-Za-z0-9.]+(?:\s+[A-Z][A-Za-z0-9.]+)?)"
    )
    # Sentence-start proper noun: capitalized word(s) at the beginning of a
    # segment, or after a sentence terminator. Catches "Uber began" at SRT
    # start, "Waymo built" at SRT start — cases brand_re misses because the
    # verb comes AFTER the name, not before.
    sentence_start_re = re.compile(
        r"(?:^|[.!?]\s+|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})(?=\s|$|[,.])"
    )
    # Stopwords that look like proper nouns at sentence start but aren't.
    SENTENCE_START_STOPWORDS = {
        "now", "then", "the", "a", "an", "and", "but", "or", "so", "yet",
        "for", "of", "in", "on", "at", "to", "by", "with", "from",
        "as", "is", "was", "were", "are", "be", "been", "being",
        "this", "that", "these", "those", "it", "they", "we", "he", "she",
        "i", "you", "his", "her", "their", "our", "my", "your",
        "today", "yesterday", "tomorrow",
        "more", "less", "much", "many", "some", "any", "all", "most",
        "still", "also", "however", "although", "though", "because",
        "after", "before", "since", "until", "while", "when", "where",
        "five", "six", "seven", "eight", "nine", "ten",
    }

    missing_hook: List[Dict[str, Any]] = []
    seen_phrases = set()

    def _append_hook(phrase: str, hint: str, srt_idx, srt_time) -> None:
        if phrase in seen_phrases:
            return
        seen_phrases.add(phrase)
        missing_hook.append({
            "phrase": phrase,
            "hint": hint,
            "srt_idx": srt_idx,
            "srt_time": round(srt_time, 3) if srt_time is not None else None,
        })

    for m in year_re.finditer(window_text):
        phrase = m.group(1)
        if _present_in_entities(phrase, hint="date"):
            continue
        char_pos = m.start()
        seg_idx, seg_time = None, None
        cum = 0
        for s in window_segs:
            if cum + len(s.text) + 1 >= char_pos:
                seg_idx, seg_time = s.index, s.start_time
                break
            cum += len(s.text) + 1
        _append_hook(phrase, "date", seg_idx, seg_time)

    for m in place_re.finditer(window_text):
        phrase = m.group(1)
        # Skip if it's clearly a sentence-starter ("In fact, In addition")
        if phrase.lower() in {"fact", "addition", "other", "particular", "general"}:
            continue
        if _present_in_entities(phrase, hint="place"):
            continue
        char_pos = m.start()
        seg_idx, seg_time = None, None
        cum = 0
        for s in window_segs:
            if cum + len(s.text) + 1 >= char_pos:
                seg_idx, seg_time = s.index, s.start_time
                break
            cum += len(s.text) + 1
        _append_hook(phrase, "place", seg_idx, seg_time)

    for m in brand_re.finditer(window_text):
        phrase = m.group(1)
        # Brands live on V1 - Lower Thirds · Names (entity_type=PERSON),
        # so type-aware matching must accept person-typed entities. Passing
        # hint=None (no type filter) is safe here because brand names are
        # rare enough that false-positive coverage from chapter/place names
        # is unlikely.
        if _present_in_entities(phrase):
            continue
        char_pos = m.start()
        seg_idx, seg_time = None, None
        cum = 0
        for s in window_segs:
            if cum + len(s.text) + 1 >= char_pos:
                seg_idx, seg_time = s.index, s.start_time
                break
            cum += len(s.text) + 1
        _append_hook(phrase, "brand", seg_idx, seg_time)

    # Sentence-start proper noun scan: per-segment, so we know exactly which
    # SRT index owns each match without char-position math.
    for s in window_segs:
        for m in sentence_start_re.finditer(s.text):
            phrase = m.group(1)
            first_word = phrase.split()[0].lower()
            if first_word in SENTENCE_START_STOPWORDS:
                continue
            # See note above: brands map to PERSON track; hint=None is
            # safer than hint="info" for type-aware coverage.
            if _present_in_entities(phrase):
                continue
            _append_hook(phrase, "brand", s.index, s.start_time)

    # 3b. missing_stats — large numeric phrases. Voiceover SRTs often
    # insert a space inside numbers ("2 ,000"), so allow \s* between
    # the leading digits and the comma group.
    stat_re = re.compile(
        r"\b\d{1,3}(?:\s*,\s*\d{3})+(?:\s+\w+)?"
        r"|\b\d{1,3}(?:\s*,\s*\d{3})+\b"
        r"|\b\d+\s+(?:robotaxi|robotaxies|vehicle|vehicles|company|companies"
        r"|city|cities|country|countries|year|years|month|months|"
        r"service|services|operator|operators|provider|providers)\b"
    )
    missing_stats: List[Dict[str, Any]] = []
    for m in stat_re.finditer(window_text):
        phrase = m.group(0)
        if phrase in seen_phrases:
            continue
        if _present_in_entities(phrase, hint="info"):
            continue
        seen_phrases.add(phrase)
        char_pos = m.start()
        seg_idx = None
        seg_time = None
        cum = 0
        for s in window_segs:
            if cum + len(s.text) + 1 >= char_pos:
                seg_idx = s.index
                seg_time = s.start_time
                break
            cum += len(s.text) + 1
        missing_stats.append({
            "phrase": phrase,
            "srt_idx": seg_idx,
            "srt_time": round(seg_time, 3) if seg_time is not None else None,
        })

    # 4. coverage_score — gate on entities to avoid vacuous-truth inflation
    # when extraction returns nothing (no misplaced / no missing would pass
    # trivially, scoring 0.55+ for an empty extraction).
    if not entities:
        score = 0.0
    else:
        score = 0.0
        score += 0.30 if len(anchored) >= 2 else (0.15 if len(anchored) == 1 else 0.0)
        score += 0.20 if not misplaced else 0.0
        score += 0.20 if not missing_hook else 0.0
        score += 0.15 if not missing_stats else 0.0
        chapter_in_window = any(
            a.get("entity_type") == "chapter" and a["start_time"] > 0
            for a in anchored
        )
        score += 0.15 if chapter_in_window else 0.0

    # 5. recommendation
    if score >= 0.85:
        recommendation = "First-minute coverage looks solid."
    elif score >= 0.6:
        recommendation = (
            "First-minute coverage is OK but has gaps. See 'missing_news_hook' "
            "and 'first_occurrence_misplaced'."
        )
    elif score >= 0.3:
        recommendation = (
            "First-minute coverage is weak. Critical news-hook entities "
            "(date, location, brand) are missing or mis-anchored."
        )
    else:
        recommendation = (
            "First-minute coverage is poor — the news-hook opener has little "
            "or no lower-third support. Re-run with stricter pre-screening "
            "or anchor first occurrences manually."
        )

    report = {
        "opening_seconds": opening_s,
        "window_end_time": round(window_end, 3),
        "segments_in_window": len(window_segs),
        "anchored_in_window": anchored,
        "first_occurrence_misplaced": misplaced,
        "missing_news_hook": missing_hook,
        "missing_stats": missing_stats,
        "coverage_score": round(score, 2),
        "recommendation": recommendation,
    }

    if print_report:
        bar = "=" * 64
        print(f"\n{bar}")
        print(f"  FIRST-MINUTE COVERAGE AUDIT  (window: 0–{opening_s:.0f}s)")
        print(bar)
        print(f"  SRT segments in window:  {len(window_segs)}")
        print(f"  Anchored in window:      {len(anchored)}")
        print(f"  Mis-anchored (later):    {len(misplaced)}")
        print(f"  Missing news-hook:       {len(missing_hook)}")
        print(f"  Missing stats:           {len(missing_stats)}")
        print(f"  Coverage score:          {score:.2f}")
        print(f"  Recommendation: {recommendation}")
        print(bar)

        if anchored:
            print("\n  ANCHORED IN WINDOW:")
            for a in anchored:
                print(f"    [{a['entity_type']:7s}] {a['start_time']:6.2f}s  "
                      f"{a['name'][:40]:40s}  srt={a['srt_indices']}")

        if misplaced:
            print("\n  FIRST OCCURRENCE MIS-ANCHORED (emitted later instead of here):")
            for m in misplaced:
                print(f"    {m['name'][:30]:30s}  first mentioned SRT {m['first_srt_idx']:>3} "
                      f"({m['first_srt_time']:6.2f}s), emitted at SRT {m['emitted_at_srt_idx']:>3} "
                      f"({m['emitted_at_time']:6.2f}s)")

        if missing_hook:
            print("\n  MISSING NEWS-HOOK (in window, not extracted):")
            for h in missing_hook:
                t = h["srt_time"] if h["srt_time"] is not None else 0.0
                print(f"    [{h['hint']:6s}] SRT {h['srt_idx']:>3} ({t:6.2f}s)  "
                      f"{h['phrase']}")

        if missing_stats:
            print("\n  MISSING STATS (in window, not extracted):")
            for st in missing_stats:
                t = st["srt_time"] if st["srt_time"] is not None else 0.0
                print(f"    SRT {st['srt_idx']:>3} ({t:6.2f}s)  {st['phrase']}")

        print(f"\n{bar}\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate lower-third name/title graphics from a voiceover SRT using "
                    "local Ollama + Pillow (image mode) or Ollama + Playwright + FFmpeg "
                    "(video mode with CSS animation). Outputs a separate OTIO file for "
                    "DaVinci Resolve."
    )
    parser.add_argument("project", help="Project directory or SRT file path")
    parser.add_argument("--srt", help="Explicit SRT path (skip auto-discovery)")
    parser.add_argument("--ollama-model", default="llama3.2", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--output-otio", help="Output OTIO path (default: <project>/lowerthirds.otio)")
    parser.add_argument("--output-dir", help="Output directory (default: <project>/lowerthirds/)")
    parser.add_argument("--resolution", default="1920x1080", help="Output WxH (default: 1920x1080)")
    parser.add_argument("--font", default=None, help="TTF font path (default: auto-detect arial)")
    parser.add_argument("--duration-min", type=float, default=2.0,
                        help="Minimum on-screen seconds per entity (default: 2.0). "
                             "Ignored when --word-anchored is active (entities end at "
                             "seg.end_time instead).")
    word_anchored_group = parser.add_mutually_exclusive_group()
    word_anchored_group.add_argument(
        "--word-anchored", dest="word_anchored", action="store_true", default=None,
        help="Anchor each entity's start_time to the spoken-word onset (via .words.json "
             "sidecar). End_time = seg.end_time. Requires the sidecar; generate it with "
             "scripts/extract_word_timings.py. (default: auto-detect from sidecar presence)")
    word_anchored_group.add_argument(
        "--no-word-anchored", dest="word_anchored", action="store_false", default=None,
        help="Force segment-level anchoring (start=seg.start, end=seg.end) even when the "
             ".words.json sidecar exists. Use this to opt out of word-onset timing on a "
             "per-run basis.")
    parser.add_argument("--mode", choices=("image", "video"), default="video",
                        help="Render mode: 'video' = animated .mov (Playwright + FFmpeg, default). "
                             "'image' = static .png (Pillow only, no animation).")
    parser.add_argument("--template", choices=sorted(TEMPLATE_REGISTRY.keys()), default="classic",
                        help="Lower-third template style: " + ", ".join(sorted(TEMPLATE_REGISTRY.keys())) +
                             " (default: classic)")
    parser.add_argument("--template-by-type", default="",
                        help="Override template per entity type. Format: person=classic,date=boxed,"
                             "info=modern. Unspecified types (including places) fall back to --template. "
                             "Empty = no overrides.")
    parser.add_argument("--animation-style", choices=ANIMATION_STYLES, default="slide",
                        help="CSS animation style for video mode: slide, pop, fade (default: slide)")
    parser.add_argument("--animation-duration", type=float, default=0.4,
                        help="Animation length in seconds before the lower-third holds (default: 0.4)")
    parser.add_argument("--fps", type=int, default=int(RATE),
                        help=f"Frame rate for video mode (default: {int(RATE)})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract entities and print a preview table; do not render or write OTIO")
    parser.add_argument("--first-minute-only", action="store_true",
                        help="Render only entities that anchor in the first 60 seconds of the SRT. "
                             "Output paths change to <project>/first_minute_lowerthirds.otio and "
                             "<project>/first_minute_lowerthirds/ so the file is distinguishable from a "
                             "full run (which writes lowerthirds.otio + lowerthirds/).")
    parser.add_argument("--audit-first-minute", action="store_true",
                        help="Audit first-minute lower-third coverage and exit. "
                             "Reports which news-hook entities (date, location, brand, "
                             "stats) are missing from the opening window or anchored to "
                             "later SRT occurrences. Useful for broadcast/news edits where "
                             "the opener must be densely captioned.")
    parser.add_argument("--entity-types", default="person,place,date,info,chapter",
                        help="Comma-separated entity types to include "
                             "(default: person,place,date,info,chapter). "
                             f"Choices: {', '.join(t.value for t in EntityType)}")
    parser.add_argument("--no-chapters", action="store_true",
                        help="Skip auto-loading chapter title overlays from "
                             "<project>/checkpoint.json (default: auto-load when present).")
    parser.add_argument("--no-pre-screening", action="store_true",
                        help="Disable Python pre-screening of entity candidates. "
                             "Sends the full SRT to Ollama (legacy mode, prone to "
                             "hallucinated names on long transcripts).")
    args = parser.parse_args()

    if args.mode == "video" and args.fps <= 0:
        logger.error("--fps must be > 0")
        return 1

    input_path = Path(args.project)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1

    if input_path.is_file() and input_path.suffix.lower() == ".srt":
        srt_path = input_path
        project_dir = input_path.parent
    elif input_path.is_file() and input_path.suffix.lower() in (".otio", ".otiod"):
        # OTIO file: treat its parent dir as the project root so we can find
        # the voiceover SRT alongside it (typical project layout has both
        # 1.otio and voiceover/voiceover_trimmed.srt in the same directory).
        project_dir = input_path.parent
        if args.srt:
            srt_path = Path(args.srt)
        else:
            srt_path = find_srt_path(project_dir)
            if not srt_path:
                logger.error(f"No SRT found in {project_dir} or voiceover/")
                return 1
    elif input_path.is_dir():
        project_dir = input_path
        if args.srt:
            srt_path = Path(args.srt)
        else:
            srt_path = find_srt_path(project_dir)
            if not srt_path:
                logger.error(f"No SRT found in {project_dir} or voiceover/")
                return 1
    else:
        logger.error(f"Input must be a project dir, .srt file, or .otio file: {input_path}")
        return 1

    logger.info(f"SRT: {srt_path}")
    logger.info(f"Project: {project_dir}")

    try:
        w, h = map(int, args.resolution.lower().split("x"))
    except ValueError:
        logger.error(f"Invalid resolution '{args.resolution}' — use WxH like 1920x1080")
        return 1

    template = TEMPLATE_REGISTRY[args.template]
    template.width = w
    template.height = h
    if args.font:
        template.font_name = args.font

    template_overrides = parse_template_by_type(args.template_by_type)
    if template_overrides:
        logger.info(
            "Template-by-type overrides: "
            + ", ".join(
                f"{t.value}={n}"
                for t, n in sorted(template_overrides.items(), key=lambda kv: kv[0].value)
            )
        )

    def _pick_template(entity: EntitySpan) -> Tuple[str, LowerThirdTemplate]:
        name = template_overrides.get(entity.entity_type, args.template)
        t = TEMPLATE_REGISTRY[name]
        t.width = w
        t.height = h
        if args.font:
            t.font_name = args.font
        return name, t

    # Word-onset anchoring auto-detect: sidecar present + user didn't override.
    sidecar_path = Path(srt_path).with_suffix(".words.json")
    sidecar_exists = sidecar_path.exists()
    if args.word_anchored is None:
        word_anchored = sidecar_exists
    else:
        word_anchored = args.word_anchored and sidecar_exists
        if args.word_anchored and not sidecar_exists:
            logger.error(
                f"--word-anchored requested but no sidecar at {sidecar_path}. "
                f"Generate it with: python scripts/extract_word_timings.py {project_dir}"
            )
            return 1
    if word_anchored:
        logger.info(
            f"Word-onset anchoring ENABLED via {sidecar_path.name} "
            f"(start=spoken word onset, end=seg.end_time)"
        )
    elif args.word_anchored is False and sidecar_exists:
        logger.info(
            "Word-onset anchoring DISABLED via --no-word-anchored "
            "(segment-level timing used despite sidecar presence)"
        )
    elif not sidecar_exists:
        logger.info(
            "Word-onset anchoring OFF — no .words.json sidecar. "
            f"Generate one with: python scripts/extract_word_timings.py {project_dir}"
        )

    words_by_idx: Optional[Dict[int, List[dict]]] = None
    if word_anchored:
        # Defer segment load until parse_srt_file runs inside the extractor —
        # load_word_timings_by_idx needs the parsed segments for time-window
        # matching, and re-parsing here would be redundant.
        from src.utils import parse_srt_file as _parse_srt
        _segments_for_words = _parse_srt(str(srt_path))
        words_by_idx = load_word_timings_by_idx(srt_path, _segments_for_words)
        if not words_by_idx:
            logger.warning(
                f"Sidecar {sidecar_path} exists but no words could be matched to "
                f"SRT segments — falling back to segment-level anchoring"
            )
            word_anchored = False

    entities, segments = extract_entities_via_ollama(
        srt_path,
        args.ollama_model,
        args.ollama_host,
        use_pre_screening=not args.no_pre_screening,
        words_by_idx=words_by_idx,
    )
    if not entities:
        logger.error("No entities extracted — check Ollama model + SRT content. Aborting.")
        return 1

    extracted_count = len(entities)
    entities = dedupe_entities(entities)
    if len(entities) < extracted_count:
        logger.info(
            f"Dedupe: kept {len(entities)}/{extracted_count} entities "
            f"(type-agnostic, case-insensitive, leading-article insensitive)"
        )

    # Optional: load chapter title overlays from checkpoint (auto-detected).
    # Chapters come from src/chapter_detection/detector.py and live at
    # <project>/checkpoint.json → chapter_data.chapters. Add `--no-chapters`
    # to skip when the user doesn't want them.
    if not args.no_chapters:
        chapter_entities = load_chapter_entities(project_dir, segments)
        if chapter_entities:
            entities.extend(chapter_entities)

    # Substring dedup runs AFTER chapters are merged so chapter titles can
    # suppress redundant standalone entities (e.g. "Zagreb" is a substring
    # of the chapter title "Zagreb's Role in Autonomous Vehicle Development").
    entities = dedupe_substring_entities(entities)

    entities = redistribute_stacked_entities(entities, word_anchored=word_anchored)
    entities = clamp_entity_durations(entities, args.duration_min, word_anchored=word_anchored)
    logger.info(f"Extracted {len(entities)} entities after redistribution + clamping")

    # Filter by --entity-types (default: all four types)
    selected_types: set = set()
    for raw in args.entity_types.split(","):
        raw = raw.strip()
        if not raw:
            continue
        normalized = _normalize_entity_type(raw)
        if normalized is None:
            logger.error(
                f"Unknown --entity-types value {raw!r}; "
                f"valid: {[t.value for t in EntityType]}"
            )
            return 1
        selected_types.add(normalized)
    if not selected_types:
        logger.error("--entity-types filter is empty; nothing to render.")
        return 1
    before = len(entities)
    entities = [e for e in entities if e.entity_type in selected_types]
    logger.info(
        f"Filtered by --entity-types ({', '.join(t.value for t in selected_types)}): "
        f"kept {len(entities)}/{before}"
    )
    if not entities:
        logger.error("No entities remain after --entity-types filter — nothing to render.")
        return 1

    if args.first_minute_only:
        before = len(entities)
        entities = [e for e in entities if e.start_time < 60.0]
        logger.info(
            f"Filtered to first minute (start_time < 60s): "
            f"kept {len(entities)}/{before}"
        )
        if not entities:
            logger.error("No entities remain in the first minute — nothing to render.")
            return 1

    if args.dry_run:
        _print_dry_run(
            entities, template,
            mode=args.mode,
            style=args.animation_style if args.mode == "video" else None,
            anim_duration_s=args.animation_duration,
        )
        return 0

    if args.audit_first_minute:
        # Audit-only path — print coverage report and exit. Does NOT
        # render or write any files. Useful as a pre-render sanity check
        # on broadcast/news edits.
        report = audit_first_minute(entities, segments)
        # Distinct path from the main lower-thirds run (which writes
        # lowerthirds.otio and lowerthirds/entities.json). Keep the audit
        # JSON at project root so editors can find it without digging into
        # the assets directory.
        audit_path = project_dir / "first_minute_coverage_audit.json"
        audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Audit report written to {audit_path}")
        return 0

    # Default output paths differ between full and first-minute runs so the
    # two artifacts don't collide on disk and are easy to tell apart.
    if args.first_minute_only:
        default_otio = project_dir / "first_minute_lowerthirds.otio"
        default_dir = project_dir / "first_minute_lowerthirds"
    else:
        default_otio = project_dir / "lowerthirds.otio"
        default_dir = project_dir / "lowerthirds"
    output_otio = Path(args.output_otio) if args.output_otio else default_otio
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    media_paths: List[Path] = []
    template_names: List[str] = []
    for idx, entity in enumerate(entities, start=1):
        template_name, t = _pick_template(entity)
        template_names.append(template_name)
        if args.mode == "video":
            out_path = output_dir / f"lowerthird_{idx:03d}.mov"
            render_lower_third_video(
                entity, out_path, t,
                style=args.animation_style,
                anim_duration_s=args.animation_duration,
                fps=args.fps,
            )
        else:
            out_path = output_dir / f"lowerthird_{idx:03d}.png"
            render_lower_third_png(entity, out_path, t)
        media_paths.append(out_path)
        logger.info(f"  rendered {out_path.name}: {entity.name}")

    # Track composition summary (filled after OTIO write so we know real per-track counts)
    tracks_summary: Dict[str, int] = {}

    summary = {
        "srt_file": str(srt_path),
        "model": args.ollama_model,
        "resolution": {"width": w, "height": h},
        "mode": args.mode,
        "entity_count": len(entities),
        "entities": [
            {
                "idx": i + 1,
                "name": e.name,
                "role": e.role,
                "entity_type": e.entity_type.value,
                "template": template_names[i],
                "start_time": e.start_time,
                "end_time": e.end_time,
                "srt_indices": e.srt_indices,
                "media_file": media_paths[i].name,
            }
            for i, e in enumerate(entities)
        ],
    }
    if args.mode == "video":
        summary["animation_style"] = args.animation_style
        summary["animation_duration_s"] = args.animation_duration
        summary["fps"] = args.fps
    summary_path = output_dir / "entities.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary: {summary_path}")

    timeline = build_lower_thirds_otio(entities, media_paths, template_names, output_otio)
    # Fill in per-track counts from the actual built timeline
    for tr in timeline.tracks:
        clip_count = sum(1 for c in tr if hasattr(c, "metadata") and c.metadata.get("entity"))
        if clip_count:
            tracks_summary[tr.name] = clip_count
    if tracks_summary:
        summary["tracks"] = tracks_summary
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"OTIO: {output_otio}")

    total_dur = max((e.end_time for e in entities), default=0.0)
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Lower-thirds generated: {len(entities)} ({args.mode} mode)")
    print(f"  Timeline duration: {total_dur:.2f}s ({int(total_dur * RATE)} frames @ {RATE:.0f}fps)")
    print(f"  OTIO: {output_otio}")
    print(f"  Media: {output_dir}")
    print(f"  Summary: {summary_path}")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
