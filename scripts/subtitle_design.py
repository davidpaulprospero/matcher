#!/usr/bin/env python3
"""
Generate CapCut-style styled subtitles from a project SRT.

Produces:
  - <output_dir>/subtitles.ass      : styled/animated ASS subtitle file (libass format)
  - <output_dir>/subtitles.srt      : styled plain SRT (no animations, fallback)
  - <output_dir>/subtitles.otio     : DaVinci-compatible OTIO with TrackKind.Text

Reuses:
  - src.utils.parse_srt_file / SRTSegment
  - src.downloader.utils.SUBPROCESS_FLAGS
  - src.otio.utils._to_windows_path
  - find_srt_path pattern from scripts/title_color_video.py

CLI:
  python scripts/subtitle_design.py --project <dir> --style mrbeast --animation pop
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import SRTSegment, parse_srt_file  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

RATE: float = 30.0  # OTIO frame rate for the text track (matches repo convention)


class Position(str, Enum):
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


class Animation(str, Enum):
    NONE = "none"
    KARAOKE = "karaoke"
    POP = "pop"
    FADE = "fade"
    FADE_UP = "fade_up"


# ASS alignment NUMPAD values (top-left=7, bottom-center=2, etc.)
_POSITION_TO_NUMPAD = {
    Position.TOP: 8,      # top-center
    Position.CENTER: 5,   # middle-center
    Position.BOTTOM: 2,   # bottom-center
}


# ============================================================
# SRT discovery
# ============================================================

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
                   # Video containers that also carry an audio stream — faster-whisper
                   # will demux the audio and transcribe it. We only need the audio track.
                   ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


def find_srt_path(project_dir: Path) -> Optional[Path]:
    """Find the appropriate SRT path for a project, preferring trimmed variant.

    Mirrors scripts/title_color_video.py:33-58 verbatim.
    """
    project_dir = Path(project_dir)
    voiceover_dir = project_dir / "voiceover"
    candidates: List[Path] = []

    if voiceover_dir.exists():
        for f in voiceover_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".srt":
                candidates.append(f)

    for f in project_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".srt" and f not in candidates:
            candidates.append(f)

    if not candidates:
        return None

    trimmed_stems = {f.stem.replace("_trimmed", "") for f in candidates if "_trimmed" in f.stem}
    if trimmed_stems:
        candidates = [f for f in candidates if "_trimmed" in f.stem or f.stem not in trimmed_stems]

    return candidates[0] if candidates else None


def transcribe_audio_to_srt(
    audio_path: Path,
    output_srt: Path,
    output_words_json: Path,
    model_size: str = "base",
    language: Optional[str] = None,
    use_cache: bool = True,
    device: str = "cpu",
    compute_type: str = "auto",
) -> Path:
    """Transcribe an audio file to SRT + .words.json sidecar using faster-whisper.

    The derived files are written next to the audio with the same stem
    (e.g. `forbidden_city_trimmed.mp3` → `forbidden_city_trimmed.srt` +
    `forbidden_city_trimmed.words.json`). On subsequent runs the derived SRT
    is reused if its mtime >= audio mtime (cache hit), unless `use_cache=False`.

    Returns the SRT path the rest of the pipeline should consume.
    """
    audio_path = Path(audio_path)
    output_srt = Path(output_srt)
    output_words_json = Path(output_words_json)

    if (
        use_cache
        and output_srt.exists()
        and output_srt.stat().st_mtime >= audio_path.stat().st_mtime
    ):
        logger.info(f"Using cached transcription: {output_srt}")
        return output_srt

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for audio transcription. "
            "Install with: pip install faster-whisper"
        ) from e

    logger.info(
        f"Transcribing {audio_path.name} with faster-whisper "
        f"({model_size}, device={device}, compute_type={compute_type})..."
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language,
        vad_filter=False,
    )

    cues: List[dict] = []
    for seg in segments_iter:
        words = [
            {
                "word": w.word,
                "start": float(w.start) if w.start is not None else float(seg.start),
                "end": float(w.end) if w.end is not None else float(seg.end),
                "probability": float(w.probability) if w.probability is not None else 1.0,
            }
            for w in (seg.words or [])
        ]
        cues.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words,
        })

    # Write SRT
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_ts(cue['start'])} --> {_srt_ts(cue['end'])}\n")
            f.write(f"{cue['text']}\n\n")

    # Write words.json in the shape load_word_timestamps() expects.
    output_words_json.write_text(
        json.dumps(cues, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        f"Transcribed {len(cues)} cues → {output_srt.name} (+{output_words_json.name})"
    )
    return output_srt


def _srt_ts(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


# ============================================================
# Word-level timestamps
# ============================================================

def load_word_timestamps(srt_path: Path) -> List[dict]:
    """Load word-level timestamps from sibling .words.json sidecar.

    Returns the raw segment dicts written by faster-whisper (each containing
    a 'words' field with {word, start, end, confidence}).

    Returns empty list if the sidecar is missing or malformed — callers
    should fall back to per-segment timing in that case.
    """
    words_path = srt_path.with_suffix(".words.json")
    if not words_path.exists():
        logger.warning(f"No word-timestamp sidecar at {words_path}; karaoke disabled")
        return []
    try:
        with open(words_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning(f"Unexpected .words.json shape (not a list): {words_path}")
        return []
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load word timestamps from {words_path}: {e}")
        return []


def words_for_segment(words_data: List[dict], seg: SRTSegment) -> List[dict]:
    """Pick the word entries that fall inside a segment's time range."""
    out: List[dict] = []
    for entry in words_data:
        ws = entry.get("words") or []
        for w in ws:
            try:
                w_start = float(w.get("start", 0.0))
            except (TypeError, ValueError):
                continue
            if seg.start_time <= w_start <= seg.end_time:
                out.append(w)
    return out


# ============================================================
# Style
# ============================================================

@dataclass
class SubtitleStyle:
    """A single subtitle style. All fields are preset-overridable."""

    font_family: str = "Arial"
    font_size: int = 48
    primary_color: str = "&H00FFFFFF"        # white
    karaoke_color: str = "&H000000FF"        # blue (SecondaryColour for karaoke highlight)
    outline_color: str = "&H00000000"        # black
    shadow_color: str = "&H80000000"         # 50% black
    outline_thickness: float = 2.0
    shadow_offset: Tuple[int, int] = (2, 2)
    bold: bool = True
    italic: bool = False
    position: Position = Position.BOTTOM
    margin_v: int = 50
    animation: Animation = Animation.NONE
    max_chars_per_line: int = 42
    auto_split_long_lines: bool = True
    # When True, switch BorderStyle to 3 (opaque back-box) so words sit inside
    # a dark rectangular background — the closest libass approximation to the
    # CapCut "rounded pill" look. Pixel-perfect pills need raster rendering.
    background_box: bool = False
    background_color: str = "&H00000000"     # opaque black (used when background_box=True)
    # Optional Comikku/Anton-style condensed bold. The writer uppercases input
    # to mimic CapCut's all-caps rendering when this is True.
    force_uppercase: bool = False
    # CapCut emphasises a single word per cue in its pink "highlight" color.
    # highlight_position picks WHICH word. Implemented as an `{\c&H...}` override.
    highlight_position: str = "first"        # "first" | "last" | "none"
    # Number of words visible at once during KARAOKE animation. Each word's
    # Dialogue line shows a centered window of `karaoke_window_size` words
    # with the current word in pink. 3 reads as the CapCut default.
    karaoke_window_size: int = 3
    # When the gap between two consecutive words exceeds this many seconds,
    # start a new batch at the later word. Lets batch boundaries land on
    # natural speaker pauses instead of forcing them at fixed index counts.
    karaoke_pause_threshold: float = 0.5
    # Duration of the fade-up "fly away" tail for Animation.FADE_UP, in ms.
    # The static Dialogue line covers [start, end - anim_duration_ms], then
    # a tail Dialogue line of length `anim_duration_ms` fades out + drifts up.
    anim_duration_ms: int = 400


# Curated bright palette for the karaoke highlight (current word) color.
# Format is &HAABBGGRR; we keep AA=00 (fully opaque). All entries are
# saturated enough to read on a transparent background with a black outline
# and a white "base" color for the non-current words. Avoid dark or
# pastel hues — they wash out against the white text outline.
BRIGHT_HIGHLIGHT_COLORS: Tuple[str, ...] = (
    "&H00C080F0",  # hot pink
    "&H000000FF",  # bright red
    "&H000080FF",  # orange
    "&H0000FFFF",  # yellow
    "&H0000FF00",  # lime
    "&H00FFFF00",  # cyan
    "&H00FF00FF",  # magenta
    "&H00FF0080",  # purple
    "&H00FF0000",  # bright blue
    "&H0080FF00",  # teal
    "&H0000AAFF",  # amber
    "&H00FF8000",  # sky blue
)


def pick_random_highlight_color(rng: Optional[random.Random] = None) -> str:
    """Return a random bright color from BRIGHT_HIGHLIGHT_COLORS.

    Uses a fresh `random.Random()` instance by default so callers don't
    influence the choice through the global RNG. Pass an explicit rng to
    make the choice deterministic (used by tests).
    """
    rng = rng or random.Random()
    return rng.choice(BRIGHT_HIGHLIGHT_COLORS)


PRESETS: dict = {
    "capcut_default": SubtitleStyle(
        font_family="Arial", font_size=56, primary_color="&H00FFFFFF",
        outline_color="&H00000000", shadow_color="&H80000000",
        outline_thickness=2.0, shadow_offset=(2, 2),
        bold=True, italic=False, position=Position.BOTTOM, margin_v=50,
        animation=Animation.NONE,
    ),
    "mrbeast": SubtitleStyle(
        font_family="Impact", font_size=80, primary_color="&H0000FFFF",
        karaoke_color="&H0000FFFF",
        outline_color="&H00000000", shadow_color="&H00000000",
        outline_thickness=4.0, shadow_offset=(0, 0),
        bold=True, italic=False, position=Position.BOTTOM, margin_v=80,
        animation=Animation.POP,
    ),
    "hormozi": SubtitleStyle(
        font_family="Arial", font_size=60, primary_color="&H00FFFFFF",
        outline_color="&H000000CC", shadow_color="&H80000000",
        outline_thickness=3.0, shadow_offset=(2, 2),
        bold=True, italic=False, position=Position.TOP, margin_v=80,
        animation=Animation.FADE,
    ),
    "tiktok": SubtitleStyle(
        font_family="Arial", font_size=52, primary_color="&H00FFFFFF",
        karaoke_color="&H0000FFFF",
        outline_color="&H00000000", shadow_color="&H80000000",
        outline_thickness=2.0, shadow_offset=(2, 2),
        bold=True, italic=False, position=Position.BOTTOM, margin_v=60,
        animation=Animation.KARAOKE,
    ),
    # Explicit "3 word karaoke" — same as tiktok but with a name that
    # describes the behaviour (default `karaoke_window_size=3` → at most
    # 3 words visible at once, current word highlighted in the karaoke color).
    # Keep this name stable: it is the default go-back-to style for the user.
    "three_word_karaoke": SubtitleStyle(
        font_family="Arial", font_size=52, primary_color="&H00FFFFFF",
        karaoke_color="&H0000FFFF",
        outline_color="&H00000000", shadow_color="&H80000000",
        outline_thickness=2.0, shadow_offset=(2, 2),
        bold=True, italic=False, position=Position.BOTTOM, margin_v=60,
        animation=Animation.KARAOKE,
    ),
    "minimal": SubtitleStyle(
        font_family="Arial", font_size=40, primary_color="&H00FFFFFF",
        outline_color="&H00000000", shadow_color="&H80000000",
        outline_thickness=0.0, shadow_offset=(0, 0),
        bold=False, italic=False, position=Position.BOTTOM, margin_v=40,
        animation=Animation.FADE,
    ),
    "news_broadcast": SubtitleStyle(
        font_family="Arial", font_size=44, primary_color="&H00FFFFFF",
        outline_color="&H00000000", shadow_color="&H00000000",
        outline_thickness=0.0, shadow_offset=(0, 0),
        bold=True, italic=False, position=Position.BOTTOM, margin_v=30,
        animation=Animation.NONE,
    ),
    # DaVinci caps preset — sampled from e:/Edit Job/Oscar/8-12/1caps_DAVINCI.mov
    # (a Croatian/Czech stock-video edit). Heavy condensed white text, each
    # spoken word briefly flashes cotton-candy pink #F080C0 as it's uttered
    # (libass {\k} karaoke highlight), set against a THICK BLACK OUTLINE
    # (BorderStyle=1, no opaque background box — the "pill" look in the source
    # is just an outline thick enough that adjacent letters visually merge).
    # All-caps rendering. Requires a <srt>.words.json sidecar from Whisper;
    # without one, falls back to highlighting just the first word of each cue.
    "davinci_caps": SubtitleStyle(
        font_family="Anton", font_size=78,
        primary_color="&H00FFFFFF",
        karaoke_color="&H00C080F0",           # pink #F080C0 in BGR
        outline_color="&H00000000",
        shadow_color="&H00000000",
        outline_thickness=6.0,                # thick black outline (the "pill" effect)
        shadow_offset=(0, 0),
        bold=True, italic=False,
        position=Position.BOTTOM, margin_v=70,
        animation=Animation.KARAOKE,          # word-by-word pink flash
        max_chars_per_line=28,
        background_box=False,                 # use outline, not opaque back-box
        background_color="&H00000000",
        force_uppercase=True,
        highlight_position="first",          # fallback when no .words.json
    ),
    # Gold text on a solid black back-box. Whole cues visible at once
    # (Animation.NONE → no per-word highlighting, no 3-word sliding window).
    # Gold #FFD700 in BGR+alpha = &H0000D7FF.
    "gold_on_black": SubtitleStyle(
        font_family="Arial", font_size=56,
        primary_color="&H0000D7FF",           # gold #FFD700
        karaoke_color="&H0000D7FF",
        outline_color="&H00000000",
        shadow_color="&H00000000",
        outline_thickness=0.0,                # box covers the glyphs; no outline needed
        shadow_offset=(0, 0),
        bold=True, italic=False,
        position=Position.BOTTOM, margin_v=50,
        animation=Animation.FADE_UP,
        background_box=True,                  # BorderStyle=3 opaque back-box
        background_color="&H00000000",        # opaque black
        highlight_position="none",            # no per-word color overrides
    ),
}


def apply_preset(name: str) -> SubtitleStyle:
    """Return a fresh SubtitleStyle for the named preset, or raise ValueError."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown style preset {name!r}. Available: {sorted(PRESETS.keys())}"
        )
    # PRESETS values are SubtitleStyle instances; copy() so caller mutations don't leak.
    from dataclasses import replace
    return replace(PRESETS[name])


# ============================================================
# Color helpers
# ============================================================

def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """Convert '#RRGGBB' or 'RRGGBB' to ASS BGR+alpha color '&HAABBGGRR'.

    alpha uses libass convention: 0 = opaque, 255 = transparent.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Hex color must be 6 chars (RRGGBB), got {hex_color!r}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


# ============================================================
# Cue splitting
# ============================================================

def _wrap_text(text: str, max_chars: int) -> str:
    """Greedy word-aware wrap; ASS renders \\N as a hard line break."""
    if len(text) <= max_chars or " " not in text:
        return text
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for w in words:
        added = len(w) + (1 if current else 0)
        if current_len + added > max_chars and current:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += added
    if current:
        lines.append(" ".join(current))
    return "\\N".join(lines)


def split_long_cues(segments: List[SRTSegment], max_chars: int) -> List[SRTSegment]:
    """Wrap each cue's text; preserves timing (in-place text rewrite)."""
    if max_chars <= 0:
        return segments
    out: List[SRTSegment] = []
    for s in segments:
        if len(s.text) > max_chars and " " in s.text:
            wrapped = _wrap_text(s.text, max_chars)
            out.append(SRTSegment(
                index=s.index, start_time=s.start_time, end_time=s.end_time,
                text=wrapped, source_file=s.source_file,
            ))
        else:
            out.append(s)
    return out


# ============================================================
# SRT writer (styled, but plain — no animations)
# ============================================================

def _fmt_srt_time(seconds: float) -> str:
    """HH:MM:SS,mmm (SRT timestamp format)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_styled_srt(segments: List[SRTSegment], path: Path) -> None:
    """Write a plain SRT (style info is in the matching .ass; SRT stays portable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = _fmt_srt_time(seg.start_time)
            end = _fmt_srt_time(seg.end_time)
            f.write(f"{i}\n{start} --> {end}\n{seg.text}\n\n")


# ============================================================
# ASS writers
# ============================================================

def _fmt_ass_time(seconds: float) -> str:
    """HH:MM:SS.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))
    h, rem = divmod(total_cs, 3600 * 100)
    m, rem = divmod(rem, 60 * 100)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_style_block(style: SubtitleStyle, style_name: str = "Default") -> str:
    """Render a single [V4+ Styles] Style: line.

    BorderStyle=1 (default): outline + shadow.
    BorderStyle=3 (when background_box=True): opaque back-box behind text
        (rectangular, not rounded — closest libass approximation to the
        CapCut per-word pill background).
    """
    bold_flag = -1 if style.bold else 0
    italic_flag = -1 if style.italic else 0
    alignment = _POSITION_TO_NUMPAD[style.position]
    border_style = 3 if style.background_box else 1
    back_colour = style.background_color if style.background_box else "&H00000000"
    return (
        f"Style: {style_name},{style.font_family},{style.font_size},"
        f"{style.primary_color},{style.karaoke_color},{style.outline_color},"
        f"{back_colour},"
        f"{bold_flag},{italic_flag},0,0,100,100,0,0,"
        f"{border_style},{style.outline_thickness:.1f},{style.shadow_offset[0]},"
        f"{alignment},10,10,{style.margin_v},1\n"
    )


def _ass_header(style: SubtitleStyle, title: str) -> str:
    """[Script Info] + [V4+ Styles] sections, no [Events] yet."""
    return (
        "[Script Info]\n"
        f"Title: {title}\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + _ass_style_block(style)
    )


def _ass_events_header() -> str:
    return (
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _ass_dialogue(start_s: float, end_s: float, text: str) -> str:
    text = text.replace("\n", "\\N")
    return (
        f"Dialogue: 0,{_fmt_ass_time(start_s)},{_fmt_ass_time(end_s)},Default,,"
        f"0,0,0,,{text}\n"
    )


def _highlight_first_word(text: str, highlight_color: str) -> str:
    """Wrap the FIRST whitespace-separated token in a colour override.

    Used by the davinci_caps preset to mimic CapCut's "first word pink, rest white".
    Returns the text unchanged when there's nothing to colour.
    """
    parts = text.split(" ", 1)
    if len(parts) < 2:
        return text
    first, rest = parts
    return f"{{\\c{highlight_color}}}{first}{{\\c&H00FFFFFF&}} {rest}"


def _animation_overrides(
    style: SubtitleStyle,
    width: int = 1920,
    height: int = 1080,
    is_animated_tail: bool = False,
) -> str:
    """ASS override tags prepended to the Text field based on style.animation.

    For FADE_UP we emit a per-cue override only on the SECOND Dialogue line
    (the tail). The static portion gets no override and stays put; the tail
    fades out and moves up over its own (short) duration. That's how we get
    the "hold still then fly away" feel — ASS's \move runs linearly across
    the whole line, so splitting into a static + animated pair is the only
    way to defer the motion to the end of the cue.
    """
    if style.animation == Animation.POP:
        # Scale Y from 110% over first 200cs back to 100% over next 200cs.
        return "{\\t(0,200,\\fscy110)\\t(200,400,\\fscy100)}"
    if style.animation == Animation.FADE:
        return "{\\fad(300,200)}"
    if style.animation == Animation.FADE_UP:
        if not is_animated_tail:
            return ""
        # Tail only — fades from full to 0 while drifting up over its own
        # (short) duration. The static line above is responsible for most of
        # the cue's duration; this tail just flies away at the end.
        cx = width // 2
        cy = height - style.margin_v
        delta = 30
        return f"{{\\fad(0,400)\\move({cx},{cy},{cx},{cy - delta})}}"
    return ""


def write_ass(segments: List[SRTSegment], style: SubtitleStyle, path: Path) -> None:
    """Write a static-styled .ass (no per-word karaoke, but with style + line animations).

    Honours style.background_box (sets BorderStyle=3) and style.highlight_position
    ("first"/"last"/"none" — colours one token with style.karaoke_color).

    For FADE_UP, each cue emits TWO Dialogue lines:
      - A "static" line covering [start, end - anim_dur_s] with no overrides.
      - A "tail" line covering [end - anim_dur_s, end] with \\fad + \\move so
        the text fades out and drifts away only at the end.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    static_prefix = _animation_overrides(style, width, height, is_animated_tail=False)
    tail_prefix = _animation_overrides(style, width, height, is_animated_tail=True)
    anim_dur_s = style.anim_duration_ms / 1000.0 if style.animation == Animation.FADE_UP else 0.0

    def _prepare(text: str, prefix: str) -> str:
        if style.force_uppercase:
            text = text.upper()
        if style.highlight_position == "first":
            text = _highlight_first_word(text, style.karaoke_color)
        elif style.highlight_position == "last":
            parts = text.rsplit(" ", 1)
            if len(parts) == 2:
                text = f"{parts[0]} {{\\c{style.karaoke_color}}}{parts[1]}{{\\c&H00FFFFFF&}}"
        return f"{prefix}{text}"

    parts: List[str] = [_ass_header(style, title=path.stem), _ass_events_header()]
    for seg in segments:
        text = _prepare(seg.text, static_prefix)
        if style.animation == Animation.FADE_UP and anim_dur_s > 0 and (seg.end_time - seg.start_time) > anim_dur_s:
            static_end = seg.end_time - anim_dur_s
            parts.append(_ass_dialogue(seg.start_time, static_end, text))
            parts.append(_ass_dialogue(static_end, seg.end_time, _prepare(seg.text, tail_prefix)))
        else:
            parts.append(_ass_dialogue(seg.start_time, seg.end_time, text))

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def _karaoke_window_text(
    window_words: List[dict],
    current_word: dict,
    style: SubtitleStyle,
) -> str:
    """Build the text for one Dialogue line: a window of words where the
    current word is pink and the other words are white.

    Pattern: per-word static `{\c}` colour overrides. The current word is
    wrapped in `{\c<karaoke>}<token>{\c<primary>}` so the colour stays local
    to that word — no leak into adjacent text. Other words are wrapped in
    `{\c<primary>}<token>{\c<primary>}` so they stay white.
    """
    parts: List[str] = []
    if style.force_uppercase:
        tokens = [ww["word"].upper() for ww in window_words]
    else:
        tokens = [ww["word"] for ww in window_words]

    for ww, token in zip(window_words, tokens):
        if ww is current_word:
            # Current word — pink (the highlight colour)
            parts.append(
                f"{{\\c{style.karaoke_color}}}{token}"
                f"{{\\c{style.primary_color}}}"
            )
        else:
            # Non-current words — white (the base colour)
            parts.append(
                f"{{\\c{style.primary_color}}}{token}"
                f"{{\\c{style.primary_color}}}"
            )
    return " ".join(parts)


def _karaoke_batch_ranges(
    valid_words: List[dict],
    batch_size: int,
    pause_threshold: float,
) -> List[Tuple[int, int]]:
    """Compute (start_idx, end_idx) batches, breaking at speaker pauses.

    A new batch starts whenever the gap between consecutive words exceeds
    `pause_threshold` seconds, OR when the running batch reaches
    `batch_size` words. Whichever comes first. The last batch may be
    shorter than `batch_size` (a one-word remainder is fine).
    """
    n = len(valid_words)
    if n == 0:
        return []
    batches: List[Tuple[int, int]] = []
    batch_start = 0
    for i in range(1, n):
        gap = valid_words[i]["start"] - valid_words[i - 1]["end"]
        reached_max = (i - batch_start) >= batch_size
        if gap > pause_threshold or reached_max:
            batches.append((batch_start, i - 1))
            batch_start = i
    batches.append((batch_start, n - 1))
    return batches


def _karaoke_for_segment(seg: SRTSegment, words: List[dict], style: SubtitleStyle) -> List[Tuple[float, float, str]]:
    """Build Dialogue-line records for one cue with batched karaoke.

    Returns a list of (start, end, text) tuples — one PER WORD. Each line
    shows the entire batch, with the current word in white and the others
    in pink. As the speaker progresses through the batch, the white moves
    left-to-right. After the last word of a batch ends, the next batch
    appears.

    Why a separate Dialogue line per word instead of one line with per-word
    `\\t` overrides? libass `\\t` colour overrides are sticky — the colour
    persists past the time window's end, so all words end up pink
    simultaneously. Static `{\c}` overrides scoped per word never leak.
    The cost is hard-cuts between batches.

    Batch boundaries break at natural speaker pauses: when the gap between
    two consecutive words exceeds `style.karaoke_pause_threshold` seconds,
    the later word starts a new batch. This avoids a long silence from
    straddling two batches, which would leave the wrong word in the wrong
    group.

    Falls back to a single plain line (the whole cue text in primary colour)
    when no usable word timings exist. Honours style.force_uppercase.
    """
    if not words:
        text = seg.text
        if style.force_uppercase:
            text = text.upper()
        return [(seg.start_time, seg.end_time, text)]

    valid_words: List[dict] = []
    seg_start = seg.start_time
    seg_end = seg.end_time
    for w in words:
        try:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start))
        except (TypeError, ValueError):
            continue
        w_start = max(w_start, seg_start)
        w_end = min(w_end, seg_end)
        if w_end <= w_start:
            continue
        token = str(w.get("word", "")).strip()
        if not token:
            continue
        valid_words.append({"word": token, "start": w_start, "end": w_end})

    if not valid_words:
        text = seg.text
        if style.force_uppercase:
            text = text.upper()
        return [(seg.start_time, seg.end_time, text)]

    valid_words.sort(key=lambda w: w["start"])
    batch_size = max(1, style.karaoke_window_size)
    pause_threshold = max(0.0, style.karaoke_pause_threshold)
    batches = _karaoke_batch_ranges(valid_words, batch_size, pause_threshold)

    windows: List[Tuple[float, float, str]] = []
    for i, w in enumerate(valid_words):
        # Find which batch this word belongs to.
        for batch_start, batch_end in batches:
            if batch_start <= i <= batch_end:
                batch_words = valid_words[batch_start:batch_end + 1]
                window_text = _karaoke_window_text(batch_words, w, style)
                windows.append((w["start"], w["end"], window_text))
                break

    return windows


def write_karaoke_ass(
    segments: List[SRTSegment],
    words_data: List[dict],
    style: SubtitleStyle,
    path: Path,
) -> None:
    """Write an .ass with per-word karaoke timing from .words.json sidecar.

    Each cue emits multiple Dialogue lines (one background + one per word).
    Falls back gracefully to write_ass() when no usable words exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not words_data:
        logger.info("No word timestamps; falling back to non-karaoke ASS write")
        write_ass(segments, style, path)
        return

    if style.animation != Animation.KARAOKE:
        from dataclasses import replace
        karaoke_style = replace(style, animation=Animation.NONE)
    else:
        karaoke_style = style

    parts: List[str] = [_ass_header(karaoke_style, title=path.stem), _ass_events_header()]
    for seg in segments:
        seg_words = words_for_segment(words_data, seg)
        line_records = _karaoke_for_segment(seg, seg_words, karaoke_style)
        for start_s, end_s, text in line_records:
            parts.append(_ass_dialogue(start_s, end_s, text))

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def write_animated_ass(
    segments: List[SRTSegment],
    words_data: List[dict],
    style: SubtitleStyle,
    path: Path,
) -> None:
    """Dispatch writer based on style.animation.

    KARAOKE → write_karaoke_ass (needs words).
    POP / FADE / NONE → write_ass (line-level animations only).
    """
    if style.animation == Animation.KARAOKE:
        write_karaoke_ass(segments, words_data, style, path)
    else:
        write_ass(segments, style, path)


# ============================================================
# OTIO Text track builder
# ============================================================

def _to_windows_path(path: str) -> str:
    """Convert to absolute forward-slash path (DaVinci Resolve requirement).

    Mirrors src/otio/utils.py:113 behaviour; inlined here so the script is
    self-contained for users running it without the full repo on path.
    """
    import os
    try:
        abs_path = str(Path(path).resolve())
    except OSError:
        abs_path = os.path.abspath(path)
    return abs_path.replace("\\", "/")


def build_subtitle_otio(
    ass_path: Path,
    segments: List[SRTSegment],
    output_path: Path,
    frame_rate: float = RATE,
) -> Path:
    """Build a standalone OTIO with one TrackKind.Text track.

    Each cue becomes a Clip with ExternalReference to the .ass sidecar and a
    source_range matching the cue's window. DaVinci Resolve reads the track
    as a subtitle layer; other tools can also re-import the .ass directly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timeline = otio.schema.Timeline(
        name="subtitles_timeline",
        metadata={"Resolve_OTIO": {"Resolve OTIO Meta Version": "1.0"}},
    )
    timeline.global_start_time = RationalTime(0, frame_rate)
    timeline.tracks.name = ""

    # Resolve a subtitle track kind. Newer OTIO versions expose TrackKind.Text;
    # older versions (and this one) only know Video/Audio. Fall back to Video
    # with metadata["is_subtitle"] so NLEs that DO read the metadata flag
    # (DaVinci via Resolve_OTIO) can still recognize it as a subtitle layer.
    text_kind = getattr(otio.schema.TrackKind, "Text", otio.schema.TrackKind.Video)
    track = otio.schema.Track(name="V1 - Subtitles", kind=text_kind)
    track.enabled = True
    track.color = None
    track.metadata["Resolve_OTIO"] = {"Locked": False, "IsSubtitle": True}
    track.metadata["is_subtitle_track"] = True

    if not segments:
        logger.warning("No segments — V1 track will be empty")
    else:
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        current_time = 0.0
        safe_ass_url = _to_windows_path(str(ass_path))
        ass_name = ass_path.name

        for idx, seg in enumerate(sorted_segs, start=1):
            start = seg.start_time
            end = seg.end_time
            duration = max(0.001, end - start)
            duration_frames = max(1, int(round(duration * frame_rate)))

            # Gap before this cue (if any).
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

            media_ref = otio.schema.ExternalReference(
                target_url=safe_ass_url,
                available_range=TimeRange(
                    duration=RationalTime(duration_frames, frame_rate),
                    start_time=RationalTime(0, frame_rate),
                ),
            )
            media_ref.name = ass_name

            clip_name = f"SUB:{idx:04d}:{seg.text[:40].replace(chr(10), ' ')}"
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
            clip.metadata["Resolve_OTIO"] = {}
            clip.metadata["subtitle_text"] = seg.text

            track.append(clip)
            current_time = end

    timeline.tracks.append(track)

    otio.adapters.write_to_file(timeline, str(output_path))
    return output_path


# ============================================================
# Transparent video render (alpha-channel overlay)
# ============================================================

def _resolve_ffmpeg_for_render() -> str:
    """Find an ffmpeg binary, preferring the project's bundled copy.

    The matcher project ships an ffmpeg at tools/ffmpeg/bin/ffmpeg.exe; if it's
    missing we fall back to imageio-ffmpeg's bundled binary, then PATH.
    """
    bundled = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # last-resort PATH lookup


def render_transparent_video(
    ass_path: Path,
    output_path: Path,
    duration_seconds: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    codec: str = "prores_ks",
    pix_fmt: str = "yuva444p12le",
    extra_ffmpeg_args: Optional[List[str]] = None,
    opaque_background: bool = False,
) -> Path:
    """Burn the .ass into a video.

    Default (opaque_background=False): ProRes 4444 with alpha — the entire
    frame is fully transparent except for the rendered subtitle glyphs.
    DaVinci Resolve reads the alpha and stacks the result as an overlay
    above any main V1 footage.

    When opaque_background=True: solid black background, no alpha plane.
    Useful when the user wants the subtitles on a visibly-black canvas
    (e.g. previewing without compositing) and doesn't need DaVinci stacking.
    Drops the ProRes 4444 alpha profile and uses the supplied pix_fmt as-is.

    Manual ffmpeg command for the transparent default:
        ffmpeg -f lavfi -i "color=c=black@0.0:s=1920x1080:d=N:rate=FPS,format=rgba" \\
               -vf "ass=subtitles.ass:alpha=1,format=rgba" -c:v prores_ks \\
               -profile:v 4444 -pix_fmt yuva444p12le -an subtitles.mov

    Manual ffmpeg command for opaque black:
        ffmpeg -f lavfi -i "color=c=black:s=1920x1080:d=N:rate=FPS" \\
               -vf "ass=subtitles.ass" -c:v prores_ks -pix_fmt yuv444p \\
               -an subtitles.mov
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run ffmpeg in a clean CWD so the relative "ass=oscar.ass" path inside the
    # filter resolves even on Windows (which chokes on long absolute paths
    # containing em-dashes / non-ASCII in ffmpeg's argument parser).
    import os, subprocess, tempfile
    ffmpeg_bin = _resolve_ffmpeg_for_render()
    work = Path(tempfile.mkdtemp(prefix="subt_"))
    ass_link = work / ass_path.name
    try:
        ass_link.symlink_to(ass_path.resolve())
    except OSError:
        # On Windows, symlinks often need elevated perms; fall back to copy.
        import shutil
        shutil.copy2(ass_path, ass_link)

    if opaque_background:
        # Opaque black canvas — no alpha pipeline, no ProRes 4444 profile.
        # libass writes opaque glyphs by default, so the result is solid
        # black with gold text on top.
        src = f"color=c=black:s={width}x{height}:d={duration_seconds}:r={fps}"
        vf = f"ass={ass_link.name}"
        cmd: List[str] = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", src,
            "-vf", vf,
            "-c:v", codec,
            "-pix_fmt", pix_fmt,
            "-an",
            str(output_path),
        ]
    else:
        # Three alpha-preserving steps are required:
        #   1. Input source must be RGBA — color=c=black@0.0 alone still emits
        #      alpha=255 unless we explicitly chain ,format=rgba onto the lavfi
        #      source. Without this, the entire frame is opaque and the
        #      "transparent" overlay renders as a solid black rectangle.
        #   2. The ass filter must be passed alpha=1 — by default libass writes
        #      opaque (alpha=255) output even when given an alpha-capable input.
        #   3. After ass, format=rgba keeps the alpha plane intact on the way to
        #      the encoder. The final -pix_fmt (e.g. yuva444p12le) handles the
        #      encoder-side conversion to ProRes' native alpha pixel format.
        src = (
            f"color=c=black@0.0:s={width}x{height}:d={duration_seconds}:r={fps}"
            f",format=rgba"
        )
        vf = f"ass={ass_link.name}:alpha=1,format=rgba"
        cmd: List[str] = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", src,
            "-vf", vf,
            "-c:v", codec,
            "-profile:v", "4444",
            "-pix_fmt", pix_fmt,
            "-an",
            str(output_path),
        ]
    if extra_ffmpeg_args:
        cmd[8:8] = list(extra_ffmpeg_args)  # splice after -vf
    try:
        r = subprocess.run(cmd, cwd=str(work), capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (rc={r.returncode}): {r.stderr or r.stdout}")
    finally:
        # Best-effort cleanup; ignore errors so we don't mask the original failure.
        try:
            ass_link.unlink()
        except OSError:
            pass
        try:
            work.rmdir()
        except OSError:
            pass

    return output_path


def render_with_background_boxes(
    ass_path: Path,
    segments: List["SRTSegment"],
    output_path: Path,
    duration_seconds: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    codec: str = "prores_ks",
    pix_fmt: str = "yuva444p12le",
    font_size: int = 56,
    max_chars_per_line: int = 42,
    margin_v: int = 50,
    box_padding_x: int = 28,
    box_padding_y: int = 4,
    char_width_factor: float = 0.43,
    line_height_factor: float = 1.0,
    box_color: Tuple[int, int, int] = (0, 0, 0),
    anim_dur_s: float = 0.0,
    fade_up_delta: int = 30,
) -> Path:
    """Render a transparent video with per-cue opaque back-boxes.

    libass's `ass` filter does NOT write alpha=255 pixels for BorderStyle=3
    back-boxes, AND ffmpeg's `drawbox` filter preserves the source's alpha
    (it cannot write alpha=255 over an alpha=0 source). To get a true opaque
    box behind text (transparent elsewhere, stackable over footage in
    DaVinci) we use Pillow to render each cue's box as a single PNG with
    full RGBA (alpha=255 inside box, alpha=0 outside), then concatenate the
    box frames into a short video clip per cue and overlay them on a
    transparent base canvas using ffmpeg's `overlay` filter.

    Box dimensions are estimated from the wrapped text. Width = max line
    length * font_size * char_width_factor + 2 * padding. Height = line_count
    * font_size * line_height_factor + 2 * padding. The constants were
    calibrated against Arial bold — other fonts may need different factors.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import subprocess, tempfile, shutil
    from PIL import Image, ImageDraw

    work = Path(tempfile.mkdtemp(prefix="subt_"))
    try:
        # Materialize a temp .ass with BorderStyle=1 (text-only, no back-box)
        # so the ass filter doesn't fight the box overlay. Some ffmpeg builds
        # don't accept force_style.
        text_only_ass = work / "text_only.ass"
        original = Path(ass_path).read_text(encoding="utf-8")
        new_lines: List[str] = []
        for line in original.splitlines():
            if line.startswith("Style:"):
                prefix, _, rest = line.partition(":")
                fields = [f.strip() for f in rest.split(",")]
                if len(fields) >= 17:
                    fields[15] = "1"   # BorderStyle=1 (text+outline, no back-box)
                new_lines.append(f"{prefix}:" + ",".join(fields))
            else:
                new_lines.append(line)
        text_only_ass.write_text("\n".join(new_lines), encoding="utf-8")

        # Pre-render each cue's box as an RGBA PNG (opaque black rect,
        # alpha=255 inside the box, alpha=0 outside). One PNG per cue.
        # Each tuple is (png_path, start, end, tail_start, y_offset) where
        # tail_start/y_offset drive the FADE_UP animation: during the tail
        # window [tail_start, end], a second overlay of the same PNG appears
        # shifted upward by y_offset pixels, mirroring the ASS \move.
        box_pngs: List[Tuple[Path, float, float, Optional[float], int]] = []
        for i, seg in enumerate(segments):
            # Respect existing \N breaks — they were set by split_long_cues
            # before the ASS writer, and libass will honour them too.
            # Calling _wrap_text again would treat a single token like
            # "started\Ndisappearing." as one word and produce extra phantom
            # lines, making the box taller than what libass actually renders.
            existing_lines = seg.text.split("\\N") if "\\N" in seg.text else [seg.text]
            # Wrap each pre-existing line individually only if it overflows.
            final_lines: List[str] = []
            for ln in existing_lines:
                if len(ln) > max_chars_per_line and " " in ln:
                    final_lines.extend(_wrap_text(ln, max_chars_per_line).split("\\N"))
                else:
                    final_lines.append(ln)
            line_count = max(1, len(final_lines))
            max_line_chars = max(len(line) for line in final_lines)
            box_w = int(max_line_chars * font_size * char_width_factor) + 2 * box_padding_x
            box_h = int(line_count * font_size * line_height_factor) + 2 * box_padding_y
            box_w = min(box_w, width - 4)
            box_h = min(box_h, height - 4)
            box_x = max(0, (width - box_w) // 2)
            box_y = max(0, height - margin_v - box_h)

            png_path = work / f"box_{i:04d}.png"
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [box_x, box_y, box_x + box_w, box_y + box_h],
                fill=(*box_color, 255),
            )
            img.save(png_path, "PNG")
            # Compute tail window for FADE_UP animation. Only cues long enough
            # to fit the tail duration get a second (rising) overlay.
            seg_duration = float(seg.end_time - seg.start_time)
            if anim_dur_s > 0 and seg_duration > anim_dur_s:
                tail_start = float(seg.end_time) - anim_dur_s
                y_offset = fade_up_delta
            else:
                tail_start = None
                y_offset = 0
            box_pngs.append((png_path, float(seg.start_time), float(seg.end_time), tail_start, y_offset))

        # Build the filter chain via the helper.
        vf = _build_box_overlay_vf(box_pngs, text_only_ass.name)

        # Build the ffmpeg command. Input layout:
        #   0:        lavfi transparent canvas (color=c=black@0.0,format=rgba)
        #   1..N:     box PNGs (one per cue), looped as still frames
        # No `-i` for the .ass file — the ass filter reads it by filename.
        cmd: List[str] = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            # `,format=rgba` MUST be in the input spec — ffmpeg's lavfi color
            # filter defaults to yuv420p without it, which discards the alpha
            # plane (alpha=0 becomes a no-op).
            f"color=c=black@0.0:s={width}x{height}:d={duration_seconds}:r={fps},format=rgba",
        ]
        for png_path, _start, _end, _tail, _yo in box_pngs:
            # `-t duration_seconds` makes the PNG input terminate at the
            # same time as the base lavfi canvas — without this, ffmpeg
            # waits indefinitely for the infinite-loop PNG input.
            cmd.extend(["-loop", "1", "-t", f"{duration_seconds:.3f}", "-i", str(png_path)])
        cmd.extend([
            "-filter_complex", vf,
            "-map", "[out]",
            "-c:v", codec,
            "-profile:v", "4444",
            "-pix_fmt", pix_fmt,
            "-an",
            str(output_path),
        ])

        r = subprocess.run(
            cmd, cwd=str(work), capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (rc={r.returncode}): {r.stderr or r.stdout}"
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    return output_path


def _build_box_overlay_vf(
    box_pngs: List[Tuple[Path, float, float, float, int]],
    ass_name: str,
) -> str:
    """Build a -filter_complex expression that:
      1. starts from a transparent RGBA canvas (lavfi input 0),
      2. overlays each box PNG (inputs 1..N) timed to its cue window,
      3. applies the ass filter to draw text glyphs on top,
      4. emits the final stream on `[out]`.

    Each entry in `box_pngs` is `(png_path, start, end, tail_start, y_offset)`:
      - start/end: visibility window for the cue.
      - tail_start: when the box should start gliding upward (or None to skip).
      - y_offset: pixels to glide upward by tail_end (positive = up).

    During the tail window [tail_start, end] we interpolate the y offset
    linearly from 0 → y_offset so the box glides up smoothly — a sudden
    snap would be visible jitter to the eye. The ASS text uses the same
    linear glide via \\move, so box and text stay aligned throughout.
    """
    n_box = len(box_pngs)
    parts: List[str] = []

    # Step 1: tag the lavfi canvas (input 0) as RGBA → [base]
    parts.append("[0:v]format=rgba[base]")

    # Step 2: chain overlays per cue. ONE overlay per cue, with y as an
    # expression that interpolates from 0 to -y_offset during the tail
    # window. Two stacked overlays would either flicker at the boundary
    # or hide the motion (one stays at original y while the other slides).
    cur = "[base]"
    overlay_idx = 0
    for i, (_png, start, end, tail_start, y_offset) in enumerate(box_pngs, start=1):
        # Build the y expression. Three regimes:
        #   t < tail_start            → 0 (box at original position)
        #   tail_start ≤ t < end      → linear glide from 0 → -y_offset
        #   t ≥ end                   → clip to -y_offset (mostly invisible
        #                               because enable=False at this point)
        if tail_start is not None and tail_start < end and y_offset > 0:
            tail_dur = end - tail_start
            y_expr = (
                f"if(lt(t,{tail_start:.3f}),0,"
                f"-{y_offset}*(t-{tail_start:.3f})/{tail_dur:.3f})"
            )
        else:
            y_expr = "0"
        overlay_idx += 1
        out = f"[b{overlay_idx}]"
        parts.append(
            f"{cur}[{i}:v]overlay=enable='between(t,{start:.3f},{end:.3f})'"
            f":y='{y_expr}'{out}"
        )
        cur = out

    # Step 3: apply ass filter to the boxed canvas — text glyphs on top.
    parts.append(f"{cur}ass={ass_name}:alpha=1,format=rgba[out]")

    return ";".join(parts)


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CapCut-style styled subtitles from a project SRT.",
    )
    parser.add_argument("--project", type=Path, required=True, help="Project directory")
    parser.add_argument(
        "--voiceover", type=Path, default=None,
        help="Override voiceover path (auto-discovered otherwise). Accepts "
             ".srt or audio (.mp3/.wav/.m4a/.flac/.ogg/.aac/.wma/.opus); "
             "audio files are transcribed on the fly.",
    )
    parser.add_argument(
        "--whisper-model", type=str, default="base",
        help="faster-whisper model size for audio transcription (default: base). "
             "Options: tiny, base, small, medium, large, large-v2, large-v3, "
             "or a local path to a custom model directory.",
    )
    parser.add_argument(
        "--whisper-device", type=str, default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="faster-whisper inference device (default: cpu). Set 'cuda' if your "
             "machine has CUDA libs on PATH (e.g. cublas64_12.dll) and you want GPU "
             "transcription. 'auto' lets faster-whisper pick — fails fast if CUDA "
             "libs are missing.",
    )
    parser.add_argument(
        "--whisper-compute-type", type=str, default="auto",
        help="faster-whisper compute_type (default: auto). Common values: "
             "int8, int8_float16, int16, float16, float32. Use 'int8' for CPU.",
    )
    parser.add_argument(
        "--whisper-language", type=str, default=None,
        help="Force a specific language code for transcription (e.g. 'en', 'es'). "
             "Default: auto-detect.",
    )
    parser.add_argument(
        "--no-transcribe-cache", action="store_true",
        help="Force re-transcription of audio input even when a cached .srt "
             "sidecar exists. Default: reuse cached .srt if newer than the audio.",
    )
    parser.add_argument(
        "--style", type=str, default="capcut_default",
        help=f"Preset name. Available: {sorted(PRESETS.keys())}",
    )
    parser.add_argument(
        "--animation", type=str, default=None,
        choices=[a.value for a in Animation],
        help="Override animation (none/karaoke/pop/fade). Default: from preset.",
    )
    parser.add_argument("--font", type=str, default=None, help="Override font family")
    parser.add_argument("--font-size", type=int, default=None, help="Override font size")
    parser.add_argument("--primary-color", type=str, default=None, help="Override primary color (e.g. #FFFF00)")
    parser.add_argument(
        "--karaoke-color", type=str, default=None,
        help="Override the highlight (current-word) color (e.g. #00FFFF). "
             "Default: random bright color from a curated palette per generation.",
    )
    parser.add_argument("--outline-color", type=str, default=None, help="Override outline color")
    parser.add_argument(
        "--background-box", action="store_true",
        help="Render an opaque back-box behind text (ASS BorderStyle=3). "
             "Pairs with --background-color. Default: from preset.",
    )
    parser.add_argument(
        "--background-color", type=str, default=None,
        help="Opaque back-box color when --background-box is set (e.g. #000000).",
    )
    parser.add_argument(
        "--position", type=str, default=None, choices=[p.value for p in Position],
        help="Override position (top/center/bottom)",
    )
    parser.add_argument("--max-chars", type=int, default=None, help="Override max chars per line")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: <project>/subtitle_design)",
    )
    parser.add_argument(
        "--formats", type=str, default="srt,ass,otio",
        help="Comma-separated outputs to write (srt, ass, otio)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing files")
    parser.add_argument(
        "--no-random-highlight", action="store_true",
        help="Skip the random highlight color; use the preset's karaoke_color exactly.",
    )
    parser.add_argument(
        "--render-video", action="store_true",
        help="Also burn the .ass into a transparent ProRes 4444 MOV (DaVinci-compatible alpha overlay)",
    )
    parser.add_argument(
        "--video-codec", type=str, default="prores_ks",
        help="ffmpeg video codec for transparent render (default: prores_ks)",
    )
    parser.add_argument(
        "--video-container", type=str, default="mov",
        help="Output video container extension (default: mov)",
    )
    parser.add_argument(
        "--video-resolution", type=str, default="1920x1080",
        help="WidthxHeight for the transparent render (default: 1920x1080)",
    )
    parser.add_argument(
        "--video-fps", type=int, default=30,
        help="Frame rate for the transparent render (default: 30)",
    )
    parser.add_argument(
        "--opaque-background", action="store_true",
        help="Render the video with a solid black background instead of "
             "transparent alpha. Use when you want to preview the subtitles "
             "without compositing — drops the ProRes 4444 alpha profile.",
    )
    parser.add_argument(
        "--video-pix-fmt", type=str, default=None,
        help="ffmpeg output pix_fmt (default: yuva444p12le for transparent, "
             "yuv444p for --opaque-background). Override only if your codec "
             "requires a different format.",
    )
    parser.add_argument(
        "--box-padding-x", type=int, default=28,
        help="Horizontal padding inside the per-cue back-box, in pixels "
             "(default: 28). Lower = tighter box around the text.",
    )
    parser.add_argument(
        "--box-padding-y", type=int, default=4,
        help="Vertical padding inside the per-cue back-box, in pixels "
             "(default: 4). Lower = tighter box around the text.",
    )
    parser.add_argument(
        "--box-char-width-factor", type=float, default=0.43,
        help="Multiplier of font_size used to estimate per-character width "
             "for box sizing (default: 0.43). Lower = narrower box. "
             "Calibrated against Arial bold at 56pt; other fonts may need "
             "tuning.",
    )
    parser.add_argument(
        "--box-line-height-factor", type=float, default=1.0,
        help="Multiplier of font_size used to estimate line height for box "
             "sizing (default: 1.0). Lower = shorter box.",
    )
    parser.add_argument(
        "--box-color", type=str, default="000000",
        help="RGB hex color for the per-cue back-box (default: 000000). "
             "Use FFFFFF for white to debug box positioning.",
    )
    return parser.parse_args(argv)


def _apply_overrides(style: SubtitleStyle, args: argparse.Namespace) -> SubtitleStyle:
    """Apply CLI overrides on top of the preset style."""
    from dataclasses import replace
    overrides = {}
    if args.animation is not None:
        overrides["animation"] = Animation(args.animation)
    if args.font is not None:
        overrides["font_family"] = args.font
    if args.font_size is not None:
        overrides["font_size"] = args.font_size
    if args.primary_color is not None:
        overrides["primary_color"] = hex_to_ass_color(args.primary_color)
    if args.karaoke_color is not None:
        overrides["karaoke_color"] = hex_to_ass_color(args.karaoke_color)
    if args.outline_color is not None:
        overrides["outline_color"] = hex_to_ass_color(args.outline_color)
    if args.background_box:
        overrides["background_box"] = True
    if args.background_color is not None:
        overrides["background_color"] = hex_to_ass_color(args.background_color)
    if args.position is not None:
        overrides["position"] = Position(args.position)
    if args.max_chars is not None:
        overrides["max_chars_per_line"] = args.max_chars
    return replace(style, **overrides) if overrides else style


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    project_dir = args.project.resolve()
    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        return 1

    srt_path_arg = args.voiceover if args.voiceover else find_srt_path(project_dir)
    if srt_path_arg and Path(srt_path_arg).suffix.lower() in AUDIO_EXTENSIONS:
        # Audio source — transcribe to derive SRT + word timestamps next to
        # the audio file. Subsequent runs reuse the cached .srt if newer than
        # the audio. Pass --no-transcribe-cache to force re-transcription.
        audio = Path(srt_path_arg)
        srt_path = audio.with_suffix(".srt")
        words_path = srt_path.with_suffix(".words.json")
        srt_path = transcribe_audio_to_srt(
            audio_path=audio,
            output_srt=srt_path,
            output_words_json=words_path,
            model_size=args.whisper_model,
            language=args.whisper_language,
            use_cache=not args.no_transcribe_cache,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
        )
    elif srt_path_arg:
        srt_path = Path(srt_path_arg)
    else:
        logger.error(
            f"No SRT or audio found in {project_dir} (or in {project_dir}/voiceover/). "
            "Pass --voiceover to specify explicitly."
        )
        return 1

    if not srt_path.exists():
        logger.error(f"Voiceover path does not exist: {srt_path}")
        return 1

    output_dir = args.output_dir or (project_dir / "subtitle_design")
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}

    try:
        style = apply_preset(args.style)
    except ValueError as e:
        logger.error(str(e))
        return 1
    style = _apply_overrides(style, args)

    # Randomize the highlight color unless the user pinned it explicitly.
    # Only meaningful for KARAOKE / per-word-highlight animations — for NONE,
    # FADE, POP there's no per-word color, so skip the picker (its chosen
    # color would otherwise leak into highlight_position="first" overrides).
    if (
        args.karaoke_color is None
        and not args.no_random_highlight
        and style.animation == Animation.KARAOKE
    ):
        from dataclasses import replace as _replace
        chosen = pick_random_highlight_color()
        style = _replace(style, karaoke_color=chosen)
        logger.info(f"Highlight color (random): {chosen}")

    logger.info(f"Project:  {project_dir}")
    logger.info(f"SRT:      {srt_path}")
    logger.info(f"Style:    {args.style} (animation={style.animation.value}, position={style.position.value})")
    logger.info(f"Output:   {output_dir}")
    logger.info(f"Formats:  {sorted(formats)}")

    segments = parse_srt_file(str(srt_path))
    if not segments:
        logger.error(f"No segments parsed from {srt_path}")
        return 1
    logger.info(f"Loaded {len(segments)} cues")

    if style.auto_split_long_lines and style.max_chars_per_line > 0:
        segments = split_long_cues(segments, style.max_chars_per_line)

    words_data = load_word_timestamps(srt_path) if "ass" in formats else []
    if words_data:
        logger.info(f"Loaded word timestamps for {sum(len(s.get('words') or []) for s in words_data)} words")

    if args.dry_run:
        logger.info("Dry run — would write:")
        if "srt" in formats:
            logger.info(f"  {output_dir / 'subtitles.srt'}")
        if "ass" in formats:
            logger.info(f"  {output_dir / 'subtitles.ass'}")
        if "otio" in formats:
            logger.info(f"  {output_dir / 'subtitles.otio'}")
        if args.render_video:
            video_ext = args.video_container.lstrip(".")
            if style.background_box and not args.opaque_background:
                mode = "transparent with per-cue back-boxes"
            elif args.opaque_background:
                mode = "opaque black"
            else:
                mode = "transparent"
            logger.info(f"  {output_dir / f'subtitles.{video_ext}'}  ({mode} {args.video_codec}, {args.video_resolution})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if "srt" in formats:
        srt_out = output_dir / "subtitles.srt"
        write_styled_srt(segments, srt_out)
        written.append(srt_out)

    if "ass" in formats:
        ass_out = output_dir / "subtitles.ass"
        write_animated_ass(segments, words_data, style, ass_out)
        written.append(ass_out)

    if "otio" in formats:
        otio_out = output_dir / "subtitles.otio"
        build_subtitle_otio(
            ass_path=output_dir / "subtitles.ass",
            segments=segments,
            output_path=otio_out,
        )
        written.append(otio_out)

    if args.render_video:
        ass_for_video = output_dir / "subtitles.ass"
        if not ass_for_video.exists():
            logger.error(f"--render-video requires subtitles.ass; rerun with --formats srt,ass")
            return 1
        video_ext = args.video_container.lstrip(".")
        video_out = output_dir / f"subtitles.{video_ext}"
        try:
            w_str, h_str = args.video_resolution.lower().split("x", 1)
            width, height = int(w_str), int(h_str)
        except ValueError:
            logger.error(f"--video-resolution must be WIDTHxHEIGHT, got {args.video_resolution!r}")
            return 1
        # Duration = last cue's end_time (with small buffer for the last cue's tail)
        duration = max(1.0, segments[-1].end_time + 0.5) if segments else 1.0

        # Pick a renderer:
        #   background_box in style → drawbox per cue (gives a true opaque
        #     box behind text; libass's BorderStyle=3 doesn't write alpha=255
        #     for the box, so we draw it ourselves and let the ass filter
        #     render text-only with force_style='BorderStyle,1').
        #   --opaque-background → solid black canvas (entire screen is black).
        #   default → transparent canvas (alpha=0 outside text glyphs).
        if style.background_box and not args.opaque_background:
            pix_fmt = args.video_pix_fmt or "yuva444p12le"
            logger.info(
                f"Rendering transparent video with per-cue back-boxes: "
                f"{width}x{height}@{args.video_fps}fps, {duration:.2f}s, "
                f"codec={args.video_codec}, pix_fmt={pix_fmt}"
            )
            # FADE_UP cues need a tail window where the box rises + fades to
            # mirror the text. Non-FADE_UP animations get anim_dur_s=0 and the
            # box overlay emits only the static (non-rising) version.
            anim_dur_s = (
                style.anim_duration_ms / 1000.0
                if style.animation == Animation.FADE_UP else 0.0
            )
            render_with_background_boxes(
                ass_path=ass_for_video,
                segments=segments,
                output_path=video_out,
                duration_seconds=duration,
                width=width,
                height=height,
                fps=args.video_fps,
                codec=args.video_codec,
                pix_fmt=pix_fmt,
                font_size=style.font_size,
                max_chars_per_line=style.max_chars_per_line,
                margin_v=style.margin_v,
                box_padding_x=args.box_padding_x,
                box_padding_y=args.box_padding_y,
                char_width_factor=args.box_char_width_factor,
                line_height_factor=args.box_line_height_factor,
                box_color=tuple(int(args.box_color[i:i+2], 16) for i in (0, 2, 4)),
                anim_dur_s=anim_dur_s,
                fade_up_delta=30,
            )
        else:
            pix_fmt = args.video_pix_fmt or ("yuv444p" if args.opaque_background else "yuva444p12le")
            mode_str = "opaque black" if args.opaque_background else "transparent"
            logger.info(
                f"Rendering {mode_str} video: {width}x{height}@{args.video_fps}fps, "
                f"{duration:.2f}s, codec={args.video_codec}, pix_fmt={pix_fmt}"
            )
            render_transparent_video(
                ass_path=ass_for_video,
                output_path=video_out,
                duration_seconds=duration,
                width=width,
                height=height,
                fps=args.video_fps,
                codec=args.video_codec,
                pix_fmt=pix_fmt,
                opaque_background=args.opaque_background,
            )
        written.append(video_out)

    logger.info("Wrote:")
    for p in written:
        logger.info(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
