#!/usr/bin/env python3
"""
distribute_ref_pictures.py — Download N reference pictures of a person/entity and
evenly distribute them as still-image clips across the span of an SRT onto a NEW
single-track OTIO file. The source OTIO is read-only and provides per-image
durations: each image's duration matches the source-OTIO clip whose [start, end)
contains the image's distribution midpoint, falling back to the covering SRT
segment, then a uniform slice.

Usage:
    python scripts/distribute_ref_pictures.py \\
        --srt "<srt_path>" \\
        --otio "<source_otio_path>" \\
        --subject "Mario Moreno Reyes" \\
        --max-images 25 \\
        [--output "<new_otio>"] [--context "..."] [--dry-run]

The output is a separate OTIO (default: <otio_stem>.ref_pictures.otio next to the
source) plus a JSON summary. The source OTIO is never modified.

Refuses to overwrite the source OTIO: --output must differ from --otio.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)

# Ensure stdout/stderr can encode non-ASCII characters (LLM-inferred
# subjects and SRT slot names routinely contain accents, umlauts, etc.).
# On Windows the default code page is cp1252 which crashes on \u0100+.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

from src.otio.utils import _to_windows_path, _has_problematic_path
from src.llm_client import create_client, LLMRequest, ResponseFormat
from src.llm_client.exceptions import LLMClientError
from src.minimax_image import MiniMaxImageProvider, aspect_for_size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("distribute-ref-pictures")

RATE_DEFAULT = 30.0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_KIND = getattr(otio.schema.TrackKind.Video, "name", "Video")


def parse_index_ranges(spec: str) -> Set[int]:
    """Parse '4-181,6-183,114-228' or '4,5,10-12' into a set of excluded indices.

    Returns an empty set on empty input. Raises ValueError on malformed entries.
    """
    out: Set[int] = set()
    spec = (spec or "").strip()
    if not spec:
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                lo, hi = int(a.strip()), int(b.strip())
            except ValueError as exc:
                raise ValueError(f"Bad range token: {token!r}") from exc
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(token))
            except ValueError as exc:
                raise ValueError(f"Bad index token: {token!r}") from exc
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _slugify(name: str, max_len: int = 40) -> str:
    """Match scripts/download_reference_pictures.py:slugify so glob slugs align."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s[:max_len] or "subject"


def _prompt_key(path: Path) -> str:
    """Strip trailing _NN variant suffix to group variants of the same prompt.

    e.g. Amish_breeder_with_draft_horses_202607262304_2.jpeg
       -> Amish_breeder_with_draft_horses_202607262304

    Variant suffixes are 1-2 digits (imagedl returns at most ~6 variants).
    Timestamps are 8+ digits and must NOT be stripped.
    """
    return re.sub(r"_\d{1,2}$", "", path.stem)


def _interleave_variants(paths: List[Path]) -> List[Path]:
    """
    Round-robin interleave images so consecutive positions never hold two variants
    of the same prompt (e.g. Amish_breeder_..._2 then Amish_breeder_..._3).
    """
    buckets: dict[str, List[Path]] = {}
    for p in paths:
        buckets.setdefault(_prompt_key(p), []).append(p)
    # Stable ordering of buckets (sorted by first occurrence via list).
    bucket_list = list(buckets.values())
    out: List[Path] = []
    max_len = max((len(b) for b in bucket_list), default=0)
    for i in range(max_len):
        for bucket in bucket_list:
            if i < len(bucket):
                out.append(bucket[i])
    return out


def _seconds_to_tc(seconds: float, rate: float) -> str:
    """Convert seconds to HH:MM:SS:FF timecode string."""
    total_frames = int(round(seconds * rate))
    ff = total_frames % int(rate)
    ss = (total_frames // int(rate)) % 60
    mm = (total_frames // (int(rate) * 60)) % 60
    hh = total_frames // (int(rate) * 3600)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def parse_srt(srt_path: Path) -> List[Tuple[float, float, str]]:
    """Parse SRT into (start_sec, end_sec, text) tuples. Inline, no extra imports."""
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    segments: List[Tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            int(lines[0].strip())
            timing = lines[1].strip()
        except (ValueError, IndexError):
            continue
        m = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            timing,
        )
        if not m:
            continue

        def _ts(ts: str) -> float:
            ts = ts.replace(",", ".")
            h, mm, s = ts.split(":")
            sec, frac = s.split(".")
            return int(h) * 3600 + int(mm) * 60 + float(sec) + float(frac) / 1000.0

        start = _ts(m.group(1))
        end = _ts(m.group(2))
        text = "\n".join(lines[2:]).strip()
        segments.append((start, end, text))
    return segments


def derive_subject(srt_path: Path) -> str:
    """Best-effort query from SRT filename: spaces, strip suffixes."""
    stem = srt_path.stem
    s = stem.replace("_", " ").strip()
    s = re.sub(r"\b(voiceover|vo|en|es|srt|final|v\d+)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s or stem


def _sample_srt_text(
    srt_segments: List[Tuple[float, float, str]],
    n: int,
) -> str:
    """Return a transcript-like string of `n` segments spread evenly across the SRT."""
    if not srt_segments:
        return ""
    if n <= 0 or n >= len(srt_segments):
        return "\n".join(text for _, _, text in srt_segments)
    # Pick indices evenly spaced across the timeline.
    step = max(1, len(srt_segments) // n)
    picked = srt_segments[::step][:n]
    return "\n".join(text for _, _, text in picked)


def _detect_name_in_segment(
    seg_text: str,
    provider: str,
    model: str,
    host: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (name, provider) for the first named entity in `seg_text`.

    Tries Ollama first (when provider is "ollama"), then falls back to
    `src.matching.scoring.extract_named_entities_from_text`. Returns
    (None, None) when nothing is found.
    """
    cleaned = " ".join(seg_text.split()).strip()
    if not cleaned:
        return None, None
    if provider == "ollama":
        name = _detect_name_ollama(cleaned, model, host)
        if name:
            return name, "ollama"
    try:
        from src.matching.scoring import extract_named_entities_from_text
    except Exception as exc:
        logger.debug(f"Heuristic name extractor unavailable: {exc}")
        return None, None
    try:
        entities = extract_named_entities_from_text(cleaned) or []
    except Exception as exc:
        logger.debug(f"Heuristic name extractor failed: {exc}")
        return None, None
    for candidate in entities:
        if candidate and candidate.strip():
            return candidate.strip(), "heuristic"
    return None, None


def _detect_name_ollama(seg_text: str, model: str, host: Optional[str]) -> Optional[str]:
    """Best-effort one-shot Ollama call asking for a JSON array of names."""
    try:
        client_kwargs = {}
        if host:
            client_kwargs["host"] = host
        client = create_client(
            provider="ollama",
            model=model,
            cache_dir=".cache/llm_responses",
            cache_ttl_hours=168,
            **client_kwargs,
        )
    except LLMClientError as exc:
        logger.debug(f"Ollama client init failed: {exc}")
        return None
    request = LLMRequest(
        prompt=(
            f"Voiceover line:\n{seg_text}\n\n"
            "Output a JSON array of the named people, places, or organizations "
            "mentioned. If none, output []. Output ONLY the JSON array, no prose."
        ),
        max_tokens=60,
        temperature=0.0,
        response_format=ResponseFormat.TEXT,
        cache_key_prefix=f"distribute_ref_pictures.name.{model}",
        use_cache=True,
        timeout=20,
    )
    try:
        response = client.generate(request)
    except LLMClientError as exc:
        logger.debug(f"Ollama name detection failed: {exc}")
        return None
    text = (getattr(response, "text", "") or "").strip()
    if not text:
        return None
    parsed: object = None
    try:
        import json as _json
        parsed = _json.loads(text)
    except Exception:
        m = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if m:
            try:
                parsed = [x.strip().strip('"').strip("'") for x in m.group(1).split(",") if x.strip()]
            except Exception:
                parsed = None
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                return item.strip()
    elif isinstance(parsed, dict):
        for key in ("names", "entities"):
            value = parsed.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
    if text and not text.startswith("["):
        first = text.splitlines()[0].strip().strip('"').strip("'")
        if 1 <= len(first) <= 80 and not first.endswith("."):
            return first
    return None



def infer_subject_with_llm(
    srt_segments: List[Tuple[float, float, str]],
    provider: str,
    model: str,
    host: Optional[str] = None,
    samples: int = 12,
    cache_dir: str = ".cache/llm_responses",
) -> Optional[str]:
    """
    Use a local or remote LLM to infer the main subject of the SRT narration.
    Returns a short image-search query, or None on any failure.
    """
    try:
        client_kwargs = {}
        if provider == "ollama":
            client_kwargs["host"] = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = create_client(
            provider=provider,
            model=model,
            cache_dir=cache_dir,
            cache_ttl_hours=168,  # 7d — transcripts are stable
            **client_kwargs,
        )
    except LLMClientError as exc:
        logger.warning(f"LLM client init failed ({provider}/{model}): {exc}")
        return None

    transcript = _sample_srt_text(srt_segments, samples)
    if not transcript:
        return None

    system_prompt = (
        "You extract the main visual subject from a voiceover transcript for an "
        "image search engine. Output a single short noun phrase (1-6 words). "
        "Names: return the full name. Topics/places/objects: return a concise "
        "descriptive phrase. Examples: 'Ada Lovelace', 'Stirling Castle', "
        "'Maya Angelou', 'Boeing 747 cockpit', '1950s New York street'. "
        "Output ONLY the phrase. No quotes, no period, no preamble, no "
        "explanation, no translation, no 'The subject is'."
    )
    request = LLMRequest(
        prompt=(
            f"Voiceover language: {transcript[:0] and 'detect'}\n"
            f"Transcript:\n{transcript}\n\n"
            f"Output exactly one short subject phrase (1-6 words) for image search:"
        ),
        system_prompt=system_prompt,
        max_tokens=20,
        temperature=0.1,
        response_format=ResponseFormat.TEXT,
        cache_key_prefix=f"distribute_ref_pictures.subject.{model}",
        use_cache=False,  # cache is keyed on prompt only, not model — bypass for subject inference
        timeout=90,
    )

    try:
        response = client.generate(request)
    except LLMClientError as exc:
        logger.warning(f"LLM subject inference failed: {exc}")
        return None

    text = (response.text or "").strip()
    # Take first line; strip quotes / bullets / "Subject:" preambles.
    text = text.splitlines()[0].strip().strip(".,;:!?'\"`*")
    text = re.sub(r"^(subject\s*:?\s*|[-*>`]|the\s+|a\s+)", "", text, flags=re.IGNORECASE).strip()
    # Reject obvious preamble / explanation text.
    bad_starts = (
        "parece", "it seems", "the text", "this text", "based on", "i think",
        "i would", "based upon", "the transcript", "here is", "of course",
        "the voiceover", "the narration",
    )
    lower = text.lower()
    if any(lower.startswith(b) for b in bad_starts):
        logger.warning(f"LLM returned preamble, discarding: {text!r}")
        return None
    # Cap to ~6 words.
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8]).rstrip(".,;:!?'\"`")
    if not text or len(text) < 2:
        return None
    logger.info(
        f"LLM subject inference: provider={provider} model={model} "
        f"samples={min(samples, len(srt_segments))}/{len(srt_segments)} -> {text!r}"
    )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Source OTIO: enumerate existing clips
# ─────────────────────────────────────────────────────────────────────────────


def collect_existing_slots(
    timeline: otio.schema.Timeline,
    track_index: int,
    rate: float,
) -> Tuple[List[Tuple[float, float, int, str]], int]:
    """
    Walk the chosen video track; return
    (slots, source_total_frames) where each slot is
    (start_sec, end_sec, dur_frames, clip_name).
    """
    video_tracks = [t for t in timeline.tracks if str(t.kind) == VIDEO_KIND]
    if not video_tracks:
        return [], 0
    if track_index < 0 or track_index >= len(video_tracks):
        logger.warning(
            f"--track-index {track_index} out of range (0..{len(video_tracks)-1}); using 0"
        )
        track_index = 0
    track = video_tracks[track_index]

    slots: List[Tuple[float, float, int, str]] = []
    cursor_frames = 0
    for item in track:
        dur_frames = int(item.duration().value)
        if dur_frames <= 0:
            cursor_frames += max(0, dur_frames)
            continue
        is_clip = isinstance(item, otio.schema.Clip)
        enabled = getattr(item, "enabled", True)
        if is_clip and enabled:
            start_sec = cursor_frames / rate
            end_sec = (cursor_frames + dur_frames) / rate
            name = getattr(item, "name", "") or f"clip_{cursor_frames}"
            slots.append((start_sec, end_sec, dur_frames, name))
        cursor_frames += dur_frames
    return slots, cursor_frames


def find_topmost_visible_ranges(
    timeline: otio.schema.Timeline,
    rate: float,
) -> List[Tuple[float, float]]:
    """
    Walk the timeline at every unique start moment and pick the highest-track
    enabled, non-gap, non-audio clip with an ExternalReference — that clip is
    what is *visible* at that moment, per the same rule used by
    /re-download-otio:find_top_most_clips. Returns merged (start_sec, end_sec)
    ranges of every visible-clip coverage region.
    """
    all_clips: List[dict] = []
    for track_idx, track in enumerate(timeline.tracks):
        if not getattr(track, "enabled", True):
            continue
        is_audio = track.kind == otio.schema.TrackKind.Audio
        tp = RationalTime(0, rate)
        for clip in track:
            dur = clip.duration()
            all_clips.append(
                {
                    "track_idx": track_idx,
                    "is_audio": is_audio,
                    "is_gap": isinstance(clip, otio.schema.Gap),
                    "clip_enabled": getattr(clip, "enabled", True),
                    "start": tp.value / rate,
                    "end": (tp + dur).value / rate,
                    "mr": getattr(clip, "media_reference", None),
                }
            )
            tp = tp + dur

    # At each unique start, find the topmost visible clip and record its end.
    coverage_end_at: dict[float, float] = {}
    for seg_start in sorted({c["start"] for c in all_clips}):
        covering = [
            c
            for c in all_clips
            if c["start"] <= seg_start < c["end"]
            and not c["is_gap"]
            and not c["is_audio"]
            and c["clip_enabled"]
            and isinstance(c["mr"], otio.schema.ExternalReference)
        ]
        if covering:
            top = max(covering, key=lambda x: x["track_idx"])
            coverage_end_at[seg_start] = top["end"]

    # Merge contiguous ranges (where the next start <= current end).
    merged: List[Tuple[float, float]] = []
    if not coverage_end_at:
        return merged
    sorted_starts = sorted(coverage_end_at)
    cur_start = sorted_starts[0]
    cur_end = coverage_end_at[cur_start]
    for s in sorted_starts[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, coverage_end_at[s])
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, coverage_end_at[s]
    merged.append((cur_start, cur_end))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Slot planning
# ─────────────────────────────────────────────────────────────────────────────


def plan_slots(
    n: int,
    srt_segments: List[Tuple[float, float, str]],
    existing_slots: List[Tuple[float, float, int, str]],
    rate: float,
) -> List[dict]:
    """
    Evenly distribute N image centers across the SRT span. For each center, find
    the source-OTIO clip containing it; fall back to the covering SRT segment,
    then nearest SRT segment, then a uniform slice.
    """
    if n <= 0:
        return []
    srt_start = srt_segments[0][0]
    srt_end = max(e for _, e, _ in srt_segments)
    total_srt_dur = max(0.0, srt_end - srt_start)
    if total_srt_dur <= 0:
        return []

    slice_dur = total_srt_dur / n
    plan: List[dict] = []
    for i in range(n):
        center_sec = srt_start + (i + 0.5) * slice_dur
        # Priority 1: SRT segment containing center (SRT-first policy)
        dur_frames: Optional[int] = None
        dur_source = "uniform"
        slot_name = ""
        for ss, se, _ in srt_segments:
            if ss <= center_sec < se:
                dur_frames = max(1, int(round((se - ss) * rate)))
                dur_source = "srt"
                break
        # Priority 2: source OTIO clip containing center (fallback)
        if dur_frames is None:
            for ss, se, df, name in existing_slots:
                if ss <= center_sec < se:
                    dur_frames = df
                    dur_source = "otio"
                    slot_name = name
                    break
        # Priority 3: nearest SRT segment by center distance
        if dur_frames is None:
            best = min(
                srt_segments,
                key=lambda seg: min(abs(center_sec - seg[0]), abs(center_sec - seg[1])),
            )
            dur_frames = max(1, int(round((best[1] - best[0]) * rate)))
            dur_source = "srt_nearest"
        # Priority 4: uniform slice (always reaches this if all else failed)
        if dur_frames is None or dur_source == "uniform":
            dur_frames = max(1, int(round(slice_dur * rate)))
        plan.append(
            {
                "index": i,
                "center_sec": center_sec,
                "dur_frames": dur_frames,
                "dur_source": dur_source,
                "slot_name": slot_name,
            }
        )
    # Resolve placements: center-anchored, clamp to prev_end to prevent overlap.
    cursor = 0
    for entry in plan:
        desired = int(round(entry["center_sec"] * rate)) - entry["dur_frames"] // 2
        start_frames = max(cursor, desired)
        entry["start_frames"] = start_frames
        cursor = start_frames + entry["dur_frames"]
    return plan


def plan_gap_slots(
    srt_segments: List[Tuple[float, float, str]],
    covered_ranges: List[Tuple[float, float]],
    rate: float,
    threshold: float = 0.5,
    max_gap_dur: Optional[float] = None,
    detect_names: bool = False,
    name_provider: str = "ollama",
    name_model: str = "mistral:7b",
    name_host: Optional[str] = None,
) -> List[dict]:
    """Plan one slot per continuous uncovered region. SRT is used only for
    query text (and entity detection when detect_names is true). Each
    gap becomes a single plan entry with dur_source="continuous_gap";
    V-Ref mirrors source coverage (has Gap items where source has video).
    """
    if not srt_segments:
        return []
    timeline_start = srt_segments[0][0]
    timeline_end = max(end for _, end, _ in srt_segments)
    coverage = []
    for start, end in covered_ranges:
        start = max(timeline_start, start)
        end = min(timeline_end, end)
        if end > start:
            coverage.append((start, end))
    coverage.sort()
    merged = []
    for start, end in coverage:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    gaps = []
    cursor = timeline_start
    for start, end in merged + [(timeline_end, timeline_end)]:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)

    gap_entries = []
    for start, end in gaps:
        duration = end - start
        if max_gap_dur is not None and duration > max_gap_dur:
            logger.info(
                f"Skipping gap [{start:.2f}s, {end:.2f}s] (duration {duration:.1f}s "
                f"> --max-gap-duration {max_gap_dur}s)"
            )
            continue
        overlapping = [text for ss, se, text in srt_segments if se > start and ss < end]
        query_text = " ".join(" ".join(overlapping).split())
        detected_name = None
        detected_provider = None
        if detect_names and overlapping:
            for seg_text in overlapping:
                name, provider = _detect_name_in_segment(
                    seg_text, name_provider, name_model, name_host
                )
                if name:
                    detected_name = name
                    detected_provider = provider
                    break
        gap_entries.append({
            "kind": "gap",
            "center_sec": (start + end) / 2,
            "start_sec": start,
            "end_sec": end,
            "dur_sec": duration,
            "dur_source": "continuous_gap",
            "slot_name": query_text[:80] or f"gap_{int(start)}",
            "srt_text": query_text,
            "detected_name": detected_name,
            "detected_provider": detected_provider,
        })

    plan = []
    for entry in gap_entries:
        plan.append({
            "index": len(plan),
            "center_sec": entry["center_sec"],
            "start_frames": max(0, int(round(entry["start_sec"] * rate))),
            "dur_frames": max(1, int(round(entry["dur_sec"] * rate))),
            "dur_source": entry["dur_source"],
            "slot_name": entry["slot_name"],
            "srt_text": entry["srt_text"],
            "detected_name": entry["detected_name"],
            "detected_provider": entry["detected_provider"],
        })
    return plan


def _evenly_subsample_plan(plan: List[dict], n: int) -> List[dict]:
    """
    Pick `n` entries from `plan`, evenly distributed across the index range so
    the timeline is covered from start to end (rather than just slicing off
    the first N). Used in --fill-gaps mode when fewer images came back than
    gaps were planned.
    """
    M = len(plan)
    if n >= M or n <= 0:
        return plan[:n] if n < M else list(plan)
    if n == 1:
        indices = [0]
    else:
        indices = sorted({int(round(i * (M - 1) / (n - 1))) for i in range(n)})
    sub = [plan[i] for i in indices]
    for i, entry in enumerate(sub):
        entry["index"] = i
    return sub


def _pad_images_to_plan_length(
    plan: List[dict],
    image_paths: List[Path],
    per_candidate: Optional[List[List[Path]]] = None,
) -> Tuple[List[dict], List[Path], Optional[List[List[Path]]]]:
    """
    In --fill-gaps mode the V-Ref track must have zero gaps — every planned
    gap region has to end up with an image. When downloads return fewer
    images than planned gaps, instead of subsampling the plan (which leaves
    Gap items between clips in the V-Ref track), pad the missing slots by
    cycling through the successful images so every gap has *some* image.

    Returns (plan, image_paths, per_candidate) with lengths aligned to
    len(plan). If image_paths is empty, returns the originals unchanged
    (the caller already warned about an empty V-Ref track).
    """
    if not plan or len(image_paths) >= len(plan):
        return plan, image_paths, per_candidate
    if not image_paths:
        return plan, image_paths, per_candidate

    padded = list(image_paths)
    while len(padded) < len(plan):
        padded.append(image_paths[len(padded) % len(image_paths)])
    logger.warning(
        f"--fill-gaps: padded {len(plan) - len(image_paths)} slot(s) by "
        f"replicating successful images to keep the V-Ref track gap-free "
        f"({len(image_paths)} unique → {len(plan)} total)"
    )

    padded_per_candidate = None
    if per_candidate is not None:
        padded_per_candidate = []
        for cand_paths in per_candidate:
            if not cand_paths or len(cand_paths) >= len(plan):
                padded_per_candidate.append(cand_paths[: len(plan)] if cand_paths else cand_paths)
                continue
            padded_cand = list(cand_paths)
            while len(padded_cand) < len(plan):
                padded_cand.append(cand_paths[len(padded_cand) % len(cand_paths)])
            padded_per_candidate.append(padded_cand)

    return plan, padded, padded_per_candidate


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────


def run_download(
    repo_root: Path,
    subject: str,
    tmp_dir: Path,
    max_images: int,
    min_size_kb: int,
    context: Optional[str],
    sources: Optional[str],
    per_source_limit: Optional[int],
    exclude_domains: Optional[str],
    keep_tmp: bool,
    query_override: Optional[str] = None,
    timeout: Optional[float] = None,
) -> int:
    argv = [
        sys.executable,
        "-u",  # unbuffered stdout so the parent's `tail` sees progress live
        str(repo_root / "scripts" / "download_reference_pictures.py"),
        "--query",
        query_override or subject,
        "--output",
        str(tmp_dir),
        "--max",
        str(max_images),
        "--min-size-kb",
        str(min_size_kb),
    ]
    if context:
        argv += ["--context", context]
    if sources:
        argv += ["--sources", sources]
    if per_source_limit is not None:
        argv += ["--per-source-limit", str(per_source_limit)]
    if exclude_domains is not None:
        argv += ["--exclude-domains", exclude_domains]
    if keep_tmp:
        argv += ["--keep-workdir"]
    logger.info(f"Running downloader: {' '.join(argv)}")
    if timeout:
        logger.info(f"Downloader timeout: {timeout:.0f}s")
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            timeout=timeout if timeout and timeout > 0 else None,
        )
        return proc.returncode
    except FileNotFoundError as exc:
        logger.error(f"Downloader not found: {exc}")
        return 3
    except subprocess.TimeoutExpired as exc:
        # pyimagedl has no per-image timeout, so a stuck Bing/Yandex request
        # can stall the whole subprocess indefinitely. Kill the child and let
        # the caller proceed with whatever images were saved before the
        # stall (the tmp_dir is collected by the parent either way).
        logger.warning(
            f"Downloader exceeded {exc.timeout:.0f}s timeout — killed child "
            f"and continuing with any images saved to {tmp_dir} before the stall."
        )
        return 4


def build_minimax_prompt(
    subject: str,
    context: Optional[str],
    slot_name: str = "",
    srt_text: str = "",
) -> str:
    """Compose a prompt from the topic and the specific voiceover line."""
    parts = [subject.strip()]
    if srt_text and srt_text.strip():
        parts.append(srt_text.strip())
    elif slot_name and slot_name.strip():
        parts.append(slot_name.strip())
    if context and context.strip():
        parts.append(context.strip())
    return ", ".join(p for p in parts if p)


def parse_size_spec(spec: str) -> Tuple[int, int]:
    """Parse a 'WxH' string (e.g. '1792x1024') into a (width, height) tuple."""
    w, _, h = spec.lower().partition("x")
    return int(w), int(h)


def generate_minimax_for_slots(
    plan: List[dict],
    subject: str,
    context: Optional[str],
    tmp_dir: Path,
    size: Tuple[int, int],
    aspect_ratio: Optional[str],
) -> Tuple[int, List[Path]]:
    """Generate one MiniMax image per slot in `plan`.

    Returns (rc, image_paths):
      rc=0 + non-empty list: success
      rc=1: bad args / missing key
      rc=2: zero usable images
      rc=3: unexpected provider exception
    """
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        logger.error("--provider minimax requires MINIMAX_API_KEY in env.")
        return 1, []
    try:
        provider = MiniMaxImageProvider(api_key=api_key)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1, []

    image_paths: List[Path] = []
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for entry in plan:
        prompt = build_minimax_prompt(
            subject,
            context,
            entry.get("slot_name", ""),
            entry.get("srt_text", ""),
        )
        try:
            out_path = provider.generate_image_to_file(
                prompt,
                tmp_dir / f"minimax_{entry['index']:03d}.jpg",
                size=size,
                aspect_ratio=aspect_ratio,
            )
        except Exception as exc:
            logger.warning(
                f"MiniMax generation failed for slot {entry['index']}: {exc}"
            )
            continue
        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.warning(f"MiniMax returned empty file for slot {entry['index']}")
            continue
        image_paths.append(out_path)

    if not image_paths:
        return 2, []
    return 0, image_paths


def collect_downloaded(
    tmp_dir: Path,
    subject: str,
    min_size_kb: int,
    exclude_domains: Optional[str] = None,
    exclude_indices: Optional[Set[int]] = None,
) -> List[Path]:
    """Glob <tmp>/<slug>_NN.<ext>, sorted by NN. Drop tiny + unicode-broken files + blacklisted domains + content duplicates + user-excluded indices."""
    slug = _slugify(subject)
    min_bytes = max(min_size_kb * 1024, 1024)
    excluded = [d.strip().lower() for d in (exclude_domains or "").split(",") if d.strip()]
    excluded_idx = exclude_indices or set()
    out: List[Tuple[int, Path]] = []
    seen_hashes: Set[str] = set()
    for f in tmp_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        if _has_problematic_path(str(f)):
            logger.warning(f"Skipping file with problematic path: {f}")
            continue
        if f.stat().st_size < min_bytes:
            logger.warning(f"Skipping {f.name} ({f.stat().st_size} bytes < {min_bytes})")
            continue
        # Safety net: re-check the sidecar JSON for blacklisted source URLs
        # (the downloader's pre-filter uses candidate URLs; if imagedl's final
        # resolved host was Alamy/etc, the sidecar will still reveal it).
        if excluded:
            sidecar = f.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    meta = {}
                blob = json.dumps(meta).lower()
                if any(d in blob for d in excluded):
                    logger.warning(f"Skipping blacklisted-domain image: {f.name}")
                    continue
        m = re.match(rf"^{re.escape(slug)}_(\d+)(?:\.[^.]+)?$", f.stem)
        if not m:
            continue
        idx = int(m.group(1))
        if idx in excluded_idx:
            logger.warning(f"Skipping user-excluded index {idx}: {f.name}")
            continue
        # Content-hash dedup so the same picture returned twice by imagedl isn't placed twice.
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()
        except OSError as exc:
            logger.warning(f"Could not hash {f.name}: {exc}")
            continue
        if h in seen_hashes:
            logger.debug(f"Skipping duplicate content: {f.name}")
            continue
        seen_hashes.add(h)
        out.append((int(m.group(1)), f))
    out.sort(key=lambda p: p[0])
    return _interleave_variants([p for _, p in out])


def collect_existing_images(
    image_dirs: List[Path],
    min_size_kb: int,
    exclude_indices: Optional[Set[int]] = None,
) -> List[Path]:
    """
    Glob all image files from given directories. Dedupes by md5 content hash so the
    same picture in two folders counts once. Drops tiny files and unicode-broken paths.
    Sort order: by directory then filename (stable, deterministic).
    If exclude_indices is given, files whose stem matches <slug>_NN.<ext> with NN
    in the set are skipped (matches filenames produced by the downloader).
    """
    min_bytes = max(min_size_kb * 1024, 1024)
    excluded_idx = exclude_indices or set()
    idx_re = re.compile(r"^(?P<slug>.+?)_(?P<nn>\d+)(?:\.[^.]+)?$")
    seen: Set[str] = set()
    out: List[Path] = []
    for img_dir in image_dirs:
        if not img_dir.exists():
            logger.warning(f"Image dir not found: {img_dir}")
            continue
        for f in sorted(img_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            if _has_problematic_path(str(f)):
                logger.warning(f"Skipping file with problematic path: {f}")
                continue
            if f.stat().st_size < min_bytes:
                logger.warning(f"Skipping {f.name} ({f.stat().st_size} bytes < {min_bytes})")
                continue
            if excluded_idx:
                m = idx_re.match(f.stem)
                if m and int(m.group("nn")) in excluded_idx:
                    logger.warning(f"Skipping user-excluded index {int(m.group('nn'))}: {f.name}")
                    continue
            try:
                h = hashlib.md5(f.read_bytes()).hexdigest()
            except OSError as exc:
                logger.warning(f"Could not hash {f.name}: {exc}")
                continue
            if h in seen:
                logger.debug(f"Skipping duplicate content: {f}")
                continue
            seen.add(h)
            out.append(f)
    logger.info(
        f"Collected {len(out)} unique images from {len(image_dirs)} dir(s)"
    )
    return _interleave_variants(out)


def verify_fill_timing(
    plan: List[dict],
    timeline: otio.schema.Timeline,
    track_names: List[str],
    rate: float,
) -> None:
    """Fail if any generated fill track's clips drift from the planned SRT frame windows."""
    if not plan:
        return
    for track_name in track_names:
        track = next((t for t in timeline.tracks if t.name == track_name), None)
        if track is None:
            raise ValueError(f"Missing generated track {track_name!r}")
        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        if len(clips) == 0 and len(plan) != 0:
            # Pass yielded no usable images (e.g. downloader timeout fired
            # before the first save completed). Skip timing verification so
            # --all can continue past the empty V-Ref track instead of
            # aborting the whole --all run.
            logger.warning(
                f"Track {track_name!r} has 0 clips but plan had {len(plan)}; "
                f"skipping fill-timing verification for this track."
            )
            continue
        if len(clips) != len(plan):
            raise ValueError(
                f"Track {track_name!r}: expected {len(plan)} fill clips, found {len(clips)}"
            )
        for entry, clip in zip(plan, clips):
            actual_start = int(round(clip.range_in_parent().start_time.value))
            actual_duration = int(round(clip.duration().value))
            expected_start = int(entry["start_frames"])
            expected_duration = int(entry["dur_frames"])
            if (actual_start, actual_duration) != (expected_start, expected_duration):
                raise ValueError(
                    f"Fill clip {entry['index']} on {track_name!r} timing mismatch: "
                    f"expected start={expected_start}, duration={expected_duration}; "
                    f"got start={actual_start}, duration={actual_duration}"
                )


def build_timeline(
    plan: List[dict],
    image_paths: List[Path],
    track_name: str,
    rate: float,
    subject: str,
    source_total_frames: int,
    match_duration: bool,
    candidates: int = 1,
    candidate_image_paths: Optional[List[List["Path"]]] = None,
    candidate_track_names: Optional[List[str]] = None,
) -> otio.schema.Timeline:
    """Build a new OTIO with one or more parallel reference tracks."""
    new_tl = otio.schema.Timeline(name="ref_pictures")
    new_tl.metadata["Resolve_OTIO"] = {"Resolve OTIO Meta Version": "1.0"}
    new_tl.global_start_time = RationalTime(0, rate)

    if candidates < 1:
        raise ValueError("candidates must be >= 1")
    if candidate_image_paths is None:
        candidate_image_paths = [image_paths]
    if candidate_track_names is None:
        candidate_track_names = [track_name]

    total_appended = 0
    final_cursor = 0
    for cand_idx, (cand_paths, cand_track_name) in enumerate(
        zip(candidate_image_paths, candidate_track_names)
    ):
        track = otio.schema.Track(
            name=cand_track_name, kind=otio.schema.TrackKind.Video
        )
        cursor = 0
        appended = 0
        for entry, file in zip(plan, cand_paths):
            start_frames = entry["start_frames"]
            dur_frames = entry["dur_frames"]
            if start_frames > cursor:
                # 1-frame slop from frame-rate rounding: nudge the clip
                # forward instead of inserting a Gap so the track stays
                # gap-free in --fill-gaps mode.
                if start_frames - cursor <= 1:
                    logger.debug(
                        f"Slop absorb: cursor={cursor}, planned start={start_frames}; "
                        f"shifting clip to cursor instead of inserting Gap."
                    )
                    start_frames = cursor
                else:
                    gap = otio.schema.Gap(
                        source_range=TimeRange(
                            start_time=RationalTime(0, rate),
                            duration=RationalTime(start_frames - cursor, rate),
                        )
                    )
                    track.append(gap)
                    cursor = start_frames

            abs_url = _to_windows_path(str(file))
            if _has_problematic_path(abs_url):
                logger.warning(f"Skipping clip with problematic path: {abs_url}")
                continue

            media_ref = otio.schema.ExternalReference(
                target_url=abs_url,
                available_range=TimeRange(
                    start_time=RationalTime(0, rate),
                    duration=RationalTime(dur_frames, rate),
                ),
            )
            media_ref.name = f"{file.parent.name}_{file.name}"

            clip = otio.schema.Clip(
                name=f"REF_{entry['index']:03d}:{file.name}",
                media_reference=media_ref,
                source_range=TimeRange(
                    start_time=RationalTime(0, rate),
                    duration=RationalTime(dur_frames, rate),
                ),
            )
            clip.effects.append(otio.schema.FreezeFrame())
            clip.metadata["Resolve_OTIO"] = {}
            clip.metadata["is_still_image"] = True
            clip.metadata["image_index"] = entry["index"]
            clip.metadata["image_path"] = str(file)
            clip.metadata["srt_center_sec"] = entry["center_sec"]
            clip.metadata["dur_source"] = entry["dur_source"]
            clip.metadata["subject"] = subject
            clip.metadata["candidate_index"] = cand_idx
            clip.metadata["candidate_track_name"] = cand_track_name
            if entry.get("slot_name"):
                clip.metadata["source_otio_clip"] = entry["slot_name"]

            track.append(clip)
            cursor += dur_frames
            appended += 1

        if match_duration and source_total_frames > cursor:
            gap = otio.schema.Gap(
                source_range=TimeRange(
                    start_time=RationalTime(0, rate),
                    duration=RationalTime(source_total_frames - cursor, rate),
                )
            )
            track.append(gap)
            cursor = source_total_frames

        new_tl.tracks.append(track)
        total_appended += appended
        final_cursor = max(final_cursor, cursor)

    logger.info(
        f"Built V-Ref timeline: {candidates} track(s), {total_appended} clips total, "
        f"{final_cursor} frames ({final_cursor / rate:.2f}s)"
    )
    return new_tl


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (candidates / summary)
# ─────────────────────────────────────────────────────────────────────────────


def candidate_track_names(
    base_name: str, prefix: Optional[str], candidates: int
) -> List[str]:
    """Return per-candidate track names like '<prefix>{NN:02d}'."""
    effective_prefix = prefix if prefix else f"{base_name}-Cand"
    return [f"{effective_prefix}{idx:02d}" for idx in range(candidates)]


def build_name_detection_block(
    *, args, plan: List[dict]
) -> Optional[dict]:
    """Build the top-level `name_detection` block for the summary JSON."""
    if not getattr(args, "detect_names", False):
        return None
    providers = sorted(
        {entry.get("detected_provider") for entry in plan if entry.get("detected_provider")}
    )
    detected = [
        {
            "index": entry["index"],
            "name": entry["detected_name"],
            "provider": entry.get("detected_provider"),
        }
        for entry in plan
        if entry.get("detected_name")
    ]
    return {
        "enabled": True,
        "provider": args.name_provider,
        "providers_used": providers,
        "detected": detected,
    }


def build_summary(
    *,
    args,
    subject: str,
    srt_path: Path,
    otio_path: Path,
    output_path: Path,
    rate: float,
    source_total_frames: int,
    srt_start: float,
    srt_end: float,
    plan: List[dict],
    per_candidate: List[List[Path]],
    track_names: List[str],
    actual: int,
    source_image_dirs: Optional[List[Path]],
    provider: Optional[str],
    name_detection: Optional[dict] = None,
) -> dict:
    """Build the per-run summary JSON, including per-candidate entries."""
    candidates = args.candidates
    flat_images = []
    per_gap_candidates = []
    for entry in plan:
        slot = {
            "track": track_names[0] if track_names else args.track_name,
            "path": str(per_candidate[0][entry["index"]])
            if per_candidate and entry["index"] < len(per_candidate[0])
            else None,
            "index_in_candidate": entry["index"],
        }
        cand_entries = []
        for cand_idx, cand_paths in enumerate(per_candidate):
            cand_entries.append(
                {
                    "track": track_names[cand_idx] if cand_idx < len(track_names) else args.track_name,
                    "path": str(cand_paths[entry["index"]])
                    if entry["index"] < len(cand_paths)
                    else None,
                    "index_in_candidate": entry["index"],
                }
            )
            if candidates == 1:
                flat_images.append(
                    {
                        "index": entry["index"],
                        "path": cand_entries[-1]["path"],
                        "center_sec": entry["center_sec"],
                        "start_frame": entry["start_frames"],
                        "dur_frames": entry["dur_frames"],
                        "dur_source": entry["dur_source"],
                        "slot_name": entry.get("slot_name", ""),
                        "srt_text": entry.get("srt_text", ""),
                        "query": build_minimax_prompt(
                            subject, args.context, entry.get("slot_name", ""), entry.get("srt_text", "")
                        ),
                    }
                )
        per_gap_candidates.append(
            {
                "index": entry["index"],
                "start_frame": entry["start_frames"],
                "dur_frames": entry["dur_frames"],
                "center_sec": entry["center_sec"],
                "srt_text": entry.get("srt_text", ""),
                "query": build_minimax_prompt(
                    subject, args.context, entry.get("slot_name", ""), entry.get("srt_text", "")
                ),
                "detected_name": entry.get("detected_name"),
                "detected_name_provider": entry.get("detected_provider"),
                "candidates": cand_entries,
            }
        )

    summary: dict = {
        "subject": subject,
        "candidates": candidates,
        "track_names": track_names,
        "srt": str(srt_path),
        "source_otio": str(otio_path),
        "output_otio": str(output_path),
        "frame_rate": rate,
        "track_name": track_names[0] if track_names else args.track_name,
        "requested": args.max_images,
        "downloaded": actual,
        "source_total_frames": source_total_frames,
        "srt_start_sec": srt_start,
        "srt_end_sec": srt_end,
        "gaps": per_gap_candidates,
    }
    if candidates == 1:
        summary["images"] = flat_images
    if source_image_dirs is not None:
        summary["source_image_dirs"] = [str(p) for p in source_image_dirs]
    if provider is not None:
        summary["provider"] = provider
    if name_detection is not None:
        summary["name_detection"] = name_detection
    return summary


def print_plan_table(
    plan: List[dict],
    rate: float,
    candidates: int,
    track_names: List[str],
    per_candidate: List[List[Path]],
) -> None:
    """Print a per-gap table; multi-candidate runs add a per-candidate path column."""
    if candidates == 1:
        print(
            f"\n{'idx':>3} | {'center (TC)':>12} | {'center (s)':>10} | "
            f"{'dur_f':>6} | {'start_f':>7} | {'dur_source':<12} | slot_name"
        )
        print("-" * 100)
        for entry in plan:
            print(
                f"{entry['index']:>3} | "
                f"{_seconds_to_tc(entry['center_sec'], rate):>12} | "
                f"{entry['center_sec']:>10.3f} | "
                f"{entry['dur_frames']:>6} | "
                f"{entry['start_frames']:>7} | "
                f"{entry['dur_source']:<12} | "
                f"{entry.get('slot_name', '')}"
            )
        return
    cand_cols = " | ".join(f"cand{i:02d}" for i in range(candidates))
    print(
        f"\n{'idx':>3} | {'center (TC)':>12} | {'center (s)':>10} | "
        f"{'dur_f':>6} | {'start_f':>7} | {cand_cols}"
    )
    print("-" * (60 + 9 * candidates))
    for entry in plan:
        paths = []
        for cand_idx, cand_paths in enumerate(per_candidate):
            idx = entry["index"]
            paths.append(
                cand_paths[idx].name
                if idx < len(cand_paths)
                else "-"
            )
        print(
            f"{entry['index']:>3} | "
            f"{_seconds_to_tc(entry['center_sec'], rate):>12} | "
            f"{entry['center_sec']:>10.3f} | "
            f"{entry['dur_frames']:>6} | "
            f"{entry['start_frames']:>7} | "
            + " | ".join(f"{p}" for p in paths)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download N reference pictures and evenly distribute them as still "
            "clips across an SRT onto a new OTIO file. The source OTIO is "
            "read-only."
        ),
    )
    p.add_argument("--srt", required=True, help="SRT file (narration schedule).")
    p.add_argument("--otio", required=True, help="Source OTIO (read-only).")
    p.add_argument(
        "--output",
        default=None,
        help="Output OTIO path. Default: <otio_stem>.ref_pictures.otio next to source.",
    )
    p.add_argument(
        "--subject",
        default=None,
        help=(
            "pyimagedl query. Default: infer with LLM (see --llm-*); "
            "fall back to SRT filename if LLM unavailable or --no-llm."
        ),
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM subject inference; use the SRT-filename heuristic.",
    )
    p.add_argument(
        "--llm-provider",
        default="ollama",
        choices=["ollama", "gemini", "anthropic"],
        help="LLM provider for subject inference (default: ollama).",
    )
    p.add_argument(
        "--llm-model",
        default="mistral:7b",
        help="LLM model name (default: mistral:7b). llama3.2 (3.2B) is too weak to follow instructions.",
    )
    p.add_argument(
        "--llm-host",
        default=None,
        help="Ollama host URL (default: $OLLAMA_HOST or http://localhost:11434).",
    )
    p.add_argument(
        "--llm-samples",
        type=int,
        default=12,
        help="Number of SRT segments to feed the LLM (default 12).",
    )
    p.add_argument("--max-images", type=int, default=25, help="Target image count (default 25). Ignored in --fill-gaps mode (image count = number of uncovered SRT segments).")
    p.add_argument(
        "--fill-gaps",
        action="store_true",
        help=(
            "Place one still image per SRT segment that lacks video coverage in "
            "the source OTIO, instead of evenly distributing N images across the SRT span."
        ),
    )
    p.add_argument(
        "--gap-coverage-mode",
        choices=["topmost", "track"],
        default="topmost",
        help=(
            "How to decide if an SRT segment is covered. 'topmost' (default) uses "
            "/re-download-otio's rule: a moment is covered iff the highest-track "
            "enabled non-gap non-audio clip covers it. 'track' looks only at "
            "--track-index (defaults to V1)."
        ),
    )
    p.add_argument(
        "--gap-threshold",
        type=float,
        default=0.5,
        help=(
            "Coverage fraction below which an SRT segment counts as a gap in "
            "--fill-gaps mode (default 0.5 = majority uncovered). 0.0 = require "
            "zero coverage; 1.0 = flag every segment."
        ),
    )
    p.add_argument(
        "--max-gap-duration",
        type=float,
        default=None,
        help=(
            "Skip SRT segments longer than this many seconds when filling gaps "
            "(default: no limit). Useful when one SRT line is 60s+ and you only "
            "want to patch small gaps."
        ),
    )
    p.add_argument(
        "--candidates",
        type=int,
        default=1,
        help=(
            "Download this many candidate pictures per gap and lay each on its own "
            "OTIO video track (--fill-gaps only, default 1). Tracks are frame-locked "
            "to identical gap start/end timing; enable/disable them in Resolve to "
            "compare candidates."
        ),
    )
    p.add_argument(
        "--candidate-track-prefix",
        default=None,
        help=(
            "Track name prefix for candidate tracks (default: '<track-name>-Cand'). "
            "Final names look like '<prefix>{00..N-1}'."
        ),
    )
    p.add_argument(
        "--detect-names",
        action="store_true",
        help=(
            "In --fill-gaps mode, detect named entities per SRT segment via "
            "Ollama (falling back to a heuristic extractor). When a name is "
            "found, the per-gap image query becomes the name itself; gaps "
            "without a detected name fall back to the standard subject+SRT query."
        ),
    )
    p.add_argument(
        "--name-provider",
        choices=["ollama"],
        default="ollama",
        help="LLM provider for name detection (default ollama).",
    )
    p.add_argument("--context", default=None, help="Optional search context phrase.")
    p.add_argument("--min-size-kb", type=int, default=8, help="Min image size in KB (default 8).")
    p.add_argument("--sources", default=None, help="Comma-separated imagedl sources (passed to downloader).")
    p.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Max candidates per (query, source) (passed to downloader).",
    )
    p.add_argument(
        "--downloader-timeout",
        type=float,
        default=600.0,
        help=(
            "Hard wall-clock timeout (seconds) for each pyimagedl downloader "
            "subprocess (default 600s = 10 min). When the downloader exceeds "
            "this, the child is killed and the run continues with whatever "
            "images were saved before the stall. pyimagedl has no per-image "
            "timeout, so a single stuck Bing/Yandex request can otherwise "
            "hang the whole pass indefinitely. Pass 0 to disable the limit."
        ),
    )
    p.add_argument(
        "--exclude-domains",
        default=(
            "alamy.com,alamy.ae,alamy.de,alamy.es,alamy.fr,alamy.it,alamy-media.co.uk,"
            "shutterstock.com,gettyimages.com,dreamstime.com,vecteezy.com,"
            "123rf.com,istockphoto.com,stocksy.com,depositphotos.com,stock.adobe.com,"
            "ftcdn.net,as.ftcdn.net,"  # Adobe Stock CDN hosts (image files served here have watermarks)
            "pond5.com,pixels.com,fineartamerica.com,redbubble.com,canstockphoto.com,"
            "pixtastock.com,yayimages.com,colourbox.com,masterfile.com"
        ),
        help=(
            "Comma-separated domains to skip (default filters Alamy/Shutterstock/Getty/"
            "Dreamstime/Vecteezy/Adobe Stock watermarks and other stock-photo hosts). "
            "Includes Adobe Stock CDN hosts (ftcdn.net) since stock.adobe.com alone doesn't "
            "match the actual image URLs. Applied at downloader AND post-download safety-net. "
            "Pass empty string to disable."
        ),
    )
    p.add_argument(
        "--exclude-indices",
        default="",
        help=(
            "Skip downloaded images whose numeric index (the NN in <slug>_NN.<ext>) "
            "falls in any of the comma-separated ranges. Examples: '4-181,114-228' "
            "or '4,6,9-12'. Index refers to the downloader's enumeration order, not "
            "the final layout position. Applied at collection time so excluded files "
            "are never placed in the OTIO."
        ),
    )
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="Image download dir. Default: <output_dir>/ref_pictures.",
    )
    p.add_argument(
        "--existing-images-dir",
        action="append",
        default=[],
        help=(
            "Use already-downloaded images from this dir instead of running the "
            "imagedl downloader. Repeatable. Implies --no-llm and requires --subject. "
            "Files are deduped by content hash across all listed dirs."
        ),
    )
    p.add_argument("--keep-tmp", action="store_true", help="Keep image dir after run.")
    p.add_argument("--frame-rate", type=float, default=RATE_DEFAULT, help="Timeline rate (default 30.0).")
    p.add_argument("--track-name", default="V-Ref", help="New track name (default 'V-Ref').")
    p.add_argument(
        "--track-index",
        type=int,
        default=0,
        help="Which video track of source OTIO to read durations from (default 0).",
    )
    p.add_argument(
        "--match-duration",
        action="store_true",
        help="Trailing gap so new track equals source OTIO total duration.",
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only; do not download or write.")
    p.add_argument("--verbose", action="store_true", help="Extra logging.")
    p.add_argument(
        "--provider",
        choices=("pyimagedl", "minimax"),
        default="pyimagedl",
        help="Image source. 'pyimagedl' (default) downloads stock photos via the existing "
             "downloader; 'minimax' generates images via MiniMax's image-01 model. "
             "Requires MINIMAX_API_KEY in env when 'minimax' is selected.",
    )
    p.add_argument(
        "--minimax-aspect",
        default=None,
        help="Optional MiniMax aspect_ratio override (e.g. '16:9'); default derived from "
             "the configured image size. Ignored unless --provider minimax.",
    )
    p.add_argument(
        "--minimax-size",
        default="1792x1024",
        help="WxH for MiniMax-generated images (default 1792x1024 = 16:9). "
             "Ignored unless --provider minimax.",
    )
    p.add_argument(
        "--minimax-fill",
        action="store_true",
        help=(
            "Shorthand for `--fill-gaps --provider minimax`: write one MiniMax-"
            "generated AI still per uncovered SRT segment to "
            "<otio_stem>_fill_gaps.otio. Requires MINIMAX_API_KEY in env. "
            "Mutually exclusive with --fill-gaps, --provider, --all, and "
            "--existing-images-dir."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run three sequential passes against the same SRT/OTIO pair and write "
            "three new OTIO files: (1) fill-gaps with MiniMax, "
            "(2) normal distribution with MiniMax, (3) normal distribution with "
            "pyimagedl. Shares the parsed SRT, source OTIO, and LLM-derived subject "
            "across passes. The pyimagedl pass uses Google+Bing+Yandex only "
            "(Wikipedia is excluded to avoid 403-PDF dead-ends). Each MiniMax pass "
            "uses --all-minimax-images (default 20). Mutually exclusive with "
            "--fill-gaps, --provider, and --minimax-fill."
        ),
    )
    p.add_argument(
        "--all-minimax-images",
        type=int,
        default=20,
        help="MiniMax image count for the second pass of --all (default 20). "
             "Only valid with --all.",
    )
    p.add_argument(
        "--all-skip-minimax",
        action="store_true",
        help="Skip the MiniMax pass in --all mode (useful when MINIMAX_API_KEY is "
             "unset). Only valid with --all.",
    )
    return p.parse_args()


def _run_all(
    args: argparse.Namespace,
    prelude: dict,
    src_tl,
    exclude_indices: Set[int],
) -> int:
    """Run three sequential passes against the same SRT/OTIO pair.

    Pass 1: fill-gaps with MiniMax (one AI still per uncovered SRT segment).
    Pass 2: normal distribution with MiniMax (--all-minimax-images, default 20).
    Pass 3: normal distribution with pyimagedl (Google + Bing + Yandex, no Wikipedia).

    Each pass produces its own OTIO + summary + _images/ dir. The shared
    prelude avoids re-parsing the SRT, re-reading the OTIO, and re-running
    the LLM subject inference.
    """
    srt_path = prelude["srt_path"]
    otio_path = prelude["otio_path"]
    srt_segments = prelude["srt_segments"]
    srt_start = prelude["srt_start"]
    srt_end = prelude["srt_end"]
    source_total_frames = prelude["source_total_frames"]
    existing_slots = prelude["existing_slots"]
    rate = prelude["rate"]
    subject = prelude["subject"]

    pyimagedl_sources = "GoogleImageClient,BingImageClient,YandexImageClient"

    # Each entry: kwargs to copy onto the cloned args namespace for the pass.
    # (output / tmp_dir default to None so each pass uses the convention derived
    # from its fill_gaps/provider values inside run_pass.)
    plans = [
        {
            "label": "fill-gaps with MiniMax",
            "fill_gaps": True,
            "provider": "minimax",
            "max_images": args.max_images,
            "sources": None,
            "description": "fill-gaps, MiniMax",
        },
        {
            "label": "normal distribution with MiniMax",
            "fill_gaps": False,
            "provider": "minimax",
            "max_images": args.all_minimax_images,
            "sources": None,
            "description": f"normal, MiniMax ({args.all_minimax_images} images)",
        },
        {
            "label": "normal distribution with pyimagedl",
            "fill_gaps": False,
            "provider": "pyimagedl",
            "max_images": args.max_images,
            "sources": pyimagedl_sources,
            "description": f"normal, pyimagedl ({args.max_images} images)",
        },
    ]
    if args.all_skip_minimax:
        plans.pop(1)

    logger.info(f"[--all] {len(plans)} passes queued:")
    for i, p in enumerate(plans, 1):
        logger.info(f"[--all]  Pass {i}/{len(plans)}: {p['description']}")

    pass_results: List[Tuple[int, str]] = []
    for i, plan_cfg in enumerate(plans, 1):
        logger.info(f"[--all] === Pass {i}/{len(plans)}: {plan_cfg['label']} ===")
        pass_args = copy.copy(args)
        pass_args.fill_gaps = plan_cfg["fill_gaps"]
        pass_args.provider = plan_cfg["provider"]
        pass_args.max_images = plan_cfg["max_images"]
        pass_args.sources = plan_cfg["sources"]
        # Pick a non-colliding output path per pass. Pass 1 (fill-gaps) and
        # pass 3 (normal pyimagedl) use the script's default suffixes; pass 2
        # (MiniMax) uses _ref_pictures_minimax.otio to avoid clobbering pass 3.
        if pass_args.fill_gaps:
            pass_args.output = str(otio_path.with_name(otio_path.stem + "_fill_gaps.otio"))
        elif pass_args.provider == "minimax":
            pass_args.output = str(
                otio_path.with_name(otio_path.stem + "_ref_pictures_minimax.otio")
            )
        else:
            pass_args.output = str(otio_path.with_name(otio_path.stem + "_ref_pictures.otio"))
        rc = run_pass(
            args=pass_args,
            prelude=prelude,
            src_tl=src_tl,
            exclude_indices=exclude_indices,
        )
        pass_results.append((rc, pass_args.output))
        logger.info(f"[--all] Pass {i}/{len(plans)} complete: rc={rc}, output={pass_args.output}")

    logger.info(f"[--all] Done. {len(plans)} passes:")
    for i, (rc, out) in enumerate(pass_results, 1):
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        logger.info(f"[--all]  Pass {i}: {status} -> {out}")
    return max((rc for rc, _ in pass_results), default=0)


def run_pass(
    args: argparse.Namespace,
    prelude: dict,
    src_tl,
    exclude_indices: Set[int],
) -> int:
    """Run a single distribution pass.

    Expects the shared prelude (SRT path / OTIO path / parsed SRT segments /
    SRT start+end / source total frames / existing slots / frame rate / subject)
    to already be resolved. Resolves output_path + tmp_dir from `args`, plans
    slots, downloads/generates images, builds the OTIO, writes the summary,
    and returns the script's exit code.
    """
    srt_path = prelude["srt_path"]
    otio_path = prelude["otio_path"]
    srt_segments = prelude["srt_segments"]
    srt_start = prelude["srt_start"]
    srt_end = prelude["srt_end"]
    source_total_frames = prelude["source_total_frames"]
    existing_slots = prelude["existing_slots"]
    rate = prelude["rate"]
    subject = prelude["subject"]

    output_path = (
        Path(args.output).resolve()
        if args.output
        else otio_path.with_name(
            otio_path.stem + ("_fill_gaps.otio" if args.fill_gaps else "_ref_pictures.otio")
        )
    )
    if output_path == otio_path:
        logger.error("Refusing to overwrite source OTIO: --output must differ from --otio")
        return 2

    if args.candidates < 1:
        logger.error("--candidates must be >= 1")
        return 2
    if args.candidates > 1 and not args.fill_gaps:
        logger.error("--candidates > 1 requires --fill-gaps (per-gap semantics)")
        return 2

    tmp_dir = (
        Path(args.tmp_dir).resolve()
        if args.tmp_dir
        else output_path.with_name(output_path.stem + "_images")
    )
    logger.info(
        f"SRT: {srt_path}  |  Source OTIO: {otio_path}  |  Output: {output_path}"
    )

    if args.fill_gaps:
        if args.gap_coverage_mode == "topmost":
            covered_ranges = find_topmost_visible_ranges(src_tl, rate)
            logger.info(
                f"Coverage mode: topmost-visible across {len(src_tl.tracks)} tracks "
                f"({len(covered_ranges)} merged coverage regions)"
            )
        else:
            covered_ranges = [(cs, ce) for cs, ce, _, _ in existing_slots]
            logger.info(
                f"Coverage mode: track {args.track_index} only "
                f"({len(covered_ranges)} clip regions)"
            )
        plan = plan_gap_slots(
            srt_segments=srt_segments,
            covered_ranges=covered_ranges,
            rate=rate,
            threshold=args.gap_threshold,
            max_gap_dur=args.max_gap_duration,
            detect_names=args.detect_names,
            name_provider=args.name_provider,
            name_model=args.llm_model,
            name_host=args.llm_host,
        )
        # In fill-gaps mode, image count is determined by the OTIO+SRT, not --max-images.
        if not plan:
            logger.warning(
                "No SRT segments below --gap-threshold "
                f"{args.gap_threshold:.2f}; nothing to fill."
            )
            return 0
        args.max_images = len(plan)
        logger.info(
            f"Fill-gaps plan: {len(plan)} image slots "
            f"(threshold={args.gap_threshold:.2f})"
        )
    else:
        plan = plan_slots(args.max_images, srt_segments, existing_slots, rate)
    if not plan:
        logger.error("Slot planning produced no entries")
        return 2

    # Existing-images branch: skip download + imagedl entirely, use files in --existing-images-dir.
    if args.existing_images_dir:
        image_dirs = [Path(p).resolve() for p in args.existing_images_dir]
        if args.candidates > 1 and len(image_dirs) == args.candidates:
            per_candidate = [
                collect_existing_images(
                    image_dirs=[d],
                    min_size_kb=args.min_size_kb,
                    exclude_indices=exclude_indices,
                )
                for d in image_dirs
            ]
        elif args.candidates > 1:
            all_images = collect_existing_images(
                image_dirs=image_dirs,
                min_size_kb=args.min_size_kb,
                exclude_indices=exclude_indices,
            )
            per_candidate = [[] for _ in range(args.candidates)]
            for i, img in enumerate(all_images):
                per_candidate[i % args.candidates].append(img)
        else:
            per_candidate = [
                collect_existing_images(
                    image_dirs=image_dirs,
                    min_size_kb=args.min_size_kb,
                    exclude_indices=exclude_indices,
                )
            ]
        track_names = candidate_track_names(args.track_name, args.candidate_track_prefix, args.candidates)
        min_count = min(len(c) for c in per_candidate)
        if min_count < len(plan):
            if args.fill_gaps:
                # Fill-gaps mode: never leave a gap in V-Ref — replicate successful images.
                plan, downloaded_unused, per_candidate = _pad_images_to_plan_length(
                    plan, per_candidate[0] if per_candidate else [], per_candidate
                )
            else:
                logger.warning(
                    f"Per-candidate short: smallest candidate has {min_count} images; "
                    f"subsampling plan to {min_count} gap(s)"
                )
                plan = _evenly_subsample_plan(plan, min_count)
        per_candidate = [c[: len(plan)] for c in per_candidate]
        actual = min_count
        downloaded = per_candidate[0] if per_candidate else []
        logger.info(
            f"Existing-images mode: {args.candidates} candidate(s); "
            f"{[len(c) for c in per_candidate]} image(s) per candidate "
            f"(requested {args.max_images})"
        )
        if actual == 0:
            logger.warning("0 images in --existing-images-dir; writing OTIO with empty V-Ref track.")
            new_tl = build_timeline(
                plan=[],
                image_paths=[],
                track_name=args.track_name,
                rate=rate,
                subject=subject,
                source_total_frames=source_total_frames,
                match_duration=args.match_duration,
                candidates=args.candidates,
                candidate_image_paths=per_candidate,
                candidate_track_names=track_names,
            )
        else:
            new_tl = build_timeline(
                plan=plan,
                image_paths=per_candidate[0],
                track_name=args.track_name,
                rate=rate,
                subject=subject,
                source_total_frames=source_total_frames,
                match_duration=args.match_duration,
                candidates=args.candidates,
                candidate_image_paths=per_candidate,
                candidate_track_names=track_names,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(new_tl, str(output_path))

        if args.fill_gaps:
            verify_fill_timing(plan, new_tl, track_names, rate)

        summary = build_summary(
            args=args,
            subject=subject,
            srt_path=srt_path,
            otio_path=otio_path,
            output_path=output_path,
            rate=rate,
            source_total_frames=source_total_frames,
            srt_start=srt_start,
            srt_end=srt_end,
            plan=plan,
            per_candidate=per_candidate,
            track_names=track_names,
            actual=actual,
            source_image_dirs=image_dirs,
            provider=None,
            name_detection=build_name_detection_block(args=args, plan=plan),
        )
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Wrote {output_path}")
        logger.info(f"Wrote {summary_path}")

        print_plan_table(plan, rate, args.candidates, track_names, per_candidate)
        return 0

    # MiniMax branch: generate AI stills instead of downloading stock photos.
    if args.provider == "minimax":
        try:
            mm_size = parse_size_spec(args.minimax_size)
        except ValueError as exc:
            logger.error(f"Bad --minimax-size {args.minimax_size!r}: {exc}")
            return 2
        logger.info(
            f"--provider minimax: generating AI stills "
            f"(size={mm_size[0]}x{mm_size[1]}, aspect={args.minimax_aspect or aspect_for_size(mm_size)})"
        )
        tmp_dir.mkdir(parents=True, exist_ok=True)
        per_candidate = []
        for cand_idx in range(args.candidates):
            cand_subdir = tmp_dir / f"cand_{cand_idx:02d}"
            cand_subject = f"{subject} cand{cand_idx:02d}" if cand_idx > 0 else subject
            rc, cand_paths = generate_minimax_for_slots(
                plan=plan,
                subject=cand_subject,
                context=args.context,
                tmp_dir=cand_subdir,
                size=mm_size,
                aspect_ratio=args.minimax_aspect,
            )
            if rc == 1:
                return 3
            per_candidate.append(cand_paths)
        track_names = candidate_track_names(args.track_name, args.candidate_track_prefix, args.candidates)
        min_count = min((len(c) for c in per_candidate), default=0)
        if min_count < len(plan):
            if args.fill_gaps:
                # Fill-gaps mode: never leave a gap in V-Ref — replicate successful images.
                plan, image_paths_unused, per_candidate = _pad_images_to_plan_length(
                    plan, per_candidate[0] if per_candidate else [], per_candidate
                )
            else:
                logger.warning(
                    f"Per-candidate short: smallest candidate has {min_count} images; "
                    f"subsampling plan to {min_count} gap(s)"
                )
                plan = _evenly_subsample_plan(plan, min_count)
        per_candidate = [c[: len(plan)] for c in per_candidate]
        image_paths = per_candidate[0]
        actual = min_count
        logger.info(f"MiniMax generated: {actual} images per candidate (requested {args.max_images})")
        logger.info(f"MiniMax generated: {actual} images (requested {args.max_images})")

        if actual == 0:
            logger.warning(
                "0 MiniMax images generated; writing OTIO with empty V-Ref track."
            )
            new_tl = build_timeline(
                plan=[],
                image_paths=[],
                track_name=args.track_name,
                rate=rate,
                subject=subject,
                source_total_frames=source_total_frames,
                match_duration=args.match_duration,
            )
        elif actual < args.max_images:
            logger.warning(
                f"Re-planning with N={actual} (had {args.max_images} requested)"
            )
            if args.fill_gaps:
                # Fill-gaps mode: keep all gaps in the plan, pad images to match.
                plan, image_paths, _ = _pad_images_to_plan_length(
                    plan, image_paths, None
                )
            else:
                plan = plan_slots(actual, srt_segments, existing_slots, rate)
            new_tl = build_timeline(
                plan=plan,
                image_paths=image_paths,
                track_name=args.track_name,
                rate=rate,
                subject=subject,
                source_total_frames=source_total_frames,
                match_duration=args.match_duration,
                candidates=args.candidates,
                candidate_image_paths=per_candidate,
                candidate_track_names=track_names,
            )
        else:
            new_tl = build_timeline(
                plan=plan,
                image_paths=image_paths[: args.max_images],
                track_name=args.track_name,
                rate=rate,
                subject=subject,
                source_total_frames=source_total_frames,
                match_duration=args.match_duration,
                candidates=args.candidates,
                candidate_image_paths=per_candidate,
                candidate_track_names=track_names,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        otio.adapters.write_to_file(new_tl, str(output_path))

        if args.fill_gaps:
            verify_fill_timing(plan, new_tl, track_names, rate)

        summary = build_summary(
            args=args,
            subject=subject,
            srt_path=srt_path,
            otio_path=otio_path,
            output_path=output_path,
            rate=rate,
            source_total_frames=source_total_frames,
            srt_start=srt_start,
            srt_end=srt_end,
            plan=plan,
            per_candidate=per_candidate,
            track_names=track_names,
            actual=actual,
            source_image_dirs=None,
            provider="minimax",
            name_detection=build_name_detection_block(args=args, plan=plan),
        )
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Wrote {output_path}")
        logger.info(f"Wrote {summary_path}")
        logger.info(f"AI stills in {tmp_dir} (use --keep-tmp to retain after future runs)")

        print_plan_table(plan, rate, args.candidates, track_names, per_candidate)
        return 0

    # Dry-run: print the original plan table and exit before downloading.
    if args.dry_run:
        print(
            f"\n{'idx':>3} | {'center (TC)':>12} | {'center (s)':>10} | "
            f"{'dur_f':>6} | {'start_f':>7} | {'dur_source':<12} | slot_name"
        )
        print("-" * 100)
        for entry in plan:
            print(
                f"{entry['index']:>3} | "
                f"{_seconds_to_tc(entry['center_sec'], rate):>12} | "
                f"{entry['center_sec']:>10.3f} | "
                f"{entry['dur_frames']:>6} | "
                f"{entry['start_frames']:>7} | "
                f"{entry['dur_source']:<12} | "
                f"{entry['slot_name']}"
            )
        logger.info("Dry run; not downloading or writing.")
        return 0

    # Download.
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    per_candidate: List[List[Path]] = []
    track_names = candidate_track_names(args.track_name, args.candidate_track_prefix, args.candidates)
    first_detected_name = next(
        (e.get("detected_name") for e in plan if e.get("detected_name")), None
    )
    for cand_idx in range(args.candidates):
        name_slug = _slugify(first_detected_name) if first_detected_name else ""
        cand_subdir = (
            tmp_dir / f"cand_{cand_idx:02d}" / name_slug
            if name_slug
            else tmp_dir / f"cand_{cand_idx:02d}"
        )
        if cand_idx == 0:
            if first_detected_name:
                query_override = first_detected_name
            else:
                query_override = None
        else:
            if first_detected_name:
                query_override = f"{first_detected_name} cand{cand_idx:02d}"
            else:
                snippet = " ".join((plan[0].get("srt_text", "") if plan else "").split())[:80]
                if snippet:
                    query_override = f"{subject} {snippet}"
                else:
                    query_override = f"{subject} cand{cand_idx:02d}"
        rc = run_download(
            repo_root=repo_root,
            subject=subject,
            tmp_dir=cand_subdir,
            max_images=args.max_images,
            min_size_kb=args.min_size_kb,
            context=args.context,
            sources=args.sources,
            per_source_limit=args.per_source_limit,
            exclude_domains=args.exclude_domains,
            keep_tmp=args.keep_tmp,
            query_override=query_override,
            timeout=args.downloader_timeout,
        )
        if rc == 1:
            logger.error(f"Candidate {cand_idx}: downloader reported bad args / no candidates (rc=1)")
            return 3
        if rc == 2:
            logger.warning(f"Candidate {cand_idx}: downloader found no images passing size filter (rc=2)")
        elif rc == 4:
            logger.warning(f"Candidate {cand_idx}: downloader hit --downloader-timeout; "
                           f"continuing with any images saved before the stall.")
        elif rc != 0:
            logger.error(f"Candidate {cand_idx}: downloader failed (rc={rc})")
            return 3
        cand_images = collect_downloaded(
            cand_subdir,
            subject,
            args.min_size_kb,
            exclude_domains=args.exclude_domains,
            exclude_indices=exclude_indices,
        )
        logger.info(
            f"Candidate {cand_idx}: {len(cand_images)} images "
            f"(requested {args.max_images}, dir={cand_subdir})"
        )
        per_candidate.append(cand_images)
    min_count = min((len(c) for c in per_candidate), default=0)
    if min_count < len(plan):
        if args.fill_gaps:
            # Fill-gaps mode: never leave a gap in V-Ref — replicate successful images.
            plan, downloaded_unused, per_candidate = _pad_images_to_plan_length(
                plan, per_candidate[0] if per_candidate else [], per_candidate
            )
        else:
            logger.warning(
                f"Per-candidate short: smallest candidate has {min_count} images; "
                f"subsampling plan to {min_count} gap(s)"
            )
            plan = _evenly_subsample_plan(plan, min_count)
    per_candidate = [c[: len(plan)] for c in per_candidate]
    downloaded = per_candidate[0] if per_candidate else []
    actual = min_count
    logger.info(
        f"Downloaded images per candidate: "
        f"{[len(c) for c in per_candidate]} (requested {args.max_images} per candidate)"
    )

    if actual == 0:
        logger.warning("0 images downloaded; writing OTIO with empty V-Ref track.")
        new_tl = build_timeline(
            plan=[],
            image_paths=[],
            track_name=args.track_name,
            rate=rate,
            subject=subject,
            source_total_frames=source_total_frames,
            match_duration=args.match_duration,
            candidates=args.candidates,
            candidate_image_paths=per_candidate,
            candidate_track_names=track_names,
        )
    else:
        new_tl = build_timeline(
            plan=plan,
            image_paths=downloaded,
            track_name=args.track_name,
            rate=rate,
            subject=subject,
            source_total_frames=source_total_frames,
            match_duration=args.match_duration,
            candidates=args.candidates,
            candidate_image_paths=per_candidate,
            candidate_track_names=track_names,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    otio.adapters.write_to_file(new_tl, str(output_path))

    if args.fill_gaps:
        verify_fill_timing(plan, new_tl, track_names, rate)

    summary = build_summary(
        args=args,
        subject=subject,
        srt_path=srt_path,
        otio_path=otio_path,
        output_path=output_path,
        rate=rate,
        source_total_frames=source_total_frames,
        srt_start=srt_start,
        srt_end=srt_end,
        plan=plan,
        per_candidate=per_candidate,
        track_names=track_names,
        actual=actual,
        source_image_dirs=None,
        provider=None,
        name_detection=build_name_detection_block(args=args, plan=plan),
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Wrote {output_path}")
    logger.info(f"Wrote {summary_path}")
    if not args.keep_tmp:
        # Leave files in place; user controls cleanup. We only log it.
        logger.info(f"Images in {tmp_dir} (use --keep-tmp to retain after future runs)")

    print_plan_table(plan, rate, args.candidates, track_names, per_candidate)
    return 0


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    srt_path = Path(args.srt).resolve()
    otio_path = Path(args.otio).resolve()
    if not srt_path.exists():
        logger.error(f"SRT not found: {srt_path}")
        return 2
    if not otio_path.exists():
        logger.error(f"OTIO not found: {otio_path}")
        return 2

    if args.minimax_fill:
        if args.fill_gaps:
            logger.error("--minimax-fill implies --fill-gaps; do not pass both.")
            return 2
        if args.provider != "pyimagedl":
            logger.error("--minimax-fill implies --provider minimax; do not pass both.")
            return 2
        if args.all:
            logger.error("--minimax-fill is mutually exclusive with --all.")
            return 2
        if args.existing_images_dir:
            logger.error("--minimax-fill is mutually exclusive with --existing-images-dir.")
            return 2
        args.fill_gaps = True
        args.provider = "minimax"
        logger.info(
            "--minimax-fill: expanding to --fill-gaps --provider minimax. "
            f"Output will be written next to {otio_path} as "
            f"{otio_path.stem}_fill_gaps.otio."
        )

    if args.all:
        if args.fill_gaps:
            logger.error("--all is mutually exclusive with --fill-gaps.")
            return 2
        if args.provider != "pyimagedl":
            logger.error("--all is mutually exclusive with --provider.")
            return 2
        if args.all_minimax_images < 1:
            logger.error("--all-minimax-images must be >= 1")
            return 2
        if args.all_skip_minimax and args.all_minimax_images:
            logger.info("--all-skip-minimax: MiniMax pass will be skipped.")
        # The pyimagedl passes want 50 images. argparse sets 25 as the script-wide
        # default; if the user didn't pass --max-images explicitly, bump to 50
        # for --all mode. (argparse doesn't expose "was this flag set by default",
        # so we infer by comparing against the default value.)
        if args.max_images == 25:
            args.max_images = 50
            logger.info("--all: defaulting --max-images to 50 for the pyimagedl passes.")
    else:
        if args.all_skip_minimax or args.all_minimax_images != 20:
            logger.error("--all-skip-minimax / --all-minimax-images require --all.")
            return 2

    rate = float(args.frame_rate)
    try:
        exclude_indices = parse_index_ranges(args.exclude_indices)
    except ValueError as exc:
        logger.error(f"Bad --exclude-indices: {exc}")
        return 2
    if exclude_indices:
        logger.info(
            f"--exclude-indices active: {len(exclude_indices)} NN index(es) will be skipped "
            f"(ranges: {args.exclude_indices})"
        )

    srt_segments = parse_srt(srt_path)
    if not srt_segments:
        logger.error(f"No SRT segments parsed from {srt_path}")
        return 2
    srt_start = srt_segments[0][0]
    srt_end = max(e for _, e, _ in srt_segments)
    if srt_end - srt_start <= 0:
        logger.error("SRT has zero/negative total duration")
        return 2
    logger.info(f"SRT span: {srt_start:.2f}s - {srt_end:.2f}s ({srt_end - srt_start:.2f}s, {len(srt_segments)} segs)")

    if args.subject:
        subject = args.subject
        logger.info(f"Subject: {subject!r} (from --subject)")
    elif args.existing_images_dir:
        logger.error(
            "--existing-images-dir requires --subject (LLM is skipped in this mode)"
        )
        return 2
    elif args.no_llm:
        subject = derive_subject(srt_path)
        logger.info(f"Subject: {subject!r} (from --no-llm + filename heuristic)")
    else:
        llm_subject = infer_subject_with_llm(
            srt_segments=srt_segments,
            provider=args.llm_provider,
            model=args.llm_model,
            host=args.llm_host,
            samples=args.llm_samples,
        )
        if llm_subject:
            subject = llm_subject
            logger.info(f"Subject: {subject!r} (from LLM)")
        else:
            subject = derive_subject(srt_path)
            logger.warning(
                f"Subject: {subject!r} (LLM unavailable, fell back to filename)"
            )

    src_tl = otio.adapters.read_from_file(str(otio_path))
    existing_slots, source_total_frames = collect_existing_slots(
        src_tl, args.track_index, rate
    )
    logger.info(
        f"Source OTIO: {len(existing_slots)} clips on track {args.track_index}, "
        f"total {source_total_frames} frames ({source_total_frames / rate:.2f}s)"
    )

    prelude: dict = {
        "srt_path": srt_path,
        "otio_path": otio_path,
        "srt_segments": srt_segments,
        "srt_start": srt_start,
        "srt_end": srt_end,
        "source_total_frames": source_total_frames,
        "existing_slots": existing_slots,
        "rate": rate,
        "subject": subject,
    }

    if args.all:
        return _run_all(
            args=args,
            prelude=prelude,
            src_tl=src_tl,
            exclude_indices=exclude_indices,
        )

    return run_pass(
        args=args,
        prelude=prelude,
        src_tl=src_tl,
        exclude_indices=exclude_indices,
    )


if __name__ == "__main__":
    sys.exit(main())
