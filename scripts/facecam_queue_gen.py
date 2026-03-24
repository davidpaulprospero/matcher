#!/usr/bin/env python3
"""
Batch Facecam Queue Generator

Reads completed projects from the Stu pipeline queue, classifies voiceover
segments as "talking points" via Ollama (free, local), merges adjacent segments
for cost efficiency, and generates facecam video clips within a hard budget.

Usage:
    # Dry run (free) - classify and show plan
    python scripts/facecam_queue_gen.py --dry-run

    # Generate for all completed projects
    python scripts/facecam_queue_gen.py

    # Generate for specific projects
    python scripts/facecam_queue_gen.py --card-ids KxQ0SoGh ujKR12Dg

    # Override budget
    python scripts/facecam_queue_gen.py --budget 5.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add project root and scripts to path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Degold"))

from auto_wan_facecam import (
    detect_channel_from_path,
    find_voiceover,
    get_next_avatar,
    _sanitize_filename,
)
from wan_facecam import WanFacecamService, DEFAULT_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_STU_QUEUE_STATE = PROJECT_ROOT / "Stu" / "pipeline_queue_state.json"
DEFAULT_BUDGET_TRACKER = PROJECT_ROOT / "Stu" / "facecam_budget.json"

DEFAULT_BUDGET_USD = 8.0
FACECAM_PERCENT = 0.05  # 5% of project runtime
CHUNK_DURATION_S = 5
COST_PER_10S_CHUNK = {
    "480P": 0.14,
    "720P": 0.28,
    "1080P": 0.56,
}
# Cost scales linearly with chunk duration
COST_PER_CHUNK = {
    res: cost * (CHUNK_DURATION_S / 10)
    for res, cost in COST_PER_10S_CHUNK.items()
}
MIN_AUDIO_DURATION_S = 3
MAX_MERGE_GAP_S = 2.0  # max gap to merge adjacent talking points
BATCH_SIZE = 20  # segments per Ollama classification call

# Windows: prevent subprocess console windows
_SUBPROCESS_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SRTSegment:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ClassifiedSegment:
    segment: SRTSegment
    is_talking_point: bool
    confidence: float
    reasoning: str = ""


@dataclass
class FacecamGroup:
    """A merged group of consecutive talking-point segments."""

    segment_indices: List[int]
    segments: List[SRTSegment]
    start_s: float
    end_s: float
    text: str
    avg_confidence: float
    whisper_aligned: bool = False
    whisper_text: str = ""
    original_start_s: float = 0.0
    original_end_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def chunks_needed(self) -> int:
        return max(1, math.ceil(self.duration_s / CHUNK_DURATION_S))

    def cost_usd(self, resolution: str = "480P") -> float:
        return self.chunks_needed * COST_PER_CHUNK.get(resolution, 0.14)


@dataclass
class BudgetTracker:
    total_budget_usd: float = DEFAULT_BUDGET_USD
    total_spent_usd: float = 0.0
    resolution: str = "480P"
    cost_per_chunk_usd: float = COST_PER_CHUNK["480P"]
    chunk_duration_s: int = CHUNK_DURATION_S
    projects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.total_budget_usd - self.total_spent_usd)


# ---------------------------------------------------------------------------
# SRT Parsing
# ---------------------------------------------------------------------------
def _parse_timestamp(ts: str) -> float:
    """Parse SRT timestamp '00:01:23,456' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def parse_srt(srt_path: Path) -> List[SRTSegment]:
    """Parse SRT file into list of SRTSegment."""
    text = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    segments: List[SRTSegment] = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Line 0: index, Line 1: timestamps, Line 2+: text
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        ts_match = re.match(
            r"(\d[\d:,\.]+)\s*-->\s*(\d[\d:,\.]+)", lines[1].strip()
        )
        if not ts_match:
            continue

        start = _parse_timestamp(ts_match.group(1))
        end = _parse_timestamp(ts_match.group(2))
        seg_text = " ".join(line.strip() for line in lines[2:] if line.strip())

        segments.append(SRTSegment(index=idx, start=start, end=end, text=seg_text))

    return segments


def find_project_srt(project_dir: Path) -> Optional[Path]:
    """Find voiceover SRT in a project directory."""
    # Check voiceover/ subfolder first
    vo_dir = project_dir / "voiceover"
    if vo_dir.is_dir():
        srts = sorted(vo_dir.glob("*.srt"))
        if srts:
            return srts[0]

    # Check project root
    srts = sorted(project_dir.glob("*.srt"))
    if srts:
        return srts[0]

    return None


# ---------------------------------------------------------------------------
# Duration detection
# ---------------------------------------------------------------------------
def get_voiceover_duration(project_dir: Path) -> float:
    """Get total voiceover duration in seconds via ffprobe, fallback to SRT."""
    # Try ffprobe on voiceover files
    vo_dir = project_dir / "voiceover"
    candidates = []
    if vo_dir.is_dir():
        for name in ["voiceover_trimmed.mp3", "voiceover.mp3"]:
            p = vo_dir / name
            if p.exists():
                candidates.append(p)
    # Also check root
    for name in ["voiceover_trimmed.mp3", "voiceover.mp3"]:
        p = project_dir / name
        if p.exists():
            candidates.append(p)

    for audio_path in candidates:
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                **_SUBPROCESS_FLAGS,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                return float(probe.stdout.strip())
        except Exception:
            continue

    # Fallback: last SRT segment end time
    srt_path = find_project_srt(project_dir)
    if srt_path:
        segments = parse_srt(srt_path)
        if segments:
            return segments[-1].end

    return 0.0


# ---------------------------------------------------------------------------
# Queue state loading
# ---------------------------------------------------------------------------
def load_completed_projects(
    queue_state_path: Path,
) -> List[Dict[str, Any]]:
    """Load completed projects from Stu queue state JSON."""
    data = json.loads(queue_state_path.read_text(encoding="utf-8"))
    completed_ids = data.get("queue", {}).get("completed_card_ids", [])

    if not completed_ids:
        return []

    pipelines = data.get("pipelines", {})
    projects = []

    for card_id in completed_ids:
        # Pipeline keys are lowercase
        key = card_id.lower()
        entry = pipelines.get(key)
        if not entry:
            logger.warning(f"Card {card_id} not found in pipelines")
            continue

        project_info = entry.get("project", {})
        local_dirs = project_info.get("local_project_dirs", [])
        if not local_dirs:
            logger.warning(f"Card {card_id} has no local project dirs")
            continue

        project_dir = Path(local_dirs[0])
        if not project_dir.is_dir():
            logger.warning(f"Project dir does not exist: {project_dir}")
            continue

        projects.append({
            "card_id": card_id,
            "title": entry.get("title", "Untitled"),
            "channel": project_info.get("channel", "STU"),
            "project_dir": project_dir,
        })

    return sorted(projects, key=lambda p: p["title"])


# ---------------------------------------------------------------------------
# Ollama talking-point classification
# ---------------------------------------------------------------------------
CLASSIFICATION_PROMPT_TEMPLATE = """\
You are a video editor analyzing voiceover segments to identify talking points — \
segments where a presenter speaking to camera would be most engaging and natural.

A TALKING POINT is:
- Speaker directly addresses the audience ("you", "we", "let me explain")
- Speaker states opinions, conclusions, or calls to action
- Speaker introduces or summarizes a topic
- Rhetorical questions or strong emotional appeal

NOT a talking point:
- Pure factual narration/description meant to accompany footage
- Short transitional phrases under 3 words
- [Music] or [Sound effect] annotations
- Lists of statistics, dates, or numbers

Video title: "{title}"

Segments:
{segments_text}

Respond with ONLY a JSON array. Each element must have:
- "index": the segment index number
- "is_talking_point": true or false
- "confidence": 0.0 to 1.0
- "reason": one brief sentence

Example: [{{"index": 1, "is_talking_point": true, "confidence": 0.85, "reason": "Direct audience address"}}]
"""


def classify_talking_points(
    segments: List[SRTSegment],
    project_title: str,
) -> List[ClassifiedSegment]:
    """Classify SRT segments as talking-point or not via Ollama.

    Batches segments into groups of BATCH_SIZE for efficiency.
    Uses built-in LLM cache so re-runs are instant and free.
    """
    from src.llm_client import create_client, LLMRequest, ResponseFormat
    from src.llm_client.providers.ollama import check_ollama_available

    # Verify Ollama is running
    available, error, models = check_ollama_available()
    if not available:
        raise RuntimeError(
            f"Ollama not available: {error}\n"
            "Start Ollama and ensure llama3.2 is pulled: ollama pull llama3.2"
        )

    client = create_client("ollama", model="llama3.2")
    results: List[ClassifiedSegment] = []

    # Process in batches
    for batch_start in range(0, len(segments), BATCH_SIZE):
        batch = segments[batch_start : batch_start + BATCH_SIZE]

        segments_text = "\n".join(
            f"[{s.index}] ({s.start:.1f}s - {s.end:.1f}s): {s.text}"
            for s in batch
        )

        prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
            title=project_title,
            segments_text=segments_text,
        )

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            cache_key_prefix="facecam_tp",
            temperature=0.3,
            max_tokens=4000,
            timeout=120,
        )

        try:
            response = client.generate(request)
            parsed = response.parsed_data
        except Exception as e:
            logger.warning(f"Ollama batch failed: {e}. Marking batch as non-talking-point.")
            parsed = None

        # Build index lookup for this batch
        batch_by_idx = {s.index: s for s in batch}

        if parsed and isinstance(parsed, list):
            classified_indices = set()
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                seg = batch_by_idx.get(idx)
                if not seg:
                    continue
                classified_indices.add(idx)
                results.append(
                    ClassifiedSegment(
                        segment=seg,
                        is_talking_point=bool(item.get("is_talking_point", False)),
                        confidence=float(item.get("confidence", 0.0)),
                        reasoning=str(item.get("reason", "")),
                    )
                )
            # Any segments not returned by LLM default to non-talking-point
            for seg in batch:
                if seg.index not in classified_indices:
                    results.append(
                        ClassifiedSegment(
                            segment=seg,
                            is_talking_point=False,
                            confidence=0.0,
                            reasoning="not classified by LLM",
                        )
                    )
        else:
            # Entire batch failed — mark all as non-talking-point
            for seg in batch:
                results.append(
                    ClassifiedSegment(
                        segment=seg,
                        is_talking_point=False,
                        confidence=0.0,
                        reasoning="classification failed",
                    )
                )

    return results


# ---------------------------------------------------------------------------
# Segment merging
# ---------------------------------------------------------------------------
def merge_adjacent_talking_points(
    classified: List[ClassifiedSegment],
    max_gap_s: float = MAX_MERGE_GAP_S,
    min_duration_s: float = MIN_AUDIO_DURATION_S,
) -> List[FacecamGroup]:
    """Merge consecutive/near-consecutive talking points into groups."""
    talking = [c for c in classified if c.is_talking_point]
    if not talking:
        return []

    talking.sort(key=lambda c: c.segment.start)

    def _build_group(members: List[ClassifiedSegment]) -> FacecamGroup:
        indices = [m.segment.index for m in members]
        segs = [m.segment for m in members]
        text = " ".join(m.segment.text for m in members)
        avg_conf = sum(m.confidence for m in members) / len(members)
        return FacecamGroup(
            segment_indices=indices,
            segments=segs,
            start_s=segs[0].start,
            end_s=segs[-1].end,
            text=text,
            avg_confidence=avg_conf,
        )

    groups: List[FacecamGroup] = []
    current: List[ClassifiedSegment] = [talking[0]]

    for i in range(1, len(talking)):
        prev_end = current[-1].segment.end
        curr_start = talking[i].segment.start
        gap = curr_start - prev_end

        if gap <= max_gap_s:
            current.append(talking[i])
        else:
            groups.append(_build_group(current))
            current = [talking[i]]

    groups.append(_build_group(current))

    # Filter: minimum duration
    groups = [g for g in groups if g.duration_s >= min_duration_s]

    return groups


# ---------------------------------------------------------------------------
# Budget-constrained selection
# ---------------------------------------------------------------------------
def select_groups_for_budget(
    groups: List[FacecamGroup],
    max_duration_s: float,
    max_cost_usd: float,
    resolution: str = "480P",
) -> List[FacecamGroup]:
    """Greedy selection of best groups within both duration and cost caps."""
    # Sort by confidence descending (best talking points first)
    ranked = sorted(groups, key=lambda g: g.avg_confidence, reverse=True)

    selected: List[FacecamGroup] = []
    total_duration = 0.0
    total_cost = 0.0

    for group in ranked:
        cost = group.cost_usd(resolution)
        dur = group.duration_s

        if total_duration + dur > max_duration_s:
            continue
        if total_cost + cost > max_cost_usd:
            continue

        selected.append(group)
        total_duration += dur
        total_cost += cost

    # Sort selected back into chronological order
    selected.sort(key=lambda g: g.start_s)
    return selected


# ---------------------------------------------------------------------------
# Whisper alignment — precise speech boundaries
# ---------------------------------------------------------------------------
WHISPER_PAD_S = 1.0  # padding around SRT timestamps for Whisper window


def whisper_align_group(
    group: FacecamGroup,
    voiceover_path: Path,
    voiceover_duration: float,
) -> FacecamGroup:
    """Re-transcribe a group's audio range with Whisper word timestamps.

    Extracts audio with padding, runs faster-whisper to get precise speech
    boundaries, and updates the group's start/end times accordingly.
    Returns the same group object with corrected timestamps.
    """
    from pydub import AudioSegment
    from src.transcription.whisper_client import WhisperClient

    # Save originals
    group.original_start_s = group.start_s
    group.original_end_s = group.end_s

    # Extract audio with padding for Whisper context
    pad_start = max(0.0, group.start_s - WHISPER_PAD_S)
    pad_end = min(voiceover_duration, group.end_s + WHISPER_PAD_S)

    with tempfile.TemporaryDirectory(prefix="whisper_align_") as tmp:
        # Extract padded audio segment
        audio = AudioSegment.from_file(str(voiceover_path))
        segment_audio = audio[int(pad_start * 1000) : int(pad_end * 1000)]
        wav_path = Path(tmp) / "align_segment.wav"
        segment_audio.export(str(wav_path), format="wav", parameters=["-ar", "16000", "-ac", "1"])

        # Run Whisper with word-level timestamps
        try:
            client = WhisperClient(
                model_name="base",
                compute_type="auto",
                gpu_transcription_timeout=120,
            )
            result = client.transcribe(
                str(wav_path),
                language="en",
                word_timestamps=True,
                vad_filter=False,  # already trimmed, don't filter
            )
            # transcribe() returns (segments_list, info) tuple
            segments = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            logger.warning(f"Whisper alignment failed: {e}. Keeping SRT timestamps.")
            return group

    if not segments:
        logger.warning("Whisper returned no segments. Keeping SRT timestamps.")
        return group

    # Collect all words with absolute timestamps (offset back to original timeline)
    all_words = []
    whisper_text_parts = []
    for seg in segments:
        whisper_text_parts.append(seg.get("text", "").strip())
        for word in seg.get("words", []):
            all_words.append({
                "word": word["word"],
                "start": word["start"] + pad_start,  # offset to original timeline
                "end": word["end"] + pad_start,
                "confidence": word.get("confidence", 0.0),
            })

    if not all_words:
        # No word-level data — use segment-level timestamps
        first_seg = segments[0]
        last_seg = segments[-1]
        group.start_s = first_seg["start"] + pad_start
        group.end_s = last_seg["end"] + pad_start
    else:
        # Use first/last word boundaries as precise speech edges
        group.start_s = all_words[0]["start"]
        group.end_s = all_words[-1]["end"]

    group.whisper_aligned = True
    group.whisper_text = " ".join(whisper_text_parts).strip()

    # Ensure minimum duration after alignment
    if group.duration_s < MIN_AUDIO_DURATION_S:
        # Expand symmetrically to meet minimum
        deficit = MIN_AUDIO_DURATION_S - group.duration_s
        group.start_s = max(0.0, group.start_s - deficit / 2)
        group.end_s = min(voiceover_duration, group.end_s + deficit / 2)

    drift = abs(group.start_s - group.original_start_s) + abs(group.end_s - group.original_end_s)
    logger.info(
        f"Whisper aligned seg {group.segment_indices[0]}-{group.segment_indices[-1]}: "
        f"{group.original_start_s:.2f}-{group.original_end_s:.2f} -> "
        f"{group.start_s:.2f}-{group.end_s:.2f} (drift={drift:.2f}s)"
    )

    return group


def whisper_align_groups(
    groups: List[FacecamGroup],
    voiceover_path: Path,
    voiceover_duration: float,
) -> List[FacecamGroup]:
    """Run Whisper alignment on all selected groups."""
    if not groups:
        return groups

    print(f"\n  Whisper alignment ({len(groups)} groups)...")
    aligned = []
    for group in groups:
        first, last = group.segment_indices[0], group.segment_indices[-1]
        label = f"seg {first}-{last}" if first != last else f"seg {first}"
        print(f"    Aligning {label} ({_fmt_duration(group.duration_s)})...", end=" ", flush=True)
        aligned_group = whisper_align_group(group, voiceover_path, voiceover_duration)
        if aligned_group.whisper_aligned:
            drift = abs(aligned_group.start_s - aligned_group.original_start_s) + \
                    abs(aligned_group.end_s - aligned_group.original_end_s)
            print(f"OK (drift={drift:.2f}s, {_fmt_duration(aligned_group.duration_s)})")
        else:
            print("skipped (Whisper failed, using SRT)")
        aligned.append(aligned_group)

    return aligned


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
def extract_segment_audio(
    voiceover_path: Path,
    start_s: float,
    end_s: float,
    output_path: Path,
) -> Path:
    """Extract audio segment using pydub, pad to minimum if needed."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(voiceover_path))
    segment = audio[int(start_s * 1000) : int(end_s * 1000)]

    # Pad short segments to API minimum
    min_ms = MIN_AUDIO_DURATION_S * 1000
    if len(segment) < min_ms:
        silence = AudioSegment.silent(duration=min_ms - len(segment))
        segment = segment + silence

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment.export(str(output_path), format="mp3")
    return output_path


# ---------------------------------------------------------------------------
# Budget tracker persistence
# ---------------------------------------------------------------------------
def load_budget_tracker(path: Path = DEFAULT_BUDGET_TRACKER) -> BudgetTracker:
    """Load or create budget tracker."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return BudgetTracker(
            total_budget_usd=data.get("total_budget_usd", DEFAULT_BUDGET_USD),
            total_spent_usd=data.get("total_spent_usd", 0.0),
            resolution=data.get("resolution", "480P"),
            cost_per_chunk_usd=data.get("cost_per_chunk_usd", COST_PER_CHUNK["480P"]),
            chunk_duration_s=data.get("chunk_duration_s", CHUNK_DURATION_S),
            projects=data.get("projects", {}),
        )
    return BudgetTracker()


def save_budget_tracker(tracker: BudgetTracker, path: Path = DEFAULT_BUDGET_TRACKER) -> None:
    """Save budget tracker atomically."""
    data = {
        "total_budget_usd": tracker.total_budget_usd,
        "total_spent_usd": round(tracker.total_spent_usd, 4),
        "resolution": tracker.resolution,
        "cost_per_chunk_usd": tracker.cost_per_chunk_usd,
        "chunk_duration_s": tracker.chunk_duration_s,
        "projects": tracker.projects,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Generation log persistence
# ---------------------------------------------------------------------------
def load_generation_log(project_dir: Path) -> Optional[Dict[str, Any]]:
    """Load generation log from project/facecam/generation_log.json."""
    log_path = project_dir / "facecam" / "generation_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8"))
    return None


def save_generation_log(log_data: Dict[str, Any], project_dir: Path) -> None:
    """Save generation log to project/facecam/generation_log.json."""
    facecam_dir = project_dir / "facecam"
    facecam_dir.mkdir(parents=True, exist_ok=True)
    log_path = facecam_dir / "generation_log.json"
    log_path.write_text(
        json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_already_generated_indices(project_dir: Path) -> set:
    """Return set of segment indices already generated (from log)."""
    log = load_generation_log(project_dir)
    if not log:
        return set()
    indices = set()
    for entry in log.get("segments", []):
        for idx in entry.get("segment_indices", []):
            indices.add(idx)
    return indices


# ---------------------------------------------------------------------------
# Facecam generation for a single group
# ---------------------------------------------------------------------------
def generate_facecam_for_group(
    group: FacecamGroup,
    voiceover_path: Path,
    avatar_path: str,
    output_dir: Path,
    svc: WanFacecamService,
    resolution: str = "480P",
) -> Optional[Dict[str, Any]]:
    """Generate facecam for a single merged group.

    Returns log entry dict on success, None on failure.
    """
    first = group.segment_indices[0]
    last = group.segment_indices[-1]
    label = f"seg{first:03d}-{last:03d}" if first != last else f"seg{first:03d}"

    # Extract audio for this group
    with tempfile.TemporaryDirectory(prefix="facecam_seg_") as tmp:
        audio_path = Path(tmp) / f"audio_{label}.mp3"
        extract_segment_audio(voiceover_path, group.start_s, group.end_s, audio_path)

        # Generate facecam using WanFacecamService
        seg_output_dir = output_dir / f"_{label}_chunks"
        seg_output_dir.mkdir(parents=True, exist_ok=True)

        max_dur = max(MIN_AUDIO_DURATION_S, math.ceil(group.duration_s))
        result = svc.generate(
            audio_path=str(audio_path),
            image_path=avatar_path,
            output_dir=str(seg_output_dir),
            prompt=DEFAULT_PROMPT,
            max_duration=max_dur,
        )

    if result.chunks_generated == 0:
        logger.error(f"Generation failed for {label}: {result.errors}")
        return None

    # Move final video to main facecam dir with descriptive name
    final_name = f"facecam_{label}.mp4"
    final_path = output_dir / final_name

    if result.concatenated_path:
        src = Path(result.concatenated_path)
        if src.exists() and str(src) != str(final_path):
            src.rename(final_path)
    elif result.video_paths:
        src = Path(result.video_paths[0])
        if src.exists():
            src.rename(final_path)

    # Clean up chunk directory
    try:
        import shutil
        shutil.rmtree(seg_output_dir, ignore_errors=True)
    except Exception:
        pass

    cost = group.cost_usd(resolution)
    logger.info(
        f"Generated {final_name}: {result.total_duration:.1f}s, "
        f"{result.chunks_generated} chunks, ${cost:.2f}"
    )

    return {
        "segment_indices": group.segment_indices,
        "text": group.text[:200],
        "start_s": round(group.start_s, 2),
        "end_s": round(group.end_s, 2),
        "duration_s": round(group.duration_s, 2),
        "chunks": result.chunks_generated,
        "cost_usd": round(cost, 4),
        "confidence": round(group.avg_confidence, 3),
        "facecam_file": final_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def display_project_plan(
    title: str,
    card_id: str,
    total_duration: float,
    segments: List[SRTSegment],
    classified: List[ClassifiedSegment],
    groups: List[FacecamGroup],
    selected: List[FacecamGroup],
    resolution: str,
    already_generated: set,
):
    """Display classification and selection results for one project."""
    tp_count = sum(1 for c in classified if c.is_talking_point)
    target_dur = total_duration * FACECAM_PERCENT
    sel_dur = sum(g.duration_s for g in selected)
    sel_cost = sum(g.cost_usd(resolution) for g in selected)
    skipped = sum(
        1 for g in groups
        if all(idx in already_generated for idx in g.segment_indices)
    )

    print(f"\n  [{card_id}] {title[:70]}")
    print(f"    VO duration: {_fmt_duration(total_duration)} | "
          f"Segments: {len(segments)} | "
          f"Talking points: {tp_count}/{len(segments)}")
    print(f"    Merged groups: {len(groups)} | "
          f"Already generated: {skipped} | "
          f"Target: {_fmt_duration(target_dur)} (5%)")
    print(f"    Selected: {len(selected)} groups | "
          f"Duration: {_fmt_duration(sel_dur)} | "
          f"Est. cost: ${sel_cost:.2f}")

    if selected:
        for g in selected:
            status = ""
            if all(idx in already_generated for idx in g.segment_indices):
                status = " [SKIP: already generated]"
            first, last = g.segment_indices[0], g.segment_indices[-1]
            label = f"seg {first}-{last}" if first != last else f"seg {first}"
            timing = f"{_fmt_duration(g.duration_s)}"
            if g.whisper_aligned:
                drift = abs(g.start_s - g.original_start_s) + abs(g.end_s - g.original_end_s)
                timing += f", aligned, drift={drift:.1f}s"
            print(
                f"      - {label} ({timing}, "
                f"conf={g.avg_confidence:.2f}, "
                f"${g.cost_usd(resolution):.2f}){status}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global CHUNK_DURATION_S, COST_PER_CHUNK
    parser = argparse.ArgumentParser(
        description="Batch facecam generation from Stu queue completed projects"
    )
    parser.add_argument(
        "--queue-state",
        type=Path,
        default=DEFAULT_STU_QUEUE_STATE,
        help=f"Path to pipeline_queue_state.json (default: {DEFAULT_STU_QUEUE_STATE})",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help=f"Total budget in USD (default: {DEFAULT_BUDGET_USD})",
    )
    parser.add_argument(
        "--facecam-percent",
        type=float,
        default=FACECAM_PERCENT,
        help=f"Target facecam as fraction of project duration (default: {FACECAM_PERCENT})",
    )
    parser.add_argument(
        "--resolution",
        default="480P",
        choices=["480P", "720P", "1080P"],
        help="Video resolution (default: 480P)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=MAX_MERGE_GAP_S,
        help=f"Max gap in seconds for merging adjacent talking points (default: {MAX_MERGE_GAP_S})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and plan but don't generate (no cost)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if segments already exist in log",
    )
    parser.add_argument(
        "--card-ids",
        nargs="*",
        help="Specific card IDs to process (default: all completed)",
    )
    parser.add_argument(
        "--region",
        default="international",
        choices=["international", "beijing"],
        help="API region (default: international)",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip Whisper alignment pass (use raw SRT timestamps)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=CHUNK_DURATION_S,
        choices=[5, 10],
        help=f"Seconds per video chunk, 5 or 10 (default: {CHUNK_DURATION_S})",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Allow runtime override of chunk duration (affects cost calculations)
    CHUNK_DURATION_S = args.chunk_duration
    COST_PER_CHUNK = {
        res: cost * (CHUNK_DURATION_S / 10)
        for res, cost in COST_PER_10S_CHUNK.items()
    }

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---------------------------------------------------------------
    # Step 1: Load completed projects
    # ---------------------------------------------------------------
    if not args.queue_state.exists():
        print(f"[ERROR] Queue state not found: {args.queue_state}")
        sys.exit(1)

    projects = load_completed_projects(args.queue_state)
    if not projects:
        print("[INFO] No completed projects found in queue.")
        sys.exit(0)

    # Filter by card IDs if specified
    if args.card_ids:
        card_set = {cid.strip() for cid in args.card_ids}
        projects = [p for p in projects if p["card_id"] in card_set]
        if not projects:
            print(f"[ERROR] No matching completed projects for card IDs: {args.card_ids}")
            sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Load budget
    # ---------------------------------------------------------------
    budget = load_budget_tracker()
    if args.budget is not None:
        budget.total_budget_usd = args.budget
    budget.resolution = args.resolution
    budget.chunk_duration_s = args.chunk_duration
    budget.cost_per_chunk_usd = COST_PER_CHUNK.get(args.resolution, 0.07)

    print("=" * 60)
    print("Batch Facecam Queue Generator")
    print("=" * 60)
    print(f"  Resolution: {args.resolution}")
    print(f"  Budget: ${budget.total_budget_usd:.2f} total | "
          f"${budget.total_spent_usd:.2f} spent | "
          f"${budget.remaining_usd:.2f} remaining")
    print(f"  Facecam target: {args.facecam_percent * 100:.0f}% of project runtime")
    print(f"  Completed projects: {len(projects)}")
    if args.dry_run:
        print("  Mode: DRY RUN (no generation)")

    # ---------------------------------------------------------------
    # Step 3: Classify and plan for each project
    # ---------------------------------------------------------------
    project_plans = []  # (project, selected_groups, voiceover_path, avatar_path)
    total_est_cost = 0.0
    remaining = budget.remaining_usd

    for proj in projects:
        card_id = proj["card_id"]
        title = proj["title"]
        project_dir = proj["project_dir"]
        channel = proj["channel"]

        # Find SRT
        srt_path = find_project_srt(project_dir)
        if not srt_path:
            print(f"\n  [{card_id}] {title[:50]}...")
            print(f"    [SKIP] No SRT file found in {project_dir}")
            continue

        # Parse segments
        segments = parse_srt(srt_path)
        if not segments:
            print(f"\n  [{card_id}] {title[:50]}...")
            print(f"    [SKIP] SRT has no segments")
            continue

        # Get duration
        total_duration = get_voiceover_duration(project_dir)
        if total_duration <= 0:
            total_duration = segments[-1].end

        # Classify with Ollama (free, cached)
        classified = classify_talking_points(segments, title)

        # Merge adjacent talking points
        groups = merge_adjacent_talking_points(classified, max_gap_s=args.max_gap)

        # Check already-generated segments
        already_generated = set() if args.force else get_already_generated_indices(project_dir)

        # Filter out already-generated groups
        new_groups = [
            g for g in groups
            if not all(idx in already_generated for idx in g.segment_indices)
        ]

        # Calculate budget caps
        max_duration_s = total_duration * args.facecam_percent
        max_cost_for_duration = math.ceil(max_duration_s / CHUNK_DURATION_S) * budget.cost_per_chunk_usd
        max_cost = min(remaining, max_cost_for_duration)

        # Select best groups within budget
        selected = select_groups_for_budget(
            new_groups, max_duration_s, max_cost, args.resolution
        )

        # Find voiceover early (needed for Whisper alignment and generation)
        voiceover_path = None
        if selected:
            try:
                voiceover_path = Path(find_voiceover(str(project_dir)))
            except FileNotFoundError:
                print(f"    [SKIP] No voiceover audio found")
                continue

        # Whisper alignment — re-transcribe selected segments for precise timing
        if selected and voiceover_path and not args.skip_whisper:
            selected = whisper_align_groups(selected, voiceover_path, total_duration)

        # Display plan
        display_project_plan(
            title, card_id, total_duration, segments,
            classified, groups, selected, args.resolution, already_generated,
        )

        if selected:
            sel_cost = sum(g.cost_usd(args.resolution) for g in selected)
            total_est_cost += sel_cost
            remaining -= sel_cost

            try:
                avatar_path = get_next_avatar(channel)
            except FileNotFoundError as e:
                print(f"    [SKIP] Avatar not found: {e}")
                continue

            project_plans.append((proj, selected, voiceover_path, avatar_path))

    # ---------------------------------------------------------------
    # Summary before generation
    # ---------------------------------------------------------------
    total_groups = sum(len(sel) for _, sel, _, _ in project_plans)
    print(f"\n{'=' * 60}")
    print(f"Plan Summary")
    print(f"{'=' * 60}")
    print(f"  Projects to generate: {len(project_plans)}")
    print(f"  Total groups: {total_groups}")
    print(f"  Estimated cost: ${total_est_cost:.2f}")
    print(f"  Budget remaining after: ${budget.remaining_usd - total_est_cost:.2f}")

    if args.dry_run:
        print(f"\n  [DRY RUN] No generation performed.")
        sys.exit(0)

    if not project_plans:
        print(f"\n  Nothing to generate.")
        sys.exit(0)

    # ---------------------------------------------------------------
    # Step 4: Check prerequisites
    # ---------------------------------------------------------------
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n[ERROR] DASHSCOPE_API_KEY environment variable not set")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 5: Generate facecams
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Generating Facecams")
    print(f"{'=' * 60}")

    svc = WanFacecamService(
        api_key=api_key,
        resolution=args.resolution,
        chunk_duration=args.chunk_duration,
        region=args.region,
    )

    total_generated = 0
    total_actual_cost = 0.0

    for proj, selected, voiceover_path, avatar_path in project_plans:
        card_id = proj["card_id"]
        title = proj["title"]
        project_dir = proj["project_dir"]
        channel = proj["channel"]

        print(f"\n  Project: {title[:60]}...")
        facecam_dir = project_dir / "facecam"
        facecam_dir.mkdir(parents=True, exist_ok=True)

        # Load or create generation log
        log_data = load_generation_log(project_dir) or {
            "card_id": card_id,
            "project_title": title,
            "channel": channel,
            "segments": [],
        }

        for group in selected:
            cost = group.cost_usd(args.resolution)

            # HARD BUDGET CHECK before every generation
            if budget.remaining_usd < cost:
                print(
                    f"    [BUDGET] Stopping: ${budget.remaining_usd:.2f} remaining, "
                    f"need ${cost:.2f}"
                )
                break

            first, last = group.segment_indices[0], group.segment_indices[-1]
            label = f"seg{first:03d}-{last:03d}" if first != last else f"seg{first:03d}"
            print(f"    Generating {label} ({_fmt_duration(group.duration_s)}, ${cost:.2f})...")

            entry = generate_facecam_for_group(
                group=group,
                voiceover_path=voiceover_path,
                avatar_path=avatar_path,
                output_dir=facecam_dir,
                svc=svc,
                resolution=args.resolution,
            )

            if entry:
                total_generated += 1
                total_actual_cost += entry["cost_usd"]

                # Update budget tracker immediately (crash-safe)
                budget.total_spent_usd += entry["cost_usd"]
                if card_id not in budget.projects:
                    budget.projects[card_id] = {
                        "title": title,
                        "spent_usd": 0.0,
                        "duration_s": 0.0,
                        "segments": 0,
                    }
                budget.projects[card_id]["spent_usd"] += entry["cost_usd"]
                budget.projects[card_id]["duration_s"] += entry["duration_s"]
                budget.projects[card_id]["segments"] += 1
                save_budget_tracker(budget)

                # Update generation log
                log_data["segments"].append(entry)
                save_generation_log(log_data, project_dir)

                print(f"      OK: {entry['facecam_file']} ({entry['duration_s']:.1f}s)")
            else:
                print(f"      FAILED: {label}")
        else:
            # Loop completed without break — no budget issue
            continue
        # Budget exhausted — stop all projects
        break

    # ---------------------------------------------------------------
    # Step 6: Final summary
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Segments generated: {total_generated}")
    print(f"  Total cost: ${total_actual_cost:.2f}")
    print(f"  Budget spent: ${budget.total_spent_usd:.2f} / ${budget.total_budget_usd:.2f}")
    print(f"  Budget remaining: ${budget.remaining_usd:.2f}")

    sys.exit(0 if total_generated > 0 or args.dry_run else 1)


if __name__ == "__main__":
    main()
