"""
Voiceover silence removal - preprocesses audio before SRT generation.

Removes long silences from voiceover audio while preserving natural
breathing room and intentional pauses. Uses pydub's detect_nonsilent()
to find speech regions, then concatenates them with padding and crossfades.

The original file is never modified; output is written to {stem}_trimmed{ext}.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def remove_voiceover_silence(
    audio_path: str,
    *,
    min_silence_len_ms: int = 700,
    silence_thresh_dbfs: int = -35,
    keep_silence_ms: int = 250,
    crossfade_ms: int = 50,
) -> Optional[str]:
    """Remove silence from voiceover audio, return path to trimmed file.

    Args:
        audio_path: Path to the input audio file.
        min_silence_len_ms: Minimum silence duration (ms) to be removed.
        silence_thresh_dbfs: Absolute silence threshold in dBFS.
        keep_silence_ms: Padding (ms) to keep on each side of speech regions.
        crossfade_ms: Crossfade duration (ms) between concatenated regions.

    Returns:
        Path to the trimmed audio file, or the original path if trimming
        was skipped (not enough silence to justify re-encoding).
        Returns None on error.
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except ImportError:
        logger.warning("pydub not installed — skipping silence removal")
        return audio_path

    src = Path(audio_path)
    if not src.exists():
        logger.warning("Audio file not found for silence removal: %s", audio_path)
        return audio_path

    # Skip if already a trimmed file
    if "_trimmed" in src.stem:
        logger.debug("Skipping silence removal — file already trimmed: %s", src.name)
        return audio_path

    # Output path: alongside original
    trimmed_path = src.with_name(f"{src.stem}_trimmed{src.suffix}")

    # Skip re-processing if trimmed file exists and is newer than original
    if trimmed_path.exists():
        if trimmed_path.stat().st_mtime >= src.stat().st_mtime:
            logger.info("Using existing trimmed file: %s", trimmed_path.name)
            return str(trimmed_path)

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as exc:
        logger.warning("Failed to load audio for silence removal: %s", exc)
        return audio_path

    original_duration_ms = len(audio)
    if original_duration_ms == 0:
        return audio_path

    # Adaptive threshold: lowers threshold for quiet recordings to avoid
    # treating quiet speech as silence. For normal/loud recordings, uses
    # the configured absolute threshold unchanged.
    adaptive_thresh = min(silence_thresh_dbfs, audio.dBFS + 10)

    # Detect non-silent (speech) regions
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=adaptive_thresh,
    )

    if not nonsilent_ranges:
        logger.warning("No speech detected in %s — returning original", src.name)
        return audio_path

    # Extend each region by keep_silence_ms on each side (clamped)
    extended = []
    for start_ms, end_ms in nonsilent_ranges:
        ext_start = max(0, start_ms - keep_silence_ms)
        ext_end = min(original_duration_ms, end_ms + keep_silence_ms)
        extended.append((ext_start, ext_end))

    # Merge overlapping regions after extension
    merged = [extended[0]]
    for start_ms, end_ms in extended[1:]:
        prev_start, prev_end = merged[-1]
        if start_ms <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_ms))
        else:
            merged.append((start_ms, end_ms))

    # Calculate how much would be removed
    kept_duration_ms = sum(end - start for start, end in merged)
    removed_ms = original_duration_ms - kept_duration_ms
    removed_pct = (removed_ms / original_duration_ms) * 100

    # If less than 3% would be removed, not worth re-encoding
    if removed_pct < 3.0:
        logger.info(
            "Silence removal: only %.1f%% silence — skipping (threshold 3%%)",
            removed_pct,
        )
        return audio_path

    # Warn if too aggressive
    if removed_pct > 40.0:
        logger.warning(
            "Silence removal would remove %.1f%% of audio — threshold may be "
            "too aggressive (file: %s, adaptive_thresh=%.1f dBFS)",
            removed_pct,
            src.name,
            adaptive_thresh,
        )

    # Concatenate speech regions with crossfade
    # Clamp crossfade to not exceed shortest segment
    min_segment_len = min(end - start for start, end in merged)
    effective_crossfade = min(crossfade_ms, min_segment_len // 2)

    trimmed_audio = audio[merged[0][0]:merged[0][1]]
    for start_ms, end_ms in merged[1:]:
        segment = audio[start_ms:end_ms]
        if effective_crossfade > 0:
            trimmed_audio = trimmed_audio.append(segment, crossfade=effective_crossfade)
        else:
            trimmed_audio = trimmed_audio + segment

    # Export
    try:
        trimmed_audio.export(str(trimmed_path), format=_get_export_format(src.suffix))
    except Exception as exc:
        logger.warning("Failed to export trimmed audio: %s", exc)
        return audio_path

    trimmed_duration_ms = len(trimmed_audio)
    logger.info(
        "Silence removed: %.1fs → %.1fs (%.1f%% removed, %d regions) → %s",
        original_duration_ms / 1000,
        trimmed_duration_ms / 1000,
        removed_pct,
        len(merged),
        trimmed_path.name,
    )

    return str(trimmed_path)


def _get_export_format(suffix: str) -> str:
    """Map file extension to pydub export format string."""
    mapping = {
        ".mp3": "mp3",
        ".wav": "wav",
        ".m4a": "mp4",
        ".aac": "adts",
        ".flac": "flac",
        ".ogg": "ogg",
    }
    return mapping.get(suffix.lower(), "mp3")
