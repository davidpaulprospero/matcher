"""
Segment utilities for audio-first pipeline.

Migrated from downloader.py lines 2563-2907.
Handles segment naming, merging, and collection from match results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .types import MatchedSegment, MergedSegment, DownloadedSegment

if TYPE_CHECKING:
    from ..state import AudioDownload

logger = logging.getLogger(__name__)


def get_segment_filename(video_id: str, start_seconds: float) -> str:
    """
    Generate filename with start time encoded.

    Migrated from downloader.py lines 2563-2575.

    Format: {video_id}_{start_seconds:04d}.mp4
    Example: abc123_0330.mp4 = video abc123, starts at 330 seconds (5:30)

    This naming allows:
    - Easy sorting by time
    - Extracting start time from filename
    - Direct mapping to original timestamps

    Args:
        video_id: YouTube video ID
        start_seconds: Start time in seconds

    Returns:
        Filename string
    """
    start_int = int(start_seconds)
    return f"{video_id}_{start_int:04d}.mp4"


def rename_segments_with_timing(
    download_dir: Path,
    video_id: str,
    merged_segments: List[MergedSegment]
) -> List[Optional[str]]:
    """
    Rename autonumber files to timestamp-based names.

    Migrated from downloader.py lines 2578-2654.

    yt-dlp with --download-sections creates files like:
        abc123_1.mp4, abc123_2.mp4, ...

    This renames them to:
        abc123_0000.mp4 (starts at 0s)
        abc123_0330.mp4 (starts at 330s)

    Args:
        download_dir: Directory containing downloaded files
        video_id: YouTube video ID
        merged_segments: Segments in order they were downloaded

    Returns:
        List of renamed file paths (None if file not found)
    """
    renamed_files = []

    for idx, segment in enumerate(merged_segments, start=1):
        new_name = download_dir / get_segment_filename(video_id, segment.start_time)

        # Try different autonumber formats and extensions
        # yt-dlp %(autonumber)s produces 5-digit padded by default: 00001, 00002
        possible_names = [
            f"{video_id}_{idx:05d}",  # 00001, 00002 (yt-dlp default)
            f"{video_id}_{idx}",       # 1, 2 (unpadded)
            f"{video_id}_{idx:02d}",   # 01, 02 (2-digit)
        ]
        extensions = ['.mp4', '.mkv', '.webm']

        found_file = None
        for name_base in possible_names:
            for ext in extensions:
                candidate = download_dir / f"{name_base}{ext}"
                if candidate.exists():
                    found_file = candidate
                    break
            if found_file:
                break

        try:
            if found_file:
                # Preserve original extension
                final_name = new_name.with_suffix(found_file.suffix)
                # Delete existing file if present (from previous run)
                if final_name.exists():
                    final_name.unlink()
                found_file.rename(final_name)
                renamed_files.append(str(final_name))
                logger.debug(f"Renamed {found_file.name} -> {final_name.name}")
            else:
                # Try glob as fallback
                pattern = f"{video_id}_*"
                matches = sorted(download_dir.glob(pattern))
                if idx <= len(matches):
                    found_file = matches[idx - 1]
                    final_name = new_name.with_suffix(found_file.suffix)
                    # Delete existing file if present
                    if final_name.exists():
                        final_name.unlink()
                    found_file.rename(final_name)
                    renamed_files.append(str(final_name))
                    logger.debug(f"Renamed (glob) {found_file.name} -> {final_name.name}")
                else:
                    logger.warning(f"Expected file not found: {video_id}_{idx}.mp4")
                    renamed_files.append(None)
        except OSError as e:
            logger.error(f"Failed to rename segment {idx} for {video_id}: {e}")
            renamed_files.append(None)

    return renamed_files


def merge_segments_with_buffer(
    segments: List[Tuple[float, float]],
    buffer_seconds: float = 30.0,
    merge_gap_seconds: float = 15.0,
    video_duration: float = None
) -> List[Tuple[float, float]]:
    """
    Merge overlapping segments after adding buffer.

    Migrated from downloader.py lines 2657-2727.

    Args:
        segments: List of (start, end) tuples
        buffer_seconds: Add this before/after each segment
        merge_gap_seconds: Merge if gap is less than this
        video_duration: Clamp end to video duration if provided

    Returns:
        List of merged (start, end) tuples
    """
    if not segments:
        return []

    # Step 0: Validate and filter segments
    validated = []
    for start, end in segments:
        # Type check
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            logger.warning(f"Invalid segment timestamps ({start}, {end}), skipping")
            continue

        # Fix negative start times
        if start < 0:
            logger.debug(f"Negative start time {start}, clamping to 0")
            start = 0

        # Skip invalid segments where end <= start
        if end <= start:
            logger.warning(f"Invalid segment [{start}, {end}] (end <= start), skipping")
            continue

        validated.append((start, end))

    if not validated:
        logger.warning("No valid segments after validation")
        return []

    # Step 1: Add buffer and clamp to valid range
    buffered = []
    for start, end in validated:
        new_start = max(0, start - buffer_seconds)
        new_end = end + buffer_seconds
        if video_duration:
            new_end = min(new_end, video_duration)
        buffered.append((new_start, new_end))

    # Step 2: Sort by start time
    buffered.sort(key=lambda x: x[0])

    # Step 3: Merge overlapping or close segments
    merged = [buffered[0]]
    for start, end in buffered[1:]:
        last_start, last_end = merged[-1]

        # Merge if overlapping OR gap is small
        if start <= last_end + merge_gap_seconds:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def collect_matched_segments(
    match_results: List,  # List[MatchResult]
    audio_downloads: Dict[str, 'AudioDownload']
) -> Dict[str, List[MatchedSegment]]:
    """
    Collect all matched segments from matching results.

    Migrated from downloader.py lines 2730-2828.

    Groups matches by video_id for efficient downloading.

    Args:
        match_results: Results from matching phase
        audio_downloads: Map of video_id -> AudioDownload

    Returns:
        Dict of video_id -> List[MatchedSegment]
    """
    segments_by_video: Dict[str, List[MatchedSegment]] = {}

    for idx, result in enumerate(match_results):
        # Process primary match (V1)
        if result.primary_match:
            seg = result.primary_match.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track="V1",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

        # Process alternatives (V2-V3)
        for alt_idx, alt in enumerate(result.alternatives or [], start=2):
            seg = alt.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track=f"V{alt_idx}",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

        # Process secondary matches (V4-V6)
        for sec_idx, sec in enumerate(result.secondary_matches or [], start=4):
            seg = sec.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track=f"V{sec_idx}",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

        # Process strategy matches (V7+)
        for strat in result.strategy_matches or []:
            seg = strat.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track="V7",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

    return segments_by_video


def _extract_video_id(file_path: str) -> Optional[str]:
    """
    Extract YouTube video ID from file path.

    Migrated from downloader.py lines 2831-2846.

    Assumes filename format: {video_id}.mp3 or {video_id}_0000.mp4

    Args:
        file_path: Path to video file

    Returns:
        Video ID or None if not found
    """
    if not file_path:
        return None

    filename = Path(file_path).stem

    # Handle segment format: abc123_0000
    if '_' in filename and filename.split('_')[-1].isdigit():
        return filename.rsplit('_', 1)[0]

    # Simple format: abc123
    return filename


def prepare_merged_segments(
    segments_by_video: Dict[str, List[MatchedSegment]],
    audio_downloads: Dict[str, 'AudioDownload'],
    buffer_seconds: float = 30.0,
    merge_gap_seconds: float = 15.0
) -> List[MergedSegment]:
    """
    Prepare merged segments for download.

    Migrated from downloader.py lines 2849-2907.

    Applies buffer, merges overlapping/close segments, and creates
    MergedSegment records ready for download.

    Args:
        segments_by_video: Dict of video_id -> List[MatchedSegment]
        audio_downloads: Map of video_id -> AudioDownload
        buffer_seconds: Padding around each match
        merge_gap_seconds: Merge if gap is smaller

    Returns:
        List of MergedSegment ready for download
    """
    all_merged = []

    for video_id, matches in segments_by_video.items():
        if not matches:
            continue

        # Get video duration for clamping
        audio = audio_downloads.get(video_id)
        video_duration = audio.duration if audio else None

        # Extract time ranges
        time_ranges = [(m.start_time, m.end_time) for m in matches]

        # Merge with buffer
        merged_ranges = merge_segments_with_buffer(
            time_ranges,
            buffer_seconds=buffer_seconds,
            merge_gap_seconds=merge_gap_seconds,
            video_duration=video_duration
        )

        # Create MergedSegment for each merged range
        for start, end in merged_ranges:
            # Find which original matches fall within this range
            contained_matches = [
                m for m in matches
                if start <= m.start_time and m.end_time <= end
            ]

            all_merged.append(MergedSegment(
                video_id=video_id,
                video_url=matches[0].video_url,
                start_time=start,
                end_time=end,
                original_matches=contained_matches,
                keyword=matches[0].keyword
            ))

    return all_merged
