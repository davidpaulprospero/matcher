"""
Timeline orchestration for OTIO generation.

Migrated from otio_builder.py create_timeline() function.
This module handles the core timeline assembly logic.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import opentimelineio as otio

from .utils import (
    _to_windows_path,
    _get_media_duration,
    get_segment_file_offset,
    get_confidence_color,
    create_clip_with_timewarp,
    optimize_timeline_gaps,
    MediaPathNormalizer,
    AUDIO_ONLY_EXTS,
    NON_MEDIA_EXTS,
    _is_audio_only,
    _has_problematic_path,
    seg_start as _seg_start,
    seg_end as _seg_end,
)
from .entities import _add_entity_images_to_track, _add_entity_videos_to_track
from .tracks import ClipBudgetTracker

# DaVinci Resolve clip count thresholds (see Rule 15 in CLAUDE.md)
# DaVinci OTIO import hangs when total clips exceed ~3130
CLIP_COUNT_WARNING_THRESHOLD = 2500  # Log warning when approaching limit
CLIP_COUNT_ERROR_THRESHOLD = 3000    # Log error when likely to fail


# Maximum clip extension factor for gap_mode='extend'
# Prevents extreme slowdowns when extending clips to fill large gaps
MAX_CLIP_EXTENSION_FACTOR = 2.0  # Max 2x original duration


def _is_non_media(file_path: str) -> bool:
    """Check if file is non-media (subtitles, text files that can't be imported)."""
    ext = Path(file_path).suffix.lower()
    return ext in NON_MEDIA_EXTS


def _find_audio_for_voiceover(vo_path: str) -> Optional[str]:
    """
    Find audio file for voiceover when given an SRT or other non-media file.

    Searches for matching audio files in the same directory:
    - Same name with audio extension (.mp3, .wav)
    - combined_output.mp3/wav (common pattern)
    - voiceover.mp3/wav (common pattern)

    When multiple candidates exist, returns the largest file by size
    (most likely the main voiceover audio rather than a click track or scratch mix).

    Returns:
        Path to audio file if found, None otherwise.
    """
    logger = logging.getLogger(__name__)
    vo_path_obj = Path(vo_path)
    vo_dir = vo_path_obj.parent
    vo_stem = vo_path_obj.stem

    candidates: List[Path] = []

    # Collect candidates: same name with audio extensions
    for ext in ['.mp3', '.wav', '.m4a', '.aac']:
        candidate = vo_dir / f"{vo_stem}{ext}"
        if candidate.exists():
            candidates.append(candidate)

    # Collect candidates: common voiceover file patterns
    for pattern in ['combined_output', 'voiceover', 'audio', 'vo']:
        for ext in ['.mp3', '.wav', '.m4a', '.aac']:
            candidate = vo_dir / f"{pattern}{ext}"
            if candidate.exists() and candidate not in candidates:
                candidates.append(candidate)

    if not candidates:
        return None

    if len(candidates) == 1:
        return str(candidates[0])

    # Multiple candidates: sort by file size descending, return largest
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    selected = candidates[0]
    size_mb = selected.stat().st_size / (1024 * 1024)
    logger.info(
        "Found %d audio candidates, selected %s (%.0fMB)",
        len(candidates), selected.name, size_mb
    )
    return str(selected)


def _is_missing_file(file_path: str) -> bool:
    """
    Check if the video file is missing from disk.

    Returns True if the file does not exist (including bare video IDs that
    failed segment resolution), False if it exists or is a URL.
    """
    # Skip check for URLs
    if file_path.startswith(('http://', 'https://', 'file://')):
        return False

    # Bare video ID (no path separators, no extension) = unresolved segment
    if '/' not in file_path and '\\' not in file_path:
        ext = Path(file_path).suffix
        if not ext:
            return True  # Bare video ID, no file on disk
        return False

    # Check if file exists
    return not Path(file_path).exists()


if TYPE_CHECKING:
    from ..config import Config
    from ..utils import MatchResult

logger = logging.getLogger(__name__)


def _count_timeline_clips(timeline: otio.schema.Timeline) -> int:
    """
    Count total number of clips (non-Gap items) across all tracks.

    This is used to detect when clip count approaches DaVinci Resolve's limit.

    Args:
        timeline: The OTIO timeline to count clips in

    Returns:
        Total number of clips (excludes Gaps)
    """
    total_clips = 0
    for track in timeline.tracks:
        for item in track:
            if isinstance(item, otio.schema.Clip):
                total_clips += 1
    return total_clips


def _log_clip_count_warnings(clip_count: int) -> None:
    """
    Log warnings if clip count approaches or exceeds DaVinci Resolve limits.

    DaVinci Resolve OTIO import hangs when total clips exceed ~3130.
    This function logs appropriate warnings to help users avoid this issue.

    Args:
        clip_count: Total number of clips in the timeline
    """
    if clip_count >= CLIP_COUNT_ERROR_THRESHOLD:
        logger.error(
            f"Timeline has {clip_count} clips, exceeding safe limit of {CLIP_COUNT_ERROR_THRESHOLD}. "
            f"DaVinci Resolve may hang during import. Consider using LITE mode (fewer tracks) "
            f"or splitting the timeline into multiple files."
        )
        print(f"  ⚠ CRITICAL: {clip_count} clips exceeds safe limit ({CLIP_COUNT_ERROR_THRESHOLD})")
        print(f"    → DaVinci Resolve may hang during OTIO import")
        print(f"    → Consider: LITE mode, fewer tracks, or split timeline")
    elif clip_count >= CLIP_COUNT_WARNING_THRESHOLD:
        logger.warning(
            f"Timeline has {clip_count} clips, approaching DaVinci limit of ~3130. "
            f"Consider reducing tracks or segments to avoid import issues."
        )
        print(f"  ⚠ Warning: {clip_count} clips approaching DaVinci limit (~3130)")


def _validate_entity_images(entity_images: Dict) -> Dict:
    """
    Validate entity images and filter out invalid entries.

    Returns a dict with only valid entities that have downloadable images.
    Handles both:
    - EntityImage objects with .file attribute
    - Plain string file paths (from checkpoint restoration)
    """
    validated = {}
    for entity_name, entity_result in entity_images.items():
        # Handle dict (from checkpoint) vs object
        if isinstance(entity_result, dict):
            images = entity_result.get('images', [])
        elif hasattr(entity_result, 'images'):
            images = entity_result.images
        else:
            continue

        if not images:
            continue

        # Filter out invalid image paths - handle both strings and objects
        valid_images = []
        for img in images:
            # Get file path - could be string or object with .file attr
            if isinstance(img, str):
                file_path = img
            elif hasattr(img, 'file'):
                file_path = img.file
            else:
                continue

            if file_path and Path(file_path).exists():
                valid_images.append(img)

        if valid_images:
            # Create validated result
            import copy
            if isinstance(entity_result, dict):
                validated_result = entity_result.copy()
                validated_result['images'] = valid_images
            else:
                validated_result = copy.copy(entity_result)
                validated_result.images = valid_images
            validated[entity_name] = validated_result

    return validated


def validate_media_paths(
    matches: List['MatchResult'],
    downloaded_segments: Optional[List] = None,
) -> Dict[str, object]:
    """
    Pre-flight validation of all media paths before timeline assembly.

    Scans every match (primary, alternatives, secondary, strategy) and checks
    each source file for common issues. Returns a summary dict so callers can
    log a concise table instead of discovering problems mid-build.

    Args:
        matches: List of MatchResult from matching stage.
        downloaded_segments: Optional segment list from audio-first download.

    Returns:
        Dict with keys:
            valid (int): count of valid paths
            audio_only (list[str]): segment indices with audio-only files
            missing (list[str]): segment indices with missing files
            problematic_path (list[str]): segment indices with problematic paths
            non_media (list[str]): segment indices with non-media files
            total (int): total paths checked
    """
    # Build segment file lookup from downloaded_segments
    segment_files: Dict[str, str] = {}
    if downloaded_segments:
        for seg in downloaded_segments:
            vid_id = seg.video_id
            if vid_id not in segment_files:
                segment_files[vid_id] = seg.file

    summary: Dict[str, object] = {
        'valid': 0,
        'audio_only': [],
        'missing': [],
        'problematic_path': [],
        'non_media': [],
        'total': 0,
    }

    def _resolve_source(source_file: str) -> str:
        """Resolve source file through downloaded segments if available."""
        if segment_files and source_file and ('/' not in source_file and '\\' not in source_file):
            # Looks like a video ID — check segment lookup
            return segment_files.get(source_file, source_file)
        return source_file

    def _check(source_file: str, segment_label: str) -> None:
        """Check a single source file and update summary."""
        resolved = _resolve_source(source_file)
        summary['total'] += 1

        if not resolved:
            summary['missing'].append(segment_label)
            return

        if _is_audio_only(resolved):
            summary['audio_only'].append(segment_label)
        elif _has_problematic_path(resolved):
            summary['problematic_path'].append(segment_label)
        elif _is_non_media(resolved):
            summary['non_media'].append(segment_label)
        elif _is_missing_file(resolved):
            summary['missing'].append(segment_label)
        else:
            summary['valid'] += 1

    for idx, match_result in enumerate(matches):
        label = f"S{idx:03d}"

        # Primary match
        _check(match_result.primary_match.video_segment.source_file, label)

        # Alternatives (V2-V3)
        for alt_i, alt in enumerate(match_result.alternatives):
            _check(alt.video_segment.source_file, f"{label}-alt{alt_i}")

        # Secondary matches (V4-V6)
        for sec_i, sec in enumerate(match_result.secondary_matches):
            _check(sec.video_segment.source_file, f"{label}-sec{sec_i}")

        # Strategy matches (V7+)
        if match_result.strategy_matches:
            for strat_i, strat in enumerate(match_result.strategy_matches):
                _check(strat.video_segment.source_file, f"{label}-strat{strat_i}")

    return summary


def _log_media_validation_summary(summary: Dict[str, object]) -> None:
    """Log a human-readable table of the pre-flight media validation results."""
    valid = summary['valid']
    audio_only = summary['audio_only']
    missing = summary['missing']
    problematic = summary['problematic_path']
    non_media = summary['non_media']
    total = summary['total']

    issue_parts = []
    if audio_only:
        ids = ','.join(audio_only[:10])
        suffix = f'... +{len(audio_only) - 10} more' if len(audio_only) > 10 else ''
        issue_parts.append(f"{len(audio_only)} audio-only ({ids}{suffix})")
    if missing:
        ids = ','.join(missing[:10])
        suffix = f'... +{len(missing) - 10} more' if len(missing) > 10 else ''
        issue_parts.append(f"{len(missing)} missing ({ids}{suffix})")
    if problematic:
        ids = ','.join(problematic[:10])
        suffix = f'... +{len(problematic) - 10} more' if len(problematic) > 10 else ''
        issue_parts.append(f"{len(problematic)} problematic-path ({ids}{suffix})")
    if non_media:
        ids = ','.join(non_media[:10])
        suffix = f'... +{len(non_media) - 10} more' if len(non_media) > 10 else ''
        issue_parts.append(f"{len(non_media)} non-media ({ids}{suffix})")

    issues_str = ', '.join(issue_parts) if issue_parts else 'none'
    logger.info(f"Media Validation: {valid} valid, {issues_str} (total: {total})")


def create_timeline(
    matches: List['MatchResult'],
    config: 'Config',
    voiceover_path: Optional[str] = None,
    frame_rate: float = 30.0,
    entity_images: Optional[Dict] = None,
    entity_videos: Optional[Dict] = None,
    downloaded_segments: Optional[List] = None,
    quality_metrics: Optional[Dict] = None
) -> otio.schema.Timeline:
    """
    Create OTIO timeline from matches.

    Track structure:
    - V1: Primary video (speed-adjusted) - enabled
    - V2: Alternative 1 (speed-adjusted) - disabled
    - V3: Alternative 2 (speed-adjusted) - disabled
    - V4: Secondary Primary (different video files from V1-V3) - disabled
    - V5: Secondary Alt 1 (different video files) - disabled
    - V6: Secondary Alt 2 (different video files) - disabled
    - V7: Embedding-Diversity strategy - disabled
    - V8: B-roll Only - disabled
    - V9: Entity Images (Google stills) - disabled
    - V10: Stock Videos (Pexels/Pixabay) - disabled
    - A1-A8: Corresponding audio tracks
    - A9: Voiceover - enabled

    Args:
        matches: List of match results from matching stage
        config: Pipeline configuration
        voiceover_path: Path to voiceover file
        frame_rate: Timeline frame rate
        entity_images: Entity images from EntityImagesStage
        entity_videos: Stock videos from EntityVideosStage
        downloaded_segments: Optional list of DownloadedSegment from audio-first mode.
            When provided, video segment files are used instead of audio files.
        quality_metrics: Optional dictionary with quality metrics to embed in metadata

    Returns:
        OTIO Timeline with all tracks populated
    """

    # Build lookup for audio-first mode: video_id -> segment info
    # This maps audio file video IDs to their downloaded video segment files
    segment_lookup = {}
    if downloaded_segments:
        for seg in downloaded_segments:
            video_id = seg.video_id
            if video_id not in segment_lookup:
                segment_lookup[video_id] = []
            segment_lookup[video_id].append({
                'file': seg.file,
                'start': seg.original_start,
                'end': seg.original_end
            })

    # =========================================================================
    # PRE-FLIGHT MEDIA VALIDATION
    # =========================================================================
    media_summary = validate_media_paths(matches, downloaded_segments)
    _log_media_validation_summary(media_summary)

    # =========================================================================
    # MEDIA PATH NORMALIZATION
    # =========================================================================
    # DaVinci Resolve hangs when the same video file is referenced from multiple
    # paths (e.g., stock/video.mp4 and broll/video.mp4). We normalize all paths
    # to use a single canonical location per unique file.
    #
    # First pass: collect all media paths from all matches
    path_normalizer = MediaPathNormalizer()

    for match_result in matches:
        # Primary match
        path_normalizer.register(match_result.primary_match.video_segment.source_file)

        # Alternatives
        for alt in match_result.alternatives:
            path_normalizer.register(alt.video_segment.source_file)

        # Secondary matches
        for sec in match_result.secondary_matches:
            path_normalizer.register(sec.video_segment.source_file)

        # Strategy matches
        if match_result.strategy_matches:
            for sm in match_result.strategy_matches:
                path_normalizer.register(sm.video_segment.source_file)

    # Also register segment files from audio-first mode
    if downloaded_segments:
        for seg in downloaded_segments:
            path_normalizer.register(seg.file)

    # Build canonical map
    path_normalizer.build_map()

    if path_normalizer.duplicates_found > 0:
        print(f"  ✓ Normalized {path_normalizer.duplicates_found} duplicate media paths")

    def resolve_video_segment(source_file: str, source_start: float) -> Tuple[str, float]:
        """
        Resolve audio file path to video segment path for audio-first mode.

        Also applies path normalization to prevent DaVinci Resolve hangs
        when the same file is referenced from multiple paths.

        Args:
            source_file: Original source file (may be audio .mp3)
            source_start: Start time in the original source

        Returns:
            Tuple of (resolved_path, adjusted_start_time)
            - If video segment found: (segment_file, time_relative_to_segment)
            - Otherwise: (original_source_file, original_source_start)
        """
        resolved_file = source_file
        adjusted_start = source_start

        if segment_lookup:
            # Extract video_id from the source file path
            # Audio files are like: /path/to/video_id.mp3 or /path/to/folder/video_id.mp3
            stem = Path(source_file).stem
            video_id = stem

            # Check if we have segment(s) for this video
            if video_id in segment_lookup:
                # Find the segment that contains this time
                segments = segment_lookup[video_id]
                for seg_info in segments:
                    # Check if source_start falls within this segment's range
                    if seg_info['start'] <= source_start <= seg_info['end']:
                        # Calculate the offset within the segment file
                        adjusted_start = source_start - seg_info['start']
                        resolved_file = seg_info['file']
                        break
                else:
                    # Find the nearest segment (closest start/end to source_start)
                    if segments:
                        best_seg = None
                        best_distance = float('inf')
                        for seg_info in segments:
                            if source_start < seg_info['start']:
                                dist = seg_info['start'] - source_start
                            elif source_start > seg_info['end']:
                                dist = source_start - seg_info['end']
                            else:
                                dist = 0
                            if dist < best_distance:
                                best_distance = dist
                                best_seg = seg_info
                        if best_seg and best_distance <= 60:
                            adjusted_start = max(0, source_start - best_seg['start'])
                            seg_duration = best_seg['end'] - best_seg['start']
                            adjusted_start = min(adjusted_start, max(0, seg_duration - 0.1))
                            resolved_file = best_seg['file']

        # Apply path normalization to prevent duplicate file references
        # which cause DaVinci Resolve to hang during OTIO import
        normalized_file = path_normalizer.get_canonical(resolved_file)

        return normalized_file, adjusted_start

    timeline = otio.schema.Timeline(name="Matched Footage")

    # Set tracks stack name to empty (DaVinci format)
    timeline.tracks.name = ""

    # Add Resolve_OTIO metadata (required for DaVinci import)
    timeline.metadata['Resolve_OTIO'] = {
        'Resolve OTIO Meta Version': '1.0'
    }

    # Add quality_summary metadata if quality_metrics provided
    if quality_metrics:
        timeline.metadata['quality_summary'] = quality_metrics

    # CRITICAL: Set global_start_time to valid RationalTime (not empty string!)
    # DaVinci Resolve hangs indefinitely if this is "" or invalid
    # Using 01:00:00:00 timecode start (86400 frames at 24fps, scaled to frame_rate)
    timeline.global_start_time = otio.opentime.RationalTime(
        int(3600 * frame_rate),  # 1 hour in frames
        frame_rate
    )

    rate = frame_rate

    # Determine number of alternative tracks (V2-V3)
    num_alternatives = config.output.num_alternatives if config.output.include_alternatives else 0

    # Secondary tracks (V4-V6) - always 3
    num_secondary = 3

    # Strategy tracks (V7+)
    strategy_names = []
    if config.output.include_strategy_tracks:
        strategy_names = list(config.output.strategy_tracks)

    strategy_display_names = {
        "embedding_diversity": "Embedding-Diversity",
        "broll_only": "B-roll Only"
    }

    # =========================================================================
    # CLIP BUDGET CHECK (proactive - before track building)
    # =========================================================================
    clip_budget = ClipBudgetTracker(
        match_count=len(matches),
        num_alternatives=num_alternatives,
        num_secondary=num_secondary,
        strategy_track_names=strategy_names,
        has_entity_images=bool(entity_images),
        has_entity_videos=bool(entity_videos),
    )
    estimated_clips = clip_budget.check_and_adjust(config)
    logger.info(f"Clip budget: {estimated_clips} estimated clips for {len(matches)} segments")

    # Re-read strategy config in case budget tracker disabled strategy tracks
    if not config.output.include_strategy_tracks:
        strategy_names = []

    # Create video tracks
    video_tracks = []

    # V1: Primary
    track = otio.schema.Track(name="V1 - Primary", kind=otio.schema.TrackKind.Video)
    track.metadata['Resolve_OTIO'] = {'Locked': False}
    video_tracks.append(track)

    # V2-V3: Alternatives
    for i in range(num_alternatives):
        track = otio.schema.Track(name=f"V{i+2} - Alternative {i+1}", kind=otio.schema.TrackKind.Video)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        video_tracks.append(track)

    # V4-V6: Secondary matches (different video files from V1-V3)
    secondary_names = ["Secondary Primary", "Secondary Alt 1", "Secondary Alt 2"]
    for i in range(num_secondary):
        track_num = 1 + num_alternatives + i + 1  # V4, V5, V6
        track = otio.schema.Track(name=f"V{track_num} - {secondary_names[i]}", kind=otio.schema.TrackKind.Video)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        video_tracks.append(track)

    # V7-V8: Strategy tracks
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + num_secondary + i + 1  # V7, V8
        track = otio.schema.Track(name=f"V{track_num} - {display_name}", kind=otio.schema.TrackKind.Video)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        video_tracks.append(track)

    # V9: Entity Images track (Google Images)
    image_track = otio.schema.Track(name="V9 - Entity Images", kind=otio.schema.TrackKind.Video)
    image_track.enabled = False  # Disabled by default, user enables as needed
    image_track.metadata['Resolve_OTIO'] = {'Locked': False}

    # V10: Stock Videos track (Pexels/Pixabay)
    stock_video_track = otio.schema.Track(name="V10 - Stock Videos", kind=otio.schema.TrackKind.Video)
    stock_video_track.enabled = False  # Disabled by default
    stock_video_track.metadata['Resolve_OTIO'] = {'Locked': False}

    # Create audio tracks for video audio
    audio_tracks = []

    # A1: Primary audio
    track = otio.schema.Track(name="A1 - Video Audio", kind=otio.schema.TrackKind.Audio)
    track.metadata['Resolve_OTIO'] = {'Locked': False}
    audio_tracks.append(track)

    # A2-A3: Alternative audio
    for i in range(num_alternatives):
        track = otio.schema.Track(name=f"A{i+2} - Alt {i+1} Audio", kind=otio.schema.TrackKind.Audio)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        audio_tracks.append(track)

    # A4-A6: Secondary audio tracks
    for i in range(num_secondary):
        track_num = 1 + num_alternatives + i + 1  # A4, A5, A6
        track = otio.schema.Track(name=f"A{track_num} - {secondary_names[i]} Audio", kind=otio.schema.TrackKind.Audio)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        audio_tracks.append(track)

    # A7-A8: Strategy audio tracks
    for i, strategy in enumerate(strategy_names):
        display_name = strategy_display_names.get(strategy, strategy)
        track_num = 1 + num_alternatives + num_secondary + i + 1  # A7, A8
        track = otio.schema.Track(name=f"A{track_num} - {display_name} Audio", kind=otio.schema.TrackKind.Audio)
        track.metadata['Resolve_OTIO'] = {'Locked': False}
        track.enabled = False
        audio_tracks.append(track)

    # Create voiceover track
    voiceover_track_num = 1 + num_alternatives + num_secondary + len(strategy_names) + 1
    voiceover_track = otio.schema.Track(
        name=f"A{voiceover_track_num} - Voiceover",
        kind=otio.schema.TrackKind.Audio
    )
    voiceover_track.metadata['Resolve_OTIO'] = {'Locked': False}

    # Track timeline position in FRAMES (integer) to avoid floating-point drift
    timeline_frames = 0

    # Get actual voiceover duration for proper timeline alignment
    actual_vo_duration = _get_media_duration(voiceover_path) if voiceover_path else None
    if actual_vo_duration:
        logger.info(f"Voiceover file duration: {actual_vo_duration:.2f}s")
        print(f"  ✓ Voiceover duration detected: {actual_vo_duration:.2f}s ({actual_vo_duration/60:.1f} min)")
    elif matches:
        # Fallback: use last segment end time + buffer for trailing content
        last_segment = matches[-1].primary_match.voiceover_segment
        fallback_duration = last_segment.end_time + 30.0  # Add 30s buffer for trailing
        logger.warning(f"ffprobe unavailable, using fallback duration: {fallback_duration:.2f}s (last segment + 30s buffer)")
        print(f"  ⚠ Using fallback VO duration: {fallback_duration:.2f}s (ffprobe unavailable)")
        actual_vo_duration = fallback_duration

    # Get the first segment's start time as timeline reference
    first_segment_start = _seg_start(matches[0].primary_match.voiceover_segment) if matches else 0.0

    # Apply voiceover offset to fix alignment when SRT timestamps don't match audio
    # Positive offset = shift clips later (audio is ahead of SRT)
    # Negative offset = shift clips earlier (audio is behind SRT)
    voiceover_offset = getattr(config.output, 'voiceover_offset', 0.0)
    if voiceover_offset != 0.0:
        logger.info(f"Applying voiceover offset: {voiceover_offset:+.2f}s")
        print(f"  ✓ Voiceover offset: {voiceover_offset:+.2f}s")

    # Gap threshold - collapse gaps smaller than this value
    # Whisper often inserts small gaps (~0.5-0.8s) between segments
    min_gap_threshold = getattr(config.output, 'min_gap_threshold', 0.0)
    if min_gap_threshold > 0.0:
        logger.info(f"Gap threshold: {min_gap_threshold:.2f}s (gaps below this will be collapsed)")
        print(f"  ✓ Gap threshold: {min_gap_threshold:.2f}s")

    # Time scale factor - stretch SRT timestamps to match actual audio duration
    # Useful when Whisper compresses timestamps (common issue)
    # Special value 0.0 = auto-calculate from actual audio duration / SRT end time
    time_scale_factor = getattr(config.output, 'time_scale_factor', 1.0)

    if time_scale_factor == 0.0 and actual_vo_duration and matches:
        # Auto-calculate: actual audio duration / last SRT segment end time
        last_srt_end = _seg_end(matches[-1].primary_match.voiceover_segment)
        if last_srt_end > 0:
            time_scale_factor = actual_vo_duration / last_srt_end
            logger.info(f"Auto-calculated time scale: {time_scale_factor:.4f} (audio {actual_vo_duration:.1f}s / SRT {last_srt_end:.1f}s)")
            print(f"  ✓ Auto time scale: {time_scale_factor:.4f}x ({actual_vo_duration:.1f}s / {last_srt_end:.1f}s)")
        else:
            time_scale_factor = 1.0
            logger.warning("Cannot auto-calculate time scale: SRT end time is 0")
    elif time_scale_factor != 1.0:
        logger.info(f"Time scale factor: {time_scale_factor:.4f} (stretching SRT timestamps)")
        print(f"  ✓ Time scale: {time_scale_factor:.4f}x")

    # =========================================================================
    # VOICEOVER DURATION vs SRT TIMELINE VALIDATION
    # =========================================================================
    if actual_vo_duration and matches:
        last_srt_end = _seg_end(matches[-1].primary_match.voiceover_segment)
        scaled_srt_end = last_srt_end * time_scale_factor
        if scaled_srt_end > 0:
            if actual_vo_duration < scaled_srt_end:
                shortfall = scaled_srt_end - actual_vo_duration
                logger.warning(
                    f"Voiceover audio ({actual_vo_duration:.1f}s) is shorter than "
                    f"scaled SRT timeline end ({scaled_srt_end:.1f}s) — "
                    f"last {shortfall:.1f}s of content will be cut off"
                )
                print(f"  ⚠ VO audio shorter than SRT timeline by {shortfall:.1f}s — content may be cut off")
            elif actual_vo_duration > scaled_srt_end + 30:
                trailing = actual_vo_duration - scaled_srt_end
                logger.info(
                    f"Voiceover audio ({actual_vo_duration:.1f}s) has {trailing:.1f}s trailing "
                    f"silence after last SRT segment ({scaled_srt_end:.1f}s) — "
                    f"potential alignment issue"
                )

    # Gap distribution mode - how to handle gaps between segments
    # - "scale": Scale SRT gaps by time_scale_factor (default)
    # - "proportional": Recalculate gaps to distribute content evenly
    # - "none": No gaps between clips
    # - "extend": Extend previous clip to fill gap (max 2x original duration)
    gap_mode = getattr(config.output, 'gap_mode', 'scale')
    if gap_mode not in ('scale', 'proportional', 'none', 'extend'):
        logger.warning(f"Invalid gap_mode '{gap_mode}', using 'scale'")
        gap_mode = 'scale'

    # Pre-calculate gap timing for proportional mode
    # This distributes gaps based on total available gap time, not SRT gaps
    proportional_gap_timing = {}
    if gap_mode == 'proportional' and actual_vo_duration and matches:
        # Calculate total segment content duration (scaled)
        total_content_duration = sum(
            (_seg_end(m.primary_match.voiceover_segment) - _seg_start(m.primary_match.voiceover_segment)) * time_scale_factor
            for m in matches
        )

        # Calculate total gap time available
        # Subtract leading silence and content from audio duration
        first_seg_start = _seg_start(matches[0].primary_match.voiceover_segment) * time_scale_factor
        total_gap_time = actual_vo_duration - first_seg_start - total_content_duration

        if total_gap_time > 0:
            # Calculate original SRT gaps for proportional distribution
            original_gaps = []
            for i in range(1, len(matches)):
                prev_end = _seg_end(matches[i-1].primary_match.voiceover_segment)
                curr_start = _seg_start(matches[i].primary_match.voiceover_segment)
                original_gap = max(0, curr_start - prev_end)
                original_gaps.append(original_gap)

            total_original_gaps = sum(original_gaps)

            # Distribute available gap time proportionally
            if total_original_gaps > 0:
                accumulated_time = first_seg_start + voiceover_offset  # Start after leading gap

                for i, match_result in enumerate(matches):
                    vo_seg = match_result.primary_match.voiceover_segment
                    segment_duration = (_seg_end(vo_seg) - _seg_start(vo_seg)) * time_scale_factor

                    proportional_gap_timing[i] = accumulated_time

                    # Add segment duration
                    accumulated_time += segment_duration

                    # Add proportional gap (except after last segment)
                    if i < len(original_gaps):
                        proportional_gap = (original_gaps[i] / total_original_gaps) * total_gap_time
                        accumulated_time += proportional_gap

                logger.info(f"Proportional gap distribution: {total_gap_time:.1f}s total gaps across {len(matches)} segments")
                print(f"  ✓ Gap mode: proportional ({total_gap_time:.1f}s distributed across {len(matches)-1} gaps)")
            else:
                # No original gaps - fall back to scale mode
                logger.warning("No gaps in SRT to distribute, falling back to scale mode")
                gap_mode = 'scale'
        else:
            # Content fills or exceeds audio - no room for gaps
            logger.warning(f"Content ({total_content_duration:.1f}s) fills audio ({actual_vo_duration:.1f}s), falling back to none mode")
            gap_mode = 'none'
    elif gap_mode == 'proportional':
        logger.warning("Cannot use proportional gap mode without audio duration, falling back to scale")
        gap_mode = 'scale'

    if gap_mode == 'none':
        logger.info("Gap mode: none - clips will be placed back-to-back")
        print(f"  ✓ Gap mode: none (back-to-back clips)")

    if gap_mode == 'extend':
        logger.info(f"Gap mode: extend - extending previous clips to fill gaps (max {MAX_CLIP_EXTENSION_FACTOR}x original duration)")
        print(f"  ✓ Gap mode: extend (max {MAX_CLIP_EXTENSION_FACTOR}x original duration)")

    # Adjusted first segment start includes the offset and scaling
    adjusted_first_segment_start = max(0.0, (first_segment_start * time_scale_factor) + voiceover_offset)

    # Add leading gap if first segment doesn't start at 0
    # This aligns video clips with the actual voiceover playback timing
    if matches and adjusted_first_segment_start > 0.1:  # More than 100ms of leading silence
        leading_frames = round(adjusted_first_segment_start * rate)
        logger.info(f"Adding {first_segment_start:.1f}s leading gap to align with voiceover start")

        leading_gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(leading_frames, rate)
            )
        )

        # Add leading gap to all video tracks
        for track in video_tracks:
            track.append(copy.deepcopy(leading_gap))
        # Add leading gap to all audio tracks
        for track in audio_tracks:
            track.append(copy.deepcopy(leading_gap))

        # Update timeline position
        timeline_frames += leading_frames

    # Track previous segment's info for extend mode
    prev_source_duration = 0.0
    prev_target_duration_frames = 0

    # Process each match
    for match_idx, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment
        vid_seg = match.video_segment

        # Check for gap before this segment (silence in voiceover)
        # The gap calculation depends on gap_mode:
        # - "scale": Use SRT gaps scaled by time_scale_factor
        # - "proportional": Use pre-calculated proportional gap positions
        # - "none": No gaps (back-to-back clips)
        # - "extend": Extend previous clip to fill gap (max 2x original duration)

        if gap_mode == 'none':
            # No gaps mode - clips are placed back-to-back
            expected_start_frames = timeline_frames
        elif gap_mode == 'extend':
            # Extend mode - calculate where this segment should start based on SRT
            # Then extend previous clip to fill the gap (handled below)
            scaled_segment_start = _seg_start(vo_seg) * time_scale_factor
            adjusted_segment_start = scaled_segment_start + voiceover_offset
            expected_start_frames = max(0, round((adjusted_segment_start - adjusted_first_segment_start) * frame_rate))
        elif gap_mode == 'proportional' and match_idx in proportional_gap_timing:
            # Proportional mode - use pre-calculated positions
            expected_start_seconds = proportional_gap_timing[match_idx]
            expected_start_frames = max(0, round(expected_start_seconds * frame_rate))
        else:
            # Scale mode (default) - use SRT gaps scaled by time_scale_factor
            scaled_segment_start = _seg_start(vo_seg) * time_scale_factor
            adjusted_segment_start = scaled_segment_start + voiceover_offset
            expected_start_frames = max(0, round((adjusted_segment_start - adjusted_first_segment_start) * frame_rate))

        if expected_start_frames > timeline_frames:
            # There's a gap - check if it's above the threshold
            gap_frames = expected_start_frames - timeline_frames
            gap_seconds = gap_frames / rate

            if gap_seconds >= min_gap_threshold:
                # Gap is significant

                if gap_mode == 'extend' and match_idx > 0 and prev_source_duration > 0:
                    # Extend mode: extend the previous clip to fill the gap
                    # Calculate maximum extension allowed (2x original duration)
                    max_extension_seconds = prev_source_duration * MAX_CLIP_EXTENSION_FACTOR

                    # Current duration of previous clip is prev_target_duration_frames / rate
                    # We want to add gap_seconds to it, but limit to max_extension_seconds total
                    prev_target_seconds = prev_target_duration_frames / rate
                    max_total_duration = max_extension_seconds
                    max_additional = max(0, max_total_duration - prev_target_seconds)

                    # How much can we actually extend?
                    extension_seconds = min(gap_seconds, max_additional)
                    extension_frames = round(extension_seconds * rate)

                    if extension_frames > 0:
                        # Extend previous clips on all tracks
                        for track in video_tracks:
                            if len(track) > 0:
                                last_item = track[-1]
                                if isinstance(last_item, otio.schema.Clip):
                                    # Extend the clip's source_range duration
                                    old_range = last_item.source_range
                                    new_duration_frames = int(old_range.duration.value) + extension_frames
                                    last_item.source_range = otio.opentime.TimeRange(
                                        start_time=old_range.start_time,
                                        duration=otio.opentime.RationalTime(new_duration_frames, rate)
                                    )

                        for track in audio_tracks:
                            if len(track) > 0:
                                last_item = track[-1]
                                if isinstance(last_item, otio.schema.Clip):
                                    old_range = last_item.source_range
                                    new_duration_frames = int(old_range.duration.value) + extension_frames
                                    last_item.source_range = otio.opentime.TimeRange(
                                        start_time=old_range.start_time,
                                        duration=otio.opentime.RationalTime(new_duration_frames, rate)
                                    )

                        logger.debug(f"Segment {match_idx}: Extended previous clip by {extension_seconds:.2f}s")
                        timeline_frames += extension_frames

                    # If there's remaining gap after max extension, insert a gap
                    remaining_gap_frames = gap_frames - extension_frames
                    if remaining_gap_frames > 0:
                        remaining_gap_seconds = remaining_gap_frames / rate
                        logger.debug(f"Segment {match_idx}: Extension limit reached, inserting {remaining_gap_seconds:.2f}s remaining gap")

                        remaining_gap_duration = otio.opentime.RationalTime(remaining_gap_frames, rate)
                        for track in video_tracks:
                            track.append(otio.schema.Gap(
                                source_range=otio.opentime.TimeRange(
                                    start_time=otio.opentime.RationalTime(0, rate),
                                    duration=remaining_gap_duration
                                )
                            ))
                        for track in audio_tracks:
                            track.append(otio.schema.Gap(
                                source_range=otio.opentime.TimeRange(
                                    start_time=otio.opentime.RationalTime(0, rate),
                                    duration=remaining_gap_duration
                                )
                            ))
                        timeline_frames += remaining_gap_frames

                else:
                    # Standard gap insertion (scale, proportional modes, or first segment in extend mode)
                    gap_duration = otio.opentime.RationalTime(gap_frames, rate)

                    logger.debug(f"Segment {match_idx}: Inserting {gap_seconds:.2f}s gap before (vo gap from {timeline_frames/rate:.2f}s to {expected_start_frames/rate:.2f}s)")

                    # Add gap to all video tracks
                    for track in video_tracks:
                        track.append(otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(0, rate),
                                duration=gap_duration
                            )
                        ))

                    # Add gap to all audio tracks
                    for track in audio_tracks:
                        track.append(otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(0, rate),
                                duration=gap_duration
                            )
                        ))

                    timeline_frames = expected_start_frames
            else:
                # Gap is below threshold - collapse it (don't insert gap, clips will be back-to-back)
                logger.debug(f"Segment {match_idx}: Collapsing {gap_seconds:.2f}s gap (below {min_gap_threshold:.2f}s threshold)")

        # Target duration = voiceover segment duration (scaled if time_scale_factor applied)
        target_duration = (_seg_end(vo_seg) - _seg_start(vo_seg)) * time_scale_factor
        duration_frames = round(target_duration * frame_rate)

        # Source duration = video segment duration
        source_duration = vid_seg.end_time - vid_seg.start_time
        source_start = vid_seg.start_time

        # Resolve audio file to video segment (audio-first mode)
        # This maps .mp3 audio files to downloaded .mp4 video segments
        resolved_source, adjusted_start = resolve_video_segment(vid_seg.source_file, source_start)

        # Also check for segment file offset from filename (legacy support)
        segment_offset = get_segment_file_offset(resolved_source)
        if segment_offset > 0 and resolved_source == vid_seg.source_file:
            # Only apply filename-based offset if we didn't already resolve
            adjusted_start = max(0, source_start - segment_offset)
            logger.debug(f"Segment file offset: {segment_offset}s, adjusted start: {adjusted_start}s")

        # Use the resolved source file and adjusted start time
        source_file_for_clip = resolved_source
        source_start = adjusted_start

        # Skip audio-only files - they cause DaVinci to hang during OTIO import
        # This happens when video segments weren't downloaded for some audio files
        # Also skip files with problematic unicode in path or missing files
        skip_reason = None
        if _is_audio_only(source_file_for_clip):
            skip_reason = "audio-only"
        elif _has_problematic_path(source_file_for_clip):
            skip_reason = "problematic path"
        elif _is_missing_file(source_file_for_clip):
            skip_reason = "missing file"
            logger.warning(f"Segment {match_idx}: Video file missing, inserting gap: {source_file_for_clip}")

        if skip_reason:
            if skip_reason != "missing file":
                logger.debug(f"Segment {match_idx}: Skipping {skip_reason}: {source_file_for_clip}")
            # Add gap instead of clip
            gap_duration = otio.opentime.RationalTime(duration_frames, rate)
            for track in video_tracks:
                track.append(otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                ))
            for track in audio_tracks:
                track.append(otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                ))
            timeline_frames += duration_frames
            continue

        # Determine clip color based on confidence
        clip_color = get_confidence_color(match.confidence)

        # Build metadata - include segment_index for post-edit analysis tracing
        segment_id = f"S{match_idx:03d}"  # S000, S001, S002, ...
        metadata = {
            'segment_index': match_idx,
            'segment_id': segment_id,
            'confidence': match.confidence,
            'reasoning': match.reasoning,
            'voiceover_text': vo_seg.text,
            'video_text': vid_seg.text,
            'is_keyword_match': match.is_keyword_match,
            'is_visual_match': match.is_visual_match,
            'embedding_similarity': match.embedding_similarity,
            'reuse_count': match.clip_reuse_count,
            'original_duration': source_duration,
            'target_duration': target_duration
        }

        # Propagate chapter title from video segment if available
        chapter_title = getattr(vid_seg, 'chapter_title', '')
        if chapter_title:
            metadata['chapter'] = chapter_title

        # Create primary video clip (V1) - prefix with segment ID for tracing
        clip_folder = Path(source_file_for_clip).parent.name
        clip_stem = Path(source_file_for_clip).stem
        v1_clip = create_clip_with_timewarp(
            name=f"[{segment_id}] {clip_folder}_{clip_stem} [{vid_seg.start_time:.1f}s]",
            source_path=source_file_for_clip,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata=metadata
        )

        # Set clip color
        v1_clip.metadata['clip_color'] = clip_color

        video_tracks[0].append(v1_clip)

        # Create primary audio clip (A1) - same source, same timing
        a1_clip = create_clip_with_timewarp(
            name=f"[{segment_id}] Audio: {clip_folder}_{clip_stem}",
            source_path=source_file_for_clip,
            source_start=source_start,
            source_duration=source_duration,
            target_duration=target_duration,
            frame_rate=frame_rate,
            metadata={'from_track': 'V1'}
        )
        audio_tracks[0].append(a1_clip)

        # Process alternatives (V2-V3, A2-A3)
        for alt_idx in range(num_alternatives):
            if alt_idx < len(match_result.alternatives):
                alt = match_result.alternatives[alt_idx]
                alt_seg = alt.video_segment

                alt_source_duration = alt_seg.end_time - alt_seg.start_time
                alt_source_start = alt_seg.start_time

                # Resolve audio file to video segment (audio-first mode)
                alt_resolved_source, alt_adjusted_start = resolve_video_segment(alt_seg.source_file, alt_source_start)

                # Legacy segment file offset support
                alt_segment_offset = get_segment_file_offset(alt_resolved_source)
                if alt_segment_offset > 0 and alt_resolved_source == alt_seg.source_file:
                    alt_adjusted_start = max(0, alt_source_start - alt_segment_offset)

                alt_source_file = alt_resolved_source
                alt_source_start = alt_adjusted_start

                # Skip audio-only files, problematic paths, and missing files
                if _is_audio_only(alt_source_file) or _has_problematic_path(alt_source_file) or _is_missing_file(alt_source_file):
                    if _is_missing_file(alt_source_file):
                        logger.warning(f"Segment {match_idx} ALT{alt_idx+1}: Video file missing, inserting gap: {alt_source_file}")
                    gap_duration = otio.opentime.RationalTime(duration_frames, rate)
                    video_tracks[alt_idx + 1].append(otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=gap_duration
                        )
                    ))
                    audio_tracks[alt_idx + 1].append(otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=gap_duration
                        )
                    ))
                    continue

                alt_metadata = {
                    'confidence': alt.confidence,
                    'reasoning': alt.reasoning,
                    'original_duration': alt_source_duration,
                    'target_duration': target_duration
                }

                # Propagate chapter title from video segment if available
                alt_chapter_title = getattr(alt_seg, 'chapter_title', '')
                if alt_chapter_title:
                    alt_metadata['chapter'] = alt_chapter_title

                # Alternative video clip - include segment ID for tracing
                alt_folder = Path(alt_source_file).parent.name
                alt_stem = Path(alt_source_file).stem
                alt_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] ALT{alt_idx+1}: {alt_folder}_{alt_stem}",
                    source_path=alt_source_file,
                    source_start=alt_source_start,
                    source_duration=alt_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=alt_metadata
                )

                # Set clip color for alternatives too
                alt_v_clip.metadata['clip_color'] = get_confidence_color(alt.confidence)

                video_tracks[alt_idx + 1].append(alt_v_clip)

                # Alternative audio clip
                alt_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio ALT{alt_idx+1}: {alt_folder}_{alt_stem}",
                    source_path=alt_source_file,
                    source_start=alt_source_start,
                    source_duration=alt_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata={'from_track': f'V{alt_idx+2}'}
                )
                audio_tracks[alt_idx + 1].append(alt_a_clip)
            else:
                # No alternative available - add gap (use rounded frames)
                gap_duration = otio.opentime.RationalTime(duration_frames, rate)

                v_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                video_tracks[alt_idx + 1].append(v_gap)

                a_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                audio_tracks[alt_idx + 1].append(a_gap)

        # Process secondary tracks (V4-V6) - different video files from V1-V3
        secondary_base_idx = 1 + num_alternatives  # Index where secondary tracks start

        for sec_idx in range(num_secondary):
            track_idx = secondary_base_idx + sec_idx

            if sec_idx < len(match_result.secondary_matches):
                sec_match = match_result.secondary_matches[sec_idx]
                sec_seg = sec_match.video_segment
                sec_source_duration = sec_seg.end_time - sec_seg.start_time
                sec_source_start = sec_seg.start_time

                # Resolve audio file to video segment (audio-first mode)
                sec_resolved_source, sec_adjusted_start = resolve_video_segment(sec_seg.source_file, sec_source_start)

                # Legacy segment file offset support
                sec_segment_offset = get_segment_file_offset(sec_resolved_source)
                if sec_segment_offset > 0 and sec_resolved_source == sec_seg.source_file:
                    sec_adjusted_start = max(0, sec_source_start - sec_segment_offset)

                sec_source_file = sec_resolved_source
                sec_source_start = sec_adjusted_start

                # Skip audio-only files, problematic paths, and missing files
                if _is_audio_only(sec_source_file) or _has_problematic_path(sec_source_file) or _is_missing_file(sec_source_file):
                    if _is_missing_file(sec_source_file):
                        logger.warning(f"Segment {match_idx} SEC{sec_idx+1}: Video file missing, inserting gap: {sec_source_file}")
                    gap_duration = otio.opentime.RationalTime(duration_frames, rate)
                    video_tracks[track_idx].append(otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=gap_duration
                        )
                    ))
                    audio_tracks[track_idx].append(otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=gap_duration
                        )
                    ))
                    continue

                sec_metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': sec_match.confidence,
                    'reasoning': sec_match.reasoning,
                    'original_duration': sec_source_duration,
                    'target_duration': target_duration,
                    'is_secondary': True
                }

                # Propagate chapter title from video segment if available
                sec_chapter_title = getattr(sec_seg, 'chapter_title', '')
                if sec_chapter_title:
                    sec_metadata['chapter'] = sec_chapter_title

                # Secondary video clip - include segment ID for tracing
                sec_label = secondary_names[sec_idx] if sec_idx < len(secondary_names) else f"Secondary {sec_idx}"
                sec_folder = Path(sec_source_file).parent.name
                sec_stem = Path(sec_source_file).stem
                sec_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] {sec_label}: {sec_folder}_{sec_stem}",
                    source_path=sec_source_file,
                    source_start=sec_source_start,
                    source_duration=sec_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=sec_metadata
                )

                # Color for secondary tracks
                secondary_colors = ["PURPLE", "BLUE", "TEAL"]
                sec_v_clip.metadata['clip_color'] = secondary_colors[sec_idx] if sec_idx < len(secondary_colors) else "GRAY"

                video_tracks[track_idx].append(sec_v_clip)

                # Secondary audio clip
                sec_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio {sec_label}: {sec_folder}_{sec_stem}",
                    source_path=sec_source_file,
                    source_start=sec_source_start,
                    source_duration=sec_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata={'from_track': f'V{track_idx+1}'}
                )
                audio_tracks[track_idx].append(sec_a_clip)
            else:
                # No secondary match available - add gap
                gap_duration = otio.opentime.RationalTime(duration_frames, rate)

                v_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                video_tracks[track_idx].append(v_gap)

                a_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                audio_tracks[track_idx].append(a_gap)

        # Process strategy tracks (V7-V8: embedding_diversity, broll_only)
        strategy_base_idx = 1 + num_alternatives + num_secondary  # Index where strategy tracks start

        for strat_idx, strategy in enumerate(strategy_names):
            track_idx = strategy_base_idx + strat_idx

            # Find strategy match for this strategy
            strat_match = None
            if match_result.strategy_matches:
                for sm in match_result.strategy_matches:
                    if sm.strategy == strategy:
                        strat_match = sm
                        break

            if strat_match:
                strat_seg = strat_match.video_segment
                strat_source_duration = strat_seg.end_time - strat_seg.start_time
                strat_source_start = strat_seg.start_time

                # Resolve audio file to video segment (audio-first mode)
                strat_resolved_source, strat_adjusted_start = resolve_video_segment(strat_seg.source_file, strat_source_start)

                # Legacy segment file offset support
                strat_segment_offset = get_segment_file_offset(strat_resolved_source)
                if strat_segment_offset > 0 and strat_resolved_source == strat_seg.source_file:
                    strat_adjusted_start = max(0, strat_source_start - strat_segment_offset)

                strat_source_file = strat_resolved_source
                strat_source_start = strat_adjusted_start

                # Skip audio-only files, problematic paths, and missing files
                if _is_audio_only(strat_source_file) or _has_problematic_path(strat_source_file) or _is_missing_file(strat_source_file):
                    if _is_missing_file(strat_source_file):
                        logger.warning(f"Segment {match_idx} {strategy.upper()}: Video file missing, inserting gap: {strat_source_file}")
                    gap_duration = otio.opentime.RationalTime(duration_frames, rate)
                    video_tracks[track_idx].append(otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=gap_duration
                        )
                    ))
                    audio_tracks[track_idx].append(otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=gap_duration
                        )
                    ))
                    continue

                strat_metadata = {
                    'segment_index': match_idx,
                    'segment_id': segment_id,
                    'confidence': strat_match.confidence,
                    'reasoning': strat_match.reasoning,
                    'strategy': strat_match.strategy,
                    'original_duration': strat_source_duration,
                    'target_duration': target_duration
                }

                # Propagate chapter title from video segment if available
                strat_chapter_title = getattr(strat_seg, 'chapter_title', '')
                if strat_chapter_title:
                    strat_metadata['chapter'] = strat_chapter_title

                # Strategy video clip - include segment ID for tracing
                strat_folder = Path(strat_source_file).parent.name
                strat_stem = Path(strat_source_file).stem
                strat_v_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] {strategy.upper()}: {strat_folder}_{strat_stem}",
                    source_path=strat_source_file,
                    source_start=strat_source_start,
                    source_duration=strat_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata=strat_metadata
                )

                # Color based on strategy
                strategy_colors = {
                    "embedding_diversity": "PINK",
                    "broll_only": "TEAL"
                }
                strat_v_clip.metadata['clip_color'] = strategy_colors.get(strategy, "GRAY")

                video_tracks[track_idx].append(strat_v_clip)

                # Strategy audio clip
                strat_a_clip = create_clip_with_timewarp(
                    name=f"[{segment_id}] Audio {strategy.upper()}: {strat_folder}_{strat_stem}",
                    source_path=strat_source_file,
                    source_start=strat_source_start,
                    source_duration=strat_source_duration,
                    target_duration=target_duration,
                    frame_rate=frame_rate,
                    metadata={'from_track': f'V{track_idx+1}', 'strategy': strategy}
                )
                audio_tracks[track_idx].append(strat_a_clip)
            else:
                # No strategy match available - add gap
                gap_duration = otio.opentime.RationalTime(duration_frames, rate)

                v_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                video_tracks[track_idx].append(v_gap)

                a_gap = otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0, rate),
                        duration=gap_duration
                    )
                )
                audio_tracks[track_idx].append(a_gap)

        # Track previous segment info for extend mode
        prev_source_duration = source_duration
        prev_target_duration_frames = duration_frames

        # Update timeline position using integer frames to avoid drift
        timeline_frames += duration_frames

    # Add trailing gap to match actual voiceover duration
    # This ensures video tracks extend to cover trailing audio (music, silence, outro)
    if actual_vo_duration and matches:
        accumulated_duration = timeline_frames / rate  # Current timeline in seconds

        if actual_vo_duration > accumulated_duration + 0.1:  # More than 100ms trailing
            trailing_seconds = actual_vo_duration - accumulated_duration
            trailing_frames = round(trailing_seconds * rate)
            logger.info(f"Adding {trailing_seconds:.1f}s trailing gap to match voiceover end")
            print(f"  ✓ Adding {trailing_seconds:.1f}s trailing gap (VO: {actual_vo_duration:.1f}s, timeline: {accumulated_duration:.1f}s)")

            trailing_gap = otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, rate),
                    duration=otio.opentime.RationalTime(trailing_frames, rate)
                )
            )

            # Add trailing gap to all video tracks
            for track in video_tracks:
                track.append(copy.deepcopy(trailing_gap))
            # Add trailing gap to all audio tracks
            for track in audio_tracks:
                track.append(copy.deepcopy(trailing_gap))

            # Update timeline position to include trailing content
            timeline_frames += trailing_frames

    # Add voiceover track
    if voiceover_path and matches:
        # Check if voiceover is a non-media file (SRT, VTT, etc.) and find audio alternative
        actual_vo_path = voiceover_path
        if _is_non_media(voiceover_path):
            audio_path = _find_audio_for_voiceover(voiceover_path)
            if audio_path:
                logger.info(f"Voiceover is subtitle file, using audio file: {audio_path}")
                actual_vo_path = audio_path
            else:
                logger.warning(f"Voiceover is subtitle file with no audio found, skipping voiceover track: {voiceover_path}")
                actual_vo_path = None

        if actual_vo_path:
            # Create absolute path for voiceover (forward slashes for DaVinci)
            abs_vo_path = _to_windows_path(actual_vo_path)
            vo_folder = Path(actual_vo_path).parent.name
            vo_filename = Path(actual_vo_path).name
            vo_unique_name = f"{vo_folder}_{vo_filename}"

            # Use actual voiceover file duration if available, otherwise use accumulated frames
            if actual_vo_duration:
                vo_total_frames = round(actual_vo_duration * rate)
                logger.info(f"Voiceover clip: using actual duration {actual_vo_duration:.2f}s ({vo_total_frames} frames)")
            else:
                vo_total_frames = timeline_frames
                logger.warning("Could not determine voiceover duration, using accumulated segment total")

            # Create proper ExternalReference with available_range
            vo_available_range = otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(vo_total_frames, rate)
            )

            vo_ref = otio.schema.ExternalReference(
                target_url=abs_vo_path,
                available_range=vo_available_range
            )
            vo_ref.name = vo_unique_name  # Unique name includes folder

            # Voiceover clip starts at 0 and uses actual file duration
            # Video/audio tracks have leading gap added to align with VO playback
            vo_clip = otio.schema.Clip(
                name="Voiceover",
                media_reference=vo_ref,
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, rate),
                    duration=otio.opentime.RationalTime(vo_total_frames, rate)
                )
            )
            vo_clip.metadata['Resolve_OTIO'] = {}  # Required for DaVinci import
            voiceover_track.append(vo_clip)

    # Add all tracks to timeline
    for track in video_tracks:
        timeline.tracks.append(track)

    for track in audio_tracks:
        timeline.tracks.append(track)

    # Only add voiceover track if it has content
    if len(voiceover_track) > 0:
        timeline.tracks.append(voiceover_track)

    # Populate image track if entity_images provided
    if entity_images:
        # Log what we received
        print(f"  [V9] Entity images received: {len(entity_images)} entities")
        for ename, eresult in entity_images.items():
            img_count = len(getattr(eresult, 'images', []))
            print(f"    • {ename}: {img_count} images")

        # Validate and filter entity images before using
        validated_entity_images = _validate_entity_images(entity_images)
        if validated_entity_images:
            total_images = sum(len(e.images) for e in validated_entity_images.values())
            print(f"  [V9] After validation: {len(validated_entity_images)} entities, {total_images} images")
            logger.info(f"Entity images: {len(validated_entity_images)} entities with valid images")
            _add_entity_images_to_track(
                image_track=image_track,
                entity_images=validated_entity_images,
                matches=matches,
                frame_rate=rate,
                config=config,
                time_scale_factor=time_scale_factor
            )
        else:
            logger.warning("No valid entity images after validation")

    # Always add V9 Entity Images track (even if empty, for manual use)
    timeline.tracks.append(image_track)

    # Populate stock video track if entity_videos provided
    if entity_videos:
        # Log what we received
        print(f"  [V10] Stock videos received: {len(entity_videos)} entities")
        for ename, eresult in entity_videos.items():
            vid_count = len(getattr(eresult, 'videos', []))
            print(f"    • {ename}: {vid_count} videos")

        _add_entity_videos_to_track(
            video_track=stock_video_track,
            entity_videos=entity_videos,
            matches=matches,
            frame_rate=rate,
            config=config,
            time_scale_factor=time_scale_factor
        )

    # Always add V10 Stock Videos track (even if empty, for manual use)
    timeline.tracks.append(stock_video_track)

    # Optimize gaps in all tracks (merge consecutive, remove trailing)
    # This improves DaVinci Resolve import performance
    optimize_timeline_gaps(timeline)

    return timeline
