"""
Unified entity track generation for images and videos.

Eliminates 200+ lines of duplication between _add_entity_images_to_track()
and _add_entity_videos_to_track() by using polymorphism for the ~15% that differs.

Key Features:
- Sticky entity matching: Last matched entity persists across segments
- Semantic matching: Word overlap scoring for relevance
- No duplicate sources: Same media won't appear twice in one segment
- Configurable matching thresholds

Supports both V9 (Entity Images - Google/Bing) and V10 (Stock Videos - Pexels/Pixabay).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Literal

import opentimelineio as otio

from .utils import _to_windows_path, _get_media_duration

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import MatchResult

logger = logging.getLogger(__name__)

# Type alias for entity type
EntityType = Literal["images", "videos"]


def _has_problematic_path(file_path: str) -> bool:
    """
    Check if file path has characters that cause DaVinci OTIO import to hang.

    Problematic patterns:
    - Corrupted unicode (replacement char U+FFFD shown as �)
    - Non-ASCII characters in paths (accents, special chars)
    """
    try:
        if '\ufffd' in file_path or '�' in file_path:
            return True
        for char in file_path:
            if ord(char) > 127:
                return True
        return False
    except Exception:
        return True


def _get_attr(obj, name: str, default=None):
    """Get attribute from either dict or object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _find_best_entity_match(
    vo_text: str,
    entity_dict: Dict,
    last_matched_entity: Optional[str] = None,
    enable_sticky: bool = False,
    semantic_threshold: float = 0.15
) -> Tuple[Optional[str], str]:
    """
    Find the best matching entity for a voiceover segment.

    Strategy:
    1. Exact match: Entity name appears in voiceover text
    2. Semantic match: Use embedding similarity between voiceover and entity query
    3. Sticky (optional): Use last matched entity if no match found

    Args:
        vo_text: Voiceover segment text (lowercase)
        entity_dict: Dict of entity_name -> EntityResult
        last_matched_entity: Previous segment's matched entity name
        enable_sticky: Whether to fall back to previous entity (creates continuous blocks if True)
        semantic_threshold: Minimum word overlap score for semantic match (0.0-1.0)

    Returns:
        (entity_name, match_type) where match_type is 'exact', 'semantic', or 'sticky'
    """
    # 1. Try exact match first
    for entity_name, entity_result in entity_dict.items():
        if entity_name.lower() in vo_text:
            assets = _get_attr(entity_result, 'images') or _get_attr(entity_result, 'videos')
            if assets:
                return entity_name, 'exact'

    # 2. Try semantic matching using entity query similarity
    best_entity = None
    best_score = 0.0

    try:
        # Simple word overlap scoring as semantic proxy
        # (Full embedding similarity would require pre-computed embeddings)
        vo_words = set(vo_text.split())

        for entity_name, entity_result in entity_dict.items():
            assets = _get_attr(entity_result, 'images') or _get_attr(entity_result, 'videos')
            if not assets:
                continue

            # Get query text for matching
            query = _get_attr(entity_result, 'query', entity_name)
            query_words = set(query.lower().split())

            # Also include entity type in matching
            entity_type = _get_attr(entity_result, 'entity_type', '')
            if entity_type:
                query_words.update(entity_type.lower().split())

            # Calculate word overlap score
            common_words = vo_words & query_words
            if common_words:
                # Jaccard-like similarity
                score = len(common_words) / (len(vo_words | query_words) + 1)
                if score > best_score:
                    best_score = score
                    best_entity = entity_name

        # Require minimum semantic score threshold
        if best_entity and best_score >= semantic_threshold:
            return best_entity, 'semantic'
    except Exception as e:
        logger.debug(f"Semantic matching failed: {e}")

    # 3. Fall back to sticky entity (if enabled)
    if enable_sticky and last_matched_entity and last_matched_entity in entity_dict:
        entity_result = entity_dict[last_matched_entity]
        assets = _get_attr(entity_result, 'images') or _get_attr(entity_result, 'videos')
        if assets:
            return last_matched_entity, 'sticky'

    return None, 'none'


def _get_video_duration_frames(video_path: str, frame_rate: float) -> Optional[int]:
    """
    Get video duration in frames using ffprobe.

    Returns None if ffprobe fails or is not available.
    """
    import subprocess

    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            duration_sec = float(result.stdout.strip())
            return int(duration_sec * frame_rate)
    except Exception:
        pass

    return None


def add_entity_media_to_track(
    track: otio.schema.Track,
    entity_data: Dict,
    matches: List['MatchResult'],
    frame_rate: float,
    config: 'Config',
    entity_type: EntityType,
    time_scale_factor: float = 1.0
):
    """
    Add entity media (images or videos) to track at segment positions.

    This is the UNIFIED function that replaces:
    - _add_entity_images_to_track() (V9 - 203 lines)
    - _add_entity_videos_to_track() (V10 - 184 lines)

    Features:
    - Sticky entity: Last matched entity persists to fill subsequent segments
    - Semantic matching: Uses word overlap to find relevant entities
    - No duplicate sources: Same media won't appear twice in one segment
    - Polymorphic handling of images vs videos

    ALL media for an entity are placed as separate clips within
    the segment, divided equally by duration.

    Args:
        track: OTIO track to add clips to (V9 or V10)
        entity_data: Dict of entity_name -> EntityImageResult or EntityVideoResult
        matches: List of MatchResult objects
        frame_rate: Timeline frame rate
        config: Pipeline configuration
        entity_type: "images" or "videos" - determines media handling

    Format matches DaVinci Resolve's OTIO export:
    - media_references dict with DEFAULT_MEDIA key
    - available_range = 1 frame (still image) or actual duration (video)
    - source_range = display duration
    - active_media_reference_key = "DEFAULT_MEDIA"
    """
    rate = frame_rate
    is_image = (entity_type == "images")
    track_name = "V9" if is_image else "V10"

    # Build segment timing map: segment_index -> (start_frame, duration_frames, duration_sec, gap_before_frames)
    # Apply time_scale_factor to match V1-V8 track timing
    # IMPORTANT: Include gaps between segments to match voiceover timing (sync with V1-V8)
    segment_timing = {}
    timeline_frame = 0

    # Get first segment start time for reference (scaled)
    first_segment_start = matches[0].primary_match.voiceover_segment.start_time * time_scale_factor if matches else 0.0

    for i, match_result in enumerate(matches):
        match = match_result.primary_match
        vo_seg = match.voiceover_segment

        # Scale segment timing to match V1-V8 tracks
        scaled_start = vo_seg.start_time * time_scale_factor
        target_duration = (vo_seg.end_time - vo_seg.start_time) * time_scale_factor
        duration_frames = round(target_duration * frame_rate)

        # Calculate expected position (where this segment should start)
        expected_start_frames = round((scaled_start - first_segment_start) * frame_rate)

        # Calculate gap before this segment (silence in voiceover)
        gap_before_frames = max(0, expected_start_frames - timeline_frame)

        segment_timing[i] = (timeline_frame, duration_frames, target_duration, gap_before_frames)

        # Update timeline position (including gap + segment duration)
        timeline_frame = expected_start_frames + duration_frames

    # Track sticky entity across segments
    last_matched_entity = None

    # Track entity match statistics
    match_stats = {'exact': 0, 'semantic': 0, 'sticky': 0, 'none': 0}
    clips_added = 0

    # Get config values once (outside loop to avoid UnboundLocalError when matches is empty)
    enable_sticky = getattr(config.image_search, 'enable_sticky_matching', False)
    semantic_threshold = getattr(config.image_search, 'semantic_match_threshold', 0.15)

    # Process each segment
    for seg_idx, (start_frame, duration_frames, duration_sec, gap_before_frames) in segment_timing.items():
        match = matches[seg_idx].primary_match
        vo_text = match.voiceover_segment.text.lower()

        # Insert gap before this segment if there's a pause in voiceover
        # This keeps V9/V10 tracks in sync with V1-V8 tracks
        if gap_before_frames > 0:
            gap = otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, rate),
                    duration=otio.opentime.RationalTime(gap_before_frames, rate)
                )
            )
            track.append(gap)

        # Find best matching entity (exact -> semantic -> sticky)
        entity_name, match_type = _find_best_entity_match(
            vo_text, entity_data, last_matched_entity,
            enable_sticky=enable_sticky,
            semantic_threshold=semantic_threshold
        )

        match_stats[match_type] += 1

        if entity_name:
            entity_result = entity_data[entity_name]

            # Update sticky entity for next segments
            last_matched_entity = entity_name

            # Get all media for this entity (polymorphic)
            # Handle both dict (from checkpoint) and object formats
            if isinstance(entity_result, dict):
                all_media = entity_result.get('images' if is_image else 'videos', [])
            else:
                all_media = entity_result.images if is_image else entity_result.videos
            num_media = len(all_media)

            # Debug: Log which media are being used for this segment
            if seg_idx < 5:  # Only log first 5 for brevity
                media_label = "images" if is_image else "videos"
                logger.debug(f"{track_name} Segment {seg_idx}: Entity '{entity_name}' ({match_type}), {num_media} {media_label}")
                for media in all_media[:2]:
                    logger.debug(f"  - {Path(media).name}")

            if num_media > 0:
                # Track used sources within this segment to prevent duplicates
                used_sources = set()
                unique_media = []
                for media_path in all_media:
                    # Use filename as source identifier
                    source_id = Path(media_path).name
                    if source_id not in used_sources:
                        used_sources.add(source_id)
                        unique_media.append(media_path)

                # Use deduplicated media
                all_media = unique_media
                num_media = len(all_media)

                # Divide segment duration equally among all media
                frames_per_media = max(1, duration_frames // num_media)
                remaining_frames = duration_frames - (frames_per_media * num_media)

                # Create a clip for each media file
                for media_idx, media_path in enumerate(all_media):
                    # Calculate this media's duration (distribute remaining frames to last clips)
                    clip_frames = frames_per_media
                    if media_idx >= num_media - remaining_frames:
                        clip_frames += 1

                    # Get folder and filename for unique reference name
                    media_path_obj = Path(media_path)

                    # For images: Verify file exists and skip if not
                    if is_image and not media_path_obj.exists():
                        logger.warning(f"Image file not found, skipping: {media_path}")
                        continue

                    # Skip files with problematic unicode in path - they cause DaVinci to hang
                    if _has_problematic_path(str(media_path)):
                        logger.warning(f"Skipping media with problematic path (unicode issues): {media_path}")
                        continue

                    media_folder = media_path_obj.parent.name
                    media_filename = media_path_obj.name

                    # Include segment ID for post-edit analysis tracing
                    segment_id = f"[S{seg_idx:03d}]"
                    media_unique_name = f"{segment_id} {media_folder}_{media_filename}"

                    # Convert to Windows path format with backslashes for Resolve
                    media_path_resolved = _to_windows_path(media_path)

                    # Create external reference (polymorphic: images vs videos)
                    if is_image:
                        # OTIO TIMING MODEL: source_range MUST fit within available_range
                        # For still images, the single frame IS available for any duration
                        # (it's a freeze frame). Set available_range = clip duration so
                        # source_range <= available_range is satisfied.
                        available_frames = clip_frames
                    else:
                        # Videos have actual duration - get from ffprobe
                        available_frames = _get_video_duration_frames(media_path, rate)
                        if available_frames is None:
                            # Fallback: assume video is long enough
                            available_frames = clip_frames * 2

                    media_ref = otio.schema.ExternalReference(
                        target_url=media_path_resolved,
                        available_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(available_frames, rate)
                        )
                    )
                    media_ref.name = media_unique_name

                    # Create clip with source_range = display duration
                    # For stills: source_range.start_time MUST be 0 (cannot advance frames that don't exist)
                    # For videos: start from beginning, play for clip_frames duration
                    media_clip = otio.schema.Clip(
                        name=media_unique_name,
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(clip_frames, rate)
                        )
                    )

                    media_clip.media_reference = media_ref

                    # Add Resolve_OTIO metadata (required for DaVinci import)
                    media_clip.metadata['Resolve_OTIO'] = {}

                    # Add metadata (polymorphic)
                    media_clip.metadata['entity_name'] = entity_name
                    media_clip.metadata['entity_type'] = entity_result.entity_type
                    media_clip.metadata['query'] = entity_result.query
                    media_clip.metadata['segment_index'] = seg_idx
                    media_clip.metadata['match_type'] = match_type

                    if is_image:
                        media_clip.metadata['image_path'] = media_path
                        media_clip.metadata['image_index'] = media_idx
                        media_clip.metadata['total_images'] = num_media
                        media_clip.metadata['is_still_image'] = True
                    else:
                        media_clip.metadata['video_path'] = media_path
                        media_clip.metadata['video_index'] = media_idx
                        media_clip.metadata['total_videos'] = num_media
                        media_clip.metadata['source'] = 'stock_video'

                    track.append(media_clip)
                    clips_added += 1

                # Successfully added media, continue to next segment
                continue

        # No entity matched even with fallbacks - add gap to maintain sync
        gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, rate),
                duration=otio.opentime.RationalTime(duration_frames, rate)
            )
        )
        track.append(gap)

    # Log entity matching statistics
    total_segments = len(segment_timing)
    matched = match_stats['exact'] + match_stats['semantic'] + match_stats['sticky']
    media_label = "Entity matching" if is_image else "Stock video matching"
    print(f"  [{track_name}] {media_label}: {matched}/{total_segments} segments" + (f", {clips_added} clips added" if not is_image else ""))
    print(f"    Exact: {match_stats['exact']} ({100*match_stats['exact']/max(1,total_segments):.1f}%)")
    print(f"    Semantic: {match_stats['semantic']} ({100*match_stats['semantic']/max(1,total_segments):.1f}%)")
    if enable_sticky:
        print(f"    Sticky: {match_stats['sticky']} ({100*match_stats['sticky']/max(1,total_segments):.1f}%)")


# Public API for backward compatibility
def _add_entity_images_to_track(
    image_track: otio.schema.Track,
    entity_images: Dict,
    matches: List['MatchResult'],
    frame_rate: float,
    config: 'Config',
    time_scale_factor: float = 1.0
):
    """Add entity images to V9 track (backward compatible wrapper)."""
    add_entity_media_to_track(image_track, entity_images, matches, frame_rate, config, "images", time_scale_factor)


def _add_entity_videos_to_track(
    video_track: otio.schema.Track,
    entity_videos: Dict,
    matches: List['MatchResult'],
    frame_rate: float,
    config: 'Config',
    time_scale_factor: float = 1.0
):
    """Add stock videos to V10 track (backward compatible wrapper)."""
    add_entity_media_to_track(video_track, entity_videos, matches, frame_rate, config, "videos", time_scale_factor)
