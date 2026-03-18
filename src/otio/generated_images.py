"""Generated images track population for V12."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import opentimelineio as otio

from .utils import _to_windows_path, _get_media_duration

logger = logging.getLogger(__name__)


def _add_generated_images_to_track(
    track: otio.schema.Track,
    generated_images: List,
    matches: List,
    frame_rate: float,
    time_scale_factor: float = 1.0,
    voiceover_offset: float = 0.0,
) -> None:
    """Add generated image clips to V12 track at batch time positions.

    Each GeneratedImageResult has start_time/end_time derived from the
    voiceover segments it covers. Creates still-image clips with gaps
    between them.

    Args:
        track: The V12 OTIO track to populate.
        generated_images: List of GeneratedImageResult objects.
        matches: List of match results (for timeline context).
        frame_rate: Timeline frame rate.
        time_scale_factor: Speed adjustment factor.
        voiceover_offset: Offset for voiceover alignment.
    """
    if not generated_images:
        return

    # Sort by start_time to ensure correct gap calculation
    sorted_images = sorted(generated_images, key=lambda r: r.start_time)
    current_time = 0.0
    clips_added = 0

    for result in sorted_images:
        file_path = result.file
        if not file_path or not Path(file_path).exists():
            logger.debug(f"Skipping generated image {result.batch_id}: file not found ({file_path})")
            continue

        start = (result.start_time + voiceover_offset) * time_scale_factor
        end = (result.end_time + voiceover_offset) * time_scale_factor
        duration = end - start

        if duration <= 0:
            logger.debug(f"Skipping generated image {result.batch_id}: zero/negative duration")
            continue

        # Insert gap if needed
        gap_duration = start - current_time
        if gap_duration > 0.001:  # Small tolerance for floating point
            gap = otio.schema.Gap(
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0, frame_rate),
                    duration=otio.opentime.RationalTime(gap_duration * frame_rate, frame_rate),
                )
            )
            track.append(gap)

        # Create clip for the generated image (still image = full duration)
        media_ref = otio.schema.ExternalReference(
            target_url=_to_windows_path(file_path),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, frame_rate),
                duration=otio.opentime.RationalTime(duration * frame_rate, frame_rate),
            ),
        )

        clip = otio.schema.Clip(
            name=f"Generated: {result.batch_id}",
            media_reference=media_ref,
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, frame_rate),
                duration=otio.opentime.RationalTime(duration * frame_rate, frame_rate),
            ),
        )

        # Add metadata
        clip.metadata['generated_image'] = {
            'batch_id': result.batch_id,
            'prompt': getattr(result, 'prompt', '')[:200],
            'width': getattr(result, 'width', 0),
            'height': getattr(result, 'height', 0),
            'segment_range': f"{result.segment_start_index}-{result.segment_end_index}",
        }

        track.append(clip)
        current_time = end
        clips_added += 1

    if clips_added:
        logger.info(f"[V12] Added {clips_added} generated image clips to track")
    else:
        logger.debug("[V12] No generated image clips added (no valid files)")
