"""Batch subtitle segments for generated still images."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

from ..state import GeneratedImageBatch, VoiceoverSegment


def _score_partition(partition: Sequence[int], target: int) -> Tuple[int, int, int]:
    """Prefer batches close to target, then fewer batches, then smoother spread."""

    distances = [abs(size - target) for size in partition]
    return (
        sum(distances),
        len(partition),
        max(distances) if distances else 0,
    )


def _find_best_partition(total_segments: int, allowed_sizes: Iterable[int], target: int) -> Optional[List[int]]:
    allowed = tuple(sorted(set(size for size in allowed_sizes if size > 0)))
    if not allowed:
        return None

    search_order = tuple(sorted(allowed, key=lambda size: (abs(size - target), size)))

    @lru_cache(maxsize=None)
    def solve(remaining: int) -> Optional[Tuple[int, ...]]:
        if remaining == 0:
            return ()

        best: Optional[Tuple[int, ...]] = None
        best_score: Optional[Tuple[int, int, int]] = None

        for size in search_order:
            if size > remaining:
                continue
            rest = solve(remaining - size)
            if rest is None:
                continue

            candidate = (size,) + rest
            candidate_score = _score_partition(candidate, target)
            if best is None or candidate_score < best_score:
                best = candidate
                best_score = candidate_score

        return best

    result = solve(total_segments)
    return list(result) if result is not None else None


def plan_generated_image_batch_sizes(total_segments: int, batch_config: object) -> List[int]:
    """Plan segment counts per generated image batch."""

    if total_segments <= 0:
        return []

    min_segments = int(getattr(batch_config, 'min_segments_per_image', 7))
    max_segments = int(getattr(batch_config, 'max_segments_per_image', 9))
    target_segments = int(getattr(batch_config, 'batch_target_segments', 8))
    allow_boundary = bool(
        getattr(batch_config, 'allow_boundary_batch_outside_range', True)
    )
    boundary_min = int(getattr(batch_config, 'boundary_min_segments', 6))
    boundary_max = int(getattr(batch_config, 'boundary_max_segments', 10))
    fallback_single_max = int(
        getattr(batch_config, 'fallback_single_batch_max_segments', boundary_max)
    )

    preferred_partition = _find_best_partition(
        total_segments,
        range(min_segments, max_segments + 1),
        target_segments,
    )
    if preferred_partition is not None:
        return preferred_partition

    if allow_boundary:
        boundary_partition = _find_best_partition(
            total_segments,
            range(boundary_min, boundary_max + 1),
            target_segments,
        )
        if boundary_partition is not None:
            return boundary_partition

        if total_segments <= fallback_single_max:
            return [total_segments]

    raise ValueError(
        "Unable to batch subtitle segments for generated images with the current "
        f"configuration (total_segments={total_segments})"
    )


def build_generated_image_batches(
    segments: Sequence[VoiceoverSegment],
    batch_config: object,
) -> List[GeneratedImageBatch]:
    """Create time-anchored generated image batches from subtitle segments."""

    batch_sizes = plan_generated_image_batch_sizes(len(segments), batch_config)
    batches: List[GeneratedImageBatch] = []
    offset = 0

    for batch_index, batch_size in enumerate(batch_sizes):
        batch_segments = list(segments[offset:offset + batch_size])
        if not batch_segments:
            continue

        first_segment = batch_segments[0]
        last_segment = batch_segments[-1]
        merged_text = " ".join(
            segment.text.strip() for segment in batch_segments if segment.text.strip()
        ).strip()

        batches.append(
            GeneratedImageBatch(
                batch_id=f"generated_{batch_index:03d}",
                segment_start_index=first_segment.index,
                segment_end_index=last_segment.index,
                segment_count=len(batch_segments),
                start_time=first_segment.start,
                end_time=last_segment.end,
                text=merged_text,
                segment_indices=[segment.index for segment in batch_segments],
            )
        )
        offset += batch_size

    return batches
