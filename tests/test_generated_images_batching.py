"""Tests for generated image batching."""

from types import SimpleNamespace

import pytest

from src.generated_images.batching import (
    build_generated_image_batches,
    plan_generated_image_batch_sizes,
)
from src.state import VoiceoverSegment


def make_batch_config():
    return SimpleNamespace(
        batch_target_segments=8,
        min_segments_per_image=7,
        max_segments_per_image=9,
        allow_boundary_batch_outside_range=True,
        boundary_min_segments=6,
        boundary_max_segments=10,
        fallback_single_batch_max_segments=11,
    )


class TestGeneratedImageBatchPlanning:
    @pytest.mark.fast
    @pytest.mark.parametrize(
        ("total_segments", "expected"),
        [
            (6, [6]),
            (16, [8, 8]),
            (17, [8, 9]),
            (18, [9, 9]),
            (19, [9, 10]),
            (20, [10, 10]),
        ],
    )
    def test_plan_batch_sizes_examples(self, total_segments, expected):
        config = make_batch_config()
        assert plan_generated_image_batch_sizes(total_segments, config) == expected

    @pytest.mark.fast
    def test_plan_batch_sizes_uses_single_batch_fallback_for_eleven(self):
        config = make_batch_config()
        assert plan_generated_image_batch_sizes(11, config) == [11]

    @pytest.mark.fast
    def test_plan_batch_sizes_strict_mode_rejects_under_minimum_batch(self):
        config = make_batch_config()
        config.allow_boundary_batch_outside_range = False

        with pytest.raises(ValueError, match="Unable to batch subtitle segments"):
            plan_generated_image_batch_sizes(6, config)

    @pytest.mark.fast
    def test_build_batches_preserves_timing_and_text(self):
        config = make_batch_config()
        segments = [
            VoiceoverSegment(
                index=index,
                start=float(index * 2),
                end=float(index * 2 + 1),
                text=f"Segment {index}",
            )
            for index in range(8)
        ]

        batches = build_generated_image_batches(segments, config)

        assert len(batches) == 1
        assert batches[0].segment_start_index == 0
        assert batches[0].segment_end_index == 7
        assert batches[0].start_time == 0.0
        assert batches[0].end_time == 15.0
        assert "Segment 0" in batches[0].text
        assert "Segment 7" in batches[0].text
        assert batches[0].segment_indices == list(range(8))
