"""
Tests for coarsen_segments() in src/stages/analyze.py.

Verifies the helper that merges every N consecutive SRT cues into a single
super-segment before CAPTION/MATCH/DOWNLOAD (--coarsen-factor CLI flag).
"""

import pytest

from src.stages.analyze import coarsen_segments
from src.state import VoiceoverSegment


def _seg(i, start, end, text):
    return VoiceoverSegment(index=i, start=start, end=end, text=text)


class TestCoarsenSegments:
    def setup_method(self):
        # 4 cues, each 1 second long, indexed 0..3
        self.segments = [
            _seg(0, 0.0, 1.0, "first"),
            _seg(1, 1.0, 2.0, "second"),
            _seg(2, 2.0, 3.0, "third"),
            _seg(3, 3.0, 4.0, "fourth"),
        ]

    def test_factor_1_is_noop(self):
        out = coarsen_segments(self.segments, 1)
        assert out == self.segments

    def test_factor_2_merges_pairs(self):
        out = coarsen_segments(self.segments, 2)
        assert len(out) == 2
        assert out[0].index == 0
        assert out[0].start == 0.0
        assert out[0].end == 2.0
        assert out[0].text == "first second"
        assert out[1].index == 2
        assert out[1].start == 2.0
        assert out[1].end == 4.0
        assert out[1].text == "third fourth"

    def test_factor_3_with_remainder(self):
        # 4 segments, factor 3 → one group of 3 + leftover 1
        out = coarsen_segments(self.segments, 3)
        assert len(out) == 2
        assert out[0].start == 0.0
        assert out[0].end == 3.0
        assert out[0].text == "first second third"
        # leftover keeps its own start/end/text
        assert out[1].start == 3.0
        assert out[1].end == 4.0
        assert out[1].text == "fourth"

    def test_factor_larger_than_length(self):
        out = coarsen_segments(self.segments, 10)
        assert len(out) == 1
        assert out[0].start == 0.0
        assert out[0].end == 4.0
        assert out[0].text == "first second third fourth"

    def test_zero_and_negative_factors_are_noop(self):
        for bad in (0, -1, -100):
            assert coarsen_segments(self.segments, bad) == self.segments

    def test_empty_input_returns_empty(self):
        assert coarsen_segments([], 2) == []

    def test_single_segment_returns_same(self):
        one = [_seg(0, 0.0, 1.0, "only")]
        out = coarsen_segments(one, 2)
        assert len(out) == 1
        assert out[0].text == "only"

    def test_indices_remain_contiguous(self):
        # downstream lookups index by .index; verify 0..N-1 across all factors
        for factor in (1, 2, 3, 4, 5):
            out = coarsen_segments(self.segments, factor)
            for i, seg in enumerate(out):
                assert seg.index == self.segments[i * factor].index, \
                    f"factor={factor} index mismatch at out[{i}]"
            # and the last index equals the last group's first-cue index
            last_group_start = (len(out) - 1) * factor
            assert out[-1].index == self.segments[last_group_start].index

    def test_duration_field_recomputed(self):
        out = coarsen_segments(self.segments, 2)
        assert out[0].duration == pytest.approx(2.0)
        assert out[1].duration == pytest.approx(2.0)

    def test_text_concatenation_uses_single_spaces(self):
        segs = [
            _seg(0, 0.0, 1.0, "alpha"),
            _seg(1, 1.0, 2.0, "beta"),
            _seg(2, 2.0, 3.0, "gamma"),
        ]
        out = coarsen_segments(segs, 3)
        assert out[0].text == "alpha beta gamma"