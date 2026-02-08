"""Tests for caption segment deduplication (US-73-007)."""
import logging
import pytest

from src.caption.models import (
    CaptionSegment,
    CaptionResult,
    deduplicate_caption_segments,
    _token_overlap,
    _temporal_overlap_ratio,
    _text_quality_score,
)


def _seg(index, start, end, text, source="vid1"):
    """Helper to create a CaptionSegment."""
    return CaptionSegment(index=index, start_time=start, end_time=end, text=text, source_file=source)


class TestTokenOverlap:
    def test_identical_text(self):
        assert _token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _token_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        result = _token_overlap("the quick brown fox", "the slow brown dog")
        # shared: {the, brown} = 2, union: {the, quick, brown, fox, slow, dog} = 6
        assert abs(result - 2 / 6) < 0.01

    def test_empty_strings(self):
        assert _token_overlap("", "") == 1.0
        assert _token_overlap("hello", "") == 0.0


class TestTemporalOverlap:
    def test_full_overlap(self):
        a = _seg(0, 0.0, 5.0, "a")
        b = _seg(1, 0.0, 5.0, "b")
        assert _temporal_overlap_ratio(a, b) == 1.0

    def test_no_overlap(self):
        a = _seg(0, 0.0, 5.0, "a")
        b = _seg(1, 6.0, 10.0, "b")
        assert _temporal_overlap_ratio(a, b) == 0.0

    def test_partial_overlap(self):
        a = _seg(0, 0.0, 10.0, "a")
        b = _seg(1, 8.0, 12.0, "b")
        # overlap = 10.0-8.0 = 2.0, shorter duration = min(10,4) = 4.0
        assert abs(_temporal_overlap_ratio(a, b) - 0.5) < 0.01

    def test_contained_segment(self):
        a = _seg(0, 0.0, 10.0, "a")
        b = _seg(1, 2.0, 4.0, "b")
        # overlap = 2.0, shorter = 2.0 => 1.0
        assert _temporal_overlap_ratio(a, b) == 1.0


class TestTextQuality:
    def test_longer_unique_text_scores_higher(self):
        short = _text_quality_score("hi")
        long = _text_quality_score("hello world this is a longer sentence")
        assert long > short

    def test_repeated_chars_score_lower(self):
        diverse = _text_quality_score("abcdefgh")
        repeated = _text_quality_score("aaaaaaaa")
        assert diverse > repeated

    def test_empty(self):
        assert _text_quality_score("") == 0.0


class TestDeduplicateCaptionSegments:
    def test_no_duplicates_pass_through(self):
        """Non-overlapping segments should pass through unchanged."""
        segs = [
            _seg(0, 0.0, 5.0, "First segment"),
            _seg(1, 5.0, 10.0, "Second segment"),
            _seg(2, 10.0, 15.0, "Third segment"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 3

    def test_identical_duplicates_merged(self):
        """Fully overlapping segments with identical text should merge."""
        segs = [
            _seg(0, 0.0, 5.0, "Hello world this is a test"),
            _seg(1, 0.0, 5.0, "Hello world this is a test"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 1

    def test_high_overlap_similar_text_merged(self):
        """Segments with >80% temporal overlap and >70% text similarity should merge."""
        # Same core text, but seg 1 is slightly truncated (auto-gen artifact)
        segs = [
            _seg(0, 0.0, 10.0, "the quick brown fox jumps over lazy dog"),
            _seg(1, 0.5, 10.0, "the quick brown fox jumps over the lazy dog"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 1

    def test_high_overlap_different_text_kept(self):
        """Segments with high temporal overlap but different text should be kept."""
        segs = [
            _seg(0, 0.0, 10.0, "completely different content here"),
            _seg(1, 0.5, 10.5, "nothing similar at all in this one"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 2

    def test_similar_text_no_overlap_kept(self):
        """Segments with similar text but no temporal overlap should be kept."""
        segs = [
            _seg(0, 0.0, 5.0, "the quick brown fox jumps"),
            _seg(1, 50.0, 55.0, "the quick brown fox jumps"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 2

    def test_keeps_higher_quality_text(self):
        """When merging, should keep the segment with higher quality text."""
        # shared: {alpha, bravo, charlie} = 3, union: 4 => Jaccard = 0.75 ✓
        segs = [
            _seg(0, 0.0, 5.0, "alpha bravo charlie"),  # Shorter
            _seg(1, 0.0, 5.0, "alpha bravo charlie delta"),  # Longer, higher quality
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 1
        assert "delta" in result[0].text

    def test_reindexes_after_dedup(self):
        """Remaining segments should have sequential indices after dedup."""
        segs = [
            _seg(0, 0.0, 5.0, "hello world test"),
            _seg(1, 0.0, 5.0, "hello world test"),
            _seg(2, 10.0, 15.0, "different segment"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 2
        assert result[0].index == 0
        assert result[1].index == 1

    def test_empty_list(self):
        assert deduplicate_caption_segments([]) == []

    def test_single_segment(self):
        segs = [_seg(0, 0.0, 5.0, "solo")]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 1

    def test_multiple_duplicates_in_chain(self):
        """Multiple overlapping duplicates should all be reduced."""
        segs = [
            _seg(0, 0.0, 5.0, "the same text repeated exactly"),
            _seg(1, 0.1, 5.1, "the same text repeated exactly"),
            _seg(2, 0.2, 5.2, "the same text repeated exactly"),
        ]
        result = deduplicate_caption_segments(segs)
        assert len(result) == 1

    def test_debug_logging(self, caplog):
        """Dedup should log at DEBUG level when merging."""
        segs = [
            _seg(0, 0.0, 5.0, "hello world test text"),
            _seg(1, 0.0, 5.0, "hello world test text"),
        ]
        with caplog.at_level(logging.DEBUG, logger="src.caption.models"):
            deduplicate_caption_segments(segs)
        assert any("Dedup:" in msg for msg in caplog.messages)


class TestCaptionResultDedup:
    def test_post_init_deduplicates(self):
        """CaptionResult.__post_init__ should run deduplication on segments."""
        segs = [
            _seg(0, 0.0, 5.0, "the quick brown fox jumps over"),
            _seg(1, 0.0, 5.0, "the quick brown fox jumps over"),
            _seg(2, 10.0, 15.0, "a different segment here"),
        ]
        result = CaptionResult(video_id="test123", segments=segs)
        assert len(result.segments) == 2

    def test_post_init_no_segments(self):
        """CaptionResult with no segments should not error."""
        result = CaptionResult(video_id="test123")
        assert len(result.segments) == 0

    def test_post_init_single_segment(self):
        """CaptionResult with one segment should pass through."""
        segs = [_seg(0, 0.0, 5.0, "solo")]
        result = CaptionResult(video_id="test123", segments=segs)
        assert len(result.segments) == 1
