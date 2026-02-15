"""
Tests for the listicle-to-chapter bridge module (US-71-010).

Verifies:
- Conversion of ListicleGroup objects to ChapterCandidate objects
- topic_keywords populated from ListicleGroup
- Merge behavior with YouTube chapters taking precedence for overlapping ranges
- Non-overlapping listicle chapters included in merge
- Unified chapter list accessible via same interface as regular chapters
"""

import pytest
from src.chapter_detection.models import ChapterCandidate, ListicleGroup
from src.chapter_detection.bridge import (
    listicle_groups_to_chapters,
    merge_chapters,
    build_unified_chapters,
    build_segment_chapter_map,
    assign_chapter_indices,
    compute_relevance_matrix,
    compute_chapter_alignment_scores,
    _ranges_overlap,
)
from src.state import TranscriptSegment


# --- Helpers ---

def _make_listicle_group(group_id, start, end, label="item", keywords=None, expected_count=None, marker_type="ordinal", confidence=0.7):
    return ListicleGroup(
        group_id=group_id,
        item_label=label,
        marker_type=marker_type,
        start_segment_idx=start,
        end_segment_idx=end,
        topic_keywords=keywords or [],
        expected_count=expected_count,
        confidence=confidence,
    )


def _make_chapter(chapter_id, start, end, title="Chapter", topics=None, strategy="topic", confidence=0.8):
    return ChapterCandidate(
        chapter_id=chapter_id,
        start_segment_idx=start,
        end_segment_idx=end,
        title=title,
        topics=topics or [],
        detection_strategy=strategy,
        confidence=confidence,
    )


# --- listicle_groups_to_chapters ---

class TestListicleGroupsToChapters:
    def test_empty_input(self):
        result = listicle_groups_to_chapters([])
        assert result == []

    def test_single_group_converts_to_chapter(self):
        group = _make_listicle_group(0, 0, 4, label="first", keywords=["travel", "paris"])
        chapters = listicle_groups_to_chapters([group])
        assert len(chapters) == 1
        ch = chapters[0]
        assert isinstance(ch, ChapterCandidate)
        assert ch.start_segment_idx == 0
        assert ch.end_segment_idx == 4
        assert ch.detection_strategy == 'listicle'

    def test_topic_keywords_populated(self):
        """Criterion: Converted chapters have topic_keywords populated from ListicleGroup."""
        group = _make_listicle_group(0, 0, 3, keywords=["cooking", "recipes", "pasta"])
        chapters = listicle_groups_to_chapters([group])
        assert chapters[0].topics == ["cooking", "recipes", "pasta"]

    def test_topic_keywords_empty_when_group_has_none(self):
        group = _make_listicle_group(0, 0, 2, keywords=[])
        chapters = listicle_groups_to_chapters([group])
        assert chapters[0].topics == []

    def test_multiple_groups_convert(self):
        groups = [
            _make_listicle_group(0, 0, 4, label="first", keywords=["intro"]),
            _make_listicle_group(1, 5, 9, label="second", keywords=["middle"]),
            _make_listicle_group(2, 10, 14, label="third", keywords=["end"]),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert len(chapters) == 3
        assert all(ch.detection_strategy == 'listicle' for ch in chapters)
        assert chapters[0].topics == ["intro"]
        assert chapters[1].topics == ["middle"]
        assert chapters[2].topics == ["end"]

    def test_title_derived_from_label_and_keywords(self):
        group = _make_listicle_group(0, 0, 3, label="first", keywords=["travel", "paris", "food"])
        chapters = listicle_groups_to_chapters([group])
        assert "first" in chapters[0].title
        assert "travel" in chapters[0].title

    def test_title_fallback_when_no_label_or_keywords(self):
        group = _make_listicle_group(0, 0, 2, label="", keywords=[])
        chapters = listicle_groups_to_chapters([group])
        assert "Item 1" in chapters[0].title

    def test_confidence_with_matching_expected_count(self):
        """When expected_count matches group count, confidence is 0.85."""
        group = _make_listicle_group(0, 0, 3, expected_count=1, marker_type="transition")
        chapters = listicle_groups_to_chapters([group])
        assert chapters[0].confidence == 0.85

    def test_confidence_without_expected_count(self):
        group = _make_listicle_group(0, 0, 3, expected_count=None, marker_type="ordinal")
        chapters = listicle_groups_to_chapters([group])
        assert chapters[0].confidence == 0.7

    def test_chapter_id_preserved_from_group_id(self):
        groups = [
            _make_listicle_group(0, 0, 4),
            _make_listicle_group(1, 5, 9),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert chapters[0].chapter_id == 0
        assert chapters[1].chapter_id == 1


# --- Graduated confidence by marker density (US-76-003) ---

class TestGraduatedConfidence:
    """Tests for graduated bridge confidence by marker type."""

    def test_transition_only_groups_get_0_6(self):
        """Transition-only markers get confidence 0.6."""
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="transition"),
            _make_listicle_group(1, 5, 9, marker_type="transition"),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert chapters[0].confidence == 0.6
        assert chapters[1].confidence == 0.6

    def test_ordinal_groups_get_0_7(self):
        """Ordinal markers get confidence 0.7."""
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="ordinal"),
            _make_listicle_group(1, 5, 9, marker_type="ordinal"),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert chapters[0].confidence == 0.7
        assert chapters[1].confidence == 0.7

    def test_numbered_groups_get_0_8(self):
        """Numbered markers get confidence 0.8."""
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="numbered"),
            _make_listicle_group(1, 5, 9, marker_type="numbered"),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert chapters[0].confidence == 0.8
        assert chapters[1].confidence == 0.8

    def test_matching_expected_count_gives_0_85(self):
        """Groups with matching expected_count get 0.85 regardless of marker type."""
        # 3 groups, expected_count=3 -> match
        groups = [
            _make_listicle_group(0, 0, 3, marker_type="transition", expected_count=3),
            _make_listicle_group(1, 4, 7, marker_type="transition", expected_count=3),
            _make_listicle_group(2, 8, 11, marker_type="transition", expected_count=3),
        ]
        chapters = listicle_groups_to_chapters(groups)
        for ch in chapters:
            assert ch.confidence == 0.85

    def test_matching_expected_count_overrides_numbered(self):
        """expected_count match overrides even numbered (0.8) to 0.85."""
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="numbered", expected_count=2),
            _make_listicle_group(1, 5, 9, marker_type="numbered", expected_count=2),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert chapters[0].confidence == 0.85
        assert chapters[1].confidence == 0.85

    def test_mismatched_expected_count_uses_marker_type(self):
        """When expected_count doesn't match group count, fall back to marker type."""
        # 2 groups, expected_count=5 -> mismatch
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="numbered", expected_count=5),
            _make_listicle_group(1, 5, 9, marker_type="numbered", expected_count=5),
        ]
        chapters = listicle_groups_to_chapters(groups)
        assert chapters[0].confidence == 0.8
        assert chapters[1].confidence == 0.8

    def test_explicit_group_confidence_used(self):
        """When group has explicit confidence (not default 0.7), it is used as base."""
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="transition", confidence=0.9),
            _make_listicle_group(1, 5, 9, marker_type="transition", confidence=0.9),
        ]
        chapters = listicle_groups_to_chapters(groups)
        # Explicit confidence 0.9 should be used instead of marker type 0.6
        assert chapters[0].confidence == 0.9
        assert chapters[1].confidence == 0.9

    def test_explicit_confidence_with_expected_count_boost(self):
        """Expected_count boost applies on top of explicit confidence."""
        groups = [
            _make_listicle_group(0, 0, 4, marker_type="numbered", confidence=0.75, expected_count=2),
            _make_listicle_group(1, 5, 9, marker_type="numbered", confidence=0.75, expected_count=2),
        ]
        chapters = listicle_groups_to_chapters(groups)
        # Expected count match boosts to 0.85, which is higher than explicit 0.75
        assert chapters[0].confidence == 0.85
        assert chapters[1].confidence == 0.85


# --- _ranges_overlap ---

class TestRangesOverlap:
    def test_no_overlap(self):
        assert not _ranges_overlap(0, 4, 5, 9)

    def test_touching_overlap(self):
        # [0,5] and [5,9] share segment 5
        assert _ranges_overlap(0, 5, 5, 9)

    def test_full_overlap(self):
        assert _ranges_overlap(0, 9, 2, 7)

    def test_partial_overlap(self):
        assert _ranges_overlap(0, 6, 4, 9)

    def test_identical_ranges(self):
        assert _ranges_overlap(3, 7, 3, 7)


# --- merge_chapters ---

class TestMergeChapters:
    def test_both_empty(self):
        assert merge_chapters([], []) == []

    def test_only_youtube(self):
        yt = [_make_chapter(0, 0, 4, title="YT1")]
        result = merge_chapters(yt, [])
        assert len(result) == 1
        assert result[0].title == "YT1"

    def test_only_listicle(self):
        lc = [_make_chapter(0, 0, 4, title="LC1", strategy="listicle")]
        result = merge_chapters([], lc)
        assert len(result) == 1
        assert result[0].title == "LC1"

    def test_youtube_takes_precedence_for_overlapping(self):
        """Criterion: YouTube chapters take precedence for overlapping ranges."""
        yt = [_make_chapter(0, 0, 9, title="YouTube Chapter")]
        lc = [_make_chapter(0, 3, 7, title="Listicle Item", strategy="listicle")]
        result = merge_chapters(yt, lc)
        # Listicle overlaps with YouTube, so only YouTube should remain
        assert len(result) == 1
        assert result[0].title == "YouTube Chapter"

    def test_non_overlapping_listicle_included(self):
        """Non-overlapping listicle chapters fill gaps."""
        yt = [_make_chapter(0, 0, 4, title="YT1")]
        lc = [_make_chapter(0, 10, 14, title="LC1", strategy="listicle")]
        result = merge_chapters(yt, lc)
        assert len(result) == 2
        assert result[0].title == "YT1"
        assert result[1].title == "LC1"

    def test_mixed_overlap_and_non_overlap(self):
        """YouTube overlaps with some listicle items, others fill gaps."""
        yt = [_make_chapter(0, 0, 9, title="YT1")]
        lc = [
            _make_chapter(0, 3, 7, title="LC-overlap", strategy="listicle"),
            _make_chapter(1, 15, 19, title="LC-gap", strategy="listicle"),
        ]
        result = merge_chapters(yt, lc)
        assert len(result) == 2
        titles = [ch.title for ch in result]
        assert "YT1" in titles
        assert "LC-gap" in titles
        assert "LC-overlap" not in titles

    def test_chapter_ids_reassigned_sequentially(self):
        yt = [_make_chapter(5, 0, 4, title="YT")]
        lc = [_make_chapter(8, 10, 14, title="LC", strategy="listicle")]
        result = merge_chapters(yt, lc)
        assert result[0].chapter_id == 0
        assert result[1].chapter_id == 1

    def test_sorted_by_start_segment_idx(self):
        yt = [_make_chapter(0, 10, 14, title="YT-later")]
        lc = [_make_chapter(0, 0, 4, title="LC-earlier", strategy="listicle")]
        result = merge_chapters(yt, lc)
        assert result[0].title == "LC-earlier"
        assert result[1].title == "YT-later"


# --- build_unified_chapters ---

class TestBuildUnifiedChapters:
    def test_both_empty(self):
        assert build_unified_chapters([], []) == []

    def test_only_listicle_groups(self):
        groups = [
            _make_listicle_group(0, 0, 4, keywords=["topic_a"]),
            _make_listicle_group(1, 5, 9, keywords=["topic_b"]),
        ]
        result = build_unified_chapters([], groups)
        assert len(result) == 2
        assert all(ch.detection_strategy == 'listicle' for ch in result)
        assert result[0].topics == ["topic_a"]
        assert result[1].topics == ["topic_b"]

    def test_only_youtube_chapters(self):
        yt = [_make_chapter(0, 0, 9, title="YT Only", topics=["travel"])]
        result = build_unified_chapters(yt, [])
        assert len(result) == 1
        assert result[0].title == "YT Only"

    def test_merge_with_overlap(self):
        yt = [_make_chapter(0, 0, 9, title="YouTube")]
        groups = [_make_listicle_group(0, 3, 7, keywords=["overlapping"])]
        result = build_unified_chapters(yt, groups)
        # Overlapping listicle excluded
        assert len(result) == 1
        assert result[0].title == "YouTube"

    def test_merge_with_gap_filling(self):
        yt = [_make_chapter(0, 0, 4, title="YouTube")]
        groups = [_make_listicle_group(0, 10, 14, keywords=["gap_fill"])]
        result = build_unified_chapters(yt, groups)
        assert len(result) == 2
        assert result[1].topics == ["gap_fill"]

    def test_unified_interface_same_as_regular_chapters(self):
        """Criterion: unified list accessible via same interface as regular chapters."""
        yt = [_make_chapter(0, 0, 4, topics=["yt_topic"])]
        groups = [_make_listicle_group(0, 10, 14, keywords=["listicle_topic"])]
        result = build_unified_chapters(yt, groups)
        # All items are ChapterCandidate -- same interface
        for ch in result:
            assert isinstance(ch, ChapterCandidate)
            assert hasattr(ch, 'topics')
            assert hasattr(ch, 'start_segment_idx')
            assert hasattr(ch, 'end_segment_idx')
            assert hasattr(ch, 'chapter_id')
            assert hasattr(ch, 'detection_strategy')
            assert hasattr(ch, 'confidence')
            assert hasattr(ch, 'segment_range')
            assert hasattr(ch, 'segment_count')


# --- build_segment_chapter_map ---

class TestBuildSegmentChapterMap:
    def test_empty_chapters(self):
        assert build_segment_chapter_map([]) == {}

    def test_single_chapter(self):
        ch = _make_chapter(0, 0, 2)
        result = build_segment_chapter_map([ch])
        assert result == {0: 0, 1: 0, 2: 0}

    def test_multiple_chapters(self):
        chapters = [_make_chapter(0, 0, 2), _make_chapter(1, 5, 7)]
        result = build_segment_chapter_map(chapters)
        assert result[0] == 0
        assert result[2] == 0
        assert result[5] == 1
        assert result[7] == 1
        assert 3 not in result  # Gap not mapped


# --- assign_chapter_indices (US-72-003) ---

def _make_segment(index, start_time, end_time, text="test"):
    return TranscriptSegment(index=index, start_time=start_time, end_time=end_time, text=text)


class TestAssignChapterIndices:
    """Tests for timestamp-based segment-to-chapter mapping (US-72-003)."""

    def test_segment_fully_inside_chapter(self):
        """Segments fully inside a chapter are assigned to it."""
        segments = [_make_segment(0, 5.0, 10.0)]
        chapters = [{'title': 'Intro', 'start_time': 0.0, 'end_time': 30.0}]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index == 0
        assert segments[0].chapter_title == 'Intro'

    def test_segment_spanning_two_chapters_majority_overlap(self):
        """Segment spanning two chapters assigned to chapter with >50% overlap."""
        # Segment: 8.0-12.0 (4s total)
        # Chapter 0: 0-10 -> overlap = 2s (50%)
        # Chapter 1: 10-20 -> overlap = 2s (50%)
        # Tie goes to first found (chapter 0) since overlap is equal
        # Make asymmetric: segment 7.0-12.0 (5s total)
        # Chapter 0: 0-10 -> overlap = 3s (60%)
        # Chapter 1: 10-20 -> overlap = 2s (40%)
        segments = [_make_segment(0, 7.0, 12.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index == 0
        assert segments[0].chapter_title == 'Chapter A'

    def test_segment_spanning_two_chapters_second_wins(self):
        """Segment with more overlap in second chapter assigned there."""
        # Segment: 9.0-15.0 (6s total)
        # Chapter 0: 0-10 -> overlap = 1s
        # Chapter 1: 10-20 -> overlap = 5s
        segments = [_make_segment(0, 9.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index == 1
        assert segments[0].chapter_title == 'Chapter B'

    def test_segment_outside_all_chapters(self):
        """Segments outside all chapter ranges get None."""
        segments = [_make_segment(0, 50.0, 55.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index is None
        assert segments[0].chapter_title == ''

    def test_empty_chapters_list(self):
        """No chapters -> segments unchanged."""
        segments = [_make_segment(0, 5.0, 10.0)]
        assign_chapter_indices(segments, [])
        assert segments[0].chapter_index is None
        assert segments[0].chapter_title == ''

    def test_empty_segments_list(self):
        """Empty segments list doesn't raise."""
        chapters = [{'title': 'Ch', 'start_time': 0.0, 'end_time': 10.0}]
        assign_chapter_indices([], chapters)  # Should not raise

    def test_multiple_segments_different_chapters(self):
        """Multiple segments map to their respective chapters."""
        segments = [
            _make_segment(0, 1.0, 5.0),   # In chapter 0
            _make_segment(1, 12.0, 18.0),  # In chapter 1
            _make_segment(2, 25.0, 28.0),  # In chapter 2
        ]
        chapters = [
            {'title': 'Intro', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Main', 'start_time': 10.0, 'end_time': 20.0},
            {'title': 'Outro', 'start_time': 20.0, 'end_time': 30.0},
        ]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index == 0
        assert segments[0].chapter_title == 'Intro'
        assert segments[1].chapter_index == 1
        assert segments[1].chapter_title == 'Main'
        assert segments[2].chapter_index == 2
        assert segments[2].chapter_title == 'Outro'

    def test_segment_in_gap_between_chapters(self):
        """Segment in gap between non-contiguous chapters gets None."""
        segments = [_make_segment(0, 15.0, 18.0)]
        chapters = [
            {'title': 'Ch A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Ch B', 'start_time': 20.0, 'end_time': 30.0},
        ]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index is None
        assert segments[0].chapter_title == ''

    def test_chapter_missing_title_defaults_empty(self):
        """Chapter dict without title key gives empty chapter_title."""
        segments = [_make_segment(0, 5.0, 8.0)]
        chapters = [{'start_time': 0.0, 'end_time': 10.0}]
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index == 0
        assert segments[0].chapter_title == ''


# --- assign_chapter_indices strategy tests (US-105-009) ---

class TestAssignChapterIndicesStrategy:
    """Tests for multi-chapter assignment strategies (US-105-009)."""

    def test_strategy_first_assigns_to_first_chapter(self):
        """Strategy 'first' assigns segment to first overlapping chapter."""
        # Segment spans 8-15, overlaps with both chapters
        # Chapter 0: 0-10 -> overlap 2s
        # Chapter 1: 10-20 -> overlap 5s
        # 'first' should assign to chapter 0
        segments = [_make_segment(0, 8.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters, strategy='first')
        assert segments[0].chapter_index == 0
        assert segments[0].chapter_title == 'Chapter A'

    def test_strategy_first_no_overlap_gives_none(self):
        """Strategy 'first' gives None when no overlap."""
        segments = [_make_segment(0, 50.0, 55.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters, strategy='first')
        assert segments[0].chapter_index is None
        assert segments[0].chapter_title == ''

    def test_strategy_best_match_assigns_to_greatest_overlap(self):
        """Strategy 'best_match' assigns to chapter with greatest overlap (default)."""
        # Segment: 8-15 overlaps more with chapter 1
        # Chapter 0: 0-10 -> overlap 2s
        # Chapter 1: 10-20 -> overlap 5s
        segments = [_make_segment(0, 8.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters, strategy='best_match')
        assert segments[0].chapter_index == 1
        assert segments[0].chapter_title == 'Chapter B'

    def test_strategy_split_falls_back_to_best_match(self):
        """Strategy 'split' currently falls back to 'best_match' (placeholder)."""
        # Segment: 8-15 overlaps more with chapter 1
        segments = [_make_segment(0, 8.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters, strategy='split')
        # Should behave like best_match
        assert segments[0].chapter_index == 1
        assert segments[0].chapter_title == 'Chapter B'

    def test_strategy_default_is_best_match(self):
        """Default strategy (no parameter) is 'best_match'."""
        # Segment: 8-15 overlaps more with chapter 1
        segments = [_make_segment(0, 8.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        # Call without strategy parameter
        assign_chapter_indices(segments, chapters)
        assert segments[0].chapter_index == 1
        assert segments[0].chapter_title == 'Chapter B'

    def test_strategy_invalid_falls_back_to_best_match(self):
        """Invalid strategy falls back to 'best_match' with warning."""
        segments = [_make_segment(0, 5.0, 8.0)]
        chapters = [{'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0}]
        # Should not raise, should fall back
        assign_chapter_indices(segments, chapters, strategy='invalid_strategy')
        assert segments[0].chapter_index == 0

    def test_strategy_first_exactly_on_boundary(self):
        """Strategy 'first' assigns when segment starts exactly at chapter boundary."""
        # Segment: 10-15 starts exactly at chapter 1's start
        # Chapter 0: 0-10
        # Chapter 1: 10-20
        segments = [_make_segment(0, 10.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters, strategy='first')
        # First chapter with overlap is chapter 1 (boundary)
        assert segments[0].chapter_index == 1

    def test_strategy_best_match_tie_goes_to_first(self):
        """Strategy 'best_match' assigns to first when overlaps are equal."""
        # Segment: 5-15 (10s total)
        # Chapter 0: 0-10 -> overlap 5s (50%)
        # Chapter 1: 10-20 -> overlap 5s (50%)
        # Tie should go to first (chapter 0)
        segments = [_make_segment(0, 5.0, 15.0)]
        chapters = [
            {'title': 'Chapter A', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Chapter B', 'start_time': 10.0, 'end_time': 20.0},
        ]
        assign_chapter_indices(segments, chapters, strategy='best_match')
        assert segments[0].chapter_index == 0


# --- compute_relevance_matrix (US-72-009) ---

class TestComputeRelevanceMatrix:
    """Tests for compute_relevance_matrix() in bridge.py (US-72-009)."""

    def test_identical_topics_gives_score_1(self):
        """ChapterCandidates with identical topics produce 1.0."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs', 'pets'])]
        vid = [_make_chapter(0, 0, 4, topics=['cats', 'dogs', 'pets'])]
        matrix = compute_relevance_matrix(vo, vid)
        assert len(matrix) == 1
        assert len(matrix[0]) == 1
        assert matrix[0][0] == pytest.approx(1.0)

    def test_disjoint_topics_gives_score_0(self):
        """No overlap produces 0.0."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['cars', 'trucks'])]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(0.0)

    def test_partial_overlap_jaccard(self):
        """Partial overlap: intersection=1, union=3 -> 1/3."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['dogs', 'fish'])]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(1.0 / 3.0)

    def test_case_insensitive(self):
        """Keywords compared case-insensitively."""
        vo = [_make_chapter(0, 0, 4, topics=['CATS', 'Dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(1.0)

    def test_matrix_dimensions(self):
        """Matrix shape is (num_vo_chapters x num_vid_chapters)."""
        vo = [
            _make_chapter(0, 0, 4, topics=['a', 'b']),
            _make_chapter(1, 5, 9, topics=['c', 'd']),
            _make_chapter(2, 10, 14, topics=['e']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['a']),
            _make_chapter(1, 5, 9, topics=['c', 'e']),
        ]
        matrix = compute_relevance_matrix(vo, vid)
        assert len(matrix) == 3
        assert all(len(row) == 2 for row in matrix)

    def test_empty_voiceover_chapters_returns_empty(self):
        """Empty voiceover chapter list returns empty matrix."""
        vid = [_make_chapter(0, 0, 4, topics=['a'])]
        assert compute_relevance_matrix([], vid) == []

    def test_empty_video_chapters_returns_empty(self):
        """Empty video chapter list returns empty matrix."""
        vo = [_make_chapter(0, 0, 4, topics=['a'])]
        assert compute_relevance_matrix(vo, []) == []

    def test_both_empty_returns_empty(self):
        """Both empty returns empty matrix."""
        assert compute_relevance_matrix([], []) == []

    def test_empty_topics_gives_zero(self):
        """Chapters with empty topics produce 0.0."""
        vo = [_make_chapter(0, 0, 4, topics=[])]
        vid = [_make_chapter(0, 0, 4, topics=['a'])]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(0.0)

    def test_both_empty_topics_gives_zero(self):
        """Both chapters with empty topics produce 0.0."""
        vo = [_make_chapter(0, 0, 4, topics=[])]
        vid = [_make_chapter(0, 0, 4, topics=[])]
        matrix = compute_relevance_matrix(vo, vid)
        assert matrix[0][0] == pytest.approx(0.0)

    def test_all_values_normalized_0_to_1(self):
        """All values in the matrix must be in [0.0, 1.0]."""
        vo = [
            _make_chapter(0, 0, 4, topics=['a', 'b', 'c']),
            _make_chapter(1, 5, 9, topics=['d']),
            _make_chapter(2, 10, 14, topics=['e', 'f']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['a', 'x']),
            _make_chapter(1, 5, 9, topics=['b', 'c', 'd', 'e']),
            _make_chapter(2, 10, 14, topics=['f']),
        ]
        matrix = compute_relevance_matrix(vo, vid)
        for row in matrix:
            for val in row:
                assert 0.0 <= val <= 1.0, f"Value {val} out of range"

    def test_multi_chapter_specific_values(self):
        """Verify specific cells in a multi-chapter matrix."""
        vo = [
            _make_chapter(0, 0, 4, topics=['paris', 'france', 'eiffel']),
            _make_chapter(1, 5, 9, topics=['tokyo', 'japan', 'sushi']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['paris', 'france', 'wine']),
            _make_chapter(1, 5, 9, topics=['tokyo', 'ramen', 'japan']),
        ]
        matrix = compute_relevance_matrix(vo, vid)
        # vo[0] vs vid[0]: {paris,france}/4 = 0.5
        assert matrix[0][0] == pytest.approx(2.0 / 4.0)
        # vo[0] vs vid[1]: no overlap
        assert matrix[0][1] == pytest.approx(0.0)
        # vo[1] vs vid[0]: no overlap
        assert matrix[1][0] == pytest.approx(0.0)
        # vo[1] vs vid[1]: {tokyo,japan}/4 = 0.5
        assert matrix[1][1] == pytest.approx(2.0 / 4.0)

    # --- Embedding blend tests (US-76-009) ---

    def test_blended_score_with_mock_embedding_fn(self):
        """With mock embedding_fn, blended score is 0.6*Jaccard + 0.4*cosine."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['dogs', 'fish'])]
        # Jaccard: intersection=1 (dogs), union=3 -> 1/3
        # Mock embedding returns 0.8
        mock_emb = lambda a, b: 0.8
        matrix = compute_relevance_matrix(vo, vid, embedding_fn=mock_emb)
        expected = 0.6 * (1.0 / 3.0) + 0.4 * 0.8
        assert matrix[0][0] == pytest.approx(expected)

    def test_no_embedding_fn_unchanged_behavior(self):
        """Without embedding_fn, output is identical to pure Jaccard."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['dogs', 'fish'])]
        matrix_no_emb = compute_relevance_matrix(vo, vid)
        matrix_none = compute_relevance_matrix(vo, vid, embedding_fn=None)
        assert matrix_no_emb == matrix_none
        # Pure Jaccard: 1/3
        assert matrix_no_emb[0][0] == pytest.approx(1.0 / 3.0)

    def test_embedding_1_jaccard_0_produces_04(self):
        """Embedding similarity 1.0 with Jaccard 0.0 produces score 0.4."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['fish', 'birds'])]
        # Jaccard: 0 intersection -> 0.0
        # Embedding: always returns 1.0
        mock_emb = lambda a, b: 1.0
        matrix = compute_relevance_matrix(vo, vid, embedding_fn=mock_emb)
        assert matrix[0][0] == pytest.approx(0.4)


# --- compute_chapter_alignment_scores (US-98-007) ---

class TestComputeChapterAlignmentScores:
    """Tests for bidirectional voiceover-video chapter alignment (US-98-007)."""

    def test_empty_voiceover_chapters_returns_empty(self):
        """Empty voiceover chapter list returns empty result."""
        vid = [_make_chapter(0, 0, 4, topics=['a'])]
        result = compute_chapter_alignment_scores([], vid)
        assert result['similarity_matrix'] == []
        assert result['best_video_chapter_per_vo'] == []

    def test_empty_video_chapters_returns_empty(self):
        """Empty video chapter list returns empty result."""
        vo = [_make_chapter(0, 0, 4, topics=['a'])]
        result = compute_chapter_alignment_scores(vo, [])
        assert result['similarity_matrix'] == []
        assert result['best_video_chapter_per_vo'] == []

    def test_both_empty_returns_empty(self):
        """Both empty returns empty result."""
        result = compute_chapter_alignment_scores([], [])
        assert result['similarity_matrix'] == []

    def test_identical_topics_and_timing(self):
        """High score when voiceover and video chapters have identical topics and timing."""
        vo = [_make_chapter(0, 0, 4, topics=['travel', 'paris'], confidence=0.8)]
        vid = [_make_chapter(0, 0, 4, topics=['travel', 'paris'])]

        result = compute_chapter_alignment_scores(vo, vid)

        # Should have 1x1 matrix
        assert len(result['similarity_matrix']) == 1
        assert len(result['similarity_matrix'][0]) == 1
        # Best match should be video chapter 0
        assert result['best_video_chapter_per_vo'][0] == 0
        # Score should be high (keyword + temporal both high)
        assert result['similarity_matrix'][0][0] > 0.7

    def test_keyword_overlap_scores(self):
        """Keyword overlap is considered in scoring."""
        vo = [
            _make_chapter(0, 0, 4, topics=['cats', 'dogs', 'pets']),
            _make_chapter(1, 5, 9, topics=['travel', 'paris']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['cats', 'dogs', 'fish']),
            _make_chapter(1, 5, 9, topics=['travel', 'japan']),
        ]

        result = compute_chapter_alignment_scores(vo, vid)

        # First VO chapter should match first video chapter better (more keyword overlap)
        assert result['similarity_matrix'][0][0] > result['similarity_matrix'][0][1]
        # Second VO chapter should match second video chapter better
        assert result['similarity_matrix'][1][1] > result['similarity_matrix'][1][0]

    def test_temporal_alignment_scores(self):
        """Temporal alignment is considered in scoring."""
        # VO chapter at segments 0-4, video chapters at different ranges
        vo = [_make_chapter(0, 0, 4, topics=['topic1'])]
        vid = [
            _make_chapter(0, 0, 4, topics=['different1']),  # Same range
            _make_chapter(1, 10, 14, topics=['different2']),  # Far apart
        ]

        result = compute_chapter_alignment_scores(vo, vid)

        # First video chapter has better temporal alignment
        assert result['temporal_scores'][0][0] > result['temporal_scores'][0][1]
        # Overall score should favor first video chapter
        assert result['similarity_matrix'][0][0] > result['similarity_matrix'][0][1]

    def test_confidence_weight_affects_scores(self):
        """Marker confidence affects alignment scores."""
        vo_low_conf = [_make_chapter(0, 0, 4, topics=['topic'], confidence=0.5)]
        vo_high_conf = [_make_chapter(0, 0, 4, topics=['topic'], confidence=0.9)]
        vid = [_make_chapter(0, 0, 4, topics=['topic'])]

        result_low = compute_chapter_alignment_scores(vo_low_conf, vid)
        result_high = compute_chapter_alignment_scores(vo_high_conf, vid)

        # Higher confidence should result in higher score
        assert result_high['similarity_matrix'][0][0] > result_low['similarity_matrix'][0][0]

    def test_best_chapter_identified_correctly(self):
        """Test that best video chapter is correctly identified for each VO chapter."""
        vo = [
            _make_chapter(0, 0, 4, topics=['cooking', 'food']),
            _make_chapter(1, 5, 9, topics=['travel', 'paris']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['cooking', 'recipe']),
            _make_chapter(1, 5, 9, topics=['travel', 'japan']),
            _make_chapter(2, 10, 14, topics=['random']),
        ]

        result = compute_chapter_alignment_scores(vo, vid)

        # First VO chapter should match first video chapter
        assert result['best_video_chapter_per_vo'][0] == 0
        # Second VO chapter should match second video chapter
        assert result['best_video_chapter_per_vo'][1] == 1

    def test_matrix_dimensions_correct(self):
        """Matrix dimensions match input chapter counts."""
        vo = [
            _make_chapter(0, 0, 4, topics=['a']),
            _make_chapter(1, 5, 9, topics=['b']),
            _make_chapter(2, 10, 14, topics=['c']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['x']),
            _make_chapter(1, 5, 9, topics=['y']),
        ]

        result = compute_chapter_alignment_scores(vo, vid)

        assert len(result['similarity_matrix']) == 3
        assert all(len(row) == 2 for row in result['similarity_matrix'])
        assert len(result['best_video_chapter_per_vo']) == 3
        assert len(result['keyword_scores']) == 3
        assert len(result['temporal_scores']) == 3
        assert len(result['confidence_weights']) == 3

    def test_scores_normalized_to_0_1(self):
        """All scores are normalized to [0, 1] range."""
        vo = [
            _make_chapter(0, 0, 4, topics=['unique1']),
            _make_chapter(1, 5, 9, topics=['unique2']),
        ]
        vid = [
            _make_chapter(0, 0, 4, topics=['different1']),
            _make_chapter(1, 5, 9, topics=['different2']),
        ]

        result = compute_chapter_alignment_scores(vo, vid)

        for row in result['similarity_matrix']:
            for val in row:
                assert 0.0 <= val <= 1.0, f"Score {val} out of range"

        for row in result['keyword_scores']:
            for val in row:
                assert 0.0 <= val <= 1.0

        for row in result['temporal_scores']:
            for val in row:
                assert 0.0 <= val <= 1.0

    def test_with_embedding_function(self):
        """Embedding function blends with keyword similarity."""
        vo = [_make_chapter(0, 0, 4, topics=['cats', 'dogs'])]
        vid = [_make_chapter(0, 0, 4, topics=['dogs', 'fish'])]

        # Mock embedding returns high similarity
        mock_emb = lambda a, b: 0.9

        result_with_emb = compute_chapter_alignment_scores(vo, vid, embedding_fn=mock_emb)
        result_without_emb = compute_chapter_alignment_scores(vo, vid)

        # With embedding, score should be different (higher when embedding is high)
        assert result_with_emb['similarity_matrix'][0][0] != result_without_emb['similarity_matrix'][0][0]

    def test_returns_all_score_components(self):
        """Result contains all score components for downstream use."""
        vo = [_make_chapter(0, 0, 4, topics=['topic'], confidence=0.8)]
        vid = [_make_chapter(0, 0, 4, topics=['topic'])]

        result = compute_chapter_alignment_scores(vo, vid)

        assert 'similarity_matrix' in result
        assert 'best_video_chapter_per_vo' in result
        assert 'temporal_scores' in result
        assert 'keyword_scores' in result
        assert 'confidence_weights' in result


# --- build_segment_chapter_map with source parameter (US-98-007) ---

class TestBuildSegmentChapterMapWithSource:
    """Tests for build_segment_chapter_map with source parameter."""

    def test_default_source_video(self):
        """Default source is 'video'."""
        ch = _make_chapter(0, 0, 2)
        result = build_segment_chapter_map([ch])
        assert result == {0: 0, 1: 0, 2: 0}

    def test_explicit_source_video(self):
        """Explicit source='video' works."""
        ch = _make_chapter(0, 0, 2)
        result = build_segment_chapter_map([ch], source='video')
        assert result == {0: 0, 1: 0, 2: 0}

    def test_source_voiceover(self):
        """source='voiceover' works."""
        ch = _make_chapter(0, 0, 2)
        result = build_segment_chapter_map([ch], source='voiceover')
        assert result == {0: 0, 1: 0, 2: 0}

    def test_multiple_chapters_with_source(self):
        """Multiple chapters map correctly with source parameter."""
        chapters = [
            _make_chapter(0, 0, 2),
            _make_chapter(1, 5, 7),
        ]
        result_vo = build_segment_chapter_map(chapters, source='voiceover')
        result_vid = build_segment_chapter_map(chapters, source='video')

        # Both should produce same mapping
        assert result_vo == result_vid
        assert result_vo[0] == 0
        assert result_vo[2] == 0
        assert result_vo[5] == 1
        assert result_vo[7] == 1
