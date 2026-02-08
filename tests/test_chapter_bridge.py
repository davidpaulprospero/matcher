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
    _ranges_overlap,
)
from src.state import TranscriptSegment


# --- Helpers ---

def _make_listicle_group(group_id, start, end, label="item", keywords=None, expected_count=None):
    return ListicleGroup(
        group_id=group_id,
        item_label=label,
        start_segment_idx=start,
        end_segment_idx=end,
        topic_keywords=keywords or [],
        expected_count=expected_count,
    )


def _make_chapter(chapter_id, start, end, title="Chapter", topics=None, strategy="topic"):
    return ChapterCandidate(
        chapter_id=chapter_id,
        start_segment_idx=start,
        end_segment_idx=end,
        title=title,
        topics=topics or [],
        detection_strategy=strategy,
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

    def test_confidence_with_expected_count(self):
        group = _make_listicle_group(0, 0, 3, expected_count=5)
        chapters = listicle_groups_to_chapters([group])
        assert chapters[0].confidence == 0.75

    def test_confidence_without_expected_count(self):
        group = _make_listicle_group(0, 0, 3, expected_count=None)
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
