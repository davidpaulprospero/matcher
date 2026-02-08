"""
Unit tests for the listicle-to-chapter bridge (US-71-010).

Tests:
- listicle_groups_to_chapters: conversion with topic_keywords populated
- merge_chapters: YouTube chapters take precedence for overlapping ranges
- build_unified_chapters: end-to-end bridge
- build_segment_chapter_map: segment index to chapter mapping
"""

import pytest

from src.chapter_detection.models import ChapterCandidate, ListicleGroup
from src.chapter_detection.bridge import (
    listicle_groups_to_chapters,
    merge_chapters,
    build_unified_chapters,
    build_segment_chapter_map,
    _ranges_overlap,
)


# --- Helpers ---

def _lg(group_id: int, start: int, end: int,
        keywords: list = None, label: str = "", expected_count=None) -> ListicleGroup:
    """Create a ListicleGroup for testing."""
    return ListicleGroup(
        group_id=group_id,
        item_label=label or f"item {group_id + 1}",
        start_segment_idx=start,
        end_segment_idx=end,
        topic_keywords=keywords or [],
        expected_count=expected_count,
    )


def _ch(chapter_id: int, start: int, end: int,
        title: str = "", strategy: str = "topic",
        topics: list = None) -> ChapterCandidate:
    """Create a ChapterCandidate for testing."""
    return ChapterCandidate(
        chapter_id=chapter_id,
        start_segment_idx=start,
        end_segment_idx=end,
        title=title or f"Chapter {chapter_id}",
        topics=topics or [],
        detection_strategy=strategy,
    )


# --- listicle_groups_to_chapters ---

class TestListicleGroupsToChapters:
    def test_empty_input(self):
        result = listicle_groups_to_chapters([])
        assert result == []

    def test_single_group(self):
        groups = [_lg(0, 0, 4, keywords=["travel", "paris", "europe"], label="first")]
        result = listicle_groups_to_chapters(groups)
        assert len(result) == 1
        ch = result[0]
        assert ch.start_segment_idx == 0
        assert ch.end_segment_idx == 4
        assert ch.detection_strategy == "listicle"
        assert ch.topics == ["travel", "paris", "europe"]
        assert "first" in ch.title

    def test_topic_keywords_populated(self):
        """AC: Converted chapters have topic_keywords populated from ListicleGroup."""
        groups = [
            _lg(0, 0, 3, keywords=["beaches", "surfing"]),
            _lg(1, 4, 7, keywords=["mountains", "hiking", "snow"]),
        ]
        result = listicle_groups_to_chapters(groups)
        assert result[0].topics == ["beaches", "surfing"]
        assert result[1].topics == ["mountains", "hiking", "snow"]

    def test_confidence_with_expected_count(self):
        """Confidence slightly higher when expected_count is set."""
        without = _lg(0, 0, 3)
        with_count = _lg(0, 0, 3, expected_count=5)
        r1 = listicle_groups_to_chapters([without])
        r2 = listicle_groups_to_chapters([with_count])
        assert r2[0].confidence > r1[0].confidence

    def test_title_from_label_and_keywords(self):
        group = _lg(0, 0, 2, keywords=["food", "culture"], label="second")
        result = listicle_groups_to_chapters([group])
        assert "second" in result[0].title
        assert "food" in result[0].title

    def test_title_fallback_no_label(self):
        group = ListicleGroup(
            group_id=0, item_label="", start_segment_idx=0,
            end_segment_idx=2, topic_keywords=[],
        )
        result = listicle_groups_to_chapters([group])
        assert "Item 1" in result[0].title


# --- _ranges_overlap ---

class TestRangesOverlap:
    def test_no_overlap(self):
        assert not _ranges_overlap(0, 3, 5, 8)

    def test_adjacent_no_overlap(self):
        assert not _ranges_overlap(0, 3, 4, 8)

    def test_overlap_partial(self):
        assert _ranges_overlap(0, 5, 3, 8)

    def test_overlap_contained(self):
        assert _ranges_overlap(0, 10, 3, 5)

    def test_overlap_exact(self):
        assert _ranges_overlap(0, 5, 0, 5)

    def test_single_point_overlap(self):
        assert _ranges_overlap(0, 3, 3, 5)


# --- merge_chapters ---

class TestMergeChapters:
    def test_empty_both(self):
        assert merge_chapters([], []) == []

    def test_only_youtube(self):
        yt = [_ch(0, 0, 5, "Intro"), _ch(1, 6, 10, "Main")]
        result = merge_chapters(yt, [])
        assert len(result) == 2

    def test_only_listicle(self):
        lc = [_ch(0, 0, 3, strategy="listicle"), _ch(1, 4, 7, strategy="listicle")]
        result = merge_chapters([], lc)
        assert len(result) == 2

    def test_youtube_precedence_overlapping(self):
        """AC: YouTube chapters take precedence for overlapping ranges."""
        yt = [_ch(0, 0, 5, "YouTube Chapter")]
        lc = [
            _ch(0, 2, 4, "Listicle Overlap", strategy="listicle"),  # overlaps with YouTube
            _ch(1, 6, 9, "Listicle Gap", strategy="listicle"),      # no overlap
        ]
        result = merge_chapters(yt, lc)
        assert len(result) == 2
        # First should be YouTube chapter
        assert result[0].title == "YouTube Chapter"
        assert result[0].start_segment_idx == 0
        # Second should be the non-overlapping listicle
        assert result[1].title == "Listicle Gap"
        assert result[1].start_segment_idx == 6

    def test_chapter_ids_reassigned(self):
        """After merge, chapter_ids are sequential."""
        yt = [_ch(5, 10, 15, "Late YouTube")]
        lc = [_ch(0, 0, 5, "Early Listicle", strategy="listicle")]
        result = merge_chapters(yt, lc)
        assert result[0].chapter_id == 0
        assert result[1].chapter_id == 1
        assert result[0].start_segment_idx < result[1].start_segment_idx

    def test_sorted_by_start_index(self):
        yt = [_ch(0, 10, 15)]
        lc = [_ch(0, 0, 5, strategy="listicle")]
        result = merge_chapters(yt, lc)
        assert result[0].start_segment_idx == 0
        assert result[1].start_segment_idx == 10

    def test_all_overlapping_keeps_only_youtube(self):
        """When all listicle groups overlap with YouTube, only YouTube chapters remain."""
        yt = [_ch(0, 0, 10)]
        lc = [_ch(0, 2, 4, strategy="listicle"), _ch(1, 6, 8, strategy="listicle")]
        result = merge_chapters(yt, lc)
        assert len(result) == 1
        assert result[0].title == "Chapter 0"


# --- build_unified_chapters ---

class TestBuildUnifiedChapters:
    def test_both_empty(self):
        assert build_unified_chapters([], []) == []

    def test_only_listicle_groups(self):
        """AC: Conversion from listicle groups to chapters works."""
        groups = [
            _lg(0, 0, 3, keywords=["intro", "overview"]),
            _lg(1, 4, 7, keywords=["details", "specs"]),
        ]
        result = build_unified_chapters([], groups)
        assert len(result) == 2
        assert result[0].detection_strategy == "listicle"
        assert result[0].topics == ["intro", "overview"]
        assert result[1].topics == ["details", "specs"]

    def test_merge_with_youtube_precedence(self):
        """AC: Merged with YouTube chapters taking precedence."""
        yt_chapters = [_ch(0, 0, 5, "YouTube Intro")]
        groups = [
            _lg(0, 3, 6, keywords=["overlap"]),  # overlaps 3-5 with YouTube
            _lg(1, 7, 10, keywords=["gap"]),      # no overlap
        ]
        result = build_unified_chapters(yt_chapters, groups)
        assert len(result) == 2
        assert result[0].title == "YouTube Intro"
        assert result[1].detection_strategy == "listicle"

    def test_unified_available_as_chapters(self):
        """AC: Unified list available via same interface as regular chapters."""
        groups = [_lg(0, 0, 3, keywords=["test"])]
        result = build_unified_chapters([], groups)
        # Should be ChapterCandidate objects usable by chapter-aware scoring
        assert isinstance(result[0], ChapterCandidate)
        assert hasattr(result[0], 'topics')
        assert hasattr(result[0], 'start_segment_idx')
        assert hasattr(result[0], 'end_segment_idx')


# --- build_segment_chapter_map ---

class TestBuildSegmentChapterMap:
    def test_empty(self):
        assert build_segment_chapter_map([]) == {}

    def test_single_chapter(self):
        chapters = [_ch(0, 2, 5)]
        result = build_segment_chapter_map(chapters)
        assert result == {2: 0, 3: 0, 4: 0, 5: 0}

    def test_multiple_chapters(self):
        chapters = [_ch(0, 0, 2), _ch(1, 3, 5)]
        result = build_segment_chapter_map(chapters)
        assert result[0] == 0
        assert result[2] == 0
        assert result[3] == 1
        assert result[5] == 1

    def test_gaps_not_in_map(self):
        chapters = [_ch(0, 0, 2), _ch(1, 5, 7)]
        result = build_segment_chapter_map(chapters)
        assert 3 not in result
        assert 4 not in result
