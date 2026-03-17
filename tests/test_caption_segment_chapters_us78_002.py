"""Tests for US-78-002: CaptionSegment chapter_index/chapter_title fields
and map_segments_to_chapters() function.

Verifies:
- CaptionSegment has optional chapter_index (int) and chapter_title (str) fields
- CaptionSegment.to_dict() includes chapter fields when present
- Cache deserialization handles missing chapter fields (backward compat)
- map_segments_to_chapters assigns based on greatest timestamp overlap
"""

import pytest

from src.caption.models import CaptionSegment, CaptionResult, map_segments_to_chapters
from src.caption.cache_models import CachedCaption


def _seg(index: int, start: float, end: float, text: str = "test",
         source_file: str = "vid1") -> CaptionSegment:
    """Helper to create a CaptionSegment."""
    return CaptionSegment(
        index=index, start_time=start, end_time=end,
        text=text, source_file=source_file,
    )


# ============================================================
# CaptionSegment field defaults
# ============================================================

@pytest.mark.fast
class TestCaptionSegmentChapterFields:
    """CaptionSegment has chapter_index and chapter_title with correct defaults."""

    def test_defaults_are_none_and_empty(self):
        seg = _seg(0, 0.0, 5.0)
        assert seg.chapter_index is None
        assert seg.chapter_title == ''

    def test_can_set_chapter_fields(self):
        seg = CaptionSegment(
            index=0, start_time=0.0, end_time=5.0, text="hi",
            chapter_index=2, chapter_title="History",
        )
        assert seg.chapter_index == 2
        assert seg.chapter_title == "History"

    def test_to_dict_omits_chapter_when_none(self):
        seg = _seg(0, 0.0, 5.0)
        d = seg.to_dict()
        assert 'chapter_index' not in d
        assert 'chapter_title' not in d

    def test_to_dict_includes_chapter_when_present(self):
        seg = CaptionSegment(
            index=0, start_time=0.0, end_time=5.0, text="hi",
            source_file="vid1", chapter_index=1, chapter_title="Intro",
        )
        d = seg.to_dict()
        assert d['chapter_index'] == 1
        assert d['chapter_title'] == "Intro"

    def test_to_dict_includes_chapter_index_zero(self):
        """chapter_index=0 is a valid value and should be serialized."""
        seg = CaptionSegment(
            index=0, start_time=0.0, end_time=5.0, text="hi",
            chapter_index=0, chapter_title="First",
        )
        d = seg.to_dict()
        assert d['chapter_index'] == 0
        assert d['chapter_title'] == "First"


# ============================================================
# CaptionResult.to_dict() includes chapter fields in segments
# ============================================================

@pytest.mark.fast
class TestCaptionResultSerializationChapterFields:
    """CaptionResult.to_dict() serializes chapter fields from segments."""

    def test_segments_with_chapters_appear_in_to_dict(self):
        seg = CaptionSegment(
            index=0, start_time=0.0, end_time=10.0, text="hello",
            source_file="vid1", chapter_index=0, chapter_title="Intro",
        )
        result = CaptionResult(video_id="vid1", segments=[seg])
        d = result.to_dict()
        assert d['segments'][0]['chapter_index'] == 0
        assert d['segments'][0]['chapter_title'] == "Intro"

    def test_segments_without_chapters_omit_fields(self):
        seg = _seg(0, 0.0, 10.0)
        result = CaptionResult(video_id="vid1", segments=[seg])
        d = result.to_dict()
        assert 'chapter_index' not in d['segments'][0]


# ============================================================
# Cache backward compatibility
# ============================================================

@pytest.mark.fast
class TestCacheBackwardCompatChapterFields:
    """Cache deserialization handles missing chapter fields gracefully."""

    def test_old_cache_without_chapter_fields(self):
        """Old cache entries without chapter fields load with defaults."""
        old_data = {
            'video_id': 'abc',
            'language': 'en',
            'segments': [
                {'index': 0, 'start': 0.0, 'end': 5.0, 'text': 'hi', 'source_file': 'abc'}
            ],
            'is_auto_generated': False,
            'format_source': 'json3',
            'fetch_timestamp': 1700000000.0,
            'duration': 5.0,
        }
        cached = CachedCaption.from_dict(old_data)
        result = cached.to_caption_result()
        seg = result.segments[0]
        assert seg.chapter_index is None
        assert seg.chapter_title == ''

    def test_new_cache_with_chapter_fields(self):
        """New cache entries with chapter fields are preserved through deserialization."""
        data = {
            'video_id': 'abc',
            'language': 'en',
            'segments': [
                {'index': 0, 'start': 0.0, 'end': 5.0, 'text': 'hi',
                 'source_file': 'abc', 'chapter_index': 1, 'chapter_title': 'Main'}
            ],
            'is_auto_generated': False,
            'format_source': 'json3',
            'fetch_timestamp': 1700000000.0,
            'duration': 5.0,
        }
        cached = CachedCaption.from_dict(data)
        result = cached.to_caption_result()
        seg = result.segments[0]
        assert seg.chapter_index == 1
        assert seg.chapter_title == 'Main'


# ============================================================
# map_segments_to_chapters function
# ============================================================

@pytest.mark.fast
class TestMapSegmentsToChapters:
    """map_segments_to_chapters assigns chapter info based on greatest overlap."""

    def test_basic_mapping(self):
        chapters = [
            {'title': 'Intro', 'start_time': 0, 'end_time': 60},
            {'title': 'Main', 'start_time': 60, 'end_time': 180},
            {'title': 'Outro', 'start_time': 180, 'end_time': 240},
        ]
        segments = [
            _seg(0, 10, 30),   # -> Intro
            _seg(1, 90, 120),  # -> Main
            _seg(2, 200, 230), # -> Outro
        ]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0].chapter_index == 0
        assert result[0].chapter_title == 'Intro'
        assert result[1].chapter_index == 1
        assert result[1].chapter_title == 'Main'
        assert result[2].chapter_index == 2
        assert result[2].chapter_title == 'Outro'

    def test_greatest_overlap_wins(self):
        """Segment spanning boundary assigned to chapter with most overlap."""
        chapters = [
            {'title': 'A', 'start_time': 0, 'end_time': 60},
            {'title': 'B', 'start_time': 60, 'end_time': 120},
        ]
        # 50-90: 10s in A (50-60), 30s in B (60-90) -> B wins
        segments = [_seg(0, 50, 90)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0].chapter_index == 1
        assert result[0].chapter_title == 'B'

    def test_equal_overlap_first_wins(self):
        """Equal overlap gives first chapter (strict > comparison)."""
        chapters = [
            {'title': 'A', 'start_time': 0, 'end_time': 60},
            {'title': 'B', 'start_time': 60, 'end_time': 120},
        ]
        # 40-80: 20s in A, 20s in B -> tie, A wins (first match)
        segments = [_seg(0, 40, 80)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0].chapter_index == 0

    def test_segment_outside_all_chapters(self):
        """Segment outside all chapters gets None/empty defaults."""
        chapters = [
            {'title': 'Main', 'start_time': 30, 'end_time': 90},
        ]
        segments = [_seg(0, 0, 20)]  # Before all chapters
        result = map_segments_to_chapters(segments, chapters)
        assert result[0].chapter_index is None
        assert result[0].chapter_title == ''

    def test_empty_chapters(self):
        """No chapters -> segments unchanged."""
        segments = [_seg(0, 0, 10)]
        result = map_segments_to_chapters(segments, [])
        assert result[0].chapter_index is None
        assert result[0].chapter_title == ''

    def test_empty_segments(self):
        """No segments -> returns empty list."""
        chapters = [{'title': 'Intro', 'start_time': 0, 'end_time': 60}]
        result = map_segments_to_chapters([], chapters)
        assert result == []

    def test_mutates_in_place(self):
        """Segments are mutated in-place and also returned."""
        chapters = [{'title': 'Ch1', 'start_time': 0, 'end_time': 100}]
        seg = _seg(0, 10, 20)
        result = map_segments_to_chapters([seg], chapters)
        assert result[0] is seg
        assert seg.chapter_index == 0
        assert seg.chapter_title == 'Ch1'

    def test_three_chapters_comprehensive(self):
        """3+ chapters with segments at midpoints, boundaries, and outside."""
        chapters = [
            {'title': 'Introduction', 'start_time': 0, 'end_time': 60},
            {'title': 'History', 'start_time': 60, 'end_time': 180},
            {'title': 'Modern Era', 'start_time': 180, 'end_time': 300},
        ]
        segments = [
            _seg(0, 25, 35, "intro mid"),       # midpoint ch0
            _seg(1, 100, 140, "history mid"),    # midpoint ch1
            _seg(2, 240, 260, "modern mid"),     # midpoint ch2
            _seg(3, 310, 330, "after all"),      # outside
        ]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0].chapter_index == 0
        assert result[0].chapter_title == 'Introduction'
        assert result[1].chapter_index == 1
        assert result[1].chapter_title == 'History'
        assert result[2].chapter_index == 2
        assert result[2].chapter_title == 'Modern Era'
        assert result[3].chapter_index is None
        assert result[3].chapter_title == ''
