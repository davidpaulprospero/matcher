"""
Unit tests for US-73-002: Map video caption segments to video chapters
during caption fetch (_populate_text_metadata).

Tests:
- _map_segments_to_video_chapters helper (dict-based chapters)
- chapter_index/chapter_title populated in text_metadata entries
- Segments at boundaries, midpoints, outside ranges
- 3+ chapters with complete coverage
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.stages.caption_stage import CaptionStage
from src.chapter_detector.detector import VideoChapter


# ============================================================
# _map_segments_to_video_chapters tests (static helper)
# ============================================================


@pytest.mark.fast
class TestMapSegmentsToVideoChapters:
    """Tests for CaptionStage._map_segments_to_video_chapters."""

    def test_basic_mapping_with_dict_chapters(self):
        """Dict-format chapters are converted and mapped correctly."""
        chapters = [
            {'title': 'Intro', 'start_time': 0, 'end_time': 60},
            {'title': 'Main', 'start_time': 60, 'end_time': 180},
            {'title': 'Outro', 'start_time': 180, 'end_time': 240},
        ]
        segments = [
            {'start': 10, 'end': 30, 'text': 'hello'},    # -> Intro
            {'start': 90, 'end': 120, 'text': 'body'},    # -> Main
            {'start': 200, 'end': 230, 'text': 'bye'},    # -> Outro
        ]
        result = CaptionStage._map_segments_to_video_chapters(segments, chapters)
        assert result[0] == (0, 'Intro')
        assert result[1] == (1, 'Main')
        assert result[2] == (2, 'Outro')

    def test_segment_at_chapter_boundary(self):
        """Segment exactly at chapter boundary maps to the chapter it falls into."""
        chapters = [
            {'title': 'Part1', 'start_time': 0, 'end_time': 60},
            {'title': 'Part2', 'start_time': 60, 'end_time': 120},
        ]
        # Segment starts exactly at boundary
        segments = [{'start': 60, 'end': 90, 'text': 'at boundary'}]
        result = CaptionStage._map_segments_to_video_chapters(segments, chapters)
        assert result[0] == (1, 'Part2')

    def test_segment_spanning_boundary_greatest_overlap(self):
        """Segment spanning chapter boundary assigned to chapter with greatest overlap."""
        chapters = [
            {'title': 'A', 'start_time': 0, 'end_time': 60},
            {'title': 'B', 'start_time': 60, 'end_time': 120},
        ]
        # 40-80: 20s in A (40-60), 20s in B (60-80) -> tie, first wins
        segments = [{'start': 40, 'end': 80, 'text': 'spanning equal'}]
        result = CaptionStage._map_segments_to_video_chapters(segments, chapters)
        # With equal overlap, first match wins (overlap > best_overlap uses strict >)
        assert result[0][0] == 0  # chapter_index 0 (A)

        # 50-90: 10s in A (50-60), 30s in B (60-90) -> B wins
        segments2 = [{'start': 50, 'end': 90, 'text': 'spanning more in B'}]
        result2 = CaptionStage._map_segments_to_video_chapters(segments2, chapters)
        assert result2[0] == (1, 'B')

    def test_segment_outside_all_chapters(self):
        """Segment outside all chapters gets chapter_index=-1 and chapter_title=''."""
        chapters = [
            {'title': 'Main', 'start_time': 30, 'end_time': 90},
        ]
        # Before first chapter
        segments = [{'start': 0, 'end': 20, 'text': 'before'}]
        result = CaptionStage._map_segments_to_video_chapters(segments, chapters)
        assert result[0] == (-1, '')

        # After last chapter
        segments2 = [{'start': 100, 'end': 120, 'text': 'after'}]
        result2 = CaptionStage._map_segments_to_video_chapters(segments2, chapters)
        assert result2[0] == (-1, '')

    def test_empty_chapters(self):
        """No chapters => all segments get chapter_index=-1."""
        segments = [{'start': 10, 'end': 50, 'text': 'hello'}]
        result = CaptionStage._map_segments_to_video_chapters(segments, [])
        assert result[0] == (-1, '')

    def test_empty_segments(self):
        """No segments => empty result."""
        chapters = [{'title': 'Intro', 'start_time': 0, 'end_time': 60}]
        result = CaptionStage._map_segments_to_video_chapters([], chapters)
        assert result == {}

    def test_videochapter_objects_passed_directly(self):
        """VideoChapter objects (not dicts) are handled without conversion."""
        chapters = [
            VideoChapter(title='Ch1', start_time=0, end_time=100),
            VideoChapter(title='Ch2', start_time=100, end_time=200),
        ]
        segments = [{'start': 50, 'end': 80, 'text': 'mid'}]
        result = CaptionStage._map_segments_to_video_chapters(segments, chapters)
        assert result[0] == (0, 'Ch1')

    def test_three_plus_chapters_comprehensive(self):
        """3+ chapters with segments at midpoints, boundaries, and outside."""
        chapters = [
            {'title': 'Introduction', 'start_time': 0, 'end_time': 60},
            {'title': 'History', 'start_time': 60, 'end_time': 180},
            {'title': 'Modern Era', 'start_time': 180, 'end_time': 300},
            {'title': 'Conclusion', 'start_time': 300, 'end_time': 360},
        ]
        segments = [
            {'start': 25, 'end': 35, 'text': 'intro mid'},          # midpoint ch0
            {'start': 55, 'end': 65, 'text': 'boundary 0-1'},       # boundary ch0/ch1
            {'start': 100, 'end': 140, 'text': 'history mid'},      # midpoint ch1
            {'start': 175, 'end': 185, 'text': 'boundary 1-2'},     # boundary ch1/ch2
            {'start': 240, 'end': 260, 'text': 'modern mid'},       # midpoint ch2
            {'start': 330, 'end': 350, 'text': 'conclusion mid'},   # midpoint ch3
            {'start': 370, 'end': 390, 'text': 'after all'},        # outside
        ]
        result = CaptionStage._map_segments_to_video_chapters(segments, chapters)

        assert result[0] == (0, 'Introduction'), "Midpoint of ch0"
        # boundary 55-65: 5s in ch0 (55-60), 5s in ch1 (60-65) -> tie, ch0 wins
        assert result[1][0] == 0, "Boundary tie -> first chapter"
        assert result[2] == (1, 'History'), "Midpoint of ch1"
        # boundary 175-185: 5s in ch1 (175-180), 5s in ch2 (180-185) -> tie, ch1 wins
        assert result[3][0] == 1, "Boundary tie -> first chapter"
        assert result[4] == (2, 'Modern Era'), "Midpoint of ch2"
        assert result[5] == (3, 'Conclusion'), "Midpoint of ch3"
        assert result[6] == (-1, ''), "Outside all chapters"


# ============================================================
# Integration: chapter fields in text_metadata via _populate_text_metadata
# ============================================================


@pytest.mark.fast
class TestPopulateTextMetadataChapterFields:
    """Tests that _populate_text_metadata adds chapter_index/chapter_title."""

    def _make_stage(self):
        """Create a minimal CaptionStage instance."""
        stage = CaptionStage.__new__(CaptionStage)
        return stage

    def _make_state(self):
        """Create a minimal mock state."""
        state = MagicMock()
        state.text_metadata = []
        state.video_search_results = []
        return state

    def test_chapter_fields_populated(self):
        """text_metadata entries have chapter_index and chapter_title."""
        stage = self._make_stage()
        state = self._make_state()

        caption_results = {
            'vid1': {
                'segments': [
                    {'start': 10, 'end': 30, 'text': 'hello'},
                    {'start': 70, 'end': 100, 'text': 'world'},
                ],
                'language': 'en',
                'video_chapters': [
                    {'title': 'Intro', 'start_time': 0, 'end_time': 60},
                    {'title': 'Main', 'start_time': 60, 'end_time': 120},
                ],
            }
        }

        stage._populate_text_metadata(state, caption_results, config=None)

        entries = state.text_metadata
        assert len(entries) == 2
        assert entries[0]['chapter_index'] == 0
        assert entries[0]['chapter_title'] == 'Intro'
        assert entries[1]['chapter_index'] == 1
        assert entries[1]['chapter_title'] == 'Main'

    def test_no_chapters_gives_default(self):
        """Without chapters, chapter_index=-1 and chapter_title=''."""
        stage = self._make_stage()
        state = self._make_state()

        caption_results = {
            'vid1': {
                'segments': [
                    {'start': 10, 'end': 30, 'text': 'hello'},
                ],
                'language': 'en',
                'video_chapters': [],
            }
        }

        stage._populate_text_metadata(state, caption_results, config=None)

        entries = state.text_metadata
        assert len(entries) == 1
        assert entries[0]['chapter_index'] == -1
        assert entries[0]['chapter_title'] == ''

    def test_missing_chapters_key_gives_default(self):
        """Missing video_chapters key defaults to no chapter mapping."""
        stage = self._make_stage()
        state = self._make_state()

        caption_results = {
            'vid1': {
                'segments': [
                    {'start': 10, 'end': 30, 'text': 'hello'},
                ],
                'language': 'en',
                # No video_chapters key
            }
        }

        stage._populate_text_metadata(state, caption_results, config=None)

        entries = state.text_metadata
        assert len(entries) == 1
        assert entries[0]['chapter_index'] == -1
        assert entries[0]['chapter_title'] == ''
