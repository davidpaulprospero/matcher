"""
Unit tests for chapter-segment mapping and chapter topic scoring (US-70-009).

Tests:
- map_segments_to_chapters: temporal overlap mapping
- apply_chapter_topic_match: keyword-based chapter topic scoring
"""

import pytest
from unittest.mock import Mock

from src.chapter_detector import VideoChapter, map_segments_to_chapters
from src.chapter_detector.mapping import map_segments_to_chapters
from src.matching.scoring import MatchScoring
from src.utils import SRTSegment


def _mock_config():
    """Create a minimal mock config for MatchScoring."""
    config = Mock()
    matching = Mock()
    matching.multimodal_enabled = True
    matching.multimodal_weights = None
    matching.pool_normalization_enabled = True
    matching.chapter_matching_enabled = False
    matching.topic_mismatch_penalty = 0.15
    matching.broll_boost = 0.1
    matching.caption_quality_adjustment_enabled = True
    matching.caption_quality_high_boost = 0.05
    matching.caption_quality_low_penalty = 0.1
    matching.apply_timing_penalty = True
    matching.skip_llm_threshold = 0.85
    matching.language_confidence_penalty = 0.0
    # Scoring sub-config: use Mock with spec to avoid duck-type check passing
    matching.scoring = None
    config.matching = matching
    global_cache = Mock()
    global_cache.current_project_boost = 0.1
    config.global_cache = global_cache
    return config


def _seg(start: float, end: float, text: str = "", index: int = 0) -> SRTSegment:
    """Helper to create a simple SRTSegment."""
    return SRTSegment(index=index, start_time=start, end_time=end, text=text)


def _ch(title: str, start: float, end: float) -> VideoChapter:
    """Helper to create a VideoChapter."""
    return VideoChapter(title=title, start_time=start, end_time=end)


# ============================================================
# map_segments_to_chapters tests
# ============================================================


@pytest.mark.fast
class TestMapSegmentsToChapters:
    """Tests for the map_segments_to_chapters utility function."""

    def test_segment_fully_inside_chapter(self):
        """Segment fully inside a chapter maps to that chapter."""
        chapters = [_ch("Intro", 0, 60), _ch("Main", 60, 120)]
        segments = [_seg(10, 50)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (0, "Intro")

    def test_segment_in_second_chapter(self):
        """Segment in second chapter maps correctly."""
        chapters = [_ch("Intro", 0, 60), _ch("Main", 60, 120)]
        segments = [_seg(70, 110)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (1, "Main")

    def test_segment_on_exact_boundary(self):
        """Segment exactly at chapter boundary maps to second chapter."""
        chapters = [_ch("Intro", 0, 60), _ch("Main", 60, 120)]
        segments = [_seg(60, 90)]
        result = map_segments_to_chapters(segments, chapters)
        # Fully inside Main chapter
        assert result[0] == (1, "Main")

    def test_segment_spanning_boundary_more_in_first(self):
        """Segment spanning boundary maps to chapter with more overlap."""
        chapters = [_ch("Intro", 0, 60), _ch("Main", 60, 120)]
        # 50-70: 10s overlap with Intro (50-60), 10s overlap with Main (60-70) -> tie, first wins
        # Use 40-70: 20s overlap with Intro (40-60), 10s overlap with Main (60-70)
        segments = [_seg(40, 70)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (0, "Intro")

    def test_segment_spanning_boundary_more_in_second(self):
        """Segment spanning boundary maps to chapter with more overlap."""
        chapters = [_ch("Intro", 0, 60), _ch("Main", 60, 120)]
        # 50-90: 10s in Intro (50-60), 30s in Main (60-90)
        segments = [_seg(50, 90)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (1, "Main")

    def test_segment_outside_all_chapters(self):
        """Segment outside all chapters gets chapter_index=-1."""
        chapters = [_ch("Intro", 10, 60)]
        segments = [_seg(70, 90)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (-1, "")

    def test_segment_before_first_chapter(self):
        """Segment before first chapter gets -1."""
        chapters = [_ch("Main", 30, 90)]
        segments = [_seg(0, 20)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (-1, "")

    def test_empty_chapters_list(self):
        """Empty chapters list maps all segments to -1."""
        segments = [_seg(10, 50), _seg(60, 90)]
        result = map_segments_to_chapters(segments, [])
        assert result[0] == (-1, "")
        assert result[1] == (-1, "")

    def test_empty_segments_list(self):
        """Empty segments list returns empty dict."""
        chapters = [_ch("Intro", 0, 60)]
        result = map_segments_to_chapters([], chapters)
        assert result == {}

    def test_multiple_segments_mapped(self):
        """Multiple segments each map to correct chapter."""
        chapters = [
            _ch("Intro", 0, 60),
            _ch("Body", 60, 180),
            _ch("Conclusion", 180, 240),
        ]
        segments = [
            _seg(5, 30),    # -> Intro
            _seg(100, 150), # -> Body
            _seg(200, 230), # -> Conclusion
            _seg(250, 270), # -> outside
        ]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (0, "Intro")
        assert result[1] == (1, "Body")
        assert result[2] == (2, "Conclusion")
        assert result[3] == (-1, "")

    def test_chapter_with_none_end_time(self):
        """Chapter with None end_time is treated as open-ended."""
        chapters = [VideoChapter(title="Open", start_time=0, end_time=None)]
        segments = [_seg(100, 200)]
        result = map_segments_to_chapters(segments, chapters)
        assert result[0] == (0, "Open")


# ============================================================
# apply_chapter_topic_match tests
# ============================================================


@pytest.mark.fast
class TestApplyChapterTopicMatch:
    """Tests for MatchScoring.apply_chapter_topic_match()."""

    def setup_method(self):
        self.scoring = MatchScoring(config=None)

    def test_no_chapter_title_no_change(self):
        """No chapter title returns unchanged confidence."""
        seg = _seg(0, 10, text="ocean wildlife documentary")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, None, vo_chapter_index=0)
        assert conf == 0.7
        assert reason == ""

    def test_empty_chapter_title_no_change(self):
        """Empty chapter title returns unchanged confidence."""
        seg = _seg(0, 10, text="ocean wildlife documentary")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "", vo_chapter_index=0)
        assert conf == 0.7
        assert reason == ""

    def test_no_vo_chapter_index_no_change(self):
        """No vo_chapter_index (None) returns unchanged confidence."""
        seg = _seg(0, 10, text="ocean wildlife documentary")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "Ocean Wildlife", vo_chapter_index=None)
        assert conf == 0.7
        assert reason == ""

    def test_matching_single_keyword_boost(self):
        """Single keyword match gives +0.05 partial boost."""
        seg = _seg(0, 10, text="the ocean is vast and deep")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "Ocean Exploration", vo_chapter_index=0)
        assert conf == pytest.approx(0.75)
        assert "partial match" in reason
        assert "ocean" in reason

    def test_matching_multiple_keywords_strong_boost(self):
        """3+ keyword matches give +0.10 strong boost."""
        seg = _seg(0, 10, text="ocean wildlife marine biology documentary")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "Ocean Wildlife Marine Documentary", vo_chapter_index=0)
        assert conf == pytest.approx(0.80)
        assert "+0.1" in reason

    def test_no_keyword_overlap_penalty(self):
        """No keyword overlap gives -0.05 mismatch penalty."""
        seg = _seg(0, 10, text="technology and innovation trends")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "Cooking Recipes", vo_chapter_index=0)
        assert conf == pytest.approx(0.65)
        assert "mismatch" in reason

    def test_stopwords_excluded(self):
        """Stopwords are excluded from matching."""
        seg = _seg(0, 10, text="the and for are but not")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "The And For", vo_chapter_index=0)
        # All words are stopwords or < 3 chars, so no keywords extracted -> no change
        assert conf == 0.7
        assert reason == ""

    def test_short_words_excluded(self):
        """Words shorter than 3 chars are excluded."""
        seg = _seg(0, 10, text="AI is an ML model")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "AI ML", vo_chapter_index=0)
        # "ai" and "ml" are only 2 chars -> excluded
        assert conf == 0.7
        assert reason == ""

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        seg = _seg(0, 10, text="OCEAN WILDLIFE documentary")
        conf, reason = self.scoring.apply_chapter_topic_match(0.7, seg, "ocean wildlife", vo_chapter_index=0)
        assert conf == pytest.approx(0.75)
        assert "partial match" in reason

    def test_recorded_in_confidence_breakdown(self):
        """Chapter topic match is recorded in confidence_breakdown."""
        scoring = MatchScoring(config=_mock_config())
        vo = _seg(0, 10, text="ocean wildlife documentary")
        vid = _seg(0, 10, text="some video caption")
        conf, reason, breakdown = scoring.apply_all_adjustments(
            0.7, vo, vid, chapter_title="Ocean Wildlife", current_chapter_index=0
        )
        chapter_entries = [b for b in breakdown if b['component'] == 'chapter_topic_match']
        assert len(chapter_entries) == 1
        assert chapter_entries[0]['adjustment'] == pytest.approx(0.05)

    def test_no_chapter_title_no_breakdown_entry(self):
        """No chapter_title produces no chapter_topic_match breakdown entry."""
        scoring = MatchScoring(config=_mock_config())
        vo = _seg(0, 10, text="ocean wildlife documentary")
        vid = _seg(0, 10, text="some video caption")
        conf, reason, breakdown = scoring.apply_all_adjustments(
            0.7, vo, vid
        )
        chapter_entries = [b for b in breakdown if b['component'] == 'chapter_topic_match']
        assert len(chapter_entries) == 0
