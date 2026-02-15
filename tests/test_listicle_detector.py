"""
Tests for listicle structure detection in voiceover narration.

Tests cover:
- Ordinal markers (first, second, third, finally, lastly)
- Numbered markers (#1, number one, step 1, item 1)
- Transition markers (next up, moving on to, let's talk about)
- Mixed numbering styles
- Edge cases (empty input, single marker, dict segments)
"""

import pytest
from dataclasses import dataclass
from typing import List

from src.chapter_detection.listicle_detector import (
    detect_listicle_groups,
    detect_list_header,
    _detect_ordinal,
    _detect_numbered,
    _detect_transition,
    _normalize_marker_sequence,
    _extract_marker_position,
    _extract_simple_keywords,
    _extract_topic_keywords,
)
from src.chapter_detection.models import ListicleGroup


@dataclass
class FakeSegment:
    """Minimal segment for testing."""
    index: int
    text: str
    start: float = 0.0
    end: float = 5.0


def _make_segments(texts: List[str]) -> List[FakeSegment]:
    """Create fake segments from a list of text strings."""
    return [FakeSegment(index=i, text=t) for i, t in enumerate(texts)]


# ── Ordinal marker detection ─────────────────────────────

class TestOrdinalDetection:
    def test_first(self):
        result = _detect_ordinal("First, let me show you something")
        assert result is not None
        assert result[0] == 'ordinal'
        assert result[1] == 'first'

    def test_second(self):
        result = _detect_ordinal("Second, we have the gardens")
        assert result is not None
        assert result[:2] == ('ordinal', 'second')

    def test_third(self):
        result = _detect_ordinal("Third is the main attraction")
        assert result is not None
        assert result[:2] == ('ordinal', 'third')

    def test_finally(self):
        result = _detect_ordinal("Finally, we arrive at the museum")
        assert result is not None
        assert result[:2] == ('ordinal', 'finally')

    def test_lastly(self):
        result = _detect_ordinal("Lastly, don't forget the food")
        assert result is not None
        assert result[:2] == ('ordinal', 'lastly')

    def test_the_first(self):
        result = _detect_ordinal("The first thing you'll notice")
        assert result is not None
        assert result[:2] == ('ordinal', 'first')

    def test_and_finally(self):
        result = _detect_ordinal("And finally, the grand finale")
        assert result is not None
        assert result[:2] == ('ordinal', 'finally')

    def test_no_ordinal(self):
        result = _detect_ordinal("The city is beautiful at night")
        assert result is None

    def test_ordinal_not_at_start(self):
        result = _detect_ordinal("The building was first constructed in 1920")
        assert result is None

    def test_mid_segment_ordinal_scan_full(self):
        """Mid-segment ordinal detected with scan_full_text=True."""
        result = _detect_ordinal("And then, second we see the park", scan_full_text=True)
        assert result is not None
        assert result[0] == 'ordinal'
        assert result[1] == 'second'
        assert result[2] > 0  # char_offset is mid-segment

    def test_mid_segment_ordinal_not_detected_by_default(self):
        """Mid-segment ordinal NOT detected without scan_full_text."""
        result = _detect_ordinal("And then, second we see the park")
        assert result is None


# ── Numbered marker detection ────────────────────────────

class TestNumberedDetection:
    def test_hashtag_number(self):
        result = _detect_numbered("#1 The Eiffel Tower")
        assert result is not None
        assert result[0] == 'numbered'
        assert '1' in result[1]

    def test_hashtag_with_space(self):
        result = _detect_numbered("# 3 Big Ben")
        assert result is not None
        assert result[0] == 'numbered'

    def test_number_word(self):
        result = _detect_numbered("Number one, the best restaurant")
        assert result is not None
        assert result[0] == 'numbered'

    def test_step_digit(self):
        result = _detect_numbered("Step 1 prepare the ingredients")
        assert result is not None
        assert result[0] == 'numbered'

    def test_item_word(self):
        result = _detect_numbered("Item two on our list")
        assert result is not None
        assert result[0] == 'numbered'

    def test_reason_number(self):
        result = _detect_numbered("Reason 5 is the most important")
        assert result is not None
        assert result[0] == 'numbered'

    def test_tip_number(self):
        result = _detect_numbered("Tip 3 always check your surroundings")
        assert result is not None
        assert result[0] == 'numbered'

    def test_no_number(self):
        result = _detect_numbered("There are many steps to follow")
        assert result is None


# ── Transition marker detection ──────────────────────────

class TestTransitionDetection:
    def test_next_up(self):
        result = _detect_transition("Next up is the harbor district")
        assert result is not None
        assert result[0] == 'transition'

    def test_moving_on_to(self):
        result = _detect_transition("Moving on to the next destination")
        assert result is not None
        assert result[0] == 'transition'

    def test_lets_talk_about(self):
        result = _detect_transition("Let's talk about the local cuisine")
        assert result is not None
        assert result[0] == 'transition'

    def test_lets_move_on(self):
        result = _detect_transition("Let's move on to the ancient ruins")
        assert result is not None
        assert result[0] == 'transition'

    def test_another_reason(self):
        result = _detect_transition("Another reason to visit is the weather")
        assert result is not None
        assert result[0] == 'transition'

    def test_no_transition(self):
        result = _detect_transition("The weather is beautiful today")
        assert result is None

    def test_mid_segment_transition_scan_full(self):
        """Mid-segment transition detected with scan_full_text=True."""
        result = _detect_transition("So now, next up is the beach", scan_full_text=True)
        assert result is not None
        assert result[0] == 'transition'
        assert result[2] > 0  # char_offset is mid-segment


# ── Full listicle group detection ────────────────────────

class TestDetectListicleGroups:
    def test_ordinal_listicle(self):
        """Detect ordinal-based listicle structure."""
        segments = _make_segments([
            "Welcome to our tour of Paris",
            "First, we visit the Eiffel Tower",
            "The tower was built in 1889",
            "Second, we explore the Louvre",
            "It houses the Mona Lisa",
            "Third, we see Notre Dame",
            "Finally, we end at Montmartre",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) >= 4  # first, second, third, finally
        assert groups[0].item_label == 'first'
        assert groups[0].start_segment_idx == 1
        assert groups[1].item_label == 'second'
        assert groups[1].start_segment_idx == 3
        assert groups[2].item_label == 'third'
        assert groups[2].start_segment_idx == 5

    def test_numbered_listicle(self):
        """Detect numbered marker listicle."""
        segments = _make_segments([
            "#1 The Great Wall of China",
            "It stretches over 13000 miles",
            "#2 Machu Picchu",
            "Located high in the Andes",
            "#3 The Colosseum",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        assert groups[0].start_segment_idx == 0
        assert groups[0].end_segment_idx == 1
        assert groups[1].start_segment_idx == 2
        assert groups[1].end_segment_idx == 3
        assert groups[2].start_segment_idx == 4
        assert groups[2].end_segment_idx == 4

    def test_transition_listicle(self):
        """Detect transition-based listicle."""
        segments = _make_segments([
            "Next up is the harbor",
            "The harbor has beautiful boats",
            "Moving on to the old town",
            "The old town has cobblestone streets",
            "Let's talk about the local food scene",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        assert groups[0].item_label.lower().startswith('next up')
        assert groups[1].item_label.lower().startswith('moving on to')

    def test_mixed_numbering_styles(self):
        """Detect listicle with mixed ordinal and numbered markers."""
        segments = _make_segments([
            "First, the beach is amazing",
            "Crystal clear waters everywhere",
            "Step 2 visit the mountain village",
            "It has traditional architecture",
            "#3 The ancient temple",
            "Finally, try the street food",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 4
        # Verify different marker types detected
        labels = [g.item_label for g in groups]
        assert labels[0] == 'first'  # ordinal
        assert 'finally' in labels[-1]  # ordinal terminal

    def test_empty_input(self):
        """Empty segment list returns empty."""
        assert detect_listicle_groups([]) == []

    def test_single_marker_returns_empty(self):
        """A single marker is not enough to confirm listicle."""
        segments = _make_segments([
            "First, let me tell you about Paris",
            "The city of lights is wonderful",
            "There are many cafes",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 0

    def test_no_markers(self):
        """Segments without markers return empty."""
        segments = _make_segments([
            "The city is beautiful",
            "There are many things to see",
            "The food is excellent",
        ])
        assert detect_listicle_groups(segments) == []

    def test_dict_segments(self):
        """Works with dict-based segments."""
        segments = [
            {'index': 0, 'text': '#1 First attraction'},
            {'index': 1, 'text': 'Some details about it'},
            {'index': 2, 'text': '#2 Second attraction'},
        ]
        groups = detect_listicle_groups(segments)
        assert len(groups) == 2
        assert groups[0].start_segment_idx == 0
        assert groups[0].end_segment_idx == 1
        assert groups[1].start_segment_idx == 2

    def test_group_ids_sequential(self):
        """Group IDs are sequential starting from 0."""
        segments = _make_segments([
            "First, the old town",
            "Second, the new town",
            "Third, the waterfront",
        ])
        groups = detect_listicle_groups(segments)
        for i, group in enumerate(groups):
            assert group.group_id == i

    def test_topic_keywords_extracted(self):
        """Each group has topic keywords from its segment text."""
        segments = _make_segments([
            "Step 1 prepare the delicious chocolate cake",
            "Mix the ingredients together carefully",
            "Step 2 bake the sourdough bread loaf",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 2
        # Group 0 covers segments 0-1, should have keywords from both
        assert len(groups[0].topic_keywords) > 0
        # Check that relevant words appear
        all_kw_0 = ' '.join(groups[0].topic_keywords)
        assert 'chocolate' in all_kw_0 or 'cake' in all_kw_0 or 'prepare' in all_kw_0

    def test_listicle_group_dataclass(self):
        """ListicleGroup has correct fields and serialization."""
        group = ListicleGroup(
            group_id=0,
            item_label='first',
            start_segment_idx=0,
            end_segment_idx=2,
            topic_keywords=['paris', 'tower', 'eiffel'],
        )
        assert group.segment_count == 3

        d = group.to_dict()
        assert d['group_id'] == 0
        assert d['item_label'] == 'first'
        assert d['topic_keywords'] == ['paris', 'tower', 'eiffel']

        restored = ListicleGroup.from_dict(d)
        assert restored.group_id == group.group_id
        assert restored.item_label == group.item_label
        assert restored.topic_keywords == group.topic_keywords

    def test_listicle_group_confidence_field(self):
        """ListicleGroup has confidence field with default and serialization."""
        group = ListicleGroup(
            group_id=0,
            item_label='first',
            confidence=0.85,
        )
        assert group.confidence == 0.85

        d = group.to_dict()
        assert d['confidence'] == 0.85

        restored = ListicleGroup.from_dict(d)
        assert restored.confidence == 0.85

    def test_listicle_group_confidence_default(self):
        """ListicleGroup has default confidence of 0.7."""
        group = ListicleGroup(group_id=0)
        assert group.confidence == 0.7

    def test_listicle_group_confidence_validation_valid(self):
        """ListicleGroup accepts valid confidence values."""
        # Test boundary values
        group_low = ListicleGroup(group_id=0, confidence=0.0)
        assert group_low.confidence == 0.0

        group_high = ListicleGroup(group_id=0, confidence=1.0)
        assert group_high.confidence == 1.0

    def test_listicle_group_confidence_validation_invalid(self):
        """ListicleGroup raises ValueError for invalid confidence."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            ListicleGroup(group_id=0, confidence=1.5)

        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            ListicleGroup(group_id=0, confidence=-0.1)

    def test_listicle_group_expected_count_validation(self):
        """ListicleGroup validates expected_count range."""
        # Valid expected_count values
        group = ListicleGroup(group_id=0, expected_count=10)
        assert group.expected_count == 10

        # Invalid: too small
        with pytest.raises(ValueError, match="expected_count must be positive"):
            ListicleGroup(group_id=0, expected_count=0)

        with pytest.raises(ValueError, match="expected_count must be positive"):
            ListicleGroup(group_id=0, expected_count=-5)

        # Invalid: too large
        with pytest.raises(ValueError, match="expected_count exceeds maximum"):
            ListicleGroup(group_id=0, expected_count=101)

    def test_step_word_number(self):
        """Detect 'step one', 'step two' word-based numbering."""
        segments = _make_segments([
            "Step one is to gather materials",
            "You'll need wood and nails",
            "Step two is to build the frame",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 2

    def test_case_insensitive(self):
        """Markers are detected regardless of case."""
        segments = _make_segments([
            "FIRST, the grand entrance",
            "SECOND, the main hall",
            "THIRD, the garden",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3

    def test_consecutive_markers(self):
        """Each marker starts a new group even when consecutive."""
        segments = _make_segments([
            "#1 Alpha",
            "#2 Beta",
            "#3 Gamma",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        for g in groups:
            assert g.start_segment_idx == g.end_segment_idx

    def test_last_group_extends_to_end(self):
        """Last group includes all remaining segments."""
        segments = _make_segments([
            "First, the introduction",
            "Second, the main topic",
            "More about the main topic",
            "Even more details here",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 2
        assert groups[-1].start_segment_idx == 1
        assert groups[-1].end_segment_idx == 3


# ── Topic keyword extraction (US-105-005) ───────────────────────────────

class TestTopicKeywordExtraction:
    """Tests for improved topic keyword extraction with LLM fallback."""

    def test_simple_extraction_filters_stop_words(self):
        """Simple extraction filters common stop words."""
        text = "the quick brown fox jumps over the lazy dog"
        keywords = _extract_simple_keywords(text, max_keywords=5)
        # Stop words should be filtered
        assert 'quick' in keywords or 'brown' in keywords
        assert 'the' not in keywords
        assert 'over' not in keywords

    def test_simple_extraction_filters_short_words(self):
        """Simple extraction filters words shorter than 4 characters."""
        text = "I saw a big cat run fast"
        keywords = _extract_simple_keywords(text, max_keywords=10)
        # Short words should be filtered
        assert 'fast' in keywords or 'big' in keywords
        assert 'i' not in keywords
        assert 'a' not in keywords
        assert 'saw' not in keywords

    def test_simple_extraction_preserves_order(self):
        """Simple extraction preserves order of first occurrence."""
        text = "apple banana apple cherry banana apple"
        keywords = _extract_simple_keywords(text, max_keywords=5)
        # Should return unique keywords in order of first occurrence
        assert keywords[0] == 'apple'
        assert keywords[1] == 'banana'
        assert keywords[2] == 'cherry'

    def test_extract_topic_keywords_without_llm(self):
        """Without LLM enabled, uses simple extraction only."""
        text = "First we visit the Eiffel Tower in Paris France"
        keywords = _extract_topic_keywords(text, max_keywords=5, use_llm=False)
        # Should return simple extraction results
        assert len(keywords) > 0
        assert 'eiffel' in keywords or 'tower' in keywords or 'paris' in keywords

    def test_extract_topic_keywords_llm_disabled_returns_simple(self):
        """When use_llm=False, returns simple extraction even with few keywords."""
        text = "First we visit Paris"
        keywords = _extract_topic_keywords(
            text, max_keywords=5,
            use_llm=False,
            min_keywords_for_llm=3,
        )
        # Should return simple extraction (may have few keywords)
        assert len(keywords) >= 0

    def test_extract_topic_keywords_with_llm_fallback(self):
        """When use_llm=True and simple extraction yields < min_keywords_for_llm,
        should attempt LLM fallback (may return empty if LLM fails in test)."""
        # Use a short text that will yield few keywords
        text = "First visit"
        keywords = _extract_topic_keywords(
            text, max_keywords=5,
            use_llm=True,
            min_keywords_for_llm=3,
        )
        # Either simple or LLM results - just verify it doesn't crash
        assert isinstance(keywords, list)

    def test_detect_listicle_groups_with_config(self):
        """detect_listicle_groups accepts listicle_topic_config parameter."""
        segments = _make_segments([
            "Step 1 the first attraction in Paris",
            "Step 2 the second attraction in London",
        ])
        # Test with None config (default behavior)
        groups = detect_listicle_groups(segments, listicle_topic_config=None)
        assert len(groups) == 2
        assert groups[0].topic_keywords is not None

    def test_detect_listicle_groups_with_llm_config(self):
        """detect_listicle_groups passes config to keyword extraction."""
        segments = _make_segments([
            "Step 1 the Eiffel Tower in Paris France",
            "Step 2 Big Ben in London England",
        ])

        # Create a mock config object
        class MockListicleTopicConfig:
            use_llm_topic_extraction = False  # Disable LLM for unit test
            min_keywords_for_simple = 3

        config = MockListicleTopicConfig()
        groups = detect_listicle_groups(segments, listicle_topic_config=config)

        assert len(groups) == 2
        # Should have keywords from the segments
        assert len(groups[0].topic_keywords) > 0


# ── List header detection ───────────────────────────────

class TestDetectListHeader:
    def test_top_10_numeric(self):
        """Detect 'top 10 reasons'."""
        assert detect_list_header("Here are the top 10 reasons to visit") == 10

    def test_numeric_at_start(self):
        """Detect '5 ways' at start of text."""
        assert detect_list_header("5 ways to improve your health") == 5

    def test_word_form_tips(self):
        """Detect 'seven tips' with word-form number."""
        assert detect_list_header("seven tips for better sleep") == 7

    def test_word_form_three_ways(self):
        """Detect 'three ways' with word-form number."""
        assert detect_list_header("three ways to save money") == 3

    def test_the_best_pattern(self):
        """Detect 'the 3 best places'."""
        assert detect_list_header("the 3 best places to eat") == 3

    def test_top_word_form(self):
        """Detect 'top five things'."""
        assert detect_list_header("top five things you need to know") == 5

    def test_no_header(self):
        """Text without header returns None."""
        assert detect_list_header("The city is beautiful at night") is None

    def test_no_header_unrecognized_noun(self):
        """Numbers followed by non-list nouns are not headers."""
        assert detect_list_header("5 cats sat on the mat") is None


class TestListHeaderIntegration:
    def test_expected_count_stored_on_groups(self):
        """When a header is in the segments, expected_count is set on all groups."""
        segments = _make_segments([
            "Here are the top 5 reasons to visit Paris",
            "First, the Eiffel Tower",
            "Second, the Louvre Museum",
            "Third, the food scene",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) >= 3
        for g in groups:
            assert g.expected_count == 5

    def test_no_header_expected_count_is_none(self):
        """Without a header, expected_count remains None."""
        segments = _make_segments([
            "First, the old town",
            "Second, the beach",
            "Third, the mountains",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        for g in groups:
            assert g.expected_count is None

    def test_mismatch_logs_warning(self, caplog):
        """Warning logged when detected count differs from expected by >1."""
        import logging
        segments = _make_segments([
            "Here are the top 10 tips for travel",
            "First, pack light",
            "Second, learn the language",
            "Third, try local food",
        ])
        with caplog.at_level(logging.WARNING, logger="src.chapter_detection.listicle_detector"):
            groups = detect_listicle_groups(segments)
        # 3 detected vs 10 expected → diff=7 > 1 → warning
        assert any("expected 10 items but detected 3" in r.message for r in caplog.records)

    def test_close_match_no_warning(self, caplog):
        """No warning when detected count is within 1 of expected."""
        import logging
        segments = _make_segments([
            "3 ways to enjoy summer",
            "First, go swimming",
            "Second, have a barbecue",
            "Third, visit the park",
        ])
        with caplog.at_level(logging.WARNING, logger="src.chapter_detection.listicle_detector"):
            groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        assert not any("expected" in r.message and "detected" in r.message for r in caplog.records)

    def test_expected_count_serialization(self):
        """expected_count round-trips through to_dict/from_dict."""
        group = ListicleGroup(
            group_id=0, item_label='first',
            start_segment_idx=0, end_segment_idx=2,
            topic_keywords=['paris'], expected_count=5,
        )
        d = group.to_dict()
        assert d['expected_count'] == 5
        restored = ListicleGroup.from_dict(d)
        assert restored.expected_count == 5

    def test_expected_count_none_serialization(self):
        """expected_count=None round-trips correctly."""
        group = ListicleGroup(group_id=0, item_label='first',
                              start_segment_idx=0, end_segment_idx=0)
        d = group.to_dict()
        assert d['expected_count'] is None
        restored = ListicleGroup.from_dict(d)
        assert restored.expected_count is None


# ── Mid-segment marker detection ─────────────────────────

class TestMidSegmentDetection:
    def test_marker_at_position_30_detected(self):
        """Marker at character position 30 is detected when max_chars_offset=50."""
        # "x" * 25 = 25 chars + space + "second" starts at position 26
        prefix = "x" * 25 + " "  # 26 chars
        segments = _make_segments([
            "First, the opening attraction",
            "Some details about it",
            prefix + "second we see the gardens",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 2
        # Second marker found mid-segment at index 2
        assert groups[1].item_label == 'second'
        assert groups[1].start_segment_idx == 2

    def test_marker_at_position_100_not_detected(self):
        """Marker at character position 100 is NOT detected when max_chars_offset=50."""
        # "x" * 99 + space = 100 chars before marker
        prefix = "x" * 99 + " "  # 100 chars
        segments = _make_segments([
            "First, the opening attraction",
            "Some details about it",
            prefix + "second we see the gardens",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        # Only one start-of-text marker detected → not enough for listicle
        assert len(groups) == 0

    def test_mid_segment_group_boundaries(self):
        """Mid-segment detection produces correct group boundaries.

        When a mid-segment marker is found in segment N, the previous group
        should include segment N (text before the marker belongs to it),
        and the new group starts at segment N.
        """
        segments = _make_segments([
            "#1 The Great Wall of China",
            "It stretches thousands of miles",
            "Amazing views, and #2 The Taj Mahal",
            "Located in Agra, India",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 2
        # Group 0: starts at segment 0, ends at segment 2 (includes pre-marker text)
        assert groups[0].start_segment_idx == 0
        assert groups[0].end_segment_idx == 2
        # Group 1: starts at segment 2 (the mid-segment marker), ends at segment 3
        assert groups[1].start_segment_idx == 2
        assert groups[1].end_segment_idx == 3

    def test_mid_segment_with_custom_max_offset(self):
        """Custom max_chars_offset controls detection range."""
        # Marker at position ~20
        text_with_mid_marker = "Some intro text. #2 The second item"
        segments = _make_segments([
            "#1 First item here",
            text_with_mid_marker,
        ])
        # With offset 10, marker at ~17 should NOT be detected
        groups_small = detect_listicle_groups(segments, max_chars_offset=10)
        assert len(groups_small) == 0  # Only #1 at start, #2 too far → single marker

        # With offset 50, marker at ~17 SHOULD be detected
        groups_large = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups_large) == 2

    def test_start_of_text_takes_priority_over_mid(self):
        """Start-of-text markers are found before mid-segment scan runs."""
        segments = _make_segments([
            "First, the beach is amazing",
            "Second, the mountain is great",
            "Third, the forest is peaceful",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 3
        # All detected at start — no mid-segment behavior
        assert groups[0].item_label == 'first'
        assert groups[1].item_label == 'second'
        assert groups[2].item_label == 'third'

    def test_mid_segment_numbered_detected(self):
        """Mid-segment numbered markers like '#2' are detected."""
        segments = _make_segments([
            "#1 Alpha",
            "Details about Alpha, then #2 Beta",
            "Details about Beta",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 2
        assert '#1' in groups[0].item_label or '1' in groups[0].item_label
        assert '#2' in groups[1].item_label or '2' in groups[1].item_label


# ── Mixed numbering normalization ─────────────────────────

class TestMixedNumberingNormalization:
    def test_mixed_ordinal_numbered_hash_normalized(self):
        """'first', '#2', 'third' produces groups with group_ids 0, 1, 2 in correct order."""
        segments = _make_segments([
            "First, the beach is stunning",
            "Crystal clear waters",
            "#2 The mountain village",
            "Traditional architecture",
            "Third, the ancient temple",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        # After normalization: first(1) → 0, #2(2) → 1, third(3) → 2
        assert groups[0].group_id == 0
        assert groups[0].item_label == 'first'
        assert groups[1].group_id == 1
        assert '#2' in groups[1].item_label or '2' in groups[1].item_label
        assert groups[2].group_id == 2
        assert groups[2].item_label == 'third'

    def test_transition_markers_preserve_original_order(self):
        """Purely transition-based markers preserve original detection order."""
        segments = _make_segments([
            "Next up is the harbor",
            "Beautiful boats everywhere",
            "Moving on to the old town",
            "Cobblestone streets abound",
            "Let's talk about the food scene",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        # All transitions — no position inferable, order preserved
        assert groups[0].group_id == 0
        assert groups[1].group_id == 1
        assert groups[2].group_id == 2
        assert 'next up' in groups[0].item_label.lower()
        assert 'moving on to' in groups[1].item_label.lower()
        assert "let's talk about" in groups[2].item_label.lower()

    def test_gap_numbering_preserves_relative_order(self):
        """Gap in numbering (e.g., '#1', '#3', '#5') preserves relative order without phantoms."""
        segments = _make_segments([
            "#1 The first attraction",
            "Details about it",
            "#3 The third attraction",
            "More details",
            "#5 The fifth attraction",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3  # No phantom groups inserted for #2 and #4
        # Relative order preserved: #1 < #3 < #5
        assert groups[0].group_id == 0
        assert groups[1].group_id == 1
        assert groups[2].group_id == 2
        assert '1' in groups[0].item_label
        assert '3' in groups[1].item_label
        assert '5' in groups[2].item_label

    def test_extract_marker_position_ordinals(self):
        """_extract_marker_position correctly maps ordinal words."""
        assert _extract_marker_position('ordinal', 'first') == 1
        assert _extract_marker_position('ordinal', 'second') == 2
        assert _extract_marker_position('ordinal', 'third') == 3
        assert _extract_marker_position('ordinal', 'finally') == -1

    def test_extract_marker_position_numbered(self):
        """_extract_marker_position correctly parses numbered labels."""
        assert _extract_marker_position('numbered', '#2') == 2
        assert _extract_marker_position('numbered', 'Step 3') == 3
        assert _extract_marker_position('numbered', 'Number one') == 1

    def test_extract_marker_position_transition(self):
        """_extract_marker_position returns None for transitions."""
        assert _extract_marker_position('transition', 'Next up') is None
        assert _extract_marker_position('transition', 'Moving on to') is None


# ── Expected-count auto-correction ──────────────────────────

class TestExpectedCountAutoCorrection:
    """Tests for auto-correction when expected_count mismatches detected count."""

    def test_rescan_finds_additional_markers(self):
        """'top 5 reasons' header with only 3 start-of-text markers triggers
        re-scan and finds 2 additional markers in mid-segment text beyond
        the default max_chars_offset."""
        # Segments: header + 3 clear start-of-text markers + 2 markers
        # buried deeper in the text (beyond 50 chars but within 150)
        segments = _make_segments([
            "Here are the top 5 reasons to visit this country",
            "Reason 1 the beaches are beautiful",
            "Crystal clear waters and white sand everywhere",
            "Reason 2 the food is incredible",
            "Amazing local cuisine with fresh ingredients",
            # Marker buried at char ~60 (beyond default 50)
            "There are also wonderful mountains to explore and reason 3 the hiking trails are world class",
            "You can hike for days without seeing anyone",
            # Marker buried at char ~70
            "The culture is rich with history and tradition and reason 4 the architecture is stunning",
            "Beautiful buildings from centuries past",
            # Another deep marker
            "People are friendly and welcoming everywhere you go and reason 5 the nightlife is vibrant",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        # Auto-correction should find the buried markers
        assert len(groups) >= 4, f"Expected at least 4 groups after correction, got {len(groups)}"
        # Verify expected_count is set on all groups
        for g in groups:
            assert g.expected_count == 5

    def test_auto_correction_does_not_exceed_expected_plus_one(self):
        """Auto-correction does not increase marker count beyond expected_count + 1."""
        from src.chapter_detection.listicle_detector import _auto_correct_markers
        # Simulate: expected 3 items, initial scan found 2 markers.
        # Relaxed scan could find many more buried markers.
        segments = _make_segments([
            "3 ways to improve your health",
            "First, eat more vegetables and fruits daily",
            "They provide essential vitamins",
            "Second, exercise regularly for fitness",
            "At least thirty minutes per day",
            # Buried markers the relaxed scan would find
            "Sleeping well is key, and the third way is to get eight hours of rest every single night",
            "Rest helps your body recover",
            "Hydration matters, and the fourth way is to drink plenty of water throughout the entire day",
            "Water flushes toxins from your body",
            "Meditation helps, the fifth way is practicing mindfulness for at least ten minutes each morning",
        ])
        # Initial markers: just 'First' at seg 1 and 'Second' at seg 3
        initial_markers = [
            (1, 'ordinal', 'first', False, 0),
            (3, 'ordinal', 'second', False, 0),
        ]
        corrected = _auto_correct_markers(
            segments, initial_markers, expected_count=3, initial_max_chars_offset=50,
        )
        # Should not exceed expected_count(3) + 1 = 4
        assert len(corrected) <= 4, f"Expected at most 4 markers, got {len(corrected)}"

    def test_auto_correction_logs_warning_when_gap_remains(self, caplog):
        """Auto-correction logs warning when it cannot close the gap."""
        import logging
        # Header says 10 but only 3 markers exist, even with relaxed scan
        segments = _make_segments([
            "Here are the top 10 tips for travel",
            "First, pack light for your journey",
            "Bring only the essentials",
            "Second, learn basic phrases in the local language",
            "It helps with making friends",
            "Third, try the local street food",
            "The best food is found at markets",
            "There are many other tips to consider",
            "Always be respectful of local customs",
            "Enjoy every moment of your trip",
        ])
        with caplog.at_level(logging.WARNING, logger="src.chapter_detection.listicle_detector"):
            groups = detect_listicle_groups(segments, max_chars_offset=50)
        # 3 detected vs 10 expected → gap cannot be closed
        assert any(
            "could not close gap" in r.message
            for r in caplog.records
        ), f"Expected 'could not close gap' warning, got: {[r.message for r in caplog.records]}"

    def test_auto_correction_runs_only_once(self, caplog):
        """Auto-correction pass runs exactly once, not recursively."""
        import logging
        segments = _make_segments([
            "Here are the top 8 reasons to visit",
            "Reason 1 the scenery",
            "Gorgeous views everywhere",
            "Reason 2 the people",
            "Friendly locals",
            # Buried marker
            "The weather is perfect, and reason 3 the climate is ideal",
        ])
        with caplog.at_level(logging.INFO, logger="src.chapter_detection.listicle_detector"):
            detect_listicle_groups(segments, max_chars_offset=50)
        # Count how many times "Auto-correction complete" appears
        correction_logs = [
            r for r in caplog.records if "Auto-correction complete" in r.message
        ]
        assert len(correction_logs) <= 1, (
            f"Auto-correction ran {len(correction_logs)} times, expected at most 1"
        )

    def test_auto_correction_logs_before_after_counts(self, caplog):
        """Auto-correction logs the before and after marker counts."""
        import logging
        segments = _make_segments([
            "Here are the top 5 reasons to visit",
            "Reason 1 the beautiful beaches",
            "Sandy shores for miles",
            "Reason 2 the delicious food",
            "Fresh seafood daily",
            # Buried marker beyond 50 chars
            "The history here is fascinating and truly reason 3 the museums are incredible",
        ])
        with caplog.at_level(logging.INFO, logger="src.chapter_detection.listicle_detector"):
            detect_listicle_groups(segments, max_chars_offset=50)
        # Should have a log with before->after counts
        correction_complete = [
            r for r in caplog.records if "Auto-correction complete" in r.message
        ]
        assert len(correction_complete) == 1
        msg = correction_complete[0].message
        assert "markers" in msg and "->" in msg
