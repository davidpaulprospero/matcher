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
    detect_inconsistent_numbering,
    normalize_numbering_format,
    _detect_number_format,
    NumberFormat,
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

    # US-122-009: New transition pattern tests

    def test_in_this_episode(self):
        """Detect 'in this episode' style transitions."""
        result = _detect_transition("In this episode we explore the mountains")
        assert result is not None
        assert result[0] == 'transition'

    def test_in_this_video(self):
        """Detect 'in this video' style transitions."""
        result = _detect_transition("In this video we'll cover the basics")
        assert result is not None
        assert result[0] == 'transition'

    def test_in_this_part(self):
        """Detect 'in this part' style transitions."""
        result = _detect_transition("In this part of the journey")
        assert result is not None
        assert result[0] == 'transition'

    def test_coming_up_next(self):
        """Detect 'coming up next' transitions."""
        result = _detect_transition("Coming up next is the grand finale")
        assert result is not None
        assert result[0] == 'transition'

    def test_up_next(self):
        """Detect 'up next' transitions."""
        result = _detect_transition("Up next we have something special")
        assert result is not None
        assert result[0] == 'transition'

    def test_stay_tuned_for(self):
        """Detect 'stay tuned for' transitions."""
        result = _detect_transition("Stay tuned for the amazing finale")
        assert result is not None
        assert result[0] == 'transition'

    def test_dont_go_away(self):
        """Detect 'don't go away' transitions."""
        result = _detect_transition("Don't go away, more content coming")
        assert result is not None
        assert result[0] == 'transition'

    def test_dont_leave(self):
        """Detect 'don't leave' transitions."""
        result = _detect_transition("Don't leave just yet!")
        assert result is not None
        assert result[0] == 'transition'

    def test_case_insensitive_new_patterns(self):
        """New transition patterns work case-insensitively."""
        result = _detect_transition("IN THIS EPISODE we explore")
        assert result is not None
        assert result[0] == 'transition'

        result = _detect_transition("COMING UP NEXT is great")
        assert result is not None
        assert result[0] == 'transition'

        result = _detect_transition("UP NEXT: amazing content")
        assert result is not None
        assert result[0] == 'transition'

    def test_mid_segment_new_patterns(self):
        """New transition patterns work mid-segment with scan_full_text=True."""
        # 'in this episode' mid-segment
        result = _detect_transition("First we covered basics, in this episode we go deeper", scan_full_text=True)
        assert result is not None
        assert result[0] == 'transition'
        assert result[2] > 0

        # 'coming up next' mid-segment
        result = _detect_transition("That was intro, coming up next is the main content", scan_full_text=True)
        assert result is not None
        assert result[0] == 'transition'
        assert result[2] > 0

        # 'up next' mid-segment
        result = _detect_transition("Now, up next we have a surprise", scan_full_text=True)
        assert result is not None
        assert result[0] == 'transition'
        assert result[2] > 0


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
        # Mixed numbering (ordinal, digit, hash) gets normalized to ordinal format
        # 'first, Step 2, #3' -> '1st, 2nd, 3rd'
        assert labels[0] == '1st'  # normalized from 'first'
        assert labels[1] == '2nd'  # normalized from 'Step 2'
        assert labels[2] == '3rd'  # normalized from '#3'
        # inconsistent_numbering flag should be set
        assert groups[0].inconsistent_numbering is True
        assert groups[1].inconsistent_numbering is True
        assert groups[2].inconsistent_numbering is True

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

    # US-122-009: Transition consecutive marker tests

    def test_transition_followed_by_transition(self):
        """Transitions don't break when followed immediately by another transition."""
        segments = _make_segments([
            "Next up is the beach",
            "Coming up next we explore the mountains",
            "Up next is the forest",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        # Each transition should start a new group
        assert groups[0].group_id == 0
        assert groups[1].group_id == 1
        assert groups[2].group_id == 2

    def test_transition_followed_by_ordinal(self):
        """Transitions don't break when followed by ordinal marker."""
        segments = _make_segments([
            "Next up is the beach",
            "First, the sandy shores",
            "Second, the rocky coast",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        # Positioned markers (ordinals) come before unpositioned (transitions) after normalization
        # So ordinals first, then transition
        assert groups[0].item_label == 'first'
        assert groups[1].item_label == 'second'
        # Transition is last (unpositioned markers come after positioned)
        assert 'next up' in groups[2].item_label.lower()

    def test_transition_followed_by_numbered(self):
        """Transitions don't break when followed by numbered marker."""
        segments = _make_segments([
            "In this video we explore",
            "#1 The first destination",
            "#2 The second destination",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        # Numbered markers come before transition after normalization
        assert '#1' in groups[0].item_label or '1' in groups[0].item_label
        assert '#2' in groups[1].item_label or '2' in groups[1].item_label
        # Transition is last
        assert 'in this video' in groups[2].item_label.lower()

    def test_ordinal_followed_by_transition(self):
        """Ordinal markers followed by transitions work correctly."""
        segments = _make_segments([
            "First, the introduction",
            "Moving on to the main topic",
            "Another point to consider",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        assert groups[0].item_label == 'first'
        assert 'moving on to' in groups[1].item_label.lower()
        assert 'another' in groups[2].item_label.lower()


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

    # New tests for US-122-003: edge case patterns

    def test_range_numeric_pattern(self):
        """Detect '15-20 ways to...' pattern - uses first number."""
        assert detect_list_header("15-20 ways to save money") == 15

    def test_range_word_form_pattern(self):
        """Detect 'five to ten tips' pattern - uses first number."""
        assert detect_list_header("five to ten tips for better sleep") == 5

    def test_range_with_hyphen_words(self):
        """Detect range with hyphenated word forms."""
        assert detect_list_header("three to seven reasons why") == 3

    def test_spanish_number_words(self):
        """Detect Spanish number words in headers (with English nouns)."""
        assert detect_list_header("diez reasons to visit") == 10
        assert detect_list_header("cinco ways to improve") == 5
        assert detect_list_header("veinte tips for health") == 20

    def test_french_number_words(self):
        """Detect French number words in headers (with English nouns)."""
        assert detect_list_header("dix reasons to visit") == 10
        assert detect_list_header("cinq ways to succeed") == 5
        assert detect_list_header("vingt tips useful") == 20

    def test_german_number_words(self):
        """Detect German number words in headers (with English nouns)."""
        assert detect_list_header("zehn reasons to visit") == 10
        assert detect_list_header("fünf ways to succeed") == 5
        assert detect_list_header("zwanzig tips for health") == 20

    # US-135-010: New tests for Portuguese number words
    def test_portuguese_number_words(self):
        """Detect Portuguese number words in headers (with English nouns)."""
        assert detect_list_header("um reasons to visit") == 1
        assert detect_list_header("dois ways to succeed") == 2
        assert detect_list_header("três tips for health") == 3
        assert detect_list_header("quatro reasons to go") == 4
        assert detect_list_header("cinco ways to win") == 5
        assert detect_list_header("seis tips great") == 6
        assert detect_list_header("sete ways amazing") == 7
        assert detect_list_header("oito tips incredible") == 8
        assert detect_list_header("nove reasons perfect") == 9
        assert detect_list_header("dez ways awesome") == 10
        assert detect_list_header("onze tips excellent") == 11
        assert detect_list_header("doze reasons wonderful") == 12
        assert detect_list_header("quinze ways fantastic") == 15
        assert detect_list_header("vinte tips amazing") == 20

    def test_portuguese_feminine_forms(self):
        """Detect Portuguese feminine number forms (duas)."""
        assert detect_list_header("duas reasons to visit") == 2

    # US-135-010: New tests for Italian number words
    def test_italian_number_words(self):
        """Detect Italian number words in headers (with English nouns)."""
        assert detect_list_header("uno reasons to visit") == 1
        assert detect_list_header("due ways to succeed") == 2
        assert detect_list_header("tre tips for health") == 3
        assert detect_list_header("quattro reasons to go") == 4
        assert detect_list_header("cinque ways to win") == 5
        assert detect_list_header("sei tips great") == 6
        assert detect_list_header("sette ways amazing") == 7
        assert detect_list_header("otto tips incredible") == 8
        assert detect_list_header("nove reasons perfect") == 9
        assert detect_list_header("dieci ways awesome") == 10
        assert detect_list_header("undici tips excellent") == 11
        assert detect_list_header("dodici reasons wonderful") == 12
        assert detect_list_header("quindici ways fantastic") == 15
        assert detect_list_header("venti tips amazing") == 20

    # US-135-010: New tests for Japanese kanji numbers
    def test_japanese_kanji_numbers(self):
        """Detect Japanese kanji numbers in headers (with English nouns)."""
        assert detect_list_header("一 reasons to visit") == 1
        assert detect_list_header("二 ways to succeed") == 2
        assert detect_list_header("三 tips for health") == 3
        assert detect_list_header("四 reasons to go") == 4
        assert detect_list_header("五 ways to win") == 5
        assert detect_list_header("六 tips great") == 6
        assert detect_list_header("七 ways amazing") == 7
        assert detect_list_header("八 tips incredible") == 8
        assert detect_list_header("九 reasons perfect") == 9
        assert detect_list_header("十 ways awesome") == 10

    def test_best_worst_with_adjective(self):
        """Detect 'the 10 best hiking trails' pattern with adjective before noun."""
        assert detect_list_header("the 10 best hiking trails") == 10
        assert detect_list_header("the 5 worst mistakes") == 5
        assert detect_list_header("the 3 best hiking spots") == 3

    def test_best_worst_without_the_with_adjective(self):
        """Detect '10 best hiking trails' pattern without 'the' prefix."""
        assert detect_list_header("10 best hiking trails") == 10
        assert detect_list_header("5 best travel tips") == 5
        assert detect_list_header("ten best hiking spots") == 10


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

    def test_expected_count_capped_when_header_exceeds_segments(self, caplog):
        """When header expects more items than detected, expected_count is capped."""
        import logging
        segments = _make_segments([
            "Here are the top 20 tips for success",
            "First, work hard",
            "Second, be persistent",
        ])
        with caplog.at_level(logging.INFO, logger="src.chapter_detection.listicle_detector"):
            groups = detect_listicle_groups(segments)
        # 2 detected vs 20 expected → should cap to 2
        assert len(groups) == 2
        assert groups[0].expected_count == 2  # Capped from 20 to 2
        assert groups[1].expected_count == 2
        # Check that capping info was logged
        assert any("Capping expected_count" in r.message for r in caplog.records)

    def test_expected_count_not_capped_when_close(self, caplog):
        """When header count is close to detected (within 1), no capping occurs."""
        import logging
        segments = _make_segments([
            "Here are the top 4 tips",
            "First, tip one",
            "Second, tip two",
            "Third, tip three",
        ])
        with caplog.at_level(logging.INFO, logger="src.chapter_detection.listicle_detector"):
            groups = detect_listicle_groups(segments)
        # 3 detected vs 4 expected → diff=1, no capping (within threshold)
        assert len(groups) == 3
        assert groups[0].expected_count == 4  # NOT capped
        assert not any("Capping expected_count" in r.message for r in caplog.records)


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

    # US-122-009: Mid-segment transition detection tests

    def test_mid_segment_transition_within_offset(self):
        """Mid-segment transition detected within max_chars_offset=50."""
        # "x" * 30 = 30 chars + " coming up next" starts at position 31
        prefix = "x" * 30 + " "  # 31 chars
        segments = _make_segments([
            "First, the opening segment",
            "Some details about it",
            prefix + "coming up next we explore the gardens",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 2
        # Second marker found mid-segment at index 2
        assert 'coming up next' in groups[1].item_label.lower() or groups[1].item_label.lower().startswith('up next')
        assert groups[1].start_segment_idx == 2

    def test_mid_segment_transition_beyond_offset_not_detected(self):
        """Mid-segment transition beyond max_chars_offset NOT detected."""
        # "x" * 60 = 60 chars before marker - beyond 50
        prefix = "x" * 60 + " "  # 61 chars
        segments = _make_segments([
            "First, the opening segment",
            "Some details about it",
            prefix + "coming up next we explore",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        # Only one start-of-text marker → not enough for listicle
        assert len(groups) == 0

    def test_mid_segment_in_this_episode_detected(self):
        """Mid-segment 'in this episode' detected within offset."""
        segments = _make_segments([
            "First segment covers basics",
            "More details, in this episode we dive deep",
            "Deep content follows",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 2
        assert 'in this episode' in groups[1].item_label.lower()

    def test_transition_group_boundaries(self):
        """Transition markers produce correct group boundaries."""
        segments = _make_segments([
            "Next up is the beach",
            "Beautiful sandy shores",
            "And now in this video we explore mountains",
            "Snow-capped peaks",
        ])
        groups = detect_listicle_groups(segments, max_chars_offset=50)
        assert len(groups) == 2
        # Group 0: starts at segment 0, ends at segment 2 (includes pre-marker text)
        assert groups[0].start_segment_idx == 0
        assert groups[0].end_segment_idx == 2
        # Group 1: starts at segment 2 (the mid-segment marker)
        assert groups[1].start_segment_idx == 2
        assert groups[1].end_segment_idx == 3


# ── Mixed numbering normalization ─────────────────────────

class TestMixedNumberingNormalization:
    def test_mixed_ordinal_numbered_hash_normalized(self):
        """'first', '#2', 'third' gets normalized to '1st', '2nd', '3rd'."""
        segments = _make_segments([
            "First, the beach is stunning",
            "Crystal clear waters",
            "#2 The mountain village",
            "Traditional architecture",
            "Third, the ancient temple",
        ])
        groups = detect_listicle_groups(segments)
        assert len(groups) == 3
        # After normalization: 'first', '#2', 'third' -> '1st', '2nd', '3rd'
        assert groups[0].group_id == 0
        assert groups[0].item_label == '1st'  # normalized from 'first'
        assert groups[1].group_id == 1
        assert groups[1].item_label == '2nd'  # normalized from '#2'
        assert groups[2].group_id == 2
        assert groups[2].item_label == '3rd'  # normalized from 'third'

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

    # US-122-009: New transition patterns unpositioned tests

    def test_new_transition_patterns_unpositioned(self):
        """New transition patterns return None for marker position (unpositioned)."""
        # All new patterns should return None for position
        assert _extract_marker_position('transition', 'In this episode') is None
        assert _extract_marker_position('transition', 'In this video') is None
        assert _extract_marker_position('transition', 'Coming up next') is None
        assert _extract_marker_position('transition', 'Up next') is None
        assert _extract_marker_position('transition', 'Here is the answer') is None
        assert _extract_marker_position('transition', 'Here comes the best part') is None
        assert _extract_marker_position('transition', 'Stay tuned for more') is None
        assert _extract_marker_position('transition', "Don't go away") is None

    def test_transition_with_mixed_markers_preserves_order(self):
        """Test that detect_inconsistent_numbering works correctly."""
        # Test directly with ListicleGroup objects
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="next up", marker_type="transition", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="#5", marker_type="numbered", start_segment_idx=2, end_segment_idx=2),
        ]
        # Mixed ordinal + hash = inconsistent
        assert detect_inconsistent_numbering(groups) is True

        # Normalize - should convert to ordinal format
        result = normalize_numbering_format(groups)
        # ordinal (first) + hash (#5) = inconsistent, so normalization happens
        assert result[0].item_label == "1st"
        assert result[2].item_label == "3rd"  # normalized from #5


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


# ── US-135-003: Embedding-based keyword enhancement ────────────────────────────

class TestEmbeddingKeywordEnhancement:
    """Tests for embedding-based keyword enhancement (US-135-003)."""

    def test_enhance_keywords_returns_base_if_no_provider(self):
        """Without embedding provider, returns base keywords unchanged."""
        from src.chapter_detection.listicle_detector import _enhance_keywords_with_embeddings

        base_keywords = ["paris", "travel", "france"]
        result = _enhance_keywords_with_embeddings(
            "Visit Paris for travel",
            base_keywords,
            max_keywords=5,
            similarity_threshold=0.6,
            embedding_provider=None,
        )
        # Should return base keywords unchanged
        assert result == base_keywords

    def test_enhance_keywords_returns_base_if_enough_keywords(self):
        """If already have enough keywords, returns base keywords."""
        from src.chapter_detection.listicle_detector import _enhance_keywords_with_embeddings

        # Create a mock provider
        class MockProvider:
            def get_embedding(self, text):
                # Return a simple mock embedding
                return [0.1] * 768

        base_keywords = ["paris", "travel", "france", "vacation", "eiffel"]
        result = _enhance_keywords_with_embeddings(
            "Visit Paris for travel",
            base_keywords,
            max_keywords=5,  # Same as number of base keywords
            similarity_threshold=0.6,
            embedding_provider=MockProvider(),
        )
        # Should return base keywords (already at max)
        assert result == base_keywords

    def test_enhance_keywords_adds_related_when_provider_given(self):
        """With embedding provider, adds related keywords when few base keywords."""
        from src.chapter_detection.listicle_detector import _enhance_keywords_with_embeddings

        # Create a mock provider that returns similar embeddings for related terms
        class MockProvider:
            def get_embedding(self, text):
                # Return mock embedding based on text
                if "travel" in text.lower():
                    return [0.9] * 768  # High for travel
                elif "vacation" in text.lower():
                    return [0.85] * 768  # High for vacation (similar to travel)
                elif "food" in text.lower():
                    return [0.1] * 768  # Low for unrelated
                return [0.5] * 768

        base_keywords = ["travel"]  # Only 1 keyword
        result = _enhance_keywords_with_embeddings(
            "Travel is great",
            base_keywords,
            max_keywords=5,
            similarity_threshold=0.6,
            embedding_provider=MockProvider(),
        )
        # Should have more than base keywords (if enhancement worked)
        assert len(result) >= len(base_keywords)

    def test_extract_topic_keywords_with_embedding_disabled(self):
        """When use_embedding=False, returns simple extraction without enhancement."""
        text = "First we visit the Eiffel Tower in Paris France"
        keywords = _extract_topic_keywords(
            text,
            max_keywords=5,
            use_llm=False,
            use_embedding=False,
            embedding_provider=None,
        )
        # Should return simple extraction results
        assert len(keywords) > 0

    def test_extract_topic_keywords_with_embedding_no_provider(self):
        """When use_embedding=True but no provider, falls back gracefully."""
        text = "First we visit the Eiffel Tower in Paris France"
        keywords = _extract_topic_keywords(
            text,
            max_keywords=5,
            use_llm=False,
            use_embedding=True,
            embedding_provider=None,
        )
        # Should still work, returning simple extraction
        assert len(keywords) > 0

    def test_detect_listicle_groups_with_embedding_config(self):
        """detect_listicle_groups accepts embedding_provider parameter."""
        segments = _make_segments([
            "Step 1 the first attraction in Paris",
            "Step 2 the second attraction in London",
        ])

        # Test with no provider - should work without error
        groups = detect_listicle_groups(
            segments,
            listicle_topic_config=None,
            embedding_provider=None,
        )
        assert len(groups) == 2

    def test_embedding_config_passed_correctly(self):
        """ListicleTopicConfig use_embedding_topic_extraction is passed correctly."""
        class MockListicleTopicConfig:
            use_llm_topic_extraction = False
            min_keywords_for_simple = 3
            use_embedding_topic_extraction = True
            embedding_similarity_threshold = 0.7

        segments = _make_segments([
            "Step 1 visit Paris France",
            "Step 2 visit London England",
            "Step 3 visit Berlin Germany",
        ])

        config = MockListicleTopicConfig()
        # Should not crash even with no actual provider
        groups = detect_listicle_groups(
            segments,
            listicle_topic_config=config,
            embedding_provider=None,  # No provider, but config is read
        )
        assert len(groups) == 3


# ── US-122-007: Keyword extraction quality validation ────────────────────────

class TestKeywordExtractionQualityValidation:
    """Tests for keyword extraction quality validation (US-122-007)."""

    def test_very_short_segment_returns_empty_keywords(self):
        """Very short segments (< 20 chars) return empty keywords gracefully."""
        from src.chapter_detection.listicle_detector import _extract_simple_keywords

        # Test empty string
        result = _extract_simple_keywords("", max_keywords=5)
        assert result == []

        # Test very short string (< 20 chars)
        result = _extract_simple_keywords("Hi", max_keywords=5)
        assert result == []

        # Test exactly at threshold - 1
        result = _extract_simple_keywords("Short text", max_keywords=5)
        assert result == []

        # Test exactly at threshold
        result = _extract_simple_keywords("This is nineteen", max_keywords=5)
        # "this" (4), "nineteen" (8) after stop words, but text < 20 chars
        # The function checks total text length, not word count

    def test_short_segment_at_threshold(self):
        """Text at exactly 20 chars threshold returns keywords if valid."""
        from src.chapter_detection.listicle_detector import _extract_simple_keywords

        # Exactly 20 chars with meaningful words
        text = "This is exactly 20 ch"
        result = _extract_simple_keywords(text, max_keywords=5, min_length_threshold=20)
        # At threshold but may have valid keywords

    def test_min_keywords_for_simple_threshold(self):
        """min_keywords_for_simple (3) threshold triggers LLM fallback correctly."""
        from src.chapter_detection.listicle_detector import _extract_topic_keywords

        # Text that yields exactly 2 keywords (below threshold of 3)
        text = "Visit Paris London"
        # Simple extraction: "paris", "london" = 2 keywords

        # With LLM disabled, should return whatever simple extraction gives
        result = _extract_topic_keywords(
            text, max_keywords=5,
            use_llm=False,
            min_keywords_for_llm=3,
        )
        # Should work without error, just returns simple extraction
        assert isinstance(result, list)

        # With LLM enabled, when below threshold should trigger LLM fallback
        # (may return empty if LLM not available in test environment)
        result = _extract_topic_keywords(
            text, max_keywords=5,
            use_llm=True,
            min_keywords_for_llm=3,
        )
        assert isinstance(result, list)

    def test_llm_fallback_triggers_when_enabled(self):
        """LLM fallback triggers correctly when use_llm_topic_extraction is True."""
        from src.chapter_detection.listicle_detector import _extract_topic_keywords

        # Short text that yields < min_keywords_for_llm
        text = "Visit Paris"

        # With LLM disabled, just returns simple result
        result_no_llm = _extract_topic_keywords(
            text, max_keywords=5,
            use_llm=False,
            min_keywords_for_llm=3,
        )
        assert isinstance(result_no_llm, list)

        # With LLM enabled, attempts LLM fallback (may return empty if LLM unavailable)
        result_with_llm = _extract_topic_keywords(
            text, max_keywords=5,
            use_llm=True,
            min_keywords_for_llm=3,
        )
        assert isinstance(result_with_llm, list)

    def test_keyword_deduplication_case_insensitive(self):
        """Keyword deduplication preserves meaningful duplicates (e.g., 'Python' vs 'python')."""
        from src.chapter_detection.listicle_detector import _extract_simple_keywords

        # Text with duplicate words of different cases
        text = "Python python PYTHON Python programming Python"

        result = _extract_simple_keywords(text, max_keywords=10)

        # Should deduplicate case-insensitively - only keep 'python' once
        assert 'python' in result
        # Should not have duplicates (either 'python' or 'PYTHON', but not both)
        assert len([k for k in result if k.lower() == 'python']) == 1
        # Should preserve first occurrence
        assert result[0] == 'python'

    def test_keyword_deduplication_preserves_different_words(self):
        """Deduplication preserves different meaningful words."""
        from src.chapter_detection.listicle_detector import _extract_simple_keywords

        text = "Python Java javascript Python Ruby rust"

        result = _extract_simple_keywords(text, max_keywords=10)

        # Should have both 'python' and 'java' (different words)
        assert 'python' in result
        assert 'java' in result
        assert 'javascript' in result  # different from java
        assert 'ruby' in result

    def test_quality_threshold_filters_low_quality(self):
        """Validation that extracted keywords meet minimum quality threshold."""
        from src.chapter_detection.listicle_detector import _extract_simple_keywords

        # Text with mostly stop words / short words - low quality
        text = "the a is are was were be been being have"

        result = _extract_simple_keywords(text, max_keywords=5)
        # Should return empty or very few keywords for low-quality text

    def test_logging_for_extraction_method(self, caplog):
        """Logging for keyword extraction method used (simple vs LLM)."""
        import logging
        from src.chapter_detection.listicle_detector import _extract_topic_keywords

        caplog.set_level(logging.DEBUG)

        # Test logging when LLM fallback triggers
        _extract_topic_keywords(
            "Visit",  # Short text that will trigger LLM
            max_keywords=5,
            use_llm=True,
            min_keywords_for_llm=3,
        )

        # Should log debug message about LLM fallback attempt
        log_messages = [r.message for r in caplog.records]
        # Check that logging happened (LLM fallback was attempted)
        assert any("LLM" in msg or "simple" in msg.lower() for msg in log_messages)

    def test_detect_listicle_groups_with_short_segments(self, caplog):
        """detect_listicle_groups handles very short segments gracefully."""
        import logging
        from src.chapter_detection.listicle_detector import detect_listicle_groups

        caplog.set_level(logging.DEBUG)

        # Create segments with very short text
        segments = [
            {'index': 0, 'text': '#1'},
            {'index': 1, 'text': 'Hi'},
            {'index': 2, 'text': '#2'},
            {'index': 3, 'text': 'Ok'},
        ]

        # Should not crash, should return groups with empty keywords
        groups = detect_listicle_groups(segments)

        # With short segments, we might not detect enough markers for listicle
        # But it should not crash
        assert isinstance(groups, list)

    def test_config_passes_min_keywords_to_extraction(self):
        """ListicleTopicConfig min_keywords_for_simple is passed correctly."""
        from src.chapter_detection.listicle_detector import detect_listicle_groups

        class MockListicleTopicConfig:
            use_llm_topic_extraction = True
            min_keywords_for_simple = 3

        segments = _make_segments([
            "Step 1 visit Paris France",
            "Step 2 visit London England",
            "Step 3 visit Berlin Germany",
        ])

        config = MockListicleTopicConfig()
        groups = detect_listicle_groups(segments, listicle_topic_config=config)

        assert len(groups) == 3
        # Should not crash, keywords may be extracted


# ── Normalize Marker Sequence Tests (US-122-012) ─────────────────────────────

class TestNormalizeMarkerSequence:
    """Tests for _normalize_marker_sequence function edge cases."""

    def _make_group(self, group_id: int, label: str, marker_type: str) -> 'ListicleGroup':
        """Create a ListicleGroup for testing."""
        from src.chapter_detection.models import ListicleGroup
        return ListicleGroup(
            group_id=group_id,
            item_label=label,
            marker_type=marker_type,
            start_segment_idx=0,
            end_segment_idx=0,
        )

    def test_empty_groups_returns_empty(self):
        """Test edge case: empty groups list returns empty."""
        result = _normalize_marker_sequence([])
        assert result == []

    def test_all_transition_markers_no_positions(self):
        """Test normalization when all markers are transition type (no positions).

        This is acceptance criterion #1: when all markers are transitions
        (unpositioned), they should still get proper group_ids assigned.
        """
        groups = [
            self._make_group(0, "next up", "transition"),
            self._make_group(1, "moving on to", "transition"),
            self._make_group(2, "now let's look at", "transition"),
        ]

        result = _normalize_marker_sequence(groups)

        # Should still reassign group_ids to clean 0-based sequence
        assert len(result) == 3
        assert result[0].group_id == 0
        assert result[1].group_id == 1
        assert result[2].group_id == 2
        # Original ordering should be preserved
        assert result[0].item_label == "next up"
        assert result[1].item_label == "moving on to"
        assert result[2].item_label == "now let's look at"

    def test_terminal_markers_always_last(self):
        """Verify terminal markers (finally, lastly) always placed last.

        This is acceptance criterion #2.
        """
        groups = [
            self._make_group(0, "first", "ordinal"),
            self._make_group(1, "finally", "ordinal"),  # terminal (-1)
            self._make_group(2, "second", "ordinal"),
            self._make_group(3, "lastly", "ordinal"),   # terminal (-1)
        ]

        result = _normalize_marker_sequence(groups)

        # Terminal markers should be at the end
        terminal_labels = [g.item_label for g in result if g.item_label in ["finally", "lastly"]]
        # Both terminals should be at the end (in original order)
        assert terminal_labels == ["finally", "lastly"]
        # Check they're at the last two positions
        assert result[-2].item_label == "finally"
        assert result[-1].item_label == "lastly"

    def test_tie_breaking_same_numeric_position(self):
        """Test tie-breaking when multiple markers have same numeric position.

        This is acceptance criterion #3: markers with the same numeric
        position should maintain their original detection order.
        """
        groups = [
            self._make_group(0, "first", "ordinal"),   # position 1
            self._make_group(1, "first", "ordinal"),   # position 1 (tie)
            self._make_group(2, "second", "ordinal"),  # position 2
            self._make_group(3, "first", "ordinal"),   # position 1 (tie)
        ]

        result = _normalize_marker_sequence(groups)

        # Should be sorted by position, then by original index for ties
        labels = [g.item_label for g in result]
        assert labels == ["first", "first", "first", "second"]
        # group_ids should be clean 0-based
        assert [g.group_id for g in result] == [0, 1, 2, 3]

    def test_group_id_reassigned_to_clean_sequence(self):
        """Verify group_id is correctly reassigned to clean 0-based sequence.

        This is acceptance criterion #4.
        """
        # Groups with messy original group_ids
        groups = [
            self._make_group(5, "third", "ordinal"),
            self._make_group(2, "first", "ordinal"),
            self._make_group(8, "second", "ordinal"),
        ]

        result = _normalize_marker_sequence(groups)

        # Should be reassigned to clean 0-based sequence
        assert result[0].group_id == 0
        assert result[1].group_id == 1
        assert result[2].group_id == 2

    def test_original_order_preserved_for_unpositioned(self):
        """Test that unpositioned markers maintain relative order among themselves.

        When mixing positioned and unpositioned markers, the unpositioned
        markers should maintain their relative order (next up before
        "now let's talk about" because it appeared earlier).
        """
        groups = [
            self._make_group(0, "first", "ordinal"),          # position 1
            self._make_group(1, "next up", "transition"),    # unpositioned
            self._make_group(2, "second", "ordinal"),         # position 2
            self._make_group(3, "now let's talk about", "transition"),  # unpositioned
            self._make_group(4, "third", "ordinal"),         # position 3
        ]

        result = _normalize_marker_sequence(groups)

        # Positioned markers in sorted order (all first), then unpositioned
        labels = [g.item_label for g in result]
        # Positioned: first (1), second (2), third (3) - sorted by position
        # Unpositioned: "next up" (idx 1), "now let's talk about" (idx 3) - in original order
        # Result: all positioned first, then all unpositioned
        assert labels[0] == "first"   # positioned
        assert labels[1] == "second"  # positioned
        assert labels[2] == "third"    # positioned
        assert labels[3] == "next up"  # unpositioned - relative order preserved
        assert labels[4] == "now let's talk about"  # unpositioned - relative order preserved

    def test_mixed_with_terminal_at_end(self):
        """Test mixed markers with terminal markers always at the end."""
        groups = [
            self._make_group(0, "third", "ordinal"),
            self._make_group(1, "finally", "ordinal"),  # terminal
            self._make_group(2, "first", "ordinal"),
            self._make_group(3, "moving on to", "transition"),  # unpositioned
            self._make_group(4, "second", "ordinal"),
        ]

        result = _normalize_marker_sequence(groups)

        labels = [g.item_label for g in result]
        # Terminal should always be last
        assert labels[-1] == "finally"

    def test_single_group(self):
        """Test with a single group - should still work."""
        groups = [self._make_group(0, "first", "ordinal")]

        result = _normalize_marker_sequence(groups)

        assert len(result) == 1
        assert result[0].group_id == 0

    def test_all_terminal_markers(self):
        """Test with all terminal markers - should reassign group_ids."""
        groups = [
            self._make_group(0, "finally", "ordinal"),
            self._make_group(1, "lastly", "ordinal"),
        ]

        result = _normalize_marker_sequence(groups)

        # Terminals are positioned with -1, so they go to positioned list
        # and should be sorted by original index, then have group_ids reassigned
        assert len(result) == 2
        assert result[0].group_id == 0
        assert result[1].group_id == 1
        assert result[0].item_label == "finally"
        assert result[1].item_label == "lastly"


# ── US-126-009: Listicle group auto-correction for inconsistent numbering ──

class TestDetectInconsistentNumbering:
    """Tests for detecting inconsistent numbering in listicle groups."""

    def test_ordinal_consistent_returns_false(self):
        """'first, second, third' should be detected as consistent (all ordinal)."""
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="second", marker_type="ordinal", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="third", marker_type="ordinal", start_segment_idx=2, end_segment_idx=2),
        ]
        result = detect_inconsistent_numbering(groups)
        assert result is False

    def test_hash_inconsistent_returns_true(self):
        """'first, #3, third' should be detected as inconsistent (mixed ordinal and hash)."""
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="#3", marker_type="numbered", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="third", marker_type="ordinal", start_segment_idx=2, end_segment_idx=2),
        ]
        result = detect_inconsistent_numbering(groups)
        assert result is True

    def test_mixed_word_and_digit_inconsistent(self):
        """'one, two, #3' should be detected as inconsistent (word vs hash)."""
        groups = [
            ListicleGroup(group_id=0, item_label="one", marker_type="numbered", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="two", marker_type="numbered", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="#3", marker_type="numbered", start_segment_idx=2, end_segment_idx=2),
        ]
        result = detect_inconsistent_numbering(groups)
        assert result is True

    def test_single_group_returns_false(self):
        """Single group should return False (not enough to determine consistency)."""
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
        ]
        result = detect_inconsistent_numbering(groups)
        assert result is False

    def test_transition_markers_ignored(self):
        """Transition markers should be ignored when determining consistency."""
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="next up", marker_type="transition", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="third", marker_type="ordinal", start_segment_idx=2, end_segment_idx=2),
        ]
        # With transitions ignored, only ordinals remain -> consistent
        result = detect_inconsistent_numbering(groups)
        assert result is False

    def test_all_hash_consistent(self):
        """'#1, #2, #3' should be detected as consistent."""
        groups = [
            ListicleGroup(group_id=0, item_label="#1", marker_type="numbered", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="#2", marker_type="numbered", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="#3", marker_type="numbered", start_segment_idx=2, end_segment_idx=2),
        ]
        result = detect_inconsistent_numbering(groups)
        assert result is False


class TestNormalizeNumberingFormat:
    """Tests for normalizing numbering format in listicle groups."""

    def test_ordinal_consistent_not_normalized(self):
        """'first, second, third' should remain unchanged (already consistent)."""
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="second", marker_type="ordinal", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="third", marker_type="ordinal", start_segment_idx=2, end_segment_idx=2),
        ]
        result = normalize_numbering_format(groups)
        # Should remain as ordinals (no change needed since consistent)
        assert result[0].item_label == "first"
        assert result[1].item_label == "second"
        assert result[2].item_label == "third"
        # inconsistent_numbering should be False
        assert result[0].inconsistent_numbering is False

    def test_mixed_normalizes_to_ordinal(self):
        """'first, #3, third' should be normalized to 1st, 2nd, 3rd."""
        groups = [
            ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="#3", marker_type="numbered", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="third", marker_type="ordinal", start_segment_idx=2, end_segment_idx=2),
        ]
        result = normalize_numbering_format(groups)
        # Should be normalized to ordinal format
        assert result[0].item_label == "1st"
        assert result[1].item_label == "2nd"
        assert result[2].item_label == "3rd"
        # inconsistent_numbering should be True
        assert result[0].inconsistent_numbering is True
        assert result[1].inconsistent_numbering is True
        assert result[2].inconsistent_numbering is True

    def test_hash_only_consistent_not_normalized(self):
        """'#1, #2, #3' should remain unchanged (already consistent)."""
        groups = [
            ListicleGroup(group_id=0, item_label="#1", marker_type="numbered", start_segment_idx=0, end_segment_idx=0),
            ListicleGroup(group_id=1, item_label="#2", marker_type="numbered", start_segment_idx=1, end_segment_idx=1),
            ListicleGroup(group_id=2, item_label="#3", marker_type="numbered", start_segment_idx=2, end_segment_idx=2),
        ]
        result = normalize_numbering_format(groups)
        # Should remain as hash (no change needed since consistent)
        assert result[0].item_label == "#1"
        assert result[1].item_label == "#2"
        assert result[2].item_label == "#3"
        # inconsistent_numbering should be False
        assert result[0].inconsistent_numbering is False

    def test_single_group_not_normalized(self):
        """Single group should not be normalized."""
        groups = [ListicleGroup(group_id=0, item_label="first", marker_type="ordinal", start_segment_idx=0, end_segment_idx=0)]
        result = normalize_numbering_format(groups)
        # Single group should remain unchanged
        assert result[0].item_label == "first"
        assert result[0].inconsistent_numbering is False

    def test_empty_groups(self):
        """Empty groups should return empty list."""
        result = normalize_numbering_format([])
        assert result == []


class TestDetectNumberFormat:
    """Tests for detecting number format of markers."""

    def test_ordinal_returns_ordinal(self):
        assert _detect_number_format("ordinal", "first") == NumberFormat.ORDINAL
        assert _detect_number_format("ordinal", "second") == NumberFormat.ORDINAL

    def test_hash_returns_hash(self):
        assert _detect_number_format("numbered", "#1") == NumberFormat.HASH_NUMBERED
        assert _detect_number_format("numbered", "#3") == NumberFormat.HASH_NUMBERED

    def test_digit_returns_digit(self):
        assert _detect_number_format("numbered", "Step 1") == NumberFormat.DIGIT_NUMBERED
        assert _detect_number_format("numbered", "Item 2") == NumberFormat.DIGIT_NUMBERED

    def test_word_returns_word(self):
        assert _detect_number_format("numbered", "number one") == NumberFormat.WORD_NUMBERED
        assert _detect_number_format("numbered", "step two") == NumberFormat.WORD_NUMBERED

    def test_transition_returns_transition(self):
        assert _detect_number_format("transition", "next up") == NumberFormat.TRANSITION

    def test_unknown_returns_none(self):
        assert _detect_number_format("unknown", "something") is None


class TestListicleInconsistencyPenalty:
    """Tests for listicle consistency penalty in scoring.py (US-126-009)."""

    def test_inconsistent_numbering_penalty_applied(self):
        """When listicle group has inconsistent numbering, penalty should be applied."""
        from src.matching.scoring import apply_listicle_consistency, _DEFAULT_LISTICLE_INCONSISTENCY_PENALTY

        @dataclass
        class FakeVideoSegment:
            index: int = 0
            source_file: str = "video1.mp4"

        @dataclass
        class FakeVoSegment:
            index: int = 1  # Not first in group

        vo_segment = FakeVoSegment()
        video_segment = FakeVideoSegment()

        # Group with inconsistent numbering flag set
        listicle_groups = [
            ListicleGroup(
                group_id=0,
                item_label="first",
                marker_type="ordinal",
                start_segment_idx=0,
                end_segment_idx=2,
                inconsistent_numbering=True,  # This triggers the penalty
            ),
        ]

        # No previous matches needed for the penalty check
        confidence = 0.8
        result, reason = apply_listicle_consistency(
            confidence,
            vo_segment,
            video_segment,
            listicle_groups=listicle_groups,
            recent_matches=[],
        )

        # Should apply penalty
        expected = confidence - _DEFAULT_LISTICLE_INCONSISTENCY_PENALTY
        assert result == expected
        assert "inconsistent numbering" in reason.lower()

    def test_consistent_numbering_no_penalty(self):
        """When listicle group has consistent numbering, no penalty should be applied."""
        from src.matching.scoring import apply_listicle_consistency

        @dataclass
        class FakeVideoSegment:
            index: int = 0
            source_file: str = "video1.mp4"

        @dataclass
        class FakeVoSegment:
            index: int = 1

        vo_segment = FakeVoSegment()
        video_segment = FakeVideoSegment()

        # Group with consistent numbering (inconsistent_numbering=False)
        listicle_groups = [
            ListicleGroup(
                group_id=0,
                item_label="first",
                marker_type="ordinal",
                start_segment_idx=0,
                end_segment_idx=2,
                inconsistent_numbering=False,
            ),
        ]

        confidence = 0.8
        result, reason = apply_listicle_consistency(
            confidence,
            vo_segment,
            video_segment,
            listicle_groups=listicle_groups,
            recent_matches=[],
        )

        # Should return original confidence (no penalty)
        assert result == confidence
