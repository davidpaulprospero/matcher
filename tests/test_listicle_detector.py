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
        assert result == ('ordinal', 'first')

    def test_second(self):
        result = _detect_ordinal("Second, we have the gardens")
        assert result is not None
        assert result == ('ordinal', 'second')

    def test_third(self):
        result = _detect_ordinal("Third is the main attraction")
        assert result is not None
        assert result == ('ordinal', 'third')

    def test_finally(self):
        result = _detect_ordinal("Finally, we arrive at the museum")
        assert result is not None
        assert result == ('ordinal', 'finally')

    def test_lastly(self):
        result = _detect_ordinal("Lastly, don't forget the food")
        assert result is not None
        assert result == ('ordinal', 'lastly')

    def test_the_first(self):
        result = _detect_ordinal("The first thing you'll notice")
        assert result is not None
        assert result == ('ordinal', 'first')

    def test_and_finally(self):
        result = _detect_ordinal("And finally, the grand finale")
        assert result is not None
        assert result == ('ordinal', 'finally')

    def test_no_ordinal(self):
        result = _detect_ordinal("The city is beautiful at night")
        assert result is None

    def test_ordinal_not_at_start(self):
        result = _detect_ordinal("The building was first constructed in 1920")
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
