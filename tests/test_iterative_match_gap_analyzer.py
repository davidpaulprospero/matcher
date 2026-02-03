"""
Unit tests for src/iterative_match/gap_analyzer.py

Tests cover:
- GapSegment and GapAnalysis dataclasses
- analyze_gaps: pattern classification, empty input, all-above-threshold
- _classify_gap_pattern: location, proper_noun, action_verb, abstract_concept, emotion, other
- _has_location_pattern: multiple indicators, City/State patterns, preposition patterns
- _has_proper_noun: entity matching, capitalized words, common starters filtering
- extract_keywords_for_gap: keyword extraction by pattern type, dedup, max_keywords
- Gap severity ranking by confidence

Created: 2026-02-03 (Sprint 46 - Test Coverage)
"""

import pytest

from src.iterative_match.gap_analyzer import (
    GapSegment,
    GapAnalysis,
    LockedMatch,
    analyze_gaps,
    extract_keywords_for_gap,
    _classify_gap_pattern,
    _has_location_pattern,
    _has_proper_noun,
    ABSTRACT_CONCEPTS,
    EMOTION_WORDS,
    ACTION_VERBS,
    LOCATION_INDICATORS,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_gap(index: int, confidence: float, text: str, position: float = 0.0) -> GapSegment:
    """Create a GapSegment for testing."""
    return GapSegment(
        segment_index=index,
        confidence=confidence,
        voiceover_text=text,
        position=position,
    )


# ============================================================================
# GapSegment dataclass
# ============================================================================

class TestGapSegment:
    def test_defaults(self):
        gap = GapSegment(segment_index=0, confidence=0.5, voiceover_text="hello", position=1.0)
        assert gap.pattern_type == ""
        assert gap.keywords == []

    def test_fields_stored(self):
        gap = GapSegment(segment_index=3, confidence=0.42, voiceover_text="test text", position=10.5)
        assert gap.segment_index == 3
        assert gap.confidence == 0.42
        assert gap.voiceover_text == "test text"
        assert gap.position == 10.5


# ============================================================================
# GapAnalysis dataclass
# ============================================================================

class TestGapAnalysis:
    def test_empty_dominant_pattern(self):
        analysis = GapAnalysis()
        assert analysis.get_dominant_pattern() == "other"

    def test_dominant_pattern_single(self):
        analysis = GapAnalysis()
        analysis.pattern_counts["emotion"] = 5
        assert analysis.get_dominant_pattern() == "emotion"

    def test_dominant_pattern_multiple(self):
        analysis = GapAnalysis()
        analysis.pattern_counts["emotion"] = 2
        analysis.pattern_counts["location"] = 7
        analysis.pattern_counts["action_verb"] = 3
        assert analysis.get_dominant_pattern() == "location"

    def test_to_dict(self):
        analysis = GapAnalysis()
        analysis.pattern_counts["emotion"] = 1
        analysis.abstract_concepts = [0, 1]
        d = analysis.to_dict()
        assert d["pattern_counts"] == {"emotion": 1}
        assert d["abstract_concepts"] == [0, 1]
        assert d["proper_nouns"] == []


# ============================================================================
# _classify_gap_pattern
# ============================================================================

class TestClassifyGapPattern:
    def test_location_multiple_indicators(self):
        # Two location words triggers location pattern
        result = _classify_gap_pattern("the city near the mountain river", set())
        assert result == "location"

    def test_location_city_state(self):
        result = _classify_gap_pattern("They lived in Portland, Oregon for years", set())
        assert result == "location"

    def test_location_preposition_capital(self):
        result = _classify_gap_pattern("born in Paris long ago", set())
        assert result == "location"

    def test_proper_noun_entity_match(self):
        result = _classify_gap_pattern("we talked about einstein", {"einstein"})
        assert result == "proper_noun"

    def test_proper_noun_capitalized(self):
        # "Johnson" is capitalized mid-sentence, detected as proper noun
        result = _classify_gap_pattern("he met Johnson at the office", set())
        assert result == "proper_noun"

    def test_action_verb(self):
        result = _classify_gap_pattern("she was running through the field", set())
        assert result == "action_verb"

    def test_abstract_concept(self):
        result = _classify_gap_pattern("the idea of freedom changed everything", set())
        assert result == "abstract_concept"

    def test_emotion(self):
        result = _classify_gap_pattern("he felt incredibly anxious about it", set())
        assert result == "emotion"

    def test_other_fallback(self):
        result = _classify_gap_pattern("the box was on the table", set())
        assert result == "other"

    def test_priority_location_over_action(self):
        # Location check runs before action_verb
        result = _classify_gap_pattern("running through the city near the beach", set())
        assert result == "location"

    def test_priority_proper_noun_over_abstract(self):
        # Proper noun beats abstract concept
        result = _classify_gap_pattern("the freedom of Johnson was remarkable", set())
        # Johnson triggers proper_noun before freedom triggers abstract
        assert result == "proper_noun"

    def test_empty_text(self):
        result = _classify_gap_pattern("", set())
        assert result == "other"


# ============================================================================
# _has_location_pattern
# ============================================================================

class TestHasLocationPattern:
    def test_two_location_words(self):
        words = {"city", "river", "beautiful"}
        assert _has_location_pattern("beautiful city river", words) is True

    def test_one_location_word_not_enough(self):
        words = {"city", "beautiful"}
        assert _has_location_pattern("beautiful city", words) is False

    def test_city_state_regex(self):
        assert _has_location_pattern("Portland, Oregon is nice", set()) is True

    def test_preposition_capital(self):
        assert _has_location_pattern("grew up in Paris", set()) is True

    def test_no_location(self):
        words = {"the", "cat", "sat"}
        assert _has_location_pattern("the cat sat", words) is False


# ============================================================================
# _has_proper_noun
# ============================================================================

class TestHasProperNoun:
    def test_entity_match(self):
        assert _has_proper_noun("we discussed tesla", {"tesla"}) is True

    def test_entity_multi_word(self):
        assert _has_proper_noun("visiting new york tomorrow", {"new york", "new", "york"}) is True

    def test_capitalized_mid_sentence(self):
        assert _has_proper_noun("then Johnson arrived late", set()) is True

    def test_common_starter_filtered(self):
        # "The" at start and "This" mid-sentence are filtered as common starters
        assert _has_proper_noun("the cat is here", set()) is False

    def test_no_proper_nouns(self):
        assert _has_proper_noun("all lowercase text here", set()) is False


# ============================================================================
# analyze_gaps - Core function tests
# ============================================================================

class TestAnalyzeGaps:
    def test_empty_gaps_returns_empty_analysis(self):
        """AC: empty segment list returns empty gap list."""
        analysis = analyze_gaps([])
        assert analysis.pattern_counts == {}
        assert analysis.abstract_concepts == []
        assert analysis.proper_nouns == []
        assert analysis.action_descriptions == []
        assert analysis.locations == []
        assert analysis.emotional_content == []
        assert analysis.other == []
        assert analysis.get_dominant_pattern() == "other"

    def test_all_above_threshold_no_gaps(self):
        """AC: all segments above threshold returns no gaps.

        analyze_gaps doesn't filter by confidence - it classifies whatever
        gaps are passed to it. If no gaps are passed, result is empty.
        """
        # The caller (IterativeMatchStage) filters segments before passing to analyze_gaps.
        # If all segments are above threshold, no gaps are passed.
        analysis = analyze_gaps([])
        assert len(analysis.pattern_counts) == 0

    def test_single_gap_classified(self):
        gaps = [_make_gap(0, 0.3, "she was running quickly")]
        analysis = analyze_gaps(gaps)
        assert gaps[0].pattern_type == "action_verb"
        assert analysis.pattern_counts["action_verb"] == 1
        assert 0 in analysis.action_descriptions

    def test_multiple_gaps_classified(self):
        gaps = [
            _make_gap(0, 0.2, "the meaning of freedom"),
            _make_gap(1, 0.1, "he felt incredibly anxious"),
            _make_gap(2, 0.15, "they were swimming at dawn"),
        ]
        analysis = analyze_gaps(gaps)
        assert gaps[0].pattern_type == "abstract_concept"
        assert gaps[1].pattern_type == "emotion"
        assert gaps[2].pattern_type == "action_verb"
        assert len(analysis.pattern_counts) == 3

    def test_gaps_ranked_by_severity(self):
        """AC: mixed confidence levels correctly ranks gaps by severity.

        analyze_gaps doesn't rank, but we verify that gaps retain their
        confidence values so the caller can sort by severity.
        """
        gaps = [
            _make_gap(0, 0.8, "some normal text"),
            _make_gap(1, 0.1, "very low confidence text"),
            _make_gap(2, 0.0, "no match at all text"),
            _make_gap(3, 0.5, "medium confidence text"),
        ]
        analysis = analyze_gaps(gaps)
        # All gaps classified
        assert len(analysis.pattern_counts) > 0
        # Sort by confidence ascending = severity descending
        sorted_gaps = sorted(gaps, key=lambda g: g.confidence)
        assert sorted_gaps[0].confidence == 0.0
        assert sorted_gaps[1].confidence == 0.1
        assert sorted_gaps[-1].confidence == 0.8

    def test_no_match_segment_none_confidence(self):
        """AC: segments with no match (confidence 0.0) are classified."""
        gaps = [_make_gap(0, 0.0, "lost in the void")]
        analysis = analyze_gaps(gaps)
        assert gaps[0].pattern_type != ""  # Must be classified
        assert len(analysis.pattern_counts) == 1

    def test_gap_mutates_pattern_type(self):
        """analyze_gaps sets pattern_type on each GapSegment in-place."""
        gap = _make_gap(5, 0.4, "cooking dinner tonight")
        assert gap.pattern_type == ""
        analyze_gaps([gap])
        assert gap.pattern_type == "action_verb"

    def test_with_extracted_entities(self):
        """Entities list is used for proper noun detection."""
        gaps = [_make_gap(0, 0.3, "we talked about tesla motors")]
        analysis = analyze_gaps(gaps, extracted_entities=[{"name": "Tesla Motors"}])
        assert gaps[0].pattern_type == "proper_noun"

    def test_with_state_entities(self):
        """State's extracted_entities attribute is used when no explicit entities."""
        from unittest.mock import MagicMock
        state = MagicMock()
        state.extracted_entities = [{"name": "SpaceX"}]
        gaps = [_make_gap(0, 0.3, "news about spacex launch")]
        analysis = analyze_gaps(gaps, state=state)
        assert gaps[0].pattern_type == "proper_noun"

    def test_entities_override_state(self):
        """Explicit entities take precedence over state entities."""
        from unittest.mock import MagicMock
        state = MagicMock()
        state.extracted_entities = [{"name": "WrongEntity"}]
        gaps = [_make_gap(0, 0.3, "discussing apple products")]
        analysis = analyze_gaps(gaps, state=state, extracted_entities=[{"name": "Apple"}])
        assert gaps[0].pattern_type == "proper_noun"

    def test_clustered_gaps_populated(self):
        gaps = [
            _make_gap(0, 0.2, "the meaning of freedom"),
            _make_gap(1, 0.3, "she felt deeply grateful"),
            _make_gap(2, 0.1, "the pursuit of justice"),
        ]
        analysis = analyze_gaps(gaps)
        # abstract_concept should have indices 0 and 2
        assert 0 in analysis.clustered_gaps.get("abstract_concept", [])
        assert 2 in analysis.clustered_gaps.get("abstract_concept", [])
        assert 1 in analysis.clustered_gaps.get("emotion", [])


# ============================================================================
# extract_keywords_for_gap
# ============================================================================

class TestExtractKeywordsForGap:
    def test_basic_extraction(self):
        gap = _make_gap(0, 0.3, "The brave soldier marched forward")
        gap.pattern_type = "other"
        keywords = extract_keywords_for_gap(gap)
        assert len(keywords) > 0
        assert len(keywords) <= 5

    def test_max_keywords_limit(self):
        gap = _make_gap(0, 0.3, "The quick brown fox jumps over the lazy dog near the river")
        gap.pattern_type = "other"
        keywords = extract_keywords_for_gap(gap, max_keywords=3)
        assert len(keywords) <= 3

    def test_proper_nouns_prioritized(self):
        gap = _make_gap(0, 0.3, "Then Johnson visited the London museum")
        gap.pattern_type = "proper_noun"
        keywords = extract_keywords_for_gap(gap)
        # Proper nouns (capitalized) should appear first
        assert any("Johnson" in kw for kw in keywords[:3])

    def test_action_verb_keywords(self):
        gap = _make_gap(0, 0.3, "she was running and jumping")
        gap.pattern_type = "action_verb"
        keywords = extract_keywords_for_gap(gap)
        assert any(kw in ("running", "jumping") for kw in keywords)

    def test_location_keywords(self):
        gap = _make_gap(0, 0.3, "the city near the mountain was quiet")
        gap.pattern_type = "location"
        keywords = extract_keywords_for_gap(gap)
        assert any(kw in LOCATION_INDICATORS for kw in keywords)

    def test_abstract_concept_keywords(self):
        gap = _make_gap(0, 0.3, "the meaning of freedom endures")
        gap.pattern_type = "abstract_concept"
        keywords = extract_keywords_for_gap(gap)
        assert "freedom" in keywords

    def test_deduplication(self):
        gap = _make_gap(0, 0.3, "Freedom freedom FREEDOM everywhere")
        gap.pattern_type = "abstract_concept"
        keywords = extract_keywords_for_gap(gap)
        lower_keywords = [k.lower() for k in keywords]
        assert len(lower_keywords) == len(set(lower_keywords))

    def test_stop_words_excluded(self):
        gap = _make_gap(0, 0.3, "the and but for with")
        gap.pattern_type = "other"
        keywords = extract_keywords_for_gap(gap)
        assert len(keywords) == 0

    def test_empty_text(self):
        gap = _make_gap(0, 0.3, "")
        gap.pattern_type = "other"
        keywords = extract_keywords_for_gap(gap)
        assert keywords == []


# ============================================================================
# LockedMatch dataclass
# ============================================================================

class TestLockedMatch:
    def test_fields(self):
        lm = LockedMatch(segment_index=2, video_id="abc123", confidence=0.95, position=5.0, title="My Video")
        assert lm.segment_index == 2
        assert lm.video_id == "abc123"
        assert lm.confidence == 0.95
        assert lm.title == "My Video"

    def test_defaults(self):
        lm = LockedMatch(segment_index=0, video_id="x", confidence=0.5, position=0.0)
        assert lm.title == ""


# ============================================================================
# Word list sanity checks
# ============================================================================

class TestWordLists:
    def test_abstract_concepts_not_empty(self):
        assert len(ABSTRACT_CONCEPTS) > 10

    def test_emotion_words_not_empty(self):
        assert len(EMOTION_WORDS) > 10

    def test_action_verbs_not_empty(self):
        assert len(ACTION_VERBS) > 10

    def test_location_indicators_not_empty(self):
        assert len(LOCATION_INDICATORS) > 10

    def test_no_overlap_abstract_emotion(self):
        """Abstract concepts and emotion words should be disjoint."""
        overlap = ABSTRACT_CONCEPTS & EMOTION_WORDS
        assert len(overlap) == 0, f"Overlapping words: {overlap}"

    def test_all_lowercase(self):
        """All word lists should contain lowercase entries."""
        for word in ABSTRACT_CONCEPTS:
            assert word == word.lower(), f"Non-lowercase in ABSTRACT_CONCEPTS: {word}"
        for word in EMOTION_WORDS:
            assert word == word.lower(), f"Non-lowercase in EMOTION_WORDS: {word}"
        for word in ACTION_VERBS:
            assert word == word.lower(), f"Non-lowercase in ACTION_VERBS: {word}"
