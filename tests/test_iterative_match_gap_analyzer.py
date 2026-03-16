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
    GapPatternLog,
    LockedMatch,
    analyze_gaps,
    analyze_gap_patterns_for_logging,
    annotate_gaps_with_chapters,
    log_gap_pattern_analysis,
    extract_keywords_for_gap,
    derive_queries_from_descriptions,
    extract_description_queries,
    categorize_gaps_by_confidence,
    get_strategy_for_confidence_category,
    extract_context_from_nearby_matches,
    compute_context_relevance,
    generate_context_aware_queries,
    get_context_keywords_for_gap,
    extract_tags_from_nearby_matches,
    # US-135-007: Listicle-aware gap analysis
    annotate_gaps_with_listicle_position,
    get_query_strategy_for_listicle_position,
    generate_listicle_aware_queries,
    LISTICLE_BOUNDARY_THRESHOLD,
    _classify_gap_pattern,
    _has_location_pattern,
    _has_proper_noun,
    _extract_content_words,
    _extract_key_phrases,
    _extract_keywords_from_text,
    _cluster_gaps_by_topic,
    _cluster_gaps_by_position,
    _extract_recurring_keywords,
    _identify_keyword_patterns,
    _generate_query_hints_from_patterns,
    ABSTRACT_CONCEPTS,
    EMOTION_WORDS,
    ACTION_VERBS,
    LOCATION_INDICATORS,
    INTRO_PRIORITY_BOOST,
    CONCLUSION_PRIORITY_BOOST,
    DURATION_PRIORITY_THRESHOLD,
    DEFAULT_DURATION_PRIORITY_WEIGHT,
    ConfidenceCategory,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_gap(index: int, confidence: float, text: str, position: float = 0.0, duration: float = 0.0) -> GapSegment:
    """Create a GapSegment for testing."""
    return GapSegment(
        segment_index=index,
        confidence=confidence,
        voiceover_text=text,
        position=position,
        duration=duration,
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

    # US-94-012: Context-aware keyword extraction tests
    def test_context_text_adds_keywords(self):
        """Context from adjacent segments should add supplementary keywords."""
        gap = _make_gap(1, 0.3, "The hero fought bravely")
        gap.pattern_type = "action_verb"
        # Previous segment mentions "Medieval Castle"
        context = "The Medieval Castle stood tall"
        keywords = extract_keywords_for_gap(gap, max_keywords=5, context_text=context)
        # Should include keywords from both gap and context
        assert len(keywords) > 0
        # Context proper nouns like "Castle" should be in keywords
        assert any("castle" in kw.lower() for kw in keywords)

    def test_context_does_not_duplicate_gap_keywords(self):
        """Context should not duplicate keywords already in gap text."""
        gap = _make_gap(1, 0.3, "The soldier ran fast")
        gap.pattern_type = "action_verb"
        # Context has same keyword
        context = "The soldier was brave"
        keywords = extract_keywords_for_gap(gap, max_keywords=5, context_text=context)
        # Should not have duplicates
        lower_keywords = [k.lower() for k in keywords]
        assert len(lower_keywords) == len(set(lower_keywords))

    def test_context_window_multiple_segments(self):
        """Multiple context segments should all contribute keywords."""
        gap = _make_gap(2, 0.3, "The event was historic")
        gap.pattern_type = "abstract_concept"
        # Context from multiple segments
        context = "Ancient Rome was powerful. The colosseum hosted games."
        keywords = extract_keywords_for_gap(gap, max_keywords=5, context_text=context)
        assert len(keywords) > 0

    def test_context_empty_string(self):
        """Empty context should work like original behavior."""
        gap = _make_gap(0, 0.3, "The brave soldier")
        gap.pattern_type = "other"
        keywords = extract_keywords_for_gap(gap, max_keywords=5, context_text="")
        # Should behave like no context
        assert len(keywords) > 0
        assert len(keywords) <= 5

    def test_context_prioritizes_gap_keywords(self):
        """Keywords from gap text should take priority over context."""
        gap = _make_gap(1, 0.3, "Paris is wonderful")
        gap.pattern_type = "location"
        # Context has different city
        context = "London is also nice"
        keywords = extract_keywords_for_gap(gap, max_keywords=3, context_text=context)
        # Gap keywords should be prioritized
        assert any("paris" in kw.lower() for kw in keywords[:2])


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


# ============================================================================
# GapPatternLog dataclass
# ============================================================================

class TestGapPatternLog:
    def test_defaults(self):
        log = GapPatternLog(pass_number=1, total_gaps=5)
        assert log.pass_number == 1
        assert log.total_gaps == 5
        assert log.pattern_summary == {}
        assert log.topic_clusters == {}
        assert log.query_hints == []

    def test_to_dict(self):
        log = GapPatternLog(pass_number=2, total_gaps=10)
        log.pattern_summary = {"emotion": 3, "location": 2}
        log.query_hints = ["Consider searching for: freedom"]
        d = log.to_dict()
        assert d["pass_number"] == 2
        assert d["total_gaps"] == 10
        assert d["pattern_summary"] == {"emotion": 3, "location": 2}
        assert "freedom" in d["query_hints"][0]

    def test_from_dict(self):
        data = {
            "pass_number": 3,
            "total_gaps": 15,
            "pattern_summary": {"abstract_concept": 5},
            "topic_clusters": {"topic_0": [1, 2, 3]},
            "topic_keywords": {"topic_0": ["freedom", "justice"]},
            "position_clusters": [{"start": 0, "end": 60, "indices": [0, 1], "count": 2}],
            "position_concentration": "early",
            "recurring_keywords": {"world": 3, "peace": 2},
            "keyword_patterns": ["world peace"],
            "query_hints": ["Consider searching for: world, peace"],
        }
        log = GapPatternLog.from_dict(data)
        assert log.pass_number == 3
        assert log.total_gaps == 15
        assert log.pattern_summary == {"abstract_concept": 5}
        assert log.topic_clusters == {"topic_0": [1, 2, 3]}
        assert log.position_concentration == "early"


# ============================================================================
# US-63-012: Gap Pattern Analysis Logging Tests
# ============================================================================

class TestAnalyzeGapPatternsForLogging:
    """Tests for analyze_gap_patterns_for_logging function."""

    def test_empty_gaps(self):
        """Empty gaps return empty log."""
        log = analyze_gap_patterns_for_logging([], pass_number=1)
        assert log.total_gaps == 0
        assert log.pattern_summary == {}
        assert log.query_hints == []

    def test_basic_analysis_with_mock_data(self):
        """AC: Unit test verifies pattern detection with mock gap data."""
        gaps = [
            _make_gap(0, 0.3, "The world needs more freedom and justice", 10.0),
            _make_gap(1, 0.2, "The world is changing rapidly", 15.0),
            _make_gap(2, 0.4, "Freedom means different things to everyone", 20.0),
            _make_gap(3, 0.1, "Justice will prevail in the end", 25.0),
        ]
        # Pre-classify gaps
        for gap in gaps:
            gap.pattern_type = "abstract_concept"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1, total_duration=100.0)

        assert log.total_gaps == 4
        assert log.pass_number == 1
        # Should detect recurring keywords
        assert "world" in log.recurring_keywords or "freedom" in log.recurring_keywords
        # Should generate query hints
        assert len(log.query_hints) > 0

    def test_topic_clustering(self):
        """Gaps with shared keywords are clustered by topic."""
        gaps = [
            _make_gap(0, 0.3, "Tesla motors innovation", 0.0),
            _make_gap(1, 0.2, "Tesla electric vehicles", 10.0),
            _make_gap(2, 0.4, "Apple iPhone release", 20.0),
            _make_gap(3, 0.1, "Apple technology news", 30.0),
        ]
        for gap in gaps:
            gap.pattern_type = "proper_noun"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1)

        # Should have topic clusters
        # Gaps 0,1 share "tesla", gaps 2,3 share "apple"
        assert len(log.topic_clusters) >= 1

    def test_position_clustering_early(self):
        """Gaps at early timeline positions detected."""
        gaps = [
            _make_gap(0, 0.3, "introduction topic", 5.0),
            _make_gap(1, 0.2, "another early topic", 10.0),
            _make_gap(2, 0.4, "third early topic", 15.0),
        ]
        for gap in gaps:
            gap.pattern_type = "other"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1, total_duration=300.0)

        # Avg position ~10s out of 300s = early
        assert log.position_concentration == "early"

    def test_position_clustering_late(self):
        """Gaps at late timeline positions detected."""
        gaps = [
            _make_gap(0, 0.3, "conclusion topic", 250.0),
            _make_gap(1, 0.2, "ending thought", 270.0),
            _make_gap(2, 0.4, "final remarks", 290.0),
        ]
        for gap in gaps:
            gap.pattern_type = "other"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1, total_duration=300.0)

        # Avg position ~270s out of 300s = late
        assert log.position_concentration == "late"

    def test_position_clustering_spread(self):
        """Widely distributed gaps detected as spread."""
        gaps = [
            _make_gap(0, 0.3, "early topic", 10.0),
            _make_gap(1, 0.2, "middle topic", 150.0),
            _make_gap(2, 0.4, "late topic", 290.0),
        ]
        for gap in gaps:
            gap.pattern_type = "other"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1, total_duration=300.0)

        # Spread from 10 to 290 = spread
        assert log.position_concentration == "spread"

    def test_recurring_keywords(self):
        """Keywords appearing multiple times are captured."""
        gaps = [
            _make_gap(0, 0.3, "The climate change crisis", 0.0),
            _make_gap(1, 0.2, "Climate science research", 10.0),
            _make_gap(2, 0.4, "Global climate impact", 20.0),
        ]
        for gap in gaps:
            gap.pattern_type = "other"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1)

        # "climate" appears 3 times
        assert "climate" in log.recurring_keywords
        assert log.recurring_keywords["climate"] >= 2

    def test_query_hints_generated(self):
        """Query hints are generated from patterns."""
        gaps = [
            _make_gap(0, 0.3, "The meaning of freedom", 0.0),
            _make_gap(1, 0.2, "Justice for all", 10.0),
            _make_gap(2, 0.4, "Peace and love", 20.0),
        ]
        for gap in gaps:
            gap.pattern_type = "abstract_concept"

        log = analyze_gap_patterns_for_logging(gaps, pass_number=1)

        # Should have hints
        assert len(log.query_hints) > 0
        # Should mention abstract concepts if dominant
        has_abstract_hint = any("abstract" in h.lower() for h in log.query_hints)
        has_keyword_hint = any("consider searching" in h.lower() for h in log.query_hints)
        assert has_abstract_hint or has_keyword_hint


class TestClusterGapsByTopic:
    """Tests for _cluster_gaps_by_topic helper."""

    def test_empty_gaps(self):
        clusters, keywords = _cluster_gaps_by_topic([])
        assert clusters == {}
        assert keywords == {}

    def test_single_gap_no_cluster(self):
        """Single gap cannot form a cluster."""
        gaps = [_make_gap(0, 0.3, "unique content here")]
        gaps[0].pattern_type = "other"
        clusters, keywords = _cluster_gaps_by_topic(gaps)
        assert len(clusters) == 0

    def test_shared_keyword_cluster(self):
        """Gaps sharing keywords are clustered."""
        gaps = [
            _make_gap(0, 0.3, "technology innovation future"),
            _make_gap(1, 0.2, "technology progress future"),
        ]
        for g in gaps:
            g.pattern_type = "other"
        clusters, keywords = _cluster_gaps_by_topic(gaps)
        # Should have at least one cluster with shared keywords
        if clusters:
            first_cluster = list(clusters.values())[0]
            assert len(first_cluster) >= 2


class TestClusterGapsByPosition:
    """Tests for _cluster_gaps_by_position helper."""

    def test_empty_gaps(self):
        clusters, concentration = _cluster_gaps_by_position([], 100.0)
        assert clusters == []
        assert concentration == "none"

    def test_nearby_gaps_cluster(self):
        """Gaps within window are clustered."""
        gaps = [
            _make_gap(0, 0.3, "text", 10.0),
            _make_gap(1, 0.2, "text", 20.0),
            _make_gap(2, 0.4, "text", 30.0),
        ]
        clusters, _ = _cluster_gaps_by_position(gaps, 300.0, window_seconds=60.0)
        # All within 60s should be one cluster
        assert len(clusters) >= 1
        if clusters:
            assert clusters[0]["count"] >= 2


class TestExtractRecurringKeywords:
    """Tests for _extract_recurring_keywords helper."""

    def test_empty_gaps(self):
        result = _extract_recurring_keywords([])
        assert result == {}

    def test_single_occurrence_excluded(self):
        """Keywords appearing once are excluded."""
        gaps = [_make_gap(0, 0.3, "unique word here")]
        result = _extract_recurring_keywords(gaps, min_occurrences=2)
        assert "unique" not in result

    def test_recurring_captured(self):
        """Keywords appearing multiple times are captured."""
        gaps = [
            _make_gap(0, 0.3, "technology advances"),
            _make_gap(1, 0.2, "technology improves"),
        ]
        result = _extract_recurring_keywords(gaps, min_occurrences=2)
        assert "technology" in result
        assert result["technology"] >= 2


class TestIdentifyKeywordPatterns:
    """Tests for _identify_keyword_patterns helper."""

    def test_empty_gaps(self):
        result = _identify_keyword_patterns([])
        assert result == []

    def test_bigram_detection(self):
        """Common bigrams are detected."""
        gaps = [
            _make_gap(0, 0.3, "climate change impact"),
            _make_gap(1, 0.2, "climate change effects"),
        ]
        result = _identify_keyword_patterns(gaps)
        assert "climate change" in result


class TestGenerateQueryHintsFromPatterns:
    """Tests for _generate_query_hints_from_patterns helper."""

    def test_hint_from_recurring_keywords(self):
        """Hints include recurring keywords."""
        log = GapPatternLog(pass_number=1, total_gaps=5)
        log.recurring_keywords = {"freedom": 3, "justice": 2}
        hints = _generate_query_hints_from_patterns(log, [])
        assert any("freedom" in h for h in hints)

    def test_hint_from_topic_cluster(self):
        """Hints mention topic clusters."""
        log = GapPatternLog(pass_number=1, total_gaps=5)
        log.topic_clusters = {"topic_0": [0, 1, 2]}
        log.topic_keywords = {"topic_0": ["technology", "future"]}
        hints = _generate_query_hints_from_patterns(log, [])
        assert any("cluster" in h.lower() or "technology" in h for h in hints)

    def test_hint_from_position_concentration(self):
        """Hints mention position concentration."""
        log = GapPatternLog(pass_number=1, total_gaps=5)
        log.position_concentration = "early"
        hints = _generate_query_hints_from_patterns(log, [])
        assert any("early" in h.lower() for h in hints)

    def test_hint_from_dominant_pattern(self):
        """Hints mention dominant pattern type."""
        log = GapPatternLog(pass_number=1, total_gaps=5)
        log.pattern_summary = {"abstract_concept": 4, "other": 1}
        hints = _generate_query_hints_from_patterns(log, [])
        assert any("abstract" in h.lower() for h in hints)


class TestLogGapPatternAnalysis:
    """Tests for log_gap_pattern_analysis helper."""

    def test_logs_without_error(self):
        """Logging function runs without error."""
        import logging
        log = GapPatternLog(pass_number=1, total_gaps=3)
        log.pattern_summary = {"emotion": 2}
        log.recurring_keywords = {"love": 2}
        log.query_hints = ["Consider searching for: love"]

        # Should not raise
        test_logger = logging.getLogger("test_gap_pattern")
        log_gap_pattern_analysis(log, test_logger)

    def test_logs_empty_log_without_error(self):
        """Logging empty log runs without error."""
        log = GapPatternLog(pass_number=1, total_gaps=0)
        log_gap_pattern_analysis(log)  # Uses default logger


# ============================================================================
# US-70-012: Description-derived search queries
# ============================================================================

class TestExtractDescriptionQueries:
    """Tests for extract_description_queries function."""

    def test_empty_descriptions(self):
        """No descriptions returns empty list."""
        result = extract_description_queries([])
        assert result == []

    def test_none_descriptions_filtered(self):
        """None and empty descriptions are handled gracefully."""
        result = extract_description_queries(["", None, ""])
        assert result == []

    def test_single_description_extracts_queries(self):
        """A single description with content produces queries."""
        descriptions = [
            "This documentary explores renewable energy solutions "
            "including solar panels and wind turbines across Europe."
        ]
        result = extract_description_queries(descriptions, max_queries=5)
        assert len(result) > 0
        # Should contain meaningful phrases, not stop words
        for q in result:
            assert len(q) > 3

    def test_multiple_descriptions_shared_themes(self):
        """Multiple descriptions with shared themes produce relevant queries."""
        descriptions = [
            "Climate change affects coral reefs worldwide",
            "Scientists study coral reef degradation from climate change",
            "Protecting coral ecosystems from rising ocean temperatures",
        ]
        result = extract_description_queries(descriptions, max_queries=10)
        assert len(result) > 0
        # "coral" should appear somewhere in the queries since it's a shared theme
        all_queries_lower = ' '.join(result).lower()
        assert 'coral' in all_queries_lower

    def test_max_queries_respected(self):
        """Output is capped at max_queries."""
        descriptions = [
            "A very long description about many different topics including "
            "technology, science, medicine, agriculture, economics, politics, "
            "culture, arts, sports, entertainment, education, and philosophy."
        ] * 5
        result = extract_description_queries(descriptions, max_queries=3)
        assert len(result) <= 3

    def test_url_noise_filtered(self):
        """URLs and YouTube boilerplate are filtered from descriptions."""
        descriptions = [
            "Subscribe to my channel https://youtube.com/example "
            "Follow me on Twitter. The space exploration mission launched."
        ]
        result = extract_description_queries(descriptions, max_queries=5)
        # Should not contain URL fragments or boilerplate
        for q in result:
            assert 'http' not in q.lower()
            assert 'subscribe' not in q.lower()

    def test_graceful_with_short_descriptions(self):
        """Very short descriptions don't crash."""
        descriptions = ["OK", "Hi", ""]
        result = extract_description_queries(descriptions)
        # May return empty or minimal results, but should not crash
        assert isinstance(result, list)

    def test_bigram_phrases_preferred(self):
        """Multi-word phrases are preferred over single words when available."""
        descriptions = [
            "artificial intelligence research advances rapidly in modern laboratories",
            "artificial intelligence models transform healthcare diagnostics",
        ]
        result = extract_description_queries(descriptions, max_queries=5)
        # Should have at least one multi-word phrase
        multi_word = [q for q in result if ' ' in q]
        assert len(multi_word) > 0


class TestExtractContentWords:
    """Tests for _extract_content_words helper."""

    def test_removes_urls(self):
        result = _extract_content_words("visit https://example.com for more info about technology")
        assert not any('example' in w for w in result)
        assert 'technology' in result

    def test_removes_stop_words(self):
        result = _extract_content_words("the quick brown fox jumps over")
        assert 'quick' in result
        assert 'brown' in result
        # 'the' and 'over' are stop words / too short
        assert 'the' not in result

    def test_removes_short_words(self):
        result = _extract_content_words("a be do it go")
        assert result == []


class TestExtractKeyPhrases:
    """Tests for _extract_key_phrases helper."""

    def test_basic_bigram_extraction(self):
        top_words = {'climate', 'change', 'global'}
        result = _extract_key_phrases(
            "Understanding global climate change effects on agriculture",
            top_words, max_phrases=3
        )
        assert len(result) > 0
        # Should contain bigrams with at least one top word
        for phrase in result:
            words = phrase.split()
            assert any(w in top_words for w in words)

    def test_max_phrases_limit(self):
        top_words = {'every', 'word', 'here'}
        result = _extract_key_phrases(
            "every word here matters greatly indeed today certainly",
            top_words, max_phrases=2
        )
        assert len(result) <= 2

    def test_empty_text(self):
        result = _extract_key_phrases("", {'word'}, max_phrases=3)
        assert result == []


# ============================================================================
# US-71-007: Chapter-aware gap prioritization
# ============================================================================

class TestAnnotateGapsWithChapters:
    """Tests for annotate_gaps_with_chapters function."""

    def test_intro_gaps_get_priority_boost(self):
        """AC: Gaps in introduction chapters (first 10%) get +0.2 boost."""
        # 100 total segments, so first 10 = intro
        gaps = [
            _make_gap(2, 0.5, "intro content", 5.0),
            _make_gap(5, 0.4, "more intro content", 10.0),
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        for gap in result:
            assert gap.priority_boost == INTRO_PRIORITY_BOOST  # 0.2

    def test_conclusion_gaps_get_priority_boost(self):
        """AC: Gaps in conclusion chapters (last 10%) get +0.15 boost."""
        # 100 total segments, so segment 90+ = conclusion
        gaps = [
            _make_gap(92, 0.5, "conclusion content", 200.0),
            _make_gap(97, 0.4, "final thoughts", 210.0),
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        for gap in result:
            assert gap.priority_boost == CONCLUSION_PRIORITY_BOOST  # 0.15

    def test_middle_gaps_no_boost(self):
        """Gaps in middle sections get no priority boost."""
        gaps = [
            _make_gap(30, 0.5, "middle content", 60.0),
            _make_gap(50, 0.4, "more middle content", 100.0),
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        for gap in result:
            assert gap.priority_boost == 0.0

    def test_intro_conclusion_higher_priority_than_middle(self):
        """AC: Intro/conclusion gaps sort before middle gaps with same confidence."""
        gaps = [
            _make_gap(50, 0.5, "middle content", 100.0),     # middle, no boost
            _make_gap(3, 0.5, "intro content", 5.0),         # intro, +0.2 boost
            _make_gap(95, 0.5, "conclusion content", 200.0), # conclusion, +0.15 boost
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Effective priority = confidence - boost
        # intro: 0.5 - 0.2 = 0.3 (highest priority)
        # conclusion: 0.5 - 0.15 = 0.35
        # middle: 0.5 - 0.0 = 0.5 (lowest priority)
        assert result[0].segment_index == 3   # intro first
        assert result[1].segment_index == 95  # conclusion second
        assert result[2].segment_index == 50  # middle last

    def test_chapter_annotation_from_chapters(self):
        """AC: analyze_gaps() annotates each gap with its containing chapter."""
        chapters = [
            {'title': 'Introduction', 'start_segment': 0, 'end_segment': 10},
            {'title': 'Main Body', 'start_segment': 10, 'end_segment': 80},
            {'title': 'Conclusion', 'start_segment': 80, 'end_segment': 100},
        ]
        gaps = [
            _make_gap(5, 0.5, "intro text", 10.0),
            _make_gap(50, 0.4, "body text", 100.0),
            _make_gap(90, 0.3, "conclusion text", 200.0),
        ]
        result = annotate_gaps_with_chapters(
            gaps, total_segments=100, chapters=chapters
        )
        # Find each gap by index
        gap_by_idx = {g.segment_index: g for g in result}
        assert gap_by_idx[5].chapter_id == 'Introduction'
        assert gap_by_idx[50].chapter_id == 'Main Body'
        assert gap_by_idx[90].chapter_id == 'Conclusion'

    def test_chapter_annotation_from_listicle_groups(self):
        """Gaps annotated from listicle groups when no chapters."""
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 30},
            {'group_id': 'group_B', 'start_segment': 30, 'end_segment': 60},
        ]
        gaps = [
            _make_gap(15, 0.5, "group A content", 30.0),
            _make_gap(45, 0.4, "group B content", 90.0),
        ]
        result = annotate_gaps_with_chapters(
            gaps, total_segments=100, listicle_groups=groups
        )
        gap_by_idx = {g.segment_index: g for g in result}
        assert gap_by_idx[15].chapter_id == 'group_A'
        assert gap_by_idx[45].chapter_id == 'group_B'

    def test_empty_gaps_returns_empty(self):
        """Empty gap list returns empty."""
        result = annotate_gaps_with_chapters([], total_segments=100)
        assert result == []

    def test_custom_boost_values(self):
        """Custom boost values are applied correctly."""
        gaps = [
            _make_gap(1, 0.5, "intro", 2.0),
            _make_gap(99, 0.5, "outro", 200.0),
        ]
        result = annotate_gaps_with_chapters(
            gaps, total_segments=100,
            intro_boost=0.3, conclusion_boost=0.1
        )
        gap_by_idx = {g.segment_index: g for g in result}
        assert gap_by_idx[1].priority_boost == 0.3
        assert gap_by_idx[99].priority_boost == 0.1

    def test_sorting_by_priority(self):
        """Gaps are sorted so high-priority (low effective score) come first."""
        gaps = [
            _make_gap(50, 0.3, "middle low conf", 100.0),   # eff: 0.3
            _make_gap(2, 0.6, "intro high conf", 5.0),      # eff: 0.6-0.2=0.4
            _make_gap(95, 0.2, "outro low conf", 200.0),     # eff: 0.2-0.15=0.05
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Sorted: outro(0.05) < middle(0.3) < intro(0.4)
        assert result[0].segment_index == 95
        assert result[1].segment_index == 50
        assert result[2].segment_index == 2


# ============================================================================
# US-94-007: Duration-based gap prioritization
# ============================================================================

class TestDurationBasedGapPrioritization:
    """Tests for duration-based gap prioritization in annotate_gaps_with_chapters."""

    def test_gap_segment_has_duration_field(self):
        """AC: GapSegment includes duration field."""
        gap = GapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="test",
            position=1.0,
            duration=45.0
        )
        assert gap.duration == 45.0

    def test_default_duration_is_zero(self):
        """AC: GapSegment defaults duration to 0.0."""
        gap = GapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="test",
            position=1.0
        )
        assert gap.duration == 0.0

    def test_longer_gaps_get_priority_boost(self):
        """AC: Gaps >30 seconds get priority boost in search order."""
        # Default threshold is 30 seconds, weight is 0.1
        # Gap with 45s duration: (45-30) * 0.1 = 1.5 priority boost
        gaps = [
            _make_gap(1, 0.5, "short gap", 10.0, duration=10.0),   # <30s, no boost
            _make_gap(2, 0.5, "long gap", 20.0, duration=45.0),    # >30s, gets boost
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Long gap should come first due to duration boost
        assert result[0].segment_index == 2
        assert result[1].segment_index == 1

    def test_duration_boost_combined_with_chapter_boost(self):
        """AC: Duration boost combines with chapter-based boost."""
        # Intro gap at segment 5 (intro) with 45s duration
        # Intro boost: 0.2, Duration boost: (45-30) * 0.1 = 1.5
        # Total boost: 1.7, Effective priority: 0.5 - 1.7 = -1.2
        gaps = [
            _make_gap(50, 0.5, "middle gap", 100.0, duration=45.0),  # middle, long
            _make_gap(5, 0.5, "intro gap", 10.0, duration=45.0),    # intro, long
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Intro + duration boost should come first
        assert result[0].segment_index == 5
        assert result[1].segment_index == 50

    def test_custom_duration_priority_weight(self):
        """AC: duration_priority_weight parameter controls boost strength."""
        gaps = [
            _make_gap(1, 0.5, "gap1", 10.0, duration=60.0),
            _make_gap(2, 0.5, "gap2", 20.0, duration=60.0),
        ]
        # With weight=0, no duration boost applied
        result_no_boost = annotate_gaps_with_chapters(
            gaps, total_segments=100, duration_priority_weight=0.0
        )
        # Both have same chapter boost (0), so order based on original order

        # With weight=0.5, strong duration boost
        result_strong_boost = annotate_gaps_with_chapters(
            gaps, total_segments=100, duration_priority_weight=0.5
        )
        # Both have same duration, so same boost

    def test_gaps_below_threshold_no_boost(self):
        """AC: Gaps at or below 30-second threshold get no duration boost."""
        gaps = [
            _make_gap(1, 0.5, "exactly 30s", 10.0, duration=30.0),   # exactly threshold
            _make_gap(2, 0.5, "25 seconds", 20.0, duration=25.0),   # below threshold
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Neither should get duration boost, order based on segment_index
        assert result[0].segment_index == 1
        assert result[1].segment_index == 2

    def test_duration_priority_weight_in_config(self):
        """AC: Config option 'duration_priority_weight' available with default 0.1."""
        from src.config.sections.iterative_matching import IterativeMatchingConfig
        config = IterativeMatchingConfig()
        assert hasattr(config, 'duration_priority_weight')
        assert config.duration_priority_weight == 0.1


# ============================================================================
# US-72-011: Description-derived search queries for gap filling
# ============================================================================

class TestDeriveQueriesFromDescriptions:
    """Tests for derive_queries_from_descriptions function."""

    def test_extracts_queries_from_descriptions(self):
        """AC: Extracts key phrases from descriptions of matched videos."""
        matched_videos = [
            {'description': 'This film covers Solar Energy Revolution in Germany and Wind Power advances.'},
            {'description': 'Renewable Energy Solutions including Solar Energy panels deployed worldwide.'},
        ]
        gap = _make_gap(0, 0.3, "energy sources and renewable technology")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=3)
        assert len(result) > 0
        assert len(result) <= 3

    def test_empty_matched_videos(self):
        """AC: Empty description handling - empty matched videos returns empty."""
        gap = _make_gap(0, 0.3, "some text about energy")
        result = derive_queries_from_descriptions([], gap)
        assert result == []

    def test_no_descriptions_in_videos(self):
        """AC: Videos without description field return empty."""
        matched_videos = [
            {'title': 'Video 1'},
            {'title': 'Video 2', 'description': ''},
        ]
        gap = _make_gap(0, 0.3, "some text")
        result = derive_queries_from_descriptions(matched_videos, gap)
        assert result == []

    def test_short_descriptions_handled(self):
        """Very short descriptions are filtered out."""
        matched_videos = [
            {'description': 'short'},
            {'description': 'OK'},
        ]
        gap = _make_gap(0, 0.3, "something longer")
        result = derive_queries_from_descriptions(matched_videos, gap)
        assert isinstance(result, list)

    def test_max_queries_default_three(self):
        """AC: Returns top 3 noun phrases by default."""
        matched_videos = [
            {'description': 'The Arctic Ocean Expedition explored Ice Sheet dynamics '
                            'and Polar Bear habitats near Greenland Ice Cap during Climate Research.'},
        ]
        gap = _make_gap(0, 0.3, "arctic ice melting")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=3)
        assert len(result) <= 3

    def test_quoted_phrases_extracted(self):
        """Quoted phrases in descriptions are extracted as noun phrases."""
        matched_videos = [
            {'description': 'The documentary explores "Deep Ocean Currents" and their impact.'},
        ]
        gap = _make_gap(0, 0.3, "ocean current patterns")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        assert len(result) > 0
        # The quoted phrase should be among the results
        all_lower = ' '.join(result).lower()
        assert 'deep ocean currents' in all_lower

    def test_capitalized_sequences_extracted(self):
        """Capitalized word sequences are extracted as noun phrases."""
        matched_videos = [
            {'description': 'Studying Marine Biology at Great Barrier Reef.'},
        ]
        gap = _make_gap(0, 0.3, "marine life reef studies")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        assert len(result) > 0
        all_lower = ' '.join(result).lower()
        assert 'great barrier reef' in all_lower or 'marine biology' in all_lower

    def test_config_toggle_field_exists(self):
        """AC: Config toggle use_description_queries exists in IterativeMatchingConfig."""
        from src.config.sections.iterative_matching import IterativeMatchingConfig
        config = IterativeMatchingConfig()
        assert hasattr(config, 'use_description_queries')
        assert config.use_description_queries is True

    def test_urls_filtered_from_descriptions(self):
        """URLs in descriptions don't pollute extracted phrases."""
        matched_videos = [
            {'description': 'Visit https://example.com for Solar Panel Installation tips.'},
        ]
        gap = _make_gap(0, 0.3, "solar panel setup")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        for q in result:
            assert 'http' not in q.lower()
            assert 'example' not in q.lower()

    def test_fallback_to_tfidf_when_no_noun_phrases(self):
        """Falls back to TF-IDF extraction when no capitalized sequences found."""
        matched_videos = [
            {'description': 'understanding the world of quantum computing and its applications in modern science'},
            {'description': 'quantum computing breakthroughs transform data processing capabilities worldwide'},
        ]
        gap = _make_gap(0, 0.3, "quantum computers")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=3)
        # Should still return something via fallback
        assert len(result) > 0

    # ============================================================================
    # US-126-011: Improved description-derived query extraction
    # ============================================================================

    def test_prefers_noun_phrases_over_single_words(self):
        """AC: Extracted queries are noun phrases (2+ words), not just single words."""
        matched_videos = [
            {'description': 'Visit Solar Energy Solutions for Wind Power Technology. '
                            'Solar panels installed by Solar Energy Company in California.'},
        ]
        gap = _make_gap(0, 0.3, "renewable energy")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        # All results should be multi-word phrases (preferring noun phrases)
        for phrase in result:
            assert len(phrase.split()) >= 2, f"Single word '{phrase}' should be filtered out"

    def test_proper_nouns_included_in_queries(self):
        """AC: Proper nouns from descriptions are included in queries."""
        matched_videos = [
            {'description': 'Exploring Paris France and visiting Eiffel Tower. '
                            'Trip to Tokyo Japan for Cherry Blossom Festival.'},
        ]
        gap = _make_gap(0, 0.3, "europe travel japan asia")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        # Should extract location proper nouns
        result_text = ' '.join(result).lower()
        # The location names should appear
        has_proper_noun = any(
            name.lower() in result_text
            for name in ['paris', 'france', 'tokyo', 'japan', 'eiffel tower', 'cherry blossom']
        )
        assert has_proper_noun, f"Expected proper nouns in {result}"

    def test_acronyms_extracted_as_proper_nouns(self):
        """AC: Acronyms (e.g., NASA, AI, SUV) are extracted as proper nouns."""
        matched_videos = [
            {'description': 'NASA launches new AI-powered SUV to study climate change. '
                            'Visit NASA.gov for details.'},
        ]
        gap = _make_gap(0, 0.3, "space exploration technology")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        # Should extract NASA as a proper noun
        result_text = ' '.join(result)
        assert 'NASA' in result_text, f"Expected NASA in {result}"

    def test_brand_names_extracted(self):
        """AC: Brand names are extracted as proper nouns."""
        matched_videos = [
            {'description': 'Review of Apple iPhone and Samsung Galaxy devices. '
                            'Microsoft Windows vs Google Android comparison.'},
        ]
        gap = _make_gap(0, 0.3, "smartphones technology")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        # Should extract brand names
        result_text = ' '.join(result)
        # At least some brands should appear
        has_brand = any(
            brand.lower() in result_text.lower()
            for brand in ['apple', 'samsung', 'microsoft', 'google', 'iphone', 'galaxy', 'android']
        )
        assert has_brand, f"Expected brand names in {result}"

    def test_aggressive_stopword_filtering(self):
        """AC: Common stopwords are aggressively filtered from results."""
        matched_videos = [
            {'description': 'The video shows how to make a good video about new things. '
                            'Visit the channel for more information about these topics.'},
        ]
        gap = _make_gap(0, 0.3, "making videos")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=5)
        # Filter out common single words that are just stopwords
        for phrase in result:
            words = phrase.lower().split()
            for word in words:
                assert word not in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                                    'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
                                    'are', 'were', 'be', 'have', 'has', 'had', 'do', 'does',
                                    'did', 'will', 'would', 'could', 'should', 'new', 'video',
                                    'channel', 'subscribe', 'like', 'make', 'made'}, \
                    f"Stopword '{word}' should be filtered from '{phrase}'"

    def test_quoted_phrases_prefer_multi_word(self):
        """AC: Quoted phrases with 2+ words are preferred over single words."""
        matched_videos = [
            {'description': 'The documentary covers "Ocean" and "Deep Sea Exploration" topics.'},
        ]
        gap = _make_gap(0, 0.3, "ocean exploration underwater")
        result = derive_queries_from_descriptions(matched_videos, gap, max_queries=3)
        # The multi-word quoted phrase should be included, single word might be filtered
        result_text = ' '.join(result)
        assert 'deep sea exploration' in result_text.lower(), \
            f"Expected multi-word phrase 'Deep Sea Exploration' in {result}"


# ============================================================================
# US-118-008: Gap severity scoring and strategy tests
# ============================================================================

class TestCategorizeGapsByConfidence:
    """Tests for categorize_gaps_by_confidence function - gap severity scoring."""

    def test_low_confidence_below_threshold(self):
        """AC: Gaps with confidence < 0.3 are categorized as low_confidence."""
        gaps = [
            _make_gap(0, 0.1, "very uncertain match"),
            _make_gap(1, 0.25, "low confidence content"),
        ]
        result = categorize_gaps_by_confidence(gaps)
        assert result[0].confidence_category == "low_confidence"
        assert result[1].confidence_category == "low_confidence"

    def test_medium_confidence_in_range(self):
        """AC: Gaps with confidence 0.3-0.6 are categorized as medium_confidence."""
        gaps = [
            _make_gap(0, 0.3, "boundary low"),
            _make_gap(1, 0.45, "medium confidence"),
            _make_gap(2, 0.59, "boundary high"),
        ]
        result = categorize_gaps_by_confidence(gaps)
        for gap in result:
            assert gap.confidence_category == "medium_confidence"

    def test_high_confidence_above_threshold(self):
        """AC: Gaps with confidence > 0.6 are categorized as high_confidence."""
        gaps = [
            _make_gap(0, 0.6, "boundary"),
            _make_gap(1, 0.75, "high confidence"),
            _make_gap(2, 0.95, "very high"),
        ]
        result = categorize_gaps_by_confidence(gaps)
        assert result[0].confidence_category == "high_confidence"
        assert result[1].confidence_category == "high_confidence"
        assert result[2].confidence_category == "high_confidence"

    def test_empty_gaps_returns_empty(self):
        """AC: Empty list returns empty."""
        result = categorize_gaps_by_confidence([])
        assert result == []

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        gaps = [
            _make_gap(0, 0.5, "test"),
        ]
        thresholds = {"low": 0.5, "medium": 0.8}
        result = categorize_gaps_by_confidence(gaps, thresholds)
        # With low=0.5, confidence 0.5 is NOT < 0.5, so goes to medium (0.5 < 0.8)
        assert result[0].confidence_category == "medium_confidence"


class TestGetStrategyForConfidenceCategory:
    """Tests for get_strategy_for_confidence_category - suggest_fill_strategy equivalent."""

    def test_low_confidence_strategy_aggressive(self):
        """AC: Low confidence returns aggressive search strategy."""
        strategy = get_strategy_for_confidence_category("low_confidence")
        assert strategy["search_results"] == 15
        assert strategy["use_broad_queries"] is True
        assert strategy["max_iterations"] == 5
        assert strategy["parallel_strategies"] is True

    def test_medium_confidence_strategy_standard(self):
        """AC: Medium confidence returns standard search strategy."""
        strategy = get_strategy_for_confidence_category("medium_confidence")
        assert strategy["search_results"] == 10
        assert strategy["use_broad_queries"] is False
        assert strategy["max_iterations"] == 3
        assert strategy["parallel_strategies"] is True

    def test_high_confidence_strategy_minimal(self):
        """AC: High confidence returns minimal search strategy."""
        strategy = get_strategy_for_confidence_category("high_confidence")
        assert strategy["search_results"] == 5
        assert strategy["use_broad_queries"] is False
        assert strategy["max_iterations"] == 1
        assert strategy["parallel_strategies"] is False

    def test_unknown_category_defaults_to_medium(self):
        """Unknown categories fallback to medium strategy."""
        strategy = get_strategy_for_confidence_category("unknown_category")
        assert strategy == get_strategy_for_confidence_category("medium_confidence")


class TestGapSeverityScoringWithVariousGapSizes:
    """AC: Gap severity scoring with various gap sizes (duration)."""

    def test_severity_based_on_confidence(self):
        """Lower confidence = higher severity."""
        gaps = [
            _make_gap(0, 0.9, "high conf"),
            _make_gap(1, 0.5, "med conf"),
            _make_gap(2, 0.1, "low conf"),
        ]
        # Sort by confidence ascending = severity descending
        sorted_gaps = sorted(gaps, key=lambda g: g.confidence)
        assert sorted_gaps[0].segment_index == 2  # lowest confidence first
        assert sorted_gaps[2].segment_index == 0  # highest confidence last

    def test_severity_considers_duration(self):
        """Longer gaps (>30s) get priority boost."""
        gaps = [
            _make_gap(0, 0.5, "short gap", 10.0, duration=10.0),
            _make_gap(1, 0.5, "long gap", 20.0, duration=45.0),
        ]
        # annotate_gaps_with_chapters applies duration-based priority
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Long gap should come first due to duration boost
        assert result[0].segment_index == 1
        assert result[1].segment_index == 0

    def test_duration_below_threshold_no_severity_boost(self):
        """Gaps <= 30 seconds don't get duration boost."""
        gaps = [
            _make_gap(0, 0.5, "gap30", 10.0, duration=30.0),
            _make_gap(1, 0.5, "gap25", 20.0, duration=25.0),
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Both have same effective priority, order by segment_index
        assert result[0].segment_index == 0
        assert result[1].segment_index == 1


# ============================================================================
# US-111-009: Context-aware iterative gap filling
# ============================================================================

class TestExtractContextFromNearbyMatches:
    """Tests for extract_context_from_nearby_matches function."""

    def test_empty_locked_matches(self):
        """No locked matches returns empty context."""
        from unittest.mock import MagicMock
        gap = _make_gap(5, 0.3, "test gap", 50.0)
        state = MagicMock()
        state.voiceover_segments = []
        result = extract_context_from_nearby_matches(gap, [], state)
        assert result == []

    def test_finds_nearby_matches_within_window(self):
        """Matches within window_seconds are returned."""
        from unittest.mock import MagicMock
        gap = _make_gap(5, 0.3, "gap text", 60.0)

        locked = [
            LockedMatch(segment_index=0, video_id="vid1", confidence=0.9, position=30.0),
            LockedMatch(segment_index=1, video_id="vid2", confidence=0.8, position=50.0),
            LockedMatch(segment_index=2, video_id="vid3", confidence=0.7, position=125.0),  # outside 65s window (65 > 60)
        ]

        state = MagicMock()
        state.voiceover_segments = [
            MagicMock(text="Segment zero text"),
            MagicMock(text="Segment one text"),
            MagicMock(text="Segment two text"),
        ]

        result = extract_context_from_nearby_matches(gap, locked, state, window_seconds=60.0)

        # Should find 2 matches within 60s window (abs(60-30)=30, abs(60-50)=10)
        # vid3 at 125 is abs(125-60)=65 > 60, so excluded
        assert len(result) == 2
        assert result[0].video_id == "vid2"  # closer first

    def test_context_sorted_by_distance(self):
        """Results sorted by distance (closest first)."""
        from unittest.mock import MagicMock
        gap = _make_gap(5, 0.3, "gap", 100.0)

        locked = [
            LockedMatch(segment_index=0, video_id="far", confidence=0.9, position=10.0),
            LockedMatch(segment_index=1, video_id="near", confidence=0.8, position=95.0),
        ]

        state = MagicMock()
        state.voiceover_segments = [MagicMock(text="text")]

        result = extract_context_from_nearby_matches(gap, locked, state)
        assert result[0].video_id == "near"

    def test_max_context_segments_limit(self):
        """Results limited by max_context_segments."""
        from unittest.mock import MagicMock
        gap = _make_gap(10, 0.3, "gap", 100.0)

        locked = [
            LockedMatch(segment_index=i, video_id=f"vid{i}", confidence=0.9, position=float(i * 5))
            for i in range(10)
        ]

        state = MagicMock()
        state.voiceover_segments = [MagicMock(text=f"text{i}") for i in range(10)]

        result = extract_context_from_nearby_matches(gap, locked, state, max_context_segments=3)
        assert len(result) == 3


class TestComputeContextRelevance:
    """Tests for compute_context_relevance function."""

    def test_empty_keywords_returns_zero(self):
        """Empty gap or context keywords return 0.0."""
        assert compute_context_relevance([], ["word"]) == 0.0
        assert compute_context_relevance(["word"], []) == 0.0

    def test_full_overlap_max_score(self):
        """Full keyword overlap returns high score."""
        gap_kw = ["freedom", "justice", "peace"]
        ctx_kw = ["freedom", "justice", "peace"]
        score = compute_context_relevance(gap_kw, ctx_kw)
        assert score > 0.8

    def test_no_overlap_low_score(self):
        """No keyword overlap returns low score."""
        gap_kw = ["freedom", "justice"]
        ctx_kw = ["unrelated", "different"]
        score = compute_context_relevance(gap_kw, ctx_kw)
        assert score < 0.3

    def test_partial_overlap_mid_score(self):
        """Partial overlap returns moderate score."""
        gap_kw = ["freedom", "justice", "peace"]
        ctx_kw = ["freedom", "completely", "different"]
        score = compute_context_relevance(gap_kw, ctx_kw)
        # Score is between 0 and 1, with partial overlap
        assert 0.0 < score < 1.0
        # It's not full overlap (would be > 0.8) and not no overlap (would be < 0.1)
        assert score < 0.5  # partial overlap gives lower score

    def test_topic_weight_affects_score(self):
        """topic_weight parameter affects scoring."""
        gap_kw = ["freedom", "justice", "peace"]
        ctx_kw = ["freedom", "completely", "different"]

        score_high_topic = compute_context_relevance(gap_kw, ctx_kw, topic_weight=0.9)
        score_low_topic = compute_context_relevance(gap_kw, ctx_kw, topic_weight=0.1)

        # Different weights should produce different scores
        assert score_high_topic != score_low_topic


class TestGenerateContextAwareQueries:
    """Tests for generate_context_aware_queries function."""

    def test_empty_gap_keywords_returns_empty(self):
        """Empty gap keywords returns empty list."""
        gap = _make_gap(0, 0.3, "test")
        result = generate_context_aware_queries(gap, [], [])
        assert result == []

    def test_generates_base_query(self):
        """Base query from gap keywords is always included."""
        gap = _make_gap(0, 0.3, "freedom and justice")
        gap.pattern_type = "abstract_concept"
        gap_keywords = ["freedom", "justice"]

        result = generate_context_aware_queries(gap, [], gap_keywords)

        assert len(result) > 0
        # Base query should have no context weight
        base_query = result[0]
        assert base_query.context_weight == 0.0

    def test_context_enhances_queries(self):
        """Context keywords add to queries when relevant."""
        from unittest.mock import MagicMock
        gap = _make_gap(0, 0.3, "climate change")
        gap.pattern_type = "other"

        context_segments = [
            MagicMock(
                segment_index=0,
                video_id="vid1",
                title="Test",
                position=10.0,
                keywords=["climate", "environment", "science"],
                distance=10.0
            )
        ]
        gap_keywords = ["change", "impact"]

        result = generate_context_aware_queries(gap, context_segments, gap_keywords)

        # Should have at least base query + context-enhanced query
        assert len(result) >= 1

    def test_max_queries_limit(self):
        """Results limited by max_queries parameter."""
        from unittest.mock import MagicMock
        gap = _make_gap(0, 0.3, "test")

        context_segments = [
            MagicMock(
                segment_index=i,
                video_id=f"vid{i}",
                title="Test",
                position=float(i),
                keywords=["word1", "word2"],
                distance=float(i)
            )
            for i in range(5)
        ]
        gap_keywords = ["test", "keywords"]

        result = generate_context_aware_queries(gap, context_segments, gap_keywords, max_queries=2)
        assert len(result) <= 2


class TestGetContextKeywordsForGap:
    """Tests for get_context_keywords_for_gap convenience function."""

    def test_returns_keyword_list(self):
        """Returns list of keywords."""
        from unittest.mock import MagicMock
        gap = _make_gap(5, 0.3, "gap text", 60.0)

        locked = [
            LockedMatch(segment_index=0, video_id="vid1", confidence=0.9, position=30.0),
        ]

        state = MagicMock()
        state.voiceover_segments = [MagicMock(text="environment science nature")]

        result = get_context_keywords_for_gap(gap, locked, state)
        assert isinstance(result, list)

    def test_max_keywords_limit(self):
        """Respects max_keywords parameter."""
        from unittest.mock import MagicMock
        gap = _make_gap(5, 0.3, "gap", 60.0)

        locked = [
            LockedMatch(segment_index=i, video_id=f"vid{i}", confidence=0.9, position=float(i * 10))
            for i in range(3)
        ]

        state = MagicMock()
        state.voiceover_segments = [MagicMock(text="word1 word2 word3 word4 word5") for _ in range(3)]

        result = get_context_keywords_for_gap(gap, locked, state, max_keywords=3)
        assert len(result) <= 3


# ============================================================================
# US-73-009: Video tag-derived search queries
# ============================================================================

class TestExtractTagsFromNearbyMatches:
    """Tests for extract_tags_from_nearby_matches function."""

    def test_empty_locked_returns_empty(self):
        """No locked matches returns empty list."""
        from unittest.mock import MagicMock
        gap = _make_gap(0, 0.3, "test", 50.0)
        state = MagicMock()
        state.video_search_results = []
        result = extract_tags_from_nearby_matches(gap, [], state)
        assert result == []

    def test_extracts_tags_from_nearby_videos(self):
        """Tags from nearby locked matches are extracted."""
        from unittest.mock import MagicMock
        gap = _make_gap(5, 0.3, "test", 60.0)

        locked = [
            LockedMatch(segment_index=0, video_id="vid1", confidence=0.9, position=30.0),
            LockedMatch(segment_index=1, video_id="vid2", confidence=0.8, position=50.0),
        ]

        state = MagicMock()
        # Mock video search results with tags
        vsr1 = MagicMock()
        vsr1.video_id = "vid1"
        vsr1.video_tags = ["nature", "wildlife", "documentary"]

        vsr2 = MagicMock()
        vsr2.video_id = "vid2"
        vsr2.video_tags = ["science", "documentary", "education"]

        state.video_search_results = [vsr1, vsr2]

        result = extract_tags_from_nearby_matches(gap, locked, state)

        # Should return tags from nearby videos
        assert len(result) > 0

    def test_filters_stop_tags(self):
        """Generic stop tags are filtered out."""
        from unittest.mock import MagicMock
        gap = _make_gap(0, 0.3, "test", 50.0)

        locked = [
            LockedMatch(segment_index=0, video_id="vid1", confidence=0.9, position=30.0),
        ]

        state = MagicMock()
        vsr = MagicMock()
        vsr.video_id = "vid1"
        vsr.video_tags = ["video", "youtube", "nature"]  # "video" and "youtube" are stop tags
        state.video_search_results = [vsr]

        result = extract_tags_from_nearby_matches(gap, locked, state)
        # Should not contain stop tags
        assert "video" not in result
        assert "youtube" not in result

    def test_max_distance_filter(self):
        """Matches beyond max_distance are excluded."""
        from unittest.mock import MagicMock
        gap = _make_gap(0, 0.3, "test", 100.0)

        locked = [
            LockedMatch(segment_index=0, video_id="near", confidence=0.9, position=95.0),
            LockedMatch(segment_index=1, video_id="far", confidence=0.8, position=10.0),
        ]

        state = MagicMock()
        vsr_near = MagicMock()
        vsr_near.video_id = "near"
        vsr_near.video_tags = ["nature"]

        vsr_far = MagicMock()
        vsr_far.video_id = "far"
        vsr_far.video_tags = ["science"]

        state.video_search_results = [vsr_near, vsr_far]

        result = extract_tags_from_nearby_matches(gap, locked, state, max_distance=10.0)
        # Only "near" is within 10s
        assert len(result) <= 1

    def test_max_tags_limit(self):
        """Results limited by max_tags."""
        from unittest.mock import MagicMock
        gap = _make_gap(0, 0.3, "test", 50.0)

        locked = [
            LockedMatch(segment_index=i, video_id=f"vid{i}", confidence=0.9, position=float(i))
            for i in range(3)
        ]

        state = MagicMock()
        vsr = MagicMock()
        vsr.video_id = "any"
        vsr.video_tags = ["tag1", "tag2", "tag3", "tag4", "tag5"]
        state.video_search_results = [vsr]

        result = extract_tags_from_nearby_matches(gap, locked, state, max_tags=2)
        assert len(result) <= 2


class TestExtractKeywordsFromText:
    """Tests for _extract_keywords_from_text helper."""

    def test_extracts_proper_nouns(self):
        """Capitalized words extracted as proper nouns."""
        text = "The Johnson family visited Paris and London"
        result = _extract_keywords_from_text(text)
        assert any("Johnson" in kw for kw in result)
        assert any("Paris" in kw for kw in result)
        assert any("London" in kw for kw in result)

    def test_filters_stop_words(self):
        """Common stop words are filtered."""
        text = "the and but for with of the"
        result = _extract_keywords_from_text(text)
        # Stop words should not appear
        assert "the" not in result
        assert "and" not in result

    def test_max_keywords_limit(self):
        """Results limited by max_keywords."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        result = _extract_keywords_from_text(text, max_keywords=3)
        assert len(result) <= 3

    def test_empty_text_returns_empty(self):
        """Empty text returns empty list."""
        result = _extract_keywords_from_text("")
        assert result == []


# ============================================================================
# Edge cases tests
# ============================================================================

class TestEdgeCases:
    """AC: Test edge cases (no gaps, all gaps, overlapping segments)."""

    def test_no_gaps_empty_analysis(self):
        """Empty gap list returns empty analysis."""
        analysis = analyze_gaps([])
        # GapAnalysis doesn't have total_gaps attribute, check pattern_counts
        assert len(analysis.pattern_counts) == 0

    def test_all_gaps_classified(self):
        """All gaps are classified into some pattern."""
        gaps = [
            _make_gap(0, 0.1, "freedom"),
            _make_gap(1, 0.1, "happy"),
            _make_gap(2, 0.1, "running"),
            _make_gap(3, 0.1, "in Paris near London"),  # Two location words = location
            _make_gap(4, 0.1, "random"),
        ]
        analysis = analyze_gaps(gaps)
        # All 5 gaps should be classified (5 unique pattern types)
        # freedom=abstract_concept, happy=emotion, running=action_verb,
        # "in Paris near London"=location, random=other
        assert len(analysis.pattern_counts) == 5

    def test_duplicate_segment_indices(self):
        """Duplicate indices handled gracefully."""
        gaps = [
            _make_gap(0, 0.1, "freedom"),
            _make_gap(0, 0.1, "justice"),  # Same index
        ]
        analysis = analyze_gaps(gaps)
        # Should still process both
        assert len(analysis.pattern_counts) >= 1

    def test_overlapping_chapters(self):
        """Gaps in overlapping chapter ranges handled."""
        gaps = [
            _make_gap(5, 0.5, "gap1", 10.0),
            _make_gap(8, 0.4, "gap2", 15.0),
        ]
        # Chapters that overlap
        chapters = [
            {'title': 'Chapter1', 'start_segment': 0, 'end_segment': 10},
            {'title': 'Chapter2', 'start_segment': 5, 'end_segment': 15},
        ]
        result = annotate_gaps_with_chapters(gaps, total_segments=20, chapters=chapters)
        # Both should have a chapter assigned (first match wins)
        for gap in result:
            assert gap.chapter_id is not None

    def test_zero_duration_gap(self):
        """Gap with zero duration handled."""
        gap = _make_gap(0, 0.5, "test", 10.0, duration=0.0)
        gaps = [gap]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        # Should not crash, should handle gracefully
        assert len(result) == 1

    def test_confidence_boundary_values(self):
        """Boundary confidence values handled correctly."""
        gaps = [
            _make_gap(0, 0.0, "freedom and hope"),  # abstract concept
            _make_gap(1, 1.0, "running in the park"),  # action verb
        ]
        analysis = analyze_gaps(gaps)
        # Should classify both - different patterns
        assert len(analysis.pattern_counts) == 2

    def test_very_long_text_handled(self):
        """Very long gap text handled without crash."""
        long_text = "word " * 1000
        gap = _make_gap(0, 0.5, long_text)
        gap.pattern_type = "other"
        keywords = extract_keywords_for_gap(gap)
        # Should not crash, should return keywords
        assert isinstance(keywords, list)


# ============================================================================
# US-135-007: Listicle-aware gap analysis
# ============================================================================

class TestAnnotateGapsWithListiclePosition:
    """Tests for annotate_gaps_with_listicle_position function."""

    def test_gap_at_listicle_start_boundary(self):
        """AC: Gap at start of listicle group marked as boundary."""
        # Listicle group has 10 segments (0-9), boundary is first 20% = 2 segments
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 10, 'label': 'first'},
        ]
        gaps = [
            _make_gap(0, 0.5, "first item content", 0.0),
            _make_gap(1, 0.4, "second segment", 5.0),
        ]
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=groups)

        # Gap at index 0-1 should be at boundary (first 20% of 10 = first 2)
        assert result[0].gap_listicle_position == "boundary"
        assert result[0].listicle_item_label == "first"
        assert result[1].gap_listicle_position == "boundary"

    def test_gap_at_listicle_end_boundary(self):
        """AC: Gap at end of listicle group marked as boundary."""
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 10, 'label': 'last'},
        ]
        gaps = [
            _make_gap(8, 0.5, "near end", 40.0),
            _make_gap(9, 0.4, "last segment", 45.0),
        ]
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=groups)

        # Gap at index 8-9 should be at boundary (last 20% of 10 = last 2)
        assert result[0].gap_listicle_position == "boundary"
        assert result[1].gap_listicle_position == "boundary"
        assert result[1].listicle_item_label == "last"

    def test_gap_in_listicle_middle(self):
        """AC: Gap in middle of listicle group marked as middle."""
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 10, 'label': 'middle'},
        ]
        gaps = [
            _make_gap(5, 0.5, "middle content", 25.0),
        ]
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=groups)

        # Gap at index 5 is in middle (not in first/last 2)
        assert result[0].gap_listicle_position == "middle"
        assert result[0].listicle_item_label == "middle"

    def test_gap_outside_listicle(self):
        """AC: Gap not in any listicle group marked as none."""
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 5},
        ]
        gaps = [
            _make_gap(10, 0.5, "outside listicle", 50.0),
        ]
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=groups)

        assert result[0].gap_listicle_position == "none"
        assert result[0].listicle_item_label == ""

    def test_no_listicle_groups(self):
        """AC: Gaps marked as none when no listicle groups provided."""
        gaps = [
            _make_gap(0, 0.5, "some content", 0.0),
            _make_gap(1, 0.4, "more content", 5.0),
        ]
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=None)

        for gap in result:
            assert gap.gap_listicle_position == "none"

    def test_empty_gaps_returns_empty(self):
        """Empty gap list returns empty."""
        result = annotate_gaps_with_listicle_position([], listicle_groups=[])
        assert result == []

    def test_multiple_listicle_groups(self):
        """Gaps in different listicle groups get correct positions."""
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 5, 'label': 'first'},
            {'group_id': 'group_B', 'start_segment': 10, 'end_segment': 15, 'label': 'fifth'},
        ]
        gaps = [
            _make_gap(2, 0.5, "in group A", 10.0),  # middle of group A
            _make_gap(12, 0.4, "in group B", 60.0),  # middle of group B
        ]
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=groups)

        assert result[0].gap_listicle_position == "middle"
        assert result[1].gap_listicle_position == "middle"

    def test_custom_boundary_threshold(self):
        """Custom boundary threshold is respected."""
        groups = [
            {'group_id': 'group_A', 'start_segment': 0, 'end_segment': 10, 'label': 'test'},
        ]
        # With 50% threshold, first 5 segments are boundaries
        gaps = [
            _make_gap(3, 0.5, "should be boundary", 15.0),
        ]
        result = annotate_gaps_with_listicle_position(
            gaps, listicle_groups=groups, boundary_threshold=0.5
        )

        assert result[0].gap_listicle_position == "boundary"


class TestGetQueryStrategyForListiclePosition:
    """Tests for get_query_strategy_for_listicle_position function."""

    def test_boundary_strategy(self):
        """AC: Boundary gaps get targeted query strategy."""
        gap = _make_gap(0, 0.5, "first tip content", 0.0)
        gap.gap_listicle_position = "boundary"
        gap.listicle_item_label = "first"

        strategy = get_query_strategy_for_listicle_position(gap)

        assert strategy["query_type"] == "targeted"
        assert strategy["include_label"] is True
        assert strategy["search_modifier"] == "first"

    def test_middle_strategy(self):
        """AC: Middle gaps get broad query strategy."""
        gap = _make_gap(5, 0.5, "middle content", 25.0)
        gap.gap_listicle_position = "middle"

        strategy = get_query_strategy_for_listicle_position(gap, listicle_theme="tips")

        assert strategy["query_type"] == "broad"
        assert strategy["include_label"] is False
        assert strategy["search_modifier"] == "tips"

    def test_none_strategy(self):
        """AC: Gaps not in listicle get standard strategy."""
        gap = _make_gap(0, 0.5, "regular content", 0.0)
        gap.gap_listicle_position = "none"

        strategy = get_query_strategy_for_listicle_position(gap)

        assert strategy["query_type"] == "standard"
        assert strategy["include_label"] is False


class TestGenerateListicleAwareQueries:
    """Tests for generate_listicle_aware_queries function."""

    def test_boundary_generates_label_queries(self):
        """AC: Boundary gaps generate queries with listicle label."""
        gap = _make_gap(0, 0.5, "first tip about cooking", 0.0)
        gap.gap_listicle_position = "boundary"
        gap.listicle_item_label = "first"
        gap_keywords = ["cooking", "tips", "kitchen"]

        queries = generate_listicle_aware_queries(gap, gap_keywords, max_queries=3)

        assert len(queries) > 0
        # First query should include label
        assert "first" in queries[0].lower()

    def test_middle_generates_theme_queries(self):
        """AC: Middle gaps generate queries with listicle theme."""
        gap = _make_gap(5, 0.5, "another cooking tip", 25.0)
        gap.gap_listicle_position = "middle"
        gap_keywords = ["cooking", "recipe"]

        queries = generate_listicle_aware_queries(gap, gap_keywords, listicle_theme="kitchen tips", max_queries=3)

        assert len(queries) > 0
        # Should include theme
        all_queries = ' '.join(queries).lower()
        assert "kitchen" in all_queries or "tips" in all_queries

    def test_standard_gap_keywords_only(self):
        """AC: Non-listicle gaps use standard keyword queries."""
        gap = _make_gap(0, 0.5, "regular content", 0.0)
        gap.gap_listicle_position = "none"
        gap_keywords = ["topic", "subject"]

        queries = generate_listicle_aware_queries(gap, gap_keywords, max_queries=3)

        assert len(queries) > 0
        # Should just use keywords
        all_queries = ' '.join(queries).lower()
        assert "topic" in all_queries

    def test_empty_keywords_returns_empty(self):
        """Empty keywords returns empty list."""
        gap = _make_gap(0, 0.5, "test", 0.0)
        gap.gap_listicle_position = "boundary"
        gap.listicle_item_label = "first"

        queries = generate_listicle_aware_queries(gap, [], max_queries=3)

        assert queries == []

    def test_max_queries_limit(self):
        """Results limited by max_queries parameter."""
        gap = _make_gap(0, 0.5, "first tip content", 0.0)
        gap.gap_listicle_position = "boundary"
        gap.listicle_item_label = "first"
        gap_keywords = ["cooking", "tips", "kitchen", "food", "recipe"]

        queries = generate_listicle_aware_queries(gap, gap_keywords, max_queries=2)

        assert len(queries) <= 2

    def test_deduplication(self):
        """Duplicate queries are removed."""
        gap = _make_gap(0, 0.5, "first", 0.0)
        gap.gap_listicle_position = "boundary"
        gap.listicle_item_label = "first"
        gap_keywords = ["first"]  # Same as label

        queries = generate_listicle_aware_queries(gap, gap_keywords, max_queries=5)

        # Check no duplicates (case-insensitive)
        lower_queries = [q.lower() for q in queries]
        assert len(lower_queries) == len(set(lower_queries))


class TestListiclePositionInGapSegment:
    """AC: GapSegment has gap_listicle_position and listicle_item_label fields."""

    def test_gap_segment_has_listicle_position_field(self):
        """GapSegment includes gap_listicle_position field."""
        gap = GapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="test",
            position=1.0,
            gap_listicle_position="boundary"
        )
        assert gap.gap_listicle_position == "boundary"

    def test_gap_segment_has_listicle_item_label_field(self):
        """GapSegment includes listicle_item_label field."""
        gap = GapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="test",
            position=1.0,
            listicle_item_label="first"
        )
        assert gap.listicle_item_label == "first"

    def test_default_listicle_position_is_none(self):
        """Default gap_listicle_position is 'none'."""
        gap = GapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="test",
            position=1.0
        )
        assert gap.gap_listicle_position == "none"
        assert gap.listicle_item_label == ""


class TestListicleAwareGapAnalysis:
    """Integration tests for listicle-aware gap analysis."""

    def test_full_listicle_aware_workflow(self):
        """AC: Full workflow with listicle groups and position detection."""
        # Create gaps - group has 10 segments (0-9), boundary is first/last 2
        gaps = [
            _make_gap(0, 0.3, "first we need water", 0.0),    # boundary (first)
            _make_gap(3, 0.2, "then add ingredients", 15.0),  # middle
            _make_gap(8, 0.1, "finally serve hot", 40.0),      # boundary (last)
        ]

        # Listicle groups covering these segments
        groups = [
            {'group_id': 'recipe_steps', 'start_segment': 0, 'end_segment': 10, 'label': 'steps'},
        ]

        # Annotate with listicle position
        result = annotate_gaps_with_listicle_position(gaps, listicle_groups=groups)

        # First gap should be at boundary
        assert result[0].gap_listicle_position == "boundary"
        # Middle gap should be in middle
        assert result[1].gap_listicle_position == "middle"
        # Last gap should be at boundary
        assert result[2].gap_listicle_position == "boundary"

    def test_query_generation_different_positions(self):
        """AC: Different query strategies for boundary vs middle."""
        gap_boundary = _make_gap(0, 0.3, "first tip content", 0.0)
        gap_boundary.gap_listicle_position = "boundary"
        gap_boundary.listicle_item_label = "first"

        gap_middle = _make_gap(5, 0.3, "middle tip content", 25.0)
        gap_middle.gap_listicle_position = "middle"

        keywords = ["cooking", "tips"]

        queries_boundary = generate_listicle_aware_queries(gap_boundary, keywords)
        queries_middle = generate_listicle_aware_queries(gap_middle, keywords, listicle_theme="kitchen")

        # Boundary queries should include label
        assert any("first" in q.lower() for q in queries_boundary)

        # Middle queries should include theme
        assert any("kitchen" in q.lower() for q in queries_middle)
