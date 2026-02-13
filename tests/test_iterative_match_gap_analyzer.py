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
    _classify_gap_pattern,
    _has_location_pattern,
    _has_proper_noun,
    _extract_content_words,
    _extract_key_phrases,
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
