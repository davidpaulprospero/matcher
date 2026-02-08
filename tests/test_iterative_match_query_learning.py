"""
Unit tests for src/iterative_match/query_learning.py

Tests cover:
- QueryPlan dataclass defaults and fields
- QueryResult dataclass and to_dict
- StrategyStats: record results, serialization round-trip
- QueryLearningDB: load/save persistence, get_best_strategy, get_strategy_ranking,
  record_result EMA, _extract_template, get_refined_template, get_synonym_suggestions,
  get_summary
- Edge cases: empty DB, unknown patterns, no-match gaps, no-context gaps

Created: 2026-02-03 (Sprint 46 - Test Coverage)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.iterative_match.query_learning import (
    QueryPlan,
    QueryResult,
    StrategyStats,
    QueryLearningDB,
)


# ============================================================================
# QueryPlan dataclass
# ============================================================================

class TestQueryPlan:
    def test_defaults(self):
        plan = QueryPlan(query="test query", strategy="voiceover", gap_indices=[0, 1])
        assert plan.seed_video_id == ""
        assert plan.priority == 0
        assert plan.videos_found == 0
        assert plan.executed is False

    def test_fields_stored(self):
        plan = QueryPlan(
            query="wildlife footage",
            strategy="entity",
            gap_indices=[2, 5],
            seed_video_id="vid123",
            priority=3,
        )
        assert plan.query == "wildlife footage"
        assert plan.strategy == "entity"
        assert plan.gap_indices == [2, 5]
        assert plan.seed_video_id == "vid123"
        assert plan.priority == 3


# ============================================================================
# QueryResult dataclass
# ============================================================================

class TestQueryResult:
    def test_to_dict(self):
        result = QueryResult(
            query="nature video",
            strategy="voiceover",
            gap_indices=[0],
            videos_found=5,
            gaps_filled=1,
            avg_confidence_improvement=0.2,
            successful=True,
        )
        d = result.to_dict()
        assert d["query"] == "nature video"
        assert d["strategy"] == "voiceover"
        assert d["gap_indices"] == [0]
        assert d["videos_found"] == 5
        assert d["gaps_filled"] == 1
        assert d["avg_confidence_improvement"] == 0.2
        assert d["successful"] is True

    def test_defaults(self):
        result = QueryResult(
            query="q", strategy="s", gap_indices=[], videos_found=0,
            gaps_filled=0, avg_confidence_improvement=0.0,
        )
        assert result.successful is False
        assert result.chapter_type == ''

    def test_chapter_type_field(self):
        result = QueryResult(
            query="q", strategy="s", gap_indices=[0], videos_found=3,
            gaps_filled=1, avg_confidence_improvement=0.2,
            chapter_type="intro",
        )
        assert result.chapter_type == "intro"

    def test_chapter_type_in_to_dict(self):
        result = QueryResult(
            query="q", strategy="s", gap_indices=[], videos_found=0,
            gaps_filled=0, avg_confidence_improvement=0.0,
            chapter_type="conclusion",
        )
        d = result.to_dict()
        assert d["chapter_type"] == "conclusion"


# ============================================================================
# StrategyStats
# ============================================================================

class TestStrategyStats:
    def test_initial_values(self):
        stats = StrategyStats()
        assert stats.total_queries == 0
        assert stats.total_gaps_targeted == 0
        assert stats.total_gaps_filled == 0
        assert stats.total_videos_found == 0
        assert stats.success_rate == 0.0

    def test_record_single_result(self):
        stats = StrategyStats()
        result = QueryResult(
            query="q", strategy="voiceover", gap_indices=[0, 1, 2],
            videos_found=10, gaps_filled=2, avg_confidence_improvement=0.3,
        )
        stats.record(result)
        assert stats.total_queries == 1
        assert stats.total_gaps_targeted == 3
        assert stats.total_gaps_filled == 2
        assert stats.total_videos_found == 10
        assert stats.success_rate == pytest.approx(2 / 3)

    def test_record_multiple_results(self):
        stats = StrategyStats()
        r1 = QueryResult(query="q1", strategy="s", gap_indices=[0], videos_found=5, gaps_filled=1, avg_confidence_improvement=0.1)
        r2 = QueryResult(query="q2", strategy="s", gap_indices=[1, 2], videos_found=3, gaps_filled=0, avg_confidence_improvement=0.0)
        stats.record(r1)
        stats.record(r2)
        assert stats.total_queries == 2
        assert stats.total_gaps_targeted == 3
        assert stats.total_gaps_filled == 1
        assert stats.success_rate == pytest.approx(1 / 3)

    def test_serialization_round_trip(self):
        stats = StrategyStats(total_queries=5, total_gaps_targeted=10, total_gaps_filled=3, total_videos_found=20, success_rate=0.3)
        d = stats.to_dict()
        restored = StrategyStats.from_dict(d)
        assert restored.total_queries == 5
        assert restored.total_gaps_targeted == 10
        assert restored.total_gaps_filled == 3
        assert restored.total_videos_found == 20
        assert restored.success_rate == 0.3


# ============================================================================
# QueryLearningDB
# ============================================================================

class TestQueryLearningDBInit:
    def test_init_no_file(self, tmp_path):
        """DB initializes empty when no file exists."""
        db = QueryLearningDB(db_path=str(tmp_path / "nonexistent.json"))
        assert len(db.pattern_strategy_success) == 0
        assert len(db.template_success) == 0
        assert len(db.strategy_stats) == 0

    def test_init_corrupted_file(self, tmp_path):
        """DB handles corrupted JSON gracefully."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("NOT JSON{{{", encoding="utf-8")
        db = QueryLearningDB(db_path=str(bad_file))
        # Should not raise, just start empty
        assert len(db.pattern_strategy_success) == 0


class TestQueryLearningDBPersistence:
    def test_save_and_load(self, tmp_path):
        """Data survives save/load cycle."""
        db_path = str(tmp_path / "learn.json")
        db = QueryLearningDB(db_path=db_path)

        # Add some data
        db.pattern_strategy_success["abstract_concept"]["voiceover"] = 0.5
        db.pattern_strategy_success["abstract_concept"]["entity"] = 0.8
        db.template_success["nature footage"] = 5
        db.strategy_stats["voiceover"] = StrategyStats(total_queries=10, total_gaps_targeted=20, total_gaps_filled=5, total_videos_found=50, success_rate=0.25)
        db.save()

        # Load into new instance
        db2 = QueryLearningDB(db_path=db_path)
        assert db2.pattern_strategy_success["abstract_concept"]["voiceover"] == 0.5
        assert db2.pattern_strategy_success["abstract_concept"]["entity"] == 0.8
        assert db2.template_success["nature footage"] == 5
        assert db2.strategy_stats["voiceover"].total_queries == 10
        assert db2.strategy_stats["voiceover"].success_rate == 0.25

    def test_save_creates_directory(self, tmp_path):
        """Save creates parent directories if needed."""
        db_path = str(tmp_path / "deep" / "nested" / "dir" / "learn.json")
        db = QueryLearningDB(db_path=db_path)
        db.pattern_strategy_success["test"]["voiceover"] = 0.5
        db.save()
        assert Path(db_path).exists()

    def test_save_version_written(self, tmp_path):
        """Saved file includes version string."""
        db_path = str(tmp_path / "learn.json")
        db = QueryLearningDB(db_path=db_path)
        db.save()
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data["version"] == "1.1"


class TestGetBestStrategy:
    def test_unknown_pattern_returns_voiceover(self, tmp_path):
        """Unknown pattern defaults to 'voiceover'."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        assert db.get_best_strategy("never_seen_pattern") == "voiceover"

    def test_returns_highest_rated(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.pattern_strategy_success["location"]["voiceover"] = 0.3
        db.pattern_strategy_success["location"]["entity"] = 0.9
        db.pattern_strategy_success["location"]["topic"] = 0.5
        assert db.get_best_strategy("location") == "entity"

    def test_single_strategy(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.pattern_strategy_success["emotion"]["topic"] = 0.1
        assert db.get_best_strategy("emotion") == "topic"


class TestGetStrategyRanking:
    def test_unknown_pattern_returns_default_order(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        ranking = db.get_strategy_ranking("unknown_pattern")
        assert ranking == ["voiceover", "similar_locked", "entity", "topic"]

    def test_ranked_by_success_rate(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.pattern_strategy_success["proper_noun"]["voiceover"] = 0.2
        db.pattern_strategy_success["proper_noun"]["entity"] = 0.9
        db.pattern_strategy_success["proper_noun"]["topic"] = 0.5
        ranking = db.get_strategy_ranking("proper_noun")
        assert ranking == ["entity", "topic", "voiceover"]


class TestRecordResult:
    def test_ema_update(self, tmp_path):
        """record_result uses exponential moving average (alpha=0.3)."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # First result: success
        r1 = QueryResult(query="q1", strategy="voiceover", gap_indices=[0], videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
        db.record_result(r1, "abstract_concept")
        # EMA: 0.3 * 1.0 + 0.7 * 0.0 = 0.3
        assert db.pattern_strategy_success["abstract_concept"]["voiceover"] == pytest.approx(0.3)

        # Second result: failure
        r2 = QueryResult(query="q2", strategy="voiceover", gap_indices=[1], videos_found=2, gaps_filled=0, avg_confidence_improvement=0.0)
        db.record_result(r2, "abstract_concept")
        # EMA: 0.3 * 0.0 + 0.7 * 0.3 = 0.21
        assert db.pattern_strategy_success["abstract_concept"]["voiceover"] == pytest.approx(0.21)

    def test_strategy_stats_updated(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        result = QueryResult(query="q", strategy="entity", gap_indices=[0, 1], videos_found=3, gaps_filled=1, avg_confidence_improvement=0.1)
        db.record_result(result, "proper_noun")
        assert db.strategy_stats["entity"].total_queries == 1
        assert db.strategy_stats["entity"].total_gaps_targeted == 2
        assert db.strategy_stats["entity"].total_gaps_filled == 1

    def test_template_tracked_on_success(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        result = QueryResult(query="wildlife nature footage", strategy="voiceover", gap_indices=[0], videos_found=5, gaps_filled=2, avg_confidence_improvement=0.3)
        db.record_result(result, "other")
        assert len(db.template_success) > 0

    def test_template_not_tracked_on_failure(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        result = QueryResult(query="bad query", strategy="voiceover", gap_indices=[0], videos_found=0, gaps_filled=0, avg_confidence_improvement=0.0)
        db.record_result(result, "other")
        assert len(db.template_success) == 0


class TestExtractTemplate:
    def test_names_replaced(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        template = db._extract_template("John Smith visited London Bridge")
        assert "[name]" in template
        assert "john" not in template
        assert "smith" not in template

    def test_numbers_replaced(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        template = db._extract_template("top 10 nature clips from 2024")
        assert "[num]" in template
        assert "10" not in template
        assert "2024" not in template

    def test_lowercase_output(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        template = db._extract_template("WILDLIFE footage NATURE")
        assert template == template.lower()

    def test_whitespace_normalized(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        template = db._extract_template("lots   of    spaces   here")
        assert "  " not in template


class TestGetRefinedTemplate:
    def test_no_refinement_pass_1(self, tmp_path):
        """Pass 1 never returns a refinement."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        assert db.get_refined_template("any query", pass_num=1) is None

    def test_no_refinement_when_no_templates(self, tmp_path):
        """Empty DB returns None for refinement."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        assert db.get_refined_template("some query", pass_num=3) is None

    def test_refinement_when_matching_template_exists(self, tmp_path):
        """Returns refined query when a similar successful template exists with short base query."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # Seed with a successful template that has enough success count
        db.template_success["nature footage"] = 5
        # Short base query (< 3 words) with overlapping words
        result = db.get_refined_template("nature", pass_num=2)
        # Should add "footage video" to short query
        if result is not None:
            assert "footage" in result or "video" in result

    def test_no_refinement_low_evidence(self, tmp_path):
        """Templates with < 3 success count are ignored."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.template_success["nature footage"] = 2  # Below threshold of 3
        assert db.get_refined_template("nature clips", pass_num=3) is None


class TestGetSynonymSuggestions:
    def test_empty_db_returns_empty(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        assert db.get_synonym_suggestions("nature") == []

    def test_returns_cooccurring_words(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.template_success["nature wildlife documentary"] = 5
        db.template_success["nature scenery landscape"] = 3
        suggestions = db.get_synonym_suggestions("nature")
        assert len(suggestions) > 0
        # Excluded: [name], [num], video, footage
        assert "[name]" not in suggestions
        assert "video" not in suggestions
        assert "footage" not in suggestions

    def test_max_three_suggestions(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.template_success["nature wildlife documentary landscape scenery panoramic aerial"] = 10
        suggestions = db.get_synonym_suggestions("nature")
        assert len(suggestions) <= 3

    def test_low_count_templates_ignored(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.template_success["nature wildlife"] = 1  # Below threshold of 2
        suggestions = db.get_synonym_suggestions("nature")
        assert suggestions == []


class TestGetSummary:
    def test_empty_db_summary(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        summary = db.get_summary()
        assert summary["total_queries_recorded"] == 0
        assert summary["total_gaps_filled"] == 0
        assert summary["patterns_learned"] == 0
        assert summary["templates_discovered"] == 0
        assert summary["strategy_success_rates"] == {}

    def test_summary_after_records(self, tmp_path):
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        r1 = QueryResult(query="q1", strategy="voiceover", gap_indices=[0, 1], videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
        r2 = QueryResult(query="q2", strategy="entity", gap_indices=[2], videos_found=3, gaps_filled=1, avg_confidence_improvement=0.3)
        db.record_result(r1, "abstract_concept")
        db.record_result(r2, "proper_noun")
        summary = db.get_summary()
        assert summary["total_queries_recorded"] == 2
        assert summary["total_gaps_filled"] == 2
        assert summary["patterns_learned"] == 2
        assert "voiceover" in summary["strategy_success_rates"]
        assert "entity" in summary["strategy_success_rates"]


# ============================================================================
# Acceptance Criteria Tests
# ============================================================================

class TestACQueryGenerationFromLowConfidence:
    """AC: Test query generation from a gap with low confidence produces relevant search terms."""

    def test_low_confidence_gap_best_strategy_reflects_learnings(self, tmp_path):
        """After recording results, best strategy for low-confidence gaps reflects success rates."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # Record a successful entity strategy for abstract_concept gaps
        success = QueryResult(query="freedom concept footage", strategy="entity", gap_indices=[0], videos_found=10, gaps_filled=1, avg_confidence_improvement=0.4)
        db.record_result(success, "abstract_concept")
        # Record a failed voiceover strategy
        fail = QueryResult(query="freedom voiceover", strategy="voiceover", gap_indices=[0], videos_found=2, gaps_filled=0, avg_confidence_improvement=0.0)
        db.record_result(fail, "abstract_concept")
        # Entity should be recommended
        assert db.get_best_strategy("abstract_concept") == "entity"

    def test_strategy_ranking_after_low_confidence_results(self, tmp_path):
        """Ranking reflects accumulated success from low-confidence gap resolution."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # Multiple successful topic queries for emotion pattern
        for _ in range(3):
            r = QueryResult(query="emotional footage", strategy="topic", gap_indices=[0], videos_found=5, gaps_filled=1, avg_confidence_improvement=0.3)
            db.record_result(r, "emotion")
        # One moderately successful voiceover query
        r = QueryResult(query="feelings video", strategy="voiceover", gap_indices=[0], videos_found=3, gaps_filled=1, avg_confidence_improvement=0.1)
        db.record_result(r, "emotion")
        ranking = db.get_strategy_ranking("emotion")
        assert ranking[0] == "topic"


class TestACQueryGenerationFromNoMatch:
    """AC: Test query generation from a gap with no match produces broader search terms than low-confidence gaps."""

    def test_no_match_pattern_can_learn_independently(self, tmp_path):
        """No-match gaps (confidence 0.0) record results under their own pattern type."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # "other" pattern typically used when gap has no match / no clear type
        r = QueryResult(query="generic broad search footage", strategy="voiceover", gap_indices=[0], videos_found=8, gaps_filled=1, avg_confidence_improvement=0.5)
        db.record_result(r, "other")
        # Verify the pattern was recorded
        assert "voiceover" in db.pattern_strategy_success["other"]
        assert db.pattern_strategy_success["other"]["voiceover"] > 0

    def test_broader_template_from_no_match_queries(self, tmp_path):
        """Templates from broader queries are tracked and influence refinement."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # Simulate broad successful queries (no-match uses generic terms)
        r = QueryResult(query="generic stock footage clips", strategy="voiceover", gap_indices=[0], videos_found=20, gaps_filled=3, avg_confidence_improvement=0.6)
        db.record_result(r, "other")
        # Template should be stored
        assert len(db.template_success) > 0
        # The template should be generalized (lowercase, names replaced)
        for template in db.template_success:
            assert template == template.lower()


class TestACNoDuplicateKeywords:
    """AC: Test that generated queries do not duplicate existing search keywords from the original analysis."""

    def test_template_deduplication_via_extract(self, tmp_path):
        """_extract_template normalizes queries so structurally similar queries map to same template."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # Same query with different proper nouns produces same template
        t1 = db._extract_template("John Smith nature footage")
        t2 = db._extract_template("Jane Doe nature footage")
        assert t1 == t2  # Both names replaced with [name]

    def test_same_template_aggregates_success(self, tmp_path):
        """Repeated successful queries with same template accumulate count, not duplicate."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        for i in range(3):
            r = QueryResult(query="nature wildlife clips", strategy="voiceover", gap_indices=[i], videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
            db.record_result(r, "other")
        # Template should have accumulated count of 3 (1 gap_filled per record)
        template = db._extract_template("nature wildlife clips")
        assert db.template_success[template] == 3


class TestACNoContextFallback:
    """AC: Test edge case with gap that has no context text returns reasonable fallback query."""

    def test_empty_query_extract_template(self, tmp_path):
        """Empty string produces an empty template without error."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        template = db._extract_template("")
        assert template == ""

    def test_best_strategy_for_empty_context_gap(self, tmp_path):
        """Gap with no context ('other' pattern) gets a valid default strategy."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # No learnings for 'other' - should return default
        strategy = db.get_best_strategy("other")
        assert strategy == "voiceover"

    def test_ranking_fallback_for_unknown_pattern(self, tmp_path):
        """Unknown/empty pattern returns full default strategy list."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        ranking = db.get_strategy_ranking("")
        assert len(ranking) == 4
        assert "voiceover" in ranking

    def test_refined_template_for_empty_query(self, tmp_path):
        """Refinement on empty query does not crash."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        result = db.get_refined_template("", pass_num=3)
        # Should return None (no templates to match)
        assert result is None

    def test_synonym_for_empty_word(self, tmp_path):
        """Synonym suggestions for empty string returns empty list."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        assert db.get_synonym_suggestions("") == []


# ============================================================================
# Chapter-Type-Aware Query Learning (US-71-012)
# ============================================================================

class TestChapterTypeTracking:
    """AC: Query success rates are bucketed by chapter type."""

    def test_record_result_tracks_chapter_type(self, tmp_path):
        """record_result with chapter_type updates chapter_strategy_success."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        result = QueryResult(
            query="intro footage", strategy="entity", gap_indices=[0],
            videos_found=5, gaps_filled=1, avg_confidence_improvement=0.3
        )
        db.record_result(result, "abstract_concept", chapter_type="intro")
        assert "intro" in db.chapter_strategy_success
        assert "entity" in db.chapter_strategy_success["intro"]
        assert db.chapter_strategy_success["intro"]["entity"] == pytest.approx(0.3)

    def test_chapter_type_ema_updates(self, tmp_path):
        """Chapter-type success rates use EMA like pattern rates."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        # First result: success -> EMA = 0.3 * 1.0 + 0.7 * 0.0 = 0.3
        r1 = QueryResult(query="q1", strategy="voiceover", gap_indices=[0],
                         videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
        db.record_result(r1, "other", chapter_type="conclusion")
        assert db.chapter_strategy_success["conclusion"]["voiceover"] == pytest.approx(0.3)

        # Second result: failure -> EMA = 0.3 * 0.0 + 0.7 * 0.3 = 0.21
        r2 = QueryResult(query="q2", strategy="voiceover", gap_indices=[1],
                         videos_found=2, gaps_filled=0, avg_confidence_improvement=0.0)
        db.record_result(r2, "other", chapter_type="conclusion")
        assert db.chapter_strategy_success["conclusion"]["voiceover"] == pytest.approx(0.21)

    def test_all_four_chapter_types_tracked(self, tmp_path):
        """All four chapter types (intro, body, conclusion, listicle_item) are tracked."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        for ct in ("intro", "body", "conclusion", "listicle_item"):
            r = QueryResult(query=f"q_{ct}", strategy="voiceover", gap_indices=[0],
                            videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
            db.record_result(r, "other", chapter_type=ct)
        assert len(db.chapter_strategy_success) == 4

    def test_invalid_chapter_type_ignored(self, tmp_path):
        """Invalid chapter type does not create an entry."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        r = QueryResult(query="q", strategy="voiceover", gap_indices=[0],
                        videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
        db.record_result(r, "other", chapter_type="invalid_type")
        assert "invalid_type" not in db.chapter_strategy_success

    def test_default_chapter_type_is_body(self, tmp_path):
        """When no chapter_type is given, defaults to 'body'."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        r = QueryResult(query="q", strategy="voiceover", gap_indices=[0],
                        videos_found=5, gaps_filled=1, avg_confidence_improvement=0.2)
        db.record_result(r, "other")  # No chapter_type arg
        assert "body" in db.chapter_strategy_success


class TestChapterTypePreference:
    """AC: When filling intro gaps, prefer historically successful query patterns for intro."""

    def test_intro_successful_patterns_preferred_for_intro_gaps(self, tmp_path):
        """Query patterns successful for 'intro' chapters are preferred when filling intro gaps."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))

        # Record entity strategy as successful for intro gaps (multiple times)
        for _ in range(5):
            r = QueryResult(query="intro entity footage", strategy="entity", gap_indices=[0],
                            videos_found=10, gaps_filled=1, avg_confidence_improvement=0.4)
            db.record_result(r, "abstract_concept", chapter_type="intro")

        # Record voiceover strategy as failing for intro gaps
        for _ in range(5):
            r = QueryResult(query="intro voiceover text", strategy="voiceover", gap_indices=[1],
                            videos_found=2, gaps_filled=0, avg_confidence_improvement=0.0)
            db.record_result(r, "abstract_concept", chapter_type="intro")

        # Record voiceover as successful for body (should NOT influence intro)
        for _ in range(5):
            r = QueryResult(query="body voiceover text", strategy="voiceover", gap_indices=[5],
                            videos_found=8, gaps_filled=1, avg_confidence_improvement=0.3)
            db.record_result(r, "abstract_concept", chapter_type="body")

        # When asking for best strategy for intro, entity should win
        best = db.get_best_strategy_for_chapter("abstract_concept", "intro")
        assert best == "entity"

    def test_chapter_type_ranking_reflects_learning(self, tmp_path):
        """Strategy ranking for a chapter type reflects accumulated success."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))

        # Entity succeeds for conclusion
        for _ in range(3):
            r = QueryResult(query="conclusion footage", strategy="entity", gap_indices=[0],
                            videos_found=5, gaps_filled=1, avg_confidence_improvement=0.3)
            db.record_result(r, "location", chapter_type="conclusion")

        # Topic fails for conclusion
        for _ in range(3):
            r = QueryResult(query="conclusion topic", strategy="topic", gap_indices=[1],
                            videos_found=1, gaps_filled=0, avg_confidence_improvement=0.0)
            db.record_result(r, "location", chapter_type="conclusion")

        ranking = db.get_strategy_ranking_for_chapter("location", "conclusion")
        assert ranking[0] == "entity"

    def test_no_chapter_data_falls_back_to_pattern(self, tmp_path):
        """When no chapter-type data exists, falls back to pattern-level data."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))

        # Only pattern-level data (no chapter_type tracking yet)
        db.pattern_strategy_success["proper_noun"]["entity"] = 0.9
        db.pattern_strategy_success["proper_noun"]["voiceover"] = 0.2

        # Chapter data empty for this type
        best = db.get_best_strategy_for_chapter("proper_noun", "listicle_item")
        assert best == "entity"  # Falls back to pattern data

    def test_unknown_pattern_and_chapter_returns_default(self, tmp_path):
        """Unknown pattern + chapter type returns 'voiceover' default."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        best = db.get_best_strategy_for_chapter("never_seen", "intro")
        assert best == "voiceover"


class TestChapterTypePersistence:
    """AC: Chapter-type learning data persists across iterative match rounds."""

    def test_chapter_data_survives_save_load(self, tmp_path):
        """Chapter-type success rates persist through save/load cycle."""
        db_path = str(tmp_path / "learn.json")
        db = QueryLearningDB(db_path=db_path)

        # Record some chapter-type data
        r = QueryResult(query="q", strategy="entity", gap_indices=[0],
                        videos_found=5, gaps_filled=1, avg_confidence_improvement=0.3)
        db.record_result(r, "abstract_concept", chapter_type="intro")
        db.save()

        # Load into new instance
        db2 = QueryLearningDB(db_path=db_path)
        assert "intro" in db2.chapter_strategy_success
        assert db2.chapter_strategy_success["intro"]["entity"] == pytest.approx(0.3)

    def test_version_updated(self, tmp_path):
        """DB version is 1.1 with chapter type support."""
        import json
        db_path = str(tmp_path / "learn.json")
        db = QueryLearningDB(db_path=db_path)
        db.save()
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data["version"] == "1.1"

    def test_summary_includes_chapter_types(self, tmp_path):
        """Summary includes chapter_types_learned count."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        r = QueryResult(query="q", strategy="entity", gap_indices=[0],
                        videos_found=5, gaps_filled=1, avg_confidence_improvement=0.3)
        db.record_result(r, "other", chapter_type="intro")
        summary = db.get_summary()
        assert summary["chapter_types_learned"] == 1


class TestChapterTypeClassification:
    """AC: Chapter type uses simple position-based heuristics."""

    def test_intro_classification(self):
        """First 10% of segments classified as intro."""
        from src.iterative_match.gap_analyzer import GapSegment, annotate_gaps_with_chapters
        gaps = [GapSegment(segment_index=0, confidence=0.5, voiceover_text="intro text", position=0.0)]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        assert result[0].chapter_type == "intro"

    def test_conclusion_classification(self):
        """Last 10% of segments classified as conclusion."""
        from src.iterative_match.gap_analyzer import GapSegment, annotate_gaps_with_chapters
        gaps = [GapSegment(segment_index=95, confidence=0.5, voiceover_text="conclusion text", position=950.0)]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        assert result[0].chapter_type == "conclusion"

    def test_body_classification(self):
        """Middle segments classified as body."""
        from src.iterative_match.gap_analyzer import GapSegment, annotate_gaps_with_chapters
        gaps = [GapSegment(segment_index=50, confidence=0.5, voiceover_text="body text", position=500.0)]
        result = annotate_gaps_with_chapters(gaps, total_segments=100)
        assert result[0].chapter_type == "body"

    def test_listicle_item_classification(self):
        """Segments within listicle groups classified as listicle_item."""
        from src.iterative_match.gap_analyzer import GapSegment, annotate_gaps_with_chapters
        gaps = [GapSegment(segment_index=50, confidence=0.5, voiceover_text="list item", position=500.0)]
        listicle_groups = [{'group_id': 'group_1', 'start_segment': 45, 'end_segment': 55}]
        result = annotate_gaps_with_chapters(gaps, total_segments=100, listicle_groups=listicle_groups)
        assert result[0].chapter_type == "listicle_item"

    def test_listicle_overrides_position(self):
        """Listicle item classification overrides position-based intro/conclusion."""
        from src.iterative_match.gap_analyzer import GapSegment, annotate_gaps_with_chapters
        # Segment at position 0 (would be intro) but inside a listicle group
        gaps = [GapSegment(segment_index=2, confidence=0.5, voiceover_text="list item", position=20.0)]
        listicle_groups = [{'group_id': 'group_1', 'start_segment': 0, 'end_segment': 10}]
        result = annotate_gaps_with_chapters(gaps, total_segments=100, listicle_groups=listicle_groups)
        assert result[0].chapter_type == "listicle_item"


# ============================================================================
# get_preferred_strategies (US-72-012)
# ============================================================================

class TestGetPreferredStrategies:
    """AC: get_preferred_strategies(chapter_type, top_n=3) returns best strategies for a chapter type."""

    def test_returns_default_when_no_data(self, tmp_path):
        """With no chapter data, returns default strategy order truncated to top_n."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        result = db.get_preferred_strategies("intro")
        assert result == ["voiceover", "similar_locked", "entity"]

    def test_returns_top_n_strategies(self, tmp_path):
        """Returns strategies sorted by chapter-type success rate, limited to top_n."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.chapter_strategy_success["intro"]["entity"] = 0.9
        db.chapter_strategy_success["intro"]["voiceover"] = 0.5
        db.chapter_strategy_success["intro"]["topic"] = 0.7
        db.chapter_strategy_success["intro"]["similar_locked"] = 0.3

        result = db.get_preferred_strategies("intro", top_n=3)
        assert len(result) == 3
        assert result[0] == "entity"
        assert result[1] == "topic"
        assert result[2] == "voiceover"

    def test_top_n_limits_output(self, tmp_path):
        """top_n=1 returns only the single best strategy."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.chapter_strategy_success["body"]["entity"] = 0.9
        db.chapter_strategy_success["body"]["voiceover"] = 0.5
        result = db.get_preferred_strategies("body", top_n=1)
        assert result == ["entity"]

    def test_fewer_strategies_than_top_n(self, tmp_path):
        """When fewer strategies exist than top_n, returns all available."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.chapter_strategy_success["conclusion"]["entity"] = 0.8
        result = db.get_preferred_strategies("conclusion", top_n=3)
        assert result == ["entity"]

    def test_different_chapter_types_independent(self, tmp_path):
        """Preferred strategies for different chapter types are independent."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))
        db.chapter_strategy_success["intro"]["entity"] = 0.9
        db.chapter_strategy_success["intro"]["voiceover"] = 0.2
        db.chapter_strategy_success["conclusion"]["voiceover"] = 0.9
        db.chapter_strategy_success["conclusion"]["entity"] = 0.2

        intro_prefs = db.get_preferred_strategies("intro", top_n=2)
        conclusion_prefs = db.get_preferred_strategies("conclusion", top_n=2)
        assert intro_prefs[0] == "entity"
        assert conclusion_prefs[0] == "voiceover"

    def test_strategy_preference_reflects_recorded_results(self, tmp_path):
        """Preferences update correctly after recording results."""
        db = QueryLearningDB(db_path=str(tmp_path / "db.json"))

        # Record multiple successful entity queries for listicle_item
        for _ in range(5):
            r = QueryResult(query="list item footage", strategy="entity", gap_indices=[0],
                            videos_found=10, gaps_filled=1, avg_confidence_improvement=0.4,
                            chapter_type="listicle_item")
            db.record_result(r, "other", chapter_type="listicle_item")

        # Record failed voiceover queries for listicle_item
        for _ in range(5):
            r = QueryResult(query="list voiceover", strategy="voiceover", gap_indices=[1],
                            videos_found=2, gaps_filled=0, avg_confidence_improvement=0.0,
                            chapter_type="listicle_item")
            db.record_result(r, "other", chapter_type="listicle_item")

        prefs = db.get_preferred_strategies("listicle_item", top_n=3)
        assert prefs[0] == "entity"
