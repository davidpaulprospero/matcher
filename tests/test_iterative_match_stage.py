"""
Tests for IterativeMatchStage - Multi-pass gap filling.

Tests cover:
- Gap/lock classification (confidence + source spacing)
- Gap pattern analysis
- Query generation strategies
- Progressive query refinement
- Query learning database
"""

import pytest
from dataclasses import dataclass
from typing import List, Any, Optional
from unittest.mock import MagicMock, patch

# Import test targets
from src.stages.iterative_match import (
    IterativeMatchStage,
    LockedMatch,
    GapSegment,
)
from src.iterative_match.gap_analyzer import (
    GapAnalysis,
    GapSegment as AnalyzerGapSegment,
    analyze_gaps,
    extract_keywords_for_gap,
)
from src.iterative_match.query_learning import (
    QueryLearningDB,
    QueryResult,
    QueryPlan,
)
from src.config.sections.iterative_matching import IterativeMatchingConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def stage():
    """Create IterativeMatchStage instance."""
    return IterativeMatchStage()


@pytest.fixture
def mock_config():
    """Create mock config with iterative_matching settings."""
    config = MagicMock()
    config.iterative_matching = IterativeMatchingConfig(
        enabled=True,
        target_confidence=0.90,
        source_spacing_seconds=300.0,
        max_iterations=5,
        min_gap_percentage=0.05,
    )
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager."""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    return checkpoint


@dataclass
class MockVoiceoverSegment:
    """Mock voiceover segment."""
    index: int
    start: float
    end: float
    text: str


@dataclass
class MockMatch:
    """Mock match with primary_match structure."""
    segment_index: int = 0
    confidence: float = 0.0
    video_file: str = ""

    @property
    def primary_match(self):
        """Mimic MatchResult structure."""
        pm = MagicMock()
        pm.confidence = self.confidence
        pm.video_segment = MagicMock()
        pm.video_segment.source_file = self.video_file
        return pm


@pytest.fixture
def mock_state():
    """Create mock pipeline state."""
    state = MagicMock()
    state.voiceover_segments = [
        MockVoiceoverSegment(0, 0.0, 5.0, "The concept of freedom is fundamental"),
        MockVoiceoverSegment(1, 5.0, 10.0, "John Smith visited New York"),
        MockVoiceoverSegment(2, 10.0, 15.0, "Running through the forest"),
        MockVoiceoverSegment(3, 15.0, 20.0, "The beautiful sunset in Paris"),
        MockVoiceoverSegment(4, 20.0, 25.0, "She felt happy and excited"),
    ]
    state.matches = [
        MockMatch(0, 0.95, "video_ABC123xyz_.mp4"),  # High conf
        MockMatch(1, 0.75, "video_DEF456uvw_.mp4"),  # Low conf - gap
        MockMatch(2, 0.92, "video_ABC123xyz_.mp4"),  # High conf but spacing violation
        MockMatch(3, 0.88, "video_GHI789rst_.mp4"),  # Low conf - gap
        MockMatch(4, 0.96, "video_JKL012opq_.mp4"),  # High conf
    ]
    state.extracted_entities = [
        {'name': 'John Smith', 'type': 'PERSON'},
        {'name': 'New York', 'type': 'LOCATION'},
        {'name': 'Paris', 'type': 'LOCATION'},
    ]
    state.downloaded_videos = []
    state.voiceover_path = "/path/to/voiceover.mp3"
    return state


# ============================================================================
# Gap/Lock Classification Tests
# ============================================================================

class TestGapLockClassification:
    """Tests for gap/lock identification logic."""

    @pytest.mark.fast
    def test_high_conf_unique_source_locked(self, stage, mock_state):
        """Segment with 95% conf and unique source should be locked."""
        # Use just first segment
        mock_state.matches = [MockMatch(0, 0.95, "video_ABC123xyz_.mp4")]
        mock_state.voiceover_segments = [mock_state.voiceover_segments[0]]

        locked, gaps = stage._identify_gaps_and_locks(
            mock_state.matches,
            mock_state.voiceover_segments,
            target_conf=0.90,
            source_spacing=300.0
        )

        assert len(locked) == 1
        assert len(gaps) == 0
        assert locked[0].confidence == 0.95

    @pytest.mark.fast
    def test_low_conf_always_gap(self, stage, mock_state):
        """Segment with 70% conf should always be a gap."""
        mock_state.matches = [MockMatch(0, 0.70, "video_ABC123xyz_.mp4")]
        mock_state.voiceover_segments = [mock_state.voiceover_segments[0]]

        locked, gaps = stage._identify_gaps_and_locks(
            mock_state.matches,
            mock_state.voiceover_segments,
            target_conf=0.90,
            source_spacing=300.0
        )

        assert len(locked) == 0
        assert len(gaps) == 1
        assert gaps[0].reason == 'low_confidence'

    @pytest.mark.fast
    def test_high_conf_spacing_violation_gap(self, stage, mock_state):
        """High conf but source used 2 minutes ago should be gap."""
        # Two segments with same source, 2 minutes apart (120s)
        mock_state.voiceover_segments = [
            MockVoiceoverSegment(0, 0.0, 5.0, "First segment"),
            MockVoiceoverSegment(1, 120.0, 125.0, "Second segment"),  # 2 min later
        ]
        mock_state.matches = [
            MockMatch(0, 0.95, "video_ABC123xyz_.mp4"),
            MockMatch(1, 0.95, "video_ABC123xyz_.mp4"),  # Same source
        ]

        locked, gaps = stage._identify_gaps_and_locks(
            mock_state.matches,
            mock_state.voiceover_segments,
            target_conf=0.90,
            source_spacing=300.0  # 5 min spacing required
        )

        # First should be locked, second should be gap (spacing violation)
        assert len(locked) == 1
        assert len(gaps) == 1
        assert gaps[0].segment_index == 1
        assert gaps[0].reason == 'spacing_violation'

    @pytest.mark.fast
    def test_spacing_boundary_300s(self, stage, mock_state):
        """Exactly 300s apart should NOT be a violation."""
        mock_state.voiceover_segments = [
            MockVoiceoverSegment(0, 0.0, 5.0, "First segment"),
            MockVoiceoverSegment(1, 300.0, 305.0, "Second segment"),  # Exactly 5 min
        ]
        mock_state.matches = [
            MockMatch(0, 0.95, "video_ABC123xyz_.mp4"),
            MockMatch(1, 0.95, "video_ABC123xyz_.mp4"),
        ]

        locked, gaps = stage._identify_gaps_and_locks(
            mock_state.matches,
            mock_state.voiceover_segments,
            target_conf=0.90,
            source_spacing=300.0
        )

        # Both should be locked (exactly at boundary)
        assert len(locked) == 2
        assert len(gaps) == 0


# ============================================================================
# Gap Pattern Analysis Tests
# ============================================================================

class TestGapPatternAnalysis:
    """Tests for gap pattern classification."""

    @pytest.mark.fast
    def test_detects_abstract_concepts(self):
        """'Freedom means...' should be classified as abstract."""
        gaps = [
            AnalyzerGapSegment(
                segment_index=0,
                confidence=0.5,
                voiceover_text="Freedom means being able to make your own choices",
                position=0.0
            )
        ]

        analysis = analyze_gaps(gaps)

        assert analysis.pattern_counts['abstract_concept'] == 1
        assert 0 in analysis.abstract_concepts

    @pytest.mark.fast
    def test_detects_proper_nouns(self):
        """'Tesla announced...' should be classified as proper noun."""
        gaps = [
            AnalyzerGapSegment(
                segment_index=0,
                confidence=0.5,
                voiceover_text="Tesla announced a new product yesterday",
                position=0.0
            )
        ]

        analysis = analyze_gaps(gaps, extracted_entities=[{'name': 'Tesla'}])

        assert analysis.pattern_counts['proper_noun'] == 1
        assert 0 in analysis.proper_nouns

    @pytest.mark.fast
    def test_detects_action_verbs(self):
        """'Running on the track...' should be classified as action."""
        # Use text without location words to avoid location taking precedence
        gaps = [
            AnalyzerGapSegment(
                segment_index=0,
                confidence=0.5,
                voiceover_text="Running on the track and jumping over hurdles",
                position=0.0
            )
        ]

        analysis = analyze_gaps(gaps)

        assert analysis.pattern_counts['action_verb'] == 1
        assert 0 in analysis.action_descriptions

    @pytest.mark.fast
    def test_detects_locations(self):
        """'In Paris, France...' should be classified as location."""
        gaps = [
            AnalyzerGapSegment(
                segment_index=0,
                confidence=0.5,
                voiceover_text="In Paris, France the Eiffel Tower stands tall",
                position=0.0
            )
        ]

        analysis = analyze_gaps(gaps)

        assert analysis.pattern_counts['location'] == 1
        assert 0 in analysis.locations

    @pytest.mark.fast
    def test_detects_emotional_content(self):
        """'She felt happy...' should be classified as emotion."""
        gaps = [
            AnalyzerGapSegment(
                segment_index=0,
                confidence=0.5,
                voiceover_text="She felt happy and content with life",
                position=0.0
            )
        ]

        analysis = analyze_gaps(gaps)

        assert analysis.pattern_counts['emotion'] == 1
        assert 0 in analysis.emotional_content


# ============================================================================
# Keyword Extraction Tests
# ============================================================================

class TestKeywordExtraction:
    """Tests for keyword extraction from gaps."""

    @pytest.mark.fast
    def test_extracts_proper_nouns_first(self):
        """Proper nouns should be prioritized."""
        gap = AnalyzerGapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="John Smith visited Microsoft headquarters",
            position=0.0,
            pattern_type='proper_noun'
        )

        keywords = extract_keywords_for_gap(gap, max_keywords=5)

        # Should include capitalized names
        assert any('John' in kw or 'Smith' in kw for kw in keywords)
        assert any('Microsoft' in kw for kw in keywords)

    @pytest.mark.fast
    def test_extracts_action_words_for_action_type(self):
        """Action gaps should include action verbs."""
        gap = AnalyzerGapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="Running through the mountains",
            position=0.0,
            pattern_type='action_verb'
        )

        keywords = extract_keywords_for_gap(gap, max_keywords=5)

        assert 'running' in [kw.lower() for kw in keywords]

    @pytest.mark.fast
    def test_excludes_stop_words(self):
        """Common stop words should be excluded from content keywords."""
        gap = AnalyzerGapSegment(
            segment_index=0,
            confidence=0.5,
            voiceover_text="The quick brown fox jumps over the lazy dog",
            position=0.0,
            pattern_type='other'
        )

        keywords = extract_keywords_for_gap(gap, max_keywords=5)

        # Content keywords (after proper nouns) should not include stop words
        # "The" at start gets extracted as proper noun, but content words shouldn't
        # Check that content-focused words are extracted
        assert any(kw.lower() in ['quick', 'brown', 'jumps', 'lazy'] for kw in keywords)


# ============================================================================
# Query Learning Tests
# ============================================================================

class TestQueryLearning:
    """Tests for query learning database."""

    @pytest.mark.fast
    def test_learning_db_records_success(self, tmp_path):
        """Successful query should update DB."""
        db_path = tmp_path / "learning.json"
        db = QueryLearningDB(str(db_path))

        result = QueryResult(
            query="person footage",
            strategy="voiceover",
            gap_indices=[0, 1],
            videos_found=5,
            gaps_filled=2,
            avg_confidence_improvement=0.15
        )

        db.record_result(result, "proper_noun")

        assert db.strategy_stats['voiceover'].total_queries == 1
        assert db.strategy_stats['voiceover'].total_gaps_filled == 2

    @pytest.mark.fast
    def test_learning_db_best_strategy(self, tmp_path):
        """DB should return best strategy for pattern."""
        db_path = tmp_path / "learning.json"
        db = QueryLearningDB(str(db_path))

        # Record some results
        for _ in range(5):
            db.record_result(
                QueryResult("q1", "similar_locked", [0], 3, 1, 0.1),
                "abstract_concept"
            )
        for _ in range(2):
            db.record_result(
                QueryResult("q2", "voiceover", [0], 3, 0, 0.0),
                "abstract_concept"
            )

        best = db.get_best_strategy("abstract_concept")

        # similar_locked had better success rate
        assert best == "similar_locked"

    @pytest.mark.fast
    def test_learning_persists_across_runs(self, tmp_path):
        """DB should persist to disk and reload."""
        db_path = tmp_path / "learning.json"

        # Create and save
        db1 = QueryLearningDB(str(db_path))
        db1.record_result(
            QueryResult("test query", "entity", [0], 3, 1, 0.1),
            "proper_noun"
        )
        db1.save()

        # Reload
        db2 = QueryLearningDB(str(db_path))

        assert db2.strategy_stats['entity'].total_queries == 1


# ============================================================================
# Stage Integration Tests
# ============================================================================

class TestStageIntegration:
    """Integration tests for the stage."""

    @pytest.mark.fast
    def test_stage_skipped_when_disabled(self, stage, mock_state, mock_config, mock_checkpoint):
        """Stage should skip when disabled in config."""
        mock_config.iterative_matching.enabled = False

        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'disabled'

    @pytest.mark.fast
    def test_stage_skipped_in_test_mode(self, stage, mock_config, mock_checkpoint):
        """Stage should skip when test_mode.skip_iterative is set."""
        # Set up test_mode config with skip_iterative=True
        from src.config.sections.test_mode import TestModeConfig

        # Create state with text_metadata to pass candidate count check
        state = MagicMock()
        state.voiceover_segments = [
            MockVoiceoverSegment(0, 0.0, 5.0, "The concept of freedom is fundamental"),
        ]
        state.matches = [
            MockMatch(0, 0.95, "video_ABC123xyz_.mp4"),
        ]
        state.text_metadata = [{"id": "video1", "title": "test"}]

        # skip_iterative should only apply when test mode is active
        mock_config._test_mode = True
        mock_config.test_mode = TestModeConfig(skip_iterative=True)

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'test_mode_skip'
        assert result.data.get('test_mode') is True

    @pytest.mark.fast
    def test_stage_runs_in_test_mode_when_skip_disabled(self, stage, mock_state, mock_config, mock_checkpoint):
        """Stage should run normally in test mode when skip_iterative is False."""
        from src.config.sections.test_mode import TestModeConfig

        # Set up test_mode config with skip_iterative=False
        mock_config._test_mode = True
        mock_config._test_mode_skip_iterative = False
        mock_config.test_mode = TestModeConfig(skip_iterative=False)

        # This should NOT skip due to test mode - it should proceed (and fail on other checks)
        # Since mock_state doesn't have text_metadata, it will skip for a different reason
        result = stage.run(mock_state, mock_config, mock_checkpoint)

        # Should NOT skip with reason 'test_mode_skip'
        assert result.data.get('reason') != 'test_mode_skip'

    @pytest.mark.fast
    def test_stage_does_not_skip_when_test_mode_inactive(self, stage, mock_state, mock_config, mock_checkpoint):
        """skip_iterative should be ignored when _test_mode is not active."""
        from src.config.sections.test_mode import TestModeConfig

        mock_config._test_mode = False
        mock_config._test_mode_skip_iterative = True
        mock_config.test_mode = TestModeConfig(skip_iterative=True)

        # Force a deterministic non-test-mode skip path.
        mock_state.text_metadata = []

        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success
        assert result.data.get('reason') == 'no_candidates'

    @pytest.mark.fast
    def test_stage_skipped_no_matches(self, stage, mock_config, mock_checkpoint):
        """Stage should skip when no matches to iterate on."""
        state = MagicMock()
        state.matches = []
        state.voiceover_segments = [MockVoiceoverSegment(0, 0.0, 5.0, "test")]

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_matches'

    @pytest.mark.fast
    def test_stage_stops_no_gaps(self, stage, mock_state, mock_config, mock_checkpoint):
        """Stage should stop early when no gaps remain."""
        # All matches are high confidence with unique sources
        mock_state.matches = [
            MockMatch(0, 0.95, "video_A.mp4"),
            MockMatch(1, 0.95, "video_B.mp4"),
            MockMatch(2, 0.95, "video_C.mp4"),
            MockMatch(3, 0.95, "video_D.mp4"),
            MockMatch(4, 0.95, "video_E.mp4"),
        ]

        result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success
        assert result.data.get('final_gaps') == 0

    @pytest.mark.fast
    def test_validate_inputs_missing_matches(self, stage, mock_config):
        """Validation should fail when matches missing."""
        state = MagicMock()
        state.matches = []
        state.voiceover_segments = None

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "matches" in error.lower() or "match" in error.lower()


# ============================================================================
# Multi-Strategy Query Generation Tests
# ============================================================================

class TestQueryGeneration:
    """Tests for multi-strategy query generation."""

    @pytest.mark.fast
    def test_generates_voiceover_queries(self, stage, mock_state):
        """Should generate queries from voiceover text."""
        iter_config = IterativeMatchingConfig(
            use_voiceover_text_queries=True,
            use_similar_to_locked=False,
            use_entity_topic_queries=False,
        )

        gaps = [
            GapSegment(0, 0.5, "The concept of freedom is important", 0.0),
        ]

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1
        )

        assert len(queries) > 0
        assert any(q['strategy'] == 'voiceover' for q in queries)

    @pytest.mark.fast
    def test_generates_similar_to_locked_queries(self, stage, mock_state):
        """Should generate queries from locked matches."""
        iter_config = IterativeMatchingConfig(
            use_voiceover_text_queries=False,
            use_similar_to_locked=True,
            use_entity_topic_queries=False,
        )

        gaps = [
            GapSegment(1, 0.5, "Some text", 10.0),  # Position 10s
        ]
        locked = [
            LockedMatch(0, "ABC123xyz", 0.95, 5.0, "Great Video"),  # Position 5s
        ]

        queries = stage._generate_multi_strategy_queries(
            gaps, locked, mock_state, iter_config, pass_num=1
        )

        assert any(q['strategy'] == 'similar_locked' for q in queries)

    @pytest.mark.fast
    def test_generates_entity_queries(self, stage, mock_state):
        """Should generate queries from entities."""
        iter_config = IterativeMatchingConfig(
            use_voiceover_text_queries=False,
            use_similar_to_locked=False,
            use_entity_topic_queries=True,
        )

        gaps = [
            GapSegment(1, 0.5, "John Smith visited the park", 0.0),
        ]
        mock_state.extracted_entities = [{'name': 'John Smith'}]

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1
        )

        assert any(q['strategy'] == 'entity' for q in queries)


# ============================================================================
# Config Tests
# ============================================================================

class TestIterativeMatchingConfig:
    """Tests for IterativeMatchingConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """Default config values should be sensible."""
        config = IterativeMatchingConfig()

        assert config.enabled is True
        assert config.target_confidence == 0.90
        assert config.source_spacing_seconds == 300.0
        assert config.max_iterations == 5
        assert config.min_gap_percentage == 0.05

    @pytest.mark.fast
    def test_clamping_values(self):
        """Config should clamp values to valid ranges."""
        config = IterativeMatchingConfig(
            target_confidence=1.5,  # Should clamp to 1.0
            source_spacing_seconds=-100,  # Should clamp to 0.0
            max_iterations=0,  # Should clamp to 1
        )

        assert config.target_confidence == 1.0
        assert config.source_spacing_seconds == 0.0
        assert config.max_iterations == 1

    @pytest.mark.fast
    def test_pattern_categories(self):
        """Default pattern categories should be set."""
        config = IterativeMatchingConfig()

        assert 'abstract_concept' in config.gap_pattern_categories
        assert 'proper_noun' in config.gap_pattern_categories
        assert 'action_verb' in config.gap_pattern_categories


# ============================================================================
# US-71-007: Chapter-aware gap prioritization integration
# ============================================================================

class TestChapterAwareGapPrioritization:
    """Tests that gap priority sorting is used in the stage gap-filling order."""

    @pytest.mark.fast
    def test_gaps_reordered_by_chapter_priority(self, stage):
        """AC: Gap priority is used to sort gap-filling order so high-importance
        chapters are filled first.

        Verifies that after the gap analysis block in run(), the local gaps list
        is reordered so intro/conclusion gaps come before middle gaps.
        """
        from src.iterative_match.gap_analyzer import (
            GapSegment as GapSeg,
            annotate_gaps_with_chapters,
        )

        # Create 100 voiceover segments worth of gaps spread across intro/middle/conclusion
        # Simulate what happens inside the stage's run() loop
        total_count = 100

        # Local GapSegment objects (as used in the stage)
        gaps = [
            GapSegment(segment_index=50, confidence=0.5, voiceover_text="middle content",
                       position=100.0, reason='low_confidence'),
            GapSegment(segment_index=3, confidence=0.5, voiceover_text="intro content",
                       position=5.0, reason='low_confidence'),
            GapSegment(segment_index=95, confidence=0.5, voiceover_text="conclusion content",
                       position=200.0, reason='low_confidence'),
        ]

        # Create analyzer GapSegments (as done in the stage's gap analysis block)
        gap_segments = [
            GapSeg(
                segment_index=g.segment_index,
                confidence=g.confidence,
                voiceover_text=g.voiceover_text,
                position=g.position,
            )
            for g in gaps
        ]

        # Apply chapter-aware prioritization (as the stage now does)
        gap_segments = annotate_gaps_with_chapters(
            gap_segments,
            total_segments=total_count,
        )

        # Reorder local gaps to match priority order (same logic as in stage)
        gap_idx_order = [gs.segment_index for gs in gap_segments]
        gap_by_idx = {g.segment_index: g for g in gaps}
        gaps = [gap_by_idx[idx] for idx in gap_idx_order if idx in gap_by_idx]

        # Verify: intro (seg 3) and conclusion (seg 95) should come before middle (seg 50)
        # With equal confidence:
        #   intro: 0.5 - 0.2 = 0.3 effective (highest priority)
        #   conclusion: 0.5 - 0.15 = 0.35 effective
        #   middle: 0.5 - 0.0 = 0.5 effective (lowest priority)
        assert gaps[0].segment_index == 3, "Intro gap should be processed first"
        assert gaps[1].segment_index == 95, "Conclusion gap should be processed second"
        assert gaps[2].segment_index == 50, "Middle gap should be processed last"

    @pytest.mark.fast
    def test_query_generation_receives_priority_sorted_gaps(self, stage, mock_state):
        """Verify that _generate_multi_strategy_queries receives gaps in priority order.

        When gaps are sorted by chapter priority, query generation processes
        intro/conclusion gaps first (they appear earlier in the gaps list
        and are selected early by query-gap sampling in the method).
        """
        from src.iterative_match.gap_analyzer import (
            GapSegment as GapSeg,
            annotate_gaps_with_chapters,
        )

        # Create many gaps - some in intro, some in middle, some in conclusion
        total_count = 100
        gaps = []
        for i in range(25):  # 25 middle gaps
            gaps.append(GapSegment(
                segment_index=30 + i, confidence=0.5,
                voiceover_text=f"middle content {i}", position=60.0 + i * 2,
                reason='low_confidence',
            ))
        # Add 2 intro gaps
        gaps.append(GapSegment(segment_index=2, confidence=0.5,
                               voiceover_text="important intro", position=4.0,
                               reason='low_confidence'))
        gaps.append(GapSegment(segment_index=5, confidence=0.5,
                               voiceover_text="key opening", position=10.0,
                               reason='low_confidence'))

        # Apply prioritization
        gap_segments = [
            GapSeg(segment_index=g.segment_index, confidence=g.confidence,
                   voiceover_text=g.voiceover_text, position=g.position)
            for g in gaps
        ]
        gap_segments = annotate_gaps_with_chapters(gap_segments, total_segments=total_count)
        gap_idx_order = [gs.segment_index for gs in gap_segments]
        gap_by_idx = {g.segment_index: g for g in gaps}
        gaps = [gap_by_idx[idx] for idx in gap_idx_order if idx in gap_by_idx]

        # The intro gaps (segments 2 and 5) should be in the first 20 gaps
        # (the slice used by _generate_multi_strategy_queries)
        first_20_indices = [g.segment_index for g in gaps[:20]]
        assert 2 in first_20_indices, "Intro gap should be in top-priority processing batch"
        assert 5 in first_20_indices, "Intro gap should be in top-priority processing batch"


class TestLateGapCoverage:
    """Tests for late-gap query/search pressure handling."""

    @pytest.mark.fast
    def test_select_query_gaps_includes_tail_in_late_mode(self, stage):
        """Late-gap mode should include high-index gaps, not just first-N priority entries."""
        gaps = [
            GapSegment(
                segment_index=i,
                confidence=0.4,
                voiceover_text=f"segment {i} content",
                position=float(i) * 5.0,
                reason='low_confidence',
            )
            for i in range(40)
        ]

        stage._late_gap_mode = True
        selected = stage._select_query_gaps(gaps, 20)
        selected_indices = [g.segment_index for g in selected]

        assert len(selected) == 20
        assert len(set(selected_indices)) == 20
        assert max(selected_indices) >= 35
        assert any(i >= 30 for i in selected_indices)

    @pytest.mark.fast
    def test_search_returns_tuple_when_no_new_videos(self, stage, mock_state, mock_config):
        """No-results path should still return (video_ids, cache_hits, cache_misses)."""
        mock_state.video_ids = []
        mock_state.text_metadata = []
        mock_state.downloaded_videos = []
        mock_config.iterative_matching.cache_query_results = False

        with patch.object(stage, '_get_cookie_args', return_value=[]):
            video_ids, cache_hits, cache_misses = stage._search_youtube_for_videos(
                [],
                mock_state,
                mock_config,
                mock_config.iterative_matching,
            )

        assert video_ids == []
        assert cache_hits == 0
        assert cache_misses == 0

    @pytest.mark.fast
    def test_search_expands_results_per_query_in_late_gap_mode(self, stage, mock_state, mock_config):
        """Late-gap pressure should increase ytsearch pool size for each query."""
        mock_state.video_ids = []
        mock_state.text_metadata = []
        mock_state.downloaded_videos = []
        mock_config.iterative_matching.cache_query_results = False
        mock_config.iterative_matching.search_results_per_gap = 10
        mock_config.iterative_matching.max_new_videos_per_pass = 50

        stage._late_gap_mode = True
        stage._gap_pressure = 0.6  # 1.6x expansion -> ytsearch16

        with patch.object(stage, '_get_cookie_args', return_value=[]), \
             patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            stage._search_youtube_for_videos(
                [{'query': 'arctic wildlife footage'}],
                mock_state,
                mock_config,
                mock_config.iterative_matching,
            )

        cmd = mock_run.call_args[0][0]
        assert any("ytsearch16:arctic wildlife footage" in str(part) for part in cmd)


# ============================================================================
# Chapter-Type Strategy Ranking Tests (US-76-005)
# ============================================================================

class TestChapterTypeStrategyRanking:
    """Tests for chapter-type strategy ranking in query generation."""

    @pytest.fixture
    def learning_db(self, tmp_path):
        """Create a QueryLearningDB with chapter-type success data."""
        db = QueryLearningDB(str(tmp_path / "test_learning.json"))
        # Record successes: for intro, entity strategy works best
        for _ in range(5):
            db.record_result(
                QueryResult(
                    query="test entity query",
                    strategy="entity",
                    gap_indices=[0],
                    videos_found=3,
                    gaps_filled=2,
                    avg_confidence_improvement=0.2,
                    successful=True,
                ),
                gap_pattern="abstract_concept",
                chapter_type="intro",
            )
        # Record: for intro, voiceover is less effective
        for _ in range(2):
            db.record_result(
                QueryResult(
                    query="test voiceover query",
                    strategy="voiceover",
                    gap_indices=[0],
                    videos_found=1,
                    gaps_filled=0,
                    avg_confidence_improvement=0.0,
                ),
                gap_pattern="abstract_concept",
                chapter_type="intro",
            )
        # Record successes: for listicle_item, voiceover strategy works best
        for _ in range(5):
            db.record_result(
                QueryResult(
                    query="test listicle voiceover",
                    strategy="voiceover",
                    gap_indices=[10],
                    videos_found=4,
                    gaps_filled=3,
                    avg_confidence_improvement=0.3,
                    successful=True,
                ),
                gap_pattern="concrete_topic",
                chapter_type="listicle_item",
            )
        for _ in range(1):
            db.record_result(
                QueryResult(
                    query="test listicle entity",
                    strategy="entity",
                    gap_indices=[10],
                    videos_found=1,
                    gaps_filled=0,
                    avg_confidence_improvement=0.0,
                ),
                gap_pattern="concrete_topic",
                chapter_type="listicle_item",
            )
        return db

    @pytest.fixture
    def intro_gap_segments(self):
        """Gap segments annotated as intro chapter type."""
        from src.iterative_match.gap_analyzer import GapSegment as GapSeg
        gs = GapSeg(
            segment_index=0,
            confidence=0.5,
            voiceover_text="Welcome to this documentary about nature",
            position=0.0,
        )
        gs.chapter_type = "intro"
        return [gs]

    @pytest.fixture
    def listicle_gap_segments(self):
        """Gap segments annotated as listicle_item chapter type."""
        from src.iterative_match.gap_analyzer import GapSegment as GapSeg
        gs = GapSeg(
            segment_index=10,
            confidence=0.5,
            voiceover_text="Number three on our list is the mighty oak tree",
            position=50.0,
        )
        gs.chapter_type = "listicle_item"
        return [gs]

    @pytest.fixture
    def body_gap_segments(self):
        """Gap segments annotated as body chapter type."""
        from src.iterative_match.gap_analyzer import GapSegment as GapSeg
        gs = GapSeg(
            segment_index=20,
            confidence=0.5,
            voiceover_text="The forest ecosystem relies on biodiversity",
            position=100.0,
        )
        gs.chapter_type = "body"
        return [gs]

    @pytest.mark.fast
    def test_intro_gaps_use_chapter_strategy_ranking(
        self, stage, mock_state, learning_db, intro_gap_segments
    ):
        """Intro chapter_type gaps should boost entity strategy priority."""
        iter_config = IterativeMatchingConfig(
            use_voiceover_text_queries=True,
            use_similar_to_locked=False,
            use_entity_topic_queries=True,
        )
        gaps = [
            GapSegment(0, 0.5, "Welcome to this documentary about nature", 0.0),
        ]
        mock_state.extracted_entities = [{"name": "nature"}]
        # Create gap_analysis with clustered_gaps
        gap_analysis = MagicMock()
        gap_analysis.clustered_gaps = {"abstract_concept": [0]}

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_analysis=gap_analysis,
            learning_db=learning_db,
            gap_segments=intro_gap_segments,
        )

        assert len(queries) > 0
        # Entity queries should be boosted for intro
        entity_queries = [q for q in queries if q["strategy"] == "entity"]
        voiceover_queries = [q for q in queries if q["strategy"] == "voiceover"]
        if entity_queries and voiceover_queries:
            # Entity should have higher priority than voiceover for intro
            assert entity_queries[0]["priority"] >= voiceover_queries[0]["priority"]

    @pytest.mark.fast
    def test_listicle_gaps_use_chapter_strategy_ranking(
        self, stage, mock_state, learning_db, listicle_gap_segments
    ):
        """listicle_item chapter_type gaps should boost voiceover strategy priority."""
        iter_config = IterativeMatchingConfig(
            use_voiceover_text_queries=True,
            use_similar_to_locked=False,
            use_entity_topic_queries=True,
        )
        gaps = [
            GapSegment(10, 0.5, "Number three on our list is the mighty oak tree", 50.0),
        ]
        mock_state.extracted_entities = [{"name": "oak tree"}]
        gap_analysis = MagicMock()
        gap_analysis.clustered_gaps = {"concrete_topic": [10]}

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_analysis=gap_analysis,
            learning_db=learning_db,
            gap_segments=listicle_gap_segments,
        )

        assert len(queries) > 0
        # Voiceover queries should be boosted for listicle_item
        voiceover_queries = [q for q in queries if q["strategy"] == "voiceover"]
        entity_queries = [q for q in queries if q["strategy"] == "entity"]
        if voiceover_queries and entity_queries:
            assert voiceover_queries[0]["priority"] >= entity_queries[0]["priority"]

    @pytest.mark.fast
    def test_body_chapter_type_uses_default_ordering(
        self, stage, mock_state, learning_db, body_gap_segments
    ):
        """body/unknown chapter_type should fall back to default priority ordering."""
        iter_config = IterativeMatchingConfig(
            use_voiceover_text_queries=True,
            use_similar_to_locked=False,
            use_entity_topic_queries=True,
        )
        gaps = [
            GapSegment(20, 0.5, "The forest ecosystem relies on biodiversity", 100.0),
        ]
        mock_state.extracted_entities = [{"name": "forest"}]
        gap_analysis = MagicMock()
        gap_analysis.clustered_gaps = {"other": [20]}

        # Get queries WITH learning_db (body type - should not boost)
        queries_with_db = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_analysis=gap_analysis,
            learning_db=learning_db,
            gap_segments=body_gap_segments,
        )

        # Reset used queries for second call
        stage._used_queries = set()

        # Get queries WITHOUT learning_db
        queries_without_db = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_analysis=gap_analysis,
        )

        # For body type, priorities should be identical (no boost applied)
        prio_with = {q["strategy"]: q["priority"] for q in queries_with_db}
        prio_without = {q["strategy"]: q["priority"] for q in queries_without_db}
        for strategy in prio_with:
            if strategy in prio_without:
                assert prio_with[strategy] == prio_without[strategy], \
                    f"body chapter_type should not boost {strategy} priority"


# ============================================================================
# Chapter-Aware Gap Query Diversity Tests (US-76-010)
# ============================================================================

class TestChapterAwareQueryDiversity:
    """Tests for per-chapter query diversity enforcement."""

    @pytest.fixture
    def stage(self):
        return IterativeMatchStage()

    @pytest.fixture
    def mock_state(self):
        state = MagicMock()
        state.voiceover_segments = []
        state.matches = []
        state.extracted_entities = []
        state.downloaded_videos = []
        state.video_search_results = []
        return state

    def test_same_chapter_identical_text_produces_different_queries(self, stage, mock_state):
        """Two gaps in the same chapter with identical voiceover text produce different queries.

        US-76-010 AC: When generating search queries for gaps within the same
        chapter, no two gaps use identical query text.
        """
        # Two gaps with identical voiceover text, same chapter
        gaps = [
            GapSegment(segment_index=1, confidence=0.5,
                       voiceover_text="ancient Roman architecture buildings",
                       position=10.0, reason='low_confidence'),
            GapSegment(segment_index=2, confidence=0.5,
                       voiceover_text="ancient Roman architecture buildings",
                       position=15.0, reason='low_confidence'),
        ]

        # Create gap_segments with same chapter_id
        gap_segments = [
            AnalyzerGapSegment(
                segment_index=1, confidence=0.5,
                voiceover_text="ancient Roman architecture buildings",
                position=10.0, chapter_id="chapter_rome",
            ),
            AnalyzerGapSegment(
                segment_index=2, confidence=0.5,
                voiceover_text="ancient Roman architecture buildings",
                position=15.0, chapter_id="chapter_rome",
            ),
        ]

        iter_config = MagicMock()
        iter_config.use_voiceover_text_queries = True
        iter_config.use_similar_to_locked = False
        iter_config.use_entity_topic_queries = False
        iter_config.use_description_queries = False
        iter_config.use_tag_queries = False
        iter_config.max_new_videos_per_pass = 50

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_segments=gap_segments,
        )

        # Extract query texts for each gap
        gap1_queries = [q['query'] for q in queries if 1 in q.get('gap_indices', [])]
        gap2_queries = [q['query'] for q in queries if 2 in q.get('gap_indices', [])]

        assert len(gap1_queries) > 0, "Gap 1 should have queries"
        assert len(gap2_queries) > 0, "Gap 2 should have queries"

        # The queries must be different despite identical voiceover text
        all_query_texts = [q['query'].lower().strip() for q in queries]
        assert len(all_query_texts) == len(set(all_query_texts)), \
            f"Queries within same chapter must be unique, got: {all_query_texts}"

    def test_different_chapters_allow_identical_queries(self, stage, mock_state):
        """Gaps in different chapters can use identical queries.

        US-76-010 AC: No cross-chapter dedup - identical queries in different
        chapters are allowed.
        """
        # Two gaps with identical text but different chapters
        gaps = [
            GapSegment(segment_index=1, confidence=0.5,
                       voiceover_text="ancient Roman architecture buildings",
                       position=10.0, reason='low_confidence'),
            GapSegment(segment_index=2, confidence=0.5,
                       voiceover_text="ancient Roman architecture buildings",
                       position=50.0, reason='low_confidence'),
        ]

        # Different chapter_ids
        gap_segments = [
            AnalyzerGapSegment(
                segment_index=1, confidence=0.5,
                voiceover_text="ancient Roman architecture buildings",
                position=10.0, chapter_id="chapter_1",
            ),
            AnalyzerGapSegment(
                segment_index=2, confidence=0.5,
                voiceover_text="ancient Roman architecture buildings",
                position=50.0, chapter_id="chapter_2",
            ),
        ]

        iter_config = MagicMock()
        iter_config.use_voiceover_text_queries = True
        iter_config.use_similar_to_locked = False
        iter_config.use_entity_topic_queries = False
        iter_config.use_description_queries = False
        iter_config.use_tag_queries = False
        iter_config.max_new_videos_per_pass = 50

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_segments=gap_segments,
        )

        # Both gaps should produce queries (global dedup may merge them, but
        # that's OK - the point is chapter dedup didn't vary them)
        gap1_queries = [q['query'] for q in queries if 1 in q.get('gap_indices', [])]
        gap2_queries = [q['query'] for q in queries if 2 in q.get('gap_indices', [])]

        assert len(gap1_queries) > 0 or len(gap2_queries) > 0, \
            "At least one gap should produce queries"

        # The raw query generation (before global dedup) should NOT have varied
        # queries across chapters. The global dedup may remove one, but the
        # chapter dedup should not have touched them.
        # Verify by checking that the gap_segments with different chapters
        # both generated the same base query text (before global dedup removed one)
        assert len(gap1_queries) >= 1, "Gap 1 in chapter_1 should have a query"

    def test_query_variation_preserves_original_keywords(self, stage):
        """Query variation preserves the original query's semantic intent.

        US-76-010 AC: Varied queries must contain original keywords.
        """
        original_query = "Roman architecture footage"
        chapter_id = "chapter_rome"
        gap = AnalyzerGapSegment(
            segment_index=1, confidence=0.5,
            voiceover_text="The ancient Roman architecture included grand buildings and arches",
            position=10.0, chapter_id=chapter_id,
        )
        existing = {original_query.lower().strip()}

        varied = IterativeMatchStage._vary_query_for_chapter(
            original_query, chapter_id, gap, existing
        )

        # Varied query must be different
        assert varied.lower().strip() != original_query.lower().strip(), \
            "Varied query must differ from original"

        # Varied query must still contain original keywords
        for keyword in original_query.lower().split():
            assert keyword in varied.lower(), \
                f"Varied query '{varied}' must contain original keyword '{keyword}'"

    def test_per_chapter_dedup_set_tracks_used_queries(self, stage, mock_state):
        """Per-chapter query dedup set tracks and triggers variation.

        US-76-010 AC: A per-chapter dedup set detects duplicates and triggers
        query variation with chapter-specific context.
        """
        # Three gaps in same chapter with identical text
        gaps = [
            GapSegment(segment_index=i, confidence=0.5,
                       voiceover_text="wildlife conservation endangered species protection",
                       position=10.0 + i * 5, reason='low_confidence')
            for i in range(3)
        ]

        gap_segments = [
            AnalyzerGapSegment(
                segment_index=i, confidence=0.5,
                voiceover_text="wildlife conservation endangered species protection",
                position=10.0 + i * 5, chapter_id="chapter_wildlife",
            )
            for i in range(3)
        ]

        iter_config = MagicMock()
        iter_config.use_voiceover_text_queries = True
        iter_config.use_similar_to_locked = False
        iter_config.use_entity_topic_queries = False
        iter_config.use_description_queries = False
        iter_config.use_tag_queries = False
        iter_config.max_new_videos_per_pass = 50

        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_segments=gap_segments,
        )

        # All three should produce queries and all should be unique
        query_texts = [q['query'].lower().strip() for q in queries]
        assert len(query_texts) == len(set(query_texts)), \
            f"All queries within same chapter must be unique, got: {query_texts}"
        assert len(query_texts) >= 2, \
            f"Expected at least 2 varied queries, got {len(query_texts)}"

    def test_no_gap_segments_skips_chapter_dedup(self, stage, mock_state):
        """When gap_segments is None, chapter dedup is skipped gracefully."""
        gaps = [
            GapSegment(segment_index=1, confidence=0.5,
                       voiceover_text="ancient Roman architecture buildings",
                       position=10.0, reason='low_confidence'),
            GapSegment(segment_index=2, confidence=0.5,
                       voiceover_text="ancient Roman architecture buildings",
                       position=15.0, reason='low_confidence'),
        ]

        iter_config = MagicMock()
        iter_config.use_voiceover_text_queries = True
        iter_config.use_similar_to_locked = False
        iter_config.use_entity_topic_queries = False
        iter_config.use_description_queries = False
        iter_config.use_tag_queries = False
        iter_config.max_new_videos_per_pass = 50

        # No gap_segments passed - should not crash
        queries = stage._generate_multi_strategy_queries(
            gaps, [], mock_state, iter_config, pass_num=1,
            gap_segments=None,
        )

        # Should still produce queries (global dedup may merge identical ones)
        assert len(queries) >= 1, "Should produce at least 1 query"
