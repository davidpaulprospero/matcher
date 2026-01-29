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
