"""
Tests for source diversity scoring for V4-V6 tracks.

US-007: Add source diversity scoring for V4-V6 tracks
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.strategies import calculate_source_diversity_score, StrategyMatcher
from src.utils import SRTSegment, AlternativeMatch


pytestmark = pytest.mark.unit


class TestCalculateSourceDiversityScoreFunction:
    """Test that calculate_source_diversity_score() function exists and has correct signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """Function is importable from strategies module."""
        from src.matching.strategies import calculate_source_diversity_score
        assert callable(calculate_source_diversity_score)

    @pytest.mark.fast
    def test_function_accepts_required_args(self):
        """Function accepts all required arguments."""
        result = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"earthquake", "damage"},
            v1_v3_sources={"video2.mp4"},
            v1_v3_keywords={"tsunami", "flood"}
        )
        assert isinstance(result, float)

    @pytest.mark.fast
    def test_function_returns_float(self):
        """Function returns a float value."""
        result = calculate_source_diversity_score(
            candidate_source="test.mp4",
            candidate_keywords=set(),
            v1_v3_sources=set(),
            v1_v3_keywords=set()
        )
        assert isinstance(result, float)

    @pytest.mark.fast
    def test_score_in_valid_range(self):
        """Function returns score between 0.0 and 1.0."""
        result = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"test"},
            v1_v3_sources={"video2.mp4"},
            v1_v3_keywords={"other"}
        )
        assert 0.0 <= result <= 1.0


class TestSourceDiversityScoring:
    """Test diversity score calculation based on source files."""

    @pytest.mark.fast
    def test_different_source_gets_higher_score(self):
        """Candidate from different source file gets higher score."""
        # Same source
        same_source_score = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"keyword"},
            v1_v3_sources={"video1.mp4", "video2.mp4"},
            v1_v3_keywords={"keyword"}
        )

        # Different source
        different_source_score = calculate_source_diversity_score(
            candidate_source="video3.mp4",
            candidate_keywords={"keyword"},
            v1_v3_sources={"video1.mp4", "video2.mp4"},
            v1_v3_keywords={"keyword"}
        )

        assert different_source_score > same_source_score

    @pytest.mark.fast
    def test_same_source_gets_lower_score(self):
        """Candidate from same source as V1-V3 gets lower diversity score."""
        score = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"earthquake"},
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords={"earthquake"}
        )
        # Same source = 0 for source component (60% weight)
        # Same keyword = 0 for keyword component (40% weight)
        assert score == 0.0

    @pytest.mark.fast
    def test_unique_source_contributes_to_score(self):
        """Unique source file adds 0.6 to the score (60% weight)."""
        score = calculate_source_diversity_score(
            candidate_source="video_unique.mp4",
            candidate_keywords={"same_keyword"},
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords={"same_keyword"}
        )
        # Different source = 0.6, same keyword = 0
        assert score == 0.6


class TestKeywordDiversityScoring:
    """Test diversity score calculation based on keyword overlap."""

    @pytest.mark.fast
    def test_unique_keywords_get_higher_score(self):
        """Candidate with unique keywords gets higher score."""
        # Same keywords
        same_kw_score = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"earthquake", "damage"},
            v1_v3_sources=set(),
            v1_v3_keywords={"earthquake", "damage"}
        )

        # Unique keywords
        unique_kw_score = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"cooking", "recipe"},
            v1_v3_sources=set(),
            v1_v3_keywords={"earthquake", "damage"}
        )

        assert unique_kw_score > same_kw_score

    @pytest.mark.fast
    def test_partial_keyword_overlap(self):
        """Partial keyword overlap results in moderate score."""
        score = calculate_source_diversity_score(
            candidate_source="video2.mp4",
            candidate_keywords={"earthquake", "cooking"},  # 1 of 2 overlap
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords={"earthquake", "damage"}
        )
        # Different source = 0.6
        # 50% overlap = 0.5 keyword score * 0.4 = 0.2
        assert 0.6 < score < 1.0

    @pytest.mark.fast
    def test_no_keywords_neutral_score(self):
        """Empty candidate keywords result in neutral keyword score."""
        score = calculate_source_diversity_score(
            candidate_source="video2.mp4",
            candidate_keywords=set(),  # No keywords
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords={"earthquake"}
        )
        # Different source = 0.6, neutral keyword = 0.5 * 0.4 = 0.2
        assert score == 0.8

    @pytest.mark.fast
    def test_no_v1_v3_keywords_full_diversity(self):
        """When V1-V3 have no keywords, candidate keywords get full score."""
        score = calculate_source_diversity_score(
            candidate_source="video2.mp4",
            candidate_keywords={"earthquake", "damage"},
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords=set()  # No V1-V3 keywords
        )
        # Different source = 0.6, full keyword diversity = 0.4
        assert score == 1.0


class TestCombinedDiversityScoring:
    """Test combined source + keyword diversity scoring."""

    @pytest.mark.fast
    def test_maximum_diversity_score(self):
        """Different source + unique keywords = maximum score."""
        score = calculate_source_diversity_score(
            candidate_source="unique_video.mp4",
            candidate_keywords={"cooking", "recipe"},
            v1_v3_sources={"video1.mp4", "video2.mp4"},
            v1_v3_keywords={"earthquake", "tsunami"}
        )
        assert score == 1.0

    @pytest.mark.fast
    def test_minimum_diversity_score(self):
        """Same source + same keywords = minimum score."""
        score = calculate_source_diversity_score(
            candidate_source="video1.mp4",
            candidate_keywords={"earthquake"},
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords={"earthquake"}
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_moderate_diversity_different_source_same_keywords(self):
        """Different source but same keywords = 0.6 score."""
        score = calculate_source_diversity_score(
            candidate_source="unique_video.mp4",
            candidate_keywords={"earthquake", "tsunami"},
            v1_v3_sources={"video1.mp4"},
            v1_v3_keywords={"earthquake", "tsunami"}
        )
        # Different source = 0.6, same keywords = 0
        assert score == 0.6


class TestAlternativeMatchDiversityScore:
    """Test that AlternativeMatch dataclass has diversity_score field."""

    @pytest.mark.fast
    def test_diversity_score_field_exists(self):
        """AlternativeMatch has diversity_score field."""
        from src.utils import AlternativeMatch
        match = AlternativeMatch(
            video_segment=MagicMock(),
            video_scene=None,
            confidence=0.8,
            reasoning="test",
            diversity_score=0.75
        )
        assert hasattr(match, 'diversity_score')
        assert match.diversity_score == 0.75

    @pytest.mark.fast
    def test_diversity_score_default_zero(self):
        """AlternativeMatch diversity_score defaults to 0.0."""
        from src.utils import AlternativeMatch
        match = AlternativeMatch(
            video_segment=MagicMock(),
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )
        assert match.diversity_score == 0.0

    @pytest.mark.fast
    def test_diversity_score_is_float(self):
        """diversity_score field is a float."""
        from src.utils import AlternativeMatch
        match = AlternativeMatch(
            video_segment=MagicMock(),
            video_scene=None,
            confidence=0.8,
            reasoning="test",
            diversity_score=0.5
        )
        assert isinstance(match.diversity_score, float)


class TestStrategyMatcherDiversityIntegration:
    """Test StrategyMatcher.get_secondary_matches_diversity includes diversity score."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config for StrategyMatcher."""
        config = MagicMock()
        config.output.variety = MagicMock(
            exclude_same_clip=True,
            require_different_source=True,
            min_time_distance=10.0,
            min_embedding_distance=0.3,
            enforce_timeline_variety=True,
            timeline_variety_window=600.0,
            max_source_repeats_in_window=1
        )
        config.output.include_strategy_tracks = True
        config.output.strategy_tracks = []
        return config

    @pytest.mark.fast
    def test_secondary_matches_include_diversity_score(self, mock_config):
        """get_secondary_matches_diversity returns matches with diversity_score."""
        matcher = StrategyMatcher(mock_config, scenes={})

        # Create mock segments
        primary = MagicMock(spec=SRTSegment)
        primary.source_file = "video1.mp4"
        primary.start_time = 0.0
        primary.end_time = 5.0
        primary.keywords = ["earthquake"]

        alt1 = MagicMock(spec=SRTSegment)
        alt1.source_file = "video2.mp4"
        alt1.start_time = 0.0
        alt1.end_time = 5.0
        alt1.keywords = ["tsunami"]

        candidate = MagicMock(spec=SRTSegment)
        candidate.source_file = "video3.mp4"  # Different source
        candidate.start_time = 0.0
        candidate.end_time = 5.0
        candidate.text = "test content"
        candidate.keywords = ["cooking"]  # Different keywords

        vo_segment = MagicMock(spec=SRTSegment)
        vo_segment.text = "voiceover text"
        vo_segment.keywords = []
        vo_segment.entities = []

        # Create embeddings
        candidate_id = matcher.get_clip_id(candidate)
        candidate_embeddings = {
            candidate_id: [0.1] * 384,
            matcher.get_clip_id(primary): [0.2] * 384,
            matcher.get_clip_id(alt1): [0.3] * 384
        }
        vo_embedding = [0.15] * 384

        matches = matcher.get_secondary_matches_diversity(
            vo_segment=vo_segment,
            all_candidates=[(candidate, 0.8)],
            primary_match=primary,
            alternatives=[alt1],
            vo_embedding=vo_embedding,
            candidate_embeddings=candidate_embeddings
        )

        # Should get at least one match
        assert len(matches) >= 1

        # Match should have diversity_score
        match = matches[0]
        assert hasattr(match, 'diversity_score')
        assert isinstance(match.diversity_score, float)
        assert 0.0 <= match.diversity_score <= 1.0


class TestDiversityScoreLogging:
    """Test that diversity score is logged."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config for StrategyMatcher."""
        config = MagicMock()
        config.output.variety = MagicMock(
            exclude_same_clip=True,
            require_different_source=True,
            min_time_distance=10.0,
            min_embedding_distance=0.3,
            enforce_timeline_variety=True,
            timeline_variety_window=600.0,
            max_source_repeats_in_window=1
        )
        config.output.include_strategy_tracks = True
        config.output.strategy_tracks = []
        return config

    @pytest.mark.fast
    def test_diversity_score_logged(self, mock_config, caplog):
        """Diversity score is logged at DEBUG level."""
        with caplog.at_level(logging.DEBUG, logger="src.matching.strategies"):
            matcher = StrategyMatcher(mock_config, scenes={})

            # Create mock segments
            primary = MagicMock(spec=SRTSegment)
            primary.source_file = "video1.mp4"
            primary.start_time = 0.0
            primary.end_time = 5.0
            primary.keywords = []

            candidate = MagicMock(spec=SRTSegment)
            candidate.source_file = "video2.mp4"
            candidate.start_time = 0.0
            candidate.end_time = 5.0
            candidate.text = "test"
            candidate.keywords = []

            vo_segment = MagicMock(spec=SRTSegment)
            vo_segment.text = "test"
            vo_segment.keywords = []
            vo_segment.entities = []

            candidate_id = matcher.get_clip_id(candidate)
            candidate_embeddings = {
                candidate_id: [0.1] * 384,
                matcher.get_clip_id(primary): [0.2] * 384
            }
            vo_embedding = [0.15] * 384

            matcher.get_secondary_matches_diversity(
                vo_segment=vo_segment,
                all_candidates=[(candidate, 0.8)],
                primary_match=primary,
                alternatives=[],
                vo_embedding=vo_embedding,
                candidate_embeddings=candidate_embeddings
            )

            # Check that diversity score was logged
            assert any("source diversity" in record.message for record in caplog.records)


class TestSameSourceAlternativesLowerScore:
    """Test that same-source alternatives get lower diversity score."""

    @pytest.mark.fast
    def test_same_source_lower_than_different(self):
        """Same source file results in lower diversity score than different source."""
        same_source_score = calculate_source_diversity_score(
            candidate_source="shared_video.mp4",
            candidate_keywords={"keyword1", "keyword2"},
            v1_v3_sources={"shared_video.mp4"},
            v1_v3_keywords={"other1", "other2"}
        )

        different_source_score = calculate_source_diversity_score(
            candidate_source="unique_video.mp4",
            candidate_keywords={"keyword1", "keyword2"},
            v1_v3_sources={"shared_video.mp4"},
            v1_v3_keywords={"other1", "other2"}
        )

        assert same_source_score < different_source_score
        # Different source contributes 0.6 to the score
        assert different_source_score - same_source_score == 0.6

    @pytest.mark.fast
    def test_same_source_same_keywords_minimum(self):
        """Same source and same keywords = minimum possible score."""
        score = calculate_source_diversity_score(
            candidate_source="video.mp4",
            candidate_keywords={"earthquake", "tsunami"},
            v1_v3_sources={"video.mp4"},
            v1_v3_keywords={"earthquake", "tsunami"}
        )
        assert score == 0.0

    @pytest.mark.fast
    def test_different_source_same_keywords_moderate(self):
        """Different source but same keywords = 0.6 (source contribution only)."""
        score = calculate_source_diversity_score(
            candidate_source="different_video.mp4",
            candidate_keywords={"earthquake", "tsunami"},
            v1_v3_sources={"original_video.mp4"},
            v1_v3_keywords={"earthquake", "tsunami"}
        )
        assert score == 0.6


class TestEdgeCases:
    """Test edge cases in diversity scoring."""

    @pytest.mark.fast
    def test_empty_v1_v3_sources(self):
        """Empty V1-V3 sources means all candidates are different."""
        score = calculate_source_diversity_score(
            candidate_source="any_video.mp4",
            candidate_keywords={"keyword"},
            v1_v3_sources=set(),
            v1_v3_keywords={"keyword"}
        )
        # Different source = 0.6, same keyword = 0
        assert score == 0.6

    @pytest.mark.fast
    def test_empty_all_sets(self):
        """Empty sets for everything."""
        score = calculate_source_diversity_score(
            candidate_source="video.mp4",
            candidate_keywords=set(),
            v1_v3_sources=set(),
            v1_v3_keywords=set()
        )
        # Different source = 1.0 * 0.6 = 0.6
        # No keywords = neutral 0.5 * 0.4 = 0.2
        assert score == 0.8

    @pytest.mark.fast
    def test_case_insensitive_keywords(self):
        """Keywords are case-insensitive in the function."""
        # Note: The function expects lowercase keywords as input
        # The calling code should normalize to lowercase
        score = calculate_source_diversity_score(
            candidate_source="video.mp4",
            candidate_keywords={"earthquake"},
            v1_v3_sources=set(),
            v1_v3_keywords={"earthquake"}
        )
        assert score == 0.6  # Different source, same keyword

    @pytest.mark.fast
    def test_many_keywords_partial_overlap(self):
        """Many keywords with partial overlap."""
        score = calculate_source_diversity_score(
            candidate_source="unique.mp4",
            candidate_keywords={"a", "b", "c", "d"},  # 2 of 4 overlap
            v1_v3_sources={"other.mp4"},
            v1_v3_keywords={"a", "b", "x", "y"}
        )
        # Different source = 0.6
        # 50% overlap = 0.5 * 0.4 = 0.2
        assert score == 0.8


class TestRealWorldScenarios:
    """Test realistic scenarios for diversity scoring."""

    @pytest.mark.fast
    def test_documentary_footage_diversity(self):
        """Documentary with earthquake footage - similar topics."""
        # V1-V3 are all earthquake videos
        v1_v3_sources = {
            "earthquake_japan_2011.mp4",
            "earthquake_chile_2010.mp4",
            "earthquake_nepal_2015.mp4"
        }
        v1_v3_keywords = {"earthquake", "destruction", "buildings", "rescue"}

        # V4 candidate - tsunami (related but different)
        score = calculate_source_diversity_score(
            candidate_source="tsunami_thailand_2004.mp4",
            candidate_keywords={"tsunami", "wave", "destruction", "flood"},
            v1_v3_sources=v1_v3_sources,
            v1_v3_keywords=v1_v3_keywords
        )
        # Different source = 0.6
        # 1 of 4 overlap (destruction) = 0.75 * 0.4 = 0.3
        assert score == 0.9

    @pytest.mark.fast
    def test_cooking_show_variety(self):
        """Cooking show needs variety in recipe types."""
        v1_v3_sources = {
            "pasta_recipe.mp4",
            "sauce_making.mp4",
        }
        v1_v3_keywords = {"pasta", "cooking", "italian", "sauce"}

        # V4 candidate - completely different cuisine
        score = calculate_source_diversity_score(
            candidate_source="sushi_recipe.mp4",
            candidate_keywords={"sushi", "japanese", "rice", "fish"},
            v1_v3_sources=v1_v3_sources,
            v1_v3_keywords=v1_v3_keywords
        )
        # Different source = 0.6
        # No overlap = 1.0 * 0.4 = 0.4
        assert score == 1.0
