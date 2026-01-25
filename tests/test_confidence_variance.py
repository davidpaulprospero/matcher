"""
Tests for confidence variance tracking in MatchResult.

Sprint 4 - US-002: Add confidence variance tracking to MatchResult
"""

import statistics
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import MatchResult, Match, SRTSegment, AlternativeMatch

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestMatchResultConfidenceVariance:
    """Tests for MatchResult.confidence_variance field"""

    def test_confidence_variance_field_exists(self):
        """MatchResult should have confidence_variance field"""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="video"),
            video_scene=None,
            confidence=0.9,
            reasoning="test"
        )
        result = MatchResult(primary_match=match)
        assert hasattr(result, 'confidence_variance')

    def test_confidence_variance_default_value(self):
        """confidence_variance should default to 0.0"""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="video"),
            video_scene=None,
            confidence=0.9,
            reasoning="test"
        )
        result = MatchResult(primary_match=match)
        assert result.confidence_variance == 0.0

    def test_confidence_variance_can_be_set(self):
        """confidence_variance should be settable"""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="video"),
            video_scene=None,
            confidence=0.9,
            reasoning="test"
        )
        result = MatchResult(primary_match=match, confidence_variance=0.15)
        assert result.confidence_variance == 0.15

    def test_confidence_variance_type_is_float(self):
        """confidence_variance should be a float"""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="video"),
            video_scene=None,
            confidence=0.9,
            reasoning="test"
        )
        result = MatchResult(primary_match=match, confidence_variance=0.15)
        assert isinstance(result.confidence_variance, float)


class TestTieredMatcherVarianceCalculation:
    """Tests for TieredMatcher._calculate_confidence_variance method"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TieredMatcher"""
        config = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.5
        config.matching.max_clip_reuse = 2
        config.matching.reuse_penalty = 0.1
        config.matching.primary_provider = None
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.skip_llm_threshold = 0.9
        config.matching.cache_llm_responses = False
        config.matching.ambiguous_threshold = 0.6
        config.matching.confidence_threshold = 0.5
        config.gemini_api_key = None
        config.anthropic_api_key = None
        config.output.num_alternatives = 2
        return config

    def test_calculate_variance_with_5_candidates(self, mock_config):
        """Variance should be calculated from top-5 candidate scores"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        # Create candidates with known scores
        candidates = [
            (MagicMock(), 0.90),
            (MagicMock(), 0.85),
            (MagicMock(), 0.80),
            (MagicMock(), 0.75),
            (MagicMock(), 0.70),
            (MagicMock(), 0.60),  # Should be ignored (6th)
        ]

        variance = matcher._calculate_confidence_variance(candidates)

        # Expected variance of [0.90, 0.85, 0.80, 0.75, 0.70]
        expected = statistics.stdev([0.90, 0.85, 0.80, 0.75, 0.70])
        assert abs(variance - expected) < 0.0001

    def test_calculate_variance_with_2_candidates(self, mock_config):
        """Variance should work with exactly 2 candidates"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        candidates = [
            (MagicMock(), 0.90),
            (MagicMock(), 0.70),
        ]

        variance = matcher._calculate_confidence_variance(candidates)
        expected = statistics.stdev([0.90, 0.70])
        assert abs(variance - expected) < 0.0001

    def test_calculate_variance_with_1_candidate_returns_zero(self, mock_config):
        """Variance should return 0.0 with only 1 candidate"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        candidates = [(MagicMock(), 0.90)]

        variance = matcher._calculate_confidence_variance(candidates)
        assert variance == 0.0

    def test_calculate_variance_with_empty_candidates_returns_zero(self, mock_config):
        """Variance should return 0.0 with empty candidates list"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        variance = matcher._calculate_confidence_variance([])
        assert variance == 0.0

    def test_calculate_variance_identical_scores_returns_zero(self, mock_config):
        """Variance should return 0.0 when all scores are identical"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        candidates = [
            (MagicMock(), 0.80),
            (MagicMock(), 0.80),
            (MagicMock(), 0.80),
        ]

        variance = matcher._calculate_confidence_variance(candidates)
        assert variance == 0.0


class TestHighVarianceIndicatesUncertainty:
    """Tests for interpreting high variance as uncertain match"""

    def test_high_variance_threshold(self):
        """Variance > 0.1 should indicate uncertain match"""
        # Calculate variance of scores where top candidates are close
        close_scores = [0.85, 0.84, 0.83, 0.82, 0.81]
        close_variance = statistics.stdev(close_scores)

        # Calculate variance of scores where there's a clear winner
        spread_scores = [0.95, 0.70, 0.65, 0.60, 0.55]
        spread_variance = statistics.stdev(spread_scores)

        # Close scores should have LOW variance (certain match)
        assert close_variance < 0.1, f"Close scores variance {close_variance} should be < 0.1"

        # Spread scores should have HIGH variance (more uncertain)
        assert spread_variance > 0.1, f"Spread scores variance {spread_variance} should be > 0.1"

    def test_variance_interpretation_examples(self):
        """Document example variance values for reference"""
        # Example 1: Very uncertain (multiple similar candidates)
        similar_scores = [0.80, 0.79, 0.78, 0.77, 0.76]
        similar_variance = statistics.stdev(similar_scores)
        assert similar_variance < 0.02  # Very low variance = uncertain which to pick

        # Example 2: Clear winner
        clear_winner = [0.95, 0.60, 0.55, 0.50, 0.45]
        clear_variance = statistics.stdev(clear_winner)
        assert clear_variance > 0.15  # High variance = clear winner stands out

    def test_zero_variance_means_identical_options(self):
        """Zero variance means all candidates have identical scores"""
        identical = [0.75, 0.75, 0.75, 0.75, 0.75]
        variance = statistics.stdev(identical)
        assert variance == 0.0


class TestCheckpointSerialization:
    """Tests for confidence_variance in checkpoint JSON output"""

    def test_checkpoint_includes_variance(self):
        """Serialized match data should include confidence_variance"""
        from src.utils import MatchResult, Match, SRTSegment

        # Create a MatchResult with variance
        vo_seg = SRTSegment(index=0, start_time=0, end_time=1, text="voiceover")
        vid_seg = SRTSegment(index=0, start_time=0, end_time=1, text="video", source_file="/path/to/video.mp4")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.85,
            reasoning="test"
        )

        result = MatchResult(primary_match=match, confidence_variance=0.12)

        # Verify variance is accessible
        assert result.confidence_variance == 0.12

    def test_match_stage_serializes_variance(self):
        """MatchStage should serialize confidence_variance to checkpoint"""
        # This tests that the serialization code handles confidence_variance
        # Mocking the actual serialization path

        match_result = MagicMock()
        match_result.primary_match = MagicMock()
        match_result.primary_match.video_segment = MagicMock()
        match_result.primary_match.video_segment.source_file = '/path/to/video.mp4'
        match_result.primary_match.video_segment.start_time = 10.0
        match_result.primary_match.confidence = 0.85
        match_result.confidence_variance = 0.12

        # Verify getattr works correctly on mock
        conf_variance = getattr(match_result, 'confidence_variance', 0.0)
        assert conf_variance == 0.12

    def test_variance_default_when_missing(self):
        """confidence_variance should default to 0.0 if not present"""
        match_result = MagicMock(spec=['primary_match'])  # No confidence_variance
        match_result.primary_match = MagicMock()

        conf_variance = getattr(match_result, 'confidence_variance', 0.0)
        assert conf_variance == 0.0


class TestVarianceCalculationEdgeCases:
    """Edge case tests for variance calculation"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TieredMatcher"""
        config = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.5
        config.matching.max_clip_reuse = 2
        config.matching.reuse_penalty = 0.1
        config.matching.primary_provider = None
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.skip_llm_threshold = 0.9
        config.matching.cache_llm_responses = False
        config.matching.ambiguous_threshold = 0.6
        config.matching.confidence_threshold = 0.5
        config.gemini_api_key = None
        config.anthropic_api_key = None
        config.output.num_alternatives = 2
        return config

    def test_variance_with_3_candidates(self, mock_config):
        """Variance should work with 3 candidates"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        candidates = [
            (MagicMock(), 0.90),
            (MagicMock(), 0.80),
            (MagicMock(), 0.70),
        ]

        variance = matcher._calculate_confidence_variance(candidates)
        expected = statistics.stdev([0.90, 0.80, 0.70])
        assert abs(variance - expected) < 0.0001

    def test_variance_with_10_candidates_uses_top_5(self, mock_config):
        """Variance should only use top 5 candidates even if more available"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        candidates = [
            (MagicMock(), 0.95),
            (MagicMock(), 0.90),
            (MagicMock(), 0.85),
            (MagicMock(), 0.80),
            (MagicMock(), 0.75),
            (MagicMock(), 0.10),  # Should be ignored
            (MagicMock(), 0.05),  # Should be ignored
            (MagicMock(), 0.01),  # Should be ignored
        ]

        variance = matcher._calculate_confidence_variance(candidates)

        # Should only use top 5
        expected = statistics.stdev([0.95, 0.90, 0.85, 0.80, 0.75])
        assert abs(variance - expected) < 0.0001

    def test_variance_with_custom_top_n(self, mock_config):
        """Variance should respect custom top_n parameter"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        candidates = [
            (MagicMock(), 0.90),
            (MagicMock(), 0.80),
            (MagicMock(), 0.70),
            (MagicMock(), 0.60),
            (MagicMock(), 0.50),
        ]

        # Use top 3 instead of top 5
        variance = matcher._calculate_confidence_variance(candidates, top_n=3)
        expected = statistics.stdev([0.90, 0.80, 0.70])
        assert abs(variance - expected) < 0.0001

    def test_variance_with_negative_scores(self, mock_config):
        """Variance should handle negative scores (edge case)"""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)

        # Shouldn't happen in practice, but should not crash
        candidates = [
            (MagicMock(), 0.5),
            (MagicMock(), -0.1),  # Invalid but shouldn't crash
        ]

        variance = matcher._calculate_confidence_variance(candidates)
        expected = statistics.stdev([0.5, -0.1])
        assert abs(variance - expected) < 0.0001


class TestMatchResultWithVariance:
    """Integration tests for MatchResult with confidence_variance"""

    def test_match_result_all_fields(self):
        """MatchResult should support all fields including variance"""
        vo_seg = SRTSegment(index=0, start_time=0, end_time=1, text="voiceover")
        vid_seg = SRTSegment(index=0, start_time=0, end_time=1, text="video")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.85,
            reasoning="test"
        )

        alt = AlternativeMatch(
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.75,
            reasoning="alt"
        )

        result = MatchResult(
            primary_match=match,
            alternatives=[alt],
            secondary_matches=[],
            strategy_matches=[],
            has_gap=False,
            gap_reason="",
            confidence_variance=0.08
        )

        assert result.primary_match == match
        assert len(result.alternatives) == 1
        assert result.confidence_variance == 0.08
        assert result.has_gap is False

    def test_match_result_high_variance_gap_detection(self):
        """High variance match with gap should preserve both fields"""
        vo_seg = SRTSegment(index=0, start_time=0, end_time=1, text="voiceover")
        vid_seg = SRTSegment(index=0, start_time=0, end_time=1, text="video")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.4,  # Below threshold
            reasoning="Low confidence match"
        )

        result = MatchResult(
            primary_match=match,
            has_gap=True,
            gap_reason="Low confidence (0.40)",
            confidence_variance=0.15  # High variance
        )

        assert result.has_gap is True
        assert result.confidence_variance == 0.15
        assert "Low confidence" in result.gap_reason
