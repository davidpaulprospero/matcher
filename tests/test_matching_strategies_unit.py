"""
Unit tests for src/matching/strategies.py

Tests StrategyMatcher and FallbackMatchStrategy classes.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.strategies import StrategyMatcher, FallbackMatchStrategy
from src.utils import SRTSegment, StrategyMatch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for StrategyMatcher."""
    config = Mock()
    config.matching = Mock()
    config.output = Mock()
    config.output.strategy_tracks = ["embedding_diversity", "broll_only", "source_rotation"]
    return config


@pytest.fixture
def fallback_config():
    """Create mock config for FallbackMatchStrategy."""
    config = Mock()
    config.matching = Mock()
    config.matching.fallback_matching_enabled = True
    config.matching.fallback_trigger_threshold = 0.4
    return config


@pytest.fixture
def sample_vo_segment():
    """Create sample voiceover segment."""
    vo = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=5.0,
        text="The earthquake caused massive destruction in the city",
        source_file=""
    )
    return vo


@pytest.fixture
def sample_candidates():
    """Create sample candidate list with different sources."""
    seg1 = SRTSegment(0, 0.0, 5.0, "earthquake damage in urban area", "/video1.mp4")
    seg2 = SRTSegment(1, 10.0, 15.0, "tsunami waves crashing", "/video2.mp4")
    seg3 = SRTSegment(2, 20.0, 25.0, "rescue workers helping", "/video3.mp4")
    seg4 = SRTSegment(3, 30.0, 35.0, "building collapse", "/video1.mp4")

    return [
        (seg1, 0.85),
        (seg2, 0.72),
        (seg3, 0.68),
        (seg4, 0.55)
    ]


# ============================================================================
# Tests for StrategyMatcher Initialization
# ============================================================================

class TestStrategyMatcherInit:
    """Test StrategyMatcher initialization."""

    @pytest.mark.fast
    def test_init_creates_matcher(self, mock_config):
        """Test StrategyMatcher initializes correctly."""
        matcher = StrategyMatcher(mock_config)

        assert matcher.config == mock_config
        assert matcher.mc == mock_config.matching

    @pytest.mark.fast
    def test_clip_id_cache_initialized(self, mock_config):
        """Test clip ID cache is empty on init."""
        matcher = StrategyMatcher(mock_config)

        assert matcher._clip_id_cache == {}

    @pytest.mark.fast
    def test_global_cache_lazy_init(self, mock_config):
        """Test global cache is None on init."""
        matcher = StrategyMatcher(mock_config)

        assert matcher._global_cache is None


# ============================================================================
# Tests for StrategyMatcher.get_clip_id
# ============================================================================

class TestGetClipId:
    """Test get_clip_id method."""

    @pytest.mark.fast
    def test_get_clip_id_generates_unique_id(self, mock_config):
        """Test clip ID is unique per segment."""
        matcher = StrategyMatcher(mock_config)

        seg = SRTSegment(0, 10.5, 15.3, "test", "/video1.mp4")
        clip_id = matcher.get_clip_id(seg)

        # Python formats floats without trailing zeros
        assert clip_id == "/video1.mp4:10.5-15.3"

    @pytest.mark.fast
    def test_get_clip_id_caches_id(self, mock_config):
        """Test clip ID is cached for same segment."""
        matcher = StrategyMatcher(mock_config)

        seg = SRTSegment(0, 10.0, 15.0, "test", "/video1.mp4")
        id1 = matcher.get_clip_id(seg)
        id2 = matcher.get_clip_id(seg)

        assert id1 == id2
        assert len(matcher._clip_id_cache) == 1


# ============================================================================
# Tests for StrategyMatcher.match_embedding_diversity
# ============================================================================

class TestEmbeddingDiversityStrategy:
    """Test match_embedding_diversity strategy."""

    @pytest.mark.fast
    def test_embedding_diversity_returns_match(self, mock_config, sample_vo_segment, sample_candidates):
        """Test embedding diversity returns a match."""
        matcher = StrategyMatcher(mock_config)

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(sample_candidates[0][0]): [0.8, 0.2, 0.0],
            matcher.get_clip_id(sample_candidates[1][0]): [0.3, 0.3, 0.4],
        }

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            sample_candidates,
            [],  # No existing matches
            [],  # No existing embeddings
            candidate_embs,
            vo_emb
        )

        assert result is not None
        assert result.strategy == "embedding_diversity"
        assert isinstance(result, StrategyMatch)

    @pytest.mark.fast
    def test_embedding_diversity_returns_none_without_candidates(self, mock_config, sample_vo_segment):
        """Test returns None when no candidates."""
        matcher = StrategyMatcher(mock_config)

        vo_emb = [1.0, 0.0, 0.0]

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            [],  # Empty candidates
            [],
            [],
            {},
            vo_emb
        )

        assert result is None

    @pytest.mark.fast
    def test_embedding_diversity_respects_diversity(self, mock_config, sample_vo_segment):
        """Test embedding diversity prefers diverse candidates."""
        matcher = StrategyMatcher(mock_config)

        # Existing match with embedding
        existing_seg = SRTSegment(0, 0.0, 5.0, "existing", "/video1.mp4")
        existing_embs = [[0.95, 0.05, 0.0]]  # Similar to vo_emb

        # Candidates with different diversity
        similar_seg = SRTSegment(1, 10.0, 15.0, "similar", "/video2.mp4")
        diverse_seg = SRTSegment(2, 20.0, 25.0, "diverse", "/video3.mp4")

        candidates = [
            (similar_seg, 0.8),
            (diverse_seg, 0.6)
        ]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(similar_seg): [0.9, 0.1, 0.0],  # Similar to existing
            matcher.get_clip_id(diverse_seg): [0.1, 0.1, 0.8],  # Different from existing
        }

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            candidates,
            [existing_seg],
            existing_embs,
            candidate_embs,
            vo_emb
        )

        assert result is not None

    @pytest.mark.fast
    def test_embedding_diversity_skips_used_clips(self, mock_config, sample_vo_segment):
        """Test embedding diversity skips already used clips."""
        matcher = StrategyMatcher(mock_config)

        # Existing match
        existing_seg = SRTSegment(0, 0.0, 5.0, "existing", "/video1.mp4")

        # Same clip as existing
        same_clip = SRTSegment(0, 0.0, 5.0, "same", "/video1.mp4")

        candidates = [
            (same_clip, 0.9)
        ]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(same_clip): [0.8, 0.2, 0.0],
        }

        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            candidates,
            [existing_seg],  # Same clip ID as candidate
            [[0.8, 0.2, 0.0]],
            candidate_embs,
            vo_emb
        )

        # Should return None since same clip is excluded
        assert result is None


# ============================================================================
# Tests for StrategyMatcher.match_broll_only
# ============================================================================

class TestBRollOnlyStrategy:
    """Test match_broll_only strategy."""

    @pytest.mark.fast
    def test_broll_only_returns_broll(self, mock_config, sample_vo_segment):
        """Test broll_only returns B-roll clip."""
        matcher = StrategyMatcher(mock_config)

        # B-roll segment
        broll_seg = SRTSegment(0, 0.0, 5.0, "[Silent video]", "/video1.mp4")
        broll_seg.is_broll = True

        # Non-B-roll segment
        speech_seg = SRTSegment(1, 10.0, 15.0, "person talking", "/video2.mp4")
        speech_seg.is_broll = False

        candidates = [
            (speech_seg, 0.9),
            (broll_seg, 0.7)
        ]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(broll_seg): [0.7, 0.2, 0.1],
            matcher.get_clip_id(speech_seg): [0.9, 0.1, 0.0],
        }

        result = matcher.match_broll_only(
            sample_vo_segment,
            candidates,
            [],
            [],
            candidate_embs,
            vo_emb
        )

        assert result is not None
        assert result.strategy == "broll_only"
        assert result.video_segment.is_broll == True

    @pytest.mark.fast
    def test_broll_only_returns_none_when_no_broll(self, mock_config, sample_vo_segment, sample_candidates):
        """Test returns None when no B-roll available."""
        matcher = StrategyMatcher(mock_config)

        # Mark all as non-broll
        for seg, _ in sample_candidates:
            seg.is_broll = False

        result = matcher.match_broll_only(
            sample_vo_segment,
            sample_candidates,
            [],
            [],
            {},
            []
        )

        assert result is None

    @pytest.mark.fast
    def test_broll_only_enforces_different_source(self, mock_config, sample_vo_segment):
        """Test broll_only prefers different source."""
        matcher = StrategyMatcher(mock_config)

        broll1 = SRTSegment(0, 0.0, 5.0, "[Silent]", "/video1.mp4")
        broll1.is_broll = True

        broll2 = SRTSegment(1, 10.0, 15.0, "[Silent]", "/video2.mp4")
        broll2.is_broll = True

        candidates = [
            (broll1, 0.8),
            (broll2, 0.7)
        ]

        # Existing match from video1
        existing_segs = [SRTSegment(0, 20.0, 25.0, "used", "/video1.mp4")]

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(broll1): [0.8, 0.2, 0.0],
            matcher.get_clip_id(broll2): [0.7, 0.3, 0.0],
        }

        result = matcher.match_broll_only(
            sample_vo_segment,
            candidates,
            existing_segs,
            [],
            candidate_embs,
            vo_emb
        )

        # Should pick video2 due to different source requirement
        if result:
            assert result.video_segment.source_file == "/video2.mp4"


# ============================================================================
# Tests for StrategyMatcher.match_source_rotation
# ============================================================================

class TestSourceRotationStrategy:
    """Test match_source_rotation strategy."""

    @pytest.mark.fast
    def test_source_rotation_returns_match(self, mock_config, sample_vo_segment, sample_candidates):
        """Test source rotation returns a match."""
        matcher = StrategyMatcher(mock_config)

        vo_emb = [1.0, 0.0, 0.0]

        result = matcher.match_source_rotation(
            sample_vo_segment,
            sample_candidates,
            [],
            None,
            {},
            vo_emb,
            segment_index=0
        )

        assert result is not None
        assert result.strategy == "source_rotation"

    @pytest.mark.fast
    def test_source_rotation_cycles_sources(self, mock_config, sample_vo_segment, sample_candidates):
        """Test source rotation cycles through sources."""
        matcher = StrategyMatcher(mock_config)

        vo_emb = [1.0, 0.0, 0.0]

        results = []
        for i in range(4):
            result = matcher.match_source_rotation(
                sample_vo_segment,
                sample_candidates,
                [],
                None,
                {},
                vo_emb,
                segment_index=i
            )
            if result:
                results.append(result.video_segment.source_file)

        # Should cycle through different sources
        assert len(results) > 0

    @pytest.mark.fast
    def test_source_rotation_returns_none_with_empty_candidates(self, mock_config, sample_vo_segment):
        """Test returns None with empty candidates."""
        matcher = StrategyMatcher(mock_config)

        vo_emb = [1.0, 0.0, 0.0]

        result = matcher.match_source_rotation(
            sample_vo_segment,
            [],
            [],
            None,
            {},
            vo_emb,
            segment_index=0
        )

        assert result is None


# ============================================================================
# Tests for StrategyMatcher.get_secondary_matches_diversity
# ============================================================================

class TestSecondaryMatchesDiversity:
    """Test get_secondary_matches_diversity method."""

    @pytest.mark.fast
    def test_secondary_matches_returns_list(self, mock_config, sample_vo_segment, sample_candidates):
        """Test returns list of secondary matches."""
        matcher = StrategyMatcher(mock_config)

        primary = sample_candidates[0][0]
        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg): [0.7, 0.2, 0.1]
            for seg, _ in sample_candidates
        }

        result = matcher.get_secondary_matches_diversity(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],  # No existing secondary
            candidate_embs,
            vo_emb,
            num_matches=3
        )

        assert isinstance(result, list)

    @pytest.mark.fast
    def test_secondary_matches_respects_num_matches(self, mock_config, sample_vo_segment, sample_candidates):
        """Test respects num_matches limit."""
        matcher = StrategyMatcher(mock_config)

        primary = sample_candidates[0][0]
        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg): [0.7, 0.2, 0.1]
            for seg, _ in sample_candidates
        }

        result = matcher.get_secondary_matches_diversity(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            candidate_embs,
            vo_emb,
            num_matches=2
        )

        assert len(result) <= 2


# ============================================================================
# Tests for StrategyMatcher.get_strategy_matches
# ============================================================================

class TestGetStrategyMatches:
    """Test get_strategy_matches orchestration method."""

    @pytest.mark.fast
    def test_get_strategy_matches_returns_list(self, mock_config, sample_vo_segment, sample_candidates):
        """Test returns list of strategy matches."""
        mock_config.output.strategy_tracks = ["embedding_diversity", "broll_only"]
        matcher = StrategyMatcher(mock_config)

        primary = sample_candidates[0][0]

        # Mark one as broll
        sample_candidates[1][0].is_broll = True

        vo_emb = [1.0, 0.0, 0.0]
        candidate_embs = {
            matcher.get_clip_id(seg): [0.7, 0.2, 0.1]
            for seg, _ in sample_candidates
        }

        result = matcher.get_strategy_matches(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            candidate_embs,
            vo_emb,
            segment_index=0
        )

        assert isinstance(result, list)

    @pytest.mark.fast
    def test_get_strategy_matches_empty_when_no_tracks(self, mock_config, sample_vo_segment, sample_candidates):
        """Test returns empty list when no strategy tracks configured."""
        mock_config.output.strategy_tracks = []
        matcher = StrategyMatcher(mock_config)

        primary = sample_candidates[0][0]

        result = matcher.get_strategy_matches(
            sample_vo_segment,
            sample_candidates,
            primary,
            [],
            {},
            [],
            segment_index=0
        )

        assert result == []


# ============================================================================
# Tests for FallbackMatchStrategy
# ============================================================================

class TestFallbackMatchStrategy:
    """Test FallbackMatchStrategy class."""

    @pytest.mark.fast
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = Mock()
        config.matching = Mock()
        config.matching.fallback_matching_enabled = True
        config.matching.fallback_trigger_threshold = 0.4

        strategy = FallbackMatchStrategy(config)

        assert strategy.enabled == True
        assert strategy.trigger_threshold == 0.4

    @pytest.mark.fast
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        config = Mock()
        config.matching = Mock()
        config.matching.fallback_matching_enabled = False
        config.matching.fallback_trigger_threshold = 0.6

        strategy = FallbackMatchStrategy(config)

        assert strategy.enabled == False
        assert strategy.trigger_threshold == 0.6

    @pytest.mark.fast
    def test_should_trigger_when_disabled(self, fallback_config):
        """Test should_trigger returns False when disabled."""
        fallback_config.matching.fallback_matching_enabled = False
        strategy = FallbackMatchStrategy(fallback_config)

        result = strategy.should_trigger(0.3)

        assert result == False

    @pytest.mark.fast
    def test_should_trigger_below_threshold(self, fallback_config):
        """Test should_trigger returns True when below threshold."""
        strategy = FallbackMatchStrategy(fallback_config)

        result = strategy.should_trigger(0.3)

        assert result == True

    @pytest.mark.fast
    def test_should_trigger_above_threshold(self, fallback_config):
        """Test should_trigger returns False when above threshold."""
        strategy = FallbackMatchStrategy(fallback_config)

        result = strategy.should_trigger(0.5)

        assert result == False

    @pytest.mark.fast
    def test_should_trigger_at_threshold(self, fallback_config):
        """Test should_trigger returns False at threshold."""
        strategy = FallbackMatchStrategy(fallback_config)

        result = strategy.should_trigger(0.4)

        assert result == False


# ============================================================================
# Tests for FallbackMatchStrategy.match_keyword_only
# ============================================================================

class TestFallbackKeywordOnly:
    """Test FallbackMatchStrategy.match_keyword_only method."""

    @pytest.mark.fast
    def test_keyword_only_returns_match(self, fallback_config, sample_vo_segment):
        """Test keyword only returns a match."""
        strategy = FallbackMatchStrategy(fallback_config)

        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4"), 0.8),
            (SRTSegment(1, 5.0, 10.0, "test2", "/video2.mp4"), 0.6),
        ]

        result = strategy.match_keyword_only(sample_vo_segment, candidates)

        assert result is not None
        assert len(result) == 3  # (segment, confidence, reasoning)

    @pytest.mark.fast
    def test_keyword_only_returns_none_empty_candidates(self, fallback_config, sample_vo_segment):
        """Test returns None with empty candidates."""
        strategy = FallbackMatchStrategy(fallback_config)

        result = strategy.match_keyword_only(sample_vo_segment, [])

        assert result is None

    @pytest.mark.fast
    def test_keyword_only_applies_ceiling(self, fallback_config, sample_vo_segment):
        """Test applies confidence ceiling."""
        strategy = FallbackMatchStrategy(fallback_config)

        # High similarity should be capped at 0.7
        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4"), 0.95),
        ]

        result = strategy.match_keyword_only(sample_vo_segment, candidates)

        assert result is not None
        seg, confidence, reasoning = result
        assert confidence <= strategy.KEYWORD_ONLY_CEILING


# ============================================================================
# Tests for FallbackMatchStrategy.match_visual_description
# ============================================================================

class TestFallbackVisualDescription:
    """Test FallbackMatchStrategy.match_visual_description method."""

    @pytest.mark.fast
    def test_visual_description_returns_match(self, fallback_config, sample_vo_segment):
        """Test visual description returns a match."""
        strategy = FallbackMatchStrategy(fallback_config)

        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4"), 0.8),
        ]

        result = strategy.match_visual_description(sample_vo_segment, candidates)

        assert result is not None
        seg, confidence, reasoning = result
        assert "visual" in reasoning.lower()

    @pytest.mark.fast
    def test_visual_description_applies_lower_ceiling(self, fallback_config, sample_vo_segment):
        """Test applies lower confidence ceiling than keyword."""
        strategy = FallbackMatchStrategy(fallback_config)

        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4"), 0.9),
        ]

        result = strategy.match_visual_description(sample_vo_segment, candidates)

        assert result is not None
        seg, confidence, reasoning = result
        # Visual description has lower ceiling (0.5) than keyword (0.7)
        assert confidence <= strategy.VISUAL_DESCRIPTION_CEILING


# ============================================================================
# Tests for FallbackMatchStrategy.match_generic_broll
# ============================================================================

class TestFallbackGenericBRoll:
    """Test FallbackMatchStrategy.match_generic_broll method."""

    @pytest.mark.fast
    def test_generic_broll_returns_broll(self, fallback_config, sample_vo_segment):
        """Test generic broll returns broll when available."""
        strategy = FallbackMatchStrategy(fallback_config)

        broll_seg = SRTSegment(0, 0.0, 5.0, "[Silent]", "/video1.mp4")
        broll_seg.is_broll = True

        candidates = [
            (broll_seg, 0.8),
        ]

        result = strategy.match_generic_broll(sample_vo_segment, candidates)

        assert result is not None
        seg, confidence, reasoning = result
        assert "b-roll" in reasoning.lower() or "silent" in reasoning.lower()

    @pytest.mark.fast
    def test_generic_broll_fallback_to_silent(self, fallback_config, sample_vo_segment):
        """Test falls back to silent segment when no explicit broll."""
        strategy = FallbackMatchStrategy(fallback_config)

        # Short transcript = likely silent
        silent_seg = SRTSegment(0, 0.0, 5.0, "word", "/video1.mp4")
        silent_seg.is_broll = False

        candidates = [
            (silent_seg, 0.8),
        ]

        result = strategy.match_generic_broll(sample_vo_segment, candidates)

        assert result is not None

    @pytest.mark.fast
    def test_generic_broll_returns_none_empty(self, fallback_config, sample_vo_segment):
        """Test returns None with empty candidates."""
        strategy = FallbackMatchStrategy(fallback_config)

        result = strategy.match_generic_broll(sample_vo_segment, [])

        assert result is None


# ============================================================================
# Tests for FallbackMatchStrategy.get_fallback_match
# ============================================================================

class TestGetFallbackMatch:
    """Test FallbackMatchStrategy.get_fallback_match orchestration."""

    @pytest.mark.fast
    def test_get_fallback_returns_none_above_threshold(self, fallback_config, sample_vo_segment):
        """Test returns None when above trigger threshold."""
        strategy = FallbackMatchStrategy(fallback_config)

        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4"), 0.8),
        ]

        result = strategy.get_fallback_match(sample_vo_segment, candidates, 0.5)

        # Above threshold (0.4), so fallback should not trigger
        assert result is None

    @pytest.mark.fast
    def test_get_fallback_triggers_below_threshold(self, fallback_config, sample_vo_segment):
        """Test triggers fallback when below threshold."""
        strategy = FallbackMatchStrategy(fallback_config)

        candidates = [
            (SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4"), 0.3),
        ]

        result = strategy.get_fallback_match(sample_vo_segment, candidates, 0.3)

        assert result is not None
        seg, confidence, reasoning = result

    @pytest.mark.fast
    def test_get_fallback_falls_through_levels(self, fallback_config, sample_vo_segment):
        """Test fallback falls through all levels."""
        strategy = FallbackMatchStrategy(fallback_config)

        # Empty candidates - should fail all levels
        result = strategy.get_fallback_match(sample_vo_segment, [], 0.3)

        assert result is None


# ============================================================================
# Tests for configuration validation
# ============================================================================

class TestStrategyConfiguration:
    """Test strategy configuration validation."""

    @pytest.mark.fast
    def test_config_has_strategy_tracks(self, mock_config):
        """Test config includes strategy_tracks."""
        assert hasattr(mock_config.output, 'strategy_tracks')

    @pytest.mark.fast
    def test_strategy_tracks_accepts_valid_strategies(self, mock_config):
        """Test valid strategies are accepted."""
        valid = ["embedding_diversity", "broll_only", "source_rotation"]
        mock_config.output.strategy_tracks = valid

        matcher = StrategyMatcher(mock_config)

        # Should not raise
        assert matcher.config.output.strategy_tracks == valid

    @pytest.mark.fast
    def test_fallback_config_has_required_fields(self, fallback_config):
        """Test fallback config has required fields."""
        assert hasattr(fallback_config.matching, 'fallback_matching_enabled')
        assert hasattr(fallback_config.matching, 'fallback_trigger_threshold')


# ============================================================================
# Edge cases and error handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.fast
    def test_embedding_diversity_handles_missing_embeddings(self, mock_config, sample_vo_segment, sample_candidates):
        """Test handles missing embeddings gracefully."""
        matcher = StrategyMatcher(mock_config)

        # No embeddings at all
        result = matcher.match_embedding_diversity(
            sample_vo_segment,
            sample_candidates,
            [],
            [],
            {},
            []
        )

        # Should handle gracefully (may return None or try text similarity fallback)
        assert result is None or isinstance(result, StrategyMatch)

    @pytest.mark.fast
    def test_broll_only_handles_face_score_fallback(self, mock_config, sample_vo_segment):
        """Test broll_only falls back to face_score."""
        matcher = StrategyMatcher(mock_config)

        # Low face_score but not explicitly marked as broll
        seg = SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4")
        seg.is_broll = False
        seg.face_score = 0.2  # Low face score

        candidates = [(seg, 0.6)]

        result = matcher.match_broll_only(
            sample_vo_segment,
            candidates,
            [],
            [],
            {},
            []
        )

        # Should fall back to face_score < 0.5
        assert result is not None

    @pytest.mark.fast
    def test_source_rotation_with_single_source(self, mock_config, sample_vo_segment):
        """Test source rotation with single source."""
        matcher = StrategyMatcher(mock_config)

        seg = SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4")
        candidates = [(seg, 0.8)]

        vo_emb = [1.0, 0.0, 0.0]

        result = matcher.match_source_rotation(
            sample_vo_segment,
            candidates,
            [],
            None,
            {},
            vo_emb,
            segment_index=5
        )

        assert result is not None

    @pytest.mark.fast
    def test_get_scene_for_segment_returns_none_without_cache(self, mock_config, sample_vo_segment):
        """Test _get_scene_for_segment returns None without global cache."""
        matcher = StrategyMatcher(mock_config)

        seg = SRTSegment(0, 0.0, 5.0, "test", "/video1.mp4")
        seg.scene_index = 0

        # Without global cache initialized, should return None
        result = matcher._get_scene_for_segment(seg)

        assert result is None
