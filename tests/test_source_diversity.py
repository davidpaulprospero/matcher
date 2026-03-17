"""
Tests for source diversity scoring in secondary matches (V4-V6 tracks).

US-007: Verify secondary matches prefer different source files.

Note: The original calculate_source_diversity_score() function was refactored
into inline scoring within get_secondary_matches_diversity(). These tests
verify the behavior of the current implementation.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.strategies import StrategyMatcher
from src.matching.types import StrategyMatch


pytestmark = pytest.mark.unit


class TestStrategyMatcherExists:
    """Test that StrategyMatcher class exists with expected methods."""

    @pytest.mark.fast
    def test_class_exists(self):
        """StrategyMatcher is importable from strategies module."""
        from src.matching.strategies import StrategyMatcher
        assert StrategyMatcher is not None

    @pytest.mark.fast
    def test_get_secondary_matches_diversity_exists(self):
        """get_secondary_matches_diversity method exists."""
        assert hasattr(StrategyMatcher, 'get_secondary_matches_diversity')
        assert callable(getattr(StrategyMatcher, 'get_secondary_matches_diversity'))


class TestStrategyMatchType:
    """Test StrategyMatch dataclass from types module."""

    @pytest.mark.fast
    def test_strategy_match_importable(self):
        """StrategyMatch is importable from matching.types."""
        from src.matching.types import StrategyMatch
        assert StrategyMatch is not None

    @pytest.mark.fast
    def test_strategy_match_has_confidence(self):
        """StrategyMatch has confidence field."""
        mock_segment = MagicMock()
        match = StrategyMatch(
            video_segment=mock_segment,
            video_scene=None,
            confidence=0.8,
            reasoning="test",
            strategy="secondary_diversity"
        )
        assert hasattr(match, 'confidence')
        assert match.confidence == 0.8

    @pytest.mark.fast
    def test_strategy_match_has_strategy_field(self):
        """StrategyMatch has strategy field."""
        mock_segment = MagicMock()
        match = StrategyMatch(
            video_segment=mock_segment,
            video_scene=None,
            confidence=0.8,
            reasoning="test",
            strategy="secondary_diversity"
        )
        assert hasattr(match, 'strategy')
        assert match.strategy == "secondary_diversity"


class TestSecondaryMatchesDiversitySignature:
    """Test get_secondary_matches_diversity method signature."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for StrategyMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.global_cache = None
        return config

    @pytest.mark.fast
    def test_method_accepts_required_args(self, mock_config):
        """Method accepts all required arguments."""
        matcher = StrategyMatcher(mock_config)

        # Create minimal mocks
        vo_segment = MagicMock()
        vo_segment.source_file = "vo.mp3"
        vo_segment.start_time = 0.0
        vo_segment.end_time = 5.0

        primary = MagicMock()
        primary.source_file = "video1.mp4"
        primary.start_time = 0.0
        primary.end_time = 5.0

        # Method should accept these args without error
        result = matcher.get_secondary_matches_diversity(
            vo_segment=vo_segment,
            all_candidates=[],
            primary_match=primary,
            secondary_matches=[],
            candidate_embeddings={},
            vo_embedding=[0.1] * 384,
            num_matches=3
        )
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_returns_list_of_strategy_match(self, mock_config):
        """Method returns list of StrategyMatch objects."""
        matcher = StrategyMatcher(mock_config)

        vo_segment = MagicMock()
        vo_segment.source_file = "vo.mp3"
        vo_segment.start_time = 0.0
        vo_segment.end_time = 5.0

        primary = MagicMock()
        primary.source_file = "video1.mp4"
        primary.start_time = 0.0
        primary.end_time = 5.0

        candidate = MagicMock()
        candidate.source_file = "video2.mp4"
        candidate.start_time = 0.0
        candidate.end_time = 5.0

        candidate_id = f"{candidate.source_file}:{candidate.start_time}-{candidate.end_time}"

        result = matcher.get_secondary_matches_diversity(
            vo_segment=vo_segment,
            all_candidates=[(candidate, 0.8)],
            primary_match=primary,
            secondary_matches=[],
            candidate_embeddings={
                candidate_id: [0.1] * 384,
                f"{primary.source_file}:{primary.start_time}-{primary.end_time}": [0.2] * 384
            },
            vo_embedding=[0.15] * 384,
            num_matches=3
        )

        assert isinstance(result, list)
        if result:
            assert isinstance(result[0], StrategyMatch)


class TestDiversityBehavior:
    """Test that secondary matches prefer different sources."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for StrategyMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.global_cache = None
        return config

    @pytest.mark.fast
    def test_skips_same_source_as_primary(self, mock_config):
        """Candidates from same source as primary are skipped."""
        matcher = StrategyMatcher(mock_config)

        vo_segment = MagicMock()
        vo_segment.source_file = "vo.mp3"
        vo_segment.start_time = 0.0
        vo_segment.end_time = 5.0

        primary = MagicMock()
        primary.source_file = "video1.mp4"
        primary.start_time = 0.0
        primary.end_time = 5.0

        # Same source as primary - should be skipped
        same_source = MagicMock()
        same_source.source_file = "video1.mp4"
        same_source.start_time = 10.0
        same_source.end_time = 15.0

        # Different source - should be selected
        diff_source = MagicMock()
        diff_source.source_file = "video2.mp4"
        diff_source.start_time = 0.0
        diff_source.end_time = 5.0

        same_id = f"{same_source.source_file}:{same_source.start_time}-{same_source.end_time}"
        diff_id = f"{diff_source.source_file}:{diff_source.start_time}-{diff_source.end_time}"
        primary_id = f"{primary.source_file}:{primary.start_time}-{primary.end_time}"

        result = matcher.get_secondary_matches_diversity(
            vo_segment=vo_segment,
            all_candidates=[(same_source, 0.9), (diff_source, 0.7)],
            primary_match=primary,
            secondary_matches=[],
            candidate_embeddings={
                same_id: [0.1] * 384,
                diff_id: [0.2] * 384,
                primary_id: [0.3] * 384
            },
            vo_embedding=[0.15] * 384,
            num_matches=1
        )

        # Should get the different source, not same source
        assert len(result) == 1
        assert result[0].video_segment.source_file == "video2.mp4"

    @pytest.mark.fast
    def test_returns_multiple_diverse_matches(self, mock_config):
        """Returns up to num_matches diverse matches."""
        matcher = StrategyMatcher(mock_config)

        vo_segment = MagicMock()
        vo_segment.source_file = "vo.mp3"
        vo_segment.start_time = 0.0
        vo_segment.end_time = 5.0

        primary = MagicMock()
        primary.source_file = "video1.mp4"
        primary.start_time = 0.0
        primary.end_time = 5.0

        # Create 3 candidates from different sources
        candidates = []
        embeddings = {}
        for i in range(2, 5):
            c = MagicMock()
            c.source_file = f"video{i}.mp4"
            c.start_time = 0.0
            c.end_time = 5.0
            candidates.append((c, 0.8 - i * 0.1))
            cid = f"{c.source_file}:{c.start_time}-{c.end_time}"
            embeddings[cid] = [0.1 * i] * 384

        primary_id = f"{primary.source_file}:{primary.start_time}-{primary.end_time}"
        embeddings[primary_id] = [0.3] * 384

        result = matcher.get_secondary_matches_diversity(
            vo_segment=vo_segment,
            all_candidates=candidates,
            primary_match=primary,
            secondary_matches=[],
            candidate_embeddings=embeddings,
            vo_embedding=[0.15] * 384,
            num_matches=3
        )

        assert len(result) == 3
        # Each match should be from different source
        sources = [m.video_segment.source_file for m in result]
        assert len(set(sources)) == 3

    @pytest.mark.fast
    def test_empty_candidates_returns_empty(self, mock_config):
        """Empty candidate list returns empty results."""
        matcher = StrategyMatcher(mock_config)

        vo_segment = MagicMock()
        vo_segment.source_file = "vo.mp3"
        vo_segment.start_time = 0.0
        vo_segment.end_time = 5.0

        primary = MagicMock()
        primary.source_file = "video1.mp4"
        primary.start_time = 0.0
        primary.end_time = 5.0

        result = matcher.get_secondary_matches_diversity(
            vo_segment=vo_segment,
            all_candidates=[],
            primary_match=primary,
            secondary_matches=[],
            candidate_embeddings={},
            vo_embedding=[0.15] * 384,
            num_matches=3
        )

        assert result == []


class TestClipIdGeneration:
    """Test get_clip_id method for deduplication."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for StrategyMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.global_cache = None
        return config

    @pytest.mark.fast
    def test_clip_id_format(self, mock_config):
        """Clip ID follows expected format."""
        matcher = StrategyMatcher(mock_config)

        segment = MagicMock()
        segment.source_file = "video.mp4"
        segment.start_time = 10.5
        segment.end_time = 15.3

        clip_id = matcher.get_clip_id(segment)
        assert clip_id == "video.mp4:10.5-15.3"

    @pytest.mark.fast
    def test_clip_id_cached(self, mock_config):
        """Clip ID is cached for same segment."""
        matcher = StrategyMatcher(mock_config)

        segment = MagicMock()
        segment.source_file = "video.mp4"
        segment.start_time = 10.5
        segment.end_time = 15.3

        id1 = matcher.get_clip_id(segment)
        id2 = matcher.get_clip_id(segment)

        assert id1 == id2
        # Should use cached value
        assert len(matcher._clip_id_cache) == 1
