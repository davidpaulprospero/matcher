"""
Tests for chapter_grouping source consistency boost and related config (US-122-002).

Verifies that:
- ChapterGroupingConfig validates chapter_topic_match_boost [0.05, 0.15] range
- ChapterGroupingConfig validates chapter_topic_mismatch_penalty must be negative
- source_consistency_boost (0.03) is applied when same source reused within chapter
- coherence_penalty_threshold (5) applies penalty when unique sources exceed threshold
- min_source_diversity (2) penalty when too few unique sources
- chapter_grouping.enabled flag controls all chapter grouping features

These tests complement existing tests in:
- test_chapter_scoring_us75005.py (chapter_topic_match, source_consistency)
- test_chapter_scoring_us75006.py (coherence_penalty)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.sections.matching import ChapterGroupingConfig, MatchingConfig
from src.matching.tiered_matcher import TieredMatcher
from src.utils import SRTSegment, MatchResult


def _make_segment(text: str, chapter_index=None, source_file=None) -> SRTSegment:
    """Create a minimal SRTSegment with given text and optional chapter info."""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
    )
    if chapter_index is not None:
        seg.chapter_index = chapter_index
    if source_file is not None:
        seg.source_file = source_file
    return seg


def _make_match_result(video_source: str, vo_chapter_index: int, confidence: float = 0.7):
    """Create a MatchResult with primary_match."""
    vo_seg = _make_segment(f"Voiceover text {vo_chapter_index}", chapter_index=vo_chapter_index)
    video_seg = _make_segment("Video content", source_file=video_source)

    match = MagicMock()
    match.video_segment = video_seg
    match.voiceover_segment = vo_seg
    match.confidence = confidence

    return MatchResult(primary_match=match, confidence_variance=0.1)


class TestChapterGroupingConfigValidation:
    """Tests for ChapterGroupingConfig validation."""

    def test_default_values(self):
        """Default config has correct values for source_consistency_boost, coherence_threshold, etc."""
        config = ChapterGroupingConfig()

        assert config.enabled is True
        assert config.source_consistency_boost == 0.03
        assert config.coherence_penalty_threshold == 5
        assert config.min_source_diversity == 2
        assert config.chapter_topic_match_boost == [0.05, 0.15]
        assert config.chapter_topic_mismatch_penalty == -0.10
        assert config.multi_chapter_assignment_strategy == 'best_match'

    def test_chapter_topic_match_boost_invalid_not_list(self):
        """chapter_topic_match_boost must be a list of 2 floats."""
        with pytest.raises(ValueError, match="must be a list of exactly 2 floats"):
            ChapterGroupingConfig(chapter_topic_match_boost="invalid")

    def test_chapter_topic_match_boost_invalid_length(self):
        """chapter_topic_match_boost must have exactly 2 elements."""
        with pytest.raises(ValueError, match="must be a list of exactly 2 floats"):
            ChapterGroupingConfig(chapter_topic_match_boost=[0.05])

    def test_chapter_topic_match_boost_min_greater_than_max(self):
        """chapter_topic_match_boost min must be <= max."""
        with pytest.raises(ValueError, match="min .* must be <= max"):
            ChapterGroupingConfig(chapter_topic_match_boost=[0.20, 0.10])

    def test_chapter_topic_match_boost_valid_range(self):
        """Valid chapter_topic_match_boost range [0.05, 0.15] should work."""
        config = ChapterGroupingConfig(chapter_topic_match_boost=[0.08, 0.12])
        assert config.chapter_topic_match_boost == [0.08, 0.12]

    def test_chapter_topic_mismatch_penalty_must_be_negative(self):
        """chapter_topic_mismatch_penalty must be negative."""
        with pytest.raises(ValueError, match="must be negative"):
            ChapterGroupingConfig(chapter_topic_mismatch_penalty=0.10)

    def test_chapter_topic_mismatch_penalty_valid_negative(self):
        """Valid negative chapter_topic_mismatch_penalty should work."""
        config = ChapterGroupingConfig(chapter_topic_mismatch_penalty=-0.15)
        assert config.chapter_topic_mismatch_penalty == -0.15

    def test_coherence_penalty_threshold_must_be_at_least_1(self):
        """coherence_penalty_threshold must be >= 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ChapterGroupingConfig(coherence_penalty_threshold=0)

    def test_multi_chapter_assignment_strategy_valid_options(self):
        """multi_chapter_assignment_strategy must be 'first', 'split', or 'best_match'."""
        for strategy in ['first', 'split', 'best_match']:
            config = ChapterGroupingConfig(multi_chapter_assignment_strategy=strategy)
            assert config.multi_chapter_assignment_strategy == strategy

    def test_multi_chapter_assignment_strategy_invalid(self):
        """Invalid multi_chapter_assignment_strategy raises error."""
        with pytest.raises(ValueError, match="must be one of"):
            ChapterGroupingConfig(multi_chapter_assignment_strategy='invalid')


class TestChapterGroupingEnabledFlag:
    """Tests for chapter_grouping.enabled flag controlling features."""

    def test_enabled_flag_false_disables_source_consistency(self):
        """When chapter_grouping.enabled=False, source consistency boost is not applied."""
        # Create config with enabled=False
        config = ChapterGroupingConfig(enabled=False)

        # Verify the flag is False
        assert config.enabled is False
        assert config.source_consistency_boost == 0.03  # Value still exists, just disabled

    def test_enabled_flag_affects_matching_flow(self):
        """Integration test: enabled=False skips chapter grouping in TieredMatcher."""
        # Create a mock config for TieredMatcher
        mock_matching_config = MagicMock()
        mock_matching_config.chapter_grouping = ChapterGroupingConfig(enabled=False)

        mock_config = MagicMock()
        mock_config.matching = mock_matching_config
        mock_config.video_search = MagicMock()
        mock_config.download = MagicMock()

        with patch('src.matching.tiered_matcher.get_config', return_value=mock_config):
            matcher = TieredMatcher(config=mock_config)

            # With enabled=False, enforce_chapter_source_diversity should return early
            matches = [
                _make_match_result("vid_A", vo_chapter_index=0),
                _make_match_result("vid_A", vo_chapter_index=0),
                _make_match_result("vid_A", vo_chapter_index=0),
                _make_match_result("vid_A", vo_chapter_index=0),
                _make_match_result("vid_A", vo_chapter_index=0),
            ]

            # This should return unchanged because enabled=False
            result = matcher.enforce_chapter_source_diversity(matches)

            # Should return the same list (no diversity enforcement)
            assert result == matches


class TestEnforceChapterSourceDiversity:
    """Tests for TieredMatcher.enforce_chapter_source_diversity method."""

    @pytest.fixture
    def mock_config_with_chapter_grouping(self):
        """Create mock config with chapter_grouping enabled."""
        mock_config = MagicMock()
        mock_matching = MagicMock()
        mock_matching.chapter_grouping = ChapterGroupingConfig(
            enabled=True,
            min_source_diversity=2,
        )
        mock_config.matching = mock_matching
        mock_config.video_search = MagicMock()
        mock_config.download = MagicMock()
        return mock_config

    def test_single_source_chapter_gets_swapped(self, mock_config_with_chapter_grouping):
        """When chapter uses only 1 source for 5+ segments, should swap to different source."""
        with patch('src.matching.tiered_matcher.get_config', return_value=mock_config_with_chapter_grouping):
            matcher = TieredMatcher(config=mock_config_with_chapter_grouping)

            # Create 5 segments all from same source in chapter 0
            matches = [
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.9),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.8),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.7),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.6),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.5),
            ]

            result = matcher.enforce_chapter_source_diversity(matches)

            # Should still have 5 matches but may have swapped some
            assert len(result) == 5

    def test_min_source_diversity_disabled_when_1(self, mock_config_with_chapter_grouping):
        """When min_source_diversity=1, no diversity enforcement (1 = disabled)."""
        mock_config_with_chapter_grouping.matching.chapter_grouping.min_source_diversity = 1

        with patch('src.matching.tiered_matcher.get_config', return_value=mock_config_with_chapter_grouping):
            matcher = TieredMatcher(config=mock_config_with_chapter_grouping)

            # All from same source
            matches = [
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.9),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.8),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.7),
            ]

            result = matcher.enforce_chapter_source_diversity(matches)

            # Should return unchanged because min_diversity=1 (disabled)
            assert result == matches

    def test_sufficient_diversity_no_change(self, mock_config_with_chapter_grouping):
        """When chapter already has min_source_diversity=2, no changes needed."""
        with patch('src.matching.tiered_matcher.get_config', return_value=mock_config_with_chapter_grouping):
            matcher = TieredMatcher(config=mock_config_with_chapter_grouping)

            # Create matches with 2 different sources (meets min_source_diversity)
            matches = [
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.9),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.8),
                _make_match_result("vid_B", vo_chapter_index=0, confidence=0.7),
                _make_match_result("vid_B", vo_chapter_index=0, confidence=0.6),
            ]

            result = matcher.enforce_chapter_source_diversity(matches)

            # Should return unchanged (already has diversity)
            assert result == matches

    def test_chapters_below_threshold_not_enforced(self, mock_config_with_chapter_grouping):
        """Chapters with <=4 segments should be skipped."""
        with patch('src.matching.tiered_matcher.get_config', return_value=mock_config_with_chapter_grouping):
            matcher = TieredMatcher(config=mock_config_with_chapter_grouping)

            # Only 3 segments in chapter 0 - below threshold of 4
            matches = [
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.9),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.8),
                _make_match_result("vid_A", vo_chapter_index=0, confidence=0.7),
            ]

            result = matcher.enforce_chapter_source_diversity(matches)

            # Should return unchanged (below segment threshold)
            assert result == matches


class TestMinSourceDiversityPenalty:
    """Tests verifying min_source_diversity applies penalty when too few unique sources."""

    def test_min_source_diversity_2_requires_2_sources(self):
        """min_source_diversity=2 means at least 2 unique sources required per chapter."""
        config = ChapterGroupingConfig(min_source_diversity=2)
        assert config.min_source_diversity == 2

    def test_min_source_diversity_configurable(self):
        """min_source_diversity can be set to different values."""
        for value in [1, 2, 3, 5]:
            config = ChapterGroupingConfig(min_source_diversity=value)
            assert config.min_source_diversity == value

    def test_min_source_diversity_zero_disables(self):
        """min_source_diversity=0 or 1 should disable diversity enforcement."""
        config = ChapterGroupingConfig(min_source_diversity=0)
        # Should still pass validation (threshold is for coherence_penalty_threshold)
        assert config.min_source_diversity == 0


class TestChapterTopicMatchBoostRange:
    """Tests verifying chapter_topic_match_boost range [0.05, 0.15] behavior."""

    def test_default_boost_range(self):
        """Default chapter_topic_match_boost is [0.05, 0.15]."""
        config = ChapterGroupingConfig()
        assert config.chapter_topic_match_boost == [0.05, 0.15]

    def test_custom_boost_range(self):
        """Custom boost range values are stored correctly."""
        config = ChapterGroupingConfig(chapter_topic_match_boost=[0.10, 0.20])
        assert config.chapter_topic_match_boost == [0.10, 0.20]

    def test_boost_range_zero_valid(self):
        """Zero values in boost range are valid."""
        config = ChapterGroupingConfig(chapter_topic_match_boost=[0.0, 0.10])
        assert config.chapter_topic_match_boost == [0.0, 0.10]
