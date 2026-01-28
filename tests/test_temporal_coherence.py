"""Tests for temporal coherence scoring for adjacent segments.

Tests compute_temporal_coherence() function in src/matching/scoring.py.
US-006: Add temporal coherence scoring for adjacent segments.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import compute_temporal_coherence, _is_jarring_context_switch
from src.utils import SRTSegment


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with default temporal coherence settings."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.temporal_coherence_enabled = True
    config.matching.temporal_coherence_same_source_boost = 0.05
    config.matching.temporal_coherence_context_switch_penalty = 0.05
    return config


@pytest.fixture
def mock_config_disabled():
    """Create mock config with temporal coherence disabled."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.temporal_coherence_enabled = False
    config.matching.temporal_coherence_same_source_boost = 0.05
    config.matching.temporal_coherence_context_switch_penalty = 0.05
    return config


@pytest.fixture
def mock_segment_factory():
    """Factory for creating mock SRTSegment objects."""
    def _create(source_file="video1.mp4", topics=None, keywords=None,
                start_time=0.0, end_time=10.0, text="Sample text"):
        seg = SRTSegment(
            index=1,
            start_time=start_time,
            end_time=end_time,
            text=text,
            source_file=source_file
        )
        if topics:
            seg.topics = topics
        if keywords:
            seg.keywords = keywords
        return seg
    return _create


# ==============================================================================
# Test Function Existence and Signature
# ==============================================================================

class TestComputeTemporalCoherenceFunction:
    """Test that compute_temporal_coherence function exists with correct signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """compute_temporal_coherence function should exist in scoring module."""
        from src.matching.scoring import compute_temporal_coherence
        assert callable(compute_temporal_coherence)

    @pytest.mark.fast
    def test_function_returns_tuple(self, mock_config, mock_segment_factory):
        """Function should return tuple of (confidence, reason)."""
        segment = mock_segment_factory(source_file="video1.mp4")
        result = compute_temporal_coherence(0.8, segment, None, None, mock_config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.fast
    def test_function_returns_float_and_string(self, mock_config, mock_segment_factory):
        """Function should return (float, str)."""
        segment = mock_segment_factory(source_file="video1.mp4")
        confidence, reason = compute_temporal_coherence(0.8, segment, None, None, mock_config)
        assert isinstance(confidence, float)
        assert isinstance(reason, str)

    @pytest.mark.fast
    def test_accepts_none_for_previous_match(self, mock_config, mock_segment_factory):
        """Function should handle None for previous_match (first segment)."""
        segment = mock_segment_factory(source_file="video1.mp4")
        confidence, reason = compute_temporal_coherence(0.8, segment, None, None, mock_config)
        assert confidence == 0.8  # No adjustment when no adjacent matches

    @pytest.mark.fast
    def test_accepts_none_for_next_match(self, mock_config, mock_segment_factory):
        """Function should handle None for next_match (last segment)."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video2.mp4")
        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)
        assert isinstance(confidence, float)


# ==============================================================================
# Test Config Option
# ==============================================================================

class TestTemporalCoherenceConfig:
    """Test temporal_coherence_enabled config option."""

    @pytest.mark.fast
    def test_config_enabled_by_default(self):
        """temporal_coherence_enabled should default to True."""
        from src.config.sections.matching import MatchingConfig
        mc = MatchingConfig()
        assert mc.temporal_coherence_enabled is True

    @pytest.mark.fast
    def test_config_same_source_boost_default(self):
        """temporal_coherence_same_source_boost should default to 0.05."""
        from src.config.sections.matching import MatchingConfig
        mc = MatchingConfig()
        assert mc.temporal_coherence_same_source_boost == 0.05

    @pytest.mark.fast
    def test_config_context_switch_penalty_default(self):
        """temporal_coherence_context_switch_penalty should default to 0.05."""
        from src.config.sections.matching import MatchingConfig
        mc = MatchingConfig()
        assert mc.temporal_coherence_context_switch_penalty == 0.05

    @pytest.mark.fast
    def test_disabled_config_returns_original_confidence(self, mock_config_disabled, mock_segment_factory):
        """When disabled, function should return original confidence unchanged."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")  # Same source

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config_disabled)

        assert confidence == 0.8
        assert reason == ""


# ==============================================================================
# Test Same-Source Boost
# ==============================================================================

class TestSameSourceBoost:
    """Test +0.05 boost for clips from same source video as adjacent."""

    @pytest.mark.fast
    def test_same_source_as_previous_applies_boost(self, mock_config, mock_segment_factory):
        """Clip from same source as previous should get +0.05 boost."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")  # Same source

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert confidence == pytest.approx(0.85, abs=0.001)
        assert "same source as prev" in reason

    @pytest.mark.fast
    def test_same_source_as_next_applies_boost(self, mock_config, mock_segment_factory):
        """Clip from same source as next should get +0.05 boost."""
        segment = mock_segment_factory(source_file="video1.mp4")
        next_match = mock_segment_factory(source_file="video1.mp4")  # Same source

        confidence, reason = compute_temporal_coherence(0.8, segment, None, next_match, mock_config)

        assert confidence == pytest.approx(0.85, abs=0.001)
        assert "same source as next" in reason

    @pytest.mark.fast
    def test_same_source_both_directions_double_boost(self, mock_config, mock_segment_factory):
        """Same source on both sides should apply boost twice (+0.10 total)."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")
        next_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, next_match, mock_config)

        assert confidence == pytest.approx(0.90, abs=0.001)
        assert "same source as prev" in reason
        assert "same source as next" in reason

    @pytest.mark.fast
    def test_boost_capped_at_1_0(self, mock_config, mock_segment_factory):
        """Confidence should never exceed 1.0."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")
        next_match = mock_segment_factory(source_file="video1.mp4")

        # Start with 0.98, would exceed 1.0 with +0.10 boost
        confidence, reason = compute_temporal_coherence(0.98, segment, prev_match, next_match, mock_config)

        assert confidence == 1.0

    @pytest.mark.fast
    def test_different_source_no_boost(self, mock_config, mock_segment_factory):
        """Different sources should not get same-source boost."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video2.mp4")  # Different source

        # Should not apply boost, but might apply penalty if topics don't match
        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert "same source as prev" not in reason


# ==============================================================================
# Test Context Switch Penalty
# ==============================================================================

class TestContextSwitchPenalty:
    """Test -0.05 penalty for jarring context switches."""

    @pytest.mark.fast
    def test_jarring_switch_applies_penalty(self, mock_config, mock_segment_factory):
        """Jarring context switch should apply -0.05 penalty."""
        segment = mock_segment_factory(
            source_file="video1.mp4",
            topics=["cooking", "kitchen"]
        )
        prev_match = mock_segment_factory(
            source_file="video2.mp4",
            topics=["astronomy", "stars"]  # Completely different
        )

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert confidence == pytest.approx(0.75, abs=0.001)
        assert "context switch" in reason

    @pytest.mark.fast
    def test_no_penalty_when_topics_overlap(self, mock_config, mock_segment_factory):
        """No penalty when topics have some overlap."""
        segment = mock_segment_factory(
            source_file="video1.mp4",
            topics=["cooking", "kitchen", "food"]
        )
        prev_match = mock_segment_factory(
            source_file="video2.mp4",
            topics=["food", "recipes"]  # "food" overlaps
        )

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert "context switch" not in reason

    @pytest.mark.fast
    def test_no_penalty_when_no_topics(self, mock_config, mock_segment_factory):
        """No penalty when segments have no topics."""
        segment = mock_segment_factory(source_file="video1.mp4")  # No topics
        prev_match = mock_segment_factory(source_file="video2.mp4")  # No topics

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        # No penalty should be applied when there's no topic info
        assert "context switch" not in reason

    @pytest.mark.fast
    def test_penalty_not_below_zero(self, mock_config, mock_segment_factory):
        """Confidence should never go below 0.0."""
        segment = mock_segment_factory(
            source_file="video1.mp4",
            topics=["cooking"]
        )
        prev_match = mock_segment_factory(
            source_file="video2.mp4",
            topics=["astronomy"]
        )
        next_match = mock_segment_factory(
            source_file="video3.mp4",
            topics=["sports"]
        )

        # Start with 0.05, would go negative with double penalty
        confidence, reason = compute_temporal_coherence(0.05, segment, prev_match, next_match, mock_config)

        assert confidence == 0.0

    @pytest.mark.fast
    def test_keywords_used_for_overlap_check(self, mock_config, mock_segment_factory):
        """Keywords should be used in addition to topics for overlap check."""
        segment = mock_segment_factory(
            source_file="video1.mp4",
            keywords=["sunset", "beach"]
        )
        prev_match = mock_segment_factory(
            source_file="video2.mp4",
            keywords=["beach", "ocean"]  # "beach" overlaps
        )

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        # Should NOT get penalty because keywords overlap
        assert "context switch" not in reason


# ==============================================================================
# Test Helper Function
# ==============================================================================

class TestIsJarringContextSwitch:
    """Test _is_jarring_context_switch helper function."""

    @pytest.mark.fast
    def test_no_overlap_is_jarring(self, mock_segment_factory):
        """No topic/keyword overlap should be jarring."""
        current = mock_segment_factory(topics=["cooking", "food"])
        adjacent = mock_segment_factory(topics=["astronomy", "space"])

        result = _is_jarring_context_switch(current, adjacent)

        assert result is True

    @pytest.mark.fast
    def test_overlap_is_not_jarring(self, mock_segment_factory):
        """Topic/keyword overlap should NOT be jarring."""
        current = mock_segment_factory(topics=["cooking", "food"])
        adjacent = mock_segment_factory(topics=["food", "nutrition"])

        result = _is_jarring_context_switch(current, adjacent)

        assert result is False

    @pytest.mark.fast
    def test_empty_topics_not_jarring(self, mock_segment_factory):
        """Empty topics should not be considered jarring."""
        current = mock_segment_factory(topics=[])
        adjacent = mock_segment_factory(topics=["astronomy"])

        result = _is_jarring_context_switch(current, adjacent)

        assert result is False

    @pytest.mark.fast
    def test_empty_adjacent_topics_not_jarring(self, mock_segment_factory):
        """Empty adjacent topics should not be considered jarring (AC4)."""
        current = mock_segment_factory(topics=["cooking", "food"])
        adjacent = mock_segment_factory(topics=[])

        result = _is_jarring_context_switch(current, adjacent)

        assert result is False

    @pytest.mark.fast
    def test_case_insensitive_comparison(self, mock_segment_factory):
        """Topic comparison should be case-insensitive."""
        current = mock_segment_factory(topics=["Cooking", "FOOD"])
        adjacent = mock_segment_factory(topics=["food", "nutrition"])

        result = _is_jarring_context_switch(current, adjacent)

        assert result is False  # "FOOD" and "food" should match

    @pytest.mark.fast
    def test_keywords_and_topics_combined(self, mock_segment_factory):
        """Both topics and keywords should be considered."""
        current = mock_segment_factory(
            topics=["cooking"],
            keywords=["sunset"]
        )
        adjacent = mock_segment_factory(
            topics=["photography"],
            keywords=["sunset"]  # Keywords overlap
        )

        result = _is_jarring_context_switch(current, adjacent)

        assert result is False  # "sunset" keyword matches


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.fast
    def test_no_source_file_no_change(self, mock_config):
        """Segment without source_file should return original confidence."""
        segment = MagicMock(spec=['text', 'start_time', 'end_time'])
        segment.source_file = None
        prev_match = MagicMock(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert confidence == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_first_segment_no_previous(self, mock_config, mock_segment_factory):
        """First segment (no previous) should work without error."""
        segment = mock_segment_factory(source_file="video1.mp4")
        next_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, None, next_match, mock_config)

        assert confidence == pytest.approx(0.85, abs=0.001)

    @pytest.mark.fast
    def test_last_segment_no_next(self, mock_config, mock_segment_factory):
        """Last segment (no next) should work without error."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert confidence == pytest.approx(0.85, abs=0.001)

    @pytest.mark.fast
    def test_isolated_segment_no_change(self, mock_config, mock_segment_factory):
        """Isolated segment (no prev, no next) should return original."""
        segment = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, None, None, mock_config)

        assert confidence == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_custom_boost_value(self, mock_segment_factory):
        """Custom boost values should be respected."""
        config = MagicMock()
        config.matching = MagicMock()
        config.matching.temporal_coherence_enabled = True
        config.matching.temporal_coherence_same_source_boost = 0.10  # Custom: +10%
        config.matching.temporal_coherence_context_switch_penalty = 0.03  # Custom: -3%

        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, config)

        assert confidence == pytest.approx(0.90, abs=0.001)  # +0.10 boost

    @pytest.mark.fast
    def test_getattr_fallback_for_missing_config(self, mock_segment_factory):
        """Function should handle missing config attributes gracefully."""
        config = MagicMock()
        config.matching = MagicMock(spec=[])  # Empty spec - no attributes

        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        # Should use defaults via getattr
        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, config)

        # Should still work with defaults
        assert isinstance(confidence, float)


# ==============================================================================
# Test Mixed Boost and Penalty
# ==============================================================================

class TestMixedBoostAndPenalty:
    """Test scenarios with both boost and penalty."""

    @pytest.mark.fast
    def test_boost_from_prev_penalty_from_next(self, mock_config, mock_segment_factory):
        """Same source prev, jarring next should net to 0 adjustment."""
        segment = mock_segment_factory(
            source_file="video1.mp4",
            topics=["cooking"]
        )
        prev_match = mock_segment_factory(
            source_file="video1.mp4"  # Same source
        )
        next_match = mock_segment_factory(
            source_file="video2.mp4",
            topics=["astronomy"]  # Jarring context
        )

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, next_match, mock_config)

        # +0.05 from same source, -0.05 from jarring = 0.80
        assert confidence == pytest.approx(0.80, abs=0.001)

    @pytest.mark.fast
    def test_penalty_from_prev_boost_from_next(self, mock_config, mock_segment_factory):
        """Jarring prev, same source next should net to 0 adjustment."""
        segment = mock_segment_factory(
            source_file="video1.mp4",
            topics=["cooking"]
        )
        prev_match = mock_segment_factory(
            source_file="video2.mp4",
            topics=["astronomy"]  # Jarring context
        )
        next_match = mock_segment_factory(
            source_file="video1.mp4"  # Same source
        )

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, next_match, mock_config)

        # -0.05 from jarring, +0.05 from same source = 0.80
        assert confidence == pytest.approx(0.80, abs=0.001)


# ==============================================================================
# Test Reason String Format
# ==============================================================================

class TestReasonStringFormat:
    """Test the format of reason strings."""

    @pytest.mark.fast
    def test_reason_includes_temporal_coherence_prefix(self, mock_config, mock_segment_factory):
        """Reason should start with 'temporal coherence:'."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert reason.startswith("temporal coherence:")

    @pytest.mark.fast
    def test_reason_includes_adjustment_value(self, mock_config, mock_segment_factory):
        """Reason should include the adjustment value."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert "+0.05" in reason or "0.05" in reason

    @pytest.mark.fast
    def test_multiple_adjustments_semicolon_separated(self, mock_config, mock_segment_factory):
        """Multiple adjustments should be semicolon-separated."""
        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")
        next_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, next_match, mock_config)

        assert ";" in reason


# ==============================================================================
# Test Integration with Real MatchingConfig
# ==============================================================================

class TestRealConfigIntegration:
    """Test with real MatchingConfig dataclass."""

    @pytest.mark.fast
    def test_with_real_matching_config(self, mock_segment_factory):
        """Function should work with real MatchingConfig."""
        from src.config.sections.matching import MatchingConfig

        config = MagicMock()
        config.matching = MatchingConfig()

        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, config)

        assert confidence == pytest.approx(0.85, abs=0.001)

    @pytest.mark.fast
    def test_config_values_used_correctly(self, mock_segment_factory):
        """Config values should be used in calculations."""
        from src.config.sections.matching import MatchingConfig

        mc = MatchingConfig()
        mc.temporal_coherence_enabled = True
        mc.temporal_coherence_same_source_boost = 0.07  # Custom

        config = MagicMock()
        config.matching = mc

        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        confidence, reason = compute_temporal_coherence(0.8, segment, prev_match, None, config)

        assert confidence == pytest.approx(0.87, abs=0.001)


# ==============================================================================
# Test Logging
# ==============================================================================

class TestLogging:
    """Test debug logging behavior."""

    @pytest.mark.fast
    def test_logs_adjustment_at_debug_level(self, mock_config, mock_segment_factory, caplog):
        """Should log adjustment at DEBUG level."""
        import logging

        segment = mock_segment_factory(source_file="video1.mp4")
        prev_match = mock_segment_factory(source_file="video1.mp4")

        with caplog.at_level(logging.DEBUG, logger="src.matching.scoring"):
            compute_temporal_coherence(0.8, segment, prev_match, None, mock_config)

        assert any("Temporal coherence adjustment" in r.message for r in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
