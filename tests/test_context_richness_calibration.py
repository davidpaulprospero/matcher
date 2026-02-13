"""Tests for US-95-010: Context richness calibration for confidence scores."""

import pytest
from src.matching.scoring import apply_context_richness_calibration


class TestContextRichnessCalibration:
    """Test suite for context richness calibration feature."""

    def test_rich_context_boost_all_signals(self):
        """Test that all 4 signals present produces max boost."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial: Building Web Apps",
            video_description="Learn how to build web applications with Python",
            video_tags=["python", "web", "tutorial", "programming"],
            video_chapter="Introduction",
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # All 4 signals present (rich context) -> should boost
        assert adjusted > confidence
        assert "signals=4/4" in reason
        assert "rich" in reason

    def test_rich_context_boost_three_signals(self):
        """Test that 3 signals present produces boost."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=["python", "programming"],
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 3 signals present -> should boost (>= 0.75 ratio)
        assert adjusted > confidence
        assert "signals=3/4" in reason
        assert "rich" in reason

    def test_moderate_context_no_adjustment(self):
        """Test that 2 signals present produces no adjustment."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",  # Has title
            video_description=None,  # No description
            video_tags=["python"],  # Has tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 2 signals -> moderate context, no adjustment
        assert adjusted == confidence
        assert "signals=2/4" in reason
        assert "moderate" in reason

    def test_sparse_context_penalty_one_signal(self):
        """Test that 1 signal present produces penalty."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",  # Has title
            video_description=None,  # No description
            video_tags=None,  # No tags
            video_chapter=None,  # No chapter
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 1 signal -> sparse context, apply penalty
        assert adjusted < confidence
        assert "signals=1/4" in reason
        assert "sparse" in reason

    def test_sparse_context_penalty_no_signals(self):
        """Test that 0 signals produces max penalty."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # 0 signals -> sparse context, max penalty
        assert adjusted < confidence
        assert "signals=0/4" in reason
        assert "sparse" in reason

    def test_disabled_calibration(self):
        """Test that disabled calibration returns original confidence."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=["python"],
            video_chapter="Intro",
            enabled=False,  # Disabled
            boost_max=0.08,
            penalty_max=0.05,
        )

        assert adjusted == confidence
        assert reason == ""

    def test_empty_string_treated_as_no_signal(self):
        """Test that empty strings are treated as no signal."""
        confidence = 0.80

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="",  # Empty string - no signal
            video_description="  ",  # Whitespace only - no signal
            video_tags=[],  # Empty list - no signal
            video_chapter="\t",  # Whitespace only - no signal
            enabled=True,
            boost_max=0.08,
            penalty_max=0.05,
        )

        # All empty/whitespace -> should apply max penalty
        assert adjusted < confidence
        assert "signals=0/4" in reason

    def test_confidence_clamped_to_max_1(self):
        """Test that boosted confidence is clamped to 1.0."""
        confidence = 0.95  # High confidence

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Python Tutorial",
            video_description="Learn Python",
            video_tags=["python"],
            video_chapter="Intro",
            enabled=True,
            boost_max=0.10,  # Large boost
            penalty_max=0.05,
        )

        # Should be clamped to 1.0
        assert adjusted <= 1.0

    def test_confidence_clamped_to_min_0(self):
        """Test that penalized confidence is clamped to 0.0."""
        confidence = 0.02  # Low confidence

        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title=None,
            video_description=None,
            video_tags=None,
            video_chapter=None,
            enabled=True,
            boost_max=0.08,
            penalty_max=0.10,  # Large penalty
        )

        # Should be clamped to 0.0
        assert adjusted >= 0.0

    def test_config_defaults(self):
        """Test that config defaults work correctly."""
        confidence = 0.80

        # Using default values (enabled=True, boost_max=0.08, penalty_max=0.05)
        adjusted, reason = apply_context_richness_calibration(
            confidence,
            video_title="Test Video",
            video_description="Test Description",
            video_tags=["tag1", "tag2"],
            video_chapter="Chapter 1",
        )

        # All 4 signals present with defaults -> should boost
        assert adjusted > confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
