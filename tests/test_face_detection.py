"""
Unit tests for face_detection module.

Tests B-roll detection and face scoring logic.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_detection import is_broll_scene


class TestBrollDetection:
    """Test B-roll scene detection based on face scores."""

    def test_broll_no_faces(self):
        """Test scene with no faces is B-roll."""
        face_score = 0.0

        is_broll = is_broll_scene(face_score)

        assert is_broll is True

    def test_broll_very_low_faces(self):
        """Test scene with very low face presence is B-roll."""
        face_score = 0.1

        is_broll = is_broll_scene(face_score)

        assert is_broll is True

    def test_broll_threshold_boundary(self):
        """Test scene at threshold boundary."""
        # Exactly at threshold (0.3) should be B-roll
        face_score = 0.29

        is_broll = is_broll_scene(face_score, threshold=0.3)

        assert is_broll is True

    def test_not_broll_above_threshold(self):
        """Test scene above threshold is not B-roll."""
        face_score = 0.5

        is_broll = is_broll_scene(face_score)

        assert is_broll is False

    def test_not_broll_high_faces(self):
        """Test scene with high face presence is not B-roll."""
        face_score = 0.9

        is_broll = is_broll_scene(face_score)

        assert is_broll is False

    def test_custom_threshold_strict(self):
        """Test with stricter threshold (0.1)."""
        face_score = 0.2

        # With default threshold (0.3), would be B-roll
        # With stricter threshold (0.1), should not be B-roll
        is_broll_default = is_broll_scene(face_score, threshold=0.3)
        is_broll_strict = is_broll_scene(face_score, threshold=0.1)

        assert is_broll_default is True
        assert is_broll_strict is False

    def test_custom_threshold_lenient(self):
        """Test with more lenient threshold (0.5)."""
        face_score = 0.4

        # With default threshold (0.3), would not be B-roll
        # With lenient threshold (0.5), should be B-roll
        is_broll_default = is_broll_scene(face_score, threshold=0.3)
        is_broll_lenient = is_broll_scene(face_score, threshold=0.5)

        assert is_broll_default is False
        assert is_broll_lenient is True


class TestFaceScoreRanges:
    """Test face score interpretation."""

    def test_zero_score(self):
        """Test zero face score (completely silent/no faces)."""
        assert is_broll_scene(0.0) is True

    def test_one_score(self):
        """Test maximum face score (always faces)."""
        assert is_broll_scene(1.0) is False

    def test_half_score(self):
        """Test half face score (faces in half of frames)."""
        # 0.5 > 0.3 threshold, so not B-roll
        assert is_broll_scene(0.5) is False

    def test_quarter_score(self):
        """Test quarter face score (faces in 25% of frames)."""
        # 0.25 < 0.3 threshold, so is B-roll
        assert is_broll_scene(0.25) is True


class TestThresholdValues:
    """Test different threshold values."""

    def test_threshold_zero(self):
        """Test with zero threshold (nothing is B-roll)."""
        # Even scenes with no faces wouldn't be B-roll
        assert is_broll_scene(0.0, threshold=0.0) is False

    def test_threshold_one(self):
        """Test with threshold of 1.0 (everything is B-roll)."""
        # Even scenes with all faces would be B-roll
        assert is_broll_scene(1.0, threshold=1.0) is False
        assert is_broll_scene(0.99, threshold=1.0) is True

    def test_threshold_typical_range(self):
        """Test typical threshold range (0.2 - 0.5)."""
        face_score = 0.35

        # Should be B-roll with low threshold
        assert is_broll_scene(face_score, threshold=0.2) is False

        # Should be B-roll with medium threshold
        assert is_broll_scene(face_score, threshold=0.4) is True

        # Should be B-roll with high threshold
        assert is_broll_scene(face_score, threshold=0.5) is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_face_score(self):
        """Test handling of negative face score (shouldn't happen)."""
        # Should treat as very low/zero
        is_broll = is_broll_scene(-0.1)
        assert is_broll is True

    def test_face_score_above_one(self):
        """Test handling of face score > 1.0 (shouldn't happen)."""
        # Should treat as high face presence
        is_broll = is_broll_scene(1.5)
        assert is_broll is False

    def test_very_small_threshold(self):
        """Test very small threshold value."""
        face_score = 0.01

        # Even tiny face presence exceeds tiny threshold
        is_broll = is_broll_scene(face_score, threshold=0.001)
        assert is_broll is False


class TestLogicConsistency:
    """Test logical consistency of B-roll detection."""

    def test_increasing_face_scores(self):
        """Test that increasing face scores eventually exceed threshold."""
        scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = [is_broll_scene(score, threshold=0.3) for score in scores]

        # First few should be B-roll, last few should not
        assert results[0] is True   # 0.0 < 0.3
        assert results[1] is True   # 0.1 < 0.3
        assert results[2] is True   # 0.2 < 0.3
        assert results[3] is False  # 0.3 >= 0.3
        assert results[4] is False  # 0.4 >= 0.3
        assert results[5] is False  # 0.5 >= 0.3

    def test_decreasing_thresholds(self):
        """Test that decreasing thresholds become stricter."""
        face_score = 0.25
        thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
        results = [is_broll_scene(face_score, threshold=t) for t in thresholds]

        # Should transition from B-roll to not-B-roll as threshold decreases
        assert results[0] is True   # 0.25 < 0.5
        assert results[1] is True   # 0.25 < 0.4
        assert results[2] is True   # 0.25 < 0.3
        assert results[3] is False  # 0.25 >= 0.2
        assert results[4] is False  # 0.25 >= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
