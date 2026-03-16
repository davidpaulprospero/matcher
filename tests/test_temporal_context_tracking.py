"""Tests for temporal context tracking for confidence adjustment.

Tests compute_temporal_confidence_adjustment() function in src/matching/scoring.py.
US-141-008: Temporal context tracking for confidence adjustment.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import compute_temporal_confidence_adjustment


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with default temporal context tracking settings."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.temporal_context_tracking_enabled = True
    config.matching.temporal_context_window = 10
    config.matching.temporal_boost_max = 0.03
    return config


@pytest.fixture
def mock_config_disabled():
    """Create mock config with temporal context tracking disabled."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.temporal_context_tracking_enabled = False
    config.matching.temporal_context_window = 10
    config.matching.temporal_boost_max = 0.03
    return config


@pytest.fixture
def sample_history():
    """Sample video metadata history with context scores."""
    return [
        {'context_score': 0.5, 'title_quality': 0.5, 'description_quality': 0.5, 'semantic_similarity': 0.5},
        {'context_score': 0.6, 'title_quality': 0.6, 'description_quality': 0.6, 'semantic_similarity': 0.6},
        {'context_score': 0.7, 'title_quality': 0.7, 'description_quality': 0.7, 'semantic_similarity': 0.7},
    ]


# ==============================================================================
# Test Function Existence and Signature
# ==============================================================================

class TestComputeTemporalConfidenceAdjustmentFunction:
    """Test that compute_temporal_confidence_adjustment function exists with correct signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """compute_temporal_confidence_adjustment function should exist in scoring module."""
        from src.matching.scoring import compute_temporal_confidence_adjustment
        assert callable(compute_temporal_confidence_adjustment)

    @pytest.mark.fast
    def test_function_returns_tuple(self, mock_config):
        """Function should return tuple of (confidence, reason)."""
        result = compute_temporal_confidence_adjustment(0.8, [], {}, mock_config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.fast
    def test_function_returns_float_and_string(self, mock_config):
        """Function should return (float, str)."""
        confidence, reason = compute_temporal_confidence_adjustment(0.8, [], {}, mock_config)
        assert isinstance(confidence, float)
        assert isinstance(reason, str)


# ==============================================================================
# Test: Feature Disabled
# ==============================================================================

class TestTemporalContextTrackingDisabled:
    """Test behavior when temporal context tracking is disabled."""

    @pytest.mark.fast
    def test_disabled_returns_original_confidence(self, mock_config_disabled):
        """When disabled, should return original confidence unchanged."""
        history = [
            {'context_score': 0.5},
            {'context_score': 0.7},
        ]
        current = {'context_score': 0.9}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config_disabled
        )

        assert confidence == 0.8
        assert reason == ""


# ==============================================================================
# Test: Empty or Insufficient History
# ==============================================================================

class TestInsufficientHistory:
    """Test behavior with empty or insufficient history."""

    @pytest.mark.fast
    def test_empty_history_returns_original(self, mock_config):
        """Empty history should return original confidence."""
        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, [], {'context_score': 0.5}, mock_config
        )

        assert confidence == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_single_item_history_returns_original(self, mock_config):
        """Single item history should return original confidence (need 2+ for trend)."""
        history = [{'context_score': 0.5}]
        current = {'context_score': 0.7}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        assert confidence == 0.8
        assert reason == ""

    @pytest.mark.fast
    def test_missing_context_score_returns_original(self, mock_config):
        """Missing context_score in current metadata should return original."""
        history = [
            {'context_score': 0.5},
            {'context_score': 0.7},
        ]
        current = {'title_quality': 0.8}  # No context_score

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        assert confidence == 0.8
        assert reason == ""


# ==============================================================================
# Test: Improving Context Trend (Boost)
# ==============================================================================

class TestImprovingContextTrend:
    """Test boost when context quality is improving."""

    @pytest.mark.fast
    def test_improving_trend_boosts_confidence(self, mock_config):
        """When context is improving, should apply positive adjustment."""
        # History shows improving trend: 0.5 -> 0.6 -> 0.7
        history = [
            {'context_score': 0.5},
            {'context_score': 0.6},
            {'context_score': 0.7},
        ]
        # Current is even better: 0.9
        current = {'context_score': 0.9}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # Should apply boost
        assert confidence > 0.8
        assert "improving" in reason.lower() or "+" in reason

    @pytest.mark.fast
    def test_improving_trend_respects_max_boost(self, mock_config):
        """Boost should not exceed temporal_boost_max."""
        # Large improvement
        history = [
            {'context_score': 0.1},
            {'context_score': 0.2},
            {'context_score': 0.3},
        ]
        current = {'context_score': 0.95}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # Max boost is 0.03, so max confidence is 0.83 (allow small epsilon for float precision)
        assert confidence <= 0.831
        assert confidence >= 0.8

    @pytest.mark.fast
    def test_small_improvement_no_adjustment(self, mock_config):
        """Small improvement below threshold should not apply adjustment."""
        history = [
            {'context_score': 0.7},
            {'context_score': 0.71},
            {'context_score': 0.72},
        ]
        current = {'context_score': 0.73}  # Only slightly better than avg

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # Should not apply boost (difference too small)
        assert confidence == 0.8
        assert reason == ""


# ==============================================================================
# Test: Degrading Context Trend (Penalty)
# ==============================================================================

class TestDegradingContextTrend:
    """Test penalty when context quality is degrading."""

    @pytest.mark.fast
    def test_degrading_trend_penalizes_confidence(self, mock_config):
        """When context is degrading, should apply negative adjustment."""
        # History shows degrading trend: 0.9 -> 0.7 -> 0.5
        history = [
            {'context_score': 0.9},
            {'context_score': 0.7},
            {'context_score': 0.5},
        ]
        # Current is even worse: 0.3
        current = {'context_score': 0.3}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # Should apply penalty
        assert confidence < 0.8
        assert "degrading" in reason.lower() or "-" in reason

    @pytest.mark.fast
    def test_degrading_trend_respects_max_penalty(self, mock_config):
        """Penalty should not exceed -temporal_boost_max."""
        # Large degradation
        history = [
            {'context_score': 0.9},
            {'context_score': 0.8},
            {'context_score': 0.7},
        ]
        current = {'context_score': 0.1}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # Max penalty is -0.03, so min confidence is 0.77
        assert confidence >= 0.77
        assert confidence <= 0.8


# ==============================================================================
# Test: Window Size
# ==============================================================================

class TestWindowSize:
    """Test behavior with different window sizes."""

    @pytest.mark.fast
    def test_uses_config_window(self, mock_config):
        """Should use temporal_context_window from config."""
        # Create history longer than default window (10)
        history = [{'context_score': 0.5 + i * 0.05} for i in range(20)]
        # Current is improving
        current = {'context_score': 0.9}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # Should still work - just uses last 10 items
        # The trend should still be detected
        assert isinstance(confidence, float)

    @pytest.mark.fast
    def test_small_window(self, mock_config):
        """Should work with small window."""
        mock_config.matching.temporal_context_window = 3

        history = [
            {'context_score': 0.5},
            {'context_score': 0.6},
            {'context_score': 0.7},
        ]
        current = {'context_score': 0.9}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        assert isinstance(confidence, float)


# ==============================================================================
# Test: Boundary Conditions
# ==============================================================================

class TestBoundaryConditions:
    """Test boundary conditions."""

    @pytest.mark.fast
    def test_confidence_clamped_to_one(self, mock_config):
        """Confidence should be clamped to max 1.0."""
        history = [
            {'context_score': 0.5},
            {'context_score': 0.6},
        ]
        current = {'context_score': 0.9}

        # Start with high confidence
        confidence, reason = compute_temporal_confidence_adjustment(
            0.99, history, current, mock_config
        )

        assert confidence <= 1.0

    @pytest.mark.fast
    def test_confidence_clamped_to_zero(self, mock_config):
        """Confidence should be clamped to min 0.0."""
        history = [
            {'context_score': 0.9},
            {'context_score': 0.8},
        ]
        current = {'context_score': 0.1}

        # Start with low confidence
        confidence, reason = compute_temporal_confidence_adjustment(
            0.01, history, current, mock_config
        )

        assert confidence >= 0.0


# ==============================================================================
# Test: Trend Detection
# ==============================================================================

class TestTrendDetection:
    """Test trend detection logic."""

    @pytest.mark.fast
    def test_positive_trend_detected(self, mock_config):
        """Positive trend should be detected."""
        # Consistent upward trend
        history = [
            {'context_score': 0.3},
            {'context_score': 0.4},
            {'context_score': 0.5},
            {'context_score': 0.6},
            {'context_score': 0.7},
        ]
        current = {'context_score': 0.8}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        assert confidence > 0.8

    @pytest.mark.fast
    def test_negative_trend_detected(self, mock_config):
        """Negative trend should be detected."""
        # Consistent downward trend
        history = [
            {'context_score': 0.8},
            {'context_score': 0.7},
            {'context_score': 0.6},
            {'context_score': 0.5},
            {'context_score': 0.4},
        ]
        current = {'context_score': 0.3}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        assert confidence < 0.8

    @pytest.mark.fast
    def test_flat_trend_no_adjustment(self, mock_config):
        """Flat trend should not apply adjustment."""
        # Flat/inconsistent trend
        history = [
            {'context_score': 0.5},
            {'context_score': 0.5},
            {'context_score': 0.5},
            {'context_score': 0.5},
        ]
        current = {'context_score': 0.5}

        confidence, reason = compute_temporal_confidence_adjustment(
            0.8, history, current, mock_config
        )

        # No significant trend, so no adjustment
        assert confidence == 0.8
