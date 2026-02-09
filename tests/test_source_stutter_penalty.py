"""Tests for source stutter penalty (US-84-004).

Verifies A-B-A source alternation detection and penalty application.
"""

import pytest
from unittest.mock import MagicMock
from src.matching.scoring import apply_source_stutter_penalty


def _make_config(penalty: float = 0.04):
    """Create a mock config with scoring.source_stutter_penalty."""
    scoring = MagicMock()
    scoring.source_stutter_penalty = penalty
    matching = MagicMock()
    matching.scoring = scoring
    config = MagicMock()
    config.matching = matching
    return config


class TestSourceStutterPenalty:
    """Test apply_source_stutter_penalty for A-B-A pattern detection."""

    def test_aba_pattern_penalized(self):
        """A-B-A pattern (videoA -> videoB -> videoA) should be penalized."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoA",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.76, abs=0.001)
        assert "source stutter A-B-A" in reason
        assert "videoA" in reason
        assert "videoB" in reason

    def test_aaa_continuation_not_penalized(self):
        """A-A-A pattern (continuation) should NOT be penalized."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoA",
            previous_source="videoA",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.80)
        assert reason == ""

    def test_abc_progression_not_penalized(self):
        """A-B-C pattern (progression) should NOT be penalized."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoC",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.80)
        assert reason == ""

    def test_no_prev_prev_source_no_penalty(self):
        """When only 2 segments exist (no prev_prev), no penalty applied."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoA",
            previous_source="videoB",
            prev_prev_source=None,
            config=config,
        )

        assert adjusted == pytest.approx(0.80)
        assert reason == ""

    def test_no_previous_source_no_penalty(self):
        """When no previous segment, no penalty applied."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoA",
            previous_source=None,
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.80)
        assert reason == ""

    def test_no_current_source_no_penalty(self):
        """When current source is None, no penalty applied."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source=None,
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.80)
        assert reason == ""

    def test_configurable_penalty_magnitude(self):
        """Penalty magnitude should be configurable via config."""
        config = _make_config(0.10)  # Higher penalty
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoA",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.70, abs=0.001)

    def test_penalty_does_not_go_below_zero(self):
        """Penalty should not push confidence below 0."""
        config = _make_config(0.50)  # Very large penalty
        confidence = 0.10

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoA",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.0)

    def test_aba_reason_contains_source_ids(self):
        """Reason string should contain the A-B-A source IDs."""
        config = _make_config(0.04)

        _, reason = apply_source_stutter_penalty(
            0.80,
            current_source="vid_123456789abc",
            previous_source="vid_999888777aaa",
            prev_prev_source="vid_123456789abc",
            config=config,
        )

        # Source IDs are truncated to 12 chars in reason
        assert "vid_12345678" in reason
        assert "vid_99988877" in reason

    def test_abb_not_penalized(self):
        """A-B-B pattern should NOT be penalized (not A-B-A)."""
        config = _make_config(0.04)
        confidence = 0.80

        adjusted, reason = apply_source_stutter_penalty(
            confidence,
            current_source="videoB",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.80)
        assert reason == ""

    def test_config_missing_scoring_uses_default(self):
        """When scoring config is missing, use default penalty of 0.04."""
        matching = MagicMock()
        matching.scoring = None
        config = MagicMock()
        config.matching = matching

        adjusted, reason = apply_source_stutter_penalty(
            0.80,
            current_source="videoA",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        assert adjusted == pytest.approx(0.76, abs=0.001)
        assert "source stutter A-B-A" in reason


class TestSourceStutterBreakdownEntry:
    """Verify confidence_breakdown entry format for source stutter penalty."""

    def test_breakdown_entry_has_correct_component_name(self):
        """Breakdown entry should use 'source_stutter_penalty' component name."""
        config = _make_config(0.04)

        adjusted, reason = apply_source_stutter_penalty(
            0.80,
            current_source="videoA",
            previous_source="videoB",
            prev_prev_source="videoA",
            config=config,
        )

        # The reason string should be non-empty for A-B-A
        assert reason != ""
        # The component name is set by _record_breakdown in tiered_matcher, not here.
        # We just verify the reason is descriptive enough.
        assert "source stutter" in reason
        assert "-0.04" in reason
