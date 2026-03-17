"""
Tests for US-84-008: Ambiguous pool detection and flagging.

When top-10 candidate similarity scores have very low variance (below threshold),
the match is flagged as 'ambiguous_pool' in match metadata and a warning is logged.
"""

import logging
import statistics
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.matching.scoring import calculate_adaptive_threshold
from src.matching.metrics import (
    MatchQualityMetrics,
    calculate_match_quality_metrics,
)
from src.utils import SRTSegment, Match, MatchResult


# ── Helpers ──────────────────────────────────────────────────────────────

def _seg(text: str = "test", source_file: str = "vid1") -> SRTSegment:
    """Create a minimal SRTSegment for testing."""
    return SRTSegment(
        index=0, start_time=0.0, end_time=5.0, text=text,
        source_file=source_file,
    )


def _make_candidates(scores: list) -> list:
    """Create (SRTSegment, score) tuples from a list of floats."""
    return [(_seg(f"candidate_{i}"), s) for i, s in enumerate(scores)]


def _make_config(variance_warning_threshold: float = 0.02):
    """Build a mock config with scoring.variance_warning_threshold."""
    scoring = MagicMock()
    scoring.confidence_floor = 0.05  # Required for duck-type check in _get_scoring_config
    scoring.variance_warning_threshold = variance_warning_threshold

    matching = MagicMock()
    matching.scoring = scoring

    config = MagicMock()
    config.matching = matching
    return config


# ══════════════════════════════════════════════════════════════════════════
# 1. Flag SET when 10 candidates score 0.87-0.89 (low variance)
# ══════════════════════════════════════════════════════════════════════════

class TestAmbiguousPoolFlagSet:
    """Verify ambiguous_pool is flagged when top candidates have near-identical scores."""

    def test_ambiguous_pool_flagged_tight_scores(self):
        """10 candidates scoring 0.87-0.89 → ambiguous_pool in reason."""
        scores = [0.890, 0.889, 0.888, 0.887, 0.886, 0.885, 0.880, 0.878, 0.875, 0.870]
        candidates = _make_candidates(scores)

        # Verify variance is indeed below 0.02
        stdev = statistics.stdev(scores)
        assert stdev < 0.02, f"Precondition failed: stdev={stdev:.4f} should be < 0.02"

        config = _make_config(variance_warning_threshold=0.02)
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="A reasonably long voiceover segment with enough text",
            candidates=candidates,
            config=config,
        )

        assert "ambiguous_pool" in reason

    def test_ambiguous_pool_all_identical_scores(self):
        """All 10 candidates with score 0.88 → ambiguous_pool flagged."""
        scores = [0.88] * 10
        candidates = _make_candidates(scores)

        config = _make_config(variance_warning_threshold=0.02)
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config,
        )

        assert "ambiguous_pool" in reason

    def test_ambiguous_pool_fifteen_candidates_uses_top_10(self):
        """15 candidates where top-10 are tight but bottom 5 spread → still ambiguous."""
        tight_top10 = [0.880 + i * 0.001 for i in range(10)]  # 0.880 to 0.889
        spread_bottom5 = [0.5, 0.4, 0.3, 0.2, 0.1]
        scores = tight_top10 + spread_bottom5
        candidates = _make_candidates(scores)

        config = _make_config(variance_warning_threshold=0.02)
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a reasonably long voiceover segment here",
            candidates=candidates,
            config=config,
        )

        # Top-10 stdev is ~0.003, well below 0.02
        assert "ambiguous_pool" in reason


# ══════════════════════════════════════════════════════════════════════════
# 2. Flag NOT SET when clear winner exists
# ══════════════════════════════════════════════════════════════════════════

class TestAmbiguousPoolNotFlagged:
    """Verify ambiguous_pool is NOT flagged when a clear winner exists."""

    def test_clear_winner_not_flagged(self):
        """0.95 vs 0.70s → clear winner, no ambiguous_pool."""
        scores = [0.95, 0.72, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]
        candidates = _make_candidates(scores)

        # Verify variance is above 0.02
        stdev = statistics.stdev(scores)
        assert stdev > 0.02, f"Precondition: stdev={stdev:.4f} should be > 0.02"

        config = _make_config(variance_warning_threshold=0.02)
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config,
        )

        assert "ambiguous_pool" not in reason

    def test_moderate_spread_not_flagged(self):
        """Scores spreading 0.90 down to 0.70 → not ambiguous."""
        scores = [0.90, 0.87, 0.84, 0.81, 0.78, 0.75, 0.74, 0.73, 0.72, 0.70]
        candidates = _make_candidates(scores)

        config = _make_config(variance_warning_threshold=0.02)
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config,
        )

        assert "ambiguous_pool" not in reason

    def test_few_candidates_no_false_alarm(self):
        """Only 3 candidates → uses top-5 logic, should not false-alarm."""
        scores = [0.88, 0.87, 0.86]
        candidates = _make_candidates(scores)

        config = _make_config(variance_warning_threshold=0.02)
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config,
        )

        # With only 3 candidates, stdev of [0.88, 0.87, 0.86] = 0.01 < 0.02
        # This IS ambiguous (correctly flagged)
        assert "ambiguous_pool" in reason


# ══════════════════════════════════════════════════════════════════════════
# 3. Warning logging
# ══════════════════════════════════════════════════════════════════════════

class TestAmbiguousPoolWarning:
    """Verify warning is logged with expected format."""

    def test_warning_logged_on_ambiguous_pool(self, caplog):
        """Warning 'Top N candidates within X range' is logged."""
        scores = [0.88] * 10
        candidates = _make_candidates(scores)

        config = _make_config(variance_warning_threshold=0.02)
        with caplog.at_level(logging.WARNING, logger="src.matching.scoring"):
            calculate_adaptive_threshold(
                base_threshold=0.85,
                voiceover_text="This is a normal length voiceover segment for testing",
                candidates=candidates,
                config=config,
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1
        msg = warnings[0].message
        assert "candidates within" in msg
        assert "match selection may be arbitrary" in msg

    def test_no_warning_on_clear_winner(self, caplog):
        """No warning logged when clear winner exists."""
        scores = [0.95, 0.72, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64]
        candidates = _make_candidates(scores)

        config = _make_config(variance_warning_threshold=0.02)
        with caplog.at_level(logging.WARNING, logger="src.matching.scoring"):
            calculate_adaptive_threshold(
                base_threshold=0.85,
                voiceover_text="This is a normal length voiceover segment for testing",
                candidates=candidates,
                config=config,
            )

        warnings = [r for r in caplog.records
                     if r.levelno >= logging.WARNING
                     and "match selection may be arbitrary" in r.message]
        assert len(warnings) == 0


# ══════════════════════════════════════════════════════════════════════════
# 4. Config field: variance_warning_threshold
# ══════════════════════════════════════════════════════════════════════════

class TestVarianceWarningThresholdConfig:
    """Verify the config field controls the threshold."""

    def test_custom_higher_threshold_flags_wider_spread(self):
        """With threshold=0.05, even moderate spread triggers ambiguous_pool."""
        # stdev of these scores ≈ 0.032 — above 0.02 but below 0.05
        scores = [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81]
        stdev = statistics.stdev(scores)
        assert 0.02 < stdev < 0.05, f"Precondition: stdev={stdev:.4f} should be between 0.02 and 0.05"

        candidates = _make_candidates(scores)

        # With default 0.02, this should NOT be flagged
        config_default = _make_config(variance_warning_threshold=0.02)
        _, reason_default = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config_default,
        )
        assert "ambiguous_pool" not in reason_default

        # With higher threshold 0.05, this SHOULD be flagged
        config_higher = _make_config(variance_warning_threshold=0.05)
        _, reason_higher = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config_higher,
        )
        assert "ambiguous_pool" in reason_higher

    def test_default_threshold_from_scoring_config(self):
        """Default variance_warning_threshold is 0.02."""
        from src.config.sections.matching import MatchingScoringConfig
        sc = MatchingScoringConfig()
        assert sc.variance_warning_threshold == 0.02


# ══════════════════════════════════════════════════════════════════════════
# 5. MatchResult.ambiguous_pool field
# ══════════════════════════════════════════════════════════════════════════

class TestMatchResultAmbiguousPool:
    """Verify MatchResult stores and serializes ambiguous_pool flag."""

    def _make_match(self) -> Match:
        return Match(
            voiceover_segment=_seg("vo"),
            video_segment=_seg("vid"),
            video_scene=None,
            confidence=0.88,
            reasoning="test",
        )

    def test_default_false(self):
        """MatchResult.ambiguous_pool defaults to False."""
        mr = MatchResult(primary_match=self._make_match())
        assert mr.ambiguous_pool is False

    def test_set_true(self):
        """MatchResult.ambiguous_pool can be set to True."""
        mr = MatchResult(primary_match=self._make_match(), ambiguous_pool=True)
        assert mr.ambiguous_pool is True

    def test_to_dict_includes_flag(self):
        """to_dict serializes ambiguous_pool."""
        mr = MatchResult(primary_match=self._make_match(), ambiguous_pool=True)
        d = mr.to_dict()
        assert d['ambiguous_pool'] is True

    def test_to_dict_default_false(self):
        """to_dict includes ambiguous_pool=False by default."""
        mr = MatchResult(primary_match=self._make_match())
        d = mr.to_dict()
        assert d['ambiguous_pool'] is False


# ══════════════════════════════════════════════════════════════════════════
# 6. Quality report: uncertain_matches_count
# ══════════════════════════════════════════════════════════════════════════

class TestUncertainMatchesCount:
    """Verify uncertain_matches_count in MatchQualityMetrics."""

    def _make_match_result(self, confidence: float, ambiguous: bool = False) -> MatchResult:
        m = Match(
            voiceover_segment=_seg("vo"),
            video_segment=_seg("vid"),
            video_scene=None,
            confidence=confidence,
            reasoning="test",
        )
        return MatchResult(primary_match=m, ambiguous_pool=ambiguous)

    def test_counts_ambiguous_matches(self):
        """uncertain_matches_count reflects number of ambiguous_pool matches."""
        results = [
            self._make_match_result(0.88, ambiguous=True),
            self._make_match_result(0.92, ambiguous=False),
            self._make_match_result(0.85, ambiguous=True),
            self._make_match_result(0.90, ambiguous=False),
        ]
        metrics = calculate_match_quality_metrics(results, total_segments=4)
        assert metrics.uncertain_matches_count == 2

    def test_zero_when_none_ambiguous(self):
        """uncertain_matches_count is 0 when no matches are ambiguous."""
        results = [
            self._make_match_result(0.92, ambiguous=False),
            self._make_match_result(0.88, ambiguous=False),
        ]
        metrics = calculate_match_quality_metrics(results, total_segments=2)
        assert metrics.uncertain_matches_count == 0

    def test_all_ambiguous(self):
        """uncertain_matches_count equals total when all are ambiguous."""
        results = [
            self._make_match_result(0.87, ambiguous=True),
            self._make_match_result(0.86, ambiguous=True),
            self._make_match_result(0.88, ambiguous=True),
        ]
        metrics = calculate_match_quality_metrics(results, total_segments=3)
        assert metrics.uncertain_matches_count == 3

    def test_serialization_roundtrip(self):
        """uncertain_matches_count survives to_dict/from_dict roundtrip."""
        original = MatchQualityMetrics(
            total_segments=10, matched_segments=8, uncertain_matches_count=3,
        )
        d = original.to_dict()
        restored = MatchQualityMetrics.from_dict(d)
        assert restored.uncertain_matches_count == 3

    def test_empty_matches(self):
        """No matches → uncertain_matches_count is 0."""
        metrics = calculate_match_quality_metrics([], total_segments=5)
        assert metrics.uncertain_matches_count == 0


# ══════════════════════════════════════════════════════════════════════════
# 7. Edge cases and defensive behavior
# ══════════════════════════════════════════════════════════════════════════

class TestAmbiguousPoolEdgeCases:
    """Edge cases for ambiguous pool detection."""

    def test_no_candidates(self):
        """Empty candidates → no crash, no ambiguous_pool in reason."""
        config = _make_config()
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="Some text here for the test",
            candidates=[],
            config=config,
        )
        assert "ambiguous_pool" not in reason

    def test_single_candidate(self):
        """Single candidate → no crash, no ambiguous_pool."""
        candidates = _make_candidates([0.88])
        config = _make_config()
        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="Some text here for the test",
            candidates=candidates,
            config=config,
        )
        assert "ambiguous_pool" not in reason

    def test_none_config_uses_default_threshold(self):
        """None config falls back to 0.02 default."""
        scores = [0.88] * 10
        candidates = _make_candidates(scores)

        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=None,
        )

        # stdev of all-identical = 0.0 < 0.02 default → ambiguous_pool
        assert "ambiguous_pool" in reason

    def test_threshold_not_reduced_on_ambiguous(self):
        """When ambiguous pool detected, threshold should NOT be reduced."""
        scores = [0.88] * 10
        candidates = _make_candidates(scores)
        config = _make_config()

        threshold, reason = calculate_adaptive_threshold(
            base_threshold=0.85,
            voiceover_text="This is a normal length voiceover segment for testing",
            candidates=candidates,
            config=config,
        )

        # Base threshold 0.85 should not be reduced (no -0.05 for low variance)
        assert threshold >= 0.85
