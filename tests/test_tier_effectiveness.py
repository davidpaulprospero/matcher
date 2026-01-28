"""Unit tests for error-pattern-to-tier success correlation (US-004 Sprint 13).

Tests cover:
  - get_tier_effectiveness() returns correct structure with success rates
  - Mixed outcomes: correct success rate calculation per tier
  - Recommendations: skip escalation when Tier 1 has >80% success
  - Recommendations: skip Tier 2 when <20% but Tier 3 >60%
  - tier_effectiveness included in RateLimitMetrics.export_to_json()
  - Empty data returns empty dict (no division-by-zero or KeyError)
"""

from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock

import pytest

from src.downloader.escalation_manager import EscalationManager
from src.downloader.rate_limit_metrics import RateLimitMetrics
from src.downloader.types import EscalationTier


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 0.0
    max_tier: int = 3


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


def _make_manager(cooldown: float = 0.0) -> EscalationManager:
    """Create an EscalationManager with zero cooldown for fast tests."""
    config = FakeExtractorArgsConfig(cooldown_seconds=cooldown)
    return EscalationManager(
        impersonation_manager=_make_impersonation_manager(),
        extractor_args_config=config,
    )


# ---------------------------------------------------------------------------
# AC 1: get_tier_effectiveness() returns correct structure
# ---------------------------------------------------------------------------

class TestTierEffectivenessStructure:
    """Test that get_tier_effectiveness() returns the expected nested dict."""

    @pytest.mark.fast
    def test_returns_dict_keyed_by_category(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('429', EscalationTier.EXTRACTOR_ARGS, success=False)

        result = mgr.get_tier_effectiveness()

        assert isinstance(result, dict)
        assert '403' in result
        assert '429' in result

    @pytest.mark.fast
    def test_tier_keys_are_tier_1_tier_2_tier_3(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)
        mgr.record_outcome('403', EscalationTier.FULL_BYPASS, success=True)

        result = mgr.get_tier_effectiveness()

        assert 'tier_1' in result['403']
        assert 'tier_2' in result['403']
        assert 'tier_3' in result['403']

    @pytest.mark.fast
    def test_success_rate_is_float_between_0_and_1(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        result = mgr.get_tier_effectiveness()

        rate = result['403']['tier_1']
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    @pytest.mark.fast
    def test_multiple_categories_tracked_independently(self):
        mgr = _make_manager()
        # 403: all succeed at Tier 1
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        # 429: all fail at Tier 1
        mgr.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, success=False)
        mgr.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, success=False)

        result = mgr.get_tier_effectiveness()

        assert result['403']['tier_1'] == 1.0
        assert result['429']['tier_1'] == 0.0


# ---------------------------------------------------------------------------
# AC 2: Mixed outcomes with correct success rate calculation
# ---------------------------------------------------------------------------

class TestMixedOutcomes:
    """Test success rate calculation with mixed success/failure outcomes."""

    @pytest.mark.fast
    def test_tier_1_80_percent_success(self):
        """10 attempts at Tier 1, 8 success → 80% rate."""
        mgr = _make_manager()
        for _ in range(8):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(2):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        result = mgr.get_tier_effectiveness()

        assert result['403']['tier_1'] == 0.8

    @pytest.mark.fast
    def test_tier_2_30_percent_success(self):
        """10 attempts at Tier 2, 3 success → 30% rate."""
        mgr = _make_manager()
        for _ in range(3):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        for _ in range(7):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)

        result = mgr.get_tier_effectiveness()

        assert result['403']['tier_2'] == 0.3

    @pytest.mark.fast
    def test_multiple_tiers_same_category(self):
        """Tier 1: 80%, Tier 2: 30% for same category."""
        mgr = _make_manager()
        # Tier 1: 8/10
        for _ in range(8):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(2):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        # Tier 2: 3/10
        for _ in range(3):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        for _ in range(7):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)

        result = mgr.get_tier_effectiveness()

        assert result['403']['tier_1'] == 0.8
        assert result['403']['tier_2'] == 0.3

    @pytest.mark.fast
    def test_100_percent_success(self):
        mgr = _make_manager()
        for _ in range(5):
            mgr.record_outcome('bot_detection', EscalationTier.FULL_BYPASS, success=True)

        result = mgr.get_tier_effectiveness()
        assert result['bot_detection']['tier_3'] == 1.0

    @pytest.mark.fast
    def test_0_percent_success(self):
        mgr = _make_manager()
        for _ in range(5):
            mgr.record_outcome('ip_blocked', EscalationTier.EXTRACTOR_ARGS, success=False)

        result = mgr.get_tier_effectiveness()
        assert result['ip_blocked']['tier_2'] == 0.0


# ---------------------------------------------------------------------------
# AC 3: Recommendation — skip escalation for categories with >80% Tier 1
# ---------------------------------------------------------------------------

class TestSkipEscalationRecommendation:
    """Test that >80% Tier 1 success generates 'skip escalation' recommendation."""

    @pytest.mark.fast
    def test_skip_escalation_when_tier_1_above_80(self):
        mgr = _make_manager()
        # 403 at Tier 1: 9/10 = 90% success
        for _ in range(9):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        recs = mgr.get_tier_recommendations()

        assert any('skip escalation for 403' in r for r in recs)

    @pytest.mark.fast
    def test_no_skip_escalation_when_tier_1_at_80(self):
        """Exactly 80% should NOT trigger recommendation (>80% required)."""
        mgr = _make_manager()
        for _ in range(8):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(2):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        recs = mgr.get_tier_recommendations()

        assert not any('skip escalation for 403' in r for r in recs)

    @pytest.mark.fast
    def test_no_skip_escalation_when_tier_1_below_80(self):
        mgr = _make_manager()
        for _ in range(7):
            mgr.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(3):
            mgr.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, success=False)

        recs = mgr.get_tier_recommendations()

        assert not any('skip escalation for 429' in r for r in recs)

    @pytest.mark.fast
    def test_skip_escalation_per_category(self):
        """Only the high-success category gets the recommendation."""
        mgr = _make_manager()
        # 403: 95% at Tier 1
        for _ in range(19):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        # 429: 50% at Tier 1
        for _ in range(5):
            mgr.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(5):
            mgr.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, success=False)

        recs = mgr.get_tier_recommendations()

        assert any('skip escalation for 403' in r for r in recs)
        assert not any('skip escalation for 429' in r for r in recs)


# ---------------------------------------------------------------------------
# AC 4: Recommendation — skip Tier 2 when <20% but Tier 3 >60%
# ---------------------------------------------------------------------------

class TestSkipTier2Recommendation:
    """Test that <20% Tier 2 + >60% Tier 3 generates 'skip Tier 2' recommendation."""

    @pytest.mark.fast
    def test_skip_tier_2_when_low_t2_high_t3(self):
        mgr = _make_manager()
        # Tier 2: 1/10 = 10% success
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        for _ in range(9):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)
        # Tier 3: 7/10 = 70% success
        for _ in range(7):
            mgr.record_outcome('403', EscalationTier.FULL_BYPASS, success=True)
        for _ in range(3):
            mgr.record_outcome('403', EscalationTier.FULL_BYPASS, success=False)

        recs = mgr.get_tier_recommendations()

        assert any('skip Tier 2 for 403' in r for r in recs)

    @pytest.mark.fast
    def test_no_skip_tier_2_when_t2_at_20(self):
        """Exactly 20% Tier 2 should NOT trigger (requires <20%)."""
        mgr = _make_manager()
        # Tier 2: 2/10 = 20%
        for _ in range(2):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        for _ in range(8):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)
        # Tier 3: 80%
        for _ in range(8):
            mgr.record_outcome('403', EscalationTier.FULL_BYPASS, success=True)
        for _ in range(2):
            mgr.record_outcome('403', EscalationTier.FULL_BYPASS, success=False)

        recs = mgr.get_tier_recommendations()

        assert not any('skip Tier 2 for 403' in r for r in recs)

    @pytest.mark.fast
    def test_no_skip_tier_2_when_t3_at_60(self):
        """Exactly 60% Tier 3 should NOT trigger (requires >60%)."""
        mgr = _make_manager()
        # Tier 2: 1/10 = 10%
        mgr.record_outcome('429', EscalationTier.EXTRACTOR_ARGS, success=True)
        for _ in range(9):
            mgr.record_outcome('429', EscalationTier.EXTRACTOR_ARGS, success=False)
        # Tier 3: 6/10 = 60%
        for _ in range(6):
            mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=True)
        for _ in range(4):
            mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=False)

        recs = mgr.get_tier_recommendations()

        assert not any('skip Tier 2 for 429' in r for r in recs)

    @pytest.mark.fast
    def test_no_skip_tier_2_when_only_t2_data(self):
        """No Tier 3 data → no skip Tier 2 recommendation."""
        mgr = _make_manager()
        # Tier 2: 0/10 = 0%
        for _ in range(10):
            mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)

        recs = mgr.get_tier_recommendations()

        assert not any('skip Tier 2 for 403' in r for r in recs)

    @pytest.mark.fast
    def test_both_recommendations_possible(self):
        """One category skips escalation, another skips Tier 2."""
        mgr = _make_manager()
        # 403: Tier 1 at 90% → skip escalation
        for _ in range(9):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        # 429: Tier 2 at 10%, Tier 3 at 70% → skip Tier 2
        mgr.record_outcome('429', EscalationTier.EXTRACTOR_ARGS, success=True)
        for _ in range(9):
            mgr.record_outcome('429', EscalationTier.EXTRACTOR_ARGS, success=False)
        for _ in range(7):
            mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=True)
        for _ in range(3):
            mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=False)

        recs = mgr.get_tier_recommendations()

        assert any('skip escalation for 403' in r for r in recs)
        assert any('skip Tier 2 for 429' in r for r in recs)


# ---------------------------------------------------------------------------
# AC 5: tier_effectiveness included in export_to_json()
# ---------------------------------------------------------------------------

class TestExportToJsonIntegration:
    """Test that tier_effectiveness is in export_to_json() under 'escalation'."""

    @pytest.mark.fast
    def test_tier_effectiveness_in_export(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        mgr.record_outcome('429', EscalationTier.EXTRACTOR_ARGS, success=True)

        metrics = RateLimitMetrics()
        export = metrics.export_to_json(escalation_manager=mgr)

        assert 'tier_effectiveness' in export['escalation']
        effectiveness = export['escalation']['tier_effectiveness']
        assert '403' in effectiveness
        assert effectiveness['403']['tier_1'] == 0.5
        assert '429' in effectiveness
        assert effectiveness['429']['tier_2'] == 1.0

    @pytest.mark.fast
    def test_tier_recommendations_in_export(self):
        mgr = _make_manager()
        # 403 at Tier 1: 95% success
        for _ in range(19):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        metrics = RateLimitMetrics()
        export = metrics.export_to_json(escalation_manager=mgr)

        assert 'tier_recommendations' in export['escalation']
        recs = export['escalation']['tier_recommendations']
        assert isinstance(recs, list)
        assert any('skip escalation for 403' in r for r in recs)

    @pytest.mark.fast
    def test_empty_effectiveness_in_export(self):
        """No outcomes recorded → empty tier_effectiveness in export."""
        mgr = _make_manager()
        metrics = RateLimitMetrics()
        export = metrics.export_to_json(escalation_manager=mgr)

        assert export['escalation']['tier_effectiveness'] == {}
        assert export['escalation']['tier_recommendations'] == []

    @pytest.mark.fast
    def test_export_without_escalation_manager(self):
        """No escalation_manager → no tier_effectiveness key."""
        metrics = RateLimitMetrics()
        export = metrics.export_to_json()

        # tier_effectiveness should NOT be present without an escalation_manager
        assert 'tier_effectiveness' not in export.get('escalation', {})


# ---------------------------------------------------------------------------
# AC 6: Empty data returns empty dict (no division-by-zero or KeyError)
# ---------------------------------------------------------------------------

class TestEmptyData:
    """Test edge cases with no data."""

    @pytest.mark.fast
    def test_empty_effectiveness_returns_empty_dict(self):
        mgr = _make_manager()
        result = mgr.get_tier_effectiveness()
        assert result == {}

    @pytest.mark.fast
    def test_empty_recommendations_returns_empty_list(self):
        mgr = _make_manager()
        recs = mgr.get_tier_recommendations()
        assert recs == []
        assert isinstance(recs, list)

    @pytest.mark.fast
    def test_reset_clears_tier_outcomes(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        assert mgr.get_tier_effectiveness() != {}

        mgr.reset_all()

        assert mgr.get_tier_effectiveness() == {}

    @pytest.mark.fast
    def test_single_attempt_no_crash(self):
        """Single attempt should not cause any issues."""
        mgr = _make_manager()
        mgr.record_outcome('age_gate', EscalationTier.FULL_BYPASS, success=False)

        result = mgr.get_tier_effectiveness()
        assert result == {'age_gate': {'tier_3': 0.0}}

    @pytest.mark.fast
    def test_recommendations_with_only_tier_2_no_crash(self):
        """Tier 2 data only (no Tier 1 or 3) should not crash."""
        mgr = _make_manager()
        for _ in range(5):
            mgr.record_outcome('ip_blocked', EscalationTier.EXTRACTOR_ARGS, success=False)

        recs = mgr.get_tier_recommendations()
        # No skip Tier 2 recommendation because no Tier 3 data
        assert not any('skip Tier 2' in r for r in recs)
        # No skip escalation because no Tier 1 data
        assert not any('skip escalation' in r for r in recs)

    @pytest.mark.fast
    def test_recommendations_with_only_tier_1_low_success(self):
        """Tier 1 at exactly 50% → no recommendations."""
        mgr = _make_manager()
        for _ in range(5):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(5):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        recs = mgr.get_tier_recommendations()
        assert recs == []
