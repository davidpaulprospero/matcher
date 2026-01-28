"""Tests for escalation timeline + tier outcomes checkpoint persistence (US-005 Sprint 14).

Tests cover:
  - to_dict() includes _escalation_timeline under 'timeline' key
  - from_dict() restores _escalation_timeline correctly (5-event round-trip)
  - to_dict() includes _tier_outcomes under 'tier_outcomes' key
  - from_dict() restores _tier_outcomes correctly (success rates preserved)
  - Round-trip with hot_keywords: events within 30 min window survive save/restore
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock

import pytest

from src.downloader.escalation_manager import EscalationManager
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
    cooldown_seconds: float = 0.0  # No cooldown for test speed
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


def _force_escalation(mgr: EscalationManager, keyword: str,
                       error_output: str = "HTTP Error 403") -> None:
    """Force an escalation by recording enough failures to cross threshold."""
    mgr.record_failure(keyword, error_output)
    mgr.record_failure(keyword, error_output)


def _restore_from(data, stale_threshold=999999.0, budget=None):
    """Helper: restore EscalationManager from serialized data."""
    return EscalationManager.from_dict(
        data=data,
        impersonation_manager=_make_impersonation_manager(),
        extractor_args_config=FakeExtractorArgsConfig(cooldown_seconds=0.0),
        budget=budget,
        stale_threshold=stale_threshold,
    )


# ---------------------------------------------------------------------------
# AC1: to_dict() includes _escalation_timeline under 'timeline' key
# ---------------------------------------------------------------------------

class TestToDictIncludesTimeline:
    """Verify to_dict() serializes the escalation timeline."""

    @pytest.mark.fast
    def test_to_dict_has_timeline_key(self):
        mgr = _make_manager()
        data = mgr.to_dict()
        assert 'timeline' in data

    @pytest.mark.fast
    def test_to_dict_empty_timeline(self):
        mgr = _make_manager()
        data = mgr.to_dict()
        assert data['timeline'] == {}

    @pytest.mark.fast
    def test_to_dict_timeline_has_events_after_escalation(self):
        mgr = _make_manager()
        _force_escalation(mgr, "beach", "HTTP Error 403")
        data = mgr.to_dict()

        assert "beach" in data['timeline']
        events = data['timeline']['beach']
        assert len(events) == 1
        event = events[0]
        assert 'timestamp' in event
        assert 'from_tier' in event
        assert 'to_tier' in event
        assert 'trigger_category' in event

    @pytest.mark.fast
    def test_to_dict_timeline_all_events_per_keyword(self):
        """All recorded events for each keyword are serialized."""
        mgr = _make_manager()
        # Tier 1 -> 2
        _force_escalation(mgr, "river", "HTTP Error 403")
        # Tier 2 -> 3
        _force_escalation(mgr, "river", "HTTP Error 403")

        data = mgr.to_dict()
        events = data['timeline']['river']
        assert len(events) == 2
        assert events[0]['from_tier'] == EscalationTier.IMPERSONATE_ONLY.value
        assert events[0]['to_tier'] == EscalationTier.EXTRACTOR_ARGS.value
        assert events[1]['from_tier'] == EscalationTier.EXTRACTOR_ARGS.value
        assert events[1]['to_tier'] == EscalationTier.FULL_BYPASS.value

    @pytest.mark.fast
    def test_to_dict_timeline_multiple_keywords(self):
        mgr = _make_manager()
        _force_escalation(mgr, "alpha", "HTTP Error 403")
        _force_escalation(mgr, "beta", "HTTP Error 429")
        data = mgr.to_dict()

        assert "alpha" in data['timeline']
        assert "beta" in data['timeline']
        assert data['timeline']['alpha'][0]['trigger_category'] == '403'
        assert data['timeline']['beta'][0]['trigger_category'] == '429'


# ---------------------------------------------------------------------------
# AC2: from_dict() restores _escalation_timeline correctly
# ---------------------------------------------------------------------------

class TestFromDictRestoresTimeline:
    """Verify from_dict() restores timeline so get_keyword_escalation_timeline() returns same data."""

    @pytest.mark.fast
    def test_roundtrip_5_events(self):
        """Record 5 events, serialize, deserialize, verify same data."""
        mgr = _make_manager()
        keyword = "roundtrip"

        # Inject 5 timeline events directly (since tiers max at 3)
        now = time.time()
        for i in range(5):
            mgr._escalation_timeline.setdefault(keyword, []).append({
                'timestamp': now + i,
                'from_tier': 1,
                'to_tier': 2,
                'trigger_category': '403',
            })

        original_timeline = mgr.get_keyword_escalation_timeline()
        data = mgr.to_dict()
        restored = _restore_from(data)
        restored_timeline = restored.get_keyword_escalation_timeline()

        assert keyword in restored_timeline
        assert len(restored_timeline[keyword]) == 5
        for orig, rest in zip(original_timeline[keyword], restored_timeline[keyword]):
            assert orig['timestamp'] == rest['timestamp']
            assert orig['from_tier'] == rest['from_tier']
            assert orig['to_tier'] == rest['to_tier']
            assert orig['trigger_category'] == rest['trigger_category']

    @pytest.mark.fast
    def test_roundtrip_preserves_multiple_keywords(self):
        mgr = _make_manager()
        _force_escalation(mgr, "kw_a", "HTTP Error 403")
        _force_escalation(mgr, "kw_b", "HTTP Error 429")

        data = mgr.to_dict()
        restored = _restore_from(data)
        timeline = restored.get_keyword_escalation_timeline()

        assert "kw_a" in timeline
        assert "kw_b" in timeline
        assert timeline["kw_a"][0]["trigger_category"] == "403"
        assert timeline["kw_b"][0]["trigger_category"] == "429"

    @pytest.mark.fast
    def test_roundtrip_empty_timeline(self):
        mgr = _make_manager()
        data = mgr.to_dict()
        restored = _restore_from(data)
        assert restored.get_keyword_escalation_timeline() == {}

    @pytest.mark.fast
    def test_from_dict_missing_timeline_key(self):
        """from_dict() handles checkpoint without 'timeline' key (backward compat)."""
        data = {
            'keyword_states': {},
            'total_403s': 0,
            'total_successes': 0,
            'total_escalations': 0,
            'escalations_per_tier': {},
            'speed_escalations': 0,
            'saved_at': time.time(),
            # No 'timeline' key
        }
        restored = _restore_from(data)
        assert restored.get_keyword_escalation_timeline() == {}


# ---------------------------------------------------------------------------
# AC3: to_dict() includes _tier_outcomes under 'tier_outcomes' key
# ---------------------------------------------------------------------------

class TestToDictIncludesTierOutcomes:
    """Verify to_dict() serializes tier outcome data."""

    @pytest.mark.fast
    def test_to_dict_has_tier_outcomes_key(self):
        mgr = _make_manager()
        data = mgr.to_dict()
        assert 'tier_outcomes' in data

    @pytest.mark.fast
    def test_to_dict_empty_tier_outcomes(self):
        mgr = _make_manager()
        data = mgr.to_dict()
        assert data['tier_outcomes'] == {}

    @pytest.mark.fast
    def test_to_dict_tier_outcomes_after_recording(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=False)

        data = mgr.to_dict()
        outcomes = data['tier_outcomes']

        assert '403' in outcomes
        assert '429' in outcomes
        # Tier 1 for 403: 2 attempts, 1 success
        tier_1_data = outcomes['403'][EscalationTier.IMPERSONATE_ONLY.value]
        assert tier_1_data['attempts'] == 2
        assert tier_1_data['successes'] == 1
        # Tier 2 for 403: 1 attempt, 1 success
        tier_2_data = outcomes['403'][EscalationTier.EXTRACTOR_ARGS.value]
        assert tier_2_data['attempts'] == 1
        assert tier_2_data['successes'] == 1
        # Tier 3 for 429: 1 attempt, 0 successes
        tier_3_data = outcomes['429'][EscalationTier.FULL_BYPASS.value]
        assert tier_3_data['attempts'] == 1
        assert tier_3_data['successes'] == 0

    @pytest.mark.fast
    def test_to_dict_tier_outcomes_matches_get_tier_effectiveness(self):
        """Serialized structure matches get_tier_effectiveness() output."""
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        effectiveness = mgr.get_tier_effectiveness()
        data = mgr.to_dict()
        outcomes = data['tier_outcomes']

        # effectiveness gives {'403': {'tier_1': 0.6667}}
        # outcomes gives {'403': {1: {'successes': 2, 'attempts': 3}}}
        tier_1 = outcomes['403'][1]
        rate = tier_1['successes'] / tier_1['attempts']
        assert abs(rate - effectiveness['403']['tier_1']) < 0.001


# ---------------------------------------------------------------------------
# AC4: from_dict() restores _tier_outcomes correctly
# ---------------------------------------------------------------------------

class TestFromDictRestoresTierOutcomes:
    """Verify from_dict() restores tier outcomes so get_tier_effectiveness() returns same rates."""

    @pytest.mark.fast
    def test_roundtrip_tier_outcomes(self):
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=False)
        mgr.record_outcome('429', EscalationTier.FULL_BYPASS, success=True)

        original_effectiveness = mgr.get_tier_effectiveness()
        data = mgr.to_dict()
        restored = _restore_from(data)
        restored_effectiveness = restored.get_tier_effectiveness()

        assert original_effectiveness == restored_effectiveness

    @pytest.mark.fast
    def test_roundtrip_tier_outcomes_rates_precise(self):
        """Record mixed outcomes, verify exact success rates after round-trip."""
        mgr = _make_manager()
        # 403: Tier 1 = 3/5 = 0.6, Tier 2 = 1/2 = 0.5
        for _ in range(3):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        for _ in range(2):
            mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=False)

        data = mgr.to_dict()
        restored = _restore_from(data)
        eff = restored.get_tier_effectiveness()

        assert eff['403']['tier_1'] == 0.6
        assert eff['403']['tier_2'] == 0.5

    @pytest.mark.fast
    def test_roundtrip_empty_tier_outcomes(self):
        mgr = _make_manager()
        data = mgr.to_dict()
        restored = _restore_from(data)
        assert restored.get_tier_effectiveness() == {}

    @pytest.mark.fast
    def test_from_dict_missing_tier_outcomes_key(self):
        """from_dict() handles checkpoint without 'tier_outcomes' key (backward compat)."""
        data = {
            'keyword_states': {},
            'total_403s': 0,
            'total_successes': 0,
            'total_escalations': 0,
            'escalations_per_tier': {},
            'speed_escalations': 0,
            'saved_at': time.time(),
            # No 'tier_outcomes' key
        }
        restored = _restore_from(data)
        assert restored.get_tier_effectiveness() == {}

    @pytest.mark.fast
    def test_from_dict_tier_outcomes_string_keys_converted(self):
        """JSON serialization converts int keys to strings; from_dict must handle this."""
        import json
        mgr = _make_manager()
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        data = mgr.to_dict()
        # Simulate JSON round-trip (int keys become strings)
        json_str = json.dumps(data)
        data_from_json = json.loads(json_str)

        restored = _restore_from(data_from_json)
        eff = restored.get_tier_effectiveness()
        assert eff['403']['tier_1'] == 0.5


# ---------------------------------------------------------------------------
# AC5: Round-trip with hot_keywords within 30 min window
# ---------------------------------------------------------------------------

class TestRoundtripHotKeywords:
    """Verify hot_keywords still detected after checkpoint save/restore."""

    @pytest.mark.fast
    def test_hot_keywords_survive_roundtrip(self):
        """Record escalations within 30 min window, save, restore, verify hot."""
        mgr = _make_manager()
        now = time.time()
        keyword = "hot_roundtrip"

        # Inject 5 recent events (within 30 min)
        for i in range(5):
            mgr._escalation_timeline.setdefault(keyword, []).append({
                'timestamp': now - i * 10,  # 0s, 10s, 20s, 30s, 40s ago
                'from_tier': 1,
                'to_tier': 2,
                'trigger_category': '403',
            })

        # Verify hot before save
        hot_before = mgr.get_hot_keywords()
        assert len(hot_before) == 1
        assert hot_before[0]['keyword'] == keyword
        assert hot_before[0]['escalation_count'] == 5

        # Save and restore
        data = mgr.to_dict()
        restored = _restore_from(data)

        # Verify hot after restore
        hot_after = restored.get_hot_keywords()
        assert len(hot_after) == 1
        assert hot_after[0]['keyword'] == keyword
        assert hot_after[0]['escalation_count'] == 5

    @pytest.mark.fast
    def test_stale_hot_keywords_not_hot_after_roundtrip(self):
        """Old events (>30 min ago) should NOT be hot after restore."""
        mgr = _make_manager()
        keyword = "stale_hot"
        old_time = time.time() - 7200  # 2 hours ago

        for i in range(5):
            mgr._escalation_timeline.setdefault(keyword, []).append({
                'timestamp': old_time + i,
                'from_tier': 1,
                'to_tier': 2,
                'trigger_category': '403',
            })

        data = mgr.to_dict()
        restored = _restore_from(data)

        hot = restored.get_hot_keywords()
        assert len(hot) == 0

    @pytest.mark.fast
    def test_mixed_hot_and_cold_keywords_roundtrip(self):
        """After round-trip, only recently escalated keywords are hot."""
        mgr = _make_manager()
        now = time.time()

        # "hot_one": 5 recent events
        for i in range(5):
            mgr._escalation_timeline.setdefault("hot_one", []).append({
                'timestamp': now - i * 10,
                'from_tier': 1,
                'to_tier': 2,
                'trigger_category': '403',
            })

        # "cold_one": 2 recent events (below threshold)
        for i in range(2):
            mgr._escalation_timeline.setdefault("cold_one", []).append({
                'timestamp': now - i * 10,
                'from_tier': 1,
                'to_tier': 2,
                'trigger_category': '403',
            })

        data = mgr.to_dict()
        restored = _restore_from(data)

        hot = restored.get_hot_keywords()
        assert len(hot) == 1
        assert hot[0]['keyword'] == "hot_one"

    @pytest.mark.fast
    def test_full_roundtrip_timeline_outcomes_and_hot(self):
        """Combined round-trip: timeline, tier_outcomes, and hot_keywords all consistent."""
        mgr = _make_manager()
        now = time.time()

        # Generate real escalations
        _force_escalation(mgr, "combined_kw", "HTTP Error 403")
        # Record outcomes
        mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)
        mgr.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, success=True)

        # Add more timeline events to make it hot
        for i in range(4):
            mgr._escalation_timeline["combined_kw"].append({
                'timestamp': now - i * 5,
                'from_tier': 2,
                'to_tier': 3,
                'trigger_category': '403',
            })

        data = mgr.to_dict()
        restored = _restore_from(data)

        # Timeline preserved
        timeline = restored.get_keyword_escalation_timeline()
        assert "combined_kw" in timeline
        assert len(timeline["combined_kw"]) >= 5  # 1 real + 4 injected

        # Tier outcomes preserved
        eff = restored.get_tier_effectiveness()
        assert '403' in eff
        assert eff['403']['tier_1'] == 0.0  # 0/1
        assert eff['403']['tier_2'] == 1.0  # 1/1

        # Hot keywords preserved
        hot = restored.get_hot_keywords()
        assert any(h['keyword'] == "combined_kw" for h in hot)
