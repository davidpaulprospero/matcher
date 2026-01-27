"""Unit tests for per-keyword escalation timeline export (US-003 Sprint 13).

Tests cover:
  - get_keyword_escalation_timeline() returns correct event structure
  - Timeline limited to last 20 events per keyword
  - get_hot_keywords() returns keywords with >3 escalations in last 30 min
  - get_hot_keywords() excludes keywords with <=3 or stale escalations
  - Timeline and hot_keywords included in export_to_json()
"""

import time
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

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


def _force_escalation(mgr: EscalationManager, keyword: str, error_output: str = "HTTP Error 403") -> None:
    """Force an escalation by recording enough failures to cross threshold."""
    mgr.record_failure(keyword, error_output)
    mgr.record_failure(keyword, error_output)


# ---------------------------------------------------------------------------
# AC1: get_keyword_escalation_timeline() returns dict keyed by keyword
#      with list of {timestamp, from_tier, to_tier, trigger_category} events
# ---------------------------------------------------------------------------

class TestEscalationTimelineStructure:
    """Test timeline event structure and content."""

    @pytest.mark.fast
    def test_timeline_returns_dict_keyed_by_keyword(self):
        mgr = _make_manager()
        _force_escalation(mgr, "sunset", "HTTP Error 403")
        timeline = mgr.get_keyword_escalation_timeline()

        assert isinstance(timeline, dict)
        assert "sunset" in timeline
        assert len(timeline["sunset"]) == 1

    @pytest.mark.fast
    def test_timeline_event_has_required_fields(self):
        mgr = _make_manager()
        _force_escalation(mgr, "ocean", "HTTP Error 403")
        timeline = mgr.get_keyword_escalation_timeline()

        event = timeline["ocean"][0]
        assert "timestamp" in event
        assert "from_tier" in event
        assert "to_tier" in event
        assert "trigger_category" in event

    @pytest.mark.fast
    def test_timeline_event_values_correct_for_403(self):
        mgr = _make_manager()
        before = time.time()
        _force_escalation(mgr, "beach", "HTTP Error 403")
        after = time.time()

        timeline = mgr.get_keyword_escalation_timeline()
        event = timeline["beach"][0]

        assert event["from_tier"] == EscalationTier.IMPERSONATE_ONLY.value
        assert event["to_tier"] == EscalationTier.EXTRACTOR_ARGS.value
        assert event["trigger_category"] == "403"
        assert before <= event["timestamp"] <= after

    @pytest.mark.fast
    def test_timeline_event_trigger_category_429(self):
        mgr = _make_manager()
        _force_escalation(mgr, "wave", "HTTP Error 429 Too Many Requests")
        timeline = mgr.get_keyword_escalation_timeline()

        assert timeline["wave"][0]["trigger_category"] == "429"

    @pytest.mark.fast
    def test_timeline_event_trigger_category_bot_detection(self):
        mgr = _make_manager()
        _force_escalation(mgr, "sky", "verify you are human")
        timeline = mgr.get_keyword_escalation_timeline()

        assert timeline["sky"][0]["trigger_category"] == "bot_detection"

    @pytest.mark.fast
    def test_timeline_event_trigger_category_unknown_when_no_output(self):
        mgr = _make_manager()
        _force_escalation(mgr, "cloud", "")
        timeline = mgr.get_keyword_escalation_timeline()

        # Empty error_output → trigger_category = 'unknown'
        assert timeline["cloud"][0]["trigger_category"] == "unknown"

    @pytest.mark.fast
    def test_multiple_keywords_isolated_in_timeline(self):
        mgr = _make_manager()
        _force_escalation(mgr, "alpha", "HTTP Error 403")
        _force_escalation(mgr, "beta", "HTTP Error 429")
        timeline = mgr.get_keyword_escalation_timeline()

        assert "alpha" in timeline
        assert "beta" in timeline
        assert len(timeline["alpha"]) == 1
        assert len(timeline["beta"]) == 1
        assert timeline["alpha"][0]["trigger_category"] == "403"
        assert timeline["beta"][0]["trigger_category"] == "429"

    @pytest.mark.fast
    def test_empty_timeline_when_no_escalations(self):
        mgr = _make_manager()
        timeline = mgr.get_keyword_escalation_timeline()
        assert timeline == {}

    @pytest.mark.fast
    def test_timeline_records_multiple_escalations_for_same_keyword(self):
        """Escalate twice: Tier 1→2, then Tier 2→3."""
        mgr = _make_manager()
        # First escalation: 1 -> 2
        _force_escalation(mgr, "river", "HTTP Error 403")
        # Second escalation: 2 -> 3
        _force_escalation(mgr, "river", "HTTP Error 403")

        timeline = mgr.get_keyword_escalation_timeline()
        events = timeline["river"]
        assert len(events) == 2
        assert events[0]["from_tier"] == 1
        assert events[0]["to_tier"] == 2
        assert events[1]["from_tier"] == 2
        assert events[1]["to_tier"] == 3


# ---------------------------------------------------------------------------
# AC2: Timeline limited to last 20 events per keyword
# ---------------------------------------------------------------------------

class TestTimelineLimit:
    """Test that timeline is capped at 20 events per keyword."""

    @pytest.mark.fast
    def test_timeline_limited_to_20_events(self):
        """Record 25 escalation events, verify only last 20 returned."""
        mgr = _make_manager()

        # We need many escalation events. Since tiers only go 1→2→3,
        # we'll inject events directly into the internal timeline.
        keyword = "prolific"
        mgr._escalation_timeline[keyword] = []
        for i in range(25):
            mgr._escalation_timeline[keyword].append({
                'timestamp': time.time() + i,
                'from_tier': 1,
                'to_tier': 2,
                'trigger_category': '403',
            })

        timeline = mgr.get_keyword_escalation_timeline()
        events = timeline[keyword]

        assert len(events) == 20
        # Verify it's the LAST 20 (most recent)
        assert events[0]['timestamp'] == mgr._escalation_timeline[keyword][5]['timestamp']
        assert events[-1]['timestamp'] == mgr._escalation_timeline[keyword][24]['timestamp']

    @pytest.mark.fast
    def test_timeline_fewer_than_20_returns_all(self):
        mgr = _make_manager()
        mgr._escalation_timeline["sparse"] = [
            {'timestamp': time.time(), 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for _ in range(5)
        ]

        timeline = mgr.get_keyword_escalation_timeline()
        assert len(timeline["sparse"]) == 5

    @pytest.mark.fast
    def test_timeline_exactly_20_returns_all(self):
        mgr = _make_manager()
        mgr._escalation_timeline["exact"] = [
            {'timestamp': time.time(), 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for _ in range(20)
        ]

        timeline = mgr.get_keyword_escalation_timeline()
        assert len(timeline["exact"]) == 20


# ---------------------------------------------------------------------------
# AC3: get_hot_keywords() returns keywords with >3 escalations in last 30 min
# ---------------------------------------------------------------------------

class TestHotKeywords:
    """Test hot keyword detection."""

    @pytest.mark.fast
    def test_hot_keywords_returns_keywords_above_threshold(self):
        mgr = _make_manager()
        now = time.time()

        # 4 recent escalations for "trending" (above threshold of 3)
        mgr._escalation_timeline["trending"] = [
            {'timestamp': now - i * 10, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(4)
        ]

        hot = mgr.get_hot_keywords()
        assert len(hot) == 1
        assert hot[0]['keyword'] == "trending"
        assert hot[0]['escalation_count'] == 4

    @pytest.mark.fast
    def test_hot_keywords_sorted_by_count_descending(self):
        mgr = _make_manager()
        now = time.time()

        # "alpha" has 6 recent escalations
        mgr._escalation_timeline["alpha"] = [
            {'timestamp': now - i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(6)
        ]
        # "beta" has 5 recent escalations
        mgr._escalation_timeline["beta"] = [
            {'timestamp': now - i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(5)
        ]

        hot = mgr.get_hot_keywords()
        assert len(hot) == 2
        assert hot[0]['keyword'] == "alpha"
        assert hot[0]['escalation_count'] == 6
        assert hot[1]['keyword'] == "beta"
        assert hot[1]['escalation_count'] == 5


# ---------------------------------------------------------------------------
# AC4: get_hot_keywords() excludes keywords with <=3 or stale escalations
# ---------------------------------------------------------------------------

class TestHotKeywordsExclusion:
    """Test that get_hot_keywords() properly excludes non-hot keywords."""

    @pytest.mark.fast
    def test_excludes_keywords_with_3_or_fewer_escalations(self):
        mgr = _make_manager()
        now = time.time()

        # Exactly 3 escalations — NOT hot (needs >3)
        mgr._escalation_timeline["lukewarm"] = [
            {'timestamp': now - i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(3)
        ]

        hot = mgr.get_hot_keywords()
        assert len(hot) == 0

    @pytest.mark.fast
    def test_excludes_keywords_with_old_escalations(self):
        mgr = _make_manager()
        # 5 escalations from 2 hours ago (>30 min window)
        old_timestamp = time.time() - 7200
        mgr._escalation_timeline["stale"] = [
            {'timestamp': old_timestamp + i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(5)
        ]

        hot = mgr.get_hot_keywords()
        assert len(hot) == 0

    @pytest.mark.fast
    def test_excludes_mixed_old_and_recent_below_threshold(self):
        """4 total escalations but only 2 are recent — not hot."""
        mgr = _make_manager()
        now = time.time()

        mgr._escalation_timeline["mixed"] = [
            # 2 old events (outside 30 min window)
            {'timestamp': now - 3600, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'},
            {'timestamp': now - 3500, 'from_tier': 2, 'to_tier': 3, 'trigger_category': '403'},
            # 2 recent events (within 30 min window)
            {'timestamp': now - 100, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '429'},
            {'timestamp': now - 50, 'from_tier': 2, 'to_tier': 3, 'trigger_category': '429'},
        ]

        hot = mgr.get_hot_keywords()
        assert len(hot) == 0

    @pytest.mark.fast
    def test_empty_timeline_returns_empty_list(self):
        mgr = _make_manager()
        hot = mgr.get_hot_keywords()
        assert hot == []

    @pytest.mark.fast
    def test_only_hot_keywords_returned_from_mixed_set(self):
        """Only "hot_one" should be returned, not "cold_one"."""
        mgr = _make_manager()
        now = time.time()

        mgr._escalation_timeline["hot_one"] = [
            {'timestamp': now - i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(5)
        ]
        mgr._escalation_timeline["cold_one"] = [
            {'timestamp': now - i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(2)
        ]

        hot = mgr.get_hot_keywords()
        assert len(hot) == 1
        assert hot[0]['keyword'] == "hot_one"


# ---------------------------------------------------------------------------
# AC5: Timeline and hot_keywords included in export_to_json()
# ---------------------------------------------------------------------------

class TestExportToJsonIntegration:
    """Test that timeline and hot_keywords appear in RateLimitMetrics.export_to_json()."""

    @pytest.mark.fast
    def test_export_includes_keyword_timelines(self):
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        mgr = _make_manager()
        _force_escalation(mgr, "export_kw", "HTTP Error 403")

        export = metrics.export_to_json(escalation_manager=mgr)

        assert "keyword_timelines" in export["escalation"]
        assert "export_kw" in export["escalation"]["keyword_timelines"]
        events = export["escalation"]["keyword_timelines"]["export_kw"]
        assert len(events) == 1
        assert events[0]["trigger_category"] == "403"

    @pytest.mark.fast
    def test_export_includes_hot_keywords(self):
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        mgr = _make_manager()
        now = time.time()

        # Inject 5 recent escalation events
        mgr._escalation_timeline["hot_export"] = [
            {'timestamp': now - i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(5)
        ]

        export = metrics.export_to_json(escalation_manager=mgr)

        assert "hot_keywords" in export["escalation"]
        hot = export["escalation"]["hot_keywords"]
        assert len(hot) == 1
        assert hot[0]["keyword"] == "hot_export"
        assert hot[0]["escalation_count"] == 5

    @pytest.mark.fast
    def test_export_without_escalation_manager_has_no_timeline_keys(self):
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        export = metrics.export_to_json()

        # Without escalation_manager, these keys should not be present
        assert "keyword_timelines" not in export["escalation"]
        assert "hot_keywords" not in export["escalation"]

    @pytest.mark.fast
    def test_export_with_empty_timeline(self):
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        mgr = _make_manager()

        export = metrics.export_to_json(escalation_manager=mgr)

        assert export["escalation"]["keyword_timelines"] == {}
        assert export["escalation"]["hot_keywords"] == []

    @pytest.mark.fast
    def test_export_timeline_respects_20_event_limit(self):
        from src.downloader.rate_limit_metrics import RateLimitMetrics

        metrics = RateLimitMetrics()
        mgr = _make_manager()

        # Inject 25 events
        mgr._escalation_timeline["overflow"] = [
            {'timestamp': time.time() + i, 'from_tier': 1, 'to_tier': 2, 'trigger_category': '403'}
            for i in range(25)
        ]

        export = metrics.export_to_json(escalation_manager=mgr)
        assert len(export["escalation"]["keyword_timelines"]["overflow"]) == 20


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------

class TestTimelineEdgeCases:
    """Edge cases and reset behavior."""

    @pytest.mark.fast
    def test_reset_all_clears_timeline(self):
        mgr = _make_manager()
        _force_escalation(mgr, "doomed", "HTTP Error 403")
        assert len(mgr.get_keyword_escalation_timeline()) > 0

        mgr.reset_all()

        assert mgr.get_keyword_escalation_timeline() == {}
        assert mgr.get_hot_keywords() == []

    @pytest.mark.fast
    def test_timeline_event_from_budget_exhausted_escalation(self):
        """When budget is exhausted, escalation skips to FULL_BYPASS."""
        budget = MagicMock()
        budget.can_rotate.return_value = False
        budget.record_attempt = MagicMock()

        config = FakeExtractorArgsConfig(cooldown_seconds=0.0)
        mgr = EscalationManager(
            impersonation_manager=_make_impersonation_manager(),
            extractor_args_config=config,
            budget=budget,
        )

        _force_escalation(mgr, "budget_kw", "HTTP Error 403")
        timeline = mgr.get_keyword_escalation_timeline()

        assert len(timeline["budget_kw"]) == 1
        event = timeline["budget_kw"][0]
        assert event["from_tier"] == EscalationTier.IMPERSONATE_ONLY.value
        assert event["to_tier"] == EscalationTier.FULL_BYPASS.value
        assert event["trigger_category"] == "403"
