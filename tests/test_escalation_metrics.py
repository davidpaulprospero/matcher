"""
Unit tests for EscalationMetrics — verifies metrics tracking independently
of EscalationManager.

US-82-004: Extract EscalationMetrics from EscalationManager.
"""

import time
import threading
import pytest

from src.downloader.escalation_metrics import EscalationMetrics
from src.downloader.types import EscalationTier


class TestEscalationMetricsRecording:
    """Test basic recording methods."""

    def test_record_success_increments_total(self):
        metrics = EscalationMetrics()
        metrics.record_success("kw1")
        metrics.record_success("kw1")
        summary = metrics.get_summary()
        assert summary['total_successes'] == 2
        assert summary['total_403s'] == 0

    def test_record_failure_increments_total(self):
        metrics = EscalationMetrics()
        metrics.record_failure("kw1")
        metrics.record_failure("kw2")
        summary = metrics.get_summary()
        assert summary['total_403s'] == 2
        assert summary['total_successes'] == 0

    def test_record_escalation_tracks_timeline_and_counters(self):
        metrics = EscalationMetrics()
        metrics.record_escalation(
            "kw1",
            EscalationTier.IMPERSONATE_ONLY,
            EscalationTier.EXTRACTOR_ARGS,
            trigger_category='403',
        )
        summary = metrics.get_summary()
        assert summary['total_escalations'] == 1
        assert summary['escalations_per_tier']['EXTRACTOR_ARGS'] == 1
        assert summary['speed_escalations'] == 0

        timeline = metrics.get_timeline()
        assert 'kw1' in timeline
        assert len(timeline['kw1']) == 1
        event = timeline['kw1'][0]
        assert event['from_tier'] == EscalationTier.IMPERSONATE_ONLY.value
        assert event['to_tier'] == EscalationTier.EXTRACTOR_ARGS.value
        assert event['trigger_category'] == '403'

    def test_record_escalation_speed_triggered(self):
        metrics = EscalationMetrics()
        metrics.record_escalation(
            "kw1",
            EscalationTier.IMPERSONATE_ONLY,
            EscalationTier.EXTRACTOR_ARGS,
            trigger_category='slow_speed',
            speed_triggered=True,
        )
        summary = metrics.get_summary()
        assert summary['speed_escalations'] == 1
        assert summary['total_escalations'] == 1


class TestEscalationMetricsOutcomes:
    """Test tier outcome tracking and effectiveness analysis."""

    def test_record_outcome_tracks_attempts_and_successes(self):
        metrics = EscalationMetrics()
        metrics.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, True)
        metrics.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, False)
        metrics.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, True)

        effectiveness = metrics.get_tier_effectiveness()
        assert '403' in effectiveness
        # 2 successes out of 3 attempts
        assert effectiveness['403']['tier_1'] == pytest.approx(0.6667, abs=0.001)

    def test_get_tier_effectiveness_empty(self):
        metrics = EscalationMetrics()
        assert metrics.get_tier_effectiveness() == {}

    def test_get_tier_recommendations_skip_escalation(self):
        metrics = EscalationMetrics()
        # 9 successes out of 10 = 90% Tier 1 success
        for _ in range(9):
            metrics.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, True)
        metrics.record_outcome('429', EscalationTier.IMPERSONATE_ONLY, False)

        recommendations = metrics.get_tier_recommendations()
        assert 'skip escalation for 429' in recommendations

    def test_get_tier_recommendations_skip_tier2(self):
        metrics = EscalationMetrics()
        # Tier 2: 1 success out of 10 = 10%
        for _ in range(9):
            metrics.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, False)
        metrics.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, True)
        # Tier 3: 7 successes out of 10 = 70%
        for _ in range(7):
            metrics.record_outcome('403', EscalationTier.FULL_BYPASS, True)
        for _ in range(3):
            metrics.record_outcome('403', EscalationTier.FULL_BYPASS, False)

        recommendations = metrics.get_tier_recommendations()
        assert 'skip Tier 2 for 403' in recommendations


class TestEscalationMetricsTimeline:
    """Test timeline features: get_timeline, get_hot_keywords."""

    def test_get_timeline_limits_to_20_events(self):
        metrics = EscalationMetrics()
        for i in range(25):
            metrics.record_escalation(
                "kw1",
                EscalationTier.IMPERSONATE_ONLY,
                EscalationTier.EXTRACTOR_ARGS,
            )
        timeline = metrics.get_timeline()
        assert len(timeline['kw1']) == 20

    def test_get_hot_keywords_filters_by_count(self):
        metrics = EscalationMetrics()
        # kw1 gets 5 recent escalations (hot)
        for _ in range(5):
            metrics.record_escalation(
                "kw1",
                EscalationTier.IMPERSONATE_ONLY,
                EscalationTier.EXTRACTOR_ARGS,
            )
        # kw2 gets only 2 escalations (not hot, needs >3)
        for _ in range(2):
            metrics.record_escalation(
                "kw2",
                EscalationTier.IMPERSONATE_ONLY,
                EscalationTier.EXTRACTOR_ARGS,
            )

        hot = metrics.get_hot_keywords(window_seconds=60.0)
        assert len(hot) == 1
        assert hot[0]['keyword'] == 'kw1'
        assert hot[0]['escalation_count'] == 5

    def test_get_hot_keywords_empty_when_no_events(self):
        metrics = EscalationMetrics()
        assert metrics.get_hot_keywords() == []


class TestEscalationMetricsSerialization:
    """Test to_dict/from_dict round-trip."""

    def test_round_trip_preserves_state(self):
        metrics = EscalationMetrics()
        metrics.record_success("kw1")
        metrics.record_failure("kw2")
        metrics.record_escalation(
            "kw2",
            EscalationTier.IMPERSONATE_ONLY,
            EscalationTier.EXTRACTOR_ARGS,
            trigger_category='403',
        )
        metrics.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, True)

        data = metrics.to_dict()
        restored = EscalationMetrics.from_dict(data)

        assert restored.get_summary() == metrics.get_summary()
        assert restored.get_timeline() == metrics.get_timeline()
        assert restored.get_tier_effectiveness() == metrics.get_tier_effectiveness()

    def test_from_dict_handles_empty_data(self):
        metrics = EscalationMetrics.from_dict({})
        assert metrics.get_summary()['total_403s'] == 0

    def test_from_dict_handles_none(self):
        metrics = EscalationMetrics.from_dict(None)
        assert metrics.get_summary()['total_escalations'] == 0


class TestEscalationMetricsReset:
    """Test reset clears all state."""

    def test_reset_clears_everything(self):
        metrics = EscalationMetrics()
        metrics.record_success("kw1")
        metrics.record_failure("kw2")
        metrics.record_escalation(
            "kw1",
            EscalationTier.IMPERSONATE_ONLY,
            EscalationTier.EXTRACTOR_ARGS,
        )
        metrics.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, True)

        metrics.reset()

        summary = metrics.get_summary()
        assert summary['total_403s'] == 0
        assert summary['total_successes'] == 0
        assert summary['total_escalations'] == 0
        assert summary['speed_escalations'] == 0
        assert metrics.get_timeline() == {}
        assert metrics.get_tier_effectiveness() == {}


class TestEscalationMetricsThreadSafety:
    """Test thread safety of concurrent recording."""

    def test_concurrent_recording(self):
        metrics = EscalationMetrics()
        errors = []

        def record_successes():
            try:
                for _ in range(100):
                    metrics.record_success("kw1")
            except Exception as e:
                errors.append(e)

        def record_failures():
            try:
                for _ in range(100):
                    metrics.record_failure("kw2")
            except Exception as e:
                errors.append(e)

        def record_escalations():
            try:
                for _ in range(100):
                    metrics.record_escalation(
                        "kw3",
                        EscalationTier.IMPERSONATE_ONLY,
                        EscalationTier.EXTRACTOR_ARGS,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_successes),
            threading.Thread(target=record_failures),
            threading.Thread(target=record_escalations),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        summary = metrics.get_summary()
        assert summary['total_successes'] == 100
        assert summary['total_403s'] == 100
        assert summary['total_escalations'] == 100


class TestEscalationMetricsIndependence:
    """Verify EscalationMetrics works without any EscalationManager."""

    def test_standalone_full_workflow(self):
        """Complete metrics workflow without EscalationManager."""
        metrics = EscalationMetrics()

        # Record some attempts
        metrics.record_failure("video_abc")
        metrics.record_failure("video_abc")
        metrics.record_success("video_abc")

        # Record an escalation
        metrics.record_escalation(
            "video_abc",
            EscalationTier.IMPERSONATE_ONLY,
            EscalationTier.EXTRACTOR_ARGS,
            trigger_category='403',
        )

        # Record tier outcomes
        metrics.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, True)
        metrics.record_outcome('403', EscalationTier.EXTRACTOR_ARGS, False)

        # Verify summary
        summary = metrics.get_summary()
        assert summary['total_403s'] == 2
        assert summary['total_successes'] == 1
        assert summary['total_escalations'] == 1

        # Verify timeline
        timeline = metrics.get_timeline()
        assert len(timeline['video_abc']) == 1

        # Verify effectiveness
        effectiveness = metrics.get_tier_effectiveness()
        assert '403' in effectiveness
        assert effectiveness['403']['tier_2'] == 0.5

        # Verify serialization round-trip
        data = metrics.to_dict()
        restored = EscalationMetrics.from_dict(data)
        assert restored.get_summary() == summary
