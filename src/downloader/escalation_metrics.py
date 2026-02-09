"""
Extracted metrics tracking for the 4-tier escalation system.

Separates metrics collection, timeline tracking, and tier effectiveness
analysis from the EscalationManager's core escalation decision logic.

This class is thread-safe: all public methods acquire the internal lock.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from .types import EscalationTier

logger = logging.getLogger(__name__)


class EscalationMetrics:
    """Tracks escalation metrics independently of escalation decisions.

    Records attempt outcomes, tier transitions, and computes effectiveness
    summaries. Can be used standalone for testing or injected into
    EscalationManager for production use.

    Thread-safe: all public methods use an internal lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_403s: int = 0
        self._total_successes: int = 0
        self._total_escalations: int = 0
        self._escalations_per_tier: Dict[str, int] = {}
        self._speed_escalations: int = 0
        # Per-keyword escalation timeline: keyword -> [{timestamp, from_tier, to_tier, trigger_category}]
        self._escalation_timeline: Dict[str, List[Dict]] = {}
        # Tier outcome tracking: {trigger_category: {tier_value: {'successes': N, 'attempts': N}}}
        self._tier_outcomes: Dict[str, Dict[int, Dict[str, int]]] = {}

    def record_attempt(
        self,
        keyword: str,
        tier: EscalationTier,
        success: bool,
        trigger_category: Optional[str] = None,
    ) -> None:
        """Record a download attempt outcome.

        Args:
            keyword: The download keyword or video ID.
            tier: The escalation tier used for this attempt.
            success: True if the download succeeded.
            trigger_category: Optional category from classify_trigger().
        """
        with self._lock:
            if success:
                self._total_successes += 1
            else:
                self._total_403s += 1

            # Record in tier outcomes if we have a trigger category
            if trigger_category:
                if trigger_category not in self._tier_outcomes:
                    self._tier_outcomes[trigger_category] = {}
                tier_val = tier.value
                if tier_val not in self._tier_outcomes[trigger_category]:
                    self._tier_outcomes[trigger_category][tier_val] = {
                        'successes': 0, 'attempts': 0,
                    }
                self._tier_outcomes[trigger_category][tier_val]['attempts'] += 1
                if success:
                    self._tier_outcomes[trigger_category][tier_val]['successes'] += 1

    def record_success(self, keyword: str) -> None:
        """Record a successful download (convenience wrapper).

        Args:
            keyword: The download keyword or video ID.
        """
        with self._lock:
            self._total_successes += 1

    def record_failure(self, keyword: str) -> None:
        """Record a failed download (convenience wrapper).

        Args:
            keyword: The download keyword or video ID.
        """
        with self._lock:
            self._total_403s += 1

    def record_escalation(
        self,
        keyword: str,
        from_tier: EscalationTier,
        to_tier: EscalationTier,
        trigger_category: Optional[str] = None,
        speed_triggered: bool = False,
    ) -> None:
        """Record an escalation event in the timeline.

        Args:
            keyword: The keyword that escalated.
            from_tier: Tier before escalation.
            to_tier: Tier after escalation.
            trigger_category: Category from classify_trigger() or None.
            speed_triggered: True if escalation was triggered by slow speed.
        """
        with self._lock:
            self._total_escalations += 1
            tier_name = to_tier.name
            self._escalations_per_tier[tier_name] = (
                self._escalations_per_tier.get(tier_name, 0) + 1
            )
            if speed_triggered:
                self._speed_escalations += 1

            # Record timeline event
            if keyword not in self._escalation_timeline:
                self._escalation_timeline[keyword] = []
            self._escalation_timeline[keyword].append({
                'timestamp': time.time(),
                'from_tier': from_tier.value,
                'to_tier': to_tier.value,
                'trigger_category': trigger_category or 'unknown',
            })

    def record_outcome(
        self, trigger_category: str, tier: EscalationTier, success: bool
    ) -> None:
        """Record a download outcome for tier effectiveness tracking.

        Each call records whether a download attempt at a specific escalation
        tier succeeded or failed for a given trigger category.

        Args:
            trigger_category: Category from classify_trigger() (e.g., '403', '429').
            tier: The escalation tier used for this attempt.
            success: True if the download succeeded, False if it failed.
        """
        with self._lock:
            if trigger_category not in self._tier_outcomes:
                self._tier_outcomes[trigger_category] = {}
            tier_val = tier.value
            if tier_val not in self._tier_outcomes[trigger_category]:
                self._tier_outcomes[trigger_category][tier_val] = {
                    'successes': 0, 'attempts': 0,
                }
            self._tier_outcomes[trigger_category][tier_val]['attempts'] += 1
            if success:
                self._tier_outcomes[trigger_category][tier_val]['successes'] += 1

    def get_timeline(self) -> Dict[str, List[Dict]]:
        """Get per-keyword escalation event timeline.

        Returns a dict keyed by keyword, where each value is a list of
        escalation event dicts with keys: timestamp, from_tier, to_tier,
        trigger_category. Limited to the last 20 events per keyword.

        Returns:
            Dict mapping keyword to list of event dicts.
        """
        with self._lock:
            result: Dict[str, List[Dict]] = {}
            for keyword, events in self._escalation_timeline.items():
                result[keyword] = list(events[-20:])
            return result

    def get_hot_keywords(self, window_seconds: float = 1800.0) -> List[Dict]:
        """Get keywords with frequent escalations in a recent time window.

        A keyword is "hot" if it has more than 3 escalation events within
        the specified window (default 30 minutes).

        Args:
            window_seconds: Time window in seconds (default: 1800 = 30 min).

        Returns:
            List of dicts with 'keyword' and 'escalation_count', sorted by
            count descending. Only includes keywords with >3 escalations.
        """
        cutoff = time.time() - window_seconds
        with self._lock:
            hot: List[Dict] = []
            for keyword, events in self._escalation_timeline.items():
                recent_count = sum(
                    1 for e in events if e['timestamp'] > cutoff
                )
                if recent_count > 3:
                    hot.append({
                        'keyword': keyword,
                        'escalation_count': recent_count,
                    })
            hot.sort(key=lambda x: x['escalation_count'], reverse=True)
            return hot

    def get_tier_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get success rate per trigger category per escalation tier.

        Returns:
            Dict mapping trigger_category to {tier_name: success_rate}.
        """
        with self._lock:
            if not self._tier_outcomes:
                return {}

            result: Dict[str, Dict[str, float]] = {}
            for category, tiers in self._tier_outcomes.items():
                tier_rates: Dict[str, float] = {}
                for tier_val, counts in tiers.items():
                    attempts = counts['attempts']
                    if attempts > 0:
                        rate = counts['successes'] / attempts
                        tier_rates[f'tier_{tier_val}'] = round(rate, 4)
                if tier_rates:
                    result[category] = tier_rates
            return result

    def get_tier_recommendations(self) -> List[str]:
        """Generate recommendations based on tier effectiveness data.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []
        effectiveness = self.get_tier_effectiveness()

        for category, tier_rates in effectiveness.items():
            tier_1_rate = tier_rates.get('tier_1', None)
            tier_2_rate = tier_rates.get('tier_2', None)
            tier_3_rate = tier_rates.get('tier_3', None)

            if tier_1_rate is not None and tier_1_rate > 0.80:
                recommendations.append(f"skip escalation for {category}")

            if (
                tier_2_rate is not None and tier_2_rate < 0.20
                and tier_3_rate is not None and tier_3_rate > 0.60
            ):
                recommendations.append(f"skip Tier 2 for {category}")

        return recommendations

    def get_summary(self) -> Dict:
        """Get escalation metrics summary.

        Returns:
            Dict with total_escalations, escalations_per_tier, total_403s,
            total_successes, speed_escalations.
        """
        with self._lock:
            return {
                'total_escalations': self._total_escalations,
                'escalations_per_tier': dict(self._escalations_per_tier),
                'total_403s': self._total_403s,
                'total_successes': self._total_successes,
                'speed_escalations': self._speed_escalations,
            }

    def to_dict(self) -> Dict:
        """Serialize metrics state for checkpoint persistence.

        Returns:
            Dict with timeline, tier_outcomes, and global counters.
        """
        with self._lock:
            timeline = {
                kw: list(events)
                for kw, events in self._escalation_timeline.items()
            }
            tier_outcomes = {
                cat: {
                    tier_val: dict(counts)
                    for tier_val, counts in tiers.items()
                }
                for cat, tiers in self._tier_outcomes.items()
            }
            return {
                'total_403s': self._total_403s,
                'total_successes': self._total_successes,
                'total_escalations': self._total_escalations,
                'escalations_per_tier': dict(self._escalations_per_tier),
                'speed_escalations': self._speed_escalations,
                'timeline': timeline,
                'tier_outcomes': tier_outcomes,
            }

    @classmethod
    def from_dict(cls, data: Dict) -> "EscalationMetrics":
        """Restore EscalationMetrics from checkpoint data.

        Args:
            data: Dict previously returned by to_dict() or from
                EscalationManager.to_dict() (backwards compatible).

        Returns:
            A new EscalationMetrics with restored state.
        """
        metrics = cls()
        if not data or not isinstance(data, dict):
            return metrics

        metrics._total_403s = data.get('total_403s', 0)
        metrics._total_successes = data.get('total_successes', 0)
        metrics._total_escalations = data.get('total_escalations', 0)
        metrics._escalations_per_tier = dict(data.get('escalations_per_tier', {}))
        metrics._speed_escalations = data.get('speed_escalations', 0)

        # Restore timeline
        saved_timeline = data.get('timeline', {})
        if isinstance(saved_timeline, dict):
            for kw, events in saved_timeline.items():
                if isinstance(events, list):
                    metrics._escalation_timeline[kw] = list(events)

        # Restore tier outcomes
        saved_outcomes = data.get('tier_outcomes', {})
        if isinstance(saved_outcomes, dict):
            for category, tiers in saved_outcomes.items():
                if isinstance(tiers, dict):
                    metrics._tier_outcomes[category] = {}
                    for tier_val, counts in tiers.items():
                        if isinstance(counts, dict):
                            try:
                                tier_key = int(tier_val)
                            except (ValueError, TypeError):
                                continue
                            metrics._tier_outcomes[category][tier_key] = {
                                'successes': counts.get('successes', 0),
                                'attempts': counts.get('attempts', 0),
                            }

        return metrics

    def reset(self) -> None:
        """Clear all metrics state."""
        with self._lock:
            self._total_403s = 0
            self._total_successes = 0
            self._total_escalations = 0
            self._escalations_per_tier.clear()
            self._speed_escalations = 0
            self._escalation_timeline.clear()
            self._tier_outcomes.clear()
