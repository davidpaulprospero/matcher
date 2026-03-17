"""Retry queue statistics tracking (US-32-008).

US-144-009: Enhanced retry queue metrics and analytics:
- Per-keyword retry success rate tracking
- Retry latency histogram (time from first failure to success)
- Category distribution metrics (% retries by error category)
- Retry efficiency score (retries per successful download)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class RetryQueueStats:
    """Statistics tracker for retry queue operations."""

    pending: int = 0
    completed_ids: Set[str] = field(default_factory=set)
    failed_ids: Set[str] = field(default_factory=set)
    current_pass: int = 0
    max_passes: int = 2
    delay_seconds: float = 120.0
    total_added: int = 0
    total_retried: int = 0
    respect_circuit_breaker: bool = True
    circuit_breaker_wait_time: float = 0.0
    wait_for_cookie_cooldown: bool = True
    cookie_cooldown_wait_time: float = 0.0
    max_combined_wait_seconds: float = 300.0
    forced_retry: bool = False
    budget_state: Optional[Dict] = None
    failure_reasons: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    # US-144-009: Per-keyword retry success rate tracking
    # Maps keyword -> {"successes": int, "failures": int, "total_retries": int}
    _keyword_retry_stats: Dict[str, Dict[str, int]] = field(default_factory=dict, repr=False)

    # US-144-009: Retry latency histogram
    # Maps video_id -> first_failure_timestamp for latency calculation
    _first_failure_time: Dict[str, float] = field(default_factory=dict, repr=False)
    # List of retry latencies (time from first failure to success) in seconds
    retry_latencies: List[float] = field(default_factory=list, repr=False)

    # US-144-009: Category distribution metrics
    # Maps category -> count of retries in that category
    retries_by_category: Dict[str, int] = field(default_factory=dict)

    def get_summary(self) -> Dict:
        """Get retry queue statistics summary for reporting."""
        return {
            'enabled': self.enabled,
            'pending': self.pending,
            'completed': len(self.completed_ids),
            'failed': len(self.failed_ids),
            'current_pass': self.current_pass,
            'max_passes': self.max_passes,
            'delay_seconds': self.delay_seconds,
            'total_added': self.total_added,
            'total_retried': self.total_retried,
            'respect_circuit_breaker': self.respect_circuit_breaker,
            'circuit_breaker_wait_time': round(self.circuit_breaker_wait_time, 1),
            'wait_for_cookie_cooldown': self.wait_for_cookie_cooldown,
            'cookie_cooldown_wait_time': round(self.cookie_cooldown_wait_time, 1),
            'max_combined_wait_seconds': self.max_combined_wait_seconds,
            'forced_retry': self.forced_retry,
            'budget_state': self.budget_state,
        }

    # ==================== US-144-009: Enhanced Retry Metrics ====================

    def record_keyword_retry(self, keyword: str) -> None:
        """US-144-009: Record a retry attempt for a keyword.

        Args:
            keyword: The search keyword associated with this retry
        """
        if keyword not in self._keyword_retry_stats:
            self._keyword_retry_stats[keyword] = {
                'successes': 0,
                'failures': 0,
                'total_retries': 0
            }
        self._keyword_retry_stats[keyword]['total_retries'] += 1

    def record_keyword_retry_success(self, keyword: str) -> None:
        """US-144-009: Record a successful retry for a keyword.

        Args:
            keyword: The search keyword associated with this success
        """
        if keyword not in self._keyword_retry_stats:
            self._keyword_retry_stats[keyword] = {
                'successes': 0,
                'failures': 0,
                'total_retries': 0
            }
        self._keyword_retry_stats[keyword]['successes'] += 1

    def record_keyword_retry_failure(self, keyword: str) -> None:
        """US-144-009: Record a failed retry for a keyword.

        Args:
            keyword: The search keyword associated with this failure
        """
        if keyword not in self._keyword_retry_stats:
            self._keyword_retry_stats[keyword] = {
                'successes': 0,
                'failures': 0,
                'total_retries': 0
            }
        self._keyword_retry_stats[keyword]['failures'] += 1

    def get_keyword_retry_stats(self) -> Dict[str, Dict[str, float]]:
        """US-144-009: Get per-keyword retry success rate.

        Returns:
            Dict mapping keyword to success rate (0.0-1.0) and counts
        """
        result = {}
        for keyword, stats in self._keyword_retry_stats.items():
            total = stats['successes'] + stats['failures']
            if total > 0:
                success_rate = stats['successes'] / total
            else:
                success_rate = 0.0
            result[keyword] = {
                'success_rate': round(success_rate, 3),
                'successes': stats['successes'],
                'failures': stats['failures'],
                'total_retries': stats['total_retries']
            }
        return result

    def record_first_failure(self, video_id: str) -> None:
        """US-144-009: Record the first failure time for a video.

        Used to calculate retry latency (time from first failure to success).

        Args:
            video_id: The YouTube video ID that failed
        """
        if video_id not in self._first_failure_time:
            self._first_failure_time[video_id] = time.time()

    def record_retry_success_latency(self, video_id: str) -> Optional[float]:
        """US-144-009: Record retry latency when a video succeeds.

        Calculates the time from first failure to this success.

        Args:
            video_id: The YouTube video ID that succeeded

        Returns:
            Latency in seconds, or None if first failure not recorded
        """
        if video_id not in self._first_failure_time:
            return None

        latency = time.time() - self._first_failure_time[video_id]
        self.retry_latencies.append(latency)
        # Clean up tracking for this video
        del self._first_failure_time[video_id]
        return latency

    def get_retry_latency_stats(self) -> Dict[str, float]:
        """US-144-009: Get retry latency histogram statistics.

        Returns:
            Dict with min, max, avg, median latency in seconds
        """
        if not self.retry_latencies:
            return {
                'min': 0.0,
                'max': 0.0,
                'avg': 0.0,
                'median': 0.0,
                'sample_count': 0
            }

        sorted_latencies = sorted(self.retry_latencies)
        n = len(sorted_latencies)
        mid = n // 2

        if n % 2 == 0:
            median = (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2
        else:
            median = sorted_latencies[mid]

        return {
            'min': round(min(sorted_latencies), 2),
            'max': round(max(sorted_latencies), 2),
            'avg': round(sum(sorted_latencies) / n, 2),
            'median': round(median, 2),
            'sample_count': n
        }

    def record_retry_by_category(self, category: str) -> None:
        """US-144-009: Record a retry attempt by error category.

        Args:
            category: Error category (e.g., 'rate_limit', 'network', 'server')
        """
        self.retries_by_category[category] = self.retries_by_category.get(category, 0) + 1

    def get_category_distribution(self) -> Dict[str, float]:
        """US-144-009: Get category distribution as percentages.

        Returns:
            Dict mapping category to percentage of total retries (0.0-1.0)
        """
        total = sum(self.retries_by_category.values())
        if total == 0:
            return {}

        return {
            category: round(count / total, 3)
            for category, count in self.retries_by_category.items()
        }

    def get_retry_efficiency_score(self) -> float:
        """US-144-009: Calculate retry efficiency score.

        Returns the average number of retries needed per successful download.
        Lower is better - 1.0 means every retry succeeds on first attempt.

        Returns:
            Average retries per successful download, or 0.0 if no data
        """
        total_successes = len(self.completed_ids)
        total_retries = self.total_retried

        if total_successes == 0:
            return 0.0

        # Efficiency = retries / successes (lower is better)
        efficiency = total_retries / total_successes
        return round(efficiency, 3)

    def get_enhanced_metrics(self) -> Dict:
        """US-144-009: Get all enhanced retry metrics.

        Returns:
            Dict containing all new metrics for reporting
        """
        return {
            'keyword_retry_stats': self.get_keyword_retry_stats(),
            'retry_latency': self.get_retry_latency_stats(),
            'category_distribution': self.get_category_distribution(),
            'retry_efficiency_score': self.get_retry_efficiency_score(),
        }

    def get_failure_reasons(self) -> Dict[str, str]:
        """Get mapping of failed video IDs to their error messages."""
        return dict(self.failure_reasons)

    def get_retry_metrics(self) -> Dict:
        """Calculate retry-specific metrics for analysis."""
        total_processed = len(self.completed_ids) + len(self.failed_ids)
        success_rate = self.total_retried / self.total_added if self.total_added > 0 else 0.0
        retry_efficiency = len(self.completed_ids) / total_processed if total_processed > 0 else 0.0
        total_wait_time = self.circuit_breaker_wait_time + self.cookie_cooldown_wait_time
        passes_used_ratio = self.current_pass / self.max_passes if self.max_passes > 0 else 0.0

        # Include enhanced metrics from US-144-009
        enhanced = self.get_enhanced_metrics()

        return {
            'success_rate': round(success_rate, 3),
            'retry_efficiency': round(retry_efficiency, 3),
            'total_processed': total_processed,
            'total_wait_time': round(total_wait_time, 1),
            'passes_used_ratio': round(passes_used_ratio, 2),
            'had_forced_retry': self.forced_retry,
            # US-144-009 enhanced metrics
            'keyword_retry_stats': enhanced['keyword_retry_stats'],
            'retry_latency': enhanced['retry_latency'],
            'category_distribution': enhanced['category_distribution'],
            'retry_efficiency_score': enhanced['retry_efficiency_score'],
        }

    def record_failure(self, video_id: str, error_message: str) -> None:
        """Record a failure reason for a video."""
        self.failure_reasons[video_id] = error_message

    def clear_failure(self, video_id: str) -> None:
        """Clear failure reason when a video succeeds."""
        self.failure_reasons.pop(video_id, None)

    def reset(self) -> None:
        """Reset all statistics for a new session."""
        self.pending = 0
        self.completed_ids.clear()
        self.failed_ids.clear()
        self.current_pass = 0
        self.total_added = 0
        self.total_retried = 0
        self.circuit_breaker_wait_time = 0.0
        self.cookie_cooldown_wait_time = 0.0
        self.forced_retry = False
        self.budget_state = None
        self.failure_reasons.clear()
        # US-144-009: Reset enhanced tracking
        self._keyword_retry_stats.clear()
        self._first_failure_time.clear()
        self.retry_latencies.clear()
        self.retries_by_category.clear()
