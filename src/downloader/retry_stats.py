"""Retry queue statistics tracking (US-32-008)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


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

        return {
            'success_rate': round(success_rate, 3),
            'retry_efficiency': round(retry_efficiency, 3),
            'total_processed': total_processed,
            'total_wait_time': round(total_wait_time, 1),
            'passes_used_ratio': round(passes_used_ratio, 2),
            'had_forced_retry': self.forced_retry,
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
