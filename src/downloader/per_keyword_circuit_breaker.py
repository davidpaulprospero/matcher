"""Per-keyword circuit breaker with coordinated global fallback.

Implements per-keyword failure tracking to isolate rate-limited keywords
while allowing other keywords to continue. Also includes a global fallback
that trips when >50% of keywords are rate-limited.

US-109-004: Per-keyword circuit breaker with coordinated global fallback.

Key features:
- Per-keyword failure tracking: each keyword has its own circuit breaker
- Per-keyword pause duration: based on keyword's own failure history
- Global fallback: trips when >50% keywords are rate-limited
- Integration with CircuitBreakerCoordinator for cross-component awareness
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase
from src.downloader.pause_calculator import PauseCalculator, PauseContext
from src.logging_templates import log_rate_limit

if TYPE_CHECKING:
    from .circuit_breaker import CircuitBreakerCoordinator
    from .speed_tracker import PerKeywordSpeedTracker

logger = logging.getLogger(__name__)


def _mock_sleep(delay: float) -> None:
    """Apply mock delay for circuit breaker pauses in test mode.

    When MOCK_RATE_LIMITS=1 env var is set, sleeps for 0.01s instead of actual delay.

    Args:
        delay: The intended delay in seconds (used for logging)
    """
    env_value = os.environ.get('MOCK_RATE_LIMITS', '').lower()
    if env_value in ('1', 'true', 'yes'):
        mock_delay = float(os.environ.get('MOCK_DELAY_SECONDS', '0.01'))
        if mock_delay > 0:
            logger.debug(f"Mock circuit breaker: sleeping {mock_delay:.3f}s instead of {delay:.1f}s")
            time.sleep(mock_delay)
    else:
        time.sleep(delay)


@dataclass
class PerKeywordCircuitBreakerConfig:
    """Configuration for per-keyword circuit breaker.

    Attributes:
        enabled: Enable per-keyword circuit breaker tracking
        consecutive_failures_threshold: Failures per keyword before its circuit trips
        pause_seconds: Base pause duration per keyword (can be scaled by history)
        max_pause_seconds: Maximum pause cap per keyword
        jitter_factor: Random jitter factor (0.0 to 1.0)
        global_fallback_threshold: Fraction of keywords rate-limited before global trip (0.5 = 50%)
        global_pause_seconds: Pause duration when global fallback trips
        global_max_pause_seconds: Maximum global pause cap
        history_window: Number of past pauses to consider for adaptive pause calculation
        # US-113-007: Speed-based circuit breaker triggering
        speed_threshold_mbps: Minimum speed in Mbps below which circuit breaker may trip (default: 0.5)
        sustained_degradation_threshold: Consecutive slow downloads before triggering circuit (default: 3)
        enable_speed_trigger: Enable speed-based circuit breaker triggering (default: True)
        # US-143-008: Auto-recovery with gradual reintroduction
        enable_recovery: Enable gradual traffic increase after circuit closes (default: True)
        recovery_max_requests: Maximum requests allowed during recovery phase (default: 3)
        recovery_backoff_base: Base for exponential backoff during recovery (default: 2.0)
        recovery_max_backoff: Maximum backoff multiplier during recovery (default: 8.0)
        recovery_success_threshold: Consecutive successes needed to consider recovery complete (default: 2)
        # US-144-004: State persistence
        persist_state: bool = True
        state_file_path: str = ""
        auto_save_on_state_change: bool = False
    """
    enabled: bool = True
    consecutive_failures_threshold: int = 3
    pause_seconds: float = 30.0
    max_pause_seconds: float = 120.0
    jitter_factor: float = 0.2
    global_fallback_threshold: float = 0.5  # 50%
    global_pause_seconds: float = 60.0
    global_max_pause_seconds: float = 300.0
    history_window: int = 5
    # US-113-007: Speed-based circuit breaker triggering
    speed_threshold_mbps: float = 0.5  # Mbps minimum
    sustained_degradation_threshold: int = 3  # Consecutive slow downloads
    enable_speed_trigger: bool = True
    # US-143-008: Auto-recovery with gradual reintroduction
    enable_recovery: bool = True
    recovery_max_requests: int = 3
    recovery_backoff_base: float = 2.0
    recovery_max_backoff: float = 8.0
    recovery_success_threshold: int = 2
    # US-144-004: State persistence
    persist_state: bool = True
    state_file_path: str = ""
    auto_save_on_state_change: bool = False


@dataclass
class KeywordCircuitState:
    """State for a single keyword's circuit breaker."""
    consecutive_failures: int = 0
    is_open: bool = False
    opened_at: Optional[float] = None
    total_trips: int = 0
    total_paused_seconds: float = 0.0
    pause_history: List[float] = field(default_factory=list)  # Recent pause durations
    # US-143-008: Recovery state
    is_recovering: bool = False  # Circuit is in recovery mode (half-open, gradual traffic)
    recovery_attempts: int = 0  # Number of recovery attempts
    recovery_requests_made: int = 0  # Requests made in current recovery cycle
    recovery_consecutive_successes: int = 0  # Consecutive successes during recovery
    recovery_backoff_multiplier: float = 1.0  # Exponential backoff multiplier
    recovery_successes: int = 0  # Total successful recoveries
    recovery_failures: int = 0  # Total failed recovery attempts


@dataclass
class GlobalCircuitState:
    """State for the global fallback circuit breaker."""
    is_open: bool = False
    opened_at: Optional[float] = None
    total_trips: int = 0
    total_paused_seconds: float = 0.0


class PerKeywordCircuitBreaker:
    """Per-keyword circuit breaker with global fallback coordination.

    Tracks failures independently per keyword and provides per-keyword pause
    durations based on each keyword's history. Also includes a global fallback
    that trips when >50% of keywords are rate-limited.

    Usage:
        cb = PerKeywordCircuitBreaker(config)

        # Before searching with a keyword
        cb.check_and_wait("python tutorial")

        # After search
        if search_failed:
            cb.record_failure("python tutorial")
        else:
            cb.record_success("python tutorial")

        # Check global fallback status
        if cb.is_global_tripped():
            cb.wait_global_recovery()
    """

    def __init__(
        self,
        config: Optional[PerKeywordCircuitBreakerConfig] = None,
        coordinator: Optional['CircuitBreakerCoordinator'] = None,
        speed_tracker: Optional['PerKeywordSpeedTracker'] = None,
    ):
        """Initialize the per-keyword circuit breaker.

        Args:
            config: Configuration. If None, uses defaults.
            coordinator: Optional CircuitBreakerCoordinator for cross-component coordination.
            speed_tracker: Optional PerKeywordSpeedTracker for speed-based circuit breaker triggering.
        """
        self._config = config or PerKeywordCircuitBreakerConfig()
        self._keyword_states: Dict[str, KeywordCircuitState] = {}
        self._global_state = GlobalCircuitState()
        self._pause_calculator = PauseCalculator()
        self._coordinator = coordinator
        self._speed_tracker = speed_tracker

    @property
    def config(self) -> PerKeywordCircuitBreakerConfig:
        return self._config

    def _get_keyword_state(self, keyword: str) -> KeywordCircuitState:
        """Get or create state for a keyword."""
        if keyword not in self._keyword_states:
            self._keyword_states[keyword] = KeywordCircuitState()
        return self._keyword_states[keyword]

    def _calculate_adaptive_pause(self, keyword_state: KeywordCircuitState) -> float:
        """Calculate adaptive pause based on keyword's own history.

        Uses exponential backoff based on the keyword's past pause durations.
        Keywords with more history get more accurate pause estimates.
        """
        base_pause = self._config.pause_seconds

        if not keyword_state.pause_history:
            return base_pause

        # Calculate average of recent pauses
        recent_pauses = keyword_state.pause_history[-self._config.history_window:]
        avg_pause = sum(recent_pauses) / len(recent_pauses)

        # Exponential backoff: increase pause based on trip count
        # Cap at 2x the average history pause
        multiplier = min(2.0, 1.0 + (keyword_state.total_trips * 0.1))
        adaptive_pause = min(avg_pause * multiplier, self._config.max_pause_seconds)

        # Ensure we don't regress below base pause
        return max(adaptive_pause, base_pause)

    def _apply_jitter(self, pause: float) -> float:
        """Apply jitter to pause duration."""
        if self._config.jitter_factor <= 0:
            return pause

        import random
        jitter_range = self._config.jitter_factor
        jitter_offset = random.uniform(-jitter_range, jitter_range)
        jittered_pause = pause * (1 + jitter_offset)

        # Cap at max
        return min(jittered_pause, self._config.max_pause_seconds)

    def _get_rate_limited_keyword_count(self) -> int:
        """Count keywords that are currently rate-limited (circuit open)."""
        return sum(1 for state in self._keyword_states.values() if state.is_open)

    def _get_active_keyword_count(self) -> int:
        """Count keywords that have been used (have state)."""
        return len(self._keyword_states)

    def check_and_wait(self, keyword: str) -> bool:
        """Check keyword's circuit state and wait if necessary.

        Args:
            keyword: The keyword to check

        Returns:
            True if search should proceed, False if circuit breaker is disabled.
        """
        if not self._config.enabled:
            return False

        # Check global fallback first
        if self._global_state.is_open:
            self._wait_global_recovery()
            # After global recovery, check keyword-specific state

        keyword_state = self._get_keyword_state(keyword)

        if not keyword_state.is_open:
            # US-169-005: Log per-keyword circuit breaker state (CLOSED)
            if keyword_state.consecutive_failures > 0:
                logger.debug(
                    f"Per-keyword CB state check: keyword '{keyword}' is CLOSED "
                    f"(consecutive_failures={keyword_state.consecutive_failures}), allowing search"
                )
            # Check if we're in recovery mode
            if self._config.enable_recovery and keyword_state.is_recovering:
                # US-143-008: Implement gradual traffic increase during recovery
                max_requests = self._config.recovery_max_requests
                if keyword_state.recovery_requests_made >= max_requests:
                    # Already made max requests in this cycle, apply backoff
                    backoff_pause = self._calculate_recovery_backoff(keyword_state)
                    logger.info(
                        f"Per-keyword CB RECOVERY: keyword '{keyword}' reached max requests "
                        f"({keyword_state.recovery_requests_made}/{max_requests}), backing off {backoff_pause:.1f}s"
                    )
                    _mock_sleep(backoff_pause)
                    # Reset for next cycle
                    keyword_state.recovery_requests_made = 0
                    keyword_state.recovery_attempts += 1
                else:
                    # Allow request, increment counter
                    keyword_state.recovery_requests_made += 1
                    logger.debug(
                        f"Per-keyword CB RECOVERY: keyword '{keyword}' request allowed "
                        f"({keyword_state.recovery_requests_made}/{max_requests})"
                    )
            return True

        # Keyword circuit is open - calculate and apply pause
        pause = self._calculate_adaptive_pause(keyword_state)
        pause = self._apply_jitter(pause)

        elapsed = time.time() - keyword_state.opened_at
        remaining = pause - elapsed

        if remaining > 0:
            # US-169-005: Log per-keyword circuit breaker state (OPEN -> transition)
            logger.info(
                f"Per-keyword CB OPEN: keyword '{keyword}' pausing {remaining:.1f}s "
                f"(elapsed={elapsed:.1f}s, total_pause={pause:.1f}s, "
                f"trip #{keyword_state.total_trips}, {keyword_state.consecutive_failures} failures)"
            )
            _mock_sleep(remaining)
            keyword_state.total_paused_seconds += remaining

        # US-169-005: Log state transition from OPEN to HALF_OPEN (recovery)
        logger.info(
            f"Per-keyword CB state transition: keyword '{keyword}' OPEN -> HALF_OPEN "
            f"(pause_duration={pause:.1f}s elapsed)"
        )

        # Transition to recovery mode (half-open with gradual traffic)
        keyword_state.is_open = False
        keyword_state.opened_at = None
        if self._config.enable_recovery:
            keyword_state.is_recovering = True
            keyword_state.recovery_requests_made = 0
            keyword_state.recovery_consecutive_successes = 0
            keyword_state.recovery_attempts = 0
            keyword_state.recovery_backoff_multiplier = 1.0
            logger.info(
                f"Per-keyword CB: pause complete for '{keyword}', entering recovery mode "
                f"(max {self._config.recovery_max_requests} requests/cycle, "
                f"need {self._config.recovery_success_threshold} consecutive successes)"
            )
        else:
            logger.debug(f"Per-keyword CB: pause complete for '{keyword}', allowing search (half-open)")

        # Log recovery from rate limit with duration
        log_rate_limit(
            logger,
            "per_keyword_circuit_breaker",
            "youtube_api",
            "recovered",
            keyword=keyword,
            pause_duration=remaining if remaining > 0 else 0
        )

        return True

    def _calculate_recovery_backoff(self, keyword_state: KeywordCircuitState) -> float:
        """Calculate exponential backoff delay for recovery phase.

        US-143-008: Implements exponential backoff for recovery attempts.

        Args:
            keyword_state: The keyword's circuit state

        Returns:
            Backoff delay in seconds
        """
        base_pause = self._config.pause_seconds
        # Exponential backoff: base * (backoff_base ^ attempts), capped at max_backoff
        backoff = min(
            base_pause * (self._config.recovery_backoff_base ** keyword_state.recovery_attempts),
            base_pause * self._config.recovery_max_backoff
        )
        return backoff

    def record_success(self, keyword: str) -> None:
        """Record a successful search for a keyword.

        Args:
            keyword: The keyword that succeeded
        """
        if not self._config.enabled:
            return

        keyword_state = self._get_keyword_state(keyword)

        # US-143-008: Track recovery success rate
        if keyword_state.is_recovering:
            keyword_state.recovery_consecutive_successes += 1
            keyword_state.recovery_successes += 1

            # Check if recovery is complete
            if keyword_state.recovery_consecutive_successes >= self._config.recovery_success_threshold:
                # Recovery successful - exit recovery mode
                keyword_state.is_recovering = False
                keyword_state.recovery_attempts = 0
                keyword_state.recovery_requests_made = 0
                keyword_state.recovery_consecutive_successes = 0
                keyword_state.consecutive_failures = 0
                logger.info(
                    f"Per-keyword CB RECOVERY COMPLETE: keyword '{keyword}' recovered successfully "
                    f"({keyword_state.recovery_successes} total successes during recovery)"
                )
            else:
                logger.debug(
                    f"Per-keyword CB RECOVERY: keyword '{keyword}' success "
                    f"({keyword_state.recovery_consecutive_successes}/{self._config.recovery_success_threshold} needed)"
                )
        elif keyword_state.consecutive_failures > 0:
            logger.debug(
                f"Per-keyword CB: keyword '{keyword}' succeeded after "
                f"{keyword_state.consecutive_failures} failures, resetting"
            )

        keyword_state.consecutive_failures = 0
        keyword_state.is_open = False
        keyword_state.opened_at = None

    def record_failure(self, keyword: str) -> bool:
        """Record a search failure for a keyword.

        Args:
            keyword: The keyword that failed

        Returns:
            True if keyword's circuit tripped (opened), False otherwise.
        """
        if not self._config.enabled:
            return False

        keyword_state = self._get_keyword_state(keyword)
        keyword_state.consecutive_failures += 1

        # US-143-008: Track recovery failure rate
        if keyword_state.is_recovering:
            keyword_state.recovery_failures += 1
            keyword_state.recovery_consecutive_successes = 0  # Reset consecutive successes
            # Increase backoff multiplier on recovery failure
            keyword_state.recovery_backoff_multiplier = min(
                keyword_state.recovery_backoff_multiplier * self._config.recovery_backoff_base,
                self._config.recovery_max_backoff
            )
            logger.warning(
                f"Per-keyword CB RECOVERY FAILED: keyword '{keyword}' failed during recovery "
                f"(attempt #{keyword_state.recovery_attempts}, backoff multiplier: {keyword_state.recovery_backoff_multiplier:.1f}x)"
            )
            # Reset recovery state to open - will go through pause again
            keyword_state.is_recovering = False

        threshold = self._config.consecutive_failures_threshold
        logger.debug(
            f"Per-keyword CB: keyword '{keyword}' failure "
            f"({keyword_state.consecutive_failures}/{threshold})"
        )

        # Check keyword-specific threshold
        if keyword_state.consecutive_failures >= threshold:
            self._trip_keyword(keyword, keyword_state)
            self._check_global_fallback()
            return True

        return False

    def _trip_keyword(self, keyword: str, keyword_state: KeywordCircuitState) -> None:
        """Trip a keyword's circuit breaker."""
        keyword_state.is_open = True
        keyword_state.opened_at = time.time()
        keyword_state.total_trips += 1

        pause = self._calculate_adaptive_pause(keyword_state)
        pause = self._apply_jitter(pause)

        # Record in history
        keyword_state.pause_history.append(pause)
        if len(keyword_state.pause_history) > self._config.history_window:
            keyword_state.pause_history.pop(0)

        logger.info(
            f"Per-keyword CB TRIPPED: keyword '{keyword}' after "
            f"{keyword_state.consecutive_failures} consecutive failures. "
            f"Pausing for {pause:.0f}s (trip #{keyword_state.total_trips})"
        )
        log_rate_limit(
            logger,
            "per_keyword_circuit_breaker",
            "youtube_api",
            "tripped",
            keyword=keyword,
            pause_seconds=pause,
            trip_number=keyword_state.total_trips
        )

    # US-113-007: Speed-based circuit breaker triggering

    def set_speed_tracker(self, speed_tracker: 'PerKeywordSpeedTracker') -> None:
        """Set the speed tracker for speed-based circuit breaker triggering.

        Args:
            speed_tracker: PerKeywordSpeedTracker instance to use for speed monitoring.
        """
        self._speed_tracker = speed_tracker
        logger.debug("Per-keyword CB: speed tracker linked for speed-based triggering")

    def check_speed_and_trip(self, keyword: str) -> bool:
        """Check speed signals and trip circuit breaker if sustained slow downloads detected.

        Analyzes the speed tracker's data for a keyword and trips the circuit breaker
        if speed has fallen below the threshold for a sustained period (consecutive slow
        downloads exceeding sustained_degradation_threshold).

        This provides early detection of rate limiting before actual search failures occur.

        Args:
            keyword: The keyword to check speed for

        Returns:
            True if circuit was tripped due to speed degradation, False otherwise.
        """
        if not self._config.enabled:
            return False

        if not self._config.enable_speed_trigger:
            return False

        if self._speed_tracker is None:
            return False

        # Get rate limit signal from speed tracker for this keyword
        signal = self._speed_tracker.detect_rate_limit_signals(keyword)

        if not signal.detected:
            return False

        # Check if sustained degradation threshold is met
        threshold = self._config.sustained_degradation_threshold
        if signal.consecutive_slow_count < threshold:
            logger.debug(
                f"Per-keyword CB: keyword '{keyword}' has {signal.consecutive_slow_count} "
                f"slow downloads, need {threshold} to trigger circuit"
            )
            return False

        # Trip the circuit breaker due to sustained slow downloads
        keyword_state = self._get_keyword_state(keyword)
        self._trip_keyword(keyword, keyword_state)
        self._check_global_fallback()

        logger.warning(
            f"Per-keyword CB TRIPPED BY SPEED: keyword '{keyword}' had "
            f"{signal.consecutive_slow_count} consecutive slow downloads "
            f"(below {self._config.speed_threshold_mbps} Mbps). "
            f"Recent speeds: {signal.recent_speeds[-5:] if signal.recent_speeds else []}"
        )

        return True

    def check_early_warning_and_trip(self, keyword: str) -> bool:
        """Check early warning signals and trip circuit breaker proactively.

        This method checks for early warnings (30%+ speed degradation) and can
        trip the circuit breaker before sustained slow downloads occur. This provides
        proactive protection against rate limiting.

        Args:
            keyword: The keyword to check early warning for

        Returns:
            True if circuit was tripped due to early warning, False otherwise.
        """
        if not self._config.enabled:
            return False

        if not self._config.enable_speed_trigger:
            return False

        if self._speed_tracker is None:
            return False

        # Get early warning signal
        warning = self._speed_tracker.detect_early_warning(keyword)

        if not warning.detected:
            return False

        # Only trip for moderate or severe warnings
        if warning.warning_level not in ('moderate', 'severe'):
            logger.debug(
                f"Per-keyword CB: early warning for '{keyword}' at level '{warning.warning_level}', "
                f"not severe enough to trip"
            )
            return False

        # Trip the circuit breaker due to early warning
        keyword_state = self._get_keyword_state(keyword)
        self._trip_keyword(keyword, keyword_state)
        self._check_global_fallback()

        logger.warning(
            f"Per-keyword CB TRIPPED BY EARLY WARNING: keyword '{keyword}' had "
            f"{warning.degradation_percentage*100:.1f}% speed degradation. "
            f"Level: {warning.warning_level}, Baseline: {warning.baseline_speed_mbps:.2f} MB/s, "
            f"Current: {warning.current_speed_mbps:.2f} MB/s"
        )

        return True

    def check_anomaly_and_trip(self, keyword: str) -> bool:
        """Check for speed anomalies and trip circuit breaker if sudden drop detected.

        This method detects sudden drops in speed (anomalies) and can trip the
        circuit breaker proactively when a significant anomaly is detected.

        Args:
            keyword: The keyword to check anomaly for

        Returns:
            True if circuit was tripped due to anomaly, False otherwise.
        """
        if not self._config.enabled:
            return False

        if not self._config.enable_speed_trigger:
            return False

        if self._speed_tracker is None:
            return False

        # Get anomaly signal
        anomaly = self._speed_tracker.detect_anomaly(keyword)

        # Only react to sudden drops, not spikes
        if not anomaly.is_anomalous or anomaly.anomaly_type != 'sudden_drop':
            return False

        # Only trip for severe anomalies (z-score < -2.5)
        if anomaly.z_score > -2.5:
            logger.debug(
                f"Per-keyword CB: anomaly for '{keyword}' z-score={anomaly.z_score:.2f}, "
                f"not severe enough to trip"
            )
            return False

        # Trip the circuit breaker due to anomaly
        keyword_state = self._get_keyword_state(keyword)
        self._trip_keyword(keyword, keyword_state)
        self._check_global_fallback()

        logger.warning(
            f"Per-keyword CB TRIPPED BY ANOMALY: keyword '{keyword}' had sudden drop "
            f"(z-score={anomaly.z_score:.2f}). Current: {anomaly.current_speed_mbps:.2f} MB/s, "
            f"Baseline: {anomaly.baseline_mean_mbps:.2f}±{anomaly.baseline_std_mbps:.2f} MB/s"
        )

        return True

    def check_all_keywords_early_warning(self) -> Dict[str, bool]:
        """Check early warning signals for all tracked keywords.

        Returns:
            Dict mapping keyword to whether circuit was tripped for that keyword.
        """
        results = {}

        if self._speed_tracker is None:
            return results

        for keyword in self._speed_tracker.get_keywords():
            results[keyword] = self.check_early_warning_and_trip(keyword)

        return results

    def check_all_keywords_anomaly(self) -> Dict[str, bool]:
        """Check anomaly signals for all tracked keywords.

        Returns:
            Dict mapping keyword to whether circuit was tripped for that keyword.
        """
        results = {}

        if self._speed_tracker is None:
            return results

        for keyword in self._speed_tracker.get_keywords():
            results[keyword] = self.check_anomaly_and_trip(keyword)

        return results

    def get_speed_warning_metrics(self, keyword: str) -> Dict[str, Any]:
        """Get speed warning metrics for a keyword.

        Args:
            keyword: The keyword to get metrics for

        Returns:
            Dict with early warning and anomaly metrics
        """
        if self._speed_tracker is None:
            return {}

        metrics = self._speed_tracker.get_warning_metrics(keyword)

        return {
            'early_warnings_issued': metrics.early_warnings_issued,
            'warnings_by_level': metrics.warnings_by_level,
            'anomaly_count': metrics.anomaly_count,
            'adaptive_threshold_adjustments': metrics.adaptive_threshold_adjustments,
            'total_speed_checks': metrics.total_speed_checks,
            # Include adaptive threshold info
            'adaptive_threshold': self._speed_tracker.get_adaptive_threshold(keyword).__dict__ if keyword in self._speed_tracker.get_keywords() else None
        }

    def check_all_keywords_speed(self) -> Dict[str, bool]:
        """Check speed signals for all tracked keywords and trip circuits as needed.

        Returns:
            Dict mapping keyword to whether circuit was tripped for that keyword.
        """
        results = {}

        if self._speed_tracker is None:
            return results

        for keyword in self._speed_tracker.get_keywords():
            results[keyword] = self.check_speed_and_trip(keyword)

        return results

    def _check_global_fallback(self) -> None:
        """Check if global fallback should trip (>50% keywords rate-limited)."""
        if self._global_state.is_open:
            return

        active_count = self._get_active_keyword_count()
        if active_count == 0:
            return

        rate_limited_count = self._get_rate_limited_keyword_count()
        rate_limited_pct = rate_limited_count / active_count

        if rate_limited_pct > self._config.global_fallback_threshold:
            self._trip_global()

    def _trip_global(self) -> None:
        """Trip the global fallback circuit."""
        self._global_state.is_open = True
        self._global_state.opened_at = time.time()
        self._global_state.total_trips += 1

        rate_limited = self._get_rate_limited_keyword_count()
        active = self._get_active_keyword_count()
        rate_limited_pct = rate_limited / active if active > 0 else 0

        logger.warning(
            f"GLOBAL FALLBACK CB TRIPPED: {rate_limited}/{active} keywords "
            f"({rate_limited_pct:.0%}) rate-limited, exceeding {self._config.global_fallback_threshold:.0%} threshold. "
            f"Global pause: {self._config.global_pause_seconds:.0f}s"
        )
        log_rate_limit(
            logger,
            "per_keyword_circuit_breaker",
            "youtube_api",
            "tripped",
            keyword="__global__",
            rate_limited_count=rate_limited,
            active_count=active,
            rate_limited_pct=rate_limited_pct,
            pause_seconds=self._config.global_pause_seconds
        )

        # Propagate to coordinator if available
        if self._coordinator:
            self._coordinator.propagate_trip("per_keyword_global")

    def _wait_global_recovery(self) -> None:
        """Wait for global fallback circuit to recover."""
        if not self._global_state.is_open:
            return

        pause = self._config.global_pause_seconds
        elapsed = time.time() - self._global_state.opened_at
        remaining = pause - elapsed

        if remaining > 0:
            logger.warning(
                f"GLOBAL FALLBACK CB OPEN: pausing {remaining:.1f}s "
                f"(trip #{self._global_state.total_trips})"
            )
            _mock_sleep(remaining)
            self._global_state.total_paused_seconds += remaining

        # Transition to half-open
        self._global_state.is_open = False
        self._global_state.opened_at = None
        logger.info("Global fallback CB: pause complete, resuming keyword searches")

        # Log recovery from rate limit with duration
        log_rate_limit(
            logger,
            "per_keyword_circuit_breaker",
            "youtube_api",
            "recovered",
            keyword="__global__",
            pause_duration=remaining if remaining > 0 else 0
        )

    def is_global_tripped(self) -> bool:
        """Check if global fallback circuit is currently tripped."""
        return self._global_state.is_open

    def is_keyword_tripped(self, keyword: str) -> bool:
        """Check if a specific keyword's circuit is currently tripped."""
        keyword_state = self._get_keyword_state(keyword)
        return keyword_state.is_open

    def get_keyword_stats(self, keyword: str) -> dict:
        """Get statistics for a specific keyword.

        Args:
            keyword: The keyword to get stats for

        Returns:
            Dict containing:
            - keyword: The keyword
            - consecutive_failures: Current consecutive failure count
            - is_open: Whether circuit is currently open
            - total_trips: Total number of times the keyword's circuit has tripped
            - total_paused_seconds: Total seconds spent paused for this keyword
            - average_pause_duration: Average pause duration in seconds
            - pause_history: List of recent pause durations
            - failure_threshold: Configured failure threshold for this keyword
            - is_recovering: Whether circuit is in recovery mode (US-143-008)
            - recovery_successes: Total successful recoveries (US-143-008)
            - recovery_failures: Total failed recovery attempts (US-143-008)
            - recovery_success_rate: Success rate during recovery (US-143-008)
        """
        keyword_state = self._get_keyword_state(keyword)

        # Calculate average pause duration
        avg_pause = 0.0
        if keyword_state.pause_history:
            avg_pause = sum(keyword_state.pause_history) / len(keyword_state.pause_history)

        # Calculate recovery success rate
        recovery_total = keyword_state.recovery_successes + keyword_state.recovery_failures
        recovery_success_rate = (
            keyword_state.recovery_successes / recovery_total if recovery_total > 0 else 0.0
        )

        return {
            'keyword': keyword,
            'consecutive_failures': keyword_state.consecutive_failures,
            'is_open': keyword_state.is_open,
            'total_trips': keyword_state.total_trips,
            'total_paused_seconds': round(keyword_state.total_paused_seconds, 1),
            'average_pause_duration': round(avg_pause, 2),
            'pause_history': keyword_state.pause_history,
            'failure_threshold': self._config.consecutive_failures_threshold,
            # US-143-008: Recovery metrics
            'is_recovering': keyword_state.is_recovering,
            'recovery_successes': keyword_state.recovery_successes,
            'recovery_failures': keyword_state.recovery_failures,
            'recovery_success_rate': round(recovery_success_rate, 3),
        }

    def get_global_stats(self) -> dict:
        """Get statistics for the global fallback circuit.

        Returns:
            Dict containing:
            - is_open: Whether global fallback circuit is currently open
            - total_trips: Total number of times global fallback has tripped
            - total_paused_seconds: Total seconds spent paused for global fallback
            - average_pause_duration: Average pause duration in seconds
            - active_keywords: Total number of unique keywords tracked
            - rate_limited_keywords: Number of keywords currently rate-limited
            - rate_limited_pct: Percentage of keywords rate-limited (triggers global trip at threshold)
            - fallback_threshold: Configured threshold for global fallback (e.g., 0.5 = 50%)
            - is_tripped: Whether global fallback is currently open (alias for is_open)
        """
        active_count = self._get_active_keyword_count()
        rate_limited_count = self._get_rate_limited_keyword_count()

        # Calculate average pause duration
        avg_pause = 0.0
        if self._global_state.total_trips > 0:
            avg_pause = self._global_state.total_paused_seconds / self._global_state.total_trips

        return {
            'is_open': self._global_state.is_open,
            'total_trips': self._global_state.total_trips,
            'total_paused_seconds': round(self._global_state.total_paused_seconds, 1),
            'average_pause_duration': round(avg_pause, 2),
            'active_keywords': active_count,
            'rate_limited_keywords': rate_limited_count,
            'rate_limited_pct': (
                rate_limited_count / active_count if active_count > 0 else 0
            ),
            'fallback_threshold': self._config.global_fallback_threshold,
            'is_tripped': self._global_state.is_open,
        }

    def get_all_health_metrics(self) -> dict:
        """Get health metrics for all keywords and global fallback.

        Returns:
            Dict containing:
            - format: 'circuit_breaker_health_metrics_v1'
            - generated_at: ISO timestamp of when metrics were generated
            - enabled: Whether per-keyword circuit breaker is enabled
            - global: Global fallback circuit metrics (see get_global_stats)
            - keywords: Dict of keyword -> keyword-specific metrics (see get_keyword_stats)
            - summary: Aggregate summary across all keywords

        Metrics Schema:
            {
                "format": "circuit_breaker_health_metrics_v1",
                "generated_at": "2026-02-18T12:00:00Z",
                "enabled": true,
                "global": {
                    "is_open": false,
                    "total_trips": 5,
                    "total_paused_seconds": 150.0,
                    "average_pause_duration": 30.0,
                    "active_keywords": 10,
                    "rate_limited_keywords": 2,
                    "rate_limited_pct": 0.2,
                    "fallback_threshold": 0.5,
                    "is_tripped": false
                },
                "keywords": {
                    "python tutorial": {
                        "keyword": "python tutorial",
                        "consecutive_failures": 0,
                        "is_open": false,
                        "total_trips": 3,
                        "total_paused_seconds": 90.0,
                        "average_pause_duration": 30.0,
                        "pause_history": [30.0, 30.0, 30.0],
                        "failure_threshold": 3
                    }
                },
                "summary": {
                    "total_keywords": 10,
                    "total_trips": 15,
                    "total_paused_seconds": 450.0,
                    "keywords_tripped": 2,
                    "average_keyword_trips": 1.5
                }
            }
        """
        global_stats = self.get_global_stats()
        keyword_stats = {
            keyword: self.get_keyword_stats(keyword)
            for keyword in self._keyword_states
        }

        # Calculate summary
        total_trips = sum(ks['total_trips'] for ks in keyword_stats.values())
        total_paused = sum(ks['total_paused_seconds'] for ks in keyword_stats.values())
        keywords_tripped = sum(1 for ks in keyword_stats.values() if ks['is_open'])
        keyword_count = len(keyword_stats)

        # US-143-008: Add state transition events to metrics
        state_transitions = self._get_state_transitions(keyword_stats)

        from datetime import datetime
        return {
            'format': 'circuit_breaker_health_metrics_v1',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'enabled': self._config.enabled,
            'recovery_enabled': self._config.enable_recovery,
            'global': global_stats,
            'keywords': keyword_stats,
            'summary': {
                'total_keywords': keyword_count,
                'total_trips': total_trips,
                'total_paused_seconds': round(total_paused, 1),
                'keywords_tripped': keywords_tripped,
                'average_keyword_trips': round(total_trips / keyword_count, 2) if keyword_count > 0 else 0.0,
            },
            'state_transitions': state_transitions,
        }

    def _get_state_transitions(self, keyword_stats: dict) -> dict:
        """Get state transition events for all keywords.

        US-143-008: Adds circuit state transition events to metrics.

        Returns:
            Dict containing state transition counts by type
        """
        total_recoveries = sum(1 for ks in keyword_stats.values() if ks.get('recovery_successes', 0) > 0)
        total_recovery_failures = sum(1 for ks in keyword_stats.values() if ks.get('recovery_failures', 0) > 0)

        # Count keywords currently in each state
        closed_count = sum(1 for ks in keyword_stats.values() if not ks['is_open'] and not ks.get('is_recovering', False))
        open_count = sum(1 for ks in keyword_stats.values() if ks['is_open'])
        recovering_count = sum(1 for ks in keyword_stats.values() if ks.get('is_recovering', False))

        return {
            'total_recovery_attempts': total_recoveries,
            'total_recovery_failures': total_recovery_failures,
            'keywords_in_closed_state': closed_count,
            'keywords_in_open_state': open_count,
            'keywords_in_recovering_state': recovering_count,
        }

    def export_as_json(self) -> str:
        """Export circuit breaker health metrics as JSON string.

        Returns:
            JSON string containing all circuit breaker health metrics
        """
        import json
        return json.dumps(self.get_all_health_metrics(), indent=2)

    def get_per_keyword_failure_counts(self) -> Dict[str, int]:
        """Get failure counts for all tracked keywords."""
        return {
            keyword: state.consecutive_failures
            for keyword, state in self._keyword_states.items()
        }

    def reset_keyword(self, keyword: str) -> None:
        """Manually reset a keyword's circuit breaker.

        Args:
            keyword: The keyword to reset
        """
        if keyword in self._keyword_states:
            keyword_state = self._keyword_states[keyword]
            keyword_state.consecutive_failures = 0
            keyword_state.is_open = False
            keyword_state.opened_at = None
            logger.debug(f"Per-keyword CB: manually reset keyword '{keyword}'")

    def reset_global(self) -> None:
        """Manually reset the global fallback circuit."""
        self._global_state.is_open = False
        self._global_state.opened_at = None
        logger.debug("Global fallback CB: manually reset")

    def reset_all(self) -> None:
        """Reset all keyword circuits and global fallback."""
        for keyword_state in self._keyword_states.values():
            keyword_state.consecutive_failures = 0
            keyword_state.is_open = False
            keyword_state.opened_at = None
        self.reset_global()
        logger.debug("Per-keyword CB: all circuits manually reset")

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence."""
        return {
            'keyword_states': {
                keyword: {
                    'consecutive_failures': state.consecutive_failures,
                    'total_trips': state.total_trips,
                    'total_paused_seconds': state.total_paused_seconds,
                    'pause_history': state.pause_history,
                    # US-143-008: Recovery state
                    'is_recovering': state.is_recovering,
                    'recovery_successes': state.recovery_successes,
                    'recovery_failures': state.recovery_failures,
                }
                for keyword, state in self._keyword_states.items()
            },
            'global_state': {
                'total_trips': self._global_state.total_trips,
                'total_paused_seconds': self._global_state.total_paused_seconds,
            },
        }

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore state from checkpoint dictionary."""
        if not data:
            return

        # Restore keyword states
        keyword_data = data.get('keyword_states', {})
        for keyword, state_data in keyword_data.items():
            state = self._get_keyword_state(keyword)
            state.total_trips = state_data.get('total_trips', 0)
            state.total_paused_seconds = state_data.get('total_paused_seconds', 0.0)
            state.pause_history = state_data.get('pause_history', [])
            # US-143-008: Restore recovery state
            state.recovery_successes = state_data.get('recovery_successes', 0)
            state.recovery_failures = state_data.get('recovery_failures', 0)
            # Reset failure state on restore
            state.consecutive_failures = 0
            state.is_open = False
            state.is_recovering = False
            state.opened_at = None

        # Restore global state
        global_data = data.get('global_state', {})
        self._global_state.total_trips = global_data.get('total_trips', 0)
        self._global_state.total_paused_seconds = global_data.get('total_paused_seconds', 0.0)
        self._global_state.is_open = False
        self._global_state.opened_at = None
