"""Circuit breaker for repeated search failures.

Implements the circuit breaker pattern to prevent hammering YouTube
when multiple consecutive searches fail. After threshold consecutive
failures, the circuit "trips" and pauses searching for a configured
duration before allowing new searches.

This protects against:
- Wasted API calls during widespread rate limiting
- Excessive retries that could worsen rate limit issues
- Unnecessarily slow pipeline execution during outages
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, List

from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase
from src.downloader.pause_calculator import PauseCalculator, PauseContext

if TYPE_CHECKING:
    from .escalation_manager import EscalationManager
    from .rate_limit_budget import RateLimitBudget
    from src.caption.circuit_breaker import CaptionCircuitBreaker

logger = logging.getLogger(__name__)


# US-89-009: Multi-circuit coordination
@dataclass
class CascadeRule:
    """A rule defining how one circuit breaker affects another."""
    source: str  # Source circuit breaker name (e.g., "search", "caption")
    target: str  # Target circuit breaker name (e.g., "download")
    on_trip: bool = True  # Propagate trip to target
    on_failure: bool = False  # Propagate failure count to target


class CircuitBreakerRegistry:
    """Registry to track all circuit breakers in the pipeline.

    US-89-009: Enables multi-circuit coordination across components.
    Allows circuit breakers to be registered and queried for health metrics,
    and supports cross-circuit trip propagation.

    Usage:
        registry = CircuitBreakerRegistry()
        registry.register("search", search_cb)
        registry.register("caption", caption_cb)
        registry.register("download", download_cb)

        # Get all health metrics
        all_metrics = registry.get_all_health_metrics()

        # Check if any circuit is tripped
        if registry.is_any_tripped():
            logger.warning(f"Circuit breaker(s) tripped: {registry.get_tripped_names()}")
    """

    def __init__(self) -> None:
        self._breakers: Dict[str, 'CircuitBreakerBase'] = {}

    def register(self, name: str, breaker: 'CircuitBreakerBase') -> None:
        """Register a circuit breaker with a given name."""
        self._breakers[name] = breaker
        logger.debug(f"CircuitBreakerRegistry: registered '{name}'")

    def unregister(self, name: str) -> None:
        """Unregister a circuit breaker by name."""
        if name in self._breakers:
            del self._breakers[name]
            logger.debug(f"CircuitBreakerRegistry: unregistered '{name}'")

    def get(self, name: str) -> Optional['CircuitBreakerBase']:
        """Get a circuit breaker by name, or None if not found."""
        return self._breakers.get(name)

    def get_all_health_metrics(self) -> Dict[str, dict]:
        """Get health metrics from all registered circuit breakers."""
        return {
            name: breaker.get_health_metrics()
            for name, breaker in self._breakers.items()
        }

    def is_any_tripped(self) -> bool:
        """Check if any circuit breaker is currently tripped (open)."""
        return any(breaker.is_open for breaker in self._breakers.values())

    def get_tripped_names(self) -> List[str]:
        """Get names of all tripped circuit breakers."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]

    def get_aggregate_stats(self) -> dict:
        """Get aggregate statistics across all circuit breakers."""
        total_trips = sum(
            cb.state.total_trips for cb in self._breakers.values()
        )
        total_paused = sum(
            cb.state.total_paused_seconds for cb in self._breakers.values()
        )
        tripped_count = len(self.get_tripped_names())

        return {
            'total_breakers': len(self._breakers),
            'tripped_count': tripped_count,
            'total_trips': total_trips,
            'total_paused_seconds': round(total_paused, 1),
            'is_any_tripped': self.is_any_tripped(),
        }


class CircuitBreakerCoordinator:
    """Singleton coordinator for multi-circuit breaker coordination.

    US-89-009: Manages cross-circuit trip propagation based on configurable
    cascade rules. When one circuit trips, it can trigger trips in other
    circuits based on configured rules.

    Default cascade rules:
    - search -> caption: trip on failure (existing US-61-003)
    - caption -> search: trip on failure (existing US-61-003)
    - search -> download: trip on trip (NEW)

    Usage:
        coordinator = CircuitBreakerCoordinator.get_instance()
        coordinator.set_registry(registry)
        coordinator.add_rule(CascadeRule(source="search", target="download", on_trip=True))

        # In each circuit breaker, call propagate when it trips
        coordinator.propagate_trip("search")
        coordinator.propagate_failure("search")
    """

    _instance: Optional['CircuitBreakerCoordinator'] = None
    _lock = None  # Will be initialized on first use

    def __init__(self) -> None:
        self._registry: Optional[CircuitBreakerRegistry] = None
        self._rules: List[CascadeRule] = []
        self._enabled: bool = True
        # Default cascade rules
        self._add_default_rules()

    @classmethod
    def get_instance(cls) -> 'CircuitBreakerCoordinator':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = CircuitBreakerCoordinator()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def _add_default_rules(self) -> None:
        """Add default cascade rules."""
        # Search -> caption: propagate failures (existing US-61-003 behavior)
        self._rules.append(CascadeRule(
            source="search",
            target="caption",
            on_failure=True,
            on_trip=True,
        ))
        # Caption -> search: propagate failures (existing US-61-003 behavior)
        self._rules.append(CascadeRule(
            source="caption",
            target="search",
            on_failure=True,
            on_trip=True,
        ))
        # Search -> download: propagate trip (NEW US-89-009)
        self._rules.append(CascadeRule(
            source="search",
            target="download",
            on_trip=True,
            on_failure=False,
        ))

    def set_registry(self, registry: CircuitBreakerRegistry) -> None:
        """Set the circuit breaker registry to coordinate."""
        self._registry = registry

    def add_rule(self, rule: CascadeRule) -> None:
        """Add a cascade rule."""
        self._rules.append(rule)
        logger.debug(
            f"CircuitBreakerCoordinator: added rule {rule.source} -> {rule.target}"
        )

    def clear_rules(self) -> None:
        """Clear all cascade rules."""
        self._rules.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable coordination."""
        self._enabled = enabled
        logger.debug(f"CircuitBreakerCoordinator: enabled={enabled}")

    def propagate_trip(self, source_name: str) -> None:
        """Propagate a trip event from source circuit breaker to targets."""
        if not self._enabled or self._registry is None:
            return

        for rule in self._rules:
            if rule.source == source_name and rule.on_trip:
                target_cb = self._registry.get(rule.target)
                if target_cb is not None:
                    # Trip the target circuit breaker
                    if not target_cb.state.is_open:
                        logger.info(
                            f"Coordinating cross-circuit trip: {source_name} -> {rule.target}"
                        )
                        target_cb.state.is_open = True
                        target_cb.state.opened_at = time.time()
                        target_cb.state.total_trips += 1

    def propagate_failure(self, source_name: str) -> None:
        """Propagate a failure event from source circuit breaker to targets."""
        if not self._enabled or self._registry is None:
            return

        for rule in self._rules:
            if rule.source == source_name and rule.on_failure:
                target_cb = self._registry.get(rule.target)
                if target_cb is not None:
                    # Increment failure count on target
                    target_cb.state.consecutive_failures += 1
                    logger.debug(
                        f"Coordinating cross-circuit failure: {source_name} -> {rule.target} "
                        f"(target failures: {target_cb.state.consecutive_failures})"
                    )

    def get_coordination_stats(self) -> dict:
        """Get coordination statistics."""
        return {
            'enabled': self._enabled,
            'rules_count': len(self._rules),
            'registry_set': self._registry is not None,
            'rules': [
                {'source': r.source, 'target': r.target,
                 'on_trip': r.on_trip, 'on_failure': r.on_failure}
                for r in self._rules
            ],
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for search failure circuit breaker.

    When multiple consecutive searches fail (no results or errors),
    the circuit breaker trips and pauses all searches for a duration.
    This prevents hammering YouTube during rate limit windows.

    Example with defaults:
      - 5 searches fail in a row → circuit trips
      - Wait 60 seconds before allowing new searches
      - On next successful search → circuit resets to closed state

    Download retry coordination:
      When block_download_retries is enabled (default), the download retry loop
      in _run_download_cmd will check the circuit breaker state before each retry
      attempt. If the circuit breaker is tripped during a retry sequence, the
      retry will wait for the circuit breaker to recover before continuing.
      The retry count is preserved across circuit breaker pauses.
    """
    # Enable/disable circuit breaker
    enabled: bool = True

    # Number of consecutive failures before circuit trips (opens)
    consecutive_failures_threshold: int = 5

    # Duration to pause after circuit trips (seconds)
    pause_seconds: float = 60.0

    # Block download retries when circuit breaker is tripped
    # When true, download retry loop waits for circuit breaker recovery
    block_download_retries: bool = True

    # Maximum pause duration cap (seconds) to prevent runaway pause scaling
    max_pause_seconds: float = 300.0

    # Jitter factor for randomizing pause durations (0.0 to 1.0)
    # Delay is computed as: base_delay * (1 + random.uniform(-jitter, +jitter))
    # Default 0.2 means ±20% randomization to prevent thundering herd
    jitter_factor: float = 0.2

    # Circuit breaker cascade (US-61-003): when enabled, failures propagate to
    # the caption circuit breaker (and vice versa) to speed up coordinated pausing
    # when YouTube is rate-limiting. Default: True.
    circuit_breaker_cascade: bool = True


@dataclass
class CircuitBreakerState(CircuitBreakerStateBase):
    """Internal state for circuit breaker."""
    pass


class CircuitBreakerBuilder:
    """Fluent builder for CircuitBreaker with validated dependency wiring.

    Ensures all desired dependencies are set before constructing the CircuitBreaker,
    preventing half-wired instances. Config is required; other dependencies are optional.

    Usage:
        breaker = (CircuitBreakerBuilder()
            .with_config(config)
            .with_escalation_manager(manager)
            .with_budget(budget)
            .with_caption_circuit_breaker(caption_cb)
            .build())
    """

    def __init__(self) -> None:
        self._config: Optional[CircuitBreakerConfig] = None
        self._escalation_manager: Optional['EscalationManager'] = None
        self._budget: Optional['RateLimitBudget'] = None
        self._caption_circuit_breaker: Optional['CaptionCircuitBreaker'] = None
        self._coordinator: Optional[CircuitBreakerCoordinator] = None
        self._name: str = "search"

    def with_config(self, config: CircuitBreakerConfig) -> 'CircuitBreakerBuilder':
        """Set the circuit breaker configuration (required)."""
        self._config = config
        return self

    def with_escalation_manager(self, manager: 'EscalationManager') -> 'CircuitBreakerBuilder':
        """Link an EscalationManager for coordinated rate-limiting."""
        self._escalation_manager = manager
        return self

    def with_budget(self, budget: 'RateLimitBudget') -> 'CircuitBreakerBuilder':
        """Link a RateLimitBudget for budget-aware pause scaling."""
        self._budget = budget
        return self

    def with_caption_circuit_breaker(self, caption_cb: 'CaptionCircuitBreaker') -> 'CircuitBreakerBuilder':
        """Link the caption circuit breaker for cascade coordination."""
        self._caption_circuit_breaker = caption_cb
        return self

    def with_coordinator(self, coordinator: 'CircuitBreakerCoordinator') -> 'CircuitBreakerBuilder':
        """Link the coordinator for multi-circuit coordination (US-89-009)."""
        self._coordinator = coordinator
        return self

    def with_name(self, name: str) -> 'CircuitBreakerBuilder':
        """Set the name for this circuit breaker (used for registry/coordinator)."""
        self._name = name
        return self

    def build(self) -> 'CircuitBreaker':
        """Build the CircuitBreaker with all configured dependencies.

        Raises:
            ValueError: If no config has been set via with_config().

        Returns:
            A fully-wired CircuitBreaker instance.
        """
        if self._config is None:
            raise ValueError(
                "CircuitBreakerBuilder.build() requires a config. "
                "Call .with_config(CircuitBreakerConfig(...)) before .build()"
            )

        return CircuitBreaker(
            config=self._config,
            escalation_manager=self._escalation_manager,
            budget=self._budget,
            caption_circuit_breaker=self._caption_circuit_breaker,
            name=self._name,
            coordinator=self._coordinator,
        )


class CircuitBreaker(CircuitBreakerBase):
    """Circuit breaker for YouTube search failures.

    Monitors consecutive search failures and temporarily pauses searches
    when a threshold is exceeded. This implements the circuit breaker
    pattern with three states:

    - CLOSED (normal): Searches allowed, failures tracked
    - OPEN (tripped): Searches paused, waiting for pause duration
    - HALF-OPEN (recovery): After pause, first search allowed as test

    Usage:
        breaker = CircuitBreaker(config)

        # Before each search
        breaker.check_and_wait()

        # After each search
        if search_successful:
            breaker.record_success()
        else:
            breaker.record_failure()

    Attributes:
        config: CircuitBreakerConfig with thresholds and timing
        state: Current state (failure count, open/closed, timing)
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        escalation_manager: Optional['EscalationManager'] = None,
        budget: Optional['RateLimitBudget'] = None,
        caption_circuit_breaker: Optional['CaptionCircuitBreaker'] = None,
        name: str = "search",
        coordinator: Optional['CircuitBreakerCoordinator'] = None,
    ):
        """Initialize circuit breaker with configuration and optional dependencies.

        Prefer using CircuitBreakerBuilder for constructing instances with
        dependencies, which validates required fields before construction.

        Args:
            config: CircuitBreakerConfig. If None, uses defaults.
            escalation_manager: Optional EscalationManager for coordinated rate-limiting.
            budget: Optional RateLimitBudget for budget-aware pause scaling.
            caption_circuit_breaker: Optional CaptionCircuitBreaker for cascade coordination.
            name: Name for this circuit breaker (used for registry/coordinator).
            coordinator: Optional CircuitBreakerCoordinator for multi-circuit coordination.
        """
        self._config = config or CircuitBreakerConfig()
        self.state = self._create_state()
        self._escalation_manager = escalation_manager
        self._budget = budget
        self._consecutive_successes: int = 0
        self._pause_calculator: PauseCalculator = PauseCalculator()
        self._caption_circuit_breaker = caption_circuit_breaker
        self._name = name
        self._coordinator = coordinator

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    @property
    def _last_jitter_applied(self) -> float:
        """Delegate jitter tracking to PauseCalculator."""
        return self._pause_calculator.last_jitter_applied

    def _create_state(self) -> CircuitBreakerState:
        return CircuitBreakerState()

    def _get_failure_threshold(self) -> int:
        return self.config.consecutive_failures_threshold

    def _get_domain_label(self) -> str:
        return "search"

    def get_health_metrics(self) -> dict:
        """Get health metrics for observability and debugging.

        Returns:
            Dict containing:
            - trip_count: Total number of times the circuit has tripped
            - recovery_count: Number of times the circuit recovered (success after trip)
            - current_state: 'closed', 'open', or 'half_open'
            - average_pause_duration: Average pause duration in seconds
            - consecutive_failures: Current consecutive failure count
            - total_paused_seconds: Total seconds spent paused
            - is_tripped: Whether circuit is currently open
        """
        total_trips = self.state.total_trips
        total_paused = self.state.total_paused_seconds

        # Calculate average pause duration
        avg_pause_duration = total_paused / total_trips if total_trips > 0 else 0.0

        # Determine current state
        if self.state.is_open:
            current_state = "open"
        elif self.state.consecutive_failures > 0:
            current_state = "half_open"
        else:
            current_state = "closed"

        return {
            'trip_count': total_trips,
            'recovery_count': self._calculate_recovery_count(),
            'current_state': current_state,
            'average_pause_duration': round(avg_pause_duration, 2),
            'consecutive_failures': self.state.consecutive_failures,
            'total_paused_seconds': round(total_paused, 1),
            'is_tripped': self.state.is_open,
            'enabled': self.config.enabled,
        }

    def _calculate_recovery_count(self) -> int:
        """Calculate number of recoveries (successes after being open)."""
        # Recovery is implied when circuit was open but now is closed
        # This is tracked via state transitions
        return self.state.total_trips  # Simplified: each trip implies a potential recovery

    # --- Cascade logic ---

    def _cascade_failure_to_caption(self) -> None:
        """Propagate failure to caption circuit breaker (US-61-003)."""
        if not self._caption_circuit_breaker:
            return

        cascade_enabled = getattr(self.config, 'circuit_breaker_cascade', True)
        if not cascade_enabled:
            return

        if not self._caption_circuit_breaker.is_enabled:
            return

        self._caption_circuit_breaker.state.consecutive_failures += 1
        logger.debug(
            f"Download CB cascading failure to caption CB "
            f"(caption failures now: {self._caption_circuit_breaker.state.consecutive_failures})"
        )

    def _cascade_trip_to_caption(self) -> None:
        """Propagate trip (open) state to caption circuit breaker (US-61-003)."""
        if not self._caption_circuit_breaker:
            return

        cascade_enabled = getattr(self.config, 'circuit_breaker_cascade', True)
        if not cascade_enabled:
            return

        if not self._caption_circuit_breaker.is_enabled:
            return

        if not self._caption_circuit_breaker.state.is_open:
            logger.info(
                "Download circuit breaker cascading trip to caption circuit breaker"
            )
            self._caption_circuit_breaker.state.is_open = True
            self._caption_circuit_breaker.state.opened_at = self.state.opened_at
            self._caption_circuit_breaker.state.total_trips += 1

    def _propagate_via_coordinator(self, event: str) -> None:
        """Propagate events through the coordinator (US-89-009)."""
        if not self._coordinator:
            return

        if event == "trip":
            self._coordinator.propagate_trip(self._name)
        elif event == "failure":
            self._coordinator.propagate_failure(self._name)

    def _on_record_failure(self) -> None:
        """Called after each failure - cascade to caption CB and coordinator."""
        self._cascade_failure_to_caption()
        self._propagate_via_coordinator("failure")

    def _on_trip(self) -> None:
        """Called after circuit trips - log and cascade to caption CB and coordinator."""
        effective_pause = self._get_effective_pause_seconds()
        logger.info(
            f"Circuit breaker TRIPPED: {self.state.consecutive_failures} consecutive "
            f"search failures. Pausing for {effective_pause:.0f}s before "
            f"allowing new searches. (trip #{self.state.total_trips})"
        )
        self._cascade_trip_to_caption()
        self._propagate_via_coordinator("trip")

    # --- Pause calculation pipeline (delegated to PauseCalculator) ---

    def _build_pause_context(self) -> PauseContext:
        """Build a PauseContext from current circuit breaker state."""
        return PauseContext(
            base_pause_seconds=self.config.pause_seconds,
            max_pause_seconds=getattr(self.config, 'max_pause_seconds', 300.0),
            jitter_factor=getattr(self.config, 'jitter_factor', 0.2),
            escalation_manager=self._escalation_manager,
            budget=self._budget,
        )

    def _base_pause(self) -> float:
        """Get the base pause duration from config."""
        return self._pause_calculator.base_pause(self._build_pause_context())

    def _escalation_adjusted_pause(self, pause: float) -> float:
        """Apply escalation-based extension to a pause duration."""
        return self._pause_calculator.escalation_adjusted(pause, self._build_pause_context())

    def _budget_adjusted_pause(self, pause: float) -> float:
        """Apply budget-aware extension to a pause duration."""
        return self._pause_calculator.budget_adjusted(pause, self._build_pause_context())

    def _cap_pause_duration(self, pause: float) -> float:
        """Cap a pause duration at max_pause_seconds."""
        return self._pause_calculator.cap_duration(pause, self._build_pause_context())

    def _get_effective_pause_seconds(self) -> float:
        """Get the effective pause duration, possibly extended by escalation and budget state.

        Delegates to PauseCalculator.calculate() for the full 5-step pipeline.
        """
        ctx = self._build_pause_context()
        result = self._pause_calculator.calculate(ctx)
        return result

    # --- Jitter ---

    def _apply_jitter_raw(self, delay: float) -> float:
        """Apply random jitter to a delay value without capping."""
        return self._pause_calculator.apply_jitter(delay, self._build_pause_context())

    def _apply_jitter(self, delay: float) -> float:
        """Apply random jitter to a delay value, capped at max_pause_seconds."""
        ctx = self._build_pause_context()
        jittered_delay = self._pause_calculator.apply_jitter(delay, ctx)
        jittered_delay = self._pause_calculator.cap_duration(jittered_delay, ctx)
        return jittered_delay

    # --- Escalation tier checks ---

    def _is_escalation_at_tier3(self) -> bool:
        """Check if escalation is at Tier 3 for any tracked keyword."""
        if self._escalation_manager is None:
            return False

        try:
            from .types import EscalationTier
        except ImportError:
            return False

        tier3_keywords = self._escalation_manager.get_keywords_at_tier(
            EscalationTier.FULL_BYPASS
        )
        return len(tier3_keywords) > 0

    # --- Overrides for domain-specific behavior ---

    def _log_pause_wait(self, remaining: float) -> None:
        """Override to include jitter % in log message."""
        jitter_pct = abs(self._last_jitter_applied) * 100
        logger.info(
            f"Circuit breaker OPEN: pausing {remaining:.1f}s "
            f"(jitter={jitter_pct:.0f}%, trip #{self.state.total_trips}, "
            f"{self.state.consecutive_failures} consecutive failures)"
        )

    def check_and_wait(self) -> bool:
        """Check circuit state and wait if necessary.

        If circuit is open (tripped), this method will:
        1. Log the pause clearly (INFO level, not hidden in debug)
        2. Sleep for the remaining pause duration
        3. Transition to half-open state (ready to test)

        Returns:
            True if search should proceed, False if circuit breaker is disabled.

        Note:
            This method blocks if circuit is open. Call before each search.
        """
        if not self.config.enabled:
            return False

        if not self.state.is_open:
            return True

        # Circuit is open - check if pause duration has elapsed
        # Use effective pause which may be extended by escalation state
        effective_pause = self._get_effective_pause_seconds()
        elapsed = time.time() - self.state.opened_at
        remaining = effective_pause - elapsed

        if remaining > 0:
            # Jitter is already applied within _get_effective_pause_seconds pipeline
            jitter_pct = abs(self._last_jitter_applied) * 100
            logger.info(
                f"Circuit breaker OPEN: pausing {remaining:.1f}s "
                f"(jitter={jitter_pct:.0f}%, trip #{self.state.total_trips}, "
                f"{self.state.consecutive_failures} consecutive failures)"
            )
            time.sleep(remaining)
            self.state.total_paused_seconds += remaining

        # Transition to half-open (closed but ready to trip quickly)
        logger.info("Circuit breaker: pause complete, allowing search (half-open)")
        self.state.is_open = False
        self.state.opened_at = None
        # Keep failure count - will reset on success or trip again on failure

        return True

    def record_success(self) -> None:
        """Record a successful search.

        Resets the consecutive failure counter and closes the circuit.
        When escalation is at Tier 3, requires 3 consecutive successes
        instead of 1 to close the circuit.

        Call after any search that returns results.
        """
        if not self.config.enabled:
            return

        self._consecutive_successes += 1

        # When escalation is at Tier 3, require 3 consecutive successes to close
        required_successes = 1
        if self._is_escalation_at_tier3():
            required_successes = 3
            if self._consecutive_successes < required_successes:
                logger.debug(
                    f"Circuit breaker: success {self._consecutive_successes}/{required_successes} "
                    f"(extended reset: escalation at Tier 3)"
                )
                return

        if self.state.consecutive_failures > 0:
            logger.debug(
                f"Circuit breaker: search succeeded after "
                f"{self.state.consecutive_failures} failures, resetting counter"
            )

        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None
        self._consecutive_successes = 0

    def record_failure(self) -> bool:
        """Record a search failure.

        Increments the consecutive failure counter. If threshold is reached,
        trips the circuit (opens it) and pauses future searches.

        US-61-003: Also cascades failure to caption circuit breaker if linked.

        Returns:
            True if circuit tripped (opened) as a result of this failure,
            False otherwise.
        """
        if not self.config.enabled:
            return False

        self.state.consecutive_failures += 1
        self._consecutive_successes = 0  # Reset success streak on failure

        logger.debug(
            f"Circuit breaker: search failure "
            f"({self.state.consecutive_failures}/{self.config.consecutive_failures_threshold})"
        )

        # US-61-003: Cascade failure to caption CB
        self._cascade_failure_to_caption()

        # Check if we've hit the threshold
        if self.state.consecutive_failures >= self.config.consecutive_failures_threshold:
            self._trip()
            return True

        return False

    def _trip(self) -> None:
        """Trip the circuit breaker (open it).

        Called internally when consecutive failures reach threshold.
        Logs clearly at INFO level so users can see the pause happening.
        Uses effective pause duration (which may be extended by escalation state).
        US-61-003: Also cascades trip to caption circuit breaker if linked.
        US-89-009: Also propagates trip to coordinator for multi-circuit coordination.
        """
        self.state.is_open = True
        self.state.opened_at = time.time()
        self.state.total_trips += 1

        effective_pause = self._get_effective_pause_seconds()
        logger.info(
            f"Circuit breaker TRIPPED: {self.state.consecutive_failures} consecutive "
            f"search failures. Pausing for {effective_pause:.0f}s before "
            f"allowing new searches. (trip #{self.state.total_trips})"
        )

        # US-61-003: Cascade trip to caption CB
        self._cascade_trip_to_caption()

        # US-89-009: Propagate trip to coordinator
        self._propagate_via_coordinator("trip")

    def reset(self) -> None:
        """Manually reset the circuit breaker.

        Clears all failure state and closes the circuit.
        Use when external conditions change (e.g., VPN switched, cookie rotated).
        """
        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None
        logger.debug("Circuit breaker: manually reset")

    def get_remaining_pause_time(self, apply_jitter: bool = False) -> float:
        """Get remaining time until circuit breaker recovers.

        Uses effective pause duration which includes jitter and is capped
        at max_pause_seconds. The apply_jitter parameter is kept for backward
        compatibility but jitter is now always applied as part of the
        effective pause calculation pipeline.

        Args:
            apply_jitter: Deprecated. Jitter is now always applied within
                         _get_effective_pause_seconds. Kept for backward compat.

        Returns:
            Remaining pause time in seconds, or 0.0 if not tripped.
        """
        if not self.config.enabled or not self.state.is_open:
            return 0.0

        if self.state.opened_at is None:
            return 0.0

        effective_pause = self._get_effective_pause_seconds()
        elapsed = time.time() - self.state.opened_at
        remaining = effective_pause - elapsed
        remaining = max(0.0, remaining)

        return remaining

    def wait_for_recovery_if_needed(self, context: str = "") -> float:
        """Wait for circuit breaker recovery if tripped.

        This is designed for download retry coordination (US-011). Unlike
        check_and_wait(), this method:
        - Returns the actual wait time for metrics tracking
        - Accepts a context string for more specific logging
        - Does not transition state (caller still needs check_and_wait for state transition)
        - Jitter is applied within the effective pause calculation pipeline

        Args:
            context: Optional context string for logging (e.g., "download retry")

        Returns:
            The number of seconds waited, or 0.0 if no wait was needed.
        """
        if not self.config.enabled:
            return 0.0

        if not self.state.is_open:
            return 0.0

        remaining = self.get_remaining_pause_time()
        if remaining <= 0:
            return 0.0

        # Jitter is already applied within _get_effective_pause_seconds pipeline
        jitter_pct = abs(self._last_jitter_applied) * 100

        # Log the wait with context
        ctx_str = f" ({context})" if context else ""
        logger.info(
            f"Circuit breaker OPEN{ctx_str}: waiting {remaining:.1f}s for recovery "
            f"(jitter={jitter_pct:.0f}%, trip #{self.state.total_trips}, "
            f"{self.state.consecutive_failures} consecutive failures)"
        )

        time.sleep(remaining)
        self.state.total_paused_seconds += remaining

        # Transition to half-open state
        logger.info(f"Circuit breaker: pause complete{ctx_str}, resuming")
        self.state.is_open = False
        self.state.opened_at = None

        return remaining

    def get_stats(self) -> dict:
        """Get circuit breaker statistics for reporting."""
        return {
            'enabled': self.config.enabled,
            'is_open': self.state.is_open,
            'consecutive_failures': self.state.consecutive_failures,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': round(self.state.total_paused_seconds, 1),
            'threshold': self.config.consecutive_failures_threshold,
            'pause_seconds': self.config.pause_seconds,
        }

    def to_checkpoint_dict(self) -> dict:
        """Serialize state to dictionary for checkpoint persistence."""
        return {
            'consecutive_failures': self.state.consecutive_failures,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': self.state.total_paused_seconds,
        }

    def from_checkpoint_dict(self, data: dict) -> None:
        """Restore state from checkpoint dictionary."""
        if not data:
            return

        self.state.total_trips = data.get('total_trips', 0)
        self.state.total_paused_seconds = data.get('total_paused_seconds', 0.0)
        self.state.consecutive_failures = 0
        self.state.is_open = False
        self.state.opened_at = None
