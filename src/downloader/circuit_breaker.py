"""Circuit breaker for repeated search failures.

Implements the circuit breaker pattern to prevent hammering YouTube
when multiple consecutive searches fail. After threshold consecutive
failures, the circuit "trips" and pauses searching for a configured
duration before allowing new searches.

This protects against:
- Wasted API calls during widespread rate limiting
- Excessive retries that could worsen rate limit issues
- Unnecessarily slow pipeline execution during outages

US-153-007: Added category-based circuit breakers for granular error handling
per error category (network, quota, auth, etc.) rather than just per-endpoint.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, List

from src.common.circuit_breaker_base import CircuitBreakerBase, CircuitBreakerStateBase
from src.downloader.pause_calculator import PauseCalculator, PauseContext, RegionalRateLimitTracker

if TYPE_CHECKING:
    from .escalation_manager import EscalationManager
    from .rate_limit_budget import RateLimitBudget
    from src.caption.circuit_breaker import CaptionCircuitBreaker

logger = logging.getLogger(__name__)


# =============================================================================
# Error Category Circuit Breaker (US-153-007)
# =============================================================================


class ErrorCategory(Enum):
    """Error categories for category-based circuit breaking.

    US-153-007: Enables granular circuit breaking per error category
    rather than just per-endpoint. Each category can have its own
    thresholds and pause durations.
    """

    NETWORK = "network"
    QUOTA = "quota"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    BOT_DETECTION = "bot_detection"
    GEO_BLOCKED = "geo_blocked"
    TIMEOUT = "timeout"
    VIDEO_SPECIFIC = "video_specific"
    UNKNOWN = "unknown"


@dataclass
class CategoryCircuitBreakerConfig:
    """Configuration for category-based circuit breaker.

    US-153-007: Each error category can have its own thresholds and
    pause durations for more granular error handling.
    """

    # Category this config applies to
    category: ErrorCategory = ErrorCategory.UNKNOWN

    # Number of consecutive failures before circuit trips
    consecutive_failures_threshold: int = 5

    # Duration to pause after circuit trips (seconds)
    pause_seconds: float = 60.0

    # Maximum pause duration cap
    max_pause_seconds: float = 300.0

    # Jitter factor
    jitter_factor: float = 0.2

    # Enable/disable this category circuit breaker
    enabled: bool = True


class CategoryCircuitBreaker:
    """Circuit breaker for a specific error category.

    US-153-007: Implements category-based circuit breaking. Each error
    category (network, quota, auth, etc.) can have its own circuit breaker
    with different thresholds and pause durations.

    Usage:
        # Create category circuit breakers
        network_cb = CategoryCircuitBreaker(ErrorCategory.NETWORK)
        quota_cb = CategoryCircuitBreaker(ErrorCategory.QUOTA)
        auth_cb = CategoryCircuitBreaker(ErrorCategory.AUTH)

        # Register errors by category
        network_cb.record_failure()  # Network error
        quota_cb.record_failure()   # Quota error
        auth_cb.record_failure()    # Auth error

        # Check if category is tripped
        if network_cb.is_tripped:
            wait_for_recovery(network_cb)
    """

    def __init__(
        self,
        category: ErrorCategory,
        config: Optional[CategoryCircuitBreakerConfig] = None,
    ):
        """Initialize category circuit breaker.

        Args:
            category: The error category this breaker handles.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.category = category
        self.config = config or CategoryCircuitBreakerConfig(category=category)
        self._is_open: bool = False
        self._opened_at: Optional[float] = None
        self._consecutive_failures: int = 0
        self._total_trips: int = 0
        self._total_paused_seconds: float = 0.0
        self._failure_history: List[dict] = []

    @property
    def is_tripped(self) -> bool:
        """Check if circuit is currently tripped (open)."""
        return self._is_open

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count."""
        return self._consecutive_failures

    def record_failure(self) -> bool:
        """Record a failure for this category.

        Returns:
            True if circuit tripped as a result of this failure.
        """
        if not self.config.enabled:
            return False

        self._consecutive_failures += 1

        # Track failure in history
        self._failure_history.append({
            'timestamp': time.time(),
            'consecutive_failures': self._consecutive_failures,
        })

        # Keep only last 100 failures
        if len(self._failure_history) > 100:
            self._failure_history = self._failure_history[-100:]

        logger.debug(
            f"CategoryCircuitBreaker ({self.category.value}): failure "
            f"({self._consecutive_failures}/{self.config.consecutive_failures_threshold})"
        )

        # Check if threshold reached
        if self._consecutive_failures >= self.config.consecutive_failures_threshold:
            self._trip()
            return True

        return False

    def record_success(self) -> None:
        """Record a success for this category.

        Resets failure counter and closes the circuit.
        """
        if not self.config.enabled:
            return

        if self._consecutive_failures > 0:
            logger.debug(
                f"CategoryCircuitBreaker ({self.category.value}): success after "
                f"{self._consecutive_failures} failures, resetting"
            )

        self._consecutive_failures = 0
        self._is_open = False
        self._opened_at = None

    def _trip(self) -> None:
        """Trip the circuit breaker (open it)."""
        self._is_open = True
        self._opened_at = time.time()
        self._total_trips += 1

        logger.info(
            f"CategoryCircuitBreaker ({self.category.value}): TRIPPED after "
            f"{self._consecutive_failures} consecutive failures. "
            f"Pausing for {self.config.pause_seconds:.0f}s. (trip #{self._total_trips})"
        )

    def check_and_wait(self) -> bool:
        """Check circuit state and wait if necessary.

        Returns:
            True if operation should proceed, False if circuit breaker disabled.
        """
        if not self.config.enabled:
            return False

        if not self._is_open:
            return True

        # Calculate remaining pause time
        if self._opened_at is not None:
            elapsed = time.time() - self._opened_at
            remaining = self.config.pause_seconds - elapsed

            if remaining > 0:
                logger.info(
                    f"CategoryCircuitBreaker ({self.category.value}): OPEN, "
                    f"pausing {remaining:.1f}s"
                )
                time.sleep(remaining)
                self._total_paused_seconds += remaining

        # Recover
        self._is_open = False
        self._opened_at = None

        return True

    def get_remaining_pause_time(self) -> float:
        """Get remaining pause time if circuit is tripped.

        Returns:
            Remaining seconds, or 0.0 if not tripped.
        """
        if not self._is_open or self._opened_at is None:
            return 0.0

        elapsed = time.time() - self._opened_at
        remaining = self.config.pause_seconds - elapsed
        return max(0.0, remaining)

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            'category': self.category.value,
            'enabled': self.config.enabled,
            'is_tripped': self._is_open,
            'consecutive_failures': self._consecutive_failures,
            'total_trips': self._total_trips,
            'total_paused_seconds': round(self._total_paused_seconds, 1),
            'threshold': self.config.consecutive_failures_threshold,
            'pause_seconds': self.config.pause_seconds,
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._consecutive_failures = 0
        self._is_open = False
        self._opened_at = None
        logger.debug(f"CategoryCircuitBreaker ({self.category.value}): manually reset")


class CategoryCircuitBreakerRegistry:
    """Registry for category-based circuit breakers.

    US-153-007: Manages multiple category circuit breakers and provides
    a unified interface for error handling across categories.

    Usage:
        registry = CategoryCircuitBreakerRegistry()

        # Record errors by category
        registry.record_error(ErrorCategory.NETWORK)
        registry.record_error(ErrorCategory.QUOTA)
        registry.record_error(ErrorCategory.AUTH)

        # Check if any category is tripped
        if registry.is_any_tripped():
            for category in registry.get_tripped_categories():
                registry.wait_for_recovery(category)
    """

    def __init__(self):
        """Initialize the category circuit breaker registry."""
        self._breakers: Dict[ErrorCategory, CategoryCircuitBreaker] = {}

        # Default configurations per category
        self._default_configs: Dict[ErrorCategory, CategoryCircuitBreakerConfig] = {
            ErrorCategory.NETWORK: CategoryCircuitBreakerConfig(
                category=ErrorCategory.NETWORK,
                consecutive_failures_threshold=5,
                pause_seconds=30.0,
                max_pause_seconds=120.0,
            ),
            ErrorCategory.QUOTA: CategoryCircuitBreakerConfig(
                category=ErrorCategory.QUOTA,
                consecutive_failures_threshold=3,
                pause_seconds=60.0,
                max_pause_seconds=300.0,
            ),
            ErrorCategory.AUTH: CategoryCircuitBreakerConfig(
                category=ErrorCategory.AUTH,
                consecutive_failures_threshold=3,
                pause_seconds=10.0,
                max_pause_seconds=30.0,
            ),
            ErrorCategory.RATE_LIMIT: CategoryCircuitBreakerConfig(
                category=ErrorCategory.RATE_LIMIT,
                consecutive_failures_threshold=5,
                pause_seconds=60.0,
                max_pause_seconds=300.0,
            ),
            ErrorCategory.BOT_DETECTION: CategoryCircuitBreakerConfig(
                category=ErrorCategory.BOT_DETECTION,
                consecutive_failures_threshold=5,
                pause_seconds=60.0,
                max_pause_seconds=300.0,
            ),
            ErrorCategory.GEO_BLOCKED: CategoryCircuitBreakerConfig(
                category=ErrorCategory.GEO_BLOCKED,
                consecutive_failures_threshold=3,
                pause_seconds=120.0,
                max_pause_seconds=600.0,
            ),
            ErrorCategory.TIMEOUT: CategoryCircuitBreakerConfig(
                category=ErrorCategory.TIMEOUT,
                consecutive_failures_threshold=5,
                pause_seconds=30.0,
                max_pause_seconds=120.0,
            ),
            ErrorCategory.VIDEO_SPECIFIC: CategoryCircuitBreakerConfig(
                category=ErrorCategory.VIDEO_SPECIFIC,
                consecutive_failures_threshold=10,
                pause_seconds=10.0,
                max_pause_seconds=30.0,
            ),
            ErrorCategory.UNKNOWN: CategoryCircuitBreakerConfig(
                category=ErrorCategory.UNKNOWN,
                consecutive_failures_threshold=5,
                pause_seconds=60.0,
                max_pause_seconds=180.0,
            ),
        }

    def get_or_create(
        self,
        category: ErrorCategory,
        config: Optional[CategoryCircuitBreakerConfig] = None,
    ) -> CategoryCircuitBreaker:
        """Get or create a circuit breaker for a category.

        Args:
            category: The error category.
            config: Optional custom config. Uses default if not provided.

        Returns:
            The CategoryCircuitBreaker for this category.
        """
        if category not in self._breakers:
            self._breakers[category] = CategoryCircuitBreaker(
                category,
                config or self._default_configs.get(category)
            )
            logger.debug(
                f"CategoryCircuitBreakerRegistry: created breaker for {category.value}"
            )

        return self._breakers[category]

    def record_error(self, category: ErrorCategory) -> bool:
        """Record an error for a category.

        Args:
            category: The error category.

        Returns:
            True if the category circuit tripped as a result.
        """
        breaker = self.get_or_create(category)
        return breaker.record_failure()

    def record_success(self, category: ErrorCategory) -> None:
        """Record a success for a category.

        Args:
            category: The error category.
        """
        breaker = self.get_or_create(category)
        breaker.record_success()

    def is_tripped(self, category: ErrorCategory) -> bool:
        """Check if a category circuit is tripped.

        Args:
            category: The error category.

        Returns:
            True if the category circuit is open.
        """
        breaker = self.get_or_create(category)
        return breaker.is_tripped

    def is_any_tripped(self) -> bool:
        """Check if any category circuit is tripped.

        Returns:
            True if any circuit is open.
        """
        return any(cb.is_tripped for cb in self._breakers.values())

    def get_tripped_categories(self) -> List[ErrorCategory]:
        """Get list of tripped categories.

        Returns:
            List of categories with open circuits.
        """
        return [
            category
            for category, breaker in self._breakers.items()
            if breaker.is_tripped
        ]

    def wait_for_recovery(self, category: ErrorCategory) -> float:
        """Wait for a category circuit to recover.

        Args:
            category: The error category.

        Returns:
            Number of seconds waited.
        """
        breaker = self.get_or_create(category)
        return breaker.get_remaining_pause_time()

    def wait_for_all_recovery(self) -> float:
        """Wait for all tripped circuits to recover.

        Returns:
            Total seconds waited.
        """
        total_wait = 0.0
        for category in self.get_tripped_categories():
            breaker = self.get_or_create(category)
            breaker.check_and_wait()
            total_wait += breaker.get_stats()['total_paused_seconds']
        return total_wait

    def get_all_stats(self) -> dict:
        """Get statistics for all category circuit breakers.

        Returns:
            Dict mapping category names to their statistics.
        """
        return {
            category.value: breaker.get_stats()
            for category, breaker in self._breakers.items()
        }

    def get_aggregate_stats(self) -> dict:
        """Get aggregate statistics across all categories.

        Returns:
            Dict with aggregate statistics.
        """
        total_trips = sum(cb._total_trips for cb in self._breakers.values())
        total_paused = sum(cb._total_paused_seconds for cb in self._breakers.values())
        tripped_count = len(self.get_tripped_categories())

        return {
            'total_categories': len(self._breakers),
            'tripped_count': tripped_count,
            'total_trips': total_trips,
            'total_paused_seconds': round(total_paused, 1),
            'is_any_tripped': self.is_any_tripped(),
        }

    def reset_category(self, category: ErrorCategory) -> None:
        """Reset a specific category circuit breaker.

        Args:
            category: The error category to reset.
        """
        breaker = self.get_or_create(category)
        breaker.reset()
        logger.info(f"CategoryCircuitBreakerRegistry: reset {category.value}")

    def reset_all(self) -> None:
        """Reset all category circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("CategoryCircuitBreakerRegistry: reset all breakers")


# Global registry instance
_category_breaker_registry: Optional[CategoryCircuitBreakerRegistry] = None


def get_category_breaker_registry() -> CategoryCircuitBreakerRegistry:
    """Get the global category circuit breaker registry.

    Returns:
        The global CategoryCircuitBreakerRegistry instance.
    """
    global _category_breaker_registry
    if _category_breaker_registry is None:
        _category_breaker_registry = CategoryCircuitBreakerRegistry()
    return _category_breaker_registry


def map_error_to_category(error_type: str) -> ErrorCategory:
    """Map an error type string to an ErrorCategory.

    Args:
        error_type: The error type string (e.g., 'quota_exceeded', 'network').

    Returns:
        The corresponding ErrorCategory.
    """
    error_type_lower = error_type.lower()

    # Quota errors
    if any(x in error_type_lower for x in ['quota', 'daily_quota']):
        return ErrorCategory.QUOTA

    # Auth errors
    if any(x in error_type_lower for x in ['invalid_key', 'invalid_project', 'disabled_project', 'permission_denied']):
        return ErrorCategory.AUTH

    # Rate limit errors
    if any(x in error_type_lower for x in ['rate_limit', 'rate_limited', 'per_second', '429']):
        return ErrorCategory.RATE_LIMIT

    # Bot detection
    if any(x in error_type_lower for x in ['bot', '403', 'forbidden', 'captcha']):
        return ErrorCategory.BOT_DETECTION

    # Geo-blocking
    if any(x in error_type_lower for x in ['geo', 'region']):
        return ErrorCategory.GEO_BLOCKED

    # Network errors
    if any(x in error_type_lower for x in ['network', 'dns', 'connection', 'timeout', 'tls', 'ssl']):
        return ErrorCategory.NETWORK

    # Timeout errors
    if 'timeout' in error_type_lower:
        return ErrorCategory.TIMEOUT

    # Unknown
    return ErrorCategory.UNKNOWN


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

    US-136-008: Extended to include caption circuit breaker and provide
    aggregate health metrics and intelligent recovery sequencing.

    Default cascade rules:
    - search -> caption: trip on failure (existing US-61-003)
    - caption -> search: trip on failure (existing US-61-003)
    - search -> download: trip on trip (existing US-89-009)
    - caption -> download: trip on trip (NEW US-136-008)

    Usage:
        coordinator = CircuitBreakerCoordinator.get_instance()
        coordinator.set_registry(registry)
        coordinator.add_rule(CascadeRule(source="search", target="download", on_trip=True))

        # In each circuit breaker, call propagate when it trips
        coordinator.propagate_trip("search")
        coordinator.propagate_failure("search")

        # Get aggregate health across all components
        health = coordinator.get_aggregate_health()

        # Intelligent recovery sequencing
        coordinator.recover_circuits()
    """

    _instance: Optional['CircuitBreakerCoordinator'] = None
    _lock = None  # Will be initialized on first use

    def __init__(self) -> None:
        self._registry: Optional[CircuitBreakerRegistry] = None
        self._rules: List[CascadeRule] = []
        self._enabled: bool = True
        # US-136-008: Coordination event metrics
        self._coordination_events: List[dict] = []
        self._max_events: int = 100  # Keep last 100 events
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
        # Search -> download: propagate trip (existing US-89-009)
        self._rules.append(CascadeRule(
            source="search",
            target="download",
            on_trip=True,
            on_failure=False,
        ))
        # Caption -> download: propagate trip (NEW US-136-008)
        self._rules.append(CascadeRule(
            source="caption",
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
                        # US-136-008: Record coordination event
                        self._record_coordination_event('trip_propagation', source_name, rule.target, {
                            'target_failures': target_cb.state.consecutive_failures,
                        })

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
                    # US-136-008: Record coordination event
                    self._record_coordination_event('failure_propagation', source_name, rule.target, {
                        'target_failures': target_cb.state.consecutive_failures,
                    })

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
            'total_coordination_events': len(self._coordination_events),
        }

    def _record_coordination_event(self, event_type: str, source: str, target: str, details: dict = None) -> None:
        """Record a cross-component coordination event for metrics.

        Args:
            event_type: Type of event (trip_propagation, failure_propagation, recovery)
            source: Source component name
            target: Target component name
            details: Additional event details
        """
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'source': source,
            'target': target,
            'details': details or {},
        }
        self._coordination_events.append(event)
        # Keep only last N events
        if len(self._coordination_events) > self._max_events:
            self._coordination_events = self._coordination_events[-self._max_events:]

    def get_aggregate_health(self) -> dict:
        """Get aggregate health across all registered circuit breakers.

        US-136-008: Returns comprehensive health status including individual
        component states and overall system health.

        Returns:
            Dict with:
            - overall_healthy: bool - True if all circuits are healthy
            - components: dict - Per-component health metrics
            - aggregate: dict - Aggregate statistics
            - coordination_events: list - Recent coordination events
        """
        if self._registry is None:
            return {
                'overall_healthy': True,
                'components': {},
                'aggregate': {'total_breakers': 0, 'tripped_count': 0},
                'coordination_events': [],
            }

        components = {}
        all_metrics = self._registry.get_all_health_metrics()
        for name, metrics in all_metrics.items():
            # Handle different metric formats from different CB implementations
            is_open = metrics.get('is_open', False)
            if is_open is False:
                # Check is_tripped as fallback (used by download CB)
                is_open = metrics.get('is_tripped', False)
            if is_open is False:
                # Check current_state as fallback
                is_open = metrics.get('current_state') == 'open'

            components[name] = {
                'is_open': is_open,
                'consecutive_failures': metrics.get('consecutive_failures', 0),
                'total_trips': metrics.get('total_trips', metrics.get('trip_count', 0)),
                'total_paused_seconds': metrics.get('total_paused_seconds', 0.0),
            }

        aggregate = self._registry.get_aggregate_stats()

        # Determine overall health
        overall_healthy = not aggregate.get('is_any_tripped', False)

        return {
            'overall_healthy': overall_healthy,
            'components': components,
            'aggregate': aggregate,
            'coordination_events': self._coordination_events[-10:],  # Last 10 events
        }

    def recover_circuits(self) -> dict:
        """Implement intelligent recovery sequencing.

        US-136-008: Attempts to recover circuits in the correct order to avoid
        cascading failures. Recovery order is based on dependency chain:
        1. First recover downstream components (download)
        2. Then recover upstream components (caption, search)

        Returns:
            Dict with recovery results including:
            - recovered: list of circuit names that were recovered
            - still_tripped: list of circuit names still open
            - recovery_order: order in which recovery was attempted
        """
        if self._registry is None:
            return {'recovered': [], 'still_tripped': [], 'recovery_order': []}

        # Define recovery order (downstream first to upstream)
        # Download is most downstream, then caption, then search
        recovery_order = ['download', 'caption', 'search']

        recovered = []
        still_tripped = []

        for name in recovery_order:
            breaker = self._registry.get(name)
            if breaker is not None and breaker.state.is_open:
                # Check if pause duration has elapsed
                if breaker.state.opened_at is not None:
                    effective_pause = getattr(breaker.config, 'pause_seconds', 60.0)
                    max_pause = getattr(breaker.config, 'max_pause_seconds', 300.0)
                    effective_pause = min(effective_pause, max_pause)

                    elapsed = time.time() - breaker.state.opened_at
                    if elapsed >= effective_pause:
                        # Recover this circuit
                        logger.info(f"CircuitBreakerCoordinator: recovering {name} circuit")
                        breaker.state.is_open = False
                        breaker.state.opened_at = None
                        breaker.state.consecutive_failures = 0
                        recovered.append(name)
                        self._record_coordination_event('recovery', 'coordinator', name, {'elapsed': elapsed})
                    else:
                        still_tripped.append(name)
                        logger.debug(f"CircuitBreakerCoordinator: {name} not ready to recover (elapsed: {elapsed:.1f}s < {effective_pause:.1f}s)")

        return {
            'recovered': recovered,
            'still_tripped': still_tripped,
            'recovery_order': recovery_order,
        }

    def get_coordination_metrics(self) -> dict:
        """Get coordination event metrics for monitoring.

        US-136-008: Returns metrics about cross-component coordination events.

        Returns:
            Dict with:
            - total_events: Total number of coordination events
            - events_by_type: Count of events grouped by type
            - events_by_source: Count of events grouped by source
            - recent_events: Last N events for detailed analysis
        """
        events_by_type = {}
        events_by_source = {}

        for event in self._coordination_events:
            event_type = event['event_type']
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1

            source = event['source']
            events_by_source[source] = events_by_source.get(source, 0) + 1

        return {
            'total_events': len(self._coordination_events),
            'events_by_type': events_by_type,
            'events_by_source': events_by_source,
            'recent_events': self._coordination_events[-20:],  # Last 20 events
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

    # US-109-002: Jitter strategy for controlling jitter behavior
    # Options:
    #   - 'random': Classic uniform jitter (default)
    #   - 'adaptive': Time-of-day based jitter (higher during peak hours)
    #   - 'deterministic': Seeded jitter based on client_id for reproducibility
    jitter_strategy: str = "random"

    # Maximum jitter factor allowed (hard cap at 50% per AC)
    # Regardless of strategy or configuration, jitter will never exceed this
    jitter_max_factor: float = 0.5

    # Enable jitter correlation check to prevent multiple clients getting similar values
    # When True, compares against recent jitter values and adjusts if too similar
    jitter_correlation_check: bool = False

    # Circuit breaker cascade (US-61-003): when enabled, failures propagate to
    # the caption circuit breaker (and vice versa) to speed up coordinated pausing
    # when YouTube is rate-limiting. Default: True.
    circuit_breaker_cascade: bool = True

    # US-113-004: Enable circuit breaker state persistence across runs
    # When true, circuit breaker state (is_open, failure counts, timestamps)
    # is saved to checkpoint and restored on pipeline resume.
    # Default: True.
    persist_state: bool = True

    # US-144-004: Path to the state file for circuit breaker persistence.
    # If not set, defaults to project checkpoint directory.
    # Set to a specific path to override the default location.
    state_file_path: str = ""

    # US-144-004: Auto-save state after each state change (trip, half_open, closed).
    # When true, state is persisted immediately after any state transition.
    # Default: False (use checkpoint save for better performance).
    auto_save_on_state_change: bool = False

    # US-109-011: Region-specific backoff configuration
    # Enable region-specific backoff multipliers based on VPN country
    # When enabled, pause duration is scaled by region multiplier:
    # - US: 1.0 (baseline)
    # - EU: 1.2 (20% longer)
    # - ASIA: 1.5 (50% longer)
    # - OTHER: 2.0 (100% longer)
    region_backoff_enabled: bool = False

    # Region-specific multiplier mapping (defaults from config)
    # Can be overridden via region_backoff config
    region_multipliers: dict = field(default_factory=lambda: {
        'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0
    })

    # Enable regional rate limit tracking (separate counters per region)
    track_per_region: bool = False


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

        # US-109-011: Initialize regional rate limit tracking
        if getattr(self._config, 'track_per_region', False):
            RegionalRateLimitTracker.enable()
        else:
            RegionalRateLimitTracker.disable()

        # US-144-004: Load state from file if persistence is enabled and file exists
        self._load_state_from_file()

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
        # US-109-011: Get region from escalation manager (MullvadVPN country)
        region = ''
        if self._escalation_manager is not None:
            try:
                region = self._escalation_manager.get_current_region() or ''
            except Exception:
                region = ''

        return PauseContext(
            base_pause_seconds=self.config.pause_seconds,
            max_pause_seconds=getattr(self.config, 'max_pause_seconds', 300.0),
            jitter_factor=getattr(self.config, 'jitter_factor', 0.2),
            # US-109-002: New jitter configuration
            jitter_strategy=getattr(self.config, 'jitter_strategy', 'random'),
            jitter_max_factor=getattr(self.config, 'jitter_max_factor', 0.5),
            jitter_correlation_check=getattr(self.config, 'jitter_correlation_check', False),
            client_id=getattr(self.config, 'client_id', ''),
            escalation_manager=self._escalation_manager,
            budget=self._budget,
            # US-109-011: Region-specific backoff
            region=region,
            region_enabled=getattr(self.config, 'region_backoff_enabled', False),
            region_multipliers=getattr(self.config, 'region_multipliers', {
                'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0
            }),
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

        # US-144-004: Auto-save state after recovery (open -> half_open)
        self._save_state_to_file()

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

        # US-144-004: Auto-save state after close (half_open/closed transition)
        self._save_state_to_file()

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

        # Track failure in history with timestamp and failure count (US-120-006)
        failure_record = {
            'timestamp': time.time(),
            'consecutive_failures': self.state.consecutive_failures
        }
        self.state.failure_history.append(failure_record)

        # Keep only last 100 failures in history
        if len(self.state.failure_history) > 100:
            self.state.failure_history = self.state.failure_history[-100:]

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

        # US-144-004: Auto-save state after trip
        self._save_state_to_file()

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
            'is_open': self.state.is_open,
            'opened_at': self.state.opened_at,
            'total_trips': self.state.total_trips,
            'total_paused_seconds': self.state.total_paused_seconds,
        }

    def from_checkpoint_dict(self, data: dict, checkpoint_age_seconds: float = 0.0) -> dict:
        """Restore state from checkpoint dictionary with stale state handling.

        Delegates to base class implementation for state restoration logic.

        Args:
            data: Dictionary with circuit breaker state from checkpoint
            checkpoint_age_seconds: Age of checkpoint in seconds for stale handling

        Returns:
            Dict with restoration info from base class
        """
        return super().from_checkpoint_dict(data, checkpoint_age_seconds)

    # US-144-004: State file persistence methods

    def _get_state_file_path(self) -> Optional[str]:
        """Get the state file path from config.

        Returns:
            Path to state file if configured, None otherwise.
        """
        state_file = getattr(self._config, 'state_file_path', '')
        return state_file if state_file else None

    def _load_state_from_file(self) -> None:
        """Load circuit breaker state from file if persistence is enabled.

        Called during initialization to restore previous state.
        Only loads if persist_state is True and state file exists.
        """
        if not getattr(self._config, 'persist_state', True):
            logger.debug(f"CircuitBreaker '{self._name}': persistence disabled, skipping state load")
            return

        state_file = self._get_state_file_path()
        if not state_file:
            logger.debug(f"CircuitBreaker '{self._name}': no state file path configured, skipping state load")
            return

        import os
        if not os.path.exists(state_file):
            logger.debug(f"CircuitBreaker '{self._name}': state file not found at {state_file}, starting fresh")
            return

        try:
            import json
            with open(state_file, 'r') as f:
                data = json.load(f)

            # Calculate checkpoint age
            checkpoint_time = data.get('saved_at', 0)
            checkpoint_age = time.time() - checkpoint_time if checkpoint_time else 0.0

            # Restore state using base class method
            result = self.from_checkpoint_dict(data, checkpoint_age)

            logger.info(
                f"CircuitBreaker '{self._name}': loaded state from {state_file}, "
                f"restored={result['restored']}, state={result['restored_state']}, "
                f"was_stale={result['was_stale']}"
            )
        except Exception as e:
            logger.warning(f"CircuitBreaker '{self._name}': failed to load state from {state_file}: {e}")

    def _save_state_to_file(self) -> None:
        """Save circuit breaker state to file if persistence is enabled.

        Called after state changes when auto_save_on_state_change is True.
        """
        if not getattr(self._config, 'persist_state', True):
            return

        if not getattr(self._config, 'auto_save_on_state_change', False):
            return

        state_file = self._get_state_file_path()
        if not state_file:
            logger.debug(f"CircuitBreaker '{self._name}': no state file path configured, skipping state save")
            return

        try:
            import os
            import json

            # Ensure directory exists
            state_dir = os.path.dirname(state_file)
            if state_dir and not os.path.exists(state_dir):
                os.makedirs(state_dir, exist_ok=True)

            # Get state and add timestamp
            state_dict = self.to_checkpoint_dict()
            state_dict['saved_at'] = time.time()
            state_dict['name'] = self._name

            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2)

            logger.debug(f"CircuitBreaker '{self._name}': saved state to {state_file}")
        except Exception as e:
            logger.warning(f"CircuitBreaker '{self._name}': failed to save state to {state_file}: {e}")
