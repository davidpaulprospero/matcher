"""
Global Rate Limit Coordinator for parallel YouTube operations.

Provides unified slot-based rate limiting across:
- Caption fetching
- Video downloading
- API calls

Uses a token bucket pattern with configurable slots per second.

Example:
    coordinator = GlobalRateLimitCoordinator()

    # Acquire a slot before making request
    if coordinator.acquire_slot('caption'):
        try:
            result = fetch_captions(video_id)
        finally:
            coordinator.release_slot('caption')
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be rate limited."""
    CAPTION = 'caption'
    DOWNLOAD = 'download'
    API = 'api'


@dataclass
class RateLimitConfig:
    """Configuration for global rate limiting.

    Attributes:
        enabled: Whether rate limiting is active
        slots_per_second: Maximum operations per second (default 2)
        burst_size: Maximum burst capacity (default 5)
        per_operation_limits: Optional per-operation type limits
    """
    enabled: bool = True
    slots_per_second: float = 2.0
    burst_size: int = 5
    per_operation_limits: Dict[str, float] = field(default_factory=dict)

    def get_limit_for_operation(self, operation_type: str) -> float:
        """Get rate limit for specific operation type."""
        return self.per_operation_limits.get(operation_type, self.slots_per_second)


@dataclass
class SlotMetrics:
    """Metrics for rate limit slot usage.

    Attributes:
        total_acquired: Total slots acquired
        total_released: Total slots released
        total_waits: Times had to wait for slot
        total_wait_time: Total time spent waiting (seconds)
        rejections: Times acquire was rejected (timeout)
    """
    total_acquired: int = 0
    total_released: int = 0
    total_waits: int = 0
    total_wait_time: float = 0.0
    rejections: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'total_acquired': self.total_acquired,
            'total_released': self.total_released,
            'total_waits': self.total_waits,
            'total_wait_time': round(self.total_wait_time, 3),
            'rejections': self.rejections,
            'active_slots': self.total_acquired - self.total_released,
        }


class GlobalRateLimitCoordinator:
    """Thread-safe global rate limit coordinator for parallel operations.

    Uses token bucket algorithm to limit requests per second across
    all YouTube operations (captions, downloads, API calls).

    Singleton pattern ensures all workers share the same coordinator.

    Example:
        coordinator = GlobalRateLimitCoordinator()

        # Before making request
        if coordinator.acquire_slot('caption'):
            try:
                result = fetch_captions(video_id)
            finally:
                coordinator.release_slot('caption')

    Configuration (via rate_limit section in config):
        rate_limit:
          global:
            enabled: true
            slots_per_second: 2
    """

    _instance: Optional['GlobalRateLimitCoordinator'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, config: Optional[RateLimitConfig] = None) -> 'GlobalRateLimitCoordinator':
        """Singleton pattern for global state sharing."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[RateLimitConfig] = None):
        if getattr(self, '_initialized', False):
            # Allow config updates on existing instance
            if config is not None:
                self._config = config
            return

        self._config = config or RateLimitConfig()
        self._state_lock = threading.RLock()

        # Token bucket state
        self._tokens: float = float(self._config.burst_size)
        self._last_refill_time: float = time.time()

        # Per-operation tracking
        self._operation_metrics: Dict[str, SlotMetrics] = {}
        self._active_slots: Dict[str, int] = {}

        self._initialized = True
        logger.debug(
            f"GlobalRateLimitCoordinator initialized: "
            f"enabled={self._config.enabled}, "
            f"slots_per_second={self._config.slots_per_second}"
        )

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time (called with lock held)."""
        now = time.time()
        elapsed = now - self._last_refill_time

        # Add tokens based on elapsed time
        new_tokens = elapsed * self._config.slots_per_second
        self._tokens = min(
            self._tokens + new_tokens,
            float(self._config.burst_size)
        )
        self._last_refill_time = now

    def _get_metrics(self, operation_type: str) -> SlotMetrics:
        """Get or create metrics for operation type (called with lock held)."""
        if operation_type not in self._operation_metrics:
            self._operation_metrics[operation_type] = SlotMetrics()
        return self._operation_metrics[operation_type]

    def acquire_slot(
        self,
        operation_type: str,
        timeout: float = 30.0,
        block: bool = True
    ) -> bool:
        """Acquire a rate limit slot for an operation.

        Args:
            operation_type: Type of operation ('caption', 'download', 'api')
            timeout: Maximum time to wait for slot (seconds)
            block: If False, return immediately if no slot available

        Returns:
            True if slot acquired, False if timeout/rejected
        """
        if not self._config.enabled:
            return True

        start_time = time.time()

        with self._state_lock:
            metrics = self._get_metrics(operation_type)

            while True:
                self._refill_tokens()

                if self._tokens >= 1.0:
                    # Slot available
                    self._tokens -= 1.0
                    metrics.total_acquired += 1
                    self._active_slots[operation_type] = \
                        self._active_slots.get(operation_type, 0) + 1

                    logger.debug(
                        f"Slot acquired for {operation_type}: "
                        f"remaining_tokens={self._tokens:.2f}"
                    )
                    return True

                if not block:
                    metrics.rejections += 1
                    return False

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    metrics.rejections += 1
                    logger.warning(
                        f"Slot acquisition timeout for {operation_type} "
                        f"after {elapsed:.1f}s"
                    )
                    return False

                # Calculate wait time until next token
                wait_time = min(
                    (1.0 - self._tokens) / self._config.slots_per_second,
                    timeout - elapsed,
                    0.1  # Max 100ms wait per iteration
                )

                if wait_time > 0:
                    metrics.total_waits += 1
                    # Release lock while waiting
                    self._state_lock.release()
                    try:
                        time.sleep(wait_time)
                        metrics.total_wait_time += wait_time
                    finally:
                        self._state_lock.acquire()

    def release_slot(self, operation_type: str) -> None:
        """Release a rate limit slot.

        Args:
            operation_type: Type of operation being released
        """
        if not self._config.enabled:
            return

        with self._state_lock:
            metrics = self._get_metrics(operation_type)
            metrics.total_released += 1

            current = self._active_slots.get(operation_type, 0)
            if current > 0:
                self._active_slots[operation_type] = current - 1

            logger.debug(f"Slot released for {operation_type}")

    def get_active_slots(self, operation_type: Optional[str] = None) -> int:
        """Get count of currently active slots.

        Args:
            operation_type: Specific type, or None for total

        Returns:
            Number of active slots
        """
        with self._state_lock:
            if operation_type:
                return self._active_slots.get(operation_type, 0)
            return sum(self._active_slots.values())

    def get_available_tokens(self) -> float:
        """Get current available token count."""
        with self._state_lock:
            self._refill_tokens()
            return self._tokens

    def get_metrics(self, operation_type: Optional[str] = None) -> dict:
        """Get metrics for specific operation or all operations.

        Args:
            operation_type: Specific type, or None for all

        Returns:
            Dictionary of metrics
        """
        with self._state_lock:
            if operation_type:
                return self._get_metrics(operation_type).to_dict()

            return {
                op_type: metrics.to_dict()
                for op_type, metrics in self._operation_metrics.items()
            }

    def get_status(self) -> dict:
        """Get current coordinator status."""
        with self._state_lock:
            self._refill_tokens()
            return {
                'enabled': self._config.enabled,
                'slots_per_second': self._config.slots_per_second,
                'burst_size': self._config.burst_size,
                'available_tokens': round(self._tokens, 2),
                'active_slots': dict(self._active_slots),
                'total_active': sum(self._active_slots.values()),
            }

    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._config.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable rate limiting."""
        with self._state_lock:
            self._config.enabled = enabled
            logger.info(f"Rate limiting {'enabled' if enabled else 'disabled'}")

    def update_config(self, config: RateLimitConfig) -> None:
        """Update configuration."""
        with self._state_lock:
            self._config = config
            # Reset tokens to new burst size
            self._tokens = min(self._tokens, float(config.burst_size))
            logger.info(
                f"Rate limit config updated: "
                f"slots_per_second={config.slots_per_second}, "
                f"burst_size={config.burst_size}"
            )
