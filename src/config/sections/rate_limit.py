"""Rate limiting configuration: Global rate limit settings for the pipeline.

Centralizes all rate limit settings in one config section (US-35-010).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict

__all__ = [
    'RateLimitConfig',
    'RateLimitBudgetConfig',
]


@dataclass
class RateLimitBudgetConfig:
    """Rate limit budget configuration for controlling resource usage.

    Controls how many resources (rotations, backoff time, VPN switches) are
    available for rate limit recovery before escalating or giving up.
    """
    # Maximum cookie rotations per session (0 = unlimited)
    max_rotations: int = 10

    # Maximum total backoff time per session (seconds)
    # After this much delay, skip backoff and escalate immediately
    max_backoff_time: float = 600.0

    # Maximum VPN switches per session (0 = unlimited)
    max_vpn_switches: int = 3

    # Tier-specific attempt limits
    # Map of tier name to max attempts
    tier_max_attempts: Optional[Dict[str, int]] = None

    # Adaptive cooldown optimization (US-123-006)
    # Enable adaptive cooldown based on historical recovery times
    adaptive_cooldown_enabled: bool = False

    # Number of recent recoveries to consider for cooldown calculation
    cooldown_history: int = 10

    # Default cooldown duration if no history available (seconds)
    default_cooldown_seconds: float = 30.0

    # Enable tier-isolated budgets (US-109-007)
    tier_isolation_enabled: bool = False

    # Cross-session state persistence (US-123-007)
    # Enable saving/loading rate limit state across pipeline sessions
    state_persistence_enabled: bool = False

    # Reset state entries older than this threshold (seconds)
    # State older than this will be skipped during restoration
    stale_state_threshold: float = 3600.0  # 1 hour default

    def __post_init__(self):
        """Convert dict fields to proper types."""
        # Handle tier_max_attempts as dict from YAML
        if isinstance(self.tier_max_attempts, dict):
            self.tier_max_attempts = self.tier_max_attempts
        elif self.tier_max_attempts is None:
            self.tier_max_attempts = {}


@dataclass
class RateLimitConfig:
    """Global rate limiting settings for downloads and API calls.

    These settings control the rate at which requests are made to external
    services to avoid triggering rate limits or getting blocked.

    Chain-of-thought: Centralized rate limiting prevents scattered config values
    Reasoning: One place to tune all rate limit behavior
    Decision: Reasonable defaults that work for most use cases
    """
    # Slots per second: how many download/request slots available per second
    # Higher values = more aggressive downloading
    slots_per_second: float = 0.5

    # Burst size: how many requests can be made in a burst before rate limiting kicks in
    # Allows short bursts above the steady-state rate
    burst_size: int = 3

    # Jitter factor: randomization factor for request timing (0.0-1.0)
    # Adds randomness to avoid detection patterns: actual_delay = delay * (1 + random * jitter)
    jitter_factor: float = 0.2

    # Maximum backoff time in seconds when rate limited
    # After this limit, requests will proceed even if rate limited
    max_backoff_seconds: float = 60.0

    # US-153-009: Predictive rate limit settings
    # Prediction window hours: how far back to look for historical patterns
    prediction_window_hours: int = 24

    # Backoff multiplier: multiplier applied to delays when historical failure rate is high
    # Higher values = more aggressive backoff when rate limits are predicted
    backoff_multiplier: float = 2.0

    # Rate limit budget configuration
    # Controls resource limits for rate limit recovery
    budget: RateLimitBudgetConfig = field(default_factory=RateLimitBudgetConfig)

    def __post_init__(self):
        """Convert dict fields to proper types."""
        # Handle budget as dict from YAML
        if self.budget is None or isinstance(self.budget, dict):
            self.budget = RateLimitBudgetConfig(**(self.budget or {}))
