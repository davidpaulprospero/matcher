"""Rate limiting configuration: Global rate limit settings for the pipeline.

Centralizes all rate limit settings in one config section (US-35-010).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    'RateLimitConfig',
]


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
