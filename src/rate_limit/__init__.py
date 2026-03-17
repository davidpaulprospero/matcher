"""
Global rate limit coordination for parallel YouTube operations.

Provides unified rate limiting across caption fetching, video downloading,
and API calls to prevent overwhelming YouTube's servers.
"""

from src.rate_limit.coordinator import GlobalRateLimitCoordinator

__all__ = ['GlobalRateLimitCoordinator']
