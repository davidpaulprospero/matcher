"""
HealerResultCache - Cache successful healing actions to prevent redundant healing.

Part of US-64-007: Add healer result caching for idempotent healing actions.

The cache:
- Keys by error signature (type + message hash + stage)
- Skips redundant healing attempts if identical error was recently healed
- Has configurable TTL (default: 5 minutes)
- Invalidates when config changes (via config_modified flag)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import HealerResult

logger = logging.getLogger(__name__)


@dataclass
class HealerCacheEntry:
    """Cached healer result entry."""
    error_signature: str
    healer_name: str
    action: str
    message: str
    cached_at: float
    config_hash: Optional[str] = None


@dataclass
class HealerCacheConfig:
    """Configuration for healer result cache."""
    enabled: bool = True
    ttl_seconds: float = 300.0  # 5 minutes default
    max_entries: int = 100


class HealerResultCache:
    """
    Cache for successful healing actions to prevent redundant healing attempts.

    The cache uses error signatures (type + message hash + stage) as keys,
    allowing the orchestrator to skip healing for errors that were recently
    healed successfully.

    Features:
    - TTL-based expiration (default 5 minutes)
    - Config-change invalidation via config_hash
    - Memory-only storage (no persistence needed)
    - Thread-safe operations

    Usage:
        cache = HealerResultCache(HealerCacheConfig(ttl_seconds=300))

        # Before attempting heal
        cached = cache.get(error, stage_name)
        if cached:
            logger.info(f"Skipping heal: identical error recently fixed by {cached.healer_name}")
            return HealerResult.success(cached.message, cached.action)

        # After successful heal
        result = healer.fix(error, state, stage_name)
        if result.success:
            cache.store(error, stage_name, result)
    """

    def __init__(self, config: Optional[HealerCacheConfig] = None):
        """
        Initialize the healer result cache.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self.config = config or HealerCacheConfig()
        self._cache: Dict[str, HealerCacheEntry] = {}
        self._config_hash: Optional[str] = None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

        logger.debug(
            f"HealerResultCache initialized: enabled={self.config.enabled}, "
            f"ttl={self.config.ttl_seconds}s, max_entries={self.config.max_entries}"
        )

    def _make_error_signature(self, error: Exception, stage_name: str) -> str:
        """
        Create error signature from error type, message hash, and stage.

        The signature uniquely identifies an error condition while avoiding
        overly long keys. Message is hashed to handle long error messages.

        Args:
            error: The exception that occurred.
            stage_name: Name of the stage where error occurred.

        Returns:
            Error signature string: "ErrorType:message_hash:stage_name"
        """
        error_type = type(error).__name__
        error_msg = str(error)

        # Hash the message to handle long error strings
        msg_hash = hashlib.md5(error_msg.encode('utf-8', errors='replace')).hexdigest()[:12]

        return f"{error_type}:{msg_hash}:{stage_name}"

    def _is_expired(self, entry: HealerCacheEntry) -> bool:
        """Check if cache entry is expired based on TTL."""
        if self.config.ttl_seconds <= 0:
            return False
        age = time.time() - entry.cached_at
        return age > self.config.ttl_seconds

    def _is_config_stale(self, entry: HealerCacheEntry) -> bool:
        """Check if entry was cached with a different config."""
        if self._config_hash is None or entry.config_hash is None:
            return False
        return entry.config_hash != self._config_hash

    def _prune_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    def _enforce_max_entries(self) -> None:
        """Enforce max_entries limit by removing oldest entries."""
        if len(self._cache) <= self.config.max_entries:
            return

        # Sort by cached_at and remove oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].cached_at
        )
        excess = len(self._cache) - self.config.max_entries
        for key, _ in sorted_entries[:excess]:
            del self._cache[key]

    def set_config_hash(self, config_hash: str) -> None:
        """
        Set current config hash for cache invalidation.

        When config changes, cached healing results may no longer be valid.
        Setting a new config_hash will cause entries with the old hash to
        be treated as stale.

        Args:
            config_hash: Hash of current configuration.
        """
        if self._config_hash is not None and self._config_hash != config_hash:
            logger.info(
                f"[healer_cache] Config changed, invalidating cache "
                f"({len(self._cache)} entries)"
            )
            self._invalidations += 1
            # Clear cache on config change
            self._cache.clear()

        self._config_hash = config_hash

    def invalidate_on_config_change(self) -> None:
        """
        Explicitly invalidate cache due to config change.

        Called when config_modified flag is True in orchestrator.
        """
        logger.info(f"[healer_cache] Invalidating cache due to config change")
        self._invalidations += 1
        self._cache.clear()

    def get(
        self,
        error: Exception,
        stage_name: str
    ) -> Optional[HealerCacheEntry]:
        """
        Get cached healer result for an error.

        Returns None if:
        - Cache is disabled
        - No cached entry exists
        - Entry is expired (TTL exceeded)
        - Entry has stale config hash

        Args:
            error: The exception that occurred.
            stage_name: Name of the stage where error occurred.

        Returns:
            HealerCacheEntry if found and valid, None otherwise.
        """
        if not self.config.enabled:
            return None

        # Periodically prune expired entries
        self._prune_expired()

        signature = self._make_error_signature(error, stage_name)

        if signature not in self._cache:
            self._misses += 1
            logger.debug(f"[healer_cache] Cache miss: {signature}")
            return None

        entry = self._cache[signature]

        # Check expiration
        if self._is_expired(entry):
            del self._cache[signature]
            self._misses += 1
            logger.debug(f"[healer_cache] Cache expired: {signature}")
            return None

        # Check config staleness
        if self._is_config_stale(entry):
            del self._cache[signature]
            self._misses += 1
            logger.debug(f"[healer_cache] Cache config stale: {signature}")
            return None

        self._hits += 1
        logger.info(
            f"[healer_cache] Cache hit: {signature} "
            f"(healer={entry.healer_name}, age={time.time() - entry.cached_at:.1f}s)"
        )
        return entry

    def store(
        self,
        error: Exception,
        stage_name: str,
        result: 'HealerResult',
        healer_name: str
    ) -> bool:
        """
        Store successful healer result in cache.

        Only successful results should be cached - failed healing attempts
        should not prevent future attempts.

        Args:
            error: The exception that was healed.
            stage_name: Name of the stage where error occurred.
            result: The successful HealerResult.
            healer_name: Name of the healer that fixed the error.

        Returns:
            True if stored, False if cache disabled or result not successful.
        """
        if not self.config.enabled:
            return False

        if not result.success:
            return False

        signature = self._make_error_signature(error, stage_name)

        entry = HealerCacheEntry(
            error_signature=signature,
            healer_name=healer_name,
            action=result.action.value if result.action else "unknown",
            message=result.message or "",
            cached_at=time.time(),
            config_hash=self._config_hash
        )

        self._cache[signature] = entry
        self._enforce_max_entries()

        logger.debug(
            f"[healer_cache] Cached result: {signature} "
            f"(healer={healer_name}, action={entry.action})"
        )
        return True

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"[healer_cache] Cleared {count} entries")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache metrics.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'enabled': self.config.enabled,
            'entries': len(self._cache),
            'max_entries': self.config.max_entries,
            'ttl_seconds': self.config.ttl_seconds,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'invalidations': self._invalidations,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for dashboard metrics (US-68-008).

        Returns:
            Dict with keys: hits, misses, hit_rate, invalidation_count, entry_count
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'invalidation_count': self._invalidations,
            'entry_count': len(self._cache),
        }

    def __repr__(self) -> str:
        return f"HealerResultCache(entries={len(self._cache)}, ttl={self.config.ttl_seconds}s)"
