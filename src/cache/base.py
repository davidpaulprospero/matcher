"""Base cache abstraction for all pipeline caches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Generic, List, Tuple
import json
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class EvictionResult:
    """Result from cache eviction operation"""
    entries_removed: int
    bytes_freed: int
    final_size_mb: float
    evicted_keys: List[str]
    dry_run: bool = False

@dataclass
class CacheEntry(Generic[T]):
    """Base cache entry with metadata"""
    data: T
    cached_at: float
    key: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCache(ABC, Generic[T]):
    """
    Abstract base class for all caches.

    Provides:
    - JSON index load/save
    - TTL expiration
    - Entry validation
    - Statistics tracking
    - Atomic writes

    Subclasses must implement:
    - _serialize_entry(): Convert entry to dict for JSON
    - _deserialize_entry(): Convert dict to entry object
    """

    def __init__(
        self,
        cache_dir: Path | str,
        index_name: str = "index.json",
        ttl_seconds: int = 0,
        auto_save: bool = True,
        pre_warm: bool = False
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Cache directory path
            index_name: Index filename (default: index.json)
            ttl_seconds: Time-to-live in seconds (0 = no expiration)
            auto_save: Auto-save index after modifications
            pre_warm: Pre-load entire index into memory at init (default: False)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.cache_dir / index_name
        self.ttl_seconds = ttl_seconds
        self.auto_save = auto_save

        self.index: Dict[str, Any] = {}
        self._load_index()

        # Statistics tracking
        self._hits: int = 0
        self._misses: int = 0
        self._bytes_saved: int = 0

        # Pre-warm cache if requested
        if pre_warm:
            self.warm_cache()

    # ==================== Abstract Methods ====================

    @abstractmethod
    def _serialize_entry(self, entry: CacheEntry[T]) -> Dict[str, Any]:
        """
        Serialize cache entry to dict for JSON storage.

        Args:
            entry: Cache entry to serialize

        Returns:
            Dict suitable for JSON serialization
        """
        pass

    @abstractmethod
    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry[T]:
        """
        Deserialize dict to cache entry.

        Args:
            data: Dict from JSON

        Returns:
            Cache entry object
        """
        pass

    # ==================== Index Management ====================

    def _load_index(self) -> None:
        """Load index from disk"""
        if not self.index_path.exists():
            self.index = self._get_default_index()
            logger.debug(f"Created new cache index at {self.index_path}")
            return

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
            logger.debug(f"Loaded cache index: {self._count_entries()} entries from {self.index_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache index: {e}. Starting fresh.")
            self.index = self._get_default_index()

    def _save_index(self) -> None:
        """Save index to disk with atomic write"""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write with temp file
            temp_path = self.index_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.index_path)

            logger.debug(f"Saved cache index: {self._count_entries()} entries to {self.index_path}")
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")

    def _get_default_index(self) -> Dict[str, Any]:
        """
        Override to provide custom default index structure.

        Returns:
            Default index dict
        """
        return {}

    def _count_entries(self) -> int:
        """
        Count total entries in index.

        Override for custom index structures (e.g., nested dicts).

        Returns:
            Number of cache entries
        """
        return len(self.index)

    # ==================== TTL & Validation ====================

    def _is_expired(self, cached_at: float) -> bool:
        """
        Check if entry is expired based on TTL.

        Args:
            cached_at: Timestamp when entry was cached

        Returns:
            True if expired, False otherwise
        """
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - cached_at) > self.ttl_seconds

    def _is_valid_entry(self, entry: CacheEntry[T]) -> bool:
        """
        Validate cache entry.

        Override to add custom validation (e.g., file existence checks).

        Args:
            entry: Cache entry to validate

        Returns:
            True if valid, False otherwise
        """
        if hasattr(entry, 'cached_at'):
            return not self._is_expired(entry.cached_at)
        return True

    # ==================== CRUD Operations ====================

    def get(self, key: str) -> Optional[CacheEntry[T]]:
        """
        Get entry by key, returns None if missing or expired.

        Args:
            key: Cache key

        Returns:
            Cache entry or None
        """
        if key not in self.index:
            self._misses += 1
            return None

        try:
            entry_data = self.index[key]
            entry = self._deserialize_entry(entry_data)

            if not self._is_valid_entry(entry):
                logger.debug(f"Cache entry expired or invalid: {key}")
                self.delete(key)
                self._misses += 1
                return None

            self._hits += 1
            # Track bytes saved (estimate from serialized entry size)
            self._bytes_saved += len(json.dumps(entry_data))
            return entry
        except Exception as e:
            logger.warning(f"Failed to deserialize cache entry {key}: {e}")
            self._misses += 1
            return None

    def set(self, key: str, value: T, metadata: Dict[str, Any] = None) -> None:
        """
        Set cache entry.

        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata dict
        """
        entry = CacheEntry(
            data=value,
            cached_at=time.time(),
            key=key,
            metadata=metadata or {}
        )

        self.index[key] = self._serialize_entry(entry)

        if self.auto_save:
            self._save_index()

    def delete(self, key: str) -> bool:
        """
        Delete cache entry.

        Args:
            key: Cache key

        Returns:
            True if entry existed, False otherwise
        """
        if key in self.index:
            del self.index[key]
            if self.auto_save:
                self._save_index()
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries"""
        self.index = self._get_default_index()
        self._save_index()
        logger.info(f"Cleared cache at {self.cache_dir}")

    def exists(self, key: str) -> bool:
        """
        Check if key exists and is valid.

        Args:
            key: Cache key

        Returns:
            True if exists and valid
        """
        return self.get(key) is not None

    # ==================== Bulk Operations ====================

    def get_all(self) -> Dict[str, CacheEntry[T]]:
        """
        Get all valid cache entries.

        Returns:
            Dict of key -> entry
        """
        entries = {}
        for key in list(self.index.keys()):
            entry = self.get(key)
            if entry:
                entries[key] = entry
        return entries

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = []
        for key in list(self.index.keys()):
            try:
                entry_data = self.index[key]
                entry = self._deserialize_entry(entry_data)
                if not self._is_valid_entry(entry):
                    expired_keys.append(key)
            except Exception as e:
                logger.warning(f"Failed to validate entry {key}: {e}")
                expired_keys.append(key)

        for key in expired_keys:
            del self.index[key]

        if expired_keys and self.auto_save:
            self._save_index()

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


    # ==================== Eviction Operations ====================

    def get_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        if self.cache_dir.exists():
            for file in self.cache_dir.rglob('*'):
                if file.is_file():
                    try:
                        total_size += file.stat().st_size
                    except OSError:
                        pass
        return total_size / (1024 * 1024)

    def get_entries_sorted(
        self,
        sort_key: str = "cached_at",
        reverse: bool = False
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get entries sorted by a key."""
        entries = []
        for key, entry_data in self.index.items():
            sort_value = entry_data.get(sort_key, entry_data.get('cached_at', 0))
            entries.append((key, entry_data, sort_value))
        entries.sort(key=lambda x: x[2], reverse=reverse)
        return [(key, data) for key, data, _ in entries]

    def evict_to_size(
        self,
        target_size_mb: float,
        strategy: str = "lru",
        dry_run: bool = False
    ) -> EvictionResult:
        """Evict entries to reach target size."""
        current_size = self.get_size_mb()
        if current_size <= target_size_mb:
            return EvictionResult(
                entries_removed=0,
                bytes_freed=0,
                final_size_mb=current_size,
                evicted_keys=[],
                dry_run=dry_run
            )

        sort_key = "last_used" if strategy == "lru" else "cached_at"
        sorted_entries = self.get_entries_sorted(sort_key, reverse=False)

        evicted_keys = []
        bytes_freed = 0
        entries_removed = 0

        for key, entry_data in sorted_entries:
            if self.get_size_mb() <= target_size_mb:
                break
            entry_size = len(json.dumps(entry_data))
            for file in self.cache_dir.glob(f"{key}*"):
                if file.is_file():
                    try:
                        entry_size += file.stat().st_size
                        if not dry_run:
                            file.unlink()
                    except OSError:
                        pass
            evicted_keys.append(key)
            bytes_freed += entry_size
            entries_removed += 1
            if not dry_run:
                if key in self.index:
                    del self.index[key]

        if not dry_run and entries_removed > 0:
            self._save_index()

        return EvictionResult(
            entries_removed=entries_removed,
            bytes_freed=bytes_freed,
            final_size_mb=self.get_size_mb(),
            evicted_keys=evicted_keys,
            dry_run=dry_run
        )

    def check_size_limit(self, max_size_mb: float, threshold: float = 0.9) -> bool:
        """Check if cache size exceeds threshold of max size."""
        if max_size_mb <= 0:
            return False
        current_size = self.get_size_mb()
        return current_size >= (max_size_mb * threshold)

    # ==================== Statistics ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics including hit/miss tracking.

        Returns:
            Dict with cache metrics including:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Ratio of hits to total requests (0.0 if no requests)
            - bytes_saved: Estimated bytes saved from cache hits
            - total_entries: Number of entries in cache
            - cache_size_mb: Cache size in MB
            - cache_dir: Cache directory path
            - index_file: Index file path
            - ttl_seconds: TTL configuration
        """
        total_entries = self._count_entries()

        # Calculate cache size
        cache_size = 0
        if self.cache_dir.exists():
            for file in self.cache_dir.rglob('*'):
                if file.is_file():
                    try:
                        cache_size += file.stat().st_size
                    except OSError:
                        pass

        # Calculate hit rate
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'bytes_saved': self._bytes_saved,
            'total_entries': total_entries,
            'cache_size_mb': cache_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'index_file': str(self.index_path),
            'ttl_seconds': self.ttl_seconds
        }

    def reset_stats(self) -> None:
        """Reset hit/miss/bytes_saved counters to zero."""
        self._hits = 0
        self._misses = 0
        self._bytes_saved = 0

    # ==================== Cache Warm-up ====================

    def warm_cache(self) -> Dict[str, Any]:
        """
        Pre-load entire index into memory for faster subsequent access.

        This method iterates through all index entries and deserializes them,
        keeping valid entries in memory. Expired or invalid entries are removed.

        Returns:
            Dict with warm-up statistics:
            - entries_loaded: Number of entries successfully loaded
            - entries_removed: Number of expired/invalid entries removed
            - elapsed_ms: Time taken in milliseconds
        """
        start_time = time.perf_counter()

        entries_loaded = 0
        entries_removed = 0
        keys_to_remove = []

        for key in list(self.index.keys()):
            try:
                entry_data = self.index[key]
                entry = self._deserialize_entry(entry_data)

                if self._is_valid_entry(entry):
                    entries_loaded += 1
                else:
                    keys_to_remove.append(key)
                    entries_removed += 1
            except Exception as e:
                logger.debug(f"Failed to deserialize entry during warm-up: {key}: {e}")
                keys_to_remove.append(key)
                entries_removed += 1

        # Remove invalid entries
        for key in keys_to_remove:
            del self.index[key]

        if keys_to_remove and self.auto_save:
            self._save_index()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(f"Cache warm-up complete: {entries_loaded} entries loaded in {elapsed_ms:.1f}ms")
        if entries_removed > 0:
            logger.debug(f"Removed {entries_removed} expired/invalid entries during warm-up")

        return {
            'entries_loaded': entries_loaded,
            'entries_removed': entries_removed,
            'elapsed_ms': elapsed_ms
        }

    def __repr__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(dir={self.cache_dir}, entries={self._count_entries()})"
