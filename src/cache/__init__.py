"""
Unified cache package for matcher-pipeline-stages.

Provides base classes and utilities for all pipeline caches.

Created: January 7, 2026
Part of cache consolidation refactoring to eliminate ~400 LOC duplication.

Usage:
    from src.cache import BaseCache, compute_hash, normalize_path

    class MyCache(BaseCache):
        def _serialize_entry(self, entry):
            return {'data': entry.data, 'cached_at': entry.cached_at}

        def _deserialize_entry(self, data):
            return CacheEntry(data=data['data'], cached_at=data['cached_at'], key='')

    cache = MyCache(cache_dir=".cache/my_cache")
    cache.set("key", "value")
    result = cache.get("key")
"""

from .base import BaseCache, CacheEntry
from .utils import (
    compute_hash,
    file_content_hash,
    file_metadata_hash,
    normalize_path,
    batch_hash,
    text_hash
)
from .types import CacheStats, CacheConfig

__all__ = [
    # Base classes
    'BaseCache',
    'CacheEntry',

    # Utilities
    'compute_hash',
    'file_content_hash',
    'file_metadata_hash',
    'normalize_path',
    'batch_hash',
    'text_hash',

    # Types
    'CacheStats',
    'CacheConfig',
]

__version__ = '1.0.0'
