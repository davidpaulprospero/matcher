"""Common cache data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    cache_size_mb: float
    cache_dir: str
    index_file: str
    ttl_seconds: int
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable string representation"""
        return (
            f"CacheStats(entries={self.total_entries}, "
            f"size={self.cache_size_mb:.2f}MB, "
            f"ttl={self.ttl_seconds}s)"
        )


@dataclass
class CacheConfig:
    """Generic cache configuration"""
    cache_dir: str
    enabled: bool = True
    ttl_hours: int = 24
    max_size_mb: int = 0  # 0 = unlimited
    auto_cleanup: bool = True

    @property
    def ttl_seconds(self) -> int:
        """Convert TTL hours to seconds"""
        return self.ttl_hours * 3600

    def __post_init__(self):
        """Validate configuration"""
        if self.ttl_hours < 0:
            raise ValueError("ttl_hours must be non-negative")
        if self.max_size_mb < 0:
            raise ValueError("max_size_mb must be non-negative")
