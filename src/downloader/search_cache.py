"""
Search Results Cache for Video Metadata

Caches YouTube search results to avoid repeated API calls for the same keyword+tier.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..cache import BaseCache, CacheEntry

logger = logging.getLogger(__name__)


@dataclass
class CachedSearchResult:
    """Cached search result data"""
    keyword: str
    tier: str
    search_pool: int
    videos: List[Dict[str, Any]]  # Video metadata list
    cached_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CachedSearchResult':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SearchResultsCache(BaseCache):
    """
    Cache for YouTube search results.
    
    Key format: hash(keyword + tier + search_pool)
    TTL: 24 hours (search results can change frequently)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        """
        Initialize search results cache.
        
        Args:
            cache_dir: Directory for cache storage (default: ~/.matcher_cache/search)
            ttl_hours: Time-to-live in hours (default: 24)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".matcher_cache" / "search"
        
        ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else 0
        
        super().__init__(
            cache_dir=cache_dir,
            index_name="search_cache_index.json",
            ttl_seconds=ttl_seconds,
            auto_save=True
        )
        
        logger.debug(f"SearchResultsCache initialized: {cache_dir} (TTL: {ttl_hours}h)")
    
    def _make_key(self, keyword: str, tier: str, search_pool: int) -> str:
        """Generate cache key from search parameters"""
        key_string = f"{keyword.lower().strip()}|{tier}|{search_pool}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _serialize_entry(self, entry: CacheEntry) -> dict:
        """Serialize cache entry to dict"""
        return {
            'data': entry.data,
            'cached_at': entry.cached_at,
            'metadata': entry.metadata
        }
    
    def _deserialize_entry(self, data: dict) -> CacheEntry:
        """Deserialize dict to cache entry"""
        return CacheEntry(
            data=data['data'],
            cached_at=data['cached_at'],
            key='',  # Will be set by caller
            metadata=data.get('metadata', {})
        )
    
    def get_search_result(
        self, 
        keyword: str, 
        tier: str, 
        search_pool: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search result if available.
        
        Args:
            keyword: Search keyword
            tier: Duration tier (short, medium, long, longer)
            search_pool: Number of results requested
            
        Returns:
            List of video metadata dicts or None if not cached/expired
        """
        key = self._make_key(keyword, tier, search_pool)
        entry = self.get(key)
        
        if entry:
            videos = entry.data.get('videos', [])
            logger.debug(f"Search cache HIT: '{keyword}' ({tier}) -> {len(videos)} videos")
            return videos
        
        logger.debug(f"Search cache MISS: '{keyword}' ({tier})")
        return None
    
    def set_search_result(
        self,
        keyword: str,
        tier: str,
        search_pool: int,
        videos: List[Dict[str, Any]]
    ) -> None:
        """
        Cache search result.
        
        Args:
            keyword: Search keyword
            tier: Duration tier
            search_pool: Number of results requested
            videos: List of video metadata dicts
        """
        key = self._make_key(keyword, tier, search_pool)
        
        from datetime import datetime
        cached_result = {
            'keyword': keyword,
            'tier': tier,
            'search_pool': search_pool,
            'videos': videos,
            'cached_at': datetime.now().isoformat()
        }
        
        self.set(key, cached_result, metadata={
            'video_count': len(videos),
            'keyword': keyword,
            'tier': tier
        })
        
        logger.debug(f"Search cache SET: '{keyword}' ({tier}) -> {len(videos)} videos")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hits': self._hits,
            'misses': self._misses,
            'entries': self._count_entries(),
            'hit_rate': self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
            'bytes_saved': self._bytes_saved
        }
    
    def invalidate_keyword(self, keyword: str) -> int:
        """
        Invalidate all cache entries for a keyword.
        
        Args:
            keyword: Keyword to invalidate
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        keyword_lower = keyword.lower().strip()
        
        for key in list(self.index.keys()):
            entry_data = self.index.get(key, {})
            metadata = entry_data.get('metadata', {})
            if metadata.get('keyword', '').lower().strip() == keyword_lower:
                self.delete(key)
                count += 1
        
        if count > 0:
            logger.info(f"Invalidated {count} cache entries for keyword: '{keyword}'")
        
        return count
