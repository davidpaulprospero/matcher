"""
LLM Filter Results Cache

Caches LLM title filtering results to avoid repeated API calls for the same keyword+video set.
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
class CachedLLMFilterResult:
    """Cached LLM filter result data"""
    keyword: str
    video_ids_hash: str
    approved_videos: List[Dict[str, Any]]  # Approved video metadata with relevance scores
    cached_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CachedLLMFilterResult':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LLMFilterCache(BaseCache):
    """
    Cache for LLM title filter results.
    
    Key format: hash(keyword + video_ids_hash)
    TTL: 7 days (relevance decisions are relatively stable)
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_days: int = 7):
        """
        Initialize LLM filter cache.
        
        Args:
            cache_dir: Directory for cache storage (default: ~/.matcher_cache/llm_filter)
            ttl_days: Time-to-live in days (default: 7)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".matcher_cache" / "llm_filter"
        
        ttl_seconds = ttl_days * 24 * 3600 if ttl_days > 0 else 0
        
        super().__init__(
            cache_dir=cache_dir,
            index_name="llm_filter_cache_index.json",
            ttl_seconds=ttl_seconds,
            auto_save=True
        )
        
        logger.debug(f"LLMFilterCache initialized: {cache_dir} (TTL: {ttl_days}d)")
    
    def _make_video_ids_hash(self, videos: List[Dict[str, Any]]) -> str:
        """Generate hash from sorted video IDs"""
        video_ids = sorted([v.get('id', '') for v in videos if v.get('id')])
        ids_string = ','.join(video_ids)
        return hashlib.md5(ids_string.encode()).hexdigest()[:16]
    
    def _make_key(self, keyword: str, video_ids_hash: str) -> str:
        """Generate cache key from filter parameters"""
        key_string = f"{keyword.lower().strip()}|{video_ids_hash}"
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
    
    def get_filter_result(
        self, 
        keyword: str, 
        videos: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached LLM filter result if available.
        
        Args:
            keyword: Search keyword
            videos: List of video metadata dicts (used to compute cache key)
            
        Returns:
            List of approved video metadata dicts or None if not cached/expired
        """
        video_ids_hash = self._make_video_ids_hash(videos)
        key = self._make_key(keyword, video_ids_hash)
        entry = self.get(key)
        
        if entry:
            approved = entry.data.get('approved_videos', [])
            logger.debug(f"LLM filter cache HIT: '{keyword}' -> {len(approved)} approved videos")
            return approved
        
        logger.debug(f"LLM filter cache MISS: '{keyword}' ({len(videos)} videos)")
        return None
    
    def set_filter_result(
        self,
        keyword: str,
        videos: List[Dict[str, Any]],
        approved_videos: List[Dict[str, Any]]
    ) -> None:
        """
        Cache LLM filter result.
        
        Args:
            keyword: Search keyword
            videos: Original list of video metadata dicts
            approved_videos: List of approved video metadata dicts with relevance scores
        """
        video_ids_hash = self._make_video_ids_hash(videos)
        key = self._make_key(keyword, video_ids_hash)
        
        from datetime import datetime
        cached_result = {
            'keyword': keyword,
            'video_ids_hash': video_ids_hash,
            'approved_videos': approved_videos,
            'cached_at': datetime.now().isoformat()
        }
        
        self.set(key, cached_result, metadata={
            'approved_count': len(approved_videos),
            'input_count': len(videos),
            'keyword': keyword
        })
        
        logger.debug(f"LLM filter cache SET: '{keyword}' -> {len(approved_videos)}/{len(videos)} approved")
    
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
            logger.info(f"Invalidated {count} LLM filter cache entries for keyword: '{keyword}'")
        
        return count
