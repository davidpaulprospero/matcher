"""
YouTube Search Cache - Persist search results across pipeline runs.

Caches YouTube video IDs by (keyword, tier) to ensure consistent results
across fresh runs and enable reuse of high-quality matches.
"""

from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class YouTubeSearchCache:
    """
    Cache YouTube search results to preserve video IDs across runs.

    Cache structure:
    .cache/youtube_search/
        {hash(keyword+tier)}.json:
            {
                "keyword": "...",
                "tier": "...",
                "video_ids": ["...", ...],
                "cached_at": "2024-01-01T12:00:00",
                "search_pool": 10
            }
    """

    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize YouTube search cache.

        Args:
            cache_dir: Base cache directory (default: .cache)
        """
        self.cache_root = Path(cache_dir) / "youtube_search"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, keyword: str, tier: str) -> str:
        """Generate cache key from keyword and tier."""
        key_str = f"{keyword}:{tier}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, keyword: str, tier: str) -> Path:
        """Get cache file path for keyword+tier."""
        cache_key = self._get_cache_key(keyword, tier)
        return self.cache_root / f"{cache_key}.json"

    def get(
        self,
        keyword: str,
        tier: str,
        max_age_days: Optional[int] = None
    ) -> Optional[List[str]]:
        """
        Get cached video IDs for keyword+tier.

        Args:
            keyword: Search keyword
            tier: Duration tier
            max_age_days: Maximum age in days (None = no limit)

        Returns:
            List of video IDs if cached, None otherwise
        """
        cache_path = self._get_cache_path(keyword, tier)

        if not cache_path.exists():
            logger.debug(f"[search_cache] GET MISS: {keyword}:{tier} (no cache file)")
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate cache structure
            if not isinstance(data, dict) or 'video_ids' not in data:
                logger.warning(f"Invalid cache structure: {cache_path}")
                return None

            # Check age if max_age_days specified
            if max_age_days is not None:
                cached_at = datetime.fromisoformat(data.get('cached_at', ''))
                age_days = (datetime.now() - cached_at).days
                if age_days > max_age_days:
                    logger.info(f"[search_cache] GET EXPIRED: {keyword}:{tier} ({age_days} days > {max_age_days} max)")
                    return None

            video_ids = data['video_ids']
            cached_at = data.get('cached_at', 'unknown')
            logger.info(f"[search_cache] GET HIT: {keyword}:{tier} → {len(video_ids)} videos (cached: {cached_at})")
            return video_ids

        except Exception as e:
            logger.warning(f"Failed to read cache for {keyword}:{tier}: {e}")
            return None

    def set(
        self,
        keyword: str,
        tier: str,
        video_ids: List[str],
        search_pool: int = 10
    ) -> None:
        """
        Cache video IDs for keyword+tier.

        Args:
            keyword: Search keyword
            tier: Duration tier
            video_ids: List of YouTube video IDs
            search_pool: Search pool size used
        """
        cache_path = self._get_cache_path(keyword, tier)

        try:
            data = {
                'keyword': keyword,
                'tier': tier,
                'video_ids': video_ids,
                'cached_at': datetime.now().isoformat(),
                'search_pool': search_pool
            }

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"[search_cache] SET: {keyword}:{tier} → {len(video_ids)} videos (pool={search_pool})")

        except Exception as e:
            logger.warning(f"Failed to write cache for {keyword}:{tier}: {e}")

    def remove_ids(
        self,
        keyword: str,
        tier: str,
        ids_to_remove: List[str]
    ) -> int:
        """
        Remove specific video IDs from cache (e.g., dead/private videos).

        Args:
            keyword: Search keyword
            tier: Duration tier
            ids_to_remove: List of video IDs to remove

        Returns:
            Number of IDs actually removed
        """
        cache_path = self._get_cache_path(keyword, tier)

        if not cache_path.exists():
            return 0

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_ids = data.get('video_ids', [])
            ids_set = set(ids_to_remove)
            new_ids = [vid for vid in original_ids if vid not in ids_set]

            removed_count = len(original_ids) - len(new_ids)

            if removed_count > 0:
                data['video_ids'] = new_ids
                data['last_pruned'] = datetime.now().isoformat()

                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"[search_cache] PRUNE: {keyword}:{tier} → removed {removed_count} dead IDs, {len(new_ids)} remaining")

            return removed_count

        except Exception as e:
            logger.warning(f"Failed to remove IDs from cache for {keyword}:{tier}: {e}")
            return 0

    def clear(self, keyword: Optional[str] = None, tier: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            keyword: Clear specific keyword (None = all)
            tier: Clear specific tier (None = all)

        Returns:
            Number of entries cleared
        """
        if keyword is None and tier is None:
            # Clear all
            count = 0
            for cache_file in self.cache_root.glob("*.json"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cache entries")
            return count

        # Clear specific keyword+tier
        cache_path = self._get_cache_path(keyword, tier)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cleared cache for {keyword}:{tier}")
            return 1

        return 0

    def list_entries(self) -> List[Tuple[str, str, int, datetime]]:
        """
        List all cache entries.

        Returns:
            List of (keyword, tier, video_count, cached_at) tuples
        """
        entries = []

        for cache_file in self.cache_root.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                entries.append((
                    data.get('keyword', '?'),
                    data.get('tier', '?'),
                    len(data.get('video_ids', [])),
                    datetime.fromisoformat(data.get('cached_at', ''))
                ))
            except Exception as e:
                logger.warning(f"Failed to read cache entry {cache_file}: {e}")

        return sorted(entries, key=lambda x: x[3], reverse=True)
