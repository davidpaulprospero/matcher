"""
Unified caching for LLM responses with TTL support.
"""

import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMCache:
    """
    File-based cache for LLM responses with TTL (time-to-live) support.

    Cache structure:
    .cache/llm_responses/
    ├── gemini/
    │   ├── prefix_abc123.json
    │   └── prefix_def456.json
    ├── anthropic/
    └── ollama/

    Each cache file contains:
    {
        "text": "response text",
        "parsed_data": {...},  # Optional
        "cached_at": 1234567890.0,
        "provider": "gemini",
        "model": "gemini-2.0-flash"
    }
    """

    def __init__(self, base_dir: str, provider: str, ttl_hours: int = 24):
        """
        Initialize cache.

        Args:
            base_dir: Base cache directory (e.g., ".cache/llm_responses")
            provider: Provider name (gemini, anthropic, ollama)
            ttl_hours: Time-to-live in hours (0 = never expire)
        """
        self.cache_dir = Path(base_dir) / provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.provider = provider

    def _cache_key(self, request: "LLMRequest") -> str:
        """
        Generate cache key from request.

        Key is based on:
        - Prompt text
        - System prompt
        - Max tokens
        - Temperature
        - Response format

        Returns:
            16-character MD5 hash
        """
        from .base import ResponseFormat

        key_data = {
            "prompt": request.prompt,
            "system_prompt": request.system_prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "response_format": request.response_format.value if isinstance(request.response_format, ResponseFormat) else str(request.response_format)
        }

        # Include images if present (hash the image bytes)
        if request.images:
            images_hash = hashlib.md5(b"".join(request.images)).hexdigest()
            key_data["images_hash"] = images_hash

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get(self, request: "LLMRequest") -> Optional[Dict[str, Any]]:
        """
        Get cached response if it exists and hasn't expired.

        Args:
            request: LLM request object

        Returns:
            Cached response dict or None if not found/expired
        """
        cache_key = self._cache_key(request)
        cache_file = self.cache_dir / f"{request.cache_key_prefix}_{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check TTL if enabled
            if self.ttl_seconds > 0:
                cached_at = data.get('cached_at', 0)
                age_seconds = time.time() - cached_at

                if age_seconds > self.ttl_seconds:
                    # Expired - delete file
                    cache_file.unlink()
                    logger.debug(f"Cache expired for key {cache_key} (age: {age_seconds/3600:.1f}h)")
                    return None

            logger.debug(f"Cache hit for key {cache_key}")
            return data

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            # Delete corrupt cache file
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None

    def set(self, request: "LLMRequest", response: "LLMResponse"):
        """
        Cache an LLM response.

        Args:
            request: LLM request object
            response: LLM response object
        """
        cache_key = self._cache_key(request)
        cache_file = self.cache_dir / f"{request.cache_key_prefix}_{cache_key}.json"

        data = {
            "text": response.text,
            "parsed_data": response.parsed_data,
            "cached_at": time.time(),
            "provider": response.provider,
            "model": response.model
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Cached response for key {cache_key}")

        except (OSError, TypeError) as e:
            logger.warning(f"Failed to cache LLM response: {e}")

    def clear(self, prefix: Optional[str] = None, older_than_hours: Optional[int] = None):
        """
        Clear cache entries.

        Args:
            prefix: Only clear entries with this prefix (None = all)
            older_than_hours: Only clear entries older than this (None = all)
        """
        cleared = 0

        for cache_file in self.cache_dir.glob("*.json"):
            # Check prefix
            if prefix and not cache_file.stem.startswith(prefix):
                continue

            # Check age
            if older_than_hours is not None:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cached_at = data.get('cached_at', 0)
                    age_hours = (time.time() - cached_at) / 3600

                    if age_hours < older_than_hours:
                        continue
                except (json.JSONDecodeError, OSError):
                    pass  # Delete corrupt files

            # Delete file
            try:
                cache_file.unlink()
                cleared += 1
            except OSError as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {cleared} cache entries for provider {self.provider}")

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with stats: total_files, total_size_mb, oldest_entry_hours
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_files = len(cache_files)
        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)

        oldest_entry_hours = None
        if cache_files:
            oldest_time = float('inf')
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cached_at = data.get('cached_at', time.time())
                    oldest_time = min(oldest_time, cached_at)
                except (json.JSONDecodeError, OSError):
                    pass

            if oldest_time != float('inf'):
                oldest_entry_hours = (time.time() - oldest_time) / 3600

        return {
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "oldest_entry_hours": round(oldest_entry_hours, 1) if oldest_entry_hours else None,
            "provider": self.provider
        }
