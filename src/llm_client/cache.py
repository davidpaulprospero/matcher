"""
Unified caching for LLM responses with TTL support.

Quality tier tracking:
- high: confidence >= 0.8
- medium: 0.5 <= confidence < 0.8
- low: confidence < 0.5
"""

import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# Quality tier thresholds
QUALITY_TIER_HIGH_THRESHOLD = 0.8
QUALITY_TIER_MEDIUM_THRESHOLD = 0.5


def compute_quality_tier(confidence: float) -> str:
    """
    Compute quality tier from confidence score.

    Args:
        confidence: Confidence score (0.0-1.0)

    Returns:
        Quality tier: 'high', 'medium', or 'low'
    """
    if confidence >= QUALITY_TIER_HIGH_THRESHOLD:
        return 'high'
    elif confidence >= QUALITY_TIER_MEDIUM_THRESHOLD:
        return 'medium'
    else:
        return 'low'


def extract_confidence_from_response(response: "LLMResponse") -> Optional[float]:
    """
    Extract confidence score from LLM response parsed data.

    The confidence is typically in the JSON response from matching LLM calls.
    We check common locations where confidence might be stored.

    Args:
        response: LLM response object

    Returns:
        Confidence score or None if not found
    """
    if response.parsed_data is None:
        return None

    # Handle list of results (batch responses)
    if isinstance(response.parsed_data, list):
        # Use first result's confidence if available
        if response.parsed_data and isinstance(response.parsed_data[0], dict):
            return response.parsed_data[0].get('confidence')
        return None

    # Handle single result dict
    if isinstance(response.parsed_data, dict):
        return response.parsed_data.get('confidence')

    return None


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
        "model": "gemini-2.0-flash",
        "quality_tier": "high"  # high, medium, or low (based on confidence)
    }

    Quality tiers:
    - high: confidence >= 0.8
    - medium: 0.5 <= confidence < 0.8
    - low: confidence < 0.5

    When skip_low_quality=True, cache hits with quality_tier='low' are skipped,
    forcing a fresh LLM evaluation.
    """

    def __init__(self, base_dir: str, provider: str, ttl_hours: int = 24, skip_low_quality: bool = False):
        """
        Initialize cache.

        Args:
            base_dir: Base cache directory (e.g., ".cache/llm_responses")
            provider: Provider name (gemini, anthropic, ollama)
            ttl_hours: Time-to-live in hours (0 = never expire)
            skip_low_quality: If True, skip cache entries with quality_tier='low'
        """
        self.cache_dir = Path(base_dir) / provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.provider = provider
        self.skip_low_quality = skip_low_quality

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
            Cached response dict or None if not found/expired/low-quality-skipped.
            The dict includes 'quality_tier' field if available.
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

            # Get quality tier (might be missing from old cache entries)
            quality_tier = data.get('quality_tier', 'unknown')

            # Check for low-quality skip
            if self.skip_low_quality and quality_tier == 'low':
                # Extract confidence for logging
                confidence = None
                if data.get('parsed_data'):
                    if isinstance(data['parsed_data'], dict):
                        confidence = data['parsed_data'].get('confidence')
                    elif isinstance(data['parsed_data'], list) and data['parsed_data']:
                        confidence = data['parsed_data'][0].get('confidence') if isinstance(data['parsed_data'][0], dict) else None

                conf_str = f" (confidence: {confidence:.2f})" if confidence is not None else ""
                logger.info(f"Cache skip (low-quality): key={cache_key}, quality_tier=low{conf_str}")
                return None

            # Log cache hit with quality tier info
            if quality_tier != 'unknown':
                # Get confidence range for logging
                conf_range = self._get_confidence_range(quality_tier)
                logger.debug(f"Cache hit for key {cache_key}, quality_tier={quality_tier} {conf_range}")
            else:
                logger.debug(f"Cache hit for key {cache_key} (legacy entry, no quality_tier)")

            return data

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            # Delete corrupt cache file
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None

    def _get_confidence_range(self, quality_tier: str) -> str:
        """Get human-readable confidence range for a quality tier."""
        if quality_tier == 'high':
            return f"(conf >= {QUALITY_TIER_HIGH_THRESHOLD})"
        elif quality_tier == 'medium':
            return f"({QUALITY_TIER_MEDIUM_THRESHOLD} <= conf < {QUALITY_TIER_HIGH_THRESHOLD})"
        elif quality_tier == 'low':
            return f"(conf < {QUALITY_TIER_MEDIUM_THRESHOLD})"
        return ""

    def set(self, request: "LLMRequest", response: "LLMResponse"):
        """
        Cache an LLM response with quality tier tracking.

        Quality tier is computed from the confidence score in parsed_data:
        - high: confidence >= 0.8
        - medium: 0.5 <= confidence < 0.8
        - low: confidence < 0.5

        Args:
            request: LLM request object
            response: LLM response object
        """
        cache_key = self._cache_key(request)
        cache_file = self.cache_dir / f"{request.cache_key_prefix}_{cache_key}.json"

        # Extract confidence and compute quality tier
        confidence = extract_confidence_from_response(response)
        quality_tier = compute_quality_tier(confidence) if confidence is not None else 'unknown'

        data = {
            "text": response.text,
            "parsed_data": response.parsed_data,
            "cached_at": time.time(),
            "provider": response.provider,
            "model": response.model,
            "quality_tier": quality_tier
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Log with quality tier info
            if confidence is not None:
                logger.debug(f"Cached response for key {cache_key}, quality_tier={quality_tier} (confidence: {confidence:.2f})")
            else:
                logger.debug(f"Cached response for key {cache_key}, quality_tier=unknown (no confidence in response)")

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

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries based on TTL.

        Returns:
            Number of entries removed
        """
        if self.ttl_seconds <= 0:
            return 0  # TTL disabled, nothing expires

        removed = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                cached_at = data.get('cached_at', 0)
                age_seconds = time.time() - cached_at

                if age_seconds > self.ttl_seconds:
                    cache_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, OSError, KeyError):
                # Remove corrupt files
                try:
                    cache_file.unlink()
                    removed += 1
                except OSError:
                    pass

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired LLM cache entries")

        return removed
