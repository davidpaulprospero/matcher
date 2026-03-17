"""API Fallback Handler - Fallback from YouTube API to yt-dlp.

Handles quota exhaustion scenarios by switching from YouTube Data API
to yt-dlp for video search operations. Also enriches search results
with engagement metrics (view_count, like_count, comment_count) from videos.list API.
"""

from __future__ import annotations

import difflib
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

# US-148-012: Import YouTube API error types for fallback handling
from .errors import (
    APIError,
    QuotaExceededError,
    YouTubeAPIError,
    YouTubeAPIQuotaExceededError,
    YouTubeAPIRateLimitedError,
    YouTubeAPIInvalidKeyError,
    YouTubeAPIPermissionDeniedError,
    YouTubeAPINetworkError,
)

# US-149-005: Import search deduplication module
from .search_deduplication import process_search_results

logger = logging.getLogger(__name__)

# Global metrics tracking for fallback events
_fallback_metrics: Dict[str, Any] = {
    "total_fallbacks": 0,
    "fallback_events": [],
}

# US-146-012: Global yt-dlp usage tracking for API vs yt-dlp ratio
_ytdlp_usage_count: int = 0

# US-153-012: Fallback level tracking for progressive degradation
_fallback_level_metrics: Dict[str, Any] = {
    "tier1_count": 0,  # Full API usage
    "tier2_count": 0,  # Reduced API (no metadata/engagement)
    "tier3_count": 0,  # yt-dlp fallback
    "current_tier": 1,  # Current active tier (1, 2, or 3)
    "tier_transitions": [],  # Record of tier transitions
}

# Fallback tier enumeration
FALLBACK_TIER_FULL_API = 1  # Full API with all enrichments
FALLBACK_TIER_REDUCED_API = 2  # Reduced API (skip metadata, engagement)
FALLBACK_TIER_YTDLP = 3  # yt-dlp fallback

# Global registry for YouTube API client (US-146-012)
# Allows metrics_exporter to access the client for API metrics
_youtube_api_client: Optional[Any] = None

# Global flag for quota reset (US-148-007)
_youtube_quota_reset_requested: bool = False


def get_youtube_api_client() -> Optional[Any]:
    """Get the global YouTube API client instance."""
    return _youtube_api_client


def set_youtube_api_client(client: Any) -> None:
    """Set the global YouTube API client instance.

    Args:
        client: YouTubeAPIClient instance
    """
    global _youtube_api_client
    _youtube_api_client = client


def set_youtube_quota_reset_requested(reset: bool = True) -> None:
    """Set the flag to reset YouTube API quota on next client initialization.

    Args:
        reset: Whether to reset quota (default: True)
    """
    global _youtube_quota_reset_requested
    _youtube_quota_reset_requested = reset


def get_youtube_quota_reset_requested() -> bool:
    """Check if quota reset was requested via CLI flag.

    Returns:
        True if --reset-youtube-quota was passed
    """
    return _youtube_quota_reset_requested


def get_fallback_metrics() -> Dict[str, Any]:
    """Get fallback metrics summary."""
    return {
        "total_fallbacks": _fallback_metrics["total_fallbacks"],
        "recent_fallbacks": _fallback_metrics["fallback_events"][-10:],  # Last 10
    }


def get_fallback_level_metrics() -> Dict[str, Any]:
    """Get fallback level metrics for US-153-012.

    Returns:
        Dict with tier counts, current tier, and recent transitions
    """
    return {
        "tier1_count": _fallback_level_metrics["tier1_count"],
        "tier2_count": _fallback_level_metrics["tier2_count"],
        "tier3_count": _fallback_level_metrics["tier3_count"],
        "current_tier": _fallback_level_metrics["current_tier"],
        "recent_transitions": _fallback_level_metrics["tier_transitions"][-10:],
    }


def record_fallback_level(tier: int) -> None:
    """Record usage of a specific fallback tier (US-153-012).

    Args:
        tier: The fallback tier used (1, 2, or 3)
    """
    global _fallback_level_metrics
    if tier == FALLBACK_TIER_FULL_API:
        _fallback_level_metrics["tier1_count"] += 1
    elif tier == FALLBACK_TIER_REDUCED_API:
        _fallback_level_metrics["tier2_count"] += 1
    elif tier == FALLBACK_TIER_YTDLP:
        _fallback_level_metrics["tier3_count"] += 1


def get_current_tier(
    api_client: Optional[Any],
    config: Optional[Any] = None,
) -> int:
    """Determine the current fallback tier based on quota levels (US-153-012).

    Args:
        api_client: YouTube API client to check quota status
        config: Config object with quota_fallback_levels settings

    Returns:
        Current fallback tier (1, 2, or 3)
    """
    global _fallback_level_metrics

    # Get fallback config
    fallback_config = None
    if config:
        video_search = getattr(config, "video_search", None)
        if video_search:
            fallback_config = getattr(video_search, "quota_fallback_levels", None)

    # If fallback is disabled, always use tier 1
    if fallback_config and not getattr(fallback_config, "enabled", True):
        _fallback_level_metrics["current_tier"] = FALLBACK_TIER_FULL_API
        return FALLBACK_TIER_FULL_API

    # Check quota status
    if api_client is None:
        # No API client, use yt-dlp fallback
        _fallback_level_metrics["current_tier"] = FALLBACK_TIER_YTDLP
        return FALLBACK_TIER_YTDLP

    try:
        quota_status = api_client.get_quota_status()
        quota_percent_remaining = quota_status.get("quota_percent_remaining", 100.0)
    except Exception:
        # Can't determine quota, default to tier 1
        _fallback_level_metrics["current_tier"] = FALLBACK_TIER_FULL_API
        return FALLBACK_TIER_FULL_API

    # Get thresholds from config or use defaults
    if fallback_config:
        tier1_threshold = getattr(fallback_config, "tier1_to_tier2_threshold", 30.0)
        tier2_threshold = getattr(fallback_config, "tier2_to_tier3_threshold", 10.0)
    else:
        tier1_threshold = 30.0
        tier2_threshold = 10.0

    # Determine tier based on quota remaining
    if quota_percent_remaining < tier2_threshold:
        new_tier = FALLBACK_TIER_YTDLP
    elif quota_percent_remaining < tier1_threshold:
        new_tier = FALLBACK_TIER_REDUCED_API
    else:
        new_tier = FALLBACK_TIER_FULL_API

    # Record tier transition if changed
    if new_tier != _fallback_level_metrics["current_tier"]:
        _record_tier_transition(
            from_tier=_fallback_level_metrics["current_tier"],
            to_tier=new_tier,
            quota_percent=quota_percent_remaining,
        )

    _fallback_level_metrics["current_tier"] = new_tier
    return new_tier


def _record_tier_transition(
    from_tier: int,
    to_tier: int,
    quota_percent: float,
) -> None:
    """Record a tier transition event (US-153-012).

    Args:
        from_tier: Previous tier
        to_tier: New tier
        quota_percent: Quota percentage remaining at time of transition
    """
    global _fallback_level_metrics

    timestamp = datetime.now().isoformat()
    transition = {
        "timestamp": timestamp,
        "from_tier": from_tier,
        "to_tier": to_tier,
        "quota_percent_remaining": quota_percent,
    }
    _fallback_level_metrics["tier_transitions"].append(transition)

    logger.warning(
        f"FALLBACK LEVEL TRANSITION: Tier {from_tier} -> Tier {to_tier} "
        f"(quota: {quota_percent:.1f}% remaining)"
    )


def reset_fallback_level() -> None:
    """Reset fallback level to tier 1 (full API).

    Called when quota is reset or new API key is used.
    """
    global _fallback_level_metrics
    _fallback_level_metrics["current_tier"] = FALLBACK_TIER_FULL_API


def record_ytdlp_usage() -> None:
    """Record yt-dlp usage (US-146-012).

    Call this when yt-dlp is used for search/download operations
    to track API vs yt-dlp usage ratio.
    """
    global _ytdlp_usage_count
    _ytdlp_usage_count += 1


def get_ytdlp_usage_count() -> int:
    """Get the total yt-dlp usage count (US-146-012).

    Returns:
        Number of times yt-dlp was used instead of YouTube API
    """
    return _ytdlp_usage_count


def get_api_vs_ytdlp_summary() -> Dict[str, Any]:
    """Get API vs yt-dlp usage summary for dashboard logging (US-146-012).

    Returns:
        Dictionary with API calls, yt-dlp usage, and ratios
    """
    # Get YouTube API client metrics if available
    api_client = get_youtube_api_client()
    api_calls = 0
    fallback_count = 0

    if api_client is not None:
        try:
            api_calls = api_client.metrics.get_total_calls()
            fallback_count = api_client.metrics.fallback_to_ytdlp
        except Exception:
            pass

    ytdlp_count = get_ytdlp_usage_count()

    total_operations = api_calls + ytdlp_count
    api_percentage = (api_calls / total_operations * 100) if total_operations > 0 else 0
    ytdlp_percentage = (ytdlp_count / total_operations * 100) if total_operations > 0 else 0

    return {
        "api_calls": api_calls,
        "ytdlp_count": ytdlp_count,
        "fallback_count": fallback_count,
        "total_operations": total_operations,
        "api_percentage": round(api_percentage, 1),
        "ytdlp_percentage": round(ytdlp_percentage, 1),
    }


def log_api_vs_ytdlp_usage() -> str:
    """Log API vs yt-dlp usage ratio to dashboard (US-146-012).

    Returns:
        Formatted log message string
    """
    summary = get_api_vs_ytdlp_summary()

    if summary["total_operations"] == 0:
        msg = "YouTube: No operations recorded"
    else:
        msg = (
            f"YouTube: API {summary['api_calls']} calls ({summary['api_percentage']:.1f}%) | "
            f"yt-dlp {summary['ytdlp_count']} uses ({summary['ytdlp_percentage']:.1f}%) | "
            f"Fallbacks: {summary['fallback_count']}"
        )

    logger.info(msg)
    return msg


def log_fallback_event(
    reason: str,
    query: str = "",
    quota_used: int = 0,
    quota_limit: int = 0,
) -> None:
    """Log a fallback event with timestamp and reason.

    Args:
        reason: Reason for fallback (e.g., "quota_exhausted", "api_error")
        query: The search query that triggered fallback
        quota_used: Current quota usage at time of fallback
        quota_limit: Quota limit at time of fallback
    """
    # US-152-012: Include timestamp in log message for debugging
    timestamp = datetime.now().isoformat()
    event = {
        "timestamp": timestamp,
        "reason": reason,
        "query": query,
        "quota_used": quota_used,
        "quota_limit": quota_limit,
        "quota_percent": (
            (quota_used / quota_limit * 100) if quota_limit > 0 else 0
        ),
    }

    _fallback_metrics["fallback_events"].append(event)
    _fallback_metrics["total_fallbacks"] += 1

    # Log the fallback event with timestamp
    logger.warning(
        f"FALLBACK: timestamp={timestamp}, reason={reason}, query='{query}', "
        f"quota: {quota_used}/{quota_limit} ({event['quota_percent']:.1f}%)"
    )


def fallback_to_ytdlp(
    query: str,
    max_results: int = 20,
    min_duration: int = 30,
    max_duration: int = 600,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Fallback to yt-dlp search when YouTube API quota is exhausted.

    This function provides the same interface as YouTube API search
    but uses yt-dlp instead, making the fallback transparent to callers.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        min_duration: Minimum video duration in seconds
        max_duration: Maximum video duration in seconds
        config: Optional config object for additional options

    Returns:
        List of dicts with video metadata (same structure as YouTube API results)
    """
    import yt_dlp
    from ..downloader.impersonation import ImpersonationManager

    logger.info(f"Falling back to yt-dlp for query: '{query}'")

    # Build yt-dlp options
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }

    # Apply impersonation if available
    try:
        imp_mgr = ImpersonationManager()
        imp_opts = imp_mgr.get_ydl_options(tier=1)
        ydl_opts.update(imp_opts)
    except Exception as e:
        logger.debug(f"Could not apply impersonation: {e}")

    results = []
    search_url = f"ytsearch{max_results * 2}:{query}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_url, download=False)
            if not info or "entries" not in info:
                return []

            seen_ids = set()
            for entry in info.get("entries", []):
                if not entry:
                    continue

                video_id = entry.get("id", "")
                if not video_id or video_id in seen_ids:
                    continue

                # Filter by duration
                duration = entry.get("duration", 0) or 0
                if duration < min_duration or duration > max_duration:
                    continue

                seen_ids.add(video_id)

                results.append({
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": entry.get("title", ""),
                    "channel": entry.get("channel", entry.get("uploader", "")),
                    "duration": duration,
                    "description": entry.get("description", ""),
                    "keyword": query,
                    "view_count": entry.get("view_count"),
                    "subscriber_count": entry.get("channel_follower_count"),
                    "source": "yt-dlp-fallback",  # Mark as fallback result
                })

                if len(results) >= max_results:
                    break

        except Exception as e:
            logger.warning(f"yt-dlp fallback search error: {e}")

    logger.info(f"yt-dlp fallback returned {len(results)} results for '{query}'")
    # US-146-012: Record yt-dlp usage for API vs yt-dlp ratio tracking
    record_ytdlp_usage()
    return results


class YouTubeAPIFallbackHandler:
    """Handler for falling back from YouTube API to yt-dlp.

    Provides a unified interface that tries YouTube API first,
    then falls back to yt-dlp on quota exhaustion or errors.
    """

    def __init__(self, api_client: Optional[Any] = None, config: Optional[Any] = None):
        """Initialize the fallback handler.

        Args:
            api_client: YouTubeAPIClient instance (if available)
            config: Config object for yt-dlp options
        """
        self.api_client = api_client
        self.config = config
        self._fallback_occurred = False
        self._fallback_reason = ""

    @property
    def fallback_occurred(self) -> bool:
        """Check if fallback has occurred."""
        return self._fallback_occurred

    @property
    def fallback_reason(self) -> str:
        """Get the reason for fallback."""
        return self._fallback_reason

    def _get_deduplication_config(self) -> Dict[str, Any]:
        """Get deduplication config from video_search config.

        Returns:
            Dict with enable_deduplication, max_title_similarity,
            enable_freshness_scoring, min_freshness_days
        """
        # Default values
        defaults = {
            "enable_deduplication": True,
            "max_title_similarity": 0.85,
            "enable_freshness_scoring": True,
            "min_freshness_days": 365,
        }

        if not self.config:
            return defaults

        # Try to get from video_search config
        try:
            video_search = getattr(self.config, "video_search", None)
            if video_search:
                return {
                    "enable_deduplication": getattr(
                        video_search, "enable_deduplication", defaults["enable_deduplication"]
                    ),
                    "max_title_similarity": getattr(
                        video_search, "max_title_similarity", defaults["max_title_similarity"]
                    ),
                    "enable_freshness_scoring": getattr(
                        video_search, "enable_freshness_scoring", defaults["enable_freshness_scoring"]
                    ),
                    "min_freshness_days": getattr(
                        video_search, "min_freshness_days", defaults["min_freshness_days"]
                    ),
                }
        except Exception as e:
            logger.debug(f"Could not get video_search config: {e}")

        return defaults

    def _apply_deduplication(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply deduplication and freshness scoring to search results.

        Args:
            results: List of video result dicts

        Returns:
            Processed results with deduplication and freshness scores
        """
        if not results:
            return results

        dedup_config = self._get_deduplication_config()

        return process_search_results(
            results,
            enable_deduplication=dedup_config["enable_deduplication"],
            max_title_similarity=dedup_config["max_title_similarity"],
            enable_freshness_scoring=dedup_config["enable_freshness_scoring"],
            min_freshness_days=dedup_config["min_freshness_days"],
        )

    def search_with_fallback(
        self,
        query: str,
        max_results: int = 20,
        min_duration: int = 30,
        max_duration: int = 600,
        published_after: str = "",
        published_before: str = "",
        video_category_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Search with automatic fallback from YouTube API to yt-dlp.

        Tries YouTube API first (if client is available and enabled),
        falls back to yt-dlp on quota exhaustion or errors. Supports
        3-tier progressive degradation (US-153-012).

        Args:
            query: Search query string
            max_results: Maximum number of results
            min_duration: Minimum video duration
            max_duration: Maximum video duration
            published_after: Filter videos published after this date
            published_before: Filter videos published before this date
            video_category_id: Filter by YouTube video category ID

        Returns:
            List of video result dicts
        """
        # US-153-012: Determine current fallback tier based on quota
        current_tier = get_current_tier(self.api_client, self.config)

        # Get fallback config for tier-specific settings
        fallback_config = None
        if self.config:
            video_search = getattr(self.config, "video_search", None)
            if video_search:
                fallback_config = getattr(video_search, "quota_fallback_levels", None)

        # Handle each tier
        if current_tier == FALLBACK_TIER_FULL_API:
            # Tier 1: Full API with all enrichments
            results = self._search_tier_full(query, max_results, min_duration, max_duration, published_after, published_before, video_category_id)
            record_fallback_level(FALLBACK_TIER_FULL_API)
            return results

        elif current_tier == FALLBACK_TIER_REDUCED_API:
            # Tier 2: Reduced API - fewer results, skip metadata/engagement
            tier_max_results = max_results
            if fallback_config:
                reduction = getattr(fallback_config, "tier2_max_results_reduction", 0.5)
                tier_max_results = max(1, int(max_results * reduction))

            logger.info(
                f"Using Tier 2 (reduced API) for query: '{query}' "
                f"(max_results: {tier_max_results} vs {max_results})"
            )

            results = self._search_tier_reduced(
                query, tier_max_results, min_duration, max_duration, fallback_config, published_after, published_before, video_category_id
            )
            record_fallback_level(FALLBACK_TIER_REDUCED_API)
            return results

        else:
            # Tier 3: yt-dlp fallback (yt-dlp doesn't support date filtering via API params)
            results = self._search_tier_ytdlp(
                query, max_results, min_duration, max_duration
            )
            record_fallback_level(FALLBACK_TIER_YTDLP)
            return results

    def _search_tier_full(
        self,
        query: str,
        max_results: int,
        min_duration: int,
        max_duration: int,
        published_after: str = "",
        published_before: str = "",
        video_category_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Tier 1: Full API search with all enrichments.

        Args:
            query: Search query
            max_results: Max results
            min_duration: Min duration
            max_duration: Max duration
            published_after: Filter videos published after this date
            published_before: Filter videos published before this date
            video_category_id: Filter by YouTube video category ID

        Returns:
            Enriched search results
        """
        # Try YouTube API first if client is available
        if self.api_client is not None:
            try:
                from .youtube_api_client import (
                    QUOTA_COST_SEARCH,
                    QuotaExceededError,
                    APIError,
                )

                # Check if quota is available
                remaining = self.api_client.get_remaining_quota()
                if remaining < QUOTA_COST_SEARCH:
                    # Quota too low, skip to fallback
                    raise QuotaExceededError(
                        f"Insufficient quota: {remaining} < {QUOTA_COST_SEARCH}"
                    )

                # US-150-007: Check for proactive fallback based on quota prediction
                if hasattr(self.api_client, 'should_proactive_fallback'):
                    if self.api_client.should_proactive_fallback(QUOTA_COST_SEARCH):
                        # Proactive fallback triggered - log and switch to yt-dlp
                        quota_status = self.api_client.get_quota_status()
                        log_fallback_event(
                            reason="proactive_quota_prediction",
                            query=query,
                            quota_used=quota_status.get("quota_used", 0),
                            quota_limit=quota_status.get("quota_limit", 10000),
                        )
                        if self.api_client and hasattr(self.api_client, "record_fallback"):
                            self.api_client.record_fallback("search", "proactive_quota_prediction", query)
                        self._fallback_occurred = True
                        self._fallback_reason = "proactive_quota_prediction"
                        # Skip to yt-dlp fallback below
                        raise QuotaExceededError(
                            f"Proactive fallback: quota predicted to exhaust soon "
                            f"({quota_status.get('quota_percent_remaining', 0):.1f}% remaining)"
                        )

                # Try API search
                api_results = self.api_client.search_videos(
                    query, max_results,
                    published_after=published_after,
                    published_before=published_before,
                    video_category_id=video_category_id,
                )

                # Convert API results to dict format (same as yt-dlp)
                # Include all available fields from API response
                results = [
                    {
                        "video_id": r.video_id,
                        "url": f"https://www.youtube.com/watch?v={r.video_id}",
                        "title": r.title,
                        "channel": r.channel_title,
                        "channel_id": r.channel_id,  # New: channel ID from API
                        "published_at": r.published_at,  # New: publish date from API
                        "duration": 0,  # API search doesn't return duration
                        "description": r.description,
                        "keyword": query,
                        "view_count": r.view_count if hasattr(r, 'view_count') else None,
                        "subscriber_count": None,
                        "thumbnail_url": r.thumbnail_url if hasattr(r, 'thumbnail_url') else "",
                        "source": "youtube-api",
                        # US-155-004: Date range info from search
                        "search_date_range": r.search_date_range if hasattr(r, 'search_date_range') else None,
                    }
                    for r in api_results
                ]

                # Enrich with channel metadata
                results = self._enrich_with_channel_metadata(results)

                # US-148-008: Enrich with engagement metrics (view_count, like_count, comment_count)
                # US-149-005: Apply deduplication and freshness scoring
                results = self._enrich_with_engagement_metrics(results)
                return self._apply_deduplication(results)

            except QuotaExceededError as e:
                # Log fallback event
                log_fallback_event(
                    reason="quota_exhausted",
                    query=query,
                    quota_used=getattr(self.api_client, "quota_used", 0),
                    quota_limit=getattr(self.api_client, "quota_limit", 10000),
                )
                # US-146-012: Record fallback on API client metrics
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", "quota_exhausted", query)
                self._fallback_occurred = True
                self._fallback_reason = "quota_exhausted"

            except APIError as e:
                # Log fallback for other API errors
                log_fallback_event(
                    reason=f"api_error: {str(e)[:50]}",
                    query=query,
                )
                # US-146-012: Record fallback on API client metrics
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", "api_error", query)
                self._fallback_occurred = True
                self._fallback_reason = f"api_error: {str(e)[:50]}"

            # US-148-012: Handle YouTubeAPIError types (quota exceeded, rate limited, etc.)
            except (
                YouTubeAPIError,
                YouTubeAPIQuotaExceededError,
                YouTubeAPIRateLimitedError,
                YouTubeAPIInvalidKeyError,
                YouTubeAPIPermissionDeniedError,
                YouTubeAPINetworkError,
            ) as e:
                # Determine fallback reason based on error type
                error_type = getattr(e, 'error_type', 'youtube_api')
                log_fallback_event(
                    reason=f"youtube_api_error: {error_type}",
                    query=query,
                )
                # Record fallback on API client metrics
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", error_type, query)
                self._fallback_occurred = True
                self._fallback_reason = f"youtube_api_error: {error_type}"

            except Exception as e:
                # Log fallback for unexpected errors
                log_fallback_event(
                    reason=f"unexpected_error: {str(e)[:50]}",
                    query=query,
                )
                # US-146-012: Record fallback on API client metrics
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", "unexpected_error", query)
                self._fallback_occurred = True
                self._fallback_reason = f"unexpected_error: {str(e)[:50]}"

        # Fallback to yt-dlp for any error in tier 1
        return self._search_tier_ytdlp(query, max_results, min_duration, max_duration)

    def _search_tier_reduced(
        self,
        query: str,
        max_results: int,
        min_duration: int,
        max_duration: int,
        fallback_config: Optional[Any] = None,
        published_after: str = "",
        published_before: str = "",
        video_category_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Tier 2: Reduced API search - skip metadata/engagement enrichment.

        Args:
            query: Search query
            max_results: Max results (reduced)
            min_duration: Min duration
            max_duration: Max duration
            fallback_config: Fallback configuration
            published_after: Filter videos published after this date
            published_before: Filter videos published before this date
            video_category_id: Filter by YouTube video category ID

        Returns:
            Search results without enrichments
        """
        # Check if we should skip channel metadata
        skip_channel = True
        skip_engagement = True
        if fallback_config:
            skip_channel = getattr(fallback_config, "tier2_skip_channel_metadata", True)
            skip_engagement = getattr(fallback_config, "tier2_skip_engagement_metrics", True)

        if self.api_client is not None:
            try:
                from .youtube_api_client import (
                    QUOTA_COST_SEARCH,
                    QuotaExceededError,
                    APIError,
                )

                # Check if quota is available
                remaining = self.api_client.get_remaining_quota()
                if remaining < QUOTA_COST_SEARCH:
                    raise QuotaExceededError(
                        f"Insufficient quota: {remaining} < {QUOTA_COST_SEARCH}"
                    )

                # Try API search with date range filtering
                api_results = self.api_client.search_videos(
                    query, max_results,
                    published_after=published_after,
                    published_before=published_before,
                    video_category_id=video_category_id,
                )

                # Convert API results to dict format
                results = [
                    {
                        "video_id": r.video_id,
                        "url": f"https://www.youtube.com/watch?v={r.video_id}",
                        "title": r.title,
                        "channel": r.channel_title,
                        "channel_id": r.channel_id,
                        "published_at": r.published_at,
                        "duration": 0,
                        "description": r.description,
                        "keyword": query,
                        "view_count": r.view_count if hasattr(r, 'view_count') else None,
                        "subscriber_count": None,
                        "thumbnail_url": r.thumbnail_url if hasattr(r, 'thumbnail_url') else "",
                        "source": "youtube-api-reduced",  # Mark as reduced tier
                    }
                    for r in api_results
                ]

                # Skip channel metadata enrichment in tier 2
                if not skip_channel:
                    results = self._enrich_with_channel_metadata(results)

                # Skip engagement metrics enrichment in tier 2
                if not skip_engagement:
                    results = self._enrich_with_engagement_metrics(results)

                return self._apply_deduplication(results)

            except (QuotaExceededError, YouTubeAPIQuotaExceededError) as e:
                log_fallback_event(
                    reason="quota_exhausted_tier2",
                    query=query,
                    quota_used=getattr(self.api_client, "quota_used", 0),
                    quota_limit=getattr(self.api_client, "quota_limit", 10000),
                )
                self._fallback_occurred = True
                self._fallback_reason = "quota_exhausted_tier2"

            except Exception as e:
                log_fallback_event(
                    reason=f"api_error_tier2: {str(e)[:50]}",
                    query=query,
                )
                self._fallback_occurred = True
                self._fallback_reason = f"api_error_tier2: {str(e)[:50]}"

        # Fallback to yt-dlp if tier 2 API fails
        return self._search_tier_ytdlp(query, max_results, min_duration, max_duration)

    def _search_tier_ytdlp(
        self,
        query: str,
        max_results: int,
        min_duration: int,
        max_duration: int,
    ) -> List[Dict[str, Any]]:
        """Tier 3: yt-dlp fallback search.

        Args:
            query: Search query
            max_results: Max results
            min_duration: Min duration
            max_duration: Max duration

        Returns:
            yt-dlp search results
        """
        # Get tier 3 max results from config
        tier3_max = max_results
        if self.config:
            video_search = getattr(self.config, "video_search", None)
            if video_search:
                fallback_config = getattr(video_search, "quota_fallback_levels", None)
                if fallback_config:
                    tier3_max = getattr(fallback_config, "tier3_max_results", max_results)

        logger.info(f"Using Tier 3 (yt-dlp) for query: '{query}'")

        # Fallback to yt-dlp
        results = fallback_to_ytdlp(
            query=query,
            max_results=tier3_max,
            min_duration=min_duration,
            max_duration=max_duration,
            config=self.config,
        )

        # For yt-dlp results, we could optionally enrich with channel metadata here
        # but that would require additional API calls, so skip for fallback

        # US-149-005: Apply deduplication and freshness scoring
        return self._apply_deduplication(results)

    async def async_search_with_fallback(
        self,
        query: str,
        max_results: int = 20,
        min_duration: int = 30,
        max_duration: int = 600,
    ) -> List[Dict[str, Any]]:
        """Async version of search_with_fallback with tiered fallback (US-153-012).

        Tries YouTube API async methods first (if client is available),
        falls back progressively through 3 tiers based on quota levels.

        Args:
            query: Search query string
            max_results: Maximum number of results
            min_duration: Minimum video duration
            max_duration: Maximum video duration

        Returns:
            List of video result dicts
        """
        # US-153-012: Determine current fallback tier based on quota
        current_tier = get_current_tier(self.api_client, self.config)

        # Get fallback config for tier-specific settings
        fallback_config = None
        if self.config:
            video_search = getattr(self.config, "video_search", None)
            if video_search:
                fallback_config = getattr(video_search, "quota_fallback_levels", None)

        # Handle each tier
        if current_tier == FALLBACK_TIER_FULL_API:
            # Tier 1: Full API with all enrichments
            results = await self._async_search_tier_full(query, max_results, min_duration, max_duration)
            record_fallback_level(FALLBACK_TIER_FULL_API)
            return results

        elif current_tier == FALLBACK_TIER_REDUCED_API:
            # Tier 2: Reduced API - fewer results, skip metadata/engagement
            tier_max_results = max_results
            if fallback_config:
                reduction = getattr(fallback_config, "tier2_max_results_reduction", 0.5)
                tier_max_results = max(1, int(max_results * reduction))

            logger.info(
                f"Using Tier 2 (reduced API) for query: '{query}' "
                f"(max_results: {tier_max_results} vs {max_results})"
            )

            results = await self._async_search_tier_reduced(
                query, tier_max_results, min_duration, max_duration, fallback_config
            )
            record_fallback_level(FALLBACK_TIER_REDUCED_API)
            return results

        else:
            # Tier 3: yt-dlp fallback
            results = self._search_tier_ytdlp(
                query, max_results, min_duration, max_duration
            )
            record_fallback_level(FALLBACK_TIER_YTDLP)
            return results

    async def _async_search_tier_full(
        self,
        query: str,
        max_results: int,
        min_duration: int,
        max_duration: int,
    ) -> List[Dict[str, Any]]:
        """Async Tier 1: Full API search with all enrichments.

        Args:
            query: Search query
            max_results: Max results
            min_duration: Min duration
            max_duration: Max duration

        Returns:
            Enriched search results
        """
        # Try YouTube API async methods first if client is available
        if self.api_client is not None and hasattr(self.api_client, 'async_search_videos'):
            try:
                from .youtube_api_client import (
                    QUOTA_COST_SEARCH,
                    QuotaExceededError,
                    APIError,
                )

                # Check if quota is available
                remaining = self.api_client.get_remaining_quota()
                if remaining < QUOTA_COST_SEARCH:
                    raise QuotaExceededError(
                        f"Insufficient quota: {remaining} < {QUOTA_COST_SEARCH}"
                    )

                # US-150-007: Check for proactive fallback based on quota prediction
                if hasattr(self.api_client, 'should_proactive_fallback'):
                    if self.api_client.should_proactive_fallback(QUOTA_COST_SEARCH):
                        # Proactive fallback triggered - log and switch to yt-dlp
                        quota_status = self.api_client.get_quota_status()
                        log_fallback_event(
                            reason="proactive_quota_prediction",
                            query=query,
                            quota_used=quota_status.get("quota_used", 0),
                            quota_limit=quota_status.get("quota_limit", 10000),
                        )
                        if self.api_client and hasattr(self.api_client, "record_fallback"):
                            self.api_client.record_fallback("search", "proactive_quota_prediction", query)
                        self._fallback_occurred = True
                        self._fallback_reason = "proactive_quota_prediction"
                        # Skip to yt-dlp fallback below
                        raise QuotaExceededError(
                            f"Proactive fallback: quota predicted to exhaust soon "
                            f"({quota_status.get('quota_percent_remaining', 0):.1f}% remaining)"
                        )

                # Use async search method
                api_results = await self.api_client.async_search_videos(query, max_results)

                # Convert API results to dict format (same as yt-dlp)
                results = [
                    {
                        "video_id": r.video_id,
                        "url": f"https://www.youtube.com/watch?v={r.video_id}",
                        "title": r.title,
                        "channel": r.channel_title,
                        "channel_id": r.channel_id,
                        "published_at": r.published_at,
                        "duration": 0,
                        "description": r.description,
                        "keyword": query,
                        "view_count": r.view_count if hasattr(r, 'view_count') else None,
                        "subscriber_count": None,
                        "thumbnail_url": r.thumbnail_url if hasattr(r, 'thumbnail_url') else "",
                        "source": "youtube-api",
                    }
                    for r in api_results
                ]

                # Enrich with channel metadata (sync version for now)
                results = self._enrich_with_channel_metadata(results)

                # Enrich with engagement metrics (sync version for now)
                results = self._enrich_with_engagement_metrics(results)

                # Apply deduplication and freshness scoring
                return self._apply_deduplication(results)

            except (QuotaExceededError, YouTubeAPIQuotaExceededError) as e:
                log_fallback_event(
                    reason="quota_exhausted",
                    query=query,
                    quota_used=getattr(self.api_client, "quota_used", 0),
                    quota_limit=getattr(self.api_client, "quota_limit", 10000),
                )
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", "quota_exhausted", query)
                self._fallback_occurred = True
                self._fallback_reason = "quota_exhausted"

            except (APIError, YouTubeAPIError, YouTubeAPIRateLimitedError,
                    YouTubeAPIInvalidKeyError, YouTubeAPIPermissionDeniedError,
                    YouTubeAPINetworkError) as e:
                error_type = getattr(e, 'error_type', 'youtube_api')
                log_fallback_event(
                    reason=f"youtube_api_error: {error_type}",
                    query=query,
                )
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", error_type, query)
                self._fallback_occurred = True
                self._fallback_reason = f"youtube_api_error: {error_type}"

            except Exception as e:
                log_fallback_event(
                    reason=f"unexpected_error: {str(e)[:50]}",
                    query=query,
                )
                if self.api_client and hasattr(self.api_client, "record_fallback"):
                    self.api_client.record_fallback("search", "unexpected_error", query)
                self._fallback_occurred = True
                self._fallback_reason = f"unexpected_error: {str(e)[:50]}"

        # Fallback to yt-dlp for any error in tier 1
        return self._search_tier_ytdlp(query, max_results, min_duration, max_duration)

    async def _async_search_tier_reduced(
        self,
        query: str,
        max_results: int,
        min_duration: int,
        max_duration: int,
        fallback_config: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Async Tier 2: Reduced API search - skip metadata/engagement enrichment.

        Args:
            query: Search query
            max_results: Max results (reduced)
            min_duration: Min duration
            max_duration: Max duration
            fallback_config: Fallback configuration

        Returns:
            Search results without enrichments
        """
        # Check if we should skip channel metadata and engagement
        skip_channel = True
        skip_engagement = True
        if fallback_config:
            skip_channel = getattr(fallback_config, "tier2_skip_channel_metadata", True)
            skip_engagement = getattr(fallback_config, "tier2_skip_engagement_metrics", True)

        if self.api_client is not None and hasattr(self.api_client, 'async_search_videos'):
            try:
                from .youtube_api_client import (
                    QUOTA_COST_SEARCH,
                    QuotaExceededError,
                )

                # Check if quota is available
                remaining = self.api_client.get_remaining_quota()
                if remaining < QUOTA_COST_SEARCH:
                    raise QuotaExceededError(
                        f"Insufficient quota: {remaining} < {QUOTA_COST_SEARCH}"
                    )

                # Use async search method
                api_results = await self.api_client.async_search_videos(query, max_results)

                # Convert API results to dict format
                results = [
                    {
                        "video_id": r.video_id,
                        "url": f"https://www.youtube.com/watch?v={r.video_id}",
                        "title": r.title,
                        "channel": r.channel_title,
                        "channel_id": r.channel_id,
                        "published_at": r.published_at,
                        "duration": 0,
                        "description": r.description,
                        "keyword": query,
                        "view_count": r.view_count if hasattr(r, 'view_count') else None,
                        "subscriber_count": None,
                        "thumbnail_url": r.thumbnail_url if hasattr(r, 'thumbnail_url') else "",
                        "source": "youtube-api-reduced",  # Mark as reduced tier
                    }
                    for r in api_results
                ]

                # Skip channel metadata enrichment in tier 2
                if not skip_channel:
                    results = self._enrich_with_channel_metadata(results)

                # Skip engagement metrics enrichment in tier 2
                if not skip_engagement:
                    results = self._enrich_with_engagement_metrics(results)

                return self._apply_deduplication(results)

            except (QuotaExceededError, YouTubeAPIQuotaExceededError) as e:
                log_fallback_event(
                    reason="quota_exhausted_tier2",
                    query=query,
                    quota_used=getattr(self.api_client, "quota_used", 0),
                    quota_limit=getattr(self.api_client, "quota_limit", 10000),
                )
                self._fallback_occurred = True
                self._fallback_reason = "quota_exhausted_tier2"

            except Exception as e:
                log_fallback_event(
                    reason=f"api_error_tier2: {str(e)[:50]}",
                    query=query,
                )
                self._fallback_occurred = True
                self._fallback_reason = f"api_error_tier2: {str(e)[:50]}"

        # Fallback to yt-dlp if tier 2 API fails
        return self._search_tier_ytdlp(query, max_results, min_duration, max_duration)

    def _enrich_with_channel_metadata(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich search results with channel metadata from channels.list API.

        Fetches subscriber_count, video_count, view_count for each unique channel
        and adds to results. Also filters by min_subscriber_count threshold
        and adds channel_quality_score for ranking.

        Args:
            results: List of video result dicts

        Returns:
            Enriched results with channel metadata
        """
        if not results or not self.api_client:
            return results

        # Get unique channel IDs
        channel_ids = list(set(
            r.get("channel_id", "") for r in results if r.get("channel_id")
        ))

        if not channel_ids:
            return results

        # Get config for min_subscriber_count
        min_subscribers = 1000  # Default
        if self.config:
            youtube_api_config = getattr(self.config, 'youtube_api', None)
            if youtube_api_config:
                min_subscribers = getattr(
                    youtube_api_config, 'min_subscriber_count', 1000
                )

        try:
            # Fetch channel metadata
            channel_metadata = self.api_client.get_channel_metadata(channel_ids)

            # Build channel_id -> metadata lookup
            channel_lookup = {
                cid: meta for cid, meta in channel_metadata.items()
            }

            # Enrich results and filter
            enriched = []
            for r in results:
                channel_id = r.get("channel_id", "")
                if not channel_id or channel_id not in channel_lookup:
                    # Keep results without channel info (they'll be filtered if needed)
                    enriched.append(r)
                    continue

                meta = channel_lookup[channel_id]
                subscriber_count = meta.get("subscriber_count", 0)

                # Filter by min_subscriber_count
                if subscriber_count < min_subscribers:
                    logger.debug(
                        f"Filtering out {r.get('title', '')[:30]}... "
                        f"(channel {subscriber_count} < {min_subscribers} subscribers)"
                    )
                    continue

                # Add channel metadata to result
                r["subscriber_count"] = subscriber_count
                r["channel_video_count"] = meta.get("video_count", 0)
                r["channel_total_views"] = meta.get("view_count", 0)
                r["channel_created_date"] = meta.get("published_at", "")

                # Calculate channel quality score (0.0 - 1.0)
                # Based on subscriber count and activity
                r["channel_quality_score"] = self._calculate_channel_quality_score(
                    subscriber_count=subscriber_count,
                    video_count=meta.get("video_count", 0),
                    total_views=meta.get("view_count", 0),
                )

                enriched.append(r)

            logger.info(
                f"Channel metadata enrichment: {len(enriched)}/{len(results)} "
                f"results after filtering (min {min_subscribers} subscribers)"
            )
            return enriched

        except Exception as e:
            logger.warning(f"Channel metadata enrichment failed: {e}")
            return results

    def _calculate_channel_quality_score(
        self,
        subscriber_count: int,
        video_count: int,
        total_views: int,
    ) -> float:
        """Calculate channel quality score for ranking.

        Score is based on:
        - Subscriber count (primary factor)
        - Video count (activity indicator)
        - Total views (popularity indicator)

        Args:
            subscriber_count: Number of subscribers
            video_count: Total videos on channel
            total_views: Total channel views

        Returns:
            Quality score between 0.0 and 1.0
        """
        import math

        # Subscriber score (log scale, 0-0.6 weight)
        # 1M+ = 0.6, 100K = 0.4, 10K = 0.2, 1K = 0.1
        if subscriber_count > 0:
            subscriber_score = min(0.6, math.log10(subscriber_count + 1) / 6.0)
        else:
            subscriber_score = 0.0

        # Activity score (0-0.2 weight) - channels with more videos are more active
        if video_count > 0:
            activity_score = min(0.2, math.log10(video_count + 1) / 7.0)
        else:
            activity_score = 0.0

        # Popularity score (0-0.2 weight) - total views
        if total_views > 0:
            popularity_score = min(0.2, math.log10(total_views + 1) / 8.0)
        else:
            popularity_score = 0.0

        return round(subscriber_score + activity_score + popularity_score, 3)

    def _enrich_with_engagement_metrics(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich search results with engagement metrics from videos.list API.

        Fetches view_count, like_count, comment_count for each video and adds
        to results. Also calculates engagement_score for ranking.

        Args:
            results: List of video result dicts

        Returns:
            Enriched results with engagement metrics
        """
        if not results or not self.api_client:
            return results

        # Get video IDs
        video_ids = [r.get("video_id", "") for r in results if r.get("video_id")]

        if not video_ids:
            return results

        # Get config for engagement ranking weights
        view_count_weight = 0.5
        like_count_weight = 0.3
        comment_count_weight = 0.2

        if self.config:
            youtube_api_config = getattr(self.config, 'youtube_api', None)
            if youtube_api_config:
                view_count_weight = getattr(youtube_api_config, 'view_count_weight', 0.5)
                like_count_weight = getattr(youtube_api_config, 'like_count_weight', 0.3)
                comment_count_weight = getattr(youtube_api_config, 'comment_count_weight', 0.2)

            # Fetch video engagement metrics (batch up to 50 per API call)
            try:
                # Import here to avoid circular import
                from .youtube_api_client import QUOTA_COST_VIDEOS

                # Check quota before making request
                remaining = self.api_client.get_remaining_quota()
                if remaining < QUOTA_COST_VIDEOS:
                    logger.debug("Insufficient quota for engagement metrics enrichment")
                    return results

                # Fetch video details with statistics
                video_details, failed_video_ids = self.api_client.get_video_details(
                    video_ids=video_ids,
                    part="contentDetails,statistics"
                )

                # US-156-007: Handle partial failures
                if failed_video_ids:
                    logger.warning(
                        f"get_video_details: {len(failed_video_ids)} videos failed to fetch"
                    )

                # video_details is now a Dict[str, VideoDetails]
                details_lookup = video_details

                # Enrich results with engagement metrics
                for r in results:
                    video_id = r.get("video_id", "")
                    if not video_id or video_id not in details_lookup:
                        continue

                    details = details_lookup[video_id]

                    # US-148-008: Get engagement metrics from VideoDetails dataclass
                    view_count = getattr(details, 'view_count', 0) or 0
                    like_count = getattr(details, 'like_count', 0) or 0
                    comment_count = getattr(details, 'comment_count', 0) or 0

                    r["view_count"] = view_count
                    r["like_count"] = like_count
                    r["comment_count"] = comment_count

                    # Calculate engagement score (0.0 - 1.0)
                    r["engagement_score"] = self._calculate_engagement_score(
                        view_count=view_count,
                        like_count=like_count,
                        comment_count=comment_count,
                        view_count_weight=view_count_weight,
                        like_count_weight=like_count_weight,
                        comment_count_weight=comment_count_weight,
                    )

                logger.info(
                    f"Engagement metrics enrichment: {len(video_details)} videos enriched "
                    f"with view_count, like_count, comment_count"
                )
                return results

            except Exception as e:
                logger.warning(f"Engagement metrics enrichment failed: {e}")
                return results

    def _calculate_engagement_score(
        self,
        view_count: int,
        like_count: int,
        comment_count: int,
        view_count_weight: float = 0.5,
        like_count_weight: float = 0.3,
        comment_count_weight: float = 0.2,
    ) -> float:
        """Calculate engagement score for ranking.

        Score is based on:
        - View count (primary factor)
        - Like count (engagement indicator)
        - Comment count (deep engagement indicator)

        Args:
            view_count: Number of views
            like_count: Number of likes
            comment_count: Number of comments
            view_count_weight: Weight for view count (default 0.5)
            like_count_weight: Weight for like count (default 0.3)
            comment_count_weight: Weight for comment count (default 0.2)

        Returns:
            Engagement score between 0.0 and 1.0
        """
        # Normalize each metric using log scale (handles wide range of values)
        # Views: 1M = 1.0, 100K = 0.8, 10K = 0.6, 1K = 0.4
        if view_count > 0:
            view_score = min(1.0, math.log10(view_count + 1) / 7.0)
        else:
            view_score = 0.0

        # Likes: 100K = 1.0, 10K = 0.8, 1K = 0.6, 100 = 0.4
        if like_count > 0:
            like_score = min(1.0, math.log10(like_count + 1) / 5.0)
        else:
            like_score = 0.0

        # Comments: 10K = 1.0, 1K = 0.8, 100 = 0.6, 10 = 0.4
        if comment_count > 0:
            comment_score = min(1.0, math.log10(comment_count + 1) / 4.0)
        else:
            comment_score = 0.0

        # Weighted combination
        return round(
            view_score * view_count_weight +
            like_score * like_count_weight +
            comment_score * comment_count_weight,
            3
        )
