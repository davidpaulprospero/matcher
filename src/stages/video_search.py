"""
Video Search Stage - Search for videos without downloading

Stage 2 of the simplified 7-stage pipeline:
- Searches YouTube for videos based on keywords
- Returns video IDs for caption fetching
- Does NOT download videos (deferred to DOWNLOAD_SEGMENTS after matching)
"""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage, validate_required_state_attrs
from ..logging_templates import (
    log_stage_start,
    log_stage_complete,
    log_stage_skip,
    log_progress,
    log_error_with_context,
    log_rate_limit,
)
from ..downloader.per_keyword_circuit_breaker import (
    PerKeywordCircuitBreaker,
    PerKeywordCircuitBreakerConfig,
)
from ..downloader.api_fallback_handler import (
    YouTubeAPIFallbackHandler,
    log_fallback_event,
    get_fallback_metrics,
    set_youtube_api_client,
)
from ..downloader.youtube_api_client import (
    QUOTA_COST_SEARCH,
    QuotaExceededError,
    YouTubeAPIClient,
)
from ..downloader.search_deduplication import (
    deduplicate_by_video_id,
)

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, VideoSearchResult

logger = logging.getLogger(__name__)


@register_stage
class VideoSearchStage(Stage):
    """
    Searches for videos based on keywords without downloading.

    Inputs:
        - state.keywords: List of keywords to search for
        - state.topic_context: Topic for search refinement

    Outputs:
        - state.video_ids: List of YouTube video IDs
        - state.video_search_results: List of VideoSearchResult with metadata
        - state.search_failed_keywords: List of keywords with no results
    """

    name = "VIDEO_SEARCH"
    description = "Search YouTube for videos (no download)"
    DEPENDS_ON = ['ANALYZE']
    PRODUCES = ['video_ids', 'video_search_results', 'search_failed_keywords']

    def __init__(self):
        self.ydl_opts = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video search stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['keywords'], self.name)

        warnings = []

        try:
            # Get search configuration first (Rule #6: handle both dict and object access)
            # video_search is at config.video_search, not config.download.video_search
            search_config = getattr(config, 'video_search', None)
            if search_config is None:
                search_config = {}
            # Handle both dict and object access patterns
            if isinstance(search_config, dict):
                results_per_keyword = search_config.get('results_per_keyword', 20)
                max_total_results = search_config.get('max_total_results', 200)
                search_budget_aware = search_config.get('search_budget_aware', True)
                auto_distribute_budget = search_config.get('auto_distribute_budget', True)
                enable_channel_diversity = search_config.get('enable_channel_diversity', True)
                max_videos_per_channel = search_config.get('max_videos_per_channel', 3)
                use_negative_context = search_config.get('use_negative_context', False)
                negative_keywords = search_config.get('negative_keywords', []) or []
                use_chapter_queries = search_config.get('use_chapter_queries', True)
                listicle_topic_as_search_terms = search_config.get('listicle_topic_as_search_terms', True)
                # US-113-002: Per-keyword circuit breaker config
                pkc_config_dict = search_config.get('per_keyword_circuit_breaker', {}) or {}
                pkc_enabled = pkc_config_dict.get('enabled', True)
            else:
                results_per_keyword = getattr(search_config, 'results_per_keyword', 20)
                max_total_results = getattr(search_config, 'max_total_results', 200)
                search_budget_aware = getattr(search_config, 'search_budget_aware', True)
                auto_distribute_budget = getattr(search_config, 'auto_distribute_budget', True)
                enable_channel_diversity = getattr(search_config, 'enable_channel_diversity', True)
                max_videos_per_channel = getattr(search_config, 'max_videos_per_channel', 3)
                use_negative_context = getattr(search_config, 'use_negative_context', False)
                negative_keywords = getattr(search_config, 'negative_keywords', []) or []
                use_chapter_queries = getattr(search_config, 'use_chapter_queries', True)
                listicle_topic_as_search_terms = getattr(search_config, 'listicle_topic_as_search_terms', True)
                # US-113-002: Per-keyword circuit breaker config
                pkc_config = getattr(search_config, 'per_keyword_circuit_breaker', None)
                pkc_enabled = getattr(pkc_config, 'enabled', True) if pkc_config else True

            # Log stage start after config is loaded
            log_stage_start(logger, "VIDEO_SEARCH",
                total_keywords=len(state.keywords),
                max_results=max_total_results,
                results_per_keyword=results_per_keyword)

            # US-113-002: Initialize per-keyword circuit breaker
            keyword_cb = None
            if pkc_enabled:
                # Build config from dict or object
                if isinstance(search_config, dict):
                    pkc_cfg = PerKeywordCircuitBreakerConfig(
                        enabled=pkc_config_dict.get('enabled', True),
                        consecutive_failures_threshold=pkc_config_dict.get('consecutive_failures_threshold', 3),
                        pause_seconds=pkc_config_dict.get('pause_seconds', 30.0),
                        max_pause_seconds=pkc_config_dict.get('max_pause_seconds', 120.0),
                        jitter_factor=pkc_config_dict.get('jitter_factor', 0.2),
                    )
                else:
                    pkc_cfg = getattr(search_config, 'per_keyword_circuit_breaker', None)
                    if pkc_cfg and not isinstance(pkc_cfg, PerKeywordCircuitBreakerConfig):
                        pkc_cfg = PerKeywordCircuitBreakerConfig(
                            enabled=getattr(pkc_cfg, 'enabled', True),
                            consecutive_failures_threshold=getattr(pkc_cfg, 'consecutive_failures_threshold', 3),
                            pause_seconds=getattr(pkc_cfg, 'pause_seconds', 30.0),
                            max_pause_seconds=getattr(pkc_cfg, 'max_pause_seconds', 120.0),
                            jitter_factor=getattr(pkc_cfg, 'jitter_factor', 0.2),
                        )
                keyword_cb = PerKeywordCircuitBreaker(pkc_cfg) if pkc_cfg else None
            else:
                results_per_keyword = getattr(search_config, 'results_per_keyword', 20)
                max_total_results = getattr(search_config, 'max_total_results', 200)
                search_budget_aware = getattr(search_config, 'search_budget_aware', True)
                auto_distribute_budget = getattr(search_config, 'auto_distribute_budget', True)
                enable_channel_diversity = getattr(search_config, 'enable_channel_diversity', True)
                max_videos_per_channel = getattr(search_config, 'max_videos_per_channel', 3)
                use_negative_context = getattr(search_config, 'use_negative_context', False)
                negative_keywords = getattr(search_config, 'negative_keywords', []) or []
                use_chapter_queries = getattr(search_config, 'use_chapter_queries', True)
                listicle_topic_as_search_terms = getattr(search_config, 'listicle_topic_as_search_terms', True)

            # US-98-008: Build listicle-specific search queries
            listicle_queries = []
            if listicle_topic_as_search_terms and hasattr(state, 'listicle_groups') and state.listicle_groups:
                listicle_queries = self._build_listicle_queries(state.listicle_groups, state.topic_context)
                if listicle_queries:
                    logger.info(f"US-98-008: Generated {len(listicle_queries)} listicle-specific queries")

            # US-98-005: Build chapter-specific search queries
            chapter_queries = []
            if use_chapter_queries and hasattr(state, 'location_chapters') and state.location_chapters:
                chapter_queries = self._build_chapter_queries(state.location_chapters, state.topic_context)
                if chapter_queries:
                    logger.info(f"US-98-005: Generated {len(chapter_queries)} chapter-specific queries")

            # Calculate adjusted results_per_keyword when keywords exceed budget capacity
            keyword_count = len(state.keywords)
            effective_results_per_keyword = results_per_keyword
            budget_info = None

            if search_budget_aware and auto_distribute_budget and keyword_count > 0:
                potential_total = results_per_keyword * keyword_count
                if potential_total > max_total_results:
                    # Distribute budget evenly - reduce results per keyword, not skip keywords
                    effective_results_per_keyword = max(1, max_total_results // keyword_count)
                    budget_info = {
                        'original': results_per_keyword,
                        'adjusted': effective_results_per_keyword,
                        'keywords': keyword_count,
                        'max_total': max_total_results
                    }
                    logger.info(
                        f"Budget distribution: {results_per_keyword} * {keyword_count} = {potential_total} "
                        f"exceeds {max_total_results}, adjusted to {effective_results_per_keyword} per keyword"
                    )

            logger.info(f"Searching for videos: {effective_results_per_keyword} per keyword, max {max_total_results} total")

            # US-167-012: Add INFO-level logging for search budget consumption
            logger.info(
                f"Search budget configuration: keywords={keyword_count}, "
                f"results_per_keyword={effective_results_per_keyword}, "
                f"max_total={max_total_results}, "
                f"search_budget_aware={search_budget_aware}, "
                f"auto_distribute_budget={auto_distribute_budget}"
            )

            all_video_ids = []
            all_search_results = []
            failed_keywords = []

            # Helper to detect rate limit errors
            def is_rate_limit_error(error: Exception) -> bool:
                error_str = str(error).lower()
                return any(x in error_str for x in ['429', 'rate limit', 'too many requests', 'quota'])

            # US-98-005: First, search using standard keywords
            total_keywords = len(state.keywords)
            for idx, keyword in enumerate(state.keywords, 1):
                # US-159-010: Progress logging at 25% intervals
                if idx == 1 or idx % max(1, total_keywords // 4) == 0 or idx == total_keywords:
                    progress_pct = int((idx / total_keywords) * 100)
                    log_progress(logger, "VIDEO_SEARCH", progress_pct, idx, total_keywords, keyword=keyword)

                logger.info(f"[{idx}/{total_keywords}] Searching: {keyword}")

                # US-113-002: Apply per-keyword circuit breaker pause if enabled
                if keyword_cb:
                    keyword_cb.check_and_wait(keyword)

                # US-153-011: Generate query variations based on content type
                video_search_config = getattr(config.download, 'video_search', None)
                query_expansion_enabled = True
                if video_search_config:
                    if isinstance(video_search_config, dict):
                        query_expansion_enabled = video_search_config.get('query_expansion_enabled', True)
                    else:
                        query_expansion_enabled = getattr(video_search_config, 'query_expansion_enabled', True)

                if query_expansion_enabled:
                    # Detect content type
                    content_type = self._detect_content_type(keyword, state.topic_context or "")
                    # Generate variations
                    variations = self._generate_query_variations(keyword, content_type, video_search_config)
                else:
                    variations = [keyword]

                # Search using each variation
                all_keyword_results = []
                for variation in variations:
                    try:
                        results = self._search_keyword(
                            keyword=variation,
                            config=config,
                            max_results=effective_results_per_keyword,
                            topic=state.topic_context,
                            state=state
                        )

                        # US-113-002: Record success after search completes
                        if keyword_cb:
                            keyword_cb.record_success(keyword)

                        if results:
                            for r in results:
                                if r['video_id'] not in all_video_ids:
                                    all_keyword_results.append(r)
                            logger.debug(f"Variation '{variation}': {len(results)} videos (keyword: {keyword}, total accumulated: {len(all_keyword_results)})")
                        else:
                            logger.debug(f"Variation '{variation}': No results (keyword: {keyword})")

                    except Exception as e:
                        log_error_with_context(logger, "SEARCH-001", f"Search failed for variation '{variation}' (keyword: {keyword}): {e}")

                        # US-113-002: Record failure for rate limit errors
                        if keyword_cb and is_rate_limit_error(e):
                            keyword_cb.record_failure(keyword)
                            log_rate_limit(
                                logger, "circuit_breaker", "youtube_api", "rate_limit_detected",
                                keyword=keyword
                            )

                        warnings.append(f"Search failed for '{variation}': {e}")

                # Add results from all variations
                if all_keyword_results:
                    for r in all_keyword_results:
                        if r['video_id'] not in all_video_ids:
                            all_video_ids.append(r['video_id'])
                            all_search_results.append(r)
                    logger.info(f"Found {len(all_keyword_results)} unique videos from {len(variations)} variations")
                else:
                    failed_keywords.append(keyword)
                    logger.warning(f"No results from any variation for keyword")

                # Check max total
                if len(all_video_ids) >= max_total_results:
                    logger.info(f"Reached max results limit ({max_total_results})")
                    break

            # US-98-005: Search using chapter-specific queries
            if chapter_queries:
                logger.info(f"Chapter-specific search ({len(chapter_queries)} queries)")
                total_chapters = len(chapter_queries)
                for idx, cq in enumerate(chapter_queries, 1):
                    # US-159-010: Progress logging at 25% intervals
                    if idx == 1 or idx % max(1, total_chapters // 4) == 0 or idx == total_chapters:
                        progress_pct = int((idx / total_chapters) * 100)
                        log_progress(
                            logger, "VIDEO_SEARCH", progress_pct, idx, total_chapters,
                            chapter=cq['chapter_title']
                        )
                    keyword = cq['keyword']
                    chapter_id = cq['chapter_id']
                    chapter_title = cq['chapter_title']

                    # Check if we still have budget
                    if len(all_video_ids) >= max_total_results:
                        logger.info(f"Reached max results limit, skipping remaining chapter queries")
                        break

                    logger.info(f"[{idx}/{len(chapter_queries)}] Chapter '{chapter_title}': {keyword}")

                    # US-113-002: Apply per-keyword circuit breaker pause if enabled
                    if keyword_cb:
                        keyword_cb.check_and_wait(keyword)

                    try:
                        results = self._search_keyword(
                            keyword=keyword,
                            config=config,
                            max_results=effective_results_per_keyword,
                            topic=state.topic_context,
                            state=state
                        )

                        # US-113-002: Record success after search completes
                        if keyword_cb:
                            keyword_cb.record_success(keyword)

                        if results:
                            for r in results:
                                if r['video_id'] not in all_video_ids:
                                    # Tag result with source chapter
                                    r['chapter_id'] = chapter_id
                                    r['chapter_title'] = chapter_title
                                    all_video_ids.append(r['video_id'])
                                    all_search_results.append(r)

                            logger.debug(f"Found {len(results)} videos")
                        else:
                            logger.debug(f"No results")

                    except Exception as e:
                        log_error_with_context(logger, "SEARCH-001", f"Chapter search failed for keyword '{keyword}': {e}")

                        # US-113-002: Record failure for rate limit errors
                        if keyword_cb and is_rate_limit_error(e):
                            keyword_cb.record_failure(keyword)
                            log_rate_limit(
                                logger, "circuit_breaker", "youtube_api", "rate_limit_detected",
                                keyword=keyword, context="chapter"
                            )

                        warnings.append(f"Chapter search failed for '{keyword}': {e}")

            # US-98-008: Search using listicle-specific queries (prioritized)
            if listicle_queries:
                logger.info(f"Listicle-specific search ({len(listicle_queries)} queries)")
                total_listicles = len(listicle_queries)
                for idx, lq in enumerate(listicle_queries, 1):
                    # US-159-010: Progress logging at 25% intervals
                    if idx == 1 or idx % max(1, total_listicles // 4) == 0 or idx == total_listicles:
                        progress_pct = int((idx / total_listicles) * 100)
                        log_progress(
                            logger, "VIDEO_SEARCH", progress_pct, idx, total_listicles,
                            listicle=lq['item_label']
                        )
                    keyword = lq['keyword']
                    group_id = lq['group_id']
                    item_label = lq['item_label']

                    # Check if we still have budget
                    if len(all_video_ids) >= max_total_results:
                        logger.info(f"Reached max results limit, skipping remaining listicle queries")
                        break

                    logger.info(f"[{idx}/{len(listicle_queries)}] Listicle '{item_label}': {keyword}")

                    # US-113-002: Apply per-keyword circuit breaker pause if enabled
                    if keyword_cb:
                        keyword_cb.check_and_wait(keyword)

                    try:
                        results = self._search_keyword(
                            keyword=keyword,
                            config=config,
                            max_results=effective_results_per_keyword,
                            topic=state.topic_context,
                            state=state
                        )

                        # US-113-002: Record success after search completes
                        if keyword_cb:
                            keyword_cb.record_success(keyword)

                        if results:
                            for r in results:
                                if r['video_id'] not in all_video_ids:
                                    # Tag result with source listicle group
                                    r['listicle_group_id'] = group_id
                                    r['listicle_item_label'] = item_label
                                    all_video_ids.append(r['video_id'])
                                    all_search_results.append(r)

                            logger.debug(f"Found {len(results)} videos")
                        else:
                            logger.debug(f"No results")

                    except Exception as e:
                        log_error_with_context(logger, "SEARCH-001", f"Listicle search failed for keyword '{keyword}': {e}")

                        # US-113-002: Record failure for rate limit errors
                        if keyword_cb and is_rate_limit_error(e):
                            keyword_cb.record_failure(keyword)
                            log_rate_limit(
                                logger, "circuit_breaker", "youtube_api", "rate_limit_detected",
                                keyword=keyword, context="listicle"
                            )

                        warnings.append(f"Listicle search failed for '{keyword}': {e}")

            # Apply channel diversity filtering (US-94-009)
            if enable_channel_diversity and all_search_results:
                # Log initial channel distribution
                pre_filter_channels: Dict[str, int] = {}
                for r in all_search_results:
                    channel = r.get('channel', 'unknown')
                    pre_filter_channels[channel] = pre_filter_channels.get(channel, 0) + 1
                logger.debug(f"Channel distribution before filtering: {dict(sorted(pre_filter_channels.items(), key=lambda x: x[1], reverse=True)[:10])}")

                channel_counts: Dict[str, int] = {}
                filtered_ids = []
                filtered_results = []

                for r in all_search_results:
                    channel = r.get('channel', '')
                    if not channel:
                        # Include videos without channel info
                        filtered_ids.append(r['video_id'])
                        filtered_results.append(r)
                        continue

                    current_count = channel_counts.get(channel, 0)
                    if current_count < max_videos_per_channel:
                        channel_counts[channel] = current_count + 1
                        filtered_ids.append(r['video_id'])
                        filtered_results.append(r)
                    else:
                        logger.debug(f"Skipping video {r['video_id']} from channel '{channel}' (max {max_videos_per_channel} reached, current: {current_count})")

                # Log final channel distribution after filtering
                post_filter_channels: Dict[str, int] = {}
                for r in filtered_results:
                    channel = r.get('channel', 'unknown')
                    post_filter_channels[channel] = post_filter_channels.get(channel, 0) + 1
                logger.debug(f"Channel distribution after filtering: {dict(sorted(post_filter_channels.items(), key=lambda x: x[1], reverse=True)[:10])}")
                logger.debug(f"Channel filtering decision: {len(all_search_results)} -> {len(filtered_results)} videos (max {max_videos_per_channel} per channel)")

                removed_count = len(all_video_ids) - len(filtered_ids)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} videos due to channel diversity limit ({max_videos_per_channel} per channel)")

                all_video_ids = filtered_ids
                all_search_results = filtered_results

            # US-154-004: Cross-keyword deduplication
            # Deduplicate across all keyword searches and track keyword matches
            if all_search_results:
                # Track which keywords each video matched (before deduplication)
                video_keyword_matches: Dict[str, List[str]] = {}
                for r in all_search_results:
                    vid = r.get('video_id', '')
                    keyword = r.get('keyword', '')
                    if vid and keyword:
                        if vid not in video_keyword_matches:
                            video_keyword_matches[vid] = []
                        if keyword not in video_keyword_matches[vid]:
                            video_keyword_matches[vid].append(keyword)

                # Apply cross-keyword deduplication using the search_deduplication module
                original_count = len(all_search_results)
                all_search_results = deduplicate_by_video_id(all_search_results)

                # Calculate and log deduplication metrics
                deduped_count = original_count - len(all_search_results)
                if deduped_count > 0:
                    dedup_rate = (deduped_count / original_count) * 100
                    logger.info(f"Cross-keyword deduplication: removed {deduped_count} duplicates ({dedup_rate:.1f}% detection rate)")

                    # Log videos that matched multiple keywords
                    multi_keyword_videos = {vid: kws for vid, kws in video_keyword_matches.items() if len(kws) > 1}
                    if multi_keyword_videos:
                        logger.info(f"US-154-004: {len(multi_keyword_videos)} videos matched multiple keywords before deduplication")

                # US-154-004: Prioritize videos that matched multiple keywords
                # Add keyword_match_count to each result for prioritization
                multi_keyword_vids = {vid for vid, kws in video_keyword_matches.items() if len(kws) > 1}
                if multi_keyword_vids:
                    # Sort: multi-keyword matches first, then by original order
                    multi_keyword_set = multi_keyword_vids
                    all_search_results.sort(
                        key=lambda r: (r.get('video_id', '') not in multi_keyword_set,  # True(1) for non-multi, False(0) for multi
                                       0)  # Stable sort preserves original order for ties
                    )
                    logger.info(f"Prioritized {len(multi_keyword_vids)} videos that matched multiple keywords")

                # Update video_ids list to match the reordered results
                all_video_ids = [r['video_id'] for r in all_search_results]

            # US-146-008: Fetch enhanced video metadata from YouTube API
            # Get topic_details, tags, duration for better matching
            video_api_metadata = {}
            if all_video_ids:
                youtube_api_config = getattr(config.download, 'youtube_api', None)
                if youtube_api_config:
                    api_enabled = getattr(youtube_api_config, 'enabled', False)
                    api_key = getattr(youtube_api_config, 'api_key', '')

                    if api_enabled and api_key:
                        try:
                            # Check if topic_matching is enabled in matching config
                            matching_config = getattr(config, 'matching', None)
                            topic_matching = False
                            if matching_config:
                                topic_matching = getattr(matching_config, 'topic_matching_enabled', False)

                            if topic_matching:
                                # US-148-009: Get project size for quota auto-scaling
                                keyword_count = len(state.keywords) if state.keywords else 0
                                voiceover_segments = len(state.voiceover_segments) if hasattr(state, 'voiceover_segments') else 0
                                auto_scale_quota = getattr(youtube_api_config, 'auto_scale_budget', getattr(youtube_api_config, 'quota_auto_scale_enabled', True))

                                # Check quota for videos.list (costs 1 unit per request)
                                # US-149-012: Use context manager for proper resource cleanup
                                # US-153-002: Add rotation_strategy for multi-key rotation
                                # US-155-007: Add webhook parameters for quota alerts
                                # US-155-003: Add predictive quota fallback config
                                with YouTubeAPIClient(
                                    api_key=api_key,
                                    quota_limit=getattr(youtube_api_config, 'quota_limit', 10000),
                                    warn_at_percent=getattr(youtube_api_config, 'warn_at_percent', 80),
                                    quota_fallback_threshold_percent=getattr(youtube_api_config, 'quota_fallback_threshold_percent', 10),
                                    quota_fallback_prediction_minutes=getattr(youtube_api_config, 'quota_fallback_prediction_minutes', 30),
                                    quota_fallback_adaptive_enabled=getattr(youtube_api_config, 'quota_fallback_adaptive_enabled', True),
                                    quota_fallback_peak_multiplier=getattr(youtube_api_config, 'quota_fallback_peak_multiplier', 1.5),
                                    quota_fallback_peak_start_hour=getattr(youtube_api_config, 'quota_fallback_peak_start_hour', 9),
                                    quota_fallback_peak_end_hour=getattr(youtube_api_config, 'quota_fallback_peak_end_hour', 21),
                                    quota_abnormal_rate_warning_enabled=getattr(youtube_api_config, 'quota_abnormal_rate_warning_enabled', True),
                                    quota_abnormal_rate_threshold=getattr(youtube_api_config, 'quota_abnormal_rate_threshold', 2.0),
                                    max_retries=getattr(youtube_api_config, 'max_retries', 3),
                                    retry_delay=getattr(youtube_api_config, 'retry_delay_seconds', 2.0),
                                    timeout=getattr(youtube_api_config, 'timeout_seconds', 30),
                                    cache_ttl=getattr(youtube_api_config, 'cache_ttl_seconds', 3600),
                                    rate_limit_rps=getattr(youtube_api_config, 'rate_limit_rps', 10.0),
                                    keyword_count=keyword_count,
                                    voiceover_segments=voiceover_segments,
                                    auto_scale_quota=auto_scale_quota,
                                    rotation_strategy=getattr(youtube_api_config, 'rotation_strategy', 'sequential'),
                                    max_concurrent_requests=getattr(youtube_api_config, 'max_concurrent_requests', 5),
                                    webhook_enabled=getattr(youtube_api_config, 'webhook_enabled', False),
                                    webhook_urls=getattr(youtube_api_config, 'webhook_urls', []),
                                    webhook_timeout=getattr(youtube_api_config, 'webhook_timeout', 10),
                                    webhook_retry_count=getattr(youtube_api_config, 'webhook_retry_count', 3),
                                    # US-155-008: Adaptive rate limiting based on response latency
                                    adaptive_rate_limiting_enabled=getattr(youtube_api_config, 'adaptive_rate_limiting_enabled', True),
                                    latency_high_threshold_ms=getattr(youtube_api_config, 'latency_high_threshold_ms', 500.0),
                                    latency_low_threshold_ms=getattr(youtube_api_config, 'latency_low_threshold_ms', 200.0),
                                    rate_decrease_factor=getattr(youtube_api_config, 'rate_decrease_factor', 0.8),
                                    rate_increase_factor=getattr(youtube_api_config, 'rate_increase_factor', 1.1),
                                    min_adaptive_rate=getattr(youtube_api_config, 'min_adaptive_rate', 1.0),
                                    max_adaptive_rate=getattr(youtube_api_config, 'max_adaptive_rate', 20.0),
                                    latency_smoothing_window=getattr(youtube_api_config, 'latency_smoothing_window', 10),
                                    min_requests_before_adjustment=getattr(youtube_api_config, 'min_requests_before_adjustment', 5),
                                    # US-156-005: Search query sanitization and deduplication
                                    deduplicate_searches=getattr(youtube_api_config, 'deduplicate_searches', True),
                                    # US-158-004: Region code for localized search results
                                    region_code=getattr(youtube_api_config, 'region_code', 'US'),
                                    # US-158-005: Safe search level for family-friendly results
                                    safe_search=getattr(youtube_api_config, 'safe_search', 'moderate'),
                                    # US-158-002: Search ordering (relevance, date, viewCount, rating, videoCount)
                                    order_by=getattr(youtube_api_config, 'order_by', 'relevance'),
                                    # US-158-003: Video duration filter (any, short, medium, long)
                                    video_duration=getattr(youtube_api_config, 'video_duration', 'any'),
                                    # US-158-006: Batch caption fetching size
                                    caption_batch_size=getattr(youtube_api_config, 'caption_batch_size', 10),
                                ) as api_client:
                                    # US-146-012: Register client for metrics
                                    set_youtube_api_client(api_client)

                                    # US-155-010: Configure per-channel API tracking
                                    per_channel_enabled = getattr(youtube_api_config, 'per_channel_tracking_enabled', False)
                                    if per_channel_enabled:
                                        api_client.configure_per_channel_tracking(
                                            enabled=per_channel_enabled,
                                            rate_limit=getattr(youtube_api_config, 'per_channel_rate_limit', 100),
                                            circuit_breaker_enabled=getattr(youtube_api_config, 'per_channel_circuit_breaker_enabled', False),
                                            circuit_breaker_threshold=getattr(youtube_api_config, 'per_channel_circuit_breaker_threshold', 5),
                                            circuit_breaker_pause=getattr(youtube_api_config, 'per_channel_circuit_breaker_pause_seconds', 60.0),
                                            graceful_no_videos=getattr(youtube_api_config, 'per_channel_graceful_no_videos', True),
                                        )

                                    remaining = api_client.get_remaining_quota()
                                    if remaining >= len(all_video_ids):  # 1 unit per video
                                        # US-167-009: DEBUG-level sub-stage timing for video details fetch
                                        details_start = time.time()
                                        video_details, failed_video_ids = api_client.get_video_details(
                                            video_ids=all_video_ids,
                                            part="contentDetails,statistics,topicDetails"
                                        )
                                        details_elapsed = time.time() - details_start
                                        logger.debug(
                                            f"[VIDEO_SEARCH] Sub-stage timing: get_video_details "
                                            f"for {len(all_video_ids)} videos took {details_elapsed:.2f}s"
                                        )

                                        # US-156-007: Handle partial failures
                                        if failed_video_ids:
                                            logger.warning(
                                                f"get_video_details: {len(failed_video_ids)} videos failed to fetch"
                                            )

                                        # Store metadata by video_id (now returns Dict[str, VideoDetails])
                                        for vd in video_details.values():
                                            video_api_metadata[vd.video_id] = {
                                                'duration': vd.duration_seconds,
                                                'tags': vd.tags,
                                                'topic_details': vd.topic_details,
                                                'topic_categories': vd.topic_categories,  # US-150-006
                                                'caption_available': vd.caption_available,
                                                'dimension': vd.dimension,
                                                'definition': vd.definition,
                                                # US-155-006: Engagement metrics
                                                'view_count': vd.view_count,
                                                'like_count': vd.like_count,
                                                'comment_count': vd.comment_count,
                                            }

                                        logger.info(f"US-146-008: Fetched API metadata for {len(video_api_metadata)} videos")

                                        # Override duration from API (more accurate than yt-dlp)
                                        # Also add engagement metrics (US-155-006)
                                        for result in all_search_results:
                                            vid = result.get('video_id', '')
                                            if vid in video_api_metadata:
                                                api_duration = video_api_metadata[vid].get('duration', 0)
                                                if api_duration > 0:
                                                    result['duration'] = api_duration
                                                    result['api_tags'] = video_api_metadata[vid].get('tags', [])
                                                    result['topic_details'] = video_api_metadata[vid].get('topic_details', {})
                                                    result['topic_categories'] = video_api_metadata[vid].get('topic_categories', [])  # US-150-006

                                                # US-155-006: Add engagement metrics to results
                                                view_count = video_api_metadata[vid].get('view_count', 0)
                                                like_count = video_api_metadata[vid].get('like_count', 0)
                                                comment_count = video_api_metadata[vid].get('comment_count', 0)

                                                result['view_count'] = view_count
                                                result['like_count'] = like_count
                                                result['comment_count'] = comment_count

                                                # Calculate engagement score (0.0 - 1.0)
                                                result['engagement_score'] = self._calculate_engagement_score(
                                                    view_count, like_count, comment_count
                                                )

                                        logger.info(f"Enriched {len(video_api_metadata)} videos with YouTube API metadata")

                        except Exception as e:
                            log_error_with_context(logger, "SEARCH-001", f"US-146-008: Failed to fetch video details from YouTube API: {e}")

            # US-146-006: Enrich with channel metadata (subscriber count, total views)
            # US-153-006: Respect include_channel_metadata config option
            channel_metadata = {}
            if all_search_results:
                # US-153-006: Check if channel metadata should be fetched
                search_config = getattr(config, 'video_search', None)
                include_channel_meta = True
                if search_config:
                    include_channel_meta = getattr(search_config, 'include_channel_metadata', True)

                if include_channel_meta:
                    youtube_api_config = getattr(config.download, 'youtube_api', None)
                    if youtube_api_config:
                        api_enabled = getattr(youtube_api_config, 'enabled', False)
                        api_key = getattr(youtube_api_config, 'api_key', '')

                        if api_enabled and api_key:
                            # Collect unique channel IDs
                            channel_ids = list(set(
                                r.get('channel_id', '') or r.get('channel', '')
                                for r in all_search_results
                                if r.get('channel_id') or r.get('channel')
                            ))
                        else:
                            channel_ids = []

                        if channel_ids:
                            try:
                                # US-148-009: Get project size for quota auto-scaling
                                keyword_count = len(state.keywords) if state.keywords else 0
                                voiceover_segments = len(state.voiceover_segments) if hasattr(state, 'voiceover_segments') else 0
                                auto_scale_quota = getattr(youtube_api_config, 'auto_scale_budget', getattr(youtube_api_config, 'quota_auto_scale_enabled', True))

                                # US-149-012: Use context manager for proper resource cleanup
                                # US-155-007: Add webhook parameters for quota alerts
                                # US-155-003: Add predictive quota fallback config
                                with YouTubeAPIClient(
                                    api_key=api_key,
                                    quota_limit=getattr(youtube_api_config, 'quota_limit', 10000),
                                    warn_at_percent=getattr(youtube_api_config, 'warn_at_percent', 80),
                                    quota_fallback_threshold_percent=getattr(youtube_api_config, 'quota_fallback_threshold_percent', 10),
                                    quota_fallback_prediction_minutes=getattr(youtube_api_config, 'quota_fallback_prediction_minutes', 30),
                                    quota_fallback_adaptive_enabled=getattr(youtube_api_config, 'quota_fallback_adaptive_enabled', True),
                                    quota_fallback_peak_multiplier=getattr(youtube_api_config, 'quota_fallback_peak_multiplier', 1.5),
                                    quota_fallback_peak_start_hour=getattr(youtube_api_config, 'quota_fallback_peak_start_hour', 9),
                                    quota_fallback_peak_end_hour=getattr(youtube_api_config, 'quota_fallback_peak_end_hour', 21),
                                    quota_abnormal_rate_warning_enabled=getattr(youtube_api_config, 'quota_abnormal_rate_warning_enabled', True),
                                    quota_abnormal_rate_threshold=getattr(youtube_api_config, 'quota_abnormal_rate_threshold', 2.0),
                                    max_retries=getattr(youtube_api_config, 'max_retries', 3),
                                    retry_delay=getattr(youtube_api_config, 'retry_delay_seconds', 2.0),
                                    timeout=getattr(youtube_api_config, 'timeout_seconds', 30),
                                    cache_ttl=getattr(youtube_api_config, 'cache_ttl_seconds', 3600),
                                    channel_cache_ttl=getattr(youtube_api_config, 'channel_metadata_cache_ttl_seconds', 604800),  # US-153-006: Default 7 days
                                    rate_limit_rps=getattr(youtube_api_config, 'rate_limit_rps', 10.0),
                                    keyword_count=keyword_count,
                                    voiceover_segments=voiceover_segments,
                                    auto_scale_quota=auto_scale_quota,
                                    rotation_strategy=getattr(youtube_api_config, 'rotation_strategy', 'sequential'),
                                    max_concurrent_requests=getattr(youtube_api_config, 'max_concurrent_requests', 5),
                                    webhook_enabled=getattr(youtube_api_config, 'webhook_enabled', False),
                                    webhook_urls=getattr(youtube_api_config, 'webhook_urls', []),
                                    webhook_timeout=getattr(youtube_api_config, 'webhook_timeout', 10),
                                    webhook_retry_count=getattr(youtube_api_config, 'webhook_retry_count', 3),
                                    # US-155-008: Adaptive rate limiting based on response latency
                                    adaptive_rate_limiting_enabled=getattr(youtube_api_config, 'adaptive_rate_limiting_enabled', True),
                                    latency_high_threshold_ms=getattr(youtube_api_config, 'latency_high_threshold_ms', 500.0),
                                    latency_low_threshold_ms=getattr(youtube_api_config, 'latency_low_threshold_ms', 200.0),
                                    rate_decrease_factor=getattr(youtube_api_config, 'rate_decrease_factor', 0.8),
                                    rate_increase_factor=getattr(youtube_api_config, 'rate_increase_factor', 1.1),
                                    min_adaptive_rate=getattr(youtube_api_config, 'min_adaptive_rate', 1.0),
                                    max_adaptive_rate=getattr(youtube_api_config, 'max_adaptive_rate', 20.0),
                                    latency_smoothing_window=getattr(youtube_api_config, 'latency_smoothing_window', 10),
                                    min_requests_before_adjustment=getattr(youtube_api_config, 'min_requests_before_adjustment', 5),
                                    # US-156-005: Search query sanitization and deduplication
                                    deduplicate_searches=getattr(youtube_api_config, 'deduplicate_searches', True),
                                    # US-158-004: Region code for localized search results
                                    region_code=getattr(youtube_api_config, 'region_code', 'US'),
                                    # US-158-005: Safe search level for family-friendly results
                                    safe_search=getattr(youtube_api_config, 'safe_search', 'moderate'),
                                    # US-158-002: Search ordering (relevance, date, viewCount, rating, videoCount)
                                    order_by=getattr(youtube_api_config, 'order_by', 'relevance'),
                                    # US-158-003: Video duration filter (any, short, medium, long)
                                    video_duration=getattr(youtube_api_config, 'video_duration', 'any'),
                                    # US-158-006: Batch caption fetching size
                                    caption_batch_size=getattr(youtube_api_config, 'caption_batch_size', 10),
                                ) as api_client:
                                    # US-146-012: Register client for metrics
                                    set_youtube_api_client(api_client)

                                    # US-155-010: Configure per-channel API tracking
                                    per_channel_enabled = getattr(youtube_api_config, 'per_channel_tracking_enabled', False)
                                    if per_channel_enabled:
                                        api_client.configure_per_channel_tracking(
                                            enabled=per_channel_enabled,
                                            rate_limit=getattr(youtube_api_config, 'per_channel_rate_limit', 100),
                                            circuit_breaker_enabled=getattr(youtube_api_config, 'per_channel_circuit_breaker_enabled', False),
                                            circuit_breaker_threshold=getattr(youtube_api_config, 'per_channel_circuit_breaker_threshold', 5),
                                            circuit_breaker_pause=getattr(youtube_api_config, 'per_channel_circuit_breaker_pause_seconds', 60.0),
                                            graceful_no_videos=getattr(youtube_api_config, 'per_channel_graceful_no_videos', True),
                                        )

                                    # Check if we have enough quota for channels.list (1 unit per request)
                                    remaining = api_client.get_remaining_quota()
                                    if remaining >= len(channel_ids):
                                        # US-167-009: DEBUG-level sub-stage timing for channel metadata fetch
                                        channel_start = time.time()
                                        channel_metadata = api_client.get_channel_metadata(channel_ids)
                                        channel_elapsed = time.time() - channel_start
                                        logger.debug(
                                            f"[VIDEO_SEARCH] Sub-stage timing: get_channel_metadata "
                                            f"for {len(channel_ids)} channels took {channel_elapsed:.2f}s"
                                        )
                                        logger.info(f"US-146-006: Fetched channel metadata for {len(channel_metadata)} channels")

                            except Exception as e:
                                log_error_with_context(logger, "SEARCH-001", f"US-146-006: Failed to fetch channel metadata for {len(channel_ids)} channels: {e}")

            # US-146-006: Apply channel metadata and filtering
            # Get subscriber threshold from video_search config (not download.youtube_api)
            min_subscriber_count = 0
            enable_channel_quality_score = False
            channel_quality_boost_factor = 0.05

            # Read from video_search config section
            if search_config := getattr(config.download, 'video_search', None):
                if isinstance(search_config, dict):
                    min_subscriber_count = search_config.get('min_subscriber_threshold', 0)
                    enable_channel_quality_score = search_config.get('enable_channel_quality_score', False)
                    channel_quality_boost_factor = search_config.get('channel_quality_boost_factor', 0.05)
                else:
                    min_subscriber_count = getattr(search_config, 'min_subscriber_threshold', 0)
                    enable_channel_quality_score = getattr(search_config, 'enable_channel_quality_score', False)
                    channel_quality_boost_factor = getattr(search_config, 'channel_quality_boost_factor', 0.05)

            if min_subscriber_count > 0:
                logger.info(f"US-146-006: Filtering videos with subscriber count < {min_subscriber_count}")

            filtered_results = []
            videos_filtered = 0
            for r in all_search_results:
                channel_id = r.get('channel_id', '') or r.get('channel', '')
                if channel_id and channel_metadata and channel_id in channel_metadata:
                    meta = channel_metadata[channel_id]
                    subscriber_count = meta.get('subscriber_count', 0)

                    # US-146-006: Filter by minimum subscriber threshold
                    if min_subscriber_count > 0 and subscriber_count < min_subscriber_count:
                        videos_filtered += 1
                        continue

                    # Add channel metadata to result
                    r['subscriber_count'] = subscriber_count
                    r['channel_total_views'] = meta.get('view_count', 0)
                    r['channel_created_date'] = meta.get('published_at', '')

                    # US-153-006: Add channel status info (verified badge not directly available via API)
                    r['channel_is_linked'] = meta.get('is_linked', False)
                    r['channel_made_for_kids'] = meta.get('made_for_kids', False)

                    # US-146-006: Calculate channel quality score
                    # Based on subscriber count and activity (video count)
                    video_count = meta.get('video_count', 0)
                    r['channel_quality_score'] = self._calculate_channel_quality_score(
                        subscriber_count, video_count, meta.get('view_count', 0)
                    )

                filtered_results.append(r)

            if videos_filtered > 0:
                logger.info(f"Filtered {videos_filtered} videos with subscriber count < {min_subscriber_count}")

            all_search_results = filtered_results

            # US-148-008: Apply engagement metrics ranking when YouTube API enabled
            youtube_api_config = getattr(config.download, 'youtube_api', None)
            if youtube_api_config:
                # Handle both dict and object access patterns
                if isinstance(youtube_api_config, dict):
                    use_engagement_ranking = youtube_api_config.get('use_engagement_ranking', False)
                    engagement_ranking = youtube_api_config.get('enable_engagement_ranking', False)
                    use_engagement_ranking = use_engagement_ranking or engagement_ranking
                else:
                    use_engagement_ranking = getattr(youtube_api_config, 'use_engagement_ranking', False) or getattr(youtube_api_config, 'enable_engagement_ranking', False)

                if use_engagement_ranking and all_search_results:
                    # Check if any results have engagement metrics
                    has_engagement = any(
                        r.get('engagement_score') is not None
                        for r in all_search_results
                    )

                    if has_engagement:
                        # Sort by engagement score (highest first)
                        all_search_results.sort(
                            key=lambda r: r.get('engagement_score', 0),
                            reverse=True
                        )

                        # Also update video_ids to match the sorted order
                        all_video_ids = [r['video_id'] for r in all_search_results]

                        logger.debug(f"Results sorted by engagement score (YouTube API)")

            # US-158-010: Apply quality boost for video engagement metrics in result ranking
            youtube_api_config = getattr(config.download, 'youtube_api', None)
            if youtube_api_config:
                # Handle both dict and object access patterns
                if isinstance(youtube_api_config, dict):
                    quality_boost_enabled = youtube_api_config.get('quality_boost_enabled', False)
                    quality_view_weight = youtube_api_config.get('quality_view_weight', 0.7)
                    quality_like_weight = youtube_api_config.get('quality_like_weight', 0.2)
                    quality_comment_weight = youtube_api_config.get('quality_comment_weight', 0.1)
                else:
                    quality_boost_enabled = getattr(youtube_api_config, 'quality_boost_enabled', False)
                    quality_view_weight = getattr(youtube_api_config, 'quality_view_weight', 0.7)
                    quality_like_weight = getattr(youtube_api_config, 'quality_like_weight', 0.2)
                    quality_comment_weight = getattr(youtube_api_config, 'quality_comment_weight', 0.1)

                if quality_boost_enabled and all_search_results:
                    # Calculate quality score for each result if not already present
                    for r in all_search_results:
                        if 'quality_score' not in r or r['quality_score'] is None:
                            view_count = r.get('view_count', 0) or 0
                            like_count = r.get('like_count', 0) or 0
                            comment_count = r.get('comment_count', 0) or 0
                            # US-158-010: Calculate quality score: viewCount * 0.7 + likeCount * 0.2 + commentCount * 0.1
                            # Handle zero engagement gracefully
                            if view_count == 0 and like_count == 0 and comment_count == 0:
                                r['quality_score'] = 0.0
                            else:
                                r['quality_score'] = (
                                    view_count * quality_view_weight +
                                    like_count * quality_like_weight +
                                    comment_count * quality_comment_weight
                                )
                                logger.debug(
                                    f"Quality score calculated for {r.get('video_id', 'unknown')}: "
                                    f"views={view_count}*{quality_view_weight} + "
                                    f"likes={like_count}*{quality_like_weight} + "
                                    f"comments={comment_count}*{quality_comment_weight} = "
                                    f"{r['quality_score']:.2f}"
                                )

                    # Log quality score distribution before sorting
                    quality_scores = [r.get('quality_score', 0) for r in all_search_results if r.get('quality_score', 0) > 0]
                    if quality_scores:
                        logger.debug(
                            f"Quality score distribution: min={min(quality_scores):.2f}, "
                            f"max={max(quality_scores):.2f}, avg={sum(quality_scores)/len(quality_scores):.2f}, "
                            f"videos_with_scores={len(quality_scores)}"
                        )

                    # Sort by quality score (highest first)
                    all_search_results.sort(
                        key=lambda r: r.get('quality_score', 0),
                        reverse=True
                    )

                    # Update video_ids to match the sorted order
                    all_video_ids = [r['video_id'] for r in all_search_results]

                    logger.debug(f"Results sorted by quality score (quality_boost_enabled: weights=view:{quality_view_weight}, like:{quality_like_weight}, comment:{quality_comment_weight})")

            # US-157-004: Apply relevance scoring to filter and score results
            # Use the last successful query as reference for relevance
            if all_search_results and 'search_query' in locals() and search_query:
                all_search_results = self._score_results_by_relevance(
                    all_search_results,
                    search_query,
                    video_search_config
                )
                # Update video_ids to match the filtered/scored results
                all_video_ids = [r['video_id'] for r in all_search_results if 'video_id' in r]

            # Store results in state
            state.video_ids = all_video_ids
            state.video_search_results = self._to_search_results(
                all_search_results,
                negative_keywords=negative_keywords if use_negative_context else []
            )
            state.search_failed_keywords = failed_keywords

            logger.info(f"Found {len(all_video_ids)} unique videos")
            if failed_keywords:
                logger.info(f"{len(failed_keywords)} keywords had no results")

            # US-167-012: Add INFO-level logging for search budget consumption summary
            actual_results_used = min(len(all_video_ids), max_total_results)
            budget_utilization = (actual_results_used / max_total_results * 100) if max_total_results > 0 else 0
            logger.info(
                f"Search budget consumption summary: "
                f"requested={max_total_results}, "
                f"actual={actual_results_used}, "
                f"utilization={budget_utilization:.1f}%, "
                f"keywords_processed={keyword_count - len(failed_keywords)}, "
                f"keywords_failed={len(failed_keywords)}"
            )

            # US-146-012: Log YouTube API vs yt-dlp usage summary
            fallback_data = get_fallback_metrics()
            total_fallbacks = fallback_data.get("total_fallbacks", 0)
            if total_fallbacks > 0:
                logger.info(
                    f"YouTube API: Fallback to yt-dlp occurred {total_fallbacks} times "
                    f"(see logs for details)"
                )

            # Prepare checkpoint data
            checkpoint_data = {
                'video_ids': all_video_ids,
                'search_results': all_search_results,
                'failed_keywords': failed_keywords,
                'video_count': len(all_video_ids),
            }

            # US-167-009: Log stage completion with timing
            elapsed = time.time() - stage_start_time
            log_stage_complete(
                logger, "VIDEO_SEARCH",
                elapsed_seconds=elapsed,
                results_count=len(all_video_ids),
                total_videos=len(all_video_ids),
                total_keywords=len(state.keywords),
                failed_keywords=len(failed_keywords)
            )

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            log_error_with_context(logger, "SEARCH-001", f"Video search stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _search_keyword(
        self,
        keyword: str,
        config: 'Config',
        max_results: int = 20,
        topic: str = "",
        state: 'PipelineState' = None
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube for videos matching a keyword.

        Returns list of dicts with video metadata (no download).
        Uses two-phase search: initial + refined with description keywords.
        """
        import yt_dlp

        # Get config
        video_search_config = config.video_search
        download_config = config.download

        # Handle both dict and object access patterns (Rule #6)
        if isinstance(video_search_config, dict):
            use_tags = video_search_config.get('use_tags_in_search', True)
            use_description_context = video_search_config.get('use_description_context', True)
            use_negative_context = video_search_config.get('use_negative_context', False)
            negative_keywords = video_search_config.get('negative_keywords', []) or []
            topic_tags_map = video_search_config.get('topic_tags', {}) or {}
            # US-153-011: Query template optimization
            query_expansion_enabled = video_search_config.get('query_expansion_enabled', True)
        else:
            use_tags = getattr(video_search_config, 'use_tags_in_search', True)
            use_description_context = getattr(video_search_config, 'use_description_context', True)
            use_negative_context = getattr(video_search_config, 'use_negative_context', False)
            negative_keywords = getattr(video_search_config, 'negative_keywords', []) or []
            topic_tags_map = getattr(video_search_config, 'topic_tags', {}) or {}
            # US-153-011: Query template optimization
            query_expansion_enabled = getattr(video_search_config, 'query_expansion_enabled', True)

        # US-153-011: Detect content type and generate query variations
        if query_expansion_enabled:
            content_type = self._detect_content_type(keyword, topic)
            query_variations = self._generate_query_variations(keyword, content_type, video_search_config)
            logger.info(f"US-153-011: Query template selection - keyword='{keyword}', content_type='{content_type}', variations={query_variations}")

            # Use the first variation for initial search
            primary_query = query_variations[0] if query_variations else keyword

            # Optimize query length
            primary_query = self._optimize_query_length(primary_query, video_search_config)

            # Build full query with tags
            search_query = self._build_search_query(primary_query, topic, use_tags, topic_tags_map)
        else:
            # Original behavior
            search_query = self._build_search_query(keyword, topic, use_tags, topic_tags_map)

        # US-157-004: Apply query preprocessing (stopwords, whitespace, special chars)
        search_query = self._preprocess_query(search_query, video_search_config)

        # US-157-004: Apply query expansion with related terms
        search_query = self._expand_query(search_query, video_search_config, topic)

        # US-157-004: Truncate to max_query_length
        search_query = self._truncate_query_length(search_query, video_search_config)

        # Get duration filter config
        min_duration = getattr(download_config, 'min_duration', 30)
        max_duration = getattr(download_config, 'max_duration', 600)

        # US-146-004: Try YouTube API first if enabled, fallback to yt-dlp on quota exhaustion
        youtube_api_config = getattr(download_config, 'youtube_api', None)
        if youtube_api_config:
            # Handle both dict and object access patterns (Rule #6)
            if isinstance(youtube_api_config, dict):
                api_enabled = youtube_api_config.get('enabled', False)
                api_key = youtube_api_config.get('api_key', '')
            else:
                api_enabled = getattr(youtube_api_config, 'enabled', False)
                api_key = getattr(youtube_api_config, 'api_key', '')

            if api_enabled and api_key:
                # US-148-009: Get project size for quota auto-scaling
                keyword_count = len(state.keywords) if state and state.keywords else 0
                voiceover_segments = len(state.voiceover_segments) if state and hasattr(state, 'voiceover_segments') else 0

                # Try YouTube API first
                try:
                    # US-149-012: Use context manager for proper resource cleanup
                    # US-155-007: Add webhook parameters for quota alerts
                    # US-155-003: Add predictive quota fallback config
                    with YouTubeAPIClient(
                        api_key=api_key,
                        quota_limit=getattr(youtube_api_config, 'quota_limit', 10000),
                        warn_at_percent=getattr(youtube_api_config, 'warn_at_percent', 80),
                        quota_fallback_threshold_percent=getattr(youtube_api_config, 'quota_fallback_threshold_percent', 10),
                        quota_fallback_prediction_minutes=getattr(youtube_api_config, 'quota_fallback_prediction_minutes', 30),
                        quota_fallback_adaptive_enabled=getattr(youtube_api_config, 'quota_fallback_adaptive_enabled', True),
                        quota_fallback_peak_multiplier=getattr(youtube_api_config, 'quota_fallback_peak_multiplier', 1.5),
                        quota_fallback_peak_start_hour=getattr(youtube_api_config, 'quota_fallback_peak_start_hour', 9),
                        quota_fallback_peak_end_hour=getattr(youtube_api_config, 'quota_fallback_peak_end_hour', 21),
                        quota_abnormal_rate_warning_enabled=getattr(youtube_api_config, 'quota_abnormal_rate_warning_enabled', True),
                        quota_abnormal_rate_threshold=getattr(youtube_api_config, 'quota_abnormal_rate_threshold', 2.0),
                        max_retries=getattr(youtube_api_config, 'max_retries', 3),
                        retry_delay=getattr(youtube_api_config, 'retry_delay_seconds', 2.0),
                        timeout=getattr(youtube_api_config, 'timeout_seconds', 30),
                        cache_ttl=getattr(youtube_api_config, 'cache_ttl_seconds', 3600),
                        channel_cache_ttl=getattr(
                            youtube_api_config, 'channel_metadata_cache_ttl_seconds', 86400
                        ),
                        rate_limit_rps=getattr(youtube_api_config, 'rate_limit_rps', 10.0),
                        keyword_count=keyword_count,
                        voiceover_segments=voiceover_segments,
                        rotation_strategy=getattr(youtube_api_config, 'rotation_strategy', 'sequential'),
                        max_concurrent_requests=getattr(youtube_api_config, 'max_concurrent_requests', 5),
                        webhook_enabled=getattr(youtube_api_config, 'webhook_enabled', False),
                        webhook_urls=getattr(youtube_api_config, 'webhook_urls', []),
                        webhook_timeout=getattr(youtube_api_config, 'webhook_timeout', 10),
                        webhook_retry_count=getattr(youtube_api_config, 'webhook_retry_count', 3),
                        # US-155-005: Parallel video details fetching
                        parallel_video_details_enabled=getattr(youtube_api_config, 'parallel_video_details_enabled', True),
                        video_details_chunk_size=getattr(youtube_api_config, 'video_details_chunk_size', 50),
                        video_details_max_workers=getattr(youtube_api_config, 'video_details_max_workers', 5),
                        # US-155-008: Adaptive rate limiting based on response latency
                        adaptive_rate_limiting_enabled=getattr(youtube_api_config, 'adaptive_rate_limiting_enabled', True),
                        latency_high_threshold_ms=getattr(youtube_api_config, 'latency_high_threshold_ms', 500.0),
                        latency_low_threshold_ms=getattr(youtube_api_config, 'latency_low_threshold_ms', 200.0),
                        rate_decrease_factor=getattr(youtube_api_config, 'rate_decrease_factor', 0.8),
                        rate_increase_factor=getattr(youtube_api_config, 'rate_increase_factor', 1.1),
                        min_adaptive_rate=getattr(youtube_api_config, 'min_adaptive_rate', 1.0),
                        max_adaptive_rate=getattr(youtube_api_config, 'max_adaptive_rate', 20.0),
                        latency_smoothing_window=getattr(youtube_api_config, 'latency_smoothing_window', 10),
                        min_requests_before_adjustment=getattr(youtube_api_config, 'min_requests_before_adjustment', 5),
                        # US-158-004: Region code for localized search results
                        region_code=getattr(youtube_api_config, 'region_code', 'US'),
                        # US-158-005: Safe search level for family-friendly results
                        safe_search=getattr(youtube_api_config, 'safe_search', 'moderate'),
                        # US-158-002: Search ordering (relevance, date, viewCount, rating, videoCount)
                        order_by=getattr(youtube_api_config, 'order_by', 'relevance'),
                        # US-158-003: Video duration filter (any, short, medium, long)
                        video_duration=getattr(youtube_api_config, 'video_duration', 'any'),
                        # US-158-006: Batch caption fetching size
                        caption_batch_size=getattr(youtube_api_config, 'caption_batch_size', 10),
                    ) as api_client:
                        # US-146-012: Register client for metrics
                        set_youtube_api_client(api_client)

                        # US-155-008: Clear deduplication cache at start of new search session
                        api_client.clear_deduplication_cache()

                        # US-155-010: Configure per-channel API tracking and clear at start of session
                        per_channel_enabled = getattr(youtube_api_config, 'per_channel_tracking_enabled', False)
                        if per_channel_enabled:
                            api_client.configure_per_channel_tracking(
                                enabled=per_channel_enabled,
                                rate_limit=getattr(youtube_api_config, 'per_channel_rate_limit', 100),
                                circuit_breaker_enabled=getattr(youtube_api_config, 'per_channel_circuit_breaker_enabled', False),
                                circuit_breaker_threshold=getattr(youtube_api_config, 'per_channel_circuit_breaker_threshold', 5),
                                circuit_breaker_pause=getattr(youtube_api_config, 'per_channel_circuit_breaker_pause_seconds', 60.0),
                                graceful_no_videos=getattr(youtube_api_config, 'per_channel_graceful_no_videos', True),
                            )
                            # Clear channel tracking for fresh session
                            api_client.clear_channel_tracking()

                        # Check quota before searching
                        remaining = api_client.get_remaining_quota()
                        if remaining >= QUOTA_COST_SEARCH:
                            # Use fallback handler for transparent fallback
                            fallback_handler = YouTubeAPIFallbackHandler(
                                api_client=api_client, config=config
                            )

                            # US-155-004: Get date range from config
                            date_range_enabled = False
                            published_after = ""
                            published_before = ""

                            # US-158-011: Get video category filter from config
                            video_category_enabled = False
                            video_category_id = ""

                            if youtube_api_config:
                                if isinstance(youtube_api_config, dict):
                                    date_range_enabled = youtube_api_config.get('date_range_enabled', False)
                                    published_after = youtube_api_config.get('published_after', '')
                                    published_before = youtube_api_config.get('published_before', '')
                                    # Use preset if enabled and no explicit dates set
                                    if date_range_enabled and not published_after:
                                        date_range_preset = youtube_api_config.get('date_range_preset', 'last_30_days')
                                        published_after = date_range_preset
                                    # US-158-011: Video category filtering
                                    video_category_enabled = youtube_api_config.get('video_category_enabled', False)
                                    video_category_ids = youtube_api_config.get('video_category_ids', [])
                                    if video_category_enabled and video_category_ids:
                                        video_category_id = video_category_ids[0]  # Use first category
                                else:
                                    date_range_enabled = getattr(youtube_api_config, 'date_range_enabled', False)
                                    published_after = getattr(youtube_api_config, 'published_after', '')
                                    published_before = getattr(youtube_api_config, 'published_before', '')
                                    if date_range_enabled and not published_after:
                                        date_range_preset = getattr(youtube_api_config, 'date_range_preset', 'last_30_days')
                                        published_after = date_range_preset
                                    # US-158-011: Video category filtering
                                    video_category_enabled = getattr(youtube_api_config, 'video_category_enabled', False)
                                    video_category_ids = getattr(youtube_api_config, 'video_category_ids', [])
                                    if video_category_enabled and video_category_ids:
                                        video_category_id = video_category_ids[0]  # Use first category

                            # DEBUG: Log API search operation details
                            logger.debug(
                                f"API search operation: query='{search_query}', "
                                f"max_results={max_results}, min_duration={min_duration}, max_duration={max_duration}, "
                                f"date_range={date_range_enabled}, video_category={video_category_id}, "
                                f"quota_remaining={api_client.get_remaining_quota()}"
                            )

                            api_results = fallback_handler.search_with_fallback(
                                query=search_query,
                                max_results=max_results,
                                min_duration=min_duration,
                                max_duration=max_duration,
                                published_after=published_after if date_range_enabled else "",
                                published_before=published_before if date_range_enabled else "",
                                video_category_id=video_category_id if video_category_enabled else "",
                            )

                            logger.debug(
                                f"API search result: query='{search_query}', "
                                f"results_count={len(api_results) if api_results else 0}, "
                                f"fallback_occurred={fallback_handler.fallback_occurred}"
                            )

                            if api_results:
                                # Log if fallback occurred
                                if fallback_handler.fallback_occurred:
                                    logger.info(
                                        f"YouTube API fallback to yt-dlp for '{keyword}': "
                                        f"reason={fallback_handler.fallback_reason}"
                                    )

                                # Store metrics about API vs fallback
                                api_source = "yt-dlp-fallback" if fallback_handler.fallback_occurred else "youtube-api"

                                # Convert to same format as yt-dlp results
                                return [
                                    {
                                        'video_id': r['video_id'],
                                        'url': r.get('url', ''),
                                        'title': r.get('title', ''),
                                        'channel': r.get('channel', ''),
                                        'duration': r.get('duration', 0),
                                        'description': r.get('description', ''),
                                        'keyword': keyword,
                                        'view_count': r.get('view_count'),
                                        'subscriber_count': r.get('subscriber_count'),
                                    }
                                    for r in api_results
                                ]
                        else:
                            # Quota too low, log fallback
                            log_fallback_event(
                                reason="quota_exhausted",
                                query=search_query,
                                quota_used=api_client.quota_used,
                                quota_limit=api_client.quota_limit,
                            )

                except QuotaExceededError as e:
                    # Log fallback event
                    log_fallback_event(
                        reason="quota_exhausted",
                        query=search_query,
                    )
                    log_rate_limit(
                        logger, "quota", "youtube_api", "exhausted",
                        query=search_query
                    )
                    log_error_with_context(logger, "SEARCH-002", f"YouTube API quota exhausted for query '{search_query}', falling back to yt-dlp: {e}")

                except Exception as e:
                    # Log fallback for any other API error
                    log_fallback_event(
                        reason=f"api_error: {str(e)[:50]}",
                        query=search_query,
                    )
                    log_error_with_context(logger, "SEARCH-001", f"YouTube API error for query '{search_query}', falling back to yt-dlp: {e}")

        # yt-dlp search options (search only, no download)
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',  # Don't download, just get metadata
            'skip_download': True,
            'ignoreerrors': True,
        }

        # Apply impersonation if available
        try:
            from ..downloader.impersonation import ImpersonationManager
            imp_mgr = ImpersonationManager()
            imp_opts = imp_mgr.get_ydl_options(tier=1)
            ydl_opts.update(imp_opts)
        except Exception as e:
            log_error_with_context(logger, "SEARCH-001", f"Impersonation setup failed, using default: {e}")

        results = []
        search_url = f"ytsearch{max_results * 2}:{search_query}"  # Get extra to filter

        # Phase 1: Initial search
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(search_url, download=False)
                if not info or 'entries' not in info:
                    return []

                initial_results = []
                for entry in info.get('entries', []):
                    if not entry:
                        continue

                    video_id = entry.get('id', '')
                    if not video_id:
                        continue

                    # Filter by duration
                    duration = entry.get('duration', 0) or 0
                    if duration < min_duration or duration > max_duration:
                        continue

                    # Apply title blacklist filter
                    title = entry.get('title', '')
                    if self._is_title_blacklisted(title, config):
                        continue

                    # US-95-012: Apply negative context filtering
                    if use_negative_context and negative_keywords:
                        description = entry.get('description', '')
                        matched_keyword = self._get_matched_negative_keyword(title, description, negative_keywords)
                        if matched_keyword:
                            logger.debug(f"Negative keyword filtering: video '{title}' (ID: {video_id}) filtered - matched keyword: '{matched_keyword}' (checking {len(negative_keywords)} negative keywords)")
                            continue

                    initial_results.append({
                        'video_id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': title,
                        'channel': entry.get('channel', entry.get('uploader', '')),
                        'duration': duration,
                        'description': entry.get('description', ''),  # US-95-003: Capture description
                        'keyword': keyword,
                        'view_count': entry.get('view_count'),  # US-111-005: For channel reputation scoring
                        'subscriber_count': entry.get('channel_follower_count'),  # US-111-005: If available
                    })

            except Exception as e:
                log_error_with_context(logger, "SEARCH-003", f"yt-dlp search error for query '{search_query}': {e}")

        # US-95-003: Extract description keywords and refine search
        if use_description_context and initial_results:
            # Extract keywords from top descriptions
            desc_keywords = self._extract_description_keywords(initial_results)

            if desc_keywords:
                # Build refined query with description keywords (weighted lower)
                # Description keywords go at the end - they're optional refinements
                refined_query = f"{keyword} {' '.join(desc_keywords[:2])}"
                refined_url = f"ytsearch{max_results}:{refined_query}"

                logger.info(f"Description-refined query: '{keyword}' -> '{refined_query}'")

                try:
                    info = ydl.extract_info(refined_url, download=False)
                    if info and 'entries' in info:
                        for entry in info.get('entries', []):
                            if not entry:
                                continue

                            video_id = entry.get('id', '')
                            if not video_id:
                                continue

                            duration = entry.get('duration', 0) or 0
                            if duration < min_duration or duration > max_duration:
                                continue

                            title = entry.get('title', '')
                            if self._is_title_blacklisted(title, config):
                                continue

                            # US-95-012: Apply negative context filtering
                            if use_negative_context and negative_keywords:
                                description = entry.get('description', '')
                                matched_keyword = self._get_matched_negative_keyword(title, description, negative_keywords)
                                if matched_keyword:
                                    logger.debug(f"Negative keyword filtering (refined): video '{title}' (ID: {video_id}) filtered - matched keyword: '{matched_keyword}' (checking {len(negative_keywords)} negative keywords)")
                                    continue

                            # Add refined results (may include duplicates)
                            initial_results.append({
                                'video_id': video_id,
                                'url': f"https://www.youtube.com/watch?v={video_id}",
                                'title': title,
                                'channel': entry.get('channel', entry.get('uploader', '')),
                                'duration': duration,
                                'description': entry.get('description', ''),
                                'keyword': keyword,
                                'view_count': entry.get('view_count'),  # US-111-005: For channel reputation scoring
                                'subscriber_count': entry.get('channel_follower_count'),  # US-111-005: If available
                            })

                except Exception as e:
                    log_error_with_context(logger, "SEARCH-003", f"yt-dlp refined search error for query '{keyword}': {e}")

        # Deduplicate and limit results
        seen_ids = set()
        for r in initial_results:
            if r['video_id'] not in seen_ids and len(results) < max_results:
                seen_ids.add(r['video_id'])
                # Remove description from final results (not stored)
                result_clean = {k: v for k, v in r.items() if k != 'description'}
                results.append(result_clean)

        return results

    def _detect_content_type(self, keyword: str, topic: str = "") -> str:
        """
        US-153-011: Detect content type from keyword and topic.

        Detects if the search is for tutorial, review, vlog, or generic content
        based on keyword patterns.

        Args:
            keyword: The search keyword
            topic: Optional topic context

        Returns:
            Content type: 'tutorial', 'review', 'vlog', or 'generic'
        """
        text = f"{keyword} {topic}".lower()

        # Tutorial indicators
        tutorial_patterns = [
            'how to', 'how-to', 'tutorial', 'guide', 'learn', 'course',
            'teach', 'step by step', 'beginner', 'explained', 'tips',
            'instructions', 'lesson', 'training', 'workshop'
        ]

        # Review indicators
        review_patterns = [
            'review', 'vs ', 'versus', 'comparison', 'compared', 'best',
            'top ', 'ranking', 'rated', 'opinion', 'thoughts', 'unboxing',
            'honest', 'pros cons', 'pros and cons'
        ]

        # Vlog/lifestyle indicators
        vlog_patterns = [
            'vlog', 'day in', 'life', 'vlog', 'vlogger', 'vlogging',
            'routine', 'morning', 'evening', 'weekend', 'travel',
            'adventure', 'experience', 'journey', 'story'
        ]

        # Check for matches
        for pattern in tutorial_patterns:
            if pattern in text:
                logger.info(f"US-153-011: Detected content type 'tutorial' for keyword '{keyword}' (pattern: '{pattern}')")
                return 'tutorial'

        for pattern in review_patterns:
            if pattern in text:
                logger.info(f"US-153-011: Detected content type 'review' for keyword '{keyword}' (pattern: '{pattern}')")
                return 'review'

        for pattern in vlog_patterns:
            if pattern in text:
                logger.info(f"US-153-011: Detected content type 'vlog' for keyword '{keyword}' (pattern: '{pattern}')")
                return 'vlog'

        logger.debug(f"US-153-011: Using generic content type for keyword '{keyword}'")
        return 'generic'

    def _generate_query_variations(
        self,
        keyword: str,
        content_type: str,
        video_search_config: Any,
    ) -> List[str]:
        """
        US-153-011: Generate query variations based on content type.

        Creates multiple query variations for A/B testing and improved search results.

        Args:
            keyword: Base keyword
            content_type: Detected content type
            video_search_config: Video search config section

        Returns:
            List of query variations
        """
        variations = []

        # Handle both dict and object access patterns
        if isinstance(video_search_config, dict):
            enable_ab_testing = video_search_config.get('enable_ab_testing', False)
            ab_test_variant = video_search_config.get('ab_test_variant', 'control')
            max_variations = video_search_config.get('max_query_variations', 3)
            query_template = video_search_config.get('query_template', None)
        else:
            enable_ab_testing = getattr(video_search_config, 'enable_ab_testing', False)
            ab_test_variant = getattr(video_search_config, 'ab_test_variant', 'control')
            max_variations = getattr(video_search_config, 'max_query_variations', 3)
            query_template = getattr(video_search_config, 'query_template', None)

        # Inline templates if not in config
        inline_templates = {
            'tutorial': [
                '{keyword} tutorial',
                '{keyword} how to',
                '{keyword} guide for beginners',
            ],
            'review': [
                '{keyword} review',
                '{keyword} vs comparison',
                '{keyword} best options',
            ],
            'vlog': [
                '{keyword} vlog',
                '{keyword} adventure',
                '{keyword} travel',
            ],
            'generic': [
                '{keyword}',
                '{keyword} video',
                '{keyword} 2024',
            ],
        }

        # A/B testing variant templates
        ab_templates = {
            'control': {
                'tutorial': ['{keyword} tutorial', '{keyword} how to', '{keyword} guide'],
                'review': ['{keyword} review', '{keyword} vs', '{keyword} best'],
                'vlog': ['{keyword} vlog', '{keyword} adventure', '{keyword} travel'],
                'generic': ['{keyword}', '{keyword} video', '{keyword} 2024'],
            },
            'treatment': {
                'tutorial': ['{keyword} tutorial for beginners', 'best {keyword} guide', '{keyword} step by step'],
                'review': ['{keyword} honest review', '{keyword} comparison 2024', 'top {keyword} recommendations'],
                'vlog': ['{keyword} daily vlog', 'amazing {keyword} journey', '{keyword} adventure travel'],
                'generic': ['{keyword} high quality', 'popular {keyword} videos', '{keyword} trending'],
            },
        }

        # Check A/B testing first
        if enable_ab_testing:
            variant = ab_test_variant
            templates = ab_templates.get(variant, ab_templates['control']).get(content_type, ab_templates['control']['generic'])
            selected_templates = templates[:max_variations]
            for tmpl in selected_templates:
                variation = tmpl.format(keyword=keyword)
                if variation not in variations:
                    variations.append(variation)
            logger.info(
                f"US-153-011: Query template selection - "
                f"source=A/B_testing, variant='{variant}', content_type='{content_type}', "
                f"max_variations={max_variations}, selected_templates={selected_templates}, "
                f"generated_count={len(variations)}"
            )
            return variations

        # Determine template source and log selection decision
        template_source = "config"
        if query_template:
            if isinstance(query_template, dict):
                templates_by_type = {
                    'tutorial': query_template.get('tutorial_templates', []),
                    'review': query_template.get('review_templates', []),
                    'vlog': query_template.get('vlog_templates', []),
                    'generic': query_template.get('generic_templates', []),
                }
            else:
                templates_by_type = {
                    'tutorial': getattr(query_template, 'tutorial_templates', []),
                    'review': getattr(query_template, 'review_templates', []),
                    'vlog': getattr(query_template, 'vlog_templates', []),
                    'generic': getattr(query_template, 'generic_templates', []),
                }

            templates = templates_by_type.get(content_type, templates_by_type.get('generic', []))
        else:
            template_source = "inline"
            templates = inline_templates.get(content_type, inline_templates['generic'])

        selected_templates = templates[:max_variations]
        for tmpl in selected_templates:
            variation = tmpl.format(keyword=keyword)
            if variation not in variations:
                variations.append(variation)

        logger.info(
            f"US-153-011: Query template selection - "
            f"source={template_source}, content_type='{content_type}', "
            f"keyword='{keyword}', max_variations={max_variations}, "
            f"selected_templates={selected_templates}, generated_count={len(variations)}"
        )
        return variations

    def _optimize_query_length(
        self,
        query: str,
        config: Any,
    ) -> str:
        """
        US-153-011: Optimize query length based on specificity.

        Shorter queries for broad topics, longer queries for specific topics.

        Args:
            query: The current query
            config: Video search config

        Returns:
            Optimized query
        """
        words = query.split()
        word_count = len(words)

        short_max = getattr(config, 'short_query_length', 3)
        long_min = getattr(config, 'long_query_length', 8)

        # If query is too long, truncate to short length
        if word_count > short_max:
            optimized = ' '.join(words[:short_max])
            logger.info(f"US-153-011: Query shortened from {word_count} to {short_max} words: '{query}' -> '{optimized}'")
            query = optimized

        # US-157-004: Also enforce max character length
        query = self._truncate_query_length(query, config)

        return query

    # US-157-004: Query optimization methods

    def _preprocess_query(
        self,
        query: str,
        config: Any,
    ) -> str:
        """
        US-157-004: Preprocess query - remove stopwords, normalize whitespace,
        handle special characters.

        Args:
            query: The raw query string
            config: Video search config

        Returns:
            Preprocessed query string
        """
        if not query:
            return query

        # Get stopwords from config
        enable_stopwords = getattr(config, 'enable_stopword_removal', True)
        stopwords = getattr(config, 'stopword_list', []) or []

        # Normalize whitespace
        processed = ' '.join(query.split())

        # Remove special characters but keep alphanumeric and spaces
        processed = re.sub(r'[^\w\s]', ' ', processed)

        # Normalize whitespace again after special char removal
        processed = ' '.join(processed.split())

        # Remove stopwords if enabled
        removed_stopwords = []
        if enable_stopwords and stopwords:
            words = processed.lower().split()
            filtered_words = [w for w in words if w not in stopwords]
            removed_stopwords = [w for w in words if w in stopwords]
            processed = ' '.join(filtered_words) if filtered_words else processed

        logger.debug(
            f"US-157-004: Query preprocessing steps: original='{query}', "
            f"whitespace_normalized, special_chars_removed, "
            f"stopwords_removed={len(removed_stopwords)} ({removed_stopwords[:5] if removed_stopwords else []}), "
            f"final='{processed}'"
        )
        return processed

    def _expand_query(
        self,
        query: str,
        config: Any,
        topic: str = None,
    ) -> str:
        """
        US-157-004: Expand query with related terms based on keywords.

        Args:
            query: The base query
            config: Video search config
            topic: Optional topic context for expansion

        Returns:
            Expanded query string
        """
        if not query:
            return query

        enable_expansion = getattr(config, 'enable_query_expansion', True)
        if not enable_expansion:
            return query

        # Get related terms from topic_tags
        topic_tags_map = getattr(config, 'topic_tags', {}) or {}
        query_lower = query.lower()

        # Find matching topics
        expanded_terms = []
        for topic_key, tags in topic_tags_map.items():
            if topic_key in query_lower:
                expanded_terms.extend(tags)

        # Also check topic parameter
        if topic:
            topic_lower = topic.lower()
            for topic_key, tags in topic_tags_map.items():
                if topic_key in topic_lower and tags not in expanded_terms:
                    expanded_terms.extend(tags)

        # Add unique expansion terms
        if expanded_terms:
            seen = set()
            unique_terms = []
            for term in expanded_terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)

            # Limit to 2 additional terms to avoid overly broad queries
            expanded_query = f"{query} {' '.join(unique_terms[:2])}"
            logger.debug(
                f"US-157-004: Query expansion: original='{query}', topic='{topic}', "
                f"matched_topics={[k for k in topic_tags_map.keys() if k in (query.lower() or '') or (topic and k in topic.lower())]}, "
                f"expansion_terms={unique_terms[:2]}, final='{expanded_query}'"
            )
            return expanded_query

        return query

    def _truncate_query_length(
        self,
        query: str,
        config: Any,
    ) -> str:
        """
        US-157-004: Truncate query to max_query_length characters.

        Args:
            query: The query string
            config: Video search config

        Returns:
            Query truncated to max_query_length
        """
        max_length = getattr(config, 'max_query_length', 256)

        if len(query) <= max_length:
            return query

        truncated = query[:max_length]
        # Try to end at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If we can cut at >80%, do so
            truncated = truncated[:last_space]

        logger.debug(f"US-157-004: Query truncated from {len(query)} to {len(truncated)} chars")
        return truncated

    def _calculate_relevance_score(
        self,
        title: str,
        description: str,
        query: str,
    ) -> float:
        """
        US-157-004: Calculate relevance score for a search result based on
        title and description matching with the query.

        Args:
            title: Video title
            description: Video description
            query: The search query

        Returns:
            Relevance score between 0 and 1
        """
        if not query or not title:
            return 0.0

        query_lower = query.lower()
        title_lower = title.lower()
        desc_lower = (description or "").lower()

        # Calculate word overlap
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        desc_words = set(desc_lower.split())

        # Title match is weighted more heavily
        title_overlap = len(query_words & title_words) / len(query_words) if query_words else 0
        desc_overlap = len(query_words & desc_words) / len(query_words) if query_words else 0

        # Combined score: 70% title, 30% description
        relevance = (0.7 * title_overlap) + (0.3 * desc_overlap)

        # Bonus for exact phrase match in title
        if query_lower in title_lower:
            relevance = min(1.0, relevance + 0.2)

        return min(1.0, relevance)

    def _score_results_by_relevance(
        self,
        results: List[Dict[str, Any]],
        query: str,
        config: Any,
    ) -> List[Dict[str, Any]]:
        """
        US-157-004: Score and filter search results by relevance.

        Args:
            results: List of search results
            query: The search query
            config: Video search config

        Returns:
            Filtered and scored results
        """
        enable_scoring = getattr(config, 'enable_relevance_scoring', True)
        min_score = getattr(config, 'min_relevance_score', 0.3)
        boost_factor = getattr(config, 'relevance_boost_factor', 0.1)

        if not enable_scoring or not results:
            return results

        scored_results = []
        for result in results:
            title = result.get('title', '')
            description = result.get('description', '')

            relevance = self._calculate_relevance_score(title, description, query)

            # Apply relevance as a boost factor to ranking/priority if applicable
            if 'relevance_score' not in result:
                result['relevance_score'] = relevance

            # Filter out low relevance results
            if relevance >= min_score:
                scored_results.append(result)
            else:
                logger.debug(f"US-157-004: Filtered low relevance result: '{title[:50]}...' (score: {relevance:.2f})")

        logger.debug(f"US-157-004: Relevance filtering: {len(results)} -> {len(scored_results)} results")
        return scored_results

    def _build_search_query(
        self,
        keyword: str,
        topic: str,
        use_tags: bool,
        topic_tags_map: dict
    ) -> str:
        """Build search query with tag expansion"""
        search_query = keyword

        # US-95-002: Expand query with related topic tags
        if use_tags and topic_tags_map:
            expanded_terms = []
            keyword_lower = keyword.lower()

            for topic_key, tags in topic_tags_map.items():
                if topic_key in keyword_lower:
                    expanded_terms.extend(tags)

            if topic:
                topic_lower = topic.lower()
                for topic_key, tags in topic_tags_map.items():
                    if topic_key in topic_lower and tags not in expanded_terms:
                        expanded_terms.extend(tags)

            if expanded_terms:
                seen = set()
                unique_terms = []
                for term in expanded_terms:
                    if term not in seen:
                        seen.add(term)
                        unique_terms.append(term)

                search_query = f"{keyword} {' '.join(unique_terms[:3])}"
                logger.info(f"Tag-expanded query: '{keyword}' -> '{search_query}'")

        if topic and topic not in search_query:
            search_query = f"{search_query} {topic}"

        return search_query

    def _build_chapter_queries(
        self,
        location_chapters: List[Any],
        topic_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        US-98-005: Build search queries from voiceover chapter topics.

        Each voiceover chapter's topics become separate search terms, tagged
        with the source chapter for downstream scoring.

        Args:
            location_chapters: List of ChapterCandidate or dict objects with topics
            topic_context: Overall topic for context

        Returns:
            List of dicts with 'keyword', 'chapter_id', 'chapter_title'
        """
        chapter_queries = []

        for chapter in location_chapters:
            # Extract chapter info - handle both dict and object access
            if isinstance(chapter, dict):
                chapter_id = chapter.get('chapter_id', 0)
                chapter_title = chapter.get('title', f'Chapter {chapter_id}')
                topics = chapter.get('topics', [])
            else:
                chapter_id = getattr(chapter, 'chapter_id', 0)
                chapter_title = getattr(chapter, 'title', f'Chapter {chapter_id}')
                topics = getattr(chapter, 'topics', [])

            # Skip chapters without topics
            if not topics:
                continue

            # Build search query from chapter topics
            # Use first 3 topics as search terms
            search_topics = topics[:3] if len(topics) > 3 else topics

            # Add topic context if available and not already included
            if topic_context:
                topic_lower = topic_context.lower()
                if not any(topic_lower in t.lower() for t in search_topics):
                    search_topics = search_topics + [topic_context]

            # Create keyword from topics
            keyword = ' '.join(search_topics)

            if keyword:
                chapter_queries.append({
                    'keyword': keyword,
                    'chapter_id': chapter_id,
                    'chapter_title': chapter_title,
                    'topics': search_topics,
                })

        return chapter_queries

    def _build_listicle_queries(
        self,
        listicle_groups: List[Any],
        topic_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        US-98-008: Build search queries from listicle group topic keywords.

        Each listicle item's topic_keywords become separate search terms, tagged
        with the source listicle group for downstream scoring.

        Args:
            listicle_groups: List of ListicleGroup objects with topic_keywords
            topic_context: Overall topic for context

        Returns:
            List of dicts with 'keyword', 'group_id', 'item_label', 'topics'
        """
        listicle_queries = []

        for group in listicle_groups:
            # Extract listicle group info - handle both dict and object access
            if isinstance(group, dict):
                group_id = group.get('group_id', 0)
                item_label = group.get('item_label', f'Item {group_id}')
                topics = group.get('topic_keywords', [])
            else:
                group_id = getattr(group, 'group_id', 0)
                item_label = getattr(group, 'item_label', f'Item {group_id}')
                topics = getattr(group, 'topic_keywords', [])

            # Skip groups without topic keywords
            if not topics:
                continue

            # Build search query from topic keywords
            # Use first 3 topics as search terms for specificity
            search_topics = topics[:3] if len(topics) > 3 else topics

            # Add topic context if available and not already included
            if topic_context:
                topic_lower = topic_context.lower()
                if not any(topic_lower in t.lower() for t in search_topics):
                    search_topics = search_topics + [topic_context]

            # Create keyword from topics, prefixed with item_label for specificity
            # Format: '{item_label} {topic_keyword_1} {topic_keyword_2}' e.g., 'first tip productivity workflow'
            keyword = f"{item_label} {' '.join(search_topics)}"

            if keyword:
                listicle_queries.append({
                    'keyword': keyword,
                    'group_id': group_id,
                    'item_label': item_label,
                    'topics': search_topics,
                })

        return listicle_queries

    def _extract_description_keywords(self, results: List[Dict]) -> List[str]:
        """US-95-003: Extract key terms from video descriptions"""
        if not results:
            return []

        # Collect descriptions from top results
        descriptions = []
        for r in results[:10]:  # Use top 10 results
            desc = r.get('description', '')
            if desc:
                descriptions.append(desc)

        if not descriptions:
            return []

        # Simple keyword extraction: split on whitespace and filter
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
            'very', 'just', 'http', 'https', 'www', 'com', 'watch', 'video', 'like',
            'subscribe', 'channel', 'youtube', 'music', 'song', 'official', 'new'
        }

        word_counts: Dict[str, int] = {}

        for desc in descriptions:
            # Clean: lowercase, remove URLs, special chars
            import re
            cleaned = re.sub(r'http\S+', '', desc)
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            cleaned = cleaned.lower()

            words = cleaned.split()
            for word in words:
                if len(word) >= 3 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Return top keywords that appear at least twice
        keywords = [w for w, count in sorted_words if count >= 2][:4]

        return keywords

    def _is_title_blacklisted(self, title: str, config: 'Config') -> bool:
        """Check if video title matches blacklist patterns"""
        if not title:
            return False

        title_lower = title.lower()

        # Get blacklist from config
        download_config = config.download
        blacklist = getattr(download_config, 'title_blacklist', [])

        if not blacklist:
            # Default blacklist patterns
            blacklist = [
                'reaction', 'review', 'gameplay', 'let\'s play',
                'unboxing', 'tutorial', 'how to', 'vlog',
                'podcast', 'interview', 'documentary'
            ]

        for pattern in blacklist:
            if pattern.lower() in title_lower:
                return True

        return False

    def _is_negative_matched(
        self,
        title: str,
        description: str,
        negative_keywords: List[str]
    ) -> bool:
        """US-95-012: Check if title or description matches negative keywords"""
        if not title or not negative_keywords:
            return False

        title_lower = title.lower()
        desc_lower = (description or "").lower()

        for keyword in negative_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower or keyword_lower in desc_lower:
                return True

        return False

    def _get_matched_negative_keyword(
        self,
        title: str,
        description: str,
        negative_keywords: List[str]
    ) -> Optional[str]:
        """US-95-012: Check if title or description matches negative keywords and return the matched keyword.

        Args:
            title: Video title
            description: Video description
            negative_keywords: List of negative keywords to check against

        Returns:
            The matched negative keyword, or None if no match
        """
        if not title or not negative_keywords:
            return None

        title_lower = title.lower()
        desc_lower = (description or "").lower()

        for keyword in negative_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower or keyword_lower in desc_lower:
                return keyword

        return None

    def _calculate_channel_quality_score(
        self,
        subscriber_count: int,
        video_count: int,
        view_count: int
    ) -> float:
        """US-146-006: Calculate channel quality score based on subscriber count and activity.

        Score is normalized 0.0-1.0 based on:
        - Subscriber count (primary factor)
        - Video count (activity indicator)
        - View count (engagement indicator)

        Returns:
            Quality score between 0.0 and 1.0
        """
        if subscriber_count == 0:
            return 0.0

        # Logarithmic scale for subscribers (more gradual)
        import math
        subscriber_score = min(1.0, math.log10(subscriber_count + 1) / 6.0)  # 1M = ~1.0

        # Video count score (more videos = more active channel)
        video_score = min(1.0, math.log10(video_count + 1) / 4.0)  # 1K videos = ~1.0

        # View count score (higher views = more popular)
        view_score = min(1.0, math.log10(view_count + 1) / 8.0)  # 100M = ~1.0

        # Weighted average: subscribers 50%, videos 20%, views 30%
        return (subscriber_score * 0.5) + (video_score * 0.2) + (view_score * 0.3)

    def _calculate_engagement_score(
        self,
        view_count: int,
        like_count: int,
        comment_count: int,
        view_count_weight: float = 0.5,
        like_count_weight: float = 0.3,
        comment_count_weight: float = 0.2,
    ) -> float:
        """Calculate engagement score for ranking (US-155-006).

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

    def _extract_negative_keywords(self, title: str, description: str, negative_keywords: List[str]) -> List[str]:
        """US-95-012: Extract negative keywords found in title/description"""
        if not negative_keywords:
            return []

        title_lower = title.lower()
        desc_lower = (description or "").lower()
        found = []

        for keyword in negative_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower or keyword_lower in desc_lower:
                found.append(keyword)

        return found

    def _to_search_results(
        self,
        results: List[Dict],
        negative_keywords: List[str] = None
    ) -> List['VideoSearchResult']:
        """Convert dict results to VideoSearchResult objects"""
        from ..state import VideoSearchResult

        negative_keywords = negative_keywords or []

        return [
            VideoSearchResult(
                video_id=r['video_id'],
                url=r.get('url', ''),
                title=r.get('title', ''),
                channel=r.get('channel', ''),
                duration=r.get('duration', 0.0),
                keyword=r.get('keyword', ''),
                description=r.get('description', ''),  # US-70-002
                negative_keywords=self._extract_negative_keywords(
                    r.get('title', ''),
                    r.get('description', ''),
                    negative_keywords
                ),  # US-95-012
                chapter_id=r.get('chapter_id', -1),  # US-98-005
                chapter_title=r.get('chapter_title', ''),  # US-98-005
                listicle_group_id=r.get('listicle_group_id', -1),  # US-98-008
                listicle_item_label=r.get('listicle_item_label', ''),  # US-98-008
                video_tags=r.get('api_tags', r.get('video_tags', [])),  # US-146-008: API tags preferred
                topic_details=r.get('topic_details', {}),  # US-146-008: Topic categories from API
                topic_categories=r.get('topic_categories', []),  # US-150-006: Dedicated topic_categories field
                # US-146-006: Channel metadata from YouTube Data API
                subscriber_count=r.get('subscriber_count', 0),
                channel_total_views=r.get('channel_total_views', 0),
                channel_created_date=r.get('channel_created_date', ''),
                channel_quality_score=r.get('channel_quality_score', 0.0),
                # US-148-008: Engagement metrics from YouTube Data API
                view_count=r.get('view_count', 0),
                like_count=r.get('like_count', 0),
                comment_count=r.get('comment_count', 0),
                engagement_score=r.get('engagement_score', 0.0),
            )
            for r in results
        ]

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if video search stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore video search stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                logger.warning(f"VIDEO_SEARCH restore: expected dict, got {type(data).__name__}")
                return False

            required_keys = {'video_ids'}
            missing = required_keys - set(data.keys())
            if missing:
                logger.warning(f"VIDEO_SEARCH restore: missing required keys: {missing}")
                return False

            if not isinstance(data['video_ids'], list):
                logger.warning(f"VIDEO_SEARCH restore: 'video_ids' expected list, got {type(data['video_ids']).__name__}")
                return False

            # Restore video IDs
            state.video_ids = data.get('video_ids', [])

            # Restore search results
            search_results = data.get('search_results', [])
            state.video_search_results = self._to_search_results(search_results)

            # Restore failed keywords
            state.search_failed_keywords = data.get('failed_keywords', [])

            logger.info(f"Restored VIDEO_SEARCH: {len(state.video_ids)} video IDs")
            return True

        except Exception as e:
            log_error_with_context(logger, "PIPE-002", f"Failed to restore VIDEO_SEARCH checkpoint: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        if not state.keywords:
            return "No keywords available for video search"
        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        return {
            'inputs': 'keywords',
            'outputs': 'video candidates',
            'input_count': len(state.keywords) if state.keywords else 0,
            'output_count': len(state.video_candidates) if state.video_candidates else None,
        }

    def get_api_estimates(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get API call estimates for dry-run preview"""
        keyword_count = len(state.keywords) if state.keywords else 0

        # Get video search config for max_results
        video_search_config = getattr(config, 'video_search', None)
        max_results = 50
        if video_search_config:
            max_results = getattr(video_search_config, 'max_results', 50)

        # Estimate YouTube API calls: 1 per keyword
        youtube_api_calls = keyword_count

        # Estimate cost: YouTube Data API v3 is $0.002/1000 quota units
        # Search endpoint costs 100 units per request
        # Each keyword search = 100 quota units
        quota_cost = keyword_count * 100 / 1000 * 0.002

        # Estimate duration: ~0.5s per search API call
        estimated_duration = keyword_count * 0.5

        return {
            'youtube_api_calls': youtube_api_calls,
            'estimated_cost_usd': round(quota_cost, 4),
            'estimated_duration_seconds': round(estimated_duration, 1),
        }
