"""
Caption Stage - Fetch YouTube Captions Before Transcription

Stage that runs AFTER VIDEO_SEARCH and BEFORE MATCH:
- Fetches YouTube captions for all video candidates
- Stores caption data in state.text_metadata for matching
- Supports checkpoint/resume from partial completion
- Falls back to TRANSCRIBE stage if captions unavailable

Caption fetching is always enabled - this is the default behavior for
faster matching with lower bandwidth. Videos without captions will be
handled by the TRANSCRIBE stage as a fallback.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageMetrics, StageResult, register_stage, validate_required_state_attrs
from ..logging_templates import (
    log_stage_start,
    log_stage_complete,
    log_stage_skip,
    log_progress,
    log_error_with_context,
)

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class CaptionStage(Stage):
    """
    Fetches YouTube captions for video candidates.

    In the simplified 7-stage pipeline, this stage receives video IDs from
    the VIDEO_SEARCH stage (not downloaded files).

    Inputs:
        - state.video_ids: List of YouTube video IDs from VIDEO_SEARCH stage

    Outputs:
        - state.caption_results: Dict mapping video_id to caption data
        - (For backward compatibility, also updates state.text_metadata)

    Language Configuration Validation (US-005 Sprint 6):
        At initialization, validates that configured language codes are valid
        ISO 639-1 codes, no duplicates exist in fallback_languages, and
        preferred_language is not redundantly in fallback_languages.
    """

    name = "CAPTION"
    description = "Fetch YouTube captions for video candidates"
    # US-108-012: Changed from ['VIDEO_SEARCH'] to ['ANALYZE'] to allow parallel
    # execution with VIDEO_SEARCH. Both stages depend on ANALYZE and can run
    # concurrently. CAPTION will wait for VIDEO_SEARCH to produce video_ids
    # internally (via validate_inputs which checks for existing video_ids).
    DEPENDS_ON = ['ANALYZE']
    PRODUCES = ['caption_results', 'text_metadata']

    def __init__(self, config: 'Config' = None):
        """Initialize CaptionStage.

        Args:
            config: Optional Config object for language validation (US-005).
                   If provided, validates language configuration at startup.
                   If not provided, validation runs during run() instead.

        Raises:
            ConfigValidationError: If language configuration is invalid.
        """
        self._fetcher = None
        self._config_validated = False

        # US-60-006: Track videos that need transcription fallback
        self.needs_transcription: List[str] = []

        # US-005: Validate language config at init if config provided
        if config is not None:
            self._validate_language_config(config)

    def _ensure_state_attributes(self, state: 'PipelineState') -> None:
        """Ensure state has required attributes with defensive initialization.

        US-37-010: Provides belt-and-suspenders protection for dynamically-added
        attributes, ensuring compatibility with both new PipelineState instances
        and legacy pipeline objects that may not have all attributes defined.

        US-39-003: Uses explicit setattr() for compatibility with object() and
        SimpleNamespace instances that may not support direct attribute assignment
        in all Python versions/contexts.

        This follows the pattern established in PipelineState.from_legacy_pipeline()
        where attributes are copied with getattr() defaults.

        Note: This method is called by _preflight_check() which provides consolidated logging.
        """
        # Ensure text_metadata exists (required by _populate_text_metadata)
        if not hasattr(state, 'text_metadata'):
            setattr(state, 'text_metadata', [])

        # Ensure caption_results exists (stores caption data by video_id)
        if not hasattr(state, 'caption_results'):
            setattr(state, 'caption_results', {})

        # Ensure video_ids exists (input from VIDEO_SEARCH stage)
        if not hasattr(state, 'video_ids'):
            setattr(state, 'video_ids', [])

        # Ensure video_search_results exists (full search metadata)
        if not hasattr(state, 'video_search_results'):
            setattr(state, 'video_search_results', [])

    def _preflight_check(self, state: 'PipelineState') -> None:
        """Validate state has all required attributes before processing.

        US-39-011: Consolidates defensive checks into single method for clarity
        and testability. Called at stage entry before main processing loop.

        Required attributes:
        - video_ids (list): Input from VIDEO_SEARCH stage
        - caption_results (dict): Stores caption data by video_id
        - text_metadata (list): Required by _populate_text_metadata

        Logs INFO listing which attributes were initialized if any were missing.

        Args:
            state: PipelineState object to validate.
        """
        # Track which attributes need initialization
        required_attrs = {
            'video_ids': [],
            'caption_results': {},
            'text_metadata': [],
        }

        initialized = []
        for attr, default in required_attrs.items():
            if not hasattr(state, attr):
                setattr(state, attr, default)
                initialized.append(attr)

        # Log consolidated message if any attributes were initialized
        if initialized:
            logger.info(f"Pre-flight: initialized {len(initialized)} missing attributes: {', '.join(initialized)}")

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the caption fetch stage.

        US-001: Uses parallel batch fetching with ThreadPoolExecutor for faster
        processing of projects with 50+ videos.

        US-005: Validates language configuration before first fetch if not
        already validated at __init__.

        US-37-010: Ensures state has required attributes before processing.

        US-39-009: Validates state type and converts legacy objects if needed.

        US-40-004: Logs retry budget summary at stage completion (success or failure).
        """
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        warnings = []
        retry_budget = None  # US-40-004: Initialize early for access in except block

        # US-60-006: Reset transcription fallback list for this run
        self.needs_transcription = []

        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['video_ids'], self.name)

        # US-43-005: Defensive text_metadata initialization at very start of run()
        # This ensures text_metadata exists even if _validate_state_type or
        # _preflight_check fail, preventing AttributeError downstream
        if not hasattr(state, 'text_metadata') or state.text_metadata is None:
            state.text_metadata = []
            logger.warning('[US-43-005] Defensive init: text_metadata was missing/None')

        try:
            # US-39-009: Validate state type at stage entry
            state = self._validate_state_type(state)

            # US-39-011: Pre-flight check validates state completeness before processing
            self._preflight_check(state)

            # Get caption config (caption fetching is always enabled)
            caption_config = getattr(config.download, 'caption_first', None)

            # US-005: Validate language config if not done at init
            if not self._config_validated:
                self._validate_language_config(config)

            # Get video IDs to fetch captions for
            video_ids = self._get_video_ids(state, config)

            if not video_ids:
                log_stage_skip(logger, "CAPTION", reason="no video IDs available")
                warnings.append("No video IDs available for caption fetch")
                return StageResult.ok({'skipped': True, 'reason': 'no_videos'}, warnings)

            # Apply test mode video limit if enabled
            test_mode_max_videos = getattr(config, '_test_mode_max_videos', None)
            if test_mode_max_videos is not None and isinstance(test_mode_max_videos, int) and len(video_ids) > test_mode_max_videos:
                original_count = len(video_ids)
                video_ids = video_ids[:test_mode_max_videos]
                logger.info(f"Test mode: limited video_ids from {original_count} to {test_mode_max_videos}")
                logger.info(f"  [Test mode] Limited {original_count} videos to {test_mode_max_videos}")

            # Get preferred language from config (needed for stage start logging)
            preferred_lang = getattr(caption_config, 'preferred_language', 'en')

            log_stage_start(logger, "CAPTION", video_count=len(video_ids), language_preference=preferred_lang)

            # US-137-004: Predictive cache warming - warm transcription cache before caption fetch
            # This checks if videos already have transcriptions in global cache
            predictive_warming_enabled = getattr(config.transcription, 'predictive_cache_warming', True)
            if predictive_warming_enabled:
                try:
                    from ..transcription.cache import TranscriptCache
                    cache_dir = getattr(config.transcription, 'cache_dir', 'transcriptions')
                    # Use project cache dir if available
                    if hasattr(state, 'project_dir') and state.project_dir:
                        cache_path = Path(state.project_dir) / '.cache' / cache_dir
                    else:
                        cache_path = Path('.cache') / cache_dir

                    transcript_cache = TranscriptCache(str(cache_path))
                    warmup_result = transcript_cache.predict_cache_warm(video_ids)

                    if warmup_result['transcript_warmed'] > 0:
                        logger.info(f"  [US-137-004] Predictive warming: {warmup_result['transcript_warmed']} transcripts pre-loaded")
                    if warmup_result['videos_found'] > 0:
                        logger.info(f"  [US-137-004] Prefetch: {warmup_result['videos_found']} videos found in global cache")
                except Exception as e:
                    logger.debug(f"Predictive cache warming failed: {e}")

            # Get preferred language from config
            preferred_lang = getattr(caption_config, 'preferred_language', 'en')
            prefer_manual = getattr(caption_config, 'prefer_human_captions', True)
            timeout = getattr(caption_config, 'timeout', 30)
            base_max_workers = getattr(caption_config, 'max_parallel_fetches', 4)

            # US-100-006: Calculate adaptive worker count based on strategy and batch size
            worker_count_strategy = getattr(caption_config, 'worker_count_strategy', 'static')
            min_workers = getattr(caption_config, 'min_workers', 4)
            max_workers_limit = getattr(caption_config, 'max_workers_limit', 8)
            batch_size_threshold = getattr(caption_config, 'batch_size_threshold', 100)

            # Get batch size for adaptive calculation
            batch_size = len(video_ids)

            # Calculate worker count based on strategy
            if worker_count_strategy == 'static':
                max_workers = base_max_workers
            elif worker_count_strategy == 'cpu_count':
                import os
                cpu_count = os.cpu_count() or 4
                max_workers = min(cpu_count, base_max_workers)
            elif worker_count_strategy == 'adaptive':
                # Adaptive scaling: start with min_workers, scale up to max_workers_limit for larger batches
                if batch_size <= batch_size_threshold:
                    # Linear interpolation from min_workers to max_workers_limit
                    ratio = batch_size / batch_size_threshold if batch_size_threshold > 0 else 0
                    max_workers = int(min_workers + ratio * (max_workers_limit - min_workers))
                else:
                    # For larger batches, use max_workers_limit
                    max_workers = max_workers_limit
                # Cap at max_workers_limit
                max_workers = min(max_workers, max_workers_limit)
            else:
                max_workers = base_max_workers

            logger.info(
                f"Caption worker count: strategy={worker_count_strategy}, "
                f"calculated_count={max_workers}, batch_size={batch_size}"
            )

            # US-004: Get coverage threshold from config
            min_coverage_threshold = getattr(caption_config, 'min_coverage_threshold', 0.5)

            logger.info(f"  Caption settings: language={preferred_lang}, prefer_manual={prefer_manual}, "
                        f"parallel_workers={max_workers}, min_coverage={min_coverage_threshold:.0%}")

            # Initialize caption fetcher
            from ..caption_fetcher import (
                CaptionFetcher,
                CaptionUnavailableError,
                CaptionFetchError,
                CaptionResult,
                CaptionMetrics,
                CaptionCache,
                CaptionBatchCheckpoint,
                ErrorPatternAbortError,
                determine_caption_quality,
            )
            from ..caption.enums import CaptionStatus

            # Create impersonation manager if enabled (US-005 Sprint 9)
            impersonation_mgr = None
            impersonation_config = getattr(config.download, 'impersonation', None)
            if impersonation_config and getattr(impersonation_config, 'enabled', False):
                from ..downloader.impersonation import ImpersonationManager
                preferred = getattr(impersonation_config, 'preferred_targets', []) or []
                detect_startup = getattr(impersonation_config, 'detect_at_startup', True)
                det_timeout = getattr(impersonation_config, 'detection_timeout', 10)
                impersonation_mgr = ImpersonationManager(
                    preferred_targets=preferred,
                    detect_at_startup=detect_startup,
                    detection_timeout=det_timeout,
                )

            # Create escalation manager if impersonation is available (US-007 Sprint 9)
            escalation_mgr = None
            if impersonation_mgr:
                extractor_args_config = getattr(config.download, 'extractor_args', None)
                from ..downloader.escalation_manager import EscalationManager
                escalation_mgr = EscalationManager(
                    impersonation_manager=impersonation_mgr,
                    extractor_args_config=extractor_args_config,
                )

            # US-146-007: Create YouTube API client for caption availability check
            youtube_api_client = None
            youtube_api_config = getattr(config.download, 'youtube_api', None)
            if youtube_api_config:
                api_enabled = getattr(youtube_api_config, 'enabled', False)
                api_key = getattr(youtube_api_config, 'api_key', '')
                if api_enabled and api_key:
                    from ..downloader.youtube_api_client import YouTubeAPIClient
                    youtube_api_client = YouTubeAPIClient(
                        api_key=api_key,
                        quota_limit=getattr(youtube_api_config, 'quota_limit', 10000),
                        warn_at_percent=getattr(youtube_api_config, 'warn_at_percent', 80),
                        quota_fallback_threshold_percent=getattr(youtube_api_config, 'quota_fallback_threshold_percent', 10),
                        # US-155-003: Predictive quota fallback config
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
                        rotation_strategy=getattr(youtube_api_config, 'rotation_strategy', 'sequential'),
                        max_concurrent_requests=getattr(youtube_api_config, 'max_concurrent_requests', 5),
                        # US-155-003: Caption language preference and filtering
                        preferred_caption_language=getattr(youtube_api_config, 'preferred_caption_language', 'en'),
                        caption_language_fallback=getattr(youtube_api_config, 'caption_language_fallback', True),
                        caption_language_metrics=getattr(youtube_api_config, 'track_caption_language_metrics', True),
                        # US-155-007: Quota alert webhook notifications
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
                        # US-156-005: Search query sanitization and deduplication
                        deduplicate_searches=getattr(youtube_api_config, 'deduplicate_searches', True),
                        # US-158-004: Region code for localized search results
                        region_code=getattr(youtube_api_config, 'region_code', 'US'),
                        # US-158-005: Safe search level for family-friendly results
                        safe_search=getattr(youtube_api_config, 'safe_search', 'moderate'),
                        # US-158-006: Batch caption fetching size
                        caption_batch_size=getattr(youtube_api_config, 'caption_batch_size', 10),
                    )
                    logger.info("YouTube API client enabled for caption availability check")

            # US-002 Sprint 7: Initialize caption cache early so it can be passed to fetcher
            caption_cache = CaptionCache(caption_config)

            self._fetcher = CaptionFetcher(
                config=config,
                impersonation_manager=impersonation_mgr,
                escalation_manager=escalation_mgr,
                caption_cache=caption_cache,
                youtube_api_client=youtube_api_client,
            )
            self._fetcher._timeout = timeout

            # Initialize metrics tracker (US-011, US-001: thread-safe)
            metrics = CaptionMetrics()
            # US-100-006: Record worker count used for this batch
            metrics.worker_count = max_workers
            metrics.worker_count_strategy = worker_count_strategy

            # US-33-009: Initialize circuit breaker for consecutive failure protection
            from ..caption.circuit_breaker import (
                CaptionCircuitBreaker,
                CaptionCircuitBreakerConfig,
            )
            circuit_breaker = None
            cb_config = getattr(caption_config, 'circuit_breaker', None)
            if cb_config:
                # Convert dict to config if needed
                if isinstance(cb_config, dict):
                    cb_config = CaptionCircuitBreakerConfig(**cb_config)
                if getattr(cb_config, 'enabled', True):
                    circuit_breaker = CaptionCircuitBreaker(cb_config)
                    logger.info(
                        f"Caption circuit breaker enabled: threshold={cb_config.threshold}, "
                        f"pause_seconds={cb_config.pause_seconds}"
                    )

            # US-33-010: Initialize retry budget for cross-video resource tracking
            from ..caption.retry_budget import (
                CaptionRetryBudget,
                CaptionRetryBudgetConfig,
            )
            rb_config = getattr(caption_config, 'retry_budget', None)
            if rb_config:
                # Convert dict to config if needed
                if isinstance(rb_config, dict):
                    rb_config = CaptionRetryBudgetConfig(**rb_config)
                if getattr(rb_config, 'enabled', True):
                    retry_budget = CaptionRetryBudget.from_config(rb_config)
                    logger.info(
                        f"Caption retry budget enabled: max_attempts={retry_budget.max_attempts}, "
                        f"max_backoff_time={retry_budget.max_backoff_time}s"
                    )
            else:
                # US-38-002: Fallback initialization when config missing
                # Create default budget with auto_scale=True to prevent exhaustion on large batches
                retry_budget = CaptionRetryBudget.from_config(None)
                logger.info("Using default retry budget (config missing)")

            # US-62-006: Link circuit breaker to retry budget for observability
            # This allows retry budget summary to include circuit breaker state
            if retry_budget and circuit_breaker:
                retry_budget.circuit_breaker = circuit_breaker
                logger.debug("Circuit breaker linked to retry budget for observability")

            # US-40-002: Restore retry budget state from checkpoint if available
            # When resuming from checkpoint, the budget limits may have been scaled for
            # a previous batch size. After restoring, we'll re-scale for the current batch.
            # US-41-012: Store checkpoint data for later integrity check when batch_size is known
            checkpoint_retry_budget_data = None
            if retry_budget:
                # US-41-008: Log retry budget state BEFORE checkpoint restore
                logger.info(
                    f"[US-41-008] Retry budget before restore: max_attempts={retry_budget.max_attempts}, "
                    f"batch_size={retry_budget.batch_size}"
                )
                try:
                    checkpoint_data = checkpoint.get_stage_data(self.name)
                    if checkpoint_data and 'retry_budget' in checkpoint_data:
                        # US-41-012: Store for later integrity check
                        checkpoint_retry_budget_data = checkpoint_data['retry_budget']
                        restored_budget = CaptionRetryBudget.from_dict(
                            checkpoint_retry_budget_data
                        )
                        # Preserve usage counters from checkpoint
                        retry_budget.attempts = restored_budget.attempts
                        retry_budget.failures = restored_budget.failures
                        retry_budget.successes = restored_budget.successes
                        retry_budget.backoff_time_spent = restored_budget.backoff_time_spent
                        retry_budget.videos_skipped = restored_budget.videos_skipped
                        retry_budget.error_counts = restored_budget.error_counts
                        retry_budget.vpn_resets_used = restored_budget.vpn_resets_used
                        retry_budget.early_terminated = restored_budget.early_terminated
                        retry_budget.early_termination_reason = restored_budget.early_termination_reason
                        retry_budget.batch_size = restored_budget.batch_size
                        # Keep max_attempts from config (will be re-scaled below)

                        # US-41-008: Log retry budget state AFTER checkpoint restore
                        logger.info(
                            f"[US-41-008] Retry budget after restore: max_attempts={retry_budget.max_attempts}, "
                            f"batch_size={retry_budget.batch_size}, attempts_used={retry_budget.attempts}"
                        )

                        # US-41-008: Warn if restored budget has attempts but no batch_size (old checkpoint)
                        if retry_budget.attempts > 0 and retry_budget.batch_size is None:
                            logger.warning(
                                f"[US-41-008] Restored budget has {retry_budget.attempts} attempts but "
                                f"batch_size is None - indicates old checkpoint format without scaling info"
                            )

                        logger.info(
                            f"Retry budget restored from checkpoint: "
                            f"{retry_budget.attempts} attempts, {retry_budget.failures} failures, "
                            f"{retry_budget.videos_skipped} skipped"
                        )

                        # US-42-012: Reset budget counters if --reset-budget flag is set
                        reset_budget_flag = getattr(config.download, 'reset_budget', False)
                        if reset_budget_flag:
                            old_attempts = retry_budget.attempts
                            old_failures = retry_budget.failures
                            old_skipped = retry_budget.videos_skipped
                            retry_budget.reset(preserve_vpn_count=True)
                            logger.info(
                                f"[US-42-012] Retry budget reset via --reset-budget flag: "
                                f"cleared {old_attempts} attempts, {old_failures} failures, "
                                f"{old_skipped} skipped"
                            )
                except Exception as e:
                    logger.debug(f"Could not restore retry budget from checkpoint: {e}")

            # US-002 Sprint 7: Apply adaptive format ordering from historical success rates
            # This reorders preferred_formats based on what worked best in previous runs
            adaptive_enabled = getattr(caption_config, 'adaptive_format_order', True)
            if adaptive_enabled and caption_cache.enabled:
                new_order = self._fetcher.apply_adaptive_format_order(cache=caption_cache)
                if self._fetcher._using_adaptive_order:
                    logger.info(f"  Using adaptive format order based on historical success rates")

            # US-67-006: Load persisted format stats from .cache/caption_format_stats.json
            # If adaptive ordering wasn't activated from in-memory cache, try the file
            format_stats_file = None
            _project_dir = getattr(state, 'project_dir', None)
            if adaptive_enabled and _project_dir:
                from ..caption.format_stats import FormatStatsFile
                format_stats_file = FormatStatsFile(_project_dir)
                if not getattr(self._fetcher, '_using_adaptive_order', False):
                    persisted_order = format_stats_file.compute_order(
                        default_formats=list(self._fetcher._default_formats),
                    )
                    if persisted_order:
                        self._fetcher._preferred_formats = persisted_order
                        self._fetcher._using_adaptive_order = True
                        logger.info(f"[US-67-006] Loaded persisted format order: {persisted_order}")
                        logger.info(f"  Using persisted format order from previous run")

            # US-004: Get video durations for coverage calculation
            video_durations = self._get_video_durations(state, config)

            # Check for already-fetched captions in checkpoint
            existing_captions = self._load_existing_captions(checkpoint)
            logger.info(f"  Found {len(existing_captions)} captions in checkpoint")

            # US-005 Sprint 8: Check for batch checkpoint from aborted run
            batch_checkpoint_path = None
            batch_checkpoint = None
            project_dir = getattr(state, 'project_dir', None)
            if project_dir:
                batch_checkpoint_path = CaptionBatchCheckpoint.get_checkpoint_path(project_dir)
                batch_checkpoint = CaptionBatchCheckpoint.load(batch_checkpoint_path)
                if batch_checkpoint:
                    if batch_checkpoint.aborted:
                        logger.warning(f"  Found aborted batch checkpoint: {batch_checkpoint.success_count} fetched, "
                                       f"{len(batch_checkpoint.remaining_video_ids)} remaining")
                        logger.warning(f"    Abort reason: {batch_checkpoint.abort_reason[:80]}...")
                    else:
                        logger.info(f"  Found batch checkpoint: {batch_checkpoint.success_count} fetched")
                    # Merge batch checkpoint results into existing captions
                    for vid, result in batch_checkpoint.results.items():
                        if vid not in existing_captions:
                            existing_captions[vid] = result

            # Prepare caption results with existing cached entries
            caption_results = {}
            skip_count = 0

            # Track cache hits from checkpoint (these were fetched in a previous run)
            for video_id in video_ids:
                if video_id in existing_captions:
                    caption_results[video_id] = existing_captions[video_id]
                    skip_count += 1
                    # US-011: Record as cache hit in metrics
                    cached_data = existing_captions[video_id]
                    if not cached_data.get('unavailable') and not cached_data.get('error'):
                        # US-004: Get coverage ratio from cache or calculate if missing
                        coverage_ratio = cached_data.get('coverage_ratio')
                        metrics.record_cache_hit(
                            video_id=video_id,
                            language=cached_data.get('language', 'en'),
                            quality=cached_data.get('caption_quality', 'medium'),
                            segment_count=cached_data.get('segment_count', 0),
                            is_auto_generated=cached_data.get('is_auto_generated', False),
                            coverage_ratio=coverage_ratio,
                            min_coverage_threshold=min_coverage_threshold,
                        )

            # IDs that need fetching (not in checkpoint)
            ids_to_fetch = [vid for vid in video_ids if vid not in existing_captions]

            # US-002: Check for live streams and skip them
            # US-007 Sprint 8: Enhanced stream state classification
            skip_live_streams = getattr(caption_config, 'skip_live_streams', True)
            handle_upcoming = getattr(caption_config, 'handle_upcoming', 'skip')
            live_stream_ids = []
            upcoming_stream_ids = []
            pending_streams: List[Dict[str, Any]] = []  # For 'queue' mode

            if ids_to_fetch and skip_live_streams:
                # Import StreamState for classification
                from ..caption_fetcher import StreamState

                logger.info(f"  Checking {len(ids_to_fetch)} videos for stream states...")
                for video_id in ids_to_fetch:
                    state_result = self._fetcher.get_stream_state(video_id)

                    if state_result.state == StreamState.LIVE:
                        # Currently live - always skip
                        live_stream_ids.append(video_id)
                        metrics.record_skipped_live_stream(video_id)
                        caption_results[video_id] = {
                            'video_id': video_id,
                            'skipped': True,
                            'reason': 'live_stream',
                            'stream_state': 'LIVE',
                            'caption_quality': 'low',
                        }
                        logger.warning(f"Skipping live stream: {state_result}")

                    elif state_result.state in (StreamState.UPCOMING, StreamState.PREMIERE):
                        # Scheduled stream/premiere - handle based on config
                        if handle_upcoming == 'skip':
                            # Treat like live - skip entirely
                            upcoming_stream_ids.append(video_id)
                            metrics.record_skipped_live_stream(video_id)
                            caption_results[video_id] = {
                                'video_id': video_id,
                                'skipped': True,
                                'reason': 'upcoming_stream',
                                'stream_state': state_result.state.name,
                                'scheduled_start': state_result.scheduled_start,
                                'caption_quality': 'low',
                            }
                            logger.info(f"Skipping upcoming: {state_result}")

                        elif handle_upcoming == 'queue':
                            # Add to pending list for later processing
                            upcoming_stream_ids.append(video_id)
                            pending_streams.append({
                                'video_id': video_id,
                                'stream_state': state_result.state.name,
                                'scheduled_start': state_result.scheduled_start,
                                'live_status': state_result.live_status,
                            })
                            caption_results[video_id] = {
                                'video_id': video_id,
                                'skipped': True,
                                'reason': 'queued_upcoming',
                                'stream_state': state_result.state.name,
                                'scheduled_start': state_result.scheduled_start,
                                'caption_quality': 'low',
                            }
                            logger.info(f"Queued for later: {state_result}")

                        elif handle_upcoming == 'check_later':
                            # Skip but don't mark as failed - can retry
                            upcoming_stream_ids.append(video_id)
                            caption_results[video_id] = {
                                'video_id': video_id,
                                'skipped': True,
                                'reason': 'check_later',
                                'stream_state': state_result.state.name,
                                'scheduled_start': state_result.scheduled_start,
                                'caption_quality': 'low',
                            }
                            logger.info(f"Will check later: {state_result}")

                    # VOD and UNKNOWN proceed to caption fetch

                # Remove live/upcoming streams from fetch list
                skip_ids = set(live_stream_ids + upcoming_stream_ids)
                if skip_ids:
                    ids_to_fetch = [vid for vid in ids_to_fetch if vid not in skip_ids]
                    # Report separately
                    if live_stream_ids:
                        logger.warning(f"  Skipped {len(live_stream_ids)} live streams (will use transcription fallback)")
                        # US-60-006: Add live streams to transcription fallback list
                        self.needs_transcription.extend(live_stream_ids)
                    if upcoming_stream_ids:
                        action = "queued" if handle_upcoming == "queue" else "skipped"
                        logger.warning(f"  {action.capitalize()} {len(upcoming_stream_ids)} upcoming/premiere streams")
                        # US-60-006: Add skipped upcoming streams to transcription fallback list
                        if handle_upcoming == 'skip':
                            self.needs_transcription.extend(upcoming_stream_ids)

                # Store pending streams in state if queued
                if pending_streams:
                    state.pending_streams = pending_streams
                    logger.info(f"Added {len(pending_streams)} streams to pending queue")

            # US-008: Pre-check caption availability to filter out videos without captions
            # US-006 Sprint 7: Use batch pre-check by channel when enabled
            # US-100-002: Use metadata language detection to skip pre-check for high-confidence predictions
            pre_check_enabled = getattr(caption_config, 'pre_check_availability', True)
            batch_precheck_enabled = getattr(caption_config, 'batch_precheck_by_channel', True)
            language_detection_enabled = getattr(caption_config, 'enable_language_detection', True)
            language_confidence_threshold = getattr(caption_config, 'language_detection_confidence_threshold', 0.8)
            no_caption_ids = []

            # US-100-002: Get video metadata for language detection
            video_metadata_map = {}
            if language_detection_enabled and ids_to_fetch:
                # Try to get metadata from video_search_results or enrich them
                from ..caption_fetcher import LanguagePrediction
                for video_id in ids_to_fetch:
                    # Get metadata from search results if available
                    title = ""
                    description = ""
                    tags = []
                    if video_id in video_search_results:
                        result = video_search_results[video_id]
                        title = getattr(result, 'title', '') or ""
                        description = getattr(result, 'description', '') or ""
                        tags = getattr(result, 'tags', []) or []

                    # Get language prediction
                    prediction = self._fetcher.get_detected_language_for_video(
                        video_id=video_id,
                        title=title,
                        description=description,
                        tags=tags
                    )
                    video_metadata_map[video_id] = prediction

                    logger.debug(
                        f"Language detection for {video_id}: {prediction.language} "
                        f"(confidence={prediction.confidence:.2f}, source={prediction.source})"
                    )

                # US-100-002: Skip pre-check for high-confidence language predictions
                # When we have high confidence in the detected language, skip pre-check
                # and go directly to fetching captions in that language
                if pre_check_enabled and language_confidence_threshold > 0:
                    videos_to_skip_precheck = [
                        video_id for video_id, pred in video_metadata_map.items()
                        if pred.confidence >= language_confidence_threshold
                    ]
                    if videos_to_skip_precheck:
                        logger.warning(f"  Language detection: skipping pre-check for {len(videos_to_skip_precheck)} "
                                       f"high-confidence videos (threshold={language_confidence_threshold})")
                        for video_id in videos_to_skip_precheck:
                            pred = video_metadata_map[video_id]
                            logger.debug(
                                f"Skipping pre-check for {video_id}: detected {pred.language} "
                                f"(confidence={pred.confidence:.2f})"
                            )
                        # These videos will skip pre-check and go directly to fetch

            if ids_to_fetch and pre_check_enabled:
                if batch_precheck_enabled and len(ids_to_fetch) > 1:
                    # US-006: Use batch pre-check with channel grouping
                    # This reduces API calls by grouping videos by channel and using
                    # representative samples when channel patterns have high confidence
                    from ..caption_fetcher import BatchPreCheckResult

                    confidence = getattr(caption_config, 'batch_precheck_confidence', 0.9)
                    min_samples = getattr(caption_config, 'batch_precheck_min_samples', 5)
                    sample_size = getattr(caption_config, 'batch_precheck_sample_size', 5)
                    clustering_enabled = getattr(caption_config, 'precheck_clustering_enabled', False)
                    min_cluster_size = getattr(caption_config, 'precheck_min_cluster_size', 3)

                    logger.info(f"  Batch pre-checking {len(ids_to_fetch)} videos (channel grouping)...")
                    batch_result = self._fetcher.batch_precheck_by_channel(
                        video_ids=ids_to_fetch,
                        cache=caption_cache,
                        metrics=metrics,
                        confidence_threshold=confidence,
                        min_samples_for_confidence=min_samples,
                        sample_size_per_channel=sample_size,
                        clustering_enabled=clustering_enabled,
                        min_cluster_size=min_cluster_size,
                    )

                    # Process batch results
                    for video_id, has_caps in batch_result.video_results.items():
                        if not has_caps:
                            no_caption_ids.append(video_id)
                            # Store as unavailable in caption results
                            caption_results[video_id] = {
                                'video_id': video_id,
                                'unavailable': True,
                                'reason': 'no_captions_available',
                                'caption_quality': 'low',
                            }
                            logger.info(f"Pre-check: No captions for {video_id}")

                    # Record API calls saved
                    metrics.set_batch_precheck_savings(batch_result.api_calls_saved)

                    if batch_result.api_calls_saved > 0:
                        logger.info(f"  + Batch pre-check saved {batch_result.api_calls_saved} API calls "
                                    f"({batch_result.skipped_by_pattern} skipped by channel pattern)")

                else:
                    # Original individual pre-check (US-008)
                    logger.info(f"  Pre-checking caption availability for {len(ids_to_fetch)} videos...")
                    for video_id in ids_to_fetch:
                        try:
                            has_caps = self._fetcher.has_captions(video_id)
                            metrics.record_pre_check(video_id, has_caps)
                            if not has_caps:
                                no_caption_ids.append(video_id)
                                # Store as unavailable in caption results
                                caption_results[video_id] = {
                                    'video_id': video_id,
                                    'unavailable': True,
                                    'reason': 'no_captions_available',
                                    'caption_quality': 'low',
                                }
                                logger.info(f"Pre-check: No captions for {video_id}")
                        except CaptionFetchError as e:
                            # Pre-check failed, but we'll still try to fetch later
                            logger.warning(f"Pre-check error for {video_id}: {e}")
                            metrics.record_pre_check(video_id, True)  # Assume available, try fetch

                # Remove videos without captions from fetch list
                if no_caption_ids:
                    ids_to_fetch = [vid for vid in ids_to_fetch if vid not in no_caption_ids]
                    logger.warning(f"  Pre-check: {len(no_caption_ids)} videos have no captions (will use transcription fallback)")
                    # US-60-006: Add no-caption videos to transcription fallback list
                    self.needs_transcription.extend(no_caption_ids)

            if ids_to_fetch:
                logger.info(f"  Fetching {len(ids_to_fetch)} new videos with {max_workers} parallel workers...")

                # US-81-004: Get progress reporter from state (set by pipeline)
                _progress_reporter = getattr(state, '_progress_reporter', None)
                if _progress_reporter:
                    _progress_reporter.update(total=len(ids_to_fetch))

                # US-009: TTY-aware progress callback for real-time streaming output
                # US-001 (Sprint 6): Thread-safe with lock for concurrent access
                import sys
                import threading
                is_tty = sys.stdout.isatty()
                last_line_length = 0  # Track for clearing overwritten lines
                progress_lock = threading.Lock()  # US-001: Protect TTY writes and state

                def on_progress(video_id: str, status: str, details: Dict) -> None:
                    """Print progress for each video fetch with TTY-aware formatting.

                    US-009: Streaming progress output showing per-video details.
                    Format: [32/100] abc123XYZ: en (auto, 45 segments, quality=medium)

                    In TTY mode: overwrites line for 'fetching', newline for final status
                    In non-TTY mode: newline for each status update

                    Thread-safety (US-001 Sprint 6):
                        This callback is invoked from multiple threads in the
                        ThreadPoolExecutor used by fetch_captions_batch(). A threading.Lock
                        protects both TTY writes and the last_line_length state variable
                        to prevent garbled output when 8+ workers call simultaneously.
                        Typical lock overhead is <1ms per callback invocation.
                    """
                    nonlocal last_line_length
                    idx = details.get('index', 0)
                    total = details.get('total', 0)

                    # US-001: Acquire lock to protect TTY writes and last_line_length
                    with progress_lock:
                        if status == 'fetching':
                            # Show in-progress indicator
                            line = f"  [{idx}/{total}] {video_id}: fetching..."
                            if is_tty:
                                # Overwrite line in TTY mode
                                padding = max(0, last_line_length - len(line))
                                logger.debug(f"\r{line}{' ' * padding}")
                                last_line_length = len(line)
                            # In non-TTY mode, skip 'fetching' status to reduce noise

                        elif status == 'success':
                            lang = details.get('language', '?')
                            quality = details.get('quality', '?')
                            segs = details.get('segment_count', 0)
                            auto_label = 'auto' if details.get('is_auto_generated') else 'human'
                            # Format: [32/100] abc123XYZ: en (auto, 45 segments, quality=medium)
                            line = f"  [{idx}/{total}] {video_id}: {lang} ({auto_label}, {segs} segments, quality={quality})"
                            # US-38-004: Add budget status when >50% consumed
                            budget_pct = details.get('budget_consumed_pct')
                            if budget_pct is not None and budget_pct > 50:
                                line += f" (budget: {budget_pct:.0f}%)"
                            if is_tty:
                                # Clear fetching line and print final status
                                padding = max(0, last_line_length - len(line))
                                logger.debug(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                logger.debug(line)

                        elif status == 'failed':
                            reason = details.get('reason', 'unknown')
                            error_msg = details.get('error', reason)
                            # Truncate error message if too long
                            if len(str(error_msg)) > 50:
                                error_msg = str(error_msg)[:47] + '...'
                            line = f"  [{idx}/{total}] {video_id}: FAILED ({error_msg})"
                            # US-38-004: Add budget status when >50% consumed
                            budget_pct = details.get('budget_consumed_pct')
                            if budget_pct is not None and budget_pct > 50:
                                line += f" (budget: {budget_pct:.0f}%)"
                            if is_tty:
                                # Clear fetching line and print final status
                                padding = max(0, last_line_length - len(line))
                                logger.debug(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                logger.debug(line)

                        elif status == 'skipped':
                            reason = details.get('reason', 'unknown')
                            line = f"  [{idx}/{total}] {video_id}: skipped ({reason})"
                            # US-38-004: Add budget status when >50% consumed
                            budget_pct = details.get('budget_consumed_pct')
                            if budget_pct is not None and budget_pct > 50:
                                line += f" (budget: {budget_pct:.0f}%)"
                            if is_tty:
                                padding = max(0, last_line_length - len(line))
                                logger.debug(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                logger.debug(line)

                        elif status == 'slow_video':
                            # US-78-010: Warn when a video takes >30s to process
                            elapsed_s = details.get('elapsed_seconds', 0)
                            line = f"  [{idx}/{total}] {video_id}: SLOW ({elapsed_s:.1f}s)"
                            if is_tty:
                                padding = max(0, last_line_length - len(line))
                                logger.debug(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                logger.debug(line)

                        # US-81-004: Update centralized progress reporter on terminal statuses
                        if _progress_reporter and status in ('success', 'failed', 'skipped'):
                            failed_delta = 1 if status == 'failed' else 0
                            completed_delta = 1 if status != 'failed' else 0
                            _progress_reporter.update(
                                completed=completed_delta,
                                failed=failed_delta,
                            )

                # US-005 Sprint 8: Create or reuse batch checkpoint for partial recovery
                checkpoint_save_interval = getattr(caption_config, 'checkpoint_save_interval', 10)
                if batch_checkpoint is None and batch_checkpoint_path:
                    batch_checkpoint = CaptionBatchCheckpoint(
                        total_requested=len(ids_to_fetch),
                        remaining_video_ids=list(ids_to_fetch)
                    )

                # US-34-002: Initialize global rate limit coordinator if enabled
                rate_limit_coordinator = None
                use_global_coordinator = getattr(caption_config, 'use_global_coordinator', True)
                if use_global_coordinator:
                    try:
                        from ..rate_limit.coordinator import GlobalRateLimitCoordinator
                        rate_limit_coordinator = GlobalRateLimitCoordinator()
                        if rate_limit_coordinator.is_enabled():
                            logger.info(
                                f"Using global rate limit coordinator: "
                                f"slots_per_second={rate_limit_coordinator._config.slots_per_second}"
                            )
                    except ImportError:
                        logger.debug("GlobalRateLimitCoordinator not available, using local rate limiting")
                    except Exception as e:
                        logger.warning(f"Failed to initialize GlobalRateLimitCoordinator: {e}")

                # US-37-003/US-37-004: Scale retry budget to batch size if auto_scale enabled
                # Default max_attempts=100 is insufficient for large batches (175+ videos)
                # US-38-003: Log budget scaling decision for debugging
                batch_size = len(ids_to_fetch)

                # US-38-010: Validate budget for batch before scaling
                # Warns user if budget may be insufficient for the batch size
                self._validate_budget_for_batch(batch_size, retry_budget)

                # US-39-010: Use ensure_scaled() convenience method
                # US-40-002: Re-scale budget after checkpoint restoration
                # US-41-003: Log budget state BEFORE and AFTER scaling for debugging
                # US-41-012: Perform batch_size integrity check when checkpoint data exists
                # This handles auto_scale check, idempotency, and logging internally
                if retry_budget:
                    # Check if budget was restored from checkpoint (has previous batch_size)
                    was_restored = retry_budget.batch_size is not None
                    old_batch_size = retry_budget.batch_size

                    # US-41-003: Log BEFORE scaling - shows initial state for debugging budget exhaustion
                    logger.info(
                        f"[US-41-003] Caption batch starting: batch_size={batch_size}, "
                        f"max_attempts={retry_budget.max_attempts} (before scaling), "
                        f"auto_scale={retry_budget.auto_scale}"
                    )

                    # US-41-012: Perform integrity check if checkpoint data exists
                    restore_info = None
                    if checkpoint_retry_budget_data:
                        _, restore_info = CaptionRetryBudget.restore_with_integrity_check(
                            checkpoint_retry_budget_data,
                            current_batch_size=batch_size,
                            auto_scale=retry_budget.auto_scale,
                            attempts_per_video=retry_budget.attempts_per_video,
                        )
                        # Log restore info for debugging
                        if restore_info.get("batch_size_changed"):
                            logger.warning(
                                f"[US-41-012] Checkpoint batch_size ({restore_info['restored_batch_size']}) "
                                f"differs from current ({batch_size})"
                            )
                        if restore_info.get("force_rescaled"):
                            # Apply the forced re-scale to our budget
                            retry_budget.max_attempts = restore_info["scaled_max_attempts"]
                            retry_budget.batch_size = batch_size
                            logger.info(
                                f"[US-41-012] Force re-scaled from checkpoint: max_attempts="
                                f"{restore_info['scaled_max_attempts']} for batch_size={batch_size}"
                            )

                    # Only call ensure_scaled if we didn't already force re-scale
                    if not (restore_info and restore_info.get("force_rescaled")):
                        scaled = retry_budget.ensure_scaled(batch_size)

                        # US-41-003: Log AFTER scaling - shows whether scaling occurred and new value
                        if scaled:
                            logger.info(
                                f"[US-41-003] Retry budget scaled: max_attempts from 100 to "
                                f"{retry_budget.max_attempts} for batch of {batch_size} videos"
                            )
                            if was_restored and old_batch_size != batch_size:
                                logger.info(
                                    f"Retry budget restored from checkpoint, re-scaling for batch of "
                                    f"{batch_size} videos (was {old_batch_size})"
                                )
                        else:
                            logger.info(
                                f"[US-41-003] Retry budget scaling not needed: max_attempts="
                                f"{retry_budget.max_attempts} sufficient for batch_size={batch_size}"
                            )
                            if was_restored:
                                logger.info(
                                    f"Retry budget restored from checkpoint, re-scaling for batch of "
                                    f"{batch_size} videos"
                                )

                    # US-41-004: Fail-fast verification - catch math errors before processing
                    # US-42-011: Use graceful degradation in non-interactive mode
                    # In non-interactive mode, continue with warning rather than failing
                    non_interactive = getattr(config.download, 'non_interactive', False)
                    degradation_info = retry_budget.verify_budget_sufficient(
                        batch_size,
                        graceful_degradation=non_interactive
                    )
                    if degradation_info:
                        # Budget is insufficient but we're continuing (graceful degradation)
                        warnings.append(
                            f"Budget insufficient: expected ~{degradation_info['expected_skips']} "
                            f"videos to be skipped (budget covers ~{degradation_info['budget_covers_pct']:.0f}%)"
                        )

                    # US-42-010: Proactive health check - log budget state before processing
                    # Helps users understand budget state after checkpoint restore
                    retry_budget.log_health_check(batch_size)

                    # US-43-010: Fail-fast check for budget exhaustion before batch start
                    # If checkpoint restores budget with 100/100 attempts already used,
                    # detect and report early instead of immediately skipping all videos
                    if retry_budget.budget_exhausted():
                        logger.error(
                            "[US-43-010] Budget already exhausted at batch start - check checkpoint restore. "
                            f"Attempts: {retry_budget.attempts}/{retry_budget.max_attempts}, "
                            f"Backoff: {retry_budget.backoff_time_spent:.1f}s/{retry_budget.max_backoff_time}s"
                        )
                        logger.warning(f"\n  ! Budget already exhausted at batch start:")
                        logger.warning(f"    - Attempts used: {retry_budget.attempts}/{retry_budget.max_attempts}")
                        logger.warning(f"    - Backoff time: {retry_budget.backoff_time_spent:.1f}s/{retry_budget.max_backoff_time}s")
                        logger.warning(f"    - {len(ids_to_fetch)} videos will be skipped")
                        logger.warning(f"    - Consider: --reset-budget flag or deleting checkpoint.json")
                        # US-100-011: Show recovery suggestions
                        logger.warning(f"\n{retry_budget.get_recovery_suggestions_formatted()}")

                        # Mark all videos as skipped due to budget exhaustion
                        for video_id in ids_to_fetch:
                            caption_results[video_id] = {
                                'video_id': video_id,
                                'skipped': True,
                                'reason': 'budget_exhausted_at_start',
                                'caption_quality': 'low',
                            }

                        # Store caption_results in state
                        state.caption_results = caption_results

                        # Populate text_metadata for consistency
                        try:
                            self._populate_text_metadata(state, caption_results, config)
                        except Exception as e:
                            logger.warning(f"Failed to populate text_metadata: {e}")

                        # Return early with warning
                        warnings.append(
                            f"Budget exhausted at batch start: {len(ids_to_fetch)} videos skipped. "
                            f"Use --reset-budget or delete checkpoint.json"
                        )

                        # Log retry budget summary for debugging
                        logger.info(retry_budget.get_formatted_summary())

                        return StageResult.ok({
                            'caption_results': caption_results,
                            'success_count': 0,
                            'skip_count': skip_count,
                            'fail_count': 0,
                            'budget_exhausted_at_start': True,
                            'skipped_due_to_budget': len(ids_to_fetch),
                            'retry_budget': retry_budget.to_dict(),
                        }, warnings)

                # US-59-008: Build video-to-channel map for list-subs deduplication
                if hasattr(state, 'video_search_results') and state.video_search_results:
                    video_channel_map = {}
                    for result in state.video_search_results:
                        vid = getattr(result, 'video_id', None)
                        ch = getattr(result, 'channel', None)
                        if vid and ch:
                            video_channel_map[vid] = ch
                    if video_channel_map:
                        self._fetcher.set_video_channel_map(video_channel_map)
                        logger.debug(
                            f"US-59-008: Set video-channel map with {len(video_channel_map)} entries"
                        )

                # US-001: Use batch fetch for parallel processing
                # US-005 Sprint 8: With checkpoint support for abort recovery
                # US-33-009: With circuit breaker for consecutive failure protection
                # US-34-002: With global rate limit coordinator for unified rate limiting
                # US-61-011: VPN rotation on caption budget exhaustion with rate limits
                try:
                    # US-61-011: Initialize VPN manager for potential rotation
                    mullvad_vpn = None
                    mullvad_config = getattr(config.download, 'mullvad', None)
                    if mullvad_config and getattr(mullvad_config, 'enabled', False):
                        from ..downloader.mullvad_vpn import MullvadVPN
                        if MullvadVPN.is_available():
                            mullvad_vpn = MullvadVPN(mullvad_config)
                            logger.debug("US-61-011: MullvadVPN initialized for caption budget exhaustion rotation")

                    # US-61-011: Track remaining videos for potential VPN rotation retry
                    remaining_ids = list(ids_to_fetch)
                    all_batch_results: Dict[str, Any] = {}

                    # US-61-011: Loop to support VPN rotation retry
                    while remaining_ids:
                        batch_results = self._fetcher.fetch_captions_batch(
                            video_ids=remaining_ids,
                            preferred_language=preferred_lang,
                            max_workers=max_workers,
                            metrics=metrics,
                            progress_callback=on_progress,
                            batch_checkpoint=batch_checkpoint,
                            checkpoint_save_interval=checkpoint_save_interval,
                            circuit_breaker=circuit_breaker,
                            retry_budget=retry_budget,
                            rate_limit_coordinator=rate_limit_coordinator,
                        )

                        # Merge results into all_batch_results
                        all_batch_results.update(batch_results)

                        # Log progress for caption fetch
                        total_captions = len(video_ids)
                        progress_pct = ((success_count + skip_count + len(all_batch_results)) / total_captions * 100) if total_captions > 0 else 0
                        log_progress(logger, "CAPTION", progress_pct, success_count + skip_count, total_captions)

                        # US-100-011: Check and warn if budget threshold exceeded
                        if retry_budget:
                            warning = retry_budget.check_and_warn_budget_threshold()
                            if warning:
                                logger.warning(f"[US-100-011] {warning}")
                                # Print to console in TTY mode
                                if sys.stdout.isatty():
                                    logger.warning(f"\n  ! {warning}")

                        # US-61-011: Check if VPN rotation should be triggered
                        if retry_budget and mullvad_vpn and retry_budget.should_trigger_vpn_rotation():
                            # Get videos that were skipped due to budget exhaustion
                            skipped_ids = [
                                vid for vid, result in batch_results.items()
                                if isinstance(result, dict)
                                and result.get('skipped')
                                and result.get('reason') == 'budget_exhausted'
                            ]

                            if skipped_ids:
                                rate_limit_pct = retry_budget.get_rate_limit_error_percentage()
                                logger.info(
                                    f"Caption budget exhausted ({rate_limit_pct:.0f}% rate-limited), "
                                    f"rotating VPN and retrying {len(skipped_ids)} videos"
                                )
                                logger.warning(f"\n  ! Caption budget exhausted ({rate_limit_pct:.0f}% rate-limited)")
                                logger.warning(f"    Rotating VPN and retrying {len(skipped_ids)} remaining videos...")

                                # Rotate VPN (also resets circuit breaker if provided)
                                if mullvad_vpn.rotate_server(circuit_breaker=circuit_breaker):
                                    # Record VPN reset and reset budget for retry
                                    retry_budget.record_vpn_reset()
                                    retry_budget.reset(preserve_vpn_count=True)

                                    # Re-scale budget for remaining videos
                                    retry_budget.ensure_scaled(len(skipped_ids))

                                    # Set remaining_ids to skipped videos for retry
                                    remaining_ids = skipped_ids
                                    logger.info(f"    VPN rotated successfully, retrying...")
                                    continue
                                else:
                                    logger.warning("VPN rotation failed, skipping retry")
                                    logger.warning(f"    ! VPN rotation failed, cannot retry")

                        # No VPN rotation needed or possible - exit loop
                        break

                    # US-100-006: Rate limit feedback loop - reduce workers if rate limit errors increased
                    rate_limit_feedback_enabled = getattr(caption_config, 'rate_limit_feedback_enabled', True)
                    if rate_limit_feedback_enabled and batch_size > 10:
                        # Check rate limit error rate from metrics
                        rate_limit_errors = metrics.error_category_counts.get('RATE_LIMIT', 0)
                        total_errors = sum(metrics.error_category_counts.values())
                        if total_errors > 0:
                            error_rate = rate_limit_errors / max(total_errors, 1)
                            rate_limit_error_threshold = getattr(caption_config, 'rate_limit_error_threshold', 0.15)

                            if error_rate > rate_limit_error_threshold:
                                # Reduce workers for next batch
                                new_worker_count = max(2, max_workers - 1)
                                logger.info(
                                    f"Rate limit feedback: reducing workers from {max_workers} to {new_worker_count} "
                                    f"(rate_limit_errors={rate_limit_errors}, error_rate={error_rate:.1%})"
                                )
                                logger.warning(f"  ! High rate limit errors ({error_rate:.1%}), reducing workers to {new_worker_count}")
                                max_workers = new_worker_count

                    # Use all_batch_results for the rest of the processing
                    batch_results = all_batch_results

                    # US-005 Sprint 8: Save final checkpoint on success
                    if batch_checkpoint and batch_checkpoint_path:
                        batch_checkpoint.save(batch_checkpoint_path)
                        logger.info(f"Batch checkpoint saved: {batch_checkpoint.success_count} successes")
                except ErrorPatternAbortError as e:
                    # US-005 Sprint 8: Checkpoint already updated in fetch_captions_batch
                    # Save checkpoint to disk before re-raising
                    if batch_checkpoint and batch_checkpoint_path:
                        batch_checkpoint.save(batch_checkpoint_path)
                        logger.warning(f"  ! Batch aborted: checkpoint saved with {batch_checkpoint.success_count} results")
                        logger.warning(f"    Resume by running the pipeline again with --resume")
                    # Use partial results from the exception
                    batch_results = e.partial_results
                    warnings.append(f"Batch fetch aborted due to error pattern: {e.pattern_result}")

                # Convert batch results to checkpoint format
                for video_id, result in batch_results.items():
                    if isinstance(result, CaptionResult):
                        # US-004: Calculate coverage ratio if video duration available
                        video_duration = video_durations.get(video_id)
                        coverage_ratio = result.calculate_coverage(video_duration)

                        # US-008 Sprint 7: Validate timing and get penalty factor
                        timing_penalty = 1.0  # Default: no penalty
                        if video_duration:
                            result.validate_timing(video_duration=video_duration)
                            timing_penalty = result.timing_penalty_factor
                            if timing_penalty < 1.0:
                                logger.debug(
                                    f"Timing penalty for {video_id}: {timing_penalty:.2f} "
                                    f"(exceeds={result.timing_validated.exceeds_ratio:.2f}, "
                                    f"coverage={result.timing_validated.coverage_ratio:.2f})"
                                )

                        # US-100-008: Validate segment continuity (gaps/overlaps)
                        timing_validation_mode = getattr(caption_config, 'timing_validation_mode', 'lenient')
                        if timing_validation_mode != 'off':
                            gap_threshold = getattr(caption_config, 'continuity_gap_threshold', 1.0)
                            overlap_tolerance = getattr(caption_config, 'overlap_tolerance', 0.1)
                            segment_validation = result.validate_segments(
                                timing_validation_mode=timing_validation_mode,
                                gap_threshold=gap_threshold,
                                overlap_tolerance=overlap_tolerance,
                            )
                            if segment_validation and not segment_validation.is_valid:
                                logger.debug(
                                    f"Segment continuity issues for {video_id}: "
                                    f"{segment_validation.gap_count} gaps, {segment_validation.overlap_count} overlaps, "
                                    f"score={segment_validation.continuity_score:.2f}"
                                )

                        # Success - convert to serializable dict
                        caption_results[video_id] = {
                            'video_id': video_id,
                            'status': CaptionStatus.SUCCESS.value,  # US-63-006: Structured status
                            'segments': [seg.to_dict() for seg in result.segments],
                            'language': result.language,
                            'is_auto_generated': result.is_auto_generated,
                            'format_source': result.format_source,
                            'segment_count': len(result.segments),
                            'caption_quality': result.caption_quality,
                            'video_duration': video_duration,  # US-004
                            'coverage_ratio': coverage_ratio,  # US-004
                            'timing_penalty': timing_penalty,  # US-008 Sprint 7
                            'video_chapters': getattr(result, 'video_chapters', []),  # US-72-002
                            'video_tags': getattr(result, 'video_tags', []),  # US-72-002
                        }

                        # US-004: Update metrics with coverage info
                        # (fetch_captions_batch already recorded basic metrics, update coverage)
                        if coverage_ratio is not None:
                            coverage_level = metrics._classify_coverage(coverage_ratio)
                            with metrics._lock:
                                metrics.coverage_distribution[coverage_level] = (
                                    metrics.coverage_distribution.get(coverage_level, 0) + 1
                                )
                                if coverage_ratio < min_coverage_threshold:
                                    metrics.low_coverage_videos.append(video_id)
                    else:
                        # Error/unavailable - already in dict format
                        caption_results[video_id] = result

            # Count results
            # US-63-006: Use status field when available, fall back to legacy flags
            success_count = sum(
                1 for r in caption_results.values()
                if r.get('status') == CaptionStatus.SUCCESS.value
                or (
                    # Legacy fallback when status not present
                    not r.get('status')
                    and not r.get('unavailable') and not r.get('error') and not r.get('skipped')
                    and not r.get('no_captions_available')
                    and r.get('segment_count', 0) > 0
                )
            ) - skip_count  # Don't double-count cached entries

            # US-63-006: Count videos with no captions available (not an error, triggers transcription)
            # Uses status field when available
            no_captions_count = sum(
                1 for r in caption_results.values()
                if r.get('status') == CaptionStatus.NO_CAPTIONS.value
                or r.get('status') == CaptionStatus.CACHED_UNAVAILABLE.value
                or (
                    # Legacy fallback when status not present
                    not r.get('status')
                    and (r.get('unavailable') and r.get('reason') == 'no_captions_available'
                         or r.get('no_captions_available'))
                )
            )

            # US-63-006: Count actual fetch failures (errors, not including no_captions_available)
            fetch_failed_count = sum(
                1 for r in caption_results.values()
                if r.get('status') == CaptionStatus.ERROR.value
                or (
                    # Legacy fallback when status not present
                    not r.get('status')
                    and (r.get('error') or (r.get('unavailable') and r.get('reason') != 'no_captions_available'))
                    and not r.get('no_captions_available')
                )
            )

            # Legacy fail_count for backwards compatibility (includes both no_captions and errors)
            fail_count = no_captions_count + fetch_failed_count

            # US-81-009: Check batch failure threshold (only for actual fetch errors, not no_captions)
            _batch_failure_threshold = getattr(
                getattr(config, 'pipeline', None), 'batch_failure_threshold', 0.5
            )
            total_processed = success_count + skip_count + fetch_failed_count
            if total_processed > 0 and _batch_failure_threshold < 1.0:
                from . import check_batch_failure_threshold, BatchFailureThresholdExceeded
                try:
                    check_batch_failure_threshold(
                        items_processed=total_processed,
                        items_failed=fetch_failed_count,
                        threshold=_batch_failure_threshold,
                    )
                except BatchFailureThresholdExceeded as e:
                    logger.error(f"[US-81-009] Caption batch: {e}")
                    logger.warning(f"\n  ! Caption batch failure threshold exceeded: {e}")
                    warnings.append(f"Batch failure threshold exceeded: {e}")

            # US-63-006: Add INFO log for count of videos with no captions
            if no_captions_count > 0:
                logger.info(
                    f"US-63-006: {no_captions_count} videos have no captions available "
                    f"(will use transcription fallback)"
                )

            # US-60-006: Add videos that failed caption fetch to transcription fallback list
            # US-63-006: Handle no_captions status gracefully - not an error, triggers fallback
            for video_id, result in caption_results.items():
                if video_id not in self.needs_transcription:
                    status = result.get('status')
                    # Add to transcription list if no_captions, error, or budget exhausted
                    if status in (CaptionStatus.NO_CAPTIONS.value, CaptionStatus.CACHED_UNAVAILABLE.value, CaptionStatus.ERROR.value):
                        self.needs_transcription.append(video_id)
                    elif result.get('unavailable') or result.get('error'):
                        # Legacy fallback
                        self.needs_transcription.append(video_id)
                    elif result.get('skipped') and result.get('reason') == 'budget_exhausted':
                        self.needs_transcription.append(video_id)

            # US-59-011: Calculate batch-wide caption availability summary
            # Count videos with captions vs without (including skipped/error/unavailable)
            total_videos_in_batch = len(caption_results)
            videos_with_captions = success_count + skip_count  # Successful + cached
            videos_without_captions = fail_count + sum(
                1 for r in caption_results.values()
                if r.get('skipped') and r.get('reason') != 'queued_upcoming'
            )
            # Log batch summary
            # US-62-007: Include no_captions vs fetch_failed in summary
            logger.info(
                f"US-59-011: Caption batch summary: {videos_with_captions}/{total_videos_in_batch} "
                f"videos have captions, {no_captions_count} no captions, {fetch_failed_count} fetch errors"
            )

            # US-59-011: Set state flag when >50% of videos have no captions
            if total_videos_in_batch > 0:
                no_caption_ratio = videos_without_captions / total_videos_in_batch
                if no_caption_ratio > 0.5:
                    state.caption_batch_low_yield = True
                    logger.warning(
                        f"US-59-011: Low caption yield detected: {videos_without_captions}/{total_videos_in_batch} "
                        f"({no_caption_ratio:.0%}) videos have no captions - downstream stages will need transcription"
                    )
                    logger.warning(f"  ! Low caption yield: {no_caption_ratio:.0%} of videos have no captions")
                else:
                    state.caption_batch_low_yield = False
            # US-002: Count skipped live streams
            skipped_live_count = sum(
                1 for r in caption_results.values()
                if r.get('skipped') and r.get('reason') == 'live_stream'
            )

            # US-008: Count pre-check filtered videos (no captions available)
            pre_check_unavailable_count = sum(
                1 for r in caption_results.values()
                if r.get('unavailable') and r.get('reason') == 'no_captions_available'
            )

            # US-38-011: Store caption_results in state BEFORE populating text_metadata
            # This ensures data is preserved even if _populate_text_metadata fails
            state.caption_results = caption_results

            # Store caption data in state.text_metadata for matching
            try:
                self._populate_text_metadata(state, caption_results, config)
            except AttributeError as e:
                # US-39-008: Log AttributeError with state type and available attributes
                state_type = type(state).__name__
                state_attrs = [attr for attr in dir(state) if not attr.startswith('_')]
                logger.error(
                    f"_populate_text_metadata AttributeError: {e}. "
                    f"State type: {state_type}, "
                    f"Available attributes: {state_attrs[:20]}{'...' if len(state_attrs) > 20 else ''}"
                )
                # US-39-008: Warning that matching stage can use caption_results directly
                logger.warning(
                    f"Matching stage can use state.caption_results directly as fallback. "
                    f"caption_results preserved with {len(caption_results)} videos, {success_count} succeeded."
                )
                raise
            except Exception as e:
                # US-38-011: Log error with caption_results summary for debugging
                logger.error(
                    f"_populate_text_metadata failed: {e}. "
                    f"caption_results preserved in state ({len(caption_results)} videos, "
                    f"{success_count} succeeded)"
                )
                raise

            # Calculate quality distribution (US-007)
            quality_distribution = self._calculate_quality_distribution(caption_results)
            human_count = sum(1 for r in caption_results.values()
                             if not r.get('is_auto_generated') and not r.get('unavailable')
                             and not r.get('error') and not r.get('skipped'))
            auto_count = sum(1 for r in caption_results.values()
                            if r.get('is_auto_generated') and not r.get('unavailable')
                            and not r.get('error') and not r.get('skipped'))

            # Summary
            # US-62-007: Show distinct counts for no_captions vs fetch_failed vs succeeded
            logger.info(f"\n  + Caption fetch complete:")
            logger.info(f"    - Succeeded: {success_count} videos (new), {skip_count} videos (cached)")
            logger.info(f"    - No captions: {no_captions_count} videos (will use transcription)")
            logger.info(f"    - Fetch failed: {fetch_failed_count} videos (errors)")
            # US-002: Report skipped live streams
            if skipped_live_count > 0:
                logger.info(f"    - Skipped live streams: {skipped_live_count} videos")
            # US-008: Report pre-check filtered videos (subset of no_captions_count)
            if pre_check_unavailable_count > 0:
                logger.info(f"    - Pre-check filtered: {pre_check_unavailable_count} videos (no captions)")
            # US-007: Report caption quality distribution
            logger.info(f"    - Caption sources: {human_count} human, {auto_count} auto, {fail_count} fallback")
            logger.info(f"    - Quality distribution: {quality_distribution['high']} high, "
                        f"{quality_distribution['medium']} medium, {quality_distribution['low']} low")

            # US-011: Display metrics summary
            logger.info(f"\n  + Caption metrics (US-011):")
            for line in metrics.summary().split('\n'):
                logger.info(f"    {line}")

            # Fallback warnings (failed + skipped live streams)
            fallback_count = fail_count + skipped_live_count
            if fallback_count > 0 and getattr(caption_config, 'fallback_to_transcription', True):
                logger.info(f"    - {fallback_count} videos will use Whisper transcription fallback")
                if fail_count > 0:
                    warnings.append(f"{fail_count} videos require transcription fallback (unavailable)")
                if skipped_live_count > 0:
                    warnings.append(f"{skipped_live_count} live streams require transcription fallback")

            # US-004: Coverage warnings for low coverage videos
            low_coverage_count = len(metrics.low_coverage_videos)
            if low_coverage_count > 0:
                logger.info(f"    - Low coverage (<{min_coverage_threshold:.0%}): {low_coverage_count} videos")
                warnings.append(f"{low_coverage_count} videos have low caption coverage (<{min_coverage_threshold:.0%})")
                # Log first few low coverage videos for debugging
                for vid in metrics.low_coverage_videos[:5]:
                    result = caption_results.get(vid, {})
                    coverage = result.get('coverage_ratio', 0.0)
                    logger.warning(
                        f"Low caption coverage for {vid}: {coverage:.1%} "
                        f"(threshold: {min_coverage_threshold:.0%})"
                    )
                if low_coverage_count > 5:
                    logger.warning(f"... and {low_coverage_count - 5} more videos with low coverage")

            # US-002 Sprint 6: Print slowest fetches summary
            slowest = metrics.get_slowest_videos(5)
            if slowest:
                slowest_str = ", ".join(f"{vid}={t:.1f}s" for vid, t in slowest)
                logger.info(f"    - Slowest fetches: {slowest_str}")

            # US-33-009: Print circuit breaker stats if used
            if circuit_breaker and circuit_breaker.is_enabled:
                cb_stats = circuit_breaker.get_stats()
                if cb_stats['total_trips'] > 0 or cb_stats['consecutive_failures'] > 0:
                    logger.info(f"    - Circuit breaker: {cb_stats['total_trips']} trips, "
                                f"{cb_stats['total_paused_seconds']:.1f}s total pause, "
                                f"{cb_stats['consecutive_failures']} recent failures")

            # US-33-010: Print retry budget stats if used
            # US-40-004: Log summary at stage completion for observability
            # US-100-011: Add progress bar and recovery suggestions
            if retry_budget:
                rb_summary = retry_budget.get_summary()
                if rb_summary['attempts'] > 0 or rb_summary['videos_skipped'] > 0:
                    # US-100-011: Show progress bar in TTY mode
                    if sys.stdout.isatty():
                        logger.info(f"    {retry_budget.get_progress_bar()}")
                    logger.info(f"    - Retry budget: {rb_summary['attempts']} attempts, "
                                f"{rb_summary['failures']} failures, "
                                f"{rb_summary['backoff_time_spent']:.1f}s backoff, "
                                f"{rb_summary['videos_skipped']} skipped")
                    if rb_summary['is_exhausted']:
                        logger.warning(f"    ! Retry budget EXHAUSTED - remaining videos skipped")
                        # US-100-011: Show recovery suggestions
                        logger.warning(f"\n{retry_budget.get_recovery_suggestions_formatted()}")

                # US-40-004: Log INFO with formatted budget summary
                logger.info(retry_budget.get_formatted_summary())

                # US-42-011: Log comparison of actual vs expected skips (if graceful degradation was used)
                retry_budget.log_skip_comparison()

                # US-40-004: Log WARNING if any videos were skipped due to budget exhaustion
                if rb_summary['videos_skipped'] > 0:
                    logger.warning(
                        f"CaptionRetryBudget: {rb_summary['videos_skipped']} videos skipped "
                        f"due to budget exhaustion"
                    )

            # US-73-008: Structured summary log for caption stage diagnostics
            # Compute avg_duration_ms from per-video fetch times
            fetch_times_ms = [t * 1000 for t in metrics.video_fetch_times.values()] if metrics.video_fetch_times else []
            avg_duration_ms = round(sum(fetch_times_ms) / len(fetch_times_ms)) if fetch_times_ms else 0
            caption_stage_metrics = {
                'total_attempts': metrics.fetch_attempts,
                'success_count': metrics.successes,
                'failure_count': metrics.failures,
                'error_type_distribution': dict(metrics.error_category_counts),
                'avg_duration_ms': avg_duration_ms,
            }
            logger.info("caption_stage_summary %s", caption_stage_metrics)

            # US-78-005: Caption coverage gap summary
            # Compute CoverageAnalysis per video and log aggregate summary
            self._log_coverage_gap_summary(caption_results, video_durations)

            # US-002 Sprint 7: Save format statistics for cross-run learning
            # This enables adaptive format ordering in future runs
            if caption_cache.enabled and metrics.format_success_counts:
                if caption_cache.save_format_statistics(metrics.format_success_counts):
                    logger.info(f"Saved format statistics: {metrics.format_success_counts}")
                else:
                    logger.warning("Failed to save format statistics to cache")

            # US-67-006: Persist format stats to .cache/caption_format_stats.json
            if format_stats_file is not None and metrics.format_success_counts:
                from ..caption.format_stats import FormatStatsFile
                stats_dict = FormatStatsFile.from_metrics_counts(
                    success_counts=metrics.format_success_counts,
                )
                if format_stats_file.save(stats_dict):
                    logger.info(f"[US-67-006] Persisted format stats to {format_stats_file.path}")
                else:
                    logger.warning("[US-67-006] Failed to persist format stats to file")

            # US-60-006: Set state.videos_needing_transcription for downstream TRANSCRIBE stage
            state.videos_needing_transcription = list(self.needs_transcription)
            if self.needs_transcription:
                logger.info(
                    f"US-60-006: {len(self.needs_transcription)} videos need transcription fallback"
                )

            # US-81-002: Build per-item failed_items list with structured error info
            failed_items = []
            for vid, r in caption_results.items():
                if r.get('error'):
                    failed_items.append({
                        'video_id': vid,
                        'error_type': r.get('reason', 'fetch_error'),
                        'message': str(r.get('error', 'unknown error')),
                    })
                elif r.get('unavailable') and r.get('reason') != 'no_captions_available':
                    failed_items.append({
                        'video_id': vid,
                        'error_type': r.get('reason', 'unavailable'),
                        'message': r.get('reason', 'unavailable'),
                    })

            # Prepare checkpoint data (US-007: include quality stats, US-011: include metrics)
            checkpoint_data = {
                'caption_results': caption_results,
                'success_count': success_count,
                'skip_count': skip_count,
                'fail_count': fail_count,
                # US-62-007: Separate counts for no_captions vs fetch_failed
                'no_captions_count': no_captions_count,
                'fetch_failed_count': fetch_failed_count,
                'skipped_live_count': skipped_live_count,  # US-002
                'pre_check_unavailable_count': pre_check_unavailable_count,  # US-008
                'total_segments': sum(
                    r.get('segment_count', 0) for r in caption_results.values()
                ),
                # US-007: Quality distribution for analysis
                'quality_distribution': quality_distribution,
                'human_count': human_count,
                'auto_count': auto_count,
                # US-011: Full metrics for cross-session aggregation
                'caption_metrics': metrics.to_dict(),
                # US-59-011: Caption batch low yield flag for downstream stages
                'caption_batch_low_yield': getattr(state, 'caption_batch_low_yield', False),
                # US-60-006: Videos needing transcription fallback for resume support
                'needs_transcription': self.needs_transcription,
                # US-73-008: Structured metrics for post-run analysis
                'caption_stage_metrics': caption_stage_metrics,
                # US-81-002: Per-item error details for batch error isolation
                'failed_items': failed_items,
            }

            # US-37-007: Save retry budget state for resume support
            if retry_budget:
                checkpoint_data['retry_budget'] = retry_budget.to_dict()

            # US-81-002: Stage metrics for pipeline observability
            # US-81-007: Compute throughput samples from per-video fetch times
            items_processed = success_count + skip_count
            throughput_samples = []
            if metrics.video_fetch_times:
                for fetch_time in metrics.video_fetch_times.values():
                    if fetch_time > 0:
                        throughput_samples.append(1.0 / fetch_time)

            # US-90-009: Populate extra_metrics with caption-specific metrics for export
            extra_metrics = {
                'fetch_success_rate': metrics.success_rate,
                'avg_fetch_time': metrics.get_summary_dict().get('avg_fetch_time', 0.0),
                'cache_hit_rate': metrics.cache_hit_rate,
                'format_distribution': metrics.get_format_statistics(),
            }

            stage_metrics = StageMetrics(
                items_processed=items_processed,
                items_failed=fetch_failed_count,
                throughput_samples=throughput_samples,
                extra_metrics=extra_metrics,
            )
            stage_metrics.compute_throughput()

            # US-100-009: Export metrics to persistent storage
            metrics_export_enabled = getattr(caption_config, 'metrics_export_enabled', True)
            if metrics_export_enabled and metrics is not None:
                try:
                    metrics_export_path = getattr(caption_config, 'metrics_export_path', '.cache/caption_metrics.json')
                    metrics_retention_runs = getattr(caption_config, 'metrics_retention_runs', 10)
                    metrics_auto_cleanup = getattr(caption_config, 'metrics_auto_cleanup', True)

                    # Construct absolute path from project directory
                    export_path = checkpoint.project_dir / metrics_export_path
                    metrics.export_json(
                        path=str(export_path),
                        project_path=str(checkpoint.project_dir),
                        config=config,
                        video_count=len(video_ids),
                        auto_cleanup=metrics_auto_cleanup,
                        retention_count=metrics_retention_runs
                    )
                except Exception as e:
                    logger.warning(f"Failed to export caption metrics: {e}")

            # Log stage completion with summary metrics
            # US-167-009: Log stage completion with timing
            elapsed = time.time() - stage_start_time
            videos_processed = success_count + skip_count
            log_stage_complete(
                logger, "CAPTION",
                elapsed_seconds=elapsed,
                videos_processed=videos_processed,
                success_count=success_count,
                failed_count=fetch_failed_count,
                cache_hits=skip_count,
                no_captions=no_captions_count,
                fallback_to_transcription=fallback_count if 'fallback_count' in dir() else 0
            )

            return StageResult.ok(checkpoint_data, warnings, stage_metrics)

        except ImportError as e:
            logger.error(f"Could not import caption_fetcher: {e}")
            return StageResult.fail(f"Caption fetcher not available: {e}", warnings)
        except Exception as e:
            logger.exception(f"Caption stage failed: {e}")
            # US-40-004: Log retry budget summary even when stage fails
            if retry_budget is not None:
                logger.info(retry_budget.get_formatted_summary())
                # US-42-011: Log comparison of actual vs expected skips
                retry_budget.log_skip_comparison()
                rb_summary = retry_budget.get_summary()
                if rb_summary['videos_skipped'] > 0:
                    logger.warning(
                        f"CaptionRetryBudget: {rb_summary['videos_skipped']} videos skipped "
                        f"due to budget exhaustion"
                    )
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if caption stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore caption stage from checkpoint.

        US-42-002: Ensures state attributes (text_metadata, caption_results, etc.)
        exist before restoration. This mirrors _preflight_check() in run().
        """
        try:
            # US-42-002: Ensure state has required attributes before restoration
            # This is critical because restore() may be called on a fresh state
            # that hasn't gone through _preflight_check() yet
            self._ensure_state_attributes(state)

            data = checkpoint.get_stage_data(self.name)
            if not data:
                # No stage data means CAPTION stage had nothing to do
                # (e.g., caption-first mode was disabled) - this is a valid no-op
                logger.info("Restored CAPTION: no caption data (caption-first may be disabled)")
                return True

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                logger.warning(f"CAPTION restore: expected dict, got {type(data).__name__}")
                return False

            if 'caption_results' in data and not isinstance(data['caption_results'], dict):
                logger.warning(f"CAPTION restore: 'caption_results' expected dict, got {type(data['caption_results']).__name__}")
                return False

            # Restore caption results to state
            caption_results = data.get('caption_results', {})
            if caption_results:
                self._populate_text_metadata(state, caption_results, config)

                # US-011: Log metrics if available
                metrics_data = data.get('caption_metrics')
                if metrics_data:
                    from ..caption.metrics import CaptionMetrics
                    metrics = CaptionMetrics.from_dict(metrics_data)
                    logger.info(
                        f"Restored CAPTION: {len(caption_results)} videos, "
                        f"{data.get('total_segments', 0)} segments, "
                        f"metrics: {metrics.successes} succeeded, "
                        f"{metrics.failures} failed, {metrics.cache_hits} cached"
                    )
                else:
                    logger.info(f"Restored CAPTION: {len(caption_results)} videos, "
                               f"{data.get('total_segments', 0)} segments")

                # US-37-007: Log retry budget state if available
                retry_budget_data = data.get('retry_budget')
                if retry_budget_data:
                    from ..caption.retry_budget import CaptionRetryBudget
                    restored_budget = CaptionRetryBudget.from_dict(retry_budget_data)
                    rb_summary = restored_budget.get_summary()
                    logger.info(
                        f"Restored retry budget: {rb_summary['attempts']} attempts, "
                        f"{rb_summary['failures']} failures, "
                        f"{rb_summary['videos_skipped']} skipped"
                    )

                # US-59-011: Restore caption_batch_low_yield flag from checkpoint
                caption_batch_low_yield = data.get('caption_batch_low_yield', False)
                state.caption_batch_low_yield = caption_batch_low_yield
                if caption_batch_low_yield:
                    logger.info(
                        "Restored caption_batch_low_yield=True: many videos will need transcription"
                    )

                # US-60-006: Restore videos_needing_transcription from checkpoint
                needs_transcription = data.get('needs_transcription', [])
                state.videos_needing_transcription = needs_transcription
                if needs_transcription:
                    logger.info(
                        f"Restored videos_needing_transcription: {len(needs_transcription)} videos"
                    )
            else:
                # Stage data exists but no caption results - valid empty case
                logger.info("Restored CAPTION: 0 videos (no captions fetched)")
                # US-60-006: Ensure videos_needing_transcription is empty for consistency
                state.videos_needing_transcription = []

            return True

        except Exception as e:
            logger.warning(f"Failed to restore CAPTION: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running."""
        # In simplified pipeline, check for video_ids from VIDEO_SEARCH stage
        if hasattr(state, 'video_ids') and state.video_ids:
            return None

        # Backward compatibility: check legacy fields
        has_videos = hasattr(state, 'downloaded_videos') and len(state.downloaded_videos) > 0
        has_audio = hasattr(state, 'downloaded_audio') and len(state.downloaded_audio) > 0

        if not has_videos and not has_audio:
            return "No video candidates available for caption fetch"

        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        # Count input videos
        input_count = 0
        if hasattr(state, 'video_ids') and state.video_ids:
            input_count = len(state.video_ids)
        elif hasattr(state, 'downloaded_videos'):
            input_count = len(state.downloaded_videos)

        # Count output captions
        output_count = None
        if hasattr(state, 'captions'):
            output_count = len(state.captions)

        return {
            'inputs': 'video candidates',
            'outputs': 'captions',
            'input_count': input_count,
            'output_count': output_count,
        }

    def get_api_estimates(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get API call estimates for dry-run preview"""
        # Count videos to fetch captions for
        video_count = 0
        if hasattr(state, 'video_ids') and state.video_ids:
            video_count = len(state.video_ids)
        elif hasattr(state, 'downloaded_videos'):
            video_count = len(state.downloaded_videos)

        # Estimate caption fetch attempts (1 per video + retries based on config)
        caption_retry_config = getattr(config.download, 'caption_first', {})
        retry_budget = caption_retry_config.get('retry_budget', {}) if isinstance(caption_retry_config, dict) else {}
        max_attempts = retry_budget.get('max_attempts', 100) if isinstance(retry_budget, dict) else 100

        # Estimate: 1 initial attempt + retry budget per video
        estimated_attempts = min(video_count, max_attempts) + min(video_count * 0.2, max_attempts * 0.1)

        # Estimate cost (YouTube caption API is free, but we'll track attempts)
        # Caption fetch is essentially free (uses existing YouTube endpoints)
        estimated_cost = 0.0

        # Estimate duration (average ~0.5s per caption with retries)
        estimated_duration = video_count * 0.5

        return {
            'caption_fetch_attempts': int(estimated_attempts),
            'estimated_cost_usd': round(estimated_cost, 4),
            'estimated_duration_seconds': round(estimated_duration, 1),
        }

    # === Helper Methods ===

    def _log_coverage_gap_summary(
        self,
        caption_results: Dict[str, Any],
        video_durations: Dict[str, float],
    ) -> None:
        """Log a structured caption coverage gap summary after batch processing.

        US-78-005: Surfaces aggregate coverage stats at INFO level so coverage
        issues are visible without --verbose. Uses CoverageAnalysis from
        src/caption/models.py as the data source.

        Args:
            caption_results: Dict mapping video_id to caption result dicts.
            video_durations: Dict mapping video_id to duration in seconds.
        """
        from ..caption.models import CoverageAnalysis, CaptionSegment, analyze_caption_coverage

        analyses: List[tuple] = []  # (video_id, CoverageAnalysis)

        for video_id, result in caption_results.items():
            segments_data = result.get('segments')
            if not segments_data:
                continue
            video_duration = result.get('video_duration') or video_durations.get(video_id)
            if not video_duration or video_duration <= 0:
                continue

            # Reconstruct CaptionSegment objects from dicts
            segments = []
            for i, seg in enumerate(segments_data):
                if isinstance(seg, dict):
                    segments.append(CaptionSegment(
                        index=seg.get('index', i),
                        start_time=seg.get('start_time', 0.0),
                        end_time=seg.get('end_time', 0.0),
                        text=seg.get('text', ''),
                    ))
                elif isinstance(seg, CaptionSegment):
                    segments.append(seg)

            if not segments:
                continue

            analysis = analyze_caption_coverage(segments, video_duration)
            if analysis:
                analyses.append((video_id, analysis))

        if not analyses:
            logger.info("caption_coverage_gap_summary: no videos with coverage data")
            return

        total_videos = len(analyses)
        captioned_count = sum(1 for _, a in analyses if a.coverage_ratio > 0)
        avg_coverage = sum(a.coverage_ratio for _, a in analyses) / total_videos
        videos_with_large_gaps = [
            (vid, a) for vid, a in analyses if a.largest_gap_seconds > 10.0
        ]

        # Top 3 largest gaps across all videos
        sorted_by_gap = sorted(analyses, key=lambda x: x[1].largest_gap_seconds, reverse=True)
        top_3_gaps = sorted_by_gap[:3]

        summary = {
            'total_videos': total_videos,
            'captioned_count': captioned_count,
            'avg_coverage_ratio': round(avg_coverage, 3),
            'videos_with_gaps_over_10s': len(videos_with_large_gaps),
            'top_3_largest_gaps': [
                {'video_id': vid, 'gap_seconds': round(a.largest_gap_seconds, 1)}
                for vid, a in top_3_gaps
            ],
        }
        logger.info("caption_coverage_gap_summary %s", summary)

        # Print user-facing summary
        logger.info(f"\n  + Caption coverage summary (US-78-005):")
        logger.info(f"    - Videos with coverage data: {total_videos}")
        logger.info(f"    - Average coverage: {avg_coverage:.0%}")
        if videos_with_large_gaps:
            logger.info(f"    - Videos with gaps >10s: {len(videos_with_large_gaps)}")
        if top_3_gaps:
            gap_strs = [f"{vid}={a.largest_gap_seconds:.1f}s" for vid, a in top_3_gaps]
            logger.info(f"    - Largest gaps: {', '.join(gap_strs)}")

    def _validate_budget_for_batch(
        self,
        batch_size: int,
        retry_budget: Optional["CaptionRetryBudget"],
    ) -> None:
        """Validate retry budget settings for the given batch size.

        US-38-010: Proactive warning helps users understand potential budget
        issues before they cause failures.

        US-39-006: Enhanced to calculate required budget (batch_size * attempts_per_video)
        and provide actionable recommendations when budget may be insufficient.

        Logs warnings for:
        - required_budget > max_attempts AND auto_scale disabled: Budget may be insufficient
        - max_attempts < batch_size: Not enough attempts for 1 per video

        Args:
            batch_size: Number of videos to be fetched.
            retry_budget: The retry budget to validate (may be None).
        """
        if retry_budget is None or batch_size <= 0:
            return

        # US-39-006: Calculate required budget: batch_size * attempts_per_video
        attempts_per_video = retry_budget.attempts_per_video
        required_budget = int(batch_size * attempts_per_video + 0.5)  # Round up

        # US-39-006/US-42-006: Log WARNING if required > max_attempts AND auto_scale is disabled
        if not retry_budget.auto_scale and retry_budget.max_attempts > 0:
            if required_budget > retry_budget.max_attempts:
                logger.warning(
                    f"[US-42-006] Budget may be insufficient: {retry_budget.max_attempts} attempts "
                    f"for {batch_size} videos (required: {batch_size} * {attempts_per_video} = {required_budget}). "
                    f"Consider enabling auto_scale in config or increasing max_attempts"
                )

        # Legacy check: max_attempts < batch_size (not enough for 1 attempt each)
        if retry_budget.max_attempts > 0 and retry_budget.max_attempts < batch_size:
            logger.warning(
                f"Retry budget may be insufficient: max_attempts={retry_budget.max_attempts} "
                f"< batch_size={batch_size} (not enough for 1 attempt per video)"
            )

    def _get_video_ids(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> List[str]:
        """Get video IDs for caption fetching.

        In the simplified 7-stage pipeline, video IDs come directly from
        state.video_ids (populated by VIDEO_SEARCH stage).

        For backward compatibility, also checks legacy downloaded_videos
        and downloaded_audio fields.
        """
        # Primary: use video_ids from VIDEO_SEARCH stage
        if hasattr(state, 'video_ids') and state.video_ids:
            return list(state.video_ids)

        # Fallback: extract from legacy fields (backward compatibility)
        video_ids = set()

        # Check legacy audio-first mode downloads
        if hasattr(state, 'downloaded_audio'):
            for audio in state.downloaded_audio:
                video_id = getattr(audio, 'video_id', None)
                if video_id and len(video_id) == 11:
                    video_ids.add(video_id)

        # Check legacy downloaded videos
        if hasattr(state, 'downloaded_videos'):
            for video in state.downloaded_videos:
                video_id = self._extract_video_id(video)
                if video_id and len(video_id) == 11:
                    video_ids.add(video_id)

        return list(video_ids)

    def _get_video_durations(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, float]:
        """Get video ID to duration mapping for coverage calculation (US-004).

        In the simplified pipeline, durations come from video_search_results.
        For backward compatibility, also checks legacy downloaded_videos/audio.

        Returns:
            Dict mapping video_id to duration in seconds.
        """
        durations = {}

        # Primary: from video_search_results (simplified pipeline)
        if hasattr(state, 'video_search_results'):
            for result in state.video_search_results:
                video_id = getattr(result, 'video_id', None)
                duration = getattr(result, 'duration', 0.0)
                if video_id and len(video_id) == 11 and duration > 0:
                    durations[video_id] = duration

        # Fallback: from legacy downloaded_audio
        if hasattr(state, 'downloaded_audio'):
            for audio in state.downloaded_audio:
                video_id = getattr(audio, 'video_id', None)
                duration = getattr(audio, 'duration', 0.0)
                if video_id and len(video_id) == 11 and duration > 0:
                    durations[video_id] = duration

        # Fallback: from legacy downloaded_videos
        if hasattr(state, 'downloaded_videos'):
            for video in state.downloaded_videos:
                video_id = self._extract_video_id(video)
                duration = getattr(video, 'duration', 0.0)
                if video_id and len(video_id) == 11 and duration > 0:
                    durations[video_id] = duration

        return durations

    def _extract_video_id(self, video) -> Optional[str]:
        """Extract YouTube video ID from a DownloadedVideo.

        Checks attributes and filename patterns to find 11-char video ID.
        """
        import re

        # Check direct video_id attribute (if it has a truthy value)
        video_id = getattr(video, 'video_id', None)
        if video_id:
            return video_id

        # Check URL for video ID
        url = getattr(video, 'url', None)
        if url:
            match = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
            if match:
                return match.group(1)

        # Check filename (videos often named with video ID)
        file_path = getattr(video, 'file', None)
        if file_path:
            filename = Path(file_path).stem
            # YouTube IDs are exactly 11 alphanumeric chars with _-
            match = re.search(r'([A-Za-z0-9_-]{11})', filename)
            if match:
                return match.group(1)

        return None

    def _load_existing_captions(
        self,
        checkpoint: 'CheckpointManager'
    ) -> Dict[str, Dict[str, Any]]:
        """Load already-fetched captions from checkpoint.

        Used for resume functionality - skips videos that already have captions.
        """
        try:
            data = checkpoint.get_stage_data(self.name)
            if data:
                return data.get('caption_results', {})
        except Exception as e:
            logger.debug(f"Could not load existing captions: {e}")

        return {}

    def _save_intermediate_checkpoint(
        self,
        checkpoint: 'CheckpointManager',
        caption_results: Dict[str, Dict[str, Any]],
        metrics: Any = None
    ):
        """Save intermediate checkpoint during long fetch operations.

        Args:
            checkpoint: CheckpointManager instance.
            caption_results: Caption results so far.
            metrics: Optional CaptionMetrics instance for US-011.
        """
        try:
            checkpoint_data = {
                'caption_results': caption_results,
                'success_count': sum(1 for r in caption_results.values()
                                    if not r.get('unavailable') and not r.get('error')),
                'partial': True,
            }
            # US-011: Include metrics in intermediate checkpoint
            if metrics is not None:
                checkpoint_data['caption_metrics'] = metrics.to_dict()
            checkpoint.save_intermediate(self.name, checkpoint_data)
        except Exception as e:
            logger.debug(f"Intermediate checkpoint save failed: {e}")

    def _calculate_quality_distribution(
        self,
        caption_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate caption quality distribution (US-007).

        Args:
            caption_results: Dict of video_id -> caption result data

        Returns:
            Dict with counts for 'high', 'medium', 'low' quality
        """
        distribution = {'high': 0, 'medium': 0, 'low': 0}

        for result in caption_results.values():
            quality = result.get('caption_quality', 'low')
            if quality in distribution:
                distribution[quality] += 1
            else:
                distribution['low'] += 1  # Unknown quality counts as low

        return distribution

    def _populate_text_metadata(
        self,
        state: 'PipelineState',
        caption_results: Dict[str, Dict[str, Any]],
        config: 'Config' = None
    ):
        """Populate state.text_metadata from caption results.

        Converts caption segments to the format expected by the matching stage.
        Compatible with TranscriptSegment format used by TRANSCRIBE stage.

        US-007: Includes caption_quality field for matching confidence adjustment.
        US-008 Sprint 7: Includes timing_penalty for timing-based confidence adjustment.
        US-41-007: Defensive state validation before attribute access.
        US-70-008: When title_enriched_embeddings is enabled, adds embedding_text field
        with '[{video_title}] {text}' prefix for richer embedding context.
        """
        # US-41-007: Belt-and-suspenders defense - validate state at usage point
        # Import PipelineState here to avoid circular import issues
        from ..state import PipelineState

        if isinstance(state, PipelineState):
            # State is proper PipelineState - use validate_state_attributes()
            if hasattr(state, 'validate_state_attributes'):
                initialized_fields = state.validate_state_attributes()
                if initialized_fields:
                    logger.warning(
                        f"US-41-007: State validation initialized fields: {initialized_fields}. "
                        "This may indicate checkpoint was loaded from legacy format or "
                        "state was not properly constructed."
                    )
        else:
            # State is not a PipelineState instance (mock, dict-like, etc.)
            # Handle case where state lacks validate_state_attributes method
            if not hasattr(state, 'text_metadata') or state.text_metadata is None:
                state.text_metadata = []
                logger.warning(
                    f"US-41-007: Non-PipelineState object ({type(state).__name__}) missing "
                    "text_metadata attribute. Manually initialized to empty list."
                )

        text_metadata = []

        # US-70-008: Build title lookup and check config for title-enriched embeddings
        title_enriched = False
        chapter_enriched = False  # US-73-004: Chapter-enriched embedding text
        description_enriched = False  # US-75-011: Description keywords in embedding text
        title_lookup = {}
        # Default values when config is None (backward compatibility)
        title_enriched = False
        chapter_enriched = True
        description_enriched = False
        channel_enriched = True
        channel_reputation_enriched = True  # US-134-004: Channel reputation in embedding
        topic_enriched = True  # US-150-006: Topic enrichment from YouTube API
        topic_enrichment_factor = 0.15  # US-150-006: Weight for topic categories
        # US-111-006: Description keyword extraction config
        # US-111-008: Multi-signal embedding context enrichment factors
        ngram_enabled = True
        max_keywords_from_description = 5
        # Default enrichment factors (for backward compatibility when config not available)
        description_enrichment_factor = 0.3
        tags_enrichment_factor = 0.2
        chapters_enrichment_factor = 0.3
        if config is not None:
            ce = getattr(getattr(config, 'matching', None), 'context_enrichment', None)
            title_enriched = getattr(ce, 'title_enriched_embeddings', False)
            chapter_enriched = getattr(ce, 'chapter_enriched_embeddings', True)
            description_enriched = getattr(ce, 'description_enriched_embeddings', False)
            channel_enriched = getattr(ce, 'embed_channel_context', True)
            # US-134-004: Get channel reputation config for embedding enrichment
            channel_reputation_enriched = getattr(ce, 'embed_channel_reputation', True)
            # US-150-006: Get topic enrichment config from YouTube API
            topic_enriched = getattr(ce, 'topic_enrichment_enabled', True)
            topic_enrichment_factor = getattr(ce, 'topic_enrichment_factor', 0.15)
            # US-111-008: Get enrichment factors for weighted metadata signals
            description_enrichment_factor = getattr(ce, 'description_enrichment_factor', 0.3)
            tags_enrichment_factor = getattr(ce, 'tags_enrichment_factor', 0.2)
            chapters_enrichment_factor = getattr(ce, 'chapters_enrichment_factor', 0.3)
            # US-111-006: Get enhanced keyword extraction settings from matching config
            # Use safe access with type checking to handle mocks in tests
            mc = getattr(config, 'matching', None)
            if mc is not None and not isinstance(mc, type(None)):
                ngram_raw = getattr(mc, 'ngram_enabled', None)
                if isinstance(ngram_raw, bool):
                    ngram_enabled = ngram_raw
                max_kw_raw = getattr(mc, 'max_keywords_from_description', None)
                if isinstance(max_kw_raw, int) and max_kw_raw > 0:
                    max_keywords_from_description = max_kw_raw

        # US-95-008: Build channel lookup for embedding enrichment
        # US-134-004: Also build subscriber_count lookup for channel reputation in embeddings
        channel_lookup: Dict[str, str] = {}
        subscriber_lookup: Dict[str, int] = {}  # US-134-004: Channel subscriber count
        if title_enriched and hasattr(state, 'video_search_results'):
            for vsr in state.video_search_results:
                vid = getattr(vsr, 'video_id', None) if not isinstance(vsr, dict) else vsr.get('video_id')
                ttl = getattr(vsr, 'title', '') if not isinstance(vsr, dict) else vsr.get('title', '')
                ch = getattr(vsr, 'channel', '') if not isinstance(vsr, dict) else vsr.get('channel', '')
                # US-134-004: Get subscriber count for channel reputation
                sub_cnt = getattr(vsr, 'subscriber_count', None) if not isinstance(vsr, dict) else vsr.get('subscriber_count')
                if vid and ttl:
                    title_lookup[vid] = ttl
                if vid and ch:
                    channel_lookup[vid] = ch
                # US-134-004: Store subscriber count if available
                if vid and sub_cnt is not None:
                    subscriber_lookup[vid] = sub_cnt

        # US-72-002: Build VSR lookup for propagating caption metadata to state
        vsr_lookup: Dict[str, Any] = {}
        if hasattr(state, 'video_search_results'):
            for vsr in state.video_search_results:
                vid = getattr(vsr, 'video_id', None) if not isinstance(vsr, dict) else vsr.get('video_id')
                if vid:
                    vsr_lookup[vid] = vsr

        for video_id, result in caption_results.items():
            # Skip unavailable/errored captions
            if result.get('unavailable') or result.get('error'):
                continue

            # US-72-002: Propagate video_chapters and video_tags to VideoSearchResult
            chapters = result.get('video_chapters', [])
            tags = result.get('video_tags', [])
            if video_id in vsr_lookup:
                vsr = vsr_lookup[video_id]
                if isinstance(vsr, dict):
                    vsr['video_chapters'] = chapters
                    vsr['video_tags'] = tags
                else:
                    vsr.video_chapters = chapters
                    vsr.video_tags = tags

            segments = result.get('segments', [])
            language = result.get('language', 'en')
            is_auto = result.get('is_auto_generated', False)
            caption_quality = result.get('caption_quality', 'medium')  # US-007
            timing_penalty = result.get('timing_penalty', 1.0)  # US-008 Sprint 7
            # US-70-008: Get video title for this video
            video_title = title_lookup.get(video_id, '')
            # US-95-008: Get video channel for embedding enrichment
            video_channel = channel_lookup.get(video_id, '') if channel_enriched else ''
            # US-134-004: Get subscriber count for channel reputation in embedding
            video_subscriber_count = subscriber_lookup.get(video_id, 0) if channel_reputation_enriched else 0
            # US-134-004: Detect channel category from channel name and video content
            video_category = None
            if channel_reputation_enriched:
                video_category = self._detect_channel_category(
                    video_channel, tags, video_title
                )
            # US-150-006: Get topic details from VideoSearchResult for topic-aware matching
            video_topic_details = {}
            video_topic_categories = []  # US-150-006: Dedicated topic_categories field
            if topic_enriched and video_id in vsr_lookup:
                vsr = vsr_lookup[video_id]
                video_topic_details = getattr(vsr, 'topic_details', {}) if not isinstance(vsr, dict) else vsr.get('topic_details', {})
                # US-150-006: Also get dedicated topic_categories field if available
                video_topic_categories = getattr(vsr, 'topic_categories', []) if not isinstance(vsr, dict) else vsr.get('topic_categories', [])
            # US-75-003: Get video description from caption result
            video_description = result.get('video_description', '')

            # US-73-002: Map caption segments to video chapters
            chapter_map = self._map_segments_to_video_chapters(segments, chapters)

            for seg_idx, seg in enumerate(segments):
                seg_text = seg.get('text', '')
                ch_idx, ch_title = chapter_map.get(seg_idx, (-1, ''))
                entry = {
                    'text': seg_text,
                    'video_path': video_id,  # In caption-first mode, this is video ID
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'source_file': video_id,
                    # Caption-specific metadata
                    'caption_source': 'youtube',
                    'caption_language': language,
                    'caption_auto_generated': is_auto,
                    'caption_quality': caption_quality,  # US-007: Quality indicator
                    'timing_penalty': timing_penalty,  # US-008 Sprint 7: Timing penalty factor
                    # US-73-002: Chapter mapping
                    'chapter_index': ch_idx,
                    'chapter_title': ch_title,
                    # US-75-003: Video description from caption result
                    'video_description': video_description,
                    # US-150-006: Topic details from YouTube API for topic-aware matching
                    'topic_details': video_topic_details,
                }
                # US-70-008 / US-73-004 / US-75-011 / US-95-008: Build embedding_text with enrichments
                if title_enriched and video_title:
                    # US-126-010: Truncate long titles (>100 chars) to avoid overly long embedding text
                    MAX_TITLE_LENGTH = 100
                    display_title = video_title[:MAX_TITLE_LENGTH] if len(video_title) > MAX_TITLE_LENGTH else video_title
                    # Build enrichment prefix: [channel | title] or [channel | title | chapter]
                    # US-111-008: Use chapters_enrichment_factor to control chapter inclusion
                    prefix_parts = []
                    if channel_enriched and video_channel:
                        prefix_parts.append(video_channel)
                    prefix_parts.append(display_title)
                    if chapter_enriched and chapters_enrichment_factor > 0 and ch_title:
                        prefix_parts.append(ch_title)
                    prefix = ' | '.join(prefix_parts)
                    embed_text = f'[{prefix}] {seg_text}'
                    # US-75-011: Append description keywords when enabled
                    # US-111-006: Enhanced with n-gram extraction
                    # US-111-008: Use description_enrichment_factor to control inclusion
                    if description_enrichment_factor > 0 and description_enriched and video_description:
                        desc_kw = self._extract_description_keywords(
                            video_description,
                            max_keywords=max_keywords_from_description,
                            ngram_enabled=ngram_enabled,
                        )
                        if desc_kw:
                            embed_text = f'{embed_text} [desc: {" ".join(desc_kw)}]'
                    # US-111-008: Append video tags when tags_enrichment_factor > 0 and tags available
                    if tags_enrichment_factor > 0 and tags:
                        # Take top tags based on factor weight (scale by factor for more/less tags)
                        num_tags = max(1, int(len(tags) * tags_enrichment_factor))
                        top_tags = tags[:num_tags]
                        embed_text = f'{embed_text} [tags: {" ".join(top_tags)}]'
                    # US-134-004: Append channel reputation info when available (subscriber count)
                    if channel_reputation_enriched and video_subscriber_count > 0:
                        # Format subscriber count as readable string (e.g., "1.5M", "500K")
                        sub_str = self._format_subscriber_count(video_subscriber_count)
                        embed_text = f'{embed_text} [sub: {sub_str}]'
                    # US-134-004: Append channel category when detected
                    if channel_reputation_enriched and video_category:
                        embed_text = f'{embed_text} [cat: {video_category}]'
                    # US-150-006: Append topic categories from YouTube API when enabled
                    # Use dedicated topic_categories field first, fallback to topic_details
                    if topic_enriched and topic_enrichment_factor > 0:
                        topic_categories = video_topic_categories or video_topic_details.get('topic_categories', [])
                        if topic_categories:
                            # Extract topic names from URIs like "http://en.wikipedia.org/wiki/..."
                            topic_names = []
                            for tc in topic_categories:
                                if isinstance(tc, str) and '/' in tc:
                                    # Extract last part of URL as topic name
                                    topic_name = tc.rstrip('/').split('/')[-1].replace('_', ' ')
                                    topic_names.append(topic_name)
                                elif isinstance(tc, str):
                                    topic_names.append(tc)
                            if topic_names:
                                # Limit topics based on factor weight
                                num_topics = max(1, int(len(topic_names) * topic_enrichment_factor))
                                top_topics = topic_names[:num_topics]
                                embed_text = f'{embed_text} [topics: {" ".join(top_topics)}]'
                    entry['embedding_text'] = embed_text
                text_metadata.append(entry)

        # US-37-010/US-41-002/US-41-007: Final safety check before extending
        # This is a secondary fallback after the validate_state_attributes() call at the start
        if not hasattr(state, 'text_metadata') or state.text_metadata is None:
            state.text_metadata = []
            logger.warning(
                "US-41-007: text_metadata still missing after validation. "
                "Auto-initialized to empty list (secondary fallback)."
            )

        # Extend existing text_metadata (don't replace, as TRANSCRIBE may add more)
        state.text_metadata.extend(text_metadata)

        logger.info(f"Populated text_metadata with {len(text_metadata)} caption segments")

    # US-75-011: YouTube/generic stop words to filter from description keywords
    _DESCRIPTION_STOP_WORDS = frozenset({
        # English stop words
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
        'or', 'if', 'while', 'about', 'up', 'its', 'it', 'this', 'that',
        'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
        'him', 'his', 'she', 'her', 'they', 'them', 'their', 'what', 'which',
        'who', 'whom',
        # YouTube-specific words
        'subscribe', 'like', 'video', 'channel', 'watch', 'click', 'link',
        'description', 'comment', 'share', 'follow', 'instagram', 'twitter',
        'facebook', 'tiktok', 'patreon', 'merch', 'discount', 'code', 'http',
        'https', 'www', 'com',
    })

    @staticmethod
    def _format_subscriber_count(count: int) -> str:
        """Format subscriber count as readable string (e.g., "1.5M", "500K").

        US-134-004: Helper to format subscriber count for embedding text.

        Args:
            count: Subscriber count as integer

        Returns:
            Formatted string like "1.5M", "500K", or "100"
        """
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.0f}K"
        else:
            return str(count)

    @staticmethod
    def _detect_channel_category(channel_name: str, video_tags: List[str], video_title: str) -> Optional[str]:
        """Detect channel category/genre from channel name and video content.

        US-134-004: Simple keyword-based category detection for channel context enrichment.

        Args:
            channel_name: YouTube channel name
            video_tags: List of video tags
            video_title: Video title

        Returns:
            Detected category string (e.g., "gaming", "music", "tech") or None
        """
        # Category keywords to look for
        CATEGORY_KEYWORDS = {
            'gaming': ['game', 'gaming', 'play', 'gamer', 'let s play', 'walkthrough', 'esports'],
            'music': ['music', 'song', 'album', 'artist', 'band', 'concert', 'lyrics', 'cover'],
            'tech': ['tech', 'technology', 'review', 'unboxing', 'device', 'computer', 'phone', 'software'],
            'cooking': ['recipe', 'cook', 'food', 'kitchen', 'baking', 'chef', 'restaurant'],
            'fitness': ['workout', 'fitness', 'exercise', 'gym', 'health', 'yoga', 'training'],
            'education': ['tutorial', 'learn', 'course', 'lesson', 'education', 'teaching', 'how to'],
            'news': ['news', 'breaking', 'update', 'report', 'journalism', 'interview'],
            'comedy': ['comedy', 'funny', 'humor', 'joke', 'laugh', 'sketch', 'stand-up'],
            'sports': ['sports', 'football', 'basketball', 'soccer', 'baseball', 'nfl', 'nba'],
            'fashion': ['fashion', 'style', 'clothing', 'makeup', 'beauty', 'outfit'],
            'science': ['science', 'experiment', 'research', 'physics', 'chemistry', 'biology'],
            'travel': ['travel', 'trip', 'vacation', 'destination', 'adventure', 'tour'],
            'politics': ['politics', 'political', 'election', 'government', 'policy', 'debate'],
            'business': ['business', 'finance', 'invest', 'stock', 'entrepreneur', 'startup'],
            'movies': ['movie', 'film', 'cinema', 'trailer', 'review', 'actor', 'director'],
        }

        # Combine channel name and video content for detection
        search_text = ' '.join([
            channel_name.lower(),
            ' '.join(video_tags).lower() if video_tags else '',
            video_title.lower()
        ])

        # Find matching category
        detected = None
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in search_text:
                    detected = category
                    break
            if detected:
                break

        return detected

    @staticmethod
    def _extract_description_keywords(
        description: str,
        max_keywords: int = 5,
        ngram_enabled: bool = True,
        ngram_min_count: int = 1,
    ) -> List[str]:
        """Extract top content keywords from video description with TF-IDF and n-grams.

        US-75-011: Simple frequency-based extraction — no LLM needed.
        US-111-006: Enhanced with n-gram extraction (bigrams, trigrams) for better phrase capture.

        Args:
            description: Video description text.
            max_keywords: Maximum keywords to return (default 5).
            ngram_enabled: Enable bigram/trigram extraction (default True).
            ngram_min_count: Minimum frequency for n-grams (default 1).

        Returns:
            List of top keywords/phrases, lowercased and deduplicated.
        """
        import re
        if not description:
            return []

        # Tokenize: split on non-alphanumeric, keep words 3+ chars
        words = re.findall(r'[a-zA-Z]{3,}', description.lower())

        # Filter stop words and YouTube-specific words
        stop_words = CaptionStage._DESCRIPTION_STOP_WORDS
        filtered = [w for w in words if w not in stop_words]

        if not filtered:
            return []

        # Count unigram frequencies
        unigram_freq: Dict[str, int] = {}
        for w in filtered:
            unigram_freq[w] = unigram_freq.get(w, 0) + 1

        keywords_with_scores: List[tuple] = []

        # Add unigrams with frequency score
        for word, count in unigram_freq.items():
            # TF-IDF-like: higher score for more frequent terms
            score = count * 1.0
            keywords_with_scores.append((word, score, 'unigram'))

        # Extract n-grams if enabled
        if ngram_enabled and len(filtered) >= 2:
            # Generate bigrams
            bigrams = [' '.join(filtered[i:i+2]) for i in range(len(filtered) - 1)]
            bigram_freq: Dict[str, int] = {}
            for bg in bigrams:
                # Only count if both words are not stop words (already filtered)
                parts = bg.split()
                if len(parts) == 2 and all(p not in stop_words for p in parts):
                    bigram_freq[bg] = bigram_freq.get(bg, 0) + 1

            for bigram, count in bigram_freq.items():
                if count >= ngram_min_count:
                    # Bigrams get a slight boost for being more specific
                    score = count * 1.2
                    keywords_with_scores.append((bigram, score, 'bigram'))

            # Generate trigrams if we have enough words
            if len(filtered) >= 3:
                trigrams = [' '.join(filtered[i:i+3]) for i in range(len(filtered) - 2)]
                trigram_freq: Dict[str, int] = {}
                for tg in trigrams:
                    parts = tg.split()
                    if len(parts) == 3 and all(p not in stop_words for p in parts):
                        trigram_freq[tg] = trigram_freq.get(tg, 0) + 1

                for trigram, count in trigram_freq.items():
                    if count >= ngram_min_count:
                        # Trigrams get a higher boost for being most specific
                        score = count * 1.5
                        keywords_with_scores.append((trigram, score, 'trigram'))

        if not keywords_with_scores:
            return []

        # Sort by score (desc), then by type (unigram < bigram < trigram for same score)
        type_order = {'unigram': 0, 'bigram': 1, 'trigram': 2}
        sorted_keywords = sorted(
            keywords_with_scores,
            key=lambda x: (-x[1], type_order.get(x[2], 3))
        )

        # Deduplicate: keep first occurrence of each term (already sorted by score)
        seen: set = set()
        result: List[str] = []
        for kw, score, ngram_type in sorted_keywords:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
                if len(result) >= max_keywords:
                    break

        return result

    @staticmethod
    def _map_segments_to_video_chapters(
        segments: List[Dict[str, Any]],
        chapters_raw: List[Dict[str, Any]],
    ) -> Dict[int, tuple]:
        """Map caption segments to video chapters using temporal overlap.

        US-73-002: Converts raw chapter dicts to VideoChapter objects and
        delegates to map_segments_to_chapters for overlap calculation.

        Args:
            segments: List of caption segment dicts with 'start' and 'end' keys.
            chapters_raw: List of chapter dicts with 'title', 'start_time', 'end_time'.

        Returns:
            Dict mapping segment index to (chapter_index, chapter_title).
        """
        if not segments or not chapters_raw:
            return {i: (-1, '') for i in range(len(segments))}

        from ..chapter_detector.detector import VideoChapter
        from ..chapter_detector.mapping import map_segments_to_chapters

        # Convert raw chapter dicts to VideoChapter objects
        video_chapters = []
        for ch in chapters_raw:
            if isinstance(ch, VideoChapter):
                video_chapters.append(ch)
            else:
                video_chapters.append(VideoChapter.from_dict(ch))

        # Create lightweight segment wrappers with start_time/end_time attributes
        class _SegProxy:
            __slots__ = ('start_time', 'end_time')
            def __init__(self, s, e):
                self.start_time = s
                self.end_time = e

        seg_proxies = [
            _SegProxy(seg.get('start', 0), seg.get('end', 0))
            for seg in segments
        ]

        return map_segments_to_chapters(seg_proxies, video_chapters)

    def _validate_language_config(self, config: 'Config') -> None:
        """Validate language configuration at stage initialization (US-005).

        Validates that all language codes in caption_first config are valid
        ISO 639-1 codes and checks for configuration errors that would cause
        silent failures during caption fetching.

        This method should be called in __init__() to fail fast on invalid config,
        or at the start of run() if config wasn't provided at init.

        Args:
            config: Pipeline Config object.

        Raises:
            ConfigValidationError: If invalid language codes or duplicates found.
                Contains field name, invalid value, reason, and suggestion.

        Example issues detected:
            - Invalid code: "eng" instead of "en" (3 letters instead of 2)
            - Duplicate: ["es", "pt", "es"] has "es" twice
            - Redundant: preferred="en" with fallback=["en", "es"] has "en" twice
        """
        from ..caption_fetcher import validate_language_config, ConfigValidationError

        # Get caption config
        caption_config = getattr(config.download, 'caption_first', None)
        if caption_config is None or not getattr(caption_config, 'enabled', False):
            # Skip validation if caption-first mode is disabled
            self._config_validated = True
            return

        # Extract language settings
        preferred_lang = getattr(caption_config, 'preferred_language', 'en')
        fallback_langs = getattr(caption_config, 'fallback_languages', [])

        # Handle case where fallback_languages might be None
        if fallback_langs is None:
            fallback_langs = []

        # Run validation (raises ConfigValidationError on failure)
        logger.debug(
            f"Validating language config: preferred={preferred_lang}, "
            f"fallback={fallback_langs}"
        )
        validate_language_config(
            preferred_language=preferred_lang,
            fallback_languages=fallback_langs,
            raise_on_error=True  # Fail fast on invalid config
        )

        self._config_validated = True
        logger.debug("Language configuration validated successfully")
