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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

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

        # US-005: Validate language config at init if config provided
        if config is not None:
            self._validate_language_config(config)

    def _ensure_state_attributes(self, state: 'PipelineState') -> None:
        """Ensure state has required attributes with defensive initialization.

        US-37-010: Provides belt-and-suspenders protection for dynamically-added
        attributes, ensuring compatibility with both new PipelineState instances
        and legacy pipeline objects that may not have all attributes defined.

        This follows the pattern established in PipelineState.from_legacy_pipeline()
        where attributes are copied with getattr() defaults.
        """
        # Ensure text_metadata exists (required by _populate_text_metadata)
        if not hasattr(state, 'text_metadata'):
            state.text_metadata = []
            logger.debug("Initialized missing text_metadata attribute on state")

        # Ensure caption_results exists (stores caption data by video_id)
        if not hasattr(state, 'caption_results'):
            state.caption_results = {}
            logger.debug("Initialized missing caption_results attribute on state")

        # Ensure video_ids exists (input from VIDEO_SEARCH stage)
        if not hasattr(state, 'video_ids'):
            state.video_ids = []
            logger.debug("Initialized missing video_ids attribute on state")

        # Ensure video_search_results exists (full search metadata)
        if not hasattr(state, 'video_search_results'):
            state.video_search_results = []
            logger.debug("Initialized missing video_search_results attribute on state")

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
        """
        warnings = []

        try:
            # US-37-010: Ensure state has required attributes (defensive initialization)
            self._ensure_state_attributes(state)

            # Get caption config (caption fetching is always enabled)
            caption_config = getattr(config.download, 'caption_first', None)

            # US-005: Validate language config if not done at init
            if not self._config_validated:
                self._validate_language_config(config)

            print(f"\n  --- Stage: CAPTION (Fetch YouTube Captions) ---")

            # Get video IDs to fetch captions for
            video_ids = self._get_video_ids(state, config)

            if not video_ids:
                print("  ! No video IDs found for caption fetch")
                warnings.append("No video IDs available for caption fetch")
                return StageResult.ok({'skipped': True, 'reason': 'no_videos'}, warnings)

            print(f"  Found {len(video_ids)} video candidates")

            # Get preferred language from config
            preferred_lang = getattr(caption_config, 'preferred_language', 'en')
            prefer_manual = getattr(caption_config, 'prefer_human_captions', True)
            timeout = getattr(caption_config, 'timeout', 30)
            max_workers = getattr(caption_config, 'max_parallel_fetches', 4)

            # US-004: Get coverage threshold from config
            min_coverage_threshold = getattr(caption_config, 'min_coverage_threshold', 0.5)

            print(f"  Caption settings: language={preferred_lang}, prefer_manual={prefer_manual}, "
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

            self._fetcher = CaptionFetcher(
                config=config,
                impersonation_manager=impersonation_mgr,
                escalation_manager=escalation_mgr,
            )
            self._fetcher._timeout = timeout

            # Initialize metrics tracker (US-011, US-001: thread-safe)
            metrics = CaptionMetrics()

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
            retry_budget = None
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

            # US-002 Sprint 7: Initialize caption cache for adaptive format ordering
            caption_cache = CaptionCache(caption_config)

            # US-002 Sprint 7: Apply adaptive format ordering from historical success rates
            # This reorders preferred_formats based on what worked best in previous runs
            adaptive_enabled = getattr(caption_config, 'adaptive_format_order', True)
            if adaptive_enabled and caption_cache.enabled:
                new_order = self._fetcher.apply_adaptive_format_order(cache=caption_cache)
                if self._fetcher._using_adaptive_order:
                    print(f"  Using adaptive format order based on historical success rates")

            # US-004: Get video durations for coverage calculation
            video_durations = self._get_video_durations(state, config)

            # Check for already-fetched captions in checkpoint
            existing_captions = self._load_existing_captions(checkpoint)
            print(f"  Found {len(existing_captions)} captions in checkpoint")

            # US-005 Sprint 8: Check for batch checkpoint from aborted run
            batch_checkpoint_path = None
            batch_checkpoint = None
            project_dir = getattr(state, 'project_dir', None)
            if project_dir:
                batch_checkpoint_path = CaptionBatchCheckpoint.get_checkpoint_path(project_dir)
                batch_checkpoint = CaptionBatchCheckpoint.load(batch_checkpoint_path)
                if batch_checkpoint:
                    if batch_checkpoint.aborted:
                        print(f"  ! Found aborted batch checkpoint: {batch_checkpoint.success_count} fetched, "
                              f"{len(batch_checkpoint.remaining_video_ids)} remaining")
                        print(f"    Abort reason: {batch_checkpoint.abort_reason[:80]}...")
                    else:
                        print(f"  Found batch checkpoint: {batch_checkpoint.success_count} fetched")
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

                print(f"  Checking {len(ids_to_fetch)} videos for stream states...")
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
                        print(f"  ! Skipped {len(live_stream_ids)} live streams (will use transcription fallback)")
                    if upcoming_stream_ids:
                        action = "queued" if handle_upcoming == "queue" else "skipped"
                        print(f"  ! {action.capitalize()} {len(upcoming_stream_ids)} upcoming/premiere streams")

                # Store pending streams in state if queued
                if pending_streams:
                    state.pending_streams = pending_streams
                    logger.info(f"Added {len(pending_streams)} streams to pending queue")

            # US-008: Pre-check caption availability to filter out videos without captions
            # US-006 Sprint 7: Use batch pre-check by channel when enabled
            pre_check_enabled = getattr(caption_config, 'pre_check_availability', True)
            batch_precheck_enabled = getattr(caption_config, 'batch_precheck_by_channel', True)
            no_caption_ids = []

            if ids_to_fetch and pre_check_enabled:
                if batch_precheck_enabled and len(ids_to_fetch) > 1:
                    # US-006: Use batch pre-check with channel grouping
                    # This reduces API calls by grouping videos by channel and using
                    # representative samples when channel patterns have high confidence
                    from ..caption_fetcher import BatchPreCheckResult

                    confidence = getattr(caption_config, 'batch_precheck_confidence', 0.9)
                    min_samples = getattr(caption_config, 'batch_precheck_min_samples', 5)
                    sample_size = getattr(caption_config, 'batch_precheck_sample_size', 5)

                    print(f"  Batch pre-checking {len(ids_to_fetch)} videos (channel grouping)...")
                    batch_result = self._fetcher.batch_precheck_by_channel(
                        video_ids=ids_to_fetch,
                        cache=caption_cache,
                        metrics=metrics,
                        confidence_threshold=confidence,
                        min_samples_for_confidence=min_samples,
                        sample_size_per_channel=sample_size,
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
                        print(f"  + Batch pre-check saved {batch_result.api_calls_saved} API calls "
                              f"({batch_result.skipped_by_pattern} skipped by channel pattern)")

                else:
                    # Original individual pre-check (US-008)
                    print(f"  Pre-checking caption availability for {len(ids_to_fetch)} videos...")
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
                    print(f"  ! Pre-check: {len(no_caption_ids)} videos have no captions (will use transcription fallback)")

            if ids_to_fetch:
                print(f"  Fetching {len(ids_to_fetch)} new videos with {max_workers} parallel workers...")

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
                                print(f"\r{line}{' ' * padding}", end='', flush=True)
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
                                print(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                print(line)

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
                                print(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                print(line)

                        elif status == 'skipped':
                            reason = details.get('reason', 'unknown')
                            line = f"  [{idx}/{total}] {video_id}: skipped ({reason})"
                            # US-38-004: Add budget status when >50% consumed
                            budget_pct = details.get('budget_consumed_pct')
                            if budget_pct is not None and budget_pct > 50:
                                line += f" (budget: {budget_pct:.0f}%)"
                            if is_tty:
                                padding = max(0, last_line_length - len(line))
                                print(f"\r{line}{' ' * padding}")
                                last_line_length = 0
                            else:
                                print(line)

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
                if retry_budget:
                    if retry_budget.auto_scale:
                        logger.info(f"Retry budget auto_scale enabled, batch_size={batch_size}")
                        old_max = retry_budget.max_attempts
                        new_max = retry_budget.scale_to_batch_size(batch_size)
                        if new_max != old_max:
                            logger.info(f"Retry budget scaled: {old_max} -> {new_max} for {batch_size} videos")
                    else:
                        # Warn if auto_scale disabled and batch is large enough to risk exhaustion
                        if batch_size > retry_budget.max_attempts:
                            logger.warning(
                                f"Retry budget auto_scale DISABLED, max_attempts={retry_budget.max_attempts} "
                                f"may be insufficient for batch of {batch_size}"
                            )

                # US-001: Use batch fetch for parallel processing
                # US-005 Sprint 8: With checkpoint support for abort recovery
                # US-33-009: With circuit breaker for consecutive failure protection
                # US-34-002: With global rate limit coordinator for unified rate limiting
                try:
                    batch_results = self._fetcher.fetch_captions_batch(
                        video_ids=ids_to_fetch,
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
                    # US-005 Sprint 8: Save final checkpoint on success
                    if batch_checkpoint and batch_checkpoint_path:
                        batch_checkpoint.save(batch_checkpoint_path)
                        logger.info(f"Batch checkpoint saved: {batch_checkpoint.success_count} successes")
                except ErrorPatternAbortError as e:
                    # US-005 Sprint 8: Checkpoint already updated in fetch_captions_batch
                    # Save checkpoint to disk before re-raising
                    if batch_checkpoint and batch_checkpoint_path:
                        batch_checkpoint.save(batch_checkpoint_path)
                        print(f"  ! Batch aborted: checkpoint saved with {batch_checkpoint.success_count} results")
                        print(f"    Resume by running the pipeline again with --resume")
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

                        # Success - convert to serializable dict
                        caption_results[video_id] = {
                            'video_id': video_id,
                            'segments': [seg.to_dict() for seg in result.segments],
                            'language': result.language,
                            'is_auto_generated': result.is_auto_generated,
                            'format_source': result.format_source,
                            'segment_count': len(result.segments),
                            'caption_quality': result.caption_quality,
                            'video_duration': video_duration,  # US-004
                            'coverage_ratio': coverage_ratio,  # US-004
                            'timing_penalty': timing_penalty,  # US-008 Sprint 7
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
            success_count = sum(
                1 for r in caption_results.values()
                if not r.get('unavailable') and not r.get('error') and not r.get('skipped')
                and r.get('segment_count', 0) > 0
            ) - skip_count  # Don't double-count cached entries
            fail_count = sum(
                1 for r in caption_results.values()
                if r.get('unavailable') or r.get('error')
            )
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

            # Store caption data in state.text_metadata for matching
            self._populate_text_metadata(state, caption_results)

            # Calculate quality distribution (US-007)
            quality_distribution = self._calculate_quality_distribution(caption_results)
            human_count = sum(1 for r in caption_results.values()
                             if not r.get('is_auto_generated') and not r.get('unavailable')
                             and not r.get('error') and not r.get('skipped'))
            auto_count = sum(1 for r in caption_results.values()
                            if r.get('is_auto_generated') and not r.get('unavailable')
                            and not r.get('error') and not r.get('skipped'))

            # Summary
            print(f"\n  + Caption fetch complete:")
            print(f"    - Success: {success_count} videos (new), {skip_count} videos (cached)")
            print(f"    - Unavailable/Error: {fail_count} videos")
            # US-002: Report skipped live streams
            if skipped_live_count > 0:
                print(f"    - Skipped live streams: {skipped_live_count} videos")
            # US-008: Report pre-check filtered videos
            if pre_check_unavailable_count > 0:
                print(f"    - Pre-check filtered: {pre_check_unavailable_count} videos (no captions)")
            # US-007: Report caption quality distribution
            print(f"    - Caption sources: {human_count} human, {auto_count} auto, {fail_count} fallback")
            print(f"    - Quality distribution: {quality_distribution['high']} high, "
                  f"{quality_distribution['medium']} medium, {quality_distribution['low']} low")

            # US-011: Display metrics summary
            print(f"\n  + Caption metrics (US-011):")
            for line in metrics.summary().split('\n'):
                print(f"    {line}")

            # Fallback warnings (failed + skipped live streams)
            fallback_count = fail_count + skipped_live_count
            if fallback_count > 0 and getattr(caption_config, 'fallback_to_transcription', True):
                print(f"    - {fallback_count} videos will use Whisper transcription fallback")
                if fail_count > 0:
                    warnings.append(f"{fail_count} videos require transcription fallback (unavailable)")
                if skipped_live_count > 0:
                    warnings.append(f"{skipped_live_count} live streams require transcription fallback")

            # US-004: Coverage warnings for low coverage videos
            low_coverage_count = len(metrics.low_coverage_videos)
            if low_coverage_count > 0:
                print(f"    - Low coverage (<{min_coverage_threshold:.0%}): {low_coverage_count} videos")
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
                print(f"    - Slowest fetches: {slowest_str}")

            # US-33-009: Print circuit breaker stats if used
            if circuit_breaker and circuit_breaker.is_enabled:
                cb_stats = circuit_breaker.get_stats()
                if cb_stats['total_trips'] > 0 or cb_stats['consecutive_failures'] > 0:
                    print(f"    - Circuit breaker: {cb_stats['total_trips']} trips, "
                          f"{cb_stats['total_paused_seconds']:.1f}s total pause, "
                          f"{cb_stats['consecutive_failures']} recent failures")

            # US-33-010: Print retry budget stats if used
            if retry_budget:
                rb_summary = retry_budget.get_summary()
                if rb_summary['attempts'] > 0 or rb_summary['videos_skipped'] > 0:
                    print(f"    - Retry budget: {rb_summary['attempts']} attempts, "
                          f"{rb_summary['failures']} failures, "
                          f"{rb_summary['backoff_time_spent']:.1f}s backoff, "
                          f"{rb_summary['videos_skipped']} skipped")
                    if rb_summary['is_exhausted']:
                        print(f"    ! Retry budget EXHAUSTED - remaining videos skipped")

            # US-002 Sprint 7: Save format statistics for cross-run learning
            # This enables adaptive format ordering in future runs
            if caption_cache.enabled and metrics.format_success_counts:
                if caption_cache.save_format_statistics(metrics.format_success_counts):
                    logger.info(f"Saved format statistics: {metrics.format_success_counts}")
                else:
                    logger.warning("Failed to save format statistics to cache")

            # Prepare checkpoint data (US-007: include quality stats, US-011: include metrics)
            checkpoint_data = {
                'caption_results': caption_results,
                'success_count': success_count,
                'skip_count': skip_count,
                'fail_count': fail_count,
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
            }

            # US-37-007: Save retry budget state for resume support
            if retry_budget:
                checkpoint_data['retry_budget'] = retry_budget.to_dict()

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            logger.error(f"Could not import caption_fetcher: {e}")
            return StageResult.fail(f"Caption fetcher not available: {e}", warnings)
        except Exception as e:
            logger.exception(f"Caption stage failed: {e}")
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
        """Restore caption stage from checkpoint."""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                # No stage data means CAPTION stage had nothing to do
                # (e.g., caption-first mode was disabled) - this is a valid no-op
                logger.info("Restored CAPTION: no caption data (caption-first may be disabled)")
                return True

            # Restore caption results to state
            caption_results = data.get('caption_results', {})
            if caption_results:
                self._populate_text_metadata(state, caption_results)

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
            else:
                # Stage data exists but no caption results - valid empty case
                logger.info("Restored CAPTION: 0 videos (no captions fetched)")

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

    # === Helper Methods ===

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
        caption_results: Dict[str, Dict[str, Any]]
    ):
        """Populate state.text_metadata from caption results.

        Converts caption segments to the format expected by the matching stage.
        Compatible with TranscriptSegment format used by TRANSCRIBE stage.

        US-007: Includes caption_quality field for matching confidence adjustment.
        US-008 Sprint 7: Includes timing_penalty for timing-based confidence adjustment.
        """
        text_metadata = []

        for video_id, result in caption_results.items():
            # Skip unavailable/errored captions
            if result.get('unavailable') or result.get('error'):
                continue

            segments = result.get('segments', [])
            language = result.get('language', 'en')
            is_auto = result.get('is_auto_generated', False)
            caption_quality = result.get('caption_quality', 'medium')  # US-007
            timing_penalty = result.get('timing_penalty', 1.0)  # US-008 Sprint 7

            for seg in segments:
                text_metadata.append({
                    'text': seg.get('text', ''),
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
                })

        # US-37-010: Defensive check before extending (belt-and-suspenders)
        if not hasattr(state, 'text_metadata'):
            state.text_metadata = []
            logger.debug("Initialized missing text_metadata in _populate_text_metadata")

        # Extend existing text_metadata (don't replace, as TRANSCRIBE may add more)
        state.text_metadata.extend(text_metadata)

        logger.info(f"Populated text_metadata with {len(text_metadata)} caption segments")

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
