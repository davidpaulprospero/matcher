"""
Iterative Match Stage - Multi-Pass Gap Filling

Stage 5 of the simplified 7-stage pipeline. Runs after MATCH to achieve
90%+ confidence across all segments while enforcing source spacing rules.

Features:
- Identifies gaps (low confidence or source spacing violation)
- Locks high-confidence matches for preservation
- Multi-strategy query generation (voiceover text, similar-to-locked, entity/topic)
- Progressive query refinement on subsequent passes
- Gap pattern analysis for smarter search
- Query learning database for cross-project improvement
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from . import Stage, StageResult, StageMetrics, register_stage, validate_required_state_attrs
from ..downloader.search_cache import SearchResultsCache
from ..logging_templates import (
    log_stage_start,
    log_stage_complete,
    log_stage_skip,
    log_error_with_context,
    log_progress,
    log_match_context,
)
from ..matching.serialization import serialize_match_for_iterative_stage, is_empty_source
from ..utils import extract_video_id

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, Match

logger = logging.getLogger(__name__)


@dataclass
class LockedMatch:
    """A segment with a locked (high-confidence, spacing-valid) match."""
    segment_index: int
    video_id: str
    confidence: float
    position: float  # Timeline position (start time)
    title: str = ""
    source_file: str = ""


@dataclass
class GapSegment:
    """A segment that needs a better match."""
    segment_index: int
    confidence: float
    voiceover_text: str
    position: float
    current_video_id: str = ""
    reason: str = ""  # Why it's a gap: 'low_confidence', 'spacing_violation'


@dataclass
class PassMetrics:
    """Metrics for a single iterative pass."""
    pass_number: int
    initial_gaps: int
    final_gaps: int
    gaps_filled: int
    new_videos_found: int
    queries_executed: int
    duration_seconds: float = 0.0
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)
    # US-94-011: Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0


@register_stage
class IterativeMatchStage(Stage):
    """
    Fill matching gaps with iterative search passes.

    Inputs:
        - state.matches: Initial matches from MATCH stage
        - state.voiceover_segments: Voiceover segments to match
        - state.text_metadata: Video segment metadata

    Outputs:
        - state.matches: Updated matches with gaps filled
        - Checkpoint data with iteration metrics

    Configuration (config.iterative_matching):
        - enabled: Enable/disable stage
        - target_confidence: Confidence threshold for locking (0.90)
        - source_spacing_seconds: Minimum seconds between same source (300)
        - max_iterations: Maximum passes to attempt (5)
        - min_gap_percentage: Stop early if below this (0.05)
    """

    name = "ITERATIVE_MATCH"
    description = "Fill matching gaps with iterative search passes"
    DEPENDS_ON = ['MATCH']
    PRODUCES = ['matches']

    def __init__(self):
        """Initialize the stage with cookie rotator."""
        super().__init__()
        self._cookie_rotator = None
        # Cross-pass tracking (reset per run)
        self._fetched_video_ids: Set[str] = set()
        self._used_queries: Set[str] = set()
        # US-101-009: Smart query retry tracking
        # Maps query_key -> {'query': str, 'strategy': str, 'retry_count': int, 'gap_indices': list}
        self._failed_queries: Dict[str, Dict[str, Any]] = {}

    def _get_cookie_rotator(self, config: 'Config'):
        """Get or initialize the cookie rotator for YouTube authentication."""
        if self._cookie_rotator is None:
            from ..downloader.cookie_rotator import CookieRotator
            cookie_rotation_config = getattr(config.download, 'cookie_rotation', None)
            if cookie_rotation_config and getattr(cookie_rotation_config, 'enabled', False):
                self._cookie_rotator = CookieRotator(cookie_rotation_config)
        return self._cookie_rotator

    def _get_cookie_args(self, config: 'Config', log_usage: bool = False) -> List[str]:
        """
        Get yt-dlp cookie arguments for YouTube authentication.

        Uses CookieRotator if enabled (for rotation), otherwise falls back
        to static cookie configuration from get_cookies_args.

        Args:
            config: Pipeline config
            log_usage: If True, log which cookie source is being used

        Returns:
            List of cookie arguments for yt-dlp (empty if no cookies configured)
        """
        # Try cookie rotator first (for rotation support)
        rotator = self._get_cookie_rotator(config)
        if rotator and rotator.is_enabled:
            current_cookie = rotator.get_current_cookie()
            if current_cookie:
                if log_usage:
                    from pathlib import Path
                    cookie_name = Path(current_cookie).stem
                    logger.info(f"Using rotated cookie: {cookie_name}")
                return ['--cookies', current_cookie]

        # Fall back to static cookie configuration
        from ..downloader.utils import get_cookies_args
        static_args = get_cookies_args(config)
        if log_usage:
            if static_args:
                logger.info(f"Using static cookie config")
            else:
                logger.debug("No cookies configured for iterative match")
        return static_args

    def _invalidate_cache_on_config_change(
        self,
        search_cache: 'SearchResultsCache',
        current_ttl: int
    ) -> None:
        """
        Invalidate cache if config has changed.

        US-94-011: Clears cache when TTL config changes, ensuring fresh results
        when user modifies caching behavior.

        Args:
            search_cache: The search cache to check
            current_ttl: Current TTL from config
        """
        import os
        import json
        from pathlib import Path

        # Check for cached TTL in a marker file
        cache_marker = search_cache.cache_dir / "config_marker.json"

        if cache_marker.exists():
            try:
                with open(cache_marker, 'r') as f:
                    marker = json.load(f)
                cached_ttl = marker.get('ttl_hours', -1)
                if cached_ttl != current_ttl:
                    # TTL changed - invalidate cache
                    logger.info(f"Cache TTL changed from {cached_ttl}h to {current_ttl}h - invalidating cache")
                    search_cache.clear()
                    marker = {'ttl_hours': current_ttl}
                    with open(cache_marker, 'w') as f:
                        json.dump(marker, f)
            except (json.JSONDecodeError, IOError):
                # Invalid marker - reset it
                marker = {'ttl_hours': current_ttl}
                try:
                    with open(cache_marker, 'w') as f:
                        json.dump(marker, f)
                except IOError:
                    pass
        else:
            # First run - create marker
            marker = {'ttl_hours': current_ttl}
            try:
                with open(cache_marker, 'w') as f:
                    json.dump(marker, f)
            except IOError:
                pass

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the iterative matching stage.

        US-39-009: Validates state type and converts legacy objects if needed.
        US-40-008: Validates required state attributes exist.
        """
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        # US-39-009: Validate state type at stage entry
        state = self._validate_state_type(state)

        # US-40-008: Validate required attributes exist
        validate_required_state_attrs(
            state,
            ['matches', 'text_metadata'],
            self.name
        )

        # Check if test mode skip_iterative is enabled
        if getattr(config, '_test_mode_skip_iterative', False):
            log_stage_skip(logger, self.name, reason="test_mode_skip")
            return StageResult.ok({
                'skipped': True,
                'reason': 'test_mode_skip'
            })

        # US-40-009: Pre-check candidate pool size
        candidate_count = len(state.text_metadata) if state.text_metadata else 0
        logger.info(f"IterativeMatch starting with {candidate_count} candidates")

        if candidate_count == 0:
            log_stage_skip(logger, self.name, reason="no_candidates_available", candidate_count=0)
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_candidates',
                'candidate_count': 0
            })

        warnings = []

        # Get config (with fallback defaults)
        iter_config = getattr(config, 'iterative_matching', None)
        if not iter_config:
            logger.info("iterative_matching config not found, using defaults")
            from ..config.sections.iterative_matching import IterativeMatchingConfig
            iter_config = IterativeMatchingConfig()

        # Check if enabled
        if not getattr(iter_config, 'enabled', True):
            log_stage_skip(logger, self.name, reason="disabled_in_config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'disabled'
            })

        # Check if test mode skip is enabled
        test_mode_config = getattr(config, 'test_mode', None)
        if test_mode_config and getattr(test_mode_config, 'skip_iterative', False):
            log_stage_skip(logger, self.name, reason="test_mode_skip")
            return StageResult.ok({
                'skipped': True,
                'reason': 'test_mode_skip',
                'test_mode': True
            })

        # Validate inputs
        if not state.matches:
            log_stage_skip(logger, self.name, reason="no_matches")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_matches'
            })

        if not state.voiceover_segments:
            log_stage_skip(logger, self.name, reason="no_voiceover_segments")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_voiceover'
            })

        # Get segment time range for logging
        if state.voiceover_segments:
            first_seg = state.voiceover_segments[0]
            last_seg = state.voiceover_segments[-1]
            seg_start = getattr(first_seg, 'start', 0)
            seg_end = getattr(last_seg, 'end', 0)
            seg_time_range = f"{seg_start:.1f}s-{seg_end:.1f}s"
        else:
            seg_time_range = "N/A"

        log_stage_start(
            logger,
            self.name,
            total_segments=len(state.voiceover_segments),
            time_range=seg_time_range,
            target_confidence=target_conf,
            max_iterations=max_iterations
        )

        try:
            # Get thresholds from config
            target_conf = getattr(iter_config, 'target_confidence', 0.90)
            source_spacing = getattr(iter_config, 'source_spacing_seconds', 300.0)
            max_iterations = getattr(iter_config, 'max_iterations', 5)
            min_gap_pct = getattr(iter_config, 'min_gap_percentage', 0.05)

            logger.info(f"Target confidence: {target_conf:.0%}, source spacing: {source_spacing:.0f}s, max iterations: {max_iterations}")

            # Initialize learning DB
            learning_db = None
            if getattr(iter_config, 'enable_query_learning', True):
                from ..iterative_match import QueryLearningDB
                db_path = getattr(iter_config, 'learning_db_path', '.cache/query_learning.json')
                learning_db = QueryLearningDB(str(Path(state.voiceover_path).parent / db_path))

            # Run iterative passes
            all_pass_metrics = []
            all_gap_pattern_logs = []  # US-63-012: Track gap patterns across passes
            total_start = time.time()
            gap_analysis = None  # Initialize before loop for early break case

            # Reset cross-pass tracking for this run
            self._fetched_video_ids = set()
            self._used_queries = set()
            # US-101-009: Reset failed queries tracking
            self._failed_queries = {}

            # Initialize local embedding storage (no longer stored on PipelineState)
            self._embeddings = None
            self._embedding_index = None
            # Track the global text_metadata index where self._embeddings[0] starts
            self._embedding_global_offset = None

            # US-89-006: Check for saved pass state from previous run
            resume_info = self._check_for_pass_checkpoint(checkpoint, max_iterations)
            start_pass_num = 1
            resumed_from_pass = False

            if resume_info:
                logger.info(f"Resuming from pass {resume_info['pass_num']}/{max_iterations}, {resume_info['gaps_remaining']} gaps remaining, {len(resume_info.get('used_queries', []))} queries already used")

                # Restore state
                start_pass_num = resume_info['pass_num']
                self._used_queries = set(resume_info.get('used_queries', []))
                self._fetched_video_ids = set(resume_info.get('fetched_video_ids', []))

                # US-89-006: Handle budget exhaustion on resume
                # If we resumed and have already used significant budget, check if we should continue
                total_queries_used = len(self._used_queries)
                budget_check = getattr(iter_config, 'resume_budget_check', True)
                if budget_check:
                    max_queries = getattr(iter_config, 'max_queries_per_run', 100)
                    if total_queries_used >= max_queries:
                        logger.warning(f"Budget exhausted on previous run ({total_queries_used} queries used), stopping")
                        # Restore matches from checkpoint and return
                        from ..state import restore_matches_from_dicts
                        saved_data = checkpoint.get_stage_data(self.name)
                        if saved_data and 'matches' in saved_data:
                            restored = restore_matches_from_dicts(
                                saved_data['matches'],
                                default_strategy='iterative_restored',
                                logger_instance=logger
                            )
                            if restored:
                                state.matches = restored
                        return StageResult.ok({
                            'resumed': True,
                            'passes_completed': resume_info['pass_num'],
                            'reason': 'budget_exhausted',
                            'queries_used': total_queries_used,
                        })

                resumed_from_pass = True

            # US-101-006: Initialize query budget tracking
            total_queries_used = len(self._used_queries)
            queries_per_pass: List[int] = []  # Track queries used per pass
            max_queries_per_pass = getattr(iter_config, 'max_queries_per_pass', 20)
            max_total_queries = getattr(iter_config, 'max_queries_per_run', 100)
            budget_warning_threshold = getattr(iter_config, 'budget_warning_threshold', 0.8)

            for pass_num in range(start_pass_num, max_iterations + 1):
                pass_start = time.time()
                queries_this_pass = 0  # Reset per-pass counter

                # US-101-006: Budget check before each search iteration
                total_queries_used = len(self._used_queries)

                # Check total budget exhaustion
                if total_queries_used >= max_total_queries:
                    logger.warning(f"Total query budget exhausted ({total_queries_used}/{max_total_queries})")
                    self._log_budget_summary(
                        total_queries_used, max_total_queries,
                        queries_per_pass, max_queries_per_pass
                    )
                    break

                # Log warning when approaching budget limit (80% threshold)
                if total_queries_used >= max_total_queries * budget_warning_threshold:
                    remaining = max_total_queries - total_queries_used
                    logger.warning(f"Approaching budget limit: {total_queries_used}/{max_total_queries} ({remaining} remaining)")

                logger.info(f"[ITERATIVE] Starting pass {pass_num}/{max_iterations}")

                # Log progress at start of each pass
                progress_pct = (pass_num / max_iterations) * 100 if max_iterations > 0 else 0
                log_progress(
                    logger,
                    self.name,
                    progress_pct=progress_pct,
                    current=pass_num,
                    total=max_iterations,
                    gaps_remaining=gap_count
                )

                # 1. Identify gaps and locks
                locked, gaps = self._identify_gaps_and_locks(
                    state.matches,
                    state.voiceover_segments,
                    target_conf,
                    source_spacing
                )

                gap_count = len(gaps)
                total_count = len(state.voiceover_segments)
                gap_pct = gap_count / total_count if total_count > 0 else 0

                # US-165-003: Store initial gap count for resolution logging
                initial_gap_count_for_resolution = gap_count

                # Log gap segment indices for debugging
                gap_indices = [g.segment_index for g in gaps] if gaps else []
                logger.info(f"[ITERATIVE] pass={pass_num} Locked: {len(locked)}, gaps: {gap_count} ({gap_pct:.1%}), gap_indices={gap_indices[:5]}{'...' if len(gap_indices) > 5 else ''}")

                # US-165-003: Log gap detection results with segment ranges
                # US-166-012: Track retry counts per gap segment (how many passes attempted this gap)
                gap_retry_counts: Dict[int, int] = {}
                for gap in gaps:
                    # A gap has been retried if it's appeared in previous passes
                    # This is tracked by checking if the segment was in gaps in prior passes
                    gap_retry_counts[gap.segment_index] = pass_num - 1  # Pass 1 = 0 retries
                log_gap_detection_results(gaps, pass_num, logger, gap_retry_counts)

                # US-159-006: Log gap analysis progress with structured logging
                logger.info("gap_analysis_progress", extra={
                    "pass": pass_num,
                    "total_gaps_found": gap_count,
                    "total_segments": total_count,
                    "gap_percentage": round(gap_pct * 100, 1),
                    "locked_count": len(locked),
                })

                # 2. Check stop conditions
                if gap_count == 0:
                    logger.info("No gaps remaining, stopping early")
                    break

                if gap_pct < min_gap_pct:
                    logger.info(f"Below {min_gap_pct:.0%} threshold, stopping")
                    break

                # 3. Analyze gap patterns
                gap_analysis = None
                gap_pattern_log = None
                gap_segments = None
                if getattr(iter_config, 'analyze_gap_patterns', True):
                    from ..iterative_match import (
                        analyze_gaps,
                        GapSegment as GapSeg,
                        analyze_gap_patterns_for_logging,
                        log_gap_pattern_analysis,
                    )
                    from ..iterative_match.gap_analyzer import (
                        annotate_gaps_with_chapters,
                        log_gap_detection_results,
                        log_gap_resolution_results,
                    )

                    # US-94-007: Build segment duration map for gap prioritization
                    segment_duration_map = {}
                    if state.voiceover_segments:
                        for seg in state.voiceover_segments:
                            seg_idx = getattr(seg, 'index', None)
                            if seg_idx is not None:
                                seg_duration = getattr(seg, 'duration', 0.0)
                                if seg_duration == 0.0:
                                    seg_duration = getattr(seg, 'end', 0.0) - getattr(seg, 'start', 0.0)
                                segment_duration_map[seg_idx] = seg_duration

                    gap_segments = [
                        GapSeg(
                            segment_index=g.segment_index,
                            confidence=g.confidence,
                            voiceover_text=g.voiceover_text,
                            position=g.position,
                            duration=segment_duration_map.get(g.segment_index, 0.0)
                        )
                        for g in gaps
                    ]
                    gap_analysis = analyze_gaps(
                        gap_segments,
                        state,
                        state.extracted_entities
                    )
                    dominant = gap_analysis.get_dominant_pattern()
                    logger.debug(f"Dominant gap pattern: {dominant}")

                    # US-159-006: Log gap pattern details
                    logger.info("gap_pattern_details", extra={
                        "pass": pass_num,
                        "dominant_pattern": dominant,
                        "pattern_counts": dict(gap_analysis.pattern_counts),
                        "abstract_concepts_count": len(gap_analysis.abstract_concepts),
                        "proper_nouns_count": len(gap_analysis.proper_nouns),
                        "action_descriptions_count": len(gap_analysis.action_descriptions),
                        "locations_count": len(gap_analysis.locations),
                        "emotional_content_count": len(gap_analysis.emotional_content),
                    })

                    # US-63-012: Detailed gap pattern analysis logging
                    total_duration = 0.0
                    if state.voiceover_segments:
                        # Get total duration from last segment
                        last_seg = state.voiceover_segments[-1]
                        total_duration = getattr(last_seg, 'end', 0.0) or getattr(last_seg, 'end_time', 0.0)

                    gap_pattern_log = analyze_gap_patterns_for_logging(
                        gap_segments,
                        pass_num,
                        total_duration
                    )
                    log_gap_pattern_analysis(gap_pattern_log, logger)

                    # Print query hints to console
                    if gap_pattern_log.query_hints:
                        for hint in gap_pattern_log.query_hints[:3]:  # Limit to 3 hints
                            logger.debug(f"Query hint: {hint}")

                    # US-63-012: Track gap pattern log for checkpoint storage
                    all_gap_pattern_logs.append(gap_pattern_log)

                    # US-71-007: Apply chapter-aware gap prioritization
                    # US-94-007: Also apply duration-based gap prioritization
                    # US-127-008: Also apply intro/conclusion chapter priority boost
                    # Get duration_priority_weight from config (default 0.1)
                    # Get intro_conclusion_boost from config (default 0.2)
                    duration_priority_weight = getattr(
                        iter_config, 'duration_priority_weight', 0.1
                    )
                    intro_conclusion_boost = getattr(
                        iter_config, 'intro_conclusion_boost', 0.2
                    )
                    gap_segments = annotate_gaps_with_chapters(
                        gap_segments,
                        total_segments=total_count,
                        intro_boost=intro_conclusion_boost,
                        conclusion_boost=intro_conclusion_boost,
                        duration_priority_weight=duration_priority_weight,
                    )
                    # Reorder the local gaps list to match the priority order
                    gap_idx_order = [gs.segment_index for gs in gap_segments]
                    gap_by_idx = {g.segment_index: g for g in gaps}
                    gaps = [gap_by_idx[idx] for idx in gap_idx_order if idx in gap_by_idx]

                # 4. Generate search queries
                queries = self._generate_multi_strategy_queries(
                    gaps, locked, state, iter_config, pass_num, gap_analysis,
                    learning_db=learning_db, gap_segments=gap_segments,
                )

                if not queries:
                    logger.info("No queries generated, stopping")
                    warnings.append(f"Pass {pass_num}: No queries generated")
                    break

                # Progress logging for query generation
                log_progress(
                    logger,
                    self.name,
                    progress_pct=10.0,  # Query generation is ~10% of pass work
                    current=1,
                    total=5,
                    phase="query_generation",
                    queries_generated=len(queries),
                    gaps_to_cover=len(gaps)
                )

                logger.info(f"Generated {len(queries)} search queries (excluding {len(self._fetched_video_ids)} videos already fetched)")

                # US-159-006: Log query generation progress
                logger.info("query_generation_progress", extra={
                    "pass": pass_num,
                    "queries_generated": len(queries),
                    "videos_already_fetched": len(self._fetched_video_ids),
                })

                # US-101-010: Batch query optimization - deduplicate similar queries
                enable_batch_opt = getattr(iter_config, 'enable_batch_optimization', True)
                if enable_batch_opt:
                    queries, saved = self._deduplicate_queries(
                        queries,
                        enable_optimization=enable_batch_opt,
                        similarity_threshold=getattr(iter_config, 'query_similarity_threshold', 0.85)
                    )
                    if saved > 0:
                        logger.debug(f"Batch optimization: {saved} queries saved (now {len(queries)} unique)")
                        # US-159-006: Log queries optimized
                        logger.info("queries_optimized", extra={
                            "pass": pass_num,
                            "queries_saved": saved,
                            "queries_remaining": len(queries),
                        })

                # 5. Apply progressive refinement on subsequent passes
                if pass_num > 1 and getattr(iter_config, 'enable_progressive_refinement', True):
                    original_queries = [q['query'] for q in queries]
                    queries = self._refine_queries_progressive(
                        queries, pass_num, all_pass_metrics, learning_db
                    )
                    # US-166-012: Log query refinement with new search terms
                    refined_queries = [q['query'] for q in queries]
                    if refined_queries != original_queries:
                        logger.info(
                            f"[ITERATIVE] Pass {pass_num} query refinement: "
                            f"{len(queries)} queries refined"
                        )
                        # Log a few example refinements
                        for i, (orig, refined) in enumerate(zip(original_queries[:3], refined_queries[:3])):
                            if orig != refined:
                                logger.debug(f"[ITERATIVE] Query refinement example: '{orig}' -> '{refined}'")

                # US-94-008: Inject negative keywords to exclude irrelevant results
                enable_neg_kw = getattr(iter_config, 'enable_negative_keywords', True)
                if enable_neg_kw and learning_db:
                    neg_patterns = getattr(iter_config, 'negative_keyword_patterns', None)
                    for q in queries:
                        query_text = q['query']
                        enhanced_query = learning_db.inject_negative_keywords(
                            query_text,
                            negative_patterns=neg_patterns,
                            enable_learning=True
                        )
                        q['query'] = enhanced_query

                # 6. Execute searches with streaming/batched caption fetching
                # Process captions in batches to avoid overwhelming the pipeline
                batch_size = getattr(iter_config, 'caption_batch_size', 10)
                fetch_delay = getattr(iter_config, 'caption_fetch_delay', 0.5)
                
                all_new_candidates = []
                gaps_filled_total = 0
                
                # Search first to get video IDs (US-94-011: also returns cache metrics)
                video_ids, cache_hits, cache_misses = self._search_youtube_for_videos(
                    queries, state, config, iter_config
                )

                if video_ids:
                    logger.info(f"Found {len(video_ids)} new video candidates")

                    # US-159-006: Log iterative search results - videos found
                    logger.info("iterative_search_videos_found", extra={
                        "pass": pass_num,
                        "videos_found": len(video_ids),
                        "gap_count": len(gaps),
                        "cache_hits": cache_hits,
                        "cache_misses": cache_misses,
                    })

                    # Process captions in batches
                    for batch_start in range(0, len(video_ids), batch_size):
                        batch_end = min(batch_start + batch_size, len(video_ids))
                        batch_ids = video_ids[batch_start:batch_end]

                        # Progress logging for video re-search loop
                        batch_num = batch_start // batch_size + 1
                        total_batches = (len(video_ids) + batch_size - 1) // batch_size
                        log_progress(
                            logger,
                            self.name,
                            progress_pct=(batch_num / total_batches) * 100,
                            current=batch_num,
                            total=total_batches,
                            gaps_remaining=len(gaps) - gaps_filled_total,
                            videos_in_batch=len(batch_ids)
                        )

                        logger.info(f"Fetching captions batch {batch_num}/{total_batches} ({len(batch_ids)} videos)...")
                        
                        batch_candidates = self._fetch_captions_for_videos(
                            batch_ids, config, iter_config
                        )
                        
                        if batch_candidates:
                            all_new_candidates.extend(batch_candidates)
                            
                            # Try to fill gaps with this batch before continuing
                            gaps_filled = self._rematch_gaps(
                                gaps, locked, batch_candidates, state, config
                            )
                            gaps_filled_total += gaps_filled
                            
                            if gaps_filled > 0:
                                logger.debug(f"Filled {gaps_filled} gaps with this batch")
                            
                            # Early exit if all gaps are filled
                            remaining_gaps = len(gaps) - gaps_filled_total
                            if remaining_gaps <= 0:
                                logger.info("All gaps filled, stopping caption fetch early")
                                break
                        
                        # Small delay between batches to avoid rate limiting
                        if batch_end < len(video_ids) and fetch_delay > 0:
                            time.sleep(fetch_delay)
                
                new_candidates = all_new_candidates
                gaps_filled = gaps_filled_total
                
                # 7. Log final results
                if gaps_filled > 0:
                    logger.info(f"[ITERATIVE] pass={pass_num} Filled {gaps_filled} gaps with new matches")

                    # US-159-006: Log iterative search results - matches found per iteration
                    logger.info("iterative_search_matches_found", extra={
                        "pass": pass_num,
                        "matches_found": gaps_filled,
                        "total_gaps": len(gaps),
                        "new_candidates": len(new_candidates) if 'new_candidates' in dir() else 0,
                    })

                # US-165-003: Log gap resolution success/failure rates
                remaining_gaps = len(gaps)
                log_gap_resolution_results(
                    initial_gap_count_for_resolution,
                    remaining_gaps,
                    pass_num,
                    logger
                )

                # 8. Update learning DB
                if learning_db and gap_analysis:
                    self._update_query_learning(
                        queries, gaps_filled, gap_analysis, learning_db,
                        gap_segments=gap_segments
                    )

                    # US-159-006: Log query learning progress
                    learning_summary = learning_db.get_summary()
                    logger.info("query_learning_progress", extra={
                        "pass": pass_num,
                        "patterns_learned": learning_summary.get('patterns_learned', 0),
                        "templates_discovered": learning_summary.get('templates_discovered', 0),
                        "total_queries_recorded": learning_summary.get('total_queries_recorded', 0),
                        "total_gaps_filled": learning_summary.get('total_gaps_filled', 0),
                        "chapter_types_learned": learning_summary.get('chapter_types_learned', 0),
                    })

                # US-101-009: Track failed queries and execute retries if enabled
                enable_retry = getattr(iter_config, 'enable_smart_retry', True)

                # Always track queries from this pass for potential retry in next pass
                if enable_retry:
                    # Track which queries didn't fill gaps this pass
                    matched_indices = {m.segment_index for m in state.matches if m.confidence >= 0.3}
                    for q in queries:
                        query_key = q['query'].lower().strip()
                        gap_indices = q.get('gap_indices', [])
                        strategy = q.get('strategy', 'unknown')
                        # Check if this query's gaps were filled
                        gaps_still_unfilled = [g for g in gap_indices if g not in matched_indices]
                        if gaps_still_unfilled:
                            self._track_failed_query(
                                q['query'], strategy, gap_indices, gaps_filled=False
                            )
                        else:
                            self._track_failed_query(
                                q['query'], strategy, gap_indices, gaps_filled=True
                            )

                    # Execute retry queries from PREVIOUS passes if eligible
                    retry_queries = self._generate_retry_queries(
                        gaps, locked, state, iter_config, learning_db, pass_num
                    )

                    if retry_queries:
                        logger.info(f"Executing {len(retry_queries)} retry queries...")
                        # Execute retry searches
                        retry_video_ids, retry_cache_hits, retry_cache_misses = self._search_youtube_for_videos(
                            retry_queries, state, config, iter_config
                        )

                        if retry_video_ids:
                            # Fetch captions for retry videos
                            batch_size = getattr(iter_config, 'caption_batch_size', 10)
                            for batch_start in range(0, len(retry_video_ids), batch_size):
                                batch_end = min(batch_start + batch_size, len(retry_video_ids))
                                batch_ids = retry_video_ids[batch_start:batch_end]

                                batch_candidates = self._fetch_captions_for_videos(
                                    batch_ids, config, iter_config
                                )

                                if batch_candidates:
                                    all_new_candidates.extend(batch_candidates)
                                    # Try to fill gaps
                                    retry_gaps_filled = self._rematch_gaps(
                                        gaps, locked, batch_candidates, state, config
                                    )
                                    gaps_filled_total += retry_gaps_filled

                                    # Track retry results
                                    matched_indices = {m.segment_index for m in state.matches if m.confidence >= 0.3}
                                    for rq in retry_queries:
                                        rq_gaps = rq.get('gap_indices', [])
                                        rq_filled = [g for g in rq_gaps if g in matched_indices]
                                        if rq_filled:
                                            self._track_failed_query(
                                                rq['query'], rq.get('strategy', 'unknown'),
                                                rq_gaps, gaps_filled=True
                                            )

                                    if retry_gaps_filled > 0:
                                        logger.debug(f"Retry filled {retry_gaps_filled} gaps")

                        # Update gaps_filled for final metrics
                        gaps_filled = gaps_filled_total

                # Log retry summary at end of pass
                if self._failed_queries:
                    self._log_retry_summary()

                # Record pass metrics
                pass_duration = time.time() - pass_start
                pass_metrics = PassMetrics(
                    pass_number=pass_num,
                    initial_gaps=gap_count,
                    final_gaps=gap_count - gaps_filled,
                    gaps_filled=gaps_filled,
                    new_videos_found=len(new_candidates),
                    queries_executed=len(queries),
                    duration_seconds=pass_duration,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses
                )
                all_pass_metrics.append(pass_metrics)

                # Log iteration convergence (improvement delta)
                if len(all_pass_metrics) > 1:
                    prev_metrics = all_pass_metrics[-2]
                    improvement_delta = prev_metrics.final_gaps - pass_metrics.final_gaps
                    logger.info("iterative_convergence", extra={
                        "pass": pass_num,
                        "previous_final_gaps": prev_metrics.final_gaps,
                        "current_final_gaps": pass_metrics.final_gaps,
                        "improvement_delta": improvement_delta,
                        "gaps_filled_this_pass": gaps_filled,
                        "total_gaps_filled": sum(pm.gaps_filled for pm in all_pass_metrics),
                    })
                else:
                    # First pass - log initial convergence
                    logger.info("iterative_convergence", extra={
                        "pass": pass_num,
                        "initial_gaps": gap_count,
                        "gaps_filled_this_pass": gaps_filled,
                        "remaining_gaps": gap_count - gaps_filled,
                    })

                # US-101-008: Log strategy effectiveness after each pass
                if learning_db:
                    effectiveness = learning_db.get_effectiveness_summary()
                    total_q = effectiveness.get('total_queries', 0)
                    total_filled = effectiveness.get('total_gaps_filled', 0)
                    strategy_rates = effectiveness.get('overall_strategy_rates', {})

                    # Build strategy rate string
                    rate_parts = []
                    for strategy, rate in sorted(strategy_rates.items(), key=lambda x: x[1], reverse=True):
                        if total_q > 0:
                            rate_parts.append(f"{strategy}:{rate:.1%}")
                    rate_str = ", ".join(rate_parts) if rate_parts else "none"

                    logger.info(f"Strategy effectiveness: {total_q} queries, {total_filled} gaps filled | {rate_str}")

                # US-101-011: Log gap coverage progress report
                self._log_gap_coverage_report(
                    gaps=gaps,
                    pass_num=pass_num,
                    initial_gap_count=gap_count,
                    gaps_filled=gaps_filled,
                    iter_config=iter_config
                )

                # US-89-006: Save intermediate checkpoint after each pass
                # This allows resuming mid-iteration if pipeline is interrupted
                self._save_pass_checkpoint(
                    checkpoint=checkpoint,
                    pass_num=pass_num,
                    gaps=gaps,
                    state=state,
                    all_pass_metrics=all_pass_metrics,
                    used_queries=list(self._used_queries),
                    fetched_video_ids=list(self._fetched_video_ids),
                )

                # Check if no progress (configurable minimum pass before giving up)
                _min_pass = getattr(iter_config, 'no_progress_min_pass', None)
                no_progress_min_pass = _min_pass if isinstance(_min_pass, int) else 3
                if gaps_filled == 0 and pass_num >= no_progress_min_pass:
                    logger.info(f"No progress in pass {pass_num}, stopping")
                    # US-101-006: Track queries for this pass
                    queries_this_pass = len(self._used_queries) - total_queries_used
                    queries_per_pass.append(queries_this_pass)
                    total_queries_used = len(self._used_queries)
                    break

                # US-101-006: Track queries for this pass
                queries_this_pass = len(self._used_queries) - total_queries_used
                queries_per_pass.append(queries_this_pass)

                # US-101-006: Check per-pass budget
                if queries_this_pass >= max_queries_per_pass:
                    remaining = max_total_queries - total_queries_used
                    logger.warning(f"Per-pass budget reached ({queries_this_pass}/{max_queries_per_pass})")
                    if remaining <= 0:
                        logger.warning("Total budget exhausted, stopping")
                        self._log_budget_summary(
                            total_queries_used, max_total_queries,
                            queries_per_pass, max_queries_per_pass
                        )
                        break

            # Save learning DB
            if learning_db:
                learning_db.save()

            # US-166-012: Warning if max iterations exceeded with gaps remaining
            passes_completed = len(all_pass_metrics)
            if passes_completed >= max_iterations:
                final_locked_check, final_gaps_check = self._identify_gaps_and_locks(
                    state.matches,
                    state.voiceover_segments,
                    target_conf,
                    source_spacing
                )
                remaining_gaps = len(final_gaps_check)
                if remaining_gaps > 0:
                    logger.warning(
                        f"[ITERATIVE] Max iterations ({max_iterations}) reached with "
                        f"{remaining_gaps} gaps remaining. Consider increasing max_iterations "
                        f"or adjusting target_confidence threshold."
                    )

            # Final summary
            total_duration = time.time() - total_start

            # Calculate final stats
            final_locked, final_gaps = self._identify_gaps_and_locks(
                state.matches,
                state.voiceover_segments,
                target_conf,
                source_spacing
            )

            initial_gaps = all_pass_metrics[0].initial_gaps if all_pass_metrics else 0
            final_gap_count = len(final_gaps)
            total_filled = initial_gaps - final_gap_count

            # US-101-006: Include budget info in final summary
            # US-167-009: Log stage completion with timing
            total_queries = len(self._used_queries)
            elapsed = time.time() - stage_start_time
            log_stage_complete(
                logger,
                self.name,
                elapsed_seconds=elapsed,
                passes=len(all_pass_metrics),
                gaps_filled=total_filled,
                final_gaps=final_gap_count,
                total_segments=len(state.voiceover_segments),
                queries_used=total_queries,
                max_queries=max_total_queries,
                duration_seconds=round(total_duration, 1)
            )

            # Serialize matches for checkpoint (preserves iterative improvements)
            # Carry forward multi-track data (V2-V8) from _raw_match_dicts since
            # flat Match objects don't store alternatives/secondary/strategy data
            raw_match_dicts = getattr(state, '_raw_match_dicts', None) or []
            serialized_matches = []
            empty_video_file_count = 0
            multi_track_carried = 0
            for i, match in enumerate(state.matches):
                try:
                    if is_empty_source(match, i):
                        logger.warning(
                            f"Match {i}: empty video_file after all fallback attempts "
                            f"(type={type(match).__name__})"
                        )
                        empty_video_file_count += 1

                    serialized = serialize_match_for_iterative_stage(match, i)

                    # If serialization produced empty multi-track data but we have
                    # raw dicts from a prior stage, carry forward the multi-track data
                    if (i < len(raw_match_dicts)
                            and not serialized.get('alternatives')
                            and not serialized.get('secondary_matches')):
                        raw = raw_match_dicts[i]
                        for key in ('alternatives', 'secondary_matches', 'strategy_matches',
                                    'has_gap', 'gap_reason'):
                            if key in raw and raw[key]:
                                serialized[key] = raw[key]
                        if raw.get('alternatives') or raw.get('secondary_matches'):
                            multi_track_carried += 1

                    serialized_matches.append(serialized)
                except Exception as e:
                    logger.warning(f"Failed to serialize match {i}: {e}")

            logger.info(
                f"Serialized {len(serialized_matches)} matches, "
                f"{empty_video_file_count} had empty video_file (skipped), "
                f"{multi_track_carried} carried forward multi-track data from prior stage"
            )

            # Build checkpoint data
            checkpoint_data = {
                'passes_completed': len(all_pass_metrics),
                'initial_gaps': initial_gaps,
                'final_gaps': final_gap_count,
                'total_gaps_filled': total_filled,
                'total_duration_seconds': total_duration,
                'matches': serialized_matches,  # Save modified matches
                'pass_metrics': [
                    {
                        'pass_number': pm.pass_number,
                        'gaps_filled': pm.gaps_filled,
                        'queries_executed': pm.queries_executed,
                        'duration_seconds': pm.duration_seconds,
                    }
                    for pm in all_pass_metrics
                ],
            }

            # Add gap analysis if available
            if gap_analysis:
                checkpoint_data['gap_analysis'] = gap_analysis.to_dict()

            # US-63-012: Store gap pattern logs for cross-run learning
            if all_gap_pattern_logs:
                checkpoint_data['gap_pattern_history'] = [
                    log.to_dict() for log in all_gap_pattern_logs
                ]

            # Stage metrics
            metrics = StageMetrics(
                items_processed=len(state.voiceover_segments),
                items_failed=final_gap_count,
                duration_seconds=total_duration
            )

            return StageResult.ok(checkpoint_data, warnings, metrics)

        except Exception as e:
            log_error_with_context(
                logger,
                "MATCH-001",
                f"Iterative matching failed: {e}"
            )
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if this stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def _save_pass_checkpoint(
        self,
        checkpoint: 'CheckpointManager',
        pass_num: int,
        gaps: List[GapSegment],
        state: 'PipelineState',
        all_pass_metrics: List[PassMetrics],
        used_queries: List[str],
        fetched_video_ids: List[str],
    ) -> None:
        """
        US-89-006: Save intermediate checkpoint after each pass.

        This allows resuming mid-iteration if the pipeline is interrupted
        during a long-running iterative matching process.
        """
        from ..matching.serialization import serialize_match_for_iterative_stage

        # Serialize current matches
        raw_match_dicts = getattr(state, '_raw_match_dicts', None) or []
        serialized_matches = []
        for i, match in enumerate(state.matches):
            try:
                serialized = serialize_match_for_iterative_stage(match, i)
                # Carry forward multi-track data
                if (i < len(raw_match_dicts)
                        and not serialized.get('alternatives')
                        and not serialized.get('secondary_matches')):
                    raw = raw_match_dicts[i]
                    for key in ('alternatives', 'secondary_matches', 'strategy_matches',
                                'has_gap', 'gap_reason'):
                        if key in raw and raw[key]:
                            serialized[key] = raw[key]
                serialized_matches.append(serialized)
            except Exception as e:
                logger.warning(f"Failed to serialize match {i} for pass checkpoint: {e}")

        # Track which gaps have been attempted (by segment index)
        attempted_gap_indices = [g.segment_index for g in gaps]

        # Build pass checkpoint data
        pass_checkpoint_data = {
            'pass_num': pass_num,
            'gaps_remaining': len(gaps),
            'attempted_gap_indices': attempted_gap_indices,
            'used_queries': used_queries,
            'fetched_video_ids': fetched_video_ids,
            'matches': serialized_matches,
            'pass_metrics': [
                {
                    'pass_number': pm.pass_number,
                    'gaps_filled': pm.gaps_filled,
                    'queries_executed': pm.queries_executed,
                    'duration_seconds': pm.duration_seconds,
                }
                for pm in all_pass_metrics
            ],
        }

        # Use save_intermediate to save without updating last_completed_stage
        checkpoint.save_intermediate(self.name, pass_checkpoint_data)
        logger.debug(f"Saved intermediate checkpoint after pass {pass_num}")

    def _check_for_pass_checkpoint(
        self,
        checkpoint: 'CheckpointManager',
        max_iterations: int,
    ) -> Optional[Dict[str, Any]]:
        """
        US-89-006: Check for saved pass state from previous run.

        Returns dict with resume info if found, None otherwise.
        """
        data = checkpoint.get_stage_data(self.name)

        if not data:
            return None

        # Check if this is a full stage completion (not a pass checkpoint)
        # Full stage completion has 'passes_completed' but not 'pass_num'
        if 'passes_completed' in data and 'pass_num' not in data:
            logger.debug("Full stage completion checkpoint found, not resuming from pass")
            return None

        # Check for pass checkpoint (has 'pass_num')
        if 'pass_num' not in data:
            return None

        pass_num = data.get('pass_num', 1)

        # Don't resume if we've already completed all passes
        if pass_num >= max_iterations:
            logger.debug(f"Already completed {pass_num} passes, not resuming")
            return None

        # Check if matches are present
        if 'matches' not in data or not data['matches']:
            logger.warning("Pass checkpoint has no matches, not resuming")
            return None

        logger.info(f"Found pass checkpoint: pass {pass_num}/{max_iterations}")

        return {
            'pass_num': pass_num + 1,  # Resume from next pass
            'gaps_remaining': data.get('gaps_remaining', 0),
            'attempted_gap_indices': data.get('attempted_gap_indices', []),
            'used_queries': data.get('used_queries', []),
            'fetched_video_ids': data.get('fetched_video_ids', []),
            'matches': data.get('matches', []),
        }

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore iterative match stage from checkpoint."""
        from ..state import restore_matches_from_dicts

        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                logger.warning(f"ITERATIVE_MATCH restore: expected dict, got {type(data).__name__}")
                return False

            if 'matches' in data and not isinstance(data['matches'], list):
                logger.warning(f"ITERATIVE_MATCH restore: 'matches' expected list, got {type(data['matches']).__name__}")
                return False

            # Restore matches using shared helper (same validation as MATCH stage)
            matches_data = data.get('matches', [])
            if matches_data:
                restored_matches = restore_matches_from_dicts(
                    matches_data, default_strategy='iterative_restored', logger_instance=logger
                )
                if restored_matches is None:
                    return False

                if restored_matches:
                    state.matches = restored_matches

                    # Merge multi-track data from MATCH checkpoint if iterative
                    # checkpoint has empty multi-track (legacy checkpoints before
                    # carry-forward fix didn't preserve V2-V8 data)
                    has_multi_track = any(
                        m.get('alternatives') or m.get('secondary_matches')
                        for m in matches_data[:5]  # Check first few
                    )
                    if not has_multi_track:
                        match_stage_data = checkpoint.get_stage_data('MATCH')
                        if match_stage_data and isinstance(match_stage_data, dict):
                            match_dicts = match_stage_data.get('matches', [])
                            if match_dicts and len(match_dicts) == len(matches_data):
                                # Merge multi-track keys from MATCH into iterative dicts
                                merged_count = 0
                                for iter_d, match_d in zip(matches_data, match_dicts):
                                    for key in ('alternatives', 'secondary_matches',
                                                'strategy_matches', 'has_gap', 'gap_reason'):
                                        if key in match_d and match_d[key] and not iter_d.get(key):
                                            iter_d[key] = match_d[key]
                                    if match_d.get('alternatives') or match_d.get('secondary_matches'):
                                        merged_count += 1
                                if merged_count:
                                    logger.info(
                                        f"Merged multi-track data from MATCH checkpoint "
                                        f"into {merged_count} iterative matches"
                                    )

                    state._raw_match_dicts = matches_data
                    logger.info(
                        f"Restored {self.name}: {len(restored_matches)} matches, "
                        f"{data.get('passes_completed', 0)} passes, "
                        f"{data.get('total_gaps_filled', 0)} gaps filled"
                    )
                    return True

            # Fallback: no matches in iterative data, rely on MATCH stage
            logger.info(
                f"Restored {self.name} metadata (no matches data): "
                f"{data.get('passes_completed', 0)} passes"
            )
            return True

        except Exception as e:
            log_error_with_context(logger, "PIPE-002", f"Failed to restore {self.name}: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running."""
        if not state.matches:
            return "No matches from MATCH stage. Run MATCH stage first."
        if not state.voiceover_segments:
            return "No voiceover segments. Run ANALYZE stage first."
        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        # Count inputs
        input_count = len(state.matches) if state.matches else 0
        input_count += len(state.voiceover_segments) if state.voiceover_segments else 0

        # Count outputs (enhanced matches)
        output_count = None
        if hasattr(state, 'matches'):
            output_count = len(state.matches)

        return {
            'inputs': 'matches + voiceover segments',
            'outputs': 'enhanced matches',
            'input_count': input_count,
            'output_count': output_count,
        }

    def get_api_estimates(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get API call estimates for dry-run preview"""
        # Count matches that need gap analysis
        match_count = len(state.matches) if state.matches else 0

        # Count voiceover segments
        segment_count = len(state.voiceover_segments) if state.voiceover_segments else 0

        # Iterative matching uses LLM for gap analysis queries
        # Estimate: ~1 LLM call per 5 segments needing additional matches
        llm_calls = max(1, segment_count // 5)

        # LLM cost estimate: ~$0.01 per call (GPT-4o mini as baseline)
        llm_cost = llm_calls * 0.01

        # Iterative matching may also trigger additional video searches
        # Estimate: ~1 search per 10 segments
        video_search_calls = max(1, segment_count // 10)

        # Search cost: $0.002 per search (100 quota units)
        search_cost = video_search_calls * 0.002

        # Total cost
        total_cost = llm_cost + search_cost

        # Estimate duration: ~2s per LLM call + ~0.5s per search
        estimated_duration = llm_calls * 2.0 + video_search_calls * 0.5

        return {
            'llm_calls': llm_calls,
            'video_search_calls': video_search_calls,
            'estimated_cost_usd': round(total_cost, 4),
            'estimated_duration_seconds': round(estimated_duration, 1),
        }

    # =========================================================================
    # Core Algorithm Methods
    # =========================================================================

    def _identify_gaps_and_locks(
        self,
        matches: List['Match'],
        vo_segments: List[Any],
        target_conf: float,
        source_spacing: float
    ) -> Tuple[List[LockedMatch], List[GapSegment]]:
        """
        Classify segments as locked or gap based on confidence + source spacing.

        A segment is LOCKED if:
        - Confidence >= target_conf AND
        - Same video source not used within source_spacing seconds

        A segment is a GAP if:
        - Confidence < target_conf OR
        - Source spacing violated

        Args:
            matches: List of Match objects from MATCH stage
            vo_segments: List of VoiceoverSegment objects
            target_conf: Confidence threshold for locking
            source_spacing: Minimum seconds between same source

        Returns:
            Tuple of (locked_list, gaps_list)
        """
        # Track video source usage by timeline position
        source_timeline: Dict[str, List[float]] = defaultdict(list)

        locked: List[LockedMatch] = []
        gaps: List[GapSegment] = []

        # Build segment lookup for text access
        segment_texts = {}
        for seg in vo_segments:
            idx = getattr(seg, 'index', None) or getattr(seg, 'segment_index', 0)
            text = getattr(seg, 'text', '')
            start = getattr(seg, 'start', 0.0)
            segment_texts[idx] = {'text': text, 'start': start}

        for i, match in enumerate(matches):
            # Extract match details
            segment_idx = i
            if hasattr(match, 'segment_index'):
                segment_idx = match.segment_index

            # Get confidence
            conf = 0.0
            if hasattr(match, 'primary_match') and match.primary_match:
                conf = getattr(match.primary_match, 'confidence', 0.0)
            elif hasattr(match, 'confidence'):
                conf = match.confidence

            # Get video source
            video_id = self._extract_video_id(match)

            # Get position from voiceover segment
            seg_info = segment_texts.get(segment_idx, {'text': '', 'start': 0.0})
            position = seg_info['start']
            vo_text = seg_info['text']

            # Check for source spacing violation
            spacing_violation = False
            if video_id:
                for prev_pos in source_timeline[video_id]:
                    if abs(position - prev_pos) < source_spacing:
                        spacing_violation = True
                        break

            # Classify
            if conf >= target_conf and not spacing_violation:
                # LOCKED
                locked.append(LockedMatch(
                    segment_index=segment_idx,
                    video_id=video_id,
                    confidence=conf,
                    position=position,
                    source_file=self._get_source_file(match)
                ))
                # Record position for spacing check
                source_timeline[video_id].append(position)
            else:
                # GAP
                reason = 'spacing_violation' if spacing_violation else 'low_confidence'
                gaps.append(GapSegment(
                    segment_index=segment_idx,
                    confidence=conf,
                    voiceover_text=vo_text,
                    position=position,
                    current_video_id=video_id,
                    reason=reason
                ))

        return locked, gaps

    def _extract_video_id(self, match: Any) -> str:
        """Extract YouTube video ID from match object."""
        source_file = ""

        if hasattr(match, 'primary_match') and match.primary_match:
            pm = match.primary_match
            if hasattr(pm, 'video_segment') and pm.video_segment:
                source_file = getattr(pm.video_segment, 'source_file', '')
        elif hasattr(match, 'video_file'):
            source_file = match.video_file

        if not source_file:
            return ""

        return extract_video_id(source_file) or Path(source_file).stem

    def _get_source_file(self, match: Any) -> str:
        """Get source file path from match object."""
        if hasattr(match, 'primary_match') and match.primary_match:
            pm = match.primary_match
            if hasattr(pm, 'video_segment') and pm.video_segment:
                return getattr(pm.video_segment, 'source_file', '')
        elif hasattr(match, 'video_file'):
            return match.video_file
        return ""

    def _generate_multi_strategy_queries(
        self,
        gaps: List[GapSegment],
        locked: List[LockedMatch],
        state: 'PipelineState',
        config: Any,
        pass_num: int,
        gap_analysis: Any = None,
        learning_db: Any = None,
        gap_segments: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate queries using multiple strategies.

        Strategies:
        1. Voiceover text keywords - extract keywords from gap text
        2. Similar-to-locked - find videos like successful matches
        3. Entity + topic queries - use extracted entities

        Args:
            gaps: Gap segments to generate queries for
            locked: Locked matches (for similar-to-locked strategy)
            state: Pipeline state
            config: Iterative matching config
            pass_num: Current pass number
            gap_analysis: Optional gap pattern analysis
            learning_db: Optional QueryLearningDB for strategy ranking
            gap_segments: Optional annotated gap segments with chapter_type

        Returns:
            List of query dictionaries
        """
        queries = []

        # US-76-010: Build chapter_id lookup from gap_segments for per-chapter dedup
        chapter_id_by_idx: Dict[int, Optional[str]] = {}
        if gap_segments:
            for gs in gap_segments:
                chapter_id_by_idx[gs.segment_index] = getattr(gs, 'chapter_id', None)

        # Strategy 1: Voiceover text keywords
        if getattr(config, 'use_voiceover_text_queries', True):
            from ..iterative_match.gap_analyzer import extract_keywords_for_gap
            from ..iterative_match import GapSegment as GapSeg

            # US-94-012: Get context window for voiceover context awareness
            context_window = getattr(config, 'context_window_segments', 1)
            voiceover_segments = state.voiceover_segments or []

            for gap in gaps[:20]:  # Limit to avoid too many queries
                # Build context text from adjacent segments
                context_parts = []
                seg_idx = gap.segment_index
                if context_window > 0 and voiceover_segments:
                    # Get segments before the gap
                    for i in range(1, context_window + 1):
                        prev_idx = seg_idx - i
                        if 0 <= prev_idx < len(voiceover_segments):
                            prev_seg = voiceover_segments[prev_idx]
                            if hasattr(prev_seg, 'text'):
                                context_parts.append(prev_seg.text)
                            elif isinstance(prev_seg, dict):
                                context_parts.append(prev_seg.get('text', ''))

                    # Get segments after the gap
                    for i in range(1, context_window + 1):
                        next_idx = seg_idx + i
                        if 0 <= next_idx < len(voiceover_segments):
                            next_seg = voiceover_segments[next_idx]
                            if hasattr(next_seg, 'text'):
                                context_parts.append(next_seg.text)
                            elif isinstance(next_seg, dict):
                                context_parts.append(next_seg.get('text', ''))

                context_text = ' '.join(context_parts)

                gap_obj = GapSeg(
                    segment_index=gap.segment_index,
                    confidence=gap.confidence,
                    voiceover_text=gap.voiceover_text,
                    position=gap.position,
                    pattern_type=gap_analysis.clustered_gaps.get(gap.segment_index, 'other')
                    if gap_analysis else 'other'
                )
                keywords = extract_keywords_for_gap(gap_obj, max_keywords=5, context_text=context_text)
                if keywords:
                    queries.append({
                        'query': ' '.join(keywords),
                        'strategy': 'voiceover',
                        'gap_indices': [gap.segment_index],
                        'chapter_id': chapter_id_by_idx.get(gap.segment_index),  # US-105-007
                        'priority': 2
                    })

        # Strategy 2: Similar-to-locked
        if getattr(config, 'use_similar_to_locked', True) and locked:
            for gap in gaps[:15]:
                # Find nearest locked match
                nearest = self._find_nearest_locked(gap, locked)
                if nearest and nearest.video_id:
                    queries.append({
                        'query': f'similar:{nearest.video_id}',
                        'strategy': 'similar_locked',
                        'gap_indices': [gap.segment_index],
                        'chapter_id': chapter_id_by_idx.get(gap.segment_index),  # US-105-007
                        'seed_video_id': nearest.video_id,
                        'priority': 3  # Higher priority
                    })

        # Strategy 3: Entity + topic queries
        if getattr(config, 'use_entity_topic_queries', True):
            entities = state.extracted_entities or []
            for gap in gaps[:15]:
                # Find entities mentioned in gap text
                gap_text_lower = gap.voiceover_text.lower()
                relevant_entities = []
                for entity in entities:
                    name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
                    if name and name.lower() in gap_text_lower:
                        relevant_entities.append(name)

                if relevant_entities:
                    queries.append({
                        'query': f'{relevant_entities[0]} footage video',
                        'strategy': 'entity',
                        'gap_indices': [gap.segment_index],
                        'chapter_id': chapter_id_by_idx.get(gap.segment_index),  # US-105-007
                        'priority': 2
                    })

        # Strategy 4: Description-derived queries (US-70-012, US-75-012)
        if getattr(config, 'use_description_queries', True) and locked:
            from ..iterative_match.gap_analyzer import (
                extract_description_queries,
                derive_queries_from_descriptions,
            )
            # Build video dicts with descriptions from search results
            vid_id_to_desc = {}
            for vsr in (state.video_search_results or []):
                desc = getattr(vsr, 'description', '') or ''
                if desc and hasattr(vsr, 'video_id'):
                    vid_id_to_desc[vsr.video_id] = desc
            matched_video_dicts = []
            descriptions = []
            for lm in locked:
                desc = vid_id_to_desc.get(lm.video_id, '')
                if desc:
                    matched_video_dicts.append({'description': desc, 'video_id': lm.video_id})
                    descriptions.append(desc)

            # 4a: Per-gap description queries (targeted, higher priority)
            if matched_video_dicts:
                for gap in gaps[:15]:
                    gap_desc_queries = derive_queries_from_descriptions(
                        matched_video_dicts, gap, max_queries=3
                    )
                    for dq in gap_desc_queries:
                        queries.append({
                            'query': dq,
                            'strategy': 'description_gap',
                            'gap_indices': [gap.segment_index],
                            'chapter_id': chapter_id_by_idx.get(gap.segment_index),  # US-105-007
                            'priority': 2
                        })

            # 4b: Broad description queries (fallback, lower priority)
            if descriptions:
                desc_queries = extract_description_queries(descriptions, max_queries=10)
                for dq in desc_queries:
                    queries.append({
                        'query': dq,
                        'strategy': 'description',
                        'gap_indices': [],  # Broad queries, not gap-specific
                        'priority': 1  # Lower priority than targeted strategies
                    })

        # Strategy 5: Video tag-derived queries (US-73-009)
        if getattr(config, 'use_tag_queries', True) and locked:
            from ..iterative_match.gap_analyzer import extract_tags_from_nearby_matches

            for gap in gaps[:20]:
                tags = extract_tags_from_nearby_matches(
                    gap, locked, state, max_tags=3
                )
                if tags:
                    tag_query = ' '.join(tags) + ' footage'
                    queries.append({
                        'query': tag_query,
                        'strategy': 'video_tags',
                        'gap_indices': [gap.segment_index],
                        'chapter_id': chapter_id_by_idx.get(gap.segment_index),  # US-105-007
                        'priority': 2
                    })

        # Strategy 6: US-111-009 Context-aware queries
        # Use context from already-matched segments near gaps to improve query generation
        if getattr(config, 'enable_context_queries', True) and locked:
            from ..iterative_match.gap_analyzer import (
                extract_context_from_nearby_matches,
                generate_context_aware_queries,
                extract_keywords_for_gap,
            )
            from ..iterative_match.gap_analyzer import GapSegment as GapSeg

            context_boost = getattr(config, 'iterative_context_boost', 0.15)
            context_window_seconds = getattr(config, 'context_boost_window_seconds', 180.0)
            topic_weight = getattr(config, 'context_topic_weight', 0.5)

            # Get voiceover segments for keyword extraction
            voiceover_segments = state.voiceover_segments or []

            for gap in gaps[:15]:
                # Create GapSegment object for the gap
                gap_obj = GapSeg(
                    segment_index=gap.segment_index,
                    confidence=gap.confidence,
                    voiceover_text=gap.voiceover_text,
                    position=gap.position,
                    pattern_type=gap_analysis.clustered_gaps.get(gap.segment_index, 'other')
                    if gap_analysis else 'other'
                )

                # Get gap keywords
                gap_keywords = extract_keywords_for_gap(gap_obj, max_keywords=5, context_text='')

                if not gap_keywords:
                    continue

                # Extract context from nearby locked matches
                context_segments = extract_context_from_nearby_matches(
                    gap=gap,
                    locked_matches=locked,
                    state=state,
                    window_seconds=context_window_seconds,
                    max_context_segments=5,
                )

                if not context_segments:
                    continue

                # Generate context-aware queries
                context_queries = generate_context_aware_queries(
                    gap=gap,
                    context_segments=context_segments,
                    gap_keywords=gap_keywords,
                    max_queries=2,
                    context_boost=context_boost,
                    topic_weight=topic_weight,
                )

                # Add context-aware queries (skip the first one if it's just the base query)
                for cq in context_queries[1:]:  # Skip base query (already covered by Strategy 1)
                    if cq.context_keywords:  # Only add queries with actual context
                        queries.append({
                            'query': cq.query,
                            'strategy': 'context_aware',
                            'gap_indices': [gap.segment_index],
                            'chapter_id': chapter_id_by_idx.get(gap.segment_index),
                            'context_keywords': cq.context_keywords,
                            'relevance_score': cq.relevance_score,
                            'priority': 2 + int(cq.context_weight * 2)  # Higher priority with more context
                        })

        # US-76-010: Per-chapter query diversity enforcement
        # Within the same chapter, duplicate queries waste search budget.
        # Vary duplicates by appending chapter-specific context keywords.
        if chapter_id_by_idx:
            chapter_query_sets: Dict[str, set] = {}  # chapter_id -> set of query keys
            for q in queries:
                gap_indices = q.get('gap_indices', [])
                if not gap_indices:
                    continue  # Broad queries (no gap) skip chapter dedup
                ch_id = chapter_id_by_idx.get(gap_indices[0])
                if ch_id is None:
                    continue  # No chapter assigned
                query_key = q['query'].lower().strip()
                if ch_id not in chapter_query_sets:
                    chapter_query_sets[ch_id] = set()
                if query_key in chapter_query_sets[ch_id]:
                    # Duplicate within chapter - vary the query
                    gap_idx = gap_indices[0]
                    gap_obj = next((g for g in gaps if g.segment_index == gap_idx), None)
                    varied = self._vary_query_for_chapter(
                        q['query'], ch_id, gap_obj, chapter_query_sets[ch_id]
                    )
                    q['query'] = varied
                    query_key = varied.lower().strip()
                chapter_query_sets[ch_id].add(query_key)

        # Deduplicate by query string AND exclude already-used queries
        seen_queries = set()
        unique_queries = []
        skipped_as_used = 0
        for q in queries:
            query_key = q['query'].lower().strip()
            if query_key in self._used_queries:
                skipped_as_used += 1
            elif query_key not in seen_queries:
                seen_queries.add(query_key)
                unique_queries.append(q)

        # On subsequent passes, generate diverse variants of remaining gaps
        if pass_num > 1 and len(unique_queries) < 10:
            variant_queries = self._generate_query_variants(gaps, pass_num)
            for vq in variant_queries:
                vq_key = vq['query'].lower().strip()
                if vq_key not in seen_queries and vq_key not in self._used_queries:
                    seen_queries.add(vq_key)
                    unique_queries.append(vq)

        # US-76-005: Boost priority using chapter-type strategy ranking
        # US-126-002: Only apply if config option is enabled
        chapter_type_enabled = getattr(config, 'query_type_by_chapter_type', True)
        if learning_db and gap_segments and chapter_type_enabled:
            # Build chapter_type lookup from annotated gap_segments
            chapter_type_by_idx: Dict[int, str] = {}
            for gs in gap_segments:
                chapter_type_by_idx[gs.segment_index] = getattr(gs, 'chapter_type', 'body')

            for q in unique_queries:
                gap_indices = q.get('gap_indices', [])
                if not gap_indices:
                    continue
                chapter_type = chapter_type_by_idx.get(gap_indices[0], 'body')
                # Only apply ranking for non-default chapter types
                if chapter_type in ('body', 'unknown'):
                    continue
                strategy = q.get('strategy', '')
                gap_pattern = 'other'
                if gap_analysis:
                    for p, indices in gap_analysis.clustered_gaps.items():
                        if gap_indices[0] in indices:
                            gap_pattern = p
                            break
                ranking = learning_db.get_strategy_ranking_for_chapter(
                    gap_pattern, chapter_type
                )
                # US-166-012: Log query learning suggestions at DEBUG level
                if ranking:
                    logger.debug(
                        f"[ITERATIVE] Query learning suggestions for pattern={gap_pattern}, "
                        f"chapter={chapter_type}: top strategies = {ranking[:3]}"
                    )
                if strategy in ranking:
                    rank_pos = ranking.index(strategy)
                    # Top-ranked strategies get +2, second +1 priority boost
                    if rank_pos == 0:
                        q['priority'] = q.get('priority', 0) + 2
                    elif rank_pos == 1:
                        q['priority'] = q.get('priority', 0) + 1

        # Sort by priority
        unique_queries.sort(key=lambda x: x.get('priority', 0), reverse=True)

        # Limit total queries
        max_queries = getattr(config, 'max_new_videos_per_pass', 50) // 2
        final_queries = unique_queries[:max_queries]

        # Log deduplication stats if any queries were skipped
        if skipped_as_used > 0:
            logger.info(f"Skipped {skipped_as_used} queries already used in previous passes")

        # Track these queries as used for future passes
        for q in final_queries:
            self._used_queries.add(q['query'].lower().strip())

        return final_queries

    def _find_nearest_locked(
        self,
        gap: GapSegment,
        locked: List[LockedMatch],
        max_distance: float = 60.0
    ) -> Optional[LockedMatch]:
        """Find the locked match nearest to a gap in timeline."""
        nearest = None
        min_distance = float('inf')

        for lock in locked:
            distance = abs(gap.position - lock.position)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest = lock

        return nearest

    def _generate_query_variants(
        self,
        gaps: List[GapSegment],
        pass_num: int
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse query variants for subsequent passes.

        Uses different framings and vocabulary to search for the same concepts
        in different ways, increasing the chance of finding useful footage.

        Args:
            gaps: Gap segments to generate variants for
            pass_num: Current pass number (affects variant strategy)

        Returns:
            List of variant query dictionaries
        """
        import re
        variants = []

        # Different framing suffixes per pass
        pass_suffixes = {
            2: ['documentary', 'stock footage', 'archive'],
            3: ['b-roll', 'cinematic', 'film'],
            4: ['historical', 'raw footage', 'clip'],
            5: ['4K', 'HD footage', 'professional'],
        }
        suffixes = pass_suffixes.get(pass_num, ['footage'])

        # Extract nouns and key phrases from gaps
        for gap in gaps[:15]:
            text = gap.voiceover_text
            if not text or len(text) < 10:
                continue

            # Extract capitalized words (likely proper nouns/entities)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

            # Extract longer words (likely meaningful content words)
            words = text.split()
            content_words = [w.strip('.,!?;:') for w in words if len(w) > 5]

            # Strategy varies by pass
            if pass_num == 2:
                # Focus on proper nouns with different framing
                for noun in proper_nouns[:2]:
                    for suffix in suffixes[:2]:
                        query = f'{noun} {suffix}'
                        if query.lower().strip() not in self._used_queries:
                            variants.append({
                                'query': query,
                                'strategy': 'variant_noun',
                                'gap_indices': [gap.segment_index],
                                'priority': 1
                            })

            elif pass_num == 3:
                # Use content words with visual descriptors
                if content_words:
                    for word in content_words[:3]:
                        for suffix in suffixes[:2]:
                            query = f'{word} {suffix}'
                            if query.lower().strip() not in self._used_queries:
                                variants.append({
                                    'query': query,
                                    'strategy': 'variant_content',
                                    'gap_indices': [gap.segment_index],
                                    'priority': 1
                                })

            else:  # pass 4+
                # Broader searches with just key terms
                all_terms = list(set(proper_nouns + content_words[:3]))
                if all_terms:
                    for term in all_terms[:2]:
                        query = f'{term} {suffixes[0] if suffixes else "footage"}'
                        if query.lower().strip() not in self._used_queries:
                            variants.append({
                                'query': query,
                                'strategy': 'variant_broad',
                                'gap_indices': [gap.segment_index],
                                'priority': 1
                            })

        return variants

    @staticmethod
    def _vary_query_for_chapter(
        query: str,
        chapter_id: str,
        gap: Optional[GapSegment],
        existing_queries: set,
    ) -> str:
        """
        Vary a duplicate query within a chapter to enforce diversity.

        US-76-010: When two gaps in the same chapter would produce identical
        queries, append chapter-specific context keywords to differentiate them
        while preserving original semantic intent.

        Args:
            query: Original query text
            chapter_id: Chapter identifier for context
            gap: Gap segment (used to extract unique keywords)
            existing_queries: Set of already-used query keys in this chapter

        Returns:
            Varied query string that differs from existing queries
        """
        import re

        # Extract unique words from gap voiceover text for context
        context_words = []
        if gap and gap.voiceover_text:
            words = gap.voiceover_text.split()
            # Pick content words (>4 chars) not already in the query
            query_lower = query.lower()
            for w in words:
                cleaned = re.sub(r'[^\w]', '', w).lower()
                if len(cleaned) > 4 and cleaned not in query_lower:
                    context_words.append(cleaned)

        # Try appending context words one by one until we get a unique query
        for cw in context_words[:5]:
            candidate = f"{query} {cw}"
            if candidate.lower().strip() not in existing_queries:
                return candidate

        # Fallback: append a chapter-derived keyword
        # Extract a short label from chapter_id (e.g., "chapter_0" -> "section 1")
        ch_label = chapter_id.split('_')[-1] if '_' in chapter_id else chapter_id
        # Use a rotation suffix to ensure uniqueness
        for i in range(1, 10):
            suffix = f"context {ch_label}" if i == 1 else f"context {ch_label} {i}"
            candidate = f"{query} {suffix}"
            if candidate.lower().strip() not in existing_queries:
                return candidate

        # Last resort: just append a number
        return f"{query} alt"

    def _refine_queries_progressive(
        self,
        queries: List[Dict[str, Any]],
        pass_num: int,
        previous_metrics: List[PassMetrics],
        learning_db: Any
    ) -> List[Dict[str, Any]]:
        """
        Modify failing queries based on pass number and learning.

        Refinements:
        - Pass 2+: Add synonyms to keywords
        - Pass 3+: Broaden scope (remove specific terms)
        - Use learned successful templates

        Args:
            queries: Queries to refine
            pass_num: Current pass number
            previous_metrics: Metrics from prior passes
            learning_db: Query learning database

        Returns:
            Refined query list
        """
        if pass_num < 2:
            return queries

        refined = []
        for q in queries:
            query_text = q['query']

            # Simple refinements
            if pass_num >= 2:
                # Add 'footage' or 'video' if not present
                if 'footage' not in query_text.lower() and 'video' not in query_text.lower():
                    query_text = f"{query_text} footage"

            if pass_num >= 3:
                # Broaden: remove very specific terms (short words)
                words = query_text.split()
                words = [w for w in words if len(w) > 3 or w.lower() in {'the', 'and', 'for'}]
                if len(words) >= 2:
                    query_text = ' '.join(words)

            # Apply learning if available
            if learning_db:
                refined_template = learning_db.get_refined_template(query_text, pass_num)
                if refined_template:
                    query_text = refined_template

            refined.append({
                **q,
                'query': query_text,
                'refined': True
            })

        return refined

    def _deduplicate_queries(
        self,
        queries: List[Dict[str, Any]],
        enable_optimization: bool = True,
        similarity_threshold: float = 0.85
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Deduplicate similar queries to avoid redundant YouTube searches.

        Uses normalized text comparison to find near-duplicate queries:
        - Same query with different casing
        - Same query with different punctuation
        - Very similar queries (e.g., "topic footage" vs "topic video")

        Args:
            queries: List of query dictionaries with 'query' key
            enable_optimization: Whether to perform deduplication
            similarity_threshold: Minimum similarity to consider as duplicate (0-1)

        Returns:
            Tuple of (deduplicated queries list, number of queries saved)
        """
        if not enable_optimization or len(queries) <= 1:
            return queries, 0

        # Normalize query for comparison
        def normalize_query(text: str) -> str:
            """Normalize query text for comparison."""
            import re
            # Lowercase
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        # Calculate Jaccard similarity between two queries
        def jaccard_similarity(text1: str, text2: str) -> float:
            """Calculate Jaccard similarity between two texts."""
            set1 = set(text1.split())
            set2 = set(text2.split())
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        seen_normalized: Dict[str, int] = {}  # normalized -> first query index
        unique_queries = []
        queries_saved = 0

        for q in queries:
            query_text = q.get('query', '')
            if not query_text:
                continue

            normalized = normalize_query(query_text)

            # Check for exact duplicate first
            if normalized in seen_normalized:
                queries_saved += 1
                continue

            # Check for similar queries
            is_duplicate = False
            for seen_norm in seen_normalized:
                similarity = jaccard_similarity(normalized, seen_norm)
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    queries_saved += 1
                    break

            if not is_duplicate:
                seen_normalized[normalized] = len(unique_queries)
                unique_queries.append(q)

        return unique_queries, queries_saved

    def _search_youtube_for_videos(
        self,
        queries: List[Dict[str, Any]],
        state: 'PipelineState',
        config: 'Config',
        iter_config: Any
    ) -> Tuple[List[str], int, int]:
        """
        Search YouTube for videos matching the queries.

        Returns a list of video IDs without fetching captions.
        This allows for batched/streamed caption fetching later.

        Args:
            queries: Search queries to execute
            state: Pipeline state
            config: Main config
            iter_config: Iterative matching config

        Returns:
            Tuple of (List of YouTube video IDs found, cache_hits, cache_misses)
        """
        import subprocess
        import json

        # Get existing video IDs to avoid duplicates
        existing_ids: Set[str] = set()

        # From state.video_ids (simplified pipeline)
        if hasattr(state, 'video_ids') and state.video_ids:
            existing_ids.update(state.video_ids)

        # From text_metadata (caption-first mode)
        for meta in (state.text_metadata or []):
            if isinstance(meta, dict):
                vid_path = meta.get('video_path', '')
                vid_id = self._extract_video_id_from_path(vid_path)
                if vid_id:
                    existing_ids.add(vid_id)

        # Backward compatibility: check legacy downloaded_videos
        if hasattr(state, 'downloaded_videos'):
            for vid in state.downloaded_videos:
                vid_file = vid.file if hasattr(vid, 'file') else vid.get('file', '')
                vid_id = self._extract_video_id_from_path(vid_file)
                if vid_id:
                    existing_ids.add(vid_id)

        # Include videos already fetched in previous passes (cross-pass dedup)
        existing_ids.update(self._fetched_video_ids)

        # Search settings
        results_per_query = getattr(iter_config, 'search_results_per_gap', 10)
        max_new_videos = getattr(iter_config, 'max_new_videos_per_pass', 50)

        # US-99-008: Duration filter from config (typical for documentary footage)
        min_duration = getattr(iter_config, 'search_min_duration', 30)
        max_duration = getattr(iter_config, 'search_max_duration', 600)  # 10 minutes max

        # Initialize search cache with config options (US-94-011)
        cache_enabled = getattr(iter_config, 'cache_query_results', True)
        cache_ttl = getattr(iter_config, 'query_cache_ttl_hours', 24)

        # US-94-011: Invalidate cache on config changes
        search_cache = None
        if cache_enabled:
            search_cache = SearchResultsCache(ttl_hours=cache_ttl)
            # Invalidate cache if TTL changed significantly (config was modified)
            self._invalidate_cache_on_config_change(search_cache, cache_ttl)

        # Collect unique video IDs from search
        new_video_ids: Set[str] = set()
        query_video_map: Dict[str, List[str]] = {}  # Track which query found which videos
        cache_hits = 0
        cache_misses = 0

        # Log cookie being used for YouTube searches
        self._get_cookie_args(config, log_usage=True)

        logger.info(f"Searching YouTube ({len(queries)} queries, {results_per_query} results each)...")

        for q in queries:
            if len(new_video_ids) >= max_new_videos:
                break

            query_text = q['query']
            if not query_text or query_text.startswith('similar:'):
                # Skip empty or similar-to queries (not supported in basic search)
                continue

            try:
                # US-94-011: Check search cache first (if enabled)
                cached_videos = None
                if search_cache is not None:
                    cached_videos = search_cache.get_search_result(
                        keyword=query_text,
                        tier='iterative',
                        search_pool=results_per_query
                    )

                query_vids = []

                if cached_videos is not None:
                    # Use cached results
                    cache_hits += 1
                    for vid_info in cached_videos:
                        vid_id = vid_info.get('id', '')
                        if vid_id and vid_id not in existing_ids and vid_id not in new_video_ids:
                            new_video_ids.add(vid_id)
                            query_vids.append(vid_id)
                            if len(new_video_ids) >= max_new_videos:
                                break
                else:
                    # Cache miss or disabled - search YouTube
                    cache_misses += 1

                    # Use yt-dlp to search YouTube (metadata only, no download)
                    cmd = [
                        'yt-dlp',
                        '--ignore-config',
                        f'ytsearch{results_per_query}:{query_text}',
                        '--dump-json',
                        '--flat-playlist',
                        '--no-download',
                        '--match-filter', f'duration>{min_duration} & duration<{max_duration} & !is_live & !was_live',
                    ]

                    # Add cookie arguments for YouTube authentication (same as pipeline)
                    cookie_args = self._get_cookie_args(config)
                    if cookie_args:
                        cmd.extend(cookie_args)

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=60
                    )

                    if result.returncode != 0:
                        logger.debug(f"Search failed for '{query_text}': {result.stderr[:200]}")
                        continue

                    # Parse JSON lines output and collect for caching
                    search_results_for_cache = []
                    for line in result.stdout.strip().split('\n'):
                        if not line:
                            continue
                        try:
                            info = json.loads(line)
                            vid_id = info.get('id', '')
                            if vid_id:
                                # Store minimal info for cache
                                search_results_for_cache.append({
                                    'id': vid_id,
                                    'title': info.get('title', ''),
                                    'duration': info.get('duration', 0)
                                })
                                if vid_id not in existing_ids and vid_id not in new_video_ids:
                                    new_video_ids.add(vid_id)
                                    query_vids.append(vid_id)
                                    if len(new_video_ids) >= max_new_videos:
                                        break
                        except json.JSONDecodeError:
                            continue

                    # Cache the search results
                    if search_results_for_cache:
                        search_cache.set_search_result(
                            keyword=query_text,
                            tier='iterative',
                            search_pool=results_per_query,
                            videos=search_results_for_cache
                        )

                if query_vids:
                    query_video_map[query_text] = query_vids
                    q['videos_found'] = len(query_vids)

            except subprocess.TimeoutExpired:
                logger.warning(f"Search timeout for '{query_text}'")
            except Exception as e:
                logger.debug(f"Search error for '{query_text}': {e}")

        # Report cache stats
        total_queries = cache_hits + cache_misses
        if total_queries > 0:
            hit_rate = (cache_hits / total_queries) * 100
            logger.debug(f"Search cache: {cache_hits}/{total_queries} hits ({hit_rate:.0f}%)")

        if not new_video_ids:
            logger.info("No new videos found from search queries")
            return []

        # Track fetched videos for cross-pass deduplication
        self._fetched_video_ids.update(new_video_ids)

        # US-94-011: Return video IDs with cache metrics
        return list(new_video_ids), cache_hits, cache_misses

    def _search_and_fetch_captions(
        self,
        queries: List[Dict[str, Any]],
        state: 'PipelineState',
        config: 'Config',
        iter_config: Any
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Execute YouTube searches and fetch captions for new videos.

        DEPRECATED: Use _search_youtube_for_videos + _fetch_captions_for_videos
        with batching for better streaming behavior.

        Args:
            queries: Search queries to execute
            state: Pipeline state
            config: Main config
            iter_config: Iterative matching config

        Returns:
            Tuple of (List of candidate dicts, cache_hits, cache_misses)

        Returns:
            List of new video candidate dictionaries with caption segments
        """
        # Search for videos first (US-94-011: also returns cache metrics)
        video_ids, cache_hits, cache_misses = self._search_youtube_for_videos(queries, state, config, iter_config)

        if not video_ids:
            return [], cache_hits, cache_misses

        logger.info(f"Found {len(video_ids)} new video candidates")

        # Fetch captions for new videos
        caption_timeout = getattr(iter_config, 'caption_timeout', 30)
        new_candidates = self._fetch_captions_for_videos(
            video_ids,
            config,
            iter_config,
            caption_timeout
        )

        # US-94-011: Return candidates with cache metrics
        return new_candidates, cache_hits, cache_misses

    def _fetch_captions_for_videos(
        self,
        video_ids: List[str],
        config: 'Config',
        iter_config: Any,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Fetch captions for a list of video IDs.

        Args:
            video_ids: YouTube video IDs to fetch captions for
            config: Main config
            iter_config: Iterative matching config
            timeout: Timeout per video

        Returns:
            List of caption result dictionaries
        """
        from ..caption_fetcher import (
            CaptionFetcher,
            CaptionFetchError,
            CaptionUnavailableError,
            CaptionCache,
            determine_caption_quality
        )

        # Get caption config
        caption_config = getattr(config.download, 'caption_first', None)
        preferred_lang = 'en'
        if caption_config:
            preferred_lang = getattr(caption_config, 'preferred_language', 'en')

        # Get cookie args (with rotation if enabled) and log
        cookie_args = self._get_cookie_args(config, log_usage=True)

        # Initialize fetcher and cache with rotated cookies
        cache = CaptionCache(caption_config)
        fetcher = CaptionFetcher(config=config, cookie_args=cookie_args, caption_cache=cache)

        candidates = []
        success_count = 0
        fail_count = 0
        cache_hits = 0

        logger.info(f"Fetching captions for {len(video_ids)} videos...")

        for i, video_id in enumerate(video_ids):
            try:
                # Check cache first
                cached = cache.get_caption(video_id, preferred_lang) if cache.enabled else None

                if cached:
                    # Use cached result
                    result = cached
                    cache_hits += 1
                else:
                    # Fetch fresh
                    result = fetcher.fetch_captions(video_id, language=preferred_lang)
                    # Save to cache
                    if result and result.segments and cache.enabled:
                        cache.store(result)

                if result and result.segments:
                    # Segments may be CaptionSegment objects (fresh fetch) or dicts (from cache)
                    first_seg = result.segments[0]
                    segments_are_dicts = isinstance(first_seg, dict)

                    # Determine caption quality
                    if segments_are_dicts:
                        total_duration = sum(
                            (s['end'] - s['start']) for s in result.segments
                        )
                    else:
                        total_duration = sum(
                            (s.end_time - s.start_time) for s in result.segments
                        )
                    quality = determine_caption_quality(
                        is_auto_generated=result.is_auto_generated,
                        segment_count=len(result.segments),
                        total_duration=total_duration
                    )

                    # Convert segments to dict format
                    if segments_are_dicts:
                        segments = [
                            {
                                'text': seg.get('text', ''),
                                'start': seg.get('start', seg.get('start_time', 0)),
                                'end': seg.get('end', seg.get('end_time', 0)),
                            }
                            for seg in result.segments
                        ]
                    else:
                        segments = [
                            {
                                'text': seg.text,
                                'start': seg.start_time,
                                'end': seg.end_time,
                            }
                            for seg in result.segments
                        ]

                    candidates.append({
                        'video_id': video_id,
                        'segments': segments,
                        'language': result.language,
                        'is_auto_generated': result.is_auto_generated,
                        'caption_quality': quality,
                        'total_duration': total_duration,  # US-94-010: Store for tier diversity
                    })
                    success_count += 1

                    # Log progress every 10 videos
                    if (i + 1) % 10 == 0:
                        logger.debug(f"[{i + 1}/{len(video_ids)}] Fetched {success_count} captions...")

            except CaptionUnavailableError:
                fail_count += 1
                logger.debug(f"No captions for {video_id}")
            except CaptionFetchError as e:
                fail_count += 1
                logger.debug(f"Caption fetch error for {video_id}: {e}")
            except Exception as e:
                fail_count += 1
                logger.debug(f"Unexpected error fetching {video_id}: {e}")

        logger.info(f"Captions: {success_count} fetched ({cache_hits} from cache), {fail_count} unavailable")

        return candidates

    # US-101-006: Query budget tracking
    def _log_budget_summary(
        self,
        total_used: int,
        max_total: int,
        queries_per_pass: List[int],
        max_per_pass: int
    ) -> None:
        """Log a summary when query budget is exhausted.

        Args:
            total_used: Total queries used
            max_total: Maximum queries allowed per run
            queries_per_pass: List of queries used per pass
            max_per_pass: Maximum queries allowed per pass
        """
        logger.info("Query budget summary")
        logger.info(f"Total queries used: {total_used}/{max_total} ({total_used/max_total:.0%})")
        for i, count in enumerate(queries_per_pass, 1):
            pct = count / max_per_pass if max_per_pass > 0 else 0
            logger.debug(f"Pass {i}: {count} queries ({pct:.0%} of per-pass budget)")

    # US-101-011: Gap coverage progress report
    def _log_gap_coverage_report(
        self,
        gaps: List['GapSegment'],
        pass_num: int,
        initial_gap_count: int,
        gaps_filled: int,
        iter_config: Any
    ) -> None:
        """Log a progress report showing gap coverage metrics.

        Args:
            gaps: Current list of gap segments
            pass_num: Current pass number
            initial_gap_count: Initial gap count at start of pass
            gaps_filled: Number of gaps filled this pass
            iter_config: Iterative matching config (for log_pass_summaries check)
        """
        # Check if logging is enabled
        log_summaries = getattr(iter_config, 'log_pass_summaries', True)
        if not log_summaries:
            return

        current_gaps = len(gaps)
        fill_pct = (gaps_filled / initial_gap_count * 100) if initial_gap_count > 0 else 0

        # Confidence distribution
        low_conf = sum(1 for g in gaps if g.confidence < 0.3)
        med_conf = sum(1 for g in gaps if 0.3 <= g.confidence < 0.6)
        high_conf = sum(1 for g in gaps if g.confidence >= 0.6)

        # Pattern distribution (based on reason)
        low_conf_reason = sum(1 for g in gaps if g.reason == 'low_confidence')
        spacing_reason = sum(1 for g in gaps if g.reason == 'spacing_violation')

        # ASCII progress bar
        bar_width = 30
        filled = int(bar_width * fill_pct / 100) if fill_pct <= 100 else bar_width
        bar = '█' * filled + '░' * (bar_width - filled)

        logger.info(f"Pass {pass_num} gap coverage: {gaps_filled} filled, {current_gaps} remaining ({fill_pct:.1f}% progress)")
        logger.debug(f"Initial gaps: {initial_gap_count}, current: {current_gaps}")
        if current_gaps > 0:
            logger.debug(f"Confidence distribution - Low: {low_conf}, Medium: {med_conf}, High: {high_conf}")
            logger.debug(f"Pattern distribution - Low confidence: {low_conf_reason}, Spacing violation: {spacing_reason}")
        else:
            logger.debug("No gaps remaining")

    def _extract_video_id_from_path(self, path: str) -> str:
        """Extract video ID from file path."""
        return extract_video_id(path) or ""

    # US-94-010: Duration tier classification for diversity enforcement
    DURATION_TIER_SHORT = "short"      # < 2 minutes
    DURATION_TIER_MEDIUM = "medium"    # 2-10 minutes
    DURATION_TIER_LONG = "long"        # > 10 minutes

    def _get_duration_tier(self, duration_seconds: float) -> str:
        """
        Classify video duration into tier for diversity enforcement.

        Args:
            duration_seconds: Video duration in seconds

        Returns:
            Tier string: 'short', 'medium', or 'long'
        """
        if duration_seconds < 120:  # < 2 minutes
            return self.DURATION_TIER_SHORT
        elif duration_seconds < 600:  # 2-10 minutes
            return self.DURATION_TIER_MEDIUM
        else:  # > 10 minutes
            return self.DURATION_TIER_LONG

    def _rematch_gaps(
        self,
        gaps: List[GapSegment],
        locked: List[LockedMatch],
        new_candidates: List[Dict[str, Any]],
        state: 'PipelineState',
        config: 'Config'
    ) -> int:
        """
        Re-match gap segments with expanded candidate pool.

        Adds new caption data to state, computes embeddings, and re-matches gaps.

        Args:
            gaps: Gap segments to re-match
            locked: Locked matches (preserved)
            new_candidates: New video candidates from search
            state: Pipeline state
            config: Main config

        Returns:
            Number of gaps successfully filled
        """
        if not new_candidates:
            return 0

        # 1. Add new caption segments to text_metadata
        original_metadata_count = len(state.text_metadata) if state.text_metadata else 0
        new_segment_count = 0

        for candidate in new_candidates:
            video_id = candidate['video_id']
            segments = candidate.get('segments', [])
            language = candidate.get('language', 'en')
            is_auto = candidate.get('is_auto_generated', False)
            quality = candidate.get('caption_quality', 'medium')
            total_duration = candidate.get('total_duration', 0)  # US-94-010: For tier diversity

            for seg in segments:
                state.text_metadata.append({
                    'text': seg.get('text', ''),
                    'video_path': video_id,
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'source_file': video_id,
                    'caption_source': 'youtube_iterative',
                    'caption_language': language,
                    'caption_auto_generated': is_auto,
                    'caption_quality': quality,
                    'iterative_pass': True,  # Mark as from iterative matching
                    'total_duration': total_duration,  # US-94-010: For tier diversity
                })
                new_segment_count += 1

        if new_segment_count == 0:
            return 0

        logger.info(f"Added {new_segment_count} caption segments from {len(new_candidates)} videos")

        # 2. Compute embeddings for new segments
        new_embeddings = self._compute_embeddings_for_new_segments(
            state, config, original_metadata_count
        )

        if new_embeddings is None or len(new_embeddings) == 0:
            logger.warning("Failed to compute embeddings for new segments")
            return 0

        # 3. Re-match gap segments using embedding similarity
        gaps_filled = self._match_gaps_to_new_candidates(
            gaps, locked, state, config, original_metadata_count
        )

        return gaps_filled

    def _compute_embeddings_for_new_segments(
        self,
        state: 'PipelineState',
        config: 'Config',
        start_index: int
    ) -> Optional[Any]:
        """
        Compute embeddings for newly added text_metadata segments.

        Args:
            state: Pipeline state
            config: Main config
            start_index: Index in text_metadata where new segments start

        Returns:
            New embeddings array or None on failure
        """
        try:
            from ..embeddings import compute_embeddings, get_embedding_provider
            from ..utils import CacheManager
            import numpy as np

            # Get new segment texts
            new_metadata = state.text_metadata[start_index:]
            if not new_metadata:
                return None

            texts = [m.get('text', '') for m in new_metadata if m.get('text')]
            if not texts:
                return None

            # Get embedding provider
            provider = get_embedding_provider(config)
            cache = CacheManager(config.cache.cache_dir)

            # Compute embeddings for new texts
            new_embeddings = compute_embeddings(
                texts=texts,
                provider=provider,
                cache=cache,
                cache_key="iterative_segments",
                show_progress=False,
                config=config,
                embed_mode="document"
            )

            # Append to local embedding storage (no longer stored on PipelineState)
            if self._embeddings is not None and len(self._embeddings) > 0:
                self._embeddings = np.vstack([self._embeddings, new_embeddings])
            else:
                self._embeddings = new_embeddings
                # Record the global text_metadata offset for the first batch
                self._embedding_global_offset = start_index

            # Rebuild embedding index with all vectors
            from ..embeddings import build_embedding_index
            self._embedding_index = build_embedding_index(self._embeddings, config)

            logger.info(f"Computed {len(new_embeddings)} new embeddings, total now {len(self._embeddings)}")
            return new_embeddings

        except Exception as e:
            log_error_with_context(logger, "MATCH-003", f"Failed to compute embeddings: {e}")
            return None

    def _match_gaps_to_new_candidates(
        self,
        gaps: List[GapSegment],
        locked: List[LockedMatch],
        state: 'PipelineState',
        config: 'Config',
        new_segment_start: int
    ) -> int:
        """
        Match gap segments against new video candidates using embedding similarity.

        Pre-computes voiceover embeddings for all gaps at once, then iterates
        through gaps for matching. This avoids per-gap embedding API calls.

        Args:
            gaps: Gap segments to fill
            locked: Locked matches (for source spacing check)
            state: Pipeline state with updated embeddings
            config: Main config
            new_segment_start: Index where new segments start in text_metadata

        Returns:
            Number of gaps successfully filled
        """
        if not gaps or self._embedding_index is None:
            return 0

        try:
            from ..embeddings import get_embedding_provider
            import numpy as np

            provider = get_embedding_provider(config)

            # PRE-COMPUTE: Batch compute all voiceover embeddings at pass start
            precomputed_embeddings = self._precompute_voiceover_embeddings(
                gaps, state, provider
            )
            logger.info(f"Pre-computed {len(precomputed_embeddings)} voiceover embeddings for gap matching")

            # Get locked video IDs for source spacing
            locked_sources: Dict[str, List[float]] = defaultdict(list)
            for lock in locked:
                locked_sources[lock.video_id].append(lock.position)

            # Get source spacing threshold
            iter_config = getattr(config, 'iterative_matching', None)
            source_spacing = 300.0
            target_conf = 0.90
            tier_diversity_weight = 0.15  # US-94-010: Default
            chapter_boost = 0.1  # US-105-007: Default chapter boost
            if iter_config:
                source_spacing = getattr(iter_config, 'source_spacing_seconds', 300.0)
                target_conf = getattr(iter_config, 'target_confidence', 0.90)
                tier_diversity_weight = getattr(iter_config, 'tier_diversity_weight', 0.15)
                chapter_boost = getattr(iter_config, 'iterative_chapter_boost', 0.1)

            # US-94-010: Track used duration tiers for diversity enforcement
            used_tiers: set = set()

            gaps_filled = 0

            # FAISS indices are local (0-based into self._embeddings).
            # Convert to global text_metadata indices using the offset.
            global_offset = self._embedding_global_offset or 0

            for gap in gaps:
                # Get voiceover segment embedding (from precomputed cache)
                vo_embedding = self._get_voiceover_embedding(
                    gap.segment_index, gap.voiceover_text, state, provider,
                    precomputed=precomputed_embeddings
                )
                if vo_embedding is None:
                    continue

                # Compute how many local embedding indices correspond to new segments
                # new_segment_start is a global text_metadata index;
                # convert to local embedding index for counting
                local_new_start = new_segment_start - global_offset
                num_new = len(self._embeddings) - max(0, local_new_start)
                k = min(20, num_new)
                if k <= 0:
                    continue

                # Query FAISS index (returns local indices into self._embeddings)
                # Normalize query vector for consistent cosine similarity with IndexFlatIP
                query_vec = np.array([vo_embedding]).astype('float32')
                norm = np.linalg.norm(query_vec)
                if norm > 0:
                    query_vec = query_vec / norm
                distances, indices = self._embedding_index.search(query_vec, k * 2)

                # Filter to only new segments and check source spacing
                best_match = None
                best_adjusted_conf = 0.0

                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0:
                        continue  # FAISS sentinel for fewer results than k
                    # Convert local FAISS index to global text_metadata index
                    global_idx = idx + global_offset
                    if global_idx < new_segment_start:
                        continue  # Skip segments from earlier passes

                    meta = state.text_metadata[global_idx]
                    video_id = self._extract_video_id_from_path(meta.get('video_path', ''))

                    # Check source spacing
                    if video_id in locked_sources:
                        violates_spacing = any(
                            abs(gap.position - pos) < source_spacing
                            for pos in locked_sources[video_id]
                        )
                        if violates_spacing:
                            continue

                    # FAISS IndexFlatIP returns inner product (cosine similarity
                    # for normalized vectors): higher = more similar, range [0, 1]
                    confidence = max(0.0, min(1.0, float(dist)))

                    # US-94-010: Apply duration tier diversity bonus
                    # Get video duration from metadata and compute tier
                    video_duration = meta.get('total_duration', 0)
                    tier = self._get_duration_tier(video_duration)

                    # Apply diversity bonus: if tier not yet used, add tier_diversity_weight
                    # This encourages using different duration tiers across gaps
                    diversity_bonus = 0.0
                    if tier not in used_tiers:
                        diversity_bonus = tier_diversity_weight

                    # US-105-007: Apply chapter-aware boost
                    # If the gap has a chapter_id, boost confidence for iterative matches
                    # This prioritizes chapter-aligned videos during iterative search
                    gap_chapter_id = getattr(gap, 'chapter_id', None)
                    chapter_bonus = chapter_boost if gap_chapter_id else 0.0

                    adjusted_conf = confidence + diversity_bonus + chapter_bonus

                    if adjusted_conf > best_adjusted_conf and confidence >= target_conf:
                        best_adjusted_conf = adjusted_conf
                        best_match = {
                            'index': global_idx,
                            'video_id': video_id,
                            'confidence': confidence,
                            'adjusted_confidence': adjusted_conf,
                            'meta': meta,
                            'tier': tier,  # US-94-010: Store tier for tracking
                        }

                # Update match if we found a good one
                if best_match:
                    tier = best_match.get('tier', 'unknown')
                    if self._update_match_for_gap(gap, best_match, state):
                        gaps_filled += 1
                        # Track this tier as used for diversity
                        used_tiers.add(tier)

                    # Track this source for future spacing checks
                    locked_sources[best_match['video_id']].append(gap.position)

            return gaps_filled

        except Exception as e:
            log_error_with_context(logger, "MATCH-001", f"Error matching gaps: {e}")
            return 0

    def _precompute_voiceover_embeddings(
        self,
        gaps: List[GapSegment],
        state: 'PipelineState',
        provider: Any
    ) -> Dict[int, Any]:
        """
        Pre-compute embeddings for all gap voiceover segments.

        Batches embedding requests to avoid per-segment API calls,
        significantly reducing latency in the gap matching loop.

        Args:
            gaps: Gap segments that need embeddings
            state: Pipeline state (may have cached embeddings)
            provider: Embedding provider

        Returns:
            Dict mapping segment_index to embedding vector
        """
        embeddings_map: Dict[int, Any] = {}

        # First, collect from cached voiceover embeddings
        if hasattr(state, 'voiceover_embeddings') and state.voiceover_embeddings is not None:
            for gap in gaps:
                if gap.segment_index < len(state.voiceover_embeddings):
                    embeddings_map[gap.segment_index] = state.voiceover_embeddings[gap.segment_index]

        # Find gaps that still need embeddings
        gaps_needing_embed = [g for g in gaps if g.segment_index not in embeddings_map]

        if not gaps_needing_embed:
            logger.debug(f"All {len(gaps)} voiceover embeddings found in cache")
            return embeddings_map

        # Batch compute missing embeddings
        logger.info(f"Pre-computing {len(gaps_needing_embed)} voiceover embeddings (batch)")
        try:
            texts = [g.voiceover_text for g in gaps_needing_embed]
            computed = provider.embed(texts, embed_mode="query")

            if computed is not None:
                for gap, emb in zip(gaps_needing_embed, computed):
                    embeddings_map[gap.segment_index] = emb

                logger.debug(f"Pre-computed {len(computed)} voiceover embeddings")

        except Exception as e:
            logger.warning(f"Batch embedding failed, will compute individually: {e}")

        return embeddings_map

    def _get_voiceover_embedding(
        self,
        segment_index: int,
        text: str,
        state: 'PipelineState',
        provider: Any,
        precomputed: Optional[Dict[int, Any]] = None
    ) -> Optional[Any]:
        """
        Get embedding for a voiceover segment.

        Checks precomputed cache first, then state cache, then computes.

        Args:
            segment_index: Index of the segment
            text: Segment text
            state: Pipeline state
            provider: Embedding provider
            precomputed: Optional dict of precomputed embeddings

        Returns:
            Embedding vector or None
        """
        try:
            # Check precomputed first (from batch computation)
            if precomputed and segment_index in precomputed:
                return precomputed[segment_index]

            # Check if we have cached voiceover embeddings
            if hasattr(state, 'voiceover_embeddings') and state.voiceover_embeddings is not None:
                if segment_index < len(state.voiceover_embeddings):
                    return state.voiceover_embeddings[segment_index]

            # Compute embedding for this text (fallback)
            logger.debug(f"Computing single voiceover embedding for segment {segment_index}")
            embeddings = provider.embed([text], embed_mode="query")
            return embeddings[0] if embeddings is not None else None

        except Exception as e:
            logger.debug(f"Failed to get voiceover embedding: {e}")
            return None

    def _update_match_for_gap(
        self,
        gap: GapSegment,
        best_match: Dict[str, Any],
        state: 'PipelineState'
    ) -> bool:
        """Update the match for a gap segment with a new candidate.

        Returns:
            True if the match was actually updated, False otherwise.
        """
        try:
            # Find the match object for this segment
            if gap.segment_index >= len(state.matches):
                return False

            match = state.matches[gap.segment_index]

            # Handle flat Match objects (from src/state.py) — have video_file, confidence, strategy
            if hasattr(match, 'video_file'):
                match.video_file = best_match['video_id']
                match.confidence = best_match['confidence']
                if hasattr(match, 'strategy'):
                    match.strategy = 'iterative_embedding'
                # Use log_match_context for traceable gap-filling
                log_match_context(
                    logger,
                    logging.DEBUG,
                    f"Gap filled with {best_match['video_id']}",
                    segment_id=gap.segment_index,
                    video_id=best_match['video_id'],
                    time_range=(gap.position, gap.position + 5.0),  # Approximate duration
                    confidence=best_match['confidence'],
                    reason=gap.reason,
                )
                return True

            # Legacy: nested primary_match structure
            if hasattr(match, 'primary_match') and match.primary_match:
                pm = match.primary_match
                if hasattr(pm, 'confidence'):
                    pm.confidence = best_match['confidence']
                if hasattr(pm, 'video_segment'):
                    vs = pm.video_segment
                    if hasattr(vs, 'source_file'):
                        vs.source_file = best_match['video_id']
                    if hasattr(vs, 'text'):
                        vs.text = best_match['meta'].get('text', '')

                # Use log_match_context for traceable gap-filling
                log_match_context(
                    logger,
                    logging.DEBUG,
                    f"Gap filled with {best_match['video_id']}",
                    segment_id=gap.segment_index,
                    video_id=best_match['video_id'],
                    time_range=(gap.position, gap.position + 5.0),
                    confidence=best_match['confidence'],
                    reason=gap.reason,
                )
                return True

            logger.debug(f"Gap {gap.segment_index}: match object has neither video_file nor primary_match")
            return False

        except Exception as e:
            logger.debug(f"Failed to update match for gap: {e}")
            return False

    def _update_query_learning(
        self,
        queries: List[Dict[str, Any]],
        gaps_filled: int,
        gap_analysis: Any,
        learning_db: Any,
        gap_segments: Optional[List[Any]] = None
    ):
        """
        Update query learning database with results.

        Args:
            queries: Queries that were executed
            gaps_filled: Total gaps filled this pass
            gap_analysis: Gap pattern analysis
            learning_db: Learning database instance
            gap_segments: Optional gap segments with chapter_type info
        """
        if not learning_db or not gap_analysis:
            return

        from ..iterative_match import QueryResult

        # Build index -> chapter_type lookup from gap_segments
        chapter_type_by_idx: Dict[int, str] = {}
        if gap_segments:
            for gs in gap_segments:
                chapter_type_by_idx[gs.segment_index] = getattr(gs, 'chapter_type', 'body')

        # Create results for each query
        # (In practice, would track per-query success)
        avg_fill = gaps_filled / len(queries) if queries else 0

        for q in queries:
            gap_indices = q.get('gap_indices', [])
            if not gap_indices:
                continue

            # Determine pattern for this gap
            pattern = 'other'
            for p, indices in gap_analysis.clustered_gaps.items():
                if gap_indices[0] in indices:
                    pattern = p
                    break

            # Determine chapter type for this gap
            chapter_type = chapter_type_by_idx.get(gap_indices[0], 'body')

            result = QueryResult(
                query=q['query'],
                strategy=q.get('strategy', 'unknown'),
                gap_indices=gap_indices,
                videos_found=q.get('videos_found', 0),
                gaps_filled=1 if avg_fill > 0.5 else 0,  # Simplified
                avg_confidence_improvement=0.1 if gaps_filled > 0 else 0.0
            )

            learning_db.record_result(result, pattern, chapter_type=chapter_type)

    # =========================================================================
    # US-101-009: Smart Query Retry Logic
    # =========================================================================

    def _broaden_query(self, query: str) -> str:
        """
        Broaden a query by removing specific terms (names, numbers, etc.).

        Args:
            query: Original search query

        Returns:
            Broadened query with specific terms removed
        """
        import re

        # Remove capitalized names (typically specific people)
        broadened = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', '', query)

        # Remove numbers
        broadened = re.sub(r'\b\d+\b', '', broadened)

        # Remove year patterns (e.g., 2020, 1995)
        broadened = re.sub(r'\b(19|20)\d{2}\b', '', broadened)

        # Clean up extra whitespace
        broadened = ' '.join(broadened.split())

        # If we removed too much, return original
        if len(broadened) < len(query) * 0.3:
            return query

        return broadened if broadened else query

    def _generate_retry_queries(
        self,
        gaps: List[Any],
        locked: List[Any],
        state: 'PipelineState',
        iter_config: Any,
        learning_db: Any,
        pass_num: int
    ) -> List[Dict[str, Any]]:
        """
        Generate retry queries for previously failed queries.

        Strategy progression: voiceover -> similar_locked -> entity
        Each retry applies broadening to remove specific terms.

        Args:
            gaps: Gap segments to generate queries for
            locked: Locked matches
            state: Pipeline state
            iter_config: Iterative matching config
            learning_db: Learning database
            pass_num: Current pass number

        Returns:
            List of retry query dictionaries
        """
        retry_queries = []
        max_retries = 2  # Max 2 retries per query

        # Get queries that haven't reached max retries
        eligible_failures = {
            k: v for k, v in self._failed_queries.items()
            if v.get('retry_count', 0) < max_retries
        }

        if not eligible_failures:
            return []

        logger.info(f"Generating retry queries for {len(eligible_failures)} failed queries...")

        # Strategy progression order
        strategy_order = ['voiceover', 'similar_locked', 'entity']

        for query_key, failure_info in eligible_failures.items():
            original_query = failure_info.get('query', '')
            current_strategy = failure_info.get('strategy', 'voiceover')
            retry_count = failure_info.get('retry_count', 0)
            gap_indices = failure_info.get('gap_indices', [])

            # Determine next strategy
            try:
                current_idx = strategy_order.index(current_strategy)
                next_strategy = strategy_order[min(current_idx + 1, len(strategy_order) - 1)]
            except (ValueError, IndexError):
                next_strategy = 'voiceover'

            # Broaden the query
            broadened_query = self._broaden_query(original_query)

            # Skip if broadening didn't help
            if broadened_query == original_query and retry_count > 0:
                # Already tried broadening, try different strategy
                pass

            # Generate query based on next strategy
            if next_strategy == 'similar_locked' and locked:
                # Use similar-to-locked strategy
                for gap in gaps:
                    if gap.segment_index in gap_indices:
                        nearest = self._find_nearest_locked(gap, locked)
                        if nearest and nearest.video_id:
                            retry_queries.append({
                                'query': f"similar:{nearest.video_id}",
                                'strategy': 'similar_locked',
                                'gap_indices': [gap.segment_index],
                                'priority': 1,
                                'is_retry': True,
                                'original_query': original_query,
                                'retry_count': retry_count + 1
                            })
                        break
            elif next_strategy == 'entity' and state.extracted_entities:
                # Use entity-based query
                entities = list(state.extracted_entities.keys())[:3]
                if entities:
                    entity_query = ' '.join(entities[:2]) + ' ' + broadened_query
                    retry_queries.append({
                        'query': entity_query,
                        'strategy': 'entity',
                        'gap_indices': gap_indices,
                        'priority': 1,
                        'is_retry': True,
                        'original_query': original_query,
                        'retry_count': retry_count + 1
                    })
            else:
                # Default: retry with broadened voiceover query
                retry_queries.append({
                    'query': broadened_query,
                    'strategy': 'voiceover',
                    'gap_indices': gap_indices,
                    'priority': 1,
                    'is_retry': True,
                    'original_query': original_query,
                    'retry_count': retry_count + 1
                })

            # Update failure info with incremented retry count
            self._failed_queries[query_key] = {
                'query': original_query,
                'strategy': next_strategy,
                'retry_count': retry_count + 1,
                'gap_indices': gap_indices
            }

        return retry_queries

    def _track_failed_query(
        self,
        query: str,
        strategy: str,
        gap_indices: List[int],
        gaps_filled: bool
    ) -> None:
        """
        Track a query as failed or successful.

        Args:
            query: The query that was executed
            strategy: Strategy used for this query
            gap_indices: Gap indices this query targeted
            gaps_filled: Whether this query filled any gaps
        """
        # Create a unique key for this query
        query_key = query.lower().strip()

        if gaps_filled:
            # Query succeeded - remove from failed tracking if it was there
            if query_key in self._failed_queries:
                del self._failed_queries[query_key]
        else:
            # Query failed - track or increment retry count
            if query_key in self._failed_queries:
                self._failed_queries[query_key]['retry_count'] += 1
            else:
                self._failed_queries[query_key] = {
                    'query': query,
                    'strategy': strategy,
                    'retry_count': 1,
                    'gap_indices': gap_indices
                }

    def _log_retry_summary(self) -> None:
        """Log summary of retry attempts."""
        if not self._failed_queries:
            return

        max_retries = 2
        permanently_failed = sum(
            1 for f in self._failed_queries.values()
            if f.get('retry_count', 0) >= max_retries
        )
        still_retriable = len(self._failed_queries) - permanently_failed

        logger.info(f"Retry summary: {len(self._failed_queries)} queries tracked, permanently failed: {permanently_failed}, still retriable: {still_retriable}")
