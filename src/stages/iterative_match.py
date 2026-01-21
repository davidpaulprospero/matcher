"""
Iterative Match Stage - High Matches Mode

Orchestrates the iterative download → match → analyze cycle to achieve
target confidence coverage. Continues until:
1. Coverage target achieved
2. Max iterations reached
3. No improvement between iterations
4. No new videos found
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage
from ..matching.coverage_analyzer import analyze_coverage, CoverageReport
from ..matching.recovery_keywords import generate_recovery_keywords
from ..matching.high_matches_logger import HighMatchesLogger
from ..state import IterativeMatchState

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class IterativeMatchStage(Stage):
    """
    Iterative matching stage for high matches mode.

    This stage wraps the normal match flow with iteration logic:
    1. Analyze initial coverage after MATCH stage
    2. If coverage < target, generate recovery keywords
    3. Download new videos based on recovery keywords
    4. Re-run matching with expanded pool
    5. Repeat until coverage target or max iterations
    """

    name = "ITERATIVE_MATCH"
    description = "Iterative matching to achieve target confidence coverage"

    def __init__(self):
        self._iteration = 0
        self._coverage_history: List[float] = []
        self._hmm_logger: Optional[HighMatchesLogger] = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
    ) -> StageResult:
        """Execute iterative matching loop."""
        logger.info("[iterative_match] === ITERATIVE MATCH STAGE STARTING ===")

        # Validate config access - handle dict or object
        hmm_config = config.matching.high_matches_mode

        # Safety: convert dict to object if needed (Rule 2: __post_init__ may not run after merge)
        if isinstance(hmm_config, dict):
            from ..config.sections.matching import HighMatchesModeConfig
            hmm_config = HighMatchesModeConfig(**hmm_config)
            config.matching.high_matches_mode = hmm_config
            logger.info("[iterative_match] Converted high_matches_mode from dict to object")

        logger.info(f"[iterative_match] Config loaded: enabled={hmm_config.enabled}")

        if not hmm_config.enabled:
            logger.info("[iterative_match] High matches mode disabled, skipping stage")
            return StageResult.ok()

        # Initialize dedicated logger for high matches mode
        log_dir = Path(state.project_dir) / "output" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._hmm_logger = HighMatchesLogger(log_dir)
        self._hmm_logger.log_session_start(config, state)

        # Initialize tracking state
        if state.iterative_match_state is None:
            logger.info("[iterative_match] Creating new IterativeMatchState")
            state.iterative_match_state = IterativeMatchState()

        iter_state = state.iterative_match_state
        target_conf = hmm_config.target_confidence
        coverage_target = hmm_config.coverage_target
        max_iterations = hmm_config.max_iterations

        # Log current state for validation
        logger.info("[iterative_match] " + "=" * 50)
        logger.info("[iterative_match] HIGH MATCHES MODE - ENABLED")
        logger.info("[iterative_match] " + "=" * 50)
        logger.info(f"[iterative_match] Configuration:")
        logger.info(f"[iterative_match]   target_confidence: {target_conf}")
        logger.info(f"[iterative_match]   coverage_target: {coverage_target}")
        logger.info(f"[iterative_match]   max_iterations: {max_iterations}")
        logger.info(f"[iterative_match]   videos_per_iteration: {hmm_config.videos_per_iteration}")
        logger.info(f"[iterative_match]   keyword_strategy: {hmm_config.keyword_strategy}")
        logger.info(f"[iterative_match] Current state:")
        logger.info(f"[iterative_match]   matches: {len(state.matches) if state.matches else 0}")
        logger.info(f"[iterative_match]   voiceover_segments: {len(state.voiceover_segments) if state.voiceover_segments else 0}")
        logger.info(f"[iterative_match]   downloaded_videos: {len(state.downloaded_videos) if state.downloaded_videos else 0}")
        logger.info(f"[iterative_match]   keywords: {len(state.keywords) if state.keywords else 0}")
        logger.info("[iterative_match] " + "=" * 50)

        # Analyze initial coverage
        coverage = self._analyze_coverage(state, target_conf)
        iter_state.coverage_history.append(coverage.coverage_ratio)
        self._hmm_logger.log_coverage_analysis(coverage, iteration=0)

        # Check if we already meet target
        if coverage.coverage_ratio >= coverage_target:
            logger.info(f"Initial coverage {coverage.coverage_ratio:.1%} meets target!")
            iter_state.target_achieved = True
            iter_state.final_coverage = coverage.coverage_ratio
            self._print_report(iter_state, coverage, config)
            self._hmm_logger.log_final_report(iter_state, coverage)
            return StageResult.ok(data=self._get_result_data(iter_state))

        # Iterative improvement loop
        logger.info("[iterative_match] " + "=" * 60)
        logger.info("[iterative_match] ENTERING ITERATION LOOP")
        logger.info("[iterative_match] " + "=" * 60)
        loop_iteration = 0
        while not self._should_stop(coverage, iter_state, config):
            loop_iteration += 1
            iter_state.iteration_count += 1
            logger.info(f"\n[iterative_match] {'=' * 60}")
            logger.info(f"[iterative_match] ITERATION {iter_state.iteration_count} START (loop #{loop_iteration})")
            logger.info(f"[iterative_match] {'=' * 60}")
            logger.info(f"[iterative_match] Current coverage: {coverage.coverage_ratio:.1%}")
            logger.info(f"[iterative_match] Weak segments: {len(coverage.weak_segments)}")
            logger.info(f"[iterative_match] text_metadata entries: {len(state.text_metadata) if state.text_metadata else 0}")
            logger.info(f"[iterative_match] video_candidates: {len(state.video_candidates) if state.video_candidates else 0}")
            logger.info(f"[iterative_match] downloaded_videos: {len(state.downloaded_videos) if state.downloaded_videos else 0}")
            self._hmm_logger.log_iteration_start(
                iter_state.iteration_count,
                coverage.coverage_ratio,
                len(coverage.weak_segments)
            )

            # Generate recovery keywords for weak segments
            keywords = generate_recovery_keywords(
                weak_segments=coverage.weak_segments,
                existing_keywords=state.keywords,
                existing_videos=state.downloaded_videos,
                strategy=hmm_config.keyword_strategy,
                max_keywords=hmm_config.videos_per_iteration,
                config=config,  # Pass config for LLM strategy
            )

            if not keywords:
                logger.warning("No recovery keywords generated - stopping iteration")
                self._hmm_logger.log_iteration_end(
                    iter_state.iteration_count,
                    coverage.coverage_ratio,
                    continue_iteration=False,
                    reason="No recovery keywords generated"
                )
                break

            logger.info(f"Recovery keywords: {keywords}")
            self._hmm_logger.log_recovery_keywords(
                keywords,
                hmm_config.keyword_strategy,
                len(coverage.weak_segments),
                iter_state.iteration_count
            )

            # Download new videos
            # In caption-first mode, count video_candidates not downloaded_videos
            # because most videos have captions and don't need audio download
            candidates_before = len(state.video_candidates) if state.video_candidates else 0
            videos_before = len(state.downloaded_videos) if state.downloaded_videos else 0
            text_metadata_before = len(state.text_metadata) if state.text_metadata else 0

            logger.info(f"[iterative_match] BEFORE DOWNLOAD:")
            logger.info(f"[iterative_match]   candidates_before={candidates_before}")
            logger.info(f"[iterative_match]   videos_before={videos_before}")
            logger.info(f"[iterative_match]   text_metadata_before={text_metadata_before}")

            success = self._download_videos(state, config, checkpoint, keywords)

            candidates_after = len(state.video_candidates) if state.video_candidates else 0
            videos_after = len(state.downloaded_videos) if state.downloaded_videos else 0
            text_metadata_after = len(state.text_metadata) if state.text_metadata else 0

            logger.info(f"[iterative_match] AFTER DOWNLOAD:")
            logger.info(f"[iterative_match]   candidates_after={candidates_after}")
            logger.info(f"[iterative_match]   videos_after={videos_after}")
            logger.info(f"[iterative_match]   text_metadata_after={text_metadata_after}")
            logger.info(f"[iterative_match]   download_success={success}")

            # Use whichever grew more - candidates (caption-first) or downloads (traditional)
            candidates_added = candidates_after - candidates_before
            downloads_added = videos_after - videos_before
            text_metadata_added = text_metadata_after - text_metadata_before
            videos_added = max(candidates_added, downloads_added)

            logger.info(f"[iterative_match] VIDEOS ADDED CALCULATION:")
            logger.info(f"[iterative_match]   candidates_added={candidates_added}")
            logger.info(f"[iterative_match]   downloads_added={downloads_added}")
            logger.info(f"[iterative_match]   text_metadata_added={text_metadata_added}")
            logger.info(f"[iterative_match]   videos_added (max of candidates/downloads)={videos_added}")

            iter_state.videos_added_per_iteration.append(videos_added)
            self._hmm_logger.log_download_result(
                videos_added,
                videos_after,
                success,
                iter_state.iteration_count
            )

            logger.info(f"[iterative_match] CHECK: videos_added == 0? ({videos_added} == 0 is {videos_added == 0})")

            if videos_added == 0:
                # YouTube found no new videos - try multiple recovery strategies
                logger.info("[iterative_match] ⚠️ YOUTUBE RETURNED 0 NEW VIDEOS")
                logger.info("[iterative_match] Attempting recovery strategies...")
                self._hmm_logger._append_text(f"[{datetime.now().strftime('%H:%M:%S')}] RECOVERY STRATEGIES (0 new videos from YouTube)\n")

                # Strategy 1: Lower relevance threshold and retry YouTube
                logger.info("[iterative_match] Strategy 1: Lowering relevance threshold to 0.3")
                videos_added = self._retry_with_lower_threshold(
                    state, config, checkpoint, keywords, threshold=0.3
                )
                if videos_added > 0:
                    logger.info(f"[iterative_match] ✅ Strategy 1 succeeded: {videos_added} new videos")
                    self._hmm_logger.log_recovery_strategy("lower_threshold", "success", videos_added, iter_state.iteration_count)
                    self._hmm_logger.log_download_result(
                        videos_added, len(state.video_candidates), True,
                        iter_state.iteration_count, source="youtube_lowthresh"
                    )
                else:
                    self._hmm_logger.log_recovery_strategy("lower_threshold", "failed", 0, iter_state.iteration_count)

                if videos_added == 0:
                    # Strategy 2: Try keyword variations
                    logger.info("[iterative_match] Strategy 2: Trying keyword variations")
                    varied_keywords = self._generate_keyword_variations(keywords)
                    candidates_before_s2 = len(state.video_candidates) if state.video_candidates else 0
                    self._download_videos(state, config, checkpoint, varied_keywords)
                    candidates_after_s2 = len(state.video_candidates) if state.video_candidates else 0
                    videos_added = candidates_after_s2 - candidates_before_s2
                    if videos_added > 0:
                        logger.info(f"[iterative_match] ✅ Strategy 2 succeeded: {videos_added} new videos")
                        self._hmm_logger.log_recovery_strategy("keyword_variations", "success", videos_added, iter_state.iteration_count)
                        self._hmm_logger.log_download_result(
                            videos_added, len(state.video_candidates), True,
                            iter_state.iteration_count, source="youtube_variations"
                        )
                    else:
                        self._hmm_logger.log_recovery_strategy("keyword_variations", "failed", 0, iter_state.iteration_count)

                if videos_added == 0:
                    # Strategy 3: Try Pexels stock footage
                    logger.info("[iterative_match] Strategy 3: Pexels fallback")
                    pexels_added = self._download_pexels_videos(state, config, keywords)
                    if pexels_added > 0:
                        videos_added = pexels_added
                        logger.info(f"[iterative_match] ✅ Strategy 3 succeeded: {pexels_added} Pexels videos")
                        self._hmm_logger.log_recovery_strategy("pexels_fallback", "success", pexels_added, iter_state.iteration_count)
                        self._hmm_logger.log_download_result(
                            pexels_added, len(state.downloaded_videos), True,
                            iter_state.iteration_count, source="pexels"
                        )
                    else:
                        self._hmm_logger.log_recovery_strategy("pexels_fallback", "failed", 0, iter_state.iteration_count)

                if videos_added == 0:
                    # Strategy 4: Generate completely new keywords from different weak segments
                    logger.info("[iterative_match] Strategy 4: Fresh keywords from random weak segments")
                    fresh_keywords = self._generate_fresh_keywords(coverage, state, config)
                    if fresh_keywords:
                        candidates_before_s4 = len(state.video_candidates) if state.video_candidates else 0
                        self._download_videos(state, config, checkpoint, fresh_keywords)
                        candidates_after_s4 = len(state.video_candidates) if state.video_candidates else 0
                        videos_added = candidates_after_s4 - candidates_before_s4
                        if videos_added > 0:
                            logger.info(f"[iterative_match] ✅ Strategy 4 succeeded: {videos_added} new videos")
                            self._hmm_logger.log_recovery_strategy("fresh_keywords", "success", videos_added, iter_state.iteration_count)
                            self._hmm_logger.log_download_result(
                                videos_added, len(state.video_candidates), True,
                                iter_state.iteration_count, source="youtube_fresh"
                            )
                        else:
                            self._hmm_logger.log_recovery_strategy("fresh_keywords", "failed", 0, iter_state.iteration_count)
                    else:
                        self._hmm_logger.log_recovery_strategy("fresh_keywords", "skipped", 0, iter_state.iteration_count)

                # Final check - if still 0, then break
                if videos_added == 0:
                    logger.warning("[iterative_match] ❌ ALL RECOVERY STRATEGIES EXHAUSTED")
                    logger.warning("[iterative_match]   Tried: lower threshold, keyword variations, Pexels, fresh keywords")
                    logger.warning("[iterative_match]   No new videos found from any source")
                    self._hmm_logger.log_iteration_end(
                        iter_state.iteration_count,
                        coverage.coverage_ratio,
                        continue_iteration=False,
                        reason="All recovery strategies exhausted"
                    )
                    break
            else:
                logger.info(f"[iterative_match] ✅ YouTube found {videos_added} new videos, continuing")

            logger.info(f"[iterative_match] PROCEEDING WITH {videos_added} NEW VIDEOS")

            # Re-run transcription for new videos
            self._transcribe_new_videos(state, config, checkpoint)

            # Re-run matching with expanded pool
            self._rematch(state, config, checkpoint)

            # Re-analyze coverage
            previous_coverage = coverage.coverage_ratio
            coverage = self._analyze_coverage(state, target_conf)
            iter_state.coverage_history.append(coverage.coverage_ratio)
            self._hmm_logger.log_coverage_analysis(coverage, iter_state.iteration_count)
            self._hmm_logger.log_rematch_result(
                coverage.coverage_ratio,
                previous_coverage,
                len(state.matches) if state.matches else 0,
                iter_state.iteration_count
            )

            # Check for improvement
            logger.info(f"[iterative_match] COVERAGE HISTORY: {iter_state.coverage_history}")
            if len(iter_state.coverage_history) >= 2:
                prev = iter_state.coverage_history[-2]
                curr = iter_state.coverage_history[-1]
                delta = curr - prev
                logger.info(f"[iterative_match] IMPROVEMENT CHECK:")
                logger.info(f"[iterative_match]   previous={prev:.3f}, current={curr:.3f}, delta={delta:.3f}")
                logger.info(f"[iterative_match]   delta < 0.01? ({delta:.3f} < 0.01 is {delta < 0.01})")
                if delta < 0.01:  # Less than 1% improvement
                    logger.warning(f"[iterative_match] ❌ MINIMAL IMPROVEMENT - BREAKING LOOP")
                    logger.warning(f"[iterative_match]   Improvement {delta:.1%} is less than 1% threshold")
                    self._hmm_logger.log_iteration_end(
                        iter_state.iteration_count,
                        curr,
                        continue_iteration=False,
                        reason=f"Minimal improvement ({delta:.1%})"
                    )
                    break
                else:
                    # Log successful iteration - continuing
                    logger.info(f"[iterative_match] ✅ GOOD IMPROVEMENT ({delta:.1%}), CONTINUING TO NEXT ITERATION")
                    self._hmm_logger.log_iteration_end(
                        iter_state.iteration_count,
                        curr,
                        continue_iteration=True,
                        reason=f"Improved by {delta:.1%}"
                    )
            else:
                logger.info(f"[iterative_match] First iteration, no previous coverage to compare")

            logger.info(f"[iterative_match] END OF ITERATION {iter_state.iteration_count}")
            logger.info(f"[iterative_match] Looping back to _should_stop check...")

        # Log why loop exited
        logger.info("[iterative_match] " + "=" * 60)
        logger.info("[iterative_match] ITERATION LOOP ENDED")
        logger.info("[iterative_match] " + "=" * 60)
        logger.info(f"[iterative_match] Total iterations completed: {iter_state.iteration_count}")
        logger.info(f"[iterative_match] Final coverage: {coverage.coverage_ratio:.1%}")
        logger.info(f"[iterative_match] Coverage target: {coverage_target:.1%}")
        logger.info(f"[iterative_match] Target achieved: {coverage.coverage_ratio >= coverage_target}")

        # Final state update
        iter_state.final_coverage = coverage.coverage_ratio
        iter_state.target_achieved = coverage.coverage_ratio >= coverage_target
        iter_state.weak_segment_count = len(coverage.weak_segments)

        # Print final report
        self._print_report(iter_state, coverage, config)
        self._hmm_logger.log_final_report(iter_state, coverage)

        # Checkpoint the iteration state
        checkpoint.save(self.name, self._get_result_data(iter_state))

        return StageResult.ok(data=self._get_result_data(iter_state))

    def _analyze_coverage(
        self,
        state: 'PipelineState',
        target_confidence: float,
    ) -> CoverageReport:
        """Analyze current match coverage."""
        logger.info(f"[iterative_match] Analyzing coverage (target: {target_confidence:.0%})")
        logger.debug(f"[iterative_match]   matches type: {type(state.matches)}, len: {len(state.matches) if state.matches else 0}")
        logger.debug(f"[iterative_match]   segments type: {type(state.voiceover_segments)}, len: {len(state.voiceover_segments) if state.voiceover_segments else 0}")

        report = analyze_coverage(
            matches=state.matches,
            voiceover_segments=state.voiceover_segments,
            target_confidence=target_confidence,
        )

        logger.info(f"[iterative_match] Coverage result: {report.coverage_ratio:.1%} ({report.high_confidence}/{report.total_segments})")
        return report

    def _should_stop(
        self,
        coverage: CoverageReport,
        iter_state: IterativeMatchState,
        config: 'Config',
    ) -> bool:
        """Check if iteration should stop."""
        hmm_config = config.matching.high_matches_mode

        logger.info(f"[_should_stop] CHECKING STOP CONDITIONS:")
        logger.info(f"[_should_stop]   coverage_ratio={coverage.coverage_ratio:.3f}, target={hmm_config.coverage_target}")
        logger.info(f"[_should_stop]   iteration_count={iter_state.iteration_count}, max={hmm_config.max_iterations}")
        logger.info(f"[_should_stop]   weak_segments={len(coverage.weak_segments)}")

        # Coverage target achieved
        if coverage.coverage_ratio >= hmm_config.coverage_target:
            logger.info(f"[_should_stop] → STOP: Coverage target {hmm_config.coverage_target:.0%} achieved!")
            return True

        # Max iterations reached
        if iter_state.iteration_count >= hmm_config.max_iterations:
            logger.info(f"[_should_stop] → STOP: Max iterations ({hmm_config.max_iterations}) reached")
            return True

        # No weak segments left
        if not coverage.weak_segments:
            logger.info("[_should_stop] → STOP: No weak segments remaining")
            return True

        logger.info(f"[_should_stop] → CONTINUE: All conditions allow more iterations")
        return False

    def _download_videos(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        keywords: List[str],
    ) -> bool:
        """Download videos for recovery keywords.

        Key change: Clears video_candidates before searching to avoid
        the dedup logic skipping videos that could match weak segments.
        Also increases search depth for better coverage.
        """
        logger.info(f"[iterative_match] Starting video download for recovery keywords")
        logger.info(f"[iterative_match]   Recovery keywords: {keywords}")

        try:
            # Import here to avoid circular imports
            from .download import DownloadStage
            from .video_metadata import VideoMetadataStage

            # Save original state
            original_keywords = state.keywords.copy() if state.keywords else []
            original_candidates = state.video_candidates.copy() if state.video_candidates else []
            original_candidate_ids = {vc.video_id for vc in original_candidates}

            logger.info(f"[iterative_match]   Original keywords: {len(original_keywords)}")
            logger.info(f"[iterative_match]   Original candidates: {len(original_candidates)}")

            # CRITICAL: Clear video_candidates so dedup doesn't skip new finds
            # This allows YouTube to return videos we already have but haven't
            # downloaded yet (caption-first mode stores IDs before downloading)
            state.video_candidates = []
            state.keywords = keywords  # Only search for recovery keywords

            # Temporarily increase search depth for recovery
            original_search_multiplier = getattr(config.download, 'search_pool_multiplier', 5)
            original_max_pool = getattr(config.download, 'max_search_pool', 100)

            # Increase search depth for recovery (2x normal)
            config.download.search_pool_multiplier = max(10, original_search_multiplier * 2)
            config.download.max_search_pool = max(200, original_max_pool * 2)

            # CRITICAL: Bypass search cache for recovery iterations
            # Without this, cached results return the same video IDs we already have
            original_bypass_cache = getattr(config.download, 'bypass_search_cache', False)
            config.download.bypass_search_cache = True

            logger.info(f"[iterative_match]   Increased search depth: multiplier={config.download.search_pool_multiplier}, max_pool={config.download.max_search_pool}")
            logger.info(f"[iterative_match]   BYPASS SEARCH CACHE: True (to find new videos)")

            # Run video metadata stage to find videos
            logger.info(f"[iterative_match]   Running VideoMetadataStage...")
            metadata_stage = VideoMetadataStage()
            result = metadata_stage.run(state, config, checkpoint)

            # Restore search config
            config.download.search_pool_multiplier = original_search_multiplier
            config.download.max_search_pool = original_max_pool
            config.download.bypass_search_cache = original_bypass_cache

            if not result.success:
                logger.warning(f"[iterative_match] Video metadata stage failed: {result.error}")
                state.keywords = original_keywords
                state.video_candidates = original_candidates
                return False

            # Count truly new candidates (not in original set)
            new_candidates = [vc for vc in state.video_candidates if vc.video_id not in original_candidate_ids]
            logger.info(f"[iterative_match]   VIDEO_METADATA RESULT:")
            logger.info(f"[iterative_match]     state.video_candidates: {len(state.video_candidates)}")
            logger.info(f"[iterative_match]     original_candidate_ids: {len(original_candidate_ids)}")
            logger.info(f"[iterative_match]     truly new candidates: {len(new_candidates)}")
            if new_candidates:
                logger.info(f"[iterative_match]     Sample new IDs: {[vc.video_id for vc in new_candidates[:5]]}")
            else:
                logger.warning(f"[iterative_match]     ⚠️ NO NEW CANDIDATES FOUND - this may cause iteration to stop")

            # Merge: add original candidates back, avoiding duplicates
            merged_ids = {vc.video_id for vc in state.video_candidates}
            for vc in original_candidates:
                if vc.video_id not in merged_ids:
                    state.video_candidates.append(vc)
                    merged_ids.add(vc.video_id)

            logger.info(f"[iterative_match]   After merge: {len(state.video_candidates)} total candidates")
            logger.info(f"[iterative_match]   VideoMetadataStage completed successfully")

            # Run CAPTION stage first (for caption-first mode)
            # This fetches captions for new candidates before download
            from .caption import CaptionStage
            text_metadata_before_caption = len(state.text_metadata) if state.text_metadata else 0
            logger.info(f"[iterative_match]   Running CaptionStage for new candidates...")
            logger.info(f"[iterative_match]   text_metadata BEFORE caption: {text_metadata_before_caption}")
            caption_stage = CaptionStage()
            caption_result = caption_stage.run(state, config, checkpoint)
            text_metadata_after_caption = len(state.text_metadata) if state.text_metadata else 0
            logger.info(f"[iterative_match]   text_metadata AFTER caption: {text_metadata_after_caption}")
            logger.info(f"[iterative_match]   Caption added {text_metadata_after_caption - text_metadata_before_caption} entries")
            if caption_result.success:
                logger.info(f"[iterative_match]   CaptionStage completed successfully")
            else:
                logger.warning(f"[iterative_match]   CaptionStage had issues: {caption_result.error}")

            # Run download stage (downloads audio for videos without captions)
            logger.info(f"[iterative_match]   Running DownloadStage...")
            download_stage = DownloadStage()
            result = download_stage.run(state, config, checkpoint)

            # Restore original keywords + new ones
            state.keywords = list(set(original_keywords + keywords))
            logger.info(f"[iterative_match]   Total keywords now: {len(state.keywords)}")

            if result.success:
                logger.info(f"[iterative_match]   DownloadStage completed successfully")
            else:
                logger.warning(f"[iterative_match]   DownloadStage failed: {result.error}")

            return result.success

        except Exception as e:
            logger.error(f"[iterative_match] Error downloading videos: {e}", exc_info=True)
            return False

    def _download_pexels_videos(
        self,
        state: 'PipelineState',
        config: 'Config',
        keywords: List[str],
    ) -> int:
        """Download videos from Pexels using recovery keywords.

        Called when YouTube returns no new videos. Searches multiple pages
        to find videos not already in the pool.

        Args:
            state: Pipeline state
            config: Config object
            keywords: Recovery keywords to search

        Returns:
            Number of new videos added
        """
        logger.info(f"[iterative_match] === PEXELS RECOVERY DOWNLOAD ===")

        # Check if Pexels is enabled
        pexels_config = getattr(config, 'pexels', None)
        if not pexels_config or not getattr(pexels_config, 'enabled', False):
            logger.info("[iterative_match] Pexels not enabled in config, skipping")
            return 0

        try:
            import os
            from ..pexels import PexelsDownloader
            from ..state import DownloadedVideo

            api_key = os.environ.get("PEXELS_API_KEY")
            if not api_key:
                logger.warning("[iterative_match] PEXELS_API_KEY not set, skipping Pexels")
                return 0

            # Pexels config - be more aggressive for recovery
            videos_per_kw = getattr(pexels_config, 'videos_per_keyword', 3)
            min_duration = getattr(pexels_config, 'min_duration', 3)
            max_duration = getattr(pexels_config, 'max_duration', 30)
            orientation = getattr(pexels_config, 'orientation', 'landscape')

            # Recovery settings - search more aggressively
            max_pages = 3  # Search pages 1, 2, 3 to find new content
            max_keywords = 20  # Use more keywords than normal
            target_new_videos = 30  # Stop after finding this many new videos

            # Output directory
            stock_dir = Path(config.downloaded_videos_dir) / "stock"
            stock_dir.mkdir(parents=True, exist_ok=True)

            downloader = PexelsDownloader(api_key=api_key, output_dir=str(stock_dir))

            # Track existing video IDs from ALL Pexels videos (downloaded + in pool)
            existing_ids = set()

            # Check downloaded_videos
            if state.downloaded_videos:
                for dv in state.downloaded_videos:
                    file_path = dv.file if hasattr(dv, 'file') else str(dv)
                    if 'pexels_' in str(file_path):
                        match = re.search(r'pexels_(\d+)_', str(file_path))
                        if match:
                            existing_ids.add(int(match.group(1)))

            # Also check stock directory for any existing files
            if stock_dir.exists():
                for f in stock_dir.glob("pexels_*.mp4"):
                    match = re.search(r'pexels_(\d+)_', f.name)
                    if match:
                        existing_ids.add(int(match.group(1)))

            logger.info(f"[iterative_match] Found {len(existing_ids)} existing Pexels videos to skip")

            new_videos = []
            videos_added = 0

            # Search each keyword across multiple pages
            for kw in keywords[:max_keywords]:
                if videos_added >= target_new_videos:
                    logger.info(f"[iterative_match] Reached target of {target_new_videos} new videos")
                    break

                # Simplify keyword for Pexels (remove "footage", "4K", etc.)
                search_kw = re.sub(r'\b(footage|4[kK]|stock|video|cinematic|S\d+)\b', '', kw).strip()
                search_kw = re.sub(r'\s+', ' ', search_kw).strip()
                if not search_kw or len(search_kw) < 3:
                    continue

                logger.info(f"[iterative_match] Pexels search: '{search_kw}'")

                # Search multiple pages to find new content
                for page in range(1, max_pages + 1):
                    if videos_added >= target_new_videos:
                        break

                    try:
                        videos = downloader.search(
                            query=search_kw,
                            per_page=15,  # Max results per page
                            page=page,
                            min_duration=min_duration,
                            max_duration=max_duration,
                            orientation=orientation,
                        )
                    except Exception as e:
                        logger.warning(f"[iterative_match] Pexels search error page {page}: {e}")
                        continue

                    if not videos:
                        break  # No more results for this keyword

                    downloaded_count = 0
                    skipped_existing = 0

                    for video in videos:
                        if video.id in existing_ids:
                            skipped_existing += 1
                            continue

                        if downloaded_count >= videos_per_kw:
                            break

                        if videos_added >= target_new_videos:
                            break

                        file_path = downloader.download(video)
                        if file_path:
                            existing_ids.add(video.id)
                            downloaded_count += 1
                            videos_added += 1

                            # Add to state.downloaded_videos
                            new_dv = DownloadedVideo(
                                file=file_path,
                                url=video.url,
                                title=f"Pexels: {video.photographer} ({video.id})",
                                duration=float(video.duration),
                                source='pexels',
                                keyword=kw,
                            )
                            state.downloaded_videos.append(new_dv)
                            new_videos.append(new_dv)

                    if downloaded_count > 0:
                        logger.info(f"[iterative_match]   Page {page}: +{downloaded_count} new (skipped {skipped_existing} existing)")
                    elif skipped_existing > 0 and page < max_pages:
                        # All results on this page were existing, try next page
                        continue
                    else:
                        break  # No new videos found, stop searching this keyword

            logger.info(f"[iterative_match] === PEXELS RECOVERY COMPLETE: {videos_added} new videos ===")

            # If Pexels found new videos, they need to be transcribed
            if videos_added > 0:
                logger.info(f"[iterative_match] New Pexels videos will be transcribed in next step")

            return videos_added

        except ImportError as e:
            logger.warning(f"[iterative_match] Pexels import failed: {e}")
            return 0
        except Exception as e:
            logger.error(f"[iterative_match] Pexels download error: {e}", exc_info=True)
            return 0

    def _retry_with_lower_threshold(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        keywords: List[str],
        threshold: float = 0.3,
    ) -> int:
        """Retry YouTube search with lower relevance threshold.

        When normal threshold (0.5) finds no new videos, lowering to 0.3
        may find relevant videos that were borderline filtered.
        """
        logger.info(f"[iterative_match] Retrying with lower threshold: {threshold}")

        try:
            # Save original threshold
            original_threshold = getattr(config.download, 'min_relevance', 0.5)

            # Lower threshold
            config.download.min_relevance = threshold
            logger.info(f"[iterative_match]   Lowered min_relevance from {original_threshold} to {threshold}")

            # Track candidates before
            candidates_before = len(state.video_candidates) if state.video_candidates else 0

            # Run download with lower threshold
            success = self._download_videos(state, config, checkpoint, keywords)

            # Restore threshold
            config.download.min_relevance = original_threshold

            # Count new candidates
            candidates_after = len(state.video_candidates) if state.video_candidates else 0
            videos_added = candidates_after - candidates_before

            logger.info(f"[iterative_match]   Lower threshold result: {videos_added} new videos")
            return videos_added

        except Exception as e:
            logger.error(f"[iterative_match] Lower threshold retry failed: {e}")
            return 0

    def _generate_keyword_variations(self, keywords: List[str]) -> List[str]:
        """Generate variations of keywords with different modifiers.

        Adds terms like 'HD', '4K', 'cinematic', 'royalty free' to
        find videos that might be indexed differently.
        """
        import random

        modifiers = [
            "HD", "4K", "cinematic", "free", "royalty free",
            "b-roll", "stock video", "compilation", "montage"
        ]

        # Also try removing existing modifiers to search more broadly
        strip_terms = ["footage", "stock", "4K", "HD", "cinematic"]

        variations = []

        for kw in keywords[:15]:  # Limit to avoid too many variations
            # Try adding modifiers
            modifier = random.choice(modifiers)
            variations.append(f"{kw} {modifier}")

            # Try stripping common terms for broader search
            stripped = kw
            for term in strip_terms:
                stripped = stripped.replace(term, "").strip()
            if stripped and stripped != kw and len(stripped) > 5:
                variations.append(stripped)

        # Remove duplicates and shuffle
        variations = list(set(variations))
        random.shuffle(variations)

        logger.info(f"[iterative_match] Generated {len(variations)} keyword variations")
        if variations:
            logger.info(f"[iterative_match]   Sample: {variations[:5]}")

        return variations[:20]  # Limit to 20 variations

    def _generate_fresh_keywords(
        self,
        coverage: 'CoverageReport',
        state: 'PipelineState',
        config: 'Config',
    ) -> List[str]:
        """Generate completely fresh keywords from different weak segments.

        Instead of using the same weak segments, pick a random sample
        of segments that haven't been targeted yet.
        """
        import random

        if not coverage.weak_segments:
            return []

        # Get segments not recently used (sample from bottom 50%)
        available_segments = coverage.weak_segments[len(coverage.weak_segments)//2:]

        if not available_segments:
            available_segments = coverage.weak_segments

        # Random sample of weak segments
        sample_size = min(10, len(available_segments))
        sample = random.sample(available_segments, sample_size)

        logger.info(f"[iterative_match] Generating fresh keywords from {sample_size} random weak segments")

        try:
            # Generate keywords with simple text extraction
            fresh_keywords = []
            for seg in sample:
                # Extract key phrases from segment text
                text = seg.text.lower()
                # Remove common words
                stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                             'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                             'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                             'from', 'as', 'into', 'through', 'during', 'before', 'after',
                             'above', 'below', 'between', 'and', 'but', 'or', 'so', 'yet',
                             'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you',
                             'he', 'she', 'they', 'we', 'what', 'which', 'who', 'when',
                             'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                             'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                             'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'}

                words = [w for w in text.split() if w not in stopwords and len(w) > 2]
                if words:
                    # Create a search phrase from key words
                    key_phrase = " ".join(words[:4])
                    fresh_keywords.append(f"{key_phrase} footage")

            logger.info(f"[iterative_match]   Generated {len(fresh_keywords)} fresh keywords")
            return fresh_keywords

        except Exception as e:
            logger.error(f"[iterative_match] Fresh keyword generation failed: {e}")
            return []

    def _transcribe_new_videos(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
    ) -> bool:
        """Transcribe newly downloaded videos."""
        logger.info(f"[iterative_match] Starting transcription for new videos")
        try:
            from .transcribe import TranscribeStage

            # Transcribe stage handles incremental transcription
            stage = TranscribeStage()
            result = stage.run(state, config, checkpoint)

            if result.success:
                logger.info(f"[iterative_match] TranscribeStage completed successfully")
                logger.info(f"[iterative_match]   text_metadata entries: {len(state.text_metadata) if state.text_metadata else 0}")
            else:
                logger.warning(f"[iterative_match] TranscribeStage failed: {result.error}")

            return result.success

        except Exception as e:
            logger.error(f"[iterative_match] Error transcribing: {e}", exc_info=True)
            return False

    def _rematch(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
    ) -> bool:
        """Re-run matching with expanded video pool."""
        logger.info(f"[iterative_match] Starting re-matching with expanded video pool")
        logger.info(f"[iterative_match]   Current matches: {len(state.matches) if state.matches else 0}")

        try:
            from .match import MatchStage

            # Clear existing matches for fresh matching
            logger.info(f"[iterative_match]   Clearing existing matches...")
            state.clear_matches()

            stage = MatchStage()
            result = stage.run(state, config, checkpoint)

            if result.success:
                logger.info(f"[iterative_match] MatchStage completed successfully")
                logger.info(f"[iterative_match]   New matches: {len(state.matches) if state.matches else 0}")
            else:
                logger.warning(f"[iterative_match] MatchStage failed: {result.error}")

            return result.success

        except Exception as e:
            logger.error(f"[iterative_match] Error matching: {e}", exc_info=True)
            return False

    def _print_report(
        self,
        iter_state: IterativeMatchState,
        coverage: CoverageReport,
        config: 'Config',
    ) -> None:
        """Print final iteration report."""
        hmm_config = config.matching.high_matches_mode

        logger.info("")
        logger.info("=" * 60)
        logger.info("HIGH MATCHES MODE REPORT")
        logger.info("=" * 60)
        logger.info(f"Target: {hmm_config.target_confidence:.0%} confidence, "
                    f"{hmm_config.coverage_target:.0%} coverage")
        logger.info("")
        logger.info(f"Iterations: {iter_state.iteration_count}")
        total_added = sum(iter_state.videos_added_per_iteration)
        logger.info(f"Videos added: {total_added}")
        logger.info("")
        logger.info("Coverage progression:")
        for i, cov in enumerate(iter_state.coverage_history):
            label = "Initial" if i == 0 else f"Iteration {i}"
            marker = " ✓ TARGET MET" if cov >= hmm_config.coverage_target else ""
            logger.info(f"  {label}: {cov:.1%} "
                        f"({int(cov * coverage.total_segments)}/{coverage.total_segments}){marker}")
        logger.info("")
        if iter_state.target_achieved:
            logger.info("✅ Target coverage achieved!")
        else:
            logger.info(f"⚠️ Target not achieved. Remaining weak segments: "
                        f"{len(coverage.weak_segments)}")
            if coverage.weak_segments:
                lowest = coverage.weak_segments[0]
                logger.info(f"  Lowest: {lowest.segment_id} "
                            f"'{lowest.text[:50]}...' ({lowest.current_confidence:.2f})")
        logger.info("=" * 60)

    def _get_result_data(self, iter_state: IterativeMatchState) -> Dict[str, Any]:
        """Get data for checkpointing."""
        return {
            'iteration_count': iter_state.iteration_count,
            'coverage_history': iter_state.coverage_history,
            'videos_added_per_iteration': iter_state.videos_added_per_iteration,
            'final_coverage': iter_state.final_coverage,
            'target_achieved': iter_state.target_achieved,
            'weak_segment_count': iter_state.weak_segment_count,
        }

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
    ) -> bool:
        """Check if stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None,
    ) -> bool:
        """Restore stage from checkpoint."""
        data = checkpoint.get_stage_data(self.name)
        if not data:
            return False

        state.iterative_match_state = IterativeMatchState(
            iteration_count=data.get('iteration_count', 0),
            coverage_history=data.get('coverage_history', []),
            videos_added_per_iteration=data.get('videos_added_per_iteration', []),
            final_coverage=data.get('final_coverage', 0.0),
            target_achieved=data.get('target_achieved', False),
            weak_segment_count=data.get('weak_segment_count', 0),
        )
        return True
