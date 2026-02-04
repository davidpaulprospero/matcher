"""
Download orchestrator - batch coordination logic extracted from core.py.

Handles:
- Batch processing across multiple keywords
- Retry queue coordination
- Progress tracking and checkpointing
- Source diversity reporting

The VideoDownloader in core.py focuses on single-video download mechanics,
while this orchestrator handles the higher-level batch coordination.
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING, Callable

from ..state import DownloadedVideo
from ..rate_limit.coordinator import GlobalRateLimitCoordinator
from .checkpoint import DownloadCheckpoint
from .rate_limit_metrics import RateLimitMetrics
from .rate_limit_budget import RateLimitBudget
from .escalation_manager import EscalationManager

if TYPE_CHECKING:
    from .core import VideoDownloader

logger = logging.getLogger(__name__)


class RateLimitHooks:
    """
    Pre/post download hooks for rate limit slot management.

    Integrates with GlobalRateLimitCoordinator to acquire slots before
    downloads and release them after, with error handling for 429 responses.
    """

    def __init__(
        self,
        coordinator: Optional[GlobalRateLimitCoordinator] = None,
        metrics: Optional[RateLimitMetrics] = None
    ):
        """
        Initialize rate limit hooks.

        Args:
            coordinator: GlobalRateLimitCoordinator instance (or uses singleton)
            metrics: RateLimitMetrics for recording events
        """
        self._coordinator = coordinator or GlobalRateLimitCoordinator()
        self._metrics = metrics

    def pre_download_hook(self, video_id: str, timeout: float = 30.0) -> bool:
        """
        Acquire a rate limit slot before downloading.

        Args:
            video_id: Video ID being downloaded (for logging)
            timeout: Max time to wait for slot

        Returns:
            True if slot acquired, False if timed out
        """
        acquired = self._coordinator.acquire_slot('download', timeout=timeout)
        if not acquired:
            logger.warning(f"Rate limit slot timeout for {video_id}")
            if self._metrics:
                self._metrics.record_slot_timeout()
        else:
            logger.debug(f"Acquired rate limit slot for {video_id}")
        return acquired

    def post_download_hook(self, video_id: str, success: bool) -> None:
        """
        Release rate limit slot after download.

        Args:
            video_id: Video ID that was downloaded
            success: Whether download succeeded
        """
        self._coordinator.release_slot('download')
        logger.debug(f"Released rate limit slot for {video_id} (success={success})")

    def on_error_hook(
        self,
        video_id: str,
        error_code: Optional[int],
        error_message: str,
        tier: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> None:
        """
        Record rate limit events on error.

        Args:
            video_id: Video ID that failed
            error_code: HTTP status code (e.g., 429, 403)
            error_message: Error message from download
            tier: Duration tier for per-tier tracking
            keyword: Search keyword for per-keyword tracking
        """
        if self._metrics is None:
            return

        # Record 429 (Too Many Requests) as rate limit events
        if error_code == 429:
            self._metrics.record_rate_limit_event(tier=tier, keyword=keyword)
            logger.info(f"Recorded 429 rate limit event for {video_id}")

        # Also check error message for rate limit indicators
        rate_limit_indicators = ['rate limit', 'too many requests', '429']
        if any(indicator in error_message.lower() for indicator in rate_limit_indicators):
            # Only record if not already recorded via error_code
            if error_code != 429:
                self._metrics.record_rate_limit_event(tier=tier, keyword=keyword)
                logger.info(f"Recorded rate limit event (from message) for {video_id}")


class DownloadOrchestrator:
    """
    Orchestrates batch downloading across multiple keywords.

    Coordinates:
    - Keyword iteration with progress tracking
    - Checkpoint management for resume capability
    - Retry queue processing for rate-limited downloads
    - Source diversity reporting

    The orchestrator delegates single-video downloads to the VideoDownloader,
    handling the batch-level coordination logic.
    """

    def __init__(self, downloader: 'VideoDownloader'):
        """
        Initialize orchestrator with a VideoDownloader instance.

        Args:
            downloader: The VideoDownloader to use for actual downloads
        """
        self.downloader = downloader

    def download_all(
        self,
        keywords: List[str],
        output_dir: Path,
        max_concurrent: int = None,
        resume: bool = False,
        topic: str = ""
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Download videos for all keywords.

        Args:
            keywords: List of search keywords
            output_dir: Output directory
            max_concurrent: Max concurrent downloads (uses config.download.parallel_workers if None)
            resume: Whether to resume from checkpoint
            topic: Topic context for LLM title filtering

        Returns:
            Tuple of (downloaded_videos, failed_keywords)
        """
        d = self.downloader

        # Use config value if not explicitly provided
        if max_concurrent is None:
            max_concurrent = getattr(d.download_config, 'parallel_workers', 4)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing downloads (file-based resume)
        existing_count = self._count_existing_videos(output_dir)
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing videos on disk (will skip)")

        # Load or create checkpoint
        keywords = self._setup_checkpoint(keywords, resume)

        all_downloaded = []
        failed_keywords = list(d.checkpoint.failed_keywords)

        # Log download configuration
        self._log_download_config(max_concurrent)

        # Process keywords sequentially
        total_videos_downloaded = 0
        print(f"\n  Downloading videos for {len(keywords)} keywords...")

        # Track which 10% milestones have been logged
        logged_milestones = set()

        for i, keyword in enumerate(keywords, 1):
            # Compact progress line
            progress_pct = (i - 1) / len(keywords) * 100
            print(f"\r  [{i}/{len(keywords)}] {progress_pct:5.1f}% | {keyword[:40]:<40} | Videos: {total_videos_downloaded}", end='', flush=True)

            # Log at 10% milestones
            self._log_milestone(i, len(keywords), logged_milestones)

            logger.info(f"[{i}/{len(keywords)}] Processing: {keyword}")

            d.checkpoint.current_keyword = keyword
            d._save_checkpoint()

            downloaded = d.download_for_keyword(keyword, output_dir, topic=topic)

            if downloaded:
                all_downloaded.extend(downloaded)
                total_videos_downloaded += len(downloaded)
                d.checkpoint.completed_keywords.append(keyword)
            else:
                failed_keywords.append(keyword)
                d.checkpoint.failed_keywords.append(keyword)

            d._save_checkpoint()

            # Delay between keywords to avoid rate limiting
            if i < len(keywords):
                time.sleep(d.download_config.delay_between_keywords)

        # Final progress line
        print(f"\r  [{len(keywords)}/{len(keywords)}] 100.0% | Done{' ' * 50}")
        print(f"  ✓ Downloaded {total_videos_downloaded} videos from {len(keywords)} keywords")

        # Log 100% milestone
        if 100 not in logged_milestones:
            logger.info(f"Download progress: 100% ({len(keywords)}/{len(keywords)} keywords)")

        # Process batch retry queue
        retry_downloaded, retry_failed = self._process_retry_queue(output_dir, topic)
        if retry_downloaded:
            all_downloaded.extend(retry_downloaded)
            total_videos_downloaded += len(retry_downloaded)
            print(f"  ✓ Batch retry recovered {len(retry_downloaded)} additional videos")

        # Final reporting
        d.log_source_diversity_report()
        self._log_cache_stats()
        self._log_budget_summary()

        # Clear checkpoint on success
        d._clear_checkpoint()

        return all_downloaded, failed_keywords

    def _count_existing_videos(self, output_dir: Path) -> int:
        """Count existing video files in output directory."""
        existing_count = 0
        if output_dir.exists():
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    videos = [f for f in os.listdir(subdir) if f.endswith(('.mp4', '.mkv', '.webm'))]
                    existing_count += len(videos)
        return existing_count

    def _setup_checkpoint(self, keywords: List[str], resume: bool) -> List[str]:
        """Setup or restore checkpoint, returning filtered keywords."""
        d = self.downloader

        if resume:
            d.checkpoint = d._load_checkpoint()
            if d.checkpoint:
                logger.info(f"Resuming from checkpoint ({len(d.checkpoint.completed_keywords)} keywords done)")
                # Filter out completed keywords
                keywords = [k for k in keywords if k not in d.checkpoint.completed_keywords]
                # Restore various states from checkpoint
                self._restore_checkpoint_state(d.checkpoint)

        if not d.checkpoint:
            d.checkpoint = DownloadCheckpoint(
                completed_keywords=[],
                completed_videos=[],
                failed_keywords=[],
                current_keyword=None,
                current_tier=None,
                timestamp=datetime.now().isoformat(),
                speed_tracker_state=None,
                last_rate_limit_timestamp=None,
                rate_limit_event_count=0
            )

        return keywords

    def _restore_checkpoint_state(self, checkpoint: DownloadCheckpoint) -> None:
        """Restore various manager states from checkpoint."""
        d = self.downloader

        # Restore speed tracker state
        if checkpoint.speed_tracker_state:
            d.speed_tracker.from_checkpoint_dict(checkpoint.speed_tracker_state)
            logger.debug(f"Restored speed tracker state: {d.speed_tracker.get_speed_stats()['samples']} samples")

        # Check rate limit cooldown
        d._in_cooldown_recovery_mode = d._check_rate_limit_cooldown(checkpoint)
        d._rate_limit_event_count = checkpoint.rate_limit_event_count

        # Restore rate limit metrics (US-006)
        if checkpoint.rate_limit_metrics:
            d.rate_limit_metrics = RateLimitMetrics.from_checkpoint(checkpoint.rate_limit_metrics)

        # Restore rate limit budget (US-004)
        if d._share_budget_across_keywords and checkpoint.rate_limit_budget:
            d.rate_limit_budget = RateLimitBudget.from_dict(checkpoint.rate_limit_budget)
            logger.debug(
                f"Restored rate limit budget: "
                f"rotations={d.rate_limit_budget.rotations_used}, "
                f"vpn_switches={d.rate_limit_budget.vpn_switches_used}, "
                f"backoff={d.rate_limit_budget.backoff_time_spent:.1f}s"
            )

        # Restore VPN manager state (US-005)
        if d.vpn_manager and checkpoint.vpn_manager_state:
            d.vpn_manager.restore_from_checkpoint(checkpoint.vpn_manager_state)

        # Restore escalation manager state (Sprint 10 US-007)
        if checkpoint.escalation_state and d.escalation_manager is not None:
            restored_mgr = EscalationManager.from_dict(
                data=checkpoint.escalation_state,
                impersonation_manager=d.impersonation_manager,
                extractor_args_config=getattr(d.download_config, 'extractor_args', None),
                budget=getattr(d, 'rate_limit_budget', None),
            )
            d.escalation_manager = restored_mgr
            # Re-share with auxiliary modules
            if hasattr(d, 'audio_first') and d.audio_first:
                d.audio_first.escalation_manager = d.escalation_manager
            if hasattr(d, 'title_filter') and d.title_filter:
                d.title_filter.escalation_manager = d.escalation_manager
            if hasattr(d, 'speech_screener') and d.speech_screener:
                d.speech_screener.escalation_manager = d.escalation_manager
            metrics = d.escalation_manager.get_metrics()
            logger.info(
                f"Restored escalation state: {metrics['total_403s']} 403s, "
                f"{metrics['total_escalations']} escalations"
            )

        # Restore per-tier backoff state (Sprint 12 US-003)
        if d._per_tier_isolation and checkpoint.tier_backoff_state:
            d._restore_tier_backoff_state(checkpoint.tier_backoff_state)

    def _log_download_config(self, max_concurrent: int) -> None:
        """Log download configuration settings."""
        d = self.downloader

        # Log title blacklist if enabled
        title_blacklist = getattr(d.download_config, 'title_blacklist', [])
        if title_blacklist:
            logger.info(f"  Title blacklist: {len(title_blacklist)} terms (e.g., {', '.join(title_blacklist[:5])}...)")

        # Log LLM filter status
        llm_config = getattr(d.download_config, 'llm_title_filter', None)
        if llm_config and getattr(llm_config, 'enabled', False):
            provider = getattr(llm_config, 'provider', 'gemini')
            logger.info(f"  LLM title filter: enabled ({provider})")

        # Log parallel workers setting
        logger.info(f"  Parallel workers: {max_concurrent}")

    def _log_milestone(self, current: int, total: int, logged_milestones: set) -> None:
        """Log progress at 10% milestones."""
        current_pct = (current - 1) / total * 100
        milestone = int(current_pct // 10) * 10
        if milestone > 0 and milestone not in logged_milestones:
            logger.info(f"Download progress: {milestone}% ({current-1}/{total} keywords)")
            logged_milestones.add(milestone)

    def _log_cache_stats(self) -> None:
        """Log cache hit/miss statistics."""
        d = self.downloader
        if hasattr(d, 'title_filter') and d.title_filter:
            try:
                cache_stats = d.title_filter.get_cache_stats()
                # Safely convert to int (handles MagicMock in tests)
                search_hits = int(cache_stats.get('search_hits', 0) or 0)
                search_misses = int(cache_stats.get('search_misses', 0) or 0)
                llm_hits = int(cache_stats.get('llm_hits', 0) or 0)
                llm_misses = int(cache_stats.get('llm_misses', 0) or 0)

                if search_hits > 0 or llm_hits > 0:
                    logger.info(
                        f"Cache stats: search={search_hits}/"
                        f"{search_hits + search_misses} hits, "
                        f"llm_filter={llm_hits}/"
                        f"{llm_hits + llm_misses} hits"
                    )
            except (TypeError, ValueError):
                # Handle MagicMock or invalid cache_stats gracefully
                pass

    def _log_budget_summary(self) -> None:
        """Log rate limit budget summary at download completion.

        US-61-010: Log budget usage summary showing rotations, VPN switches,
        backoff time, and keywords that triggered rate limit events.
        """
        d = self.downloader
        if hasattr(d, 'rate_limit_budget') and d.rate_limit_budget:
            budget = d.rate_limit_budget
            # Only log if any budget was consumed
            if (
                budget.rotations_used > 0 or
                budget.vpn_switches_used > 0 or
                budget.backoff_time_spent > 0
            ):
                logger.info(budget.budget_summary())

    def _process_retry_queue(
        self,
        output_dir: Path,
        topic: str = ""
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Process the batch retry queue after main download completes.

        Retries all rate-limited keyword/tier combinations with a delay
        between passes to allow rate limit windows to pass.

        Args:
            output_dir: Output directory for downloads
            topic: Topic context for LLM filtering

        Returns:
            Tuple of (recovered_videos, still_failed_keywords)
        """
        d = self.downloader

        if not d.retry_queue.is_enabled or not d.retry_queue.has_pending():
            return [], []

        # Pass budget state to retry queue so it knows remaining budget
        if d._share_budget_across_keywords:
            d.retry_queue.set_budget_state(d.rate_limit_budget.get_summary())

        recovered = []
        still_failed = []

        # Process retry passes
        while d.retry_queue.has_pending():
            # Start retry pass (applies delay)
            pass_num = d.retry_queue.start_retry_pass()
            if pass_num == 0:
                break

            # Get items to retry
            items = d.retry_queue.get_pending_items()
            logger.info(f"Batch retry pass {pass_num}: attempting {len(items)} keyword/tier combinations")

            for item in items:
                keyword = item.keyword
                tier = item.tier

                # Reset rate limit flag before retry
                d._last_download_rate_limited = False

                # Try downloading again
                downloaded = d._download_single(keyword, tier, output_dir, topic)

                if downloaded:
                    # Success - mark in queue and add to recovered
                    d.retry_queue.mark_success(item.video_id)
                    recovered.extend(downloaded)

                    # Update tier counts and sources
                    with d._lock:
                        d.tier_download_counts[tier] = d.tier_download_counts.get(tier, 0) + len(downloaded)
                        d.sources.extend(downloaded)
                        d._save_sources()

                    logger.info(f"  Batch retry: recovered {len(downloaded)} video(s) for '{keyword}' ({tier})")
                else:
                    # Still failing
                    d.retry_queue.mark_failed(item.video_id)

            # Finish this pass (moves exhausted items to permanently failed)
            d.retry_queue.finish_retry_pass()

        # Collect still-failed keywords
        stats = d.retry_queue.get_stats()
        if stats['failed'] > 0:
            logger.warning(
                f"Batch retry: {stats['failed']} keyword/tier combinations "
                f"still failed after {stats['max_passes']} retry passes"
            )

        return recovered, still_failed

    def coordinate_retries(
        self,
        failed_items: List[Dict],
        output_dir: Path,
        topic: str = ""
    ) -> Tuple[List[DownloadedVideo], List[Dict]]:
        """
        Coordinate retry attempts for a list of failed download items.

        This is a simplified interface for retry coordination that can be
        used independently of the full download_all flow.

        Args:
            failed_items: List of dicts with 'keyword', 'tier', 'video_id' keys
            output_dir: Output directory
            topic: Topic context for LLM filtering

        Returns:
            Tuple of (recovered_videos, still_failed_items)
        """
        d = self.downloader
        recovered = []
        still_failed = []

        for item in failed_items:
            keyword = item.get('keyword', '')
            tier = item.get('tier', 'medium')
            video_id = item.get('video_id', f"{keyword}|{tier}")

            # Reset rate limit flag
            d._last_download_rate_limited = False

            # Attempt download
            downloaded = d._download_single(keyword, tier, output_dir, topic)

            if downloaded:
                recovered.extend(downloaded)
                # Update tier counts
                with d._lock:
                    d.tier_download_counts[tier] = d.tier_download_counts.get(tier, 0) + len(downloaded)
                    d.sources.extend(downloaded)
                    d._save_sources()
            else:
                still_failed.append(item)

        return recovered, still_failed

    def aggregate_results(
        self,
        results: List[Tuple[List[DownloadedVideo], List[str]]]
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Aggregate results from multiple download batches.

        Args:
            results: List of (downloaded, failed) tuples

        Returns:
            Combined (all_downloaded, all_failed) tuple
        """
        all_downloaded = []
        all_failed = []

        for downloaded, failed in results:
            all_downloaded.extend(downloaded)
            all_failed.extend(failed)

        return all_downloaded, all_failed

    def get_batch_stats(self) -> Dict:
        """
        Get statistics for the current/last batch operation.

        Returns:
            Dict with batch statistics
        """
        d = self.downloader

        stats = {
            'total_sources': len(d.sources),
            'tier_counts': dict(d.tier_download_counts),
            'retry_queue_stats': d.retry_queue.get_stats() if hasattr(d, 'retry_queue') else {},
        }

        if d.checkpoint:
            stats['checkpoint'] = {
                'completed_keywords': len(d.checkpoint.completed_keywords),
                'failed_keywords': len(d.checkpoint.failed_keywords),
                'completed_videos': len(d.checkpoint.completed_videos),
            }

        return stats
