"""
Download orchestrator - batch coordination logic extracted from core.py.

Handles:
- Batch processing across multiple keywords (DownloadOrchestrator)
- Segment-level download abstraction (SegmentDownloadOrchestrator, US-82-007)
- Retry queue coordination
- Progress tracking and checkpointing
- Source diversity reporting

The VideoDownloader in core.py focuses on single-video download mechanics,
while these orchestrators handle higher-level coordination logic.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, TYPE_CHECKING, Callable

from ..state import DownloadedVideo
from ..rate_limit.coordinator import GlobalRateLimitCoordinator
from ..logging_templates import log_rate_limit, log_progress
from .checkpoint import DownloadCheckpoint
from .rate_limit_metrics import RateLimitMetrics
from .rate_limit_budget import RateLimitBudget
from .escalation_manager import EscalationManager
from .pause_calculator import RegionSuccessTracker

# US-129-003: ETA calculation imports
# US-143-007: Enhanced ETA with congestion factor and confidence intervals
from ..stages.download_segments import (
    calculate_eta_seconds,
    calculate_bandwidth_utilization,
    format_eta_display,
    format_progress_line,
    calculate_network_congestion_factor,
    calculate_eta_confidence_interval,
    format_eta_confidence_display,
    ETAHistory,
)

if TYPE_CHECKING:
    from .core import VideoDownloader
    from ..config import Config

logger = logging.getLogger(__name__)


class CancellationToken:
    """Token for signaling cancellation of long-running operations.

    US-129-009: Provides a thread-safe way to signal and check for cancellation
    during download operations, enabling graceful shutdown with cleanup.
    """

    def __init__(self):
        """Initialize cancellation token with not-cancelled state."""
        self._cancelled = False
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signal cancellation."""
        with self._lock:
            self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        with self._lock:
            return self._cancelled

    def reset(self) -> None:
        """Reset cancellation state (for testing or reuse)."""
        with self._lock:
            self._cancelled = False


class TieredSlotManager:
    """Manages per-tier concurrent download slots (US-114-011).

    Provides tier-aware slot management where each duration tier
    (short, medium, long, longer) has its own concurrent limit,
    with optional slot borrowing between tiers.
    """

    def __init__(
        self,
        max_concurrent_per_tier: Dict[str, int] = None,
        allow_borrowing: bool = True,
        max_total_concurrent: int = None
    ):
        """
        Initialize tiered slot manager.

        Args:
            max_concurrent_per_tier: Max concurrent downloads per tier
            allow_borrowing: Whether tiers can borrow slots from other tiers
            max_total_concurrent: Safety cap on total concurrent downloads
        """
        # Default per-tier limits
        self._max_per_tier = max_concurrent_per_tier or {
            'short': 2,
            'medium': 1,
            'long': 1,
            'longer': 1,
        }
        self._allow_borrowing = allow_borrowing
        self._max_total = max_total_concurrent

        # Track active downloads per tier
        self._active_per_tier: Dict[str, int] = {tier: 0 for tier in self._max_per_tier}
        self._active_downloads: Dict[str, Any] = {}  # download_id -> {tier, started}

        # Create semaphores per tier
        self._tier_semaphores: Dict[str, threading.Semaphore] = {
            tier: threading.Semaphore(slots)
            for tier, slots in self._max_per_tier.items()
        }

        # Global semaphore for total concurrency cap
        total_slots = max_total_concurrent or sum(self._max_per_tier.values())
        self._global_semaphore = threading.Semaphore(total_slots)

        self._lock = threading.Lock()

    @property
    def max_per_tier(self) -> Dict[str, int]:
        """Return max concurrent per tier."""
        return self._max_per_tier.copy()

    def get_active_count(self, tier: str = None) -> int:
        """Get active count for a tier or total."""
        with self._lock:
            if tier:
                return self._active_per_tier.get(tier, 0)
            return sum(self._active_per_tier.values())

    def get_available_slots(self, tier: str = None) -> int:
        """Get available slots for a tier or total."""
        with self._lock:
            if tier:
                return self._max_per_tier.get(tier, 0) - self._active_per_tier.get(tier, 0)
            total_active = sum(self._active_per_tier.values())
            total_max = self._max_total or sum(self._max_per_tier.values())
            return total_max - total_active

    def acquire(self, tier: str, download_id: str = None, timeout: float = None) -> bool:
        """
        Acquire a slot for the specified tier.

        Args:
            tier: Duration tier (short, medium, long, longer)
            download_id: Optional identifier for tracking
            timeout: Max time to wait for slot (None = blocking)

        Returns:
            True if slot acquired, False if timeout
        """
        # Normalize tier name
        tier = tier.lower() if tier else 'short'
        if tier not in self._max_per_tier:
            logger.warning(f"Unknown tier '{tier}', defaulting to 'short'")
            tier = 'short'

        # First, acquire global slot (for total cap)
        acquired_global = self._global_semaphore.acquire(timeout=timeout if timeout else 0)
        if not acquired_global:
            logger.debug(f"TieredSlotManager: global slot timeout for tier={tier}")
            return False

        # Try to acquire tier-specific slot
        tier_sem = self._tier_semaphores[tier]
        acquired_tier = tier_sem.acquire(timeout=timeout if timeout else 0)

        if not acquired_tier:
            # Tier slot not available, try borrowing if enabled
            if self._allow_borrowing:
                borrowed = self._try_borrow_slot(tier, timeout)
                if borrowed:
                    with self._lock:
                        self._active_per_tier[tier] += 1
                        if download_id:
                            self._active_downloads[download_id] = {
                                'started': time.time(),
                                'id': download_id,
                                'tier': tier,
                                'borrowed': True
                            }
                    logger.debug(
                        f"TieredSlotManager: acquired borrowed slot for tier={tier} "
                        f"({self._active_per_tier[tier]}/{self._max_per_tier[tier]} + borrowed)"
                    )
                    return True

            # Can't acquire or borrow - release global and return failure
            self._global_semaphore.release()
            logger.debug(f"TieredSlotManager: tier slot unavailable for tier={tier}")
            return False

        # Successfully acquired tier slot
        with self._lock:
            self._active_per_tier[tier] += 1
            if download_id:
                self._active_downloads[download_id] = {
                    'started': time.time(),
                    'id': download_id,
                    'tier': tier,
                    'borrowed': False
                }

        logger.debug(
            f"TieredSlotManager: acquired slot for tier={tier} "
            f"({self._active_per_tier[tier]}/{self._max_per_tier[tier]} active)"
        )
        return True

    def _try_borrow_slot(self, requesting_tier: str, timeout: float = None) -> bool:
        """Try to borrow a slot from another tier.

        Args:
            requesting_tier: Tier that needs a slot
            timeout: Max time to wait

        Returns:
            True if borrowed successfully
        """
        with self._lock:
            # Find tiers with available slots
            available_tiers = [
                tier for tier, count in self._active_per_tier.items()
                if tier != requesting_tier and count < self._max_per_tier[tier]
            ]

        if not available_tiers:
            return False

        # Try to acquire from available tiers (prefer same or higher tier)
        # Sort by tier priority: longer > long > medium > short (lower priority gets borrowed first)
        tier_priority = {'longer': 0, 'long': 1, 'medium': 2, 'short': 3}
        available_tiers.sort(key=lambda t: tier_priority.get(t, 99))

        for source_tier in available_tiers:
            if self._tier_semaphores[source_tier].acquire(timeout=0):
                logger.debug(f"TieredSlotManager: borrowed slot from {source_tier} for {requesting_tier}")
                return True

        return False

    def release(self, tier: str, download_id: str = None, success: bool = True) -> None:
        """
        Release a slot.

        Args:
            tier: Duration tier (short, medium, long, longer)
            download_id: Optional identifier that was used for tracking
            success: Whether the download succeeded
        """
        tier = tier.lower() if tier else 'short'
        if tier not in self._max_per_tier:
            tier = 'short'

        with self._lock:
            was_borrowed = False
            if download_id and download_id in self._active_downloads:
                info = self._active_downloads.pop(download_id)
                was_borrowed = info.get('borrowed', False)

            if self._active_per_tier.get(tier, 0) > 0:
                self._active_per_tier[tier] -= 1

        # Release tier semaphore
        if tier in self._tier_semaphores:
            self._tier_semaphores[tier].release()

        # Always release global semaphore
        self._global_semaphore.release()

        logger.debug(
            f"TieredSlotManager: released slot for tier={tier} "
            f"({self._active_per_tier.get(tier, 0)}/{self._max_per_tier[tier]} active, "
            f"borrowed={was_borrowed})"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get status of all tiers."""
        with self._lock:
            return {
                'active_per_tier': self._active_per_tier.copy(),
                'max_per_tier': self._max_per_tier.copy(),
                'total_active': sum(self._active_per_tier.values()),
                'total_max': self._max_total or sum(self._max_per_tier.values()),
                'allow_borrowing': self._allow_borrowing,
                'active_downloads': list(self._active_downloads.keys()),
            }

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._active_per_tier = {tier: 0 for tier in self._max_per_tier}
            self._active_downloads.clear()

        # Recreate semaphores
        self._tier_semaphores = {
            tier: threading.Semaphore(slots)
            for tier, slots in self._max_per_tier.items()
        }
        total_slots = self._max_total or sum(self._max_per_tier.values())
        self._global_semaphore = threading.Semaphore(total_slots)


class DownloadCoordinator:
    """
    Global download coordinator to manage concurrent downloads.

    Tracks active downloads across parallel workers and enforces
    max concurrent downloads limit to prevent resource exhaustion.

    This enables true parallel downloading while respecting resource constraints.

    Supports per-tier slot management (US-114-011) via TieredSlotManager.
    """

    def __init__(
        self,
        max_concurrent: int = None,
        tier_config: Dict[str, int] = None,
        enable_tier_slots: bool = False,
        allow_borrowing: bool = True,
        max_total_concurrent: int = None
    ):
        """
        Initialize download coordinator.

        Args:
            max_concurrent: Maximum concurrent downloads (used if tier slots disabled)
            tier_config: Max concurrent per tier (used if tier slots enabled)
            enable_tier_slots: Enable per-tier slot management
            allow_borrowing: Allow borrowing between tiers
            max_total_concurrent: Safety cap on total concurrent downloads
        """
        self._max_concurrent = max_concurrent or 4
        self._enable_tier_slots = enable_tier_slots
        self._tier_config = dict(tier_config or {
            'short': 2,
            'medium': 1,
            'long': 1,
            'longer': 1,
        })
        self._allow_borrowing = allow_borrowing
        self._max_total_concurrent = max_total_concurrent

        if enable_tier_slots:
            # Use tiered slot manager
            self._tier_manager = TieredSlotManager(
                max_concurrent_per_tier=self._tier_config,
                allow_borrowing=self._allow_borrowing,
                max_total_concurrent=self._max_total_concurrent
            )
            self._semaphore = None
        else:
            # Use simple global semaphore
            self._semaphore = threading.Semaphore(self._max_concurrent)
            self._tier_manager = None

        self._active_count = 0
        self._lock = threading.Lock()
        self._active_downloads: Dict[str, Any] = {}
        self._completed_count = 0
        self._failed_count = 0

    @property
    def max_concurrent(self) -> int:
        """Return max concurrent downloads setting."""
        return self._max_concurrent

    @property
    def active_count(self) -> int:
        """Return number of currently active downloads."""
        with self._lock:
            return self._active_count

    @property
    def completed_count(self) -> int:
        """Return total completed downloads."""
        with self._lock:
            return self._completed_count

    @property
    def failed_count(self) -> int:
        """Return total failed downloads."""
        with self._lock:
            return self._failed_count

    @property
    def is_tier_slots_enabled(self) -> bool:
        """Return whether tier slot management is enabled."""
        return self._enable_tier_slots

    def acquire(self, download_id: str = None, tier: str = None, timeout: float = None) -> bool:
        """
        Acquire a download slot. Blocks if at max capacity.

        Args:
            download_id: Optional identifier for tracking
            tier: Duration tier for per-tier management (short, medium, long, longer)
            timeout: Max time to wait for slot

        Returns:
            True when slot acquired
        """
        if self._enable_tier_slots and self._tier_manager:
            # Use tiered slot manager
            acquired = self._tier_manager.acquire(tier or 'short', download_id, timeout)
            if acquired:
                with self._lock:
                    self._active_count += 1
            return acquired
        else:
            # Use simple semaphore
            if timeout is None:
                acquired = self._semaphore.acquire()
            else:
                acquired = self._semaphore.acquire(timeout=timeout)
            if not acquired:
                logger.debug("DownloadCoordinator: slot acquisition timed out")
                return False
            with self._lock:
                self._active_count += 1
                if download_id:
                    self._active_downloads[download_id] = {
                        'started': time.time(),
                        'id': download_id,
                        'tier': tier
                    }
            logger.debug(f"DownloadCoordinator: acquired slot ({self._active_count}/{self._max_concurrent} active)")
            return True

    def release(self, download_id: str = None, success: bool = True, tier: str = None) -> None:
        """
        Release a download slot.

        Args:
            download_id: Optional identifier that was used for tracking
            success: Whether the download succeeded
            tier: Duration tier for per-tier management
        """
        if self._enable_tier_slots and self._tier_manager:
            # Use tiered slot manager
            self._tier_manager.release(tier or 'short', download_id, success)
            with self._lock:
                self._active_count -= 1
                if download_id and download_id in self._active_downloads:
                    del self._active_downloads[download_id]
                if success:
                    self._completed_count += 1
                else:
                    self._failed_count += 1
        else:
            # Use simple semaphore
            with self._lock:
                if download_id and download_id in self._active_downloads:
                    del self._active_downloads[download_id]
                self._active_count -= 1
                if success:
                    self._completed_count += 1
                else:
                    self._failed_count += 1

            self._semaphore.release()
            logger.debug(f"DownloadCoordinator: released slot ({self._active_count}/{self._max_concurrent} active)")

    def get_status(self) -> Dict[str, Any]:
        """
        Get coordinator status.

        Returns:
            Dict with active, max_concurrent, completed, failed, and active_downloads info
        """
        with self._lock:
            status = {
                'active': self._active_count,
                'max_concurrent': self._max_concurrent,
                'completed': self._completed_count,
                'failed': self._failed_count,
                'active_downloads': list(self._active_downloads.keys()),
                'available_slots': self._max_concurrent - self._active_count,
                'tier_slots_enabled': self._enable_tier_slots,
            }

            if self._enable_tier_slots and self._tier_manager:
                status['tier_status'] = self._tier_manager.get_status()

            return status

    def reset(self) -> None:
        """Reset all counters (for testing or new batch)."""
        with self._lock:
            self._active_count = 0
            self._completed_count = 0
            self._failed_count = 0
            self._active_downloads.clear()

        if self._enable_tier_slots and self._tier_manager:
            self._tier_manager.reset()
        else:
            # Recreate semaphore to clear any pending acquires
            self._semaphore = threading.Semaphore(self._max_concurrent)

    # =========================================================================
    # SELF-REGULATION THROTTLING (US-123-010)
    # =========================================================================

    def set_concurrency(self, new_concurrency: int) -> None:
        """Set new concurrency limit, recreating semaphore if needed.

        Args:
            new_concurrency: New max concurrent downloads
        """
        if new_concurrency < 1:
            logger.warning(f"Cannot set concurrency < 1, using 1")
            new_concurrency = 1

        old_concurrency = self._max_concurrent
        self._max_concurrent = new_concurrency

        if self._enable_tier_slots and self._tier_manager:
            self._max_total_concurrent = new_concurrency
            self._tier_manager = TieredSlotManager(
                max_concurrent_per_tier=self._tier_config,
                allow_borrowing=self._allow_borrowing,
                max_total_concurrent=self._max_total_concurrent
            )
        else:
            # Recreate semaphore with new limit
            self._semaphore = threading.Semaphore(self._max_concurrent)

        logger.info(f"Concurrency adjusted: {old_concurrency} → {new_concurrency}")

    def get_current_concurrency(self) -> int:
        """Get current concurrency setting.

        Returns:
            Current max concurrent downloads
        """
        return self._max_concurrent


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
            log_rate_limit(logger, "slot_acquisition", "download", "timeout", video_id=video_id)
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
            log_rate_limit(logger, "http_response", "youtube_api", "rate_limit_detected", video_id=video_id)

        # Also check error message for rate limit indicators
        rate_limit_indicators = ['rate limit', 'too many requests', '429']
        if any(indicator in error_message.lower() for indicator in rate_limit_indicators):
            # Only record if not already recorded via error_code
            if error_code != 429:
                self._metrics.record_rate_limit_event(tier=tier, keyword=keyword)
                log_rate_limit(logger, "error_message", "youtube_api", "rate_limit_detected", video_id=video_id)


class DownloadOrchestrator:
    """
    Orchestrates batch downloading across multiple keywords.

    Coordinates:
    - Keyword iteration with progress tracking
    - Checkpoint management for resume capability
    - Retry queue processing for rate-limited downloads
    - Source diversity reporting
    - Cancellation support with graceful cleanup (US-129-009)

    The orchestrator delegates single-video downloads to the VideoDownloader,
    handling the batch-level coordination logic.
    """

    def __init__(self, downloader: 'VideoDownloader', cancellation_token: Optional[CancellationToken] = None):
        """
        Initialize orchestrator with a VideoDownloader instance.

        Args:
            downloader: The VideoDownloader to use for actual downloads
            cancellation_token: Optional token for signaling cancellation
        """
        self.downloader = downloader
        self.cancellation_token = cancellation_token or CancellationToken()
        self._coordinator: Optional[DownloadCoordinator] = None
        # US-129-003: Progress tracking state
        self._progress_update_interval: float = 5.0  # seconds
        self._progress_start_time: Optional[float] = None
        self._progress_bytes_downloaded: int = 0
        self._progress_bytes_total: int = 0
        self._progress_videos_completed: int = 0
        self._progress_videos_total: int = 0
        self._progress_last_update: float = 0
        self._progress_current_video: str = ""
        self._progress_download_speeds: List[float] = []  # Track recent download speeds
        # US-129-009: Track partial files for cleanup on cancellation
        self._partial_files: List[Path] = []
        self._output_dir: Optional[Path] = None

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
            max_concurrent = getattr(d.download_config, 'max_concurrent', None)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            max_concurrent = getattr(d.download_config, 'parallel_workers', 4)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            max_concurrent = 4

        tier_slot_config = getattr(d.download_config, 'tier_slot_management', None)
        enable_tier_slots = False
        tier_config = None
        allow_borrowing = True
        max_total_concurrent = None
        if isinstance(tier_slot_config, dict):
            enable_tier_slots = tier_slot_config.get('enabled') is True
            if enable_tier_slots:
                tier_config = tier_slot_config.get('max_concurrent_per_tier')
                allow_borrowing = tier_slot_config.get('allow_borrowing', True)
                max_total_concurrent = tier_slot_config.get('max_total_concurrent')
        elif tier_slot_config is not None:
            enable_tier_slots = getattr(tier_slot_config, 'enabled', False) is True
            if enable_tier_slots:
                tier_config = getattr(tier_slot_config, 'max_concurrent_per_tier', None)
                allow_borrowing = getattr(tier_slot_config, 'allow_borrowing', True)
                max_total_concurrent = getattr(tier_slot_config, 'max_total_concurrent', None)

        # Initialize download coordinator for parallel download management
        self._coordinator = DownloadCoordinator(
            max_concurrent=max_concurrent,
            tier_config=tier_config,
            enable_tier_slots=enable_tier_slots,
            allow_borrowing=allow_borrowing,
            max_total_concurrent=max_total_concurrent,
        )
        d.download_coordinator = self._coordinator
        logger.info(
            "DownloadCoordinator initialized: "
            f"max_concurrent={max_concurrent}, "
            f"tier_slots_enabled={enable_tier_slots}, "
            f"max_total_concurrent={max_total_concurrent}"
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir  # Store for cleanup on cancellation

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

        # US-129-003: Initialize progress tracking
        self._init_progress_tracking(keywords)

        # Process keywords sequentially
        total_videos_downloaded = 0
        logger.info(f"Starting video download for {len(keywords)} keywords")

        # Track which 10% milestones have been logged
        logged_milestones = set()

        for i, keyword in enumerate(keywords, 1):
            # US-129-003: Track current video for progress display
            self._progress_current_video = keyword

            # Compact progress line with ETA
            self._update_progress_display(i, len(keywords), total_videos_downloaded, existing_count)

            # Log at 10% milestones
            self._log_milestone(i, len(keywords), logged_milestones)

            logger.info(f"[{i}/{len(keywords)}] Processing: {keyword}")

            d.checkpoint.current_keyword = keyword
            d._save_checkpoint()

            # Track download start time for speed calculation
            keyword_start_time = time.time()

            downloaded = d.download_for_keyword(keyword, output_dir, topic=topic)

            # US-129-003: Calculate bytes downloaded and update progress
            keyword_bytes = 0
            if downloaded:
                for vid in downloaded:
                    if vid.file_path:
                        try:
                            keyword_bytes += Path(vid.file_path).stat().st_size
                        except OSError:
                            pass

            self._update_progress_stats(
                bytes_downloaded=keyword_bytes,
                videos_completed=len(downloaded) if downloaded else 0,
                elapsed_seconds=time.time() - keyword_start_time
            )

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

            # US-129-009: Check for cancellation
            if self.cancellation_token.is_cancelled:
                logger.info(f"Download cancelled at keyword {i}/{len(keywords)}")
                return self._handle_cancellation(all_downloaded, failed_keywords)

        # Final progress line
        logger.info(f"Download progress: 100% ({len(keywords)}/{len(keywords)} keywords) - Complete")
        logger.info(f"Downloaded {total_videos_downloaded} videos from {len(keywords)} keywords")

        # Log 100% milestone
        if 100 not in logged_milestones:
            logger.info(f"Download progress: 100% ({len(keywords)}/{len(keywords)} keywords)")

        # Process batch retry queue
        retry_downloaded, retry_failed = self._process_retry_queue(output_dir, topic)
        if retry_downloaded:
            all_downloaded.extend(retry_downloaded)
            total_videos_downloaded += len(retry_downloaded)
            logger.info(f"Batch retry recovered {len(retry_downloaded)} additional videos")

        # Final reporting
        d.log_source_diversity_report()
        self._log_cache_stats()
        self._log_budget_summary()

        # Log coordinator status
        if self._coordinator:
            status = self._coordinator.get_status()
            logger.info(f"DownloadCoordinator final status: {status}")

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

    # =========================================================================
    # US-129-009: Cancellation handling
    # =========================================================================

    def _handle_cancellation(
        self,
        all_downloaded: List[DownloadedVideo],
        failed_keywords: List[str]
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """Handle download cancellation with graceful cleanup.

        Args:
            all_downloaded: Videos already downloaded before cancellation
            failed_keywords: Keywords that failed before cancellation

        Returns:
            Tuple of (downloaded_videos, failed_keywords) for resume
        """
        d = self.downloader
        cleaned_files = 0

        # Clean up partial download files
        if self._output_dir and self._output_dir.exists():
            cleaned_files = self._cleanup_partial_files(self._output_dir)

        # Log cancellation event with cleanup summary
        logger.info(
            f"Download cancelled: {len(all_downloaded)} videos downloaded, "
            f"{len(failed_keywords)} keywords failed, {cleaned_files} partial files cleaned up"
        )

        # Save checkpoint for resume capability
        if d.checkpoint:
            d._save_checkpoint()
            logger.info(f"Checkpoint saved for resume: {len(d.checkpoint.completed_keywords)} keywords completed")

        return all_downloaded, failed_keywords

    def _cleanup_partial_files(self, output_dir: Path) -> int:
        """Clean up partial download files in the output directory.

        Partial files are identified by yt-dlp's .part extension or
        files that are being actively written to.

        Args:
            output_dir: Directory to clean

        Returns:
            Number of partial files cleaned up
        """
        cleaned_count = 0

        # Patterns for partial files
        partial_patterns = ['.part', '.ytdl', '.tmp', '.download']

        try:
            for item in output_dir.rglob('*'):
                if item.is_file():
                    # Check if file matches partial patterns
                    name_lower = item.name.lower()
                    if any(pattern in name_lower for pattern in partial_patterns):
                        try:
                            item.unlink()
                            cleaned_count += 1
                            logger.debug(f"Removed partial file: {item}")
                        except OSError as e:
                            logger.warning(f"Failed to remove partial file {item}: {e}")
        except OSError as e:
            logger.warning(f"Error scanning for partial files: {e}")

        return cleaned_count

    def cancel(self) -> None:
        """Request cancellation of ongoing download operations.

        This method signals the cancellation token and triggers cleanup.
        The actual cleanup happens when the download loop checks the token.
        """
        if not self.cancellation_token.is_cancelled:
            self.cancellation_token.cancel()
            logger.info("Cancellation requested for download orchestrator")

            # If we're not in the middle of a download_all call,
            # perform immediate cleanup
            if hasattr(self, '_output_dir') and self._output_dir:
                cleaned = self._cleanup_partial_files(self._output_dir)
                logger.info(f"Immediate cleanup: {cleaned} partial files removed")

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

        # Restore MullvadVPN state for rotation limit persistence (US-129-007)
        if d.mullvad_vpn and checkpoint.mullvad_vpn_state:
            d.mullvad_vpn.restore_from_checkpoint(checkpoint.mullvad_vpn_state)

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

        # Restore impersonation manager state from checkpoint
        impersonation_state = getattr(checkpoint, 'impersonation_state', None)
        if impersonation_state and d.impersonation_manager is not None:
            d.impersonation_manager.restore_state(impersonation_state)

        # Restore per-tier backoff state (Sprint 12 US-003)
        if d._per_tier_isolation and checkpoint.tier_backoff_state:
            d._restore_tier_backoff_state(checkpoint.tier_backoff_state)

        # US-129-002: Restore retry budget state for per-video retry limits
        if hasattr(d, '_retry_budget') and d._retry_budget and checkpoint.retry_budget_state:
            d._retry_budget.from_checkpoint_dict(checkpoint.retry_budget_state)
            logger.debug(
                f"Restored retry budget state: {len(checkpoint.retry_budget_state.get('video_states', {}))} videos tracked"
            )

        # US-136-007: Restore region success tracking state for dynamic region backoff
        if checkpoint.region_success_state:
            RegionSuccessTracker.restore_from_checkpoint(checkpoint.region_success_state)
            logger.debug(
                f"Restored region success tracking state: "
                f"{checkpoint.region_success_state.get('region_attempts', {})}"
            )

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

    # =========================================================================
    # US-129-003: Progress Tracking with ETA
    # =========================================================================

    def _init_progress_tracking(self, keywords: List[str]) -> None:
        """Initialize progress tracking state for batch download.

        Args:
            keywords: List of keywords to process
        """
        self._progress_start_time = time.time()
        self._progress_bytes_downloaded = 0
        self._progress_bytes_total = 0  # Will be estimated as we go
        self._progress_videos_completed = 0
        self._progress_videos_total = 0
        self._progress_last_update = 0
        self._progress_current_video = ""
        self._progress_download_speeds = []

        # Estimate total based on keywords count (rough estimate: 5 videos per keyword)
        self._progress_videos_total = len(keywords) * 5

        logger.debug(
            f"Progress tracking initialized: {len(keywords)} keywords, "
            f"estimated {self._progress_videos_total} videos"
        )

    def _update_progress_stats(
        self,
        bytes_downloaded: int,
        videos_completed: int,
        elapsed_seconds: float
    ) -> None:
        """Update progress tracking statistics after each keyword download.

        Args:
            bytes_downloaded: Bytes downloaded in this keyword
            videos_completed: Number of videos completed
            elapsed_seconds: Time taken for this keyword
        """
        self._progress_bytes_downloaded += bytes_downloaded
        self._progress_videos_completed += videos_completed

        # Track download speed (bytes per second)
        if elapsed_seconds > 0 and bytes_downloaded > 0:
            speed_bps = bytes_downloaded / elapsed_seconds
            self._progress_download_speeds.append(speed_bps)
            # Keep only last 10 speed samples for moving average
            if len(self._progress_download_speeds) > 10:
                self._progress_download_speeds.pop(0)

    def _update_progress_display(
        self,
        current_index: int,
        total_keywords: int,
        videos_completed: int,
        existing_count: int = 0
    ) -> None:
        """Update the progress display with ETA calculation.

        Args:
            current_index: Current keyword index (1-based)
            total_keywords: Total number of keywords
            videos_completed: Number of videos downloaded so far
            existing_count: Number of existing videos already downloaded
        """
        current_time = time.time()

        # Check if we should update (based on interval)
        if (current_time - self._progress_last_update) < self._progress_update_interval:
            # Still update the basic progress line for responsiveness
            progress_pct = (current_index - 1) / total_keywords * 100
            # Quick progress update - log at debug level to avoid spam
            logger.debug(f"Download progress: {progress_pct:.0f}% ({current_index}/{total_keywords}) - {self._progress_current_video[:40]}")
            return

        self._progress_last_update = current_time

        # Calculate elapsed time and speed
        elapsed_total = current_time - self._progress_start_time if self._progress_start_time else 0

        # Calculate average speed from tracked speeds
        avg_speed_bps = 0.0
        if self._progress_download_speeds:
            avg_speed_bps = sum(self._progress_download_speeds) / len(self._progress_download_speeds)

        # Estimate remaining videos
        avg_videos_per_keyword = 5  # Assume 5 videos per keyword
        remaining_keywords = total_keywords - current_index + 1
        estimated_remaining_videos = remaining_keywords * avg_videos_per_keyword

        # Estimate total bytes based on average bytes per video
        estimated_total_bytes = self._progress_bytes_downloaded
        if videos_completed > 0:
            avg_bytes_per_video = self._progress_bytes_downloaded / videos_completed
            estimated_total_bytes = int(avg_bytes_per_video * (videos_completed + estimated_remaining_videos))

        # Calculate ETA with congestion factor
        remaining_bytes = max(0, estimated_total_bytes - self._progress_bytes_downloaded)

        # Get recent speeds for congestion factor calculation
        recent_speeds = self._progress_download_speeds[-10:] if len(self._progress_download_speeds) > 10 else self._progress_download_speeds

        # Calculate congestion factor based on speed variance
        congestion_factor = calculate_network_congestion_factor(recent_speeds) if recent_speeds else 1.0

        # Calculate base ETA
        base_eta = calculate_eta_seconds(
            self._progress_bytes_downloaded,
            estimated_total_bytes,
            avg_speed_bps if avg_speed_bps > 0 else None
        )

        # Apply congestion factor to ETA (add padding for network instability)
        if base_eta is not None and base_eta > 0:
            eta_seconds = base_eta * congestion_factor
        else:
            eta_seconds = None

        # Calculate confidence interval for ETA
        lower_bound, upper_bound = calculate_eta_confidence_interval(eta_seconds, recent_speeds)

        # Calculate bandwidth utilization
        bandwidth_util = calculate_bandwidth_utilization(avg_speed_bps if avg_speed_bps > 0 else None)

        # Format the progress line
        progress_pct = (current_index - 1) / total_keywords * 100
        downloaded_mb = self._progress_bytes_downloaded / (1024 * 1024)
        total_mb = estimated_total_bytes / (1024 * 1024)
        speed_kbps = (avg_speed_bps / 1024) if avg_speed_bps > 0 else 0

        # Format ETA with confidence interval
        eta_str = format_eta_confidence_display(eta_seconds, lower_bound, upper_bound)

        # Log full progress with ETA and bandwidth using structured logging
        log_progress(
            logger,
            stage_name="DOWNLOAD",
            progress_pct=progress_pct,
            current=current_index,
            total=total_keywords,
            keyword=self._progress_current_video[:30],
            downloaded_mb=f"{downloaded_mb:.1f}",
            estimated_total_mb=f"{total_mb:.1f}",
            speed_kbps=int(speed_kbps),
            eta=eta_str,
            bandwidth_util=f"{bandwidth_util:.0f}%",
            videos=videos_completed + existing_count
        )

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

            # Get items to retry (filter out budget-exhausted videos)
            items = d.retry_queue.get_items_with_budget()
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

                    # US-129-002: Reset retry budget on successful download
                    if d._retry_budget:
                        d._retry_budget.reset(item.video_id)

                    recovered.extend(downloaded)

                    # Update tier counts and sources
                    with d._lock:
                        d.tier_download_counts[tier] = d.tier_download_counts.get(tier, 0) + len(downloaded)
                        d.sources.extend(downloaded)
                        d._save_sources()

                    logger.info(f"  Batch retry: recovered {len(downloaded)} video(s) for '{keyword}' ({tier})")
                else:
                    # Still failing - record in retry budget
                    d.retry_queue.mark_failed(item.video_id)
                    # US-129-002: Record retry attempt for budget tracking
                    if d._retry_budget:
                        d._retry_budget.record_attempt(item.video_id, backoff_seconds=0.0)

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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics including coordinator status.

        Returns:
            Dict with coordinator and batch stats
        """
        stats: Dict[str, Any] = {}

        if self._coordinator:
            stats['coordinator'] = self._coordinator.get_status()

        stats['batch'] = self.get_batch_stats()

        return stats


@dataclass
class SegmentDownloadResult:
    """Result of a single segment download attempt.

    US-82-007: Typed result returned by SegmentDownloadOrchestrator.download_segment().
    """

    success: bool
    duration: float = 0.0
    error_msg: str = ''
    file_missing: bool = False
    output_path: str = ''
    impersonation_target: Optional[str] = None


class SegmentDownloadOrchestrator:
    """Orchestrates individual segment downloads for the DOWNLOAD_SEGMENTS stage.

    US-82-007: Encapsulates VideoDownloader instantiation, escalation arg
    application, ydl_opts construction, and single-segment download execution.

    The DownloadVideoSegmentsStage delegates download mechanics to this class
    instead of directly constructing a VideoDownloader and managing yt-dlp opts.

    US-129-009: Supports cancellation token for graceful shutdown.
    """

    def __init__(self, config: 'Config', cancellation_token: Optional[CancellationToken] = None) -> None:
        """Instantiate a VideoDownloader and wire up escalation/cookie managers.

        Args:
            config: Application Config (used to construct VideoDownloader).
            cancellation_token: Optional token for signaling cancellation
        """
        from .core import VideoDownloader

        self._config = config
        self._downloader = VideoDownloader(config=config)
        self.cancellation_token = cancellation_token or CancellationToken()

    # -- public properties for stage-level access ----------------------------

    @property
    def downloader(self) -> 'VideoDownloader':
        """Expose underlying VideoDownloader for retry-queue and checkpoint access."""
        return self._downloader

    @property
    def escalation_manager(self):
        return getattr(self._downloader, 'escalation_manager', None)

    @property
    def cookie_rotator(self):
        return getattr(self._downloader, 'cookie_rotator', None)

    @property
    def circuit_breaker(self):
        return getattr(self._downloader, 'circuit_breaker', None)

    @property
    def impersonation_manager(self):
        return getattr(self._downloader, 'impersonation_manager', None)

    @property
    def retry_queue(self):
        return getattr(self._downloader, 'retry_queue', None)

    @property
    def download_config(self):
        return getattr(self._downloader, 'download_config', None)

    # -- core download method ------------------------------------------------

    def download_segment(
        self,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        *,
        progress_hooks: Optional[List] = None,
        stall_timeout: int = 120,
        cookie_file_override: Optional[str] = None,
    ) -> SegmentDownloadResult:
        """Download a single video segment via yt-dlp Python API.

        Encapsulates ydl_opts construction, escalation application, cookie
        propagation, and stall-timeout execution.

        Args:
            video_id: YouTube video ID.
            start: Segment start time in seconds (with buffer already applied).
            end: Segment end time in seconds (with buffer already applied).
            output_file: Destination file path.
            progress_hooks: Optional yt-dlp progress hook callables.
            stall_timeout: Seconds before killing a stalled download (0=no timeout).

        Returns:
            SegmentDownloadResult with success/error info.
        """
        import yt_dlp

        url = f"https://www.youtube.com/watch?v={video_id}"

        ydl_opts, escalation_result, imp_target = self._build_ydl_opts(
            video_id=video_id,
            start=start,
            end=end,
            output_file=output_file,
            progress_hooks=progress_hooks,
            cookie_file_override=cookie_file_override,
        )

        if escalation_result:
            try:
                if escalation_result.tier.value > 1:
                    logger.info(
                        f"Segment {video_id}: using escalation tier "
                        f"{escalation_result.tier.name}"
                    )
            except (TypeError, AttributeError):
                pass

        seg_start_time = time.time()
        try:
            if stall_timeout and stall_timeout > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    def _do_download():
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])

                    future = executor.submit(_do_download)
                    try:
                        future.result(timeout=stall_timeout)
                    except concurrent.futures.TimeoutError:
                        elapsed = time.time() - seg_start_time
                        logger.warning(
                            f"Segment {video_id}: ydl.download() stalled for "
                            f"{elapsed:.1f}s (timeout={stall_timeout}s) — killing"
                        )
                        raise TimeoutError(
                            f"ydl.download() stalled for {elapsed:.1f}s "
                            f"(segment_stall_timeout={stall_timeout}s)"
                        )
            else:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            duration = time.time() - seg_start_time
            if output_file.exists():
                return SegmentDownloadResult(
                    success=True,
                    duration=duration,
                    output_path=str(output_file),
                    impersonation_target=imp_target,
                )
            return SegmentDownloadResult(
                success=False,
                file_missing=True,
                duration=duration,
                impersonation_target=imp_target,
            )
        except Exception as e:
            return SegmentDownloadResult(
                success=False,
                error_msg=str(e),
                impersonation_target=imp_target,
            )

    # -- get_stats -----------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the orchestrator's underlying components.

        Returns:
            Dict with circuit_breaker, escalation, and retry_queue info.
        """
        stats: Dict[str, Any] = {}

        cb = self.circuit_breaker
        if cb:
            stats['circuit_breaker'] = {
                'total_trips': cb.state.total_trips,
                'total_paused_seconds': round(cb.state.total_paused_seconds, 1),
            }

        esc_mgr = self.escalation_manager
        if esc_mgr and hasattr(esc_mgr, 'get_metrics'):
            try:
                stats['escalation'] = esc_mgr.get_metrics()
            except Exception:
                pass

        rq = self.retry_queue
        if rq:
            try:
                stats['retry_queue'] = rq.get_stats()
            except Exception:
                pass

        return stats

    # -- private helpers (moved from download_segments stage) -----------------

    def _build_ydl_opts(
        self,
        *,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        progress_hooks: Optional[List] = None,
        cookie_file_override: Optional[str] = None,
    ) -> tuple:
        """Build ydl_opts dict for a yt-dlp Python API download call.

        Encapsulates base options, cookie propagation, and escalation application.

        Returns:
            (ydl_opts, escalation_result) tuple. escalation_result may be None.
        """
        dl_cfg = self.download_config
        esc_mgr = self.escalation_manager
        cookie_rotator = self.cookie_rotator

        # Read segment config from download config with fallback defaults
        _socket_timeout = 30
        _max_res = 1080
        _seg_format = 'best[height<={segment_max_resolution}]'
        if dl_cfg:
            _seg_sock = getattr(dl_cfg, 'segment_socket_timeout', 0)
            _socket_timeout = _seg_sock if _seg_sock else getattr(dl_cfg, 'socket_timeout', 30)
            _max_res = getattr(dl_cfg, 'segment_max_resolution', 1080)
            _seg_format = getattr(dl_cfg, 'segment_format', _seg_format)

        # Build format string with fallback chain
        _primary_format = _seg_format.format(segment_max_resolution=_max_res)
        _format_with_fallback = f'{_primary_format}/best/bestvideo+bestaudio'

        ydl_opts: Dict[str, Any] = {
            'format': _format_with_fallback,
            'outtmpl': str(output_file),
            'quiet': True,
            'no_warnings': True,
            'ignore_no_formats_error': True,
            'remote_components': {'ejs:github'},
            'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
            'force_keyframes_at_cuts': True,
            'socket_timeout': _socket_timeout,
            'retries': 10,
            'fragment_retries': 10,
        }

        if progress_hooks:
            ydl_opts['progress_hooks'] = progress_hooks

        # Cookie propagation
        if dl_cfg:
            _browser = getattr(dl_cfg, 'cookies_from_browser', '')
            if _browser:
                ydl_opts['cookiesfrombrowser'] = [_browser]
            else:
                _cookies_path = getattr(dl_cfg, 'cookies_path', '')
                if not _cookies_path:
                    _cookie_rotation = getattr(dl_cfg, 'cookie_rotation', None)
                    if _cookie_rotation:
                        _cookie_files = getattr(_cookie_rotation, 'cookie_files', [])
                        if _cookie_files:
                            _cookies_path = _cookie_files[0]
                if _cookies_path:
                    ydl_opts['cookiefile'] = _cookies_path

        # Concurrent worker cookie override — takes precedence over config defaults
        if cookie_file_override:
            ydl_opts['cookiefile'] = cookie_file_override
            ydl_opts.pop('cookiesfrombrowser', None)

        # Escalation application
        escalation_result = None
        imp_target = None
        if esc_mgr:
            try:
                escalation_result = esc_mgr.get_escalation_args(video_id)
                _apply_escalation_to_ydl_opts(ydl_opts, escalation_result)
                imp_target = getattr(escalation_result, 'impersonation_target', None)

                # Tier 3: apply cookie rotation (skip if worker has dedicated cookie)
                if escalation_result.rotate_cookies and cookie_rotator and not cookie_file_override:
                    cookie_path = cookie_rotator.get_current_cookie()
                    if cookie_path:
                        ydl_opts['cookiefile'] = cookie_path
                        ydl_opts.pop('cookiesfrombrowser', None)
            except Exception as esc_err:
                logger.debug(f"Escalation lookup failed for {video_id}: {esc_err}")
        elif self.impersonation_manager:
            try:
                imp_args, imp_target = self.impersonation_manager.get_impersonate_args_with_target()
                if len(imp_args) >= 2 and imp_args[0] == '--impersonate':
                    ydl_opts['impersonate'] = imp_args[1]
            except Exception:
                pass

        return ydl_opts, escalation_result, imp_target


def _apply_escalation_to_ydl_opts(ydl_opts: Dict[str, Any], escalation_result) -> None:
    """Translate EscalationResult CLI args to yt-dlp Python API ydl_opts.

    The EscalationManager returns CLI args (e.g., ['--impersonate', 'X',
    '--extractor-args', 'youtube:player_client=a,b']). This function
    translates them to ydl_opts dict keys for the Python API.

    Translation:
        --impersonate X           -> ydl_opts['impersonate'] = 'X'
        --extractor-args youtube:player_client=X  -> ydl_opts['extractor_args'] = ...

    Cookie rotation is handled separately via cookiefile, not via CLI args.
    """
    if not escalation_result or not escalation_result.args:
        return

    args = escalation_result.args
    i = 0
    while i < len(args):
        if args[i] == '--impersonate' and i + 1 < len(args):
            try:
                from yt_dlp.networking.impersonate import ImpersonateTarget
            except ImportError:
                logger.warning(
                    "ImpersonateTarget not available in this yt-dlp version; "
                    "skipping impersonation for '%s'", args[i + 1]
                )
                i += 2
                continue
            try:
                target = ImpersonateTarget.from_str(args[i + 1])
                target = ImpersonateTarget(
                    client=target.client.lower() if target.client else None,
                    version=target.version,
                    os=target.os.lower() if target.os else None,
                    os_version=target.os_version,
                )
                ydl_opts['impersonate'] = target
            except Exception:
                logger.warning(
                    "Failed to parse impersonate target '%s'; skipping impersonation",
                    args[i + 1]
                )
            i += 2
        elif args[i] == '--extractor-args' and i + 1 < len(args):
            logger.debug(
                "Skipping escalation extractor_args for segment download: %s",
                args[i + 1],
            )
            i += 2
        else:
            i += 1
