"""
Caption Stage - Fetch YouTube captions for transcript-first matching.

When enabled, fetches YouTube captions/subtitles before any media download.
Videos with captions skip audio download and Whisper transcription entirely.
Videos without captions are marked for audio fallback.

In caption-first mode, this stage runs AFTER VIDEO_METADATA and BEFORE DOWNLOAD:
  VIDEO_METADATA -> CAPTION -> DOWNLOAD (only uncaptioned) -> TRANSCRIBE (only uncaptioned)

This enables massive bandwidth savings since caption fetch is ~1KB vs
audio download being 5-50MB per video.

Supports parallel fetching with multiple proxies for faster processing of large
video batches. When parallel mode is enabled, each worker gets a dedicated proxy.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, CaptionDownload
    from ..downloader.caption_fetcher import CaptionFetcher, CaptionResult
    from ..downloader.proxy_manager import ProxyPool, ProxyInfo

logger = logging.getLogger(__name__)


def _safe_print(*args, **kwargs):
    """Print with flush, but handle Windows OSError when stdout is redirected."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


@dataclass
class WorkerStats:
    """Per-worker statistics for parallel caption fetching."""
    worker_id: int
    proxy_url: Optional[str]
    processed: int = 0
    cache_hits: int = 0
    network_fetches: int = 0
    successes: int = 0
    failures: int = 0
    rate_limits: int = 0
    total_time: float = 0.0

    def log_summary(self) -> None:
        """Log worker statistics summary."""
        logger.info(
            f"[worker_{self.worker_id}] SUMMARY: "
            f"processed={self.processed}, cache={self.cache_hits}, network={self.network_fetches}, "
            f"success={self.successes}, fail={self.failures}, 429s={self.rate_limits}, "
            f"time={self.total_time:.1f}s, proxy={self.proxy_url or 'direct'}"
        )


class ParallelCaptionFetcher:
    """
    Parallel caption fetching with dedicated proxy per worker.

    Uses ThreadPoolExecutor to fetch captions from multiple videos simultaneously,
    with each worker using a different proxy to distribute rate limits.

    Features:
    - N workers with dedicated proxies
    - Shared work queue for load balancing
    - Per-worker delay only after network requests (cache hits are instant)
    - Comprehensive logging with worker IDs
    - Thread-safe result collection
    """

    def __init__(
        self,
        caption_fetcher: 'CaptionFetcher',
        proxy_pool: 'ProxyPool',
        num_workers: int = 4,
        per_worker_delay: float = 10.0,
        queue_timeout: float = 60.0,
        worker_timeout: float = 120.0,
        log_activity: bool = True,
        languages: List[str] = None,
        prefer_manual: bool = True,
    ):
        """
        Initialize ParallelCaptionFetcher.

        Args:
            caption_fetcher: CaptionFetcher instance for actual fetching
            proxy_pool: ProxyPool with available proxies
            num_workers: Number of parallel workers
            per_worker_delay: Delay between network requests per worker (seconds)
            queue_timeout: Max time to wait for queue item (seconds)
            worker_timeout: Max time per fetch operation (seconds)
            log_activity: Enable detailed per-worker logging
            languages: Preferred caption languages
            prefer_manual: Prefer manual captions over auto-generated
        """
        self.caption_fetcher = caption_fetcher
        self.proxy_pool = proxy_pool
        self.num_workers = num_workers
        self.per_worker_delay = per_worker_delay
        self.queue_timeout = queue_timeout
        self.worker_timeout = worker_timeout
        self.log_activity = log_activity
        self.languages = languages or ["en", "en-US", "en-GB"]
        self.prefer_manual = prefer_manual

        self._results: Dict[str, Any] = {}  # video_id -> CaptionResult or None
        self._results_lock = threading.Lock()
        self._worker_stats: Dict[int, WorkerStats] = {}

    def fetch_all(self, video_ids: List[str]) -> Dict[str, Any]:
        """
        Fetch captions for all video IDs using parallel workers.

        Args:
            video_ids: List of YouTube video IDs to fetch

        Returns:
            Dict mapping video_id to CaptionResult (or None if failed)
        """
        overall_start = time.time()

        logger.info(
            f"[parallel_caption] Starting parallel fetch: "
            f"{len(video_ids)} videos, {self.num_workers} workers"
        )
        _safe_print(f"  >> Parallel caption fetch: {len(video_ids)} videos, {self.num_workers} workers")

        # Allocate proxies
        proxies = self.proxy_pool.get_n_proxies(self.num_workers)
        logger.info(f"[parallel_caption] Allocated {len(proxies)} proxies")

        # Create work queue
        work_queue: Queue[str] = Queue()
        for vid in video_ids:
            work_queue.put(vid)

        # Initialize worker stats
        for i in range(self.num_workers):
            proxy_url = proxies[i].url if i < len(proxies) else None
            self._worker_stats[i] = WorkerStats(worker_id=i, proxy_url=proxy_url)

        # Run workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._worker_loop,
                    worker_id=i,
                    work_queue=work_queue,
                    proxy_url=proxies[i].url if i < len(proxies) else None
                )
                for i in range(self.num_workers)
            ]

            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"[parallel_caption] Worker failed: {e}")

        # Log final stats
        overall_elapsed = time.time() - overall_start
        total_processed = sum(s.processed for s in self._worker_stats.values())
        total_cache = sum(s.cache_hits for s in self._worker_stats.values())
        total_network = sum(s.network_fetches for s in self._worker_stats.values())
        total_success = sum(s.successes for s in self._worker_stats.values())
        total_fail = sum(s.failures for s in self._worker_stats.values())

        rate = total_processed / overall_elapsed if overall_elapsed > 0 else 0

        # === COMPREHENSIVE LOGGING: Parallel Fetch Summary ===
        logger.info("=" * 60)
        logger.info("[parallel_caption] PARALLEL CAPTION FETCH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"[parallel_caption] Workers: {self.num_workers}")
        logger.info(f"[parallel_caption] Videos processed: {total_processed}/{len(video_ids)}")
        logger.info(f"[parallel_caption] Time elapsed: {overall_elapsed:.1f}s")
        logger.info(f"[parallel_caption] Throughput: {rate:.1f} videos/sec")
        logger.info(f"[parallel_caption] Results:")
        logger.info(f"[parallel_caption]   - Cache hits: {total_cache} (instant)")
        logger.info(f"[parallel_caption]   - Network fetches: {total_network}")
        logger.info(f"[parallel_caption]   - Successes: {total_success}")
        logger.info(f"[parallel_caption]   - Failures: {total_fail}")
        if total_cache > 0:
            cache_percent = total_cache / total_processed * 100 if total_processed > 0 else 0
            logger.info(f"[parallel_caption] Cache efficiency: {cache_percent:.1f}%")
        logger.info("=" * 60)

        _safe_print(
            f"  >> Parallel complete: {total_processed} videos in {overall_elapsed:.1f}s "
            f"({rate:.1f}/sec, {total_cache} cached)"
        )

        # Log per-worker summaries
        for stats in self._worker_stats.values():
            stats.log_summary()

        return self._results

    def _worker_loop(
        self,
        worker_id: int,
        work_queue: Queue,
        proxy_url: Optional[str]
    ) -> None:
        """
        Worker loop: pull from queue, fetch, track stats.

        Args:
            worker_id: Worker identifier (0-indexed)
            work_queue: Shared queue of video IDs
            proxy_url: Proxy URL for this worker (or None for direct)
        """
        stats = self._worker_stats[worker_id]
        log_prefix = f"[worker_{worker_id}]"

        logger.info(f"{log_prefix} Started with proxy: {proxy_url or 'direct'}")

        while True:
            try:
                video_id = work_queue.get(timeout=2.0)
            except Empty:
                logger.debug(f"{log_prefix} Queue empty, exiting")
                break

            start_time = time.time()

            try:
                # Fetch caption using the main fetcher (with proxy and worker_id)
                result = self.caption_fetcher.fetch_captions(
                    video_id,
                    languages=self.languages,
                    prefer_manual=self.prefer_manual,
                    timeout=int(self.worker_timeout),
                    proxy=proxy_url,
                    worker_id=worker_id
                )

                # Track if from cache
                from_cache = result is not None and getattr(result, 'from_cache', False)

                if from_cache:
                    stats.cache_hits += 1
                else:
                    stats.network_fetches += 1

                # Track result
                if result is not None:
                    stats.successes += 1
                else:
                    stats.failures += 1

                # Store result
                with self._results_lock:
                    self._results[video_id] = result

                elapsed = time.time() - start_time
                stats.processed += 1
                stats.total_time += elapsed

                if self.log_activity:
                    status = '✓' if result else '✗'
                    source = 'cache' if from_cache else 'network'
                    logger.info(
                        f"{log_prefix} {status} {video_id} "
                        f"({source}, {elapsed:.2f}s) "
                        f"[{stats.processed} done]"
                    )

                # Delay only after network requests
                if not from_cache and self.per_worker_delay > 0:
                    logger.debug(f"{log_prefix} Sleeping {self.per_worker_delay}s after network fetch")
                    time.sleep(self.per_worker_delay)

            except Exception as e:
                logger.error(f"{log_prefix} Error fetching {video_id}: {e}")
                stats.failures += 1
                elapsed = time.time() - start_time
                stats.processed += 1
                stats.total_time += elapsed

                # Check if rate limit error
                if "429" in str(e) or "rate" in str(e).lower():
                    stats.rate_limits += 1
                    logger.warning(f"{log_prefix} Rate limited on {video_id}")

                with self._results_lock:
                    self._results[video_id] = None

            finally:
                work_queue.task_done()

        logger.info(f"{log_prefix} Finished: {stats.processed} videos processed")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        return {
            "workers": {
                i: {
                    "processed": s.processed,
                    "cache_hits": s.cache_hits,
                    "network_fetches": s.network_fetches,
                    "successes": s.successes,
                    "failures": s.failures,
                    "rate_limits": s.rate_limits,
                    "total_time": s.total_time,
                    "proxy": s.proxy_url,
                }
                for i, s in self._worker_stats.items()
            },
            "totals": {
                "processed": sum(s.processed for s in self._worker_stats.values()),
                "cache_hits": sum(s.cache_hits for s in self._worker_stats.values()),
                "network_fetches": sum(s.network_fetches for s in self._worker_stats.values()),
                "successes": sum(s.successes for s in self._worker_stats.values()),
                "failures": sum(s.failures for s in self._worker_stats.values()),
                "rate_limits": sum(s.rate_limits for s in self._worker_stats.values()),
            }
        }


@register_stage
class CaptionStage(Stage):
    """
    Fetches YouTube captions for transcript-first matching.

    Inputs (in order of priority):
        - state.video_candidates: List of VideoCandidate from VIDEO_METADATA stage
        - state.downloaded_audio: List of AudioDownload (audio-first mode fallback)
        - state.downloaded_videos: List of DownloadedVideo (traditional mode fallback)
        - config.download.caption_first: Caption-first configuration

    Outputs:
        - state.caption_downloads: List of CaptionDownload
        - state.transcripts: Dict[video_id, List[segment]] (from captions)
        - state.videos_need_audio: List of video_ids that need Whisper fallback
        - state.video_candidates: Updated with caption info (has_captions, is_auto_caption)
    """

    name = "CAPTION"
    description = "Fetch YouTube captions for videos"

    def __init__(self):
        self.caption_fetcher = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the caption fetching stage"""
        warnings = []

        # Check if caption-first mode is enabled
        caption_config = getattr(config.download, 'caption_first', None)
        # Handle both dict and object config patterns (Rule 6)
        if isinstance(caption_config, dict):
            is_enabled = caption_config.get('enabled', False)
        else:
            is_enabled = getattr(caption_config, 'enabled', False) if caption_config else False
        if not is_enabled:
            logger.info("Caption-first mode not enabled, skipping CAPTION stage")
            return StageResult.ok({'skipped': True, 'reason': 'not_enabled'}, warnings)

        try:
            # Initialize caption fetcher with project-specific cache
            from ..downloader.caption_fetcher import CaptionFetcher

            # Use project directory for cache (not global cache)
            project_dir = getattr(state, 'project_dir', None)
            if project_dir:
                cache_dir = Path(project_dir) / ".cache" / "captions"
            else:
                cache_dir = Path(config.cache.cache_dir) / "captions"
            self.caption_fetcher = CaptionFetcher(config, cache_dir)

            # Get video IDs to fetch captions for
            video_ids = self._get_video_ids(state, config)
            if not video_ids:
                logger.warning("No video IDs to fetch captions for")
                return StageResult.ok({'skipped': True, 'reason': 'no_videos'}, warnings)

            print(f"  >> Fetching captions for {len(video_ids)} videos...")
            logger.info(f"Fetching captions for {len(video_ids)} videos")

            # Fetch captions
            results = self._fetch_captions_batch(
                video_ids,
                state,
                config,
                caption_config
            )

            # Report results
            caption_count = len(state.caption_downloads)
            fallback_count = len(state.videos_need_audio)

            print(f"  >> Captions: {caption_count} found, {fallback_count} need audio fallback")
            logger.info(f"Caption results: {caption_count} with captions, {fallback_count} need audio")

            if caption_count == 0 and fallback_count > 0:
                warnings.append(f"No captions found for any videos - all {fallback_count} will use Whisper fallback")

            # Return full checkpoint data in StageResult (pipeline saves this)
            checkpoint_data = {
                'caption_downloads': [
                    {
                        'file': cd.file,
                        'video_id': cd.video_id,
                        'url': cd.url,
                        'title': cd.title,
                        'duration': cd.duration,
                        'keyword': cd.keyword,
                        'language': cd.language,
                        'is_auto_generated': cd.is_auto_generated,
                    }
                    for cd in state.caption_downloads
                ],
                'videos_need_audio': state.videos_need_audio,
                'transcripts_from_captions': list(state.transcripts.keys()),
                'caption_count': caption_count,
                'fallback_count': fallback_count,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Caption stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if caption stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore caption stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            from ..state import CaptionDownload

            # Restore caption downloads
            caption_downloads = data.get('caption_downloads', [])
            state.caption_downloads = [
                CaptionDownload(**cd) if isinstance(cd, dict) else cd
                for cd in caption_downloads
            ]

            # Restore videos needing audio fallback
            state.videos_need_audio = data.get('videos_need_audio', [])

            # Restore transcripts from captions (re-parse caption files)
            if config:
                self._restore_transcripts(state, config)

            logger.info(f"Restored CAPTION: {len(state.caption_downloads)} captions, {len(state.videos_need_audio)} need audio")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore CAPTION: {e}")
            return False

    def _get_video_ids(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> List[str]:
        """
        Get list of video IDs to fetch captions for.

        Priority order:
        1. state.video_candidates (caption-first mode - VIDEO_METADATA stage output)
        2. state.downloaded_audio (audio-first mode)
        3. state.downloaded_videos (traditional full download mode)
        """
        video_ids = []

        # Priority 1: From video candidates (caption-first mode)
        # This is the primary input when VIDEO_METADATA stage has run
        if state.video_candidates:
            for candidate in state.video_candidates:
                if candidate.video_id:
                    video_ids.append(candidate.video_id)
            logger.debug(f"Using {len(video_ids)} video IDs from video_candidates")

        # Priority 2: From audio downloads (audio-first mode fallback)
        if not video_ids:
            for audio in state.downloaded_audio:
                if audio.video_id:
                    video_ids.append(audio.video_id)
            if video_ids:
                logger.debug(f"Using {len(video_ids)} video IDs from downloaded_audio")

        # Priority 3: From downloaded videos (full download mode fallback)
        if not video_ids:
            for video in state.downloaded_videos:
                video_id = self._extract_video_id(video.url or video.file)
                if video_id:
                    video_ids.append(video_id)
            if video_ids:
                logger.debug(f"Using {len(video_ids)} video IDs from downloaded_videos")

        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid)
                unique_ids.append(vid)

        return unique_ids

    def _extract_video_id(self, url_or_path: str) -> Optional[str]:
        """Extract YouTube video ID from URL or filename"""
        import re

        # Try URL patterns
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})(?:\.mp[34]|\.webm|\.mkv)?$',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_path)
            if match:
                return match.group(1)

        return None

    def _fetch_captions_batch(
        self,
        video_ids: List[str],
        state: 'PipelineState',
        config: 'Config',
        caption_config
    ) -> Dict[str, Any]:
        """Fetch captions for a batch of videos"""
        from ..state import CaptionDownload
        import time as time_mod
        stage_start_time = time_mod.time()

        # Helper for Rule 6: handle both dict and object config patterns
        def get_cfg(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        languages = get_cfg(caption_config, 'languages', ["en", "en-US", "en-GB"])
        prefer_manual = get_cfg(caption_config, 'prefer_manual_captions', True)
        timeout = get_cfg(caption_config, 'fetch_timeout', 30)

        # Check parallel mode settings
        parallel_enabled = get_cfg(caption_config, 'parallel_enabled', False)
        parallel_workers = get_cfg(caption_config, 'parallel_workers', 4)
        parallel_min_batch = get_cfg(caption_config, 'parallel_min_batch', 10)
        per_worker_delay = get_cfg(caption_config, 'per_worker_delay', 10.0)
        worker_timeout = get_cfg(caption_config, 'worker_timeout', 120.0)
        log_worker_activity = get_cfg(caption_config, 'log_worker_activity', True)

        # Build video info lookup (for title, duration, keyword)
        # Also build video_id -> video_path mapping for transcript keying
        # Priority: video_candidates > downloaded_audio > downloaded_videos
        video_info = {}
        video_id_to_path = {}
        video_candidate_lookup = {}  # To update VideoCandidate objects

        # From video_candidates (caption-first mode primary source)
        for candidate in state.video_candidates:
            video_info[candidate.video_id] = {
                'url': candidate.url,
                'title': candidate.title,
                'duration': candidate.duration,
                'keyword': candidate.keyword,
                'file': '',  # No file downloaded yet in caption-first mode
            }
            video_candidate_lookup[candidate.video_id] = candidate

        # From downloaded_audio (audio-first mode)
        for audio in state.downloaded_audio:
            video_info[audio.video_id] = {
                'url': audio.url,
                'title': audio.title,
                'duration': audio.duration,
                'keyword': audio.keyword,
                'file': audio.file,
            }
            video_id_to_path[audio.video_id] = audio.file

        # From downloaded_videos (traditional mode)
        for video in state.downloaded_videos:
            vid = self._extract_video_id(video.url or video.file)
            if vid:
                video_info[vid] = {
                    'url': video.url,
                    'title': video.title,
                    'duration': video.duration,
                    'keyword': video.keyword,
                    'file': video.file,
                }
                video_id_to_path[vid] = video.file

        caption_downloads = []
        videos_need_audio = []

        # Determine if we should use parallel mode
        has_proxy = self._has_proxy_pool(config)
        logger.debug(f"[caption] Parallel check: enabled={parallel_enabled}, videos={len(video_ids)}>={parallel_min_batch}, has_proxy={has_proxy}")
        use_parallel = (
            parallel_enabled and
            len(video_ids) >= parallel_min_batch and
            has_proxy
        )

        if use_parallel:
            # === PARALLEL MODE ===
            logger.info(f"[caption] Using PARALLEL mode: {len(video_ids)} videos, {parallel_workers} workers")
            _safe_print(f"  >> Using parallel mode ({parallel_workers} workers)")

            # Get proxy pool
            from ..downloader.proxy_manager import ProxyPool
            proxy_pool = self._get_proxy_pool(config)

            # Create parallel fetcher
            parallel_fetcher = ParallelCaptionFetcher(
                caption_fetcher=self.caption_fetcher,
                proxy_pool=proxy_pool,
                num_workers=parallel_workers,
                per_worker_delay=per_worker_delay,
                worker_timeout=worker_timeout,
                log_activity=log_worker_activity,
                languages=languages,
                prefer_manual=prefer_manual,
            )

            # Fetch all captions in parallel
            results = parallel_fetcher.fetch_all(video_ids)

            # Process results
            for video_id in video_ids:
                result = results.get(video_id)
                self._process_caption_result(
                    result=result,
                    video_id=video_id,
                    state=state,
                    video_info=video_info,
                    video_id_to_path=video_id_to_path,
                    video_candidate_lookup=video_candidate_lookup,
                    caption_downloads=caption_downloads,
                    videos_need_audio=videos_need_audio,
                )

        else:
            # === SEQUENTIAL MODE ===
            logger.info(f"[caption] Using SEQUENTIAL mode: {len(video_ids)} videos")

            for i, video_id in enumerate(video_ids):
                _safe_print(f"    [{i+1}/{len(video_ids)}] {video_id}...", end=" ", flush=True)

                result = self.caption_fetcher.fetch_captions(
                    video_id,
                    languages=languages,
                    prefer_manual=prefer_manual,
                    timeout=timeout
                )

                # Process result using shared helper
                self._process_caption_result(
                    result=result,
                    video_id=video_id,
                    state=state,
                    video_info=video_info,
                    video_id_to_path=video_id_to_path,
                    video_candidate_lookup=video_candidate_lookup,
                    caption_downloads=caption_downloads,
                    videos_need_audio=videos_need_audio,
                    print_result=True,  # Print to console in sequential mode
                )

                # Rate limit prevention: delay between caption fetches (skip for cache hits)
                from_cache = result is not None and result.from_cache
                delay = getattr(config.download, 'delay_between_downloads', 3.0)
                if delay > 0 and i < len(video_ids) - 1 and not from_cache:
                    logger.debug(f"[rate_limit] Sleeping {delay}s between caption fetches ({i+1}/{len(video_ids)})")
                    time.sleep(delay)

        state.caption_downloads = caption_downloads
        state.videos_need_audio = videos_need_audio

        # === COMPREHENSIVE LOGGING: Caption Stage Summary ===
        import time as time_mod
        elapsed = time_mod.time() - stage_start_time if 'stage_start_time' in dir() else 0

        # Count result types
        cache_hits = sum(1 for cd in caption_downloads if getattr(cd, 'from_cache', False))
        fresh_fetches = len(caption_downloads) - cache_hits
        no_caption_count = len(videos_need_audio)

        logger.info("=" * 60)
        logger.info("[caption] CAPTION STAGE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"[caption] Total videos processed: {len(video_ids)}")
        logger.info(f"[caption] Mode: {'PARALLEL (' + str(parallel_workers) + ' workers)' if use_parallel else 'SEQUENTIAL'}")
        logger.info(f"[caption] Results:")
        logger.info(f"[caption]   - Captions fetched: {len(caption_downloads)}")
        logger.info(f"[caption]     - Cache hits: {cache_hits} (instant)")
        logger.info(f"[caption]     - Fresh fetches: {fresh_fetches}")
        logger.info(f"[caption]   - No captions available: {no_caption_count}")
        logger.info(f"[caption] Success rate: {len(caption_downloads) / len(video_ids) * 100:.1f}%")
        if elapsed > 0:
            logger.info(f"[caption] Elapsed time: {elapsed:.1f}s ({len(video_ids) / elapsed:.1f} videos/sec)")
        logger.info("=" * 60)

        _safe_print(f"  >> Caption stage complete: {len(caption_downloads)} captions, {no_caption_count} need audio fallback")

        return {
            'caption_count': len(caption_downloads),
            'fallback_count': len(videos_need_audio),
            'cache_hits': cache_hits,
            'fresh_fetches': fresh_fetches,
        }

    def _has_proxy_pool(self, config: 'Config') -> bool:
        """Check if a proxy pool is available for parallel fetching."""
        try:
            # Handle both dict and object config patterns (Rule 6)
            download_cfg = config.download
            if isinstance(download_cfg, dict):
                fallback_cfg = download_cfg.get('fallback', None)
            else:
                fallback_cfg = getattr(download_cfg, 'fallback', None)
            if not fallback_cfg:
                return False

            if isinstance(fallback_cfg, dict):
                proxy_cfg = fallback_cfg.get('proxy', None)
            else:
                proxy_cfg = getattr(fallback_cfg, 'proxy', None)
            if not proxy_cfg:
                return False

            # Check enabled
            if isinstance(proxy_cfg, dict):
                enabled = proxy_cfg.get('enabled', False)
            else:
                enabled = getattr(proxy_cfg, 'enabled', False)
            return enabled
        except Exception as e:
            logger.debug(f"[caption] _has_proxy_pool error: {e}")
            return False

    def _get_proxy_pool(self, config: 'Config') -> 'ProxyPool':
        """Get the proxy pool for parallel fetching."""
        from ..downloader.proxy_manager import ProxyManager
        manager = ProxyManager(config)
        return manager.pool

    def _process_caption_result(
        self,
        result: Optional['CaptionResult'],
        video_id: str,
        state: 'PipelineState',
        video_info: Dict[str, Any],
        video_id_to_path: Dict[str, str],
        video_candidate_lookup: Dict[str, Any],
        caption_downloads: List['CaptionDownload'],
        videos_need_audio: List[str],
        print_result: bool = False
    ) -> None:
        """
        Process a caption fetch result and update state.

        Args:
            result: CaptionResult from fetch (or None if failed)
            video_id: Video ID being processed
            state: Pipeline state to update
            video_info: Video info lookup dict
            video_id_to_path: Video ID to file path mapping
            video_candidate_lookup: VideoCandidate lookup by ID
            caption_downloads: List to append successful downloads
            videos_need_audio: List to append videos needing audio fallback
            print_result: Whether to print result to console
        """
        from ..state import CaptionDownload

        if result:
            # Parse caption file into transcript segments
            segments = self.caption_fetcher.parse_caption_file(result.file)

            if segments:
                # Set transcript source on each segment
                source = 'manual_caption' if not result.is_auto_generated else 'auto_caption'
                for seg in segments:
                    seg['transcript_source'] = source

                # Update cache with segment count
                duration_covered = segments[-1].get('end', 0.0) if segments else 0.0
                self.caption_fetcher.update_segment_count(
                    video_id,
                    len(segments),
                    duration_covered
                )

                # Store in state.transcripts keyed by video_path (for embedding compatibility)
                # Fall back to video_id if no path available
                transcript_key = video_id_to_path.get(video_id, video_id)
                state.transcripts[transcript_key] = segments

                # Also store by video_id for easy lookup during filtering
                if transcript_key != video_id:
                    state.transcripts[video_id] = segments

                # Create CaptionDownload record
                info = video_info.get(video_id, {})
                cd = CaptionDownload(
                    file=result.file,
                    video_id=video_id,
                    url=info.get('url', f"https://youtube.com/watch?v={video_id}"),
                    title=info.get('title', ''),
                    duration=info.get('duration', 0.0),
                    keyword=info.get('keyword', ''),
                    language=result.language,
                    is_auto_generated=result.is_auto_generated,
                )
                caption_downloads.append(cd)

                # Update VideoCandidate if exists (caption-first mode)
                if video_id in video_candidate_lookup:
                    candidate = video_candidate_lookup[video_id]
                    candidate.has_captions = True
                    candidate.caption_language = result.language
                    candidate.is_auto_caption = result.is_auto_generated
                    candidate.transcript_source = source

                if print_result:
                    caption_type = "auto" if result.is_auto_generated else "manual"
                    print(f"{caption_type} ({len(segments)} segments)")
            else:
                # Caption file exists but couldn't parse
                if print_result:
                    print("parse failed -> audio fallback")
                videos_need_audio.append(video_id)
        else:
            if print_result:
                print("no captions -> audio fallback")
            videos_need_audio.append(video_id)

    def _restore_transcripts(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> None:
        """Re-parse caption files to restore transcripts"""
        from ..downloader.caption_fetcher import CaptionFetcher

        # Use project directory for cache (same as run method)
        project_dir = getattr(state, 'project_dir', None)
        if project_dir:
            cache_dir = Path(project_dir) / ".cache" / "captions"
        else:
            cache_dir = Path(config.cache.cache_dir) / "captions"
        fetcher = CaptionFetcher(config, cache_dir)

        # Build video_id -> video_path mapping for proper keying
        # Also build video_candidate_lookup to restore caption flags
        video_id_to_path = {}
        video_candidate_lookup = {}

        for candidate in state.video_candidates:
            video_candidate_lookup[candidate.video_id] = candidate

        for audio in state.downloaded_audio:
            video_id_to_path[audio.video_id] = audio.file
        for video in state.downloaded_videos:
            vid = self._extract_video_id(video.url or video.file)
            if vid:
                video_id_to_path[vid] = video.file

        for cd in state.caption_downloads:
            if Path(cd.file).exists():
                segments = fetcher.parse_caption_file(cd.file)
                if segments:
                    source = 'manual_caption' if not cd.is_auto_generated else 'auto_caption'
                    for seg in segments:
                        seg['transcript_source'] = source

                    # Key by video_path for embedding compatibility, fall back to video_id
                    transcript_key = video_id_to_path.get(cd.video_id, cd.video_id)
                    state.transcripts[transcript_key] = segments

                    # Also store by video_id for easy lookup
                    if transcript_key != cd.video_id:
                        state.transcripts[cd.video_id] = segments

                    # Restore VideoCandidate caption flags
                    if cd.video_id in video_candidate_lookup:
                        candidate = video_candidate_lookup[cd.video_id]
                        candidate.has_captions = True
                        candidate.caption_language = cd.language
                        candidate.is_auto_caption = cd.is_auto_generated
                        candidate.transcript_source = source
