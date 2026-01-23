"""
Audio-first download pipeline.

Phase 1: Download audio only
Phase 3: Download matched video segments

Migrated from VideoDownloader audio-first methods (lines 2078-2556).
"""

from __future__ import annotations

import subprocess
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from ..state import AudioDownload
from .types import MergedSegment, DownloadedSegment
from . import segment_utils
from . import utils

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5  # seconds


@dataclass
class VideoDownloadResult:
    """Result from downloading a single video's segments."""
    video_id: str
    segments: List[DownloadedSegment]
    cache_hit: bool = False
    success: bool = False
    error: Optional[str] = None
    download_time: float = 0.0
    tier: str = "medium"
DEFAULT_SEGMENT_TIMEOUT = 300  # 5 minutes per video segment download


class AudioFirstPipeline:
    """Orchestrates audio-first download workflow.

    Migrated from VideoDownloader audio-first methods.
    """

    def __init__(
        self,
        config: 'Config',
        get_tier_value_func,
        search_metadata_func,
        filter_titles_func,
        cleanup_partial_func,
        tier_download_counts: dict,
        lock
    ):
        """
        Initialize AudioFirstPipeline.

        Args:
            config: Config object
            get_tier_value_func: Function to get tier config values
            search_metadata_func: Function to search YouTube metadata
            filter_titles_func: Function to filter titles with LLM
            cleanup_partial_func: Function to clean up partial files
            tier_download_counts: Dict tracking downloads per tier
            lock: Threading lock for tier_download_counts
        """
        self.config = config
        self.download_config = config.download
        self._get_tier_value = get_tier_value_func
        self._search_video_metadata = search_metadata_func
        self._filter_titles_with_llm = filter_titles_func
        self._cleanup_partial_files = cleanup_partial_func
        self.tier_download_counts = tier_download_counts
        self._lock = lock

    def download_audio_for_keyword(
        self,
        keyword: str,
        output_dir: Path,
        tier: str,
        topic: str = ""
    ) -> List[AudioDownload]:
        """
        Download audio only (MP3) for videos matching keyword.

        Migrated from downloader.py lines 2078-2282.

        Phase 1 of audio-first pipeline. Downloads lightweight MP3 files
        for transcription and matching, before video segments.

        Args:
            keyword: Search keyword
            output_dir: Base output directory
            tier: Duration tier ('short', 'medium', 'long', 'longer')
            topic: Optional topic for LLM filter context

        Returns:
            List of AudioDownload records
        """
        audio_config = getattr(self.download_config, 'audio_first', None)
        if not audio_config:
            logger.error("Audio-first config not found")
            return []

        # Check max_total limit for this tier (e.g., only 1 LONGER video total)
        max_total = self._get_tier_value(tier, 'max_total', 0)  # 0 = no limit
        if max_total > 0 and self.tier_download_counts.get(tier, 0) >= max_total:
            logger.debug(f"  [{tier}] Skipped (max_total={max_total} reached)")
            return []

        # Get tier settings
        tier_min = self._get_tier_value(tier, 'min', 20)
        tier_max = self._get_tier_value(tier, 'max', 120)
        per_keyword = self._get_tier_value(tier, 'per_keyword', 5)

        # Create audio output directory
        max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
        safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
        tier_short = tier[0]
        audio_dir = output_dir / f"{safe_keyword}_{tier_short}_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Search for videos
        search_count = max(per_keyword * 5, 40)
        try:
            search_results = self._search_video_metadata(keyword, tier, max_results=search_count)
        except Exception as e:
            logger.error(f"Search failed for '{keyword}': {e}")
            return []

        if not search_results:
            logger.warning(f"No search results for '{keyword}'")
            return []

        # Filter by duration (handle None duration values)
        filtered = [
            v for v in search_results
            if (v.get('duration') or 0) >= tier_min
            and (v.get('duration') or 0) <= tier_max
            and not v.get('is_live', False)   # Skip current live streams
            and not v.get('was_live', False)  # Skip completed livestreams
        ]

        if not filtered:
            logger.warning(f"No videos in duration range for '{keyword}'")
            return []

        # LLM title filter if enabled
        if getattr(self.download_config, 'llm_title_filter', None):
            filter_config = self.download_config.llm_title_filter
            if getattr(filter_config, 'enabled', False):
                filtered = self._filter_titles_with_llm(filtered, keyword, topic)

        # Take top N
        to_download = filtered[:per_keyword]

        # Clean up any leftover .part files from previous failed downloads
        if audio_dir.exists():
            for part_file in audio_dir.glob('*.part*'):
                try:
                    part_file.unlink()
                    logger.debug(f"Cleaned up stale partial file: {part_file.name}")
                except Exception:
                    pass

        # Download audio for each
        audio_downloads = []
        audio_quality = getattr(audio_config, 'audio_quality', 5)

        for video_info in to_download:
            video_id = video_info.get('id', '')
            video_url = video_info.get('webpage_url', f"https://www.youtube.com/watch?v={video_id}")

            # Skip if already downloaded (check multiple audio formats)
            existing_file = None
            for ext in ['.mp3', '.m4a', '.mp4', '.opus', '.webm', '.ogg', '.wav']:
                candidate = audio_dir / f"{video_id}{ext}"
                if candidate.exists():
                    existing_file = candidate
                    break

            if existing_file:
                logger.debug(f"Audio already exists: {existing_file.name}")
                audio_downloads.append(AudioDownload(
                    file=str(existing_file),
                    video_id=video_id,
                    url=video_url,
                    title=video_info.get('title', ''),
                    duration=video_info.get('duration', 0),
                    keyword=keyword
                ))
                continue

            # Build yt-dlp command for audio only
            cmd = [
                'yt-dlp',
                '--sleep-interval', '5',
                video_url,
                '-f', 'bestaudio/best',
                '-x',  # Extract/convert audio
                '--audio-format', 'mp3',
                '--audio-quality', str(audio_quality),
                '-o', str(audio_dir / '%(id)s.%(ext)s'),
                '--no-playlist',
                '--no-warnings',
                '--no-keep-video',
            ]

            # Add ffmpeg location if configured
            ffmpeg_loc = getattr(self.download_config, 'ffmpeg_location', '')
            if ffmpeg_loc:
                cmd.extend(['--ffmpeg-location', ffmpeg_loc])

            # Add base args (JS runtime for challenge solving), cookies, and proxy
            cmd.extend(utils.get_ytdlp_base_args())
            cmd.extend(utils.get_cookies_args(self.config))
            cmd.extend(utils.get_proxy_args(self.config))

            try:
                # Use tier-specific timeout
                tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
                if isinstance(tier_timeouts, dict):
                    audio_timeout = tier_timeouts.get(tier, 120)
                else:
                    audio_timeout = getattr(tier_timeouts, tier, 120)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=audio_timeout
                )

                # Find the actual downloaded file
                actual_file = None
                if result.returncode == 0:
                    matches = list(audio_dir.glob(f"{video_id}.*"))
                    if matches:
                        actual_file = matches[0]
                        logger.debug(f"Found audio file: {actual_file.name}")

                if actual_file:
                    # Report success to proxy manager
                    utils.report_proxy_result(self.config, success=True)

                    audio_downloads.append(AudioDownload(
                        file=str(actual_file),
                        video_id=video_id,
                        url=video_url,
                        title=video_info.get('title', ''),
                        duration=video_info.get('duration', 0),
                        keyword=keyword
                    ))
                    logger.debug(f"Downloaded audio: {actual_file.name}")
                else:
                    err_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                    logger.warning(f"Audio download failed for {video_id} (rc={result.returncode})")
                    logger.warning(f"  Error output: {err_msg}")
                    self._cleanup_partial_files(audio_dir, video_id)

                    # Rate limit detection: exponential backoff + proxy rotation
                    if 'rate-limited' in str(result.stderr).lower() or '429' in str(result.stderr):
                        # Report to proxy manager for automatic rotation
                        utils.report_proxy_result(self.config, success=False, error_text=str(result.stderr))

                        backoff_delay = 30  # Start with 30 seconds for rate limit
                        logger.warning(f"[rate_limit] ⚠️ RATE LIMIT DETECTED for {video_id}")
                        logger.warning(f"[rate_limit] YouTube has rate-limited this account")
                        logger.warning(f"[rate_limit] Applying {backoff_delay}s backoff before continuing...")
                        time.sleep(backoff_delay)
                        logger.info(f"[rate_limit] Backoff complete, resuming downloads")
                    else:
                        # Report non-rate-limit failure
                        utils.report_proxy_result(self.config, success=False, error_text=str(result.stderr))

            except subprocess.TimeoutExpired:
                logger.warning(f"Audio download timeout for {video_id}")
                self._cleanup_partial_files(audio_dir, video_id)
            except Exception as e:
                logger.warning(f"Audio download error for {video_id}: {e}")
                self._cleanup_partial_files(audio_dir, video_id)

            # Rate limit prevention: delay between downloads
            delay = getattr(self.download_config, 'delay_between_downloads', 3.0)
            if delay > 0:
                logger.debug(f"[rate_limit] Sleeping {delay}s between downloads")
                time.sleep(delay)

        # Update tier download count
        if audio_downloads:
            with self._lock:
                self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(audio_downloads)

        logger.info(f"  Downloaded {len(audio_downloads)} audio files for '{keyword}' ({tier})")
        return audio_downloads

    def download_video_segments(
        self,
        merged_segments: List[MergedSegment],
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, List['DownloadedSegment']], None]] = None
    ) -> List[DownloadedSegment]:
        """
        Download video segments using --download-sections.

        Migrated from downloader.py lines 2284-2429.

        Phase 3 of audio-first pipeline. Downloads only the matched portions
        of videos, not the full files.

        Args:
            merged_segments: List of merged segments with buffer applied
            output_dir: Base output directory
            progress_callback: Optional callback(current, total, segments) for progress/checkpointing

        Returns:
            List of DownloadedSegment records with timing info
        """
        audio_config = getattr(self.download_config, 'audio_first', None)
        fallback_full = getattr(audio_config, 'fallback_full_video', True) if audio_config else True

        # Checkpoint save interval (save after every N videos)
        checkpoint_interval = getattr(self.download_config, 'checkpoint_interval', 10)

        # Get caption_first config for tier-based speed settings
        caption_config = getattr(self.download_config, 'caption_first', None)

        # Tier thresholds
        short_threshold = getattr(caption_config, 'segment_tier_short_threshold', 60.0) if caption_config else 60.0
        medium_threshold = getattr(caption_config, 'segment_tier_medium_threshold', 180.0) if caption_config else 180.0

        # Helper to get tier-based settings
        def get_tier_settings(total_duration: float) -> Tuple[str, float, int, float]:
            """Get (tier_name, sleep_interval, concurrent_frags, delay_after) based on duration."""
            if total_duration < short_threshold:
                return (
                    "short",
                    getattr(caption_config, 'segment_tier_short_sleep', 2.0) if caption_config else 2.0,
                    getattr(caption_config, 'segment_tier_short_concurrent', 2) if caption_config else 2,
                    getattr(caption_config, 'segment_tier_short_delay_after', 3.0) if caption_config else 3.0,
                )
            elif total_duration < medium_threshold:
                return (
                    "medium",
                    getattr(caption_config, 'segment_tier_medium_sleep', 1.0) if caption_config else 1.0,
                    getattr(caption_config, 'segment_tier_medium_concurrent', 4) if caption_config else 4,
                    getattr(caption_config, 'segment_tier_medium_delay_after', 1.0) if caption_config else 1.0,
                )
            else:
                return (
                    "long",
                    getattr(caption_config, 'segment_tier_long_sleep', 0.5) if caption_config else 0.5,
                    getattr(caption_config, 'segment_tier_long_concurrent', 6) if caption_config else 6,
                    getattr(caption_config, 'segment_tier_long_delay_after', 0.0) if caption_config else 0.0,
                )

        # Get cookie info for logging
        cookie_info = "none"
        cookie_rotation = getattr(self.download_config, 'cookie_rotation', None)
        if cookie_rotation and getattr(cookie_rotation, 'enabled', False):
            cookie_info = "rotation"
        elif getattr(self.download_config, 'cookies_from_browser', ''):
            cookie_info = f"browser:{getattr(self.download_config, 'cookies_from_browser', '')}"
        elif getattr(self.download_config, 'cookies_path', ''):
            cookie_info = "file"

        downloaded_segments = []

        # Track progress for summary
        cache_hits = 0
        new_downloads = 0
        failed_downloads = 0
        total_download_time = 0.0
        total_segments_downloaded = 0
        tier_counts = {"short": 0, "medium": 0, "long": 0}

        # Group by video_id
        by_video: Dict[str, List[MergedSegment]] = {}
        for seg in merged_segments:
            if seg.video_id not in by_video:
                by_video[seg.video_id] = []
            by_video[seg.video_id].append(seg)

        total_videos = len(by_video)
        current_video = 0

        # Check for parallel mode
        parallel_enabled = getattr(caption_config, 'parallel_segment_downloads', False) if caption_config else False
        parallel_workers = getattr(caption_config, 'parallel_segment_workers', 3) if caption_config else 3
        parallel_stagger = getattr(caption_config, 'parallel_segment_stagger', 2.0) if caption_config else 2.0

        # Get cookie manager for parallel mode (each worker gets dedicated cookie)
        cookie_manager = None
        worker_cookies = []
        if parallel_enabled:
            cookie_rotation = getattr(self.download_config, 'cookie_rotation', None)
            if cookie_rotation and getattr(cookie_rotation, 'enabled', False):
                try:
                    from .cookie_manager import CookieManager
                    cookie_manager = CookieManager.get_instance(None)  # Get existing instance
                    if cookie_manager and cookie_manager.enabled:
                        available = cookie_manager.get_available_accounts()
                        # Assign dedicated cookies to workers (up to parallel_workers)
                        worker_cookies = available[:parallel_workers]
                        if len(worker_cookies) < parallel_workers:
                            logger.warning(f"[SEGMENT_DOWNLOAD] Only {len(worker_cookies)} cookies available for {parallel_workers} workers")
                            parallel_workers = max(1, len(worker_cookies))
                except Exception as e:
                    logger.warning(f"[SEGMENT_DOWNLOAD] Cookie manager setup failed: {e}, using sequential mode")
                    parallel_enabled = False

        # Log configuration
        logger.info(f"[SEGMENT_DOWNLOAD] === STARTING SEGMENT DOWNLOADS ===")
        logger.info(f"[SEGMENT_DOWNLOAD] Videos: {total_videos}, Segments: {len(merged_segments)}")
        logger.info(f"[SEGMENT_DOWNLOAD] Tier thresholds: short<{short_threshold}s, medium<{medium_threshold}s, long>={medium_threshold}s")
        logger.info(f"[SEGMENT_DOWNLOAD] Tier settings: short(sleep=2s,frag=2), medium(sleep=1s,frag=4), long(sleep=0.5s,frag=6)")
        logger.info(f"[SEGMENT_DOWNLOAD] Cookies: {cookie_info}")
        if parallel_enabled:
            cookie_labels = [acc.label for acc in worker_cookies] if worker_cookies else ["default"]
            logger.info(f"[SEGMENT_DOWNLOAD] Mode: PARALLEL ({parallel_workers} workers, stagger={parallel_stagger}s)")
            logger.info(f"[SEGMENT_DOWNLOAD] Worker cookies: {cookie_labels}")
        else:
            logger.info(f"[SEGMENT_DOWNLOAD] Mode: SEQUENTIAL")
        logger.info(f"[SEGMENT_DOWNLOAD] Output: {output_dir}")

        # Thread-safe counters for parallel mode
        stats_lock = threading.Lock()

        # === PARALLEL MODE ===
        if parallel_enabled and parallel_workers > 1:
            # Convert to list for indexing
            video_items = [(vid, segs) for vid, segs in by_video.items() if segs]
            results_queue = Queue()
            video_index = [0]  # Use list for mutable counter in closure
            index_lock = threading.Lock()

            def download_worker(worker_id: int, cookie_account):
                """Worker that downloads videos using dedicated cookie."""
                worker_label = cookie_account.label if cookie_account else f"worker-{worker_id}"
                logger.info(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} started with cookie '{worker_label}'")

                while True:
                    # Get next video to process
                    with index_lock:
                        if video_index[0] >= len(video_items):
                            break
                        idx = video_index[0]
                        video_index[0] += 1

                    video_id, segments = video_items[idx]
                    first_seg = segments[0]
                    video_url = first_seg.video_url
                    keyword = first_seg.keyword
                    total_seg_duration = sum(seg.end_time - seg.start_time for seg in segments)

                    # Get tier-based settings
                    tier_name, sleep_interval, concurrent_frags, delay_after = get_tier_settings(total_seg_duration)

                    logger.info(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} [{idx+1}/{len(video_items)}] {video_id} ({len(segments)} seg, {total_seg_duration:.0f}s) tier={tier_name}")

                    # Create output directory
                    max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
                    safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
                    safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
                    video_dir = output_dir / f"{safe_keyword}_segments"
                    video_dir.mkdir(parents=True, exist_ok=True)

                    result = VideoDownloadResult(video_id=video_id, segments=[], tier=tier_name)

                    # Check cache
                    existing_segments = self._check_existing_segments(video_dir, video_id, segments)
                    existing_count = sum(1 for p in existing_segments if p is not None)

                    if existing_count == len(segments):
                        # Cache hit
                        logger.info(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} [{idx+1}/{len(video_items)}] {video_id} CACHE_HIT ({existing_count} seg)")
                        for seg, file_path in zip(segments, existing_segments):
                            if file_path and Path(file_path).exists():
                                result.segments.append(DownloadedSegment(
                                    file=str(file_path),
                                    video_id=video_id,
                                    original_start=seg.start_time,
                                    original_end=seg.end_time,
                                    file_duration=seg.end_time - seg.start_time,
                                    matches=seg.original_matches,
                                    keyword=keyword
                                ))
                        result.cache_hit = True
                        result.success = True
                        results_queue.put(result)
                        continue

                    # Build download command
                    download_start = time.time()
                    section_args = []
                    for seg in segments:
                        start_str = utils.format_time(seg.start_time)
                        end_str = utils.format_time(seg.end_time)
                        section_args.extend(['--download-sections', f'*{start_str}-{end_str}'])

                    cmd = [
                        'yt-dlp',
                        '--sleep-interval', str(int(sleep_interval)),
                        '--concurrent-fragments', str(concurrent_frags),
                        video_url,
                        *section_args,
                        '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                        '--merge-output-format', 'mp4',
                        '-o', str(video_dir / f'{video_id}_%(autonumber)s.%(ext)s'),
                        '--no-playlist',
                        '--no-warnings',
                    ]

                    ffmpeg_loc = getattr(self.download_config, 'ffmpeg_location', '')
                    if ffmpeg_loc:
                        cmd.extend(['--ffmpeg-location', ffmpeg_loc])

                    cmd.extend(utils.get_ytdlp_base_args())

                    # Use worker's dedicated cookie
                    if cookie_account and cookie_manager:
                        cookie_args = cookie_manager.get_cookies_args_for_account(cookie_account)
                        cmd.extend(cookie_args)
                    else:
                        cmd.extend(utils.get_cookies_args(self.config))

                    cmd.extend(utils.get_proxy_args(self.config))

                    # Calculate timeout
                    tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
                    base_timeout = tier_timeouts.get('long', 600) if isinstance(tier_timeouts, dict) else 600
                    timeout = max(DEFAULT_SEGMENT_TIMEOUT, int(total_seg_duration * 3))
                    timeout = min(timeout, base_timeout)

                    try:
                        proc_result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

                        if proc_result.returncode == 0:
                            # Success - rename and collect segments
                            downloaded = segment_utils.rename_segments_with_timing(video_dir, video_id, segments)
                            for seg, file_path in zip(segments, downloaded):
                                if file_path and Path(file_path).exists():
                                    result.segments.append(DownloadedSegment(
                                        file=str(file_path),
                                        video_id=video_id,
                                        original_start=seg.start_time,
                                        original_end=seg.end_time,
                                        file_duration=seg.end_time - seg.start_time,
                                        matches=seg.original_matches,
                                        keyword=keyword
                                    ))

                            result.success = True
                            result.download_time = time.time() - download_start

                            if cookie_account and cookie_manager:
                                cookie_manager.report_success_for_account(cookie_account)

                            logger.info(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} [{idx+1}/{len(video_items)}] {video_id} DOWNLOADED ({len(result.segments)}/{len(segments)} seg) in {result.download_time:.1f}s")
                        else:
                            # Check for rate limit
                            if '429' in str(proc_result.stderr) or 'rate' in str(proc_result.stderr).lower():
                                if cookie_account and cookie_manager:
                                    new_cookie = cookie_manager.report_rate_limit_for_account(cookie_account)
                                    if new_cookie:
                                        cookie_account = new_cookie
                                        logger.info(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} rotated to cookie '{cookie_account.label}'")

                            result.error = proc_result.stderr[:100] if proc_result.stderr else "Unknown error"
                            logger.warning(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} [{idx+1}/{len(video_items)}] {video_id} FAILED: {result.error}")

                    except subprocess.TimeoutExpired:
                        result.error = f"Timeout after {timeout}s"
                        logger.warning(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} [{idx+1}/{len(video_items)}] {video_id} TIMEOUT")
                    except Exception as e:
                        result.error = str(e)
                        logger.error(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} [{idx+1}/{len(video_items)}] {video_id} ERROR: {e}")

                    # Apply tier delay after download
                    if delay_after > 0:
                        time.sleep(delay_after)

                    results_queue.put(result)

                logger.info(f"[SEGMENT_DOWNLOAD] Worker-{worker_id} finished")

            # Start workers with staggered launch
            threads = []
            for i in range(parallel_workers):
                cookie = worker_cookies[i] if i < len(worker_cookies) else None
                t = threading.Thread(target=download_worker, args=(i, cookie), daemon=True)
                threads.append(t)
                t.start()
                if i < parallel_workers - 1:
                    time.sleep(parallel_stagger)

            # Wait for all threads
            for t in threads:
                t.join()

            # Collect results
            while not results_queue.empty():
                result = results_queue.get()
                downloaded_segments.extend(result.segments)
                if result.cache_hit:
                    cache_hits += 1
                elif result.success:
                    new_downloads += 1
                    total_download_time += result.download_time
                else:
                    failed_downloads += 1
                tier_counts[result.tier] += 1

            # Log summary for parallel mode
            avg_time = total_download_time / new_downloads if new_downloads > 0 else 0
            logger.info(f"[SEGMENT_DOWNLOAD] === DOWNLOAD COMPLETE (PARALLEL) ===")
            logger.info(f"[SEGMENT_DOWNLOAD] Total segments: {len(downloaded_segments)} from {total_videos} videos")
            logger.info(f"[SEGMENT_DOWNLOAD] Results: cache_hits={cache_hits}, new_downloads={new_downloads}, failed={failed_downloads}")
            logger.info(f"[SEGMENT_DOWNLOAD] Tiers: short={tier_counts['short']}, medium={tier_counts['medium']}, long={tier_counts['long']}")
            logger.info(f"[SEGMENT_DOWNLOAD] Timing: total={total_download_time:.1f}s, avg={avg_time:.1f}s/video")

            if cookie_manager:
                cookie_manager.log_summary()

            return downloaded_segments

        # === SEQUENTIAL MODE ===
        for video_id, segments in by_video.items():
            if not segments:
                continue

            current_video += 1

            first_seg = segments[0]
            video_url = first_seg.video_url
            keyword = first_seg.keyword

            total_seg_duration = sum(seg.end_time - seg.start_time for seg in segments)

            # Get tier-based settings for this video
            tier_name, sleep_interval, concurrent_frags, delay_after = get_tier_settings(total_seg_duration)
            tier_counts[tier_name] += 1

            print(f"  [{current_video}/{total_videos}] {video_id} ({len(segments)} seg, {total_seg_duration:.0f}s) [{tier_name}]")
            logger.info(f"[SEGMENT_DOWNLOAD] [{current_video}/{total_videos}] {video_id} ({len(segments)} seg, {total_seg_duration:.0f}s) tier={tier_name}")

            # Create output directory
            max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
            safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
            safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
            video_dir = output_dir / f"{safe_keyword}_segments"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Check if segments already exist (skip re-download)
            existing_segments = self._check_existing_segments(video_dir, video_id, segments)
            # Count actual existing files (non-None entries)
            existing_count = sum(1 for p in existing_segments if p is not None)
            if existing_count > 0:
                logger.debug(f"Found {existing_count}/{len(segments)} existing segments for {video_id}")
                all_exist = existing_count == len(segments)
                if all_exist:
                    print(f"      [OK] Already downloaded ({existing_count} segments)")
                    logger.info(f"[SEGMENT_DOWNLOAD] [{current_video}/{total_videos}] {video_id} CACHE_HIT ({existing_count} seg)")
                    cache_hits += 1
                    # Add existing segments to results
                    for seg, file_path in zip(segments, existing_segments):
                        if file_path and Path(file_path).exists():
                            file_duration = seg.end_time - seg.start_time
                            downloaded_segments.append(DownloadedSegment(
                                file=str(file_path),
                                video_id=video_id,
                                original_start=seg.start_time,
                                original_end=seg.end_time,
                                file_duration=file_duration,
                                matches=seg.original_matches,
                                keyword=keyword
                            ))
                    continue  # Skip to next video

            # Build --download-sections arguments
            section_args = []
            for seg in segments:
                start_str = utils.format_time(seg.start_time)
                end_str = utils.format_time(seg.end_time)
                section_args.extend(['--download-sections', f'*{start_str}-{end_str}'])

            # Start timing
            download_start_time = time.time()

            # Build yt-dlp command with configurable speed settings (using settings from start)
            cmd = [
                'yt-dlp',
                '--sleep-interval', str(int(sleep_interval)),
                '--concurrent-fragments', str(concurrent_frags),
                video_url,
                *section_args,
                '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                '--merge-output-format', 'mp4',
                '-o', str(video_dir / f'{video_id}_%(autonumber)s.%(ext)s'),
                '--no-playlist',
                '--no-warnings',
            ]

            # Add ffmpeg location
            ffmpeg_loc = getattr(self.download_config, 'ffmpeg_location', '')
            if ffmpeg_loc:
                cmd.extend(['--ffmpeg-location', ffmpeg_loc])

            # Add base args (JS runtime for challenge solving) and cookies
            cmd.extend(utils.get_ytdlp_base_args())
            cmd.extend(utils.get_cookies_args(self.config))

            # Store base command (without proxy) for fallback
            base_cmd = cmd.copy()

            # Add proxy args - will be removed on proxy failure
            proxy_args = utils.get_proxy_args(self.config)
            cmd.extend(proxy_args)
            use_proxy = bool(proxy_args)

            # Get timeout - use segment-specific timeout (shorter than full video)
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            base_timeout = tier_timeouts.get('long', 600)
            # Scale timeout based on total segment duration, with minimum of DEFAULT_SEGMENT_TIMEOUT
            timeout = max(DEFAULT_SEGMENT_TIMEOUT, int(total_seg_duration * 3))
            timeout = min(timeout, base_timeout)  # Cap at configured max

            # Retry configuration
            max_retries = getattr(self.download_config, 'max_retries', DEFAULT_MAX_RETRIES)
            retry_delay = getattr(self.download_config, 'retry_delay', DEFAULT_RETRY_DELAY)

            segment_success = False
            last_error = None

            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"      ↻ Retry {attempt}/{max_retries-1} after {retry_delay}s...")
                    logger.info(f"Retrying {video_id} (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    # Exponential backoff for subsequent retries
                    retry_delay = min(retry_delay * 2, 60)

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )

                    if result.returncode != 0:
                        last_error = result.stderr[-200:] if result.stderr else 'Unknown error'
                        # Check if error is retryable (network issues, rate limiting)
                        if self._is_retryable_error(result.stderr):
                            # Check if this is a SOCKS proxy error - fall back to direct
                            if use_proxy and ('SOCKS' in str(result.stderr) or 'WinError 10061' in str(result.stderr)):
                                logger.warning(f"Proxy connection failed for {video_id}, falling back to direct connection")
                                print(f"      → Proxy unavailable, switching to direct...")
                                cmd = base_cmd  # Use command without proxy
                                use_proxy = False
                            logger.warning(f"Retryable error for {video_id}: {last_error}")
                            continue
                        else:
                            print(f"      [X] Failed: {last_error}")
                            logger.warning(f"Segment download failed for {video_id}: {result.stderr[:200]}")
                            break
                    else:
                        segment_success = True
                        # Rename files from autonumber to timestamp-based names
                        downloaded = segment_utils.rename_segments_with_timing(video_dir, video_id, segments)

                        success_count = 0
                        for seg, file_path in zip(segments, downloaded):
                            if file_path and Path(file_path).exists():
                                success_count += 1
                                file_duration = seg.end_time - seg.start_time

                                downloaded_segments.append(DownloadedSegment(
                                    file=str(file_path),
                                    video_id=video_id,
                                    original_start=seg.start_time,
                                    original_end=seg.end_time,
                                    file_duration=file_duration,
                                    matches=seg.original_matches,
                                    keyword=keyword
                                ))

                        # Calculate timing
                        download_elapsed = time.time() - download_start_time
                        total_download_time += download_elapsed
                        total_segments_downloaded += success_count

                        print(f"      [OK] Downloaded {success_count}/{len(segments)} segments in {download_elapsed:.1f}s")
                        logger.info(f"[SEGMENT_DOWNLOAD] [{current_video}/{total_videos}] {video_id} DOWNLOADED ({success_count}/{len(segments)} seg) in {download_elapsed:.1f}s")
                        new_downloads += 1
                        break  # Success, exit retry loop

                except subprocess.TimeoutExpired:
                    last_error = f"Timeout after {timeout}s"
                    print(f"      [X] {last_error} (attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"Segment download timeout for {video_id}")
                    # Timeout is retryable
                    continue
                except Exception as e:
                    last_error = str(e)
                    print(f"      [X] Error: {e}")
                    logger.error(f"Segment download error for {video_id}: {e}")
                    break  # Non-retryable error

            # Log final failure if all retries exhausted
            if not segment_success and last_error:
                logger.error(f"All {max_retries} attempts failed for {video_id}: {last_error}")
                logger.info(f"[SEGMENT_DOWNLOAD] [{current_video}/{total_videos}] {video_id} FAILED: {last_error[:50]}")
                failed_downloads += 1

            # Fallback to full video if segment download failed
            fallback_segments = []  # Initialize for delay_after check
            if not segment_success and fallback_full:
                print(f"      → Falling back to full video download...")
                logger.info(f"[SEGMENT_DOWNLOAD] [{current_video}/{total_videos}] {video_id} FALLBACK to full video")
                fallback_segments = self._download_full_video_fallback(
                    video_id=video_id,
                    video_url=video_url,
                    video_dir=video_dir,
                    segments=segments,
                    keyword=keyword,
                    timeout=timeout
                )
                downloaded_segments.extend(fallback_segments)
                if fallback_segments:
                    failed_downloads -= 1  # Fallback succeeded, remove from failed count
                    new_downloads += 1
                    logger.info(f"[SEGMENT_DOWNLOAD] [{current_video}/{total_videos}] {video_id} FALLBACK_OK ({len(fallback_segments)} seg)")

            # Apply tier-based delay after download (prevents rate limiting for short videos)
            if delay_after > 0 and (segment_success or fallback_segments):
                logger.debug(f"[SEGMENT_DOWNLOAD] Applying {delay_after}s delay after {tier_name} video")
                time.sleep(delay_after)

            # Periodic checkpoint save to allow resume if interrupted
            if progress_callback and current_video % checkpoint_interval == 0:
                logger.info(f"Checkpoint save at video {current_video}/{total_videos}")
                try:
                    progress_callback(current_video, total_videos, downloaded_segments)
                except Exception as e:
                    logger.warning(f"Checkpoint callback failed: {e}")

            # Periodic progress summary (every 25 videos)
            if current_video % 25 == 0:
                logger.info(f"[SEGMENT_DOWNLOAD] Progress: {current_video}/{total_videos} (cache={cache_hits}, new={new_downloads}, fail={failed_downloads}) tiers: S={tier_counts['short']}, M={tier_counts['medium']}, L={tier_counts['long']}")

        # Final progress callback
        if progress_callback:
            try:
                progress_callback(total_videos, total_videos, downloaded_segments)
            except Exception as e:
                logger.warning(f"Final checkpoint callback failed: {e}")

        # Log comprehensive summary
        avg_time = total_download_time / new_downloads if new_downloads > 0 else 0
        logger.info(f"[SEGMENT_DOWNLOAD] === DOWNLOAD COMPLETE ===")
        logger.info(f"[SEGMENT_DOWNLOAD] Total segments: {len(downloaded_segments)} from {total_videos} videos")
        logger.info(f"[SEGMENT_DOWNLOAD] Results: cache_hits={cache_hits}, new_downloads={new_downloads}, failed={failed_downloads}")
        logger.info(f"[SEGMENT_DOWNLOAD] Tiers: short={tier_counts['short']}, medium={tier_counts['medium']}, long={tier_counts['long']}")
        logger.info(f"[SEGMENT_DOWNLOAD] Timing: total={total_download_time:.1f}s, avg={avg_time:.1f}s/video")
        return downloaded_segments

    def _download_full_video_fallback(
        self,
        video_id: str,
        video_url: str,
        video_dir: Path,
        segments: List[MergedSegment],
        keyword: str,
        timeout: int = 600
    ) -> List[DownloadedSegment]:
        """
        Download full video as fallback when segment download fails.

        Migrated from downloader.py lines 2458-2540.

        Args:
            video_id: YouTube video ID
            video_url: Full YouTube URL
            video_dir: Output directory
            segments: Original segments (for match info)
            keyword: Source keyword
            timeout: Download timeout in seconds

        Returns:
            List with single DownloadedSegment covering full video
        """
        logger.info(f"  Downloading full video fallback: {video_id}")

        output_file = video_dir / f"{video_id}_0000.mp4"

        # Get speed settings from caption_first config
        caption_config = getattr(self.download_config, 'caption_first', None)
        sleep_interval = getattr(caption_config, 'segment_sleep_interval', 1.0) if caption_config else 1.0
        concurrent_frags = getattr(caption_config, 'segment_concurrent_fragments', 4) if caption_config else 4

        cmd = [
            'yt-dlp',
            '--sleep-interval', str(int(sleep_interval)),
            '--concurrent-fragments', str(concurrent_frags),
            video_url,
            '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            '--merge-output-format', 'mp4',
            '-o', str(output_file),
            '--no-playlist',
            '--no-warnings',
        ]

        # Add ffmpeg location
        ffmpeg_loc = getattr(self.download_config, 'ffmpeg_location', '')
        if ffmpeg_loc:
            cmd.extend(['--ffmpeg-location', ffmpeg_loc])

        # Add base args (JS runtime for challenge solving) and cookies
        cmd.extend(utils.get_ytdlp_base_args())
        cmd.extend(utils.get_cookies_args(self.config))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0 and output_file.exists():
                # Get video duration
                video_duration = self._get_video_duration(output_file)
                if video_duration is None:
                    # Estimate from segments
                    video_duration = max(seg.end_time for seg in segments) + 60

                # Collect all original matches
                all_matches = []
                for seg in segments:
                    all_matches.extend(seg.original_matches)

                logger.info(f"  [OK] Full video fallback success: {video_id}")
                return [DownloadedSegment(
                    file=str(output_file),
                    video_id=video_id,
                    original_start=0,
                    original_end=video_duration,
                    file_duration=video_duration,
                    matches=all_matches,
                    keyword=keyword
                )]
            else:
                logger.error(f"Full video fallback failed for {video_id}: {result.stderr[:200]}")
                return []

        except subprocess.TimeoutExpired:
            logger.error(f"Full video fallback timeout for {video_id}")
            return []
        except Exception as e:
            logger.error(f"Full video fallback error for {video_id}: {e}")
            return []

    def _check_existing_segments(
        self,
        video_dir: Path,
        video_id: str,
        segments: List['MergedSegment']
    ) -> List[Optional[str]]:
        """
        Check if video segments already exist on disk.

        Looks for files matching the expected naming pattern from segment_utils.
        Returns list of existing file paths (None for missing segments).

        Args:
            video_dir: Directory where segments are stored
            video_id: YouTube video ID
            segments: List of segments to check

        Returns:
            List of file paths (or None) for each segment
        """
        from . import segment_utils

        existing = []
        for seg in segments:
            # Expected filename pattern: {video_id}_{start_seconds:04d}.mp4
            # Using same logic as segment_utils.get_segment_filename
            start_int = int(seg.start_time)
            expected_base = f"{video_id}_{start_int:04d}"

            # Check for file with any video extension
            found = None
            for ext in ['.mp4', '.mkv', '.webm', '.m4v']:
                candidate = video_dir / f"{expected_base}{ext}"
                if candidate.exists():
                    found = str(candidate)
                    break

            existing.append(found)

        return existing

    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """
        Get video duration using ffprobe.

        Migrated from downloader.py lines 2542-2556.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds or None
        """
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def _is_retryable_error(self, error_text: str) -> bool:
        """
        Check if an error is retryable (transient network/rate limiting issues).

        Args:
            error_text: Error message from yt-dlp stderr

        Returns:
            True if the error is likely transient and worth retrying
        """
        if not error_text:
            return False

        error_lower = error_text.lower()

        # Network-related errors (retryable)
        network_errors = [
            'connection reset',
            'connection refused',
            'connection timed out',
            'timeout',
            'network unreachable',
            'temporary failure',
            'name resolution',
            'dns',
            'ssl',
            'certificate',
            'read timed out',
            'socket',
            'broken pipe',
            'connection aborted',
            'incomplete read',
        ]

        # Rate limiting errors (retryable with delay)
        rate_limit_errors = [
            'rate limit',
            'too many requests',
            '429',
            'quota exceeded',
            'throttl',
            'please try again',
            'temporary',
        ]

        # Server-side errors (potentially retryable)
        server_errors = [
            '500',
            '502',
            '503',
            '504',
            'internal server error',
            'bad gateway',
            'service unavailable',
            'gateway timeout',
        ]

        # Check for retryable patterns
        for pattern in network_errors + rate_limit_errors + server_errors:
            if pattern in error_lower:
                return True

        # Non-retryable errors (video unavailable, geo-blocked, etc.)
        non_retryable = [
            'video unavailable',
            'private video',
            'removed',
            'deleted',
            'copyright',
            'blocked',
            'not available',
            'age-restricted',
            'sign in',
            'members only',
            'premiere',
        ]

        for pattern in non_retryable:
            if pattern in error_lower:
                return False

        # Default: retry unknown errors once
        return True
