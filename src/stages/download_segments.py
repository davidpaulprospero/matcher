"""
Download Segments Stage - Download matched video segments

Stage 6 of the simplified 7-stage pipeline:
- Downloads only the matched video segments (not full videos)
- Runs after MATCH and ITERATIVE_MATCH stages
- Efficient: only downloads portions of videos that are actually used
- US-48-005: Integrates 4-tier escalation (impersonation, extractor_args,
  cookie rotation) via EscalationManager for resilient downloading
"""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageMetrics, StageResult, register_stage, validate_required_state_attrs
from .error_aggregator import ErrorAggregator

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo

logger = logging.getLogger(__name__)

# US-52-006: Error classification delegated to shared module
from ..downloader.error_classification import (
    is_network_failure as _is_network_failure,
    is_escalation_error as _is_escalation_error,
    classify_error_category,
    NETWORK_FAILURE_THRESHOLD,
    BOT_DETECTION_ABORT_THRESHOLD,
)


@dataclass
class _DownloadLoopContext:
    """Internal shared state for the download loop sub-methods.

    US-57-007: Holds references that the decomposed loop methods need,
    avoiding long parameter lists while keeping each method focused.
    """

    stats: 'SegmentDownloadStats'
    escalation_mgr: Any = None
    cookie_rotator: Any = None
    circuit_breaker: Any = None
    download_config: Any = None
    bot_floor_threshold: int = 5
    bot_abort_threshold: int = 0
    network_failure_threshold: int = 0
    consecutive_network_failures: int = 0
    consecutive_bot_detections: int = 0


@dataclass
class SegmentDownloadStats:
    """Typed container for segment download statistics.

    Replaces the previously untyped ``stats`` dict in ``_download_segments()``,
    providing IDE autocompletion, typo-safe attribute access, and explicit
    defaults for every field.
    """

    succeeded: int = 0
    failed: int = 0
    cached: int = 0
    attempted: int = 0
    total: int = 0
    retry_count: int = 0
    segment_durations: List[float] = field(default_factory=list)
    total_bytes: int = 0
    error_categories: Dict[str, int] = field(default_factory=dict)
    error_aggregator: ErrorAggregator = field(default_factory=ErrorAggregator)
    progress_hooks_data: Dict[str, int] = field(default_factory=lambda: {
        'total_downloaded_bytes': 0,
        'segments_with_progress': 0,
        'segments_finished': 0,
    })
    download_speeds: List[float] = field(default_factory=list)
    # US-81-002: Per-item error tracking for batch error isolation
    failed_items: List[Dict[str, Any]] = field(default_factory=list)

    # -- convenience mutators --------------------------------------------------

    def increment_success(self, duration: float = 0.0, file_bytes: int = 0) -> None:
        """Record a successful segment download."""
        self.succeeded += 1
        self.attempted += 1
        if duration:
            self.segment_durations.append(duration)
        if file_bytes:
            self.total_bytes += file_bytes
        if duration > 0 and file_bytes > 0:
            self.download_speeds.append(file_bytes / duration)

    def increment_failure(
        self,
        category: str | None = None,
        error_msg: str | None = None,
        video_id: str | None = None,
    ) -> None:
        """Record a failed segment download, optionally categorised.

        US-81-002: Also appends structured error info to failed_items list.
        """
        self.failed += 1
        self.attempted += 1
        if category:
            self.error_categories[category] = self.error_categories.get(category, 0) + 1
        if error_msg and category:
            self.error_aggregator.record(error_msg, category)
        # US-81-002: Track per-item failure details
        if video_id:
            self.failed_items.append({
                'video_id': video_id,
                'error_type': category or 'unknown',
                'message': error_msg or 'unknown error',
            })

    def increment_cached(self, file_bytes: int = 0) -> None:
        """Record a segment served from cache."""
        self.cached += 1
        self.attempted += 1
        if file_bytes:
            self.total_bytes += file_bytes

    def average_speed(self) -> float:
        """Return mean download speed in bytes/second across all segments.

        Returns 0.0 if no speed measurements have been recorded.
        """
        if not self.download_speeds:
            return 0.0
        return sum(self.download_speeds) / len(self.download_speeds)

    def is_speed_degrading(self, window: int = 5, threshold: float = 0.5) -> bool:
        """Detect sustained speed degradation over recent downloads.

        Returns True if the last *window* speeds are all below *threshold*
        of the overall average speed.  This indicates gradual throttling
        rather than a single slow download.

        Args:
            window: Number of most-recent speeds to inspect.
            threshold: Fraction of overall average below which a speed is
                       considered degraded (0.5 = 50 %).

        Returns:
            True if the tail window is consistently slow.
        """
        if len(self.download_speeds) < window:
            return False
        overall_avg = self.average_speed()
        if overall_avg <= 0:
            return False
        cutoff = overall_avg * threshold
        tail = self.download_speeds[-window:]
        return all(s < cutoff for s in tail)


def _apply_escalation_to_ydl_opts(ydl_opts: Dict[str, Any], escalation_result) -> None:
    """Translate EscalationResult CLI args to yt-dlp Python API ydl_opts.

    The EscalationManager returns CLI args (e.g., ['--impersonate', 'X',
    '--extractor-args', 'youtube:player_client=a,b']). This function
    translates them to ydl_opts dict keys for the Python API.

    Translation:
        --impersonate X           → ydl_opts['impersonate'] = 'X'
        --extractor-args youtube:player_client=X  → ydl_opts['extractor_args'] = {'youtube': {'player_client': 'X'}}

    Cookie rotation is handled separately via cookiefile, not via CLI args.

    Args:
        ydl_opts: The yt-dlp options dict to modify in-place.
        escalation_result: EscalationResult from EscalationManager.get_escalation_args().
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
                # Lowercase client/os fields - available targets are lowercase
                # but from_str() preserves original case
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
                # Fall back to no impersonation rather than crashing
            i += 2
        elif args[i] == '--extractor-args' and i + 1 < len(args):
            # Skip escalation's extractor_args — alternative player_clients
            # (web_safari, tv_downgraded, web) can trigger YouTube SABR
            # (Server-side Adaptive Bitrate) which returns no downloadable
            # format URLs, causing "No video formats found!".
            # We override with 'tv' client below instead.
            logger.debug(
                "Skipping escalation extractor_args for segment download: %s",
                args[i + 1],
            )
            i += 2
        else:
            i += 1


@register_stage
class DownloadVideoSegmentsStage(Stage):
    """
    Downloads video segments after matching.

    In the simplified 7-stage pipeline, this stage downloads only the
    portions of YouTube videos that were matched to voiceover segments.

    Inputs:
        - state.matches: List of Match objects with video IDs and time ranges
        - state.video_ids: List of video IDs from VIDEO_SEARCH stage

    Outputs:
        - state.downloaded_segments: List of DownloadedVideo with local file paths
        - Updated state.matches with local file paths
    """

    name = "DOWNLOAD_SEGMENTS"
    description = "Download matched video segments"

    # Browsers that yt-dlp supports for cookie extraction
    _KNOWN_BROWSERS = ('firefox', 'chrome', 'edge', 'safari', 'opera', 'brave')

    def __init__(self):
        self.downloader = None

    @staticmethod
    def _validate_cookie_config(download_config) -> None:
        """US-50-012 / US-52-009: Validate cookie configuration at stage init.

        Emits warnings when:
        - No cookie source is configured (neither cookies_from_browser,
          cookies_path, nor cookie_rotation with files)
        - cookies_from_browser is set to a browser that isn't installed

        These warnings appear once at stage startup, not per-download.
        """
        cookies_from_browser = getattr(download_config, 'cookies_from_browser', '')
        cookies_path = getattr(download_config, 'cookies_path', '')

        # US-52-009: Also check cookie_rotation as a valid cookie source
        cookie_rotation = getattr(download_config, 'cookie_rotation', None)
        has_cookie_rotation = False
        if cookie_rotation is not None:
            rotation_enabled = getattr(cookie_rotation, 'enabled', False)
            rotation_files = getattr(cookie_rotation, 'cookie_files', [])
            has_cookie_rotation = rotation_enabled and bool(rotation_files)

        if not cookies_from_browser and not cookies_path and not has_cookie_rotation:
            logger.warning(
                "No cookie source configured. YouTube will likely block all download "
                "requests with 403/bot-detection errors. "
                "Set download.cookies_from_browser to your browser name "
                "(firefox, chrome, edge, safari, opera, brave) in config.yaml"
            )
            return

        if cookies_from_browser:
            # Best-effort check: see if the browser executable is on PATH
            browser_exe = cookies_from_browser.lower()
            # Map browser names to common executable names
            _exe_map = {
                'firefox': 'firefox',
                'chrome': 'google-chrome' if shutil.which('google-chrome') else 'chrome',
                'edge': 'msedge',
                'safari': 'safari',
                'opera': 'opera',
                'brave': 'brave',
            }
            exe_name = _exe_map.get(browser_exe, browser_exe)
            if not shutil.which(exe_name) and not shutil.which(browser_exe):
                logger.warning(
                    f"cookies_from_browser is set to '{cookies_from_browser}' but "
                    f"'{cookies_from_browser}' does not appear to be installed "
                    f"(not found on PATH). Cookie extraction may fail. "
                    f"Verify the browser is installed or use a cookies_path file instead."
                )

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video segment download stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['matches'], self.name)

        warnings = []

        try:
            if not state.matches:
                print("  >> No matches to download")
                return StageResult.ok({'skipped': True, 'reason': 'no_matches'}, warnings)

            print(f"\n  --- Stage 6: DOWNLOAD VIDEO SEGMENTS ---")

            # US-59-011: Check caption_batch_low_yield flag from CAPTION stage
            # If set, many videos will need transcription after download
            caption_batch_low_yield = getattr(state, 'caption_batch_low_yield', False)
            if caption_batch_low_yield:
                logger.warning(
                    "US-59-011: caption_batch_low_yield=True - many videos in the batch had no "
                    "captions available. These videos will require transcription after download."
                )
                print("  ! Warning: Many videos have no captions - transcription will be needed")

            # Get download settings from config
            download_config = config.download
            buffer_seconds = download_config.segment_buffer

            # US-50-012: Validate cookie configuration early (before download loop)
            self._validate_cookie_config(download_config)

            print(f"  Downloading matched segments")
            print(f"    Buffer: {buffer_seconds}s before/after each match")

            # Collect segments to download (US-48-008: merge overlapping/adjacent)
            segments_to_download = self._collect_matched_segments(state, buffer_seconds)

            if not segments_to_download:
                print("  ! No valid segments to download")
                return StageResult.ok({'skipped': True, 'reason': 'no_valid_segments'}, warnings)

            print(f"    Total segments: {len(segments_to_download)}")

            # Initialize downloader
            from ..downloader import VideoDownloader
            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            # US-51-010: Restore retry queue from checkpoint on resume
            restored_rq_data = getattr(state, '_restored_retry_queue', None)
            if restored_rq_data and self.downloader.retry_queue:
                self.downloader.retry_queue.from_checkpoint_dict(restored_rq_data)
                pending_count = len(self.downloader.retry_queue.items)
                failed_count = len(self.downloader.retry_queue._failed_ids)
                if pending_count > 0:
                    logger.info(
                        f"Restored retry queue: {pending_count} videos to retry, "
                        f"{failed_count} permanently skipped"
                    )
                    print(f"    Retry queue: {pending_count} previously-failed videos to retry first")
                # Clean up temporary state attribute
                delattr(state, '_restored_retry_queue')

            # US-81-003: Shared mutable container for partial_progress tracking
            # Updated by _download_segments and read by checkpoint_progress
            _partial_progress = {
                'completed_ids': [],
                'failed_ids': [],
                'total_count': len(segments_to_download),
            }

            # US-81-003: Configurable checkpoint I/O frequency
            _checkpoint_every_n = int(getattr(
                download_config, 'segment_checkpoint_every_n', 10
            ))
            if _checkpoint_every_n <= 0:
                _checkpoint_every_n = 1
            _checkpoint_counter = [0]  # mutable for closure

            # Download segments with progress callback for checkpointing
            def checkpoint_progress(current: int, total: int, downloaded: list):
                """Save progress checkpoint during download.

                US-81-003: Includes partial_progress dict with completed_ids,
                failed_ids, total_count for partial resume on interruption.
                Only writes to disk every N calls to reduce I/O overhead.
                Always writes on abort (current < total check in caller).
                """
                _checkpoint_counter[0] += 1
                # Write every N items, or on final item, or when aborting
                is_final = (current >= total)
                if _checkpoint_counter[0] < _checkpoint_every_n and not is_final:
                    return
                _checkpoint_counter[0] = 0

                checkpoint_data = {
                    'segment_count': len(downloaded),
                    'segments_completed': current,
                    'segments_total': total,
                    'in_progress': current < total,
                    # US-81-003: Partial progress for resume (deep copy lists)
                    'partial_progress': {
                        'completed_ids': list(_partial_progress['completed_ids']),
                        'failed_ids': list(_partial_progress['failed_ids']),
                        'total_count': _partial_progress['total_count'],
                    },
                }
                # US-51-010: Include retry queue state in intermediate checkpoints
                if self.downloader and self.downloader.retry_queue:
                    rq = self.downloader.retry_queue
                    if rq.items or rq._failed_ids:
                        checkpoint_data['retry_queue'] = rq.to_checkpoint_dict()
                checkpoint.save_intermediate('DOWNLOAD_SEGMENTS', checkpoint_data)

            stage_start_time = time.time()

            # US-51-010: Process restored retry queue BEFORE new segments
            pre_retry_downloaded = []
            if (restored_rq_data and self.downloader and self.downloader.retry_queue
                    and self.downloader.retry_queue.has_pending()):
                logger.info("Processing restored retry queue before new segments")
                self._process_retry_queue(
                    output_dir, buffer_seconds, pre_retry_downloaded,
                    len(segments_to_download), checkpoint_progress
                )
                if pre_retry_downloaded:
                    logger.info(
                        f"Restored retry queue: {len(pre_retry_downloaded)} "
                        f"videos recovered from previous session"
                    )

            downloaded_segments, download_stats = self._download_segments(
                segments_to_download,
                output_dir,
                buffer_seconds,
                checkpoint_progress,
                partial_progress=_partial_progress,
            )

            # US-51-010: Merge pre-retry downloads into main list
            if pre_retry_downloaded:
                downloaded_segments.extend(pre_retry_downloaded)

            elapsed = time.time() - stage_start_time

            # Print end-of-stage summary
            self._print_summary(download_stats, elapsed)

            # Update matches to reference local files
            self._update_matches_with_local_paths(state, downloaded_segments)

            # Store downloaded segments in state
            state.downloaded_segments = downloaded_segments

            checkpoint_data = {
                'segment_count': len(downloaded_segments),
                'total_matches': len(state.matches),
                'retry_count': download_stats.retry_count,
                # US-81-002: Per-item error details for batch error isolation
                'failed_items': download_stats.failed_items,
                # US-81-003: Clear partial_progress on full completion
                'partial_progress': None,
            }

            # US-50-008: Include circuit breaker metrics in checkpoint
            if self.downloader:
                _cb = getattr(self.downloader, 'circuit_breaker', None)
                if _cb:
                    checkpoint_data['circuit_breaker'] = {
                        'total_trips': _cb.state.total_trips,
                        'total_paused_seconds': round(_cb.state.total_paused_seconds, 1),
                    }

            # US-49-012: Collect escalation summary from escalation manager
            escalation_summary = {}
            if self.downloader:
                _esc_mgr = getattr(self.downloader, 'escalation_manager', None)
                if _esc_mgr and hasattr(_esc_mgr, 'get_metrics'):
                    try:
                        esc_metrics = _esc_mgr.get_metrics()
                        # Build serializable summary: videos per tier, totals, effectiveness
                        escalation_summary = {
                            'total_escalations': esc_metrics.get('total_escalations', 0),
                            'videos_per_tier': {
                                tier: len(keywords)
                                for tier, keywords in esc_metrics.get('keywords_at_each_tier', {}).items()
                            },
                            'escalations_per_tier': esc_metrics.get('escalations_per_tier', {}),
                            'total_403s': esc_metrics.get('total_403s', 0),
                            'total_successes': esc_metrics.get('total_successes', 0),
                            'average_tier': esc_metrics.get('average_tier', 1.0),
                        }
                        # Include tier effectiveness if available
                        if hasattr(_esc_mgr, 'get_tier_effectiveness'):
                            escalation_summary['tier_effectiveness'] = _esc_mgr.get_tier_effectiveness()
                    except Exception as esc_err:
                        logger.debug(f"Could not collect escalation summary: {esc_err}")

            # US-52-011: Add tier_distribution from escalation_mgr.keyword_states
            # Shows count of downloads at each escalation tier for diagnosing
            # whether YouTube is broadly blocking (all Tier 3+) vs isolated failures.
            if self.downloader:
                _esc_mgr_td = getattr(self.downloader, 'escalation_manager', None)
                if _esc_mgr_td and hasattr(_esc_mgr_td, 'keyword_states'):
                    try:
                        tier_distribution: Dict[str, int] = {}
                        for _kw, _state in _esc_mgr_td.keyword_states.items():
                            tier_name = _state.current_tier.name
                            tier_distribution[tier_name] = tier_distribution.get(tier_name, 0) + 1
                        escalation_summary['tier_distribution'] = tier_distribution
                    except Exception as td_err:
                        logger.debug(f"Could not collect tier distribution: {td_err}")

            # US-50-009: Add bot_detection_count and network_failure_count from error categories
            escalation_summary['bot_detection_count'] = download_stats.error_categories.get('bot_detection', 0)
            escalation_summary['network_failure_count'] = download_stats.error_categories.get('network', 0)

            # US-50-009: Log structured escalation summary at stage completion
            self._log_escalation_summary(escalation_summary)

            # US-51-006: Include progress hook data in escalation_summary for checkpoint
            if download_stats.progress_hooks_data:
                escalation_summary['download_progress'] = download_stats.progress_hooks_data

            # US-51-010: Persist retry queue to checkpoint for resume
            if self.downloader and self.downloader.retry_queue:
                rq = self.downloader.retry_queue
                if rq.items or rq._failed_ids:
                    checkpoint_data['retry_queue'] = rq.to_checkpoint_dict()

            # Stage metrics for pipeline observability (US-49-009 + US-49-012)
            metrics = StageMetrics(
                items_processed=download_stats.succeeded + download_stats.cached,
                items_failed=download_stats.failed,
                duration_seconds=elapsed,
                error_categories=download_stats.error_categories,
                escalation_summary=escalation_summary,
            )

            return StageResult.ok(checkpoint_data, warnings, metrics)

        except Exception as e:
            logger.exception(f"Video segment download failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _collect_matched_segments(
        self, state: 'PipelineState', buffer_seconds: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Collect segment info from matches for downloading.

        US-48-008: Uses exact float values for dedup keys (not round()) to
        preserve precision for segments differing by <0.5s. Also merges
        overlapping/adjacent segments from the same video to reduce downloads.
        """
        raw_segments = []

        for match in state.matches:
            # Handle MatchResult structure (has primary_match)
            if hasattr(match, 'primary_match') and match.primary_match:
                pm = match.primary_match
                if hasattr(pm, 'video_segment') and pm.video_segment:
                    video_id = getattr(pm.video_segment, 'source_file', '')
                    start_time = getattr(pm.video_segment, 'start_time', 0.0)
                    end_time = getattr(pm.video_segment, 'end_time', start_time + 10.0)
                else:
                    continue
            # Handle plain Match structure
            elif hasattr(match, 'video_file'):
                video_id = match.video_file
                start_time = getattr(match, 'video_start', 0.0)
                end_time = getattr(match, 'video_end', start_time + 10.0)
            else:
                continue

            # Skip if no video ID
            if not video_id:
                continue

            raw_segments.append({
                'video_id': video_id,
                'start': start_time,
                'end': end_time,
            })

        # Deduplicate exact matches using (video_id, start, end) tuple
        seen = set()
        deduped = []
        for seg in raw_segments:
            key = (seg['video_id'], seg['start'], seg['end'])
            if key not in seen:
                seen.add(key)
                deduped.append(seg)

        # Merge overlapping/adjacent segments from the same video
        return self._merge_segments(deduped, buffer_seconds)

    @staticmethod
    def _merge_segments(
        segments: List[Dict[str, Any]], buffer_seconds: float
    ) -> List[Dict[str, Any]]:
        """Merge overlapping or adjacent segments from the same video.

        Two segments from the same video are merged if they overlap or
        the gap between them is less than 2 * buffer_seconds (since both
        would have buffer applied, their downloaded ranges would overlap).

        Args:
            segments: Deduplicated segment list.
            buffer_seconds: Per-segment buffer (used to compute merge threshold).

        Returns:
            Merged segment list.
        """
        if not segments:
            return []

        # Group by video_id
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for seg in segments:
            by_video.setdefault(seg['video_id'], []).append(seg)

        merged = []
        merge_gap = 2 * buffer_seconds

        for video_id, segs in by_video.items():
            # Sort by start time
            segs.sort(key=lambda s: s['start'])

            current = dict(segs[0])  # copy first segment
            for seg in segs[1:]:
                # Merge if overlapping or gap < 2*buffer
                if seg['start'] <= current['end'] + merge_gap:
                    current['end'] = max(current['end'], seg['end'])
                else:
                    merged.append(current)
                    current = dict(seg)
            merged.append(current)

        return merged

    def _download_segments(
        self,
        segments: List[Dict[str, Any]],
        output_dir: Path,
        buffer_seconds: float,
        progress_callback,
        partial_progress: Optional[Dict[str, Any]] = None,
    ):
        """Download video segments via sub-methods: prepare, check, execute, handle.

        US-57-007: Returns (downloaded_segments list, SegmentDownloadStats).
        US-81-003: Tracks partial_progress (completed_ids, failed_ids, total_count)
        and saves incremental checkpoint every N downloads for partial resume.

        Args:
            partial_progress: Mutable dict shared with checkpoint callback.
                Updated in-place with completed_ids, failed_ids, total_count.
        """
        from ..state import DownloadedVideo

        downloaded = []
        total = len(segments)
        stats = SegmentDownloadStats(total=total)
        ctx = self._prepare_download_context(stats)

        # US-81-003: Track completed/failed IDs via shared partial_progress dict
        if partial_progress is None:
            partial_progress = {'completed_ids': [], 'failed_ids': [], 'total_count': total}

        # Adaptive request delay to avoid YouTube rate-limiting
        dl_cfg = ctx.download_config
        base_delay = float(getattr(dl_cfg, 'segment_request_delay', 1.0)) if dl_cfg else 1.0
        max_delay = float(getattr(dl_cfg, 'segment_request_delay_max', 30.0)) if dl_cfg else 30.0
        current_delay = base_delay
        _did_network_request = False

        for idx, seg in enumerate(segments, 1):
            # Sleep between network requests (skip before first, skip after cache hits)
            if _did_network_request and current_delay > 0:
                time.sleep(current_delay)
            _did_network_request = False

            video_id = seg['video_id']
            start = max(0, seg['start'] - buffer_seconds)
            end = seg['end'] + buffer_seconds
            seg_key = f"{video_id}_{int(start)}_{int(end)}"
            output_file = output_dir / f"{seg_key}.mp4"

            # Check cache hit
            if output_file.exists():
                logger.info(f"Segment already exists: {output_file}")
                downloaded.append(DownloadedVideo(
                    file=str(output_file),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    source='segment_cache'
                ))
                try:
                    _cached_bytes = output_file.stat().st_size
                except OSError:
                    _cached_bytes = 0
                stats.increment_cached(file_bytes=_cached_bytes)
                partial_progress['completed_ids'].append(seg_key)
                ctx.consecutive_network_failures = 0
                if ctx.consecutive_bot_detections > 0:
                    ctx.consecutive_bot_detections = 0
                    if ctx.escalation_mgr:
                        ctx.escalation_mgr.clear_tier_floor()
                self._print_progress(idx, total, stats)
                if progress_callback:
                    progress_callback(idx, total, downloaded)
                continue

            # Check preconditions (circuit breaker, etc.)
            skip_reason = self._check_preconditions(ctx, video_id, start, end, output_file)
            if skip_reason:
                stats.increment_failure(
                    category='precondition',
                    error_msg=skip_reason,
                    video_id=video_id,
                )
                partial_progress['failed_ids'].append(seg_key)
                self._print_progress(idx, total, stats)
                if progress_callback:
                    progress_callback(idx, total, downloaded)
                continue

            # Execute the download
            _did_network_request = True
            result = self._execute_download(ctx, video_id, start, end, output_file)

            # Handle the result (success, failure, abort signals)
            abort = self._handle_result(
                ctx, result, video_id, start, end, output_file,
                downloaded, idx, total, progress_callback,
            )
            # US-81-003: Track outcome
            if result.get('success'):
                partial_progress['completed_ids'].append(seg_key)
            else:
                partial_progress['failed_ids'].append(seg_key)
            self._print_progress(idx, total, stats)

            # Adaptive delay: back off on failure, reset on success
            if result.get('success'):
                current_delay = base_delay
            else:
                current_delay = min(current_delay * 2, max_delay)

            if abort:
                break

            # Checkpoint progress
            if progress_callback:
                progress_callback(idx, total, downloaded)

        # Process retry queue if there are pending items
        self._process_retry_queue(output_dir, buffer_seconds, downloaded, total, progress_callback, stats)

        # US-49-009: Log structured error summary with actionable diagnostics
        self._log_error_summary(stats)

        return downloaded, stats

    def _prepare_download_context(
        self, stats: SegmentDownloadStats
    ) -> _DownloadLoopContext:
        """Initialise shared loop state from downloader and config.

        US-57-007: Extracts the one-time setup that previously sat at the
        top of _download_segments into its own method.

        Returns:
            A populated _DownloadLoopContext.
        """
        escalation_mgr = None
        cookie_rotator = None
        circuit_breaker = None
        if self.downloader:
            escalation_mgr = getattr(self.downloader, 'escalation_manager', None)
            cookie_rotator = getattr(self.downloader, 'cookie_rotator', None)
            circuit_breaker = getattr(self.downloader, 'circuit_breaker', None)

            # US-49-007: Wire circuit breaker into escalation manager
            if escalation_mgr and circuit_breaker:
                escalation_mgr.set_circuit_breaker(circuit_breaker)

        dl_cfg = getattr(self.downloader, 'download_config', None) if self.downloader else None

        bot_floor_threshold = 5
        bot_abort_threshold = BOT_DETECTION_ABORT_THRESHOLD
        network_failure_threshold = NETWORK_FAILURE_THRESHOLD
        if dl_cfg:
            bot_floor_threshold = int(getattr(dl_cfg, 'bot_detection_tier_floor_threshold', 5))
            bot_abort_threshold = int(getattr(
                dl_cfg, 'bot_detection_abort_threshold', BOT_DETECTION_ABORT_THRESHOLD
            ))
            network_failure_threshold = int(getattr(
                dl_cfg, 'network_failure_threshold', NETWORK_FAILURE_THRESHOLD
            ))

        return _DownloadLoopContext(
            stats=stats,
            escalation_mgr=escalation_mgr,
            cookie_rotator=cookie_rotator,
            circuit_breaker=circuit_breaker,
            download_config=dl_cfg,
            bot_floor_threshold=bot_floor_threshold,
            bot_abort_threshold=bot_abort_threshold,
            network_failure_threshold=network_failure_threshold,
        )

    def _check_preconditions(
        self,
        ctx: _DownloadLoopContext,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
    ) -> Optional[str]:
        """Check whether the download should be skipped or paused.

        US-57-007: Extracts circuit-breaker logic from the main loop.

        Returns:
            A skip reason string if the segment should be skipped,
            or ``None`` if the download may proceed.
        """
        cb = ctx.circuit_breaker
        if not cb or not cb.is_open:
            return None

        at_max_tier = False
        if ctx.escalation_mgr:
            from ..downloader.types import EscalationTier
            kw_state = ctx.escalation_mgr._get_state(video_id)
            at_max_tier = kw_state.current_tier >= EscalationTier.VPN_ROTATION

        if at_max_tier:
            logger.info(
                f"Circuit breaker open + max tier reached for {video_id} "
                f"— skipping to retry queue"
            )
            if self.downloader and self.downloader.retry_queue:
                self.downloader.retry_queue.add(
                    video_id=f"{video_id}_{int(start)}_{int(end)}",
                    keyword='segment',
                    tier='segment',
                    error_message='circuit_breaker_open_max_tier',
                    error_category='video_specific',
                    escalation_tier=int(kw_state.current_tier),
                )
            return 'circuit_breaker_open_max_tier'

        # Not at max tier — pause and wait for circuit recovery
        cb.check_and_wait()
        return None

    def _execute_download(
        self,
        ctx: _DownloadLoopContext,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
    ) -> Dict[str, Any]:
        """Run the actual yt-dlp download for a single segment.

        US-57-007: Extracts download execution from the main loop.

        Returns:
            A dict with keys ``success`` (bool), and optionally ``duration``,
            ``error_msg``, or ``file_missing``.
        """
        import yt_dlp

        url = f"https://www.youtube.com/watch?v={video_id}"
        _progress_hook = self._make_progress_hook(video_id, ctx.stats)

        ydl_opts, escalation_result = self._build_ydl_opts(
            video_id=video_id,
            start=start,
            end=end,
            output_file=output_file,
            download_config=ctx.download_config,
            escalation_mgr=ctx.escalation_mgr,
            cookie_rotator=ctx.cookie_rotator,
            progress_hooks=[_progress_hook],
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

        # US-49-004: Read stall timeout for process-level hang detection
        _stall_timeout = 120
        if ctx.download_config:
            _raw_stall = getattr(ctx.download_config, 'segment_stall_timeout', 120)
            try:
                _stall_timeout = int(_raw_stall)
            except (TypeError, ValueError):
                _stall_timeout = 120

        seg_start_time = time.time()
        try:
            if _stall_timeout and _stall_timeout > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    def _do_download():
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([url])

                    future = executor.submit(_do_download)
                    try:
                        future.result(timeout=_stall_timeout)
                    except concurrent.futures.TimeoutError:
                        elapsed = time.time() - seg_start_time
                        logger.warning(
                            f"Segment {video_id}: ydl.download() stalled for "
                            f"{elapsed:.1f}s (timeout={_stall_timeout}s) — killing"
                        )
                        raise TimeoutError(
                            f"ydl.download() stalled for {elapsed:.1f}s "
                            f"(segment_stall_timeout={_stall_timeout}s)"
                        )
            else:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            duration = time.time() - seg_start_time
            if output_file.exists():
                return {'success': True, 'duration': duration}
            return {'success': False, 'file_missing': True, 'duration': duration}
        except Exception as e:
            return {'success': False, 'error_msg': str(e)}

    def _handle_result(
        self,
        ctx: _DownloadLoopContext,
        result: Dict[str, Any],
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        downloaded: list,
        idx: int,
        total: int,
        progress_callback,
    ) -> bool:
        """Process the outcome of a single segment download attempt.

        US-57-007: Dispatches to success or error handling. Returns
        ``True`` if the loop should abort (bot / network threshold hit).
        """
        from ..state import DownloadedVideo
        stats = ctx.stats

        # --- success path ---
        if result.get('success'):
            url = f"https://www.youtube.com/watch?v={video_id}"
            downloaded.append(DownloadedVideo(
                file=str(output_file), url=url, source='segment_download'
            ))
            try:
                _dl_bytes = output_file.stat().st_size
            except OSError:
                _dl_bytes = 0
            stats.increment_success(duration=result.get('duration', 0), file_bytes=_dl_bytes)
            ctx.consecutive_network_failures = 0
            if ctx.consecutive_bot_detections > 0:
                ctx.consecutive_bot_detections = 0
                if ctx.escalation_mgr:
                    ctx.escalation_mgr.clear_tier_floor()
            if ctx.escalation_mgr:
                ctx.escalation_mgr.record_success(video_id)
            return False

        # --- file-missing edge case (download didn't error but file absent) ---
        if result.get('file_missing'):
            stats.failed += 1
            stats.attempted += 1
            stats.segment_durations.append(result.get('duration', 0))
            # US-81-002: Track as failed item
            stats.failed_items.append({
                'video_id': video_id,
                'error_type': 'file_missing',
                'message': f'Download succeeded but file not found: {output_file}',
            })
            logger.warning(f"Download succeeded but file not found: {output_file}")
            return False

        # --- error path (delegated) ---
        return self._handle_download_error(
            ctx, result.get('error_msg', 'unknown error'),
            video_id, start, end, downloaded, idx, total, progress_callback,
        )

    def _handle_download_error(
        self,
        ctx: _DownloadLoopContext,
        error_msg: str,
        video_id: str,
        start: float,
        end: float,
        downloaded: list,
        idx: int,
        total: int,
        progress_callback,
    ) -> bool:
        """Handle a failed download: classify, escalate, and check abort thresholds.

        US-57-007: Extracted from _handle_result to keep each method under 80 lines.

        Returns:
            ``True`` if the loop should abort.
        """
        stats = ctx.stats
        _err_cat = classify_error_category(error_msg)
        stats.increment_failure(category=_err_cat, error_msg=error_msg, video_id=video_id)
        logger.warning(f"Failed to download segment {video_id}: {error_msg}")

        is_bot_error = _is_escalation_error(error_msg)
        if ctx.escalation_mgr and is_bot_error:
            ctx.escalation_mgr.record_failure(video_id, error_msg)
            if ctx.cookie_rotator and getattr(ctx.cookie_rotator, 'should_rotate', None):
                if ctx.cookie_rotator.should_rotate(error_msg):
                    ctx.cookie_rotator.rotate()

        # Bot-detection tracking and abort
        if is_bot_error:
            ctx.consecutive_bot_detections += 1
            if (
                ctx.bot_floor_threshold > 0
                and ctx.consecutive_bot_detections >= ctx.bot_floor_threshold
                and ctx.escalation_mgr
            ):
                from ..downloader.types import EscalationTier
                ctx.escalation_mgr.set_tier_floor(EscalationTier.VPN_ROTATION)
                logger.warning(
                    f"Bot-detection tier floor activated: "
                    f"{ctx.consecutive_bot_detections} consecutive bot-detection "
                    f"errors across video IDs — new downloads start at max tier"
                )
            if self._should_abort_bot_detection(ctx, stats, idx, total, downloaded, progress_callback):
                return True

        # Network failure tracking and abort
        if _is_network_failure(error_msg):
            ctx.consecutive_network_failures += 1
            logger.warning(
                f"Network failure detected ({ctx.consecutive_network_failures}/"
                f"{ctx.network_failure_threshold}): {error_msg}"
            )
            if ctx.consecutive_network_failures >= ctx.network_failure_threshold:
                remaining = total - idx
                logger.error(
                    f"Aborting download loop: {ctx.consecutive_network_failures} consecutive "
                    f"network failures indicate systemic network issue. "
                    f"Skipping {remaining} remaining segment(s)."
                )
                print(
                    f"  !! Network unavailable — aborting after "
                    f"{ctx.consecutive_network_failures} consecutive DNS/network failures "
                    f"({remaining} segments skipped)"
                )
                if progress_callback:
                    progress_callback(idx, total, downloaded)
                return True
        elif not is_bot_error:
            ctx.consecutive_network_failures = 0

        # Add to retry queue
        self._enqueue_retry(ctx, video_id, start, end, error_msg)
        return False

    def _should_abort_bot_detection(
        self,
        ctx: _DownloadLoopContext,
        stats: SegmentDownloadStats,
        idx: int,
        total: int,
        downloaded: list,
        progress_callback,
    ) -> bool:
        """Check whether bot-detection count has exceeded the abort threshold.

        US-57-007: Extracted from _handle_download_error for line budget.
        """
        if ctx.bot_abort_threshold <= 0:
            return False
        if ctx.consecutive_bot_detections < ctx.bot_abort_threshold:
            return False

        remaining = total - idx
        stats.error_categories['bot_detection_abort'] = 1
        stats.error_aggregator.record(
            f"Bot-detection abort: {ctx.consecutive_bot_detections} "
            f"consecutive bot errors exceeded threshold "
            f"({ctx.bot_abort_threshold})",
            'bot_detection_abort',
        )
        logger.error(
            f"Aborting download loop: {ctx.consecutive_bot_detections} "
            f"consecutive bot-detection errors (threshold: "
            f"{ctx.bot_abort_threshold}). YouTube is broadly blocking "
            f"requests. Skipping {remaining} remaining segment(s). "
            f"Check cookie configuration."
        )
        logger.error(
            "Suggested actions to resolve bot-detection:\n"
            "  1. Check/refresh your browser cookies "
            "(cookies_from_browser or cookies_path in config.yaml)\n"
            "  2. Enable Mullvad VPN rotation "
            "(download.mullvad.enabled: true)\n"
            "  3. Wait 15-30 minutes before retrying "
            "(YouTube rate limits are temporary)\n"
            "  4. Run with --resume to continue from this checkpoint"
        )
        print(
            f"  !! Bot-detection abort — {ctx.consecutive_bot_detections} "
            f"bot errors exceeded threshold ({ctx.bot_abort_threshold}). "
            f"{remaining} segments skipped.\n"
            f"     Fix: check cookies, enable VPN, or wait before "
            f"--resume"
        )
        if progress_callback:
            progress_callback(idx, total, downloaded)
        return True

    def _enqueue_retry(
        self,
        ctx: _DownloadLoopContext,
        video_id: str,
        start: float,
        end: float,
        error_msg: str,
    ) -> None:
        """Add a failed segment to the retry queue.

        US-57-007: Extracted from _handle_download_error for clarity.
        """
        if not self.downloader or not self.downloader.retry_queue:
            return
        category = classify_error_category(error_msg)
        _esc_tier = 1
        if ctx.escalation_mgr:
            try:
                _esc_state = ctx.escalation_mgr._get_state(video_id)
                _esc_tier = int(_esc_state.current_tier)
            except Exception:
                pass
        self.downloader.retry_queue.add(
            video_id=f"{video_id}_{int(start)}_{int(end)}",
            keyword='segment',
            tier='segment',
            error_message=error_msg,
            error_category=category,
            escalation_tier=_esc_tier,
        )
        logger.debug(
            f"Added {video_id} to retry queue "
            f"(category={category}, escalation_tier={_esc_tier})"
        )

    @staticmethod
    def _make_progress_hook(video_id: str, stats: SegmentDownloadStats) -> callable:
        """Create a yt-dlp progress_hooks callback for per-download observability.

        US-51-006: Logs download progress at INFO level for segments taking >30s,
        and logs final file size/time on completion. Accumulates totals into
        stats.progress_hooks_data for stage metrics.

        Args:
            video_id: YouTube video ID being downloaded.
            stats: The stage stats dataclass; progress data is accumulated
                   under stats.progress_hooks_data.

        Returns:
            A callable suitable for ydl_opts['progress_hooks'].
        """
        hook_data = stats.progress_hooks_data
        _last_log_elapsed = [0.0]  # mutable container for closure

        def _hook(d: Dict[str, Any]) -> None:
            status = d.get('status', '')
            elapsed = d.get('elapsed', 0.0) or 0.0

            if status == 'downloading' and elapsed > 30:
                # Throttle: only log every 15s of elapsed time
                if elapsed - _last_log_elapsed[0] >= 15:
                    _last_log_elapsed[0] = elapsed
                    downloaded = d.get('downloaded_bytes') or 0
                    total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                    speed = d.get('speed') or 0
                    speed_str = f"{speed / 1024:.0f} KB/s" if speed else "unknown"
                    total_str = f"{total_bytes / (1024 * 1024):.1f}MB" if total_bytes else "unknown"
                    logger.info(
                        f"Segment {video_id}: downloading — "
                        f"{downloaded / (1024 * 1024):.1f}MB / {total_str} "
                        f"@ {speed_str} (elapsed {elapsed:.0f}s)"
                    )
                    hook_data['segments_with_progress'] += 1

            elif status == 'finished':
                total_bytes = d.get('total_bytes') or d.get('downloaded_bytes') or 0
                if total_bytes:
                    hook_data['total_downloaded_bytes'] += total_bytes
                hook_data['segments_finished'] += 1
                if elapsed and elapsed > 0:
                    logger.info(
                        f"Segment {video_id}: finished — "
                        f"{total_bytes / (1024 * 1024):.1f}MB in {elapsed:.1f}s"
                    )

        return _hook

    def _build_ydl_opts(
        self,
        *,
        video_id: str,
        start: float,
        end: float,
        output_file: Path,
        download_config,
        escalation_mgr=None,
        cookie_rotator=None,
        progress_hooks: Optional[List] = None,
    ) -> tuple:
        """Build ydl_opts dict for a yt-dlp Python API download call.

        US-52-005: Extracts the repeated ydl_opts construction from
        _download_segments and the retry loop into a single builder.

        Encapsulates:
        - Base options (format, output, ranges, timeouts, retries)
        - Cookie propagation (cookiesfrombrowser / cookiefile fallback)
        - Escalation application (impersonation, extractor_args, cookie rotation)

        Args:
            video_id: YouTube video ID.
            start: Segment start time in seconds.
            end: Segment end time in seconds.
            output_file: Path for the downloaded file.
            download_config: Download config object (or None).
            escalation_mgr: Optional EscalationManager instance.
            cookie_rotator: Optional CookieRotator instance.
            progress_hooks: Optional list of progress hook callables.

        Returns:
            (ydl_opts, escalation_result) tuple. escalation_result may be None.
        """
        # Read segment config from download config with fallback defaults
        _socket_timeout = 30
        _max_res = 1080
        _seg_format = 'best[height<={segment_max_resolution}]'
        if download_config:
            _seg_sock = getattr(download_config, 'segment_socket_timeout', 0)
            _socket_timeout = _seg_sock if _seg_sock else getattr(download_config, 'socket_timeout', 30)
            _max_res = getattr(download_config, 'segment_max_resolution', 1080)
            _seg_format = getattr(download_config, 'segment_format', _seg_format)

        # Build format string with fallback chain to handle Tier 2+ escalation
        # where alternative player_clients may not expose formats matching the
        # height filter (causes "Requested format is not available")
        _primary_format = _seg_format.format(segment_max_resolution=_max_res)
        _format_with_fallback = f'{_primary_format}/best/bestvideo+bestaudio'

        ydl_opts: Dict[str, Any] = {
            'format': _format_with_fallback,
            'outtmpl': str(output_file),
            'quiet': True,
            'no_warnings': True,
            # Skip gracefully when rate-limited/unavailable (no formats returned);
            # caller checks output_file.exists() to detect actual failure
            'ignore_no_formats_error': True,
            # Auto-update EJS challenge solver scripts from GitHub so yt-dlp can
            # solve YouTube's n-sig challenges (required for format extraction)
            'remote_components': {'ejs:github'},
            # Time-based download options
            'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
            'force_keyframes_at_cuts': True,
            # Network resilience (matches core.py subprocess args)
            'socket_timeout': _socket_timeout,
            'retries': 10,
            'fragment_retries': 10,
        }

        if progress_hooks:
            ydl_opts['progress_hooks'] = progress_hooks

        # US-49-002: Propagate cookie auth to Python API
        if download_config:
            _browser = getattr(download_config, 'cookies_from_browser', '')
            if _browser:
                ydl_opts['cookiesfrombrowser'] = [_browser]
            else:
                _cookies_path = getattr(download_config, 'cookies_path', '')
                if not _cookies_path:
                    _cookie_rotation = getattr(download_config, 'cookie_rotation', None)
                    if _cookie_rotation:
                        _cookie_files = getattr(_cookie_rotation, 'cookie_files', [])
                        if _cookie_files:
                            _cookies_path = _cookie_files[0]
                if _cookies_path:
                    ydl_opts['cookiefile'] = _cookies_path

        # US-48-005: Apply escalation tiers (impersonation + extractor_args + cookies)
        escalation_result = None
        if escalation_mgr:
            try:
                escalation_result = escalation_mgr.get_escalation_args(video_id)
                _apply_escalation_to_ydl_opts(ydl_opts, escalation_result)

                # Tier 3: apply cookie rotation (overrides baseline cookies)
                if escalation_result.rotate_cookies and cookie_rotator:
                    cookie_path = cookie_rotator.get_current_cookie()
                    if cookie_path:
                        ydl_opts['cookiefile'] = cookie_path
                        # Remove browser cookies when using rotated cookie file
                        ydl_opts.pop('cookiesfrombrowser', None)
            except Exception as esc_err:
                logger.debug(f"Escalation lookup failed for {video_id}: {esc_err}")
        elif self.downloader and getattr(self.downloader, 'impersonation_manager', None):
            # Fallback: direct impersonation only (no escalation manager)
            try:
                imp_args = self.downloader.impersonation_manager.get_impersonate_args()
                if len(imp_args) >= 2 and imp_args[0] == '--impersonate':
                    ydl_opts['impersonate'] = imp_args[1]
            except Exception:
                pass

        return ydl_opts, escalation_result

    @staticmethod
    def _print_progress(current: int, total: int, stats: SegmentDownloadStats) -> None:
        """Print running progress line after each download attempt."""
        ok = stats.succeeded + stats.cached
        attempted = stats.attempted
        rate = (ok / attempted * 100) if attempted > 0 else 0.0
        print(
            f"  [{current}/{total}] "
            f"ok={ok} fail={stats.failed} cached={stats.cached} "
            f"({rate:.0f}% success)"
        )

    @staticmethod
    def _print_summary(stats: SegmentDownloadStats, elapsed: float) -> None:
        """Print end-of-stage summary."""
        ok = stats.succeeded + stats.cached
        attempted = stats.attempted
        rate = (ok / attempted * 100) if attempted > 0 else 0.0

        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            time_str = f"{h}h {m}m"

        print(f"\n  --- Download Summary ---")
        print(f"    Attempted: {attempted}/{stats.total}")
        print(f"    Succeeded: {stats.succeeded}")
        print(f"    Cached:    {stats.cached}")
        print(f"    Failed:    {stats.failed}")
        print(f"    Success rate: {rate:.0f}%")
        print(f"    Total time: {time_str}")

        # Per-segment duration stats
        durations = stats.segment_durations
        if durations:
            avg_dur = statistics.mean(durations)
            median_dur = statistics.median(durations)
            print(f"    Avg segment time: {avg_dur:.1f}s")
            print(f"    Median segment time: {median_dur:.1f}s")

        # Total bytes downloaded
        if stats.total_bytes > 0:
            if stats.total_bytes < 1024 * 1024:
                size_str = f"{stats.total_bytes / 1024:.1f} KB"
            elif stats.total_bytes < 1024 * 1024 * 1024:
                size_str = f"{stats.total_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{stats.total_bytes / (1024 * 1024 * 1024):.2f} GB"
            print(f"    Total size: {size_str}")

        # Retry count
        if stats.retry_count > 0:
            print(f"    Retried: {stats.retry_count}")

    @staticmethod
    def _log_error_summary(stats: SegmentDownloadStats) -> None:
        """US-49-009 + US-51-011: Log structured error summary with per-category breakdown.

        Logs at INFO level with category counts and sample messages, and at
        WARNING level with actionable guidance when >50% of failures are
        bot-detection.
        """
        if not stats.failed:
            return  # No errors to summarize

        summary_parts = [f"{cat}={count}" for cat, count in sorted(stats.error_categories.items())]
        logger.info(
            f"Download error summary: total={stats.total} "
            f"succeeded={stats.succeeded} failed={stats.failed} "
            f"cached={stats.cached} skipped="
            f"{stats.total - stats.attempted} | "
            f"errors by category: {', '.join(summary_parts) if summary_parts else 'uncategorized'}"
        )

        # US-51-011: Log categorized table with sample messages via ErrorAggregator
        if stats.error_aggregator.total_errors > 0:
            stats.error_aggregator.log_summary(stage_name='DOWNLOAD_SEGMENTS')

        # Actionable guidance when >50% of failures are bot-detection
        bot_count = stats.error_categories.get('bot_detection', 0)
        if bot_count > 0 and (bot_count / stats.failed) > 0.5:
            logger.warning(
                "Most failures are bot-detection. "
                "Check cookie configuration (cookies_from_browser or cookies_path in config.yaml)."
            )

    @staticmethod
    def _log_escalation_summary(escalation_summary: Dict[str, Any]) -> None:
        """US-50-009: Log structured escalation tier effectiveness summary.

        Logs at INFO level with tier distribution, success/failure counts,
        and per-tier effectiveness rates.
        """
        if not escalation_summary or not escalation_summary.get('total_escalations', 0):
            # No escalation data to report (all downloads succeeded at tier 1)
            if escalation_summary:
                logger.info(
                    "Escalation summary: no escalations needed "
                    f"(bot_detection={escalation_summary.get('bot_detection_count', 0)} "
                    f"network_failures={escalation_summary.get('network_failure_count', 0)})"
                )
            return

        vpt = escalation_summary.get('videos_per_tier', {})
        tier_parts = [f"{tier}={count}" for tier, count in sorted(vpt.items())]

        logger.info(
            f"Escalation summary: total_escalations={escalation_summary.get('total_escalations', 0)} "
            f"average_tier={escalation_summary.get('average_tier', 1.0)} "
            f"videos_per_tier=[{', '.join(tier_parts)}] "
            f"bot_detection={escalation_summary.get('bot_detection_count', 0)} "
            f"network_failures={escalation_summary.get('network_failure_count', 0)}"
        )

        # Log tier effectiveness if available
        tier_eff = escalation_summary.get('tier_effectiveness', {})
        if tier_eff:
            for category, rates in tier_eff.items():
                rate_parts = [f"{t}={r:.1%}" for t, r in sorted(rates.items())]
                logger.info(
                    f"Tier effectiveness [{category}]: {', '.join(rate_parts)}"
                )

    def _process_retry_queue(
        self,
        output_dir: Path,
        buffer_seconds: float,
        downloaded: List['DownloadedVideo'],
        total: int,
        progress_callback,
        stats: Optional[SegmentDownloadStats] = None
    ) -> None:
        """Process any failed downloads in the retry queue.

        Attempts to retry failed segment downloads using the downloader's
        retry queue infrastructure. US-48-005: Uses escalation tiers for
        retries (items that originally failed at Tier 1 will retry at
        the escalated tier).
        """
        from ..state import DownloadedVideo
        import yt_dlp

        if not self.downloader or not self.downloader.retry_queue:
            return

        retry_queue = self.downloader.retry_queue
        if not retry_queue.has_pending():
            return

        pending = retry_queue.get_pending_items()
        logger.info(f"Processing {len(pending)} items from retry queue")

        # Start retry pass (applies configured delay)
        retry_queue.start_retry_pass()

        # Get escalation manager and cookie rotator for retry pass
        escalation_mgr = getattr(self.downloader, 'escalation_manager', None)
        cookie_rotator = getattr(self.downloader, 'cookie_rotator', None)

        # Apply inter-request delay during retries too
        dl_cfg = getattr(self.downloader, 'download_config', None)
        _retry_delay = float(getattr(dl_cfg, 'segment_request_delay', 1.0)) if dl_cfg else 1.0

        retry_attempts = 0
        for item in pending:
            # Parse video_id from the retry item (format: video_id_start_end)
            # US-48-008: Use rsplit to handle video IDs with underscores
            parts = item.video_id.rsplit('_', 2)
            if len(parts) < 3:
                logger.warning(f"Invalid retry item format: {item.video_id}")
                retry_queue.mark_failed(item.video_id)
                continue

            video_id = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                logger.warning(f"Invalid time range in retry item: {item.video_id}")
                retry_queue.mark_failed(item.video_id)
                continue

            output_file = output_dir / f"{video_id}_{start}_{end}.mp4"

            if output_file.exists():
                retry_queue.mark_success(item.video_id)
                continue

            retry_attempts += 1
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"

                # US-49-010: Apply stored escalation tier floor before getting args.
                # This ensures the retry starts at the tier where the original
                # download failed (or higher), avoiding wasted lower-tier attempts.
                if escalation_mgr and item.escalation_tier > 1:
                    try:
                        from ..downloader.types import EscalationTier
                        stored_tier = EscalationTier(item.escalation_tier)
                        esc_state = escalation_mgr._get_state(video_id)
                        if esc_state.current_tier < stored_tier:
                            esc_state.current_tier = stored_tier
                            logger.debug(
                                f"Retry {video_id}: elevated escalation tier to "
                                f"{stored_tier.name} (from retry queue)"
                            )
                    except (ValueError, Exception):
                        pass

                # US-52-005: Build ydl_opts via builder (retry path)
                _dl_cfg = getattr(self.downloader, 'download_config', None)
                ydl_opts, _esc_result = self._build_ydl_opts(
                    video_id=video_id,
                    start=start,
                    end=end,
                    output_file=output_file,
                    download_config=_dl_cfg,
                    escalation_mgr=escalation_mgr,
                    cookie_rotator=cookie_rotator,
                )

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                if output_file.exists():
                    downloaded.append(DownloadedVideo(
                        file=str(output_file),
                        url=url,
                        source='segment_retry'
                    ))
                    retry_queue.mark_success(item.video_id)
                    if escalation_mgr:
                        escalation_mgr.record_success(video_id)
                    logger.info(f"Retry succeeded for {video_id}")
                else:
                    retry_queue.mark_failed(item.video_id)

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Retry failed for {video_id}: {error_msg}")
                retry_queue.mark_failed(item.video_id)
                # Record failure for escalation progression on next retry
                if escalation_mgr and _is_escalation_error(error_msg):
                    escalation_mgr.record_failure(video_id, error_msg)

            # Delay between retry requests to avoid rate-limiting
            if _retry_delay > 0:
                time.sleep(_retry_delay)

            # Update checkpoint with retry progress
            if progress_callback:
                progress_callback(len(downloaded), total, downloaded)

        # Update stats with retry count
        if stats is not None:
            stats.retry_count = retry_attempts

    def _update_matches_with_local_paths(
        self,
        state: 'PipelineState',
        downloaded_segments: List['DownloadedVideo']
    ):
        """Update match objects to reference local file paths.

        US-48-008: Uses rsplit('_', 2) to extract video_id from filename
        format '{video_id}_{start}_{end}.mp4', correctly handling video IDs
        that contain underscores (e.g., 'abc_def_0_15.mp4' → 'abc_def').
        """
        # Build mapping from video_id to local file
        file_map = {}
        for seg in downloaded_segments:
            # Extract video_id from filename: {video_id}_{start}_{end}.mp4
            filename = Path(seg.file).stem
            parts = filename.rsplit('_', 2)
            if len(parts) == 3:
                video_id = parts[0]
                file_map[video_id] = seg.file
            elif parts:
                # Fallback for unexpected format
                video_id = parts[0]
                file_map[video_id] = seg.file

        # Update matches
        updated_count = 0
        for match in state.matches:
            if hasattr(match, 'video_file') and match.video_file:
                video_id = match.video_file
                if video_id in file_map:
                    match.video_file = file_map[video_id]
                    updated_count += 1

        logger.info(f"Updated {updated_count} matches with local file paths")

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore from checkpoint.

        US-48-009: Validates restored segment files exist on disk and have
        non-zero size. Filters out deleted or empty files. Cross-references
        against state.matches to identify segments that still need downloading.

        US-51-010: Restores retry queue from checkpoint so previously-failed
        videos are retried before processing new segments on resume.
        """
        try:
            if config:
                output_dir = Path(config.downloaded_videos_dir)
                if output_dir.exists():
                    from ..state import DownloadedVideo
                    segments = []
                    skipped = 0
                    for f in output_dir.glob('*_*_*.mp4'):
                        if not f.exists() or f.stat().st_size == 0:
                            skipped += 1
                            continue
                        segments.append(DownloadedVideo(
                            file=str(f),
                            source='restored'
                        ))
                    state.downloaded_segments = segments
                    logger.info(
                        f"Restored DOWNLOAD_SEGMENTS: {len(segments)} valid, "
                        f"{skipped} invalid (missing or empty)"
                    )

                    # Cross-reference against matches to find segments needing download
                    if state.matches:
                        restored_ids = set()
                        for seg in segments:
                            fname = Path(seg.file).stem
                            parts = fname.rsplit('_', 2)
                            if len(parts) >= 3:
                                restored_ids.add(parts[0])

                        matched_ids = set()
                        for match in state.matches:
                            vid = getattr(match, 'video_file', '') or ''
                            if vid:
                                matched_ids.add(vid)

                        missing = matched_ids - restored_ids
                        if missing:
                            logger.info(
                                f"DOWNLOAD_SEGMENTS restore: {len(missing)} matched "
                                f"video(s) have no restored segments on disk"
                            )

            # US-51-010: Restore retry queue from checkpoint data
            cp_data = getattr(checkpoint, 'data', None)
            if cp_data:
                ds_data = getattr(cp_data, 'download_segments', {}) or {}
                retry_queue_data = ds_data.get('retry_queue')
                if retry_queue_data:
                    # Store on state for the run() method to pick up
                    if not hasattr(state, '_restored_retry_queue'):
                        state._restored_retry_queue = retry_queue_data
                    logger.info(
                        f"DOWNLOAD_SEGMENTS restore: found retry queue with "
                        f"{len(retry_queue_data.get('items', []))} pending items, "
                        f"{len(retry_queue_data.get('failed_ids', []))} permanently failed"
                    )

            return True
        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD_SEGMENTS: {e}")
            return True  # Non-critical, proceed anyway

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs"""
        if not state.matches:
            return "No matches available for segment download"
        return None
