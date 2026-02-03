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
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageMetrics, StageResult, register_stage, validate_required_state_attrs

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo

logger = logging.getLogger(__name__)

# Error patterns indicating systemic network failures (not video-specific)
# These match both subprocess stderr AND Python API DownloadError messages,
# which wrap exceptions as "ERROR: [youtube] ID: <original exception text>".
_NETWORK_FAILURE_PATTERNS = (
    'getaddrinfo failed',
    'Name or service not known',
    'Errno 11001',           # Windows DNS resolution failure
    'nodename nor servname',  # macOS DNS failure
    'Network is unreachable',
    'No address associated with hostname',
    'Temporary failure in name resolution',
    'URLError',              # Python urllib wrapper (e.g. URLError: <urlopen error ...>)
    'ConnectionResetError',  # Python API: connection dropped mid-transfer
)

# ffmpeg exit code 0xFFFFFEC6 = 4294967158 unsigned = -314 signed (network error)
_FFMPEG_NETWORK_EXIT_CODE = '4294967158'


def _is_network_failure(error_msg: str) -> bool:
    """Check if an error message indicates a systemic network failure.

    These are failures that affect ALL downloads (DNS down, no internet),
    as opposed to video-specific errors (403, removed, age-gated).

    Args:
        error_msg: The exception message string.

    Returns:
        True if the error indicates a systemic network issue.
    """
    error_lower = error_msg.lower()
    for pattern in _NETWORK_FAILURE_PATTERNS:
        if pattern.lower() in error_lower:
            return True
    # Check for ffmpeg network exit code
    if _FFMPEG_NETWORK_EXIT_CODE in error_msg:
        return True
    return False


# Default threshold for consecutive network failures before aborting
NETWORK_FAILURE_THRESHOLD = 3

# Default threshold for total bot-detection errors before aborting the stage.
# When YouTube is broadly blocking (broken cookies, defeated impersonation),
# every remaining segment hits the block wall with doomed Tier 1 requests.
# This threshold triggers an early abort with actionable guidance.
BOT_DETECTION_ABORT_THRESHOLD = 10


def classify_error_category(error_msg: str) -> str:
    """Classify a download error into a diagnostic category.

    Categories (most specific first):
        'network'       — DNS failure, no connectivity (systemic)
        'bot_detection' — 403/bot/captcha/sign-in errors
        'timeout'       — stall timeouts, socket timeouts
        'video_specific'— removed, age-gated, unavailable, etc.

    Args:
        error_msg: The exception message string.

    Returns:
        One of 'network', 'bot_detection', 'timeout', 'video_specific'.
    """
    if _is_network_failure(error_msg):
        return 'network'
    if _is_escalation_error(error_msg):
        return 'bot_detection'
    lower = error_msg.lower()
    if any(p in lower for p in ('timeout', 'timed out', 'stalled')):
        return 'timeout'
    return 'video_specific'


def _is_escalation_error(error_msg: str) -> bool:
    """Check if an error message indicates a 403/bot-detection/auth error.

    These errors warrant escalation to a higher bypass tier via the
    EscalationManager (Tier 2 extractor_args, Tier 3 cookies).

    Args:
        error_msg: The exception message string.

    Returns:
        True if the error matches 403/bot/auth patterns.
    """
    try:
        from ..downloader.escalation_manager import is_escalation_trigger
        return is_escalation_trigger(error_msg)
    except ImportError:
        # Fallback: simple pattern match if escalation_manager unavailable
        lower = error_msg.lower()
        return any(p in lower for p in ('403', 'forbidden', 'sign in', 'bot', 'captcha'))


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
                # Fall back to no impersonation rather than crashing
                pass
            i += 2
        elif args[i] == '--extractor-args' and i + 1 < len(args):
            # Parse "youtube:player_client=X,Y,Z" format
            raw = args[i + 1]
            if ':' in raw:
                namespace, kv = raw.split(':', 1)
                if '=' in kv:
                    key, value = kv.split('=', 1)
                    if 'extractor_args' not in ydl_opts:
                        ydl_opts['extractor_args'] = {}
                    if namespace not in ydl_opts['extractor_args']:
                        ydl_opts['extractor_args'][namespace] = {}
                    ydl_opts['extractor_args'][namespace][key] = value
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

    def __init__(self):
        self.downloader = None

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

            # Get download settings from config
            download_config = config.download
            buffer_seconds = download_config.segment_buffer

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

            # Download segments with progress callback for checkpointing
            def checkpoint_progress(current: int, total: int, downloaded: list):
                """Save progress checkpoint during download"""
                checkpoint_data = {
                    'segment_count': len(downloaded),
                    'segments_completed': current,
                    'segments_total': total,
                    'in_progress': current < total,
                }
                checkpoint.save_intermediate('DOWNLOAD_SEGMENTS', checkpoint_data)

            stage_start_time = time.time()

            downloaded_segments, download_stats = self._download_segments(
                segments_to_download,
                output_dir,
                buffer_seconds,
                checkpoint_progress
            )

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
                'retry_count': download_stats.get('retry_count', 0),
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

            # Stage metrics for pipeline observability (US-49-009 + US-49-012)
            metrics = StageMetrics(
                items_processed=download_stats['succeeded'] + download_stats['cached'],
                items_failed=download_stats['failed'],
                duration_seconds=elapsed,
                error_categories=download_stats.get('error_categories', {}),
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
        progress_callback
    ):
        """Download video segments using VideoDownloader's retry queue.

        Uses the downloader's impersonation, escalation, and retry queue
        infrastructure instead of raw yt-dlp calls.

        US-48-005: Uses EscalationManager for per-segment tier progression.
        On 403/bot errors, escalates to Tier 2 (extractor_args) and Tier 3
        (cookie rotation). Escalation state is tracked per video_id.

        Returns:
            Tuple of (downloaded_segments list, stats dict).
        """
        from ..state import DownloadedVideo
        import yt_dlp

        downloaded = []
        total = len(segments)
        consecutive_network_failures = 0
        consecutive_bot_detections = 0  # US-49-005: stage-level bot-detection counter

        # Progress counters
        stats = {
            'succeeded': 0,
            'failed': 0,
            'cached': 0,
            'attempted': 0,
            'total': total,
            'retry_count': 0,
            'segment_durations': [],   # Per-segment download durations (seconds)
            'total_bytes': 0,          # Total bytes downloaded (from output file sizes)
            'error_categories': {},    # US-49-009: Per-category error counts
        }

        # Get escalation manager, circuit breaker, and cookie rotator from downloader
        escalation_mgr = None
        cookie_rotator = None
        circuit_breaker = None
        if self.downloader:
            escalation_mgr = getattr(self.downloader, 'escalation_manager', None)
            cookie_rotator = getattr(self.downloader, 'cookie_rotator', None)
            circuit_breaker = getattr(self.downloader, 'circuit_breaker', None)

            # US-49-007: Wire circuit breaker into escalation manager so that
            # open-circuit state informs escalation decisions (skip to max tier)
            if escalation_mgr and circuit_breaker:
                escalation_mgr.set_circuit_breaker(circuit_breaker)

        # US-49-005: Read bot-detection tier floor threshold from config
        _dl_cfg_top = getattr(self.downloader, 'download_config', None) if self.downloader else None
        _bot_floor_threshold = 5  # default
        if _dl_cfg_top:
            _bot_floor_threshold = int(getattr(
                _dl_cfg_top, 'bot_detection_tier_floor_threshold', 5
            ))

        # US-49-008: Read bot-detection abort threshold from config
        _bot_abort_threshold = BOT_DETECTION_ABORT_THRESHOLD  # module-level default
        if _dl_cfg_top:
            _bot_abort_threshold = int(getattr(
                _dl_cfg_top, 'bot_detection_abort_threshold', BOT_DETECTION_ABORT_THRESHOLD
            ))

        for idx, seg in enumerate(segments, 1):
            video_id = seg['video_id']
            start = max(0, seg['start'] - buffer_seconds)
            end = seg['end'] + buffer_seconds

            # Create output filename
            output_file = output_dir / f"{video_id}_{int(start)}_{int(end)}.mp4"

            if output_file.exists():
                logger.info(f"Segment already exists: {output_file}")
                downloaded.append(DownloadedVideo(
                    file=str(output_file),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    source='segment_cache'
                ))
                stats['cached'] += 1
                stats['attempted'] += 1
                try:
                    stats['total_bytes'] += output_file.stat().st_size
                except OSError:
                    pass
                consecutive_network_failures = 0  # Cached file counts as success
                # US-49-005: Reset bot-detection counter on success
                if consecutive_bot_detections > 0:
                    consecutive_bot_detections = 0
                    if escalation_mgr:
                        escalation_mgr.clear_tier_floor()
                self._print_progress(idx, total, stats)
                continue

            # US-49-007: Check circuit breaker before download attempt.
            # If circuit is open and escalation is already at max tier,
            # skip the video and add to retry queue for later.
            if circuit_breaker and circuit_breaker.is_open:
                at_max_tier = False
                if escalation_mgr:
                    from ..downloader.types import EscalationTier
                    kw_state = escalation_mgr._get_state(video_id)
                    at_max_tier = kw_state.current_tier >= EscalationTier.VPN_ROTATION

                if at_max_tier:
                    stats['failed'] += 1
                    stats['attempted'] += 1
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
                    self._print_progress(idx, total, stats)
                    continue

            try:
                # Download segment using yt-dlp with downloader's infrastructure
                url = f"https://www.youtube.com/watch?v={video_id}"

                # Read segment config from download config with fallback defaults
                _dl_cfg = getattr(self.downloader, 'download_config', None) if self.downloader else None
                _socket_timeout = 30
                _max_res = 1080
                _seg_format = 'best[height<={segment_max_resolution}]'
                if _dl_cfg:
                    _seg_sock = getattr(_dl_cfg, 'segment_socket_timeout', 0)
                    _socket_timeout = _seg_sock if _seg_sock else getattr(_dl_cfg, 'socket_timeout', 30)
                    _max_res = getattr(_dl_cfg, 'segment_max_resolution', 1080)
                    _seg_format = getattr(_dl_cfg, 'segment_format', _seg_format)

                ydl_opts = {
                    'format': _seg_format.format(segment_max_resolution=_max_res),
                    'outtmpl': str(output_file),
                    'quiet': True,
                    'no_warnings': True,
                    # Time-based download options
                    'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
                    'force_keyframes_at_cuts': True,
                    # Network resilience (matches core.py subprocess args)
                    'socket_timeout': _socket_timeout,
                    'retries': 10,
                    'fragment_retries': 10,
                }

                # US-49-002: Propagate cookie auth to Python API (yt-dlp doesn't read CLI config)
                # The yt-dlp Python API key is 'cookiesfrombrowser' (list) not --cookies-from-browser
                if _dl_cfg:
                    _browser = getattr(_dl_cfg, 'cookies_from_browser', '')
                    if _browser:
                        ydl_opts['cookiesfrombrowser'] = [_browser]
                    else:
                        # Fallback: use cookies_path or first cookie_rotation file
                        _cookies_path = getattr(_dl_cfg, 'cookies_path', '')
                        if not _cookies_path:
                            _cookie_rotation = getattr(_dl_cfg, 'cookie_rotation', None)
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

                        if escalation_result.tier.value > 1:
                            logger.info(
                                f"Segment {video_id}: using escalation tier "
                                f"{escalation_result.tier.name}"
                            )
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

                # US-49-004: Read stall timeout for process-level hang detection
                _stall_timeout = 120  # fallback
                if _dl_cfg:
                    _raw_stall = getattr(_dl_cfg, 'segment_stall_timeout', 120)
                    try:
                        _stall_timeout = int(_raw_stall)
                    except (TypeError, ValueError):
                        _stall_timeout = 120

                seg_start_time = time.time()
                if _stall_timeout and _stall_timeout > 0:
                    # US-49-004: Wrap ydl.download() in ThreadPoolExecutor to detect
                    # process-level stalls (ffmpeg hangs, stream stalls with no data).
                    # socket_timeout only covers HTTP sockets; this covers the entire call.
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
                seg_duration = time.time() - seg_start_time

                if output_file.exists():
                    downloaded.append(DownloadedVideo(
                        file=str(output_file),
                        url=url,
                        source='segment_download'
                    ))
                    stats['succeeded'] += 1
                    stats['attempted'] += 1
                    stats['segment_durations'].append(seg_duration)
                    try:
                        stats['total_bytes'] += output_file.stat().st_size
                    except OSError:
                        pass
                    consecutive_network_failures = 0  # Reset on success
                    # US-49-005: Reset bot-detection counter on success
                    if consecutive_bot_detections > 0:
                        consecutive_bot_detections = 0
                        if escalation_mgr:
                            escalation_mgr.clear_tier_floor()
                    self._print_progress(idx, total, stats)
                    # Record success with escalation manager
                    if escalation_mgr:
                        escalation_mgr.record_success(video_id)
                else:
                    stats['failed'] += 1
                    stats['attempted'] += 1
                    stats['segment_durations'].append(seg_duration)
                    self._print_progress(idx, total, stats)
                    logger.warning(f"Download succeeded but file not found: {output_file}")

            except Exception as e:
                error_msg = str(e)
                stats['failed'] += 1
                stats['attempted'] += 1
                # US-49-009: Track error by category for end-of-stage summary
                _err_cat = classify_error_category(error_msg)
                stats['error_categories'][_err_cat] = stats['error_categories'].get(_err_cat, 0) + 1
                logger.warning(f"Failed to download segment {video_id}: {error_msg}")

                # US-48-005: Record failure with escalation manager for tier progression
                is_bot_error = _is_escalation_error(error_msg)
                if escalation_mgr and is_bot_error:
                    escalation_mgr.record_failure(video_id, error_msg)
                    # Advance cookie rotation on Tier 3+ auth errors
                    if cookie_rotator and getattr(cookie_rotator, 'should_rotate', None):
                        if cookie_rotator.should_rotate(error_msg):
                            cookie_rotator.rotate()

                # US-49-005: Track stage-level bot-detection counter
                if is_bot_error:
                    consecutive_bot_detections += 1
                    if (
                        _bot_floor_threshold > 0
                        and consecutive_bot_detections >= _bot_floor_threshold
                        and escalation_mgr
                    ):
                        from ..downloader.types import EscalationTier
                        escalation_mgr.set_tier_floor(EscalationTier.VPN_ROTATION)
                        logger.warning(
                            f"Bot-detection tier floor activated: "
                            f"{consecutive_bot_detections} consecutive bot-detection "
                            f"errors across video IDs — new downloads start at max tier"
                        )

                    # US-49-008: Abort stage when total bot-detection errors exceed threshold
                    if (
                        _bot_abort_threshold > 0
                        and consecutive_bot_detections >= _bot_abort_threshold
                    ):
                        remaining = total - idx
                        logger.error(
                            f"Aborting download loop: {consecutive_bot_detections} "
                            f"consecutive bot-detection errors (threshold: "
                            f"{_bot_abort_threshold}). YouTube is broadly blocking "
                            f"requests. Skipping {remaining} remaining segment(s)."
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
                            f"  !! Bot-detection abort — {consecutive_bot_detections} "
                            f"bot errors exceeded threshold ({_bot_abort_threshold}). "
                            f"{remaining} segments skipped.\n"
                            f"     Fix: check cookies, enable VPN, or wait before "
                            f"--resume"
                        )
                        # Checkpoint progress before aborting so --resume works
                        if progress_callback:
                            progress_callback(idx, total, downloaded)
                        break

                # Track consecutive network failures for early abort
                # US-49-005: Bot-detection errors do NOT reset the network failure counter
                if _is_network_failure(error_msg):
                    consecutive_network_failures += 1
                    logger.warning(
                        f"Network failure detected ({consecutive_network_failures}/"
                        f"{NETWORK_FAILURE_THRESHOLD}): {error_msg}"
                    )
                    if consecutive_network_failures >= NETWORK_FAILURE_THRESHOLD:
                        remaining = total - idx
                        logger.error(
                            f"Aborting download loop: {consecutive_network_failures} consecutive "
                            f"network failures indicate systemic network issue. "
                            f"Skipping {remaining} remaining segment(s)."
                        )
                        print(
                            f"  !! Network unavailable — aborting after "
                            f"{consecutive_network_failures} consecutive DNS/network failures "
                            f"({remaining} segments skipped)"
                        )
                        # Checkpoint before aborting
                        if progress_callback:
                            progress_callback(idx, total, downloaded)
                        break
                elif not is_bot_error:
                    # Only non-network, non-bot errors reset the counter
                    # (e.g., video removed, age-gated without bot detection)
                    consecutive_network_failures = 0

                self._print_progress(idx, total, stats)

                # Add failed download to retry queue for batch retry later
                if self.downloader and self.downloader.retry_queue:
                    category = classify_error_category(error_msg)
                    # US-49-010: Capture current escalation tier so retry starts
                    # at this tier or higher (avoids wasting time on lower tiers)
                    _esc_tier = 1
                    if escalation_mgr:
                        try:
                            _esc_state = escalation_mgr._get_state(video_id)
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

            # Checkpoint progress
            if progress_callback:
                progress_callback(idx, total, downloaded)

        # Process retry queue if there are pending items
        self._process_retry_queue(output_dir, buffer_seconds, downloaded, total, progress_callback, stats)

        # US-49-009: Log structured error summary with actionable diagnostics
        self._log_error_summary(stats)

        return downloaded, stats

    @staticmethod
    def _print_progress(current: int, total: int, stats: Dict[str, int]) -> None:
        """Print running progress line after each download attempt."""
        ok = stats['succeeded'] + stats['cached']
        attempted = stats['attempted']
        rate = (ok / attempted * 100) if attempted > 0 else 0.0
        print(
            f"  [{current}/{total}] "
            f"ok={ok} fail={stats['failed']} cached={stats['cached']} "
            f"({rate:.0f}% success)"
        )

    @staticmethod
    def _print_summary(stats: Dict[str, Any], elapsed: float) -> None:
        """Print end-of-stage summary."""
        ok = stats['succeeded'] + stats['cached']
        attempted = stats['attempted']
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
        print(f"    Attempted: {attempted}/{stats['total']}")
        print(f"    Succeeded: {stats['succeeded']}")
        print(f"    Cached:    {stats['cached']}")
        print(f"    Failed:    {stats['failed']}")
        print(f"    Success rate: {rate:.0f}%")
        print(f"    Total time: {time_str}")

        # Per-segment duration stats
        durations = stats.get('segment_durations', [])
        if durations:
            avg_dur = statistics.mean(durations)
            median_dur = statistics.median(durations)
            print(f"    Avg segment time: {avg_dur:.1f}s")
            print(f"    Median segment time: {median_dur:.1f}s")

        # Total bytes downloaded
        total_bytes = stats.get('total_bytes', 0)
        if total_bytes > 0:
            if total_bytes < 1024 * 1024:
                size_str = f"{total_bytes / 1024:.1f} KB"
            elif total_bytes < 1024 * 1024 * 1024:
                size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"
            print(f"    Total size: {size_str}")

        # Retry count
        retry_count = stats.get('retry_count', 0)
        if retry_count > 0:
            print(f"    Retried: {retry_count}")

    @staticmethod
    def _log_error_summary(stats: Dict[str, Any]) -> None:
        """US-49-009: Log structured error summary with per-category breakdown.

        Logs at INFO level with category counts, and at WARNING level with
        actionable guidance when >50% of failures are bot-detection.
        """
        error_cats = stats.get('error_categories', {})
        failed = stats.get('failed', 0)
        if not failed:
            return  # No errors to summarize

        summary_parts = [f"{cat}={count}" for cat, count in sorted(error_cats.items())]
        logger.info(
            f"Download error summary: total={stats.get('total', 0)} "
            f"succeeded={stats.get('succeeded', 0)} failed={failed} "
            f"cached={stats.get('cached', 0)} skipped="
            f"{stats.get('total', 0) - stats.get('attempted', 0)} | "
            f"errors by category: {', '.join(summary_parts) if summary_parts else 'uncategorized'}"
        )

        # Actionable guidance when >50% of failures are bot-detection
        bot_count = error_cats.get('bot_detection', 0)
        if bot_count > 0 and (bot_count / failed) > 0.5:
            logger.warning(
                "Most failures are bot-detection. "
                "Check cookie configuration (cookies_from_browser or cookies_path in config.yaml)."
            )

    def _process_retry_queue(
        self,
        output_dir: Path,
        buffer_seconds: float,
        downloaded: List['DownloadedVideo'],
        total: int,
        progress_callback,
        stats: Optional[Dict[str, Any]] = None
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

                # Read segment config from download config with fallback defaults
                _dl_cfg = getattr(self.downloader, 'download_config', None)
                _socket_timeout = 30
                _max_res = 1080
                _seg_format = 'best[height<={segment_max_resolution}]'
                if _dl_cfg:
                    _seg_sock = getattr(_dl_cfg, 'segment_socket_timeout', 0)
                    _socket_timeout = _seg_sock if _seg_sock else getattr(_dl_cfg, 'socket_timeout', 30)
                    _max_res = getattr(_dl_cfg, 'segment_max_resolution', 1080)
                    _seg_format = getattr(_dl_cfg, 'segment_format', _seg_format)

                ydl_opts = {
                    'format': _seg_format.format(segment_max_resolution=_max_res),
                    'outtmpl': str(output_file),
                    'quiet': True,
                    'no_warnings': True,
                    'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
                    'force_keyframes_at_cuts': True,
                    # Network resilience (matches core.py subprocess args)
                    'socket_timeout': _socket_timeout,
                    'retries': 10,
                    'fragment_retries': 10,
                }

                # US-49-002: Propagate cookie auth to Python API (retry path)
                if _dl_cfg:
                    _browser = getattr(_dl_cfg, 'cookies_from_browser', '')
                    if _browser:
                        ydl_opts['cookiesfrombrowser'] = [_browser]
                    else:
                        _cookies_path = getattr(_dl_cfg, 'cookies_path', '')
                        if not _cookies_path:
                            _cookie_rotation = getattr(_dl_cfg, 'cookie_rotation', None)
                            if _cookie_rotation:
                                _cookie_files = getattr(_cookie_rotation, 'cookie_files', [])
                                if _cookie_files:
                                    _cookies_path = _cookie_files[0]
                        if _cookies_path:
                            ydl_opts['cookiefile'] = _cookies_path

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

                # US-48-005: Apply escalation tiers for retry
                if escalation_mgr:
                    try:
                        escalation_result = escalation_mgr.get_escalation_args(video_id)
                        _apply_escalation_to_ydl_opts(ydl_opts, escalation_result)
                        if escalation_result.rotate_cookies and cookie_rotator:
                            cookie_path = cookie_rotator.get_current_cookie()
                            if cookie_path:
                                ydl_opts['cookiefile'] = cookie_path
                                ydl_opts.pop('cookiesfrombrowser', None)
                    except Exception:
                        pass
                elif self.downloader.impersonation_manager:
                    try:
                        imp_args = self.downloader.impersonation_manager.get_impersonate_args()
                        if len(imp_args) >= 2 and imp_args[0] == '--impersonate':
                            ydl_opts['impersonate'] = imp_args[1]
                    except Exception:
                        pass

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

            # Update checkpoint with retry progress
            if progress_callback:
                progress_callback(len(downloaded), total, downloaded)

        # Update stats with retry count
        if stats is not None:
            stats['retry_count'] = retry_attempts

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
