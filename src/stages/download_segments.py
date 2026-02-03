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

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage, validate_required_state_attrs

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo

logger = logging.getLogger(__name__)

# Error patterns indicating systemic network failures (not video-specific)
_NETWORK_FAILURE_PATTERNS = (
    'getaddrinfo failed',
    'Name or service not known',
    'Errno 11001',           # Windows DNS resolution failure
    'nodename nor servname',  # macOS DNS failure
    'Network is unreachable',
    'No address associated with hostname',
    'Temporary failure in name resolution',
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


def classify_error_category(error_msg: str) -> str:
    """Classify an error as 'network_systemic' or 'video_specific'.

    Network-systemic errors (DNS failure, no connectivity) affect ALL
    downloads and retrying individual items won't help. Video-specific
    errors (403, removed, age-gated) may succeed on retry with escalation.

    Args:
        error_msg: The exception message string.

    Returns:
        'network_systemic' for DNS/connectivity failures, 'video_specific' otherwise.
    """
    if _is_network_failure(error_msg):
        return 'network_systemic'
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
            ydl_opts['impersonate'] = args[i + 1]
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
            buffer_seconds = getattr(download_config, 'segment_buffer', 5.0)

            print(f"  Downloading matched segments")
            print(f"    Buffer: {buffer_seconds}s before/after each match")

            # Collect segments to download
            segments_to_download = self._collect_matched_segments(state)

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
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Video segment download failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _collect_matched_segments(self, state: 'PipelineState') -> List[Dict[str, Any]]:
        """Collect segment info from matches for downloading"""
        segments = []
        seen = set()  # Avoid duplicate downloads

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

            # Create unique key for deduplication
            key = (video_id, round(start_time), round(end_time))
            if key in seen:
                continue
            seen.add(key)

            segments.append({
                'video_id': video_id,
                'start': start_time,
                'end': end_time,
            })

        return segments

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

        # Progress counters
        stats = {
            'succeeded': 0,
            'failed': 0,
            'cached': 0,
            'attempted': 0,
            'total': total,
        }

        # Get escalation manager and cookie rotator from downloader
        escalation_mgr = None
        cookie_rotator = None
        if self.downloader:
            escalation_mgr = getattr(self.downloader, 'escalation_manager', None)
            cookie_rotator = getattr(self.downloader, 'cookie_rotator', None)

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
                consecutive_network_failures = 0  # Cached file counts as success
                self._print_progress(idx, total, stats)
                continue

            try:
                # Download segment using yt-dlp with downloader's infrastructure
                url = f"https://www.youtube.com/watch?v={video_id}"

                # Read socket_timeout from download config with fallback default
                _socket_timeout = 30
                if self.downloader and hasattr(self.downloader, 'download_config'):
                    _socket_timeout = getattr(self.downloader.download_config, 'socket_timeout', 30)

                ydl_opts = {
                    'format': 'best[height<=1080]',
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

                # US-48-005: Apply escalation tiers (impersonation + extractor_args + cookies)
                escalation_result = None
                if escalation_mgr:
                    try:
                        escalation_result = escalation_mgr.get_escalation_args(video_id)
                        _apply_escalation_to_ydl_opts(ydl_opts, escalation_result)

                        # Tier 3: apply cookie rotation
                        if escalation_result.rotate_cookies and cookie_rotator:
                            cookie_path = cookie_rotator.get_current_cookie()
                            if cookie_path:
                                ydl_opts['cookiefile'] = cookie_path

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

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                if output_file.exists():
                    downloaded.append(DownloadedVideo(
                        file=str(output_file),
                        url=url,
                        source='segment_download'
                    ))
                    stats['succeeded'] += 1
                    stats['attempted'] += 1
                    consecutive_network_failures = 0  # Reset on success
                    self._print_progress(idx, total, stats)
                    # Record success with escalation manager
                    if escalation_mgr:
                        escalation_mgr.record_success(video_id)
                else:
                    stats['failed'] += 1
                    stats['attempted'] += 1
                    self._print_progress(idx, total, stats)
                    logger.warning(f"Download succeeded but file not found: {output_file}")

            except Exception as e:
                error_msg = str(e)
                stats['failed'] += 1
                stats['attempted'] += 1
                logger.warning(f"Failed to download segment {video_id}: {error_msg}")

                # US-48-005: Record failure with escalation manager for tier progression
                if escalation_mgr and _is_escalation_error(error_msg):
                    escalation_mgr.record_failure(video_id, error_msg)
                    # Advance cookie rotation on Tier 3+ auth errors
                    if cookie_rotator and getattr(cookie_rotator, 'should_rotate', None):
                        if cookie_rotator.should_rotate(error_msg):
                            cookie_rotator.rotate()

                # Track consecutive network failures for early abort
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
                else:
                    # Non-network error (403, removed, etc.) — reset counter
                    consecutive_network_failures = 0

                self._print_progress(idx, total, stats)

                # Add failed download to retry queue for batch retry later
                if self.downloader and self.downloader.retry_queue:
                    category = classify_error_category(error_msg)
                    self.downloader.retry_queue.add(
                        video_id=f"{video_id}_{int(start)}_{int(end)}",
                        keyword='segment',
                        tier='segment',
                        error_message=error_msg,
                        error_category=category,
                    )
                    logger.debug(
                        f"Added {video_id} to retry queue (category={category})"
                    )

            # Checkpoint progress
            if progress_callback:
                progress_callback(idx, total, downloaded)

        # Process retry queue if there are pending items
        self._process_retry_queue(output_dir, buffer_seconds, downloaded, total, progress_callback)

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
    def _print_summary(stats: Dict[str, int], elapsed: float) -> None:
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

    def _process_retry_queue(
        self,
        output_dir: Path,
        buffer_seconds: float,
        downloaded: List['DownloadedVideo'],
        total: int,
        progress_callback
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

        for item in pending:
            # Parse video_id from the retry item (format: video_id_start_end)
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

            try:
                url = f"https://www.youtube.com/watch?v={video_id}"

                # Read socket_timeout from download config with fallback default
                _socket_timeout = 30
                if hasattr(self.downloader, 'download_config'):
                    _socket_timeout = getattr(self.downloader.download_config, 'socket_timeout', 30)

                ydl_opts = {
                    'format': 'best[height<=1080]',
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

                # US-48-005: Apply escalation tiers for retry
                if escalation_mgr:
                    try:
                        escalation_result = escalation_mgr.get_escalation_args(video_id)
                        _apply_escalation_to_ydl_opts(ydl_opts, escalation_result)
                        if escalation_result.rotate_cookies and cookie_rotator:
                            cookie_path = cookie_rotator.get_current_cookie()
                            if cookie_path:
                                ydl_opts['cookiefile'] = cookie_path
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

    def _update_matches_with_local_paths(
        self,
        state: 'PipelineState',
        downloaded_segments: List['DownloadedVideo']
    ):
        """Update match objects to reference local file paths"""
        # Build mapping from video_id to local file
        file_map = {}
        for seg in downloaded_segments:
            # Extract video_id from filename
            filename = Path(seg.file).stem
            parts = filename.split('_')
            if parts:
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
        """Restore from checkpoint"""
        # Segment files are on disk, scan for them
        try:
            if config:
                output_dir = Path(config.downloaded_videos_dir)
                if output_dir.exists():
                    from ..state import DownloadedVideo
                    segments = []
                    for f in output_dir.glob('*_*_*.mp4'):
                        segments.append(DownloadedVideo(
                            file=str(f),
                            source='restored'
                        ))
                    state.downloaded_segments = segments
                    logger.info(f"Restored DOWNLOAD_SEGMENTS: {len(segments)} segments from disk")
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
