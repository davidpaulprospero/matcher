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
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from ..state import AudioDownload
from .types import MergedSegment, DownloadedSegment
from .cookie_rotator import CookieRotator
from .impersonation import ImpersonationManager
from . import segment_utils
from . import utils

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5  # seconds
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
        lock,
        cookie_rotator: Optional[CookieRotator] = None,
        impersonation_manager: Optional[ImpersonationManager] = None
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
            cookie_rotator: Optional CookieRotator for cookie rotation on errors
            impersonation_manager: Optional ImpersonationManager for TLS fingerprint bypass
        """
        self.config = config
        self.download_config = config.download
        self._get_tier_value = get_tier_value_func
        self._search_video_metadata = search_metadata_func
        self._filter_titles_with_llm = filter_titles_func
        self._cleanup_partial_files = cleanup_partial_func
        self.tier_download_counts = tier_download_counts
        self._lock = lock
        self.cookie_rotator = cookie_rotator
        self.impersonation_manager = impersonation_manager

        # Log cookie rotation status
        if self.cookie_rotator and self.cookie_rotator.is_enabled:
            logger.info(f"AudioFirstPipeline: Cookie rotation enabled ({self.cookie_rotator.available_cookies} cookies)")
        else:
            logger.debug("AudioFirstPipeline: Using static cookies")

        # Log impersonation status
        if self.impersonation_manager and self.impersonation_manager.target_count > 0:
            logger.info(f"AudioFirstPipeline: Impersonation enabled ({self.impersonation_manager.target_count} targets)")
        else:
            logger.debug("AudioFirstPipeline: Impersonation not available")

    def _get_cookie_args(self) -> List[str]:
        """
        Get cookie arguments for yt-dlp command.

        Uses CookieRotator if enabled, otherwise falls back to static cookies.

        Returns:
            List of yt-dlp cookie arguments (e.g., ['--cookies', '/path/to/cookies.txt'])
        """
        # Use cookie rotator if enabled
        if self.cookie_rotator and self.cookie_rotator.is_enabled:
            current_cookie = self.cookie_rotator.get_current_cookie()
            if current_cookie:
                return ['--cookies', current_cookie]

        # Fallback to static cookie configuration
        return utils.get_cookies_args(self.config)

    def _add_impersonation_to_cmd(self, cmd: list) -> None:
        """Add browser impersonation args to yt-dlp command.

        Injects --impersonate with the next rotated target from the
        shared ImpersonationManager. Must be called BEFORE cookie args
        to maintain correct yt-dlp argument ordering.

        When impersonation is unavailable or no targets detected,
        this is a no-op (command unchanged).
        """
        if self.impersonation_manager:
            args = self.impersonation_manager.get_impersonate_args()
            if args:
                cmd.extend(args)

    def rotate_cookie_on_error(self, error_message: str) -> bool:
        """
        Attempt to rotate cookie based on error message.

        Args:
            error_message: Error message from yt-dlp stderr

        Returns:
            True if cookie was rotated, False otherwise
        """
        if not self.cookie_rotator or not self.cookie_rotator.is_enabled:
            return False

        if self.cookie_rotator.should_rotate(error_message):
            new_cookie = self.cookie_rotator.rotate()
            if new_cookie:
                logger.info(f"AudioFirstPipeline: Rotated to new cookie: {Path(new_cookie).name}")
                return True
            else:
                logger.warning("AudioFirstPipeline: Cookie rotation exhausted - no more cookies available")

        return False

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
            v for v in search_results.videos
            if (v.get('duration') or 0) >= tier_min
            and (v.get('duration') or 0) <= tier_max
            and not v.get('is_live', False)  # Skip live videos
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

            # Get tier-specific timeout
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            if isinstance(tier_timeouts, dict):
                audio_timeout = tier_timeouts.get(tier, 120)
            else:
                audio_timeout = getattr(tier_timeouts, tier, 120)

            # Retry loop with cookie rotation
            max_cookie_rotations = 2  # Try up to 2 cookie rotations per video
            actual_file = None

            for rotation_attempt in range(max_cookie_rotations + 1):
                # Rebuild command with impersonation + cookie on each attempt
                download_cmd = cmd.copy()
                self._add_impersonation_to_cmd(download_cmd)
                download_cmd.extend(self._get_cookie_args())

                try:
                    result = subprocess.run(
                        download_cmd,
                        capture_output=True,
                        text=True,
                        timeout=audio_timeout
                    )

                    # Find the actual downloaded file
                    if result.returncode == 0:
                        matches = list(audio_dir.glob(f"{video_id}.*"))
                        if matches:
                            actual_file = matches[0]
                            logger.debug(f"Found audio file: {actual_file.name}")
                            break  # Success, exit retry loop

                    # Check for errors that warrant cookie rotation
                    err_msg = result.stderr if result.stderr else ''
                    if self.rotate_cookie_on_error(err_msg):
                        logger.info(f"Retrying {video_id} with rotated cookie (attempt {rotation_attempt + 2})")
                        self._cleanup_partial_files(audio_dir, video_id)
                        time.sleep(2)  # Brief pause before retry
                        continue  # Try again with new cookie

                    # Non-rotatable error, log and break
                    err_snippet = err_msg[-500:] if len(err_msg) > 500 else err_msg
                    logger.warning(f"Audio download failed for {video_id} (rc={result.returncode})")
                    logger.warning(f"  Error output: {err_snippet}")
                    self._cleanup_partial_files(audio_dir, video_id)
                    break

                except subprocess.TimeoutExpired:
                    logger.warning(f"Audio download timeout for {video_id}")
                    self._cleanup_partial_files(audio_dir, video_id)
                    break  # Don't retry on timeout
                except Exception as e:
                    logger.warning(f"Audio download error for {video_id}: {e}")
                    self._cleanup_partial_files(audio_dir, video_id)
                    break  # Don't retry on unknown errors

            if actual_file:
                audio_downloads.append(AudioDownload(
                    file=str(actual_file),
                    video_id=video_id,
                    url=video_url,
                    title=video_info.get('title', ''),
                    duration=video_info.get('duration', 0),
                    keyword=keyword
                ))
                logger.debug(f"Downloaded audio: {actual_file.name}")

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

        downloaded_segments = []

        # Group by video_id
        by_video: Dict[str, List[MergedSegment]] = {}
        for seg in merged_segments:
            if seg.video_id not in by_video:
                by_video[seg.video_id] = []
            by_video[seg.video_id].append(seg)

        total_videos = len(by_video)
        current_video = 0

        for video_id, segments in by_video.items():
            if not segments:
                continue

            current_video += 1

            first_seg = segments[0]
            video_url = first_seg.video_url
            keyword = first_seg.keyword

            total_seg_duration = sum(seg.end_time - seg.start_time for seg in segments)

            print(f"  [{current_video}/{total_videos}] {video_id} ({len(segments)} segments, {total_seg_duration:.0f}s)")

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
                    print(f"      ✓ Already downloaded ({existing_count} segments)")
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

            # Build base yt-dlp command (cookies added in retry loop)
            base_cmd = [
                'yt-dlp',
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
                base_cmd.extend(['--ffmpeg-location', ffmpeg_loc])

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

                # Build command with impersonation + cookies (may have rotated)
                cmd = base_cmd.copy()
                self._add_impersonation_to_cmd(cmd)
                cmd.extend(self._get_cookie_args())

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )

                    if result.returncode != 0:
                        last_error = result.stderr[-200:] if result.stderr else 'Unknown error'
                        # Try cookie rotation first for auth/rate-limit errors
                        if self.rotate_cookie_on_error(result.stderr or ''):
                            logger.info(f"Cookie rotated for {video_id}, retrying...")
                            continue
                        # Check if error is retryable (network issues, rate limiting)
                        if self._is_retryable_error(result.stderr):
                            logger.warning(f"Retryable error for {video_id}: {last_error}")
                            continue
                        else:
                            print(f"      ✗ Failed: {last_error}")
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

                        print(f"      ✓ Downloaded {success_count}/{len(segments)} segments")
                        break  # Success, exit retry loop

                except subprocess.TimeoutExpired:
                    last_error = f"Timeout after {timeout}s"
                    print(f"      ✗ {last_error} (attempt {attempt + 1}/{max_retries})")
                    logger.warning(f"Segment download timeout for {video_id}")
                    # Timeout is retryable
                    continue
                except Exception as e:
                    last_error = str(e)
                    print(f"      ✗ Error: {e}")
                    logger.error(f"Segment download error for {video_id}: {e}")
                    break  # Non-retryable error

            # Log final failure if all retries exhausted
            if not segment_success and last_error:
                logger.error(f"All {max_retries} attempts failed for {video_id}: {last_error}")

            # Fallback to full video if segment download failed
            if not segment_success and fallback_full:
                print(f"      → Falling back to full video download...")
                fallback_segments = self._download_full_video_fallback(
                    video_id=video_id,
                    video_url=video_url,
                    video_dir=video_dir,
                    segments=segments,
                    keyword=keyword,
                    timeout=timeout
                )
                downloaded_segments.extend(fallback_segments)

            # Periodic checkpoint save to allow resume if interrupted
            if progress_callback and current_video % checkpoint_interval == 0:
                logger.info(f"Checkpoint save at video {current_video}/{total_videos}")
                try:
                    progress_callback(current_video, total_videos, downloaded_segments)
                except Exception as e:
                    logger.warning(f"Checkpoint callback failed: {e}")

        # Final progress callback
        if progress_callback:
            try:
                progress_callback(total_videos, total_videos, downloaded_segments)
            except Exception as e:
                logger.warning(f"Final checkpoint callback failed: {e}")

        logger.info(f"Downloaded {len(downloaded_segments)} video segments")
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

        base_cmd = [
            'yt-dlp',
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
            base_cmd.extend(['--ffmpeg-location', ffmpeg_loc])

        # Retry with cookie rotation (1 retry)
        max_cookie_rotations = 1

        for rotation_attempt in range(max_cookie_rotations + 1):
            cmd = base_cmd.copy()
            self._add_impersonation_to_cmd(cmd)
            cmd.extend(self._get_cookie_args())

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

                    logger.info(f"  ✓ Full video fallback success: {video_id}")
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
                    # Try cookie rotation on error
                    if self.rotate_cookie_on_error(result.stderr or ''):
                        logger.info(f"Cookie rotated for {video_id} fallback, retrying...")
                        time.sleep(2)
                        continue
                    logger.error(f"Full video fallback failed for {video_id}: {result.stderr[:200]}")
                    return []

            except subprocess.TimeoutExpired:
                logger.error(f"Full video fallback timeout for {video_id}")
                return []
            except Exception as e:
                logger.error(f"Full video fallback error for {video_id}: {e}")
                return []

        return []  # All attempts exhausted

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
