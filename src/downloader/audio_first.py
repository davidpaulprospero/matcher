"""
Audio-first download pipeline.

Phase 1: Download audio only
Phase 3: Download matched video segments

Migrated from VideoDownloader audio-first methods (lines 2078-2556).
"""

from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from ..state import AudioDownload
from .types import MergedSegment, DownloadedSegment
from . import segment_utils
from . import utils

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


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

            # Add cookies
            cmd.extend(utils.get_cookies_args(self.config))

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

            except subprocess.TimeoutExpired:
                logger.warning(f"Audio download timeout for {video_id}")
                self._cleanup_partial_files(audio_dir, video_id)
            except Exception as e:
                logger.warning(f"Audio download error for {video_id}: {e}")
                self._cleanup_partial_files(audio_dir, video_id)

        # Update tier download count
        if audio_downloads:
            with self._lock:
                self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(audio_downloads)

        logger.info(f"  Downloaded {len(audio_downloads)} audio files for '{keyword}' ({tier})")
        return audio_downloads

    def download_video_segments(
        self,
        merged_segments: List[MergedSegment],
        output_dir: Path
    ) -> List[DownloadedSegment]:
        """
        Download video segments using --download-sections.

        Migrated from downloader.py lines 2284-2429.

        Phase 3 of audio-first pipeline. Downloads only the matched portions
        of videos, not the full files.

        Args:
            merged_segments: List of merged segments with buffer applied
            output_dir: Base output directory

        Returns:
            List of DownloadedSegment records with timing info
        """
        audio_config = getattr(self.download_config, 'audio_first', None)
        fallback_full = getattr(audio_config, 'fallback_full_video', True) if audio_config else True

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

            # Build --download-sections arguments
            section_args = []
            for seg in segments:
                start_str = utils.format_time(seg.start_time)
                end_str = utils.format_time(seg.end_time)
                section_args.extend(['--download-sections', f'*{start_str}-{end_str}'])

            # Build yt-dlp command
            cmd = [
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
                cmd.extend(['--ffmpeg-location', ffmpeg_loc])

            # Add cookies
            cmd.extend(utils.get_cookies_args(self.config))

            # Get timeout
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            timeout = tier_timeouts.get('long', 600)

            segment_success = False
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode != 0:
                    print(f"      ✗ Failed: {result.stderr[-200:] if result.stderr else 'Unknown error'}")
                    logger.warning(f"Segment download failed for {video_id}: {result.stderr[:200]}")
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

            except subprocess.TimeoutExpired:
                print(f"      ✗ Timeout after {timeout}s")
                logger.warning(f"Segment download timeout for {video_id}")
            except Exception as e:
                print(f"      ✗ Error: {e}")
                logger.error(f"Segment download error for {video_id}: {e}")

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

        cmd = [
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
            cmd.extend(['--ffmpeg-location', ffmpeg_loc])

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
                logger.error(f"Full video fallback failed for {video_id}: {result.stderr[:200]}")
                return []

        except subprocess.TimeoutExpired:
            logger.error(f"Full video fallback timeout for {video_id}")
            return []
        except Exception as e:
            logger.error(f"Full video fallback error for {video_id}: {e}")
            return []

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
