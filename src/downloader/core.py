"""
Streamlined VideoDownloader core orchestrator.

Refactored from monolithic downloader.py (2,906 lines) into modular architecture.
This core orchestrator (~650 lines) delegates to specialized modules:
- CheckpointManager: Checkpoint/resume and duration tier management
- TranscodingManager: FFmpeg transcoding and codec detection
- TitleFilter: LLM-based title filtering
- SpeechScreener: Whisper VAD speech detection
- SearchOptimizer: Download search optimization with adaptive pools and keyword alternatives
- AudioFirstPipeline: Audio-first download workflow
- segment_utils, utils: Helper functions

Original methods extracted: ~2,050 lines across 8 modules (71% reduction).
Core orchestration logic retained: download_all, download_for_keyword, _download_single,
_download_by_ids, _run_download_cmd (subprocess management, transcoding workflow).
"""

from __future__ import annotations

import os
import json
import subprocess
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

from ..config import Config, get_config
from ..state import DownloadedVideo

from .checkpoint import CheckpointManager, DownloadCheckpoint
from .transcoding import TranscodingManager
from .title_filter import TitleFilter
from .speech_screening import SpeechScreener
from .keyword_remix import SearchOptimizer
from .audio_first import AudioFirstPipeline
from . import utils

logger = logging.getLogger(__name__)


class VideoDownloader:
    """
    Advanced video downloader with modular architecture.

    Features:
    - Three duration tiers (short/medium/long/longer)
    - Parallel downloads with rate limiting
    - DaVinci Resolve transcoding (optional)
    - LLM title filtering (optional)
    - Speech screening for B-roll detection (optional)
    - Keyword remixing when searches fail
    - Audio-first pipeline for bandwidth savings
    - Checkpoint/resume support
    - Source attribution tracking
    """

    def __init__(self, config: Config = None):
        """
        Initialize VideoDownloader with all specialized managers.

        Args:
            config: Config object (defaults to get_config() if None)
        """
        self.config = config or get_config()
        self.download_config = self.config.download

        # Initialize checkpoint manager
        checkpoint_file = Path(self.config.cache_dir) / "download_checkpoint.json"
        sources_file = Path(self.config.downloaded_videos_dir) / "sources.json"

        self.checkpoint_mgr = CheckpointManager(
            config=self.config,
            checkpoint_file=checkpoint_file,
            sources_file=sources_file
        )

        # Initialize transcoding manager
        self.transcoding_mgr = TranscodingManager(self.config)

        # Initialize title filter (needs cookies and tier functions)
        cookies_args = utils.get_cookies_args(self.config)
        self.title_filter = TitleFilter(
            config=self.config,
            cookies_args=cookies_args,
            get_tier_value_func=self.checkpoint_mgr.get_tier_value
        )

        # Initialize speech screener
        self.speech_screener = SpeechScreener(
            config=self.config,
            cookies_args=cookies_args
        )

        # Initialize search optimizer (needs LLM functions from title_filter)
        self.search_optimizer = SearchOptimizer(
            config=self.config,
            llm_call_gemini_func=self.title_filter._call_gemini,
            llm_call_anthropic_func=self.title_filter._call_anthropic
        )

        # Initialize audio-first pipeline
        self._lock = threading.RLock()
        self.tier_download_counts: Dict[str, int] = {
            'short': 0, 'medium': 0, 'long': 0, 'longer': 0
        }

        self.audio_first = AudioFirstPipeline(
            config=self.config,
            get_tier_value_func=self.checkpoint_mgr.get_tier_value,
            search_metadata_func=self.title_filter.search_video_metadata,
            filter_titles_func=self.title_filter.filter_titles_with_llm,
            cleanup_partial_func=self._cleanup_partial_files,
            tier_download_counts=self.tier_download_counts,
            lock=self._lock
        )

        # Core state
        self.sources: List[DownloadedVideo] = self.checkpoint_mgr.load_sources()
        self.checkpoint: Optional[DownloadCheckpoint] = None
        self.DURATION_TIERS = self.checkpoint_mgr.duration_tiers
        self._last_download_timed_out = False

        # Cookie authentication
        self._cookies_from_browser = getattr(self.download_config, 'cookies_from_browser', '')
        self._cookies_path = self._find_cookies_file() if not self._cookies_from_browser else None

        if self._cookies_from_browser:
            logger.info(f"Using cookies from browser: {self._cookies_from_browser}")
        elif self._cookies_path:
            logger.info(f"Found cookies file: {self._cookies_path}")
        else:
            logger.warning("No cookies configured - YouTube downloads may fail!")

    # =========================================================================
    # DELEGATION METHODS (Delegate to specialized managers)
    # =========================================================================

    def _get_tier_value(self, tier: str, key: str, default: int = 0) -> int:
        """Delegate to CheckpointManager."""
        return self.checkpoint_mgr.get_tier_value(tier, key, default)

    def _load_checkpoint(self) -> Optional[DownloadCheckpoint]:
        """Delegate to CheckpointManager."""
        return self.checkpoint_mgr.load_checkpoint()

    def _save_checkpoint(self):
        """Delegate to CheckpointManager."""
        if self.checkpoint:
            self.checkpoint_mgr.save_checkpoint(self.checkpoint)

    def _clear_checkpoint(self):
        """Delegate to CheckpointManager."""
        self.checkpoint_mgr.clear_checkpoint()
        self.checkpoint = None

    def _save_sources(self):
        """Delegate to CheckpointManager."""
        self.checkpoint_mgr.save_sources(self.sources)

    def _search_video_metadata(self, keyword: str, tier: str, max_results: int = 50):
        """Delegate to TitleFilter."""
        return self.title_filter.search_video_metadata(keyword, tier, max_results)

    def _filter_titles_with_llm(self, videos, keyword, topic=""):
        """Delegate to TitleFilter."""
        return self.title_filter.filter_titles_with_llm(videos, keyword, topic)

    def _screen_approved_videos(self, approved_videos, keyword):
        """Delegate to SpeechScreener."""
        return self.speech_screener.screen_approved_videos(approved_videos, keyword)

    def _get_remix_keyword(self, keyword, topic=""):
        """Delegate to SearchOptimizer."""
        return self.search_optimizer.get_remix_keyword(keyword, topic)

    def _get_retry_keyword(self, keyword: str, retry_count: int) -> str:
        """Delegate to SearchOptimizer."""
        return self.search_optimizer.get_retry_keyword(keyword, retry_count)

    def _get_adaptive_search_pool(self, keyword, max_downloads):
        """Delegate to SearchOptimizer."""
        return self.search_optimizer.get_adaptive_search_pool(keyword, max_downloads)

    def _record_search_pass_rate(self, keyword, searched, approved):
        """Delegate to SearchOptimizer."""
        self.search_optimizer.record_search_pass_rate(keyword, searched, approved)

    def _record_source_for_keyword(self, keyword: str, video_id: str):
        """Delegate to SearchOptimizer."""
        self.search_optimizer.record_source_for_keyword(keyword, video_id)

    def log_source_diversity_report(self):
        """Delegate to SearchOptimizer."""
        self.search_optimizer.log_source_diversity_report()

    def _build_format_string(self):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.build_format_string()

    def _build_filter_string(self, tier):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.build_filter_string(tier, self.DURATION_TIERS)

    def _needs_transcoding(self, video_path):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.needs_transcoding(video_path)

    def _get_ffmpeg_transcode_cmd(self, input_path, output_path):
        """Delegate to TranscodingManager."""
        return self.transcoding_mgr.get_ffmpeg_transcode_cmd(input_path, output_path)

    # =========================================================================
    # CORE ORCHESTRATION METHODS (Keep in core - complex coordination logic)
    # =========================================================================

    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if yt-dlp and ffmpeg are installed.

        Returns:
            Tuple of (success: bool, message: str)
        """
        messages = []

        # Check yt-dlp
        try:
            result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
            messages.append(f"✓ yt-dlp {result.stdout.strip()}")
        except FileNotFoundError:
            return False, "✗ yt-dlp not found! Install with: pip install yt-dlp"

        # Check ffmpeg (required for DaVinci mode)
        if self.download_config.davinci_mode:
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                hw_accel = self.transcoding_mgr.hw_accel
                hw_names = {
                    'nvidia': 'NVIDIA NVENC',
                    'amd': 'AMD AMF',
                    'intel': 'Intel QuickSync',
                    'mac': 'Apple VideoToolbox',
                    'none': 'CPU'
                }
                messages.append(f"✓ FFmpeg ({hw_names.get(hw_accel, 'Unknown')})")
            except FileNotFoundError:
                return False, "✗ FFmpeg not found! Required for DaVinci mode."

        return True, "\n".join(messages)

    def _cleanup_partial_files(self, directory: Path, video_id: str) -> None:
        """
        Remove partial download files for a video ID.

        Cleans up .part, .ytdl, and other temporary files left by failed/timed out downloads.
        """
        if not directory.exists():
            return

        patterns = [f"{video_id}.*part*", f"{video_id}.*.ytdl", f"{video_id}.ytdl"]
        for pattern in patterns:
            for partial_file in directory.glob(pattern):
                try:
                    partial_file.unlink()
                    logger.debug(f"Cleaned up partial file: {partial_file.name}")
                except Exception as e:
                    logger.debug(f"Could not remove {partial_file.name}: {e}")

    def _find_cookies_file(self) -> Optional[Path]:
        """
        Find cookies.txt file for YouTube authentication.

        Searches in order:
        1. Explicit path from config (download.cookies_path)
        2. Install directory (same as config.yaml)
        3. Project directory
        4. Current working directory
        5. User home directory
        """
        search_locations = []

        # 0. Check if explicit path is set in config
        if hasattr(self.download_config, 'cookies_path') and self.download_config.cookies_path:
            explicit_path = Path(self.download_config.cookies_path)
            if explicit_path.exists():
                return explicit_path
            else:
                logger.warning(f"Configured cookies_path does not exist: {explicit_path}")

        # 1. Install directory (where config.yaml is)
        if hasattr(self.config, '_config_path') and self.config._config_path:
            install_dir = Path(self.config._config_path).parent
            search_locations.append(install_dir / 'cookies.txt')

        # 2. Project directory
        if hasattr(self.config, 'project_dir') and self.config.project_dir:
            search_locations.append(Path(self.config.project_dir) / 'cookies.txt')

        # 3. Current working directory
        search_locations.append(Path.cwd() / 'cookies.txt')

        # 4. User home directory
        search_locations.append(Path.home() / 'cookies.txt')

        for path in search_locations:
            if path.exists():
                return path

        return None

    def _add_cookies_to_cmd(self, cmd: list) -> None:
        """Add cookie authentication to yt-dlp command."""
        if self._cookies_from_browser:
            cmd.extend(['--cookies-from-browser', self._cookies_from_browser])
        elif self._cookies_path:
            cmd.extend(['--cookies', str(self._cookies_path)])

    def download_all(
        self,
        keywords: List[str],
        output_dir: Path,
        max_concurrent: int = 3,
        resume: bool = False,
        topic: str = ""
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Download videos for all keywords.

        Args:
            keywords: List of search keywords
            output_dir: Output directory
            max_concurrent: Max concurrent downloads (unused, sequential for now)
            resume: Whether to resume from checkpoint
            topic: Topic context for LLM title filtering

        Returns:
            Tuple of (downloaded_videos, failed_keywords)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scan existing downloads (file-based resume)
        existing_count = 0
        if output_dir.exists():
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    videos = [f for f in os.listdir(subdir) if f.endswith(('.mp4', '.mkv', '.webm'))]
                    existing_count += len(videos)
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing videos on disk (will skip)")

        # Load or create checkpoint
        if resume:
            self.checkpoint = self._load_checkpoint()
            if self.checkpoint:
                logger.info(f"Resuming from checkpoint ({len(self.checkpoint.completed_keywords)} keywords done)")
                # Filter out completed keywords
                keywords = [k for k in keywords if k not in self.checkpoint.completed_keywords]

        if not self.checkpoint:
            self.checkpoint = DownloadCheckpoint(
                completed_keywords=[],
                completed_videos=[],
                failed_keywords=[],
                current_keyword=None,
                current_tier=None,
                timestamp=datetime.now().isoformat()
            )

        all_downloaded = []
        failed_keywords = list(self.checkpoint.failed_keywords)

        # Log title blacklist if enabled
        title_blacklist = getattr(self.download_config, 'title_blacklist', [])
        if title_blacklist:
            logger.info(f"  Title blacklist: {len(title_blacklist)} terms (e.g., {', '.join(title_blacklist[:5])}...)")

        # Log LLM filter status
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        if llm_config and getattr(llm_config, 'enabled', False):
            provider = getattr(llm_config, 'provider', 'gemini')
            logger.info(f"  LLM title filter: enabled ({provider})")

        # Process keywords sequentially
        total_videos_downloaded = 0
        print(f"\n  Downloading videos for {len(keywords)} keywords...")

        for i, keyword in enumerate(keywords, 1):
            # Compact progress line
            progress_pct = (i - 1) / len(keywords) * 100
            print(f"\r  [{i}/{len(keywords)}] {progress_pct:5.1f}% | {keyword[:40]:<40} | Videos: {total_videos_downloaded}", end='', flush=True)

            logger.info(f"[{i}/{len(keywords)}] Processing: {keyword}")

            self.checkpoint.current_keyword = keyword
            self._save_checkpoint()

            downloaded = self.download_for_keyword(keyword, output_dir, topic=topic)

            if downloaded:
                all_downloaded.extend(downloaded)
                total_videos_downloaded += len(downloaded)
                self.checkpoint.completed_keywords.append(keyword)
            else:
                failed_keywords.append(keyword)
                self.checkpoint.failed_keywords.append(keyword)

            self._save_checkpoint()

            # Delay between keywords to avoid rate limiting
            if i < len(keywords):
                time.sleep(self.download_config.delay_between_keywords)

        # Final progress line
        print(f"\r  [{len(keywords)}/{len(keywords)}] 100.0% | Done{' ' * 50}")
        print(f"  ✓ Downloaded {total_videos_downloaded} videos from {len(keywords)} keywords")

        # Log inter-keyword source diversity report
        self.log_source_diversity_report()

        # Clear checkpoint on success
        self._clear_checkpoint()

        return all_downloaded, failed_keywords

    def download_for_keyword(
        self,
        keyword: str,
        output_dir: Path,
        tiers: List[str] = None,
        topic: str = ""
    ) -> List[DownloadedVideo]:
        """
        Download videos for a keyword across specified tiers.

        If download times out, will retry with modified keyword (up to 2 retries).

        Args:
            keyword: Search keyword
            output_dir: Output directory
            tiers: List of tier names (defaults to all tiers)
            topic: Topic context for LLM filtering

        Returns:
            List of DownloadedVideo objects
        """
        if tiers is None:
            tiers = list(self.DURATION_TIERS.keys())

        all_downloaded = []

        for tier in tiers:
            # Skip tiers with per_keyword=0
            per_kw = self._get_tier_value(tier, 'per_keyword', 5)
            if per_kw <= 0:
                logger.debug(f"  [{tier}] Skipped (0/kw)")
                continue

            # Check max_total limit for this tier
            max_total = self._get_tier_value(tier, 'max_total', 0)
            if max_total > 0 and self.tier_download_counts.get(tier, 0) >= max_total:
                logger.debug(f"  [{tier}] Skipped (max_total={max_total} reached)")
                continue

            # FILE-BASED SKIP: Check if videos already exist
            max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
            safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
            safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
            tier_short = tier[0]
            keyword_dir = output_dir / f"{safe_keyword}_{tier_short}"

            if keyword_dir.exists():
                existing_videos = [f for f in os.listdir(keyword_dir)
                                   if f.endswith(('.mp4', '.mkv', '.webm'))]
                per_kw = self._get_tier_value(tier, 'per_keyword', 5)
                if len(existing_videos) >= per_kw:
                    logger.debug(f"  [{tier}] Already have {len(existing_videos)} videos (skipping)")
                    # Count existing toward tier total
                    with self._lock:
                        self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(existing_videos)
                    continue
                elif existing_videos:
                    logger.debug(f"  [{tier}] Found {len(existing_videos)} existing, need {per_kw - len(existing_videos)} more")

            # Checkpoint-based skip (secondary check)
            if self.checkpoint:
                video_key = f"{keyword}|{tier}"
                if video_key in self.checkpoint.completed_videos:
                    logger.debug(f"Skipping {keyword} ({tier}) - checkpoint says completed")
                    continue

            logger.debug(f"  [{tier}] Downloading...")
            downloaded = self._download_single(keyword, tier, output_dir, topic)

            # Check for timeout and retry with modified keywords
            retry_attempt = 0
            while not downloaded and getattr(self, '_last_download_timed_out', False) and retry_attempt < 2:
                alt_keyword = self._get_retry_keyword(keyword, retry_attempt)
                if alt_keyword and alt_keyword != keyword:
                    logger.debug(f"  [{tier}] Timeout - retrying with: '{alt_keyword}'")
                    downloaded = self._download_single(alt_keyword, tier, output_dir, topic)
                    retry_attempt += 1
                else:
                    break

            if downloaded:
                logger.debug(f"  [{tier}] ✓ {len(downloaded)} video(s)")
                all_downloaded.extend(downloaded)

                # Update tier download count
                with self._lock:
                    self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(downloaded)

                # Update sources
                with self._lock:
                    self.sources.extend(downloaded)
                    self._save_sources()
            else:
                # ZERO-DOWNLOAD REMIX: Try LLM-based keyword remix when 0 results
                remix_keyword = self._get_remix_keyword(keyword, topic)
                if remix_keyword and remix_keyword != keyword:
                    logger.info(f"  [{tier}] No results - trying remix: '{remix_keyword}'")
                    downloaded = self._download_single(remix_keyword, tier, output_dir, topic)
                    if downloaded:
                        logger.debug(f"  [{tier}] ✓ Remix success: {len(downloaded)} video(s)")
                        all_downloaded.extend(downloaded)
                        with self._lock:
                            self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(downloaded)
                            self.sources.extend(downloaded)
                            self._save_sources()
                    else:
                        logger.debug(f"  [{tier}] Remix also returned no results")
                else:
                    logger.debug(f"  [{tier}] No results")

            # Update checkpoint
            if self.checkpoint:
                logger.debug(f"  [{tier}] Saving checkpoint...")
                self.checkpoint.completed_videos.append(f"{keyword}|{tier}")
                self._save_checkpoint()
                logger.debug(f"  [{tier}] Checkpoint saved")

        return all_downloaded

    def _download_single(
        self,
        keyword: str,
        tier: str,
        output_dir: Path,
        topic: str = ""
    ) -> List[DownloadedVideo]:
        """
        Download videos for a single keyword and tier with optional LLM filtering.

        Implements two flows:
        1. LLM filter enabled: Search metadata → LLM filter → Speech screen → Download by IDs
        2. LLM filter disabled: Direct yt-dlp search with built-in filters

        Args:
            keyword: Search keyword
            tier: Duration tier
            output_dir: Output directory
            topic: Topic context for LLM

        Returns:
            List of DownloadedVideo objects
        """
        max_downloads = self._get_tier_value(tier, 'per_keyword', 5)

        # ADAPTIVE SEARCH POOL: Adjust based on historical pass rates
        search_pool = self._get_adaptive_search_pool(keyword, max_downloads)

        # Get path length settings from config
        max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)

        # Create keyword subdirectory
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
        safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
        tier_short = tier[0]
        keyword_dir = output_dir / f"{safe_keyword}_{tier_short}"
        keyword_dir.mkdir(parents=True, exist_ok=True)

        # Get existing files
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()

        # Check if LLM filtering is enabled
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        use_llm_filter = llm_config and getattr(llm_config, 'enabled', False)

        if use_llm_filter:
            # NEW FLOW: Search metadata first, filter with LLM, then download specific videos
            logger.debug(f"    Searching {search_pool} videos for LLM filtering...")

            videos = self._search_video_metadata(keyword, tier, search_pool)

            if not videos:
                logger.debug(f"    No videos found for '{keyword}'")
                return []

            logger.debug(f"    Found {len(videos)} candidate videos")

            # Apply blacklist filter first (fast, no API cost)
            title_blacklist = getattr(self.download_config, 'title_blacklist', [])
            if title_blacklist:
                before_count = len(videos)
                videos = [v for v in videos if not any(
                    term.lower() in v['title'].lower() for term in title_blacklist
                )]
                if before_count > len(videos):
                    logger.debug(f"    Blacklist filter: {len(videos)}/{before_count} passed")

            # Apply LLM filter
            approved_videos = self._filter_titles_with_llm(videos[:search_pool], keyword, topic)

            # Track pass rate for adaptive pool sizing
            searched_count = min(len(videos), search_pool)
            approved_count = len(approved_videos) if approved_videos else 0
            self._record_search_pass_rate(keyword, searched_count, approved_count)

            if not approved_videos:
                logger.debug(f"    No videos passed LLM filter for '{keyword}'")
                return []

            # Speech screening: filter out videos with speech
            speech_config = getattr(self.download_config, 'speech_screening', None)
            speech_enabled = speech_config and getattr(speech_config, 'enabled', False)
            speech_tiers = getattr(speech_config, 'tiers', ['long', 'longer']) if speech_config else []

            if speech_enabled and tier in speech_tiers:
                logger.info(f"[SPEECH SCREEN] Tier '{tier}' - screening {len(approved_videos)} approved videos...")
                approved_videos = self._screen_approved_videos(approved_videos, keyword)
                logger.info(f"[SPEECH SCREEN] After screening: {len(approved_videos)} videos remain")

                if not approved_videos:
                    logger.debug(f"    No videos passed speech screening for '{keyword}'")
                    return []

            # Download only approved videos (by ID)
            video_ids = [v['id'] for v in approved_videos[:max_downloads]]
            logger.debug(f"    Downloading {len(video_ids)} approved videos...")

            return self._download_by_ids(video_ids, keyword_dir, output_dir, keyword, tier)

        else:
            # ORIGINAL FLOW: Direct search and download with yt-dlp filters
            cmd = [
                'yt-dlp',
                f'ytsearch{search_pool}:{keyword}',
                '-f', self._build_format_string(),
                '--match-filter', self._build_filter_string(tier),
                '--max-downloads', str(max_downloads),
                '--merge-output-format', 'mp4',
                '--no-playlist',
                '--write-info-json',
                '--restrict-filenames',
                '--no-overwrites',
                '--no-continue',
                '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
                '--progress',
                '--newline',
                '--quiet',
                '--no-warnings',
            ]

            self._add_cookies_to_cmd(cmd)

            logger.debug(f"    Search: ytsearch{search_pool}, Max: {max_downloads}")

            return self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)

    def _download_by_ids(
        self,
        video_ids: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str
    ) -> List[DownloadedVideo]:
        """
        Download specific videos by their YouTube IDs.

        Automatically skips videos that already exist on disk (crash-resilient).

        Args:
            video_ids: List of YouTube video IDs
            keyword_dir: Keyword-specific directory
            output_dir: Base output directory
            keyword: Search keyword
            tier: Duration tier

        Returns:
            List of DownloadedVideo objects (including already-downloaded ones)
        """
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()

        # Check which video IDs are already downloaded (file-based, not checkpoint-based)
        already_downloaded = []
        missing_ids = []

        for vid_id in video_ids:
            # Check if any file contains this video ID
            found = False
            for existing_file in existing_before:
                if vid_id in existing_file and existing_file.endswith(('.mp4', '.mkv', '.webm')):
                    found = True
                    logger.debug(f"    Skipping {vid_id} - already exists: {existing_file}")
                    # Create DownloadedVideo for existing file
                    video_path = keyword_dir / existing_file
                    already_downloaded.append(DownloadedVideo(
                        file=str(video_path.relative_to(output_dir)),
                        url=f"https://www.youtube.com/watch?v={vid_id}",
                        title=existing_file,
                        channel="",
                        upload_date="",
                        duration=0,
                        duration_tier=tier,
                        keyword=keyword,
                        download_date="",
                        license="Unknown"
                    ))
                    break
            if not found:
                missing_ids.append(vid_id)

        if already_downloaded:
            logger.debug(f"    {len(already_downloaded)} already downloaded, {len(missing_ids)} to fetch")

        if not missing_ids:
            # All videos already exist
            return already_downloaded

        # Get filename length from config
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)

        # Build URLs from IDs
        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in missing_ids]

        cmd = [
            'yt-dlp',
            '-f', self._build_format_string(),
            '--merge-output-format', 'mp4',
            '--no-playlist',
            '--write-info-json',
            '--restrict-filenames',
            '--no-overwrites',
            '--no-continue',
            '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
            '--quiet',
            '--no-warnings',
            '--progress',
        ] + urls

        self._add_cookies_to_cmd(cmd)

        newly_downloaded = self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)

        # Combine already downloaded + newly downloaded
        return already_downloaded + newly_downloaded

    def _run_download_cmd(
        self,
        cmd: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set,
        timeout_override: int = None
    ) -> List[DownloadedVideo]:
        """
        Execute download command and process results.

        This is the most complex method in the core orchestrator (199 lines in original).
        Handles:
        - Subprocess execution with timeout
        - Cleanup of partial downloads
        - Transcoding if DaVinci mode enabled
        - Metadata extraction from info.json
        - Filename sanitization for NLE compatibility
        - Source tracking for diversity analysis

        Args:
            cmd: yt-dlp command to execute
            keyword_dir: Keyword-specific directory
            output_dir: Base output directory
            keyword: Search keyword
            tier: Duration tier
            existing_before: Set of files that existed before download
            timeout_override: Optional timeout override

        Returns:
            List of DownloadedVideo objects, or empty list on failure/timeout.
            Sets self._last_download_timed_out = True if timeout occurred.
        """
        # Clean up any partial downloads from previous crashes
        if keyword_dir.exists():
            for part_file in keyword_dir.glob('*.part'):
                try:
                    part_file.unlink()
                    logger.debug(f"Cleaned up partial download: {part_file.name}")
                except Exception:
                    pass
            for ytdl_file in keyword_dir.glob('*.ytdl'):
                try:
                    ytdl_file.unlink()
                except Exception:
                    pass

        # Use override, tier-specific timeout, config default, or fallback
        if timeout_override:
            download_timeout = timeout_override
        else:
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            if tier and tier in tier_timeouts:
                download_timeout = tier_timeouts[tier]
            else:
                download_timeout = getattr(self.download_config, 'download_timeout', 120)

        # Track timeout for retry logic
        self._last_download_timed_out = False
        process = None

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=download_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                logger.warning(f"Timeout downloading '{keyword}' ({tier}) after {download_timeout}s")
                self._last_download_timed_out = True
                return []
            finally:
                # Ensure process is cleaned up
                if process is not None and process.poll() is None:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass

            # Only log actual errors
            if stderr:
                for line in stderr.strip().split('\n'):
                    if line and 'WARNING' not in line and 'ERROR' in line:
                        logger.warning(f"    yt-dlp: {line}")

            # Find new files
            existing_after = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
            new_files = existing_after - existing_before

            # Filter to video files
            video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
            new_videos = [f for f in new_files if Path(f).suffix.lower() in video_extensions]

            if new_videos:
                logger.debug(f"    Downloaded {len(new_videos)} video(s)")

            downloaded = []

            for idx, video_file in enumerate(new_videos, 1):
                video_path = keyword_dir / video_file

                # Try to get metadata from info.json
                info_file = video_path.with_suffix('.info.json')
                metadata = {}
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass

                # Transcode for DaVinci if enabled AND necessary
                final_path = video_path
                if self.download_config.davinci_mode:
                    needs_transcode, reason = self._needs_transcoding(str(video_path))

                    if not needs_transcode:
                        logger.debug(f"    No transcode needed: {reason}")
                        final_path = video_path
                    else:
                        logger.debug(f"    Transcoding {video_file[:40]}...")
                        transcode_cmd, output_path = self._get_ffmpeg_transcode_cmd(
                            str(video_path), str(video_path)
                        )
                        temp_output = Path(output_path).with_stem(Path(output_path).stem + '_davinci')
                        transcode_cmd[-1] = str(temp_output)

                        transcode_process = None
                        try:
                            transcode_process = subprocess.Popen(
                                transcode_cmd,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                text=True
                            )

                            try:
                                _, stderr = transcode_process.communicate(timeout=download_timeout)
                                if transcode_process.returncode != 0:
                                    logger.warning(f"FFmpeg error: {stderr[-500:] if stderr else 'unknown'}")
                            except subprocess.TimeoutExpired:
                                transcode_process.kill()
                                transcode_process.communicate()
                                logger.warning(f"Transcode timeout for {video_file}")
                            finally:
                                if transcode_process is not None and transcode_process.poll() is None:
                                    transcode_process.kill()
                                    try:
                                        transcode_process.wait(timeout=5)
                                    except subprocess.TimeoutExpired:
                                        pass

                            if temp_output.exists() and temp_output.stat().st_size > 0:
                                if self.download_config.delete_original:
                                    video_path.unlink()
                                final_path = temp_output.rename(temp_output.with_stem(
                                    temp_output.stem.replace('_davinci', '')
                                ))
                                logger.info(f"    ↳ ✓ Transcode complete")
                            else:
                                logger.warning(f"    ↳ Transcode produced no output, using original")
                                final_path = video_path
                        except Exception as e:
                            logger.warning(f"Transcode failed for {video_file}: {e}")
                            final_path = video_path

                # Sanitize filename for NLE compatibility
                final_path = utils.sanitize_filename_for_nle(final_path)

                # Create source record
                source = DownloadedVideo(
                    file=str(final_path.relative_to(output_dir)),
                    url=metadata.get('webpage_url', metadata.get('url', 'Unknown')),
                    title=metadata.get('title', video_file),
                    channel=metadata.get('uploader', metadata.get('channel', 'Unknown')),
                    upload_date=metadata.get('upload_date', 'Unknown'),
                    duration=metadata.get('duration', 0),
                    duration_tier=tier,
                    keyword=keyword,
                    download_date=datetime.now().strftime('%Y-%m-%d'),
                    license=metadata.get('license', 'Unknown')
                )

                downloaded.append(source)

                # Track source for inter-keyword diversity analysis
                video_id = metadata.get('id', '')
                if video_id:
                    self._record_source_for_keyword(keyword, video_id)

                # Clean up info.json
                if info_file.exists():
                    info_file.unlink()

            return downloaded

        except Exception as e:
            logger.error(f"Error downloading '{keyword}' ({tier}): {e}")
            return []

    # =========================================================================
    # REPORTING METHODS
    # =========================================================================

    def get_download_estimate(self, num_keywords: int) -> Dict:
        """
        Estimate download stats.

        Args:
            num_keywords: Number of keywords to download

        Returns:
            Dict with estimates (videos, storage, time)
        """
        videos_per_keyword = sum(
            self._get_tier_value(t, 'per_keyword', 0)
            for t in self.DURATION_TIERS.keys()
        )
        total_videos = num_keywords * videos_per_keyword

        # Rough estimates
        avg_size_mb = 150
        avg_download_time = 30

        return {
            'keywords': num_keywords,
            'videos_per_keyword': videos_per_keyword,
            'total_videos': total_videos,
            'est_storage_gb': round(total_videos * avg_size_mb / 1024, 1),
            'est_time_minutes': round(total_videos * avg_download_time / 60, 0),
            'tiers': {
                name: f"{self._get_tier_value(name, 'min', 0)}s-{self._get_tier_value(name, 'max', 120)}s ({self._get_tier_value(name, 'per_keyword', 5)}/kw)"
                for name in self.DURATION_TIERS.keys()
            }
        }

    def get_inventory_report(self, output_dir: Path) -> Dict:
        """
        Report on existing footage.

        Args:
            output_dir: Output directory to scan

        Returns:
            Dict with inventory stats
        """
        output_dir = Path(output_dir)

        if not output_dir.exists():
            return {'total_videos': 0, 'total_duration': 0, 'keywords_covered': []}

        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
        videos = []

        for ext in video_extensions:
            videos.extend(output_dir.rglob(f'*{ext}'))

        # Get unique keywords from directory names
        keywords_covered = set()
        for video in videos:
            parent = video.parent.name
            # Extract keyword from dir name (remove tier suffix)
            for tier in self.DURATION_TIERS.keys():
                if parent.endswith(f'_{tier[0]}'):  # Match tier short code
                    keyword = parent[:-2]  # Remove _s, _m, _l
                    keywords_covered.add(keyword)
                    break

        # Calculate total duration from sources.json
        total_duration = sum(s.duration for s in self.sources) if self.sources else 0

        return {
            'total_videos': len(videos),
            'total_duration_hours': round(total_duration / 3600, 2),
            'keywords_covered': list(keywords_covered),
            'num_keywords_covered': len(keywords_covered)
        }
