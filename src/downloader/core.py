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
from .search_cache import YouTubeSearchCache
from . import utils

logger = logging.getLogger(__name__)


def _safe_print(*args, **kwargs):
    """Print with flush, but handle Windows OSError when stdout is redirected."""
    try:
        print(*args, **kwargs)
    except OSError:
        # Windows can throw OSError: [Errno 22] Invalid argument
        # when stdout is redirected and flush=True is used
        pass


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

        # Initialize search cache for YouTube search result persistence
        self.search_cache = YouTubeSearchCache(cache_dir=self.config.cache_dir)

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

        # Healer integration (set via set_healer())
        self._download_healer = None

        # Global video ID deduplication across all keywords
        # Prevents downloading same video for different keywords (saves disk space + improves variety)
        self._downloaded_video_ids: Set[str] = set()
        self._init_downloaded_video_ids()

        # Cookie authentication
        self._cookies_from_browser = getattr(self.download_config, 'cookies_from_browser', '')
        self._cookies_path = self._find_cookies_file() if not self._cookies_from_browser else None

        if self._cookies_from_browser:
            logger.info(f"Using cookies from browser: {self._cookies_from_browser}")
        elif self._cookies_path:
            logger.info(f"Found cookies file: {self._cookies_path}")
        else:
            # No legacy cookies - ensure bypass tier uses browser cookies (Tier 4)
            bypass_config = getattr(self.download_config, 'rate_limit_bypass', None)
            if bypass_config:
                start_tier = getattr(bypass_config, 'start_tier', 1)
                if isinstance(bypass_config, dict):
                    start_tier = bypass_config.get('start_tier', 1)
                    current_tier = bypass_config.get('_current_tier', 1)
                    if current_tier < 4 and start_tier >= 4:
                        bypass_config['_current_tier'] = start_tier
                        logger.info(f"Forcing bypass tier {start_tier} (browser cookies)")
                    elif current_tier < 4:
                        bypass_config['_current_tier'] = 4
                        logger.info("No cookies configured - forcing bypass Tier 4 (browser cookies)")
                else:
                    current_tier = getattr(bypass_config, '_current_tier', 1)
                    if current_tier < 4 and start_tier >= 4:
                        bypass_config._current_tier = start_tier
                        logger.info(f"Forcing bypass tier {start_tier} (browser cookies)")
                    elif current_tier < 4:
                        bypass_config._current_tier = 4
                        logger.info("No cookies configured - forcing bypass Tier 4 (browser cookies)")
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

    def _init_downloaded_video_ids(self):
        """
        Initialize the set of already-downloaded video IDs by scanning existing files.

        Scans the download directory for video files and extracts YouTube video IDs
        from filenames (pattern: {title}_{video_id}.mp4).

        This enables cross-keyword deduplication: if a video was downloaded for
        keyword A, it won't be re-downloaded for keyword B.
        """
        import re

        # Get the root video directory
        video_dir = Path(self.config.downloaded_videos_dir)
        if not video_dir.exists():
            return

        # Scan for video files and extract video IDs
        video_extensions = {'.mp4', '.mkv', '.webm', '.m4v'}
        count = 0

        for video_file in video_dir.rglob('*'):
            if video_file.suffix.lower() not in video_extensions:
                continue

            filename = video_file.stem

            # Pattern 1: Audio-first segment: {video_id}_{offset:04d}
            match = re.match(r'^([a-zA-Z0-9_-]{11})_(\d{4})$', filename)
            if match:
                self._downloaded_video_ids.add(match.group(1))
                count += 1
                continue

            # Pattern 2: Regular YouTube: {title}_{video_id}
            match = re.search(r'_([a-zA-Z0-9_-]{11})$', filename)
            if match:
                self._downloaded_video_ids.add(match.group(1))
                count += 1

        if count > 0:
            logger.info(f"Initialized video ID tracker: {len(self._downloaded_video_ids)} unique videos from {count} files")

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
    # HEALER INTEGRATION
    # =========================================================================

    def set_healer(self, healer):
        """Set healer reference for success callbacks and tier management.

        Args:
            healer: DownloadHealer instance from self-healing system
        """
        self._download_healer = healer
        logger.debug(f"Download healer attached: {healer.name if healer else None}")

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
        """Add base args (JS runtime), bypass args, and cookie authentication to yt-dlp command."""
        # Add JS runtime for YouTube challenge solving
        cmd.extend(['--js-runtimes', 'node'])

        # Add bypass args (handles tier-specific authentication)
        bypass_args = self._build_bypass_args()
        cmd.extend(bypass_args)

        # Skip cookies if using impersonation (Tier 1) or browser auth (Tier 2)
        # Bypass args already handle authentication for those tiers
        if '--no-cookies' in bypass_args or '--cookies-from-browser' in bypass_args:
            return

        # Tier 3 or bypass disabled: use legacy cookie auth
        if self._cookies_from_browser:
            cmd.extend(['--cookies-from-browser', self._cookies_from_browser])
        elif self._cookies_path:
            cmd.extend(['--cookies', str(self._cookies_path)])

    def _build_bypass_args(self, for_subtitles: bool = False) -> List[str]:
        """Build yt-dlp arguments for current bypass tier.

        Args:
            for_subtitles: If True, use subtitle-optimized settings (tv_embedded)

        Returns list of arguments to add to yt-dlp command.
        Tier state is managed by RateLimitBypassConfig (set by healer).

        Tier order:
        1. Impersonate Chrome (fastest, no auth)
        2. Impersonate Safari (different fingerprint)
        3. tv_embedded player (best for subtitles)
        4. Browser cookies + web player
        5. ios_creator player
        6. android_vr player
        7. Standard yt-dlp (fallback)
        """
        args = []

        bypass_config = getattr(self.download_config, 'rate_limit_bypass', None)
        if not bypass_config:
            logger.debug("[BYPASS] No bypass config, using standard yt-dlp")
            return args

        # Handle both dataclass and dict (for backwards compatibility)
        if isinstance(bypass_config, dict):
            current_tier = bypass_config.get('_current_tier', 1)
            tier1_enabled = bypass_config.get('tier1_enabled', True)
            tier1_target = bypass_config.get('tier1_target', 'Chrome-131:Android-14')
            tier2_enabled = bypass_config.get('tier2_enabled', True)
            tier2_target = bypass_config.get('tier2_target', 'Safari-18.2:macOS-15')
            tier3_enabled = bypass_config.get('tier3_enabled', True)
            tier3_player = bypass_config.get('tier3_player', 'tv_embedded')
            tier4_enabled = bypass_config.get('tier4_enabled', True)
            tier4_browser = bypass_config.get('tier4_browser', 'firefox')
            tier4_player = bypass_config.get('tier4_player', 'web')
            tier5_enabled = bypass_config.get('tier5_enabled', True)
            tier5_player = bypass_config.get('tier5_player', 'ios_creator')
            tier6_enabled = bypass_config.get('tier6_enabled', True)
            tier6_player = bypass_config.get('tier6_player', 'android_vr')
            subtitle_tv_embedded = bypass_config.get('subtitle_always_tv_embedded', True)
        else:
            current_tier = bypass_config._current_tier
            tier1_enabled = bypass_config.tier1_enabled
            tier1_target = bypass_config.tier1_target
            tier2_enabled = bypass_config.tier2_enabled
            tier2_target = bypass_config.tier2_target
            tier3_enabled = bypass_config.tier3_enabled
            tier3_player = bypass_config.tier3_player
            tier4_enabled = bypass_config.tier4_enabled
            tier4_browser = bypass_config.tier4_browser
            tier4_player = bypass_config.tier4_player
            tier5_enabled = bypass_config.tier5_enabled
            tier5_player = bypass_config.tier5_player
            tier6_enabled = bypass_config.tier6_enabled
            tier6_player = bypass_config.tier6_player
            subtitle_tv_embedded = bypass_config.subtitle_always_tv_embedded

        # For subtitles, always prefer tv_embedded (no PO Token needed)
        if for_subtitles and subtitle_tv_embedded:
            logger.debug("[BYPASS] Subtitle mode: using tv_embedded player")
            args.extend(['--extractor-args', 'youtube:player_client=tv_embedded'])
            return args

        # Tier 1: Chrome impersonation
        if current_tier == 1 and tier1_enabled:
            logger.debug(f"[BYPASS] Tier 1: --impersonate {tier1_target}")
            args.extend(['--impersonate', tier1_target])
            args.append('--no-cookies')

        # Tier 2: Safari impersonation
        elif current_tier == 2 and tier2_enabled:
            logger.debug(f"[BYPASS] Tier 2: --impersonate {tier2_target}")
            args.extend(['--impersonate', tier2_target])
            args.append('--no-cookies')

        # Tier 3: tv_embedded player (no auth, works for subtitles)
        elif current_tier == 3 and tier3_enabled:
            logger.debug(f"[BYPASS] Tier 3: player_client={tier3_player}")
            args.extend(['--extractor-args', f'youtube:player_client={tier3_player}'])

        # Tier 4: Browser cookies + web player
        elif current_tier == 4 and tier4_enabled:
            logger.debug(f"[BYPASS] Tier 4: cookies from {tier4_browser}, player={tier4_player}")
            args.extend(['--cookies-from-browser', tier4_browser])
            args.extend(['--extractor-args', f'youtube:player_client={tier4_player}'])

        # Tier 5: ios_creator player
        elif current_tier == 5 and tier5_enabled:
            logger.debug(f"[BYPASS] Tier 5: player_client={tier5_player}")
            args.extend(['--extractor-args', f'youtube:player_client={tier5_player}'])

        # Tier 6: android_vr player
        elif current_tier == 6 and tier6_enabled:
            logger.debug(f"[BYPASS] Tier 6: player_client={tier6_player}")
            args.extend(['--extractor-args', f'youtube:player_client={tier6_player}'])

        # Tier 7: Standard yt-dlp
        else:
            logger.debug("[BYPASS] Tier 7: Standard yt-dlp (no impersonation or special player)")

        return args

    def _escalate_bypass_tier(self, reason: str = "unknown") -> bool:
        """Escalate to next bypass tier after connection hang or auth failure.

        Args:
            reason: Why escalation is happening (for logging)

        Returns:
            True if escalated, False if already at max tier
        """
        bypass_config = getattr(self.download_config, 'rate_limit_bypass', None)
        if not bypass_config:
            logger.debug("[BYPASS] No bypass config, cannot escalate")
            return False

        # Get current tier and max tier
        if isinstance(bypass_config, dict):
            current_tier = bypass_config.get('_current_tier', 1)
            max_tier = bypass_config.get('max_tier', 3)
        else:
            current_tier = bypass_config._current_tier
            max_tier = getattr(bypass_config, 'max_tier', 3)

        if current_tier >= max_tier:
            logger.warning(f"[BYPASS] Already at max tier {max_tier}, cannot escalate further")
            return False

        # Escalate
        new_tier = current_tier + 1
        if isinstance(bypass_config, dict):
            bypass_config['_current_tier'] = new_tier
        else:
            bypass_config._current_tier = new_tier

        tier_names = {
            1: "Chrome Impersonate",
            2: "Safari Impersonate",
            3: "tv_embedded",
            4: "Browser Cookies",
            5: "ios_creator",
            6: "android_vr",
            7: "Standard"
        }
        logger.warning(
            f"[BYPASS] Tier escalation: {current_tier} ({tier_names.get(current_tier, '?')}) → "
            f"{new_tier} ({tier_names.get(new_tier, '?')}) [reason: {reason}]"
        )

        # Notify healer if attached (for statistics tracking)
        if self._download_healer:
            self._download_healer.escalation_history.append({
                'from_tier': current_tier,
                'to_tier': new_tier,
                'reason': reason,
                'source': 'downloader_direct'
            })

        return True

    def _retry_with_full_timeout(
        self,
        cmd: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set,
        full_timeout: int
    ) -> List[DownloadedVideo]:
        """Retry a slow download with the full timeout instead of first-byte timeout.

        Called when first-byte timeout expired but output was being produced,
        indicating a slow but working download rather than a connection hang.
        """
        logger.debug(f"    Retrying '{keyword}' with full {full_timeout}s timeout...")

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
                stdout, stderr = process.communicate(timeout=full_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""

                logger.warning(f"Download timeout for '{keyword}' even with full {full_timeout}s timeout")
                self._last_download_timed_out = True
                return []
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass

            # Process completed - continue with normal post-download handling
            # (This duplicates some code from _run_download_cmd, but keeps the logic clear)
            if stderr:
                for line in stderr.strip().split('\n'):
                    if line and 'WARNING' not in line and 'ERROR' in line:
                        logger.warning(f"    yt-dlp: {line}")

            # Find new files and process them
            return self._process_downloaded_files(
                keyword_dir, output_dir, keyword, tier, existing_before
            )

        except Exception as e:
            logger.error(f"Error in retry download for '{keyword}': {e}")
            return []

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
            _safe_print(f"\r  [{i}/{len(keywords)}] {progress_pct:5.1f}% | {keyword[:40]:<40} | Videos: {total_videos_downloaded}", end='', flush=True)

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

            # Check search cache for previously approved video IDs
            cached_ids = self.search_cache.get(keyword, tier, max_age_days=30)

            if cached_ids:
                # Cache HIT: Use cached video IDs (skip search + LLM filter)
                logger.info(f"    Using {len(cached_ids)} cached video IDs for '{keyword}' (tier: {tier})")
                video_ids = cached_ids[:max_downloads]
                return self._download_by_ids(video_ids, keyword_dir, output_dir, keyword, tier)

            # Cache MISS: Do full search + filter pipeline
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

            # Cache approved video IDs for future runs
            video_ids = [v['id'] for v in approved_videos[:max_downloads]]
            self.search_cache.set(keyword, tier, video_ids, search_pool=search_pool)

            # Download only approved videos (by ID)
            logger.debug(f"    Downloading {len(video_ids)} approved videos...")

            return self._download_by_ids(video_ids, keyword_dir, output_dir, keyword, tier)

        else:
            # ORIGINAL FLOW: Direct search and download with yt-dlp filters
            cmd = [
                'yt-dlp',
                f'ytsearch{search_pool}:{keyword}',
                '--sleep-interval', '5',
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

    def _validate_video_ids(self, video_ids: List[str]) -> List[str]:
        """
        Validate video IDs before download by checking accessibility.

        Uses yt-dlp's --skip-download + info extraction to quickly verify
        each video is accessible without downloading. Removes invalid IDs
        (deleted, private, geoblocked, age-gated, etc.) from cache.

        Args:
            video_ids: List of YouTube video IDs to validate

        Returns:
            List of valid video IDs (accessible and downloadable)
        """
        if not video_ids:
            return []

        valid_ids = []
        validation_timeout = getattr(self.download_config, 'validation_timeout', 30)

        logger.debug(f"    Validating {len(video_ids)} cached video IDs...")

        for vid_id in video_ids:
            url = f"https://www.youtube.com/watch?v={vid_id}"

            cmd = [
                'yt-dlp',
                '--skip-download',
                '--dump-json',
                '--quiet',
                '--no-warnings',
                url
            ]

            self._add_cookies_to_cmd(cmd)

            try:
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                try:
                    stdout, stderr = process.communicate(timeout=validation_timeout)

                    # If we got JSON output, video is accessible
                    if stdout and process.returncode == 0:
                        try:
                            json.loads(stdout)
                            valid_ids.append(vid_id)
                            logger.debug(f"      ✓ {vid_id} - valid")
                        except json.JSONDecodeError:
                            logger.debug(f"      ✗ {vid_id} - invalid JSON response")
                    else:
                        logger.debug(f"      ✗ {vid_id} - unavailable ({stderr[:50] if stderr else 'unknown error'})")

                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    logger.debug(f"      ✗ {vid_id} - validation timeout")

            except Exception as e:
                logger.debug(f"      ✗ {vid_id} - validation error: {e}")

        if len(valid_ids) < len(video_ids):
            logger.info(f"    Validation: {len(valid_ids)}/{len(video_ids)} videos accessible (removed {len(video_ids) - len(valid_ids)} dead IDs)")
        else:
            logger.debug(f"    Validation: All {len(valid_ids)} videos accessible")

        return valid_ids

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
        skipped_global = 0

        for vid_id in video_ids:
            # GLOBAL DEDUP: Skip if already downloaded for another keyword
            if vid_id in self._downloaded_video_ids:
                # Check if file exists in this keyword folder (for local tracking)
                local_exists = any(
                    vid_id in f and f.endswith(('.mp4', '.mkv', '.webm'))
                    for f in existing_before
                )
                if not local_exists:
                    skipped_global += 1
                    logger.debug(f"    Skipping {vid_id} - already downloaded for another keyword")
                    continue

            # Check if any file contains this video ID in this folder
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

        if skipped_global > 0:
            logger.info(f"    Global dedup: skipped {skipped_global} videos (already downloaded for other keywords)")

        if already_downloaded:
            logger.debug(f"    {len(already_downloaded)} already downloaded, {len(missing_ids)} to fetch")

        if not missing_ids:
            # All videos already exist
            return already_downloaded

        # VALIDATION PHASE: Check which IDs are actually accessible before attempting download
        # This prevents timeouts on dead/private/geoblocked videos
        valid_ids = self._validate_video_ids(missing_ids)

        if not valid_ids:
            logger.warning(f"    No valid videos found after validation (all {len(missing_ids)} IDs are inaccessible)")
            return already_downloaded

        if len(valid_ids) < len(missing_ids):
            # Update search cache to remove invalid IDs (actually do it, not just log!)
            removed_ids = list(set(missing_ids) - set(valid_ids))
            self.search_cache.remove_ids(keyword, tier, removed_ids)
            logger.debug(f"    Pruned {len(removed_ids)} invalid IDs from search cache")

        # Build URLs from IDs
        # URLs are now processed individually to ensure timeouts apply per-video, not per-batch

        newly_downloaded = []

        # Get filename length from config
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)

        # Get per-video timeout (shorter than batch timeout for faster failure on bad videos)
        per_video_timeout = getattr(self.download_config, 'per_video_timeout', 60)

        # Download videos one by one (only valid IDs)
        for i, vid_id in enumerate(valid_ids, 1):
            url = f"https://www.youtube.com/watch?v={vid_id}"

            logger.debug(f"    Downloading {i}/{len(valid_ids)}: {vid_id}")
            
            cmd = [
                'yt-dlp',
                '--sleep-interval', '5',
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
                url
            ]

            self._add_cookies_to_cmd(cmd)

            # Pass just this video's download to the runner with per-video timeout
            # existing_before is updated implicitly by the runner detecting new files
            # but for safety in the loop, we should refresh it or rely on the runner's return

            # Note: _run_download_cmd calculates "new files" by looking at the directory
            # We need to be careful if running in parallel, but here it is sequential.

            batch_result = self._run_download_cmd(
                cmd, keyword_dir, output_dir, keyword, tier, existing_before,
                timeout_override=per_video_timeout
            )
            
            if batch_result:
                newly_downloaded.extend(batch_result)
                # Update existing_before so next iteration doesn't think this file is "new" again
                # (though _run_download_cmd logic handles new files by diffing, updating the set
                # prevents potential double-counting if logic changes)
                for video in batch_result:
                    filename = Path(video.file).name
                    existing_before.add(filename)
                # Track video ID globally to prevent re-download for other keywords
                self._downloaded_video_ids.add(vid_id)

            # Small delay between individual downloads to be nice to YouTube
            if i < len(valid_ids):
                time.sleep(5)

        # Combine already downloaded + newly downloaded
        return already_downloaded + newly_downloaded

    def _process_downloaded_files(
        self,
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set
    ) -> List[DownloadedVideo]:
        """Process newly downloaded files: transcode, sanitize, create records.

        Extracted from _run_download_cmd to allow reuse in retry logic.

        Args:
            keyword_dir: Keyword-specific directory
            output_dir: Base output directory
            keyword: Search keyword
            tier: Duration tier
            existing_before: Set of files that existed before download

        Returns:
            List of DownloadedVideo objects for new files
        """
        # Find new files
        existing_after = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
        new_files = existing_after - existing_before

        # Filter to video files
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
        new_videos = [f for f in new_files if Path(f).suffix.lower() in video_extensions]

        if new_videos:
            logger.debug(f"    Downloaded {len(new_videos)} video(s)")

        downloaded = []

        # Get download_timeout for transcoding
        tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
        download_timeout = tier_timeouts.get(tier, getattr(self.download_config, 'download_timeout', 120))

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

        # Notify healer of successful download batch (if healer is attached)
        if downloaded and self._download_healer:
            self._download_healer.mark_success()

        return downloaded

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

        # First-byte timeout for detecting connection hangs early (before full timeout)
        # This is critical for Tier 1 --impersonate which hangs on TLS handshake
        first_byte_timeout = getattr(self.download_config, 'first_byte_timeout', 30)

        # Track timeout for retry logic
        self._last_download_timed_out = False
        self._last_download_was_hang = False  # Distinguishes hang vs slow download
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
                # Phase 1: Use first-byte timeout to detect connection hangs quickly
                # This prevents waiting full 60s+ when --impersonate hangs on TLS
                stdout, stderr = process.communicate(timeout=first_byte_timeout)
                # If we get here, download completed within first_byte_timeout - success!

            except subprocess.TimeoutExpired:
                # First-byte timeout expired - check if this is a connection hang or slow download
                process.kill()
                try:
                    # CRITICAL: Must have timeout here too - process.kill() may not
                    # immediately terminate yt-dlp with curl_cffi impersonation on Windows
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    # Process refused to die cleanly after kill()
                    # This happens with --impersonate on Windows where curl_cffi
                    # holds connections open even after SIGTERM
                    stdout, stderr = "", ""
                    logger.debug("Process cleanup timed out after kill(), forcing termination")

                # Detect connection hang vs slow download
                # If no output at all after first_byte_timeout, this is a connection hang
                # If some output, download was progressing (should retry with longer timeout)
                is_connection_hang = not stdout and not stderr

                if is_connection_hang:
                    logger.warning(
                        f"Connection hang downloading '{keyword}' ({tier}) after {first_byte_timeout}s "
                        f"(no output produced - likely tier issue)"
                    )
                    # ACTUALLY escalate the tier (not just log it!)
                    self._escalate_bypass_tier(reason="connection_hang")
                    self._last_download_was_hang = True
                else:
                    # Slow download - had output but didn't finish in first_byte_timeout
                    # This is NOT a connection hang - retry with full timeout
                    logger.info(
                        f"Slow download for '{keyword}' ({tier}) - retrying with full timeout "
                        f"(had output: {len(stdout or '')} stdout, {len(stderr or '')} stderr bytes)"
                    )
                    # Retry with full download_timeout instead of first_byte_timeout
                    return self._retry_with_full_timeout(
                        cmd, keyword_dir, output_dir, keyword, tier, existing_before, download_timeout
                    )

                self._last_download_timed_out = True
                return []

            # Only log actual errors
            if stderr:
                for line in stderr.strip().split('\n'):
                    if line and 'WARNING' not in line and 'ERROR' in line:
                        logger.warning(f"    yt-dlp: {line}")

            # Process downloaded files (transcode, sanitize, create records)
            return self._process_downloaded_files(
                keyword_dir, output_dir, keyword, tier, existing_before
            )

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
