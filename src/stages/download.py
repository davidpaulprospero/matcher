"""
Download Stage - Video and Audio Download

Stage 2 of the video matching pipeline:
- Downloads videos from YouTube based on keywords
- Supports audio-first mode for efficient matching
- Supports caption-first mode (only downloads uncaptioned videos)
- Handles global cache reuse
- Downloads video segments after matching (audio-first)

In caption-first mode, this stage runs AFTER VIDEO_METADATA and CAPTION stages:
  VIDEO_METADATA -> CAPTION -> DOWNLOAD (only uncaptioned) -> TRANSCRIBE (only uncaptioned)

Videos that have captions (has_captions=True in video_candidates) are skipped,
as they already have transcripts from the CAPTION stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo, AudioDownload

logger = logging.getLogger(__name__)


@register_stage
class DownloadStage(Stage):
    """
    Downloads video/audio footage based on keywords.

    Inputs:
        - state.keywords: List of keywords to search for
        - state.topic_context: Topic for search refinement
        - state.videos_need_audio: List of video IDs needing Whisper fallback (caption-first mode)
        - state.video_candidates: List of VideoCandidate (caption-first mode)

    Outputs:
        - state.downloaded_videos: List of DownloadedVideo
        - state.downloaded_audio: List of AudioDownload (audio-first mode)
        - state.failed_keywords: List of failed keywords

    In caption-first mode, only downloads audio for videos in state.videos_need_audio
    (those without captions that need Whisper transcription).
    """

    name = "DOWNLOAD"
    description = "Download video footage from YouTube"

    def __init__(self):
        self.downloader = None
        self.global_cache = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the download stage"""
        warnings = []

        try:
            if config.pipeline.skip_download:
                print("  >> Skipping download (config: skip_download=true)")
                logger.info("Skipping DOWNLOAD stage (config: skip_download=true)")
                self._load_existing_videos(state, config)
                return StageResult.ok({'skipped': True}, warnings)

            # Pre-flight check: verify proxy/YouTube access before starting
            preflight_warning = self._run_preflight_check(config)
            if preflight_warning:
                warnings.append(preflight_warning)

            # Check if caption-first mode
            if self._is_caption_first_enabled(config):
                result = self._run_caption_first_download(state, config, checkpoint, warnings)
            # Check if audio-first mode
            elif self._is_audio_first_enabled(config):
                result = self._run_audio_first(state, config, checkpoint, warnings)
            else:
                result = self._run_full_download(state, config, checkpoint, warnings)

            # Log cookie rotation summary at end of download stage
            self._log_cookie_rotation_summary()

            return result

        except Exception as e:
            logger.exception(f"Download stage failed: {e}")
            self._log_cookie_rotation_summary()
            return StageResult.fail(str(e), warnings)

    def _log_cookie_rotation_summary(self):
        """Log cookie rotation statistics if enabled."""
        try:
            from ..downloader.cookie_manager import CookieManager
            cm = CookieManager.get_instance()
            if cm and cm.enabled and cm.accounts:
                cm.log_summary()
        except Exception:
            pass

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if download stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore download stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                # Fallback: try to restore from disk if config available
                if config and self._restore_from_disk(state, config):
                    logger.info("Restored DOWNLOAD from disk (no checkpoint data)")
                    return True
                return False

            # Restore downloaded videos
            from ..state import DownloadedVideo, AudioDownload
            videos = data.get('downloaded_videos', [])
            state.downloaded_videos = [
                DownloadedVideo(**v) if isinstance(v, dict) else v
                for v in videos
            ]

            # Restore downloaded audio (audio-first mode)
            audio_downloads = data.get('audio_downloads', [])
            restored_audio = []
            for a in audio_downloads:
                if isinstance(a, dict):
                    # Handle backward compatibility: old checkpoints used different field names
                    a_copy = dict(a)  # Create copy to avoid modifying checkpoint data

                    # Map 'audio_file' → 'file'
                    if 'audio_file' in a_copy and 'file' not in a_copy:
                        a_copy['file'] = a_copy.pop('audio_file')

                    # Map 'video_url' → 'url'
                    if 'video_url' in a_copy and 'url' not in a_copy:
                        a_copy['url'] = a_copy.pop('video_url')

                    # Remove obsolete fields that are no longer in AudioDownload
                    # (channel, duration_tier, upload_date, license were removed in refactoring)
                    for obsolete_field in ['channel', 'duration_tier', 'upload_date', 'license']:
                        a_copy.pop(obsolete_field, None)

                    restored_audio.append(AudioDownload(**a_copy))
                else:
                    restored_audio.append(a)
            state.downloaded_audio = restored_audio

            # Restore failed keywords
            state.failed_keywords = data.get('failed_keywords', [])

            logger.info(f"Restored DOWNLOAD: {len(state.downloaded_videos)} videos, {len(state.downloaded_audio)} audio")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD: {e}")
            return False

    def _restore_from_disk(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> bool:
        """Fallback: restore download state by scanning disk for existing files"""
        from ..state import AudioDownload
        from pathlib import Path

        try:
            # Get videos root directory from config
            root_dir = getattr(config.download, 'root_dir', None)
            if not root_dir:
                return False

            # Find project subdirectory (must include project name)
            root_path = Path(root_dir)
            project_name = getattr(config.project, 'name', '') or ''
            if project_name:
                project_path = root_path / project_name[:15]
            else:
                project_path = root_path
            
            if not project_path.exists():
                return False

            # Look for audio directories (audio-first mode)
            audio_files = []
            for audio_dir in project_path.rglob('*_audio'):
                for mp3 in audio_dir.glob('*.mp3'):
                    audio_files.append(AudioDownload(
                        file=str(mp3),
                        url="",  # Unknown from disk
                        video_id=mp3.stem,
                        title=mp3.stem,
                        duration=0.0,
                        keyword=audio_dir.name.replace('_audio', '').replace('_l', '').replace('_m', '')
                    ))

            if audio_files:
                state.downloaded_audio = audio_files
                logger.info(f"Restored DOWNLOAD from disk: {len(audio_files)} audio files")
                return True

            # Look for video files
            video_files = list(project_path.rglob('*.mp4')) + list(root_path.rglob('*.webm'))
            if video_files:
                from ..state import DownloadedVideo
                state.downloaded_videos = [
                    DownloadedVideo(
                        file=str(v),
                        duration_tier='m',
                        keyword=v.parent.name.replace('_segments', '').replace('_l', '').replace('_m', ''),
                        title=v.stem,
                        video_id=v.stem
                    )
                    for v in video_files
                ]
                logger.info(f"Restored DOWNLOAD from disk: {len(state.downloaded_videos)} videos")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD from disk: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        if not state.keywords:
            return "No keywords available for download"
        return None

    # === Main Download Methods ===

    def _run_full_download(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        warnings: List[str]
    ) -> StageResult:
        """Run full video download mode"""
        print(f"\n  --- Stage 2: DOWNLOAD FOOTAGE ---")

        try:
            from ..downloader import VideoDownloader

            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            # Print tier info
            print(f"  Duration tiers:")
            for tier_name in ['short', 'medium', 'long', 'longer']:
                min_s = self.downloader._get_tier_value(tier_name, 'min', 0)
                max_s = self.downloader._get_tier_value(tier_name, 'max', 120)
                per_kw = self.downloader._get_tier_value(tier_name, 'per_keyword', 5)
                print(f"    - {tier_name}: {min_s}-{max_s}s ({per_kw}/kw)")

            # Check global cache first
            keywords_to_download, reusable_videos = self._check_global_cache(
                state.keywords, config
            )

            # Download videos
            if keywords_to_download:
                downloaded_videos, failed = self.downloader.download_all(
                    keywords=keywords_to_download,
                    output_dir=output_dir,
                    resume=True,
                    topic=state.topic_context or ""
                )
            else:
                downloaded_videos = []
                failed = []
                print(f"  >> All keywords covered by global cache")

            # Store results
            self._store_download_results(
                state, downloaded_videos, reusable_videos, failed
            )

            print(f"\n  + Downloaded {len(downloaded_videos)} new videos")
            if reusable_videos:
                print(f"  + Reusing {len(reusable_videos)} videos from global cache")

            # Prepare checkpoint data
            checkpoint_data = {
                'downloaded_videos': [
                    self._video_to_dict(dv) for dv in state.downloaded_videos
                ],
                'failed_keywords': state.failed_keywords,
                'video_count': len(state.downloaded_videos),
            }

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            return StageResult.fail(f"Could not import downloader: {e}", warnings)

    def _run_caption_first_download(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        warnings: List[str]
    ) -> StageResult:
        """
        Run caption-first download mode.

        Only downloads audio for videos that don't have captions
        (those in state.videos_need_audio).
        """
        print(f"\n  --- Stage 2: DOWNLOAD (Caption-First Mode) ---")

        # Check if there are any videos needing audio download
        if not state.videos_need_audio:
            captioned_count = len([vc for vc in state.video_candidates if vc.has_captions])
            print(f"  >> All {captioned_count} videos have captions - no download needed!")
            logger.info(f"Caption-first: All {captioned_count} videos have captions, skipping download")
            return StageResult.ok({
                'skipped': True,
                'reason': 'all_videos_have_captions',
                'captioned_count': captioned_count,
            }, warnings)

        print(f"  >> {len(state.videos_need_audio)} videos need audio download (no captions)")
        logger.info(f"Caption-first: {len(state.videos_need_audio)} videos need audio fallback")

        try:
            from ..downloader import VideoDownloader
            from ..state import AudioDownload

            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            # Build lookup of video candidates by video_id
            candidate_lookup = {vc.video_id: vc for vc in state.video_candidates}

            all_audio_downloads = []
            total_videos = len(state.videos_need_audio)
            failed_videos = []

            for idx, video_id in enumerate(state.videos_need_audio, 1):
                candidate = candidate_lookup.get(video_id)
                if not candidate:
                    logger.warning(f"Video candidate not found for {video_id}")
                    continue

                print(f"\n  [{idx}/{total_videos}] {video_id}: {candidate.title[:50]}...")

                try:
                    # Download audio for this specific video
                    audio_download = self._download_audio_for_video(
                        video_id=video_id,
                        url=candidate.url,
                        title=candidate.title,
                        keyword=candidate.keyword,
                        output_dir=output_dir,
                        config=config
                    )

                    if audio_download:
                        all_audio_downloads.append(audio_download)
                        print(f"    ✓ Downloaded audio")
                    else:
                        failed_videos.append(video_id)
                        print(f"    ✗ Failed to download")

                    # Rate limit prevention: delay between downloads
                    # Skip delay for cache hits (file already existed)
                    from_cache = audio_download and getattr(audio_download, 'from_cache', False)
                    delay = getattr(config.download, 'delay_between_downloads', 3.0)
                    if delay > 0 and idx < total_videos and not from_cache:
                        import time
                        logger.debug(f"[rate_limit] Sleeping {delay}s between downloads ({idx}/{total_videos})")
                        time.sleep(delay)

                except Exception as e:
                    logger.error(f"Audio download failed for '{video_id}': {e}")
                    failed_videos.append(video_id)

            # Store results
            state.downloaded_audio = all_audio_downloads

            if failed_videos:
                warnings.append(f"{len(failed_videos)} videos failed to download audio")
                print(f"\n  ! {len(failed_videos)} videos failed")

            print(f"\n  + Downloaded {len(all_audio_downloads)} audio files (caption fallback)")

            checkpoint_data = {
                'audio_downloads': [
                    self._audio_to_dict(ad) for ad in all_audio_downloads
                ],
                'failed_videos': failed_videos,
                'audio_count': len(all_audio_downloads),
                'mode': 'caption_first_fallback',
            }

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            return StageResult.fail(f"Could not import downloader: {e}", warnings)

    def _download_audio_for_video(
        self,
        video_id: str,
        url: str,
        title: str,
        keyword: str,
        output_dir: Path,
        config: 'Config'
    ) -> Optional['AudioDownload']:
        """Download audio for a specific video by URL."""
        from ..state import AudioDownload
        import subprocess
        import os

        # Create output directory for this keyword
        keyword_dir = output_dir / f"{keyword.replace(' ', '_')}_audio"
        keyword_dir.mkdir(parents=True, exist_ok=True)

        # Output filename
        output_file = keyword_dir / f"{video_id}.mp3"

        # Skip if already downloaded (cache hit - no rate limit delay needed)
        if output_file.exists():
            logger.info(f"Audio already exists: {output_file}")
            return AudioDownload(
                file=str(output_file),
                url=url,
                video_id=video_id,
                title=title,
                duration=0.0,
                keyword=keyword,
                from_cache=True  # Skip rate-limit delay for cache hits
            )

        # Build yt-dlp command for audio extraction
        cmd = [
            'yt-dlp',
            url,
            '-x',  # Extract audio
            '--audio-format', 'mp3',
            '--audio-quality', '192K',
            '-o', str(output_file),
            '--no-playlist',
            '--no-warnings',
        ]

        # Add base args (JS runtime for challenge solving), cookies, and proxy
        from ..downloader.utils import get_cookies_args, get_ytdlp_base_args, get_proxy_args
        cmd.extend(get_ytdlp_base_args())
        cmd.extend(get_cookies_args(config))
        cmd.extend(get_proxy_args(config))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                encoding='utf-8',
                errors='ignore',
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode == 0 and output_file.exists():
                # Get duration from file
                duration = self._get_audio_duration(output_file)

                # Report success to cookie rotation
                try:
                    from ..downloader.cookie_manager import CookieManager
                    cm = CookieManager.get_instance()
                    if cm and cm.enabled:
                        cm.report_success()
                except Exception:
                    pass

                return AudioDownload(
                    file=str(output_file),
                    url=url,
                    video_id=video_id,
                    title=title,
                    duration=duration,
                    keyword=keyword
                )
            else:
                logger.warning(f"yt-dlp failed for {video_id}: {result.stderr[:200] if result.stderr else 'unknown'}")

                # Rate limit detection: rotate proxy and backoff
                stderr_lower = str(result.stderr).lower() if result.stderr else ''
                if 'rate-limited' in stderr_lower or '429' in stderr_lower:
                    import time
                    backoff_delay = 30  # 30 seconds for rate limit
                    logger.warning(f"[rate_limit] ⚠️ RATE LIMIT DETECTED for {video_id}")
                    logger.warning(f"[rate_limit] YouTube has rate-limited this account")

                    # Report rate limit to cookie rotation
                    try:
                        from ..downloader.cookie_manager import CookieManager
                        cm = CookieManager.get_instance()
                        if cm and cm.enabled:
                            cm.report_rate_limit()
                    except Exception:
                        pass

                    # Rotate proxy on rate limit
                    try:
                        from ..pot_utils.proxy_manager import ProxyManager
                        pm = ProxyManager.get_instance()
                        if pm:
                            old_proxy = pm.get_current_proxy()
                            pm.report_failure(old_proxy or "direct", is_rate_limit=True)
                            new_proxy = pm.get_next_proxy()  # Force rotation to next proxy
                            logger.warning(f"[rate_limit] Rotated proxy: {old_proxy or 'direct'} -> {new_proxy or 'direct'}")
                            print(f"    Rotated proxy -> {new_proxy or 'direct'}")
                    except Exception as e:
                        logger.warning(f"[rate_limit] Proxy rotation failed: {e}")

                    logger.warning(f"[rate_limit] Applying {backoff_delay}s backoff before continuing...")
                    print(f"    ⚠️ Rate limited! Waiting {backoff_delay}s...")
                    time.sleep(backoff_delay)
                    logger.info(f"[rate_limit] Backoff complete, resuming downloads")

                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout downloading audio for {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error downloading audio for {video_id}: {e}")
            return None

    def _get_audio_duration(self, audio_file: Path) -> float:
        """Get duration of audio file using ffprobe."""
        import subprocess
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_file)],
                capture_output=True,
                text=True,
                timeout=10
            )
            return float(result.stdout.strip()) if result.stdout.strip() else 0.0
        except Exception:
            return 0.0

    def _run_audio_first(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        warnings: List[str]
    ) -> StageResult:
        """Run audio-first download mode"""
        print(f"\n  --- Stage 2A: DOWNLOAD AUDIO (Audio-First Mode) ---")

        audio_config = getattr(config.download, 'audio_first', None)
        print(f"  Audio-first mode: Downloading audio for transcription")
        print(f"    Buffer: {getattr(audio_config, 'buffer_seconds', 30)}s")
        print(f"    Merge gap: {getattr(audio_config, 'merge_gap_seconds', 15)}s")

        try:
            from ..downloader import VideoDownloader
            from ..state import AudioDownload

            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            all_audio_downloads = []
            tiers = list(self.downloader.DURATION_TIERS.keys())
            total_keywords = len(state.keywords)
            failed_keywords = []

            for idx, keyword in enumerate(state.keywords, 1):
                print(f"\n  [{idx}/{total_keywords}] {keyword}")

                try:
                    for tier in tiers:
                        per_kw = self.downloader._get_tier_value(tier, 'per_keyword', 5)
                        if per_kw <= 0:
                            continue

                        audio_downloads = self.downloader.audio_first.download_audio_for_keyword(
                            keyword=keyword,
                            output_dir=output_dir,
                            tier=tier,
                            topic=state.topic_context or ""
                        )
                        all_audio_downloads.extend(audio_downloads)
                except Exception as e:
                    logger.error(f"Audio download failed for '{keyword}': {e}")
                    failed_keywords.append(keyword)

            # Store results
            state.downloaded_audio = all_audio_downloads
            state.failed_keywords = failed_keywords

            if failed_keywords:
                print(f"\n  ! {len(failed_keywords)} keywords failed")
            print(f"\n  + Downloaded {len(all_audio_downloads)} audio files")

            checkpoint_data = {
                'audio_downloads': [
                    self._audio_to_dict(ad) for ad in all_audio_downloads
                ],
                'failed_keywords': failed_keywords,
                'audio_count': len(all_audio_downloads),
                'mode': 'audio_first',
            }

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            return StageResult.fail(f"Could not import downloader: {e}", warnings)

    # === Helper Methods ===

    def _run_preflight_check(self, config: 'Config') -> Optional[str]:
        """
        Run pre-flight check to verify proxy and YouTube access.

        Tests the configured proxy against YouTube before starting downloads.
        This catches issues early rather than failing mid-download.

        Returns:
            Warning message if check failed, None if successful
        """
        try:
            fallback_cfg = getattr(config.download, 'fallback', None)
            if not fallback_cfg:
                return None

            proxy_cfg = getattr(fallback_cfg, 'proxy', None)
            if not proxy_cfg:
                return None

            # Check if proxy is enabled
            enabled = proxy_cfg.get('enabled', False) if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'enabled', False)
            if not enabled:
                return None

            # Get rate limit handler (already initialized with proxies)
            from ..downloader.rate_limit_handler import get_rate_limit_handler
            handler = get_rate_limit_handler(config)

            if not handler.has_proxy:
                return None

            # Test YouTube access via proxy
            print("  >> Pre-flight: Testing YouTube access via proxy...")
            logger.info("Running pre-flight proxy check against YouTube")

            import httpx
            test_url = "https://www.youtube.com/robots.txt"
            proxy = handler.get_proxy()

            try:
                with httpx.Client(proxy=proxy, timeout=15.0) as client:
                    response = client.get(test_url)

                    if response.status_code == 200:
                        print(f"  >> Pre-flight: SUCCESS via {proxy or 'direct'}")
                        logger.info(f"Pre-flight passed via proxy")
                        return None

                    elif response.status_code == 429:
                        print("  >> Pre-flight: Rate limited - rotating proxy...")
                        logger.warning("Pre-flight hit rate limit, rotating proxy")
                        handler.on_rate_limit("preflight")

                        # Try again with rotated proxy
                        new_proxy = handler.get_proxy()
                        with httpx.Client(proxy=new_proxy, timeout=15.0) as retry_client:
                            retry_response = retry_client.get(test_url)
                            if retry_response.status_code == 200:
                                print(f"  >> Pre-flight: SUCCESS after rotation")
                                return None

                        return "Pre-flight: Rate limited even after proxy rotation"

                    else:
                        return f"Pre-flight: Unexpected status {response.status_code}"

            except httpx.ProxyError as e:
                logger.warning(f"Pre-flight proxy error: {e}")
                return f"Pre-flight: Proxy connection failed - {e}"

            except httpx.TimeoutException:
                logger.warning("Pre-flight timeout")
                return "Pre-flight: Request timed out"

            except Exception as e:
                logger.warning(f"Pre-flight error: {e}")
                return f"Pre-flight: {e}"

        except Exception as e:
            logger.debug(f"Pre-flight check error: {e}")
            return None  # Don't block on pre-flight errors

    def _is_audio_first_enabled(self, config: 'Config') -> bool:
        """Check if audio-first mode is enabled"""
        audio_config = getattr(config.download, 'audio_first', None)
        return audio_config and getattr(audio_config, 'enabled', False)

    def _is_caption_first_enabled(self, config: 'Config') -> bool:
        """Check if caption-first mode is enabled"""
        caption_config = getattr(config.download, 'caption_first', None)
        # Handle both dict and object config patterns (Rule 6)
        if isinstance(caption_config, dict):
            return caption_config.get('enabled', False)
        return caption_config and getattr(caption_config, 'enabled', False)

    def _init_global_cache(self, config: 'Config'):
        """Initialize global cache manager if enabled"""
        global_config = getattr(config, 'global_cache', None)
        if not global_config:
            return None

        if not getattr(global_config, 'enabled', False):
            return None

        if not getattr(global_config, 'check_before_download', True):
            return None

        try:
            from ..global_cache import GlobalCacheManager
            cache_dir = getattr(global_config, 'cache_dir', None)
            return GlobalCacheManager(cache_dir=cache_dir, config=global_config)
        except Exception as e:
            logger.warning(f"Failed to initialize global cache: {e}")
            return None

    def _check_global_cache(
        self,
        keywords: List[str],
        config: 'Config',
        topics: List[str] = None
    ) -> tuple:
        """Check global cache for reusable videos"""
        global_config = getattr(config, 'global_cache', None)

        # Skip if global cache is disabled
        if not global_config or not getattr(global_config, 'enabled', False):
            return keywords, []

        # Skip if check_before_download is False
        if not getattr(global_config, 'check_before_download', True):
            return keywords, []

        # Try to initialize cache
        try:
            cache = self._init_global_cache(config)
            if not cache:
                return keywords, []

            # Query cache for relevant videos
            result = cache.find_videos_for_keywords(keywords, topics=topics)

            # Return uncovered keywords and reusable videos
            return result.uncovered_keywords or keywords, []
        except Exception as e:
            logger.warning(f"Global cache check failed: {e}")
            return keywords, []

    def _register_downloaded_videos(
        self,
        videos: List[Any],
        config: 'Config',
        project_id: str = ""
    ):
        """Register downloaded videos in global cache"""
        if not hasattr(self, 'global_cache') or self.global_cache is None:
            return

        for video in videos:
            if isinstance(video, dict):
                video_path = video.get('file', video.get('path', ''))
                if video_path and Path(video_path).exists():
                    try:
                        self.global_cache.register_video(
                            video_path=video_path,
                            download_keyword=video.get('keyword', ''),
                            topics=video.get('topics', []),
                            project_id=project_id
                        )
                    except Exception as e:
                        logger.debug(f"Failed to register video in cache: {e}")

    def _store_download_results(
        self,
        state: 'PipelineState',
        downloaded_videos: List[Any],
        reusable_videos: List[Dict],
        failed: List[str]
    ):
        """Store download results in state"""
        from ..state import DownloadedVideo

        state.downloaded_videos = []

        # Add newly downloaded videos
        for dv in downloaded_videos:
            if isinstance(dv, dict):
                state.downloaded_videos.append(DownloadedVideo(
                    file=dv.get('file', dv.get('path', '')),
                    url=dv.get('url', ''),
                    title=dv.get('title', ''),
                    channel=dv.get('channel', ''),
                    upload_date=dv.get('upload_date', ''),
                    duration=dv.get('duration', 0.0),
                    duration_tier=dv.get('duration_tier', dv.get('tier', '')),
                    keyword=dv.get('keyword', ''),
                    download_date=dv.get('download_date', ''),
                    source='download',
                ))
            else:
                state.downloaded_videos.append(dv)

        # Add reusable videos from global cache
        for gv in reusable_videos:
            state.downloaded_videos.append(DownloadedVideo(
                file=gv.get('path', ''),
                source='global_cache',
                face_score=gv.get('face_score', 0.5),
            ))

        state.failed_keywords = failed

    def _load_existing_videos(
        self,
        state: 'PipelineState',
        config: 'Config'
    ):
        """Load existing videos when skipping download"""
        from ..state import DownloadedVideo

        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
        videos_dir = Path(config.downloaded_videos_dir)

        if not videos_dir.exists():
            logger.warning(f"Videos directory not found: {videos_dir}")
            return

        video_files = []

        # Direct files
        for f in videos_dir.iterdir():
            if f.is_file() and f.suffix.lower() in video_extensions:
                video_files.append(f)

        # One level deep in subfolders
        if not video_files:
            for subdir in videos_dir.iterdir():
                if subdir.is_dir():
                    for f in subdir.iterdir():
                        if f.is_file() and f.suffix.lower() in video_extensions:
                            video_files.append(f)

        state.downloaded_videos = [
            DownloadedVideo(
                file=str(vf),
                source='existing',
            )
            for vf in video_files
        ]

        print(f"  + Loaded {len(state.downloaded_videos)} existing videos")

    def _video_to_dict(self, video: 'DownloadedVideo') -> Dict[str, Any]:
        """Convert DownloadedVideo to dict for checkpointing"""
        if hasattr(video, '__dict__'):
            return {k: v for k, v in video.__dict__.items()}
        return dict(video) if isinstance(video, dict) else {}

    def _audio_to_dict(self, audio: 'AudioDownload') -> Dict[str, Any]:
        """Convert AudioDownload to dict for checkpointing"""
        if hasattr(audio, '__dict__'):
            return {k: v for k, v in audio.__dict__.items()}
        return dict(audio) if isinstance(audio, dict) else {}


@register_stage
class DownloadVideoSegmentsStage(Stage):
    """
    Downloads video segments after matching (audio-first mode).

    This is Stage 2B in audio-first mode, run after matching to
    download only the portions of videos actually used.

    Inputs:
        - state.matches: List of Match objects
        - state.downloaded_audio: List of AudioDownload

    Outputs:
        - Updated state.downloaded_videos with segment files
    """

    name = "DOWNLOAD_SEGMENTS"
    description = "Download matched video segments (audio-first)"

    def __init__(self):
        self.downloader = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video segment download stage"""
        warnings = []

        try:
            # Check which mode we're in
            has_audio = state.downloaded_audio and len(state.downloaded_audio) > 0
            has_video_candidates = state.video_candidates and len(state.video_candidates) > 0
            is_caption_first = self._is_caption_first_enabled(config) and has_video_candidates

            # Caption-first mode: download segments for YouTube videos matched via captions
            if is_caption_first:
                return self._run_caption_first_segments(state, config, checkpoint, warnings)

            # Edge case: skip_download=true BUT we have audio files
            # This means user interrupted audio-first pipeline before video download
            if config.pipeline.skip_download and has_audio:
                warning_msg = (
                    "WARNING: skip_download=true but audio files exist. "
                    "This indicates incomplete audio-first pipeline. "
                    "Video segments have NOT been downloaded yet. "
                    "OTIO will reference audio files (.mp3) instead of video files. "
                    "To fix: Set skip_download=false and re-run to download matched video segments."
                )
                print(f"\n  ! WARNING: {warning_msg}")
                logger.warning(warning_msg)
                warnings.append("Incomplete audio-first pipeline - no video segments")
                return StageResult.ok({'skipped': True, 'reason': 'skip_download_with_audio'}, warnings)

            # Normal skip: not in audio-first mode
            if config.pipeline.skip_download or not has_audio:
                print("  >> Skipping segment download (not audio-first mode)")
                logger.info("Skipping DOWNLOAD_SEGMENTS (not audio-first mode)")
                return StageResult.ok({'skipped': True}, warnings)

            if not state.matches:
                return StageResult.fail("No matches - run matching first", warnings)

            print(f"\n  --- Stage 2B: DOWNLOAD VIDEO SEGMENTS ---")

            from ..downloader import (
                VideoDownloader,
                collect_matched_segments,
                prepare_merged_segments,
            )

            audio_config = getattr(config.download, 'audio_first', None)
            buffer_seconds = getattr(audio_config, 'buffer_seconds', 30.0)
            merge_gap = getattr(audio_config, 'merge_gap_seconds', 15.0)

            print(f"  Downloading matched video segments")
            print(f"    Buffer: {buffer_seconds}s, Merge gap: {merge_gap}s")

            # Build audio downloads by ID
            audio_downloads_by_id = {
                a.video_id if hasattr(a, 'video_id') else a.get('video_id', ''): a
                for a in state.downloaded_audio
            }

            # Collect matched segments
            segments_by_video = collect_matched_segments(
                state.matches,
                audio_downloads_by_id
            )

            total_matches = sum(len(segs) for segs in segments_by_video.values())
            print(f"    Matched segments: {total_matches} across {len(segments_by_video)} videos")

            # Merge segments with buffer
            merged_segments = prepare_merged_segments(
                segments_by_video,
                audio_downloads_by_id,
                buffer_seconds=buffer_seconds,
                merge_gap_seconds=merge_gap
            )

            print(f"    After merge: {len(merged_segments)} segments to download")

            # Download segments
            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            # Create checkpoint callback for periodic saves during long downloads
            def checkpoint_progress(current: int, total: int, segments: list):
                """Save progress checkpoint during download"""
                checkpoint_data = {
                    'segment_count': len(segments),
                    'total_matches': total_matches,
                    'videos_completed': current,
                    'videos_total': total,
                    'in_progress': current < total,
                }
                # Save intermediate checkpoint (doesn't update last_completed_stage)
                checkpoint.save_intermediate('DOWNLOAD_SEGMENTS', checkpoint_data)

            downloaded_segments = self.downloader.audio_first.download_video_segments(
                merged_segments,
                output_dir,
                progress_callback=checkpoint_progress
            )

            print(f"\n  + Downloaded {len(downloaded_segments)} video segments")

            # Update Match objects to reference downloaded video segments (.mp4) instead of audio files (.mp3)
            self._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

            # Update text_metadata to reference video files instead of audio files
            self._remap_text_metadata_to_video_files(state, downloaded_segments, audio_downloads_by_id)

            checkpoint_data = {
                'segment_count': len(downloaded_segments),
                'total_matches': total_matches,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Video segment download failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _remap_matches_to_video_segments(
        self,
        state: 'PipelineState',
        downloaded_segments: List,
        audio_downloads_by_id: dict
    ) -> None:
        """
        Update Match objects to reference downloaded video segment files instead of audio files.

        In audio-first mode, matches initially reference .mp3 audio files. After downloading
        video segments, we need to remap them to the actual .mp4 segment files.

        Args:
            state: Pipeline state with matches to update
            downloaded_segments: List of DownloadedSegment objects
            audio_downloads_by_id: Dict mapping video_id to AudioDownload
        """
        from ..downloader.types import DownloadedSegment

        # Build mapping: (video_id, original_time) -> segment_file
        # Each DownloadedSegment contains multiple matches within its time range
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                # Key by video_id and the original video time
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Helper function to remap a single Match object
        def remap_match(match_obj):
            """Remap a single Match object to video segment file"""
            # Handle two different Match structures:
            # 1. state.Match: has video_file, video_start fields
            # 2. utils.Match: has video_segment.source_file field

            if hasattr(match_obj, 'video_file'):
                # state.Match structure
                if not match_obj.video_file:
                    # Skip matches with empty video_file (shouldn't happen, but defensive)
                    return False
                audio_file = Path(match_obj.video_file).stem
                start_time = match_obj.video_start
            elif hasattr(match_obj, 'video_segment'):
                # utils.Match structure - use the video_segment
                if not hasattr(match_obj.video_segment, 'source_file'):
                    return False
                if not match_obj.video_segment.source_file:
                    # Skip matches with empty source_file
                    return False
                audio_file = Path(match_obj.video_segment.source_file).stem
                start_time = match_obj.video_segment.start_time
            else:
                logger.warning(f"Unknown Match structure: {type(match_obj)}")
                return False

            # Skip empty audio files (defensive check)
            if not audio_file:
                return False

            # Skip stock videos (pexels_, pixabay_) and entity videos - they don't need remapping
            if audio_file.startswith(('pexels_', 'pixabay_', 'entity_')):
                return False

            # Find corresponding video_id from audio downloads
            # Strip timestamp suffix (_0000, _1234, etc.) from audio_file if present
            base_audio_file = audio_file
            if '_' in audio_file:
                # Check if last part after underscore is all digits (timestamp)
                parts = audio_file.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_audio_file = parts[0]

            video_id = None
            for vid, audio in audio_downloads_by_id.items():
                audio_path = audio.file if hasattr(audio, 'file') else audio.get('file', '')
                if Path(audio_path).stem == base_audio_file:
                    video_id = vid
                    break

            if not video_id:
                logger.warning(f"Could not find video_id for audio file: {audio_file} (base: {base_audio_file})")
                return False

            # Look up the downloaded segment containing this match
            key = (video_id, start_time)
            if key in segment_map:
                segment_file, original_start = segment_map[key]

                # Update match to reference video segment file
                if hasattr(match_obj, 'video_file'):
                    old_file = match_obj.video_file
                    match_obj.video_file = segment_file
                else:
                    old_file = match_obj.video_segment.source_file
                    match_obj.video_segment.source_file = segment_file

                logger.debug(f"Remapped match: {old_file} -> {segment_file}")
                return True
            else:
                logger.warning(f"Could not find downloaded segment for match: video_id={video_id}, time={start_time}")
                return False

        # Helper to remap video_segment.source_file (for AlternativeMatch/StrategyMatch)
        def remap_segment(video_segment):
            """Remap video_segment.source_file to video segment file"""
            if not hasattr(video_segment, 'source_file'):
                return False

            audio_file = Path(video_segment.source_file).stem

            # Skip stock videos (pexels_, pixabay_) and entity videos - they don't need remapping
            if audio_file.startswith(('pexels_', 'pixabay_', 'entity_')):
                return False

            # Strip timestamp suffix (_0000, _1234, etc.) from audio_file if present
            base_audio_file = audio_file
            if '_' in audio_file:
                # Check if last part after underscore is all digits (timestamp)
                parts = audio_file.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_audio_file = parts[0]

            # Find corresponding video_id
            video_id = None
            for vid, audio in audio_downloads_by_id.items():
                audio_path = audio.file if hasattr(audio, 'file') else audio.get('file', '')
                if Path(audio_path).stem == base_audio_file:
                    video_id = vid
                    break

            if not video_id:
                return False

            # Look up segment file
            # For video_segments, we need to use start_time instead of video_start
            key = (video_id, video_segment.start_time)
            if key in segment_map:
                segment_file, _ = segment_map[key]
                old_file = video_segment.source_file
                video_segment.source_file = segment_file
                logger.debug(f"Remapped segment: {old_file} -> {segment_file}")
                return True
            return False

        # Update all Match objects (handling both Match and MatchResult)
        updated_count = 0
        for item in state.matches:
            # Check if this is a MatchResult wrapper or a plain Match
            if hasattr(item, 'primary_match'):
                # MatchResult object - update primary match
                if remap_match(item.primary_match):
                    updated_count += 1

                # Update alternatives (V2-V3) - these have video_segment field
                for alt in item.alternatives:
                    if hasattr(alt, 'video_segment') and remap_segment(alt.video_segment):
                        updated_count += 1

                # Update secondary matches (V4-V6) - these have video_segment field
                for sec in item.secondary_matches:
                    if hasattr(sec, 'video_segment') and remap_segment(sec.video_segment):
                        updated_count += 1

                # Update strategy matches (V7+) - these have video_segment field
                for strat in item.strategy_matches:
                    if hasattr(strat, 'video_segment') and remap_segment(strat.video_segment):
                        updated_count += 1
            else:
                # Plain Match object
                if remap_match(item):
                    updated_count += 1

        logger.info(f"Remapped {updated_count} match objects to video segment files")
        print(f"  ✓ Updated {updated_count} matches to reference video segments")

    def _remap_text_metadata_to_video_files(
        self,
        state: 'PipelineState',
        downloaded_segments: List,
        audio_downloads_by_id: dict
    ) -> None:
        """
        Update text_metadata to reference video files instead of audio files.

        In audio-first mode, text_metadata initially references .mp3 audio files.
        After downloading video segments, we need to remap them to .mp4 files.

        Args:
            state: Pipeline state containing text_metadata
            downloaded_segments: List of DownloadedSegment objects
            audio_downloads_by_id: Dict mapping video_id to AudioDownload
        """
        from ..downloader.types import DownloadedSegment
        from ..downloader.segment_utils import _extract_video_id

        if not state.text_metadata:
            return

        # Build mapping: audio_file -> set of video_segment_files
        audio_to_video_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            if video_id in audio_downloads_by_id:
                audio_file = audio_downloads_by_id[video_id].file
                if audio_file not in audio_to_video_map:
                    audio_to_video_map[audio_file] = set()
                audio_to_video_map[audio_file].add(seg.file)

        # Remap text_metadata entries
        updated_count = 0
        for meta in state.text_metadata:
            if not isinstance(meta, dict):
                continue

            video_path = meta.get('video_path', '')
            if not video_path:
                continue

            # Check if this is an audio file that has video segments
            if video_path in audio_to_video_map:
                # Prefer the first video segment file
                video_files = list(audio_to_video_map[video_path])
                new_path = video_files[0]
                meta['video_path'] = new_path
                logger.debug(f"Remapped text_metadata: {video_path} -> {new_path}")
                updated_count += 1

        if updated_count > 0:
            logger.info(f"Remapped {updated_count} text_metadata entries to video files")
            print(f"  ✓ Updated {updated_count} text_metadata entries to reference video files")

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
        # Segment files are on disk, no state to restore
        return True

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs"""
        # Skip validation if not in audio-first or caption-first mode
        # (stage will skip itself in run())
        has_audio = state.downloaded_audio and len(state.downloaded_audio) > 0
        has_video_candidates = state.video_candidates and len(state.video_candidates) > 0
        is_caption_first = self._is_caption_first_enabled(config) and has_video_candidates

        if config.pipeline.skip_download or (not has_audio and not is_caption_first):
            return None

        if not state.matches:
            return "No matches available for segment download"
        return None

    def _is_caption_first_enabled(self, config: 'Config') -> bool:
        """Check if caption-first mode is enabled"""
        caption_config = getattr(config.download, 'caption_first', None)
        # Handle both dict and object config patterns (Rule 6)
        if isinstance(caption_config, dict):
            return caption_config.get('enabled', False)
        return caption_config and getattr(caption_config, 'enabled', False)

    def _run_caption_first_segments(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        warnings: List[str]
    ) -> StageResult:
        """
        Download video segments for caption-first mode.

        In caption-first mode, videos were never fully downloaded. Matching was done
        against caption transcripts. Now we download only the matched segments.
        """
        if not state.matches:
            return StageResult.fail("No matches - run matching first", warnings)

        print(f"\n  --- Stage: DOWNLOAD VIDEO SEGMENTS (Caption-First) ---")

        from ..downloader import (
            VideoDownloader,
            collect_matched_segments_caption_first,
            prepare_merged_segments_caption_first,
        )

        # Get caption-first config
        caption_config = getattr(config.download, 'caption_first', None)
        buffer_seconds = getattr(caption_config, 'segment_buffer_seconds', 5.0)
        merge_gap = getattr(caption_config, 'merge_gap_seconds', 10.0)

        print(f"  Downloading matched video segments")
        print(f"    Buffer: {buffer_seconds}s, Merge gap: {merge_gap}s")

        # Build video candidates lookup by ID
        video_candidates_by_id = {
            vc.video_id: vc for vc in state.video_candidates
        }

        # Collect matched segments from caption-first matches
        segments_by_video = collect_matched_segments_caption_first(
            state.matches,
            video_candidates_by_id
        )

        total_matches = sum(len(segs) for segs in segments_by_video.values())
        print(f"    Matched segments: {total_matches} across {len(segments_by_video)} videos")

        if not segments_by_video:
            print("  ! No YouTube video segments to download (may be all B-roll/stock)")
            logger.info("Caption-first: No YouTube segments to download")
            return StageResult.ok({'skipped': True, 'reason': 'no_youtube_segments'}, warnings)

        # Merge segments with buffer
        merged_segments = prepare_merged_segments_caption_first(
            segments_by_video,
            video_candidates_by_id,
            buffer_seconds=buffer_seconds,
            merge_gap_seconds=merge_gap
        )

        print(f"    After merge: {len(merged_segments)} segments to download")

        # Download segments
        self.downloader = VideoDownloader(config=config)
        output_dir = Path(config.downloaded_videos_dir)

        downloaded_segments = self.downloader.audio_first.download_video_segments(
            merged_segments,
            output_dir,
            progress_callback=None
        )

        print(f"\n  + Downloaded {len(downloaded_segments)} video segments")

        # Update Match objects to reference downloaded video segments
        self._remap_matches_caption_first(state, downloaded_segments, video_candidates_by_id)

        # Update text_metadata to reference video files instead of video IDs
        self._remap_text_metadata_caption_first(state, downloaded_segments)

        checkpoint_data = {
            'segment_count': len(downloaded_segments),
            'total_matches': total_matches,
            'mode': 'caption_first',
        }

        return StageResult.ok(checkpoint_data, warnings)

    def _remap_matches_caption_first(
        self,
        state: 'PipelineState',
        downloaded_segments: List,
        video_candidates_by_id: dict
    ) -> None:
        """
        Update Match objects to reference downloaded video segment files.

        In caption-first mode, matches initially reference video_ids (not file paths).
        After downloading segments, we remap them to actual .mp4 files.

        CRITICAL: Match objects have video_segment.source_file, NOT video_file!
        """
        from ..downloader.types import DownloadedSegment
        from ..downloader.segment_utils import _extract_video_id

        # Build mapping: (video_id, start_time) -> segment_file
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Update all Match objects and alternatives/secondary matches
        updated_count = 0

        for match_result in state.matches:
            # Extract the actual Match object (may be wrapped in MatchResult)
            if hasattr(match_result, 'primary_match'):
                # MatchResult wrapper
                match = match_result.primary_match
            else:
                # Direct Match object
                match = match_result

            # Match objects have video_segment.source_file, not video_file
            if not hasattr(match, 'video_segment') or not match.video_segment:
                continue

            # In caption-first mode, source_file is video_id or contains it
            video_id = _extract_video_id(match.video_segment.source_file)
            if not video_id:
                continue

            # Look up segment file
            key = (video_id, match.video_segment.start_time)
            if key in segment_map:
                segment_file, original_start = segment_map[key]
                old_ref = match.video_segment.source_file
                match.video_segment.source_file = segment_file
                # Adjust video timings to be relative to segment file
                # match.video_segment.start_time -= original_start
                # match.video_segment.end_time -= original_start
                logger.debug(f"Remapped: {old_ref} -> {segment_file}")
                updated_count += 1

            # Also remap alternatives if they exist
            if hasattr(match_result, 'alternatives'):
                for alt in match_result.alternatives:
                    if hasattr(alt, 'video_segment') and alt.video_segment:
                        video_id = _extract_video_id(alt.video_segment.source_file)
                        if video_id:
                            key = (video_id, alt.video_segment.start_time)
                            if key in segment_map:
                                segment_file, _ = segment_map[key]
                                alt.video_segment.source_file = segment_file
                                updated_count += 1

            # Also remap secondary matches if they exist
            if hasattr(match_result, 'secondary_matches'):
                for sec in match_result.secondary_matches:
                    if hasattr(sec, 'video_segment') and sec.video_segment:
                        video_id = _extract_video_id(sec.video_segment.source_file)
                        if video_id:
                            key = (video_id, sec.video_segment.start_time)
                            if key in segment_map:
                                segment_file, _ = segment_map[key]
                                sec.video_segment.source_file = segment_file
                                updated_count += 1

        # Also remap transcript keys from video_id to actual file paths
        self._remap_transcript_keys(state, downloaded_segments)

        logger.info(f"Remapped {updated_count} match video references to segment files")
        print(f"  + Updated {updated_count} match references to video segments")

    def _remap_transcript_keys(
        self,
        state: 'PipelineState',
        downloaded_segments: List
    ) -> None:
        """
        Remap transcript dictionary keys from video_id to actual file paths.

        In caption-first mode, transcripts are initially keyed by video_id.
        After downloading segments, we need to remap them to actual file paths
        so OTIO generation can find the transcripts.
        """
        from ..downloader.segment_utils import _extract_video_id

        # Build mapping: video_id -> set of file paths
        video_id_to_files = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            if video_id not in video_id_to_files:
                video_id_to_files[video_id] = set()
            video_id_to_files[video_id].add(seg.file)

        # Remap transcript keys
        new_transcripts = {}
        remapped_count = 0

        for key, segments in state.transcripts.items():
            # Check if this key is a video_id
            video_id = _extract_video_id(key)

            # If we have downloaded files for this video_id, remap to the first file
            if video_id and video_id in video_id_to_files:
                # Use the first file as the canonical key
                file_paths = list(video_id_to_files[video_id])
                new_key = file_paths[0]

                # Add transcript for all downloaded files from this video
                for file_path in file_paths:
                    new_transcripts[file_path] = segments

                # Keep the old video_id key for backward compatibility
                new_transcripts[video_id] = segments

                logger.debug(f"Remapped transcript key: {key} -> {new_key} (and {len(file_paths)-1} other segments)")
                remapped_count += 1
            else:
                # Not a video_id or no downloaded files, keep as is
                new_transcripts[key] = segments

        state.transcripts = new_transcripts
        logger.info(f"Remapped {remapped_count} transcript keys to file paths")

    def _remap_text_metadata_caption_first(
        self,
        state: 'PipelineState',
        downloaded_segments: List
    ) -> None:
        """
        Update text_metadata to reference video files instead of video IDs.

        In caption-first mode, text_metadata initially references video_ids (11-char strings).
        After downloading segments, we need to remap them to actual .mp4 files.

        Args:
            state: Pipeline state containing text_metadata
            downloaded_segments: List of DownloadedSegment objects
        """
        from ..downloader.segment_utils import _extract_video_id

        if not state.text_metadata:
            return

        # Build mapping: video_id -> set of video_segment_files
        video_id_to_files = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            if video_id not in video_id_to_files:
                video_id_to_files[video_id] = set()
            video_id_to_files[video_id].add(seg.file)

        # Remap text_metadata entries
        updated_count = 0
        for meta in state.text_metadata:
            if not isinstance(meta, dict):
                continue

            video_path = meta.get('video_path', '')
            if not video_path:
                continue

            # Check if this is a video_id that has downloaded segments
            video_id = _extract_video_id(video_path)
            if video_id and video_id in video_id_to_files:
                # Prefer the first video segment file
                video_files = list(video_id_to_files[video_id])
                new_path = video_files[0]
                meta['video_path'] = new_path
                logger.debug(f"Remapped text_metadata: {video_path} -> {new_path}")
                updated_count += 1

        if updated_count > 0:
            logger.info(f"Remapped {updated_count} text_metadata entries from video IDs to video files")
            print(f"  ✓ Updated {updated_count} text_metadata entries to reference video files")
