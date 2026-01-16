"""
Video Metadata Stage - Fetch video URLs and metadata without downloading.

This stage is the entry point for caption-first mode. It:
1. Searches YouTube for video candidates based on keywords
2. Fetches metadata (title, duration, channel) without downloading
3. Populates state.video_candidates for downstream stages

The CAPTION stage then fetches captions for these candidates,
and only videos without captions go to DOWNLOAD for audio/video download.

This enables massive bandwidth savings since caption fetch is ~1KB vs
audio download being 5-50MB per video.
"""

from __future__ import annotations

import json
import subprocess
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..state import PipelineState
    from ..checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


@register_stage
class VideoMetadataStage(Stage):
    """
    Fetch video metadata without downloading media.

    In caption-first mode, this stage replaces the download-centric approach
    with a metadata-first approach:

    1. Search YouTube for each keyword
    2. Apply duration tier filters
    3. Apply title blacklist filters
    4. Store candidates in state.video_candidates
    5. Exit without downloading anything

    The CAPTION stage then processes these candidates.

    Inputs:
        - state.keywords: List of search keywords
        - state.topic_context: Topic for search refinement
        - config.download.caption_first: Caption-first configuration

    Outputs:
        - state.video_candidates: List of VideoCandidate objects
    """

    name = "VIDEO_METADATA"
    description = "Search YouTube and fetch video metadata without downloading"

    def __init__(self):
        """Initialize VideoMetadataStage."""
        super().__init__()
        self._cookies_args: List[str] = []

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """
        Fetch video metadata for all keywords.

        Args:
            state: Pipeline state with keywords
            config: Config with download settings
            checkpoint: Checkpoint manager for saving progress

        Returns:
            StageResult with success/failure and data for checkpointing
        """
        warnings = []

        # Check if caption-first mode is enabled
        caption_first_config = self._get_caption_first_config(config)
        # Handle both dict and object config patterns (Rule 6)
        if not caption_first_config:
            is_enabled = False
        elif isinstance(caption_first_config, dict):
            is_enabled = caption_first_config.get('enabled', False)
        else:
            is_enabled = getattr(caption_first_config, 'enabled', False)
        if not is_enabled:
            logger.info("Caption-first mode disabled, skipping VIDEO_METADATA stage")
            return StageResult.ok({'skipped': True, 'reason': 'not_enabled'}, warnings)

        # Validate inputs
        if not state.keywords:
            logger.warning("No keywords available for video search")
            return StageResult.ok({'skipped': True, 'reason': 'no_keywords'}, warnings)

        try:
            # Setup cookie args for yt-dlp
            self._setup_cookies(config)

            # Get duration tier config
            tier_config = self._get_tier_config(config)

            # Process each keyword
            total_candidates = 0
            keywords = state.keywords
            topic = state.topic_context or ""

            print(f"  >> Searching {len(keywords)} keywords for video candidates...")
            logger.info(f"VIDEO_METADATA: Searching {len(keywords)} keywords for video candidates")

            for i, keyword in enumerate(keywords):
                print(f"  >> [{i+1}/{len(keywords)}] Searching: {keyword}")
                logger.info(f"  [{i+1}/{len(keywords)}] Searching: {keyword}")

                candidates = self._search_keyword(
                    keyword=keyword,
                    topic=topic,
                    tier_config=tier_config,
                    config=config
                )

                if candidates:
                    # Add to state, avoiding duplicates by video_id
                    new_count = 0
                    for candidate in candidates:
                        if not any(vc.video_id == candidate.video_id for vc in state.video_candidates):
                            state.video_candidates.append(candidate)
                            new_count += 1

                    total_candidates += new_count
                    logger.info(f"    Found {len(candidates)} candidates (+{new_count} new, {total_candidates} total)")
                else:
                    warnings.append(f"No candidates found for '{keyword}'")
                    logger.warning(f"    No candidates found for '{keyword}'")

                # Rate limiting between keywords
                delay = getattr(config.download, 'delay_between_keywords', 1.0)
                if i < len(keywords) - 1 and delay > 0:
                    time.sleep(delay)

            print(f"  >> Found {total_candidates} unique video candidates")
            logger.info(f"VIDEO_METADATA complete: {total_candidates} video candidates from {len(keywords)} keywords")

            # Return full checkpoint data in StageResult (pipeline saves this)
            checkpoint_data = {
                'candidate_count': len(state.video_candidates),
                'keywords_processed': len(keywords),
                'video_ids': [vc.video_id for vc in state.video_candidates],
                'video_candidates': [vc.to_dict() for vc in state.video_candidates]
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"VIDEO_METADATA stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if video metadata stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """
        Restore video candidates from checkpoint.

        Args:
            state: Pipeline state to restore into
            checkpoint: Checkpoint manager with saved data
            config: Optional config object

        Returns:
            True if restoration successful
        """
        from ..state import VideoCandidate

        try:
            stage_data = checkpoint.get_stage_data(self.name)
            if not stage_data:
                return False

            # First try to restore full video_candidates
            video_candidates_data = stage_data.get('video_candidates', [])
            if video_candidates_data:
                for vc_data in video_candidates_data:
                    if isinstance(vc_data, dict):
                        video_id = vc_data.get('video_id', '')
                        if video_id and not any(vc.video_id == video_id for vc in state.video_candidates):
                            state.video_candidates.append(VideoCandidate.from_dict(vc_data))

                logger.info(f"Restored {len(state.video_candidates)} video candidates from checkpoint")
                return True

            # Fallback: restore just video IDs
            video_ids = stage_data.get('video_ids', [])
            if video_ids:
                logger.info(f"Restoring {len(video_ids)} video candidates (IDs only) from checkpoint")

                for video_id in video_ids:
                    if video_id and not any(vc.video_id == video_id for vc in state.video_candidates):
                        state.video_candidates.append(VideoCandidate(
                            video_id=video_id,
                            url=f"https://www.youtube.com/watch?v={video_id}"
                        ))

                logger.info(f"  Restored {len(state.video_candidates)} candidates")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to restore VIDEO_METADATA: {e}")
            return False

    def validate_inputs(self, state: 'PipelineState', config: 'Config') -> Optional[str]:
        """Validate that required inputs are present."""
        caption_first_config = self._get_caption_first_config(config)
        # Handle both dict and object config patterns (Rule 6)
        if not caption_first_config:
            is_enabled = False
        elif isinstance(caption_first_config, dict):
            is_enabled = caption_first_config.get('enabled', False)
        else:
            is_enabled = getattr(caption_first_config, 'enabled', False)
        if not is_enabled:
            return None  # Stage will be skipped, no validation needed

        if not state.keywords:
            return "No keywords available. Run ANALYZE stage first."

        return None

    def _search_keyword(
        self,
        keyword: str,
        topic: str,
        tier_config: Dict,
        config: 'Config'
    ) -> List['VideoCandidate']:
        """
        Search YouTube for a keyword and return video candidates.

        Args:
            keyword: Search keyword
            topic: Topic context for filtering
            tier_config: Duration tier configuration
            config: Config object

        Returns:
            List of VideoCandidate objects
        """
        from ..state import VideoCandidate

        # Get tier settings (default to 'medium' tier for caption-first)
        tier = 'medium'
        min_dur = tier_config.get(tier, {}).get('min', 60)
        max_dur = tier_config.get(tier, {}).get('max', 600)

        # Search pool size
        download_config = config.download
        max_results = getattr(download_config, 'search_pool_multiplier', 5) * 10
        max_results = min(max_results, getattr(download_config, 'max_search_pool', 100))

        # Build yt-dlp command
        cmd = [
            'yt-dlp',
            f'ytsearch{max_results}:{keyword}',
            '--dump-json',
            '--flat-playlist',
            '--no-download',
            '--match-filter', f"duration>{min_dur} & duration<{max_dur} & !is_live",
        ]

        # Add base args (JS runtime for challenge solving) and cookies
        from ..downloader.utils import get_ytdlp_base_args
        cmd.extend(get_ytdlp_base_args())
        cmd.extend(self._cookies_args)

        try:
            search_timeout = getattr(download_config, 'search_timeout', 60)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=search_timeout,
                encoding='utf-8',
                errors='ignore',
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            if result.returncode != 0:
                logger.warning(f"yt-dlp search failed for '{keyword}': {result.stderr[:200] if result.stderr else 'unknown'}")

            candidates = []
            stdout_lines = result.stdout.strip().split('\n') if result.stdout else []

            # Get blacklist
            blacklist = getattr(download_config, 'title_blacklist', [])

            for line in stdout_lines:
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    video_id = data.get('id', '')
                    title = data.get('title', '')

                    # Skip if no ID
                    if not video_id:
                        continue

                    # Apply title blacklist
                    if self._is_blacklisted(title, blacklist):
                        logger.debug(f"    Blacklisted: {title[:50]}")
                        continue

                    # Determine duration tier
                    duration = data.get('duration', 0) or 0
                    duration_tier = self._get_duration_tier(duration, tier_config)

                    candidates.append(VideoCandidate(
                        video_id=video_id,
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        title=title,
                        channel=data.get('channel', data.get('uploader', '')),
                        duration=duration,
                        duration_tier=duration_tier,
                        keyword=keyword,
                        upload_date=data.get('upload_date', ''),
                    ))

                except json.JSONDecodeError:
                    continue

            return candidates

        except subprocess.TimeoutExpired:
            logger.warning(f"Search timeout for '{keyword}'")
            return []
        except Exception as e:
            logger.error(f"Search error for '{keyword}': {e}")
            return []

    def _is_blacklisted(self, title: str, blacklist: List[str]) -> bool:
        """Check if title contains blacklisted terms."""
        title_lower = title.lower()
        return any(term.lower() in title_lower for term in blacklist)

    def _get_duration_tier(self, duration: float, tier_config: Dict) -> str:
        """Determine duration tier for a video."""
        for tier_name, tier_data in tier_config.items():
            min_dur = tier_data.get('min', 0)
            max_dur = tier_data.get('max', float('inf'))
            if min_dur <= duration < max_dur:
                return tier_name
        return 'medium'

    def _get_caption_first_config(self, config: 'Config'):
        """Get caption_first config section."""
        download_config = getattr(config, 'download', None)
        if download_config:
            return getattr(download_config, 'caption_first', None)
        return None

    def _get_tier_config(self, config: 'Config') -> Dict:
        """Get duration tier configuration."""
        # Default tier config if not in config
        default_tiers = {
            'short': {'min': 10, 'max': 120},
            'medium': {'min': 120, 'max': 600},
            'long': {'min': 600, 'max': 1500},
            'longer': {'min': 1500, 'max': 3000},
        }

        # Try to get from config
        duration_tiers = getattr(config, 'duration_tiers', None)
        if duration_tiers:
            if isinstance(duration_tiers, dict):
                return duration_tiers
            # Convert from config object to dict
            try:
                return {
                    'short': {'min': getattr(duration_tiers.short, 'min', 10), 'max': getattr(duration_tiers.short, 'max', 120)},
                    'medium': {'min': getattr(duration_tiers.medium, 'min', 120), 'max': getattr(duration_tiers.medium, 'max', 600)},
                    'long': {'min': getattr(duration_tiers.long, 'min', 600), 'max': getattr(duration_tiers.long, 'max', 1500)},
                    'longer': {'min': getattr(duration_tiers.longer, 'min', 1500), 'max': getattr(duration_tiers.longer, 'max', 3000)},
                }
            except AttributeError:
                pass

        return default_tiers

    def _setup_cookies(self, config: 'Config'):
        """Setup cookie arguments for yt-dlp."""
        from ..downloader.utils import get_cookies_args
        self._cookies_args = get_cookies_args(config)
