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
from ..downloader.search_cache import YouTubeSearchCache

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
            # Initialize caption cache for pre-filtering videos with known no-captions
            from ..cache import CaptionCache
            cache_dir = Path(config.download.root_dir if hasattr(config.download, 'root_dir') else '.cache') / 'captions'
            self._caption_cache = CaptionCache(cache_dir=cache_dir)
            no_caption_count = sum(1 for key in self._caption_cache.index if self._caption_cache.has_no_captions(key))
            self._total_skipped_no_captions = 0  # Track total skipped for summary
            if no_caption_count > 0:
                logger.info(f"[pre-filter] Loaded negative caption cache: {no_caption_count} videos known to have no captions")
                logger.info(f"[pre-filter] These videos will be automatically skipped during candidate selection")
            else:
                logger.debug(f"[pre-filter] No negative caption cache entries found")

            # Setup cookie args for yt-dlp
            self._setup_cookies(config)

            # Get duration tier config
            tier_config = self._get_tier_config(config)

            # Initialize search cache for skipping already-searched keywords
            search_cache_dir = Path(config.download.root_dir if hasattr(config.download, 'root_dir') else '.cache')
            self._search_cache = YouTubeSearchCache(cache_dir=search_cache_dir)

            # Check if bypass_search_cache is set (for iterative recovery mode)
            bypass_cache = getattr(config.download, 'bypass_search_cache', False)

            # Process each keyword
            total_candidates = 0
            cache_hits = 0
            keywords = state.keywords
            topic = state.topic_context or ""

            print(f"  >> Searching {len(keywords)} keywords for video candidates...")
            logger.info(f"VIDEO_METADATA: Searching {len(keywords)} keywords for video candidates")

            for i, keyword in enumerate(keywords):
                # Check search cache first (unless bypassed)
                if not bypass_cache:
                    cached_ids = self._search_cache.get(keyword, tier='metadata', max_age_days=7)
                    if cached_ids:
                        cache_hits += 1
                        logger.info(f"  [{i+1}/{len(keywords)}] CACHE HIT: {keyword} ({len(cached_ids)} videos)")
                        # Convert cached IDs to VideoCandidate objects
                        candidates = self._ids_to_candidates(cached_ids, keyword, config)
                        if candidates:
                            self._add_candidates_to_state(candidates, state)
                            total_candidates += len(candidates)
                            logger.info(f"    Restored {len(candidates)} candidates from cache")
                        continue

                print(f"  >> [{i+1}/{len(keywords)}] Searching: {keyword}")
                logger.info(f"  [{i+1}/{len(keywords)}] Searching: {keyword}")

                candidates = self._search_keyword(
                    keyword=keyword,
                    topic=topic,
                    tier_config=tier_config,
                    config=config
                )

                # Cache the results for future runs
                if candidates and not bypass_cache:
                    video_ids = [c.video_id for c in candidates]
                    self._search_cache.set(keyword, tier='metadata', video_ids=video_ids)

                if candidates:
                    # Add to state, avoiding duplicates by video_id
                    # Also skip videos known to have no captions (negative cache)
                    new_count = 0
                    skipped_no_captions = 0
                    skipped_duplicate = 0
                    for candidate in candidates:
                        # Skip if already in list
                        if any(vc.video_id == candidate.video_id for vc in state.video_candidates):
                            skipped_duplicate += 1
                            continue
                        # Skip if known to have no captions
                        if hasattr(self, '_caption_cache') and self._caption_cache.has_no_captions(candidate.video_id):
                            skipped_no_captions += 1
                            logger.debug(f"[pre-filter] SKIP {candidate.video_id}: no captions (negative cache)")
                            continue
                        state.video_candidates.append(candidate)
                        new_count += 1

                    total_candidates += new_count
                    self._total_skipped_no_captions = getattr(self, '_total_skipped_no_captions', 0) + skipped_no_captions

                    # Detailed logging
                    parts = [f"+{new_count} new"]
                    if skipped_duplicate > 0:
                        parts.append(f"{skipped_duplicate} dup")
                    if skipped_no_captions > 0:
                        parts.append(f"{skipped_no_captions} no-caption")
                    parts.append(f"{total_candidates} total")
                    logger.info(f"    Found {len(candidates)} candidates ({', '.join(parts)})")
                else:
                    warnings.append(f"No candidates found for '{keyword}'")
                    logger.warning(f"    No candidates found for '{keyword}'")

                # Rate limiting between keywords
                delay = getattr(config.download, 'delay_between_keywords', 1.0)
                if i < len(keywords) - 1 and delay > 0:
                    time.sleep(delay)

            # === COMPREHENSIVE LOGGING: VIDEO_METADATA Summary ===
            total_skipped = getattr(self, '_total_skipped_no_captions', 0)
            logger.info("=" * 60)
            logger.info("[video_metadata] VIDEO_METADATA STAGE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"[video_metadata] Keywords processed: {len(keywords)}")
            logger.info(f"[video_metadata] Cache hits: {cache_hits}/{len(keywords)} ({100*cache_hits//len(keywords) if keywords else 0}% saved)")
            logger.info(f"[video_metadata] Unique candidates found: {total_candidates}")
            if total_skipped > 0:
                logger.info(f"[video_metadata] Pre-filtered (no captions): {total_skipped}")
                logger.info(f"[video_metadata]   -> Saved ~{total_skipped * 15}s by skipping known no-caption videos")
            logger.info("=" * 60)

            print(f"  >> Found {total_candidates} unique video candidates")
            if total_skipped > 0:
                print(f"  >> Pre-filtered {total_skipped} videos with no captions (from cache)")
            logger.info(f"VIDEO_METADATA complete: {total_candidates} video candidates from {len(keywords)} keywords")

            # Apply LLM title filter if enabled
            filtered_count = self._apply_llm_filter(state, config, topic)
            if filtered_count > 0:
                print(f"  >> LLM filter removed {filtered_count} irrelevant videos, {len(state.video_candidates)} remaining")
                logger.info(f"LLM title filter: removed {filtered_count}, {len(state.video_candidates)} remaining")

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
            '--match-filter', f"duration>{min_dur} & duration<{max_dur} & !is_live & !was_live",
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

    def _ids_to_candidates(
        self,
        video_ids: List[str],
        keyword: str,
        config: 'Config'
    ) -> List['VideoCandidate']:
        """
        Convert cached video IDs to VideoCandidate objects.

        Creates lightweight candidates from cached IDs. These have minimal
        metadata but are sufficient for the pipeline to proceed.
        """
        from ..state import VideoCandidate

        candidates = []
        for video_id in video_ids:
            candidates.append(VideoCandidate(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title='',  # Will be fetched during caption stage
                channel='',
                duration=0,  # Unknown from cache
                duration_tier='medium',
                keyword=keyword,
                upload_date='',
            ))
        return candidates

    def _add_candidates_to_state(
        self,
        candidates: List['VideoCandidate'],
        state: 'PipelineState'
    ) -> int:
        """
        Add candidates to state, avoiding duplicates.

        Returns number of new candidates added.
        """
        new_count = 0
        for candidate in candidates:
            # Skip if already in list
            if any(vc.video_id == candidate.video_id for vc in state.video_candidates):
                continue
            # Skip if known to have no captions
            if hasattr(self, '_caption_cache') and self._caption_cache.has_no_captions(candidate.video_id):
                logger.debug(f"[pre-filter] SKIP {candidate.video_id}: no captions (negative cache)")
                continue
            state.video_candidates.append(candidate)
            new_count += 1
        return new_count

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

    def _apply_llm_filter(
        self,
        state: 'PipelineState',
        config: 'Config',
        topic: str
    ) -> int:
        """
        Apply LLM title filter to video candidates.

        This filters out videos with irrelevant titles using an LLM to assess
        relevance to the topic context.

        Args:
            state: Pipeline state with video_candidates
            config: Config with download.llm_title_filter settings
            topic: Topic context for relevance checking

        Returns:
            Number of candidates removed
        """
        # Check if LLM filter is enabled
        download_config = getattr(config, 'download', None)
        if not download_config:
            return 0

        llm_config = getattr(download_config, 'llm_title_filter', None)
        if not llm_config or not getattr(llm_config, 'enabled', False):
            logger.debug("LLM title filter disabled, skipping")
            return 0

        if not state.video_candidates:
            return 0

        original_count = len(state.video_candidates)

        # Convert VideoCandidate objects to dicts for the filter
        video_dicts = []
        for vc in state.video_candidates:
            video_dicts.append({
                'id': vc.video_id,
                'title': vc.title or '',
                'channel': vc.channel or '',
                'duration': vc.duration or 0,
                'keyword': vc.keyword or '',
            })

        # Get filter settings
        provider = getattr(llm_config, 'provider', 'gemini')
        model = getattr(llm_config, 'model', 'gemini-2.0-flash')
        min_relevance = getattr(llm_config, 'min_relevance', 0.5)
        batch_size = getattr(llm_config, 'batch_size', 20)

        # Process in batches
        approved_ids = set()

        for i in range(0, len(video_dicts), batch_size):
            batch = video_dicts[i:i + batch_size]
            batch_approved = self._filter_batch_with_llm(
                batch, topic, provider, model, min_relevance
            )
            approved_ids.update(batch_approved)

        # Filter state.video_candidates to only approved ones
        state.video_candidates = [
            vc for vc in state.video_candidates
            if vc.video_id in approved_ids
        ]

        return original_count - len(state.video_candidates)

    def _filter_batch_with_llm(
        self,
        videos: List[Dict],
        topic: str,
        provider: str,
        model: str,
        min_relevance: float
    ) -> set:
        """
        Filter a batch of videos using LLM.

        Args:
            videos: List of video dicts with id, title, channel, duration
            topic: Topic context
            provider: LLM provider (gemini, anthropic)
            model: Model name
            min_relevance: Minimum relevance score (0-1)

        Returns:
            Set of approved video IDs
        """
        import json as json_module

        if not videos:
            return set()

        # Build video data string
        video_lines = []
        for v in videos:
            line = f"- ID: {v['id']} | Title: {v['title']}"
            if v.get('channel'):
                line += f" | Channel: {v['channel']}"
            if v.get('duration'):
                line += f" | Duration: {v['duration']}s"
            video_lines.append(line)

        video_data = '\n'.join(video_lines)

        prompt = f"""You are filtering YouTube videos for a documentary/educational video project.

TOPIC: {topic if topic else 'General educational content'}

VIDEO DATA:
{video_data}

For each video, rate relevance from 0.0 to 1.0 (higher = more relevant footage for the topic).

REJECT (relevance=0) videos that are:
- Audio tests, frequency tests, white noise, ASMR, sleep sounds
- Live streams, webcams, 24/7 streams, ambient cameras
- Music videos, lyric videos, karaoke, song covers
- Gaming content, Let's Play, walkthroughs, game clips
- Reaction videos, commentary on other videos
- Compilation of memes, fails, or unrelated clips
- Videos completely unrelated to the topic

ACCEPT (relevance>0.5) videos with:
- Documentary footage, news reports about the topic
- Educational content, explainers, analysis
- Relevant B-roll footage (establishments, locations, events)
- Interviews, discussions related to the topic

Return a JSON array of objects with "id" and "relevance" (0.0-1.0).
Example: [{{"id": "abc123", "relevance": 0.8}}, {{"id": "xyz789", "relevance": 0.0}}]

ONLY return the JSON array, no other text."""

        try:
            # Use LLM client
            from ..llm_client import create_client, LLMRequest, ResponseFormat

            client = create_client(provider=provider, model=model)
            request = LLMRequest(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1,
                response_format=ResponseFormat.JSON_ARRAY,
                cache_key_prefix="title_filter",
                use_cache=True
            )
            response = client.generate(request)

            if not response or not response.text:
                logger.warning("LLM filter returned empty response, approving all")
                return {v['id'] for v in videos}

            # Parse JSON response
            # Extract JSON array from response (handle markdown code blocks)
            response_text = response.text.strip()
            if response_text.startswith('```'):
                # Remove markdown code block
                lines = response_text.split('\n')
                response_text = '\n'.join(
                    line for line in lines
                    if not line.startswith('```')
                )

            results = json_module.loads(response_text)

            # Extract approved IDs
            approved = set()
            for item in results:
                if isinstance(item, dict):
                    vid_id = item.get('id', '')
                    relevance = item.get('relevance', 0)
                    if relevance >= min_relevance:
                        approved.add(vid_id)
                    else:
                        logger.debug(f"    Filtered: {vid_id} (relevance={relevance:.2f})")

            logger.info(f"    LLM filter: {len(approved)}/{len(videos)} approved (min_relevance={min_relevance})")
            return approved

        except json_module.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM filter response: {e}")
            # On parse error, approve all to avoid data loss
            return {v['id'] for v in videos}
        except Exception as e:
            logger.warning(f"LLM filter error: {e}")
            # On error, approve all to avoid data loss
            return {v['id'] for v in videos}
