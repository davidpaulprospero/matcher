"""
Video Search Stage - Search for videos without downloading

Stage 2 of the simplified 7-stage pipeline:
- Searches YouTube for videos based on keywords
- Returns video IDs for caption fetching
- Does NOT download videos (deferred to DOWNLOAD_SEGMENTS after matching)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage, validate_required_state_attrs

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, VideoSearchResult

logger = logging.getLogger(__name__)


@register_stage
class VideoSearchStage(Stage):
    """
    Searches for videos based on keywords without downloading.

    Inputs:
        - state.keywords: List of keywords to search for
        - state.topic_context: Topic for search refinement

    Outputs:
        - state.video_ids: List of YouTube video IDs
        - state.video_search_results: List of VideoSearchResult with metadata
        - state.search_failed_keywords: List of keywords with no results
    """

    name = "VIDEO_SEARCH"
    description = "Search YouTube for videos (no download)"
    DEPENDS_ON = ['ANALYZE']
    PRODUCES = ['video_ids', 'video_search_results', 'search_failed_keywords']

    def __init__(self):
        self.ydl_opts = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video search stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['keywords'], self.name)

        warnings = []

        try:
            print(f"\n  --- Stage 2: VIDEO SEARCH ---")

            # Get search configuration
            search_config = getattr(config.download, 'video_search', None) or {}
            if isinstance(search_config, dict):
                results_per_keyword = search_config.get('results_per_keyword', 20)
                max_total_results = search_config.get('max_total_results', 200)
                search_budget_aware = search_config.get('search_budget_aware', True)
                auto_distribute_budget = search_config.get('auto_distribute_budget', True)
            else:
                results_per_keyword = getattr(search_config, 'results_per_keyword', 20)
                max_total_results = getattr(search_config, 'max_total_results', 200)
                search_budget_aware = getattr(search_config, 'search_budget_aware', True)
                auto_distribute_budget = getattr(search_config, 'auto_distribute_budget', True)

            # Calculate adjusted results_per_keyword when keywords exceed budget capacity
            keyword_count = len(state.keywords)
            effective_results_per_keyword = results_per_keyword
            budget_info = None

            if search_budget_aware and auto_distribute_budget and keyword_count > 0:
                potential_total = results_per_keyword * keyword_count
                if potential_total > max_total_results:
                    # Distribute budget evenly - reduce results per keyword, not skip keywords
                    effective_results_per_keyword = max(1, max_total_results // keyword_count)
                    budget_info = {
                        'original': results_per_keyword,
                        'adjusted': effective_results_per_keyword,
                        'keywords': keyword_count,
                        'max_total': max_total_results
                    }
                    logger.info(
                        f"Budget distribution: {results_per_keyword} * {keyword_count} = {potential_total} "
                        f"exceeds {max_total_results}, adjusted to {effective_results_per_keyword} per keyword"
                    )

            print(f"  Searching for videos: {effective_results_per_keyword} per keyword, max {max_total_results} total")

            all_video_ids = []
            all_search_results = []
            failed_keywords = []

            for idx, keyword in enumerate(state.keywords, 1):
                print(f"\n  [{idx}/{len(state.keywords)}] Searching: {keyword}")

                # Use effective_results_per_keyword for each keyword
                try:
                    results = self._search_keyword(
                        keyword=keyword,
                        config=config,
                        max_results=effective_results_per_keyword,
                        topic=state.topic_context
                    )

                    if results:
                        for r in results:
                            if r['video_id'] not in all_video_ids:
                                all_video_ids.append(r['video_id'])
                                all_search_results.append(r)

                        print(f"    Found {len(results)} videos")
                    else:
                        failed_keywords.append(keyword)
                        print(f"    No results")

                except Exception as e:
                    logger.warning(f"Search failed for '{keyword}': {e}")
                    failed_keywords.append(keyword)
                    warnings.append(f"Search failed for '{keyword}': {e}")

                # Check max total
                if len(all_video_ids) >= max_total_results:
                    print(f"\n  Reached max results limit ({max_total_results})")
                    break

            # Store results in state
            state.video_ids = all_video_ids
            state.video_search_results = self._to_search_results(all_search_results)
            state.search_failed_keywords = failed_keywords

            print(f"\n  + Found {len(all_video_ids)} unique videos")
            if failed_keywords:
                print(f"  - {len(failed_keywords)} keywords had no results")

            # Prepare checkpoint data
            checkpoint_data = {
                'video_ids': all_video_ids,
                'search_results': all_search_results,
                'failed_keywords': failed_keywords,
                'video_count': len(all_video_ids),
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Video search stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _search_keyword(
        self,
        keyword: str,
        config: 'Config',
        max_results: int = 20,
        topic: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube for videos matching a keyword.

        Returns list of dicts with video metadata (no download).
        """
        import yt_dlp

        # Build search query
        search_query = keyword
        if topic:
            search_query = f"{keyword} {topic}"

        # Get duration filter config
        download_config = config.download
        min_duration = getattr(download_config, 'min_duration', 30)
        max_duration = getattr(download_config, 'max_duration', 600)

        # yt-dlp search options (search only, no download)
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',  # Don't download, just get metadata
            'skip_download': True,
            'ignoreerrors': True,
        }

        # Apply impersonation if available
        try:
            from ..downloader.impersonation import ImpersonationManager
            imp_mgr = ImpersonationManager()
            imp_opts = imp_mgr.get_ydl_options(tier=1)
            ydl_opts.update(imp_opts)
        except Exception:
            pass

        results = []
        search_url = f"ytsearch{max_results * 2}:{search_query}"  # Get extra to filter

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(search_url, download=False)
                if not info or 'entries' not in info:
                    return []

                for entry in info.get('entries', []):
                    if not entry:
                        continue

                    video_id = entry.get('id', '')
                    if not video_id:
                        continue

                    # Filter by duration
                    duration = entry.get('duration', 0) or 0
                    if duration < min_duration or duration > max_duration:
                        continue

                    # Apply title blacklist filter
                    title = entry.get('title', '')
                    if self._is_title_blacklisted(title, config):
                        continue

                    results.append({
                        'video_id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': title,
                        'channel': entry.get('channel', entry.get('uploader', '')),
                        'duration': duration,
                        'keyword': keyword,
                    })

                    if len(results) >= max_results:
                        break

            except Exception as e:
                logger.warning(f"yt-dlp search error: {e}")

        return results

    def _is_title_blacklisted(self, title: str, config: 'Config') -> bool:
        """Check if video title matches blacklist patterns"""
        if not title:
            return False

        title_lower = title.lower()

        # Get blacklist from config
        download_config = config.download
        blacklist = getattr(download_config, 'title_blacklist', [])

        if not blacklist:
            # Default blacklist patterns
            blacklist = [
                'reaction', 'review', 'gameplay', 'let\'s play',
                'unboxing', 'tutorial', 'how to', 'vlog',
                'podcast', 'interview', 'documentary'
            ]

        for pattern in blacklist:
            if pattern.lower() in title_lower:
                return True

        return False

    def _to_search_results(self, results: List[Dict]) -> List['VideoSearchResult']:
        """Convert dict results to VideoSearchResult objects"""
        from ..state import VideoSearchResult

        return [
            VideoSearchResult(
                video_id=r['video_id'],
                url=r.get('url', ''),
                title=r.get('title', ''),
                channel=r.get('channel', ''),
                duration=r.get('duration', 0.0),
                keyword=r.get('keyword', ''),
            )
            for r in results
        ]

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if video search stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore video search stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                logger.warning(f"VIDEO_SEARCH restore: expected dict, got {type(data).__name__}")
                return False

            required_keys = {'video_ids'}
            missing = required_keys - set(data.keys())
            if missing:
                logger.warning(f"VIDEO_SEARCH restore: missing required keys: {missing}")
                return False

            if not isinstance(data['video_ids'], list):
                logger.warning(f"VIDEO_SEARCH restore: 'video_ids' expected list, got {type(data['video_ids']).__name__}")
                return False

            # Restore video IDs
            state.video_ids = data.get('video_ids', [])

            # Restore search results
            search_results = data.get('search_results', [])
            state.video_search_results = self._to_search_results(search_results)

            # Restore failed keywords
            state.search_failed_keywords = data.get('failed_keywords', [])

            logger.info(f"Restored VIDEO_SEARCH: {len(state.video_ids)} video IDs")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore VIDEO_SEARCH: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        if not state.keywords:
            return "No keywords available for video search"
        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        return {
            'inputs': 'keywords',
            'outputs': 'video candidates',
            'input_count': len(state.keywords) if state.keywords else 0,
            'output_count': len(state.video_candidates) if state.video_candidates else None,
        }
