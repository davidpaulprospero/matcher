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
                enable_channel_diversity = search_config.get('enable_channel_diversity', True)
                max_videos_per_channel = search_config.get('max_videos_per_channel', 3)
                use_negative_context = search_config.get('use_negative_context', False)
                negative_keywords = search_config.get('negative_keywords', []) or []
                use_chapter_queries = search_config.get('use_chapter_queries', True)
            else:
                results_per_keyword = getattr(search_config, 'results_per_keyword', 20)
                max_total_results = getattr(search_config, 'max_total_results', 200)
                search_budget_aware = getattr(search_config, 'search_budget_aware', True)
                auto_distribute_budget = getattr(search_config, 'auto_distribute_budget', True)
                enable_channel_diversity = getattr(search_config, 'enable_channel_diversity', True)
                max_videos_per_channel = getattr(search_config, 'max_videos_per_channel', 3)
                use_negative_context = getattr(search_config, 'use_negative_context', False)
                negative_keywords = getattr(search_config, 'negative_keywords', []) or []
                use_chapter_queries = getattr(search_config, 'use_chapter_queries', True)

            # US-98-005: Build chapter-specific search queries
            chapter_queries = []
            if use_chapter_queries and hasattr(state, 'location_chapters') and state.location_chapters:
                chapter_queries = self._build_chapter_queries(state.location_chapters, state.topic_context)
                if chapter_queries:
                    logger.info(f"US-98-005: Generated {len(chapter_queries)} chapter-specific queries")

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

            # US-98-005: First, search using standard keywords
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

            # US-98-005: Search using chapter-specific queries
            if chapter_queries:
                print(f"\n  --- Chapter-specific search ({len(chapter_queries)} queries) ---")
                for idx, cq in enumerate(chapter_queries, 1):
                    keyword = cq['keyword']
                    chapter_id = cq['chapter_id']
                    chapter_title = cq['chapter_title']

                    # Check if we still have budget
                    if len(all_video_ids) >= max_total_results:
                        print(f"  Reached max results limit, skipping remaining chapter queries")
                        break

                    print(f"\n  [{idx}/{len(chapter_queries)}] Chapter '{chapter_title}': {keyword}")

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
                                    # Tag result with source chapter
                                    r['chapter_id'] = chapter_id
                                    r['chapter_title'] = chapter_title
                                    all_video_ids.append(r['video_id'])
                                    all_search_results.append(r)

                            print(f"    Found {len(results)} videos")
                        else:
                            print(f"    No results")

                    except Exception as e:
                        logger.warning(f"Chapter search failed for '{keyword}': {e}")
                        warnings.append(f"Chapter search failed for '{keyword}': {e}")

            # Apply channel diversity filtering (US-94-009)
            if enable_channel_diversity and all_search_results:
                channel_counts: Dict[str, int] = {}
                filtered_ids = []
                filtered_results = []

                for r in all_search_results:
                    channel = r.get('channel', '')
                    if not channel:
                        # Include videos without channel info
                        filtered_ids.append(r['video_id'])
                        filtered_results.append(r)
                        continue

                    current_count = channel_counts.get(channel, 0)
                    if current_count < max_videos_per_channel:
                        channel_counts[channel] = current_count + 1
                        filtered_ids.append(r['video_id'])
                        filtered_results.append(r)
                    else:
                        logger.debug(f"Skipping video {r['video_id']} from channel '{channel}' (max {max_videos_per_channel} reached)")

                removed_count = len(all_video_ids) - len(filtered_ids)
                if removed_count > 0:
                    print(f"  - Removed {removed_count} videos due to channel diversity limit ({max_videos_per_channel} per channel)")

                all_video_ids = filtered_ids
                all_search_results = filtered_results

            # Store results in state
            state.video_ids = all_video_ids
            state.video_search_results = self._to_search_results(
                all_search_results,
                negative_keywords=negative_keywords if use_negative_context else []
            )
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
        Uses two-phase search: initial + refined with description keywords.
        """
        import yt_dlp

        # Get config
        download_config = config.download
        video_search_config = download_config.video_search

        # Handle both dict and object access patterns (Rule #6)
        if isinstance(video_search_config, dict):
            use_tags = video_search_config.get('use_tags_in_search', True)
            use_description_context = video_search_config.get('use_description_context', True)
            use_negative_context = video_search_config.get('use_negative_context', False)
            negative_keywords = video_search_config.get('negative_keywords', []) or []
            topic_tags_map = video_search_config.get('topic_tags', {}) or {}
        else:
            use_tags = getattr(video_search_config, 'use_tags_in_search', True)
            use_description_context = getattr(video_search_config, 'use_description_context', True)
            use_negative_context = getattr(video_search_config, 'use_negative_context', False)
            negative_keywords = getattr(video_search_config, 'negative_keywords', []) or []
            topic_tags_map = getattr(video_search_config, 'topic_tags', {}) or {}

        # Build initial search query
        search_query = self._build_search_query(keyword, topic, use_tags, topic_tags_map)

        # Get duration filter config
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

        # Phase 1: Initial search
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(search_url, download=False)
                if not info or 'entries' not in info:
                    return []

                initial_results = []
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

                    # US-95-012: Apply negative context filtering
                    if use_negative_context and negative_keywords:
                        description = entry.get('description', '')
                        if self._is_negative_matched(title, description, negative_keywords):
                            logger.debug(f"Filtered out video '{title}' due to negative keywords")
                            continue

                    initial_results.append({
                        'video_id': video_id,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'title': title,
                        'channel': entry.get('channel', entry.get('uploader', '')),
                        'duration': duration,
                        'description': entry.get('description', ''),  # US-95-003: Capture description
                        'keyword': keyword,
                    })

            except Exception as e:
                logger.warning(f"yt-dlp search error: {e}")

        # US-95-003: Extract description keywords and refine search
        if use_description_context and initial_results:
            # Extract keywords from top descriptions
            desc_keywords = self._extract_description_keywords(initial_results)

            if desc_keywords:
                # Build refined query with description keywords (weighted lower)
                # Description keywords go at the end - they're optional refinements
                refined_query = f"{keyword} {' '.join(desc_keywords[:2])}"
                refined_url = f"ytsearch{max_results}:{refined_query}"

                logger.info(f"Description-refined query: '{keyword}' -> '{refined_query}'")

                try:
                    info = ydl.extract_info(refined_url, download=False)
                    if info and 'entries' in info:
                        for entry in info.get('entries', []):
                            if not entry:
                                continue

                            video_id = entry.get('id', '')
                            if not video_id:
                                continue

                            duration = entry.get('duration', 0) or 0
                            if duration < min_duration or duration > max_duration:
                                continue

                            title = entry.get('title', '')
                            if self._is_title_blacklisted(title, config):
                                continue

                            # US-95-012: Apply negative context filtering
                            if use_negative_context and negative_keywords:
                                description = entry.get('description', '')
                                if self._is_negative_matched(title, description, negative_keywords):
                                    logger.debug(f"Filtered out refined video '{title}' due to negative keywords")
                                    continue

                            # Add refined results (may include duplicates)
                            initial_results.append({
                                'video_id': video_id,
                                'url': f"https://www.youtube.com/watch?v={video_id}",
                                'title': title,
                                'channel': entry.get('channel', entry.get('uploader', '')),
                                'duration': duration,
                                'description': entry.get('description', ''),
                                'keyword': keyword,
                            })

                except Exception as e:
                    logger.warning(f"yt-dlp refined search error: {e}")

        # Deduplicate and limit results
        seen_ids = set()
        for r in initial_results:
            if r['video_id'] not in seen_ids and len(results) < max_results:
                seen_ids.add(r['video_id'])
                # Remove description from final results (not stored)
                result_clean = {k: v for k, v in r.items() if k != 'description'}
                results.append(result_clean)

        return results

    def _build_search_query(
        self,
        keyword: str,
        topic: str,
        use_tags: bool,
        topic_tags_map: dict
    ) -> str:
        """Build search query with tag expansion"""
        search_query = keyword

        # US-95-002: Expand query with related topic tags
        if use_tags and topic_tags_map:
            expanded_terms = []
            keyword_lower = keyword.lower()

            for topic_key, tags in topic_tags_map.items():
                if topic_key in keyword_lower:
                    expanded_terms.extend(tags)

            if topic:
                topic_lower = topic.lower()
                for topic_key, tags in topic_tags_map.items():
                    if topic_key in topic_lower and tags not in expanded_terms:
                        expanded_terms.extend(tags)

            if expanded_terms:
                seen = set()
                unique_terms = []
                for term in expanded_terms:
                    if term not in seen:
                        seen.add(term)
                        unique_terms.append(term)

                search_query = f"{keyword} {' '.join(unique_terms[:3])}"
                logger.info(f"Tag-expanded query: '{keyword}' -> '{search_query}'")

        if topic and topic not in search_query:
            search_query = f"{search_query} {topic}"

        return search_query

    def _build_chapter_queries(
        self,
        location_chapters: List[Any],
        topic_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        US-98-005: Build search queries from voiceover chapter topics.

        Each voiceover chapter's topics become separate search terms, tagged
        with the source chapter for downstream scoring.

        Args:
            location_chapters: List of ChapterCandidate or dict objects with topics
            topic_context: Overall topic for context

        Returns:
            List of dicts with 'keyword', 'chapter_id', 'chapter_title'
        """
        chapter_queries = []

        for chapter in location_chapters:
            # Extract chapter info - handle both dict and object access
            if isinstance(chapter, dict):
                chapter_id = chapter.get('chapter_id', 0)
                chapter_title = chapter.get('title', f'Chapter {chapter_id}')
                topics = chapter.get('topics', [])
            else:
                chapter_id = getattr(chapter, 'chapter_id', 0)
                chapter_title = getattr(chapter, 'title', f'Chapter {chapter_id}')
                topics = getattr(chapter, 'topics', [])

            # Skip chapters without topics
            if not topics:
                continue

            # Build search query from chapter topics
            # Use first 3 topics as search terms
            search_topics = topics[:3] if len(topics) > 3 else topics

            # Add topic context if available and not already included
            if topic_context:
                topic_lower = topic_context.lower()
                if not any(topic_lower in t.lower() for t in search_topics):
                    search_topics = search_topics + [topic_context]

            # Create keyword from topics
            keyword = ' '.join(search_topics)

            if keyword:
                chapter_queries.append({
                    'keyword': keyword,
                    'chapter_id': chapter_id,
                    'chapter_title': chapter_title,
                    'topics': search_topics,
                })

        return chapter_queries

    def _extract_description_keywords(self, results: List[Dict]) -> List[str]:
        """US-95-003: Extract key terms from video descriptions"""
        if not results:
            return []

        # Collect descriptions from top results
        descriptions = []
        for r in results[:10]:  # Use top 10 results
            desc = r.get('description', '')
            if desc:
                descriptions.append(desc)

        if not descriptions:
            return []

        # Simple keyword extraction: split on whitespace and filter
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
            'very', 'just', 'http', 'https', 'www', 'com', 'watch', 'video', 'like',
            'subscribe', 'channel', 'youtube', 'music', 'song', 'official', 'new'
        }

        word_counts: Dict[str, int] = {}

        for desc in descriptions:
            # Clean: lowercase, remove URLs, special chars
            import re
            cleaned = re.sub(r'http\S+', '', desc)
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            cleaned = cleaned.lower()

            words = cleaned.split()
            for word in words:
                if len(word) >= 3 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        # Return top keywords that appear at least twice
        keywords = [w for w, count in sorted_words if count >= 2][:4]

        return keywords

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

    def _is_negative_matched(
        self,
        title: str,
        description: str,
        negative_keywords: List[str]
    ) -> bool:
        """US-95-012: Check if title or description matches negative keywords"""
        if not title or not negative_keywords:
            return False

        title_lower = title.lower()
        desc_lower = (description or "").lower()

        for keyword in negative_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower or keyword_lower in desc_lower:
                return True

        return False

    def _extract_negative_keywords(
        self,
        title: str,
        description: str,
        negative_keywords: List[str]
    ) -> List[str]:
        """US-95-012: Extract negative keywords found in title/description"""
        if not negative_keywords:
            return []

        title_lower = title.lower()
        desc_lower = (description or "").lower()
        found = []

        for keyword in negative_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower or keyword_lower in desc_lower:
                found.append(keyword)

        return found

    def _to_search_results(
        self,
        results: List[Dict],
        negative_keywords: List[str] = None
    ) -> List['VideoSearchResult']:
        """Convert dict results to VideoSearchResult objects"""
        from ..state import VideoSearchResult

        negative_keywords = negative_keywords or []

        return [
            VideoSearchResult(
                video_id=r['video_id'],
                url=r.get('url', ''),
                title=r.get('title', ''),
                channel=r.get('channel', ''),
                duration=r.get('duration', 0.0),
                keyword=r.get('keyword', ''),
                description=r.get('description', ''),  # US-70-002
                negative_keywords=self._extract_negative_keywords(
                    r.get('title', ''),
                    r.get('description', ''),
                    negative_keywords
                ),  # US-95-012
                chapter_id=r.get('chapter_id', -1),  # US-98-005
                chapter_title=r.get('chapter_title', ''),  # US-98-005
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
