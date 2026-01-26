"""
LLM-based title filtering and metadata search.

Migrated from VideoDownloader search and filter methods (lines 582-842).
Already uses unified src.llm_client (Rule 9 compliant).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional

if TYPE_CHECKING:
    from ..config import Config
    from .impersonation import ImpersonationManager
    from .escalation_manager import EscalationManager

logger = logging.getLogger(__name__)


class SearchResult:
    """Result of a search operation with metadata about the search itself.

    Attributes:
        videos: List of video metadata dicts
        timed_out: True if the search timed out
        error: Error message if search failed for other reasons
    """
    __slots__ = ('videos', 'timed_out', 'error')

    def __init__(self, videos: List[Dict], timed_out: bool = False, error: Optional[str] = None):
        self.videos = videos
        self.timed_out = timed_out
        self.error = error

    def __bool__(self) -> bool:
        """Returns True if there are videos (for backward compatibility)."""
        return bool(self.videos)


class TitleFilter:
    """Filters video titles using LLM relevance ranking.

    Migrated from VideoDownloader LLM filter methods.
    Already uses unified LLM client (Rule 9 compliant).
    """

    def __init__(
        self,
        config: 'Config',
        cookies_args: List[str],
        get_tier_value_func,
        impersonation_manager: Optional['ImpersonationManager'] = None,
        escalation_manager: Optional['EscalationManager'] = None,
    ):
        """
        Initialize TitleFilter.

        Args:
            config: Config object with download.llm_title_filter settings
            cookies_args: Cookie arguments for yt-dlp
            get_tier_value_func: Function to get tier config values (from CheckpointManager)
            impersonation_manager: Optional ImpersonationManager for TLS fingerprint bypass
            escalation_manager: Optional EscalationManager for 3-tier bypass orchestration
        """
        self.config = config
        self.download_config = config.download
        self.cookies_args = cookies_args
        self._get_tier_value = get_tier_value_func
        self.impersonation_manager = impersonation_manager
        self.escalation_manager = escalation_manager

    def search_video_metadata(
        self,
        keyword: str,
        tier: str,
        max_results: int = 50
    ) -> SearchResult:
        """
        Search YouTube and get video metadata WITHOUT downloading.

        Migrated from downloader.py lines 582-640.

        Used for LLM title filtering.

        Args:
            keyword: Search keyword
            tier: Duration tier (short, medium, long, longer)
            max_results: Maximum search results

        Returns:
            SearchResult object containing:
            - videos: List of dicts with: id, title, duration, channel, url
            - timed_out: True if search timed out
            - error: Error message if search failed for other reasons
        """
        min_dur = self._get_tier_value(tier, 'min', 0)
        max_dur = self._get_tier_value(tier, 'max', 120)

        cmd = [
            'yt-dlp',
            f'ytsearch{max_results}:{keyword}',
            '--dump-json',  # Get metadata only, no download
            '--flat-playlist',  # Faster - don't extract full info
            '--no-download',
            '--match-filter', f"duration>{min_dur} & duration<{max_dur} & !is_live",
        ]

        # Add escalation/impersonation args before cookies for correct argument ordering
        if self.escalation_manager:
            esc_result = self.escalation_manager.get_escalation_args(keyword)
            if esc_result.args:
                cmd.extend(esc_result.args)
        elif self.impersonation_manager:
            imp_args = self.impersonation_manager.get_impersonate_args()
            if imp_args:
                cmd.extend(imp_args)

        cmd.extend(self.cookies_args)

        logger.debug(f"Searching YouTube: {keyword} (max_results={max_results}, tier={tier}, {min_dur}-{max_dur}s)")

        try:
            # Use config timeout or default (30s for search, separate from download timeout)
            search_timeout = getattr(self.download_config, 'search_timeout', 30)

            # Use shell=False for better subprocess handling on Windows
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=search_timeout,
                encoding='utf-8',
                errors='ignore',  # Ignore encoding errors in output
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )

            # Check return code
            if result.returncode != 0:
                logger.warning(f"yt-dlp exited with code {result.returncode} for '{keyword}'")
                if result.stderr:
                    logger.warning(f"  Error: {result.stderr[:200]}")

            # Log stderr if present (helps diagnose issues)
            if result.stderr:
                # Only log warnings/errors, not progress bars
                stderr_lines = [line for line in result.stderr.split('\n')
                               if line and not line.startswith('[download]')
                               and not line.startswith('[youtube]')
                               and ('WARNING' in line or 'ERROR' in line)]
                if stderr_lines:
                    logger.warning(f"yt-dlp stderr for '{keyword}': {stderr_lines[0]}")

            videos = []
            stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
            logger.debug(f"  Received {len(stdout_lines)} lines of output")

            for line in stdout_lines:
                if line:
                    try:
                        data = json.loads(line)
                        videos.append({
                            'id': data.get('id', ''),
                            'title': data.get('title', ''),
                            'duration': data.get('duration', 0),
                            'channel': data.get('channel', data.get('uploader', '')),
                            'url': f"https://www.youtube.com/watch?v={data.get('id', '')}"
                        })
                    except json.JSONDecodeError:
                        continue

            logger.debug(f"  Parsed {len(videos)} valid videos")
            return SearchResult(videos=videos, timed_out=False)

        except subprocess.TimeoutExpired as e:
            logger.warning(f"Search timeout for '{keyword}' (>{search_timeout}s)")
            logger.debug(f"  Command: {' '.join(cmd[:5])}... (cookies arg present: {any('cookie' in arg for arg in cmd)})")
            return SearchResult(videos=[], timed_out=True, error=f"Search timeout after {search_timeout}s")
        except Exception as e:
            logger.warning(f"Error searching metadata: {e}")
            return SearchResult(videos=[], timed_out=False, error=str(e))

    def filter_titles_with_llm(
        self,
        videos: List[Dict],
        keyword: str,
        topic: str = ""
    ) -> List[Dict]:
        """
        Filter and RANK video titles using LLM to check relevance.

        Migrated from downloader.py lines 642-786.

        Args:
            videos: List of video metadata dicts
            keyword: The search keyword
            topic: Optional topic context

        Returns:
            List of approved videos, SORTED by relevance score (highest first)
        """
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        if not llm_config or not getattr(llm_config, 'enabled', False):
            return videos  # Return all if LLM filter disabled

        if not videos:
            return []

        provider = getattr(llm_config, 'provider', 'gemini')
        model = getattr(llm_config, 'model', 'gemini-2.0-flash')
        min_relevance = getattr(llm_config, 'min_relevance', 0.7)
        batch_size = getattr(llm_config, 'batch_size', 20)

        approved = []

        # Process in batches
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]

            # Build prompt - now includes relevance scoring
            titles_list = "\n".join([f"{j+1}. {v['title']}" for j, v in enumerate(batch)])

            prompt = f"""You are filtering and ranking YouTube video titles for a video editing project.

SEARCH KEYWORD: "{keyword}"
{f'TOPIC CONTEXT: {topic}' if topic else ''}

VIDEO TITLES:
{titles_list}

For each title, determine if it would provide relevant B-roll footage for the keyword/topic.
Also rate its relevance from 0.0 to 1.0 (higher = more relevant/useful footage).

REJECT (relevance=0) videos that are:
- Live streams, webcams, 24/7 streams, live cams
- Sports highlights, game recaps, match footage
- Music videos, lyric videos, karaoke
- Gaming content, Let's Play, walkthroughs
- Personal vlogs unrelated to the topic
- News commentary/opinion pieces (unless specifically needed)
- Reaction videos
- Compilations of memes/fails

APPROVE and RATE videos that are:
- Documentary or educational content (0.8-1.0)
- Stock footage, travel footage, city views (0.7-0.9)
- Nature, landscapes, aerial shots (0.6-0.8)
- Professional productions about the topic (0.8-1.0)
- News reports with actual footage (0.6-0.8)
- Explainer videos with relevant visuals (0.5-0.7)

Respond with a JSON array of objects, one per video:
[
  {{"index": 1, "approve": true, "relevance": 0.9, "reason": "Documentary about topic"}},
  {{"index": 2, "approve": false, "relevance": 0.0, "reason": "Sports highlights"}}
]

Only output the JSON array, no other text."""

            try:
                if provider == 'gemini':
                    response = self._call_gemini(prompt, model)
                else:
                    response = self._call_anthropic(prompt, model)

                # Parse response - extract JSON array
                # Strip markdown code blocks if present
                clean_response = response.strip()
                if clean_response.startswith('```'):
                    # Remove ```json or ``` prefix and trailing ```
                    lines = clean_response.split('\n')
                    if lines[0].startswith('```'):
                        lines = lines[1:]  # Remove opening ```json
                    if lines and lines[-1].strip() == '```':
                        lines = lines[:-1]  # Remove closing ```
                    clean_response = '\n'.join(lines)

                json_match = re.search(r'\[[\s\S]*\]', clean_response)
                if not json_match:
                    logger.warning(f"No JSON array found in LLM response (len={len(response)})")
                    raise ValueError("No JSON array in response")

                json_str = json_match.group()
                try:
                    results = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix truncated JSON by closing brackets
                    json_str = json_str.rstrip()
                    if not json_str.endswith(']'):
                        # Find last complete object
                        last_brace = json_str.rfind('}')
                        if last_brace > 0:
                            json_str = json_str[:last_brace + 1] + ']'
                            try:
                                results = json.loads(json_str)
                                logger.debug("Fixed truncated JSON response")
                            except json.JSONDecodeError:
                                raise
                        else:
                            raise
                    else:
                        raise

                for result in results:
                    idx = result.get('index', 0) - 1
                    if 0 <= idx < len(batch) and result.get('approve', False):
                        video = batch[idx]
                        video['llm_reason'] = result.get('reason', 'Approved')
                        # Store relevance score for ranking (default 0.7 for backwards compat)
                        video['llm_relevance'] = float(result.get('relevance', 0.7))
                        approved.append(video)
                        logger.debug(f"    ✓ Approved ({video['llm_relevance']:.1f}): {video['title'][:50]}...")
                    elif 0 <= idx < len(batch):
                        logger.debug(f"    ✗ Rejected: {batch[idx]['title'][:50]}... ({result.get('reason', 'No reason')})")

            except Exception as e:
                logger.warning(f"LLM title filter error: {e}")
                # On error, approve all in batch with default relevance (fail open)
                for v in batch:
                    v['llm_relevance'] = 0.5  # Lower default for error case
                approved.extend(batch)

        # SORT by relevance score (highest first) before returning
        approved.sort(key=lambda v: v.get('llm_relevance', 0.5), reverse=True)

        logger.info(f"    LLM filter: {len(approved)}/{len(videos)} videos approved (sorted by relevance)")
        return approved

    def _call_gemini(self, prompt: str, model: str) -> str:
        """
        Call Gemini API for title filtering.

        Migrated from downloader.py lines 788-811.
        Uses unified LLM client (Rule 9 compliant).

        Args:
            prompt: LLM prompt
            model: Gemini model name

        Returns:
            LLM response text
        """
        try:
            from src.llm_client import create_client, LLMRequest, ResponseFormat

            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logger.warning("No Gemini API key found")
                return "[]"

            client = create_client("gemini", api_key=api_key, model=model)
            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.JSON_ARRAY,
                cache_key_prefix="title_filter"
            )
            response = client.generate(request)

            # Return raw text for backward compatibility
            return response.text

        except Exception as e:
            logger.warning(f"Gemini API error: {e}")
            return "[]"

    def _call_anthropic(self, prompt: str, model: str) -> str:
        """
        Call Anthropic API for title filtering.

        Migrated from downloader.py lines 813-842.
        Uses unified LLM client (Rule 9 compliant).

        Args:
            prompt: LLM prompt
            model: Anthropic model name

        Returns:
            LLM response text
        """
        try:
            from src.llm_client import create_client, LLMRequest, ResponseFormat

            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("No Anthropic API key found")
                return "[]"

            # Get max_tokens from config if available
            max_tokens = 2000
            if hasattr(self.config, 'llm'):
                max_tokens = getattr(self.config.llm, 'max_tokens', 2000)

            client = create_client("anthropic", api_key=api_key, model=model)
            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.JSON_ARRAY,
                cache_key_prefix="title_filter",
                max_tokens=max_tokens
            )
            response = client.generate(request)

            # Return raw text for backward compatibility
            return response.text

        except Exception as e:
            logger.warning(f"Anthropic API error: {e}")
            return "[]"
