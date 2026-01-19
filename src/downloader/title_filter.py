"""
LLM-based title filtering and metadata search.

Migrated from VideoDownloader search and filter methods (lines 582-842).
Already uses unified src.llm_client (Rule 9 compliant).

Enhanced with:
- YouTube Data API v3 for fast metadata fetching (50 videos/request)
- Content filter presets (documentary, stock_footage, raw)
- Duration-aware filtering
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Any

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# Default presets (loaded from config.yaml at runtime)
DEFAULT_PRESETS = {
    "raw": {
        "description": "No additional content filtering",
        "rejection_prompt": "",
        "acceptance_prompt": "",
    },
    "documentary": {
        "description": "Filter trainers, animated, vlogs for documentary footage",
        "rejection_prompt": """- Professional dog training content (trainers like Cesar Millan, Zac George,
  Victoria Stilwell, Kikopup, McCann, or any channel focused on dog training)
- Training tutorials, obedience lessons, behavior modification guides
- Animated, cartoon, CGI, or 3D animated content
- Personal vlogs, reaction videos, commentary without useful footage
- Long videos (>10 minutes) - likely contain intros, outros, filler content""",
        "acceptance_prompt": """PREFER (in order):
1. YouTube Shorts (<60 seconds) - quick emotional moments, viral clips
2. Medium videos (1-5 minutes) - focused content, specific moments
3. Documentary footage, news coverage, adoption events, shelter footage
AVOID: Long videos (>10 min) unless exceptionally relevant""",
    },
    "stock_footage": {
        "description": "Silent/B-roll footage only",
        "rejection_prompt": """- Any video with significant talking/narration/voiceover
- Vlogs, commentary, reactions, podcasts
- Animated or cartoon content
- Music videos, lyric videos""",
        "acceptance_prompt": "PREFER: Silent footage, nature shots, B-roll, aerial, time-lapse",
    },
}


class TitleFilter:
    """Filters video titles using LLM relevance ranking.

    Migrated from VideoDownloader LLM filter methods.
    Already uses unified LLM client (Rule 9 compliant).

    Enhanced with:
    - YouTube Data API v3 for fast metadata fetching
    - Content filter presets support
    - Duration-aware filtering
    """

    def __init__(self, config: 'Config', cookies_args: List[str], get_tier_value_func):
        """
        Initialize TitleFilter.

        Args:
            config: Config object with download.llm_title_filter settings
            cookies_args: Cookie arguments for yt-dlp
            get_tier_value_func: Function to get tier config values (from CheckpointManager)
        """
        self.config = config
        self.download_config = config.download
        self.cookies_args = cookies_args
        self._get_tier_value = get_tier_value_func

        # Initialize YouTube Data API client (optional - for faster metadata)
        self.youtube = None
        self._init_youtube_api()

        # Load content filter presets from config
        self.presets = self._load_presets()

    def _init_youtube_api(self):
        """Initialize YouTube Data API v3 client if API key is available."""
        api_key = (
            os.environ.get('YOUTUBE_API_KEY') or
            getattr(self.download_config, 'youtube_api_key', '')
        )
        if not api_key:
            logger.debug("No YouTube API key found, will use yt-dlp for metadata")
            return

        try:
            from googleapiclient.discovery import build
            self.youtube = build('youtube', 'v3', developerKey=api_key)
            logger.info("YouTube Data API v3 initialized (fast metadata fetching enabled)")
        except ImportError:
            logger.warning("google-api-python-client not installed, using yt-dlp for metadata")
        except Exception as e:
            logger.warning(f"Failed to initialize YouTube API: {e}")

    def _load_presets(self) -> Dict[str, Dict[str, str]]:
        """Load content filter presets from config or use defaults."""
        presets = DEFAULT_PRESETS.copy()

        # Try to load from config
        if hasattr(self.config, 'content_filter_presets'):
            config_presets = self.config.content_filter_presets
            if isinstance(config_presets, dict):
                for name, preset in config_presets.items():
                    if isinstance(preset, dict):
                        presets[name] = {
                            'description': preset.get('description', ''),
                            'rejection_prompt': preset.get('rejection_prompt', ''),
                            'acceptance_prompt': preset.get('acceptance_prompt', ''),
                        }
        return presets

    def search_video_metadata(
        self,
        keyword: str,
        tier: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Search YouTube and get video metadata WITHOUT downloading.

        Migrated from downloader.py lines 582-640.

        Used for LLM title filtering.

        Args:
            keyword: Search keyword
            tier: Duration tier (short, medium, long, longer)
            max_results: Maximum search results

        Returns:
            List of dicts with: id, title, duration, channel, url
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

        # Add base args (JS runtime for challenge solving) and cookies
        from . import utils
        cmd.extend(utils.get_ytdlp_base_args())
        cmd.extend(self.cookies_args)

        logger.debug(f"Searching YouTube: {keyword} (max_results={max_results}, tier={tier}, {min_dur}-{max_dur}s)")

        try:
            # Use config timeout or default
            search_timeout = getattr(self.download_config, 'search_timeout', 60)

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
            return videos

        except subprocess.TimeoutExpired as e:
            logger.warning(f"Timeout searching metadata for '{keyword}' (>{search_timeout}s)")
            logger.debug(f"  Command: {' '.join(cmd[:5])}... (cookies arg present: {any('cookie' in arg for arg in cmd)})")
            return []
        except Exception as e:
            logger.warning(f"Error searching metadata: {e}")
            return []

    def fetch_full_metadata(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch full metadata (including description) for specific video IDs.

        Uses YouTube Data API v3 if available (fast, batches 50 videos per request).
        Falls back to yt-dlp if no API key (slower, 1 request per video).

        Called AFTER title blacklist filter to minimize API calls.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            Dict mapping video_id to metadata dict with:
            - title, channel, description (first 200 chars), tags, category_id
        """
        if not video_ids:
            return {}

        if self.youtube:
            return self._fetch_with_youtube_api(video_ids)
        else:
            return self._fetch_with_ytdlp(video_ids)

    def _fetch_with_youtube_api(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch metadata using YouTube Data API v3 (fast, batches 50 videos)."""
        metadata = {}

        # Batch in groups of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]
            try:
                request = self.youtube.videos().list(
                    part='snippet',
                    id=','.join(batch)
                )
                response = request.execute()

                for item in response.get('items', []):
                    vid_id = item['id']
                    snippet = item.get('snippet', {})
                    metadata[vid_id] = {
                        'title': snippet.get('title', ''),
                        'channel': snippet.get('channelTitle', ''),
                        'description': snippet.get('description', '')[:200],
                        'tags': snippet.get('tags', []),
                        'category_id': snippet.get('categoryId', ''),
                    }

                logger.debug(f"YouTube API: Fetched metadata for {len(response.get('items', []))}/{len(batch)} videos")

            except Exception as e:
                logger.warning(f"YouTube API error for batch: {e}")
                # Fall back to yt-dlp for this batch
                metadata.update(self._fetch_with_ytdlp(batch))

        return metadata

    def _fetch_with_ytdlp(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fallback: fetch metadata with yt-dlp (slower, no API key needed)."""
        metadata = {}

        for vid_id in video_ids:
            try:
                cmd = [
                    'yt-dlp',
                    f'https://www.youtube.com/watch?v={vid_id}',
                    '--dump-json',
                    '--skip-download',
                    '--no-warnings',
                    '--no-playlist',
                ]
                cmd.extend(self.cookies_args)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='ignore',
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )

                if result.returncode == 0 and result.stdout:
                    data = json.loads(result.stdout)
                    metadata[vid_id] = {
                        'title': data.get('title', ''),
                        'channel': data.get('channel', data.get('uploader', '')),
                        'description': data.get('description', '')[:200],
                        'tags': data.get('tags', []),
                        'category_id': '',  # yt-dlp uses different field
                    }
            except subprocess.TimeoutExpired:
                logger.debug(f"yt-dlp timeout for video {vid_id}")
            except Exception as e:
                logger.debug(f"yt-dlp error for video {vid_id}: {e}")

        logger.debug(f"yt-dlp: Fetched metadata for {len(metadata)}/{len(video_ids)} videos")
        return metadata

    def _format_duration(self, seconds: int) -> str:
        """Format duration with category hint for LLM context."""
        if seconds <= 0:
            return "unknown"
        elif seconds < 60:
            return f"{seconds}s (YouTube Short)"
        elif seconds < 300:
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins}m {secs}s (Short clip)"
        elif seconds < 600:
            return f"{seconds // 60}m (Medium)"
        else:
            return f"{seconds // 60}m (Long - likely contains filler)"

    def _apply_title_blacklist(self, videos: List[Dict]) -> List[Dict]:
        """Apply title blacklist filter (fast, free, no API calls)."""
        blacklist = getattr(self.download_config, 'title_blacklist', [])
        if not blacklist:
            return videos

        filtered = []
        for video in videos:
            title_lower = video.get('title', '').lower()
            channel_lower = video.get('channel', '').lower()

            # Check if any blacklist term is in title or channel
            blocked = False
            for term in blacklist:
                term_lower = term.lower()
                if term_lower in title_lower or term_lower in channel_lower:
                    logger.debug(f"    Blacklist: {video['title'][:50]}... (matched: {term})")
                    blocked = True
                    break

            if not blocked:
                filtered.append(video)

        if len(filtered) < len(videos):
            logger.info(f"    Title blacklist: {len(videos) - len(filtered)} videos filtered out")

        return filtered

    def _get_preset_prompts(self) -> tuple:
        """Get rejection and acceptance prompts from preset + custom config."""
        # Get preset name from config
        preset_name = getattr(self.download_config, 'content_filter_preset', 'raw')
        preset = self.presets.get(preset_name, self.presets.get('raw', {}))

        # Start with preset prompts
        rejection_prompt = preset.get('rejection_prompt', '')
        acceptance_prompt = preset.get('acceptance_prompt', '')

        # Extend with custom prompts from project config
        custom_rejection = getattr(self.download_config, 'custom_rejection_prompt', '')
        custom_acceptance = getattr(self.download_config, 'custom_acceptance_prompt', '')

        if custom_rejection:
            rejection_prompt = f"{rejection_prompt}\n{custom_rejection}" if rejection_prompt else custom_rejection
        if custom_acceptance:
            acceptance_prompt = f"{acceptance_prompt}\n{custom_acceptance}" if acceptance_prompt else custom_acceptance

        if preset_name != 'raw':
            logger.debug(f"Using content filter preset: {preset_name}")

        return rejection_prompt, acceptance_prompt

    def _build_video_data_string(self, videos: List[Dict]) -> str:
        """Build formatted video data string for LLM prompt with full metadata."""
        lines = []
        for i, v in enumerate(videos, 1):
            duration_str = self._format_duration(v.get('duration', 0))
            description = v.get('description', '')[:200]
            if description:
                description = description.replace('\n', ' ')
                if len(description) >= 200:
                    description += '...'

            lines.append(f"""{i}. Title: {v['title']}
   Channel: {v.get('channel', 'Unknown')}
   Duration: {duration_str}
   Description: {description if description else 'N/A'}""")

        return "\n\n".join(lines)

    def filter_titles_with_llm(
        self,
        videos: List[Dict],
        keyword: str,
        topic: str = ""
    ) -> List[Dict]:
        """
        Filter and RANK video titles using LLM to check relevance.

        Enhanced flow:
        1. Apply title blacklist (fast, free)
        2. Fetch full metadata for remaining videos (YouTube API or yt-dlp)
        3. Build prompt with title + channel + duration + description
        4. Apply content filter preset + custom prompts
        5. Send to LLM for filtering and ranking

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

        # Step 1: Apply title blacklist first (fast, no API cost)
        videos = self._apply_title_blacklist(videos)
        if not videos:
            logger.info("    All videos filtered by title blacklist")
            return []

        # Step 2: Fetch full metadata for remaining videos (description, tags)
        video_ids = [v['id'] for v in videos if v.get('id')]
        if video_ids:
            full_metadata = self.fetch_full_metadata(video_ids)

            # Enrich videos with description from full metadata
            for v in videos:
                if v.get('id') in full_metadata:
                    meta = full_metadata[v['id']]
                    v['description'] = meta.get('description', '')
                    # Update channel if we got better info
                    if meta.get('channel') and not v.get('channel'):
                        v['channel'] = meta['channel']

        # Get preset prompts
        rejection_prompt, acceptance_prompt = self._get_preset_prompts()

        provider = getattr(llm_config, 'provider', 'gemini')
        model = getattr(llm_config, 'model', 'gemini-2.0-flash')
        min_relevance = getattr(llm_config, 'min_relevance', 0.7)
        batch_size = getattr(llm_config, 'batch_size', 20)

        approved = []

        # Process in batches
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]

            # Build video data with full metadata
            video_data = self._build_video_data_string(batch)

            # Build prompt with preset criteria
            prompt = f"""You are filtering and ranking YouTube videos for a video editing project.

SEARCH KEYWORD: "{keyword}"
{f'TOPIC CONTEXT: {topic}' if topic else ''}

VIDEO DATA:
{video_data}

For each video, determine if it would provide relevant B-roll footage for the keyword/topic.
Rate relevance from 0.0 to 1.0 (higher = more relevant/useful footage).

REJECT (relevance=0) videos that are:
- Live streams, webcams, 24/7 streams, live cams
- Sports highlights, game recaps, match footage
- Music videos, lyric videos, karaoke
- Gaming content, Let's Play, walkthroughs
- Reaction videos
- Compilations of memes/fails
{rejection_prompt}

APPROVE and RATE videos based on:
- Documentary or educational content (0.8-1.0)
- Stock footage, travel footage, city views (0.7-0.9)
- Nature, landscapes, aerial shots (0.6-0.8)
- Professional productions about the topic (0.8-1.0)
- News reports with actual footage (0.6-0.8)
{acceptance_prompt}

Respond with a JSON array of objects, one per video:
[
  {{"index": 1, "approve": true, "relevance": 0.9, "reason": "Documentary about topic"}},
  {{"index": 2, "approve": false, "relevance": 0.0, "reason": "Professional trainer content"}}
]

Only output the JSON array, no other text."""

            try:
                if provider == 'gemini':
                    response = self._call_gemini(prompt, model)
                else:
                    response = self._call_anthropic(prompt, model)

                # Parse response - extract JSON array
                results = self._parse_llm_response(response)

                for result in results:
                    idx = result.get('index', 0) - 1
                    if 0 <= idx < len(batch) and result.get('approve', False):
                        video = batch[idx]
                        video['llm_reason'] = result.get('reason', 'Approved')
                        video['llm_relevance'] = float(result.get('relevance', 0.7))
                        approved.append(video)
                        logger.debug(f"    ✓ Approved ({video['llm_relevance']:.1f}): {video['title'][:50]}...")
                    elif 0 <= idx < len(batch):
                        reason = result.get('reason', 'No reason')
                        logger.debug(f"    ✗ Rejected: {batch[idx]['title'][:50]}... ({reason})")

            except Exception as e:
                logger.warning(f"LLM title filter error: {e}")
                # On error, approve all in batch with default relevance (fail open)
                for v in batch:
                    v['llm_relevance'] = 0.5
                approved.extend(batch)

        # SORT by relevance score (highest first) before returning
        approved.sort(key=lambda v: v.get('llm_relevance', 0.5), reverse=True)

        logger.info(f"    LLM filter: {len(approved)}/{len(videos)} videos approved (sorted by relevance)")
        return approved

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse JSON array from LLM response, handling markdown and truncation."""
        # Strip markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith('```'):
            lines = clean_response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            clean_response = '\n'.join(lines)

        json_match = re.search(r'\[[\s\S]*\]', clean_response)
        if not json_match:
            logger.warning(f"No JSON array found in LLM response (len={len(response)})")
            raise ValueError("No JSON array in response")

        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix truncated JSON
            json_str = json_str.rstrip()
            if not json_str.endswith(']'):
                last_brace = json_str.rfind('}')
                if last_brace > 0:
                    json_str = json_str[:last_brace + 1] + ']'
                    try:
                        results = json.loads(json_str)
                        logger.debug("Fixed truncated JSON response")
                        return results
                    except json.JSONDecodeError:
                        raise
            raise

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
