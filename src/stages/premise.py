"""
Premise Stage - Extract Video Topic/Theme Summaries

Stage that runs after TRANSCRIBE to extract a brief premise (topic + context)
for each video. This premise is used for theme-based matching instead of
literal word-for-word transcript matching.

Example premises:
- "Documentary examining Denny's restaurant closures amid economic challenges"
- "Music lyric video for a love song called '24 Hours'"
- "B-roll footage of empty restaurant interiors at night"
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)

# Maximum transcript length to send to LLM (chars)
MAX_TRANSCRIPT_LENGTH = 2000

PREMISE_EXTRACTION_PROMPT = """Analyze this video and provide a brief premise summary.

Title: {title}
Description: {description}
Transcript excerpt: {transcript_excerpt}

Respond with a single sentence describing:
1. What type of content this is (documentary, music video, tutorial, news clip, B-roll footage, stock footage, etc.)
2. The main topic or subject matter

Format your response as: "[Content type] about/of/for [topic/subject]"

Examples:
- "Documentary about fast food restaurant closures in America"
- "Music lyric video for a love song"
- "B-roll footage of urban cityscapes at night"
- "News report covering inflation impact on food prices"
- "Stock footage of people eating at outdoor cafe"
- "Tutorial on restaurant kitchen operations"

Premise:"""


@register_stage
class PremiseStage(Stage):
    """
    Extract topic + context premise for each video.

    Inputs:
        - state.downloaded_videos: List of DownloadedVideo
        - state.transcripts: Dict[video_path, List[segments]]
        - state.broll_downloads: B-roll video metadata

    Outputs:
        - state.video_premises: Dict[video_id, premise_string]
        - Updates DownloadedVideo.premise field

    The premise is a brief LLM-generated summary describing what type
    of content the video is and its main topic. This enables theme-based
    matching instead of literal transcript matching.
    """

    name = "PREMISE"
    description = "Extract video topic/theme premises"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the premise extraction stage."""
        warnings = []

        try:
            # Check if premise scoring is enabled
            premise_config = getattr(config.matching, 'premise_scoring', None)
            if premise_config and not getattr(premise_config, 'enabled', True):
                logger.info("Premise scoring disabled in config, skipping stage")
                return StageResult.ok({'skipped': True, 'reason': 'disabled'})

            print(f"\n  --- Stage: PREMISE ---")

            # Initialize premises dict in state if not exists
            if not hasattr(state, 'video_premises') or state.video_premises is None:
                state.video_premises = {}

            # Get cache directory
            cache_dir = Path(state.project_dir) / ".cache" / "premises"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Collect all videos to process
            videos_to_process = self._collect_videos(state)
            print(f"  Found {len(videos_to_process)} videos to extract premises for")

            if not videos_to_process:
                return StageResult.ok({'premise_count': 0}, warnings)

            # Process each video
            extracted_count = 0
            cached_count = 0

            for video_info in videos_to_process:
                video_id = video_info['video_id']
                video_path = video_info.get('path', '')

                # Check cache first
                cached_premise = self._load_from_cache(cache_dir, video_id)
                if cached_premise:
                    state.video_premises[video_id] = cached_premise
                    cached_count += 1
                    continue

                # Get transcript for this video
                transcript = self._get_transcript(state, video_path, video_id)

                # Extract premise via LLM
                premise = self._extract_premise(
                    title=video_info.get('title', ''),
                    description=video_info.get('description', ''),
                    transcript=transcript,
                    config=config
                )

                if premise:
                    state.video_premises[video_id] = premise
                    self._save_to_cache(cache_dir, video_id, premise, video_info)
                    extracted_count += 1
                else:
                    # Fallback: use title or generic description
                    fallback = self._generate_fallback_premise(video_info)
                    state.video_premises[video_id] = fallback
                    warnings.append(f"Used fallback premise for {video_id}")

            print(f"  + Extracted {extracted_count} new premises")
            print(f"  + Loaded {cached_count} premises from cache")
            print(f"  Total: {len(state.video_premises)} video premises")

            # Log sample premises
            if state.video_premises:
                sample_items = list(state.video_premises.items())[:3]
                print(f"  Sample premises:")
                for vid, premise in sample_items:
                    print(f"    {vid[:15]}...: {premise[:60]}...")

            return StageResult.ok({
                'premise_count': len(state.video_premises),
                'extracted': extracted_count,
                'cached': cached_count
            }, warnings)

        except Exception as e:
            logger.exception(f"Premise stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if premise stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore premises from cache."""
        try:
            cache_dir = Path(state.project_dir) / ".cache" / "premises"
            if not cache_dir.exists():
                return False

            # Initialize premises dict
            if not hasattr(state, 'video_premises') or state.video_premises is None:
                state.video_premises = {}

            # Load all cached premises
            loaded = 0
            for cache_file in cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    video_id = data.get('video_id', cache_file.stem)
                    premise = data.get('premise', '')
                    if premise:
                        state.video_premises[video_id] = premise
                        loaded += 1
                except Exception as e:
                    logger.warning(f"Could not load premise cache {cache_file}: {e}")

            print(f"  Restored PREMISE: {loaded} video premises from cache")
            logger.info(f"Restored {loaded} premises from cache")
            return loaded > 0

        except Exception as e:
            logger.warning(f"Could not restore premises: {e}")
            return False

    def _collect_videos(self, state: 'PipelineState') -> List[Dict[str, Any]]:
        """Collect all videos that need premise extraction."""
        videos = []

        # Regular downloaded videos
        for dv in state.downloaded_videos:
            video_id = self._extract_video_id(dv.file)
            videos.append({
                'video_id': video_id,
                'path': dv.file,
                'title': dv.title or '',
                'description': '',  # Could add if available
                'source': 'download'
            })

        # B-roll downloads
        for broll in state.broll_downloads:
            if isinstance(broll, dict):
                video_id = broll.get('video_id', self._extract_video_id(broll.get('file', '')))
                videos.append({
                    'video_id': video_id,
                    'path': broll.get('file', ''),
                    'title': broll.get('title', ''),
                    'description': broll.get('description', ''),
                    'source': 'broll'
                })

        # Stock videos from stock/ folder
        stock_dir = Path(state.project_dir) / "stock"
        if stock_dir.exists():
            for video_file in stock_dir.glob("*.mp4"):
                video_id = self._extract_video_id(str(video_file))
                # Try to load metadata from .meta.json
                meta_file = video_file.with_suffix('.meta.json')
                title = ''
                description = ''
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        title = meta.get('title', meta.get('description', ''))
                        description = meta.get('description', '')
                    except Exception:
                        pass
                videos.append({
                    'video_id': video_id,
                    'path': str(video_file),
                    'title': title,
                    'description': description,
                    'source': 'stock'
                })
            logger.info(f"Collected {sum(1 for v in videos if v['source'] == 'stock')} stock videos for premise extraction")

        # Deduplicate by video_id
        seen = set()
        unique_videos = []
        for v in videos:
            if v['video_id'] not in seen:
                seen.add(v['video_id'])
                unique_videos.append(v)

        return unique_videos

    def _extract_video_id(self, file_path: str) -> str:
        """Extract video ID from file path."""
        import re
        filename = Path(file_path).stem

        # Audio-first segment: {video_id}_{offset:04d}
        match = re.match(r'^([a-zA-Z0-9_-]{11})_(\d{4})$', filename)
        if match:
            return match.group(1)

        # Regular YouTube: {title}_{video_id}
        match = re.search(r'_([a-zA-Z0-9_-]{11})$', filename)
        if match:
            return match.group(1)

        # Stock footage - use filename hash
        if filename.startswith(('pexels_', 'pixabay_')):
            return filename

        # Fallback: hash the filename
        return hashlib.md5(filename.encode()).hexdigest()[:11]

    def _get_transcript(
        self,
        state: 'PipelineState',
        video_path: str,
        video_id: str
    ) -> str:
        """Get transcript text for a video."""
        # Try by video path
        if video_path and video_path in state.transcripts:
            segments = state.transcripts[video_path]
            return self._segments_to_text(segments)

        # Try by video ID patterns
        for path, segments in state.transcripts.items():
            if video_id in path:
                return self._segments_to_text(segments)

        return ""

    def _segments_to_text(self, segments: List[Dict[str, Any]]) -> str:
        """Convert transcript segments to plain text."""
        texts = []
        for seg in segments:
            if isinstance(seg, dict):
                texts.append(seg.get('text', ''))
            elif hasattr(seg, 'text'):
                texts.append(seg.text)
        return " ".join(texts)

    def _load_from_cache(self, cache_dir: Path, video_id: str) -> Optional[str]:
        """Load premise from cache if exists."""
        cache_file = cache_dir / f"{video_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('premise')
            except Exception as e:
                logger.debug(f"Could not load premise cache for {video_id}: {e}")
        return None

    def _save_to_cache(
        self,
        cache_dir: Path,
        video_id: str,
        premise: str,
        video_info: Dict[str, Any]
    ):
        """Save premise to cache."""
        cache_file = cache_dir / f"{video_id}.json"
        try:
            import datetime
            data = {
                'video_id': video_id,
                'premise': premise,
                'extracted_at': datetime.datetime.now().isoformat(),
                'source': {
                    'title': video_info.get('title', ''),
                    'has_transcript': bool(video_info.get('has_transcript', True)),
                }
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save premise cache for {video_id}: {e}")

    def _extract_premise(
        self,
        title: str,
        description: str,
        transcript: str,
        config: 'Config'
    ) -> Optional[str]:
        """Extract premise using LLM."""
        try:
            from ..llm_client import create_client, LLMRequest

            # Get LLM settings from config
            premise_config = getattr(config.matching, 'premise_scoring', None)
            provider = 'gemini'
            model = 'gemini-2.0-flash'

            if premise_config:
                provider = getattr(premise_config, 'provider', 'gemini')
                model = getattr(premise_config, 'model', 'gemini-2.0-flash')

            # Truncate transcript if too long
            transcript_excerpt = transcript[:MAX_TRANSCRIPT_LENGTH]
            if len(transcript) > MAX_TRANSCRIPT_LENGTH:
                transcript_excerpt += "..."

            # Build prompt
            prompt = PREMISE_EXTRACTION_PROMPT.format(
                title=title or "(No title)",
                description=description or "(No description)",
                transcript_excerpt=transcript_excerpt or "(No transcript)"
            )

            # Call LLM
            client = create_client(provider=provider, model=model)
            request = LLMRequest(
                prompt=prompt,
                max_tokens=150,
                temperature=0.3
            )
            response = client.generate(request)

            # Parse response - LLMResponse has .text property
            premise = response.text.strip() if response.text else ""

            # Clean up common artifacts
            if premise.startswith('"') and premise.endswith('"'):
                premise = premise[1:-1]
            if premise.startswith("Premise:"):
                premise = premise[8:].strip()

            return premise if premise else None

        except Exception as e:
            logger.warning(f"LLM premise extraction failed: {e}")
            return None

    def _generate_fallback_premise(self, video_info: Dict[str, Any]) -> str:
        """Generate fallback premise from metadata when LLM fails."""
        title = video_info.get('title', '')
        source = video_info.get('source', '')

        if 'pexels' in video_info.get('video_id', '').lower():
            return f"Stock footage from Pexels"
        if 'pixabay' in video_info.get('video_id', '').lower():
            return f"Stock footage from Pixabay"
        if title:
            return f"Video titled '{title[:50]}'"
        return "Video content (premise unavailable)"
