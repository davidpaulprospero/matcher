"""
Caption Stage - Fetch YouTube captions for transcript-first matching.

When enabled, fetches YouTube captions/subtitles before any media download.
Videos with captions skip audio download and Whisper transcription entirely.
Videos without captions are marked for audio fallback.

In caption-first mode, this stage runs AFTER VIDEO_METADATA and BEFORE DOWNLOAD:
  VIDEO_METADATA -> CAPTION -> DOWNLOAD (only uncaptioned) -> TRANSCRIBE (only uncaptioned)

This enables massive bandwidth savings since caption fetch is ~1KB vs
audio download being 5-50MB per video.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, CaptionDownload

logger = logging.getLogger(__name__)


@register_stage
class CaptionStage(Stage):
    """
    Fetches YouTube captions for transcript-first matching.

    Inputs (in order of priority):
        - state.video_candidates: List of VideoCandidate from VIDEO_METADATA stage
        - state.downloaded_audio: List of AudioDownload (audio-first mode fallback)
        - state.downloaded_videos: List of DownloadedVideo (traditional mode fallback)
        - config.download.caption_first: Caption-first configuration

    Outputs:
        - state.caption_downloads: List of CaptionDownload
        - state.transcripts: Dict[video_id, List[segment]] (from captions)
        - state.videos_need_audio: List of video_ids that need Whisper fallback
        - state.video_candidates: Updated with caption info (has_captions, is_auto_caption)
    """

    name = "CAPTION"
    description = "Fetch YouTube captions for videos"

    def __init__(self):
        self.caption_fetcher = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the caption fetching stage"""
        warnings = []

        # Check if caption-first mode is enabled
        caption_config = getattr(config.download, 'caption_first', None)
        # Handle both dict and object config patterns (Rule 6)
        if isinstance(caption_config, dict):
            is_enabled = caption_config.get('enabled', False)
        else:
            is_enabled = getattr(caption_config, 'enabled', False) if caption_config else False
        if not is_enabled:
            logger.info("Caption-first mode not enabled, skipping CAPTION stage")
            return StageResult.ok({'skipped': True, 'reason': 'not_enabled'}, warnings)

        try:
            # Initialize caption fetcher
            from ..downloader.caption_fetcher import CaptionFetcher

            cache_dir = Path(config.cache.cache_dir) / "captions"
            self.caption_fetcher = CaptionFetcher(config, cache_dir)

            # Get video IDs to fetch captions for
            video_ids = self._get_video_ids(state, config)
            if not video_ids:
                logger.warning("No video IDs to fetch captions for")
                return StageResult.ok({'skipped': True, 'reason': 'no_videos'}, warnings)

            print(f"  >> Fetching captions for {len(video_ids)} videos...")
            logger.info(f"Fetching captions for {len(video_ids)} videos")

            # Fetch captions
            results = self._fetch_captions_batch(
                video_ids,
                state,
                config,
                caption_config
            )

            # Report results
            caption_count = len(state.caption_downloads)
            fallback_count = len(state.videos_need_audio)

            print(f"  >> Captions: {caption_count} found, {fallback_count} need audio fallback")
            logger.info(f"Caption results: {caption_count} with captions, {fallback_count} need audio")

            if caption_count == 0 and fallback_count > 0:
                warnings.append(f"No captions found for any videos - all {fallback_count} will use Whisper fallback")

            # Return full checkpoint data in StageResult (pipeline saves this)
            checkpoint_data = {
                'caption_downloads': [
                    {
                        'file': cd.file,
                        'video_id': cd.video_id,
                        'url': cd.url,
                        'title': cd.title,
                        'duration': cd.duration,
                        'keyword': cd.keyword,
                        'language': cd.language,
                        'is_auto_generated': cd.is_auto_generated,
                    }
                    for cd in state.caption_downloads
                ],
                'videos_need_audio': state.videos_need_audio,
                'transcripts_from_captions': list(state.transcripts.keys()),
                'caption_count': caption_count,
                'fallback_count': fallback_count,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Caption stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if caption stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore caption stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            from ..state import CaptionDownload

            # Restore caption downloads
            caption_downloads = data.get('caption_downloads', [])
            state.caption_downloads = [
                CaptionDownload(**cd) if isinstance(cd, dict) else cd
                for cd in caption_downloads
            ]

            # Restore videos needing audio fallback
            state.videos_need_audio = data.get('videos_need_audio', [])

            # Restore transcripts from captions (re-parse caption files)
            if config:
                self._restore_transcripts(state, config)

            logger.info(f"Restored CAPTION: {len(state.caption_downloads)} captions, {len(state.videos_need_audio)} need audio")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore CAPTION: {e}")
            return False

    def _get_video_ids(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> List[str]:
        """
        Get list of video IDs to fetch captions for.

        Priority order:
        1. state.video_candidates (caption-first mode - VIDEO_METADATA stage output)
        2. state.downloaded_audio (audio-first mode)
        3. state.downloaded_videos (traditional full download mode)
        """
        video_ids = []

        # Priority 1: From video candidates (caption-first mode)
        # This is the primary input when VIDEO_METADATA stage has run
        if state.video_candidates:
            for candidate in state.video_candidates:
                if candidate.video_id:
                    video_ids.append(candidate.video_id)
            logger.debug(f"Using {len(video_ids)} video IDs from video_candidates")

        # Priority 2: From audio downloads (audio-first mode fallback)
        if not video_ids:
            for audio in state.downloaded_audio:
                if audio.video_id:
                    video_ids.append(audio.video_id)
            if video_ids:
                logger.debug(f"Using {len(video_ids)} video IDs from downloaded_audio")

        # Priority 3: From downloaded videos (full download mode fallback)
        if not video_ids:
            for video in state.downloaded_videos:
                video_id = self._extract_video_id(video.url or video.file)
                if video_id:
                    video_ids.append(video_id)
            if video_ids:
                logger.debug(f"Using {len(video_ids)} video IDs from downloaded_videos")

        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for vid in video_ids:
            if vid not in seen:
                seen.add(vid)
                unique_ids.append(vid)

        return unique_ids

    def _extract_video_id(self, url_or_path: str) -> Optional[str]:
        """Extract YouTube video ID from URL or filename"""
        import re

        # Try URL patterns
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})(?:\.mp[34]|\.webm|\.mkv)?$',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_path)
            if match:
                return match.group(1)

        return None

    def _fetch_captions_batch(
        self,
        video_ids: List[str],
        state: 'PipelineState',
        config: 'Config',
        caption_config
    ) -> Dict[str, Any]:
        """Fetch captions for a batch of videos"""
        from ..state import CaptionDownload

        languages = getattr(caption_config, 'languages', ["en", "en-US", "en-GB"])
        prefer_manual = getattr(caption_config, 'prefer_manual_captions', True)
        timeout = getattr(caption_config, 'fetch_timeout', 30)

        # Build video info lookup (for title, duration, keyword)
        # Also build video_id -> video_path mapping for transcript keying
        # Priority: video_candidates > downloaded_audio > downloaded_videos
        video_info = {}
        video_id_to_path = {}
        video_candidate_lookup = {}  # To update VideoCandidate objects

        # From video_candidates (caption-first mode primary source)
        for candidate in state.video_candidates:
            video_info[candidate.video_id] = {
                'url': candidate.url,
                'title': candidate.title,
                'duration': candidate.duration,
                'keyword': candidate.keyword,
                'file': '',  # No file downloaded yet in caption-first mode
            }
            video_candidate_lookup[candidate.video_id] = candidate

        # From downloaded_audio (audio-first mode)
        for audio in state.downloaded_audio:
            video_info[audio.video_id] = {
                'url': audio.url,
                'title': audio.title,
                'duration': audio.duration,
                'keyword': audio.keyword,
                'file': audio.file,
            }
            video_id_to_path[audio.video_id] = audio.file

        # From downloaded_videos (traditional mode)
        for video in state.downloaded_videos:
            vid = self._extract_video_id(video.url or video.file)
            if vid:
                video_info[vid] = {
                    'url': video.url,
                    'title': video.title,
                    'duration': video.duration,
                    'keyword': video.keyword,
                    'file': video.file,
                }
                video_id_to_path[vid] = video.file

        caption_downloads = []
        videos_need_audio = []

        for i, video_id in enumerate(video_ids):
            print(f"    [{i+1}/{len(video_ids)}] {video_id}...", end=" ", flush=True)

            result = self.caption_fetcher.fetch_captions(
                video_id,
                languages=languages,
                prefer_manual=prefer_manual,
                timeout=timeout
            )

            if result:
                # Parse caption file into transcript segments
                segments = self.caption_fetcher.parse_caption_file(result.file)

                if segments:
                    # Set transcript source on each segment
                    source = 'manual_caption' if not result.is_auto_generated else 'auto_caption'
                    for seg in segments:
                        seg['transcript_source'] = source

                    # Update cache with segment count
                    duration_covered = segments[-1].get('end', 0.0) if segments else 0.0
                    self.caption_fetcher.update_segment_count(
                        video_id,
                        len(segments),
                        duration_covered
                    )

                    # Store in state.transcripts keyed by video_path (for embedding compatibility)
                    # Fall back to video_id if no path available
                    transcript_key = video_id_to_path.get(video_id, video_id)
                    state.transcripts[transcript_key] = segments

                    # Also store by video_id for easy lookup during filtering
                    if transcript_key != video_id:
                        state.transcripts[video_id] = segments

                    # Create CaptionDownload record
                    info = video_info.get(video_id, {})
                    cd = CaptionDownload(
                        file=result.file,
                        video_id=video_id,
                        url=info.get('url', f"https://youtube.com/watch?v={video_id}"),
                        title=info.get('title', ''),
                        duration=info.get('duration', 0.0),
                        keyword=info.get('keyword', ''),
                        language=result.language,
                        is_auto_generated=result.is_auto_generated,
                    )
                    caption_downloads.append(cd)

                    # Update VideoCandidate if exists (caption-first mode)
                    if video_id in video_candidate_lookup:
                        candidate = video_candidate_lookup[video_id]
                        candidate.has_captions = True
                        candidate.caption_language = result.language
                        candidate.is_auto_caption = result.is_auto_generated
                        candidate.transcript_source = source

                    caption_type = "auto" if result.is_auto_generated else "manual"
                    print(f"{caption_type} ({len(segments)} segments)")
                else:
                    # Caption file exists but couldn't parse
                    print("parse failed -> audio fallback")
                    videos_need_audio.append(video_id)
            else:
                print("no captions -> audio fallback")
                videos_need_audio.append(video_id)

        state.caption_downloads = caption_downloads
        state.videos_need_audio = videos_need_audio

        return {
            'caption_count': len(caption_downloads),
            'fallback_count': len(videos_need_audio),
        }

    def _restore_transcripts(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> None:
        """Re-parse caption files to restore transcripts"""
        from ..downloader.caption_fetcher import CaptionFetcher

        cache_dir = Path(config.cache.cache_dir) / "captions"
        fetcher = CaptionFetcher(config, cache_dir)

        # Build video_id -> video_path mapping for proper keying
        # Also build video_candidate_lookup to restore caption flags
        video_id_to_path = {}
        video_candidate_lookup = {}

        for candidate in state.video_candidates:
            video_candidate_lookup[candidate.video_id] = candidate

        for audio in state.downloaded_audio:
            video_id_to_path[audio.video_id] = audio.file
        for video in state.downloaded_videos:
            vid = self._extract_video_id(video.url or video.file)
            if vid:
                video_id_to_path[vid] = video.file

        for cd in state.caption_downloads:
            if Path(cd.file).exists():
                segments = fetcher.parse_caption_file(cd.file)
                if segments:
                    source = 'manual_caption' if not cd.is_auto_generated else 'auto_caption'
                    for seg in segments:
                        seg['transcript_source'] = source

                    # Key by video_path for embedding compatibility, fall back to video_id
                    transcript_key = video_id_to_path.get(cd.video_id, cd.video_id)
                    state.transcripts[transcript_key] = segments

                    # Also store by video_id for easy lookup
                    if transcript_key != cd.video_id:
                        state.transcripts[cd.video_id] = segments

                    # Restore VideoCandidate caption flags
                    if cd.video_id in video_candidate_lookup:
                        candidate = video_candidate_lookup[cd.video_id]
                        candidate.has_captions = True
                        candidate.caption_language = cd.language
                        candidate.is_auto_caption = cd.is_auto_generated
                        candidate.transcript_source = source
