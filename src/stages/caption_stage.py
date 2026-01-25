"""
Caption Stage - Fetch YouTube Captions Before Transcription

Stage that runs AFTER DOWNLOAD and BEFORE TRANSCRIBE:
- Fetches YouTube captions for all video candidates
- Stores caption data in state.text_metadata for matching
- Supports checkpoint/resume from partial completion
- Falls back to transcription if captions unavailable

When caption-first mode is enabled (config.download.caption_first.enabled),
this stage fetches captions BEFORE video download, enabling faster matching
with lower bandwidth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class CaptionStage(Stage):
    """
    Fetches YouTube captions for video candidates.

    Inputs:
        - state.downloaded_videos: List of DownloadedVideo (for video IDs)
        - state.downloaded_audio: List of AudioDownload (for video IDs in audio-first mode)

    Outputs:
        - state.text_metadata: Updated with caption data for matching
        - state.caption_results: Dict mapping video_id to caption data (for checkpoint)

    When caption-first mode is enabled, this stage fetches captions from YouTube
    before the TRANSCRIBE stage, enabling matching without downloading video content.
    """

    name = "CAPTION"
    description = "Fetch YouTube captions for video candidates"

    def __init__(self):
        self._fetcher = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the caption fetch stage."""
        warnings = []

        try:
            # Check if caption-first mode is enabled
            caption_config = getattr(config.download, 'caption_first', None)
            if not caption_config or not getattr(caption_config, 'enabled', False):
                print("  >> Skipping caption fetch (caption-first mode disabled)")
                logger.info("Skipping CAPTION stage (caption_first.enabled=false)")
                return StageResult.ok({'skipped': True, 'reason': 'disabled'}, warnings)

            print(f"\n  --- Stage: CAPTION (Fetch YouTube Captions) ---")

            # Get video IDs to fetch captions for
            video_ids = self._get_video_ids(state, config)

            if not video_ids:
                print("  ! No video IDs found for caption fetch")
                warnings.append("No video IDs available for caption fetch")
                return StageResult.ok({'skipped': True, 'reason': 'no_videos'}, warnings)

            print(f"  Found {len(video_ids)} video candidates")

            # Get preferred language from config
            preferred_lang = getattr(caption_config, 'preferred_language', 'en')
            prefer_manual = getattr(caption_config, 'prefer_human_captions', True)
            timeout = getattr(caption_config, 'timeout', 30)

            print(f"  Caption settings: language={preferred_lang}, prefer_manual={prefer_manual}")

            # Initialize caption fetcher
            from ..caption_fetcher import (
                CaptionFetcher,
                CaptionUnavailableError,
                CaptionFetchError,
                CaptionResult,
            )

            self._fetcher = CaptionFetcher(config=config)
            self._fetcher._timeout = timeout

            # Check for already-fetched captions in checkpoint
            existing_captions = self._load_existing_captions(checkpoint)
            print(f"  Found {len(existing_captions)} captions in checkpoint")

            # Fetch captions for each video
            caption_results = {}
            success_count = 0
            skip_count = 0
            fail_count = 0

            for idx, video_id in enumerate(video_ids, 1):
                # Skip if already fetched
                if video_id in existing_captions:
                    caption_results[video_id] = existing_captions[video_id]
                    skip_count += 1
                    continue

                print(f"  [{idx}/{len(video_ids)}] Fetching captions for {video_id}...", end=' ')

                try:
                    result = self._fetcher.fetch_captions_auto_language(
                        video_id,
                        preferred_language=preferred_lang
                    )

                    caption_results[video_id] = {
                        'video_id': video_id,
                        'segments': [seg.to_dict() for seg in result.segments],
                        'language': result.language,
                        'is_auto_generated': result.is_auto_generated,
                        'format_source': result.format_source,
                        'segment_count': len(result.segments),
                    }
                    success_count += 1
                    print(f"✓ {len(result.segments)} segments ({result.language})")

                except CaptionUnavailableError as e:
                    caption_results[video_id] = {
                        'video_id': video_id,
                        'unavailable': True,
                        'reason': str(e.reason),
                    }
                    fail_count += 1
                    print(f"✗ No captions available")
                    logger.debug(f"Captions unavailable for {video_id}: {e}")

                except CaptionFetchError as e:
                    caption_results[video_id] = {
                        'video_id': video_id,
                        'error': True,
                        'reason': str(e.reason),
                    }
                    fail_count += 1
                    print(f"✗ Fetch error")
                    logger.warning(f"Caption fetch error for {video_id}: {e}")

                # Periodic checkpoint save
                if idx % 10 == 0:
                    self._save_intermediate_checkpoint(checkpoint, caption_results)

            # Store caption data in state.text_metadata for matching
            self._populate_text_metadata(state, caption_results)

            # Summary
            print(f"\n  + Caption fetch complete:")
            print(f"    - Success: {success_count} videos")
            print(f"    - Skipped (cached): {skip_count} videos")
            print(f"    - Unavailable/Error: {fail_count} videos")

            if fail_count > 0 and getattr(caption_config, 'fallback_to_transcription', True):
                print(f"    - {fail_count} videos will use Whisper transcription fallback")
                warnings.append(f"{fail_count} videos require transcription fallback")

            # Prepare checkpoint data
            checkpoint_data = {
                'caption_results': caption_results,
                'success_count': success_count,
                'skip_count': skip_count,
                'fail_count': fail_count,
                'total_segments': sum(
                    r.get('segment_count', 0) for r in caption_results.values()
                ),
            }

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            logger.error(f"Could not import caption_fetcher: {e}")
            return StageResult.fail(f"Caption fetcher not available: {e}", warnings)
        except Exception as e:
            logger.exception(f"Caption stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if caption stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore caption stage from checkpoint."""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # Restore caption results to state
            caption_results = data.get('caption_results', {})
            if caption_results:
                self._populate_text_metadata(state, caption_results)
                logger.info(f"Restored CAPTION: {len(caption_results)} videos, "
                           f"{data.get('total_segments', 0)} segments")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to restore CAPTION: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running."""
        # Need either downloaded videos or audio files (which have video IDs)
        has_videos = len(state.downloaded_videos) > 0
        has_audio = len(state.downloaded_audio) > 0

        if not has_videos and not has_audio:
            return "No video candidates available for caption fetch"

        return None

    # === Helper Methods ===

    def _get_video_ids(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> List[str]:
        """Extract video IDs from downloaded videos/audio.

        Returns unique video IDs from state.downloaded_videos or
        state.downloaded_audio (in audio-first mode).
        """
        video_ids = set()

        # Check audio-first mode downloads
        for audio in state.downloaded_audio:
            video_id = getattr(audio, 'video_id', None)
            if video_id and len(video_id) == 11:
                video_ids.add(video_id)

        # Check downloaded videos
        for video in state.downloaded_videos:
            # Try to extract video ID from file path or direct attribute
            video_id = self._extract_video_id(video)
            if video_id and len(video_id) == 11:
                video_ids.add(video_id)

        return list(video_ids)

    def _extract_video_id(self, video) -> Optional[str]:
        """Extract YouTube video ID from a DownloadedVideo.

        Checks attributes and filename patterns to find 11-char video ID.
        """
        import re

        # Check direct video_id attribute (if it has a truthy value)
        video_id = getattr(video, 'video_id', None)
        if video_id:
            return video_id

        # Check URL for video ID
        url = getattr(video, 'url', None)
        if url:
            match = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
            if match:
                return match.group(1)

        # Check filename (videos often named with video ID)
        file_path = getattr(video, 'file', None)
        if file_path:
            filename = Path(file_path).stem
            # YouTube IDs are exactly 11 alphanumeric chars with _-
            match = re.search(r'([A-Za-z0-9_-]{11})', filename)
            if match:
                return match.group(1)

        return None

    def _load_existing_captions(
        self,
        checkpoint: 'CheckpointManager'
    ) -> Dict[str, Dict[str, Any]]:
        """Load already-fetched captions from checkpoint.

        Used for resume functionality - skips videos that already have captions.
        """
        try:
            data = checkpoint.get_stage_data(self.name)
            if data:
                return data.get('caption_results', {})
        except Exception as e:
            logger.debug(f"Could not load existing captions: {e}")

        return {}

    def _save_intermediate_checkpoint(
        self,
        checkpoint: 'CheckpointManager',
        caption_results: Dict[str, Dict[str, Any]]
    ):
        """Save intermediate checkpoint during long fetch operations."""
        try:
            checkpoint_data = {
                'caption_results': caption_results,
                'success_count': sum(1 for r in caption_results.values()
                                    if not r.get('unavailable') and not r.get('error')),
                'partial': True,
            }
            checkpoint.save_intermediate(self.name, checkpoint_data)
        except Exception as e:
            logger.debug(f"Intermediate checkpoint save failed: {e}")

    def _populate_text_metadata(
        self,
        state: 'PipelineState',
        caption_results: Dict[str, Dict[str, Any]]
    ):
        """Populate state.text_metadata from caption results.

        Converts caption segments to the format expected by the matching stage.
        Compatible with TranscriptSegment format used by TRANSCRIBE stage.
        """
        text_metadata = []

        for video_id, result in caption_results.items():
            # Skip unavailable/errored captions
            if result.get('unavailable') or result.get('error'):
                continue

            segments = result.get('segments', [])
            language = result.get('language', 'en')
            is_auto = result.get('is_auto_generated', False)

            for seg in segments:
                text_metadata.append({
                    'text': seg.get('text', ''),
                    'video_path': video_id,  # In caption-first mode, this is video ID
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'source_file': video_id,
                    # Caption-specific metadata
                    'caption_source': 'youtube',
                    'caption_language': language,
                    'caption_auto_generated': is_auto,
                })

        # Extend existing text_metadata (don't replace, as TRANSCRIBE may add more)
        state.text_metadata.extend(text_metadata)

        logger.info(f"Populated text_metadata with {len(text_metadata)} caption segments")
