"""
Remix Stage - Keyword-Based Video Filtering

Stage 2.5 of the video matching pipeline:
- Filters downloaded videos by keyword relevance before transcription
- Reduces transcription workload by excluding low-relevance videos
- Supports both video files and audio files (audio-first mode)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class RemixStage(Stage):
    """
    Filters downloaded content by keyword relevance.

    Inputs:
        - state.keywords: Keywords for relevance scoring
        - state.downloaded_videos: Videos to filter (OR)
        - state.downloaded_audio: Audio files to filter (audio-first mode)

    Outputs:
        - Updates state.downloaded_videos or state.downloaded_audio in-place (filters out irrelevant files)

    Configuration:
        - config.remix.enabled: Master on/off switch
        - config.remix.trigger_after_download: Must be true to run
        - config.remix.min_relevance_score: Minimum score threshold (default: 0.1)
        - config.remix.max_files_to_include: Maximum files to keep (default: 100)
        - config.remix.fuzzy_match: Enable fuzzy matching
        - config.remix.interactive_curation: Prompt user for confirmation
        - config.remix.auto_accept_filter: "filtered", "all", or "prompt"
    """

    name = "REMIX"
    description = "Filter videos by keyword relevance (zero-download remix)"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the remix stage"""
        warnings = []

        # Check if enabled
        if not getattr(config, 'remix', None) or not config.remix.enabled:
            logger.debug("Remix disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'disabled'
            })

        # Check if trigger_after_download is enabled
        if not getattr(config.remix, 'trigger_after_download', True):
            logger.debug("Remix trigger_after_download disabled")
            return StageResult.ok({
                'skipped': True,
                'reason': 'trigger_disabled'
            })

        # Check if keywords exist
        if not state.keywords:
            logger.debug("No keywords available for remix")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_keywords'
            })

        print(f"\n  --- Stage 2.5: KEYWORD REMIX (Zero-Download) ---")

        try:
            from ..keyword_remix import remix_downloaded_videos, remix_audio_files, RemixConfig

            # Build RemixConfig from config
            remix_config = RemixConfig(
                enabled=config.remix.enabled,
                trigger_after_download=config.remix.trigger_after_download,
                min_relevance_score=getattr(config.remix, 'min_relevance_score', 0.1),
                high_relevance_threshold=getattr(config.remix, 'high_relevance_threshold', 0.5),
                max_files_to_process=getattr(config.remix, 'max_files_to_process', 500),
                max_files_to_include=getattr(config.remix, 'max_files_to_include', 100),
                fuzzy_match=getattr(config.remix, 'fuzzy_match', True),
                case_sensitive=getattr(config.remix, 'case_sensitive', False),
                interactive_curation=getattr(config.remix, 'interactive_curation', False),
                auto_accept_filter=getattr(config.remix, 'auto_accept_filter', 'filtered')
            )

            # Check if audio-first mode
            mode = self._detect_mode(state)

            if mode == 'audio':
                included_paths, result = self._remix_audio(
                    state,
                    remix_config
                )
            elif mode == 'video':
                included_paths, result = self._remix_videos(
                    state,
                    config,
                    remix_config
                )
            else:
                # No files to remix
                return StageResult.ok({
                    'skipped': True,
                    'reason': 'no_files',
                    'mode': mode
                })

            # Update state with filtered files
            if mode == 'audio':
                # Filter audio_downloads
                included_set = set(included_paths)
                original_count = len(state.downloaded_audio)
                state.downloaded_audio = [
                    ad for ad in state.downloaded_audio
                    if ad.file in included_set
                ]
                filtered_count = len(state.downloaded_audio)
                print(f"  + Filtered: {original_count} -> {filtered_count} audio files")

            else:  # video mode
                # Filter downloaded_videos
                included_set = set(included_paths)
                original_count = len(state.downloaded_videos)
                state.downloaded_videos = [
                    dv for dv in state.downloaded_videos
                    if dv.file in included_set
                ]
                filtered_count = len(state.downloaded_videos)
                print(f"  + Filtered: {original_count} -> {filtered_count} videos")

            # Build checkpoint data
            checkpoint_data = {
                'included_count': len(included_paths),
                'excluded_count': result.total_files - len(included_paths),
                'avg_score': result.avg_match_score,
                'remixed_paths': included_paths,
                'mode': mode,
                'processing_time_sec': result.processing_time_seconds
            }

            return StageResult.ok(checkpoint_data, warnings=warnings)

        except ImportError as e:
            error_msg = f"Could not import keyword_remix module: {e}"
            logger.error(error_msg)
            print(f"  ! Remix module not available")
            return StageResult.fail(error_msg)

        except Exception as e:
            logger.error(f"Remix failed: {e}", exc_info=True)
            print(f"  ! Remix failed: {e}")
            return StageResult.fail(str(e))

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if this stage can be skipped (already completed)"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """
        Restore from checkpoint.

        Remix doesn't need explicit restore - the filtered state was already
        saved in the DOWNLOAD checkpoint. We just skip re-running the filter.
        """
        data = checkpoint.get_stage_data(self.name)
        if not data:
            logger.warning(f"No checkpoint data for {self.name}")
            return False

        # Log what was done
        mode = data.get('mode', 'unknown')
        included = data.get('included_count', 0)
        excluded = data.get('excluded_count', 0)

        logger.info(f"Restored {self.name}: {included} files included, {excluded} excluded ({mode} mode)")
        return True

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """
        Validate inputs before running.

        This is an optional stage, so we don't enforce hard requirements.
        """
        # Optional stage - no hard requirements
        return None

    def _detect_mode(
        self,
        state: 'PipelineState'
    ) -> str:
        """
        Detect whether we're in audio-first or video mode.

        Returns:
            'audio', 'video', or 'none'
        """
        if state.downloaded_audio and len(state.downloaded_audio) > 0:
            return 'audio'
        elif state.downloaded_videos and len(state.downloaded_videos) > 0:
            return 'video'
        else:
            return 'none'

    def _remix_audio(
        self,
        state: 'PipelineState',
        remix_config
    ):
        """
        Remix audio files (audio-first mode).

        Returns:
            Tuple of (included_paths, result)
        """
        from ..keyword_remix import remix_audio_files

        # Extract audio file paths
        audio_files = [ad.file for ad in state.downloaded_audio]

        print(f"  Analyzing {len(audio_files)} audio files...")

        # Run remix
        included, result = remix_audio_files(
            audio_files,
            state.keywords,
            config=remix_config,
            interactive=remix_config.interactive_curation,
            show_progress=True
        )

        return included, result

    def _remix_videos(
        self,
        state: 'PipelineState',
        config: 'Config',
        remix_config
    ):
        """
        Remix video files.

        Returns:
            Tuple of (included_paths, result)
        """
        from ..keyword_remix import remix_downloaded_videos

        # Get video directory
        video_dir = Path(config.downloaded_videos_dir)

        print(f"  Analyzing videos in {video_dir}...")

        # Run remix
        included, result = remix_downloaded_videos(
            video_dir,
            state.keywords,
            config=remix_config,
            interactive=remix_config.interactive_curation,
            show_progress=True
        )

        return included, result
