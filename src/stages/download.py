"""
Download Stage - Video and Audio Download

Stage 2 of the video matching pipeline:
- Downloads videos from YouTube based on keywords
- Supports audio-first mode for efficient matching
- Handles global cache reuse
- Downloads video segments after matching (audio-first)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo, AudioDownload

logger = logging.getLogger(__name__)


@register_stage
class DownloadStage(Stage):
    """
    Downloads video/audio footage based on keywords.

    Inputs:
        - state.keywords: List of keywords to search for
        - state.topic_context: Topic for search refinement

    Outputs:
        - state.downloaded_videos: List of DownloadedVideo
        - state.downloaded_audio: List of AudioDownload (audio-first mode)
        - state.failed_keywords: List of failed keywords
    """

    name = "DOWNLOAD"
    description = "Download video footage from YouTube"

    def __init__(self):
        self.downloader = None
        self.global_cache = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the download stage"""
        warnings = []

        try:
            if config.pipeline.skip_download:
                print("  >> Skipping download (config: skip_download=true)")
                logger.info("Skipping DOWNLOAD stage (config: skip_download=true)")
                self._load_existing_videos(state, config)
                return StageResult.ok({'skipped': True}, warnings)

            # Check if audio-first mode
            if self._is_audio_first_enabled(config):
                return self._run_audio_first(state, config, checkpoint, warnings)
            else:
                return self._run_full_download(state, config, checkpoint, warnings)

        except Exception as e:
            logger.exception(f"Download stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if download stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Restore download stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # Restore downloaded videos
            from ..state import DownloadedVideo
            videos = data.get('downloaded_videos', [])
            state.downloaded_videos = [
                DownloadedVideo(**v) if isinstance(v, dict) else v
                for v in videos
            ]

            # Restore failed keywords
            state.failed_keywords = data.get('failed_keywords', [])

            logger.info(f"Restored DOWNLOAD: {len(state.downloaded_videos)} videos")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        if not state.keywords:
            return "No keywords available for download"
        return None

    # === Main Download Methods ===

    def _run_full_download(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        warnings: List[str]
    ) -> StageResult:
        """Run full video download mode"""
        print(f"\n  --- Stage 2: DOWNLOAD FOOTAGE ---")

        try:
            from ..downloader import VideoDownloader

            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            # Print tier info
            print(f"  Duration tiers:")
            for tier_name in ['short', 'medium', 'long', 'longer']:
                min_s = self.downloader._get_tier_value(tier_name, 'min', 0)
                max_s = self.downloader._get_tier_value(tier_name, 'max', 120)
                per_kw = self.downloader._get_tier_value(tier_name, 'per_keyword', 5)
                print(f"    - {tier_name}: {min_s}-{max_s}s ({per_kw}/kw)")

            # Check global cache first
            keywords_to_download, reusable_videos = self._check_global_cache(
                state.keywords, config
            )

            # Download videos
            if keywords_to_download:
                downloaded_videos, failed = self.downloader.download_all(
                    keywords=keywords_to_download,
                    output_dir=output_dir,
                    resume=True,
                    topic=state.topic_context or ""
                )
            else:
                downloaded_videos = []
                failed = []
                print(f"  >> All keywords covered by global cache")

            # Store results
            self._store_download_results(
                state, downloaded_videos, reusable_videos, failed
            )

            print(f"\n  + Downloaded {len(downloaded_videos)} new videos")
            if reusable_videos:
                print(f"  + Reusing {len(reusable_videos)} videos from global cache")

            # Prepare checkpoint data
            checkpoint_data = {
                'downloaded_videos': [
                    self._video_to_dict(dv) for dv in state.downloaded_videos
                ],
                'failed_keywords': state.failed_keywords,
                'video_count': len(state.downloaded_videos),
            }

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            return StageResult.fail(f"Could not import downloader: {e}", warnings)

    def _run_audio_first(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager',
        warnings: List[str]
    ) -> StageResult:
        """Run audio-first download mode"""
        print(f"\n  --- Stage 2A: DOWNLOAD AUDIO (Audio-First Mode) ---")

        audio_config = getattr(config.download, 'audio_first', None)
        print(f"  Audio-first mode: Downloading audio for transcription")
        print(f"    Buffer: {getattr(audio_config, 'buffer_seconds', 30)}s")
        print(f"    Merge gap: {getattr(audio_config, 'merge_gap_seconds', 15)}s")

        try:
            from ..downloader import VideoDownloader, AudioDownload

            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            all_audio_downloads = []
            tiers = list(self.downloader.DURATION_TIERS.keys())
            total_keywords = len(state.keywords)
            failed_keywords = []

            for idx, keyword in enumerate(state.keywords, 1):
                print(f"\n  [{idx}/{total_keywords}] {keyword}")

                try:
                    for tier in tiers:
                        per_kw = self.downloader._get_tier_value(tier, 'per_keyword', 5)
                        if per_kw <= 0:
                            continue

                        audio_downloads = self.downloader.download_audio_for_keyword(
                            keyword=keyword,
                            output_dir=output_dir,
                            tier=tier,
                            topic=state.topic_context or ""
                        )
                        all_audio_downloads.extend(audio_downloads)
                except Exception as e:
                    logger.error(f"Audio download failed for '{keyword}': {e}")
                    failed_keywords.append(keyword)

            # Store results
            state.downloaded_audio = all_audio_downloads
            state.failed_keywords = failed_keywords

            if failed_keywords:
                print(f"\n  ! {len(failed_keywords)} keywords failed")
            print(f"\n  + Downloaded {len(all_audio_downloads)} audio files")

            checkpoint_data = {
                'audio_downloads': [
                    self._audio_to_dict(ad) for ad in all_audio_downloads
                ],
                'failed_keywords': failed_keywords,
                'audio_count': len(all_audio_downloads),
                'mode': 'audio_first',
            }

            return StageResult.ok(checkpoint_data, warnings)

        except ImportError as e:
            return StageResult.fail(f"Could not import downloader: {e}", warnings)

    # === Helper Methods ===

    def _is_audio_first_enabled(self, config: 'Config') -> bool:
        """Check if audio-first mode is enabled"""
        audio_config = getattr(config.download, 'audio_first', None)
        return audio_config and getattr(audio_config, 'enabled', False)

    def _check_global_cache(
        self,
        keywords: List[str],
        config: 'Config'
    ) -> tuple:
        """Check global cache for reusable videos"""
        # TODO: Implement global cache checking
        # For now, return all keywords for download
        return keywords, []

    def _store_download_results(
        self,
        state: 'PipelineState',
        downloaded_videos: List[Any],
        reusable_videos: List[Dict],
        failed: List[str]
    ):
        """Store download results in state"""
        from ..state import DownloadedVideo

        state.downloaded_videos = []

        # Add newly downloaded videos
        for dv in downloaded_videos:
            if isinstance(dv, dict):
                state.downloaded_videos.append(DownloadedVideo(
                    file=dv.get('file', dv.get('path', '')),
                    url=dv.get('url', ''),
                    title=dv.get('title', ''),
                    channel=dv.get('channel', ''),
                    upload_date=dv.get('upload_date', ''),
                    duration=dv.get('duration', 0.0),
                    duration_tier=dv.get('duration_tier', dv.get('tier', '')),
                    keyword=dv.get('keyword', ''),
                    download_date=dv.get('download_date', ''),
                    source='download',
                ))
            else:
                state.downloaded_videos.append(dv)

        # Add reusable videos from global cache
        for gv in reusable_videos:
            state.downloaded_videos.append(DownloadedVideo(
                file=gv.get('path', ''),
                source='global_cache',
                face_score=gv.get('face_score', 0.5),
            ))

        state.failed_keywords = failed

    def _load_existing_videos(
        self,
        state: 'PipelineState',
        config: 'Config'
    ):
        """Load existing videos when skipping download"""
        from ..state import DownloadedVideo

        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
        videos_dir = Path(config.downloaded_videos_dir)

        if not videos_dir.exists():
            logger.warning(f"Videos directory not found: {videos_dir}")
            return

        video_files = []

        # Direct files
        for f in videos_dir.iterdir():
            if f.is_file() and f.suffix.lower() in video_extensions:
                video_files.append(f)

        # One level deep in subfolders
        if not video_files:
            for subdir in videos_dir.iterdir():
                if subdir.is_dir():
                    for f in subdir.iterdir():
                        if f.is_file() and f.suffix.lower() in video_extensions:
                            video_files.append(f)

        state.downloaded_videos = [
            DownloadedVideo(
                file=str(vf),
                source='existing',
            )
            for vf in video_files
        ]

        print(f"  + Loaded {len(state.downloaded_videos)} existing videos")

    def _video_to_dict(self, video: 'DownloadedVideo') -> Dict[str, Any]:
        """Convert DownloadedVideo to dict for checkpointing"""
        if hasattr(video, '__dict__'):
            return {k: v for k, v in video.__dict__.items()}
        return dict(video) if isinstance(video, dict) else {}

    def _audio_to_dict(self, audio: 'AudioDownload') -> Dict[str, Any]:
        """Convert AudioDownload to dict for checkpointing"""
        if hasattr(audio, '__dict__'):
            return {k: v for k, v in audio.__dict__.items()}
        return dict(audio) if isinstance(audio, dict) else {}


@register_stage
class DownloadVideoSegmentsStage(Stage):
    """
    Downloads video segments after matching (audio-first mode).

    This is Stage 2B in audio-first mode, run after matching to
    download only the portions of videos actually used.

    Inputs:
        - state.matches: List of Match objects
        - state.downloaded_audio: List of AudioDownload

    Outputs:
        - Updated state.downloaded_videos with segment files
    """

    name = "DOWNLOAD_SEGMENTS"
    description = "Download matched video segments (audio-first)"

    def __init__(self):
        self.downloader = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video segment download stage"""
        warnings = []

        try:
            if not state.matches:
                return StageResult.fail("No matches - run matching first", warnings)

            if not state.downloaded_audio:
                return StageResult.fail("No audio downloads - run audio download first", warnings)

            print(f"\n  --- Stage 2B: DOWNLOAD VIDEO SEGMENTS ---")

            from ..downloader import (
                VideoDownloader,
                collect_matched_segments,
                prepare_merged_segments,
            )

            audio_config = getattr(config.download, 'audio_first', None)
            buffer_seconds = getattr(audio_config, 'buffer_seconds', 30.0)
            merge_gap = getattr(audio_config, 'merge_gap_seconds', 15.0)

            print(f"  Downloading matched video segments")
            print(f"    Buffer: {buffer_seconds}s, Merge gap: {merge_gap}s")

            # Build audio downloads by ID
            audio_downloads_by_id = {
                a.video_id if hasattr(a, 'video_id') else a.get('video_id', ''): a
                for a in state.downloaded_audio
            }

            # Collect matched segments
            segments_by_video = collect_matched_segments(
                state.matches,
                audio_downloads_by_id
            )

            total_matches = sum(len(segs) for segs in segments_by_video.values())
            print(f"    Matched segments: {total_matches} across {len(segments_by_video)} videos")

            # Merge segments with buffer
            merged_segments = prepare_merged_segments(
                segments_by_video,
                audio_downloads_by_id,
                buffer_seconds=buffer_seconds,
                merge_gap_seconds=merge_gap
            )

            print(f"    After merge: {len(merged_segments)} segments to download")

            # Download segments
            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            downloaded_segments = self.downloader.download_video_segments(
                merged_segments,
                output_dir
            )

            print(f"\n  + Downloaded {len(downloaded_segments)} video segments")

            checkpoint_data = {
                'segment_count': len(downloaded_segments),
                'total_matches': total_matches,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Video segment download failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Restore from checkpoint"""
        # Segment files are on disk, no state to restore
        return True

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs"""
        if not state.matches:
            return "No matches available for segment download"
        if not state.downloaded_audio:
            return "No audio downloads available"
        return None
