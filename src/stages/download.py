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
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore download stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                # Fallback: try to restore from disk if config available
                if config and self._restore_from_disk(state, config):
                    logger.info("Restored DOWNLOAD from disk (no checkpoint data)")
                    return True
                return False

            # Restore downloaded videos
            from ..state import DownloadedVideo, AudioDownload
            videos = data.get('downloaded_videos', [])
            state.downloaded_videos = [
                DownloadedVideo(**v) if isinstance(v, dict) else v
                for v in videos
            ]

            # Restore downloaded audio (audio-first mode)
            audio_downloads = data.get('audio_downloads', [])
            restored_audio = []
            for a in audio_downloads:
                if isinstance(a, dict):
                    # Handle backward compatibility: old checkpoints used different field names
                    a_copy = dict(a)  # Create copy to avoid modifying checkpoint data

                    # Map 'audio_file' → 'file'
                    if 'audio_file' in a_copy and 'file' not in a_copy:
                        a_copy['file'] = a_copy.pop('audio_file')

                    # Map 'video_url' → 'url'
                    if 'video_url' in a_copy and 'url' not in a_copy:
                        a_copy['url'] = a_copy.pop('video_url')

                    # Remove obsolete fields that are no longer in AudioDownload
                    # (channel, duration_tier, upload_date, license were removed in refactoring)
                    for obsolete_field in ['channel', 'duration_tier', 'upload_date', 'license']:
                        a_copy.pop(obsolete_field, None)

                    restored_audio.append(AudioDownload(**a_copy))
                else:
                    restored_audio.append(a)
            state.downloaded_audio = restored_audio

            # Restore failed keywords
            state.failed_keywords = data.get('failed_keywords', [])

            logger.info(f"Restored DOWNLOAD: {len(state.downloaded_videos)} videos, {len(state.downloaded_audio)} audio")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD: {e}")
            return False

    def _restore_from_disk(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> bool:
        """Fallback: restore download state by scanning disk for existing files"""
        from ..state import AudioDownload
        from pathlib import Path

        try:
            # Get videos root directory from config
            root_dir = getattr(config.download, 'root_dir', None)
            if not root_dir:
                return False

            # Find project subdirectory (must include project name)
            root_path = Path(root_dir)
            project_name = getattr(config.project, 'name', '') or ''
            if project_name:
                project_path = root_path / project_name[:15]
            else:
                project_path = root_path
            
            if not project_path.exists():
                return False

            # Look for audio directories (audio-first mode)
            audio_files = []
            for audio_dir in project_path.rglob('*_audio'):
                for mp3 in audio_dir.glob('*.mp3'):
                    audio_files.append(AudioDownload(
                        file=str(mp3),
                        url="",  # Unknown from disk
                        video_id=mp3.stem,
                        title=mp3.stem,
                        duration=0.0,
                        keyword=audio_dir.name.replace('_audio', '').replace('_l', '').replace('_m', '')
                    ))

            if audio_files:
                state.downloaded_audio = audio_files
                logger.info(f"Restored DOWNLOAD from disk: {len(audio_files)} audio files")
                return True

            # Look for video files
            video_files = list(project_path.rglob('*.mp4')) + list(root_path.rglob('*.webm'))
            if video_files:
                from ..state import DownloadedVideo
                state.downloaded_videos = [
                    DownloadedVideo(
                        file=str(v),
                        duration_tier='m',
                        keyword=v.parent.name.replace('_segments', '').replace('_l', '').replace('_m', ''),
                        title=v.stem,
                        video_id=v.stem
                    )
                    for v in video_files
                ]
                logger.info(f"Restored DOWNLOAD from disk: {len(state.downloaded_videos)} videos")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD from disk: {e}")
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
            from ..downloader import VideoDownloader
            from ..state import AudioDownload

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

                        audio_downloads = self.downloader.audio_first.download_audio_for_keyword(
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
            # Check if we have audio downloads (audio-first mode indicator)
            has_audio = state.downloaded_audio and len(state.downloaded_audio) > 0

            # Edge case: skip_download=true BUT we have audio files
            # This means user interrupted audio-first pipeline before video download
            if config.pipeline.skip_download and has_audio:
                warning_msg = (
                    "WARNING: skip_download=true but audio files exist. "
                    "This indicates incomplete audio-first pipeline. "
                    "Video segments have NOT been downloaded yet. "
                    "OTIO will reference audio files (.mp3) instead of video files. "
                    "To fix: Set skip_download=false and re-run to download matched video segments."
                )
                print(f"\n  ! WARNING: {warning_msg}")
                logger.warning(warning_msg)
                warnings.append("Incomplete audio-first pipeline - no video segments")
                return StageResult.ok({'skipped': True, 'reason': 'skip_download_with_audio'}, warnings)

            # Normal skip: not in audio-first mode
            if config.pipeline.skip_download or not has_audio:
                print("  >> Skipping segment download (not audio-first mode)")
                logger.info("Skipping DOWNLOAD_SEGMENTS (not audio-first mode)")
                return StageResult.ok({'skipped': True}, warnings)

            if not state.matches:
                return StageResult.fail("No matches - run matching first", warnings)

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

            downloaded_segments = self.downloader.audio_first.download_video_segments(
                merged_segments,
                output_dir
            )

            print(f"\n  + Downloaded {len(downloaded_segments)} video segments")

            # Update Match objects to reference downloaded video segments (.mp4) instead of audio files (.mp3)
            self._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

            checkpoint_data = {
                'segment_count': len(downloaded_segments),
                'total_matches': total_matches,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Video segment download failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _remap_matches_to_video_segments(
        self,
        state: 'PipelineState',
        downloaded_segments: List,
        audio_downloads_by_id: dict
    ) -> None:
        """
        Update Match objects to reference downloaded video segment files instead of audio files.

        In audio-first mode, matches initially reference .mp3 audio files. After downloading
        video segments, we need to remap them to the actual .mp4 segment files.

        Args:
            state: Pipeline state with matches to update
            downloaded_segments: List of DownloadedSegment objects
            audio_downloads_by_id: Dict mapping video_id to AudioDownload
        """
        from ..downloader.types import DownloadedSegment

        # Build mapping: (video_id, original_time) -> segment_file
        # Each DownloadedSegment contains multiple matches within its time range
        segment_map = {}
        for seg in downloaded_segments:
            video_id = seg.video_id
            for match in seg.matches:
                # Key by video_id and the original video time
                key = (video_id, match.start_time)
                segment_map[key] = (seg.file, seg.original_start)

        # Helper function to remap a single Match object
        def remap_match(match_obj):
            """Remap a single Match object to video segment file"""
            # Handle two different Match structures:
            # 1. state.Match: has video_file, video_start fields
            # 2. utils.Match: has video_segment.source_file field

            if hasattr(match_obj, 'video_file'):
                # state.Match structure
                if not match_obj.video_file:
                    # Skip matches with empty video_file (shouldn't happen, but defensive)
                    return False
                audio_file = Path(match_obj.video_file).stem
                start_time = match_obj.video_start
            elif hasattr(match_obj, 'video_segment'):
                # utils.Match structure - use the video_segment
                if not hasattr(match_obj.video_segment, 'source_file'):
                    return False
                if not match_obj.video_segment.source_file:
                    # Skip matches with empty source_file
                    return False
                audio_file = Path(match_obj.video_segment.source_file).stem
                start_time = match_obj.video_segment.start_time
            else:
                logger.warning(f"Unknown Match structure: {type(match_obj)}")
                return False

            # Skip empty audio files (defensive check)
            if not audio_file:
                return False

            # Skip stock videos (pexels_, pixabay_) and entity videos - they don't need remapping
            if audio_file.startswith(('pexels_', 'pixabay_', 'entity_')):
                return False

            # Find corresponding video_id from audio downloads
            # Strip timestamp suffix (_0000, _1234, etc.) from audio_file if present
            base_audio_file = audio_file
            if '_' in audio_file:
                # Check if last part after underscore is all digits (timestamp)
                parts = audio_file.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_audio_file = parts[0]

            video_id = None
            for vid, audio in audio_downloads_by_id.items():
                audio_path = audio.file if hasattr(audio, 'file') else audio.get('file', '')
                if Path(audio_path).stem == base_audio_file:
                    video_id = vid
                    break

            if not video_id:
                logger.warning(f"Could not find video_id for audio file: {audio_file} (base: {base_audio_file})")
                return False

            # Look up the downloaded segment containing this match
            key = (video_id, start_time)
            if key in segment_map:
                segment_file, original_start = segment_map[key]

                # Update match to reference video segment file
                if hasattr(match_obj, 'video_file'):
                    old_file = match_obj.video_file
                    match_obj.video_file = segment_file
                else:
                    old_file = match_obj.video_segment.source_file
                    match_obj.video_segment.source_file = segment_file

                logger.debug(f"Remapped match: {old_file} -> {segment_file}")
                return True
            else:
                logger.warning(f"Could not find downloaded segment for match: video_id={video_id}, time={start_time}")
                return False

        # Helper to remap video_segment.source_file (for AlternativeMatch/StrategyMatch)
        def remap_segment(video_segment):
            """Remap video_segment.source_file to video segment file"""
            if not hasattr(video_segment, 'source_file'):
                return False

            audio_file = Path(video_segment.source_file).stem

            # Skip stock videos (pexels_, pixabay_) and entity videos - they don't need remapping
            if audio_file.startswith(('pexels_', 'pixabay_', 'entity_')):
                return False

            # Strip timestamp suffix (_0000, _1234, etc.) from audio_file if present
            base_audio_file = audio_file
            if '_' in audio_file:
                # Check if last part after underscore is all digits (timestamp)
                parts = audio_file.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_audio_file = parts[0]

            # Find corresponding video_id
            video_id = None
            for vid, audio in audio_downloads_by_id.items():
                audio_path = audio.file if hasattr(audio, 'file') else audio.get('file', '')
                if Path(audio_path).stem == base_audio_file:
                    video_id = vid
                    break

            if not video_id:
                return False

            # Look up segment file
            # For video_segments, we need to use start_time instead of video_start
            key = (video_id, video_segment.start_time)
            if key in segment_map:
                segment_file, _ = segment_map[key]
                old_file = video_segment.source_file
                video_segment.source_file = segment_file
                logger.debug(f"Remapped segment: {old_file} -> {segment_file}")
                return True
            return False

        # Update all Match objects (handling both Match and MatchResult)
        updated_count = 0
        for item in state.matches:
            # Check if this is a MatchResult wrapper or a plain Match
            if hasattr(item, 'primary_match'):
                # MatchResult object - update primary match
                if remap_match(item.primary_match):
                    updated_count += 1

                # Update alternatives (V2-V3) - these have video_segment field
                for alt in item.alternatives:
                    if hasattr(alt, 'video_segment') and remap_segment(alt.video_segment):
                        updated_count += 1

                # Update secondary matches (V4-V6) - these have video_segment field
                for sec in item.secondary_matches:
                    if hasattr(sec, 'video_segment') and remap_segment(sec.video_segment):
                        updated_count += 1

                # Update strategy matches (V7+) - these have video_segment field
                for strat in item.strategy_matches:
                    if hasattr(strat, 'video_segment') and remap_segment(strat.video_segment):
                        updated_count += 1
            else:
                # Plain Match object
                if remap_match(item):
                    updated_count += 1

        logger.info(f"Remapped {updated_count} match objects to video segment files")
        print(f"  ✓ Updated {updated_count} matches to reference video segments")

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
        # Skip validation if not in audio-first mode
        # (stage will skip itself in run())
        if config.pipeline.skip_download or not state.downloaded_audio:
            return None

        if not state.matches:
            return "No matches available for segment download"
        return None
