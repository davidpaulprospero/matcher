"""
Download Segments Stage - Download matched video segments

Stage 6 of the simplified 7-stage pipeline:
- Downloads only the matched video segments (not full videos)
- Runs after MATCH and ITERATIVE_MATCH stages
- Efficient: only downloads portions of videos that are actually used
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage, validate_required_state_attrs

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, DownloadedVideo

logger = logging.getLogger(__name__)


@register_stage
class DownloadVideoSegmentsStage(Stage):
    """
    Downloads video segments after matching.

    In the simplified 7-stage pipeline, this stage downloads only the
    portions of YouTube videos that were matched to voiceover segments.

    Inputs:
        - state.matches: List of Match objects with video IDs and time ranges
        - state.video_ids: List of video IDs from VIDEO_SEARCH stage

    Outputs:
        - state.downloaded_segments: List of DownloadedVideo with local file paths
        - Updated state.matches with local file paths
    """

    name = "DOWNLOAD_SEGMENTS"
    description = "Download matched video segments"

    def __init__(self):
        self.downloader = None

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the video segment download stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['matches'], self.name)

        warnings = []

        try:
            if not state.matches:
                print("  >> No matches to download")
                return StageResult.ok({'skipped': True, 'reason': 'no_matches'}, warnings)

            print(f"\n  --- Stage 6: DOWNLOAD VIDEO SEGMENTS ---")

            # Get download settings from config
            download_config = config.download
            buffer_seconds = getattr(download_config, 'segment_buffer', 5.0)

            print(f"  Downloading matched segments")
            print(f"    Buffer: {buffer_seconds}s before/after each match")

            # Collect segments to download
            segments_to_download = self._collect_matched_segments(state)

            if not segments_to_download:
                print("  ! No valid segments to download")
                return StageResult.ok({'skipped': True, 'reason': 'no_valid_segments'}, warnings)

            print(f"    Total segments: {len(segments_to_download)}")

            # Initialize downloader
            from ..downloader import VideoDownloader
            self.downloader = VideoDownloader(config=config)
            output_dir = Path(config.downloaded_videos_dir)

            # Download segments with progress callback for checkpointing
            def checkpoint_progress(current: int, total: int, downloaded: list):
                """Save progress checkpoint during download"""
                checkpoint_data = {
                    'segment_count': len(downloaded),
                    'segments_completed': current,
                    'segments_total': total,
                    'in_progress': current < total,
                }
                checkpoint.save_intermediate('DOWNLOAD_SEGMENTS', checkpoint_data)

            downloaded_segments = self._download_segments(
                segments_to_download,
                output_dir,
                buffer_seconds,
                checkpoint_progress
            )

            print(f"\n  + Downloaded {len(downloaded_segments)} video segments")

            # Update matches to reference local files
            self._update_matches_with_local_paths(state, downloaded_segments)

            # Store downloaded segments in state
            state.downloaded_segments = downloaded_segments

            checkpoint_data = {
                'segment_count': len(downloaded_segments),
                'total_matches': len(state.matches),
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Video segment download failed: {e}")
            return StageResult.fail(str(e), warnings)

    def _collect_matched_segments(self, state: 'PipelineState') -> List[Dict[str, Any]]:
        """Collect segment info from matches for downloading"""
        segments = []
        seen = set()  # Avoid duplicate downloads

        for match in state.matches:
            # Handle MatchResult structure (has primary_match)
            if hasattr(match, 'primary_match') and match.primary_match:
                pm = match.primary_match
                if hasattr(pm, 'video_segment') and pm.video_segment:
                    video_id = getattr(pm.video_segment, 'source_file', '')
                    start_time = getattr(pm.video_segment, 'start_time', 0.0)
                    end_time = getattr(pm.video_segment, 'end_time', start_time + 10.0)
                else:
                    continue
            # Handle plain Match structure
            elif hasattr(match, 'video_file'):
                video_id = match.video_file
                start_time = getattr(match, 'video_start', 0.0)
                end_time = getattr(match, 'video_end', start_time + 10.0)
            else:
                continue

            # Skip if no video ID
            if not video_id:
                continue

            # Create unique key for deduplication
            key = (video_id, round(start_time), round(end_time))
            if key in seen:
                continue
            seen.add(key)

            segments.append({
                'video_id': video_id,
                'start': start_time,
                'end': end_time,
            })

        return segments

    def _download_segments(
        self,
        segments: List[Dict[str, Any]],
        output_dir: Path,
        buffer_seconds: float,
        progress_callback
    ) -> List['DownloadedVideo']:
        """Download video segments using VideoDownloader's retry queue.

        Uses the downloader's impersonation, escalation, and retry queue
        infrastructure instead of raw yt-dlp calls.
        """
        from ..state import DownloadedVideo
        import yt_dlp

        downloaded = []
        total = len(segments)

        for idx, seg in enumerate(segments, 1):
            video_id = seg['video_id']
            start = max(0, seg['start'] - buffer_seconds)
            end = seg['end'] + buffer_seconds

            # Create output filename
            output_file = output_dir / f"{video_id}_{int(start)}_{int(end)}.mp4"

            if output_file.exists():
                logger.info(f"Segment already exists: {output_file}")
                downloaded.append(DownloadedVideo(
                    file=str(output_file),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    source='segment_cache'
                ))
                continue

            try:
                # Download segment using yt-dlp with downloader's infrastructure
                url = f"https://www.youtube.com/watch?v={video_id}"

                ydl_opts = {
                    'format': 'best[height<=1080]',
                    'outtmpl': str(output_file),
                    'quiet': True,
                    'no_warnings': True,
                    # Time-based download options
                    'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
                    'force_keyframes_at_cuts': True,
                }

                # Apply impersonation from downloader if available
                if self.downloader and self.downloader.impersonation_manager:
                    try:
                        imp_opts = self.downloader.impersonation_manager.get_ydl_options(tier=1)
                        ydl_opts.update(imp_opts)
                    except Exception:
                        pass

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                if output_file.exists():
                    downloaded.append(DownloadedVideo(
                        file=str(output_file),
                        url=url,
                        source='segment_download'
                    ))
                    print(f"  [{idx}/{total}] Downloaded {video_id} ({start:.0f}s-{end:.0f}s)")
                else:
                    logger.warning(f"Download succeeded but file not found: {output_file}")

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to download segment {video_id}: {error_msg}")

                # Add failed download to retry queue for batch retry later
                if self.downloader and self.downloader.retry_queue:
                    self.downloader.retry_queue.add(
                        video_id=f"{video_id}_{int(start)}_{int(end)}",
                        keyword='segment',
                        tier='segment',
                        error_message=error_msg
                    )
                    logger.debug(f"Added {video_id} to retry queue")

            # Checkpoint progress
            if progress_callback:
                progress_callback(idx, total, downloaded)

        # Process retry queue if there are pending items
        self._process_retry_queue(output_dir, buffer_seconds, downloaded, total, progress_callback)

        return downloaded

    def _process_retry_queue(
        self,
        output_dir: Path,
        buffer_seconds: float,
        downloaded: List['DownloadedVideo'],
        total: int,
        progress_callback
    ) -> None:
        """Process any failed downloads in the retry queue.

        Attempts to retry failed segment downloads using the downloader's
        retry queue infrastructure.
        """
        from ..state import DownloadedVideo
        import yt_dlp

        if not self.downloader or not self.downloader.retry_queue:
            return

        retry_queue = self.downloader.retry_queue
        if not retry_queue.has_pending():
            return

        pending = retry_queue.get_pending_items()
        logger.info(f"Processing {len(pending)} items from retry queue")

        # Start retry pass (applies configured delay)
        retry_queue.start_retry_pass()

        for item in pending:
            # Parse video_id from the retry item (format: video_id_start_end)
            parts = item.video_id.rsplit('_', 2)
            if len(parts) < 3:
                logger.warning(f"Invalid retry item format: {item.video_id}")
                retry_queue.mark_failed(item.video_id)
                continue

            video_id = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                logger.warning(f"Invalid time range in retry item: {item.video_id}")
                retry_queue.mark_failed(item.video_id)
                continue

            output_file = output_dir / f"{video_id}_{start}_{end}.mp4"

            if output_file.exists():
                retry_queue.mark_success(item.video_id)
                continue

            try:
                url = f"https://www.youtube.com/watch?v={video_id}"

                ydl_opts = {
                    'format': 'best[height<=1080]',
                    'outtmpl': str(output_file),
                    'quiet': True,
                    'no_warnings': True,
                    'download_ranges': lambda info, ydl: [{'start_time': start, 'end_time': end}],
                    'force_keyframes_at_cuts': True,
                }

                # Apply impersonation
                if self.downloader.impersonation_manager:
                    try:
                        imp_opts = self.downloader.impersonation_manager.get_ydl_options(tier=1)
                        ydl_opts.update(imp_opts)
                    except Exception:
                        pass

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                if output_file.exists():
                    downloaded.append(DownloadedVideo(
                        file=str(output_file),
                        url=url,
                        source='segment_retry'
                    ))
                    retry_queue.mark_success(item.video_id)
                    logger.info(f"Retry succeeded for {video_id}")
                else:
                    retry_queue.mark_failed(item.video_id)

            except Exception as e:
                logger.warning(f"Retry failed for {video_id}: {e}")
                retry_queue.mark_failed(item.video_id)

            # Update checkpoint with retry progress
            if progress_callback:
                progress_callback(len(downloaded), total, downloaded)

    def _update_matches_with_local_paths(
        self,
        state: 'PipelineState',
        downloaded_segments: List['DownloadedVideo']
    ):
        """Update match objects to reference local file paths"""
        # Build mapping from video_id to local file
        file_map = {}
        for seg in downloaded_segments:
            # Extract video_id from filename
            filename = Path(seg.file).stem
            parts = filename.split('_')
            if parts:
                video_id = parts[0]
                file_map[video_id] = seg.file

        # Update matches
        updated_count = 0
        for match in state.matches:
            if hasattr(match, 'video_file') and match.video_file:
                video_id = match.video_file
                if video_id in file_map:
                    match.video_file = file_map[video_id]
                    updated_count += 1

        logger.info(f"Updated {updated_count} matches with local file paths")

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
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore from checkpoint"""
        # Segment files are on disk, scan for them
        try:
            if config:
                output_dir = Path(config.downloaded_videos_dir)
                if output_dir.exists():
                    from ..state import DownloadedVideo
                    segments = []
                    for f in output_dir.glob('*_*_*.mp4'):
                        segments.append(DownloadedVideo(
                            file=str(f),
                            source='restored'
                        ))
                    state.downloaded_segments = segments
                    logger.info(f"Restored DOWNLOAD_SEGMENTS: {len(segments)} segments from disk")
            return True
        except Exception as e:
            logger.warning(f"Failed to restore DOWNLOAD_SEGMENTS: {e}")
            return True  # Non-critical, proceed anyway

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs"""
        if not state.matches:
            return "No matches available for segment download"
        return None
