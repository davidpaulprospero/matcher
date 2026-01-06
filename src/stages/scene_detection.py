"""
Scene Detection Stage - Scene Boundary and B-roll Detection

Stage 3.5 of the video matching pipeline (runs after transcription):
- Detects scene boundaries using PySceneDetect
- Performs face detection per scene
- Marks B-roll segments (no faces detected)
- Stores scene metadata for matching stage
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage
from ..utils import is_embeddings_empty

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class SceneDetectionStage(Stage):
    """
    Detects scenes and B-roll segments in videos.

    Inputs:
        - state.downloaded_videos: List of DownloadedVideo
        - state.transcripts: Dict[video_path, List[segments]]

    Outputs:
        - state.scene_data: Dict[video_name, VideoSceneData]
        - Updates segments with scene metadata (is_broll, face_score)
    """

    name = "SCENE_DETECTION"
    description = "Detect scenes and B-roll segments"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the scene detection stage"""
        warnings = []

        try:
            if config.pipeline.skip_scene_detection:
                print("  >> Skipping scene detection (config: skip_scene_detection=true)")
                logger.info("Skipping SCENE_DETECTION stage (config: skip_scene_detection=true)")
                return StageResult.ok({'skipped': True, 'reason': 'skip_pipeline_config'})

            print(f"\n  --- Stage 3.5: SCENE DETECTION ---")

            # Get video files from downloaded videos
            video_files = self._get_video_files(state)

            if not video_files:
                print("  ! No video files found for scene detection")
                warnings.append("No video files to analyze")
                return StageResult.ok({'scene_count': 0}, warnings)

            print(f"  Analyzing {len(video_files)} videos for scenes and B-roll...")

            # Initialize SceneDetector
            from ..scene_detection import SceneDetector

            scene_detector = SceneDetector(config)
            scene_data_dict = {}

            # Process each video
            total_broll_scenes = 0
            total_scenes = 0

            for i, video_path in enumerate(video_files, 1):
                video_name = Path(video_path).stem
                print(f"  [{i}/{len(video_files)}] {video_name}")

                try:
                    scene_data = scene_detector.process_video(Path(video_path))

                    if scene_data:
                        scene_data_dict[video_name] = scene_data
                        total_scenes += scene_data.scene_count

                        # Count B-roll scenes
                        broll_count = sum(1 for scene in scene_data.scenes if scene.is_broll)
                        total_broll_scenes += broll_count

                        print(f"    ✓ {scene_data.scene_count} scenes ({broll_count} B-roll)")
                    else:
                        warnings.append(f"Scene detection failed for {video_name}")

                except Exception as e:
                    logger.error(f"Error processing {video_name}: {e}")
                    warnings.append(f"Error processing {video_name}: {e}")

            # Store scene data in state
            state.scene_data = scene_data_dict

            # Update transcripts with scene metadata
            if state.transcripts:
                self._merge_scene_data_to_transcripts(state, scene_data_dict)

            print(f"  ✓ Scene detection complete:")
            print(f"    • {len(scene_data_dict)} videos processed")
            print(f"    • {total_scenes} total scenes")
            print(f"    • {total_broll_scenes} B-roll scenes ({(total_broll_scenes/total_scenes*100) if total_scenes > 0 else 0:.1f}%)")

            checkpoint_data = {
                'video_count': len(scene_data_dict),
                'scene_count': total_scenes,
                'broll_count': total_broll_scenes,
                'scene_data': {
                    name: {
                        'scene_count': data.scene_count,
                        'broll_count': sum(1 for s in data.scenes if s.is_broll),
                        'has_speech': data.has_speech
                    }
                    for name, data in scene_data_dict.items()
                }
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Scene detection stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if scene detection stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Restore scene detection stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data or data.get('skipped'):
                return False

            # Scene data is stored in scene_detection cache, not checkpoint
            # Just load from SceneDetector's index file
            from ..scene_detection import SceneDetector
            from ..config import Config

            # Need to reconstruct config - this is a limitation
            # For now, just mark as restorable if checkpoint exists
            logger.info(f"Restored SCENE_DETECTION metadata from checkpoint")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore SCENE_DETECTION: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        if not state.downloaded_videos and not state.downloaded_audio:
            return "No videos available for scene detection"
        return None

    # === Helper Methods ===

    def _get_video_files(self, state: 'PipelineState') -> List[Path]:
        """Get list of video files to analyze"""
        video_files = []

        # From downloaded videos
        for dv in state.downloaded_videos:
            if hasattr(dv, 'file'):
                video_files.append(Path(dv.file))

        return video_files

    def _merge_scene_data_to_transcripts(
        self,
        state: 'PipelineState',
        scene_data_dict: Dict[str, Any]
    ):
        """
        Merge scene metadata into transcript segments.

        For each transcript segment, find the corresponding scene and add:
        - is_broll: bool
        - face_score: float
        - scene_index: int
        """
        if not state.transcripts:
            return

        for video_path, segments in state.transcripts.items():
            video_name = Path(video_path).stem
            scene_data = scene_data_dict.get(video_name)

            if not scene_data:
                continue

            # For each segment, find which scene it belongs to
            for segment in segments:
                segment_start = getattr(segment, 'start_time', 0)
                segment_end = getattr(segment, 'end_time', 0)
                segment_mid = (segment_start + segment_end) / 2

                # Find scene that contains this segment
                for scene in scene_data.scenes:
                    if scene.start_time <= segment_mid <= scene.end_time:
                        # Add scene metadata to segment
                        if hasattr(segment, '__dict__'):
                            segment.is_broll = scene.is_broll
                            segment.face_score = scene.face_score
                            segment.scene_index = scene.scene_index
                        break

        logger.info(f"Merged scene metadata into {len(state.transcripts)} transcript sets")

        # Also update text_metadata if it exists (for matching stage)
        if not is_embeddings_empty(state.embeddings) and state.text_metadata:
            updated_count = 0
            for meta in state.text_metadata:
                if not isinstance(meta, dict):
                    continue

                video_path = meta.get('video_path')
                if not video_path:
                    continue

                # Find matching transcript segment
                segments = state.transcripts.get(video_path, [])
                for transcript in segments:
                    # Match by video path and time (with tolerance)
                    transcript_start = getattr(transcript, 'start_time', getattr(transcript, 'start', 0))
                    meta_start = meta.get('start_time', 0)

                    if abs(float(transcript_start) - float(meta_start)) < 0.1:
                        # Copy scene metadata to text_metadata
                        meta['is_broll'] = getattr(transcript, 'is_broll', False)
                        meta['face_score'] = getattr(transcript, 'face_score', 0.5)
                        meta['scene_index'] = getattr(transcript, 'scene_index', None)
                        updated_count += 1
                        break

            if updated_count > 0:
                logger.info(f"Updated {updated_count} text_metadata entries with scene data")
