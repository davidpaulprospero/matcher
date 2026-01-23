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
            video_files = self._get_video_files(state, config)

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
                self._merge_scene_data_to_transcripts(state, scene_data_dict, config)

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
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore scene detection stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data or data.get('skipped'):
                return False

            if not config:
                logger.warning("Cannot restore SCENE_DETECTION without config")
                return False

            # Load scene data from SceneDetector's cache (scene_index.json)
            from ..scene_detection import SceneDetector

            scene_detector = SceneDetector(config)
            scene_data_dict = scene_detector.scene_index

            if not scene_data_dict:
                logger.warning("No scene data in cache during restore")
                return False

            # Store in state for use by other stages
            state.scene_data = scene_data_dict

            # CRITICAL: Merge B-roll flags into text_metadata
            # This ensures B-roll flags are available for matching stage
            if state.text_metadata:
                self._merge_scene_data_to_transcripts(state, scene_data_dict, config)
                broll_count = sum(1 for m in state.text_metadata if isinstance(m, dict) and m.get('is_broll'))
                logger.info(f"Restored SCENE_DETECTION: {len(scene_data_dict)} videos, {broll_count} B-roll entries in text_metadata")
            else:
                logger.info(f"Restored SCENE_DETECTION metadata: {len(scene_data_dict)} videos (text_metadata not yet available)")

            return True

        except Exception as e:
            logger.warning(f"Failed to restore SCENE_DETECTION: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        # Check multiple sources for videos:
        # 1. downloaded_videos (normal flow)
        # 2. transcripts (resume flow - videos discovered during transcript rebuild)
        # 3. remix_files (resume flow - video paths from REMIX stage)
        has_videos = (
            state.downloaded_videos or
            state.downloaded_audio or
            state.transcripts or
            state.remix_files
        )
        if not has_videos:
            return "No videos available for scene detection"
        return None

    # === Helper Methods ===

    def _get_video_files(self, state: 'PipelineState', config: 'Config') -> List[Path]:
        """Get list of video files to analyze (excludes stock footage)"""
        video_files = []

        # Get base video directory for resolving relative paths
        videos_dir = Path(config.downloaded_videos_dir)

        # From downloaded videos (normal flow)
        for dv in state.downloaded_videos:
            if hasattr(dv, 'file'):
                p = Path(dv.file)
                # Resolve relative paths against videos_dir
                video_files.append(p if p.is_absolute() else videos_dir / p)

        # If no downloaded_videos, try transcripts (resume flow)
        if not video_files and state.transcripts:
            for video_path in state.transcripts.keys():
                if video_path and Path(video_path).exists():
                    video_files.append(Path(video_path))

        # If still no videos, try remix_files (resume flow)
        if not video_files and state.remix_files:
            for video_path in state.remix_files:
                if video_path and Path(video_path).exists():
                    video_files.append(Path(video_path))

        # Filter out stock footage - already curated clips, no scene detection needed
        filtered_files = []
        skipped_count = 0
        for p in video_files:
            # Skip files in stock/ folder or with pexels_/pixabay_ prefix
            is_stock = (
                'stock' in p.parts or
                p.stem.startswith('pexels_') or
                p.stem.startswith('pixabay_')
            )
            if is_stock:
                skipped_count += 1
            else:
                filtered_files.append(p)

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} stock footage files (already curated clips)")

        return filtered_files

    def _merge_scene_data_to_transcripts(
        self,
        state: 'PipelineState',
        scene_data_dict: Dict[str, Any],
        config: 'Config'
    ):
        """
        Merge scene metadata into transcript segments.

        For each transcript segment, find the corresponding scene and add:
        - is_broll: bool
        - face_score: float
        - scene_index: int
        """
        logger.info("===== SCENE_DETECTION _merge_scene_data_to_transcripts START =====")
        logger.info(f"  text_metadata entries: {len(state.text_metadata)}")
        logger.info(f"  embeddings empty: {is_embeddings_empty(state.embeddings)}")
        logger.info(f"  transcripts: {len(state.transcripts)} video sets")

        if not state.transcripts:
            logger.info("  ! No transcripts to update")
            return

        broll_set_count = 0  # Debug: count how many segments get is_broll=True
        scene_data_matches = 0  # Debug: count videos with scene data

        # Debug: log sample keys for troubleshooting
        sample_transcript_keys = [Path(k).stem for k in list(state.transcripts.keys())[:3]]
        sample_scene_keys = list(scene_data_dict.keys())[:3]
        logger.info(f"  Sample transcript stems: {sample_transcript_keys}")
        logger.info(f"  Sample scene_data keys: {sample_scene_keys}")

        # Build a lookup that handles segment suffixes in scene_data keys
        # Scene data keys may be like "video_id_0123" while transcript stems are just "video_id"
        # Build mapping: base_video_id -> list of scene_data entries
        scene_data_by_video_id: Dict[str, Any] = {}
        for key, data in scene_data_dict.items():
            # Try to extract base video ID (handle _XXXX segment suffix)
            # YouTube IDs are 11 chars, but with underscore segment it becomes "ID_XXXX"
            parts = key.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
                base_id = parts[0]
            else:
                base_id = key
            # Store the scene_data (use first if multiple segments)
            if base_id not in scene_data_by_video_id:
                scene_data_by_video_id[base_id] = data

        for video_path, segments in state.transcripts.items():
            video_name = Path(video_path).stem
            # Try exact match first, then base ID match
            scene_data = scene_data_dict.get(video_name) or scene_data_by_video_id.get(video_name)

            # If still no match, try normalizing the transcript stem too
            # (transcript may be like "video_id_0504" while scene_data is "video_id_0123")
            if not scene_data:
                parts = video_name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
                    base_name = parts[0]
                    scene_data = scene_data_dict.get(base_name) or scene_data_by_video_id.get(base_name)

            if not scene_data:
                continue
            scene_data_matches += 1

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
                            if scene.is_broll:
                                broll_set_count += 1
                        break

        logger.info(f"Merged scene metadata into {len(state.transcripts)} transcript sets")
        logger.info(f"  Scene data found for {scene_data_matches}/{len(state.transcripts)} videos")
        logger.info(f"  B-roll set on {broll_set_count} transcript segments")

        # Create text_metadata entries for silent videos (stock footage)
        # These videos have scene data but no transcripts, so we need to create
        # text_metadata entries manually for them to be available for matching
        silent_videos_processed = self._create_text_metadata_for_silent_videos(
            state, scene_data_dict, config
        )
        if silent_videos_processed > 0:
            logger.info(f"Created text_metadata entries for {silent_videos_processed} silent videos")

        # Also update text_metadata if it exists (for matching stage)
        logger.info(f"  Checking text_metadata update condition:")
        logger.info(f"    embeddings empty? {is_embeddings_empty(state.embeddings)}")
        logger.info(f"    text_metadata exists? {bool(state.text_metadata)}")
        if not is_embeddings_empty(state.embeddings) and state.text_metadata:
            updated_count = 0
            broll_found = 0
            direct_updates = 0  # For videos with no transcripts

            # Build video_name -> segments mapping (handles audio vs video path mismatch)
            # state.transcripts keys might be audio paths (.mp3) while text_metadata
            # has video paths (.mp4), so we normalize by video name (stem)
            transcript_by_name: Dict[str, list] = {}
            for path, segs in state.transcripts.items():
                video_name = Path(path).stem
                transcript_by_name[video_name] = segs

            logger.info(f"  Built transcript lookup: {len(transcript_by_name)} videos")
            # Debug: check if any transcripts have is_broll set
            broll_in_transcripts = sum(
                1 for segs in transcript_by_name.values()
                for seg in segs
                if getattr(seg, 'is_broll', False)
            )
            logger.info(f"  Transcripts with is_broll=True: {broll_in_transcripts}")

            for meta in state.text_metadata:
                if not isinstance(meta, dict):
                    continue

                video_path = meta.get('video_path')
                if not video_path:
                    continue

                # Try to find matching transcript segment by video name (not full path)
                video_name = Path(video_path).stem
                segments = transcript_by_name.get(video_name, [])
                matched = False

                for transcript in segments:
                    # Match by video path and time (with tolerance)
                    transcript_start = getattr(transcript, 'start_time', getattr(transcript, 'start', 0))
                    meta_start = meta.get('start_time', 0)

                    if abs(float(transcript_start) - float(meta_start)) < 0.1:
                        # Copy scene metadata from transcript
                        is_broll_val = getattr(transcript, 'is_broll', False)
                        meta['is_broll'] = is_broll_val
                        meta['face_score'] = getattr(transcript, 'face_score', 0.5)
                        meta['scene_index'] = getattr(transcript, 'scene_index', None)
                        updated_count += 1
                        if is_broll_val:
                            broll_found += 1
                        matched = True
                        break

                # If no transcript match (silent video), get scene data directly
                if not matched:
                    video_name = Path(video_path).stem
                    scene_data = scene_data_dict.get(video_name)

                    if scene_data:
                        meta_start = meta.get('start_time', 0)
                        meta_end = meta.get('end_time', meta_start + 1.0)
                        meta_mid = (meta_start + meta_end) / 2

                        # Find which scene this text_metadata entry belongs to
                        for scene in scene_data.scenes:
                            if scene.start_time <= meta_mid <= scene.end_time:
                                # Update directly from scene data
                                meta['is_broll'] = scene.is_broll
                                meta['face_score'] = scene.face_score
                                meta['scene_index'] = scene.scene_index
                                direct_updates += 1
                                if scene.is_broll:
                                    broll_found += 1
                                break
                    elif direct_updates == 0:  # Log only first few misses
                        logger.debug(f"  No scene data for silent video: {video_name}")

            if updated_count > 0 or direct_updates > 0:
                logger.info(f"Updated {updated_count} text_metadata entries from transcripts")
                logger.info(f"Updated {direct_updates} text_metadata entries directly from scene data (silent videos)")
                logger.info(f"  B-roll entries found during update: {broll_found}")

            # Count B-roll entries for debugging
            broll_count = sum(1 for m in state.text_metadata if isinstance(m, dict) and m.get('is_broll'))
            logger.info(f"  After update: {broll_count}/{len(state.text_metadata)} entries have is_broll=True")

            if broll_count == 0 and broll_found > 0:
                logger.warning(f"  !!! B-roll data LOST: found {broll_found} during update but final count is 0")
        else:
            logger.info("  ! SKIPPED text_metadata update (embeddings empty or text_metadata missing)")

    def _create_text_metadata_for_silent_videos(
        self,
        state: 'PipelineState',
        scene_data_dict: Dict[str, Any],
        config: 'Config'
    ) -> int:
        """
        Create text_metadata entries for silent videos (stock footage).

        Silent videos have scene data but no transcripts, so they're never
        added to text_metadata during transcription. We need to create entries
        manually so they can be used for matching (especially B-roll matching).

        Optionally uses Vision API to generate descriptions instead of placeholder text.

        Returns: Number of silent videos processed
        """
        from pathlib import Path

        # Find videos in downloaded_videos that have scene data but no transcripts
        silent_videos = []
        for dv in state.downloaded_videos:
            video_path = getattr(dv, 'file', None)
            if not video_path:
                continue

            video_name = Path(video_path).stem

            # Has scene data but not in transcripts = silent video
            if video_name in scene_data_dict and video_path not in state.transcripts:
                silent_videos.append((video_path, video_name))

        if not silent_videos:
            return 0

        # Initialize vision processor if enabled
        vision_processor = None
        use_vision = getattr(config.vision, 'enabled', False)

        if use_vision:
            try:
                from src.vision import VisionProcessor
                vision_processor = VisionProcessor(config)
                if not vision_processor.is_available():
                    logger.info("  Vision API not available (no API key), using placeholder text")
                    vision_processor = None
                else:
                    logger.info(f"  Vision API enabled for {len(silent_videos)} silent videos")
            except Exception as e:
                logger.warning(f"  Failed to initialize vision processor: {e}")
                vision_processor = None

        # Create text_metadata entries for each scene in silent videos
        new_entries = []
        vision_calls = 0

        for video_path, video_name in silent_videos:
            scene_data = scene_data_dict.get(video_name)
            if not scene_data:
                continue

            # Create one text_metadata entry per scene
            for scene in scene_data.scenes:
                # Try to get vision description if available
                description = None
                if vision_processor:
                    try:
                        scene_dict = {
                            'start_time': scene.start_time,
                            'end_time': scene.end_time
                        }
                        description = vision_processor.describe_scene(
                            video_path,
                            scene_dict,
                            cache_dir=config.cache.cache_dir
                        )
                        if description:
                            vision_calls += 1
                    except Exception as e:
                        logger.debug(f"  Vision API failed for {video_name} scene {scene.scene_index}: {e}")

                # Use vision description or fallback to placeholder
                text = description if description else f"[Silent video: {video_name}]"

                entry = {
                    'video_path': video_path,
                    'text': text,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'is_broll': scene.is_broll,
                    'face_score': scene.face_score,
                    'scene_index': scene.scene_index,
                }
                new_entries.append(entry)

        # Add to state.text_metadata
        if new_entries:
            if state.text_metadata is None:
                state.text_metadata = []
            state.text_metadata.extend(new_entries)

            broll_count = sum(1 for e in new_entries if e.get('is_broll'))
            logger.info(f"  Created {len(new_entries)} text_metadata entries for {len(silent_videos)} silent videos ({broll_count} B-roll)")

            if vision_processor and vision_calls > 0:
                stats = vision_processor.get_stats()
                logger.info(f"  Vision API: {vision_calls} scenes described, estimated cost: ${stats['estimated_cost']:.4f}")

            # Compute embeddings for the new entries and add to FAISS index
            self._compute_embeddings_for_silent_videos(state, new_entries)

        return len(silent_videos)

    def _compute_embeddings_for_silent_videos(
        self,
        state: 'PipelineState',
        new_entries: List[Dict[str, Any]]
    ):
        """
        Compute embeddings for silent video entries and add to FA ISS index.

        Silent video entries are created after the main embedding computation,
        so we need to compute their embeddings separately and add them to the index.
        """
        if not new_entries:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Get existing embeddings
            if is_embeddings_empty(state.embeddings):
                logger.warning("  Cannot add silent video embeddings: no existing embeddings")
                return

            # Extract text from new entries
            texts = [e['text'] for e in new_entries]

            # Load embedding model
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

            # Compute embeddings
            new_embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

            # Add to state.embeddings
            state.embeddings = np.vstack([state.embeddings, new_embeddings])

            # Rebuild FAISS index with new embeddings
            if state.embedding_index is not None:
                import faiss
                dimension = state.embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(state.embeddings.astype('float32'))
                state.embedding_index = index

                logger.info(f"  Added {len(new_embeddings)} embeddings for silent videos to index")

        except Exception as e:
            logger.warning(f"Failed to compute embeddings for silent videos: {e}")
