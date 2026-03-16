"""
Entity Videos Stage - Download Entity Stock Videos

Stage 1.6 of the video matching pipeline:
- Downloads stock videos for entities (people, places, organizations) from Pexels, Pixabay
- Maps entities to voiceover segments for timeline placement
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import Stage, StageResult
from ..logging_templates import log_error_with_context, log_stage_complete

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


class EntityVideosStage(Stage):
    """
    Downloads stock videos for entities extracted from voiceover.

    This is an **optional stage** not included in the default 7-stage pipeline.
    It is not registered via @register_stage and must be added manually using
    ``pipeline.add_stage(EntityVideosStage())``, or by using the
    ``create_entity_enhanced_pipeline()`` factory which inserts entity stages
    between ANALYZE and VIDEO_SEARCH.

    Inputs:
        - state.extracted_entities: List of entity dicts with 'text', 'type', 'context'
        - state.topic_context: Documentary topic string
        - state.voiceover_segments: For entity-to-segment mapping

    Outputs:
        - state.entity_videos: Dict[str, EntityVideoResult]

    Configuration:
        - config.image_search.enabled: Master on/off switch (reuses image_search config)
        - config.image_search.use_stock_apis: Must be true to download videos
        - config.image_search.entity_types: List of entity types to search
        - config.image_search.max_entities: Maximum entities to process
        - config.image_search.videos_per_entity: Target videos per entity (default: 3)
        - config.image_search.root_dir: Optional short path override
        - config.image_search.folder_name: Subfolder name (default: "images")
    """

    name = "ENTITY_VIDEOS"
    description = "Download entity stock videos from Pexels/Pixabay"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the entity videos stage"""
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        warnings = []

        # Note: skip_download only affects YouTube downloads, not stock videos
        # Stock videos are from Pexels/Pixabay and should run independently

        # Check if skipped via pipeline config
        if config.pipeline.skip_image_search:
            logger.info("Skipping ENTITY_VIDEOS stage (config: skip_image_search=true)")

            # Try to restore existing entity videos from checkpoint for V10 track
            # Checkpoint stores: {video_count, total_videos, entities: {name: {videos, segment_indices, ...}}}
            checkpoint_data = checkpoint.get_stage_data('entity_videos')
            if checkpoint_data and 'entities' in checkpoint_data:
                from types import SimpleNamespace

                restored_videos = {}
                raw_entities = checkpoint_data['entities']
                for entity_name, entity_data in raw_entities.items():
                    if isinstance(entity_data, dict):
                        videos = entity_data.get('videos', [])
                        segment_indices = entity_data.get('segment_indices', [])
                        restored_videos[entity_name] = SimpleNamespace(
                            videos=videos,
                            segment_indices=segment_indices,
                            entity_name=entity_name,
                            entity_type=entity_data.get('entity_type', 'unknown'),
                            context=entity_data.get('context', ''),
                            query=entity_data.get('query', entity_name)
                        )

                if restored_videos:
                    state.entity_videos = restored_videos
                    total_videos = sum(len(getattr(e, 'videos', [])) for e in restored_videos.values())
                    logger.info(f"Restored {len(restored_videos)} entities with {total_videos} videos from checkpoint")

            return StageResult.ok({
                'skipped': True,
                'reason': 'skip_pipeline_config',
                'restored_from_checkpoint': bool(state.entity_videos)
            })

        # Check if enabled
        if not config.image_search.enabled:
            logger.debug("Image/video search disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'disabled'
            })

        # Check if stock APIs are enabled
        if not config.image_search.use_stock_apis:
            logger.debug("Stock APIs disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'stock_apis_disabled'
            })

        # Check if entities exist
        if not state.extracted_entities:
            logger.debug("No entities available for stock video search")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_entities'
            })

        logger.info("Starting STOCK VIDEO SEARCH stage")

        try:
            from ..media_sources import download_entity_videos, map_entities_to_segments

            # Filter entities by configured types
            allowed_types = config.image_search.entity_types
            entities_to_search = [
                e for e in state.extracted_entities
                if e.get('type', '') in allowed_types
            ]

            if not entities_to_search:
                logger.info(f"No entities of types {allowed_types} to search")
                return StageResult.ok({
                    'skipped': True,
                    'reason': 'no_matching_types',
                    'video_count': 0
                })

            # Apply max_entities limit if configured
            max_entities = getattr(config.image_search, 'max_entities', 0)
            if max_entities > 0 and len(entities_to_search) > max_entities:
                logger.info(f"Limiting to {max_entities} entities (from {len(entities_to_search)})")
                entities_to_search = entities_to_search[:max_entities]

            # Get videos_per_entity from config (default 3)
            videos_per_entity = getattr(config.image_search, 'videos_per_entity', 3)

            # US-99-008: Get duration from StockVideoConfig (was hardcoded 3.0/30.0)
            stock_video_config = getattr(config.image_search, 'stock_video', None)
            if stock_video_config:
                min_duration = getattr(stock_video_config, 'min_duration', 3.0)
                max_duration = getattr(stock_video_config, 'max_duration', 30.0)
            else:
                min_duration = 3.0
                max_duration = 30.0

            logger.info(f"Searching stock videos for {len(entities_to_search)} entities, types: {', '.join(allowed_types)}, videos_per_entity: {videos_per_entity}, duration: {min_duration}s-{max_duration}s")

            # Determine output directory (same as entity_images)
            output_dir = self._get_output_dir(config, checkpoint)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Entity videos output directory: {output_dir}")

            # Download stock videos
            entity_results = download_entity_videos(
                entities=entities_to_search,
                output_dir=str(output_dir),
                topic=state.topic_context or "",
                videos_per_entity=videos_per_entity,
                min_duration=min_duration,
                max_duration=max_duration,
                pexels_key=os.getenv("PEXELS_API_KEY"),
                pixabay_key=os.getenv("PIXABAY_API_KEY"),
                config=config
            )

            # Map entities to segments for timeline placement
            if entity_results and isinstance(entity_results, dict):
                entity_segments = map_entities_to_segments(
                    entities_to_search,
                    state.voiceover_segments
                )

                # Update entity results with segment info
                for entity_name, result in entity_results.items():
                    result.segment_indices = entity_segments.get(entity_name, [])

                # Update state
                state.entity_videos = entity_results

                # US-99-008: Get entity_display_limit from config (default 5)
                entity_display_limit = getattr(config.image_search, 'entity_display_limit', 5)

                # Summary
                total_videos = sum(len(r.videos) for r in entity_results.values())
                logger.info(f"Downloaded {total_videos} stock videos for {len(entity_results)} entities")

                # Show what was found (limited by entity_display_limit)
                display_count = min(entity_display_limit, len(entity_results))
                for name, result in list(entity_results.items())[:display_count]:
                    segments_str = f"segments: {result.segment_indices[:3]}" if result.segment_indices else "no segment matches"
                    logger.debug(f"Entity video: {name} ({result.entity_type}): {len(result.videos)} videos, {segments_str}")

                if len(entity_results) > display_count:
                    logger.debug(f"... and {len(entity_results) - display_count} more entities")
            else:
                logger.info("No stock videos downloaded")
                state.entity_videos = {}

            # Build checkpoint data
            checkpoint_data = self._build_checkpoint_data(entity_results)

            # US-167-009: Log stage completion with timing
            elapsed = time.time() - stage_start_time
            entity_count = len(entity_results) if isinstance(entity_results, dict) else 0
            log_stage_complete(
                logger, "ENTITY_VIDEOS",
                elapsed_seconds=elapsed,
                entities_processed=entity_count,
                videos_downloaded=entity_count
            )

            return StageResult.ok(checkpoint_data, warnings=warnings)

        except ImportError as e:
            error_msg = f"Could not import entity_images module: {e}"
            log_error_with_context(logger, "PIPE-001", error_msg)
            return StageResult.fail(error_msg)

        except Exception as e:
            log_error_with_context(logger, "SEARCH-001", f"Stock video search failed: {e}")
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
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore entity videos from checkpoint data"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # Get config from checkpoint manager
            config = getattr(checkpoint, '_config', None)
            if not config:
                logger.warning("No config available in checkpoint manager")
                return False

            # Restore entity_videos from checkpoint
            entities_data = data.get('entities', {})

            # Reconstruct EntityVideoResult objects
            from ..media_sources.models import EntityVideoResult

            entity_videos = {}
            for entity_name, entity_info in entities_data.items():
                # Verify video files still exist
                video_paths = entity_info.get('videos', [])
                existing_paths = [p for p in video_paths if Path(p).exists()]

                if existing_paths:
                    entity_videos[entity_name] = EntityVideoResult(
                        entity_name=entity_name,
                        entity_type=entity_info.get('entity_type', ''),
                        context=entity_info.get('context', ''),
                        query=entity_info.get('query', ''),
                        videos=existing_paths,
                        segment_indices=entity_info.get('segment_indices', [])
                    )
                else:
                    logger.warning(f"No video files found for entity: {entity_name}")

            state.entity_videos = entity_videos

            video_count = len(state.entity_videos)
            logger.info(f"Restored {self.name}: {video_count} entities")

            if video_count == 0:
                logger.warning(f"No entity videos restored")
                return False

            return True

        except Exception as e:
            log_error_with_context(logger, "PIPE-002", f"Failed to restore {self.name}: {e}")
            return False

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

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        return {
            'inputs': 'entities from state',
            'outputs': 'entity videos',
            'input_count': None,
            'output_count': None,
        }

    def _get_output_dir(
        self,
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> Path:
        """
        Determine output directory based on config.

        Uses same logic as EntityImagesStage (shares same output directory).
        """
        image_cfg = config.image_search

        # Short path mode (e.g., E:/i/ProjectName)
        if getattr(image_cfg, 'root_dir', '') and image_cfg.root_dir:
            project_name = checkpoint.project_dir.name[:15]
            return Path(image_cfg.root_dir) / project_name

        # Project-relative mode (e.g., project_dir/images)
        folder_name = getattr(image_cfg, 'folder_name', 'images')
        return checkpoint.project_dir / folder_name

    def _build_checkpoint_data(
        self,
        entity_results
    ) -> dict:
        """
        Build checkpoint data from entity results.

        Format matches the checkpoint data structure expected by restore().
        """
        if not entity_results or not isinstance(entity_results, dict):
            return {
                'video_count': 0,
                'entities': {}
            }

        entities_data = {}
        for name, result in entity_results.items():
            entities_data[name] = {
                'entity_type': result.entity_type,
                'context': result.context,
                'query': result.query,
                'videos': result.videos,
                'segment_indices': result.segment_indices
            }

        return {
            'video_count': len(entity_results),
            'total_videos': sum(len(r.videos) for r in entity_results.values()),
            'entities': entities_data
        }
