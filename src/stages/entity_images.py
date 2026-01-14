"""
Entity Images Stage - Download Entity Images

Stage 1.5 of the video matching pipeline:
- Downloads images for entities (people, places, organizations) from Google, Bing, Pexels, Pixabay
- Uses local cache and optional global entity cache
- Maps entities to voiceover segments for timeline placement
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class EntityImagesStage(Stage):
    """
    Downloads images for entities extracted from voiceover.

    Inputs:
        - state.extracted_entities: List of entity dicts with 'text', 'type', 'context'
        - state.topic_context: Documentary topic string
        - state.voiceover_segments: For entity-to-segment mapping

    Outputs:
        - state.entity_images: Dict[str, EntityImageResult]

    Configuration:
        - config.image_search.enabled: Master on/off switch
        - config.image_search.entity_types: List of entity types to search
        - config.image_search.max_entities: Maximum entities to process
        - config.image_search.images_per_entity: Target images per entity
        - config.image_search.min_size_mb: Minimum image file size
        - config.image_search.use_google: Enable Google Images
        - config.image_search.use_bing: Enable Bing Images
        - config.image_search.use_stock_apis: Enable Pexels/Pixabay
        - config.image_search.root_dir: Optional short path override
        - config.image_search.folder_name: Subfolder name (default: "images")
        - config.image_search.entity_cache.enabled: Cross-project cache
        - config.image_search.download_timeout: Per-image timeout
        - config.image_search.max_search_time: Total search timeout per entity
    """

    name = "ENTITY_IMAGES"
    description = "Download entity images from Google/Bing/Pexels/Pixabay"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the entity images stage"""
        warnings = []

        # Check if skipped via pipeline config
        if config.pipeline.skip_image_search:
            print("  >> Skipping image search (config: skip_image_search=true)")
            logger.info("Skipping ENTITY_IMAGES stage (config: skip_image_search=true)")

            # Try to restore existing entity images from checkpoint for V9 track
            # Checkpoint stores: {entity_count, total_images, entities: {name: {images, segment_indices, ...}}}
            checkpoint_data = checkpoint.get_stage_data('entity_images')
            if checkpoint_data and 'entities' in checkpoint_data:
                from types import SimpleNamespace

                restored_images = {}
                raw_entities = checkpoint_data['entities']
                for entity_name, entity_data in raw_entities.items():
                    if isinstance(entity_data, dict):
                        images = entity_data.get('images', [])
                        segment_indices = entity_data.get('segment_indices', [])
                        # Create EntityImageResult-like structure
                        restored_images[entity_name] = SimpleNamespace(
                            images=images,
                            segment_indices=segment_indices,
                            entity_name=entity_name,
                            entity_type=entity_data.get('entity_type', 'unknown'),
                            context=entity_data.get('context', ''),
                            query=entity_data.get('query', entity_name)
                        )

                if restored_images:
                    state.entity_images = restored_images
                    total_images = sum(len(getattr(e, 'images', [])) for e in restored_images.values())
                    print(f"  >> Restored {len(restored_images)} entities with {total_images} images from checkpoint")
                    logger.info(f"Restored {len(restored_images)} entity images from checkpoint")

            return StageResult.ok({
                'skipped': True,
                'reason': 'skip_pipeline_config',
                'restored_from_checkpoint': bool(state.entity_images)
            })

        # Check if enabled
        if not config.image_search.enabled:
            logger.debug("Image search disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'disabled'
            })

        # Check if entities exist
        if not state.extracted_entities:
            logger.debug("No entities available for image search")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_entities'
            })

        print(f"\n  ─── Stage 1.5: ENTITY IMAGE SEARCH ───")

        try:
            from ..media_sources import download_entity_images, map_entities_to_segments

            # Filter entities by configured types
            allowed_types = config.image_search.entity_types
            entities_to_search = [
                e for e in state.extracted_entities
                if e.get('type', '') in allowed_types
            ]

            if not entities_to_search:
                print(f"  No entities of types {allowed_types} to search")
                return StageResult.ok({
                    'skipped': True,
                    'reason': 'no_matching_types',
                    'entity_count': 0
                })

            # Apply max_entities limit if configured
            max_entities = getattr(config.image_search, 'max_entities', 0)
            if max_entities > 0 and len(entities_to_search) > max_entities:
                print(f"  Limiting to {max_entities} entities (from {len(entities_to_search)})")
                entities_to_search = entities_to_search[:max_entities]

            print(f"  Searching images for {len(entities_to_search)} entities")
            print(f"  Entity types: {', '.join(allowed_types)}")
            print(f"  Images per entity: {config.image_search.images_per_entity}")
            print(f"  Minimum size: {config.image_search.min_size_mb}MB")

            # Determine output directory
            output_dir = self._get_output_dir(config, checkpoint)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Output directory: {output_dir}")

            # Initialize entity cache if enabled
            entity_cache = self._init_entity_cache(config, checkpoint)

            # Download images (checks local cache first, then global cache if enabled)
            entity_results = download_entity_images(
                entities=entities_to_search,
                output_dir=str(output_dir),
                topic=state.topic_context or "",
                images_per_entity=config.image_search.images_per_entity,
                min_size_mb=config.image_search.min_size_mb,
                use_google=config.image_search.use_google,
                use_bing=getattr(config.image_search, 'use_bing', False),
                use_stock_apis=config.image_search.use_stock_apis,
                pexels_key=os.getenv("PEXELS_API_KEY"),
                pixabay_key=os.getenv("PIXABAY_API_KEY"),
                download_timeout=getattr(config.image_search, 'download_timeout', 10),
                max_search_time=getattr(config.image_search, 'max_search_time', 300),
                max_results_to_check=getattr(config.image_search, 'max_results_to_check', 500),
                search_until_found=getattr(config.image_search, 'search_until_found', True),
                entity_cache=entity_cache,
                source_project=checkpoint.project_dir.name,
                skip_local_cache=getattr(checkpoint, 'refresh_entities', False),
                config=config
            )

            # Map entities to segments for timeline placement
            if entity_results:
                entity_segments = map_entities_to_segments(
                    entities_to_search,
                    state.voiceover_segments
                )

                # Update entity results with segment info
                for entity_name, result in entity_results.items():
                    result.segment_indices = entity_segments.get(entity_name, [])

                # Update state
                state.entity_images = entity_results

                # Summary
                total_images = sum(len(r.images) for r in entity_results.values())
                print(f"\n  ✓ Downloaded {total_images} images for {len(entity_results)} entities")

                # Show what was found
                for name, result in list(entity_results.items())[:5]:
                    segments_str = f"segments: {result.segment_indices[:3]}" if result.segment_indices else "no segment matches"
                    print(f"    • {name} ({result.entity_type}): {len(result.images)} images, {segments_str}")

                if len(entity_results) > 5:
                    print(f"    ... and {len(entity_results) - 5} more entities")
            else:
                print(f"  ⚠ No images downloaded")
                state.entity_images = {}

            # Build checkpoint data
            checkpoint_data = self._build_checkpoint_data(entity_results)

            return StageResult.ok(checkpoint_data, warnings=warnings)

        except ImportError as e:
            error_msg = f"Could not import entity_images module: {e}"
            logger.error(error_msg)
            print(f"  ⚠ Image search module not available")
            return StageResult.fail(error_msg)

        except Exception as e:
            logger.error(f"Image search failed: {e}", exc_info=True)
            print(f"  ⚠ Image search failed: {e}")
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
        """Restore entity images from disk using .entity.json metadata files"""
        try:
            from ..media_sources import restore_entity_images_from_disk

            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # Get config from checkpoint manager
            config = getattr(checkpoint, '_config', None)
            if not config:
                logger.warning("No config available in checkpoint manager")
                return False

            # Get images directory
            images_dir = self._get_output_dir(config, checkpoint)
            if not images_dir.exists():
                logger.warning(f"Images directory not found: {images_dir}")
                return False

            # Restore from disk
            state.entity_images = restore_entity_images_from_disk(
                str(images_dir),
                voiceover_segments=state.voiceover_segments
            )

            entity_count = len(state.entity_images)
            logger.info(f"Restored {self.name}: {entity_count} entities")

            if entity_count == 0:
                logger.warning(f"No entity images restored from {images_dir}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to restore {self.name}: {e}", exc_info=True)
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

    def _get_output_dir(
        self,
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> Path:
        """
        Determine output directory based on config.

        Supports two modes:
        1. Short path mode: config.image_search.root_dir (e.g., "E:/i")
        2. Project-relative mode: checkpoint.project_dir / folder_name
        """
        image_cfg = config.image_search

        # Short path mode (e.g., E:/i/ProjectName)
        if getattr(image_cfg, 'root_dir', '') and image_cfg.root_dir:
            project_name = checkpoint.project_dir.name[:15]
            return Path(image_cfg.root_dir) / project_name

        # Project-relative mode (e.g., project_dir/images)
        folder_name = getattr(image_cfg, 'folder_name', 'images')
        return checkpoint.project_dir / folder_name

    def _init_entity_cache(
        self,
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ):
        """
        Initialize cross-project entity cache if enabled.

        Returns:
            EntityCache instance or None if disabled/unavailable
        """
        cache_config = getattr(config.image_search, 'entity_cache', None)

        # Debug: Check what we got
        logger.debug(f"Entity cache config: {cache_config}")
        logger.debug(f"Entity cache enabled: {getattr(cache_config, 'enabled', False) if cache_config else 'N/A'}")

        if not cache_config:
            logger.debug("No entity_cache config found")
            return None

        if not getattr(cache_config, 'enabled', False):
            logger.debug("Entity cache is disabled in config")
            return None

        try:
            from ..entity_cache import EntityCache
            entity_cache = EntityCache(cache_config)
            stats = entity_cache.get_stats()
            total = stats.get('total_entities', 0)
            if total > 0:
                print(f"  Global entity cache: {total} entities available")
            else:
                print(f"  Global entity cache: enabled (empty, will populate)")
            return entity_cache
        except ImportError as e:
            logger.warning(f"EntityCache module not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize entity cache: {e}", exc_info=True)
            return None

    def _build_checkpoint_data(
        self,
        entity_results
    ) -> dict:
        """
        Build checkpoint data from entity results.

        Format matches the checkpoint data structure expected by restore().
        """
        if not entity_results:
            return {
                'entity_count': 0,
                'total_images': 0,
                'entities': {}
            }

        entities_data = {}
        for name, result in entity_results.items():
            entities_data[name] = {
                'entity_type': result.entity_type,
                'context': result.context,
                'query': result.query,
                'images': result.images,
                'segment_indices': result.segment_indices
            }

        return {
            'entity_count': len(entity_results),
            'total_images': sum(len(r.images) for r in entity_results.values()),
            'entities': entities_data
        }
