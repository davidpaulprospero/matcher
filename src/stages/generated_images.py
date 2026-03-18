"""
Generated Images Stage - AI Image Generation from Voiceover

Generates still images using Imagen API based on voiceover segment text.
Images are batched by segment groups and placed on the V12 track.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult
from ..logging_templates import log_error_with_context, log_stage_complete, log_stage_start

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


class GeneratedImagesStage(Stage):
    """
    Generate AI images from voiceover text using Imagen.

    This is an **optional stage** that generates still images from batched
    voiceover segments. Images are placed on the V12 track in the OTIO timeline.

    Inputs:
        - state.voiceover_segments: Voiceover segments for batching
        - state.topic_context: Documentary topic for prompt context

    Outputs:
        - state.generated_images: List[GeneratedImageResult]

    Configuration:
        - config.generated_images.enabled: Master on/off switch
        - config.generated_images.output_dir: Directory for saved images
        - config.generated_images.image_size: Width/height for generation
        - config.generated_images.provider: Image generation provider
    """

    name = "GENERATED_IMAGES"
    description = "Generate AI images from voiceover text using Imagen"
    DEPENDS_ON = ["ANALYZE"]
    PRODUCES = ["generated_images"]

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the generated images stage."""
        stage_start_time = time.time()

        gen_config = getattr(config, 'generated_images', None)
        if not gen_config or not getattr(gen_config, 'enabled', False):
            logger.debug("Generated images disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'disabled'
            })

        if not state.voiceover_segments:
            logger.debug("No voiceover segments available for image generation")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_segments'
            })

        log_stage_start(logger, self.name)

        try:
            from ..generated_images import build_generated_image_batches, GeneratedImageService

            # Build batches from voiceover segments
            batches = build_generated_image_batches(state.voiceover_segments, gen_config)
            logger.info(f"Built {len(batches)} image batches from {len(state.voiceover_segments)} segments")

            if not batches:
                return StageResult.ok({
                    'skipped': True,
                    'reason': 'no_batches',
                    'segment_count': len(state.voiceover_segments)
                })

            # Determine output directory — always relative to project dir.
            # config._resolve_paths() may have resolved against the wrong base
            # (dev repo vs actual project), so use the raw relative name.
            raw_dir = getattr(gen_config, 'output_dir', 'generated_images')
            # Strip any pre-resolved absolute path back to the basename
            raw_basename = Path(raw_dir).name if Path(raw_dir).is_absolute() else raw_dir
            output_dir = str(checkpoint.project_dir / raw_basename)

            # Generate images — pass Gemini key from config for .env support
            api_key = getattr(getattr(config, 'api_keys', None), 'gemini_api_key', '')
            service = GeneratedImageService(gen_config, api_key=api_key)
            results = service.generate_for_batches(
                batches=batches,
                topic_context=state.topic_context or "",
                output_dir=output_dir,
            )

            # Store results in state
            state.generated_images = results

            # Build checkpoint data
            total_cost = sum(r.cost_usd for r in results)
            checkpoint_data = {
                'batch_count': len(batches),
                'generated_count': len(results),
                'total_cost_usd': total_cost,
                'output_dir': output_dir,
                'results': [
                    {
                        'batch_id': r.batch_id,
                        'file': r.file,
                        'prompt': r.prompt,
                        'width': r.width,
                        'height': r.height,
                        'start_time': r.start_time,
                        'end_time': r.end_time,
                        'segment_start_index': r.segment_start_index,
                        'segment_end_index': r.segment_end_index,
                        'segment_indices': r.segment_indices,
                        'cost_usd': r.cost_usd,
                    }
                    for r in results
                ]
            }

            elapsed = time.time() - stage_start_time
            log_stage_complete(
                logger, self.name,
                elapsed_seconds=elapsed,
                batches=len(batches),
                images_generated=len(results),
                total_cost_usd=total_cost,
            )

            return StageResult.ok(checkpoint_data)

        except ImportError as e:
            error_msg = f"Could not import generated_images module: {e}"
            log_error_with_context(logger, "PIPE-001", error_msg)
            return StageResult.fail(error_msg)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_error_with_context(logger, "GENIMG-001", f"Image generation failed: {e}\n{tb}")
            return StageResult.fail(str(e))

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if this stage can be skipped (already completed)."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore generated images from checkpoint data."""
        try:
            from ..state import GeneratedImageResult

            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            results_data = data.get('results', [])
            if not results_data:
                logger.info(f"No generated image results in checkpoint")
                return True  # Stage ran but produced no images (e.g., disabled)

            restored = []
            for r in results_data:
                if not isinstance(r, dict):
                    continue
                restored.append(GeneratedImageResult(
                    batch_id=r.get('batch_id', ''),
                    file=r.get('file', ''),
                    prompt=r.get('prompt', ''),
                    width=r.get('width', 0),
                    height=r.get('height', 0),
                    start_time=r.get('start_time', 0.0),
                    end_time=r.get('end_time', 0.0),
                    segment_start_index=r.get('segment_start_index', 0),
                    segment_end_index=r.get('segment_end_index', 0),
                    cost_usd=r.get('cost_usd', 0.0),
                    segment_indices=r.get('segment_indices', []),
                ))

            state.generated_images = restored
            logger.info(f"Restored {self.name}: {len(restored)} generated images")
            return True

        except Exception as e:
            log_error_with_context(logger, "PIPE-002", f"Failed to restore {self.name}: {e}")
            return False
