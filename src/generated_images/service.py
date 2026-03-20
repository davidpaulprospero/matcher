"""Orchestrator service for generating images from voiceover batches."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .prompt_builder import GeneratedImagePromptBuilder
from .validation import validate_image_bytes
from ..state import GeneratedImageResult

if TYPE_CHECKING:
    from .providers.imagen import ImagenProvider
    from ..state import GeneratedImageBatch

logger = logging.getLogger(__name__)

# Imagen 4 per-image pricing (USD) by model variant.
# Source: Google Cloud billing actuals (March 2026) — list price is $0.04
# but billed rate is ~$0.048/image. Using observed rates for accuracy.
IMAGEN_COST_PER_IMAGE: Dict[str, float] = {
    'imagen-4.0-generate-001': 0.048,       # standard (billed ~$0.048)
    'imagen-4.0-fast-generate-001': 0.02,    # fast
    'imagen-4.0-ultra-generate-001': 0.06,   # ultra
}
IMAGEN_COST_DEFAULT = 0.048  # fallback for unknown model variants


class GeneratedImageService:
    """Orchestrates batch image generation: prompt -> API -> validate -> save."""

    def __init__(self, config: object, provider: Optional['ImagenProvider'] = None, api_key: str = ""):
        self.config = config
        self.prompt_builder = GeneratedImagePromptBuilder(config)
        self._provider = provider
        self._api_key = api_key

    def generate_for_batches(
        self,
        batches: List['GeneratedImageBatch'],
        topic_context: str,
        output_dir: str,
    ) -> List[GeneratedImageResult]:
        """Generate images for each batch and save to disk.

        Args:
            batches: List of GeneratedImageBatch with segment text and timing.
            topic_context: Documentary topic for prompt context.
            output_dir: Directory to save generated PNG files.

        Returns:
            List of GeneratedImageResult with file paths and timing info.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        provider = self._get_provider()
        width = getattr(getattr(self.config, 'image_size', None), 'width', 1792)
        height = getattr(getattr(self.config, 'image_size', None), 'height', 1024)
        quality = getattr(self.config, 'quality', 'standard')

        model = getattr(self.config, 'model', 'imagen-4.0-generate-001')
        cost_per_image = IMAGEN_COST_PER_IMAGE.get(model, IMAGEN_COST_DEFAULT)

        results: List[GeneratedImageResult] = []
        total_cost = 0.0

        logger.info(f"Imagen pricing: ${cost_per_image:.2f}/image ({model}), "
                     f"estimated total: ${cost_per_image * len(batches):.2f} for {len(batches)} batches")

        budget_usd = getattr(self.config, 'budget_usd', 0.0)
        budget_hit = False

        if budget_usd > 0:
            logger.info(f"Budget cap: ${budget_usd:.2f} "
                        f"(~{int(budget_usd / cost_per_image)} images max)")

        for batch in batches:
            if budget_usd > 0 and (total_cost + cost_per_image) > budget_usd:
                logger.warning(f"Budget cap reached: ${total_cost:.2f} of "
                               f"${budget_usd:.2f}. Skipping remaining batches.")
                budget_hit = True
                break

            prompt = self.prompt_builder.build(batch.text, topic_context)
            logger.info(f"Generating image for batch {batch.batch_id} "
                        f"(segments {batch.segment_start_index}-{batch.segment_end_index})")
            logger.debug(f"Prompt: {prompt[:200]}...")

            try:
                image_bytes = provider.generate_image(
                    prompt=prompt,
                    size=(width, height),
                    quality=quality,
                )
            except Exception as e:
                logger.warning(f"Image generation failed for batch {batch.batch_id}: {e}")
                continue

            valid, msg, dims = validate_image_bytes(image_bytes)
            if not valid:
                logger.warning(f"Validation failed for batch {batch.batch_id}: {msg}")
                continue

            total_cost += cost_per_image

            file_name = f"{batch.batch_id}.png"
            file_path = output_path / file_name
            file_path.write_bytes(image_bytes)
            logger.info(f"Saved {file_name} ({dims[0]}x{dims[1]}) — "
                        f"${cost_per_image:.2f} (running total: ${total_cost:.2f})")

            results.append(GeneratedImageResult(
                batch_id=batch.batch_id,
                file=str(file_path),
                prompt=prompt,
                width=dims[0],
                height=dims[1],
                start_time=batch.start_time,
                end_time=batch.end_time,
                segment_start_index=batch.segment_start_index,
                segment_end_index=batch.segment_end_index,
                segment_indices=list(batch.segment_indices),
                cost_usd=cost_per_image,
            ))

        logger.info(f"Generated {len(results)}/{len(batches)} images — "
                     f"total cost: ${total_cost:.2f}"
                     f"{' (budget cap hit)' if budget_hit else ''}")
        return results

    def _get_provider(self) -> 'ImagenProvider':
        """Lazy-create ImagenProvider from config + env var."""
        if self._provider is not None:
            return self._provider

        from .providers.imagen import ImagenProvider

        api_key = (
            self._api_key
            or os.environ.get('GOOGLE_API_KEY', '')
            or os.environ.get('GEMINI_API_KEY', '')
        )
        if not api_key:
            raise RuntimeError(
                "No API key found for Imagen image generation. "
                "Set GEMINI_API_KEY in .env or config api_keys.gemini_api_key"
            )

        model = getattr(self.config, 'model', 'imagen-4.0-generate-001')
        self._provider = ImagenProvider(api_key=api_key, model=model)
        return self._provider
