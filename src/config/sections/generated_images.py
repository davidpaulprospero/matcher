"""Generated images configuration: AI image generation from voiceover text.

Controls Imagen-based still image generation for the V12 track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

__all__ = [
    'GeneratedImageSizeConfig',
    'GeneratedImagesConfig',
]

# Imagen-supported size tuples (width, height).
# Copied from ImagenProvider.SIZE_MAP keys to avoid circular import.
IMAGEN_SUPPORTED_SIZES = {
    (1024, 1024),
    (1024, 1792),
    (1792, 1024),
    (2048, 2048),
    (1536, 2048),
    (2048, 1536),
}


@dataclass
class GeneratedImageSizeConfig:
    """Dimensions for generated images."""
    width: int = 1792
    height: int = 1024


@dataclass
class GeneratedImagesConfig:
    """Configuration for AI-generated still images from voiceover text.

    Generates images using Imagen API based on voiceover segment text,
    placed on the V12 track in the OTIO timeline.
    """
    enabled: bool = False
    output_dir: str = "generated_images"
    image_size: GeneratedImageSizeConfig = field(default_factory=GeneratedImageSizeConfig)
    track_source: str = "entity"

    # Prompt knobs
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    include_topic_context: bool = True
    use_prompt_enhancement: bool = True
    provider_keywords: List[str] = field(default_factory=list)
    quality_keywords: List[str] = field(default_factory=list)
    quality: str = "standard"

    # Batching knobs
    min_segments_per_image: int = 7
    max_segments_per_image: int = 9
    batch_target_segments: int = 8
    allow_boundary_batch_outside_range: bool = True
    boundary_min_segments: int = 6
    boundary_max_segments: int = 10
    fallback_single_batch_max_segments: int = 10

    # Provider
    provider: str = "imagen"
    model: str = "imagen-4.0-generate-001"

    def __post_init__(self):
        """Coerce nested dicts and validate image size."""
        if isinstance(self.image_size, dict):
            self.image_size = GeneratedImageSizeConfig(**self.image_size)

        size_tuple = (self.image_size.width, self.image_size.height)
        if size_tuple not in IMAGEN_SUPPORTED_SIZES:
            supported = sorted(IMAGEN_SUPPORTED_SIZES)
            raise ValueError(
                f"Imagen-supported sizes are {supported}, "
                f"got ({self.image_size.width}, {self.image_size.height})"
            )
