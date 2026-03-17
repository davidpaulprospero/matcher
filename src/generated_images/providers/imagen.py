"""Google Imagen provider for generated subtitle stills."""

from __future__ import annotations

from typing import Tuple

from .base import GeneratedImageProvider


class ImagenProvider(GeneratedImageProvider):
    """Generate images with the Google Imagen API via google-genai."""

    SIZE_MAP = {
        (1024, 1024): {"image_size": "1K", "aspect_ratio": "1:1"},
        (1024, 1792): {"image_size": "1K", "aspect_ratio": "9:16"},
        (1792, 1024): {"image_size": "1K", "aspect_ratio": "16:9"},
        (2048, 2048): {"image_size": "2K", "aspect_ratio": "1:1"},
        (1536, 2048): {"image_size": "2K", "aspect_ratio": "3:4"},
        (2048, 1536): {"image_size": "2K", "aspect_ratio": "4:3"},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "imagen-4.0-generate-001",
        sdk_preference: str = "auto",
    ):
        self.api_key = api_key
        self.model = model
        self.sdk_preference = sdk_preference
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "Imagen generation requires the google-genai package. "
                    "Install it with: pip install google-genai"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_image(
        self,
        prompt: str,
        size: Tuple[int, int],
        quality: str = "standard",
    ) -> bytes:
        size_config = self.SIZE_MAP.get(size, self.SIZE_MAP[(1024, 1024)])
        response = self.client.models.generate_images(
            model=self.model,
            prompt=prompt,
            config={
                "image_size": size_config["image_size"],
                "aspect_ratio": size_config["aspect_ratio"],
                "number_of_images": 1,
            },
        )

        generated_images = getattr(response, 'generated_images', None) or []
        if not generated_images:
            raise RuntimeError("Imagen response did not include generated_images")

        image = generated_images[0]
        image_bytes = getattr(getattr(image, 'image', None), 'image_bytes', None)
        if not image_bytes:
            raise RuntimeError("Imagen response did not include image bytes")

        return image_bytes
