"""MiniMax image generation.

Thin client around https://api.minimax.io/v1/image_generation . Returns decoded
image bytes (one or many) for a text prompt. Not tied to the matcher pipeline —
used by skills like distribute-ref-pictures to add AI-generation as an
alternative to pyimagedl stock downloads.

Auth: pass `api_key` to the constructor or set MINIMAX_API_KEY in the env.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional, Tuple

import requests


DEFAULT_MODEL = "image-01"
DEFAULT_ENDPOINT = "https://api.minimax.io/v1/image_generation"
DEFAULT_SIZE: Tuple[int, int] = (1792, 1024)

ASPECT_MAP: dict[tuple[int, int], str] = {
    # 16:9
    (1920, 1080): "16:9",
    (1792, 1024): "16:9",
    (1408, 768): "16:9",
    (1280, 720): "16:9",
    (1024, 576): "16:9",
    # 9:16
    (1080, 1920): "9:16",
    (1024, 1792): "9:16",
    (768, 1408): "9:16",
    (720, 1280): "9:16",
    (576, 1024): "9:16",
    # 1:1
    (2048, 2048): "1:1",
    (1024, 1024): "1:1",
    (512, 512): "1:1",
    # 4:3 / 3:4
    (2048, 1536): "4:3",
    (1024, 768): "4:3",
    (1536, 2048): "3:4",
    (768, 1024): "3:4",
}

DEFAULT_ASPECT = "16:9"


def aspect_for_size(size: tuple[int, int]) -> str:
    """Map a (width, height) tuple to MiniMax aspect_ratio string."""
    return ASPECT_MAP.get(tuple(size), DEFAULT_ASPECT)


class MiniMaxImageProvider:
    """Sync client for MiniMax's image-01 generation API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = 120,
    ):
        resolved = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not resolved:
            raise RuntimeError(
                "MINIMAX_API_KEY not set. Provide via constructor api_key= or "
                "set the MINIMAX_API_KEY environment variable."
            )
        self.api_key = resolved
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate_image(
        self,
        prompt: str,
        *,
        size: tuple[int, int] = DEFAULT_SIZE,
        aspect_ratio: Optional[str] = None,
    ) -> bytes:
        """Generate one image and return decoded bytes.

        `aspect_ratio` wins over `size` when provided; otherwise the size tuple
        is mapped via `ASPECT_MAP`. Unknown sizes fall back to "16:9".
        """
        aspect = aspect_ratio or aspect_for_size(size)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "aspect_ratio": aspect,
            "response_format": "base64",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        images = ((body.get("data") or {}).get("image_base64")) or []
        if not images:
            raise RuntimeError(
                f"MiniMax returned no images. status={response.status_code} "
                f"body_preview={response.text[:300]!r}"
            )
        return base64.b64decode(images[0])

    def generate_image_to_file(
        self,
        prompt: str,
        output_path,
        *,
        size: tuple[int, int] = DEFAULT_SIZE,
        aspect_ratio: Optional[str] = None,
    ) -> Path:
        """Generate one image and write it to `output_path`.

        The file extension is auto-corrected based on the decoded bytes'
        magic-byte signature (JPEG `b"\\xff\\xd8\\xff"` vs PNG `b"\\x89PNG"`).
        Returns the final `Path` actually written.
        """
        image_bytes = self.generate_image(
            prompt, size=size, aspect_ratio=aspect_ratio
        )
        out = Path(output_path)
        if image_bytes.startswith(b"\xff\xd8\xff"):
            correct_ext = ".jpg"
        elif image_bytes.startswith(b"\x89PNG"):
            correct_ext = ".png"
        else:
            correct_ext = out.suffix or ".jpg"
        if out.suffix.lower() != correct_ext.lower():
            out = out.with_suffix(correct_ext)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(image_bytes)
        return out


def generate_image(prompt: str, **kwargs) -> bytes:
    """Module-level convenience wrapper."""
    return MiniMaxImageProvider().generate_image(prompt, **kwargs)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_ENDPOINT",
    "DEFAULT_SIZE",
    "ASPECT_MAP",
    "MiniMaxImageProvider",
    "aspect_for_size",
    "generate_image",
]
