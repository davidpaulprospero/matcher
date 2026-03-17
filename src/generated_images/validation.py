"""Validation helpers for generated image bytes."""

from __future__ import annotations

import io
from typing import Tuple

from PIL import Image


def validate_image_bytes(
    image_data: bytes,
    min_size_bytes: int = 100,
) -> Tuple[bool, str, Tuple[int, int]]:
    """Validate image bytes and return dimensions when valid."""

    if not image_data:
        return False, "image data is empty", (0, 0)

    if len(image_data) < min_size_bytes:
        return (
            False,
            f"image too small ({len(image_data)} bytes), minimum is {min_size_bytes} bytes",
            (0, 0),
        )

    try:
        image = Image.open(io.BytesIO(image_data))
        image.load()
    except Exception as exc:
        return False, f"invalid image: {exc}", (0, 0)

    width, height = image.size
    if width < 1 or height < 1:
        return False, f"invalid dimensions: {width}x{height}", (width, height)

    return True, "valid", (width, height)
