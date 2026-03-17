"""Base classes for generated image providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple


class GeneratedImageProvider(ABC):
    """Abstract generated image provider."""

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        size: Tuple[int, int],
        quality: str = "standard",
    ) -> bytes:
        """Generate a single image."""
