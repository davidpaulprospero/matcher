"""Tests for src.config.sections.generated_images."""

import pytest

from src.config.sections.generated_images import (
    GeneratedImageSizeConfig,
    GeneratedImagesConfig,
    IMAGEN_SUPPORTED_SIZES,
)


class TestDefaults:
    """Test default values of GeneratedImagesConfig."""

    def test_default_model(self):
        """Default model is 'gemini-2.5-flash-image'."""
        config = GeneratedImagesConfig()
        assert config.model == "gemini-2.5-flash-image"

    def test_default_provider(self):
        """Default provider is 'gemini-flash'."""
        config = GeneratedImagesConfig()
        assert config.provider == "gemini-flash"


class TestImagenSupportedSizes:
    """Test IMAGEN_SUPPORTED_SIZES contains expected values."""

    def test_1792_1024_in_supported_sizes(self):
        """(1792, 1024) is in IMAGEN_SUPPORTED_SIZES."""
        assert (1792, 1024) in IMAGEN_SUPPORTED_SIZES

    def test_1408_768_in_supported_sizes(self):
        """(1408, 768) is in IMAGEN_SUPPORTED_SIZES."""
        assert (1408, 768) in IMAGEN_SUPPORTED_SIZES


class TestValidation:
    """Test GeneratedImagesConfig validation in __post_init__."""

    def test_unsupported_size_raises_value_error(self):
        """Config with size not in supported sizes raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GeneratedImagesConfig(
                image_size=GeneratedImageSizeConfig(width=999, height=999)
            )
        assert "Imagen-supported sizes are" in str(exc_info.value)

    def test_negative_budget_raises_value_error(self):
        """Config with negative budget_usd raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            GeneratedImagesConfig(budget_usd=-1.0)
        assert "budget_usd must be >= 0" in str(exc_info.value)

    def test_dict_image_size_converts_to_config(self):
        """Config with dict image_size correctly converts to GeneratedImageSizeConfig."""
        config = GeneratedImagesConfig(
            image_size={"width": 1024, "height": 576}
        )
        assert isinstance(config.image_size, GeneratedImageSizeConfig)
        assert config.image_size.width == 1024
        assert config.image_size.height == 576
