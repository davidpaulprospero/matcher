"""Tests for generated image config wiring."""

from pathlib import Path

import pytest

from src.config import Config
from src.config.sections.generated_images import GeneratedImagesConfig, GeneratedImageSizeConfig


class TestGeneratedImagesConfig:
    @pytest.mark.fast
    def test_generated_images_config_coerces_nested_image_size(self):
        config = GeneratedImagesConfig(
            image_size={"width": 1792, "height": 1024}
        )

        assert isinstance(config.image_size, GeneratedImageSizeConfig)
        assert config.image_size.width == 1792
        assert config.image_size.height == 1024
        assert config.track_source == "entity"

    @pytest.mark.fast
    def test_config_from_yaml_loads_generated_images_section(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    f"project_dir: {tmp_path.as_posix()}",
                    "generated_images:",
                    "  enabled: true",
                    "  output_dir: generated_images",
                    "  image_size:",
                    "    width: 1792",
                    "    height: 1024",
                ]
            ),
            encoding="utf-8",
        )

        config = Config.from_yaml(str(config_path), skip_final_validation=True)

        assert config.generated_images.enabled is True
        assert config.generated_images.image_size.width == 1792
        assert config.generated_images.output_dir == str(tmp_path / "generated_images")

    @pytest.mark.fast
    def test_generated_images_config_rejects_unsupported_imagen_size(self):
        with pytest.raises(ValueError, match="Imagen-supported sizes"):
            GeneratedImagesConfig(
                image_size={"width": 1500, "height": 900}
            )
