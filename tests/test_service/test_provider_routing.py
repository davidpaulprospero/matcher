"""Tests for provider routing and cost lookup in GeneratedImageService."""

import pytest
from unittest.mock import MagicMock, patch

from src.generated_images.service import (
    GeneratedImageService,
    IMAGEN_COST_PER_IMAGE,
    IMAGEN_COST_DEFAULT,
)


class TestCostLookup:
    """Test IMAGEN_COST_PER_IMAGE cost map."""

    def test_imagen_cost_default_is_0_04(self):
        """IMAGEN_COST_DEFAULT should be 0.04."""
        assert IMAGEN_COST_DEFAULT == 0.04

    def test_gemini_2_5_flash_image_cost_is_0_039(self):
        """gemini-2.5-flash-image costs $0.039/image."""
        assert IMAGEN_COST_PER_IMAGE['gemini-2.5-flash-image'] == 0.039

    def test_imagen_4_0_generate_001_cost_is_0_04(self):
        """imagen-4.0-generate-001 costs $0.04/image."""
        assert IMAGEN_COST_PER_IMAGE['imagen-4.0-generate-001'] == 0.04

    def test_imagen_4_0_fast_generate_001_cost_is_0_02(self):
        """imagen-4.0-fast-generate-001 costs $0.02/image."""
        assert IMAGEN_COST_PER_IMAGE['imagen-4.0-fast-generate-001'] == 0.02

    def test_imagen_4_0_ultra_generate_001_cost_is_0_06(self):
        """imagen-4.0-ultra-generate-001 costs $0.06/image."""
        assert IMAGEN_COST_PER_IMAGE['imagen-4.0-ultra-generate-001'] == 0.06

    def test_unknown_model_falls_back_to_default_cost(self):
        """Unknown model uses IMAGEN_COST_DEFAULT (0.04)."""
        cost = IMAGEN_COST_PER_IMAGE.get('unknown-model', IMAGEN_COST_DEFAULT)
        assert cost == 0.04

    def test_all_costs_are_positive(self):
        """All costs in the map must be positive."""
        for model, cost in IMAGEN_COST_PER_IMAGE.items():
            assert cost > 0, f"Cost for {model} should be positive"


class TestProviderRouting:
    """Test provider routing based on model name prefix."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = MagicMock()
        config.model = 'imagen-4.0-generate-001'
        config.image_size.width = 1408
        config.image_size.height = 768
        config.quality = 'standard'
        config.budget_usd = 0.0
        return config

    def test_gemini_prefix_routes_to_gemini_flash_provider(self, mock_config):
        """model starting with 'gemini-' routes to GeminiFlashProvider."""
        mock_config.model = 'gemini-2.5-flash-image'
        service = GeneratedImageService(config=mock_config, api_key="test_key")

        with patch('src.generated_images.providers.gemini_flash.GeminiFlashProvider') as mock_gemini_cls:
            mock_instance = MagicMock()
            mock_gemini_cls.return_value = mock_instance
            provider = service._get_provider()
            mock_gemini_cls.assert_called_once_with(api_key="test_key", model='gemini-2.5-flash-image')

    def test_gemini_3_preview_routes_to_gemini_flash_provider(self, mock_config):
        """model starting with 'gemini-' (any variant) routes to GeminiFlashProvider."""
        mock_config.model = 'gemini-3-pro-image-preview'
        service = GeneratedImageService(config=mock_config, api_key="test_key")

        with patch('src.generated_images.providers.gemini_flash.GeminiFlashProvider') as mock_gemini_cls:
            mock_instance = MagicMock()
            mock_gemini_cls.return_value = mock_instance
            provider = service._get_provider()
            mock_gemini_cls.assert_called_once_with(api_key="test_key", model='gemini-3-pro-image-preview')

    def test_imagen_prefix_routes_to_imagen_provider(self, mock_config):
        """model starting with 'imagen-' routes to ImagenProvider."""
        mock_config.model = 'imagen-4.0-generate-001'
        service = GeneratedImageService(config=mock_config, api_key="test_key")

        with patch('src.generated_images.providers.imagen.ImagenProvider') as mock_imagen_cls:
            mock_instance = MagicMock()
            mock_imagen_cls.return_value = mock_instance
            provider = service._get_provider()
            mock_imagen_cls.assert_called_once_with(api_key="test_key", model='imagen-4.0-generate-001')

    def test_imagen_fast_routes_to_imagen_provider(self, mock_config):
        """imagen-4.0-fast-generate-001 routes to ImagenProvider."""
        mock_config.model = 'imagen-4.0-fast-generate-001'
        service = GeneratedImageService(config=mock_config, api_key="test_key")

        with patch('src.generated_images.providers.imagen.ImagenProvider') as mock_imagen_cls:
            mock_instance = MagicMock()
            mock_imagen_cls.return_value = mock_instance
            provider = service._get_provider()
            mock_imagen_cls.assert_called_once_with(api_key="test_key", model='imagen-4.0-fast-generate-001')

    def test_provider_is_cached(self, mock_config):
        """_get_provider returns the same instance on subsequent calls."""
        mock_config.model = 'gemini-2.5-flash-image'
        service = GeneratedImageService(config=mock_config, api_key="test_key")

        with patch('src.generated_images.providers.gemini_flash.GeminiFlashProvider') as mock_gemini_cls:
            mock_instance = MagicMock()
            mock_gemini_cls.return_value = mock_instance
            provider1 = service._get_provider()
            provider2 = service._get_provider()
            assert provider1 is provider2
            assert mock_gemini_cls.call_count == 1
