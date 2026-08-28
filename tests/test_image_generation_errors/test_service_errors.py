"""Tests for GeneratedImageService error handling."""

import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from src.generated_images.service import GeneratedImageService, IMAGEN_COST_PER_IMAGE


def _create_mock_image_bytes(width=1792, height=1024):
    """Create minimal valid PNG image bytes using PIL."""
    try:
        from PIL import Image
        import io
        img = Image.new('RGB', (width, height), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except ImportError:
        # Fallback if PIL not available
        return b'\x89PNG\r\n\x1a\n' + b'\x00' * 100


def _make_mock_config(model="imagen-4.0-generate-001", budget_usd=0.0):
    """Create a properly configured mock config for image generation."""
    mock_config = Mock()
    mock_config.image_size = Mock(width=1792, height=1024)
    mock_config.quality = "standard"
    mock_config.model = model
    mock_config.budget_usd = budget_usd
    # Prompt builder attributes
    mock_config.prompt_prefix = ""
    mock_config.include_topic_context = True
    mock_config.provider_keywords = []
    mock_config.quality_keywords = []
    mock_config.use_prompt_enhancement = False
    mock_config.prompt_suffix = ""
    return mock_config


def _make_batch(batch_id, text="Test batch", start_idx=0, end_idx=5):
    """Helper to create a mock batch."""
    batch = Mock()
    batch.batch_id = batch_id
    batch.text = text
    batch.segment_start_index = start_idx
    batch.segment_end_index = end_idx
    batch.start_time = float(start_idx)
    batch.end_time = float(end_idx)
    batch.segment_indices = list(range(start_idx, end_idx))
    return batch


class TestGenerateForBatchesSkipsFailingBatches:
    """Test that generate_for_batches skips batches that raise exceptions and continues."""

    def test_skips_batch_on_provider_exception(self, tmp_path):
        """generate_for_batches should catch provider exceptions and continue to next batch."""
        mock_config = _make_mock_config()
        batch1 = _make_batch("batch_001", "Test batch 1", 0, 5)
        batch2 = _make_batch("batch_002", "Test batch 2", 5, 10)

        # Mock provider - first fails, second succeeds
        mock_provider = Mock()
        mock_provider.generate_image.side_effect = [
            RuntimeError("API error: rate limit exceeded"),
            _create_mock_image_bytes(),
        ]

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches([batch1, batch2], "test topic", str(output_dir))

        # Should have processed batch2 successfully despite batch1 failure
        assert len(results) == 1
        assert results[0].batch_id == "batch_002"

        # Verify first batch was attempted
        assert mock_provider.generate_image.call_count == 2

    def test_skips_batch_on_validation_failure(self, tmp_path):
        """generate_for_batches should skip batches with invalid image data."""
        mock_config = _make_mock_config()
        batch1 = _make_batch("batch_001", "Test batch 1", 0, 5)
        batch2 = _make_batch("batch_002", "Test batch 2", 5, 10)

        # Mock provider - first returns too small bytes (validation failure), second succeeds
        mock_provider = Mock()
        mock_provider.generate_image.side_effect = [
            b"tiny",  # Too small to be valid image
            _create_mock_image_bytes(),
        ]

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches([batch1, batch2], "test topic", str(output_dir))

        # Should have processed batch2 successfully despite batch1 validation failure
        assert len(results) == 1
        assert results[0].batch_id == "batch_002"

    def test_continues_after_multiple_failures(self, tmp_path):
        """generate_for_batches should continue processing after multiple failures."""
        mock_config = _make_mock_config()

        # Create 5 batches
        batches = [_make_batch(f"batch_{i:03d}", f"Test batch {i}", i * 5, (i + 1) * 5) for i in range(5)]

        # Mock provider - fail on batches 1 and 3, succeed on 0, 2, 4
        mock_provider = Mock()
        mock_provider.generate_image.side_effect = [
            RuntimeError("API error"),
            _create_mock_image_bytes(),
            RuntimeError("Timeout"),
            _create_mock_image_bytes(),
            _create_mock_image_bytes(),
        ]

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches(batches, "test topic", str(output_dir))

        # Should have processed 3 batches successfully
        assert len(results) == 3
        result_ids = [r.batch_id for r in results]
        assert "batch_001" in result_ids
        assert "batch_003" in result_ids
        assert "batch_004" in result_ids


class TestBudgetCapRespected:
    """Test that generate_for_batches respects budget_usd cap and stops when exceeded."""

    def test_stops_when_budget_exceeded(self, tmp_path):
        """Should stop generating images when budget_usd would be exceeded."""
        mock_config = _make_mock_config(
            model="imagen-4.0-generate-001",  # $0.04 per image
            budget_usd=0.07  # Enough for 1 image, not 2
        )

        # Create 5 batches
        batches = [_make_batch(f"batch_{i:03d}", f"Test batch {i}", i * 5, (i + 1) * 5) for i in range(5)]

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches(batches, "test topic", str(output_dir))

        # Should only process 1 image (0.04) before budget (0.07) would be exceeded
        # with the second image (would total 0.08)
        assert len(results) == 1
        assert results[0].batch_id == "batch_000"

        # Verify we didn't try to generate more than budget allows
        assert mock_provider.generate_image.call_count == 1

    def test_budget_allows_exact_one_image(self, tmp_path):
        """Should allow exactly one image when budget equals cost_per_image."""
        mock_config = _make_mock_config(
            model="imagen-4.0-generate-001",  # $0.04 per image
            budget_usd=0.04  # Exactly enough for 1 image
        )

        batch1 = _make_batch("batch_001", "Test batch 1", 0, 5)
        batch2 = _make_batch("batch_002", "Test batch 2", 5, 10)

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches([batch1, batch2], "test topic", str(output_dir))

        # Should process exactly 1 image (cost exactly equals budget)
        assert len(results) == 1
        assert results[0].batch_id == "batch_001"

    def test_budget_zero_means_no_limit(self, tmp_path):
        """Should process all batches when budget_usd is 0 (no limit)."""
        mock_config = _make_mock_config(budget_usd=0.0)  # No budget limit

        batches = [_make_batch(f"batch_{i:03d}", f"Test batch {i}", i * 5, (i + 1) * 5) for i in range(3)]

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches(batches, "test topic", str(output_dir))

        # Should process all 3 batches when budget is 0
        assert len(results) == 3

    def test_budget_uses_correct_cost_per_model(self, tmp_path):
        """Should use model-specific cost when checking budget."""
        mock_config = _make_mock_config(
            model="gemini-2.5-flash-image",  # $0.039 per image
            budget_usd=0.08  # Enough for 2 gemini images (0.078)
        )

        batches = [_make_batch(f"batch_{i:03d}", f"Test batch {i}", i * 5, (i + 1) * 5) for i in range(4)]

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches(batches, "test topic", str(output_dir))

        # Should process 2 images at $0.039 each = $0.078, but 3rd would be $0.117 > $0.08
        assert len(results) == 2

    def test_budget_logs_warning_when_exceeded(self, tmp_path, caplog):
        """Should log warning when budget cap is hit."""
        import logging

        # Budget of 0.03 is less than cost_per_image (0.04), so we can't even process 1
        mock_config = _make_mock_config(budget_usd=0.03)

        batch1 = _make_batch("batch_001", "Test batch 1", 0, 5)
        batch2 = _make_batch("batch_002", "Test batch 2", 5, 10)

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        with caplog.at_level(logging.WARNING):
            results = service.generate_for_batches([batch1, batch2], "test topic", str(output_dir))

        # Check that budget warning was logged (budget exceeded before processing batch1)
        assert any("Budget cap reached" in msg for msg in caplog.messages)


class TestServiceRaisesRuntimeError:
    """Test that GeneratedImageService raises RuntimeError when no API key is available."""

    def test_raises_when_no_api_key_and_no_env_var(self, tmp_path):
        """Should raise RuntimeError when no API key is available."""
        mock_config = _make_mock_config()
        service = GeneratedImageService(mock_config, provider=None, api_key="")

        # Clear any environment variables that might provide a key
        env = {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            batch = _make_batch("batch_001", "Test", 0, 5)

            with pytest.raises(RuntimeError, match="No API key found"):
                service.generate_for_batches([batch], "topic", str(tmp_path / "output"))

    def test_raises_when_gemini_api_key_not_in_env(self, tmp_path):
        """Should raise RuntimeError when GEMINI_API_KEY is not set."""
        mock_config = _make_mock_config(model="gemini-2.5-flash-image")
        service = GeneratedImageService(mock_config, provider=None, api_key="")

        # Only GOOGLE_API_KEY set to empty, GEMINI_API_KEY not set
        env = {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            batch = _make_batch("batch_001", "Test", 0, 5)

            with pytest.raises(RuntimeError, match="No API key found"):
                service.generate_for_batches([batch], "topic", str(tmp_path / "output"))

    def test_uses_provided_api_key(self, tmp_path):
        """Should use the api_key provided to constructor when provider is set."""
        mock_config = _make_mock_config()

        # Service created with explicit API key and pre-set provider
        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key_123")

        batch = _make_batch("batch_001", "Test", 0, 5)

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        # When provider is set, _get_provider is not called and env is not checked
        results = service.generate_for_batches([batch], "topic", str(output_dir))

        # Should have processed successfully with provided provider
        assert len(results) == 1

    def test_raises_when_no_provider_and_no_key_available(self, tmp_path):
        """Should raise when no provider set and no API key in env or constructor."""
        mock_config = _make_mock_config()

        # Service with no provider and no API key
        service = GeneratedImageService(mock_config, provider=None, api_key="")

        env = {"GOOGLE_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            batch = _make_batch("batch_001", "Test", 0, 5)

            with pytest.raises(RuntimeError, match="No API key found"):
                service.generate_for_batches([batch], "topic", str(tmp_path / "output"))


class TestGenerateForBatchesWithValidProvider:
    """Test generate_for_batches with properly mocked provider and validation."""

    def test_generates_images_with_proper_mocked_calls(self, tmp_path):
        """Test that generate_for_batches correctly calls provider methods."""
        mock_config = _make_mock_config()

        batch1 = _make_batch("batch_001", "Test batch 1", 0, 5)
        batch2 = _make_batch("batch_002", "Test batch 2", 5, 10)

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches([batch1, batch2], "test topic", str(output_dir))

        assert len(results) == 2

        # Verify provider was called with correct arguments
        assert mock_provider.generate_image.call_count == 2

        # Check first call arguments - prompt is built from batch.text via prompt_builder
        first_call = mock_provider.generate_image.call_args_list[0]
        assert batch1.text in first_call.kwargs['prompt']
        assert first_call.kwargs['size'] == (1792, 1024)
        assert first_call.kwargs['quality'] == 'standard'

    def test_saves_files_with_correct_naming(self, tmp_path):
        """Test that generated images are saved with correct batch_id filenames."""
        mock_config = _make_mock_config()

        batch = _make_batch("my_batch_123", "Test batch", 0, 5)

        mock_provider = Mock()
        mock_provider.generate_image.return_value = _create_mock_image_bytes()

        service = GeneratedImageService(mock_config, provider=mock_provider, api_key="test_key")

        output_dir = tmp_path / "images"
        output_dir.mkdir()

        results = service.generate_for_batches([batch], "test topic", str(output_dir))

        assert len(results) == 1

        # Verify file was saved with batch_id name
        expected_path = output_dir / "my_batch_123.png"
        assert expected_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])