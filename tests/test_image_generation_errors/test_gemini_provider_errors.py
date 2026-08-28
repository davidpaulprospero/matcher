"""Tests for GeminiFlashProvider error handling."""

from unittest.mock import Mock, MagicMock, patch

import pytest

from src.generated_images.providers.gemini_flash import GeminiFlashProvider


def _make_mock_response_with_candidates(candidates, content_parts=None):
    """Helper to create mock response with candidates and optional content."""
    mock_response = Mock()
    mock_response.candidates = candidates

    if content_parts is not None:
        mock_content = Mock()
        mock_content.parts = content_parts
        if candidates and candidates[0] is not None:
            candidates[0].content = mock_content

    return mock_response


class TestGeminiFlashProviderNoCandidates:
    """Test that GeminiFlashProvider raises RuntimeError when response.candidates is empty."""

    def test_raises_when_candidates_empty(self):
        """Should raise RuntimeError when response.candidates is empty/None."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        # Set up mock client with mock response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.candidates = []
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="no candidates"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )

    def test_raises_when_candidates_none(self):
        """Should raise RuntimeError when response.candidates is None."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.candidates = None
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="no candidates"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )


class TestGeminiFlashProviderNoInlineData:
    """Test that GeminiFlashProvider raises RuntimeError when no inline_data image is found."""

    def test_raises_when_no_inline_data_in_parts(self):
        """Should raise RuntimeError when no part has inline_data."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        # Create mock parts without inline_data
        mock_part1 = Mock()
        mock_part1.inline_data = None

        mock_part2 = Mock()
        mock_part2.inline_data = None

        # Create content with parts
        mock_content = Mock()
        mock_content.parts = [mock_part1, mock_part2]

        # Create candidate with content
        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="did not include image bytes"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )

    def test_raises_when_inline_data_has_wrong_mime_type(self):
        """Should raise RuntimeError when inline_data has non-image mime type."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        # Create part with text inline_data (not image)
        mock_inline_data = Mock()
        mock_inline_data.mime_type = "text/plain"
        mock_inline_data.data = b"some text data"

        mock_part = Mock()
        mock_part.inline_data = mock_inline_data

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="did not include image bytes"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )

    def test_raises_when_inline_data_data_is_none(self):
        """Should raise RuntimeError when inline_data.data is None."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        # Create part with image inline_data but no data
        mock_inline_data = Mock()
        mock_inline_data.mime_type = "image/png"
        mock_inline_data.data = None

        mock_part = Mock()
        mock_part.inline_data = mock_inline_data

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="did not include image bytes"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )

    def test_returns_bytes_when_valid_inline_data_found(self):
        """Should return bytes when inline_data contains valid image."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Valid PNG header

        mock_inline_data = Mock()
        mock_inline_data.mime_type = "image/png"
        mock_inline_data.data = image_bytes

        mock_part = Mock()
        mock_part.inline_data = mock_inline_data

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        result = provider.generate_image(
            prompt="test prompt",
            size=(1792, 1024),
            quality="standard"
        )

        assert result == bytes(image_bytes)


class TestGeminiFlashProviderNoContent:
    """Test that GeminiFlashProvider raises RuntimeError when response has no content."""

    def test_raises_when_candidate_has_no_content(self):
        """Should raise RuntimeError when candidate.content is None."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        mock_candidate = Mock()
        mock_candidate.content = None

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="no content parts"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )

    def test_raises_when_content_has_no_parts(self):
        """Should raise RuntimeError when content.parts is None."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        mock_content = Mock()
        mock_content.parts = None

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="no content parts"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )

    def test_raises_when_parts_list_is_empty(self):
        """Should raise RuntimeError when content.parts is empty list."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        mock_content = Mock()
        mock_content.parts = []

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        with pytest.raises(RuntimeError, match="no content parts"):
            provider.generate_image(
                prompt="test prompt",
                size=(1792, 1024),
                quality="standard"
            )


class TestGeminiFlashProviderMultipleParts:
    """Test handling of multiple parts in Gemini response."""

    def test_skips_non_image_parts_and_finds_image(self):
        """Should skip parts without image inline_data and find image in later part."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        # First part has text, second has image
        mock_text_inline = Mock()
        mock_text_inline.mime_type = "text/plain"
        mock_text_inline.data = b"some text"

        mock_text_part = Mock()
        mock_text_part.inline_data = mock_text_inline

        mock_image_inline = Mock()
        mock_image_inline.mime_type = "image/png"
        mock_image_inline.data = image_bytes

        mock_image_part = Mock()
        mock_image_part.inline_data = mock_image_inline

        mock_content = Mock()
        mock_content.parts = [mock_text_part, mock_image_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        result = provider.generate_image(
            prompt="test prompt",
            size=(1792, 1024),
            quality="standard"
        )

        assert result == bytes(image_bytes)


class TestGeminiFlashProviderImportErrors:
    """Test that GeminiFlashProvider handles import errors gracefully."""

    def test_raises_when_google_genai_not_installed_client(self):
        """Should raise RuntimeError when google.genai cannot be imported for client."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")
        provider._client = None  # Reset to force re-import

        with patch.dict('sys.modules', {'google': None, 'google.genai': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'google'")):
                with pytest.raises(RuntimeError, match="google-genai package"):
                    _ = provider.client

    def test_raises_when_types_import_fails(self):
        """Should raise RuntimeError when types module cannot be imported."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        mock_client = Mock()
        provider._client = mock_client

        with patch('builtins.__import__', side_effect=ImportError("No module named 'google.genai.types'")):
            with pytest.raises(RuntimeError, match="google-genai package required"):
                provider.generate_image(
                    prompt="test prompt",
                    size=(1792, 1024),
                    quality="standard"
                )


class TestGeminiFlashProviderAspectRatio:
    """Test aspect ratio mapping in GeminiFlashProvider."""

    def test_uses_default_for_unknown_size(self):
        """Should use default 16:9 for unmapped size."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_inline_data = Mock()
        mock_inline_data.mime_type = "image/png"
        mock_inline_data.data = image_bytes

        mock_part = Mock()
        mock_part.inline_data = mock_inline_data

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]

        # Use unmapped size (1234, 5678)
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        # Should not raise, uses default aspect ratio
        result = provider.generate_image(
            prompt="test prompt",
            size=(1234, 5678),
            quality="standard"
        )

        assert result == bytes(image_bytes)

    def test_maps_common_16_9_sizes(self):
        """Should correctly map 16:9 aspect ratio sizes."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        sizes_16_9 = [
            (1024, 576), (1408, 768), (1792, 1024),
            (2048, 1152), (2816, 1536)
        ]

        for width, height in sizes_16_9:
            aspect = provider.ASPECT_RATIO_MAP.get((width, height))
            assert aspect == "16:9", f"Size ({width}, {height}) should map to 16:9"

    def test_maps_9_16_sizes(self):
        """Should correctly map 9:16 aspect ratio sizes."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        sizes_9_16 = [
            (576, 1024), (768, 1408), (1024, 1792), (1536, 2816)
        ]

        for width, height in sizes_9_16:
            aspect = provider.ASPECT_RATIO_MAP.get((width, height))
            assert aspect == "9:16", f"Size ({width}, {height}) should map to 9:16"

    def test_maps_square_sizes(self):
        """Should correctly map 1:1 aspect ratio sizes."""
        provider = GeminiFlashProvider(api_key="test_key", model="gemini-2.5-flash-image")

        sizes_1_1 = [(1024, 1024), (2048, 2048)]

        for width, height in sizes_1_1:
            aspect = provider.ASPECT_RATIO_MAP.get((width, height))
            assert aspect == "1:1", f"Size ({width}, {height}) should map to 1:1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])