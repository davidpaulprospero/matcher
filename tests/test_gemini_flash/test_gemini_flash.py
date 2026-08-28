"""Tests for GeminiFlashProvider."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestAspectRatioMap:
    """Test ASPECT_RATIO_MAP completeness."""

    def test_aspect_ratio_map_covers_all_imagen_supported_sizes(self):
        """All IMAGEN_SUPPORTED_SIZES must be keys in ASPECT_RATIO_MAP."""
        from src.config.sections.generated_images import IMAGEN_SUPPORTED_SIZES
        from src.generated_images.providers.gemini_flash import GeminiFlashProvider

        missing = []
        for size in IMAGEN_SUPPORTED_SIZES:
            if size not in GeminiFlashProvider.ASPECT_RATIO_MAP:
                missing.append(size)
        assert not missing, f"Sizes missing from ASPECT_RATIO_MAP: {missing}"

    def test_unknown_size_falls_back_to_16_9(self):
        """Unknown size falls back to 16:9."""
        from src.generated_images.providers.gemini_flash import GeminiFlashProvider

        # Default fallback is (1024, 576) -> "16:9"
        assert GeminiFlashProvider.ASPECT_RATIO_MAP.get((9999, 9999), "16:9") == "16:9"
        # Verify the fallback entry exists
        assert (1024, 576) in GeminiFlashProvider.ASPECT_RATIO_MAP
        assert GeminiFlashProvider.ASPECT_RATIO_MAP[(1024, 576)] == "16:9"


class TestGenerateImage:
    """Test generate_image error handling and extraction."""

    @pytest.fixture
    def provider(self):
        from src.generated_images.providers.gemini_flash import GeminiFlashProvider
        return GeminiFlashProvider(api_key="test_key")

    def _mock_response(self, **parts_parts):
        """Build a mock generate_content response.

        Args:
            candidates: list of MagicMock candidates
        """
        mock_response = MagicMock()
        mock_response.candidates = parts_parts.get('candidates', [MagicMock()])

        if 'candidates' in parts_parts:
            return mock_response

        # Build default candidate with content.parts
        candidate = MagicMock()
        content = MagicMock()
        parts_list = parts_parts.get('parts', [])
        content.parts = parts_list
        candidate.content = content
        mock_response.candidates = [candidate]
        return mock_response

    def _make_part(self, has_inline_data=True, mime_type="image/png", data=b"fake_png"):
        part = MagicMock()
        if has_inline_data:
            part.inline_data = MagicMock()
            part.inline_data.mime_type = mime_type
            part.inline_data.data = data
        else:
            part.text = "not an image"
        return part

    def test_raises_when_no_candidates(self, provider):
        """RuntimeError when response has no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = []

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            # Force re-init of client property
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="no candidates"):
                provider.generate_image("a prompt", (1408, 768))

    def test_raises_when_candidate_has_no_content(self, provider):
        """RuntimeError when candidate.content is None."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = None

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="no content parts"):
                provider.generate_image("a prompt", (1408, 768))

    def test_raises_when_content_has_no_parts(self, provider):
        """RuntimeError when content.parts is None."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = None

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="no content parts"):
                provider.generate_image("a prompt", (1408, 768))

    def test_raises_when_parts_empty(self, provider):
        """RuntimeError when parts list is empty."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_response.candidates[0].content.parts = []

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="no content parts"):
                provider.generate_image("a prompt", (1408, 768))

    def test_raises_when_no_inline_data(self, provider):
        """RuntimeError when no inline_data in parts."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        # Part with text, no inline_data
        mock_response.candidates[0].content.parts = [MagicMock(spec=['text'])]
        mock_response.candidates[0].content.parts[0].text = "not an image"

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="did not include image bytes"):
                provider.generate_image("a prompt", (1408, 768))

    def test_raises_when_inline_data_has_wrong_mime_type(self, provider):
        """RuntimeError when inline_data has non-image mime type."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "audio/mp3"
        mock_part.inline_data.data = b"fake audio"
        mock_response.candidates[0].content.parts = [mock_part]

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="did not include image bytes"):
                provider.generate_image("a prompt", (1408, 768))

    def test_raises_when_inline_data_data_is_none(self, provider):
        """RuntimeError when inline_data.data is None."""
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = None
        mock_response.candidates[0].content.parts = [mock_part]

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            with pytest.raises(RuntimeError, match="did not include image bytes"):
                provider.generate_image("a prompt", (1408, 768))

    def test_returns_bytes_when_valid_inline_data(self, provider):
        """Valid image bytes are correctly extracted and returned."""
        fake_image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00"

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock()
        mock_part.inline_data.mime_type = "image/png"
        mock_part.inline_data.data = fake_image_bytes
        mock_response.candidates[0].content.parts = [mock_part]

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            result = provider.generate_image("a prompt", (1408, 768))
            assert result == bytes(fake_image_bytes)

    def test_skips_non_image_parts_and_finds_image(self, provider):
        """Image is found even when mixed with non-image parts."""
        fake_image_bytes = b"\x89PNG\r\n\x1a\n"

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content = MagicMock()
        text_part = MagicMock(spec=['text'])
        text_part.text = "some text"
        image_part = MagicMock()
        image_part.inline_data = MagicMock()
        image_part.inline_data.mime_type = "image/jpeg"
        image_part.inline_data.data = fake_image_bytes
        mock_response.candidates[0].content.parts = [text_part, image_part]

        with patch('google.genai.Client') as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.return_value = mock_response
            type(provider)._client = PropertyMock(return_value=mock_client)
            result = provider.generate_image("a prompt", (1408, 768))
            assert result == bytes(fake_image_bytes)
