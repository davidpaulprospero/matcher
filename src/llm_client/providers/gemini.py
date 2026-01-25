"""
Google Gemini LLM client implementation.
"""

import logging
from typing import Optional
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """
    Client for Google Gemini models via google-generativeai SDK.

    Supports:
    - Text generation
    - Vision (images in request)
    - Configurable models
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        cache_dir: str = ".cache/llm_responses",
        cache_ttl_hours: int = 24
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
            cache_dir: Cache directory
            cache_ttl_hours: TTL for cached LLM responses in hours (0 = never expire)
        """
        super().__init__(api_key, model, cache_dir, cache_ttl_hours)

        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel(model)
        except ImportError:
            raise LLMProviderError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Gemini client: {e}")

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _call_api(self, request: LLMRequest) -> str:
        """
        Call Gemini API.

        Supports:
        - Text-only prompts
        - Vision (prompt + images)

        Args:
            request: LLM request

        Returns:
            Raw text response

        Raises:
            LLMProviderError: On API errors
            LLMTimeoutError: On timeout
        """
        try:
            # Build content for API call
            if request.images:
                # Vision API call
                content = [request.prompt]

                # Add images
                import PIL.Image
                import io

                for img_bytes in request.images:
                    try:
                        pil_image = PIL.Image.open(io.BytesIO(img_bytes))
                        content.append(pil_image)
                    except Exception as e:
                        logger.warning(f"Failed to load image for vision API: {e}")
                        continue

                # Make vision API call
                response = self.genai_model.generate_content(
                    content,
                    request_options={"timeout": request.timeout}
                )
            else:
                # Regular text API call
                response = self.genai_model.generate_content(
                    request.prompt,
                    request_options={"timeout": request.timeout}
                )

            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            else:
                # Handle cases where response doesn't have text (safety filters, etc.)
                logger.warning(f"Gemini response has no text attribute: {response}")
                return ""

        except Exception as e:
            error_msg = str(e).lower()

            # Check for timeout
            if "timeout" in error_msg or "deadline" in error_msg:
                raise LLMTimeoutError(f"Gemini request timed out: {e}")

            # Check for common API errors
            if "api_key" in error_msg or "authentication" in error_msg:
                raise LLMProviderError(f"Gemini authentication error: {e}")
            elif "quota" in error_msg or "rate" in error_msg:
                raise LLMProviderError(f"Gemini quota/rate limit error: {e}")
            elif "safety" in error_msg:
                logger.warning(f"Gemini safety filter triggered: {e}")
                return ""  # Return empty string for safety filter
            else:
                raise LLMProviderError(f"Gemini API error: {e}")
