"""
Google Gemini LLM client implementation.
"""

import logging
import time
from typing import Optional
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError
from ..cost import calculate_llm_cost
from src.logging_templates import log_error_with_context

logger = logging.getLogger(__name__)


def _format_correlation(correlation_id: Optional[str]) -> str:
    """Format correlation ID into log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


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
        cache_ttl_hours: int = 24,
        cache_skip_low_quality: bool = False
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google API key
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
            cache_dir: Cache directory
            cache_ttl_hours: TTL for cached LLM responses in hours (0 = never expire)
            cache_skip_low_quality: If True, skip cache entries with quality_tier='low'
        """
        super().__init__(api_key, model, cache_dir, cache_ttl_hours, cache_skip_low_quality)

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
        start_time = time.time()
        prompt_length = len(request.prompt)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Get correlation ID
        correlation_id = getattr(request, 'correlation_id', None)
        corr = _format_correlation(correlation_id)

        # Log request start at INFO level with provider, model, prompt length, timestamp
        vision_mode = bool(request.images)
        logger.info(
            f"[GEMINI] LLM API request{corr}: "
            f"model={self.model}, prompt_length={prompt_length}, "
            f"vision_mode={vision_mode}, timestamp={timestamp}"
        )

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

                # Log request details at DEBUG level
                logger.debug(
                    f"[GEMINI] API request{corr}: model={self.model}, "
                    f"vision_mode=True, num_images={len(request.images)}"
                )

                # Make vision API call
                response = self.genai_model.generate_content(
                    content,
                    request_options={"timeout": request.timeout}
                )
            else:
                # Regular text API call
                # Log request details at DEBUG level (truncated prompt)
                prompt_preview = request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt
                logger.debug(
                    f"[GEMINI] API request{corr}: model={self.model}, "
                    f"vision_mode=False, prompt={prompt_preview}"
                )

                response = self.genai_model.generate_content(
                    request.prompt,
                    request_options={"timeout": request.timeout}
                )

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[GEMINI] LLM API response{corr}: "
                f"model={self.model}, status=success, latency_ms={response_time_ms:.1f}"
            )

            # Extract token usage for cost tracking (US-162-010)
            # Gemini provides usage metadata in the response
            self._last_tokens_used = None
            self._last_input_tokens = None
            self._last_output_tokens = None

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                # Gemini 2.0 uses usage_metadata
                usage = response.usage_metadata
                self._last_input_tokens = getattr(usage, 'prompt_token_count', None)
                self._last_output_tokens = getattr(usage, 'candidates_token_count', None)
                if self._last_input_tokens and self._last_output_tokens:
                    self._last_tokens_used = self._last_input_tokens + self._last_output_tokens
                elif self._last_input_tokens:
                    self._last_tokens_used = self._last_input_tokens
                elif self._last_output_tokens:
                    self._last_tokens_used = self._last_output_tokens

            # Log token usage and cost estimate at INFO level
            if self._last_tokens_used:
                cost = calculate_llm_cost(
                    provider="gemini",
                    model=self.model,
                    input_tokens=self._last_input_tokens,
                    output_tokens=self._last_output_tokens
                )
                logger.info(
                    f"[GEMINI] Token usage{corr}: tokens_used={self._last_tokens_used}, "
                    f"input={self._last_input_tokens}, output={self._last_output_tokens}, "
                    f"estimated_cost=${cost:.6f}, latency_ms={response_time_ms:.1f}"
                )

            # Extract text from response
            if hasattr(response, 'text'):
                result = response.text
                logger.debug(f"[GEMINI] Response extracted{corr}: {len(result)} chars")
                return result
            else:
                # Handle cases where response doesn't have text (safety filters, etc.)
                logger.warning(f"[GEMINI] Response has no text attribute{corr}: {response}")
                return ""

        except Exception as e:
            error_msg = str(e).lower()

            # Check for timeout
            if "timeout" in error_msg or "deadline" in error_msg:
                log_error_with_context(logger, "MATCH-005", f"Gemini request timed out: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, prompt_length=prompt_length)
                raise LLMTimeoutError(f"Gemini request timed out: {e}")

            # Check for common API errors
            if "api_key" in error_msg or "authentication" in error_msg:
                log_error_with_context(logger, "MATCH-004", f"Gemini authentication error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model)
                raise LLMProviderError(f"Gemini authentication error: {e}")
            elif "quota" in error_msg or "rate" in error_msg:
                # Log rate limit at WARNING level for detection/tracking
                logger.warning(f"[MATCH-006] Gemini rate limit detected{corr}: {e}")
                log_error_with_context(logger, "MATCH-006", f"Gemini quota/rate limit error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, prompt_length=prompt_length)
                raise LLMProviderError(f"Gemini quota/rate limit error: {e}")
            elif "safety" in error_msg:
                logger.warning(f"[GEMINI] Safety filter triggered{corr}: {e}")
                return ""  # Return empty string for safety filter
            else:
                log_error_with_context(logger, "MATCH-001", f"Gemini API error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, prompt_length=prompt_length)
                raise LLMProviderError(f"Gemini API error: {e}")
