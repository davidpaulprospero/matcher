"""
Ollama local LLM client implementation.
"""

import logging
import time
from typing import List, Optional, Tuple
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError
from src.logging_templates import log_error_with_context

logger = logging.getLogger(__name__)


def _format_correlation(correlation_id: Optional[str]) -> str:
    """Format correlation ID into log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


def check_ollama_available(host: str = "http://localhost:11434", timeout: int = 5) -> Tuple[bool, Optional[str], List[str]]:
    """Check if Ollama is running and return available models.

    Args:
        host: Ollama server URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_available, error_message, list_of_models)
        - is_available: True if Ollama is responding
        - error_message: None if available, else reason for failure
        - list_of_models: List of installed model names (empty if unavailable)
    """
    try:
        import requests
    except ImportError:
        return False, "requests package not installed", []

    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        if response.status_code != 200:
            return False, f"Ollama returned status {response.status_code}", []

        data = response.json()
        models = data.get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        return True, None, model_names

    except Exception as e:
        if "ConnectionRefusedError" in str(type(e).__name__) or "Connection refused" in str(e):
            return False, "Ollama not running (connection refused)", []
        elif "ConnectTimeout" in str(type(e).__name__) or "timed out" in str(e).lower():
            return False, "Ollama not responding (timeout)", []
        else:
            return False, f"Error checking Ollama: {e}", []


def check_ollama_model_available(model: str, host: str = "http://localhost:11434", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """Check if a specific model is available in Ollama.

    Args:
        model: Model name to check (e.g., "llama3.2")
        host: Ollama server URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_available, error_message)
    """
    available, error, models = check_ollama_available(host, timeout)

    if not available:
        return False, error

    # Check if model is in list (handle both "llama3.2" and "llama3.2:latest" formats)
    model_base = model.split(":")[0]
    for m in models:
        if m == model or m.startswith(model_base):
            return True, None

    return False, f"Model '{model}' not found. Available: {models}. Run: ollama pull {model}"


class OllamaClient(LLMClient):
    """
    Client for local Ollama models.

    Supports:
    - Local LLM inference
    - Configurable host
    - No API key required
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        cache_dir: str = ".cache/llm_responses",
        cache_ttl_hours: int = 24,
        cache_skip_low_quality: bool = False
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "llama3.2", "mistral")
            host: Ollama server URL
            cache_dir: Cache directory
            cache_ttl_hours: TTL for cached LLM responses in hours (0 = never expire)
            cache_skip_low_quality: If True, skip cache entries with quality_tier='low'
        """
        # Ollama doesn't need API key
        super().__init__("", model, cache_dir, cache_ttl_hours, cache_skip_low_quality)
        self.host = host.rstrip('/')

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _call_api(self, request: LLMRequest) -> str:
        """
        Call Ollama API.

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
        logger.info(
            f"[OLLAMA] LLM API request{corr}: "
            f"model={self.model}, host={self.host}, prompt_length={prompt_length}, "
            f"max_tokens={request.max_tokens}, timestamp={timestamp}"
        )

        try:
            import requests
        except ImportError:
            raise LLMProviderError(
                "requests package not installed. "
                "Install with: pip install requests"
            )

        try:
            # Build request payload
            payload = {
                "model": self.model,
                "prompt": request.prompt,
                "stream": False
            }

            # Add optional parameters
            if request.temperature != 0.7:
                payload["temperature"] = request.temperature

            # Log request details at DEBUG level (truncated)
            prompt_preview = request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt
            logger.debug(
                f"[OLLAMA] API request{corr}: model={self.model}, "
                f"host={self.host}, max_tokens={request.max_tokens}, "
                f"temperature={request.temperature}, prompt={prompt_preview}"
            )

            # Make API call
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=request.timeout
            )

            response.raise_for_status()

            # Parse response
            result = response.json()
            result_text = result.get('response', '')

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Log successful response at INFO level with status and latency
            logger.info(
                f"[OLLAMA] LLM API response{corr}: "
                f"model={self.model}, status=success, latency_ms={response_time_ms:.1f}"
            )

            # Try to extract token usage if available (Ollama may not always provide this)
            prompt_tokens = result.get('prompt_eval_count')
            completion_tokens = result.get('eval_count')
            if prompt_tokens or completion_tokens:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
                logger.info(
                    f"[OLLAMA] Token usage{corr}: tokens_used={total_tokens}, "
                    f"input={prompt_tokens}, output={completion_tokens}, "
                    f"latency_ms={response_time_ms:.1f}"
                )

            return result_text

        except requests.exceptions.Timeout as e:
            log_error_with_context(logger, "MATCH-005", f"Ollama request timed out: {e}",
                                   correlation_id=correlation_id,
                                   model=self.model, host=self.host, prompt_length=prompt_length)
            raise LLMTimeoutError(f"Ollama request timed out: {e}")

        except requests.exceptions.ConnectionError as e:
            log_error_with_context(logger, "MATCH-004", f"Ollama connection error: {e}",
                                   correlation_id=correlation_id,
                                   model=self.model, host=self.host)
            raise LLMProviderError(
                f"Failed to connect to Ollama at {self.host}. "
                f"Is Ollama running? Error: {e}"
            )

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            error_msg = str(e).lower()

            # Check for rate limiting (429 Too Many Requests)
            if status_code == 429:
                # Log rate limit at WARNING level for detection/tracking
                logger.warning(f"[MATCH-006] Ollama rate limit detected{corr}: {e}")
                log_error_with_context(logger, "MATCH-006", f"Ollama rate limit exceeded (HTTP 429)",
                                       correlation_id=correlation_id,
                                       model=self.model, host=self.host)
                raise LLMProviderError(f"Ollama rate limit exceeded: {e}")

            if status_code == 404:
                log_error_with_context(logger, "MATCH-001", f"Ollama model not found: {self.model}",
                                       correlation_id=correlation_id,
                                       model=self.model, host=self.host)
                raise LLMProviderError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
            else:
                log_error_with_context(logger, "MATCH-001", f"Ollama HTTP error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, host=self.host, status_code=status_code)
                raise LLMProviderError(f"Ollama API error: {e}")

        except Exception as e:
            log_error_with_context(logger, "MATCH-001", f"Ollama error: {e}",
                                   correlation_id=correlation_id,
                                   model=self.model, host=self.host)
            raise LLMProviderError(f"Ollama error: {e}")
