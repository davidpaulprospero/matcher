"""
Ollama local LLM client implementation.
"""

import logging
from typing import List, Optional, Tuple
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError

logger = logging.getLogger(__name__)


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

            # Make API call
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=request.timeout
            )

            response.raise_for_status()

            # Parse response
            result = response.json()
            return result.get('response', '')

        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(f"Ollama request timed out: {e}")

        except requests.exceptions.ConnectionError as e:
            raise LLMProviderError(
                f"Failed to connect to Ollama at {self.host}. "
                f"Is Ollama running? Error: {e}"
            )

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None

            if status_code == 404:
                raise LLMProviderError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
            else:
                raise LLMProviderError(f"Ollama API error: {e}")

        except Exception as e:
            raise LLMProviderError(f"Ollama error: {e}")
