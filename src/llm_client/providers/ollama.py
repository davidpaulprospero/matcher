"""
Ollama local LLM client implementation.
"""

import logging
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError

logger = logging.getLogger(__name__)


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
        cache_dir: str = ".cache/llm_responses"
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "llama3.2", "mistral")
            host: Ollama server URL
            cache_dir: Cache directory
        """
        # Ollama doesn't need API key
        super().__init__("", model, cache_dir)
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
