"""
MiniMax (MiniMax M2) LLM client stub.

This is a placeholder module created because the upstream commit (ab13c49)
added `from .minimax import MiniMaxClient` to providers/__init__.py and
references in factory.py, but the actual file was missing.

The welsh-farmer-birds pipeline does not use this provider (it uses Gemini
or Anthropic), so this stub only needs to satisfy the import chain.

TODO: Implement actual MiniMax API integration when the upstream file lands.
"""

import logging
from typing import Optional

from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class MiniMaxClient(LLMClient):
    """Stub client for the MiniMax provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "MiniMax-M2.7",
        cache_dir: str = ".cache/llm_responses",
        cache_ttl_hours: int = 24,
        cache_skip_low_quality: bool = False,
        base_url: Optional[str] = None,
    ):
        super().__init__(api_key, model, cache_dir, cache_ttl_hours, cache_skip_low_quality)
        self.base_url = base_url or "https://api.minimax.chat/v1"

    @property
    def provider_name(self) -> str:
        return "minimax"

    def _call_api(self, request: LLMRequest) -> str:
        raise LLMProviderError(
            "MiniMax provider is not yet implemented in this checkout. "
            "The minimax.py module was missing from upstream commit ab13c49. "
            "Use --provider gemini or --provider anthropic instead."
        )