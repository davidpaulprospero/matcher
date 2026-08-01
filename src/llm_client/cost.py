"""
API Cost Tracking Module (US-162-010)

Provides cost calculation for LLM and embedding API calls based on model pricing.
Prices are in USD per million tokens (or per 1000 tokens for embeddings).

Usage:
    from src.llm_client.cost import calculate_llm_cost, calculate_embedding_cost, get_cost_tracker

    # Calculate cost for an LLM call
    cost = calculate_llm_cost(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        input_tokens=1000,
        output_tokens=500
    )

    # Track cumulative costs
    tracker = get_cost_tracker()
    tracker.add_llm_cost(0.005)
    tracker.get_total_cost()
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import threading

# LLM Pricing (USD per million tokens)
# Updated February 2026
LLM_PRICING = {
    # Anthropic Claude
    "anthropic": {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00,  # $3.00 per million input
            "output": 15.00,  # $15.00 per million output
        },
        "claude-3-5-haiku-20241022": {
            "input": 0.80,
            "output": 4.00,
        },
        "claude-3-haiku-20240307": {
            "input": 0.25,
            "output": 1.25,
        },
        "claude-3-sonnet-20240229": {
            "input": 3.00,
            "output": 15.00,
        },
    },
    # MiniMax
    "minimax": {
        "MiniMax-M2.7": {
            "input": 0.30,   # $0.30 per million input
            "output": 1.20,  # $1.20 per million output
        },
        "MiniMax-M2.5": {
            "input": 0.30,
            "output": 1.20,
        },
        "MiniMax-M2.5-HighSpeed": {
            "input": 0.30,
            "output": 1.20,
        },
    },
    # Google Gemini
    "gemini": {
        "gemini-2.0-flash": {
            "input": 0.10,  # $0.10 per million input
            "output": 0.40,  # $0.40 per million output
        },
        "gemini-2.0-flash-001": {
            "input": 0.10,
            "output": 0.40,
        },
        "gemini-2.5-flash": {
            "input": 0.30,  # $0.30 per million input (≤200K context)
            "output": 2.50,  # $2.50 per million output
        },
        "gemini-1.5-pro": {
            "input": 1.25,
            "output": 5.00,
        },
        "gemini-1.5-flash": {
            "input": 0.075,
            "output": 0.30,
        },
        "gemini-1.5-flash-8b": {
            "input": 0.0375,
            "output": 0.15,
        },
    },
    # OpenAI (for reference)
    "openai": {
        "gpt-4o": {
            "input": 2.50,
            "output": 10.00,
        },
        "gpt-4o-mini": {
            "input": 0.15,
            "output": 0.60,
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00,
        },
    },
}

# Embedding Pricing (USD per million tokens)
EMBEDDING_PRICING = {
    # Google Gemini Embeddings
    "gemini-embedding": {
        "models/gemini-embedding-001": 0.10,  # $0.10 per million tokens
    },
    # Voyage AI
    "voyage": {
        "voyage-2": 0.12,
        "voyage-large-2": 0.12,
    },
    # OpenAI
    "openai": {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
    },
}


def calculate_llm_cost(
    provider: str,
    model: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None
) -> float:
    """
    Calculate cost for an LLM API call.

    Args:
        provider: Provider name (e.g., "anthropic", "gemini")
        model: Model name (e.g., "claude-3-5-sonnet-20241022")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens (used if input/output not available)

    Returns:
        Cost in USD
    """
    provider_lower = provider.lower()
    model_lower = model.lower()

    # Get pricing for provider
    provider_pricing = LLM_PRICING.get(provider_lower, {})

    # Try exact match first, then partial match
    pricing = provider_pricing.get(model)
    if not pricing:
        # Try partial match (e.g., "gemini-2.0-flash" matches "gemini-2.0-flash-001")
        for model_name, model_pricing in provider_pricing.items():
            if model_name in model_lower or model_lower in model_name:
                pricing = model_pricing
                break

    if not pricing:
        # Default fallback - return 0 to avoid breaking
        return 0.0

    # Calculate cost
    cost = 0.0

    if input_tokens is not None and "input" in pricing:
        cost += (input_tokens / 1_000_000) * pricing["input"]

    if output_tokens is not None and "output" in pricing:
        cost += (output_tokens / 1_000_000) * pricing["output"]

    # Fallback to total tokens if input/output not available
    if cost == 0.0 and total_tokens is not None:
        # Use average of input/output price
        avg_price = (pricing.get("input", 0) + pricing.get("output", 0)) / 2
        cost = (total_tokens / 1_000_000) * avg_price

    return cost


def calculate_embedding_cost(
    provider: str,
    model: str,
    token_count: int
) -> float:
    """
    Calculate cost for an embedding API call.

    Args:
        provider: Provider name (e.g., "gemini", "voyage")
        model: Model name (e.g., "voyage-2")
        token_count: Number of tokens processed

    Returns:
        Cost in USD
    """
    provider_lower = provider.lower()

    # Get pricing for provider
    provider_pricing = EMBEDDING_PRICING.get(provider_lower, {})

    # Try exact match first
    price_per_million = provider_pricing.get(model)
    if not price_per_million:
        # Try partial match
        for model_name, price in provider_pricing.items():
            if model_name in model or model in model_name:
                price_per_million = price
                break

    if not price_per_million:
        return 0.0

    return (token_count / 1_000_000) * price_per_million


@dataclass
class CostTracker:
    """Track cumulative API costs across the pipeline."""

    llm_cost: float = 0.0
    embedding_cost: float = 0.0
    llm_call_count: int = 0
    embedding_call_count: int = 0
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Per-provider breakdown
    cost_by_provider: Dict[str, float] = field(default_factory=dict)

    # Thread lock for safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add_llm_cost(
        self,
        cost: float,
        provider: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0
    ):
        """Add LLM cost to the tracker."""
        with self._lock:
            self.llm_cost += cost
            self.llm_call_count += 1
            self.total_tokens += total_tokens
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            # Update provider breakdown
            if provider not in self.cost_by_provider:
                self.cost_by_provider[provider] = 0.0
            self.cost_by_provider[provider] += cost

    def add_embedding_cost(
        self,
        cost: float,
        provider: str = "unknown"
    ):
        """Add embedding cost to the tracker."""
        with self._lock:
            self.embedding_cost += cost
            self.embedding_call_count += 1

            # Update provider breakdown
            if provider not in self.cost_by_provider:
                self.cost_by_provider[provider] = 0.0
            self.cost_by_provider[provider] += cost

    def get_total_cost(self) -> float:
        """Get total cost (LLM + embedding)."""
        return self.llm_cost + self.embedding_cost

    def get_summary(self) -> Dict:
        """Get cost summary as a dictionary."""
        with self._lock:
            return {
                "total_cost": self.get_total_cost(),
                "llm_cost": self.llm_cost,
                "embedding_cost": self.embedding_cost,
                "llm_call_count": self.llm_call_count,
                "embedding_call_count": self.embedding_call_count,
                "total_tokens": self.total_tokens,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "cost_by_provider": dict(self.cost_by_provider),
            }

    def reset(self):
        """Reset all costs."""
        with self._lock:
            self.llm_cost = 0.0
            self.embedding_cost = 0.0
            self.llm_call_count = 0
            self.embedding_call_count = 0
            self.total_tokens = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.cost_by_provider.clear()


# Global cost tracker instance
_global_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_cost_tracker
    if _global_cost_tracker is not None:
        _global_cost_tracker.reset()
