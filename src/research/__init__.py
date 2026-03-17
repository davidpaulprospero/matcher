# Research module - Multi-backend research using Perplexity, Gemini, and Grok

from .perplexity_client import PerplexityClient
from .gemini_client import GeminiClient
from .grok_client import GrokClient

# Re-export ResearchResult from perplexity (they're compatible)
from .perplexity_client import ResearchResult

__all__ = [
    "PerplexityClient",
    "GeminiClient",
    "GrokClient",
    "ResearchResult",
    "create_client",
    "research",
]


def create_client(backend: str = "perplexity"):
    """
    Create a research client for the specified backend.

    Args:
        backend: One of "perplexity", "gemini", "grok"

    Returns:
        Research client instance

    Raises:
        ValueError: If backend is unknown or API key is missing
    """
    backend = backend.lower()

    if backend == "perplexity":
        return PerplexityClient()
    elif backend == "gemini":
        return GeminiClient()
    elif backend == "grok":
        return GrokClient()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'perplexity', 'gemini', or 'grok'")


def research(query: str, backend: str = "perplexity", depth: str = "comprehensive"):
    """
    Quick research function using the specified backend.

    Args:
        query: Research query
        backend: One of "perplexity", "gemini", "grok"
        depth: "quick" or "comprehensive"

    Returns:
        ResearchResult with content and sources
    """
    client = create_client(backend)
    return client.research(query, depth=depth)
