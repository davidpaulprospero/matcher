"""
Custom exceptions for LLM client operations.
"""


class LLMClientError(Exception):
    """Base exception for all LLM client errors."""
    pass


class LLMTimeoutError(LLMClientError):
    """Raised when an LLM request times out."""
    pass


class LLMParseError(LLMClientError):
    """Raised when LLM response cannot be parsed."""
    pass


class LLMProviderError(LLMClientError):
    """Raised when there's an error with the LLM provider (API error, authentication, etc.)."""
    pass
