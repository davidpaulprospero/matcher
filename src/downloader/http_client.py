"""
Shared HTTP client factory with proxy support.

Provides a unified way to create httpx clients that automatically use
proxy configuration from the RateLimitHandler.

Usage:
    from src.downloader.http_client import create_httpx_client, get_proxy_for_httpx

    # Create a client with proxy support
    client = create_httpx_client(handler=rate_limit_handler, timeout=30.0)
    response = client.get(url)

    # Or get proxy for one-off requests
    proxy = get_proxy_for_httpx(handler)
    response = httpx.get(url, proxy=proxy)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from .rate_limit_handler import RateLimitHandler

logger = logging.getLogger(__name__)


def get_proxy_for_httpx(handler: Optional["RateLimitHandler"]) -> Optional[str]:
    """
    Get proxy URL formatted for httpx.

    Args:
        handler: RateLimitHandler instance (may be None)

    Returns:
        Proxy URL string (e.g., "socks5://127.0.0.1:1080") or None if no proxy available
    """
    if handler is None:
        return None

    if not handler.has_proxy:
        return None

    proxy_url = handler.get_proxy()
    if proxy_url:
        logger.debug(f"Using proxy: {proxy_url[:30]}...")
    return proxy_url


def get_socks5_proxy_for_httpx(handler: Optional["RateLimitHandler"]) -> Optional[str]:
    """
    Get SOCKS5 proxy URL (preferring VPN if available).

    Args:
        handler: RateLimitHandler instance (may be None)

    Returns:
        SOCKS5 proxy URL or None
    """
    if handler is None:
        return None

    # Try SOCKS5 first (VPN), fall back to regular proxy
    socks5 = handler.get_socks5_proxy()
    if socks5:
        logger.debug(f"Using SOCKS5 proxy: {socks5[:30]}...")
        return socks5

    return get_proxy_for_httpx(handler)


def create_httpx_client(
    handler: Optional["RateLimitHandler"] = None,
    timeout: float = 30.0,
    follow_redirects: bool = True,
    **kwargs,
) -> httpx.Client:
    """
    Create httpx client with proxy from RateLimitHandler if available.

    This factory ensures all HTTP clients use the same proxy configuration
    from the centralized rate limit handler.

    Args:
        handler: RateLimitHandler instance for proxy configuration
        timeout: Request timeout in seconds (default: 30.0)
        follow_redirects: Whether to follow redirects (default: True)
        **kwargs: Additional arguments passed to httpx.Client

    Returns:
        Configured httpx.Client instance

    Example:
        handler = get_rate_limit_handler(config)
        client = create_httpx_client(handler, timeout=60.0)
        response = client.get("https://example.com")
    """
    proxy = get_proxy_for_httpx(handler)

    # Build client kwargs
    client_kwargs = {
        "timeout": timeout,
        "follow_redirects": follow_redirects,
        **kwargs,
    }

    # Add proxy if available
    if proxy:
        client_kwargs["proxy"] = proxy
        logger.debug(f"Created httpx client with proxy")
    else:
        logger.debug("Created httpx client without proxy")

    return httpx.Client(**client_kwargs)


def create_async_httpx_client(
    handler: Optional["RateLimitHandler"] = None,
    timeout: float = 30.0,
    follow_redirects: bool = True,
    **kwargs,
) -> httpx.AsyncClient:
    """
    Create async httpx client with proxy from RateLimitHandler if available.

    Args:
        handler: RateLimitHandler instance for proxy configuration
        timeout: Request timeout in seconds (default: 30.0)
        follow_redirects: Whether to follow redirects (default: True)
        **kwargs: Additional arguments passed to httpx.AsyncClient

    Returns:
        Configured httpx.AsyncClient instance
    """
    proxy = get_proxy_for_httpx(handler)

    client_kwargs = {
        "timeout": timeout,
        "follow_redirects": follow_redirects,
        **kwargs,
    }

    if proxy:
        client_kwargs["proxy"] = proxy
        logger.debug("Created async httpx client with proxy")
    else:
        logger.debug("Created async httpx client without proxy")

    return httpx.AsyncClient(**client_kwargs)


class ProxyAwareSession:
    """
    Context manager for httpx client that updates proxy on rate limit.

    Provides a session that automatically rotates proxy when rate limited.

    Usage:
        with ProxyAwareSession(handler) as session:
            response = session.get(url)
            if response.status_code == 429:
                session.rotate_proxy()
                response = session.get(url)  # Retry with new proxy
    """

    def __init__(
        self,
        handler: Optional["RateLimitHandler"] = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        self.handler = handler
        self.timeout = timeout
        self.kwargs = kwargs
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "ProxyAwareSession":
        self._client = create_httpx_client(
            handler=self.handler,
            timeout=self.timeout,
            **self.kwargs,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        """Get the underlying httpx client."""
        if self._client is None:
            raise RuntimeError("Session not entered. Use 'with ProxyAwareSession() as session:'")
        return self._client

    def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return self.client.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return self.client.post(url, **kwargs)

    def rotate_proxy(self) -> bool:
        """
        Rotate to a new proxy after rate limit.

        Closes current client and creates new one with rotated proxy.

        Returns:
            True if proxy was rotated, False if no handler or no proxies
        """
        if not self.handler:
            return False

        # Trigger rotation in handler
        self.handler.on_rate_limit("http_session")

        # Close old client
        if self._client:
            self._client.close()

        # Create new client with (potentially) rotated proxy
        self._client = create_httpx_client(
            handler=self.handler,
            timeout=self.timeout,
            **self.kwargs,
        )

        return self.handler.has_proxy

    def on_success(self) -> None:
        """Report successful request to handler."""
        if self.handler:
            self.handler.on_success()
