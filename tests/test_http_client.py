"""
Tests for httpx client factory with proxy support.

Tests cover:
- Client creation with and without proxy
- Proxy URL extraction from handler
- ProxyAwareSession context manager
- Proxy rotation on rate limit
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.downloader.http_client import (
    get_proxy_for_httpx,
    get_socks5_proxy_for_httpx,
    create_httpx_client,
    create_async_httpx_client,
    ProxyAwareSession,
)


# =============================================================================
# get_proxy_for_httpx Tests
# =============================================================================


class TestGetProxyForHttpx:
    """Tests for get_proxy_for_httpx function."""

    def test_returns_none_when_handler_is_none(self):
        """Should return None when no handler provided."""
        result = get_proxy_for_httpx(None)
        assert result is None

    def test_returns_none_when_handler_has_no_proxy(self):
        """Should return None when handler.has_proxy is False."""
        handler = Mock()
        handler.has_proxy = False

        result = get_proxy_for_httpx(handler)

        assert result is None
        handler.get_proxy.assert_not_called()

    def test_returns_proxy_url_when_available(self):
        """Should return proxy URL from handler."""
        handler = Mock()
        handler.has_proxy = True
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        result = get_proxy_for_httpx(handler)

        assert result == "socks5://127.0.0.1:1080"
        handler.get_proxy.assert_called_once()

    def test_returns_none_when_get_proxy_returns_none(self):
        """Should return None if get_proxy returns None (proxy exhausted)."""
        handler = Mock()
        handler.has_proxy = True
        handler.get_proxy.return_value = None

        result = get_proxy_for_httpx(handler)

        assert result is None

    @pytest.mark.parametrize("proxy_url", [
        "http://proxy.example.com:8080",
        "https://user:pass@proxy.example.com:8080",
        "socks5://127.0.0.1:1080",
        "socks5://user:pass@socks.example.com:1080",
    ])
    def test_various_proxy_formats(self, proxy_url):
        """Should handle various proxy URL formats."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = proxy_url

        result = get_proxy_for_httpx(handler)

        assert result == proxy_url


# =============================================================================
# get_socks5_proxy_for_httpx Tests
# =============================================================================


class TestGetSocks5ProxyForHttpx:
    """Tests for get_socks5_proxy_for_httpx function."""

    def test_returns_none_when_handler_is_none(self):
        """Should return None when no handler provided."""
        result = get_socks5_proxy_for_httpx(None)
        assert result is None

    def test_prefers_socks5_proxy(self):
        """Should prefer SOCKS5 proxy from VPN if available."""
        handler = Mock()
        handler.get_socks5_proxy.return_value = "socks5://vpn.example.com:1080"
        handler.has_proxy = True
        handler.get_proxy.return_value = "http://regular.proxy.com:8080"

        result = get_socks5_proxy_for_httpx(handler)

        assert result == "socks5://vpn.example.com:1080"
        handler.get_socks5_proxy.assert_called_once()
        handler.get_proxy.assert_not_called()

    def test_falls_back_to_regular_proxy(self):
        """Should fall back to regular proxy when no SOCKS5."""
        handler = Mock()
        handler.get_socks5_proxy.return_value = None
        handler.has_proxy = True
        handler.get_proxy.return_value = "http://regular.proxy.com:8080"

        result = get_socks5_proxy_for_httpx(handler)

        assert result == "http://regular.proxy.com:8080"


# =============================================================================
# create_httpx_client Tests
# =============================================================================


class TestCreateHttpxClient:
    """Tests for create_httpx_client factory function."""

    def test_creates_client_without_proxy_when_no_handler(self):
        """Should create client without proxy when handler is None."""
        client = create_httpx_client(handler=None, timeout=30.0)

        assert client is not None
        # httpx.Client doesn't expose proxy directly, but we can verify it was created
        assert client.timeout.connect == 30.0
        client.close()

    def test_creates_client_without_proxy_when_handler_has_no_proxy(self):
        """Should create client without proxy when handler has no proxies."""
        handler = Mock(has_proxy=False)

        client = create_httpx_client(handler=handler, timeout=45.0)

        assert client is not None
        assert client.timeout.connect == 45.0
        handler.get_proxy.assert_not_called()
        client.close()

    def test_creates_client_with_proxy_when_available(self):
        """Should create client with proxy from handler."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        with patch('src.downloader.http_client.httpx.Client') as mock_client_cls:
            mock_client = Mock()
            mock_client_cls.return_value = mock_client

            result = create_httpx_client(handler=handler, timeout=30.0)

            mock_client_cls.assert_called_once_with(
                timeout=30.0,
                follow_redirects=True,
                proxy="socks5://127.0.0.1:1080",
            )

    def test_respects_follow_redirects_false(self):
        """Should respect follow_redirects=False."""
        with patch('src.downloader.http_client.httpx.Client') as mock_client_cls:
            create_httpx_client(handler=None, follow_redirects=False)

            mock_client_cls.assert_called_once()
            call_kwargs = mock_client_cls.call_args.kwargs
            assert call_kwargs["follow_redirects"] is False

    def test_passes_additional_kwargs(self):
        """Should pass additional kwargs to httpx.Client."""
        headers = {"User-Agent": "CustomAgent/1.0"}

        with patch('src.downloader.http_client.httpx.Client') as mock_client_cls:
            create_httpx_client(handler=None, headers=headers)

            mock_client_cls.assert_called_once()
            call_kwargs = mock_client_cls.call_args.kwargs
            assert call_kwargs["headers"] == headers


# =============================================================================
# create_async_httpx_client Tests
# =============================================================================


class TestCreateAsyncHttpxClient:
    """Tests for create_async_httpx_client factory function."""

    def test_creates_async_client_without_proxy(self):
        """Should create async client without proxy when no handler."""
        with patch('src.downloader.http_client.httpx.AsyncClient') as mock_cls:
            create_async_httpx_client(handler=None, timeout=30.0)

            mock_cls.assert_called_once_with(
                timeout=30.0,
                follow_redirects=True,
            )

    def test_creates_async_client_with_proxy(self):
        """Should create async client with proxy from handler."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        with patch('src.downloader.http_client.httpx.AsyncClient') as mock_cls:
            create_async_httpx_client(handler=handler, timeout=60.0)

            mock_cls.assert_called_once_with(
                timeout=60.0,
                follow_redirects=True,
                proxy="socks5://127.0.0.1:1080",
            )


# =============================================================================
# ProxyAwareSession Tests
# =============================================================================


class TestProxyAwareSession:
    """Tests for ProxyAwareSession context manager."""

    def test_context_manager_creates_and_closes_client(self):
        """Should create client on enter and close on exit."""
        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            with ProxyAwareSession(handler=None, timeout=30.0) as session:
                assert session._client is mock_client

            mock_client.close.assert_called_once()

    def test_get_request_delegates_to_client(self):
        """Should delegate GET request to underlying client."""
        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_client = Mock()
            mock_response = Mock(status_code=200)
            mock_client.get.return_value = mock_response
            mock_create.return_value = mock_client

            with ProxyAwareSession() as session:
                response = session.get("https://example.com", headers={"X-Test": "1"})

            mock_client.get.assert_called_once_with("https://example.com", headers={"X-Test": "1"})
            assert response == mock_response

    def test_post_request_delegates_to_client(self):
        """Should delegate POST request to underlying client."""
        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_client = Mock()
            mock_response = Mock(status_code=200)
            mock_client.post.return_value = mock_response
            mock_create.return_value = mock_client

            with ProxyAwareSession() as session:
                response = session.post("https://example.com", json={"key": "value"})

            mock_client.post.assert_called_once_with("https://example.com", json={"key": "value"})

    def test_rotate_proxy_calls_handler_on_rate_limit(self):
        """Should call handler.on_rate_limit when rotating."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            with ProxyAwareSession(handler=handler) as session:
                result = session.rotate_proxy()

            handler.on_rate_limit.assert_called_once_with("http_session")
            assert result is True

    def test_rotate_proxy_recreates_client(self):
        """Should close old client and create new one with rotated proxy."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.side_effect = ["proxy1", "proxy2"]

        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_client1 = Mock()
            mock_client2 = Mock()
            mock_create.side_effect = [mock_client1, mock_client2]

            with ProxyAwareSession(handler=handler) as session:
                assert session._client is mock_client1

                session.rotate_proxy()

                mock_client1.close.assert_called_once()
                assert session._client is mock_client2

    def test_rotate_proxy_returns_false_when_no_handler(self):
        """Should return False when no handler available."""
        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_create.return_value = Mock()

            with ProxyAwareSession(handler=None) as session:
                result = session.rotate_proxy()

            assert result is False

    def test_on_success_calls_handler(self):
        """Should call handler.on_success when reported."""
        handler = Mock(has_proxy=True)
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_create.return_value = Mock()

            with ProxyAwareSession(handler=handler) as session:
                session.on_success()

            handler.on_success.assert_called_once()

    def test_on_success_does_nothing_when_no_handler(self):
        """Should not raise when no handler and on_success called."""
        with patch('src.downloader.http_client.create_httpx_client') as mock_create:
            mock_create.return_value = Mock()

            with ProxyAwareSession(handler=None) as session:
                # Should not raise
                session.on_success()

    def test_raises_when_accessing_client_outside_context(self):
        """Should raise RuntimeError when accessing client outside context."""
        session = ProxyAwareSession()

        with pytest.raises(RuntimeError, match="Session not entered"):
            _ = session.client


# =============================================================================
# Integration Tests
# =============================================================================


class TestHttpClientIntegration:
    """Integration tests for http_client module."""

    def test_client_factory_with_mock_handler(self):
        """End-to-end test with mocked handler - verifies configuration flow."""
        handler = Mock()
        handler.has_proxy = True
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"
        handler.on_rate_limit = Mock()
        handler.on_success = Mock()

        # Mock httpx.Client to avoid needing socksio package
        with patch('src.downloader.http_client.httpx.Client') as mock_client_cls:
            mock_client = Mock()
            mock_client_cls.return_value = mock_client

            client = create_httpx_client(handler=handler, timeout=30.0)

            # Verify handler was consulted
            handler.get_proxy.assert_called_once()

            # Verify client was created with proxy
            mock_client_cls.assert_called_once_with(
                timeout=30.0,
                follow_redirects=True,
                proxy="socks5://127.0.0.1:1080",
            )

    def test_session_full_workflow(self):
        """Test full session workflow: request, 429, rotate, retry."""
        handler = Mock()
        handler.has_proxy = True
        handler.get_proxy.side_effect = ["proxy1", "proxy2", "proxy3"]

        with patch('src.downloader.http_client.httpx.Client') as mock_client_cls:
            # Create mock clients for each creation
            mock_clients = [Mock() for _ in range(3)]
            mock_client_cls.side_effect = mock_clients

            # First request returns 429
            mock_clients[0].get.return_value = Mock(status_code=429)
            # Second request (after rotation) returns 200
            mock_clients[1].get.return_value = Mock(status_code=200)

            with ProxyAwareSession(handler=handler) as session:
                # First request
                response1 = session.get("https://api.example.com")
                assert response1.status_code == 429

                # Rotate proxy
                session.rotate_proxy()

                # Retry
                response2 = session.get("https://api.example.com")
                assert response2.status_code == 200

                # Report success
                session.on_success()

            # Verify workflow
            handler.on_rate_limit.assert_called_once_with("http_session")
            handler.on_success.assert_called_once()
