"""Comprehensive tests for proxy_manager.py - proxy rotation for rate limit bypass.

Tests cover:
- Happy path (normal usage)
- Edge cases (empty, null, boundary values, unicode)
- Error cases (invalid input, failures, exceptions)
- Branch coverage for all conditional paths
"""

import os
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import httpx

from src.downloader.proxy_manager import (
    ProxyType,
    ProxyStatus,
    ProxyInfo,
    ProxyPool,
    ProxyManager,
    ProxiedHttpClient,
    load_proxies_from_env,
    check_proxy,
)


# ============================================================================
# ProxyInfo Tests
# ============================================================================

class TestProxyInfo:
    """Tests for ProxyInfo dataclass."""

    # --- Happy Path Tests ---

    @pytest.mark.parametrize("status,expected", [
        (ProxyStatus.HEALTHY, True),
        (ProxyStatus.UNKNOWN, True),
        (ProxyStatus.COOLDOWN, False),
        (ProxyStatus.DEAD, False),
    ])
    def test_is_available_all_statuses(self, status, expected):
        """Should correctly report availability for all status types."""
        proxy = ProxyInfo(url="http://proxy:8080", status=status)
        assert proxy.is_available is expected

    @pytest.mark.parametrize("success,failure,expected_rate", [
        (0, 0, 0.0),      # No requests
        (10, 0, 100.0),   # All success
        (0, 10, 0.0),     # All failures
        (8, 2, 80.0),     # 80% success
        (1, 3, 25.0),     # 25% success
        (50, 50, 50.0),   # 50% success
    ])
    def test_success_rate_calculation(self, success, failure, expected_rate):
        """Success rate should be calculated correctly for various counts."""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=success, failure_count=failure)
        assert proxy.success_rate == expected_rate

    def test_to_httpx_format(self):
        """Should return URL for httpx."""
        proxy = ProxyInfo(url="http://proxy:8080")
        assert proxy.to_httpx_format() == "http://proxy:8080"

    def test_to_httpx_format_socks5(self):
        """Should return SOCKS5 URL unchanged."""
        proxy = ProxyInfo(url="socks5://user:pass@proxy:1080")
        assert proxy.to_httpx_format() == "socks5://user:pass@proxy:1080"

    # --- Edge Cases ---

    def test_str_masks_password(self):
        """String representation should mask password."""
        proxy = ProxyInfo(url="http://user:secret@proxy:8080")
        result = str(proxy)
        assert "secret" not in result
        assert "****" in result or "user" in result

    def test_str_no_password(self):
        """String representation without password."""
        proxy = ProxyInfo(url="http://proxy:8080")
        result = str(proxy)
        assert "proxy:8080" in result

    def test_proxy_types(self):
        """Should detect proxy types correctly."""
        http_proxy = ProxyInfo(url="http://proxy:8080", proxy_type=ProxyType.HTTP)
        assert http_proxy.proxy_type == ProxyType.HTTP

        socks_proxy = ProxyInfo(url="socks5://proxy:1080", proxy_type=ProxyType.SOCKS5)
        assert socks_proxy.proxy_type == ProxyType.SOCKS5

    def test_all_proxy_types(self):
        """Should support all proxy types."""
        types = [ProxyType.HTTP, ProxyType.HTTPS, ProxyType.SOCKS4, ProxyType.SOCKS5]
        for proxy_type in types:
            proxy = ProxyInfo(url=f"{proxy_type.value}://proxy:8080", proxy_type=proxy_type)
            assert proxy.proxy_type == proxy_type

    def test_masked_url_special_characters_in_password(self):
        """Should handle special characters in password for masking."""
        proxy = ProxyInfo(url="http://user:p@ss%3Dword!@proxy:8080")
        # Should not raise an error
        result = str(proxy)
        assert proxy is not None

    # --- Boundary Tests ---

    def test_success_rate_large_numbers(self):
        """Success rate with very large counts."""
        proxy = ProxyInfo(url="http://proxy:8080", success_count=1000000, failure_count=1)
        assert proxy.success_rate > 99.99

    def test_proxy_info_defaults(self):
        """Test default values."""
        proxy = ProxyInfo(url="http://proxy:8080")
        assert proxy.status == ProxyStatus.UNKNOWN
        assert proxy.last_used == 0.0
        assert proxy.last_failure == 0.0
        assert proxy.failure_count == 0
        assert proxy.success_count == 0
        assert proxy.avg_latency_ms == 0.0
        assert proxy.source == "manual"


# ============================================================================
# ProxyPool Tests
# ============================================================================

class TestProxyPool:
    """Tests for ProxyPool class."""

    # --- Happy Path Tests ---

    def test_add_proxy(self):
        """Should add proxy to pool."""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy1:8080")
        pool.add(proxy)
        assert len(pool.proxies) == 1

    def test_add_multiple_proxies(self):
        """Should add multiple unique proxies."""
        pool = ProxyPool()
        for i in range(5):
            pool.add(ProxyInfo(url=f"http://proxy{i}:8080"))
        assert len(pool.proxies) == 5

    def test_add_duplicate_ignored(self):
        """Should not add duplicate proxies."""
        pool = ProxyPool()
        proxy1 = ProxyInfo(url="http://proxy1:8080")
        proxy2 = ProxyInfo(url="http://proxy1:8080")  # Same URL
        pool.add(proxy1)
        pool.add(proxy2)
        assert len(pool.proxies) == 1

    def test_get_next_round_robin(self):
        """Should rotate through proxies."""
        pool = ProxyPool()
        pool.add(ProxyInfo(url="http://proxy1:8080"))
        pool.add(ProxyInfo(url="http://proxy2:8080"))
        pool.add(ProxyInfo(url="http://proxy3:8080"))

        # Should rotate
        assert pool.get_next().url == "http://proxy1:8080"
        assert pool.get_next().url == "http://proxy2:8080"
        assert pool.get_next().url == "http://proxy3:8080"
        assert pool.get_next().url == "http://proxy1:8080"

    def test_get_next_skips_cooldown(self):
        """Should skip proxies on cooldown."""
        pool = ProxyPool(cooldown_seconds=300)
        proxy1 = ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.COOLDOWN, last_failure=time.time())
        proxy2 = ProxyInfo(url="http://proxy2:8080", status=ProxyStatus.HEALTHY)
        pool.add(proxy1)
        pool.add(proxy2)

        # Should skip proxy1 and return proxy2
        assert pool.get_next().url == "http://proxy2:8080"

    def test_get_next_skips_dead(self):
        """Should skip dead proxies."""
        pool = ProxyPool()
        dead = ProxyInfo(url="http://dead:8080", status=ProxyStatus.DEAD)
        alive = ProxyInfo(url="http://alive:8080", status=ProxyStatus.HEALTHY)
        pool.add(dead)
        pool.add(alive)

        # Should skip dead and return alive
        result = pool.get_next()
        assert result.url == "http://alive:8080"

    def test_get_next_none_available(self):
        """Should return None when no proxies available."""
        pool = ProxyPool()
        assert pool.get_next() is None

    def test_get_random(self):
        """Should return random available proxy."""
        pool = ProxyPool()
        pool.add(ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.HEALTHY))
        pool.add(ProxyInfo(url="http://proxy2:8080", status=ProxyStatus.HEALTHY))

        # Should return one of the proxies
        proxy = pool.get_random()
        assert proxy is not None
        assert proxy.url in ["http://proxy1:8080", "http://proxy2:8080"]

    def test_get_random_skips_unavailable(self):
        """get_random should only return available proxies."""
        pool = ProxyPool()
        pool.add(ProxyInfo(url="http://dead:8080", status=ProxyStatus.DEAD))
        pool.add(ProxyInfo(url="http://cooldown:8080", status=ProxyStatus.COOLDOWN))
        pool.add(ProxyInfo(url="http://healthy:8080", status=ProxyStatus.HEALTHY))

        for _ in range(10):
            result = pool.get_random()
            assert result.url == "http://healthy:8080"

    def test_get_random_empty(self):
        """get_random should return None when no available proxies."""
        pool = ProxyPool()
        pool.add(ProxyInfo(url="http://dead:8080", status=ProxyStatus.DEAD))
        assert pool.get_random() is None

    def test_mark_success(self):
        """Should update proxy on success."""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.UNKNOWN)
        pool.add(proxy)

        pool.mark_success(proxy, latency_ms=100)

        assert proxy.status == ProxyStatus.HEALTHY
        assert proxy.success_count == 1
        assert proxy.avg_latency_ms == 100
        assert proxy.failure_count == 0

    def test_mark_success_updates_latency(self):
        """Should calculate running average latency."""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.HEALTHY)
        proxy.avg_latency_ms = 100
        pool.add(proxy)

        pool.mark_success(proxy, latency_ms=200)
        assert proxy.avg_latency_ms == 150  # (100 + 200) / 2

    def test_mark_success_no_latency(self):
        """Should not update latency when not provided."""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy1:8080")
        pool.add(proxy)

        pool.mark_success(proxy, latency_ms=0)
        assert proxy.avg_latency_ms == 0.0

    def test_mark_failure(self):
        """Should update proxy on failure."""
        pool = ProxyPool(max_failures=3)
        proxy = ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.HEALTHY)
        pool.add(proxy)

        pool.mark_failure(proxy)
        assert proxy.failure_count == 1
        assert proxy.status == ProxyStatus.HEALTHY  # Not cooldown yet

        pool.mark_failure(proxy)
        pool.mark_failure(proxy)
        assert proxy.status == ProxyStatus.COOLDOWN  # Now on cooldown

    def test_mark_failure_rate_limit(self):
        """Rate limit should immediately put proxy on cooldown."""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.HEALTHY)
        pool.add(proxy)

        pool.mark_failure(proxy, is_rate_limit=True)
        assert proxy.status == ProxyStatus.COOLDOWN

    def test_mark_failure_eventually_dies(self):
        """Proxy should die after too many failures."""
        pool = ProxyPool(max_failures=2)
        proxy = ProxyInfo(url="http://proxy1:8080")
        pool.add(proxy)

        # 2*3=6 failures should mark it dead
        for _ in range(6):
            pool.mark_failure(proxy)

        assert proxy.status == ProxyStatus.DEAD

    def test_get_healthy_count(self):
        """Should count healthy/available proxies."""
        pool = ProxyPool()
        pool.add(ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.HEALTHY))
        pool.add(ProxyInfo(url="http://proxy2:8080", status=ProxyStatus.COOLDOWN))
        pool.add(ProxyInfo(url="http://proxy3:8080", status=ProxyStatus.UNKNOWN))
        pool.add(ProxyInfo(url="http://proxy4:8080", status=ProxyStatus.DEAD))

        assert pool.get_healthy_count() == 2  # HEALTHY + UNKNOWN

    def test_reset_all(self):
        """Should reset all proxy states."""
        pool = ProxyPool()
        pool.add(ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.COOLDOWN, failure_count=5))
        pool.add(ProxyInfo(url="http://proxy2:8080", status=ProxyStatus.DEAD, failure_count=10))

        pool.reset_all()

        for proxy in pool.proxies:
            assert proxy.status == ProxyStatus.UNKNOWN
            assert proxy.failure_count == 0

    # --- Edge Cases ---

    def test_cooldown_expires(self):
        """Proxy should be available after cooldown expires."""
        pool = ProxyPool(cooldown_seconds=0.1)  # Very short cooldown
        proxy = ProxyInfo(
            url="http://proxy1:8080",
            status=ProxyStatus.COOLDOWN,
            last_failure=time.time() - 1  # 1 second ago
        )
        pool.add(proxy)

        # Should now be available
        result = pool.get_next()
        assert result is not None
        assert result.status == ProxyStatus.UNKNOWN  # Reset from cooldown
        assert result.failure_count == 0

    def test_get_next_all_cooldown(self):
        """Should return None when all proxies on cooldown."""
        pool = ProxyPool(cooldown_seconds=300)
        now = time.time()
        pool.add(ProxyInfo(url="http://proxy1:8080", status=ProxyStatus.COOLDOWN, last_failure=now))
        pool.add(ProxyInfo(url="http://proxy2:8080", status=ProxyStatus.COOLDOWN, last_failure=now))

        assert pool.get_next() is None

    def test_get_next_updates_last_used(self):
        """get_next should update last_used timestamp."""
        pool = ProxyPool()
        proxy = ProxyInfo(url="http://proxy1:8080")
        pool.add(proxy)

        before = time.time()
        result = pool.get_next()
        after = time.time()

        assert before <= result.last_used <= after


# ============================================================================
# ProxyManager Tests
# ============================================================================

class TestProxyManager:
    """Tests for ProxyManager class."""

    # --- Happy Path Tests ---

    def test_init_no_config(self):
        """Should initialize without config."""
        manager = ProxyManager()
        assert manager.has_proxies is False

    def test_add_proxy(self):
        """Should add proxy manually."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")
        assert manager.has_proxies is True
        assert manager.available_count == 1

    def test_add_proxy_auto_scheme(self):
        """Should add scheme when missing."""
        manager = ProxyManager()
        manager.add_proxy("proxy:8080")  # No scheme
        proxy = manager.get_proxy()
        assert proxy.startswith("http://")

    def test_add_socks5(self):
        """Should add SOCKS5 proxy."""
        manager = ProxyManager()
        manager.add_socks5("127.0.0.1", 1080)
        assert manager.has_proxies is True

        proxy = manager.get_proxy()
        assert "socks5://" in proxy
        assert "127.0.0.1:1080" in proxy

    def test_add_socks5_with_auth(self):
        """Should add SOCKS5 proxy with authentication."""
        manager = ProxyManager()
        manager.add_socks5("proxy.example.com", 1080, "user", "pass")

        proxy = manager.get_proxy()
        assert proxy == "socks5://user:pass@proxy.example.com:1080"

    def test_get_proxy_rotates(self):
        """Should rotate through proxies."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")
        manager.add_proxy("http://proxy2:8080")

        proxy1 = manager.get_proxy()
        proxy2 = manager.get_proxy()

        assert proxy1 != proxy2

    def test_get_random_proxy(self):
        """Should return random proxy."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")
        manager.add_proxy("http://proxy2:8080")

        proxy = manager.get_random_proxy()
        assert proxy in ["http://proxy1:8080", "http://proxy2:8080"]

    def test_report_success(self):
        """Should mark proxy as successful."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")

        manager.get_proxy()  # Sets current proxy
        manager.report_success(latency_ms=50)

        stats = manager.get_stats()
        assert stats["healthy"] == 1

    def test_report_rate_limit(self):
        """Should put proxy on cooldown after rate limit."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")
        manager.add_proxy("http://proxy2:8080")

        manager.get_proxy()  # Get proxy1
        manager.report_rate_limit()  # Rate limit proxy1

        # Should now get proxy2
        proxy = manager.get_proxy()
        assert "proxy2" in proxy

    def test_get_stats(self):
        """Should return pool statistics."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")
        manager.add_proxy("http://proxy2:8080")

        stats = manager.get_stats()
        assert stats["total"] == 2
        assert stats["available"] == 2
        assert stats["healthy"] == 0
        assert stats["unknown"] == 2

    def test_reset(self):
        """Should reset all proxy states."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")
        manager.get_proxy()
        manager.report_rate_limit()

        manager.reset()
        stats = manager.get_stats()
        assert stats["cooldown"] == 0

    # --- Edge Cases ---

    def test_get_proxy_disabled(self):
        """Should return None when disabled."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")
        manager._enabled = False

        assert manager.get_proxy() is None

    def test_get_proxy_empty_pool(self):
        """Should return None with no proxies."""
        manager = ProxyManager()
        assert manager.get_proxy() is None

    def test_get_random_proxy_disabled(self):
        """get_random_proxy returns None when disabled."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")
        manager._enabled = False

        assert manager.get_random_proxy() is None

    def test_report_success_no_current(self):
        """report_success should not fail without current proxy."""
        manager = ProxyManager()
        manager.report_success(100)  # Should not raise

    def test_report_failure_no_current(self):
        """report_failure should not fail without current proxy."""
        manager = ProxyManager()
        manager.report_failure()  # Should not raise

    @pytest.mark.parametrize("url,expected_type", [
        ("http://proxy:8080", ProxyType.HTTP),
        ("https://proxy:8080", ProxyType.HTTPS),
        ("socks5://proxy:1080", ProxyType.SOCKS5),
        ("socks4://proxy:1080", ProxyType.SOCKS4),
        ("proxy:8080", ProxyType.HTTP),  # Default
    ])
    def test_add_proxy_url_types(self, url, expected_type):
        """Should detect proxy type from URL."""
        manager = ProxyManager()
        manager._add_proxy_url(url, "test")
        assert manager.pool.proxies[0].proxy_type == expected_type

    def test_add_proxy_url_empty(self):
        """Should skip empty URLs."""
        manager = ProxyManager()
        manager._add_proxy_url("", "test")
        manager._add_proxy_url("   ", "test")
        assert len(manager.pool.proxies) == 0

    # --- Config Loading Tests ---

    def test_load_from_config_dict(self):
        """Should load proxies from config dict."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {
            'enabled': True,
            'sources': [
                {'type': 'url', 'url': 'http://config-proxy:8080'},
            ]
        }

        manager = ProxyManager(mock_config)
        assert manager.has_proxies is True

    def test_load_from_config_object(self):
        """Should load proxies from config object."""
        mock_config = Mock()
        mock_proxy = Mock()
        mock_proxy.enabled = True
        mock_proxy.sources = [{'type': 'url', 'url': 'http://config-proxy:8080'}]
        mock_config.download.fallback.proxy = mock_proxy

        manager = ProxyManager(mock_config)
        assert manager.has_proxies is True

    def test_disabled_in_config(self):
        """Should not load proxies when disabled."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {
            'enabled': False,
            'sources': [
                {'type': 'url', 'url': 'http://config-proxy:8080'},
            ]
        }

        manager = ProxyManager(mock_config)
        assert manager.has_proxies is False

    def test_load_from_config_no_fallback(self):
        """Should handle missing fallback config."""
        mock_config = Mock()
        mock_config.download.fallback = None

        manager = ProxyManager(mock_config)
        assert manager.has_proxies is False

    def test_load_from_config_no_proxy(self):
        """Should handle missing proxy config."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = None

        manager = ProxyManager(mock_config)
        assert manager.has_proxies is False

    def test_load_from_config_error(self):
        """Should handle config loading errors gracefully."""
        mock_config = Mock()
        mock_config.download.fallback = Mock(side_effect=Exception("Config error"))

        # Should not raise
        manager = ProxyManager(mock_config)

    def test_load_source_list_type(self):
        """Should load from file source type."""
        manager = ProxyManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("http://file-proxy1:8080\n")
            f.write("# Comment line\n")
            f.write("http://file-proxy2:8080\n")
            f.write("\n")  # Empty line
            temp_path = f.name

        try:
            manager._load_source({'type': 'list', 'file': temp_path})
            assert len(manager.pool.proxies) == 2
        finally:
            os.unlink(temp_path)

    def test_load_from_file_not_found(self):
        """Should handle missing proxy file."""
        manager = ProxyManager()
        manager._load_from_file(Path("/nonexistent/proxy_list.txt"))
        assert len(manager.pool.proxies) == 0

    def test_load_source_env_type(self):
        """Should load from environment variable."""
        manager = ProxyManager()

        with patch.dict(os.environ, {'TEST_PROXY': 'http://env-proxy:8080'}):
            manager._load_source({'type': 'env', 'env': 'TEST_PROXY'})
            assert len(manager.pool.proxies) == 1
            assert manager.pool.proxies[0].url == "http://env-proxy:8080"

    def test_load_source_socks5_type(self):
        """Should load SOCKS5 source type."""
        manager = ProxyManager()
        manager._load_source({'type': 'socks5', 'url': 'socks5://proxy:1080'})
        assert len(manager.pool.proxies) == 1

    def test_load_source_string(self):
        """Should handle direct string sources."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {
            'enabled': True,
            'sources': ['http://direct-proxy:8080']  # Direct string, not dict
        }

        manager = ProxyManager(mock_config)
        assert manager.has_proxies is True


# ============================================================================
# ProxiedHttpClient Tests
# ============================================================================

class TestProxiedHttpClient:
    """Tests for ProxiedHttpClient wrapper."""

    def test_init(self):
        """Should initialize with manager."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")

        client = ProxiedHttpClient(manager, max_retries=3)
        assert client.max_retries == 3
        assert client.timeout == 30.0

    def test_get_success(self):
        """Should make successful GET request."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")

        client = ProxiedHttpClient(manager, max_retries=1)

        with patch('httpx.Client') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200

            mock_client_instance = MagicMock()
            mock_client_instance.request.return_value = mock_response
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            response = client.get("http://example.com")
            assert response.status_code == 200

    def test_post_success(self):
        """Should make successful POST request."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")

        client = ProxiedHttpClient(manager, max_retries=1)

        with patch('httpx.Client') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 201

            mock_client_instance = MagicMock()
            mock_client_instance.request.return_value = mock_response
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            response = client.post("http://example.com", json={"data": "test"})
            assert response.status_code == 201

    def test_retries_on_rate_limit(self):
        """Should retry with different proxy on rate limit."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy1:8080")
        manager.add_proxy("http://proxy2:8080")

        client = ProxiedHttpClient(manager, max_retries=3)

        with patch('httpx.Client') as mock_client_class:
            call_count = 0

            def create_response(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                response = Mock()
                response.status_code = 429 if call_count < 3 else 200
                response.request = Mock()
                return response

            mock_client_instance = MagicMock()
            mock_client_instance.request.side_effect = create_response
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            response = client.get("http://example.com")
            assert response.status_code == 200
            assert call_count == 3

    def test_retries_on_timeout(self):
        """Should retry on timeout."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")

        client = ProxiedHttpClient(manager, max_retries=2)

        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.request.side_effect = httpx.TimeoutException("Timeout")
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            with pytest.raises(httpx.TimeoutException):
                client.get("http://example.com")

    def test_retries_on_proxy_error(self):
        """Should retry on proxy error."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")

        client = ProxiedHttpClient(manager, max_retries=2)

        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.request.side_effect = httpx.ProxyError("Proxy failed")
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            with pytest.raises(httpx.ProxyError):
                client.get("http://example.com")

    def test_all_retries_fail(self):
        """Should raise after all retries fail."""
        manager = ProxyManager()
        manager.add_proxy("http://proxy:8080")

        client = ProxiedHttpClient(manager, max_retries=2)

        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.request.side_effect = Exception("Network error")
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            with pytest.raises(Exception, match="Network error"):
                client.get("http://example.com")


# ============================================================================
# Utility Functions Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_load_proxies_from_env(self):
        """Should load proxies from environment variables."""
        with patch.dict('os.environ', {'HTTP_PROXY': 'http://env-proxy:8080'}):
            proxies = load_proxies_from_env()
            assert 'http://env-proxy:8080' in proxies

    def test_load_proxies_from_env_multiple(self):
        """Should load from multiple env vars."""
        with patch.dict('os.environ', {
            'HTTP_PROXY': 'http://http-proxy:8080',
            'HTTPS_PROXY': 'http://https-proxy:8080',
        }):
            proxies = load_proxies_from_env()
            assert 'http://http-proxy:8080' in proxies
            assert 'http://https-proxy:8080' in proxies

    def test_load_proxies_from_env_lowercase(self):
        """Should check lowercase env vars too."""
        with patch.dict('os.environ', {'http_proxy': 'http://lower-proxy:8080'}, clear=True):
            proxies = load_proxies_from_env()
            assert 'http://lower-proxy:8080' in proxies

    def test_load_proxies_from_env_empty(self):
        """Should return empty list when no env vars set."""
        with patch.dict('os.environ', {}, clear=True):
            proxies = load_proxies_from_env()
            assert isinstance(proxies, list)

    def test_check_proxy_success(self):
        """Should return success for working proxy."""
        with patch('httpx.Client') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 200

            mock_client_instance = MagicMock()
            mock_client_instance.get.return_value = mock_response
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            success, latency, message = check_proxy("http://good-proxy:8080")
            assert success is True
            assert "OK" in message

    def test_check_proxy_http_error(self):
        """Should return failure for non-200 response."""
        with patch('httpx.Client') as mock_client_class:
            mock_response = Mock()
            mock_response.status_code = 403

            mock_client_instance = MagicMock()
            mock_client_instance.get.return_value = mock_response
            mock_client_instance.__enter__ = Mock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = Mock(return_value=False)

            mock_client_class.return_value = mock_client_instance

            success, latency, message = check_proxy("http://bad-proxy:8080")
            assert success is False
            assert "403" in message

    def test_check_proxy_connection_error(self):
        """Should return failure for connection error."""
        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = Mock(side_effect=Exception("Connection refused"))
            mock_client_class.return_value = mock_client_instance

            success, latency, message = check_proxy("http://invalid:9999")
            assert success is False
            assert "Error" in message

    def test_check_proxy_timeout(self):
        """Should return failure for timeout."""
        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = Mock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_class.return_value = mock_client_instance

            success, latency, message = check_proxy("http://slow-proxy:8080")
            assert success is False
            assert "Timeout" in message

    def test_check_proxy_proxy_error(self):
        """Should return failure for proxy error."""
        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = MagicMock()
            mock_client_instance.__enter__ = Mock(side_effect=httpx.ProxyError("Proxy failed"))
            mock_client_class.return_value = mock_client_instance

            success, latency, message = check_proxy("http://bad-proxy:8080")
            assert success is False
            assert "Proxy error" in message


# ============================================================================
# Unicode and Special Characters Tests
# ============================================================================

class TestUnicodeAndSpecialCharacters:
    """Tests for unicode and special character handling."""

    def test_proxy_url_with_unicode_password(self):
        """Should handle unicode in password."""
        proxy = ProxyInfo(url="http://user:密码@proxy:8080")
        # Should not raise
        result = str(proxy)
        assert proxy.to_httpx_format() == "http://user:密码@proxy:8080"

    def test_proxy_url_with_special_chars(self):
        """Should handle special characters in URL."""
        manager = ProxyManager()
        manager.add_proxy("http://user:p%40ss@proxy:8080")  # @ encoded as %40
        assert manager.has_proxies is True

    def test_proxy_file_with_unicode(self):
        """Should handle unicode in proxy file."""
        manager = ProxyManager()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("http://user:пароль@proxy:8080\n")  # Russian "password"
            temp_path = f.name

        try:
            manager._load_from_file(Path(temp_path))
            assert len(manager.pool.proxies) == 1
        finally:
            os.unlink(temp_path)


# ============================================================================
# Concurrency Edge Cases
# ============================================================================

class TestConcurrencyEdgeCases:
    """Tests for potential concurrency issues."""

    def test_rapid_get_proxy_calls(self):
        """Should handle rapid sequential get_proxy calls."""
        manager = ProxyManager()
        for i in range(10):
            manager.add_proxy(f"http://proxy{i}:8080")

        # Rapid calls should rotate correctly
        seen = set()
        for _ in range(30):
            proxy = manager.get_proxy()
            seen.add(proxy)

        # Should have seen all 10 proxies
        assert len(seen) == 10

    def test_rapid_failure_reports(self):
        """Should handle rapid failure reports."""
        manager = ProxyManager(max_failures=2)
        manager.add_proxy("http://proxy:8080")

        # Rapid failures
        manager.get_proxy()
        for _ in range(10):
            manager.report_failure()

        # Proxy should be dead
        stats = manager.get_stats()
        assert stats["dead"] >= 1 or stats["cooldown"] >= 1
