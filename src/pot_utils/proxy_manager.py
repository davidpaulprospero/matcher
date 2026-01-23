"""
Proxy Manager - Automatic proxy rotation for YouTube access.

Handles multiple proxies with automatic failover when rate limited.
Integrates with yt-dlp and caption fetching.
"""

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ProxyStats:
    """Statistics for a single proxy."""
    url: str
    success_count: int = 0
    failure_count: int = 0
    rate_limit_count: int = 0
    last_used: float = 0.0
    last_rate_limit: float = 0.0
    cooldown_until: float = 0.0
    is_disabled: bool = False

    @property
    def total_requests(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def is_cooling_down(self) -> bool:
        return time.time() < self.cooldown_until


@dataclass
class ProxyConfig:
    """Configuration for proxy rotation."""
    proxies: List[str] = field(default_factory=list)
    rotation_strategy: str = "round_robin"  # round_robin, random, least_used, performance
    cooldown_on_rate_limit: float = 300.0  # 5 minutes cooldown after rate limit
    max_failures_before_disable: int = 5
    re_enable_after: float = 1800.0  # Re-enable disabled proxy after 30 minutes
    auto_detect_env_proxy: bool = True


class ProxyManager:
    """
    Manages proxy rotation for YouTube downloads.

    Features:
    - Multiple rotation strategies
    - Automatic cooldown on rate limit
    - Performance-based selection
    - Thread-safe operation
    """

    _instance: Optional['ProxyManager'] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or ProxyConfig()
        self.proxies: Dict[str, ProxyStats] = {}
        self.current_index = 0
        self._lock = threading.Lock()

        # Initialize proxies
        self._init_proxies()

    @classmethod
    def get_instance(cls, config: Optional[ProxyConfig] = None) -> 'ProxyManager':
        """Get singleton instance of ProxyManager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            elif config is not None:
                # Update config if provided
                cls._instance.config = config
                cls._instance._init_proxies()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _init_proxies(self):
        """Initialize proxy list from config and environment."""
        proxy_urls = list(self.config.proxies)

        # Auto-detect from environment
        if self.config.auto_detect_env_proxy:
            for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                env_proxy = os.environ.get(env_var)
                if env_proxy and env_proxy not in proxy_urls:
                    proxy_urls.append(env_proxy)
                    logger.info(f"[proxy] Auto-detected proxy from {env_var}: {self._mask_proxy(env_proxy)}")

        # Add "no proxy" option as fallback
        if "direct" not in proxy_urls:
            proxy_urls.append("direct")

        # Initialize stats for each proxy
        for url in proxy_urls:
            if url not in self.proxies:
                self.proxies[url] = ProxyStats(url=url)

        logger.info(f"[proxy] Initialized {len(self.proxies)} proxies")

    def _mask_proxy(self, url: str) -> str:
        """Mask proxy URL for logging (hide credentials)."""
        if url == "direct":
            return "direct"
        try:
            parsed = urlparse(url)
            if parsed.password:
                return f"{parsed.scheme}://{parsed.username}:****@{parsed.hostname}:{parsed.port}"
            return url
        except Exception:
            return "****"

    def add_proxy(self, url: str):
        """Add a new proxy to the rotation."""
        with self._lock:
            if url not in self.proxies:
                self.proxies[url] = ProxyStats(url=url)
                logger.info(f"[proxy] Added proxy: {self._mask_proxy(url)}")

    def remove_proxy(self, url: str):
        """Remove a proxy from rotation."""
        with self._lock:
            if url in self.proxies and url != "direct":
                del self.proxies[url]
                logger.info(f"[proxy] Removed proxy: {self._mask_proxy(url)}")

    def get_available_proxies(self) -> List[str]:
        """Get list of currently available (not cooling down or disabled) proxies."""
        now = time.time()
        available = []

        for url, stats in self.proxies.items():
            # Re-enable disabled proxies after timeout
            if stats.is_disabled and now > stats.cooldown_until + self.config.re_enable_after:
                stats.is_disabled = False
                stats.failure_count = 0
                stats.rate_limit_count = 0
                logger.info(f"[proxy] Re-enabled proxy: {self._mask_proxy(url)}")

            if not stats.is_disabled and not stats.is_cooling_down:
                available.append(url)

        return available

    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy based on rotation strategy."""
        with self._lock:
            available = self.get_available_proxies()

            if not available:
                logger.warning("[proxy] No available proxies! All are cooling down or disabled.")
                # Return direct connection as last resort
                return "direct"

            strategy = self.config.rotation_strategy

            if strategy == "round_robin":
                self.current_index = (self.current_index + 1) % len(available)
                selected = available[self.current_index % len(available)]

            elif strategy == "random":
                selected = random.choice(available)

            elif strategy == "least_used":
                selected = min(available, key=lambda u: self.proxies[u].total_requests)

            elif strategy == "performance":
                # Prefer proxies with better success rates
                selected = max(available, key=lambda u: self.proxies[u].success_rate)

            else:
                selected = available[0]

            self.proxies[selected].last_used = time.time()
            return selected

    def get_current_proxy(self) -> Optional[str]:
        """Get current proxy without rotating."""
        available = self.get_available_proxies()
        if not available:
            return "direct"
        return available[self.current_index % len(available)] if available else None

    def report_success(self, proxy_url: str):
        """Report successful request through proxy."""
        with self._lock:
            if proxy_url in self.proxies:
                self.proxies[proxy_url].success_count += 1
                logger.debug(f"[proxy] Success via {self._mask_proxy(proxy_url)}")

    def report_failure(self, proxy_url: str, is_rate_limit: bool = False):
        """Report failed request through proxy."""
        with self._lock:
            if proxy_url not in self.proxies:
                return

            stats = self.proxies[proxy_url]
            stats.failure_count += 1

            if is_rate_limit:
                stats.rate_limit_count += 1
                stats.last_rate_limit = time.time()
                stats.cooldown_until = time.time() + self.config.cooldown_on_rate_limit
                logger.warning(f"[proxy] Rate limited on {self._mask_proxy(proxy_url)}, "
                             f"cooldown for {self.config.cooldown_on_rate_limit}s")

            # Disable proxy after too many failures
            if stats.failure_count >= self.config.max_failures_before_disable:
                stats.is_disabled = True
                stats.cooldown_until = time.time()
                logger.warning(f"[proxy] Disabled proxy {self._mask_proxy(proxy_url)} "
                             f"after {stats.failure_count} failures")

    def get_proxy_for_ytdlp(self) -> Optional[str]:
        """Get proxy URL formatted for yt-dlp --proxy argument."""
        proxy = self.get_next_proxy()
        if proxy == "direct" or proxy is None:
            return None
        return proxy

    def get_proxy_for_requests(self) -> Optional[Dict[str, str]]:
        """Get proxy dict formatted for requests library."""
        proxy = self.get_next_proxy()
        if proxy == "direct" or proxy is None:
            return None
        return {
            "http": proxy,
            "https": proxy,
        }

    def get_stats(self) -> Dict[str, dict]:
        """Get statistics for all proxies."""
        return {
            url: {
                "success_count": stats.success_count,
                "failure_count": stats.failure_count,
                "rate_limit_count": stats.rate_limit_count,
                "success_rate": f"{stats.success_rate:.1%}",
                "is_disabled": stats.is_disabled,
                "is_cooling_down": stats.is_cooling_down,
            }
            for url, stats in self.proxies.items()
        }

    def print_stats(self):
        """Print proxy statistics to console."""
        print("\n  Proxy Statistics:")
        print("  " + "-" * 60)
        for url, stats in self.proxies.items():
            status = "DISABLED" if stats.is_disabled else ("COOLDOWN" if stats.is_cooling_down else "OK")
            print(f"    {self._mask_proxy(url)[:40]:<40} "
                  f"{stats.success_count:>4} ok / {stats.failure_count:>3} fail / "
                  f"{stats.rate_limit_count:>2} 429  [{status}]")
        print("  " + "-" * 60)


def is_rate_limit_error(error_text: str) -> bool:
    """Check if error text indicates a rate limit."""
    if not error_text:
        return False
    error_lower = error_text.lower()
    return any(indicator in error_lower for indicator in [
        'rate-limited',
        'rate limited',
        '429',
        'too many requests',
        'try again later',
    ])


def with_proxy_rotation(func: Callable, max_retries: int = 3):
    """
    Decorator/wrapper for functions that should use proxy rotation.

    Usage:
        result = with_proxy_rotation(download_function, max_retries=3)(video_id)
    """
    def wrapper(*args, **kwargs):
        manager = ProxyManager.get_instance()
        last_error = None

        for attempt in range(max_retries):
            proxy = manager.get_next_proxy()

            try:
                # Inject proxy into kwargs if function accepts it
                if 'proxy' in func.__code__.co_varnames:
                    kwargs['proxy'] = proxy if proxy != "direct" else None

                result = func(*args, **kwargs)
                manager.report_success(proxy)
                return result

            except Exception as e:
                last_error = e
                error_text = str(e)
                is_rate_limit = is_rate_limit_error(error_text)
                manager.report_failure(proxy, is_rate_limit=is_rate_limit)

                if is_rate_limit:
                    logger.warning(f"[proxy] Rate limit on attempt {attempt + 1}, rotating proxy...")
                else:
                    logger.warning(f"[proxy] Error on attempt {attempt + 1}: {error_text[:100]}")

                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry

        raise last_error

    return wrapper
