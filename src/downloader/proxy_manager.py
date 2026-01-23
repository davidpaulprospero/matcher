"""
Proxy rotation manager for bypassing rate limits.

Supports multiple proxy sources:
- Static proxy list (file or config)
- SOCKS5 proxies (including VPN SOCKS5 endpoints)
- HTTP/HTTPS proxies
- Free proxy APIs (with health checking)

Usage:
    manager = ProxyManager(config)

    # Get a proxy for requests
    proxy = manager.get_proxy()
    response = httpx.get(url, proxy=proxy)

    # Report rate limit - proxy goes to cooldown
    manager.report_rate_limit(proxy)

    # Get next proxy (rotates automatically)
    proxy = manager.get_proxy()
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .fallback_logging import FallbackLogger

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class ProxyType(Enum):
    """Supported proxy types."""
    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"
    SOCKS4 = "socks4"


class ProxyStatus(Enum):
    """Proxy health status."""
    HEALTHY = "healthy"
    COOLDOWN = "cooldown"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class ProxyInfo:
    """Information about a proxy."""
    url: str
    proxy_type: ProxyType = ProxyType.HTTP
    status: ProxyStatus = ProxyStatus.UNKNOWN
    last_used: float = 0.0
    last_failure: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    source: str = "manual"  # manual, file, api

    # NEW: IP tracking for cooldown across different proxies with same IP
    last_rate_limit: float = 0.0  # Timestamp of last rate limit
    extracted_ip: Optional[str] = None  # Cached extracted IP

    @property
    def is_available(self) -> bool:
        """Check if proxy is available for use."""
        return self.status in (ProxyStatus.HEALTHY, ProxyStatus.UNKNOWN)

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100

    @property
    def success_rate_normalized(self) -> float:
        """Get success rate as 0-1 ratio (for weighting)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0  # Untested = assume good
        return self.success_count / total

    @property
    def is_healthy(self) -> bool:
        """Check if proxy meets minimum success rate threshold."""
        total = self.success_count + self.failure_count
        if total < 3:
            return True  # Not enough data yet
        return self.success_rate_normalized >= 0.3  # 30% threshold

    def extract_ip(self) -> Optional[str]:
        """Extract IP address from proxy URL."""
        if self.extracted_ip:
            return self.extracted_ip

        try:
            parsed = urlparse(self.url)
            host = parsed.hostname
            if host:
                # Check if it's an IP address
                ip_match = re.match(r'^(\d{1,3}\.){3}\d{1,3}$', host)
                if ip_match:
                    self.extracted_ip = host
                    return host
        except Exception:
            pass
        return None

    def to_httpx_format(self) -> str:
        """Get proxy URL in httpx format."""
        return self.url

    def __str__(self) -> str:
        return f"{self.proxy_type.value}://{self._masked_url}"

    @property
    def _masked_url(self) -> str:
        """Get URL with password masked."""
        parsed = urlparse(self.url)
        if parsed.password:
            return self.url.replace(parsed.password, "****")
        return self.url.replace(f"{parsed.scheme}://", "")


@dataclass
class ProxyPool:
    """Pool of proxies with rotation, health tracking, and smart selection."""
    proxies: list[ProxyInfo] = field(default_factory=list)
    cooldown_seconds: float = 300.0  # 5 minutes
    max_failures: int = 3  # Mark dead after this many consecutive failures
    current_index: int = 0

    # NEW: IP cooldown tracking (shared across proxies)
    _rate_limited_ips: Dict[str, float] = field(default_factory=dict)
    ip_cooldown_seconds: float = 300.0  # 5 min cooldown for rate-limited IPs

    def add(self, proxy: ProxyInfo) -> None:
        """Add a proxy to the pool."""
        # Avoid duplicates
        if not any(p.url == proxy.url for p in self.proxies):
            self.proxies.append(proxy)

    def get_next(self, strategy: str = "round_robin", ip_cooldown: float = None) -> Optional[ProxyInfo]:
        """
        Get next available proxy using specified strategy.

        Args:
            strategy: Selection strategy - round_robin, weighted, random, least_used
            ip_cooldown: IP cooldown in seconds (None = use pool default)

        Returns:
            Next available ProxyInfo or None
        """
        if not self.proxies:
            return None

        now = time.time()
        cooldown = ip_cooldown if ip_cooldown is not None else self.ip_cooldown_seconds

        # Get available proxies (not in cooldown, not dead, IP not rate-limited)
        available = self._get_available_proxies(now, cooldown)
        if not available:
            logger.warning("[proxy] No available proxies (all in cooldown or dead)")
            return None

        # Select using strategy
        if strategy == "round_robin":
            proxy = self._select_round_robin(available)
        elif strategy == "weighted":
            proxy = self._select_weighted(available)
        elif strategy == "random":
            proxy = random.choice(available)
        elif strategy == "least_used":
            proxy = min(available, key=lambda p: p.last_used)
        else:
            # Default to round robin
            proxy = self._select_round_robin(available)

        proxy.last_used = now
        return proxy

    def _get_available_proxies(self, now: float, ip_cooldown: float) -> List[ProxyInfo]:
        """Get list of proxies that are available for use."""
        available = []

        for proxy in self.proxies:
            # Check if on cooldown
            if proxy.status == ProxyStatus.COOLDOWN:
                if now - proxy.last_failure >= self.cooldown_seconds:
                    proxy.status = ProxyStatus.UNKNOWN
                    proxy.failure_count = 0
                else:
                    continue

            # Skip dead proxies
            if proxy.status == ProxyStatus.DEAD:
                continue

            # Check IP cooldown (rate-limited IPs across different proxies)
            ip = proxy.extract_ip()
            if ip and ip in self._rate_limited_ips:
                if now - self._rate_limited_ips[ip] < ip_cooldown:
                    continue  # IP still in cooldown
                else:
                    del self._rate_limited_ips[ip]  # Cooldown expired

            available.append(proxy)

        return available

    def _select_round_robin(self, available: List[ProxyInfo]) -> ProxyInfo:
        """Select using round-robin from available proxies."""
        # Find the next available proxy in original order
        for i in range(len(self.proxies)):
            idx = (self.current_index + i) % len(self.proxies)
            proxy = self.proxies[idx]
            if proxy in available:
                self.current_index = (idx + 1) % len(self.proxies)
                return proxy
        return available[0]  # Fallback

    def _select_weighted(self, available: List[ProxyInfo]) -> ProxyInfo:
        """Select using weighted random by success rate."""
        # Weight by success rate (higher = more likely)
        weights = []
        for p in available:
            # Combine success rate with inverse latency
            weight = p.success_rate_normalized
            if p.avg_latency_ms > 0:
                # Lower latency = higher weight (normalize to 0-1)
                latency_factor = max(0.1, 1.0 - (p.avg_latency_ms / 5000))
                weight = weight * 0.7 + latency_factor * 0.3
            weights.append(max(0.1, weight))  # Minimum weight

        total = sum(weights)
        if total == 0:
            return random.choice(available)

        return random.choices(available, weights=weights, k=1)[0]

    def get_random(self) -> Optional[ProxyInfo]:
        """Get a random available proxy."""
        available = [p for p in self.proxies if p.is_available]
        if not available:
            return None
        return random.choice(available)

    def get_n_proxies(self, n: int) -> List[ProxyInfo]:
        """
        Allocate N unique proxies for parallel workers.

        Returns fewer proxies if not enough are available.
        Prioritizes healthy proxies with lower latency.

        Args:
            n: Number of proxies to allocate

        Returns:
            List of ProxyInfo objects (may be fewer than n)
        """
        now = time.time()

        # Get available proxies (healthy or unknown, not in cooldown/dead)
        available = self._get_available_proxies(now, self.ip_cooldown_seconds)

        # Sort by health: healthy first, then by latency
        def sort_key(p: ProxyInfo) -> tuple:
            # Healthy with data first, then unknown, sorted by latency
            health_priority = 0 if p.status == ProxyStatus.HEALTHY else 1
            return (health_priority, p.avg_latency_ms)

        available.sort(key=sort_key)

        # Allocate up to n proxies
        allocated = available[:n]

        # Log allocation
        logger.info(
            f"[proxy_pool] Allocating {len(allocated)}/{n} requested proxies "
            f"({len(available)} available, {len(self.proxies)} total)"
        )

        for i, proxy in enumerate(allocated):
            proxy.last_used = now
            logger.debug(f"[proxy_pool] Worker {i} → {proxy._masked_url}")

        return allocated

    def get_proxy_for_worker(self, worker_id: int) -> Optional[str]:
        """
        Get consistent proxy URL for specific worker ID.

        Args:
            worker_id: Worker identifier (0-indexed)

        Returns:
            Proxy URL string or None if not enough proxies
        """
        healthy = [p for p in self.proxies if p.is_available]
        if worker_id < len(healthy):
            return healthy[worker_id].url
        return None

    def report_worker_result(
        self,
        worker_id: int,
        proxy_url: str,
        success: bool,
        response_time: float
    ) -> None:
        """
        Track per-worker proxy performance.

        Args:
            worker_id: Worker identifier
            proxy_url: Proxy URL used
            success: Whether the request succeeded
            response_time: Response time in seconds
        """
        for proxy in self.proxies:
            if proxy.url == proxy_url:
                if success:
                    proxy.success_count += 1
                    # Update rolling average latency
                    latency_ms = response_time * 1000
                    if proxy.avg_latency_ms == 0:
                        proxy.avg_latency_ms = latency_ms
                    else:
                        proxy.avg_latency_ms = (proxy.avg_latency_ms + latency_ms) / 2
                else:
                    proxy.failure_count += 1
                logger.debug(
                    f"[proxy_pool] Worker {worker_id} result: "
                    f"{'✓' if success else '✗'} via {proxy._masked_url} ({response_time:.2f}s)"
                )
                break

    def mark_rate_limited_ip(self, proxy: ProxyInfo) -> None:
        """Mark proxy's IP as rate-limited."""
        ip = proxy.extract_ip()
        if ip:
            self._rate_limited_ips[ip] = time.time()
            logger.debug(f"[proxy] Marked IP {ip} as rate-limited")

    # === Health Check Methods ===

    def test_proxy_sync(
        self,
        proxy: ProxyInfo,
        test_url: str = "https://www.youtube.com/robots.txt",
        timeout: float = 10.0
    ) -> Tuple[bool, float]:
        """
        Test a single proxy synchronously.

        Returns:
            Tuple of (success, latency_ms)
        """
        try:
            start = time.time()
            with httpx.Client(proxy=proxy.url, timeout=timeout) as client:
                response = client.get(test_url)
                latency_ms = (time.time() - start) * 1000

                if response.status_code < 400:
                    return True, latency_ms
                elif response.status_code == 429:
                    logger.debug(f"[proxy] {proxy.url} got 429 during health check")
                    return False, latency_ms
                else:
                    return False, latency_ms
        except Exception as e:
            logger.debug(f"[proxy] {proxy.url} health check failed: {e}")
            return False, 0.0

    def test_all_sync(
        self,
        test_url: str = "https://www.youtube.com/robots.txt",
        timeout: float = 10.0
    ) -> Dict[str, bool]:
        """
        Test all proxies synchronously.

        Returns:
            Dict mapping proxy URL to health status
        """
        results = {}

        for proxy in self.proxies:
            success, latency = self.test_proxy_sync(proxy, test_url, timeout)
            results[proxy.url] = success

            if success:
                proxy.success_count += 1
                proxy.avg_latency_ms = latency
                proxy.status = ProxyStatus.HEALTHY
            else:
                proxy.failure_count += 1
                if proxy.failure_count >= self.max_failures:
                    proxy.status = ProxyStatus.DEAD

        healthy_count = sum(1 for v in results.values() if v)
        logger.info(f"[proxy] Health check: {healthy_count}/{len(results)} proxies healthy")

        return results

    async def test_all_async(
        self,
        test_url: str = "https://www.youtube.com/robots.txt",
        timeout: float = 10.0
    ) -> Dict[str, bool]:
        """
        Test all proxies asynchronously (faster for many proxies).

        Returns:
            Dict mapping proxy URL to health status
        """
        results = {}

        async def test_proxy(proxy: ProxyInfo) -> Tuple[str, bool, float]:
            try:
                start = time.time()
                async with httpx.AsyncClient(proxy=proxy.url, timeout=timeout) as client:
                    response = await client.get(test_url)
                    latency_ms = (time.time() - start) * 1000
                    success = response.status_code < 400
                    return proxy.url, success, latency_ms
            except Exception as e:
                logger.debug(f"[proxy] {proxy.url} async health check failed: {e}")
                return proxy.url, False, 0.0

        tasks = [test_proxy(p) for p in self.proxies]
        task_results = await asyncio.gather(*tasks)

        for url, success, latency in task_results:
            results[url] = success
            # Update proxy stats
            proxy = next((p for p in self.proxies if p.url == url), None)
            if proxy:
                if success:
                    proxy.success_count += 1
                    proxy.avg_latency_ms = latency
                    proxy.status = ProxyStatus.HEALTHY
                else:
                    proxy.failure_count += 1

        healthy_count = sum(1 for v in results.values() if v)
        logger.info(f"[proxy] Async health check: {healthy_count}/{len(results)} proxies healthy")

        return results

    # === Persistence Methods ===

    def save_state(self, filepath: str) -> None:
        """
        Save proxy statistics to file for persistence between runs.

        Args:
            filepath: Path to state file (JSON)
        """
        state = {
            "saved_at": time.time(),
            "proxies": [
                {
                    "url": p.url,
                    "success_count": p.success_count,
                    "failure_count": p.failure_count,
                    "avg_latency_ms": p.avg_latency_ms,
                    "last_rate_limit": p.last_rate_limit,
                    "status": p.status.value,
                }
                for p in self.proxies
            ],
            "rate_limited_ips": self._rate_limited_ips,
        }

        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"[proxy] Saved state to {filepath}")
        except Exception as e:
            logger.warning(f"[proxy] Could not save state: {e}")

    def load_state(self, filepath: str) -> bool:
        """
        Load proxy statistics from file.

        Args:
            filepath: Path to state file (JSON)

        Returns:
            True if state was loaded successfully
        """
        if not Path(filepath).exists():
            return False

        try:
            with open(filepath) as f:
                state = json.load(f)

            # Check if state is too old (> 24 hours)
            saved_at = state.get("saved_at", 0)
            if time.time() - saved_at > 86400:
                logger.info("[proxy] State file too old, ignoring")
                return False

            # Merge with current proxies
            state_map = {p["url"]: p for p in state.get("proxies", [])}
            for proxy in self.proxies:
                if proxy.url in state_map:
                    saved = state_map[proxy.url]
                    proxy.success_count = saved.get("success_count", 0)
                    proxy.failure_count = saved.get("failure_count", 0)
                    proxy.avg_latency_ms = saved.get("avg_latency_ms", 0.0)
                    proxy.last_rate_limit = saved.get("last_rate_limit", 0.0)
                    # Restore status if not dead
                    status_str = saved.get("status", "unknown")
                    if status_str != "dead":
                        try:
                            proxy.status = ProxyStatus(status_str)
                        except ValueError:
                            proxy.status = ProxyStatus.UNKNOWN

            # Restore rate-limited IPs (only recent ones)
            now = time.time()
            for ip, timestamp in state.get("rate_limited_ips", {}).items():
                if now - timestamp < self.ip_cooldown_seconds:
                    self._rate_limited_ips[ip] = timestamp

            logger.info(f"[proxy] Loaded state from {filepath}")
            return True

        except Exception as e:
            logger.warning(f"[proxy] Could not load state: {e}")
            return False

    def mark_success(self, proxy: ProxyInfo, latency_ms: float = 0.0) -> None:
        """Mark proxy as successful."""
        proxy.status = ProxyStatus.HEALTHY
        proxy.success_count += 1
        proxy.failure_count = 0  # Reset consecutive failures

        # Update average latency
        if latency_ms > 0:
            if proxy.avg_latency_ms == 0:
                proxy.avg_latency_ms = latency_ms
            else:
                proxy.avg_latency_ms = (proxy.avg_latency_ms + latency_ms) / 2

    def mark_failure(self, proxy: ProxyInfo, is_rate_limit: bool = False) -> None:
        """Mark proxy as failed."""
        proxy.failure_count += 1
        proxy.last_failure = time.time()

        if is_rate_limit:
            proxy.status = ProxyStatus.COOLDOWN
            proxy.last_rate_limit = time.time()
            # Also mark the IP as rate-limited globally
            self.mark_rate_limited_ip(proxy)
        elif proxy.failure_count >= self.max_failures:
            proxy.status = ProxyStatus.COOLDOWN

        # Mark as dead after too many total failures
        if proxy.failure_count >= self.max_failures * 3:
            proxy.status = ProxyStatus.DEAD

    def get_healthy_count(self) -> int:
        """Get count of healthy/available proxies."""
        return sum(1 for p in self.proxies if p.is_available)

    def reset_all(self) -> None:
        """Reset all proxies to unknown status."""
        for proxy in self.proxies:
            proxy.status = ProxyStatus.UNKNOWN
            proxy.failure_count = 0
        self._rate_limited_ips.clear()


class ProxyManager:
    """
    Manages proxy rotation for HTTP requests.

    Features:
    - Multiple proxy sources (config, file, API)
    - Health checking with cooldown
    - Automatic rotation on rate limits
    - SOCKS5 support for VPN integration

    Usage:
        manager = ProxyManager(config)

        # With httpx
        proxy = manager.get_proxy()
        if proxy:
            client = httpx.Client(proxy=proxy)
            response = client.get(url)
            if response.status_code == 429:
                manager.report_rate_limit()
    """

    def __init__(
        self,
        config: "Config | None" = None,
        cooldown_seconds: float = 300.0,
        max_failures: int = 3,
    ):
        self.config = config
        self.pool = ProxyPool(
            cooldown_seconds=cooldown_seconds,
            max_failures=max_failures,
        )
        self.log = FallbackLogger("PROXY_MANAGER")
        self._current_proxy: Optional[ProxyInfo] = None
        self._enabled = True

        # Load proxies from config
        self._load_from_config()

    def _load_from_config(self) -> None:
        """Load proxies from config."""
        if not self.config:
            return

        try:
            fallback_cfg = getattr(self.config.download, 'fallback', None)
            if not fallback_cfg:
                return

            proxy_cfg = getattr(fallback_cfg, 'proxy', None)
            if not proxy_cfg:
                return

            # Check if enabled
            if isinstance(proxy_cfg, dict):
                self._enabled = proxy_cfg.get('enabled', True)
                sources = proxy_cfg.get('sources', [])
            else:
                self._enabled = getattr(proxy_cfg, 'enabled', True)
                sources = getattr(proxy_cfg, 'sources', [])

            if not self._enabled:
                self.log.debug("Proxy rotation disabled in config")
                return

            # Load from sources
            for source in sources:
                if isinstance(source, dict):
                    self._load_source(source)
                elif isinstance(source, str):
                    # Direct proxy URL
                    self._add_proxy_url(source, "config")

            self.log.info(f"Loaded {len(self.pool.proxies)} proxies from config")

        except Exception as e:
            self.log.debug(f"Error loading proxies from config: {e}")

    def _load_source(self, source: dict) -> None:
        """Load proxies from a source definition."""
        source_type = source.get('type', 'url')

        if source_type in ('url', 'socks5', 'http', 'https'):
            url = source.get('url', '')
            if url:
                self._add_proxy_url(url, "config")

        elif source_type == 'list':
            # Load from file
            file_path = source.get('file', '')
            if file_path:
                self._load_from_file(Path(file_path))

        elif source_type == 'env':
            # Load from environment variable
            import os
            env_var = source.get('env', 'HTTP_PROXY')
            url = os.environ.get(env_var, '')
            if url:
                self._add_proxy_url(url, f"env:{env_var}")

    def _load_from_file(self, file_path: Path) -> None:
        """Load proxies from a file (one per line)."""
        if not file_path.exists():
            self.log.warning(f"Proxy file not found: {file_path}")
            return

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self._add_proxy_url(line, f"file:{file_path.name}")

            self.log.debug(f"Loaded proxies from {file_path}")
        except Exception as e:
            self.log.warning(f"Error reading proxy file: {e}")

    def _add_proxy_url(self, url: str, source: str) -> None:
        """Add a proxy URL to the pool."""
        # Normalize URL
        url = url.strip()
        if not url:
            return

        # Detect proxy type
        proxy_type = ProxyType.HTTP
        if url.startswith('socks5://'):
            proxy_type = ProxyType.SOCKS5
        elif url.startswith('socks4://'):
            proxy_type = ProxyType.SOCKS4
        elif url.startswith('https://'):
            proxy_type = ProxyType.HTTPS
        elif not url.startswith('http://'):
            # Add default scheme
            url = f"http://{url}"

        proxy = ProxyInfo(
            url=url,
            proxy_type=proxy_type,
            source=source,
        )
        self.pool.add(proxy)

    def add_proxy(self, url: str, proxy_type: ProxyType = ProxyType.HTTP) -> None:
        """Manually add a proxy."""
        if not url.startswith(('http://', 'https://', 'socks')):
            url = f"{proxy_type.value}://{url}"

        proxy = ProxyInfo(url=url, proxy_type=proxy_type, source="manual")
        self.pool.add(proxy)
        self.log.debug(f"Added proxy: {proxy}")

    def add_socks5(self, host: str, port: int, username: str = None, password: str = None) -> None:
        """Add a SOCKS5 proxy (commonly used with VPNs)."""
        if username and password:
            url = f"socks5://{username}:{password}@{host}:{port}"
        else:
            url = f"socks5://{host}:{port}"

        proxy = ProxyInfo(url=url, proxy_type=ProxyType.SOCKS5, source="manual")
        self.pool.add(proxy)
        self.log.debug(f"Added SOCKS5 proxy: {host}:{port}")

    def get_proxy(self) -> Optional[str]:
        """
        Get the next available proxy URL.

        Returns:
            Proxy URL string for httpx, or None if no proxies available
        """
        if not self._enabled or not self.pool.proxies:
            return None

        proxy = self.pool.get_next()
        if proxy:
            self._current_proxy = proxy
            self.log.debug(f"Using proxy: {proxy}")
            return proxy.to_httpx_format()

        self.log.warning("No available proxies")
        return None

    def get_random_proxy(self) -> Optional[str]:
        """Get a random available proxy."""
        if not self._enabled or not self.pool.proxies:
            return None

        proxy = self.pool.get_random()
        if proxy:
            self._current_proxy = proxy
            return proxy.to_httpx_format()
        return None

    def report_success(self, latency_ms: float = 0.0) -> None:
        """Report that the current proxy succeeded."""
        if self._current_proxy:
            self.pool.mark_success(self._current_proxy, latency_ms)
            self.log.debug(f"Proxy success: {self._current_proxy} ({latency_ms:.0f}ms)")

    def report_failure(self, is_rate_limit: bool = False) -> None:
        """Report that the current proxy failed."""
        if self._current_proxy:
            self.pool.mark_failure(self._current_proxy, is_rate_limit)
            if is_rate_limit:
                self.log.warning(f"Proxy rate limited: {self._current_proxy} -> cooldown")
            else:
                self.log.debug(f"Proxy failure: {self._current_proxy}")

    def report_rate_limit(self) -> None:
        """Shortcut for reporting rate limit on current proxy."""
        self.report_failure(is_rate_limit=True)

    @property
    def has_proxies(self) -> bool:
        """Check if any proxies are configured."""
        return len(self.pool.proxies) > 0

    @property
    def available_count(self) -> int:
        """Get count of available proxies."""
        return self.pool.get_healthy_count()

    def get_stats(self) -> dict:
        """Get proxy pool statistics."""
        total = len(self.pool.proxies)
        healthy = sum(1 for p in self.pool.proxies if p.status == ProxyStatus.HEALTHY)
        cooldown = sum(1 for p in self.pool.proxies if p.status == ProxyStatus.COOLDOWN)
        dead = sum(1 for p in self.pool.proxies if p.status == ProxyStatus.DEAD)
        unknown = sum(1 for p in self.pool.proxies if p.status == ProxyStatus.UNKNOWN)

        return {
            "total": total,
            "healthy": healthy,
            "cooldown": cooldown,
            "dead": dead,
            "unknown": unknown,
            "available": healthy + unknown,
        }

    def reset(self) -> None:
        """Reset all proxy states."""
        self.pool.reset_all()
        self.log.info("Reset all proxy states")


class ProxiedHttpClient:
    """
    HTTP client wrapper with automatic proxy rotation.

    Usage:
        client = ProxiedHttpClient(proxy_manager)
        response = client.get(url)  # Automatically uses/rotates proxies
    """

    def __init__(
        self,
        proxy_manager: ProxyManager,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.proxy_manager = proxy_manager
        self.timeout = timeout
        self.max_retries = max_retries
        self.log = FallbackLogger("PROXIED_CLIENT")

    def get(self, url: str, **kwargs) -> httpx.Response:
        """GET request with automatic proxy rotation on failure."""
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        """POST request with automatic proxy rotation on failure."""
        return self._request("POST", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make request with retry and proxy rotation."""
        last_error = None

        for attempt in range(self.max_retries):
            proxy = self.proxy_manager.get_proxy()

            try:
                start_time = time.time()

                with httpx.Client(
                    proxy=proxy,
                    timeout=self.timeout,
                    follow_redirects=True,
                ) as client:
                    response = client.request(method, url, **kwargs)

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 429:
                    self.log.warning(f"Rate limited via proxy (attempt {attempt + 1})")
                    self.proxy_manager.report_rate_limit()
                    last_error = httpx.HTTPStatusError(
                        "Rate limited",
                        request=response.request,
                        response=response,
                    )
                    continue

                self.proxy_manager.report_success(latency_ms)
                return response

            except httpx.TimeoutException as e:
                self.log.debug(f"Timeout via proxy (attempt {attempt + 1})")
                self.proxy_manager.report_failure()
                last_error = e
            except httpx.ProxyError as e:
                self.log.debug(f"Proxy error (attempt {attempt + 1}): {e}")
                self.proxy_manager.report_failure()
                last_error = e
            except Exception as e:
                self.log.debug(f"Request error (attempt {attempt + 1}): {e}")
                self.proxy_manager.report_failure()
                last_error = e

        # All retries failed
        if last_error:
            raise last_error
        raise httpx.RequestError(f"All {self.max_retries} attempts failed")


# Convenience functions

def load_proxies_from_env() -> list[str]:
    """Load proxy URLs from common environment variables."""
    import os

    proxies = []
    env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY', 'ALL_PROXY']

    for var in env_vars:
        value = os.environ.get(var, '') or os.environ.get(var.lower(), '')
        if value:
            proxies.append(value)

    return proxies


def check_proxy(proxy_url: str, test_url: str = "https://httpbin.org/ip", timeout: float = 10.0) -> tuple[bool, float, str]:
    """
    Test if a proxy is working.

    Args:
        proxy_url: The proxy URL to test
        test_url: URL to fetch for testing
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, latency_ms, message)
    """
    try:
        start_time = time.time()

        with httpx.Client(proxy=proxy_url, timeout=timeout) as client:
            response = client.get(test_url)

        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            return True, latency_ms, f"OK ({latency_ms:.0f}ms)"
        else:
            return False, latency_ms, f"HTTP {response.status_code}"

    except httpx.ProxyError as e:
        return False, 0, f"Proxy error: {e}"
    except httpx.TimeoutException:
        return False, 0, "Timeout"
    except Exception as e:
        return False, 0, f"Error: {e}"
