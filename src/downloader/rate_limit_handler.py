"""
Unified rate limit handler that coordinates proxy rotation and VPN switching.

Provides a single interface for handling rate limits across all fallback tiers.
Automatically escalates through strategies:
1. Proxy rotation (if proxies configured)
2. VPN server rotation (if VPN available)
3. Backoff delays (always available)

Usage:
    handler = RateLimitHandler(config)

    # When rate limited
    if response.status_code == 429:
        handler.on_rate_limit()
        # Handler automatically rotates proxy/VPN or applies backoff

    # Get current proxy for requests
    proxy = handler.get_proxy()
    response = httpx.get(url, proxy=proxy)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from .fallback_logging import FallbackLogger
from .proxy_manager import ProxyManager
from .tor_manager import TorManager, get_tor_manager
from .vpn_manager import VPNManager, VPNRotationHelper

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limit handling strategies in escalation order."""
    PROXY_ROTATION = auto()
    TOR_CIRCUIT_REFRESH = auto()
    VPN_ROTATION = auto()
    BACKOFF = auto()


@dataclass
class RateLimitStats:
    """Statistics for rate limit handling."""
    rate_limits_total: int = 0
    proxy_rotations: int = 0
    tor_refreshes: int = 0
    vpn_rotations: int = 0
    backoffs_applied: int = 0
    total_backoff_seconds: float = 0.0
    last_rate_limit: float = 0.0
    last_strategy_used: Optional[RateLimitStrategy] = None


class RateLimitHandler:
    """
    Unified rate limit handler for the fallback system.

    Coordinates proxy rotation, VPN switching, and backoff delays
    to handle rate limits from YouTube and other services.

    Features:
    - Automatic proxy rotation on 429 errors
    - VPN server rotation when proxies exhausted
    - Exponential backoff as final fallback
    - Statistics tracking for debugging
    - Configurable cooldowns and limits

    Usage:
        handler = RateLimitHandler(config)

        # In request loop
        proxy = handler.get_proxy()
        response = httpx.get(url, proxy=proxy)

        if response.status_code == 429:
            handler.on_rate_limit()

        # After successful request
        handler.on_success()
    """

    def __init__(
        self,
        config: "Config | None" = None,
        base_backoff: float = 30.0,
        max_backoff: float = 300.0,
        backoff_multiplier: float = 2.0,
    ):
        self.config = config
        self.log = FallbackLogger("RATE_LIMIT_HANDLER")

        # Backoff settings
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._backoff_multiplier = backoff_multiplier
        self._current_backoff = base_backoff
        self._consecutive_rate_limits = 0

        # Initialize managers
        self._proxy_manager: Optional[ProxyManager] = None
        self._vpn_manager: Optional[VPNManager] = None
        self._vpn_helper: Optional[VPNRotationHelper] = None
        self._tor_manager: Optional[TorManager] = None

        # State persistence
        self._state_file: Optional[str] = None

        # Statistics
        self.stats = RateLimitStats()

        # Load from config
        self._init_from_config()

    def _init_from_config(self) -> None:
        """Initialize proxy, Tor, and VPN managers from config."""
        if not self.config:
            return

        try:
            fallback_cfg = getattr(self.config.download, 'fallback', None)
            if not fallback_cfg:
                return

            # Initialize proxy manager
            proxy_cfg = getattr(fallback_cfg, 'proxy', None)
            if proxy_cfg:
                enabled = proxy_cfg.get('enabled', False) if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'enabled', False)
                if enabled:
                    self._proxy_manager = ProxyManager(self.config)
                    if self._proxy_manager.has_proxies:
                        self.log.info(f"Proxy rotation enabled ({self._proxy_manager.available_count} proxies)")

                        # Load persistent state
                        persist = proxy_cfg.get('persist_state', False) if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'persist_state', False)
                        if persist:
                            state_file = proxy_cfg.get('state_file', '.cache/proxy_state.json') if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'state_file', '.cache/proxy_state.json')
                            self._state_file = state_file
                            if self._proxy_manager.pool.load_state(state_file):
                                self.log.info(f"Loaded proxy state from {state_file}")

                        # Run health check on startup
                        health_check = proxy_cfg.get('health_check_on_startup', False) if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'health_check_on_startup', False)
                        if health_check:
                            self._run_health_check(proxy_cfg)
                    else:
                        self.log.debug("Proxy enabled but no proxies configured")
                        self._proxy_manager = None

            # Initialize Tor manager
            self._tor_manager = get_tor_manager(self.config)
            if self._tor_manager:
                if self._tor_manager.is_available():
                    self.log.info(f"Tor circuit refresh enabled (port {self._tor_manager.socks_port})")
                else:
                    self.log.debug("Tor configured but not available")
                    self._tor_manager = None

            # Initialize VPN manager
            vpn_cfg = getattr(fallback_cfg, 'vpn', None)
            if vpn_cfg:
                enabled = vpn_cfg.get('enabled', False) if isinstance(vpn_cfg, dict) else getattr(vpn_cfg, 'enabled', False)
                if enabled:
                    self._vpn_manager = VPNManager(self.config)
                    if self._vpn_manager.is_available:
                        self._vpn_helper = VPNRotationHelper(self._vpn_manager)
                        self.log.info(f"VPN rotation enabled ({self._vpn_manager.detected_provider.value})")
                    else:
                        self.log.debug("VPN enabled but no provider detected")
                        self._vpn_manager = None

        except Exception as e:
            self.log.debug(f"Error initializing rate limit handler: {e}")

    def _run_health_check(self, proxy_cfg) -> None:
        """Run health check on all proxies."""
        if not self._proxy_manager:
            return

        try:
            test_url = proxy_cfg.get('health_check_url', 'https://www.youtube.com/robots.txt') if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'health_check_url', 'https://www.youtube.com/robots.txt')
            timeout = proxy_cfg.get('health_check_timeout', 10.0) if isinstance(proxy_cfg, dict) else getattr(proxy_cfg, 'health_check_timeout', 10.0)

            self.log.info(f"Running proxy health check against {test_url}...")
            results = self._proxy_manager.pool.test_all_sync(test_url=test_url, timeout=timeout)

            healthy = sum(1 for v in results.values() if v)
            if healthy == 0:
                self.log.warning("No healthy proxies found during health check!")
            else:
                self.log.info(f"Health check complete: {healthy}/{len(results)} proxies healthy")

        except Exception as e:
            self.log.warning(f"Health check failed: {e}")

    @property
    def has_proxy(self) -> bool:
        """Check if proxy rotation is available."""
        return self._proxy_manager is not None and self._proxy_manager.has_proxies

    @property
    def has_vpn(self) -> bool:
        """Check if VPN rotation is available."""
        return self._vpn_manager is not None and self._vpn_manager.is_available

    @property
    def has_tor(self) -> bool:
        """Check if Tor circuit refresh is available."""
        return self._tor_manager is not None and self._tor_manager.is_available()

    def get_proxy(self) -> Optional[str]:
        """
        Get current proxy URL for requests.

        Returns:
            Proxy URL string or None if no proxy available
        """
        if self._proxy_manager:
            return self._proxy_manager.get_proxy()
        return None

    def get_socks5_proxy(self) -> Optional[str]:
        """
        Get SOCKS5 proxy URL (from VPN if available).

        Returns:
            SOCKS5 proxy URL or None
        """
        if self._vpn_manager and self._vpn_manager.socks5_proxy:
            return self._vpn_manager.socks5_proxy
        return self.get_proxy()

    def on_rate_limit(self, source: str = "unknown") -> RateLimitStrategy:
        """
        Handle rate limit event.

        Automatically tries strategies in order:
        1. Proxy rotation
        2. VPN rotation
        3. Backoff delay

        Args:
            source: Description of what got rate limited

        Returns:
            Strategy that was applied
        """
        self.stats.rate_limits_total += 1
        self.stats.last_rate_limit = time.time()
        self._consecutive_rate_limits += 1

        self.log.warning(f"Rate limit detected from {source} (consecutive: {self._consecutive_rate_limits})")

        # Try proxy rotation first
        if self._try_proxy_rotation():
            self.stats.proxy_rotations += 1
            self.stats.last_strategy_used = RateLimitStrategy.PROXY_ROTATION
            self._save_state()
            return RateLimitStrategy.PROXY_ROTATION

        # Try Tor circuit refresh
        if self._try_tor_refresh():
            self.stats.tor_refreshes += 1
            self.stats.last_strategy_used = RateLimitStrategy.TOR_CIRCUIT_REFRESH
            return RateLimitStrategy.TOR_CIRCUIT_REFRESH

        # Try VPN rotation
        if self._try_vpn_rotation():
            self.stats.vpn_rotations += 1
            self.stats.last_strategy_used = RateLimitStrategy.VPN_ROTATION
            return RateLimitStrategy.VPN_ROTATION

        # Apply backoff as fallback
        self._apply_backoff()
        self.stats.backoffs_applied += 1
        self.stats.last_strategy_used = RateLimitStrategy.BACKOFF
        return RateLimitStrategy.BACKOFF

    def _try_proxy_rotation(self) -> bool:
        """Try rotating to a new proxy."""
        if not self._proxy_manager:
            return False

        if self._proxy_manager.available_count <= 1:
            self.log.debug("No available proxies for rotation")
            return False

        self._proxy_manager.report_rate_limit()
        new_proxy = self._proxy_manager.get_proxy()

        if new_proxy:
            self.log.info(f"Rotated to new proxy")
            return True

        return False

    def _try_tor_refresh(self) -> bool:
        """Try refreshing Tor circuit for new exit IP."""
        if not self._tor_manager:
            return False

        if not self._tor_manager.is_available():
            self.log.debug("Tor not available for circuit refresh")
            return False

        success = self._tor_manager.on_rate_limit()
        if success:
            self.log.info("Tor circuit refreshed - new exit IP")
            return True

        return False

    def _try_vpn_rotation(self) -> bool:
        """Try rotating VPN server."""
        if not self._vpn_helper:
            return False

        if not self._vpn_helper.should_rotate():
            self.log.debug("VPN rotation not available (cooldown or limit)")
            return False

        success = self._vpn_helper.rotate_on_rate_limit()
        if success:
            self.log.info("Rotated VPN server")
            # Reset proxy manager after VPN change (new IP)
            if self._proxy_manager:
                self._proxy_manager.reset()
            return True

        return False

    def _save_state(self) -> None:
        """Save proxy state to file."""
        if self._state_file and self._proxy_manager:
            try:
                self._proxy_manager.pool.save_state(self._state_file)
            except Exception as e:
                self.log.debug(f"Could not save proxy state: {e}")

    def _apply_backoff(self) -> None:
        """Apply backoff delay."""
        delay = min(self._current_backoff, self._max_backoff)
        self.log.warning(f"Applying {delay:.0f}s backoff delay...")

        self.stats.total_backoff_seconds += delay
        time.sleep(delay)

        # Increase backoff for next time
        self._current_backoff = min(
            self._current_backoff * self._backoff_multiplier,
            self._max_backoff
        )

    def on_success(self) -> None:
        """Handle successful request (reset backoff)."""
        if self._consecutive_rate_limits > 0:
            self.log.debug(f"Request succeeded after {self._consecutive_rate_limits} rate limits")

        self._consecutive_rate_limits = 0
        self._current_backoff = self._base_backoff

        if self._proxy_manager:
            self._proxy_manager.report_success()

        # Track Tor request for auto-refresh
        if self._tor_manager:
            self._tor_manager.on_request()

    def reset(self) -> None:
        """Reset all handlers."""
        self._consecutive_rate_limits = 0
        self._current_backoff = self._base_backoff

        if self._proxy_manager:
            self._proxy_manager.reset()

        self.stats = RateLimitStats()
        self.log.info("Rate limit handler reset")

    def get_stats(self) -> dict:
        """Get rate limit handling statistics."""
        stats = {
            "rate_limits_total": self.stats.rate_limits_total,
            "proxy_rotations": self.stats.proxy_rotations,
            "tor_refreshes": self.stats.tor_refreshes,
            "vpn_rotations": self.stats.vpn_rotations,
            "backoffs_applied": self.stats.backoffs_applied,
            "total_backoff_seconds": self.stats.total_backoff_seconds,
            "consecutive_rate_limits": self._consecutive_rate_limits,
            "current_backoff": self._current_backoff,
            "has_proxy": self.has_proxy,
            "has_tor": self.has_tor,
            "has_vpn": self.has_vpn,
        }

        if self._proxy_manager:
            stats["proxy_stats"] = self._proxy_manager.get_stats()

        if self._tor_manager:
            stats["tor_stats"] = self._tor_manager.get_stats()

        if self._vpn_manager:
            stats["vpn_stats"] = self._vpn_manager.get_stats()

        return stats

    def log_summary(self) -> None:
        """Log summary of rate limit handling."""
        if self.stats.rate_limits_total == 0:
            self.log.info("No rate limits encountered")
            return

        self.log.info(
            f"Rate limit summary: "
            f"{self.stats.rate_limits_total} total | "
            f"{self.stats.proxy_rotations} proxy rotations | "
            f"{self.stats.tor_refreshes} Tor refreshes | "
            f"{self.stats.vpn_rotations} VPN rotations | "
            f"{self.stats.backoffs_applied} backoffs | "
            f"{self.stats.total_backoff_seconds:.0f}s total delay"
        )


# Singleton instance for easy access
_handler_instance: Optional[RateLimitHandler] = None


def get_rate_limit_handler(config: "Config | None" = None) -> RateLimitHandler:
    """Get or create the global rate limit handler."""
    global _handler_instance

    if _handler_instance is None:
        _handler_instance = RateLimitHandler(config)

    return _handler_instance


def reset_rate_limit_handler() -> None:
    """Reset the global rate limit handler."""
    global _handler_instance

    if _handler_instance:
        _handler_instance.reset()
    _handler_instance = None
