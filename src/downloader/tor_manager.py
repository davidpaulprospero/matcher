"""
Tor circuit management via control port.

Allows refreshing IP identity (exit node) without restarting Tor.
This is useful for bypassing rate limits by getting a new IP.

Requirements:
    - Tor running with control port enabled:
      tor --SocksPort 9050 --ControlPort 9051
    - Or use scripts/start_tor.ps1

Usage:
    from src.downloader.tor_manager import TorManager, get_tor_manager

    # Create from config
    tor = get_tor_manager(config)
    if tor and tor.is_available():
        print(f"Tor proxy: {tor.get_proxy_url()}")

    # Refresh circuit on rate limit
    if rate_limited:
        tor.on_rate_limit()

    # Auto-refresh after N requests
    tor.on_request()  # Call after each request
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TorStatus:
    """Status information for Tor connection."""
    connected: bool = False
    circuit_established: bool = False
    exit_ip: Optional[str] = None
    requests_since_refresh: int = 0
    last_refresh_time: float = 0.0
    total_refreshes: int = 0


@dataclass
class TorStats:
    """Statistics for Tor usage."""
    total_requests: int = 0
    total_refreshes: int = 0
    refresh_on_rate_limit: int = 0
    refresh_on_interval: int = 0
    refresh_failures: int = 0
    last_exit_ip: Optional[str] = None


class TorManager:
    """
    Manage Tor SOCKS proxy and circuit refresh via control port.

    Tor provides anonymous IP rotation through the onion network.
    Unlike VPN, Tor can refresh circuits (get new exit IP) instantly
    via the control port without disconnecting.

    Features:
    - SOCKS5 proxy for HTTP requests
    - Circuit refresh on rate limit (429)
    - Automatic circuit refresh after N requests
    - Minimum interval between refreshes (Tor recommends 10s)
    """

    def __init__(
        self,
        socks_port: int = 9050,
        control_port: int = 9051,
        control_password: str = "",
        refresh_after_requests: int = 50,
        min_refresh_interval: float = 10.0,
        refresh_on_rate_limit: bool = True,
    ):
        """
        Initialize TorManager.

        Args:
            socks_port: Tor SOCKS5 proxy port (default: 9050)
            control_port: Tor control port for circuit management (default: 9051)
            control_password: Control port password (empty for no auth)
            refresh_after_requests: Refresh circuit after N requests
            min_refresh_interval: Minimum seconds between refreshes
            refresh_on_rate_limit: Refresh circuit on 429 errors
        """
        self.socks_port = socks_port
        self.control_port = control_port
        self.control_password = control_password
        self.refresh_after_requests = refresh_after_requests
        self.min_refresh_interval = min_refresh_interval
        self.refresh_on_rate_limit_enabled = refresh_on_rate_limit

        self._requests_count = 0
        self._last_refresh = 0.0
        self._status = TorStatus()
        self._stats = TorStats()

    def get_proxy_url(self) -> str:
        """Get SOCKS5 proxy URL for Tor."""
        return f"socks5://127.0.0.1:{self.socks_port}"

    def is_available(self) -> bool:
        """
        Check if Tor SOCKS port is accepting connections.

        Returns:
            True if Tor is running and accepting connections
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect(("127.0.0.1", self.socks_port))
                self._status.connected = True
                return True
        except (socket.error, socket.timeout):
            self._status.connected = False
            return False

    def is_control_available(self) -> bool:
        """Check if Tor control port is accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect(("127.0.0.1", self.control_port))
                return True
        except (socket.error, socket.timeout):
            return False

    def refresh_circuit(self) -> bool:
        """
        Request new Tor circuit (new exit IP).

        Sends SIGNAL NEWNYM to Tor control port. This tells Tor
        to build new circuits for future requests, giving you
        a new exit node and IP address.

        Returns:
            True if circuit refresh was successful
        """
        # Check minimum interval
        now = time.time()
        elapsed = now - self._last_refresh
        if elapsed < self.min_refresh_interval:
            remaining = self.min_refresh_interval - elapsed
            logger.debug(f"[tor] Skipping refresh, {remaining:.1f}s until next allowed")
            return False

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)
                s.connect(("127.0.0.1", self.control_port))

                # Read welcome banner
                s.recv(1024)

                # Authenticate
                if self.control_password:
                    auth_cmd = f'AUTHENTICATE "{self.control_password}"\r\n'
                else:
                    auth_cmd = 'AUTHENTICATE\r\n'
                s.sendall(auth_cmd.encode())
                response = s.recv(1024).decode()

                if "250 OK" not in response:
                    logger.error(f"[tor] Authentication failed: {response.strip()}")
                    self._stats.refresh_failures += 1
                    return False

                # Request new circuit
                s.sendall(b"SIGNAL NEWNYM\r\n")
                response = s.recv(1024).decode()

                if "250 OK" in response:
                    self._last_refresh = now
                    self._requests_count = 0
                    self._status.last_refresh_time = now
                    self._status.requests_since_refresh = 0
                    self._stats.total_refreshes += 1
                    logger.info("[tor] Circuit refreshed - new exit IP will be used")
                    return True
                else:
                    logger.error(f"[tor] NEWNYM failed: {response.strip()}")
                    self._stats.refresh_failures += 1
                    return False

        except socket.timeout:
            logger.error("[tor] Control port timeout")
            self._stats.refresh_failures += 1
            return False
        except ConnectionRefusedError:
            logger.error("[tor] Control port refused connection - is Tor running?")
            self._stats.refresh_failures += 1
            return False
        except Exception as e:
            logger.error(f"[tor] Failed to refresh circuit: {e}")
            self._stats.refresh_failures += 1
            return False

    def on_request(self) -> bool:
        """
        Track request count, auto-refresh if threshold reached.

        Call this after each successful request through Tor.

        Returns:
            True if circuit was refreshed
        """
        self._requests_count += 1
        self._status.requests_since_refresh = self._requests_count
        self._stats.total_requests += 1

        if self._requests_count >= self.refresh_after_requests:
            logger.info(f"[tor] {self._requests_count} requests reached, refreshing circuit")
            if self.refresh_circuit():
                self._stats.refresh_on_interval += 1
                return True
        return False

    def on_rate_limit(self) -> bool:
        """
        Called when 429 rate limit detected - refresh circuit immediately.

        Returns:
            True if circuit was refreshed successfully
        """
        if not self.refresh_on_rate_limit_enabled:
            logger.debug("[tor] Rate limit refresh disabled")
            return False

        logger.info("[tor] Rate limit detected, refreshing circuit")
        if self.refresh_circuit():
            self._stats.refresh_on_rate_limit += 1
            return True
        return False

    def get_exit_ip(self) -> Optional[str]:
        """
        Get current Tor exit IP via check.torproject.org.

        Returns:
            Exit IP address or None if unavailable
        """
        try:
            import httpx
            response = httpx.get(
                "https://check.torproject.org/api/ip",
                proxy=self.get_proxy_url(),
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                ip = data.get("IP")
                self._status.exit_ip = ip
                self._stats.last_exit_ip = ip
                return ip
        except Exception as e:
            logger.debug(f"[tor] Could not get exit IP: {e}")
        return None

    def verify_tor_connection(self) -> bool:
        """
        Verify that traffic is actually going through Tor.

        Returns:
            True if check.torproject.org confirms Tor usage
        """
        try:
            import httpx
            response = httpx.get(
                "https://check.torproject.org/api/ip",
                proxy=self.get_proxy_url(),
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                is_tor = data.get("IsTor", False)
                if is_tor:
                    logger.info(f"[tor] Verified: Traffic going through Tor (IP: {data.get('IP')})")
                    self._status.circuit_established = True
                else:
                    logger.warning("[tor] WARNING: Traffic NOT going through Tor!")
                    self._status.circuit_established = False
                return is_tor
        except Exception as e:
            logger.error(f"[tor] Could not verify Tor connection: {e}")
        return False

    def get_status(self) -> TorStatus:
        """Get current Tor status."""
        return self._status

    def get_stats(self) -> dict:
        """Get Tor usage statistics as dict."""
        return {
            "total_requests": self._stats.total_requests,
            "total_refreshes": self._stats.total_refreshes,
            "refresh_on_rate_limit": self._stats.refresh_on_rate_limit,
            "refresh_on_interval": self._stats.refresh_on_interval,
            "refresh_failures": self._stats.refresh_failures,
            "last_exit_ip": self._stats.last_exit_ip,
            "requests_since_refresh": self._requests_count,
            "socks_port": self.socks_port,
            "control_port": self.control_port,
        }

    def __repr__(self) -> str:
        available = "available" if self.is_available() else "unavailable"
        return f"TorManager(socks={self.socks_port}, control={self.control_port}, {available})"


def get_tor_manager(config: "Config") -> Optional[TorManager]:
    """
    Factory function to create TorManager from config.

    Args:
        config: Application configuration

    Returns:
        TorManager instance or None if Tor is disabled
    """
    try:
        fallback_cfg = getattr(config.download, 'fallback', None)
        if not fallback_cfg:
            return None

        tor_cfg = getattr(fallback_cfg, 'tor', None)
        if not tor_cfg:
            return None

        # Check if enabled (handle both dict and dataclass)
        if isinstance(tor_cfg, dict):
            enabled = tor_cfg.get('enabled', False)
            if not enabled:
                return None
            return TorManager(
                socks_port=tor_cfg.get('socks_port', 9050),
                control_port=tor_cfg.get('control_port', 9051),
                control_password=tor_cfg.get('control_password', ''),
                refresh_after_requests=tor_cfg.get('refresh_after_requests', 50),
                min_refresh_interval=tor_cfg.get('min_refresh_interval', 10.0),
                refresh_on_rate_limit=tor_cfg.get('refresh_on_rate_limit', True),
            )
        else:
            if not getattr(tor_cfg, 'enabled', False):
                return None
            return TorManager(
                socks_port=getattr(tor_cfg, 'socks_port', 9050),
                control_port=getattr(tor_cfg, 'control_port', 9051),
                control_password=getattr(tor_cfg, 'control_password', ''),
                refresh_after_requests=getattr(tor_cfg, 'refresh_after_requests', 50),
                min_refresh_interval=getattr(tor_cfg, 'min_refresh_interval', 10.0),
                refresh_on_rate_limit=getattr(tor_cfg, 'refresh_on_rate_limit', True),
            )

    except Exception as e:
        logger.debug(f"[tor] Could not create TorManager from config: {e}")
        return None
