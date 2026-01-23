"""
VPN CLI manager for rotating IP addresses to bypass rate limits.

Supports multiple VPN providers via their CLI tools:
- NordVPN (nordvpn)
- Mullvad (mullvad)
- ProtonVPN (protonvpn-cli)
- OpenVPN (openvpn)
- WireGuard (wg-quick)

Usage:
    manager = VPNManager()

    # Connect to VPN
    if manager.connect():
        # Make requests...
        pass

    # Rotate to new server on rate limit
    if rate_limited:
        manager.rotate()

    # Disconnect when done
    manager.disconnect()
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .fallback_logging import FallbackLogger

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


# Windows-specific CLI paths for VPN providers
WINDOWS_CLI_PATHS = {
    "mullvad": [
        r"C:\Program Files\Mullvad VPN\resources\mullvad.exe",
        r"C:\Program Files (x86)\Mullvad VPN\resources\mullvad.exe",
    ],
    "nordvpn": [
        r"C:\Program Files\NordVPN\nordvpn.exe",
        r"C:\Program Files (x86)\NordVPN\nordvpn.exe",
    ],
}


def find_cli_path(command: str) -> Optional[str]:
    """Find CLI executable path, checking PATH and common Windows locations."""
    # First try PATH
    path = shutil.which(command)
    if path:
        return path

    # On Windows, check known installation paths
    if platform.system() == "Windows":
        paths = WINDOWS_CLI_PATHS.get(command, [])
        for p in paths:
            if Path(p).exists():
                return p

    return None


class VPNProvider(Enum):
    """Supported VPN providers."""
    NORDVPN = "nordvpn"
    MULLVAD = "mullvad"
    PROTONVPN = "protonvpn"
    OPENVPN = "openvpn"
    WIREGUARD = "wireguard"
    UNKNOWN = "unknown"


class VPNStatus(Enum):
    """VPN connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class VPNConnection:
    """Information about current VPN connection."""
    provider: VPNProvider
    status: VPNStatus
    server: str = ""
    country: str = ""
    city: str = ""
    ip_address: str = ""
    connected_at: float = 0.0
    protocol: str = ""

    @property
    def is_connected(self) -> bool:
        return self.status == VPNStatus.CONNECTED

    @property
    def duration_seconds(self) -> float:
        if self.connected_at > 0:
            return time.time() - self.connected_at
        return 0.0


@dataclass
class VPNProviderConfig:
    """Configuration for a VPN provider."""
    provider: VPNProvider
    cli_command: str
    connect_args: list[str] = field(default_factory=list)
    disconnect_args: list[str] = field(default_factory=list)
    status_args: list[str] = field(default_factory=list)
    rotate_args: list[str] = field(default_factory=list)
    preferred_countries: list[str] = field(default_factory=lambda: ["US", "UK", "CA", "DE", "NL"])
    socks5_port: int = 0  # If provider exposes SOCKS5


# Pre-configured provider settings
PROVIDER_CONFIGS = {
    VPNProvider.NORDVPN: VPNProviderConfig(
        provider=VPNProvider.NORDVPN,
        cli_command="nordvpn",
        connect_args=["connect"],
        disconnect_args=["disconnect"],
        status_args=["status"],
        rotate_args=["connect", "--group", "P2P"],
        socks5_port=1080,  # NordVPN SOCKS5 proxy
    ),
    VPNProvider.MULLVAD: VPNProviderConfig(
        provider=VPNProvider.MULLVAD,
        cli_command="mullvad",
        connect_args=["connect"],
        disconnect_args=["disconnect"],
        status_args=["status"],
        rotate_args=["relay", "set", "location"],
        socks5_port=1080,  # Mullvad SOCKS5
    ),
    VPNProvider.PROTONVPN: VPNProviderConfig(
        provider=VPNProvider.PROTONVPN,
        cli_command="protonvpn-cli",
        connect_args=["connect", "--fastest"],
        disconnect_args=["disconnect"],
        status_args=["status"],
        rotate_args=["connect", "--random"],
    ),
    VPNProvider.WIREGUARD: VPNProviderConfig(
        provider=VPNProvider.WIREGUARD,
        cli_command="wg-quick",
        connect_args=["up"],
        disconnect_args=["down"],
        status_args=["show"],  # Actually uses 'wg show'
        rotate_args=["up"],
    ),
}


class VPNManager:
    """
    Manages VPN connections via CLI for IP rotation.

    Auto-detects installed VPN clients and provides a unified interface
    for connecting, disconnecting, and rotating servers.

    Usage:
        manager = VPNManager()

        # Check what's available
        print(f"Detected VPN: {manager.detected_provider}")

        # Connect
        if manager.connect(country="US"):
            print(f"Connected to {manager.current_server}")

        # Rotate on rate limit
        manager.rotate()

        # Check status
        status = manager.get_status()
        print(f"Status: {status.status}, IP: {status.ip_address}")

        # Disconnect
        manager.disconnect()
    """

    def __init__(
        self,
        config: "Config | None" = None,
        preferred_provider: VPNProvider = None,
        preferred_countries: list[str] = None,
    ):
        self.config = config
        self.log = FallbackLogger("VPN_MANAGER")
        self._connection: Optional[VPNConnection] = None
        self._provider_config: Optional[VPNProviderConfig] = None
        self._enabled = True
        self._rotation_count = 0
        self._last_rotation = 0.0
        self._cooldown_seconds = 30.0  # Minimum time between rotations
        self._cli_path: Optional[str] = None  # Resolved CLI executable path

        # Override preferred countries if provided
        self._preferred_countries = preferred_countries or ["US", "UK", "CA", "DE", "NL"]

        # Load from config
        self._load_from_config()

        # Detect provider
        if preferred_provider:
            self._provider_config = PROVIDER_CONFIGS.get(preferred_provider)
        else:
            self._detect_provider()

    def _load_from_config(self) -> None:
        """Load VPN settings from config."""
        if not self.config:
            return

        try:
            fallback_cfg = getattr(self.config.download, 'fallback', None)
            if not fallback_cfg:
                return

            vpn_cfg = getattr(fallback_cfg, 'vpn', None)
            if not vpn_cfg:
                return

            if isinstance(vpn_cfg, dict):
                self._enabled = vpn_cfg.get('enabled', True)
                provider_name = vpn_cfg.get('provider', '')
                self._preferred_countries = vpn_cfg.get('preferred_countries', self._preferred_countries)
                self._cooldown_seconds = vpn_cfg.get('rotation_cooldown', 30.0)
            else:
                self._enabled = getattr(vpn_cfg, 'enabled', True)
                provider_name = getattr(vpn_cfg, 'provider', '')
                self._preferred_countries = getattr(vpn_cfg, 'preferred_countries', self._preferred_countries)
                self._cooldown_seconds = getattr(vpn_cfg, 'rotation_cooldown', 30.0)

            # Set preferred provider from config
            if provider_name:
                try:
                    provider = VPNProvider(provider_name.lower())
                    self._provider_config = PROVIDER_CONFIGS.get(provider)
                except ValueError:
                    self.log.warning(f"Unknown VPN provider: {provider_name}")

        except Exception as e:
            self.log.debug(f"Error loading VPN config: {e}")

    def _detect_provider(self) -> None:
        """Auto-detect installed VPN client."""
        self.log.debug("Auto-detecting VPN provider...")

        # Check each provider in order of preference
        detection_order = [
            VPNProvider.NORDVPN,
            VPNProvider.MULLVAD,
            VPNProvider.PROTONVPN,
            VPNProvider.WIREGUARD,
        ]

        for provider in detection_order:
            config = PROVIDER_CONFIGS.get(provider)
            if config and self._is_cli_available(config.cli_command):
                self._provider_config = config
                self.log.info(f"Detected VPN provider: {provider.value}")
                return

        self.log.debug("No VPN provider detected")

    def _is_cli_available(self, command: str) -> bool:
        """Check if a CLI command is available."""
        path = find_cli_path(command)
        if path:
            # Store the resolved path for later use
            self._cli_path = path
            return True
        return False

    @property
    def detected_provider(self) -> Optional[VPNProvider]:
        """Get the detected VPN provider."""
        if self._provider_config:
            return self._provider_config.provider
        return None

    @property
    def is_available(self) -> bool:
        """Check if VPN management is available."""
        return self._enabled and self._provider_config is not None

    @property
    def cli_command(self) -> str:
        """Get the CLI command/path to use."""
        # Use resolved path if available, otherwise fallback to config
        if self._cli_path:
            return self._cli_path
        if self._provider_config:
            return self.cli_command
        return ""

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to VPN."""
        status = self.get_status()
        return status.is_connected

    @property
    def current_server(self) -> str:
        """Get current server name."""
        if self._connection:
            return self._connection.server
        return ""

    @property
    def socks5_proxy(self) -> Optional[str]:
        """Get SOCKS5 proxy URL if available."""
        if self._provider_config and self._provider_config.socks5_port:
            return f"socks5://127.0.0.1:{self._provider_config.socks5_port}"
        return None

    def connect(self, country: str = None, server: str = None) -> bool:
        """
        Connect to VPN.

        Args:
            country: Country code (e.g., "US", "UK")
            server: Specific server name (provider-dependent)

        Returns:
            True if connection successful
        """
        if not self._provider_config:
            self.log.warning("No VPN provider available")
            return False

        self.log.start_operation("connect", "vpn")
        self.log.info(f"Connecting to VPN ({self._provider_config.provider.value})...")

        try:
            provider = self._provider_config.provider

            # Mullvad-specific: set location separately, then connect
            if provider == VPNProvider.MULLVAD:
                target_country = country or (self._preferred_countries[0] if self._preferred_countries else None)
                if target_country:
                    set_location_cmd = [self.cli_command, "relay", "set", "location", target_country.lower()]
                    self.log.debug(f"Setting location: {' '.join(set_location_cmd)}")
                    subprocess.run(set_location_cmd, capture_output=True, text=True, timeout=30)

                # Connect without arguments
                cmd = [self.cli_command, "connect"]
            else:
                cmd = [self.cli_command]
                cmd.extend(self._provider_config.connect_args)

                # Add country/server if specified (for other providers)
                if server:
                    cmd.append(server)
                elif country:
                    cmd.append(country)
                elif self._preferred_countries:
                    # Use first preferred country
                    cmd.append(self._preferred_countries[0])

            self.log.debug(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                self._connection = VPNConnection(
                    provider=self._provider_config.provider,
                    status=VPNStatus.CONNECTED,
                    server=server or country or "",
                    country=country or "",
                    connected_at=time.time(),
                )
                self.log.success(f"Connected to VPN")
                return True
            else:
                self.log.failure("Connection failed", reason=result.stderr[:100] if result.stderr else "Unknown")
                return False

        except subprocess.TimeoutExpired:
            self.log.warning("Connection timeout")
            return False
        except Exception as e:
            self.log.error("Connection error", error=e)
            return False

    def disconnect(self) -> bool:
        """Disconnect from VPN."""
        if not self._provider_config:
            return False

        self.log.start_operation("disconnect", "vpn")
        self.log.info("Disconnecting from VPN...")

        try:
            cmd = [self.cli_command]
            cmd.extend(self._provider_config.disconnect_args)

            self.log.debug(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self._connection = None
                self.log.success("Disconnected from VPN")
                return True
            else:
                self.log.warning(f"Disconnect may have failed: {result.stderr[:50]}")
                return False

        except Exception as e:
            self.log.error("Disconnect error", error=e)
            return False

    def rotate(self, country: str = None) -> bool:
        """
        Rotate to a new VPN server.

        This disconnects and reconnects to get a new IP address.

        Args:
            country: Specific country to connect to

        Returns:
            True if rotation successful
        """
        if not self._provider_config:
            self.log.warning("No VPN provider available for rotation")
            return False

        # Check cooldown
        now = time.time()
        if now - self._last_rotation < self._cooldown_seconds:
            wait_time = self._cooldown_seconds - (now - self._last_rotation)
            self.log.debug(f"Rotation cooldown, waiting {wait_time:.0f}s")
            time.sleep(wait_time)

        self._rotation_count += 1
        self._last_rotation = time.time()

        self.log.start_operation("rotate", "vpn")
        self.log.info(f"Rotating VPN server (rotation #{self._rotation_count})...")

        # Get next country from preferred list
        if not country and self._preferred_countries:
            idx = self._rotation_count % len(self._preferred_countries)
            country = self._preferred_countries[idx]

        try:
            # Provider-specific rotation logic
            provider = self._provider_config.provider

            if provider == VPNProvider.MULLVAD:
                # Mullvad: set location first, then reconnect
                if country:
                    set_location_cmd = [self.cli_command, "relay", "set", "location", country.lower()]
                    self.log.debug(f"Setting location: {' '.join(set_location_cmd)}")
                    subprocess.run(set_location_cmd, capture_output=True, text=True, timeout=30)

                # Reconnect to apply new location / get new server
                reconnect_cmd = [self.cli_command, "reconnect"]
                self.log.debug(f"Reconnecting: {' '.join(reconnect_cmd)}")
                result = subprocess.run(reconnect_cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    # Wait for connection to establish
                    time.sleep(3)
                    self._connection = VPNConnection(
                        provider=self._provider_config.provider,
                        status=VPNStatus.CONNECTED,
                        country=country or "",
                        connected_at=time.time(),
                    )
                    self.log.success(f"Rotated to new server (country: {country or 'auto'})")
                    return True

            # Method 1: Use provider's rotate command if available
            elif self._provider_config.rotate_args:
                cmd = [self.cli_command]
                cmd.extend(self._provider_config.rotate_args)
                if country:
                    cmd.append(country.lower())

                self.log.debug(f"Executing: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    self._connection = VPNConnection(
                        provider=self._provider_config.provider,
                        status=VPNStatus.CONNECTED,
                        country=country or "",
                        connected_at=time.time(),
                    )
                    self.log.success(f"Rotated to new server (country: {country or 'auto'})")
                    return True

            # Method 2: Disconnect and reconnect
            self.log.debug("Using disconnect/reconnect method")
            self.disconnect()
            time.sleep(2)  # Brief pause
            return self.connect(country=country)

        except Exception as e:
            self.log.error("Rotation error", error=e)
            return False

    def get_status(self) -> VPNConnection:
        """Get current VPN connection status."""
        if not self._provider_config:
            return VPNConnection(
                provider=VPNProvider.UNKNOWN,
                status=VPNStatus.UNKNOWN,
            )

        try:
            cmd = [self.cli_command]
            cmd.extend(self._provider_config.status_args)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return self._parse_status(result.stdout)
            else:
                return VPNConnection(
                    provider=self._provider_config.provider,
                    status=VPNStatus.DISCONNECTED,
                )

        except Exception as e:
            self.log.debug(f"Status check error: {e}")
            return VPNConnection(
                provider=self._provider_config.provider if self._provider_config else VPNProvider.UNKNOWN,
                status=VPNStatus.ERROR,
            )

    def _parse_status(self, output: str) -> VPNConnection:
        """Parse VPN status output."""
        provider = self._provider_config.provider if self._provider_config else VPNProvider.UNKNOWN
        output_lower = output.lower()

        # Check for disconnected first (more specific)
        disconnected_patterns = [
            "disconnected",
            "status: disconnected",
            "not connected",
            "connection: inactive",
            "tunnel: stopped",
        ]

        is_disconnected = any(pattern in output_lower for pattern in disconnected_patterns)

        # Common patterns for connected status
        connected_patterns = [
            "status: connected",
            "connection: active",
            "tunnel: running",
        ]

        # Only check connected if not explicitly disconnected
        is_connected = not is_disconnected and (
            any(pattern in output_lower for pattern in connected_patterns) or
            ("connected" in output_lower and "disconnected" not in output_lower)
        )

        connection = VPNConnection(
            provider=provider,
            status=VPNStatus.CONNECTED if is_connected else VPNStatus.DISCONNECTED,
        )

        # Try to extract server/country from output
        if provider == VPNProvider.NORDVPN:
            # NordVPN: "Current server: us1234.nordvpn.com"
            import re
            server_match = re.search(r'current server:\s*(\S+)', output_lower)
            if server_match:
                connection.server = server_match.group(1)
            country_match = re.search(r'country:\s*(\w+)', output_lower)
            if country_match:
                connection.country = country_match.group(1)

        elif provider == VPNProvider.MULLVAD:
            # Mullvad: "Connected to se-got-wg-001"
            import re
            server_match = re.search(r'connected to\s+(\S+)', output_lower)
            if server_match:
                connection.server = server_match.group(1)

        return connection

    def get_current_ip(self) -> Optional[str]:
        """Get current public IP address."""
        import httpx

        ip_services = [
            "https://api.ipify.org",
            "https://icanhazip.com",
            "https://checkip.amazonaws.com",
        ]

        for service in ip_services:
            try:
                response = httpx.get(service, timeout=10)
                if response.status_code == 200:
                    return response.text.strip()
            except Exception:
                continue

        return None

    def get_stats(self) -> dict:
        """Get VPN manager statistics."""
        status = self.get_status()
        return {
            "provider": self.detected_provider.value if self.detected_provider else None,
            "available": self.is_available,
            "connected": status.is_connected,
            "server": status.server,
            "country": status.country,
            "rotation_count": self._rotation_count,
            "socks5_proxy": self.socks5_proxy,
        }


class VPNRotationHelper:
    """
    Helper class for integrating VPN rotation with the fallback system.

    Automatically rotates VPN on rate limits and tracks effectiveness.

    Usage:
        helper = VPNRotationHelper(vpn_manager)

        # In your request loop
        response = make_request()
        if response.status_code == 429:
            if helper.should_rotate():
                helper.rotate_on_rate_limit()
    """

    def __init__(
        self,
        vpn_manager: VPNManager,
        max_rotations_per_hour: int = 10,
        min_rotation_interval: float = 60.0,
    ):
        self.vpn = vpn_manager
        self.max_rotations_per_hour = max_rotations_per_hour
        self.min_rotation_interval = min_rotation_interval
        self.log = FallbackLogger("VPN_ROTATION")

        self._rotation_times: list[float] = []
        self._last_rotation = 0.0
        self._rate_limits_since_rotation = 0

    def should_rotate(self) -> bool:
        """Check if we should rotate VPN based on rate limits."""
        if not self.vpn.is_available:
            return False

        now = time.time()

        # Check minimum interval
        if now - self._last_rotation < self.min_rotation_interval:
            return False

        # Check hourly limit
        one_hour_ago = now - 3600
        recent_rotations = sum(1 for t in self._rotation_times if t > one_hour_ago)
        if recent_rotations >= self.max_rotations_per_hour:
            self.log.warning(f"Rotation limit reached ({recent_rotations}/hr)")
            return False

        return True

    def rotate_on_rate_limit(self) -> bool:
        """Rotate VPN after encountering rate limit."""
        self._rate_limits_since_rotation += 1

        if not self.should_rotate():
            return False

        self.log.info(f"Rate limit detected, rotating VPN (rate limits since last: {self._rate_limits_since_rotation})")

        success = self.vpn.rotate()

        if success:
            now = time.time()
            self._rotation_times.append(now)
            self._last_rotation = now
            self._rate_limits_since_rotation = 0

            # Clean up old rotation times
            one_hour_ago = now - 3600
            self._rotation_times = [t for t in self._rotation_times if t > one_hour_ago]

        return success

    def report_success(self) -> None:
        """Report successful request (no rate limit)."""
        # Could be used to track effectiveness
        pass

    def get_stats(self) -> dict:
        """Get rotation statistics."""
        now = time.time()
        one_hour_ago = now - 3600
        recent_rotations = sum(1 for t in self._rotation_times if t > one_hour_ago)

        return {
            "rotations_this_hour": recent_rotations,
            "max_rotations_per_hour": self.max_rotations_per_hour,
            "rate_limits_since_rotation": self._rate_limits_since_rotation,
            "last_rotation_ago": now - self._last_rotation if self._last_rotation > 0 else None,
            "vpn_stats": self.vpn.get_stats(),
        }


# Provider-specific helper functions

def nordvpn_connect(country: str = "US", group: str = None) -> bool:
    """Quick connect via NordVPN."""
    cmd = ["nordvpn", "connect"]
    if group:
        cmd.extend(["--group", group])
    cmd.append(country)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


def mullvad_set_location(country: str, city: str = None) -> bool:
    """Set Mullvad relay location."""
    cmd = ["mullvad", "relay", "set", "location", country.lower()]
    if city:
        cmd.append(city.lower())

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False


def protonvpn_random() -> bool:
    """Connect to random ProtonVPN server."""
    try:
        result = subprocess.run(
            ["protonvpn-cli", "connect", "--random"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


# Detect available VPN on module load
def detect_vpn() -> Optional[VPNProvider]:
    """Detect which VPN CLI is available."""
    for provider, config in PROVIDER_CONFIGS.items():
        if shutil.which(config.cli_command):
            return provider
    return None
