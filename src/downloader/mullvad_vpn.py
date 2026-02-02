"""
Mullvad VPN manager for IP rotation on rate limits.

Extends VPNManager with Mullvad-specific CLI commands:
- `mullvad connect` / `mullvad disconnect`
- `mullvad relay set location` for server rotation
- `mullvad status` for connection status
- https://am.i.mullvad.net/json for verification
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from .vpn_manager import VPNManager

if TYPE_CHECKING:
    from ..config.sections.download import VPNConfig
    from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Mullvad relay locations for geographic rotation
# Format: country code (2-letter ISO)
MULLVAD_COUNTRIES = [
    "us",  # United States
    "gb",  # United Kingdom
    "de",  # Germany
    "nl",  # Netherlands
    "se",  # Sweden
    "ch",  # Switzerland
    "ca",  # Canada
    "au",  # Australia
    "jp",  # Japan
    "sg",  # Singapore
]


class MullvadVPN(VPNManager):
    """
    Mullvad-specific VPN manager with CLI integration.

    Extends VPNManager to use Mullvad's CLI commands directly:
    - connect/disconnect via `mullvad connect/disconnect`
    - server rotation via `mullvad relay set location`
    - status checking via `mullvad status`
    - verification via am.i.mullvad.net API

    Usage:
        manager = MullvadVPN(config.download.vpn)

        if manager.can_switch():
            success = manager.rotate_server()  # Switch to new country
            if success:
                # Continue downloading with new IP
    """

    @staticmethod
    def is_available() -> bool:
        """
        Check if Mullvad CLI is available on the system.

        Runs 'mullvad --version' to verify the CLI is installed and accessible.
        Should be called before instantiating MullvadVPN to avoid runtime errors.

        Returns:
            True if mullvad CLI is available and can be executed
        """
        try:
            result = subprocess.run(
                ["mullvad", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.debug(f"Mullvad CLI available: {version}")
                return True
            else:
                logger.debug(f"Mullvad CLI check failed: {result.stderr.strip()}")
                return False
        except FileNotFoundError:
            logger.debug("Mullvad CLI not found in PATH")
            return False
        except subprocess.TimeoutExpired:
            logger.debug("Mullvad CLI check timed out")
            return False
        except Exception as e:
            logger.debug(f"Mullvad CLI check error: {e}")
            return False

    def __init__(self, config: 'VPNConfig'):
        """
        Initialize Mullvad VPN manager.

        Args:
            config: VPNConfig with VPN settings
        """
        super().__init__(config)
        self._current_country: Optional[str] = None
        self._used_countries: List[str] = []
        self._last_verified_ip: Optional[str] = None

        if self.is_enabled:
            logger.info("Mullvad VPN manager initialized")

    def connect(self) -> bool:
        """
        Connect to Mullvad VPN.

        Uses `mullvad connect` CLI command.

        Returns:
            True if connection was successful
        """
        logger.info("Connecting to Mullvad VPN...")

        try:
            result = subprocess.run(
                ["mullvad", "connect"],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode == 0:
                logger.info("Mullvad connect command succeeded")
                # Wait for connection to establish
                time.sleep(self.config.switch_delay_seconds)
                return True
            else:
                logger.error(f"Mullvad connect failed: {result.stderr.strip()}")
                return False

        except FileNotFoundError:
            logger.error("Mullvad CLI not found. Is Mullvad VPN installed?")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Mullvad connect timed out")
            return False
        except Exception as e:
            logger.error(f"Mullvad connect error: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Mullvad VPN.

        Uses `mullvad disconnect` CLI command.

        Returns:
            True if disconnection was successful
        """
        logger.info("Disconnecting from Mullvad VPN...")

        try:
            result = subprocess.run(
                ["mullvad", "disconnect"],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode == 0:
                logger.info("Mullvad disconnected")
                self._current_country = None
                return True
            else:
                logger.error(f"Mullvad disconnect failed: {result.stderr.strip()}")
                return False

        except FileNotFoundError:
            logger.error("Mullvad CLI not found")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Mullvad disconnect timed out")
            return False
        except Exception as e:
            logger.error(f"Mullvad disconnect error: {e}")
            return False

    def rotate_server(
        self,
        country: Optional[str] = None,
        circuit_breaker: Optional['CircuitBreaker'] = None,
    ) -> bool:
        """
        Rotate to a new Mullvad server in a different country.

        Uses `mullvad relay set location <country>` to change server.
        If no country specified, picks a random country not recently used.

        Args:
            country: Optional 2-letter country code (e.g., "us", "de")
            circuit_breaker: Optional CircuitBreaker to reset on successful rotation.
                            When provided, the circuit breaker is reset after VPN
                            rotation since a new IP has fresh rate limit budget.

        Returns:
            True if rotation was successful
        """
        if not self.can_switch():
            return False

        # Pick a country if not specified
        if country is None:
            country = self._pick_next_country()

        logger.info(f"Rotating Mullvad to {country.upper()}...")

        try:
            # Set relay location
            result = subprocess.run(
                ["mullvad", "relay", "set", "location", country],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                logger.error(f"Mullvad relay set failed: {result.stderr.strip()}")
                return False

            # Reconnect to apply new location
            reconnect_result = subprocess.run(
                ["mullvad", "reconnect"],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )

            if reconnect_result.returncode != 0:
                logger.warning(f"Mullvad reconnect warning: {reconnect_result.stderr.strip()}")
                # Try connect as fallback
                if not self.connect():
                    return False

            # Update tracking
            self._switch_count += 1
            self._last_switch_time = time.time()
            self._current_country = country
            self._used_countries.append(country)

            # Wait for connection
            if self.config.switch_delay_seconds > 0:
                logger.debug(f"Waiting {self.config.switch_delay_seconds}s for VPN connection...")
                time.sleep(self.config.switch_delay_seconds)

            # Verify if enabled
            if self.config.verify_connection and not getattr(self.config, 'skip_verification', False):
                if not self.verify_connection():
                    logger.warning("Mullvad verification failed, but continuing...")

            # Reset circuit breaker if provided (new IP = fresh rate limit budget)
            if circuit_breaker is not None:
                circuit_breaker.reset()
                logger.info(
                    f"Circuit breaker reset due to VPN rotation to {country.upper()} "
                    f"(new IP has fresh rate limit budget)"
                )

            logger.info(f"Mullvad rotation successful to {country.upper()} (total: {self._switch_count})")
            return True

        except FileNotFoundError:
            logger.error("Mullvad CLI not found. Is Mullvad VPN installed?")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Mullvad rotation timed out")
            return False
        except Exception as e:
            logger.error(f"Mullvad rotation error: {e}")
            return False

    def _pick_next_country(self) -> str:
        """
        Pick the next country for rotation.

        Uses config.preferred_countries if set and non-empty,
        otherwise falls back to hardcoded MULLVAD_COUNTRIES list.
        Avoids recently used countries when possible.

        Returns:
            2-letter country code
        """
        # Use config.preferred_countries if available and non-empty,
        # otherwise fall back to hardcoded default list
        config_countries = getattr(self.config, 'preferred_countries', None)
        country_pool = config_countries if config_countries else MULLVAD_COUNTRIES

        # Get countries not recently used
        available = [c for c in country_pool if c not in self._used_countries[-3:]]

        # If all used recently, reset and use all
        if not available:
            self._used_countries = []
            available = list(country_pool)

        # Avoid current country if possible
        if self._current_country and self._current_country in available and len(available) > 1:
            available.remove(self._current_country)

        return random.choice(available)

    def get_status(self) -> Dict:
        """
        Get current Mullvad connection status.

        Parses output of `mullvad status` command.

        Returns:
            Dict with connection status info:
                - connected: bool
                - country: str or None
                - city: str or None
                - ip: str or None
                - raw_status: str (full status output)
        """
        status = {
            "connected": False,
            "country": None,
            "city": None,
            "ip": None,
            "raw_status": "",
        }

        try:
            result = subprocess.run(
                ["mullvad", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )

            output = result.stdout.strip()
            status["raw_status"] = output

            if result.returncode == 0 and output:
                # Parse status output
                # Example: "Connected to se-got-wg-001 in Gothenburg, Sweden"
                # Example: "Disconnected"
                if output.lower().startswith("connected"):
                    status["connected"] = True

                    # Try to parse location
                    if " in " in output:
                        location_part = output.split(" in ", 1)[1]
                        parts = location_part.split(", ")
                        if len(parts) >= 2:
                            status["city"] = parts[0]
                            status["country"] = parts[1]
                        elif len(parts) == 1:
                            status["country"] = parts[0]

            logger.debug(f"Mullvad status: {status}")

        except FileNotFoundError:
            logger.error("Mullvad CLI not found")
            status["raw_status"] = "Mullvad CLI not found"
        except subprocess.TimeoutExpired:
            logger.error("Mullvad status timed out")
            status["raw_status"] = "Status check timed out"
        except Exception as e:
            logger.error(f"Mullvad status error: {e}")
            status["raw_status"] = str(e)

        return status

    def verify_connection(self) -> bool:
        """
        Verify Mullvad VPN connection using am.i.mullvad.net API.

        Checks if traffic is going through Mullvad by calling their
        verification API which returns connection details.

        Returns:
            True if connected via Mullvad (mullvad_exit_ip is True)
        """
        logger.debug("Verifying Mullvad connection via am.i.mullvad.net...")

        try:
            # Use curl to fetch from am.i.mullvad.net/json
            result = subprocess.run(
                ["curl", "-s", "--max-time", "10", "https://am.i.mullvad.net/json"],
                capture_output=True,
                text=True,
                timeout=15,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                logger.warning(f"am.i.mullvad.net request failed: {result.stderr.strip()}")
                return self._fallback_verification()

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                logger.warning("Failed to parse am.i.mullvad.net response")
                return self._fallback_verification()

            # Check mullvad_exit_ip boolean field
            is_mullvad = data.get("mullvad_exit_ip", False)
            ip_address = data.get("ip", "unknown")
            country = data.get("country", "unknown")
            city = data.get("city", "unknown")

            self._last_verified_ip = ip_address

            if is_mullvad:
                logger.info(f"Mullvad verified: {ip_address} in {city}, {country}")
                return True
            else:
                logger.warning(f"Not connected via Mullvad. IP: {ip_address}")
                return False

        except FileNotFoundError:
            logger.warning("curl not found, using fallback verification")
            return self._fallback_verification()
        except subprocess.TimeoutExpired:
            logger.warning("am.i.mullvad.net request timed out")
            return self._fallback_verification()
        except Exception as e:
            logger.warning(f"Mullvad verification error: {e}")
            return self._fallback_verification()

    def _fallback_verification(self) -> bool:
        """
        Fallback verification using generic connectivity check.

        Used when am.i.mullvad.net is unreachable.

        Returns:
            True if basic connectivity works
        """
        logger.debug("Using fallback ping verification...")
        return super()._verify_connection()

    def switch(self) -> bool:
        """
        Switch VPN server (alias for rotate_server).

        Overrides base class switch() to use Mullvad-specific rotation.

        Returns:
            True if switch was successful
        """
        return self.rotate_server()

    def get_status_extended(self) -> Dict:
        """
        Get extended status including base class info.

        Returns:
            Dict with full VPN manager status
        """
        base_status = super().get_status()
        mullvad_status = self.get_status()

        return {
            **base_status,
            "mullvad_connected": mullvad_status["connected"],
            "mullvad_country": mullvad_status["country"],
            "mullvad_city": mullvad_status["city"],
            "current_country": self._current_country,
            "used_countries": self._used_countries.copy(),
            "last_verified_ip": self._last_verified_ip,
        }
