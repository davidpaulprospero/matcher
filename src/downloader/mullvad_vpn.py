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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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

# Region mapping for geographic proximity scoring (US-109-012)
REGION_MAP = {
    "us": "na",    # North America
    "ca": "na",
    "gb": "eu",    # Europe
    "de": "eu",
    "nl": "eu",
    "se": "eu",
    "ch": "eu",
    "au": "apac",  # Asia Pacific
    "jp": "apac",
    "sg": "apac",
}

# Default preferred countries from config
DEFAULT_PREFERRED_COUNTRIES = ["us", "gb", "de", "nl"]


@dataclass
class ServerSuccessRecord:
    """Tracks success/failure history for a specific server (country)."""
    country: str
    attempts: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        """Return success rate as a float between 0 and 1."""
        if self.attempts == 0:
            return 0.5  # Default neutral rate for untested servers
        return self.successes / self.attempts

    def record_success(self) -> None:
        """Record a successful download attempt."""
        self.attempts += 1
        self.successes += 1

    def record_failure(self) -> None:
        """Record a failed download attempt."""
        self.attempts += 1

    def should_swap(self, threshold: int = 3) -> bool:
        """Check if server should be swapped (3+ consecutive failures)."""
        if self.attempts == 0:
            return False
        consecutive_failures = self.attempts - self.successes
        return consecutive_failures >= threshold


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
            config: VPNConfig with VPN settings (typically MullvadConfig)
        """
        super().__init__(config)
        self._current_country: Optional[str] = None
        self._used_countries: List[str] = []
        self._last_verified_ip: Optional[str] = None
        # Initialize max rotations from config (MullvadConfig.max_rotations_per_session)
        self._max_rotations: int = getattr(config, 'max_rotations_per_session', 5)
        # Track last rotation time for cooldown enforcement (US-35-012)
        self._last_rotation_time: float = 0.0
        # Rotation delay cooldown from config (seconds between rotations)
        self._rotation_delay: float = float(getattr(config, 'rotation_delay_seconds', 5))
        # Exponential backoff for rotation delays (US-67-007)
        self._backoff_count: int = 0
        self._initial_rotation_delay: float = float(
            getattr(config, 'initial_rotation_delay_seconds', 5.0)
        )
        self._max_rotation_delay: float = float(
            getattr(config, 'max_rotation_delay_seconds', 60.0)
        )

        # Server success history tracking (US-109-012)
        self._server_history: Dict[str, ServerSuccessRecord] = {}
        self._current_server_record: Optional[ServerSuccessRecord] = None
        # Minimum success rate threshold for adaptive selection
        self._min_success_rate: float = getattr(config, 'min_server_success_rate', 0.7)
        # Consecutive failure threshold for swap recommendation
        self._swap_failure_threshold: int = getattr(config, 'swap_failure_threshold', 3)
        # Track region of last successful download for geographic proximity
        self._last_successful_region: Optional[str] = None

        # VPN degradation tracking (US-113-012)
        # Track consecutive VPN rotation failures for graceful degradation
        self._consecutive_vpn_failures: int = 0
        self._max_consecutive_vpn_failures: int = getattr(
            config, 'max_consecutive_vpn_failures', 5
        )
        # Manual override to continue with degraded mode
        self._degradation_override: bool = False
        # Whether VPN has been degraded to cookie-only mode
        self._is_degraded: bool = False

        # Latency-based server selection (US-114-008)
        # Track measured latency per country (ms)
        self._server_latency: Dict[str, float] = {}
        # Enable latency-based selection
        self._prefer_low_latency: bool = getattr(config, 'prefer_low_latency', True)
        # Maximum acceptable latency in ms (0 = no threshold)
        self._max_latency_ms: int = getattr(config, 'max_latency_ms', 200)
        # Timeout for latency measurement
        self._latency_timeout: float = getattr(config, 'latency_measurement_timeout', 3.0)

        if self.is_enabled:
            logger.info(
                f"Mullvad VPN manager initialized (max_rotations={self._max_rotations}, "
                f"rotation_delay={self._rotation_delay}s)"
            )

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
            True if rotation was successful, False if max rotations reached or failed
        """
        # Check max rotations limit before attempting rotation
        if self._switch_count >= self._max_rotations:
            logger.warning(
                f"Mullvad max rotations reached ({self._switch_count}/{self._max_rotations}). "
                "No more VPN rotations available this session."
            )
            return False

        # Check rotation cooldown (US-35-012)
        if self._rotation_delay > 0 and self._last_rotation_time > 0:
            elapsed = time.time() - self._last_rotation_time
            if elapsed < self._rotation_delay:
                remaining = self._rotation_delay - elapsed
                logger.warning(
                    f"Mullvad rotation cooldown active ({remaining:.1f}s remaining). "
                    f"Wait {self._rotation_delay}s between rotations."
                )
                return False

        if not self.can_switch():
            return False

        # Pick a country if not specified
        if country is None:
            country = self._pick_next_country()

        # Initialize server record for tracking (US-109-012)
        if country not in self._server_history:
            self._server_history[country] = ServerSuccessRecord(country)
        self._current_server_record = self._server_history[country]

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
            previous_country = self._current_country  # Save before updating
            self._switch_count += 1
            self._last_switch_time = time.time()
            self._last_rotation_time = time.time()  # For cooldown enforcement (US-35-012)
            self._current_country = country
            self._used_countries.append(country)

            # Wait with exponential backoff delay (US-67-007)
            backoff_delay = self._compute_backoff_delay()
            if backoff_delay > 0:
                logger.info(
                    f"Mullvad rotation backoff: waiting {backoff_delay:.1f}s "
                    f"(attempt {self._backoff_count}, base={self._initial_rotation_delay}s, "
                    f"cap={self._max_rotation_delay}s)"
                )
                time.sleep(backoff_delay)
            # Increment backoff counter after applying delay
            self._backoff_count += 1

            # Additional connection stabilization delay if configured
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

            # Record server switch event for metrics (US-143-006)
            self.record_server_switch_event(
                from_country=previous_country,
                to_country=country,
                reason="rotation",
            )

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
        Pick the next country for rotation using geographic proximity and adaptive selection.

        Uses geographic proximity scoring (US-109-012):
        1. Prefer servers in same region as previous successful downloads
        2. Use servers with >70% historical success rate first
        3. Avoid recently used countries when possible

        Latency-based selection (US-114-008):
        - When prefer_low_latency is enabled, measure latency to candidate servers
        - Prefer servers with latency below max_latency_ms threshold
        - Lower latency = higher score boost

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

        # Measure latency to available countries if enabled (US-114-008)
        if self._prefer_low_latency:
            for country in available:
                if country not in self._server_latency:
                    self.measure_latency(country)

        # Apply geographic proximity and adaptive selection scoring
        scored = self._score_countries_for_selection(available)

        # Return highest scoring country, or fall back to random
        if scored:
            return scored[0][0]  # Return country with highest score

        return random.choice(available) if available else country_pool[0]

    def _score_countries_for_selection(self, countries: List[str]) -> List[Tuple[str, float]]:
        """
        Score countries based on geographic proximity, historical success rate, and latency.

        Scoring factors (US-109-012):
        - Geographic proximity: +0.3 if same region as last successful download
        - Historical success rate: +0.5 if >70% success rate
        - Base score: 0.0

        Latency scoring (US-114-008):
        - Low latency boost: +0.4 if latency < max_latency_ms threshold
        - High latency penalty: -0.2 if latency > max_latency_ms but measurement exists
        - Unknown latency: no penalty (measurement will be attempted)

        Args:
            countries: List of candidate country codes

        Returns:
            List of (country, score) tuples sorted by score descending
        """
        scored = []

        for country in countries:
            score = 0.0

            # Geographic proximity boost (US-109-012)
            if self._last_successful_region:
                country_region = REGION_MAP.get(country, "other")
                if country_region == self._last_successful_region:
                    score += 0.3
                    logger.debug(f"Geographic proximity boost for {country}: +0.3 (region: {country_region})")

            # Historical success rate boost (US-109-012)
            record = self._server_history.get(country)
            if record and record.attempts > 0:
                if record.success_rate >= self._min_success_rate:
                    score += 0.5
                    logger.debug(
                        f"Success rate boost for {country}: +0.5 "
                        f"(rate: {record.success_rate:.1%}, attempts: {record.attempts})"
                    )
                elif record.success_rate < 0.3:
                    score -= 0.3  # Penalize poor performers
                    logger.debug(
                        f"Poor success rate penalty for {country}: -0.3 "
                        f"(rate: {record.success_rate:.1%}, attempts: {record.attempts})"
                    )

            # Latency-based scoring (US-114-008)
            if self._prefer_low_latency and self._max_latency_ms > 0:
                latency = self._server_latency.get(country)
                if latency is not None:
                    if latency <= self._max_latency_ms:
                        # Low latency - apply boost proportional to how low it is
                        # Max boost of +0.4 for very low latency (<50ms)
                        boost = 0.4 * (1 - min(latency / 50.0, 1.0))
                        score += boost
                        logger.debug(
                            f"Low latency boost for {country}: +{boost:.2f} "
                            f"(latency: {latency}ms, threshold: {self._max_latency_ms}ms)"
                        )
                    else:
                        # High latency - apply penalty
                        score -= 0.2
                        logger.debug(
                            f"High latency penalty for {country}: -0.2 "
                            f"(latency: {latency}ms, threshold: {self._max_latency_ms}ms)"
                        )

            scored.append((country, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Log top 3 choices for debugging
        if scored:
            logger.debug(f"Country selection scores: {scored[:3]}")

        return scored

    def measure_latency(self, country: str) -> Optional[float]:
        """
        Measure latency to a Mullvad server in the given country.

        Uses ping to measure round-trip time to a common server in that region.
        For Mullvad servers, we ping a server in the target country or use
        a nearby IP as a proxy.

        Args:
            country: 2-letter country code (e.g., "us", "de")

        Returns:
            Latency in milliseconds, or None if measurement failed
        """
        if not self._prefer_low_latency:
            return None

        # Check if we already have a recent measurement (within last 5 minutes)
        if country in self._server_latency:
            logger.debug(f"Using cached latency for {country}: {self._server_latency[country]}ms")
            return self._server_latency[country]

        # Map country to a ping target (Mullvad relay IPs or public IPs in region)
        ping_targets = self._get_ping_targets(country)

        for target in ping_targets:
            latency = self._ping_latency(target)
            if latency is not None:
                self._server_latency[country] = latency
                logger.info(f"Measured latency to {country}: {latency}ms (via {target})")
                return latency

        logger.warning(f"Could not measure latency for {country}")
        return None

    def _get_ping_targets(self, country: str) -> List[str]:
        """
        Get ping targets for a given country.

        Uses Mullvad relay IPs or public IPs as ping targets.

        Args:
            country: 2-letter country code

        Returns:
            List of IP addresses to ping
        """
        # Mullvad relay IPs by country (approximate - these change periodically)
        # Using public DNS servers in each region as reliable ping targets
        targets_by_country = {
            "us": ["8.8.8.8", "1.1.1.1"],  # US DNS
            "gb": ["8.8.8.8", "1.1.1.1"],  # UK can use US DNS as proxy
            "de": ["8.8.4.4", "1.1.1.1"],  # Germany
            "nl": ["8.8.4.4", "1.1.1.1"],  # Netherlands
            "se": ["8.8.4.4"],  # Sweden
            "ch": ["8.8.4.4"],  # Switzerland
            "ca": ["8.8.8.8"],  # Canada
            "au": ["1.1.1.1"],  # Australia
            "jp": ["1.1.1.1", "8.8.8.8"],  # Japan
            "sg": ["1.1.1.1"],  # Singapore
        }
        return targets_by_country.get(country, ["8.8.8.8", "1.1.1.1"])

    def _ping_latency(self, host: str) -> Optional[float]:
        """
        Measure latency to a host using ping.

        Args:
            host: IP address or hostname to ping

        Returns:
            Latency in milliseconds, or None if ping failed
        """
        import platform
        is_windows = platform.system().lower() == "windows"

        try:
            # Use ping with single attempt and short timeout
            if is_windows:
                cmd = ["ping", "-n", "1", "-w", str(int(self._latency_timeout * 1000)), host]
            else:
                cmd = ["ping", "-c", "1", "-W", str(int(self._latency_timeout)), host]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._latency_timeout + 1,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                return None

            # Parse latency from ping output
            # Windows: "Reply from x.x.x.x: bytes=32 time=XXms TTL=XX"
            # Linux: "time=XX.X ms"
            output = result.stdout

            if is_windows:
                # Windows format: time=XXms
                import re
                match = re.search(r'time[=<](\d+)ms', output, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            else:
                # Linux format: time=XX.X ms
                import re
                match = re.search(r'time[=<](\d+\.?\d*)\s*ms', output, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            return None

        except subprocess.TimeoutExpired:
            logger.debug(f"Ping timeout for {host}")
            return None
        except Exception as e:
            logger.debug(f"Ping error for {host}: {e}")
            return None

    def get_swap_recommendation(self) -> Optional[str]:
        """
        Check if current server should be swapped due to repeated failures.

        Returns recommendation (US-109-012):
        - None if current server is performing well
        - Country code of recommended alternative if current server failed 3+ times

        Returns:
            Recommended country code, or None if no swap needed
        """
        if not self._current_server_record:
            return None

        if self._current_server_record.should_swap(self._swap_failure_threshold):
            logger.warning(
                f"Server swap recommended: {self._current_country} has "
                f"{self._current_server_record.attempts - self._current_server_record.successes} "
                f"consecutive failures (threshold: {self._swap_failure_threshold})"
            )

            # Find best alternative in different region
            current_region = REGION_MAP.get(self._current_country, "other")
            alternatives = [
                c for c in MULLVAD_COUNTRIES
                if c != self._current_country and REGION_MAP.get(c, "other") != current_region
            ]

            if alternatives:
                # Score alternatives and return best
                scored = self._score_countries_for_selection(alternatives)
                if scored:
                    return scored[0][0]

            # Fallback to random alternative
            return self._pick_next_country()

        return None

    def _compute_backoff_delay(self) -> float:
        """
        Compute the exponential backoff delay for the current rotation.

        Formula: base_delay * 2^(rotation_count) capped at max_rotation_delay.
        First rotation (backoff_count=0): base_delay * 2^0 = base_delay (5s).

        Returns:
            Delay in seconds
        """
        delay = self._initial_rotation_delay * (2 ** self._backoff_count)
        return min(delay, self._max_rotation_delay)

    def reset_backoff(self) -> None:
        """
        Reset the exponential backoff counter after a successful download.

        Call this when a download succeeds after VPN rotation to indicate
        that the rotation was effective and future rotations should start
        with the base delay again.
        """
        if self._backoff_count > 0:
            logger.info(
                f"Mullvad backoff reset (was at count={self._backoff_count}, "
                f"delay would have been {self._compute_backoff_delay():.1f}s)"
            )
            self._backoff_count = 0

    def record_server_success(self, country: Optional[str] = None) -> None:
        """
        Record a successful download for the current or specified server.

        Updates server success history (US-109-012) for adaptive selection
        and geographic proximity tracking.

        Args:
            country: Country code to record success for (defaults to current)
        """
        target_country = country or self._current_country
        if not target_country:
            return

        # Get or create server record
        if target_country not in self._server_history:
            self._server_history[target_country] = ServerSuccessRecord(target_country)

        record = self._server_history[target_country]
        record.record_success()

        # Update current server record reference
        self._current_server_record = record

        # Update last successful region for geographic proximity
        region = REGION_MAP.get(target_country, "other")
        self._last_successful_region = region

        logger.debug(
            f"Recorded server success for {target_country}: "
            f"rate={record.success_rate:.1%}, attempts={record.attempts}"
        )

    def record_server_failure(self, country: Optional[str] = None) -> None:
        """
        Record a failed download for the current or specified server.

        Updates server failure history (US-109-012) for swap recommendations.

        Args:
            country: Country code to record failure for (defaults to current)
        """
        target_country = country or self._current_country
        if not target_country:
            return

        # Get or create server record
        if target_country not in self._server_history:
            self._server_history[target_country] = ServerSuccessRecord(target_country)

        record = self._server_history[target_country]
        record.record_failure()

        # Update current server record reference
        self._current_server_record = record

        logger.debug(
            f"Recorded server failure for {target_country}: "
            f"rate={record.success_rate:.1%}, attempts={record.attempts}, "
            f"failures={record.attempts - record.successes}"
        )

        # Check if swap is recommended
        if record.should_swap(self._swap_failure_threshold):
            recommendation = self.get_swap_recommendation()
            if recommendation:
                logger.warning(
                    f"Server {target_country} has failed {self._swap_failure_threshold}+ times. "
                    f"Consider rotating to {recommendation.upper()}"
                )

    def get_server_stats(self) -> Dict:
        """
        Get server statistics including success rates and latency.

        Returns:
            Dict with server history and statistics
        """
        history = {}
        for country, record in self._server_history.items():
            history[country] = {
                "attempts": record.attempts,
                "successes": record.successes,
                "success_rate": record.success_rate,
                "consecutive_failures": record.attempts - record.successes,
                "latency_ms": self._server_latency.get(country),
            }

        return {
            "current_country": self._current_country,
            "last_successful_region": self._last_successful_region,
            "server_history": history,
            "total_servers_tested": len(self._server_history),
            "server_latency": self._server_latency.copy(),
            "prefer_low_latency": self._prefer_low_latency,
            "max_latency_ms": self._max_latency_ms,
            # US-143-006: Server switch metrics
            "switch_events": self.get_switch_events(),
            "switch_metrics": self.get_switch_metrics(),
        }

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

    # ============== VPN Degradation (US-113-012) ==============

    def record_vpn_failure(self) -> None:
        """Record a VPN rotation failure for degradation tracking.

        Called when a VPN rotation attempt fails. Increments the consecutive
        failure counter and checks if degradation threshold is exceeded.
        """
        self._consecutive_vpn_failures += 1
        logger.warning(
            f"VPN rotation failure recorded: {self._consecutive_vpn_failures}/"
            f"{self._max_consecutive_vpn_failures} consecutive failures"
        )

        # Check if we should degrade
        if self._should_degrade():
            self._trigger_degradation()

    def record_vpn_success(self) -> None:
        """Record a successful VPN rotation or download.

        Resets the consecutive failure counter on success.
        """
        if self._consecutive_vpn_failures > 0:
            logger.info(
                f"VPN success recorded, resetting failure counter "
                f"(was {self._consecutive_vpn_failures})"
            )
        self._consecutive_vpn_failures = 0
        # Clear degraded state on success
        if self._is_degraded:
            self._is_degraded = False
            logger.info("VPN degraded state cleared due to successful operation")

    def _should_degrade(self) -> bool:
        """Check if VPN should degrade to cookie-only mode.

        Returns:
            True if consecutive failures exceed threshold and degradation not overridden.
        """
        if self._degradation_override:
            logger.info(
                "VPN degradation overridden by manual config, continuing with VPN"
            )
            return False
        if self._max_consecutive_vpn_failures <= 0:
            return False
        return self._consecutive_vpn_failures >= self._max_consecutive_vpn_failures

    def _trigger_degradation(self) -> None:
        """Trigger graceful degradation to cookie-only mode.

        Logs clear warning about VPN unavailability and sets degraded state.
        """
        self._is_degraded = True
        logger.warning(
            f"VPN DEGRADATION: Maximum consecutive failures exceeded "
            f"({self._consecutive_vpn_failures} >= {self._max_consecutive_vpn_failures}). "
            f"Degrading to cookie-only mode (Tier 3). VPN will be disabled for this session. "
            f"Use enable_vpn(override_degradation=True) to re-enable if needed."
        )

    def is_degraded(self) -> bool:
        """Check if VPN has been degraded to cookie-only mode.

        Returns:
            True if VPN is currently degraded.
        """
        return self._is_degraded

    def can_use_vpn(self) -> bool:
        """Check if VPN can be used for rotation.

        Returns False if:
        - VPN is disabled in config
        - Max rotations reached
        - Has been degraded (unless override is set)

        Returns:
            True if VPN rotation is available.
        """
        if not self.is_enabled:
            return False
        if self._switch_count >= self._max_rotations:
            return False
        if self._is_degraded and not self._degradation_override:
            return False
        return True

    def enable_vpn(self, override_degradation: bool = False) -> None:
        """Re-enable VPN usage, optionally overriding degradation state.

        Args:
            override_degradation: If True, re-enable VPN even if degraded.
                Use this for manual override to continue with VPN.
        """
        if override_degradation:
            self._degradation_override = True
            self._is_degraded = False
            self._consecutive_vpn_failures = 0
            logger.info(
                "VPN degradation override enabled - VPN re-activated manually"
            )
        else:
            self._degradation_override = False

    def disable_vpn_degradation_override(self) -> None:
        """Clear the manual degradation override.

        After calling this, if VPN is degraded, it will remain degraded.
        """
        was_override = self._degradation_override
        self._degradation_override = False
        if was_override:
            logger.info("VPN degradation override cleared")

    def get_degradation_status(self) -> Dict:
        """Get the current VPN degradation status.

        Returns:
            Dict with degradation state information:
                - is_degraded: bool
                - consecutive_failures: int
                - max_failures: int
                - override_enabled: bool
                - can_use_vpn: bool
        """
        return {
            "is_degraded": self._is_degraded,
            "consecutive_failures": self._consecutive_vpn_failures,
            "max_consecutive_vpn_failures": self._max_consecutive_vpn_failures,
            "degradation_override": self._degradation_override,
            "can_use_vpn": self.can_use_vpn(),
        }

    def reset_degradation(self) -> None:
        """Reset degradation state and failure counter.

        Use this to clear degradation after addressing the underlying issue.
        """
        self._consecutive_vpn_failures = 0
        self._is_degraded = False
        self._degradation_override = False
        logger.info("VPN degradation state fully reset")

    # ============== Checkpoint Persistence (US-129-007) ==============

    def to_checkpoint_state(self) -> Dict:
        """
        Serialize MullvadVPN state for checkpoint persistence.

        Overrides parent to include Mullvad-specific state:
        - switch_count: Number of rotations made
        - last_switch_timestamp: ISO timestamp of last rotation
        - last_rotation_time: Epoch timestamp for cooldown enforcement
        - backoff_count: Current backoff level
        - current_country: Current VPN exit country
        - used_countries: List of all used countries
        - consecutive_vpn_failures: Failure tracking for degradation
        - is_degraded: Degradation state

        Returns:
            Dict with serialized MullvadVPN state
        """
        # Get base state from parent
        state = {
            "type": "mullvad",
            "switch_count": self._switch_count,
            "last_switch_timestamp": (
                None
                if self._last_switch_time is None
                else self._format_timestamp(self._last_switch_time)
            ),
            "last_rotation_time": (
                None
                if self._last_rotation_time is None
                else self._format_timestamp(self._last_rotation_time)
            ),
            "backoff_count": self._backoff_count,
            "current_country": self._current_country,
            "used_countries": self._used_countries,
            "consecutive_vpn_failures": self._consecutive_vpn_failures,
            "is_degraded": self._is_degraded,
        }
        return state

    def restore_from_checkpoint(self, state: Dict) -> None:
        """
        Restore MullvadVPN state from checkpoint.

        Overrides parent to restore Mullvad-specific state.
        Used when resuming from checkpoint to continue session.

        Args:
            state: Dict with MullvadVPN state from checkpoint
        """
        if not state:
            return

        # Check if this is a MullvadVPN state (has type marker)
        if state.get("type") != "mullvad":
            logger.warning(
                f"MullvadVPN: Unexpected state type '{state.get('type')}', "
                f"expected 'mullvad'. Skipping restoration."
            )
            return

        # Restore switch count
        old_count = self._switch_count
        self._switch_count = state.get("switch_count", 0)

        # Restore last switch time
        timestamp_str = state.get("last_switch_timestamp")
        if timestamp_str:
            self._last_switch_time = self._parse_timestamp(timestamp_str)
        else:
            self._last_switch_time = None

        # Restore last rotation time (for cooldown)
        rotation_timestamp_str = state.get("last_rotation_time")
        if rotation_timestamp_str:
            self._last_rotation_time = self._parse_timestamp(rotation_timestamp_str)
        else:
            self._last_rotation_time = None

        # Restore backoff count
        self._backoff_count = state.get("backoff_count", 0)

        # Restore country state
        self._current_country = state.get("current_country")
        self._used_countries = state.get("used_countries", [])

        # Restore degradation state
        self._consecutive_vpn_failures = state.get("consecutive_vpn_failures", 0)
        self._is_degraded = state.get("is_degraded", False)

        # Log restoration
        logger.info(
            f"MullvadVPN restored from checkpoint: {old_count} -> {self._switch_count} rotations, "
            f"degraded={self._is_degraded}, country={self._current_country}"
        )

    # ============== Server Switch Metrics (US-143-006) ==============

    def record_server_switch_event(
        self,
        from_country: Optional[str],
        to_country: str,
        reason: str = "manual",
    ) -> None:
        """
        Record a server switch event for metrics tracking.

        Args:
            from_country: Previous country code (None if initial connection)
            to_country: New country code
            reason: Reason for switch (manual, rate_limit, failure, etc.)
        """
        import time as time_module

        event = {
            "timestamp": time_module.time(),
            "from_country": from_country,
            "to_country": to_country,
            "reason": reason,
            "total_switches": self._switch_count,
        }

        # Store in instance for retrieval
        if not hasattr(self, "_switch_events"):
            self._switch_events: List[Dict] = []
        self._switch_events.append(event)

        # Keep only last 100 events
        if len(self._switch_events) > 100:
            self._switch_events = self._switch_events[-100:]

        logger.info(
            f"VPN server switch event recorded: {from_country or 'initial'} -> {to_country} "
            f"(reason: {reason}, total: {self._switch_count})"
        )

    def get_switch_events(self) -> List[Dict]:
        """
        Get recorded server switch events.

        Returns:
            List of switch event dictionaries
        """
        return getattr(self, "_switch_events", []).copy()

    def get_switch_metrics(self) -> Dict:
        """
        Get server switch metrics summary.

        Returns:
            Dict with switch count and event summary
        """
        events = self.get_switch_events()

        # Count by reason
        by_reason: Dict[str, int] = {}
        by_country: Dict[str, int] = {}
        for event in events:
            reason = event.get("reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1

            to_country = event.get("to_country", "unknown")
            by_country[to_country] = by_country.get(to_country, 0) + 1

        return {
            "total_switches": self._switch_count,
            "max_rotations": self._max_rotations,
            "rotation_limit_reached": self._switch_count >= self._max_rotations,
            "events_count": len(events),
            "switches_by_reason": by_reason,
            "switches_by_country": by_country,
        }

    # ============== End Server Switch Metrics ==============
