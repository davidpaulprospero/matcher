"""
VPN manager for IP rotation on rate limits.

Switches VPN servers when cookie rotation is exhausted to get a new
IP address. Requires a VPN client with CLI support.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from ..config.sections.download import VPNConfig

logger = logging.getLogger(__name__)


class VPNManager:
    """
    Manages VPN server switching for IP rotation.

    Features:
    - Execute VPN switch commands via subprocess
    - Connection verification after switch
    - Session-level switch limits
    - Configurable switch delay for connection establishment

    Usage:
        manager = VPNManager(config.download.vpn)

        if manager.can_switch():
            success = manager.switch()
            if success:
                # Continue downloading with new IP
    """

    def __init__(self, config: 'VPNConfig'):
        """
        Initialize the VPN manager.

        Args:
            config: VPNConfig with VPN settings
        """
        self.config = config
        self._switch_count = 0
        self._last_switch_time: Optional[float] = None

        if self.is_enabled:
            logger.info(
                f"VPN manager initialized, max switches: {config.max_switches_per_session}"
            )
        else:
            logger.debug("VPN manager disabled or no switch command configured")

    @property
    def is_enabled(self) -> bool:
        """Check if VPN switching is enabled and configured."""
        return self.config.enabled and bool(self.config.switch_command)

    @property
    def switch_count(self) -> int:
        """Get the number of VPN switches this session."""
        return self._switch_count

    def can_switch(self) -> bool:
        """
        Check if VPN switching is possible.

        Returns:
            True if VPN is enabled and under switch limit
        """
        if not self.is_enabled:
            return False

        # Check switch limit
        if self.config.max_switches_per_session > 0:
            if self._switch_count >= self.config.max_switches_per_session:
                logger.debug(
                    f"VPN switch limit reached ({self.config.max_switches_per_session})"
                )
                return False

        return True

    def switch(self) -> bool:
        """
        Switch to a new VPN server.

        Executes the configured switch command, waits for connection,
        and optionally verifies connectivity.

        Returns:
            True if switch was successful, False otherwise
        """
        if not self.can_switch():
            return False

        logger.info(f"Switching VPN server (attempt {self._switch_count + 1})...")

        try:
            # Execute switch command
            result = self._run_command(self.config.switch_command)

            if not result:
                logger.error("VPN switch command failed")
                return False

            self._switch_count += 1
            self._last_switch_time = time.time()

            # Wait for connection to establish
            if self.config.switch_delay_seconds > 0:
                logger.debug(f"Waiting {self.config.switch_delay_seconds}s for VPN connection...")
                time.sleep(self.config.switch_delay_seconds)

            # Verify connection if enabled (skip_verification overrides verify_connection)
            if self.config.verify_connection and not getattr(self.config, 'skip_verification', False):
                if not self._verify_connection():
                    logger.warning("VPN connection verification failed, but continuing...")
                    # Don't return False - the switch command succeeded
            elif getattr(self.config, 'skip_verification', False):
                logger.debug("VPN verification skipped (skip_verification=True)")

            logger.info(f"VPN switch successful (total: {self._switch_count})")
            return True

        except Exception as e:
            logger.error(f"VPN switch error: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from VPN.

        Executes the configured disconnect command if set.

        Returns:
            True if disconnect was successful or no command configured
        """
        if not self.config.disconnect_command:
            logger.debug("No disconnect command configured")
            return True

        logger.info("Disconnecting VPN...")

        try:
            result = self._run_command(self.config.disconnect_command)
            if result:
                logger.info("VPN disconnected")
            return result

        except Exception as e:
            logger.error(f"VPN disconnect error: {e}")
            return False

    def _run_command(self, command: str) -> bool:
        """
        Execute a shell command.

        Args:
            command: Command string to execute

        Returns:
            True if command succeeded (exit code 0)
        """
        logger.debug(f"Running command: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout for VPN commands
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode == 0:
                if result.stdout:
                    logger.debug(f"Command output: {result.stdout.strip()}")
                return True
            else:
                logger.error(
                    f"Command failed (exit {result.returncode}): "
                    f"{result.stderr.strip() if result.stderr else 'no error output'}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return False
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False

    def _verify_connection(self) -> bool:
        """
        Verify VPN connection is working.

        Attempts to connect to a reliable host to verify connectivity.
        Uses configured endpoints (verification_endpoint for HTTPS, verification_ip for ping).

        Returns:
            True if connection is verified
        """
        # Get configured endpoints (with defaults for backward compatibility)
        endpoint = getattr(self.config, 'verification_endpoint', 'https://www.google.com')
        ip_address = getattr(self.config, 'verification_ip', '8.8.8.8')

        logger.debug(f"Verifying VPN connection (endpoint={endpoint}, ip={ip_address})...")

        # Try multiple verification methods
        verification_commands = [
            # Try curl to configured endpoint
            f"curl -s --max-time 10 -o /dev/null -w '%{{http_code}}' {endpoint}",
            # Fallback: ping to configured IP (works on most systems)
            f"ping -c 1 -W 10 {ip_address}" if not self._is_windows() else f"ping -n 1 -w 10000 {ip_address}",
        ]

        for cmd in verification_commands:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.verify_timeout,
                    encoding='utf-8',
                    errors='replace'
                )

                if result.returncode == 0:
                    logger.debug("VPN connection verified")
                    return True

            except subprocess.TimeoutExpired:
                logger.debug(f"Verification command timed out: {cmd}")
                continue
            except Exception as e:
                logger.debug(f"Verification command failed: {cmd} - {e}")
                continue

        # Log failure with endpoints used for troubleshooting
        logger.warning(
            f"Could not verify VPN connection (tried endpoint={endpoint}, ip={ip_address})"
        )
        return False

    def _is_windows(self) -> bool:
        """Check if running on Windows."""
        import platform
        return platform.system().lower() == "windows"

    def reset(self) -> None:
        """Reset VPN manager state (clear switch count)."""
        self._switch_count = 0
        self._last_switch_time = None
        logger.info("VPN manager reset")

    def get_status(self) -> Dict:
        """
        Get current VPN manager status for logging/debugging.

        Returns:
            Dict with VPN status info
        """
        return {
            "enabled": self.is_enabled,
            "switch_count": self._switch_count,
            "switches": self._switch_count,  # Alias for compatibility with rate_limit_metrics
            "max_switches": self.config.max_switches_per_session,
            "can_switch": self.can_switch(),
            "last_switch_time": self._last_switch_time,
            "switch_command": self.config.switch_command[:50] + "..." if len(self.config.switch_command) > 50 else self.config.switch_command,
        }

    def to_checkpoint_state(self) -> Dict:
        """
        Serialize VPN manager state for checkpoint persistence.

        Saves switch_count and last_switch_timestamp so that the VPN manager
        can enforce max_switches_per_session across resume boundaries.

        Returns:
            Dict with serialized state:
                - switch_count: Number of switches made in session
                - last_switch_timestamp: ISO timestamp of last switch (or None)
        """
        return {
            "switch_count": self._switch_count,
            "last_switch_timestamp": (
                self._last_switch_time
                if self._last_switch_time is None
                else self._format_timestamp(self._last_switch_time)
            ),
        }

    @classmethod
    def from_checkpoint_state(cls, state: Dict, config: 'VPNConfig') -> 'VPNManager':
        """
        Create VpnManager from checkpoint state.

        Restores switch_count and last_switch_timestamp from a previous session.
        This allows max_switches_per_session to be enforced across resume boundaries.

        Args:
            state: Dict with switch_count and last_switch_timestamp
            config: VPNConfig instance

        Returns:
            VPNManager initialized with restored state
        """
        manager = cls(config)

        # Restore state from checkpoint
        if state:
            manager._switch_count = state.get("switch_count", 0)
            timestamp_str = state.get("last_switch_timestamp")
            if timestamp_str:
                manager._last_switch_time = cls._parse_timestamp(timestamp_str)
            else:
                manager._last_switch_time = None

        return manager

    def restore_from_checkpoint(self, state: Dict) -> None:
        """
        Restore state from checkpoint into existing VpnManager instance.

        Used when resuming from checkpoint to restore previous session's state.
        Logs the restoration status.

        Args:
            state: Dict with switch_count and last_switch_timestamp from checkpoint
        """
        if not state:
            return

        old_count = self._switch_count
        self._switch_count = state.get("switch_count", 0)

        timestamp_str = state.get("last_switch_timestamp")
        if timestamp_str:
            self._last_switch_time = self._parse_timestamp(timestamp_str)
        else:
            self._last_switch_time = None

        # Log restoration status
        max_switches = self.config.max_switches_per_session
        logger.info(
            f"Resuming VPN manager: {self._switch_count}/{max_switches} switches used"
        )

    @staticmethod
    def _format_timestamp(epoch_time: float) -> str:
        """Convert epoch timestamp to ISO format string."""
        from datetime import datetime
        return datetime.fromtimestamp(epoch_time).isoformat()

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> Optional[float]:
        """Parse ISO format timestamp to epoch time."""
        from datetime import datetime
        try:
            return datetime.fromisoformat(timestamp_str).timestamp()
        except (ValueError, TypeError):
            return None
