"""
Cookie rotation for YouTube rate limit evasion.

Rotates between multiple cookie files when hitting rate limits or
authentication errors. Each cookie file should be exported from a
different browser profile or YouTube account.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..config.sections.download import CookieRotationConfig

logger = logging.getLogger(__name__)


class CookieRotator:
    """
    Manages rotation between multiple cookie files for YouTube downloads.

    Features:
    - Multiple rotation strategies (on_error, round_robin, random)
    - Cooldown tracking to avoid reusing rate-limited cookies too soon
    - Error pattern matching for automatic rotation triggers
    - Session-level rotation limits
    - Startup validation of cookie file existence and readability

    Usage:
        rotator = CookieRotator(config.download.cookie_rotation)
        cookie_path = rotator.get_current_cookie()

        # On error:
        if rotator.should_rotate(error_message):
            new_cookie = rotator.rotate()
    """

    def __init__(self, config: 'CookieRotationConfig', continue_on_partial: bool = True):
        """
        Initialize the cookie rotator.

        Args:
            config: CookieRotationConfig with rotation settings
            continue_on_partial: If True (default), allow startup with some invalid cookies.
                                 If False, raise ValueError when any cookie is invalid.
        """
        self.config = config
        self._current_index = 0
        self._rotation_count = 0
        self._continue_on_partial = continue_on_partial

        # Track failed cookies with timestamps for cooldown
        # {cookie_path: timestamp_when_failed}
        self._failed_cookies: Dict[str, float] = {}

        # Validate and filter to existing AND readable cookie files
        self._cookie_files, self._invalid_cookies = self._validate_cookie_files(
            config.cookie_files
        )

        # Log summary of validation
        total = len(config.cookie_files)
        valid = len(self._cookie_files)
        invalid = len(self._invalid_cookies)

        if self._cookie_files:
            logger.info(
                f"Cookie rotator initialized with {valid}/{total} valid cookies, "
                f"strategy: {config.rotation_strategy}"
            )
            if invalid > 0:
                logger.warning(f"{invalid} cookie file(s) failed validation")
        else:
            logger.warning("Cookie rotator has no valid cookie files!")

        # Raise error if continue_on_partial=False and some cookies are invalid
        if not continue_on_partial and invalid > 0:
            reasons = "; ".join([f"{p}: {r}" for p, r in self._invalid_cookies.items()])
            raise ValueError(f"Cookie validation failed: {reasons}")

    def _validate_cookie_files(
        self, cookie_files: List[str]
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Validate cookie file paths: check existence AND readability.

        Args:
            cookie_files: List of cookie file paths from config

        Returns:
            Tuple of:
            - List of validated, readable cookie file paths
            - Dict of invalid paths mapped to failure reasons
        """
        valid_files: List[str] = []
        invalid_files: Dict[str, str] = {}

        for cookie_path in cookie_files:
            path = Path(cookie_path)

            # Check existence
            if not path.exists():
                reason = "file not found"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
                continue

            # Check if it's a file (not directory)
            if not path.is_file():
                reason = "not a file"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
                continue

            # Check readability
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    # Read first 100 bytes to verify readability
                    f.read(100)
                valid_files.append(str(path))
                logger.debug(f"Cookie file validated: {cookie_path}")
            except PermissionError:
                reason = "permission denied"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
            except UnicodeDecodeError:
                # Cookie files should be plain text, but try binary read as fallback
                try:
                    with open(path, 'rb') as f:
                        f.read(100)
                    valid_files.append(str(path))
                    logger.debug(f"Cookie file validated (binary): {cookie_path}")
                except (OSError, IOError) as e:
                    reason = f"read error: {e}"
                    invalid_files[cookie_path] = reason
                    logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
            except (OSError, IOError) as e:
                reason = f"read error: {e}"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")

        return valid_files, invalid_files

    @property
    def is_enabled(self) -> bool:
        """Check if rotation is enabled and has valid cookies."""
        return self.config.enabled and len(self._cookie_files) > 0

    @property
    def available_cookies(self) -> int:
        """Get count of cookies not in cooldown."""
        return sum(1 for cf in self._cookie_files if self.is_available(cf))

    def get_current_cookie(self) -> Optional[str]:
        """
        Get the current active cookie file path.

        Returns:
            Path to current cookie file, or None if no valid cookies
        """
        if not self._cookie_files:
            return None

        # Ensure index is valid
        if self._current_index >= len(self._cookie_files):
            self._current_index = 0

        current = self._cookie_files[self._current_index]

        # If current cookie is in cooldown, try to find an available one
        if not self.is_available(current):
            available = self._find_available_cookie()
            if available:
                self._current_index = self._cookie_files.index(available)
                return available
            # All cookies in cooldown - return current anyway (will fail but that's expected)
            logger.warning("All cookies in cooldown, using current cookie anyway")

        return current

    def _find_available_cookie(self) -> Optional[str]:
        """Find the first cookie not in cooldown."""
        for cookie_path in self._cookie_files:
            if self.is_available(cookie_path):
                return cookie_path
        return None

    def is_available(self, cookie_path: str) -> bool:
        """
        Check if a cookie is available (not in cooldown).

        Args:
            cookie_path: Path to cookie file

        Returns:
            True if cookie can be used, False if in cooldown
        """
        if cookie_path not in self._failed_cookies:
            return True

        failed_time = self._failed_cookies[cookie_path]
        elapsed = time.time() - failed_time

        if elapsed >= self.config.cooldown_seconds:
            # Cooldown expired, remove from failed list
            del self._failed_cookies[cookie_path]
            logger.debug(f"Cookie cooldown expired: {cookie_path}")
            return True

        remaining = self.config.cooldown_seconds - elapsed
        logger.debug(f"Cookie in cooldown ({remaining:.0f}s remaining): {cookie_path}")
        return False

    def should_rotate(self, error_message: str) -> bool:
        """
        Check if the error message should trigger cookie rotation.

        Args:
            error_message: Error string from yt-dlp or download process

        Returns:
            True if error matches rotation triggers
        """
        if not self.is_enabled:
            return False

        error_lower = error_message.lower()

        for pattern in self.config.rotate_on_errors:
            if pattern.lower() in error_lower:
                logger.debug(f"Error matches rotation trigger: {pattern}")
                return True

        return False

    def rotate(self) -> Optional[str]:
        """
        Rotate to the next cookie file.

        Marks current cookie as failed (enters cooldown) and switches
        to the next available cookie based on rotation strategy.

        Returns:
            Path to new cookie file, or None if rotation exhausted
        """
        if not self._cookie_files:
            return None

        # Check rotation limit
        if self.config.max_rotations_per_session > 0:
            if self._rotation_count >= self.config.max_rotations_per_session:
                logger.warning(
                    f"Max rotations reached ({self.config.max_rotations_per_session}), "
                    "cannot rotate further"
                )
                return None

        # Mark current cookie as failed
        current = self.get_current_cookie()
        if current:
            self.mark_failed(current)

        # Select next cookie based on strategy
        new_cookie = self._select_next_cookie()

        if new_cookie:
            self._rotation_count += 1
            logger.info(
                f"Rotated cookie ({self._rotation_count}): {Path(current).name if current else 'none'} "
                f"-> {Path(new_cookie).name}"
            )

        return new_cookie

    def _select_next_cookie(self) -> Optional[str]:
        """
        Select the next cookie based on rotation strategy.

        Returns:
            Path to next cookie, or None if all exhausted
        """
        strategy = self.config.rotation_strategy

        if strategy == "random":
            return self._select_random()
        elif strategy == "round_robin":
            return self._select_round_robin()
        else:  # "on_error" - same as round_robin
            return self._select_round_robin()

    def _select_round_robin(self) -> Optional[str]:
        """Select next available cookie in sequence."""
        start_index = self._current_index
        tried = 0

        while tried < len(self._cookie_files):
            self._current_index = (self._current_index + 1) % len(self._cookie_files)
            candidate = self._cookie_files[self._current_index]

            if self.is_available(candidate):
                return candidate

            tried += 1

        # All cookies exhausted
        logger.warning("All cookies in cooldown or exhausted")
        return None

    def _select_random(self) -> Optional[str]:
        """Select a random available cookie."""
        available = [cf for cf in self._cookie_files if self.is_available(cf)]

        if not available:
            logger.warning("No available cookies for random selection")
            return None

        selected = random.choice(available)
        self._current_index = self._cookie_files.index(selected)
        return selected

    def mark_failed(self, cookie_path: str) -> None:
        """
        Mark a cookie as failed (enters cooldown).

        Args:
            cookie_path: Path to cookie file that failed
        """
        self._failed_cookies[cookie_path] = time.time()
        logger.debug(f"Cookie marked failed (cooldown {self.config.cooldown_seconds}s): {cookie_path}")

    def reset(self) -> None:
        """Reset rotation state (clear cooldowns and counters)."""
        self._current_index = 0
        self._rotation_count = 0
        self._failed_cookies.clear()
        logger.info("Cookie rotator reset")

    def get_status(self) -> Dict:
        """
        Get current rotation status for logging/debugging.

        Returns:
            Dict with rotation status info including:
            - total_cookies: All configured cookie files (valid + invalid)
            - valid_cookies: Cookie files that passed validation (exist + readable)
            - available_cookies: Valid cookies not currently in cooldown
            - invalid_cookies: Dict mapping invalid paths to failure reasons
        """
        return {
            "enabled": self.is_enabled,
            "total_cookies": len(self._cookie_files) + len(self._invalid_cookies),
            "valid_cookies": len(self._cookie_files),
            "available_cookies": self.available_cookies,
            "current_cookie": self.get_current_cookie(),
            "rotation_count": self._rotation_count,
            "max_rotations": self.config.max_rotations_per_session,
            "strategy": self.config.rotation_strategy,
            "cookies_in_cooldown": list(self._failed_cookies.keys()),
            "invalid_cookies": dict(self._invalid_cookies),
        }

    def can_rotate(self) -> bool:
        """
        Check if rotation is possible.

        Returns:
            True if there are available cookies to rotate to
        """
        if not self.is_enabled:
            return False

        # Check rotation limit
        if self.config.max_rotations_per_session > 0:
            if self._rotation_count >= self.config.max_rotations_per_session:
                return False

        # Check if there are available cookies (besides current)
        return self.available_cookies > 1 or (
            self.available_cookies == 1 and
            not self.is_available(self.get_current_cookie() or "")
        )
