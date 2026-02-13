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
    - Health tracking: success/failure rates per cookie for intelligent prioritization
    - Cooldown tracking to avoid reusing rate-limited cookies too soon
    - Error pattern matching for automatic rotation triggers
    - Session-level rotation limits
    - Startup validation of cookie file existence and readability

    Usage:
        rotator = CookieRotator(config.download.cookie_rotation)
        cookie_path = rotator.get_current_cookie()

        # On success:
        rotator.mark_success(cookie_path)

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

        # Health tracking: success/failure counts per cookie (US-93-010)
        # {cookie_path: {"success": int, "failure": int}}
        self._cookie_health: Dict[str, Dict[str, int]] = {}

        # Minimum attempts before evaluating health (prevents premature removal)
        self._health_min_attempts = getattr(config, 'health_min_attempts', 5)

        # Success rate threshold for auto-removal (US-93-010)
        # Cookies with success rate below this threshold after min_attempts will be removed
        self._success_rate_threshold = getattr(config, 'success_rate_threshold', 0.3)

        # Maximum consecutive failures before auto-removal
        self._max_consecutive_failures = getattr(config, 'max_consecutive_failures', 3)

        # Validate and filter to existing AND readable cookie files
        self._cookie_files, self._invalid_cookies = self._validate_cookie_files(
            config.cookie_files
        )

        # Log summary of validation
        total = len(config.cookie_files)
        valid = len(self._cookie_files)
        invalid = len(self._invalid_cookies)

        logger.info(
            f"Cookie rotator: {valid}/{total} cookies valid ({invalid} invalid)"
        )

        if invalid > 0 and self._continue_on_partial and self._invalid_cookies:
            reasons = "; ".join(
                f"{Path(p).name}: {r}" for p, r in self._invalid_cookies.items()
            )
            logger.warning(
                f"{invalid} cookie file(s) failed validation: {reasons}"
            )

        if not self._cookie_files:
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

    def _is_cookie_file_valid(self, cookie_path: str) -> bool:
        """
        Check if a cookie file still exists and is non-empty.

        Used for mid-session validation to detect files deleted or
        truncated after initial startup validation.

        Args:
            cookie_path: Path to cookie file

        Returns:
            True if file exists and has content, False otherwise
        """
        path = Path(cookie_path)
        if not path.exists():
            return False
        if not path.is_file():
            return False
        try:
            return path.stat().st_size > 0
        except OSError:
            return False

    def _remove_invalid_cookie(self, cookie_path: str, reason: str) -> None:
        """
        Remove a cookie file from the active list after mid-session invalidation.

        Args:
            cookie_path: Path to the invalid cookie file
            reason: Reason for removal (for logging)
        """
        if cookie_path in self._cookie_files:
            self._cookie_files.remove(cookie_path)
            self._invalid_cookies[cookie_path] = reason
            # Clean up cooldown tracking for removed cookie (prevents unbounded dict growth)
            self._failed_cookies.pop(cookie_path, None)
            logger.warning(f"Cookie file invalidated mid-session: {cookie_path} ({reason})")
            # Adjust current index if needed
            if self._current_index >= len(self._cookie_files):
                self._current_index = 0

    def get_current_cookie(self) -> Optional[str]:
        """
        Get the current active cookie file path.

        Performs mid-session validation to detect deleted or empty cookie files.

        Returns:
            Path to current cookie file, or None if no valid cookies
        """
        if not self._cookie_files:
            return None

        # Ensure index is valid
        if self._current_index >= len(self._cookie_files):
            self._current_index = 0

        current = self._cookie_files[self._current_index]

        # Mid-session validation: check if file still exists and is non-empty
        if not self._is_cookie_file_valid(current):
            reason = "deleted" if not Path(current).exists() else "empty file"
            self._remove_invalid_cookie(current, reason)
            # Try to find another valid cookie
            return self._find_valid_cookie()

        # If current cookie is in cooldown, try to find an available one
        if not self.is_available(current):
            available = self._find_available_cookie()
            if available:
                self._current_index = self._cookie_files.index(available)
                return available
            # All cookies in cooldown - return current anyway (will fail but that's expected)
            logger.warning("All cookies in cooldown, using current cookie anyway")

        return current

    def _find_valid_cookie(self) -> Optional[str]:
        """Find the first cookie that still exists, is non-empty, and not in cooldown."""
        # Remove any invalidated cookies first
        to_remove = []
        for cookie_path in self._cookie_files:
            if not self._is_cookie_file_valid(cookie_path):
                reason = "deleted" if not Path(cookie_path).exists() else "empty file"
                to_remove.append((cookie_path, reason))

        for cookie_path, reason in to_remove:
            self._remove_invalid_cookie(cookie_path, reason)

        if not self._cookie_files:
            logger.warning("All cookie files deleted or empty — no cookies available")
            return None

        # Find available (not in cooldown)
        available = self._find_available_cookie()
        if available:
            self._current_index = self._cookie_files.index(available)
            return available

        # All valid cookies in cooldown — return first valid anyway
        self._current_index = 0
        return self._cookie_files[0]

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
        elif strategy == "health":  # US-93-010: prioritize by health score
            return self._select_by_health()
        else:  # "on_error" - same as round_robin
            return self._select_round_robin()

    def _select_round_robin(self) -> Optional[str]:
        """Select next available cookie in sequence."""
        if not self._cookie_files:
            return None

        start_index = self._current_index
        tried = 0

        while tried < len(self._cookie_files):
            self._current_index = (self._current_index + 1) % len(self._cookie_files)
            candidate = self._cookie_files[self._current_index]

            # Mid-session validation: skip deleted/empty files
            if not self._is_cookie_file_valid(candidate):
                reason = "deleted" if not Path(candidate).exists() else "empty file"
                self._remove_invalid_cookie(candidate, reason)
                if not self._cookie_files:
                    return None
                # Adjust tried count since list shrank
                continue

            if self.is_available(candidate):
                return candidate

            tried += 1

        # All cookies exhausted
        logger.warning("All cookies in cooldown or exhausted")
        return None

    def _select_random(self) -> Optional[str]:
        """Select a random available cookie."""
        # Mid-session validation: filter out deleted/empty files
        to_remove = []
        for cf in self._cookie_files:
            if not self._is_cookie_file_valid(cf):
                reason = "deleted" if not Path(cf).exists() else "empty file"
                to_remove.append((cf, reason))
        for cf, reason in to_remove:
            self._remove_invalid_cookie(cf, reason)

        available = [cf for cf in self._cookie_files if self.is_available(cf)]

        if not available:
            logger.warning("No available cookies for random selection")
            return None

        selected = random.choice(available)
        self._current_index = self._cookie_files.index(selected)
        return selected

    def mark_success(self, cookie_path: str) -> None:
        """
        Mark a cookie as successful (US-93-010).

        Args:
            cookie_path: Path to cookie file that succeeded
        """
        if cookie_path not in self._cookie_health:
            self._cookie_health[cookie_path] = {"success": 0, "failure": 0}

        self._cookie_health[cookie_path]["success"] += 1

        # Clear from failed cookies on success
        if cookie_path in self._failed_cookies:
            del self._failed_cookies[cookie_path]

        logger.debug(f"Cookie marked successful: {cookie_path} "
                     f"(health: {self._cookie_health[cookie_path]})")

    def mark_failed(self, cookie_path: str) -> None:
        """
        Mark a cookie as failed (enters cooldown).

        Args:
            cookie_path: Path to cookie file that failed
        """
        self._failed_cookies[cookie_path] = time.time()

        # Track failure for health scoring (US-93-010)
        if cookie_path not in self._cookie_health:
            self._cookie_health[cookie_path] = {"success": 0, "failure": 0}
        self._cookie_health[cookie_path]["failure"] += 1

        # Check for auto-removal of consistently failing cookies (US-93-010)
        self._check_and_remove_failing_cookie(cookie_path)

        logger.debug(f"Cookie marked failed (cooldown {self.config.cooldown_seconds}s): {cookie_path}")

    def _check_and_remove_failing_cookie(self, cookie_path: str) -> None:
        """
        Check if a cookie should be automatically removed due to poor health (US-93-010).

        Removes cookie if:
        - Has exceeded max consecutive failures, OR
        - Has success rate below threshold after minimum attempts
        """
        if cookie_path not in self._cookie_health:
            return

        health = self._cookie_health[cookie_path]
        total_attempts = health["success"] + health["failure"]

        # Check consecutive failures (if tracking)
        consecutive_failures = getattr(self, '_consecutive_failures', {}).get(cookie_path, 0)
        if consecutive_failures >= self._max_consecutive_failures:
            logger.warning(
                f"Cookie auto-removed due to {consecutive_failures} consecutive failures: {cookie_path}"
            )
            self._remove_invalid_cookie(cookie_path, f"consecutive_failures:{consecutive_failures}")
            return

        # Check success rate threshold after minimum attempts
        if total_attempts >= self._health_min_attempts:
            success_rate = health["success"] / total_attempts
            if success_rate < self._success_rate_threshold:
                logger.warning(
                    f"Cookie auto-removed due to low success rate: {success_rate:.1%} "
                    f"(below {self._success_rate_threshold:.0%} threshold) after {total_attempts} attempts: {cookie_path}"
                )
                self._remove_invalid_cookie(cookie_path, f"low_success_rate:{success_rate:.2f}")

    def get_health_score(self, cookie_path: str) -> float:
        """
        Get health score for a cookie (0.0 to 1.0) (US-93-010).

        Returns:
            Health score based on success rate. Returns 0.5 for unknown cookies.
        """
        if cookie_path not in self._cookie_health:
            return 0.5  # Unknown cookies get neutral score

        health = self._cookie_health[cookie_path]
        total = health["success"] + health["failure"]

        if total == 0:
            return 0.5

        return health["success"] / total

    def get_cookie_health_report(self) -> Dict:
        """Get detailed health report for all cookies (US-93-010)."""
        report = {}
        for cookie_path in self._cookie_files:
            health = self._cookie_health.get(cookie_path, {"success": 0, "failure": 0})
            total = health["success"] + health["failure"]
            success_rate = health["success"] / total if total > 0 else 0.0

            report[cookie_path] = {
                "success": health["success"],
                "failure": health["failure"],
                "total_attempts": total,
                "success_rate": success_rate,
                "health_score": self.get_health_score(cookie_path),
                "in_cooldown": not self.is_available(cookie_path),
            }

        return report

    def _select_by_health(self) -> Optional[str]:
        """
        Select the best available cookie based on health score (US-93-010).

        Returns:
            Path to highest-health available cookie, or None if all exhausted
        """
        available = [
            (cf, self.get_health_score(cf))
            for cf in self._cookie_files
            if self.is_available(cf)
        ]

        if not available:
            logger.warning("No available cookies for health-based selection")
            return None

        # Sort by health score descending (highest first)
        available.sort(key=lambda x: x[1], reverse=True)

        selected = available[0][0]
        score = available[0][1]

        logger.info(
            f"Cookie health-based selection: {Path(selected).name} "
            f"(health score: {score:.1%}, {len(available)} available)"
        )

        self._current_index = self._cookie_files.index(selected)
        return selected

    def reset(self) -> None:
        """Reset rotation state (clear cooldowns and counters, but preserve health for session continuity)."""
        self._current_index = 0
        self._rotation_count = 0
        self._failed_cookies.clear()
        logger.info("Cookie rotator reset (health data preserved for session)")

    def get_status(self) -> Dict:
        """
        Get current rotation status for logging/debugging.

        Returns:
            Dict with rotation status info including:
            - total: All configured cookie files (valid + invalid)
            - valid: Cookie files that passed validation (exist + readable)
            - invalid: Count of cookie files that failed validation
            - cooldown: Count of valid cookies currently in cooldown
            - exhausted: True if no cookies are available (all invalid or in cooldown)
            - health_report: Detailed health info per cookie (US-93-010)
            - Plus legacy keys for backward compatibility
        """
        total = len(self._cookie_files) + len(self._invalid_cookies)
        valid = len(self._cookie_files)
        invalid = len(self._invalid_cookies)
        cooldown = valid - self.available_cookies
        exhausted = self.available_cookies == 0

        return {
            "enabled": self.is_enabled,
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "cooldown": cooldown,
            "exhausted": exhausted,
            # Health tracking (US-93-010)
            "health_report": self.get_cookie_health_report(),
            "success_rate_threshold": self._success_rate_threshold,
            # Legacy keys for backward compatibility
            "total_cookies": total,
            "valid_cookies": valid,
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
