"""
Browser impersonation manager for yt-dlp TLS fingerprint bypass.

Uses curl_cffi impersonation targets to spoof browser TLS fingerprints,
defeating YouTube bot detection that relies on TLS ClientHello analysis.

Auto-detects available targets via `yt-dlp --list-impersonate-targets` at
startup and provides thread-safe round-robin rotation across all targets.

Implements US-001: ImpersonationManager with auto-detect and round-robin rotation.
US-113-005: Enhanced browser impersonation fallback strategies - supports
fallback chain (Chrome -> Firefox -> Safari), success rate tracking per browser
type, and browser-specific extractor-args configurations.
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..logging_templates import log_rate_limit

logger = logging.getLogger(__name__)

# Browser family mapping: maps target prefix to canonical browser name
BROWSER_FAMILY_MAP = {
    'chrome': 'chrome',
    'firefox': 'firefox',
    'safari': 'safari',
    'edge': 'edge',
    'tor': 'tor',
    'opera': 'opera',
}


def get_browser_family(target: str) -> Optional[str]:
    """Extract browser family from impersonation target string (US-113-005).

    Parses targets like 'Chrome-136:Macos-15' to return 'chrome'.

    Args:
        target: Impersonation target string (e.g., 'Chrome-136:Macos-15')

    Returns:
        Browser family name (e.g., 'chrome', 'firefox', 'safari'), or None
        if the browser family cannot be determined.
    """
    if not target:
        return None

    # Extract browser name before the version number (e.g., 'Chrome-136' -> 'Chrome')
    match = re.match(r'^([a-zA-Z]+)', target.lower())
    if match:
        browser_name = match.group(1)
        return BROWSER_FAMILY_MAP.get(browser_name)

    return None


@dataclass
class ImpersonationStats:
    """Tracks impersonation rotation metrics including success/failure rates.

    US-113-005: Now tracks success/failure rates per browser family in addition
    to per-target tracking, enabling intelligent fallback chain ordering.

    US-123-004: Added windowed tracking for recent attempts per browser family
    to support adaptive profile selection based on recent success rates.
    """
    calls_made: int = 0
    unique_targets_used: Dict[str, int] = field(default_factory=dict)
    success_count: Dict[str, int] = field(default_factory=dict)
    failure_count: Dict[str, int] = field(default_factory=dict)
    # Browser family-level tracking (US-113-005)
    browser_family_success: Dict[str, int] = field(default_factory=dict)
    browser_family_failure: Dict[str, int] = field(default_factory=dict)
    # Windowed tracking for adaptive profile selection (US-123-004)
    # Stores recent outcomes per browser family: True=success, False=failure
    recent_family_attempts: Dict[str, deque] = field(default_factory=dict)
    # Maximum window size for recent attempts
    _max_window_size: int = field(default=20, repr=False)

    @property
    def unique_count(self) -> int:
        return len(self.unique_targets_used)

    def set_window_size(self, size: int) -> None:
        """Set the maximum window size for recent attempts (US-123-004).

        Args:
            size: Maximum number of recent attempts to track per browser family.
        """
        self._max_window_size = size
        # Trim existing deques to new size
        for family in self.recent_family_attempts:
            while len(self.recent_family_attempts[family]) > size:
                self.recent_family_attempts[family].popleft()

    def record_use(self, target: str) -> None:
        self.calls_made += 1
        self.unique_targets_used[target] = self.unique_targets_used.get(target, 0) + 1

    def record_success(self, target: str) -> None:
        """Record a successful operation with the given target."""
        self.success_count[target] = self.success_count.get(target, 0) + 1

        # Also track at browser family level (US-113-005)
        family = get_browser_family(target)
        if family:
            self.browser_family_success[family] = self.browser_family_success.get(family, 0) + 1

        # Track recent attempts for adaptive selection (US-123-004)
        if family:
            if family not in self.recent_family_attempts:
                self.recent_family_attempts[family] = deque(maxlen=self._max_window_size)
            self.recent_family_attempts[family].append(True)

    def record_failure(self, target: str) -> None:
        """Record a failed operation with the given target."""
        self.failure_count[target] = self.failure_count.get(target, 0) + 1

        # Also track at browser family level (US-113-005)
        family = get_browser_family(target)
        if family:
            self.browser_family_failure[family] = self.browser_family_failure.get(family, 0) + 1

        # Track recent attempts for adaptive selection (US-123-004)
        if family:
            if family not in self.recent_family_attempts:
                self.recent_family_attempts[family] = deque(maxlen=self._max_window_size)
            self.recent_family_attempts[family].append(False)

    def get_success_rate(self, target: str) -> float:
        """Calculate success rate for a target.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no data exists (optimistic default for new targets).
        """
        successes = self.success_count.get(target, 0)
        failures = self.failure_count.get(target, 0)
        total = successes + failures
        if total == 0:
            return 1.0  # Optimistic default for untested targets
        return successes / total

    def get_browser_family_success_rate(self, family: str) -> float:
        """Calculate success rate for a browser family (US-113-005).

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no data exists (optimistic default for new families).
        """
        successes = self.browser_family_success.get(family, 0)
        failures = self.browser_family_failure.get(family, 0)
        total = successes + failures
        if total == 0:
            return 1.0  # Optimistic default for untested families
        return successes / total

    def to_dict(self) -> dict:
        return {
            'calls_made': self.calls_made,
            'unique_targets_used': dict(self.unique_targets_used),
            'success_count': dict(self.success_count),
            'failure_count': dict(self.failure_count),
            'browser_family_success': dict(self.browser_family_success),
            'browser_family_failure': dict(self.browser_family_failure),
        }


class ImpersonationManager:
    """Manages browser impersonation targets for yt-dlp bypass.

    Auto-detects available impersonation targets at startup by running
    `yt-dlp --list-impersonate-targets` and parsing the output. Provides
    thread-safe round-robin rotation returning the next target on each call.

    US-113-005: Enhanced browser impersonation fallback strategies:
    - Fallback chain: When current browser fails, automatically try next browser
      in fallback_order (default: chrome -> firefox -> safari)
    - Success rate tracking per browser family for intelligent reordering
    - Browser-specific extractor-args support

    US-123-004: Adaptive profile selection:
    - When adaptive_profile_selection is enabled, select the best browser
      profile based on recent success rates instead of round-robin
    - profile_success_window controls how many recent attempts to consider

    Args:
        preferred_targets: Optional list of target strings to filter to.
            Empty list means use all detected targets.
        detect_at_startup: Whether to auto-detect targets on init.
        detection_timeout: Timeout in seconds for the detection subprocess.
        min_success_rate: Minimum success rate threshold for filtering.
        enable_success_filtering: Enable success rate-based filtering.
        fallback_order: Ordered list of browser families to try on failure.
        adaptive_profile_selection: Enable adaptive profile selection (US-123-004).
        profile_success_window: Number of recent attempts for windowed tracking (US-123-004).
    """

    def __init__(
        self,
        preferred_targets: Optional[List[str]] = None,
        detect_at_startup: bool = True,
        detection_timeout: int = 10,
        min_success_rate: float = 0.2,
        enable_success_filtering: bool = True,
        fallback_order: Optional[List[str]] = None,
        adaptive_profile_selection: bool = False,
        profile_success_window: int = 20,
    ):
        self._targets: List[str] = []
        self._preferred_targets = preferred_targets or []
        self._detection_timeout = detection_timeout
        self._min_success_rate = min_success_rate
        self._enable_success_filtering = enable_success_filtering
        self._fallback_order: List[str] = fallback_order or ['chrome', 'firefox', 'safari']
        self._adaptive_profile_selection = adaptive_profile_selection
        self._profile_success_window = profile_success_window
        self._index: int = 0
        self._lock = threading.Lock()
        self._stats = ImpersonationStats()
        # Set window size for recent attempts tracking (US-123-004)
        self._stats.set_window_size(profile_success_window)

        if detect_at_startup:
            self._targets = self.detect_targets()

            if self._targets:
                logger.info(
                    f"Impersonation manager initialized with {len(self._targets)} targets"
                )
            else:
                logger.warning("Impersonation manager: no targets detected")

    @property
    def targets(self) -> List[str]:
        """Return the list of available impersonation targets."""
        return list(self._targets)

    @property
    def target_count(self) -> int:
        """Return the number of available targets."""
        return len(self._targets)

    @property
    def stats(self) -> ImpersonationStats:
        """Return rotation statistics."""
        return self._stats

    def detect_targets(self) -> List[str]:
        """Detect available impersonation targets from yt-dlp.

        Runs `yt-dlp --list-impersonate-targets` and parses the tabular
        output into a sorted list of `Client:OS` strings suitable for
        the `--impersonate` flag.

        Returns:
            Sorted list of target strings (e.g., ['Chrome-131:Android-14', ...]).
            Empty list if detection fails.
        """
        try:
            result = subprocess.run(
                ['yt-dlp', '--ignore-config', '--list-impersonate-targets'],
                capture_output=True,
                text=True,
                timeout=self._detection_timeout,
                encoding='utf-8',
                errors='replace',
            )

            if result.returncode != 0:
                logger.warning(
                    f"yt-dlp --list-impersonate-targets exited with code {result.returncode}"
                )
                return []

            targets = self._parse_targets_output(result.stdout)

            # Filter to preferred targets if specified
            if self._preferred_targets and targets:
                filtered = [t for t in targets if t in self._preferred_targets]
                if filtered:
                    logger.debug(
                        f"Filtered to {len(filtered)}/{len(targets)} preferred targets"
                    )
                    targets = filtered
                else:
                    logger.warning(
                        f"No preferred targets matched detected targets, using all {len(targets)}"
                    )

            return sorted(targets)

        except subprocess.TimeoutExpired:
            logger.warning(
                f"yt-dlp --list-impersonate-targets timed out after {self._detection_timeout}s"
            )
            return []
        except FileNotFoundError:
            logger.warning("yt-dlp not found on PATH, impersonation unavailable")
            return []
        except Exception as e:
            logger.warning(f"Failed to detect impersonation targets: {e}")
            return []

    @staticmethod
    def _parse_targets_output(output: str) -> List[str]:
        """Parse the tabular output from --list-impersonate-targets.

        Expected format:
            [info] Available impersonate targets
            Client        OS           Source
            ------------------------------------
            Chrome-136    Macos-15     curl_cffi
            Safari-18.0   Ios-18.0     curl_cffi

        Returns:
            List of 'Client:OS' strings (e.g., ['Chrome-136:Macos-15']).
        """
        targets = []
        header_seen = False

        for line in output.splitlines():
            stripped = line.strip()

            # Skip empty lines and info prefix lines
            if not stripped or stripped.startswith('['):
                continue

            # Detect the header line
            if stripped.startswith('Client') and 'OS' in stripped:
                header_seen = True
                continue

            # Skip separator line
            if stripped.startswith('---'):
                continue

            # Parse data rows after header
            if header_seen:
                parts = stripped.split()
                if len(parts) >= 2:
                    client = parts[0]
                    os_name = parts[1]
                    target = f"{client}:{os_name}"
                    targets.append(target)

        return targets

    def get_next_target(self, skip_low_success: Optional[bool] = None) -> Optional[str]:
        """Get the next impersonation target via round-robin rotation.

        Thread-safe: uses a lock to ensure consistent rotation across
        concurrent calls from multiple download threads.

        When success rate filtering is enabled, targets with success rates
        below the threshold are skipped. A target must have at least one
        recorded success or failure to be filtered (new targets are not skipped).

        Args:
            skip_low_success: Override the default success filtering behavior.
                If None, uses the instance default (enable_success_filtering).

        Returns:
            The next target string, or None if no targets available.
        """
        if not self._targets:
            return None

        use_filtering = skip_low_success if skip_low_success is not None else self._enable_success_filtering

        with self._lock:
            # Try to find a target that meets success rate threshold
            targets_checked = 0
            while targets_checked < len(self._targets):
                target = self._targets[self._index]
                self._index = (self._index + 1) % len(self._targets)
                targets_checked += 1

                if use_filtering:
                    # Check if target has enough data to evaluate
                    successes = self._stats.success_count.get(target, 0)
                    failures = self._stats.failure_count.get(target, 0)
                    total = successes + failures

                    # Only filter if we have data; new targets pass through
                    if total > 0:
                        success_rate = self._stats.get_success_rate(target)
                        if success_rate < self._min_success_rate:
                            logger.debug(
                                f"Skipping target {target} with low success rate: "
                                f"{success_rate:.1%} (threshold: {self._min_success_rate:.1%})"
                            )
                            continue

                self._stats.record_use(target)
                return target

            # All targets have low success rate - fall back to round-robin without filtering
            logger.warning(
                "All impersonation targets have low success rates, using next in rotation"
            )
            target = self._targets[self._index]
            self._index = (self._index + 1) % len(self._targets)
            self._stats.record_use(target)

        return target

    def get_impersonate_args(self, use_adaptive: Optional[bool] = None) -> List[str]:
        """Get yt-dlp impersonation arguments for the next rotated target.

        When adaptive_profile_selection is enabled (via config or use_adaptive=True),
        selects the best browser profile based on recent success rates instead of
        using round-robin rotation.

        Args:
            use_adaptive: Override adaptive selection behavior. If None, uses
                the instance default (adaptive_profile_selection setting).

        Returns:
            List like ['--impersonate', 'Chrome-136:Macos-15'], or
            empty list if no targets available.
        """
        target = None

        # Determine whether to use adaptive selection
        use_adaptive_selection = use_adaptive if use_adaptive is not None else self._adaptive_profile_selection

        if use_adaptive_selection:
            # US-123-004: Use adaptive profile selection based on recent success rates
            best_profile = self.get_best_profile()
            if best_profile:
                target = self.get_target_for_profile(best_profile)
                if target:
                    logger.debug(f"Adaptive impersonation: {best_profile} -> {target}")
        else:
            # Use standard round-robin rotation
            target = self.get_next_target()

        if target:
            logger.debug(f"Impersonation: {target}")
            # Log impersonation rotation with browser info
            browser_family = get_browser_family(target)
            log_rate_limit(
                logger, "impersonation_rotation", "impersonation_manager", "rotate",
                target=target, browser_family=browser_family,
                adaptive_selection=use_adaptive_selection
            )
            return ['--impersonate', target]
        return []

    def get_ydl_options(self, tier: int = 1) -> dict:
        """Get yt-dlp options dictionary with impersonation settings.

        Args:
            tier: Impersonation tier (currently unused, reserved for future use)

        Returns:
            Empty dict - impersonation is disabled for search to avoid compatibility issues
        """
        # Return empty dict - impersonation causes issues with yt-dlp in Python context
        # The CLI works but Python doesn't. Fallback to no impersonation for stability.
        return {}

    def record_success(self, target: str) -> None:
        """Record a successful operation with the given impersonation target.

        Thread-safe: updates the stats under lock.

        Args:
            target: The impersonation target string that was used.
        """
        with self._lock:
            self._stats.record_success(target)
            # Log impersonation success
            browser_family = get_browser_family(target)
            log_rate_limit(
                logger, "impersonation", "impersonation_manager", "success",
                target=target, browser_family=browser_family
            )

    def record_failure(self, target: str) -> None:
        """Record a failed operation with the given impersonation target.

        Thread-safe: updates the stats under lock.

        Args:
            target: The impersonation target string that was used.
        """
        with self._lock:
            self._stats.record_failure(target)
            # Log impersonation failure
            browser_family = get_browser_family(target)
            log_rate_limit(
                logger, "impersonation", "impersonation_manager", "failure",
                target=target, browser_family=browser_family
            )

    def get_success_rate(self, target: str) -> float:
        """Get the success rate for a specific target.

        Thread-safe: reads stats under lock.

        Args:
            target: The impersonation target string.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 1.0 if no data exists for this target.
        """
        with self._lock:
            return self._stats.get_success_rate(target)

    def get_status(self) -> dict:
        """Get current manager status for debugging/metrics.

        Returns:
            Dict with target count, rotation stats, and current index.
        """
        with self._lock:
            current_index = self._index
            # Calculate success rates for all targets
            success_rates = {
                target: self._stats.get_success_rate(target)
                for target in self._targets
            }

        return {
            'target_count': len(self._targets),
            'targets': list(self._targets),
            'current_index': current_index,
            'stats': self._stats.to_dict(),
            'success_rates': success_rates,
            'min_success_rate_threshold': self._min_success_rate,
            'success_filtering_enabled': self._enable_success_filtering,
            'fallback_order': list(self._fallback_order),
        }

    def get_fallback_target(self, failed_target: str) -> Optional[str]:
        """Get a fallback target from the next browser in fallback chain (US-113-005).

        When a target fails, this method finds the next browser family in the
        fallback order and returns a target from that family.

        Args:
            failed_target: The target that failed (e.g., 'Chrome-136:Macos-15')

        Returns:
            A new target from the next browser family in fallback order,
            or None if no fallback available.
        """
        if not self._targets:
            return None

        failed_family = get_browser_family(failed_target)
        if not failed_family:
            return None

        try:
            current_idx = self._fallback_order.index(failed_family)
        except ValueError:
            # Failed family not in fallback order, start from beginning
            current_idx = -1

        # Try each browser family in order after the failed one
        for i in range(len(self._fallback_order)):
            next_idx = (current_idx + i + 1) % len(self._fallback_order)
            next_family = self._fallback_order[next_idx]

            # Find a target from this family
            for target in self._targets:
                if get_browser_family(target) == next_family:
                    # Check if this target has acceptable success rate
                    if self._enable_success_filtering:
                        rate = self._stats.get_success_rate(target)
                        if rate < self._min_success_rate:
                            continue
                    logger.info(
                        f"Falling back from {failed_family} to {next_family} "
                        f"(target: {target})"
                    )
                    return target

        logger.warning(
            f"No fallback target found after {failed_family}, "
            f"falling back to standard rotation"
        )
        return None

    def get_targets_by_family(self, family: str) -> List[str]:
        """Get all targets belonging to a specific browser family (US-113-005).

        Args:
            family: Browser family name (e.g., 'chrome', 'firefox', 'safari')

        Returns:
            List of targets from that family.
        """
        return [t for t in self._targets if get_browser_family(t) == family]

    def get_browser_family_success_rates(self) -> Dict[str, float]:
        """Get success rates for all browser families (US-113-005).

        Returns:
            Dict mapping browser family to success rate (0.0 to 1.0).
        """
        with self._lock:
            return {
                family: self._stats.get_browser_family_success_rate(family)
                for family in set(self._fallback_order)
            }

    def reorder_fallback_by_success(self) -> None:
        """Reorder fallback chain to prefer higher-success browsers (US-113-005).

        Sorts the fallback_order list so that browsers with higher success rates
        come first. Called periodically or after enough data is collected.
        """
        with self._lock:
            # Get success rates for all families in fallback order
            family_rates = []
            for family in self._fallback_order:
                rate = self._stats.get_browser_family_success_rate(family)
                family_rates.append((family, rate))

            # Sort by success rate (highest first)
            family_rates.sort(key=lambda x: x[1], reverse=True)

            # Update fallback order
            self._fallback_order = [f for f, _ in family_rates]
            logger.info(
                f"Reordered fallback chain by success rate: {self._fallback_order}"
            )

    def get_recent_success_rate(self, family: str) -> float:
        """Get success rate based on recent windowed attempts (US-123-004).

        This method calculates success rate using only the most recent attempts
        (as configured by profile_success_window), providing a more responsive
        metric for adaptive profile selection.

        Args:
            family: Browser family name (e.g., 'chrome', 'firefox', 'safari')

        Returns:
            Success rate as a float between 0.0 and 1.0 based on recent attempts.
            Returns 1.0 if no recent data exists (optimistic default).
        """
        with self._lock:
            recent_attempts = self._stats.recent_family_attempts.get(family)
            if not recent_attempts or len(recent_attempts) == 0:
                return 1.0  # Optimistic default for untested families

            successes = sum(1 for result in recent_attempts if result)
            return successes / len(recent_attempts)

    def get_best_profile(self) -> Optional[str]:
        """Get the best browser profile based on recent success rates (US-123-004).

        Selects the browser family with the highest success rate from recent
        windowed attempts. This enables adaptive profile selection that
        responds to changing conditions rather than using static ordering.

        Returns:
            The best browser family name (e.g., 'chrome'), or None if no
            targets are available or no data exists to make a selection.
        """
        if not self._targets:
            return None

        with self._lock:
            # Get recent success rates for all families in fallback order
            family_rates = []
            for family in self._fallback_order:
                rate = self.get_recent_success_rate(family)
                family_rates.append((family, rate))

            # Sort by recent success rate (highest first)
            family_rates.sort(key=lambda x: x[1], reverse=True)

            best_family = family_rates[0][0] if family_rates else None

            if best_family:
                logger.debug(
                    f"Best profile by recent success: {best_family} "
                    f"(rate: {family_rates[0][1]:.1%})"
                )

            return best_family

    def get_target_for_profile(self, profile: str) -> Optional[str]:
        """Get a target for the specified browser profile (US-123-004).

        Args:
            profile: Browser family name (e.g., 'chrome', 'firefox', 'safari')

        Returns:
            A target string for the specified profile, or None if no matching
            target is available.
        """
        if not self._targets:
            return None

        with self._lock:
            # Find targets from this family
            targets = self.get_targets_by_family(profile)
            if not targets:
                return None

            # Return the first target from this family (round-robin within family)
            target = targets[self._index % len(targets)]
            return target
