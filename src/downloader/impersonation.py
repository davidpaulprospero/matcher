"""
Browser impersonation manager for yt-dlp TLS fingerprint bypass.

Uses curl_cffi impersonation targets to spoof browser TLS fingerprints,
defeating YouTube bot detection that relies on TLS ClientHello analysis.

Auto-detects available targets via `yt-dlp --list-impersonate-targets` at
startup and provides thread-safe round-robin rotation across all targets.

Implements US-001: ImpersonationManager with auto-detect and round-robin rotation.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ImpersonationStats:
    """Tracks impersonation rotation metrics including success/failure rates."""
    calls_made: int = 0
    unique_targets_used: Dict[str, int] = field(default_factory=dict)
    success_count: Dict[str, int] = field(default_factory=dict)
    failure_count: Dict[str, int] = field(default_factory=dict)

    @property
    def unique_count(self) -> int:
        return len(self.unique_targets_used)

    def record_use(self, target: str) -> None:
        self.calls_made += 1
        self.unique_targets_used[target] = self.unique_targets_used.get(target, 0) + 1

    def record_success(self, target: str) -> None:
        """Record a successful operation with the given target."""
        self.success_count[target] = self.success_count.get(target, 0) + 1

    def record_failure(self, target: str) -> None:
        """Record a failed operation with the given target."""
        self.failure_count[target] = self.failure_count.get(target, 0) + 1

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

    def to_dict(self) -> dict:
        return {
            'calls_made': self.calls_made,
            'unique_targets_used': dict(self.unique_targets_used),
            'success_count': dict(self.success_count),
            'failure_count': dict(self.failure_count),
        }


class ImpersonationManager:
    """Manages browser impersonation targets for yt-dlp bypass.

    Auto-detects available impersonation targets at startup by running
    `yt-dlp --list-impersonate-targets` and parsing the output. Provides
    thread-safe round-robin rotation returning the next target on each call.

    Args:
        preferred_targets: Optional list of target strings to filter to.
            Empty list means use all detected targets.
        detect_at_startup: Whether to auto-detect targets on init.
        detection_timeout: Timeout in seconds for the detection subprocess.
    """

    def __init__(
        self,
        preferred_targets: Optional[List[str]] = None,
        detect_at_startup: bool = True,
        detection_timeout: int = 10,
        min_success_rate: float = 0.2,
        enable_success_filtering: bool = True,
    ):
        self._targets: List[str] = []
        self._preferred_targets = preferred_targets or []
        self._detection_timeout = detection_timeout
        self._min_success_rate = min_success_rate
        self._enable_success_filtering = enable_success_filtering
        self._index: int = 0
        self._lock = threading.Lock()
        self._stats = ImpersonationStats()

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

    def get_impersonate_args(self) -> List[str]:
        """Get yt-dlp impersonation arguments for the next rotated target.

        Returns:
            List like ['--impersonate', 'Chrome-136:Macos-15'], or
            empty list if no targets available.
        """
        target = self.get_next_target()
        if target:
            logger.debug(f"Impersonation: {target}")
            return ['--impersonate', target]
        return []

    def record_success(self, target: str) -> None:
        """Record a successful operation with the given impersonation target.

        Thread-safe: updates the stats under lock.

        Args:
            target: The impersonation target string that was used.
        """
        with self._lock:
            self._stats.record_success(target)

    def record_failure(self, target: str) -> None:
        """Record a failed operation with the given impersonation target.

        Thread-safe: updates the stats under lock.

        Args:
            target: The impersonation target string that was used.
        """
        with self._lock:
            self._stats.record_failure(target)

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
        }
