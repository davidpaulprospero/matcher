"""
Healing Strategy configuration.

Defines how aggressively the orchestrator should attempt healing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class HealingMode(Enum):
    """Healing strategy modes."""

    # Aggressive: Try all fixes, modify config freely, minimal user interaction
    AGGRESSIVE = "aggressive"

    # Conservative: Only safe fixes, preserve config where possible
    CONSERVATIVE = "conservative"

    # Interactive: Ask user before major changes
    INTERACTIVE = "interactive"

    # Minimal: Only critical fixes, fail fast
    MINIMAL = "minimal"


@dataclass
class HealingStrategy:
    """
    Configuration for healing behavior.

    Controls how healers operate and when to escalate to user.
    """

    # Overall mode
    mode: HealingMode = HealingMode.CONSERVATIVE

    # Maximum heal attempts per stage
    max_attempts_per_stage: int = 3

    # Maximum total heals before giving up
    max_total_heals: int = 20

    # Delay between heal attempts (seconds)
    heal_delay: float = 2.0

    # Run preflight checks before pipeline
    run_preflight: bool = True

    # Auto-fix issues found in preflight
    auto_fix_preflight: bool = True

    # Enable config rollback on failure
    enable_rollback: bool = True

    # Healers to skip (by name)
    skip_healers: Set[str] = field(default_factory=set)

    # Priority order for healers (first = highest priority)
    healer_priority: List[str] = field(default_factory=lambda: [
        "checkpoint-healer",  # Try checkpoint recovery first
        "api-healer",         # API issues are common
        "download-healer",    # Download failures
        "disk-healer",        # Disk space
        "path-healer",        # Path issues
        "otio-healer",        # Timeline generation
    ])

    # Errors that should always escalate to user
    always_escalate: Set[str] = field(default_factory=lambda: {
        "permission denied",
        "authentication failed",
        "api key invalid",
        "quota exceeded",
        "account suspended",
    })

    # Config keys that should not be auto-modified
    protected_config_keys: Set[str] = field(default_factory=lambda: {
        "api_key",
        "password",
        "token",
        "secret",
    })

    # Per-healer timeout configuration (seconds)
    # If a healer is not in this dict, DEFAULT_HEALER_TIMEOUT is used
    healer_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "api-healer": 120.0,       # API healer may wait for rate limits
        "download-healer": 60.0,  # Download retries need time
        "checkpoint-healer": 30.0,
        "disk-healer": 10.0,      # Disk checks should be fast
        "path-healer": 10.0,      # Path fixes should be fast
        "otio-healer": 30.0,
        "caption-healer": 60.0,   # Caption fetching may retry
        "llm-healer": 180.0,      # LLM analysis takes longer
    })

    # Per-healer max attempts (separate from global max_attempts_per_stage)
    # If a healer is not in this dict, max_attempts_per_stage is used
    healer_max_attempts: Dict[str, int] = field(default_factory=lambda: {
        "api-healer": 5,          # API healer can retry more (rate limit waits)
        "download-healer": 4,     # Downloads worth retrying
        "checkpoint-healer": 2,   # Checkpoint recovery rarely needs more
        "disk-healer": 1,         # Disk issues usually need user intervention
        "path-healer": 2,
        "otio-healer": 2,
        "caption-healer": 3,
        "llm-healer": 2,          # LLM analysis is expensive
    })

    # Default timeout for healers not in healer_timeouts dict
    DEFAULT_HEALER_TIMEOUT: float = 30.0

    @classmethod
    def aggressive(cls) -> 'HealingStrategy':
        """Create aggressive healing strategy."""
        return cls(
            mode=HealingMode.AGGRESSIVE,
            max_attempts_per_stage=5,
            max_total_heals=50,
            heal_delay=1.0,
            run_preflight=True,
            auto_fix_preflight=True,
            enable_rollback=True,
        )

    @classmethod
    def conservative(cls) -> 'HealingStrategy':
        """Create conservative healing strategy."""
        return cls(
            mode=HealingMode.CONSERVATIVE,
            max_attempts_per_stage=3,
            max_total_heals=20,
            heal_delay=2.0,
            run_preflight=True,
            auto_fix_preflight=True,
            enable_rollback=True,
        )

    @classmethod
    def interactive(cls) -> 'HealingStrategy':
        """Create interactive healing strategy (asks user)."""
        return cls(
            mode=HealingMode.INTERACTIVE,
            max_attempts_per_stage=3,
            max_total_heals=30,
            heal_delay=2.0,
            run_preflight=True,
            auto_fix_preflight=False,  # Ask user first
            enable_rollback=True,
        )

    @classmethod
    def minimal(cls) -> 'HealingStrategy':
        """Create minimal healing strategy (fail fast)."""
        return cls(
            mode=HealingMode.MINIMAL,
            max_attempts_per_stage=1,
            max_total_heals=5,
            heal_delay=0.5,
            run_preflight=True,
            auto_fix_preflight=False,
            enable_rollback=False,
        )

    @classmethod
    def overnight(cls) -> 'HealingStrategy':
        """Create overnight/batch processing strategy.

        Optimized for unattended execution with maximum resilience:
        - High retry limits to survive transient failures
        - Long delays between attempts to respect rate limits
        - All auto-fix enabled (no user to ask)
        - Full rollback support to recover from bad states
        - Never escalates to user (no one is watching)
        """
        return cls(
            mode=HealingMode.AGGRESSIVE,
            max_attempts_per_stage=8,
            max_total_heals=100,
            heal_delay=5.0,
            run_preflight=True,
            auto_fix_preflight=True,
            enable_rollback=True,
        )

    @classmethod
    def development(cls) -> 'HealingStrategy':
        """Create development/debug strategy.

        Optimized for developer iteration with fast feedback:
        - Minimal retries (fail fast to surface issues)
        - Short delays (don't waste developer time)
        - No auto-fix (let developer see the raw error)
        - No rollback (developer controls state manually)
        - Preflight still runs to catch environment issues
        """
        return cls(
            mode=HealingMode.MINIMAL,
            max_attempts_per_stage=1,
            max_total_heals=3,
            heal_delay=0.0,
            run_preflight=True,
            auto_fix_preflight=False,
            enable_rollback=False,
        )

    @classmethod
    def production(cls) -> 'HealingStrategy':
        """Create production strategy balancing resilience and efficiency.

        Balanced approach for supervised production runs:
        - Moderate retries (recover from transient errors)
        - Standard delays (respect rate limits without stalling)
        - Auto-fix for preflight issues only
        - Rollback enabled as safety net
        - Escalates critical issues to user
        """
        return cls(
            mode=HealingMode.CONSERVATIVE,
            max_attempts_per_stage=4,
            max_total_heals=30,
            heal_delay=3.0,
            run_preflight=True,
            auto_fix_preflight=True,
            enable_rollback=True,
        )


@dataclass
class ConfigSnapshot:
    """Snapshot of config state for rollback."""

    stage_name: str
    timestamp: float
    config_values: Dict[str, any] = field(default_factory=dict)

    def restore(self, config) -> bool:
        """Restore config values from snapshot."""
        restored = 0
        for key, value in self.config_values.items():
            try:
                # Handle nested keys like "output.gap_mode"
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None:
                    setattr(obj, parts[-1], value)
                    restored += 1
            except Exception:
                pass
        return restored > 0


@dataclass
class HealingMetrics:
    """Metrics collected during healing."""

    total_heals: int = 0
    successful_heals: int = 0
    failed_heals: int = 0

    heals_by_stage: Dict[str, int] = field(default_factory=dict)
    heals_by_healer: Dict[str, int] = field(default_factory=dict)

    # US-64-005: Track successful heals per healer for utilization
    successful_heals_by_healer: Dict[str, int] = field(default_factory=dict)

    # US-64-005: Track heal times per healer for average calculation
    heal_times_by_healer: Dict[str, List[float]] = field(default_factory=dict)

    # US-64-005: Track error categories from watcher classifications
    error_categories: Dict[str, int] = field(default_factory=dict)

    time_spent_healing: float = 0.0

    preflight_issues_found: int = 0
    preflight_issues_fixed: int = 0

    rollbacks_performed: int = 0
    user_escalations: int = 0

    errors_encountered: List[str] = field(default_factory=list)

    def record_heal(self, healer_name: str, stage_name: str, success: bool,
                    heal_time_ms: float = 0.0):
        """Record a healing attempt.

        Args:
            healer_name: Name of the healer
            stage_name: Name of the stage
            success: Whether the heal was successful
            heal_time_ms: Time taken in milliseconds (US-64-005)
        """
        self.total_heals += 1
        if success:
            self.successful_heals += 1
            # Track successful heals per healer
            self.successful_heals_by_healer[healer_name] = (
                self.successful_heals_by_healer.get(healer_name, 0) + 1
            )
        else:
            self.failed_heals += 1

        self.heals_by_stage[stage_name] = self.heals_by_stage.get(stage_name, 0) + 1
        self.heals_by_healer[healer_name] = self.heals_by_healer.get(healer_name, 0) + 1

        # Track heal times for average calculation
        if heal_time_ms > 0:
            if healer_name not in self.heal_times_by_healer:
                self.heal_times_by_healer[healer_name] = []
            self.heal_times_by_healer[healer_name].append(heal_time_ms)

    def record_error_category(self, category: str):
        """Record an error category from watcher classification (US-64-005).

        Args:
            category: Error category (api, disk, path, checkpoint, download, etc.)
        """
        self.error_categories[category] = self.error_categories.get(category, 0) + 1

    def get_heal_success_rate(self) -> float:
        """Get overall heal success rate (US-64-005).

        Returns:
            Success rate as percentage (0.0 - 100.0)
        """
        if self.total_heals == 0:
            return 0.0
        return (self.successful_heals / self.total_heals) * 100.0

    def get_average_heal_time_ms(self) -> float:
        """Get average heal time across all healers (US-64-005).

        Returns:
            Average heal time in milliseconds
        """
        all_times = []
        for times in self.heal_times_by_healer.values():
            all_times.extend(times)
        if not all_times:
            return 0.0
        return sum(all_times) / len(all_times)

    def get_healer_success_rate(self, healer_name: str) -> float:
        """Get success rate for a specific healer (US-64-005).

        Args:
            healer_name: Name of the healer

        Returns:
            Success rate as percentage (0.0 - 100.0)
        """
        total = self.heals_by_healer.get(healer_name, 0)
        if total == 0:
            return 0.0
        successful = self.successful_heals_by_healer.get(healer_name, 0)
        return (successful / total) * 100.0

    def get_healer_average_time_ms(self, healer_name: str) -> float:
        """Get average heal time for a specific healer (US-64-005).

        Args:
            healer_name: Name of the healer

        Returns:
            Average heal time in milliseconds
        """
        times = self.heal_times_by_healer.get(healer_name, [])
        if not times:
            return 0.0
        return sum(times) / len(times)

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Total heals: {self.total_heals} ({self.successful_heals} successful, {self.failed_heals} failed)",
        ]

        if self.heals_by_healer:
            healer_str = ", ".join(f"{k}: {v}" for k, v in self.heals_by_healer.items())
            lines.append(f"By healer: {healer_str}")

        if self.preflight_issues_found > 0:
            lines.append(f"Preflight: {self.preflight_issues_fixed}/{self.preflight_issues_found} issues fixed")

        if self.rollbacks_performed > 0:
            lines.append(f"Rollbacks: {self.rollbacks_performed}")

        return "\n".join(lines)
