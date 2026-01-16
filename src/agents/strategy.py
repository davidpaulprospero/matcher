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
        "caption-healer",     # Caption fetch errors (before download-healer)
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

    time_spent_healing: float = 0.0

    preflight_issues_found: int = 0
    preflight_issues_fixed: int = 0

    rollbacks_performed: int = 0
    user_escalations: int = 0

    errors_encountered: List[str] = field(default_factory=list)

    def record_heal(self, healer_name: str, stage_name: str, success: bool):
        """Record a healing attempt."""
        self.total_heals += 1
        if success:
            self.successful_heals += 1
        else:
            self.failed_heals += 1

        self.heals_by_stage[stage_name] = self.heals_by_stage.get(stage_name, 0) + 1
        self.heals_by_healer[healer_name] = self.heals_by_healer.get(healer_name, 0) + 1

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
