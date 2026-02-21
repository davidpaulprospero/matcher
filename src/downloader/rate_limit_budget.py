"""
Cross-keyword rate limit budget tracking.

Tracks rate limit recovery resources used across all keywords in a download session.
When processing multiple keywords, rate limit events from keyword A affect the budget
available for keyword B.

This prevents wasting resources:
- If keyword A exhausted all cookie rotations, keyword B should skip directly to VPN
- If total backoff time exceeds budget, skip backoff and escalate immediately

Tier-Isolated Budgets (US-109-007):
- Each escalation tier (tier1, tier2, tier3, tier4) has its own independent budget
- Cross-tier borrowing: if tier1 exhausted, borrow from tier2 if unused
- Budget priority: tier4 (VPN) is last to borrow, tier1 is first to lend
- Tier definitions: tier1=impersonate, tier2=extractor-args, tier3=cookie, tier4=VPN
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


# Global mock rate limit settings (can be set programmatically for testing)
_mock_rate_limits_enabled: bool = False
_mock_delay_seconds: float = 0.01


def set_mock_rate_limits(enabled: bool, delay_seconds: float = 0.01) -> None:
    """Programmatically enable/disable mock rate limits.

    Args:
        enabled: Whether to enable mock rate limiting
        delay_seconds: Delay to use when mock is enabled (default: 0.01s)
    """
    global _mock_rate_limits_enabled, _mock_delay_seconds
    _mock_rate_limits_enabled = enabled
    _mock_delay_seconds = delay_seconds
    logger.debug(f"Mock rate limits: enabled={enabled}, delay={delay_seconds}s")


def is_mock_rate_limits_enabled() -> bool:
    """Check if mock rate limits are enabled.

    Checks:
    1. Environment variable MOCK_RATE_LIMITS=1
    2. Programmatically set via set_mock_rate_limits()

    Returns:
        True if mock rate limiting is enabled
    """
    # Check environment variable first
    if os.environ.get("MOCK_RATE_LIMITS", "").lower() in ("1", "true", "yes"):
        return True
    # Check programmatic setting
    return _mock_rate_limits_enabled


def get_mock_delay_seconds() -> float:
    """Get the mock delay duration in seconds.

    Returns:
        Mock delay seconds (default: 0.01s)
    """
    return _mock_delay_seconds


def mock_rate_limit_delay(delay_seconds: float, config=None) -> None:
    """Perform a rate limit delay, bypassing actual delay in mock mode.

    When mock mode is enabled (via MOCK_RATE_LIMITS env var or programmatically),
    this function returns immediately or after a minimal delay instead of the
    full requested delay.

    Args:
        delay_seconds: The intended delay duration in seconds
        config: Optional config object to check for test_mode.mock_rate_limits
    """
    # Check if mock mode is enabled via environment variable
    if is_mock_rate_limits_enabled():
        mock_delay = get_mock_delay_seconds()
        if mock_delay > 0:
            time.sleep(mock_delay)
        return

    # Check config.test_mode.mock_rate_limits if config provided
    if config is not None:
        test_mode_config = getattr(config, 'test_mode', None)
        if test_mode_config:
            mock_enabled = getattr(test_mode_config, 'mock_rate_limits', False)
            if mock_enabled:
                mock_delay = getattr(test_mode_config, 'mock_delay_seconds', 0.01)
                if mock_delay > 0:
                    time.sleep(mock_delay)
                return

    # Real mode - perform actual delay
    if delay_seconds > 0:
        time.sleep(delay_seconds)


class EscalationTier(Enum):
    """Escalation tiers for download retry strategy.

    Tiers represent increasing levels of bypass sophistication:
    - tier1: Browser impersonation (Chrome/Mobile Safari)
    - tier2: Alternative extractor clients (web_safari, tv, etc.)
    - tier3: Cookie rotation with fresh cookies
    - tier4: VPN IP rotation (Mullvad)
    """
    TIER1 = "tier1"  # Impersonation
    TIER2 = "tier2"  # Extractor args
    TIER3 = "tier3"  # Cookie rotation
    TIER4 = "tier4"  # VPN rotation

    @classmethod
    def from_string(cls, value: str) -> "EscalationTier":
        """Parse tier from string."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError:
            return cls.TIER1  # Default to tier1

    def priority(self) -> int:
        """Get borrowing priority (lower = lend first, higher = borrow last).

        Tier priority for cross-tier borrowing:
        - tier1 (priority=1): First to lend, last to borrow from
        - tier2 (priority=2): Second priority
        - tier3 (priority=3): Third priority
        - tier4 (priority=4): Last to lend, first to borrow from
        """
        tier_priorities = {
            EscalationTier.TIER1: 1,
            EscalationTier.TIER2: 2,
            EscalationTier.TIER3: 3,
            EscalationTier.TIER4: 4,
        }
        return tier_priorities.get(self, 1)


# All tiers in priority order (for borrowing search)
TIER_PRIORITY_ORDER = [
    EscalationTier.TIER1,
    EscalationTier.TIER2,
    EscalationTier.TIER3,
    EscalationTier.TIER4,
]


class PriorityLevel(Enum):
    """Keyword priority levels for budget allocation."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_string(cls, value: str) -> "PriorityLevel":
        """Parse priority from string (case-insensitive)."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError:
            return cls.MEDIUM  # Default to medium


# Default budget allocation percentages by priority
PRIORITY_ALLOCATION = {
    PriorityLevel.HIGH: 0.50,    # 50% of budget
    PriorityLevel.MEDIUM: 0.30,  # 30% of budget
    PriorityLevel.LOW: 0.20,     # 20% of budget
}


@dataclass
class TierBudget:
    """Budget tracking for a single escalation tier.

    Each tier has independent budget tracking, allowing tier1 to be exhausted
    while tier2 still has capacity. Cross-tier borrowing enables borrowing
    from lower-priority tiers when a higher-priority tier is exhausted.

    Attributes:
        tier: The escalation tier this budget tracks
        attempts_used: Number of retry attempts used within this tier
        max_attempts: Maximum attempts allowed in this tier (0 = unlimited)
        borrowed_from: Source tier this tier borrowed from (if any)
        lent_to: Destination tier this tier lent to (if any)
    """
    tier: EscalationTier = EscalationTier.TIER1
    attempts_used: int = 0
    max_attempts: int = 5  # Default 5 attempts per tier
    borrowed_from: Optional[EscalationTier] = None
    lent_to: Optional[EscalationTier] = None

    # Track borrowed amount for logging
    _borrowed_amount: int = 0
    _lent_amount: int = 0

    def can_attempt(self) -> bool:
        """Check if this tier can attempt within its budget.

        Returns:
            True if attempts are unlimited or under limit
        """
        if self.max_attempts <= 0:
            return True
        return self.attempts_used < self.max_attempts

    def attempts_remaining(self) -> Optional[int]:
        """Get remaining attempts in this tier.

        Returns:
            Number of attempts remaining, or None if unlimited
        """
        if self.max_attempts <= 0:
            return None
        return max(0, self.max_attempts - self.attempts_used)

    def record_attempt(self) -> None:
        """Record a retry attempt in this tier."""
        self.attempts_used += 1
        logger.debug(f"Tier budget {self.tier.value}: attempt recorded (total: {self.attempts_used})")

    def is_exhausted(self) -> bool:
        """Check if this tier's budget is exhausted.

        Returns:
            True if no attempts remain in this tier
        """
        return not self.can_attempt()

    def has_unused_capacity(self, threshold: int = 1) -> bool:
        """Check if this tier has unused capacity to lend.

        Args:
            threshold: Minimum unused attempts required to consider lending

        Returns:
            True if tier has capacity above threshold
        """
        remaining = self.attempts_remaining()
        if remaining is None:
            return True  # Unlimited
        return remaining >= threshold

    def borrow_from(self, donor: 'TierBudget', amount: int = 1) -> bool:
        """Borrow budget capacity from another tier.

        Args:
            donor: The tier budget to borrow from
            amount: Number of attempts to borrow

        Returns:
            True if borrow was successful
        """
        if not donor.has_unused_capacity(amount):
            logger.debug(
                f"Tier budget {self.tier.value}: cannot borrow {amount} from {donor.tier.value} "
                f"(remaining: {donor.attempts_remaining()})"
            )
            return False

        # Record the borrowing
        self.borrowed_from = donor.tier
        self._borrowed_amount += amount
        donor.lent_to = self.tier
        donor._lent_amount += amount

        logger.info(
            f"Tier budget CROSS-TIER BORROW: {self.tier.value} borrowed {amount} attempt(s) "
            f"from {donor.tier.value} (borrower remaining: {self.attempts_remaining()}, "
            f"donor remaining: {donor.attempts_remaining()})"
        )
        return True

    def get_usage_percentage(self) -> Optional[float]:
        """Get usage percentage of this tier's budget.

        Returns:
            Percentage (0-100) or None if unlimited
        """
        if self.max_attempts <= 0:
            return None
        return (self.attempts_used / self.max_attempts) * 100

    def to_dict(self) -> Dict:
        """Serialize tier budget for checkpoint.

        Returns:
            Dict with tier budget state
        """
        return {
            "tier": self.tier.value,
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "borrowed_from": self.borrowed_from.value if self.borrowed_from else None,
            "lent_to": self.lent_to.value if self.lent_to else None,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'TierBudget':
        """Create tier budget from checkpoint data.

        Args:
            data: Dict with tier budget state

        Returns:
            TierBudget instance
        """
        if not data:
            return cls()

        return cls(
            tier=EscalationTier.from_string(data.get("tier", "tier1")),
            attempts_used=data.get("attempts_used", 0),
            max_attempts=data.get("max_attempts", 5),
            borrowed_from=EscalationTier.from_string(data["borrowed_from"]) if data.get("borrowed_from") else None,
            lent_to=EscalationTier.from_string(data["lent_to"]) if data.get("lent_to") else None,
        )


class TierBudgetManager:
    """Manages tier-isolated budgets with cross-tier borrowing.

    This class coordinates budget across all escalation tiers:
    - Each tier has independent budget tracking
    - Cross-tier borrowing allows exhausted tiers to borrow from unused tiers
    - Budget priority: tier1 first to lend, tier4 last to borrow

    Borrowing rules:
    - tier1 (impersonate) is first to lend, last to borrow
    - tier4 (VPN) is last to lend, first to borrow
    - Only borrow if donor has unused capacity (threshold=1)
    """

    def __init__(self, tier_configs: Optional[Dict[EscalationTier, int]] = None):
        """Initialize tier budget manager.

        Args:
            tier_configs: Optional dict mapping tier to max_attempts.
                         If None, uses defaults (5 per tier).
        """
        self._tier_budgets: Dict[EscalationTier, TierBudget] = {}

        # Initialize budgets for all tiers with default or custom limits
        default_limits = {
            EscalationTier.TIER1: 5,   # 5 impersonation attempts
            EscalationTier.TIER2: 5,   # 5 extractor-args attempts
            EscalationTier.TIER3: 5,   # 5 cookie rotation attempts
            EscalationTier.TIER4: 3,   # 3 VPN rotations (expensive)
        }

        configs = tier_configs or default_limits
        for tier, max_attempts in configs.items():
            self._tier_budgets[tier] = TierBudget(tier=tier, max_attempts=max_attempts)

    def get_budget(self, tier: EscalationTier) -> TierBudget:
        """Get the budget for a specific tier.

        Args:
            tier: The escalation tier

        Returns:
            TierBudget for the tier
        """
        if tier not in self._tier_budgets:
            # Auto-create if missing
            self._tier_budgets[tier] = TierBudget(tier=tier)
        return self._tier_budgets[tier]

    def can_attempt(self, tier: EscalationTier) -> bool:
        """Check if a tier can attempt within its budget.

        Also attempts cross-tier borrowing if local budget exhausted.

        Args:
            tier: The escalation tier to check

        Returns:
            True if tier can attempt (with or without borrowing)
        """
        budget = self.get_budget(tier)
        if budget.can_attempt():
            return True

        # Try to borrow from other tiers
        return self._try_borrow(tier)

    def _try_borrow(self, borrower_tier: EscalationTier) -> bool:
        """Attempt to borrow budget from other tiers.

        Searches tiers in borrowing priority order (tier4 first, tier1 last).

        Args:
            borrower_tier: The tier that needs to borrow

        Returns:
            True if borrow was successful
        """
        borrower_budget = self.get_budget(borrower_tier)

        # Borrowing priority: lower priority tiers lend first
        # Reverse order: tier4 first, then tier3, tier2, tier1
        lending_order = list(reversed(TIER_PRIORITY_ORDER))

        for donor_tier in lending_order:
            # Skip the borrower itself
            if donor_tier == borrower_tier:
                continue

            donor_budget = self.get_budget(donor_tier)

            # Skip if donor is exhausted
            if not donor_budget.has_unused_capacity():
                continue

            # Attempt to borrow
            if borrower_budget.borrow_from(donor_budget):
                return True

        logger.warning(
            f"Tier budget EXHAUSTED: {borrower_tier.value} exhausted and cannot borrow "
            f"from any other tier (all tiers at capacity)"
        )
        return False

    def record_attempt(self, tier: EscalationTier) -> bool:
        """Record an attempt in a tier, with cross-tier borrowing support.

        Args:
            tier: The tier that attempted

        Returns:
            True if attempt was recorded successfully
        """
        budget = self.get_budget(tier)

        # Check if we can attempt (includes borrowing attempt)
        if not self.can_attempt(tier):
            logger.warning(
                f"Tier budget: {tier.value} cannot attempt (exhausted with no borrowing available)"
            )
            return False

        budget.record_attempt()
        return True

    def get_tier_status(self) -> Dict:
        """Get status of all tiers for logging.

        Returns:
            Dict mapping tier to status info
        """
        status = {}
        for tier in TIER_PRIORITY_ORDER:
            budget = self.get_budget(tier)
            status[tier.value] = {
                "attempts_used": budget.attempts_used,
                "attempts_remaining": budget.attempts_remaining(),
                "max_attempts": budget.max_attempts,
                "usage_pct": budget.get_usage_percentage(),
                "is_exhausted": budget.is_exhausted(),
                "borrowed_from": budget.borrowed_from.value if budget.borrowed_from else None,
                "lent_to": budget.lent_to.value if budget.lent_to else None,
            }
        return status

    def is_any_tier_available(self) -> bool:
        """Check if any tier has available budget.

        Returns:
            True if at least one tier can attempt
        """
        for tier in TIER_PRIORITY_ORDER:
            if self.can_attempt(tier):
                return True
        return False

    def get_next_available_tier(self) -> Optional[EscalationTier]:
        """Get the next tier that has available budget.

        Returns:
            The highest-priority tier with available budget, or None
        """
        for tier in TIER_PRIORITY_ORDER:
            if self.can_attempt(tier):
                return tier
        return None

    def to_dict(self) -> Dict:
        """Serialize tier budgets for checkpoint.

        Returns:
            Dict with all tier budget states
        """
        return {
            "tier_budgets": {
                tier.value: budget.to_dict()
                for tier, budget in self._tier_budgets.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'TierBudgetManager':
        """Create tier budget manager from checkpoint data.

        Args:
            data: Dict with tier budget states

        Returns:
            TierBudgetManager instance
        """
        if not data or "tier_budgets" not in data:
            return cls()

        manager = cls()
        manager._tier_budgets = {}

        for tier_str, tier_data in data.get("tier_budgets", {}).items():
            tier = EscalationTier.from_string(tier_str)
            manager._tier_budgets[tier] = TierBudget.from_dict(tier_data)

        return manager

    def clear(self) -> None:
        """Clear all tier budgets for new session."""
        self._tier_budgets = {}
        # Re-initialize with defaults
        default_limits = {
            EscalationTier.TIER1: 5,
            EscalationTier.TIER2: 5,
            EscalationTier.TIER3: 5,
            EscalationTier.TIER4: 3,
        }
        for tier, max_attempts in default_limits.items():
            self._tier_budgets[tier] = TierBudget(tier=tier, max_attempts=max_attempts)
        logger.debug("Tier budget manager cleared")


@dataclass
class RateLimitBudget:
    """Tracks rate limit recovery resources used across all keywords.

    When share_budget_across_keywords is enabled (default), all keywords in a
    download session share the same budget. This prevents wasting resources:

    - If keyword A exhausted all cookie rotations, keyword B should skip
      directly to VPN switching instead of trying rotations again.
    - If total backoff time exceeds the budget, skip backoff and escalate.

    Tier-Isolated Budgets (US-109-007):
    - When tier_isolation_enabled=True, each escalation tier has independent budget
    - Cross-tier borrowing: exhausted tier1 can borrow from tier2 if unused
    - Budget priority: tier4 (VPN) is last to borrow, tier1 is first to lend
    - Provides granular control over retry strategies per tier

    Attributes:
        rotations_used: Number of cookie rotations used this session
        vpn_switches_used: Number of VPN switches used this session
        backoff_time_spent: Total time spent in backoff delays (seconds)
        keywords_rate_limited: Keywords that triggered rate limit events
        last_escalation_level: Current escalation level (backoff, cookie, vpn)
        tier_budget_manager: Manages tier-isolated budgets with cross-tier borrowing
        tier_isolation_enabled: Enable tier-isolated budgets (US-109-007)
    """
    rotations_used: int = 0
    vpn_switches_used: int = 0
    backoff_time_spent: float = 0.0
    keywords_rate_limited: List[str] = field(default_factory=list)
    last_escalation_level: str = "none"  # none, backoff, cookie, vpn
    successes: int = 0
    failures: int = 0

    # Per-keyword priority tracking
    keyword_priorities: Dict[str, PriorityLevel] = field(default_factory=dict)
    keyword_consecutive_successes: Dict[str, int] = field(default_factory=dict)

    # Budget limits (set from config)
    max_rotations: int = 0  # 0 = unlimited
    max_vpn_switches: int = 10
    max_backoff_time: float = 300.0  # 5 minutes total backoff budget

    # Priority-based allocation config
    priority_allocation_enabled: bool = True
    consecutive_success_boost_threshold: int = 3  # After N consecutive successes, enable boost
    consecutive_success_boost_multiplier: float = 1.5  # Boost high-priority by this multiplier

    # Tier-isolated budget (US-109-007)
    tier_isolation_enabled: bool = False  # Disabled by default for backward compatibility
    tier_budget_manager: Optional[TierBudgetManager] = None  # Initialized on demand

    # Adaptive cooldown optimization (US-123-006)
    # Track recovery times to optimize cooldown duration
    adaptive_cooldown_enabled: bool = False  # Enable/disable adaptive cooldown
    cooldown_history: int = 10  # Number of recent recoveries to consider
    recovery_times: List[float] = field(default_factory=list)  # Historical recovery times in seconds
    last_rate_limit_timestamp: Optional[float] = None  # Timestamp of last rate limit event
    default_cooldown_seconds: float = 30.0  # Default cooldown if no history

    @classmethod
    def from_config(cls, budget_config) -> "RateLimitBudget":
        """Create a RateLimitBudget from a RateLimitBudgetConfig dataclass.

        Args:
            budget_config: RateLimitBudgetConfig instance or dict with budget settings.
                           If None, returns default budget.

        Returns:
            RateLimitBudget with limits set from config.
        """
        budget = cls()
        if budget_config is None:
            return budget

        budget.max_rotations = int(getattr(budget_config, 'max_rotations', 10))
        budget.max_backoff_time = float(getattr(budget_config, 'max_backoff_time', 600.0))
        budget.max_vpn_switches = int(getattr(budget_config, 'max_vpn_switches', 3))
        logger.debug(
            f"Budget initialized from config: max_rotations={budget.max_rotations}, "
            f"max_backoff_time={budget.max_backoff_time}s, "
            f"max_vpn_switches={budget.max_vpn_switches}"
        )

        # Initialize tier budget manager if enabled
        budget.tier_isolation_enabled = bool(getattr(budget_config, 'tier_isolation_enabled', False))
        if budget.tier_isolation_enabled:
            budget.tier_budget_manager = TierBudgetManager()
            logger.debug("Tier-isolated budgets enabled")

        # Initialize adaptive cooldown settings (US-123-006)
        budget.adaptive_cooldown_enabled = bool(getattr(budget_config, 'adaptive_cooldown_enabled', False))
        budget.cooldown_history = int(getattr(budget_config, 'cooldown_history', 10))
        budget.default_cooldown_seconds = float(getattr(budget_config, 'default_cooldown_seconds', 30.0))
        if budget.adaptive_cooldown_enabled:
            logger.debug(
                f"Adaptive cooldown enabled: history={budget.cooldown_history}, "
                f"default={budget.default_cooldown_seconds}s"
            )

        return budget

    def _ensure_tier_manager(self) -> TierBudgetManager:
        """Ensure tier budget manager is initialized.

        Returns:
            The tier budget manager instance
        """
        if self.tier_budget_manager is None:
            self.tier_budget_manager = TierBudgetManager()
        return self.tier_budget_manager

    def enable_tier_isolation(self, tier_configs: Optional[Dict[EscalationTier, int]] = None) -> None:
        """Enable tier-isolated budgets with optional custom tier limits.

        Args:
            tier_configs: Optional dict mapping tier to max_attempts.
                         If None, uses default limits (5 per tier, 3 for VPN).
        """
        self.tier_isolation_enabled = True
        self.tier_budget_manager = TierBudgetManager(tier_configs)
        logger.info(
            f"Tier-isolated budgets enabled with config: "
            f"{ {t.value: m for t, m in (tier_configs or {}).items()} or 'defaults'}"
        )

    def disable_tier_isolation(self) -> None:
        """Disable tier-isolated budgets and use global budget only."""
        self.tier_isolation_enabled = False
        self.tier_budget_manager = None
        logger.info("Tier-isolated budgets disabled")

    def can_attempt_tier(self, tier: EscalationTier) -> bool:
        """Check if a tier can attempt within its budget (with cross-tier borrowing).

        Args:
            tier: The escalation tier to check

        Returns:
            True if tier can attempt (with or without borrowing)
        """
        if not self.tier_isolation_enabled:
            # Fall back to legacy global budget check
            return self.can_rotate() if tier == EscalationTier.TIER3 else True

        return self._ensure_tier_manager().can_attempt(tier)

    def record_tier_attempt(self, tier: EscalationTier) -> bool:
        """Record an attempt in a tier, with cross-tier borrowing support.

        Args:
            tier: The tier that attempted

        Returns:
            True if attempt was recorded successfully
        """
        if not self.tier_isolation_enabled:
            # Fall back to legacy global tracking
            return True

        return self._ensure_tier_manager().record_attempt(tier)

    def get_tier_status(self) -> Dict:
        """Get status of all tiers for logging.

        Returns:
            Dict mapping tier to status info, or empty dict if tier isolation disabled
        """
        if not self.tier_isolation_enabled or self.tier_budget_manager is None:
            return {}

        return self.tier_budget_manager.get_tier_status()

    def get_tier_budget_summary(self) -> str:
        """Get a formatted summary of tier budgets for logging.

        Returns:
            Formatted summary string showing all tier statuses
        """
        status = self.get_tier_status()
        if not status:
            return "Tier budgets: disabled"

        parts = []
        for tier_name, tier_info in status.items():
            remaining = tier_info.get("attempts_remaining")
            max_attempts = tier_info.get("max_attempts")
            borrowed_from = tier_info.get("borrowed_from")
            lent_to = tier_info.get("lent_to")

            remaining_str = f"{remaining}/{max_attempts}" if remaining is not None else "unlimited"

            # Add borrowing info if applicable
            borrow_info = ""
            if borrowed_from:
                borrow_info = f" (borrowed from {borrowed_from})"
            elif lent_to:
                borrow_info = f" (lent to {lent_to})"

            parts.append(f"{tier_name}: {remaining_str}{borrow_info}")

        return f"Tier budgets: {', '.join(parts)}"

    def is_tier_exhausted(self, tier: EscalationTier) -> bool:
        """Check if a specific tier is exhausted.

        Args:
            tier: The escalation tier to check

        Returns:
            True if tier has no budget remaining and cannot borrow
        """
        if not self.tier_isolation_enabled:
            # Legacy behavior: check global budget
            if tier == EscalationTier.TIER3:
                return not self.can_rotate()
            if tier == EscalationTier.TIER4:
                return not self.can_switch_vpn()
            return False

        budget = self._ensure_tier_manager().get_budget(tier)
        return budget.is_exhausted() and not self.can_attempt_tier(tier)

    def get_next_available_tier(self) -> Optional[EscalationTier]:
        """Get the next tier that has available budget.

        Returns:
            The highest-priority tier with available budget, or None
        """
        if not self.tier_isolation_enabled:
            # Legacy: return tier based on current state
            if self.can_rotate():
                return EscalationTier.TIER3
            if self.can_switch_vpn():
                return EscalationTier.TIER4
            return None

        return self._ensure_tier_manager().get_next_available_tier()

    def scale_for_keywords(self, keyword_count: int, auto_scale: bool = True) -> None:
        """Scale budget limits based on keyword count.

        When processing many keywords, the base budget limits may be too low.
        This scales limits proportionally: ceil(keyword_count / 5), capped at 5x.

        Args:
            keyword_count: Number of keywords being processed
            auto_scale: If False, do nothing (config override)
        """
        if not auto_scale:
            return

        multiplier = min(math.ceil(keyword_count / 5), 5)

        if multiplier <= 1:
            return  # No scaling needed

        base_rotations = self.max_rotations
        base_backoff = self.max_backoff_time
        base_vpn = self.max_vpn_switches

        # Scale limits (only non-zero/unlimited limits)
        if self.max_rotations > 0:
            self.max_rotations = self.max_rotations * multiplier
        if self.max_backoff_time > 0:
            self.max_backoff_time = self.max_backoff_time * multiplier
        if self.max_vpn_switches > 0:
            self.max_vpn_switches = self.max_vpn_switches * multiplier

        logger.info(
            f"Rate limit budget auto-scaled for {keyword_count} keywords: "
            f"rotations={self.max_rotations}, backoff={self.max_backoff_time}s"
        )

    def record_rotation(self, keyword: str = None) -> None:
        """Record a cookie rotation.

        Args:
            keyword: Keyword that triggered the rotation (optional for tracking)
        """
        self.rotations_used += 1
        self.last_escalation_level = "cookie"
        if keyword is not None and keyword not in self.keywords_rate_limited:
            self.keywords_rate_limited.append(keyword)
        logger.debug(f"Budget: cookie rotation recorded (total: {self.rotations_used})")

    def record_vpn_switch(self, keyword: str = None) -> None:
        """Record a VPN switch.

        Args:
            keyword: Keyword that triggered the switch (optional for tracking)
        """
        self.vpn_switches_used += 1
        self.last_escalation_level = "vpn"
        if keyword is not None and keyword not in self.keywords_rate_limited:
            self.keywords_rate_limited.append(keyword)
        logger.debug(f"Budget: VPN switch recorded (total: {self.vpn_switches_used})")

    def record_vpn_rotation(self, keyword: str = None) -> None:
        """Record a VPN rotation (alias for record_vpn_switch).

        This method is an alias for record_vpn_switch() to match the
        VPN_ROTATION tier naming in EscalationManager (Tier 4).

        Args:
            keyword: Keyword that triggered the rotation (optional for tracking)
        """
        self.record_vpn_switch(keyword)

    def record_attempt(self, keyword: str = None) -> None:
        """Record a download attempt for tracking total attempts across keywords.

        Args:
            keyword: Keyword that triggered the attempt (optional for tracking)
        """
        # Lightweight tracking - increments are used by EscalationManager
        # to understand overall download volume.
        logger.debug(f"Budget: download attempt recorded (keyword={keyword})")

    def record_success(self, keyword: str = None) -> None:
        """Record a successful download to track success rate.

        Args:
            keyword: Keyword that succeeded (optional for tracking)
        """
        self.successes += 1
        logger.debug(f"Budget: success recorded (total: {self.successes}, keyword={keyword})")

    def record_failure(self, keyword: str = None) -> None:
        """Record a failed download to track failure rate.

        Args:
            keyword: Keyword that failed (optional for tracking)
        """
        self.failures += 1
        if keyword is not None and keyword not in self.keywords_rate_limited:
            self.keywords_rate_limited.append(keyword)
        logger.debug(f"Budget: failure recorded (total: {self.failures}, keyword={keyword})")

    def record_backoff(self, seconds: float, keyword: str = None) -> None:
        """Record backoff time spent.

        Args:
            seconds: Duration of backoff delay
            keyword: Keyword that triggered the backoff (optional for tracking)
        """
        self.backoff_time_spent += seconds
        self.last_escalation_level = "backoff"
        if keyword is not None and keyword not in self.keywords_rate_limited:
            self.keywords_rate_limited.append(keyword)
        logger.debug(
            f"Budget: backoff {seconds:.1f}s recorded "
            f"(total: {self.backoff_time_spent:.1f}s)"
        )

    def record_rate_limit_event(self, timestamp: Optional[float] = None) -> None:
        """Record a rate limit event to track for recovery time calculation.

        This marks the start of a rate limit period. When a successful download
        occurs after this, the recovery time can be calculated.

        Args:
            timestamp: Unix timestamp of the rate limit event. If None, uses current time.
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        self.last_rate_limit_timestamp = timestamp
        logger.debug(f"Budget: rate limit event recorded at {timestamp}")

    def record_recovery(self, success_timestamp: Optional[float] = None) -> Optional[float]:
        """Record a successful recovery from rate limit and calculate recovery time.

        When a download succeeds after a rate limit event, call this to record
        the recovery time. This builds historical data for adaptive cooldown.

        Args:
            success_timestamp: Unix timestamp of successful download. If None, uses current time.

        Returns:
            The recovery time in seconds, or None if no rate limit event was recorded.
        """
        import time
        if success_timestamp is None:
            success_timestamp = time.time()

        if self.last_rate_limit_timestamp is None:
            logger.debug("Budget: recovery recorded but no prior rate limit event")
            return None

        recovery_time = success_timestamp - self.last_rate_limit_timestamp

        # Add to recovery history (keeping only recent history)
        self.recovery_times.append(recovery_time)
        if len(self.recovery_times) > self.cooldown_history:
            self.recovery_times.pop(0)

        # Clear the rate limit timestamp (ready for next event)
        self.last_rate_limit_timestamp = None

        logger.debug(
            f"Budget: recovery recorded: {recovery_time:.1f}s "
            f"(history count: {len(self.recovery_times)})"
        )
        return recovery_time

    def get_optimal_cooldown(self) -> float:
        """Get the optimal cooldown duration based on historical recovery times.

        Uses adaptive cooldown if enabled and history is available. Otherwise,
        returns the default cooldown duration.

        The optimal cooldown is calculated as:
        - If adaptive enabled and history exists: use the 75th percentile of recovery times
          (conservative - ensures most recoveries succeed)
        - Otherwise: return default cooldown

        Returns:
            Recommended cooldown duration in seconds
        """
        if not self.adaptive_cooldown_enabled:
            return self.default_cooldown_seconds

        if not self.recovery_times:
            logger.debug(
                f"Budget: no recovery history, using default cooldown "
                f"({self.default_cooldown_seconds}s)"
            )
            return self.default_cooldown_seconds

        # Calculate 75th percentile for conservative cooldown (covers most cases)
        sorted_times = sorted(self.recovery_times)
        n = len(sorted_times)
        if n == 1:
            optimal = sorted_times[0]
        else:
            # 75th percentile index
            idx = int(n * 0.75)
            if idx >= n:
                idx = n - 1
            optimal = sorted_times[idx]

        # Ensure at least the default cooldown
        optimal = max(optimal, self.default_cooldown_seconds)

        logger.debug(
            f"Budget: optimal cooldown calculated: {optimal:.1f}s "
            f"(from {len(self.recovery_times)} samples, "
            f"min={min(self.recovery_times):.1f}s, "
            f"max={max(self.recovery_times):.1f}s)"
        )
        return optimal

    def get_cooldown_stats(self) -> Dict:
        """Get statistics about cooldown history.

        Returns:
            Dict with cooldown statistics including count, min, max, mean, and optimal
        """
        if not self.recovery_times:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "optimal": self.default_cooldown_seconds,
                "adaptive_enabled": self.adaptive_cooldown_enabled,
            }

        return {
            "count": len(self.recovery_times),
            "min": min(self.recovery_times),
            "max": max(self.recovery_times),
            "mean": sum(self.recovery_times) / len(self.recovery_times),
            "optimal": self.get_optimal_cooldown(),
            "adaptive_enabled": self.adaptive_cooldown_enabled,
        }

    def reset_cooldown_history(self) -> None:
        """Clear all cooldown history data."""
        self.recovery_times.clear()
        self.last_rate_limit_timestamp = None
        logger.debug("Budget: cooldown history reset")

    def can_rotate(self) -> bool:
        """Check if cookie rotation is available within budget.

        Returns:
            True if rotations are unlimited or under limit
        """
        if self.max_rotations <= 0:
            return True  # Unlimited
        return self.rotations_used < self.max_rotations

    def can_switch_vpn(self) -> bool:
        """Check if VPN switching is available within budget.

        Returns:
            True if VPN switches are unlimited or under limit
        """
        if self.max_vpn_switches <= 0:
            return True  # Unlimited
        return self.vpn_switches_used < self.max_vpn_switches

    def can_rotate_vpn(self) -> bool:
        """Check if VPN rotation is available within budget (alias for can_switch_vpn).

        This method is an alias for can_switch_vpn() to match the
        VPN_ROTATION tier naming in EscalationManager (Tier 4).

        Returns:
            True if VPN rotations are unlimited or under limit
        """
        return self.can_switch_vpn()

    def can_backoff(self, additional_seconds: float = 0.0) -> bool:
        """Check if backoff time is available within budget.

        Args:
            additional_seconds: Planned backoff duration to check

        Returns:
            True if total backoff time would be under limit
        """
        if self.max_backoff_time <= 0:
            return True  # Unlimited
        return (self.backoff_time_spent + additional_seconds) <= self.max_backoff_time

    def should_skip_backoff(self, proposed_seconds: float) -> bool:
        """Check if backoff should be skipped because budget would be exceeded.

        This is the inverse of can_backoff() - returns True when the proposed
        backoff duration would exceed the remaining budget.

        Args:
            proposed_seconds: Proposed backoff duration to check

        Returns:
            True if backoff should be skipped (budget would be exceeded)
        """
        return not self.can_backoff(proposed_seconds)

    def rotations_remaining(self) -> Optional[int]:
        """Get remaining cookie rotations.

        Returns:
            Number of rotations remaining, or None if unlimited
        """
        if self.max_rotations <= 0:
            return None
        return max(0, self.max_rotations - self.rotations_used)

    def vpn_switches_remaining(self) -> Optional[int]:
        """Get remaining VPN switches.

        Returns:
            Number of VPN switches remaining, or None if unlimited
        """
        if self.max_vpn_switches <= 0:
            return None
        return max(0, self.max_vpn_switches - self.vpn_switches_used)

    def backoff_time_remaining(self) -> Optional[float]:
        """Get remaining backoff time budget.

        Returns:
            Seconds of backoff remaining, or None if unlimited
        """
        if self.max_backoff_time <= 0:
            return None
        return max(0.0, self.max_backoff_time - self.backoff_time_spent)

    def set_keyword_priority(self, keyword: str, priority: PriorityLevel) -> None:
        """Set the priority level for a keyword.

        Args:
            keyword: The keyword to set priority for
            priority: The priority level (high, medium, low)
        """
        self.keyword_priorities[keyword] = priority
        if keyword not in self.keyword_consecutive_successes:
            self.keyword_consecutive_successes[keyword] = 0
        logger.debug(f"Budget: set priority for '{keyword}' to {priority.value}")

    def set_keyword_priority_string(self, keyword: str, priority: str) -> None:
        """Set keyword priority from string.

        Args:
            keyword: The keyword to set priority for
            priority: String priority level ('high', 'medium', 'low')
        """
        priority_level = PriorityLevel.from_string(priority)
        self.set_keyword_priority(keyword, priority_level)

    def get_keyword_priority(self, keyword: str) -> PriorityLevel:
        """Get the priority level for a keyword.

        Args:
            keyword: The keyword to get priority for

        Returns:
            PriorityLevel, defaults to MEDIUM if not set
        """
        return self.keyword_priorities.get(keyword, PriorityLevel.MEDIUM)

    def get_budget_for_keyword(self, keyword: str) -> Dict[str, Optional[int]]:
        """Get budget allocation for a specific keyword based on priority.

        Args:
            keyword: The keyword to get budget for

        Returns:
            Dict with 'rotations', 'vpn_switches', 'backoff_time' allocation
        """
        if not self.priority_allocation_enabled:
            # Return full budget if allocation disabled
            return {
                "rotations": self.rotations_remaining(),
                "vpn_switches": self.vpn_switches_remaining(),
                "backoff_time": self.backoff_time_remaining(),
            }

        priority = self.get_keyword_priority(keyword)
        allocation_pct = PRIORITY_ALLOCATION[priority]

        # Check for consecutive success boost
        consecutive = self.keyword_consecutive_successes.get(keyword, 0)
        if consecutive >= self.consecutive_success_boost_threshold:
            allocation_pct = min(1.0, allocation_pct * self.consecutive_success_boost_multiplier)
            logger.debug(
                f"Budget: high-priority boost applied for '{keyword}' "
                f"(consecutive successes: {consecutive})"
            )

        return {
            "rotations": self._allocated_remaining(self.rotations_remaining(), allocation_pct),
            "vpn_switches": self._allocated_remaining(self.vpn_switches_remaining(), allocation_pct),
            "backoff_time": self._allocated_remaining(self.backoff_time_remaining(), allocation_pct),
        }

    def _allocated_remaining(self, remaining: Optional[int | float], percentage: float) -> Optional[int | float]:
        """Apply percentage allocation to remaining budget.

        Args:
            remaining: The remaining budget (None if unlimited)
            percentage: The percentage to allocate (0.0-1.0)

        Returns:
            Allocated amount, or None if unlimited
        """
        if remaining is None:
            return None
        return max(0, int(remaining * percentage)) if isinstance(remaining, int) else max(0.0, remaining * percentage)

    def can_attempt_for_keyword(self, keyword: str, attempt_cost: int = 1) -> bool:
        """Check if a keyword can attempt a download within its priority budget.

        Args:
            keyword: The keyword to check
            attempt_cost: The cost of this attempt (default 1)

        Returns:
            True if keyword has budget remaining for this attempt
        """
        budget = self.get_budget_for_keyword(keyword)

        # Check rotations budget
        if budget["rotations"] is not None and budget["rotations"] < attempt_cost:
            return False

        # Check VPN budget
        if budget["vpn_switches"] is not None and budget["vpn_switches"] < attempt_cost:
            return False

        # Check backoff budget (use minimum expected backoff of 1 second)
        if budget["backoff_time"] is not None and budget["backoff_time"] < 1.0:
            return False

        return True

    def predict_exhaustion(self, keyword: str, attempts_remaining: int = 5) -> Optional[Dict]:
        """Predict when a keyword will exhaust its budget.

        Args:
            keyword: The keyword to predict exhaustion for
            attempts_remaining: Number of attempts to simulate (default 5)

        Returns:
            Dict with exhaustion prediction or None if not predictable:
            - will_exhaust: bool
            - attempts_until_exhaustion: int or None
            - warning: str or None
        """
        budget = self.get_budget_for_keyword(keyword)

        # Calculate effective remaining budget
        effective_rotations = budget["rotations"] if budget["rotations"] is not None else float('inf')
        effective_vpn = budget["vpn_switches"] if budget["vpn_switches"] is not None else float('inf')
        effective_backoff = budget["backoff_time"] if budget["backoff_time"] is not None else float('inf')

        # Simple model: predict based on rotations (most common bottleneck)
        if effective_rotations == float('inf'):
            return None  # Unlimited budget

        # Estimate attempts until exhaustion
        # Assume each failed download uses ~1 rotation worth of budget
        attempts_until_exhaustion = int(effective_rotations)

        will_exhaust = attempts_until_exhaustion <= attempts_remaining
        warning = None
        if will_exhaust:
            warning = (
                f"Keyword '{keyword}' will exhaust budget in ~{attempts_until_exhaustion} attempts "
                f"(priority: {self.get_keyword_priority(keyword).value})"
            )
            logger.warning(warning)

        return {
            "will_exhaust": will_exhaust,
            "attempts_until_exhaustion": attempts_until_exhaustion,
            "warning": warning,
        }

    def record_success_for_keyword(self, keyword: str) -> None:
        """Record success for a keyword and track consecutive successes.

        Args:
            keyword: The keyword that succeeded
        """
        self.record_success(keyword)
        # Track consecutive successes for priority boost
        self.keyword_consecutive_successes[keyword] = (
            self.keyword_consecutive_successes.get(keyword, 0) + 1
        )
        logger.debug(
            f"Budget: consecutive successes for '{keyword}': "
            f"{self.keyword_consecutive_successes[keyword]}"
        )

    def record_failure_for_keyword(self, keyword: str) -> None:
        """Record failure for a keyword and reset consecutive successes.

        Args:
            keyword: The keyword that failed
        """
        self.record_failure(keyword)
        # Reset consecutive successes on failure
        self.keyword_consecutive_successes[keyword] = 0
        logger.debug(f"Budget: consecutive successes reset for '{keyword}'")

    def steal_unused_budget(self, donor_keywords: List[str], recipient_keyword: str) -> None:
        """Allow high-priority keyword to steal unused budget from low-priority keywords.

        This is called when a high-priority keyword is about to exhaust its budget
        but other keywords have unused budget.

        Args:
            donor_keywords: Keywords that may have unused budget
            recipient_keyword: High-priority keyword needing budget
        """
        recipient_priority = self.get_keyword_priority(recipient_keyword)
        if recipient_priority != PriorityLevel.HIGH:
            logger.debug(f"Budget steal: recipient '{recipient_keyword}' is not high-priority")
            return

        # Check if recipient actually needs budget
        if self.can_attempt_for_keyword(recipient_keyword):
            return  # No need to steal

        # Find donors with unused budget (low-priority keywords)
        for donor in donor_keywords:
            donor_priority = self.get_keyword_priority(donor)
            if donor_priority == PriorityLevel.LOW:
                donor_budget = self.get_budget_for_keyword(donor)
                # If donor has unused budget, log the steal opportunity
                if donor_budget["rotations"] and donor_budget["rotations"] > 2:
                    logger.info(
                        f"Budget steal opportunity: low-priority '{donor}' has "
                        f"{donor_budget['rotations']} rotations unused"
                    )

    def get_recommended_escalation(self) -> str:
        """Get recommended next escalation level based on budget state.

        Returns:
            Recommended action: 'backoff', 'cookie', 'vpn', or 'exhausted'
        """
        # Check if backoff budget remains
        if self.can_backoff(5.0):  # Assume 5s minimum backoff
            return "backoff"

        # Check if cookie rotation is available
        if self.can_rotate():
            return "cookie"

        # Check if VPN switching is available
        if self.can_switch_vpn():
            return "vpn"

        # All options exhausted
        return "exhausted"

    def get_budget_status(self) -> Dict:
        """Get detailed budget status for diagnostics.

        Returns:
            Dict with current budget state including:
            - rotations: used/remaining/max
            - vpn_switches: used/remaining/max
            - backoff_time: spent/remaining/max
            - tier_status: status of each escalation tier (if enabled)
            - is_exhausted: overall exhaustion state
            - is_nearly_exhausted: state when any resource >80% used
        """
        return {
            "rotations": {
                "used": self.rotations_used,
                "remaining": self.rotations_remaining(),
                "max": self.max_rotations if self.max_rotations > 0 else None,
                "unlimited": self.max_rotations <= 0,
            },
            "vpn_switches": {
                "used": self.vpn_switches_used,
                "remaining": self.vpn_switches_remaining(),
                "max": self.max_vpn_switches if self.max_vpn_switches > 0 else None,
                "unlimited": self.max_vpn_switches <= 0,
            },
            "backoff_time": {
                "spent": round(self.backoff_time_spent, 1),
                "remaining": round(self.backoff_time_remaining(), 1) if self.backoff_time_remaining() is not None else None,
                "max": self.max_backoff_time if self.max_backoff_time > 0 else None,
                "unlimited": self.max_backoff_time <= 0,
            },
            "tier_status": self.get_tier_status(),
            "is_exhausted": self.is_exhausted(),
            "is_nearly_exhausted": self.is_nearly_exhausted(),
            "keywords_affected": len(self.keywords_rate_limited),
            "success_rate": round(self.successes / (self.successes + self.failures), 2) if (self.successes + self.failures) > 0 else None,
        }

    def get_budget_advice(self) -> str:
        """Get actionable advice based on current budget exhaustion state.

        Returns advice strings for callers to decide next action:
        - 'continue': Budget resources still available, proceed normally
        - 'skip_to_vpn': Cookie rotations exhausted, skip directly to VPN
        - 'abort_keyword': All recovery resources exhausted, abandon keyword

        Returns:
            Advice string: 'continue', 'skip_to_vpn', or 'abort_keyword'
        """
        if not self.can_rotate():
            if not self.can_switch_vpn():
                return "abort_keyword"
            return "skip_to_vpn"
        return "continue"

    def is_nearly_exhausted(self) -> bool:
        """Check if any single resource exceeds 80% usage.

        Used by the circuit breaker to extend pause duration when budget
        is running low but not yet fully exhausted.

        Returns:
            True if any resource (rotations, VPN switches, or backoff time)
            exceeds 80% of its budget limit.
        """
        # Check rotation budget (only if limited)
        if self.max_rotations > 0:
            if self.rotations_used / self.max_rotations > 0.8:
                return True

        # Check VPN switches budget (only if limited)
        if self.max_vpn_switches > 0:
            if self.vpn_switches_used / self.max_vpn_switches > 0.8:
                return True

        # Check backoff time budget (only if limited)
        if self.max_backoff_time > 0:
            if self.backoff_time_spent / self.max_backoff_time > 0.8:
                return True

        return False

    def is_exhausted(self) -> bool:
        """Check if all rate limit budget is exhausted.

        Returns:
            True if no recovery options remain
        """
        return self.get_recommended_escalation() == "exhausted"

    def get_exhaustion_details(self) -> Optional[Dict]:
        """Get details about which budget limit was hit.

        Returns:
            Dict with exhaustion details if exhausted, None otherwise:
            - is_exhausted: bool
            - exhausted_resources: list of resource names that are exhausted
            - remaining_options: list of options still available
            - recommendation: recommended action
        """
        if not self.is_exhausted():
            return None

        exhausted = []
        remaining = []

        # Check each resource - use minimum expected backoff of 5 seconds
        # to match logic in get_recommended_escalation
        if not self.can_rotate():
            exhausted.append("rotations")
        else:
            remaining.append("rotations")

        if not self.can_switch_vpn():
            exhausted.append("vpn_switches")
        else:
            remaining.append("vpn_switches")

        if not self.can_backoff(5.0):  # Use 5s minimum like get_recommended_escalation
            exhausted.append("backoff_time")
        else:
            remaining.append("backoff_time")

        # Log the exhaustion event
        logger.warning(
            f"BUDGET EXHAUSTED: {', '.join(exhausted)} exhausted. "
            f"Remaining: {', '.join(remaining) if remaining else 'none'}. "
            f"Keywords affected: {len(self.keywords_rate_limited)}"
        )

        return {
            "is_exhausted": True,
            "exhausted_resources": exhausted,
            "remaining_options": remaining,
            "recommendation": self.get_recommended_escalation(),
            "keywords_affected": len(self.keywords_rate_limited),
        }

    def get_summary(self) -> Dict:
        """Get a summary of budget usage for reporting.

        Returns:
            Dict with budget usage statistics
        """
        # Count keywords by priority
        priority_counts = {p.value: 0 for p in PriorityLevel}
        for kw, priority in self.keyword_priorities.items():
            priority_counts[priority.value] += 1

        return {
            "rotations_used": self.rotations_used,
            "rotations_remaining": self.rotations_remaining(),
            "vpn_switches_used": self.vpn_switches_used,
            "vpn_switches_remaining": self.vpn_switches_remaining(),
            "backoff_time_spent": round(self.backoff_time_spent, 1),
            "backoff_time_remaining": (
                round(self.backoff_time_remaining(), 1)
                if self.backoff_time_remaining() is not None
                else None
            ),
            "keywords_affected": len(self.keywords_rate_limited),
            "keyword_priorities": priority_counts,
            "priority_allocation_enabled": self.priority_allocation_enabled,
            "last_escalation": self.last_escalation_level,
            "is_exhausted": self.is_exhausted(),
            "successes": self.successes,
            "failures": self.failures,
        }

    def to_dict(self) -> Dict:
        """Serialize budget state for checkpoint persistence.

        Returns:
            Dict with all budget state data
        """
        result = {
            "rotations_used": self.rotations_used,
            "vpn_switches_used": self.vpn_switches_used,
            "backoff_time_spent": self.backoff_time_spent,
            "keywords_rate_limited": list(self.keywords_rate_limited),
            "last_escalation_level": self.last_escalation_level,
            "max_rotations": self.max_rotations,
            "max_vpn_switches": self.max_vpn_switches,
            "max_backoff_time": self.max_backoff_time,
            "successes": self.successes,
            "failures": self.failures,
            "keyword_priorities": {k: v.value for k, v in self.keyword_priorities.items()},
            "keyword_consecutive_successes": dict(self.keyword_consecutive_successes),
            "priority_allocation_enabled": self.priority_allocation_enabled,
            "tier_isolation_enabled": self.tier_isolation_enabled,
        }

        # Add tier budget state if enabled
        if self.tier_isolation_enabled and self.tier_budget_manager is not None:
            result["tier_budgets"] = self.tier_budget_manager.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "RateLimitBudget":
        """Create budget from checkpoint data.

        Args:
            data: Checkpoint data dict (may be None for new sessions)

        Returns:
            RateLimitBudget with restored state
        """
        if not data:
            return cls()

        budget = cls(
            rotations_used=data.get("rotations_used", 0),
            vpn_switches_used=data.get("vpn_switches_used", 0),
            backoff_time_spent=data.get("backoff_time_spent", 0.0),
            keywords_rate_limited=data.get("keywords_rate_limited", []),
            last_escalation_level=data.get("last_escalation_level", "none"),
            successes=data.get("successes", 0),
            failures=data.get("failures", 0),
        )

        # Restore tier isolation setting
        budget.tier_isolation_enabled = data.get("tier_isolation_enabled", False)

        # Restore tier budget manager if enabled
        if budget.tier_isolation_enabled and "tier_budgets" in data:
            budget.tier_budget_manager = TierBudgetManager.from_dict(data.get("tier_budgets"))

        # Restore budget limits
        budget.max_rotations = data.get("max_rotations", 0)
        budget.max_vpn_switches = data.get("max_vpn_switches", 10)
        budget.max_backoff_time = data.get("max_backoff_time", 300.0)

        # Restore priority-related fields
        priorities_data = data.get("keyword_priorities", {})
        budget.keyword_priorities = {
            k: PriorityLevel.from_string(v) for k, v in priorities_data.items()
        }
        budget.keyword_consecutive_successes = data.get("keyword_consecutive_successes", {})
        budget.priority_allocation_enabled = data.get("priority_allocation_enabled", True)

        return budget

    def clear(self) -> None:
        """Clear all budget state for a new session."""
        self.rotations_used = 0
        self.vpn_switches_used = 0
        self.backoff_time_spent = 0.0
        self.keywords_rate_limited = []
        self.last_escalation_level = "none"
        self.successes = 0
        self.failures = 0
        self.keyword_priorities = {}
        self.keyword_consecutive_successes = {}

        # Clear tier budgets if enabled
        if self.tier_isolation_enabled and self.tier_budget_manager is not None:
            self.tier_budget_manager.clear()

        logger.debug("Rate limit budget cleared")

    def reset_on_ip_change(self) -> None:
        """Reset rotation and backoff budgets after VPN IP change.

        When the VPN rotates to a new IP address, rate limits from YouTube
        are tied to the old IP. This method resets the budgets that benefit
        from a fresh IP while preserving VPN rotation count (to track total
        VPN rotations in the session).

        Resets:
            - rotations_used (cookie rotations)
            - backoff_time_spent
            - last_escalation_level (reset to 'none')

        Preserves:
            - vpn_switches_used (tracks total VPN rotations)
            - keywords_rate_limited (historical tracking)
            - successes/failures (session statistics)
            - max_* limits (budget configuration)
        """
        old_rotations = self.rotations_used
        old_backoff = self.backoff_time_spent

        self.rotations_used = 0
        self.backoff_time_spent = 0.0
        self.last_escalation_level = "none"

        # Reset tier budgets on IP change if enabled
        if self.tier_isolation_enabled and self.tier_budget_manager is not None:
            self.tier_budget_manager.clear()
            logger.debug("Tier budgets cleared on IP change")

        logger.info(
            f"Budget reset on IP change: rotations {old_rotations}→0, "
            f"backoff {old_backoff:.1f}s→0s (VPN rotations: {self.vpn_switches_used})"
        )

    def budget_summary(self) -> str:
        """Get a formatted budget summary string for logging.

        Returns a human-readable summary showing:
        - Rotations used vs max with percentage
        - VPN switches used vs max with percentage
        - Backoff time spent vs max with percentage
        - Keywords that triggered rate limit events

        Returns:
            Formatted summary string suitable for logging.
        """
        parts = []

        # Rotations (cookie rotations) - only show if any were used
        if self.rotations_used > 0:
            if self.max_rotations > 0:
                rotation_pct = (self.rotations_used / self.max_rotations) * 100
                parts.append(
                    f"rotations {self.rotations_used}/{self.max_rotations} ({rotation_pct:.0f}%)"
                )
            else:
                parts.append(f"rotations {self.rotations_used}/unlimited")

        # VPN switches - only show if any were used
        if self.vpn_switches_used > 0:
            if self.max_vpn_switches > 0:
                vpn_pct = (self.vpn_switches_used / self.max_vpn_switches) * 100
                parts.append(
                    f"VPN switches {self.vpn_switches_used}/{self.max_vpn_switches} ({vpn_pct:.0f}%)"
                )
            else:
                parts.append(f"VPN switches {self.vpn_switches_used}/unlimited")

        # Backoff time - only show if any time was spent
        if self.backoff_time_spent > 0:
            if self.max_backoff_time > 0:
                backoff_pct = (self.backoff_time_spent / self.max_backoff_time) * 100
                parts.append(
                    f"backoff {self.backoff_time_spent:.0f}s/{self.max_backoff_time:.0f}s ({backoff_pct:.0f}%)"
                )
            else:
                parts.append(f"backoff {self.backoff_time_spent:.0f}s/unlimited")

        # Build summary string
        if parts:
            summary = f"Budget used: {', '.join(parts)}"
        else:
            summary = "Budget used: none"

        # Add keywords that triggered rate limit events
        if self.keywords_rate_limited:
            # Show up to 5 keywords to keep log readable
            keywords_display = self.keywords_rate_limited[:5]
            if len(self.keywords_rate_limited) > 5:
                keywords_display.append(f"... +{len(self.keywords_rate_limited) - 5} more")
            summary += f" | Rate-limited keywords: {', '.join(keywords_display)}"

        return summary

    # === US-123-007: Cross-session rate limit state persistence ===

    def serialize_rate_limit_state(self) -> Dict:
        """
        Serialize rate limit state for checkpoint persistence.

        Exports the current rate limit budget state along with a timestamp
        for staleness detection during restoration.

        Returns:
            Dict with serialized rate limit state including timestamp
        """
        import time

        state = self.to_dict()
        # Add timestamp for staleness detection
        state["saved_at"] = time.time()
        state["saved_at_iso"] = datetime.now().isoformat()

        logger.debug(
            f"Serialized rate limit state: rotations={self.rotations_used}, "
            f"vpn_switches={self.vpn_switches_used}, backoff_time={self.backoff_time_spent:.1f}s"
        )
        return state

    @classmethod
    def deserialize_rate_limit_state(
        cls,
        data: Optional[Dict],
        stale_state_threshold: float = 3600.0,
    ) -> Optional["RateLimitBudget"]:
        """
        Deserialize rate limit state from checkpoint data.

        Automatically filters out stale state entries that are older than
        the stale_state_threshold to prevent using outdated rate limit info.

        Args:
            data: Checkpoint data dict with rate limit state
            stale_state_threshold: Seconds after which state is considered stale
                                  and skipped (default: 1 hour)

        Returns:
            RateLimitBudget instance with restored state, or None if data is empty/stale
        """
        import time

        if not data:
            logger.debug("No rate limit state data to deserialize")
            return None

        # Check for staleness
        saved_at = data.get("saved_at")
        if saved_at is not None:
            age = time.time() - saved_at
            if age > stale_state_threshold:
                logger.info(
                    f"Skipping stale rate limit state: age={age:.0f}s "
                    f"(threshold={stale_state_threshold:.0f}s)"
                )
                return None
            logger.debug(f"Restoring fresh rate limit state: age={age:.0f}s")

        # Remove internal metadata before passing to from_dict
        state_data = {k: v for k, v in data.items() if k not in ("saved_at", "saved_at_iso")}

        budget = cls.from_dict(state_data)

        logger.info(
            f"Restored rate limit state: rotations={budget.rotations_used}, "
            f"vpn_switches={budget.vpn_switches_used}, backoff_time={budget.backoff_time_spent:.1f}s"
        )
        return budget
