"""
Cross-keyword rate limit budget tracking.

Tracks rate limit recovery resources used across all keywords in a download session.
When processing multiple keywords, rate limit events from keyword A affect the budget
available for keyword B.

This prevents wasting resources:
- If keyword A exhausted all cookie rotations, keyword B should skip directly to VPN
- If total backoff time exceeds budget, skip backoff and escalate immediately
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class RateLimitBudget:
    """Tracks rate limit recovery resources used across all keywords.

    When share_budget_across_keywords is enabled (default), all keywords in a
    download session share the same budget. This prevents wasting resources:

    - If keyword A exhausted all cookie rotations, keyword B should skip
      directly to VPN switching instead of trying rotations again.
    - If total backoff time exceeds the budget, skip backoff and escalate.

    Attributes:
        rotations_used: Number of cookie rotations used this session
        vpn_switches_used: Number of VPN switches used this session
        backoff_time_spent: Total time spent in backoff delays (seconds)
        keywords_rate_limited: Keywords that triggered rate limit events
        last_escalation_level: Current escalation level (backoff, cookie, vpn)
    """
    rotations_used: int = 0
    vpn_switches_used: int = 0
    backoff_time_spent: float = 0.0
    keywords_rate_limited: List[str] = field(default_factory=list)
    last_escalation_level: str = "none"  # none, backoff, cookie, vpn
    successes: int = 0
    failures: int = 0

    # Budget limits (set from config)
    max_rotations: int = 0  # 0 = unlimited
    max_vpn_switches: int = 10
    max_backoff_time: float = 300.0  # 5 minutes total backoff budget

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
        return budget

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

    def is_exhausted(self) -> bool:
        """Check if all rate limit budget is exhausted.

        Returns:
            True if no recovery options remain
        """
        return self.get_recommended_escalation() == "exhausted"

    def get_summary(self) -> Dict:
        """Get a summary of budget usage for reporting.

        Returns:
            Dict with budget usage statistics
        """
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
        return {
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
        }

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

        # Restore budget limits
        budget.max_rotations = data.get("max_rotations", 0)
        budget.max_vpn_switches = data.get("max_vpn_switches", 10)
        budget.max_backoff_time = data.get("max_backoff_time", 300.0)

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
        logger.debug("Rate limit budget cleared")
