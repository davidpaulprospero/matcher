"""
Escalation strategy - decision logic for tier progression.

Separates 'what to escalate' (strategy decisions) from 'how to escalate'
(manager execution). The EscalationStrategy class encapsulates:
  - Level progression logic (when to advance tiers)
  - Cooldown checks (when escalation is suppressed)
  - Trigger matching (which errors warrant escalation)

This class is stateless regarding execution - it reads state and config
to make decisions, but does not modify state directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .types import EscalationState, EscalationTier

if TYPE_CHECKING:
    from ..config.sections.download import ExtractorArgsConfig

logger = logging.getLogger(__name__)


@dataclass
class EscalationDecision:
    """Result of an escalation strategy decision.

    Attributes:
        should_escalate: Whether escalation should proceed.
        target_tier: The tier to escalate to (if should_escalate is True).
        reason: Human-readable reason for the decision.
        skip_to_max: If True, skip intermediate tiers and go to max tier.
    """
    should_escalate: bool
    target_tier: Optional[EscalationTier] = None
    reason: str = ""
    skip_to_max: bool = False


class EscalationStrategy:
    """Strategy for making escalation decisions.

    This class encapsulates all decision logic for escalation:
    - When to escalate (threshold, cooldown)
    - What tier to escalate to (level progression)
    - Whether to skip tiers (budget exhaustion)

    The strategy is configured via ExtractorArgsConfig and consulted
    by EscalationManager to make escalation decisions.

    Args:
        config: ExtractorArgsConfig with escalation settings.
            Expected attributes:
            - escalation_threshold (int): 403 count before escalation
            - cooldown_seconds (float): Minimum time between escalations
            - max_tier (int): Maximum tier value (default: 3)
    """

    def __init__(self, config: Optional["ExtractorArgsConfig"] = None):
        self._config = config

    @property
    def threshold(self) -> int:
        """Get the escalation threshold (403 count before escalation)."""
        if self._config is None:
            return 2
        return getattr(self._config, 'escalation_threshold', 2)

    @property
    def cooldown_seconds(self) -> float:
        """Get the cooldown period in seconds."""
        if self._config is None:
            return 300.0
        return getattr(self._config, 'cooldown_seconds', 300.0)

    @property
    def max_tier(self) -> EscalationTier:
        """Get the maximum escalation tier."""
        if self._config is None:
            return EscalationTier.VPN_ROTATION
        max_val = getattr(self._config, 'max_tier', 4)
        # Clamp to valid range (1-4)
        max_val = max(1, min(max_val, 4))
        return EscalationTier(max_val)

    def get_next_tier(self, current_tier: EscalationTier) -> Optional[EscalationTier]:
        """Get the next tier in the progression.

        Args:
            current_tier: The current escalation tier.

        Returns:
            The next tier, or None if already at max tier.
        """
        if current_tier >= self.max_tier:
            return None

        next_value = current_tier.value + 1
        if next_value > EscalationTier.VPN_ROTATION.value:
            return None

        return EscalationTier(next_value)

    def is_at_max_tier(self, current_tier: EscalationTier) -> bool:
        """Check if currently at the maximum escalation tier.

        Args:
            current_tier: The current escalation tier.

        Returns:
            True if at max tier, False otherwise.
        """
        return current_tier >= self.max_tier

    def is_past_cooldown(self, state: EscalationState) -> bool:
        """Check if enough time has passed since the last escalation.

        Args:
            state: The per-keyword escalation state.

        Returns:
            True if past cooldown (or never escalated), False if in cooldown.
        """
        if state.last_escalation_time is None:
            return True

        elapsed = time.time() - state.last_escalation_time
        return elapsed >= self.cooldown_seconds

    def get_cooldown_remaining(self, state: EscalationState) -> float:
        """Get remaining cooldown seconds.

        Args:
            state: The per-keyword escalation state.

        Returns:
            Seconds remaining in cooldown, or 0.0 if not in cooldown.
        """
        if state.last_escalation_time is None:
            return 0.0

        elapsed = time.time() - state.last_escalation_time
        remaining = self.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def should_escalate_on_failure(
        self,
        state: EscalationState,
        budget_exhausted: bool = False,
    ) -> EscalationDecision:
        """Decide whether to escalate after a failure.

        This is the main decision method for failure-triggered escalation.
        It checks:
        1. Whether the 403 threshold has been reached
        2. Whether the cooldown period has passed
        3. Whether to skip to max tier (if budget exhausted)

        Args:
            state: The per-keyword escalation state (after incrementing 403 count).
            budget_exhausted: If True and escalation is warranted, skip to max tier.

        Returns:
            EscalationDecision with the escalation verdict.
        """
        # Check threshold
        if not state.should_escalate(self.threshold):
            return EscalationDecision(
                should_escalate=False,
                reason=f"Below threshold ({state.consecutive_403s}/{self.threshold})",
            )

        # Check if already at max tier
        if self.is_at_max_tier(state.current_tier):
            return EscalationDecision(
                should_escalate=False,
                reason="Already at max tier",
            )

        # Check cooldown
        if not self.is_past_cooldown(state):
            remaining = self.get_cooldown_remaining(state)
            return EscalationDecision(
                should_escalate=False,
                reason=f"Cooldown active ({remaining:.0f}s remaining)",
            )

        # Determine target tier
        if budget_exhausted:
            # Skip to max tier when budget exhausted
            return EscalationDecision(
                should_escalate=True,
                target_tier=self.max_tier,
                reason="Budget exhausted, skipping to max tier",
                skip_to_max=True,
            )

        # Normal escalation to next tier
        next_tier = self.get_next_tier(state.current_tier)
        if next_tier is None:
            return EscalationDecision(
                should_escalate=False,
                reason="No next tier available",
            )

        return EscalationDecision(
            should_escalate=True,
            target_tier=next_tier,
            reason=f"Threshold reached ({state.consecutive_403s} 403s)",
        )

    def should_escalate_on_slow_speed(
        self,
        state: EscalationState,
        consecutive_slow_count: int,
        slow_threshold: int = 3,
    ) -> EscalationDecision:
        """Decide whether to escalate due to slow download speed.

        Speed-triggered escalation is preemptive and does not consume
        budget rotations.

        Args:
            state: The per-keyword escalation state.
            consecutive_slow_count: Number of consecutive slow speed signals.
            slow_threshold: Number of signals before escalation (default: 3).

        Returns:
            EscalationDecision with the escalation verdict.
        """
        if consecutive_slow_count < slow_threshold:
            return EscalationDecision(
                should_escalate=False,
                reason=f"Below slow speed threshold ({consecutive_slow_count}/{slow_threshold})",
            )

        if self.is_at_max_tier(state.current_tier):
            return EscalationDecision(
                should_escalate=False,
                reason="Already at max tier",
            )

        if not self.is_past_cooldown(state):
            remaining = self.get_cooldown_remaining(state)
            return EscalationDecision(
                should_escalate=False,
                reason=f"Cooldown active ({remaining:.0f}s remaining)",
            )

        next_tier = self.get_next_tier(state.current_tier)
        if next_tier is None:
            return EscalationDecision(
                should_escalate=False,
                reason="No next tier available",
            )

        return EscalationDecision(
            should_escalate=True,
            target_tier=next_tier,
            reason=f"Sustained slow speed ({consecutive_slow_count} signals)",
        )

    def should_shortcut_to_max(
        self,
        current_tier: EscalationTier,
        circuit_breaker_open: bool,
    ) -> EscalationDecision:
        """Decide whether to shortcut to max tier due to circuit breaker.

        When the circuit breaker is open, we skip lower tiers entirely
        and go straight to full bypass mode.

        Args:
            current_tier: The current escalation tier.
            circuit_breaker_open: Whether the circuit breaker is open.

        Returns:
            EscalationDecision indicating whether to shortcut.
        """
        if not circuit_breaker_open:
            return EscalationDecision(
                should_escalate=False,
                reason="Circuit breaker not open",
            )

        if current_tier >= EscalationTier.FULL_BYPASS:
            return EscalationDecision(
                should_escalate=False,
                reason="Already at full bypass",
            )

        return EscalationDecision(
            should_escalate=True,
            target_tier=EscalationTier.FULL_BYPASS,
            reason="Circuit breaker open, shortcutting to full bypass",
            skip_to_max=True,
        )
