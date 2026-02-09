"""Composable pause duration calculator for circuit breakers.

Extracted from CircuitBreaker's 5-step pause pipeline to make pause logic
independently testable and reusable. Each step is a standalone method that
takes a PauseContext and returns an updated pause value.

Pipeline order: base_pause → escalation_adjusted → budget_adjusted → apply_jitter → cap_duration

Extracted as part of US-82-008.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .escalation_manager import EscalationManager
    from .rate_limit_budget import RateLimitBudget

# Use the circuit_breaker logger to preserve backward-compatible log filtering
logger = logging.getLogger('src.downloader.circuit_breaker')


@dataclass
class PauseContext:
    """Context for pause calculation pipeline steps.

    Carries the current pause value plus all inputs needed by
    individual pipeline steps. Each step reads from this context
    and returns the updated pause value.
    """
    current_pause: float = 0.0
    base_pause_seconds: float = 60.0
    max_pause_seconds: float = 300.0
    jitter_factor: float = 0.2
    escalation_manager: Optional['EscalationManager'] = None
    budget: Optional['RateLimitBudget'] = None


class PauseCalculator:
    """Composable pause duration calculator.

    Composes 5 steps into a pipeline:
    1. base_pause() - Get base duration from config
    2. escalation_adjusted() - Double pause when >50% keywords at Tier 3
    3. budget_adjusted() - Scale pause based on rate limit budget state
    4. apply_jitter() - Randomize ±jitter_factor to prevent thundering herd
    5. cap_duration() - Enforce max_pause_seconds ceiling

    Each step can be called independently for unit testing, or composed
    via calculate() for the full pipeline.
    """

    def __init__(self) -> None:
        self._last_jitter_applied: float = 0.0

    @property
    def last_jitter_applied(self) -> float:
        """Last jitter multiplier offset applied (for debugging/metrics)."""
        return self._last_jitter_applied

    def calculate(self, ctx: PauseContext) -> float:
        """Run the full 5-step pause calculation pipeline.

        Args:
            ctx: PauseContext with all inputs for the pipeline.

        Returns:
            Final pause duration in seconds.
        """
        pause = self.base_pause(ctx)
        pause = self.escalation_adjusted(pause, ctx)
        pause = self.budget_adjusted(pause, ctx)
        pause = self.apply_jitter(pause, ctx)
        pause = self.cap_duration(pause, ctx)
        return pause

    def base_pause(self, ctx: PauseContext) -> float:
        """Step 1: Get the base pause duration from context."""
        return ctx.base_pause_seconds

    def escalation_adjusted(self, pause: float, ctx: PauseContext) -> float:
        """Step 2: Apply escalation-based extension to a pause duration.

        Doubles the pause when >50% of active keywords are at Tier 3.
        """
        if ctx.escalation_manager is None:
            return pause

        try:
            from .types import EscalationTier

            total_keywords = ctx.escalation_manager.get_active_keyword_count()
            if total_keywords > 0:
                tier3_keywords = ctx.escalation_manager.get_keywords_at_tier(
                    EscalationTier.FULL_BYPASS
                )
                tier3_pct = len(tier3_keywords) / total_keywords

                if tier3_pct > 0.5:
                    adjusted = pause * 2.0
                    logger.info(
                        f"Circuit breaker extended: {tier3_pct:.0%} keywords at Tier 3 "
                        f"(pause {ctx.base_pause_seconds:.0f}s -> {adjusted:.0f}s)"
                    )
                    return adjusted
        except ImportError:
            pass

        return pause

    def budget_adjusted(self, pause: float, ctx: PauseContext) -> float:
        """Step 3: Apply budget-aware extension to a pause duration."""
        if ctx.budget is None:
            return pause

        original_pause = pause
        if ctx.budget.is_exhausted():
            pause = pause * 2.5
            budget_status = "exhausted"
        elif ctx.budget.is_nearly_exhausted():
            pause = pause * 1.5
            budget_status = "nearly exhausted"
        else:
            budget_status = None

        if budget_status is not None:
            logger.info(
                f"Circuit breaker pause extended {original_pause:.0f}s -> "
                f"{pause:.0f}s (budget {budget_status})"
            )

        return pause

    def apply_jitter(self, delay: float, ctx: PauseContext) -> float:
        """Step 4: Apply random jitter to a delay value.

        Jitter helps prevent thundering herd when multiple downloads
        resume simultaneously after circuit breaker recovery.

        The jitter formula is: delay * (1 + random.uniform(-jitter, +jitter))
        """
        jitter_factor = ctx.jitter_factor

        if jitter_factor < 0.0:
            jitter_factor = 0.0
        elif jitter_factor > 1.0:
            jitter_factor = 1.0

        if jitter_factor > 0.0:
            jitter_multiplier = 1 + random.uniform(-jitter_factor, jitter_factor)
            jittered_delay = delay * jitter_multiplier
            self._last_jitter_applied = jitter_multiplier - 1.0
        else:
            jittered_delay = delay
            self._last_jitter_applied = 0.0

        return jittered_delay

    def cap_duration(self, pause: float, ctx: PauseContext) -> float:
        """Step 5: Cap a pause duration at max_pause_seconds."""
        max_pause = ctx.max_pause_seconds
        if pause > max_pause:
            logger.debug(
                f"Circuit breaker pause capped: {pause:.1f}s -> {max_pause:.0f}s "
                f"(max_pause_seconds={max_pause:.0f})"
            )
            pause = max_pause
        return pause
