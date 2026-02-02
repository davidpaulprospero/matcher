"""
4-tier escalation manager for yt-dlp bypass orchestration.

Manages per-keyword escalation state across four tiers:
  - Tier 1 (IMPERSONATE_ONLY): --impersonate only (delegated to ImpersonationManager)
  - Tier 2 (EXTRACTOR_ARGS): --impersonate + --extractor-args player_client
  - Tier 3 (FULL_BYPASS): Both + cookie rotation flag
  - Tier 4 (VPN_ROTATION): All above + Mullvad VPN server rotation

On consecutive 403/bot-detection errors, the manager escalates to the next tier.
Escalation is sticky per-session: success resets the 403 counter but does not
de-escalate to a lower tier.

Thread-safe: uses one threading.Lock per keyword for concurrent download access.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .types import EscalationState, EscalationTier

logger = logging.getLogger(__name__)

# Compiled regex for 403/bot-detection patterns in yt-dlp stderr output.
# These are SEPARATE from ERROR_SEVERITY_PATTERNS in core.py, which handles
# rate-limit severity classification. Escalation triggers specifically detect
# when yt-dlp is being blocked and a higher bypass tier is needed.
_ESCALATION_TRIGGER_RE = re.compile(
    # 403 patterns (original)
    r'HTTP Error 403'
    # Bot/captcha patterns (original)
    r'|Sign in to confirm'
    r'|bot'
    r'|captcha'
    r'|blocked'
    r'|verify you are human'
    # 429 / rate-limit patterns
    r'|HTTP Error 429'
    r'|429'
    r'|Too Many Requests'
    r'|rate.?limit'
    # IP-based restriction patterns
    r'|IP address'
    r'|ip.*block'
    r'|access denied'
    r'|geo.?block'
    # Age-gate patterns
    r'|age.?gate'
    r'|age.?restrict'
    r'|sign.*in.*to.*confirm.*age',
    re.IGNORECASE,
)

# Category-specific compiled regexes for classify_trigger().
# Order matters: more specific patterns checked first.
_TRIGGER_CATEGORIES: List[Tuple[str, "re.Pattern[str]"]] = [
    ('429', re.compile(
        r'HTTP Error 429|Too Many Requests|rate.?limit',
        re.IGNORECASE,
    )),
    ('age_gate', re.compile(
        r'age.?gate|age.?restrict|sign.*in.*to.*confirm.*age',
        re.IGNORECASE,
    )),
    ('ip_blocked', re.compile(
        r'IP address|ip.*block|access denied|geo.?block',
        re.IGNORECASE,
    )),
    ('bot_detection', re.compile(
        r'bot|captcha|verify you are human',
        re.IGNORECASE,
    )),
    ('403', re.compile(
        r'HTTP Error 403|Sign in to confirm|blocked',
        re.IGNORECASE,
    )),
]


def is_escalation_trigger(stderr_output: str) -> bool:
    """Check if yt-dlp stderr output indicates a 403/bot-detection error.

    This is used by download call sites (core.py, audio_first.py, etc.) to
    decide whether to call ``record_failure()`` on the EscalationManager.

    Note: This is a separate concern from ``ERROR_SEVERITY_PATTERNS`` in
    ``core.py``, which classifies error *severity* for backoff timing.
    ``is_escalation_trigger`` detects whether the error warrants *escalation*
    to a higher bypass tier.

    Args:
        stderr_output: Raw stderr text from a yt-dlp subprocess.

    Returns:
        True if the output matches any 403/bot-detection pattern.
    """
    if not stderr_output:
        return False
    return bool(_ESCALATION_TRIGGER_RE.search(stderr_output))


def classify_trigger(stderr_output: str) -> Optional[str]:
    """Classify the escalation trigger category from yt-dlp stderr output.

    Returns a specific category string for metrics granularity, or None
    if the output does not match any known escalation trigger.

    Categories (checked in order of specificity):
        - ``'429'``: Rate-limit / HTTP 429 / Too Many Requests
        - ``'age_gate'``: Age verification required
        - ``'ip_blocked'``: IP-based blocking / geo-blocking / access denied
        - ``'bot_detection'``: Bot / captcha / human verification
        - ``'403'``: HTTP 403 / sign-in / generic blocking

    Args:
        stderr_output: Raw stderr text from a yt-dlp subprocess.

    Returns:
        Category string or None if no trigger matched.
    """
    if not stderr_output:
        return None
    for category, pattern in _TRIGGER_CATEGORIES:
        if pattern.search(stderr_output):
            return category
    return None


# Type alias for the config - avoid circular import by using duck typing.
# Expects: .enabled, .player_clients, .escalation_threshold, .cooldown_seconds, .max_tier
try:
    from ..config.sections.download import ExtractorArgsConfig
except ImportError:  # pragma: no cover
    ExtractorArgsConfig = None  # type: ignore[misc,assignment]

try:
    from ..config.sections.download import MullvadConfig
except ImportError:  # pragma: no cover
    MullvadConfig = None  # type: ignore[misc,assignment]

try:
    from .impersonation import ImpersonationManager
except ImportError:  # pragma: no cover
    ImpersonationManager = None  # type: ignore[misc,assignment]

try:
    from .escalation_strategy import EscalationStrategy
except ImportError:  # pragma: no cover
    EscalationStrategy = None  # type: ignore[misc,assignment]

try:
    from .rate_limit_budget import RateLimitBudget
except ImportError:  # pragma: no cover
    RateLimitBudget = None  # type: ignore[misc,assignment]

try:
    from .circuit_breaker import CircuitBreaker
except ImportError:  # pragma: no cover
    CircuitBreaker = None  # type: ignore[misc,assignment]

try:
    from .mullvad_vpn import MullvadVPN
except ImportError:  # pragma: no cover
    MullvadVPN = None  # type: ignore[misc,assignment]


@dataclass
class EscalationResult:
    """Result from get_escalation_args() with args and metadata.

    Attributes:
        args: List of yt-dlp CLI arguments (--impersonate, --extractor-args, etc.)
        tier: The escalation tier used to generate these args.
        rotate_cookies: If True, caller should trigger cookie rotation (Tier 3).
        rotate_vpn: If True, caller should trigger VPN server rotation (Tier 4).
    """
    args: List[str] = field(default_factory=list)
    tier: EscalationTier = EscalationTier.IMPERSONATE_ONLY
    rotate_cookies: bool = False
    rotate_vpn: bool = False


class EscalationManager:
    """Orchestrates 4-tier yt-dlp bypass escalation per keyword.

    Per-keyword tracking ensures that one keyword hitting 403 errors does not
    affect the escalation state of other keywords.

    Tier progression:
        - Tier 1 (IMPERSONATE_ONLY): --impersonate only
        - Tier 2 (EXTRACTOR_ARGS): --impersonate + --extractor-args player_client
        - Tier 3 (FULL_BYPASS): Both + cookie rotation flag
        - Tier 4 (VPN_ROTATION): All above + VPN server rotation

    Args:
        impersonation_manager: Provides Tier 1 --impersonate args.
        extractor_args_config: Configuration for Tier 2 player_client rotation.
            If None, a default config is used with escalation disabled.
        budget: Optional RateLimitBudget for budget-aware escalation.
            When provided, escalation decisions consult the budget:
            - record_failure() calls budget.record_rotation() on tier advances
            - If budget is exhausted, skip intermediate tiers to max tier
            - get_escalation_args() calls budget.record_attempt()
        mullvad_vpn: Optional MullvadVPN manager for Tier 4 VPN rotation.
        on_vpn_rotation_needed: Optional callback invoked when escalating to Tier 4.
            Called with (keyword: str) to allow caller to handle VPN rotation.
    """

    def __init__(
        self,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
        strategy: Optional["EscalationStrategy"] = None,
        mullvad_vpn: Optional["MullvadVPN"] = None,
        on_vpn_rotation_needed: Optional[Callable[[str], None]] = None,
    ):
        self._impersonation_manager = impersonation_manager
        self._extractor_config = extractor_args_config
        self._budget = budget
        # Create strategy if not provided (for backwards compatibility)
        self._strategy = strategy or EscalationStrategy(extractor_args_config)
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        self._mullvad_vpn: Optional["MullvadVPN"] = mullvad_vpn
        self._on_vpn_rotation_needed = on_vpn_rotation_needed
        self._keyword_states: Dict[str, EscalationState] = {}
        self._keyword_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._total_403s: int = 0
        self._total_successes: int = 0
        self._total_escalations: int = 0
        self._escalations_per_tier: Dict[str, int] = {}  # tier_name -> count
        self._slow_speed_counts: Dict[str, int] = {}  # keyword -> consecutive slow count
        self._speed_escalations: int = 0  # Total speed-triggered escalations
        # Per-keyword escalation timeline: keyword -> [{timestamp, from_tier, to_tier, trigger_category}]
        self._escalation_timeline: Dict[str, List[Dict]] = {}
        # Tier outcome tracking: {trigger_category: {tier_value: {'successes': N, 'attempts': N}}}
        self._tier_outcomes: Dict[str, Dict[int, Dict[str, int]]] = {}

    def _get_lock(self, keyword: str) -> threading.Lock:
        """Get or create a per-keyword lock (thread-safe)."""
        with self._global_lock:
            if keyword not in self._keyword_locks:
                self._keyword_locks[keyword] = threading.Lock()
            return self._keyword_locks[keyword]

    def _get_state(self, keyword: str) -> EscalationState:
        """Get or create the escalation state for a keyword."""
        if keyword not in self._keyword_states:
            self._keyword_states[keyword] = EscalationState()
        return self._keyword_states[keyword]

    def set_circuit_breaker(self, circuit_breaker: "CircuitBreaker") -> None:
        """Link a CircuitBreaker for coordinated rate-limiting.

        When linked, get_escalation_args() will return Tier 3 args
        immediately during circuit breaker pause (skip lower tiers).

        Args:
            circuit_breaker: The CircuitBreaker to consult.
        """
        self._circuit_breaker = circuit_breaker

    def set_mullvad_vpn(self, mullvad_vpn: "MullvadVPN") -> None:
        """Link a MullvadVPN manager for Tier 4 VPN rotation.

        When linked, escalation to Tier 4 will trigger VPN server rotation
        for IP-based rate limit bypass.

        Args:
            mullvad_vpn: The MullvadVPN manager to use.
        """
        self._mullvad_vpn = mullvad_vpn

    def set_vpn_rotation_callback(self, callback: Callable[[str], None]) -> None:
        """Set or replace the on_vpn_rotation_needed callback (US-36-003).

        This allows the pipeline to wire up budget reset logic after
        the EscalationManager is instantiated. Called by HealingOrchestrator
        to ensure budget.reset_on_ip_change() is invoked on Tier 4 escalation.

        Args:
            callback: Callable taking keyword (str) to invoke on VPN rotation.
        """
        self._on_vpn_rotation_needed = callback

    @property
    def keyword_states(self) -> Dict[str, EscalationState]:
        """Read-only access to keyword states (for metrics/debugging)."""
        return dict(self._keyword_states)

    @property
    def strategy(self) -> "EscalationStrategy":
        """Access the escalation strategy (for testing/inspection)."""
        return self._strategy

    def get_escalation_args(self, keyword: str) -> EscalationResult:
        """Get yt-dlp arguments for the current escalation tier of a keyword.

        Tier 1: --impersonate <target> only
        Tier 2: --impersonate <target> + --extractor-args "youtube:player_client=X,Y,Z"
        Tier 3: All of Tier 2 + rotate_cookies=True flag
        Tier 4: All of Tier 3 + rotate_vpn=True flag

        Also records the attempt in the budget (if available) to track total
        download attempts across keywords.

        Args:
            keyword: The download keyword or video ID.

        Returns:
            EscalationResult with args list, tier, cookie rotation, and VPN rotation flags.
        """
        # Track attempt in budget (outside lock - budget has its own thread safety)
        if self._budget is not None:
            self._budget.record_attempt(keyword)

        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            tier = state.current_tier

            # Delegate circuit breaker shortcut decision to strategy
            cb_open = self._circuit_breaker is not None and self._circuit_breaker.is_open
            shortcut_decision = self._strategy.should_shortcut_to_max(tier, cb_open)
            if shortcut_decision.should_escalate:
                logger.info(
                    f"Circuit breaker open: shortcutting keyword={keyword} "
                    f"from {tier.name} to FULL_BYPASS"
                )
                tier = shortcut_decision.target_tier

            # Tier 1: impersonation only
            args = self._impersonation_manager.get_impersonate_args()

            result = EscalationResult(
                args=list(args),
                tier=tier,
                rotate_cookies=False,
                rotate_vpn=False,
            )

            # Tier 2+: add extractor-args
            if tier >= EscalationTier.EXTRACTOR_ARGS:
                extractor_args = self._build_extractor_args(state)
                if extractor_args:
                    result.args.extend(extractor_args)

            # Tier 3+: signal cookie rotation
            if tier >= EscalationTier.FULL_BYPASS:
                result.rotate_cookies = True

            # Tier 4: signal VPN rotation (only if Mullvad is configured)
            if tier >= EscalationTier.VPN_ROTATION and self._mullvad_vpn is not None:
                result.rotate_vpn = True

            logger.debug(
                f"Escalation args: keyword={keyword} tier={tier.name} "
                f"args_count={len(result.args)} rotate_cookies={result.rotate_cookies} "
                f"rotate_vpn={result.rotate_vpn}"
            )

            return result

    def _build_extractor_args(self, state: EscalationState) -> List[str]:
        """Build --extractor-args for Tier 2+ from ExtractorArgsConfig.

        Rotates the player_client list starting position using
        state.extractor_args_index so successive escalations try different
        client orderings.

        Returns:
            ['--extractor-args', 'youtube:player_client=X,Y,Z'] or empty list.
        """
        if self._extractor_config is None:
            return []
        if not getattr(self._extractor_config, 'enabled', True):
            return []

        clients = getattr(self._extractor_config, 'player_clients', [])
        if not clients:
            return []

        # Rotate starting position
        idx = state.extractor_args_index % len(clients)
        rotated = clients[idx:] + clients[:idx]
        client_str = ','.join(rotated)

        return ['--extractor-args', f'youtube:player_client={client_str}']

    def _record_timeline_event(
        self, keyword: str, from_tier: EscalationTier, to_tier: EscalationTier,
        trigger_category: Optional[str] = None,
    ) -> None:
        """Record an escalation event in the per-keyword timeline.

        Args:
            keyword: The keyword that escalated.
            from_tier: Tier before escalation.
            to_tier: Tier after escalation.
            trigger_category: Category from classify_trigger() or None.
        """
        if keyword not in self._escalation_timeline:
            self._escalation_timeline[keyword] = []
        self._escalation_timeline[keyword].append({
            'timestamp': time.time(),
            'from_tier': from_tier.value,
            'to_tier': to_tier.value,
            'trigger_category': trigger_category or 'unknown',
        })

    def record_failure(self, keyword: str, error_output: str = "") -> None:
        """Record a download failure for a keyword.

        Increments the consecutive 403 counter. If the threshold is reached,
        escalates to the next tier. When a budget is available:
        - Records a rotation on tier advance (Tier 2 or Tier 3)
        - If budget is exhausted (can_rotate() is False), skips intermediate
          tiers and jumps directly to max tier (FULL_BYPASS)

        Args:
            keyword: The download keyword or video ID.
            error_output: stderr output from the failed subprocess.
        """
        # Classify trigger category for timeline tracking
        trigger_category = classify_trigger(error_output) if error_output else None

        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            state.consecutive_403s += 1
            self._total_403s += 1

            # Delegate escalation decision to strategy
            budget_exhausted = self._budget is not None and not self._budget.can_rotate()
            decision = self._strategy.should_escalate_on_failure(state, budget_exhausted)

            if decision.should_escalate:
                old_tier = state.current_tier
                n_403s = state.consecutive_403s

                if decision.skip_to_max:
                    # Budget exhausted: skip to max tier
                    logger.warning(
                        f"Budget exhausted for keyword={keyword}: "
                        f"skipping to FULL_BYPASS (was {state.current_tier.name})"
                    )
                    state.current_tier = decision.target_tier
                    state.last_escalation_time = time.time()
                    state.escalation_history.append(
                        (state.last_escalation_time, state.current_tier)
                    )
                    state.consecutive_403s = 0
                    state.extractor_args_index += 1
                else:
                    # Normal escalation
                    state.escalate()
                    # Increment extractor_args_index on Tier 2 escalation
                    if state.current_tier >= EscalationTier.EXTRACTOR_ARGS:
                        state.extractor_args_index += 1

                self._total_escalations += 1
                tier_name = state.current_tier.name
                self._escalations_per_tier[tier_name] = (
                    self._escalations_per_tier.get(tier_name, 0) + 1
                )

                # Budget tracking: record rotation when advancing to Tier 2 or Tier 3
                if self._budget is not None and state.current_tier > old_tier:
                    self._budget.record_rotation(keyword)

                # Record timeline event
                self._record_timeline_event(
                    keyword, old_tier, state.current_tier, trigger_category
                )

                logger.info(
                    f"Escalation: keyword={keyword} tier {old_tier.name}->{state.current_tier.name} "
                    f"after {n_403s} consecutive 403s"
                )

                if state.current_tier == EscalationTier.FULL_BYPASS:
                    logger.warning(
                        f"Tier 3 reached for keyword={keyword}, engaging full bypass with cookies"
                    )
                elif state.current_tier == EscalationTier.VPN_ROTATION:
                    logger.warning(
                        f"Max escalation (Tier 4) reached for keyword={keyword}, "
                        f"engaging VPN rotation"
                    )
                    # Invoke callback for VPN rotation handling
                    if self._on_vpn_rotation_needed is not None:
                        try:
                            self._on_vpn_rotation_needed(keyword)
                        except Exception as e:
                            logger.error(
                                f"VPN rotation callback failed for keyword={keyword}: {e}"
                            )

    def record_success(self, keyword: str) -> None:
        """Record a successful download for a keyword.

        Resets the 403 counter but keeps the current tier (sticky escalation).

        Args:
            keyword: The download keyword or video ID.
        """
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            state.record_success()
            self._total_successes += 1

    def record_slow_speed(self, keyword: str, speed_mbps: float = 0.0) -> None:
        """Record a slow download speed signal for preemptive escalation.

        When called 3+ times for the same keyword, preemptively escalates
        one tier without waiting for a 403 error. Speed-triggered escalations
        do NOT count toward budget rotations (they are preventive, not reactive).

        Args:
            keyword: The download keyword or video ID.
            speed_mbps: The detected download speed in MB/s (for logging).
        """
        lock = self._get_lock(keyword)
        with lock:
            # Increment slow speed count for this keyword
            count = self._slow_speed_counts.get(keyword, 0) + 1
            self._slow_speed_counts[keyword] = count

            state = self._get_state(keyword)
            # Delegate decision to strategy
            decision = self._strategy.should_escalate_on_slow_speed(state, count)

            if decision.should_escalate:
                old_tier = state.current_tier
                state.escalate()
                self._speed_escalations += 1
                self._total_escalations += 1
                tier_name = state.current_tier.name
                self._escalations_per_tier[tier_name] = (
                    self._escalations_per_tier.get(tier_name, 0) + 1
                )

                logger.info(
                    f"Preemptive escalation for {keyword}: sustained low speed "
                    f"({speed_mbps:.3f} MB/s) - "
                    f"{old_tier.name} -> {state.current_tier.name} "
                    f"(after {count} slow speed signals)"
                )

                # Reset slow speed count after escalation
                self._slow_speed_counts[keyword] = 0
                # NOTE: No budget.record_rotation() here - speed signals
                # are preventive, not reactive, so they don't consume budget
            elif self._strategy.is_at_max_tier(state.current_tier):
                logger.debug(
                    f"Slow speed for {keyword} ({speed_mbps:.3f} MB/s) "
                    f"but already at max tier"
                )
                # Reset counter since we can't escalate further
                self._slow_speed_counts[keyword] = 0
            elif "cooldown" in decision.reason.lower():
                logger.debug(
                    f"Speed escalation suppressed for {keyword}: cooldown active"
                )
                # Reset counter so signals accumulate again after cooldown
                self._slow_speed_counts[keyword] = 0

    def _should_escalate(self, state: EscalationState) -> bool:
        """Check if escalation should proceed, considering cooldown.

        Returns False if the keyword was escalated within the cooldown
        period, even if the 403 threshold has been reached again.

        Note: This method delegates to the strategy but is kept for
        backward compatibility with any external code that may call it.

        Args:
            state: The per-keyword escalation state.

        Returns:
            True if escalation should proceed, False if in cooldown.
        """
        decision = self._strategy.should_escalate_on_failure(state, budget_exhausted=False)
        return decision.should_escalate

    def _is_past_cooldown(self, state: EscalationState) -> bool:
        """Check if enough time has passed since the last escalation.

        Used by record_slow_speed() to prevent rapid speed-triggered
        escalations within the cooldown window.

        Note: This method delegates to the strategy but is kept for
        backward compatibility.

        Args:
            state: The per-keyword escalation state.

        Returns:
            True if past cooldown (or never escalated), False if in cooldown.
        """
        return self._strategy.is_past_cooldown(state)

    def get_cooldown_remaining(self, keyword: str) -> float:
        """Get remaining cooldown seconds for a keyword.

        Args:
            keyword: The download keyword or video ID.

        Returns:
            Seconds remaining in cooldown, or 0.0 if not in cooldown.
        """
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            return self._strategy.get_cooldown_remaining(state)

    def reset_keyword(self, keyword: str) -> None:
        """Clear all escalation state for a keyword.

        Useful for healer or manual recovery scenarios.

        Args:
            keyword: The download keyword or video ID to reset.
        """
        lock = self._get_lock(keyword)
        with lock:
            if keyword in self._keyword_states:
                del self._keyword_states[keyword]
                logger.debug(f"Escalation state reset for keyword={keyword}")

    def reset_all(self) -> None:
        """Clear all keyword escalation states.

        Useful for session restart or full recovery.
        """
        with self._global_lock:
            self._keyword_states.clear()
            self._keyword_locks.clear()
            self._total_403s = 0
            self._total_successes = 0
            self._total_escalations = 0
            self._escalations_per_tier.clear()
            self._slow_speed_counts.clear()
            self._speed_escalations = 0
            self._escalation_timeline.clear()
            self._tier_outcomes.clear()
            logger.debug("All escalation states reset")

    def get_active_keyword_count(self) -> int:
        """Get the number of keywords with tracked escalation state.

        Used by circuit breaker to determine what percentage of keywords
        are at Tier 3, which informs whether pause duration should be extended.

        Returns:
            Number of keywords currently tracked.
        """
        with self._global_lock:
            return len(self._keyword_states)

    def get_keywords_at_tier(self, tier: EscalationTier) -> List[str]:
        """Get list of keywords currently at a specific escalation tier.

        Used by circuit breaker to check how many keywords are at Tier 3
        and decide whether to extend pause duration.

        Args:
            tier: The escalation tier to query.

        Returns:
            List of keyword strings at the given tier.
        """
        with self._global_lock:
            return [
                kw for kw, state in self._keyword_states.items()
                if state.current_tier == tier
            ]

    def to_dict(self) -> Dict:
        """Serialize all keyword escalation states for checkpoint persistence.

        Returns:
            Dict with keyword states, global counters, timeline, outcomes, and a timestamp.
            Format: {
                'keyword_states': {keyword: {tier, consecutive_403s,
                    extractor_args_index, last_escalation_time}},
                'total_403s': int,
                'total_successes': int,
                'total_escalations': int,
                'escalations_per_tier': {tier_name: count},
                'speed_escalations': int,
                'timeline': {keyword: [{timestamp, from_tier, to_tier, trigger_category}]},
                'tier_outcomes': {trigger_category: {tier_value: {successes, attempts}}},
                'saved_at': float (epoch timestamp)
            }
        """
        with self._global_lock:
            keyword_states = {}
            for keyword, state in self._keyword_states.items():
                keyword_states[keyword] = {
                    'tier': state.current_tier.value,
                    'consecutive_403s': state.consecutive_403s,
                    'extractor_args_index': state.extractor_args_index,
                    'last_escalation_time': state.last_escalation_time,
                }
            # Deep-copy timeline events (list of dicts per keyword)
            timeline = {
                kw: list(events)
                for kw, events in self._escalation_timeline.items()
            }
            # Deep-copy tier outcomes (nested dicts)
            tier_outcomes = {
                cat: {
                    tier_val: dict(counts)
                    for tier_val, counts in tiers.items()
                }
                for cat, tiers in self._tier_outcomes.items()
            }
            return {
                'keyword_states': keyword_states,
                'total_403s': self._total_403s,
                'total_successes': self._total_successes,
                'total_escalations': self._total_escalations,
                'escalations_per_tier': dict(self._escalations_per_tier),
                'speed_escalations': self._speed_escalations,
                'timeline': timeline,
                'tier_outcomes': tier_outcomes,
                'saved_at': time.time(),
            }

    @classmethod
    def from_dict(
        cls,
        data: Dict,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
        stale_threshold: float = 3600.0,
        strategy: Optional["EscalationStrategy"] = None,
    ) -> "EscalationManager":
        """Restore an EscalationManager from checkpoint data.

        Handles stale state: if the checkpoint data is older than
        ``stale_threshold`` seconds, all keywords are de-escalated by one tier
        (YouTube may have relaxed blocking since the last session).

        Args:
            data: Dict previously returned by ``to_dict()``.
            impersonation_manager: Provides Tier 1 --impersonate args.
            extractor_args_config: Configuration for Tier 2 player_client rotation.
            budget: Optional RateLimitBudget for budget-aware escalation.
            stale_threshold: Seconds after which saved data is considered stale
                and keywords are de-escalated by one tier. Default: 3600 (1 hour).
            strategy: Optional EscalationStrategy for decision logic.
                If not provided, a default strategy is created from config.

        Returns:
            A new EscalationManager with restored keyword states.
        """
        manager = cls(
            impersonation_manager=impersonation_manager,
            extractor_args_config=extractor_args_config,
            budget=budget,
            strategy=strategy,
        )

        if not data or not isinstance(data, dict):
            logger.warning("Empty or invalid escalation checkpoint data, starting fresh")
            return manager

        # Check staleness
        saved_at = data.get('saved_at', 0.0)
        age = time.time() - saved_at
        is_stale = age > stale_threshold

        if is_stale:
            logger.info(
                f"Escalation checkpoint is stale ({age:.0f}s > {stale_threshold:.0f}s threshold), "
                f"de-escalating all keywords by one tier"
            )

        # Restore keyword states
        keyword_states = data.get('keyword_states', {})
        for keyword, state_data in keyword_states.items():
            tier_value = state_data.get('tier', EscalationTier.IMPERSONATE_ONLY.value)
            # Clamp to valid tier range
            tier_value = max(
                EscalationTier.IMPERSONATE_ONLY.value,
                min(tier_value, EscalationTier.FULL_BYPASS.value),
            )

            if is_stale and tier_value > EscalationTier.IMPERSONATE_ONLY.value:
                tier_value -= 1

            state = EscalationState(
                current_tier=EscalationTier(tier_value),
                consecutive_403s=state_data.get('consecutive_403s', 0),
                last_escalation_time=state_data.get('last_escalation_time'),
                extractor_args_index=state_data.get('extractor_args_index', 0),
            )
            manager._keyword_states[keyword] = state

        # Restore global counters
        manager._total_403s = data.get('total_403s', 0)
        manager._total_successes = data.get('total_successes', 0)
        manager._total_escalations = data.get('total_escalations', 0)
        manager._escalations_per_tier = dict(data.get('escalations_per_tier', {}))
        manager._speed_escalations = data.get('speed_escalations', 0)

        # Restore escalation timeline (per-keyword event history)
        saved_timeline = data.get('timeline', {})
        if isinstance(saved_timeline, dict):
            for kw, events in saved_timeline.items():
                if isinstance(events, list):
                    manager._escalation_timeline[kw] = list(events)

        # Restore tier outcomes (per-category, per-tier success/attempt counts)
        saved_outcomes = data.get('tier_outcomes', {})
        if isinstance(saved_outcomes, dict):
            for category, tiers in saved_outcomes.items():
                if isinstance(tiers, dict):
                    manager._tier_outcomes[category] = {}
                    for tier_val, counts in tiers.items():
                        if isinstance(counts, dict):
                            # tier_val may be string from JSON; convert to int
                            try:
                                tier_key = int(tier_val)
                            except (ValueError, TypeError):
                                continue
                            manager._tier_outcomes[category][tier_key] = {
                                'successes': counts.get('successes', 0),
                                'attempts': counts.get('attempts', 0),
                            }

        restored_count = len(keyword_states)
        logger.info(
            f"Restored escalation state for {restored_count} keywords"
            f"{' (de-escalated due to stale data)' if is_stale else ''}"
        )

        return manager

    @classmethod
    def create_with_mullvad(
        cls,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        mullvad_config: Optional["MullvadConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
        strategy: Optional["EscalationStrategy"] = None,
    ) -> "EscalationManager":
        """Create an EscalationManager with MullvadVPN pre-wired if enabled.

        Factory method that instantiates both EscalationManager and MullvadVPN
        (if mullvad_config.enabled is True), wiring them together automatically.
        This simplifies setup compared to manually creating both and calling
        set_mullvad_vpn().

        Args:
            impersonation_manager: Provides Tier 1 --impersonate args.
            extractor_args_config: Configuration for Tier 2 player_client rotation.
            mullvad_config: MullvadConfig for Tier 4 VPN rotation. If None or
                mullvad_config.enabled is False, MullvadVPN is not created.
            budget: Optional RateLimitBudget for budget-aware escalation.
            strategy: Optional EscalationStrategy for decision logic.

        Returns:
            A new EscalationManager with MullvadVPN wired if enabled.
        """
        manager = cls(
            impersonation_manager=impersonation_manager,
            extractor_args_config=extractor_args_config,
            budget=budget,
            strategy=strategy,
        )

        # Wire MullvadVPN if config is enabled
        if mullvad_config and getattr(mullvad_config, 'enabled', False):
            # Import here to avoid circular import at module level
            from .mullvad_vpn import MullvadVPN
            mullvad_vpn = MullvadVPN(mullvad_config)
            manager.set_mullvad_vpn(mullvad_vpn)
            logger.debug(
                "EscalationManager created with MullvadVPN for Tier 4 bypass"
            )
        else:
            logger.debug(
                "EscalationManager created without MullvadVPN (disabled or no config)"
            )

        return manager

    def get_metrics(self) -> Dict:
        """Get escalation metrics summary.

        Returns:
            Dict with:
                total_escalations: Total number of tier escalations
                escalations_per_tier: Dict mapping tier name to escalation count
                keywords_at_each_tier: Dict mapping tier name to list of keywords
                total_403s: Total 403/bot-detection errors recorded
                total_successes: Total successful downloads recorded
                average_tier: Weighted average tier across all tracked keywords (1.0-3.0)
                speed_escalations: Total escalations triggered by slow speed signals
        """
        with self._global_lock:
            keywords_at_each_tier: Dict[str, List[str]] = {}
            tier_sum = 0.0
            keyword_count = 0

            for keyword, state in self._keyword_states.items():
                tier_name = state.current_tier.name
                if tier_name not in keywords_at_each_tier:
                    keywords_at_each_tier[tier_name] = []
                keywords_at_each_tier[tier_name].append(keyword)
                tier_sum += float(state.current_tier.value)
                keyword_count += 1

            average_tier = round(tier_sum / keyword_count, 2) if keyword_count > 0 else 1.0

            return {
                'total_escalations': self._total_escalations,
                'escalations_per_tier': dict(self._escalations_per_tier),
                'keywords_at_each_tier': keywords_at_each_tier,
                'total_403s': self._total_403s,
                'total_successes': self._total_successes,
                'average_tier': average_tier,
                'speed_escalations': self._speed_escalations,
            }

    def get_keyword_escalation_timeline(self) -> Dict[str, List[Dict]]:
        """Get per-keyword escalation event timeline.

        Returns a dict keyed by keyword, where each value is a list of
        escalation event dicts with keys: timestamp, from_tier, to_tier,
        trigger_category. Limited to the last 20 events per keyword.

        Returns:
            Dict mapping keyword to list of event dicts.
        """
        with self._global_lock:
            result: Dict[str, List[Dict]] = {}
            for keyword, events in self._escalation_timeline.items():
                # Limit to last 20 events per keyword
                result[keyword] = list(events[-20:])
            return result

    def get_hot_keywords(self, window_seconds: float = 1800.0) -> List[Dict]:
        """Get keywords with frequent escalations in a recent time window.

        A keyword is "hot" if it has more than 3 escalation events within
        the specified window (default 30 minutes). Results are sorted by
        escalation count descending.

        Args:
            window_seconds: Time window in seconds (default: 1800 = 30 min).

        Returns:
            List of dicts with 'keyword' and 'escalation_count', sorted by
            count descending. Only includes keywords with >3 escalations.
        """
        cutoff = time.time() - window_seconds
        with self._global_lock:
            hot: List[Dict] = []
            for keyword, events in self._escalation_timeline.items():
                recent_count = sum(
                    1 for e in events if e['timestamp'] > cutoff
                )
                if recent_count > 3:
                    hot.append({
                        'keyword': keyword,
                        'escalation_count': recent_count,
                    })
            hot.sort(key=lambda x: x['escalation_count'], reverse=True)
            return hot

    def record_outcome(
        self, trigger_category: str, tier: EscalationTier, success: bool
    ) -> None:
        """Record a download outcome for tier effectiveness tracking.

        Each call records whether a download attempt at a specific escalation
        tier succeeded or failed for a given trigger category. This data is
        used by ``get_tier_effectiveness()`` to compute success rates per
        trigger category per tier.

        Args:
            trigger_category: Category from classify_trigger() (e.g., '403', '429').
            tier: The escalation tier used for this attempt.
            success: True if the download succeeded, False if it failed.
        """
        with self._global_lock:
            if trigger_category not in self._tier_outcomes:
                self._tier_outcomes[trigger_category] = {}
            tier_val = tier.value
            if tier_val not in self._tier_outcomes[trigger_category]:
                self._tier_outcomes[trigger_category][tier_val] = {
                    'successes': 0, 'attempts': 0,
                }
            self._tier_outcomes[trigger_category][tier_val]['attempts'] += 1
            if success:
                self._tier_outcomes[trigger_category][tier_val]['successes'] += 1

    def get_tier_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Get success rate per trigger category per escalation tier.

        Returns a dict keyed by trigger category, where each value maps
        tier names ('tier_1', 'tier_2', 'tier_3') to their success rate
        (0.0 to 1.0). Only tiers with recorded attempts are included.

        Returns:
            Dict mapping trigger_category to {tier_name: success_rate}.
            Empty dict if no outcomes have been recorded.
        """
        with self._global_lock:
            if not self._tier_outcomes:
                return {}

            result: Dict[str, Dict[str, float]] = {}
            for category, tiers in self._tier_outcomes.items():
                tier_rates: Dict[str, float] = {}
                for tier_val, counts in tiers.items():
                    attempts = counts['attempts']
                    if attempts > 0:
                        rate = counts['successes'] / attempts
                        tier_rates[f'tier_{tier_val}'] = round(rate, 4)
                if tier_rates:
                    result[category] = tier_rates
            return result

    def get_tier_recommendations(self) -> List[str]:
        """Generate recommendations based on tier effectiveness data.

        Analyzes per-category, per-tier success rates and returns
        actionable recommendations:
        - If a category has >80% Tier 1 success: 'skip escalation for {category}'
        - If a category has <20% Tier 2 but >60% Tier 3: 'skip Tier 2 for {category}'

        Returns:
            List of recommendation strings. Empty list if no data or
            no recommendations apply.
        """
        recommendations: List[str] = []
        effectiveness = self.get_tier_effectiveness()

        for category, tier_rates in effectiveness.items():
            tier_1_rate = tier_rates.get('tier_1', None)
            tier_2_rate = tier_rates.get('tier_2', None)
            tier_3_rate = tier_rates.get('tier_3', None)

            # Recommend skipping escalation if Tier 1 is highly effective
            if tier_1_rate is not None and tier_1_rate > 0.80:
                recommendations.append(
                    f"skip escalation for {category}"
                )

            # Recommend skipping Tier 2 if it's ineffective but Tier 3 works
            if (
                tier_2_rate is not None and tier_2_rate < 0.20
                and tier_3_rate is not None and tier_3_rate > 0.60
            ):
                recommendations.append(
                    f"skip Tier 2 for {category}"
                )

        return recommendations
