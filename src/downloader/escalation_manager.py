"""
3-tier escalation manager for yt-dlp bypass orchestration.

Manages per-keyword escalation state across three tiers:
  - Tier 1 (IMPERSONATE_ONLY): --impersonate only (delegated to ImpersonationManager)
  - Tier 2 (EXTRACTOR_ARGS): --impersonate + --extractor-args player_client
  - Tier 3 (FULL_BYPASS): Both + cookie rotation flag

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
from typing import Dict, List, Optional, Tuple

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
    from .impersonation import ImpersonationManager
except ImportError:  # pragma: no cover
    ImpersonationManager = None  # type: ignore[misc,assignment]

try:
    from .rate_limit_budget import RateLimitBudget
except ImportError:  # pragma: no cover
    RateLimitBudget = None  # type: ignore[misc,assignment]

try:
    from .circuit_breaker import CircuitBreaker
except ImportError:  # pragma: no cover
    CircuitBreaker = None  # type: ignore[misc,assignment]


@dataclass
class EscalationResult:
    """Result from get_escalation_args() with args and metadata.

    Attributes:
        args: List of yt-dlp CLI arguments (--impersonate, --extractor-args, etc.)
        tier: The escalation tier used to generate these args.
        rotate_cookies: If True, caller should trigger cookie rotation (Tier 3).
    """
    args: List[str] = field(default_factory=list)
    tier: EscalationTier = EscalationTier.IMPERSONATE_ONLY
    rotate_cookies: bool = False


class EscalationManager:
    """Orchestrates 3-tier yt-dlp bypass escalation per keyword.

    Per-keyword tracking ensures that one keyword hitting 403 errors does not
    affect the escalation state of other keywords.

    Args:
        impersonation_manager: Provides Tier 1 --impersonate args.
        extractor_args_config: Configuration for Tier 2 player_client rotation.
            If None, a default config is used with escalation disabled.
        budget: Optional RateLimitBudget for budget-aware escalation.
            When provided, escalation decisions consult the budget:
            - record_failure() calls budget.record_rotation() on tier advances
            - If budget is exhausted, skip intermediate tiers to max tier
            - get_escalation_args() calls budget.record_attempt()
    """

    def __init__(
        self,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
        budget: Optional["RateLimitBudget"] = None,
    ):
        self._impersonation_manager = impersonation_manager
        self._extractor_config = extractor_args_config
        self._budget = budget
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        self._keyword_states: Dict[str, EscalationState] = {}
        self._keyword_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._total_403s: int = 0
        self._total_successes: int = 0
        self._total_escalations: int = 0
        self._escalations_per_tier: Dict[str, int] = {}  # tier_name -> count
        self._slow_speed_counts: Dict[str, int] = {}  # keyword -> consecutive slow count
        self._speed_escalations: int = 0  # Total speed-triggered escalations

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

    @property
    def keyword_states(self) -> Dict[str, EscalationState]:
        """Read-only access to keyword states (for metrics/debugging)."""
        return dict(self._keyword_states)

    def get_escalation_args(self, keyword: str) -> EscalationResult:
        """Get yt-dlp arguments for the current escalation tier of a keyword.

        Tier 1: --impersonate <target> only
        Tier 2: --impersonate <target> + --extractor-args "youtube:player_client=X,Y,Z"
        Tier 3: All of Tier 2 + rotate_cookies=True flag

        Also records the attempt in the budget (if available) to track total
        download attempts across keywords.

        Args:
            keyword: The download keyword or video ID.

        Returns:
            EscalationResult with args list, tier, and cookie rotation flag.
        """
        # Track attempt in budget (outside lock - budget has its own thread safety)
        if self._budget is not None:
            self._budget.record_attempt(keyword)

        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            tier = state.current_tier

            # Circuit breaker shortcut: when circuit breaker is open (paused),
            # return Tier 3 args immediately to skip lower tiers
            if (
                self._circuit_breaker is not None
                and self._circuit_breaker.is_open
                and tier < EscalationTier.FULL_BYPASS
            ):
                logger.info(
                    f"Circuit breaker open: shortcutting keyword={keyword} "
                    f"from {tier.name} to FULL_BYPASS"
                )
                tier = EscalationTier.FULL_BYPASS

            # Tier 1: impersonation only
            args = self._impersonation_manager.get_impersonate_args()

            result = EscalationResult(
                args=list(args),
                tier=tier,
                rotate_cookies=False,
            )

            # Tier 2+: add extractor-args
            if tier >= EscalationTier.EXTRACTOR_ARGS:
                extractor_args = self._build_extractor_args(state)
                if extractor_args:
                    result.args.extend(extractor_args)

            # Tier 3: signal cookie rotation
            if tier >= EscalationTier.FULL_BYPASS:
                result.rotate_cookies = True

            logger.debug(
                f"Escalation args: keyword={keyword} tier={tier.name} "
                f"args_count={len(result.args)} rotate_cookies={result.rotate_cookies}"
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
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            state.consecutive_403s += 1
            self._total_403s += 1

            if self._should_escalate(state):
                old_tier = state.current_tier
                n_403s = state.consecutive_403s

                # Budget-aware escalation: if budget exhausted, skip to max tier
                if self._budget is not None and not self._budget.can_rotate():
                    if state.current_tier < EscalationTier.FULL_BYPASS:
                        logger.warning(
                            f"Budget exhausted for keyword={keyword}: "
                            f"skipping to FULL_BYPASS (was {state.current_tier.name})"
                        )
                        state.current_tier = EscalationTier.FULL_BYPASS
                        state.last_escalation_time = time.time()
                        state.escalation_history.append(
                            (state.last_escalation_time, state.current_tier)
                        )
                        state.consecutive_403s = 0
                        state.extractor_args_index += 1
                        self._total_escalations += 1
                        tier_name = state.current_tier.name
                        self._escalations_per_tier[tier_name] = (
                            self._escalations_per_tier.get(tier_name, 0) + 1
                        )
                        return

                state.escalate()
                self._total_escalations += 1
                tier_name = state.current_tier.name
                self._escalations_per_tier[tier_name] = (
                    self._escalations_per_tier.get(tier_name, 0) + 1
                )
                # Increment extractor_args_index on Tier 2 escalation
                if state.current_tier >= EscalationTier.EXTRACTOR_ARGS:
                    state.extractor_args_index += 1

                # Budget tracking: record rotation when advancing to Tier 2 or Tier 3
                if self._budget is not None and state.current_tier > old_tier:
                    self._budget.record_rotation(keyword)

                logger.info(
                    f"Escalation: keyword={keyword} tier {old_tier.name}->{state.current_tier.name} "
                    f"after {n_403s} consecutive 403s"
                )

                if state.current_tier == EscalationTier.FULL_BYPASS:
                    logger.warning(
                        f"Max escalation reached for keyword={keyword}, engaging full bypass"
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

            if count >= 3:
                state = self._get_state(keyword)
                if state.current_tier < EscalationTier.FULL_BYPASS:
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
                else:
                    logger.debug(
                        f"Slow speed for {keyword} ({speed_mbps:.3f} MB/s) "
                        f"but already at max tier"
                    )
                    # Reset counter since we can't escalate further
                    self._slow_speed_counts[keyword] = 0

    def _should_escalate(self, state: EscalationState) -> bool:
        """Check if escalation should proceed, considering cooldown.

        Returns False if the keyword was escalated within the cooldown
        period, even if the 403 threshold has been reached again.

        Args:
            state: The per-keyword escalation state.

        Returns:
            True if escalation should proceed, False if in cooldown.
        """
        cooldown = 300.0
        if self._extractor_config is not None:
            cooldown = getattr(self._extractor_config, 'cooldown_seconds', 300.0)

        threshold = 2
        if self._extractor_config is not None:
            threshold = getattr(self._extractor_config, 'escalation_threshold', 2)

        if not state.should_escalate(threshold):
            return False

        # Check cooldown: if recently escalated, suppress
        if state.last_escalation_time is not None:
            elapsed = time.time() - state.last_escalation_time
            if elapsed < cooldown:
                logger.debug(
                    f"Escalation suppressed: cooldown active "
                    f"({elapsed:.0f}s / {cooldown:.0f}s elapsed)"
                )
                return False

        return True

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
            if state.last_escalation_time is None:
                return 0.0

            cooldown = 300.0
            if self._extractor_config is not None:
                cooldown = getattr(self._extractor_config, 'cooldown_seconds', 300.0)

            elapsed = time.time() - state.last_escalation_time
            remaining = cooldown - elapsed
            return max(0.0, remaining)

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
            Dict with keyword states, global counters, and a timestamp.
            Format: {
                'keyword_states': {keyword: {tier, consecutive_403s, total_403s,
                    extractor_args_index, last_escalation_time}},
                'total_403s': int,
                'total_successes': int,
                'total_escalations': int,
                'escalations_per_tier': {tier_name: count},
                'speed_escalations': int,
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
            return {
                'keyword_states': keyword_states,
                'total_403s': self._total_403s,
                'total_successes': self._total_successes,
                'total_escalations': self._total_escalations,
                'escalations_per_tier': dict(self._escalations_per_tier),
                'speed_escalations': self._speed_escalations,
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

        Returns:
            A new EscalationManager with restored keyword states.
        """
        manager = cls(
            impersonation_manager=impersonation_manager,
            extractor_args_config=extractor_args_config,
            budget=budget,
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

        restored_count = len(keyword_states)
        logger.info(
            f"Restored escalation state for {restored_count} keywords"
            f"{' (de-escalated due to stale data)' if is_stale else ''}"
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
