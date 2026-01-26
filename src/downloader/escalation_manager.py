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
    r'HTTP Error 403'
    r'|Sign in to confirm'
    r'|bot'
    r'|captcha'
    r'|blocked'
    r'|verify you are human',
    re.IGNORECASE,
)


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
    """

    def __init__(
        self,
        impersonation_manager: "ImpersonationManager",
        extractor_args_config: Optional["ExtractorArgsConfig"] = None,
    ):
        self._impersonation_manager = impersonation_manager
        self._extractor_config = extractor_args_config
        self._keyword_states: Dict[str, EscalationState] = {}
        self._keyword_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

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

    @property
    def keyword_states(self) -> Dict[str, EscalationState]:
        """Read-only access to keyword states (for metrics/debugging)."""
        return dict(self._keyword_states)

    def get_escalation_args(self, keyword: str) -> EscalationResult:
        """Get yt-dlp arguments for the current escalation tier of a keyword.

        Tier 1: --impersonate <target> only
        Tier 2: --impersonate <target> + --extractor-args "youtube:player_client=X,Y,Z"
        Tier 3: All of Tier 2 + rotate_cookies=True flag

        Args:
            keyword: The download keyword or video ID.

        Returns:
            EscalationResult with args list, tier, and cookie rotation flag.
        """
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            tier = state.current_tier

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
        escalates to the next tier.

        Args:
            keyword: The download keyword or video ID.
            error_output: stderr output from the failed subprocess.
        """
        lock = self._get_lock(keyword)
        with lock:
            state = self._get_state(keyword)
            state.consecutive_403s += 1

            if self._should_escalate(state):
                old_tier = state.current_tier
                n_403s = state.consecutive_403s
                state.escalate()
                # Increment extractor_args_index on Tier 2 escalation
                if state.current_tier >= EscalationTier.EXTRACTOR_ARGS:
                    state.extractor_args_index += 1

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
            logger.debug("All escalation states reset")
