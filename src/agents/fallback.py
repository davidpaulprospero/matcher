"""Fallback chain and pattern-based routing for healing system."""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

# Lazy import requests to avoid import-time failure
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False

if TYPE_CHECKING:
    from .healing_logger import HealingLogger

logger = logging.getLogger(__name__)


# Pattern-based routing when watcher is unavailable
# IMPORTANT: Use word boundaries (\b) to prevent false positives
# Examples of past false positives we're preventing:
# - "Singapore" matching "gap" -> now requires \bgap\b
# - "video_401.mp4" matching "401" -> now requires \b401\b or HTTP context
# - "author" matching "auth" -> now requires \bauth(entication|orization)?\b
PATTERN_ROUTING: Dict[str, Tuple[str, str]] = {
    # API errors - HTTP status codes need context or word boundaries
    r"\b429\b|rate[_\s-]?limit|too[_\s-]?many[_\s-]?requests|quota[_\s-]?(exceeded|limit)": ("api", "api-healer"),
    r"HTTP[_\s]*(401|403)\b|\bauth(entication|orization)?[_\s-]?(error|fail|invalid)|\bunauthorized\b|\bforbidden\b": ("api", "api-healer"),
    r"\b(request|connection|read|socket)[_\s-]?time[d]?[_\s-]?out\b|deadline[_\s-]?exceeded": ("api", "api-healer"),
    r"\bconnection[_\s-]?refused\b|ECONNREFUSED|\bconnect[_\s-]?error\b": ("api", "api-healer"),

    # Disk errors - specific error patterns
    r"\bdisk[_\s-]?full\b|\bno[_\s-]?space[_\s-]?left\b|ENOSPC|\berrno[_\s:]*28\b": ("disk", "disk-healer"),
    r"\bpermission[_\s-]?denied\b|EACCES|\berrno[_\s:]*13\b": ("disk", "disk-healer"),
    r"\bread[_\s-]?only[_\s-]?file[_\s-]?system\b|EROFS": ("disk", "disk-healer"),

    # Path errors - Windows-specific patterns
    r"\bpath[_\s-]?too[_\s-]?long\b|260[_\s-]?char|filename[_\s-]?too[_\s-]?long\b|MAX_PATH": ("path", "path-healer"),
    r"\bUnicodeDecodeError\b|\bUnicodeEncodeError\b|\bcodec[_\s]can't[_\s](decode|encode)\b": ("path", "path-healer"),
    r"\binvalid[_\s-]?(file)?path\b|\billegal[_\s-]?char(acter)?\b": ("path", "path-healer"),

    # Checkpoint errors - JSON parsing specific
    r"\bcheckpoint[_\s-]?(corrupt|invalid|failed|error)\b": ("checkpoint", "checkpoint-healer"),
    r"\bJSONDecodeError\b|json\.(decoder\.)?JSONDecodeError|Expecting[_\s]value": ("checkpoint", "checkpoint-healer"),
    r"\b(checkpoint|json)[_\s-]?(file)?[_\s-]?(corrupt|malformed)\b": ("checkpoint", "checkpoint-healer"),

    # Caption errors - format discovery and subtitle issues
    r"\b(caption|subtitle)s?[_\s-]?(unavailable|not[_\s-]?found|error|failed)\b": ("caption", "caption-healer"),
    r"\bformat[_\s-]?(unavailable|not[_\s-]?available)\b|\bjson3\b|\bsrv[123]\b": ("caption", "caption-healer"),
    r"\bno[_\s-]?(caption|subtitle)s?\b|\bthere[_\s-]?are[_\s-]?no[_\s-]?subtitles\b": ("caption", "caption-healer"),
    r"\b(write[_\s-]?auto[_\s-]?sub|auto[_\s-]?generated[_\s-]?caption)\b": ("caption", "caption-healer"),

    # Download errors - YouTube/video specific
    r"\byoutube\b|\byt-?dlp\b.*\b(error|fail)": ("download", "download-healer"),
    r"\bvideo[_\s-]?(unavailable|not[_\s-]?found|removed|deleted|private)\b": ("download", "download-healer"),
    r"\bage[_\s-]?restrict|\bsign[_\s-]?in[_\s-]?required\b": ("download", "download-healer"),

    # OTIO errors - timeline specific, NOT generic words
    r"\bopentimelineio\b|\botio\b.*\b(error|exception|invalid)\b": ("otio", "otio-healer"),
    r"\b(timeline|track|clip)[_\s-]?(generation|creation)?[_\s-]?(error|failed|invalid)\b": ("otio", "otio-healer"),
    r"\binvalid[_\s-]?time[_\s-]?range\b|\bnegative[_\s-]?duration\b": ("otio", "otio-healer"),
    r"\b(source|available)[_\s-]?range\b.*\b(error|invalid|mismatch)\b": ("otio", "otio-healer"),
    r"\bmedia[_\s-]?reference[_\s-]?(missing|invalid|error)\b": ("otio", "otio-healer"),
}


@dataclass
class PatternClassification:
    """Result from pattern-based classification."""
    category: str
    suggested_healer: str
    severity: str = "recoverable"
    confidence: float = 0.6  # Lower confidence than watcher
    needs_llm_healer: bool = False
    reasoning: str = "Pattern-based classification"


def pattern_route(error: str) -> PatternClassification:
    """Route error to healer based on regex patterns.

    Args:
        error: Error message string

    Returns:
        PatternClassification with category and suggested healer
    """
    error_lower = error.lower()

    for pattern, (category, healer) in PATTERN_ROUTING.items():
        if re.search(pattern, error_lower, re.IGNORECASE):
            return PatternClassification(
                category=category,
                suggested_healer=healer,
                severity="recoverable",
                confidence=0.7,
                needs_llm_healer=False,
                reasoning=f"Matched pattern: {pattern}"
            )

    # No pattern matched - escalate to LLM healer
    return PatternClassification(
        category="unknown",
        suggested_healer="",
        severity="recoverable",
        confidence=0.3,
        needs_llm_healer=True,
        reasoning="No pattern matched, escalating to LLM healer"
    )


class FallbackChain:
    """Manages fallback logic for healing components.

    Provides a multi-level fallback system:
    - Level 0: WatcherAgent (Ollama local)
    - Level 1: Pattern-based routing
    - Level 2: Standard healers
    - Level 3: LLM Healer (Claude)
    - Level 4: User escalation
    - Level 5: Graceful failure

    Thread-safe for concurrent access. Uses sentinel values to prevent
    thundering herd problem when multiple threads check availability.
    """

    # Sentinel value to indicate check in progress
    _CHECKING = "checking"

    # Maximum time to wait for a check in progress (seconds)
    _CHECK_WAIT_TIMEOUT = 10.0

    def __init__(self, config: 'HealingConfig', healing_logger: Optional['HealingLogger'] = None):
        """Initialize fallback chain.

        Args:
            config: HealingConfig with watcher and llm_healer sections
            healing_logger: Optional logger for fallback events
        """
        self.config = config
        self.healing_logger = healing_logger
        self._lock = threading.Lock()
        self._check_condition = threading.Condition(self._lock)

        self.fallback_state: Dict[str, Any] = {
            "watcher_available": None,  # None=unknown, True/False=known, "checking"=in progress
            "llm_healer_available": None,
            "watcher_failures": 0,
            "llm_healer_failures": 0,
            "last_watcher_check": 0.0,
            "last_llm_healer_check": 0.0,
            "watcher_check_started": 0.0,  # Track when check started for timeout
            "llm_healer_check_started": 0.0,
        }

    def check_watcher_available(self) -> bool:
        """Check if Ollama is running and model available.

        Thread-safe with thundering herd prevention. Only one thread
        performs the actual network check while others wait.

        Returns:
            True if watcher can be used, False otherwise
        """
        if not REQUESTS_AVAILABLE:
            return False

        with self._check_condition:
            # Check if we should re-check after previous failure
            if self.fallback_state["watcher_available"] is False:
                last_check = self.fallback_state.get("last_watcher_check", 0.0)
                recheck_interval = getattr(
                    getattr(self.config, 'watcher', None),
                    'recheck_interval_seconds', 300.0
                )
                if time.time() - last_check > recheck_interval:
                    logger.info("[fallback] Re-checking watcher availability...")
                    self.fallback_state["watcher_available"] = None
                    self.fallback_state["watcher_failures"] = 0

            # Return cached result if definitively known
            state = self.fallback_state["watcher_available"]
            if state is True or state is False:
                return state

            # Another thread is checking - wait for result
            if state == self._CHECKING:
                check_started = self.fallback_state.get("watcher_check_started", 0.0)
                # Wait with timeout to prevent deadlock if checking thread crashed
                deadline = check_started + self._CHECK_WAIT_TIMEOUT
                while self.fallback_state["watcher_available"] == self._CHECKING:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        # Timeout - assume check failed, allow re-check
                        logger.warning("[fallback] Watcher check timed out waiting for another thread")
                        self.fallback_state["watcher_available"] = None
                        break
                    self._check_condition.wait(timeout=remaining)

                # Re-check state after waiting
                state = self.fallback_state["watcher_available"]
                if state is True or state is False:
                    return state
                # If still None, fall through to perform check ourselves

            # Mark as checking to prevent thundering herd
            self.fallback_state["watcher_available"] = self._CHECKING
            self.fallback_state["watcher_check_started"] = time.time()

        # Perform actual check OUTSIDE lock to avoid blocking other threads
        try:
            result = self._do_watcher_check()
            return result
        finally:
            # Always set a result and notify waiting threads
            with self._check_condition:
                # If still in checking state (not changed by another path), ensure it's set
                if self.fallback_state["watcher_available"] == self._CHECKING:
                    self.fallback_state["watcher_available"] = False
                self._check_condition.notify_all()

    def _do_watcher_check(self) -> bool:
        """Actually perform the Ollama availability check.

        Called outside the lock. Must always call _set_watcher_* to set result.
        """
        watcher_config = getattr(self.config, 'watcher', None)
        if not watcher_config or not getattr(watcher_config, 'enabled', True):
            self._set_watcher_unavailable("Watcher disabled in config")
            return False

        # Only check Ollama if the watcher provider is ollama
        provider = getattr(watcher_config, 'provider', 'ollama')
        if provider != 'ollama':
            self._set_watcher_unavailable(f"Watcher uses {provider}, not Ollama")
            return False

        host = getattr(watcher_config, 'host', 'http://localhost:11434')
        model = getattr(watcher_config, 'model', 'llama3.2')
        fallback_model = getattr(watcher_config, 'fallback_model', 'llama3.1')

        try:
            # Quick health check with reasonable timeout
            response = requests.get(f"{host}/api/tags", timeout=5)
            if response.status_code != 200:
                self._set_watcher_unavailable(f"Ollama returned status {response.status_code}")
                return False

            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            # Check primary model
            if model in model_names or any(model in name for name in model_names):
                self._set_watcher_available(model)
                return True

            # Check fallback model
            if fallback_model and (fallback_model in model_names or
                                   any(fallback_model in name for name in model_names)):
                logger.warning(f"[fallback] Primary model {model} not found, using {fallback_model}")
                self._set_watcher_available(fallback_model)
                return True

            # No model available
            self._set_watcher_unavailable(
                f"Models {model}/{fallback_model} not found. Available: {model_names}"
            )
            return False

        except requests.exceptions.ConnectionError:
            self._set_watcher_unavailable("Ollama not running")
            return False
        except requests.exceptions.Timeout:
            self._set_watcher_unavailable("Ollama health check timed out")
            return False
        except Exception as e:
            self._set_watcher_unavailable(f"Error checking Ollama: {e}")
            return False

    def _set_watcher_available(self, model: str):
        """Mark watcher as available."""
        with self._lock:
            self.fallback_state["watcher_available"] = True
            self.fallback_state["last_watcher_check"] = time.time()
            self.fallback_state["active_model"] = model
        logger.info(f"[fallback] Watcher ready: ollama/{model}")

    def _set_watcher_unavailable(self, reason: str):
        """Mark watcher as unavailable and log fallback."""
        with self._lock:
            self.fallback_state["watcher_available"] = False
            self.fallback_state["last_watcher_check"] = time.time()

        if self.healing_logger:
            self.healing_logger.log_fallback("INIT", "watcher", "pattern_routing", reason)
        else:
            logger.warning(f"[fallback] Watcher unavailable: {reason}")

    def check_llm_healer_available(self) -> bool:
        """Check if LLM healer can be used.

        Returns:
            True if LLM healer can be used, False otherwise
        """
        with self._lock:
            # Check if we should re-check after previous failure
            if self.fallback_state["llm_healer_available"] == False:
                last_check = self.fallback_state.get("last_llm_healer_check", 0)
                recheck_interval = getattr(
                    getattr(self.config, 'llm_healer', None),
                    'recheck_interval_seconds', 300.0
                )
                if time.time() - last_check > recheck_interval:
                    logger.info("[fallback] Re-checking LLM healer availability...")
                    self.fallback_state["llm_healer_available"] = None
                    self.fallback_state["llm_healer_failures"] = 0

            # Return cached result if available
            if self.fallback_state["llm_healer_available"] is not None:
                return self.fallback_state["llm_healer_available"]

        # Perform actual check
        llm_config = getattr(self.config, 'llm_healer', None)
        if not llm_config or not getattr(llm_config, 'enabled', True):
            self._set_llm_healer_unavailable("LLM healer disabled in config")
            return False

        provider = getattr(llm_config, 'provider', 'anthropic')

        # Check API key based on provider
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                self._set_llm_healer_unavailable("ANTHROPIC_API_KEY not set")
                return False
        elif provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                self._set_llm_healer_unavailable("GEMINI_API_KEY not set")
                return False
        # Ollama doesn't need API key

        with self._lock:
            self.fallback_state["llm_healer_available"] = True
            self.fallback_state["last_llm_healer_check"] = time.time()
        logger.info(f"[fallback] LLM healer ready: {provider}")
        return True

    def _set_llm_healer_unavailable(self, reason: str):
        """Mark LLM healer as unavailable and log fallback."""
        with self._lock:
            self.fallback_state["llm_healer_available"] = False
            self.fallback_state["last_llm_healer_check"] = time.time()

        if self.healing_logger:
            self.healing_logger.log_fallback("INIT", "llm_healer", "user_escalation", reason)
        else:
            logger.warning(f"[fallback] LLM healer unavailable: {reason}")

    def record_watcher_failure(self):
        """Track watcher failure for circuit breaker."""
        with self._lock:
            self.fallback_state["watcher_failures"] += 1
            failures = self.fallback_state["watcher_failures"]
            max_failures = getattr(
                getattr(self.config, 'watcher', None),
                'max_failures', 3
            )

            if failures >= max_failures:
                self.fallback_state["watcher_available"] = False
                self.fallback_state["last_watcher_check"] = time.time()

                if self.healing_logger:
                    self.healing_logger.log_fallback(
                        "RUNTIME", "watcher", "pattern_routing",
                        f"Watcher disabled after {failures} consecutive failures"
                    )
                else:
                    logger.warning(f"[fallback] Watcher disabled after {failures} failures")

    def record_watcher_success(self):
        """Reset watcher failure counter on success."""
        with self._lock:
            self.fallback_state["watcher_failures"] = 0

    def record_llm_healer_failure(self):
        """Track LLM healer failure for circuit breaker."""
        with self._lock:
            self.fallback_state["llm_healer_failures"] += 1
            failures = self.fallback_state["llm_healer_failures"]
            max_failures = getattr(
                getattr(self.config, 'llm_healer', None),
                'max_failures', 3
            )

            if failures >= max_failures:
                self.fallback_state["llm_healer_available"] = False
                self.fallback_state["last_llm_healer_check"] = time.time()

                if self.healing_logger:
                    self.healing_logger.log_fallback(
                        "RUNTIME", "llm_healer", "user_escalation",
                        f"LLM healer disabled after {failures} consecutive failures"
                    )
                else:
                    logger.warning(f"[fallback] LLM healer disabled after {failures} failures")

    def record_llm_healer_success(self):
        """Reset LLM healer failure counter on success."""
        with self._lock:
            self.fallback_state["llm_healer_failures"] = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current fallback chain status."""
        with self._lock:
            return {
                "watcher_available": self.fallback_state["watcher_available"],
                "llm_healer_available": self.fallback_state["llm_healer_available"],
                "watcher_failures": self.fallback_state["watcher_failures"],
                "llm_healer_failures": self.fallback_state["llm_healer_failures"],
                "active_model": self.fallback_state.get("active_model"),
            }
