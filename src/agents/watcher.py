"""WatcherAgent - Local LLM for error classification and routing."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .healing_logger import HealingLogger
    from .fallback import FallbackChain

logger = logging.getLogger(__name__)


@dataclass
class ErrorClassification:
    """Result from watcher error classification."""
    category: str           # api, disk, path, checkpoint, download, otio, config, unknown
    severity: str           # critical, recoverable, transient
    suggested_healer: str   # Name of suggested healer
    confidence: float       # 0.0 - 1.0
    needs_llm_healer: bool  # Whether to escalate to LLM healer
    reasoning: str          # Explanation from watcher


# Watcher prompt template
WATCHER_PROMPT = """You are an error classification agent for a video processing pipeline.

Classify the following error into one of these categories:
- api: Rate limits, authentication errors, quota exceeded, timeouts, connection errors
- disk: Storage full, permission denied, read-only filesystem
- path: Long paths (>260 chars on Windows), encoding issues, invalid characters
- checkpoint: JSON corruption, malformed checkpoint files
- download: Video unavailable, private video, age-restricted content
- otio: Timeline generation errors, invalid time ranges, media reference issues
- config: Configuration errors, missing required fields
- unknown: Cannot classify with confidence

Severity levels:
- critical: Cannot continue without user intervention
- recoverable: Can be fixed automatically
- transient: Temporary issue, retry may succeed

Error message:
{error_message}

Error type:
{error_type}

Stage:
{stage_name}

Additional context:
{context}

Respond with JSON only (no markdown, no explanation):
{{
    "category": "string",
    "severity": "transient|recoverable|critical",
    "suggested_healer": "api-healer|disk-healer|path-healer|checkpoint-healer|download-healer|otio-healer|",
    "confidence": 0.0-1.0,
    "needs_llm_healer": true|false,
    "reasoning": "brief explanation"
}}"""


class WatcherAgent:
    """Local LLM agent for fast error classification.

    Uses Ollama with a small model (llama3.2) for quick error triage.
    Decides which healer to try first and whether to escalate to LLM healer.
    """

    def __init__(self, config: 'WatcherConfig', project_dir: Optional[Path] = None,
                 healing_logger: Optional['HealingLogger'] = None,
                 fallback_chain: Optional['FallbackChain'] = None):
        """Initialize watcher agent.

        Args:
            config: WatcherConfig with provider, model, timeout settings
            project_dir: Optional project directory for context
            healing_logger: Optional logger for classification events
            fallback_chain: Optional fallback chain for failure tracking
        """
        self.config = config
        self.project_dir = project_dir
        self.healing_logger = healing_logger
        self.fallback_chain = fallback_chain

        self.provider = getattr(config, 'provider', 'ollama')
        self.model = getattr(config, 'model', 'llama3.2')
        self.fallback_model = getattr(config, 'fallback_model', 'llama3.1')
        self.host = getattr(config, 'host', 'http://localhost:11434')
        self.timeout = getattr(config, 'timeout', 30.0)
        self.escalate_threshold = getattr(config, 'escalate_threshold', 0.7)

        self.client = None
        self._initialized = False

        # Circuit breaker for repeated low-confidence classifications
        self._low_confidence_threshold = 0.3
        self._cb_failure_limit = 5
        self._cb_cooldown_seconds = 600.0  # 10 minutes
        self._cb_consecutive_low = 0
        self._cb_tripped = False
        self._cb_tripped_at: Optional[float] = None

    def _init_client(self):
        """Initialize LLM client lazily."""
        if self._initialized:
            return

        try:
            # For Ollama, verify it's running before creating client
            if self.provider == 'ollama':
                from src.llm_client.providers.ollama import check_ollama_model_available
                model_ok, error = check_ollama_model_available(self.model, self.host, timeout=5)
                if not model_ok:
                    # Try fallback model
                    if self.fallback_model:
                        fallback_ok, fallback_error = check_ollama_model_available(
                            self.fallback_model, self.host, timeout=5
                        )
                        if fallback_ok:
                            logger.info(f"[watcher] Using fallback model {self.fallback_model}")
                            self.model = self.fallback_model
                        else:
                            logger.warning(f"[watcher] Ollama unavailable: {error}")
                            self._initialized = False
                            return
                    else:
                        logger.warning(f"[watcher] Ollama unavailable: {error}")
                        self._initialized = False
                        return

            from src.llm_client import create_client
            self.client = create_client(
                self.provider,
                model=self.model,
                host=self.host if self.provider == 'ollama' else None
            )
            self._initialized = True
            logger.info(f"[watcher] Initialized with {self.provider}/{self.model}")
        except Exception as e:
            logger.warning(f"[watcher] Failed to initialize client: {e}")
            self._initialized = False

    def warmup(self) -> bool:
        """Warm up the model with a simple prompt.

        Returns:
            True if warmup successful, False otherwise
        """
        logger.info(f"[watcher] Warming up model {self.model}...")
        start = time.time()

        try:
            self._init_client()
            if not self.client:
                return False

            from src.llm_client import LLMRequest, ResponseFormat

            # Simple warmup prompt
            request = LLMRequest(
                prompt="Respond with: {\"status\": \"ready\"}",
                response_format=ResponseFormat.JSON,
                timeout=self.timeout
            )

            response = self.client.generate(request)
            elapsed = time.time() - start
            logger.info(f"[watcher] Model ready (warmup: {elapsed:.1f}s)")
            return True

        except Exception as e:
            logger.warning(f"[watcher] Warmup failed: {e}")
            return False

    def is_circuit_breaker_tripped(self) -> bool:
        """Check if the low-confidence circuit breaker is currently active.

        If the cooldown period has elapsed, auto-resets the circuit breaker.

        Returns:
            True if circuit breaker is tripped and cooldown has not elapsed.
        """
        if not self._cb_tripped:
            return False

        elapsed = time.time() - (self._cb_tripped_at or 0.0)
        if elapsed >= self._cb_cooldown_seconds:
            logger.info(
                "[watcher] Low-confidence circuit breaker auto-reset "
                f"after {elapsed:.0f}s cooldown"
            )
            self._cb_tripped = False
            self._cb_tripped_at = None
            self._cb_consecutive_low = 0
            return False

        return True

    def reset_circuit_breaker(self) -> None:
        """Manually reset the low-confidence circuit breaker.

        Can be called externally to re-enable the watcher before the
        cooldown period expires.
        """
        was_tripped = self._cb_tripped
        self._cb_consecutive_low = 0
        self._cb_tripped = False
        self._cb_tripped_at = None
        if was_tripped:
            logger.info("[watcher] Low-confidence circuit breaker manually reset")

    def _record_low_confidence(self, confidence: float) -> None:
        """Record a low-confidence classification and trip if threshold reached."""
        self._cb_consecutive_low += 1
        logger.debug(
            f"[watcher] Low-confidence classification ({confidence:.2f} < "
            f"{self._low_confidence_threshold}), "
            f"consecutive: {self._cb_consecutive_low}/{self._cb_failure_limit}"
        )

        if self._cb_consecutive_low >= self._cb_failure_limit:
            self._cb_tripped = True
            self._cb_tripped_at = time.time()
            logger.warning(
                f"[watcher] Circuit breaker TRIPPED: "
                f"{self._cb_consecutive_low} consecutive low-confidence "
                f"classifications (< {self._low_confidence_threshold}). "
                f"Falling through to pattern routing for "
                f"{self._cb_cooldown_seconds:.0f}s"
            )

    def _record_good_confidence(self) -> None:
        """Record an adequate-confidence classification, resetting the counter."""
        if self._cb_consecutive_low > 0:
            logger.debug(
                f"[watcher] Good confidence, resetting low-confidence counter "
                f"(was {self._cb_consecutive_low})"
            )
        self._cb_consecutive_low = 0

    def classify_error(self, error: Exception, context: Dict[str, Any]) -> Optional[ErrorClassification]:
        """Classify an error using the local LLM.

        Args:
            error: The exception that occurred
            context: Additional context (stage_name, config section, etc.)

        Returns:
            ErrorClassification or None if classification failed
        """
        # Check low-confidence circuit breaker first
        if self.is_circuit_breaker_tripped():
            logger.info("[watcher] Circuit breaker active, falling through to pattern routing")
            return None

        stage_name = context.get('stage', context.get('stage_name', 'UNKNOWN'))
        start_time = time.time()

        logger.info(f"[watcher] Classifying error... ({self.model})")

        try:
            self._init_client()
            if not self.client:
                logger.warning("[watcher] Client not initialized, falling back")
                if self.fallback_chain:
                    self.fallback_chain.record_watcher_failure()
                return None

            from src.llm_client import LLMRequest, ResponseFormat

            # Build prompt
            prompt = WATCHER_PROMPT.format(
                error_message=str(error)[:500],
                error_type=type(error).__name__,
                stage_name=stage_name,
                context=str(context.get('extra', ''))[:200]
            )

            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.JSON,
                timeout=self.timeout
            )

            response = self.client.generate(request)
            elapsed_ms = (time.time() - start_time) * 1000

            # Parse response
            classification = self._parse_response(response, error)

            if classification:
                # Track low-confidence vs good-confidence for circuit breaker
                if classification.confidence < self._low_confidence_threshold:
                    self._record_low_confidence(classification.confidence)
                else:
                    self._record_good_confidence()

                # Log success
                if self.fallback_chain:
                    self.fallback_chain.record_watcher_success()

                if self.healing_logger:
                    self.healing_logger.log_classification(
                        stage_name, error, classification, elapsed_ms
                    )

                return classification
            else:
                # Parse failed
                if self.fallback_chain:
                    self.fallback_chain.record_watcher_failure()
                return None

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"[watcher] Classification failed after {elapsed_ms:.0f}ms: {e}")

            if self.fallback_chain:
                self.fallback_chain.record_watcher_failure()

            if self.healing_logger:
                self.healing_logger.log_fallback(
                    stage_name, "watcher", "pattern_routing",
                    f"Classification failed: {e}"
                )

            return None

    def _parse_response(self, response: Any, error: Exception) -> Optional[ErrorClassification]:
        """Parse LLM response into ErrorClassification.

        Args:
            response: LLM response object
            error: Original error for fallback classification

        Returns:
            ErrorClassification or None if parsing failed
        """
        try:
            # Try to get parsed JSON
            if hasattr(response, 'parsed_data') and response.parsed_data:
                data = response.parsed_data
            elif hasattr(response, 'text'):
                import json
                # Try to extract JSON from text
                text = response.text.strip()
                # Handle markdown code blocks
                if text.startswith('```'):
                    lines = text.split('\n')
                    text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
                data = json.loads(text)
            else:
                logger.warning("[watcher] Response has no parseable content")
                return None

            # Validate and extract fields
            category = data.get('category', 'unknown')
            if category not in ('api', 'disk', 'path', 'checkpoint', 'download', 'otio', 'config', 'unknown'):
                category = 'unknown'

            severity = data.get('severity', 'recoverable')
            if severity not in ('critical', 'recoverable', 'transient'):
                severity = 'recoverable'

            confidence = float(data.get('confidence', 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            return ErrorClassification(
                category=category,
                severity=severity,
                suggested_healer=data.get('suggested_healer', ''),
                confidence=confidence,
                needs_llm_healer=data.get('needs_llm_healer', confidence < self.escalate_threshold),
                reasoning=data.get('reasoning', '')[:200]
            )

        except Exception as e:
            logger.warning(f"[watcher] Failed to parse response: {e}")
            return None

    def should_escalate(self, classification: Optional[ErrorClassification],
                       healer_result: Optional['HealerResult'] = None) -> bool:
        """Determine if error should be escalated to LLM healer.

        Args:
            classification: Classification from watcher (or None)
            healer_result: Optional result from standard healer attempt

        Returns:
            True if should escalate to LLM healer
        """
        # No classification = unknown error = escalate
        if not classification:
            return True

        # Watcher explicitly requested LLM healer
        if classification.needs_llm_healer:
            return True

        # Low confidence = escalate
        if classification.confidence < 0.5:
            return True

        # Standard healer failed = escalate if moderate confidence
        if healer_result and not healer_result.success:
            return classification.confidence < 0.8

        return False

    def get_healer_priority(self, classification: ErrorClassification) -> list:
        """Get ordered list of healers to try based on classification.

        Args:
            classification: Error classification from watcher

        Returns:
            List of healer names in priority order
        """
        # Category to healer mapping
        category_healers = {
            'api': ['api-healer'],
            'disk': ['disk-healer'],
            'path': ['path-healer'],
            'checkpoint': ['checkpoint-healer'],
            'download': ['download-healer'],
            'otio': ['otio-healer'],
            'config': [],  # Config errors often need LLM healer
            'unknown': [],
        }

        primary = category_healers.get(classification.category, [])

        # Add suggested healer if different
        if classification.suggested_healer and classification.suggested_healer not in primary:
            primary = [classification.suggested_healer] + primary

        return primary
