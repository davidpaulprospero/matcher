"""LLMHealer - Claude-powered healer with self-healing capability."""

import logging
import os
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from src.state import PipelineState
    from ..healing_logger import HealingLogger
    from ..fallback import FallbackChain

logger = logging.getLogger(__name__)


# LLM Healer prompt template
LLM_HEALER_PROMPT = """You are a debugging expert for a video processing pipeline called voiceover-matcher.

## Error Details
Error type: {error_type}
Error message: {error_message}

## Stack Trace
{stack_trace}

## Failed Healers
The following standard healers have already tried and failed:
{failed_healers}

## Relevant Configuration
{config_context}

## Task
Analyze this error and determine how to fix it.

1. If fixable via configuration change:
   - Provide exact config key and new value
   - Common fixes: timeout increases, mode changes, path adjustments

2. If needs to skip this item:
   - Return skip action with explanation

3. If cannot be fixed automatically:
   - Return abort with clear explanation for user

Respond with JSON only:
{{
    "root_cause": "brief explanation of why this error occurred",
    "fix_type": "config|skip|abort",
    "config_changes": {{"section.field": "new_value"}},
    "confidence": 0.0-1.0,
    "reasoning": "step by step explanation"
}}"""


class LLMHealer(Healer):
    """Claude-powered healer with self-healing capability.

    Uses Claude Sonnet (or fallback providers) to analyze complex errors
    that standard pattern-based healers cannot handle.

    Features:
    - Context-aware error analysis
    - Config modification suggestions
    - Self-healing on API failures (timeout increase, backoff, provider switch)
    - Provider fallback chain (anthropic -> gemini -> ollama)
    """

    name = "llm-healer"
    description = "Uses Claude to analyze and fix complex errors"
    error_patterns = []  # Handles anything escalated to it
    exception_types = []

    # Self-healing configuration
    MAX_SELF_HEAL_ATTEMPTS = 3
    FALLBACK_PROVIDERS = ["anthropic", "gemini", "ollama"]
    INITIAL_TIMEOUT = 60.0
    MAX_TIMEOUT = 300.0
    INITIAL_BACKOFF = 2.0
    MAX_CONTEXT_CHARS = 12000

    def __init__(self, config: 'LLMHealerConfig', project_dir: Path,
                 healing_logger: Optional['HealingLogger'] = None,
                 fallback_chain: Optional['FallbackChain'] = None):
        """Initialize LLM healer.

        Args:
            config: LLMHealerConfig with provider, model, timeout settings
            project_dir: Project directory for context
            healing_logger: Optional logger for healing events
            fallback_chain: Optional fallback chain for failure tracking
        """
        super().__init__(config, project_dir)
        self.llm_config = config
        self.healing_logger = healing_logger
        self.fallback_chain = fallback_chain

        self.current_provider = getattr(config, 'provider', 'anthropic')
        self.model = getattr(config, 'model', 'claude-sonnet-4-20250514')
        self.timeout = getattr(config, 'timeout', self.INITIAL_TIMEOUT)
        self.max_tokens = getattr(config, 'max_tokens', 4096)
        self.backoff = self.INITIAL_BACKOFF

        self.include_stack_trace = getattr(config, 'include_stack_trace', True)
        self.include_config_context = getattr(config, 'include_config_context', True)
        self.max_context_chars = getattr(config, 'max_context_chars', self.MAX_CONTEXT_CHARS)

        self.client = None
        self._initialized = False
        self._failed_healers: List[str] = []

    def _init_client(self, provider: Optional[str] = None):
        """Initialize LLM client for provider.

        Args:
            provider: Provider to use (defaults to current_provider)
        """
        provider = provider or self.current_provider

        try:
            from src.llm_client import create_client

            if provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                model = self.model or "claude-sonnet-4-20250514"
            elif provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
                model = "gemini-2.0-flash"
            elif provider == "ollama":
                api_key = None
                model = "llama3.2"
            else:
                raise ValueError(f"Unknown provider: {provider}")

            self.client = create_client(provider, api_key=api_key, model=model)
            self.current_provider = provider
            self._initialized = True
            logger.info(f"[llm-healer] Initialized with {provider}/{model}")

        except Exception as e:
            logger.warning(f"[llm-healer] Failed to initialize {provider}: {e}")
            raise

    def can_handle(self, error: Exception) -> bool:
        """LLM healer can handle any error escalated to it."""
        return True

    def add_failed_healer(self, healer_name: str):
        """Record a healer that already tried and failed.

        Args:
            healer_name: Name of the healer that failed
        """
        if healer_name not in self._failed_healers:
            self._failed_healers.append(healer_name)

    def clear_failed_healers(self):
        """Clear the list of failed healers."""
        self._failed_healers.clear()

    def fix(self, error: Exception, state: Optional['PipelineState'],
            stage_name: str, error_stack_trace: Optional[str] = None) -> HealerResult:
        """Attempt to fix error using LLM analysis, with self-healing.

        Args:
            error: The exception to fix
            state: Current pipeline state (for context)
            stage_name: Name of the stage that failed
            error_stack_trace: Optional pre-captured stack trace of the original error

        Returns:
            HealerResult indicating success/failure and action to take
        """
        # Store pre-captured stack trace for use in context building
        self._error_stack_trace = error_stack_trace
        self.log_attempt(f"Analyzing error in {stage_name} with {self.current_provider}")

        for attempt in range(self.MAX_SELF_HEAL_ATTEMPTS):
            try:
                return self._attempt_fix(error, state, stage_name)

            except Exception as e:
                error_str = str(e).lower()

                # Timeout - increase timeout
                if "timeout" in error_str or "timed out" in error_str:
                    old_timeout = self.timeout
                    self.timeout = min(self.timeout * 1.5, self.MAX_TIMEOUT)

                    if self.healing_logger:
                        self.healing_logger.log_self_heal(
                            "llm-healer", attempt + 1, self.MAX_SELF_HEAL_ATTEMPTS,
                            "timeout",
                            f"Increased timeout: {old_timeout:.0f}s → {self.timeout:.0f}s"
                        )

                    if self.timeout >= self.MAX_TIMEOUT:
                        # Can't increase further, try provider switch
                        if not self._switch_provider():
                            break
                    continue

                # Rate limit - exponential backoff
                elif "rate" in error_str or "429" in error_str:
                    if self.healing_logger:
                        self.healing_logger.log_self_heal(
                            "llm-healer", attempt + 1, self.MAX_SELF_HEAL_ATTEMPTS,
                            "rate_limit",
                            f"Backoff {self.backoff:.1f}s"
                        )

                    time.sleep(self.backoff)
                    self.backoff = min(self.backoff * 2, 60.0)
                    continue

                # Auth error - switch provider
                elif "api_key" in error_str or "auth" in error_str or "401" in error_str:
                    if self.healing_logger:
                        self.healing_logger.log_self_heal(
                            "llm-healer", attempt + 1, self.MAX_SELF_HEAL_ATTEMPTS,
                            "auth",
                            "Switching provider"
                        )

                    if not self._switch_provider():
                        break
                    continue

                # Unknown error - try provider switch
                else:
                    if self.healing_logger:
                        self.healing_logger.log_self_heal(
                            "llm-healer", attempt + 1, self.MAX_SELF_HEAL_ATTEMPTS,
                            "provider_error",
                            f"Error: {str(e)[:50]}, switching provider"
                        )

                    if not self._switch_provider():
                        break
                    continue

        # All attempts exhausted
        self.log_failure(f"Failed after {self.MAX_SELF_HEAL_ATTEMPTS} self-heal attempts")

        if self.fallback_chain:
            self.fallback_chain.record_llm_healer_failure()

        return HealerResult(
            success=False,
            action=HealerAction.ABORT,
            message=f"LLM healer exhausted all options",
            details={
                "attempts": self.MAX_SELF_HEAL_ATTEMPTS,
                "final_provider": self.current_provider
            }
        )

    def _attempt_fix(self, error: Exception, state: Optional['PipelineState'],
                    stage_name: str) -> HealerResult:
        """Single attempt to fix error using LLM.

        Args:
            error: The exception to fix
            state: Current pipeline state
            stage_name: Name of the failed stage

        Returns:
            HealerResult with fix instructions
        """
        if not self._initialized:
            self._init_client()

        from src.llm_client import LLMRequest, ResponseFormat

        # Build context
        context = self._build_context(error, state, stage_name)

        request = LLMRequest(
            prompt=context,
            response_format=ResponseFormat.JSON,
            timeout=self.timeout,
            max_tokens=self.max_tokens
        )

        start_time = time.time()
        response = self.client.generate(request)
        elapsed_ms = (time.time() - start_time) * 1000

        # Parse and apply fix
        result = self._process_response(response, error, state, stage_name)

        # Log success
        if result.success and self.fallback_chain:
            self.fallback_chain.record_llm_healer_success()

        if self.healing_logger:
            # Use pre-captured stack trace, not traceback.format_exc() which would be empty/wrong here
            self.healing_logger.log_healer_attempt(
                stage_name, self.name, error, result, elapsed_ms,
                stack_trace=getattr(self, '_error_stack_trace', None) if self.include_stack_trace else None
            )

        return result

    # Unicode characters that can be used for text direction attacks or obfuscation
    _UNICODE_DANGEROUS_CHARS = (
        # Bidirectional text controls (RTL attacks)
        '\u202a'  # LRE - Left-to-Right Embedding
        '\u202b'  # RLE - Right-to-Left Embedding
        '\u202c'  # PDF - Pop Directional Formatting
        '\u202d'  # LRO - Left-to-Right Override
        '\u202e'  # RLO - Right-to-Left Override (most dangerous)
        '\u2066'  # LRI - Left-to-Right Isolate
        '\u2067'  # RLI - Right-to-Left Isolate
        '\u2068'  # FSI - First Strong Isolate
        '\u2069'  # PDI - Pop Directional Isolate
        '\u200e'  # LRM - Left-to-Right Mark
        '\u200f'  # RLM - Right-to-Left Mark
        # Zero-width characters (obfuscation)
        '\u200b'  # Zero Width Space
        '\u200c'  # Zero Width Non-Joiner
        '\u200d'  # Zero Width Joiner
        '\ufeff'  # BOM / Zero Width No-Break Space
        '\u2060'  # Word Joiner
        # Other invisible/confusing characters
        '\u00ad'  # Soft Hyphen
        '\u034f'  # Combining Grapheme Joiner
        '\u061c'  # Arabic Letter Mark
        '\u115f'  # Hangul Choseong Filler
        '\u1160'  # Hangul Jungseong Filler
        '\u17b4'  # Khmer Vowel Inherent Aq
        '\u17b5'  # Khmer Vowel Inherent Aa
        '\u180e'  # Mongolian Vowel Separator
        '\u3164'  # Hangul Filler
        '\uffa0'  # Halfwidth Hangul Filler
    )

    # Cyrillic/Greek homoglyphs that look like Latin (for obfuscation detection)
    _HOMOGLYPH_MAP = {
        '\u0430': 'a',  # Cyrillic а -> Latin a
        '\u0435': 'e',  # Cyrillic е -> Latin e
        '\u043e': 'o',  # Cyrillic о -> Latin o
        '\u0440': 'p',  # Cyrillic р -> Latin p
        '\u0441': 'c',  # Cyrillic с -> Latin c
        '\u0443': 'y',  # Cyrillic у -> Latin y
        '\u0445': 'x',  # Cyrillic х -> Latin x
        '\u0410': 'A',  # Cyrillic А -> Latin A
        '\u0412': 'B',  # Cyrillic В -> Latin B
        '\u0415': 'E',  # Cyrillic Е -> Latin E
        '\u041a': 'K',  # Cyrillic К -> Latin K
        '\u041c': 'M',  # Cyrillic М -> Latin M
        '\u041d': 'H',  # Cyrillic Н -> Latin H
        '\u041e': 'O',  # Cyrillic О -> Latin O
        '\u0420': 'P',  # Cyrillic Р -> Latin P
        '\u0421': 'C',  # Cyrillic С -> Latin C
        '\u0422': 'T',  # Cyrillic Т -> Latin T
        '\u0425': 'X',  # Cyrillic Х -> Latin X
        '\u0392': 'B',  # Greek Β -> Latin B
        '\u0395': 'E',  # Greek Ε -> Latin E
        '\u0397': 'H',  # Greek Η -> Latin H
        '\u0399': 'I',  # Greek Ι -> Latin I
        '\u039a': 'K',  # Greek Κ -> Latin K
        '\u039c': 'M',  # Greek Μ -> Latin M
        '\u039d': 'N',  # Greek Ν -> Latin N
        '\u039f': 'O',  # Greek Ο -> Latin O
        '\u03a1': 'P',  # Greek Ρ -> Latin P
        '\u03a4': 'T',  # Greek Τ -> Latin T
        '\u03a7': 'X',  # Greek Χ -> Latin X
        '\u03a5': 'Y',  # Greek Υ -> Latin Y
        '\u0391': 'A',  # Greek Α -> Latin A
    }

    def _sanitize_for_prompt(self, text: str, max_length: int = 500) -> str:
        """Sanitize text for inclusion in LLM prompt.

        Comprehensive prompt injection defense that handles:
        1. Control characters removal
        2. Unicode direction override attacks (RTL, LRO, etc.)
        3. Zero-width character obfuscation
        4. Homoglyph attacks (Cyrillic/Greek lookalikes)
        5. Mid-string injection keywords (IGNORE, DISREGARD, etc.)
        6. Role/instruction patterns anywhere in text
        7. Code blocks and template syntax
        8. JSON injection attempts

        Security model: Attacker controls video titles, filenames, error messages
        from external APIs. All text positions must be sanitized.

        Args:
            text: Raw text to sanitize (potentially attacker-controlled)
            max_length: Maximum length after sanitization

        Returns:
            Sanitized text safe for prompt inclusion
        """
        import re

        if not text:
            return ""

        # Convert to string if needed
        text = str(text)

        # LAYER 1: Remove dangerous Unicode characters
        # These can manipulate text display (RTL attacks) or hide content
        for char in self._UNICODE_DANGEROUS_CHARS:
            text = text.replace(char, '')

        # LAYER 2: Normalize homoglyphs to ASCII equivalents
        # Prevents bypass using Cyrillic 'а' instead of Latin 'a'
        for homoglyph, ascii_char in self._HOMOGLYPH_MAP.items():
            text = text.replace(homoglyph, ascii_char)

        # LAYER 3: Remove control characters except newline/tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # LAYER 4: Neutralize injection keywords ANYWHERE in text
        # These are dangerous regardless of position
        # Wrap in brackets to make them inert data, not instructions
        injection_keywords = [
            # Direct instruction overrides
            r'\bIGNORE\b',
            r'\bDISREGARD\b',
            r'\bFORGET\b',
            r'\bOVERRIDE\b',
            r'\bBYPASS\b',
            r'\bSKIP\b',
            # Instruction references
            r'\bPREVIOUS\s+INSTRUCTIONS?\b',
            r'\bALL\s+INSTRUCTIONS?\b',
            r'\bABOVE\s+INSTRUCTIONS?\b',
            r'\bYOUR\s+INSTRUCTIONS?\b',
            r'\bTHESE\s+INSTRUCTIONS?\b',
            r'\bSYSTEM\s+PROMPT\b',
            # Context manipulation
            r'\bIGNORE\s+ABOVE\b',
            r'\bIGNORE\s+EVERYTHING\b',
            r'\bDISREGARD\s+ABOVE\b',
            r'\bDISREGARD\s+EVERYTHING\b',
            r'\bFORGET\s+EVERYTHING\b',
            r'\bFORGET\s+ABOVE\b',
            # Mode switches
            r'\bDEBUG\s+MODE\b',
            r'\bDEVELOPER\s+MODE\b',
            r'\bADMIN\s+MODE\b',
            r'\bTEST\s+MODE\b',
            r'\bRAW\s+CONFIG\b',
            r'\bRAW\s+OUTPUT\b',
            # Role impersonation
            r'\bYOU\s+ARE\s+NOW\b',
            r'\bACT\s+AS\b',
            r'\bPRETEND\s+TO\s+BE\b',
            r'\bROLE\s*:\s*SYSTEM\b',
            r'\bROLE\s*:\s*ADMIN\b',
        ]
        for pattern in injection_keywords:
            text = re.sub(pattern, lambda m: f'[{m.group(0)}]', text, flags=re.IGNORECASE)

        # LAYER 5: Escape role markers at any position
        # Pattern: word followed by colon that could be role marker
        role_patterns = [
            (r'\b(system)\s*:', r'[\1]:'),
            (r'\b(user)\s*:', r'[\1]:'),
            (r'\b(assistant)\s*:', r'[\1]:'),
            (r'\b(admin)\s*:', r'[\1]:'),
            (r'\b(human)\s*:', r'[\1]:'),
            (r'\b(ai)\s*:', r'[\1]:'),
            (r'\b(claude)\s*:', r'[\1]:'),
            (r'\b(gpt)\s*:', r'[\1]:'),
            (r'\b(instruction)\s*:', r'[\1]:'),
            (r'\b(command)\s*:', r'[\1]:'),
            (r'\b(task)\s*:', r'[\1]:'),
            (r'\b(prompt)\s*:', r'[\1]:'),
        ]
        for pattern, replacement in role_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # LAYER 6: Break code blocks (prevent code injection)
        text = re.sub(r'```', '[code]', text)
        text = re.sub(r'~~~', '[code]', text)

        # LAYER 7: Neutralize dangerous JSON keys ANYWHERE in text
        # These keys could confuse the LLM's response parsing
        # Wrap in brackets to make them inert data
        dangerous_json_keys = [
            r'"fix_type"',
            r'"action"',
            r'"abort"',
            r'"config"',
            r'"execute"',
            r'"eval"',
            r'"exec"',
            r'"cmd"',
            r'"command"',
            r'"shell"',
            r'"code"',
            r'"script"',
            r'"config_changes"',
            r'"root_cause"',
            r'"confidence"',
            r'"reasoning"',
        ]
        for key_pattern in dangerous_json_keys:
            text = re.sub(key_pattern, lambda m: f'[{m.group(0)}]', text, flags=re.IGNORECASE)

        # LAYER 8: Break template/placeholder patterns
        text = re.sub(r'\{\{', '{ {', text)
        text = re.sub(r'\}\}', '} }', text)
        text = re.sub(r'\$\{', '$ {', text)  # JS template literals
        text = re.sub(r'<%', '< %', text)    # ERB/ASP templates
        text = re.sub(r'%>', '% >', text)

        # LAYER 9: Truncate to safe length
        if len(text) > max_length:
            text = text[:max_length - 3] + "..."

        return text

    def _build_context(self, error: Exception, state: Optional['PipelineState'],
                      stage_name: str) -> str:
        """Build context for LLM with budget management.

        Sanitizes all user-controllable inputs to prevent prompt injection.

        Args:
            error: The exception
            state: Pipeline state
            stage_name: Stage name

        Returns:
            Formatted prompt string
        """
        budget = self.max_context_chars

        # Error type is safe (Python class name)
        error_type = type(error).__name__

        # Error message needs sanitization (could contain adversarial content)
        error_message = self._sanitize_for_prompt(str(error), 500)
        budget -= len(error_message)

        # Stack trace - use pre-captured trace if available, else try to get current
        stack_trace = "Not included"
        if self.include_stack_trace and budget > 1000:
            # Prefer pre-captured stack trace (captures the actual error context)
            raw_stack = getattr(self, '_error_stack_trace', None)
            if not raw_stack:
                # Fallback to current exception context (may not be the original error)
                raw_stack = traceback.format_exc()
                if raw_stack == "NoneType: None\n":
                    raw_stack = f"No stack trace available. Error: {error}"
            stack_trace = self._sanitize_for_prompt(raw_stack, min(2000, budget // 2))
            budget -= len(stack_trace)

        # Config context - this is internal data, but sanitize anyway
        config_context = "Not available"
        if self.include_config_context and state and budget > 500:
            raw_config = self._format_config(state)
            config_context = self._sanitize_for_prompt(raw_config, budget // 3)
            budget -= len(config_context)

        # Failed healers - these are internal healer names, safe
        # But validate they look like healer names (alphanumeric + dash)
        import re
        safe_healers = [h for h in self._failed_healers if re.match(r'^[\w-]+$', h)]
        failed_healers = "\n".join(f"- {h}" for h in safe_healers) or "None"

        return LLM_HEALER_PROMPT.format(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            failed_healers=failed_healers,
            config_context=config_context
        )

    def _format_config(self, state: 'PipelineState') -> str:
        """Format relevant config for context.

        Args:
            state: Pipeline state with config

        Returns:
            Formatted config string
        """
        if not state or not hasattr(state, 'config'):
            return "No config available"

        config = state.config
        lines = []

        # Include key config sections
        for section in ['download', 'transcription', 'matching', 'output', 'healing']:
            if hasattr(config, section):
                section_config = getattr(config, section)
                lines.append(f"[{section}]")
                if hasattr(section_config, '__dict__'):
                    for key, value in section_config.__dict__.items():
                        if not key.startswith('_'):
                            # Format nested configs (like caption_first) specially
                            if hasattr(value, '__dict__'):
                                lines.append(f"  [{key}]")
                                for nested_key, nested_val in value.__dict__.items():
                                    if not nested_key.startswith('_'):
                                        lines.append(f"    {nested_key}: {nested_val}")
                            else:
                                lines.append(f"  {key}: {value}")
                elif isinstance(section_config, dict):
                    for key, value in section_config.items():
                        lines.append(f"  {key}: {value}")

        return "\n".join(lines[:60])  # Limit lines

    def _process_response(self, response: Any, error: Exception,
                         state: Optional['PipelineState'],
                         stage_name: str) -> HealerResult:
        """Process LLM response and apply fix.

        Args:
            response: LLM response
            error: Original error
            state: Pipeline state
            stage_name: Stage name

        Returns:
            HealerResult with action to take
        """
        try:
            # Parse JSON response
            if hasattr(response, 'parsed_data') and response.parsed_data:
                data = response.parsed_data
            elif hasattr(response, 'text'):
                import json
                text = response.text.strip()
                if text.startswith('```'):
                    lines = text.split('\n')
                    text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
                data = json.loads(text)
            else:
                return HealerResult(
                    success=False,
                    action=HealerAction.ABORT,
                    message="LLM response not parseable"
                )

            root_cause = data.get('root_cause', 'Unknown')
            fix_type = data.get('fix_type', 'abort')
            config_changes = data.get('config_changes', {})
            confidence = float(data.get('confidence', 0.5))
            reasoning = data.get('reasoning', '')

            self.log_attempt(f"Root cause: {root_cause[:60]}")

            if fix_type == 'config' and config_changes:
                # Apply config changes
                applied = self._apply_config_changes(config_changes, state)
                if applied:
                    self.log_success(f"Config modified: {list(config_changes.keys())}")
                    return HealerResult(
                        success=True,
                        action=HealerAction.RETRY,
                        message=f"Config modified: {', '.join(config_changes.keys())}",
                        modified_config=True,
                        details={
                            "root_cause": root_cause,
                            "config_changes": config_changes,
                            "confidence": confidence,
                            "reasoning": reasoning
                        }
                    )
                else:
                    return HealerResult(
                        success=False,
                        action=HealerAction.ABORT,
                        message="Failed to apply config changes",
                        details={"config_changes": config_changes}
                    )

            elif fix_type == 'skip':
                self.log_attempt(f"Recommending skip: {reasoning[:60]}")
                return HealerResult(
                    success=True,
                    action=HealerAction.SKIP,
                    message=f"Skip recommended: {root_cause}",
                    details={
                        "root_cause": root_cause,
                        "reasoning": reasoning
                    }
                )

            else:  # abort
                self.log_failure(f"Cannot fix: {root_cause[:60]}")
                return HealerResult(
                    success=False,
                    action=HealerAction.ABORT,
                    message=f"Cannot fix automatically: {root_cause}",
                    details={
                        "root_cause": root_cause,
                        "reasoning": reasoning
                    }
                )

        except Exception as e:
            logger.error(f"[llm-healer] Failed to process response: {e}")
            return HealerResult(
                success=False,
                action=HealerAction.ABORT,
                message=f"Failed to process LLM response: {e}"
            )

    # Whitelist of config keys that LLM can safely modify
    # Format: "section.field" -> (min_value, max_value) for numeric, or allowed_values for string
    # None means any value is acceptable (for strings without restrictions)
    #
    # NOTE: Only 2-level deep paths (section.field) are supported for security.
    # Nested configs like download.caption_first.* are handled by specialized healers
    # (CaptionHealer) rather than LLMHealer to prevent prompt injection via deep paths.
    SAFE_CONFIG_KEYS: Dict[str, Any] = {
        # Download timeouts and retries
        "download.timeout": (5.0, 300.0),
        "download.max_retries": (1, 10),
        "download.buffer_seconds": (0.0, 120.0),
        "download.merge_gap_seconds": (0.0, 60.0),
        # Transcription settings
        "transcription.timeout": (30.0, 600.0),
        "transcription.chunk_length": (10, 60),
        # Matching thresholds
        "matching.min_score": (0.0, 1.0),
        "matching.max_candidates": (1, 100),
        # Output settings
        "output.gap_mode": ("none", "fill", "extend", "black"),
        "output.track_count": (1, 10),
        "output.include_disabled_tracks": (True, False),
        # Healing settings (only safe ones)
        "healing.heal_delay": (0.5, 30.0),
        "healing.max_attempts_per_stage": (1, 10),
        # API settings
        "api.timeout": (5.0, 300.0),
        "api.max_retries": (1, 10),
    }

    def _validate_config_value(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate a config value against whitelist constraints.

        Args:
            key: Config key in "section.field" format
            value: Proposed new value

        Returns:
            (is_valid, error_message) tuple
        """
        # Runtime mitigation: Reject deeply nested paths (LLM hallucination protection)
        # Valid keys are "section.field" format (exactly one dot)
        if key.count('.') > 1:
            logger.warning(f"[llm-healer] Rejected deeply nested config key: {key}")
            return False, f"Key '{key}' has invalid depth (max 1 level: section.field)"

        if key not in self.SAFE_CONFIG_KEYS:
            return False, f"Key '{key}' not in allowed config whitelist"

        constraint = self.SAFE_CONFIG_KEYS[key]

        if constraint is None:
            return True, ""

        if isinstance(constraint, tuple):
            # Check if it's a numeric range (exactly 2 numeric values)
            if len(constraint) == 2:
                min_val, max_val = constraint
                # Handle boolean tuple FIRST (True, False) or (False, True)
                # Must check before numeric because bool is subclass of int in Python
                if isinstance(min_val, bool) and isinstance(max_val, bool):
                    if value not in (True, False):
                        return False, f"Value must be True or False"
                    return True, ""
                # Handle numeric range (check type explicitly excludes bool)
                elif (isinstance(min_val, (int, float)) and not isinstance(min_val, bool) and
                      isinstance(max_val, (int, float)) and not isinstance(max_val, bool)):
                    try:
                        num_value = float(value)
                        if num_value < min_val or num_value > max_val:
                            return False, f"Value {value} outside range [{min_val}, {max_val}]"
                        return True, ""
                    except (TypeError, ValueError):
                        return False, f"Value '{value}' is not numeric"

            # Handle string enum as tuple (any length tuple of strings)
            if all(isinstance(v, str) for v in constraint):
                if value not in constraint:
                    return False, f"Value '{value}' not in allowed values: {constraint}"
                return True, ""

        return False, f"Unknown constraint type for {key}"

    def _apply_config_changes(self, changes: Dict[str, Any],
                             state: Optional['PipelineState']) -> bool:
        """Apply configuration changes with whitelist validation.

        Only allows modification of pre-approved config keys to prevent
        prompt injection attacks where LLM suggests dangerous config changes.

        Args:
            changes: Dict of "section.field" -> value
            state: Pipeline state with config

        Returns:
            True if at least one change was applied successfully
        """
        if not state or not hasattr(state, 'config'):
            logger.warning("[llm-healer] No config to modify")
            return False

        config = state.config
        applied = False
        rejected = []

        for key, value in changes.items():
            # Validate against whitelist
            is_valid, error_msg = self._validate_config_value(key, value)
            if not is_valid:
                logger.warning(f"[llm-healer] Rejected config change: {key}={value} - {error_msg}")
                rejected.append((key, error_msg))
                continue

            try:
                parts = key.split('.')
                if len(parts) != 2:
                    logger.warning(f"[llm-healer] Invalid key format: {key}")
                    continue

                section, field = parts
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    if hasattr(section_obj, field):
                        old_value = getattr(section_obj, field)
                        # Type coercion for numeric values
                        if isinstance(old_value, int) and not isinstance(value, int):
                            value = int(float(value))
                        elif isinstance(old_value, float) and not isinstance(value, float):
                            value = float(value)
                        setattr(section_obj, field, value)
                        logger.info(f"[llm-healer] {key}: {old_value} → {value}")
                        applied = True
                    elif isinstance(section_obj, dict) and field in section_obj:
                        old_value = section_obj.get(field)
                        section_obj[field] = value
                        logger.info(f"[llm-healer] {key}: {old_value} → {value}")
                        applied = True
                    else:
                        logger.warning(f"[llm-healer] Field '{field}' not found in section '{section}'")
                else:
                    logger.warning(f"[llm-healer] Section '{section}' not found in config")
            except Exception as e:
                logger.warning(f"[llm-healer] Failed to set {key}: {e}")

        if rejected:
            logger.warning(f"[llm-healer] {len(rejected)} config changes rejected by whitelist")

        return applied

    def _switch_provider(self) -> bool:
        """Switch to next available provider.

        Returns:
            True if switched successfully, False if no more providers
        """
        try:
            current_idx = self.FALLBACK_PROVIDERS.index(self.current_provider)
        except ValueError:
            current_idx = -1

        for provider in self.FALLBACK_PROVIDERS[current_idx + 1:]:
            try:
                if self.healing_logger:
                    self.healing_logger.log_provider_switch(self.current_provider, provider)

                self._init_client(provider)
                self.timeout = self.INITIAL_TIMEOUT  # Reset timeout
                self.backoff = self.INITIAL_BACKOFF  # Reset backoff
                return True

            except Exception as e:
                if self.healing_logger:
                    self.healing_logger.log_provider_switch_failed(provider, str(e))
                continue

        return False  # No more fallbacks
