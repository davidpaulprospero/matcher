"""
Healing Orchestrator - Coordinates self-healing pipeline execution.

The orchestrator manages:
- Preflight checks before pipeline runs
- Healer selection and prioritization
- Config snapshots and rollback
- User escalation for complex issues
- Metrics collection and reporting
- Two-tier LLM delegation (watcher + LLM healer)
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .base import Healer, HealerResult, HealerAction
from .strategy import (
    HealingStrategy,
    HealingMode,
    HealingMetrics,
    ConfigSnapshot,
)
from .healers import HEALER_REGISTRY
from .fallback import FallbackChain, pattern_route, PatternClassification
from .healer_cache import HealerResultCache, HealerCacheConfig

if TYPE_CHECKING:
    from ..config import Config
    from ..state import PipelineState
    from ..stages import Stage, StageResult
    from ..checkpoint import CheckpointManager
    from .watcher import WatcherAgent, ErrorClassification
    from .healing_logger import HealingLogger
    from .healers.llm_healer import LLMHealer

logger = logging.getLogger(__name__)


@dataclass
class PreflightIssue:
    """Issue detected during preflight checks."""
    category: str
    severity: str  # "critical", "warning", "info"
    message: str
    auto_fixable: bool = False
    healer: Optional[str] = None


@dataclass
class EscalationRequest:
    """Request for user intervention."""
    stage_name: str
    error: str
    options: List[str]
    recommendation: Optional[str] = None


class HealingOrchestrator:
    """
    Coordinates self-healing across the pipeline.

    Features:
    - Preflight checks before running
    - Smart healer selection based on error type
    - Config rollback on failure
    - Cross-healer coordination
    - User escalation for complex issues
    - Comprehensive metrics
    """

    def __init__(
        self,
        config: 'Config',
        project_dir: Path,
        strategy: HealingStrategy = None,
        healers: List[Type[Healer]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Pipeline configuration
            project_dir: Project directory
            strategy: Healing strategy (defaults to conservative)
            healers: Optional list of healer classes
        """
        self.config = config
        self.project_dir = Path(project_dir)
        self.strategy = strategy or HealingStrategy.conservative()

        # Initialize healers
        healer_classes = healers or HEALER_REGISTRY
        self._healer_instances: Dict[str, Healer] = {}
        for cls in healer_classes:
            healer = cls(config, project_dir)
            if healer.name not in self.strategy.skip_healers:
                self._healer_instances[healer.name] = healer

        # Reorder by priority
        self.healers = self._prioritize_healers()

        # State tracking
        self.metrics = HealingMetrics()
        self.config_snapshots: List[ConfigSnapshot] = []
        self.current_stage: Optional[str] = None
        self.escalation_callback: Optional[Callable[[EscalationRequest], str]] = None

        # Rate limit metrics from download stage (set via set_rate_limit_metrics)
        self._rate_limit_metrics: Optional[Any] = None
        # Escalation metrics from download stage (set via set_escalation_metrics)
        self._escalation_metrics: Optional[Dict[str, Any]] = None

        # Unified aggregated metrics (set via set_aggregated_metrics)
        self._aggregated_metrics = None

        # MullvadVPN instance for Tier 4 bypass (US-35-002)
        # Set via set_mullvad_vpn() when config.download.mullvad.enabled=true
        self._mullvad_vpn = None

        # Cross-healer state
        self._healer_state: Dict[str, Any] = {}

        # Thread safety for recent errors tracking
        self._errors_lock = threading.Lock()
        self.recent_errors: Dict[str, int] = {}
        self._current_error_stack: Optional[str] = None  # Pre-captured stack trace for LLM healer

        # Two-tier LLM delegation components
        self.healing_logger: Optional['HealingLogger'] = None
        self.fallback_chain: Optional[FallbackChain] = None
        self.watcher: Optional['WatcherAgent'] = None
        self.llm_healer: Optional['LLMHealer'] = None

        # Healer result cache for idempotent healing (US-64-007)
        self._healer_cache = self._init_healer_cache()

        # Initialize two-tier LLM system if enabled
        self._init_llm_delegation()

    def _init_healer_cache(self) -> HealerResultCache:
        """Initialize healer result cache from config (US-64-007).

        Reads cache settings from healing.healer_cache config section,
        or uses defaults if not configured.
        """
        healing_config = getattr(self.config, 'healing', None)
        cache_config = None

        if healing_config:
            healer_cache_cfg = getattr(healing_config, 'healer_cache', None)
            if healer_cache_cfg:
                cache_config = HealerCacheConfig(
                    enabled=getattr(healer_cache_cfg, 'enabled', True),
                    ttl_seconds=getattr(healer_cache_cfg, 'ttl_seconds', 300.0),
                    max_entries=getattr(healer_cache_cfg, 'max_entries', 100)
                )

        cache = HealerResultCache(cache_config)
        logger.debug(f"[orchestrator] Healer cache initialized: {cache}")
        return cache

    def _init_llm_delegation(self):
        """Initialize two-tier LLM delegation system (watcher + LLM healer)."""
        healing_config = getattr(self.config, 'healing', None)
        if not healing_config:
            return

        # Check if watcher/LLM healer are configured
        watcher_config = getattr(healing_config, 'watcher', None)
        llm_healer_config = getattr(healing_config, 'llm_healer', None)
        logging_config = getattr(healing_config, 'logging', None)

        # Initialize healing logger
        if logging_config and getattr(logging_config, 'enabled', True):
            try:
                from .healing_logger import HealingLogger
                log_dir = self.project_dir / getattr(logging_config, 'log_dir', 'logs')
                self.healing_logger = HealingLogger(
                    log_dir,
                    json_log=getattr(logging_config, 'json_log', True),
                    console_format=getattr(logging_config, 'console_format', 'box')
                )
                logger.info("[orchestrator] Healing logger initialized")
            except Exception as e:
                logger.warning(f"[orchestrator] Could not initialize healing logger: {e}")

        # Initialize fallback chain
        self.fallback_chain = FallbackChain(healing_config, self.healing_logger)

        # Initialize watcher
        if watcher_config and getattr(watcher_config, 'enabled', True):
            try:
                from .watcher import WatcherAgent
                self.watcher = WatcherAgent(
                    watcher_config,
                    self.project_dir,
                    self.healing_logger,
                    self.fallback_chain
                )
                logger.info("[orchestrator] Watcher agent initialized")
            except Exception as e:
                logger.warning(f"[orchestrator] Could not initialize watcher: {e}")

        # Initialize LLM healer
        if llm_healer_config and getattr(llm_healer_config, 'enabled', True):
            try:
                from .healers.llm_healer import LLMHealer
                self.llm_healer = LLMHealer(
                    llm_healer_config,
                    self.project_dir,
                    self.healing_logger,
                    self.fallback_chain
                )
                # Add to healer instances so it can be selected
                self._healer_instances['llm-healer'] = self.llm_healer
                logger.info("[orchestrator] LLM healer initialized")
            except Exception as e:
                logger.warning(f"[orchestrator] Could not initialize LLM healer: {e}")

    def _prioritize_healers(self) -> List[Healer]:
        """Order healers by strategy priority.

        Logs warning if a healer specified in priority list is not found,
        which helps catch configuration typos.
        """
        ordered = []
        seen = set()

        # Add in priority order, warn about missing healers
        for name in self.strategy.healer_priority:
            if name in self._healer_instances:
                if name not in seen:
                    ordered.append(self._healer_instances[name])
                    seen.add(name)
            else:
                logger.warning(
                    f"[orchestrator] Healer '{name}' in priority list not found. "
                    f"Available: {list(self._healer_instances.keys())}"
                )

        # Add remaining healers (except LLM healer - it's tried last)
        for name, healer in self._healer_instances.items():
            if name not in seen and name != 'llm-healer':
                ordered.append(healer)

        return ordered

    # =========================================================================
    # PREFLIGHT CHECKS
    # =========================================================================

    def run_preflight(self, state: 'PipelineState') -> List[PreflightIssue]:
        """
        Run preflight checks before pipeline execution.

        Returns list of issues found.
        """
        if not self.strategy.run_preflight:
            return []

        logger.info("Running preflight checks...")
        issues: List[PreflightIssue] = []

        # Warm up watcher (local LLM) if enabled
        if self.watcher:
            watcher_config = getattr(getattr(self.config, 'healing', None), 'watcher', None)
            if watcher_config and getattr(watcher_config, 'warmup_on_preflight', True):
                logger.info("[preflight] Warming up watcher model...")
                if not self.watcher.warmup():
                    issues.append(PreflightIssue(
                        category="watcher",
                        severity="warning",
                        message="Watcher model warmup failed, will use pattern routing as fallback",
                        auto_fixable=False,
                    ))

        # Check disk space
        disk_issues = self._check_disk_space()
        issues.extend(disk_issues)

        # Check API keys
        api_issues = self._check_api_keys()
        issues.extend(api_issues)

        # Check Ollama availability for watcher
        ollama_issues = self._check_ollama()
        issues.extend(ollama_issues)

        # Check paths
        path_issues = self._check_paths()
        issues.extend(path_issues)

        # Check state/matches (for mid-pipeline runs)
        if hasattr(state, 'matches') and state.matches:
            match_issues = self._check_matches(state)
            issues.extend(match_issues)

        # Run healer-specific preflight checks
        for healer in self.healers:
            if hasattr(healer, 'preflight_check'):
                healer_issues = healer.preflight_check(state)
                for issue_msg in healer_issues:
                    issues.append(PreflightIssue(
                        category=healer.name,
                        severity="warning",
                        message=issue_msg,
                        auto_fixable=True,
                        healer=healer.name,
                    ))

        self.metrics.preflight_issues_found = len(issues)

        # Log issues
        for issue in issues:
            if issue.severity == "critical":
                logger.error(f"Preflight CRITICAL: {issue.message}")
            elif issue.severity == "warning":
                logger.warning(f"Preflight warning: {issue.message}")
            else:
                logger.info(f"Preflight info: {issue.message}")

        return issues

    def fix_preflight_issues(
        self,
        issues: List[PreflightIssue],
        state: 'PipelineState'
    ) -> Tuple[int, int]:
        """
        Attempt to fix preflight issues.

        Returns (fixed_count, remaining_count).
        """
        if not self.strategy.auto_fix_preflight:
            return 0, len(issues)

        fixed = 0
        for issue in issues:
            if not issue.auto_fixable:
                continue

            if issue.healer and issue.healer in self._healer_instances:
                healer = self._healer_instances[issue.healer]
                try:
                    # Create a synthetic error for the healer
                    error = Exception(issue.message)
                    result = healer.fix(error, state, "PREFLIGHT")
                    if result.success:
                        fixed += 1
                        logger.info(f"Fixed preflight issue: {issue.message}")
                except Exception as e:
                    logger.warning(f"Could not fix preflight issue: {e}")

        self.metrics.preflight_issues_fixed = fixed
        return fixed, len(issues) - fixed

    def _check_disk_space(self) -> List[PreflightIssue]:
        """Check available disk space."""
        issues = []
        try:
            import shutil
            usage = shutil.disk_usage(self.project_dir)
            free_gb = usage.free / (1024 ** 3)

            if free_gb < 1.0:
                issues.append(PreflightIssue(
                    category="disk",
                    severity="critical",
                    message=f"Low disk space: {free_gb:.1f} GB free",
                    auto_fixable=True,
                    healer="disk-healer",
                ))
            elif free_gb < 5.0:
                issues.append(PreflightIssue(
                    category="disk",
                    severity="warning",
                    message=f"Disk space warning: {free_gb:.1f} GB free",
                    auto_fixable=False,
                ))
        except Exception:
            pass
        return issues

    def _check_api_keys(self) -> List[PreflightIssue]:
        """Check required API keys are set."""
        import os
        issues = []

        # Check for common API keys
        key_checks = [
            ("GEMINI_API_KEY", "Gemini LLM", "warning"),
            ("ANTHROPIC_API_KEY", "Anthropic LLM", "info"),
            ("PEXELS_API_KEY", "Pexels stock media", "info"),
            ("PIXABAY_API_KEY", "Pixabay stock media", "info"),
        ]

        for env_var, service, severity in key_checks:
            if not os.environ.get(env_var):
                issues.append(PreflightIssue(
                    category="api",
                    severity=severity,
                    message=f"{env_var} not set ({service} unavailable)",
                    auto_fixable=False,
                ))

        return issues

    def _check_ollama(self) -> List[PreflightIssue]:
        """Check if Ollama is available for watcher agent."""
        issues = []

        # Check if watcher is enabled
        healing_config = getattr(self.config, 'healing', None)
        if not healing_config:
            return issues

        watcher_config = getattr(healing_config, 'watcher', None)
        if not watcher_config or not getattr(watcher_config, 'enabled', True):
            return issues

        # Only check Ollama if the watcher provider is ollama
        provider = getattr(watcher_config, 'provider', 'ollama')
        if provider != 'ollama':
            return issues

        # Get Ollama settings
        host = getattr(watcher_config, 'host', 'http://localhost:11434')
        model = getattr(watcher_config, 'model', 'llama3.2')
        fallback_model = getattr(watcher_config, 'fallback_model', 'llama3.1')

        try:
            from src.llm_client.providers.ollama import check_ollama_available, check_ollama_model_available

            # Check if Ollama is running
            available, error, models = check_ollama_available(host, timeout=5)

            if not available:
                issues.append(PreflightIssue(
                    category="watcher",
                    severity="warning",
                    message=f"Ollama not available: {error}. Error classification will use pattern routing.",
                    auto_fixable=False,
                ))
                return issues

            # Check if required model is installed
            model_ok, model_error = check_ollama_model_available(model, host, timeout=5)
            if not model_ok:
                # Try fallback model
                fallback_ok, _ = check_ollama_model_available(fallback_model, host, timeout=5)
                if fallback_ok:
                    issues.append(PreflightIssue(
                        category="watcher",
                        severity="info",
                        message=f"Primary model '{model}' not found, will use fallback '{fallback_model}'",
                        auto_fixable=False,
                    ))
                else:
                    issues.append(PreflightIssue(
                        category="watcher",
                        severity="warning",
                        message=f"{model_error}",
                        auto_fixable=False,
                    ))

        except ImportError:
            # llm_client not available
            pass
        except Exception as e:
            logger.debug(f"[preflight] Could not check Ollama: {e}")

        return issues

    def _check_paths(self) -> List[PreflightIssue]:
        """Check path configuration."""
        issues = []

        # Check project dir exists
        if not self.project_dir.exists():
            issues.append(PreflightIssue(
                category="path",
                severity="critical",
                message=f"Project directory does not exist: {self.project_dir}",
                auto_fixable=False,
            ))

        # Check for long paths on Windows
        project_str = str(self.project_dir)
        if len(project_str) > 200:
            issues.append(PreflightIssue(
                category="path",
                severity="warning",
                message=f"Project path is long ({len(project_str)} chars), may hit Windows limit",
                auto_fixable=True,
                healer="path-healer",
            ))

        return issues

    def _check_matches(self, state: 'PipelineState') -> List[PreflightIssue]:
        """Check match state for issues."""
        issues = []

        if not state.matches:
            return issues

        missing_media = 0
        for match in state.matches:
            video_path = None
            for attr in ['video_path', 'source_file', 'file_path', 'path', 'file']:
                if hasattr(match, attr):
                    video_path = getattr(match, attr)
                    break

            if video_path and not Path(video_path).exists():
                missing_media += 1

        if missing_media > 0:
            issues.append(PreflightIssue(
                category="media",
                severity="warning",
                message=f"{missing_media} media files not found",
                auto_fixable=True,
                healer="otio-healer",
            ))

        return issues

    # =========================================================================
    # CONFIG SNAPSHOT / ROLLBACK
    # =========================================================================

    def snapshot_config(self, stage_name: str) -> ConfigSnapshot:
        """Create a snapshot of current config for rollback."""
        snapshot = ConfigSnapshot(
            stage_name=stage_name,
            timestamp=time.time(),
            config_values={},
        )

        # Snapshot key config sections
        sections_to_snapshot = ['output', 'download', 'llm', 'matching']

        for section_name in sections_to_snapshot:
            section = getattr(self.config, section_name, None)
            if section is None:
                continue

            # Get all non-private attributes
            for attr in dir(section):
                if attr.startswith('_'):
                    continue
                try:
                    value = getattr(section, attr)
                    # Only snapshot simple types
                    if isinstance(value, (str, int, float, bool, type(None))):
                        key = f"{section_name}.{attr}"
                        if not any(p in key.lower() for p in self.strategy.protected_config_keys):
                            snapshot.config_values[key] = value
                except Exception:
                    pass

        self.config_snapshots.append(snapshot)
        return snapshot

    def rollback_config(self, to_stage: str = None) -> bool:
        """
        Rollback config to a previous snapshot.

        Args:
            to_stage: Stage name to rollback to (defaults to most recent)

        Returns:
            True if rollback was successful
        """
        if not self.strategy.enable_rollback:
            logger.warning("Rollback disabled in strategy")
            return False

        if not self.config_snapshots:
            logger.warning("No config snapshots available for rollback")
            return False

        # Find snapshot
        snapshot = None
        if to_stage:
            for s in reversed(self.config_snapshots):
                if s.stage_name == to_stage:
                    snapshot = s
                    break
        else:
            snapshot = self.config_snapshots[-1]

        if not snapshot:
            logger.warning(f"No snapshot found for stage: {to_stage}")
            return False

        # Restore
        success = snapshot.restore(self.config)
        if success:
            self.metrics.rollbacks_performed += 1
            logger.info(f"Rolled back config to stage: {snapshot.stage_name}")

        return success

    # =========================================================================
    # HEALER COORDINATION
    # =========================================================================

    def select_healers(
        self,
        error: Exception,
        stage_name: str
    ) -> List[Healer]:
        """
        Select appropriate healers for an error.

        Returns healers in priority order.
        """
        applicable = []

        for healer in self.healers:
            if healer.can_handle(error, stage_name):
                applicable.append(healer)

        # In aggressive mode, return all applicable healers
        if self.strategy.mode == HealingMode.AGGRESSIVE:
            return applicable

        # In minimal mode, return only the first match
        if self.strategy.mode == HealingMode.MINIMAL:
            return applicable[:1]

        # Conservative/Interactive: return top 3
        return applicable[:3]

    def coordinate_heal(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str,
        error_stack: Optional[str] = None
    ) -> HealerResult:
        """
        Coordinate healing attempt across multiple healers.

        Uses two-tier LLM delegation:
        1. Watcher (local Ollama) classifies error
        2. Standard healers try to fix based on classification
        3. LLM healer (Claude) handles complex issues

        Args:
            error: The exception that occurred
            state: Pipeline state
            stage_name: Name of the stage that failed
            error_stack: Pre-captured stack trace for LLM healer context
        """
        start_time = time.time()
        self.current_stage = stage_name
        self._current_error_stack = error_stack  # Store for LLM healer

        # Check for circular healing
        if not self._should_attempt_heal(error, stage_name):
            return HealerResult.failed("Healing loop detected, aborting")

        # US-64-007: Check healer cache for recently-healed identical error
        cached_result = self._healer_cache.get(error, stage_name)
        if cached_result:
            logger.info(
                f"[orchestrator] Skipping heal: identical error recently fixed "
                f"by {cached_result.healer_name} ({cached_result.action})"
            )
            self.metrics.time_spent_healing += time.time() - start_time
            # Return a success result based on cached data
            action = HealerAction.RETRY  # Default action for cached results
            try:
                action = HealerAction(cached_result.action)
            except (ValueError, KeyError):
                pass
            return HealerResult.fixed(
                f"Cached heal: {cached_result.message}",
                action=action,
                cached=True,
                original_healer=cached_result.healer_name
            )

        # Check if we should escalate immediately
        error_str = str(error).lower()
        if any(p in error_str for p in self.strategy.always_escalate):
            return self._escalate_to_user(error, stage_name)

        # Step 1: Get classification from watcher or pattern routing
        classification = self._classify_error(error, stage_name)

        # Step 2: Select and try healers based on classification
        result, healer_name = self._try_healers_with_classification(
            error, state, stage_name, classification
        )

        if result.success:
            self.metrics.time_spent_healing += time.time() - start_time
            # US-64-007: Cache successful result
            if healer_name:
                self._healer_cache.store(error, stage_name, result, healer_name)
                # Invalidate cache if config was modified
                if result.modified_config:
                    self._healer_cache.invalidate_on_config_change()
            return result

        # Step 3: Escalate to LLM healer if needed
        if self._should_escalate_to_llm(classification, result):
            llm_result = self._try_llm_healer(error, state, stage_name)
            if llm_result:
                self.metrics.time_spent_healing += time.time() - start_time
                # US-64-007: Cache LLM healer result
                self._healer_cache.store(error, stage_name, llm_result, "llm_healer")
                if llm_result.modified_config:
                    self._healer_cache.invalidate_on_config_change()
                return llm_result

        # All healers failed
        self.metrics.time_spent_healing += time.time() - start_time

        # Escalate in interactive mode
        if self.strategy.mode == HealingMode.INTERACTIVE:
            return self._escalate_to_user(error, stage_name)

        return HealerResult.failed(f"All healers failed for: {type(error).__name__}")

    def _classify_error(self, error: Exception, stage_name: str) -> Optional[Any]:
        """Classify error using watcher or pattern routing.

        Returns ErrorClassification or PatternClassification.
        Also records error category in metrics for dashboard (US-64-005).
        """
        context = {'stage': stage_name, 'stage_name': stage_name}

        # Try watcher first
        if self.watcher and self.fallback_chain and self.fallback_chain.check_watcher_available():
            classification = self.watcher.classify_error(error, context)
            if classification:
                # Record error category for dashboard metrics (US-64-005)
                category = getattr(classification, 'category', 'unknown')
                self.metrics.record_error_category(category)
                return classification

        # Fallback to pattern routing
        classification = pattern_route(str(error))

        # Record error category from pattern classification (US-64-005)
        if classification:
            category = getattr(classification, 'category', 'unknown')
            self.metrics.record_error_category(category)

        return classification

    def _try_healers_with_classification(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str,
        classification: Optional[Any]
    ) -> Tuple[HealerResult, Optional[str]]:
        """Try healers based on classification confidence.

        Returns:
            Tuple of (HealerResult, healer_name) where healer_name is the name
            of the healer that succeeded, or None if all failed.
        """
        failed_healers: List[str] = []

        # Get suggested healer from classification
        suggested_healer = None
        confidence = 0.5

        if classification:
            suggested_healer = getattr(classification, 'suggested_healer', None)
            confidence = getattr(classification, 'confidence', 0.5)

        # High confidence: try suggested healer first
        if confidence >= 0.8 and suggested_healer:
            healer = self._healer_instances.get(suggested_healer)
            if healer:
                result = self._try_healer(healer, error, state, stage_name)
                if result.success:
                    return result, healer.name
                failed_healers.append(healer.name)

        # Medium confidence: try suggested, then others
        elif confidence >= 0.5 and suggested_healer:
            healer = self._healer_instances.get(suggested_healer)
            if healer:
                result = self._try_healer(healer, error, state, stage_name)
                if result.success:
                    return result, healer.name
                failed_healers.append(healer.name)

        # Try all applicable healers
        healers = self.select_healers(error, stage_name)
        for healer in healers:
            if healer.name in failed_healers:
                continue

            result = self._try_healer(healer, error, state, stage_name)
            if result.success:
                return result, healer.name
            failed_healers.append(healer.name)

        # Record failed healers for LLM healer context
        if self.llm_healer:
            for name in failed_healers:
                self.llm_healer.add_failed_healer(name)

        return HealerResult.failed("Standard healers exhausted"), None

    def _get_healer_timeout(self, healer_name: str) -> float:
        """Get timeout for a specific healer.

        Returns the per-healer timeout from strategy.healer_timeouts dict,
        or DEFAULT_HEALER_TIMEOUT if healer not configured.
        """
        return self.strategy.healer_timeouts.get(
            healer_name,
            self.strategy.DEFAULT_HEALER_TIMEOUT
        )

    def _get_healer_max_attempts(self, healer_name: str) -> int:
        """Get max attempts for a specific healer.

        Returns the per-healer max_attempts from strategy.healer_max_attempts dict,
        or max_attempts_per_stage if healer not configured.
        """
        return self.strategy.healer_max_attempts.get(
            healer_name,
            self.strategy.max_attempts_per_stage
        )

    def _try_healer(
        self,
        healer: Healer,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Try a single healer and record metrics.

        Respects per-healer timeout from strategy.healer_timeouts.
        """
        healer_timeout = self._get_healer_timeout(healer.name)
        logger.info(f"Trying healer: {healer.name} (timeout: {healer_timeout}s)")
        start_time = time.time()

        try:
            # Run healer with timeout
            # Note: Using shutdown(wait=False) allows us to return immediately
            # on timeout without waiting for the thread to complete
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(healer.fix, error, state, stage_name)
                try:
                    result = future.result(timeout=healer_timeout)
                except FuturesTimeoutError:
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.warning(
                        f"Healer {healer.name} timed out after {healer_timeout}s"
                    )
                    self.metrics.errors_encountered.append(
                        f"{healer.name} timed out after {healer_timeout}s"
                    )
                    result = HealerResult.failed(
                        f"Healer timed out after {healer_timeout}s"
                    )
                    # Record timeout as failed heal
                    self.metrics.record_heal(healer.name, stage_name, False)
                    if self.healing_logger:
                        self.healing_logger.log_healer_attempt(
                            stage_name, healer.name, error, result, elapsed_ms
                        )
                    return result
            finally:
                # Don't wait for the thread to finish on timeout
                executor.shutdown(wait=False)

            elapsed_ms = (time.time() - start_time) * 1000

            # Record metrics with heal time (US-64-005)
            self.metrics.record_heal(healer.name, stage_name, result.success, elapsed_ms)

            # Log to healing logger
            if self.healing_logger:
                self.healing_logger.log_healer_attempt(
                    stage_name, healer.name, error, result, elapsed_ms
                )

            if result.success:
                # Notify other healers of changes
                self._notify_healers(healer.name, result)

            return result

        except Exception as heal_error:
            logger.error(f"Healer {healer.name} raised exception: {heal_error}")
            self.metrics.errors_encountered.append(str(heal_error))
            return HealerResult.failed(f"Healer exception: {heal_error}")

    def _should_escalate_to_llm(
        self,
        classification: Optional[Any],
        healer_result: HealerResult
    ) -> bool:
        """Determine if should escalate to LLM healer."""
        if not self.llm_healer:
            return False

        if not self.fallback_chain or not self.fallback_chain.check_llm_healer_available():
            return False

        # Classification explicitly requested LLM healer
        if classification and getattr(classification, 'needs_llm_healer', False):
            return True

        # Low confidence classification
        if classification and getattr(classification, 'confidence', 1.0) < 0.5:
            return True

        # Standard healers failed
        if not healer_result.success:
            return True

        return False

    def _try_llm_healer(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> Optional[HealerResult]:
        """Try LLM healer for complex issues."""
        if not self.llm_healer:
            return None

        if self.healing_logger:
            self.healing_logger.log_escalation(
                stage_name, error, "Standard healers failed, escalating to LLM healer"
            )

        logger.info("[orchestrator] Escalating to LLM healer")

        try:
            # Pass the pre-captured stack trace for accurate context
            result = self.llm_healer.fix(
                error, state, stage_name,
                error_stack_trace=getattr(self, '_current_error_stack', None)
            )

            self.metrics.record_heal("llm-healer", stage_name, result.success)

            if result.success:
                self._notify_healers("llm-healer", result)

            # Clear failed healers for next error
            self.llm_healer.clear_failed_healers()

            return result

        except Exception as e:
            logger.error(f"LLM healer raised exception: {e}")
            self.metrics.errors_encountered.append(str(e))
            return None

    # Maximum number of unique errors to track (prevents memory growth)
    MAX_RECENT_ERRORS = 100

    def _should_attempt_heal(self, error: Exception, stage_name: str) -> bool:
        """Check for circular healing (same error repeating)."""
        error_key = f"{stage_name}:{type(error).__name__}:{str(error)[:50]}"

        with self._errors_lock:
            # Prevent unbounded memory growth - clear oldest entries when limit reached
            if len(self.recent_errors) >= self.MAX_RECENT_ERRORS:
                # Clear half the entries (oldest by insertion order in Python 3.7+)
                keys_to_remove = list(self.recent_errors.keys())[:self.MAX_RECENT_ERRORS // 2]
                for key in keys_to_remove:
                    del self.recent_errors[key]
                logger.debug(f"[orchestrator] Pruned recent_errors to {len(self.recent_errors)} entries")

            if error_key in self.recent_errors:
                count = self.recent_errors[error_key]
                if count >= 2:
                    if self.healing_logger:
                        self.healing_logger.log_fallback(
                            stage_name, "heal_loop", "abort",
                            f"Same error repeated {count} times, aborting"
                        )
                    return False
                self.recent_errors[error_key] = count + 1
            else:
                self.recent_errors[error_key] = 1

        return True

    def _notify_healers(self, source_healer: str, result: HealerResult):
        """
        Notify healers of changes made by another healer.

        Enables cross-healer coordination.
        """
        # Store in shared state
        self._healer_state[f"{source_healer}_last_result"] = result

        # Specific notifications
        if result.modified_config:
            # If config was modified, reset backoff states
            for healer in self.healers:
                if hasattr(healer, 'reset_backoff'):
                    healer.reset_backoff()

        # If disk healer cleaned cache, checkpoint healer needs to know
        if source_healer == "disk-healer":
            if "checkpoint-healer" in self._healer_instances:
                ch = self._healer_instances["checkpoint-healer"]
                if hasattr(ch, 'cache_was_cleaned'):
                    ch.cache_was_cleaned = True

    def _escalate_to_user(
        self,
        error: Exception,
        stage_name: str
    ) -> HealerResult:
        """Escalate issue to user for decision."""
        self.metrics.user_escalations += 1

        request = EscalationRequest(
            stage_name=stage_name,
            error=str(error),
            options=[
                "retry",      # Retry the stage
                "skip",       # Skip to next stage
                "abort",      # Stop pipeline
                "rollback",   # Rollback config and retry
            ],
            recommendation="retry" if "rate limit" in str(error).lower() else "abort",
        )

        # If callback is set, use it
        if self.escalation_callback:
            try:
                decision = self.escalation_callback(request)
                return self._handle_user_decision(decision, error, stage_name)
            except Exception:
                pass

        # Default: log and return failed
        logger.error(f"User escalation required for {stage_name}: {error}")
        print(f"\n{'='*60}")
        print(f"ESCALATION REQUIRED: {stage_name}")
        print(f"Error: {error}")
        print(f"Options: {request.options}")
        print(f"Recommendation: {request.recommendation}")
        print(f"{'='*60}\n")

        return HealerResult.failed(f"User escalation required: {error}")

    def _handle_user_decision(
        self,
        decision: str,
        error: Exception,
        stage_name: str
    ) -> HealerResult:
        """Handle user's decision from escalation."""
        if decision == "retry":
            return HealerResult.fixed("User requested retry", action=HealerAction.RETRY)
        elif decision == "skip":
            return HealerResult.fixed("User requested skip", action=HealerAction.SKIP)
        elif decision == "rollback":
            if self.rollback_config():
                return HealerResult.fixed("Config rolled back", action=HealerAction.RETRY)
            return HealerResult.failed("Rollback failed")
        else:  # abort
            return HealerResult.failed("User requested abort")

    # =========================================================================
    # METRICS AND REPORTING
    # =========================================================================

    def get_metrics(self) -> HealingMetrics:
        """Get current healing metrics."""
        return self.metrics

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get structured metrics for healing dashboard (US-64-005).

        Returns a dictionary containing:
        - success_rate: Overall heal success rate (0-100)
        - average_heal_time_ms: Average heal time in milliseconds
        - total_heals: Total number of healing attempts
        - successful_heals: Number of successful heals
        - failed_heals: Number of failed heals
        - healer_utilization: Dict mapping healer name to utilization metrics
        - error_categories: Dict mapping error category to count
        - time_spent_healing: Total time spent healing in seconds
        - preflight_issues: Dict with found and fixed counts
        - user_escalations: Number of user escalations
        - rollbacks: Number of rollbacks performed
        """
        healer_utilization = {}
        for healer_name, total_attempts in self.metrics.heals_by_healer.items():
            healer_utilization[healer_name] = {
                'total_attempts': total_attempts,
                'successful_heals': self.metrics.successful_heals_by_healer.get(healer_name, 0),
                'success_rate': self.metrics.get_healer_success_rate(healer_name),
                'average_time_ms': self.metrics.get_healer_average_time_ms(healer_name),
            }

        return {
            'success_rate': self.metrics.get_heal_success_rate(),
            'average_heal_time_ms': self.metrics.get_average_heal_time_ms(),
            'total_heals': self.metrics.total_heals,
            'successful_heals': self.metrics.successful_heals,
            'failed_heals': self.metrics.failed_heals,
            'healer_utilization': healer_utilization,
            'error_categories': dict(self.metrics.error_categories),
            'heals_by_stage': dict(self.metrics.heals_by_stage),
            'time_spent_healing': self.metrics.time_spent_healing,
            'preflight_issues': {
                'found': self.metrics.preflight_issues_found,
                'fixed': self.metrics.preflight_issues_fixed,
            },
            'user_escalations': self.metrics.user_escalations,
            'rollbacks': self.metrics.rollbacks_performed,
        }

    def format_dashboard(self) -> str:
        """Format healing dashboard for human-readable console output (US-64-005).

        Returns a formatted string suitable for printing to console.
        """
        metrics = self.get_dashboard_metrics()
        lines = []

        # Header
        lines.append("HEALING DASHBOARD")
        lines.append("-" * 50)

        # Overall stats
        success_rate = metrics['success_rate']
        avg_time = metrics['average_heal_time_ms']
        lines.append(f"Success Rate: {success_rate:.1f}%")
        lines.append(f"Average Heal Time: {avg_time:.0f}ms")
        lines.append(f"Total Heals: {metrics['total_heals']} "
                     f"({metrics['successful_heals']} successful, {metrics['failed_heals']} failed)")
        lines.append(f"Time Spent Healing: {metrics['time_spent_healing']:.1f}s")

        # Healer utilization breakdown
        if metrics['healer_utilization']:
            lines.append("")
            lines.append("Healer Utilization:")
            for healer_name, util in sorted(metrics['healer_utilization'].items()):
                rate = util['success_rate']
                avg_ms = util['average_time_ms']
                attempts = util['total_attempts']
                success = util['successful_heals']
                lines.append(f"  {healer_name}:")
                lines.append(f"    Attempts: {attempts} ({success} successful)")
                lines.append(f"    Success Rate: {rate:.1f}%")
                if avg_ms > 0:
                    lines.append(f"    Avg Time: {avg_ms:.0f}ms")

        # Error category distribution
        if metrics['error_categories']:
            lines.append("")
            lines.append("Error Categories:")
            sorted_cats = sorted(
                metrics['error_categories'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for category, count in sorted_cats:
                lines.append(f"  {category}: {count}")

        # Heals by stage
        if metrics['heals_by_stage']:
            lines.append("")
            lines.append("Heals by Stage:")
            for stage, count in sorted(metrics['heals_by_stage'].items()):
                lines.append(f"  {stage}: {count}")

        # Preflight issues
        preflight = metrics['preflight_issues']
        if preflight['found'] > 0:
            lines.append("")
            lines.append(f"Preflight Issues: {preflight['fixed']}/{preflight['found']} fixed")

        # Escalations and rollbacks
        if metrics['user_escalations'] > 0:
            lines.append(f"User Escalations: {metrics['user_escalations']}")
        if metrics['rollbacks'] > 0:
            lines.append(f"Rollbacks: {metrics['rollbacks']}")

        return "\n".join(lines)

    def set_rate_limit_metrics(self, metrics) -> None:
        """Set rate limit metrics from download stage for inclusion in report.

        Args:
            metrics: RateLimitMetrics instance from VideoDownloader
        """
        self._rate_limit_metrics = metrics

    def set_escalation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set escalation metrics from EscalationManager for inclusion in report.

        Args:
            metrics: Dict from EscalationManager.get_metrics()
        """
        self._escalation_metrics = metrics

    def set_aggregated_metrics(self, aggregator) -> None:
        """Set a RateLimitMetricsAggregator for unified reporting.

        Args:
            aggregator: RateLimitMetricsAggregator instance
        """
        self._aggregated_metrics = aggregator

    def wire_escalation_manager(self, escalation_manager) -> None:
        """Wire a shared EscalationManager into the DownloadHealer.

        Called by the pipeline after the download stage's VideoDownloader
        is initialized, so the healer uses the same escalation state
        instead of creating a duplicate CookieRotator.

        Args:
            escalation_manager: EscalationManager from VideoDownloader
        """
        healer = self._healer_instances.get('download-healer')
        if healer is not None:
            healer.escalation_manager = escalation_manager
            logger.info("Orchestrator: Wired shared EscalationManager into DownloadHealer")

        # US-35-002: Wire pre-instantiated MullvadVPN into EscalationManager
        if self._mullvad_vpn is not None and hasattr(escalation_manager, 'set_mullvad_vpn'):
            escalation_manager.set_mullvad_vpn(self._mullvad_vpn)
            logger.debug("Orchestrator: Wired MullvadVPN into EscalationManager for Tier 4 bypass")

    def set_mullvad_vpn(self, mullvad_vpn) -> None:
        """Store MullvadVPN instance for wiring into EscalationManager (US-35-002).

        Called by create_healing_pipeline() when config.download.mullvad.enabled=true.
        The stored MullvadVPN is wired into EscalationManager later when
        wire_escalation_manager() is called after stage downloader initialization.

        Args:
            mullvad_vpn: MullvadVPN instance from create_healing_pipeline()
        """
        self._mullvad_vpn = mullvad_vpn
        logger.debug("Orchestrator: MullvadVPN instance stored for Tier 4 bypass wiring")

    def get_vpn_status(self) -> Optional[Dict[str, Any]]:
        """Get current VPN status if MullvadVPN is available (US-35-006).

        Returns:
            Dict with VPN status info if MullvadVPN is configured, None otherwise.
            Dict keys:
                - connected: bool - whether VPN is currently connected
                - country: str or None - current connected country
                - rotation_count: int - number of VPN rotations performed
                - countries_used: List[str] - list of countries rotated through
                - last_verified_ip: str or None - last verified exit IP
        """
        if self._mullvad_vpn is None:
            return None

        try:
            # Get extended status from MullvadVPN
            extended = self._mullvad_vpn.get_status_extended()
            return {
                'connected': extended.get('mullvad_connected', False),
                'country': extended.get('current_country'),
                'rotation_count': extended.get('switch_count', 0),
                'countries_used': extended.get('used_countries', []),
                'last_verified_ip': extended.get('last_verified_ip'),
            }
        except Exception as e:
            logger.debug(f"Could not get VPN status: {e}")
            return None

    def print_report(self):
        """Print healing summary report."""
        print("\n" + "=" * 60)
        print("HEALING ORCHESTRATOR REPORT")
        print("=" * 60)

        print(f"\nStrategy: {self.strategy.mode.value}")
        print(f"Healers active: {len(self.healers)}")

        # US-64-005: Print healing dashboard if there was healing activity
        if self.metrics.total_heals > 0:
            print("\n" + "-" * 60)
            print(self.format_dashboard())
        else:
            print(f"\n{self.metrics.summary()}")

        if self.metrics.errors_encountered:
            print(f"\nErrors encountered: {len(self.metrics.errors_encountered)}")
            for err in self.metrics.errors_encountered[:5]:
                print(f"  - {err[:80]}...")

        # Print rate limiting section if metrics available (US-010)
        if self._rate_limit_metrics:
            print("\n" + "-" * 60)
            print("RATE LIMITING")
            print("-" * 60)
            print(self._rate_limit_metrics.summary())

            # Print config recommendations if rate limiting was significant
            recommendations = self._rate_limit_metrics.get_config_recommendations()
            if recommendations:
                print("\nRecommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")

        # Print aggregated metrics if available (US-004 Sprint 10)
        if self._aggregated_metrics is not None:
            try:
                agg = self._aggregated_metrics.aggregate()
                health = self._aggregated_metrics.get_health_status()

                print("\n" + "-" * 60)
                print(f"UNIFIED RATE-LIMIT STATUS: {health.upper()}")
                print("-" * 60)

                # Escalation summary from aggregated data
                esc = agg.get('escalation', {})
                if esc.get('total_escalations', 0) > 0 or esc.get('total_403s', 0) > 0:
                    print(f"Total 403/bot errors: {esc.get('total_403s', 0)}")
                    print(f"Total successes: {esc.get('total_successes', 0)}")
                    print(f"Total escalations: {esc.get('total_escalations', 0)}")
                    print(f"Average tier: {esc.get('average_tier', 1.0)}")

                    per_tier = esc.get('escalations_per_tier', {})
                    if per_tier:
                        tier_str = ", ".join(f"{k}: {v}" for k, v in per_tier.items())
                        print(f"Escalations by tier: {tier_str}")

                    kw_tiers = esc.get('keywords_at_each_tier', {})
                    if kw_tiers:
                        for tier_name, keywords in kw_tiers.items():
                            print(f"  {tier_name}: {len(keywords)} keywords")

                # Trigger category breakdown
                triggers = agg.get('trigger_categories', {})
                if triggers:
                    trig_str = ", ".join(f"{k}: {v}" for k, v in sorted(triggers.items()))
                    print(f"Trigger categories: {trig_str}")

                # Budget summary
                budget = agg.get('budget', {})
                if budget:
                    print(f"Budget: rotations {budget.get('rotations_used', 0)}/{budget.get('max_rotations', '?')}, "
                          f"backoff {budget.get('backoff_time_spent', 0):.0f}s/{budget.get('max_backoff_time', '?')}s")

                # Circuit breaker
                cb = agg.get('circuit_breaker', {})
                if cb.get('total_trips', 0) > 0:
                    print(f"Circuit breaker: {cb['total_trips']} trips, "
                          f"{cb.get('total_paused_seconds', 0):.0f}s paused")
            except Exception:
                pass  # Non-critical

        # Fallback: print escalation metrics if no aggregator (US-008 Sprint 9)
        elif self._escalation_metrics:
            m = self._escalation_metrics
            if m.get('total_escalations', 0) > 0 or m.get('total_403s', 0) > 0:
                print("\n" + "-" * 60)
                print("BYPASS ESCALATION")
                print("-" * 60)
                print(f"Total 403/bot errors: {m.get('total_403s', 0)}")
                print(f"Total successes: {m.get('total_successes', 0)}")
                print(f"Total escalations: {m.get('total_escalations', 0)}")
                print(f"Average tier: {m.get('average_tier', 1.0)}")

                per_tier = m.get('escalations_per_tier', {})
                if per_tier:
                    tier_str = ", ".join(f"{k}: {v}" for k, v in per_tier.items())
                    print(f"Escalations by tier: {tier_str}")

                kw_tiers = m.get('keywords_at_each_tier', {})
                if kw_tiers:
                    for tier_name, keywords in kw_tiers.items():
                        print(f"  {tier_name}: {len(keywords)} keywords")

                # US-1-012: VPN rotation metrics
                vpn_rotations = m.get('vpn_rotation_count', 0)
                vpn_countries = m.get('vpn_countries_used', [])
                if vpn_rotations > 0 or vpn_countries:
                    print(f"\nVPN Rotation:")
                    print(f"  Total rotations: {vpn_rotations}")
                    if vpn_countries:
                        print(f"  Countries used: {', '.join(vpn_countries)}")

        # US-1-012: Tier-by-tier escalation breakdown
        self._print_tier_breakdown()

        # US-35-006: VPN status section when VPN was used
        self._print_vpn_status()

        print("=" * 60 + "\n")

    def _print_tier_breakdown(self) -> None:
        """Print detailed tier-by-tier escalation breakdown (US-1-012).

        Shows which tiers were reached and how often, helping users understand
        escalation patterns and effectiveness.
        """
        # Try aggregated metrics first, then escalation metrics
        esc_data = None
        if self._aggregated_metrics is not None:
            try:
                agg = self._aggregated_metrics.aggregate()
                esc_data = agg.get('escalation', {})
            except Exception:
                pass

        if esc_data is None and self._escalation_metrics:
            esc_data = self._escalation_metrics

        if not esc_data:
            return

        per_tier = esc_data.get('escalations_per_tier', {})
        kw_tiers = esc_data.get('keywords_at_each_tier', {})

        # Only print if there's meaningful data
        if not per_tier and not kw_tiers:
            return

        # Define tier order for consistent display
        tier_order = ['IMPERSONATE_ONLY', 'EXTRACTOR_ARGS', 'FULL_BYPASS', 'VPN_ROTATION']
        tier_labels = {
            'IMPERSONATE_ONLY': 'Tier 1 (Impersonate)',
            'EXTRACTOR_ARGS': 'Tier 2 (Extractor Args)',
            'FULL_BYPASS': 'Tier 3 (Full Bypass)',
            'VPN_ROTATION': 'Tier 4 (VPN Rotation)',
        }

        print("\n" + "-" * 60)
        print("TIER-BY-TIER BREAKDOWN")
        print("-" * 60)

        for tier_name in tier_order:
            label = tier_labels.get(tier_name, tier_name)
            escalations = per_tier.get(tier_name, 0)
            keywords = kw_tiers.get(tier_name, [])
            keyword_count = len(keywords) if keywords else 0

            if escalations > 0 or keyword_count > 0:
                print(f"{label}:")
                if escalations > 0:
                    print(f"  Escalations to this tier: {escalations}")
                if keyword_count > 0:
                    print(f"  Keywords currently at this tier: {keyword_count}")
                    # Show first 5 keywords if any
                    if keyword_count <= 5:
                        print(f"    {', '.join(keywords)}")
                    else:
                        print(f"    {', '.join(keywords[:5])}... (+{keyword_count - 5} more)")

    def _print_vpn_status(self) -> None:
        """Print VPN status section when VPN was used during the run (US-35-006).

        Only prints if MullvadVPN is configured and was used (rotation_count > 0).
        """
        vpn_status = self.get_vpn_status()
        if vpn_status is None:
            return

        rotation_count = vpn_status.get('rotation_count', 0)
        countries_used = vpn_status.get('countries_used', [])

        # Only print section if VPN was actually used
        if rotation_count == 0 and not countries_used:
            return

        print("\n" + "-" * 60)
        print("VPN STATUS")
        print("-" * 60)

        connected = vpn_status.get('connected', False)
        current_country = vpn_status.get('country')

        print(f"Connected: {'Yes' if connected else 'No'}")
        if current_country:
            print(f"Current country: {current_country.upper()}")

        print(f"Total rotations: {rotation_count}")

        if countries_used:
            countries_str = ', '.join(c.upper() for c in countries_used)
            print(f"Countries used: {countries_str}")

        last_ip = vpn_status.get('last_verified_ip')
        if last_ip:
            print(f"Last verified IP: {last_ip}")

    def reset(self):
        """Reset orchestrator state for new run."""
        self.metrics = HealingMetrics()
        self.config_snapshots = []
        self.current_stage = None
        self._healer_state = {}
        self._rate_limit_metrics = None
        self._escalation_metrics = None
        self._aggregated_metrics = None

        # Reset healers
        for healer in self.healers:
            if hasattr(healer, 'reset_backoff'):
                healer.reset_backoff()


def create_orchestrated_pipeline(
    config: 'Config',
    project_dir: Path,
    strategy: HealingStrategy = None,
) -> Tuple['PipelineOrchestrator', 'HealingOrchestrator', 'ResilientRunner']:
    """
    Create a fully orchestrated pipeline with healing.

    Returns:
        Tuple of (pipeline, healing_orchestrator, runner)
    """
    from ..pipeline import create_default_pipeline
    from .runner import ResilientRunner

    # Create components
    pipeline = create_default_pipeline(config, project_dir)
    orchestrator = HealingOrchestrator(config, project_dir, strategy)
    runner = ResilientRunner(config, project_dir)

    # Connect runner to orchestrator
    runner.orchestrator = orchestrator

    return pipeline, orchestrator, runner
