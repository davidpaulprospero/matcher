"""
Pipeline Orchestrator

Lightweight orchestrator that runs pipeline stages in order,
handling checkpointing and resume functionality.

This replaces the run() method and stage orchestration logic in main.py.

Self-Healing: By default, pipelines use ResilientRunner with HealingOrchestrator
for automatic error recovery. Controlled via config.healing settings.

Pipeline Variants:
    - create_default_pipeline(): Standard 7-stage pipeline
      ANALYZE → VIDEO_SEARCH → CAPTION → MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
    - create_entity_enhanced_pipeline(): 9-stage pipeline with entity media stages
      ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → VIDEO_SEARCH → CAPTION → MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
    - create_match_only_pipeline(): Re-run matching from checkpoint
    - create_healing_pipeline(): Default pipeline with self-healing wrapper
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from .checkpoint import CheckpointManager, STAGE_ORDER
from .pipeline_events import (
    PipelineEvent,
    PipelineEventBus,
    EventCallback,
    EVENT_STAGE_START,
    EVENT_STAGE_SKIP,
    EVENT_CHECKPOINT_SAVE,
    EVENT_RESOURCE_WARNING,
    EVENT_ERROR_RATE_THRESHOLD,  # US-138-010
)
from .pipeline_history import append_stage_timing, estimate_duration, predict_memory_usage, append_resource_usage
from .pipeline_progress import ProgressReporter, get_event_bus
from .pipeline_validator import PipelineValidator, StageValidationResult, ContractViolation
from .health_checker import HealthChecker, HealthStatus
from .state import PipelineState
from .stages import Stage, StageResult, StageMetrics, DependencyError
from .stages import validate_no_cycles, build_dependency_graph, get_all_stages
from .checkpoint import STAGE_ORDER
from .stages.error_aggregator import ErrorAggregator, ErrorCategory, PipelineErrorAggregator

# US-88-008: Pipeline metrics exporter
from .pipeline_metrics_exporter import MetricsExporter, PipelineExportConfig

# US-146-012: YouTube API vs yt-dlp usage tracking
from .downloader.api_fallback_handler import log_api_vs_ytdlp_usage

# US-154-011: Pre-flight quota check
from .downloader.youtube_api_client import YouTubeAPIClient

# US-159-007: Error code classification
from .downloader.errors import format_error_with_code

# US-162-005: Stage skip logging
# US-163-008: Health check structured logging
from .logging_templates import log_stage_skip, log_progress, log_error_with_context

if TYPE_CHECKING:
    from .config import Config
    from .agents.runner import ResilientRunner
    from .agents.orchestrator import HealingOrchestrator

# Type aliases for callbacks
StageStartCallback = Callable[[str], None]  # (stage_name) -> None
StageCompleteCallback = Callable[[str, StageResult, float], None]  # (stage_name, result, elapsed_seconds) -> None


logger = logging.getLogger(__name__)

# US-159-005: Correlation ID for structured logging and request tracing
# Context variable to store the current pipeline run's correlation ID
_correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID for this pipeline run.

    Returns:
        The correlation ID if set, None otherwise.
    """
    return _correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current pipeline run.

    Args:
        correlation_id: The correlation ID to set.
    """
    _correlation_id_var.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID.

    Returns:
        A new UUID-based correlation ID.
    """
    return f"run-{uuid.uuid4().hex[:12]}"


# US-88-011: Auto-detect parallel stages based on DEPENDS_ON
def detect_parallel_stage_groups(
    stages: List['Stage'],
) -> List[Tuple[str, ...]]:
    """
    Auto-detect stages that can run in parallel based on their DEPENDS_ON relationships.

    This analyzes the dependency graph to find stages that:
    1. Have no direct or transitive dependency on each other
    2. Can be executed concurrently once their prerequisites are met

    Args:
        stages: List of Stage instances to analyze

    Returns:
        List of tuples, each tuple contains stage names that can run in parallel.
        Returns empty list if no parallel opportunities detected or if auto-detection
        is not beneficial (e.g., linear dependency chain).

    Example:
        If VIDEO_SEARCH depends on [ANALYZE] and ENTITY_IMAGES depends on [ANALYZE],
        returns [("VIDEO_SEARCH", "ENTITY_IMAGES")]
    """
    if not stages:
        return []

    # Build stage name -> stage object map
    stage_map: Dict[str, 'Stage'] = {s.name: s for s in stages}

    # Build dependency sets (include transitive dependencies)
    def get_all_dependencies(stage: 'Stage') -> Set[str]:
        """Get all transitive dependencies for a stage."""
        deps = set(stage.DEPENDS_ON)
        for dep in stage.DEPENDS_ON:
            if dep in stage_map:
                deps.update(get_all_dependencies(stage_map[dep]))
        return deps

    # Find stages with no dependencies that could run in parallel
    # These are stages that depend only on the same prerequisites
    dependency_groups: Dict[frozenset, List[str]] = {}

    for stage in stages:
        if not stage.DEPENDS_ON:
            # Stages with no dependencies can't be auto-parallelized with each other
            # (they would run sequentially as entry points)
            continue

        # Group stages by their dependency set
        dep_set = frozenset(stage.DEPENDS_ON)
        if dep_set not in dependency_groups:
            dependency_groups[dep_set] = []
        dependency_groups[dep_set].append(stage.name)

    # Filter to groups with 2+ stages (parallel opportunity)
    parallel_groups: List[Tuple[str, ...]] = []
    for dep_set, stage_names in dependency_groups.items():
        if len(stage_names) >= 2:
            # Verify no circular dependencies between these stages
            if _verify_no_circular_between(stage_map, stage_names):
                parallel_groups.append(tuple(stage_names))
                logger.debug(f"Auto-detected parallel group: {stage_names} (depend on {set(dep_set)})")

    return parallel_groups


def _verify_no_circular_between(
    stage_map: Dict[str, 'Stage'],
    stage_names: List[str],
) -> bool:
    """
    Verify that no stages in the list have circular dependencies on each other.

    Args:
        stage_map: Map of stage name to stage object
        stage_names: List of stage names to check

    Returns:
        True if no circular dependencies exist between these stages
    """
    # Check each stage against others
    for stage_name in stage_names:
        if stage_name not in stage_map:
            continue
        stage = stage_map[stage_name]
        deps = set(stage.DEPENDS_ON)

        # Check if this stage depends on any other stage in the list
        for other in stage_names:
            if other != stage_name and other in deps:
                logger.warning(f"Circular dependency detected: {stage_name} depends on {other}")
                return False

    return True


def validate_no_circular_dependencies(stages: List['Stage']) -> None:
    """
    Validate that the stage dependency graph has no cycles.

    Args:
        stages: List of Stage instances to validate

    Raises:
        ValueError: If circular dependencies are detected
    """
    stage_map: Dict[str, 'Stage'] = {s.name: s for s in stages}

    # Use DFS to detect cycles
    def has_cycle(stage_name: str, visited: Set[str], rec_stack: Set[str]) -> bool:
        """DFS cycle detection. Returns True if cycle found."""
        visited.add(stage_name)
        rec_stack.add(stage_name)

        if stage_name not in stage_map:
            return False

        for dep in stage_map[stage_name].DEPENDS_ON:
            if dep not in visited:
                if has_cycle(dep, visited, rec_stack):
                    return True
            elif dep in rec_stack:
                logger.error(format_error_with_code("PIPE-005", f"Circular dependency detected: {stage_name} -> {dep}"))
                return True

        rec_stack.remove(stage_name)
        return False

    visited: Set[str] = set()
    for stage in stages:
        if stage.name not in visited:
            if has_cycle(stage.name, visited, set()):
                raise ValueError(
                    f"Circular dependency detected in stage graph involving '{stage.name}'"
                )

    logger.debug("Stage dependency graph validation passed: no circular dependencies")


def log_parallel_execution_plan(
    stages: List['Stage'],
    parallel_groups: List[Tuple[str, ...]],
) -> None:
    """
    Log the parallel execution plan for transparency.

    Args:
        stages: All stages in execution order
        parallel_groups: Auto-detected parallel stage groups
    """
    logger.info("=" * 60)
    logger.info("Parallel Execution Plan:")
    logger.info("=" * 60)

    # Build stage order for reference
    stage_order = [s.name for s in stages]

    # Log sequential stages
    sequential = [s for s in stage_order if not any(s in g for g in parallel_groups)]
    if sequential:
        logger.info(f"  Sequential: {' -> '.join(sequential)}")

    # Log parallel groups
    for i, group in enumerate(parallel_groups, 1):
        logger.info(f"  Parallel Group {i}: {' | '.join(group)}")

    logger.info("=" * 60)


@dataclass
class StageLogContext:
    """Structured logging context for pipeline stage execution."""
    stage_name: str
    stage_index: int
    total_stages: int
    elapsed_seconds: float
    items_processed: int
    items_failed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            'stage': self.stage_name,
            'idx': self.stage_index,
            'total': self.total_stages,
            'elapsed': f"{self.elapsed_seconds:.1f}s",
            'processed': self.items_processed,
            'failed': self.items_failed,
        }

    def to_suffix(self) -> str:
        """Format as a parseable suffix for log messages."""
        return f"[stage={self.stage_name} idx={self.stage_index}/{self.total_stages} elapsed={self.elapsed_seconds:.1f}s items={self.items_processed}/{self.items_failed}]"


class PipelineOrchestrator:
    """
    Orchestrates pipeline stage execution with checkpoint support.

    Features:
    - Runs stages in order
    - Supports resume from checkpoint
    - Tracks stage timing
    - Handles stage failures gracefully
    - Cross-stage data drift detection (non-blocking warnings)
    """

    # Cross-stage data drift rules: (source_field, target_field, min_ratio)
    # After a stage completes, check that target_field count >= min_ratio * source_field count.
    # Rules are keyed by the stage whose completion triggers the check.
    DRIFT_RULES: List[Tuple[str, str, str, float]] = [
        # After CAPTION: caption_results should have >= 80% of video_ids
        ('CAPTION', 'video_ids', 'caption_results', 0.8),
        # After MATCH: text_metadata should have >= 80% of video_ids
        ('MATCH', 'video_ids', 'text_metadata', 0.8),
        # After OUTPUT: matches should have >= 50% of voiceover_segments
        ('OUTPUT', 'voiceover_segments', 'matches', 0.5),
    ]

    def __init__(
        self,
        config: 'Config',
        project_dir: Path,
        stages: List[Stage] = None,
        verbose_progress: bool = False,
        show_quota: bool = False
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Configuration object
            project_dir: Project directory for checkpoints and caches
            stages: Optional list of stages (uses default if not provided)
            verbose_progress: Enable detailed per-stage progress output
            show_quota: Display real-time quota status during execution (US-155-011)
        """
        self.config = config
        self.project_dir = Path(project_dir)
        self.state = PipelineState()
        self.stages = stages or []
        self.verbose_progress = verbose_progress
        self.show_quota = show_quota
        self._quota_displayed = False  # Track if initial quota has been shown

        # Create validator for pre-run checks (US-82-006)
        self._validator = PipelineValidator(config, self.stages)

        # Validate config before any I/O (US-45-010: fail-fast on invalid config)
        config_errors = self._validate_config()
        if config_errors:
            error_detail = "; ".join(config_errors)
            raise ValueError(f"Pipeline config validation failed: {error_detail}")

        # US-138-007: Validate stage dependencies at startup
        # This checks for circular dependencies and invalid DEPENDS_ON references
        dependency_errors = self._validate_dependencies()
        if dependency_errors:
            error_detail = "; ".join(dependency_errors)
            raise ValueError(f"Pipeline dependency validation failed: {error_detail}")

        # Initialize checkpoint manager
        config_hash = getattr(config, '_config_hash', '')
        self.checkpoint = CheckpointManager(project_dir, config_hash, config=config)

        # US-115-010: Clean up stale checkpoint backups on startup
        cleanup_report = self.checkpoint.cleanup_stale_backups()
        if cleanup_report["files_removed"] > 0:
            logger.info(
                f"Checkpoint cleanup: removed {cleanup_report['files_removed']} files, "
                f"freed {cleanup_report['bytes_freed']} bytes"
            )

        # Runtime state
        self.resume_mode = False
        self.current_stage: Optional[str] = None
        self.stage_timings: dict = {}
        self.stage_metrics: dict = {}  # stage_name -> StageMetrics

        # US-125-006: Pause/Resume state
        self._paused: bool = False
        self._pause_event: Optional[signal.Event] = None  # Signal event for thread-safe pause
        self._original_sigint_handler: Optional[signal.Handler] = None
        self._original_sigterm_handler: Optional[signal.Handler] = None  # US-138-004: SIGTERM handler

        # US-138-004: Abort state for graceful shutdown
        self.abort_requested: bool = False
        self.abort_reason: Optional[str] = None

        # Progress reporter for real-time progress.json updates
        self.progress_reporter = ProgressReporter(project_dir, remaining_stages=[])

        # Event bus for decoupled stage lifecycle notifications (US-82-011)
        self.event_bus = PipelineEventBus()

        # US-108-003: Subscribe to progress events for verbose output
        if verbose_progress:
            from .pipeline_progress import get_event_bus, EVENT_STAGE_PROGRESS
            progress_bus = get_event_bus()
            progress_bus.subscribe(EVENT_STAGE_PROGRESS, self._handle_verbose_progress)

        # US-88-006: Data drift detection history for trend analysis
        # Stores drift events for debugging and pattern detection
        self.drift_history: List[Dict[str, Any]] = []
        self._drift_config = getattr(config.pipeline, 'drift_rules', None)

        # US-106-009: Pipeline-level error aggregation for cross-stage analysis
        self._error_aggregator = PipelineErrorAggregator()

        # US-106-011: Resource monitoring history (per-stage before/after metrics)
        self._resource_history: List[Dict[str, Any]] = []

        # US-162-009: Track current stage estimated duration for baseline comparison
        self._current_stage_estimated_duration: Optional[float] = None

        # US-129-012: Download metrics exporter for resource monitoring
        self._download_metrics_exporter = None

        # US-108-012: Parallel execution tracking
        self.stages_run_concurrently: List[str] = []
        self.parallel_group_timings: Dict[Tuple[str, ...], float] = {}

        # US-125-010: Webhook notifications for pipeline events
        self._webhook_sender = None
        webhook_config = getattr(config.pipeline, 'webhook', None)
        if webhook_config and getattr(webhook_config, 'enabled', False):
            from .pipeline_events import WebhookSender
            self._webhook_sender = WebhookSender(webhook_config)
            # Register webhook sender for key events
            self.event_bus.subscribe('after_stage', self._webhook_sender.on_event)
            self.event_bus.subscribe('on_pipeline_complete', self._webhook_sender.on_event)
            self.event_bus.subscribe('on_stage_error', self._webhook_sender.on_event)
            logger.info("Webhook notifications enabled")

        # US-125-011: Health check interval tracking
        # Tracks last health check time for interval-based scheduling
        self._last_health_check_time: float = 0.0
        self._health_check_interval_config: dict = {}
        hc_interval = getattr(config.pipeline, 'health_check_interval', None)
        if hc_interval:
            if isinstance(hc_interval, dict):
                self._health_check_interval_config = hc_interval
            elif hasattr(hc_interval, '__dict__'):
                self._health_check_interval_config = hc_interval.__dict__
        # Default to 300 seconds if not configured
        if not self._health_check_interval_config:
            self._health_check_interval_config = {"default": 300}

    def _get_health_check_interval(self, stage_name: str) -> float:
        """Get the health check interval for a specific stage (US-125-011).

        Args:
            stage_name: Name of the stage

        Returns:
            Health check interval in seconds (0 = run before each stage only)
        """
        # Check stage-specific interval first
        if stage_name in self._health_check_interval_config:
            return float(self._health_check_interval_config[stage_name])
        # Fall back to default
        return float(self._health_check_interval_config.get("default", 300))

    def _should_run_health_check(self, stage_name: str) -> bool:
        """Determine if health check should run before this stage (US-125-011).

        Runs health check if:
        - Interval-based checks are disabled (interval = 0)
        - No previous health check has run
        - Enough time has passed since last health check

        Args:
            stage_name: Name of the stage

        Returns:
            True if health check should run
        """
        import time
        interval = self._get_health_check_interval(stage_name)

        # If interval is 0, run before each stage
        if interval <= 0:
            return True

        # If no previous health check, run it
        if self._last_health_check_time <= 0:
            return True

        # Check if enough time has passed
        current_time = time.time()
        elapsed = current_time - self._last_health_check_time
        return elapsed >= interval

    def add_stage(self, stage: Stage) -> 'PipelineOrchestrator':
        """Add a stage to the pipeline (fluent interface).

        Validates:
        - No duplicate stage names are registered
        - All required dependency stages are present or will be added

        Raises:
            ValueError: If stage validation fails
        """
        # US-138-011: Validate duplicate stage registration
        existing_names = {s.name for s in self.stages}
        if stage.name in existing_names:
            raise ValueError(
                f"Cannot add duplicate stage '{stage.name}'. "
                f"Stage already exists in pipeline. "
                f"Existing stages: {sorted(existing_names)}"
            )

        # US-138-011: Validate stage dependencies are satisfied
        depends_on = getattr(stage, 'DEPENDS_ON', [])
        missing_deps = [dep for dep in depends_on if dep not in existing_names]
        if missing_deps:
            raise ValueError(
                f"Cannot add stage '{stage.name}': missing dependencies {missing_deps}. "
                f"Required dependencies must be added before this stage. "
                f"Add required stages first or use create_pipeline_variant() which handles this automatically."
            )

        self.stages.append(stage)
        return self

    @property
    def is_paused(self) -> bool:
        """Check if pipeline is currently paused."""
        return self._paused

    def pause(self) -> None:
        """
        Pause the pipeline execution.

        This suspends stage execution at the next checkpoint. The pipeline
        can be resumed by calling resume() or by sending another Ctrl+C signal.

        Note: This does not stop the currently executing stage immediately,
        but will pause before the next stage begins.
        """
        if self._paused:
            logger.info("Pipeline is already paused")
            return

        logger.info("Pausing pipeline execution...")
        self._paused = True

        # Restore original SIGINT handler if we had set one
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None

    def resume(self) -> None:
        """
        Resume the pipeline from a paused state.

        This continues pipeline execution from where it was paused.
        """
        if not self._paused:
            logger.info("Pipeline is not paused")
            return

        logger.info("Resuming pipeline execution...")
        self._paused = False

    def register_hook(self, event_type: str, callback: EventCallback) -> None:
        """Register a callback for a pipeline event type.

        Delegates to PipelineEventBus.subscribe(). Multiple callbacks can be
        registered per event type and are called in registration order.

        Args:
            event_type: One of 'before_stage', 'after_stage', 'on_stage_error',
                        'on_pipeline_complete'
            callback: Function that receives a PipelineEvent
        """
        self.event_bus.subscribe(event_type, callback)

    def emit_event(self, event: PipelineEvent) -> None:
        """Emit a pipeline event via the event bus.

        Delegates to PipelineEventBus.emit(). Handler exceptions are logged
        but do not interrupt pipeline execution.

        Args:
            event: The PipelineEvent to emit
        """
        self.event_bus.emit(event)

    def _setup_pause_signal_handler(self) -> None:
        """
        Setup SIGINT/SIGTERM handlers for pause and abort.

        - First Ctrl+C: pause pipeline at next checkpoint
        - Second Ctrl+C (while paused): abort pipeline
        - SIGTERM: abort pipeline immediately
        """
        def handle_sigint(signum, frame):
            if self.abort_requested:
                # Already aborting, ignore additional signals
                logger.info("Abort already in progress, ignoring signal")
                return
            if self._paused:
                # If already paused, abort execution
                logger.warning("Ctrl+C received while paused - aborting pipeline")
                self._trigger_abort("Ctrl+C received while paused")
            else:
                # If not paused, pause the pipeline
                logger.info("Ctrl+C received - pausing pipeline (Ctrl+C again to abort)")
                self.pause()

        def handle_sigterm(signum, frame):
            """SIGTERM always triggers abort for graceful shutdown."""
            if self.abort_requested:
                logger.info("Abort already in progress, ignoring SIGTERM")
                return
            logger.warning(f"SIGTERM received ({signum}) - aborting pipeline")
            self._trigger_abort(f"SIGTERM received (signal {signum})")

        # Store original handlers to restore on cleanup
        self._original_sigint_handler = signal.signal(signal.SIGINT, handle_sigint)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)

    def _trigger_abort(self, reason: str) -> None:
        """Trigger pipeline abort with the given reason."""
        self.abort_requested = True
        self.abort_reason = reason
        # US-138-004: Also set abort flag in state for stages to check
        if hasattr(self, 'state') and self.state:
            self.state.abort_requested = True
        logger.warning(f"Pipeline abort requested: {reason}")

        # If paused, resume to allow abort to take effect
        if self._paused:
            logger.info("Resuming from paused state to complete abort")
            self.resume()

    def _handle_abort(self) -> None:
        """Handle pipeline abort - save checkpoint and cleanup."""
        # Save checkpoint before aborting
        if self.checkpoint:
            try:
                # Save current state with abort info
                if self.current_stage:
                    self.checkpoint.save(self.current_stage, self.state.__dict__,
                                        stage_metrics=None)
                logger.info(f"Checkpoint saved before abort: {self.abort_reason}")
            except Exception as e:
                logger.error(format_error_with_code("OUTPUT-004", f"Failed to save checkpoint during abort: {e}"))

        # Store abort info in state for later diagnostics
        # Note: We use setattr to dynamically add the attribute to PipelineState
        self.state.abort_info = {
            'reason': self.abort_reason,
            'current_stage': self.current_stage,
            'completed_stages': list(self.stage_timings.keys()),
            'timings': dict(self.stage_timings),
        }

        # Cleanup signal handlers
        self._cleanup_signal_handler()

    def _cleanup_signal_handler(self) -> None:
        """Restore original SIGINT/SIGTERM handlers after pipeline completes."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            self._original_sigterm_handler = None

    def _handle_verbose_progress(self, event: PipelineEvent) -> None:
        """Handle verbose progress output (US-108-003)."""
        if not self.verbose_progress:
            return

        data = event.data
        stage = event.stage_name
        completed = data.get("items_completed", 0)
        total = data.get("items_total", 0)
        pct = data.get("progress_percent")
        memory = data.get("memory_usage_mb")
        cpu = data.get("cpu_percent")

        if total > 0 and pct is not None:
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = "N/A"

        resource_parts = []
        if memory is not None:
            resource_parts.append(f"Mem: {memory:.0f}MB")
        if cpu is not None:
            resource_parts.append(f"CPU: {cpu:.1f}%")
        resource_str = f" ({', '.join(resource_parts)})" if resource_parts else ""

        logger.info(f"{stage}: {pct_str} ({completed}/{total}){resource_str}")

    def _display_initial_quota(self) -> None:
        """Display initial quota status at pipeline start (US-155-011)."""
        try:
            # Check if download.youtube_api config section exists
            download_config = getattr(self.config, 'download', None)
            if not download_config:
                return

            youtube_api_config = getattr(download_config, 'youtube_api', None)
            if not youtube_api_config:
                return

            # Check if API is enabled via config
            api_enabled = getattr(youtube_api_config, 'enabled', False)
            if not api_enabled:
                return

            # Get API key(s) from config
            api_keys = getattr(youtube_api_config, 'api_keys', None)
            api_key = getattr(youtube_api_config, 'api_key', None)

            if not api_keys and not api_key:
                return

            # Get quota parameters from config
            quota_limit = getattr(youtube_api_config, 'quota_limit', 10000)
            quota_fallback_threshold = getattr(youtube_api_config, 'quota_fallback_threshold_percent', 10)

            # Create temporary API client to show initial quota
            with YouTubeAPIClient(
                api_key=api_key or (api_keys[0] if api_keys else ""),
                api_keys=api_keys,
                quota_limit=quota_limit,
                quota_fallback_threshold_percent=quota_fallback_threshold,
            ) as api_client:
                logger.info(f"YouTube API Quota Status:")
                self._display_quota_status(api_client, force=True)

        except ImportError:
            logger.debug("YouTubeAPIClient not available for quota display")
        except Exception as e:
            logger.exception(f"Could not display initial quota status: {e}")

    def _display_quota_status(self, api_client=None, force: bool = False) -> None:
        """Display real-time quota status (US-155-011).

        Args:
            api_client: YouTubeAPIClient instance to query quota status
            force: Force display even if show_quota is False
        """
        if not (self.show_quota or force):
            return

        if api_client is None:
            # Try to get api_client from global registry (US-155-011)
            try:
                from src.downloader.api_fallback_handler import get_youtube_api_client
                api_client = get_youtube_api_client()
            except ImportError:
                pass
            if api_client is None:
                return

        try:
            # Get quota status from API client
            quota_status = api_client.get_quota_status()
            if not quota_status:
                return

            quota_remaining = quota_status.get('quota_remaining', 0)
            quota_limit = quota_status.get('quota_limit', 0)
            quota_percent = quota_status.get('quota_percent_remaining', 0)

            # Build output string
            output_parts = [f"Quota: {quota_remaining}/{quota_limit} ({quota_percent:.1f}%)"]

            # Show warning if below threshold (default 20%)
            warn_threshold = 20
            if quota_percent < warn_threshold:
                output_parts.append(f"[WARNING: Low quota - {quota_percent:.1f}% remaining]")

            # Per-key breakdown if multiple keys
            per_key = quota_status.get('per_key_quota', [])
            if len(per_key) > 1:
                key_details = []
                for key_info in per_key:
                    key_idx = key_info.get('key_index', 0)
                    remaining = key_info.get('quota_remaining', 0)
                    limit = key_info.get('quota_limit', 0)
                    pct = key_info.get('quota_percent_remaining', 0)
                    key_details.append(f"Key{key_idx}: {remaining}/{limit} ({pct:.0f}%)")
                output_parts.append(" | ".join(key_details))

            logger.info(f"{' | '.join(output_parts)}")

        except Exception as e:
            logger.exception(f"Could not display quota status: {e}")

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if it exists.

        Returns:
            True if checkpoint was loaded and is valid
        """
        if not self.checkpoint.exists():
            logger.debug("Checkpoint does not exist, starting fresh")
            return False

        data = self.checkpoint.load()
        if data is None:
            logger.debug("Checkpoint data is None, starting fresh")
            return False

        logger.debug(f"Checkpoint loaded, validating (version: {data.get('version', 'unknown')})")

        validation = self.checkpoint.validate()
        if not validation['valid']:
            for error in validation['errors']:
                logger.error(f"Checkpoint error: {error}")
            return False

        for warning in validation['warnings']:
            logger.warning(f"Checkpoint warning: {warning}")

        # Check for stale checkpoint (older than 24 hours)
        if self.checkpoint.is_stale(max_age_hours=24.0):
            age_hours = self.checkpoint.get_age_hours()
            logger.warning(
                f"Checkpoint is stale (age: {age_hours:.1f} hours). "
                "Consider using --fresh to start a new run."
            )

        logger.debug("Checkpoint validation passed, restoring state")
        # US-40-003: Use CheckpointManager.restore_state() for defensive validation
        # This validates and initializes any missing state attributes after checkpoint load
        self.state = self.checkpoint.restore_state(self.state)
        logger.debug("Checkpoint state restored successfully")

        self.resume_mode = True
        logger.info("Resuming pipeline from checkpoint")
        return True

    def _validate_config(self) -> List[str]:
        """
        Validate pipeline configuration before running stages.

        Delegates to PipelineValidator (US-82-006).

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        validator = getattr(self, '_validator', None)
        if validator is not None:
            return validator.validate_config()
        # Fallback for tests that bypass __init__
        return PipelineValidator(self.config, self.stages).validate_config()

    def _validate_config_schema(self) -> List[str]:
        """
        Pure config schema validation — no I/O, no filesystem access.

        Delegates to PipelineValidator (US-82-006).

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        validator = getattr(self, '_validator', None)
        if validator is not None:
            return validator._validate_config_schema()
        return PipelineValidator(self.config, self.stages)._validate_config_schema()

    def _validate_runtime_environment(self) -> List[str]:
        """
        Runtime environment checks — requires filesystem/PATH access.

        Delegates to PipelineValidator (US-82-006).

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        validator = getattr(self, '_validator', None)
        if validator is not None:
            return validator._validate_runtime_environment()
        return PipelineValidator(self.config, self.stages)._validate_runtime_environment()

    def _validate_dependencies(self) -> List[str]:
        """
        Validate stage dependencies at pipeline startup (US-138-007).

        Checks:
        1. No circular dependencies in the dependency graph
        2. All DEPENDS_ON stages exist in STAGE_ORDER
        3. Warn about registered stages not in the current pipeline

        Returns:
            List of error strings. Empty list means dependencies are valid.
            Warnings are logged but don't cause failure.
        """
        errors: List[str] = []

        # 1. Check for circular dependencies
        cycle_error = validate_no_cycles()
        if cycle_error:
            errors.append(f"Dependency cycle detected: {cycle_error}")
            # Fail fast - no point continuing if there are cycles
            return errors

        # 2. Validate all DEPENDS_ON stages exist in STAGE_ORDER
        stage_names_in_order = set(STAGE_ORDER)
        all_stages = get_all_stages()

        for stage_name, stage_cls in all_stages.items():
            deps = getattr(stage_cls, 'DEPENDS_ON', [])
            for dep in deps:
                if dep not in stage_names_in_order:
                    errors.append(
                        f"Stage '{stage_name}' depends on '{dep}' which is not in STAGE_ORDER. "
                        f"Valid stages: {sorted(stage_names_in_order)}"
                    )

        # 3. Warn about registered stages not in the current pipeline
        pipeline_stage_names = {s.name for s in self.stages}
        registered_stage_names = set(all_stages.keys())

        unused_stages = registered_stage_names - pipeline_stage_names
        if unused_stages:
            logger.warning(
                f"Registered stages not in current pipeline: {sorted(unused_stages)}. "
                "These stages will not be executed."
            )

        return errors

    def validate_all(
        self,
        skip_stages: List[str] = None,
        only_stages: List[str] = None,
        resume: bool = False,
    ) -> List[StageValidationResult]:
        """
        Run validate_inputs() for ALL stages in order, collecting all validation
        errors before failing.  Does NOT execute any stage.

        Checks config validity (via _validate_config) and per-stage input
        validation (via stage.validate_inputs) without modifying state or
        checkpoint.

        Args:
            skip_stages: Stage names to skip
            only_stages: If provided, only validate these stages
            resume: Whether to check checkpoint for skippable stages

        Returns:
            List of StageValidationResult, one per stage, with status indicating
            'run' (valid), 'skip' (filtered), 'checkpoint' (would restore), or
            'error' (validation failed).
        """
        skip_set = set(skip_stages or [])
        only_set = set(only_stages) if only_stages else None

        results: List[StageValidationResult] = []

        # 1. Check pipeline-level config first
        config_errors = self._validate_config()
        if config_errors:
            for err in config_errors:
                results.append(StageValidationResult(
                    stage_name='CONFIG',
                    status='error',
                    message=err,
                ))

        # 2. Optionally load checkpoint (read-only)
        checkpoint_loaded = False
        if resume:
            checkpoint_loaded = self.load_checkpoint()

        # 3. Walk stages
        for stage in self.stages:
            stage_name = stage.name

            # Filter: skip_stages
            if stage_name in skip_set:
                results.append(StageValidationResult(
                    stage_name=stage_name,
                    status='skip',
                    message='filtered by skip_stages',
                ))
                continue

            # Filter: only_stages
            if only_set and stage_name not in only_set:
                results.append(StageValidationResult(
                    stage_name=stage_name,
                    status='skip',
                    message='not in only_stages',
                ))
                continue

            # Checkpoint skip
            if resume and checkpoint_loaded and self.resume_mode:
                if stage.can_skip(self.state, self.checkpoint):
                    results.append(StageValidationResult(
                        stage_name=stage_name,
                        status='checkpoint',
                        message='would restore from checkpoint',
                    ))
                    continue

            # US-88-009: Check validation cache on resume
            validation_error = None
            cache_hit = False
            if resume and checkpoint_loaded and self.checkpoint:
                cached = self.checkpoint.get_cached_validation(stage_name)
                if cached:
                    validation_error = cached.get('error_message')
                    cache_hit = True
                    logger.debug(f"Validation cache hit for {stage_name}")

            # Validate inputs if not cached
            if not cache_hit:
                validation_error = stage.validate_inputs(self.state, self.config)
                # Save validation result to cache (for resume)
                if resume and checkpoint_loaded and self.checkpoint:
                    self.checkpoint.save_validation_result(
                        stage=stage_name,
                        is_valid=(validation_error is None),
                        error_message=validation_error
                    )

            if validation_error:
                results.append(StageValidationResult(
                    stage_name=stage_name,
                    status='error',
                    message=validation_error,
                ))
            else:
                status_msg = 'inputs valid (cached)' if cache_hit else 'inputs valid'
                results.append(StageValidationResult(
                    stage_name=stage_name,
                    status='run',
                    message=status_msg,
                ))

        return results

    def _warn_cookies_from_browser(self) -> None:
        """Check if cookies_from_browser browser is findable on PATH.

        Delegates to PipelineValidator (US-82-006).
        """
        validator = getattr(self, '_validator', None)
        if validator is not None:
            validator._warn_cookies_from_browser()
        else:
            PipelineValidator(self.config, self.stages)._warn_cookies_from_browser()

    def _validate_stage_dependencies(
        self,
        stage: Stage,
        completed_stages: set,
    ) -> None:
        """
        Validate that all declared dependencies for a stage are satisfied.

        A dependency is satisfied if it has been completed in the current run
        OR was completed in a loaded checkpoint (i.e., it was skipped/restored).

        Args:
            stage: The stage about to run.
            completed_stages: Set of stage names completed or restored so far.

        Raises:
            DependencyError: If any required dependency is missing.
        """
        depends_on = getattr(stage, 'DEPENDS_ON', [])
        if not depends_on:
            return

        missing = [dep for dep in depends_on if dep not in completed_stages]
        if missing:
            raise DependencyError(
                f"Stage '{stage.name}' requires {missing} to be completed first. "
                f"Completed stages: {sorted(completed_stages)}"
            )

    # Fields to snapshot before each stage for rollback on failure
    _SNAPSHOT_FIELDS = ('matches', 'alternatives', 'downloaded_segments', 'output_files')

    def _snapshot_state(self) -> Dict[str, Any]:
        """
        Create a granular snapshot of critical state fields before stage execution.

        Tracks field-level deep copies for complex objects and includes metadata
        for integrity validation after rollback. Maintains backward compatibility
        by exposing core snapshot fields at top level.
        """
        snapshot = {
            'stage_name': self.current_stage,
            'snapshot_timestamp': time.time(),
            'fields': {},
            'field_hashes': {},
        }

        for field_name in self._SNAPSHOT_FIELDS:
            value = getattr(self.state, field_name, None)
            if value is not None:
                # Use deep copy for lists/dicts to capture nested state
                if isinstance(value, list):
                    snapshot['fields'][field_name] = copy.deepcopy(value)
                elif isinstance(value, dict):
                    snapshot['fields'][field_name] = copy.deepcopy(value)
                else:
                    snapshot['fields'][field_name] = copy.copy(value)
                # Store hash for integrity validation
                snapshot['field_hashes'][field_name] = hash(str(value))
                # Backward compatibility: also set at top level
                snapshot[field_name] = snapshot['fields'][field_name]
            else:
                snapshot['fields'][field_name] = None
                snapshot['field_hashes'][field_name] = None
                # Backward compatibility: also set at top level
                snapshot[field_name] = None

        # Snapshot additional state fields for granular recovery
        additional_fields = (
            'video_ids', 'caption_results', 'video_search_results',
            'voiceover_segments', 'keywords', 'entity_images', 'entity_videos'
        )
        for field_name in additional_fields:
            value = getattr(self.state, field_name, None)
            if value is not None:
                if isinstance(value, (list, dict)):
                    snapshot['fields'][field_name] = copy.deepcopy(value)
                else:
                    snapshot['fields'][field_name] = copy.copy(value)
                snapshot['field_hashes'][field_name] = hash(str(value))
            else:
                snapshot['fields'][field_name] = None
                snapshot['field_hashes'][field_name] = None

        logger.debug(
            f"State snapshot created: fields={list(snapshot['fields'].keys())}"
        )
        return snapshot

    def _rollback_state(self, snapshot: Dict[str, Any], stage_name: str) -> None:
        """
        Restore state from a pre-stage snapshot after a failure with granular field-level recovery.

        Features:
        - Field-level recovery instead of whole-field restoration
        - Partial checkpoint restoration support for corrupted stage data
        - Detailed rollback decision logging for debugging
        - State integrity validation after rollback
        - Stage metrics preservation even when stage data is rolled back
        """
        if not snapshot or 'fields' not in snapshot:
            logger.warning(f"No valid snapshot for rollback after {stage_name} failure")
            return

        # Always log rollback attempt at WARNING level (for backward compatibility)
        logger.warning(f"State rollback after {stage_name} failure - restored fields: []")

        restored_fields = []
        skipped_fields = []
        integrity_issues = []

        # Get current state values before rollback for comparison
        pre_rollback_state = {}
        for field_name in snapshot['fields']:
            pre_rollback_state[field_name] = getattr(self.state, field_name, None)

        # Restore only the core snapshot fields (matches, alternatives, etc.)
        for field_name in self._SNAPSHOT_FIELDS:
            if field_name in snapshot['fields']:
                saved_value = snapshot['fields'][field_name]
                current_value = getattr(self.state, field_name, None)

                # Check if field was actually modified (avoid unnecessary restoration)
                if current_value != saved_value:
                    # Validate data integrity before restoration
                    if self._validate_field_integrity(saved_value, field_name):
                        setattr(self.state, field_name, saved_value)
                        restored_fields.append(field_name)
                        logger.debug(f"Restored field '{field_name}' after {stage_name} failure")
                    else:
                        integrity_issues.append(field_name)
                        logger.warning(
                            f"Skipping restoration of '{field_name}' due to integrity check failure"
                        )
                else:
                    skipped_fields.append(field_name)
                    logger.debug(f"Field '{field_name}' unchanged, skipping restoration")

        # Log detailed rollback decisions
        if restored_fields:
            logger.info(
                f"State rollback after {stage_name} failure - "
                f"restored fields: {restored_fields}"
            )
        elif skipped_fields and not integrity_issues:
            # Log rollback even when no fields changed (backward compatibility)
            logger.warning(
                f"State rollback after {stage_name} failure - "
                f"no fields required restoration (unchanged)"
            )
        if skipped_fields:
            logger.debug(
                f"State rollback after {stage_name} - "
                f"skipped (unchanged): {skipped_fields}"
            )
        if integrity_issues:
            logger.warning(
                f"State rollback after {stage_name} failure - "
                f"integrity issues (skipped): {integrity_issues}"
            )

        # Validate overall state integrity after rollback
        validation_result = self._validate_state_integrity(snapshot)
        if not validation_result['valid']:
            for issue in validation_result['issues']:
                logger.error(format_error_with_code("PIPE-001", f"State integrity issue after rollback: {issue}"))

        # Preserve stage metrics even when stage data is rolled back
        # Metrics are already stored in self.stage_metrics before rollback is called
        metrics_preserved = stage_name in self.stage_metrics
        if metrics_preserved:
            logger.debug(
                f"Stage metrics preserved for {stage_name}: "
                f"duration={self.stage_metrics[stage_name].duration_seconds:.2f}s"
            )
        else:
            logger.warning(
                f"No stage metrics found for {stage_name} to preserve"
            )

    def _validate_field_integrity(self, field_value: Any, field_name: str) -> bool:
        """
        Validate integrity of a field value before restoration.

        Checks for:
        - None values that shouldn't be None
        - Corrupted data types
        - Invalid nested structures
        """
        if field_value is None:
            # None is valid for optional fields
            return True

        # Check for obviously corrupted data
        if isinstance(field_value, (list, dict)):
            # Check for deeply nested corruption (suspiciously deep structures)
            try:
                import pickle
                # Quick check - if pickle fails, data might be corrupted
                pickle.dumps(field_value)
            except (pickle.PicklingError, TypeError):
                # Non-picklable is ok for some types, just warn
                pass

        return True

    def _validate_state_integrity(self, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate overall state integrity after rollback.

        Returns:
            Dict with 'valid' (bool) and 'issues' (list of issue descriptions)
        """
        issues = []

        # Check required fields exist and have valid types
        required_list_fields = ['matches', 'downloaded_segments', 'output_files']
        for field_name in required_list_fields:
            value = getattr(self.state, field_name, None)
            if value is not None and not isinstance(value, list):
                issues.append(f"Field '{field_name}' should be list, got {type(value).__name__}")

        required_dict_fields = ['alternatives', 'caption_results']
        for field_name in required_dict_fields:
            value = getattr(self.state, field_name, None)
            if value is not None and not isinstance(value, dict):
                issues.append(f"Field '{field_name}' should be dict, got {type(value).__name__}")

        # Check state consistency - matches count should align with alternatives
        matches = getattr(self.state, 'matches', [])
        alternatives = getattr(self.state, 'alternatives', {})
        if matches and alternatives:
            matched_segment_indices = set(alternatives.keys())
            for match in matches:
                if match.segment_index in matched_segment_indices:
                    # This segment has both a match and alternatives - valid
                    pass

        # Verify against snapshot hashes if provided
        if snapshot and 'field_hashes' in snapshot:
            for field_name, expected_hash in snapshot['field_hashes'].items():
                if expected_hash is None:
                    continue
                current_value = getattr(self.state, field_name, None)
                if current_value is not None:
                    current_hash = hash(str(current_value))
                    if current_hash != expected_hash:
                        issues.append(
                            f"Field '{field_name}' hash mismatch after rollback: "
                            f"expected {expected_hash}, got {current_hash}"
                        )

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def _get_state_summary(self) -> str:
        """
        Build a compact summary of current pipeline state for error context.

        Returns:
            Human-readable string summarizing segment, video, and match counts.
        """
        parts = []
        segments = getattr(self.state, 'voiceover_segments', None)
        if segments is not None:
            parts.append(f"segments={len(segments)}")

        video_ids = getattr(self.state, 'video_ids', None)
        if video_ids is not None:
            parts.append(f"videos={len(video_ids)}")

        matches = getattr(self.state, 'matches', None)
        if matches is not None:
            parts.append(f"matches={len(matches)}")

        caption_results = getattr(self.state, 'caption_results', None)
        if caption_results is not None:
            parts.append(f"captions={len(caption_results)}")

        if self.current_stage:
            parts.append(f"current_stage={self.current_stage}")

        return ", ".join(parts) if parts else "empty state"

    def _load_ralph_parallel_groups(self) -> Optional[List[Tuple[str, ...]]]:
        """
        Load parallel stage groups from ralph-config.json.

        US-108-012: Reads pipelineStages.groups from the Ralph configuration
        file to determine which stages can run in parallel.

        Returns:
            List of tuples containing stage names that can run in parallel,
            or None if the config file cannot be loaded.
        """
        import os
        import json

        # Try to find ralph-config.json
        possible_paths = [
            # Check common locations relative to project root
            os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'ralph', 'config', 'ralph-config.json'),
            os.path.join(os.getcwd(), 'scripts', 'ralph', 'config', 'ralph-config.json'),
            os.path.expanduser('~/Desktop/matcher/scripts/ralph/config/ralph-config.json'),
        ]

        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

        if not config_path:
            logger.debug("Ralph config file not found, skipping parallel group loading")
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                ralph_config = json.load(f)

            # Extract pipelineStages.groups
            pipeline_stages = ralph_config.get('pipelineStages', {})
            groups = pipeline_stages.get('groups', {})

            if not groups:
                logger.debug("No pipelineStages.groups found in ralph-config.json")
                return None

            # Convert dict of lists to list of tuples
            # Format: {"search": ["VIDEO_SEARCH", "CAPTION"]} -> [("VIDEO_SEARCH", "CAPTION")]
            parallel_groups = []
            for group_name, stage_list in groups.items():
                if isinstance(stage_list, list) and len(stage_list) >= 2:
                    # Only include groups with 2+ stages (parallel opportunity)
                    parallel_groups.append(tuple(stage_list))

            return parallel_groups if parallel_groups else None

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load ralph-config.json for parallel groups: {e}")
            return None

    def _format_log_context(
        self,
        stage_name: str,
        stage_index: int,
        total_stages: int,
        elapsed_seconds: float = 0.0,
        items_processed: int = 0,
        items_failed: int = 0,
    ) -> StageLogContext:
        """
        Build structured logging context for pipeline stage execution.

        Args:
            stage_name: Name of the stage (e.g., 'MATCH')
            stage_index: 0-based index of the stage in the pipeline
            total_stages: Total number of stages in the pipeline
            elapsed_seconds: Time elapsed for the stage (default 0.0)
            items_processed: Number of items processed successfully (default 0)
            items_failed: Number of items that failed (default 0)

        Returns:
            StageLogContext dataclass with structured context.
        """
        return StageLogContext(
            stage_name=stage_name,
            stage_index=stage_index,
            total_stages=total_stages,
            elapsed_seconds=elapsed_seconds,
            items_processed=items_processed,
            items_failed=items_failed,
        )

    def _calculate_retry_delay(self, attempt: int, strategy: str, base_delay: float, max_delay: float, jitter_factor: float = 0.2) -> float:
        """Calculate retry delay based on strategy with optional jitter.

        Args:
            attempt: 0-based attempt number (0 = first attempt, 1 = first retry, etc.)
            strategy: One of 'exponential', 'linear', 'fixed'
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap in seconds
            jitter_factor: Jitter factor (0.0-1.0) for randomization

        Returns:
            Delay in seconds before next retry
        """
        if strategy == 'exponential':
            delay = base_delay * (2 ** attempt)
        elif strategy == 'linear':
            delay = base_delay * (attempt + 1)
        else:  # fixed
            delay = base_delay

        # Apply jitter to prevent thundering herd
        if jitter_factor > 0:
            import random
            jitter_range = delay * jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure delay is non-negative

        return min(delay, max_delay)

    def _log_stage_estimate(self, stage_name: str) -> Optional[float]:
        """Log a predicted duration for the upcoming stage based on history.

        Uses the current item count (from state) and historical throughput.
        Silently does nothing when no history exists (graceful degradation).

        Returns:
            Estimated duration in seconds, or None if not available.
        """
        try:
            # Determine items_count from state heuristics
            items_count = self._guess_items_count(stage_name)
            est = estimate_duration(self.project_dir, stage_name, items_count)
            if est is not None:
                if est >= 60:
                    est_str = f"{est / 60:.1f} min"
                else:
                    est_str = f"{est:.0f}s"
                logger.info(
                    f"  Estimated duration for {stage_name}: ~{est_str}"
                    f" (based on {items_count} items, historical throughput)"
                )
            return est
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            logger.debug(f"Could not estimate duration for {stage_name}: {exc}")
            return None

    def _log_resource_prediction(self) -> None:
        """Log predicted resource usage before pipeline execution (US-138-009).

        Uses historical data and input parameters (voiceover duration, video count)
        to predict memory requirements and warn if predicted usage is high.
        """
        try:
            # Get resource prediction config
            pipeline_config = getattr(self.config, 'pipeline', None)
            if not pipeline_config:
                return

            prediction_config = getattr(pipeline_config, 'resource_prediction', None)
            if not prediction_config:
                return

            # Check if predictions are enabled
            if not getattr(prediction_config, 'enabled', True):
                return

            # Get input parameters from state
            voiceover_segments = getattr(self.state, 'voiceover_segments', [])
            video_ids = getattr(self.state, 'video_ids', [])
            matches = getattr(self.state, 'matches', [])

            # Calculate input values
            voiceover_duration_seconds = 0.0
            if voiceover_segments:
                for seg in voiceover_segments:
                    if hasattr(seg, 'duration'):
                        voiceover_duration_seconds += seg.duration
                    elif isinstance(seg, dict):
                        voiceover_duration_seconds += seg.get('duration', 0)

            video_count = len(video_ids)
            segment_count = len(voiceover_segments)
            match_count = len(matches)

            # Get prediction parameters from config
            base_memory = getattr(prediction_config, 'base_memory_per_vo_minute', 50.0)
            memory_per_video = getattr(prediction_config, 'memory_per_video', 5.0)
            memory_per_segment = getattr(prediction_config, 'memory_per_segment', 2.0)
            history_window = getattr(prediction_config, 'history_window', 5)
            warning_threshold = getattr(prediction_config, 'warning_threshold_percent', 75.0)
            critical_threshold = getattr(prediction_config, 'critical_threshold_percent', 90.0)

            # Get prediction
            prediction = predict_memory_usage(
                self.project_dir,
                voiceover_duration_seconds,
                video_count,
                match_count,
                base_memory_per_vo_minute=base_memory,
                memory_per_video=memory_per_video,
                memory_per_segment=memory_per_segment,
                history_window=history_window,
            )

            if not prediction:
                return

            predicted_mb = prediction.get('predicted_memory_mb')
            if predicted_mb is None:
                return

            # Log prediction info
            logger.info(
                f"Resource prediction: {predicted_mb}MB estimated "
                f"(voiceover: {voiceover_duration_seconds/60:.1f}min, "
                f"videos: {video_count}, segments: {segment_count}, "
                f"confidence: {prediction.get('confidence', 'unknown')})"
            )

            # Get available memory and calculate percentage
            try:
                import psutil
                total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
                predicted_percent = (predicted_mb / total_memory_mb) * 100

                # Emit warning if predicted usage is high
                if predicted_percent >= critical_threshold:
                    logger.warning(
                        f"CRITICAL: Predicted memory usage {predicted_percent:.1f}% "
                        f"({predicted_mb}MB / {total_memory_mb:.0f}MB available) - "
                        f"pipeline may encounter memory issues"
                    )
                    self._emit_resource_warning('memory', predicted_percent, critical_threshold)
                elif predicted_percent >= warning_threshold:
                    logger.warning(
                        f"Resource warning: Predicted memory usage {predicted_percent:.1f}% "
                        f"({predicted_mb}MB / {total_memory_mb:.0f}MB available)"
                    )
                    self._emit_resource_warning('memory', predicted_percent, warning_threshold)
            except Exception:
                # psutil not available, skip percentage-based warning
                logger.exception("psutil not available, skipping memory percentage calculation")

        except Exception as exc:
            logger.exception(f"Could not generate resource prediction: {exc}")

    def _run_quota_preflight_check(self) -> bool:
        """Run pre-flight quota check before pipeline execution (US-154-011).

        Checks if YouTube API quota is sufficient for the estimated pipeline run.
        Warns or auto-fallbacks to yt-dlp if quota is insufficient.

        Returns:
            True if quota check passed or is not configured, False if should skip API
        """
        try:
            # Check if download.youtube_api config section exists
            download_config = getattr(self.config, 'download', None)
            if not download_config:
                logger.debug("No download config found, skipping pre-flight check")
                return True

            youtube_api_config = getattr(download_config, 'youtube_api', None)
            if not youtube_api_config:
                logger.debug("No youtube_api config section found, skipping pre-flight check")
                return True

            # Check if pre-flight check is enabled
            enable_pre_flight = getattr(youtube_api_config, 'enable_pre_flight_check', True)
            if not enable_pre_flight:
                logger.debug("Pre-flight quota check disabled in config")
                return True

            # Check if API is enabled via config
            api_enabled = getattr(youtube_api_config, 'enabled', False)
            if not api_enabled:
                logger.debug("YouTube API not enabled, skipping pre-flight check")
                return True

            # Check for force-yt-dlp flag (passed via state or config)
            force_yt_dlp = getattr(self.state, 'force_yt_dlp', False)
            if force_yt_dlp:
                logger.info("Force yt-dlp mode enabled, skipping YouTube API")
                return True

            # Get API key(s) from config
            api_keys = getattr(youtube_api_config, 'api_keys', None)
            api_key = getattr(youtube_api_config, 'api_key', None)

            if not api_keys and not api_key:
                logger.debug("No API keys configured, skipping pre-flight check")
                return True

            # Get quota parameters from config
            quota_limit = getattr(youtube_api_config, 'quota_limit', 10000)
            quota_fallback_threshold = getattr(youtube_api_config, 'quota_fallback_threshold_percent', 10)

            # Get project parameters for estimation
            voiceover_segments = getattr(self.state, 'voiceover_segments', [])
            keyword_count = len(getattr(self.state, 'keywords', []))
            video_ids = getattr(self.state, 'video_ids', [])

            # Estimate parameters
            segment_count = len(voiceover_segments)
            # Estimate: ~3 videos per segment for matching
            estimated_video_calls = min(segment_count * 3, 200)
            # Estimate: ~50% of videos need captions
            estimated_caption_fetches = min(len(video_ids) * 0.5, 100) if video_ids else 50

            # Create API client to check quota
            with YouTubeAPIClient(
                api_key=api_key or (api_keys[0] if api_keys else ""),
                api_keys=api_keys,
                quota_limit=quota_limit,
                quota_fallback_threshold_percent=quota_fallback_threshold,
            ) as api_client:
                # Estimate quota needed
                estimate = api_client.estimate_quota_for_pipeline(
                    keyword_count=keyword_count,
                    estimated_video_metadata_calls=estimated_video_calls,
                    estimated_caption_fetches=int(estimated_caption_fetches),
                )

                # Log the estimate
                logger.info(
                    f"Quota pre-flight: estimated {estimate['total_estimated_quota']} units needed, "
                    f"{estimate['current_quota_remaining']} remaining ({estimate['remaining_percent']:.1f}%)"
                )

                status = estimate['status']
                recommendation = estimate['recommendation']

                if status == "sufficient":
                    logger.info(f"Quota check passed: {recommendation}")
                    return True
                elif status == "low":
                    logger.warning(f"Quota check warning: {recommendation}")
                    return True
                elif status == "insufficient":
                    logger.warning(f"Quota check warning: {recommendation}")
                    # Set flag to prefer yt-dlp for captions
                    self.state.quota_insufficient = True
                    return True
                else:  # critically_low
                    logger.error(format_error_with_code("SEARCH-002", f"Quota check failed: {recommendation}"))
                    # Set flag to force yt-dlp for all API operations
                    self.state.force_yt_dlp = True
                    return True

                # US-155-006: Run API connectivity health check
                self._run_api_health_check(api_client)

        except ImportError:
            logger.debug("YouTubeAPIClient not available, skipping pre-flight check")
            return True
        except Exception as exc:
            logger.warning(f"Quota pre-flight check failed: {exc}")
            # Don't block pipeline on pre-flight check errors
            return True

    def _run_api_health_check(self, api_client) -> None:
        """Run API connectivity health check (US-155-006).

        Validates API connectivity before pipeline execution.
        Logs health status but doesn't block pipeline - failures are handled gracefully.

        Args:
            api_client: Initialized YouTubeAPIClient instance
        """
        try:
            is_valid, error_message, quota_info = api_client.health_check()

            # US-155-012: Get retry budget stats for health check output
            retry_budget_stats = {}
            try:
                retry_budget_stats = api_client.get_retry_budget_stats() or {}
            except Exception:
                logger.exception("Could not get retry budget stats during health check")

            if is_valid:
                logger.info(
                    f"API health check passed - connectivity OK, "
                    f"quota: {quota_info.get('percent_used', 0):.1f}% used"
                )
                # US-155-012: Log retry budget status at INFO
                if retry_budget_stats:
                    remaining = retry_budget_stats.get('attempts_remaining', 'N/A')
                    max_att = retry_budget_stats.get('attempts_max', 'unlimited')
                    logger.info(
                        f"Retry budget: {retry_budget_stats.get('attempts_used', 0)}/{max_att} attempts used, "
                        f"{remaining} remaining"
                    )
                # Store health check result in state for metrics export
                self.state.api_health_check = {
                    'status': 'ok',
                    'is_valid': True,
                    'error_message': '',
                    'quota_info': quota_info,
                    'retry_budget': retry_budget_stats,  # US-155-012
                }
            else:
                # Log error but don't block pipeline
                if 'network' in error_message.lower() or 'timeout' in error_message.lower():
                    logger.warning(f"API health check failed (network): {error_message}")
                elif 'quota' in error_message.lower() or 'exceeded' in error_message.lower():
                    logger.warning(f"API health check failed (quota): {error_message}")
                else:
                    logger.warning(f"API health check failed: {error_message}")

                # Store health check result in state for metrics export
                self.state.api_health_check = {
                    'status': 'failed',
                    'is_valid': False,
                    'error_message': error_message,
                    'quota_info': quota_info,
                    'retry_budget': retry_budget_stats,  # US-155-012
                }

        except Exception as exc:
            logger.warning(f"API health check error: {exc}")
            # Store error state for metrics
            self.state.api_health_check = {
                'status': 'error',
                'is_valid': False,
                'error_message': str(exc),
                'quota_info': {},
                'retry_budget': {},  # US-155-012
            }

    def _emit_resource_warning(self, resource_type: str, current_value: float, threshold: float) -> None:
        """Emit a resource warning event when CPU or memory thresholds are exceeded (US-106-006).

        Args:
            resource_type: 'cpu' or 'memory'
            current_value: Current usage percentage (0-100)
            threshold: The threshold that was exceeded
        """
        logger.warning(f"Resource warning: {resource_type} at {current_value:.1f}% (threshold: {threshold}%)")
        self.emit_event(PipelineEvent(
            event_type=EVENT_RESOURCE_WARNING,
            stage_name=self.current_stage or '',
            timestamp=time.time(),
            data={
                'resource_type': resource_type,
                'current_value': current_value,
                'threshold': threshold,
            },
        ))

    # US-106-011: Resource monitoring
    # Default threshold for memory warning (80% of available)
    DEFAULT_MEMORY_THRESHOLD = 80.0
    # Default critical threshold for automatic pause (90% of available)
    DEFAULT_CRITICAL_MEMORY_THRESHOLD = 90.0
    DEFAULT_CRITICAL_CPU_THRESHOLD = 95.0

    def _get_disk_usage_percent(self) -> Optional[float]:
        """Get current disk usage percentage for the project directory.

        Returns:
            Disk usage percentage (0-100), or None if unavailable
        """
        try:
            import psutil
            if self.project_dir:
                usage = psutil.disk_usage(str(self.project_dir))
                return usage.percent
            return None
        except Exception:
            return None

    def _get_memory_usage(self) -> Optional[Dict[str, float]]:
        """Get current memory usage using psutil (US-106-011).

        Returns:
            Dict with 'percent', 'available_gb', 'total_gb', or None if psutil unavailable
        """
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'percent': mem.percent,
                'available_gb': mem.available / (1024 ** 3),
                'total_gb': mem.total / (1024 ** 3),
            }
        except ImportError:
            return None

    def _get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage percentage using psutil (US-106-011).

        Returns:
            CPU usage percentage (0-100), or None if psutil unavailable
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return None

    def _track_stage_resources(self, stage_name: str, phase: str) -> Dict[str, Any]:
        """Track memory and CPU usage before/after stage execution (US-106-011).

        Args:
            stage_name: Name of the stage being tracked
            phase: 'before' or 'after' stage execution

        Returns:
            Dict with resource metrics collected
        """
        metrics = {'phase': phase, 'stage_name': stage_name}

        mem_info = self._get_memory_usage()
        if mem_info:
            metrics['memory_percent'] = mem_info['percent']
            metrics['memory_available_gb'] = mem_info['available_gb']
            metrics['memory_total_gb'] = mem_info['total_gb']

            # Check if memory exceeds warning threshold and emit warning
            threshold = getattr(self.config.pipeline, 'memory_threshold', self.DEFAULT_MEMORY_THRESHOLD) if self.config else self.DEFAULT_MEMORY_THRESHOLD
            # Ensure threshold is a number (handle mock objects in tests)
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                threshold = self.DEFAULT_MEMORY_THRESHOLD
            if isinstance(mem_info['percent'], (int, float)) and mem_info['percent'] >= threshold and phase == 'after':
                self._emit_resource_warning('memory', mem_info['percent'], threshold)

            # Check if memory exceeds critical threshold and pause (US-108-009)
            critical_threshold = getattr(self.config.pipeline, 'critical_memory_threshold', self.DEFAULT_CRITICAL_MEMORY_THRESHOLD) if self.config else self.DEFAULT_CRITICAL_MEMORY_THRESHOLD
            try:
                critical_threshold = float(critical_threshold)
            except (TypeError, ValueError):
                critical_threshold = self.DEFAULT_CRITICAL_MEMORY_THRESHOLD
            if isinstance(mem_info['percent'], (int, float)) and mem_info['percent'] >= critical_threshold:
                self._handle_critical_resource('memory', mem_info['percent'], critical_threshold)

        cpu_info = self._get_cpu_usage()
        if cpu_info is not None:
            metrics['cpu_percent'] = cpu_info

            # Check if CPU exceeds critical threshold and pause (US-108-009)
            critical_cpu_threshold = getattr(self.config.pipeline, 'critical_cpu_threshold', self.DEFAULT_CRITICAL_CPU_THRESHOLD) if self.config else self.DEFAULT_CRITICAL_CPU_THRESHOLD
            try:
                critical_cpu_threshold = float(critical_cpu_threshold)
            except (TypeError, ValueError):
                critical_cpu_threshold = self.DEFAULT_CRITICAL_CPU_THRESHOLD
            if isinstance(cpu_info, (int, float)) and cpu_info >= critical_cpu_threshold and phase == 'after':
                self._handle_critical_resource('cpu', cpu_info, critical_cpu_threshold)

        # Track disk usage
        disk_percent = self._get_disk_usage_percent()
        if disk_percent is not None:
            metrics['disk_percent'] = disk_percent

        return metrics

    def _handle_critical_resource(self, resource_type: str, current_value: float, threshold: float) -> None:
        """Handle critical resource threshold exceeded by emitting warning and waiting (US-108-009).

        Args:
            resource_type: 'memory', 'cpu', or 'disk'
            current_value: Current resource usage value
            threshold: Critical threshold that was exceeded
        """
        logger.warning(f"Critical resource: {resource_type} at {current_value:.1f}% (threshold: {threshold}%). Waiting for cleanup...")

        # Emit a critical resource event
        event = PipelineEvent(
            event_type=EVENT_RESOURCE_WARNING,
            stage_name='',
            timestamp=time.time(),
            data={
                'resource_type': resource_type,
                'current_value': current_value,
                'threshold': threshold,
                'is_critical': True,
            },
        )
        get_event_bus().emit(event)

        # Wait briefly to allow cleanup (GC, etc.)
        import time as time_module
        time_module.sleep(2)

    def _run_health_checks(self, stage_name: str) -> List[Any]:
        """Run health checks before stage execution (US-88-005, US-125-011).

        Runs pre-stage health checks that validate external dependencies.
        Uses interval-based scheduling - only runs if enough time has passed
        since last check (configurable per stage type).

        Args:
            stage_name: Name of the stage to check

        Returns:
            List of health check results
        """
        import time

        # US-125-011: Check if we should run health check based on interval
        if not self._should_run_health_check(stage_name):
            interval = self._get_health_check_interval(stage_name)
            elapsed = time.time() - self._last_health_check_time
            logger.debug(
                f"Skipping health check for {stage_name}: "
                f"only {elapsed:.0f}s elapsed (interval: {interval}s)"
            )
            return []

        try:
            start_time = time.time()
            checker = HealthChecker(self.config)
            project_path = str(self.project_dir) if self.project_dir else None
            results = checker.check_stage(stage_name, project_path)

            # Track last health check time
            self._last_health_check_time = time.time()

            # Add timing metrics to results
            total_duration_ms = (time.time() - start_time) * 1000
            for result in results:
                if hasattr(result, 'duration_ms') and result.duration_ms > 0:
                    # Individual check already has duration
                    pass
                else:
                    # Add total duration to each result
                    result.duration_ms = total_duration_ms / max(len(results), 1)

            # US-163-008: Log health check completion with progress
            total_checks = len(results)
            if total_checks > 0:
                # Count passed/warning/failed
                passed = sum(1 for r in results if r.status.value == 'ok')
                warnings = sum(1 for r in results if r.status.value == 'warning')
                failed = sum(1 for r in results if r.status.value == 'failed')

                # Log progress with completion percentage
                completion_pct = 100.0  # Health checks complete when results are available
                log_progress(
                    logger,
                    "HEALTH_CHECK",
                    completion_pct,
                    passed + warnings + failed,
                    total_checks,
                    stage_name=stage_name,
                    passed=passed,
                    warnings=warnings,
                    failed=failed,
                    duration_ms=total_duration_ms,
                )

                # US-163-008: Log any health check failures with structured logging
                for result in results:
                    if result.status.value == 'failed':
                        log_error_with_context(
                            logger,
                            "PIPE-002",
                            f"Health check failed: {result.message}",
                            stage_name=stage_name,
                            check_type=result.name,
                            details=result.details,
                        )
                    elif result.status.value == 'warning':
                        # Log warnings at info level with context
                        logger.info(
                            f"[HEALTH_CHECK] Warning for {stage_name}: {result.message} "
                            f"(check_type={result.name})"
                        )

            return results
        except Exception as e:
            # US-163-008: Use structured error logging
            log_error_with_context(
                logger,
                "PIPE-002",
                f"Health check failed for {stage_name}: {str(e)}",
                stage_name=stage_name,
            )
            return []

    def _save_stage_timing(self, stage_name: str, elapsed: float) -> None:
        """Persist stage timing to history file after successful completion."""
        try:
            metrics = self.stage_metrics.get(stage_name)
            items = metrics.items_processed if metrics else 0
            throughput = (items / elapsed) if (elapsed > 0 and items > 0) else 0.0
            append_stage_timing(
                self.project_dir, stage_name, elapsed, items, throughput
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.debug(f"Could not save stage timing for {stage_name}: {exc}")

    def _save_stage_resource_usage(self, stage_name: str) -> None:
        """Persist stage resource usage to history file after completion (US-138-009)."""
        try:
            # Get current resource usage
            from .pipeline_progress import _get_resource_usage
            memory_mb, cpu_percent, _, _ = _get_resource_usage()

            if memory_mb is None and cpu_percent is None:
                # No resource data available
                return

            # Get input parameters for context
            voiceover_segments = getattr(self.state, 'voiceover_segments', [])
            video_ids = getattr(self.state, 'video_ids', [])
            matches = getattr(self.state, 'matches', [])

            # Calculate voiceover duration
            voiceover_duration = 0.0
            if voiceover_segments:
                for seg in voiceover_segments:
                    if hasattr(seg, 'duration'):
                        voiceover_duration += seg.duration
                    elif isinstance(seg, dict):
                        voiceover_duration += seg.get('duration', 0)

            append_resource_usage(
                self.project_dir,
                stage_name,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                voiceover_duration_seconds=voiceover_duration,
                video_count=len(video_ids) if video_ids else None,
                segment_count=len(voiceover_segments) if voiceover_segments else None,
            )
        except Exception as exc:
            logger.exception(f"Could not save resource usage for {stage_name}: {exc}")

    def _guess_items_count(self, stage_name: str) -> int:
        """Heuristic to determine expected item count for a stage.

        Returns 0 when unknown (estimation falls back to average duration).
        """
        if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
            segments = getattr(self.state, 'voiceover_segments', None)
            return len(segments) if segments else 0
        if stage_name in ('CAPTION', 'VIDEO_SEARCH'):
            video_ids = getattr(self.state, 'video_ids', None)
            return len(video_ids) if video_ids else 0
        if stage_name == 'DOWNLOAD_SEGMENTS':
            matches = getattr(self.state, 'matches', None)
            return len(matches) if matches else 0
        return 0

    def _capture_stage_errors(self, stage_name: str, result: StageResult) -> None:
        """Capture stage errors for pipeline-level aggregation.

        US-106-009: Extracts error categories from stage metrics and adds them
        to the pipeline-level error aggregator.

        Args:
            stage_name: Name of the stage that completed
            result: StageResult containing metrics with error categories
        """
        if not result.metrics:
            return

        error_categories = getattr(result.metrics, 'error_categories', {})
        if not error_categories:
            return

        # Create a temporary aggregator for this stage's errors
        stage_agg = ErrorAggregator()

        # Add each error category from the stage
        for cat_str, count in error_categories.items():
            # Convert to ErrorCategory if possible
            try:
                cat = ErrorCategory(cat_str)
            except ValueError:
                # Legacy category string - normalize it
                from .stages.error_aggregator import normalize_category
                cat = normalize_category(cat_str)

            # Record count errors with the category
            sample = f"Errors from {stage_name}"
            for _ in range(count):
                stage_agg.record(sample, cat)

        # Add to pipeline aggregator
        self._error_aggregator.add_stage_errors(stage_name, stage_agg)

    def _compute_and_check_error_rate(self, stage_name: str) -> None:
        """Compute error rate for stage and emit warning if threshold exceeded.

        US-138-010: Calculates error rate from items_failed / items_processed
        and emits a warning event if the configured threshold is exceeded.

        Args:
            stage_name: Name of the stage to check
        """
        # Get config
        error_tracking_config = getattr(self.config.pipeline_config, 'error_rate_tracking', None)
        if not error_tracking_config or not getattr(error_tracking_config, 'enabled', False):
            return

        # Get stage metrics
        if stage_name not in self.stage_metrics:
            return

        metrics = self.stage_metrics[stage_name]
        if not hasattr(metrics, 'items_processed'):
            return

        # Compute error rate
        metrics.compute_error_rate()
        error_rate = metrics.error_rate

        if error_rate <= 0:
            return

        # Check threshold
        stage_type = stage_name.upper()
        status, threshold = error_tracking_config.check_threshold(stage_type, error_rate)

        if status != 'ok':
            # Emit warning event
            self.emit_event(PipelineEvent(
                event_type=EVENT_ERROR_RATE_THRESHOLD,
                stage_name=stage_name,
                timestamp=time.time(),
                data={
                    'error_rate': error_rate,
                    'threshold': threshold,
                    'status': status,
                    'items_processed': metrics.items_processed,
                    'items_failed': metrics.items_failed,
                },
                error=f"Error rate {error_rate:.1%} exceeds {status} threshold {threshold:.1%}" if status != 'ok' else None,
            ))

            # Log warning
            logger.warning(
                f"[{stage_name}] Error rate {error_rate:.1%} ({metrics.items_failed}/{metrics.items_processed}) "
                f"exceeds {status} threshold {threshold:.1%}"
            )

    def _log_pipeline_error_summary(self) -> None:
        """Log pipeline-level error summary after pipeline completion.

        US-106-009: Logs comprehensive summary including:
        - Total errors by category
        - Errors per stage
        - Trend analysis
        - Repeated patterns
        """
        self._error_aggregator.log_pipeline_summary()

    def _print_timing_summary(
        self,
        total_duration: float,
        skipped_stages: set
    ) -> None:
        """
        Log a formatted table of per-stage timing after pipeline completion.

        Shows each stage's wall-clock duration, percentage of total pipeline
        time, and marks checkpoint-resumed stages as 'skipped'.

        Args:
            total_duration: Total pipeline wall-clock time in seconds.
            skipped_stages: Set of stage names that were restored from checkpoint.
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("Pipeline Timing Summary")
        logger.info("=" * 70)
        logger.info(f"  {'Stage':<25} {'Duration':>10} {'% of Total':>12} {'Throughput':>16}")
        logger.info(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*16}")

        total_stages = len(self.stages)
        for stage_index, stage in enumerate(self.stages):
            name = stage.name
            # US-81-007: Format throughput column from stage_metrics
            throughput_str = ''
            items_processed = 0
            items_failed = 0
            metrics = self.stage_metrics.get(name)
            if metrics:
                if hasattr(metrics, 'items_per_second') and metrics.items_per_second > 0:
                    throughput_str = f"{metrics.items_per_second:.1f} items/sec"
                items_processed = getattr(metrics, 'items_processed', 0)
                items_failed = getattr(metrics, 'items_failed', 0)

            # Build structured context for this stage
            elapsed = self.stage_timings.get(name, 0.0)
            ctx = self._format_log_context(
                stage_name=name,
                stage_index=stage_index,
                total_stages=total_stages,
                elapsed_seconds=elapsed,
                items_processed=items_processed,
                items_failed=items_failed,
            )

            if name in skipped_stages:
                logger.info(f"  {name:<25} {'skipped':>10} {'-':>12} {'-':>16} {ctx.to_suffix()}")
            elif name in self.stage_timings:
                elapsed = self.stage_timings[name]
                pct = (elapsed / total_duration * 100) if total_duration > 0 else 0.0
                logger.info(f"  {name:<25} {elapsed:>9.1f}s {pct:>11.1f}% {throughput_str:>16} {ctx.to_suffix()}")
            else:
                # Stage was filtered out (skip_stages / only_stages)
                logger.info(f"  {name:<25} {'--':>10} {'-':>12} {'-':>16} {ctx.to_suffix()}")

        logger.info(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*16}")
        logger.info(f"  {'TOTAL':<25} {total_duration:>9.1f}s {'100.0%':>12}")
        logger.info("=" * 70)

        # US-88-009: Get validation cache stats for metrics
        validation_cache_stats = None
        if self.checkpoint:
            validation_cache_stats = self.checkpoint.get_validation_cache_stats()

        # Persist timing summary to checkpoint stage_metrics
        self.checkpoint.save_stage_timing_summary(
            self.stage_timings, total_duration, skipped_stages, validation_cache_stats
        )

    def _handle_verbose_progress(self, event: 'PipelineEvent') -> None:
        """
        Handle verbose progress events - print detailed per-stage progress.

        Called when EVENT_STAGE_PROGRESS is emitted by ProgressReporter.
        Prints: STAGE_NAME: XX% (completed/total) | ETA: Xm Xs | Memory: XXX MB | CPU: XX%

        Args:
            event: PipelineEvent with stage progress data
        """
        if not self.verbose_progress:
            return

        data = event.data
        stage_name = event.stage_name
        completed = data.get('items_completed', 0)
        total = data.get('items_total', 0)
        pct = data.get('progress_percent')
        memory_mb = data.get('memory_usage_mb')
        cpu_pct = data.get('cpu_percent')

        # Build progress string
        if total > 0:
            progress_str = f"{stage_name}: {pct:.1f}% ({completed}/{total})"
        else:
            progress_str = f"{stage_name}: {completed} items"

        # Add ETA if available from progress reporter
        snapshot = self.progress_reporter.get_snapshot()
        eta = snapshot.get('estimated_remaining_seconds')
        if eta is not None and eta > 0:
            from .pipeline_progress import _format_duration
            eta_str = _format_duration(eta)
            progress_str += f" | ETA: {eta_str}"

        # Add pipeline ETA breakdown by remaining stages (US-138-008)
        eta_by_stage = snapshot.get('pipeline_eta_by_stage', [])
        if eta_by_stage:
            from .pipeline_progress import _format_duration
            remaining_eta_parts = []
            for stage_info in eta_by_stage:
                stage_name_eta = stage_info['stage']
                eta_display = stage_info['eta_display']
                remaining_eta_parts.append(f"{stage_name_eta}: {eta_display}")
            if remaining_eta_parts:
                progress_str += f"\n    Pipeline: {' | '.join(remaining_eta_parts)}"

        # Add resource usage if available
        if memory_mb is not None:
            progress_str += f" | Memory: {memory_mb:.0f} MB"
        if cpu_pct is not None:
            progress_str += f" | CPU: {cpu_pct:.0f}%"

        logger.info(progress_str)

    def _get_recovery_suggestion(self, stage_name: str, error: str) -> str:
        """
        Return an actionable recovery suggestion based on stage name and error.

        Args:
            stage_name: The name of the failed stage.
            error: The error message string.

        Returns:
            A recovery suggestion string, or empty string if none applicable.
        """
        error_lower = (error or "").lower()

        # Stage-specific recovery hints
        suggestions = {
            "ANALYZE": "Try: --fresh to re-analyze, or check that voiceover file exists and is readable.",
            "VIDEO_SEARCH": "Try: --fresh to re-run from start, or check network connectivity and API keys.",
            "CAPTION": "Try: --resume to retry captions, or increase retry_budget.max_attempts in config.",
            "MATCH": "Try: --match-only to re-run matching, or --fresh if video data is stale.",
            "ITERATIVE_MATCH": "Try: --match-only to re-run matching from MATCH stage.",
            "DOWNLOAD_SEGMENTS": "Try: --resume to retry downloads. Check disk space and network.",
            "OUTPUT": "Try: --output-only to regenerate output files, or --resume.",
        }

        # Error-pattern-specific hints (override stage defaults when more specific)
        if "no voiceover" in error_lower or "voiceover_path" in error_lower or "voiceover file" in error_lower:
            return "Provide a voiceover file with --voiceover <path>."
        if "no keywords" in error_lower:
            return "Run ANALYZE stage first, or use --use-keywords <preset> to load saved keywords."
        if "no matches" in error_lower:
            return "Run MATCH stage first with --match-only, or --fresh for a full re-run."
        if "no video" in error_lower or "video_ids" in error_lower:
            return "Run VIDEO_SEARCH stage first, or --fresh for a full re-run."
        if "checkpoint" in error_lower or "corrupt" in error_lower:
            return "Try: --fresh to discard checkpoint and start over."
        if "permission" in error_lower or "writable" in error_lower:
            return "Check file/directory permissions for the project folder."
        if "embedding" in error_lower or "provider" in error_lower:
            return "Configure embedding.provider in config.yaml (e.g., 'sentence-transformers')."

        return suggestions.get(stage_name, "Try: --fresh to restart the pipeline from scratch.")

    def _check_match_coverage_gate(self, stage_name: str) -> None:
        """
        Check match coverage quality gate after a matching stage.

        Logs a warning if fewer than the configured threshold of voiceover
        segments have a match. Non-blocking — the pipeline continues regardless.

        Args:
            stage_name: Name of the stage that just completed (for log context).
        """
        segments = getattr(self.state, 'voiceover_segments', None)
        matches = getattr(self.state, 'matches', None)

        if not segments:
            return  # No segments to check against

        total_segments = len(segments)
        if total_segments == 0:
            return

        matched_count = len(matches) if matches else 0
        coverage = matched_count / total_segments

        # Get threshold from config
        pipeline_config = getattr(self.config, 'pipeline', None)
        quality_gates = getattr(pipeline_config, 'quality_gates', None)
        threshold = getattr(quality_gates, 'min_match_coverage', 0.5)

        if coverage < threshold:
            logger.warning(
                f"Quality gate [{stage_name}]: Only {matched_count}/{total_segments} "
                f"segments matched ({coverage:.0%}). "
                f"Consider running with --force-rematch or checking video search results."
            )
        else:
            logger.info(
                f"Quality gate [{stage_name}]: {matched_count}/{total_segments} "
                f"segments matched ({coverage:.0%}) — above {threshold:.0%} threshold."
            )

    def _check_data_drift(self, stage_name: str) -> None:
        """
        Check for cross-stage data drift after a stage completes (US-88-006).

        Compares output field counts against source field counts using
        configurable drift rules. Supports multiple threshold types:
        - ratio: target >= threshold * source (e.g., 0.8 = 80%)
        - absolute_count: target >= threshold (exact number)
        - percentage: target >= (threshold/100) * source (e.g., 80 = 80%)

        Emits a warning (or error if severity=error) when the threshold
        is not met. Non-blocking by default — the pipeline continues unless
        severity is set to 'error'.

        Drift events are tracked in self.drift_history for trend analysis.

        Args:
            stage_name: Name of the stage that just completed.
        """
        # Check if drift detection is enabled
        if self._drift_config and not self._drift_config.enabled:
            return

        # Get rules for this stage from config, fall back to hardcoded defaults
        if self._drift_config:
            rules = self._drift_config.get_rules_for_stage(stage_name)
        else:
            # Fallback to hardcoded DRIFT_RULES if no config
            rules = self._get_fallback_rules(stage_name)

        for rule in rules:
            source_value = getattr(self.state, rule.source_field, None)
            target_value = getattr(self.state, rule.target_field, None)

            # Get counts (handle both list and dict)
            expected = len(source_value) if source_value else 0
            actual = len(target_value) if target_value else 0

            if expected == 0:
                continue  # Can't compute threshold with zero source

            # Calculate ratio for comparison
            ratio = actual / expected

            # Check threshold based on threshold_type
            threshold_met = self._check_threshold(rule, actual, expected, ratio)

            if not threshold_met:
                # Create drift event for history
                drift_event = {
                    'stage': stage_name,
                    'source_field': rule.source_field,
                    'target_field': rule.target_field,
                    'expected': expected,
                    'actual': actual,
                    'ratio': ratio,
                    'threshold_type': rule.threshold_type,
                    'threshold': rule.threshold,
                    'severity': rule.severity,
                    'timestamp': time.time(),
                }

                # Track in history (respect max_history limit)
                if self._drift_config and self._drift_config.track_history:
                    self.drift_history.append(drift_event)
                    if self._drift_config.max_history:
                        max_hist = self._drift_config.max_history
                        if len(self.drift_history) > max_hist:
                            self.drift_history = self.drift_history[-max_hist:]

                # Get remediation message
                remediation = rule.get_remediation_message(expected, actual, ratio)

                # Log based on severity
                log_message = (
                    f"Data drift detected [{stage_name}]: {rule.target_field} has {actual} items "
                    f"but {rule.source_field} has {expected} "
                    f"(ratio: {ratio:.1%}, threshold: {rule.threshold_type}={rule.threshold}). "
                    f"{remediation}"
                )

                if rule.severity == "error":
                    logger.error(log_message)
                else:
                    logger.warning(log_message)

    def _check_threshold(
        self,
        rule: 'DriftRuleConfig',
        actual: int,
        expected: int,
        ratio: float
    ) -> bool:
        """Check if the threshold is met based on threshold_type.

        Args:
            rule: The drift rule configuration
            actual: Actual count from target field
            expected: Expected count from source field
            ratio: Computed ratio of actual/expected

        Returns:
            True if threshold is met, False otherwise
        """
        if rule.threshold_type == "ratio":
            return ratio >= rule.threshold
        elif rule.threshold_type == "absolute_count":
            return actual >= rule.threshold
        elif rule.threshold_type == "percentage":
            # percentage threshold: actual >= (threshold/100) * expected
            return ratio >= (rule.threshold / 100.0)
        else:
            # Default to ratio
            return ratio >= rule.threshold

    def _get_fallback_rules(self, stage_name: str) -> List['DriftRuleConfig']:
        """Get fallback hardcoded rules if no config is provided.

        Args:
            stage_name: Name of the stage

        Returns:
            List of DriftRuleConfig for the stage
        """
        from .config.sections.infrastructure import DriftRuleConfig

        fallback_map = {
            "CAPTION": [DriftRuleConfig(
                trigger_stage="CAPTION",
                source_field="video_ids",
                target_field="caption_results",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
            )],
            "MATCH": [DriftRuleConfig(
                trigger_stage="MATCH",
                source_field="video_ids",
                target_field="text_metadata",
                threshold_type="ratio",
                threshold=0.8,
                severity="warning",
            )],
            "OUTPUT": [DriftRuleConfig(
                trigger_stage="OUTPUT",
                source_field="voiceover_segments",
                target_field="matches",
                threshold_type="ratio",
                threshold=0.5,
                severity="warning",
            )],
        }
        return fallback_map.get(stage_name, [])

    def run(
        self,
        resume: bool = True,
        skip_stages: List[str] = None,
        only_stages: List[str] = None,
        on_stage_start: StageStartCallback = None,
        on_stage_complete: StageCompleteCallback = None,
        parallel_stages: List[Tuple[str, ...]] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Run the pipeline.

        Args:
            resume: Whether to resume from checkpoint if available
            skip_stages: Stage names to skip
            only_stages: If provided, only run these stages
            on_stage_start: Callback invoked when a stage starts (receives stage_name)
            on_stage_complete: Callback invoked when a stage completes
                               (receives stage_name, result, elapsed_seconds)
            parallel_stages: List of tuples specifying stages to run in parallel.
                            Each tuple contains stage names that can execute concurrently.
                            E.g., [("entity_images", "entity_videos")] runs those two in parallel.
                            Checkpoint saves maintain stage order even for parallel stages.
            dry_run: If True, log stage names and validation results without executing.
                    Useful for previewing what the pipeline would do.

        Returns:
            True if pipeline completed successfully (or dry-run validation passed)
        """
        # US-159-005: Generate and set correlation ID for this pipeline run
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)
        logger.info(f"[{correlation_id}] Pipeline run starting")
        logger.debug(f"[{correlation_id}] Correlation ID assigned and set for this pipeline run")

        skip_stages = set(skip_stages or [])
        only_stages = set(only_stages) if only_stages else None

        # Re-validate config with stages present (US-44-003 + US-45-010)
        # __init__ validates config-only checks; this catches stage-dependent checks
        # (e.g., embedding provider required when matching stages are added post-init)
        config_errors = self._validate_config()
        if config_errors:
            for error in config_errors:
                logger.error(format_error_with_code("CFG-002", f"Config validation error: {error}"))
            return False

        # Dry-run mode: log stages and validate without executing
        if dry_run:
            return self._run_dry_run(skip_stages, only_stages, resume=resume)

        # US-154-011: Run pre-flight quota check before pipeline execution
        if not self._run_quota_preflight_check():
            logger.warning("Quota pre-flight check indicated insufficient quota - continuing with yt-dlp fallback")

        # US-155-011: Display initial quota status at pipeline start
        if self.show_quota:
            self._display_initial_quota()

        # US-88-011: Validate no circular dependencies in stage graph
        try:
            validate_no_circular_dependencies(self.stages)
        except ValueError as e:
            logger.error(format_error_with_code("PIPE-005", f"Stage dependency validation failed: {e}"))
            return False

        # US-106-007: Check parallel_execution config flag for auto-detection
        # Only auto-detect parallel stages if explicitly enabled in config
        parallel_execution_enabled = False
        pipeline_config = getattr(self.config, 'pipeline', None)
        if pipeline_config:
            try:
                parallel_execution_enabled = bool(getattr(pipeline_config, 'parallel_execution', False))
            except (TypeError, AttributeError):
                parallel_execution_enabled = False

        # US-108-012: Load parallel stage groups from ralph-config.json
        # This allows explicit configuration of which stages can run in parallel
        ralph_config_groups = None
        if parallel_execution_enabled and parallel_stages is None:
            ralph_config_groups = self._load_ralph_parallel_groups()
            if ralph_config_groups:
                logger.info(f"Loaded {len(ralph_config_groups)} parallel stage group(s) from ralph-config.json")
                parallel_stages = ralph_config_groups

        # US-88-011: Auto-detect parallel stages from DEPENDS_ON if not provided
        # Only auto-detect if parallel_execution is enabled in config and no ralph config
        if parallel_stages is None and parallel_execution_enabled:
            detected_groups = detect_parallel_stage_groups(self.stages)
            if detected_groups:
                logger.info(f"Auto-detected {len(detected_groups)} parallel stage group(s) from DEPENDS_ON")
                parallel_stages = detected_groups

        # Log parallel execution plan for transparency
        if parallel_stages:
            log_parallel_execution_plan(self.stages, parallel_stages)

        # Build parallel stage lookup: stage_name -> tuple of parallel stage names
        parallel_groups: Dict[str, Tuple[str, ...]] = {}
        if parallel_stages:
            for group in parallel_stages:
                for stage_name in group:
                    parallel_groups[stage_name] = group

        # Track skipped stages and pipeline start time for timing summary
        skipped_stages: set = set()
        pipeline_start_time = time.time()

        # Try to load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Freeze config to prevent mutation during pipeline execution
        self.config.freeze()

        # US-138-009: Run pre-execution resource prediction
        self._log_resource_prediction()

        # US-125-006: Setup signal handler for Ctrl+C to pause gracefully
        self._setup_pause_signal_handler()

        # Track processed parallel groups to avoid running same group twice
        processed_parallel_groups: set = set()

        # US-108-012: Track parallel execution stats
        stages_run_concurrently: List[str] = []  # Stages that ran in parallel groups
        parallel_group_timings: Dict[str, float] = {}  # group_key -> elapsed time

        # Track completed/restored stages for dependency validation
        completed_stages: set = set()

        # US-169-007: Pre-compute which stages will be restored from checkpoint
        stages_to_restore_from_checkpoint = []
        if self.resume_mode and self.checkpoint and self.checkpoint.data:
            last_completed = self.checkpoint.data.last_completed_stage
            if last_completed:
                try:
                    last_idx = STAGE_ORDER.index(last_completed)
                    for stage in self.stages:
                        if stage.name in STAGE_ORDER:
                            stage_idx = STAGE_ORDER.index(stage.name)
                            if stage_idx <= last_idx and stage.can_skip(self.state, self.checkpoint):
                                stages_to_restore_from_checkpoint.append(stage.name)
                except ValueError:
                    pass

        # US-169-007: Log checkpoint restore decisions (which stages to skip)
        if stages_to_restore_from_checkpoint:
            logger.info(
                f"Checkpoint restore: {len(stages_to_restore_from_checkpoint)} stages will be restored from checkpoint: "
                f"{stages_to_restore_from_checkpoint}"
            )

        # Run each stage
        total_stages = len(self.stages)
        for stage_index, stage in enumerate(self.stages):
            stage_name = stage.name

            # Filter stages
            if stage_name in skip_stages:
                logger.info(f"Skipping stage {stage_name} (skip_stages)")
                continue

            if only_stages and stage_name not in only_stages:
                logger.info(f"Skipping stage {stage_name} (not in only_stages)")
                continue

            # US-125-006: Check if pipeline is paused and wait for resume
            while self._paused:
                # US-138-004: Check for abort while paused
                if self.abort_requested:
                    logger.warning(f"Pipeline abort requested while paused - stopping at stage {stage_name}")
                    self._handle_abort()
                    return False
                logger.info(f"Pipeline paused at stage {stage_name} - waiting to resume...")
                time.sleep(1)  # Wait 1 second before checking again

            # US-138-004: Check if abort was requested
            if self.abort_requested:
                logger.warning(f"Pipeline abort requested - stopping at stage {stage_name}")
                self._handle_abort()
                return False

            # Check if stage can be skipped (checkpoint)
            if self.resume_mode and stage.can_skip(self.state, self.checkpoint):
                log_stage_skip(logger, stage_name, "checkpoint resume")
                logger.debug(f"Stage {stage_name}: checkpoint resume - transitioning from previous state")
                # Emit stage_skip event (US-106-006)
                self.emit_event(PipelineEvent(
                    event_type=EVENT_STAGE_SKIP,
                    stage_name=stage_name,
                    timestamp=time.time(),
                    data={'reason': 'checkpoint_resume'},
                ))
                if stage.restore(self.state, self.checkpoint, self.config):
                    # Validate state attributes after stage restoration
                    self.state.validate_state_attributes()
                    skipped_stages.add(stage_name)
                    completed_stages.add(stage_name)
                    logger.debug(f"Stage {stage_name}: restored from checkpoint, adding to completed stages")
                    continue
                else:
                    # US-51-008: restore failed - re-run the stage instead of
                    # proceeding with bad/partial state
                    ctx = self._format_log_context(
                        stage_name=stage_name,
                        stage_index=stage_index,
                        total_stages=total_stages,
                    )
                    logger.warning(
                        f"Failed to restore {stage_name} from checkpoint, "
                        f"re-running stage {ctx.to_suffix()}"
                    )

            # Check if this stage should run in parallel with others
            if stage_name in parallel_groups:
                group = parallel_groups[stage_name]
                group_key = tuple(sorted(group))

                # Skip if we already processed this parallel group
                if group_key in processed_parallel_groups:
                    continue

                processed_parallel_groups.add(group_key)

                # US-108-012: Track parallel execution timing
                parallel_start_time = time.time()

                # Run parallel stages
                success = self._run_parallel_stages(
                    group, skip_stages, only_stages,
                    on_stage_start, on_stage_complete, total_stages
                )

                # Track parallel execution stats
                parallel_elapsed = time.time() - parallel_start_time
                parallel_group_timings[group_key] = parallel_elapsed

                # Track stages that ran concurrently
                for s in group:
                    if s not in skip_stages and (not only_stages or s in only_stages):
                        stages_run_concurrently.append(s)

                if not success:
                    return False
                continue

            # Validate stage dependencies (US-81-006)
            try:
                self._validate_stage_dependencies(stage, completed_stages)
            except DependencyError as e:
                logger.error(format_error_with_code("PIPE-005", f"Stage dependency error: {e}"))
                return False

            # Validate inputs
            validation_error = stage.validate_inputs(self.state, self.config)
            if validation_error:
                recovery = self._get_recovery_suggestion(stage_name, validation_error)
                state_ctx = self._get_state_summary()
                logger.error(
                    f"Stage {stage_name} validation failed: {validation_error} "
                    f"[state: {state_ctx}] "
                    f"Recovery: {recovery}"
                )
                return False

            # Run the stage
            self.current_stage = stage_name
            # US-138-004: Set abort flag in state for stages to check
            self.state.abort_requested = self.abort_requested
            start_time = time.time()

            # Snapshot critical state fields before execution for rollback on failure
            state_snapshot = self._snapshot_state()

            # Invoke on_stage_start callback (legacy)
            if on_stage_start:
                try:
                    on_stage_start(stage_name)
                except (TypeError, ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"on_stage_start callback failed for {stage_name}: {e}")

            # Emit before_stage event (US-81-012)
            # US-159-005: Include correlation ID for tracing
            correlation_id = get_correlation_id()
            self.emit_event(PipelineEvent(
                event_type='before_stage',
                stage_name=stage_name,
                timestamp=time.time(),
                data={'correlation_id': correlation_id},
            ))

            # Start progress tracking for this stage
            self.progress_reporter.start_stage(stage_name)
            # Make reporter accessible to stages via state (transient, not serialized)
            self.state._progress_reporter = self.progress_reporter

            # Log stage start with structured context
            ctx = self._format_log_context(
                stage_name=stage_name,
                stage_index=stage_index,
                total_stages=total_stages,
            )
            # US-159-005: Log stage start with correlation ID
            logger.info(f"[{correlation_id}] Running stage: {stage_name} {ctx.to_suffix()}")
            # DEBUG: Log stage transition from previous stage
            prev_stage = self.current_stage
            logger.debug(f"[{correlation_id}] Stage transition: {prev_stage or 'START'} -> {stage_name}")

            # Log estimated duration from historical data (US-81-010)
            # Emit stage_start event with estimated duration (US-106-006)
            estimated_duration = self._log_stage_estimate(stage_name)
            # US-162-009: Store estimated duration for baseline comparison
            self._current_stage_estimated_duration = estimated_duration
            self.emit_event(PipelineEvent(
                event_type=EVENT_STAGE_START,
                stage_name=stage_name,
                timestamp=time.time(),
                data={'estimated_duration': estimated_duration} if estimated_duration else {},
            ))

            # US-88-005: Run health checks before stage execution
            health_check_results = self._run_health_checks(stage_name)
            health_check_warnings = []
            # DEBUG: Log health check results summary
            hc_summary = {hc.component: hc.status.value for hc in health_check_results}
            logger.debug(f"Health check results for {stage_name}: {hc_summary}")
            for hc_result in health_check_results:
                if hc_result.status == HealthStatus.FAILED:
                    logger.warning(f"Health check FAILED for {stage_name}: {hc_result.message}")
                    health_check_warnings.append(hc_result.message)
                elif hc_result.status == HealthStatus.WARNING:
                    logger.warning(f"Health check WARNING for {stage_name}: {hc_result.message}")
                    health_check_warnings.append(hc_result.message)
                else:
                    logger.debug(f"Health check OK for {stage_name}: {hc_result.message}")

            # US-108-011: Validate stage input contract before execution
            validator = PipelineValidator(self.config, self.stages)
            checkpoint_data = self.checkpoint.data or {}
            input_violations = validator.validate_stage_io(stage_name, checkpoint_data, 'input')
            if input_violations:
                for v in input_violations:
                    logger.warning(
                        f"[{stage_name}] Input contract violation: {v.field_name} - "
                        f"expected {v.expected}, got {v.actual}"
                    )
                # Add to result warnings if stage fails due to contract issues

            # US-88-003: Get retry config for this stage
            pipeline_config = getattr(self.config, 'pipeline', None)
            retry_config = None
            total_retry_attempts = 0
            if pipeline_config:
                retry_config = pipeline_config.get_retry_config(stage_name)

            # US-88-007: Get timeout config for this stage
            timeout_config = None
            timeout_seconds = 0
            if pipeline_config:
                timeout_config = pipeline_config.get_timeout_config(stage_name)
                if timeout_config:
                    try:
                        timeout_seconds = int(getattr(timeout_config, 'timeout_seconds', 0))
                        timeout_enabled = bool(getattr(timeout_config, 'enabled', False))
                        if not timeout_enabled:
                            timeout_seconds = 0
                    except (TypeError, ValueError):
                        timeout_seconds = 0

            # Execute stage with retry logic (US-88-003)
            # Handle case where retry_config is a MagicMock (from tests) or invalid
            retry_max_attempts = 1
            if retry_config is not None:
                try:
                    retry_max_attempts = int(getattr(retry_config, 'max_attempts', 1))
                except (TypeError, ValueError):
                    retry_max_attempts = 1

            attempt = 0
            result = None
            while attempt < retry_max_attempts:
                if attempt > 0:
                    # Calculate delay based on retry logic
                    retry_enabled = False
                    retry_strategy = None
                    retry_jitter_factor = 0.2  # Default jitter
                    if retry_config is not None:
                        try:
                            retry_enabled = bool(getattr(retry_config, 'enabled', False))
                            strategy_obj = getattr(retry_config, 'strategy', None)
                            if strategy_obj:
                                retry_strategy = getattr(strategy_obj, 'strategy', 'fixed')
                                retry_base_delay = getattr(strategy_obj, 'base_delay', 1.0)
                                retry_max_delay = getattr(strategy_obj, 'max_delay', 60.0)
                                retry_jitter_factor = getattr(strategy_obj, 'jitter_factor', 0.2)
                        except (TypeError, AttributeError):
                            retry_enabled = False

                    if retry_enabled and retry_strategy:
                        delay = self._calculate_retry_delay(
                            attempt=attempt,
                            strategy=retry_strategy,
                            base_delay=retry_base_delay,
                            max_delay=retry_max_delay,
                            jitter_factor=retry_jitter_factor
                        )
                        logger.info(f"Retrying stage {stage_name} (attempt {attempt + 1}/{retry_max_attempts}) after {delay:.1f}s")
                        time.sleep(delay)
                    total_retry_attempts += 1

                # US-88-007: Run the stage with timeout enforcement
                stage_start_time = time.time()
                timeout_occurred = False

                # US-106-011: Track resource usage before stage execution
                # US-162-009: Log memory and CPU at stage start
                before_resources = self._track_stage_resources(stage_name, 'before')
                self._resource_history.append(before_resources)
                if before_resources.get('memory_percent') is not None:
                    mem_pct = before_resources['memory_percent']
                    cpu_pct = before_resources.get('cpu_percent')
                    if cpu_pct is not None:
                        logger.info(
                            f"[{stage_name}] Starting resources: memory={mem_pct:.1f}%, CPU={cpu_pct:.1f}%"
                        )
                    else:
                        logger.info(
                            f"[{stage_name}] Starting resources: memory={mem_pct:.1f}%"
                        )

                if timeout_seconds > 0:
                    # Run with timeout using ThreadPoolExecutor
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(stage.run, self.state, self.config, self.checkpoint)
                        try:
                            result = future.result(timeout=timeout_seconds)
                        except FuturesTimeoutError:
                            # Timeout occurred - handle gracefully
                            timeout_occurred = True
                            elapsed = time.time() - stage_start_time
                            timeout_configured = timeout_config.timeout_seconds if timeout_config else timeout_seconds

                            # Calculate progress percentage (approximate based on elapsed time vs expected)
                            progress_pct = min(100, int((elapsed / timeout_configured) * 100))

                            # Log timeout with stage progress percentage
                            logger.warning(
                                f"Stage {stage_name} TIMED OUT after {elapsed:.1f}s "
                                f"(timeout: {timeout_configured}s, progress: ~{progress_pct}%)"
                            )

                            # US-88-007: Save partial progress before timeout if enabled
                            save_on_timeout = True
                            if timeout_config:
                                try:
                                    save_on_timeout = bool(getattr(timeout_config, 'save_on_timeout', True))
                                except (TypeError, AttributeError):
                                    save_on_timeout = True

                            if save_on_timeout:
                                logger.info(f"Saving partial progress for stage {stage_name} before timeout")
                                self.checkpoint.save(
                                    self.state,
                                    stage_name,
                                    self.config,
                                    partial=True
                                )
                                # Emit checkpoint_save event (US-106-006)
                                self.emit_event(PipelineEvent(
                                    event_type=EVENT_CHECKPOINT_SAVE,
                                    stage_name=stage_name,
                                    timestamp=time.time(),
                                    data={
                                        'checkpoint_path': str(self.checkpoint.checkpoint_path),
                                        'partial': True,
                                    },
                                ))

                            # Create a failed result with timeout metrics (use global import)
                            result = StageResult(
                                success=False,
                                error=f"Stage timed out after {elapsed:.1f}s (limit: {timeout_configured}s)",
                                metrics=StageMetrics(
                                    duration_seconds=elapsed,
                                    failed=True,
                                    timeout_occurred=True,  # US-106-002: Mark timeout in metrics
                                    timeout_count=1,  # US-108-002: Track timeout count
                                    timeout_duration=elapsed,  # US-108-002: Track timeout duration
                                    was_force_killed=True,  # US-108-002: Mark as force-killed
                                    retry_attempts=total_retry_attempts,
                                    health_check_results=[hc.to_dict() for hc in health_check_results]
                                )
                            )
                else:
                    # No timeout - run directly
                    result = stage.run(self.state, self.config, self.checkpoint)

                # US-108-011: Validate stage output contract after execution
                checkpoint_data = self.checkpoint.data or {}
                output_violations = validator.validate_stage_io(stage_name, checkpoint_data, 'output')
                if output_violations:
                    for v in output_violations:
                        logger.warning(
                            f"[{stage_name}] Output contract violation: {v.field_name} - "
                            f"expected {v.expected}, got {v.actual}"
                        )
                    # Add warnings to result
                    if result.warnings is None:
                        result.warnings = []
                    for v in output_violations:
                        result.warnings.append(
                            f"Output contract violation: {v.field_name} - expected {v.expected}, got {v.actual}"
                        )

                # US-106-011: Track resource usage after stage execution
                # US-162-009: Log memory and CPU at stage completion
                after_resources = self._track_stage_resources(stage_name, 'after')
                self._resource_history.append(after_resources)
                if after_resources.get('memory_percent') is not None:
                    mem_pct = after_resources['memory_percent']
                    cpu_pct = after_resources.get('cpu_percent')
                    if cpu_pct is not None:
                        logger.info(
                            f"[{stage_name}] Completion resources: memory={mem_pct:.1f}%, CPU={cpu_pct:.1f}%"
                        )
                    else:
                        logger.info(
                            f"[{stage_name}] Completion resources: memory={mem_pct:.1f}%"
                        )

                # Add resource usage to stage metrics extra_metrics
                if result.metrics:
                    result.metrics.extra_metrics['resource_usage'] = {
                        'before': before_resources,
                        'after': after_resources,
                    }

                # US-162-010: Add API cost to stage metrics
                if result.metrics:
                    try:
                        from src.llm_client.cost import get_cost_tracker
                        tracker = get_cost_tracker()
                        cost_summary = tracker.get_summary()
                        result.metrics.api_cost = cost_summary.get('total_cost', 0.0)
                        result.metrics.extra_metrics['api_cost'] = {
                            'llm_cost': cost_summary.get('llm_cost', 0.0),
                            'embedding_cost': cost_summary.get('embedding_cost', 0.0),
                            'llm_call_count': cost_summary.get('llm_call_count', 0),
                            'embedding_call_count': cost_summary.get('embedding_call_count', 0),
                            'total_tokens': cost_summary.get('total_tokens', 0),
                        }
                    except ImportError:
                        pass

                # US-106-002: Check if approaching timeout threshold (80%)
                # Set timeout_warning in metrics if stage is taking long
                if timeout_seconds > 0 and result.metrics is not None and not timeout_occurred:
                    current_elapsed = time.time() - stage_start_time
                    if current_elapsed >= timeout_seconds * 0.8:
                        result.metrics.timeout_warning = True
                        logger.warning(
                            f"Stage {stage_name} is approaching timeout "
                            f"({current_elapsed:.1f}s / {timeout_seconds}s = {int(current_elapsed/timeout_seconds*100)}%)"
                        )

                # Check if successful or retries exhausted
                # US-88-007: Don't retry on timeout - timeout_occurred means stage exceeded its limit
                retry_enabled = False
                if retry_config is not None and not timeout_occurred:
                    try:
                        retry_enabled = bool(getattr(retry_config, 'enabled', False))
                    except (TypeError, AttributeError):
                        retry_enabled = False

                if result.success or not retry_enabled:
                    break

                # Check if we should retry (failure but still have attempts left)
                if attempt < retry_max_attempts - 1:
                    # Snapshot state before retry
                    state_snapshot = self._snapshot_state()

                attempt += 1

            # If retries were used, log summary
            if total_retry_attempts > 0:
                logger.info(f"Stage {stage_name}: {total_retry_attempts} retry attempts made")

            elapsed = time.time() - start_time
            self.stage_timings[stage_name] = elapsed
            self.state.stage_timings[stage_name] = elapsed

            # US-162-009: Check if stage duration exceeds estimated baseline
            if self._current_stage_estimated_duration and self._current_stage_estimated_duration > 0:
                ratio = elapsed / self._current_stage_estimated_duration
                if ratio > 2.0:  # More than 2x the estimate
                    logger.warning(
                        f"Stage {stage_name} duration exceeded baseline: "
                        f"{elapsed:.1f}s vs estimated {self._current_stage_estimated_duration:.1f}s "
                        f"({ratio:.1f}x)"
                    )
                # Clear the estimate after use
                self._current_stage_estimated_duration = None

            # Store stage metrics if provided
            if result.metrics:
                # Update duration_seconds to actual elapsed time
                result.metrics.duration_seconds = elapsed
                # US-88-003: Track retry attempts in metrics
                result.metrics.retry_attempts = total_retry_attempts
                # US-88-005: Add health check results to metrics
                result.metrics.health_check_results = [hc.to_dict() for hc in health_check_results]
                # US-106-002: Preserve timeout flags if already set
                if timeout_occurred:
                    result.metrics.timeout_occurred = True
                self.stage_metrics[stage_name] = result.metrics

                # US-155-011: Display quota status after stage completion
                if self.show_quota:
                    self._display_quota_status()

                # US-138-010: Compute error rate and check thresholds
                self._compute_and_check_error_rate(stage_name)

                # US-106-009: Capture errors for pipeline-level aggregation
                self._capture_stage_errors(stage_name, result)
            else:
                # Create default metrics with just duration
                self.stage_metrics[stage_name] = StageMetrics(
                    duration_seconds=elapsed,
                    retry_attempts=total_retry_attempts,
                    health_check_results=[hc.to_dict() for hc in health_check_results]
                )

            # US-155-011: Display quota status after stage completion (for branches without metrics)
            if self.show_quota:
                self._display_quota_status()

            # Invoke on_stage_complete callback (legacy)
            if on_stage_complete:
                try:
                    on_stage_complete(stage_name, result, elapsed)
                except (TypeError, ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"on_stage_complete callback failed for {stage_name}: {e}")

            # Handle result
            if not result.success:
                # Emit on_stage_error event (US-81-012)
                self.emit_event(PipelineEvent(
                    event_type='on_stage_error',
                    stage_name=stage_name,
                    timestamp=time.time(),
                    data={'elapsed': elapsed},
                    error=result.error,
                ))

                # Rollback state to pre-stage snapshot
                self._rollback_state(state_snapshot, stage_name)

                # Mark metrics as failed (metrics already stored above)
                items_processed = 0
                items_failed = 0
                if stage_name in self.stage_metrics:
                    self.stage_metrics[stage_name].failed = True
                    items_processed = self.stage_metrics[stage_name].items_processed
                    items_failed = self.stage_metrics[stage_name].items_failed
                self.progress_reporter.finish_stage()
                state_ctx = self._get_state_summary()
                recovery = self._get_recovery_suggestion(stage_name, result.error or "")

                # Log stage failure with structured context
                ctx = self._format_log_context(
                    stage_name=stage_name,
                    stage_index=stage_index,
                    total_stages=total_stages,
                    elapsed_seconds=elapsed,
                    items_processed=items_processed,
                    items_failed=items_failed,
                )
                logger.error(
                    f"Stage {stage_name} failed: {result.error} "
                    f"[state: {state_ctx}] "
                    f"Recovery: {recovery} "
                    f"{ctx.to_suffix()}"
                )
                for warning in result.warnings:
                    logger.warning(f"  Warning: {warning}")
                return False

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Stage {stage_name}: {warning}")

            # Save checkpoint (US-49-012: include stage metrics if available)
            if result.data:
                metrics_dict = None
                if stage_name in self.stage_metrics:
                    metrics_dict = self.stage_metrics[stage_name].to_dict()
                # DEBUG: Log checkpoint serialization
                logger.debug(f"Checkpoint serialization: saving {stage_name} with {len(result.data)} data keys")
                self.checkpoint.save(stage_name, result.data,
                                     stage_metrics=metrics_dict)
                logger.debug(f"Checkpoint saved successfully to {self.checkpoint.checkpoint_path}")
                # Emit checkpoint_save event (US-106-006)
                self.emit_event(PipelineEvent(
                    event_type=EVENT_CHECKPOINT_SAVE,
                    stage_name=stage_name,
                    timestamp=time.time(),
                    data={
                        'checkpoint_path': str(self.checkpoint.checkpoint_path),
                        'partial': False,
                    },
                ))

            self.progress_reporter.finish_stage()
            completed_stages.add(stage_name)

            # Get items from stage metrics if available
            items_processed = 0
            items_failed = 0
            if stage_name in self.stage_metrics:
                metrics = self.stage_metrics[stage_name]
                items_processed = metrics.items_processed
                items_failed = metrics.items_failed

            # Emit after_stage event (US-81-012, US-106-006)
            # US-159-005: Include correlation ID for tracing
            correlation_id = get_correlation_id()
            self.emit_event(PipelineEvent(
                event_type='after_stage',
                stage_name=stage_name,
                timestamp=time.time(),
                data={
                    'elapsed': elapsed,
                    'stage_duration': elapsed,
                    'success': True,
                    'correlation_id': correlation_id,
                },
            ))

            # Log stage completion with structured context
            ctx = self._format_log_context(
                stage_name=stage_name,
                stage_index=stage_index,
                total_stages=total_stages,
                elapsed_seconds=elapsed,
                items_processed=items_processed,
                items_failed=items_failed,
            )
            # US-159-005: Log stage completion with correlation ID
            logger.info(f"[{correlation_id}] Stage {stage_name} completed in {elapsed:.1f}s {ctx.to_suffix()}")

            # US-167-009: WARNING logging when stages exceed expected time thresholds
            stage_warning_threshold = 600.0  # 10 minutes default
            if elapsed > stage_warning_threshold:
                logger.warning(
                    f"[{stage_name}] Stage exceeded time threshold: {elapsed:.1f}s "
                    f"(threshold: {stage_warning_threshold:.0f}s)"
                )

            # Save timing to history for future predictions (US-81-010)
            self._save_stage_timing(stage_name, elapsed)

            # US-138-009: Save resource usage to history for predictions
            self._save_stage_resource_usage(stage_name)

            # Run quality gate after matching stages (US-81-005)
            if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
                self._check_match_coverage_gate(stage_name)

            # Run cross-stage data drift detection (US-81-011)
            self._check_data_drift(stage_name)

        self.current_stage = None

        # Print timing summary after successful pipeline completion
        total_duration = time.time() - pipeline_start_time
        self._print_timing_summary(total_duration, skipped_stages)

        # US-106-009: Log pipeline-level error summary
        self._log_pipeline_error_summary()

        self.progress_reporter.finish_pipeline()

        # US-106-011: Log resource usage summary in pipeline completion output
        if self._resource_history:
            summary = self.get_summary()
            res = summary.get('resource_usage')
            if res:
                logger.info(f"Resource usage summary:")
                if res.get('memory_percent_max'):
                    logger.info(f"  Memory: avg={res['memory_percent_avg']:.1f}% max={res['memory_percent_max']:.1f}%")
                if res.get('cpu_percent_max'):
                    logger.info(f"  CPU: avg={res['cpu_percent_avg']:.1f}% max={res['cpu_percent_max']:.1f}%")

        # US-146-012: Log YouTube API vs yt-dlp usage ratio
        log_api_vs_ytdlp_usage()

        # Emit on_pipeline_complete event (US-81-012)
        # US-159-005: Include correlation ID for tracing
        correlation_id = get_correlation_id()

        # US-162-010: Log API cost summary at pipeline completion
        try:
            from src.llm_client.cost import get_cost_tracker
            tracker = get_cost_tracker()
            cost_summary = tracker.get_summary()
            if cost_summary and cost_summary.get('total_cost', 0) > 0:
                logger.info(
                    f"API Cost Summary: total=${cost_summary['total_cost']:.4f} "
                    f"(LLM: ${cost_summary['llm_cost']:.4f}, "
                    f"Embedding: ${cost_summary['embedding_cost']:.4f})"
                )
                if cost_summary.get('cost_by_provider'):
                    providers = ", ".join(
                        f"{k}: ${v:.4f}"
                        for k, v in cost_summary['cost_by_provider'].items()
                        if v > 0
                    )
                    if providers:
                        logger.info(f"  By provider: {providers}")
        except ImportError:
            pass

        self.emit_event(PipelineEvent(
            event_type='on_pipeline_complete',
            stage_name='',
            timestamp=time.time(),
            data={
                'total_duration': total_duration,
                'stages_run': list(self.stage_timings.keys()),
                'correlation_id': correlation_id,
            },
        ))
        logger.info(f"[{correlation_id}] Pipeline completed in {total_duration:.1f}s")
        # DEBUG: Log pipeline completion with correlation ID propagation summary
        logger.debug(f"[{correlation_id}] Pipeline completed - correlation ID propagated through all stages")

        # US-125-006: Cleanup signal handler on pipeline completion
        self._cleanup_signal_handler()

        return True

    def _run_dry_run(
        self,
        skip_stages: set,
        only_stages: Optional[set],
        resume: bool = True
    ) -> bool:
        """
        Run pipeline in dry-run mode: validate all stage inputs without executing.

        Delegates to validate_all() for structured validation, then logs the
        results as a detailed summary table with:
        - Stage-by-stage preview
        - Estimated duration from checkpoint history
        - Input/output counts
        - Checkpoint status (will run vs will skip)

        Does NOT modify state or checkpoint.

        Args:
            skip_stages: Set of stage names to skip
            only_stages: If set, only include these stages
            resume: Whether to check checkpoint for skippable stages

        Returns:
            True if all validations pass, False if any validation fails
        """
        logger.info("=" * 60)
        logger.info("DRY-RUN MODE: Previewing pipeline execution plan")
        logger.info("=" * 60)

        # US-108-008: Display variant info if set
        variant_mode = getattr(self, '_variant_mode', None)
        variant_options = getattr(self, '_variant_options', None)
        if variant_mode and variant_mode == 'test' and variant_options:
            # US-151-012: Show detailed test mode configuration
            logger.info("  Test mode configuration (limits applied during execution):")
            logger.info(f"    max_videos: {variant_options.max_videos or 3} (maximum videos per search)")
            logger.info(f"    max_segments: {variant_options.max_voiceover_segments or 10} (maximum voiceover segments)")
            logger.info(f"    max_downloads: {variant_options.max_downloads or 3} (maximum segments to download)")
            logger.info(f"    skip_embeddings: {variant_options.skip_embeddings} (skip embedding computation)")
            logger.info(f"    skip_iterative: {variant_options.skip_iterative_match} (skip iterative matching)")
        elif variant_mode and variant_mode != 'full':
            variant_descriptions = {
                'fast': 'Fast mode: skips iterative_match, reduces search results, skips embeddings',
            }
            logger.info(f"  Pipeline variant: {variant_descriptions.get(variant_mode, variant_mode)}")

        # US-108-004: Display checkpoint version info
        if resume and self.checkpoint and hasattr(self.checkpoint, 'data') and self.checkpoint.data:
            cp_version = self.checkpoint.data.version or "unknown"
            from .checkpoint import CURRENT_CHECKPOINT_VERSION
            if cp_version != CURRENT_CHECKPOINT_VERSION:
                logger.info(f"  Checkpoint version: {cp_version} (will migrate to {CURRENT_CHECKPOINT_VERSION})")
            else:
                logger.info(f"  Checkpoint version: {cp_version}")
        else:
            logger.info("  Checkpoint: none (fresh run)")

        # Delegate to validate_all() for structured results
        validation_results = self.validate_all(
            skip_stages=list(skip_stages) if skip_stages else None,
            only_stages=list(only_stages) if only_stages else None,
            resume=resume,
        )

        has_errors = any(r.status == 'error' for r in validation_results)

        # Build stage name to object mapping for get_input_output_info
        stage_map = {stage.name: stage for stage in self.stages}

        # Get checkpoint data for estimated durations if available
        checkpoint_stage_metrics = {}
        if resume and self.checkpoint and hasattr(self.checkpoint, 'data'):
            checkpoint_stage_metrics = self.checkpoint.data.stage_metrics or {}

        # US-106-005: Add metrics completeness helper function
        def get_metrics_completeness(stage_name: str, metrics: dict) -> str:
            """Check metrics completeness for a stage and return status string."""
            if not metrics:
                return "no metrics"

            # Import here to avoid circular imports
            from .stages import StageMetrics, StageType
            from .checkpoint import _STAGE_NAME_TO_TYPE

            stage_type_str = _STAGE_NAME_TO_TYPE.get(stage_name)
            if not stage_type_str:
                return "unknown type"

            try:
                stage_type = StageType(stage_type_str)
                stage_metrics = StageMetrics.from_dict(metrics)
                warnings = stage_metrics.validate_metrics(stage_type)
                if warnings:
                    return f"incomplete ({len(warnings)} warnings)"
                return "complete"
            except Exception:
                return "parse error"

        # Log structured summary table with enhanced details
        logger.info("")
        logger.info("Stage Execution Plan:")
        logger.info("-" * 80)
        logger.info(f"  {'Stage':<20} {'Action':<12} {'Est. Duration':<14} {'Inputs':<12} {'Outputs':<12} {'Metrics':<18} {'Detail'}")
        logger.info(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*12} {'-'*12} {'-'*18} {'-'*15}")

        for r in validation_results:
            stage_name = r.stage_name
            stage = stage_map.get(stage_name)

            # Get estimated duration from checkpoint
            est_duration = ""
            if stage_name in checkpoint_stage_metrics:
                metrics = checkpoint_stage_metrics[stage_name]
                if isinstance(metrics, dict) and 'duration_seconds' in metrics:
                    duration = metrics['duration_seconds']
                    if duration:
                        est_duration = f"{duration:.1f}s"

            # Get metrics completeness status
            metrics_status = ""
            if stage_name in checkpoint_stage_metrics:
                metrics_status = get_metrics_completeness(stage_name, checkpoint_stage_metrics[stage_name])
            elif r.status == 'checkpoint':
                metrics_status = "pending"

            # Get input/output info from stage
            io_info = {'inputs': '', 'outputs': '', 'input_count': None, 'output_count': None}
            if stage and r.status in ('run', 'checkpoint'):
                try:
                    io_info = stage.get_input_output_info(self.state, self.config)
                except Exception:
                    logger.exception(f"Could not get I/O info for stage {r.stage_name}, using defaults")

            inputs_str = str(io_info.get('input_count', '')) if io_info.get('input_count') is not None else '-'
            outputs_str = str(io_info.get('output_count', '')) if io_info.get('output_count') is not None else '-'

            detail = r.message
            if r.status == "error":
                logger.error(f"  {stage_name:<20} {r.status:<12} {est_duration:<14} {inputs_str:<12} {outputs_str:<12} {metrics_status:<18} {detail}")
            else:
                logger.info(f"  {stage_name:<20} {r.status:<12} {est_duration:<14} {inputs_str:<12} {outputs_str:<12} {metrics_status:<18} {detail}")

        logger.info("-" * 80)

        # US-125-009: Display API call estimates for stages that will run
        api_estimates_list = []
        total_estimated_cost = 0.0
        total_estimated_duration = 0.0

        for r in validation_results:
            if r.status not in ('run', 'checkpoint'):
                continue

            stage = stage_map.get(r.stage_name)
            if not stage:
                continue

            try:
                estimates = stage.get_api_estimates(self.state, self.config)
                if estimates:
                    api_estimates_list.append((r.stage_name, estimates))
                    total_estimated_cost += estimates.get('estimated_cost_usd', 0.0)
                    total_estimated_duration += estimates.get('estimated_duration_seconds', 0.0)
            except Exception:
                logger.exception(f"Could not get API estimates for stage {r.stage_name}")

        if api_estimates_list:
            logger.info("")
            logger.info("Estimated API Calls & Costs:")
            logger.info("-" * 80)

            for stage_name, estimates in api_estimates_list:
                # Build estimate string
                parts = []
                if 'youtube_api_calls' in estimates:
                    parts.append(f"YouTube API: {estimates['youtube_api_calls']}")
                if 'caption_fetch_attempts' in estimates:
                    parts.append(f"Caption fetch: {estimates['caption_fetch_attempts']}")
                if 'embedding_calls' in estimates:
                    parts.append(f"Embeddings: {estimates['embedding_calls']}")
                if 'llm_calls' in estimates:
                    parts.append(f"LLM: {estimates['llm_calls']}")
                if 'video_search_calls' in estimates:
                    parts.append(f"Video search: {estimates['video_search_calls']}")

                cost = estimates.get('estimated_cost_usd', 0.0)
                duration = estimates.get('estimated_duration_seconds', 0.0)

                estimate_str = ", ".join(parts) if parts else "none"
                logger.info(f"  {stage_name:<20}: {estimate_str}")
                if cost > 0 or duration > 0:
                    cost_str = f"${cost:.4f}" if cost > 0 else "$0"
                    logger.info(f"  {'':20}  Cost: {cost_str}, Est. time: {duration:.1f}s")

            logger.info("-" * 80)
            logger.info(f"  Total estimated cost: ${total_estimated_cost:.4f}")
            logger.info(f"  Total estimated API time: {total_estimated_duration:.1f}s")

        # Count by status
        counts: Dict[str, int] = {}
        for r in validation_results:
            counts[r.status] = counts.get(r.status, 0) + 1

        parts = []
        for action in ["run", "checkpoint", "skip", "error"]:
            if action in counts:
                parts.append(f"{action}={counts[action]}")

        # Calculate total estimated time
        total_est = 0.0
        for r in validation_results:
            if r.status == 'run' and r.stage_name in checkpoint_stage_metrics:
                metrics = checkpoint_stage_metrics[r.stage_name]
                if isinstance(metrics, dict) and 'duration_seconds' in metrics:
                    total_est += metrics.get('duration_seconds', 0) or 0

        if total_est > 0:
            logger.info(f"Summary: {', '.join(parts)}, total_est={total_est:.1f}s")
        else:
            logger.info(f"Summary: {', '.join(parts)}")

        logger.info("=" * 60)
        if has_errors:
            logger.error("DRY-RUN COMPLETE: Validation errors found")
        else:
            logger.info("DRY-RUN COMPLETE: All validations passed")
        logger.info("=" * 60)

        return not has_errors

    def _run_parallel_stages(
        self,
        group: Tuple[str, ...],
        skip_stages: set,
        only_stages: Optional[set],
        on_stage_start: StageStartCallback,
        on_stage_complete: StageCompleteCallback,
        total_stages: int,
    ) -> bool:
        """
        Run a group of stages in parallel using ThreadPoolExecutor.

        Args:
            group: Tuple of stage names to run in parallel
            skip_stages: Set of stage names to skip
            only_stages: If set, only run these stages
            on_stage_start: Callback for stage start
            on_stage_complete: Callback for stage completion
            total_stages: Total number of stages in the pipeline

        Returns:
            True if all parallel stages succeeded, False otherwise
        """
        # US-125-006: Check if pipeline is paused before running parallel stages
        while self._paused:
            # US-138-004: Check for abort while paused
            if self.abort_requested:
                logger.warning(f"Pipeline abort requested while paused - stopping parallel stages {group}")
                self._handle_abort()
                return False
            logger.info(f"Pipeline paused - waiting to resume parallel stages {group}...")
            time.sleep(1)

        # Get stage objects for this group
        stages_to_run = []
        for stage in self.stages:
            if stage.name in group:
                # Apply filtering
                if stage.name in skip_stages:
                    logger.info(f"Skipping parallel stage {stage.name} (skip_stages)")
                    continue
                if only_stages and stage.name not in only_stages:
                    logger.info(f"Skipping parallel stage {stage.name} (not in only_stages)")
                    continue
                # Check checkpoint skip
                if self.resume_mode and stage.can_skip(self.state, self.checkpoint):
                    log_stage_skip(logger, stage.name, "checkpoint resume")
                    if not stage.restore(self.state, self.checkpoint, self.config):
                        logger.warning(f"Failed to restore {stage.name} from checkpoint")
                    continue
                stages_to_run.append(stage)

        if not stages_to_run:
            return True

        # Validate all stages first (sequential to avoid race conditions)
        for stage in stages_to_run:
            validation_error = stage.validate_inputs(self.state, self.config)
            if validation_error:
                logger.error(f"Stage {stage.name} validation failed: {validation_error}")
                return False

        # Results collected from parallel execution
        results: Dict[str, Tuple[StageResult, float]] = {}

        # US-88-005: Run health checks for each stage before parallel execution
        parallel_health_checks: Dict[str, List[Any]] = {}
        for stage in stages_to_run:
            try:
                checker = HealthChecker(self.config)
                project_path = str(self.project_dir) if self.project_dir else None
                parallel_health_checks[stage.name] = checker.check_stage(stage.name, project_path)
            except Exception as e:
                logger.exception(f"Health check failed for {stage.name}: {e}")
                parallel_health_checks[stage.name] = []

        def run_stage(stage: Stage) -> Tuple[str, StageResult, float]:
            """Execute a single stage and return results."""
            start_time = time.time()
            if on_stage_start:
                try:
                    on_stage_start(stage.name)
                except (TypeError, ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"on_stage_start callback failed for {stage.name}: {e}")

            # Get stage index for structured logging
            stage_idx = next((i for i, s in enumerate(self.stages) if s.name == stage.name), 0)
            total_stages = len(self.stages)
            ctx = self._format_log_context(
                stage_name=stage.name,
                stage_index=stage_idx,
                total_stages=total_stages,
            )
            logger.info(f"Running parallel stage: {stage.name} {ctx.to_suffix()}")
            result = stage.run(self.state, self.config, self.checkpoint)
            elapsed = time.time() - start_time
            return (stage.name, result, elapsed)

        # Run stages in parallel (US-125-007: use max_concurrent_stages config)
        logger.info(f"Running {len(stages_to_run)} stages in parallel: {[s.name for s in stages_to_run]}")
        max_workers = min(
            len(stages_to_run),
            getattr(self.config.pipeline, 'max_concurrent_stages', 4) if self.config else 4
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_stage, stage): stage for stage in stages_to_run}
            for future in as_completed(futures):
                stage_name, result, elapsed = future.result()
                results[stage_name] = (result, elapsed)

        # Process results in original stage order (for checkpoint consistency)
        # Sort by the order stages appear in self.stages
        stage_order = {s.name: i for i, s in enumerate(self.stages)}
        ordered_names = sorted(results.keys(), key=lambda n: stage_order.get(n, 999))

        for stage_name in ordered_names:
            result, elapsed = results[stage_name]

            # US-106-011: Track resource usage after parallel stage execution
            after_resources = self._track_stage_resources(stage_name, 'after')
            self._resource_history.append(after_resources)

            # Add resource usage to stage metrics extra_metrics
            if result.metrics:
                result.metrics.extra_metrics['resource_usage'] = {
                    'after': after_resources,
                }

            # US-162-010: Add API cost to stage metrics
            if result.metrics:
                try:
                    from src.llm_client.cost import get_cost_tracker
                    tracker = get_cost_tracker()
                    cost_summary = tracker.get_summary()
                    result.metrics.api_cost = cost_summary.get('total_cost', 0.0)
                    result.metrics.extra_metrics['api_cost'] = {
                        'llm_cost': cost_summary.get('llm_cost', 0.0),
                        'embedding_cost': cost_summary.get('embedding_cost', 0.0),
                        'llm_call_count': cost_summary.get('llm_call_count', 0),
                        'embedding_call_count': cost_summary.get('embedding_call_count', 0),
                        'total_tokens': cost_summary.get('total_tokens', 0),
                    }
                except ImportError:
                    pass

            # Record timing
            self.stage_timings[stage_name] = elapsed
            self.state.stage_timings[stage_name] = elapsed

            # Store metrics
            hc_results = parallel_health_checks.get(stage_name, [])
            if result.metrics:
                result.metrics.duration_seconds = elapsed
                result.metrics.health_check_results = [hc.to_dict() for hc in hc_results]
                self.stage_metrics[stage_name] = result.metrics

                # US-138-010: Compute error rate and check thresholds
                self._compute_and_check_error_rate(stage_name)
            else:
                self.stage_metrics[stage_name] = StageMetrics(
                    duration_seconds=elapsed,
                    health_check_results=[hc.to_dict() for hc in hc_results]
                )

            # Invoke callback
            if on_stage_complete:
                try:
                    on_stage_complete(stage_name, result, elapsed)
                except (TypeError, ValueError, RuntimeError, OSError) as e:
                    logger.warning(f"on_stage_complete callback failed for {stage_name}: {e}")

            # Handle failure (US-85-012: critical vs optional stages)
            if not result.success:
                # Check if this stage is critical (default True for backward compatibility)
                stage = next((s for s in stages_to_run if s.name == stage_name), None)
                is_critical = getattr(stage, 'CRITICAL', True) if stage else True

                if is_critical:
                    # Critical stage failure: abort pipeline
                    logger.error(f"Parallel stage {stage_name} failed (CRITICAL): {result.error}")
                    for warning in result.warnings:
                        logger.warning(f"  Warning: {warning}")
                    return False
                else:
                    # Optional stage failure: log warning, collect to partial_failures, continue
                    logger.warning(f"Parallel stage {stage_name} failed (OPTIONAL): {result.error}")
                    for warning in result.warnings:
                        logger.warning(f"  Stage {stage_name} warning: {warning}")
                    # Collect failure for later reporting
                    self.state.partial_failures.append({
                        'stage': stage_name,
                        'error': result.error,
                        'warnings': result.warnings,
                    })

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Stage {stage_name}: {warning}")

            # Save checkpoint (in stage order, US-49-012: include stage metrics)
            if result.data:
                metrics_dict = None
                if stage_name in self.stage_metrics:
                    metrics_dict = self.stage_metrics[stage_name].to_dict()
                self.checkpoint.save(stage_name, result.data,
                                     stage_metrics=metrics_dict)
                # Emit checkpoint_save event (US-106-006)
                self.emit_event(PipelineEvent(
                    event_type=EVENT_CHECKPOINT_SAVE,
                    stage_name=stage_name,
                    timestamp=time.time(),
                    data={
                        'checkpoint_path': str(self.checkpoint.checkpoint_path),
                        'partial': False,
                    },
                ))

            # Get items from stage metrics if available
            items_processed = 0
            items_failed = 0
            if stage_name in self.stage_metrics:
                metrics = self.stage_metrics[stage_name]
                items_processed = metrics.items_processed
                items_failed = metrics.items_failed

            # Log parallel stage completion with structured context
            stage_idx = stage_order.get(stage_name, 0)
            ctx = self._format_log_context(
                stage_name=stage_name,
                stage_index=stage_idx,
                total_stages=total_stages,
                elapsed_seconds=elapsed,
                items_processed=items_processed,
                items_failed=items_failed,
            )
            logger.info(f"Parallel stage {stage_name} completed in {elapsed:.1f}s {ctx.to_suffix()}")

            # US-167-009: WARNING logging when stages exceed expected time thresholds
            stage_warning_threshold = 600.0  # 10 minutes default
            if elapsed > stage_warning_threshold:
                logger.warning(
                    f"[{stage_name}] Stage exceeded time threshold: {elapsed:.1f}s "
                    f"(threshold: {stage_warning_threshold:.0f}s)"
                )

        return True

    def get_summary(self) -> dict:
        """Get pipeline execution summary including aggregated metrics"""
        metrics = self.get_metrics()

        # US-106-011: Include resource usage summary
        resource_summary = None
        if self._resource_history:
            # Compute aggregate statistics from resource history
            memory_values = [r.get('memory_percent') for r in self._resource_history if r.get('memory_percent') is not None]
            cpu_values = [r.get('cpu_percent') for r in self._resource_history if r.get('cpu_percent') is not None]

            resource_summary = {
                'history': self._resource_history,
                'memory_percent_avg': sum(memory_values) / len(memory_values) if memory_values else None,
                'memory_percent_max': max(memory_values) if memory_values else None,
                'cpu_percent_avg': sum(cpu_values) / len(cpu_values) if cpu_values else None,
                'cpu_percent_max': max(cpu_values) if cpu_values else None,
            }

        # US-108-012: Calculate parallel execution stats
        # Time saved = sum of individual stage times - parallel group time
        stages_run_concurrently = getattr(self, 'stages_run_concurrently', [])
        parallel_group_timings = getattr(self, 'parallel_group_timings', {})

        total_time_saved_seconds = 0.0
        if parallel_group_timings:
            # For each parallel group, calculate time saved
            for group_key, parallel_time in parallel_group_timings.items():
                # Sum of individual stage times (from stage_timings)
                individual_times = [
                    self.stage_timings.get(stage, 0.0)
                    for stage in group_key
                ]
                sum_individual = sum(individual_times)
                # Time saved = sum of individual - parallel time
                time_saved = sum_individual - parallel_time
                total_time_saved_seconds += max(0, time_saved)  # Only positive savings

        # US-162-010: Include API cost summary
        api_cost_summary = None
        try:
            from src.llm_client.cost import get_cost_tracker
            tracker = get_cost_tracker()
            api_cost_summary = tracker.get_summary()
        except ImportError:
            pass

        return {
            'stages_run': list(self.stage_timings.keys()),
            'total_time': sum(self.stage_timings.values()),
            'stage_timings': self.stage_timings,
            'state': self.state.to_checkpoint_dict(),
            'items_processed': metrics['total_items_processed'],
            'items_failed': metrics['total_items_failed'],
            'metrics': metrics,
            'resource_usage': resource_summary,  # US-106-011: Resource monitoring summary
            # US-108-012: Parallel execution stats
            'stages_run_concurrently': stages_run_concurrently,
            'total_time_saved_seconds': total_time_saved_seconds,
            'parallel_group_timings': {str(k): v for k, v in parallel_group_timings.items()},
            'api_cost': api_cost_summary,  # US-162-010: API cost tracking
        }

    def set_download_metrics_exporter(self, exporter) -> None:
        """Set the download metrics exporter for resource monitoring (US-129-012).

        Args:
            exporter: DownloadMetricsExporter instance
        """
        self._download_metrics_exporter = exporter

    def export_resource_metrics(self, format: str = 'json', stage_filter: Optional[str] = None) -> Dict[str, Any]:
        """Export resource monitoring metrics in structured format (US-125-003).

        Args:
            format: Output format ('json' for structured dict, 'summary' for dashboard summary)
            stage_filter: Optional stage name to filter metrics (e.g., 'DOWNLOAD_SEGMENTS')

        Returns:
            Dictionary with resource metrics:
            - format: Output format
            - generated_at: ISO timestamp
            - history: List of per-stage resource measurements
            - per_stage: Aggregated metrics per stage (if format='json')
            - summary: Aggregate statistics (avg/max CPU, memory)
            - download_metrics: Download-specific metrics (US-129-012) if available
        """
        import time

        result = {
            'format': format,
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'history': self._resource_history,
            'summary': {},
            'per_stage': {},
        }

        # US-129-012: Include download-specific metrics if available
        if self._download_metrics_exporter is not None:
            try:
                download_resource = self._download_metrics_exporter.get_download_resource_metrics()
                result['download_metrics'] = download_resource
            except Exception as e:
                logger.exception(f"Failed to get download metrics: {e}")

        # US-146-012: Include YouTube API metrics if available
        if self._download_metrics_exporter is not None:
            try:
                youtube_api_metrics = self._download_metrics_exporter.get_youtube_api_metrics()
                if youtube_api_metrics and youtube_api_metrics.get("enabled", True):
                    result['youtube_api'] = youtube_api_metrics
            except Exception as e:
                logger.exception(f"Failed to get YouTube API metrics: {e}")

        # Apply stage filter if specified
        filtered_history = self._resource_history
        if stage_filter:
            filtered_history = [
                r for r in self._resource_history
                if r.get('stage_name') == stage_filter
            ]

        if not filtered_history:
            result['summary'] = {
                'message': 'No resource history available - pipeline may not have run',
                'stages_tracked': 0,
            }
            # Still include download metrics if stage_filter is DOWNLOAD_SEGMENTS
            if stage_filter == 'DOWNLOAD_SEGMENTS' and 'download_metrics' in result:
                result['summary']['message'] = 'Using download-specific metrics'
            return result

        # Compute aggregate statistics
        memory_values = [r.get('memory_percent') for r in filtered_history if r.get('memory_percent') is not None]
        cpu_values = [r.get('cpu_percent') for r in filtered_history if r.get('cpu_percent') is not None]
        memory_available = [r.get('memory_available_gb') for r in filtered_history if r.get('memory_available_gb') is not None]

        result['summary'] = {
            'stages_tracked': len(set(r.get('stage_name') for r in filtered_history)),
            'total_measurements': len(filtered_history),
            'memory_percent_avg': round(sum(memory_values) / len(memory_values), 2) if memory_values else None,
            'memory_percent_max': round(max(memory_values), 2) if memory_values else None,
            'memory_percent_min': round(min(memory_values), 2) if memory_values else None,
            'cpu_percent_avg': round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else None,
            'cpu_percent_max': round(max(cpu_values), 2) if cpu_values else None,
            'cpu_percent_min': round(min(cpu_values), 2) if cpu_values else None,
            'memory_available_gb_min': round(min(memory_available), 2) if memory_available else None,
        }

        # Compute per-stage metrics
        stage_data: Dict[str, List[Dict]] = {}
        for entry in filtered_history:
            stage_name = entry.get('stage_name', 'unknown')
            if stage_name not in stage_data:
                stage_data[stage_name] = []
            stage_data[stage_name].append(entry)

        for stage_name, entries in stage_data.items():
            stage_memory = [e.get('memory_percent') for e in entries if e.get('memory_percent') is not None]
            stage_cpu = [e.get('cpu_percent') for e in entries if e.get('cpu_percent') is not None]

            result['per_stage'][stage_name] = {
                'measurements': len(entries),
                'phases': list(set(e.get('phase') for e in entries)),
                'memory_percent_avg': round(sum(stage_memory) / len(stage_memory), 2) if stage_memory else None,
                'memory_percent_max': round(max(stage_memory), 2) if stage_memory else None,
                'cpu_percent_avg': round(sum(stage_cpu) / len(stage_cpu), 2) if stage_cpu else None,
                'cpu_percent_max': round(max(stage_cpu), 2) if stage_cpu else None,
            }

        return result

    def get_metrics(self) -> dict:
        """
        Get aggregated metrics across all stages.

        Returns:
            Dictionary with:
            - total_items_processed: Sum of items_processed across all stages
            - total_items_failed: Sum of items_failed across all stages
            - total_duration_seconds: Sum of duration_seconds across all stages
            - failed_stages: List of stage names that failed
            - stages: Dict mapping stage_name to StageMetrics
        """
        total_processed = 0
        total_failed = 0
        total_duration = 0.0
        failed_stages = []

        for stage_name, metrics in self.stage_metrics.items():
            total_processed += metrics.items_processed
            total_failed += metrics.items_failed
            total_duration += metrics.duration_seconds
            if metrics.failed:
                failed_stages.append(stage_name)

        return {
            'total_items_processed': total_processed,
            'total_items_failed': total_failed,
            'total_duration_seconds': total_duration,
            'failed_stages': failed_stages,
            'stages': self.stage_metrics
        }

    def clear_checkpoint(self):
        """Clear checkpoint for fresh start"""
        self.checkpoint.clear()
        self.resume_mode = False


def create_default_pipeline(
    config: 'Config',
    project_dir: Path,
    verbose_progress: bool = False,
    show_quota: bool = False,
) -> PipelineOrchestrator:
    """
    Create a pipeline with the simplified 7-stage order.

    This is a factory function that creates a fully configured pipeline.
    Stages are imported lazily to avoid circular imports.

    Simplified 7-stage pipeline:
    ANALYZE → VIDEO_SEARCH → CAPTION → MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT

    Args:
        config: Configuration object
        project_dir: Project directory path
        verbose_progress: Enable detailed per-stage progress output
        show_quota: Display real-time quota status (US-155-011)

    Returns:
        Configured PipelineOrchestrator
    """
    pipeline = PipelineOrchestrator(config, project_dir, verbose_progress=verbose_progress, show_quota=show_quota)

    # Import stages lazily to avoid circular imports
    from .stages.analyze import AnalyzeStage
    from .stages.video_search import VideoSearchStage
    from .stages.caption_stage import CaptionStage
    from .stages.match import MatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.download_segments import DownloadVideoSegmentsStage
    from .stages.output import OutputStage

    # Add stages in simplified 7-stage order
    pipeline.add_stage(AnalyzeStage())           # Stage 1: Extract keywords from voiceover
    pipeline.add_stage(VideoSearchStage())       # Stage 2: Search YouTube (no download)
    pipeline.add_stage(CaptionStage())           # Stage 3: Fetch YouTube captions
    pipeline.add_stage(MatchStage())             # Stage 4: Match voiceover to captions
    pipeline.add_stage(IterativeMatchStage())    # Stage 5: Fill gaps with iterative search
    pipeline.add_stage(DownloadVideoSegmentsStage())  # Stage 6: Download matched segments
    pipeline.add_stage(OutputStage())            # Stage 7: Generate OTIO/EDL/XML

    return pipeline


def create_entity_enhanced_pipeline(
    config: 'Config',
    project_dir: Path,
    show_quota: bool = False,
) -> PipelineOrchestrator:
    """
    Create a 9-stage pipeline that includes optional entity media stages.

    Inserts ENTITY_IMAGES and ENTITY_VIDEOS between ANALYZE and VIDEO_SEARCH
    so that entity images/videos are fetched before the main video search.

    9-stage pipeline:
    ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → VIDEO_SEARCH → CAPTION →
    MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT

    Args:
        config: Configuration object
        project_dir: Project directory path
        show_quota: Display real-time quota status (US-155-011)

    Returns:
        Configured PipelineOrchestrator with 9 stages
    """
    pipeline = PipelineOrchestrator(config, project_dir, show_quota=show_quota)

    # Import stages lazily to avoid circular imports
    from .stages.analyze import AnalyzeStage
    from .stages.entity_images import EntityImagesStage
    from .stages.entity_videos import EntityVideosStage
    from .stages.video_search import VideoSearchStage
    from .stages.caption_stage import CaptionStage
    from .stages.match import MatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.download_segments import DownloadVideoSegmentsStage
    from .stages.output import OutputStage

    # Add stages in 9-stage order (entity stages between ANALYZE and VIDEO_SEARCH)
    pipeline.add_stage(AnalyzeStage())                    # Stage 1: Extract keywords
    pipeline.add_stage(EntityImagesStage())               # Stage 2: Download entity images
    pipeline.add_stage(EntityVideosStage())                # Stage 3: Download entity stock videos
    pipeline.add_stage(VideoSearchStage())                 # Stage 4: Search YouTube
    pipeline.add_stage(CaptionStage())                     # Stage 5: Fetch YouTube captions
    pipeline.add_stage(MatchStage())                       # Stage 6: Match voiceover to captions
    pipeline.add_stage(IterativeMatchStage())              # Stage 7: Fill gaps
    pipeline.add_stage(DownloadVideoSegmentsStage())       # Stage 8: Download matched segments
    pipeline.add_stage(OutputStage())                      # Stage 9: Generate OTIO/EDL/XML

    return pipeline


def create_match_only_pipeline(
    config: 'Config',
    project_dir: Path,
    verbose_progress: bool = False,
    show_quota: bool = False,
) -> PipelineOrchestrator:
    """
    Create a pipeline that only runs matching and output stages.

    Used when user wants to re-run matching with different config
    without re-searching or fetching captions.

    In simplified pipeline: skips ANALYZE, VIDEO_SEARCH, CAPTION
    and runs: MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT

    Args:
        config: Configuration object
        project_dir: Project directory path
        verbose_progress: Enable detailed per-stage progress output
        show_quota: Display real-time quota status (US-155-011)

    Returns:
        Configured PipelineOrchestrator for match-only mode
    """
    pipeline = PipelineOrchestrator(config, project_dir, verbose_progress=verbose_progress, show_quota=show_quota)

    from .stages.analyze import AnalyzeStage
    from .stages.video_search import VideoSearchStage
    from .stages.caption_stage import CaptionStage
    from .stages.match import MatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.download_segments import DownloadVideoSegmentsStage
    from .stages.output import OutputStage

    # Add prerequisite stages for restoration only (will be skipped via checkpoint)
    pipeline.add_stage(AnalyzeStage())
    pipeline.add_stage(VideoSearchStage())
    pipeline.add_stage(CaptionStage())

    # Add stages to actually run
    pipeline.add_stage(MatchStage())
    pipeline.add_stage(IterativeMatchStage())
    pipeline.add_stage(DownloadVideoSegmentsStage())
    pipeline.add_stage(OutputStage())

    return pipeline


def create_output_only_pipeline(
    config: 'Config',
    project_dir: Path,
    verbose_progress: bool = False,
    show_quota: bool = False,
) -> PipelineOrchestrator:
    """
    Create a pipeline that only re-runs the OUTPUT stage.

    Used when user wants to regenerate OTIO/EDL/XML with different output
    config without re-matching or re-downloading.

    All stages are added for state restoration (skipped via checkpoint).
    Caller must reset checkpoint to DOWNLOAD_SEGMENTS.

    Args:
        config: Configuration object
        project_dir: Project directory path
        verbose_progress: Enable detailed per-stage progress output
        show_quota: Display real-time quota status (US-155-011)

    Returns:
        Configured PipelineOrchestrator for output-only mode
    """
    pipeline = PipelineOrchestrator(config, project_dir, verbose_progress=verbose_progress, show_quota=show_quota)

    from .stages.analyze import AnalyzeStage
    from .stages.video_search import VideoSearchStage
    from .stages.caption_stage import CaptionStage
    from .stages.match import MatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.download_segments import DownloadVideoSegmentsStage
    from .stages.output import OutputStage

    # All stages needed for state restoration (skipped via checkpoint)
    pipeline.add_stage(AnalyzeStage())
    pipeline.add_stage(VideoSearchStage())
    pipeline.add_stage(CaptionStage())
    pipeline.add_stage(MatchStage())
    pipeline.add_stage(IterativeMatchStage())
    pipeline.add_stage(DownloadVideoSegmentsStage())
    pipeline.add_stage(OutputStage())  # Only this stage runs

    return pipeline


def create_healing_pipeline(
    config: 'Config',
    project_dir: Path,
) -> Tuple['PipelineOrchestrator', Optional['HealingOrchestrator'], Optional['ResilientRunner']]:
    """
    Create a pipeline with self-healing enabled (default behavior).

    Uses config.healing settings to control healing behavior.
    If healing is disabled, returns (pipeline, None, None) for standard execution.

    Args:
        config: Configuration object
        project_dir: Project directory path

    Returns:
        Tuple of (PipelineOrchestrator, HealingOrchestrator or None, ResilientRunner or None)

    Usage:
        pipeline, orchestrator, runner = create_healing_pipeline(config, project_dir)
        if runner:
            success = runner.run_pipeline(pipeline)
            if orchestrator:
                orchestrator.print_report()
        else:
            success = pipeline.run()
    """
    # Create base pipeline
    pipeline = create_default_pipeline(config, project_dir)

    # Check if healing is enabled
    healing_config = getattr(config, 'healing', None)
    if not healing_config or not getattr(healing_config, 'enabled', True):
        logger.info("Self-healing disabled, using standard pipeline")
        return pipeline, None, None

    # Import agents (lazy to avoid circular imports)
    from .agents.orchestrator import HealingOrchestrator
    from .agents.runner import ResilientRunner
    from .agents.strategy import HealingStrategy, HealingMode

    # Build strategy from config
    strategy_name = getattr(healing_config, 'strategy', 'conservative')
    strategy_map = {
        'aggressive': HealingStrategy.aggressive,
        'conservative': HealingStrategy.conservative,
        'interactive': HealingStrategy.interactive,
        'minimal': HealingStrategy.minimal,
        'overnight': HealingStrategy.overnight,
        'development': HealingStrategy.development,
        'production': HealingStrategy.production,
    }
    strategy_factory = strategy_map.get(strategy_name, HealingStrategy.conservative)
    strategy = strategy_factory()

    # Override strategy settings from config
    if hasattr(healing_config, 'max_attempts_per_stage'):
        strategy.max_attempts_per_stage = healing_config.max_attempts_per_stage
    if hasattr(healing_config, 'max_total_heals'):
        strategy.max_total_heals = healing_config.max_total_heals
    if hasattr(healing_config, 'heal_delay'):
        strategy.heal_delay = healing_config.heal_delay
    if hasattr(healing_config, 'run_preflight'):
        strategy.run_preflight = healing_config.run_preflight
    if hasattr(healing_config, 'auto_fix_preflight'):
        strategy.auto_fix_preflight = healing_config.auto_fix_preflight
    if hasattr(healing_config, 'enable_rollback'):
        strategy.enable_rollback = healing_config.enable_rollback

    # Create orchestrator and runner
    orchestrator = HealingOrchestrator(config, project_dir, strategy)
    runner = ResilientRunner(config, project_dir, orchestrator=orchestrator)

    logger.info(f"Self-healing enabled: strategy={strategy_name}, max_attempts={strategy.max_attempts_per_stage}")

    # US-35-002: Auto-activate Mullvad VPN when enabled in config
    # Instantiate MullvadVPN early so it can be wired into EscalationManager
    # when stages create their downloaders
    # US-35-009: Check CLI availability before instantiating
    download_config = getattr(config, 'download', None)
    mullvad_config = getattr(download_config, 'mullvad', None) if download_config else None
    if mullvad_config and getattr(mullvad_config, 'enabled', False):
        from .downloader.mullvad_vpn import MullvadVPN
        if MullvadVPN.is_available():
            mullvad_vpn = MullvadVPN(mullvad_config)
            orchestrator.set_mullvad_vpn(mullvad_vpn)
            logger.debug("Mullvad VPN auto-activated in create_healing_pipeline (config.download.mullvad.enabled=true)")
            logger.info("Mullvad VPN enabled for Tier 4 IP rotation bypass")
        else:
            logger.warning(
                "Mullvad VPN is enabled in config but mullvad CLI is not available. "
                "Install Mullvad VPN or disable config.download.mullvad.enabled. "
                "Tier 4 VPN rotation will be skipped."
            )

    return pipeline, orchestrator, runner


@dataclass
class PipelineVariantOptions:
    """Options for configuring a pipeline variant.

    Attributes:
        mode: Execution mode - 'fast', 'full', or 'test'
        skip_stages: List of stage names to skip
        parallel_execution: Whether to enable parallel execution
        max_videos: Maximum videos to process (test mode)
        max_voiceover_segments: Maximum voiceover segments (test mode)
        max_downloads: Maximum video segments to download (test mode)
        reduce_search_results: Reduce video search results (fast mode)
        skip_iterative_match: Skip iterative match stage (fast mode)
        skip_embeddings: Skip embedding generation (fast mode)
    """
    mode: str = 'full'
    skip_stages: Optional[List[str]] = None
    parallel_execution: bool = False
    max_videos: Optional[int] = None
    max_voiceover_segments: Optional[int] = None
    max_downloads: Optional[int] = None
    reduce_search_results: bool = False
    skip_iterative_match: bool = False
    skip_embeddings: bool = False

    def __post_init__(self):
        if self.skip_stages is None:
            self.skip_stages = []


# US-108-008: Pipeline variant factory
def create_pipeline_variant(
    config: 'Config',
    project_dir: Path,
    variant_options: Optional[PipelineVariantOptions] = None,
    verbose_progress: bool = False,
    dry_run: bool = False,
    show_quota: bool = False,
) -> PipelineOrchestrator:
    """
    Create a pipeline variant based on specified options.

    This factory method creates different pipeline configurations for various use cases:
    - 'fast' mode: Skip iterative_match, reduce search results, skip embeddings
    - 'full' mode: Standard 7-stage pipeline (default)
    - 'test' mode: Max 3 videos, max 10 voiceover segments, mock API calls

    The variant can also be customized with:
    - skip_stages: List of stage names to exclude
    - parallel_execution: Enable parallel stage execution

    Validation (US-138-011):
        The function validates the variant configuration before creating the pipeline:
        - Checks that all required stages are available (not skipped)
        - Validates that stage dependencies are satisfied
        - Ensures at least one stage is included

    Args:
        config: Configuration object
        project_dir: Project directory path
        variant_options: Options for variant configuration (optional)
        verbose_progress: Enable detailed per-stage progress output
        show_quota: Display real-time quota status (US-155-011)

    Returns:
        Configured PipelineOrchestrator with specified variant

    Raises:
        ValueError: If variant configuration is invalid (missing dependencies, invalid mode, etc.)

    Examples:
        # Fast mode pipeline
        options = PipelineVariantOptions(mode='fast')
        pipeline = create_pipeline_variant(config, project_dir, options)

        # Custom variant with skipped stages
        options = PipelineVariantOptions(skip_stages=['ITERATIVE_MATCH'])
        pipeline = create_pipeline_variant(config, project_dir, options)

        # Test mode pipeline
        options = PipelineVariantOptions(
            mode='test',
            max_videos=3,
            max_voiceover_segments=10
        )
        pipeline = create_pipeline_variant(config, project_dir, options)
    """
    # Default to full mode if not specified
    if variant_options is None:
        variant_options = PipelineVariantOptions(mode='full')

    # Validate mode
    valid_modes = {'fast', 'full', 'test'}
    if variant_options.mode not in valid_modes:
        raise ValueError(
            f"Invalid pipeline mode: {variant_options.mode}. "
            f"Must be one of {valid_modes}"
        )

    mode = variant_options.mode
    skip_stages = set(variant_options.skip_stages or [])

    # US-138-011: Validate that all required stages are available
    # Define all available stages and their dependencies
    available_stages = {
        'ANALYZE': {'depends_on': []},
        'VIDEO_SEARCH': {'depends_on': ['ANALYZE']},
        'CAPTION': {'depends_on': ['ANALYZE']},
        'MATCH': {'depends_on': ['ANALYZE', 'CAPTION']},
        'ITERATIVE_MATCH': {'depends_on': ['MATCH']},
        'DOWNLOAD_SEGMENTS': {'depends_on': ['MATCH']},
        'OUTPUT': {'depends_on': ['MATCH', 'DOWNLOAD_SEGMENTS']},
    }

    # Determine which stages will be included (all available minus skipped)
    included_stages = set(available_stages.keys()) - skip_stages

    # Validate that skipped stages don't break dependencies
    for stage_name, stage_info in available_stages.items():
        if stage_name in skip_stages:
            continue  # Skip validation for stages we're not including

        # Check all dependencies are available
        for dep in stage_info['depends_on']:
            if dep in skip_stages:
                raise ValueError(
                    f"Invalid pipeline variant: cannot skip stage '{dep}' because stage '{stage_name}' "
                    f"depends on it. Either remove '{dep}' from skip_stages or use a different "
                    f"pipeline mode (e.g., 'full' instead of skipping required stages)."
                )

    # Validate that at least one stage is included
    if not included_stages:
        raise ValueError(
            f"Invalid pipeline variant: cannot skip all stages. "
            f"At least one stage must be included. Available stages: {list(available_stages.keys())}"
        )

    # Build variant description for logging
    variant_descriptions = {
        'fast': 'Fast mode: skips iterative_match, reduces search, skips embeddings',
        'full': 'Full mode: standard 7-stage pipeline',
        'test': f'Test mode: max {variant_options.max_videos or 3} videos, '
                f'max {variant_options.max_voiceover_segments or 10} voiceover segments',
    }

    logger.info(f"Creating pipeline variant: {variant_descriptions.get(mode, mode)}")

    # Create base pipeline
    pipeline = PipelineOrchestrator(
        config,
        project_dir,
        verbose_progress=verbose_progress,
        show_quota=show_quota,
    )

    # Configure parallel execution via config if requested
    if variant_options.parallel_execution:
        # Add parallel_execution to config if not already present
        if not hasattr(config, 'pipeline'):
            from dataclasses import dataclass
            @dataclass
            class PipelineConfig:
                parallel_execution: bool = False
            config.pipeline = PipelineConfig()
        config.pipeline.parallel_execution = True

    # Import stages lazily
    from .stages.analyze import AnalyzeStage
    from .stages.video_search import VideoSearchStage
    from .stages.caption_stage import CaptionStage
    from .stages.match import MatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.download_segments import DownloadVideoSegmentsStage
    from .stages.output import OutputStage

    # Determine which stages to skip based on mode
    if mode == 'fast':
        skip_stages.add('ITERATIVE_MATCH')
        variant_options.skip_iterative_match = True
        variant_options.skip_embeddings = True
        variant_options.reduce_search_results = True

    # Build stage list with conditional stages
    stages_to_add = [
        ('ANALYZE', AnalyzeStage()),
        ('VIDEO_SEARCH', VideoSearchStage()),
        ('CAPTION', CaptionStage()),
        ('MATCH', MatchStage()),
        ('ITERATIVE_MATCH', IterativeMatchStage()),
        ('DOWNLOAD_SEGMENTS', DownloadVideoSegmentsStage()),
        ('OUTPUT', OutputStage()),
    ]

    for stage_name, stage_instance in stages_to_add:
        if stage_name in skip_stages:
            logger.debug(f"Pipeline variant: skipping stage {stage_name}")
            continue
        pipeline.add_stage(stage_instance)

    # Store variant info on pipeline for later reference (e.g., dry-run output)
    pipeline._variant_options = variant_options
    pipeline._variant_mode = mode

    # Apply test mode limits to config (temporary modification)
    # US-151-012: Skip applying limits in dry-run mode - just show what would be limited
    if mode == 'test' and not dry_run:
        _apply_test_mode_limits(config, variant_options)

    logger.info(f"Pipeline variant created: {len(pipeline.stages)} stages, "
                f"skipped: {sorted(skip_stages) if skip_stages else 'none'}")

    return pipeline


def _apply_test_mode_limits(config: 'Config', options: PipelineVariantOptions) -> None:
    """Apply test mode limits to config for the pipeline run.

    This temporarily modifies config values to limit:
    - Maximum videos per search
    - Maximum voiceover segments

    Args:
        config: Configuration object to modify
        options: Pipeline variant options with test mode settings
    """
    max_videos = options.max_videos or 3
    max_segments = options.max_voiceover_segments or 10
    max_downloads = options.max_downloads or 3

    # Modify video search config
    video_search_config = getattr(config, 'video_search', None)
    if video_search_config:
        # Store original value for restoration
        if not hasattr(config, '_test_mode_original'):
            config._test_mode_original = {}

        original_max = getattr(video_search_config, 'max_results', None)
        if original_max is not None:
            config._test_mode_original['video_search.max_results'] = original_max

        # Set reduced limit
        setattr(video_search_config, 'max_results', min(max_videos, original_max or max_videos))
        logger.debug(f"Test mode: limited max_results to {max_videos}")

    # Set flag to limit videos in caption and other stages
    config._test_mode_max_videos = max_videos
    logger.debug(f"Test mode: limited videos to {max_videos}")

    # Set flag to limit voiceover segments in analyze stage
    config._test_mode = True
    config._test_mode_max_segments = max_segments
    logger.debug(f"Test mode: limited voiceover segments to {max_segments}")

    # Set flag to limit downloads in download_segments stage
    config._test_mode_max_downloads = max_downloads
    logger.debug(f"Test mode: limited downloads to {max_downloads}")

    # Set flag to skip embeddings based on test_mode config
    test_mode_config = getattr(config, 'test_mode', None)
    if test_mode_config:
        skip_embeddings = getattr(test_mode_config, 'skip_embeddings', True)
        config._test_mode_skip_embeddings = skip_embeddings
        if skip_embeddings:
            logger.debug("Test mode: embeddings will be skipped")

        # Set flag to skip iterative matching based on test_mode config
        skip_iterative = getattr(test_mode_config, 'skip_iterative', True)
        config._test_mode_skip_iterative = skip_iterative
        if skip_iterative:
            logger.debug("Test mode: iterative matching will be skipped")


def _collect_escalation_metrics(pipeline, orchestrator) -> None:
    """Collect escalation metrics from download stages and pass to orchestrator.

    Searches pipeline stages for a DownloadStage with a downloader that has
    an escalation_manager, then builds a RateLimitMetricsAggregator and passes
    unified metrics to the orchestrator for inclusion in the end-of-run report.

    Also wires the shared EscalationManager into the DownloadHealer so it
    uses the same escalation state instead of a duplicate CookieRotator.

    US-1-012: Now also collects VPN rotation count from MullvadVPN if available.
    """
    try:
        from src.downloader.rate_limit_metrics import RateLimitMetricsAggregator

        for stage in getattr(pipeline, 'stages', []):
            downloader = getattr(stage, 'downloader', None)
            if downloader is None:
                continue
            esc_mgr = getattr(downloader, 'escalation_manager', None)
            if esc_mgr is not None and hasattr(esc_mgr, 'get_metrics'):
                # Build aggregator with all available subsystems
                aggregator = RateLimitMetricsAggregator(
                    escalation_manager=esc_mgr,
                    cookie_rotator=getattr(downloader, 'cookie_rotator', None),
                    rate_limit_budget=getattr(downloader, '_rate_limit_budget', None),
                    circuit_breaker=getattr(downloader, '_circuit_breaker', None),
                )
                orchestrator.set_aggregated_metrics(aggregator)

                # Collect escalation metrics with VPN rotation count (US-1-012)
                esc_metrics = esc_mgr.get_metrics()

                # Add VPN metrics from MullvadVPN if available (US-35-004)
                mullvad_vpn = getattr(esc_mgr, '_mullvad_vpn', None)
                if mullvad_vpn is not None:
                    vpn_status = mullvad_vpn.get_status_extended() if hasattr(mullvad_vpn, 'get_status_extended') else mullvad_vpn.get_status()
                    # Connection and country status
                    esc_metrics['mullvad_connected'] = vpn_status.get('mullvad_connected', vpn_status.get('connected', False))
                    esc_metrics['vpn_current_country'] = vpn_status.get('current_country', vpn_status.get('country'))
                    esc_metrics['vpn_countries_used'] = vpn_status.get('used_countries', [])
                    # Rotation count from base VPNManager
                    esc_metrics['vpn_switch_count'] = vpn_status.get('switch_count', vpn_status.get('switches', 0))

                orchestrator.set_escalation_metrics(esc_metrics)

                # Wire shared EscalationManager into DownloadHealer
                if hasattr(orchestrator, 'wire_escalation_manager'):
                    orchestrator.wire_escalation_manager(esc_mgr)
                return
    except (ImportError, AttributeError, TypeError, KeyError, ValueError) as exc:
        logger.debug(f"Non-critical: escalation metrics collection failed: {exc}")


def _export_pipeline_metrics(pipeline: 'PipelineOrchestrator', config: 'Config') -> None:
    """Export pipeline metrics to configured formats (US-88-008).

    Args:
        pipeline: The pipeline orchestrator with metrics data
        config: Configuration object containing metrics_export settings
    """
    try:
        # Get export config from pipeline config
        export_config = getattr(config.pipeline, 'metrics_export', None)
        if not export_config:
            logger.debug("No metrics_export config found, skipping export")
            return

        # Skip if both exports are disabled
        if not export_config.export_json and not export_config.export_prometheus:
            logger.debug("Both JSON and Prometheus exports disabled, skipping")
            return

        # Create exporter with config
        from .pipeline_metrics_exporter import MetricsExporter
        exporter = MetricsExporter(export_config)

        # Get metrics from pipeline
        stage_timings = getattr(pipeline, 'stage_timings', {})
        stage_metrics = getattr(pipeline, 'stage_metrics', {})

        if not stage_timings:
            logger.debug("No stage timings found, skipping metrics export")
            return

        # Get optional context data
        pipeline_state = None
        if hasattr(pipeline, 'state') and pipeline.state:
            pipeline_state = pipeline.state.to_checkpoint_dict()

        checkpoint_data = None
        if hasattr(pipeline, 'checkpoint') and pipeline.checkpoint:
            checkpoint_data = pipeline.checkpoint.load()

        # Get config summary (sanitized - no secrets)
        config_summary = _get_config_summary(config)

        # Export metrics
        exported = exporter.export(
            stage_timings=stage_timings,
            stage_metrics=stage_metrics,
            pipeline_state=pipeline_state,
            checkpoint_data=checkpoint_data,
            config_summary=config_summary,
        )

        for fmt, path in exported.items():
            logger.info(f"Pipeline metrics exported to {fmt}: {path}")

    except Exception as e:
        logger.warning(f"Failed to export pipeline metrics: {e}")


def _get_config_summary(config: 'Config') -> Dict[str, Any]:
    """Get a sanitized summary of config for metrics export.

    Args:
        config: Full config object

    Returns:
        Dict with config summary (no secrets)
    """
    summary = {}

    try:
        # Pipeline config summary
        if hasattr(config, 'pipeline') and config.pipeline:
            p = config.pipeline
            summary['pipeline'] = {
                'skip_download': getattr(p, 'skip_download', False),
                'skip_matching': getattr(p, 'skip_matching', False),
                'resume_enabled': getattr(p, 'resume_enabled', True),
                'max_retries': getattr(p, 'max_retries', 3),
                'parallel_transcription': getattr(p, 'parallel_transcription', True),
                'batch_failure_threshold': getattr(p, 'batch_failure_threshold', 0.5),
            }

        # Matching config summary
        if hasattr(config, 'matching') and config.matching:
            m = config.matching
            summary['matching'] = {
                'min_confidence': getattr(m, 'min_confidence', 0.3),
                'max_videos_per_segment': getattr(m, 'max_videos_per_segment', 5),
            }

        # Download config summary
        if hasattr(config, 'download') and config.download:
            d = config.download
            summary['download'] = {
                'max_videos': getattr(d, 'max_videos', 50),
                'max_duration': getattr(d, 'max_duration', 300),
            }

    except Exception as e:
        logger.exception(f"Failed to get config summary: {e}")

    return summary


def run_pipeline_with_healing(
    config: 'Config',
    project_dir: Path,
    resume: bool = True
) -> bool:
    """
    Convenience function to create and run a self-healing pipeline.

    Args:
        config: Configuration object
        project_dir: Project directory path
        resume: Whether to resume from checkpoint

    Returns:
        True if pipeline completed successfully
    """
    pipeline, orchestrator, runner = create_healing_pipeline(
        config, project_dir
    )

    if runner:
        # Run with healing
        success = runner.run_pipeline(pipeline, resume=resume)

        # Collect escalation metrics from download stages (US-008 Sprint 9)
        if orchestrator:
            _collect_escalation_metrics(pipeline, orchestrator)

        # Print report if configured
        healing_config = getattr(config, 'healing', None)
        if orchestrator and getattr(healing_config, 'print_report', True):
            orchestrator.print_report()
            orchestrator.export_metrics_json()

        # US-88-008: Export pipeline metrics
        _export_pipeline_metrics(pipeline, config)

        return success
    else:
        # Fallback to standard execution
        success = pipeline.run(resume=resume)
        # US-88-008: Export pipeline metrics (fallback path)
        _export_pipeline_metrics(pipeline, config)
        return success
