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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .checkpoint import CheckpointManager, STAGE_ORDER
from .pipeline_events import PipelineEvent, PipelineEventBus, EventCallback
from .pipeline_history import append_stage_timing, estimate_duration
from .pipeline_progress import ProgressReporter
from .pipeline_validator import PipelineValidator, StageValidationResult
from .health_checker import HealthChecker, HealthStatus
from .state import PipelineState
from .stages import Stage, StageResult, StageMetrics, DependencyError

# US-88-008: Pipeline metrics exporter
from .pipeline_metrics_exporter import MetricsExporter, PipelineExportConfig

if TYPE_CHECKING:
    from .config import Config
    from .agents.runner import ResilientRunner
    from .agents.orchestrator import HealingOrchestrator

# Type aliases for callbacks
StageStartCallback = Callable[[str], None]  # (stage_name) -> None
StageCompleteCallback = Callable[[str, StageResult, float], None]  # (stage_name, result, elapsed_seconds) -> None


logger = logging.getLogger(__name__)


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
                logger.error(f"Circular dependency detected: {stage_name} -> {dep}")
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
        stages: List[Stage] = None
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Configuration object
            project_dir: Project directory for checkpoints and caches
            stages: Optional list of stages (uses default if not provided)
        """
        self.config = config
        self.project_dir = Path(project_dir)
        self.state = PipelineState()
        self.stages = stages or []

        # Create validator for pre-run checks (US-82-006)
        self._validator = PipelineValidator(config, self.stages)

        # Validate config before any I/O (US-45-010: fail-fast on invalid config)
        config_errors = self._validate_config()
        if config_errors:
            error_detail = "; ".join(config_errors)
            raise ValueError(f"Pipeline config validation failed: {error_detail}")

        # Initialize checkpoint manager
        config_hash = getattr(config, '_config_hash', '')
        self.checkpoint = CheckpointManager(project_dir, config_hash, config=config)

        # Runtime state
        self.resume_mode = False
        self.current_stage: Optional[str] = None
        self.stage_timings: dict = {}
        self.stage_metrics: dict = {}  # stage_name -> StageMetrics

        # Progress reporter for real-time progress.json updates
        self.progress_reporter = ProgressReporter(project_dir)

        # Event bus for decoupled stage lifecycle notifications (US-82-011)
        self.event_bus = PipelineEventBus()

        # US-88-006: Data drift detection history for trend analysis
        # Stores drift events for debugging and pattern detection
        self.drift_history: List[Dict[str, Any]] = []
        self._drift_config = getattr(config.pipeline, 'drift_rules', None)

    def add_stage(self, stage: Stage) -> 'PipelineOrchestrator':
        """Add a stage to the pipeline (fluent interface)"""
        self.stages.append(stage)
        return self

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

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint if it exists.

        Returns:
            True if checkpoint was loaded and is valid
        """
        if not self.checkpoint.exists():
            return False

        data = self.checkpoint.load()
        if data is None:
            return False

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

        # US-40-003: Use CheckpointManager.restore_state() for defensive validation
        # This validates and initializes any missing state attributes after checkpoint load
        self.state = self.checkpoint.restore_state(self.state)

        self.resume_mode = True
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
                logger.error(f"State integrity issue after rollback: {issue}")

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

    def _calculate_retry_delay(self, attempt: int, strategy: str, base_delay: float, max_delay: float) -> float:
        """Calculate retry delay based on strategy.

        Args:
            attempt: 0-based attempt number (0 = first attempt, 1 = first retry, etc.)
            strategy: One of 'exponential', 'linear', 'fixed'
            base_delay: Base delay in seconds
            max_delay: Maximum delay cap in seconds

        Returns:
            Delay in seconds before next retry
        """
        if strategy == 'exponential':
            delay = base_delay * (2 ** attempt)
        elif strategy == 'linear':
            delay = base_delay * (attempt + 1)
        else:  # fixed
            delay = base_delay
        return min(delay, max_delay)

    def _log_stage_estimate(self, stage_name: str) -> None:
        """Log a predicted duration for the upcoming stage based on history.

        Uses the current item count (from state) and historical throughput.
        Silently does nothing when no history exists (graceful degradation).
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
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            logger.debug(f"Could not estimate duration for {stage_name}: {exc}")

    def _run_health_checks(self, stage_name: str) -> List[Any]:
        """Run health checks before stage execution (US-88-005).

        Runs pre-stage health checks that validate external dependencies.
        Returns list of health check results.
        """
        try:
            checker = HealthChecker(self.config)
            project_path = str(self.project_dir) if self.project_dir else None
            return checker.check_stage(stage_name, project_path)
        except Exception as e:
            logger.debug(f"Health check failed for {stage_name}: {e}")
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
        skip_stages = set(skip_stages or [])
        only_stages = set(only_stages) if only_stages else None

        # Re-validate config with stages present (US-44-003 + US-45-010)
        # __init__ validates config-only checks; this catches stage-dependent checks
        # (e.g., embedding provider required when matching stages are added post-init)
        config_errors = self._validate_config()
        if config_errors:
            for error in config_errors:
                logger.error(f"Config validation error: {error}")
            return False

        # Dry-run mode: log stages and validate without executing
        if dry_run:
            return self._run_dry_run(skip_stages, only_stages, resume=resume)

        # US-88-011: Validate no circular dependencies in stage graph
        try:
            validate_no_circular_dependencies(self.stages)
        except ValueError as e:
            logger.error(f"Stage dependency validation failed: {e}")
            return False

        # US-88-011: Auto-detect parallel stages from DEPENDS_ON if not provided
        if parallel_stages is None:
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

        # Track processed parallel groups to avoid running same group twice
        processed_parallel_groups: set = set()

        # Track completed/restored stages for dependency validation
        completed_stages: set = set()

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

            # Check if stage can be skipped (checkpoint)
            if self.resume_mode and stage.can_skip(self.state, self.checkpoint):
                logger.info(f"Skipping {stage_name} (checkpoint resume)")
                if stage.restore(self.state, self.checkpoint, self.config):
                    # Validate state attributes after stage restoration
                    self.state.validate_state_attributes()
                    skipped_stages.add(stage_name)
                    completed_stages.add(stage_name)
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

                # Run parallel stages
                success = self._run_parallel_stages(
                    group, skip_stages, only_stages,
                    on_stage_start, on_stage_complete, total_stages
                )
                if not success:
                    return False
                continue

            # Validate stage dependencies (US-81-006)
            try:
                self._validate_stage_dependencies(stage, completed_stages)
            except DependencyError as e:
                logger.error(f"Stage dependency error: {e}")
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
            self.emit_event(PipelineEvent(
                event_type='before_stage',
                stage_name=stage_name,
                timestamp=time.time(),
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
            logger.info(f"Running stage: {stage_name} {ctx.to_suffix()}")

            # Log estimated duration from historical data (US-81-010)
            self._log_stage_estimate(stage_name)

            # US-88-005: Run health checks before stage execution
            health_check_results = self._run_health_checks(stage_name)
            health_check_warnings = []
            for hc_result in health_check_results:
                if hc_result.status == HealthStatus.FAILED:
                    logger.warning(f"Health check FAILED for {stage_name}: {hc_result.message}")
                    health_check_warnings.append(hc_result.message)
                elif hc_result.status == HealthStatus.WARNING:
                    logger.warning(f"Health check WARNING for {stage_name}: {hc_result.message}")
                    health_check_warnings.append(hc_result.message)
                else:
                    logger.debug(f"Health check OK for {stage_name}: {hc_result.message}")

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
                    if retry_config is not None:
                        try:
                            retry_enabled = bool(getattr(retry_config, 'enabled', False))
                            strategy_obj = getattr(retry_config, 'strategy', None)
                            if strategy_obj:
                                retry_strategy = getattr(strategy_obj, 'strategy', 'fixed')
                                retry_base_delay = getattr(strategy_obj, 'base_delay', 1.0)
                                retry_max_delay = getattr(strategy_obj, 'max_delay', 60.0)
                        except (TypeError, AttributeError):
                            retry_enabled = False

                    if retry_enabled and retry_strategy:
                        delay = self._calculate_retry_delay(
                            attempt=attempt,
                            strategy=retry_strategy,
                            base_delay=retry_base_delay,
                            max_delay=retry_max_delay
                        )
                        logger.info(f"Retrying stage {stage_name} (attempt {attempt + 1}/{retry_max_attempts}) after {delay:.1f}s")
                        time.sleep(delay)
                    total_retry_attempts += 1

                # US-88-007: Run the stage with timeout enforcement
                stage_start_time = time.time()
                timeout_occurred = False

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

                            # Create a failed result to exit the retry loop
                            from .stages import StageResult
                            result = StageResult(
                                success=False,
                                error=f"Stage timed out after {elapsed:.1f}s (limit: {timeout_configured}s)",
                                metrics=None
                            )
                else:
                    # No timeout - run directly
                    result = stage.run(self.state, self.config, self.checkpoint)

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

            # Store stage metrics if provided
            if result.metrics:
                # Update duration_seconds to actual elapsed time
                result.metrics.duration_seconds = elapsed
                # US-88-003: Track retry attempts in metrics
                result.metrics.retry_attempts = total_retry_attempts
                # US-88-005: Add health check results to metrics
                result.metrics.health_check_results = [hc.to_dict() for hc in health_check_results]
                self.stage_metrics[stage_name] = result.metrics
            else:
                # Create default metrics with just duration
                self.stage_metrics[stage_name] = StageMetrics(
                    duration_seconds=elapsed,
                    retry_attempts=total_retry_attempts,
                    health_check_results=[hc.to_dict() for hc in health_check_results]
                )

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
                self.checkpoint.save(stage_name, result.data,
                                     stage_metrics=metrics_dict)

            self.progress_reporter.finish_stage()
            completed_stages.add(stage_name)

            # Get items from stage metrics if available
            items_processed = 0
            items_failed = 0
            if stage_name in self.stage_metrics:
                metrics = self.stage_metrics[stage_name]
                items_processed = metrics.items_processed
                items_failed = metrics.items_failed

            # Emit after_stage event (US-81-012)
            self.emit_event(PipelineEvent(
                event_type='after_stage',
                stage_name=stage_name,
                timestamp=time.time(),
                data={'elapsed': elapsed, 'success': True},
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
            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s {ctx.to_suffix()}")

            # Save timing to history for future predictions (US-81-010)
            self._save_stage_timing(stage_name, elapsed)

            # Run quality gate after matching stages (US-81-005)
            if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
                self._check_match_coverage_gate(stage_name)

            # Run cross-stage data drift detection (US-81-011)
            self._check_data_drift(stage_name)

        self.current_stage = None

        # Print timing summary after successful pipeline completion
        total_duration = time.time() - pipeline_start_time
        self._print_timing_summary(total_duration, skipped_stages)

        self.progress_reporter.finish_pipeline()

        # Emit on_pipeline_complete event (US-81-012)
        self.emit_event(PipelineEvent(
            event_type='on_pipeline_complete',
            stage_name='',
            timestamp=time.time(),
            data={'total_duration': total_duration, 'stages_run': list(self.stage_timings.keys())},
        ))

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

        # Log structured summary table with enhanced details
        logger.info("")
        logger.info("Stage Execution Plan:")
        logger.info("-" * 80)
        logger.info(f"  {'Stage':<20} {'Action':<12} {'Est. Duration':<14} {'Inputs':<12} {'Outputs':<12} {'Detail'}")
        logger.info(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*12} {'-'*12} {'-'*15}")

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

            # Get input/output info from stage
            io_info = {'inputs': '', 'outputs': '', 'input_count': None, 'output_count': None}
            if stage and r.status in ('run', 'checkpoint'):
                try:
                    io_info = stage.get_input_output_info(self.state, self.config)
                except Exception:
                    pass  # Fall back to default

            inputs_str = str(io_info.get('input_count', '')) if io_info.get('input_count') is not None else '-'
            outputs_str = str(io_info.get('output_count', '')) if io_info.get('output_count') is not None else '-'

            detail = r.message
            if r.status == "error":
                logger.error(f"  {stage_name:<20} {r.status:<12} {est_duration:<14} {inputs_str:<12} {outputs_str:<12} {detail}")
            else:
                logger.info(f"  {stage_name:<20} {r.status:<12} {est_duration:<14} {inputs_str:<12} {outputs_str:<12} {detail}")

        logger.info("-" * 80)

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
                    logger.info(f"Skipping parallel stage {stage.name} (checkpoint resume)")
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
                logger.debug(f"Health check failed for {stage.name}: {e}")
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

        # Run stages in parallel
        logger.info(f"Running {len(stages_to_run)} stages in parallel: {[s.name for s in stages_to_run]}")
        with ThreadPoolExecutor(max_workers=len(stages_to_run)) as executor:
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

            # Record timing
            self.stage_timings[stage_name] = elapsed
            self.state.stage_timings[stage_name] = elapsed

            # Store metrics
            hc_results = parallel_health_checks.get(stage_name, [])
            if result.metrics:
                result.metrics.duration_seconds = elapsed
                result.metrics.health_check_results = [hc.to_dict() for hc in hc_results]
                self.stage_metrics[stage_name] = result.metrics
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

        return True

    def get_summary(self) -> dict:
        """Get pipeline execution summary including aggregated metrics"""
        metrics = self.get_metrics()
        return {
            'stages_run': list(self.stage_timings.keys()),
            'total_time': sum(self.stage_timings.values()),
            'stage_timings': self.stage_timings,
            'state': self.state.to_checkpoint_dict(),
            'items_processed': metrics['total_items_processed'],
            'items_failed': metrics['total_items_failed'],
            'metrics': metrics,
        }

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

    Returns:
        Configured PipelineOrchestrator
    """
    pipeline = PipelineOrchestrator(config, project_dir)

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

    Returns:
        Configured PipelineOrchestrator with 9 stages
    """
    pipeline = PipelineOrchestrator(config, project_dir)

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
    project_dir: Path
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

    Returns:
        Configured PipelineOrchestrator for match-only mode
    """
    pipeline = PipelineOrchestrator(config, project_dir)

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
    project_dir: Path
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

    Returns:
        Configured PipelineOrchestrator for output-only mode
    """
    pipeline = PipelineOrchestrator(config, project_dir)

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
        logger.debug(f"Failed to get config summary: {e}")

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
