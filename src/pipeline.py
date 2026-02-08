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
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .checkpoint import CheckpointManager, STAGE_ORDER
from .pipeline_progress import ProgressReporter
from .state import PipelineState
from .stages import Stage, StageResult, StageMetrics

if TYPE_CHECKING:
    from .config import Config
    from .agents.runner import ResilientRunner
    from .agents.orchestrator import HealingOrchestrator

# Type aliases for callbacks
StageStartCallback = Callable[[str], None]  # (stage_name) -> None
StageCompleteCallback = Callable[[str, StageResult, float], None]  # (stage_name, result, elapsed_seconds) -> None

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates pipeline stage execution with checkpoint support.

    Features:
    - Runs stages in order
    - Supports resume from checkpoint
    - Tracks stage timing
    - Handles stage failures gracefully
    """

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

    def add_stage(self, stage: Stage) -> 'PipelineOrchestrator':
        """Add a stage to the pipeline (fluent interface)"""
        self.stages.append(stage)
        return self

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

        Combines schema validation (pure config checks) with runtime environment
        checks (filesystem/PATH). See _validate_config_schema() and
        _validate_runtime_environment() for details.

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        errors = self._validate_config_schema()
        errors.extend(self._validate_runtime_environment())
        return errors

    def _validate_config_schema(self) -> List[str]:
        """
        Pure config schema validation — no I/O, no filesystem access.

        Checks required fields, type constraints, and value ranges that can be
        validated from config values alone. Safe to call in CI without a real
        filesystem.

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        errors: List[str] = []

        # Check cache_dir is configured
        cache_dir = getattr(getattr(self.config, 'cache', None), 'cache_dir', None)
        if not cache_dir:
            errors.append("No cache directory configured (config.cache.cache_dir)")

        # Check embedding provider when matching stages are enabled
        matching_stage_names = {'MATCH', 'ITERATIVE_MATCH'}
        has_matching_stages = any(
            getattr(stage, 'name', '') in matching_stage_names
            for stage in self.stages
        )
        if has_matching_stages:
            embedding_config = getattr(self.config, 'embedding', None)
            provider = getattr(embedding_config, 'provider', None) if embedding_config else None
            if not provider:
                errors.append(
                    "Embedding provider not configured (config.embedding.provider) "
                    "but matching stages require embeddings"
                )

        return errors

    def _validate_runtime_environment(self) -> List[str]:
        """
        Runtime environment checks — requires filesystem/PATH access.

        Checks cache_dir writability, browser availability, and other
        I/O-dependent preconditions.

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        errors: List[str] = []

        # Check cache_dir is writable or can be created
        cache_dir = getattr(getattr(self.config, 'cache', None), 'cache_dir', None)
        if cache_dir and isinstance(cache_dir, str):
            cache_path = Path(cache_dir)
            if cache_path.exists():
                if not os.access(str(cache_path), os.W_OK):
                    errors.append(
                        f"Cache directory is not writable: {cache_dir}"
                    )
            else:
                # Check if parent is writable so we can create it
                parent = cache_path.parent
                if parent.exists() and not os.access(str(parent), os.W_OK):
                    errors.append(
                        f"Cannot create cache directory (parent not writable): {cache_dir}"
                    )

        # US-57-004: Warn if cookies_from_browser is set but browser not on PATH
        self._warn_cookies_from_browser()

        return errors

    def _warn_cookies_from_browser(self) -> None:
        """Check if cookies_from_browser browser is findable on PATH.

        Emits a WARNING if the configured browser cannot be found.
        Does NOT block pipeline startup — Tier 3 cookie extraction will
        simply fail at download time.
        """
        download_config = getattr(self.config, 'download', None)
        if download_config is None:
            return
        browser = getattr(download_config, 'cookies_from_browser', '')
        if not browser:
            return

        # Map browser config names to common executable names
        _exe_map = {
            'firefox': 'firefox',
            'chrome': 'chrome',
            'edge': 'msedge',
            'safari': 'safari',
            'opera': 'opera',
            'brave': 'brave',
        }
        exe_name = _exe_map.get(browser.lower(), browser)
        if not shutil.which(exe_name) and not shutil.which(browser):
            logger.warning(
                f'cookies_from_browser is set to "{browser}" but it is not '
                f'found on PATH. Tier 3 cookie extraction will fail. '
                f'Set download.cookies_path instead or install {browser}.'
            )

    # Fields to snapshot before each stage for rollback on failure
    _SNAPSHOT_FIELDS = ('matches', 'alternatives', 'downloaded_segments', 'output_files')

    def _snapshot_state(self) -> Dict[str, Any]:
        """Create a shallow copy of critical state fields before stage execution."""
        snapshot = {}
        for field_name in self._SNAPSHOT_FIELDS:
            value = getattr(self.state, field_name, None)
            if value is not None:
                snapshot[field_name] = copy.copy(value)
            else:
                snapshot[field_name] = None
        return snapshot

    def _rollback_state(self, snapshot: Dict[str, Any], stage_name: str) -> None:
        """Restore state from a pre-stage snapshot after a failure."""
        restored_fields = []
        for field_name, saved_value in snapshot.items():
            setattr(self.state, field_name, saved_value)
            restored_fields.append(field_name)
        logger.warning(
            f"State rollback after {stage_name} failure: "
            f"restored fields: {restored_fields}"
        )

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
        logger.info("=" * 60)
        logger.info("Pipeline Timing Summary")
        logger.info("=" * 60)
        logger.info(f"  {'Stage':<25} {'Duration':>10} {'% of Total':>12}")
        logger.info(f"  {'-'*25} {'-'*10} {'-'*12}")

        for stage in self.stages:
            name = stage.name
            if name in skipped_stages:
                logger.info(f"  {name:<25} {'skipped':>10} {'-':>12}")
            elif name in self.stage_timings:
                elapsed = self.stage_timings[name]
                pct = (elapsed / total_duration * 100) if total_duration > 0 else 0.0
                logger.info(f"  {name:<25} {elapsed:>9.1f}s {pct:>11.1f}%")
            else:
                # Stage was filtered out (skip_stages / only_stages)
                logger.info(f"  {name:<25} {'--':>10} {'-':>12}")

        logger.info(f"  {'-'*25} {'-'*10} {'-'*12}")
        logger.info(f"  {'TOTAL':<25} {total_duration:>9.1f}s {'100.0%':>12}")
        logger.info("=" * 60)

        # Persist timing summary to checkpoint stage_metrics
        self.checkpoint.save_stage_timing_summary(
            self.stage_timings, total_duration, skipped_stages
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

        # Run each stage
        for stage in self.stages:
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
                    continue
                else:
                    # US-51-008: restore failed - re-run the stage instead of
                    # proceeding with bad/partial state
                    logger.warning(
                        f"Failed to restore {stage_name} from checkpoint, "
                        f"re-running stage"
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
                    on_stage_start, on_stage_complete
                )
                if not success:
                    return False
                continue

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

            # Invoke on_stage_start callback
            if on_stage_start:
                try:
                    on_stage_start(stage_name)
                except Exception as e:
                    logger.warning(f"on_stage_start callback failed for {stage_name}: {e}")

            # Start progress tracking for this stage
            self.progress_reporter.start_stage(stage_name)
            # Make reporter accessible to stages via state (transient, not serialized)
            self.state._progress_reporter = self.progress_reporter

            logger.info(f"Running stage: {stage_name}")
            result = stage.run(self.state, self.config, self.checkpoint)

            elapsed = time.time() - start_time
            self.stage_timings[stage_name] = elapsed
            self.state.stage_timings[stage_name] = elapsed

            # Store stage metrics if provided
            if result.metrics:
                # Update duration_seconds to actual elapsed time
                result.metrics.duration_seconds = elapsed
                self.stage_metrics[stage_name] = result.metrics
            else:
                # Create default metrics with just duration
                self.stage_metrics[stage_name] = StageMetrics(duration_seconds=elapsed)

            # Invoke on_stage_complete callback
            if on_stage_complete:
                try:
                    on_stage_complete(stage_name, result, elapsed)
                except Exception as e:
                    logger.warning(f"on_stage_complete callback failed for {stage_name}: {e}")

            # Handle result
            if not result.success:
                # Rollback state to pre-stage snapshot
                self._rollback_state(state_snapshot, stage_name)

                # Mark metrics as failed (metrics already stored above)
                if stage_name in self.stage_metrics:
                    self.stage_metrics[stage_name].failed = True
                self.progress_reporter.finish_stage()
                state_ctx = self._get_state_summary()
                recovery = self._get_recovery_suggestion(stage_name, result.error or "")
                logger.error(
                    f"Stage {stage_name} failed: {result.error} "
                    f"[state: {state_ctx}] "
                    f"Recovery: {recovery}"
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
            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s")

            # Run quality gate after matching stages (US-81-005)
            if stage_name in ('MATCH', 'ITERATIVE_MATCH'):
                self._check_match_coverage_gate(stage_name)

        self.current_stage = None

        # Print timing summary after successful pipeline completion
        total_duration = time.time() - pipeline_start_time
        self._print_timing_summary(total_duration, skipped_stages)

        self.progress_reporter.finish_pipeline()

        return True

    def _run_dry_run(
        self,
        skip_stages: set,
        only_stages: Optional[set],
        resume: bool = True
    ) -> bool:
        """
        Run pipeline in dry-run mode: validate all stage inputs without executing.

        Iterates all stages in order, determines which would run vs skip
        (via checkpoint), validates inputs for stages that would run, and
        produces a structured summary table.

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

        # Load checkpoint for skip detection (read-only, does not modify state)
        checkpoint_loaded = False
        if resume:
            checkpoint_loaded = self.load_checkpoint()
            if checkpoint_loaded:
                logger.info("Checkpoint found - checking which stages can be skipped")
            else:
                logger.info("No checkpoint found - all stages would run")

        # Categorize each stage: skip, checkpoint, run, error
        # action: "skip" (filtered), "checkpoint" (would restore), "run", "error"
        summary = []  # list of (stage_name, action, detail)
        has_errors = False

        for stage in self.stages:
            stage_name = stage.name

            # Check skip/only filters
            if stage_name in skip_stages:
                summary.append((stage_name, "skip", "filtered by skip_stages"))
                continue

            if only_stages and stage_name not in only_stages:
                summary.append((stage_name, "skip", "not in only_stages"))
                continue

            # Check checkpoint skip
            if resume and checkpoint_loaded and self.resume_mode:
                if stage.can_skip(self.state, self.checkpoint):
                    summary.append((stage_name, "checkpoint", "would restore from checkpoint"))
                    continue

            # Validate inputs for stages that would actually run
            validation_error = stage.validate_inputs(self.state, self.config)
            if validation_error:
                summary.append((stage_name, "error", validation_error))
                has_errors = True
            else:
                summary.append((stage_name, "run", "inputs valid"))

        # Log structured summary table
        logger.info("")
        logger.info("Stage Execution Plan:")
        logger.info("-" * 60)
        logger.info(f"  {'Stage':<25} {'Action':<12} {'Detail'}")
        logger.info(f"  {'-'*25} {'-'*12} {'-'*20}")
        for stage_name, action, detail in summary:
            if action == "error":
                logger.error(f"  {stage_name:<25} {action:<12} {detail}")
            else:
                logger.info(f"  {stage_name:<25} {action:<12} {detail}")
        logger.info("-" * 60)

        # Count by action
        counts = {}
        for _, action, _ in summary:
            counts[action] = counts.get(action, 0) + 1

        parts = []
        for action in ["run", "checkpoint", "skip", "error"]:
            if action in counts:
                parts.append(f"{action}={counts[action]}")
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
        on_stage_complete: StageCompleteCallback
    ) -> bool:
        """
        Run a group of stages in parallel using ThreadPoolExecutor.

        Args:
            group: Tuple of stage names to run in parallel
            skip_stages: Set of stage names to skip
            only_stages: If set, only run these stages
            on_stage_start: Callback for stage start
            on_stage_complete: Callback for stage completion

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

        def run_stage(stage: Stage) -> Tuple[str, StageResult, float]:
            """Execute a single stage and return results."""
            start_time = time.time()
            if on_stage_start:
                try:
                    on_stage_start(stage.name)
                except Exception as e:
                    logger.warning(f"on_stage_start callback failed for {stage.name}: {e}")

            logger.info(f"Running parallel stage: {stage.name}")
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
            if result.metrics:
                result.metrics.duration_seconds = elapsed
                self.stage_metrics[stage_name] = result.metrics
            else:
                self.stage_metrics[stage_name] = StageMetrics(duration_seconds=elapsed)

            # Invoke callback
            if on_stage_complete:
                try:
                    on_stage_complete(stage_name, result, elapsed)
                except Exception as e:
                    logger.warning(f"on_stage_complete callback failed for {stage_name}: {e}")

            # Handle failure
            if not result.success:
                logger.error(f"Parallel stage {stage_name} failed: {result.error}")
                for warning in result.warnings:
                    logger.warning(f"  Warning: {warning}")
                return False

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

            logger.info(f"Parallel stage {stage_name} completed in {elapsed:.1f}s")

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
    except Exception:
        pass  # Non-critical: metrics collection should never break the pipeline


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

        return success
    else:
        # Fallback to standard execution
        return pipeline.run(resume=resume)
