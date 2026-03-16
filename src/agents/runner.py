"""
Resilient Runner - Self-healing pipeline executor.

Wraps the PipelineOrchestrator with automatic error recovery.
When a stage fails, it attempts to heal the error and retry.
Integrates with HealingOrchestrator for coordinated healing.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from .base import Healer, HealerResult, HealerAction, SupportsBackoff
from .healers import HEALER_REGISTRY
from ..logging_templates import (
    log_error_with_context,
    log_healing_attempt,
    log_healing_success,
    log_healing_failure,
    log_no_healer_available,
    log_max_healing_exceeded,
    log_manual_intervention,
    log_healing_action_details,
)

if TYPE_CHECKING:
    from ..config import Config
    from ..state import PipelineState
    from ..stages import Stage, StageResult
    from ..checkpoint import CheckpointManager
    from .orchestrator import HealingOrchestrator
    from .strategy import HealingStrategy

logger = logging.getLogger(__name__)


class ResilientRunner:
    """
    Self-healing pipeline runner.

    Wraps stage execution with automatic error detection and recovery.
    Uses a registry of healers or a HealingOrchestrator for coordinated healing.

    Usage:
        # Simple usage
        runner = ResilientRunner(config, project_dir)
        success = runner.run_pipeline(pipeline)

        # With orchestrator (recommended)
        from src.agents import HealingOrchestrator, HealingStrategy
        orchestrator = HealingOrchestrator(config, project_dir, HealingStrategy.aggressive())
        runner = ResilientRunner(config, project_dir, orchestrator=orchestrator)
        success = runner.run_pipeline(pipeline)
    """

    # Maximum healing attempts per stage (can be overridden by orchestrator)
    MAX_HEAL_ATTEMPTS = 3

    # Delay between heal attempts
    HEAL_DELAY_SECONDS = 2.0

    def __init__(
        self,
        config: 'Config',
        project_dir: Path,
        healers: List[Type[Healer]] = None,
        orchestrator: 'HealingOrchestrator' = None
    ):
        """
        Initialize the resilient runner.

        Args:
            config: Pipeline configuration
            project_dir: Project directory path
            healers: Optional list of healer classes (uses defaults if not provided)
            orchestrator: Optional HealingOrchestrator for coordinated healing
        """
        self.config = config
        self.project_dir = Path(project_dir)
        self.orchestrator = orchestrator

        # Log runner initialization
        logger.info(f"[RUNNER] ResilientRunner initialized (max_heal_attempts: {self.MAX_HEAL_ATTEMPTS}, heal_delay: {self.HEAL_DELAY_SECONDS}s)")
        if orchestrator:
            logger.info(f"[RUNNER] Connected to HealingOrchestrator with strategy: {orchestrator.strategy.mode.value}")

        # Initialize healers (used if no orchestrator)
        if orchestrator:
            self.healers = orchestrator.healers
            self.MAX_HEAL_ATTEMPTS = orchestrator.strategy.max_attempts_per_stage
            self.HEAL_DELAY_SECONDS = orchestrator.strategy.heal_delay
        else:
            healer_classes = healers or HEALER_REGISTRY
            self.healers: List[Healer] = [
                cls(config, project_dir) for cls in healer_classes
            ]

        # Tracking
        self.heal_history: List[Dict] = []
        self.stage_attempts: Dict[str, int] = {}
        self.total_heals = 0

    def run_pipeline(
        self,
        pipeline: 'PipelineOrchestrator',
        resume: bool = True
    ) -> bool:
        """
        Run the entire pipeline with healing.

        Args:
            pipeline: The pipeline to run
            resume: Whether to resume from checkpoint

        Returns:
            True if pipeline completed successfully
        """
        # Run preflight checks if orchestrator is available
        if self.orchestrator:
            preflight_issues = self.orchestrator.run_preflight(pipeline.state)
            if preflight_issues:
                critical = [i for i in preflight_issues if i.severity == "critical"]
                if critical:
                    logger.error(f"Critical preflight issues found: {len(critical)}")
                    # Try to fix
                    fixed, remaining = self.orchestrator.fix_preflight_issues(
                        preflight_issues, pipeline.state
                    )
                    if remaining > 0:
                        critical_remaining = [i for i in preflight_issues
                                             if i.severity == "critical" and not i.auto_fixable]
                        if critical_remaining:
                            logger.error("Cannot proceed with unfixed critical issues")
                            return False

        # Load checkpoint if resuming
        if resume:
            pipeline.load_checkpoint()

        # Run each stage with healing
        for stage in pipeline.stages:
            stage_name = stage.name

            # Take config snapshot if orchestrator available
            if self.orchestrator and self.orchestrator.strategy.enable_rollback:
                self.orchestrator.snapshot_config(stage_name)

            # Check if stage can be skipped
            if pipeline.resume_mode and stage.can_skip(pipeline.state, pipeline.checkpoint):
                logger.info(f"Skipping {stage_name} (checkpoint resume)")
                if not stage.restore(pipeline.state, pipeline.checkpoint, pipeline.config):
                    logger.warning(f"Failed to restore {stage_name} from checkpoint")
                # Validate state attributes after stage restoration
                pipeline.state.validate_state_attributes()
                continue

            # Validate inputs
            validation_error = stage.validate_inputs(pipeline.state, pipeline.config)
            if validation_error:
                logger.error(f"Stage {stage_name} validation failed: {validation_error}")
                return False

            # Run stage with healing
            start_time = time.time()
            result = self.run_stage(
                stage,
                pipeline.state,
                pipeline.config,
                pipeline.checkpoint
            )

            elapsed = time.time() - start_time
            pipeline.stage_timings[stage_name] = elapsed
            pipeline.state.stage_timings[stage_name] = elapsed

            if not result.success:
                logger.error(f"Stage {stage_name} failed: {result.error}")

                # Try rollback if available
                if self.orchestrator and self.orchestrator.strategy.enable_rollback:
                    logger.info("Attempting config rollback...")
                    if self.orchestrator.rollback_config(stage_name):
                        # Could retry here, but for now just report failure
                        pass

                return False

            # Save checkpoint
            if result.data:
                pipeline.checkpoint.save(stage_name, result.data)

            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s")

        # Clean up session recovery file after successful completion (US-68-009)
        if self.orchestrator:
            self.orchestrator.cleanup_session_file()

        return True

    def run_stage(
        self,
        stage: 'Stage',
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> 'StageResult':
        """
        Run a single stage with healing.

        Args:
            stage: The stage to run
            state: Pipeline state
            config: Configuration
            checkpoint: Checkpoint manager

        Returns:
            StageResult from the stage
        """
        from ..stages import StageResult

        stage_name = stage.name
        attempts = 0
        max_attempts = self.MAX_HEAL_ATTEMPTS

        # Check orchestrator limits
        if self.orchestrator:
            total_heals = self.orchestrator.metrics.total_heals
            if total_heals >= self.orchestrator.strategy.max_total_heals:
                logger.error("Maximum total heals reached, aborting")
                return StageResult.fail("Maximum healing attempts exceeded")

        while attempts < max_attempts:
            attempts += 1
            self.stage_attempts[stage_name] = attempts

            try:
                logger.info(f"Running stage: {stage_name} (attempt {attempts})")
                result = stage.run(state, config, checkpoint)

                if result.success:
                    # Reset any healer backoff states on success
                    self._reset_healers()
                    return result

                # Stage returned failure (not exception)
                if result.error:
                    error = Exception(result.error)
                    healed = self._try_heal(error, state, stage_name)

                    if not healed:
                        return result

                    # Healed - continue to retry
                    time.sleep(self.HEAL_DELAY_SECONDS)
                    continue

                return result

            except Exception as e:
                # Capture stack trace immediately while exception context is active
                error_stack = traceback.format_exc()
                logger.error(f"Stage {stage_name} raised exception: {e}")

                # Try to heal (pass stack trace for context)
                healed = self._try_heal(e, state, stage_name, error_stack)

                if not healed:
                    return StageResult.fail(str(e))

                # Healed - continue to retry
                time.sleep(self.HEAL_DELAY_SECONDS)

        # Exhausted attempts - log with structured format
        log_max_healing_exceeded(
            logger,
            target=stage_name,
            max_attempts=max_attempts,
            total_heals=self.total_heals
        )

        return StageResult.fail(
            f"Stage {stage_name} failed after {max_attempts} heal attempts"
        )

    def _try_heal(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str,
        error_stack: Optional[str] = None
    ) -> bool:
        """
        Attempt to heal an error.

        Uses orchestrator if available, otherwise uses direct healer calls.

        Args:
            error: The exception that occurred
            state: Pipeline state
            stage_name: Name of the stage that failed
            error_stack: Pre-captured stack trace (captured at exception time)

        Returns:
            True if error was healed and stage should retry
        """
        # Get current attempt number for logging
        attempt = self.stage_attempts.get(stage_name, 1)
        error_type = type(error).__name__

        # Use orchestrator for coordinated healing
        if self.orchestrator:
            # Log healing attempt with structured format
            log_healing_attempt(
                logger,
                healer_name="orchestrator",
                target=stage_name,
                attempt=attempt,
                max_attempts=self.MAX_HEAL_ATTEMPTS,
                error_type=error_type,
                error_message=str(error)[:100]
            )

            start_time = time.time()
            result = self.orchestrator.coordinate_heal(error, state, stage_name, error_stack)
            duration_ms = (time.time() - start_time) * 1000

            # Record in local history
            self.heal_history.append({
                'stage': stage_name,
                'healer': 'orchestrator',
                'error': str(error)[:200],
                'success': result.success,
                'action': result.action.value,
                'message': result.message,
                'duration_ms': duration_ms,
            })

            if result.success:
                self.total_heals += 1
                # Log healing success with details
                affected = list(result.details.get('affected_components', [])) if result.details else []
                log_healing_success(
                    logger,
                    healer_name="orchestrator",
                    target=stage_name,
                    action=result.action.value,
                    affected_components=affected,
                    duration_ms=duration_ms,
                    error_message=result.message
                )

                if result.action in (HealerAction.RETRY, HealerAction.MODIFY_CONFIG, HealerAction.RESTORE):
                    return True

            # Log failure if not successful
            log_healing_failure(
                logger,
                healer_name="orchestrator",
                target=stage_name,
                error_message=result.message
            )

            # Check if manual intervention is required
            if result.action == HealerAction.ABORT:
                log_manual_intervention(
                    logger,
                    target=stage_name,
                    reason=result.message
                )

            return False

        # Direct healer calls (no orchestrator)
        for healer in self.healers:
            if healer.can_handle(error, stage_name):
                # Log healing attempt with structured format
                log_healing_attempt(
                    logger,
                    healer_name=healer.name,
                    target=stage_name,
                    attempt=attempt,
                    max_attempts=self.MAX_HEAL_ATTEMPTS,
                    error_type=error_type,
                    error_message=str(error)[:100]
                )

                try:
                    start_time = time.time()
                    result = healer.fix(error, state, stage_name)
                    duration_ms = (time.time() - start_time) * 1000

                    # Record heal attempt
                    self.heal_history.append({
                        'stage': stage_name,
                        'healer': healer.name,
                        'error': str(error)[:200],
                        'success': result.success,
                        'action': result.action.value,
                        'message': result.message,
                        'duration_ms': duration_ms,
                    })

                    if result.success:
                        self.total_heals += 1

                        # Extract details for logging
                        files_modified = result.details.get('files_modified') if result.details else None
                        config_changes = result.details.get('config_changes') if result.details else None
                        affected = result.details.get('affected_components', []) if result.details else []

                        # Log healing success with details
                        log_healing_success(
                            logger,
                            healer_name=healer.name,
                            target=stage_name,
                            action=result.action.value,
                            affected_components=affected,
                            duration_ms=duration_ms,
                            error_message=result.message
                        )

                        # Log detailed action info
                        log_healing_action_details(
                            logger,
                            healer_name=healer.name,
                            target=stage_name,
                            action=result.action.value,
                            files_modified=files_modified,
                            config_changes=config_changes
                        )

                        if result.action == HealerAction.RETRY:
                            return True
                        elif result.action == HealerAction.MODIFY_CONFIG:
                            return True
                        elif result.action == HealerAction.RESTORE:
                            return True
                        elif result.action == HealerAction.SKIP:
                            log_manual_intervention(
                                logger,
                                target=stage_name,
                                reason=f"Healer chose to skip: {result.message}"
                            )
                            return False
                        elif result.action == HealerAction.ABORT:
                            log_manual_intervention(
                                logger,
                                target=stage_name,
                                reason=result.message
                            )
                            return False

                    else:
                        # Log healing failure
                        log_healing_failure(
                            logger,
                            healer_name=healer.name,
                            target=stage_name,
                            error_message=result.message
                        )

                except Exception as heal_error:
                    # Log healer exception with error code
                    log_error_with_context(
                        logger,
                        "HEAL-006",
                        f"Healer {healer.name} raised exception",
                        healer=healer.name,
                        target=stage_name,
                        error=str(heal_error)[:200]
                    )

        # Log when no healer could fix the error
        log_no_healer_available(
            logger,
            target=stage_name,
            error_type=error_type,
            error_message=str(error)[:200]
        )

        return False

    def _reset_healers(self):
        """Reset healer states after successful operation."""
        for healer in self.healers:
            if isinstance(healer, SupportsBackoff):
                healer.reset_backoff()

    def get_summary(self) -> Dict:
        """Get healing summary."""
        summary = {
            'total_heals': self.total_heals,
            'stage_attempts': self.stage_attempts,
            'heal_history': self.heal_history,
        }

        # Include orchestrator metrics if available
        if self.orchestrator:
            summary['orchestrator_metrics'] = {
                'total_heals': self.orchestrator.metrics.total_heals,
                'successful_heals': self.orchestrator.metrics.successful_heals,
                'preflight_issues': self.orchestrator.metrics.preflight_issues_found,
                'preflight_fixed': self.orchestrator.metrics.preflight_issues_fixed,
                'rollbacks': self.orchestrator.metrics.rollbacks_performed,
            }

        return summary

    def print_summary(self):
        """Print healing summary to console."""
        # Use orchestrator report if available
        if self.orchestrator:
            self.orchestrator.print_report()
            return

        if not self.heal_history:
            logger.info("[AGENT] No healing was needed during this run.")
            return

        logger.info(f"[AGENT] Healing Summary")
        logger.info(f"[AGENT] Total heals applied: {self.total_heals}")

        # Group by stage
        by_stage: Dict[str, List] = {}
        for entry in self.heal_history:
            stage = entry['stage']
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(entry)

        for stage, entries in by_stage.items():
            logger.info(f"[AGENT] Stage: {stage}")
            for entry in entries:
                status = "✓" if entry['success'] else "✗"
                logger.info(f"[AGENT]   {status} [{entry['healer']}] {entry['message'][:60]}")


def create_resilient_pipeline(
    config: 'Config',
    project_dir: Path,
    strategy: 'HealingStrategy' = None
) -> tuple:
    """
    Create a pipeline with resilient runner.

    Args:
        config: Pipeline configuration
        project_dir: Project directory path
        strategy: Optional healing strategy

    Returns:
        Tuple of (PipelineOrchestrator, ResilientRunner)
    """
    from ..pipeline import create_default_pipeline
    from .orchestrator import HealingOrchestrator

    pipeline = create_default_pipeline(config, project_dir)

    # Create orchestrator if strategy provided
    orchestrator = None
    if strategy:
        orchestrator = HealingOrchestrator(config, project_dir, strategy)

    runner = ResilientRunner(config, project_dir, orchestrator=orchestrator)

    return pipeline, runner


def create_orchestrated_pipeline(
    config: 'Config',
    project_dir: Path,
    strategy: 'HealingStrategy' = None,
    recover_session: bool = True,
) -> tuple:
    """
    Create a fully orchestrated pipeline with healing.

    Args:
        config: Pipeline configuration
        project_dir: Project directory path
        strategy: Optional healing strategy
        recover_session: Whether to attempt session recovery from crash (US-64-012)

    Returns:
        Tuple of (PipelineOrchestrator, HealingOrchestrator, ResilientRunner)
    """
    from ..pipeline import create_default_pipeline
    from .orchestrator import HealingOrchestrator
    from .strategy import HealingStrategy

    strategy = strategy or HealingStrategy.conservative()

    pipeline = create_default_pipeline(config, project_dir)
    orchestrator = HealingOrchestrator(config, project_dir, strategy)
    runner = ResilientRunner(config, project_dir, orchestrator=orchestrator)

    # US-64-012: Attempt session recovery if enabled
    if recover_session:
        recovered = orchestrator.load_session()
        if recovered:
            logger.info(
                "[runner] Recovered healing session from previous run "
                f"(stage={orchestrator.current_stage}, "
                f"heals={orchestrator.metrics.total_heals})"
            )

    return pipeline, orchestrator, runner
