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

from .base import Healer, HealerResult, HealerAction
from .healers import HEALER_REGISTRY

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

        # Exhausted attempts
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
        # Use orchestrator for coordinated healing
        if self.orchestrator:
            result = self.orchestrator.coordinate_heal(error, state, stage_name, error_stack)

            # Record in local history
            self.heal_history.append({
                'stage': stage_name,
                'healer': 'orchestrator',
                'error': str(error)[:200],
                'success': result.success,
                'action': result.action.value,
                'message': result.message,
            })

            if result.success:
                self.total_heals += 1
                logger.info(f"Healed via orchestrator: {result.message}")

                if result.action in (HealerAction.RETRY, HealerAction.MODIFY_CONFIG, HealerAction.RESTORE):
                    return True

            return False

        # Direct healer calls (no orchestrator)
        for healer in self.healers:
            if healer.can_handle(error, stage_name):
                logger.info(f"Attempting heal with {healer.name}...")

                try:
                    result = healer.fix(error, state, stage_name)

                    # Record heal attempt
                    self.heal_history.append({
                        'stage': stage_name,
                        'healer': healer.name,
                        'error': str(error)[:200],
                        'success': result.success,
                        'action': result.action.value,
                        'message': result.message,
                    })

                    if result.success:
                        self.total_heals += 1
                        logger.info(f"Healed by {healer.name}: {result.message}")

                        if result.action == HealerAction.RETRY:
                            return True
                        elif result.action == HealerAction.MODIFY_CONFIG:
                            return True
                        elif result.action == HealerAction.RESTORE:
                            return True
                        elif result.action == HealerAction.SKIP:
                            return False
                        elif result.action == HealerAction.ABORT:
                            return False

                    else:
                        logger.warning(f"Healer {healer.name} could not fix: {result.message}")

                except Exception as heal_error:
                    logger.error(f"Healer {healer.name} raised exception: {heal_error}")

        logger.warning(f"No healer could fix error in {stage_name}")
        return False

    def _reset_healers(self):
        """Reset healer states after successful operation."""
        for healer in self.healers:
            if hasattr(healer, 'reset_backoff'):
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
            print("\n  No healing was needed during this run.")
            return

        print(f"\n  === Healing Summary ===")
        print(f"  Total heals applied: {self.total_heals}")

        # Group by stage
        by_stage: Dict[str, List] = {}
        for entry in self.heal_history:
            stage = entry['stage']
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(entry)

        for stage, entries in by_stage.items():
            print(f"\n  Stage: {stage}")
            for entry in entries:
                status = "✓" if entry['success'] else "✗"
                print(f"    {status} [{entry['healer']}] {entry['message'][:60]}")


def create_resilient_pipeline(
    config: 'Config',
    project_dir: Path,
    audio_first_mode: bool = False,
    strategy: 'HealingStrategy' = None
) -> tuple:
    """
    Create a pipeline with resilient runner.

    Args:
        config: Pipeline configuration
        project_dir: Project directory path
        audio_first_mode: Use audio-first download mode
        strategy: Optional healing strategy

    Returns:
        Tuple of (PipelineOrchestrator, ResilientRunner)
    """
    from ..pipeline import create_default_pipeline
    from .orchestrator import HealingOrchestrator

    pipeline = create_default_pipeline(config, project_dir, audio_first_mode)

    # Create orchestrator if strategy provided
    orchestrator = None
    if strategy:
        orchestrator = HealingOrchestrator(config, project_dir, strategy)

    runner = ResilientRunner(config, project_dir, orchestrator=orchestrator)

    return pipeline, runner


def create_orchestrated_pipeline(
    config: 'Config',
    project_dir: Path,
    audio_first_mode: bool = False,
    strategy: 'HealingStrategy' = None
) -> tuple:
    """
    Create a fully orchestrated pipeline with healing.

    Returns:
        Tuple of (PipelineOrchestrator, HealingOrchestrator, ResilientRunner)
    """
    from ..pipeline import create_default_pipeline
    from .orchestrator import HealingOrchestrator
    from .strategy import HealingStrategy

    strategy = strategy or HealingStrategy.conservative()

    pipeline = create_default_pipeline(config, project_dir, audio_first_mode)
    orchestrator = HealingOrchestrator(config, project_dir, strategy)
    runner = ResilientRunner(config, project_dir, orchestrator=orchestrator)

    return pipeline, orchestrator, runner
