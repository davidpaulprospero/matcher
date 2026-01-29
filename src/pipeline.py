"""
Pipeline Orchestrator

Lightweight orchestrator that runs pipeline stages in order,
handling checkpointing and resume functionality.

This replaces the run() method and stage orchestration logic in main.py.

Self-Healing: By default, pipelines use ResilientRunner with HealingOrchestrator
for automatic error recovery. Controlled via config.healing settings.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from .checkpoint import CheckpointManager, STAGE_ORDER
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

        # Initialize checkpoint manager
        config_hash = getattr(config, '_config_hash', '')
        self.checkpoint = CheckpointManager(project_dir, config_hash, config=config)

        # Runtime state
        self.resume_mode = False
        self.current_stage: Optional[str] = None
        self.stage_timings: dict = {}
        self.stage_metrics: dict = {}  # stage_name -> StageMetrics

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

        self.resume_mode = True
        return True

    def run(
        self,
        resume: bool = True,
        skip_stages: List[str] = None,
        only_stages: List[str] = None,
        on_stage_start: StageStartCallback = None,
        on_stage_complete: StageCompleteCallback = None
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

        Returns:
            True if pipeline completed successfully
        """
        skip_stages = set(skip_stages or [])
        only_stages = set(only_stages) if only_stages else None

        # Try to load checkpoint if resuming
        if resume:
            self.load_checkpoint()

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
                if not stage.restore(self.state, self.checkpoint, self.config):
                    logger.warning(f"Failed to restore {stage_name} from checkpoint")
                continue

            # Validate inputs
            validation_error = stage.validate_inputs(self.state, self.config)
            if validation_error:
                logger.error(f"Stage {stage_name} validation failed: {validation_error}")
                return False

            # Run the stage
            self.current_stage = stage_name
            start_time = time.time()

            # Invoke on_stage_start callback
            if on_stage_start:
                try:
                    on_stage_start(stage_name)
                except Exception as e:
                    logger.warning(f"on_stage_start callback failed for {stage_name}: {e}")

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
                logger.error(f"Stage {stage_name} failed: {result.error}")
                for warning in result.warnings:
                    logger.warning(f"  Warning: {warning}")
                return False

            # Log warnings
            for warning in result.warnings:
                logger.warning(f"Stage {stage_name}: {warning}")

            # Save checkpoint
            if result.data:
                self.checkpoint.save(stage_name, result.data)

            logger.info(f"Stage {stage_name} completed in {elapsed:.1f}s")

        self.current_stage = None
        return True

    def get_summary(self) -> dict:
        """Get pipeline execution summary"""
        return {
            'stages_run': list(self.stage_timings.keys()),
            'total_time': sum(self.stage_timings.values()),
            'stage_timings': self.stage_timings,
            'state': self.state.to_checkpoint_dict(),
        }

    def get_metrics(self) -> dict:
        """
        Get aggregated metrics across all stages.

        Returns:
            Dictionary with:
            - total_items_processed: Sum of items_processed across all stages
            - total_items_failed: Sum of items_failed across all stages
            - total_duration_seconds: Sum of duration_seconds across all stages
            - stages: Dict mapping stage_name to StageMetrics
        """
        total_processed = 0
        total_failed = 0
        total_duration = 0.0

        for stage_name, metrics in self.stage_metrics.items():
            total_processed += metrics.items_processed
            total_failed += metrics.items_failed
            total_duration += metrics.duration_seconds

        return {
            'total_items_processed': total_processed,
            'total_items_failed': total_failed,
            'total_duration_seconds': total_duration,
            'stages': self.stage_metrics
        }

    def clear_checkpoint(self):
        """Clear checkpoint for fresh start"""
        self.checkpoint.clear()
        self.resume_mode = False


def create_default_pipeline(
    config: 'Config',
    project_dir: Path,
    audio_first_mode: bool = False
) -> PipelineOrchestrator:
    """
    Create a pipeline with the default stage order.

    This is a factory function that creates a fully configured pipeline.
    Stages are imported lazily to avoid circular imports.

    Args:
        config: Configuration object
        project_dir: Project directory path
        audio_first_mode: If True, uses audio-first download pipeline

    Returns:
        Configured PipelineOrchestrator
    """
    pipeline = PipelineOrchestrator(config, project_dir)

    # Import stages lazily to avoid circular imports
    from .stages.analyze import AnalyzeStage
    from .stages.entity_images import EntityImagesStage
    from .stages.entity_videos import EntityVideosStage
    from .stages.download import DownloadStage, DownloadVideoSegmentsStage
    from .stages.stock import StockVideoStage
    from .stages.broll_download import BrollDownloadStage
    from .stages.remix import RemixStage
    from .stages.caption_stage import CaptionStage
    from .stages.transcribe import TranscribeStage
    from .stages.scene_detection import SceneDetectionStage
    from .stages.match import MatchStage
    from .stages.broll_match import BrollMatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.output import OutputStage

    # Add stages in STAGE_ORDER
    pipeline.add_stage(AnalyzeStage())
    pipeline.add_stage(EntityImagesStage())
    pipeline.add_stage(EntityVideosStage())
    pipeline.add_stage(DownloadStage())
    pipeline.add_stage(StockVideoStage())
    pipeline.add_stage(BrollDownloadStage())  # B-roll specific downloads
    pipeline.add_stage(RemixStage())
    pipeline.add_stage(CaptionStage())  # Fetch YouTube captions before transcription
    pipeline.add_stage(TranscribeStage())
    pipeline.add_stage(SceneDetectionStage())  # Scene detection with B-roll marking
    pipeline.add_stage(MatchStage())
    pipeline.add_stage(BrollMatchStage())  # Match silent scenes for V8 track
    pipeline.add_stage(IterativeMatchStage())  # Multi-pass gap filling

    # Audio-first mode adds video segment download after matching
    if audio_first_mode:
        pipeline.add_stage(DownloadVideoSegmentsStage())

    pipeline.add_stage(OutputStage())

    return pipeline


def create_match_only_pipeline(
    config: 'Config',
    project_dir: Path
) -> PipelineOrchestrator:
    """
    Create a pipeline that only runs matching and output stages.

    Used when user wants to re-run matching with different config
    without re-downloading or transcribing.

    Args:
        config: Configuration object
        project_dir: Project directory path

    Returns:
        Configured PipelineOrchestrator for match-only mode
    """
    pipeline = PipelineOrchestrator(config, project_dir)

    from .stages.analyze import AnalyzeStage
    from .stages.entity_images import EntityImagesStage
    from .stages.entity_videos import EntityVideosStage
    from .stages.download import DownloadStage
    from .stages.stock import StockVideoStage
    from .stages.broll_download import BrollDownloadStage
    from .stages.remix import RemixStage
    from .stages.caption_stage import CaptionStage
    from .stages.transcribe import TranscribeStage
    from .stages.scene_detection import SceneDetectionStage
    from .stages.match import MatchStage
    from .stages.broll_match import BrollMatchStage
    from .stages.iterative_match import IterativeMatchStage
    from .stages.output import OutputStage

    # Add prerequisite stages for restoration only (will be skipped via checkpoint)
    pipeline.add_stage(AnalyzeStage())
    pipeline.add_stage(EntityImagesStage())  # For V9 track
    pipeline.add_stage(EntityVideosStage())  # For V10 track
    pipeline.add_stage(DownloadStage())
    pipeline.add_stage(StockVideoStage())    # For general B-roll
    pipeline.add_stage(BrollDownloadStage()) # B-roll specific downloads
    pipeline.add_stage(RemixStage())         # Filter videos by relevance
    pipeline.add_stage(CaptionStage())       # Fetch YouTube captions
    pipeline.add_stage(TranscribeStage())
    pipeline.add_stage(SceneDetectionStage())  # Scene detection with B-roll marking

    # Add stages to actually run
    pipeline.add_stage(MatchStage())
    pipeline.add_stage(BrollMatchStage())    # Match silent scenes for V8 track
    pipeline.add_stage(IterativeMatchStage())  # Multi-pass gap filling
    pipeline.add_stage(OutputStage())

    return pipeline


def create_healing_pipeline(
    config: 'Config',
    project_dir: Path,
    audio_first_mode: bool = False
) -> Tuple['PipelineOrchestrator', Optional['HealingOrchestrator'], Optional['ResilientRunner']]:
    """
    Create a pipeline with self-healing enabled (default behavior).

    Uses config.healing settings to control healing behavior.
    If healing is disabled, returns (pipeline, None, None) for standard execution.

    Args:
        config: Configuration object
        project_dir: Project directory path
        audio_first_mode: If True, uses audio-first download pipeline

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
    pipeline = create_default_pipeline(config, project_dir, audio_first_mode)

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

    return pipeline, orchestrator, runner


def _collect_escalation_metrics(pipeline, orchestrator) -> None:
    """Collect escalation metrics from download stages and pass to orchestrator.

    Searches pipeline stages for a DownloadStage with a downloader that has
    an escalation_manager, then builds a RateLimitMetricsAggregator and passes
    unified metrics to the orchestrator for inclusion in the end-of-run report.

    Also wires the shared EscalationManager into the DownloadHealer so it
    uses the same escalation state instead of a duplicate CookieRotator.
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
                # Also keep backward-compat escalation metrics
                orchestrator.set_escalation_metrics(esc_mgr.get_metrics())
                # Wire shared EscalationManager into DownloadHealer
                if hasattr(orchestrator, 'wire_escalation_manager'):
                    orchestrator.wire_escalation_manager(esc_mgr)
                return
    except Exception:
        pass  # Non-critical: metrics collection should never break the pipeline


def run_pipeline_with_healing(
    config: 'Config',
    project_dir: Path,
    audio_first_mode: bool = False,
    resume: bool = True
) -> bool:
    """
    Convenience function to create and run a self-healing pipeline.

    Args:
        config: Configuration object
        project_dir: Project directory path
        audio_first_mode: If True, uses audio-first download pipeline
        resume: Whether to resume from checkpoint

    Returns:
        True if pipeline completed successfully
    """
    pipeline, orchestrator, runner = create_healing_pipeline(
        config, project_dir, audio_first_mode
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

        return success
    else:
        # Fallback to standard execution
        return pipeline.run(resume=resume)
