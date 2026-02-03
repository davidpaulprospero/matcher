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
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

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

        Checks:
        - cache_dir is writable (or can be created)
        - Embedding provider is configured when matching stages are enabled

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        errors: List[str] = []

        # Check cache_dir is writable or can be created
        cache_dir = getattr(getattr(self.config, 'cache', None), 'cache_dir', None)
        if cache_dir:
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
        else:
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

        # Validate config before running any stages (US-44-003: fail-fast)
        config_errors = self._validate_config()
        if config_errors:
            for error in config_errors:
                logger.error(f"Config validation error: {error}")
            return False

        # Dry-run mode: log stages and validate without executing
        if dry_run:
            return self._run_dry_run(skip_stages, only_stages)

        # Build parallel stage lookup: stage_name -> tuple of parallel stage names
        parallel_groups: Dict[str, Tuple[str, ...]] = {}
        if parallel_stages:
            for group in parallel_stages:
                for stage_name in group:
                    parallel_groups[stage_name] = group

        # Try to load checkpoint if resuming
        if resume:
            self.load_checkpoint()

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
                if not stage.restore(self.state, self.checkpoint, self.config):
                    logger.warning(f"Failed to restore {stage_name} from checkpoint")
                # Validate state attributes after stage restoration
                self.state.validate_state_attributes()
                continue

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

    def _run_dry_run(
        self,
        skip_stages: set,
        only_stages: Optional[set]
    ) -> bool:
        """
        Run pipeline in dry-run mode: log stages and validate without executing.

        Args:
            skip_stages: Set of stage names to skip
            only_stages: If set, only include these stages

        Returns:
            True if all validations pass, False if any validation fails
        """
        logger.info("=" * 60)
        logger.info("DRY-RUN MODE: Previewing pipeline execution plan")
        logger.info("=" * 60)

        all_valid = True
        stages_to_run = []

        for stage in self.stages:
            stage_name = stage.name

            # Filter stages
            if stage_name in skip_stages:
                logger.info(f"  [SKIP] {stage_name} (skip_stages)")
                continue

            if only_stages and stage_name not in only_stages:
                logger.info(f"  [SKIP] {stage_name} (not in only_stages)")
                continue

            stages_to_run.append(stage)

        logger.info(f"\nStages to execute ({len(stages_to_run)}):")
        for i, stage in enumerate(stages_to_run, 1):
            logger.info(f"  {i}. {stage.name}")

        logger.info("\nValidating stage inputs:")
        for stage in stages_to_run:
            validation_error = stage.validate_inputs(self.state, self.config)
            if validation_error:
                logger.error(f"  [FAIL] {stage.name}: {validation_error}")
                all_valid = False
            else:
                logger.info(f"  [OK]   {stage.name}: inputs valid")

        logger.info("=" * 60)
        if all_valid:
            logger.info("DRY-RUN COMPLETE: All validations passed")
        else:
            logger.error("DRY-RUN COMPLETE: Validation errors found")
        logger.info("=" * 60)

        return all_valid

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

            # Save checkpoint (in stage order)
            if result.data:
                self.checkpoint.save(stage_name, result.data)

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
    audio_first_mode: bool = False  # Deprecated: now always uses caption-first
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
        audio_first_mode: Deprecated, ignored (caption-first is now default)

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
