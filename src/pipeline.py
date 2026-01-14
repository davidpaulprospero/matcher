"""
Pipeline Orchestrator

Lightweight orchestrator that runs pipeline stages in order,
handling checkpointing and resume functionality.

This replaces the run() method and stage orchestration logic in main.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .checkpoint import CheckpointManager, STAGE_ORDER
from .state import PipelineState
from .stages import Stage, StageResult

if TYPE_CHECKING:
    from .config import Config

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
        only_stages: List[str] = None
    ) -> bool:
        """
        Run the pipeline.

        Args:
            resume: Whether to resume from checkpoint if available
            skip_stages: Stage names to skip
            only_stages: If provided, only run these stages

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

            logger.info(f"Running stage: {stage_name}")
            result = stage.run(self.state, self.config, self.checkpoint)

            elapsed = time.time() - start_time
            self.stage_timings[stage_name] = elapsed
            self.state.stage_timings[stage_name] = elapsed

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
    from .stages.transcribe import TranscribeStage
    from .stages.scene_detection import SceneDetectionStage
    from .stages.match import MatchStage
    from .stages.broll_match import BrollMatchStage
    from .stages.output import OutputStage

    # Add stages in STAGE_ORDER
    pipeline.add_stage(AnalyzeStage())
    pipeline.add_stage(EntityImagesStage())
    pipeline.add_stage(EntityVideosStage())
    pipeline.add_stage(DownloadStage())
    pipeline.add_stage(StockVideoStage())
    pipeline.add_stage(BrollDownloadStage())  # B-roll specific downloads
    pipeline.add_stage(RemixStage())
    pipeline.add_stage(TranscribeStage())
    pipeline.add_stage(SceneDetectionStage())  # Scene detection with B-roll marking
    pipeline.add_stage(MatchStage())
    pipeline.add_stage(BrollMatchStage())  # Match silent scenes for V8 track

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
    from .stages.transcribe import TranscribeStage
    from .stages.scene_detection import SceneDetectionStage
    from .stages.match import MatchStage
    from .stages.broll_match import BrollMatchStage
    from .stages.output import OutputStage

    # Add prerequisite stages for restoration only (will be skipped via checkpoint)
    pipeline.add_stage(AnalyzeStage())
    pipeline.add_stage(EntityImagesStage())  # For V9 track
    pipeline.add_stage(EntityVideosStage())  # For V10 track
    pipeline.add_stage(DownloadStage())
    pipeline.add_stage(StockVideoStage())    # For general B-roll
    pipeline.add_stage(BrollDownloadStage()) # B-roll specific downloads
    pipeline.add_stage(RemixStage())         # Filter videos by relevance
    pipeline.add_stage(TranscribeStage())
    pipeline.add_stage(SceneDetectionStage())  # Scene detection with B-roll marking

    # Add stages to actually run
    pipeline.add_stage(MatchStage())
    pipeline.add_stage(BrollMatchStage())    # Match silent scenes for V8 track
    pipeline.add_stage(OutputStage())

    return pipeline
