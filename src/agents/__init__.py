"""
Self-healing agents for pipeline execution.

This module provides resilient pipeline execution with automatic error recovery.
Each healer specializes in a category of errors and knows how to fix them.
The orchestrator coordinates healers and manages the healing process.

Usage:
    # Simple usage (no orchestrator)
    from src.agents import ResilientRunner, create_resilient_pipeline

    pipeline, runner = create_resilient_pipeline(config, project_dir)
    success = runner.run_pipeline(pipeline)
    runner.print_summary()

    # With orchestrator (recommended for production)
    from src.agents import (
        create_orchestrated_pipeline,
        HealingStrategy,
        HealingOrchestrator,
    )

    pipeline, orchestrator, runner = create_orchestrated_pipeline(
        config, project_dir,
        strategy=HealingStrategy.aggressive()
    )

    # Run preflight checks
    issues = orchestrator.run_preflight(pipeline.state)
    if issues:
        orchestrator.fix_preflight_issues(issues, pipeline.state)

    # Run pipeline
    success = runner.run_pipeline(pipeline)
    orchestrator.print_report()
"""

from .base import Healer, HealerResult, HealerAction
from .strategy import (
    HealingStrategy,
    HealingMode,
    HealingMetrics,
    ConfigSnapshot,
)
from .orchestrator import (
    HealingOrchestrator,
    PreflightIssue,
    EscalationRequest,
    create_orchestrated_pipeline,
)
from .runner import (
    ResilientRunner,
    create_resilient_pipeline,
)
from .healers import (
    OTIOHealer,
    APIHealer,
    CheckpointHealer,
    DownloadHealer,
    DiskHealer,
    PathHealer,
    HEALER_REGISTRY,
)

__all__ = [
    # Base classes
    'Healer',
    'HealerResult',
    'HealerAction',

    # Strategy
    'HealingStrategy',
    'HealingMode',
    'HealingMetrics',
    'ConfigSnapshot',

    # Orchestrator
    'HealingOrchestrator',
    'PreflightIssue',
    'EscalationRequest',

    # Runner
    'ResilientRunner',

    # Factory functions
    'create_resilient_pipeline',
    'create_orchestrated_pipeline',

    # Individual healers
    'OTIOHealer',
    'APIHealer',
    'CheckpointHealer',
    'DownloadHealer',
    'DiskHealer',
    'PathHealer',
    'HEALER_REGISTRY',
]
