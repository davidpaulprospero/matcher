"""
Pipeline Stage Base Classes

Provides the abstract Stage class that all pipeline stages inherit from,
and the StageResult type for returning success/failure with data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from .state import PipelineState


@dataclass
class StageResult:
    """
    Result of a pipeline stage execution.

    Attributes:
        success: Whether the stage completed successfully
        data: Stage output data (for checkpointing)
        error: Error message if failed
        warnings: Non-fatal warnings encountered
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def ok(cls, data: Dict[str, Any] = None, warnings: List[str] = None) -> 'StageResult':
        """Create a successful result"""
        return cls(success=True, data=data or {}, warnings=warnings or [])

    @classmethod
    def fail(cls, error: str, warnings: List[str] = None) -> 'StageResult':
        """Create a failed result"""
        return cls(success=False, error=error, warnings=warnings or [])

    def __bool__(self) -> bool:
        return self.success


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage represents a discrete step in the video matching pipeline:
    - ANALYZE: Extract keywords from voiceover
    - DOWNLOAD: Download video footage
    - TRANSCRIBE: Transcribe videos and compute embeddings
    - MATCH: Match voiceover segments to video clips
    - OUTPUT: Generate OTIO timeline

    Stages are:
    - Independently testable
    - Checkpointable (can resume from failure)
    - Configurable via Config object
    """

    # Stage identifier (must be unique, matches STAGE_ORDER in checkpoint.py)
    name: str = ""

    # Human-readable description
    description: str = ""

    @abstractmethod
    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """
        Execute the stage.

        Args:
            state: Pipeline state object (modified in place)
            config: Configuration object
            checkpoint: Checkpoint manager for saving progress

        Returns:
            StageResult with success/failure and data for checkpointing
        """
        pass

    @abstractmethod
    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """
        Check if this stage can be skipped (already completed in checkpoint).

        Args:
            state: Pipeline state object
            checkpoint: Checkpoint manager

        Returns:
            True if stage can be skipped
        """
        pass

    @abstractmethod
    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """
        Restore stage output from checkpoint.

        Called when can_skip() returns True to restore state.

        Args:
            state: Pipeline state object (modified in place)
            checkpoint: Checkpoint manager

        Returns:
            True if restore succeeded
        """
        pass

    def validate_inputs(self, state: 'PipelineState', config: 'Config') -> Optional[str]:
        """
        Validate that required inputs are present before running.

        Override in subclasses to add input validation.

        Args:
            state: Pipeline state object
            config: Configuration object

        Returns:
            Error message if validation fails, None if valid
        """
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


# Stage registry for dynamic loading
_stage_registry: Dict[str, type] = {}


def register_stage(stage_class: type) -> type:
    """Decorator to register a stage class"""
    if hasattr(stage_class, 'name') and stage_class.name:
        _stage_registry[stage_class.name] = stage_class
    return stage_class


def get_stage(name: str) -> Optional[type]:
    """Get a registered stage class by name"""
    return _stage_registry.get(name)


def list_stages() -> List[str]:
    """List all registered stage names"""
    return list(_stage_registry.keys())
