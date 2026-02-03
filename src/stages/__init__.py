"""
Pipeline Stage Base Classes

Provides the abstract Stage class that all pipeline stages inherit from,
and the StageResult type for returning success/failure with data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from .state import PipelineState

_stages_logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """
    Metrics collected during stage execution.

    Attributes:
        items_processed: Number of items successfully processed
        items_failed: Number of items that failed processing
        duration_seconds: Time taken to execute the stage
        failed: Whether the stage failed (True) or succeeded (False)
    """
    items_processed: int = 0
    items_failed: int = 0
    duration_seconds: float = 0.0
    failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'items_processed': self.items_processed,
            'items_failed': self.items_failed,
            'duration_seconds': self.duration_seconds,
            'failed': self.failed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageMetrics':
        """Create StageMetrics from dictionary."""
        return cls(
            items_processed=data.get('items_processed', 0),
            items_failed=data.get('items_failed', 0),
            duration_seconds=data.get('duration_seconds', 0.0),
            failed=data.get('failed', False)
        )


@dataclass
class StageResult:
    """
    Result of a pipeline stage execution.

    Attributes:
        success: Whether the stage completed successfully
        data: Stage output data (for checkpointing)
        error: Error message if failed
        warnings: Non-fatal warnings encountered
        metrics: Stage execution metrics (items processed, failed, duration)
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Optional[StageMetrics] = None

    @classmethod
    def ok(cls, data: Dict[str, Any] = None, warnings: List[str] = None, metrics: StageMetrics = None) -> 'StageResult':
        """Create a successful result"""
        return cls(success=True, data=data or {}, warnings=warnings or [], metrics=metrics)

    @classmethod
    def fail(cls, error: str, warnings: List[str] = None, metrics: StageMetrics = None) -> 'StageResult':
        """Create a failed result"""
        return cls(success=False, error=error, warnings=warnings or [], metrics=metrics)

    def __bool__(self) -> bool:
        return self.success


class Stage(ABC):
    """
    Abstract base class for pipeline stages.

    Simplified 7-stage pipeline:
    - ANALYZE: Extract keywords from voiceover
    - VIDEO_SEARCH: Search YouTube for videos (no download)
    - CAPTION: Fetch YouTube captions
    - MATCH: Match voiceover segments to video clips
    - ITERATIVE_MATCH: Fill gaps with iterative search
    - DOWNLOAD_SEGMENTS: Download matched video segments
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

    def _validate_state_type(self, state: Any) -> 'PipelineState':
        """
        Validate that state is a PipelineState instance.

        US-39-009: Ensures type safety at stage entry points. Legacy pipeline
        objects may not have all required attributes.

        If state is not a PipelineState instance:
        - Logs a WARNING with the actual state type
        - Attempts to convert using PipelineState.from_legacy_pipeline()
        - Returns the converted (or original) state

        Args:
            state: The state object passed to run()

        Returns:
            PipelineState instance (original if valid, converted if legacy)
        """
        # Import here to avoid circular imports
        from ..state import PipelineState

        if isinstance(state, PipelineState):
            return state

        # Non-standard state object - log warning
        state_type = type(state).__name__
        _stages_logger.warning(f"Non-standard state object: {state_type}")

        # Attempt to convert from legacy pipeline
        try:
            converted = PipelineState.from_legacy_pipeline(state)
            _stages_logger.info(f"Converted {state_type} to PipelineState")
            return converted
        except Exception as e:
            _stages_logger.error(f"Failed to convert {state_type} to PipelineState: {e}")
            # Return original state - stage will handle missing attributes
            return state

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


# Default values for state attributes when missing
# Must match default_factory types from PipelineState (src/state.py)
_STATE_ATTR_DEFAULTS = {
    # INPUT STATE
    'voiceover_path': '',
    'voiceover_segments': [],
    'keywords': [],
    'topic_context': '',
    'extracted_entities': [],
    # VIDEO SEARCH STATE
    'video_ids': [],
    'video_search_results': [],
    'search_failed_keywords': [],
    # CAPTION STATE
    'caption_results': {},
    'text_metadata': [],
    # MATCHING STATE
    'matches': [],
    'alternatives': {},
    # DOWNLOAD STATE
    'downloaded_segments': [],
    # OUTPUT STATE
    'output_files': [],
    'otio_files': [],
    # RUNTIME STATE
    'face_preference': 'neutral',
    'location_chapters': [],
    'stage_timings': {},
}


def validate_required_state_attrs(
    state: 'PipelineState',
    required_attrs: List[str],
    stage_name: str
) -> None:
    """
    Validate and initialize required state attributes at stage entry.

    US-40-008: Prevents AttributeError crashes when checkpoint restoration
    fails to initialize all fields. Missing attributes are initialized to
    sensible defaults and logged as warnings.

    Args:
        state: Pipeline state object to validate
        required_attrs: List of attribute names required by this stage
        stage_name: Name of the calling stage (for logging)

    Returns:
        None (modifies state in place)

    Example:
        validate_required_state_attrs(state, ['text_metadata', 'matches'], 'MATCH')
    """
    for attr in required_attrs:
        if not hasattr(state, attr):
            default = _STATE_ATTR_DEFAULTS.get(attr, None)
            _stages_logger.warning(
                f"[{stage_name}] Missing state attribute '{attr}', initializing to {type(default).__name__}"
            )
            setattr(state, attr, default)
