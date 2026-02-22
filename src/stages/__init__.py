"""
Pipeline Stage Base Classes

Provides the abstract Stage class that all pipeline stages inherit from,
and the StageResult type for returning success/failure with data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

from ..logging_templates import log_error_with_context

_stages_logger = logging.getLogger(__name__)


class StageType(Enum):
    """Categorizes stages by their primary function for metrics validation."""
    DOWNLOAD = "download"       # Downloads files from external sources
    PROCESSING = "processing"  # Transforms/processes data
    ANALYSIS = "analysis"       # Analyzes and matches data
    OUTPUT = "output"           # Generates output artifacts


# Mapping of stage types to their required StageMetrics fields
# US-106-005: Extended to include all required fields per stage type
STAGE_METRICS_SCHEMA: Dict[StageType, Set[str]] = {
    StageType.DOWNLOAD: {
        'items_processed',
        'items_failed',
        'duration_seconds',
    },
    StageType.PROCESSING: {
        'duration_seconds',
    },
    StageType.ANALYSIS: {
        'items_processed',
    },
    StageType.OUTPUT: {
        'items_processed',
    },
}


class DependencyError(Exception):
    """Raised when a stage's dependencies are not satisfied."""
    pass


class BatchFailureThresholdExceeded(Exception):
    """Raised when a batch stage's failure rate exceeds the configured threshold.

    US-81-009: Allows stages to abort early when too many items are failing,
    rather than grinding through an entire batch with a high failure rate.
    """

    def __init__(self, failure_rate: float, threshold: float, processed: int,
                 failed: int, failed_items: list = None):
        self.failure_rate = failure_rate
        self.threshold = threshold
        self.processed = processed
        self.failed = failed
        self.failed_items = failed_items or []
        super().__init__(
            f"Batch failure rate {failure_rate:.1%} exceeds threshold {threshold:.1%} "
            f"({failed}/{processed} items failed)"
        )


def check_batch_failure_threshold(
    items_processed: int,
    items_failed: int,
    threshold: float,
    failed_items: list = None,
) -> None:
    """Check if the batch failure rate exceeds the configured threshold.

    US-81-009: Called after each batch item completes. Raises
    BatchFailureThresholdExceeded if the failure rate exceeds the threshold.

    Args:
        items_processed: Total items processed so far (successes + failures).
        items_failed: Number of items that have failed so far.
        threshold: Maximum allowed failure rate (0.0 to 1.0). Values >= 1.0 disable the check.
        failed_items: Optional list of failure details for error reporting.

    Raises:
        BatchFailureThresholdExceeded: When failure_rate > threshold.
    """
    if threshold >= 1.0 or items_processed == 0:
        return
    failure_rate = items_failed / items_processed
    if failure_rate > threshold:
        raise BatchFailureThresholdExceeded(
            failure_rate=failure_rate,
            threshold=threshold,
            processed=items_processed,
            failed=items_failed,
            failed_items=failed_items,
        )


@dataclass
class StageMetrics:
    """
    Metrics collected during stage execution.

    Attributes:
        items_processed: Number of items successfully processed
        items_failed: Number of items that failed processing
        duration_seconds: Time taken to execute the stage
        failed: Whether the stage failed (True) or succeeded (False)
        error_categories: Per-category error counts (e.g. {'bot_detection': 5, 'network': 2})
        error_rate: Error rate as ratio (errors / items_processed) (US-138-010)
        escalation_summary: Escalation tier breakdown (US-49-012)
        items_per_second: Overall throughput (items_processed / duration)
        peak_items_per_second: Peak throughput from sliding window samples
        throughput_samples: Per-item throughput samples for sliding window analysis
        retry_attempts: Total retry attempts made during stage execution (US-88-003)
        extra_metrics: Stage-specific metrics dict (US-90-009)
    """
    items_processed: int = 0
    items_failed: int = 0
    duration_seconds: float = 0.0
    failed: bool = False
    error_categories: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0  # US-138-010: Error rate as ratio (errors / items_processed)
    escalation_summary: Dict[str, Any] = field(default_factory=dict)
    items_per_second: float = 0.0
    peak_items_per_second: float = 0.0
    throughput_samples: List[float] = field(default_factory=list)
    retry_attempts: int = 0  # US-88-003: Track retry attempts for observability
    timeout_warning: bool = False  # US-106-002: Stage approached timeout threshold
    timeout_occurred: bool = False  # US-106-002: Stage timed out
    timeout_count: int = 0  # US-108-002: Number of timeouts for this stage
    timeout_duration: float = 0.0  # US-108-002: Total duration of timeout before cancellation
    was_force_killed: bool = False  # US-108-002: Whether stage was force-killed due to timeout
    health_check_results: List[Dict[str, Any]] = field(default_factory=list)  # US-88-005: Health check results
    extra_metrics: Dict[str, Any] = field(default_factory=dict)  # US-90-009: Stage-specific metrics (e.g., caption metrics)
    api_cost: float = 0.0  # US-162-010: Cumulative API cost at stage completion

    def compute_throughput(self) -> None:
        """Compute items_per_second from throughput_samples using sliding window.

        Uses a sliding window of the last 10 samples for recent throughput,
        and tracks peak throughput across all windows.
        """
        if not self.throughput_samples:
            # Fall back to overall average
            if self.duration_seconds > 0 and self.items_processed > 0:
                self.items_per_second = self.items_processed / self.duration_seconds
            return

        window_size = 10
        peak = 0.0
        for i in range(len(self.throughput_samples)):
            window = self.throughput_samples[max(0, i - window_size + 1):i + 1]
            window_avg = sum(window) / len(window) if window else 0.0
            if window_avg > peak:
                peak = window_avg

        # Recent throughput = average of last `window_size` samples
        recent = self.throughput_samples[-window_size:]
        self.items_per_second = sum(recent) / len(recent) if recent else 0.0
        self.peak_items_per_second = peak

    def compute_error_rate(self) -> None:
        """Compute error_rate from items_failed / items_processed.

        US-138-010: Calculate error rate as ratio of failed items to processed items.
        """
        if self.items_processed > 0:
            self.error_rate = self.items_failed / self.items_processed
        else:
            self.error_rate = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        d = {
            'items_processed': self.items_processed,
            'items_failed': self.items_failed,
            'duration_seconds': self.duration_seconds,
            'failed': self.failed
        }
        if self.error_categories:
            d['error_categories'] = dict(self.error_categories)
        if self.error_rate > 0:  # US-138-010: Include error rate
            d['error_rate'] = round(self.error_rate, 4)
        if self.escalation_summary:
            d['escalation_summary'] = dict(self.escalation_summary)
        if self.items_per_second > 0:
            d['items_per_second'] = round(self.items_per_second, 3)
        if self.peak_items_per_second > 0:
            d['peak_items_per_second'] = round(self.peak_items_per_second, 3)
        if self.throughput_samples:
            d['throughput_samples'] = [round(s, 3) for s in self.throughput_samples]
        if self.retry_attempts > 0:  # US-88-003: Include retry_attempts in serialization
            d['retry_attempts'] = self.retry_attempts
        if self.health_check_results:  # US-88-005: Include health check results
            d['health_check_results'] = self.health_check_results
        if self.extra_metrics:  # US-90-009: Include stage-specific metrics
            d['extra_metrics'] = dict(self.extra_metrics)
        # US-108-002: Include timeout metrics
        if self.timeout_warning:
            d['timeout_warning'] = True
        if self.timeout_occurred:
            d['timeout_occurred'] = True
        if self.timeout_count > 0:
            d['timeout_count'] = self.timeout_count
        if self.timeout_duration > 0:
            d['timeout_duration'] = round(self.timeout_duration, 3)
        if self.was_force_killed:
            d['was_force_killed'] = True
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageMetrics':
        """Create StageMetrics from dictionary."""
        return cls(
            items_processed=data.get('items_processed', 0),
            items_failed=data.get('items_failed', 0),
            duration_seconds=data.get('duration_seconds', 0.0),
            failed=data.get('failed', False),
            error_categories=data.get('error_categories', {}),
            error_rate=data.get('error_rate', 0.0),  # US-138-010
            escalation_summary=data.get('escalation_summary', {}),
            items_per_second=data.get('items_per_second', 0.0),
            peak_items_per_second=data.get('peak_items_per_second', 0.0),
            throughput_samples=data.get('throughput_samples', []),
            health_check_results=data.get('health_check_results', []),
            extra_metrics=data.get('extra_metrics', {}),
            # US-108-002: Timeout metrics
            timeout_warning=data.get('timeout_warning', False),
            timeout_occurred=data.get('timeout_occurred', False),
            timeout_count=data.get('timeout_count', 0),
            timeout_duration=data.get('timeout_duration', 0.0),
            was_force_killed=data.get('was_force_killed', False),
        )

    def validate_metrics(self, stage_type: StageType) -> List[str]:
        """Validate that required fields are populated for the given stage type.

        Args:
            stage_type: The type of stage that generated these metrics

        Returns:
            List of warning messages for missing required fields (empty if valid)
        """
        warnings: List[str] = []
        required_fields = STAGE_METRICS_SCHEMA.get(stage_type, set())

        for field_name in required_fields:
            value = getattr(self, field_name, None)
            # Check if the field is populated (non-zero/non-empty)
            if value == 0 or value == 0.0 or value == {} or value is None:
                warnings.append(
                    f"StageMetrics: required field '{field_name}' not populated for stage type '{stage_type.value}'"
                )

        return warnings


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
        stage_type: Type of stage for metrics validation (optional)
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Optional[StageMetrics] = None
    stage_type: Optional[StageType] = None

    def __post_init__(self) -> None:
        """Validate metrics if stage_type is provided."""
        if self.metrics and self.stage_type:
            metric_warnings = self.metrics.validate_metrics(self.stage_type)
            for warning in metric_warnings:
                self.warnings.append(warning)

    @classmethod
    def ok(cls, data: Dict[str, Any] = None, warnings: List[str] = None,
           metrics: StageMetrics = None, stage_type: StageType = None) -> 'StageResult':
        """Create a successful result"""
        return cls(success=True, data=data or {}, warnings=warnings or [], metrics=metrics, stage_type=stage_type)

    @classmethod
    def fail(cls, error: str, warnings: List[str] = None,
             metrics: StageMetrics = None, stage_type: StageType = None) -> 'StageResult':
        """Create a failed result"""
        return cls(success=False, error=error, warnings=warnings or [], metrics=metrics, stage_type=stage_type)

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

    # Explicit stage dependencies: list of stage names that must complete before this stage.
    # Defaults to empty list (no dependencies) for backward compatibility.
    DEPENDS_ON: List[str] = []

    # State attributes this stage produces (sets on PipelineState).
    # Used for documentation and future dependency resolution.
    PRODUCES: List[str] = []

    # Whether this stage is critical to pipeline success.
    # If False, failure logs a warning but pipeline continues (US-85-012).
    # Default is True (all stages critical by default for backward compatibility).
    CRITICAL: bool = True

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

    def is_abort_requested(self, state: 'PipelineState') -> bool:
        """
        Check if pipeline abort has been requested.

        US-138-004: Stages can call this periodically to check if they should
        abort gracefully. This allows for clean shutdown during long operations.

        Args:
            state: Pipeline state object

        Returns:
            True if abort has been requested, False otherwise
        """
        # Check if abort was requested via state (set by orchestrator)
        return getattr(state, 'abort_requested', False)

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """
        Get input/output information for dry-run preview.

        Override in subclasses to provide accurate counts.

        Args:
            state: Pipeline state object
            config: Configuration object

        Returns:
            Dict with 'inputs' and 'outputs' descriptions and optional counts
        """
        return {
            'inputs': 'state data',
            'outputs': 'state data',
            'input_count': None,
            'output_count': None,
            'api_estimates': {},
        }

    def get_api_estimates(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """
        Get estimated API calls for dry-run preview.

        Override in subclasses to provide API call estimates.

        Args:
            state: Pipeline state object
            config: Configuration object

        Returns:
            Dict with estimated API calls:
            - 'youtube_api_calls': Estimated YouTube Data API calls
            - 'caption_fetch_attempts': Estimated caption fetch attempts
            - 'embedding_calls': Estimated embedding API calls
            - 'llm_calls': Estimated LLM API calls
            - 'estimated_cost_usd': Estimated cost in USD
            - 'estimated_duration_seconds': Estimated duration in seconds
        """
        return {}

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
            log_error_with_context(_stages_logger, "PIPE-002", f"Failed to convert {state_type} to PipelineState: {e}")
            # Return original state - stage will handle missing attributes
            return state

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


# Stage registry for dynamic loading
_stage_registry: Dict[str, type] = {}


def register_stage(stage_class: type) -> type:
    """Decorator to register a stage class.

    Logs warnings for:
    - Duplicate registration (last-wins behavior preserved)
    - Orphan stages whose name is not in STAGE_ORDER
    """
    if hasattr(stage_class, 'name') and stage_class.name:
        stage_name = stage_class.name

        # Warn on duplicate registration
        if stage_name in _stage_registry:
            _stages_logger.warning(
                f"Duplicate stage registration: '{stage_name}' "
                f"(replacing {_stage_registry[stage_name].__name__} "
                f"with {stage_class.__name__})"
            )

        _stage_registry[stage_name] = stage_class

        # Warn if stage name is not in STAGE_ORDER (orphan stage)
        from ..checkpoint import STAGE_ORDER
        if stage_name not in STAGE_ORDER:
            _stages_logger.warning(
                f"Orphan stage registered: '{stage_name}' is not in STAGE_ORDER "
                f"(no checkpoint support)"
            )

    return stage_class


def get_stage(name: str) -> Optional[type]:
    """Get a registered stage class by name"""
    return _stage_registry.get(name)


def list_stages() -> List[str]:
    """List all registered stage names"""
    return list(_stage_registry.keys())


def get_orphan_stages() -> Dict[str, type]:
    """Return registered stages whose names are not in STAGE_ORDER.

    Orphan stages have no checkpoint support and won't be executed
    by the standard pipeline runner.

    Returns:
        Dict mapping orphan stage names to their classes.
    """
    from ..checkpoint import STAGE_ORDER
    return {
        name: cls for name, cls in _stage_registry.items()
        if name not in STAGE_ORDER
    }


# Auto-generate default values for state attributes from PipelineState dataclass.
# This eliminates manual drift between PipelineState field definitions and the
# defaults dict used by validate_required_state_attrs().
def _build_state_attr_defaults() -> Dict[str, Any]:
    """Build defaults dict from PipelineState dataclass field annotations."""
    import dataclasses as _dc
    from ..state import PipelineState
    defaults = {}
    for f in _dc.fields(PipelineState):
        if f.default is not _dc.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not _dc.MISSING:
            defaults[f.name] = f.default_factory()
        # Fields with no default at all are skipped (none in PipelineState currently)
    return defaults

_STATE_ATTR_DEFAULTS: Dict[str, Any] = _build_state_attr_defaults()


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


# =============================================================================
# US-89-007: Stage Input/Output Validation Layer
# =============================================================================

@dataclass
class StageContract:
    """
    Contract defining a stage's input and output requirements.

    US-89-007: Validates that stages have required inputs before execution,
    catching dict-vs-object bugs early and preventing runtime failures.

    Attributes:
        required_inputs: List of state attribute names that must exist before stage runs.
            Missing required inputs will cause stage to fail.
        optional_inputs: List of state attribute names that are optional.
            Missing optional inputs will log a warning but stage will continue.
        outputs: List of state attribute names that this stage produces.
            Used for validation after stage completion.
    """
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


class StageValidator:
    """
    Validates stage input/output contracts.

    US-89-007: Checks that PipelineState has required fields before a stage runs,
    and validates output completeness after stage completion. This prevents
    AttributeError crashes from dict-vs-object bugs and missing state data.
    """

    @staticmethod
    def validate_inputs(
        state: 'PipelineState',
        contract: StageContract,
        stage_name: str
    ) -> tuple[List[str], List[str]]:
        """
        Validate that state has required inputs before stage execution.

        Args:
            state: Pipeline state to validate
            contract: Stage contract defining input requirements
            stage_name: Name of stage for logging

        Returns:
            Tuple of (error_messages, warning_messages).
            Errors are for missing required inputs, warnings for missing optional.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required inputs - value must not be None (catches checkpoint restoration failures)
        for attr in contract.required_inputs:
            value = getattr(state, attr, None)
            if value is None:
                errors.append(
                    f"[{stage_name}] Missing required input '{attr}'"
                )

        # Check optional inputs - these are warnings only
        for attr in contract.optional_inputs:
            value = getattr(state, attr, None)
            if value is None:
                warnings.append(
                    f"[{stage_name}] Missing optional input '{attr}'"
                )

        return errors, warnings

    @staticmethod
    def validate_outputs(
        state: 'PipelineState',
        contract: StageContract,
        stage_name: str
    ) -> List[str]:
        """
        Validate that stage produced required outputs after completion.

        Args:
            state: Pipeline state to validate
            contract: Stage contract defining output requirements
            stage_name: Name of stage for logging

        Returns:
            List of warning messages for missing outputs (not errors - output
            validation is informational).
        """
        warnings: List[str] = []

        for attr in contract.outputs:
            value = getattr(state, attr, None)
            # Only warn if the output is None (truly missing), not if empty
            # Empty outputs are valid - the stage might have legitimately found nothing
            if value is None:
                warnings.append(
                    f"[{stage_name}] Missing expected output '{attr}'"
                )

        return warnings


def contract_validation(contract: StageContract):
    """
    Decorator to validate stage input/output contracts.

    Wraps stage run() method to validate:
    1. Required inputs exist before execution (raises error if missing)
    2. Optional inputs exist before execution (logs warning if missing)
    3. Outputs are populated after execution (logs warning if missing)

    Usage:
        @contract_validation(StageContract(
            required_inputs=['voiceover_segments', 'keywords'],
            optional_inputs=['text_metadata'],
            outputs=['matches']
        ))
        class MatchStage(Stage):
            ...
    """
    def decorator(stage_class: type) -> type:
        # Store contract on the class
        stage_class.CONTRACT = contract

        # Wrap the run method if it exists
        original_run = getattr(stage_class, 'run', None)
        if original_run and hasattr(original_run, '__func__'):
            def wrapped_run(self, state, config, checkpoint):
                stage_name = self.name or stage_class.__name__

                # Validate inputs BEFORE running
                errors, warnings = StageValidator.validate_inputs(state, contract, stage_name)

                # Log warnings for optional inputs
                for warning in warnings:
                    _stages_logger.warning(warning)

                # Fail for missing required inputs
                if errors:
                    for error in errors:
                        log_error_with_context(_stages_logger, "PIPE-001", error)
                    return StageResult.fail(
                        f"Missing required inputs: {[e.split(']', 1)[1].strip() for e in errors]}"
                    )

                # Execute the original run method
                result = original_run(self, state, config, checkpoint)

                # Validate outputs AFTER completion (if stage succeeded)
                if result.success:
                    output_warnings = StageValidator.validate_outputs(state, contract, stage_name)
                    for warning in output_warnings:
                        _stages_logger.warning(warning)
                    # Add output warnings to result
                    if output_warnings:
                        result.warnings.extend(output_warnings)

                return result

            # Replace the run method
            stage_class.run = wrapped_run

        return stage_class
    return decorator


# =============================================================================
# US-89-011: Stage Dependency Graph for Debugging
# =============================================================================

# =============================================================================
# US-108-011: Stage Input/Output Validation Contracts
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type


@dataclass
class FieldSchema:
    """Schema definition for a single field in stage input/output.

    US-108-011: Defines the expected type and constraints for a field.

    Attributes:
        field_name: Name of the field
        expected_type: Expected Python type (e.g., dict, list, str)
        required: Whether the field is required (True) or optional (False)
        description: Human-readable description of the field
        nested_schema: Optional schema for nested dataclass validation
    """
    field_name: str
    expected_type: Type | str  # Type or type name string for flexibility
    required: bool = True
    description: str = ""
    nested_schema: Optional['StageInputSchema' | 'StageOutputSchema'] = None


@dataclass
class StageInputSchema:
    """Schema definition for stage input data.

    US-108-011: Defines what data a stage expects to receive from checkpoint.

    Attributes:
        stage_name: Name of the stage this schema applies to
        fields: List of field schemas defining expected input fields
    """
    stage_name: str
    fields: List[FieldSchema] = field(default_factory=list)

    def get_required_fields(self) -> Set[str]:
        """Get set of required field names."""
        return {f.field_name for f in self.fields if f.required}

    def get_all_fields(self) -> Set[str]:
        """Get set of all field names (required + optional)."""
        return {f.field_name for f in self.fields}


@dataclass
class StageOutputSchema:
    """Schema definition for stage output data.

    US-108-011: Defines what data a stage produces and saves to checkpoint.

    Attributes:
        stage_name: Name of the stage this schema applies to
        fields: List of field schemas defining expected output fields
    """
    stage_name: str
    fields: List[FieldSchema] = field(default_factory=list)

    def get_required_fields(self) -> Set[str]:
        """Get set of required field names."""
        return {f.field_name for f in self.fields if f.required}

    def get_all_fields(self) -> Set[str]:
        """Get set of all field names (required + optional)."""
        return {f.field_name for f in self.fields}


# Stage schema registry - maps stage name to input/output schemas
_STAGE_SCHEMAS: Dict[str, Dict[str, StageInputSchema | StageOutputSchema]] = {}

# Valid PipelineState attributes for output field validation
VALID_PIPELINE_STATE_ATTRIBUTES: Set[str] = {
    "voiceover_path",
    "voiceover_segments",
    "keywords",
    "topic_context",
    "extracted_entities",
    "video_ids",
    "video_search_results",
    "search_failed_keywords",
    "caption_results",
    "text_metadata",
    "matches",
    "alternatives",
    "downloaded_segments",
    "output_files",
    "otio_files",
    "entity_images",
    "entity_videos",
    "voiceover_embeddings",
    "face_preference",
    "location_chapters",
    "listicle_groups",
    "stage_timings",
    "partial_failures",
}

logger = logging.getLogger(__name__)


def validate_stage_schema(
    input_schema: Optional[StageInputSchema] = None,
    output_schema: Optional[StageOutputSchema] = None,
) -> List[str]:
    """Validate stage schema(s) at registration time.

    US-125-004: Validates schemas when they are registered to catch issues early.

    Checks:
    - Required vs optional input fields don't overlap (duplicate field names)
    - Output fields reference valid PipelineState attributes
    - Field names are valid identifiers

    Args:
        input_schema: Schema defining expected inputs
        output_schema: Schema defining expected outputs

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []

    # Validate input schema
    if input_schema:
        stage_name = input_schema.stage_name

        # Check for duplicate field names in input schema
        field_names = [f.field_name for f in input_schema.fields]
        if len(field_names) != len(set(field_names)):
            duplicates = [f for f in field_names if field_names.count(f) > 1]
            errors.append(
                f"Stage '{stage_name}' input schema has duplicate fields: {set(duplicates)}"
            )

        # Validate field names are valid identifiers
        for f in input_schema.fields:
            if not f.field_name.isidentifier():
                errors.append(
                    f"Stage '{stage_name}' input field '{f.field_name}' is not a valid identifier"
                )

    # Validate output schema
    if output_schema:
        stage_name = output_schema.stage_name

        # Check for duplicate field names in output schema
        field_names = [f.field_name for f in output_schema.fields]
        if len(field_names) != len(set(field_names)):
            duplicates = [f for f in field_names if field_names.count(f) > 1]
            errors.append(
                f"Stage '{stage_name}' output schema has duplicate fields: {set(duplicates)}"
            )

        # Validate output fields reference valid PipelineState attributes
        for f in output_schema.fields:
            if f.field_name not in VALID_PIPELINE_STATE_ATTRIBUTES:
                errors.append(
                    f"Stage '{stage_name}' output field '{f.field_name}' "
                    f"is not a valid PipelineState attribute. "
                    f"Valid attributes: {sorted(VALID_PIPELINE_STATE_ATTRIBUTES)}"
                )

            # Validate field names are valid identifiers
            if not f.field_name.isidentifier():
                errors.append(
                    f"Stage '{stage_name}' output field '{f.field_name}' is not a valid identifier"
                )

    return errors


def register_stage_schemas(
    input_schema: Optional[StageInputSchema] = None,
    output_schema: Optional[StageOutputSchema] = None,
) -> None:
    """Register input/output schemas for a stage.

    US-108-011: Called by stages to register their contract schemas.
    US-125-004: Validates schemas at registration time.

    Args:
        input_schema: Schema defining expected inputs
        output_schema: Schema defining expected outputs
    """
    # Validate schemas at registration time (US-125-004)
    validation_errors = validate_stage_schema(input_schema, output_schema)
    if validation_errors:
        for error in validation_errors:
            log_error_with_context(logger, "CFG-001", f"Schema validation error: {error}")

    if input_schema:
        stage_name = input_schema.stage_name
        if stage_name not in _STAGE_SCHEMAS:
            _STAGE_SCHEMAS[stage_name] = {}
        _STAGE_SCHEMAS[stage_name]['input'] = input_schema
        logger.debug(f"Registered input schema for stage '{stage_name}'")

    if output_schema:
        stage_name = output_schema.stage_name
        if stage_name not in _STAGE_SCHEMAS:
            _STAGE_SCHEMAS[stage_name] = {}
        _STAGE_SCHEMAS[stage_name]['output'] = output_schema
        logger.debug(f"Registered output schema for stage '{stage_name}'")


def get_stage_schemas(stage_name: str) -> Dict[str, StageInputSchema | StageOutputSchema]:
    """Get registered schemas for a stage.

    Args:
        stage_name: Name of the stage

    Returns:
        Dict with 'input' and/or 'output' schema if registered
    """
    return _STAGE_SCHEMAS.get(stage_name, {})


def get_all_schemas() -> Dict[str, Dict[str, StageInputSchema | StageOutputSchema]]:
    """Get all registered stage schemas.

    Returns:
        Dict mapping stage name to its schemas
    """
    return dict(_STAGE_SCHEMAS)


def check_orphan_stages() -> List[str]:
    """Check for orphan stages (registered but without schemas).

    US-125-004: Logs warning for stages that are registered but don't have schemas.

    Returns:
        List of warning messages for orphan stages
    """
    warnings: List[str] = []

    # Ensure stages are registered first
    _ensure_stages_registered()

    # Get all registered stages
    all_stages = get_all_stages()

    # Check each stage for schema
    for stage_name in all_stages.keys():
        if stage_name not in _STAGE_SCHEMAS:
            warnings.append(
                f"Stage '{stage_name}' is registered but has no contract schema defined. "
                f"Consider adding input/output schemas for contract validation."
            )
            logger.warning(warnings[-1])

    return warnings


def _ensure_stages_registered() -> None:
    """Ensure all pipeline stages are registered.

    Imports stage modules to trigger @register_stage decorators.
    This must be called before building the dependency graph.
    """
    global _stage_registry
    if not _stage_registry:
        # Import all stage modules to trigger registration
        from .analyze import AnalyzeStage
        from .video_search import VideoSearchStage
        from .caption_stage import CaptionStage
        from .match import MatchStage
        from .iterative_match import IterativeMatchStage
        from .download_segments import DownloadVideoSegmentsStage
        from .output import OutputStage


def get_all_stages() -> Dict[str, type]:
    """Get all registered stages that have DEPENDS_ON defined.

    Returns:
        Dict mapping stage name to stage class.
    """
    _ensure_stages_registered()
    return dict(_stage_registry)


def build_dependency_graph() -> Dict[str, List[str]]:
    """Build a dependency graph from registered stages.

    Returns:
        Dict mapping stage name to list of stage names it depends on.
    """
    _ensure_stages_registered()
    graph: Dict[str, List[str]] = {}
    for name, cls in _stage_registry.items():
        deps = getattr(cls, 'DEPENDS_ON', [])
        if deps:
            graph[name] = deps
        else:
            graph[name] = []
    return graph


def validate_no_cycles() -> Optional[str]:
    """Validate that the dependency graph has no circular dependencies.

    Returns:
        Error message if cycles found, None if graph is valid.
    """
    graph = build_dependency_graph()

    def has_cycle_from(node: str, visited: Set[str], rec_stack: Set[str]) -> Optional[List[str]]:
        """DFS to find cycle, returns cycle path if found."""
        visited.add(node)
        rec_stack.add(node)

        for dep in graph.get(node, []):
            if dep not in visited:
                cycle = has_cycle_from(dep, visited, rec_stack)
                if cycle:
                    return [node] + cycle
            elif dep in rec_stack:
                # Found cycle
                return [node, dep]

        rec_stack.remove(node)
        return None

    visited: Set[str] = set()
    for node in graph:
        if node not in visited:
            cycle = has_cycle_from(node, visited, set())
            if cycle:
                return f"Circular dependency detected: {' -> '.join(cycle)}"

    return None


def get_parallel_groups() -> List[List[str]]:
    """Get stages grouped by parallelization opportunity.

    Stages in the same group have no dependencies on each other
    and can potentially run in parallel (if the pipeline supported it).

    Returns:
        List of stage name groups, ordered by execution priority.
    """
    graph = build_dependency_graph()
    all_stages = set(graph.keys())

    # Topologically sort with level tracking
    levels: Dict[str, int] = {}
    remaining = set(all_stages)

    while remaining:
        # Find nodes with no dependencies on remaining nodes
        ready = []
        for stage in remaining:
            deps = graph.get(stage, [])
            # Stage is ready if all its dependencies are already assigned a level
            if all(d not in remaining for d in deps):
                ready.append(stage)

        if not ready:
            # Circular dependency or missing dependency
            break

        # Assign level to all ready stages
        for stage in ready:
            # Level is max of dependencies + 1, or 0 if no deps
            deps = graph.get(stage, [])
            max_dep_level = max((levels.get(d, -1) for d in deps), default=-1)
            levels[stage] = max_dep_level + 1
            remaining.remove(stage)

    # Group by level
    level_groups: Dict[int, List[str]] = {}
    for stage, level in levels.items():
        level_groups.setdefault(level, []).append(stage)

    return [level_groups[l] for l in sorted(level_groups.keys())]


def generate_dot_graph() -> str:
    """Generate DOT graph representation of stage dependencies.

    Returns:
        DOT-formatted string for visualization with graphviz.
    """
    graph = build_dependency_graph()

    lines = [
        "digraph pipeline_stages {",
        '  rankdir=LR;',
        '  node [shape=box, style=rounded];',
        '  edge [color=gray50];',
        "",
    ]

    # Add nodes
    for stage in sorted(graph.keys()):
        deps = graph.get(stage, [])
        if deps:
            label = f"{stage}\\n(depends on: {', '.join(deps)})"
        else:
            label = f"{stage}\\n(root)"
        lines.append(f'  "{stage}" [label="{label}"];')

    lines.append("")

    # Add edges
    for stage, deps in sorted(graph.items()):
        for dep in deps:
            lines.append(f'  "{dep}" -> "{stage}";')

    lines.append("}")
    return "\n".join(lines)


def print_dependency_summary() -> str:
    """Generate human-readable dependency summary.

    Returns:
        Formatted string with dependency info and parallelization groups.
    """
    graph = build_dependency_graph()

    lines = [
        "=== Stage Dependency Summary ===",
        "",
    ]

    # Group by dependency count
    roots = []
    dependents = []

    for stage in sorted(graph.keys()):
        deps = graph.get(stage, [])
        if not deps:
            roots.append(stage)
        else:
            dependents.append((stage, deps))

    lines.append("Root stages (no dependencies):")
    for stage in roots:
        lines.append(f"  - {stage}")

    lines.append("")
    lines.append("Dependent stages:")
    for stage, deps in sorted(dependents):
        lines.append(f"  - {stage} depends on: {', '.join(deps)}")

    # Add parallelization info
    lines.append("")
    lines.append("=== Parallelization Opportunities ===")
    lines.append("(Stages in same group can run in parallel)")

    parallel_groups = get_parallel_groups()
    for i, group in enumerate(parallel_groups):
        lines.append(f"  Level {i}: {', '.join(group)}")

    # Add cycle validation
    lines.append("")
    lines.append("=== Dependency Validation ===")
    cycle_error = validate_no_cycles()
    if cycle_error:
        lines.append(f"  ERROR: {cycle_error}")
    else:
        lines.append("  No circular dependencies detected.")

    return "\n".join(lines)
