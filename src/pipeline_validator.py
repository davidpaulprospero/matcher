"""
Pipeline Validator - US-82-006

Extracted pre-run validation logic from PipelineOrchestrator.
Independently constructable with just config and stage list (no orchestrator dependency).

Validates config, stages, and checkpoint compatibility before pipeline execution.

US-108-011: Extended with validate_stage_io() for stage input/output contract validation.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .checkpoint import STAGE_ORDER

if TYPE_CHECKING:
    from .config import Config
    from .stages import Stage
    from .stages import StageInputSchema, StageOutputSchema, FieldSchema

logger = logging.getLogger(__name__)


@dataclass
class StageValidationResult:
    """Structured validation result for a single pipeline stage.

    Attributes:
        stage_name: Name of the stage
        status: One of 'run', 'skip', 'checkpoint', 'error'
        message: Human-readable detail about the validation outcome
    """
    stage_name: str
    status: str  # 'run' | 'skip' | 'checkpoint' | 'error'
    message: str


@dataclass
class InputValidationResult:
    """Result of input file/directory validation.

    Attributes:
        path: Path that was validated
        valid: Whether the path exists/is valid
        is_warning: True for warnings, False for errors
        message: Human-readable message
    """
    path: str
    valid: bool
    is_warning: bool = False
    message: str = ""


@dataclass
class ContractViolation:
    """US-108-011: Details of a stage contract violation.

    Attributes:
        stage_name: Name of the stage
        violation_type: 'input' or 'output'
        field_name: Name of the field that violated the contract
        expected: Description of what was expected
        actual: Description of what was actually found
        severity: 'error' or 'warning'
    """
    stage_name: str
    violation_type: str  # 'input' or 'output'
    field_name: str
    expected: str
    actual: str
    severity: str = "error"  # 'error' or 'warning'


class PipelineValidator:
    """Pre-run validation for pipeline configuration, stages, and checkpoints.

    Independently constructable with just config and stage list — no orchestrator
    dependency. Returns structured StageValidationResult objects.

    Usage:
        validator = PipelineValidator(config, stages)
        results = validator.validate_all()
        errors = [r for r in results if r.status == 'error']
    """

    def __init__(self, config: 'Config', stages: List['Stage']):
        self.config = config
        self.stages = stages

    def validate_config(self) -> List[str]:
        """Validate pipeline configuration (schema + runtime).

        Returns:
            List of validation error strings. Empty list means config is valid.
        """
        errors = self._validate_config_schema()
        errors.extend(self._validate_runtime_environment())
        return errors

    def _validate_config_schema(self) -> List[str]:
        """Pure config schema validation — no I/O, no filesystem access.

        Checks required fields, type constraints, and value ranges.

        Returns:
            List of validation error strings.
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
        """Runtime environment checks — requires filesystem/PATH access.

        Returns:
            List of validation error strings.
        """
        errors: List[str] = []

        cache_dir = getattr(getattr(self.config, 'cache', None), 'cache_dir', None)
        if cache_dir and isinstance(cache_dir, str):
            cache_path = Path(cache_dir)
            if cache_path.exists():
                if not os.access(str(cache_path), os.W_OK):
                    errors.append(
                        f"Cache directory is not writable: {cache_dir}"
                    )
            else:
                parent = cache_path.parent
                if parent.exists() and not os.access(str(parent), os.W_OK):
                    errors.append(
                        f"Cannot create cache directory (parent not writable): {cache_dir}"
                    )

        # Warn if cookies_from_browser is set but browser not on PATH
        self._warn_cookies_from_browser()

        return errors

    def _warn_cookies_from_browser(self) -> None:
        """Check if cookies_from_browser browser is findable on PATH."""
        download_config = getattr(self.config, 'download', None)
        if download_config is None:
            return
        browser = getattr(download_config, 'cookies_from_browser', '')
        if not browser:
            return

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

    def validate_input_files(
        self,
        state=None,
    ) -> List[InputValidationResult]:
        """Validate existence of input files (voiceover, config, etc.).

        Args:
            state: Optional PipelineState containing voiceover_path and project_dir.

        Returns:
            List of InputValidationResult for each validated path.
        """
        results: List[InputValidationResult] = []

        # Validate voiceover file
        if state is not None:
            voiceover_path = getattr(state, 'voiceover_path', None)
            if voiceover_path:
                voiceover_path = str(voiceover_path)
                if not Path(voiceover_path).exists():
                    results.append(InputValidationResult(
                        path=voiceover_path,
                        valid=False,
                        is_warning=False,
                        message=f"Voiceover file not found: {voiceover_path}",
                    ))
                else:
                    results.append(InputValidationResult(
                        path=voiceover_path,
                        valid=True,
                        message=f"Voiceover file exists",
                    ))

        # Validate config file path if provided
        config_path = getattr(self.config, 'config_path', None)
        if config_path:
            config_path = str(config_path)
            if not Path(config_path).exists():
                results.append(InputValidationResult(
                    path=config_path,
                    valid=False,
                    is_warning=False,
                    message=f"Config file not found: {config_path}",
                ))
            else:
                results.append(InputValidationResult(
                    path=config_path,
                    valid=True,
                    message=f"Config file exists",
                ))

        return results

    def validate_output_directory(
        self,
        state=None,
    ) -> List[InputValidationResult]:
        """Validate output directory exists and is writable.

        Args:
            state: Optional PipelineState containing project_dir.

        Returns:
            List of InputValidationResult for output directory validation.
        """
        results: List[InputValidationResult] = []

        # Get project directory from state or config
        project_dir = None
        if state is not None:
            project_dir = getattr(state, 'project_dir', None)

        if not project_dir:
            project_dir = getattr(self.config, 'project_dir', '.')

        project_dir = str(project_dir)
        project_path = Path(project_dir)

        # Check if directory exists
        if not project_path.exists():
            # Try to create it
            try:
                project_path.mkdir(parents=True, exist_ok=True)
                results.append(InputValidationResult(
                    path=project_dir,
                    valid=True,
                    is_warning=True,
                    message=f"Output directory created: {project_dir}",
                ))
            except Exception as e:
                results.append(InputValidationResult(
                    path=project_dir,
                    valid=False,
                    is_warning=False,
                    message=f"Cannot create output directory: {e}",
                ))
        else:
            # Check write permissions
            if not os.access(project_dir, os.W_OK):
                results.append(InputValidationResult(
                    path=project_dir,
                    valid=False,
                    is_warning=False,
                    message=f"Output directory is not writable: {project_dir}",
                ))
            else:
                results.append(InputValidationResult(
                    path=project_dir,
                    valid=True,
                    message=f"Output directory is writable",
                ))

        return results

    def check_optional_dependencies(self) -> List[InputValidationResult]:
        """Check for optional dependencies (ffmpeg, etc.).

        Logs warnings but does not fail the validation.

        Returns:
            List of InputValidationResult for each dependency check.
        """
        results: List[InputValidationResult] = []

        # Check for ffmpeg (needed for transcoding, audio extraction)
        ffmpeg_location = getattr(getattr(self.config, 'download', None), 'ffmpeg_location', '')
        if ffmpeg_location:
            # Custom ffmpeg location configured
            ffmpeg_path = Path(ffmpeg_location)
            if not ffmpeg_path.exists():
                results.append(InputValidationResult(
                    path=ffmpeg_location,
                    valid=False,
                    is_warning=True,
                    message=f"Configured ffmpeg_location not found: {ffmpeg_location}",
                ))
            elif not shutil.which('ffmpeg') and not ffmpeg_path.name == 'ffmpeg':
                # Also check if it's in PATH
                results.append(InputValidationResult(
                    path=ffmpeg_location,
                    valid=True,
                    is_warning=True,
                    message=f"ffmpeg at custom location: {ffmpeg_location}",
                ))
        else:
            # Check if ffmpeg is in PATH
            if not shutil.which('ffmpeg'):
                results.append(InputValidationResult(
                    path='ffmpeg',
                    valid=True,
                    is_warning=True,
                    message="ffmpeg not found in PATH - some features may be limited",
                ))

        # Check for yt-dlp (required for downloads)
        if not shutil.which('yt-dlp'):
            results.append(InputValidationResult(
                path='yt-dlp',
                valid=True,
                is_warning=True,
                message="yt-dlp not found in PATH - downloads will fail",
            ))

        return results

    def validate_stages(self) -> List[StageValidationResult]:
        """Validate stage ordering against the canonical STAGE_ORDER.

        Returns:
            List of StageValidationResult for any ordering issues.
        """
        results: List[StageValidationResult] = []
        stage_names = [s.name for s in self.stages]

        for name in stage_names:
            if name not in STAGE_ORDER:
                # Unknown stage names are allowed (entity stages, custom stages)
                # but log for awareness
                continue

        # Check ordering: stages that are in STAGE_ORDER should appear
        # in the same relative order
        known_stages = [n for n in stage_names if n in STAGE_ORDER]
        expected_order = [s for s in STAGE_ORDER if s in known_stages]

        if known_stages != expected_order:
            results.append(StageValidationResult(
                stage_name='STAGE_ORDER',
                status='error',
                message=(
                    f"Stage order mismatch: got {known_stages}, "
                    f"expected {expected_order}"
                ),
            ))

        return results

    def validate_checkpoint_compatibility(
        self,
        checkpoint_data: Optional[dict] = None,
    ) -> List[StageValidationResult]:
        """Validate checkpoint compatibility with current pipeline config.

        Args:
            checkpoint_data: Raw checkpoint data dict (from checkpoint.load()).

        Returns:
            List of StageValidationResult for compatibility issues.
        """
        results: List[StageValidationResult] = []

        if checkpoint_data is None:
            return results

        # Check version compatibility
        cp_version = checkpoint_data.get('version')
        if cp_version is not None and not isinstance(cp_version, (int, float)):
            results.append(StageValidationResult(
                stage_name='CHECKPOINT',
                status='error',
                message=f"Invalid checkpoint version type: {type(cp_version).__name__}",
            ))

        # Check last_completed_stage is valid
        last_stage = checkpoint_data.get('last_completed_stage')
        if last_stage and last_stage not in STAGE_ORDER:
            results.append(StageValidationResult(
                stage_name='CHECKPOINT',
                status='error',
                message=f"Checkpoint references unknown stage: {last_stage}",
            ))

        # Check config_hash compatibility
        cp_hash = checkpoint_data.get('config_hash', '')
        current_hash = getattr(self.config, '_config_hash', '')
        if cp_hash and current_hash and cp_hash != current_hash:
            results.append(StageValidationResult(
                stage_name='CHECKPOINT',
                status='error',
                message=(
                    f"Checkpoint config hash mismatch: checkpoint={cp_hash[:8]}... "
                    f"vs current={current_hash[:8]}... "
                    f"Config may have changed since last run."
                ),
            ))

        return results

    def validate_all(
        self,
        state=None,
        checkpoint_data: Optional[dict] = None,
    ) -> List[StageValidationResult]:
        """Run all validations and return combined results.

        Args:
            state: Optional PipelineState for per-stage input validation.
            checkpoint_data: Optional checkpoint data for compatibility check.

        Returns:
            List of StageValidationResult from all validation checks.
        """
        results: List[StageValidationResult] = []

        # 1. Config validation
        config_errors = self.validate_config()
        for err in config_errors:
            results.append(StageValidationResult(
                stage_name='CONFIG',
                status='error',
                message=err,
            ))

        # 2. Stage order validation
        results.extend(self.validate_stages())

        # 3. Checkpoint compatibility
        results.extend(self.validate_checkpoint_compatibility(checkpoint_data))

        # 4. Input file validation (voiceover, config)
        input_results = self.validate_input_files(state)
        for ir in input_results:
            results.append(StageValidationResult(
                stage_name='INPUT_FILES',
                status='error' if (not ir.valid and not ir.is_warning) else 'run',
                message=ir.message,
            ))

        # 5. Output directory validation
        output_results = self.validate_output_directory(state)
        for out_result in output_results:
            results.append(StageValidationResult(
                stage_name='OUTPUT_DIR',
                status='error' if not out_result.valid else 'run',
                message=out_result.message,
            ))

        # 6. Optional dependency checks (warnings only)
        dep_results = self.check_optional_dependencies()
        for dr in dep_results:
            if dr.is_warning:
                results.append(StageValidationResult(
                    stage_name='DEPS',
                    status='run',
                    message=dr.message,
                ))

        # 7. Per-stage input validation (if state provided)
        if state is not None:
            for stage in self.stages:
                validation_error = stage.validate_inputs(state, self.config)
                if validation_error:
                    results.append(StageValidationResult(
                        stage_name=stage.name,
                        status='error',
                        message=validation_error,
                    ))
                else:
                    results.append(StageValidationResult(
                        stage_name=stage.name,
                        status='run',
                        message='inputs valid',
                    ))

        return results

    def validate_stage_io(
        self,
        stage_name: str,
        checkpoint_data: Dict[str, Any],
        validation_type: str = 'both',
    ) -> List[ContractViolation]:
        """US-108-011: Validate stage input/output against registered schemas.

        Validates that checkpoint data matches the expected schema for a stage's
        input (data from previous stages) and output (data produced by this stage).

        Args:
            stage_name: Name of the stage to validate
            checkpoint_data: Raw checkpoint data dict
            validation_type: What to validate - 'input', 'output', or 'both' (default)

        Returns:
            List of ContractViolation objects for any schema mismatches
        """
        violations: List[ContractViolation] = []

        # Import here to avoid circular imports
        from .stages import get_stage_schemas

        schemas = get_stage_schemas(stage_name)

        # Validate input (data expected BEFORE stage runs)
        if validation_type in ('input', 'both') and 'input' in schemas:
            input_schema = schemas['input']
            violations.extend(self._validate_against_schema(
                stage_name=stage_name,
                schema=input_schema,
                data=checkpoint_data,
                violation_type='input',
            ))

        # Validate output (data expected AFTER stage runs)
        if validation_type in ('output', 'both') and 'output' in schemas:
            output_schema = schemas['output']
            violations.extend(self._validate_against_schema(
                stage_name=stage_name,
                schema=output_schema,
                data=checkpoint_data,
                violation_type='output',
            ))

        return violations

    def _validate_against_schema(
        self,
        stage_name: str,
        schema,
        data: Dict[str, Any],
        violation_type: str,
    ) -> List[ContractViolation]:
        """Validate checkpoint data against a schema.

        Args:
            stage_name: Name of the stage
            schema: StageInputSchema or StageOutputSchema
            data: Checkpoint data to validate
            violation_type: 'input' or 'output'

        Returns:
            List of ContractViolation objects
        """
        from .stages import StageInputSchema, StageOutputSchema

        violations: List[ContractViolation] = []

        for field_schema in schema.fields:
            field_name = field_schema.field_name
            value = data.get(field_name)

            # Check if field is required but missing
            if field_schema.required and value is None:
                violations.append(ContractViolation(
                    stage_name=stage_name,
                    violation_type=violation_type,
                    field_name=field_name,
                    expected=f"required field of type {field_schema.expected_type}",
                    actual="field not present in checkpoint",
                    severity="error",
                ))
                continue

            # If field is present, validate its type
            if value is not None:
                type_valid = self._check_field_type(value, field_schema.expected_type)
                if not type_valid:
                    violations.append(ContractViolation(
                        stage_name=stage_name,
                        violation_type=violation_type,
                        field_name=field_name,
                        expected=f"type {field_schema.expected_type}",
                        actual=f"type {type(value).__name__}",
                        severity="error",
                    ))

        return violations

    def _check_field_type(self, value: Any, expected_type: Any) -> bool:
        """Check if a value matches the expected type.

        Args:
            value: The value to check
            expected_type: Expected type (Type or string name)

        Returns:
            True if type matches, False otherwise
        """
        from .stages import StageInputSchema, StageOutputSchema

        actual_type = type(value)

        # Handle string type names
        if isinstance(expected_type, str):
            expected_name = expected_type.lower()
            actual_name = actual_type.__name__.lower()

            # Handle common type aliases
            type_aliases = {
                'dict': 'dict',
                'list': 'list',
                'str': 'str',
                'string': 'str',
                'int': 'int',
                'integer': 'int',
                'float': 'float',
                'bool': 'bool',
                'boolean': 'bool',
            }

            expected_name = type_aliases.get(expected_name, expected_name)

            # Handle dict-like types (PipelineState, etc.)
            if expected_name == 'dict':
                return actual_name in ('dict', 'dict')

            return actual_name == expected_name

        # Handle Type objects
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # expected_type might be a string that failed to resolve
            return False

    def validate_all_stages_io(
        self,
        checkpoint_data: Dict[str, Any],
    ) -> Dict[str, List[ContractViolation]]:
        """Validate I/O contracts for all stages in the pipeline.

        US-108-011: Validates input contracts for all stages that would run,
        and output contracts for all stages that have completed.

        Args:
            checkpoint_data: Raw checkpoint data dict

        Returns:
            Dict mapping stage_name to list of violations
        """
        results: Dict[str, List[ContractViolation]] = {}

        # Get the last completed stage
        last_completed = checkpoint_data.get('last_completed_stage')

        # Validate each stage
        for stage_name in STAGE_ORDER:
            # For stages before last_completed: validate both input and output
            # For stages at/after last_completed: validate input only
            if last_completed and STAGE_ORDER.index(stage_name) <= STAGE_ORDER.index(last_completed):
                validation_type = 'both'
            else:
                validation_type = 'input'

            violations = self.validate_stage_io(stage_name, checkpoint_data, validation_type)
            if violations:
                results[stage_name] = violations

        return results
