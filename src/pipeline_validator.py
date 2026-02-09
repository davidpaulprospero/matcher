"""
Pipeline Validator - US-82-006

Extracted pre-run validation logic from PipelineOrchestrator.
Independently constructable with just config and stage list (no orchestrator dependency).

Validates config, stages, and checkpoint compatibility before pipeline execution.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .checkpoint import STAGE_ORDER

if TYPE_CHECKING:
    from .config import Config
    from .stages import Stage

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

        # 4. Per-stage input validation (if state provided)
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
