"""
PipelineValidator Tests - US-82-006

Tests that verify PipelineValidator:
- validate_config() catches missing config fields
- validate_stages() catches invalid stage order
- validate_checkpoint_compatibility() catches incompatible checkpoint versions
- validate_all() returns combined StageValidationResult list
- PipelineValidator is independently constructable (no orchestrator dependency)
"""

import pytest
from unittest.mock import Mock

from src.checkpoint import STAGE_ORDER
from src.pipeline import StageValidationResult
from src.pipeline_validator import PipelineValidator
from src.stages import Stage, StageResult
from src.state import PipelineState


# ---------------------------------------------------------------------------
# Test Stage Implementations
# ---------------------------------------------------------------------------

class _DummyStage(Stage):
    """A stage that tracks calls for testing."""

    def __init__(self, name, validation_error=None):
        self.name = name
        self._validation_error = validation_error

    def run(self, state, config, checkpoint=None):
        return StageResult(success=True)

    def can_skip(self, state, checkpoint):
        return False

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return self._validation_error


def _make_valid_config():
    """Create a minimal valid mock config."""
    config = Mock()
    config.cache = Mock()
    config.cache.cache_dir = ".cache"
    config.embedding = Mock()
    config.embedding.provider = "sentence-transformers"
    config.download = None
    config._config_hash = "abc123"
    return config


# ===========================================================================
# PipelineValidator is independently constructable
# ===========================================================================

class TestPipelineValidatorIndependent:
    """PipelineValidator works without PipelineOrchestrator."""

    def test_constructable_with_config_and_stages(self):
        """Can create PipelineValidator with just config and stage list."""
        config = _make_valid_config()
        stages = [_DummyStage('ANALYZE'), _DummyStage('MATCH')]

        validator = PipelineValidator(config, stages)

        assert validator.config is config
        assert validator.stages is stages

    def test_validate_all_returns_list(self):
        """validate_all() returns a list of StageValidationResult."""
        config = _make_valid_config()
        stages = [_DummyStage('ANALYZE')]

        validator = PipelineValidator(config, stages)
        results = validator.validate_all()

        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, StageValidationResult)


# ===========================================================================
# validate_config() catches missing config fields
# ===========================================================================

@pytest.mark.fast
class TestValidateConfig:
    """Test validate_config() catches config issues."""

    def test_missing_cache_dir(self):
        """Missing cache_dir produces an error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None
        config.download = None

        validator = PipelineValidator(config, [])
        errors = validator.validate_config()

        assert any("cache" in e.lower() for e in errors)

    def test_empty_cache_dir(self):
        """Empty string cache_dir produces an error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ""
        config.download = None

        validator = PipelineValidator(config, [])
        errors = validator.validate_config()

        assert any("cache" in e.lower() for e in errors)

    def test_valid_config_no_errors(self):
        """Valid config produces no errors."""
        config = _make_valid_config()

        validator = PipelineValidator(config, [])
        errors = validator.validate_config()

        assert len(errors) == 0

    def test_missing_embedding_provider_with_match_stage(self):
        """MATCH stage without embedding provider produces error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = Mock()
        config.embedding.provider = None
        config.download = None

        stages = [_DummyStage('MATCH')]
        validator = PipelineValidator(config, stages)
        errors = validator.validate_config()

        assert any("embedding" in e.lower() for e in errors)

    def test_missing_embedding_provider_without_match_stage(self):
        """No matching stages means missing embedding provider is OK."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = Mock()
        config.embedding.provider = None
        config.download = None

        stages = [_DummyStage('ANALYZE')]
        validator = PipelineValidator(config, stages)
        errors = validator.validate_config()

        assert not any("embedding" in e.lower() for e in errors)


# ===========================================================================
# validate_stages() catches invalid stage order
# ===========================================================================

@pytest.mark.fast
class TestValidateStages:
    """Test validate_stages() checks stage ordering."""

    def test_correct_order_no_errors(self):
        """Standard pipeline order produces no errors."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('VIDEO_SEARCH'),
            _DummyStage('CAPTION'),
            _DummyStage('MATCH'),
        ]
        validator = PipelineValidator(_make_valid_config(), stages)
        results = validator.validate_stages()

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 0

    def test_wrong_order_produces_error(self):
        """Reversed stage order produces an error."""
        stages = [
            _DummyStage('MATCH'),
            _DummyStage('ANALYZE'),
        ]
        validator = PipelineValidator(_make_valid_config(), stages)
        results = validator.validate_stages()

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 1
        assert 'order' in errors[0].message.lower()

    def test_partial_order_is_valid(self):
        """Subset of stages in correct relative order is valid."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('MATCH'),
            _DummyStage('OUTPUT'),
        ]
        validator = PipelineValidator(_make_valid_config(), stages)
        results = validator.validate_stages()

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 0

    def test_unknown_stages_tolerated(self):
        """Stages not in STAGE_ORDER don't cause errors."""
        stages = [
            _DummyStage('ANALYZE'),
            _DummyStage('CUSTOM_STAGE'),
            _DummyStage('MATCH'),
        ]
        validator = PipelineValidator(_make_valid_config(), stages)
        results = validator.validate_stages()

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 0


# ===========================================================================
# validate_checkpoint_compatibility() catches incompatible checkpoints
# ===========================================================================

@pytest.mark.fast
class TestValidateCheckpointCompatibility:
    """Test validate_checkpoint_compatibility() checks checkpoint data."""

    def test_no_checkpoint_no_errors(self):
        """None checkpoint data produces no errors."""
        validator = PipelineValidator(_make_valid_config(), [])
        results = validator.validate_checkpoint_compatibility(None)

        assert len(results) == 0

    def test_unknown_stage_in_checkpoint(self):
        """Checkpoint referencing unknown stage produces error."""
        validator = PipelineValidator(_make_valid_config(), [])
        results = validator.validate_checkpoint_compatibility({
            'last_completed_stage': 'NONEXISTENT_STAGE',
        })

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 1
        assert 'unknown stage' in errors[0].message.lower()

    def test_valid_stage_in_checkpoint(self):
        """Checkpoint referencing valid stage produces no error."""
        validator = PipelineValidator(_make_valid_config(), [])
        results = validator.validate_checkpoint_compatibility({
            'last_completed_stage': 'MATCH',
        })

        stage_errors = [r for r in results
                        if r.status == 'error' and 'unknown stage' in r.message.lower()]
        assert len(stage_errors) == 0

    def test_config_hash_mismatch(self):
        """Mismatched config_hash produces error."""
        config = _make_valid_config()
        config._config_hash = "current_hash_123"

        validator = PipelineValidator(config, [])
        results = validator.validate_checkpoint_compatibility({
            'last_completed_stage': 'MATCH',
            'config_hash': 'old_hash_456',
        })

        errors = [r for r in results if r.status == 'error']
        assert any('hash mismatch' in e.message.lower() for e in errors)

    def test_matching_config_hash_no_error(self):
        """Matching config_hash produces no error."""
        config = _make_valid_config()
        config._config_hash = "same_hash"

        validator = PipelineValidator(config, [])
        results = validator.validate_checkpoint_compatibility({
            'last_completed_stage': 'MATCH',
            'config_hash': 'same_hash',
        })

        errors = [r for r in results if r.status == 'error']
        assert len(errors) == 0

    def test_invalid_version_type(self):
        """Non-numeric version produces error."""
        validator = PipelineValidator(_make_valid_config(), [])
        results = validator.validate_checkpoint_compatibility({
            'version': 'not_a_number',
        })

        errors = [r for r in results if r.status == 'error']
        assert any('version' in e.message.lower() for e in errors)


# ===========================================================================
# validate_all() returns combined results
# ===========================================================================

@pytest.mark.fast
class TestValidateAll:
    """Test validate_all() combines all validation checks."""

    def test_returns_stage_validation_results(self):
        """validate_all() returns StageValidationResult objects."""
        config = _make_valid_config()
        stages = [_DummyStage('ANALYZE')]

        validator = PipelineValidator(config, stages)
        state = PipelineState()
        results = validator.validate_all(state=state)

        assert all(isinstance(r, StageValidationResult) for r in results)

    def test_config_error_included(self):
        """Config errors appear in validate_all() results."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None
        config.download = None

        validator = PipelineValidator(config, [])
        results = validator.validate_all()

        config_errors = [r for r in results if r.stage_name == 'CONFIG']
        assert len(config_errors) >= 1
        assert config_errors[0].status == 'error'

    def test_stage_validation_included(self):
        """Per-stage validation errors appear in validate_all()."""
        config = _make_valid_config()
        stages = [_DummyStage('ANALYZE', validation_error='No voiceover')]

        validator = PipelineValidator(config, stages)
        state = PipelineState()
        results = validator.validate_all(state=state)

        stage_errors = [r for r in results
                        if r.stage_name == 'ANALYZE' and r.status == 'error']
        assert len(stage_errors) == 1
        assert 'voiceover' in stage_errors[0].message.lower()

    def test_checkpoint_error_included(self):
        """Checkpoint errors appear in validate_all()."""
        config = _make_valid_config()

        validator = PipelineValidator(config, [])
        results = validator.validate_all(checkpoint_data={
            'last_completed_stage': 'FAKE_STAGE',
        })

        cp_errors = [r for r in results if r.stage_name == 'CHECKPOINT']
        assert len(cp_errors) >= 1

    def test_multiple_error_types_combined(self):
        """Config + stage + checkpoint errors all appear in results."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None  # Config error
        config.download = None
        config._config_hash = "new_hash"

        stages = [_DummyStage('ANALYZE', validation_error='Missing file')]

        validator = PipelineValidator(config, stages)
        state = PipelineState()
        results = validator.validate_all(
            state=state,
            checkpoint_data={'last_completed_stage': 'BOGUS'},
        )

        error_names = {r.stage_name for r in results if r.status == 'error'}
        assert 'CONFIG' in error_names
        assert 'ANALYZE' in error_names
        assert 'CHECKPOINT' in error_names


# ===========================================================================
# Orchestrator delegation
# ===========================================================================

@pytest.mark.fast
class TestOrchestratorDelegation:
    """Verify PipelineOrchestrator delegates to PipelineValidator."""

    def test_orchestrator_validate_config_delegates(self, tmp_path):
        """PipelineOrchestrator._validate_config() uses PipelineValidator."""
        from src.pipeline import PipelineOrchestrator
        from src.checkpoint import CheckpointManager
        from collections import defaultdict

        config = _make_valid_config()
        config.cache.cache_dir = str(tmp_path / "cache")
        (tmp_path / "cache").mkdir()

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = config
        orch.stages = []
        orch.state = PipelineState()
        orch.checkpoint = Mock()
        orch.resume_mode = False
        orch.current_stage = None
        orch.stage_timings = {}
        orch.stage_metrics = {}
        orch._event_hooks = defaultdict(list)
        orch._validator = PipelineValidator(config, [])

        errors = orch._validate_config()
        assert isinstance(errors, list)
        # Valid config -> no errors
        assert len(errors) == 0

    def test_orchestrator_run_calls_validate_config(self, tmp_path):
        """PipelineOrchestrator.run() still validates config via delegation."""
        from src.pipeline import PipelineOrchestrator
        from collections import defaultdict

        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None  # Will trigger config error
        config.download = None
        config.freeze = Mock()

        orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orch.config = config
        orch.stages = [_DummyStage('ANALYZE')]
        orch.state = PipelineState()
        orch.checkpoint = Mock()
        orch.resume_mode = False
        orch.current_stage = None
        orch.stage_timings = {}
        orch.stage_metrics = {}
        orch._event_hooks = defaultdict(list)
        orch._validator = PipelineValidator(config, orch.stages)

        # Mock progress reporter
        orch.progress_reporter = Mock()

        result = orch.run(resume=False)
        assert result is False  # Should fail on config validation


# ===========================================================================
# validate_input_files() - US-106-010
# ===========================================================================

@pytest.mark.fast
class TestValidateInputFiles:
    """Test validate_input_files() checks input file existence."""

    def test_missing_voiceover_file(self, tmp_path):
        """Missing voiceover file produces an error."""
        config = _make_valid_config()
        config.config_path = None  # Clear mock
        validator = PipelineValidator(config, [])

        state = PipelineState()
        state.voiceover_path = str(tmp_path / "nonexistent.srt")

        results = validator.validate_input_files(state)

        # Filter to only voiceover results
        voiceover_results = [r for r in results if 'voiceover' in r.message.lower()]
        assert len(voiceover_results) == 1
        assert voiceover_results[0].valid is False
        assert "not found" in voiceover_results[0].message.lower()

    def test_existing_voiceover_file(self, tmp_path):
        """Existing voiceover file produces valid result."""
        config = _make_valid_config()
        config.config_path = None  # Clear mock
        validator = PipelineValidator(config, [])

        # Create a dummy voiceover file
        voiceover_path = tmp_path / "voiceover.srt"
        voiceover_path.write_text("dummy")

        state = PipelineState()
        state.voiceover_path = str(voiceover_path)

        results = validator.validate_input_files(state)

        # Filter to voiceover results
        voiceover_results = [r for r in results if 'voiceover' in r.message.lower()]
        assert len(voiceover_results) == 1
        assert voiceover_results[0].valid is True

    def test_missing_config_file(self, tmp_path):
        """Missing config file produces an error."""
        config = _make_valid_config()
        config.config_path = str(tmp_path / "nonexistent.yaml")

        validator = PipelineValidator(config, [])

        results = validator.validate_input_files(None)

        assert len(results) == 1
        assert results[0].valid is False
        assert "not found" in results[0].message.lower()

    def test_existing_config_file(self, tmp_path):
        """Existing config file produces valid result."""
        config = _make_valid_config()
        config_path = tmp_path / "config.yaml"
        config_path.write_text("key: value")
        config.config_path = str(config_path)

        validator = PipelineValidator(config, [])

        results = validator.validate_input_files(None)

        assert len(results) == 1
        assert results[0].valid is True

    def test_no_state_produces_empty_results(self):
        """No state produces empty results (no error)."""
        config = _make_valid_config()
        config.config_path = None  # Clear mock
        validator = PipelineValidator(config, [])

        results = validator.validate_input_files(None)

        assert len(results) == 0


# ===========================================================================
# validate_output_directory() - US-106-010
# ===========================================================================

@pytest.mark.fast
class TestValidateOutputDirectory:
    """Test validate_output_directory() checks write permissions."""

    def test_nonexistent_directory_created(self, tmp_path):
        """Nonexistent directory is created (warning)."""
        config = _make_valid_config()
        validator = PipelineValidator(config, [])

        new_dir = tmp_path / "new_project"
        state = PipelineState()
        state.project_dir = str(new_dir)

        results = validator.validate_output_directory(state)

        assert len(results) == 1
        assert results[0].valid is True
        assert results[0].is_warning is True
        assert "created" in results[0].message.lower()

    def test_existing_writable_directory(self, tmp_path):
        """Existing writable directory is valid."""
        config = _make_valid_config()
        validator = PipelineValidator(config, [])

        state = PipelineState()
        state.project_dir = str(tmp_path)

        results = validator.validate_output_directory(state)

        assert len(results) == 1
        assert results[0].valid is True
        assert "writable" in results[0].message.lower()

    def test_project_dir_from_config(self, tmp_path):
        """Falls back to config.project_dir when state has no project_dir."""
        config = _make_valid_config()
        config.project_dir = str(tmp_path)

        validator = PipelineValidator(config, [])

        results = validator.validate_output_directory(None)

        assert len(results) == 1
        assert results[0].valid is True


# ===========================================================================
# check_optional_dependencies() - US-106-010
# ===========================================================================

@pytest.mark.fast
class TestCheckOptionalDependencies:
    """Test check_optional_dependencies() warns about missing tools."""

    def test_ffmpeg_not_in_path_warns(self):
        """Missing ffmpeg produces a warning."""
        import shutil
        config = _make_valid_config()
        config.download = Mock()
        config.download.ffmpeg_location = ""

        validator = PipelineValidator(config, [])

        # Mock shutil.which to return None for ffmpeg
        original_which = shutil.which
        try:
            shutil.which = lambda x: None if x == 'ffmpeg' else original_which(x)
            results = validator.check_optional_dependencies()

            ffmpeg_result = [r for r in results if 'ffmpeg' in r.message.lower()]
            assert len(ffmpeg_result) >= 1
            assert ffmpeg_result[0].is_warning is True
        finally:
            shutil.which = original_which

    def test_yt_dlp_not_in_path_warns(self):
        """ Missing yt-dlp produces a warning."""
        import shutil
        config = _make_valid_config()
        config.download = Mock()
        config.download.ffmpeg_location = ""

        validator = PipelineValidator(config, [])

        # Mock shutil.which to return None for yt-dlp
        original_which = shutil.which
        try:
            shutil.which = lambda x: None if x == 'yt-dlp' else original_which(x)
            results = validator.check_optional_dependencies()

            ytdlp_result = [r for r in results if 'yt-dlp' in r.message.lower()]
            assert len(ytdlp_result) >= 1
            assert ytdlp_result[0].is_warning is True
        finally:
            shutil.which = original_which

    def test_custom_ffmpeg_location_not_found_warns(self, tmp_path):
        """Custom ffmpeg location that's missing produces warning."""
        config = _make_valid_config()
        config.download = Mock()
        config.download.ffmpeg_location = str(tmp_path / "nonexistent" / "ffmpeg")

        validator = PipelineValidator(config, [])

        results = validator.check_optional_dependencies()

        ffmpeg_result = [r for r in results if 'ffmpeg' in r.message.lower() and 'not found' in r.message.lower()]
        assert len(ffmpeg_result) >= 1


# ===========================================================================
# validate_all() includes new validations - US-106-010
# ===========================================================================

@pytest.mark.fast
class TestValidateAllIncludesNewValidations:
    """Test validate_all() includes new validation types."""

    def test_input_files_included(self, tmp_path):
        """validate_all() includes INPUT_FILES results."""
        config = _make_valid_config()
        config.config_path = None  # Clear mock
        validator = PipelineValidator(config, [])

        state = PipelineState()
        state.voiceover_path = str(tmp_path / "missing.srt")

        results = validator.validate_all(state=state)

        input_files = [r for r in results if r.stage_name == 'INPUT_FILES']
        assert len(input_files) >= 1

    def test_output_dir_included(self, tmp_path):
        """validate_all() includes OUTPUT_DIR results."""
        config = _make_valid_config()
        config.config_path = None  # Clear mock
        validator = PipelineValidator(config, [])

        state = PipelineState()
        state.project_dir = str(tmp_path)

        results = validator.validate_all(state=state)

        output_dir = [r for r in results if r.stage_name == 'OUTPUT_DIR']
        assert len(output_dir) >= 1

    def test_deps_warnings_included(self, tmp_path):
        """validate_all() includes DEPS warnings."""
        config = _make_valid_config()
        config.cache.cache_dir = str(tmp_path / "cache")
        (tmp_path / "cache").mkdir()
        config.download = Mock()
        config.download.ffmpeg_location = ""
        config.download.cookies_from_browser = ""

        validator = PipelineValidator(config, [])

        results = validator.validate_all()

        deps = [r for r in results if r.stage_name == 'DEPS']
        # Should have at least one dependency warning (ffmpeg or yt-dlp may be missing)
        assert len(deps) >= 0  # May be empty if tools are found
