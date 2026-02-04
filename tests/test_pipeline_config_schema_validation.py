"""
Tests for US-57-012: Pipeline config validation split into schema and runtime checks.

Verifies that _validate_config_schema() performs pure config checks without any
filesystem access, and _validate_runtime_environment() handles I/O-dependent checks.
"""

import pytest
from unittest.mock import Mock, patch

from src.pipeline import PipelineOrchestrator
from src.state import PipelineState
from tests.fixtures import create_mock_config


def _make_orchestrator_no_io(config, stages=None):
    """Create PipelineOrchestrator without __init__ (bypasses all I/O)."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.config = config
    orch.stages = stages or []
    orch.state = PipelineState()
    orch.checkpoint = Mock()
    orch.resume_mode = False
    orch.current_stage = None
    orch.stage_timings = {}
    orch.stage_metrics = {}
    orch.healing_enabled = False
    return orch


# =============================================================================
# _validate_config_schema() — pure checks, no I/O
# =============================================================================

@pytest.mark.fast
class TestValidateConfigSchema:
    """Test _validate_config_schema() returns errors for invalid configs without filesystem."""

    def test_missing_cache_dir_returns_error(self):
        """Config with no cache_dir produces a schema error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None

        orch = _make_orchestrator_no_io(config)
        errors = orch._validate_config_schema()

        assert any("cache" in e.lower() and "directory" in e.lower() for e in errors)

    def test_empty_cache_dir_returns_error(self):
        """Config with empty string cache_dir produces a schema error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ""

        orch = _make_orchestrator_no_io(config)
        errors = orch._validate_config_schema()

        assert any("cache" in e.lower() for e in errors)

    def test_valid_cache_dir_no_error(self):
        """Config with a cache_dir value produces no cache-related schema error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = Mock()
        config.embedding.provider = "gemini"

        orch = _make_orchestrator_no_io(config)
        errors = orch._validate_config_schema()

        assert len(errors) == 0

    def test_missing_embedding_provider_with_match_stage(self):
        """Matching stages without embedding provider produces a schema error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = Mock()
        config.embedding.provider = None

        match_stage = Mock()
        match_stage.name = "MATCH"

        orch = _make_orchestrator_no_io(config, stages=[match_stage])
        errors = orch._validate_config_schema()

        assert any("embedding" in e.lower() and "provider" in e.lower() for e in errors)

    def test_missing_embedding_provider_without_match_stage(self):
        """No matching stages means missing embedding provider is not an error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = Mock()
        config.embedding.provider = None

        analyze_stage = Mock()
        analyze_stage.name = "ANALYZE"

        orch = _make_orchestrator_no_io(config, stages=[analyze_stage])
        errors = orch._validate_config_schema()

        assert len(errors) == 0

    def test_no_filesystem_access(self):
        """_validate_config_schema() must not call os.access or Path.exists."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = "/nonexistent/path"
        config.embedding = Mock()
        config.embedding.provider = "gemini"

        orch = _make_orchestrator_no_io(config)

        with patch("src.pipeline.os.access") as mock_access, \
             patch("src.pipeline.Path.exists") as mock_exists:
            errors = orch._validate_config_schema()

            mock_access.assert_not_called()
            mock_exists.assert_not_called()

        assert len(errors) == 0

    def test_iterative_match_stage_also_requires_embedding(self):
        """ITERATIVE_MATCH stage also triggers embedding provider check."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = None

        iter_stage = Mock()
        iter_stage.name = "ITERATIVE_MATCH"

        orch = _make_orchestrator_no_io(config, stages=[iter_stage])
        errors = orch._validate_config_schema()

        assert any("embedding" in e.lower() for e in errors)


# =============================================================================
# _validate_runtime_environment() — I/O-dependent checks
# =============================================================================

@pytest.mark.fast
class TestValidateRuntimeEnvironment:
    """Test _validate_runtime_environment() handles filesystem checks."""

    def test_unwritable_cache_dir_returns_error(self, tmp_path):
        """Existing but unwritable cache_dir produces a runtime error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = str(tmp_path)
        config.download = None

        orch = _make_orchestrator_no_io(config)

        with patch("src.pipeline.os.access", return_value=False):
            errors = orch._validate_runtime_environment()

        assert any("not writable" in e for e in errors)

    def test_writable_cache_dir_no_error(self, tmp_path):
        """Writable cache_dir produces no runtime error."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = str(cache_dir)
        config.download = None

        orch = _make_orchestrator_no_io(config)
        errors = orch._validate_runtime_environment()

        assert len(errors) == 0

    def test_no_cache_dir_no_runtime_error(self):
        """Missing cache_dir is a schema error, not a runtime error."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None
        config.download = None

        orch = _make_orchestrator_no_io(config)
        errors = orch._validate_runtime_environment()

        # Runtime check should not duplicate the schema check
        assert len(errors) == 0


# =============================================================================
# _validate_config() — combined
# =============================================================================

@pytest.mark.fast
class TestValidateConfigCombined:
    """Test _validate_config() combines schema + runtime errors."""

    def test_combines_both_error_sources(self):
        """_validate_config() returns errors from both schema and runtime checks."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = None  # schema error: no cache dir
        config.download = None

        orch = _make_orchestrator_no_io(config)
        errors = orch._validate_config()

        # Should have at least the schema error for missing cache_dir
        assert any("cache" in e.lower() for e in errors)

    def test_returns_list_of_strings(self):
        """Both methods return List[str]."""
        config = Mock()
        config.cache = Mock()
        config.cache.cache_dir = ".cache"
        config.embedding = Mock()
        config.embedding.provider = "gemini"
        config.download = None

        orch = _make_orchestrator_no_io(config)

        schema_errors = orch._validate_config_schema()
        runtime_errors = orch._validate_runtime_environment()
        combined = orch._validate_config()

        assert isinstance(schema_errors, list)
        assert isinstance(runtime_errors, list)
        assert isinstance(combined, list)
        assert all(isinstance(e, str) for e in combined)
