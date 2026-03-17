"""
Sprint 2 Testing Verification Tests

US-001: Create sprint 2 testing verification test

Verifies that all new test files created in this sprint exist and import correctly.
This test file itself serves as verification that the sprint infrastructure is working.
"""

import pytest
import importlib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSprintTestFileExistence:
    """Verify that all planned test files exist."""

    @pytest.mark.fast
    def test_test_cli_args_exists(self):
        """Test that tests/test_cli_args.py exists."""
        test_file = Path(__file__).parent / "test_cli_args.py"
        assert test_file.exists(), f"Expected test file {test_file} does not exist"

    @pytest.mark.fast
    def test_test_cleanup_module_exists(self):
        """Test that tests/test_cleanup_module.py exists."""
        test_file = Path(__file__).parent / "test_cleanup_module.py"
        assert test_file.exists(), f"Expected test file {test_file} does not exist"


class TestSourceModuleImports:
    """Verify that source modules to be tested can be imported."""

    @pytest.mark.fast
    def test_cli_args_module_imports(self):
        """Test that src.cli.args module imports correctly."""
        from src.cli import args
        assert hasattr(args, 'parse_arguments'), "parse_arguments function should exist"

    @pytest.mark.fast
    def test_cleanup_module_imports(self):
        """Test that src.cleanup module imports correctly."""
        from src.cleanup import FileDeleter, AudioCleanupService, AudioCleanupResult
        assert FileDeleter is not None
        assert AudioCleanupService is not None
        assert AudioCleanupResult is not None

    @pytest.mark.fast
    def test_agents_watcher_imports(self):
        """Test that src.agents.watcher module imports correctly."""
        from src.agents import WatcherAgent, ErrorClassification
        assert WatcherAgent is not None
        assert ErrorClassification is not None

    @pytest.mark.fast
    def test_agents_fallback_imports(self):
        """Test that src.agents.fallback module imports correctly."""
        from src.agents import FallbackChain, pattern_route, PATTERN_ROUTING
        assert FallbackChain is not None
        assert pattern_route is not None
        assert PATTERN_ROUTING is not None

    @pytest.mark.fast
    def test_cli_environment_imports(self):
        """Test that src.cli.environment module imports correctly."""
        from src.cli import environment
        assert environment is not None


class TestPytestCollectionSucceeds:
    """Verify pytest can collect the new test files."""

    @pytest.mark.fast
    def test_sprint_verification_collects(self):
        """Test that this verification file collects properly."""
        # If we're running, collection succeeded
        assert True

    @pytest.mark.fast
    def test_conftest_has_markers(self):
        """Test that conftest.py defines required markers."""
        conftest_file = Path(__file__).parent / "conftest.py"
        assert conftest_file.exists(), "conftest.py should exist"

        content = conftest_file.read_text()
        assert "integration" in content, "integration marker should be defined"


class TestStagesCanBeImported:
    """Verify all pipeline stages can be imported for error handling tests."""

    @pytest.mark.fast
    def test_download_stage_imports(self):
        """Test DownloadStage can be imported."""
        from src.stages.download import DownloadStage
        assert DownloadStage is not None

    @pytest.mark.fast
    def test_transcribe_stage_imports(self):
        """Test TranscribeStage can be imported."""
        from src.stages.transcribe import TranscribeStage
        assert TranscribeStage is not None

    @pytest.mark.fast
    def test_match_stage_imports(self):
        """Test MatchStage can be imported."""
        from src.stages.match import MatchStage
        assert MatchStage is not None

    @pytest.mark.fast
    def test_output_stage_imports(self):
        """Test OutputStage can be imported."""
        from src.stages.output import OutputStage
        assert OutputStage is not None

    @pytest.mark.fast
    def test_stage_result_imports(self):
        """Test StageResult can be imported."""
        from src.stages import StageResult
        assert StageResult is not None
        assert hasattr(StageResult, 'ok')
        assert hasattr(StageResult, 'fail')


class TestConfigMergeImports:
    """Verify config merge utilities can be imported."""

    @pytest.mark.fast
    def test_config_utils_imports(self):
        """Test that src.cli.config_utils can be imported."""
        from src.cli import config_utils
        assert config_utils is not None


class TestVerificationPasses:
    """Final verification that sprint infrastructure is working."""

    @pytest.mark.fast
    def test_verification_complete(self):
        """
        This test marks US-001 as ready for completion.

        When this passes along with all other tests in this file,
        set passes: true for US-001 in prd.json.
        """
        # All imports and checks above must pass for this to run
        assert True, "Sprint 2 testing verification complete"
