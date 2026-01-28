"""
Tests for scripts/run_integration_tests.py

Verifies the integration test runner's core functionality:
- Isolated environment setup and cleanup
- Test discovery
- Timeout handling
- Result parsing
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_integration_tests import (
    IsolatedTestEnvironment,
    IntegrationTestRunner,
    TestResult as IntegrationTestResult,
    RunSummary,
    DEFAULT_TIMEOUT,
    INTEGRATION_MARKERS,
)


class TestIsolatedTestEnvironment:
    """Tests for IsolatedTestEnvironment class."""

    def test_setup_creates_temp_directory(self):
        """Test that setup creates temp directory with subdirectories."""
        env = IsolatedTestEnvironment()
        temp_dir = env.setup()

        try:
            assert temp_dir.exists()
            assert (temp_dir / "cache").exists()
            assert (temp_dir / "output").exists()
            assert (temp_dir / "downloads").exists()
            assert (temp_dir / "projects").exists()
        finally:
            env.cleanup()

    def test_setup_sets_environment_variables(self):
        """Test that setup sets isolation environment variables."""
        env = IsolatedTestEnvironment()
        temp_dir = env.setup()

        try:
            assert os.environ.get("INTEGRATION_TEST_TEMP_DIR") == str(temp_dir)
            assert os.environ.get("INTEGRATION_TEST_CACHE_DIR") == str(temp_dir / "cache")
            assert os.environ.get("INTEGRATION_TEST_OUTPUT_DIR") == str(temp_dir / "output")
            assert os.environ.get("INTEGRATION_TEST_DOWNLOADS_DIR") == str(temp_dir / "downloads")
            assert os.environ.get("INTEGRATION_TEST_NO_CACHE") == "1"
        finally:
            env.cleanup()

    def test_cleanup_removes_temp_directory(self):
        """Test that cleanup removes temp directory when keep_artifacts=False."""
        env = IsolatedTestEnvironment(keep_artifacts=False)
        temp_dir = env.setup()
        temp_dir_path = str(temp_dir)

        env.cleanup()

        assert not Path(temp_dir_path).exists()

    def test_cleanup_preserves_temp_when_keep_artifacts(self):
        """Test that cleanup preserves temp directory when keep_artifacts=True."""
        env = IsolatedTestEnvironment(keep_artifacts=True)
        temp_dir = env.setup()

        try:
            env.cleanup()
            # Directory should still exist
            assert temp_dir.exists()
        finally:
            # Manual cleanup for test
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_cleanup_restores_environment(self):
        """Test that cleanup restores original environment variables."""
        # Set a marker variable
        original_value = os.environ.get("INTEGRATION_TEST_TEMP_DIR")

        env = IsolatedTestEnvironment()
        env.setup()
        env.cleanup()

        # Should be restored (None or original value)
        assert os.environ.get("INTEGRATION_TEST_TEMP_DIR") == original_value

    def test_context_manager_protocol(self):
        """Test that IsolatedTestEnvironment works as context manager."""
        with IsolatedTestEnvironment() as temp_dir:
            assert temp_dir.exists()
            assert os.environ.get("INTEGRATION_TEST_TEMP_DIR") == str(temp_dir)
            temp_dir_path = str(temp_dir)

        # After exiting, should be cleaned up
        assert not Path(temp_dir_path).exists()

    def test_custom_base_dir(self, tmp_path):
        """Test that custom base_dir is used for temp directory."""
        env = IsolatedTestEnvironment(base_dir=tmp_path)
        temp_dir = env.setup()

        try:
            assert temp_dir.parent == tmp_path
            assert "integration_test_" in temp_dir.name
        finally:
            env.cleanup()


class TestIntegrationTestRunner:
    """Tests for IntegrationTestRunner class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        runner = IntegrationTestRunner()

        assert runner.timeout == DEFAULT_TIMEOUT
        assert runner.parallel == 1
        assert runner.verbose is False
        assert runner.pattern is None
        assert runner.markers == INTEGRATION_MARKERS
        assert runner.keep_artifacts is False

    def test_custom_configuration(self):
        """Test custom configuration values."""
        runner = IntegrationTestRunner(
            timeout=600,
            parallel=4,
            verbose=True,
            pattern="test_caption*",
            markers=["integration"],
            keep_artifacts=True,
        )

        assert runner.timeout == 600
        assert runner.parallel == 4
        assert runner.verbose is True
        assert runner.pattern == "test_caption*"
        assert runner.markers == ["integration"]
        assert runner.keep_artifacts is True

    def test_discover_tests_returns_list(self):
        """Test that discover_tests returns a list."""
        runner = IntegrationTestRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="tests/test_example.py::test_one\ntests/test_example.py::test_two\n",
                stderr="",
                returncode=0,
            )

            tests = runner.discover_tests()

            assert isinstance(tests, list)
            assert len(tests) == 2
            assert "tests/test_example.py::test_one" in tests
            assert "tests/test_example.py::test_two" in tests

    def test_discover_tests_handles_timeout(self):
        """Test that discover_tests handles timeout gracefully."""
        runner = IntegrationTestRunner()

        import subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=[], timeout=60)

            tests = runner.discover_tests()

            assert tests == []

    def test_discover_tests_filters_summary_lines(self):
        """Test that discover_tests filters out pytest summary lines."""
        runner = IntegrationTestRunner()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="""tests/test_one.py::test_a
tests/test_two.py::test_b
===== 2 items collected =====
<Module tests/test_one.py>
""",
                stderr="",
                returncode=0,
            )

            tests = runner.discover_tests()

            assert len(tests) == 2
            assert "====" not in str(tests)
            assert "<Module" not in str(tests)

    def test_run_single_test_success(self, tmp_path):
        """Test running a single test that passes."""
        runner = IntegrationTestRunner(timeout=10)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="PASSED",
                stderr="",
                returncode=0,
            )

            result = runner.run_single_test("tests/test_example.py::test_pass", tmp_path)

            assert result.status == "passed"
            assert result.duration >= 0  # Can be very fast when mocked
            assert result.name == "tests/test_example.py::test_pass"

    def test_run_single_test_failure(self, tmp_path):
        """Test running a single test that fails."""
        runner = IntegrationTestRunner(timeout=10)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="FAILED",
                stderr="AssertionError",
                returncode=1,
            )

            result = runner.run_single_test("tests/test_example.py::test_fail", tmp_path)

            assert result.status == "failed"

    def test_run_single_test_timeout(self, tmp_path):
        """Test running a single test that times out."""
        runner = IntegrationTestRunner(timeout=1)

        import subprocess
        with patch("subprocess.run") as mock_run:
            exc = subprocess.TimeoutExpired(cmd=[], timeout=1)
            exc.stdout = b"partial output"  # Set after construction
            mock_run.side_effect = exc

            result = runner.run_single_test("tests/test_example.py::test_slow", tmp_path)

            assert result.status == "timeout"
            assert "timed out" in result.error.lower()

    def test_run_single_test_skipped(self, tmp_path):
        """Test running a single test that is skipped."""
        runner = IntegrationTestRunner(timeout=10)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="SKIPPED",
                stderr="",
                returncode=5,  # pytest exit code for no tests
            )

            result = runner.run_single_test("tests/test_example.py::test_skip", tmp_path)

            assert result.status == "skipped"


class TestIntegrationTestResult:
    """Tests for IntegrationTestResult dataclass."""

    def test_create_passed_result(self):
        """Test creating a passed test result."""
        result = IntegrationTestResult(
            name="test_example",
            status="passed",
            duration=1.5,
            output="test output",
        )

        assert result.name == "test_example"
        assert result.status == "passed"
        assert result.duration == 1.5
        assert result.output == "test output"
        assert result.error == ""

    def test_create_failed_result(self):
        """Test creating a failed test result."""
        result = IntegrationTestResult(
            name="test_fail",
            status="failed",
            duration=2.0,
            error="AssertionError: expected 1 got 2",
        )

        assert result.status == "failed"
        assert "AssertionError" in result.error


class TestRunSummary:
    """Tests for RunSummary dataclass."""

    def test_create_empty_summary(self):
        """Test creating an empty summary."""
        summary = RunSummary()

        assert summary.total == 0
        assert summary.passed == 0
        assert summary.failed == 0
        assert summary.results == []

    def test_create_populated_summary(self):
        """Test creating a populated summary."""
        results = [
            IntegrationTestResult("test1", "passed", 1.0),
            IntegrationTestResult("test2", "failed", 2.0),
            IntegrationTestResult("test3", "passed", 1.5),
        ]

        summary = RunSummary(
            total=3,
            passed=2,
            failed=1,
            duration=4.5,
            results=results,
        )

        assert summary.total == 3
        assert summary.passed == 2
        assert summary.failed == 1
        assert len(summary.results) == 3


class TestParallelExecution:
    """Tests for parallel execution support."""

    def test_parallel_disabled_runs_sequential(self, tmp_path):
        """Test that parallel=1 runs tests sequentially."""
        runner = IntegrationTestRunner(parallel=1)

        tests = ["test1", "test2"]

        with patch.object(runner, "run_single_test") as mock_run:
            mock_run.return_value = IntegrationTestResult("test", "passed", 1.0)

            results = runner._run_sequential(tests, tmp_path)

            assert mock_run.call_count == 2
            assert len(results) == 2

    def test_parallel_fallback_on_xdist_missing(self, tmp_path):
        """Test fallback to sequential when pytest-xdist is missing."""
        runner = IntegrationTestRunner(parallel=4)

        tests = ["test1", "test2"]

        import subprocess
        with patch("subprocess.run") as mock_run:
            # First call for parallel fails with xdist error
            mock_run.side_effect = [
                Exception("pytest-xdist not installed"),
                MagicMock(stdout="PASSED", returncode=0),
                MagicMock(stdout="PASSED", returncode=0),
            ]

            with patch.object(runner, "_run_sequential") as mock_seq:
                mock_seq.return_value = [
                    IntegrationTestResult("test1", "passed", 1.0),
                    IntegrationTestResult("test2", "passed", 1.0),
                ]

                results = runner.run_parallel(tests, tmp_path)

                # Should have fallen back to sequential
                mock_seq.assert_called_once()
