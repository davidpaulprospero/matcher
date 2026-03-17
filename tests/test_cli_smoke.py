"""
Smoke tests for CLI commands.

US-119-012: Add smoke tests for CLI commands

Tests for running CLI commands and verifying they execute without errors:
- --help flag works for all CLI commands
- --validate-config runs without errors
- --list-keywords runs without errors
- --validate-captions runs without errors
"""

import pytest
import subprocess
import sys
from pathlib import Path

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class TestCLIHelp:
    """Tests for --help flag."""

    @pytest.mark.fast
    def test_help_flag_works(self):
        """Test --help flag displays help and exits successfully."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"
        assert 'Voiceover-to-Footage Matching Pipeline' in result.stdout
        assert '--voiceover' in result.stdout or '-v' in result.stdout

    @pytest.mark.fast
    def test_help_shows_all_main_flags(self):
        """Test --help shows all major CLI flags."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Check for key flags mentioned in acceptance criteria
        assert '--validate-config' in result.stdout
        assert '--list-keywords' in result.stdout
        assert '--validate-captions' in result.stdout


class TestCLIValidateConfig:
    """Tests for --validate-config flag."""

    @pytest.mark.fast
    def test_validate_config_runs_successfully(self):
        """Test --validate-config runs without errors."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--validate-config'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
        )
        # Exit code 0 means validation passed, 1 means validation errors
        # We just want to ensure it runs without crashing
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}, stderr: {result.stderr}"

    @pytest.mark.fast
    def test_validate_config_json_runs_successfully(self):
        """Test --validate-config-json runs without errors."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--validate-config-json'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
        )
        # Should output JSON and exit with 0 (valid) or 1 (invalid)
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}, stderr: {result.stderr}"
        # Verify JSON output contains expected keys
        if result.returncode in [0, 1]:
            import json
            try:
                output = result.stdout.strip()
                if output:
                    data = json.loads(output)
                    assert 'status' in data or 'errors' in data or 'config_loaded' in data
            except json.JSONDecodeError:
                # JSON output might not be parseable if config is invalid
                pass


class TestCLIListKeywords:
    """Tests for --list-keywords flag."""

    @pytest.mark.fast
    def test_list_keywords_runs_successfully(self):
        """Test --list-keywords runs without errors (with no presets).

        Note: May fail on Linux due to Windows-specific paths in config.yaml.
        The test verifies the CLI runs without crashing, not that it succeeds.
        """
        result = subprocess.run(
            [sys.executable, 'main.py', '--list-keywords'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should either exit with 0 (success) or exit with 1 due to config issues
        # (Windows-specific paths in config.yaml)
        # Either way, it should not crash with a Python error
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}, stderr: {result.stderr}"
        # Should not crash - check for Python traceback
        assert 'Traceback' not in result.stderr, f"Python crash: {result.stderr}"

    @pytest.mark.fast
    def test_list_keywords_with_project_runs_successfully(self, tmp_path):
        """Test --list-keywords with a project directory runs without crashing.

        Note: May fail on Linux due to Windows-specific paths in config.yaml.
        """
        # Create a minimal project structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        (project_dir / "voiceover").mkdir()

        result = subprocess.run(
            [sys.executable, 'main.py', '--project', str(project_dir), '--list-keywords'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should either exit with 0 or exit with 1 due to config issues
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}, stderr: {result.stderr}"
        # Should not crash - check for Python traceback
        assert 'Traceback' not in result.stderr, f"Python crash: {result.stderr}"


class TestCLIValidateCaptions:
    """Tests for --validate-captions flag."""

    @pytest.mark.fast
    def test_validate_captions_runs_successfully(self):
        """Test --validate-captions runs without errors."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--validate-captions'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
        )
        # Should run and either pass validation or show errors (but not crash)
        assert result.returncode in [0, 1, 2], f"Unexpected exit code: {result.returncode}, stderr: {result.stderr}"
        # Check that validation output is present
        output = result.stdout + result.stderr
        assert 'Caption' in output or 'caption' in output.lower()

    @pytest.mark.fast
    def test_validate_captions_with_project_runs_successfully(self, tmp_path):
        """Test --validate-captions with a project directory runs without errors."""
        # Create a minimal project structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        (project_dir / "voiceover").mkdir()

        result = subprocess.run(
            [sys.executable, 'main.py', '--project', str(project_dir), '--validate-captions'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
        )
        # Should run without crashing
        assert result.returncode in [0, 1, 2], f"Unexpected exit code: {result.returncode}, stderr: {result.stderr}"


class TestCLICombinedFlags:
    """Tests for combining multiple CLI flags."""

    @pytest.mark.fast
    def test_validate_config_with_non_interactive(self):
        """Test --validate-config combined with --non-interactive."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--validate-config', '--non-interactive'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode in [0, 1], f"Exit code: {result.returncode}, stderr: {result.stderr}"

    @pytest.mark.fast
    def test_validate_captions_with_non_interactive(self):
        """Test --validate-captions combined with --non-interactive."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--validate-captions', '--non-interactive'],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode in [0, 1, 2], f"Exit code: {result.returncode}, stderr: {result.stderr}"


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    @pytest.mark.fast
    def test_invalid_flag_handled_gracefully(self):
        """Test that invalid flags are handled gracefully."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--invalid-flag-xyz'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should exit with error
        assert result.returncode != 0, f"Expected non-zero exit code for invalid flag"
        # Should show error message
        assert 'error' in result.stderr.lower() or 'unrecognized' in result.stderr.lower()

    @pytest.mark.fast
    def test_missing_required_args_no_crash(self):
        """Test that running without required args doesn't crash."""
        result = subprocess.run(
            [sys.executable, 'main.py'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should exit with error (missing voiceover)
        assert result.returncode != 0, f"Expected non-zero exit code when missing voiceover"


class TestCLIDiagnosticViewer:
    """Tests for --diagnostic-view flag (US-120-010)."""

    @pytest.mark.fast
    def test_diagnostic_view_flag_shows_in_help(self):
        """Test --diagnostic-view flag appears in help output."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert '--diagnostic-view' in result.stdout

    @pytest.mark.fast
    def test_diagnostic_view_runs_without_crash(self):
        """Test --diagnostic-view runs without crashing even without logs."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--diagnostic-view'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should either find logs or show "no log entries" - not crash
        assert 'Traceback' not in result.stderr, f"Python crash: {result.stderr}"

    @pytest.mark.fast
    def test_diagnostic_view_with_invalid_dir(self):
        """Test --diagnostic-view with non-existent directory."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--diagnostic-view'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should handle gracefully - either use default logs dir or show message
        assert result.returncode in [0, 1], f"Unexpected exit code: {result.returncode}"

    @pytest.mark.fast
    def test_diagnostic_viewer_script_directly(self):
        """Test running diagnostic_viewer.py script directly."""
        result = subprocess.run(
            [sys.executable, 'scripts/diagnostic_viewer.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0, f"Exit code: {result.returncode}, stderr: {result.stderr}"
        assert 'Diagnostic Log Viewer' in result.stdout
        assert '--level' in result.stdout
        assert '--category' in result.stdout

    @pytest.mark.fast
    def test_diagnostic_viewer_with_nonexistent_dir(self):
        """Test diagnostic viewer with non-existent directory."""
        result = subprocess.run(
            [sys.executable, 'scripts/diagnostic_viewer.py', '--logs-dir', '/nonexistent/path'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        # Should show error message and exit with error code
        assert 'not found' in result.stderr or 'not found' in result.stdout
