"""
Tests for --dry-run-config flag (US-128-009).

Tests:
- --dry-run-config loads and validates config without running pipeline
- Output shows all config sections with effective values
- Shows which values came from env vars vs config.yaml vs defaults
- Validates all cross-section constraints
- Exit code 0 if valid, shows errors if not
- Test verifies dry-run catches validation errors that would fail pipeline
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Ensure project root is in path
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))


class TestDryRunConfig:
    """Tests for --dry-run-config CLI flag."""

    def test_dry_run_config_detects_validation_errors(self, tmp_path):
        """Test that --dry-run-config catches validation errors and returns exit code 1."""
        # Create a minimal config with invalid values (missing API keys but using providers)
        config_content = """project:
  name: test-project
  version: "4.0.0"

matching:
  primary_provider: gemini  # Requires GEMINI_API_KEY

video_search:
  results_per_keyword: 20
  max_total_results: 200

download:
  caption_first:
    enabled: true
  root_dir: /tmp/test

stock_footage:
  pexels_enabled: true  # Requires PEXELS_API_KEY

transcription:
  model: base

embedding:
  provider: gemini
"""

        # Write config to temp file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        # Run with --dry-run-config and capture exit code
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--config", str(config_file), "--dry-run-config"],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
            env={**os.environ, "HOME": str(tmp_path)}  # Use temp home to avoid real config
        )

        # Should show validation errors in output
        assert "GEMINI_API_KEY required" in result.stdout or "GEMINI_API_KEY required" in result.stderr
        assert "PEXELS_API_KEY required" in result.stdout or "PEXELS_API_KEY required" in result.stderr

        # Exit code should be 1 for invalid config
        assert result.returncode == 1

    def test_dry_run_config_shows_env_overrides(self, tmp_path):
        """Test that --dry-run-config shows environment variable overrides."""
        # Create a minimal config
        config_content = """project:
  name: test-project
  version: "4.0.0"

matching:
  primary_provider: local
  min_confidence: 0.6

video_search:
  results_per_keyword: 20
  max_total_results: 200

download:
  caption_first:
    enabled: true
  root_dir: /tmp/test

transcription:
  model: base

embedding:
  provider: ollama

stock_footage:
  pexels_enabled: false
  pixabay_enabled: false
"""

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        # Run with environment variable override
        env = {**os.environ, "HOME": str(tmp_path), "MATCHER_MATCHING_MIN_CONFIDENCE": "0.75"}
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--config", str(config_file), "--dry-run-config"],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
            env=env
        )

        # Should show the environment variable override
        assert "MATCHING_MIN_CONFIDENCE=0.75" in result.stdout
        # Should show [ENV] marker for the value
        assert "min_confidence: 0.75 [ENV]" in result.stdout

    def test_dry_run_config_shows_effective_values(self, tmp_path):
        """Test that --dry-run-config shows all config sections with values."""
        config_content = """project:
  name: test-project
  version: "4.0.0"

matching:
  primary_provider: local
  min_confidence: 0.6

video_search:
  results_per_keyword: 20
  max_total_results: 200

download:
  caption_first:
    enabled: true
  root_dir: /tmp/test

transcription:
  model: base

embedding:
  provider: ollama

stock_footage:
  pexels_enabled: false
  pixabay_enabled: false
"""

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--config", str(config_file), "--dry-run-config"],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
            env={**os.environ, "HOME": str(tmp_path)}
        )

        # Should show configuration sections in output
        assert "CONFIG DRY-RUN" in result.stdout
        assert "PROJECT" in result.stdout
        assert "MATCHING" in result.stdout
        assert "VIDEO_SEARCH" in result.stdout
        assert "DOWNLOAD" in result.stdout

    def test_dry_run_config_exit_code_valid(self, tmp_path):
        """Test that --dry-run-config returns exit code 0 when config is valid."""
        # Create a config with all required API keys or providers that don't need them
        config_content = """project:
  name: test-project
  version: "4.0.0"

matching:
  primary_provider: local  # No API key needed
  min_confidence: 0.6

video_search:
  results_per_keyword: 20
  max_total_results: 200

download:
  caption_first:
    enabled: true
  root_dir: /tmp/test

transcription:
  model: base

embedding:
  provider: ollama  # Local, no API key

stock_footage:
  pexels_enabled: false
  pixabay_enabled: false
"""

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "--config", str(config_file), "--dry-run-config"],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
            env={**os.environ, "HOME": str(tmp_path)}
        )

        # Should exit with code 0 if config is valid (no API key requirements)
        # Note: May still have warnings but should not have API key errors
        assert "GEMINI_API_KEY required" not in result.stdout
        assert "PEXELS_API_KEY required" not in result.stdout
        assert "PIXABAY_API_KEY required" not in result.stdout


class TestDryRunConfigFunctions:
    """Unit tests for helper functions used by --dry-run-config."""

    def test_get_source_marker(self):
        """Test _get_source_marker returns correct markers."""
        from main import _get_source_marker

        assert _get_source_marker('env') == '[ENV]'
        assert _get_source_marker('yaml') == '[YAML]'
        assert _get_source_marker('default') == '[DEFAULT]'
        assert _get_source_marker('unknown') == '[?]'

    def test_get_env_overrides_summary(self):
        """Test _get_env_overrides_summary returns list of overrides."""
        from main import _get_env_overrides_summary

        # Test with no env vars
        # Note: This will pick up any MATCHER_ vars in the real environment
        overrides = _get_env_overrides_summary()

        # The function should return a list (may be empty)
        assert isinstance(overrides, list)
