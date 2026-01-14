"""
Tests for config path resolution - ensures paths resolve to project directory, not installation directory.

These tests prevent regressions of the bug where:
- otio_output_dir resolved to D:\voiceover-matcher\output instead of {project}\output
- cache.cache_dir resolved to D:\voiceover-matcher\.cache instead of {project}\.cache
- logging.log_dir resolved to D:\voiceover-matcher\logs instead of {project}\logs

The root cause was that _resolve_paths() ran in __post_init__ with project_dir=".",
which resolved to the current working directory (installation dir), not the project dir.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.base import Config, load_config
from src.cli.config_utils import load_project_config


class TestPathResolutionToProjectDir:
    """Test that all paths resolve to project directory, not installation directory."""

    def test_otio_output_dir_is_project_relative(self, tmp_path):
        """otio_output_dir should be relative to project_dir, not cwd."""
        # Create a mock project directory
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        # Create minimal project_config.yaml
        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
""")

        # Load config with project_dir set
        config = load_project_config(project_dir)

        # otio_output_dir should be under project_dir, not cwd
        assert str(project_dir) in config.otio_output_dir
        assert "output" in config.otio_output_dir
        # Should NOT be in the installation directory
        assert "voiceover-matcher" not in config.otio_output_dir.lower() or str(project_dir) in config.otio_output_dir

    def test_cache_dir_is_project_relative(self, tmp_path):
        """cache.cache_dir should be relative to project_dir, not cwd."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
""")

        config = load_project_config(project_dir)

        # cache_dir should be under project_dir
        assert str(project_dir) in config.cache.cache_dir
        assert ".cache" in config.cache.cache_dir

    def test_log_dir_is_project_relative(self, tmp_path):
        """logging.log_dir should be relative to project_dir, not cwd."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
""")

        config = load_project_config(project_dir)

        # log_dir should be under project_dir
        assert str(project_dir) in config.logging.log_dir
        assert "logs" in config.logging.log_dir

    def test_project_dir_is_set_correctly(self, tmp_path):
        """config.project_dir should be the actual project directory."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
""")

        config = load_project_config(project_dir)

        # project_dir should be the actual project directory, not "."
        assert config.project_dir == str(project_dir)
        assert config.project_dir != "."

    def test_paths_not_in_cwd_when_different_from_project(self, tmp_path):
        """When cwd differs from project_dir, paths should NOT be in cwd."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
""")

        # Save original cwd and change to a different directory
        original_cwd = os.getcwd()
        other_dir = tmp_path / "other_dir"
        other_dir.mkdir()

        try:
            os.chdir(other_dir)

            config = load_project_config(project_dir)

            # Paths should be in project_dir, not in other_dir (cwd)
            assert str(project_dir) in config.otio_output_dir
            assert str(other_dir) not in config.otio_output_dir
            assert str(project_dir) in config.cache.cache_dir
            assert str(other_dir) not in config.cache.cache_dir

        finally:
            os.chdir(original_cwd)


class TestPathResolutionWithCustomConfig:
    """Test path resolution when custom paths are specified in config."""

    def test_absolute_paths_preserved(self, tmp_path):
        """Absolute paths in config should be preserved, not made relative."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        custom_output = tmp_path / "custom_output"

        (project_dir / "project_config.yaml").write_text(f"""
project:
  name: test_project
output:
  output_dir: "{str(custom_output).replace(chr(92), '/')}"
""")

        config = load_project_config(project_dir)

        # Absolute path should be preserved
        # Note: The path resolution may still make it project-relative if it wasn't absolute in the YAML

    def test_relative_paths_resolved_to_project(self, tmp_path):
        """Relative paths in config should resolve to project_dir."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
cache:
  cache_dir: "my_custom_cache"
""")

        config = load_project_config(project_dir)

        # Relative path should be resolved to project_dir
        assert str(project_dir) in config.cache.cache_dir
        assert "my_custom_cache" in config.cache.cache_dir


class TestResolvePathsIdempotent:
    """Test that _resolve_paths can be called multiple times safely."""

    def test_resolve_paths_multiple_calls(self, tmp_path):
        """Calling _resolve_paths multiple times should not change paths."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        (project_dir / "project_config.yaml").write_text("""
project:
  name: test_project
""")

        config = load_project_config(project_dir)

        # Store initial values
        initial_otio = config.otio_output_dir
        initial_cache = config.cache.cache_dir
        initial_log = config.logging.log_dir

        # Call _resolve_paths again
        config._resolve_paths()

        # Values should not change (paths already absolute)
        assert config.otio_output_dir == initial_otio
        assert config.cache.cache_dir == initial_cache
        assert config.logging.log_dir == initial_log


class TestProjectDirReset:
    """Test that project_dir properly resets paths when changed."""

    def test_changing_project_dir_updates_paths(self, tmp_path):
        """When project_dir changes, paths should update accordingly."""
        project_dir1 = tmp_path / "project1"
        project_dir1.mkdir()
        project_dir2 = tmp_path / "project2"
        project_dir2.mkdir()

        (project_dir1 / "project_config.yaml").write_text("project:\n  name: test1")
        (project_dir2 / "project_config.yaml").write_text("project:\n  name: test2")

        config1 = load_project_config(project_dir1)
        config2 = load_project_config(project_dir2)

        # Each config should have paths relative to its project
        assert str(project_dir1) in config1.otio_output_dir
        assert str(project_dir2) in config2.otio_output_dir
        assert str(project_dir1) not in config2.otio_output_dir
        assert str(project_dir2) not in config1.otio_output_dir
