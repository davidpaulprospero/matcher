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

# This module uses os.chdir() and must run serially to avoid affecting other tests
pytestmark = pytest.mark.serial
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.base import Config, load_config
from src.cli.config_utils import load_project_config


class TestPathResolutionToProjectDir:
    """Test that all paths resolve to project directory, not installation directory."""

    @pytest.mark.fast
    def test_otio_output_dir_is_project_relative(self, tmp_path):
        """otio_output_dir should be relative to project_dir, not cwd."""
        # Create a mock project directory
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        # Load config with project_dir set
        config = load_project_config(project_dir)

        # otio_output_dir should be under project_dir, not cwd
        assert str(project_dir) in config.otio_output_dir
        assert "output" in config.otio_output_dir
        # Should NOT be in the installation directory
        assert "voiceover-matcher" not in config.otio_output_dir.lower() or str(project_dir) in config.otio_output_dir

    @pytest.mark.fast
    def test_cache_dir_is_project_relative(self, tmp_path):
        """cache.cache_dir should be relative to project_dir, not cwd."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = load_project_config(project_dir)

        # cache_dir should be under project_dir
        assert str(project_dir) in config.cache.cache_dir
        assert ".cache" in config.cache.cache_dir

    @pytest.mark.fast
    def test_log_dir_is_project_relative(self, tmp_path):
        """logging.log_dir should be relative to project_dir, not cwd."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = load_project_config(project_dir)

        # log_dir should be under project_dir
        assert str(project_dir) in config.logging.log_dir
        assert "logs" in config.logging.log_dir

    @pytest.mark.fast
    def test_project_dir_is_set_correctly(self, tmp_path):
        """config.project_dir should be the actual project directory."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = load_project_config(project_dir)

        # project_dir should be the actual project directory, not "."
        assert config.project_dir == str(project_dir)
        assert config.project_dir != "."

    @pytest.mark.fast
    def test_paths_not_in_cwd_when_different_from_project(self, tmp_path):
        """When cwd differs from project_dir, paths should NOT be in cwd."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

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

    @pytest.mark.fast
    def test_default_paths_resolved_to_project(self, tmp_path):
        """Default relative paths in config should resolve to project_dir."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = load_project_config(project_dir)

        # Default relative path should be resolved to project_dir
        assert str(project_dir) in config.cache.cache_dir
        assert ".cache" in config.cache.cache_dir


class TestResolvePathsIdempotent:
    """Test that _resolve_paths can be called multiple times safely."""

    @pytest.mark.fast
    def test_resolve_paths_multiple_calls(self, tmp_path):
        """Calling _resolve_paths multiple times should not change paths."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

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

    @pytest.mark.fast
    def test_changing_project_dir_updates_paths(self, tmp_path):
        """When project_dir changes, paths should update accordingly."""
        project_dir1 = tmp_path / "project1"
        project_dir1.mkdir()
        project_dir2 = tmp_path / "project2"
        project_dir2.mkdir()

        config1 = load_project_config(project_dir1)
        config2 = load_project_config(project_dir2)

        # Each config should have paths relative to its project
        assert str(project_dir1) in config1.otio_output_dir
        assert str(project_dir2) in config2.otio_output_dir
        assert str(project_dir1) not in config2.otio_output_dir
        assert str(project_dir2) not in config1.otio_output_dir


class TestUnifiedPathResolution:
    """Test that path resolution is unified - _resolve_paths() is the single source of truth."""

    @pytest.mark.fast
    def test_relative_paths_resolve_to_absolute_exactly_once(self, tmp_path):
        """Loading config with relative paths resolves them to absolute project-relative paths exactly once."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = load_project_config(project_dir)

        project_str = str(project_dir)

        # All section-level paths should be absolute and under project_dir
        assert Path(config.output.output_dir).is_absolute()
        assert project_str in config.output.output_dir

        assert Path(config.cache.cache_dir).is_absolute()
        assert project_str in config.cache.cache_dir

        assert Path(config.logging.log_dir).is_absolute()
        assert project_str in config.logging.log_dir

        assert Path(config.transcription.cache_dir).is_absolute()
        assert project_str in config.transcription.cache_dir

        # Convenience paths too
        assert Path(config.otio_output_dir).is_absolute()
        assert project_str in config.otio_output_dir

        # Call _resolve_paths again - should be idempotent (no double-resolution)
        snapshot = {
            'output.output_dir': config.output.output_dir,
            'cache.cache_dir': config.cache.cache_dir,
            'logging.log_dir': config.logging.log_dir,
            'transcription.cache_dir': config.transcription.cache_dir,
            'otio_output_dir': config.otio_output_dir,
        }

        config._resolve_paths()

        assert config.output.output_dir == snapshot['output.output_dir']
        assert config.cache.cache_dir == snapshot['cache.cache_dir']
        assert config.logging.log_dir == snapshot['logging.log_dir']
        assert config.transcription.cache_dir == snapshot['transcription.cache_dir']
        assert config.otio_output_dir == snapshot['otio_output_dir']

    @pytest.mark.fast
    def test_mixed_separator_paths_resolve_correctly(self, tmp_path):
        """Paths with mixed forward/backslashes resolve correctly on Windows."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        # Set paths with mixed separators
        config.output.output_dir = "sub/dir\\output"
        config.cache.cache_dir = "data\\cache/store"
        config.project_dir = str(project_dir)
        config._resolve_paths()

        # Both should resolve to valid absolute paths under project_dir
        assert Path(config.output.output_dir).is_absolute()
        assert str(project_dir) in config.output.output_dir
        assert Path(config.cache.cache_dir).is_absolute()
        assert str(project_dir) in config.cache.cache_dir

    @pytest.mark.fast
    def test_make_paths_delegates_to_resolve_paths(self, tmp_path):
        """make_paths_project_relative delegates to _resolve_paths (no duplicate logic)."""
        from src.cli.config_utils import make_paths_project_relative

        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        result = make_paths_project_relative(config, project_dir)

        project_str = str(project_dir)

        # Should resolve all the same paths as load_project_config
        assert result is config
        assert project_str in config.output.output_dir
        assert project_str in config.cache.cache_dir
        assert project_str in config.transcription.cache_dir
        assert project_str in config.logging.log_dir

    @pytest.mark.fast
    def test_absolute_paths_preserved(self, tmp_path):
        """Absolute paths in config sections are not modified by _resolve_paths."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        abs_output = str(tmp_path / "custom_output")

        config = Config()
        config.project_dir = str(project_dir)
        config.output.output_dir = abs_output
        config._resolve_paths()

        # Absolute path should remain unchanged
        assert config.output.output_dir == abs_output

    @pytest.mark.fast
    def test_forward_slash_paths_resolve_correctly(self, tmp_path):
        """Paths with forward slashes resolve correctly on Windows."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        config.output.output_dir = "sub/dir/output"
        config.logging.log_dir = "logs/app"
        config.project_dir = str(project_dir)
        config._resolve_paths()

        assert Path(config.output.output_dir).is_absolute()
        assert str(project_dir) in config.output.output_dir
        assert Path(config.logging.log_dir).is_absolute()
        assert str(project_dir) in config.logging.log_dir

    @pytest.mark.fast
    def test_backslash_paths_resolve_correctly(self, tmp_path):
        """Paths with backslashes resolve correctly on Windows."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        config.output.output_dir = "sub\\dir\\output"
        config.cache.cache_dir = "data\\cache"
        config.project_dir = str(project_dir)
        config._resolve_paths()

        assert Path(config.output.output_dir).is_absolute()
        assert str(project_dir) in config.output.output_dir
        assert Path(config.cache.cache_dir).is_absolute()
        assert str(project_dir) in config.cache.cache_dir

    @pytest.mark.fast
    def test_no_double_resolution_of_project_relative_paths(self, tmp_path):
        """Project-relative paths are only resolved once - no double-resolution."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        config.project_dir = str(project_dir)

        # First resolution: relative -> absolute
        config.output.output_dir = "output"
        config.cache.cache_dir = ".cache"
        config.logging.log_dir = "logs"
        config.transcription.cache_dir = "transcriptions"
        config._resolve_paths()

        # All should be absolute and under project_dir exactly once
        expected_output = str(project_dir / "output")
        expected_cache = str(project_dir / ".cache")
        expected_log = str(project_dir / "logs")
        expected_trans = str(project_dir / "transcriptions")

        assert config.output.output_dir == expected_output
        assert config.cache.cache_dir == expected_cache
        assert config.logging.log_dir == expected_log
        assert config.transcription.cache_dir == expected_trans

        # Second call should NOT double-resolve (idempotent)
        config._resolve_paths()

        assert config.output.output_dir == expected_output
        assert config.cache.cache_dir == expected_cache
        assert config.logging.log_dir == expected_log
        assert config.transcription.cache_dir == expected_trans

    @pytest.mark.fast
    def test_load_project_config_delegates_to_shared_resolution(self, tmp_path):
        """load_project_config delegates to make_paths_project_relative (single implementation)."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = load_project_config(project_dir)
        project_str = str(project_dir)

        # All paths should be resolved exactly as make_paths_project_relative does
        assert Path(config.output.output_dir).is_absolute()
        assert project_str in config.output.output_dir
        assert Path(config.cache.cache_dir).is_absolute()
        assert project_str in config.cache.cache_dir
        assert Path(config.logging.log_dir).is_absolute()
        assert project_str in config.logging.log_dir
        assert Path(config.transcription.cache_dir).is_absolute()
        assert project_str in config.transcription.cache_dir


class TestProjectNameTruncation:
    """Test that project name truncation uses max_name_display_length from config."""

    @pytest.mark.fast
    def test_long_project_name_truncated_to_default_length(self, tmp_path):
        """Project names longer than default max_name_display_length (15) are truncated."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        config.project_dir = str(project_dir)
        config.download.root_dir = str(tmp_path / "v")
        config.project.name = "a_very_long_project_name_exceeding_fifteen"
        config._resolve_paths()

        # Default max_name_display_length is 15
        truncated = "a_very_long_pro"  # first 15 chars
        assert config.downloaded_videos_dir.endswith(truncated)

    @pytest.mark.fast
    def test_custom_max_name_display_length_truncates_correctly(self, tmp_path):
        """Setting max_name_display_length changes the truncation point."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        config.project_dir = str(project_dir)
        config.download.root_dir = str(tmp_path / "v")
        config.project.name = "a_very_long_project_name_exceeding_fifteen"
        config.project.max_name_display_length = 10
        config._resolve_paths()

        truncated = "a_very_lon"  # first 10 chars
        assert config.downloaded_videos_dir.endswith(truncated)

    @pytest.mark.fast
    def test_short_project_name_not_truncated(self, tmp_path):
        """Project names shorter than max_name_display_length are not truncated."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()

        config = Config()
        config.project_dir = str(project_dir)
        config.download.root_dir = str(tmp_path / "v")
        config.project.name = "short"
        config._resolve_paths()

        assert config.downloaded_videos_dir.endswith("short")
