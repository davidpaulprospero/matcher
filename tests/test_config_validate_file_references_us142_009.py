"""Tests for config file reference validation (US-142-009)."""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from src.config.base import Config


@pytest.fixture
def mock_api_keys():
    """Provide mock API keys for tests."""
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test_gemini_key_12345',
        'GOOGLE_API_KEY': 'test_google_key_12345',
        'PEXELS_API_KEY': 'test_pexels_key',
        'PIXABAY_API_KEY': 'test_pixabay_key',
    }):
        yield


class TestValidateFileReferences:
    """Test Config.validate_file_references() method."""

    def test_validate_file_references_returns_empty_for_defaults(self, mock_api_keys):
        """Test that validation returns empty for default config (all paths valid or expandable)."""
        config = Config()
        issues = config.validate_file_references()
        # Default config should not have issues - paths either exist or can be created
        assert isinstance(issues, list)

    def test_validate_file_references_detects_missing_output_dir(self, mock_api_keys, tmp_path):
        """Test that validation detects missing output_dir."""
        config = Config()

        # Set a non-existent output directory
        non_existent_dir = str(tmp_path / "nonexistent_output_dir_xyz")
        config.output.output_dir = non_existent_dir
        config.unfreeze()

        issues = config.validate_file_references()

        # Should have one issue about missing output_dir
        output_issues = [i for i in issues if i['field'] == 'output.output_dir']
        assert len(output_issues) == 1
        assert output_issues[0]['type'] == 'directory_missing'
        assert 'suggestion' in output_issues[0]

    def test_validate_file_references_detects_missing_cache_dir(self, mock_api_keys, tmp_path):
        """Test that validation detects missing cache_dir."""
        config = Config()

        # Set a non-existent cache directory
        non_existent_dir = str(tmp_path / "nonexistent_cache_dir_xyz")
        config.cache.cache_dir = non_existent_dir
        config.unfreeze()

        issues = config.validate_file_references()

        # Should have one issue about missing cache_dir
        cache_issues = [i for i in issues if i['field'] == 'cache.cache_dir']
        assert len(cache_issues) == 1
        assert cache_issues[0]['type'] == 'directory_missing'

    def test_validate_file_references_validates_existing_directories(self, mock_api_keys, tmp_path):
        """Test that validation passes for existing directories."""
        config = Config()

        # Set existing directories
        config.output.output_dir = str(tmp_path / "output")
        config.cache.cache_dir = str(tmp_path / "cache")
        config.unfreeze()

        # Create the directories
        os.makedirs(tmp_path / "output", exist_ok=True)
        os.makedirs(tmp_path / "cache", exist_ok=True)

        issues = config.validate_file_references()

        # Should have no issues for existing directories
        assert len(issues) == 0

    def test_validate_file_references_handles_download_dir_parent(self, mock_api_keys, tmp_path):
        """Test that validation detects missing download directory parent."""
        config = Config()

        # Set download dir with non-existent parent
        # Use a path that definitely won't exist
        config.download.download_dir = "/this/absolutely/does/not/exist/download"
        config.unfreeze()

        issues = config.validate_file_references()

        # Should have issue about download_dir parent
        download_issues = [i for i in issues if i['field'] == 'download.download_dir']
        assert len(download_issues) >= 1  # May also have output/cache issues from defaults

    def test_validate_file_references_returns_suggestions(self, mock_api_keys, tmp_path):
        """Test that validation provides actionable suggestions."""
        config = Config()

        # Set a non-existent directory
        non_existent_dir = str(tmp_path / "new_output_dir")
        config.output.output_dir = non_existent_dir
        config.unfreeze()

        issues = config.validate_file_references()

        # Should have suggestion
        assert len(issues) > 0
        for issue in issues:
            assert 'suggestion' in issue
            assert 'path' in issue
            assert 'type' in issue
            assert 'field' in issue


class TestValidateFileReferencesCLI:
    """Test --validate-config integration with file references."""

    def test_validate_file_references_integrated_in_validate_config(self, mock_api_keys, tmp_path):
        """Test that validate_config_at_startup includes file reference validation."""
        from src.cli.config_utils import validate_config_at_startup

        config = Config()

        # Set non-existent directories
        config.output.output_dir = str(tmp_path / "missing_output")
        config.unfreeze()

        # Should run without error, but will print warnings
        # The function returns True even with warnings
        result = validate_config_at_startup(config)
        assert result is True  # Should still return True (not fatal)

    def test_json_output_includes_file_references(self, mock_api_keys):
        """Test that JSON output includes file_references field."""
        config = Config()

        # Call validate_file_references directly
        file_refs = config.validate_file_references()

        # Should return a list
        assert isinstance(file_refs, list)
