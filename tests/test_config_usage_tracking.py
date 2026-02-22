"""Tests for config usage tracking (US-142-003)."""

import pytest
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import patch

from src.config.base import Config


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = """
project:
  name: test_project
  version: "4.0.0"

matching:
  min_confidence: 0.7

download:
  max_workers: 4
"""
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


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


class TestConfigUsageTracking:
    """Test config usage tracking functionality."""

    def test_usage_tracker_initialized_empty(self):
        """Test that _usage_tracker is initialized as empty dict."""
        config = Config()
        # Reset to ensure clean state
        config.reset_usage_stats()
        assert config._usage_tracker == {}
        assert config.get_usage_stats() == {}

    def test_track_field_access_via_getattribute(self):
        """Test that accessing fields increments usage count."""
        config = Config()
        # Reset first to get clean state
        initial_stats = config.get_usage_stats()
        config.reset_usage_stats()

        # Access some fields
        _ = config.matching
        _ = config.download
        _ = config.matching  # Access again

        stats = config.get_usage_stats()
        # After reset, we should have tracking
        assert stats.get('matching', 0) >= 2
        assert stats.get('download', 0) >= 1

    def test_track_usage_manual(self):
        """Test manual track_usage method."""
        config = Config()
        config.reset_usage_stats()

        # Manually track some fields
        config.track_usage('matching.min_confidence')
        config.track_usage('matching.min_confidence')
        config.track_usage('download.max_workers')

        stats = config.get_usage_stats()
        assert stats.get('matching.min_confidence', 0) == 2
        assert stats.get('download.max_workers', 0) == 1

    def test_reset_usage_stats(self):
        """Test resetting usage statistics."""
        config = Config()

        # Access some fields
        _ = config.matching
        _ = config.download

        # Reset
        config.reset_usage_stats()

        assert config.get_usage_stats() == {}

    def test_private_fields_not_tracked(self):
        """Test that private fields are not tracked."""
        config = Config()
        config.reset_usage_stats()

        # Access private field
        _ = config._config_path
        _ = config._loaded_at
        _ = config._frozen

        # Access public field
        _ = config.matching

        stats = config.get_usage_stats()
        # Private fields should not be tracked
        assert '_config_path' not in stats
        assert '_loaded_at' not in stats
        # Public field should be tracked
        assert 'matching' in stats

    def test_nested_field_access_tracked(self):
        """Test that nested field access is tracked on parent."""
        config = Config()
        config.reset_usage_stats()

        # Access nested fields (matching is a dataclass section)
        _ = config.matching
        _ = config.matching.min_confidence

        stats = config.get_usage_stats()
        # The __getattribute__ tracks on the parent object access
        assert 'matching' in stats

    def test_usage_stats_in_json_export(self, temp_config_file, mock_api_keys):
        """Test that usage stats are included in JSON export metadata."""
        config = Config.from_yaml(temp_config_file)
        config.reset_usage_stats()

        # Access some fields
        _ = config.matching
        _ = config.download
        _ = config.matching

        # Export to JSON
        json_output = config.to_json()

        # Parse and verify
        import json
        data = json.loads(json_output)

        assert 'metadata' in data
        assert 'usage_stats' in data['metadata']
        # After reset and accessing matching 2x and download 1x
        assert data['metadata']['usage_stats'].get('matching', 0) >= 2
        assert data['metadata']['usage_stats'].get('download', 0) >= 1

    def test_get_usage_stats_returns_copy(self):
        """Test that get_usage_stats returns a copy, not the original."""
        config = Config()
        config.reset_usage_stats()

        _ = config.matching

        stats1 = config.get_usage_stats()
        stats2 = config.get_usage_stats()

        # Should be equal
        assert stats1 == stats2

        # Modifying returned dict should not affect internal tracker
        if 'matching' in stats1:
            stats1['matching'] = 999
            stats3 = config.get_usage_stats()
            assert stats3.get('matching', 0) != 999

    def test_non_existent_field_no_error(self):
        """Test that accessing non-existent fields doesn't break tracking."""
        config = Config()
        config.reset_usage_stats()

        # This should raise AttributeError as normal
        with pytest.raises(AttributeError):
            _ = config.nonexistent_field

        # But tracking should still work
        config.track_usage('test_field')
        assert config.get_usage_stats().get('test_field', 0) == 1
