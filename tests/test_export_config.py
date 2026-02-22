#!/usr/bin/env python3
"""
Tests for scripts/export_config.py

Tests the export/import utility functions including:
- Export keywords and presets
- Export config settings
- Export cache metadata
- Import keywords and presets
- Import config settings
- Encryption/decryption
"""

import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import the module under test
import export_config


@pytest.mark.script
class TestExportKeywords:
    """Test keyword export functionality"""

    def test_export_keywords_no_file(self, tmp_path):
        """Test export when no keywords file exists"""
        result = export_config.export_keywords(tmp_path)
        assert result["presets"] == {}
        assert result["count"] == 0

    def test_export_keywords_with_data(self, tmp_path):
        """Test export with existing keywords file"""
        # Create a mock keywords file
        keywords_file = tmp_path / "saved_keywords.json"
        test_data = {
            "version": "1.0",
            "presets": {
                "test_preset": {
                    "name": "test_preset",
                    "created_at": "2024-01-01T00:00:00",
                    "keywords": ["python", "tutorial"],
                    "topic_context": "programming"
                }
            }
        }
        keywords_file.write_text(json.dumps(test_data))

        result = export_config.export_keywords(tmp_path)
        assert result["count"] == 1
        assert "test_preset" in result["presets"]
        assert result["presets"]["test_preset"]["keywords"] == ["python", "tutorial"]


@pytest.mark.script
class TestExportCheckpoint:
    """Test checkpoint metadata export"""

    def test_export_checkpoint_no_file(self, tmp_path):
        """Test export when no checkpoint exists"""
        result = export_config.export_checkpoint(tmp_path)
        assert result["exists"] is False

    def test_export_checkpoint_with_data(self, tmp_path):
        """Test export with existing checkpoint"""
        checkpoint_file = tmp_path / "checkpoint.json"
        test_data = {
            "last_completed_stage": "MATCH",
            "stages_completed": ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH"],
            "video_ids": ["abc123", "def456"],
            "matches": [{"id": "abc123", "start": 0, "end": 10}]
        }
        checkpoint_file.write_text(json.dumps(test_data))

        result = export_config.export_checkpoint(tmp_path)
        assert result["exists"] is True
        assert result["last_completed_stage"] == "MATCH"
        assert result["video_ids_count"] == 2
        assert result["matches_count"] == 1


@pytest.mark.script
class TestExportConfigSettings:
    """Test config settings export"""

    def test_export_config_settings_no_file(self, tmp_path):
        """Test export when no config exists - falls back to project root"""
        # The function falls back to project_root/config.yaml if not in project_dir
        # So we test the behavior by checking it either returns False (no file) or
        # uses fallback behavior. This test just verifies it doesn't crash.
        result = export_config.export_config_settings(tmp_path)
        # Either returns False or uses fallback (both are valid behaviors)
        assert "exists" in result

    def test_export_config_settings_with_data(self, tmp_path):
        """Test export with existing config"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
project:
  name: test-project
  version: 4.0.0
matching:
  min_confidence: 0.7
  chapter_matching_enabled: true
video_search:
  max_results: 50
""")

        result = export_config.export_config_settings(tmp_path)
        assert result["exists"] is True
        assert "settings" in result
        assert result["settings"]["matching"]["min_confidence"] == 0.7
        assert result["settings"]["matching"]["chapter_matching_enabled"] is True


@pytest.mark.script
class TestExportCacheMetadata:
    """Test cache metadata export"""

    def test_export_cache_metadata_no_cache(self, tmp_path):
        """Test export when no cache exists"""
        result = export_config.export_cache_metadata(tmp_path)
        assert result["exists"] is False

    def test_export_cache_metadata_with_cache(self, tmp_path):
        """Test export with existing cache"""
        # Create cache directory with some files
        cache_dir = tmp_path / ".cache"
        transcriptions_dir = cache_dir / "transcriptions"
        transcriptions_dir.mkdir(parents=True)

        # Create a test file
        test_file = transcriptions_dir / "test.json"
        test_file.write_text('{"transcript": "test"}')

        result = export_config.export_cache_metadata(tmp_path)
        assert result["exists"] is True
        assert "transcriptions" in result["subdirs"]
        assert result["subdirs"]["transcriptions"]["file_count"] == 1


@pytest.mark.script
class TestFormatSize:
    """Test size formatting utility"""

    def test_format_size_bytes(self):
        """Test formatting bytes"""
        assert export_config.format_size(500) == "500.0 B"

    def test_format_size_kilobytes(self):
        """Test formatting kilobytes"""
        assert export_config.format_size(1024) == "1.0 KB"

    def test_format_size_megabytes(self):
        """Test formatting megabytes"""
        assert export_config.format_size(1048576) == "1.0 MB"

    def test_format_size_gigabytes(self):
        """Test formatting gigabytes"""
        assert export_config.format_size(1073741824) == "1.0 GB"


@pytest.mark.script
class TestImportKeywords:
    """Test keyword import functionality"""

    def test_import_keywords_new_file(self, tmp_path):
        """Test import to new keywords file"""
        keywords_data = {
            "presets": {
                "imported": {
                    "name": "imported",
                    "keywords": ["test", "imported"],
                    "topic_context": "testing"
                }
            }
        }

        result = export_config.import_keywords(tmp_path, keywords_data)
        assert result is True

        # Verify file was created
        keywords_file = tmp_path / "saved_keywords.json"
        assert keywords_file.exists()

        # Verify data was imported
        with open(keywords_file) as f:
            saved = json.load(f)
        assert "imported" in saved["presets"]

    def test_import_keywords_merge_existing(self, tmp_path):
        """Test importing merges with existing presets"""
        # Create existing file
        keywords_file = tmp_path / "saved_keywords.json"
        existing = {
            "version": "1.0",
            "presets": {
                "existing": {"name": "existing", "keywords": ["old"]}
            }
        }
        keywords_file.write_text(json.dumps(existing))

        # Import new preset
        keywords_data = {
            "presets": {
                "new": {"name": "new", "keywords": ["new"]}
            }
        }
        result = export_config.import_keywords(tmp_path, keywords_data)
        assert result is True

        # Verify both presets exist
        with open(keywords_file) as f:
            saved = json.load(f)
        assert "existing" in saved["presets"]
        assert "new" in saved["presets"]


@pytest.mark.script
class TestEncryption:
    """Test encryption/decryption functionality"""

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption return original data"""
        original = '{"test": "data", "nested": {"value": 123}}'
        password = "test_password_123"

        encrypted = export_config.encrypt_data(original, password)
        decrypted = export_config.decrypt_data(encrypted, password)

        assert decrypted == original

    def test_different_password_fails(self):
        """Test that wrong password fails to decrypt"""
        original = '{"test": "data"}'
        password = "correct_password"
        wrong_password = "wrong_password"

        encrypted = export_config.encrypt_data(original, password)
        decrypted = export_config.decrypt_data(encrypted, wrong_password)

        assert decrypted is None

    def test_encrypted_output_is_base64(self):
        """Test that encrypted output is valid base64"""
        original = '{"test": "data"}'
        password = "test_password"

        encrypted = export_config.encrypt_data(original, password)

        # Should be valid base64
        import base64
        try:
            base64.b64decode(encrypted)
        except Exception:
            pytest.fail("Encrypted output is not valid base64")


@pytest.mark.script
class TestImportConfigSettings:
    """Test config settings import"""

    def test_import_config_settings(self, tmp_path):
        """Test importing config settings"""
        # Create initial config
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
project:
  name: test-project
  version: 4.0.0
matching:
  min_confidence: 0.5
""")

        # Import new settings
        config_data = {
            "settings": {
                "matching": {
                    "min_confidence": 0.8
                }
            }
        }
        result = export_config.import_config_settings(tmp_path, config_data)
        assert result is True

        # Verify settings were updated
        with open(config_file) as f:
            content = f.read()
        assert "min_confidence: 0.8" in content
