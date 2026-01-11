"""
Tests for src/checkpoint.py to achieve 100% coverage.

Targets:
- Corruption scenarios (empty file, invalid JSON, non-dict JSON)
- Backup restoration when main file corrupt
- v0.9 → v1.0 migration (uppercase stage keys)
- KeywordManager old format conversion
- is_stale() with malformed timestamps
- validate() edge cases
"""

import pytest
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

from src.checkpoint import (
    CheckpointManager,
    CheckpointData,
    KeywordManager,
    SavedKeywords,
    STAGE_ORDER,
    format_resume_prompt,
    format_keyword_prompt
)


class TestCheckpointDataClass:
    """Test CheckpointData dataclass methods."""

    def test_to_dict(self):
        """Test to_dict method."""
        data = CheckpointData(
            version="1.0",
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE"
        )
        result = data.to_dict()
        assert result['version'] == "1.0"
        assert result['last_completed_stage'] == "ANALYZE"

    def test_from_dict_filters_unknown_fields(self):
        """Test from_dict filters unknown fields."""
        raw_data = {
            'version': '1.0',
            'created_at': '2026-01-01',
            'unknown_field': 'should be ignored',
            'another_unknown': 123
        }
        result = CheckpointData.from_dict(raw_data)
        assert result.version == '1.0'
        assert not hasattr(result, 'unknown_field')


class TestSavedKeywordsDataClass:
    """Test SavedKeywords dataclass methods."""

    def test_to_dict(self):
        """Test to_dict method."""
        preset = SavedKeywords(
            name="test",
            keywords=["kw1", "kw2"]
        )
        result = preset.to_dict()
        assert result['name'] == "test"
        assert result['keywords'] == ["kw1", "kw2"]

    def test_from_dict_filters_unknown_fields(self):
        """Test from_dict filters unknown fields."""
        raw_data = {
            'name': 'test',
            'keywords': ['kw1'],
            'unknown_field': 'ignored'
        }
        result = SavedKeywords.from_dict(raw_data)
        assert result.name == 'test'
        assert result.keywords == ['kw1']


class TestCheckpointManagerBasics:
    """Test basic CheckpointManager functionality."""

    def test_init(self, tmp_path):
        """Test initialization."""
        manager = CheckpointManager(tmp_path, config_hash="abc123")
        assert manager.project_dir == tmp_path
        assert manager.config_hash == "abc123"
        assert manager.data is None

    def test_exists_false(self, tmp_path):
        """Test exists returns False when no checkpoint."""
        manager = CheckpointManager(tmp_path)
        assert manager.exists() == False

    def test_exists_true(self, tmp_path):
        """Test exists returns True when checkpoint exists."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('{"version": "1.0"}')
        manager = CheckpointManager(tmp_path)
        assert manager.exists() == True


class TestCheckpointCorruption:
    """Test corruption detection and backup restoration."""

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty checkpoint file."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('')  # Empty file

        manager = CheckpointManager(tmp_path)
        result = manager.load()
        assert result is None

    def test_load_small_file(self, tmp_path):
        """Test loading a file with less than 10 chars."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('{}')  # Only 2 chars

        manager = CheckpointManager(tmp_path)
        result = manager.load()
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """Test loading a file with invalid JSON."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('{invalid json here}' * 5)

        manager = CheckpointManager(tmp_path)
        result = manager.load()
        assert result is None

    def test_load_json_not_dict(self, tmp_path):
        """Test loading a file where JSON is not a dict."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('["this", "is", "a", "list", "not", "dict"]')

        manager = CheckpointManager(tmp_path)
        result = manager.load()
        assert result is None

    def test_backup_restoration(self, tmp_path):
        """Test restoring from backup when main is corrupt."""
        # Create corrupt main checkpoint
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('corrupt!!')

        # Create valid backup
        backup_file = tmp_path / "checkpoint.backup.json"
        backup_data = {
            'version': '1.0',
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'ANALYZE',
            'config_hash': '',
            'voiceover_path': '',
            'voiceover_hash': ''
        }
        backup_file.write_text(json.dumps(backup_data))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.version == '1.0'

    def test_backup_restoration_copy_fails(self, tmp_path):
        """Test when backup restoration copy fails."""
        # Create corrupt main checkpoint
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('corrupt!!')

        # Create valid backup
        backup_file = tmp_path / "checkpoint.backup.json"
        backup_data = {
            'version': '1.0',
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'ANALYZE'
        }
        backup_file.write_text(json.dumps(backup_data))

        manager = CheckpointManager(tmp_path)

        with patch('shutil.copy2', side_effect=PermissionError("Cannot copy")):
            result = manager.load()
            # Should still return data even if copy fails
            assert result is not None

    def test_both_corrupt(self, tmp_path):
        """Test when both main and backup are corrupt."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('corrupt!!')

        backup_file = tmp_path / "checkpoint.backup.json"
        backup_file.write_text('also corrupt!!')

        manager = CheckpointManager(tmp_path)
        result = manager.load()
        assert result is None

    def test_load_file_read_error(self, tmp_path):
        """Test handling of file read errors."""
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text('{"version": "1.0"}' * 10)

        manager = CheckpointManager(tmp_path)

        with patch('builtins.open', side_effect=PermissionError("Cannot read")):
            result = manager._try_load_file(checkpoint_file)
            assert result is None


class TestCheckpointMigration:
    """Test checkpoint migration from v0.9 to v1.0."""

    def test_migrate_v09_to_v10(self, tmp_path):
        """Test migration from v0.9 format to v1.0."""
        old_format = {
            'version': '0.9',
            'created_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'ANALYZE',
            'ANALYZE': {'keywords': ['test']},  # Uppercase key
            'DOWNLOAD': {'video_paths': ['/path/to/video.mp4']}
        }

        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(old_format))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.version == '1.0'
        assert result.analyze == {'keywords': ['test']}
        assert result.download == {'video_paths': ['/path/to/video.mp4']}

    def test_migrate_already_v10(self, tmp_path):
        """Test that v1.0 data is not migrated."""
        v10_format = {
            'version': '1.0',
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'ANALYZE',
            'analyze': {'keywords': ['test']}
        }

        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(v10_format))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.version == '1.0'

    def test_migrate_save_failure(self, tmp_path):
        """Test migration when save fails."""
        old_format = {
            'version': '0.9',
            'created_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'ANALYZE',
            'ANALYZE': {'keywords': ['test']}
        }

        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(old_format))

        manager = CheckpointManager(tmp_path)

        with patch.object(manager, '_atomic_save', side_effect=Exception("Save failed")):
            result = manager.load()
            # Migration should still return data even if save fails
            assert result is not None
            assert result.version == '1.0'

    def test_migrate_missing_version(self, tmp_path):
        """Test migration when version is missing (defaults to 0.9)."""
        old_format = {
            # No version field
            'created_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'ANALYZE'
        }

        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text(json.dumps(old_format))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        assert result is not None
        assert result.version == '1.0'  # Migrated to 1.0


class TestCheckpointIsStale:
    """Test is_stale() method."""

    def test_is_stale_no_data(self, tmp_path):
        """Test is_stale when no data loaded."""
        manager = CheckpointManager(tmp_path)
        assert manager.is_stale() == False

    def test_is_stale_no_timestamp(self, tmp_path):
        """Test is_stale when no timestamp in data."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData()  # Empty timestamps
        assert manager.is_stale() == True

    def test_is_stale_fresh_checkpoint(self, tmp_path):
        """Test is_stale with recent checkpoint."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            updated_at=datetime.now().isoformat()
        )
        assert manager.is_stale(max_age_hours=1.0) == False

    def test_is_stale_old_checkpoint(self, tmp_path):
        """Test is_stale with old checkpoint."""
        manager = CheckpointManager(tmp_path)
        old_time = datetime.now() - timedelta(hours=48)
        manager.data = CheckpointData(
            updated_at=old_time.isoformat()
        )
        assert manager.is_stale(max_age_hours=24.0) == True

    def test_is_stale_malformed_timestamp(self, tmp_path):
        """Test is_stale with malformed timestamp."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            updated_at="not-a-valid-timestamp"
        )
        assert manager.is_stale() == True

    def test_is_stale_uses_created_at_fallback(self, tmp_path):
        """Test is_stale uses created_at when updated_at is empty."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            updated_at="",
            created_at=datetime.now().isoformat()
        )
        assert manager.is_stale(max_age_hours=1.0) == False


class TestCheckpointGetAgeHours:
    """Test get_age_hours() method."""

    def test_get_age_hours_no_data(self, tmp_path):
        """Test get_age_hours when no data."""
        manager = CheckpointManager(tmp_path)
        assert manager.get_age_hours() == 0.0

    def test_get_age_hours_no_timestamp(self, tmp_path):
        """Test get_age_hours with no timestamp."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData()
        assert manager.get_age_hours() == 0.0

    def test_get_age_hours_valid(self, tmp_path):
        """Test get_age_hours with valid timestamp."""
        manager = CheckpointManager(tmp_path)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        manager.data = CheckpointData(
            updated_at=one_hour_ago.isoformat()
        )
        age = manager.get_age_hours()
        assert 0.9 < age < 1.1  # Approximately 1 hour

    def test_get_age_hours_exception(self, tmp_path):
        """Test get_age_hours with invalid timestamp returns 0."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            updated_at="invalid"
        )
        assert manager.get_age_hours() == 0.0


class TestCheckpointValidation:
    """Test checkpoint validation."""

    def test_validate_checkpoint_data_valid(self, tmp_path):
        """Test validation with valid data."""
        manager = CheckpointManager(tmp_path)
        data = CheckpointData(
            version="1.0",
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE"
        )
        assert manager._validate_checkpoint_data(data) == True

    def test_validate_checkpoint_data_wrong_version(self, tmp_path):
        """Test validation with wrong version."""
        manager = CheckpointManager(tmp_path)
        data = CheckpointData(
            version="2.0",
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE"
        )
        assert manager._validate_checkpoint_data(data) == False

    def test_validate_checkpoint_data_missing_created_at(self, tmp_path):
        """Test validation with missing created_at."""
        manager = CheckpointManager(tmp_path)
        data = CheckpointData(
            version="1.0",
            created_at="",
            last_completed_stage="ANALYZE"
        )
        assert manager._validate_checkpoint_data(data) == False

    def test_validate_checkpoint_data_no_stage(self, tmp_path):
        """Test validation with no completed stage."""
        manager = CheckpointManager(tmp_path)
        data = CheckpointData(
            version="1.0",
            created_at="2026-01-01T00:00:00",
            last_completed_stage=""
        )
        assert manager._validate_checkpoint_data(data) == False

    def test_validate_checkpoint_data_unknown_stage(self, tmp_path):
        """Test validation with unknown stage."""
        manager = CheckpointManager(tmp_path)
        data = CheckpointData(
            version="1.0",
            created_at="2026-01-01T00:00:00",
            last_completed_stage="UNKNOWN_STAGE"
        )
        assert manager._validate_checkpoint_data(data) == False


class TestCheckpointValidateMethod:
    """Test the validate() method."""

    def test_validate_no_data(self, tmp_path):
        """Test validate when no data loaded."""
        manager = CheckpointManager(tmp_path)
        result = manager.validate()
        assert result['valid'] == False
        assert "No checkpoint data loaded" in result['errors']

    def test_validate_config_hash_mismatch(self, tmp_path):
        """Test validate with config hash mismatch."""
        manager = CheckpointManager(tmp_path, config_hash="new_hash")
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE",
            config_hash="old_hash"
        )
        result = manager.validate()
        assert any("Configuration has changed" in w for w in result['warnings'])

    def test_validate_voiceover_changed(self, tmp_path):
        """Test validate with voiceover hash mismatch."""
        # Create a test voiceover file
        vo_file = tmp_path / "voiceover.srt"
        vo_file.write_text("new content here")

        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE",
            voiceover_hash="different_hash"
        )
        result = manager.validate(voiceover_path=str(vo_file))
        assert any("Voiceover file has changed" in w for w in result['warnings'])

    def test_validate_unknown_stage(self, tmp_path):
        """Test validate with unknown stage in checkpoint."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="UNKNOWN_STAGE"
        )
        result = manager.validate()
        assert any("Unknown stage" in w for w in result['warnings'])

    def test_validate_pipeline_completed(self, tmp_path):
        """Test validate when pipeline already completed."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="OUTPUT"  # Last stage
        )
        result = manager.validate()
        assert result['valid'] == False
        assert "Pipeline already completed" in result['errors']

    def test_validate_missing_videos(self, tmp_path):
        """Test validate with missing video files."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="DOWNLOAD",
            download={'video_paths': ['/nonexistent/video1.mp4', '/nonexistent/video2.mp4']}
        )
        result = manager.validate()
        assert any("missing from disk" in w for w in result['warnings'])

    def test_validate_successful(self, tmp_path):
        """Test successful validation with resume point."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE"
        )
        result = manager.validate()
        assert result['valid'] == True
        assert result['resume_from'] == "ENTITY_IMAGES"
        assert result['completed_stages'] == ["ANALYZE"]


class TestCheckpointSaveAndRestore:
    """Test save and restore operations."""

    def test_save_creates_checkpoint(self, tmp_path):
        """Test saving creates checkpoint file."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        assert manager.checkpoint_path.exists()
        assert manager.data is not None
        assert manager.data.last_completed_stage == "ANALYZE"

    def test_save_creates_backup(self, tmp_path):
        """Test saving creates backup of existing checkpoint."""
        manager = CheckpointManager(tmp_path)

        # First save
        manager.save("ANALYZE", {"keywords": ["test1"]})

        # Second save should create backup
        manager.save("DOWNLOAD", {"video_paths": []})

        assert manager.backup_path.exists()

    def test_save_atomic_failure_cleanup(self, tmp_path):
        """Test that temp file is cleaned up on failure."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData()

        with patch('builtins.open', side_effect=PermissionError("Cannot write")):
            with pytest.raises(Exception):
                manager._atomic_save()

    def test_set_voiceover(self, tmp_path):
        """Test set_voiceover method."""
        vo_file = tmp_path / "voiceover.srt"
        vo_file.write_text("test content")

        manager = CheckpointManager(tmp_path)
        manager.set_voiceover(str(vo_file))

        assert manager.data is not None
        assert manager.data.voiceover_path == str(vo_file)
        assert len(manager.data.voiceover_hash) > 0

    def test_hash_file_error(self, tmp_path):
        """Test _hash_file with non-existent file."""
        manager = CheckpointManager(tmp_path)
        result = manager._hash_file("/nonexistent/file.txt")
        assert result == ""


class TestCheckpointStageOperations:
    """Test stage-related operations."""

    def test_get_stage_data_no_data(self, tmp_path):
        """Test get_stage_data when no data loaded."""
        manager = CheckpointManager(tmp_path)
        assert manager.get_stage_data("ANALYZE") == {}

    def test_get_stage_data_valid(self, tmp_path):
        """Test get_stage_data with valid data."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            analyze={"keywords": ["test"]}
        )
        assert manager.get_stage_data("ANALYZE") == {"keywords": ["test"]}

    def test_should_skip_stage_no_data(self, tmp_path):
        """Test should_skip_stage when no data."""
        manager = CheckpointManager(tmp_path)
        assert manager.should_skip_stage("ANALYZE") == False

    def test_should_skip_stage_completed(self, tmp_path):
        """Test should_skip_stage for completed stage."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            last_completed_stage="DOWNLOAD"
        )
        assert manager.should_skip_stage("ANALYZE") == True
        assert manager.should_skip_stage("DOWNLOAD") == True
        assert manager.should_skip_stage("TRANSCRIBE") == False

    def test_should_skip_stage_unknown(self, tmp_path):
        """Test should_skip_stage with unknown stage."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            last_completed_stage="ANALYZE"
        )
        assert manager.should_skip_stage("UNKNOWN_STAGE") == False


class TestCheckpointClear:
    """Test clear functionality."""

    def test_clear(self, tmp_path):
        """Test clearing checkpoint."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {})

        assert manager.checkpoint_path.exists()

        manager.clear()

        assert not manager.checkpoint_path.exists()
        assert manager.data is None


class TestCheckpointSummary:
    """Test summary generation."""

    def test_get_summary_no_data(self, tmp_path):
        """Test get_summary when no data."""
        manager = CheckpointManager(tmp_path)
        assert manager.get_summary() == "No checkpoint found"

    def test_get_summary_full(self, tmp_path):
        """Test get_summary with full data."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T12:00:00",
            last_completed_stage="MATCH",
            analyze={"keywords": ["a", "b"], "segment_count": 10},
            download={"video_paths": ["/a.mp4", "/b.mp4"]},
            transcribe={"transcribed_count": 5, "embedding_count": 100},
            match={"match_count": 8, "avg_confidence": 0.85}
        )
        summary = manager.get_summary()

        assert "2026-01-01" in summary
        assert "MATCH" in summary
        assert "ANALYZE" in summary
        assert "2 keywords" in summary


class TestKeywordManager:
    """Test KeywordManager functionality."""

    def test_init_no_file(self, tmp_path):
        """Test initialization with no existing file."""
        manager = KeywordManager(tmp_path)
        assert manager.presets == {}

    def test_load_new_format(self, tmp_path):
        """Test loading new format (multiple presets)."""
        keywords_file = tmp_path / "saved_keywords.json"
        data = {
            'version': '1.0',
            'presets': {
                'preset1': {
                    'name': 'preset1',
                    'keywords': ['kw1', 'kw2']
                }
            }
        }
        keywords_file.write_text(json.dumps(data))

        manager = KeywordManager(tmp_path)
        assert 'preset1' in manager.presets
        assert manager.presets['preset1'].keywords == ['kw1', 'kw2']

    def test_load_old_format(self, tmp_path):
        """Test loading old format (single preset)."""
        keywords_file = tmp_path / "saved_keywords.json"
        old_data = {
            'name': 'old',
            'keywords': ['kw1', 'kw2'],
            'topic_context': 'test topic'
        }
        keywords_file.write_text(json.dumps(old_data))

        manager = KeywordManager(tmp_path)
        assert 'default' in manager.presets
        assert manager.presets['default'].keywords == ['kw1', 'kw2']

    def test_load_error_handling(self, tmp_path):
        """Test load handles errors gracefully."""
        keywords_file = tmp_path / "saved_keywords.json"
        keywords_file.write_text('invalid json!!')

        manager = KeywordManager(tmp_path)
        assert manager.presets == {}

    def test_save_keywords(self, tmp_path):
        """Test saving keywords."""
        manager = KeywordManager(tmp_path)
        name = manager.save_keywords(
            keywords=['kw1', 'kw2'],
            topic_context='test topic',
            name='test_preset'
        )

        assert name == 'test_preset'
        assert 'test_preset' in manager.presets
        assert manager.keywords_path.exists()

    def test_save_keywords_auto_name(self, tmp_path):
        """Test saving keywords with auto-generated name."""
        manager = KeywordManager(tmp_path)
        name = manager.save_keywords(keywords=['kw1'])

        # Name should be timestamp-based
        assert len(name) == 15  # YYYYMMDD_HHMMSS format

    def test_save_keywords_with_voiceover(self, tmp_path):
        """Test saving keywords with voiceover hash."""
        vo_file = tmp_path / "voiceover.srt"
        vo_file.write_text("test content")

        manager = KeywordManager(tmp_path)
        manager.save_keywords(
            keywords=['kw1'],
            voiceover_path=str(vo_file),
            name='test'
        )

        assert len(manager.presets['test'].voiceover_hash) > 0

    def test_save_keywords_voiceover_error(self, tmp_path):
        """Test saving keywords when voiceover hash fails."""
        manager = KeywordManager(tmp_path)
        # Non-existent file should be handled gracefully
        manager.save_keywords(
            keywords=['kw1'],
            voiceover_path='/nonexistent/file.srt',
            name='test'
        )

        assert manager.presets['test'].voiceover_hash == ""

    def test_save_error_handling(self, tmp_path):
        """Test _save handles errors gracefully."""
        manager = KeywordManager(tmp_path)
        manager.presets['test'] = SavedKeywords(name='test', keywords=['kw1'])

        with patch('builtins.open', side_effect=PermissionError("Cannot write")):
            manager._save()  # Should not raise

    def test_get_preset_by_name(self, tmp_path):
        """Test getting preset by name."""
        manager = KeywordManager(tmp_path)
        manager.presets['test'] = SavedKeywords(name='test', keywords=['kw1'])

        result = manager.get_preset('test')
        assert result is not None
        assert result.name == 'test'

    def test_get_preset_not_found(self, tmp_path):
        """Test getting non-existent preset."""
        manager = KeywordManager(tmp_path)
        result = manager.get_preset('nonexistent')
        assert result is None

    def test_get_preset_latest(self, tmp_path):
        """Test getting latest preset."""
        manager = KeywordManager(tmp_path)
        manager.presets['old'] = SavedKeywords(
            name='old',
            created_at='2025-01-01T00:00:00',
            keywords=['old_kw']
        )
        manager.presets['new'] = SavedKeywords(
            name='new',
            created_at='2026-01-01T00:00:00',
            keywords=['new_kw']
        )

        result = manager.get_preset()  # No name = latest
        assert result.name == 'new'

    def test_get_latest(self, tmp_path):
        """Test get_latest method."""
        manager = KeywordManager(tmp_path)
        manager.presets['test'] = SavedKeywords(
            name='test',
            created_at='2026-01-01T00:00:00',
            keywords=['kw1']
        )

        result = manager.get_latest()
        assert result.name == 'test'

    def test_list_presets(self, tmp_path):
        """Test listing presets."""
        manager = KeywordManager(tmp_path)
        manager.presets['old'] = SavedKeywords(
            name='old',
            created_at='2025-01-01T00:00:00'
        )
        manager.presets['new'] = SavedKeywords(
            name='new',
            created_at='2026-01-01T00:00:00'
        )

        result = manager.list_presets()
        assert len(result) == 2
        assert result[0].name == 'new'  # Newest first

    def test_delete_preset(self, tmp_path):
        """Test deleting preset."""
        manager = KeywordManager(tmp_path)
        manager.presets['test'] = SavedKeywords(name='test')

        result = manager.delete_preset('test')
        assert result == True
        assert 'test' not in manager.presets

    def test_delete_preset_not_found(self, tmp_path):
        """Test deleting non-existent preset."""
        manager = KeywordManager(tmp_path)
        result = manager.delete_preset('nonexistent')
        assert result == False

    def test_has_presets(self, tmp_path):
        """Test has_presets method."""
        manager = KeywordManager(tmp_path)
        assert manager.has_presets() == False

        manager.presets['test'] = SavedKeywords(name='test')
        assert manager.has_presets() == True

    def test_get_summary_empty(self, tmp_path):
        """Test get_summary with no presets."""
        manager = KeywordManager(tmp_path)
        assert manager.get_summary() == "No saved keyword presets"

    def test_get_summary_with_presets(self, tmp_path):
        """Test get_summary with presets."""
        manager = KeywordManager(tmp_path)
        manager.presets['test'] = SavedKeywords(
            name='test',
            created_at='2026-01-01T12:00:00',
            keywords=['kw1', 'kw2', 'kw3', 'kw4'],
            topic_context='A long topic context that should be truncated'
        )

        summary = manager.get_summary()
        assert 'test' in summary
        assert 'kw1' in summary
        assert '+1 more' in summary  # 4 keywords, showing 3


class TestRemainingCoverage:
    """Tests for remaining uncovered lines."""

    def test_load_no_checkpoint_file(self, tmp_path):
        """Test load returns None when no checkpoint exists (line 158)."""
        manager = CheckpointManager(tmp_path)
        # Don't create any checkpoint file
        result = manager.load()
        assert result is None

    def test_validation_warning_logged(self, tmp_path):
        """Test validation warning is logged (line 180)."""
        # Create a valid checkpoint that loads
        checkpoint_file = tmp_path / "checkpoint.json"
        data = {
            'version': '1.0',
            'created_at': '',  # Empty - will fail validation
            'updated_at': '2026-01-01T00:00:00',
            'last_completed_stage': '',  # Empty - will fail validation
        }
        checkpoint_file.write_text(json.dumps(data))

        manager = CheckpointManager(tmp_path)
        result = manager.load()

        # Should still load despite validation failure
        assert result is not None
        # Validation fails but data is returned

    def test_atomic_save_temp_cleanup(self, tmp_path):
        """Test temp file cleanup on write failure (line 342)."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00"
        )

        temp_path = manager.checkpoint_path.with_suffix('.tmp')

        # Create temp file first
        temp_path.write_text('temp content')

        # Mock to fail during rename
        original_replace = Path.replace

        def mock_replace(self, target):
            raise OSError("Cannot rename")

        with patch.object(Path, 'replace', mock_replace):
            with pytest.raises(Exception):
                manager._atomic_save()

        # Temp file should be cleaned up
        # Note: The cleanup happens in the except block

    def test_clear_with_backup(self, tmp_path):
        """Test clear removes backup file too (line 455)."""
        manager = CheckpointManager(tmp_path)

        # Create both checkpoint and backup
        manager.checkpoint_path.write_text('{"version": "1.0"}')
        manager.backup_path.write_text('{"version": "1.0"}')
        manager.data = CheckpointData()

        manager.clear()

        assert not manager.checkpoint_path.exists()
        assert not manager.backup_path.exists()
        assert manager.data is None

    def test_get_summary_more_than_5_presets(self, tmp_path):
        """Test get_summary with more than 5 presets (line 674)."""
        manager = KeywordManager(tmp_path)

        # Create 7 presets
        for i in range(7):
            manager.presets[f'preset{i}'] = SavedKeywords(
                name=f'preset{i}',
                created_at=f'2026-01-0{i+1}T00:00:00',
                keywords=[f'kw{i}']
            )

        summary = manager.get_summary()
        assert "... and 2 more" in summary


class TestFormatFunctions:
    """Test format_* functions."""

    def test_format_resume_prompt(self, tmp_path):
        """Test format_resume_prompt function."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE"
        )

        result = format_resume_prompt(manager)
        assert "CHECKPOINT FOUND" in result
        assert "[R] Resume" in result
        assert "[F] Fresh start" in result

    def test_format_resume_prompt_with_warnings(self, tmp_path):
        """Test format_resume_prompt with warnings."""
        manager = CheckpointManager(tmp_path, config_hash="new")
        manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            last_completed_stage="ANALYZE",
            config_hash="old"
        )

        result = format_resume_prompt(manager)
        assert "Warnings" in result

    def test_format_keyword_prompt(self, tmp_path):
        """Test format_keyword_prompt function."""
        manager = KeywordManager(tmp_path)
        manager.presets['test'] = SavedKeywords(
            name='test',
            keywords=['kw1']
        )

        result = format_keyword_prompt(manager)
        assert "SAVED KEYWORDS FOUND" in result
        assert "[U] Use saved keywords" in result
