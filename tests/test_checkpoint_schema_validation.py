"""
Tests for checkpoint backup JSON schema validation before restore (US-68-003).

Covers:
- AC1: _validate_checkpoint_schema accepts well-formed backups
- AC2: _validate_checkpoint_schema rejects backups missing required keys
- AC3: _validate_checkpoint_schema rejects backups with invalid stage names
- AC4: _restore_from_backup returns HealerResult.failed on invalid schema
- AC5: _restore_from_backup succeeds on valid schema
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.healers.checkpoint import CheckpointHealer
from src.agents.base import HealerAction
from src.checkpoint import STAGE_ORDER


def _make_healer(tmp_path):
    """Create a CheckpointHealer with a mock config and tmp project dir."""
    config = MagicMock()
    return CheckpointHealer(config, str(tmp_path))


def _valid_checkpoint():
    """Return a well-formed checkpoint dict."""
    return {
        "last_completed_stage": "MATCH",
        "stages": {"ANALYZE": {}, "VIDEO_SEARCH": {}, "CAPTION": {}, "MATCH": {}},
        "timestamp": "2026-02-05T12:00:00",
    }


class TestValidateCheckpointSchemaSuccess:
    """AC1 / AC4: Validation passes for well-formed backup."""

    @pytest.mark.fast
    def test_valid_checkpoint_passes(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = _valid_checkpoint()
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is True
        assert errors == []

    @pytest.mark.fast
    def test_valid_checkpoint_with_all_stages(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": "OUTPUT",
            "stages": {s: {} for s in STAGE_ORDER},
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is True
        assert errors == []

    @pytest.mark.fast
    def test_valid_checkpoint_empty_stages(self, tmp_path):
        """Empty stages dict is valid — no invalid stage names."""
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": "ANALYZE",
            "stages": {},
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is True

    @pytest.mark.fast
    def test_valid_checkpoint_with_none_last_stage(self, tmp_path):
        """last_completed_stage=None is valid (fresh checkpoint)."""
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": None,
            "stages": {},
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is True


class TestValidateCheckpointSchemaMissingKeys:
    """AC2 / AC5: Validation fails when required keys are missing."""

    @pytest.mark.fast
    def test_missing_last_completed_stage(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = {"stages": {}, "timestamp": "2026-02-05T12:00:00"}
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("last_completed_stage" in e for e in errors)

    @pytest.mark.fast
    def test_missing_stages(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = {"last_completed_stage": "MATCH", "timestamp": "2026-02-05T12:00:00"}
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("stages" in e for e in errors)

    @pytest.mark.fast
    def test_missing_timestamp(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = {"last_completed_stage": "MATCH", "stages": {}}
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("timestamp" in e for e in errors)

    @pytest.mark.fast
    def test_empty_dict_missing_all_keys(self, tmp_path):
        healer = _make_healer(tmp_path)
        is_valid, errors = healer._validate_checkpoint_schema({})
        assert is_valid is False
        assert len(errors) >= 1
        # All three keys should be mentioned
        error_text = " ".join(errors)
        for key in ("last_completed_stage", "stages", "timestamp"):
            assert key in error_text


class TestValidateCheckpointSchemaInvalidStages:
    """AC3 / AC6: Validation fails when backup has invalid stage names."""

    @pytest.mark.fast
    def test_invalid_last_completed_stage(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": "NONEXISTENT_STAGE",
            "stages": {},
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("NONEXISTENT_STAGE" in e for e in errors)

    @pytest.mark.fast
    def test_invalid_stage_name_in_stages_dict(self, tmp_path):
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": "MATCH",
            "stages": {"ANALYZE": {}, "BOGUS_STAGE": {}},
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("BOGUS_STAGE" in e for e in errors)

    @pytest.mark.fast
    def test_legacy_stage_name_rejected(self, tmp_path):
        """Legacy stage names like DOWNLOAD should be rejected."""
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": "MATCH",
            "stages": {"DOWNLOAD": {}, "MATCH": {}},
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("DOWNLOAD" in e for e in errors)

    @pytest.mark.fast
    def test_stages_not_dict_rejected(self, tmp_path):
        """stages must be a dict, not a list or other type."""
        healer = _make_healer(tmp_path)
        data = {
            "last_completed_stage": "MATCH",
            "stages": ["ANALYZE", "MATCH"],
            "timestamp": "2026-02-05T12:00:00",
        }
        is_valid, errors = healer._validate_checkpoint_schema(data)
        assert is_valid is False
        assert any("dict" in e for e in errors)


class TestRestoreFromBackupWithValidation:
    """Integration: _restore_from_backup uses schema validation."""

    @pytest.mark.fast
    def test_restore_succeeds_with_valid_backup(self, tmp_path):
        healer = _make_healer(tmp_path)
        state = MagicMock()

        # Create corrupted primary
        (tmp_path / "checkpoint.json").write_text("{bad json")
        # Create valid backup
        (tmp_path / "checkpoint.backup.json").write_text(
            json.dumps(_valid_checkpoint())
        )

        result = healer._restore_from_backup(json.JSONDecodeError("", "", 0), state)
        assert result.success is True
        assert result.action == HealerAction.RESTORE

    @pytest.mark.fast
    def test_restore_fails_with_missing_keys_backup(self, tmp_path):
        healer = _make_healer(tmp_path)
        state = MagicMock()

        (tmp_path / "checkpoint.json").write_text("{bad json")
        # Backup missing required keys
        (tmp_path / "checkpoint.backup.json").write_text(
            json.dumps({"some_random_key": True})
        )

        result = healer._restore_from_backup(json.JSONDecodeError("", "", 0), state)
        assert result.success is False
        assert result.action == HealerAction.ABORT
        assert "validation_errors" in result.details

    @pytest.mark.fast
    def test_restore_fails_with_invalid_stages_backup(self, tmp_path):
        healer = _make_healer(tmp_path)
        state = MagicMock()

        (tmp_path / "checkpoint.json").write_text("{bad json")
        (tmp_path / "checkpoint.backup.json").write_text(
            json.dumps({
                "last_completed_stage": "FAKE_STAGE",
                "stages": {"ALSO_FAKE": {}},
                "timestamp": "2026-02-05T12:00:00",
            })
        )

        result = healer._restore_from_backup(json.JSONDecodeError("", "", 0), state)
        assert result.success is False
        assert "FAKE_STAGE" in result.message or "ALSO_FAKE" in result.message
