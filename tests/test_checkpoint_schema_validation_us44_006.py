"""
Tests for US-44-006: Checkpoint data schema validation to prevent silent corruption.

Covers:
- AC1: from_dict() validates stage data dicts contain expected types
- AC2: validate() method checks internal consistency (stages before last_completed have data)
- AC3: load() calls validate() and logs warnings for inconsistencies
- AC4: save() validates stage_key maps to actual CheckpointData field
- AC5: Loading checkpoint with missing intermediate stage data produces warning
"""

import json
import logging
import pytest
from datetime import datetime
from pathlib import Path

from src.checkpoint import (
    CheckpointManager,
    CheckpointData,
    STAGE_ORDER,
    STAGE_FIELD_MAP,
    CURRENT_CHECKPOINT_VERSION,
)


# ============================================================================
# AC1: from_dict() validates stage data types
# ============================================================================

@pytest.mark.fast
class TestFromDictValidation:
    """AC1: from_dict() validates that stage data dicts contain expected keys."""

    def test_from_dict_accepts_valid_stage_dicts(self):
        """Valid stage data (dicts) are accepted without modification."""
        raw = {
            'version': '2.0',
            'created_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'MATCH',
            'analyze': {'keywords': ['test']},
            'match': {'match_count': 5},
        }
        result = CheckpointData.from_dict(raw)
        assert result.analyze == {'keywords': ['test']}
        assert result.match == {'match_count': 5}

    def test_from_dict_resets_non_dict_stage_data_to_empty(self):
        """Stage data that is not a dict gets reset to empty dict."""
        raw = {
            'version': '2.0',
            'created_at': '2026-01-01T00:00:00',
            'last_completed_stage': 'MATCH',
            'analyze': "this should be a dict",
            'match': ['also', 'wrong'],
        }
        result = CheckpointData.from_dict(raw)
        assert result.analyze == {}
        assert result.match == {}

    def test_from_dict_handles_none_stage_data(self):
        """Stage data that is None gets reset to empty dict."""
        raw = {
            'version': '2.0',
            'created_at': '2026-01-01T00:00:00',
            'analyze': None,
        }
        result = CheckpointData.from_dict(raw)
        # None is not a dict, so it should be reset
        assert result.analyze == {}

    def test_from_dict_logs_warning_for_invalid_type(self, caplog):
        """from_dict logs a warning when resetting invalid stage data."""
        with caplog.at_level(logging.WARNING):
            raw = {
                'version': '2.0',
                'analyze': "invalid_string",
            }
            CheckpointData.from_dict(raw)
        assert "analyze" in caplog.text
        assert "expected dict" in caplog.text

    def test_from_dict_still_filters_unknown_fields(self):
        """Unknown fields are still filtered out."""
        raw = {
            'version': '2.0',
            'unknown_field': 'ignored',
            'analyze': {'keywords': ['ok']},
        }
        result = CheckpointData.from_dict(raw)
        assert not hasattr(result, 'unknown_field')
        assert result.analyze == {'keywords': ['ok']}

    def test_from_dict_integer_stage_data_reset(self):
        """Integer stage data is reset to empty dict."""
        raw = {
            'version': '2.0',
            'caption': 42,
        }
        result = CheckpointData.from_dict(raw)
        assert result.caption == {}


# ============================================================================
# AC2: validate() method checks internal consistency
# ============================================================================

@pytest.mark.fast
class TestCheckpointDataValidate:
    """AC2: validate() checks that stages before last_completed have data."""

    def test_validate_empty_stage_returns_no_warnings(self):
        """No warnings when last_completed_stage is empty."""
        data = CheckpointData()
        warnings = data.validate()
        assert warnings == []

    def test_validate_all_stages_populated_no_warnings(self):
        """No warnings when all stages up to last_completed have data."""
        data = CheckpointData(
            last_completed_stage='MATCH',
            analyze={'keywords': ['test']},
            video_search={'video_ids': ['vid1']},
            caption={'caption_count': 3},
            match={'match_count': 5},
        )
        warnings = data.validate()
        assert warnings == []

    def test_validate_missing_intermediate_stage_warns(self):
        """Warning when an intermediate stage has no data."""
        data = CheckpointData(
            last_completed_stage='MATCH',
            analyze={'keywords': ['test']},
            # video_search is empty (default)
            caption={'caption_count': 3},
            match={'match_count': 5},
        )
        warnings = data.validate()
        assert len(warnings) == 1
        assert 'VIDEO_SEARCH' in warnings[0]
        assert 'no data' in warnings[0]

    def test_validate_multiple_missing_stages_warn(self):
        """Multiple warnings when several intermediate stages have no data."""
        data = CheckpointData(
            last_completed_stage='MATCH',
            # analyze empty
            # video_search empty
            # caption empty
            match={'match_count': 5},
        )
        warnings = data.validate()
        assert len(warnings) == 3
        stage_names = [w for w in warnings]
        assert any('ANALYZE' in w for w in stage_names)
        assert any('VIDEO_SEARCH' in w for w in stage_names)
        assert any('CAPTION' in w for w in stage_names)

    def test_validate_first_stage_only_no_warnings(self):
        """No warnings when last_completed is first stage and it has data."""
        data = CheckpointData(
            last_completed_stage='ANALYZE',
            analyze={'keywords': ['test']},
        )
        warnings = data.validate()
        assert warnings == []

    def test_validate_first_stage_empty_warns(self):
        """Warning when first stage is completed but has no data."""
        data = CheckpointData(
            last_completed_stage='ANALYZE',
            # analyze is empty (default)
        )
        warnings = data.validate()
        assert len(warnings) == 1
        assert 'ANALYZE' in warnings[0]

    def test_validate_unknown_stage_warns(self):
        """Warning for unknown last_completed_stage."""
        data = CheckpointData(
            last_completed_stage='UNKNOWN_STAGE',
        )
        warnings = data.validate()
        assert len(warnings) == 1
        assert 'Unknown' in warnings[0]

    def test_validate_output_stage_no_field_check(self):
        """OUTPUT stage has no field in CheckpointData, should not warn about it."""
        data = CheckpointData(
            last_completed_stage='OUTPUT',
            analyze={'keywords': ['test']},
            video_search={'video_ids': ['vid1']},
            caption={'caption_count': 3},
            match={'match_count': 5},
            iterative_match={'passes': 1},
            download_segments={'count': 3},
        )
        warnings = data.validate()
        assert warnings == []

    def test_validate_download_segments_stage(self):
        """Validate with last_completed at DOWNLOAD_SEGMENTS."""
        data = CheckpointData(
            last_completed_stage='DOWNLOAD_SEGMENTS',
            analyze={'keywords': ['test']},
            video_search={'video_ids': ['vid1']},
            caption={'caption_count': 3},
            match={'match_count': 5},
            iterative_match={'passes': 1},
            download_segments={'count': 3},
        )
        warnings = data.validate()
        assert warnings == []


# ============================================================================
# AC3: load() calls validate() and logs warnings
# ============================================================================

@pytest.mark.fast
class TestLoadCallsValidate:
    """AC3: CheckpointManager.load() calls validate() and logs warnings."""

    def test_load_logs_consistency_warning_for_missing_stage_data(self, tmp_path, caplog):
        """Loading a checkpoint with missing intermediate stage data logs warning."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Checkpoint where last_completed is MATCH but analyze data is empty
        checkpoint_data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "analyze": {},  # Empty — should trigger consistency warning
            "video_search": {"video_ids": ["vid1"]},
            "caption": {"caption_count": 3},
            "match": {"match_count": 5},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.WARNING):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        assert "Checkpoint consistency" in caplog.text
        assert "ANALYZE" in caplog.text

    def test_load_no_consistency_warning_for_valid_checkpoint(self, tmp_path, caplog):
        """Loading a valid checkpoint does not log consistency warnings."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "ANALYZE",
            "analyze": {"keywords": ["test"]},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.WARNING):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        assert "Checkpoint consistency" not in caplog.text

    def test_load_multiple_consistency_warnings(self, tmp_path, caplog):
        """Loading checkpoint with multiple missing stages logs multiple warnings."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            # All stages empty except match
            "analyze": {},
            "video_search": {},
            "caption": {},
            "match": {"match_count": 5},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.WARNING):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        # Should have warnings for ANALYZE, VIDEO_SEARCH, CAPTION
        consistency_warnings = [r for r in caplog.records if "Checkpoint consistency" in r.message]
        assert len(consistency_warnings) == 3


# ============================================================================
# AC4: save() validates stage_key maps to a real field
# ============================================================================

@pytest.mark.fast
class TestSaveStageKeyValidation:
    """AC4: save() validates stage_key maps to actual CheckpointData field."""

    def test_save_valid_stage_persists_data(self, tmp_path):
        """save() with a valid stage name persists data correctly."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        assert manager.data.analyze == {"keywords": ["test"]}
        assert manager.data.last_completed_stage == "ANALYZE"

    def test_save_unknown_stage_logs_warning(self, tmp_path, caplog):
        """save() with an unknown stage name logs warning and doesn't crash."""
        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.WARNING):
            manager.save("NONEXISTENT_STAGE", {"data": "test"})

        assert "does not map to a CheckpointData field" in caplog.text
        assert "nonexistent_stage" in caplog.text
        # last_completed_stage is still set (that's the stage name, not the field)
        assert manager.data.last_completed_stage == "NONEXISTENT_STAGE"

    def test_save_output_stage_without_data_is_fine(self, tmp_path):
        """save() for OUTPUT with no data is fine (no field needed)."""
        manager = CheckpointManager(tmp_path)
        # OUTPUT has no field in CheckpointData but save with no data is OK
        manager.save("OUTPUT")
        assert manager.data.last_completed_stage == "OUTPUT"

    def test_save_output_stage_with_data_no_warning(self, tmp_path, caplog):
        """save() for terminal OUTPUT with data logs debug, not warning."""
        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.DEBUG):
            manager.save("OUTPUT", {"some": "data"})

        assert "does not map to a CheckpointData field" not in caplog.text
        assert "terminal" in caplog.text
        assert manager.data.last_completed_stage == "OUTPUT"

    def test_save_intermediate_unknown_stage_warns(self, tmp_path, caplog):
        """save_intermediate() with unknown stage logs warning."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(created_at=datetime.now().isoformat())

        with caplog.at_level(logging.WARNING):
            manager.save_intermediate("FAKE_STAGE", {"data": "test"})

        assert "does not map to a CheckpointData field" in caplog.text
        assert "fake_stage" in caplog.text

    def test_save_intermediate_valid_stage_persists(self, tmp_path):
        """save_intermediate() with valid stage persists data without changing last_completed."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["initial"]})

        manager.save_intermediate("CAPTION", {"caption_count": 5})

        assert manager.data.caption == {"caption_count": 5}
        assert manager.data.last_completed_stage == "ANALYZE"  # Unchanged


# ============================================================================
# AC5: Loading checkpoint with missing intermediate stage data warns
# ============================================================================

@pytest.mark.fast
class TestMissingIntermediateStageDataWarning:
    """AC5: Loading checkpoint with missing intermediate stage data produces warning."""

    def test_checkpoint_missing_early_stage_warns_on_load(self, tmp_path, caplog):
        """Checkpoint at DOWNLOAD_SEGMENTS but missing ANALYZE data warns."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "DOWNLOAD_SEGMENTS",
            "analyze": {},  # Missing!
            "video_search": {"video_ids": ["vid1"]},
            "caption": {"caption_count": 3},
            "match": {"match_count": 5},
            "iterative_match": {"passes": 1},
            "download_segments": {"count": 3},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.WARNING):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        assert "ANALYZE" in caplog.text
        assert "no data" in caplog.text

    def test_checkpoint_all_empty_up_to_last_completed_warns(self, tmp_path, caplog):
        """Checkpoint where all stages before last_completed are empty warns for each."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": "2.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "CAPTION",
            "analyze": {},
            "video_search": {},
            "caption": {},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.WARNING):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        consistency_msgs = [r.message for r in caplog.records if "Checkpoint consistency" in r.message]
        assert len(consistency_msgs) == 3  # ANALYZE, VIDEO_SEARCH, CAPTION


# ============================================================================
# STAGE_FIELD_MAP correctness
# ============================================================================

@pytest.mark.fast
class TestStageFieldMap:
    """Verify STAGE_FIELD_MAP covers all expected stages."""

    def test_stage_field_map_covers_non_output_stages(self):
        """STAGE_FIELD_MAP has entries for all stages except OUTPUT."""
        for stage in STAGE_ORDER:
            if stage == "OUTPUT":
                assert stage not in STAGE_FIELD_MAP
            else:
                assert stage in STAGE_FIELD_MAP, f"Missing STAGE_FIELD_MAP entry for {stage}"

    def test_stage_field_map_values_are_dataclass_fields(self):
        """All STAGE_FIELD_MAP values correspond to CheckpointData fields."""
        for stage, field_name in STAGE_FIELD_MAP.items():
            assert field_name in CheckpointData.__dataclass_fields__, \
                f"STAGE_FIELD_MAP[{stage}]={field_name} not in CheckpointData fields"


# ============================================================================
# US-57-003: Version constant — no false-positive warning for current version
# ============================================================================

@pytest.mark.fast
class TestCheckpointVersionConstant:
    """US-57-003: Centralized version constant eliminates false-positive warnings."""

    def test_current_version_checkpoint_no_version_warning(self, tmp_path, caplog):
        """Checkpoint with CURRENT_CHECKPOINT_VERSION triggers no version warning."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": CURRENT_CHECKPOINT_VERSION,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "ANALYZE",
            "analyze": {"keywords": ["test"]},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.DEBUG):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        # No version-related warnings or debug messages about incompatibility
        version_msgs = [
            r for r in caplog.records
            if "version" in r.message.lower() and "may not be" in r.message.lower()
            or "differs from current" in r.message.lower()
        ]
        assert len(version_msgs) == 0, (
            f"Unexpected version warning for current version: {[r.message for r in version_msgs]}"
        )

    def test_old_version_checkpoint_logs_migration_notice(self, tmp_path, caplog):
        """Checkpoint with version '1.0' logs a migration/upgrade notice."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "ANALYZE",
            "analyze": {"keywords": ["test"]},
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        with caplog.at_level(logging.INFO):
            manager = CheckpointManager(project_dir)
            result = manager.load()

        assert result is not None
        # Should log migration notice (not a false alarm)
        migration_msgs = [
            r for r in caplog.records
            if "migrat" in r.message.lower()
        ]
        assert len(migration_msgs) > 0, (
            f"Expected migration notice for v1.0 checkpoint, got: {[r.message for r in caplog.records]}"
        )

    def test_checkpoint_data_default_version_matches_constant(self):
        """CheckpointData default version field matches CURRENT_CHECKPOINT_VERSION."""
        data = CheckpointData()
        assert data.version == CURRENT_CHECKPOINT_VERSION

    def test_save_writes_current_version(self, tmp_path):
        """save() writes checkpoint with CURRENT_CHECKPOINT_VERSION."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Read raw JSON to verify version field
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        assert raw["version"] == CURRENT_CHECKPOINT_VERSION
