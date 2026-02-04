"""
Tests for US-57-010: STAGE_FIELD_MAP auto-generation from CheckpointData dataclass.

Verifies:
- AC1: STAGE_FIELD_MAP is auto-generated (not hardcoded) from CheckpointData fields
- AC2: Naming convention maps stage names to lowercased field names
- AC3: Module-load validation catches missing CheckpointData fields
- AC4: STAGE_FIELD_MAP covers all non-terminal STAGE_ORDER entries
- AC5: Adding a stage to STAGE_ORDER without a CheckpointData field raises ImportError
"""

import pytest
from unittest.mock import patch
from dataclasses import dataclass, field
from typing import Dict, Any

from src.checkpoint import (
    STAGE_ORDER,
    STAGE_FIELD_MAP,
    CheckpointData,
    _STAGE_FIELD_MAP_TERMINAL,
    _build_stage_field_map,
    _validate_stage_field_map,
)


class TestStageFieldMapCoversStageOrder:
    """AC4: STAGE_FIELD_MAP contains entries for all non-terminal stages in STAGE_ORDER."""

    def test_all_non_terminal_stages_have_field_map_entry(self):
        """Every STAGE_ORDER entry (except OUTPUT) must be in STAGE_FIELD_MAP."""
        for stage in STAGE_ORDER:
            if stage in _STAGE_FIELD_MAP_TERMINAL:
                continue
            assert stage in STAGE_FIELD_MAP, (
                f"Stage '{stage}' is in STAGE_ORDER but missing from STAGE_FIELD_MAP. "
                f"Add a Dict[str, Any] field named '{stage.lower()}' to CheckpointData."
            )

    def test_no_extra_entries_in_field_map(self):
        """STAGE_FIELD_MAP should not contain stages that aren't in STAGE_ORDER."""
        stage_set = set(STAGE_ORDER)
        for stage in STAGE_FIELD_MAP:
            assert stage in stage_set, (
                f"STAGE_FIELD_MAP contains '{stage}' which is not in STAGE_ORDER"
            )

    def test_terminal_stages_excluded(self):
        """Terminal stages (OUTPUT) must NOT be in STAGE_FIELD_MAP."""
        for stage in _STAGE_FIELD_MAP_TERMINAL:
            assert stage not in STAGE_FIELD_MAP, (
                f"Terminal stage '{stage}' should not be in STAGE_FIELD_MAP"
            )


class TestStageFieldMapNamingConvention:
    """AC2: Stage names map to lowercased CheckpointData field names."""

    def test_field_names_are_lowercased_stage_names(self):
        """Each STAGE_FIELD_MAP value must be the lowercased key."""
        for stage, field_name in STAGE_FIELD_MAP.items():
            assert field_name == stage.lower(), (
                f"STAGE_FIELD_MAP['{stage}'] = '{field_name}', "
                f"expected '{stage.lower()}'"
            )

    def test_checkpoint_data_has_matching_fields(self):
        """Each STAGE_FIELD_MAP field must exist on CheckpointData."""
        for stage, field_name in STAGE_FIELD_MAP.items():
            assert field_name in CheckpointData.__dataclass_fields__, (
                f"CheckpointData missing field '{field_name}' for stage '{stage}'"
            )


class TestAutoGenerationMechanism:
    """AC1: STAGE_FIELD_MAP is auto-generated, not hardcoded."""

    def test_build_returns_dict(self):
        """_build_stage_field_map returns a dict."""
        result = _build_stage_field_map()
        assert isinstance(result, dict)

    def test_build_matches_module_level_map(self):
        """Calling _build_stage_field_map() produces the same result as STAGE_FIELD_MAP."""
        assert _build_stage_field_map() == STAGE_FIELD_MAP

    def test_build_excludes_terminal_stages(self):
        """Terminal stages are excluded from the built map."""
        result = _build_stage_field_map()
        for stage in _STAGE_FIELD_MAP_TERMINAL:
            assert stage not in result


class TestMissingFieldRaisesError:
    """AC3/AC5: Adding a stage to STAGE_ORDER without a CheckpointData field raises ImportError."""

    def test_missing_field_raises_import_error(self):
        """_validate_stage_field_map raises ImportError when a stage has no CheckpointData field."""
        fake_stage = "NONEXISTENT_STAGE"
        with patch("src.checkpoint.STAGE_ORDER", STAGE_ORDER + [fake_stage]):
            with pytest.raises(ImportError, match="STAGE_ORDER / CheckpointData drift"):
                _validate_stage_field_map()

    def test_error_message_contains_missing_stage(self):
        """The ImportError message identifies which stage is missing."""
        fake_stage = "BRAND_NEW_STAGE"
        with patch("src.checkpoint.STAGE_ORDER", STAGE_ORDER + [fake_stage]):
            with pytest.raises(ImportError, match=fake_stage):
                _validate_stage_field_map()

    def test_valid_config_does_not_raise(self):
        """Current STAGE_ORDER + CheckpointData should not raise."""
        # Should not raise — this is the normal case
        _validate_stage_field_map()
