"""Smoke tests for checkpoint auto-rollback on missing critical stage data."""

import pytest
import json
import gzip
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.checkpoint import (
    CheckpointManager,
    CheckpointData,
    STAGE_ORDER,
    STAGE_FIELD_MAP,
)


def _write_checkpoint(path: Path, data: dict):
    """Write a checkpoint dict as gzip JSON (matches real pipeline format)."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def _make_checkpoint_dict(last_completed: str, filled_stages: set[str]) -> dict:
    """Build a minimal checkpoint dict with given stages filled."""
    d = {"last_completed_stage": last_completed, "version": "3.0"}
    for stage in STAGE_ORDER:
        field = STAGE_FIELD_MAP.get(stage)
        if field:
            d[field] = {"_ok": True} if stage in filled_stages else {}
    return d


class TestCheckpointAutoRollback:
    """Verify the auto-rollback fires for missing critical stages."""

    def test_rollback_on_empty_video_search(self, tmp_path):
        """VIDEO_SEARCH empty + last_completed=CAPTION → rolls back before VIDEO_SEARCH."""
        cp_path = tmp_path / "checkpoint.json"
        data = _make_checkpoint_dict(
            "CAPTION",
            filled_stages={"ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS",
                           "STOCK_FOOTAGE", "CAPTION"},
        )
        _write_checkpoint(cp_path, data)

        mgr = CheckpointManager(project_dir=tmp_path)
        mgr.load()

        assert mgr.data.last_completed_stage == "GENERATED_IMAGES"

    def test_rollback_on_empty_match(self, tmp_path):
        """MATCH empty + last_completed=DOWNLOAD_SEGMENTS → rolls back before MATCH."""
        cp_path = tmp_path / "checkpoint.json"
        data = _make_checkpoint_dict(
            "DOWNLOAD_SEGMENTS",
            filled_stages={"ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS",
                           "STOCK_FOOTAGE", "VIDEO_SEARCH", "CAPTION",
                           "DOWNLOAD_SEGMENTS"},
        )
        _write_checkpoint(cp_path, data)

        mgr = CheckpointManager(project_dir=tmp_path)
        mgr.load()

        assert mgr.data.last_completed_stage == "CAPTION"

    def test_rollback_on_empty_caption(self, tmp_path):
        """CAPTION empty + last_completed=MATCH → rolls back before CAPTION."""
        cp_path = tmp_path / "checkpoint.json"
        data = _make_checkpoint_dict(
            "MATCH",
            filled_stages={"ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS",
                           "STOCK_FOOTAGE", "VIDEO_SEARCH", "MATCH"},
        )
        _write_checkpoint(cp_path, data)

        mgr = CheckpointManager(project_dir=tmp_path)
        mgr.load()

        assert mgr.data.last_completed_stage == "VIDEO_SEARCH"

    def test_no_rollback_when_optional_stage_empty(self, tmp_path):
        """GENERATED_IMAGES empty (not critical) → no rollback."""
        cp_path = tmp_path / "checkpoint.json"
        data = _make_checkpoint_dict(
            "CAPTION",
            filled_stages={"ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS",
                           "STOCK_FOOTAGE", "VIDEO_SEARCH", "CAPTION"},
            # GENERATED_IMAGES deliberately missing — it's optional
        )
        _write_checkpoint(cp_path, data)

        mgr = CheckpointManager(project_dir=tmp_path)
        mgr.load()

        assert mgr.data.last_completed_stage == "CAPTION"

    def test_no_rollback_when_all_stages_filled(self, tmp_path):
        """All stages have data → no rollback."""
        cp_path = tmp_path / "checkpoint.json"
        data = _make_checkpoint_dict(
            "DOWNLOAD_SEGMENTS",
            filled_stages={"ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS",
                           "STOCK_FOOTAGE", "GENERATED_IMAGES", "VIDEO_SEARCH",
                           "CAPTION", "MATCH", "DOWNLOAD_SEGMENTS"},
        )
        _write_checkpoint(cp_path, data)

        mgr = CheckpointManager(project_dir=tmp_path)
        mgr.load()

        assert mgr.data.last_completed_stage == "DOWNLOAD_SEGMENTS"

    def test_rollback_on_empty_download_segments(self, tmp_path):
        """DOWNLOAD_SEGMENTS empty + last_completed=OUTPUT → rolls back."""
        cp_path = tmp_path / "checkpoint.json"
        data = _make_checkpoint_dict(
            "OUTPUT",
            filled_stages={"ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS",
                           "STOCK_FOOTAGE", "VIDEO_SEARCH", "CAPTION", "MATCH"},
        )
        _write_checkpoint(cp_path, data)

        mgr = CheckpointManager(project_dir=tmp_path)
        mgr.load()

        # ITERATIVE_MATCH comes between MATCH and DOWNLOAD_SEGMENTS
        assert mgr.data.last_completed_stage == "ITERATIVE_MATCH"
