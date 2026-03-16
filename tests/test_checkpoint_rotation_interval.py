"""
Tests for checkpoint backup rotation interval gating (US-85-003).

Covers:
- AC4: Rapid successive intermediate saves do NOT trigger backup rotation
        when min_rotation_interval_seconds has not elapsed.
- AC5: Backup rotation IS triggered when a stage boundary is crossed
        regardless of the rotation interval.
- Config: min_rotation_interval_seconds is read from pipeline config.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData


class _FakePipelineConfig:
    """Minimal pipeline config stub for testing."""
    def __init__(self, backup_count=3, min_rotation_interval=60):
        self.checkpoint_backup_count = backup_count
        self.min_rotation_interval_seconds = min_rotation_interval


class _FakeConfig:
    """Minimal config stub for testing."""
    def __init__(self, backup_count=3, min_rotation_interval=60):
        self.pipeline = _FakePipelineConfig(backup_count, min_rotation_interval)


class TestRotationIntervalSkipsOnRapidIntermediateSaves:
    """AC4: Rapid intermediate saves skip backup rotation when interval not elapsed."""

    @pytest.mark.fast
    def test_rapid_intermediate_saves_skip_rotation(self, tmp_path):
        """Multiple save_intermediate calls within interval should rotate only once."""
        config = _FakeConfig(backup_count=3, min_rotation_interval=300)
        manager = CheckpointManager(tmp_path, config=config)

        # Two stage saves to create checkpoint + backup
        # (first save creates checkpoint.json; second rotates it to backup)
        manager.save("ANALYZE", {"step": 0})
        manager.save("VIDEO_SEARCH", {"step": 1})
        assert (tmp_path / "checkpoint.json").exists()

        # Record backup state after stage saves
        backup_path = tmp_path / "checkpoint.backup.json"
        assert backup_path.exists()
        backup_mtime_after_stage = backup_path.stat().st_mtime

        # Now do rapid intermediate saves — rotation should be skipped
        # because interval (300s) has not elapsed since the last stage save
        for i in range(5):
            manager.save_intermediate("DOWNLOAD_SEGMENTS", {"downloaded": i})

        # Backup file should NOT have been updated (rotation skipped)
        assert backup_path.stat().st_mtime == backup_mtime_after_stage

        # But checkpoint.json itself should be updated (save still happens)
        with open(tmp_path / "checkpoint.json", 'r') as f:
            cp = json.load(f)
        assert cp.get("download_segments", {}).get("downloaded") == 4

    @pytest.mark.fast
    def test_intermediate_saves_rotate_after_interval_elapses(self, tmp_path):
        """Intermediate save should rotate when interval has elapsed."""
        # Use a tiny interval so time.time() drift naturally exceeds it
        config = _FakeConfig(backup_count=3, min_rotation_interval=0)
        manager = CheckpointManager(tmp_path, config=config)

        # Two stage saves to create checkpoint + backup
        manager.save("ANALYZE", {"step": 0})
        manager.save("VIDEO_SEARCH", {"step": 1})
        backup_path = tmp_path / "checkpoint.backup.json"
        assert backup_path.exists()

        # With interval=0, every intermediate save should rotate
        backup_mtime_before = backup_path.stat().st_mtime
        # Small sleep to ensure mtime changes
        time.sleep(0.05)
        manager.save_intermediate("DOWNLOAD_SEGMENTS", {"downloaded": 1})

        # Backup should have been rotated (updated)
        assert backup_path.stat().st_mtime > backup_mtime_before


class TestStageBoundaryAlwaysRotates:
    """AC5: Stage completion saves always rotate regardless of interval."""

    @pytest.mark.fast
    def test_stage_save_rotates_even_within_interval(self, tmp_path):
        """save() should always rotate backups even if interval hasn't elapsed."""
        config = _FakeConfig(backup_count=3, min_rotation_interval=9999)
        manager = CheckpointManager(tmp_path, config=config)

        # First save creates checkpoint
        manager.save("ANALYZE", {"step": 0})
        backup_path = tmp_path / "checkpoint.backup.json"

        # Record state
        if backup_path.exists():
            mtime_1 = backup_path.stat().st_mtime
        else:
            mtime_1 = 0

        # Second stage save — should rotate despite huge interval
        time.sleep(0.05)
        manager.save("VIDEO_SEARCH", {"step": 1})

        assert backup_path.exists()
        assert backup_path.stat().st_mtime > mtime_1

    @pytest.mark.fast
    def test_stage_save_rotates_with_rotate_backups_call_count(self, tmp_path):
        """Verify _rotate_backups is called for every save() but not every save_intermediate()."""
        config = _FakeConfig(backup_count=3, min_rotation_interval=9999)
        manager = CheckpointManager(tmp_path, config=config)

        with patch.object(manager, '_rotate_backups', wraps=manager._rotate_backups) as mock_rotate:
            # Stage save — should call _rotate_backups
            manager.save("ANALYZE", {"step": 0})
            assert mock_rotate.call_count == 1

            # Intermediate saves — should NOT call _rotate_backups (interval=9999)
            manager.save_intermediate("DOWNLOAD_SEGMENTS", {"downloaded": 0})
            manager.save_intermediate("DOWNLOAD_SEGMENTS", {"downloaded": 1})
            manager.save_intermediate("DOWNLOAD_SEGMENTS", {"downloaded": 2})
            assert mock_rotate.call_count == 1  # Still 1

            # Another stage save — should call _rotate_backups again
            manager.save("VIDEO_SEARCH", {"step": 1})
            assert mock_rotate.call_count == 2


class TestRotationIntervalConfig:
    """Test that min_rotation_interval_seconds is correctly read from config."""

    @pytest.mark.fast
    def test_default_interval_without_config(self, tmp_path):
        """Without config, default interval should be 60."""
        manager = CheckpointManager(tmp_path)
        assert manager._min_rotation_interval == 60

    @pytest.mark.fast
    def test_custom_interval_from_config(self, tmp_path):
        """Custom interval from config should be respected."""
        config = _FakeConfig(min_rotation_interval=120)
        manager = CheckpointManager(tmp_path, config=config)
        assert manager._min_rotation_interval == 120

    @pytest.mark.fast
    def test_last_rotation_time_initialized_to_zero(self, tmp_path):
        """_last_rotation_time should start at 0 so first save always rotates."""
        manager = CheckpointManager(tmp_path)
        assert manager._last_rotation_time == 0.0
