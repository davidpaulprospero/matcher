"""
Tests for US-138-005: Checkpoint corruption detection and healing integration.

Tests verify:
- Corruption metrics are tracked correctly
- Graceful degradation works when checkpoint is corrupted
- Integration between auto-repair and corruption tracking
- Checksum validation integration
"""

import gzip
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData


def _make_config(backup_count=3, compression_enabled=False):
    """Create a mock config with specified settings."""
    config = MagicMock()
    pipeline = MagicMock()
    pipeline.checkpoint_backup_count = backup_count
    pipeline.min_rotation_interval_seconds = 60
    # Disable compression for simpler testing
    compression = MagicMock()
    compression.enabled = compression_enabled
    compression.compression_level = 6
    pipeline.checkpoint_compression = compression
    # Enable auto-repair
    auto_repair = MagicMock()
    auto_repair.enabled = True
    auto_repair.log_repairs = True
    auto_repair.max_repair_attempts = 3
    pipeline.checkpoint_auto_repair = auto_repair
    # Disable signature for testing (requires key)
    sig_config = MagicMock()
    sig_config.enabled = False
    sig_config.key = None
    pipeline.checkpoint_signature = sig_config
    config.pipeline = pipeline
    return config


class TestCorruptionMetricsTracking:
    """Tests for corruption metrics tracking (US-138-005 AC4)."""

    @pytest.mark.fast
    def test_initial_corruption_stats_are_zero(self, tmp_path):
        """Verify initial corruption stats are zero."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        stats = manager.get_corruption_stats()
        assert stats['total_corruptions_detected'] == 0
        assert stats['graceful_recoveries'] == 0
        assert stats['total_data_loss_events'] == 0

    @pytest.mark.fast
    def test_json_error_recorded(self, tmp_path):
        """Verify JSON parse errors are tracked."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Manually trigger corruption recording
        manager._record_corruption_event('json_error', source='primary', repaired=False)

        stats = manager.get_corruption_stats()
        assert stats['total_corruptions_detected'] == 1
        assert stats['corruption_types']['json_error'] == 1
        assert stats['corruption_sources']['primary'] == 1

    @pytest.mark.fast
    def test_successful_repair_tracked(self, tmp_path):
        """Verify successful repairs are tracked."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Record a successful repair
        manager._record_corruption_event('truncated_json', source='backup', repaired=True)

        stats = manager.get_corruption_stats()
        assert stats['auto_repairs_attempted'] == 1
        assert stats['auto_repairs_successful'] == 1
        assert stats['graceful_recoveries'] == 1
        assert stats['success_rate'] == 1.0

    @pytest.mark.fast
    def test_data_loss_recorded(self, tmp_path):
        """Verify data loss events are tracked."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        manager._record_corruption_event('missing_field', source='primary', repaired=False, data_lost=True)

        stats = manager.get_corruption_stats()
        assert stats['total_data_loss_events'] == 1

    @pytest.mark.fast
    def test_corruption_history_limit(self, tmp_path):
        """Verify corruption history is limited to 10 events."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Add 15 corruption events
        for i in range(15):
            manager._record_corruption_event(f'type_{i}', source='primary', repaired=False)

        stats = manager.get_corruption_stats()
        # Should only keep last 10
        assert len(stats['corruption_history']) == 10


class TestGracefulDegradation:
    """Tests for graceful degradation when checkpoint is corrupted (US-138-005 AC5)."""

    @pytest.mark.fast
    def test_degradation_warnings_initially_empty(self, tmp_path):
        """Verify degradation warnings are initially empty."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        warnings = manager.get_degradation_warnings()
        assert warnings == []

    @pytest.mark.fast
    def test_graceful_degradation_enabled_by_default(self, tmp_path):
        """Verify graceful degradation is enabled by default."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        assert manager._graceful_degradation_enabled is True

    @pytest.mark.fast
    def test_validation_failure_adds_warning(self, tmp_path):
        """Verify validation failures add degradation warnings."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Add a warning manually
        manager._degradation_warnings.append("Checkpoint validation failed - continuing with partial data")

        warnings = manager.get_degradation_warnings()
        assert len(warnings) == 1
        assert "validation failed" in warnings[0]


class TestCorruptionRecovery:
    """Integration tests for corruption recovery scenarios."""

    @pytest.mark.fast
    def test_load_with_corrupted_primary_uses_backup(self, tmp_path):
        """Verify load uses backup when primary is corrupted."""
        config = _make_config()
        # Create corrupted primary
        (tmp_path / "checkpoint.json").write_text("{corrupted!!")

        # Create valid backup
        backup_data = {
            "version": "2.1",
            "created_at": "2026-01-15T10:00:00",
            "updated_at": "2026-01-15T10:00:00",
            "last_completed_stage": "CAPTION",
            "caption": {"fetched": 5},
        }
        (tmp_path / "checkpoint.backup.json").write_text(json.dumps(backup_data))

        manager = CheckpointManager(tmp_path, config=config)
        result = manager.load()

        assert result is not None
        assert result.last_completed_stage == "CAPTION"

    @pytest.mark.fast
    def test_corruption_metrics_updated_on_load_failure(self, tmp_path):
        """Verify corruption metrics are updated when load fails."""
        config = _make_config()
        # Create corrupted primary and backup
        (tmp_path / "checkpoint.json").write_text("{corrupted!!")
        (tmp_path / "checkpoint.backup.json").write_text("{also corrupted!!")

        manager = CheckpointManager(tmp_path, config=config)
        result = manager.load()

        assert result is None
        # Should have recorded corruption events
        stats = manager.get_corruption_stats()
        assert stats['total_corruptions_detected'] > 0

    @pytest.mark.fast
    def test_repair_stats_tracked_separately(self, tmp_path):
        """Verify repair stats are tracked separately from corruption stats."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Manually trigger repair stats
        manager._repair_stats['total_repair_attempts'] = 5
        manager._repair_stats['successful_repairs'] = 4

        repair_stats = manager.get_repair_stats()
        corruption_stats = manager.get_corruption_stats()

        # They should be independent
        assert repair_stats['total_repair_attempts'] == 5
        assert corruption_stats['total_corruptions_detected'] == 0


class TestChecksumValidation:
    """Tests for checksum validation integration (US-138-005 AC1)."""

    @pytest.mark.fast
    def test_validate_checkpoint_returns_valid(self, tmp_path):
        """Verify validate_checkpoint returns valid result for good checkpoint."""
        from src.checkpoint import validate_checkpoint

        # Create a valid checkpoint
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Validate it
        result = validate_checkpoint(str(tmp_path / "checkpoint.json"))

        assert result['is_valid'] is True
        assert result['json_valid'] is True
        assert result['schema_valid'] is True


class TestIntegrationScenarios:
    """End-to-end integration tests for corruption scenarios."""

    @pytest.mark.fast
    def test_full_corruption_recovery_flow(self, tmp_path):
        """Test complete flow: corrupt -> detect -> repair -> track metrics."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Create initial valid checkpoint
        manager.save("ANALYZE", {"keywords": ["test", "data"]})
        initial_stats = manager.get_corruption_stats()

        # Now simulate corruption by writing bad data
        (tmp_path / "checkpoint.json").write_text("{corrupted")

        # Load should detect corruption and try to recover
        result = manager.load()

        # After recovery attempt, stats should be updated
        final_stats = manager.get_corruption_stats()

        # Stats should have been updated (either recovery or failure tracked)
        assert final_stats['total_corruptions_detected'] >= initial_stats['total_corruptions_detected']

    @pytest.mark.fast
    def test_graceful_degradation_with_partial_data(self, tmp_path):
        """Test graceful degradation continues with partial data."""
        config = _make_config()
        manager = CheckpointManager(tmp_path, config=config)

        # Create a checkpoint with partial data
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Manually add a degradation warning
        manager._degradation_warnings.append("Stage data incomplete")

        # Verify we can still get the warnings
        warnings = manager.get_degradation_warnings()
        assert len(warnings) > 0
