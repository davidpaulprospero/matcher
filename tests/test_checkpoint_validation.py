"""
Test script for checkpoint validation and integrity features (US-108-010).

Run: python tests/test_checkpoint_validation.py
"""

import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.checkpoint import (
    CheckpointManager, CheckpointData, CURRENT_CHECKPOINT_VERSION
)


@pytest.mark.integration
def test_verify_integrity_valid_checkpoint():
    """Test verify_integrity with a valid checkpoint"""
    print("\n" + "=" * 60)
    print("  TEST: verify_integrity with valid checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create checkpoint manager
        cm = CheckpointManager(tmpdir, config_hash="test123")

        # Save first checkpoint (creates the checkpoint file)
        cm.set_voiceover(str(tmpdir / "test.srt"))
        cm.save("ANALYZE", {
            'keywords': ['keyword1', 'keyword2'],
            'segment_count': 5,
        })

        # Save second checkpoint (creates the backup from first save)
        cm.save("VIDEO_SEARCH", {
            'video_ids': ['abc123'],
            'count': 1,
        })

        # Verify integrity
        result = cm.verify_integrity()

        print(f"  Result: {result}")

        assert result['is_valid'] == True, "Checkpoint should be valid"
        assert result['checksum'] != '', "Checksum should be computed"
        assert result['schema_valid'] == True, "Schema should be valid"
        assert result['version'] == CURRENT_CHECKPOINT_VERSION, "Version should match"
        assert result['last_stage'] == 'VIDEO_SEARCH', "Last stage should be VIDEO_SEARCH"
        assert result['backup_count'] >= 1, "Should have at least 1 backup"

        print("  ✅ verify_integrity PASSED")


@pytest.mark.integration
def test_verify_integrity_missing_checkpoint():
    """Test verify_integrity when checkpoint doesn't exist"""
    print("\n" + "=" * 60)
    print("  TEST: verify_integrity with missing checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test123")
        result = cm.verify_integrity()

        print(f"  Result: {result}")

        assert result['is_valid'] == False, "Should be invalid when missing"
        assert "does not exist" in result['issues'][0], "Should mention missing file"

        print("  ✅ verify_integrity missing checkpoint PASSED")


@pytest.mark.integration
def test_verify_integrity_corrupt_checkpoint():
    """Test verify_integrity with corrupt JSON"""
    print("\n" + "=" * 60)
    print("  TEST: verify_integrity with corrupt checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write corrupt JSON
        corrupt_path = tmpdir / "checkpoint.json"
        with open(corrupt_path, 'w') as f:
            f.write("{ invalid json }")

        cm = CheckpointManager(tmpdir, config_hash="test123")
        result = cm.verify_integrity()

        print(f"  Result: {result}")

        assert result['is_valid'] == False, "Should be invalid when corrupt"
        assert result['schema_valid'] == False, "Schema should be invalid"
        assert any("Invalid JSON" in issue for issue in result['issues']), "Should mention JSON error"

        print("  ✅ verify_integrity corrupt checkpoint PASSED")


@pytest.mark.integration
def test_restore_from_backup():
    """Test restore_from_backup method"""
    print("\n" + "=" * 60)
    print("  TEST: restore_from_backup")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test123")

        # Save first checkpoint
        cm.set_voiceover(str(tmpdir / "test.srt"))
        cm.save("ANALYZE", {'keywords': ['keyword1'], 'segment_count': 1})
        first_checksum = cm.verify_integrity()['checksum']

        # Save second checkpoint (different data)
        cm.save("VIDEO_SEARCH", {'video_ids': ['abc123'], 'count': 1})
        second_data = cm.data

        # Restore from backup index 0
        success = cm.restore_from_backup(0)

        assert success == True, "Restore should succeed"
        assert cm.data.last_completed_stage == "ANALYZE", "Should be restored to ANALYZE stage"

        print(f"  Restored to stage: {cm.data.last_completed_stage}")
        print("  ✅ restore_from_backup PASSED")


@pytest.mark.integration
def test_restore_from_invalid_backup():
    """Test restore_from_backup with invalid index"""
    print("\n" + "=" * 60)
    print("  TEST: restore_from_backup with invalid index")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test123")

        # Try to restore from non-existent backup
        success = cm.restore_from_backup(99)

        assert success == False, "Should fail with invalid index"

        print("  ✅ restore_from_invalid_backup PASSED")


@pytest.mark.integration
def test_backup_rotation_5_copies():
    """Test that backup rotation keeps 5 backups"""
    print("\n" + "=" * 60)
    print("  TEST: backup rotation with 5 copies")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create manager with 5 backups
        cm = CheckpointManager(tmpdir, config_hash="test123")
        cm._backup_count = 5

        # Save 6 checkpoints to trigger rotation
        for i in range(6):
            cm.set_voiceover(str(tmpdir / "test.srt"))
            cm.save("ANALYZE", {'iteration': i, 'keywords': [f'kw{i}']})

        # Check how many backups exist
        backup_count = 0
        for i in range(5):
            if cm._get_backup_path(i).exists():
                backup_count += 1

        print(f"  Backup count: {backup_count}")
        assert backup_count == 5, f"Should have 5 backups, got {backup_count}"

        print("  ✅ backup_rotation_5_copies PASSED")


@pytest.mark.integration
def test_get_backup_info():
    """Test get_backup_info method"""
    print("\n" + "=" * 60)
    print("  TEST: get_backup_info")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test123")
        cm._backup_count = 5

        # Save first checkpoint (creates the checkpoint file)
        cm.set_voiceover(str(tmpdir / "test.srt"))
        cm.save("ANALYZE", {'keywords': ['kw1']})

        # Save second checkpoint (creates the backup)
        cm.save("VIDEO_SEARCH", {'video_ids': ['abc']})

        # Get backup info
        backups = cm.get_backup_info()

        print(f"  Backups: {backups}")
        assert len(backups) >= 1, "Should have at least 1 backup"
        assert 'index' in backups[0], "Backup info should have index"
        assert 'path' in backups[0], "Backup info should have path"
        assert 'size_bytes' in backups[0], "Backup info should have size"

        print("  ✅ get_backup_info PASSED")


@pytest.mark.integration
def test_checkpoint_info_display():
    """Test checkpoint info display with all required fields"""
    print("\n" + "=" * 60)
    print("  TEST: checkpoint info display")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test123")
        cm.set_voiceover(str(tmpdir / "test.srt"))
        cm.save("MATCH", {
            'keywords': ['test'],
            'match_count': 10,
            'avg_confidence': 0.85,
        })

        # Get integrity info (for --checkpoint-info display)
        integrity = cm.verify_integrity()

        # Get backup info
        backups = cm.get_backup_info()

        # Display format similar to --checkpoint-info output
        print(f"\n  === Checkpoint Info ===")
        print(f"  Version: {integrity['version']}")
        print(f"  Last Stage: {integrity['last_stage']}")
        print(f"  Backup Count: {integrity['backup_count']}")
        print(f"  Integrity Status: {'VALID' if integrity['is_valid'] else 'INVALID'}")
        print(f"  Checksum: {integrity['checksum'][:16]}...")
        print(f"  Available Backups:")
        for b in backups:
            print(f"    - {b['path']}: {b['size_bytes']} bytes")

        assert integrity['version'] != '', "Should have version"
        assert integrity['last_stage'] != '', "Should have last stage"
        assert integrity['backup_count'] >= 0, "Should have backup count"

        print("  ✅ checkpoint_info_display PASSED")


# Tests for validate_checkpoint standalone function (US-120-007)
from src.checkpoint import validate_checkpoint, STAGE_ORDER, CURRENT_CHECKPOINT_VERSION


@pytest.mark.integration
def test_validate_checkpoint_valid():
    """Test validate_checkpoint with a valid checkpoint"""
    print("\n" + "=" * 60)
    print("  TEST: validate_checkpoint with valid checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a valid checkpoint file
        checkpoint_path = tmpdir / "checkpoint.json"
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).isoformat()

        checkpoint_data = {
            "version": CURRENT_CHECKPOINT_VERSION,
            "last_completed_stage": "VIDEO_SEARCH",
            "created_at": created_at,
            "completed_stages": ["ANALYZE", "VIDEO_SEARCH"],
            "voiceover_hash": "abc123",
            "video_search": {"video_ids": ["vid1", "vid2"], "count": 2}
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        # Validate
        result = validate_checkpoint(str(checkpoint_path))

        print(f"  Result: {result}")

        assert result['is_valid'] == True, "Checkpoint should be valid"
        assert result['json_valid'] == True, "JSON should be valid"
        assert result['schema_valid'] == True, "Schema should be valid"
        assert result['stage_order_valid'] == True, "Stage order should be valid"
        assert result['version'] == CURRENT_CHECKPOINT_VERSION, "Version should match"
        assert result['last_stage'] == "VIDEO_SEARCH", "Last stage should match"

        print("  ✅ validate_checkpoint valid PASSED")


@pytest.mark.integration
def test_validate_checkpoint_missing():
    """Test validate_checkpoint when checkpoint doesn't exist"""
    print("\n" + "=" * 60)
    print("  TEST: validate_checkpoint with missing checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        missing_path = tmpdir / "nonexistent.json"

        result = validate_checkpoint(str(missing_path))

        print(f"  Result: {result}")

        assert result['is_valid'] == False, "Should be invalid when missing"
        assert result['json_valid'] == False, "JSON should be invalid"
        assert "does not exist" in result['issues'][0], "Should mention missing file"

        print("  ✅ validate_checkpoint missing PASSED")


@pytest.mark.integration
def test_validate_checkpoint_corrupt():
    """Test validate_checkpoint with corrupt JSON"""
    print("\n" + "=" * 60)
    print("  TEST: validate_checkpoint with corrupt checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write corrupt JSON
        corrupt_path = tmpdir / "checkpoint.json"
        with open(corrupt_path, 'w') as f:
            f.write("{ invalid json }")

        result = validate_checkpoint(str(corrupt_path))

        print(f"  Result: {result}")

        assert result['is_valid'] == False, "Should be invalid when corrupt"
        assert result['json_valid'] == False, "JSON should be invalid"
        assert any("Invalid JSON" in issue for issue in result['issues']), "Should mention JSON error"

        print("  ✅ validate_checkpoint corrupt PASSED")


@pytest.mark.integration
def test_validate_checkpoint_invalid_stage_order():
    """Test validate_checkpoint with invalid stage order"""
    print("\n" + "=" * 60)
    print("  TEST: validate_checkpoint with invalid stage order")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create checkpoint with invalid stage order (OUTPUT before ANALYZE)
        checkpoint_path = tmpdir / "checkpoint.json"
        from datetime import datetime, timezone
        created_at = datetime.now(timezone.utc).isoformat()

        checkpoint_data = {
            "version": CURRENT_CHECKPOINT_VERSION,
            "last_completed_stage": "ANALYZE",
            "created_at": created_at,
            "completed_stages": ["OUTPUT", "ANALYZE"],  # Invalid order
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        result = validate_checkpoint(str(checkpoint_path))

        print(f"  Result: {result}")

        assert result['is_valid'] == False, "Should be invalid with bad stage order"
        assert result['stage_order_valid'] == False, "Stage order should be invalid"
        assert any("order violation" in issue.lower() for issue in result['issues']), "Should mention order violation"

        print("  ✅ validate_checkpoint invalid stage order PASSED")


@pytest.mark.integration
def test_validate_checkpoint_stale():
    """Test validate_checkpoint with stale checkpoint (>24h old)"""
    print("\n" + "=" * 60)
    print("  TEST: validate_checkpoint with stale checkpoint")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create checkpoint with old timestamp (>24 hours)
        checkpoint_path = tmpdir / "checkpoint.json"
        from datetime import datetime, timezone, timedelta
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)

        checkpoint_data = {
            "version": CURRENT_CHECKPOINT_VERSION,
            "last_completed_stage": "MATCH",
            "created_at": old_time.isoformat(),
            "completed_stages": ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH"],
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        result = validate_checkpoint(str(checkpoint_path))

        print(f"  Result: {result}")

        assert result['is_stale'] == True, "Checkpoint should be marked as stale"
        assert result['age_hours'] > 24.0, "Age should be > 24 hours"
        assert any("stale" in w.lower() for w in result['warnings']), "Should warn about staleness"

        print("  ✅ validate_checkpoint stale PASSED")


@pytest.mark.integration
def test_validate_checkpoint_missing_fields():
    """Test validate_checkpoint with missing required fields"""
    print("\n" + "=" * 60)
    print("  TEST: validate_checkpoint with missing required fields")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create checkpoint missing required fields
        checkpoint_path = tmpdir / "checkpoint.json"
        checkpoint_data = {
            "version": CURRENT_CHECKPOINT_VERSION,
            # Missing: last_completed_stage, created_at
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        result = validate_checkpoint(str(checkpoint_path))

        print(f"  Result: {result}")

        assert result['is_valid'] == False, "Should be invalid with missing fields"
        assert result['schema_valid'] == False, "Schema should be invalid"
        assert any("last_completed_stage" in issue for issue in result['issues']), "Should mention missing field"
        assert any("created_at" in issue for issue in result['issues']), "Should mention missing field"

        print("  ✅ validate_checkpoint missing fields PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
