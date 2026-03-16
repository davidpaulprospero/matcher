"""
Tests for checkpoint load/save timing logging.

Verifies that CheckpointManager logs timing information for load() and _atomic_save() methods.
"""

import json
import logging
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class TestLoadTimingLogged:
    """Tests for checkpoint load timing being logged."""

    @pytest.mark.fast
    def test_load_logs_timing_message(self, tmp_path, caplog):
        """Verify that load() logs timing in milliseconds."""
        # Create valid checkpoint file
        checkpoint_data = {
            "version": "1.0",
            "created_at": "2026-01-25T10:00:00",
            "updated_at": "2026-01-25T10:00:00",
            "last_completed_stage": "DOWNLOAD",
            "config_hash": "abc123",
            "voiceover_path": "",
            "voiceover_hash": "",
            "analyze": {},
            "download": {},
            "transcribe": {},
            "match": {}
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.INFO, logger='src.checkpoint'):
            result = manager.load()

        assert result is not None
        # Check that timing message was logged
        timing_messages = [r.message for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_messages) == 1
        assert 'ms' in timing_messages[0]

    @pytest.mark.fast
    def test_load_timing_includes_milliseconds(self, tmp_path, caplog):
        """Verify that load timing message includes milliseconds unit."""
        checkpoint_data = {
            "version": "1.0",
            "created_at": "2026-01-25T10:00:00",
            "updated_at": "2026-01-25T10:00:00",
            "last_completed_stage": "ANALYZE"
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.INFO, logger='src.checkpoint'):
            manager.load()

        timing_messages = [r.message for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_messages) == 1
        assert 'ms' in timing_messages[0]
        # Should match pattern like "Checkpoint loaded in X.Xms"
        assert 'Checkpoint loaded in' in timing_messages[0]

    @pytest.mark.fast
    def test_load_timing_is_info_level(self, tmp_path, caplog):
        """Verify that load timing is logged at INFO level."""
        checkpoint_data = {
            "version": "1.0",
            "created_at": "2026-01-25T10:00:00",
            "last_completed_stage": "TRANSCRIBE"
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager.load()

        timing_records = [r for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_records) == 1
        assert timing_records[0].levelno == logging.INFO

    @pytest.mark.fast
    def test_load_no_timing_when_file_not_exists(self, tmp_path, caplog):
        """Verify that no timing is logged when checkpoint doesn't exist."""
        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.INFO, logger='src.checkpoint'):
            result = manager.load()

        assert result is None
        timing_messages = [r.message for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_messages) == 0

    @pytest.mark.fast
    def test_load_timing_positive_value(self, tmp_path, caplog):
        """Verify that load timing value is positive."""
        checkpoint_data = {
            "version": "1.0",
            "created_at": "2026-01-25T10:00:00",
            "last_completed_stage": "MATCH"
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.INFO, logger='src.checkpoint'):
            manager.load()

        timing_messages = [r.message for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_messages) == 1
        # Extract the numeric value from the message
        import re
        match = re.search(r'(\d+\.?\d*)ms', timing_messages[0])
        assert match is not None
        timing_value = float(match.group(1))
        assert timing_value >= 0  # Timing should be non-negative


class TestSaveTimingLogged:
    """Tests for checkpoint save timing being logged."""

    @pytest.mark.fast
    def test_atomic_save_logs_timing_message(self, tmp_path, caplog):
        """Verify that _atomic_save() logs timing in milliseconds."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="ANALYZE"
        )

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager._atomic_save()

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1
        assert 'ms' in timing_messages[0]

    @pytest.mark.fast
    def test_atomic_save_timing_includes_milliseconds(self, tmp_path, caplog):
        """Verify that save timing message includes milliseconds unit."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="DOWNLOAD"
        )

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager._atomic_save()

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1
        assert 'Checkpoint saved in' in timing_messages[0]
        assert 'ms' in timing_messages[0]

    @pytest.mark.fast
    def test_atomic_save_timing_is_debug_level(self, tmp_path, caplog):
        """Verify that save timing is logged at DEBUG level."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="TRANSCRIBE"
        )

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager._atomic_save()

        timing_records = [r for r in caplog.records if 'saved in' in r.message]
        assert len(timing_records) == 1
        assert timing_records[0].levelno == logging.DEBUG

    @pytest.mark.fast
    def test_save_calls_atomic_save_with_timing(self, tmp_path, caplog):
        """Verify that save() method logs timing via _atomic_save()."""
        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager.save("ANALYZE", {"keywords": ["test"]})

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1

    @pytest.mark.fast
    def test_atomic_save_timing_positive_value(self, tmp_path, caplog):
        """Verify that save timing value is positive."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="MATCH"
        )

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager._atomic_save()

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1
        # Extract the numeric value from the message
        import re
        match = re.search(r'(\d+\.?\d*)ms', timing_messages[0])
        assert match is not None
        timing_value = float(match.group(1))
        assert timing_value >= 0  # Timing should be non-negative


class TestTimingFormat:
    """Tests for timing message format."""

    @pytest.mark.fast
    def test_load_timing_format(self, tmp_path, caplog):
        """Verify load timing format is 'Checkpoint loaded in X.Xms'."""
        checkpoint_data = {
            "version": "1.0",
            "created_at": "2026-01-25T10:00:00",
            "last_completed_stage": "ANALYZE"
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.INFO, logger='src.checkpoint'):
            manager.load()

        timing_messages = [r.message for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_messages) == 1
        import re
        # Should match "Checkpoint loaded in X.Xms" format
        assert re.match(r'Checkpoint loaded in \d+\.?\d*ms', timing_messages[0])

    @pytest.mark.fast
    def test_save_timing_format(self, tmp_path, caplog):
        """Verify save timing format is 'Checkpoint saved in X.Xms'."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="ANALYZE"
        )

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager._atomic_save()

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1
        import re
        # Should match "Checkpoint saved in X.Xms" format
        assert re.match(r'Checkpoint saved in \d+\.?\d*ms', timing_messages[0])


class TestTimingWithLargeCheckpoint:
    """Tests for timing with larger checkpoint data."""

    @pytest.mark.fast
    def test_load_timing_large_checkpoint(self, tmp_path, caplog):
        """Verify timing is logged for large checkpoints."""
        # Create checkpoint with substantial data
        checkpoint_data = {
            "version": "1.0",
            "created_at": "2026-01-25T10:00:00",
            "updated_at": "2026-01-25T10:00:00",
            "last_completed_stage": "MATCH",
            "config_hash": "abc123",
            "analyze": {
                "keywords": [f"keyword_{i}" for i in range(100)],
                "entities": [{"name": f"entity_{i}", "type": "person"} for i in range(50)]
            },
            "download": {
                "video_paths": [f"/path/to/video_{i}.mp4" for i in range(200)]
            },
            "transcribe": {
                "transcriptions": {f"video_{i}": {"text": f"Transcript for video {i} " * 100} for i in range(50)}
            },
            "match": {
                "matches": [{"segment_index": i, "video_file": f"video_{i}.mp4", "confidence": 0.85} for i in range(100)]
            }
        }
        checkpoint_path = tmp_path / "checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(tmp_path)

        with caplog.at_level(logging.INFO, logger='src.checkpoint'):
            result = manager.load()

        assert result is not None
        timing_messages = [r.message for r in caplog.records if 'loaded in' in r.message]
        assert len(timing_messages) == 1

    @pytest.mark.fast
    def test_save_timing_large_checkpoint(self, tmp_path, caplog):
        """Verify timing is logged for large checkpoint saves."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="MATCH"
        )
        # Add substantial data
        manager.data.analyze = {
            "keywords": [f"keyword_{i}" for i in range(100)],
            "entities": [{"name": f"entity_{i}", "type": "person"} for i in range(50)]
        }
        manager.data.download = {
            "video_paths": [f"/path/to/video_{i}.mp4" for i in range(200)]
        }
        manager.data.match = {
            "matches": [{"segment_index": i, "video_file": f"video_{i}.mp4", "confidence": 0.85} for i in range(100)]
        }

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager._atomic_save()

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1


class TestIntermediateSaveTiming:
    """Tests for save_intermediate() timing."""

    @pytest.mark.fast
    def test_save_intermediate_logs_timing(self, tmp_path, caplog):
        """Verify that save_intermediate() logs timing via _atomic_save()."""
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="DOWNLOAD"
        )

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager.save_intermediate("DOWNLOAD_SEGMENTS", {"downloaded": ["video1.mp4"]})

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1


class TestMarkStageIncompleteTiming:
    """Tests for mark_stage_incomplete() timing."""

    @pytest.mark.fast
    def test_mark_stage_incomplete_logs_timing(self, tmp_path, caplog):
        """Verify that mark_stage_incomplete() logs timing via _atomic_save()."""
        # Create initial checkpoint
        manager = CheckpointManager(tmp_path)
        manager.data = CheckpointData(
            version="1.0",
            created_at="2026-01-25T10:00:00",
            last_completed_stage="MATCH"
        )
        manager._atomic_save()

        # Clear logs and mark stage incomplete
        caplog.clear()

        with caplog.at_level(logging.DEBUG, logger='src.checkpoint'):
            manager.mark_stage_incomplete("MATCH")

        timing_messages = [r.message for r in caplog.records if 'saved in' in r.message]
        assert len(timing_messages) == 1
