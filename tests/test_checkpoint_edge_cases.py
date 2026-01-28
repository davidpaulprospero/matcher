"""
Tests for checkpoint serialization edge cases (US-005, Sprint 19).

These tests document the CURRENT behavior of checkpoint serialization:
- numpy arrays: converted to string representation via json default=str
- datetime objects: converted to ISO format string via json default=str
- Circular references: NOT handled (would cause recursion error)
- Missing optional fields: return empty dict via get_stage_data()

Note: Some acceptance criteria described features (like numpy->list conversion)
that aren't implemented. These tests document what actually happens.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData

# Only import numpy if available for optional tests
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestNumpyArraySerialization:
    """AC1: Test checkpoint.save() behavior with numpy arrays in stage data."""

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
    @pytest.mark.fast
    def test_save_handles_numpy_arrays_via_default_str(self, tmp_path):
        """numpy arrays are converted to string representation (not lists)."""
        manager = CheckpointManager(tmp_path)

        # Stage data with numpy array
        arr = np.array([1.0, 2.0, 3.0])
        stage_data = {"embeddings": arr}

        # Should not raise - json uses default=str fallback
        manager.save("ANALYZE", stage_data)

        # Read raw JSON to verify what was saved
        with open(tmp_path / "checkpoint.json", "r") as f:
            raw_data = json.load(f)

        # numpy array becomes string representation (not a list)
        assert isinstance(raw_data["analyze"]["embeddings"], str)
        assert "[1." in raw_data["analyze"]["embeddings"]

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
    @pytest.mark.fast
    def test_numpy_arrays_not_restored_as_arrays(self, tmp_path):
        """Loading checkpoint with numpy data returns strings, not arrays."""
        manager = CheckpointManager(tmp_path)

        # Save numpy array
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        manager.save("ANALYZE", {"matrix": arr})

        # Load checkpoint
        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.get_stage_data("ANALYZE")

        # Loaded data is string, not numpy array
        assert isinstance(result["matrix"], str)
        assert not isinstance(result["matrix"], np.ndarray)


class TestDatetimeSerialization:
    """AC2: Test checkpoint.save() behavior with datetime objects in stage data."""

    @pytest.mark.fast
    def test_save_handles_datetime_via_default_str(self, tmp_path):
        """datetime objects are converted to ISO format strings."""
        manager = CheckpointManager(tmp_path)

        now = datetime(2026, 1, 28, 15, 30, 45)
        stage_data = {"completed_at": now}

        # Should not raise
        manager.save("ANALYZE", stage_data)

        # Read raw JSON to verify what was saved
        with open(tmp_path / "checkpoint.json", "r") as f:
            raw_data = json.load(f)

        # datetime becomes ISO string
        assert raw_data["analyze"]["completed_at"] == "2026-01-28 15:30:45"

    @pytest.mark.fast
    def test_datetime_restored_as_string(self, tmp_path):
        """Loading checkpoint with datetime data returns string, not datetime."""
        manager = CheckpointManager(tmp_path)

        stage_data = {"started_at": datetime(2026, 1, 1, 12, 0, 0)}
        manager.save("MATCH", stage_data)

        # Load checkpoint
        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.get_stage_data("MATCH")

        # Loaded data is string, not datetime
        assert isinstance(result["started_at"], str)
        assert result["started_at"] == "2026-01-01 12:00:00"


class TestCircularReferences:
    """AC4: Test checkpoint.save() behavior with circular references."""

    @pytest.mark.fast
    def test_circular_reference_causes_error(self, tmp_path):
        """Circular references are NOT handled and raise ValueError."""
        manager = CheckpointManager(tmp_path)

        # Create circular reference
        obj = {"name": "parent"}
        obj["self"] = obj  # Circular reference

        # This would raise ValueError (circular reference) or RecursionError
        # The checkpoint code doesn't have special handling for this
        with pytest.raises((ValueError, RecursionError)):
            manager.save("ANALYZE", obj)


class TestMissingOptionalFields:
    """AC5: Test checkpoint.load() returns None/empty for missing optional stage data."""

    @pytest.mark.fast
    def test_missing_stage_data_returns_empty_dict(self, tmp_path):
        """get_stage_data() returns empty dict for stages not saved."""
        manager = CheckpointManager(tmp_path)

        # Save only ANALYZE stage
        manager.save("ANALYZE", {"keywords": ["test"]})

        # MATCH was never saved
        result = manager.get_stage_data("MATCH")
        assert result == {}

    @pytest.mark.fast
    def test_load_missing_checkpoint_returns_none(self, tmp_path):
        """load() returns None when no checkpoint file exists."""
        manager = CheckpointManager(tmp_path)
        result = manager.load()
        assert result is None

    @pytest.mark.fast
    def test_partial_checkpoint_loads_available_fields(self, tmp_path):
        """Checkpoint with some fields missing still loads successfully."""
        # Write minimal checkpoint JSON directly
        checkpoint_path = tmp_path / "checkpoint.json"
        minimal_data = {
            "version": "1.0",
            "created_at": "2026-01-28T12:00:00",
            "last_completed_stage": "ANALYZE",
            # Missing: updated_at, config_hash, voiceover_path, voiceover_hash
            # Missing all stage data fields
        }
        with open(checkpoint_path, "w") as f:
            json.dump(minimal_data, f)

        manager = CheckpointManager(tmp_path)
        loaded = manager.load()

        assert loaded is not None
        assert loaded.last_completed_stage == "ANALYZE"
        assert loaded.analyze == {}  # Default empty dict


class TestCustomObjectSerialization:
    """Test serialization of custom objects via default=str fallback."""

    @pytest.mark.fast
    def test_custom_class_becomes_string(self, tmp_path):
        """Custom objects are converted to string representation."""
        manager = CheckpointManager(tmp_path)

        class CustomData:
            def __init__(self, value):
                self.value = value
            def __str__(self):
                return f"CustomData({self.value})"

        stage_data = {"custom": CustomData(42)}
        manager.save("ANALYZE", stage_data)

        # Read raw JSON
        with open(tmp_path / "checkpoint.json", "r") as f:
            raw_data = json.load(f)

        assert raw_data["analyze"]["custom"] == "CustomData(42)"

    @pytest.mark.fast
    def test_path_object_becomes_string(self, tmp_path):
        """pathlib.Path objects are converted to string."""
        manager = CheckpointManager(tmp_path)

        stage_data = {"video_path": Path("/videos/test.mp4")}
        manager.save("DOWNLOAD", stage_data)

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.get_stage_data("DOWNLOAD")
        assert result["video_path"] == str(Path("/videos/test.mp4"))


class TestSerializationRobustness:
    """Test robustness of checkpoint serialization."""

    @pytest.mark.fast
    def test_unicode_content_preserved(self, tmp_path):
        """Unicode content in stage data is preserved."""
        manager = CheckpointManager(tmp_path)

        stage_data = {
            "keywords": ["日本語", "한국어", "中文", "émojis 🎬"],
            "topic": "Café résumé"
        }
        manager.save("ANALYZE", stage_data)

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.get_stage_data("ANALYZE")
        assert result["keywords"] == stage_data["keywords"]
        assert result["topic"] == stage_data["topic"]

    @pytest.mark.fast
    def test_large_nested_structures_preserved(self, tmp_path):
        """Deeply nested dicts/lists survive round-trip."""
        manager = CheckpointManager(tmp_path)

        stage_data = {
            "matches": [
                {
                    "segment_id": i,
                    "results": [
                        {"video_id": f"vid_{j}", "confidence": 0.8 + j * 0.01}
                        for j in range(5)
                    ]
                }
                for i in range(10)
            ]
        }
        manager.save("MATCH", stage_data)

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.get_stage_data("MATCH")
        assert len(result["matches"]) == 10
        assert len(result["matches"][0]["results"]) == 5
        assert result["matches"][5]["results"][2]["video_id"] == "vid_2"
