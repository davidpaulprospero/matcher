"""
Snapshot tests for OTIO output validation.

Provides regression protection for OTIO timeline generation.
"""

import pytest
import json
from pathlib import Path
import sys
from unittest.mock import Mock, MagicMock
import opentimelineio as otio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.helpers.snapshot import SnapshotManager
from src.otio.utils import create_clip_with_timewarp
from src.otio.timeline import create_timeline


# Snapshot directory
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


@pytest.fixture
def snapshot_manager():
    """Fixture providing snapshot manager."""
    return SnapshotManager(SNAPSHOT_DIR)


class TestOTIOSnapshot:
    """Test OTIO output against stored snapshots."""

    @pytest.mark.fast
    def test_clip_creation_snapshot(self, snapshot_manager):
        """Test that clip creation produces consistent output."""
        # Create a sample clip using the existing function
        clip = create_clip_with_timewarp(
            name="Test Clip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=24.0,
            metadata={
                "confidence": 0.85,
                "strategy": "primary",
                "vo_index": 0,
                "vo_text": "Test voiceover segment"
            }
        )

        # Compare against snapshot
        matches, diff = snapshot_manager.compare("clip_creation", clip)

        if not matches:
            # If snapshot doesn't exist, create it
            if diff and "does not exist" in diff:
                snapshot_manager.save_snapshot("clip_creation", clip)
                pytest.skip("Snapshot created - run test again to verify")
            else:
                pytest.fail(f"Clip creation snapshot mismatch:\n{diff}")

    @pytest.mark.fast
    def test_clip_with_timewarp_snapshot(self, snapshot_manager):
        """Test clip with timewarp produces consistent output."""
        # Create a clip with slowdown (source 2.5s -> target 3.75s = 1.5x slowdown)
        clip = create_clip_with_timewarp(
            name="Slowdown Clip",
            source_path="C:/Videos/test.mp4",
            source_start=4.17,  # 100 frames / 24 fps
            source_duration=2.5,  # 60 frames / 24 fps
            target_duration=3.75,  # 90 frames / 24 fps
            frame_rate=24.0,
            metadata={
                "confidence": 0.78,
                "strategy": "primary",
                "vo_index": 1,
                "vo_text": "Another test segment"
            }
        )

        matches, diff = snapshot_manager.compare("clip_with_timewarp", clip)

        if not matches:
            if diff and "does not exist" in diff:
                snapshot_manager.save_snapshot("clip_with_timewarp", clip)
                pytest.skip("Snapshot created - run test again to verify")
            else:
                pytest.fail(f"Clip with timewarp snapshot mismatch:\n{diff}")

    @pytest.mark.fast
    def test_timeline_structure_snapshot(self, snapshot_manager):
        """Test timeline structure matches expected format."""
        # Create a minimal timeline
        timeline = otio.schema.Timeline(name="test_timeline")

        # Add metadata via the metadata property
        timeline.metadata["project_name"] = "Test Project"
        timeline.metadata["total_vo_segments"] = 3

        # Add a video track
        video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        timeline.tracks.append(video_track)

        matches, diff = snapshot_manager.compare("timeline_structure", timeline)

        if not matches:
            if diff and "does not exist" in diff:
                snapshot_manager.save_snapshot("timeline_structure", timeline)
                pytest.skip("Snapshot created - run test again to verify")
            else:
                pytest.fail(f"Timeline structure snapshot mismatch:\n{diff}")

    @pytest.mark.fast
    def test_metadata_serialization_snapshot(self, snapshot_manager):
        """Test metadata serialization is consistent."""
        metadata = {
            "confidence": 0.92,
            "strategy": "primary",
            "vo_index": 0,
            "vo_text": "Welcome to this documentary.",
            "vo_start": 0.0,
            "vo_end": 5.5,
            "video_source_file": "dQw4w9WgXcQ",
            "video_title": "Test Video",
            "video_start": 0.0,
            "video_end": 5.5,
            "matched_keywords": ["documentary", "welcome", "test"]
        }

        matches, diff = snapshot_manager.compare("metadata_serialization", metadata)

        if not matches:
            if diff and "does not exist" in diff:
                snapshot_manager.save_snapshot("metadata_serialization", metadata)
                pytest.skip("Snapshot created - run test again to verify")
            else:
                pytest.fail(f"Metadata serialization snapshot mismatch:\n{diff}")


class TestSnapshotUpdate:
    """Tests for updating snapshots when OTIO changes are intentional."""

    @pytest.mark.fast
    def test_update_snapshot(self, snapshot_manager):
        """Test that snapshots can be updated for intentional changes."""
        # This test demonstrates the update mechanism
        test_data = {
            "version": "2.0",
            "features": ["new_feature_1", "new_feature_2"]
        }

        # Save initial snapshot
        snapshot_manager.save_snapshot("update_test", test_data)

        # Verify it was saved
        loaded = snapshot_manager.load_snapshot("update_test")
        assert loaded == test_data

        # Update with new data
        updated_data = {
            "version": "2.1",
            "features": ["new_feature_1", "new_feature_2", "new_feature_3"]
        }
        snapshot_manager.update_snapshot("update_test", updated_data)

        # Verify update worked
        loaded = snapshot_manager.load_snapshot("update_test")
        assert loaded == updated_data


class TestSnapshotRegression:
    """Tests to verify snapshot tests catch regressions."""

    @pytest.mark.fast
    def test_regression_clip_metadata_change(self, snapshot_manager):
        """Test that changes to clip metadata are detected."""
        # Load existing snapshot or create baseline
        baseline = snapshot_manager.load_snapshot("metadata_serialization")

        if baseline is None:
            # Create baseline for this test
            baseline = {
                "confidence": 0.92,
                "strategy": "primary",
                "vo_index": 0
            }
            snapshot_manager.save_snapshot("metadata_serialization", baseline)

        # Simulate a regression: different confidence value
        current = {
            "confidence": 0.50,  # This is a regression!
            "strategy": "primary",
            "vo_index": 0
        }

        matches, diff = snapshot_manager.compare("metadata_serialization", current)

        # Should NOT match (regression detected)
        assert not matches, "Regression not detected: confidence changed from 0.92 to 0.50"

    @pytest.mark.fast
    def test_regression_new_field(self, snapshot_manager):
        """Test that new fields in output are detected."""
        baseline = {
            "field_a": "value_a",
            "field_b": "value_b"
        }
        snapshot_manager.save_snapshot("regression_test", baseline)

        # Current has a new field
        current = {
            "field_a": "value_a",
            "field_b": "value_b",
            "field_c": "new_field"  # New field - should be detected
        }

        matches, diff = snapshot_manager.compare("regression_test", current)

        # Should NOT match (new field detected)
        assert not matches, "New field not detected"
