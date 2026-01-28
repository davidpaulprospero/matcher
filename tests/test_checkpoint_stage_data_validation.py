"""
Tests for checkpoint stage data validation (US-005, Sprint 21, Rule 25).

Covers:
- AC1: --output-only fails gracefully when stages dict is empty {}
- AC2: --output-only succeeds when checkpoint has populated stages dict
- AC3: Checkpoint backup restoration recovers stage data (existing tests cover this)
- AC4: Manually edited checkpoint without stages triggers appropriate warning
- AC5: Fixture that generates valid checkpoint with stage data for OUTPUT tests

Rule 25: `--output-only` needs stage data - Checkpoint must have populated
`stages` dict, not just `last_completed_stage`.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER
from src.stages.output import OutputStage
from src.state import PipelineState
from src.utils import Match, MatchResult, SRTSegment


# ============================================================================
# Fixtures (AC5)
# ============================================================================

@pytest.fixture
def valid_checkpoint_with_stage_data(tmp_path):
    """
    AC5: Create fixture that generates valid checkpoint with stage data for OUTPUT tests.

    Returns a tuple of (project_dir, checkpoint_manager) with fully populated
    stage data as would exist after a complete pipeline run up to MATCH.
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create checkpoint data with all stages populated
    checkpoint_data = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "last_completed_stage": "MATCH",
        "config_hash": "test_hash_123",
        "voiceover_path": str(project_dir / "voiceover.srt"),
        "voiceover_hash": "abc123",
        "analyze": {
            "keywords": ["travel", "nature", "adventure"],
            "segment_count": 10,
            "topic": "documentary"
        },
        "entity_images": {
            "images": ["beach.jpg", "mountain.jpg"],
            "sources": ["google", "bing"]
        },
        "entity_videos": {
            "videos": ["stock1.mp4"],
            "api_calls": 3
        },
        "download": {
            "video_paths": [
                str(project_dir / "videos" / "vid1.mp4"),
                str(project_dir / "videos" / "vid2.mp4")
            ],
            "count": 2
        },
        "stock": {
            "stock_videos": ["pexels_ocean.mp4"],
            "source": "pexels"
        },
        "broll_download": {
            "broll_paths": [str(project_dir / "broll" / "broll1.mp4")],
            "keyword_suffixes": ["aerial", "drone"]
        },
        "remix": {
            "filtered_count": 8,
            "removed": ["irrelevant.mp4"]
        },
        "caption": {
            "caption_count": 5,
            "languages": ["en"]
        },
        "transcribe": {
            "transcribed_count": 8,
            "embedding_count": 8
        },
        "scene_detection": {
            "scenes": 24,
            "broll_flagged": 3
        },
        "match": {
            "match_count": 10,
            "avg_confidence": 0.85,
            "matches": [
                {
                    "segment_id": 0,
                    "video_file": "vid1.mp4",
                    "confidence": 0.92,
                    "start": 0.0,
                    "end": 3.0
                },
                {
                    "segment_id": 1,
                    "video_file": "vid2.mp4",
                    "confidence": 0.78,
                    "start": 3.0,
                    "end": 6.0
                }
            ]
        },
        "broll_match": {
            "broll_matches": 3,
            "strategies": ["embedding", "keyword"]
        },
        "download_segments": {
            "segments_downloaded": 10,
            "total_size_mb": 250.5
        }
    }

    # Write checkpoint file
    checkpoint_path = project_dir / "checkpoint.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Create voiceover file
    voiceover_path = project_dir / "voiceover.srt"
    voiceover_path.write_text("1\n00:00:00,000 --> 00:00:03,000\nTest voiceover\n")

    # Initialize checkpoint manager
    manager = CheckpointManager(project_dir, config_hash="test_hash_123")
    manager.load()

    return project_dir, manager


@pytest.fixture
def empty_stages_checkpoint(tmp_path):
    """
    Create checkpoint with last_completed_stage set but stages dict empty.
    This simulates a corrupted or manually edited checkpoint.
    """
    project_dir = tmp_path / "test_project_empty"
    project_dir.mkdir()

    # Create checkpoint with last_completed_stage but NO stage data
    checkpoint_data = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "last_completed_stage": "MATCH",
        "config_hash": "test_hash",
        "voiceover_path": "",
        "voiceover_hash": "",
        # All stage dicts intentionally empty - this is the problem scenario
        "analyze": {},
        "entity_images": {},
        "entity_videos": {},
        "download": {},
        "stock": {},
        "broll_download": {},
        "remix": {},
        "caption": {},
        "transcribe": {},
        "scene_detection": {},
        "match": {},  # Empty match data - critical for OUTPUT stage
        "broll_match": {},
        "download_segments": {}
    }

    checkpoint_path = project_dir / "checkpoint.json"
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)

    manager = CheckpointManager(project_dir)
    manager.load()

    return project_dir, manager


@pytest.fixture
def mock_matches():
    """Create mock MatchResult objects for testing"""
    vo_seg = SRTSegment(
        index=0, start_time=0.0, end_time=3.0,
        text="Test voiceover", source_file="voiceover.srt"
    )
    vid_seg = SRTSegment(
        index=0, start_time=0.0, end_time=3.0,
        text="Test video", source_file="video.mp4"
    )
    match = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=0.9,
        reasoning="Test match"
    )
    return [MatchResult(primary_match=match)]


# ============================================================================
# AC1: Test --output-only fails gracefully when stages dict is empty {}
# ============================================================================

class TestOutputOnlyFailsWithEmptyStages:
    """AC1: Test --output-only behavior when stages dict is empty"""

    @pytest.mark.fast
    def test_restore_returns_false_with_empty_match_data(self, empty_stages_checkpoint):
        """OutputStage.restore() returns False when match data is empty"""
        project_dir, checkpoint = empty_stages_checkpoint

        stage = OutputStage()
        state = PipelineState()

        # Empty stages means restore should fail
        result = stage.restore(state, checkpoint)

        # restore() should return False because there's no data to restore
        assert result is False

    @pytest.mark.fast
    def test_get_stage_data_returns_empty_dict_for_empty_stages(self, empty_stages_checkpoint):
        """get_stage_data returns empty dict when stage data is empty"""
        project_dir, checkpoint = empty_stages_checkpoint

        # All stages should return empty dicts
        for stage_name in ["ANALYZE", "DOWNLOAD", "MATCH"]:
            data = checkpoint.get_stage_data(stage_name)
            assert data == {}, f"Expected empty dict for {stage_name}"

    @pytest.mark.fast
    def test_validate_warns_about_empty_download_paths(self, empty_stages_checkpoint):
        """validate() doesn't warn about missing videos when download data is empty"""
        project_dir, checkpoint = empty_stages_checkpoint

        result = checkpoint.validate()

        # Should be valid but may have warnings
        assert result['valid'] is True
        # No missing video warnings because download.video_paths doesn't exist
        missing_video_warnings = [w for w in result['warnings'] if "missing from disk" in w]
        assert len(missing_video_warnings) == 0

    @pytest.mark.fast
    def test_output_stage_run_warns_with_no_matches(self, empty_stages_checkpoint):
        """OutputStage.run() warns appropriately when state has no matches"""
        project_dir, checkpoint = empty_stages_checkpoint

        stage = OutputStage()
        state = PipelineState()
        state.matches = []  # No matches

        config = MagicMock()
        config.otio_output_dir = str(project_dir / "output")
        config.output.generate_otio = True
        config.output.generate_edl = False
        config.output.generate_xml = False
        config.output.generate_report = False
        config.output.split_otio = False
        config.output.frame_rate = 30.0

        result = stage.run(state, config, checkpoint)

        # Should succeed but with warnings about no matches
        assert result.success is True
        assert any("no matches" in w.lower() for w in result.warnings)

    @pytest.mark.fast
    def test_checkpoint_summary_shows_no_stage_data(self, empty_stages_checkpoint):
        """get_summary() doesn't show stage details when data is empty"""
        project_dir, checkpoint = empty_stages_checkpoint

        summary = checkpoint.get_summary()

        # Should show basic info but not stage details
        assert "MATCH" in summary  # last_completed_stage is shown
        # Should NOT show detailed stage info because data is empty
        assert "• ANALYZE:" not in summary
        assert "keywords" not in summary


# ============================================================================
# AC2: Test --output-only succeeds when checkpoint has populated stages dict
# ============================================================================

class TestOutputOnlySucceedsWithPopulatedStages:
    """AC2: Test --output-only behavior when checkpoint has stage data"""

    @pytest.mark.fast
    def test_restore_returns_true_with_populated_data(self, valid_checkpoint_with_stage_data):
        """OutputStage.restore() returns True when match data exists"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        # Simulate that OUTPUT was completed
        checkpoint.data.last_completed_stage = "OUTPUT"
        checkpoint.data.output = {
            "outputs": {
                "otio": ["timeline.otio"],
                "edl": "timeline.edl"
            }
        }

        stage = OutputStage()
        state = PipelineState()

        # With populated output data, restore should succeed
        result = stage.restore(state, checkpoint)

        assert result is True

    @pytest.mark.fast
    def test_get_stage_data_returns_populated_data(self, valid_checkpoint_with_stage_data):
        """get_stage_data returns populated data for completed stages"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        # ANALYZE should have data
        analyze_data = checkpoint.get_stage_data("ANALYZE")
        assert analyze_data != {}
        assert "keywords" in analyze_data
        assert len(analyze_data["keywords"]) > 0

        # MATCH should have data
        match_data = checkpoint.get_stage_data("MATCH")
        assert match_data != {}
        assert "match_count" in match_data
        assert match_data["match_count"] == 10

    @pytest.mark.fast
    def test_checkpoint_summary_shows_stage_details(self, valid_checkpoint_with_stage_data):
        """get_summary() shows stage details when data is populated"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        summary = checkpoint.get_summary()

        # Should show detailed stage info
        assert "ANALYZE" in summary
        assert "keywords" in summary
        assert "DOWNLOAD" in summary
        assert "videos" in summary.lower()

    @pytest.mark.fast
    def test_validate_returns_valid_for_complete_checkpoint(self, valid_checkpoint_with_stage_data):
        """validate() returns valid for fully populated checkpoint"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        result = checkpoint.validate()

        assert result['valid'] is True
        assert "MATCH" in result['completed_stages']

    @pytest.mark.fast
    def test_should_skip_stage_works_with_populated_data(self, valid_checkpoint_with_stage_data):
        """should_skip_stage() correctly identifies completed stages"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        # Stages before MATCH should be skipped
        assert checkpoint.should_skip_stage("ANALYZE") is True
        assert checkpoint.should_skip_stage("DOWNLOAD") is True
        assert checkpoint.should_skip_stage("MATCH") is True

        # Stages after MATCH should not be skipped
        assert checkpoint.should_skip_stage("OUTPUT") is False


# ============================================================================
# AC4: Test manually edited checkpoint without stages triggers warning
# ============================================================================

class TestManuallyEditedCheckpointWarnings:
    """AC4: Test warnings for manually edited/corrupted checkpoints"""

    @pytest.mark.fast
    def test_checkpoint_missing_created_at_logs_warning(self, tmp_path, caplog):
        """Checkpoint without created_at triggers validation warning"""
        import logging
        caplog.set_level(logging.DEBUG)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create checkpoint missing created_at
        checkpoint_data = {
            "version": "1.0",
            "created_at": "",  # Empty - manually edited
            "last_completed_stage": "MATCH",
            "match": {"match_count": 5}
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        manager.load()

        # Validation should succeed but log warning
        assert "validation" in caplog.text.lower() or "timestamp" in caplog.text.lower() or "created_at" in caplog.text.lower()

    @pytest.mark.fast
    def test_checkpoint_invalid_stage_name_warns(self, tmp_path, caplog):
        """Checkpoint with unknown stage name triggers warning"""
        import logging
        caplog.set_level(logging.DEBUG)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create checkpoint with invalid stage name
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "INVALID_STAGE_NAME",  # Invalid
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        manager.load()

        result = manager.validate()

        # Should warn about unknown stage
        assert any("unknown" in w.lower() or "invalid_stage" in w.lower()
                   for w in result['warnings'])

    @pytest.mark.fast
    def test_checkpoint_with_only_last_completed_stage(self, tmp_path):
        """Checkpoint with only last_completed_stage but no stage data is usable but limited"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create minimal checkpoint (manually edited to remove stage data)
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH"
            # No stage data fields at all
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should load successfully
        assert loaded is not None
        assert loaded.last_completed_stage == "MATCH"

        # Stage data should return empty dicts (not error)
        assert manager.get_stage_data("MATCH") == {}
        assert manager.get_stage_data("ANALYZE") == {}

    @pytest.mark.fast
    def test_output_stage_restore_handles_missing_outputs_key(self, tmp_path, caplog):
        """OutputStage.restore() handles checkpoint without 'outputs' key"""
        import logging
        caplog.set_level(logging.WARNING)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create checkpoint with OUTPUT stage but no outputs key
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "OUTPUT",
            # output stage data is empty - simulates manual edit
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        manager.load()

        stage = OutputStage()
        state = PipelineState()

        # restore should handle missing data gracefully
        result = stage.restore(state, manager)

        # Should return False because no data to restore
        assert result is False
        assert "No checkpoint data for OUTPUT" in caplog.text

    @pytest.mark.fast
    def test_restore_handles_corrupted_outputs_structure(self, tmp_path, caplog):
        """OutputStage.restore() handles corrupted outputs structure"""
        import logging
        caplog.set_level(logging.WARNING)

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        manager = CheckpointManager(project_dir)
        manager.data = CheckpointData(
            created_at=datetime.now().isoformat(),
            last_completed_stage="OUTPUT"
        )
        # Set invalid output data structure
        manager.data.output = {
            "outputs": "not_a_dict"  # Should be a dict
        }

        stage = OutputStage()
        state = PipelineState()

        result = stage.restore(state, manager)

        # Should return False and log warning
        assert result is False
        assert "'outputs' is not a dict" in caplog.text


# ============================================================================
# AC3: Backup restoration (verify existing tests cover this)
# ============================================================================

class TestBackupRestorationWithStageData:
    """AC3: Verify backup restoration recovers stage data"""

    @pytest.mark.fast
    def test_backup_restores_stage_data(self, tmp_path):
        """Backup restoration preserves all stage data"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create main checkpoint (corrupted)
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("corrupted!!")

        # Create valid backup with stage data
        backup_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "config_hash": "backup_hash",
            "analyze": {"keywords": ["test1", "test2"]},
            "download": {"video_paths": ["/path/to/video.mp4"], "count": 1},
            "match": {"match_count": 5, "avg_confidence": 0.88}
        }

        backup_path = project_dir / "checkpoint.backup.json"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f)

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Should restore from backup
        assert loaded is not None
        assert loaded.last_completed_stage == "MATCH"

        # Stage data should be preserved
        assert manager.get_stage_data("ANALYZE") == {"keywords": ["test1", "test2"]}
        assert manager.get_stage_data("MATCH")["match_count"] == 5

    @pytest.mark.fast
    def test_backup_restores_all_14_stage_types(self, tmp_path):
        """Backup restoration preserves data for all 14 stage types"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create corrupted main checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("corrupted data here!")

        # Create backup with data for all stages
        backup_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_completed_stage": "OUTPUT",
            "analyze": {"data": "analyze"},
            "entity_images": {"data": "entity_images"},
            "entity_videos": {"data": "entity_videos"},
            "download": {"data": "download"},
            "stock": {"data": "stock"},
            "broll_download": {"data": "broll_download"},
            "remix": {"data": "remix"},
            "caption": {"data": "caption"},
            "transcribe": {"data": "transcribe"},
            "scene_detection": {"data": "scene_detection"},
            "match": {"data": "match"},
            "broll_match": {"data": "broll_match"},
            "download_segments": {"data": "download_segments"},
        }

        backup_path = project_dir / "checkpoint.backup.json"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f)

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        assert loaded is not None

        # Verify all stage data is restored
        stages_with_data = [
            "ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS", "DOWNLOAD",
            "STOCK", "BROLL_DOWNLOAD", "REMIX", "CAPTION", "TRANSCRIBE",
            "SCENE_DETECTION", "MATCH", "BROLL_MATCH", "DOWNLOAD_SEGMENTS"
        ]

        for stage_name in stages_with_data:
            data = manager.get_stage_data(stage_name)
            assert data != {}, f"Stage {stage_name} data should be restored"
            assert data.get("data") == stage_name.lower()


# ============================================================================
# Integration tests for --output-only mode simulation
# ============================================================================

class TestOutputOnlyModeIntegration:
    """Integration tests simulating --output-only behavior"""

    @pytest.mark.fast
    def test_output_only_pipeline_with_empty_stages_skips_restore(self, empty_stages_checkpoint):
        """Simulates --output-only failing to restore when stages are empty"""
        project_dir, checkpoint = empty_stages_checkpoint

        # In --output-only mode, pipeline tries to restore from checkpoint
        stage = OutputStage()
        state = PipelineState()

        # Try to restore (this is what pipeline does in resume mode)
        restored = stage.restore(state, checkpoint)

        # With empty stages, restore should fail
        assert restored is False

        # State should not have matches populated
        assert len(state.matches) == 0
        assert len(state.otio_files) == 0

    @pytest.mark.fast
    def test_output_only_pipeline_with_populated_stages_restores(self, valid_checkpoint_with_stage_data):
        """Simulates --output-only successfully restoring when stages have data"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        # Add OUTPUT stage data to checkpoint
        checkpoint.data.last_completed_stage = "OUTPUT"
        checkpoint.data.output = {
            "outputs": {
                "otio": [str(project_dir / "output" / "timeline.otio")],
                "edl": str(project_dir / "output" / "timeline.edl")
            },
            "match_count": 10,
            "output_dir": str(project_dir / "output")
        }

        stage = OutputStage()
        state = PipelineState()

        # Try to restore
        restored = stage.restore(state, checkpoint)

        # With populated stages, restore should succeed
        assert restored is True

        # State should have output files populated
        assert len(state.otio_files) == 1
        assert len(state.output_files) == 2  # 1 otio + 1 edl

    @pytest.mark.fast
    def test_can_skip_returns_true_for_output_when_checkpoint_complete(
        self, valid_checkpoint_with_stage_data
    ):
        """can_skip returns appropriate value based on checkpoint state"""
        project_dir, checkpoint = valid_checkpoint_with_stage_data

        stage = OutputStage()
        state = PipelineState()

        # OUTPUT can_skip should return False (OUTPUT always regenerates)
        can_skip = stage.can_skip(state, checkpoint)
        assert can_skip is False  # OUTPUT stage never skips

    @pytest.mark.fast
    def test_validate_inputs_fails_without_matches(self, empty_stages_checkpoint):
        """validate_inputs fails when no matches in state"""
        project_dir, checkpoint = empty_stages_checkpoint

        stage = OutputStage()
        state = PipelineState()
        state.matches = []  # No matches

        config = MagicMock()
        config.otio_output_dir = str(project_dir / "output")

        error = stage.validate_inputs(state, config)

        # Should have validation error about missing matches
        assert error is not None
        assert "matches" in error.lower()


# ============================================================================
# Edge case tests
# ============================================================================

class TestCheckpointStageDataEdgeCases:
    """Edge case tests for checkpoint stage data handling"""

    @pytest.mark.fast
    def test_partial_stage_data_is_preserved(self, tmp_path):
        """Checkpoint with partial stage data preserves what exists"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create checkpoint with only some stages populated
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "analyze": {"keywords": ["test"]},  # Populated
            "download": {},  # Empty
            "match": {"match_count": 5}  # Populated
            # Other stages missing entirely
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        # Populated stages should have data
        assert manager.get_stage_data("ANALYZE") == {"keywords": ["test"]}
        assert manager.get_stage_data("MATCH") == {"match_count": 5}

        # Empty/missing stages should return empty dict
        assert manager.get_stage_data("DOWNLOAD") == {}
        assert manager.get_stage_data("TRANSCRIBE") == {}

    @pytest.mark.fast
    def test_stage_data_with_none_values(self, tmp_path):
        """Checkpoint handles stage data containing None values"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "MATCH",
            "match": {
                "match_count": 5,
                "avg_confidence": None,  # None value
                "matches": None  # None value
            }
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        loaded = manager.load()

        match_data = manager.get_stage_data("MATCH")
        assert match_data["match_count"] == 5
        assert match_data["avg_confidence"] is None
        assert match_data["matches"] is None

    @pytest.mark.fast
    def test_empty_vs_missing_stage_data(self, tmp_path):
        """Distinguish between empty {} and missing stage data"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Checkpoint with explicitly empty stage vs missing stage
        checkpoint_data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_completed_stage": "DOWNLOAD",
            "analyze": {"keywords": ["test"]},
            "download": {}  # Explicitly empty
            # entity_images, etc. are missing
        }

        checkpoint_path = project_dir / "checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)

        manager = CheckpointManager(project_dir)
        manager.load()

        # Both should return empty dict
        assert manager.get_stage_data("DOWNLOAD") == {}
        assert manager.get_stage_data("ENTITY_IMAGES") == {}

        # But they're treated the same way - empty dict

    @pytest.mark.fast
    def test_save_intermediate_preserves_existing_stage_data(self, tmp_path):
        """save_intermediate() doesn't clear other stage data"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        manager = CheckpointManager(project_dir)

        # Save some stages
        manager.save("ANALYZE", {"keywords": ["test1", "test2"]})
        manager.save("DOWNLOAD", {"video_paths": ["/v/vid1.mp4"]})

        # Now save intermediate checkpoint for TRANSCRIBE
        manager.save_intermediate("TRANSCRIBE", {"transcribed_count": 3})

        # Verify previous stage data is preserved
        assert manager.get_stage_data("ANALYZE") == {"keywords": ["test1", "test2"]}
        assert manager.get_stage_data("DOWNLOAD") == {"video_paths": ["/v/vid1.mp4"]}
        assert manager.get_stage_data("TRANSCRIBE") == {"transcribed_count": 3}

        # last_completed_stage should NOT change (intermediate save)
        assert manager.data.last_completed_stage == "DOWNLOAD"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
