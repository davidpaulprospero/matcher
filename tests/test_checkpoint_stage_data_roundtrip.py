"""
Tests for checkpoint stage data round-trip (US-006, Sprint 15).

Covers:
- AC1: save/load round-trip preserves stage data for all 7 stage types
- AC2: get_stage_data() returns empty dict for stages not yet completed
- AC3: should_skip_stage() correctly skips completed stages during resume
- AC4: config_hash mismatch detection via validate()
- AC5: voiceover_hash comparison detects voiceover file changes

Uses the 7-stage pipeline structure (v4.0):
ANALYZE → VIDEO_SEARCH → CAPTION → MATCH → ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
"""

import json
import hashlib
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER


class TestSaveLoadRoundTrip:
    """AC1: save/load round-trip preserves stage data for all 7 stage types."""

    # Sample data for each stage (7-stage pipeline, lowercase keys matching CheckpointData fields)
    STAGE_DATA = {
        "ANALYZE": {"keywords": ["travel", "nature"], "segment_count": 20, "topic": "wildlife"},
        "VIDEO_SEARCH": {"video_ids": ["vid1", "vid2"], "count": 2},
        "CAPTION": {"caption_count": 8, "languages": ["en", "es"]},
        "MATCH": {"match_count": 18, "avg_confidence": 0.85, "matches": [{"id": 1}]},
        "ITERATIVE_MATCH": {"passes": 2, "gaps_filled": 5},
        "DOWNLOAD_SEGMENTS": {"segments_downloaded": 18, "total_size_mb": 450.5},
        "OUTPUT": {"otio_path": "/output/timeline.otio", "edl_path": "/output/timeline.edl"},
    }

    @pytest.mark.fast
    def test_round_trip_all_7_stages(self, tmp_path):
        """Save data for all 7 stages sequentially, load, verify each stage dict matches."""
        manager = CheckpointManager(tmp_path, config_hash="test_hash")

        # Save each stage with its data
        for stage in STAGE_ORDER:
            data = self.STAGE_DATA.get(stage, {})
            manager.save(stage, data)

        # Create a new manager and load the checkpoint
        manager2 = CheckpointManager(tmp_path, config_hash="test_hash")
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.last_completed_stage == STAGE_ORDER[-1]  # OUTPUT

        # Verify each stage's data that has a corresponding CheckpointData field
        for stage in STAGE_ORDER:
            stage_key = stage.lower()
            expected = self.STAGE_DATA.get(stage, {})
            if hasattr(loaded, stage_key):
                actual = getattr(loaded, stage_key)
                assert actual == expected, f"Stage {stage} data mismatch: {actual} != {expected}"

    @pytest.mark.fast
    def test_round_trip_preserves_metadata(self, tmp_path):
        """Verify config_hash, voiceover_path, and timestamps survive round-trip."""
        manager = CheckpointManager(tmp_path, config_hash="cfg_abc123")
        manager.save("ANALYZE", {"keywords": ["test"]})
        manager.data.voiceover_path = "/project/script.srt"
        manager.data.voiceover_hash = "vo_deadbeef"
        manager._atomic_save()

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.config_hash == "cfg_abc123"
        assert loaded.voiceover_path == "/project/script.srt"
        assert loaded.voiceover_hash == "vo_deadbeef"
        assert loaded.created_at != ""
        assert loaded.updated_at != ""

    @pytest.mark.fast
    def test_round_trip_single_stage(self, tmp_path):
        """Verify round-trip works for a single stage save."""
        manager = CheckpointManager(tmp_path)
        manager.save("MATCH", {"match_count": 25, "avg_confidence": 0.92})

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.last_completed_stage == "MATCH"
        assert loaded.match == {"match_count": 25, "avg_confidence": 0.92}

    @pytest.mark.fast
    def test_round_trip_nested_stage_data(self, tmp_path):
        """Verify deeply nested stage data survives round-trip."""
        nested_data = {
            "results": [
                {"video_id": "abc", "score": 0.95, "metadata": {"channel": "NatGeo"}},
                {"video_id": "def", "score": 0.82, "metadata": {"channel": "BBC"}},
            ],
            "stats": {"mean": 0.885, "std": 0.065},
        }
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", nested_data)

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        assert loaded.analyze == nested_data
        assert loaded.analyze["results"][0]["metadata"]["channel"] == "NatGeo"

    @pytest.mark.fast
    def test_round_trip_empty_stage_data(self, tmp_path):
        """Verify saving with None/empty stage_data preserves empty dict."""
        manager = CheckpointManager(tmp_path)
        manager.save("VIDEO_SEARCH", None)  # No stage data

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        assert loaded is not None
        assert loaded.last_completed_stage == "VIDEO_SEARCH"
        # Empty dict is the default for unset stage data
        assert loaded.video_search == {}

    @pytest.mark.fast
    def test_round_trip_overwrite_stage_data(self, tmp_path):
        """Verify saving same stage twice overwrites with latest data."""
        manager = CheckpointManager(tmp_path)
        manager.save("MATCH", {"match_count": 10, "avg_confidence": 0.7})
        manager.save("MATCH", {"match_count": 20, "avg_confidence": 0.9})

        manager2 = CheckpointManager(tmp_path)
        loaded = manager2.load()

        assert loaded.match == {"match_count": 20, "avg_confidence": 0.9}


class TestGetStageData:
    """AC2: get_stage_data() returns empty dict for stages not yet completed."""

    @pytest.mark.fast
    def test_returns_empty_dict_before_any_stage_runs(self, tmp_path):
        """Verify no KeyError when accessing stage data before any stage runs."""
        manager = CheckpointManager(tmp_path)
        # No data loaded at all
        result = manager.get_stage_data("ANALYZE")
        assert result == {}

    @pytest.mark.fast
    def test_returns_empty_dict_for_uncompleted_stage(self, tmp_path):
        """Verify empty dict for stage that hasn't been saved yet."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        # VIDEO_SEARCH was never saved
        result = manager.get_stage_data("VIDEO_SEARCH")
        assert result == {}

    @pytest.mark.fast
    def test_returns_data_for_completed_stage(self, tmp_path):
        """Verify correct data returned for a saved stage."""
        manager = CheckpointManager(tmp_path)
        expected = {"keywords": ["nature", "travel"], "segment_count": 15}
        manager.save("ANALYZE", expected)

        result = manager.get_stage_data("ANALYZE")
        assert result == expected

    @pytest.mark.fast
    def test_returns_empty_dict_for_unknown_stage(self, tmp_path):
        """Verify empty dict for a stage name not in STAGE_ORDER."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Unknown stage name
        result = manager.get_stage_data("NONEXISTENT_STAGE")
        assert result == {}

    @pytest.mark.fast
    def test_returns_empty_dict_for_future_stages(self, tmp_path):
        """Verify all stages after last_completed return empty dicts."""
        manager = CheckpointManager(tmp_path)
        manager.save("VIDEO_SEARCH", {"video_ids": ["vid1"]})

        # All stages after VIDEO_SEARCH should return empty
        for stage in ["CAPTION", "MATCH", "OUTPUT"]:
            result = manager.get_stage_data(stage)
            assert result == {}, f"Expected empty dict for {stage}, got {result}"

    @pytest.mark.fast
    def test_all_7_stages_return_empty_when_no_data(self, tmp_path):
        """Verify every stage in STAGE_ORDER returns empty dict on fresh manager."""
        manager = CheckpointManager(tmp_path)
        for stage in STAGE_ORDER:
            result = manager.get_stage_data(stage)
            assert result == {}, f"Expected empty dict for {stage}"


class TestShouldSkipStage:
    """AC3: should_skip_stage() correctly skips completed stages during resume."""

    @pytest.mark.fast
    def test_skips_stages_before_last_completed(self, tmp_path):
        """Verify returns False for stages before last_completed_stage, True for stages after."""
        manager = CheckpointManager(tmp_path)
        # Set last completed to VIDEO_SEARCH (index 1 in STAGE_ORDER)
        manager.save("VIDEO_SEARCH", {"count": 5})

        vs_idx = STAGE_ORDER.index("VIDEO_SEARCH")

        # Stages up to and including VIDEO_SEARCH should be skipped
        for stage in STAGE_ORDER[:vs_idx + 1]:
            assert manager.should_skip_stage(stage) is True, \
                f"Expected should_skip_stage('{stage}') to be True (completed)"

        # Stages after VIDEO_SEARCH should NOT be skipped
        for stage in STAGE_ORDER[vs_idx + 1:]:
            assert manager.should_skip_stage(stage) is False, \
                f"Expected should_skip_stage('{stage}') to be False (not yet completed)"

    @pytest.mark.fast
    def test_no_checkpoint_skips_nothing(self, tmp_path):
        """Verify all stages run when no checkpoint exists."""
        manager = CheckpointManager(tmp_path)
        # No data, no checkpoint

        for stage in STAGE_ORDER:
            assert manager.should_skip_stage(stage) is False, \
                f"Expected should_skip_stage('{stage}') to be False (no checkpoint)"

    @pytest.mark.fast
    def test_first_stage_completed(self, tmp_path):
        """Verify only first stage skipped when it's the only completed one."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        assert manager.should_skip_stage("ANALYZE") is True
        # All subsequent stages should run
        for stage in STAGE_ORDER[1:]:
            assert manager.should_skip_stage(stage) is False

    @pytest.mark.fast
    def test_last_stage_completed(self, tmp_path):
        """Verify all stages skipped when last stage (OUTPUT) is completed."""
        manager = CheckpointManager(tmp_path)
        manager.save("OUTPUT", {"otio_path": "/output/timeline.otio"})

        for stage in STAGE_ORDER:
            assert manager.should_skip_stage(stage) is True, \
                f"Expected should_skip_stage('{stage}') to be True (pipeline complete)"

    @pytest.mark.fast
    def test_mid_pipeline_resume(self, tmp_path):
        """Verify correct split for mid-pipeline resume from CAPTION."""
        manager = CheckpointManager(tmp_path)
        manager.save("CAPTION", {"caption_count": 10})

        caption_idx = STAGE_ORDER.index("CAPTION")

        # Count skipped and non-skipped
        skipped = [s for s in STAGE_ORDER if manager.should_skip_stage(s)]
        to_run = [s for s in STAGE_ORDER if not manager.should_skip_stage(s)]

        assert len(skipped) == caption_idx + 1
        assert len(to_run) == len(STAGE_ORDER) - caption_idx - 1
        assert "CAPTION" in skipped
        assert "MATCH" in to_run  # Next stage after CAPTION

    @pytest.mark.fast
    def test_unknown_stage_returns_false(self, tmp_path):
        """Verify unknown stage name returns False (don't skip unknown stages)."""
        manager = CheckpointManager(tmp_path)
        manager.save("MATCH", {"match_count": 10})

        assert manager.should_skip_stage("UNKNOWN_STAGE") is False


class TestConfigHashMismatch:
    """AC4: config_hash mismatch detection via validate()."""

    @pytest.mark.fast
    def test_validate_warns_on_config_hash_mismatch(self, tmp_path):
        """Verify validate() warns when checkpoint config_hash differs from current."""
        # Save checkpoint with one config hash
        manager = CheckpointManager(tmp_path, config_hash="original_hash")
        manager.save("VIDEO_SEARCH", {"count": 5})

        # Load with a different config hash (simulating config change between runs)
        manager2 = CheckpointManager(tmp_path, config_hash="changed_hash")
        manager2.load()

        result = manager2.validate()

        assert result['valid'] is True  # Still valid, just warned
        assert len(result['warnings']) >= 1
        assert any("changed" in w.lower() or "config" in w.lower() for w in result['warnings'])

    @pytest.mark.fast
    def test_validate_no_warning_when_config_hash_matches(self, tmp_path):
        """Verify no warning when config hash matches between save and load."""
        manager = CheckpointManager(tmp_path, config_hash="same_hash")
        manager.save("VIDEO_SEARCH", {"count": 5})

        manager2 = CheckpointManager(tmp_path, config_hash="same_hash")
        manager2.load()

        result = manager2.validate()

        config_warnings = [w for w in result['warnings']
                           if "config" in w.lower() or "changed" in w.lower()]
        assert len(config_warnings) == 0

    @pytest.mark.fast
    def test_validate_no_warning_when_no_original_hash(self, tmp_path):
        """Verify no warning when checkpoint has no config_hash (old checkpoint)."""
        # Save with no config hash (empty string)
        manager = CheckpointManager(tmp_path, config_hash="")
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Load with a config hash
        manager2 = CheckpointManager(tmp_path, config_hash="new_hash")
        manager2.load()

        result = manager2.validate()

        config_warnings = [w for w in result['warnings']
                           if "config" in w.lower()]
        assert len(config_warnings) == 0

    @pytest.mark.fast
    def test_validate_no_warning_when_no_current_hash(self, tmp_path):
        """Verify no warning when current manager has no config_hash."""
        manager = CheckpointManager(tmp_path, config_hash="saved_hash")
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Load with no config hash
        manager2 = CheckpointManager(tmp_path, config_hash="")
        manager2.load()

        result = manager2.validate()

        config_warnings = [w for w in result['warnings']
                           if "config" in w.lower()]
        assert len(config_warnings) == 0

    @pytest.mark.fast
    def test_validate_resume_from_correct_after_hash_mismatch(self, tmp_path):
        """Verify resume_from is still correct even with config hash mismatch."""
        manager = CheckpointManager(tmp_path, config_hash="hash_v1")
        manager.save("MATCH", {"match_count": 10})

        manager2 = CheckpointManager(tmp_path, config_hash="hash_v2")
        manager2.load()

        result = manager2.validate()

        # Despite mismatch warning, resume_from should be the next stage
        match_idx = STAGE_ORDER.index("MATCH")
        expected_next = STAGE_ORDER[match_idx + 1]
        assert result['resume_from'] == expected_next


class TestVoiceoverHashComparison:
    """AC5: voiceover_hash comparison detects when voiceover file changed."""

    @pytest.mark.fast
    def test_validate_warns_on_voiceover_hash_mismatch(self, tmp_path):
        """Verify checkpoint detects when voiceover file changed between runs."""
        # Create original voiceover file
        vo_file = tmp_path / "script.srt"
        vo_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nOriginal text")

        manager = CheckpointManager(tmp_path, config_hash="hash1")
        manager.save("ANALYZE", {"keywords": ["test"]})
        manager.set_voiceover(str(vo_file))
        manager._atomic_save()

        # Modify voiceover file (simulating user edit between runs)
        vo_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nModified text with changes")

        # Load checkpoint and validate with changed voiceover
        manager2 = CheckpointManager(tmp_path, config_hash="hash1")
        manager2.load()

        result = manager2.validate(voiceover_path=str(vo_file))

        assert any("voiceover" in w.lower() for w in result['warnings']), \
            f"Expected voiceover change warning, got warnings: {result['warnings']}"

    @pytest.mark.fast
    def test_validate_no_warning_when_voiceover_unchanged(self, tmp_path):
        """Verify no warning when voiceover file hasn't changed."""
        vo_file = tmp_path / "script.srt"
        vo_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nSame text")

        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})
        manager.set_voiceover(str(vo_file))
        manager._atomic_save()

        # Reload with same file
        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.validate(voiceover_path=str(vo_file))

        vo_warnings = [w for w in result['warnings'] if "voiceover" in w.lower()]
        assert len(vo_warnings) == 0

    @pytest.mark.fast
    def test_validate_no_warning_when_no_voiceover_path(self, tmp_path):
        """Verify no warning when no voiceover_path is provided to validate."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})
        manager.data.voiceover_hash = "some_old_hash"
        manager._atomic_save()

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        # Validate without voiceover_path
        result = manager2.validate()

        vo_warnings = [w for w in result['warnings'] if "voiceover" in w.lower()]
        assert len(vo_warnings) == 0

    @pytest.mark.fast
    def test_validate_no_warning_when_checkpoint_has_no_hash(self, tmp_path):
        """Verify no warning when checkpoint has empty voiceover_hash."""
        vo_file = tmp_path / "script.srt"
        vo_file.write_text("1\n00:00:01,000 --> 00:00:05,000\nText")

        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})
        # Don't set voiceover - hash will be empty

        manager2 = CheckpointManager(tmp_path)
        manager2.load()

        result = manager2.validate(voiceover_path=str(vo_file))

        vo_warnings = [w for w in result['warnings'] if "voiceover" in w.lower()]
        assert len(vo_warnings) == 0

    @pytest.mark.fast
    def test_voiceover_hash_stored_correctly(self, tmp_path):
        """Verify set_voiceover() computes and stores hash correctly."""
        vo_file = tmp_path / "script.srt"
        content = "1\n00:00:01,000 --> 00:00:05,000\nTest content for hashing"
        vo_file.write_text(content)

        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})
        manager.set_voiceover(str(vo_file))

        # Compute expected hash from raw bytes (same as _hash_file reads 'rb')
        expected_hash = hashlib.md5(vo_file.read_bytes()).hexdigest()[:16]

        assert manager.data.voiceover_hash == expected_hash
        assert manager.data.voiceover_path == str(vo_file)

    @pytest.mark.fast
    def test_voiceover_hash_missing_file_returns_empty(self, tmp_path):
        """Verify set_voiceover() handles missing file gracefully."""
        manager = CheckpointManager(tmp_path)
        manager.save("ANALYZE", {"keywords": ["test"]})

        # Set voiceover with non-existent file
        manager.set_voiceover(str(tmp_path / "nonexistent.srt"))

        assert manager.data.voiceover_hash == ""
        assert "nonexistent.srt" in manager.data.voiceover_path
