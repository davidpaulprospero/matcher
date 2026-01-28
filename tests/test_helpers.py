"""
Tests for the test helpers module.

US-007: Add test helper module for common assertion patterns
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock


class TestAssertValidMatchResult:
    """Tests for assert_valid_match_result helper."""

    def test_valid_basic_result(self):
        """Valid minimal match result passes."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.85,
            "video_file": "video.mp4",
            "start": 0.0,
            "end": 5.0,
        }
        # Should not raise
        assert_valid_match_result(result)

    def test_missing_segment_index(self):
        """Missing segment_index raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {"confidence": 0.85, "video_file": "video.mp4"}
        with pytest.raises(AssertionError, match="missing 'segment_index'"):
            assert_valid_match_result(result)

    def test_missing_confidence(self):
        """Missing confidence raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {"segment_index": 0, "video_file": "video.mp4"}
        with pytest.raises(AssertionError, match="missing 'confidence'"):
            assert_valid_match_result(result)

    def test_confidence_below_min(self):
        """Confidence below minimum raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.3,
            "video_file": "video.mp4",
        }
        with pytest.raises(AssertionError, match="below minimum threshold"):
            assert_valid_match_result(result, min_confidence=0.5)

    def test_confidence_at_min_passes(self):
        """Confidence at minimum threshold passes."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.5,
            "video_file": "video.mp4",
        }
        # Should not raise
        assert_valid_match_result(result, min_confidence=0.5)

    def test_confidence_out_of_range(self):
        """Confidence outside [0, 1] range raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 1.5,
            "video_file": "video.mp4",
        }
        with pytest.raises(AssertionError, match="must be in range"):
            assert_valid_match_result(result)

    def test_empty_video_file(self):
        """Empty video_file raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.85,
            "video_file": "",
        }
        with pytest.raises(AssertionError, match="non-empty string"):
            assert_valid_match_result(result)

    def test_video_file_not_required(self):
        """Missing video_file passes when not required."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.85,
        }
        # Should not raise
        assert_valid_match_result(result, require_video_file=False, require_timing=False)

    def test_invalid_strategy(self):
        """Invalid strategy raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.85,
            "video_file": "video.mp4",
            "strategy": "unknown",
        }
        with pytest.raises(AssertionError, match="not in valid strategies"):
            assert_valid_match_result(
                result,
                require_strategy=True,
                valid_strategies=["embedding", "llm"],
            )

    def test_valid_strategy(self):
        """Valid strategy passes."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.85,
            "video_file": "video.mp4",
            "strategy": "embedding",
        }
        assert_valid_match_result(
            result,
            require_strategy=True,
            valid_strategies=["embedding", "llm"],
        )


class TestAssertCheckpointConsistent:
    """Tests for assert_checkpoint_consistent helper."""

    def test_valid_basic_checkpoint(self):
        """Valid minimal checkpoint passes."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "MATCH",
        }
        # Should not raise
        assert_checkpoint_consistent(checkpoint)

    def test_missing_version(self):
        """Missing version raises assertion when required."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {"last_completed_stage": "MATCH"}
        with pytest.raises(AssertionError, match="missing 'version'"):
            assert_checkpoint_consistent(checkpoint, require_version=True)

    def test_missing_last_completed_stage(self):
        """Missing last_completed_stage raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {"version": "1.0"}
        with pytest.raises(AssertionError, match="missing 'last_completed_stage'"):
            assert_checkpoint_consistent(checkpoint)

    def test_invalid_stage_name(self):
        """Invalid stage name raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "INVALID_STAGE",
        }
        with pytest.raises(AssertionError, match="Invalid stage"):
            assert_checkpoint_consistent(checkpoint)

    def test_expected_stage_mismatch(self):
        """Stage mismatch with expected raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "DOWNLOAD",
        }
        with pytest.raises(AssertionError, match="Expected stage 'MATCH'"):
            assert_checkpoint_consistent(checkpoint, expected_stage="MATCH")

    def test_require_stages_with_empty(self):
        """Empty stages dict raises assertion when required."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "MATCH",
            "stages": {},
        }
        with pytest.raises(AssertionError, match="no stage data"):
            assert_checkpoint_consistent(checkpoint, require_stages=True)

    def test_require_stages_with_populated(self):
        """Populated stages dict passes."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "MATCH",
            "stages": {
                "match": {"matches": [], "match_count": 0}
            },
        }
        # Should not raise
        assert_checkpoint_consistent(checkpoint, require_stages=True)

    def test_min_matches_requirement(self):
        """Insufficient matches raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "MATCH",
            "stages": {
                "match": {"matches": [{"segment_index": 0}], "match_count": 1}
            },
        }
        with pytest.raises(AssertionError, match="has 1 matches, expected at least 5"):
            assert_checkpoint_consistent(checkpoint, min_matches=5)


class TestAssertConfigValid:
    """Tests for assert_config_valid helper."""

    def test_valid_mock_config(self):
        """Valid mock config passes."""
        from tests.helpers import assert_config_valid

        config = Mock(spec=[])  # Empty spec prevents auto-creating validate()
        config.matching = Mock(
            min_confidence=0.5,
            primary_provider="gemini",
            embedding_candidates=50,
            high_confidence_threshold=0.85,
        )
        config.transcription = Mock(max_workers=4)
        config.keyword = Mock(max_keywords=10)

        errors = assert_config_valid(config)
        assert not errors

    def test_invalid_confidence_range(self):
        """Invalid confidence range returns error."""
        from tests.helpers import assert_config_valid

        config = Mock(spec=[])  # Empty spec prevents auto-creating validate()
        config.matching = Mock(
            min_confidence=1.5,  # Invalid - must be 0-1
            primary_provider="gemini",
            embedding_candidates=50,
        )

        errors = assert_config_valid(config, check_constraints=False)
        assert any("min_confidence" in e for e in errors)

    def test_invalid_provider(self):
        """Invalid provider returns error."""
        from tests.helpers import assert_config_valid

        config = Mock(spec=[])  # Empty spec prevents auto-creating validate()
        config.matching = Mock(
            min_confidence=0.5,
            primary_provider="invalid_provider",
            embedding_candidates=50,
            high_confidence_threshold=0.85,  # Required for constraint check
        )

        errors = assert_config_valid(config)
        assert any("primary_provider" in e for e in errors)

    def test_constraint_violation(self):
        """Constraint violation returns error."""
        from tests.helpers import assert_config_valid

        config = Mock(spec=[])  # Empty spec prevents auto-creating validate()
        config.matching = Mock(
            min_confidence=0.9,
            high_confidence_threshold=0.8,  # Less than min_confidence
            primary_provider="gemini",
            embedding_candidates=50,
        )

        errors = assert_config_valid(config, check_constraints=True)
        assert any("should be <=" in e for e in errors)

    def test_allowed_errors_filtering(self):
        """Allowed errors are filtered out."""
        from tests.helpers import assert_config_valid

        config = Mock(spec=[])  # Empty spec prevents auto-creating validate()
        config.matching = Mock(
            min_confidence=1.5,  # Would be error
            primary_provider="gemini",
            embedding_candidates=50,
            high_confidence_threshold=0.85,
        )

        errors = assert_config_valid(
            config,
            allowed_errors=["min_confidence", "should be <="],  # Filter both errors
        )
        assert not errors  # All filtered out

    def test_none_config_raises(self):
        """None config raises assertion."""
        from tests.helpers import assert_config_valid

        with pytest.raises(AssertionError, match="Config is None"):
            assert_config_valid(None)

    def test_config_with_validate_method(self):
        """Config with validate() method uses it."""
        from tests.helpers import assert_config_valid

        config = Mock(spec=['validate', 'matching'])
        config.validate.return_value = ["error1", "error2"]
        config.matching = Mock()

        errors = assert_config_valid(config)
        assert errors == ["error1", "error2"]
        config.validate.assert_called_once()


class TestAssertValidOtioTimeline:
    """Tests for assert_valid_otio_timeline helper."""

    def test_rejects_non_timeline(self):
        """Non-Timeline object raises assertion."""
        from tests.helpers import assert_valid_otio_timeline

        with pytest.raises(AssertionError, match="Expected opentimelineio"):
            assert_valid_otio_timeline({"not": "a timeline"})

    def test_rejects_string(self):
        """String raises assertion."""
        from tests.helpers import assert_valid_otio_timeline

        with pytest.raises(AssertionError, match="Expected opentimelineio"):
            assert_valid_otio_timeline("not a timeline")

    @pytest.mark.skipif(
        not pytest.importorskip("opentimelineio", reason="opentimelineio not installed"),
        reason="opentimelineio not installed"
    )
    def test_valid_timeline_passes(self):
        """Valid OTIO timeline passes validation."""
        from tests.helpers import assert_valid_otio_timeline
        import opentimelineio as otio

        # Create a simple valid timeline
        timeline = otio.schema.Timeline(name="Test Timeline")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

        # Add a clip with valid source_range and media_reference
        clip = otio.schema.Clip(
            name="Test Clip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30),
                duration=otio.opentime.RationalTime(30, 30),
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="file:///path/to/video.mp4",
            ),
        )
        track.append(clip)
        timeline.tracks.append(track)

        # Should not raise
        assert_valid_otio_timeline(timeline)


class TestUtilityAssertions:
    """Tests for utility assertion helpers."""

    def test_assert_file_exists_missing(self, tmp_path):
        """Missing file raises assertion."""
        from tests.helpers import assert_file_exists

        fake_path = tmp_path / "nonexistent.txt"
        with pytest.raises(AssertionError, match="not found"):
            assert_file_exists(fake_path, description="Test file")

    def test_assert_file_exists_is_dir(self, tmp_path):
        """Directory instead of file raises assertion."""
        from tests.helpers import assert_file_exists

        with pytest.raises(AssertionError, match="is not a file"):
            assert_file_exists(tmp_path, description="Test file")

    def test_assert_file_exists_valid(self, tmp_path):
        """Existing file passes."""
        from tests.helpers import assert_file_exists

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should not raise
        assert_file_exists(test_file)

    def test_assert_dir_exists_missing(self, tmp_path):
        """Missing directory raises assertion."""
        from tests.helpers import assert_dir_exists

        fake_path = tmp_path / "nonexistent"
        with pytest.raises(AssertionError, match="not found"):
            assert_dir_exists(fake_path)

    def test_assert_dir_exists_is_file(self, tmp_path):
        """File instead of directory raises assertion."""
        from tests.helpers import assert_dir_exists

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(AssertionError, match="is not a directory"):
            assert_dir_exists(test_file)

    def test_assert_json_structure_missing_key(self):
        """Missing required key raises assertion."""
        from tests.helpers import assert_json_structure

        data = {"a": 1, "b": 2}
        with pytest.raises(AssertionError, match="Missing required key: c"):
            assert_json_structure(data, required_keys=["a", "b", "c"])

    def test_assert_json_structure_wrong_type(self):
        """Wrong type raises assertion."""
        from tests.helpers import assert_json_structure

        data = {"count": "not a number"}
        with pytest.raises(AssertionError, match="should be int"):
            assert_json_structure(data, type_checks={"count": int})

    def test_assert_json_structure_valid(self):
        """Valid structure passes."""
        from tests.helpers import assert_json_structure

        data = {"name": "test", "count": 5, "items": []}

        # Should not raise
        assert_json_structure(
            data,
            required_keys=["name", "count"],
            type_checks={"name": str, "count": int, "items": list},
        )
