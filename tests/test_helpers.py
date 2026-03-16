"""
Tests for the test helpers module.

US-007: Add test helper module for common assertion patterns
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock


class TestAssertValidMatchResult:
    """Tests for assert_valid_match_result helper."""

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_missing_segment_index(self):
        """Missing segment_index raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {"confidence": 0.85, "video_file": "video.mp4"}
        with pytest.raises(AssertionError, match="missing 'segment_index'"):
            assert_valid_match_result(result)

    @pytest.mark.fast
    def test_missing_confidence(self):
        """Missing confidence raises assertion."""
        from tests.helpers import assert_valid_match_result

        result = {"segment_index": 0, "video_file": "video.mp4"}
        with pytest.raises(AssertionError, match="missing 'confidence'"):
            assert_valid_match_result(result)

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_video_file_not_required(self):
        """Missing video_file passes when not required."""
        from tests.helpers import assert_valid_match_result

        result = {
            "segment_index": 0,
            "confidence": 0.85,
        }
        # Should not raise
        assert_valid_match_result(result, require_video_file=False, require_timing=False)

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_valid_basic_checkpoint(self):
        """Valid minimal checkpoint passes."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "MATCH",
        }
        # Should not raise
        assert_checkpoint_consistent(checkpoint)

    @pytest.mark.fast
    def test_missing_version(self):
        """Missing version raises assertion when required."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {"last_completed_stage": "MATCH"}
        with pytest.raises(AssertionError, match="missing 'version'"):
            assert_checkpoint_consistent(checkpoint, require_version=True)

    @pytest.mark.fast
    def test_missing_last_completed_stage(self):
        """Missing last_completed_stage raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {"version": "1.0"}
        with pytest.raises(AssertionError, match="missing 'last_completed_stage'"):
            assert_checkpoint_consistent(checkpoint)

    @pytest.mark.fast
    def test_invalid_stage_name(self):
        """Invalid stage name raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "INVALID_STAGE",
        }
        with pytest.raises(AssertionError, match="Invalid stage"):
            assert_checkpoint_consistent(checkpoint)

    @pytest.mark.fast
    def test_expected_stage_mismatch(self):
        """Stage mismatch with expected raises assertion."""
        from tests.helpers import assert_checkpoint_consistent

        checkpoint = {
            "version": "1.0",
            "last_completed_stage": "DOWNLOAD",
        }
        with pytest.raises(AssertionError, match="Expected stage 'MATCH'"):
            assert_checkpoint_consistent(checkpoint, expected_stage="MATCH")

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_none_config_raises(self):
        """None config raises assertion."""
        from tests.helpers import assert_config_valid

        with pytest.raises(AssertionError, match="Config is None"):
            assert_config_valid(None)

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_rejects_non_timeline(self):
        """Non-Timeline object raises assertion."""
        from tests.helpers import assert_valid_otio_timeline

        with pytest.raises(AssertionError, match="Expected opentimelineio"):
            assert_valid_otio_timeline({"not": "a timeline"})

    @pytest.mark.fast
    def test_rejects_string(self):
        """String raises assertion."""
        from tests.helpers import assert_valid_otio_timeline

        with pytest.raises(AssertionError, match="Expected opentimelineio"):
            assert_valid_otio_timeline("not a timeline")

    @pytest.mark.skipif(
        not pytest.importorskip("opentimelineio", reason="opentimelineio not installed"),
        reason="opentimelineio not installed"
    )
    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_assert_file_exists_missing(self, tmp_path):
        """Missing file raises assertion."""
        from tests.helpers import assert_file_exists

        fake_path = tmp_path / "nonexistent.txt"
        with pytest.raises(AssertionError, match="not found"):
            assert_file_exists(fake_path, description="Test file")

    @pytest.mark.fast
    def test_assert_file_exists_is_dir(self, tmp_path):
        """Directory instead of file raises assertion."""
        from tests.helpers import assert_file_exists

        with pytest.raises(AssertionError, match="is not a file"):
            assert_file_exists(tmp_path, description="Test file")

    @pytest.mark.fast
    def test_assert_file_exists_valid(self, tmp_path):
        """Existing file passes."""
        from tests.helpers import assert_file_exists

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Should not raise
        assert_file_exists(test_file)

    @pytest.mark.fast
    def test_assert_dir_exists_missing(self, tmp_path):
        """Missing directory raises assertion."""
        from tests.helpers import assert_dir_exists

        fake_path = tmp_path / "nonexistent"
        with pytest.raises(AssertionError, match="not found"):
            assert_dir_exists(fake_path)

    @pytest.mark.fast
    def test_assert_dir_exists_is_file(self, tmp_path):
        """File instead of directory raises assertion."""
        from tests.helpers import assert_dir_exists

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        with pytest.raises(AssertionError, match="is not a directory"):
            assert_dir_exists(test_file)

    @pytest.mark.fast
    def test_assert_json_structure_missing_key(self):
        """Missing required key raises assertion."""
        from tests.helpers import assert_json_structure

        data = {"a": 1, "b": 2}
        with pytest.raises(AssertionError, match="Missing required key: c"):
            assert_json_structure(data, required_keys=["a", "b", "c"])

    @pytest.mark.fast
    def test_assert_json_structure_wrong_type(self):
        """Wrong type raises assertion."""
        from tests.helpers import assert_json_structure

        data = {"count": "not a number"}
        with pytest.raises(AssertionError, match="should be int"):
            assert_json_structure(data, type_checks={"count": int})

    @pytest.mark.fast
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


# =============================================================================
# Agent Healer Assertion Tests
# =============================================================================

class TestAssertHealerAttemptLogged:
    """Tests for assert_healer_attempt_logged helper."""

    @pytest.mark.fast
    def test_healer_with_dict_context(self):
        """Healer attempt in dict context passes."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "attempts": [
                {"healer": "api-healer", "message": "Rate limit detected"},
                {"healer": "checkpoint-healer", "message": "Restoring from backup"},
            ]
        }

        # Should not raise
        assert_healer_attempt_logged(context, "api-healer")
        assert_healer_attempt_logged(context, "checkpoint-healer")

    @pytest.mark.fast
    def test_healer_with_healers_dict(self):
        """Healer in healers dict context passes."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "healers": {
                "download-healer": [
                    {"message": "Retrying download"},
                    {"message": "Increasing timeout"},
                ],
            }
        }

        assert_healer_attempt_logged(context, "download-healer")

    @pytest.mark.fast
    def test_healer_not_found_raises(self):
        """Missing healer raises assertion."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "attempts": [
                {"healer": "api-healer", "message": "Rate limit"},
            ]
        }

        with pytest.raises(AssertionError, match="did not log any attempts"):
            assert_healer_attempt_logged(context, "checkpoint-healer")

    @pytest.mark.fast
    def test_specific_attempt_number(self):
        """Specific attempt number validation."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "healers": {
                "api-healer": [
                    {"message": "First attempt"},
                    {"message": "Second attempt"},
                ],
            }
        }

        # Should pass for attempt 1 and 2
        assert_healer_attempt_logged(context, "api-healer", 1)
        assert_healer_attempt_logged(context, "api-healer", 2)

        # Should fail for attempt 3
        with pytest.raises(AssertionError, match="only logged 2 attempts"):
            assert_healer_attempt_logged(context, "api-healer", 3)

    @pytest.mark.fast
    def test_expected_message_found(self):
        """Message substring matching passes."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "attempts": [
                {"healer": "disk-healer", "message": "Cleaning up temp files"},
            ]
        }

        assert_healer_attempt_logged(
            context, "disk-healer", expected_message="Cleaning up"
        )

    @pytest.mark.fast
    def test_expected_message_not_found(self):
        """Missing message raises assertion."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "attempts": [
                {"healer": "disk-healer", "message": "Checking disk space"},
            ]
        }

        with pytest.raises(AssertionError, match="No attempts.*contain message"):
            assert_healer_attempt_logged(
                context, "disk-healer", expected_message="Cleaning up"
            )

    @pytest.mark.fast
    def test_healer_name_normalization(self):
        """Healer names are normalized (underscores to dashes)."""
        from tests.helpers import assert_healer_attempt_logged

        context = {
            "attempts": [
                {"healer": "api-healer", "message": "Test"},
            ]
        }

        # Both formats should work
        assert_healer_attempt_logged(context, "api-healer")
        assert_healer_attempt_logged(context, "api_healer")

    @pytest.mark.fast
    def test_mock_with_healer_attempts_attr(self):
        """Context with _healer_attempts attribute works."""
        from tests.helpers import assert_healer_attempt_logged

        class MockOrchestrator:
            _healer_attempts = {
                "path-healer": [{"message": "Path too long"}],
            }

        context = MockOrchestrator()
        assert_healer_attempt_logged(context, "path-healer")


class TestAssertHealingStrategyApplied:
    """Tests for assert_healing_strategy_applied helper."""

    @pytest.mark.fast
    def test_direct_strategy_object(self):
        """Direct strategy object validation passes."""
        from tests.helpers import assert_healing_strategy_applied

        class MockStrategy:
            mode = Mock(value="conservative")
            max_attempts_per_stage = 3
            heal_delay = 2.0

        strategy = MockStrategy()
        assert_healing_strategy_applied(strategy, "conservative")

    @pytest.mark.fast
    def test_strategy_from_orchestrator(self):
        """Strategy extracted from orchestrator passes."""
        from tests.helpers import assert_healing_strategy_applied

        class MockOrchestrator:
            class MockStrategy:
                mode = Mock(value="aggressive")
                max_attempts_per_stage = 5
                heal_delay = 1.0
            strategy = MockStrategy()

        orchestrator = MockOrchestrator()
        assert_healing_strategy_applied(orchestrator, "aggressive")

    @pytest.mark.fast
    def test_strategy_from_dict(self):
        """Strategy extracted from dict passes."""
        from tests.helpers import assert_healing_strategy_applied

        context = {
            "strategy": {
                "mode": "minimal",
                "max_attempts_per_stage": 1,
                "heal_delay": 0.5,
            }
        }

        assert_healing_strategy_applied(context, "minimal")

    @pytest.mark.fast
    def test_invalid_strategy_name(self):
        """Invalid strategy name raises assertion."""
        from tests.helpers import assert_healing_strategy_applied

        class MockStrategy:
            mode = Mock(value="conservative")

        with pytest.raises(AssertionError, match="Invalid strategy"):
            assert_healing_strategy_applied(MockStrategy(), "invalid_strategy")

    @pytest.mark.fast
    def test_strategy_mismatch(self):
        """Wrong strategy raises assertion."""
        from tests.helpers import assert_healing_strategy_applied

        class MockStrategy:
            mode = Mock(value="conservative")
            max_attempts_per_stage = 3  # Required for _extract_strategy

        with pytest.raises(AssertionError, match="Expected strategy 'aggressive'"):
            assert_healing_strategy_applied(MockStrategy(), "aggressive")

    @pytest.mark.fast
    def test_check_max_attempts(self):
        """Max attempts parameter check."""
        from tests.helpers import assert_healing_strategy_applied

        class MockStrategy:
            mode = Mock(value="aggressive")
            max_attempts_per_stage = 5
            heal_delay = 1.0

        # Should pass
        assert_healing_strategy_applied(
            MockStrategy(), "aggressive", check_max_attempts=5
        )

        # Should fail
        with pytest.raises(AssertionError, match="max_attempts_per_stage is 5"):
            assert_healing_strategy_applied(
                MockStrategy(), "aggressive", check_max_attempts=3
            )

    @pytest.mark.fast
    def test_check_heal_delay(self):
        """Heal delay parameter check."""
        from tests.helpers import assert_healing_strategy_applied

        class MockStrategy:
            mode = Mock(value="conservative")
            max_attempts_per_stage = 3
            heal_delay = 2.0

        # Should pass
        assert_healing_strategy_applied(
            MockStrategy(), "conservative", check_heal_delay=2.0
        )

        # Should fail
        with pytest.raises(AssertionError, match="heal_delay is 2.0"):
            assert_healing_strategy_applied(
                MockStrategy(), "conservative", check_heal_delay=1.0
            )


class TestAssertRecoveryMetricsValid:
    """Tests for assert_recovery_metrics_valid helper."""

    @pytest.mark.fast
    def test_valid_metrics_dict(self):
        """Valid metrics dict passes."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {
            "total_heals": 3,
            "successful_heals": 2,
            "failed_heals": 1,
            "heals_by_healer": {"api-healer": 2, "download-healer": 1},
            "heals_by_stage": {"DOWNLOAD": 2, "MATCH": 1},
        }

        # Should not raise
        assert_recovery_metrics_valid(metrics)

    @pytest.mark.fast
    def test_expected_attempts(self):
        """Expected attempts check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {"total_heals": 5, "successful_heals": 3, "failed_heals": 2}

        assert_recovery_metrics_valid(metrics, expected_attempts=5)

        with pytest.raises(AssertionError, match="Expected 10 heal attempts"):
            assert_recovery_metrics_valid(metrics, expected_attempts=10)

    @pytest.mark.fast
    def test_expected_success_true(self):
        """Expected success=True check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {"total_heals": 2, "successful_heals": 1, "failed_heals": 1}
        assert_recovery_metrics_valid(metrics, expected_success=True)

        metrics_no_success = {"total_heals": 2, "successful_heals": 0, "failed_heals": 2}
        with pytest.raises(AssertionError, match="no successful heals"):
            assert_recovery_metrics_valid(metrics_no_success, expected_success=True)

    @pytest.mark.fast
    def test_expected_success_false(self):
        """Expected success=False check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {"total_heals": 2, "successful_heals": 0, "failed_heals": 2}
        assert_recovery_metrics_valid(metrics, expected_success=False)

        metrics_all_success = {"total_heals": 2, "successful_heals": 2, "failed_heals": 0}
        with pytest.raises(AssertionError, match="no failed heals"):
            assert_recovery_metrics_valid(metrics_all_success, expected_success=False)

    @pytest.mark.fast
    def test_min_successful_heals(self):
        """Minimum successful heals check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {"total_heals": 5, "successful_heals": 3, "failed_heals": 2}

        assert_recovery_metrics_valid(metrics, min_successful_heals=3)

        with pytest.raises(AssertionError, match="Expected at least 5 successful"):
            assert_recovery_metrics_valid(metrics, min_successful_heals=5)

    @pytest.mark.fast
    def test_max_failed_heals(self):
        """Maximum failed heals check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {"total_heals": 5, "successful_heals": 3, "failed_heals": 2}

        assert_recovery_metrics_valid(metrics, max_failed_heals=2)

        with pytest.raises(AssertionError, match="Expected at most 1 failed"):
            assert_recovery_metrics_valid(metrics, max_failed_heals=1)

    @pytest.mark.fast
    def test_expected_healers_used(self):
        """Expected healers used check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {
            "total_heals": 3,
            "successful_heals": 2,
            "failed_heals": 1,
            "heals_by_healer": {"api-healer": 2, "download-healer": 1},
        }

        assert_recovery_metrics_valid(
            metrics, expected_healers_used=["api-healer", "download-healer"]
        )

        with pytest.raises(AssertionError, match="Expected healers not used"):
            assert_recovery_metrics_valid(
                metrics, expected_healers_used=["checkpoint-healer"]
            )

    @pytest.mark.fast
    def test_expected_stages_healed(self):
        """Expected stages healed check."""
        from tests.helpers import assert_recovery_metrics_valid

        metrics = {
            "total_heals": 3,
            "successful_heals": 2,
            "failed_heals": 1,
            "heals_by_stage": {"DOWNLOAD": 2, "MATCH": 1},
        }

        assert_recovery_metrics_valid(
            metrics, expected_stages_healed=["DOWNLOAD", "MATCH"]
        )

        with pytest.raises(AssertionError, match="Expected stages not healed"):
            assert_recovery_metrics_valid(
                metrics, expected_stages_healed=["OUTPUT"]
            )

    @pytest.mark.fast
    def test_consistency_check(self):
        """Metrics consistency validation."""
        from tests.helpers import assert_recovery_metrics_valid

        # successful + failed exceeds total - should fail
        metrics = {"total_heals": 2, "successful_heals": 2, "failed_heals": 2}
        with pytest.raises(AssertionError, match="exceeds total_heals"):
            assert_recovery_metrics_valid(metrics)

    @pytest.mark.fast
    def test_negative_values_rejected(self):
        """Negative metric values are rejected."""
        from tests.helpers import assert_recovery_metrics_valid

        with pytest.raises(AssertionError, match="cannot be negative"):
            assert_recovery_metrics_valid({"total_heals": -1, "successful_heals": 0, "failed_heals": 0})


class TestAssertHealerChainExecuted:
    """Tests for assert_healer_chain_executed helper."""

    @pytest.mark.fast
    def test_chain_with_all_healers(self):
        """All expected healers executed passes."""
        from tests.helpers import assert_healer_chain_executed

        context = {
            "healers": {
                "checkpoint-healer": [{"message": "Restoring"}],
                "api-healer": [{"message": "Rate limit"}],
                "download-healer": [{"message": "Retrying"}],
            }
        }

        assert_healer_chain_executed(
            context,
            ["checkpoint-healer", "api-healer", "download-healer"],
        )

    @pytest.mark.fast
    def test_chain_missing_healer(self):
        """Missing required healer raises assertion."""
        from tests.helpers import assert_healer_chain_executed

        context = {
            "healers": {
                "api-healer": [{"message": "Rate limit"}],
            }
        }

        with pytest.raises(AssertionError, match="Required healers not executed"):
            assert_healer_chain_executed(
                context,
                ["checkpoint-healer", "api-healer"],
            )

    @pytest.mark.fast
    def test_chain_partial_match(self):
        """Partial chain match with all_required=False."""
        from tests.helpers import assert_healer_chain_executed

        context = {
            "healers": {
                "api-healer": [{"message": "Rate limit"}],
            }
        }

        # Should pass - only api-healer needed
        assert_healer_chain_executed(
            context,
            ["checkpoint-healer", "api-healer"],
            all_required=False,
        )

    @pytest.mark.fast
    def test_chain_in_order(self):
        """Chain order validation."""
        from tests.helpers import assert_healer_chain_executed

        context = {
            "attempts": [
                {"healer": "checkpoint-healer", "message": "First"},
                {"healer": "api-healer", "message": "Second"},
                {"healer": "download-healer", "message": "Third"},
            ]
        }

        # Correct order - should pass
        assert_healer_chain_executed(
            context,
            ["checkpoint-healer", "api-healer"],
            in_order=True,
        )

    @pytest.mark.fast
    def test_chain_wrong_order(self):
        """Wrong chain order raises assertion."""
        from tests.helpers import assert_healer_chain_executed

        context = {
            "attempts": [
                {"healer": "download-healer", "message": "First"},
                {"healer": "api-healer", "message": "Second"},
                {"healer": "checkpoint-healer", "message": "Third"},
            ]
        }

        with pytest.raises(AssertionError, match="not executed in expected order"):
            assert_healer_chain_executed(
                context,
                ["checkpoint-healer", "api-healer"],
                in_order=True,
            )

    @pytest.mark.fast
    def test_empty_healers_list_rejected(self):
        """Empty expected_healers raises assertion."""
        from tests.helpers import assert_healer_chain_executed

        context = {"healers": {"api-healer": [{"message": "Test"}]}}

        with pytest.raises(AssertionError, match="cannot be empty"):
            assert_healer_chain_executed(context, [])

    @pytest.mark.fast
    def test_no_healers_logged(self):
        """No healers logged raises assertion."""
        from tests.helpers import assert_healer_chain_executed

        context = {"healers": {}, "attempts": []}

        with pytest.raises(AssertionError, match="No healers logged"):
            assert_healer_chain_executed(context, ["api-healer"])


class TestHealerHelperConstants:
    """Tests for healer helper constants."""

    @pytest.mark.fast
    def test_valid_healers_constant(self):
        """VALID_HEALERS constant has expected values."""
        from tests.helpers import VALID_HEALERS

        assert "checkpoint-healer" in VALID_HEALERS
        assert "api-healer" in VALID_HEALERS
        assert "download-healer" in VALID_HEALERS
        assert "disk-healer" in VALID_HEALERS
        assert "path-healer" in VALID_HEALERS
        assert "otio-healer" in VALID_HEALERS
        assert "llm-healer" in VALID_HEALERS

    @pytest.mark.fast
    def test_valid_strategies_constant(self):
        """VALID_STRATEGIES constant has expected values."""
        from tests.helpers import VALID_STRATEGIES

        assert "aggressive" in VALID_STRATEGIES
        assert "conservative" in VALID_STRATEGIES
        assert "interactive" in VALID_STRATEGIES
        assert "minimal" in VALID_STRATEGIES
