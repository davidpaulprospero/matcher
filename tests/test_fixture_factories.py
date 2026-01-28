"""
Tests for fixture factory patterns (US-006, Sprint 19).

Verifies that the factory fixtures in conftest.py work correctly
and can be used to create customizable test objects.
"""

import pytest
from pathlib import Path


class TestSRTSegmentFactory:
    """Test srt_segment_factory fixture."""

    @pytest.mark.fast
    def test_creates_with_defaults(self, srt_segment_factory):
        """Factory creates segment with sensible defaults."""
        segment = srt_segment_factory()

        assert segment.index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Sample voiceover text"
        assert segment.is_broll is False

    @pytest.mark.fast
    def test_accepts_custom_values(self, srt_segment_factory):
        """Factory accepts custom values for all fields."""
        segment = srt_segment_factory(
            index=5,
            start_time=10.0,
            end_time=20.0,
            text="Custom text",
            is_broll=True,
            keywords=["travel", "nature"],
        )

        assert segment.index == 5
        assert segment.start_time == 10.0
        assert segment.end_time == 20.0
        assert segment.text == "Custom text"
        assert segment.is_broll is True
        assert segment.keywords == ["travel", "nature"]

    @pytest.mark.fast
    def test_multiple_segments_independent(self, srt_segment_factory):
        """Multiple factory calls create independent objects."""
        seg1 = srt_segment_factory(text="First")
        seg2 = srt_segment_factory(text="Second")

        assert seg1.text != seg2.text
        assert seg1 is not seg2


class TestConfigFactory:
    """Test config_factory fixture."""

    @pytest.mark.fast
    def test_creates_with_defaults(self, config_factory):
        """Factory creates config with all sections."""
        config = config_factory()

        assert config.cache is not None
        assert config.output is not None
        assert config.transcription is not None
        assert config.matching is not None
        assert config.keyword is not None
        assert config.download is not None

    @pytest.mark.fast
    def test_accepts_section_overrides(self, config_factory):
        """Factory accepts section-level overrides."""
        config = config_factory(
            matching={'min_confidence': 0.9},
            keyword={'max_keywords': 20},
        )

        assert config.matching.min_confidence == 0.9
        assert config.keyword.max_keywords == 20

    @pytest.mark.fast
    def test_creates_directories(self, config_factory):
        """Factory creates required directories."""
        config = config_factory()

        assert Path(config.cache.cache_dir).exists()
        assert Path(config.output.output_dir).exists()
        assert Path(config.download.root_dir).exists()


class TestPipelineStateFactory:
    """Test pipeline_state_factory fixture."""

    @pytest.mark.fast
    def test_creates_empty_state(self, pipeline_state_factory):
        """Factory creates empty state with defaults."""
        state = pipeline_state_factory()

        assert state.voiceover_path == ""
        assert state.voiceover_segments == []
        assert state.keywords == []
        assert state.matches == []

    @pytest.mark.fast
    def test_accepts_field_values(self, pipeline_state_factory):
        """Factory accepts field values."""
        state = pipeline_state_factory(
            voiceover_path="/path/to/vo.srt",
            keywords=["travel", "nature"],
            topic_context="Wildlife documentary",
        )

        assert state.voiceover_path == "/path/to/vo.srt"
        assert state.keywords == ["travel", "nature"]
        assert state.topic_context == "Wildlife documentary"


class TestMatchResultFactory:
    """Test match_result_factory fixture."""

    @pytest.mark.fast
    def test_creates_with_defaults(self, match_result_factory):
        """Factory creates result with default confidence."""
        result = match_result_factory()

        assert result.primary_match.confidence == 0.85
        assert result.primary_match.video_segment.source_file == "video.mp4"
        assert result.has_gap is False
        assert result.alternatives == []

    @pytest.mark.fast
    def test_accepts_custom_confidence(self, match_result_factory):
        """Factory accepts custom confidence."""
        result = match_result_factory(confidence=0.95)

        assert result.primary_match.confidence == 0.95

    @pytest.mark.fast
    def test_creates_gap_result(self, match_result_factory):
        """Factory creates gap result."""
        result = match_result_factory(has_gap=True, gap_reason="No suitable match")

        assert result.has_gap is True
        assert result.gap_reason == "No suitable match"

    @pytest.mark.fast
    def test_creates_alternatives(self, match_result_factory):
        """Factory creates alternatives with decreasing confidence."""
        result = match_result_factory(confidence=0.90, num_alternatives=3)

        assert len(result.alternatives) == 3
        assert result.alternatives[0].confidence == 0.85  # 0.90 - 0.05
        assert result.alternatives[1].confidence == 0.80  # 0.90 - 0.10
        assert result.alternatives[2].confidence == 0.75  # 0.90 - 0.15


class TestMatchFactory:
    """Test match_factory fixture."""

    @pytest.mark.fast
    def test_creates_with_defaults(self, match_factory):
        """Factory creates match with sensible defaults."""
        match = match_factory()

        assert match.video_segment.source_file == "video.mp4"
        assert match.confidence == 0.85

    @pytest.mark.fast
    def test_accepts_custom_values(self, match_factory):
        """Factory accepts custom values."""
        match = match_factory(
            video_source_file="custom.mp4",
            confidence=0.99,
            vo_text="Custom voiceover",
            video_text="Custom video text",
        )

        assert match.video_segment.source_file == "custom.mp4"
        assert match.confidence == 0.99
        assert match.voiceover_segment.text == "Custom voiceover"
        assert match.video_segment.text == "Custom video text"


# =============================================================================
# US-010: Complex Coordination Scenario Fixture Tests
# =============================================================================


class TestCheckpointWithPopulatedStagesFixture:
    """Test create_checkpoint_with_populated_stages fixture (AC1)."""

    @pytest.mark.fast
    def test_creates_checkpoint_with_all_stages(self):
        """Fixture creates checkpoint with populated stages dict."""
        from tests.fixtures import create_checkpoint_with_populated_stages

        cp = create_checkpoint_with_populated_stages()

        # All major stages should have data
        assert "analyze" in cp
        assert "download" in cp
        assert "transcribe" in cp
        assert "match" in cp
        assert "scene_detection" in cp

    @pytest.mark.fast
    def test_match_count_customization(self):
        """Fixture accepts custom match_count parameter."""
        from tests.fixtures import create_checkpoint_with_populated_stages

        cp = create_checkpoint_with_populated_stages(match_count=20)

        assert cp["match"]["match_count"] == 20
        assert len(cp["match"]["matches"]) == 20

    @pytest.mark.fast
    def test_avg_confidence_customization(self):
        """Fixture accepts custom avg_confidence parameter."""
        from tests.fixtures import create_checkpoint_with_populated_stages

        cp = create_checkpoint_with_populated_stages(avg_confidence=0.92)

        assert cp["match"]["avg_confidence"] == 0.92

    @pytest.mark.fast
    def test_video_count_affects_download_paths(self):
        """Fixture generates correct number of video paths."""
        from tests.fixtures import create_checkpoint_with_populated_stages

        cp = create_checkpoint_with_populated_stages(video_count=8)

        assert cp["download"]["count"] == 8
        assert len(cp["download"]["video_paths"]) == 8

    @pytest.mark.fast
    def test_stage_overrides(self):
        """Fixture accepts stage-specific overrides."""
        from tests.fixtures import create_checkpoint_with_populated_stages

        cp = create_checkpoint_with_populated_stages(
            analyze={"keywords": ["custom", "keywords", "list"]}
        )

        assert cp["analyze"]["keywords"] == ["custom", "keywords", "list"]

    @pytest.mark.fast
    def test_required_fields_for_output_stage(self):
        """Fixture generates all fields needed for OUTPUT stage (Rule 25)."""
        from tests.fixtures import create_checkpoint_with_populated_stages

        cp = create_checkpoint_with_populated_stages()

        # These fields are required for --output-only to work
        assert "last_completed_stage" in cp
        assert "config_hash" in cp
        assert "match" in cp and "matches" in cp["match"]
        assert "download_segments" in cp


class TestConcurrentEscalationFixture:
    """Test create_concurrent_escalation_fixture fixture (AC2)."""

    @pytest.mark.fast
    def test_generates_keywords(self):
        """Fixture generates correct number of keywords."""
        from tests.fixtures import create_concurrent_escalation_fixture

        fixture = create_concurrent_escalation_fixture(num_keywords=5)

        assert len(fixture["keywords"]) == 5
        assert fixture["keywords"][0] == "test_keyword_0"
        assert fixture["keywords"][4] == "test_keyword_4"

    @pytest.mark.fast
    def test_ext_config_structure(self):
        """Fixture generates valid ExtractorArgs config."""
        from tests.fixtures import create_concurrent_escalation_fixture

        fixture = create_concurrent_escalation_fixture()

        cfg = fixture["ext_config"]
        assert cfg["enabled"] is True
        assert cfg["escalation_threshold"] == 2
        assert "player_clients" in cfg
        assert isinstance(cfg["player_clients"], list)

    @pytest.mark.fast
    def test_initial_states_at_tier1(self):
        """All keywords start at tier 1."""
        from tests.fixtures import create_concurrent_escalation_fixture

        fixture = create_concurrent_escalation_fixture(num_keywords=3)

        for kw in fixture["keywords"]:
            state = fixture["initial_states"][kw]
            assert state["tier"] == 1
            assert state["consecutive_403s"] == 0

    @pytest.mark.fast
    def test_custom_escalation_threshold(self):
        """Fixture accepts custom escalation threshold."""
        from tests.fixtures import create_concurrent_escalation_fixture

        fixture = create_concurrent_escalation_fixture(escalation_threshold=5)

        assert fixture["ext_config"]["escalation_threshold"] == 5

    @pytest.mark.fast
    def test_expected_tier_args(self):
        """Fixture provides expected args for each tier."""
        from tests.fixtures import create_concurrent_escalation_fixture

        fixture = create_concurrent_escalation_fixture()

        # Should have impersonate args
        assert "--impersonate" in fixture["impersonate_args"]
        # Should have tier 2 extractor args
        assert "--extractor-args" in fixture["expected_tier2_args"]
        # Tier 3 should rotate cookies
        assert fixture["expected_tier3_rotate_cookies"] is True


class TestOtioTimelineFixture:
    """Test create_otio_timeline_fixture fixture (AC3)."""

    @pytest.mark.fast
    def test_at_threshold_boundary(self):
        """Fixture correctly identifies at-threshold condition."""
        from tests.fixtures import create_otio_timeline_fixture

        fixture = create_otio_timeline_fixture(3000)

        assert fixture["at_threshold"] is True
        assert fixture["above_threshold"] is False
        assert fixture["below_threshold"] is False
        assert fixture["expected_parts"] == 1

    @pytest.mark.fast
    def test_above_threshold(self):
        """Fixture correctly calculates parts above threshold."""
        from tests.fixtures import create_otio_timeline_fixture

        fixture = create_otio_timeline_fixture(3001)

        assert fixture["at_threshold"] is False
        assert fixture["above_threshold"] is True
        assert fixture["expected_parts"] == 2

    @pytest.mark.fast
    def test_below_threshold(self):
        """Fixture correctly identifies below-threshold condition."""
        from tests.fixtures import create_otio_timeline_fixture

        fixture = create_otio_timeline_fixture(2999)

        assert fixture["below_threshold"] is True
        assert fixture["expected_parts"] == 1

    @pytest.mark.fast
    def test_large_timeline_parts_calculation(self):
        """Fixture correctly calculates parts for large timelines."""
        from tests.fixtures import create_otio_timeline_fixture

        # 6500 segments should split into 3 parts (ceil(6500/3000))
        fixture = create_otio_timeline_fixture(6500)

        assert fixture["expected_parts"] == 3

    @pytest.mark.fast
    def test_track_config(self):
        """Fixture generates correct track configuration."""
        from tests.fixtures import create_otio_timeline_fixture

        fixture = create_otio_timeline_fixture(100, include_audio=True)

        assert "V1" in fixture["track_config"]
        assert "A1" in fixture["track_config"]
        assert fixture["track_count"] == 2

    @pytest.mark.fast
    def test_constants_available(self):
        """Fixture provides threshold constants for reference."""
        from tests.fixtures import create_otio_timeline_fixture

        fixture = create_otio_timeline_fixture(100)

        assert fixture["max_segments_per_part"] == 3000
        assert fixture["warning_threshold"] == 2500
        assert fixture["error_threshold"] == 3000


class TestBrollPropagationChainStateFixture:
    """Test create_broll_propagation_chain_state fixture (AC4)."""

    @pytest.mark.fast
    def test_generates_scenes(self):
        """Fixture generates correct number of scenes."""
        from tests.fixtures import create_broll_propagation_chain_state

        state = create_broll_propagation_chain_state(num_scenes=15)

        assert len(state["scenes"]) == 15

    @pytest.mark.fast
    def test_broll_ratio(self):
        """Fixture respects B-roll ratio."""
        from tests.fixtures import create_broll_propagation_chain_state

        state = create_broll_propagation_chain_state(num_scenes=10, broll_ratio=0.4)

        broll_count = len([s for s in state["scenes"] if s["is_broll"]])
        assert broll_count == 4
        assert state["expected_broll_count"] == 4

    @pytest.mark.fast
    def test_detection_methods(self):
        """Fixture splits B-roll between face and silent detection."""
        from tests.fixtures import create_broll_propagation_chain_state

        state = create_broll_propagation_chain_state(num_scenes=10, broll_ratio=0.4)

        # Should have both detection methods
        assert state["face_detected_broll"] > 0
        assert state["silent_detected_broll"] > 0
        assert state["face_detected_broll"] + state["silent_detected_broll"] == state["expected_broll_count"]

    @pytest.mark.fast
    def test_text_metadata_propagation(self):
        """Fixture generates text_metadata with is_broll flag preserved."""
        from tests.fixtures import create_broll_propagation_chain_state

        state = create_broll_propagation_chain_state(num_scenes=5, include_text_metadata=True)

        assert len(state["text_metadata"]) == 5

        # Check that is_broll is preserved in text_metadata
        for scene, metadata in zip(state["scenes"], state["text_metadata"]):
            assert scene["is_broll"] == metadata["is_broll"]

    @pytest.mark.fast
    def test_threshold_customization(self):
        """Fixture accepts custom detection thresholds."""
        from tests.fixtures import create_broll_propagation_chain_state

        state = create_broll_propagation_chain_state(
            face_score_threshold=0.25,
            min_words_threshold=5
        )

        assert state["thresholds"]["face_score"] == 0.25
        assert state["thresholds"]["min_words"] == 5

    @pytest.mark.fast
    def test_v8_expectations(self):
        """Fixture provides V8 track expectations."""
        from tests.fixtures import create_broll_propagation_chain_state

        state = create_broll_propagation_chain_state(num_scenes=10, broll_ratio=0.3)

        # V8 entries should match total B-roll count
        assert state["expected_v8_entries"] == state["expected_broll_count"]
        assert (
            state["v8_from_face_detection"] + state["v8_from_silent_detection"]
            == state["expected_v8_entries"]
        )
