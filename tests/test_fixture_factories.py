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
