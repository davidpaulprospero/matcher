"""Tests for segment post-processing (US-124-011).

Tests for intelligent segmentation post-processing:
- Merging segments with same speaker
- Splitting segments at natural language boundaries
"""

import pytest
from src.transcription.utils import (
    merge_segments_by_speaker,
    split_segment_at_punctuation,
    post_process_segments,
)
from src.config.sections.core import SegmentPostProcessingConfig


class TestMergeSegmentsBySpeaker:
    """Tests for segment merging based on speaker labels."""

    def test_no_speakers_returns_original(self):
        """If no speaker labels, segments should remain unchanged."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
            {"start": 1.0, "end": 2.0, "text": "This is a test"},
        ]
        result = merge_segments_by_speaker(segments)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"
        assert result[1]["text"] == "This is a test"

    def test_merge_consecutive_same_speaker(self):
        """Consecutive segments with same speaker should be merged."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_0"},
            {"start": 1.0, "end": 1.5, "text": "world", "speaker": "SPEAKER_0"},
        ]
        result = merge_segments_by_speaker(segments)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 1.5

    def test_no_merge_different_speaker(self):
        """Different speakers should not be merged."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_0"},
            {"start": 1.0, "end": 2.0, "text": "world", "speaker": "SPEAKER_1"},
        ]
        result = merge_segments_by_speaker(segments)
        assert len(result) == 2

    def test_no_merge_large_gap(self):
        """Large gaps between segments should prevent merging."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello", "speaker": "SPEAKER_0"},
            {"start": 2.0, "end": 3.0, "text": "world", "speaker": "SPEAKER_0"},
        ]
        result = merge_segments_by_speaker(segments, max_gap_seconds=0.5)
        assert len(result) == 2


class TestSplitSegmentAtPunctuation:
    """Tests for splitting segments at punctuation boundaries."""

    def test_split_at_period(self):
        """Split at period followed by space and text."""
        segment = {
            "start": 0.0,
            "end": 3.0,
            "text": "This is sentence one. This is sentence two."
        }
        result = split_segment_at_punctuation(segment)
        assert len(result) == 2
        assert "sentence one." in result[0]["text"]
        assert "sentence two" in result[1]["text"]

    def test_split_at_question_mark(self):
        """Split at question mark."""
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "What is your name? My name is test."
        }
        result = split_segment_at_punctuation(segment)
        assert len(result) >= 2
        assert "?" in result[0]["text"]

    def test_ignore_abbreviations(self):
        """Abbreviations like 'Mr.' should not trigger splits."""
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "Mr. Smith arrived. Dr. Jones was there."
        }
        result = split_segment_at_punctuation(segment)
        # Should not split at "Mr." or "Dr."
        assert len(result) == 1 or (len(result) == 2 and "Smith arrived." in result[0]["text"] or "Smith arrived" in result[0]["text"])

    def test_short_segment_not_split(self):
        """Very short segments should not be split."""
        segment = {
            "start": 0.0,
            "end": 1.0,
            "text": "Hi."
        }
        result = split_segment_at_punctuation(segment, min_duration=0.5)
        assert len(result) == 1


class TestPostProcessSegments:
    """Tests for the full post-processing pipeline."""

    def test_disabled_returns_original(self):
        """When disabled, original segments returned."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
        ]
        config = SegmentPostProcessingConfig(enabled=False)
        result = post_process_segments(segments, config)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_full_pipeline(self):
        """Test full pipeline with merging and splitting."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello.", "speaker": "SPEAKER_0"},
            {"start": 1.0, "end": 1.5, "text": "World.", "speaker": "SPEAKER_0"},
            {"start": 2.0, "end": 3.0, "text": "Test one. Test two.", "speaker": "SPEAKER_1"},
        ]
        config = SegmentPostProcessingConfig(
            enabled=True,
            merge_same_speaker=True,
            split_at_punctuation=True
        )
        result = post_process_segments(segments, config)

        # Should have merged speaker 0 segments and split speaker 1 segments
        assert len(result) >= 3  # At least 3 segments after splitting

    def test_empty_segments(self):
        """Empty segment list should return empty list."""
        result = post_process_segments([])
        assert result == []


class TestSegmentPostProcessingConfig:
    """Tests for SegmentPostProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SegmentPostProcessingConfig()

        assert config.enabled is True
        assert config.merge_same_speaker is True
        assert config.merge_max_gap_seconds == 0.5
        assert config.merge_min_duration == 1.0
        assert config.split_at_punctuation is True
        assert config.split_punctuation == ".!?"
        assert config.split_min_duration == 0.5
        assert "mr" in config.abbreviations

    def test_custom_abbreviations(self):
        """Test custom abbreviations list."""
        config = SegmentPostProcessingConfig(abbreviations=["custom", "test"])
        assert "custom" in config.abbreviations
        assert "test" in config.abbreviations
