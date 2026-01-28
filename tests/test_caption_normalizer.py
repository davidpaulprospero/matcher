"""
Tests for caption timestamp normalization (US-009).

Tests the CaptionNormalizer class which handles:
- Overlapping segments (merge/split/truncate strategies)
- Gap segments (placeholder/extend/ignore strategies)
- Invalid timestamps (negative, end < start)
- Timestamp format conversion (VTT, SRT, JSON3)
"""

import pytest
from src.caption_fetcher import (
    CaptionSegment,
    CaptionNormalizer,
    CaptionNormalizationError,
    NormalizationConfig,
)


@pytest.mark.fast
class TestNormalizationConfig:
    """Test NormalizationConfig dataclass"""

    def test_default_config(self):
        """Test default config values"""
        config = NormalizationConfig()

        assert config.overlap_strategy == "truncate"
        assert config.gap_strategy == "ignore"
        assert config.max_gap_to_extend == 1.0
        assert config.min_segment_duration == 0.1
        assert config.validate_timestamps is True

    def test_custom_config(self):
        """Test custom config values"""
        config = NormalizationConfig(
            overlap_strategy="merge",
            gap_strategy="extend",
            max_gap_to_extend=2.5,
            min_segment_duration=0.5,
            validate_timestamps=False
        )

        assert config.overlap_strategy == "merge"
        assert config.gap_strategy == "extend"
        assert config.max_gap_to_extend == 2.5
        assert config.min_segment_duration == 0.5
        assert config.validate_timestamps is False


@pytest.mark.fast
class TestCaptionNormalizerInit:
    """Test CaptionNormalizer initialization"""

    def test_default_init(self):
        """Test initialization with default config"""
        normalizer = CaptionNormalizer()

        assert normalizer.config.overlap_strategy == "truncate"
        assert normalizer.config.gap_strategy == "ignore"

    def test_custom_config_init(self):
        """Test initialization with custom config"""
        config = NormalizationConfig(overlap_strategy="merge")
        normalizer = CaptionNormalizer(config)

        assert normalizer.config.overlap_strategy == "merge"


@pytest.mark.fast
class TestNormalizeEmptyInput:
    """Test normalize with empty or edge case input"""

    def test_normalize_empty_list(self):
        """Test normalizing empty segment list"""
        normalizer = CaptionNormalizer()

        result, _ = normalizer.normalize([])

        assert result == []

    def test_normalize_single_segment(self):
        """Test normalizing single segment"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 0.0, 5.0, "Hello", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].text == "Hello"
        assert result[0].start_time == 0.0
        assert result[0].end_time == 5.0


@pytest.mark.fast
class TestValidateSegment:
    """Test individual segment validation"""

    def test_valid_segment_unchanged(self):
        """Test valid segment passes through unchanged"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 1.0, 5.0, "Valid", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 1.0
        assert result[0].end_time == 5.0

    def test_negative_start_time_fixed(self):
        """Test negative start_time is fixed to 0"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, -1.0, 5.0, "Text", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 5.0

    def test_negative_end_time_removed(self):
        """Test segment with negative end_time is removed"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 1.0, -2.0, "Text", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 0

    def test_end_before_start_fixed(self):
        """Test end_time < start_time is fixed by extending end"""
        config = NormalizationConfig(min_segment_duration=0.5)
        normalizer = CaptionNormalizer(config)
        segments = [CaptionSegment(0, 5.0, 3.0, "Text", "vid1")]  # end < start

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 5.0
        assert result[0].end_time == 5.5  # Extended by min_segment_duration

    def test_end_equals_start_fixed(self):
        """Test end_time == start_time is fixed"""
        config = NormalizationConfig(min_segment_duration=0.1)
        normalizer = CaptionNormalizer(config)
        segments = [CaptionSegment(0, 5.0, 5.0, "Text", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].end_time == 5.1

    def test_short_duration_extended(self):
        """Test segment with duration below minimum is extended"""
        config = NormalizationConfig(min_segment_duration=1.0)
        normalizer = CaptionNormalizer(config)
        segments = [CaptionSegment(0, 5.0, 5.05, "Text", "vid1")]  # 0.05s duration

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].end_time == 6.0  # Extended to min_segment_duration

    def test_empty_text_removed(self):
        """Test segment with empty text is removed"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 0.0, 5.0, "", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 0

    def test_whitespace_only_text_removed(self):
        """Test segment with whitespace-only text is removed"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 0.0, 5.0, "   \t\n  ", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 0

    def test_validation_can_be_disabled(self):
        """Test that validation can be disabled"""
        config = NormalizationConfig(validate_timestamps=False)
        normalizer = CaptionNormalizer(config)
        # This segment would normally be fixed
        segments = [CaptionSegment(0, -1.0, 5.0, "Text", "vid1")]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == -1.0  # Not fixed

    def test_source_file_set_from_video_id(self):
        """Test that source_file is set from video_id if missing"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 0.0, 5.0, "Text", "")]  # Empty source_file

        result, _ = normalizer.normalize(segments, video_id="test_video")

        assert result[0].source_file == "test_video"


@pytest.mark.fast
class TestOverlapHandling:
    """Test overlapping segment handling"""

    def test_no_overlap_unchanged(self):
        """Test segments without overlap pass through"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 6.0, 10.0, "Second", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].end_time == 5.0
        assert result[1].start_time == 6.0

    def test_overlap_truncate_strategy(self):
        """Test truncate strategy for overlapping segments"""
        config = NormalizationConfig(overlap_strategy="truncate")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 8.0, "First", "vid1"),
            CaptionSegment(1, 5.0, 12.0, "Second", "vid1"),  # Overlaps 5-8
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].start_time == 0.0
        assert result[0].end_time == 5.0  # Truncated to second's start
        assert result[1].start_time == 5.0
        assert result[1].end_time == 12.0

    def test_overlap_merge_strategy(self):
        """Test merge strategy for overlapping segments"""
        config = NormalizationConfig(overlap_strategy="merge")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 8.0, "First", "vid1"),
            CaptionSegment(1, 5.0, 12.0, "Second", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1  # Merged into one
        assert result[0].start_time == 0.0
        assert result[0].end_time == 12.0
        assert "First" in result[0].text
        assert "Second" in result[0].text

    def test_overlap_split_strategy(self):
        """Test split strategy for overlapping segments"""
        config = NormalizationConfig(overlap_strategy="split")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 8.0, "First", "vid1"),
            CaptionSegment(1, 6.0, 12.0, "Second", "vid1"),  # Overlaps 6-8
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        # Midpoint of overlap: (8 + 6) / 2 = 7
        assert result[0].end_time == 7.0
        assert result[1].start_time == 7.0

    def test_multiple_overlaps(self):
        """Test handling multiple consecutive overlaps"""
        config = NormalizationConfig(overlap_strategy="truncate")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(1, 4.0, 8.0, "B", "vid1"),  # Overlaps with A
            CaptionSegment(2, 7.0, 12.0, "C", "vid1"),  # Overlaps with B
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 3
        # Each segment's end should be truncated to next's start
        assert result[0].end_time == 4.0
        assert result[1].start_time == 4.0
        assert result[1].end_time == 7.0
        assert result[2].start_time == 7.0

    def test_complete_overlap_merge(self):
        """Test merging when one segment completely contains another"""
        config = NormalizationConfig(overlap_strategy="merge")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 10.0, "Outer", "vid1"),
            CaptionSegment(1, 3.0, 7.0, "Inner", "vid1"),  # Completely inside first
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 10.0
        assert "Outer" in result[0].text
        assert "Inner" in result[0].text


@pytest.mark.fast
class TestGapHandling:
    """Test gap segment handling"""

    def test_gap_ignore_strategy(self):
        """Test ignore strategy leaves gaps as-is"""
        config = NormalizationConfig(gap_strategy="ignore")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 10.0, 15.0, "Second", "vid1"),  # 5s gap
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].end_time == 5.0
        assert result[1].start_time == 10.0  # Gap preserved

    def test_gap_extend_small_gap(self):
        """Test extend strategy extends for small gaps"""
        config = NormalizationConfig(
            gap_strategy="extend",
            max_gap_to_extend=2.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 6.0, 10.0, "Second", "vid1"),  # 1s gap (small)
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].end_time == 6.0  # Extended to fill gap
        assert result[1].start_time == 6.0

    def test_gap_extend_large_gap_placeholder(self):
        """Test extend strategy uses placeholder for large gaps"""
        config = NormalizationConfig(
            gap_strategy="extend",
            max_gap_to_extend=2.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 10.0, 15.0, "Second", "vid1"),  # 5s gap (large)
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 3  # Placeholder inserted
        assert result[0].end_time == 5.0
        assert result[1].start_time == 5.0  # Placeholder fills gap
        assert result[1].end_time == 10.0
        assert result[1].text == ""  # Placeholder has empty text
        assert result[2].start_time == 10.0

    def test_gap_placeholder_strategy(self):
        """Test placeholder strategy always inserts placeholders"""
        config = NormalizationConfig(gap_strategy="placeholder")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 6.0, 10.0, "Second", "vid1"),  # 1s gap
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 3  # Placeholder always inserted
        assert result[1].start_time == 5.0
        assert result[1].end_time == 6.0
        assert result[1].text == ""

    def test_no_gap_no_change(self):
        """Test contiguous segments unchanged"""
        config = NormalizationConfig(gap_strategy="extend")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "Second", "vid1"),  # No gap
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].end_time == 5.0
        assert result[1].start_time == 5.0


@pytest.mark.fast
class TestReindexing:
    """Test segment reindexing"""

    def test_segments_reindexed_sequentially(self):
        """Test segments are reindexed from 0"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(5, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(10, 6.0, 10.0, "B", "vid1"),
            CaptionSegment(3, 11.0, 15.0, "C", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].index == 0
        assert result[1].index == 1
        assert result[2].index == 2

    def test_sorting_by_start_time(self):
        """Test segments are sorted by start_time before reindexing"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 10.0, 15.0, "Third", "vid1"),
            CaptionSegment(1, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(2, 6.0, 9.0, "Second", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].text == "First"
        assert result[0].index == 0
        assert result[1].text == "Second"
        assert result[1].index == 1
        assert result[2].text == "Third"
        assert result[2].index == 2


@pytest.mark.fast
class TestValidateContinuity:
    """Test validate_continuity method"""

    def test_valid_segments_no_warnings(self):
        """Test valid segments produce no warnings"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "B", "vid1"),
        ]

        warnings = normalizer.validate_continuity(segments)

        assert len(warnings) == 0

    def test_end_before_start_warning(self):
        """Test end < start produces warning"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 5.0, 3.0, "A", "vid1")]

        warnings = normalizer.validate_continuity(segments)

        assert len(warnings) >= 1
        assert any("end_time" in w and "start_time" in w for w in warnings)

    def test_zero_duration_warning(self):
        """Test zero duration produces warning"""
        normalizer = CaptionNormalizer()
        segments = [CaptionSegment(0, 5.0, 5.0, "A", "vid1")]

        warnings = normalizer.validate_continuity(segments)

        assert len(warnings) >= 1
        assert any("duration" in w for w in warnings)

    def test_overlap_warning(self):
        """Test overlapping segments produce warning"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 8.0, "A", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "B", "vid1"),
        ]

        warnings = normalizer.validate_continuity(segments)

        assert len(warnings) >= 1
        assert any("overlap" in w.lower() for w in warnings)


@pytest.mark.fast
class TestTimestampConversion:
    """Test timestamp format conversion"""

    def test_convert_vtt_timestamp(self):
        """Test converting VTT timestamp to seconds"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("01:30:45.123")

        assert result == 5445.123

    def test_convert_srt_timestamp(self):
        """Test converting SRT timestamp to seconds"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("01:30:45,123")

        assert result == 5445.123

    def test_convert_short_vtt_timestamp(self):
        """Test converting short VTT timestamp (MM:SS.mmm)"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("05:30.500")

        assert result == 330.5

    def test_convert_float_string(self):
        """Test converting float string directly"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("123.456")

        assert result == 123.456

    def test_convert_timestamp_without_millis(self):
        """Test converting timestamp without milliseconds"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("01:30:45")

        assert result == 5445.0

    def test_convert_empty_string(self):
        """Test converting empty string returns None"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("")

        assert result is None

    def test_convert_invalid_string(self):
        """Test converting invalid string returns None"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("not a timestamp")

        assert result is None

    def test_convert_with_whitespace(self):
        """Test converting timestamp with surrounding whitespace"""
        result = CaptionNormalizer.convert_timestamp_to_seconds("  01:30:45.123  ")

        assert result == 5445.123


@pytest.mark.fast
class TestSecondsToTimestamp:
    """Test seconds to timestamp conversion"""

    def test_seconds_to_vtt(self):
        """Test converting seconds to VTT format"""
        result = CaptionNormalizer.seconds_to_vtt_timestamp(5445.123)

        assert result == "01:30:45.123"

    def test_seconds_to_vtt_zero(self):
        """Test converting 0 to VTT format"""
        result = CaptionNormalizer.seconds_to_vtt_timestamp(0.0)

        assert result == "00:00:00.000"

    def test_seconds_to_vtt_negative(self):
        """Test negative seconds defaults to 0"""
        result = CaptionNormalizer.seconds_to_vtt_timestamp(-10.0)

        assert result == "00:00:00.000"

    def test_seconds_to_srt(self):
        """Test converting seconds to SRT format"""
        result = CaptionNormalizer.seconds_to_srt_timestamp(5445.123)

        assert result == "01:30:45,123"

    def test_seconds_to_vtt_hours(self):
        """Test converting large duration with hours"""
        result = CaptionNormalizer.seconds_to_vtt_timestamp(7323.456)

        assert result == "02:02:03.456"


@pytest.mark.fast
class TestCaptionNormalizationError:
    """Test CaptionNormalizationError exception"""

    def test_error_with_reason(self):
        """Test error with reason message"""
        error = CaptionNormalizationError("Invalid format")

        assert "Invalid format" in str(error)
        assert error.reason == "Invalid format"

    def test_error_without_reason(self):
        """Test error without reason message"""
        error = CaptionNormalizationError()

        assert "normalization failed" in str(error).lower()
        assert error.reason == ""


@pytest.mark.fast
class TestComplexScenarios:
    """Test complex normalization scenarios"""

    def test_all_segments_invalid(self):
        """Test when all segments are invalid"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "", "vid1"),  # Empty text
            CaptionSegment(1, 1.0, -2.0, "Text", "vid1"),  # Negative end
        ]

        result, _ = normalizer.normalize(segments)

        assert result == []

    def test_mixed_valid_invalid_segments(self):
        """Test mix of valid and invalid segments"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Valid", "vid1"),
            CaptionSegment(1, 6.0, 10.0, "", "vid1"),  # Invalid (empty)
            CaptionSegment(2, 11.0, 15.0, "Also valid", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].text == "Valid"
        assert result[1].text == "Also valid"
        assert result[0].index == 0
        assert result[1].index == 1

    def test_multiple_merges_chain(self):
        """Test chained merges when multiple segments overlap"""
        config = NormalizationConfig(overlap_strategy="merge")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(1, 4.0, 8.0, "B", "vid1"),  # Overlaps A
            CaptionSegment(2, 7.0, 12.0, "C", "vid1"),  # Overlaps B (merged)
        ]

        result, _ = normalizer.normalize(segments)

        # A+B merged, then result+C merged
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 12.0
        assert "A" in result[0].text
        assert "B" in result[0].text
        assert "C" in result[0].text

    def test_overlap_and_gap_combined(self):
        """Test handling both overlaps and gaps"""
        config = NormalizationConfig(
            overlap_strategy="truncate",
            gap_strategy="extend",
            max_gap_to_extend=1.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(1, 4.0, 8.0, "B", "vid1"),  # Overlaps A
            CaptionSegment(2, 8.5, 12.0, "C", "vid1"),  # Small gap after B
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 3
        # A truncated to B's start
        assert result[0].end_time == 4.0
        # B extended to fill small gap
        assert result[1].end_time == 8.5
        assert result[2].start_time == 8.5

    def test_unsorted_input(self):
        """Test that unsorted input is properly sorted"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(2, 20.0, 25.0, "Third", "vid1"),
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 10.0, 15.0, "Second", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].text == "First"
        assert result[1].text == "Second"
        assert result[2].text == "Third"

    def test_real_world_vtt_scenario(self):
        """Test with realistic VTT caption timing patterns"""
        normalizer = CaptionNormalizer()
        # Simulating typical YouTube auto-caption timing
        segments = [
            CaptionSegment(0, 0.0, 2.5, "Hello everyone", "vid1"),
            CaptionSegment(1, 2.4, 4.8, "welcome to this video", "vid1"),  # Slight overlap
            CaptionSegment(2, 4.7, 7.2, "today we'll learn", "vid1"),  # Slight overlap
            CaptionSegment(3, 7.1, 9.5, "about normalization", "vid1"),  # Slight overlap
        ]

        result, _ = normalizer.normalize(segments)

        # All segments should be present with overlaps resolved
        assert len(result) == 4
        # No overlaps in result
        for i in range(len(result) - 1):
            assert result[i].end_time <= result[i + 1].start_time

    def test_preserve_source_file(self):
        """Test that source_file is preserved through normalization"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "A", "original_video"),
            CaptionSegment(1, 6.0, 10.0, "B", "original_video"),
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].source_file == "original_video"
        assert result[1].source_file == "original_video"


@pytest.mark.fast
class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_overlap(self):
        """Test handling very small overlaps (< 0.01s)"""
        config = NormalizationConfig(overlap_strategy="truncate")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.001, "A", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "B", "vid1"),  # 0.001s overlap
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        assert result[0].end_time <= result[1].start_time

    def test_very_small_gap(self):
        """Test handling very small gaps (< 0.01s)"""
        config = NormalizationConfig(gap_strategy="extend", max_gap_to_extend=0.1)
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 4.999, "A", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "B", "vid1"),  # 0.001s gap
        ]

        result, _ = normalizer.normalize(segments)

        assert len(result) == 2
        # Gap should be extended
        assert result[0].end_time == 5.0

    def test_unicode_text_preserved(self):
        """Test that unicode text is preserved"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Привет 世界 🌍", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].text == "Привет 世界 🌍"

    def test_very_long_segment(self):
        """Test handling very long segment (hours)"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 7200.0, "Two hour segment", "vid1"),  # 2 hours
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].end_time == 7200.0

    def test_millisecond_precision(self):
        """Test millisecond precision is maintained"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.001, 5.999, "Precise", "vid1"),
        ]

        result, _ = normalizer.normalize(segments)

        assert result[0].start_time == 0.001
        assert result[0].end_time == 5.999


@pytest.mark.fast
class TestPartialRecovery:
    """Test partial recovery from malformed segments (US-005)."""

    def test_skipped_count_zero_for_valid_segments(self):
        """Test skipped_count is 0 when all segments are valid"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Valid 1", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "Valid 2", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 2
        assert skipped == 0

    def test_skipped_count_for_filtered_segments(self):
        """Test that intentionally filtered segments don't count as skipped"""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Valid", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "", "vid1"),  # Empty text - filtered
            CaptionSegment(2, 10.0, 15.0, "Also valid", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 2
        # Skipped count is 0 because empty text filtering is intentional, not an error
        assert skipped == 0

    def test_partial_recovery_with_malformed_segment(self):
        """Test that malformed segments are skipped but valid ones are kept"""
        normalizer = CaptionNormalizer()

        # Create a mock segment with broken __getattribute__ to simulate parsing error
        class MalformedSegment:
            def __init__(self, idx):
                self.index = idx
                self.start_time = 0.0
                self.end_time = 5.0
                self.source_file = "vid1"
                self._text_accessed = False

            @property
            def text(self):
                if not self._text_accessed:
                    self._text_accessed = True
                    raise ValueError("Simulated parse error")
                return "Valid"

        segments = [
            CaptionSegment(0, 0.0, 5.0, "Valid 1", "vid1"),
            MalformedSegment(1),  # This will throw when text is accessed
            CaptionSegment(2, 10.0, 15.0, "Valid 2", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments, video_id="test")

        # Only valid segments should be returned
        assert len(result) == 2
        assert result[0].text == "Valid 1"
        assert result[1].text == "Valid 2"
        # One segment was skipped due to error
        assert skipped == 1

    def test_partial_recovery_threshold_raises_when_exceeded(self):
        """Test that CaptionNormalizationError is raised when error ratio exceeds threshold"""
        normalizer = CaptionNormalizer()

        # Create multiple malformed segments
        class MalformedSegment:
            def __init__(self, idx):
                self.index = idx
                self.start_time = float(idx)
                self.end_time = float(idx + 1)
                self.source_file = "vid1"

            @property
            def text(self):
                raise ValueError("Simulated parse error")

        # 8 malformed, 2 valid = 80% error rate > 50% threshold
        segments = [MalformedSegment(i) for i in range(8)]
        segments.extend([
            CaptionSegment(8, 8.0, 9.0, "Valid 1", "vid1"),
            CaptionSegment(9, 9.0, 10.0, "Valid 2", "vid1"),
        ])

        with pytest.raises(CaptionNormalizationError) as exc_info:
            normalizer.normalize(segments, video_id="test")

        assert "parse errors" in str(exc_info.value).lower()

    def test_partial_recovery_threshold_customizable(self):
        """Test that min_success_ratio parameter controls the threshold"""
        normalizer = CaptionNormalizer()

        class MalformedSegment:
            def __init__(self, idx):
                self.index = idx
                self.start_time = float(idx)
                self.end_time = float(idx + 1)
                self.source_file = "vid1"

            @property
            def text(self):
                raise ValueError("Simulated parse error")

        # 3 malformed, 7 valid = 30% error rate
        segments = [MalformedSegment(i) for i in range(3)]
        segments.extend([
            CaptionSegment(i, float(i), float(i + 1), f"Valid {i}", "vid1")
            for i in range(3, 10)
        ])

        # With default 50% threshold (error tolerance 50%), 30% error rate should pass
        result, skipped = normalizer.normalize(segments, video_id="test", min_success_ratio=0.5)
        assert len(result) == 7
        assert skipped == 3

        # With stricter 80% threshold (error tolerance 20%), 30% error rate should fail
        with pytest.raises(CaptionNormalizationError):
            normalizer.normalize(segments, video_id="test", min_success_ratio=0.8)

    def test_skipped_segments_logged(self, caplog):
        """Test that skipped segments are logged with warning"""
        import logging
        normalizer = CaptionNormalizer()

        class MalformedSegment:
            def __init__(self):
                self.index = 1
                self.start_time = 5.0
                self.end_time = 10.0
                self.source_file = "vid1"

            @property
            def text(self):
                raise ValueError("Test error message")

        segments = [
            CaptionSegment(0, 0.0, 5.0, "Valid", "vid1"),
            MalformedSegment(),
        ]

        with caplog.at_level(logging.WARNING):
            result, skipped = normalizer.normalize(segments, video_id="test_video")

        assert skipped == 1
        assert any("Skipped malformed segment" in record.message for record in caplog.records)
        assert any("test_video" in record.message for record in caplog.records)

    def test_empty_list_returns_zero_skipped(self):
        """Test that empty input returns zero skipped count"""
        normalizer = CaptionNormalizer()

        result, skipped = normalizer.normalize([])

        assert result == []
        assert skipped == 0

    def test_all_filtered_returns_empty_with_zero_skipped(self):
        """Test that all segments filtered returns empty list with zero skipped"""
        normalizer = CaptionNormalizer()
        # All segments have empty text - filtered, not errors
        segments = [
            CaptionSegment(0, 0.0, 5.0, "", "vid1"),
            CaptionSegment(1, 5.0, 10.0, "   ", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert result == []
        assert skipped == 0  # Filtering is not an error


@pytest.mark.fast
class TestCaptionParseWarning:
    """Test CaptionParseWarning exception class (US-005)."""

    def test_warning_with_all_fields(self):
        """Test CaptionParseWarning with all fields"""
        from src.caption_fetcher import CaptionParseWarning

        warning = CaptionParseWarning("abc123", 5, "Invalid timestamp format")

        assert warning.video_id == "abc123"
        assert warning.segment_index == 5
        assert warning.reason == "Invalid timestamp format"
        assert "abc123" in str(warning)
        assert "5" in str(warning)
        assert "Invalid timestamp format" in str(warning)

    def test_warning_without_reason(self):
        """Test CaptionParseWarning without reason"""
        from src.caption_fetcher import CaptionParseWarning

        warning = CaptionParseWarning("xyz789", 0)

        assert warning.video_id == "xyz789"
        assert warning.segment_index == 0
        assert warning.reason == ""
        assert "xyz789" in str(warning)

    def test_warning_inherits_from_caption_error(self):
        """Test CaptionParseWarning inherits from CaptionError"""
        from src.caption_fetcher import CaptionParseWarning, CaptionError

        warning = CaptionParseWarning("vid", 0)

        assert isinstance(warning, CaptionError)
        assert isinstance(warning, Exception)


@pytest.mark.fast
class TestCaptionResultSkippedCount:
    """Test CaptionResult.skipped_segments_count field (US-005)."""

    def test_default_skipped_count_is_zero(self):
        """Test that skipped_segments_count defaults to 0"""
        from src.caption_fetcher import CaptionResult

        result = CaptionResult(video_id="abc123")

        assert result.skipped_segments_count == 0

    def test_skipped_count_in_constructor(self):
        """Test setting skipped_segments_count in constructor"""
        from src.caption_fetcher import CaptionResult

        result = CaptionResult(video_id="abc123", skipped_segments_count=5)

        assert result.skipped_segments_count == 5

    def test_skipped_count_in_to_dict(self):
        """Test that skipped_segments_count is included in to_dict()"""
        from src.caption_fetcher import CaptionResult

        result = CaptionResult(video_id="abc123", skipped_segments_count=3)
        data = result.to_dict()

        assert "skipped_segments_count" in data
        assert data["skipped_segments_count"] == 3

    def test_skipped_count_zero_in_to_dict(self):
        """Test that skipped_segments_count=0 is included in to_dict()"""
        from src.caption_fetcher import CaptionResult

        result = CaptionResult(video_id="abc123")
        data = result.to_dict()

        assert "skipped_segments_count" in data
        assert data["skipped_segments_count"] == 0


@pytest.mark.fast
class TestNormalizerEdgeCasesSprint6:
    """Test normalizer edge cases for corrupted segments (US-009 Sprint 6).

    These tests verify the CaptionNormalizer handles edge cases like:
    - Negative start times (clipped to 0)
    - Negative durations (start > end) - extended to min_duration
    - Overlapping segments with text merge
    - Duplicate segment indices re-indexed
    - Large gaps creating placeholder segments
    """

    def test_negative_start_time_severe_fixed_to_zero(self):
        """Test segment with start=-5, end=2 is fixed to start=0.

        AC: segment with start=-5, end=2 fixed to start=0, end>=min_duration

        Expected behavior:
        - start_time < 0 is clamped to 0
        - If duration becomes too short, end is extended to min_segment_duration
        """
        config = NormalizationConfig(min_segment_duration=0.5)
        normalizer = CaptionNormalizer(config)
        # start=-5 should become 0, end=2 remains (duration 2s > min 0.5s)
        segments = [CaptionSegment(0, -5.0, 2.0, "Negative start text", "vid1")]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 0.0  # Fixed from -5 to 0
        assert result[0].end_time == 2.0  # Unchanged (duration 2s is valid)
        assert result[0].text == "Negative start text"
        assert skipped == 0

    def test_negative_start_requiring_duration_extension(self):
        """Test negative start where fixing creates short duration.

        When start=-5, end=0.01, clamping start to 0 creates 0.01s duration.
        If min_segment_duration is 0.5s, end should be extended to 0.5s.
        """
        config = NormalizationConfig(min_segment_duration=0.5)
        normalizer = CaptionNormalizer(config)
        # start=-5 becomes 0, end=0.01 creates 0.01s duration < 0.5s min
        segments = [CaptionSegment(0, -5.0, 0.01, "Short after fix", "vid1")]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 0.0  # Clamped from -5
        assert result[0].end_time >= 0.5  # Extended to meet min duration
        assert result[0].text == "Short after fix"

    def test_negative_duration_start_greater_than_end(self):
        """Test segment with negative duration (start > end) gets fixed.

        AC: segment with negative duration (start > end) gets swapped or skipped

        When start=8, end=3 (negative 5s duration), the normalizer:
        - Detects end <= start condition
        - Extends end to start + min_segment_duration
        """
        config = NormalizationConfig(min_segment_duration=0.5)
        normalizer = CaptionNormalizer(config)
        # start=8, end=3 -> negative duration, end should be extended
        segments = [CaptionSegment(0, 8.0, 3.0, "Inverted times", "vid1")]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 8.0
        # end < start condition triggers fix: end = start + min_segment_duration
        assert result[0].end_time == 8.5
        assert result[0].text == "Inverted times"

    def test_negative_duration_with_larger_min_duration(self):
        """Test negative duration fixed to larger minimum duration.

        Verifies the end time extension uses the configured min_segment_duration.
        """
        config = NormalizationConfig(min_segment_duration=2.0)
        normalizer = CaptionNormalizer(config)
        segments = [CaptionSegment(0, 10.0, 5.0, "Inverted 5s", "vid1")]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 10.0
        assert result[0].end_time == 12.0  # 10 + 2.0 min duration
        assert skipped == 0

    def test_overlapping_segments_merged_with_text_concatenation(self):
        """Test overlapping segments (0-5, 3-8) merged with concatenated text.

        AC: overlapping segments (0-5, 3-8) merged with concatenated text

        With merge strategy:
        - Segments (0-5) and (3-8) overlap from 3-5
        - Result should be single segment (0-8) with "First Second" text
        """
        config = NormalizationConfig(overlap_strategy="merge")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 3.0, 8.0, "Second", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 8.0
        # Text should be concatenated
        assert "First" in result[0].text
        assert "Second" in result[0].text
        assert result[0].text == "First Second"
        assert skipped == 0

    def test_three_overlapping_segments_merged(self):
        """Test chain of three overlapping segments all merged.

        Segments (0-5), (4-9), (8-12) should merge into single (0-12).
        """
        config = NormalizationConfig(overlap_strategy="merge")
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "One", "vid1"),
            CaptionSegment(1, 4.0, 9.0, "Two", "vid1"),
            CaptionSegment(2, 8.0, 12.0, "Three", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 12.0
        assert "One" in result[0].text
        assert "Two" in result[0].text
        assert "Three" in result[0].text

    def test_duplicate_indices_reindexed_sequentially(self):
        """Test duplicate segment indices are re-indexed sequentially.

        AC: duplicate segment indices re-indexed sequentially

        All segments have index=5, after normalization they should be 0, 1, 2.
        """
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(5, 0.0, 3.0, "All", "vid1"),
            CaptionSegment(5, 4.0, 7.0, "have", "vid1"),
            CaptionSegment(5, 8.0, 11.0, "same index", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 3
        assert result[0].index == 0
        assert result[1].index == 1
        assert result[2].index == 2
        # Text preserved correctly
        assert result[0].text == "All"
        assert result[1].text == "have"
        assert result[2].text == "same index"
        assert skipped == 0

    def test_non_sequential_indices_reindexed(self):
        """Test non-sequential indices (100, 50, 200) become (0, 1, 2)."""
        normalizer = CaptionNormalizer()
        segments = [
            CaptionSegment(100, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(50, 6.0, 10.0, "B", "vid1"),
            CaptionSegment(200, 11.0, 15.0, "C", "vid1"),
        ]

        result, skipped = normalizer.normalize(segments)

        assert len(result) == 3
        assert [s.index for s in result] == [0, 1, 2]

    def test_large_gap_creates_placeholder_segment(self):
        """Test gap > max_gap creates placeholder segment preserving timing.

        AC: gap > max_gap creates placeholder segment preserving timing

        With extend strategy and max_gap_to_extend=2.0:
        - Gap of 10s (5-15) exceeds threshold
        - Placeholder segment should fill 5.0-15.0
        """
        config = NormalizationConfig(
            gap_strategy="extend",
            max_gap_to_extend=2.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Before gap", "vid1"),
            CaptionSegment(1, 15.0, 20.0, "After gap", "vid1"),  # 10s gap
        ]

        result, skipped = normalizer.normalize(segments)

        # Should have 3 segments: original + placeholder + original
        assert len(result) == 3

        # First segment unchanged
        assert result[0].start_time == 0.0
        assert result[0].end_time == 5.0
        assert result[0].text == "Before gap"

        # Placeholder segment fills the gap
        assert result[1].start_time == 5.0
        assert result[1].end_time == 15.0
        assert result[1].text == ""  # Placeholder has empty text

        # Last segment unchanged
        assert result[2].start_time == 15.0
        assert result[2].end_time == 20.0
        assert result[2].text == "After gap"

        # Indices should be sequential
        assert result[0].index == 0
        assert result[1].index == 1
        assert result[2].index == 2

    def test_gap_exactly_at_threshold_extends_not_placeholder(self):
        """Test gap exactly at max_gap_to_extend extends, doesn't create placeholder."""
        config = NormalizationConfig(
            gap_strategy="extend",
            max_gap_to_extend=2.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 7.0, 10.0, "Second", "vid1"),  # 2s gap (exactly at threshold)
        ]

        result, skipped = normalizer.normalize(segments)

        # Gap <= threshold means extend, not placeholder
        assert len(result) == 2
        assert result[0].end_time == 7.0  # Extended to fill gap
        assert result[1].start_time == 7.0

    def test_gap_just_over_threshold_creates_placeholder(self):
        """Test gap just over max_gap_to_extend creates placeholder."""
        config = NormalizationConfig(
            gap_strategy="extend",
            max_gap_to_extend=2.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First", "vid1"),
            CaptionSegment(1, 7.01, 10.0, "Second", "vid1"),  # 2.01s gap (just over)
        ]

        result, skipped = normalizer.normalize(segments)

        # Gap > threshold means placeholder
        assert len(result) == 3
        assert result[1].text == ""  # Placeholder
        assert result[1].start_time == 5.0
        assert result[1].end_time == 7.01

    def test_multiple_gaps_with_mixed_sizes(self):
        """Test multiple gaps with some under and some over threshold."""
        config = NormalizationConfig(
            gap_strategy="extend",
            max_gap_to_extend=2.0
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "A", "vid1"),
            CaptionSegment(1, 6.0, 10.0, "B", "vid1"),  # 1s gap (extend)
            CaptionSegment(2, 20.0, 25.0, "C", "vid1"),  # 10s gap (placeholder)
        ]

        result, skipped = normalizer.normalize(segments)

        # First gap extended, second gap placeholder
        assert len(result) == 4  # A, B, placeholder, C
        assert result[0].end_time == 6.0  # Extended
        assert result[2].text == ""  # Placeholder
        assert result[2].start_time == 10.0
        assert result[2].end_time == 20.0

    def test_combined_edge_cases(self):
        """Test combined edge cases: negative start, overlap, gap.

        Comprehensive test combining multiple edge cases in one scenario.
        """
        config = NormalizationConfig(
            overlap_strategy="truncate",
            gap_strategy="extend",
            max_gap_to_extend=1.0,
            min_segment_duration=0.5
        )
        normalizer = CaptionNormalizer(config)
        segments = [
            CaptionSegment(99, -2.0, 3.0, "Negative start", "vid1"),  # Fix start to 0
            CaptionSegment(99, 2.0, 6.0, "Overlaps prev", "vid1"),  # Overlap, truncate
            CaptionSegment(99, 10.0, 15.0, "After gap", "vid1"),  # 4s gap > 1s threshold
        ]

        result, skipped = normalizer.normalize(segments)

        # Should have: fixed segment, truncated segment, placeholder, last segment
        assert len(result) == 4

        # First segment: start fixed from -2 to 0, end truncated to 2.0
        assert result[0].start_time == 0.0
        assert result[0].end_time == 2.0  # Truncated due to overlap

        # Second segment starts at 2.0
        assert result[1].start_time == 2.0
        assert result[1].end_time == 6.0

        # Placeholder for gap
        assert result[2].text == ""
        assert result[2].start_time == 6.0
        assert result[2].end_time == 10.0

        # Last segment unchanged
        assert result[3].start_time == 10.0
        assert result[3].end_time == 15.0

        # All re-indexed sequentially
        assert [s.index for s in result] == [0, 1, 2, 3]
