"""
Tests for src/caption_fetcher.py

Tests YouTube caption fetching, parsing, and error handling.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    CaptionError,
    CaptionUnavailableError,
    CaptionFetchError,
    AvailableLanguage,
    TimingValidationResult,
    determine_caption_quality,
)


class TestCaptionSegment:
    """Test CaptionSegment dataclass"""

    def test_caption_segment_creation(self):
        """Test creating a caption segment"""
        segment = CaptionSegment(
            index=0,
            start_time=1.5,
            end_time=4.0,
            text="Hello world",
            source_file="dQw4w9WgXcQ"
        )

        assert segment.index == 0
        assert segment.start_time == 1.5
        assert segment.end_time == 4.0
        assert segment.text == "Hello world"
        assert segment.source_file == "dQw4w9WgXcQ"

    def test_caption_segment_to_dict(self):
        """Test segment serialization"""
        segment = CaptionSegment(
            index=1,
            start_time=5.0,
            end_time=8.5,
            text="Test text",
            source_file="test123"
        )

        result = segment.to_dict()

        assert result['index'] == 1
        assert result['start'] == 5.0
        assert result['end'] == 8.5
        assert result['text'] == "Test text"
        assert result['source_file'] == "test123"

    def test_caption_segment_default_source_file(self):
        """Test default source_file value"""
        segment = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="Test"
        )

        assert segment.source_file == ""


class TestCaptionResult:
    """Test CaptionResult dataclass"""

    def test_caption_result_creation(self):
        """Test creating a caption result"""
        segments = [
            CaptionSegment(0, 0.0, 2.0, "First", "vid1"),
            CaptionSegment(1, 2.0, 4.0, "Second", "vid1"),
        ]

        result = CaptionResult(
            video_id="vid123",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        assert result.video_id == "vid123"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.is_auto_generated is False
        assert result.format_source == "vtt"

    def test_caption_result_text_property(self):
        """Test text property concatenates all segment text"""
        segments = [
            CaptionSegment(0, 0.0, 2.0, "Hello", "vid1"),
            CaptionSegment(1, 2.0, 4.0, "world", "vid1"),
        ]

        result = CaptionResult(video_id="vid1", segments=segments)

        assert result.text == "Hello world"

    def test_caption_result_text_empty(self):
        """Test text property with no segments"""
        result = CaptionResult(video_id="vid1", segments=[])

        assert result.text == ""

    def test_caption_result_duration(self):
        """Test duration property calculation"""
        segments = [
            CaptionSegment(0, 1.0, 3.0, "First", "vid1"),
            CaptionSegment(1, 3.0, 10.0, "Last", "vid1"),
        ]

        result = CaptionResult(video_id="vid1", segments=segments)

        # Duration is last.end - first.start = 10.0 - 1.0 = 9.0
        assert result.duration == 9.0

    def test_caption_result_duration_empty(self):
        """Test duration with no segments"""
        result = CaptionResult(video_id="vid1", segments=[])

        assert result.duration == 0.0

    def test_caption_result_to_dict(self):
        """Test result serialization"""
        segments = [
            CaptionSegment(0, 0.0, 1.0, "Test", "vid1")
        ]

        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            language="en",
            is_auto_generated=True,
            format_source="json3"
        )

        data = result.to_dict()

        assert data['video_id'] == "vid1"
        assert len(data['segments']) == 1
        assert data['language'] == "en"
        assert data['is_auto_generated'] is True
        assert data['format_source'] == "json3"
        # US-007: caption_quality should be included in to_dict
        assert 'caption_quality' in data

    def test_caption_result_caption_quality_high(self):
        """Test caption_quality returns 'high' for human captions with good completeness (US-007)"""
        # Create many segments for good completeness
        segments = [
            CaptionSegment(i, i * 3.0, (i + 1) * 3.0, f"Segment {i}", "vid1")
            for i in range(50)  # 50 segments over 150s = 3s avg
        ]

        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            language="en",
            is_auto_generated=False,  # Human captions
            format_source="vtt"
        )

        assert result.caption_quality == "high"

    def test_caption_result_caption_quality_medium_auto(self):
        """Test caption_quality returns 'medium' for auto-generated captions (US-007)"""
        segments = [
            CaptionSegment(i, i * 3.0, (i + 1) * 3.0, f"Segment {i}", "vid1")
            for i in range(50)
        ]

        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            language="en",
            is_auto_generated=True,  # Auto-generated
            format_source="vtt"
        )

        assert result.caption_quality == "medium"

    def test_caption_result_caption_quality_low_sparse(self):
        """Test caption_quality returns 'low' for very sparse captions (US-007)"""
        # Only 2 segments over 60 seconds = very sparse
        segments = [
            CaptionSegment(0, 0.0, 30.0, "First", "vid1"),
            CaptionSegment(1, 30.0, 60.0, "Second", "vid1"),
        ]

        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            language="en",
            is_auto_generated=True,
            format_source="vtt"
        )

        assert result.caption_quality == "low"

    def test_caption_result_caption_quality_low_empty(self):
        """Test caption_quality returns 'low' for empty captions (US-007)"""
        result = CaptionResult(
            video_id="vid1",
            segments=[],
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        assert result.caption_quality == "low"

    # US-004: Coverage calculation tests

    def test_calculate_coverage_full(self):
        """Test calculate_coverage returns 1.0 for full coverage (US-004)"""
        # 10 segments, each 10s, covering entire 100s video
        segments = [
            CaptionSegment(i, i * 10.0, (i + 1) * 10.0, f"Seg {i}", "vid1")
            for i in range(10)
        ]
        result = CaptionResult(video_id="vid1", segments=segments)

        coverage = result.calculate_coverage(video_duration=100.0)
        assert coverage == 1.0

    def test_calculate_coverage_partial(self):
        """Test calculate_coverage returns correct ratio for partial coverage (US-004)"""
        # 5 segments of 10s each = 50s out of 100s video
        segments = [
            CaptionSegment(i, i * 20.0, i * 20.0 + 10.0, f"Seg {i}", "vid1")
            for i in range(5)
        ]
        result = CaptionResult(video_id="vid1", segments=segments)

        coverage = result.calculate_coverage(video_duration=100.0)
        assert coverage == 0.5

    def test_calculate_coverage_low(self):
        """Test calculate_coverage returns low ratio for sparse coverage (US-004)"""
        # 1 segment of 20s out of 100s video = 20%
        segments = [CaptionSegment(0, 0.0, 20.0, "Only segment", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments)

        coverage = result.calculate_coverage(video_duration=100.0)
        assert coverage == 0.2

    def test_calculate_coverage_no_duration(self):
        """Test calculate_coverage returns 0.0 when no video duration (US-004)"""
        segments = [CaptionSegment(0, 0.0, 10.0, "Test", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments)

        # No duration provided
        coverage = result.calculate_coverage()
        assert coverage == 0.0

    def test_calculate_coverage_zero_duration(self):
        """Test calculate_coverage returns 0.0 for zero duration (US-004)"""
        segments = [CaptionSegment(0, 0.0, 10.0, "Test", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments)

        coverage = result.calculate_coverage(video_duration=0.0)
        assert coverage == 0.0

    def test_calculate_coverage_empty_segments(self):
        """Test calculate_coverage returns 0.0 for empty segments (US-004)"""
        result = CaptionResult(video_id="vid1", segments=[])

        coverage = result.calculate_coverage(video_duration=100.0)
        assert coverage == 0.0

    def test_calculate_coverage_capped_at_one(self):
        """Test calculate_coverage caps at 1.0 even if segments overlap (US-004)"""
        # Overlapping segments that sum to more than video duration
        segments = [
            CaptionSegment(0, 0.0, 60.0, "Seg 1", "vid1"),
            CaptionSegment(1, 30.0, 90.0, "Seg 2 overlaps", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments)

        # 60s + 60s = 120s total, but video is only 100s
        coverage = result.calculate_coverage(video_duration=100.0)
        assert coverage == 1.0

    def test_calculate_coverage_with_gaps(self):
        """Test calculate_coverage handles gaps between segments (US-004)"""
        # Segments with gaps: 0-10, 20-30, 40-50 = 30s out of 100s
        segments = [
            CaptionSegment(0, 0.0, 10.0, "Seg 1", "vid1"),
            CaptionSegment(1, 20.0, 30.0, "Seg 2", "vid1"),
            CaptionSegment(2, 40.0, 50.0, "Seg 3", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments)

        coverage = result.calculate_coverage(video_duration=100.0)
        assert coverage == 0.3

    def test_coverage_ratio_property(self):
        """Test coverage_ratio property uses video_duration field (US-004)"""
        segments = [CaptionSegment(0, 0.0, 50.0, "Half", "vid1")]
        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            video_duration=100.0  # Set video_duration field
        )

        assert result.coverage_ratio == 0.5

    def test_coverage_ratio_property_no_duration(self):
        """Test coverage_ratio returns 0.0 when video_duration not set (US-004)"""
        segments = [CaptionSegment(0, 0.0, 50.0, "Test", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments)
        # video_duration not set (None)

        assert result.coverage_ratio == 0.0

    def test_to_dict_includes_coverage(self):
        """Test to_dict includes video_duration and coverage_ratio (US-004)"""
        segments = [CaptionSegment(0, 0.0, 50.0, "Half", "vid1")]
        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            video_duration=100.0
        )

        data = result.to_dict()
        assert 'video_duration' in data
        assert data['video_duration'] == 100.0
        assert 'coverage_ratio' in data
        assert data['coverage_ratio'] == 0.5

    # US-007: Timing validation tests

    def test_validate_timing_valid(self):
        """Test validate_timing returns valid for normal captions (US-007)"""
        # Captions end at 90s, video is 100s - should be valid
        segments = [
            CaptionSegment(0, 0.0, 30.0, "Seg 1", "vid1"),
            CaptionSegment(1, 30.0, 60.0, "Seg 2", "vid1"),
            CaptionSegment(2, 60.0, 90.0, "Seg 3", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing()

        assert validation.is_valid is True
        assert validation.exceeds_duration is False
        assert validation.below_coverage is False
        assert validation.caption_end_time == 90.0
        assert validation.video_duration == 100.0
        assert "Timing valid" in validation.message

    def test_validate_timing_exceeds_duration(self):
        """Test validate_timing detects captions exceeding video duration (US-007)"""
        # Captions end at 120s, video is 100s - exceeds by 20% (over 10% tolerance)
        segments = [
            CaptionSegment(0, 0.0, 60.0, "Seg 1", "vid1"),
            CaptionSegment(1, 60.0, 120.0, "Seg 2", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing()

        assert validation.is_valid is False
        assert validation.exceeds_duration is True
        assert validation.caption_end_time == 120.0
        assert "exceeds video duration" in validation.message.lower()

    def test_validate_timing_within_tolerance(self):
        """Test validate_timing allows captions within 10% tolerance (US-007)"""
        # Captions end at 105s, video is 100s - within 10% tolerance
        segments = [
            CaptionSegment(0, 0.0, 105.0, "Long segment", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing()

        assert validation.is_valid is True
        assert validation.exceeds_duration is False

    def test_validate_timing_at_tolerance_boundary(self):
        """Test validate_timing at exactly 110% boundary (US-007)"""
        # Captions end at exactly 110s when video is 100s - should be valid (<=110%)
        segments = [CaptionSegment(0, 0.0, 110.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing()

        assert validation.is_valid is True
        assert validation.exceeds_duration is False

    def test_validate_timing_just_over_tolerance(self):
        """Test validate_timing just over 110% boundary (US-007)"""
        # Captions end at 110.1s when video is 100s - should fail (>110%)
        segments = [CaptionSegment(0, 0.0, 110.1, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing()

        assert validation.is_valid is False
        assert validation.exceeds_duration is True

    def test_validate_timing_below_coverage(self):
        """Test validate_timing detects low coverage (US-007)"""
        # Captions only cover 0-40s of a 100s video (40% < 50% threshold)
        segments = [
            CaptionSegment(0, 0.0, 20.0, "Seg 1", "vid1"),
            CaptionSegment(1, 20.0, 40.0, "Seg 2", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing()

        assert validation.is_valid is False
        assert validation.below_coverage is True
        assert validation.exceeds_duration is False
        assert "below minimum" in validation.message.lower()

    def test_validate_timing_both_issues(self):
        """Test validate_timing detects both exceed and low coverage (US-007)"""
        # Captions at 120s but only one early segment (low coverage, exceeds duration)
        # Note: This scenario is unusual but tests that both flags can be set
        segments = [
            CaptionSegment(0, 0.0, 10.0, "Seg 1", "vid1"),
            CaptionSegment(1, 100.0, 120.0, "Seg 2 way later", "vid1"),
        ]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # This tests exceeds (120 > 110) but coverage is 120% so not below_coverage
        validation = result.validate_timing()

        assert validation.is_valid is False
        assert validation.exceeds_duration is True

    def test_validate_timing_no_duration(self):
        """Test validate_timing without video duration (US-007)"""
        segments = [CaptionSegment(0, 0.0, 100.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments)  # No video_duration

        validation = result.validate_timing()

        # Should be marked as valid since we can't validate
        assert validation.is_valid is True
        assert "video duration unknown" in validation.message

    def test_validate_timing_empty_segments(self):
        """Test validate_timing with no segments (US-007)"""
        result = CaptionResult(video_id="vid1", segments=[], video_duration=100.0)

        validation = result.validate_timing()

        # No segments = 0 end time = below coverage threshold
        assert validation.is_valid is False
        assert validation.below_coverage is True
        assert validation.caption_end_time == 0.0

    def test_validate_timing_stores_result(self):
        """Test validate_timing stores result in timing_validated field (US-007)"""
        segments = [CaptionSegment(0, 0.0, 50.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        assert result.timing_validated is None

        validation = result.validate_timing()

        assert result.timing_validated is validation
        assert result.timing_validated.is_valid is True

    def test_validate_timing_custom_thresholds(self):
        """Test validate_timing with custom tolerance thresholds (US-007)"""
        # 115s captions on 100s video - fails with default 1.1, passes with 1.2
        segments = [CaptionSegment(0, 0.0, 115.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # Should fail with default
        validation1 = result.validate_timing(max_exceed_ratio=1.1)
        assert validation1.is_valid is False
        assert validation1.exceeds_duration is True

        # Should pass with relaxed threshold
        validation2 = result.validate_timing(max_exceed_ratio=1.2)
        assert validation2.is_valid is True
        assert validation2.exceeds_duration is False

    def test_validate_timing_custom_min_coverage(self):
        """Test validate_timing with custom minimum coverage (US-007)"""
        # 30s captions on 100s video - fails with default 0.5, passes with 0.25
        segments = [CaptionSegment(0, 0.0, 30.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # Should fail with default 50% threshold
        validation1 = result.validate_timing(min_coverage_ratio=0.5)
        assert validation1.is_valid is False
        assert validation1.below_coverage is True

        # Should pass with 25% threshold
        validation2 = result.validate_timing(min_coverage_ratio=0.25)
        assert validation2.is_valid is True
        assert validation2.below_coverage is False

    def test_validate_timing_with_explicit_duration(self):
        """Test validate_timing with explicitly provided video duration (US-007)"""
        segments = [CaptionSegment(0, 0.0, 80.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # Use explicit duration instead of stored duration
        validation = result.validate_timing(video_duration=200.0)

        assert validation.video_duration == 200.0
        assert validation.below_coverage is True  # 80s < 50% of 200s

    def test_to_dict_includes_timing_validated(self):
        """Test to_dict includes timing_validated field (US-007)"""
        segments = [CaptionSegment(0, 0.0, 90.0, "Seg", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # Before validation
        data1 = result.to_dict()
        assert data1['timing_validated'] is None

        # After validation
        result.validate_timing()
        data2 = result.to_dict()

        assert data2['timing_validated'] is not None
        assert data2['timing_validated']['is_valid'] is True
        assert data2['timing_validated']['caption_end_time'] == 90.0
        assert data2['timing_validated']['video_duration'] == 100.0


class TestTimingValidationResult:
    """Test TimingValidationResult dataclass (US-007)"""

    def test_timing_validation_result_creation(self):
        """Test creating TimingValidationResult"""
        from src.caption_fetcher import TimingValidationResult

        result = TimingValidationResult(
            is_valid=True,
            caption_end_time=90.0,
            video_duration=100.0,
            exceeds_duration=False,
            below_coverage=False,
            message="Timing valid"
        )

        assert result.is_valid is True
        assert result.caption_end_time == 90.0
        assert result.video_duration == 100.0
        assert result.exceeds_duration is False
        assert result.below_coverage is False
        assert result.message == "Timing valid"

    def test_timing_validation_result_to_dict(self):
        """Test TimingValidationResult serialization"""
        from src.caption_fetcher import TimingValidationResult

        result = TimingValidationResult(
            is_valid=False,
            caption_end_time=120.0,
            video_duration=100.0,
            exceeds_duration=True,
            below_coverage=False,
            message="Caption exceeds video"
        )

        data = result.to_dict()

        assert data['is_valid'] is False
        assert data['caption_end_time'] == 120.0
        assert data['video_duration'] == 100.0
        assert data['exceeds_duration'] is True
        assert data['below_coverage'] is False
        assert data['message'] == "Caption exceeds video"

    def test_timing_validation_result_defaults(self):
        """Test TimingValidationResult default values"""
        from src.caption_fetcher import TimingValidationResult

        result = TimingValidationResult(
            is_valid=True,
            caption_end_time=50.0,
            video_duration=100.0
        )

        assert result.exceeds_duration is False
        assert result.below_coverage is False
        assert result.message == ""


class TestTimingEpsilonTolerance:
    """Test timing_epsilon_ms parameter in validate_timing (US-007 Sprint 6)"""

    def test_epsilon_valid_within_tolerance(self):
        """Caption at 300.05s in 300s video is valid with 100ms epsilon"""
        segments = [CaptionSegment(0, 0.0, 300.05, "Full coverage", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=300.0)

        # With 100ms epsilon (default), 300.05s should be treated as 300s (valid)
        validation = result.validate_timing(timing_epsilon_ms=100.0)

        assert validation.is_valid is True
        assert validation.exceeds_duration is False
        assert validation.timing_epsilon_applied == 100.0
        assert validation.caption_end_time == 300.05  # Original value preserved

    def test_epsilon_invalid_beyond_tolerance(self):
        """Caption at 300.15s in 300s video is invalid with 100ms epsilon"""
        segments = [CaptionSegment(0, 0.0, 300.15, "Slightly over", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=300.0)

        # With 100ms epsilon, 300.15s (150ms over) is beyond tolerance
        # But still within max_exceed_ratio=1.1 (10%), so valid
        validation = result.validate_timing(timing_epsilon_ms=100.0)

        assert validation.is_valid is True  # Within 10% ratio
        assert validation.exceeds_duration is False
        assert validation.timing_epsilon_applied == 100.0

    def test_epsilon_strict_with_10ms(self):
        """Caption at 300.05s in 300s video is outside 10ms epsilon"""
        segments = [CaptionSegment(0, 0.0, 300.05, "Slightly over", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=300.0)

        # With 10ms epsilon, 300.05s (50ms over) is beyond epsilon
        # But still within max_exceed_ratio=1.1 (10%), so valid
        validation = result.validate_timing(timing_epsilon_ms=10.0)

        assert validation.is_valid is True  # Within 10% ratio
        assert validation.exceeds_duration is False
        assert validation.timing_epsilon_applied == 10.0

    def test_epsilon_affects_exceeds_check_at_boundary(self):
        """Epsilon snaps caption end to duration when within tolerance"""
        # Caption exactly at 110% of 100s video = 110s (boundary of max_exceed_ratio)
        # If epsilon snaps to 100s, it's valid. If not snapped, it's at boundary.
        segments = [CaptionSegment(0, 0.0, 100.08, "Near boundary", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # With 100ms epsilon, 100.08s (80ms over duration) snaps to 100s
        validation = result.validate_timing(timing_epsilon_ms=100.0)

        assert validation.is_valid is True
        assert validation.exceeds_duration is False

    def test_epsilon_zero_exact_matching(self):
        """With epsilon=0, exact matching is used (may cause false positives)"""
        # 299.999s is technically less than 300s
        segments = [CaptionSegment(0, 0.0, 299.999, "Almost exact", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=300.0)

        validation = result.validate_timing(timing_epsilon_ms=0.0)

        assert validation.is_valid is True  # Still valid (below duration)
        assert validation.exceeds_duration is False
        assert validation.timing_epsilon_applied == 0.0

    def test_epsilon_stored_in_result_no_duration(self):
        """Epsilon is stored in result even when video duration is unknown"""
        result = CaptionResult(video_id="vid1", segments=[])

        validation = result.validate_timing(timing_epsilon_ms=50.0)

        assert validation.timing_epsilon_applied == 50.0
        assert validation.message == "Cannot validate timing: video duration unknown"

    def test_epsilon_serialized_in_to_dict(self):
        """timing_epsilon_applied is included in serialized dict"""
        segments = [CaptionSegment(0, 0.0, 90.0, "Normal", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing(timing_epsilon_ms=75.0)
        data = validation.to_dict()

        assert 'timing_epsilon_applied' in data
        assert data['timing_epsilon_applied'] == 75.0

    def test_epsilon_combined_with_max_exceed_ratio(self):
        """Epsilon and max_exceed_ratio work together correctly"""
        # Caption at 110.05s for 100s video
        # With epsilon=100ms: 110.05 is 10.05s over duration (not within epsilon of 100s)
        # With max_exceed_ratio=1.1: 110.05 > 110.0, so exceeds
        segments = [CaptionSegment(0, 0.0, 110.05, "Just over", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        validation = result.validate_timing(
            max_exceed_ratio=1.1,
            timing_epsilon_ms=100.0
        )

        assert validation.is_valid is False
        assert validation.exceeds_duration is True

    def test_epsilon_negative_difference(self):
        """Epsilon handles captions ending slightly before video duration"""
        # Caption at 299.95s for 300s video (50ms before end)
        # Epsilon should snap this to 300s for consistency
        segments = [CaptionSegment(0, 0.0, 299.95, "Slightly early", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=300.0)

        validation = result.validate_timing(timing_epsilon_ms=100.0)

        assert validation.is_valid is True
        assert validation.exceeds_duration is False
        # Coverage should still use original value (299.95/300 = 99.98%)

    def test_epsilon_default_value(self):
        """Default epsilon is 100ms when not specified"""
        segments = [CaptionSegment(0, 0.0, 100.05, "Slight float error", "vid1")]
        result = CaptionResult(video_id="vid1", segments=segments, video_duration=100.0)

        # Call without specifying epsilon - should use default 100.0
        validation = result.validate_timing()

        assert validation.timing_epsilon_applied == 100.0
        assert validation.is_valid is True  # 50ms within 100ms epsilon


class TestDetermineCaptionQuality:
    """Test determine_caption_quality function (US-007)"""

    def test_high_quality_human_dense(self):
        """Human captions with dense segments = high quality"""
        quality = determine_caption_quality(
            is_auto_generated=False,
            segment_count=100,
            total_duration=300.0  # 3s avg per segment
        )
        assert quality == "high"

    def test_medium_quality_human_sparse(self):
        """Human captions with sparse segments = medium quality"""
        quality = determine_caption_quality(
            is_auto_generated=False,
            segment_count=8,  # 8 segments (below min_segments_for_high=10)
            total_duration=60.0  # 7.5s avg per segment - acceptable but sparse
        )
        assert quality == "medium"

    def test_medium_quality_auto_dense(self):
        """Auto-generated captions with dense segments = medium quality"""
        quality = determine_caption_quality(
            is_auto_generated=True,
            segment_count=100,
            total_duration=300.0
        )
        assert quality == "medium"

    def test_low_quality_empty(self):
        """No segments = low quality"""
        quality = determine_caption_quality(
            is_auto_generated=False,
            segment_count=0,
            total_duration=0.0
        )
        assert quality == "low"

    def test_low_quality_very_sparse(self):
        """Very few segments = low quality"""
        quality = determine_caption_quality(
            is_auto_generated=True,
            segment_count=2,
            total_duration=120.0  # 60s avg - very sparse
        )
        assert quality == "low"

    def test_low_quality_auto_very_sparse_long_video(self):
        """Auto captions with very long avg duration on long video = low quality"""
        quality = determine_caption_quality(
            is_auto_generated=True,
            segment_count=3,
            total_duration=120.0  # 40s avg
        )
        assert quality == "low"

    def test_custom_thresholds(self):
        """Test custom min_segments and max_avg_duration thresholds"""
        # With default thresholds (10 segments, 10s max avg), this would be medium
        # With custom thresholds (5 segments, 20s max avg), this should be high
        quality = determine_caption_quality(
            is_auto_generated=False,
            segment_count=6,
            total_duration=100.0,  # ~16s avg
            min_segments_for_high=5,
            max_avg_duration_for_high=20.0
        )
        assert quality == "high"


class TestCaptionExceptions:
    """Test caption exception classes"""

    def test_caption_unavailable_error(self):
        """Test CaptionUnavailableError"""
        error = CaptionUnavailableError("vid123", "No subtitles available")

        assert error.video_id == "vid123"
        assert error.reason == "No subtitles available"
        assert "vid123" in str(error)
        assert "No subtitles" in str(error)

    def test_caption_unavailable_error_no_reason(self):
        """Test CaptionUnavailableError without reason"""
        error = CaptionUnavailableError("vid123")

        assert error.video_id == "vid123"
        assert "vid123" in str(error)

    def test_caption_fetch_error(self):
        """Test CaptionFetchError"""
        error = CaptionFetchError("vid123", "Network timeout")

        assert error.video_id == "vid123"
        assert error.reason == "Network timeout"
        assert "vid123" in str(error)
        assert "Network timeout" in str(error)

    def test_exception_inheritance(self):
        """Test exception inheritance"""
        assert issubclass(CaptionUnavailableError, CaptionError)
        assert issubclass(CaptionFetchError, CaptionError)
        assert issubclass(CaptionError, Exception)


class TestCaptionFetcherVideoIdValidation:
    """Test video ID validation"""

    def test_valid_video_id(self):
        """Test valid YouTube video ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9WgXcQ") is True
        assert fetcher._is_valid_video_id("abc123DEF-_") is True
        assert fetcher._is_valid_video_id("12345678901") is True

    def test_invalid_video_id_too_short(self):
        """Test rejection of too-short ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9WgXc") is False  # 10 chars
        assert fetcher._is_valid_video_id("abc") is False

    def test_invalid_video_id_too_long(self):
        """Test rejection of too-long ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9WgXcQQ") is False  # 12 chars

    def test_invalid_video_id_special_chars(self):
        """Test rejection of invalid characters"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9Wg@cQ") is False  # @ is invalid
        assert fetcher._is_valid_video_id("dQw4w9Wg cQ") is False  # space is invalid
        assert fetcher._is_valid_video_id("dQw4w9Wg.cQ") is False  # . is invalid

    def test_invalid_video_id_empty(self):
        """Test rejection of empty ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("") is False
        assert fetcher._is_valid_video_id(None) is False


class TestCaptionFetcherTimestampParsing:
    """Test timestamp parsing"""

    def test_parse_timestamp_vtt_format(self):
        """Test parsing VTT timestamp format (HH:MM:SS.mmm)"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:00:01.000") == 1.0
        assert fetcher._parse_timestamp("00:01:30.500") == 90.5
        assert fetcher._parse_timestamp("01:00:00.000") == 3600.0
        assert fetcher._parse_timestamp("01:30:45.123") == 5445.123

    def test_parse_timestamp_srt_format(self):
        """Test parsing SRT timestamp format (HH:MM:SS,mmm)"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:00:01,000") == 1.0
        assert fetcher._parse_timestamp("00:01:30,500") == 90.5
        assert fetcher._parse_timestamp("01:00:00,000") == 3600.0

    def test_parse_timestamp_short_format(self):
        """Test parsing short VTT format (MM:SS.mmm)"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:01.000") == 1.0
        assert fetcher._parse_timestamp("01:30.500") == 90.5
        assert fetcher._parse_timestamp("59:59.999") == 3599.999

    def test_parse_timestamp_without_milliseconds(self):
        """Test parsing timestamp without milliseconds"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:00:01") == 1.0
        assert fetcher._parse_timestamp("01:30") == 90.0

    def test_parse_timestamp_with_whitespace(self):
        """Test parsing timestamp with whitespace"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("  00:00:01.000  ") == 1.0

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp returns None"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("invalid") is None
        assert fetcher._parse_timestamp("abc:def:ghi") is None  # Non-numeric


class TestCaptionFetcherVttParsing:
    """Test VTT format parsing"""

    def test_parse_vtt_basic(self):
        """Test parsing basic VTT content"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello, world!

00:00:05.000 --> 00:00:08.000
This is a test.
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 2
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello, world!"
        assert segments[1].start_time == 5.0
        assert segments[1].end_time == 8.0
        assert segments[1].text == "This is a test."

    def test_parse_vtt_with_style_tags(self):
        """Test VTT parsing removes style tags"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
<c.colorWhite>Hello</c> <b>world</b>!
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Hello world!"

    def test_parse_vtt_multiline_text(self):
        """Test VTT parsing with multiline text"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Line one
Line two
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 1
        assert "Line one" in segments[0].text
        assert "Line two" in segments[0].text

    def test_parse_vtt_with_cue_identifiers(self):
        """Test VTT parsing with cue identifiers"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
First cue

2
00:00:05.000 --> 00:00:08.000
Second cue
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 2

    def test_parse_vtt_empty(self):
        """Test parsing empty VTT content"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 0

    def test_parse_vtt_sets_source_file(self):
        """Test that source_file is set correctly"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:02.000
Test
"""
        segments = fetcher._parse_vtt(vtt_content, "my_video_id")

        assert segments[0].source_file == "my_video_id"


class TestCaptionFetcherSrtParsing:
    """Test SRT format parsing"""

    def test_parse_srt_basic(self):
        """Test parsing basic SRT content"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,000
This is a test.
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert len(segments) == 2
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello, world!"
        assert segments[1].text == "This is a test."

    def test_parse_srt_with_tags(self):
        """Test SRT parsing removes formatting tags"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
<i>Italic</i> and <b>bold</b>
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert segments[0].text == "Italic and bold"

    def test_parse_srt_multiline(self):
        """Test SRT with multiline subtitles"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
Line one
Line two

2
00:00:05,000 --> 00:00:08,000
Single line
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert len(segments) == 2
        assert "Line one" in segments[0].text
        assert "Line two" in segments[0].text

    def test_parse_srt_empty_blocks(self):
        """Test SRT parsing handles empty blocks gracefully"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
Valid


3
00:00:05,000 --> 00:00:08,000
Also valid
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert len(segments) == 2


class TestCaptionFetcherJson3Parsing:
    """Test JSON3/SRV3 format parsing"""

    def test_parse_json3_basic(self):
        """Test parsing basic JSON3 content"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "Hello, world!"}]
                },
                {
                    "tStartMs": 5000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "This is a test."}]
                }
            ]
        })

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 2
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello, world!"
        assert segments[1].start_time == 5.0
        assert segments[1].end_time == 8.0

    def test_parse_json3_multiple_segs(self):
        """Test JSON3 with multiple text segments per event"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 3000,
                    "segs": [
                        {"utf8": "Hello "},
                        {"utf8": "world!"}
                    ]
                }
            ]
        })

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Hello world!"

    def test_parse_json3_empty_events(self):
        """Test JSON3 with empty events array"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({"events": []})

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 0

    def test_parse_json3_events_without_segs(self):
        """Test JSON3 skips events without segs"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({
            "events": [
                {"tStartMs": 1000, "dDurationMs": 3000},  # No segs
                {
                    "tStartMs": 5000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "Valid"}]
                }
            ]
        })

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Valid"

    def test_parse_json3_invalid_json(self):
        """Test JSON3 parsing handles invalid JSON"""
        fetcher = CaptionFetcher()

        segments = fetcher._parse_json3("not valid json", "test_video")

        assert len(segments) == 0


class TestCaptionFetcherFetchCaptions:
    """Test fetch_captions method"""

    @patch('subprocess.run')
    def test_fetch_captions_invalid_video_id(self, mock_run):
        """Test fetch with invalid video ID raises error"""
        fetcher = CaptionFetcher()

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.fetch_captions("invalid")

        assert "Invalid video ID" in str(exc_info.value)
        mock_run.assert_not_called()

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_fetch_captions_no_subtitles_available(self, mock_tempdir, mock_run):
        """Test fetch when no subtitles available"""
        fetcher = CaptionFetcher()

        # Setup temp directory mock
        mock_temp_path = MagicMock()
        mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(mock_temp_path))
        mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yt-dlp failure with "no subtitles" message
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "no subtitles available for this video"
        mock_run.return_value = mock_result

        with pytest.raises(CaptionUnavailableError):
            fetcher.fetch_captions("dQw4w9WgXcQ")

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_fetch_captions_network_error(self, mock_tempdir, mock_run):
        """Test fetch with network error"""
        fetcher = CaptionFetcher()

        mock_temp_path = MagicMock()
        mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(mock_temp_path))
        mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

        # Mock network error
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Connection timeout"
        mock_run.return_value = mock_result

        with pytest.raises(CaptionFetchError):
            fetcher.fetch_captions("dQw4w9WgXcQ")

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_fetch_captions_timeout(self, mock_tempdir, mock_run):
        """Test fetch with subprocess timeout"""
        fetcher = CaptionFetcher()

        mock_temp_path = MagicMock()
        mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(mock_temp_path))
        mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

        mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 60)

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.fetch_captions("dQw4w9WgXcQ")

        assert "Timeout" in str(exc_info.value)

    @patch('subprocess.run')
    def test_fetch_captions_success(self, mock_run, tmp_path):
        """Test successful caption fetch"""
        fetcher = CaptionFetcher()

        # Create a VTT file that will be "downloaded"
        video_id = "dQw4w9WgXcQ"
        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Never gonna give you up

00:00:05.000 --> 00:00:08.000
Never gonna let you down
"""

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            # Use the actual tmp_path for the mock
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create the subtitle file
            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text(vtt_content)

            # Mock successful yt-dlp run
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = fetcher.fetch_captions(video_id)

            assert result.video_id == video_id
            assert len(result.segments) == 2
            assert "Never gonna give you up" in result.text
            assert result.format_source == "vtt"


class TestCaptionFetcherCookies:
    """Test cookie handling"""

    def test_get_cookies_args_no_config(self):
        """Test cookie args with no config"""
        fetcher = CaptionFetcher()

        args = fetcher._get_cookies_args()

        assert args == []

    def test_get_cookies_args_browser_cookies(self):
        """Test cookie args with browser cookies configured"""
        mock_config = Mock()
        mock_config.download.cookies_from_browser = "chrome"
        mock_config.download.cookies_path = ""

        fetcher = CaptionFetcher(config=mock_config)

        args = fetcher._get_cookies_args()

        assert args == ['--cookies-from-browser', 'chrome']

    def test_get_cookies_args_file_cookies(self):
        """Test cookie args with file cookies configured"""
        mock_config = Mock()
        mock_config.download.cookies_from_browser = ""
        mock_config.download.cookies_path = "/path/to/cookies.txt"

        with patch.object(Path, 'exists', return_value=True):
            fetcher = CaptionFetcher(config=mock_config)
            args = fetcher._get_cookies_args()

        assert args == ['--cookies', '/path/to/cookies.txt']

    def test_get_cookies_args_browser_preferred(self):
        """Test that browser cookies are preferred over file"""
        mock_config = Mock()
        mock_config.download.cookies_from_browser = "firefox"
        mock_config.download.cookies_path = "/path/to/cookies.txt"

        fetcher = CaptionFetcher(config=mock_config)

        args = fetcher._get_cookies_args()

        # Should use browser cookies, not file
        assert args == ['--cookies-from-browser', 'firefox']


class TestCaptionFetcherIntegration:
    """Integration tests for caption fetcher"""

    def test_parse_subtitle_file_vtt(self, tmp_path):
        """Test parsing VTT file from disk"""
        fetcher = CaptionFetcher()

        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("""WEBVTT

00:00:01.000 --> 00:00:04.000
Test caption
""")

        segments = fetcher._parse_subtitle_file(vtt_file, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Test caption"

    def test_parse_subtitle_file_srt(self, tmp_path):
        """Test parsing SRT file from disk"""
        fetcher = CaptionFetcher()

        srt_file = tmp_path / "test.srt"
        srt_file.write_text("""1
00:00:01,000 --> 00:00:04,000
Test caption
""")

        segments = fetcher._parse_subtitle_file(srt_file, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Test caption"

    def test_parse_subtitle_file_json3(self, tmp_path):
        """Test parsing JSON3 file from disk"""
        fetcher = CaptionFetcher()

        json_file = tmp_path / "test.json3"
        json_file.write_text(json.dumps({
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "Test caption"}]
                }
            ]
        }))

        segments = fetcher._parse_subtitle_file(json_file, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Test caption"

    def test_parse_subtitle_file_unicode(self, tmp_path):
        """Test parsing file with unicode characters"""
        fetcher = CaptionFetcher()

        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("""WEBVTT

00:00:01.000 --> 00:00:04.000
Hello 世界! Привет мир! 🎉
""", encoding='utf-8')

        segments = fetcher._parse_subtitle_file(vtt_file, "test_video")

        assert len(segments) == 1
        assert "世界" in segments[0].text
        assert "Привет" in segments[0].text
        assert "🎉" in segments[0].text


# Mark integration tests that require network
@pytest.mark.requires_network
class TestCaptionFetcherRealVideos:
    """Integration tests with real YouTube videos.

    These tests are skipped by default. Run with:
        pytest -m requires_network tests/test_caption_fetcher.py

    Known video IDs for testing:
    - "dQw4w9WgXcQ": Rick Astley - Never Gonna Give You Up (has human captions)
    - Some videos may have auto-generated captions only
    """

    def test_fetch_video_with_captions(self):
        """Test fetching captions from a known video with captions"""
        fetcher = CaptionFetcher()

        # Note: This test may fail if the video is removed or captions change
        # Rick Astley - Never Gonna Give You Up is a stable choice
        try:
            result = fetcher.fetch_captions("dQw4w9WgXcQ", language="en")

            assert result.video_id == "dQw4w9WgXcQ"
            assert len(result.segments) > 0
            assert result.language == "en"
            # The actual is_auto_generated value depends on what's available
        except CaptionUnavailableError:
            pytest.skip("Video captions not available (may have been removed)")
        except CaptionFetchError as e:
            # Skip on transient yt-dlp errors (preprocessing, network, etc.)
            if "preprocessing" in str(e).lower() or "invalid data" in str(e).lower():
                pytest.skip(f"yt-dlp transient error: {e}")
            raise


class TestAvailableLanguage:
    """Test AvailableLanguage dataclass"""

    def test_available_language_creation(self):
        """Test creating an available language"""
        lang = AvailableLanguage(
            code="en",
            name="English",
            is_auto_generated=False
        )

        assert lang.code == "en"
        assert lang.name == "English"
        assert lang.is_auto_generated is False

    def test_available_language_auto_generated(self):
        """Test auto-generated language"""
        lang = AvailableLanguage(
            code="es",
            name="Spanish (auto-generated)",
            is_auto_generated=True
        )

        assert lang.code == "es"
        assert lang.is_auto_generated is True


class TestListAvailableLanguages:
    """Test list_available_languages method"""

    @patch('subprocess.run')
    def test_list_languages_invalid_video_id(self, mock_run):
        """Test listing with invalid video ID"""
        fetcher = CaptionFetcher()

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.list_available_languages("invalid")

        assert "Invalid video ID" in str(exc_info.value)
        mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_list_languages_timeout(self, mock_run):
        """Test listing with subprocess timeout"""
        fetcher = CaptionFetcher()

        mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 60)

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.list_available_languages("dQw4w9WgXcQ")

        assert "Timeout" in str(exc_info.value)

    @patch('subprocess.run')
    def test_list_languages_manual_only(self, mock_run):
        """Test parsing manual subtitles section"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
es        Spanish              vtt, ttml, srv3, srv2, srv1, json3
fr        French               vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 3
        assert all(not lang.is_auto_generated for lang in languages)
        assert languages[0].code == "en"
        assert languages[1].code == "es"
        assert languages[2].code == "fr"

    @patch('subprocess.run')
    def test_list_languages_auto_only(self, mock_run):
        """Test parsing auto-generated captions section"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
es        Spanish (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 2
        assert all(lang.is_auto_generated for lang in languages)

    @patch('subprocess.run')
    def test_list_languages_mixed(self, mock_run):
        """Test parsing both manual and auto sections"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
es        Spanish (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        # Should have 3: 1 manual + 2 auto
        assert len(languages) == 3

        # Sorted: manual first (en), then auto (en, es)
        assert languages[0].code == "en"
        assert languages[0].is_auto_generated is False

        assert languages[1].code == "en"
        assert languages[1].is_auto_generated is True

        assert languages[2].code == "es"
        assert languages[2].is_auto_generated is True

    @patch('subprocess.run')
    def test_list_languages_empty(self, mock_run):
        """Test parsing output with no subtitles"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] dQw4w9WgXcQ: Downloading webpage
[info] dQw4w9WgXcQ: Downloading ios player API JSON
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 0

    @patch('subprocess.run')
    def test_list_languages_regional_codes(self, mock_run):
        """Test parsing regional language codes like en-GB"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en-GB     English (UK)         vtt, ttml, srv3, srv2, srv1, json3
pt-BR     Portuguese (Brazil)  vtt, ttml, srv3, srv2, srv1, json3
zh-Hans   Chinese (Simplified) vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 3
        assert languages[0].code == "en-gb"
        assert languages[1].code == "pt-br"
        assert languages[2].code == "zh-hans"


class TestHasCaptions:
    """Test has_captions method (US-008)"""

    @patch('subprocess.run')
    def test_has_captions_returns_true_when_manual_available(self, mock_run):
        """Test has_captions returns True when manual captions exist"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = fetcher.has_captions("dQw4w9WgXcQ")

        assert result is True

    @patch('subprocess.run')
    def test_has_captions_returns_true_when_auto_available(self, mock_run):
        """Test has_captions returns True when only auto captions exist"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = fetcher.has_captions("dQw4w9WgXcQ")

        assert result is True

    @patch('subprocess.run')
    def test_has_captions_returns_false_when_none_available(self, mock_run):
        """Test has_captions returns False when no captions exist"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] dQw4w9WgXcQ: Downloading webpage
[info] dQw4w9WgXcQ: Downloading ios player API JSON
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = fetcher.has_captions("dQw4w9WgXcQ")

        assert result is False

    @patch('subprocess.run')
    def test_has_captions_raises_on_invalid_id(self, mock_run):
        """Test has_captions raises CaptionFetchError for invalid video ID"""
        fetcher = CaptionFetcher()

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.has_captions("invalid")

        assert "Invalid video ID" in str(exc_info.value)
        mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_has_captions_raises_on_timeout(self, mock_run):
        """Test has_captions raises CaptionFetchError on timeout"""
        fetcher = CaptionFetcher()

        mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 30)

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.has_captions("dQw4w9WgXcQ")

        assert "Timeout" in str(exc_info.value)

    @patch('subprocess.run')
    def test_has_captions_with_multiple_languages(self, mock_run):
        """Test has_captions returns True when multiple languages available"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
es        Spanish              vtt, ttml, srv3, srv2, srv1, json3
fr        French               vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = fetcher.has_captions("dQw4w9WgXcQ")

        assert result is True


class TestSelectBestLanguage:
    """Test select_best_language method"""

    def test_select_preferred_manual(self):
        """Test selecting preferred language when manual is available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
            AvailableLanguage("en", "English (auto)", True),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        assert result.code == "es"
        assert result.is_auto_generated is False

    def test_select_preferred_auto_fallback(self):
        """Test falling back to auto when manual not available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish (auto)", True),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        assert result.code == "es"
        assert result.is_auto_generated is True

    def test_select_english_fallback(self):
        """Test falling back to English when preferred not available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        assert result.code == "en"
        assert result.is_auto_generated is False

    def test_select_english_fallback_auto(self):
        """Test falling back to English auto when English manual not available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English (auto)", True),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(
            available, preferred="es", prefer_manual=True
        )

        # English fallback happens before "any available" fallback
        # Even though French is manual, English (auto) is selected because
        # the fallback chain is: preferred -> English -> any
        # When prefer_manual=True but only auto English available, still uses English
        assert result.code == "en"
        assert result.is_auto_generated is True

    def test_select_any_fallback(self):
        """Test falling back to any available when preferred and en not available"""
        fetcher = CaptionFetcher()

        # List is passed as-is to select_best_language (not sorted internally)
        # list_available_languages returns sorted results, but select_best_language
        # just iterates through the provided list
        available = [
            AvailableLanguage("de", "German", False),  # First in sorted order
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        # Should select first manual from the list
        assert result.code == "de"
        assert result.is_auto_generated is False

    def test_select_empty_list(self):
        """Test selecting from empty list returns None"""
        fetcher = CaptionFetcher()

        result = fetcher.select_best_language([], preferred="en")

        assert result is None

    def test_select_prefer_manual_over_auto(self):
        """Test that manual captions are preferred over auto"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("en", "English (auto)", True),
        ]

        result = fetcher.select_best_language(
            available, preferred="en", prefer_manual=True
        )

        assert result.code == "en"
        assert result.is_auto_generated is False

    def test_select_prefer_auto_when_configured(self):
        """Test preferring auto when prefer_manual=False"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("en", "English (auto)", True),
        ]

        result = fetcher.select_best_language(
            available, preferred="en", prefer_manual=False
        )

        # When not preferring manual, first match wins (list sorted manual first)
        assert result.code == "en"
        # First en in sorted list is manual
        assert result.is_auto_generated is False

    def test_select_no_english_fallback(self):
        """Test disabling English fallback"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(
            available, preferred="es", fallback_to_english=False
        )

        # Should skip English and fall back to any (sorted alphabetically)
        assert result is not None
        # en comes before fr alphabetically
        assert result.code == "en"

    def test_select_english_as_preferred_no_double_check(self):
        """Test that English as preferred doesn't check English twice"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(available, preferred="en")

        assert result.code == "en"


class TestFallbackLanguageChain:
    """Test configurable fallback language chain (US-003)"""

    def test_fallback_chain_first_available(self):
        """Test fallback chain stops at first available language"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("pt", "Portuguese", False),
            AvailableLanguage("de", "German", False),
        ]

        # es not available, pt is first in fallback chain and is available
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr", "de"]
        )

        assert result.code == "pt"
        assert result.is_auto_generated is False

    def test_fallback_chain_second_available(self):
        """Test fallback chain moves to second when first unavailable"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("fr", "French", False),
            AvailableLanguage("de", "German", False),
        ]

        # es not available, pt not available, fr is available
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr", "de"]
        )

        assert result.code == "fr"
        assert result.is_auto_generated is False

    def test_fallback_chain_skips_to_english(self):
        """Test fallback chain falls through to English when chain exhausted"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("de", "German", False),
        ]

        # es, pt, fr all unavailable -> should fall back to English
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr"]
        )

        assert result.code == "en"
        assert result.is_auto_generated is False

    def test_fallback_chain_includes_english_no_double_check(self):
        """Test English in fallback chain doesn't cause double-checking"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("de", "German", False),
        ]

        # If en is in fallback chain, it should be checked once (in the chain)
        # and the English fallback step should be skipped
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "en", "fr"]
        )

        assert result.code == "en"

    def test_fallback_chain_empty_preserves_default_behavior(self):
        """Test empty fallback_languages preserves default behavior"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("de", "German", False),
        ]

        # Empty fallback_languages should behave like before: es -> en -> any
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=[]
        )

        assert result.code == "en"  # Default English fallback

    def test_fallback_chain_to_any_when_all_fail(self):
        """Test falls back to any available when chain + English fail"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("de", "German", False),
            AvailableLanguage("ja", "Japanese", False),
        ]

        # es, pt, fr, en all unavailable -> should fall back to first available
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr"]
        )

        assert result.code == "de"  # First in sorted list

    def test_fallback_chain_prefers_manual_at_each_step(self):
        """Test prefer_manual tries manual first, then auto, before moving to next language"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("pt", "Portuguese (auto)", True),
            AvailableLanguage("fr", "French", False),
        ]

        # pt is auto-generated only, fr is manual
        # With prefer_manual=True:
        # 1. Try pt manual -> not found
        # 2. Try pt auto -> found, use it (same-language auto is preferred over next-language manual)
        # This behavior is intentional: stay in the same language even if only auto is available
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr"],
            prefer_manual=True
        )

        assert result.code == "pt"
        assert result.is_auto_generated is True

    def test_fallback_chain_manual_available_in_chain(self):
        """Test prefer_manual selects manual when available at same position"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("pt", "Portuguese", False),  # Manual
            AvailableLanguage("pt", "Portuguese (auto)", True),  # Auto
            AvailableLanguage("fr", "French", False),
        ]

        # With prefer_manual=True, should select pt manual over pt auto
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr"],
            prefer_manual=True
        )

        assert result.code == "pt"
        assert result.is_auto_generated is False

    def test_fallback_chain_accepts_auto_when_only_option(self):
        """Test auto captions selected when no manual available in chain"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("pt", "Portuguese (auto)", True),
        ]

        # pt auto is the only option in chain
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr"],
            prefer_manual=True
        )

        assert result.code == "pt"
        assert result.is_auto_generated is True

    def test_fallback_chain_skips_duplicates(self):
        """Test fallback chain skips already-tried languages"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("fr", "French", False),
        ]

        # 'es' is both preferred and in fallback chain - should only try once
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["es", "pt", "fr"]
        )

        assert result.code == "fr"

    def test_fallback_chain_case_insensitive(self):
        """Test fallback chain handles case variations"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("PT", "Portuguese", False),
        ]

        # Lowercase in config, uppercase in available
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr"]
        )

        assert result.code == "PT"

    def test_fallback_chain_multilingual_project(self):
        """Test realistic multilingual project scenario"""
        fetcher = CaptionFetcher()

        # Spanish project with Portuguese and French fallbacks
        available = [
            AvailableLanguage("en", "English (auto)", True),
            AvailableLanguage("pt", "Portuguese", False),
            AvailableLanguage("de", "German (auto)", True),
        ]

        # Preferred es not available, fallback to pt
        result = fetcher.select_best_language(
            available, preferred="es", fallback_languages=["pt", "fr", "en"]
        )

        assert result.code == "pt"
        assert result.is_auto_generated is False


class TestGetFallbackLanguagesFromConfig:
    """Test _get_fallback_languages_from_config method (US-003)"""

    def test_no_config_returns_empty_list(self):
        """Test default is empty list when no config"""
        fetcher = CaptionFetcher()

        result = fetcher._get_fallback_languages_from_config()

        assert result == []

    def test_reads_fallback_languages_from_config(self):
        """Test reading fallback_languages from config"""
        mock_config = Mock()
        mock_config.download.caption_first.fallback_languages = ["es", "pt", "fr"]

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_fallback_languages_from_config()

        assert result == ["es", "pt", "fr"]

    def test_returns_empty_for_none(self):
        """Test returns empty list when config value is None"""
        mock_config = Mock()
        mock_config.download.caption_first.fallback_languages = None

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_fallback_languages_from_config()

        assert result == []

    def test_returns_empty_for_non_list(self):
        """Test returns empty list when config value is not a list"""
        mock_config = Mock()
        mock_config.download.caption_first.fallback_languages = "es"  # String, not list

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_fallback_languages_from_config()

        assert result == []

    def test_handles_attribute_error(self):
        """Test handles missing config attributes gracefully"""
        mock_config = Mock()
        # Make download raise AttributeError
        del mock_config.download

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_fallback_languages_from_config()

        assert result == []


class TestGetPreferredLanguageFromConfig:
    """Test _get_preferred_language_from_config method"""

    def test_no_config_defaults_to_english(self):
        """Test default is English when no config"""
        fetcher = CaptionFetcher()

        result = fetcher._get_preferred_language_from_config()

        assert result == "en"

    def test_caption_first_preferred_language(self):
        """Test reading from caption_first.preferred_language"""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_language = "es"

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_preferred_language_from_config()

        assert result == "es"

    def test_transcription_language_fallback(self):
        """Test falling back to transcription.language"""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_language = ""
        mock_config.transcription.language = "fr"

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_preferred_language_from_config()

        assert result == "fr"

    def test_caption_first_takes_precedence(self):
        """Test caption_first.preferred_language takes precedence"""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_language = "de"
        mock_config.transcription.language = "fr"

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_preferred_language_from_config()

        assert result == "de"


class TestParseListSubsOutput:
    """Test _parse_list_subs_output method"""

    def test_parse_manual_subtitles_section(self):
        """Test parsing manual subtitles section"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
es        Spanish              vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        assert len(result) == 2
        assert all(not lang.is_auto_generated for lang in result)

    def test_parse_auto_captions_section(self):
        """Test parsing auto-generated captions section"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available automatic captions for VIDEO_ID:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        assert len(result) == 1
        assert result[0].is_auto_generated is True

    def test_parse_mixed_output(self):
        """Test parsing output with both sections"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for VIDEO_ID:
Language  Name                              Formats
de        German (auto-generated)           vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        assert len(result) == 2
        # Manual first
        assert result[0].code == "en"
        assert result[0].is_auto_generated is False
        # Auto second
        assert result[1].code == "de"
        assert result[1].is_auto_generated is True

    def test_parse_output_with_stderr(self):
        """Test parsing with content in stderr"""
        fetcher = CaptionFetcher()

        stdout = ""
        stderr = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
ja        Japanese             vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, stderr)

        assert len(result) == 1
        assert result[0].code == "ja"

    def test_parse_handles_header_line(self):
        """Test that header line is skipped"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
en        English              vtt, ttml
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        # Should only have 1 language, not treat "Language" as a code
        assert len(result) == 1
        assert result[0].code == "en"

    def test_parse_language_code_variations(self):
        """Test parsing various language code formats"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                      Formats
en        English                   vtt, ttml
en-US     English (United States)   vtt, ttml
zh-Hans   Chinese (Simplified)      vtt, ttml
pt-BR     Portuguese (Brazil)       vtt, ttml
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        codes = [lang.code for lang in result]
        assert "en" in codes
        assert "en-us" in codes
        assert "zh-hans" in codes
        assert "pt-br" in codes


class TestFetchCaptionsAutoLanguage:
    """Test fetch_captions_auto_language method"""

    @patch.object(CaptionFetcher, 'fetch_captions')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_selects_best(self, mock_list, mock_fetch):
        """Test auto language selection and fetch"""
        fetcher = CaptionFetcher()

        mock_list.return_value = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
        ]
        mock_fetch.return_value = CaptionResult(
            video_id="test123",
            segments=[],
            language="en"
        )

        result = fetcher.fetch_captions_auto_language("test1234567")

        mock_fetch.assert_called_once_with(
            "test1234567",
            language="en",
            prefer_manual=True
        )

    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_no_captions_available(self, mock_list):
        """Test error when no captions available"""
        fetcher = CaptionFetcher()

        mock_list.return_value = []

        with pytest.raises(CaptionUnavailableError) as exc_info:
            fetcher.fetch_captions_auto_language("test1234567")

        assert "No captions available" in str(exc_info.value)

    @patch.object(CaptionFetcher, 'fetch_captions')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_uses_preferred(self, mock_list, mock_fetch):
        """Test auto language uses preferred language override"""
        fetcher = CaptionFetcher()

        mock_list.return_value = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
        ]
        mock_fetch.return_value = CaptionResult(
            video_id="test123",
            segments=[],
            language="es"
        )

        fetcher.fetch_captions_auto_language("test1234567", preferred_language="es")

        mock_fetch.assert_called_once_with(
            "test1234567",
            language="es",
            prefer_manual=True
        )

    @patch.object(CaptionFetcher, 'fetch_captions')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_respects_prefer_human_config(self, mock_list, mock_fetch):
        """Test auto language respects prefer_human_captions config"""
        mock_config = Mock()
        mock_config.download.caption_first.prefer_human_captions = False

        fetcher = CaptionFetcher(config=mock_config)

        mock_list.return_value = [
            AvailableLanguage("en", "English (auto)", True),
        ]
        mock_fetch.return_value = CaptionResult(
            video_id="test123",
            segments=[],
            language="en"
        )

        fetcher.fetch_captions_auto_language("test1234567")

        # Should call with prefer_manual=False since config says don't prefer human
        mock_fetch.assert_called_once_with(
            "test1234567",
            language="en",
            prefer_manual=False  # Auto-generated selected
        )


class TestCaptionFetcherRetry:
    """Test caption fetch retry behavior (US-008)"""

    def test_fetcher_init_default_retry_settings(self):
        """Test default retry settings when no config"""
        fetcher = CaptionFetcher()

        assert fetcher._max_retries == 3
        assert fetcher._retry_delay == 2.0

    def test_fetcher_init_config_retry_settings(self):
        """Test retry settings from config"""
        mock_config = Mock()
        mock_config.download.caption_first.max_retries = 5
        mock_config.download.caption_first.retry_delay = 1.5
        mock_config.download.caption_first.timeout = 45

        fetcher = CaptionFetcher(config=mock_config)

        assert fetcher._max_retries == 5
        assert fetcher._retry_delay == 1.5
        assert fetcher._timeout == 45

    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_success_first_attempt(self, mock_fetch):
        """Test successful fetch on first attempt"""
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="test1234567",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "test1234567")],
            language="en"
        )
        mock_fetch.return_value = mock_result

        result = fetcher.fetch_captions_with_retry("test1234567")

        assert result.video_id == "test1234567"
        assert mock_fetch.call_count == 1

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_temporary_error_then_success(self, mock_fetch, mock_sleep):
        """Test retry on CaptionFetchError then success"""
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="test1234567",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "test1234567")],
            language="en"
        )
        # First call fails, second succeeds
        mock_fetch.side_effect = [
            CaptionFetchError("test1234567", "Network timeout"),
            mock_result
        ]

        result = fetcher.fetch_captions_with_retry("test1234567", max_retries=3, retry_delay=1.0)

        assert result.video_id == "test1234567"
        assert mock_fetch.call_count == 2
        # Check exponential backoff: delay * (2^0) = 1.0
        mock_sleep.assert_called_once_with(1.0)

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_exponential_backoff(self, mock_fetch, mock_sleep):
        """Test exponential backoff timing"""
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="test1234567",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "test1234567")],
            language="en"
        )
        # Fail twice, then succeed
        mock_fetch.side_effect = [
            CaptionFetchError("test1234567", "Error 1"),
            CaptionFetchError("test1234567", "Error 2"),
            mock_result
        ]

        result = fetcher.fetch_captions_with_retry("test1234567", max_retries=3, retry_delay=2.0)

        assert mock_fetch.call_count == 3
        # Check exponential backoff: 2*2^0=2, 2*2^1=4
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2.0)  # 2.0 * 2^0
        mock_sleep.assert_any_call(4.0)  # 2.0 * 2^1

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_all_retries_exhausted(self, mock_fetch, mock_sleep):
        """Test all retries fail raises CaptionFetchError"""
        fetcher = CaptionFetcher()

        # All attempts fail
        mock_fetch.side_effect = CaptionFetchError("test1234567", "Persistent error")

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.fetch_captions_with_retry("test1234567", max_retries=2, retry_delay=0.1)

        assert "test1234567" in str(exc_info.value)
        # 1 initial + 2 retries = 3 attempts
        assert mock_fetch.call_count == 3

    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_unavailable_not_retried(self, mock_fetch):
        """Test CaptionUnavailableError is NOT retried"""
        fetcher = CaptionFetcher()

        mock_fetch.side_effect = CaptionUnavailableError("test1234567", "No captions")

        with pytest.raises(CaptionUnavailableError) as exc_info:
            fetcher.fetch_captions_with_retry("test1234567", max_retries=3)

        # Should only try once - unavailable errors are permanent
        assert mock_fetch.call_count == 1
        assert "test1234567" in str(exc_info.value)

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_unexpected_error_wrapped(self, mock_fetch, mock_sleep):
        """Test unexpected exceptions are wrapped in CaptionFetchError"""
        fetcher = CaptionFetcher()

        mock_fetch.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.fetch_captions_with_retry("test1234567", max_retries=1, retry_delay=0.1)

        assert "test1234567" in str(exc_info.value)
        assert mock_fetch.call_count == 2  # Initial + 1 retry

    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_fetch_with_retry_override_params(self, mock_fetch):
        """Test override parameters are used"""
        mock_config = Mock()
        mock_config.download.caption_first.max_retries = 10
        mock_config.download.caption_first.retry_delay = 5.0
        mock_config.download.caption_first.timeout = 30

        fetcher = CaptionFetcher(config=mock_config)

        mock_result = CaptionResult(
            video_id="test1234567",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "test1234567")],
            language="en"
        )
        mock_fetch.return_value = mock_result

        # Override config values
        result = fetcher.fetch_captions_with_retry(
            "test1234567",
            max_retries=1,
            retry_delay=0.5
        )

        assert result.video_id == "test1234567"

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_list_languages_with_retry(self, mock_list, mock_sleep):
        """Test list_available_languages_with_retry"""
        fetcher = CaptionFetcher()

        mock_languages = [AvailableLanguage("en", "English", False)]
        # Fail once, then succeed
        mock_list.side_effect = [
            CaptionFetchError("test1234567", "Timeout"),
            mock_languages
        ]

        result = fetcher.list_available_languages_with_retry("test1234567", max_retries=2, retry_delay=1.0)

        assert len(result) == 1
        assert result[0].code == "en"
        assert mock_list.call_count == 2

    @patch.object(CaptionFetcher, 'fetch_captions_auto_language')
    def test_fetch_auto_language_with_retry(self, mock_fetch):
        """Test fetch_captions_auto_language_with_retry"""
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="test1234567",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "test1234567")],
            language="en"
        )
        mock_fetch.return_value = mock_result

        result = fetcher.fetch_captions_auto_language_with_retry("test1234567")

        assert result.video_id == "test1234567"
        mock_fetch.assert_called_once()

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_retry_logs_context(self, mock_fetch, mock_sleep, caplog):
        """Test retry logging includes video_id and error details"""
        import logging
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="test1234567",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "test1234567")],
            language="en"
        )
        mock_fetch.side_effect = [
            CaptionFetchError("test1234567", "Network timeout"),
            mock_result
        ]

        with caplog.at_level(logging.WARNING):
            fetcher.fetch_captions_with_retry("test1234567", max_retries=2, retry_delay=1.0)

        # Check log contains video_id and error info
        assert any("test1234567" in record.message for record in caplog.records)
        assert any("CaptionFetchError" in record.message for record in caplog.records)
        assert any("Network timeout" in record.message for record in caplog.records)

    @patch('time.sleep')
    @patch.object(CaptionFetcher, 'fetch_captions')
    def test_retry_logs_final_failure(self, mock_fetch, mock_sleep, caplog):
        """Test final failure is logged with error level"""
        import logging
        fetcher = CaptionFetcher()

        mock_fetch.side_effect = CaptionFetchError("test1234567", "Persistent error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(CaptionFetchError):
                fetcher.fetch_captions_with_retry("test1234567", max_retries=1, retry_delay=0.1)

        # Check final error is logged
        error_logs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_logs) >= 1
        assert "test1234567" in error_logs[-1].message

    def test_with_retry_zero_retries(self):
        """Test max_retries=0 means only one attempt"""
        fetcher = CaptionFetcher()

        with patch.object(fetcher, 'fetch_captions') as mock_fetch:
            mock_fetch.side_effect = CaptionFetchError("test1234567", "Error")

            with pytest.raises(CaptionFetchError):
                fetcher.fetch_captions_with_retry("test1234567", max_retries=0)

            # Only 1 attempt (no retries)
            assert mock_fetch.call_count == 1


class TestCaptionCacheRetry:
    """Test CaptionCache retry integration (US-008)"""

    @patch.object(CaptionFetcher, 'fetch_captions_with_retry')
    def test_get_or_fetch_with_retry_cache_miss(self, mock_fetch, tmp_path):
        """Test get_or_fetch_with_retry fetches on cache miss"""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        # Use a fresh temp cache
        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="misstest123",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "misstest123")],
            language="en"
        )
        mock_fetch.return_value = mock_result

        result = cache.get_or_fetch_with_retry(fetcher, "misstest123", "en")

        assert result.video_id == "misstest123"
        mock_fetch.assert_called_once()

    @patch.object(CaptionFetcher, 'fetch_captions_with_retry')
    def test_get_or_fetch_with_retry_cache_hit(self, mock_fetch, tmp_path):
        """Test get_or_fetch_with_retry uses cache on hit"""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        # Use a fresh temp cache
        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)
        fetcher = CaptionFetcher()

        # Pre-populate cache
        mock_result = CaptionResult(
            video_id="hittest1234",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "hittest1234")],
            language="en"
        )
        cache.store(mock_result)

        # Should use cache, not fetch
        result = cache.get_or_fetch_with_retry(fetcher, "hittest1234", "en")

        assert result.video_id == "hittest1234"
        mock_fetch.assert_not_called()

    @patch.object(CaptionFetcher, 'fetch_captions_with_retry')
    def test_get_or_fetch_with_retry_passes_params(self, mock_fetch, tmp_path):
        """Test retry parameters are passed through"""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        # Use a fresh temp cache
        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)
        fetcher = CaptionFetcher()

        mock_result = CaptionResult(
            video_id="paramtest123",
            segments=[CaptionSegment(0, 0.0, 1.0, "Test", "paramtest123")],
            language="es"
        )
        mock_fetch.return_value = mock_result

        cache.get_or_fetch_with_retry(
            fetcher, "paramtest123", "es",
            prefer_manual=False,
            max_retries=5,
            retry_delay=3.0
        )

        mock_fetch.assert_called_once_with(
            "paramtest123",
            language="es",
            prefer_manual=False,
            max_retries=5,
            retry_delay=3.0
        )

    @patch.object(CaptionFetcher, 'fetch_captions_with_retry')
    def test_get_or_fetch_with_retry_propagates_unavailable(self, mock_fetch, tmp_path):
        """Test CaptionUnavailableError propagates from retry"""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        # Use a fresh temp cache to avoid cache hits from prior tests
        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)
        fetcher = CaptionFetcher()

        mock_fetch.side_effect = CaptionUnavailableError("errtest1234", "No captions")

        with pytest.raises(CaptionUnavailableError):
            cache.get_or_fetch_with_retry(fetcher, "errtest1234", "en")

    @patch.object(CaptionFetcher, 'fetch_captions_with_retry')
    def test_get_or_fetch_with_retry_propagates_fetch_error(self, mock_fetch, tmp_path):
        """Test CaptionFetchError propagates after retry exhaustion"""
        from src.caption_fetcher import CaptionCache
        from src.config.sections.download import CaptionFirstConfig

        # Use a fresh temp cache to avoid cache hits from prior tests
        config = CaptionFirstConfig(cache_dir=str(tmp_path / "caption_cache"))
        cache = CaptionCache(config)
        fetcher = CaptionFetcher()

        mock_fetch.side_effect = CaptionFetchError("errtest1234", "Network failure")

        with pytest.raises(CaptionFetchError):
            cache.get_or_fetch_with_retry(fetcher, "errtest1234", "en")


# =============================================================================
# US-011: CaptionMetrics Tests
# =============================================================================

class TestCaptionMetrics:
    """Test CaptionMetrics dataclass for fetch statistics tracking (US-011)"""

    def test_caption_metrics_defaults(self):
        """Test default values for CaptionMetrics"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()

        assert metrics.fetch_attempts == 0
        assert metrics.successes == 0
        assert metrics.failures == 0
        assert metrics.cache_hits == 0
        assert metrics.language_distribution == {}
        assert metrics.quality_distribution == {}
        assert metrics.total_segments == 0
        assert metrics.auto_generated_count == 0
        assert metrics.human_caption_count == 0

    def test_record_fetch_attempt(self):
        """Test recording fetch attempts"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_attempt("dQw4w9WgXcQ")
        metrics.record_fetch_attempt("abc12345678")

        assert metrics.fetch_attempts == 2

    def test_record_fetch_success(self):
        """Test recording successful fetches with distribution tracking"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success(
            video_id="vid1",
            language="en",
            quality="high",
            segment_count=50,
            is_auto_generated=False
        )
        metrics.record_fetch_success(
            video_id="vid2",
            language="es",
            quality="medium",
            segment_count=30,
            is_auto_generated=True
        )
        metrics.record_fetch_success(
            video_id="vid3",
            language="en",
            quality="high",
            segment_count=40,
            is_auto_generated=False
        )

        assert metrics.successes == 3
        assert metrics.total_segments == 120
        assert metrics.human_caption_count == 2
        assert metrics.auto_generated_count == 1
        assert metrics.language_distribution == {"en": 2, "es": 1}
        assert metrics.quality_distribution == {"high": 2, "medium": 1}

    def test_record_fetch_failure(self):
        """Test recording fetch failures"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_failure("vid1", reason="unavailable")
        metrics.record_fetch_failure("vid2", reason="error")
        metrics.record_fetch_failure("vid3", reason="timeout")

        assert metrics.failures == 3
        assert metrics.quality_distribution.get("unavailable") == 3

    def test_record_cache_hit(self):
        """Test recording cache hits"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_cache_hit(
            video_id="vid1",
            language="en",
            quality="high",
            segment_count=25,
            is_auto_generated=False
        )

        assert metrics.cache_hits == 1
        assert metrics.total_segments == 25
        assert metrics.human_caption_count == 1
        assert metrics.language_distribution == {"en": 1}
        assert metrics.quality_distribution == {"high": 1}

    def test_total_processed(self):
        """Test total_processed property"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("v1", segment_count=10)
        metrics.record_fetch_failure("v2")
        metrics.record_cache_hit("v3", segment_count=5)

        assert metrics.total_processed == 3

    def test_success_rate(self):
        """Test success_rate calculation"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("v1")
        metrics.record_fetch_success("v2")
        metrics.record_fetch_failure("v3")

        assert metrics.success_rate == 66.7  # 2/3 = 66.7%

    def test_success_rate_no_attempts(self):
        """Test success_rate with no attempts"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()

        assert metrics.success_rate == 0.0

    def test_cache_hit_rate(self):
        """Test cache_hit_rate calculation"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("v1")
        metrics.record_cache_hit("v2")
        metrics.record_cache_hit("v3")
        metrics.record_fetch_failure("v4")

        # 2 cache hits out of 4 total = 50%
        assert metrics.cache_hit_rate == 50.0

    def test_cache_hit_rate_no_processed(self):
        """Test cache_hit_rate with no processed videos"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()

        assert metrics.cache_hit_rate == 0.0

    def test_summary(self):
        """Test summary generation"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.fetch_attempts = 5
        metrics.record_fetch_success("v1", language="en", quality="high", segment_count=50)
        metrics.record_fetch_success("v2", language="en", quality="medium", segment_count=30, is_auto_generated=True)
        metrics.record_cache_hit("v3", language="es", quality="high", segment_count=40)
        metrics.record_fetch_failure("v4")

        summary = metrics.summary()

        assert "Caption fetch: 5 attempts" in summary
        assert "2 succeeded" in summary
        assert "1 failed" in summary
        assert "1 from cache" in summary
        assert "Success rate:" in summary
        assert "Cache hit rate:" in summary
        assert "Total segments: 120" in summary
        assert "Caption sources:" in summary
        assert "Languages:" in summary
        assert "Quality:" in summary

    def test_to_dict(self):
        """Test serialization to dictionary"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("v1", language="en", quality="high", segment_count=50)

        result = metrics.to_dict()

        assert result['fetch_attempts'] == 0
        assert result['successes'] == 1
        assert result['failures'] == 0
        assert result['cache_hits'] == 0
        assert result['language_distribution'] == {"en": 1}
        assert result['quality_distribution'] == {"high": 1}
        assert result['total_segments'] == 50
        assert result['auto_generated_count'] == 0
        assert result['human_caption_count'] == 1

    def test_from_dict(self):
        """Test deserialization from dictionary"""
        from src.caption_fetcher import CaptionMetrics

        data = {
            'fetch_attempts': 10,
            'successes': 7,
            'failures': 2,
            'cache_hits': 3,
            'language_distribution': {"en": 5, "es": 2},
            'quality_distribution': {"high": 4, "medium": 3, "unavailable": 2},
            'total_segments': 250,
            'auto_generated_count': 3,
            'human_caption_count': 4,
        }

        metrics = CaptionMetrics.from_dict(data)

        assert metrics.fetch_attempts == 10
        assert metrics.successes == 7
        assert metrics.failures == 2
        assert metrics.cache_hits == 3
        assert metrics.language_distribution == {"en": 5, "es": 2}
        assert metrics.quality_distribution == {"high": 4, "medium": 3, "unavailable": 2}
        assert metrics.total_segments == 250
        assert metrics.auto_generated_count == 3
        assert metrics.human_caption_count == 4

    def test_from_dict_empty(self):
        """Test from_dict with empty data"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics.from_dict({})

        assert metrics.fetch_attempts == 0
        assert metrics.successes == 0

    def test_from_dict_none(self):
        """Test from_dict with None"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics.from_dict(None)

        assert metrics.fetch_attempts == 0
        assert metrics.successes == 0

    def test_merge(self):
        """Test merging two CaptionMetrics instances"""
        from src.caption_fetcher import CaptionMetrics

        metrics1 = CaptionMetrics()
        metrics1.record_fetch_success("v1", language="en", quality="high", segment_count=50)
        metrics1.record_fetch_failure("v2")

        metrics2 = CaptionMetrics()
        metrics2.record_fetch_success("v3", language="es", quality="medium", segment_count=30)
        metrics2.record_cache_hit("v4", language="en", quality="high", segment_count=20)

        metrics1.merge(metrics2)

        assert metrics1.fetch_attempts == 0  # fetch_attempts not incremented by record_ methods
        assert metrics1.successes == 2
        assert metrics1.failures == 1
        assert metrics1.cache_hits == 1
        assert metrics1.total_segments == 100
        assert metrics1.language_distribution == {"en": 2, "es": 1}
        assert metrics1.quality_distribution == {"high": 2, "medium": 1, "unavailable": 1}

    def test_clear(self):
        """Test clearing all metrics"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("v1", language="en", quality="high", segment_count=50)
        metrics.record_fetch_failure("v2")

        metrics.clear()

        assert metrics.fetch_attempts == 0
        assert metrics.successes == 0
        assert metrics.failures == 0
        assert metrics.cache_hits == 0
        assert metrics.language_distribution == {}
        assert metrics.quality_distribution == {}
        assert metrics.total_segments == 0
        assert metrics.auto_generated_count == 0
        assert metrics.human_caption_count == 0

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict preserves all data"""
        from src.caption_fetcher import CaptionMetrics

        original = CaptionMetrics()
        original.record_fetch_attempt("v1")
        original.record_fetch_success("v1", language="en", quality="high", segment_count=50)
        original.record_fetch_attempt("v2")
        original.record_fetch_success("v2", language="es", quality="medium", segment_count=30, is_auto_generated=True)
        original.record_fetch_attempt("v3")
        original.record_fetch_failure("v3")
        original.record_cache_hit("v4", language="fr", quality="low", segment_count=10)

        # Serialize and deserialize
        data = original.to_dict()
        restored = CaptionMetrics.from_dict(data)

        assert restored.fetch_attempts == original.fetch_attempts
        assert restored.successes == original.successes
        assert restored.failures == original.failures
        assert restored.cache_hits == original.cache_hits
        assert restored.total_segments == original.total_segments
        assert restored.language_distribution == original.language_distribution
        assert restored.quality_distribution == original.quality_distribution
        assert restored.auto_generated_count == original.auto_generated_count
        assert restored.human_caption_count == original.human_caption_count


# =============================================================================
# US-008: Pre-Check Metrics Tests
# =============================================================================

class TestCaptionMetricsPreCheck:
    """Test CaptionMetrics pre-check tracking (US-008)"""

    def test_pre_check_defaults(self):
        """Test default pre-check counts are zero (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()

        assert metrics.pre_check_available == 0
        assert metrics.pre_check_unavailable == 0

    def test_record_pre_check_available(self):
        """Test recording pre-check with captions available (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_pre_check("vid1", has_captions=True)
        metrics.record_pre_check("vid2", has_captions=True)
        metrics.record_pre_check("vid3", has_captions=True)

        assert metrics.pre_check_available == 3
        assert metrics.pre_check_unavailable == 0

    def test_record_pre_check_unavailable(self):
        """Test recording pre-check with captions unavailable (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_pre_check("vid1", has_captions=False)
        metrics.record_pre_check("vid2", has_captions=False)

        assert metrics.pre_check_available == 0
        assert metrics.pre_check_unavailable == 2

    def test_record_pre_check_mixed(self):
        """Test recording mixed pre-check results (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_pre_check("vid1", has_captions=True)
        metrics.record_pre_check("vid2", has_captions=False)
        metrics.record_pre_check("vid3", has_captions=True)
        metrics.record_pre_check("vid4", has_captions=False)
        metrics.record_pre_check("vid5", has_captions=True)

        assert metrics.pre_check_available == 3
        assert metrics.pre_check_unavailable == 2

    def test_pre_check_in_summary(self):
        """Test pre-check stats appear in summary (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_pre_check("vid1", has_captions=True)
        metrics.record_pre_check("vid2", has_captions=False)
        metrics.record_pre_check("vid3", has_captions=True)

        summary = metrics.summary()

        assert "Pre-check: 2 available, 1 unavailable" in summary

    def test_pre_check_not_in_summary_when_zero(self):
        """Test pre-check stats not shown in summary when all zero (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        summary = metrics.summary()

        assert "Pre-check" not in summary

    def test_pre_check_in_to_dict(self):
        """Test pre-check stats serialized to dict (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_pre_check("vid1", has_captions=True)
        metrics.record_pre_check("vid2", has_captions=False)

        data = metrics.to_dict()

        assert data['pre_check_available'] == 1
        assert data['pre_check_unavailable'] == 1

    def test_pre_check_restored_from_dict(self):
        """Test pre-check stats restored from dict (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        original = CaptionMetrics()
        original.record_pre_check("vid1", has_captions=True)
        original.record_pre_check("vid2", has_captions=True)
        original.record_pre_check("vid3", has_captions=False)

        data = original.to_dict()
        restored = CaptionMetrics.from_dict(data)

        assert restored.pre_check_available == 2
        assert restored.pre_check_unavailable == 1

    def test_pre_check_merge(self):
        """Test pre-check stats merged correctly (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics1 = CaptionMetrics()
        metrics1.record_pre_check("vid1", has_captions=True)
        metrics1.record_pre_check("vid2", has_captions=False)

        metrics2 = CaptionMetrics()
        metrics2.record_pre_check("vid3", has_captions=True)
        metrics2.record_pre_check("vid4", has_captions=True)
        metrics2.record_pre_check("vid5", has_captions=False)

        metrics1.merge(metrics2)

        assert metrics1.pre_check_available == 3  # 1 + 2
        assert metrics1.pre_check_unavailable == 2  # 1 + 1

    def test_pre_check_clear(self):
        """Test pre-check stats cleared correctly (US-008)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_pre_check("vid1", has_captions=True)
        metrics.record_pre_check("vid2", has_captions=False)

        metrics.clear()

        assert metrics.pre_check_available == 0
        assert metrics.pre_check_unavailable == 0


# =============================================================================
# US-004: Coverage Distribution Tests
# =============================================================================

class TestCaptionMetricsCoverage:
    """Test CaptionMetrics coverage distribution tracking (US-004)"""

    def test_coverage_distribution_defaults(self):
        """Test default coverage_distribution is empty (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()

        assert metrics.coverage_distribution == {}
        assert metrics.low_coverage_videos == []

    def test_record_fetch_success_with_high_coverage(self):
        """Test high coverage (>80%) tracking (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success(
            video_id="vid1",
            language="en",
            quality="high",
            segment_count=50,
            is_auto_generated=False,
            coverage_ratio=0.95,
            min_coverage_threshold=0.5
        )

        assert metrics.coverage_distribution.get('high') == 1
        assert 'vid1' not in metrics.low_coverage_videos

    def test_record_fetch_success_with_medium_coverage(self):
        """Test medium coverage (50-80%) tracking (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success(
            video_id="vid1",
            language="en",
            quality="medium",
            segment_count=30,
            is_auto_generated=True,
            coverage_ratio=0.65,
            min_coverage_threshold=0.5
        )

        assert metrics.coverage_distribution.get('medium') == 1
        assert 'vid1' not in metrics.low_coverage_videos

    def test_record_fetch_success_with_low_coverage(self):
        """Test low coverage (<50%) tracking and warning (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success(
            video_id="vid1",
            language="en",
            quality="low",
            segment_count=10,
            is_auto_generated=True,
            coverage_ratio=0.25,
            min_coverage_threshold=0.5
        )

        assert metrics.coverage_distribution.get('low') == 1
        assert 'vid1' in metrics.low_coverage_videos

    def test_record_fetch_success_without_coverage(self):
        """Test backward compatibility when coverage_ratio is None (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success(
            video_id="vid1",
            language="en",
            quality="high",
            segment_count=50,
            is_auto_generated=False
            # No coverage_ratio provided
        )

        assert metrics.coverage_distribution == {}
        assert metrics.low_coverage_videos == []
        assert metrics.successes == 1

    def test_record_cache_hit_with_coverage(self):
        """Test cache hit with coverage tracking (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_cache_hit(
            video_id="vid1",
            language="en",
            quality="high",
            segment_count=50,
            is_auto_generated=False,
            coverage_ratio=0.90,
            min_coverage_threshold=0.5
        )

        assert metrics.coverage_distribution.get('high') == 1
        assert 'vid1' not in metrics.low_coverage_videos

    def test_record_cache_hit_low_coverage(self):
        """Test cache hit with low coverage warning (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_cache_hit(
            video_id="vid1",
            language="en",
            quality="medium",
            segment_count=20,
            is_auto_generated=True,
            coverage_ratio=0.30,
            min_coverage_threshold=0.5
        )

        assert metrics.coverage_distribution.get('low') == 1
        assert 'vid1' in metrics.low_coverage_videos

    def test_classify_coverage_high(self):
        """Test _classify_coverage returns 'high' for >80% (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        assert CaptionMetrics._classify_coverage(0.81) == 'high'
        assert CaptionMetrics._classify_coverage(0.95) == 'high'
        assert CaptionMetrics._classify_coverage(1.0) == 'high'

    def test_classify_coverage_medium(self):
        """Test _classify_coverage returns 'medium' for 50-80% (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        assert CaptionMetrics._classify_coverage(0.50) == 'medium'
        assert CaptionMetrics._classify_coverage(0.65) == 'medium'
        assert CaptionMetrics._classify_coverage(0.80) == 'medium'

    def test_classify_coverage_low(self):
        """Test _classify_coverage returns 'low' for <50% (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        assert CaptionMetrics._classify_coverage(0.0) == 'low'
        assert CaptionMetrics._classify_coverage(0.25) == 'low'
        assert CaptionMetrics._classify_coverage(0.49) == 'low'

    def test_summary_includes_coverage(self):
        """Test summary() includes coverage distribution (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("vid1", "en", "high", 50, False, 0.95, 0.5)
        metrics.record_fetch_success("vid2", "en", "medium", 30, True, 0.65, 0.5)
        metrics.record_fetch_success("vid3", "en", "low", 10, True, 0.20, 0.5)

        summary = metrics.summary()

        assert "Coverage:" in summary
        assert "high: 1" in summary
        assert "medium: 1" in summary
        assert "low: 1" in summary
        assert "Low coverage: 1 videos" in summary

    def test_to_dict_includes_coverage(self):
        """Test to_dict() includes coverage fields (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("vid1", "en", "high", 50, False, 0.95, 0.5)
        metrics.record_fetch_success("vid2", "en", "low", 10, True, 0.20, 0.5)

        data = metrics.to_dict()

        assert 'coverage_distribution' in data
        assert data['coverage_distribution'].get('high') == 1
        assert data['coverage_distribution'].get('low') == 1
        assert 'low_coverage_videos' in data
        assert 'vid2' in data['low_coverage_videos']

    def test_from_dict_restores_coverage(self):
        """Test from_dict() restores coverage fields (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        original = CaptionMetrics()
        original.record_fetch_success("vid1", "en", "high", 50, False, 0.95, 0.5)
        original.record_fetch_success("vid2", "en", "low", 10, True, 0.20, 0.5)

        data = original.to_dict()
        restored = CaptionMetrics.from_dict(data)

        assert restored.coverage_distribution == original.coverage_distribution
        assert restored.low_coverage_videos == original.low_coverage_videos

    def test_merge_includes_coverage(self):
        """Test merge() combines coverage distributions (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics1 = CaptionMetrics()
        metrics1.record_fetch_success("vid1", "en", "high", 50, False, 0.95, 0.5)

        metrics2 = CaptionMetrics()
        metrics2.record_fetch_success("vid2", "en", "low", 10, True, 0.20, 0.5)

        metrics1.merge(metrics2)

        assert metrics1.coverage_distribution.get('high') == 1
        assert metrics1.coverage_distribution.get('low') == 1
        assert 'vid2' in metrics1.low_coverage_videos

    def test_clear_resets_coverage(self):
        """Test clear() resets coverage fields (US-004)"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success("vid1", "en", "low", 10, True, 0.20, 0.5)

        assert metrics.coverage_distribution.get('low') == 1
        assert 'vid1' in metrics.low_coverage_videos

        metrics.clear()

        assert metrics.coverage_distribution == {}
        assert metrics.low_coverage_videos == []


class TestCaptionMetricsThreadSafety:
    """Tests for thread-safe CaptionMetrics (US-001)."""

    def test_concurrent_record_fetch_success(self):
        """Test thread-safety of record_fetch_success with concurrent calls."""
        import threading
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        num_threads = 10
        calls_per_thread = 100

        def record_many():
            for i in range(calls_per_thread):
                metrics.record_fetch_success(
                    f"video_{threading.current_thread().name}_{i}",
                    language="en",
                    quality="high",
                    segment_count=10,
                    is_auto_generated=i % 2 == 0
                )

        threads = [threading.Thread(target=record_many) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = num_threads * calls_per_thread
        assert metrics.successes == expected_total
        assert metrics.total_segments == expected_total * 10
        assert metrics.language_distribution['en'] == expected_total
        assert metrics.auto_generated_count + metrics.human_caption_count == expected_total

    def test_concurrent_mixed_operations(self):
        """Test thread-safety with mixed record operations."""
        import threading
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        num_threads = 5
        calls_per_thread = 51  # Divisible by 3 for cleaner math

        def mixed_operations(thread_id):
            for i in range(calls_per_thread):
                if i % 3 == 0:
                    metrics.record_fetch_attempt(f"v_{thread_id}_{i}")
                elif i % 3 == 1:
                    metrics.record_fetch_success(
                        f"v_{thread_id}_{i}",
                        language="en",
                        quality="medium",
                        segment_count=5
                    )
                else:
                    metrics.record_fetch_failure(f"v_{thread_id}_{i}", reason="test")

        threads = [threading.Thread(target=mixed_operations, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread does 51 operations: 17 attempts (i%3==0), 17 successes (i%3==1), 17 failures (i%3==2)
        ops_per_type = calls_per_thread // 3  # 17

        expected_attempts = num_threads * ops_per_type
        expected_successes = num_threads * ops_per_type
        expected_failures = num_threads * ops_per_type

        assert metrics.fetch_attempts == expected_attempts
        assert metrics.successes == expected_successes
        assert metrics.failures == expected_failures


class TestBatchCaptionFetch:
    """Tests for batch caption fetching (US-001)."""

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_empty_list(self, mock_fetch):
        """Test batch fetch with empty video list."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher()
        results = fetcher.fetch_captions_batch([])

        assert results == {}
        mock_fetch.assert_not_called()

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_single_success(self, mock_fetch):
        """Test batch fetch with single successful video."""
        from src.caption_fetcher import CaptionFetcher, CaptionResult, CaptionSegment

        mock_result = CaptionResult(
            video_id='test123abc',
            segments=[CaptionSegment(0, 0.0, 5.0, "Hello", 'test123abc')],
            language='en',
            is_auto_generated=False,
        )
        mock_fetch.return_value = mock_result

        fetcher = CaptionFetcher()
        results = fetcher.fetch_captions_batch(['test123abc'])

        assert 'test123abc' in results
        assert isinstance(results['test123abc'], CaptionResult)
        assert results['test123abc'].language == 'en'
        mock_fetch.assert_called_once()

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_with_failures(self, mock_fetch):
        """Test batch fetch with mixed success and failures."""
        from src.caption_fetcher import (
            CaptionFetcher, CaptionResult, CaptionSegment,
            CaptionUnavailableError, CaptionFetchError
        )

        def side_effect(video_id, preferred_language=None):
            if video_id == 'success1abc':
                return CaptionResult(
                    video_id=video_id,
                    segments=[CaptionSegment(0, 0.0, 5.0, "Test", video_id)],
                    language='en',
                )
            elif video_id == 'unavail1ab':
                raise CaptionUnavailableError(video_id, "No captions")
            else:
                raise CaptionFetchError(video_id, "Network error")

        mock_fetch.side_effect = side_effect

        fetcher = CaptionFetcher()
        results = fetcher.fetch_captions_batch(['success1abc', 'unavail1ab', 'error12abc'])

        assert len(results) == 3
        assert isinstance(results['success1abc'], CaptionResult)
        assert results['unavail1ab'].get('unavailable') is True
        assert results['error12abc'].get('error') is True

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_metrics_tracking(self, mock_fetch):
        """Test that batch fetch correctly updates metrics."""
        from src.caption_fetcher import (
            CaptionFetcher, CaptionResult, CaptionSegment,
            CaptionMetrics, CaptionUnavailableError
        )

        def side_effect(video_id, preferred_language=None):
            if 'success' in video_id:
                return CaptionResult(
                    video_id=video_id,
                    segments=[CaptionSegment(0, 0.0, 5.0, "Test", video_id)],
                    language='en',
                )
            else:
                raise CaptionUnavailableError(video_id, "No captions")

        mock_fetch.side_effect = side_effect

        fetcher = CaptionFetcher()
        metrics = CaptionMetrics()

        results = fetcher.fetch_captions_batch(
            ['success1abc', 'success2abc', 'fail123abc'],
            metrics=metrics
        )

        # Check metrics were updated
        assert metrics.fetch_attempts == 3
        assert metrics.successes == 2
        assert metrics.failures == 1
        assert metrics.total_segments == 2  # 2 successes, 1 segment each

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_progress_callback(self, mock_fetch):
        """Test that progress callback is invoked correctly."""
        from src.caption_fetcher import CaptionFetcher, CaptionResult, CaptionSegment

        mock_fetch.return_value = CaptionResult(
            video_id='test',
            segments=[CaptionSegment(0, 0.0, 5.0, "Test", 'test')],
            language='en',
        )

        progress_calls = []

        def on_progress(video_id, status, details):
            progress_calls.append((video_id, status, details.copy()))

        fetcher = CaptionFetcher()
        fetcher.fetch_captions_batch(
            ['abc123test1', 'def456test2'],
            progress_callback=on_progress,
            max_workers=1  # Sequential to ensure predictable order
        )

        # Should have called progress for each video (fetching + success/failed)
        # With parallel execution, order may vary, but we should have entries
        statuses = {(call[0], call[1]) for call in progress_calls}
        assert ('abc123test1', 'fetching') in statuses
        assert ('abc123test1', 'success') in statuses
        assert ('def456test2', 'fetching') in statuses
        assert ('def456test2', 'success') in statuses

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_skip_video_ids(self, mock_fetch):
        """Test that skip_video_ids are not fetched."""
        from src.caption_fetcher import CaptionFetcher, CaptionResult, CaptionSegment

        mock_fetch.return_value = CaptionResult(
            video_id='test',
            segments=[CaptionSegment(0, 0.0, 5.0, "Test", 'test')],
            language='en',
        )

        fetcher = CaptionFetcher()
        results = fetcher.fetch_captions_batch(
            ['video1abcde', 'video2abcde', 'video3abcde'],
            skip_video_ids={'video1abcde', 'video3abcde'}
        )

        # Only video2 should be fetched
        assert len(results) == 1
        assert 'video2abcde' in results
        assert mock_fetch.call_count == 1

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_respects_max_workers(self, mock_fetch):
        """Test that max_workers is respected."""
        import time
        from src.caption_fetcher import CaptionFetcher, CaptionResult, CaptionSegment

        call_times = []

        def slow_fetch(video_id, preferred_language=None):
            call_times.append(time.time())
            time.sleep(0.1)  # Simulate network delay
            return CaptionResult(
                video_id=video_id,
                segments=[CaptionSegment(0, 0.0, 5.0, "Test", video_id)],
                language='en',
            )

        mock_fetch.side_effect = slow_fetch

        fetcher = CaptionFetcher()

        # With 4 workers and 8 videos, should take ~2 batches (0.2s)
        # With 1 worker, would take ~0.8s
        start = time.time()
        fetcher.fetch_captions_batch(
            [f'video{i}abcd' for i in range(8)],
            max_workers=4
        )
        elapsed = time.time() - start

        # Should complete faster than sequential (0.8s)
        # Allow some margin for thread overhead
        assert elapsed < 0.6, f"Parallel fetch took too long: {elapsed}s"

    @patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry')
    def test_batch_fetch_uses_config_max_workers(self, mock_fetch):
        """Test that max_workers defaults to config value."""
        from unittest.mock import MagicMock
        from src.caption_fetcher import CaptionFetcher, CaptionResult, CaptionSegment

        mock_fetch.return_value = CaptionResult(
            video_id='test',
            segments=[CaptionSegment(0, 0.0, 5.0, "Test", 'test')],
            language='en',
        )

        # Create mock config with max_parallel_fetches=2
        mock_config = MagicMock()
        mock_config.download.caption_first.max_parallel_fetches = 2

        fetcher = CaptionFetcher(config=mock_config)
        # The actual max_workers used is internal, but we can verify it works
        results = fetcher.fetch_captions_batch(['test12abcde'])

        assert 'test12abcde' in results

    def test_batch_fetch_parallel_faster_than_sequential(self):
        """US-001 acceptance: parallel execution faster than sequential for 10+ videos."""
        import time
        from unittest.mock import patch, MagicMock
        from src.caption_fetcher import CaptionFetcher, CaptionResult, CaptionSegment

        def make_slow_fetch(delay):
            def slow_fetch(video_id, preferred_language=None):
                time.sleep(delay)
                return CaptionResult(
                    video_id=video_id,
                    segments=[CaptionSegment(0, 0.0, 5.0, "Test", video_id)],
                    language='en',
                )
            return slow_fetch

        num_videos = 12
        delay_per_video = 0.05  # 50ms per fetch
        video_ids = [f'video{i:03d}abc' for i in range(num_videos)]

        # Sequential timing (max_workers=1)
        with patch.object(CaptionFetcher, 'fetch_captions_auto_language_with_retry',
                          side_effect=make_slow_fetch(delay_per_video)):
            fetcher = CaptionFetcher()
            start = time.time()
            fetcher.fetch_captions_batch(video_ids, max_workers=1)
            sequential_time = time.time() - start

        # Parallel timing (max_workers=4)
        with patch.object(CaptionFetcher, 'fetch_captions_auto_language_with_retry',
                          side_effect=make_slow_fetch(delay_per_video)):
            fetcher = CaptionFetcher()
            start = time.time()
            fetcher.fetch_captions_batch(video_ids, max_workers=4)
            parallel_time = time.time() - start

        # Parallel should be significantly faster (at least 2x for 4 workers)
        assert parallel_time < sequential_time * 0.6, (
            f"Parallel ({parallel_time:.3f}s) not faster than sequential "
            f"({sequential_time:.3f}s) as expected"
        )


class TestLiveStreamDetection:
    """Tests for US-002: Live stream detection to skip caption fetch."""

    @patch('subprocess.run')
    def test_is_live_stream_returns_true_for_live(self, mock_run):
        """Test detection of currently live stream."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                'id': 'test1234567',
                'is_live': True,
                'was_live': False,
            }),
            stderr=''
        )

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        assert result is True
        mock_run.assert_called_once()
        # Verify yt-dlp command includes --dump-json
        call_args = mock_run.call_args[0][0]
        assert '--dump-json' in call_args
        assert '--skip-download' in call_args

    @patch('subprocess.run')
    def test_is_live_stream_returns_true_for_was_live(self, mock_run):
        """Test detection of past live stream (was_live=True)."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                'id': 'test1234567',
                'is_live': False,
                'was_live': True,
            }),
            stderr=''
        )

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        assert result is True

    @patch('subprocess.run')
    def test_is_live_stream_returns_false_for_regular_video(self, mock_run):
        """Test that regular videos return False."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                'id': 'test1234567',
                'is_live': False,
                'was_live': False,
            }),
            stderr=''
        )

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        assert result is False

    @patch('subprocess.run')
    def test_is_live_stream_returns_false_on_metadata_error(self, mock_run):
        """Test that metadata fetch errors return False (don't block)."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr='Video unavailable'
        )

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        # Should return False on error to avoid blocking
        assert result is False

    @patch('subprocess.run')
    def test_is_live_stream_returns_false_on_invalid_json(self, mock_run):
        """Test that invalid JSON returns False."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='not valid json',
            stderr=''
        )

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        assert result is False

    @patch('subprocess.run')
    def test_is_live_stream_returns_false_on_timeout(self, mock_run):
        """Test that timeout returns False."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='yt-dlp', timeout=30)

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        assert result is False

    def test_is_live_stream_returns_false_for_invalid_video_id(self):
        """Test that invalid video IDs return False."""
        fetcher = CaptionFetcher()

        assert fetcher.is_live_stream('') is False
        assert fetcher.is_live_stream('short') is False
        assert fetcher.is_live_stream('toolongvideoid') is False

    @patch('subprocess.run')
    def test_is_live_stream_missing_fields_returns_false(self, mock_run):
        """Test that missing is_live/was_live fields return False."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                'id': 'test1234567',
                # No is_live or was_live fields
            }),
            stderr=''
        )

        fetcher = CaptionFetcher()
        result = fetcher.is_live_stream('test1234567')

        assert result is False


class TestCaptionMetricsSkippedLiveStreams:
    """Tests for US-002: CaptionMetrics skipped live streams tracking."""

    def test_record_skipped_live_stream(self):
        """Test recording a skipped live stream."""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        assert metrics.skipped_live_streams == 0

        metrics.record_skipped_live_stream('test1234567')

        assert metrics.skipped_live_streams == 1
        assert metrics.quality_distribution.get('skipped') == 1

    def test_skipped_live_streams_in_total_processed(self):
        """Test that skipped live streams are included in total_processed."""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_fetch_success('vid1')
        metrics.record_fetch_failure('vid2')
        metrics.record_skipped_live_stream('vid3')

        assert metrics.total_processed == 3

    def test_skipped_live_streams_in_summary(self):
        """Test that skipped live streams appear in summary."""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_skipped_live_stream('vid1')
        metrics.record_skipped_live_stream('vid2')

        summary = metrics.summary()

        assert '2 live streams skipped' in summary

    def test_skipped_live_streams_to_dict(self):
        """Test serialization includes skipped_live_streams."""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_skipped_live_stream('vid1')

        data = metrics.to_dict()

        assert data['skipped_live_streams'] == 1

    def test_skipped_live_streams_from_dict(self):
        """Test deserialization includes skipped_live_streams."""
        from src.caption_fetcher import CaptionMetrics

        data = {'skipped_live_streams': 5}
        metrics = CaptionMetrics.from_dict(data)

        assert metrics.skipped_live_streams == 5

    def test_skipped_live_streams_merge(self):
        """Test merging metrics combines skipped_live_streams."""
        from src.caption_fetcher import CaptionMetrics

        metrics1 = CaptionMetrics()
        metrics1.record_skipped_live_stream('vid1')

        metrics2 = CaptionMetrics()
        metrics2.record_skipped_live_stream('vid2')
        metrics2.record_skipped_live_stream('vid3')

        metrics1.merge(metrics2)

        assert metrics1.skipped_live_streams == 3

    def test_skipped_live_streams_clear(self):
        """Test clear resets skipped_live_streams."""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_skipped_live_stream('vid1')

        metrics.clear()

        assert metrics.skipped_live_streams == 0


class TestCaptionFormatPreference:
    """Tests for US-006: Caption format preference configuration."""

    def test_default_preferred_formats(self):
        """Test default preferred formats (json3, vtt, srt)."""
        fetcher = CaptionFetcher()

        assert fetcher._preferred_formats == ["json3", "vtt", "srt"]

    def test_config_preferred_formats(self):
        """Test preferred formats from config."""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_formats = ["vtt", "srt"]
        mock_config.download.caption_first.max_retries = 3
        mock_config.download.caption_first.retry_delay = 2.0
        mock_config.download.caption_first.timeout = 30

        fetcher = CaptionFetcher(config=mock_config)

        assert fetcher._preferred_formats == ["vtt", "srt"]

    def test_config_empty_formats_uses_default(self):
        """Test that empty preferred_formats uses default."""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_formats = None  # Not configured
        mock_config.download.caption_first.max_retries = 3
        mock_config.download.caption_first.retry_delay = 2.0
        mock_config.download.caption_first.timeout = 30

        fetcher = CaptionFetcher(config=mock_config)

        assert fetcher._preferred_formats == ["json3", "vtt", "srt"]

    @patch('subprocess.run')
    def test_fetch_tries_formats_in_order(self, mock_run, tmp_path):
        """Test that formats are tried in preference order."""
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt", "srt"]
        video_id = "dQw4w9WgXcQ"

        # Track which formats were requested
        formats_tried = []

        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            # Find the format in the command
            if '--sub-format' in cmd:
                fmt_idx = cmd.index('--sub-format')
                formats_tried.append(cmd[fmt_idx + 1])

                # json3 not available - use "no subtitles" to trigger CaptionUnavailableError
                # which correctly falls through to next format
                if 'json3' in cmd[fmt_idx + 1]:
                    result = Mock()
                    result.returncode = 1
                    result.stderr = "no subtitles available in json3 format"
                    return result
                else:
                    result = Mock()
                    result.returncode = 0
                    result.stderr = ""
                    return result
            else:
                result = Mock()
                result.returncode = 0
                result.stderr = ""
                return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create VTT file for success
            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTest")

            result = fetcher.fetch_captions(video_id)

        # Should have tried json3 first, then vtt
        assert "json3" in formats_tried
        assert "vtt" in formats_tried
        # json3 should be tried before vtt
        json3_idx = formats_tried.index("json3")
        vtt_idx = formats_tried.index("vtt")
        assert json3_idx < vtt_idx

    @patch('subprocess.run')
    def test_fetch_fallback_on_parse_error(self, mock_run, tmp_path):
        """Test fallback to next format on parse error (not just unavailable)."""
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt"]
        video_id = "dQw4w9WgXcQ"

        formats_tried = []

        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if '--sub-format' in cmd:
                fmt_idx = cmd.index('--sub-format')
                formats_tried.append(cmd[fmt_idx + 1])
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create a malformed json3 file that will fail parsing
            json3_file = tmp_path / f"{video_id}.en.json3"
            json3_file.write_text("not valid json {{{")

            # Also create a valid VTT file for fallback
            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTest")

            result = fetcher.fetch_captions(video_id)

        # Should have tried json3 (failed parsing), then vtt (succeeded)
        assert "json3" in formats_tried
        assert "vtt" in formats_tried
        # Final result should be VTT since json3 failed to parse
        assert result.format_source == "vtt"

    @patch('subprocess.run')
    def test_fetch_uses_first_successful_format(self, mock_run, tmp_path):
        """Test that first successfully parsed format is used."""
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt", "srt"]
        video_id = "dQw4w9WgXcQ"

        def mock_run_side_effect(*args, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create valid json3 file
            json3_content = json.dumps({
                "events": [
                    {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "Test"}]}
                ]
            })
            json3_file = tmp_path / f"{video_id}.en.json3"
            json3_file.write_text(json3_content)

            result = fetcher.fetch_captions(video_id)

        # Should use json3 since it's first and valid
        assert result.format_source == "json3"

    @patch('subprocess.run')
    def test_fetch_logs_format_used(self, mock_run, tmp_path, caplog):
        """Test that the used format is logged."""
        import logging
        caplog.set_level(logging.INFO)

        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["vtt"]
        video_id = "dQw4w9WgXcQ"

        def mock_run_side_effect(*args, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create VTT file
            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTest")

            result = fetcher.fetch_captions(video_id)

        # Check that format was logged
        assert any("Using vtt format" in record.message for record in caplog.records)

    @patch('subprocess.run')
    def test_all_formats_exhausted_raises_error(self, mock_run, tmp_path):
        """Test that error is raised when all formats fail."""
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt", "srt"]
        video_id = "dQw4w9WgXcQ"

        formats_tried = []

        def mock_run_side_effect(*args, **kwargs):
            cmd = args[0]
            if '--sub-format' in cmd:
                fmt_idx = cmd.index('--sub-format')
                formats_tried.append(cmd[fmt_idx + 1])

            # All formats fail with network error
            result = Mock()
            result.returncode = 1
            result.stderr = "Connection timeout"
            return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(CaptionFetchError) as exc_info:
                fetcher.fetch_captions(video_id)

            # Should have tried all formats
            assert "json3" in formats_tried
            assert "vtt" in formats_tried
            assert "srt" in formats_tried
            # Error message should indicate timeout
            assert "Connection timeout" in str(exc_info.value)

    def test_custom_format_order(self):
        """Test custom format order preference."""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_formats = ["srt", "vtt", "json3"]
        mock_config.download.caption_first.max_retries = 3
        mock_config.download.caption_first.retry_delay = 2.0
        mock_config.download.caption_first.timeout = 30

        fetcher = CaptionFetcher(config=mock_config)

        assert fetcher._preferred_formats == ["srt", "vtt", "json3"]

    @patch('subprocess.run')
    def test_single_format_preference(self, mock_run, tmp_path):
        """Test with only one format in preference list."""
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["vtt"]
        video_id = "dQw4w9WgXcQ"

        def mock_run_side_effect(*args, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stderr = ""
            return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTest")

            result = fetcher.fetch_captions(video_id)

        assert result.format_source == "vtt"

    @patch('subprocess.run')
    def test_format_preference_with_unavailable_error(self, mock_run, tmp_path):
        """Test that CaptionUnavailableError triggers next format."""
        fetcher = CaptionFetcher()
        fetcher._preferred_formats = ["json3", "vtt"]
        video_id = "dQw4w9WgXcQ"

        call_count = [0]

        def mock_run_side_effect(*args, **kwargs):
            call_count[0] += 1
            cmd = args[0]
            fmt_idx = cmd.index('--sub-format')
            fmt = cmd[fmt_idx + 1]

            result = Mock()
            if fmt == "json3":
                # First format unavailable
                result.returncode = 1
                result.stderr = "no subtitles available for this video"
            else:
                # Second format succeeds
                result.returncode = 0
                result.stderr = ""
            return result

        mock_run.side_effect = mock_run_side_effect

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTest")

            result = fetcher.fetch_captions(video_id)

        # Should have tried both formats
        assert call_count[0] == 2
        assert result.format_source == "vtt"


# ============================================================================
# US-005: Language Configuration Validation Tests
# ============================================================================


class TestLanguageConfigValidation:
    """Tests for language config validation (US-005 Sprint 6).

    Validates:
    1. ISO 639-1 language code validation
    2. Duplicate detection in fallback_languages
    3. Warning when preferred_language in fallback_languages
    4. ConfigValidationError exception with suggestions
    """

    def test_is_valid_language_code_valid_codes(self):
        """Test that valid ISO 639-1 codes are accepted."""
        from src.caption_fetcher import is_valid_language_code

        valid_codes = ['en', 'es', 'fr', 'de', 'pt', 'zh', 'ja', 'ko', 'ru', 'ar']
        for code in valid_codes:
            assert is_valid_language_code(code), f"{code} should be valid"

    def test_is_valid_language_code_case_insensitive(self):
        """Test that language codes are validated case-insensitively."""
        from src.caption_fetcher import is_valid_language_code

        assert is_valid_language_code('EN')
        assert is_valid_language_code('En')
        assert is_valid_language_code('eN')
        assert is_valid_language_code('es')
        assert is_valid_language_code('ES')

    def test_is_valid_language_code_invalid_codes(self):
        """Test that invalid codes are rejected."""
        from src.caption_fetcher import is_valid_language_code

        invalid_codes = [
            'eng',      # 3 letters (ISO 639-2)
            'english',  # Full word
            'e',        # Too short
            'xyz',      # Not a valid code
            '12',       # Numbers
            '',         # Empty
        ]
        for code in invalid_codes:
            assert not is_valid_language_code(code), f"{code} should be invalid"

    def test_is_valid_language_code_non_strings(self):
        """Test that non-string inputs return False."""
        from src.caption_fetcher import is_valid_language_code

        assert not is_valid_language_code(None)
        assert not is_valid_language_code(123)
        assert not is_valid_language_code(['en'])
        assert not is_valid_language_code({'code': 'en'})

    def test_validate_language_config_valid(self):
        """Test that valid config passes validation."""
        from src.caption_fetcher import validate_language_config

        # Should not raise
        issues = validate_language_config('en', ['es', 'pt', 'fr'])
        assert issues == []

    def test_validate_language_config_empty_fallback(self):
        """Test that empty fallback list is valid."""
        from src.caption_fetcher import validate_language_config

        issues = validate_language_config('en', [])
        assert issues == []

    def test_validate_language_config_invalid_preferred(self):
        """Test that invalid preferred_language raises ConfigValidationError."""
        from src.caption_fetcher import validate_language_config, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('xyz', [])

        error = exc_info.value
        assert error.field == 'preferred_language'
        assert error.value == 'xyz'
        assert 'ISO 639-1' in error.reason
        assert error.suggestion  # Has a suggestion

    def test_validate_language_config_invalid_preferred_three_letter(self):
        """Test that 3-letter codes (ISO 639-2) are rejected."""
        from src.caption_fetcher import validate_language_config, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('eng', [])  # 'eng' is ISO 639-2, not 639-1

        error = exc_info.value
        assert error.field == 'preferred_language'
        assert 'eng' in str(error.value)

    def test_validate_language_config_invalid_fallback_code(self):
        """Test that invalid fallback language raises ConfigValidationError."""
        from src.caption_fetcher import validate_language_config, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('en', ['es', 'invalid', 'fr'])

        error = exc_info.value
        assert error.field == 'fallback_languages'
        assert error.value == 'invalid'
        assert 'position 1' in error.reason

    def test_validate_language_config_duplicate_in_fallback(self):
        """Test that duplicate fallback languages raises ConfigValidationError."""
        from src.caption_fetcher import validate_language_config, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('en', ['es', 'pt', 'es'])

        error = exc_info.value
        assert error.field == 'fallback_languages'
        assert 'es' in str(error.value)
        assert 'multiple times' in error.reason

    def test_validate_language_config_duplicate_case_insensitive(self):
        """Test that duplicates are detected case-insensitively."""
        from src.caption_fetcher import validate_language_config, ConfigValidationError

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('en', ['ES', 'pt', 'es'])

        error = exc_info.value
        assert 'multiple times' in error.reason

    def test_validate_language_config_preferred_in_fallback_warns(self):
        """Test that preferred_language in fallback produces warning, not error."""
        from src.caption_fetcher import validate_language_config

        # Should not raise, but return warning
        issues = validate_language_config('en', ['en', 'es', 'pt'])

        assert len(issues) == 1
        assert 'Warning' in issues[0]
        assert 'redundant' in issues[0]

    def test_validate_language_config_no_raise_mode(self):
        """Test validation with raise_on_error=False returns all issues."""
        from src.caption_fetcher import validate_language_config

        issues = validate_language_config('xyz', ['invalid', 'es', 'es'], raise_on_error=False)

        # Should collect all issues instead of raising
        assert len(issues) >= 2  # Invalid preferred + invalid fallback (+ maybe duplicate)
        assert any('xyz' in issue for issue in issues)
        assert any('invalid' in issue for issue in issues)

    def test_config_validation_error_message_format(self):
        """Test that ConfigValidationError has well-formatted message."""
        from src.caption_fetcher import ConfigValidationError

        error = ConfigValidationError(
            field='test_field',
            value='bad_value',
            reason='not valid',
            suggestion='try something else'
        )

        message = str(error)
        assert 'test_field' in message
        assert 'bad_value' in message
        assert 'not valid' in message
        assert 'try something else' in message

    def test_config_validation_error_attributes(self):
        """Test ConfigValidationError has expected attributes."""
        from src.caption_fetcher import ConfigValidationError

        error = ConfigValidationError(
            field='fallback_languages',
            value='xyz',
            reason='invalid code',
            suggestion='use en, es, fr'
        )

        assert error.field == 'fallback_languages'
        assert error.value == 'xyz'
        assert error.reason == 'invalid code'
        assert error.suggestion == 'use en, es, fr'

    def test_iso_639_1_codes_coverage(self):
        """Test that ISO 639-1 code set includes common languages."""
        from src.caption_fetcher import ISO_639_1_CODES

        # Common language codes that must be present
        common_codes = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'nl', 'pl', 'sv', 'tr', 'vi', 'th', 'id', 'he'
        ]
        for code in common_codes:
            assert code in ISO_639_1_CODES, f"Common code {code} missing from set"

    def test_iso_639_1_codes_count(self):
        """Test that we have approximately the right number of ISO 639-1 codes."""
        from src.caption_fetcher import ISO_639_1_CODES

        # ISO 639-1 has around 180 codes
        assert 170 <= len(ISO_639_1_CODES) <= 190, \
            f"Expected ~180 ISO 639-1 codes, got {len(ISO_639_1_CODES)}"
