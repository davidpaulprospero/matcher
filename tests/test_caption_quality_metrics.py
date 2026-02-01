"""Tests for caption quality metrics calculation (US-34-011).

Tests the calculate_segment_metrics function and SegmentQualityMetrics class
for quality scoring of caption segments.
"""

import pytest
from dataclasses import dataclass


@dataclass
class CaptionSegment:
    """Test fixture for CaptionSegment."""
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""


# Import the actual implementation
from src.caption_fetcher import calculate_segment_metrics, SegmentQualityMetrics


class TestCalculateSegmentMetricsQualitySamples:
    """Test calculate_segment_metrics() with high/medium/low quality samples."""

    def test_high_quality_human_captions(self):
        """Human captions with optimal density and completeness score high."""
        # 12 segments per minute (optimal), 100 chars average, round timestamps
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.5,
                           "A" * 100,  # 100 chars - in optimal range
                           "vid")
            for i in range(12)
        ]
        # 12 segments over 60 seconds = 12/min (optimal density)
        metrics = calculate_segment_metrics(segments, 60.0)

        assert metrics.quality_score >= 0.7, f"Expected high quality, got {metrics.quality_score}"
        assert metrics.density_score >= 0.9, f"Density should be near optimal, got {metrics.density_score}"
        assert metrics.text_completeness == 1.0, "100 chars is in optimal range"

    def test_medium_quality_auto_captions(self):
        """Auto-generated captions with moderate density score medium."""
        # 25 segments per minute, short text, precise timestamps
        segments = [
            CaptionSegment(i, i * 2.4, i * 2.4 + 2.3,
                           "A" * 30,  # 30 chars - below optimal
                           "vid")
            for i in range(25)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        assert 0.3 <= metrics.quality_score <= 0.8, f"Expected medium quality, got {metrics.quality_score}"
        # Text completeness should be reduced for short segments
        assert metrics.text_completeness < 1.0, "30 chars is below optimal"

    def test_low_quality_sparse_captions(self):
        """Very sparse captions with minimal content score low."""
        # Only 2 segments per minute, very short text
        segments = [
            CaptionSegment(i, i * 30.0, i * 30.0 + 2.0,
                           "Hi",  # 2 chars - very short
                           "vid")
            for i in range(2)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        assert metrics.quality_score <= 0.5, f"Expected low quality, got {metrics.quality_score}"
        assert metrics.density_score < 0.6, f"Sparse captions should have low density score"
        assert metrics.text_completeness < 0.5, "2 chars is very below optimal"

    def test_low_quality_overly_dense_micro_segments(self):
        """Very dense micro-segments (auto-generated pattern) score lower."""
        # 120 segments per minute, single words
        segments = [
            CaptionSegment(i, i * 0.5, i * 0.5 + 0.4,
                           "word",  # 4 chars
                           "vid")
            for i in range(120)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # Very high density penalizes score
        assert metrics.density_score < 0.7, f"Overly dense should reduce density score"
        assert metrics.text_completeness < 0.2, "4 chars is very short"

    def test_empty_segments_returns_zero_quality(self):
        """Empty segment list returns all zero metrics."""
        metrics = calculate_segment_metrics([], 60.0)

        assert metrics.quality_score == 0.0
        assert metrics.density_score == 0.0
        assert metrics.timing_precision == 0.0
        assert metrics.text_completeness == 0.0
        assert metrics.segment_count == 0


class TestDensityScoreCalculation:
    """Test density_score calculation with various segment densities."""

    def test_optimal_density_12_per_minute(self):
        """12 segments per minute is optimal, should score ~1.0."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "text", "vid")
            for i in range(12)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        assert metrics.density_score >= 0.95, f"12/min should be near 1.0, got {metrics.density_score}"

    def test_low_density_2_per_minute(self):
        """2 segments per minute is very sparse."""
        segments = [
            CaptionSegment(i, i * 30.0, i * 30.0 + 5.0, "text", "vid")
            for i in range(2)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # (12 - 2) / 12 = 0.833 deviation, 1 - 0.833 * 0.5 = 0.583
        assert metrics.density_score < 0.6, f"2/min should be low, got {metrics.density_score}"

    def test_high_density_30_per_minute(self):
        """30 segments per minute is too dense."""
        segments = [
            CaptionSegment(i, i * 2.0, i * 2.0 + 1.5, "text", "vid")
            for i in range(30)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # (30 - 12) / 12 = 1.5 deviation, 1 - 1.5 * 0.5 = 0.25
        assert metrics.density_score < 0.5, f"30/min should reduce score, got {metrics.density_score}"

    def test_very_high_density_60_per_minute(self):
        """60 segments per minute (1/sec) is very dense."""
        segments = [
            CaptionSegment(i, i * 1.0, i * 1.0 + 0.9, "text", "vid")
            for i in range(60)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # (60 - 12) / 12 = 4.0 deviation, 1 - 4.0 * 0.5 = -1.0 -> clamped to 0
        assert metrics.density_score == 0.0, f"60/min should bottom out, got {metrics.density_score}"

    def test_density_with_inferred_duration(self):
        """When total_duration not provided, infer from segment span."""
        segments = [
            CaptionSegment(0, 0.0, 10.0, "text", "vid"),
            CaptionSegment(1, 50.0, 60.0, "text", "vid"),
        ]
        # Duration should be inferred as 60 - 0 = 60 seconds
        metrics = calculate_segment_metrics(segments, None)

        assert metrics.total_duration == 60.0
        # 2 segments / 1 minute = 2/min density
        assert metrics.density_score < 0.6


class TestTimingPrecisionDetection:
    """Test timing_precision detection for auto vs human captions."""

    def test_round_timestamps_indicate_human_captions(self):
        """Round timestamps (1.0, 2.0, 3.0) indicate human curation."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "text", "vid")
            for i in range(10)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # All timestamps end in .0 (round 100ms intervals)
        assert metrics.timing_precision == 0.0, "Round timestamps = 0 precision"

    def test_precise_timestamps_indicate_auto_captions(self):
        """Precise timestamps (1.234, 2.567) indicate auto-generation."""
        segments = [
            CaptionSegment(i, i * 5.123, i * 5.123 + 4.456, "text", "vid")
            for i in range(10)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # All timestamps have non-round milliseconds
        assert metrics.timing_precision == 1.0, "Precise timestamps = 1.0 precision"

    def test_mixed_timestamps_moderate_precision(self):
        """Mixed round and precise timestamps give moderate precision."""
        segments = [
            # Half with round timestamps
            CaptionSegment(0, 0.0, 5.0, "text", "vid"),
            CaptionSegment(1, 5.0, 10.0, "text", "vid"),
            # Half with precise timestamps
            CaptionSegment(2, 10.123, 15.456, "text", "vid"),
            CaptionSegment(3, 15.789, 20.234, "text", "vid"),
        ]
        metrics = calculate_segment_metrics(segments, 30.0)

        # 2 of 4 segments have precise timestamps
        assert 0.4 <= metrics.timing_precision <= 0.6

    def test_optimal_precision_around_40_percent(self):
        """Precision around 0.4 is considered optimal for quality scoring."""
        # Create segments where ~40% have precise timestamps
        segments = []
        for i in range(10):
            if i < 4:  # 4 of 10 = 40% with precise
                segments.append(CaptionSegment(i, i * 5.123, i * 5.123 + 4.456, "A" * 100, "vid"))
            else:  # 6 of 10 = 60% with round
                segments.append(CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 100, "vid"))

        metrics = calculate_segment_metrics(segments, 60.0)

        # ~40% precision is optimal, should contribute to higher quality
        assert metrics.timing_precision == 0.4
        assert metrics.quality_score >= 0.7


class TestTextCompletenessCalculation:
    """Test text_completeness with short/normal/long segments."""

    def test_optimal_length_50_to_200_chars(self):
        """Segments with 50-200 chars get full completeness score."""
        for char_count in [50, 100, 150, 200]:
            segments = [
                CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * char_count, "vid")
                for i in range(12)
            ]
            metrics = calculate_segment_metrics(segments, 60.0)

            assert metrics.text_completeness == 1.0, f"{char_count} chars should be optimal"

    def test_short_segments_reduce_completeness(self):
        """Segments shorter than 50 chars reduce completeness linearly."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 25, "vid")
            for i in range(12)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # 25 chars / 50 optimal = 0.5 completeness
        assert metrics.text_completeness == 0.5

    def test_very_short_segments_low_completeness(self):
        """Very short segments (< 10 chars) have very low completeness."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "Hi", "vid")
            for i in range(12)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # 2 chars / 50 optimal = 0.04 completeness
        assert metrics.text_completeness < 0.1

    def test_long_segments_reduce_completeness(self):
        """Segments longer than 200 chars reduce completeness gradually."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 400, "vid")
            for i in range(12)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # overage = (400 - 200) / 400 = 0.5, completeness = 1 - 0.5 = 0.5
        assert metrics.text_completeness == 0.5

    def test_very_long_segments_low_completeness(self):
        """Very long segments (600+ chars) have low completeness."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 600, "vid")
            for i in range(12)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        # overage = (600 - 200) / 400 = 1.0, completeness = 1 - 1.0 = 0.0
        assert metrics.text_completeness == 0.0

    def test_empty_text_segments(self):
        """Segments with empty text have zero completeness."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "", "vid")
            for i in range(12)
        ]
        metrics = calculate_segment_metrics(segments, 60.0)

        assert metrics.text_completeness == 0.0


class TestSegmentQualityMetricsSerialization:
    """Test SegmentQualityMetrics serialization roundtrip."""

    def test_to_dict_contains_all_fields(self):
        """to_dict() exports all metric fields."""
        metrics = SegmentQualityMetrics(
            density_score=0.85,
            timing_precision=0.42,
            text_completeness=0.95,
            quality_score=0.78,
            segment_count=24,
            total_duration=120.0,
        )

        d = metrics.to_dict()

        assert d['density_score'] == 0.85
        assert d['timing_precision'] == 0.42
        assert d['text_completeness'] == 0.95
        assert d['quality_score'] == 0.78
        assert d['segment_count'] == 24
        assert d['total_duration'] == 120.0

    def test_from_dict_restores_all_fields(self):
        """from_dict() restores all metric fields."""
        data = {
            'density_score': 0.85,
            'timing_precision': 0.42,
            'text_completeness': 0.95,
            'quality_score': 0.78,
            'segment_count': 24,
            'total_duration': 120.0,
        }

        metrics = SegmentQualityMetrics.from_dict(data)

        assert metrics.density_score == 0.85
        assert metrics.timing_precision == 0.42
        assert metrics.text_completeness == 0.95
        assert metrics.quality_score == 0.78
        assert metrics.segment_count == 24
        assert metrics.total_duration == 120.0

    def test_roundtrip_preserves_all_values(self):
        """Full roundtrip: create -> to_dict -> from_dict preserves values."""
        original = SegmentQualityMetrics(
            density_score=0.75,
            timing_precision=0.33,
            text_completeness=0.88,
            quality_score=0.69,
            segment_count=15,
            total_duration=90.5,
        )

        restored = SegmentQualityMetrics.from_dict(original.to_dict())

        assert restored.density_score == original.density_score
        assert restored.timing_precision == original.timing_precision
        assert restored.text_completeness == original.text_completeness
        assert restored.quality_score == original.quality_score
        assert restored.segment_count == original.segment_count
        assert restored.total_duration == original.total_duration

    def test_from_dict_handles_missing_fields(self):
        """from_dict() uses defaults for missing fields."""
        data = {}  # Empty dict

        metrics = SegmentQualityMetrics.from_dict(data)

        assert metrics.density_score == 0.0
        assert metrics.timing_precision == 0.0
        assert metrics.text_completeness == 0.0
        assert metrics.quality_score == 0.0
        assert metrics.segment_count == 0
        assert metrics.total_duration == 0.0

    def test_from_dict_handles_partial_data(self):
        """from_dict() handles partial data with defaults."""
        data = {
            'density_score': 0.9,
            'segment_count': 10,
        }

        metrics = SegmentQualityMetrics.from_dict(data)

        assert metrics.density_score == 0.9
        assert metrics.segment_count == 10
        assert metrics.timing_precision == 0.0  # default
        assert metrics.quality_score == 0.0  # default

    def test_serialization_with_calculated_metrics(self):
        """Serialization works with metrics from calculate_segment_metrics."""
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 100, "vid")
            for i in range(12)
        ]
        original = calculate_segment_metrics(segments, 60.0)

        restored = SegmentQualityMetrics.from_dict(original.to_dict())

        assert restored.density_score == original.density_score
        assert restored.timing_precision == original.timing_precision
        assert restored.text_completeness == original.text_completeness
        assert restored.quality_score == original.quality_score
        assert restored.segment_count == original.segment_count
        assert restored.total_duration == original.total_duration


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_segment(self):
        """Single segment calculates metrics correctly."""
        segments = [CaptionSegment(0, 0.0, 10.0, "A" * 100, "vid")]

        metrics = calculate_segment_metrics(segments, 60.0)

        assert metrics.segment_count == 1
        assert metrics.text_completeness == 1.0  # 100 chars optimal
        assert metrics.density_score < 0.6  # 1/min is sparse

    def test_zero_duration_uses_minimum(self):
        """Zero total_duration uses 1 second minimum."""
        segments = [CaptionSegment(0, 0.0, 0.0, "text", "vid")]

        metrics = calculate_segment_metrics(segments, 0.0)

        assert metrics.total_duration == 1.0  # minimum

    def test_negative_duration_uses_minimum(self):
        """Negative total_duration uses 1 second minimum."""
        segments = [CaptionSegment(0, 0.0, 0.0, "text", "vid")]

        metrics = calculate_segment_metrics(segments, -10.0)

        assert metrics.total_duration == 1.0  # minimum

    def test_quality_score_bounded_0_to_1(self):
        """Quality score is always between 0 and 1."""
        # Very bad metrics
        bad_segments = [CaptionSegment(i, i * 0.1, i * 0.1 + 0.05, "x", "vid") for i in range(600)]
        metrics = calculate_segment_metrics(bad_segments, 60.0)
        assert 0.0 <= metrics.quality_score <= 1.0

        # Very good metrics
        good_segments = [CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 100, "vid") for i in range(12)]
        metrics = calculate_segment_metrics(good_segments, 60.0)
        assert 0.0 <= metrics.quality_score <= 1.0

    def test_all_component_scores_bounded(self):
        """All component scores are bounded between 0 and 1."""
        segments = [CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "A" * 100, "vid") for i in range(12)]
        metrics = calculate_segment_metrics(segments, 60.0)

        assert 0.0 <= metrics.density_score <= 1.0
        assert 0.0 <= metrics.timing_precision <= 1.0
        assert 0.0 <= metrics.text_completeness <= 1.0
        assert 0.0 <= metrics.quality_score <= 1.0
