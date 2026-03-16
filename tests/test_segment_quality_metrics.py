"""
Tests for sophisticated segment quality scoring heuristics (US-006 Sprint 8).

Tests cover:
- SegmentQualityMetrics dataclass serialization
- calculate_segment_metrics() function with various segment patterns
- Density score calculation (segments per minute, normalized 0-1)
- Timing precision detection (exact millisecond timestamps)
- Text completeness scoring (average chars per segment vs expected 50-200)
- Combined quality_score with weights: 0.4*density + 0.3*precision + 0.3*completeness
- Verification that sparse human captions score higher than dense auto-generated
"""

import pytest
from src.caption_fetcher import (
    CaptionSegment,
    SegmentQualityMetrics,
    calculate_segment_metrics,
)


class TestSegmentQualityMetrics:
    """Test suite for SegmentQualityMetrics dataclass (US-006 Sprint 8)."""

    @pytest.mark.fast
    def test_metrics_dataclass_creation(self):
        """Test SegmentQualityMetrics can be created with all fields."""
        metrics = SegmentQualityMetrics(
            density_score=0.8,
            timing_precision=0.4,
            text_completeness=0.9,
            quality_score=0.72,
            segment_count=12,
            total_duration=60.0,
        )

        assert metrics.density_score == 0.8
        assert metrics.timing_precision == 0.4
        assert metrics.text_completeness == 0.9
        assert metrics.quality_score == 0.72
        assert metrics.segment_count == 12
        assert metrics.total_duration == 60.0

    @pytest.mark.fast
    def test_metrics_to_dict(self):
        """Test SegmentQualityMetrics serialization to dict."""
        metrics = SegmentQualityMetrics(
            density_score=0.8,
            timing_precision=0.4,
            text_completeness=0.9,
            quality_score=0.72,
            segment_count=12,
            total_duration=60.0,
        )

        data = metrics.to_dict()

        assert data['density_score'] == 0.8
        assert data['timing_precision'] == 0.4
        assert data['text_completeness'] == 0.9
        assert data['quality_score'] == 0.72
        assert data['segment_count'] == 12
        assert data['total_duration'] == 60.0

    @pytest.mark.fast
    def test_metrics_from_dict(self):
        """Test SegmentQualityMetrics deserialization from dict."""
        data = {
            'density_score': 0.75,
            'timing_precision': 0.5,
            'text_completeness': 0.85,
            'quality_score': 0.68,
            'segment_count': 15,
            'total_duration': 90.0,
        }

        metrics = SegmentQualityMetrics.from_dict(data)

        assert metrics.density_score == 0.75
        assert metrics.timing_precision == 0.5
        assert metrics.text_completeness == 0.85
        assert metrics.quality_score == 0.68
        assert metrics.segment_count == 15
        assert metrics.total_duration == 90.0

    @pytest.mark.fast
    def test_metrics_from_dict_defaults(self):
        """Test SegmentQualityMetrics from_dict with missing fields uses defaults."""
        data = {'density_score': 0.5}  # Only partial data

        metrics = SegmentQualityMetrics.from_dict(data)

        assert metrics.density_score == 0.5
        assert metrics.timing_precision == 0.0  # Default
        assert metrics.text_completeness == 0.0  # Default
        assert metrics.quality_score == 0.0  # Default
        assert metrics.segment_count == 0  # Default
        assert metrics.total_duration == 0.0  # Default


class TestCalculateSegmentMetrics:
    """Test suite for calculate_segment_metrics() function (US-006 Sprint 8)."""

    @pytest.mark.fast
    def test_empty_segments_returns_zero_metrics(self):
        """Test empty segment list returns all-zero metrics."""
        metrics = calculate_segment_metrics([])

        assert metrics.density_score == 0.0
        assert metrics.timing_precision == 0.0
        assert metrics.text_completeness == 0.0
        assert metrics.quality_score == 0.0
        assert metrics.segment_count == 0
        assert metrics.total_duration == 0.0

    @pytest.mark.fast
    def test_single_segment_calculates_metrics(self):
        """Test single segment produces valid metrics."""
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Hello world, this is a test.", "vid")
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        assert metrics.segment_count == 1
        assert metrics.total_duration == 60.0
        # 1 segment / 1 minute = 1 seg/min, far from optimal 12
        # Density score formula: 1.0 - |deviation| * 0.5
        # |1 - 12| / 12 = 11/12 = 0.917, 0.917 * 0.5 = 0.458, 1.0 - 0.458 = 0.542
        assert metrics.density_score < 0.6  # Below optimal

    @pytest.mark.fast
    def test_density_score_optimal_12_segments_per_minute(self):
        """Test density_score is 1.0 for ~12 segments per minute (optimal).

        Acceptance criteria: density_score = segments per minute (normalize to 0-1 scale).
        """
        # Create 12 segments over 60 seconds
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, "Test segment text " * 3, "vid")
            for i in range(12)
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # 12 segments / 1 minute = optimal density
        assert metrics.density_score == 1.0
        assert metrics.segment_count == 12

    @pytest.mark.fast
    def test_density_score_too_sparse(self):
        """Test density_score drops for sparse captions (<5 segments/minute)."""
        # Create only 2 segments over 60 seconds
        segments = [
            CaptionSegment(0, 0.0, 10.0, "First segment with enough text here.", "vid"),
            CaptionSegment(1, 30.0, 40.0, "Second segment with enough text.", "vid"),
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # 2 segments / 1 minute = very sparse
        assert metrics.density_score < 0.6
        assert metrics.segment_count == 2

    @pytest.mark.fast
    def test_density_score_too_dense(self):
        """Test density_score drops for very dense captions (>25 segments/minute)."""
        # Create 60 segments over 60 seconds (1 per second)
        segments = [
            CaptionSegment(i, i * 1.0, i * 1.0 + 0.9, "Word", "vid")
            for i in range(60)
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # 60 segments / 1 minute = very dense (auto-generated pattern)
        assert metrics.density_score < 0.7
        assert metrics.segment_count == 60

    @pytest.mark.fast
    def test_timing_precision_round_numbers(self):
        """Test timing_precision for captions with round timestamps (human style).

        Acceptance criteria: timing_precision = percentage with exact millisecond timestamps.
        """
        # Human-style captions: round number timestamps (0.0, 5.0, 10.0, etc.)
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First segment here.", "vid"),
            CaptionSegment(1, 5.0, 10.0, "Second segment.", "vid"),
            CaptionSegment(2, 10.0, 15.0, "Third segment.", "vid"),
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # All timestamps are round 100ms intervals, so precision should be 0.0
        assert metrics.timing_precision == 0.0

    @pytest.mark.fast
    def test_timing_precision_millisecond_timestamps(self):
        """Test timing_precision for captions with precise millisecond timestamps (auto style)."""
        # Auto-generated style: precise timestamps (1.234, 2.891, etc.)
        segments = [
            CaptionSegment(0, 0.123, 1.456, "First segment.", "vid"),
            CaptionSegment(1, 1.789, 3.012, "Second segment.", "vid"),
            CaptionSegment(2, 3.345, 4.678, "Third segment.", "vid"),
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # All timestamps have sub-100ms precision, so timing_precision should be 1.0
        assert metrics.timing_precision == 1.0

    @pytest.mark.fast
    def test_timing_precision_mixed(self):
        """Test timing_precision for mixed round and precise timestamps."""
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Round timestamps.", "vid"),  # Round
            CaptionSegment(1, 5.123, 10.456, "Precise timestamps.", "vid"),  # Precise
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # 1/2 segments have precise timestamps
        assert 0.4 < metrics.timing_precision < 0.6

    @pytest.mark.fast
    def test_text_completeness_optimal_range(self):
        """Test text_completeness = 1.0 for average 50-200 chars per segment.

        Acceptance criteria: text_completeness = average chars per segment vs expected (50-200 chars).
        """
        # Create segments with ~100 chars each (in optimal range)
        text = "This is a test segment with enough characters to be in the optimal range. " * 2
        segments = [
            CaptionSegment(0, 0.0, 5.0, text[:100], "vid"),
            CaptionSegment(1, 5.0, 10.0, text[:100], "vid"),
            CaptionSegment(2, 10.0, 15.0, text[:100], "vid"),
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        assert metrics.text_completeness == 1.0

    @pytest.mark.fast
    def test_text_completeness_too_short(self):
        """Test text_completeness drops for very short segments (<50 chars avg)."""
        # Create segments with ~20 chars each
        segments = [
            CaptionSegment(0, 0.0, 5.0, "Short text here.", "vid"),  # 16 chars
            CaptionSegment(1, 5.0, 10.0, "Also short text.", "vid"),  # 16 chars
            CaptionSegment(2, 10.0, 15.0, "Brief segment.", "vid"),  # 14 chars
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # Avg ~15 chars, well below 50 minimum
        assert metrics.text_completeness < 0.5

    @pytest.mark.fast
    def test_text_completeness_too_long(self):
        """Test text_completeness drops for very long segments (>200 chars avg)."""
        # Create segments with ~300 chars each
        long_text = "A" * 300
        segments = [
            CaptionSegment(0, 0.0, 5.0, long_text, "vid"),
            CaptionSegment(1, 5.0, 10.0, long_text, "vid"),
            CaptionSegment(2, 10.0, 15.0, long_text, "vid"),
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # Avg 300 chars, above 200 maximum
        assert metrics.text_completeness < 1.0

    @pytest.mark.fast
    def test_quality_score_weights_correct(self):
        """Test combined quality_score uses weights: 0.4*density + 0.3*precision + 0.3*completeness.

        Acceptance criteria: Combine into quality_score: 0.4*density + 0.3*precision + 0.3*completeness.
        """
        # Create optimal segments (12/min, mixed precision, optimal text)
        text = "This is a test segment with enough characters to be in the optimal range here."
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, text, "vid")
            for i in range(12)
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # Manually verify the weight formula
        expected_quality = (
            0.4 * metrics.density_score +
            0.3 * (1.0 - abs(metrics.timing_precision - 0.4) * 1.5) +  # precision_quality
            0.3 * metrics.text_completeness
        )
        expected_quality = max(0.0, min(1.0, expected_quality))

        # Allow small floating point tolerance
        assert abs(metrics.quality_score - expected_quality) < 0.01

    @pytest.mark.fast
    def test_quality_score_bounded_0_to_1(self):
        """Test quality_score is always bounded between 0.0 and 1.0."""
        # Test with various edge cases
        test_cases = [
            [],  # Empty
            [CaptionSegment(0, 0.0, 0.1, "X", "vid")],  # Single tiny segment
            [CaptionSegment(i, i, i + 0.001, "A" * 500, "vid") for i in range(1000)],  # Extreme
        ]

        for segments in test_cases:
            metrics = calculate_segment_metrics(segments)
            assert 0.0 <= metrics.quality_score <= 1.0

    @pytest.mark.fast
    def test_auto_calculates_duration_from_segments(self):
        """Test total_duration is calculated from segments if not provided."""
        segments = [
            CaptionSegment(0, 0.0, 10.0, "First segment here.", "vid"),
            CaptionSegment(1, 10.0, 30.0, "Second segment longer.", "vid"),
            CaptionSegment(2, 30.0, 60.0, "Third segment ends at 60.", "vid"),
        ]

        # Don't provide total_duration
        metrics = calculate_segment_metrics(segments)

        # Should calculate from last.end_time - first.start_time = 60 - 0 = 60
        assert metrics.total_duration == 60.0


class TestSparseHumanVsDenseAuto:
    """Test that sparse human captions score higher than dense auto-generated (US-006).

    This is the key acceptance criteria: verify the scoring heuristics correctly
    identify human-curated captions as higher quality for matching purposes.
    """

    @pytest.mark.fast
    def test_sparse_human_scores_higher_than_dense_auto(self):
        """Test sparse human captions score higher than dense auto-generated.

        Acceptance criteria: Tests verify sparse human captions score higher than dense auto-generated.

        Human captions characteristics:
        - Sparse: ~6-8 segments per minute
        - Round timestamps: 0.0, 5.0, 10.0, etc.
        - Complete sentences: 80-150 chars per segment

        Auto-generated characteristics:
        - Dense: ~30-60 segments per minute
        - Precise timestamps: 0.123, 1.456, etc.
        - Short fragments: 10-30 chars per segment
        """
        # Human-style captions: 6 segments in 60 seconds, round times, full sentences
        human_segments = [
            CaptionSegment(
                i, i * 10.0, i * 10.0 + 8.0,
                "This is a complete sentence with proper grammar and punctuation. " * 2,
                "vid"
            )
            for i in range(6)
        ]

        # Auto-generated captions: 30 segments in 60 seconds, precise times, short fragments
        auto_segments = [
            CaptionSegment(
                i, i * 2.0 + 0.123, i * 2.0 + 1.789,
                "word fragment here",
                "vid"
            )
            for i in range(30)
        ]

        human_metrics = calculate_segment_metrics(human_segments, total_duration=60.0)
        auto_metrics = calculate_segment_metrics(auto_segments, total_duration=60.0)

        # Human should score higher overall
        assert human_metrics.quality_score > auto_metrics.quality_score, (
            f"Human quality ({human_metrics.quality_score:.3f}) should be > "
            f"auto quality ({auto_metrics.quality_score:.3f})"
        )

        # Human has better text completeness (full sentences vs fragments)
        assert human_metrics.text_completeness > auto_metrics.text_completeness

    @pytest.mark.fast
    def test_human_with_round_timestamps_scores_well(self):
        """Test human captions with round timestamps get good precision_quality score.

        Human captions often have round timestamps (1.0, 5.0, 10.0) because
        they're manually timed. This shouldn't penalize them.
        """
        # Human-style with all round timestamps
        segments = [
            CaptionSegment(0, 0.0, 5.0, "First line of dialogue with context.", "vid"),
            CaptionSegment(1, 5.0, 10.0, "Second line with more dialogue here.", "vid"),
            CaptionSegment(2, 10.0, 15.0, "Third line continues the scene.", "vid"),
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # Round timestamps (timing_precision=0.0) should still give decent quality
        # The precision_quality formula gives ~0.4 for all-round timestamps
        assert metrics.quality_score > 0.3

    @pytest.mark.fast
    def test_optimal_human_captions_score_near_perfect(self):
        """Test that optimal human captions (12/min, good text) score very high."""
        # Optimal: 12 segments/minute, round times, ~100 chars each
        text = "This is an optimal human caption with complete sentences and proper length."
        segments = [
            CaptionSegment(i, i * 5.0, i * 5.0 + 4.0, text + f" ({i})", "vid")
            for i in range(12)
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # Should score very high (>0.8)
        assert metrics.quality_score > 0.8
        assert metrics.density_score == 1.0  # Optimal 12/min

    @pytest.mark.fast
    def test_very_dense_auto_penalized(self):
        """Test very dense auto-generated captions (60+/min) are penalized."""
        # 60 segments per minute with precise timestamps and short text
        segments = [
            CaptionSegment(i, i * 1.0 + 0.123, i * 1.0 + 0.789, "word", "vid")
            for i in range(60)
        ]

        metrics = calculate_segment_metrics(segments, total_duration=60.0)

        # Should be penalized for being too dense and too short text
        assert metrics.quality_score < 0.5
        assert metrics.text_completeness < 0.2  # "word" is only 4 chars
