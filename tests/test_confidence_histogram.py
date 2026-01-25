"""
Tests for confidence histogram logging functionality.

Tests the log_confidence_histogram() function in src/matching/metrics.py.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock


class TestHistogramFunctionExists:
    """Test that the histogram function exists and is importable."""

    def test_function_importable_from_metrics(self):
        """log_confidence_histogram should be importable from metrics module."""
        from src.matching.metrics import log_confidence_histogram
        assert callable(log_confidence_histogram)

    def test_function_importable_from_matching_package(self):
        """log_confidence_histogram should be exported from matching package."""
        from src.matching import log_confidence_histogram
        assert callable(log_confidence_histogram)

    def test_function_in_matching_all(self):
        """log_confidence_histogram should be in matching __all__."""
        from src import matching
        assert 'log_confidence_histogram' in matching.__all__


class TestHistogramBuckets:
    """Test that histogram buckets are correct."""

    def test_six_buckets_defined(self):
        """Should have exactly 6 buckets."""
        from src.matching.metrics import log_confidence_histogram

        # Single value should produce 6 bucket lines
        result = log_confidence_histogram([0.75])
        lines = result.split('\n')

        bucket_lines = [l for l in lines if '|' in l and '-' in l.split('|')[0]]
        assert len(bucket_lines) == 6

    def test_bucket_ranges_correct(self):
        """Buckets should be: 0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0."""
        from src.matching.metrics import log_confidence_histogram

        result = log_confidence_histogram([0.5])

        expected_labels = ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        for label in expected_labels:
            assert label in result, f"Expected bucket {label} in histogram"


class TestBucketCounting:
    """Test that confidences are correctly bucketed."""

    def test_low_confidence_bucket(self):
        """Values 0.0-0.5 should go in first bucket."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.1, 0.2, 0.3, 0.4, 0.49]
        result = log_confidence_histogram(confidences)

        # First bucket should have 5 items
        lines = result.split('\n')
        first_bucket_line = [l for l in lines if '0.0-0.5' in l][0]
        assert '5' in first_bucket_line or '5 ' in first_bucket_line

    def test_medium_confidence_bucket(self):
        """Values 0.6-0.7 should go in third bucket."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.65, 0.68, 0.69]
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        bucket_line = [l for l in lines if '0.6-0.7' in l][0]
        assert '3' in bucket_line

    def test_high_confidence_bucket(self):
        """Values 0.9-1.0 should go in last bucket."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.91, 0.95, 0.99, 1.0]
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        last_bucket_line = [l for l in lines if '0.9-1.0' in l][0]
        assert '4' in last_bucket_line

    def test_boundary_value_half(self):
        """Value 0.5 should go in 0.5-0.6 bucket (not 0.0-0.5)."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.5]
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        # First bucket should be empty
        first_bucket = [l for l in lines if '0.0-0.5' in l][0]
        assert '0 ' in first_bucket or '   0' in first_bucket

        # Second bucket should have 1
        second_bucket = [l for l in lines if '0.5-0.6' in l][0]
        assert '1' in second_bucket

    def test_boundary_value_one(self):
        """Value 1.0 should go in 0.9-1.0 bucket."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [1.0]
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        last_bucket = [l for l in lines if '0.9-1.0' in l][0]
        assert '1' in last_bucket


class TestHistogramFormatting:
    """Test histogram output formatting."""

    def test_includes_bar_characters(self):
        """Histogram should include bar characters."""
        from src.matching.metrics import log_confidence_histogram

        result = log_confidence_histogram([0.75, 0.75, 0.75])
        assert '█' in result

    def test_includes_segment_counts(self):
        """Each bucket should show segment count."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.3, 0.55, 0.65, 0.75, 0.85, 0.95]
        result = log_confidence_histogram(confidences)

        # Each bucket should have exactly 1
        for label in ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']:
            bucket_line = [l for l in result.split('\n') if label in l][0]
            assert '1 ' in bucket_line or '   1' in bucket_line

    def test_includes_percentages(self):
        """Histogram should show percentages."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.75] * 4  # 100% in 0.7-0.8 bucket
        result = log_confidence_histogram(confidences)

        assert '100.0%' in result or '100%' in result

    def test_includes_total_segments(self):
        """Histogram should show total segment count."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = log_confidence_histogram(confidences)

        assert 'Total segments: 5' in result

    def test_includes_header_and_footer(self):
        """Histogram should have header and footer."""
        from src.matching.metrics import log_confidence_histogram

        result = log_confidence_histogram([0.75])

        assert 'Confidence Distribution' in result
        assert '===' in result


class TestHistogramLogging:
    """Test that histogram is logged at INFO level."""

    def test_logs_at_info_level(self):
        """Histogram should be logged at INFO level."""
        from src.matching.metrics import log_confidence_histogram

        with patch('src.matching.metrics.logger') as mock_logger:
            log_confidence_histogram([0.75, 0.85])

            # Should have called info multiple times (header + buckets + footer)
            assert mock_logger.info.call_count >= 8  # At least header + 6 buckets + footer

    def test_all_bucket_lines_logged(self):
        """All bucket lines should be logged."""
        from src.matching.metrics import log_confidence_histogram

        with patch('src.matching.metrics.logger') as mock_logger:
            log_confidence_histogram([0.75])

            logged_text = ' '.join(str(call) for call in mock_logger.info.call_args_list)

            for label in ['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']:
                assert label in logged_text


class TestHistogramEdgeCases:
    """Test edge cases for histogram."""

    def test_empty_confidences(self):
        """Empty confidences list should not crash."""
        from src.matching.metrics import log_confidence_histogram

        result = log_confidence_histogram([])

        assert 'Total segments: 0' in result

    def test_single_confidence(self):
        """Single confidence value should work."""
        from src.matching.metrics import log_confidence_histogram

        result = log_confidence_histogram([0.75])

        assert 'Total segments: 1' in result

    def test_many_confidences(self):
        """Large number of confidences should work."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [i / 100 for i in range(101)]  # 0.0 to 1.0
        result = log_confidence_histogram(confidences)

        assert 'Total segments: 101' in result

    def test_all_same_confidence(self):
        """All same confidence values should work."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.75] * 50
        result = log_confidence_histogram(confidences)

        # 0.7-0.8 bucket should have 50, others 0
        lines = result.split('\n')
        bucket_0708 = [l for l in lines if '0.7-0.8' in l][0]
        assert '50' in bucket_0708

    def test_zero_confidence(self):
        """Zero confidence should go in first bucket."""
        from src.matching.metrics import log_confidence_histogram

        confidences = [0.0]
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        first_bucket = [l for l in lines if '0.0-0.5' in l][0]
        assert '1' in first_bucket


class TestHistogramBarScaling:
    """Test that histogram bars scale correctly."""

    def test_max_bucket_gets_full_bar(self):
        """The bucket with the most items should have the longest bar."""
        from src.matching.metrics import log_confidence_histogram

        # Put lots of items in 0.7-0.8, few elsewhere
        confidences = [0.75] * 100 + [0.55] * 10
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        bucket_0708 = [l for l in lines if '0.7-0.8' in l][0]
        bucket_0506 = [l for l in lines if '0.5-0.6' in l][0]

        # 0.7-0.8 should have more █ characters than 0.5-0.6
        bars_0708 = bucket_0708.count('█')
        bars_0506 = bucket_0506.count('█')

        assert bars_0708 > bars_0506

    def test_empty_buckets_have_no_bar(self):
        """Empty buckets should have no bar characters."""
        from src.matching.metrics import log_confidence_histogram

        # Only put items in one bucket
        confidences = [0.75] * 10
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')
        # First bucket should be empty
        first_bucket = [l for l in lines if '0.0-0.5' in l][0]

        # Should have no █ characters
        assert first_bucket.count('█') == 0

    def test_custom_bar_width(self):
        """Custom bar_width parameter should be respected."""
        from src.matching.metrics import log_confidence_histogram

        result_narrow = log_confidence_histogram([0.75] * 10, bar_width=10)
        result_wide = log_confidence_histogram([0.75] * 10, bar_width=50)

        # Wide histogram should have more █ characters
        assert result_wide.count('█') >= result_narrow.count('█')


class TestHistogramReturnValue:
    """Test that histogram returns the string for testing."""

    def test_returns_string(self):
        """Function should return the histogram string."""
        from src.matching.metrics import log_confidence_histogram

        result = log_confidence_histogram([0.75])

        assert isinstance(result, str)
        assert len(result) > 0

    def test_return_matches_logged(self):
        """Returned string should match what was logged."""
        from src.matching.metrics import log_confidence_histogram

        with patch('src.matching.metrics.logger') as mock_logger:
            result = log_confidence_histogram([0.75])

            # Each line in result should have been logged
            for line in result.split('\n'):
                # At least check a sample
                if 'Confidence Distribution' in line:
                    logged = any('Confidence Distribution' in str(call) for call in mock_logger.info.call_args_list)
                    assert logged


class TestMatchStageIntegration:
    """Test that histogram is integrated into match stage."""

    def test_histogram_import_in_match_stage(self):
        """log_confidence_histogram should be used in match stage."""
        # Read the match stage source to verify integration
        import inspect
        from src.stages import match

        source = inspect.getsource(match)
        assert 'log_confidence_histogram' in source

    def test_histogram_called_after_quality_summary(self):
        """Histogram should be called after quality summary."""
        import inspect
        from src.stages import match

        source = inspect.getsource(match)

        # Find positions
        quality_pos = source.find('log_quality_summary')
        histogram_pos = source.find('log_confidence_histogram')

        assert quality_pos > 0, "log_quality_summary not found in match stage"
        assert histogram_pos > 0, "log_confidence_histogram not found in match stage"
        assert histogram_pos > quality_pos, "Histogram should come after quality summary"


class TestDistributionVerification:
    """Test that histogram correctly reflects confidence distribution."""

    def test_uniform_distribution(self):
        """Uniform distribution should show similar counts across buckets."""
        from src.matching.metrics import log_confidence_histogram

        # 2 values in each bucket (10 total, but 0.5 goes to second bucket)
        confidences = [0.25, 0.45, 0.55, 0.58, 0.65, 0.68, 0.75, 0.78, 0.85, 0.88, 0.95, 0.98]
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')

        # Each bucket should have 2 items
        for label in ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']:
            bucket_line = [l for l in lines if label in l][0]
            # Extract count - should be 2
            parts = bucket_line.split('|')
            count_part = parts[-1].strip()
            assert '2' in count_part.split()[0]

    def test_skewed_distribution(self):
        """Skewed distribution should show appropriate counts."""
        from src.matching.metrics import log_confidence_histogram

        # Most values high confidence
        confidences = [0.3] + [0.95] * 99
        result = log_confidence_histogram(confidences)

        lines = result.split('\n')

        # First bucket should have 1
        first_bucket = [l for l in lines if '0.0-0.5' in l][0]
        # Last bucket should have 99
        last_bucket = [l for l in lines if '0.9-1.0' in l][0]

        assert '1' in first_bucket
        assert '99' in last_bucket
