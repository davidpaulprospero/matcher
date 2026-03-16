"""
Tests for US-106-009: Enhanced error aggregation and analysis.

Tests cover:
  - Error category hierarchy (network.timeout -> network -> all)
  - Pipeline-level error aggregation
  - Error trend analysis (increasing/decreasing over stages)
  - Grouping similar errors to avoid repetition
  - Pipeline error summary logging
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.stages.error_aggregator import (
    ErrorAggregator,
    ErrorCategory,
    PipelineErrorAggregator,
    classify_into_subcategory,
    normalize_category,
)


# ---------------------------------------------------------------------------
# Test: Error category hierarchy
# ---------------------------------------------------------------------------

class TestErrorCategoryHierarchy:
    """US-106-009: Error category hierarchy support."""

    def test_get_parent_for_subcategory(self):
        """Subcategories have parent category."""
        assert ErrorCategory.get_parent('network.timeout') == 'network'
        assert ErrorCategory.get_parent('auth.403') == 'auth'
        assert ErrorCategory.get_parent('timeout.socket') == 'timeout'
        assert ErrorCategory.get_parent('validation.input') == 'validation'

    def test_get_parent_for_base_category(self):
        """Base categories have no parent."""
        assert ErrorCategory.get_parent('network') is None
        assert ErrorCategory.get_parent('auth') is None

    def test_get_all_ancestors(self):
        """Get all ancestors including 'all'."""
        ancestors = ErrorCategory.get_all_ancestors('network.timeout')
        assert ancestors == ['network', 'all']

        ancestors = ErrorCategory.get_all_ancestors('network')
        assert ancestors == ['all']

    def test_classify_into_subcategory_network(self):
        """Classify network errors into subcategories."""
        assert classify_into_subcategory(
            ErrorCategory.NETWORK, "Connection timeout"
        ) == 'network.timeout'
        assert classify_into_subcategory(
            ErrorCategory.NETWORK, "DNS failure: getaddrinfo failed"
        ) == 'network.dns'
        assert classify_into_subcategory(
            ErrorCategory.NETWORK, "Connection refused"
        ) == 'network.connection'
        assert classify_into_subcategory(
            ErrorCategory.NETWORK, "SSL certificate error"
        ) == 'network.ssl'
        assert classify_into_subcategory(
            ErrorCategory.NETWORK, "Network unreachable"
        ) == 'network.unreachable'

    def test_classify_into_subcategory_auth(self):
        """Classify auth errors into subcategories."""
        assert classify_into_subcategory(
            ErrorCategory.AUTH, "HTTP 403 Forbidden"
        ) == 'auth.403'
        assert classify_into_subcategory(
            ErrorCategory.AUTH, "Captcha required"
        ) == 'auth.captcha'
        assert classify_into_subcategory(
            ErrorCategory.AUTH, "Rate limit exceeded"
        ) == 'auth.rate_limit'
        assert classify_into_subcategory(
            ErrorCategory.AUTH, "Bot detected"
        ) == 'auth.bot_detection'

    def test_classify_into_subcategory_timeout(self):
        """Classify timeout errors into subcategories."""
        assert classify_into_subcategory(
            ErrorCategory.TIMEOUT, "Socket timeout"
        ) == 'timeout.socket'
        assert classify_into_subcategory(
            ErrorCategory.TIMEOUT, "Stall detected"
        ) == 'timeout.stall'
        assert classify_into_subcategory(
            ErrorCategory.TIMEOUT, "Read timeout"
        ) == 'timeout.read'


# ---------------------------------------------------------------------------
# Test: ErrorAggregator subcategory tracking
# ---------------------------------------------------------------------------

class TestErrorAggregatorSubcategories:
    """US-106-009: ErrorAggregator tracks subcategories."""

    def test_subcategory_tracking(self):
        """Subcategories are tracked alongside main categories."""
        agg = ErrorAggregator()
        agg.record("Connection timeout", ErrorCategory.NETWORK)
        agg.record("DNS failed", ErrorCategory.NETWORK)
        agg.record("HTTP 403", ErrorCategory.AUTH)

        # Check subcategory counts
        assert agg._subcategory_counts.get('network.timeout') == 1
        assert agg._subcategory_counts.get('network.dns') == 1
        assert agg._subcategory_counts.get('auth.403') == 1

    def test_hierarchical_counts(self):
        """Hierarchical counts include both subcategories and parents."""
        agg = ErrorAggregator()
        agg.record("Connection timeout", ErrorCategory.NETWORK)
        agg.record("DNS failed", ErrorCategory.NETWORK)

        hierarchical = agg.get_hierarchical_counts()
        assert hierarchical.get('network.timeout') == 1
        assert hierarchical.get('network.dns') == 1
        assert hierarchical.get('network') == 2  # Parent total
        assert hierarchical.get('all') == 2  # Grand total

    def test_subcategory_samples(self):
        """Subcategory samples are preserved."""
        agg = ErrorAggregator()
        agg.record("Connection timeout error message", ErrorCategory.NETWORK)

        assert agg._subcategory_samples.get('network.timeout') is not None
        assert 'Connection timeout' in agg._subcategory_samples.get('network.timeout', '')


# ---------------------------------------------------------------------------
# Test: ErrorAggregator similar error grouping
# ---------------------------------------------------------------------------

class TestErrorAggregatorSimilarGrouping:
    """US-106-009: Group similar errors to avoid repetition."""

    def test_record_similar_groups_errors(self):
        """record_similar groups similar errors together."""
        agg = ErrorAggregator()
        agg.record_similar("Failed to download video abc123def456", ErrorCategory.NETWORK)
        agg.record_similar("Failed to download video xyz789uvw012", ErrorCategory.NETWORK)
        agg.record_similar("Failed to download video qrs345tuv678", ErrorCategory.NETWORK)

        # Should have grouped these into one pattern
        similar = agg.get_top_similar_errors(limit=5)
        assert len(similar) == 1
        assert similar[0][1] == 3  # count

    def test_normalize_for_grouping(self):
        """Error messages are normalized for grouping."""
        agg = ErrorAggregator()

        # These should normalize to the same pattern
        normalized1 = agg._normalize_for_grouping("Failed to download video abc123def456")
        normalized2 = agg._normalize_for_grouping("Failed to download video xyz789uvw012")

        assert normalized1 == normalized2

    def test_different_errors_not_grouped(self):
        """Different error patterns are not grouped together."""
        agg = ErrorAggregator()
        agg.record_similar("Connection timeout for video abc123", ErrorCategory.NETWORK)
        agg.record_similar("DNS failure for video abc123", ErrorCategory.NETWORK)

        # Should have two different patterns (use min_count=1 to include all patterns)
        similar = agg.get_top_similar_errors(limit=5, min_count=1)
        assert len(similar) == 2


# ---------------------------------------------------------------------------
# Test: PipelineErrorAggregator
# ---------------------------------------------------------------------------

class TestPipelineErrorAggregator:
    """US-106-009: Pipeline-level error aggregation."""

    def test_add_stage_errors(self):
        """Errors can be added from individual stages."""
        pipeline_agg = PipelineErrorAggregator()

        # Add errors from DOWNLOAD_SEGMENTS stage
        agg1 = ErrorAggregator()
        agg1.record("HTTP 403", ErrorCategory.AUTH)
        agg1.record("HTTP 403", ErrorCategory.AUTH)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg1)

        # Add errors from MATCH stage
        agg2 = ErrorAggregator()
        agg2.record("Connection timeout", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('MATCH', agg2)

        assert pipeline_agg.total_errors == 3

    def test_stages_with_errors(self):
        """Can query which stages had errors."""
        pipeline_agg = PipelineErrorAggregator()

        agg1 = ErrorAggregator()
        agg1.record("error", ErrorCategory.AUTH)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg1)

        agg2 = ErrorAggregator()
        agg2.record("error", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('MATCH', agg2)

        stages = pipeline_agg.stages_with_errors
        assert 'DOWNLOAD_SEGMENTS' in stages
        assert 'MATCH' in stages

    def test_get_category_totals(self):
        """Can get totals by category across stages."""
        pipeline_agg = PipelineErrorAggregator()

        agg1 = ErrorAggregator()
        agg1.record("error1", ErrorCategory.AUTH)
        agg1.record("error2", ErrorCategory.AUTH)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg1)

        agg2 = ErrorAggregator()
        agg2.record("error3", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('MATCH', agg2)

        totals = pipeline_agg.get_category_totals()
        assert totals.get('auth') == 2
        assert totals.get('network') == 1

    def test_hierarchical_totals(self):
        """Hierarchical totals include subcategories."""
        pipeline_agg = PipelineErrorAggregator()

        agg = ErrorAggregator()
        agg.record("Connection timeout", ErrorCategory.NETWORK)
        agg.record("DNS failed", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg)

        totals = pipeline_agg.get_hierarchical_totals()
        assert totals.get('network.timeout') == 1
        assert totals.get('network.dns') == 1
        assert totals.get('network') == 2

    def test_analyze_trends_increasing(self):
        """Can detect increasing error trends."""
        pipeline_agg = PipelineErrorAggregator()

        # Early stage has fewer errors
        agg1 = ErrorAggregator()
        agg1.record("error", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('VIDEO_SEARCH', agg1)

        # Later stage has more errors
        agg2 = ErrorAggregator()
        agg2.record("error1", ErrorCategory.NETWORK)
        agg2.record("error2", ErrorCategory.NETWORK)
        agg2.record("error3", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg2)

        trends = pipeline_agg.analyze_trends()
        assert trends.get('network') == 'increasing'

    def test_analyze_trends_decreasing(self):
        """Can detect decreasing error trends."""
        pipeline_agg = PipelineErrorAggregator()

        # Early stage has more errors
        agg1 = ErrorAggregator()
        agg1.record("error1", ErrorCategory.NETWORK)
        agg1.record("error2", ErrorCategory.NETWORK)
        agg1.record("error3", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('VIDEO_SEARCH', agg1)

        # Later stage has fewer errors
        agg2 = ErrorAggregator()
        agg2.record("error", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg2)

        trends = pipeline_agg.analyze_trends()
        assert trends.get('network') == 'decreasing'

    def test_analyze_trends_stable(self):
        """Can detect stable error trends."""
        pipeline_agg = PipelineErrorAggregator()

        agg1 = ErrorAggregator()
        agg1.record("error1", ErrorCategory.NETWORK)
        agg1.record("error2", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('VIDEO_SEARCH', agg1)

        agg2 = ErrorAggregator()
        agg2.record("error1", ErrorCategory.NETWORK)
        agg2.record("error2", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('MATCH', agg2)

        trends = pipeline_agg.analyze_trends()
        assert trends.get('network') == 'stable'

    def test_get_repeated_error_patterns(self):
        """Can identify errors that repeat across stages."""
        pipeline_agg = PipelineErrorAggregator()

        # Record same pattern multiple times within a stage to get repeated errors
        agg1 = ErrorAggregator()
        agg1.record_similar("Failed video abc123def", ErrorCategory.NETWORK)
        agg1.record_similar("Failed video xyz789uvw", ErrorCategory.NETWORK)
        agg1.record_similar("Failed video abc123def", ErrorCategory.NETWORK)  # Repeat
        pipeline_agg.add_stage_errors('VIDEO_SEARCH', agg1)

        agg2 = ErrorAggregator()
        agg2.record_similar("Failed video qrs456tuv", ErrorCategory.NETWORK)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg2)

        patterns = pipeline_agg.get_repeated_error_patterns()
        # Should have grouped similar errors (abc123def pattern repeats)
        assert len(patterns) > 0

    def test_log_pipeline_summary(self, caplog):
        """Pipeline summary logs comprehensive error information."""
        pipeline_agg = PipelineErrorAggregator()

        agg1 = ErrorAggregator()
        agg1.record("HTTP 403", ErrorCategory.AUTH)
        pipeline_agg.add_stage_errors('DOWNLOAD_SEGMENTS', agg1)

        with caplog.at_level(logging.INFO, logger='src.stages.error_aggregator'):
            pipeline_agg.log_pipeline_summary()

        # Should log summary
        assert any('PIPELINE ERROR SUMMARY' in r.message for r in caplog.records)
        assert any('Total errors: 1' in r.message for r in caplog.records)

    def test_empty_pipeline_no_errors(self, caplog):
        """Empty pipeline logs no errors."""
        pipeline_agg = PipelineErrorAggregator()

        with caplog.at_level(logging.INFO, logger='src.stages.error_aggregator'):
            pipeline_agg.log_pipeline_summary()

        assert any('no errors' in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: ErrorAggregator merge with subcategories
# ---------------------------------------------------------------------------

class TestErrorAggregatorMerge:
    """US-106-009: Merging preserves subcategories and similar groups."""

    def test_merge_subcategories(self):
        """Merging preserves subcategory counts."""
        agg1 = ErrorAggregator()
        agg1.record("Connection timeout", ErrorCategory.NETWORK)

        agg2 = ErrorAggregator()
        agg2.record("DNS failed", ErrorCategory.NETWORK)

        agg1.merge(agg2)

        assert agg1._subcategory_counts.get('network.timeout') == 1
        assert agg1._subcategory_counts.get('network.dns') == 1

    def test_merge_similar_groups(self):
        """Merging preserves similar error groups."""
        agg1 = ErrorAggregator()
        agg1.record_similar("Failed video abc123", ErrorCategory.NETWORK)

        agg2 = ErrorAggregator()
        agg2.record_similar("Failed video xyz789", ErrorCategory.NETWORK)

        agg1.merge(agg2)

        # With min_count=1, we get all patterns (2 different patterns with 1 each)
        similar = agg1.get_top_similar_errors(limit=5, min_count=1)
        assert len(similar) == 2

        # With default min_count=2, only repeated patterns are returned (none in this case)
        repeated = agg1.get_top_similar_errors(limit=5)
        assert len(repeated) == 0


# ---------------------------------------------------------------------------
# Test: Integration with PipelineOrchestrator
# ---------------------------------------------------------------------------

class TestPipelineOrchestratorErrorAggregation:
    """US-106-009: PipelineOrchestrator captures stage errors."""

    def test_capture_stage_errors(self):
        """PipelineOrchestrator captures errors from stage metrics."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import StageResult, StageMetrics

        # Create a mock orchestrator directly
        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orchestrator._error_aggregator = PipelineErrorAggregator()
        orchestrator.stage_metrics = {}

        # Create a result with error categories
        metrics = StageMetrics()
        metrics.error_categories = {'network': 2, 'auth': 1}

        result = StageResult(success=True, data={}, metrics=metrics)

        # Capture errors
        orchestrator._capture_stage_errors('DOWNLOAD_SEGMENTS', result)

        # Verify errors were captured (3 total: 2 network + 1 auth)
        assert orchestrator._error_aggregator.total_errors == 3
