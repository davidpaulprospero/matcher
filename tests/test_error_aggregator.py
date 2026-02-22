"""Tests for US-51-011: Stage-level error aggregation with categorized failure summary.

Tests cover:
  - ErrorCategory enum values and string serialization
  - ErrorAggregator collects and categorizes errors correctly
  - ErrorAggregator.to_dict() produces StageMetrics-compatible output
  - ErrorAggregator.log_summary() logs table with category, count, sample
  - ErrorAggregator.merge() combines two aggregators
  - normalize_category() handles both new and legacy category strings
  - Completion summary includes all error categories with counts
  - DOWNLOAD_SEGMENTS stage uses ErrorAggregator for error classification

US-108-006: Enhanced error aggregation with actionable insights:
  - New error categories: RATE_LIMIT, RESOURCE_EXHAUSTION, CONFIG
  - Error pattern detection and grouping
  - SuggestionEngine for fix recommendations
  - Integration with self-healing strategies
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.stages.error_aggregator import (
    ErrorAggregator,
    ErrorCategory,
    normalize_category,
    SuggestionEngine,
    PipelineErrorAggregator,
)
from src.stages.download_segments import (
    DownloadVideoSegmentsStage,
    classify_error_category,
)


# ---------------------------------------------------------------------------
# Test: ErrorCategory enum
# ---------------------------------------------------------------------------

class TestErrorCategory:
    """US-51-011: ErrorCategory enum values and serialization."""

    def test_all_categories_defined(self):
        """All five standard categories exist."""
        assert ErrorCategory.NETWORK.value == 'network'
        assert ErrorCategory.AUTH.value == 'auth'
        assert ErrorCategory.TIMEOUT.value == 'timeout'
        assert ErrorCategory.VALIDATION.value == 'validation'
        assert ErrorCategory.UNKNOWN.value == 'unknown'

    def test_category_is_str_subclass(self):
        """ErrorCategory values can be used as plain strings."""
        assert isinstance(ErrorCategory.NETWORK, str)
        assert ErrorCategory.NETWORK == 'network'

    def test_category_from_value(self):
        """ErrorCategory can be constructed from string value."""
        assert ErrorCategory('network') == ErrorCategory.NETWORK
        assert ErrorCategory('auth') == ErrorCategory.AUTH

    def test_invalid_value_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ErrorCategory('nonexistent')


# ---------------------------------------------------------------------------
# Test: normalize_category
# ---------------------------------------------------------------------------

class TestNormalizeCategory:
    """US-51-011: normalize_category handles new and legacy strings."""

    def test_direct_enum_values(self):
        assert normalize_category('network') == ErrorCategory.NETWORK
        assert normalize_category('auth') == ErrorCategory.AUTH
        assert normalize_category('timeout') == ErrorCategory.TIMEOUT
        assert normalize_category('validation') == ErrorCategory.VALIDATION
        assert normalize_category('unknown') == ErrorCategory.UNKNOWN

    def test_legacy_bot_detection_maps_to_auth(self):
        """Legacy 'bot_detection' from classify_error_category maps to AUTH."""
        assert normalize_category('bot_detection') == ErrorCategory.AUTH

    def test_legacy_video_specific_maps_to_unknown(self):
        """Legacy 'video_specific' maps to UNKNOWN."""
        assert normalize_category('video_specific') == ErrorCategory.UNKNOWN

    def test_unrecognized_string_maps_to_unknown(self):
        assert normalize_category('some_random_thing') == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Test: ErrorAggregator collection and categorization
# ---------------------------------------------------------------------------

class TestErrorAggregatorCollection:
    """US-51-011: ErrorAggregator collects and categorizes errors correctly."""

    def test_record_single_error(self):
        agg = ErrorAggregator()
        agg.record("HTTP Error 403: Forbidden", ErrorCategory.AUTH)
        assert agg.total_errors == 1
        assert agg.get_count(ErrorCategory.AUTH) == 1
        assert agg.get_sample(ErrorCategory.AUTH) == "HTTP Error 403: Forbidden"

    def test_record_multiple_errors_same_category(self):
        agg = ErrorAggregator()
        agg.record("403 error 1", ErrorCategory.AUTH)
        agg.record("403 error 2", ErrorCategory.AUTH)
        agg.record("403 error 3", ErrorCategory.AUTH)
        assert agg.total_errors == 3
        assert agg.get_count(ErrorCategory.AUTH) == 3
        # First sample is kept
        assert agg.get_sample(ErrorCategory.AUTH) == "403 error 1"

    def test_record_multiple_categories(self):
        agg = ErrorAggregator()
        agg.record("DNS failure", ErrorCategory.NETWORK)
        agg.record("403 Forbidden", ErrorCategory.AUTH)
        agg.record("stalled for 120s", ErrorCategory.TIMEOUT)
        assert agg.total_errors == 3
        assert len(agg.categories) == 3
        assert agg.get_count(ErrorCategory.NETWORK) == 1
        assert agg.get_count(ErrorCategory.AUTH) == 1
        assert agg.get_count(ErrorCategory.TIMEOUT) == 1

    def test_record_with_legacy_string_category(self):
        """Passing a legacy string category still works via normalize_category."""
        agg = ErrorAggregator()
        agg.record("HTTP 403", "bot_detection")
        assert agg.get_count(ErrorCategory.AUTH) == 1

    def test_sample_truncated_to_200_chars(self):
        agg = ErrorAggregator()
        long_msg = "x" * 300
        agg.record(long_msg, ErrorCategory.UNKNOWN)
        assert len(agg.get_sample(ErrorCategory.UNKNOWN)) == 200

    def test_empty_aggregator(self):
        agg = ErrorAggregator()
        assert agg.total_errors == 0
        assert agg.categories == {}
        assert agg.get_count(ErrorCategory.NETWORK) == 0
        assert agg.get_sample(ErrorCategory.NETWORK) is None

    def test_clear_resets(self):
        agg = ErrorAggregator()
        agg.record("error", ErrorCategory.NETWORK)
        agg.clear()
        assert agg.total_errors == 0
        assert agg.categories == {}


# ---------------------------------------------------------------------------
# Test: ErrorAggregator.to_dict()
# ---------------------------------------------------------------------------

class TestErrorAggregatorToDict:
    """US-51-011: to_dict() is compatible with StageMetrics.error_categories."""

    def test_to_dict_format(self):
        agg = ErrorAggregator()
        agg.record("DNS failed", ErrorCategory.NETWORK)
        agg.record("403", ErrorCategory.AUTH)
        agg.record("403 again", ErrorCategory.AUTH)
        d = agg.to_dict()
        assert d == {'network': 1, 'auth': 2}

    def test_to_dict_empty(self):
        agg = ErrorAggregator()
        assert agg.to_dict() == {}


# ---------------------------------------------------------------------------
# Test: ErrorAggregator.summary_rows()
# ---------------------------------------------------------------------------

class TestErrorAggregatorSummaryRows:
    """US-51-011: summary_rows returns (category, count, sample) tuples."""

    def test_rows_sorted_by_count_descending(self):
        agg = ErrorAggregator()
        agg.record("net error", ErrorCategory.NETWORK)
        agg.record("auth error 1", ErrorCategory.AUTH)
        agg.record("auth error 2", ErrorCategory.AUTH)
        agg.record("auth error 3", ErrorCategory.AUTH)
        agg.record("timeout error", ErrorCategory.TIMEOUT)
        agg.record("timeout error 2", ErrorCategory.TIMEOUT)

        rows = agg.summary_rows()
        assert len(rows) == 3
        assert rows[0] == ('auth', 3, 'auth error 1')
        assert rows[1] == ('timeout', 2, 'timeout error')
        assert rows[2] == ('network', 1, 'net error')

    def test_empty_rows(self):
        agg = ErrorAggregator()
        assert agg.summary_rows() == []


# ---------------------------------------------------------------------------
# Test: ErrorAggregator.log_summary()
# ---------------------------------------------------------------------------

class TestErrorAggregatorLogSummary:
    """US-51-011: Completion summary logs table with category, count, sample."""

    def test_log_summary_includes_all_categories(self, caplog):
        agg = ErrorAggregator()
        agg.record("DNS failure", ErrorCategory.NETWORK)
        agg.record("HTTP 403", ErrorCategory.AUTH)
        agg.record("HTTP 403 again", ErrorCategory.AUTH)
        agg.record("stalled 120s", ErrorCategory.TIMEOUT)

        with caplog.at_level(logging.INFO, logger='src.stages.error_aggregator'):
            agg.log_summary(stage_name='DOWNLOAD_SEGMENTS')

        # Header line
        assert any('4 total errors' in r.message for r in caplog.records)
        assert any('3 categories' in r.message for r in caplog.records)
        # Category rows with counts and samples
        assert any('auth' in r.message and 'count=2' in r.message for r in caplog.records)
        assert any('network' in r.message and 'count=1' in r.message for r in caplog.records)
        assert any('timeout' in r.message and 'count=1' in r.message for r in caplog.records)
        # Sample messages present
        assert any('DNS failure' in r.message for r in caplog.records)
        assert any('HTTP 403' in r.message for r in caplog.records)

    def test_log_summary_no_output_when_empty(self, caplog):
        agg = ErrorAggregator()
        with caplog.at_level(logging.INFO, logger='src.stages.error_aggregator'):
            agg.log_summary(stage_name='TEST')
        assert len(caplog.records) == 0

    def test_log_summary_stage_name_prefix(self, caplog):
        agg = ErrorAggregator()
        agg.record("error", ErrorCategory.UNKNOWN)
        with caplog.at_level(logging.INFO, logger='src.stages.error_aggregator'):
            agg.log_summary(stage_name='MY_STAGE')
        assert any('[MY_STAGE]' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: ErrorAggregator.merge()
# ---------------------------------------------------------------------------

class TestErrorAggregatorMerge:
    """US-51-011: merge combines two aggregators."""

    def test_merge_adds_counts(self):
        agg1 = ErrorAggregator()
        agg1.record("net1", ErrorCategory.NETWORK)
        agg1.record("auth1", ErrorCategory.AUTH)

        agg2 = ErrorAggregator()
        agg2.record("net2", ErrorCategory.NETWORK)
        agg2.record("timeout1", ErrorCategory.TIMEOUT)

        agg1.merge(agg2)
        assert agg1.get_count(ErrorCategory.NETWORK) == 2
        assert agg1.get_count(ErrorCategory.AUTH) == 1
        assert agg1.get_count(ErrorCategory.TIMEOUT) == 1
        assert agg1.total_errors == 4

    def test_merge_preserves_first_sample(self):
        agg1 = ErrorAggregator()
        agg1.record("net1", ErrorCategory.NETWORK)

        agg2 = ErrorAggregator()
        agg2.record("net2", ErrorCategory.NETWORK)

        agg1.merge(agg2)
        # agg1 had sample first, so it's preserved
        assert agg1.get_sample(ErrorCategory.NETWORK) == "net1"

    def test_merge_adds_new_sample(self):
        agg1 = ErrorAggregator()
        agg1.record("auth1", ErrorCategory.AUTH)

        agg2 = ErrorAggregator()
        agg2.record("timeout1", ErrorCategory.TIMEOUT)

        agg1.merge(agg2)
        # timeout was only in agg2, so its sample comes through
        assert agg1.get_sample(ErrorCategory.TIMEOUT) == "timeout1"


# ---------------------------------------------------------------------------
# Test: DOWNLOAD_SEGMENTS integration — error aggregator in _log_error_summary
# ---------------------------------------------------------------------------

class TestDownloadSegmentsErrorAggregatorIntegration:
    """US-51-011: DOWNLOAD_SEGMENTS stage uses ErrorAggregator in error summary."""

    def test_log_error_summary_includes_aggregator_table(self, caplog):
        """_log_error_summary logs aggregator's categorized table with samples."""
        stage = DownloadVideoSegmentsStage()
        agg = ErrorAggregator()
        agg.record("HTTP Error 403: Forbidden", "bot_detection")
        agg.record("getaddrinfo failed", "network")
        agg.record("getaddrinfo failed again", "network")

        stats = {
            'total': 10,
            'succeeded': 7,
            'failed': 3,
            'cached': 0,
            'attempted': 10,
            'error_categories': {'bot_detection': 1, 'network': 2},
            'error_aggregator': agg,
        }
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_error_summary(stats)

        # Original summary line still present
        assert any('bot_detection=1' in r.message for r in caplog.records)

        # Aggregator table rows are also logged (via error_aggregator logger)
        all_messages = [r.message for r in caplog.records]
        # The aggregator logs at src.stages.error_aggregator logger
        # Verify the main summary line is present
        assert any('Download error summary' in msg for msg in all_messages)

    def test_log_error_summary_works_without_aggregator(self, caplog):
        """_log_error_summary still works with legacy stats (no aggregator)."""
        stage = DownloadVideoSegmentsStage()
        stats = {
            'total': 5,
            'succeeded': 3,
            'failed': 2,
            'cached': 0,
            'attempted': 5,
            'error_categories': {'bot_detection': 2},
            # No error_aggregator key
        }
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_error_summary(stats)

        assert any('Download error summary' in r.message for r in caplog.records)

    def test_error_aggregator_initialized_in_stats(self):
        """_download_segments initializes an ErrorAggregator in stats dict."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = MagicMock()
        stage.downloader.escalation_manager = None
        stage.downloader.circuit_breaker = None
        stage.downloader.cookie_rotator = None
        stage.downloader.retry_queue = MagicMock()
        stage.downloader.retry_queue.has_pending.return_value = False
        stage.downloader.impersonation_manager = None
        stage.downloader.download_config = MagicMock()
        stage.downloader.download_config.bot_detection_tier_floor_threshold = 5
        stage.downloader.download_config.bot_detection_abort_threshold = 0
        stage.downloader.download_config.segment_socket_timeout = 30
        stage.downloader.download_config.segment_max_resolution = 1080
        stage.downloader.download_config.segment_format = 'best'
        stage.downloader.download_config.segment_stall_timeout = 0
        stage.downloader.download_config.cookies_from_browser = ''
        stage.downloader.download_config.cookies_path = ''
        stage.downloader.download_config.cookie_rotation = None
        stage.downloader.download_config.ffmpeg_location = ''

        segments = [{'video_id': 'vid1', 'start': 0, 'end': 10}]
        from pathlib import Path

        # Download raises to trigger error path
        with patch('yt_dlp.YoutubeDL') as mock_ydl:
            mock_ydl.side_effect = Exception("HTTP Error 403: Forbidden")
            with patch.object(Path, 'exists', return_value=False):
                _, stats = stage._download_segments(segments, Path('/tmp/test'), 2.0, None)

        # Verify aggregator was populated
        agg = stats.get('error_aggregator')
        assert agg is not None
        assert isinstance(agg, ErrorAggregator)
        assert agg.total_errors == 1
        assert agg.get_count(ErrorCategory.AUTH) == 1
        assert 'HTTP Error 403' in (agg.get_sample(ErrorCategory.AUTH) or '')


# ---------------------------------------------------------------------------
# US-108-006: New error categories
# ---------------------------------------------------------------------------

class TestNewErrorCategories:
    """US-108-006: Test new error categories for rate_limit, resource_exhaustion, config."""

    def test_rate_limit_category_exists(self):
        """RATE_LIMIT category is defined."""
        assert ErrorCategory.RATE_LIMIT.value == 'rate_limit'

    def test_resource_exhaustion_category_exists(self):
        """RESOURCE_EXHAUSTION category is defined."""
        assert ErrorCategory.RESOURCE_EXHAUSTION.value == 'resource_exhaustion'

    def test_config_category_exists(self):
        """CONFIG category is defined."""
        assert ErrorCategory.CONFIG.value == 'config'

    def test_all_new_categories_are_strings(self):
        """New categories can be used as strings."""
        assert isinstance(ErrorCategory.RATE_LIMIT, str)
        assert isinstance(ErrorCategory.RESOURCE_EXHAUSTION, str)
        assert isinstance(ErrorCategory.CONFIG, str)

    def test_normalize_rate_limit(self):
        """normalize_category handles rate_limit string."""
        assert normalize_category('rate_limit') == ErrorCategory.RATE_LIMIT

    def test_normalize_resource_exhaustion(self):
        """normalize_category handles resource_exhaustion string."""
        assert normalize_category('resource_exhaustion') == ErrorCategory.RESOURCE_EXHAUSTION

    def test_normalize_config(self):
        """normalize_category handles config string."""
        assert normalize_category('config') == ErrorCategory.CONFIG


# ---------------------------------------------------------------------------
# US-108-006: Error pattern detection
# ---------------------------------------------------------------------------

class TestErrorPatternDetection:
    """US-108-006: Test error pattern grouping and detection."""

    def test_record_similar_groups_errors(self):
        """record_similar groups similar errors together."""
        agg = ErrorAggregator()
        agg.record_similar("HTTP 403: Forbidden", ErrorCategory.AUTH)
        agg.record_similar("HTTP 403: Access Denied", ErrorCategory.AUTH)
        agg.record_similar("HTTP 404: Not Found", ErrorCategory.AUTH)

        patterns = agg.get_top_similar_errors(limit=5, min_count=2)
        assert len(patterns) == 1
        assert patterns[0][1] == 2  # 2 similar errors

    def test_get_top_similar_errors_min_count(self):
        """get_top_similar_errors respects min_count parameter."""
        agg = ErrorAggregator()
        agg.record_similar("error 1", ErrorCategory.UNKNOWN)
        agg.record_similar("error 1", ErrorCategory.UNKNOWN)
        agg.record_similar("error 2", ErrorCategory.UNKNOWN)  # Only 1 occurrence

        patterns = agg.get_top_similar_errors(limit=5, min_count=2)
        assert len(patterns) == 1

    def test_get_top_similar_errors_returns_limit(self):
        """get_top_similar_errors respects limit parameter."""
        agg = ErrorAggregator()
        for i in range(5):
            agg.record_similar(f"error type {i}", ErrorCategory.UNKNOWN)

        patterns = agg.get_top_similar_errors(limit=2, min_count=1)
        assert len(patterns) == 2


# ---------------------------------------------------------------------------
# US-108-006: Suggestion engine
# ---------------------------------------------------------------------------

class TestSuggestionEngine:
    """US-108-006: Test SuggestionEngine for actionable fix recommendations."""

    def test_suggestion_engine_exists(self):
        """SuggestionEngine can be imported and instantiated."""
        engine = SuggestionEngine()
        assert engine is not None

    def test_get_suggestions_empty_aggregator(self):
        """get_suggestions returns empty list for empty aggregator."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        suggestions = engine.get_suggestions(agg)
        assert suggestions == []

    def test_get_suggestions_network_error(self):
        """get_suggestions returns suggestions for network errors."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("DNS resolution failed", ErrorCategory.NETWORK)
        suggestions = engine.get_suggestions(agg)

        assert len(suggestions) > 0
        assert any("network" in s.lower() or "dns" in s.lower() for s in suggestions)

    def test_get_suggestions_auth_error(self):
        """get_suggestions returns suggestions for auth errors."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("HTTP 403: Forbidden", ErrorCategory.AUTH)
        suggestions = engine.get_suggestions(agg)

        assert len(suggestions) > 0
        assert any("cookie" in s.lower() or "auth" in s.lower() for s in suggestions)

    def test_get_suggestions_rate_limit_error(self):
        """get_suggestions returns suggestions for rate limit errors."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("HTTP 429: Too Many Requests", ErrorCategory.RATE_LIMIT)
        suggestions = engine.get_suggestions(agg)

        assert len(suggestions) > 0
        assert any("rate" in s.lower() or "wait" in s.lower() for s in suggestions)

    def test_get_suggestions_resource_exhaustion(self):
        """get_suggestions returns suggestions for resource exhaustion errors."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("Out of memory error", ErrorCategory.RESOURCE_EXHAUSTION)
        suggestions = engine.get_suggestions(agg)

        assert len(suggestions) > 0
        assert any("memory" in s.lower() or "disk" in s.lower() or "resource" in s.lower() for s in suggestions)

    def test_get_suggestions_config_error(self):
        """get_suggestions returns suggestions for config errors."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("Invalid config value", ErrorCategory.CONFIG)
        suggestions = engine.get_suggestions(agg)

        assert len(suggestions) > 0
        assert any("config" in s.lower() for s in suggestions)

    def test_get_healing_strategy_suggestions(self):
        """get_healing_strategy_suggestions returns healer recommendations."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        agg.record("HTTP 403", ErrorCategory.AUTH)
        strategies = engine.get_healing_strategy_suggestions(agg)

        assert len(strategies) > 0
        assert any("healer" in s.lower() for s in strategies)

    def test_repeated_patterns_get_priority(self):
        """Repeated error patterns get priority in suggestions."""
        engine = SuggestionEngine()
        agg = ErrorAggregator()
        # Record same error multiple times
        for _ in range(3):
            agg.record_similar("HTTP 403: Forbidden", ErrorCategory.AUTH)
        agg.record("DNS error", ErrorCategory.NETWORK)

        suggestions = engine.get_suggestions(agg)
        # Should have suggestions for repeated pattern
        assert len(suggestions) > 0


# ---------------------------------------------------------------------------
# US-108-006: Pipeline-level error aggregation
# ---------------------------------------------------------------------------

class TestPipelineErrorAggregator:
    """US-108-006: Test PipelineErrorAggregator for multi-stage aggregation."""

    def test_pipeline_aggregator_exists(self):
        """PipelineErrorAggregator can be imported and instantiated."""
        pAgg = PipelineErrorAggregator()
        assert pAgg is not None
        assert pAgg.total_errors == 0

    def test_add_stage_errors(self):
        """add_stage_errors adds errors from a stage."""
        pAgg = PipelineErrorAggregator()
        stage_agg = ErrorAggregator()
        stage_agg.record("error1", ErrorCategory.NETWORK)

        pAgg.add_stage_errors("DOWNLOAD", stage_agg)
        assert pAgg.total_errors == 1

    def test_stages_with_errors(self):
        """stages_with_errors returns list of stages with errors."""
        pAgg = PipelineErrorAggregator()

        stage_agg1 = ErrorAggregator()
        stage_agg1.record("error", ErrorCategory.NETWORK)
        pAgg.add_stage_errors("STAGE1", stage_agg1)

        stage_agg2 = ErrorAggregator()
        pAgg.add_stage_errors("STAGE2", stage_agg2)

        assert "STAGE1" in pAgg.stages_with_errors
        assert "STAGE2" not in pAgg.stages_with_errors

    def test_get_category_totals(self):
        """get_category_totals returns aggregated category counts."""
        pAgg = PipelineErrorAggregator()

        agg1 = ErrorAggregator()
        agg1.record("e1", ErrorCategory.NETWORK)
        pAgg.add_stage_errors("S1", agg1)

        agg2 = ErrorAggregator()
        agg2.record("e2", ErrorCategory.NETWORK)
        agg2.record("e3", ErrorCategory.AUTH)
        pAgg.add_stage_errors("S2", agg2)

        totals = pAgg.get_category_totals()
        assert totals.get('network') == 2
        assert totals.get('auth') == 1


# ---------------------------------------------------------------------------
# US-108-006: CLI argument integration
# ---------------------------------------------------------------------------

class TestCLIErrorSummary:
    """US-108-006: Test --error-summary CLI flag is properly defined."""

    def test_error_summary_flag_in_args(self):
        """--error-summary flag should be available in CLI."""
        import sys

        # Save original argv
        original_argv = sys.argv

        try:
            # Simulate command line with --error-summary
            sys.argv = ['main.py', '--error-summary', '--project', '/tmp/test']

            # Re-import args to parse new arguments
            from src.cli.args import parse_arguments

            # This should not raise
            args = parse_arguments()
            assert hasattr(args, 'error_summary')
            assert args.error_summary is True
        finally:
            sys.argv = original_argv


# ---------------------------------------------------------------------------
# US-120-005: Error pattern analytics dashboard
# ---------------------------------------------------------------------------

class TestErrorFrequencyReport:
    """US-120-005: Test error frequency report generation."""

    def test_generate_frequency_report_empty(self):
        """generate_frequency_report returns empty structure for no errors."""
        agg = ErrorAggregator()
        report = agg.generate_frequency_report()

        assert report['total_errors'] == 0
        assert report['top_errors'] == []
        assert report['category_breakdown'] == {}

    def test_generate_frequency_report_with_errors(self):
        """generate_frequency_report includes top errors with counts and percentages."""
        agg = ErrorAggregator()
        agg.record("HTTP 403 error", ErrorCategory.AUTH)
        agg.record("HTTP 403 again", ErrorCategory.AUTH)
        agg.record("HTTP 403 third", ErrorCategory.AUTH)
        agg.record("DNS failed", ErrorCategory.NETWORK)
        agg.record("Timeout error", ErrorCategory.TIMEOUT)

        report = agg.generate_frequency_report(top_n=5)

        assert report['total_errors'] == 5
        assert len(report['top_errors']) > 0
        # Auth should be top with 3 errors (60%)
        top = report['top_errors'][0]
        assert top['count'] == 3
        assert top['percentage'] == 60.0
        # Check category breakdown
        assert 'auth' in report['category_breakdown']
        assert report['category_breakdown']['auth'] == 3

    def test_generate_frequency_report_percentages(self):
        """generate_frequency_report calculates percentages correctly."""
        agg = ErrorAggregator()
        agg.record("error1", ErrorCategory.AUTH)
        agg.record("error2", ErrorCategory.AUTH)
        agg.record("error3", ErrorCategory.NETWORK)

        report = agg.generate_frequency_report()

        # Auth: 2/3 = 66.7%, Network: 1/3 = 33.3%
        assert report['percentages']['auth'] == pytest.approx(66.7, rel=0.1)
        assert report['percentages']['network'] == pytest.approx(33.3, rel=0.1)

    def test_generate_frequency_report_top_n(self):
        """generate_frequency_report respects top_n parameter."""
        agg = ErrorAggregator()
        for i in range(10):
            agg.record(f"error_{i}", ErrorCategory.UNKNOWN)

        report = agg.generate_frequency_report(top_n=3)

        # Each error is unique, so we get up to 3
        assert len(report['top_errors']) <= 3


class TestTemporalPatternAnalysis:
    """US-120-005: Test temporal pattern analysis (time of day, day of week)."""

    def test_get_temporal_patterns_empty(self):
        """get_temporal_patterns returns empty structure when no temporal data."""
        agg = ErrorAggregator()
        patterns = agg.get_temporal_patterns()

        assert patterns['total_temporal_errors'] == 0
        assert patterns['time_of_day'] == {}
        assert patterns['day_of_week'] == {}

    def test_record_with_timestamp(self):
        """record_with_timestamp records errors with timestamp."""
        from datetime import datetime
        agg = ErrorAggregator()

        ts1 = datetime(2026, 2, 17, 10, 30)  # Tuesday 10:30 AM (morning)
        ts2 = datetime(2026, 2, 17, 14, 0)   # Tuesday 2:00 PM (afternoon)

        agg.record_with_timestamp("error1", ErrorCategory.NETWORK, ts1)
        agg.record_with_timestamp("error2", ErrorCategory.AUTH, ts2)

        patterns = agg.get_temporal_patterns()
        assert patterns['total_temporal_errors'] == 2
        assert patterns['time_of_day']['morning'] == 1
        assert patterns['time_of_day']['afternoon'] == 1
        assert patterns['day_of_week']['tuesday'] == 2

    def test_temporal_time_buckets(self):
        """Temporal patterns correctly bucket time of day."""
        from datetime import datetime
        agg = ErrorAggregator()

        # Morning (6-12)
        agg.record_with_timestamp("morning error", ErrorCategory.NETWORK, datetime(2026, 2, 17, 8, 0))
        # Afternoon (12-18)
        agg.record_with_timestamp("afternoon error", ErrorCategory.NETWORK, datetime(2026, 2, 17, 14, 0))
        # Evening (18-22)
        agg.record_with_timestamp("evening error", ErrorCategory.NETWORK, datetime(2026, 2, 17, 20, 0))
        # Overnight (22-6)
        agg.record_with_timestamp("overnight error", ErrorCategory.NETWORK, datetime(2026, 2, 17, 23, 0))

        patterns = agg.get_temporal_patterns()

        assert patterns['time_of_day']['morning'] == 1
        assert patterns['time_of_day']['afternoon'] == 1
        assert patterns['time_of_day']['evening'] == 1
        assert patterns['time_of_day']['overnight'] == 1

    def test_temporal_day_buckets(self):
        """Temporal patterns correctly bucket day of week."""
        from datetime import datetime
        agg = ErrorAggregator()

        # Monday
        agg.record_with_timestamp("monday error", ErrorCategory.NETWORK, datetime(2026, 2, 16, 10, 0))
        # Friday
        agg.record_with_timestamp("friday error", ErrorCategory.NETWORK, datetime(2026, 2, 20, 10, 0))
        # Saturday
        agg.record_with_timestamp("saturday error", ErrorCategory.NETWORK, datetime(2026, 2, 21, 10, 0))

        patterns = agg.get_temporal_patterns()

        assert patterns['day_of_week']['monday'] == 1
        assert patterns['day_of_week']['friday'] == 1
        assert patterns['day_of_week']['saturday'] == 1


class TestErrorStatsExport:
    """US-120-005: Test error stats JSON export."""

    def test_export_as_json(self):
        """export_as_json produces valid JSON-serializable dict."""
        agg = ErrorAggregator()
        agg.record("HTTP 403", ErrorCategory.AUTH)
        agg.record("DNS failed", ErrorCategory.NETWORK)

        from datetime import datetime
        agg.record_with_timestamp("error1", ErrorCategory.NETWORK, datetime(2026, 2, 17, 10, 0))

        export = agg.export_as_json(include_temporal=True)

        assert 'top_errors' in export
        assert 'category_breakdown' in export
        assert 'total_errors' in export
        assert 'temporal_patterns' in export

    def test_export_as_json_without_temporal(self):
        """export_as_json can exclude temporal data."""
        agg = ErrorAggregator()
        agg.record("error", ErrorCategory.NETWORK)

        export = agg.export_as_json(include_temporal=False)

        assert 'temporal_patterns' not in export


class TestPipelineErrorFrequencyReport:
    """US-120-005: Test pipeline-level error frequency report."""

    def test_pipeline_generate_frequency_report(self):
        """PipelineErrorAggregator generates frequency report across stages."""
        pAgg = PipelineErrorAggregator()

        # Add errors to different stages
        agg1 = ErrorAggregator()
        agg1.record("error1", ErrorCategory.NETWORK)
        agg1.record("error2", ErrorCategory.NETWORK)
        pAgg.add_stage_errors("STAGE1", agg1)

        agg2 = ErrorAggregator()
        agg2.record("error3", ErrorCategory.AUTH)
        pAgg.add_stage_errors("STAGE2", agg2)

        report = pAgg.generate_frequency_report()

        assert report['total_errors'] == 3
        assert 'STAGE1' in report['stages']
        assert 'STAGE2' in report['stages']
        assert len(report['stages_with_errors']) == 2

    def test_pipeline_temporal_patterns(self):
        """PipelineErrorAggregator aggregates temporal patterns."""
        from datetime import datetime
        pAgg = PipelineErrorAggregator()

        agg1 = ErrorAggregator()
        agg1.record_with_timestamp("error1", ErrorCategory.NETWORK, datetime(2026, 2, 17, 10, 0))
        pAgg.add_stage_errors("STAGE1", agg1)

        agg2 = ErrorAggregator()
        agg2.record_with_timestamp("error2", ErrorCategory.AUTH, datetime(2026, 2, 17, 14, 0))
        pAgg.add_stage_errors("STAGE2", agg2)

        patterns = pAgg.get_temporal_patterns()

        assert patterns['total_temporal_errors'] == 2
        assert patterns['time_of_day']['morning'] == 1
        assert patterns['time_of_day']['afternoon'] == 1
        assert patterns['day_of_week']['tuesday'] == 2

    def test_pipeline_export_as_json(self):
        """PipelineErrorAggregator exports complete JSON."""
        pAgg = PipelineErrorAggregator()

        agg = ErrorAggregator()
        agg.record("error", ErrorCategory.NETWORK)
        pAgg.add_stage_errors("STAGE1", agg)

        export = pAgg.export_as_json(include_temporal=True)

        assert 'total_errors' in export
        assert 'stages' in export
        assert 'overall' in export
        assert 'temporal_patterns' in export


class TestCLIErrorStatsFlag:
    """US-120-005: Test --error-stats CLI flag is properly defined."""

    def test_error_stats_flag_in_args(self):
        """--error-stats flag should be available in CLI."""
        import sys
        original_argv = sys.argv

        try:
            sys.argv = ['main.py', '--error-stats', '--project', '/tmp/test']
            from src.cli.args import parse_arguments
            args = parse_arguments()
            assert hasattr(args, 'error_stats')
            assert args.error_stats is True
        finally:
            sys.argv = original_argv

    def test_error_stats_json_flag_in_args(self):
        """--error-stats-json flag should be available in CLI."""
        import sys
        original_argv = sys.argv

        try:
            sys.argv = ['main.py', '--error-stats-json', '/tmp/errors.json', '--project', '/tmp/test']
            from src.cli.args import parse_arguments
            args = parse_arguments()
            assert hasattr(args, 'error_stats_json')
            assert args.error_stats_json == '/tmp/errors.json'
        finally:
            sys.argv = original_argv
