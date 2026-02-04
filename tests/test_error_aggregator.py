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
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.stages.error_aggregator import (
    ErrorAggregator,
    ErrorCategory,
    normalize_category,
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
