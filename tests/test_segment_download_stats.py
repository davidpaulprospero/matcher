"""Tests for US-52-010: SegmentDownloadStats dataclass.

Verifies:
  - Default field values on construction
  - increment_success() updates succeeded, attempted, segment_durations, total_bytes
  - increment_failure() updates failed, attempted, error_categories, error_aggregator
  - increment_cached() updates cached, attempted, total_bytes
  - Fields are independent (no cross-contamination between instances)
"""

import pytest

from src.stages.download_segments import SegmentDownloadStats
from src.stages.error_aggregator import ErrorAggregator


class TestSegmentDownloadStatsDefaults:
    """Verify all fields initialise to sensible zero/empty values."""

    def test_numeric_defaults(self):
        stats = SegmentDownloadStats()
        assert stats.succeeded == 0
        assert stats.failed == 0
        assert stats.cached == 0
        assert stats.attempted == 0
        assert stats.total == 0
        assert stats.retry_count == 0
        assert stats.total_bytes == 0

    def test_collection_defaults(self):
        stats = SegmentDownloadStats()
        assert stats.segment_durations == []
        assert stats.error_categories == {}
        assert isinstance(stats.error_aggregator, ErrorAggregator)

    def test_total_from_constructor(self):
        stats = SegmentDownloadStats(total=42)
        assert stats.total == 42
        assert stats.succeeded == 0

    def test_progress_hooks_data_defaults(self):
        stats = SegmentDownloadStats()
        assert stats.progress_hooks_data == {
            'total_downloaded_bytes': 0,
            'segments_with_progress': 0,
            'segments_finished': 0,
        }


class TestIncrementSuccess:
    """Verify increment_success() mutator."""

    def test_basic_increment(self):
        stats = SegmentDownloadStats(total=5)
        stats.increment_success()
        assert stats.succeeded == 1
        assert stats.attempted == 1
        assert stats.failed == 0

    def test_with_duration_and_bytes(self):
        stats = SegmentDownloadStats(total=3)
        stats.increment_success(duration=2.5, file_bytes=1024)
        assert stats.succeeded == 1
        assert stats.attempted == 1
        assert stats.segment_durations == [2.5]
        assert stats.total_bytes == 1024

    def test_accumulates_across_calls(self):
        stats = SegmentDownloadStats(total=10)
        stats.increment_success(duration=1.0, file_bytes=100)
        stats.increment_success(duration=2.0, file_bytes=200)
        assert stats.succeeded == 2
        assert stats.attempted == 2
        assert stats.segment_durations == [1.0, 2.0]
        assert stats.total_bytes == 300

    def test_zero_duration_not_appended(self):
        stats = SegmentDownloadStats()
        stats.increment_success(duration=0.0, file_bytes=50)
        assert stats.segment_durations == []
        assert stats.total_bytes == 50


class TestIncrementFailure:
    """Verify increment_failure() mutator."""

    def test_basic_increment(self):
        stats = SegmentDownloadStats(total=5)
        stats.increment_failure()
        assert stats.failed == 1
        assert stats.attempted == 1
        assert stats.succeeded == 0

    def test_with_category(self):
        stats = SegmentDownloadStats()
        stats.increment_failure(category='bot_detection')
        assert stats.error_categories == {'bot_detection': 1}

    def test_category_accumulates(self):
        stats = SegmentDownloadStats()
        stats.increment_failure(category='network')
        stats.increment_failure(category='network')
        stats.increment_failure(category='timeout')
        assert stats.error_categories == {'network': 2, 'timeout': 1}
        assert stats.failed == 3
        assert stats.attempted == 3

    def test_with_category_and_error_msg(self):
        stats = SegmentDownloadStats()
        stats.increment_failure(category='bot_detection', error_msg='HTTP Error 403')
        assert stats.error_categories == {'bot_detection': 1}
        assert stats.error_aggregator.total_errors == 1

    def test_error_msg_without_category_not_recorded_in_aggregator(self):
        """Error aggregator requires a category to record."""
        stats = SegmentDownloadStats()
        stats.increment_failure(error_msg='some error')
        assert stats.failed == 1
        assert stats.error_aggregator.total_errors == 0


class TestIncrementCached:
    """Verify increment_cached() mutator."""

    def test_basic_increment(self):
        stats = SegmentDownloadStats(total=5)
        stats.increment_cached()
        assert stats.cached == 1
        assert stats.attempted == 1
        assert stats.succeeded == 0
        assert stats.failed == 0

    def test_with_bytes(self):
        stats = SegmentDownloadStats()
        stats.increment_cached(file_bytes=2048)
        assert stats.total_bytes == 2048

    def test_accumulates(self):
        stats = SegmentDownloadStats()
        stats.increment_cached(file_bytes=100)
        stats.increment_cached(file_bytes=200)
        assert stats.cached == 2
        assert stats.attempted == 2
        assert stats.total_bytes == 300


class TestInstanceIsolation:
    """Verify that separate instances don't share mutable defaults."""

    def test_separate_segment_durations(self):
        a = SegmentDownloadStats()
        b = SegmentDownloadStats()
        a.segment_durations.append(1.0)
        assert b.segment_durations == []

    def test_separate_error_categories(self):
        a = SegmentDownloadStats()
        b = SegmentDownloadStats()
        a.error_categories['network'] = 5
        assert b.error_categories == {}

    def test_separate_error_aggregators(self):
        a = SegmentDownloadStats()
        b = SegmentDownloadStats()
        a.error_aggregator.record('err', 'cat')
        assert b.error_aggregator.total_errors == 0

    def test_separate_progress_hooks_data(self):
        a = SegmentDownloadStats()
        b = SegmentDownloadStats()
        a.progress_hooks_data['segments_finished'] = 10
        assert b.progress_hooks_data['segments_finished'] == 0
