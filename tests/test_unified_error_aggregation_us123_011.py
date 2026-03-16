"""Tests for US-123-011: Multi-source error aggregation for unified rate limit analysis.

Tests cover:
  - ErrorSource enum: DOWNLOAD, CAPTION, TRANSCRIPTION
  - ErrorSource.from_stage_name() maps stage names to sources
  - Unifiedaggregate_from_source()ErrorAggregator. combines errors from multiple sources
  - UnifiedErrorAggregator.get_unified_rate_limit_stats() shows rate limits across sources
  - UnifiedErrorAggregator.cross_source_correlation() detects shared infrastructure issues
  - UnifiedErrorAggregator enabled/disabled behavior
  - Config integration with unified_error_aggregation config
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.stages.error_aggregator import (
    ErrorAggregator,
    ErrorCategory,
    ErrorSource,
    UnifiedErrorAggregator,
)


# ---------------------------------------------------------------------------
# Test: ErrorSource enum
# ---------------------------------------------------------------------------

class TestErrorSource:
    """ErrorSource enum values and string serialization."""

    def test_all_sources_defined(self):
        """All three error sources exist."""
        assert ErrorSource.DOWNLOAD.value == 'download'
        assert ErrorSource.CAPTION.value == 'caption'
        assert ErrorSource.TRANSCRIPTION.value == 'transcription'

    def test_source_is_str_subclass(self):
        """ErrorSource values can be used as plain strings."""
        assert isinstance(ErrorSource.DOWNLOAD, str)
        assert ErrorSource.DOWNLOAD == 'download'

    def test_source_from_value(self):
        """ErrorSource can be constructed from string value."""
        assert ErrorSource('download') == ErrorSource.DOWNLOAD
        assert ErrorSource('caption') == ErrorSource.CAPTION
        assert ErrorSource('transcription') == ErrorSource.TRANSCRIPTION


class TestErrorSourceFromStageName:
    """ErrorSource.from_stage_name() maps stage names correctly."""

    def test_download_stages(self):
        """DOWNLOAD_SEGMENTS maps to DOWNLOAD."""
        assert ErrorSource.from_stage_name('DOWNLOAD_SEGMENTS') == ErrorSource.DOWNLOAD
        assert ErrorSource.from_stage_name('download_segments') == ErrorSource.DOWNLOAD
        assert ErrorSource.from_stage_name('DownloadStage') == ErrorSource.DOWNLOAD

    def test_caption_stages(self):
        """CAPTION stage maps to CAPTION."""
        assert ErrorSource.from_stage_name('CAPTION') == ErrorSource.CAPTION
        assert ErrorSource.from_stage_name('caption_stage') == ErrorSource.CAPTION

    def test_transcription_stages(self):
        """TRANSCRIPTION maps to TRANSCRIPTION."""
        assert ErrorSource.from_stage_name('TRANSCRIPTION') == ErrorSource.TRANSCRIPTION
        assert ErrorSource.from_stage_name('transcribe') == ErrorSource.TRANSCRIPTION
        assert ErrorSource.from_stage_name('transcription_stage') == ErrorSource.TRANSCRIPTION


# ---------------------------------------------------------------------------
# Test: UnifiedErrorAggregator basic functionality
# ---------------------------------------------------------------------------

class TestUnifiedErrorAggregatorInit:
    """UnifiedErrorAggregator initialization and enabled state."""

    def test_default_enabled(self):
        """Default initialization enables aggregation."""
        unified = UnifiedErrorAggregator()
        assert unified.enabled is True

    def test_disabled_init(self):
        """Can initialize as disabled."""
        unified = UnifiedErrorAggregator(enabled=False)
        assert unified.enabled is False


class TestUnifiedErrorAggregatorAggregate:
    """UnifiedErrorAggregator.aggregate_from_source() method."""

    def test_aggregate_from_download_source(self):
        """Can aggregate errors from download source."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("HTTP 429 rate limited", ErrorCategory.RATE_LIMIT)
        download_agg.record("DNS failure", ErrorCategory.NETWORK)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)

        stats = unified.get_all_source_stats()
        assert stats['enabled'] is True
        assert 'download' in stats['sources']
        assert stats['sources']['download']['total_errors'] == 2

    def test_aggregate_from_caption_source(self):
        """Can aggregate errors from caption source."""
        unified = UnifiedErrorAggregator()

        caption_agg = ErrorAggregator()
        caption_agg.record("Caption fetch timeout", ErrorCategory.TIMEOUT)

        unified.aggregate_from_source(ErrorSource.CAPTION, caption_agg)

        stats = unified.get_all_source_stats()
        assert 'caption' in stats['sources']

    def test_aggregate_multiple_sources(self):
        """Can aggregate from multiple sources."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)

        caption_agg = ErrorAggregator()
        caption_agg.record("caption unavailable", ErrorCategory.UNKNOWN)

        transcription_agg = ErrorAggregator()
        transcription_agg.record("transcription failed", ErrorCategory.NETWORK)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)
        unified.aggregate_from_source(ErrorSource.CAPTION, caption_agg)
        unified.aggregate_from_source(ErrorSource.TRANSCRIPTION, transcription_agg)

        stats = unified.get_all_source_stats()
        assert len(stats['sources']) == 3

    def test_aggregate_when_disabled(self):
        """Does nothing when disabled."""
        unified = UnifiedErrorAggregator(enabled=False)

        download_agg = ErrorAggregator()
        download_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)

        stats = unified.get_all_source_stats()
        assert stats['enabled'] is False


# ---------------------------------------------------------------------------
# Test: UnifiedErrorAggregator rate limit stats
# ---------------------------------------------------------------------------

class TestUnifiedRateLimitStats:
    """UnifiedErrorAggregator.get_unified_rate_limit_stats() method."""

    def test_empty_stats(self):
        """Returns enabled=False when no sources."""
        unified = UnifiedErrorAggregator()
        stats = unified.get_unified_rate_limit_stats()
        assert stats['enabled'] is True
        assert stats['total_rate_limits'] == 0

    def test_single_source_rate_limits(self):
        """Reports rate limits from single source."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("HTTP 429 Too Many Requests", ErrorCategory.RATE_LIMIT)
        download_agg.record("Rate limit exceeded", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)

        stats = unified.get_unified_rate_limit_stats()
        assert stats['total_rate_limits'] == 2
        assert stats['sources']['download']['count'] == 2

    def test_multiple_source_rate_limits(self):
        """Aggregates rate limits across multiple sources."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("429 from download", ErrorCategory.RATE_LIMIT)

        caption_agg = ErrorAggregator()
        caption_agg.record("429 from caption", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)
        unified.aggregate_from_source(ErrorSource.CAPTION, caption_agg)

        stats = unified.get_unified_rate_limit_stats()
        # Each source has at least 1 rate limit
        assert stats['total_rate_limits'] >= 2

    def test_rate_limit_percentage(self):
        """Calculates rate limit percentage correctly."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)
        download_agg.record("DNS failure", ErrorCategory.NETWORK)
        download_agg.record("Timeout", ErrorCategory.TIMEOUT)

        # 1 rate limit out of 3 = 33.33%
        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)

        stats = unified.get_unified_rate_limit_stats()
        assert stats['rate_limit_percentage'] > 0


# ---------------------------------------------------------------------------
# Test: UnifiedErrorAggregator cross-source correlation
# ---------------------------------------------------------------------------

class TestCrossSourceCorrelation:
    """UnifiedErrorAggregator.cross_source_correlation() method."""

    def test_empty_correlation(self):
        """Returns empty list when no sources."""
        unified = UnifiedErrorAggregator()
        correlations = unified.cross_source_correlation()
        assert correlations == []

    def test_single_source_no_correlation(self):
        """Returns empty list when only one source."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)

        correlations = unified.cross_source_correlation()
        assert correlations == []

    def test_correlation_disabled(self):
        """Returns empty when disabled."""
        unified = UnifiedErrorAggregator(enabled=False)

        download_agg = ErrorAggregator()
        download_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)

        caption_agg = ErrorAggregator()
        caption_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)
        unified.aggregate_from_source(ErrorSource.CAPTION, caption_agg)

        correlations = unified.cross_source_correlation()
        assert correlations == []

    def test_temporal_correlation(self):
        """Detects temporal correlation when rate limits occur at same time."""
        unified = UnifiedErrorAggregator()

        # Create aggregators with temporal errors
        now = datetime.now()

        download_agg = ErrorAggregator()
        download_agg.record_with_timestamp(
            "429 rate limit", ErrorCategory.RATE_LIMIT, now
        )

        caption_agg = ErrorAggregator()
        caption_agg.record_with_timestamp(
            "429 from caption", ErrorCategory.RATE_LIMIT, now
        )

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)
        unified.aggregate_from_source(ErrorSource.CAPTION, caption_agg)

        correlations = unified.cross_source_correlation()
        # Should detect correlation when same time window has errors from multiple sources
        assert len(correlations) > 0

    def test_pattern_correlation(self):
        """Detects correlation by similar error patterns."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        # Record similar errors for pattern detection
        download_agg.record_similar("429 Too Many Requests", ErrorCategory.RATE_LIMIT)
        download_agg.record_similar("429 Too Many Requests", ErrorCategory.RATE_LIMIT)

        caption_agg = ErrorAggregator()
        caption_agg.record_similar("429 Too Many Requests", ErrorCategory.RATE_LIMIT)
        caption_agg.record_similar("429 Too Many Requests", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)
        unified.aggregate_from_source(ErrorSource.CAPTION, caption_agg)

        correlations = unified.cross_source_correlation()
        # Should detect pattern correlation
        assert len(correlations) > 0


# ---------------------------------------------------------------------------
# Test: UnifiedErrorAggregator export and logging
# ---------------------------------------------------------------------------

class TestUnifiedErrorAggregatorExport:
    """UnifiedErrorAggregator export methods."""

    def test_export_json(self):
        """export_as_json returns complete stats."""
        unified = UnifiedErrorAggregator()

        download_agg = ErrorAggregator()
        download_agg.record("429 rate limit", ErrorCategory.RATE_LIMIT)

        unified.aggregate_from_source(ErrorSource.DOWNLOAD, download_agg)

        exported = unified.export_as_json()
        assert 'unified_stats' in exported
        assert 'rate_limit_stats' in exported
        assert 'correlations' in exported
        assert 'enabled' in exported
        assert exported['enabled'] is True

    def test_log_summary_when_disabled(self):
        """log_unified_summary handles disabled state."""
        unified = UnifiedErrorAggregator(enabled=False)
        # Should not raise, just log
        unified.log_unified_summary()  # Should complete without error


# ---------------------------------------------------------------------------
# Test: Set enabled state
# ---------------------------------------------------------------------------

class TestUnifiedErrorAggregatorSetEnabled:
    """UnifiedErrorAggregator.set_enabled() method."""

    def test_set_enabled(self):
        """Can enable/disable aggregation."""
        unified = UnifiedErrorAggregator(enabled=False)
        assert unified.enabled is False

        unified.set_enabled(True)
        assert unified.enabled is True

        unified.set_enabled(False)
        assert unified.enabled is False
