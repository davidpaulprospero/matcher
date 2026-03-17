"""Tests for US-49-009: download_segments error summary with actionable diagnostics.

Tests cover:
  - Error summary is generated with correct per-category counts
  - Bot-detection guidance logged when >50% failures are bot-detection
  - No guidance logged when bot-detection is <=50% of failures
  - Error categories are included in StageMetrics
  - No summary logged when there are zero failures
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import (
    DownloadVideoSegmentsStage,
    SegmentDownloadStats,
    classify_error_category,
)
from src.stages import StageMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_downloader():
    """Create a mock downloader with required attributes."""
    mock_dl = MagicMock()
    mock_dl.escalation_manager = None
    mock_dl.circuit_breaker = None
    mock_dl.cookie_rotator = None
    mock_dl.retry_queue = MagicMock()
    mock_dl.retry_queue.has_pending.return_value = False
    mock_dl.impersonation_manager = None

    mock_dl.download_config = MagicMock()
    mock_dl.download_config.bot_detection_tier_floor_threshold = 5
    mock_dl.download_config.bot_detection_abort_threshold = 0  # disable abort
    mock_dl.download_config.network_failure_threshold = 3
    mock_dl.download_config.segment_socket_timeout = 30
    mock_dl.download_config.segment_max_resolution = 1080
    mock_dl.download_config.segment_format = 'best[height<={segment_max_resolution}]'
    mock_dl.download_config.segment_stall_timeout = 0
    mock_dl.download_config.cookies_from_browser = ''
    mock_dl.download_config.cookies_path = ''
    mock_dl.download_config.cookie_rotation = None
    mock_dl.download_config.ffmpeg_location = ''
    return mock_dl


def _make_segments(count):
    """Generate a list of fake segments."""
    return [
        {'video_id': f'vid_{i}', 'start': 10.0, 'end': 20.0}
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Test: classify_error_category granular categories
# ---------------------------------------------------------------------------

class TestClassifyErrorCategoryGranular:
    """US-49-009: Verify granular error categories."""

    def test_network_failure_returns_network(self):
        assert classify_error_category("getaddrinfo failed") == 'network'

    def test_bot_detection_returns_bot_detection(self):
        assert classify_error_category("HTTP Error 403: Forbidden") == 'bot_detection'

    def test_sign_in_bot_detection(self):
        assert classify_error_category("Sign in to confirm you're not a bot") == 'bot_detection'

    def test_timeout_returns_timeout(self):
        assert classify_error_category("ydl.download() stalled for 120s") == 'timeout'

    def test_connection_timed_out_returns_network(self):
        """US-51-005: 'Connection timed out' is now classified as network (systemic)."""
        assert classify_error_category("Connection timed out") == 'network'

    def test_video_unavailable_returns_video_specific(self):
        assert classify_error_category("Video unavailable") == 'video_specific'

    def test_empty_string_returns_video_specific(self):
        assert classify_error_category('') == 'video_specific'


# ---------------------------------------------------------------------------
# Test: Error summary with category counts
# ---------------------------------------------------------------------------

class TestErrorSummary:
    """US-49-009: Verify error summary logging and category breakdown."""

    def test_error_summary_logs_category_counts(self, caplog):
        """Error summary includes per-category counts in log output."""
        stage = DownloadVideoSegmentsStage()
        stats = SegmentDownloadStats(
            total=10,
            succeeded=5,
            failed=4,
            cached=1,
            attempted=10,
            error_categories={
                'bot_detection': 3,
                'video_specific': 1,
            },
        )
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_error_summary(stats)

        assert any('bot_detection=3' in r.message for r in caplog.records)
        assert any('video_specific=1' in r.message for r in caplog.records)
        assert any('failed=4' in r.message for r in caplog.records)

    def test_no_summary_when_zero_failures(self, caplog):
        """No error summary logged when there are zero failures."""
        stage = DownloadVideoSegmentsStage()
        stats = SegmentDownloadStats(
            total=5,
            succeeded=4,
            failed=0,
            cached=1,
            attempted=5,
        )
        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_error_summary(stats)

        # Should not log anything
        assert not any('error summary' in r.message.lower() for r in caplog.records)

    def test_bot_detection_guidance_when_majority(self, caplog):
        """Log cookie guidance when >50% of failures are bot-detection."""
        stage = DownloadVideoSegmentsStage()
        stats = SegmentDownloadStats(
            total=10,
            succeeded=3,
            failed=7,
            cached=0,
            attempted=10,
            error_categories={
                'bot_detection': 5,  # 5/7 = 71% > 50%
                'video_specific': 2,
            },
        )
        with caplog.at_level(logging.WARNING, logger='src.stages.download_segments'):
            stage._log_error_summary(stats)

        assert any(
            'Most failures are bot-detection' in r.message
            for r in caplog.records
        )
        assert any(
            'cookie configuration' in r.message.lower()
            for r in caplog.records
        )

    def test_no_bot_guidance_when_minority(self, caplog):
        """No cookie guidance when bot-detection is <=50% of failures."""
        stage = DownloadVideoSegmentsStage()
        stats = SegmentDownloadStats(
            total=10,
            succeeded=4,
            failed=6,
            cached=0,
            attempted=10,
            error_categories={
                'bot_detection': 2,  # 2/6 = 33% < 50%
                'network': 3,
                'video_specific': 1,
            },
        )
        with caplog.at_level(logging.WARNING, logger='src.stages.download_segments'):
            stage._log_error_summary(stats)

        assert not any(
            'Most failures are bot-detection' in r.message
            for r in caplog.records
        )

    def test_error_categories_in_stats_after_download(self):
        """Error categories are tracked in stats dict during _download_segments."""
        stage = DownloadVideoSegmentsStage()
        stage.downloader = _make_mock_downloader()

        segments = _make_segments(3)
        output_dir = Path('/tmp/test_segments')

        # All downloads raise 403 errors
        with patch('yt_dlp.YoutubeDL') as mock_ydl_cls:
            mock_ydl_cls.return_value.__enter__ = MagicMock(
                side_effect=Exception("HTTP Error 403: Forbidden")
            )
            mock_ydl_cls.side_effect = Exception("HTTP Error 403: Forbidden")
            with patch.object(Path, 'exists', return_value=False):
                _, stats = stage._download_segments(segments, output_dir, 2.0, None)

        assert stats.error_categories.get('bot_detection', 0) == 3
        assert stats.failed == 3


# ---------------------------------------------------------------------------
# Test: StageMetrics includes error_categories
# ---------------------------------------------------------------------------

class TestStageMetricsErrorCategories:
    """US-49-009: StageMetrics serializes error_categories."""

    def test_to_dict_includes_error_categories(self):
        metrics = StageMetrics(
            items_processed=5,
            items_failed=3,
            duration_seconds=10.0,
            error_categories={'bot_detection': 2, 'network': 1},
        )
        d = metrics.to_dict()
        assert d['error_categories'] == {'bot_detection': 2, 'network': 1}

    def test_to_dict_omits_empty_error_categories(self):
        metrics = StageMetrics(items_processed=5, items_failed=0)
        d = metrics.to_dict()
        assert 'error_categories' not in d

    def test_from_dict_restores_error_categories(self):
        data = {
            'items_processed': 5,
            'items_failed': 3,
            'duration_seconds': 10.0,
            'error_categories': {'bot_detection': 2, 'timeout': 1},
        }
        metrics = StageMetrics.from_dict(data)
        assert metrics.error_categories == {'bot_detection': 2, 'timeout': 1}

    def test_from_dict_defaults_empty_when_missing(self):
        """Old checkpoints without error_categories default to empty dict."""
        data = {
            'items_processed': 5,
            'items_failed': 0,
            'duration_seconds': 10.0,
        }
        metrics = StageMetrics.from_dict(data)
        assert metrics.error_categories == {}
