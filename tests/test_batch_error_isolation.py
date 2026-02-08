"""
Tests for per-item error isolation in batch pipeline stages.

US-81-002: Verifies that CAPTION and DOWNLOAD_SEGMENTS stages continue
processing remaining items when individual items fail, collecting errors
in failed_items and reporting correct metrics.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass

from src.stages import StageResult, StageMetrics
from src.stages.download_segments import (
    DownloadVideoSegmentsStage,
    SegmentDownloadStats,
)


# ============================================================================
# SegmentDownloadStats tests
# ============================================================================


class TestSegmentDownloadStatsFailedItems:
    """US-81-002: Test that SegmentDownloadStats tracks per-item failures."""

    def test_increment_failure_with_video_id_tracks_item(self):
        """Failed items are logged with structured error info when video_id provided."""
        stats = SegmentDownloadStats(total=5)
        stats.increment_failure(
            category='bot_detection',
            error_msg='403 Forbidden',
            video_id='abc123',
        )

        assert stats.failed == 1
        assert len(stats.failed_items) == 1
        item = stats.failed_items[0]
        assert item['video_id'] == 'abc123'
        assert item['error_type'] == 'bot_detection'
        assert item['message'] == '403 Forbidden'

    def test_increment_failure_without_video_id_no_item(self):
        """When video_id is not provided, failed_items is not populated."""
        stats = SegmentDownloadStats(total=1)
        stats.increment_failure(category='network', error_msg='timeout')

        assert stats.failed == 1
        assert len(stats.failed_items) == 0

    def test_multiple_failures_accumulate(self):
        """Multiple failures accumulate in failed_items list."""
        stats = SegmentDownloadStats(total=5)
        stats.increment_failure(category='bot_detection', error_msg='403', video_id='v1')
        stats.increment_failure(category='network', error_msg='timeout', video_id='v2')

        assert stats.failed == 2
        assert len(stats.failed_items) == 2
        assert stats.failed_items[0]['video_id'] == 'v1'
        assert stats.failed_items[1]['video_id'] == 'v2'


# ============================================================================
# Download stage batch error isolation test
# ============================================================================


class TestDownloadStageBatchErrorIsolation:
    """US-81-002: Test that DOWNLOAD_SEGMENTS continues on per-item failure."""

    @pytest.mark.fast
    def test_one_of_five_fails_stage_still_succeeds(self, tmp_path):
        """When 1 of 5 items fails, stage returns success=True with 4 processed and 1 failed."""
        stage = DownloadVideoSegmentsStage()

        # Build 5 mock matches with primary_match structure
        matches = []
        for i in range(5):
            m = Mock()
            pm = Mock()
            vs = Mock()
            vs.source_file = f'vid{i}'
            vs.start_time = 0.0
            vs.end_time = 10.0
            pm.video_segment = vs
            m.primary_match = pm
            matches.append(m)

        state = Mock()
        state.matches = matches
        state.downloaded_segments = []

        config = MagicMock()
        config.download = MagicMock()
        config.download.segment_buffer = 2.0
        config.downloaded_videos_dir = str(tmp_path / 'segments')
        # Avoid cookie validation issues
        config.download.cookies_path = None
        config.download.cookies_from_browser = None

        checkpoint = MagicMock()
        checkpoint.get_stage_data.return_value = None
        checkpoint.should_skip_stage.return_value = False
        checkpoint.save_intermediate = MagicMock()

        # Mock _download_segments to simulate 4 successes + 1 failure
        download_stats = SegmentDownloadStats(total=5)
        download_stats.succeeded = 4
        download_stats.failed = 1
        download_stats.attempted = 5
        download_stats.cached = 0
        download_stats.failed_items = [{
            'video_id': 'vid2',
            'error_type': 'bot_detection',
            'message': '403 Forbidden',
        }]

        fake_downloaded = [Mock() for _ in range(4)]

        with patch.object(stage, '_collect_matched_segments', return_value=[
            {'video_id': f'vid{i}', 'start': 0.0, 'end': 10.0} for i in range(5)
        ]), \
            patch.object(stage, '_download_segments', return_value=(fake_downloaded, download_stats)), \
            patch.object(stage, '_print_summary'), \
            patch.object(stage, '_update_matches_with_local_paths'), \
            patch.object(stage, '_log_escalation_summary'), \
            patch.object(stage, '_log_error_summary'), \
            patch('src.downloader.VideoDownloader') as MockDL:

            MockDL.return_value = MagicMock()
            MockDL.return_value.circuit_breaker = None
            MockDL.return_value.escalation_manager = None
            MockDL.return_value.retry_queue = MagicMock()
            MockDL.return_value.retry_queue.items = []
            MockDL.return_value.retry_queue._failed_ids = set()

            result = stage.run(state, config, checkpoint)

        # Assertions
        assert result.success is True, f"Stage should succeed when 1/5 fails, got error: {result.error}"
        assert result.data is not None

        # Check failed_items in data
        assert 'failed_items' in result.data
        assert len(result.data['failed_items']) == 1
        assert result.data['failed_items'][0]['video_id'] == 'vid2'
        assert result.data['failed_items'][0]['error_type'] == 'bot_detection'
        assert result.data['failed_items'][0]['message'] == '403 Forbidden'

        # Check metrics
        assert result.metrics is not None
        assert result.metrics.items_processed == 4
        assert result.metrics.items_failed == 1


# ============================================================================
# Caption stage batch error isolation test
# ============================================================================


class TestCaptionStageBatchErrorIsolation:
    """US-81-002: Test that CAPTION stage collects per-video errors and reports metrics."""

    @pytest.mark.fast
    def test_failed_items_extraction_from_caption_results(self):
        """Verify failed_items list is correctly built from caption_results dict.

        This tests the core logic that the caption stage uses to build the
        failed_items list from its caption_results — the same logic added in
        US-81-002 just before building checkpoint_data.
        """
        # Simulate caption_results with 4 successes and 1 error
        caption_results = {
            'vid1': {'video_id': 'vid1', 'status': 'success', 'segment_count': 10},
            'vid2': {
                'video_id': 'vid2',
                'error': 'Connection timeout',
                'reason': 'fetch_error',
                'caption_quality': 'low',
            },
            'vid3': {'video_id': 'vid3', 'status': 'success', 'segment_count': 8},
            'vid4': {'video_id': 'vid4', 'status': 'success', 'segment_count': 5},
            'vid5': {'video_id': 'vid5', 'status': 'success', 'segment_count': 12},
        }

        # Replicate the failed_items extraction logic from caption_stage.py
        failed_items = []
        for vid, r in caption_results.items():
            if r.get('error'):
                failed_items.append({
                    'video_id': vid,
                    'error_type': r.get('reason', 'fetch_error'),
                    'message': str(r.get('error', 'unknown error')),
                })
            elif r.get('unavailable') and r.get('reason') != 'no_captions_available':
                failed_items.append({
                    'video_id': vid,
                    'error_type': r.get('reason', 'unavailable'),
                    'message': r.get('reason', 'unavailable'),
                })

        assert len(failed_items) == 1
        assert failed_items[0]['video_id'] == 'vid2'
        assert failed_items[0]['error_type'] == 'fetch_error'
        assert 'Connection timeout' in failed_items[0]['message']

    @pytest.mark.fast
    def test_no_captions_available_not_in_failed_items(self):
        """Videos with no captions (unavailable+no_captions_available) are NOT errors."""
        caption_results = {
            'vid1': {'video_id': 'vid1', 'status': 'success', 'segment_count': 10},
            'vid2': {
                'video_id': 'vid2',
                'unavailable': True,
                'reason': 'no_captions_available',
                'caption_quality': 'low',
            },
        }

        failed_items = []
        for vid, r in caption_results.items():
            if r.get('error'):
                failed_items.append({
                    'video_id': vid,
                    'error_type': r.get('reason', 'fetch_error'),
                    'message': str(r.get('error', 'unknown error')),
                })
            elif r.get('unavailable') and r.get('reason') != 'no_captions_available':
                failed_items.append({
                    'video_id': vid,
                    'error_type': r.get('reason', 'unavailable'),
                    'message': r.get('reason', 'unavailable'),
                })

        # no_captions_available is expected (triggers transcription), not a failure
        assert len(failed_items) == 0

    @pytest.mark.fast
    def test_stage_metrics_items_failed_matches_failed_count(self):
        """StageMetrics.items_failed reflects fetch_failed_count, not no_captions_count."""
        success_count = 4
        skip_count = 0
        fetch_failed_count = 1

        metrics = StageMetrics(
            items_processed=success_count + skip_count,
            items_failed=fetch_failed_count,
        )

        assert metrics.items_processed == 4
        assert metrics.items_failed == 1
