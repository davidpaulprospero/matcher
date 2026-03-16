"""Tests for caption coverage gap summary in CaptionStage (US-78-005).

Verifies _log_coverage_gap_summary produces correct structured summary
with varying coverage ratios, gap sizes, and edge cases.
"""

import logging
from unittest.mock import MagicMock

import pytest

from src.stages.caption_stage import CaptionStage


def _make_caption_result(video_id, segments_timings, video_duration):
    """Build a caption result dict with segment dicts from (start, end) tuples."""
    segments = [
        {
            'index': i,
            'start_time': s,
            'end_time': e,
            'text': f'text {i}',
        }
        for i, (s, e) in enumerate(segments_timings)
    ]
    return {
        'video_id': video_id,
        'segments': segments,
        'segment_count': len(segments),
        'video_duration': video_duration,
        'coverage_ratio': None,  # Will be computed by analyze_caption_coverage
    }


class TestCoverageGapSummary:
    """Tests for CaptionStage._log_coverage_gap_summary()."""

    def setup_method(self):
        self.stage = CaptionStage()

    def test_summary_with_varying_coverage(self, caplog):
        """Summary includes total, captioned count, avg coverage, and gaps >10s."""
        caption_results = {
            'vid_high': _make_caption_result(
                'vid_high',
                [(0.0, 30.0), (30.0, 60.0)],  # Full coverage
                video_duration=60.0,
            ),
            'vid_medium': _make_caption_result(
                'vid_medium',
                [(0.0, 10.0), (30.0, 50.0)],  # 20s/60s covered, 20s gap
                video_duration=60.0,
            ),
            'vid_low': _make_caption_result(
                'vid_low',
                [(0.0, 5.0)],  # 5s/60s covered, 55s trailing gap
                video_duration=60.0,
            ),
        }
        video_durations = {
            'vid_high': 60.0,
            'vid_medium': 60.0,
            'vid_low': 60.0,
        }

        with caplog.at_level(logging.INFO):
            self.stage._log_coverage_gap_summary(caption_results, video_durations)

        # Find the structured summary log
        summary_logs = [r for r in caplog.records if 'caption_coverage_gap_summary' in r.message]
        assert len(summary_logs) == 1

        msg = summary_logs[0].message
        # All 3 videos should be counted
        assert "'total_videos': 3" in msg
        # All have some coverage
        assert "'captioned_count': 3" in msg
        # vid_medium (20s gap) and vid_low (55s gap) have gaps >10s
        assert "'videos_with_gaps_over_10s': 2" in msg

    def test_top_3_largest_gaps(self, caplog):
        """Top 3 largest gaps are reported with video_id and duration."""
        caption_results = {}
        video_durations = {}

        # Create 5 videos with different gap sizes
        gaps = [5.0, 15.0, 25.0, 35.0, 45.0]
        for i, gap_size in enumerate(gaps):
            vid = f'vid_{i}'
            # Each video: caption from 0 to 10, then gap, then video ends
            duration = 10.0 + gap_size
            caption_results[vid] = _make_caption_result(
                vid,
                [(0.0, 10.0)],
                video_duration=duration,
            )
            video_durations[vid] = duration

        with caplog.at_level(logging.INFO):
            self.stage._log_coverage_gap_summary(caption_results, video_durations)

        summary_logs = [r for r in caplog.records if 'caption_coverage_gap_summary' in r.message]
        assert len(summary_logs) == 1
        msg = summary_logs[0].message

        # Top 3 should be the largest gaps (45s, 35s, 25s)
        assert "'gap_seconds': 45.0" in msg
        assert "'gap_seconds': 35.0" in msg
        assert "'gap_seconds': 25.0" in msg
        # The 15s and 5s gaps should NOT be in top 3
        assert "'gap_seconds': 15.0" not in msg
        assert "'gap_seconds': 5.0" not in msg

    def test_no_segments_produces_no_summary(self, caplog):
        """Videos without segments produce a 'no data' log."""
        caption_results = {
            'vid_empty': {
                'video_id': 'vid_empty',
                'segments': [],
                'video_duration': 60.0,
            },
            'vid_error': {
                'video_id': 'vid_error',
                'unavailable': True,
                'reason': 'no_captions_available',
            },
        }

        with caplog.at_level(logging.INFO):
            self.stage._log_coverage_gap_summary(caption_results, {})

        summary_logs = [r for r in caplog.records if 'caption_coverage_gap_summary' in r.message]
        assert len(summary_logs) == 1
        assert 'no videos with coverage data' in summary_logs[0].message

    def test_duration_from_video_durations_dict(self, caplog):
        """Fallback to video_durations dict when result has no video_duration."""
        caption_results = {
            'vid_a': {
                'video_id': 'vid_a',
                'segments': [
                    {'index': 0, 'start_time': 0.0, 'end_time': 20.0, 'text': 'hello'},
                ],
                'segment_count': 1,
                # No video_duration in result
            },
        }
        video_durations = {'vid_a': 30.0}

        with caplog.at_level(logging.INFO):
            self.stage._log_coverage_gap_summary(caption_results, video_durations)

        summary_logs = [r for r in caplog.records if 'caption_coverage_gap_summary' in r.message]
        assert len(summary_logs) == 1
        msg = summary_logs[0].message
        assert "'total_videos': 1" in msg
        # 20/30 = 0.667 coverage
        assert "'avg_coverage_ratio': 0.667" in msg

    def test_logged_at_info_level(self, caplog):
        """Summary is logged at INFO level, not DEBUG."""
        caption_results = {
            'vid_a': _make_caption_result('vid_a', [(0.0, 50.0)], 60.0),
        }

        with caplog.at_level(logging.INFO):
            self.stage._log_coverage_gap_summary(caption_results, {'vid_a': 60.0})

        summary_logs = [r for r in caplog.records if 'caption_coverage_gap_summary' in r.message]
        assert len(summary_logs) == 1
        assert summary_logs[0].levelno == logging.INFO
