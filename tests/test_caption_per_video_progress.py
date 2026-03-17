"""
Tests for per-video progress tracking in caption stage batch processing (US-78-010).

Verifies:
- Progress callback receives correct video_index/total_videos ratio
- Progress includes current video_id and elapsed time
- Slow videos (>30s) trigger warning log
- Progress is TTY-aware compatible
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


@dataclass
class MockCaptionResult:
    """Mock CaptionResult for testing."""
    video_id: str
    language: str = "en"
    caption_quality: str = "medium"
    segments: list = None
    is_auto_generated: bool = False
    format_source: str = "vtt"

    def __post_init__(self):
        if self.segments is None:
            self.segments = []


def _make_mock_fetcher(fetch_side_effect=None):
    """Create a mock fetcher with common attributes."""
    mock_fetcher = MagicMock()
    mock_fetcher._timeout = 30.0
    mock_fetcher._preferred_formats = ["vtt"]
    mock_fetcher._sort_videos_by_channel_success = lambda x, y: x
    if fetch_side_effect:
        mock_fetcher.fetch_captions_auto_language_with_retry.side_effect = fetch_side_effect
    return mock_fetcher


class TestPerVideoProgressCallback:
    """Tests for per-video progress tracking (US-78-010)."""

    def test_progress_callback_invoked_with_correct_video_index_for_3_video_batch(self):
        """Progress callback is invoked with correct video_index for a 3-video batch.

        Verifies AC5: Unit test verifies progress callback is invoked with
        correct video_index for a 3-video batch.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[{"text": "ok"}])

        mock_fetcher = _make_mock_fetcher(mock_fetch)

        # Single worker for deterministic ordering
        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Track all progress events
        progress_events: List[Dict[str, Any]] = []

        def on_progress(video_id: str, status: str, details: Dict[str, Any]) -> None:
            progress_events.append({
                'video_id': video_id,
                'status': status,
                'index': details.get('index'),
                'total': details.get('total'),
            })

        videos = ["vid_A", "vid_B", "vid_C"]
        result = processor.process(videos, progress_callback=on_progress)

        assert result.success_count == 3

        # Filter to fetching and success events (per-video progress)
        fetching_events = [e for e in progress_events if e['status'] == 'fetching']
        success_events = [e for e in progress_events if e['status'] == 'success']

        # Each of the 3 videos should get a fetching event
        assert len(fetching_events) == 3
        # Each should get a success event
        assert len(success_events) == 3

        # Verify index/total ratio for fetching events (1-based index)
        fetching_indices = sorted(e['index'] for e in fetching_events)
        assert fetching_indices == [1, 2, 3]
        assert all(e['total'] == 3 for e in fetching_events)

        # Verify index/total ratio for success events
        success_indices = sorted(e['index'] for e in success_events)
        assert success_indices == [1, 2, 3]
        assert all(e['total'] == 3 for e in success_events)

        # Verify video_ids are present in events
        fetching_video_ids = set(e['video_id'] for e in fetching_events)
        assert fetching_video_ids == {"vid_A", "vid_B", "vid_C"}

    def test_progress_includes_video_id_and_elapsed_time(self):
        """Progress includes current video_id and elapsed time (AC2).

        Success/failed events include elapsed_seconds.
        Fetching events include start_time for computing elapsed.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        def mock_fetch(video_id, preferred_language=None):
            time.sleep(0.01)  # Small delay to ensure elapsed > 0
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher = _make_mock_fetcher(mock_fetch)

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        progress_events: List[Dict[str, Any]] = []

        def on_progress(video_id: str, status: str, details: Dict[str, Any]) -> None:
            progress_events.append({
                'video_id': video_id,
                'status': status,
                'details': details.copy(),
            })

        processor.process(["vid1"], progress_callback=on_progress)

        # Check fetching event has start_time
        fetching = [e for e in progress_events if e['status'] == 'fetching']
        assert len(fetching) == 1
        assert fetching[0]['video_id'] == "vid1"
        assert 'start_time' in fetching[0]['details']

        # Check success event has elapsed_seconds
        success = [e for e in progress_events if e['status'] == 'success']
        assert len(success) == 1
        assert success[0]['video_id'] == "vid1"
        assert 'elapsed_seconds' in success[0]['details']
        assert success[0]['details']['elapsed_seconds'] > 0

    def test_slow_video_warning_emitted_over_30s(self, caplog):
        """Slow videos (>30s processing time) trigger warning log (AC3).

        Uses a mock that simulates a slow fetch to verify the warning is emitted.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        caplog.set_level(logging.WARNING)

        def mock_slow_fetch(video_id, preferred_language=None):
            # We can't actually wait 30s in a test, but we can verify the slow_video
            # progress callback is emitted by checking the elapsed time path.
            # Instead, mock the perf_counter to simulate elapsed time.
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher = _make_mock_fetcher(mock_slow_fetch)

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        # Track slow_video events
        slow_events: List[Dict[str, Any]] = []

        def on_progress(video_id: str, status: str, details: Dict[str, Any]) -> None:
            if status == 'slow_video':
                slow_events.append({
                    'video_id': video_id,
                    'elapsed_seconds': details.get('elapsed_seconds'),
                })

        # Use time mock to simulate 35s elapsed
        import unittest.mock
        original_perf_counter = time.perf_counter

        call_count = [0]
        base_time = [original_perf_counter()]

        def mock_perf_counter():
            # First call (start_time) returns base, second call (elapsed) returns base+35
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return base_time[0] + 35.0  # 35 seconds later
            else:
                base_time[0] = original_perf_counter()
                return base_time[0]

        with unittest.mock.patch('src.caption.batch_processor.time') as mock_time:
            mock_time.perf_counter = mock_perf_counter

            processor.process(["slow_vid"], progress_callback=on_progress)

        # Should have emitted slow_video progress
        assert len(slow_events) == 1
        assert slow_events[0]['video_id'] == "slow_vid"
        assert slow_events[0]['elapsed_seconds'] == 35.0

        # Should also have logged a warning
        slow_warnings = [
            r for r in caplog.records
            if "Slow caption fetch" in r.message and "slow_vid" in r.message
        ]
        assert len(slow_warnings) == 1

    def test_progress_ratio_computed_correctly(self):
        """video_index/total_videos ratio is correct (AC1).

        For a batch of 5 videos, index goes 1-5 and total stays 5.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher = _make_mock_fetcher(mock_fetch)

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        ratios: List[float] = []

        def on_progress(video_id: str, status: str, details: Dict[str, Any]) -> None:
            if status == 'success':
                idx = details.get('index', 0)
                total = details.get('total', 1)
                ratios.append(idx / total)

        videos = [f"v{i}" for i in range(5)]
        processor.process(videos, progress_callback=on_progress)

        # Ratios should be 1/5, 2/5, 3/5, 4/5, 5/5
        assert len(ratios) == 5
        assert sorted(ratios) == [0.2, 0.4, 0.6, 0.8, 1.0]

    def test_slow_video_threshold_is_30_seconds(self):
        """Slow video progress notification uses 30s threshold (AC3).

        Videos completing in <30s should NOT get slow_video events.
        Videos completing in >30s SHOULD get slow_video events.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig
        import unittest.mock

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(video_id=video_id, segments=[])

        mock_fetcher = _make_mock_fetcher(mock_fetch)

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        slow_events: List[str] = []

        def on_progress(video_id: str, status: str, details: Dict[str, Any]) -> None:
            if status == 'slow_video':
                slow_events.append(video_id)

        # Simulate: vid_fast finishes in 5s, vid_slow finishes in 40s
        original_perf_counter = time.perf_counter
        call_idx = [0]
        # Two videos, each gets 2 perf_counter calls (start, end)
        elapsed_values = [5.0, 40.0]  # fast, slow

        def mock_perf_counter():
            video_idx = call_idx[0] // 2
            is_end = call_idx[0] % 2 == 1
            call_idx[0] += 1
            base = 1000.0 + video_idx * 100
            if is_end:
                return base + elapsed_values[video_idx]
            return base

        with unittest.mock.patch('src.caption.batch_processor.time') as mock_time:
            mock_time.perf_counter = mock_perf_counter

            processor.process(["vid_fast", "vid_slow"], progress_callback=on_progress)

        # Only vid_slow should trigger slow_video
        assert "vid_fast" not in slow_events
        assert "vid_slow" in slow_events

    def test_progress_callback_compatible_with_tty_output(self):
        """Progress details are structured for TTY-aware display (AC4).

        Verifies the callback provides all fields needed by the TTY output formatter.
        """
        from src.caption.batch_processor import BatchProcessor, BatchProcessorConfig

        def mock_fetch(video_id, preferred_language=None):
            return MockCaptionResult(
                video_id=video_id,
                language="en",
                caption_quality="medium",
                segments=[{"text": "test"}],
                is_auto_generated=True,
                format_source="vtt",
            )

        mock_fetcher = _make_mock_fetcher(mock_fetch)

        config = BatchProcessorConfig(max_workers=1)
        processor = BatchProcessor(mock_fetcher, config=config)

        all_events: List[Dict[str, Any]] = []

        def on_progress(video_id: str, status: str, details: Dict[str, Any]) -> None:
            all_events.append({'video_id': video_id, 'status': status, 'details': details})

        processor.process(["vid1"], progress_callback=on_progress)

        # Fetching event should have index, total for TTY display
        fetching = [e for e in all_events if e['status'] == 'fetching'][0]
        assert 'index' in fetching['details']
        assert 'total' in fetching['details']

        # Success event should have all TTY display fields
        success = [e for e in all_events if e['status'] == 'success'][0]
        assert 'index' in success['details']
        assert 'total' in success['details']
        assert 'language' in success['details']
        assert 'quality' in success['details']
        assert 'segment_count' in success['details']
        assert 'is_auto_generated' in success['details']
        assert 'elapsed_seconds' in success['details']
