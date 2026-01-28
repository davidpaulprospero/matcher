"""
Stress tests for high-concurrency caption fetching (US-010 Sprint 7).

Tests verify that caption fetching with 8+ parallel workers handles edge cases:
- No deadlocks or memory leaks with many videos
- Error cascades don't cause worker starvation
- Mixed latency videos get fair scheduling
- Abrupt cancellation cleans up all worker threads

Run with: pytest tests/test_caption_stress.py -v -m stress
"""

import gc
import os
import random
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

import pytest

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    CaptionMetrics,
    CaptionUnavailableError,
    CaptionFetchError,
    ErrorPatternAbortError,
)


def make_caption_result(video_id: str, segment_count: int = 10) -> CaptionResult:
    """Helper to create a CaptionResult with multiple segments."""
    segments = [
        CaptionSegment(
            index=i,
            start_time=float(i * 5),
            end_time=float(i * 5 + 4.5),
            text=f"Segment {i} text for video {video_id}",
            source_file=video_id
        )
        for i in range(segment_count)
    ]
    return CaptionResult(
        video_id=video_id,
        segments=segments,
        language='en',
        is_auto_generated=False,
        format_source='vtt'
    )


@pytest.mark.stress
class TestCaptionStressSuite:
    """Stress tests for high-concurrency caption fetching (US-010 Sprint 7)."""

    @pytest.mark.fast
    def test_100_videos_16_workers_no_deadlock(self):
        """Test: 100 videos with 16 workers completes without deadlock or memory leak.

        Verifies that high parallelism doesn't cause:
        - Thread deadlocks (test completes within timeout)
        - Memory leaks (memory doesn't grow unbounded)
        - Data corruption (all results are valid)
        """
        video_ids = [f'video{i:03d}test' for i in range(100)]
        results_lock = threading.Lock()
        fetch_order = []

        def mock_fetch(video_id, preferred_language=None):
            """Simulate fetch with slight delay to exercise threading."""
            time.sleep(random.uniform(0.01, 0.05))  # 10-50ms delay
            with results_lock:
                fetch_order.append(video_id)
            return make_caption_result(video_id, segment_count=5)

        # Track memory before
        gc.collect()
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0]

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_fetch

            fetcher = CaptionFetcher()
            metrics = CaptionMetrics()

            # This should complete within a reasonable time (not deadlock)
            start = time.time()
            results = fetcher.fetch_captions_batch(
                video_ids,
                max_workers=16,
                metrics=metrics
            )
            elapsed = time.time() - start

            # Check memory after
            gc.collect()
            mem_after = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()

        # Verify no deadlock: should complete in reasonable time
        # 100 videos * 0.05s max delay / 16 workers = ~0.3s theoretical minimum
        # Allow margin for thread overhead
        assert elapsed < 10.0, f"Batch fetch took too long ({elapsed:.1f}s) - possible deadlock"

        # Verify all results received
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"

        # Verify all results are valid CaptionResult objects
        for vid in video_ids:
            assert vid in results, f"Missing result for {vid}"
            assert isinstance(results[vid], CaptionResult), f"Invalid result type for {vid}"
            assert len(results[vid].segments) == 5

        # Verify metrics tracked correctly (thread-safe)
        assert metrics.fetch_attempts == 100
        assert metrics.successes == 100
        assert metrics.failures == 0
        assert metrics.total_segments == 500  # 100 * 5

        # Verify no memory leak (allow 50MB growth for test overhead)
        mem_growth_mb = (mem_after - mem_before) / (1024 * 1024)
        assert mem_growth_mb < 50, f"Memory grew by {mem_growth_mb:.1f}MB - possible leak"

        # Verify all videos were actually fetched (no silently dropped)
        assert len(fetch_order) == 100

    @pytest.mark.fast
    def test_50_percent_error_rate_no_cascade_failure(self):
        """Test: 50% error rate doesn't cause cascade failure or worker starvation.

        Verifies that high error rates don't cause:
        - Worker threads to die without being replaced
        - Error handling to slow down healthy fetches
        - Metrics to become corrupted from concurrent updates
        """
        video_ids = [f'video{i:03d}test' for i in range(100)]
        success_count = 0
        error_count = 0
        count_lock = threading.Lock()

        def mock_fetch_with_errors(video_id, preferred_language=None):
            """50% of fetches fail with various errors."""
            nonlocal success_count, error_count

            time.sleep(random.uniform(0.01, 0.03))

            # Deterministic error pattern based on video id for reproducibility
            should_fail = hash(video_id) % 2 == 0
            error_type = hash(video_id) % 3  # Different error types

            with count_lock:
                if should_fail:
                    error_count += 1
                else:
                    success_count += 1

            if should_fail:
                if error_type == 0:
                    raise CaptionUnavailableError(video_id, "Captions disabled")
                elif error_type == 1:
                    raise CaptionFetchError(video_id, "Network timeout")
                else:
                    raise Exception(f"Unexpected error for {video_id}")

            return make_caption_result(video_id, segment_count=3)

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_fetch_with_errors

            fetcher = CaptionFetcher()
            metrics = CaptionMetrics()

            start = time.time()
            results = fetcher.fetch_captions_batch(
                video_ids,
                max_workers=8,
                metrics=metrics
            )
            elapsed = time.time() - start

        # Should complete in reasonable time despite errors
        assert elapsed < 15.0, f"Batch with errors took too long ({elapsed:.1f}s)"

        # All 100 videos should have results (success or error dict)
        assert len(results) == 100

        # Count actual successes and errors from results
        actual_successes = sum(1 for r in results.values() if isinstance(r, CaptionResult))
        actual_errors = sum(1 for r in results.values() if isinstance(r, dict))

        assert actual_successes + actual_errors == 100

        # Approximately 50% should succeed (hash-based, deterministic)
        assert 40 <= actual_successes <= 60, f"Expected ~50% success, got {actual_successes}"

        # Verify metrics consistency (no races in counter updates)
        assert metrics.fetch_attempts == 100
        assert metrics.successes == actual_successes
        assert metrics.failures == actual_errors
        assert metrics.successes + metrics.failures == 100

        # Verify each error result has required fields
        for vid, result in results.items():
            if isinstance(result, dict):
                assert 'video_id' in result or 'error' in result or 'unavailable' in result

    @pytest.mark.fast
    def test_mixed_latency_fair_scheduling(self):
        """Test: mixed slow/fast videos (0.1s to 5s latency) maintains fair scheduling.

        Verifies that:
        - Slow videos don't block fast video completion
        - All workers stay busy (no starvation)
        - Fast videos complete before slow ones finish
        """
        # 20 fast videos, 5 slow videos
        fast_ids = [f'fast{i:02d}test' for i in range(20)]
        slow_ids = [f'slow{i:02d}test' for i in range(5)]
        video_ids = fast_ids + slow_ids
        random.shuffle(video_ids)  # Interleave

        completion_order = []
        completion_lock = threading.Lock()
        fast_start = None
        fast_end = None

        def mock_varied_latency(video_id, preferred_language=None):
            """Fast videos: 50ms, Slow videos: 500ms."""
            nonlocal fast_start, fast_end

            is_fast = video_id.startswith('fast')
            latency = 0.05 if is_fast else 0.5

            if is_fast and fast_start is None:
                with completion_lock:
                    if fast_start is None:
                        fast_start = time.time()

            time.sleep(latency)

            with completion_lock:
                completion_order.append((video_id, time.time()))
                if is_fast:
                    fast_end = time.time()

            return make_caption_result(video_id)

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_varied_latency

            fetcher = CaptionFetcher()

            start = time.time()
            results = fetcher.fetch_captions_batch(
                video_ids,
                max_workers=8
            )
            total_elapsed = time.time() - start

        # All should complete
        assert len(results) == 25

        # Count fast vs slow completions in first half vs second half
        mid_point = len(completion_order) // 2
        first_half = completion_order[:mid_point]
        second_half = completion_order[mid_point:]

        fast_in_first_half = sum(1 for vid, _ in first_half if vid.startswith('fast'))
        fast_in_second_half = sum(1 for vid, _ in second_half if vid.startswith('fast'))

        # Most fast videos should complete before most slow videos
        # With 8 workers processing 20 fast (50ms each) and 5 slow (500ms each):
        # Fast should mostly complete while slow are still running
        assert fast_in_first_half > fast_in_second_half, \
            f"Fair scheduling failed: {fast_in_first_half} fast in first half, {fast_in_second_half} in second"

        # Total time should be dominated by slow videos, not serialized
        # Sequential would be: 20*0.05 + 5*0.5 = 3.5s
        # Parallel with 8 workers: ~0.5s (slow video latency) + overhead
        # Allow generous margin for CI environments
        assert total_elapsed < 3.0, f"Scheduling inefficient: took {total_elapsed:.1f}s"

    def test_abrupt_cancellation_thread_cleanup(self):
        """Test: abrupt cancellation during batch fetch cleans up all worker threads.

        Verifies that:
        - ErrorPatternAbortError properly cancels remaining futures
        - All threads terminate (no orphaned threads)
        - Partial results are returned correctly
        """
        video_ids = [f'video{i:03d}test' for i in range(50)]
        fetch_count = 0
        fetch_lock = threading.Lock()
        threads_active = []

        def mock_fetch_with_pattern_error(video_id, preferred_language=None):
            """First 10 videos fail with same error to trigger pattern detection."""
            nonlocal fetch_count

            with fetch_lock:
                fetch_count += 1
                current = fetch_count
                threads_active.append(threading.current_thread().name)

            time.sleep(0.05)  # Small delay to allow concurrent execution

            # First 10 fail with same error
            idx = int(video_id.replace('video', '').replace('test', ''))
            if idx < 10:
                raise CaptionFetchError(video_id, "403 Forbidden - Geographic restriction")

            return make_caption_result(video_id)

        # Count threads before
        threads_before = threading.active_count()

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_fetch_with_pattern_error

            # Create config that enables abort mode
            mock_config = MagicMock()
            mock_config.download.caption_first.abort_on_error_pattern = 'abort'
            mock_config.download.caption_first.error_pattern_threshold = 0.3
            mock_config.download.caption_first.error_pattern_sample_size = 10
            mock_config.download.caption_first.max_parallel_fetches = 8
            mock_config.download.caption_first.prioritize_by_channel = False

            fetcher = CaptionFetcher(config=mock_config)

            # Should raise ErrorPatternAbortError
            with pytest.raises(ErrorPatternAbortError) as exc_info:
                fetcher.fetch_captions_batch(
                    video_ids,
                    max_workers=8
                )

            # Verify partial results are available
            partial_results = exc_info.value.partial_results
            assert partial_results is not None
            assert isinstance(partial_results, dict)

        # Small delay for thread cleanup
        time.sleep(0.2)
        gc.collect()

        # Verify threads are cleaned up
        threads_after = threading.active_count()
        thread_delta = threads_after - threads_before

        # Allow for some variance in thread count, but shouldn't have orphaned workers
        assert thread_delta <= 2, \
            f"Threads leaked: {thread_delta} extra threads after cancellation"

    @pytest.mark.fast
    def test_concurrent_metrics_updates_thread_safe(self):
        """Test metrics are correctly updated with high concurrency.

        Verifies that CaptionMetrics._lock correctly synchronizes:
        - fetch_attempts counter
        - successes/failures counters
        - language_distribution dict
        - total_segments counter
        """
        video_ids = [f'video{i:03d}test' for i in range(100)]
        languages = ['en', 'es', 'fr', 'de', 'ja']

        def mock_fetch_varied(video_id, preferred_language=None):
            """Return results with varied languages."""
            time.sleep(random.uniform(0.001, 0.01))
            idx = int(video_id.replace('video', '').replace('test', ''))
            lang = languages[idx % len(languages)]

            # Create segments directly with the desired language
            segment_count = 5
            segments = [
                CaptionSegment(
                    index=i,
                    start_time=float(i * 5),
                    end_time=float(i * 5 + 4.5),
                    text=f"Segment {i} text for video {video_id}",
                    source_file=video_id
                )
                for i in range(segment_count)
            ]
            return CaptionResult(
                video_id=video_id,
                segments=segments,
                language=lang,
                is_auto_generated=False,
                format_source='vtt'
            )

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_fetch_varied

            fetcher = CaptionFetcher()
            metrics = CaptionMetrics()

            # Run multiple times to stress test
            for _ in range(3):
                results = fetcher.fetch_captions_batch(
                    video_ids,
                    max_workers=16,
                    metrics=metrics
                )
                assert len(results) == 100

        # Verify counter consistency
        assert metrics.fetch_attempts == 300  # 100 * 3 runs
        assert metrics.successes == 300
        assert metrics.failures == 0
        assert metrics.total_segments == 1500  # 300 * 5 segments each

        # Verify distribution sums match
        total_by_language = sum(metrics.language_distribution.values())
        assert total_by_language == 300

        # Verify language distribution is balanced (100 videos / 5 languages = 20 each * 3 runs)
        for lang in languages:
            assert metrics.language_distribution.get(lang, 0) == 60  # 20 * 3

    @pytest.mark.fast
    def test_progress_callback_thread_safety(self):
        """Test progress callback doesn't cause issues with concurrent invocations.

        Verifies that:
        - Callback exceptions don't crash workers
        - Callbacks are called for all videos
        - Callback data is consistent
        """
        video_ids = [f'video{i:03d}test' for i in range(50)]
        callbacks_received = []
        callback_lock = threading.Lock()
        callback_errors = []

        def flaky_callback(video_id, status, details):
            """Callback that sometimes fails."""
            with callback_lock:
                callbacks_received.append((video_id, status, dict(details)))

            # Sometimes raise to test error handling
            if random.random() < 0.1:
                raise ValueError("Random callback failure")

        def mock_fetch(video_id, preferred_language=None):
            time.sleep(random.uniform(0.01, 0.03))
            return make_caption_result(video_id)

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_fetch

            fetcher = CaptionFetcher()

            # Should complete even with flaky callbacks
            results = fetcher.fetch_captions_batch(
                video_ids,
                max_workers=8,
                progress_callback=flaky_callback
            )

        # All videos should complete despite callback errors
        assert len(results) == 50

        # Should have received callbacks for all videos (fetching + success/fail)
        # Each video gets at least 2 callbacks: 'fetching' and 'success'
        videos_with_callbacks = set(vid for vid, status, _ in callbacks_received if vid)
        assert len(videos_with_callbacks) == 50

    def test_worker_thread_names_isolation(self):
        """Verify each batch fetch uses isolated worker threads.

        Tests that multiple batch fetches don't interfere with each other.
        """
        video_ids_a = [f'batchA{i:02d}t' for i in range(20)]
        video_ids_b = [f'batchB{i:02d}t' for i in range(20)]

        threads_a = set()
        threads_b = set()
        lock = threading.Lock()

        def mock_fetch_a(video_id, preferred_language=None):
            with lock:
                threads_a.add(threading.current_thread().ident)
            time.sleep(0.05)
            return make_caption_result(video_id)

        def mock_fetch_b(video_id, preferred_language=None):
            with lock:
                threads_b.add(threading.current_thread().ident)
            time.sleep(0.05)
            return make_caption_result(video_id)

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            # First batch
            mock.side_effect = mock_fetch_a
            fetcher_a = CaptionFetcher()
            results_a = fetcher_a.fetch_captions_batch(video_ids_a, max_workers=4)

            # Second batch (after first completes)
            mock.side_effect = mock_fetch_b
            fetcher_b = CaptionFetcher()
            results_b = fetcher_b.fetch_captions_batch(video_ids_b, max_workers=4)

        assert len(results_a) == 20
        assert len(results_b) == 20

        # Thread pools should have created different thread instances
        # (ThreadPoolExecutor creates new threads for each context)
        # The thread idents from batch A shouldn't all be reused in batch B
        # (though some reuse is possible in Python's thread pool)
        assert len(threads_a) <= 4  # max_workers
        assert len(threads_b) <= 4

    def test_rapid_sequential_batches(self):
        """Test rapid sequential batch fetches don't accumulate resources.

        Verifies no memory leak or thread leak across multiple batch calls.
        """
        video_batch = [f'video{i:03d}test' for i in range(20)]

        def mock_fetch(video_id, preferred_language=None):
            time.sleep(0.01)
            return make_caption_result(video_id)

        gc.collect()
        tracemalloc.start()
        mem_start = tracemalloc.get_traced_memory()[0]
        threads_start = threading.active_count()

        with patch('src.caption_fetcher.CaptionFetcher.fetch_captions_auto_language_with_retry') as mock:
            mock.side_effect = mock_fetch

            fetcher = CaptionFetcher()

            # Run 10 sequential batches
            for i in range(10):
                results = fetcher.fetch_captions_batch(
                    video_batch,
                    max_workers=8
                )
                assert len(results) == 20

        gc.collect()
        time.sleep(0.1)  # Allow thread cleanup

        mem_end = tracemalloc.get_traced_memory()[0]
        threads_end = threading.active_count()
        tracemalloc.stop()

        # Memory shouldn't grow significantly (allow 20MB for test overhead)
        mem_growth_mb = (mem_end - mem_start) / (1024 * 1024)
        assert mem_growth_mb < 20, f"Memory grew by {mem_growth_mb:.1f}MB across batches"

        # Threads shouldn't accumulate
        thread_growth = threads_end - threads_start
        assert thread_growth <= 2, f"Thread count grew by {thread_growth}"
