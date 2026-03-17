"""
Thread safety tests for CaptionMetrics.

Verifies that concurrent access from multiple threads produces correct
results without race condition overwrites, count overruns, or lost updates.

Created: February 1, 2026
User Story: US-34-004 - Extract caption metrics to dedicated module
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.caption.metrics import CaptionMetrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def fresh_metrics():
    """Fresh CaptionMetrics instance for each test."""
    return CaptionMetrics()


# ============================================================================
# AC1: Concurrent increment_success() from multiple threads
# ============================================================================

class TestConcurrentIncrementSuccess:
    """Test concurrent increment_success() — verify no race condition overwrites."""

    @pytest.mark.fast
    def test_10_threads_each_increment_once(self, fresh_metrics):
        """10 threads each call increment_success() once — successes must equal 10."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def increment_once():
            barrier.wait()  # Synchronize start
            metrics.increment_success()

        threads = [threading.Thread(target=increment_once) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.successes == 10

    @pytest.mark.fast
    def test_10_threads_each_increment_100_times(self, fresh_metrics):
        """10 threads each call increment_success() 100 times — total must be 1000."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def increment_many():
            barrier.wait()
            for _ in range(100):
                metrics.increment_success()

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.successes == 1000


# ============================================================================
# AC2: Concurrent increment_failure() from multiple threads
# ============================================================================

class TestConcurrentIncrementFailure:
    """Test concurrent increment_failure() — verify no lost updates."""

    @pytest.mark.fast
    def test_10_threads_each_increment_once(self, fresh_metrics):
        """10 threads each call increment_failure() once — failures must equal 10."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def increment_once():
            barrier.wait()
            metrics.increment_failure()

        threads = [threading.Thread(target=increment_once) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.failures == 10

    @pytest.mark.fast
    def test_10_threads_each_increment_100_times(self, fresh_metrics):
        """10 threads each call increment_failure() 100 times — total must be 1000."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def increment_many():
            barrier.wait()
            for _ in range(100):
                metrics.increment_failure()

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.failures == 1000


# ============================================================================
# AC3: Concurrent increment_cached() from multiple threads
# ============================================================================

class TestConcurrentIncrementCached:
    """Test concurrent increment_cached() — verify no lost updates."""

    @pytest.mark.fast
    def test_10_threads_each_increment_once(self, fresh_metrics):
        """10 threads each call increment_cached() once — cache_hits must equal 10."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def increment_once():
            barrier.wait()
            metrics.increment_cached()

        threads = [threading.Thread(target=increment_once) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.cache_hits == 10

    @pytest.mark.fast
    def test_10_threads_each_increment_100_times(self, fresh_metrics):
        """10 threads each call increment_cached() 100 times — total must be 1000."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def increment_many():
            barrier.wait()
            for _ in range(100):
                metrics.increment_cached()

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.cache_hits == 1000


# ============================================================================
# AC4: Mixed concurrent operations
# ============================================================================

class TestMixedConcurrentOperations:
    """Test concurrent success/failure/cached recording — total must equal call count."""

    @pytest.mark.fast
    def test_mixed_operations_from_30_threads(self, fresh_metrics):
        """30 threads: 10 success, 10 failure, 10 cached — counts must match."""
        metrics = fresh_metrics
        barrier = threading.Barrier(30)

        def record_success():
            barrier.wait()
            for _ in range(100):
                metrics.increment_success()

        def record_failure():
            barrier.wait()
            for _ in range(100):
                metrics.increment_failure()

        def record_cached():
            barrier.wait()
            for _ in range(100):
                metrics.increment_cached()

        threads = []
        for i in range(30):
            if i < 10:
                threads.append(threading.Thread(target=record_success))
            elif i < 20:
                threads.append(threading.Thread(target=record_failure))
            else:
                threads.append(threading.Thread(target=record_cached))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.successes == 1000   # 10 threads x 100 increments
        assert metrics.failures == 1000    # 10 threads x 100 increments
        assert metrics.cache_hits == 1000  # 10 threads x 100 increments

    @pytest.mark.fast
    def test_interleaved_operations_same_threads(self, fresh_metrics):
        """Each thread records success, failure, and cached in sequence."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def record_all():
            barrier.wait()
            for _ in range(50):
                metrics.increment_success()
                metrics.increment_failure()
                metrics.increment_cached()

        threads = [threading.Thread(target=record_all) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.successes == 500    # 10 threads x 50
        assert metrics.failures == 500     # 10 threads x 50
        assert metrics.cache_hits == 500   # 10 threads x 50


# ============================================================================
# AC5: Concurrent get_summary_dict() while updating
# ============================================================================

class TestConcurrentSummaryAccess:
    """Test concurrent get_summary_dict() while metrics are being updated."""

    @pytest.mark.fast
    def test_summary_during_updates(self, fresh_metrics):
        """get_summary_dict() returns consistent data during concurrent updates."""
        metrics = fresh_metrics
        barrier = threading.Barrier(11)  # 10 updaters + 1 reader
        summaries = []
        lock = threading.Lock()

        def update_metrics():
            barrier.wait()
            for _ in range(100):
                metrics.increment_success()
                metrics.increment_failure()
                metrics.increment_cached()

        def read_summaries():
            barrier.wait()
            for _ in range(50):
                summary = metrics.get_summary_dict()
                with lock:
                    summaries.append(summary)

        updaters = [threading.Thread(target=update_metrics) for _ in range(10)]
        reader = threading.Thread(target=read_summaries)

        for t in updaters:
            t.start()
        reader.start()

        for t in updaters:
            t.join()
        reader.join()

        # All summaries should be consistent (successes + failures + cache_hits = total_processed)
        for summary in summaries:
            expected_total = summary['successes'] + summary['failures'] + summary['cache_hits']
            assert summary['total_processed'] == expected_total

        # Final state should have correct totals
        final = metrics.get_summary_dict()
        assert final['successes'] == 1000
        assert final['failures'] == 1000
        assert final['cache_hits'] == 1000

    @pytest.mark.fast
    def test_hit_rate_calculation_consistent(self, fresh_metrics):
        """hit_rate in summary is always between 0 and 100."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)
        summaries = []
        lock = threading.Lock()

        def update_and_read():
            barrier.wait()
            for _ in range(50):
                metrics.increment_success()
                metrics.increment_cached()
                summary = metrics.get_summary_dict()
                with lock:
                    summaries.append(summary)

        threads = [threading.Thread(target=update_and_read) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All hit rates should be valid percentages
        for summary in summaries:
            assert 0 <= summary['hit_rate'] <= 100
            assert 0 <= summary['success_rate'] <= 100


# ============================================================================
# AC6: Concurrent record_fetch_success() with full data
# ============================================================================

class TestConcurrentRecordFetchSuccess:
    """Test concurrent record_fetch_success() — verify all data tracked correctly."""

    @pytest.mark.fast
    def test_10_threads_record_fetch_success(self, fresh_metrics):
        """10 threads each record success with different languages."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)
        languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'ru']

        def record_success(lang):
            barrier.wait()
            for _ in range(10):
                metrics.record_fetch_success(
                    video_id=f"vid_{lang}_{_}",
                    language=lang,
                    quality="high",
                    segment_count=100,
                    is_auto_generated=False,
                    coverage_ratio=0.9
                )

        threads = [
            threading.Thread(target=record_success, args=(lang,))
            for lang in languages
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All successes recorded
        assert metrics.successes == 100  # 10 threads x 10 recordings

        # All languages tracked
        assert len(metrics.language_distribution) == 10
        for lang in languages:
            assert metrics.language_distribution[lang] == 10

        # Total segments tracked
        assert metrics.total_segments == 10000  # 100 recordings x 100 segments


# ============================================================================
# AC7: Stress test with many threads
# ============================================================================

class TestStressConcurrency:
    """Stress test with many concurrent threads."""

    @pytest.mark.fast
    def test_50_threads_increment_success(self, fresh_metrics):
        """50 threads each increment success 200 times — total must be 10000."""
        metrics = fresh_metrics
        barrier = threading.Barrier(50)

        def increment_many():
            barrier.wait()
            for _ in range(200):
                metrics.increment_success()

        threads = [threading.Thread(target=increment_many) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.successes == 10000

    @pytest.mark.fast
    def test_merge_concurrent_metrics(self, fresh_metrics):
        """Merge operation is thread-safe."""
        metrics = fresh_metrics
        barrier = threading.Barrier(10)

        def create_and_merge():
            barrier.wait()
            other = CaptionMetrics()
            other.successes = 10
            other.failures = 5
            other.cache_hits = 3
            metrics.merge(other)

        threads = [threading.Thread(target=create_and_merge) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 merges, each adding 10 successes
        assert metrics.successes == 100
        assert metrics.failures == 50
        assert metrics.cache_hits == 30
