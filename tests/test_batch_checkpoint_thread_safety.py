"""
Thread safety tests for CaptionBatchCheckpoint.

Verifies that concurrent update() and mark_aborted() calls from multiple
threads produce correct counts without race conditions or lost updates.

User Story: US-66-009 - Add thread safety to BatchCaptionCheckpoint.update_progress
"""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.caption_fetcher import CaptionBatchCheckpoint


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def checkpoint():
    """Fresh CaptionBatchCheckpoint with 200 remaining video IDs."""
    ids = [f"vid_{i:04d}" for i in range(200)]
    return CaptionBatchCheckpoint(
        total_requested=200,
        remaining_video_ids=list(ids),
    )


@pytest.fixture
def small_checkpoint():
    """Small checkpoint for focused tests."""
    ids = [f"vid_{i:02d}" for i in range(20)]
    return CaptionBatchCheckpoint(
        total_requested=20,
        remaining_video_ids=list(ids),
    )


# ============================================================================
# AC4a: Concurrent update() calls produce correct success counts
# ============================================================================

class TestConcurrentUpdate:
    """Concurrent update() from multiple threads must produce correct counts."""

    @pytest.mark.fast
    def test_10_threads_each_update_10_videos(self, checkpoint):
        """10 threads each update 10 unique videos — success_count must be 100."""
        barrier = threading.Barrier(10)

        def update_batch(thread_idx):
            barrier.wait()
            for i in range(10):
                vid_id = f"vid_{thread_idx * 10 + i:04d}"
                checkpoint.update(vid_id, {
                    'video_id': vid_id,
                    'segments': [],
                    'language': 'en',
                })

        threads = [threading.Thread(target=update_batch, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert checkpoint.success_count == 100
        assert checkpoint.error_count == 0
        assert len(checkpoint.results) == 100
        # All 100 videos should be removed from remaining
        for t in range(10):
            for i in range(10):
                vid_id = f"vid_{t * 10 + i:04d}"
                assert vid_id not in checkpoint.remaining_video_ids

    @pytest.mark.fast
    def test_concurrent_success_and_error_updates(self, checkpoint):
        """Half threads report success, half report errors — counts must sum correctly."""
        n_threads = 10
        per_thread = 10
        barrier = threading.Barrier(n_threads)

        def update_batch(thread_idx):
            barrier.wait()
            for i in range(per_thread):
                vid_id = f"vid_{thread_idx * per_thread + i:04d}"
                if thread_idx < n_threads // 2:
                    # Success result (dict without error key)
                    checkpoint.update(vid_id, {
                        'video_id': vid_id,
                        'segments': [],
                        'language': 'en',
                    })
                else:
                    # Error result
                    checkpoint.update(vid_id, {
                        'video_id': vid_id,
                        'error': '403 Forbidden',
                    })

        threads = [threading.Thread(target=update_batch, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n_threads * per_thread
        assert checkpoint.success_count + checkpoint.error_count == total
        assert checkpoint.success_count == (n_threads // 2) * per_thread
        assert checkpoint.error_count == (n_threads // 2) * per_thread
        assert len(checkpoint.results) == total

    @pytest.mark.fast
    def test_high_contention_100_threads(self, checkpoint):
        """100 threads each update 1 video — no lost updates."""
        barrier = threading.Barrier(100)

        def update_one(idx):
            barrier.wait()
            vid_id = f"vid_{idx:04d}"
            checkpoint.update(vid_id, {
                'video_id': vid_id,
                'segments': [],
                'language': 'en',
            })

        threads = [threading.Thread(target=update_one, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert checkpoint.success_count == 100
        assert len(checkpoint.results) == 100


# ============================================================================
# AC4b: Concurrent mark_aborted() calls
# ============================================================================

class TestConcurrentMarkAborted:
    """Concurrent mark_aborted() must not corrupt shared state."""

    @pytest.mark.fast
    def test_concurrent_mark_aborted(self, small_checkpoint):
        """Multiple threads calling mark_aborted — state must be consistent."""
        barrier = threading.Barrier(5)

        def abort_call(idx):
            barrier.wait()
            small_checkpoint.mark_aborted(
                reason=f"Error pattern {idx}",
                remaining_ids=[f"remaining_{idx}"],
            )

        threads = [threading.Thread(target=abort_call, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After concurrent aborts, state must be consistent (one of the abort calls wins)
        assert small_checkpoint.aborted is True
        assert small_checkpoint.abort_reason.startswith("Error pattern")
        assert len(small_checkpoint.remaining_video_ids) == 1


# ============================================================================
# AC4c: Mixed update() and mark_aborted() concurrency
# ============================================================================

class TestMixedConcurrency:
    """Interleaved update() and mark_aborted() must not corrupt state."""

    @pytest.mark.fast
    def test_updates_interleaved_with_abort(self):
        """Updates and aborts running concurrently — counts must be consistent."""
        # Use a larger checkpoint so each thread gets unique IDs
        ids = [f"vid_{i:04d}" for i in range(50)]
        cp = CaptionBatchCheckpoint(
            total_requested=50,
            remaining_video_ids=list(ids),
        )
        barrier = threading.Barrier(6)

        def do_updates(thread_idx):
            barrier.wait()
            for i in range(10):
                vid_id = f"vid_{thread_idx * 10 + i:04d}"
                cp.update(vid_id, {
                    'video_id': vid_id,
                    'segments': [],
                    'language': 'en',
                })

        def do_abort():
            barrier.wait()
            cp.mark_aborted(
                reason="Rate limited",
                remaining_ids=["leftover_1", "leftover_2"],
            )

        update_threads = [threading.Thread(target=do_updates, args=(t,)) for t in range(5)]
        abort_thread = threading.Thread(target=do_abort)

        for t in update_threads:
            t.start()
        abort_thread.start()

        for t in update_threads:
            t.join()
        abort_thread.join()

        # success + error must equal number of unique results
        assert checkpoint_counts_consistent(cp)
        assert cp.aborted is True
        assert cp.success_count == 50


# ============================================================================
# AC5: No deadlocks when calling update_progress from locked context
# ============================================================================

class TestNoDeadlock:
    """RLock must allow re-entrant acquisition without deadlock."""

    @pytest.mark.fast
    def test_reentrant_lock_no_deadlock(self, small_checkpoint):
        """Acquiring lock then calling update() (which also acquires lock) must not deadlock.

        Uses RLock so re-entrant acquisition from the same thread is safe.
        """
        # Manually acquire the lock, then call update — RLock allows this
        with small_checkpoint._lock:
            small_checkpoint.update("vid_00", {
                'video_id': 'vid_00',
                'segments': [],
                'language': 'en',
            })

        assert small_checkpoint.success_count == 1
        assert "vid_00" in small_checkpoint.results

    @pytest.mark.fast
    def test_reentrant_lock_mark_aborted(self, small_checkpoint):
        """Acquiring lock then calling mark_aborted() must not deadlock."""
        with small_checkpoint._lock:
            small_checkpoint.mark_aborted(reason="test abort")

        assert small_checkpoint.aborted is True

    @pytest.mark.fast
    def test_to_dict_under_external_lock(self, small_checkpoint):
        """to_dict() called while holding external lock must not deadlock."""
        small_checkpoint.update("vid_00", {
            'video_id': 'vid_00',
            'segments': [],
            'language': 'en',
        })

        with small_checkpoint._lock:
            d = small_checkpoint.to_dict()

        assert d['success_count'] == 1


# ============================================================================
# Helpers
# ============================================================================

def checkpoint_counts_consistent(cp: CaptionBatchCheckpoint) -> bool:
    """Verify success_count + error_count equals number of results."""
    return cp.success_count + cp.error_count == len(cp.results)
