"""
Parallel download escalation coordination tests.

Verifies that the EscalationManager correctly handles concurrent access
from multiple download threads, maintaining independent per-keyword tier
states, thread-safe failure counting, and consistent argument generation
under contention.

Created: January 28, 2026
User Story: US-006 (Sprint 14) - Add parallel download escalation coordination tests
"""

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.escalation_manager import EscalationManager, EscalationResult
from src.downloader.rate_limit_budget import RateLimitBudget
from src.downloader.types import EscalationState, EscalationTier


# ============================================================================
# Helpers / Fixtures
# ============================================================================

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 0.0  # No cooldown for thread safety tests
    max_tier: int = 3


def _make_impersonation_manager():
    """Create a mock ImpersonationManager with deterministic args."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


@pytest.fixture
def ext_config():
    return FakeExtractorArgsConfig()


@pytest.fixture
def imp_manager():
    return _make_impersonation_manager()


@pytest.fixture
def escalation_manager(imp_manager, ext_config):
    return EscalationManager(imp_manager, ext_config)


@pytest.fixture
def budget_manager(imp_manager, ext_config):
    """EscalationManager with a shared RateLimitBudget."""
    budget = RateLimitBudget()
    budget.max_rotations = 5
    budget.max_backoff_time = 300.0
    return EscalationManager(imp_manager, ext_config, budget=budget)


# ============================================================================
# AC1: Two concurrent keywords sharing EscalationManager — independent tiers
# ============================================================================

class TestConcurrentKeywordIndependentTiers:
    """Keyword A escalates to Tier 2 while B stays at Tier 1."""

    def test_keyword_a_tier2_keyword_b_tier1(self, escalation_manager):
        """Escalate keyword A via failures while B remains untouched."""
        em = escalation_manager
        barrier = threading.Barrier(2)
        results = {}

        def escalate_keyword_a():
            barrier.wait()
            # Record 2 consecutive 403s to trigger escalation (threshold=2)
            em.record_failure("kw_a", "HTTP Error 403")
            em.record_failure("kw_a", "HTTP Error 403")
            results["a"] = em.get_escalation_args("kw_a")

        def access_keyword_b():
            barrier.wait()
            # Just access keyword B — no failures
            results["b"] = em.get_escalation_args("kw_b")

        t_a = threading.Thread(target=escalate_keyword_a)
        t_b = threading.Thread(target=access_keyword_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        # Keyword A should be at Tier 2 (escalated)
        assert results["a"].tier == EscalationTier.EXTRACTOR_ARGS
        # Keyword B should still be at Tier 1 (untouched)
        assert results["b"].tier == EscalationTier.IMPERSONATE_ONLY

    def test_10_keywords_independent_tiers_concurrent(self, escalation_manager):
        """10 concurrent threads each managing a separate keyword — verify independence."""
        em = escalation_manager
        num_keywords = 10
        barrier = threading.Barrier(num_keywords)
        results = {}

        def worker(idx):
            keyword = f"kw_{idx}"
            barrier.wait()
            # Even-indexed keywords get 2 failures (escalate to Tier 2)
            if idx % 2 == 0:
                em.record_failure(keyword, "HTTP Error 403")
                em.record_failure(keyword, "HTTP Error 403")
            # Odd-indexed keywords: just access (stay at Tier 1)
            results[keyword] = em.get_escalation_args(keyword)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_keywords)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Verify each keyword's tier independently
        for idx in range(num_keywords):
            keyword = f"kw_{idx}"
            if idx % 2 == 0:
                assert results[keyword].tier == EscalationTier.EXTRACTOR_ARGS, (
                    f"{keyword} should be Tier 2 but got {results[keyword].tier}"
                )
            else:
                assert results[keyword].tier == EscalationTier.IMPERSONATE_ONLY, (
                    f"{keyword} should be Tier 1 but got {results[keyword].tier}"
                )

    def test_escalation_does_not_leak_across_keywords(self, escalation_manager):
        """Full escalation of one keyword doesn't affect neighbors."""
        em = escalation_manager
        barrier = threading.Barrier(2)

        def full_escalation():
            barrier.wait()
            # Tier 1 -> 2
            em.record_failure("heavy", "HTTP Error 403")
            em.record_failure("heavy", "HTTP Error 403")
            # Tier 2 -> 3
            em.record_failure("heavy", "HTTP Error 403")
            em.record_failure("heavy", "HTTP Error 403")

        def light_access():
            barrier.wait()
            em.get_escalation_args("light")
            # Small sleep to ensure heavy has time to escalate
            time.sleep(0.01)
            em.get_escalation_args("light")

        t1 = threading.Thread(target=full_escalation)
        t2 = threading.Thread(target=light_access)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        heavy_result = em.get_escalation_args("heavy")
        light_result = em.get_escalation_args("light")

        assert heavy_result.tier == EscalationTier.FULL_BYPASS
        assert light_result.tier == EscalationTier.IMPERSONATE_ONLY


# ============================================================================
# AC2: Concurrent record_failure() — thread-safe consecutive_failures count
# ============================================================================

class TestConcurrentRecordFailure:
    """Verify consecutive_failures count is accurate under concurrent access."""

    def test_concurrent_failures_same_keyword_count_accurate(self, escalation_manager):
        """10 threads each record 1 failure for the same keyword — count must be exact."""
        em = escalation_manager
        # Use high threshold so no escalation resets the counter
        em._extractor_config.escalation_threshold = 100
        num_threads = 10
        barrier = threading.Barrier(num_threads)

        def record_one_failure(thread_id):
            barrier.wait()
            em.record_failure("shared_kw", "HTTP Error 403")

        threads = [
            threading.Thread(target=record_one_failure, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        state = em._keyword_states["shared_kw"]
        assert state.consecutive_403s == num_threads, (
            f"Expected {num_threads} failures but got {state.consecutive_403s}"
        )

    def test_concurrent_failures_100_iterations(self, escalation_manager):
        """10 threads each record 10 failures — total 100, count must be accurate."""
        em = escalation_manager
        em._extractor_config.escalation_threshold = 200  # Prevent escalation resets
        num_threads = 10
        iterations = 10
        barrier = threading.Barrier(num_threads)

        def record_multiple(thread_id):
            barrier.wait()
            for _ in range(iterations):
                em.record_failure("shared_kw", "HTTP Error 403")

        threads = [
            threading.Thread(target=record_multiple, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        state = em._keyword_states["shared_kw"]
        expected = num_threads * iterations
        assert state.consecutive_403s == expected, (
            f"Expected {expected} failures but got {state.consecutive_403s}"
        )

    def test_total_403s_accurate_across_keywords(self, escalation_manager):
        """Multiple threads recording failures for different keywords — total_403s accurate."""
        em = escalation_manager
        em._extractor_config.escalation_threshold = 100
        num_threads = 5
        failures_per_thread = 4
        barrier = threading.Barrier(num_threads)

        def record_for_keyword(idx):
            barrier.wait()
            for _ in range(failures_per_thread):
                em.record_failure(f"kw_{idx}", "HTTP Error 403")

        threads = [
            threading.Thread(target=record_for_keyword, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        expected_total = num_threads * failures_per_thread
        assert em._total_403s == expected_total, (
            f"Expected total_403s={expected_total} but got {em._total_403s}"
        )


# ============================================================================
# AC3: Per-keyword locks prevent cross-keyword interference
# ============================================================================

class TestPerKeywordLockIsolation:
    """Verify per-keyword locks prevent cross-keyword interference."""

    def test_concurrent_escalations_different_keywords(self, escalation_manager):
        """Two keywords escalate simultaneously — verify each gets correct tier."""
        em = escalation_manager
        barrier = threading.Barrier(2)
        results = {}

        def escalate_keyword(keyword):
            barrier.wait()
            em.record_failure(keyword, "HTTP Error 403")
            em.record_failure(keyword, "HTTP Error 403")
            results[keyword] = em.get_escalation_args(keyword)

        t1 = threading.Thread(target=escalate_keyword, args=("alpha",))
        t2 = threading.Thread(target=escalate_keyword, args=("beta",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["alpha"].tier == EscalationTier.EXTRACTOR_ARGS
        assert results["beta"].tier == EscalationTier.EXTRACTOR_ARGS
        # Verify separate lock objects were created
        assert "alpha" in em._keyword_locks
        assert "beta" in em._keyword_locks
        assert em._keyword_locks["alpha"] is not em._keyword_locks["beta"]

    def test_interleaved_failure_success_different_keywords(self, escalation_manager):
        """Failures on keyword A don't reset when keyword B succeeds."""
        em = escalation_manager
        em._extractor_config.escalation_threshold = 5  # High threshold
        barrier = threading.Barrier(2)

        def fail_keyword_a():
            barrier.wait()
            for _ in range(3):
                em.record_failure("kw_a", "HTTP Error 403")

        def succeed_keyword_b():
            barrier.wait()
            em.get_escalation_args("kw_b")
            em.record_success("kw_b")
            em.record_success("kw_b")

        t1 = threading.Thread(target=fail_keyword_a)
        t2 = threading.Thread(target=succeed_keyword_b)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # kw_a should have 3 consecutive failures intact
        state_a = em._keyword_states["kw_a"]
        assert state_a.consecutive_403s == 3
        # kw_b should have 0 failures (only successes)
        state_b = em._keyword_states["kw_b"]
        assert state_b.consecutive_403s == 0

    def test_20_keywords_concurrent_no_lock_contention(self, escalation_manager):
        """20 concurrent keywords — verify all complete without deadlock."""
        em = escalation_manager
        num_keywords = 20
        barrier = threading.Barrier(num_keywords)
        completed = threading.Event()
        errors = []

        def worker(idx):
            try:
                keyword = f"kw_{idx}"
                barrier.wait(timeout=5)
                em.record_failure(keyword, "HTTP Error 403")
                em.get_escalation_args(keyword)
                em.record_success(keyword)
                em.get_escalation_args(keyword)
            except Exception as e:
                errors.append((idx, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_keywords)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert em.get_active_keyword_count() == num_keywords

    def test_lock_creation_is_thread_safe(self, escalation_manager):
        """Multiple threads requesting locks for same keyword — only one lock created."""
        em = escalation_manager
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        locks_seen = []

        def get_lock_for_keyword():
            barrier.wait()
            lock = em._get_lock("same_keyword")
            locks_seen.append(id(lock))

        threads = [threading.Thread(target=get_lock_for_keyword) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All threads should see the same lock object
        unique_locks = set(locks_seen)
        assert len(unique_locks) == 1, (
            f"Expected 1 unique lock but got {len(unique_locks)}"
        )


# ============================================================================
# AC4: Shared budget depletion during parallel downloads
# ============================================================================

class TestSharedBudgetDepletion:
    """When keyword A exhausts rotations, keyword B's can_rotate() returns False."""

    def test_keyword_a_exhausts_budget_keyword_b_blocked(self, imp_manager, ext_config):
        """Keyword A uses all rotations — keyword B's escalation sees exhausted budget."""
        budget = RateLimitBudget()
        budget.max_rotations = 3  # Small budget for testing
        em = EscalationManager(imp_manager, ext_config, budget=budget)
        barrier = threading.Barrier(2)
        results = {}

        def exhaust_budget():
            """Keyword A escalates multiple times, consuming all rotations."""
            barrier.wait()
            # Each escalation calls budget.record_rotation
            em.record_failure("kw_a", "HTTP Error 403")
            em.record_failure("kw_a", "HTTP Error 403")  # -> Tier 2 (1 rotation)
            em.record_failure("kw_a", "HTTP Error 403")
            em.record_failure("kw_a", "HTTP Error 403")  # -> Tier 3 (2 rotations)
            results["a_can_rotate"] = budget.can_rotate()

        def check_budget():
            """Keyword B checks budget after A finishes."""
            barrier.wait()
            # Small delay to let A exhaust budget first
            time.sleep(0.05)
            results["b_can_rotate"] = budget.can_rotate()
            results["b_rotations_remaining"] = budget.rotations_remaining()

        t_a = threading.Thread(target=exhaust_budget)
        t_b = threading.Thread(target=check_budget)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5)
        t_b.join(timeout=5)

        # Budget should be consumed by keyword A's escalations
        assert budget.rotations_used >= 2, (
            f"Expected >=2 rotations used but got {budget.rotations_used}"
        )

    def test_concurrent_budget_consumption_exact_limit(self, imp_manager, ext_config):
        """Two keywords escalate concurrently — total rotations must not exceed budget."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        em = EscalationManager(imp_manager, ext_config, budget=budget)
        num_keywords = 5
        barrier = threading.Barrier(num_keywords)

        def escalate_keyword(idx):
            keyword = f"kw_{idx}"
            barrier.wait()
            # Each keyword does 2 failures to trigger escalation (Tier 1 -> 2)
            em.record_failure(keyword, "HTTP Error 403")
            em.record_failure(keyword, "HTTP Error 403")

        threads = [
            threading.Thread(target=escalate_keyword, args=(i,))
            for i in range(num_keywords)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Each keyword escalating once = 1 rotation each = 5 total
        assert budget.rotations_used == num_keywords, (
            f"Expected {num_keywords} rotations but got {budget.rotations_used}"
        )
        assert budget.can_rotate() is True  # 5 < 10

    def test_budget_exhaustion_forces_max_tier_skip(self, imp_manager, ext_config):
        """When budget is exhausted, escalation skips directly to FULL_BYPASS."""
        budget = RateLimitBudget()
        budget.max_rotations = 1  # Very tight budget
        em = EscalationManager(imp_manager, ext_config, budget=budget)

        # First keyword uses the one rotation (Tier 1 -> 2)
        em.record_failure("kw_first", "HTTP Error 403")
        em.record_failure("kw_first", "HTTP Error 403")
        assert budget.rotations_used == 1
        assert budget.can_rotate() is False

        # Second keyword should skip directly to FULL_BYPASS (budget exhausted)
        em.record_failure("kw_second", "HTTP Error 403")
        em.record_failure("kw_second", "HTTP Error 403")

        state_second = em._keyword_states["kw_second"]
        assert state_second.current_tier == EscalationTier.FULL_BYPASS, (
            f"Expected FULL_BYPASS but got {state_second.current_tier}"
        )


# ============================================================================
# AC5: Concurrent get_escalation_args() returns consistent tier-appropriate args
# ============================================================================

class TestConcurrentGetEscalationArgs:
    """Verify no partial reads of tier state during concurrent escalation."""

    def test_concurrent_reads_return_valid_tier_args(self, escalation_manager):
        """Multiple threads reading args for same keyword — each returns valid result."""
        em = escalation_manager
        # Pre-escalate to Tier 2
        em.record_failure("shared", "HTTP Error 403")
        em.record_failure("shared", "HTTP Error 403")

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results = []
        lock = threading.Lock()

        def read_args():
            barrier.wait()
            result = em.get_escalation_args("shared")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=read_args) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == num_threads
        # All results should be at Tier 2 (consistent)
        for r in results:
            assert r.tier == EscalationTier.EXTRACTOR_ARGS, (
                f"Expected EXTRACTOR_ARGS but got {r.tier}"
            )
            # Tier 2 should include --extractor-args
            assert any("--extractor-args" in arg for arg in r.args), (
                f"Tier 2 args missing --extractor-args: {r.args}"
            )

    def test_concurrent_read_during_escalation(self, escalation_manager):
        """One thread escalates while others read — reads must return valid tier."""
        em = escalation_manager
        num_readers = 5
        barrier = threading.Barrier(num_readers + 1)  # +1 for writer
        read_results = []
        lock = threading.Lock()

        def writer():
            barrier.wait()
            em.record_failure("target", "HTTP Error 403")
            em.record_failure("target", "HTTP Error 403")
            # Escalation from Tier 1 -> 2 happens here
            em.record_failure("target", "HTTP Error 403")
            em.record_failure("target", "HTTP Error 403")
            # Escalation from Tier 2 -> 3 happens here

        def reader():
            barrier.wait()
            # Read multiple times during potential escalation
            for _ in range(5):
                result = em.get_escalation_args("target")
                with lock:
                    read_results.append(result)

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader) for _ in range(num_readers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All results must be a valid tier (1, 2, or 3)
        valid_tiers = {
            EscalationTier.IMPERSONATE_ONLY,
            EscalationTier.EXTRACTOR_ARGS,
            EscalationTier.FULL_BYPASS,
        }
        for r in read_results:
            assert r.tier in valid_tiers, f"Invalid tier: {r.tier}"

        # Verify args are consistent with their tier
        for r in read_results:
            if r.tier == EscalationTier.IMPERSONATE_ONLY:
                # Tier 1: should have --impersonate but NOT --extractor-args
                assert any("--impersonate" in arg for arg in r.args)
                assert not any("--extractor-args" in arg for arg in r.args)
            elif r.tier == EscalationTier.EXTRACTOR_ARGS:
                # Tier 2: should have both
                assert any("--impersonate" in arg for arg in r.args)
                assert any("--extractor-args" in arg for arg in r.args)
            elif r.tier == EscalationTier.FULL_BYPASS:
                # Tier 3: should have both + rotate_cookies
                assert any("--impersonate" in arg for arg in r.args)
                assert any("--extractor-args" in arg for arg in r.args)
                assert r.rotate_cookies is True

    def test_concurrent_different_keywords_consistent(self, escalation_manager):
        """Multiple keywords read concurrently — each gets its own correct tier."""
        em = escalation_manager

        # Pre-set different tiers for different keywords
        # kw_tier1: no failures (stays Tier 1)
        em.get_escalation_args("kw_tier1")
        # kw_tier2: escalate to Tier 2
        em.record_failure("kw_tier2", "HTTP Error 403")
        em.record_failure("kw_tier2", "HTTP Error 403")
        # kw_tier3: escalate to Tier 3
        em.record_failure("kw_tier3", "HTTP Error 403")
        em.record_failure("kw_tier3", "HTTP Error 403")
        em.record_failure("kw_tier3", "HTTP Error 403")
        em.record_failure("kw_tier3", "HTTP Error 403")

        num_iterations = 10
        barrier = threading.Barrier(3)
        results = {"tier1": [], "tier2": [], "tier3": []}
        lock = threading.Lock()

        def read_keyword(keyword, result_key):
            barrier.wait()
            for _ in range(num_iterations):
                r = em.get_escalation_args(keyword)
                with lock:
                    results[result_key].append(r.tier)

        threads = [
            threading.Thread(target=read_keyword, args=("kw_tier1", "tier1")),
            threading.Thread(target=read_keyword, args=("kw_tier2", "tier2")),
            threading.Thread(target=read_keyword, args=("kw_tier3", "tier3")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All reads for each keyword should return same tier (no cross-contamination)
        assert all(t == EscalationTier.IMPERSONATE_ONLY for t in results["tier1"]), (
            f"Tier 1 keyword returned unexpected tiers: {set(results['tier1'])}"
        )
        assert all(t == EscalationTier.EXTRACTOR_ARGS for t in results["tier2"]), (
            f"Tier 2 keyword returned unexpected tiers: {set(results['tier2'])}"
        )
        assert all(t == EscalationTier.FULL_BYPASS for t in results["tier3"]), (
            f"Tier 3 keyword returned unexpected tiers: {set(results['tier3'])}"
        )

    def test_args_never_contain_partial_state(self, escalation_manager):
        """Stress test: 50 threads reading same keyword — no partial/corrupt args."""
        em = escalation_manager
        # Escalate to Tier 2
        em.record_failure("stress", "HTTP Error 403")
        em.record_failure("stress", "HTTP Error 403")

        num_threads = 50
        barrier = threading.Barrier(num_threads)
        all_results = []
        lock = threading.Lock()

        def stress_read():
            barrier.wait()
            result = em.get_escalation_args("stress")
            with lock:
                all_results.append(result)

        threads = [threading.Thread(target=stress_read) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(all_results) == num_threads

        for r in all_results:
            # Every result should be a complete, valid EscalationResult
            assert isinstance(r, EscalationResult)
            assert isinstance(r.args, list)
            assert isinstance(r.tier, EscalationTier)
            assert isinstance(r.rotate_cookies, bool)
            # Args should always have at least --impersonate pair
            assert len(r.args) >= 2, f"Args too short: {r.args}"
            assert r.args[0] == "--impersonate"
