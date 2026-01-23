"""
Comprehensive tests for parallel caption fetching.

Tests cover:
- WorkerStats tracking
- ParallelCaptionFetcher initialization and operation
- Proxy allocation and distribution
- Cache hit vs network fetch handling
- Delay behavior
- Thread safety
- Error handling
- Logging
"""

import pytest
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Optional
from unittest.mock import Mock, MagicMock, patch, PropertyMock


# === Mock CaptionResult for testing ===

@dataclass
class MockCaptionResult:
    """Mock CaptionResult for testing."""
    video_id: str
    file: str = "/path/to/caption.srt"
    language: str = "en"
    is_auto_generated: bool = False
    format: str = "srt"
    from_cache: bool = False


# === WorkerStats Tests ===

class TestWorkerStats:
    """Test worker statistics tracking."""

    def test_stats_initialization(self):
        """WorkerStats should initialize with correct defaults."""
        from src.stages.caption import WorkerStats

        stats = WorkerStats(worker_id=0, proxy_url="socks5://127.0.0.1:1080")
        assert stats.worker_id == 0
        assert stats.proxy_url == "socks5://127.0.0.1:1080"
        assert stats.processed == 0
        assert stats.cache_hits == 0
        assert stats.network_fetches == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.rate_limits == 0
        assert stats.total_time == 0.0

    def test_stats_initialization_no_proxy(self):
        """WorkerStats should work without proxy."""
        from src.stages.caption import WorkerStats

        stats = WorkerStats(worker_id=1, proxy_url=None)
        assert stats.worker_id == 1
        assert stats.proxy_url is None

    def test_stats_log_summary(self, caplog):
        """WorkerStats should log summary correctly."""
        from src.stages.caption import WorkerStats
        import logging

        stats = WorkerStats(
            worker_id=1,
            proxy_url="http://proxy:8080",
            processed=100,
            cache_hits=80,
            network_fetches=20,
            successes=95,
            failures=5,
            rate_limits=2,
            total_time=45.5
        )

        with caplog.at_level(logging.INFO):
            stats.log_summary()

        assert "worker_1" in caplog.text
        assert "processed=100" in caplog.text
        assert "cache=80" in caplog.text
        assert "network=20" in caplog.text
        assert "success=95" in caplog.text
        assert "fail=5" in caplog.text
        assert "429s=2" in caplog.text


# === ParallelCaptionFetcher Tests ===

class TestParallelCaptionFetcher:
    """Test parallel fetching logic."""

    @pytest.fixture
    def mock_caption_fetcher(self):
        """Create mock CaptionFetcher."""
        fetcher = Mock()
        fetcher.fetch_captions.return_value = MockCaptionResult(
            video_id="test_vid",
            from_cache=False
        )
        return fetcher

    @pytest.fixture
    def mock_proxy_pool(self):
        """Create mock ProxyPool."""
        pool = Mock()
        # Create mock proxy infos
        mock_proxies = [
            Mock(url="socks5://proxy1:1080"),
            Mock(url="socks5://proxy2:1080"),
            Mock(url="socks5://proxy3:1080"),
            Mock(url="socks5://proxy4:1080"),
        ]
        pool.get_n_proxies.return_value = mock_proxies
        return pool

    def test_initialization(self, mock_caption_fetcher, mock_proxy_pool):
        """ParallelCaptionFetcher should initialize correctly."""
        from src.stages.caption import ParallelCaptionFetcher

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=4,
            per_worker_delay=10.0,
            log_activity=True
        )

        assert fetcher.num_workers == 4
        assert fetcher.per_worker_delay == 10.0
        assert fetcher.log_activity is True

    def test_allocates_proxies_to_workers(self, mock_caption_fetcher, mock_proxy_pool):
        """Each worker should get a unique proxy."""
        from src.stages.caption import ParallelCaptionFetcher

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=4,
            per_worker_delay=0.01
        )

        results = fetcher.fetch_all(["vid1", "vid2", "vid3", "vid4"])

        mock_proxy_pool.get_n_proxies.assert_called_once_with(4)

    def test_all_videos_processed(self, mock_caption_fetcher, mock_proxy_pool):
        """All video IDs should be processed."""
        from src.stages.caption import ParallelCaptionFetcher

        # Configure mock to return results with correct video_id
        def mock_fetch(video_id, **kwargs):
            return MockCaptionResult(video_id=video_id, from_cache=False)

        mock_caption_fetcher.fetch_captions.side_effect = mock_fetch

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=4,
            per_worker_delay=0.01
        )

        video_ids = [f"vid_{i}" for i in range(20)]
        results = fetcher.fetch_all(video_ids)

        assert len(results) == 20
        assert all(vid in results for vid in video_ids)

    def test_cache_hits_skip_delay(self, mock_caption_fetcher, mock_proxy_pool):
        """Cache hits should not trigger per-worker delay."""
        from src.stages.caption import ParallelCaptionFetcher

        # All cache hits
        mock_caption_fetcher.fetch_captions.return_value = MockCaptionResult(
            video_id="test",
            from_cache=True
        )

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=2,
            per_worker_delay=10.0  # Long delay that would be obvious
        )

        start = time.time()
        results = fetcher.fetch_all(["vid1", "vid2", "vid3", "vid4"])
        elapsed = time.time() - start

        # Should be fast since all cache hits (no 10s delays)
        assert elapsed < 5.0
        assert all(r.from_cache for r in results.values() if r)

    def test_worker_stats_tracked(self, mock_caption_fetcher, mock_proxy_pool):
        """Worker stats should be accurately tracked."""
        from src.stages.caption import ParallelCaptionFetcher

        # First 3 calls return cache hits, rest return network fetches
        call_count = [0]

        def mock_fetch(video_id, **kwargs):
            call_count[0] += 1
            return MockCaptionResult(
                video_id=video_id,
                from_cache=(call_count[0] <= 3)
            )

        mock_caption_fetcher.fetch_captions.side_effect = mock_fetch

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=2,
            per_worker_delay=0.01
        )

        results = fetcher.fetch_all([f"vid_{i}" for i in range(6)])

        stats = fetcher.get_stats()
        total_cache = stats["totals"]["cache_hits"]
        total_network = stats["totals"]["network_fetches"]

        assert total_cache == 3
        assert total_network == 3

    def test_handles_fetch_errors(self, mock_caption_fetcher, mock_proxy_pool):
        """Errors should be caught and recorded."""
        from src.stages.caption import ParallelCaptionFetcher

        mock_caption_fetcher.fetch_captions.return_value = None  # Simulates failure

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=2,
            per_worker_delay=0.01
        )

        results = fetcher.fetch_all(["vid1", "vid2"])

        assert len(results) == 2
        assert all(r is None for r in results.values())

        stats = fetcher.get_stats()
        total_failures = stats["totals"]["failures"]
        assert total_failures == 2

    def test_handles_exceptions(self, mock_caption_fetcher, mock_proxy_pool):
        """Exceptions should be caught and recorded."""
        from src.stages.caption import ParallelCaptionFetcher

        mock_caption_fetcher.fetch_captions.side_effect = Exception("Network error")

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=2,
            per_worker_delay=0.01
        )

        results = fetcher.fetch_all(["vid1", "vid2"])

        assert len(results) == 2
        assert all(r is None for r in results.values())

        stats = fetcher.get_stats()
        assert stats["totals"]["failures"] == 2

    def test_rate_limits_tracked(self, mock_caption_fetcher, mock_proxy_pool):
        """Rate limit errors should be tracked separately."""
        from src.stages.caption import ParallelCaptionFetcher

        mock_caption_fetcher.fetch_captions.side_effect = Exception("429 rate limited")

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=mock_caption_fetcher,
            proxy_pool=mock_proxy_pool,
            num_workers=2,
            per_worker_delay=0.01
        )

        results = fetcher.fetch_all(["vid1", "vid2", "vid3"])

        stats = fetcher.get_stats()
        total_rate_limits = stats["totals"]["rate_limits"]
        assert total_rate_limits == 3


class TestProxyDistribution:
    """Test that proxies are properly distributed across workers."""

    def test_each_worker_uses_assigned_proxy(self):
        """Verify workers use their assigned proxy, not others."""
        from src.stages.caption import ParallelCaptionFetcher

        proxy_usage = {}
        proxy_usage_lock = threading.Lock()

        def tracking_fetch(video_id, proxy=None, worker_id=None, **kwargs):
            with proxy_usage_lock:
                if worker_id not in proxy_usage:
                    proxy_usage[worker_id] = set()
                proxy_usage[worker_id].add(proxy)
            return MockCaptionResult(video_id=video_id, from_cache=False)

        caption_fetcher = Mock()
        caption_fetcher.fetch_captions.side_effect = tracking_fetch

        proxy_pool = Mock()
        proxy_pool.get_n_proxies.return_value = [
            Mock(url="proxy_0"),
            Mock(url="proxy_1"),
            Mock(url="proxy_2"),
        ]

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=caption_fetcher,
            proxy_pool=proxy_pool,
            num_workers=3,
            per_worker_delay=0.01
        )

        fetcher.fetch_all([f"vid_{i}" for i in range(12)])

        # Each worker should only use its assigned proxy
        for worker_id, proxies_used in proxy_usage.items():
            assert len(proxies_used) == 1, f"Worker {worker_id} used multiple proxies: {proxies_used}"


class TestThreadSafety:
    """Test thread safety of parallel fetching."""

    def test_results_dict_thread_safe(self):
        """Results should be safely written from multiple threads."""
        from src.stages.caption import ParallelCaptionFetcher

        caption_fetcher = Mock()
        caption_fetcher.fetch_captions.return_value = MockCaptionResult(
            video_id="test",
            from_cache=False
        )

        proxy_pool = Mock()
        proxy_pool.get_n_proxies.return_value = [Mock(url=f"proxy_{i}") for i in range(4)]

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=caption_fetcher,
            proxy_pool=proxy_pool,
            num_workers=4,
            per_worker_delay=0.001
        )

        # Large batch to stress test
        video_ids = [f"vid_{i}" for i in range(100)]
        results = fetcher.fetch_all(video_ids)

        # All results should be present
        assert len(results) == 100

        # No duplicates or missing
        assert set(results.keys()) == set(video_ids)


class TestLogging:
    """Test logging output."""

    def test_logs_worker_activity(self, caplog):
        """Detailed logs should be emitted when enabled."""
        from src.stages.caption import ParallelCaptionFetcher
        import logging

        caption_fetcher = Mock()
        caption_fetcher.fetch_captions.return_value = MockCaptionResult(
            video_id="test",
            from_cache=True
        )

        proxy_pool = Mock()
        proxy_pool.get_n_proxies.return_value = [Mock(url="proxy_0")]

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=caption_fetcher,
            proxy_pool=proxy_pool,
            num_workers=1,
            per_worker_delay=0.01,
            log_activity=True
        )

        with caplog.at_level(logging.INFO):
            fetcher.fetch_all(["vid_1"])

        # Should have logged worker activity
        assert any("worker_0" in record.message or "Started" in record.message for record in caplog.records)

    def test_logs_summary_on_complete(self, caplog):
        """Summary should be logged on completion."""
        from src.stages.caption import ParallelCaptionFetcher
        import logging

        caption_fetcher = Mock()
        caption_fetcher.fetch_captions.return_value = MockCaptionResult(
            video_id="test",
            from_cache=True
        )

        proxy_pool = Mock()
        proxy_pool.get_n_proxies.return_value = [Mock(url="proxy_0")]

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=caption_fetcher,
            proxy_pool=proxy_pool,
            num_workers=1,
            per_worker_delay=0.01
        )

        with caplog.at_level(logging.INFO):
            fetcher.fetch_all(["vid_1", "vid_2"])

        # Should have COMPLETE and SUMMARY logs
        log_text = " ".join(r.message for r in caplog.records)
        assert "COMPLETE" in log_text
        assert "SUMMARY" in log_text


# === ProxyPool.get_n_proxies Tests ===

class TestProxyPoolGetNProxies:
    """Test the get_n_proxies method on ProxyPool."""

    def test_get_n_proxies_returns_requested_count(self):
        """get_n_proxies should return up to N proxies."""
        from src.downloader.proxy_manager import ProxyPool, ProxyInfo, ProxyType

        pool = ProxyPool()
        for i in range(5):
            pool.add(ProxyInfo(url=f"http://proxy{i}:8080", proxy_type=ProxyType.HTTP))

        proxies = pool.get_n_proxies(3)
        assert len(proxies) == 3

    def test_get_n_proxies_returns_fewer_if_not_enough(self):
        """get_n_proxies should return fewer if not enough available."""
        from src.downloader.proxy_manager import ProxyPool, ProxyInfo, ProxyType

        pool = ProxyPool()
        for i in range(2):
            pool.add(ProxyInfo(url=f"http://proxy{i}:8080", proxy_type=ProxyType.HTTP))

        proxies = pool.get_n_proxies(5)
        assert len(proxies) == 2

    def test_get_n_proxies_empty_pool(self):
        """get_n_proxies should return empty list for empty pool."""
        from src.downloader.proxy_manager import ProxyPool

        pool = ProxyPool()
        proxies = pool.get_n_proxies(4)
        assert len(proxies) == 0

    def test_get_proxy_for_worker(self):
        """get_proxy_for_worker should return consistent proxy."""
        from src.downloader.proxy_manager import ProxyPool, ProxyInfo, ProxyType

        pool = ProxyPool()
        for i in range(3):
            pool.add(ProxyInfo(url=f"http://proxy{i}:8080", proxy_type=ProxyType.HTTP))

        url1 = pool.get_proxy_for_worker(0)
        url2 = pool.get_proxy_for_worker(1)
        url3 = pool.get_proxy_for_worker(2)
        url4 = pool.get_proxy_for_worker(3)  # Out of range

        assert url1 is not None
        assert url2 is not None
        assert url3 is not None
        assert url4 is None  # No proxy for worker 3


# === Integration Tests ===

@pytest.mark.integration
class TestIntegration:
    """Integration tests with real components."""

    def test_with_real_proxy_pool(self):
        """Test with real ProxyPool (no proxies configured)."""
        from src.downloader.proxy_manager import ProxyPool
        from src.stages.caption import ParallelCaptionFetcher

        caption_fetcher = Mock()
        caption_fetcher.fetch_captions.return_value = MockCaptionResult(
            video_id="test",
            from_cache=True
        )

        pool = ProxyPool()  # Empty pool

        fetcher = ParallelCaptionFetcher(
            caption_fetcher=caption_fetcher,
            proxy_pool=pool,
            num_workers=4,
            per_worker_delay=0.01
        )

        # Should work with None proxies (direct connection)
        results = fetcher.fetch_all(["vid_1", "vid_2"])
        assert len(results) == 2


# === CaptionStage Parallel Mode Tests ===

class TestCaptionStageParallelMode:
    """Test CaptionStage parallel mode switching."""

    def test_has_proxy_pool_false_without_config(self):
        """_has_proxy_pool should return False without proxy config."""
        from src.stages.caption import CaptionStage

        stage = CaptionStage()

        config = Mock()
        config.download = Mock()
        config.download.fallback = None

        assert stage._has_proxy_pool(config) is False

    def test_has_proxy_pool_false_when_disabled(self):
        """_has_proxy_pool should return False when proxy disabled."""
        from src.stages.caption import CaptionStage

        stage = CaptionStage()

        config = Mock()
        config.download = Mock()
        config.download.fallback = Mock()
        config.download.fallback.proxy = Mock()
        config.download.fallback.proxy.enabled = False

        assert stage._has_proxy_pool(config) is False

    def test_has_proxy_pool_true_when_enabled(self):
        """_has_proxy_pool should return True when proxy enabled."""
        from src.stages.caption import CaptionStage

        stage = CaptionStage()

        config = Mock()
        config.download = Mock()
        config.download.fallback = Mock()
        config.download.fallback.proxy = Mock()
        config.download.fallback.proxy.enabled = True

        assert stage._has_proxy_pool(config) is True
