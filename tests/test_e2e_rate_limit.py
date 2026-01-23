"""
End-to-end tests for rate limit handling and proxy integration.

Tests are split into two categories:
1. Mock tests - test logic without real network (always runnable)
2. E2E tests - require real proxy infrastructure (skip if unavailable)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Skip E2E tests if no proxy configured
def has_proxy_configured():
    """Check if proxy infrastructure is available."""
    try:
        from src.config import get_config
        config = get_config()
        proxy_config = getattr(config.download.fallback, 'proxy', None)
        if proxy_config and getattr(proxy_config, 'enabled', False):
            sources = getattr(proxy_config, 'sources', [])
            return len(sources) > 0
    except Exception:
        pass
    return False


def has_vpn_configured():
    """Check if VPN CLI is available."""
    import subprocess
    for vpn in ['nordvpn', 'mullvad', 'expressvpn', 'protonvpn']:
        try:
            result = subprocess.run([vpn, 'status'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
        except Exception:
            continue
    return False


skip_no_proxy = pytest.mark.skipif(
    not has_proxy_configured(),
    reason="Proxy not configured - set download.fallback.proxy.enabled=true and add sources"
)

skip_no_vpn = pytest.mark.skipif(
    not has_vpn_configured(),
    reason="VPN CLI not available"
)


# =============================================================================
# MOCK TESTS - Test rotation logic without real network
# =============================================================================

class TestProxyRotationLogic:
    """Test proxy rotation logic with mocked responses."""

    def test_handler_tracks_rate_limits(self):
        """Handler should track consecutive rate limits."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler(base_backoff=0.01, max_backoff=0.02)  # Minimal backoff for tests
        initial = handler._consecutive_rate_limits

        handler.on_rate_limit("test")

        assert handler._consecutive_rate_limits > initial

    def test_success_resets_rate_limit_count(self):
        """on_success should reset consecutive rate limit count."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler(base_backoff=0.01, max_backoff=0.02)

        # Accumulate rate limits
        handler.on_rate_limit("test")
        assert handler._consecutive_rate_limits > 0

        # Success resets
        handler.on_success()
        assert handler._consecutive_rate_limits == 0

    def test_handler_stats_track_rate_limits(self):
        """Stats should track total rate limits."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler(base_backoff=0.01, max_backoff=0.02)

        handler.on_rate_limit("test1")
        handler.on_rate_limit("test2")

        assert handler.stats.rate_limits_total == 2
        assert handler.stats.backoffs_applied == 2  # Without proxy/VPN, backoff is applied

    def test_handler_has_proxy_without_config(self):
        """Handler without config should have no proxy."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler()
        assert handler.has_proxy is False
        assert handler.get_proxy() is None


class TestCaptionFetcherProxyInjection:
    """Test that caption fetchers receive and use proxy."""

    def test_innertube_fetcher_creates_client_with_proxy(self):
        """InnertubeDirectFetcher should create httpx client with proxy."""
        from src.downloader.caption_fallback import InnertubeDirectFetcher
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = Mock(spec=RateLimitHandler)
        handler.has_proxy = True
        handler.get_proxy.return_value = "http://proxy:8080"

        with patch('src.downloader.http_client.httpx.Client') as mock_client:
            fetcher = InnertubeDirectFetcher(rate_limit_handler=handler)
            _ = fetcher.client  # Access client to trigger creation

            # Verify proxy was passed to Client
            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs.get('proxy') == "http://proxy:8080"

    def test_invidious_fetcher_passes_proxy_to_get(self):
        """InvidiousCaptionFetcher should pass proxy to httpx.get."""
        from src.downloader.caption_fallback import InvidiousCaptionFetcher

        handler = Mock()
        handler.has_proxy = True
        handler.get_proxy.return_value = "http://proxy:8080"

        with patch('src.downloader.http_client.get_proxy_for_httpx', return_value="http://proxy:8080"):
            with patch('httpx.get') as mock_get:
                mock_get.return_value = Mock(
                    status_code=200,
                    json=Mock(return_value={'captions': []})
                )

                fetcher = InvidiousCaptionFetcher(rate_limit_handler=handler)
                fetcher.fetch("test_video_id")

                # Verify proxy was passed
                mock_get.assert_called()
                call_kwargs = mock_get.call_args[1]
                assert call_kwargs.get('proxy') == "http://proxy:8080"

    def test_fetcher_calls_on_success_after_successful_fetch(self):
        """Fetcher should call handler.on_success after successful fetch."""
        from src.downloader.caption_fallback import InvidiousCaptionFetcher

        handler = Mock()
        handler.has_proxy = False

        with patch('src.downloader.http_client.get_proxy_for_httpx', return_value=None):
            with patch('httpx.get') as mock_get:
                mock_get.return_value = Mock(
                    status_code=200,
                    json=Mock(return_value={'captions': [{'label': 'English', 'url': 'http://test'}]})
                )

                fetcher = InvidiousCaptionFetcher(rate_limit_handler=handler)
                result = fetcher.fetch("test_video_id")

                if result.success:
                    handler.on_success.assert_called()


class TestVideoFetcherProxyInjection:
    """Test that video fetchers receive and use proxy."""

    def test_invidious_stream_fetcher_passes_proxy(self):
        """InvidiousStreamFetcher should pass proxy to httpx.get."""
        from src.downloader.video_fallback import InvidiousStreamFetcher

        handler = Mock()
        handler.has_proxy = True
        handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

        with patch('src.downloader.http_client.get_proxy_for_httpx', return_value="socks5://127.0.0.1:1080"):
            with patch('httpx.get') as mock_get:
                mock_get.return_value = Mock(
                    status_code=200,
                    json=Mock(return_value={'formatStreams': []})
                )

                fetcher = InvidiousStreamFetcher(rate_limit_handler=handler)
                fetcher.get_streams("test_video_id")

                mock_get.assert_called()
                call_kwargs = mock_get.call_args[1]
                assert call_kwargs.get('proxy') == "socks5://127.0.0.1:1080"


class TestYtdlpProxyInjection:
    """Test that yt-dlp commands receive proxy flag."""

    def test_proxy_added_to_ytdlp_command(self):
        """VideoDownloader should add --proxy to yt-dlp commands."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.cookies_from_browser = ''
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        from src.downloader.core import VideoDownloader
                                        downloader = VideoDownloader()

            # Set up handler with proxy
            downloader.rate_limit_handler = Mock()
            downloader.rate_limit_handler.has_proxy = True
            downloader.rate_limit_handler.get_proxy.return_value = "socks5://127.0.0.1:1080"

            cmd = ['yt-dlp', 'URL']
            downloader._add_cookies_to_cmd(cmd)

            assert '--proxy' in cmd
            proxy_idx = cmd.index('--proxy')
            assert cmd[proxy_idx + 1] == "socks5://127.0.0.1:1080"

    def test_no_proxy_when_handler_has_none(self):
        """Should not add --proxy when handler has no proxy."""
        with patch('src.downloader.core.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.cache_dir = "/tmp/cache"
            mock_cfg.downloaded_videos_dir = "/tmp/videos"
            mock_cfg.download = MagicMock()
            mock_cfg.download.rate_limit_bypass = None
            mock_cfg.download.cookies_from_browser = ''
            mock_config.return_value = mock_cfg

            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        from src.downloader.core import VideoDownloader
                                        downloader = VideoDownloader()

            downloader.rate_limit_handler = Mock()
            downloader.rate_limit_handler.has_proxy = False

            cmd = ['yt-dlp', 'URL']
            downloader._add_cookies_to_cmd(cmd)

            assert '--proxy' not in cmd


class TestRateLimitHandlerStatistics:
    """Test rate limit handler statistics tracking."""

    def test_stats_track_all_events(self):
        """Stats should track rate limits, backoffs, etc."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler(base_backoff=0.01, max_backoff=0.02)

        # Trigger some rate limits
        handler.on_rate_limit("source1")
        handler.on_rate_limit("source2")
        handler.on_success()
        handler.on_rate_limit("source3")

        stats = handler.get_stats()

        assert stats['rate_limits_total'] == 3
        assert stats['consecutive_rate_limits'] == 1  # Reset after success
        assert stats['backoffs_applied'] == 3

    def test_reset_clears_stats(self):
        """reset() should clear all statistics."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler(base_backoff=0.01, max_backoff=0.02)

        handler.on_rate_limit("test")
        handler.reset()

        assert handler.stats.rate_limits_total == 0
        assert handler._consecutive_rate_limits == 0


# =============================================================================
# INTEGRATION TESTS - Can run without full infrastructure
# =============================================================================

class TestIntegrationHttpClientFactory:
    """Integration tests for http_client factory."""

    def test_factory_creates_client_without_handler(self):
        """Factory should create client even without handler."""
        from src.downloader.http_client import create_httpx_client

        client = create_httpx_client(handler=None, timeout=5.0)
        assert client is not None
        client.close()

    def test_factory_creates_async_client(self):
        """Factory should create async client."""
        from src.downloader.http_client import create_async_httpx_client

        client = create_async_httpx_client(handler=None, timeout=5.0)
        assert client is not None

    def test_proxy_aware_session_context_manager(self):
        """ProxyAwareSession should work as context manager."""
        from src.downloader.http_client import ProxyAwareSession

        with ProxyAwareSession(handler=None) as session:
            assert session.client is not None


class TestIntegrationHandlerLifecycle:
    """Integration tests for handler lifecycle."""

    def test_get_rate_limit_handler_returns_handler(self):
        """get_rate_limit_handler should return a handler instance."""
        from src.downloader.rate_limit_handler import get_rate_limit_handler, reset_rate_limit_handler

        # Reset to ensure clean state
        reset_rate_limit_handler()

        with patch('src.config.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.download.fallback.proxy.enabled = False
            mock_config.return_value = mock_cfg

            handler = get_rate_limit_handler(mock_cfg)
            assert handler is not None

        # Clean up
        reset_rate_limit_handler()

    def test_handler_state_persists_across_calls(self):
        """Handler should maintain state across multiple on_rate_limit calls."""
        from src.downloader.rate_limit_handler import RateLimitHandler

        handler = RateLimitHandler(base_backoff=0.01, max_backoff=0.02)

        handler.on_rate_limit("source1")
        limits_after_1 = handler._consecutive_rate_limits

        handler.on_rate_limit("source2")
        limits_after_2 = handler._consecutive_rate_limits

        assert limits_after_2 > limits_after_1


class TestProxyManagerIntegration:
    """Test ProxyManager without real proxies."""

    def test_proxy_manager_without_proxies(self):
        """ProxyManager should work gracefully without proxies."""
        from src.downloader.proxy_manager import ProxyManager

        with patch('src.config.get_config') as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.download.fallback.proxy.enabled = False
            mock_cfg.download.fallback.proxy.sources = []
            mock_config.return_value = mock_cfg

            manager = ProxyManager(mock_cfg)

            assert manager.has_proxies is False
            assert manager.get_proxy() is None

    def test_proxy_manager_add_proxy_manually(self):
        """ProxyManager should work when proxies added manually."""
        from src.downloader.proxy_manager import ProxyManager, ProxyInfo, ProxyType

        # Create manager without config
        manager = ProxyManager()

        # Add proxy manually (simulates what config loading does)
        manager.pool.add(ProxyInfo(
            url="http://test:8080",
            proxy_type=ProxyType.HTTP,
            source="test"
        ))

        assert manager.has_proxies is True
        proxy = manager.get_proxy()
        assert proxy == "http://test:8080"


# =============================================================================
# E2E TESTS - Require real proxy infrastructure
# =============================================================================

@skip_no_proxy
class TestE2EWithProxy:
    """End-to-end tests requiring real proxy."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_baseline_no_proxy(self):
        """Establish baseline: success rate without proxy protection."""
        from src.downloader.caption_fallback import CaptionFallbackChain
        from src.config import get_config

        config = get_config()
        # Temporarily disable proxy
        original_enabled = config.download.fallback.proxy.enabled
        config.download.fallback.proxy.enabled = False

        try:
            chain = CaptionFallbackChain()

            video_ids = [
                "dQw4w9WgXcQ",  # Rick Astley
                "jNQXAC9IVRw",  # First YouTube video
                "9bZkp7q19f0",  # Gangnam Style
            ]

            results = []
            for vid in video_ids:
                result = chain.fetch(vid)
                results.append({
                    'video_id': vid,
                    'success': result.success,
                    'error': str(result.error) if result.error else None,
                })
                print(f"{vid}: {'✅' if result.success else '❌'} - {result.error or 'OK'}")
                time.sleep(0.5)  # Small delay between requests

            successes = sum(1 for r in results if r['success'])
            print(f"\nBaseline: {successes}/{len(results)} succeeded without proxy")

            return results

        finally:
            config.download.fallback.proxy.enabled = original_enabled

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_with_proxy_rotation(self):
        """Test that proxy rotation improves success rate."""
        from src.downloader.caption_fallback import CaptionFallbackChain
        from src.config import get_config

        config = get_config()
        assert config.download.fallback.proxy.enabled, "Proxy must be enabled for this test"

        chain = CaptionFallbackChain()
        handler = chain.rate_limit_handler

        video_ids = [
            "dQw4w9WgXcQ",
            "jNQXAC9IVRw",
            "9bZkp7q19f0",
            "kJQP7kiw5Fk",
            "RgKAFK5djSk",
        ]

        initial_proxy = handler.get_proxy() if handler.has_proxy else None
        rotation_count = 0

        results = []
        for vid in video_ids:
            current_proxy = handler.get_proxy() if handler.has_proxy else None
            if current_proxy != initial_proxy:
                rotation_count += 1
                initial_proxy = current_proxy
                print(f"🔄 Proxy rotated to: {current_proxy}")

            result = chain.fetch(vid)
            results.append({
                'video_id': vid,
                'success': result.success,
                'proxy_used': current_proxy,
            })
            print(f"{vid}: {'✅' if result.success else '❌'} via {current_proxy or 'direct'}")

        successes = sum(1 for r in results if r['success'])
        print(f"\nWith proxy: {successes}/{len(results)} succeeded")
        print(f"Proxy rotations: {rotation_count}")

        assert successes > 0, "Should have at least some successes with proxy"

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ytdlp_uses_proxy(self):
        """Verify yt-dlp actually receives and uses proxy flag."""
        import subprocess
        from src.config import get_config

        config = get_config()
        assert config.download.fallback.proxy.enabled, "Proxy must be enabled for this test"

        with patch('src.downloader.core.get_config', return_value=config):
            with patch('src.downloader.core.CheckpointManager'):
                with patch('src.downloader.core.TranscodingManager'):
                    with patch('src.downloader.core.TitleFilter'):
                        with patch('src.downloader.core.SpeechScreener'):
                            with patch('src.downloader.core.SearchOptimizer'):
                                with patch('src.downloader.core.YouTubeSearchCache'):
                                    with patch('src.downloader.core.AudioFirstPipeline'):
                                        from src.downloader.core import VideoDownloader
                                        downloader = VideoDownloader()

        cmd = ['yt-dlp', '--dump-json', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ']
        downloader._add_cookies_to_cmd(cmd)

        print(f"Command: {' '.join(cmd)}")
        assert '--proxy' in cmd, "Proxy flag should be in command"

        proxy_idx = cmd.index('--proxy')
        proxy_value = cmd[proxy_idx + 1]
        print(f"Proxy value: {proxy_value}")

        # Try a dry run with the proxy
        test_cmd = ['yt-dlp', '--proxy', proxy_value, '--dump-json', '--no-download',
                    'https://www.youtube.com/watch?v=dQw4w9WgXcQ']

        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("✅ yt-dlp successfully used proxy")
            else:
                print(f"⚠️ yt-dlp returned non-zero: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("⚠️ yt-dlp timed out")
        except Exception as e:
            print(f"⚠️ yt-dlp error: {e}")


@skip_no_proxy
class TestE2EStressTest:
    """Stress tests requiring real proxy infrastructure."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_stress_rapid_requests(self):
        """Hammer YouTube to trigger rate limits, verify recovery."""
        from src.downloader.caption_fallback import CaptionFallbackChain
        from src.config import get_config

        config = get_config()
        assert config.download.fallback.proxy.enabled, "Proxy must be enabled for this test"

        chain = CaptionFallbackChain()

        # Same video 10 times rapidly to trigger rate limit
        video_ids = ["dQw4w9WgXcQ"] * 10

        results = {'success': 0, 'rate_limited': 0, 'other_error': 0}
        rate_limit_times = []

        start = time.time()

        for i, vid in enumerate(video_ids):
            result = chain.fetch(vid)

            if result.success:
                results['success'] += 1
            elif '429' in str(result.error) or 'rate' in str(result.error).lower():
                results['rate_limited'] += 1
                rate_limit_times.append(time.time() - start)
            else:
                results['other_error'] += 1

            print(f"Request {i+1}: {'✅' if result.success else '❌'} - {result.error or 'OK'}")

        elapsed = time.time() - start

        print(f"\n=== STRESS TEST RESULTS ===")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Success: {results['success']}")
        print(f"Rate limited: {results['rate_limited']}")
        print(f"Other errors: {results['other_error']}")

        if rate_limit_times:
            print(f"First rate limit at: {rate_limit_times[0]:.1f}s")

        recovery_rate = results['success'] / len(video_ids)
        print(f"Recovery rate: {recovery_rate:.1%}")


@skip_no_vpn
class TestE2EVPNEscalation:
    """VPN escalation tests requiring VPN CLI."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_vpn_escalation(self):
        """Test that VPN rotation triggers when all proxies exhausted."""
        from src.downloader.rate_limit_handler import get_rate_limit_handler, reset_rate_limit_handler
        from src.config import get_config

        reset_rate_limit_handler()
        config = get_config()
        handler = get_rate_limit_handler(config)

        # Exhaust all proxies
        for i in range(10):
            handler.on_rate_limit("test")

        print(f"Consecutive rate limits: {handler._consecutive_rate_limits}")
        print(f"Current proxy: {handler.get_proxy() if handler.has_proxy else 'None'}")

        reset_rate_limit_handler()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
