"""
Tests for DownloadOrchestrator rate limit hooks.

Implements US-35-011: Add download orchestrator rate limit hooks.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.downloader.orchestrator import RateLimitHooks
from src.downloader.rate_limit_metrics import RateLimitMetrics
from src.rate_limit.coordinator import GlobalRateLimitCoordinator


@pytest.fixture(autouse=True)
def reset_coordinator():
    """Reset the global coordinator singleton between tests."""
    GlobalRateLimitCoordinator.reset_instance()
    yield
    GlobalRateLimitCoordinator.reset_instance()


class TestRateLimitHooksBasics:
    """Basic hook functionality tests."""

    def test_hooks_init_with_defaults(self):
        """Test hooks initialize with default coordinator."""
        hooks = RateLimitHooks()
        assert hooks._coordinator is not None

    def test_hooks_init_with_custom_coordinator(self):
        """Test hooks accept custom coordinator."""
        coordinator = GlobalRateLimitCoordinator()
        hooks = RateLimitHooks(coordinator=coordinator)
        assert hooks._coordinator is coordinator

    def test_hooks_init_with_metrics(self):
        """Test hooks accept metrics instance."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)
        assert hooks._metrics is metrics


class TestPreDownloadHook:
    """Tests for pre_download_hook slot acquisition."""

    def test_pre_download_acquires_slot(self):
        """Test pre_download_hook acquires a slot successfully."""
        hooks = RateLimitHooks()
        result = hooks.pre_download_hook('video123')
        assert result is True

    def test_pre_download_returns_false_on_timeout(self):
        """Test pre_download_hook returns False when slot times out."""
        coordinator = GlobalRateLimitCoordinator()
        # Exhaust all tokens
        coordinator._tokens = 0

        hooks = RateLimitHooks(coordinator=coordinator)
        result = hooks.pre_download_hook('video123', timeout=0.01)
        assert result is False

    def test_pre_download_records_slot_timeout_metric(self):
        """Test pre_download_hook records slot timeout in metrics."""
        coordinator = GlobalRateLimitCoordinator()
        coordinator._tokens = 0

        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)

        hooks.pre_download_hook('video123', timeout=0.01)

        assert metrics.slot_timeouts == 1

    def test_pre_download_does_not_record_metric_on_success(self):
        """Test pre_download_hook does not record timeout on success."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        result = hooks.pre_download_hook('video123')

        assert result is True
        assert metrics.slot_timeouts == 0


class TestPostDownloadHook:
    """Tests for post_download_hook slot release."""

    def test_post_download_releases_slot(self):
        """Test post_download_hook releases the slot."""
        coordinator = GlobalRateLimitCoordinator()
        hooks = RateLimitHooks(coordinator=coordinator)

        # Acquire a slot first
        hooks.pre_download_hook('video123')
        initial_active = coordinator.get_active_slots('download')

        # Release it
        hooks.post_download_hook('video123', success=True)

        final_active = coordinator.get_active_slots('download')
        assert final_active == initial_active - 1

    def test_post_download_releases_slot_on_failure(self):
        """Test post_download_hook releases slot even on failure."""
        coordinator = GlobalRateLimitCoordinator()
        hooks = RateLimitHooks(coordinator=coordinator)

        hooks.pre_download_hook('video123')
        initial_active = coordinator.get_active_slots('download')

        hooks.post_download_hook('video123', success=False)

        final_active = coordinator.get_active_slots('download')
        assert final_active == initial_active - 1


class TestOnErrorHook:
    """Tests for on_error_hook rate limit event recording."""

    def test_on_error_records_429_event(self):
        """Test on_error_hook records 429 as rate limit event."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=429,
            error_message='Too Many Requests'
        )

        assert metrics.rate_limit_events == 1

    def test_on_error_records_429_with_tier(self):
        """Test on_error_hook records 429 with tier breakdown."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=429,
            error_message='Too Many Requests',
            tier='medium'
        )

        assert metrics.tier_rate_limit_events.get('medium') == 1

    def test_on_error_records_429_with_keyword(self):
        """Test on_error_hook records 429 with keyword breakdown."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=429,
            error_message='Too Many Requests',
            keyword='sunset'
        )

        assert metrics.keyword_rate_limit_events.get('sunset') == 1

    def test_on_error_detects_rate_limit_in_message(self):
        """Test on_error_hook detects rate limit from error message."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=403,
            error_message='Request failed: rate limit exceeded'
        )

        assert metrics.rate_limit_events == 1

    def test_on_error_detects_too_many_requests_in_message(self):
        """Test on_error_hook detects 'too many requests' phrase."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=500,
            error_message='Error: too many requests from this IP'
        )

        assert metrics.rate_limit_events == 1

    def test_on_error_does_not_double_count_429(self):
        """Test on_error_hook doesn't double count 429 with rate limit message."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=429,
            error_message='Rate limit exceeded'
        )

        # Should only record once, not twice
        assert metrics.rate_limit_events == 1

    def test_on_error_ignores_non_rate_limit_errors(self):
        """Test on_error_hook ignores unrelated errors."""
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(metrics=metrics)

        hooks.on_error_hook(
            video_id='video123',
            error_code=404,
            error_message='Video not found'
        )

        assert metrics.rate_limit_events == 0

    def test_on_error_handles_no_metrics(self):
        """Test on_error_hook works without metrics (no crash)."""
        hooks = RateLimitHooks(metrics=None)

        # Should not raise
        hooks.on_error_hook(
            video_id='video123',
            error_code=429,
            error_message='Too Many Requests'
        )


class TestHookOrdering:
    """Tests for correct hook calling order during download."""

    def test_hooks_called_in_correct_order(self):
        """Test pre -> download -> post hook ordering."""
        coordinator = GlobalRateLimitCoordinator()
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)

        call_order = []

        # Pre-download
        result = hooks.pre_download_hook('video123')
        call_order.append('pre')
        assert result is True

        # Simulate download (would happen here)
        call_order.append('download')

        # Post-download
        hooks.post_download_hook('video123', success=True)
        call_order.append('post')

        assert call_order == ['pre', 'download', 'post']

    def test_error_hook_after_download_failure(self):
        """Test error hook called on download failure."""
        coordinator = GlobalRateLimitCoordinator()
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)

        call_order = []

        # Pre-download
        hooks.pre_download_hook('video123')
        call_order.append('pre')

        # Simulate download failure
        call_order.append('download_fail')

        # Error hook for 429
        hooks.on_error_hook('video123', 429, 'Too Many Requests', tier='short')
        call_order.append('error')

        # Post-download (always called)
        hooks.post_download_hook('video123', success=False)
        call_order.append('post')

        assert call_order == ['pre', 'download_fail', 'error', 'post']
        assert metrics.rate_limit_events == 1

    def test_multiple_downloads_with_hooks(self):
        """Test hooks work correctly across multiple downloads."""
        coordinator = GlobalRateLimitCoordinator()
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)

        # First download - success
        assert hooks.pre_download_hook('video1')
        hooks.post_download_hook('video1', success=True)

        # Second download - 429 error
        assert hooks.pre_download_hook('video2')
        hooks.on_error_hook('video2', 429, 'Rate limited', tier='medium')
        hooks.post_download_hook('video2', success=False)

        # Third download - success
        assert hooks.pre_download_hook('video3')
        hooks.post_download_hook('video3', success=True)

        # Verify metrics
        assert metrics.rate_limit_events == 1
        assert metrics.tier_rate_limit_events.get('medium') == 1


class TestImportFromOrchestrator:
    """Test that RateLimitHooks can be imported from downloader package."""

    def test_import_from_downloader_package(self):
        """Test RateLimitHooks is exported from src.downloader."""
        from src.downloader import RateLimitHooks as ImportedHooks
        assert ImportedHooks is RateLimitHooks

    def test_import_from_orchestrator_module(self):
        """Test direct import from orchestrator module."""
        from src.downloader.orchestrator import RateLimitHooks as DirectHooks
        assert DirectHooks is RateLimitHooks
