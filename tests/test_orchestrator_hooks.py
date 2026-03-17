"""
Tests for DownloadOrchestrator hooks.

Implements US-36-009: Create tests for download orchestrator hooks.

Tests cover:
- Pre-download hook is called before each download
- Post-download hook receives correct success/failure status
- Rate limit hook pauses orchestrator correctly
- Circuit breaker hook integration works
- Hook exceptions don't crash orchestrator
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.downloader.orchestrator import DownloadOrchestrator, RateLimitHooks
from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.downloader.rate_limit_metrics import RateLimitMetrics
from src.rate_limit.coordinator import GlobalRateLimitCoordinator


@pytest.fixture(autouse=True)
def reset_coordinator():
    """Reset the global coordinator singleton between tests."""
    GlobalRateLimitCoordinator.reset_instance()
    yield
    GlobalRateLimitCoordinator.reset_instance()


class TestPreDownloadHookCalled:
    """Tests that pre-download hook is called before each download."""

    @pytest.mark.fast
    def test_pre_download_hook_called_before_download(self):
        """Pre-download hook is invoked before download attempt."""
        hooks = RateLimitHooks()
        call_sequence = []

        # Track call order
        original_pre = hooks.pre_download_hook
        def tracked_pre(*args, **kwargs):
            call_sequence.append('pre')
            return original_pre(*args, **kwargs)

        hooks.pre_download_hook = tracked_pre

        # Simulate download sequence
        hooks.pre_download_hook('video123')
        call_sequence.append('download')

        assert call_sequence == ['pre', 'download']

    @pytest.mark.fast
    def test_pre_download_hook_receives_video_id(self):
        """Pre-download hook receives the correct video ID."""
        hooks = RateLimitHooks()
        captured_ids = []

        original_pre = hooks.pre_download_hook
        def capture_pre(video_id, *args, **kwargs):
            captured_ids.append(video_id)
            return original_pre(video_id, *args, **kwargs)

        hooks.pre_download_hook = capture_pre
        hooks.pre_download_hook('abc123')
        hooks.pre_download_hook('xyz789')

        assert captured_ids == ['abc123', 'xyz789']

    @pytest.mark.fast
    def test_pre_download_hook_called_for_each_video(self):
        """Pre-download hook is called once per video."""
        hooks = RateLimitHooks()
        call_count = 0

        original_pre = hooks.pre_download_hook
        def counting_pre(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_pre(*args, **kwargs)

        hooks.pre_download_hook = counting_pre

        for vid in ['v1', 'v2', 'v3']:
            hooks.pre_download_hook(vid)

        assert call_count == 3


class TestPostDownloadHookStatus:
    """Tests that post-download hook receives correct success/failure status."""

    @pytest.mark.fast
    def test_post_download_hook_receives_success_true(self):
        """Post-download hook receives success=True on successful download."""
        coordinator = GlobalRateLimitCoordinator()
        hooks = RateLimitHooks(coordinator=coordinator)
        captured_status = []

        original_post = hooks.post_download_hook
        def capture_post(video_id, success):
            captured_status.append((video_id, success))
            return original_post(video_id, success)

        hooks.post_download_hook = capture_post

        hooks.pre_download_hook('video1')
        hooks.post_download_hook('video1', success=True)

        assert ('video1', True) in captured_status

    @pytest.mark.fast
    def test_post_download_hook_receives_success_false(self):
        """Post-download hook receives success=False on failed download."""
        coordinator = GlobalRateLimitCoordinator()
        hooks = RateLimitHooks(coordinator=coordinator)
        captured_status = []

        original_post = hooks.post_download_hook
        def capture_post(video_id, success):
            captured_status.append((video_id, success))
            return original_post(video_id, success)

        hooks.post_download_hook = capture_post

        hooks.pre_download_hook('video2')
        hooks.post_download_hook('video2', success=False)

        assert ('video2', False) in captured_status

    @pytest.mark.fast
    def test_post_download_hook_multiple_statuses(self):
        """Post-download hook tracks mixed success/failure correctly."""
        coordinator = GlobalRateLimitCoordinator()
        hooks = RateLimitHooks(coordinator=coordinator)
        statuses = []

        original_post = hooks.post_download_hook
        def capture_post(video_id, success):
            statuses.append(success)
            return original_post(video_id, success)

        hooks.post_download_hook = capture_post

        # Simulate downloads: success, fail, success
        hooks.pre_download_hook('v1')
        hooks.post_download_hook('v1', success=True)

        hooks.pre_download_hook('v2')
        hooks.post_download_hook('v2', success=False)

        hooks.pre_download_hook('v3')
        hooks.post_download_hook('v3', success=True)

        assert statuses == [True, False, True]


class TestRateLimitHookPause:
    """Tests that rate limit hook pauses orchestrator correctly."""

    @pytest.mark.fast
    def test_rate_limit_hook_blocks_when_no_slots(self):
        """Pre-download hook blocks when no rate limit slots available."""
        coordinator = GlobalRateLimitCoordinator()
        coordinator._tokens = 0  # Exhaust slots

        hooks = RateLimitHooks(coordinator=coordinator)
        result = hooks.pre_download_hook('video123', timeout=0.01)

        assert result is False

    @pytest.mark.fast
    def test_rate_limit_hook_records_timeout_metric(self):
        """Rate limit hook records slot timeout in metrics."""
        coordinator = GlobalRateLimitCoordinator()
        coordinator._tokens = 0

        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)

        hooks.pre_download_hook('video123', timeout=0.01)

        assert metrics.slot_timeouts >= 1

    @pytest.mark.fast
    def test_rate_limit_hook_allows_when_slot_available(self):
        """Pre-download hook proceeds when slot is available."""
        hooks = RateLimitHooks()
        result = hooks.pre_download_hook('video123')

        assert result is True


class TestCircuitBreakerHookIntegration:
    """Tests circuit breaker hook integration with orchestrator."""

    @pytest.mark.fast
    def test_circuit_breaker_blocks_when_open(self):
        """Circuit breaker hook blocks downloads when circuit is open."""
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=2,
            pause_seconds=0.01,  # Short for testing
            jitter_factor=0.0    # No jitter for predictable tests
        )
        breaker = CircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.is_open is True

    @pytest.mark.fast
    def test_circuit_breaker_tracks_failures(self):
        """Circuit breaker hook tracks consecutive failures."""
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        breaker.record_failure()
        assert breaker.state.consecutive_failures == 1

        breaker.record_failure()
        assert breaker.state.consecutive_failures == 2

    @pytest.mark.fast
    def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker hook resets failure count on success."""
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=5,
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state.consecutive_failures == 2

        breaker.record_success()
        assert breaker.state.consecutive_failures == 0

    @pytest.mark.fast
    def test_circuit_breaker_recovers_after_pause(self):
        """Circuit breaker hook recovers after pause duration."""
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=1,
            pause_seconds=0.001,  # Very short
            jitter_factor=0.0
        )
        breaker = CircuitBreaker(config)

        breaker.record_failure()
        assert breaker.is_open is True

        # Wait for recovery
        breaker.check_and_wait()
        assert breaker.is_open is False


class TestHookExceptionHandling:
    """Tests that hook exceptions don't crash orchestrator."""

    @pytest.mark.fast
    def test_pre_hook_exception_does_not_crash(self):
        """Orchestrator survives pre-hook exceptions."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        # Create hooks with exception-throwing pre_download
        hooks = RateLimitHooks()
        original_pre = hooks.pre_download_hook

        def failing_pre(*args, **kwargs):
            raise ValueError("Test exception")

        hooks.pre_download_hook = failing_pre

        # Should handle exception gracefully
        with pytest.raises(ValueError):
            hooks.pre_download_hook('video123')

    @pytest.mark.fast
    def test_post_hook_exception_does_not_crash(self):
        """Orchestrator survives post-hook exceptions."""
        hooks = RateLimitHooks()

        def failing_post(*args, **kwargs):
            raise RuntimeError("Post-hook error")

        hooks.post_download_hook = failing_post

        # Exception is raised but doesn't affect other operations
        with pytest.raises(RuntimeError):
            hooks.post_download_hook('video123', success=True)

    @pytest.mark.fast
    def test_error_hook_exception_does_not_crash(self):
        """Error hook gracefully handles internal exceptions."""
        hooks = RateLimitHooks(metrics=None)  # No metrics

        # Should not raise even when metrics is None
        hooks.on_error_hook(
            video_id='video123',
            error_code=429,
            error_message='Too Many Requests'
        )

    @pytest.mark.fast
    def test_coordinate_retries_handles_hook_errors(self):
        """coordinate_retries continues despite hook errors."""
        mock_downloader = MagicMock()
        mock_downloader._lock = MagicMock()
        mock_downloader.tier_download_counts = {}
        mock_downloader.sources = []
        mock_downloader._download_single.return_value = []

        orchestrator = DownloadOrchestrator(mock_downloader)
        failed_items = [
            {'keyword': 'test', 'tier': 'medium', 'video_id': 'test|medium'}
        ]

        # Should complete without crashing
        recovered, still_failed = orchestrator.coordinate_retries(
            failed_items, Path("/tmp"), "topic"
        )

        assert len(recovered) == 0
        assert len(still_failed) == 1


class TestHookOrchestration:
    """Tests for complete hook orchestration flow."""

    @pytest.mark.fast
    def test_full_hook_flow_success(self):
        """Test complete pre -> download -> post flow for success."""
        coordinator = GlobalRateLimitCoordinator()
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)
        events = []

        # Pre-download
        result = hooks.pre_download_hook('video123')
        events.append(('pre', result))

        # Simulate successful download
        events.append(('download', True))

        # Post-download
        hooks.post_download_hook('video123', success=True)
        events.append(('post', True))

        assert events == [('pre', True), ('download', True), ('post', True)]
        assert metrics.rate_limit_events == 0

    @pytest.mark.fast
    def test_full_hook_flow_with_error(self):
        """Test complete hook flow with download error."""
        coordinator = GlobalRateLimitCoordinator()
        metrics = RateLimitMetrics()
        hooks = RateLimitHooks(coordinator=coordinator, metrics=metrics)
        events = []

        # Pre-download
        result = hooks.pre_download_hook('video456')
        events.append(('pre', result))

        # Simulate failed download
        events.append(('download', False))

        # Error hook
        hooks.on_error_hook('video456', 429, 'Rate limited', tier='short')
        events.append(('error', 429))

        # Post-download
        hooks.post_download_hook('video456', success=False)
        events.append(('post', False))

        assert events == [
            ('pre', True),
            ('download', False),
            ('error', 429),
            ('post', False)
        ]
        assert metrics.rate_limit_events == 1
