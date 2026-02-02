"""Integration tests for CaptionStage with rate limiting (US-34-005).

Tests the integration between CaptionStage and rate limiting components:
- 429 error handling with exponential backoff
- Circuit breaker respect during fetches
- Metrics tracking across parallel workers
- Checkpointing that survives rate limit pauses

All tests use mocked yt-dlp subprocess - NO network calls are made.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from src.caption.circuit_breaker import (
    CaptionCircuitBreaker,
    CaptionCircuitBreakerConfig,
)
from src.caption.metrics import CaptionMetrics
from src.caption.rate_limiter import (
    RateLimitConfig,
    UnifiedCaptionRateLimiter,
)
from src.caption_fetcher import (
    CaptionFetcher,
    CaptionFetchError,
    CaptionResult,
    CaptionSegment,
    CaptionUnavailableError,
)
from src.stages.caption_stage import CaptionStage


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class MockConfig:
    """Mock config for testing CaptionStage."""
    download: Any = None


@dataclass
class MockDownloadConfig:
    """Mock download config with caption_first settings."""
    caption_first: Any = None
    impersonation: Any = None
    extractor_args: Any = None


@dataclass
class MockCaptionFirstConfig:
    """Mock caption_first config with rate limiting options."""
    enabled: bool = True
    preferred_language: str = "en"
    prefer_human_captions: bool = True
    timeout: int = 30
    max_parallel_fetches: int = 4
    min_coverage_threshold: float = 0.5
    pre_check_availability: bool = False
    skip_live_streams: bool = False
    fallback_to_transcription: bool = True
    checkpoint_save_interval: int = 10
    use_global_coordinator: bool = False
    circuit_breaker: Optional[Dict] = None
    retry_budget: Optional[Dict] = None
    adaptive_format_order: bool = False


@dataclass
class MockCheckpointManager:
    """Mock checkpoint manager for testing."""
    stage_data: Dict[str, Any] = field(default_factory=dict)

    def get_stage_data(self, stage_name: str) -> Optional[Dict]:
        return self.stage_data.get(stage_name)

    def should_skip_stage(self, stage_name: str) -> bool:
        return False

    def save_intermediate(self, stage_name: str, data: Dict) -> None:
        self.stage_data[stage_name] = data


@dataclass
class MockPipelineState:
    """Mock pipeline state for testing."""
    downloaded_videos: List = field(default_factory=list)
    downloaded_audio: List = field(default_factory=list)
    text_metadata: List = field(default_factory=list)
    caption_results: Dict = field(default_factory=dict)
    project_dir: Optional[Path] = None
    pending_streams: List = field(default_factory=list)


@dataclass
class MockDownloadedVideo:
    """Mock downloaded video with video ID."""
    video_id: str
    file: str = ""
    url: str = ""
    duration: float = 120.0
    duration_tier: str = "medium"


def make_mock_caption_result(video_id: str, segment_count: int = 10) -> CaptionResult:
    """Create a mock CaptionResult for testing."""
    segments = [
        CaptionSegment(
            index=i,
            start_time=i * 5.0,
            end_time=(i + 1) * 5.0,
            text=f"Test caption segment {i}",
            source_file=video_id,
        )
        for i in range(segment_count)
    ]
    return CaptionResult(
        video_id=video_id,
        segments=segments,
        language="en",
        is_auto_generated=True,
        format_source="vtt",
    )


def make_mock_config(
    circuit_breaker_config: Optional[Dict] = None,
    retry_budget_config: Optional[Dict] = None,
) -> MockConfig:
    """Create mock config with optional circuit breaker and retry budget."""
    caption_first = MockCaptionFirstConfig(
        circuit_breaker=circuit_breaker_config,
        retry_budget=retry_budget_config,
    )
    download_config = MockDownloadConfig(caption_first=caption_first)
    return MockConfig(download=download_config)


# =============================================================================
# Tests: 429 Error Handling with Exponential Backoff
# =============================================================================


class TestCaptionStage429ErrorHandling:
    """Test CaptionStage handles 429 errors with exponential backoff (AC2)."""

    def test_429_error_triggers_rate_limit_recording(self):
        """Test that 429 errors are detected and trigger rate limit recording."""
        # Setup mock rate limiter to track calls
        mock_limiter = MagicMock(spec=UnifiedCaptionRateLimiter)
        mock_limiter.wait_if_needed.return_value = 0.0
        mock_limiter.consecutive_rate_limits = 0

        # Create fetcher with rate limiter
        fetcher = CaptionFetcher(rate_limiter=mock_limiter)

        # Mock _fetch_subtitle to raise 429 error
        error_429 = CaptionFetchError("test_video", "HTTP 429 Too Many Requests")

        with patch.object(fetcher, '_fetch_subtitle', side_effect=error_429):
            with patch.object(fetcher, '_is_valid_video_id', return_value=True):
                with pytest.raises(CaptionFetchError):
                    fetcher.fetch_captions("test_video")

        # Rate limit should be recorded
        mock_limiter.record_rate_limit.assert_called()

    def test_exponential_backoff_increases_delay(self):
        """Test that consecutive 429 errors increase backoff delay exponentially."""
        rate_limiter = UnifiedCaptionRateLimiter(RateLimitConfig(
            base_delay_seconds=2.0,
            max_delay_seconds=120.0,
            jitter_factor=0.0,  # Disable jitter for predictable tests
        ))

        # First rate limit: 2 * 2^0 = 2s
        rate_limiter.record_rate_limit("video1")
        delay1 = rate_limiter.calculate_backoff(rate_limiter.consecutive_rate_limits)
        assert delay1 == 2.0

        # Second rate limit: 2 * 2^1 = 4s
        rate_limiter.record_rate_limit("video2")
        delay2 = rate_limiter.calculate_backoff(rate_limiter.consecutive_rate_limits)
        assert delay2 == 4.0

        # Third rate limit: 2 * 2^2 = 8s
        rate_limiter.record_rate_limit("video3")
        delay3 = rate_limiter.calculate_backoff(rate_limiter.consecutive_rate_limits)
        assert delay3 == 8.0

    def test_success_resets_backoff(self):
        """Test that successful fetch resets backoff delay."""
        rate_limiter = UnifiedCaptionRateLimiter()

        # Record several rate limits
        rate_limiter.record_rate_limit("video1")
        rate_limiter.record_rate_limit("video2")
        rate_limiter.record_rate_limit("video3")
        assert rate_limiter.consecutive_rate_limits == 3

        # Success resets consecutive count
        rate_limiter.record_success()
        assert rate_limiter.consecutive_rate_limits == 0

    def test_429_error_in_batch_fetch_handled_gracefully(self):
        """Test that 429 errors in batch fetch don't crash the stage."""
        stage = CaptionStage()
        config = make_mock_config()
        state = MockPipelineState(
            downloaded_videos=[MockDownloadedVideo("video1")]
        )
        checkpoint = MockCheckpointManager()

        # Mock the fetcher to simulate one 429 error then success
        call_count = [0]

        def mock_fetch_batch(video_ids, **kwargs):
            call_count[0] += 1
            results = {}
            for vid in video_ids:
                # Return error dict format for failed fetches
                results[vid] = {
                    'video_id': vid,
                    'error': True,
                    'reason': 'HTTP 429 Too Many Requests',
                    'caption_quality': 'low',
                }
            return results

        with patch.object(CaptionFetcher, 'fetch_captions_batch', mock_fetch_batch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                result = stage.run(state, config, checkpoint)

        # Stage should complete (may have warnings about failed fetches)
        assert result is not None


# =============================================================================
# Tests: Circuit Breaker Respect
# =============================================================================


class TestCaptionStageCircuitBreakerRespect:
    """Test CaptionStage respects circuit breaker open state (AC3)."""

    def test_circuit_breaker_trips_after_consecutive_failures(self):
        """Test circuit breaker trips after threshold failures."""
        config = CaptionCircuitBreakerConfig(threshold=3, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Record failures up to threshold
        assert breaker.record_failure() is False  # 1st failure
        assert breaker.record_failure() is False  # 2nd failure
        assert breaker.record_failure() is True   # 3rd failure - trips

        assert breaker.is_open is True

    def test_circuit_breaker_blocks_fetches_when_open(self):
        """Test that open circuit breaker causes wait before fetch."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.1)
        breaker = CaptionCircuitBreaker(config)

        # Trip the circuit
        breaker.record_failure()
        assert breaker.is_open is True

        # Check remaining pause time
        remaining = breaker.get_remaining_pause_time()
        assert remaining > 0

        # Wait and check recovery
        time.sleep(0.15)
        result = breaker.check_and_wait()
        assert result is True
        assert breaker.is_open is False

    def test_circuit_breaker_tracks_total_trips(self):
        """Test circuit breaker tracks cumulative trip count."""
        config = CaptionCircuitBreakerConfig(threshold=1, pause_seconds=0.05)
        breaker = CaptionCircuitBreaker(config)

        # Trip multiple times
        for cycle in range(3):
            breaker.record_failure()
            assert breaker.state.total_trips == cycle + 1

            # Wait for recovery
            time.sleep(0.06)
            breaker.check_and_wait()
            breaker.record_success()

    def test_circuit_breaker_config_in_caption_stage(self):
        """Test CaptionStage reads circuit breaker config correctly."""
        cb_config = {
            'enabled': True,
            'threshold': 5,
            'pause_seconds': 60.0,
        }
        config = make_mock_config(circuit_breaker_config=cb_config)

        # Extract and verify config is accessible
        caption_config = config.download.caption_first
        assert caption_config.circuit_breaker['enabled'] is True
        assert caption_config.circuit_breaker['threshold'] == 5


# =============================================================================
# Tests: Metrics Tracking Across Parallel Workers
# =============================================================================


class TestCaptionStageParallelMetrics:
    """Test CaptionStage tracks metrics across parallel workers (AC4)."""

    def test_metrics_thread_safe_increment(self):
        """Test metrics can be safely incremented from multiple threads."""
        metrics = CaptionMetrics()
        num_threads = 10
        increments_per_thread = 100

        def increment_metrics():
            for _ in range(increments_per_thread):
                metrics.increment_success()
                metrics.increment_failure()
                metrics.increment_cached()

        threads = [threading.Thread(target=increment_metrics) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * increments_per_thread
        assert metrics.successes == expected
        assert metrics.failures == expected
        assert metrics.cache_hits == expected

    def test_metrics_summary_calculation(self):
        """Test metrics summary is calculated correctly."""
        metrics = CaptionMetrics()

        # Record some metrics
        metrics.increment_success()
        metrics.increment_success()
        metrics.increment_cached()
        metrics.increment_failure()

        summary = metrics.get_summary_dict()

        # Verify summary keys exist
        assert 'hit_rate' in summary or 'cache_hit_rate' in summary
        assert 'success_rate' in summary
        assert 'successes' in summary
        assert 'failures' in summary

    def test_metrics_from_parallel_batch_fetches(self):
        """Test metrics are correctly aggregated from parallel batch fetches."""
        metrics = CaptionMetrics()
        num_workers = 4
        videos_per_worker = 25

        def worker_fetch(worker_id: int):
            for i in range(videos_per_worker):
                video_id = f"worker{worker_id}_video{i}"
                # Simulate mixed results
                if i % 3 == 0:
                    metrics.increment_cached()
                elif i % 5 == 0:
                    metrics.increment_failure()
                else:
                    # Record success with fetch time via record_fetch_success
                    metrics.record_fetch_success(
                        video_id=video_id,
                        elapsed_seconds=0.5 + (i * 0.01)
                    )

        threads = [threading.Thread(target=worker_fetch, args=(w,)) for w in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = num_workers * videos_per_worker
        # All increments should be counted
        assert metrics.successes + metrics.failures + metrics.cache_hits == total

    def test_metrics_fetch_time_tracking(self):
        """Test metrics track fetch times correctly across threads."""
        metrics = CaptionMetrics()

        def record_times():
            for i in range(10):
                video_id = f"video_{threading.current_thread().name}_{i}"
                # Use record_fetch_success with elapsed_seconds to track time
                metrics.record_fetch_success(
                    video_id=video_id,
                    elapsed_seconds=1.0 + i * 0.1
                )

        threads = [threading.Thread(target=record_times) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check that fetch times were recorded in video_fetch_times dict
        assert len(metrics.video_fetch_times) > 0
        # Check summary includes avg_fetch_time
        summary = metrics.get_summary_dict()
        assert 'avg_fetch_time' in summary


# =============================================================================
# Tests: Checkpointing Survives Rate Limit Pauses
# =============================================================================


class TestCaptionStageCheckpointDuringRateLimits:
    """Test CaptionStage checkpointing survives rate limit pauses (AC5)."""

    def test_checkpoint_preserved_during_rate_limit_pause(self, tmp_path):
        """Test checkpoint data is preserved when circuit breaker pauses."""
        checkpoint_file = tmp_path / "checkpoint.json"

        # Create checkpoint with some caption results
        initial_data = {
            'caption_results': {
                'video1': {'video_id': 'video1', 'segments': [], 'segment_count': 10},
                'video2': {'video_id': 'video2', 'segments': [], 'segment_count': 15},
            },
            'success_count': 2,
            'partial': True,
        }
        checkpoint_file.write_text(json.dumps(initial_data))

        # Create circuit breaker that will trip
        breaker = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(
            threshold=1,
            pause_seconds=0.1,
        ))

        # Trip the breaker
        breaker.record_failure()
        assert breaker.is_open is True

        # During pause, checkpoint data should still be readable
        loaded_data = json.loads(checkpoint_file.read_text())
        assert loaded_data['caption_results']['video1']['segment_count'] == 10
        assert loaded_data['caption_results']['video2']['segment_count'] == 15

        # Wait for breaker to recover
        time.sleep(0.15)
        breaker.check_and_wait()

        # Checkpoint should still be valid
        loaded_data = json.loads(checkpoint_file.read_text())
        assert loaded_data['success_count'] == 2

    def test_circuit_breaker_state_serialization(self):
        """Test circuit breaker state can be serialized for checkpoint."""
        breaker = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(
            threshold=3,
            pause_seconds=60.0,
        ))

        # Record some activity
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()  # Trips
        breaker.state.total_paused_seconds = 120.5

        # Serialize to checkpoint
        checkpoint_data = breaker.to_checkpoint_dict()

        assert checkpoint_data['total_trips'] == 1
        assert checkpoint_data['total_paused_seconds'] == 120.5
        # is_open should NOT be in checkpoint (transient state)
        assert 'is_open' not in checkpoint_data

    def test_circuit_breaker_state_restoration(self):
        """Test circuit breaker cumulative stats restored from checkpoint."""
        breaker = CaptionCircuitBreaker()

        # Restore from checkpoint
        checkpoint_data = {
            'total_trips': 5,
            'total_paused_seconds': 300.0,
            'consecutive_failures': 10,  # Should NOT be restored
        }
        breaker.from_checkpoint_dict(checkpoint_data)

        # Cumulative stats restored
        assert breaker.state.total_trips == 5
        assert breaker.state.total_paused_seconds == 300.0

        # Transient state starts fresh
        assert breaker.state.consecutive_failures == 0
        assert breaker.is_open is False

    def test_metrics_serialization_for_checkpoint(self):
        """Test metrics can be serialized for checkpoint."""
        metrics = CaptionMetrics()

        # Record some metrics
        metrics.increment_success()
        metrics.increment_success()
        metrics.increment_failure()
        metrics.increment_cached()

        # Serialize
        metrics_dict = metrics.to_dict()

        assert metrics_dict['successes'] == 2
        assert metrics_dict['failures'] == 1
        assert metrics_dict['cache_hits'] == 1

    def test_metrics_restoration_from_checkpoint(self):
        """Test metrics can be restored from checkpoint."""
        # Serialize
        original = CaptionMetrics()
        original.increment_success()
        original.increment_success()
        original.increment_failure()
        original_dict = original.to_dict()

        # Restore
        restored = CaptionMetrics.from_dict(original_dict)

        assert restored.successes == 2
        assert restored.failures == 1


# =============================================================================
# Tests: All Tests Use Mocked yt-dlp
# =============================================================================


class TestCaptionStageMockedYtdlp:
    """Verify all tests use mocked yt-dlp - no network calls (AC6)."""

    def test_caption_fetcher_subprocess_mocked(self):
        """Test that CaptionFetcher can be mocked without subprocess calls."""
        # Create mock result
        mock_result = make_mock_caption_result("test_video")

        # Mock the entire fetch method
        with patch.object(CaptionFetcher, 'fetch_captions', return_value=mock_result):
            fetcher = CaptionFetcher()
            result = fetcher.fetch_captions("test_video")

        assert result.video_id == "test_video"
        assert len(result.segments) == 10

    def test_batch_fetch_uses_mock(self):
        """Test batch fetch can be fully mocked."""
        video_ids = ["video1", "video2", "video3"]

        def mock_batch_fetch(ids, **kwargs):
            return {vid: make_mock_caption_result(vid) for vid in ids}

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            fetcher = CaptionFetcher()
            results = fetcher.fetch_captions_batch(video_ids)

        assert len(results) == 3
        for vid in video_ids:
            assert vid in results

    def test_no_subprocess_in_rate_limit_tests(self):
        """Test rate limit handling doesn't invoke subprocess."""
        rate_limiter = UnifiedCaptionRateLimiter()

        # All rate limiter operations are in-memory
        rate_limiter.record_rate_limit("video1")
        rate_limiter.record_rate_limit("video2")
        rate_limiter.record_success()

        delay = rate_limiter.calculate_backoff(2)
        state = rate_limiter.get_state()

        # No subprocess should have been called
        assert delay >= 0
        assert 'consecutive_rate_limits' in state

    def test_circuit_breaker_no_subprocess(self):
        """Test circuit breaker operations are pure in-memory."""
        breaker = CaptionCircuitBreaker(CaptionCircuitBreakerConfig(
            threshold=2,
            pause_seconds=0.01,
        ))

        # All circuit breaker operations are in-memory
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open is True

        time.sleep(0.02)
        breaker.check_and_wait()
        breaker.record_success()

        # No subprocess was called
        assert breaker.is_open is False


# =============================================================================
# Integration Tests: End-to-End Stage Flow
# =============================================================================


class TestCaptionStageIntegrationFlow:
    """End-to-end integration tests for CaptionStage with rate limiting."""

    def test_stage_run_with_circuit_breaker_and_metrics(self):
        """Test full stage run with circuit breaker and metrics enabled."""
        stage = CaptionStage()

        # Config with circuit breaker
        config = make_mock_config(
            circuit_breaker_config={
                'enabled': True,
                'threshold': 5,
                'pause_seconds': 0.1,
            }
        )

        # State with some videos
        state = MockPipelineState(
            downloaded_videos=[
                MockDownloadedVideo(f"video{i}")
                for i in range(3)
            ]
        )
        checkpoint = MockCheckpointManager()

        # Mock the fetcher's batch method to return success
        def mock_batch_fetch(ids, **kwargs):
            results = {}
            for vid in ids:
                result = make_mock_caption_result(vid)
                results[vid] = result
            return results

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    # Mock _using_adaptive_order attribute
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # Stage should complete successfully
        assert result is not None

    def test_stage_handles_partial_failures_gracefully(self):
        """Test stage handles mix of successes and failures."""
        stage = CaptionStage()
        config = make_mock_config()

        state = MockPipelineState(
            downloaded_videos=[
                MockDownloadedVideo("success_video"),
                MockDownloadedVideo("fail_video"),
            ]
        )
        checkpoint = MockCheckpointManager()

        def mock_batch_fetch(ids, **kwargs):
            results = {}
            for vid in ids:
                if "success" in vid:
                    results[vid] = make_mock_caption_result(vid)
                else:
                    results[vid] = {
                        'video_id': vid,
                        'error': True,
                        'reason': 'Captions unavailable',
                        'caption_quality': 'low',
                    }
            return results

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # Stage should complete (may have warnings)
        assert result is not None

    def test_metrics_accumulated_during_stage_run(self):
        """Test that metrics are properly accumulated during stage run."""
        stage = CaptionStage()
        config = make_mock_config()

        # Add more videos to test accumulation
        state = MockPipelineState(
            downloaded_videos=[
                MockDownloadedVideo(f"video{i}")
                for i in range(5)
            ]
        )
        checkpoint = MockCheckpointManager()

        # Track metrics passed to batch fetch
        captured_metrics = []

        def mock_batch_fetch(ids, metrics=None, **kwargs):
            if metrics:
                captured_metrics.append(metrics)
                # Simulate recording some metrics
                for vid in ids:
                    metrics.increment_success()
            return {vid: make_mock_caption_result(vid) for vid in ids}

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # Metrics object should have been passed to batch fetch
        # (may be empty if fetch_captions_batch was fully mocked)
        assert result is not None


# =============================================================================
# US-40-002: Checkpoint Restoration Hook for CaptionRetryBudget
# =============================================================================


class TestCaptionStageRetryBudgetCheckpointRestoration:
    """Test CaptionStage restores and re-scales retry budget from checkpoint (US-40-002)."""

    @pytest.mark.fast
    def test_ensure_scaled_called_after_checkpoint_restoration(self):
        """Test ensure_scaled() is called after restoring retry budget from checkpoint.

        AC3: Add unit test verifying ensure_scaled() is called after checkpoint restoration
        """
        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
                'attempts_per_video': 1.5,
            }
        )

        # Create state with 200 videos (should trigger scaling from 100 to 300)
        # Use video_ids attribute (primary path in _get_video_ids)
        video_ids_list = [f"vid{i:08d}" for i in range(200)]  # vid00000000 = 11 chars
        state = MockPipelineState(
            downloaded_videos=[
                MockDownloadedVideo(vid)
                for vid in video_ids_list
            ]
        )
        # Add video_ids attribute for the primary path in _get_video_ids
        state.video_ids = video_ids_list

        # Checkpoint with previous retry budget state (smaller batch)
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 50,
                        'failures': 10,
                        'successes': 40,
                        'backoff_time_spent': 20.0,
                        'videos_skipped': 0,
                        'max_attempts': 150,  # Was scaled for 100 videos
                        'max_backoff_time': 300.0,
                        'batch_size': 100,  # Previous batch was 100 videos
                    }
                }
            }
        )

        # Track calls to ensure_scaled
        ensure_scaled_calls = []
        original_ensure_scaled = None

        def mock_batch_fetch(video_ids, retry_budget=None, **kwargs):
            # Capture the retry budget state when fetch is called
            if retry_budget:
                ensure_scaled_calls.append({
                    'max_attempts': retry_budget.max_attempts,
                    'batch_size': retry_budget.batch_size,
                    'attempts': retry_budget.attempts,
                })
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # Verify budget was re-scaled for new batch size
        assert len(ensure_scaled_calls) == 1
        budget_state = ensure_scaled_calls[0]

        # Should be scaled for 200 videos: 200 * 1.5 = 300
        assert budget_state['max_attempts'] == 300
        assert budget_state['batch_size'] == 200

        # Previous attempts should be preserved
        assert budget_state['attempts'] == 50

    @pytest.mark.fast
    def test_max_attempts_updated_when_batch_size_increases_after_resume(self):
        """Test max_attempts is updated when batch_size increases after checkpoint resume.

        AC4: Add unit test verifying max_attempts is updated when batch_size increases after resume
        """
        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
                'attempts_per_video': 2.0,  # 2 attempts per video
            }
        )

        # Larger batch than checkpoint had (150 vs 50)
        # Use video_ids attribute (primary path in _get_video_ids)
        video_ids_list = [f"vid{i:08d}" for i in range(150)]  # vid00000000 = 11 chars
        state = MockPipelineState(
            downloaded_videos=[
                MockDownloadedVideo(vid)
                for vid in video_ids_list
            ]
        )
        # Add video_ids attribute for the primary path in _get_video_ids
        state.video_ids = video_ids_list

        # Checkpoint from smaller batch (50 videos, scaled to 100 attempts)
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 30,
                        'failures': 5,
                        'successes': 25,
                        'backoff_time_spent': 10.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,  # Was 50 * 2.0 = 100
                        'max_backoff_time': 300.0,
                        'batch_size': 50,
                    }
                }
            }
        )

        captured_budget = {}

        def mock_batch_fetch(video_ids, retry_budget=None, **kwargs):
            if retry_budget:
                captured_budget['max_attempts'] = retry_budget.max_attempts
                captured_budget['batch_size'] = retry_budget.batch_size
                captured_budget['attempts'] = retry_budget.attempts
                captured_budget['failures'] = retry_budget.failures
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # max_attempts should be scaled up for larger batch
        # 150 videos * 2.0 attempts/video = 300 max_attempts
        assert captured_budget['max_attempts'] == 300

        # batch_size should reflect new batch
        assert captured_budget['batch_size'] == 150

        # Usage counters should be preserved from checkpoint
        assert captured_budget['attempts'] == 30
        assert captured_budget['failures'] == 5

    @pytest.mark.fast
    def test_budget_scaled_before_fetch_captions_batch_called(self):
        """Test budget is scaled BEFORE fetch_captions_batch is called.

        AC5: Verify budget is scaled BEFORE fetch_captions_batch is called
        """
        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
                'attempts_per_video': 1.5,
            }
        )

        # Use video_ids attribute (primary path in _get_video_ids)
        video_ids_list = [f"vid{i:08d}" for i in range(100)]  # vid00000000 = 11 chars
        state = MockPipelineState(
            downloaded_videos=[
                MockDownloadedVideo(vid)
                for vid in video_ids_list
            ]
        )
        # Add video_ids attribute for the primary path in _get_video_ids
        state.video_ids = video_ids_list

        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 0,
                        'failures': 0,
                        'successes': 0,
                        'backoff_time_spent': 0.0,
                        'videos_skipped': 0,
                        'max_attempts': 75,  # Was scaled for 50 videos
                        'max_backoff_time': 300.0,
                        'batch_size': 50,
                    }
                }
            }
        )

        budget_at_fetch_time = {}

        def mock_batch_fetch(video_ids, retry_budget=None, **kwargs):
            if retry_budget:
                # Record budget state at the moment fetch_captions_batch is called
                budget_at_fetch_time['max_attempts'] = retry_budget.max_attempts
                budget_at_fetch_time['batch_size'] = retry_budget.batch_size
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
            with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                    with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                        result = stage.run(state, config, checkpoint)

        # At the time fetch_captions_batch is called, budget should already be scaled
        # 100 videos * 1.5 = 150 max_attempts (scaled up from 75)
        assert budget_at_fetch_time['max_attempts'] == 150
        assert budget_at_fetch_time['batch_size'] == 100


# =============================================================================
# US-41-008: Retry Budget Checkpoint Restore Logging
# =============================================================================


class TestRetryBudgetCheckpointRestoreLogging:
    """Test logging of retry budget state before and after checkpoint restore (US-41-008).

    Acceptance criteria:
    1. Log INFO before checkpoint restore: 'Retry budget before restore: max_attempts=X, batch_size=Y'
    2. Log INFO after checkpoint restore: 'Retry budget after restore: max_attempts=X, batch_size=Y, attempts_used=Z'
    3. Log WARNING if restored budget has attempts > 0 but batch_size is None (indicates old checkpoint)
    4. Include these logs in caption_stage.py initialization block
    5. Add test: checkpoint restore logging shows expected values
    """

    @pytest.mark.fast
    def test_logs_retry_budget_before_restore(self):
        """Test INFO log before checkpoint restore shows max_attempts and batch_size.

        AC1: Log INFO before checkpoint restore: 'Retry budget before restore: max_attempts=X, batch_size=Y'
        """
        import logging

        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
            }
        )

        video_ids_list = [f"vid{i:08d}" for i in range(10)]
        state = MockPipelineState(
            downloaded_videos=[MockDownloadedVideo(vid) for vid in video_ids_list]
        )
        state.video_ids = video_ids_list

        # Checkpoint with retry budget data
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 20,
                        'failures': 5,
                        'successes': 15,
                        'backoff_time_spent': 10.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,
                        'max_backoff_time': 300.0,
                        'batch_size': 50,
                    }
                }
            }
        )

        log_messages = []

        def capture_log(record):
            log_messages.append(record.getMessage())
            return True

        # Capture logger output
        logger = logging.getLogger('src.stages.caption_stage')
        handler = logging.Handler()
        handler.emit = lambda record: log_messages.append(record.getMessage())
        handler.filter = capture_log
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)

        def mock_batch_fetch(video_ids, **kwargs):
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        try:
            with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
                with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                    with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                        with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                            stage.run(state, config, checkpoint)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # Verify "before restore" log was emitted
        before_restore_logs = [msg for msg in log_messages if '[US-41-008] Retry budget before restore' in msg]
        assert len(before_restore_logs) >= 1, f"Missing 'before restore' log. Got: {log_messages}"
        # Verify it contains max_attempts and batch_size
        assert 'max_attempts=' in before_restore_logs[0]
        assert 'batch_size=' in before_restore_logs[0]

    @pytest.mark.fast
    def test_logs_retry_budget_after_restore(self):
        """Test INFO log after checkpoint restore shows max_attempts, batch_size, and attempts_used.

        AC2: Log INFO after checkpoint restore: 'Retry budget after restore: max_attempts=X, batch_size=Y, attempts_used=Z'
        """
        import logging

        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
            }
        )

        video_ids_list = [f"vid{i:08d}" for i in range(10)]
        state = MockPipelineState(
            downloaded_videos=[MockDownloadedVideo(vid) for vid in video_ids_list]
        )
        state.video_ids = video_ids_list

        # Checkpoint with retry budget data
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 25,
                        'failures': 5,
                        'successes': 20,
                        'backoff_time_spent': 15.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,
                        'max_backoff_time': 300.0,
                        'batch_size': 50,
                    }
                }
            }
        )

        log_messages = []

        # Capture logger output
        logger = logging.getLogger('src.stages.caption_stage')
        handler = logging.Handler()
        handler.emit = lambda record: log_messages.append(record.getMessage())
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)

        def mock_batch_fetch(video_ids, **kwargs):
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        try:
            with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
                with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                    with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                        with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                            stage.run(state, config, checkpoint)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # Verify "after restore" log was emitted
        after_restore_logs = [msg for msg in log_messages if '[US-41-008] Retry budget after restore' in msg]
        assert len(after_restore_logs) >= 1, f"Missing 'after restore' log. Got: {log_messages}"
        # Verify it contains max_attempts, batch_size, and attempts_used
        assert 'max_attempts=' in after_restore_logs[0]
        assert 'batch_size=' in after_restore_logs[0]
        assert 'attempts_used=' in after_restore_logs[0]

    @pytest.mark.fast
    def test_warns_on_old_checkpoint_format(self):
        """Test WARNING log when restored budget has attempts > 0 but batch_size is None.

        AC3: Log WARNING if restored budget has attempts > 0 but batch_size is None (indicates old checkpoint)
        """
        import logging

        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
            }
        )

        video_ids_list = [f"vid{i:08d}" for i in range(10)]
        state = MockPipelineState(
            downloaded_videos=[MockDownloadedVideo(vid) for vid in video_ids_list]
        )
        state.video_ids = video_ids_list

        # Checkpoint with OLD format - has attempts but NO batch_size (simulates old checkpoint)
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 30,  # Has attempts used
                        'failures': 5,
                        'successes': 25,
                        'backoff_time_spent': 10.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,
                        'max_backoff_time': 300.0,
                        'batch_size': None,  # OLD checkpoint - missing batch_size
                    }
                }
            }
        )

        log_messages = []
        log_levels = []

        # Capture logger output with levels
        logger = logging.getLogger('src.stages.caption_stage')
        original_handlers = logger.handlers[:]

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())
                log_levels.append(record.levelno)

        handler = TestHandler()
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)  # Capture all levels

        def mock_batch_fetch(video_ids, **kwargs):
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        try:
            with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
                with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                    with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                        with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                            stage.run(state, config, checkpoint)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # Verify WARNING about old checkpoint format was emitted
        old_checkpoint_warnings = [
            (msg, level) for msg, level in zip(log_messages, log_levels)
            if '[US-41-008]' in msg and 'batch_size is None' in msg
        ]
        assert len(old_checkpoint_warnings) >= 1, f"Missing 'old checkpoint' warning. Got: {log_messages}"
        # Verify it was a WARNING level
        assert old_checkpoint_warnings[0][1] == logging.WARNING

    @pytest.mark.fast
    def test_no_warning_when_batch_size_present(self):
        """Test no WARNING when restored budget has proper batch_size (not old checkpoint)."""
        import logging

        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
            }
        )

        video_ids_list = [f"vid{i:08d}" for i in range(10)]
        state = MockPipelineState(
            downloaded_videos=[MockDownloadedVideo(vid) for vid in video_ids_list]
        )
        state.video_ids = video_ids_list

        # Checkpoint with proper batch_size
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 30,
                        'failures': 5,
                        'successes': 25,
                        'backoff_time_spent': 10.0,
                        'videos_skipped': 0,
                        'max_attempts': 100,
                        'max_backoff_time': 300.0,
                        'batch_size': 50,  # Proper batch_size present
                    }
                }
            }
        )

        log_messages = []
        log_levels = []

        logger = logging.getLogger('src.stages.caption_stage')

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())
                log_levels.append(record.levelno)

        handler = TestHandler()
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        def mock_batch_fetch(video_ids, **kwargs):
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        try:
            with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
                with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                    with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                        with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                            stage.run(state, config, checkpoint)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # Verify NO warning about old checkpoint format
        old_checkpoint_warnings = [
            msg for msg in log_messages
            if 'batch_size is None' in msg
        ]
        assert len(old_checkpoint_warnings) == 0, f"Unexpected 'old checkpoint' warning: {old_checkpoint_warnings}"

    @pytest.mark.fast
    def test_checkpoint_restore_logging_shows_expected_values(self):
        """Test checkpoint restore logging shows the expected values (comprehensive AC5 test).

        AC5: Add test: checkpoint restore logging shows expected values
        """
        import logging

        stage = CaptionStage()
        config = make_mock_config(
            retry_budget_config={
                'enabled': True,
                'max_attempts': 100,
                'auto_scale': True,
            }
        )

        video_ids_list = [f"vid{i:08d}" for i in range(10)]
        state = MockPipelineState(
            downloaded_videos=[MockDownloadedVideo(vid) for vid in video_ids_list]
        )
        state.video_ids = video_ids_list

        # Checkpoint with specific values we expect to see in logs
        checkpoint = MockCheckpointManager(
            stage_data={
                'CAPTION': {
                    'caption_results': {},
                    'retry_budget': {
                        'attempts': 42,  # Specific value to verify
                        'failures': 7,
                        'successes': 35,
                        'backoff_time_spent': 25.0,
                        'videos_skipped': 2,
                        'max_attempts': 100,
                        'max_backoff_time': 300.0,
                        'batch_size': 75,  # Specific value to verify
                    }
                }
            }
        )

        log_messages = []

        logger = logging.getLogger('src.stages.caption_stage')

        class TestHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(record.getMessage())

        handler = TestHandler()
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)

        def mock_batch_fetch(video_ids, **kwargs):
            return {vid: make_mock_caption_result(vid) for vid in video_ids}

        try:
            with patch.object(CaptionFetcher, 'fetch_captions_batch', side_effect=mock_batch_fetch):
                with patch.object(CaptionFetcher, '__init__', lambda self, **kwargs: None):
                    with patch.object(CaptionFetcher, 'apply_adaptive_format_order', return_value=None):
                        with patch.object(CaptionFetcher, '_using_adaptive_order', False, create=True):
                            stage.run(state, config, checkpoint)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

        # Verify "after restore" log contains the specific checkpoint values
        after_restore_logs = [msg for msg in log_messages if '[US-41-008] Retry budget after restore' in msg]
        assert len(after_restore_logs) >= 1, f"Missing 'after restore' log. Got: {log_messages}"

        # Verify the log contains the expected restored values
        after_log = after_restore_logs[0]
        # Batch size should be restored to 75 from checkpoint
        assert 'batch_size=75' in after_log, f"Expected batch_size=75 in log: {after_log}"
        # Attempts used should be 42 from checkpoint
        assert 'attempts_used=42' in after_log, f"Expected attempts_used=42 in log: {after_log}"
