"""Tests for GlobalRateLimitCoordinator integration with VideoDownloader (US-35-002).

Tests that VideoDownloader properly acquires and releases rate limit slots
when downloading videos through _download_by_ids.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.rate_limit.coordinator import GlobalRateLimitCoordinator, RateLimitConfig


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_coordinator():
    """Reset the singleton coordinator before each test."""
    GlobalRateLimitCoordinator.reset_instance()
    yield
    GlobalRateLimitCoordinator.reset_instance()


@pytest.fixture
def coordinator():
    """Create a fresh coordinator with test config."""
    config = RateLimitConfig(
        enabled=True,
        slots_per_second=10.0,  # Fast for tests
        burst_size=5
    )
    return GlobalRateLimitCoordinator(config)


@pytest.fixture
def mock_config():
    """Create a mock config for VideoDownloader."""
    config = MagicMock()
    config.cache_dir = "/tmp/test_cache"
    config.downloaded_videos_dir = "/tmp/test_downloads"
    config.download = MagicMock()
    config.download.max_keyword_len = 8
    config.download.max_filename_len = 10
    config.download.cookies_from_browser = ''
    config.download.cookie_rotation = None
    config.download.impersonation = None
    config.download.vpn = None
    config.download.speed_tracking = None
    config.download.circuit_breaker = None
    config.download.batch_retry = None
    config.download.rate_limit = MagicMock()
    config.download.rate_limit.per_tier_isolation = False
    config.download.rate_limit.global_ = MagicMock()
    config.download.rate_limit.global_.enabled = True
    config.download.rate_limit.global_.slots_per_second = 10.0
    config.download.rate_limit.global_.burst_size = 5
    # Use getattr mock for safe attribute access
    type(config.download.rate_limit).global_ = PropertyMock(return_value=None)
    config.download.rate_limit_budget = None
    config.download.llm_title_filter = None
    return config


# =============================================================================
# Test: Slot Acquisition/Release
# =============================================================================


class TestCoordinatorSlotManagement:
    """Test basic slot acquisition and release."""

    def test_acquire_slot_success(self, coordinator):
        """Should acquire slot when tokens available."""
        acquired = coordinator.acquire_slot('download', timeout=1.0)
        assert acquired is True
        assert coordinator.get_active_slots('download') == 1

    def test_release_slot(self, coordinator):
        """Should release slot after download."""
        coordinator.acquire_slot('download', timeout=1.0)
        coordinator.release_slot('download')
        assert coordinator.get_active_slots('download') == 0

    def test_slot_metrics_tracked(self, coordinator):
        """Should track slot acquisition metrics."""
        coordinator.acquire_slot('download', timeout=1.0)
        coordinator.release_slot('download')

        metrics = coordinator.get_metrics('download')
        assert metrics['total_acquired'] == 1
        assert metrics['total_released'] == 1
        assert metrics['active_slots'] == 0

    def test_multiple_slots_within_burst(self, coordinator):
        """Should allow burst_size slots immediately."""
        for i in range(5):
            acquired = coordinator.acquire_slot('download', timeout=0.1, block=False)
            assert acquired is True, f"Slot {i+1} should be acquired"

        # 6th slot should fail (burst exhausted, no time to refill)
        acquired = coordinator.acquire_slot('download', timeout=0.01, block=False)
        assert acquired is False

    def test_disabled_coordinator_always_allows(self):
        """Disabled coordinator should always return True."""
        config = RateLimitConfig(enabled=False)
        coord = GlobalRateLimitCoordinator(config)

        for _ in range(100):
            assert coord.acquire_slot('download', block=False) is True


# =============================================================================
# Test: VideoDownloader Integration
# =============================================================================


class TestVideoDownloaderSlotIntegration:
    """Test that VideoDownloader uses coordinator for downloads."""

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    def test_acquire_download_slot_method(
        self,
        mock_audio, mock_search, mock_speech, mock_title,
        mock_transcode, mock_checkpoint
    ):
        """VideoDownloader.acquire_download_slot should use coordinator."""
        from src.downloader.core import VideoDownloader

        # Setup mocks
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}, 'medium': {}, 'long': {}, 'longer': {}}

        with patch('src.downloader.core.get_config') as mock_get_config:
            config = MagicMock()
            config.cache_dir = "/tmp/test"
            config.downloaded_videos_dir = "/tmp/test"
            config.download = MagicMock()
            config.download.cookies_from_browser = ''
            config.download.cookie_rotation = None
            config.download.impersonation = None
            config.download.vpn = None
            config.download.speed_tracking = None
            config.download.circuit_breaker = None
            config.download.batch_retry = None
            config.download.rate_limit = None
            config.download.rate_limit_budget = None
            mock_get_config.return_value = config

            downloader = VideoDownloader(config)

            # Test slot acquisition
            acquired = downloader.acquire_download_slot(timeout=1.0)
            assert acquired is True

            # Verify coordinator state
            assert downloader.rate_limit_coordinator.get_active_slots('download') == 1

            # Test slot release
            downloader.release_download_slot()
            assert downloader.rate_limit_coordinator.get_active_slots('download') == 0

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    def test_slot_timeout_records_metric(
        self,
        mock_audio, mock_search, mock_speech, mock_title,
        mock_transcode, mock_checkpoint
    ):
        """Failed slot acquisition should record metric."""
        from src.downloader.core import VideoDownloader

        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}, 'medium': {}, 'long': {}, 'longer': {}}

        with patch('src.downloader.core.get_config') as mock_get_config:
            config = MagicMock()
            config.cache_dir = "/tmp/test"
            config.downloaded_videos_dir = "/tmp/test"
            config.download = MagicMock()
            config.download.cookies_from_browser = ''
            config.download.cookie_rotation = None
            config.download.impersonation = None
            config.download.vpn = None
            config.download.speed_tracking = None
            config.download.circuit_breaker = None
            config.download.batch_retry = None
            config.download.rate_limit = MagicMock()
            config.download.rate_limit.per_tier_isolation = False
            config.download.rate_limit.global_ = MagicMock()
            config.download.rate_limit.global_.enabled = True
            config.download.rate_limit.global_.slots_per_second = 0.01  # Very slow
            config.download.rate_limit.global_.burst_size = 1
            config.download.rate_limit_budget = None
            mock_get_config.return_value = config

            # Reset singleton to get fresh coordinator with config
            GlobalRateLimitCoordinator.reset_instance()

            downloader = VideoDownloader(config)

            # Exhaust the only slot
            downloader.acquire_download_slot(timeout=0.1)

            # Second acquisition should fail quickly
            acquired = downloader.acquire_download_slot(timeout=0.05)
            assert acquired is False

            # Check metric recorded
            assert downloader.rate_limit_metrics.slot_timeouts >= 1


class TestDownloadByIdsSlotUsage:
    """Test that _download_by_ids acquires/releases slots."""

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @patch('src.downloader.core.os.listdir')
    def test_download_by_ids_acquires_slot_per_video(
        self,
        mock_listdir,
        mock_audio, mock_search, mock_speech, mock_title,
        mock_transcode, mock_checkpoint
    ):
        """Each video download should acquire and release a slot."""
        from src.downloader.core import VideoDownloader

        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}, 'medium': {}, 'long': {}, 'longer': {}}
        mock_listdir.return_value = []

        with patch('src.downloader.core.get_config') as mock_get_config:
            config = MagicMock()
            config.cache_dir = "/tmp/test"
            config.downloaded_videos_dir = "/tmp/test"
            config.download = MagicMock()
            config.download.cookies_from_browser = ''
            config.download.cookie_rotation = None
            config.download.impersonation = None
            config.download.vpn = None
            config.download.speed_tracking = None
            config.download.circuit_breaker = None
            config.download.batch_retry = None
            config.download.rate_limit = None
            config.download.rate_limit_budget = None
            config.download.max_filename_len = 10
            mock_get_config.return_value = config

            downloader = VideoDownloader(config)

            # Mock the download command to return empty (simulating failed download)
            with patch.object(downloader, '_run_download_cmd', return_value=[]):
                with patch.object(downloader, '_add_escalation_to_cmd'):
                    with patch.object(downloader, '_add_cookies_to_cmd'):
                        with patch.object(downloader, '_build_format_string', return_value='bestvideo+bestaudio'):
                            # Track slot acquisitions
                            acquire_calls = []
                            release_calls = []

                            original_acquire = downloader.acquire_download_slot
                            original_release = downloader.release_download_slot

                            def track_acquire(*args, **kwargs):
                                result = original_acquire(*args, **kwargs)
                                acquire_calls.append(result)
                                return result

                            def track_release(*args, **kwargs):
                                release_calls.append(True)
                                return original_release(*args, **kwargs)

                            downloader.acquire_download_slot = track_acquire
                            downloader.release_download_slot = track_release

                            # Create temp directory
                            import tempfile
                            with tempfile.TemporaryDirectory() as tmpdir:
                                keyword_dir = Path(tmpdir) / "test_s"
                                keyword_dir.mkdir()
                                output_dir = Path(tmpdir)

                                # Download 3 videos
                                video_ids = ['video1', 'video2', 'video3']
                                downloader._download_by_ids(
                                    video_ids, keyword_dir, output_dir, 'test', 'short'
                                )

                                # Should have acquired and released 3 slots
                                assert len(acquire_calls) == 3, f"Expected 3 acquires, got {len(acquire_calls)}"
                                assert len(release_calls) == 3, f"Expected 3 releases, got {len(release_calls)}"
                                assert all(acquire_calls), "All acquisitions should succeed"

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @patch('src.downloader.core.os.listdir')
    def test_slot_released_on_download_failure(
        self,
        mock_listdir,
        mock_audio, mock_search, mock_speech, mock_title,
        mock_transcode, mock_checkpoint
    ):
        """Slot should be released even when download fails."""
        from src.downloader.core import VideoDownloader

        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}, 'medium': {}, 'long': {}, 'longer': {}}
        mock_listdir.return_value = []

        with patch('src.downloader.core.get_config') as mock_get_config:
            config = MagicMock()
            config.cache_dir = "/tmp/test"
            config.downloaded_videos_dir = "/tmp/test"
            config.download = MagicMock()
            config.download.cookies_from_browser = ''
            config.download.cookie_rotation = None
            config.download.impersonation = None
            config.download.vpn = None
            config.download.speed_tracking = None
            config.download.circuit_breaker = None
            config.download.batch_retry = None
            config.download.rate_limit = None
            config.download.rate_limit_budget = None
            config.download.max_filename_len = 10
            mock_get_config.return_value = config

            downloader = VideoDownloader(config)

            # Mock download to raise an exception
            def failing_download(*args, **kwargs):
                raise Exception("Download failed!")

            with patch.object(downloader, '_run_download_cmd', side_effect=failing_download):
                with patch.object(downloader, '_add_escalation_to_cmd'):
                    with patch.object(downloader, '_add_cookies_to_cmd'):
                        with patch.object(downloader, '_build_format_string', return_value='bestvideo+bestaudio'):
                            import tempfile
                            with tempfile.TemporaryDirectory() as tmpdir:
                                keyword_dir = Path(tmpdir) / "test_s"
                                keyword_dir.mkdir()
                                output_dir = Path(tmpdir)

                                # Download should fail but not crash
                                try:
                                    downloader._download_by_ids(
                                        ['video1'], keyword_dir, output_dir, 'test', 'short'
                                    )
                                except Exception:
                                    pass  # Expected

                                # Slot should still be released (via finally)
                                assert downloader.rate_limit_coordinator.get_active_slots('download') == 0


# =============================================================================
# Test: Coordinator Status
# =============================================================================


class TestCoordinatorStatus:
    """Test coordinator status reporting."""

    def test_get_status(self, coordinator):
        """Should return accurate status dict."""
        coordinator.acquire_slot('download', timeout=1.0)
        coordinator.acquire_slot('download', timeout=1.0)

        status = coordinator.get_status()

        assert status['enabled'] is True
        assert status['slots_per_second'] == 10.0
        assert status['burst_size'] == 5
        assert status['total_active'] == 2
        assert status['active_slots']['download'] == 2

    def test_is_enabled(self, coordinator):
        """Should report enabled state."""
        assert coordinator.is_enabled() is True

        coordinator.set_enabled(False)
        assert coordinator.is_enabled() is False


# =============================================================================
# Test: Integration with Rate Limit Metrics
# =============================================================================


class TestRateLimitMetricsIntegration:
    """Test that slot timeouts are recorded in rate limit metrics."""

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    def test_slot_timeout_increments_metrics(
        self,
        mock_audio, mock_search, mock_speech, mock_title,
        mock_transcode, mock_checkpoint
    ):
        """Slot timeout should increment rate_limit_metrics.slot_timeouts."""
        from src.downloader.core import VideoDownloader

        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}, 'medium': {}, 'long': {}, 'longer': {}}

        # Reset singleton
        GlobalRateLimitCoordinator.reset_instance()

        with patch('src.downloader.core.get_config') as mock_get_config:
            config = MagicMock()
            config.cache_dir = "/tmp/test"
            config.downloaded_videos_dir = "/tmp/test"
            config.download = MagicMock()
            config.download.cookies_from_browser = ''
            config.download.cookie_rotation = None
            config.download.impersonation = None
            config.download.vpn = None
            config.download.speed_tracking = None
            config.download.circuit_breaker = None
            config.download.batch_retry = None
            config.download.rate_limit = MagicMock()
            config.download.rate_limit.per_tier_isolation = False
            config.download.rate_limit.global_ = MagicMock()
            config.download.rate_limit.global_.enabled = True
            config.download.rate_limit.global_.slots_per_second = 100.0
            config.download.rate_limit.global_.burst_size = 1
            config.download.rate_limit_budget = None
            mock_get_config.return_value = config

            downloader = VideoDownloader(config)

            # Force slow coordinator for timeout
            downloader.rate_limit_coordinator._config.slots_per_second = 0.001

            # Exhaust burst
            downloader.acquire_download_slot(timeout=0.1)

            # This should timeout
            initial_timeouts = downloader.rate_limit_metrics.slot_timeouts
            acquired = downloader.acquire_download_slot(timeout=0.01)

            assert acquired is False
            assert downloader.rate_limit_metrics.slot_timeouts == initial_timeouts + 1
