"""
Tests for batch pre-check by channel (US-006 Sprint 7)

Tests the channel-based batch pre-checking optimization that reduces API calls
by grouping videos by channel and using representative samples.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionFetchError,
    CaptionCache,
    CaptionMetrics,
    ChannelCaptionPattern,
    BatchPreCheckResult,
)


class TestChannelCaptionPattern:
    """Test ChannelCaptionPattern dataclass (US-006 Sprint 7)"""

    def test_pattern_creation(self):
        """Test creating a channel caption pattern"""
        pattern = ChannelCaptionPattern(
            channel_id="UCtest123",
            videos_checked=10,
            captions_found=9,
            success_rate=0.9,
            last_updated=1234567890.0,
        )

        assert pattern.channel_id == "UCtest123"
        assert pattern.videos_checked == 10
        assert pattern.captions_found == 9
        assert pattern.success_rate == 0.9
        assert pattern.last_updated == 1234567890.0

    def test_pattern_update_with_caption(self):
        """Test updating pattern when video has captions"""
        pattern = ChannelCaptionPattern(
            channel_id="UCtest123",
            videos_checked=5,
            captions_found=4,
            success_rate=0.8,
        )

        pattern.update(has_captions=True)

        assert pattern.videos_checked == 6
        assert pattern.captions_found == 5
        assert pattern.success_rate == pytest.approx(5/6)
        assert pattern.last_updated > 0

    def test_pattern_update_without_caption(self):
        """Test updating pattern when video has no captions"""
        pattern = ChannelCaptionPattern(
            channel_id="UCtest123",
            videos_checked=5,
            captions_found=5,
            success_rate=1.0,
        )

        pattern.update(has_captions=False)

        assert pattern.videos_checked == 6
        assert pattern.captions_found == 5
        assert pattern.success_rate == pytest.approx(5/6)

    def test_pattern_to_dict(self):
        """Test pattern serialization"""
        pattern = ChannelCaptionPattern(
            channel_id="UCtest123",
            videos_checked=10,
            captions_found=9,
            success_rate=0.9,
            last_updated=1234567890.0,
        )

        result = pattern.to_dict()

        assert result['channel_id'] == "UCtest123"
        assert result['videos_checked'] == 10
        assert result['captions_found'] == 9
        assert result['success_rate'] == 0.9
        assert result['last_updated'] == 1234567890.0

    def test_pattern_from_dict(self):
        """Test pattern deserialization"""
        data = {
            'channel_id': "UCtest123",
            'videos_checked': 10,
            'captions_found': 8,
            'success_rate': 0.8,
            'last_updated': 1234567890.0,
        }

        pattern = ChannelCaptionPattern.from_dict(data)

        assert pattern.channel_id == "UCtest123"
        assert pattern.videos_checked == 10
        assert pattern.captions_found == 8
        assert pattern.success_rate == 0.8

    def test_pattern_initial_state(self):
        """Test pattern with default initial state"""
        pattern = ChannelCaptionPattern(channel_id="UCnew123")

        assert pattern.videos_checked == 0
        assert pattern.captions_found == 0
        assert pattern.success_rate == 0.0

    def test_pattern_first_update(self):
        """Test updating fresh pattern"""
        pattern = ChannelCaptionPattern(channel_id="UCnew123")

        pattern.update(has_captions=True)

        assert pattern.videos_checked == 1
        assert pattern.captions_found == 1
        assert pattern.success_rate == 1.0


class TestBatchPreCheckResult:
    """Test BatchPreCheckResult dataclass (US-006 Sprint 7)"""

    def test_result_creation(self):
        """Test creating batch pre-check result"""
        result = BatchPreCheckResult(
            total_videos=50,
            actual_checks=10,
            skipped_by_pattern=40,
            api_calls_saved=40,
        )

        assert result.total_videos == 50
        assert result.actual_checks == 10
        assert result.skipped_by_pattern == 40
        assert result.api_calls_saved == 40

    def test_result_video_results(self):
        """Test video results dictionary"""
        result = BatchPreCheckResult()
        result.video_results = {
            "vid1": True,
            "vid2": True,
            "vid3": False,
        }

        assert result.video_results["vid1"] is True
        assert result.video_results["vid3"] is False

    def test_result_to_dict(self):
        """Test result serialization"""
        result = BatchPreCheckResult(
            total_videos=10,
            actual_checks=5,
            skipped_by_pattern=5,
            api_calls_saved=5,
        )
        result.video_results = {"vid1": True, "vid2": False}

        data = result.to_dict()

        assert data['total_videos'] == 10
        assert data['actual_checks'] == 5
        assert data['video_results'] == {"vid1": True, "vid2": False}


class TestCaptionCacheChannelPatterns:
    """Test CaptionCache channel pattern methods (US-006 Sprint 7)"""

    @pytest.fixture
    def cache_config(self, tmp_path):
        """Create a mock config for CaptionCache"""
        mock_config = Mock()
        mock_config.cache_dir = str(tmp_path)
        mock_config.max_cache_age_days = 30
        mock_config.enabled = True
        return mock_config

    def test_save_and_load_patterns(self, cache_config):
        """Test saving and loading channel patterns"""
        cache = CaptionCache(cache_config)

        # Create patterns
        patterns = {
            "UCtest1": ChannelCaptionPattern("UCtest1", 10, 9, 0.9, time.time()),
            "UCtest2": ChannelCaptionPattern("UCtest2", 5, 0, 0.0, time.time()),
        }

        # Save patterns
        result = cache.save_channel_patterns(patterns)
        assert result is True

        # Load patterns
        loaded = cache.load_channel_patterns()

        assert "UCtest1" in loaded
        assert loaded["UCtest1"].videos_checked == 10
        assert loaded["UCtest1"].success_rate == 0.9
        assert "UCtest2" in loaded
        assert loaded["UCtest2"].captions_found == 0

    def test_get_channel_pattern(self, cache_config):
        """Test getting single channel pattern"""
        cache = CaptionCache(cache_config)

        # Save a pattern
        patterns = {"UCtest1": ChannelCaptionPattern("UCtest1", 5, 4, 0.8, time.time())}
        cache.save_channel_patterns(patterns)

        # Get specific pattern
        pattern = cache.get_channel_pattern("UCtest1")
        assert pattern is not None
        assert pattern.channel_id == "UCtest1"

        # Non-existent pattern
        missing = cache.get_channel_pattern("UCmissing")
        assert missing is None

    def test_update_channel_pattern(self, cache_config):
        """Test updating channel pattern"""
        cache = CaptionCache(cache_config)

        # Update new pattern
        pattern = cache.update_channel_pattern("UCnew", has_captions=True)

        assert pattern.channel_id == "UCnew"
        assert pattern.videos_checked == 1
        assert pattern.captions_found == 1

        # Update again
        pattern = cache.update_channel_pattern("UCnew", has_captions=True)
        assert pattern.videos_checked == 2
        assert pattern.captions_found == 2

        # Update with no captions
        pattern = cache.update_channel_pattern("UCnew", has_captions=False)
        assert pattern.videos_checked == 3
        assert pattern.captions_found == 2
        assert pattern.success_rate == pytest.approx(2/3)

    def test_patterns_disabled_cache(self, tmp_path):
        """Test pattern methods with disabled cache"""
        mock_config = Mock()
        mock_config.cache_dir = str(tmp_path)
        mock_config.max_cache_age_days = 30
        mock_config.cache_captions = False  # This disables the cache

        cache = CaptionCache(mock_config)

        # With cache disabled, save should return False
        patterns = {"UCtest": ChannelCaptionPattern("UCtest", 5, 5, 1.0, time.time())}
        assert cache.save_channel_patterns(patterns) is False

        # Load should return empty
        loaded = cache.load_channel_patterns()
        assert loaded == {}


class TestCaptionMetricsBatchPrecheck:
    """Test CaptionMetrics batch pre-check tracking (US-006 Sprint 7)"""

    def test_record_pre_check_batched_checked(self):
        """Test recording batch pre-check with actual check"""
        metrics = CaptionMetrics()

        metrics.record_pre_check_batched("vid1", skipped=False)

        assert metrics.pre_check_batched_total == 1
        assert metrics.pre_check_batched_checked == 1
        assert metrics.pre_check_batched_skipped == 0

    def test_record_pre_check_batched_skipped(self):
        """Test recording batch pre-check skipped by pattern"""
        metrics = CaptionMetrics()

        metrics.record_pre_check_batched("vid1", skipped=True)

        assert metrics.pre_check_batched_total == 1
        assert metrics.pre_check_batched_checked == 0
        assert metrics.pre_check_batched_skipped == 1

    def test_record_pre_check_batched_multiple(self):
        """Test recording multiple batch pre-checks"""
        metrics = CaptionMetrics()

        # 5 actual checks
        for i in range(5):
            metrics.record_pre_check_batched(f"vid{i}", skipped=False)

        # 10 skipped by pattern
        for i in range(5, 15):
            metrics.record_pre_check_batched(f"vid{i}", skipped=True)

        assert metrics.pre_check_batched_total == 15
        assert metrics.pre_check_batched_checked == 5
        assert metrics.pre_check_batched_skipped == 10

    def test_set_batch_precheck_savings(self):
        """Test setting API calls saved"""
        metrics = CaptionMetrics()

        metrics.set_batch_precheck_savings(45)

        assert metrics.pre_check_api_calls_saved == 45

    def test_metrics_serialization(self):
        """Test batch pre-check metrics in to_dict/from_dict"""
        metrics = CaptionMetrics()
        metrics.pre_check_batched_total = 50
        metrics.pre_check_batched_skipped = 40
        metrics.pre_check_batched_checked = 10
        metrics.pre_check_api_calls_saved = 40

        data = metrics.to_dict()

        assert data['pre_check_batched_total'] == 50
        assert data['pre_check_batched_skipped'] == 40

        # Restore from dict
        restored = CaptionMetrics.from_dict(data)
        assert restored.pre_check_batched_total == 50
        assert restored.pre_check_api_calls_saved == 40

    def test_metrics_summary_includes_batch_info(self):
        """Test batch pre-check info in summary"""
        metrics = CaptionMetrics()
        metrics.pre_check_batched_total = 50
        metrics.pre_check_batched_skipped = 45
        metrics.pre_check_batched_checked = 5
        metrics.pre_check_api_calls_saved = 45

        summary = metrics.summary()

        assert "Batch pre-check:" in summary
        assert "5 checked" in summary
        assert "45 skipped" in summary
        assert "45 API calls saved" in summary


class TestBatchPrecheckByChannel:
    """Test CaptionFetcher.batch_precheck_by_channel (US-006 Sprint 7)"""

    @pytest.fixture
    def mock_fetcher(self):
        """Create a fetcher with mocked methods"""
        with patch.object(CaptionFetcher, '__init__', lambda x, config=None: None):
            fetcher = CaptionFetcher()
            fetcher._timeout = 30
            fetcher._cookies_file = None
            return fetcher

    def test_empty_video_list(self, mock_fetcher):
        """Test with empty video list"""
        result = mock_fetcher.batch_precheck_by_channel([])

        assert result.total_videos == 0
        assert result.actual_checks == 0
        assert result.api_calls_saved == 0

    def test_single_video_no_channel_info(self, mock_fetcher):
        """Test single video without pre-computed channel info"""
        # Mock get_video_metadata and has_captions
        mock_fetcher.get_video_metadata = Mock(return_value={'channel_id': 'UCtest1'})
        mock_fetcher.has_captions = Mock(return_value=True)

        result = mock_fetcher.batch_precheck_by_channel(["vid1"])

        assert result.total_videos == 1
        assert "vid1" in result.video_results
        assert result.video_results["vid1"] is True

    def test_with_precomputed_channel_info(self, mock_fetcher):
        """Test with pre-computed channel_info mapping"""
        mock_fetcher.has_captions = Mock(return_value=True)

        channel_info = {
            "vid1": "UCchannel1",
            "vid2": "UCchannel1",
            "vid3": "UCchannel1",
        }

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=["vid1", "vid2", "vid3"],
            channel_info=channel_info,
        )

        assert result.total_videos == 3
        # With channel_info provided, no metadata fetches needed
        # But individual has_captions calls still made for new channel
        assert mock_fetcher.has_captions.call_count >= 1

    def test_high_confidence_pattern_skips_checks(self, mock_fetcher, tmp_path):
        """Test that high-confidence patterns skip individual checks"""
        mock_fetcher.has_captions = Mock(return_value=True)
        mock_fetcher.get_video_metadata = Mock(return_value={'channel_id': 'UCtest1'})

        # Create cache with high-confidence pattern
        mock_config = Mock()
        mock_config.cache_dir = str(tmp_path)
        mock_config.max_cache_age_days = 30
        mock_config.enabled = True

        cache = CaptionCache(mock_config)
        # Pre-populate with high-confidence pattern (10 videos, 10 successes)
        pattern = ChannelCaptionPattern("UCtest1", 10, 10, 1.0, time.time())
        cache.save_channel_patterns({"UCtest1": pattern})

        # Provide channel info directly
        channel_info = {f"vid{i}": "UCtest1" for i in range(50)}
        video_ids = [f"vid{i}" for i in range(50)]

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=video_ids,
            channel_info=channel_info,
            cache=cache,
            confidence_threshold=0.9,
            min_samples_for_confidence=5,
        )

        # With high-confidence pattern, all 50 videos should use pattern
        assert result.total_videos == 50
        assert result.skipped_by_pattern == 50
        # No has_captions calls made
        mock_fetcher.has_captions.assert_not_called()

    def test_low_confidence_pattern_checks_samples(self, mock_fetcher, tmp_path):
        """Test that low-confidence patterns check sample videos"""
        mock_fetcher.has_captions = Mock(return_value=True)
        mock_fetcher.get_video_metadata = Mock(return_value={'channel_id': 'UCtest1'})

        mock_config = Mock()
        mock_config.cache_dir = str(tmp_path)
        mock_config.max_cache_age_days = 30
        mock_config.enabled = True

        cache = CaptionCache(mock_config)
        # Pre-populate with low-confidence pattern (only 2 videos checked)
        pattern = ChannelCaptionPattern("UCtest1", 2, 2, 1.0, time.time())
        cache.save_channel_patterns({"UCtest1": pattern})

        channel_info = {f"vid{i}": "UCtest1" for i in range(10)}
        video_ids = [f"vid{i}" for i in range(10)]

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=video_ids,
            channel_info=channel_info,
            cache=cache,
            confidence_threshold=0.9,
            min_samples_for_confidence=5,
            sample_size_per_channel=5,
        )

        # With low confidence (only 2 samples), need to check more
        # Should check up to sample_size_per_channel videos
        assert result.total_videos == 10
        assert mock_fetcher.has_captions.call_count >= 3  # At least 3 more to reach min_samples

    def test_mixed_channels(self, mock_fetcher, tmp_path):
        """Test batch with videos from multiple channels"""
        mock_fetcher.has_captions = Mock(return_value=True)
        mock_fetcher.get_video_metadata = Mock(return_value={'channel_id': 'UCunknown'})

        channel_info = {
            "vid1": "UCchannel1",
            "vid2": "UCchannel1",
            "vid3": "UCchannel2",
            "vid4": "UCchannel2",
            "vid5": "UCchannel3",
        }
        video_ids = ["vid1", "vid2", "vid3", "vid4", "vid5"]

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=video_ids,
            channel_info=channel_info,
        )

        assert result.total_videos == 5
        assert len(result.video_results) == 5

    def test_metrics_tracking(self, mock_fetcher):
        """Test that metrics are properly tracked during batch pre-check"""
        mock_fetcher.has_captions = Mock(return_value=True)
        mock_fetcher.get_video_metadata = Mock(return_value={'channel_id': 'UCtest1'})

        metrics = CaptionMetrics()
        channel_info = {"vid1": "UCtest1", "vid2": "UCtest1"}

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=["vid1", "vid2"],
            channel_info=channel_info,
            metrics=metrics,
        )

        # Metrics should have been updated
        assert metrics.pre_check_batched_total >= 2
        assert metrics.pre_check_available >= 2  # Both videos have captions

    def test_unknown_channel_fallback(self, mock_fetcher):
        """Test handling videos where channel_id cannot be determined"""
        mock_fetcher.has_captions = Mock(return_value=True)
        mock_fetcher.get_video_metadata = Mock(return_value={})  # No channel_id

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=["vid1", "vid2"],
            channel_info=None,  # Force metadata lookup
        )

        # Videos with unknown channel should still be checked
        assert result.total_videos == 2
        assert len(result.video_results) == 2
        assert mock_fetcher.has_captions.call_count == 2

    def test_error_handling_in_precheck(self, mock_fetcher):
        """Test that pre-check errors are handled gracefully"""
        mock_fetcher.has_captions = Mock(side_effect=CaptionFetchError("Network error"))
        mock_fetcher.get_video_metadata = Mock(return_value={'channel_id': 'UCtest1'})

        channel_info = {"vid1": "UCtest1"}

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=["vid1"],
            channel_info=channel_info,
        )

        # On error, should assume available (True) to avoid false negatives
        assert result.video_results["vid1"] is True

    def test_50_videos_same_channel_uses_5_checks(self, mock_fetcher, tmp_path):
        """Test acceptance criteria: 50 videos from same channel use ~5 sample checks"""
        mock_fetcher.has_captions = Mock(return_value=True)

        mock_config = Mock()
        mock_config.cache_dir = str(tmp_path)
        mock_config.max_cache_age_days = 30
        mock_config.enabled = True

        cache = CaptionCache(mock_config)
        # Fresh cache, no pre-existing patterns

        # 50 videos from same channel
        channel_info = {f"vid{i}": "UCsame_channel" for i in range(50)}
        video_ids = [f"vid{i}" for i in range(50)]

        result = mock_fetcher.batch_precheck_by_channel(
            video_ids=video_ids,
            channel_info=channel_info,
            cache=cache,
            confidence_threshold=0.9,
            min_samples_for_confidence=5,
            sample_size_per_channel=5,
        )

        # Should only make ~5 actual has_captions calls (sample size)
        # Then infer the rest based on pattern
        assert result.total_videos == 50
        assert mock_fetcher.has_captions.call_count == 5
        assert result.actual_checks == 5  # Only sample size checked
        assert result.skipped_by_pattern == 45
        assert result.api_calls_saved == 45

        # Pattern should be updated in cache
        loaded_patterns = cache.load_channel_patterns()
        assert "UCsame_channel" in loaded_patterns
        assert loaded_patterns["UCsame_channel"].videos_checked == 5


class TestBatchPrecheckIntegration:
    """Integration tests for batch pre-check feature"""

    @pytest.mark.slow
    def test_cache_persistence_across_runs(self, tmp_path):
        """Test that channel patterns persist across fetcher instances"""
        mock_config = Mock()
        mock_config.cache_dir = str(tmp_path)
        mock_config.max_cache_age_days = 30
        mock_config.enabled = True

        # First run: build patterns
        cache1 = CaptionCache(mock_config)
        pattern = ChannelCaptionPattern("UCtest1", 10, 9, 0.9, time.time())
        cache1.save_channel_patterns({"UCtest1": pattern})

        # Second run: load patterns
        cache2 = CaptionCache(mock_config)
        loaded = cache2.load_channel_patterns()

        assert "UCtest1" in loaded
        assert loaded["UCtest1"].videos_checked == 10
        assert loaded["UCtest1"].success_rate == 0.9
