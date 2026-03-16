"""Unit tests for DownloadRetryBudget (US-129-002)."""

import pytest
from dataclasses import asdict
from src.downloader.retry_queue import DownloadRetryBudget, VideoRetryState
from src.config.sections.download import DownloadRetryBudgetConfig


class TestVideoRetryState:
    """Tests for VideoRetryState dataclass."""

    def test_create_with_defaults(self):
        """Test creating VideoRetryState with default values."""
        state = VideoRetryState(video_id="abc123")
        assert state.video_id == "abc123"
        assert state.attempts == 0
        assert state.total_backoff_seconds == 0.0

    def test_create_with_values(self):
        """Test creating VideoRetryState with custom values."""
        state = VideoRetryState(video_id="abc123", attempts=3, total_backoff_seconds=120.5)
        assert state.video_id == "abc123"
        assert state.attempts == 3
        assert state.total_backoff_seconds == 120.5


class TestDownloadRetryBudget:
    """Tests for DownloadRetryBudget class."""

    def test_create_with_defaults(self):
        """Test creating DownloadRetryBudget with default config."""
        budget = DownloadRetryBudget()
        assert budget.enabled is True
        assert budget._config.max_attempts == 5
        assert budget._config.max_backoff_time_seconds == 300.0

    def test_create_with_custom_config(self):
        """Test creating DownloadRetryBudget with custom config."""
        config = DownloadRetryBudgetConfig(
            enabled=True,
            max_attempts=10,
            max_backoff_time_seconds=600.0
        )
        budget = DownloadRetryBudget(config)
        assert budget.enabled is True
        assert budget._config.max_attempts == 10
        assert budget._config.max_backoff_time_seconds == 600.0

    def test_disabled_budget(self):
        """Test that disabled budget always returns not exhausted."""
        config = DownloadRetryBudgetConfig(enabled=False, max_attempts=1)
        budget = DownloadRetryBudget(config)

        # Record many attempts - should still return not exhausted when disabled
        for i in range(100):
            budget.record_attempt(f"video_{i}")

        assert budget.is_exhausted("video_0") is False

    def test_record_attempt(self):
        """Test recording retry attempts."""
        budget = DownloadRetryBudget()

        budget.record_attempt("video_1", backoff_seconds=10.0)
        assert budget._video_states["video_1"].attempts == 1
        assert budget._video_states["video_1"].total_backoff_seconds == 10.0

        budget.record_attempt("video_1", backoff_seconds=20.0)
        assert budget._video_states["video_1"].attempts == 2
        assert budget._video_states["video_1"].total_backoff_seconds == 30.0

    def test_is_exhausted_by_attempts(self):
        """Test budget exhaustion by max_attempts."""
        config = DownloadRetryBudgetConfig(enabled=True, max_attempts=3, max_backoff_time_seconds=0)
        budget = DownloadRetryBudget(config)

        # Should not be exhausted after 2 attempts
        budget.record_attempt("video_1")
        budget.record_attempt("video_1")
        assert budget.is_exhausted("video_1") is False

        # Should be exhausted after 3 attempts
        budget.record_attempt("video_1")
        assert budget.is_exhausted("video_1") is True

    def test_is_exhausted_by_backoff(self):
        """Test budget exhaustion by max_backoff_time_seconds."""
        config = DownloadRetryBudgetConfig(enabled=True, max_attempts=0, max_backoff_time_seconds=100.0)
        budget = DownloadRetryBudget(config)

        # Should not be exhausted after 50s backoff
        budget.record_attempt("video_1", backoff_seconds=50.0)
        assert budget.is_exhausted("video_1") is False

        # Should be exhausted after 60s more (total 110s)
        budget.record_attempt("video_1", backoff_seconds=60.0)
        assert budget.is_exhausted("video_1") is True

    def test_is_exhausted_combined(self):
        """Test budget exhaustion with both limits."""
        config = DownloadRetryBudgetConfig(enabled=True, max_attempts=3, max_backoff_time_seconds=50.0)
        budget = DownloadRetryBudget(config)

        # 2 attempts = 20s backoff, should not be exhausted
        budget.record_attempt("video_1", backoff_seconds=10.0)
        budget.record_attempt("video_1", backoff_seconds=10.0)
        assert budget.is_exhausted("video_1") is False

        # 3rd attempt pushes backoff to 30s, but max_attempts=3 exhausted
        budget.record_attempt("video_1", backoff_seconds=10.0)
        assert budget.is_exhausted("video_1") is True

    def test_is_exhausted_untracked_video(self):
        """Test that untracked videos return False for is_exhausted."""
        budget = DownloadRetryBudget()
        assert budget.is_exhausted("nonexistent_video") is False

    def test_get_remaining_attempts(self):
        """Test getting remaining attempts for a video."""
        config = DownloadRetryBudgetConfig(enabled=True, max_attempts=5)
        budget = DownloadRetryBudget(config)

        # New video should have all attempts remaining
        assert budget.get_remaining_attempts("video_1") == 5

        # After 2 attempts, 3 remaining
        budget.record_attempt("video_1")
        budget.record_attempt("video_1")
        assert budget.get_remaining_attempts("video_1") == 3

        # After exhausting, 0 remaining
        budget.record_attempt("video_1")
        budget.record_attempt("video_1")
        budget.record_attempt("video_1")
        assert budget.get_remaining_attempts("video_1") == 0

    def test_get_remaining_attempts_unlimited(self):
        """Test get_remaining_attempts when max_attempts=0 (unlimited)."""
        config = DownloadRetryBudgetConfig(enabled=True, max_attempts=0)
        budget = DownloadRetryBudget(config)

        assert budget.get_remaining_attempts("video_1") == -1

    def test_reset(self):
        """Test resetting budget for a video."""
        budget = DownloadRetryBudget()

        # Record some attempts
        budget.record_attempt("video_1")
        budget.record_attempt("video_1")
        assert budget.is_exhausted("video_1") is False

        # Reset should remove the video state
        budget.reset("video_1")
        assert "video_1" not in budget._video_states

        # Video should now have full budget
        assert budget.get_remaining_attempts("video_1") == 5

    def test_reset_all(self):
        """Test resetting all budget states."""
        budget = DownloadRetryBudget()

        budget.record_attempt("video_1")
        budget.record_attempt("video_2")
        budget.record_attempt("video_3")

        assert len(budget._video_states) == 3

        budget.reset_all()

        assert len(budget._video_states) == 0

    def test_to_checkpoint_dict(self):
        """Test serializing budget state to checkpoint."""
        budget = DownloadRetryBudget()

        budget.record_attempt("video_1", backoff_seconds=10.0)
        budget.record_attempt("video_1", backoff_seconds=20.0)
        budget.record_attempt("video_2", backoff_seconds=5.0)

        checkpoint = budget.to_checkpoint_dict()

        assert "video_states" in checkpoint
        assert checkpoint["video_states"]["video_1"]["attempts"] == 2
        assert checkpoint["video_states"]["video_1"]["total_backoff_seconds"] == 30.0
        assert checkpoint["video_states"]["video_2"]["attempts"] == 1
        assert checkpoint["video_states"]["video_2"]["total_backoff_seconds"] == 5.0

    def test_from_checkpoint_dict(self):
        """Test restoring budget state from checkpoint."""
        data = {
            "video_states": {
                "video_1": {"attempts": 3, "total_backoff_seconds": 150.0},
                "video_2": {"attempts": 1, "total_backoff_seconds": 10.0},
            }
        }

        budget = DownloadRetryBudget()
        budget.from_checkpoint_dict(data)

        assert len(budget._video_states) == 2
        assert budget._video_states["video_1"].attempts == 3
        assert budget._video_states["video_1"].total_backoff_seconds == 150.0
        assert budget._video_states["video_2"].attempts == 1

    def test_from_checkpoint_dict_empty(self):
        """Test restoring from empty checkpoint."""
        budget = DownloadRetryBudget()

        budget.record_attempt("video_1")
        budget.from_checkpoint_dict({})

        # Original state should be preserved
        assert "video_1" in budget._video_states

    def test_from_checkpoint_dict_none(self):
        """Test restoring from None checkpoint."""
        budget = DownloadRetryBudget()

        budget.record_attempt("video_1")
        budget.from_checkpoint_dict(None)

        # Original state should be preserved
        assert "video_1" in budget._video_states


class TestDownloadRetryBudgetConfig:
    """Tests for DownloadRetryBudgetConfig validation."""

    def test_valid_config(self):
        """Test creating valid config."""
        config = DownloadRetryBudgetConfig(enabled=True, max_attempts=5, max_backoff_time_seconds=300.0)
        assert config.enabled is True
        assert config.max_attempts == 5
        assert config.max_backoff_time_seconds == 300.0

    def test_invalid_max_attempts(self):
        """Test that negative max_attempts raises error."""
        with pytest.raises(ValueError, match="max_attempts must be >= 0"):
            DownloadRetryBudgetConfig(max_attempts=-1)

    def test_invalid_max_backoff(self):
        """Test that negative max_backoff_time_seconds raises error."""
        with pytest.raises(ValueError, match="max_backoff_time_seconds must be >= 0"):
            DownloadRetryBudgetConfig(max_backoff_time_seconds=-10.0)

    def test_default_values(self):
        """Test default config values."""
        config = DownloadRetryBudgetConfig()
        assert config.enabled is True
        assert config.max_attempts == 5
        assert config.max_backoff_time_seconds == 300.0


class TestCategoryBackoffConfig:
    """Tests for CategoryBackoffConfig (US-144-003)."""

    def test_default_category_backoff_values(self):
        """Test default category backoff values."""
        from src.config.sections.download import CategoryBackoffConfig
        config = CategoryBackoffConfig()

        # Base times
        assert config.rate_limit_base == 30.0
        assert config.network_base == 10.0
        assert config.format_base == 5.0
        assert config.server_base == 15.0
        assert config.bot_detection_base == 30.0
        assert config.timeout_base == 15.0
        assert config.geo_blocked_base == 60.0

        # Max backoffs
        assert config.rate_limit_max == 300.0
        assert config.network_max == 120.0
        assert config.format_max == 60.0

        # Exponential base
        assert config.exponential_base == 2.0

    def test_custom_category_backoff_values(self):
        """Test custom category backoff values."""
        from src.config.sections.download import CategoryBackoffConfig
        config = CategoryBackoffConfig(
            rate_limit_base=60.0,
            rate_limit_max=600.0,
            exponential_base=1.5
        )

        assert config.rate_limit_base == 60.0
        assert config.rate_limit_max == 600.0
        assert config.exponential_base == 1.5


class TestCategoryAwareBackoff:
    """Tests for category-aware exponential backoff (US-144-003)."""

    def test_calculate_category_backoff_rate_limit(self):
        """Test rate_limit category backoff calculation."""
        budget = DownloadRetryBudget()

        # First attempt should return base time (30s) with some jitter
        backoff = budget.calculate_category_backoff("video1", "rate_limit")

        # Base is 30s, so with jitter it should be around 27-33s
        assert 27.0 <= backoff <= 33.0

    def test_calculate_category_backoff_network(self):
        """Test network category backoff calculation."""
        budget = DownloadRetryBudget()

        backoff = budget.calculate_category_backoff("video1", "network")

        # Base is 10s
        assert 9.0 <= backoff <= 11.0

    def test_calculate_category_backoff_format(self):
        """Test format category backoff calculation."""
        budget = DownloadRetryBudget()

        backoff = budget.calculate_category_backoff("video1", "format")

        # Base is 5s
        assert 4.5 <= backoff <= 5.5

    def test_calculate_category_backoff_exponential_growth(self):
        """Test exponential backoff growth with retries."""
        budget = DownloadRetryBudget()

        # Record attempts for rate_limit category
        budget.record_attempt("video1", error_category="rate_limit")  # 1st attempt
        budget.record_attempt("video1", error_category="rate_limit")  # 2nd attempt
        budget.record_attempt("video1", error_category="rate_limit")  # 3rd attempt

        # Next backoff should be exponential: 30 * 2^3 = 240s
        backoff = budget.calculate_category_backoff("video1", "rate_limit")

        # Should be around 240s (216-264s with jitter)
        assert 200.0 <= backoff <= 280.0

    def test_calculate_category_backoff_caps_at_max(self):
        """Test that backoff caps at max value."""
        budget = DownloadRetryBudget()

        # Record many attempts to trigger max cap
        for _ in range(10):
            budget.record_attempt("video1", error_category="rate_limit")

        # Backoff should cap at 300s (rate_limit_max)
        backoff = budget.calculate_category_backoff("video1", "rate_limit")

        # Should be around 300s (270-330s with jitter)
        assert 250.0 <= backoff <= 350.0

    def test_calculate_category_backoff_unknown_category(self):
        """Test unknown category defaults to network settings."""
        budget = DownloadRetryBudget()

        backoff = budget.calculate_category_backoff("video1", "unknown_category")

        # Should default to network base (10s)
        assert 9.0 <= backoff <= 11.0

    def test_backoff_reduction_on_success(self):
        """Test that successful retries reduce backoff for future retries."""
        config = DownloadRetryBudgetConfig(
            reduce_backoff_on_success=True,
            success_backoff_reduction_factor=0.5
        )
        budget = DownloadRetryBudget(config)

        # Record some attempts
        budget.record_attempt("video1", error_category="rate_limit")  # 1st attempt
        budget.record_attempt("video1", error_category="rate_limit")  # 2nd attempt
        budget.record_attempt("video1", error_category="rate_limit")  # 3rd attempt

        # Now record a successful retry - this should reduce effective attempts
        budget.record_successful_retry("video1", "rate_limit")

        # Effective attempts = 3 - 1 = 2, so backoff = 30 * 2^2 = 120s
        backoff = budget.calculate_category_backoff("video1", "rate_limit")

        # Should be around 120s (108-132s with jitter)
        assert 100.0 <= backoff <= 140.0

    def test_backoff_no_reduction_when_disabled(self):
        """Test that backoff is not reduced when feature is disabled."""
        config = DownloadRetryBudgetConfig(reduce_backoff_on_success=False)
        budget = DownloadRetryBudget(config)

        # Record some attempts
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="rate_limit")

        # Record successful retry (but feature is disabled)
        budget.record_successful_retry("video1", "rate_limit")

        # Effective attempts = 2 (reduction disabled)
        backoff = budget.calculate_category_backoff("video1", "rate_limit")

        # Should be around 120s (108-132s with jitter)
        assert 100.0 <= backoff <= 140.0

    def test_different_categories_have_different_backoffs(self):
        """Test that different categories have different base backoffs."""
        budget = DownloadRetryBudget()

        rate_limit_backoff = budget.calculate_category_backoff("video1", "rate_limit")
        network_backoff = budget.calculate_category_backoff("video1", "network")
        format_backoff = budget.calculate_category_backoff("video1", "format")
        geo_blocked_backoff = budget.calculate_category_backoff("video1", "geo_blocked")

        # rate_limit (30s) > network (10s) > format (5s)
        # geo_blocked (60s) should be highest
        assert geo_blocked_backoff > rate_limit_backoff
        assert rate_limit_backoff > network_backoff
        assert network_backoff > format_backoff

    def test_record_attempt_with_category_backoff(self):
        """Test that record_attempt calculates category-aware backoff when not provided."""
        budget = DownloadRetryBudget()

        # record_attempt without backoff should calculate category-aware backoff
        budget.record_attempt("video1", error_category="rate_limit")

        # Check that backoff was recorded
        state = budget._video_states["video1"]
        assert state.attempts == 1
        assert state.total_backoff_seconds > 0

    def test_backoff_improves_over_time(self):
        """Test that backoff increases with repeated failures."""
        budget = DownloadRetryBudget()

        backoffs = []
        for i in range(5):
            backoff = budget.calculate_category_backoff("video1", "network")
            backoffs.append(backoff)
            budget.record_attempt("video1", error_category="network")

        # Each backoff should be roughly double the previous
        # network: 10 -> 20 -> 40 -> 80 -> 120 (caps at 120)
        for i in range(1, len(backoffs)):
            # Each subsequent backoff should be at least as large (due to cap)
            assert backoffs[i] >= backoffs[i - 1] * 0.9  # Allow small variance for jitter


class TestMixedErrorCategoriesIntegration:
    """Integration tests for mixed error categories (US-144-003)."""

    def test_mixed_error_categories_independent_backoff(self):
        """Test that different videos with different error categories have independent backoffs."""
        budget = DownloadRetryBudget()

        # Video 1: rate_limit errors
        # Video 2: network errors
        # Video 3: format errors

        # Record attempts for each video with different categories
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="rate_limit")

        budget.record_attempt("video2", error_category="network")
        budget.record_attempt("video2", error_category="network")

        budget.record_attempt("video3", error_category="format")
        budget.record_attempt("video3", error_category="format")

        # Calculate backoffs for each
        backoff1 = budget.calculate_category_backoff("video1", "rate_limit")
        backoff2 = budget.calculate_category_backoff("video2", "network")
        backoff3 = budget.calculate_category_backoff("video3", "format")

        # rate_limit should have highest backoff, then network, then format
        # rate_limit: 30 * 2^2 = 120s
        # network: 10 * 2^2 = 40s
        # format: 5 * 2^2 = 20s
        assert backoff1 > backoff2 > backoff3
        assert 100.0 <= backoff1 <= 140.0  # ~120s
        assert 35.0 <= backoff2 <= 45.0    # ~40s
        assert 15.0 <= backoff3 <= 25.0    # ~20s

    def test_same_video_different_categories(self):
        """Test that same video with different error categories tracks separately."""
        budget = DownloadRetryBudget()

        # Video has both rate_limit and network errors
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="network")
        budget.record_attempt("video1", error_category="network")

        # Get backoff for each category
        rate_limit_backoff = budget.calculate_category_backoff("video1", "rate_limit")
        network_backoff = budget.calculate_category_backoff("video1", "network")

        # Each should have 2 attempts independently
        # rate_limit: 30 * 2^2 = 120s
        # network: 10 * 2^2 = 40s
        assert rate_limit_backoff > network_backoff
        assert 100.0 <= rate_limit_backoff <= 140.0
        assert 35.0 <= network_backoff <= 45.0

    def test_mixed_batch_processing(self):
        """Test batch processing with mixed error categories simulates real scenario."""
        budget = DownloadRetryBudget()

        # Simulate a batch of downloads with various error categories
        videos = [
            ("video1", "rate_limit"),
            ("video2", "network"),
            ("video3", "format"),
            ("video4", "rate_limit"),
            ("video5", "network"),
            ("video6", "server"),
            ("video7", "bot_detection"),
            ("video8", "geo_blocked"),
        ]

        # Each video gets 1 retry attempt
        for video_id, category in videos:
            budget.record_attempt(video_id, error_category=category)

        # Calculate backoffs for all
        # After 1 attempt: backoff = base * (exponential_base ^ 1) = base * 2
        backoffs = {video_id: budget.calculate_category_backoff(video_id, category)
                    for video_id, category in videos}

        # Verify ordering based on base values (allow for jitter in similar values):
        # geo_blocked (60s) > rate_limit/bot_detection (30s) > server/timeout (15s) > network (10s) > format (5s)
        # geo_blocked should be clearly highest
        assert backoffs["video8"] > backoffs["video2"] * 2  # geo_blocked >> network
        assert backoffs["video8"] > backoffs["video3"] * 5  # geo_blocked >> format

        # server should be between network and rate_limit
        assert backoffs["video6"] > backoffs["video2"]  # server > network
        assert backoffs["video2"] > backoffs["video3"]  # network > format

        # Verify reasonable ranges for first retry (after 1 attempt: base * 2, with ±10% jitter)
        # geo_blocked: 60 * 2 = 120, jitter ±12 = 108-132
        # rate_limit/bot_detection: 30 * 2 = 60, jitter ±6 = 54-66
        # server: 15 * 2 = 30, jitter ±3 = 27-33
        # network: 10 * 2 = 20, jitter ±2 = 18-22
        # format: 5 * 2 = 10, jitter ±1 = 9-11
        assert 100.0 <= backoffs["video8"] <= 140.0    # geo_blocked ~120s
        assert 50.0 <= backoffs["video1"] <= 70.0      # rate_limit ~60s
        assert 25.0 <= backoffs["video6"] <= 35.0      # server ~30s
        assert 15.0 <= backoffs["video2"] <= 25.0       # network ~20s
        assert 8.0 <= backoffs["video3"] <= 12.0       # format ~10s

    def test_category_backoff_with_checkpoint_persistence(self):
        """Test that category backoff state persists through checkpoint save/restore."""
        budget = DownloadRetryBudget()

        # Record attempts with different categories
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video2", error_category="network")

        # Verify attempts are tracked correctly before checkpoint
        assert budget._video_states["video1"].attempts_by_category.get("rate_limit", 0) == 2
        assert budget._video_states["video2"].attempts_by_category.get("network", 0) == 1

        # Save checkpoint
        checkpoint = budget.to_checkpoint_dict()

        # Verify checkpoint contains category data
        assert "attempts_by_category" in checkpoint["video_states"]["video1"]
        assert checkpoint["video_states"]["video1"]["attempts_by_category"]["rate_limit"] == 2

        # Create new budget and restore
        new_budget = DownloadRetryBudget()
        new_budget.from_checkpoint_dict(checkpoint)

        # Verify attempts are restored correctly
        assert new_budget._video_states["video1"].attempts_by_category.get("rate_limit", 0) == 2
        assert new_budget._video_states["video2"].attempts_by_category.get("network", 0) == 1

        # Get backoffs after restore - should calculate based on restored state
        # Both budgets should now have same attempt counts, so backoffs will be similar
        # (may differ slightly due to new jitter calculation, but within same order of magnitude)
        backoff1_after = new_budget.calculate_category_backoff("video1", "rate_limit")
        backoff2_after = new_budget.calculate_category_backoff("video2", "network")

        # Verify the backoff values are in reasonable ranges based on restored attempt counts
        # rate_limit with 2 attempts: 30 * 2^2 = 120, with jitter ~108-132
        # network with 1 attempt: 10 * 2^1 = 20, with jitter ~18-22
        assert 100.0 <= backoff1_after <= 140.0
        assert 15.0 <= backoff2_after <= 25.0

    def test_success_reduction_respects_category(self):
        """Test that successful retry reduction is category-specific."""
        config = DownloadRetryBudgetConfig(reduce_backoff_on_success=True)
        budget = DownloadRetryBudget(config)

        # Record attempts for two different categories
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="rate_limit")
        budget.record_attempt("video1", error_category="network")
        budget.record_attempt("video1", error_category="network")

        # Record success only for rate_limit
        budget.record_successful_retry("video1", "rate_limit")

        # Get backoffs
        rate_limit_backoff = budget.calculate_category_backoff("video1", "rate_limit")
        network_backoff = budget.calculate_category_backoff("video1", "network")

        # rate_limit had 2 attempts + 1 success = effective 1 attempt = 30 * 2^1 = 60s
        # network had 2 attempts = 10 * 2^2 = 40s
        # Note: rate_limit still higher than network because base is 3x higher
        # But reduction does occur - without success, rate_limit would be 30 * 2^2 = 120s
        assert rate_limit_backoff > network_backoff  # rate_limit still higher due to higher base
        assert 50.0 <= rate_limit_backoff <= 70.0    # 60s with jitter
        assert 35.0 <= network_backoff <= 45.0       # 40s with jitter

        # Verify that without success_reduction, rate_limit would be higher
        # (this is implicit - we can see that reduction happened because it's 60 not 120)
