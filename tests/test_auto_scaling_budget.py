"""Tests for auto-scaling budget limits (US-005 Sprint 13).

Tests that RateLimitBudget.scale_for_keywords() scales budget limits
based on keyword count using ceil(n/5) multiplier, capped at 5x.
"""

import logging
import math
import pytest

from src.downloader.rate_limit_budget import RateLimitBudget


class TestScaleForKeywordsBase:
    """Test scale_for_keywords(1) returns base limits unchanged."""

    def test_single_keyword_no_scaling(self):
        """ceil(1/5) = 1x multiplier, limits unchanged."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(1)
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3

    def test_five_keywords_no_scaling(self):
        """ceil(5/5) = 1x multiplier, limits unchanged."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(5)
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3

    def test_zero_keywords_no_scaling(self):
        """ceil(0/5) = 0, clipped to 1x by min, no scaling."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(0)
        # ceil(0/5) = 0 which is <= 1, so no change
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3


class TestScaleForKeywords2x:
    """Test scale_for_keywords(10) returns 2x base limits."""

    def test_ten_keywords_doubles_limits(self):
        """ceil(10/5) = 2x multiplier."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(10)
        assert budget.max_rotations == 20
        assert budget.max_backoff_time == 1200.0
        assert budget.max_vpn_switches == 6

    def test_six_keywords_doubles_limits(self):
        """ceil(6/5) = 2x multiplier."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(6)
        assert budget.max_rotations == 20
        assert budget.max_backoff_time == 1200.0
        assert budget.max_vpn_switches == 6


class TestScaleForKeywords5x:
    """Test scale_for_keywords(25) returns 5x base limits."""

    def test_25_keywords_5x_limits(self):
        """ceil(25/5) = 5x multiplier."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(25)
        assert budget.max_rotations == 50
        assert budget.max_backoff_time == 3000.0
        assert budget.max_vpn_switches == 15

    def test_21_keywords_5x_limits(self):
        """ceil(21/5) = 5x multiplier."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(21)
        assert budget.max_rotations == 50
        assert budget.max_backoff_time == 3000.0
        assert budget.max_vpn_switches == 15


class TestScaleForKeywordsCapped:
    """Test scale_for_keywords(100) is capped at 5x."""

    def test_100_keywords_capped_at_5x(self):
        """ceil(100/5) = 20, but capped at 5x. max_rotations <= 50, max_backoff_time <= 3000."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(100)
        assert budget.max_rotations == 50
        assert budget.max_rotations <= 50
        assert budget.max_backoff_time == 3000.0
        assert budget.max_backoff_time <= 3000.0
        assert budget.max_vpn_switches == 15

    def test_500_keywords_capped_at_5x(self):
        """Even 500 keywords only get 5x."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(500)
        assert budget.max_rotations == 50
        assert budget.max_backoff_time == 3000.0
        assert budget.max_vpn_switches == 15

    def test_cap_prevents_20x(self):
        """Explicit check that 100 keywords don't produce 20x."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(100)
        # Without cap, would be 200, 12000, 60
        assert budget.max_rotations != 200
        assert budget.max_backoff_time != 12000.0
        assert budget.max_vpn_switches != 60


class TestAutoScaleDisabled:
    """Test auto_scale_budget=false ignores keyword count."""

    def test_auto_scale_false_preserves_base(self):
        """When auto_scale=False, limits stay at base regardless of keyword count."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(100, auto_scale=False)
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3

    def test_auto_scale_false_with_25_keywords(self):
        """Even 25 keywords don't scale when disabled."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(25, auto_scale=False)
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3

    def test_auto_scale_false_with_10_keywords(self):
        """10 keywords also don't scale when disabled."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(10, auto_scale=False)
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
        assert budget.max_vpn_switches == 3


class TestScaleForKeywordsLogging:
    """Test scaled budget logged at INFO level."""

    def test_scaling_logs_info(self, caplog):
        """Verify INFO log: 'Rate limit budget auto-scaled for {n} keywords: rotations={x}, backoff={y}s'."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        with caplog.at_level(logging.INFO, logger="src.downloader.rate_limit_budget"):
            budget.scale_for_keywords(10)
        assert any("Rate limit budget auto-scaled for 10 keywords" in r.message for r in caplog.records)
        assert any("rotations=20" in r.message for r in caplog.records)
        assert any("backoff=1200.0s" in r.message for r in caplog.records)

    def test_no_log_when_no_scaling(self, caplog):
        """No INFO log when multiplier is 1x (no scaling needed)."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        with caplog.at_level(logging.INFO, logger="src.downloader.rate_limit_budget"):
            budget.scale_for_keywords(1)
        assert not any("auto-scaled" in r.message for r in caplog.records)

    def test_no_log_when_disabled(self, caplog):
        """No INFO log when auto_scale=False."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        with caplog.at_level(logging.INFO, logger="src.downloader.rate_limit_budget"):
            budget.scale_for_keywords(100, auto_scale=False)
        assert not any("auto-scaled" in r.message for r in caplog.records)

    def test_log_contains_keyword_count(self, caplog):
        """Log message contains the actual keyword count."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        with caplog.at_level(logging.INFO, logger="src.downloader.rate_limit_budget"):
            budget.scale_for_keywords(25)
        assert any("for 25 keywords" in r.message for r in caplog.records)


class TestScaleForKeywordsEdgeCases:
    """Edge case tests for scale_for_keywords."""

    def test_unlimited_rotations_not_scaled(self):
        """max_rotations=0 (unlimited) stays unlimited after scaling."""
        budget = RateLimitBudget(max_rotations=0, max_backoff_time=600.0, max_vpn_switches=3)
        budget.scale_for_keywords(25)
        assert budget.max_rotations == 0  # Still unlimited

    def test_unlimited_backoff_not_scaled(self):
        """max_backoff_time=0 (unlimited) stays unlimited after scaling."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=0, max_vpn_switches=3)
        budget.scale_for_keywords(25)
        assert budget.max_backoff_time == 0  # Still unlimited
        assert budget.max_rotations == 50  # Other limits still scale

    def test_unlimited_vpn_not_scaled(self):
        """max_vpn_switches=0 (unlimited) stays unlimited after scaling."""
        budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=0)
        budget.scale_for_keywords(25)
        assert budget.max_vpn_switches == 0  # Still unlimited

    def test_multiplier_math_correctness(self):
        """Verify ceil(n/5) math for various keyword counts."""
        test_cases = [
            (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
            (6, 2), (10, 2),
            (11, 3), (15, 3),
            (16, 4), (20, 4),
            (21, 5), (25, 5),
            (26, 5),  # Capped at 5
            (100, 5),  # Capped at 5
        ]
        for keyword_count, expected_multiplier in test_cases:
            budget = RateLimitBudget(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
            budget.scale_for_keywords(keyword_count)
            actual_multiplier = min(math.ceil(keyword_count / 5), 5)
            if actual_multiplier <= 1:
                assert budget.max_rotations == 10, f"Failed for {keyword_count} keywords"
            else:
                assert budget.max_rotations == 10 * actual_multiplier, f"Failed for {keyword_count} keywords"


class TestAutoScaleConfigIntegration:
    """Test that auto_scale_budget config field exists and works."""

    def test_config_field_exists(self):
        """RateLimitBudgetConfig has auto_scale_budget field."""
        from src.config.sections.download import RateLimitBudgetConfig
        config = RateLimitBudgetConfig()
        assert hasattr(config, 'auto_scale_budget')
        assert config.auto_scale_budget is True  # Default is True

    def test_config_field_false(self):
        """RateLimitBudgetConfig accepts auto_scale_budget=False."""
        from src.config.sections.download import RateLimitBudgetConfig
        config = RateLimitBudgetConfig(auto_scale_budget=False)
        assert config.auto_scale_budget is False

    def test_from_config_then_scale(self):
        """Full integration: create budget from config, then scale."""
        from src.config.sections.download import RateLimitBudgetConfig
        config = RateLimitBudgetConfig(max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3)
        budget = RateLimitBudget.from_config(config)
        budget.scale_for_keywords(10, auto_scale=config.auto_scale_budget)
        assert budget.max_rotations == 20
        assert budget.max_backoff_time == 1200.0

    def test_from_config_disabled_no_scale(self):
        """Integration: config with auto_scale_budget=False prevents scaling."""
        from src.config.sections.download import RateLimitBudgetConfig
        config = RateLimitBudgetConfig(
            max_rotations=10, max_backoff_time=600.0, max_vpn_switches=3,
            auto_scale_budget=False
        )
        budget = RateLimitBudget.from_config(config)
        budget.scale_for_keywords(100, auto_scale=config.auto_scale_budget)
        assert budget.max_rotations == 10
        assert budget.max_backoff_time == 600.0
