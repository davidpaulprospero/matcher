"""Comprehensive tests for rate_limit_handler.py - unified rate limit handling.

Tests cover:
- Happy path (normal usage)
- Edge cases (empty, null, boundary values)
- Error cases (invalid input, failures, exceptions)
- Branch coverage for all conditional paths
"""

import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.downloader.rate_limit_handler import (
    RateLimitStrategy,
    RateLimitStats,
    RateLimitHandler,
    get_rate_limit_handler,
    reset_rate_limit_handler,
)
from src.downloader.proxy_manager import ProxyStatus
from src.downloader.vpn_manager import VPNProvider, VPNStatus, VPNConnection


# ============================================================================
# RateLimitStrategy Enum Tests
# ============================================================================

class TestRateLimitStrategy:
    """Tests for RateLimitStrategy enum."""

    def test_strategy_values(self):
        """All strategies should have unique values."""
        values = [s.value for s in RateLimitStrategy]
        assert len(values) == len(set(values))

    def test_all_strategies(self):
        """Should have all expected strategies."""
        strategies = list(RateLimitStrategy)
        assert RateLimitStrategy.PROXY_ROTATION in strategies
        assert RateLimitStrategy.VPN_ROTATION in strategies
        assert RateLimitStrategy.BACKOFF in strategies


# ============================================================================
# RateLimitStats Tests
# ============================================================================

class TestRateLimitStats:
    """Tests for RateLimitStats dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        stats = RateLimitStats()
        assert stats.rate_limits_total == 0
        assert stats.proxy_rotations == 0
        assert stats.vpn_rotations == 0
        assert stats.backoffs_applied == 0
        assert stats.total_backoff_seconds == 0.0
        assert stats.last_rate_limit == 0.0
        assert stats.last_strategy_used is None

    def test_custom_values(self):
        """Should accept custom values."""
        stats = RateLimitStats(
            rate_limits_total=10,
            proxy_rotations=5,
            vpn_rotations=2,
            backoffs_applied=3,
            total_backoff_seconds=120.0,
            last_strategy_used=RateLimitStrategy.PROXY_ROTATION,
        )
        assert stats.rate_limits_total == 10
        assert stats.proxy_rotations == 5
        assert stats.last_strategy_used == RateLimitStrategy.PROXY_ROTATION


# ============================================================================
# RateLimitHandler Tests
# ============================================================================

class TestRateLimitHandler:
    """Tests for RateLimitHandler class."""

    # --- Initialization Tests ---

    def test_init_no_config(self):
        """Should initialize without config."""
        handler = RateLimitHandler()
        assert handler.has_proxy is False
        assert handler.has_vpn is False

    def test_init_with_backoff_settings(self):
        """Should accept custom backoff settings."""
        handler = RateLimitHandler(
            base_backoff=60.0,
            max_backoff=600.0,
            backoff_multiplier=3.0,
        )
        assert handler._base_backoff == 60.0
        assert handler._max_backoff == 600.0
        assert handler._backoff_multiplier == 3.0

    def test_init_with_proxy_config(self):
        """Should initialize proxy manager from config."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {
            'enabled': True,
            'sources': [{'type': 'url', 'url': 'http://proxy:8080'}]
        }
        mock_config.download.fallback.vpn = {'enabled': False}

        handler = RateLimitHandler(mock_config)
        assert handler.has_proxy is True

    def test_init_with_vpn_config(self):
        """Should initialize VPN manager from config."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {'enabled': False}
        mock_config.download.fallback.vpn = {
            'enabled': True,
            'provider': 'nordvpn'
        }

        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            handler = RateLimitHandler(mock_config)
            assert handler.has_vpn is True

    def test_init_with_both(self):
        """Should initialize both proxy and VPN managers."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {
            'enabled': True,
            'sources': [{'type': 'url', 'url': 'http://proxy:8080'}]
        }
        mock_config.download.fallback.vpn = {
            'enabled': True,
            'provider': 'nordvpn'
        }

        with patch('shutil.which', return_value='/usr/bin/nordvpn'):
            handler = RateLimitHandler(mock_config)
            assert handler.has_proxy is True
            assert handler.has_vpn is True

    def test_init_config_no_fallback(self):
        """Should handle missing fallback config."""
        mock_config = Mock()
        mock_config.download.fallback = None

        handler = RateLimitHandler(mock_config)
        assert handler.has_proxy is False
        assert handler.has_vpn is False

    def test_init_config_error(self):
        """Should handle config errors gracefully."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = Mock(side_effect=Exception("Error"))

        # Should not raise
        handler = RateLimitHandler(mock_config)

    # --- get_proxy Tests ---

    def test_get_proxy_with_manager(self):
        """Should return proxy when manager configured."""
        handler = RateLimitHandler()
        handler._proxy_manager = Mock()
        handler._proxy_manager.get_proxy.return_value = "http://proxy:8080"

        result = handler.get_proxy()
        assert result == "http://proxy:8080"

    def test_get_proxy_no_manager(self):
        """Should return None without proxy manager."""
        handler = RateLimitHandler()
        assert handler.get_proxy() is None

    # --- get_socks5_proxy Tests ---

    def test_get_socks5_proxy_from_vpn(self):
        """Should return SOCKS5 proxy from VPN when available."""
        handler = RateLimitHandler()
        handler._vpn_manager = Mock()
        handler._vpn_manager.socks5_proxy = "socks5://127.0.0.1:1080"

        result = handler.get_socks5_proxy()
        assert result == "socks5://127.0.0.1:1080"

    def test_get_socks5_proxy_fallback_to_proxy(self):
        """Should fallback to regular proxy when no SOCKS5."""
        handler = RateLimitHandler()
        handler._vpn_manager = Mock()
        handler._vpn_manager.socks5_proxy = None
        handler._proxy_manager = Mock()
        handler._proxy_manager.get_proxy.return_value = "http://proxy:8080"

        result = handler.get_socks5_proxy()
        assert result == "http://proxy:8080"

    def test_get_socks5_proxy_none(self):
        """Should return None when nothing available."""
        handler = RateLimitHandler()
        assert handler.get_socks5_proxy() is None

    # --- on_rate_limit Tests ---

    def test_on_rate_limit_increments_stats(self):
        """Should increment rate limit stats."""
        handler = RateLimitHandler()

        with patch.object(handler, '_apply_backoff'):
            handler.on_rate_limit(source="test")

        assert handler.stats.rate_limits_total == 1
        assert handler._consecutive_rate_limits == 1

    def test_on_rate_limit_uses_proxy_first(self):
        """Should try proxy rotation first."""
        handler = RateLimitHandler()

        # Mock proxy manager with multiple proxies
        mock_proxy = Mock()
        mock_proxy.has_proxies = True
        mock_proxy.available_count = 3
        mock_proxy.get_proxy.return_value = "http://proxy2:8080"
        handler._proxy_manager = mock_proxy

        result = handler.on_rate_limit(source="test")

        assert result == RateLimitStrategy.PROXY_ROTATION
        assert handler.stats.proxy_rotations == 1

    def test_on_rate_limit_uses_vpn_when_proxy_exhausted(self):
        """Should use VPN when proxies exhausted."""
        handler = RateLimitHandler()

        # Mock exhausted proxy manager
        mock_proxy = Mock()
        mock_proxy.has_proxies = True
        mock_proxy.available_count = 1  # Only one left
        handler._proxy_manager = mock_proxy

        # Mock VPN helper
        mock_vpn_helper = Mock()
        mock_vpn_helper.should_rotate.return_value = True
        mock_vpn_helper.rotate_on_rate_limit.return_value = True
        handler._vpn_helper = mock_vpn_helper

        result = handler.on_rate_limit(source="test")

        assert result == RateLimitStrategy.VPN_ROTATION
        assert handler.stats.vpn_rotations == 1

    def test_on_rate_limit_uses_backoff_as_fallback(self):
        """Should use backoff when no other options."""
        handler = RateLimitHandler()

        with patch.object(handler, '_apply_backoff') as mock_backoff:
            result = handler.on_rate_limit(source="test")

        assert result == RateLimitStrategy.BACKOFF
        assert handler.stats.backoffs_applied == 1
        mock_backoff.assert_called_once()

    def test_on_rate_limit_tracks_last_rate_limit_time(self):
        """Should track time of last rate limit."""
        handler = RateLimitHandler()

        before = time.time()
        with patch.object(handler, '_apply_backoff'):
            handler.on_rate_limit()
        after = time.time()

        assert before <= handler.stats.last_rate_limit <= after

    def test_on_rate_limit_tracks_strategy(self):
        """Should track last strategy used."""
        handler = RateLimitHandler()

        with patch.object(handler, '_apply_backoff'):
            handler.on_rate_limit()

        assert handler.stats.last_strategy_used == RateLimitStrategy.BACKOFF

    # --- _try_proxy_rotation Tests ---

    def test_try_proxy_rotation_success(self):
        """Should rotate proxy successfully."""
        handler = RateLimitHandler()

        mock_proxy = Mock()
        mock_proxy.available_count = 3
        mock_proxy.get_proxy.return_value = "http://proxy2:8080"
        handler._proxy_manager = mock_proxy

        result = handler._try_proxy_rotation()

        assert result is True
        mock_proxy.report_rate_limit.assert_called_once()

    def test_try_proxy_rotation_no_manager(self):
        """Should return False without proxy manager."""
        handler = RateLimitHandler()
        assert handler._try_proxy_rotation() is False

    def test_try_proxy_rotation_no_proxies(self):
        """Should return False with only one proxy."""
        handler = RateLimitHandler()

        mock_proxy = Mock()
        mock_proxy.available_count = 1
        handler._proxy_manager = mock_proxy

        result = handler._try_proxy_rotation()
        assert result is False

    def test_try_proxy_rotation_no_next(self):
        """Should return False when get_proxy returns None."""
        handler = RateLimitHandler()

        mock_proxy = Mock()
        mock_proxy.available_count = 3
        mock_proxy.get_proxy.return_value = None
        handler._proxy_manager = mock_proxy

        result = handler._try_proxy_rotation()
        assert result is False

    # --- _try_vpn_rotation Tests ---

    def test_try_vpn_rotation_success(self):
        """Should rotate VPN successfully."""
        handler = RateLimitHandler()

        mock_vpn_helper = Mock()
        mock_vpn_helper.should_rotate.return_value = True
        mock_vpn_helper.rotate_on_rate_limit.return_value = True
        handler._vpn_helper = mock_vpn_helper

        result = handler._try_vpn_rotation()

        assert result is True
        mock_vpn_helper.rotate_on_rate_limit.assert_called_once()

    def test_try_vpn_rotation_resets_proxy(self):
        """Should reset proxy manager after VPN rotation."""
        handler = RateLimitHandler()

        mock_vpn_helper = Mock()
        mock_vpn_helper.should_rotate.return_value = True
        mock_vpn_helper.rotate_on_rate_limit.return_value = True
        handler._vpn_helper = mock_vpn_helper

        mock_proxy = Mock()
        handler._proxy_manager = mock_proxy

        handler._try_vpn_rotation()

        mock_proxy.reset.assert_called_once()

    def test_try_vpn_rotation_no_helper(self):
        """Should return False without VPN helper."""
        handler = RateLimitHandler()
        assert handler._try_vpn_rotation() is False

    def test_try_vpn_rotation_blocked(self):
        """Should return False when rotation blocked."""
        handler = RateLimitHandler()

        mock_vpn_helper = Mock()
        mock_vpn_helper.should_rotate.return_value = False
        handler._vpn_helper = mock_vpn_helper

        result = handler._try_vpn_rotation()
        assert result is False

    def test_try_vpn_rotation_failure(self):
        """Should return False when rotation fails."""
        handler = RateLimitHandler()

        mock_vpn_helper = Mock()
        mock_vpn_helper.should_rotate.return_value = True
        mock_vpn_helper.rotate_on_rate_limit.return_value = False
        handler._vpn_helper = mock_vpn_helper

        result = handler._try_vpn_rotation()
        assert result is False

    # --- _apply_backoff Tests ---

    def test_apply_backoff_sleeps(self):
        """Should sleep for backoff duration."""
        handler = RateLimitHandler(base_backoff=1.0, max_backoff=100.0)

        with patch('time.sleep') as mock_sleep:
            handler._apply_backoff()
            mock_sleep.assert_called_once_with(1.0)

    def test_apply_backoff_tracks_total(self):
        """Should track total backoff seconds."""
        handler = RateLimitHandler(base_backoff=5.0)

        with patch('time.sleep'):
            handler._apply_backoff()
            handler._apply_backoff()

        assert handler.stats.total_backoff_seconds >= 10.0

    def test_apply_backoff_exponential(self):
        """Should increase backoff exponentially."""
        handler = RateLimitHandler(
            base_backoff=1.0,
            max_backoff=100.0,
            backoff_multiplier=2.0,
        )

        backoffs = []
        with patch('time.sleep') as mock_sleep:
            for _ in range(4):
                handler._apply_backoff()
                backoffs.append(mock_sleep.call_args[0][0])

        assert backoffs == [1.0, 2.0, 4.0, 8.0]

    def test_apply_backoff_respects_max(self):
        """Should not exceed max backoff."""
        handler = RateLimitHandler(
            base_backoff=100.0,
            max_backoff=50.0,  # Lower than base
        )

        with patch('time.sleep') as mock_sleep:
            handler._apply_backoff()
            mock_sleep.assert_called_once_with(50.0)

    # --- on_success Tests ---

    def test_on_success_resets_consecutive(self):
        """Should reset consecutive rate limit count."""
        handler = RateLimitHandler()
        handler._consecutive_rate_limits = 5

        handler.on_success()

        assert handler._consecutive_rate_limits == 0

    def test_on_success_resets_backoff(self):
        """Should reset backoff to base."""
        handler = RateLimitHandler(base_backoff=30.0)
        handler._current_backoff = 240.0  # Was escalated

        handler.on_success()

        assert handler._current_backoff == 30.0

    def test_on_success_reports_proxy_success(self):
        """Should report success to proxy manager."""
        handler = RateLimitHandler()
        mock_proxy = Mock()
        handler._proxy_manager = mock_proxy

        handler.on_success()

        mock_proxy.report_success.assert_called_once()

    def test_on_success_no_proxy(self):
        """Should not fail without proxy manager."""
        handler = RateLimitHandler()
        handler.on_success()  # Should not raise

    # --- reset Tests ---

    def test_reset_clears_state(self):
        """Should reset all state."""
        handler = RateLimitHandler(base_backoff=30.0)
        handler._consecutive_rate_limits = 10
        handler._current_backoff = 240.0
        handler.stats.rate_limits_total = 50

        handler.reset()

        assert handler._consecutive_rate_limits == 0
        assert handler._current_backoff == 30.0
        assert handler.stats.rate_limits_total == 0

    def test_reset_resets_proxy_manager(self):
        """Should reset proxy manager."""
        handler = RateLimitHandler()
        mock_proxy = Mock()
        handler._proxy_manager = mock_proxy

        handler.reset()

        mock_proxy.reset.assert_called_once()

    # --- get_stats Tests ---

    def test_get_stats_basic(self):
        """Should return basic statistics."""
        handler = RateLimitHandler()

        with patch('time.sleep'):
            handler.on_rate_limit()

        stats = handler.get_stats()

        assert "rate_limits_total" in stats
        assert "proxy_rotations" in stats
        assert "vpn_rotations" in stats
        assert "backoffs_applied" in stats
        assert "has_proxy" in stats
        assert "has_vpn" in stats

    def test_get_stats_includes_proxy_stats(self):
        """Should include proxy manager stats."""
        handler = RateLimitHandler()
        mock_proxy = Mock()
        mock_proxy.get_stats.return_value = {"total": 5}
        handler._proxy_manager = mock_proxy

        stats = handler.get_stats()

        assert "proxy_stats" in stats
        assert stats["proxy_stats"]["total"] == 5

    def test_get_stats_includes_vpn_stats(self):
        """Should include VPN manager stats."""
        handler = RateLimitHandler()
        mock_vpn = Mock()
        mock_vpn.get_stats.return_value = {"provider": "nordvpn"}
        handler._vpn_manager = mock_vpn

        stats = handler.get_stats()

        assert "vpn_stats" in stats
        assert stats["vpn_stats"]["provider"] == "nordvpn"

    # --- log_summary Tests ---

    def test_log_summary_no_rate_limits(self):
        """Should handle no rate limits gracefully."""
        handler = RateLimitHandler()
        handler.log_summary()  # Should not raise

    def test_log_summary_with_rate_limits(self):
        """Should log summary of rate limiting."""
        handler = RateLimitHandler()
        handler.stats.rate_limits_total = 10
        handler.stats.proxy_rotations = 5
        handler.stats.vpn_rotations = 2
        handler.stats.backoffs_applied = 3
        handler.stats.total_backoff_seconds = 120.0

        handler.log_summary()  # Should not raise

    # --- Properties Tests ---

    def test_has_proxy_true(self):
        """has_proxy should return True when configured."""
        handler = RateLimitHandler()
        mock_proxy = Mock()
        mock_proxy.has_proxies = True
        handler._proxy_manager = mock_proxy

        assert handler.has_proxy is True

    def test_has_proxy_false(self):
        """has_proxy should return False when not configured."""
        handler = RateLimitHandler()
        assert handler.has_proxy is False

    def test_has_proxy_false_empty(self):
        """has_proxy should return False when pool empty."""
        handler = RateLimitHandler()
        mock_proxy = Mock()
        mock_proxy.has_proxies = False
        handler._proxy_manager = mock_proxy

        assert handler.has_proxy is False

    def test_has_vpn_true(self):
        """has_vpn should return True when configured."""
        handler = RateLimitHandler()
        mock_vpn = Mock()
        mock_vpn.is_available = True
        handler._vpn_manager = mock_vpn

        assert handler.has_vpn is True

    def test_has_vpn_false(self):
        """has_vpn should return False when not configured."""
        handler = RateLimitHandler()
        assert handler.has_vpn is False


# ============================================================================
# Global Handler Functions Tests
# ============================================================================

class TestGlobalHandlerFunctions:
    """Tests for global handler functions."""

    def test_get_rate_limit_handler_creates(self):
        """Should create handler if none exists."""
        reset_rate_limit_handler()  # Clear any existing

        handler = get_rate_limit_handler()
        assert handler is not None

    def test_get_rate_limit_handler_reuses(self):
        """Should reuse existing handler."""
        reset_rate_limit_handler()

        handler1 = get_rate_limit_handler()
        handler2 = get_rate_limit_handler()

        assert handler1 is handler2

    def test_get_rate_limit_handler_with_config(self):
        """Should accept config on first call."""
        reset_rate_limit_handler()

        mock_config = Mock()
        mock_config.download.fallback = None

        handler = get_rate_limit_handler(mock_config)
        assert handler is not None

    def test_reset_rate_limit_handler(self):
        """Should reset global handler."""
        get_rate_limit_handler()  # Ensure one exists

        reset_rate_limit_handler()

        # Getting again should create new instance
        handler = get_rate_limit_handler()
        assert handler.stats.rate_limits_total == 0

    def test_reset_rate_limit_handler_when_none(self):
        """Should not fail when no handler exists."""
        reset_rate_limit_handler()
        reset_rate_limit_handler()  # Should not raise


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for rate limit handling flow."""

    def test_full_escalation_path(self):
        """Should escalate through all strategies."""
        handler = RateLimitHandler()

        # Mock proxy manager with controlled available_count
        mock_proxy = Mock()
        mock_proxy.available_count = 3
        mock_proxy.get_proxy.return_value = "http://proxy:8080"
        handler._proxy_manager = mock_proxy

        # First call - proxy rotation (3 proxies available)
        result1 = handler.on_rate_limit()
        assert result1 == RateLimitStrategy.PROXY_ROTATION

        # Exhaust proxies by setting available_count to 1
        mock_proxy.available_count = 1

        # Second call - backoff (only 1 proxy, no VPN configured)
        with patch('time.sleep'):
            result2 = handler.on_rate_limit()
        assert result2 == RateLimitStrategy.BACKOFF

    def test_multiple_rate_limits_track_correctly(self):
        """Should track multiple rate limits correctly."""
        handler = RateLimitHandler()

        with patch('time.sleep'):
            for _ in range(5):
                handler.on_rate_limit()

        assert handler.stats.rate_limits_total == 5
        assert handler._consecutive_rate_limits == 5

    def test_success_after_rate_limits(self):
        """Success should reset state after rate limits."""
        handler = RateLimitHandler(base_backoff=1.0, backoff_multiplier=2.0)

        with patch('time.sleep'):
            handler.on_rate_limit()
            handler.on_rate_limit()
            handler.on_rate_limit()

        # After 3 rate limits: 1 -> 2 -> 4 -> 8 (stored for next attempt)
        assert handler._current_backoff == 8.0
        assert handler._consecutive_rate_limits == 3

        handler.on_success()

        assert handler._current_backoff == 1.0  # Reset to base
        assert handler._consecutive_rate_limits == 0


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================

class TestEdgeCasesAndBoundaries:
    """Tests for edge cases and boundary conditions."""

    def test_rapid_rate_limits(self):
        """Should handle rapid rate limit events."""
        handler = RateLimitHandler()

        with patch('time.sleep'):
            for _ in range(100):
                handler.on_rate_limit()

        assert handler.stats.rate_limits_total == 100
        assert handler.stats.backoffs_applied == 100

    def test_backoff_overflow_protection(self):
        """Should not overflow with many escalations."""
        handler = RateLimitHandler(
            base_backoff=1.0,
            max_backoff=1000.0,
            backoff_multiplier=10.0,
        )

        with patch('time.sleep'):
            for _ in range(100):
                handler.on_rate_limit()

        assert handler._current_backoff == 1000.0  # Capped at max

    def test_zero_backoff_settings(self):
        """Should handle zero backoff settings."""
        handler = RateLimitHandler(
            base_backoff=0.0,
            max_backoff=0.0,
        )

        with patch('time.sleep') as mock_sleep:
            handler.on_rate_limit()
            mock_sleep.assert_called_once_with(0.0)

    def test_source_parameter_in_on_rate_limit(self):
        """Should handle various source parameter values."""
        handler = RateLimitHandler()

        with patch('time.sleep'):
            handler.on_rate_limit(source="youtube")
            handler.on_rate_limit(source="")
            handler.on_rate_limit(source=None)

        assert handler.stats.rate_limits_total == 3

    def test_config_with_proxy_but_no_sources(self):
        """Should handle proxy config with empty sources."""
        mock_config = Mock()
        mock_config.download.fallback.proxy = {
            'enabled': True,
            'sources': []  # No sources
        }
        mock_config.download.fallback.vpn = {'enabled': False}

        handler = RateLimitHandler(mock_config)
        assert handler.has_proxy is False  # No actual proxies
