"""Tests for RateLimitMetricsAggregator (US-004 Sprint 10).

Tests:
- aggregate() collects from all subsystems
- trigger_categories use classify_trigger() breakdown
- get_health_status() returns healthy/degraded/critical
- Pipeline integration via print_report()
- Safe fallback when subsystems are None
"""

import pytest
from unittest.mock import MagicMock, patch

from src.downloader.rate_limit_metrics import RateLimitMetricsAggregator


class TestAggregateMethod:
    """Test aggregate() collects from all subsystems."""

    def test_aggregate_empty_subsystems(self):
        """aggregate() returns empty dicts when no subsystems provided."""
        agg = RateLimitMetricsAggregator()
        result = agg.aggregate()

        assert result['escalation'] == {}
        assert result['cookies'] == {}
        assert result['budget'] == {}
        assert result['circuit_breaker'] == {}
        assert result['trigger_categories'] == {}

    def test_aggregate_with_escalation_manager(self):
        """aggregate() includes escalation metrics when manager provided."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'total_escalations': 5,
            'total_403s': 10,
            'total_successes': 50,
            'average_tier': 1.5,
            'escalations_per_tier': {'EXTRACTOR_ARGS': 3, 'FULL_BYPASS': 2},
            'keywords_at_each_tier': {
                'IMPERSONATE': ['kw1'],
                'EXTRACTOR_ARGS': ['kw2'],
                'FULL_BYPASS': ['kw3'],
            },
        }

        agg = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        result = agg.aggregate()

        assert result['escalation']['total_escalations'] == 5
        assert result['escalation']['total_403s'] == 10
        assert result['escalation']['average_tier'] == 1.5

    def test_aggregate_with_cookie_rotator(self):
        """aggregate() includes cookie status when rotator provided."""
        mock_cookie = MagicMock()
        mock_cookie.get_status.return_value = {
            'enabled': True,
            'rotation_count': 3,
            'valid_cookies': 4,
            'strategy': 'on_error',
        }

        agg = RateLimitMetricsAggregator(cookie_rotator=mock_cookie)
        result = agg.aggregate()

        assert result['cookies']['enabled'] is True
        assert result['cookies']['rotation_count'] == 3

    def test_aggregate_with_rate_limit_budget(self):
        """aggregate() includes budget data when budget provided."""
        mock_budget = MagicMock()
        mock_budget.to_dict.return_value = {
            'rotations_used': 3,
            'vpn_switches_used': 1,
            'backoff_time_spent': 120.5,
            'max_rotations': 10,
            'max_backoff_time': 600,
        }

        agg = RateLimitMetricsAggregator(rate_limit_budget=mock_budget)
        result = agg.aggregate()

        assert result['budget']['rotations_used'] == 3
        assert result['budget']['max_rotations'] == 10

    def test_aggregate_with_circuit_breaker(self):
        """aggregate() includes circuit breaker stats when provided."""
        mock_cb = MagicMock()
        mock_cb.get_stats.return_value = {
            'enabled': True,
            'is_open': False,
            'total_trips': 2,
            'total_paused_seconds': 120.0,
        }

        agg = RateLimitMetricsAggregator(circuit_breaker=mock_cb)
        result = agg.aggregate()

        assert result['circuit_breaker']['total_trips'] == 2
        assert result['circuit_breaker']['is_open'] is False

    def test_aggregate_all_subsystems(self):
        """aggregate() collects from all 4 subsystems simultaneously."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {'total_escalations': 3}

        mock_cookie = MagicMock()
        mock_cookie.get_status.return_value = {'rotation_count': 2}

        mock_budget = MagicMock()
        mock_budget.to_dict.return_value = {'rotations_used': 1}

        mock_cb = MagicMock()
        mock_cb.get_stats.return_value = {'total_trips': 1}

        agg = RateLimitMetricsAggregator(
            escalation_manager=mock_esc,
            cookie_rotator=mock_cookie,
            rate_limit_budget=mock_budget,
            circuit_breaker=mock_cb,
        )
        result = agg.aggregate()

        assert result['escalation']['total_escalations'] == 3
        assert result['cookies']['rotation_count'] == 2
        assert result['budget']['rotations_used'] == 1
        assert result['circuit_breaker']['total_trips'] == 1

    def test_aggregate_handles_subsystem_error(self):
        """aggregate() returns empty dict when subsystem raises exception."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.side_effect = RuntimeError("connection lost")

        agg = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        result = agg.aggregate()

        assert result['escalation'] == {}


class TestTriggerCategories:
    """Test trigger_categories using classify_trigger()."""

    def test_record_trigger_403(self):
        """record_trigger() categorizes 403 errors."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("ERROR: HTTP Error 403: Forbidden")
        result = agg.aggregate()

        assert '403' in result['trigger_categories']
        assert result['trigger_categories']['403'] == 1

    def test_record_trigger_429(self):
        """record_trigger() categorizes 429 rate-limit errors."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("ERROR: HTTP Error 429: Too Many Requests")
        result = agg.aggregate()

        assert '429' in result['trigger_categories']
        assert result['trigger_categories']['429'] == 1

    def test_record_trigger_bot_detection(self):
        """record_trigger() categorizes bot detection."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("ERROR: verify you are human")
        result = agg.aggregate()

        assert 'bot_detection' in result['trigger_categories']

    def test_record_trigger_ip_blocked(self):
        """record_trigger() categorizes IP blocking."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("ERROR: access denied")
        result = agg.aggregate()

        assert 'ip_blocked' in result['trigger_categories']

    def test_record_trigger_age_gate(self):
        """record_trigger() categorizes age-gate triggers."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("Sign in to confirm your age")
        result = agg.aggregate()

        assert 'age_gate' in result['trigger_categories']

    def test_record_trigger_multiple_categories(self):
        """record_trigger() accumulates across multiple categories."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("HTTP Error 403: Forbidden")
        agg.record_trigger("HTTP Error 403: Forbidden")
        agg.record_trigger("HTTP Error 429: Too Many Requests")
        agg.record_trigger("access denied")

        result = agg.aggregate()
        cats = result['trigger_categories']

        assert cats.get('403', 0) == 2
        assert cats.get('429', 0) == 1
        assert cats.get('ip_blocked', 0) == 1

    def test_record_trigger_no_match_ignored(self):
        """record_trigger() ignores unrecognized errors."""
        agg = RateLimitMetricsAggregator()
        agg.record_trigger("ERROR: video not available")
        agg.record_trigger("")

        result = agg.aggregate()
        assert result['trigger_categories'] == {}


class TestHealthStatus:
    """Test get_health_status() with threshold logic."""

    def test_healthy_no_subsystems(self):
        """get_health_status() returns 'healthy' with no subsystems."""
        agg = RateLimitMetricsAggregator()
        assert agg.get_health_status() == 'healthy'

    def test_healthy_low_escalation(self):
        """get_health_status() returns 'healthy' when most keywords at Tier 1."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'average_tier': 1.2,
            'keywords_at_each_tier': {
                'IMPERSONATE': ['kw1', 'kw2', 'kw3', 'kw4', 'kw5'],
            },
        }

        agg = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        assert agg.get_health_status() == 'healthy'

    def test_degraded_moderate_escalation(self):
        """get_health_status() returns 'degraded' with significant escalation."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'average_tier': 2.0,
            'keywords_at_each_tier': {
                'IMPERSONATE': ['kw1', 'kw2'],
                'EXTRACTOR_ARGS': ['kw3', 'kw4'],
                'FULL_BYPASS': ['kw5'],
            },
        }

        agg = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        assert agg.get_health_status() == 'degraded'

    def test_critical_high_escalation(self):
        """get_health_status() returns 'critical' when >50% at max tier."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'average_tier': 2.8,
            'keywords_at_each_tier': {
                'IMPERSONATE': ['kw1'],
                'FULL_BYPASS': ['kw2', 'kw3', 'kw4'],
            },
        }

        agg = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        assert agg.get_health_status() == 'critical'

    def test_critical_budget_exhausted(self):
        """get_health_status() returns 'critical' when budget nearly exhausted."""
        mock_budget = MagicMock()
        mock_budget.to_dict.return_value = {
            'rotations_used': 9,
            'max_rotations': 10,
            'backoff_time_spent': 550.0,
            'max_backoff_time': 600,
        }

        agg = RateLimitMetricsAggregator(rate_limit_budget=mock_budget)
        assert agg.get_health_status() == 'critical'

    def test_degraded_budget_half_consumed(self):
        """get_health_status() returns 'degraded' when budget half consumed."""
        mock_budget = MagicMock()
        mock_budget.to_dict.return_value = {
            'rotations_used': 5,
            'max_rotations': 10,
            'backoff_time_spent': 100.0,
            'max_backoff_time': 600,
        }

        agg = RateLimitMetricsAggregator(rate_limit_budget=mock_budget)
        assert agg.get_health_status() == 'degraded'

    def test_critical_circuit_breaker_open(self):
        """get_health_status() returns 'critical' when circuit breaker is open."""
        mock_cb = MagicMock()
        mock_cb.get_stats.return_value = {
            'is_open': True,
            'total_trips': 5,
        }

        agg = RateLimitMetricsAggregator(circuit_breaker=mock_cb)
        assert agg.get_health_status() == 'critical'

    def test_degraded_circuit_breaker_frequent_trips(self):
        """get_health_status() returns 'degraded' with many circuit trips."""
        mock_cb = MagicMock()
        mock_cb.get_stats.return_value = {
            'is_open': False,
            'total_trips': 4,
        }

        agg = RateLimitMetricsAggregator(circuit_breaker=mock_cb)
        assert agg.get_health_status() == 'degraded'

    def test_combined_signals_take_worst(self):
        """get_health_status() uses worst signal across all subsystems."""
        # Escalation: healthy (all Tier 1)
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'average_tier': 1.0,
            'keywords_at_each_tier': {'IMPERSONATE': ['kw1']},
        }
        # Budget: critical (90% consumed)
        mock_budget = MagicMock()
        mock_budget.to_dict.return_value = {
            'rotations_used': 9, 'max_rotations': 10,
            'backoff_time_spent': 0, 'max_backoff_time': 600,
        }

        agg = RateLimitMetricsAggregator(
            escalation_manager=mock_esc,
            rate_limit_budget=mock_budget,
        )
        assert agg.get_health_status() == 'critical'

    def test_health_safe_with_failing_subsystem(self):
        """get_health_status() doesn't crash when subsystem raises."""
        mock_esc = MagicMock()
        mock_esc.get_metrics.side_effect = RuntimeError("crash")

        agg = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        # Should not raise, falls back to healthy
        assert agg.get_health_status() == 'healthy'


class TestPipelineIntegration:
    """Test aggregator integration with orchestrator print_report()."""

    def test_print_report_with_aggregator(self, capsys):
        """print_report() uses aggregator when set."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import patch as _patch

        mock_config = MagicMock()
        mock_config.healing = None

        with _patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        # Create aggregator with mock escalation
        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'total_escalations': 3,
            'total_403s': 8,
            'total_successes': 42,
            'average_tier': 1.5,
            'escalations_per_tier': {'EXTRACTOR_ARGS': 3},
            'keywords_at_each_tier': {'IMPERSONATE': ['kw1'], 'EXTRACTOR_ARGS': ['kw2']},
        }

        aggregator = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        orchestrator.set_aggregated_metrics(aggregator)
        orchestrator.print_report()

        captured = capsys.readouterr()
        assert 'UNIFIED RATE-LIMIT STATUS: HEALTHY' in captured.out
        assert 'Total 403/bot errors: 8' in captured.out

    def test_print_report_shows_health_critical(self, capsys):
        """print_report() shows CRITICAL status appropriately."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import patch as _patch

        mock_config = MagicMock()
        mock_config.healing = None

        with _patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        mock_esc = MagicMock()
        mock_esc.get_metrics.return_value = {
            'total_escalations': 10,
            'total_403s': 20,
            'total_successes': 5,
            'average_tier': 2.8,
            'escalations_per_tier': {'FULL_BYPASS': 10},
            'keywords_at_each_tier': {
                'IMPERSONATE': ['kw1'],
                'FULL_BYPASS': ['kw2', 'kw3', 'kw4'],
            },
        }

        aggregator = RateLimitMetricsAggregator(escalation_manager=mock_esc)
        orchestrator.set_aggregated_metrics(aggregator)
        orchestrator.print_report()

        captured = capsys.readouterr()
        assert 'UNIFIED RATE-LIMIT STATUS: CRITICAL' in captured.out

    def test_reset_clears_aggregated_metrics(self):
        """reset() clears the aggregated metrics."""
        from src.agents.orchestrator import HealingOrchestrator
        from unittest.mock import patch as _patch

        mock_config = MagicMock()
        mock_config.healing = None

        with _patch('src.agents.orchestrator.HEALER_REGISTRY', []):
            orchestrator = HealingOrchestrator(
                config=mock_config,
                project_dir='/tmp/test'
            )

        aggregator = RateLimitMetricsAggregator()
        orchestrator.set_aggregated_metrics(aggregator)
        assert orchestrator._aggregated_metrics is not None

        orchestrator.reset()
        assert orchestrator._aggregated_metrics is None
