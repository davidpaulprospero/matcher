"""Unit tests for healing metrics dashboard (US-64-005)."""

import pytest
from unittest.mock import MagicMock

from src.agents.strategy import HealingStrategy, HealingMetrics
from src.agents.orchestrator import HealingOrchestrator


class TestHealingMetricsEnhancements:
    """Test HealingMetrics enhancements for dashboard (US-64-005)."""

    @pytest.mark.fast
    def test_record_heal_with_time(self):
        """Test that record_heal stores heal time."""
        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", True, heal_time_ms=150.0)

        assert "api-healer" in metrics.heal_times_by_healer
        assert metrics.heal_times_by_healer["api-healer"] == [150.0]

    @pytest.mark.fast
    def test_record_heal_tracks_successful_by_healer(self):
        """Test that successful heals are tracked per healer."""
        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", True)
        metrics.record_heal("api-healer", "DOWNLOAD", False)
        metrics.record_heal("api-healer", "MATCH", True)

        assert metrics.successful_heals_by_healer["api-healer"] == 2
        assert metrics.heals_by_healer["api-healer"] == 3

    @pytest.mark.fast
    def test_record_error_category(self):
        """Test error category recording."""
        metrics = HealingMetrics()
        metrics.record_error_category("api")
        metrics.record_error_category("api")
        metrics.record_error_category("disk")

        assert metrics.error_categories["api"] == 2
        assert metrics.error_categories["disk"] == 1

    @pytest.mark.fast
    def test_get_heal_success_rate_zero_heals(self):
        """Test success rate returns 0 when no heals."""
        metrics = HealingMetrics()
        assert metrics.get_heal_success_rate() == 0.0

    @pytest.mark.fast
    def test_get_heal_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", True)
        metrics.record_heal("api-healer", "DOWNLOAD", True)
        metrics.record_heal("api-healer", "DOWNLOAD", False)

        # 2 successful out of 3 = 66.67%
        rate = metrics.get_heal_success_rate()
        assert 66.6 < rate < 66.7

    @pytest.mark.fast
    def test_get_average_heal_time_ms(self):
        """Test average heal time calculation."""
        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", True, heal_time_ms=100.0)
        metrics.record_heal("disk-healer", "OUTPUT", True, heal_time_ms=200.0)
        metrics.record_heal("api-healer", "MATCH", False, heal_time_ms=300.0)

        # Average of [100, 200, 300] = 200
        assert metrics.get_average_heal_time_ms() == 200.0

    @pytest.mark.fast
    def test_get_average_heal_time_ms_empty(self):
        """Test average heal time returns 0 when no times recorded."""
        metrics = HealingMetrics()
        assert metrics.get_average_heal_time_ms() == 0.0

    @pytest.mark.fast
    def test_get_healer_success_rate(self):
        """Test per-healer success rate calculation."""
        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", True)
        metrics.record_heal("api-healer", "DOWNLOAD", False)
        metrics.record_heal("disk-healer", "OUTPUT", True)

        # api-healer: 1 success, 2 total = 50%
        assert metrics.get_healer_success_rate("api-healer") == 50.0
        # disk-healer: 1 success, 1 total = 100%
        assert metrics.get_healer_success_rate("disk-healer") == 100.0

    @pytest.mark.fast
    def test_get_healer_success_rate_unknown(self):
        """Test per-healer success rate for unknown healer."""
        metrics = HealingMetrics()
        assert metrics.get_healer_success_rate("unknown-healer") == 0.0

    @pytest.mark.fast
    def test_get_healer_average_time_ms(self):
        """Test per-healer average time calculation."""
        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", True, heal_time_ms=100.0)
        metrics.record_heal("api-healer", "MATCH", True, heal_time_ms=200.0)
        metrics.record_heal("disk-healer", "OUTPUT", True, heal_time_ms=50.0)

        # api-healer average: (100 + 200) / 2 = 150
        assert metrics.get_healer_average_time_ms("api-healer") == 150.0
        # disk-healer average: 50 / 1 = 50
        assert metrics.get_healer_average_time_ms("disk-healer") == 50.0

    @pytest.mark.fast
    def test_get_healer_average_time_ms_unknown(self):
        """Test per-healer average time for unknown healer."""
        metrics = HealingMetrics()
        assert metrics.get_healer_average_time_ms("unknown-healer") == 0.0


from src.agents.base import Healer, HealerResult


class NoopHealer(Healer):
    """Minimal healer for testing."""
    name = "noop-healer"
    error_patterns = []

    def fix(self, error, state, stage_name):
        return HealerResult.failed("Noop")


class TestOrchestratorDashboard:
    """Test HealingOrchestrator dashboard methods (US-64-005)."""

    def _create_orchestrator(self) -> HealingOrchestrator:
        """Create orchestrator with mocked config to avoid real healer init."""
        config = MagicMock()
        config.healing = None
        strategy = HealingStrategy()
        # Pass NoopHealer to avoid initializing real healers
        orchestrator = HealingOrchestrator(
            config, "/tmp/project", strategy=strategy, healers=[NoopHealer]
        )
        return orchestrator

    @pytest.mark.fast
    def test_get_dashboard_metrics_empty(self):
        """Test get_dashboard_metrics with no heals."""
        orchestrator = self._create_orchestrator()
        metrics = orchestrator.get_dashboard_metrics()

        assert metrics['success_rate'] == 0.0
        assert metrics['average_heal_time_ms'] == 0.0
        assert metrics['total_heals'] == 0
        assert metrics['healer_utilization'] == {}
        assert metrics['error_categories'] == {}

    @pytest.mark.fast
    def test_get_dashboard_metrics_with_heals(self):
        """Test get_dashboard_metrics with healing activity."""
        orchestrator = self._create_orchestrator()

        # Simulate some heals
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", True, 100.0)
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", False, 50.0)
        orchestrator.metrics.record_heal("disk-healer", "OUTPUT", True, 25.0)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("disk")

        metrics = orchestrator.get_dashboard_metrics()

        # 2 out of 3 successful = 66.67%
        assert 66.6 < metrics['success_rate'] < 66.7
        # Average of [100, 50, 25] = 58.33
        assert 58.3 < metrics['average_heal_time_ms'] < 58.4
        assert metrics['total_heals'] == 3
        assert metrics['successful_heals'] == 2
        assert metrics['failed_heals'] == 1

        # Healer utilization
        assert 'api-healer' in metrics['healer_utilization']
        api_util = metrics['healer_utilization']['api-healer']
        assert api_util['total_attempts'] == 2
        assert api_util['successful_heals'] == 1
        assert api_util['success_rate'] == 50.0

        # Error categories
        assert metrics['error_categories'] == {'api': 1, 'disk': 1}

    @pytest.mark.fast
    def test_get_dashboard_metrics_structure(self):
        """Test get_dashboard_metrics returns expected structure."""
        orchestrator = self._create_orchestrator()
        metrics = orchestrator.get_dashboard_metrics()

        # Verify all expected keys exist
        expected_keys = [
            'success_rate',
            'average_heal_time_ms',
            'total_heals',
            'successful_heals',
            'failed_heals',
            'healer_utilization',
            'error_categories',
            'heals_by_stage',
            'time_spent_healing',
            'preflight_issues',
            'user_escalations',
            'rollbacks',
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

        # Verify preflight_issues structure
        assert 'found' in metrics['preflight_issues']
        assert 'fixed' in metrics['preflight_issues']

    @pytest.mark.fast
    def test_format_dashboard_empty(self):
        """Test format_dashboard with no heals."""
        orchestrator = self._create_orchestrator()
        output = orchestrator.format_dashboard()

        assert "HEALING DASHBOARD" in output
        assert "Success Rate: 0.0%" in output
        assert "Total Heals: 0" in output

    @pytest.mark.fast
    def test_format_dashboard_with_heals(self):
        """Test format_dashboard with healing activity."""
        orchestrator = self._create_orchestrator()

        # Simulate heals
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", True, 100.0)
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 150.0)
        orchestrator.metrics.record_heal("disk-healer", "OUTPUT", False, 50.0)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("disk")

        output = orchestrator.format_dashboard()

        # Verify dashboard sections present
        assert "HEALING DASHBOARD" in output
        assert "Healer Utilization:" in output
        assert "api-healer:" in output
        assert "disk-healer:" in output
        assert "Error Categories:" in output
        assert "api: 2" in output
        assert "disk: 1" in output

    @pytest.mark.fast
    def test_format_dashboard_includes_stage_breakdown(self):
        """Test format_dashboard includes heals by stage."""
        orchestrator = self._create_orchestrator()

        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", True, 100.0)
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 150.0)

        output = orchestrator.format_dashboard()

        assert "Heals by Stage:" in output
        assert "DOWNLOAD: 1" in output
        assert "MATCH: 1" in output

    @pytest.mark.fast
    def test_format_dashboard_shows_preflight_issues(self):
        """Test format_dashboard shows preflight issues when present."""
        orchestrator = self._create_orchestrator()

        orchestrator.metrics.preflight_issues_found = 5
        orchestrator.metrics.preflight_issues_fixed = 3

        output = orchestrator.format_dashboard()

        assert "Preflight Issues: 3/5 fixed" in output

    @pytest.mark.fast
    def test_format_dashboard_shows_escalations(self):
        """Test format_dashboard shows user escalations when present."""
        orchestrator = self._create_orchestrator()

        orchestrator.metrics.user_escalations = 2

        output = orchestrator.format_dashboard()

        assert "User Escalations: 2" in output

    @pytest.mark.fast
    def test_format_dashboard_shows_rollbacks(self):
        """Test format_dashboard shows rollbacks when present."""
        orchestrator = self._create_orchestrator()

        orchestrator.metrics.rollbacks_performed = 1

        output = orchestrator.format_dashboard()

        assert "Rollbacks: 1" in output


class TestOrchestratorPrintReport:
    """Test print_report integration with dashboard (US-64-005)."""

    def _create_orchestrator(self) -> HealingOrchestrator:
        """Create orchestrator with mocked config to avoid real healer init."""
        config = MagicMock()
        config.healing = None
        strategy = HealingStrategy()
        # Pass NoopHealer to avoid initializing real healers
        orchestrator = HealingOrchestrator(
            config, "/tmp/project", strategy=strategy, healers=[NoopHealer]
        )
        return orchestrator

    @pytest.mark.fast
    def test_print_report_shows_dashboard_when_heals_exist(self, capsys):
        """Test that print_report shows dashboard when there are heals."""
        orchestrator = self._create_orchestrator()

        # Add some heals
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", True, 100.0)

        orchestrator.print_report()

        captured = capsys.readouterr()
        assert "HEALING DASHBOARD" in captured.out
        assert "Healer Utilization:" in captured.out

    @pytest.mark.fast
    def test_print_report_shows_summary_when_no_heals(self, capsys):
        """Test that print_report shows summary when there are no heals."""
        orchestrator = self._create_orchestrator()

        orchestrator.print_report()

        captured = capsys.readouterr()
        # Should show the summary format, not dashboard
        assert "Total heals: 0" in captured.out
        # Dashboard header should not appear
        assert "HEALING DASHBOARD" not in captured.out


class TestExportMetricsJson:
    """Test export_metrics_json() for healing metrics persistence (US-68-004)."""

    def _create_orchestrator(self, project_dir) -> HealingOrchestrator:
        """Create orchestrator with mocked config."""
        config = MagicMock()
        config.healing = None
        strategy = HealingStrategy()
        orchestrator = HealingOrchestrator(
            config, project_dir, strategy=strategy, healers=[NoopHealer]
        )
        return orchestrator

    @pytest.mark.fast
    def test_export_creates_valid_json_with_all_fields(self, tmp_path):
        """Test that export_metrics_json creates valid JSON with all required fields."""
        import json

        orchestrator = self._create_orchestrator(tmp_path)

        # Populate some metrics
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", True, 120.0)
        orchestrator.metrics.record_heal("disk-healer", "OUTPUT", False, 80.0)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("disk")
        orchestrator.metrics.preflight_issues_found = 3
        orchestrator.metrics.rollbacks_performed = 1

        path = orchestrator.export_metrics_json()

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))

        # Verify all required fields from acceptance criteria
        assert "success_rate" in data
        assert "total_heals" in data
        assert "heals_by_stage" in data
        assert "heals_by_healer" in data
        assert "error_category_distribution" in data
        assert "average_heal_time_ms" in data
        assert "preflight_issues_found" in data
        assert "rollback_count" in data

        # Verify values
        assert data["total_heals"] == 2
        assert data["success_rate"] == 50.0
        assert data["heals_by_stage"] == {"DOWNLOAD": 1, "OUTPUT": 1}
        assert data["heals_by_healer"] == {"api-healer": 1, "disk-healer": 1}
        assert data["error_category_distribution"] == {"api": 1, "disk": 1}
        assert data["average_heal_time_ms"] == 100.0
        assert data["preflight_issues_found"] == 3
        assert data["rollback_count"] == 1
        assert "timestamp" in data

    @pytest.mark.fast
    def test_export_default_path(self, tmp_path):
        """Test that export writes to <project>/healing_metrics.json by default."""
        orchestrator = self._create_orchestrator(tmp_path)

        path = orchestrator.export_metrics_json()

        assert path == tmp_path / "healing_metrics.json"
        assert path.exists()

    @pytest.mark.fast
    def test_export_custom_path(self, tmp_path):
        """Test that export can write to a custom path."""
        orchestrator = self._create_orchestrator(tmp_path)
        custom_path = tmp_path / "custom" / "metrics.json"
        custom_path.parent.mkdir(parents=True)

        path = orchestrator.export_metrics_json(output_path=custom_path)

        assert path == custom_path
        assert path.exists()

    @pytest.mark.fast
    def test_export_empty_metrics(self, tmp_path):
        """Test export with no healing activity produces valid JSON."""
        import json

        orchestrator = self._create_orchestrator(tmp_path)

        path = orchestrator.export_metrics_json()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_heals"] == 0
        assert data["success_rate"] == 0.0
        assert data["heals_by_stage"] == {}
        assert data["heals_by_healer"] == {}
        assert data["error_category_distribution"] == {}
        assert data["average_heal_time_ms"] == 0.0
        assert data["preflight_issues_found"] == 0
        assert data["rollback_count"] == 0

    @pytest.mark.fast
    def test_metrics_accumulate_across_multiple_heal_attempts(self, tmp_path):
        """Test metrics accumulate correctly across multiple heal attempts."""
        import json

        orchestrator = self._create_orchestrator(tmp_path)

        # Simulate multiple heal attempts across different stages and healers
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", True, 100.0)
        orchestrator.metrics.record_heal("api-healer", "DOWNLOAD", False, 200.0)
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 150.0)
        orchestrator.metrics.record_heal("disk-healer", "OUTPUT", True, 50.0)
        orchestrator.metrics.record_heal("checkpoint-healer", "DOWNLOAD", True, 75.0)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("disk")
        orchestrator.metrics.preflight_issues_found = 5
        orchestrator.metrics.rollbacks_performed = 2

        path = orchestrator.export_metrics_json()
        data = json.loads(path.read_text(encoding="utf-8"))

        # Verify accumulation
        assert data["total_heals"] == 5
        # 4 successes out of 5 = 80%
        assert data["success_rate"] == 80.0
        # Heals by stage: DOWNLOAD=3, MATCH=1, OUTPUT=1
        assert data["heals_by_stage"]["DOWNLOAD"] == 3
        assert data["heals_by_stage"]["MATCH"] == 1
        assert data["heals_by_stage"]["OUTPUT"] == 1
        # Heals by healer: api-healer=3, disk-healer=1, checkpoint-healer=1
        assert data["heals_by_healer"]["api-healer"] == 3
        assert data["heals_by_healer"]["disk-healer"] == 1
        assert data["heals_by_healer"]["checkpoint-healer"] == 1
        # Error categories accumulated
        assert data["error_category_distribution"]["api"] == 2
        assert data["error_category_distribution"]["disk"] == 1
        # Average heal time: (100+200+150+50+75)/5 = 115.0
        assert data["average_heal_time_ms"] == 115.0
        assert data["preflight_issues_found"] == 5
        assert data["rollback_count"] == 2
