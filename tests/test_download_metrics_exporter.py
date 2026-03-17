"""Unit tests for enhanced download metrics exporter (US-114-003).

Tests per-keyword, per-tier, per-region metrics tracking and CSV export.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.downloader.metrics_exporter import (
    DownloadMetricsConfig,
    DownloadMetrics,
    DownloadMetricsExporter,
    create_metrics_exporter,
)


class TestDownloadMetricsConfig:
    """Tests for DownloadMetricsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DownloadMetricsConfig()
        assert config.enabled is True
        assert config.export_interval_seconds == 60.0
        assert config.output_dir == "output/download_metrics"
        assert config.export_prometheus is True
        assert config.export_json is True
        assert config.export_csv is False
        assert config.metrics_export_format == "json"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DownloadMetricsConfig(
            enabled=False,
            export_interval_seconds=30.0,
            output_dir="custom/path",
            export_prometheus=False,
            export_json=False,
            export_csv=True,
            metrics_export_format="csv",
        )
        assert config.enabled is False
        assert config.export_interval_seconds == 30.0
        assert config.output_dir == "custom/path"
        assert config.export_prometheus is False
        assert config.export_json is False
        assert config.export_csv is True
        assert config.metrics_export_format == "csv"


class TestDownloadMetrics:
    """Tests for DownloadMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = DownloadMetrics()
        assert metrics.download_total == 0
        assert metrics.download_successful == 0
        assert metrics.download_failed == 0
        assert metrics.keyword_success == {}
        assert metrics.keyword_failure == {}
        assert metrics.tier_success == {}
        assert metrics.tier_failure == {}
        assert metrics.region_success == {}
        assert metrics.region_failure == {}
        assert metrics.rate_limit_events_log == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DownloadMetrics()
        metrics.download_total = 10
        metrics.download_successful = 8
        metrics.download_failed = 2

        d = metrics.to_dict()
        assert d["download_total"] == 10
        assert d["download_successful"] == 8
        assert d["download_failed"] == 2
        assert "keyword_success" in d
        assert "tier_success" in d
        assert "region_success" in d

    def test_reset(self):
        """Test reset clears all counters."""
        metrics = DownloadMetrics()
        metrics.download_total = 100
        metrics.download_successful = 90
        metrics.keyword_success["test_keyword"] = 50
        metrics.tier_success["short"] = 30
        metrics.region_success["us"] = 20

        metrics.reset()

        assert metrics.download_total == 0
        assert metrics.download_successful == 0
        assert metrics.keyword_success == {}
        assert metrics.tier_success == {}
        assert metrics.region_success == {}


class TestDownloadMetricsExporter:
    """Tests for DownloadMetricsExporter class."""

    def test_initialization(self):
        """Test exporter initialization with defaults."""
        exporter = DownloadMetricsExporter()
        assert exporter.config.enabled is True
        assert exporter._metrics is not None

    def test_record_download_with_keyword(self):
        """Test recording download with keyword tracking."""
        exporter = DownloadMetricsExporter()

        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
            keyword="sunset",
        )

        metrics = exporter.get_current_metrics()
        assert metrics["keyword_success"]["sunset"] == 1

        exporter.record_download_complete(
            duration=3.0,
            size_bytes=512000,
            success=False,
            keyword="ocean",
        )

        metrics = exporter.get_current_metrics()
        assert metrics["keyword_success"]["sunset"] == 1
        assert metrics["keyword_failure"]["ocean"] == 1

    def test_record_download_with_tier(self):
        """Test recording download with tier tracking."""
        exporter = DownloadMetricsExporter()

        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
            tier="short",
        )
        exporter.record_download_complete(
            duration=10.0,
            size_bytes=2048000,
            success=True,
            tier="medium",
        )
        exporter.record_download_complete(
            duration=3.0,
            size_bytes=512000,
            success=False,
            tier="short",
        )

        metrics = exporter.get_current_metrics()
        assert metrics["tier_success"]["short"] == 1
        assert metrics["tier_success"]["medium"] == 1
        assert metrics["tier_failure"]["short"] == 1

    def test_record_download_with_region(self):
        """Test recording download with region tracking."""
        exporter = DownloadMetricsExporter()

        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
            region="us",
        )
        exporter.record_download_complete(
            duration=10.0,
            size_bytes=2048000,
            success=True,
            region="de",
        )
        exporter.record_download_complete(
            duration=3.0,
            size_bytes=512000,
            success=False,
            region="us",
        )

        metrics = exporter.get_current_metrics()
        assert metrics["region_success"]["us"] == 1
        assert metrics["region_success"]["de"] == 1
        assert metrics["region_failure"]["us"] == 1

    def test_record_rate_limit_event(self):
        """Test recording rate limit events with time-series."""
        exporter = DownloadMetricsExporter()

        exporter.record_rate_limit_event(keyword="sunset", tier="short", region="us")
        exporter.record_rate_limit_event(keyword="ocean", tier="medium", region="de")

        timeseries = exporter.get_rate_limit_timeseries()
        assert len(timeseries) == 2
        assert timeseries[0]["keyword"] == "sunset"
        assert timeseries[0]["tier"] == "short"
        assert timeseries[0]["region"] == "us"
        assert "timestamp" in timeseries[0]

    def test_json_export(self):
        """Test JSON export includes per-keyword/tier/region data."""
        exporter = DownloadMetricsExporter()
        exporter.config.output_dir = tempfile.mkdtemp()

        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
            keyword="sunset",
            tier="short",
            region="us",
        )
        exporter.record_rate_limit_event(keyword="sunset", tier="short", region="us")

        files = exporter.export()
        assert "json" in files

        # Verify JSON content
        with open(files["json"]) as f:
            data = json.load(f)

        assert "downloads" in data
        assert data["downloads"]["keyword_success"]["sunset"] == 1
        assert data["downloads"]["tier_success"]["short"] == 1
        assert data["downloads"]["region_success"]["us"] == 1
        assert len(data["timeseries"]["rate_limit_events"]) == 1


class TestPrometheusExport:
    """Tests for Prometheus format export (US-143-009)."""

    def test_prometheus_export_basic_counters(self):
        """Test Prometheus export includes basic counter metrics."""
        exporter = DownloadMetricsExporter()
        exporter.config.export_prometheus = True
        exporter.config.export_json = False
        exporter.config.export_csv = False
        exporter.config.output_dir = tempfile.mkdtemp()

        # Record some downloads
        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
        )
        exporter.record_download_complete(
            duration=3.0,
            size_bytes=512000,
            success=False,
        )
        exporter.record_retry()
        exporter.record_retry()

        files = exporter.export()
        assert "prometheus" in files

        # Verify Prometheus format content
        with open(files["prometheus"]) as f:
            content = f.read()

        assert "# HELP download_total" in content
        assert "# TYPE download_total counter" in content
        assert "download_total 2" in content
        assert "download_successful 1" in content
        assert "download_failed 1" in content
        assert "download_errors 1" in content
        assert "download_retry_count 2" in content

    def test_prometheus_export_gauges(self):
        """Test Prometheus export includes gauge metrics."""
        exporter = DownloadMetricsExporter()
        exporter.config.export_prometheus = True
        exporter.config.export_json = False
        exporter.config.output_dir = tempfile.mkdtemp()

        exporter.record_download_start()
        exporter.record_download_start()
        exporter.record_queue_size(10)

        files = exporter.export()
        assert "prometheus" in files

        with open(files["prometheus"]) as f:
            content = f.read()

        assert "# HELP download_active" in content
        assert "# TYPE download_active gauge" in content
        assert "download_active 2" in content
        assert "# HELP download_queue_size" in content
        assert "# TYPE download_queue_size gauge" in content
        assert "download_queue_size 10" in content

    def test_prometheus_export_histogram(self):
        """Test Prometheus export includes histogram buckets (US-143-009)."""
        exporter = DownloadMetricsExporter()
        exporter.config.export_prometheus = True
        exporter.config.export_json = False
        exporter.config.output_dir = tempfile.mkdtemp()

        # Record downloads with various durations
        # 0.1s (bucket 0.1), 0.3s (bucket 0.5), 1.5s (bucket 2)
        # 3.0s (bucket 5), 8.0s (bucket 10), 45s (bucket 60)
        exporter.record_download_complete(duration=0.1, size_bytes=100000, success=True)
        exporter.record_download_complete(duration=0.3, size_bytes=100000, success=True)
        exporter.record_download_complete(duration=1.5, size_bytes=100000, success=True)
        exporter.record_download_complete(duration=3.0, size_bytes=100000, success=True)
        exporter.record_download_complete(duration=8.0, size_bytes=100000, success=True)
        exporter.record_download_complete(duration=45.0, size_bytes=100000, success=True)

        files = exporter.export()
        assert "prometheus" in files

        with open(files["prometheus"]) as f:
            content = f.read()

        # Check histogram format
        assert "# HELP download_duration_seconds Download duration histogram" in content
        assert "# TYPE download_duration_seconds histogram" in content
        assert 'download_duration_seconds_bucket{le="0.1"}' in content
        assert 'download_duration_seconds_bucket{le="0.5"}' in content
        assert 'download_duration_seconds_bucket{le="1"}' in content
        assert 'download_duration_seconds_bucket{le="2"}' in content
        assert 'download_duration_seconds_bucket{le="5"}' in content
        assert 'download_duration_seconds_bucket{le="10"}' in content
        assert 'download_duration_seconds_bucket{le="60"}' in content
        assert 'download_duration_seconds_bucket{le="+Inf"}' in content

        # Verify cumulative counts
        assert 'le="0.1"} 1' in content  # 1 <= 0.1
        assert 'le="0.5"} 2' in content  # 2 <= 0.5
        assert 'le="1"} 2' in content     # only 2 <= 1
        assert 'le="2"} 3' in content     # 3 <= 2
        assert 'le="5"} 4' in content     # 4 <= 5
        assert 'le="10"} 5' in content    # 5 <= 10
        assert 'le="60"} 6' in content    # 6 <= 60

        # sum and count
        assert "download_duration_seconds_sum 57.9" in content
        assert "download_duration_seconds_count 6" in content

    def test_generate_prometheus_format(self):
        """Test _generate_prometheus_format method (US-143-009)."""
        exporter = DownloadMetricsExporter()

        # Record some data
        exporter.record_download_complete(duration=5.0, size_bytes=1024000, success=True)
        exporter.record_queue_size(15)

        # Get the data and generate Prometheus format
        data = exporter._get_metrics_data()
        prom_output = exporter._generate_prometheus_format(data)

        assert "# HELP download_total" in prom_output
        assert "download_total 1" in prom_output
        assert "download_queue_size 15" in prom_output
        assert "# HELP download_duration_seconds Download duration histogram" in prom_output


class TestCsvExport:
    """Tests for CSV export functionality (US-114-003)."""

    def test_csv_export_creates_files(self):
        """Test CSV export creates separate CSV files."""
        exporter = DownloadMetricsExporter()
        exporter.config.export_csv = True
        exporter.config.metrics_export_format = "csv"
        exporter.config.output_dir = tempfile.mkdtemp()

        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
            keyword="sunset",
            tier="short",
            region="us",
        )
        exporter.record_download_complete(
            duration=3.0,
            size_bytes=512000,
            success=False,
            keyword="ocean",
            tier="medium",
            region="de",
        )
        exporter.record_rate_limit_event(keyword="sunset", tier="short", region="us")

        files = exporter.export()
        assert "csv" in files

        base = str(files["csv"]).replace(".csv", "")

        # Check main CSV
        assert os.path.exists(files["csv"])

        # Check keyword CSV
        keyword_csv = f"{base}_keywords.csv"
        assert os.path.exists(keyword_csv)
        with open(keyword_csv) as f:
            content = f.read()
            assert "sunset" in content
            assert "ocean" in content

        # Check tier CSV
        tier_csv = f"{base}_tier.csv"
        assert os.path.exists(tier_csv)
        with open(tier_csv) as f:
            content = f.read()
            assert "short" in content
            assert "medium" in content

        # Check region CSV
        region_csv = f"{base}_region.csv"
        assert os.path.exists(region_csv)
        with open(region_csv) as f:
            content = f.read()
            assert "us" in content
            assert "de" in content

    def test_csv_export_both_format(self):
        """Test CSV export with 'both' format exports both JSON and CSV."""
        exporter = DownloadMetricsExporter()
        exporter.config.export_json = True
        exporter.config.export_csv = True
        exporter.config.metrics_export_format = "both"
        exporter.config.output_dir = tempfile.mkdtemp()

        exporter.record_download_complete(
            duration=5.0,
            size_bytes=1024000,
            success=True,
            keyword="sunset",
        )

        files = exporter.export()
        assert "json" in files
        assert "csv" in files


class TestCreateMetricsExporter:
    """Tests for create_metrics_exporter factory function."""

    def test_with_config_dict(self):
        """Test creating exporter from config dict."""
        config = {
            "enabled": False,
            "export_csv": True,
            "metrics_export_format": "csv",
        }
        exporter = create_metrics_exporter(config)

        assert exporter.config.enabled is False
        assert exporter.config.export_csv is True
        assert exporter.config.metrics_export_format == "csv"

    def test_with_none_config(self):
        """Test creating exporter with None config uses defaults."""
        exporter = create_metrics_exporter(None)

        assert exporter.config.enabled is True
        assert exporter.config.metrics_export_format == "json"


class TestDownloadQualityMetrics:
    """Tests for US-144-012: Download quality metrics."""

    def test_tier_duration_tracking(self):
        """Test tracking average duration per tier."""
        exporter = DownloadMetricsExporter()

        # Record downloads with different tiers
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=True, tier="short")
        exporter.record_download_complete(duration=10.0, size_bytes=2000000, success=True, tier="short")
        exporter.record_download_complete(duration=15.0, size_bytes=3000000, success=True, tier="medium")
        exporter.record_download_complete(duration=20.0, size_bytes=4000000, success=True, tier="long")

        metrics = exporter.get_current_metrics()

        # Verify tier duration averages
        assert metrics["tier_duration_avg_seconds"]["short"] == 7.5  # (5 + 10) / 2
        assert metrics["tier_duration_avg_seconds"]["medium"] == 15.0
        assert metrics["tier_duration_avg_seconds"]["long"] == 20.0

    def test_success_rate_by_tier(self):
        """Test success rate calculation by tier."""
        exporter = DownloadMetricsExporter()

        # Short: 3 success, 1 failure = 75%
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=True, tier="short")
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=True, tier="short")
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=True, tier="short")
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=False, tier="short")

        # Medium: 1 success, 1 failure = 50%
        exporter.record_download_complete(duration=10.0, size_bytes=2000000, success=True, tier="medium")
        exporter.record_download_complete(duration=10.0, size_bytes=2000000, success=False, tier="medium")

        # Long: 2 success, 0 failure = 100%
        exporter.record_download_complete(duration=15.0, size_bytes=3000000, success=True, tier="long")
        exporter.record_download_complete(duration=15.0, size_bytes=3000000, success=True, tier="long")

        metrics = exporter.get_current_metrics()

        assert metrics["tier_success_rate_percent"]["short"] == 75.0
        assert metrics["tier_success_rate_percent"]["medium"] == 50.0
        assert metrics["tier_success_rate_percent"]["long"] == 100.0

    def test_session_bytes_tracking(self):
        """Test session-level bytes tracking (cumulative across exports)."""
        exporter = DownloadMetricsExporter()

        # Record downloads
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=True)
        exporter.record_download_complete(duration=5.0, size_bytes=2000000, success=True)
        exporter.record_download_complete(duration=5.0, size_bytes=3000000, success=True)

        # Session bytes should accumulate
        metrics = exporter.get_current_metrics()
        assert metrics["session_bytes_total"] == 6000000

        # After export, session bytes should persist (not reset)
        exporter.config.output_dir = tempfile.mkdtemp()
        exporter.export()

        metrics = exporter.get_current_metrics()
        # session_bytes_total is NOT reset after export
        assert metrics["session_bytes_total"] == 6000000

    def test_retry_effectiveness_tracking(self):
        """Test retry effectiveness metrics (US-144-012)."""
        exporter = DownloadMetricsExporter()

        # Record some retry events
        exporter.record_retry()
        exporter.record_retry()

        # Record successful retries
        exporter.record_retry_success(attempts=1)  # Succeeded after 1 retry
        exporter.record_retry_success(attempts=2)  # Succeeded after 2 retries
        exporter.record_retry_success(attempts=3)  # Succeeded after 3 retries

        # Record a final retry that didn't lead to success
        exporter.record_retry_failure()

        metrics = exporter.get_current_metrics()

        # 3 videos saved by retry out of 4 total = 75%
        assert metrics["videos_saved_by_retry"] == 3
        assert metrics["videos_failed_after_retries"] == 1
        assert metrics["retry_effectiveness_percent"] == 75.0

        # Average attempts before success: (1 + 2 + 3) / 3 = 2.0
        assert metrics["avg_attempts_before_success"] == 2.0

    def test_retry_effectiveness_with_no_retries(self):
        """Test retry effectiveness when no retries occurred."""
        exporter = DownloadMetricsExporter()

        # No retries recorded
        metrics = exporter.get_current_metrics()

        assert metrics["videos_saved_by_retry"] == 0
        assert metrics["retry_effectiveness_percent"] == 0.0
        assert metrics["avg_attempts_before_success"] == 0.0

    def test_json_export_includes_quality_metrics(self):
        """Test JSON export includes new quality metrics."""
        exporter = DownloadMetricsExporter()
        exporter.config.output_dir = tempfile.mkdtemp()

        # Record downloads with various tiers
        exporter.record_download_complete(duration=5.0, size_bytes=1000000, success=True, tier="short")
        exporter.record_download_complete(duration=10.0, size_bytes=2000000, success=False, tier="short")
        exporter.record_download_complete(duration=15.0, size_bytes=3000000, success=True, tier="medium")

        # Record retry success
        exporter.record_retry_success(attempts=1)

        # Get metrics BEFORE export (export resets)
        metrics = exporter.get_current_metrics()
        assert metrics["tier_duration_avg_seconds"]["short"] == 7.5  # (5 + 10) / 2
        assert metrics["tier_success_rate_percent"]["short"] == 50.0
        assert metrics["session_bytes_total"] == 6000000
        assert metrics["videos_saved_by_retry"] == 1
        assert metrics["videos_failed_after_retries"] == 0
        assert metrics["retry_effectiveness_percent"] == 100.0  # 1/1 = 100%

        # Now export and verify JSON includes metrics
        files = exporter.export()
        assert "json" in files

        with open(files["json"]) as f:
            data = json.load(f)

        downloads = data["downloads"]

        # Verify quality metrics are in export
        assert "tier_duration_avg_seconds" in downloads
        assert "tier_success_rate_percent" in downloads
        assert "session_bytes_total" in downloads
        assert "videos_saved_by_retry" in downloads
        assert "retry_effectiveness_percent" in downloads
        assert "avg_attempts_before_success" in downloads

    def test_all_tiers_tracked(self):
        """Test tracking for all standard tiers: short, medium, long, longer."""
        exporter = DownloadMetricsExporter()

        # short: < 1 min
        exporter.record_download_complete(duration=30.0, size_bytes=1000000, success=True, tier="short")
        # medium: 1-5 min
        exporter.record_download_complete(duration=180.0, size_bytes=5000000, success=True, tier="medium")
        # long: 5-20 min
        exporter.record_download_complete(duration=600.0, size_bytes=20000000, success=True, tier="long")
        # longer: > 20 min
        exporter.record_download_complete(duration=1500.0, size_bytes=50000000, success=True, tier="longer")

        metrics = exporter.get_current_metrics()

        # All tiers should have duration averages
        assert "short" in metrics["tier_duration_avg_seconds"]
        assert "medium" in metrics["tier_duration_avg_seconds"]
        assert "long" in metrics["tier_duration_avg_seconds"]
        assert "longer" in metrics["tier_duration_avg_seconds"]

        # All should have 100% success
        assert metrics["tier_success_rate_percent"]["short"] == 100.0
        assert metrics["tier_success_rate_percent"]["medium"] == 100.0
        assert metrics["tier_success_rate_percent"]["long"] == 100.0
        assert metrics["tier_success_rate_percent"]["longer"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
