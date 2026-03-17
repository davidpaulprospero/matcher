"""Pipeline metrics export functionality.

Supports export to JSON and Prometheus formats.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .stages import StageMetrics


class MetricsExporter:
    """Export pipeline metrics to various formats."""

    def __init__(self, config: Optional['PipelineExportConfig'] = None):
        """Initialize metrics exporter.

        Args:
            config: Export configuration. If None, uses defaults.
        """
        self.config = config or PipelineExportConfig()

    def export(
        self,
        stage_timings: Dict[str, float],
        stage_metrics: Dict[str, StageMetrics],
        pipeline_state: Optional[Dict[str, Any]] = None,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        config_summary: Optional[Dict[str, Any]] = None,
        circuit_breaker_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Export metrics to configured formats.

        Args:
            stage_timings: Dict of stage name to elapsed time in seconds
            stage_metrics: Dict of stage name to StageMetrics
            pipeline_state: Optional pipeline state dict for context
            checkpoint_data: Optional checkpoint data for context
            config_summary: Optional config summary dict
            circuit_breaker_metrics: Optional circuit breaker health metrics

        Returns:
            Dict mapping format name to output file path
        """
        exported_files: Dict[str, Path] = {}

        # Build the export data
        export_data = self._build_export_data(
            stage_timings, stage_metrics, pipeline_state, checkpoint_data, config_summary,
            circuit_breaker_metrics
        )

        # Export to each configured format
        if self.config.export_json:
            output_path = self._get_output_path("json")
            self._export_json(export_data, output_path)
            exported_files["json"] = output_path

        if self.config.export_prometheus:
            output_path = self._get_output_path("prometheus")
            self._export_prometheus(export_data, output_path)
            exported_files["prometheus"] = output_path

        return exported_files

    def _build_export_data(
        self,
        stage_timings: Dict[str, float],
        stage_metrics: Dict[str, StageMetrics],
        pipeline_state: Optional[Dict[str, Any]],
        checkpoint_data: Optional[Dict[str, Any]],
        config_summary: Optional[Dict[str, Any]],
        circuit_breaker_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the complete export data structure."""
        # Calculate aggregate metrics
        total_time = sum(stage_timings.values())
        total_items_processed = sum(m.items_processed for m in stage_metrics.values())
        total_items_failed = sum(m.items_failed for m in stage_metrics.values())

        # Build stage-level data
        stages_data = []
        for stage_name in stage_timings.keys():
            metrics = stage_metrics.get(stage_name)
            stages_data.append({
                "name": stage_name,
                "timing_seconds": stage_timings.get(stage_name, 0.0),
                "items_processed": metrics.items_processed if metrics else 0,
                "items_failed": metrics.items_failed if metrics else 0,
                "duration_seconds": metrics.duration_seconds if metrics else 0.0,
                "failed": metrics.failed if metrics else False,
                "error_categories": metrics.error_categories if metrics else {},
                "items_per_second": metrics.items_per_second if metrics else 0.0,
                "retry_attempts": metrics.retry_attempts if metrics else 0,
                "extra_metrics": metrics.extra_metrics if metrics else {},  # US-90-009: Stage-specific metrics
            })

        # Calculate error rates
        error_rates = {}
        for stage_name, metrics in stage_metrics.items():
            total = metrics.items_processed + metrics.items_failed
            if total > 0:
                error_rates[stage_name] = metrics.items_failed / total
            else:
                error_rates[stage_name] = 0.0

        # Build export structure
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline": {
                "total_duration_seconds": total_time,
                "total_items_processed": total_items_processed,
                "total_items_failed": total_items_failed,
                "overall_error_rate": (
                    total_items_failed / (total_items_processed + total_items_failed)
                    if (total_items_processed + total_items_failed) > 0
                    else 0.0
                ),
            },
            "stages": stages_data,
            "timings": stage_timings,
            "error_rates": error_rates,
        }

        # Add optional context if provided
        if pipeline_state:
            export_data["pipeline_state"] = pipeline_state
        if checkpoint_data:
            export_data["checkpoint"] = checkpoint_data
        if config_summary:
            export_data["config"] = config_summary
        if circuit_breaker_metrics:
            export_data["circuit_breaker"] = circuit_breaker_metrics

        return export_data

    def _get_output_path(self, format: str) -> Path:
        """Get output path for the given format."""
        base_dir = Path(self.config.output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = f"pipeline_metrics.{format}"
        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_metrics_{timestamp}.{format}"

        return base_dir / filename

    def _export_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export metrics to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _export_prometheus(self, data: Dict[str, Any], output_path: Path) -> None:
        """Export metrics to Prometheus text format."""
        lines = [
            "# HELP pipeline_total_duration_seconds Total pipeline execution time in seconds",
            "# TYPE pipeline_total_duration_seconds gauge",
            f'pipeline_total_duration_seconds {data["pipeline"]["total_duration_seconds"]}',
            "",
            "# HELP pipeline_items_processed_total Total items processed by pipeline",
            "# TYPE pipeline_items_processed_total counter",
            f'pipeline_items_processed_total {data["pipeline"]["total_items_processed"]}',
            "",
            "# HELP pipeline_items_failed_total Total items failed in pipeline",
            "# TYPE pipeline_items_failed_total counter",
            f'pipeline_items_failed_total {data["pipeline"]["total_items_failed"]}',
            "",
            "# HELP pipeline_error_rate Overall pipeline error rate",
            "# TYPE pipeline_error_rate gauge",
            f'pipeline_error_rate {data["pipeline"]["overall_error_rate"]}',
            "",
        ]

        # Stage-level metrics
        for stage in data.get("stages", []):
            stage_name = stage["name"].replace("-", "_").lower()
            lines.extend([
                f"# HELP pipeline_stage_duration_seconds Duration of {stage['name']} stage",
                f"# TYPE pipeline_stage_duration_seconds gauge",
                f'pipeline_stage_duration_seconds{{stage="{stage["name"]}"}} {stage["timing_seconds"]}',
                "",
                f"# HELP pipeline_stage_items_processed Items processed by {stage['name']} stage",
                f"# TYPE pipeline_stage_items_processed counter",
                f'pipeline_stage_items_processed{{stage="{stage["name"]}"}} {stage["items_processed"]}',
                "",
                f"# HELP pipeline_stage_items_failed Items failed by {stage['name']} stage",
                f"# TYPE pipeline_stage_items_failed counter",
                f'pipeline_stage_items_failed{{stage="{stage["name"]}"}} {stage["items_failed"]}',
                "",
                f"# HELP pipeline_stage_throughput Throughput of {stage['name']} stage (items/sec)",
                f"# TYPE pipeline_stage_throughput gauge",
                f'pipeline_stage_throughput{{stage="{stage["name"]}"}} {stage["items_per_second"]}',
                "",
            ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


class PipelineExportConfig:
    """Configuration for pipeline metrics export."""

    def __init__(
        self,
        output_dir: str = "output/metrics",
        export_json: bool = True,
        export_prometheus: bool = False,
        include_timestamp: bool = True,
    ):
        """Initialize export configuration.

        Args:
            output_dir: Directory to write exported metrics
            export_json: Whether to export to JSON format
            export_prometheus: Whether to export to Prometheus format
            include_timestamp: Whether to include timestamp in filename
        """
        self.output_dir = output_dir
        self.export_json = export_json
        self.export_prometheus = export_prometheus
        self.include_timestamp = include_timestamp

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineExportConfig':
        """Create config from dictionary."""
        return cls(
            output_dir=data.get("output_dir", "output/metrics"),
            export_json=data.get("export_json", True),
            export_prometheus=data.get("export_prometheus", False),
            include_timestamp=data.get("include_timestamp", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "output_dir": self.output_dir,
            "export_json": self.export_json,
            "export_prometheus": self.export_prometheus,
            "include_timestamp": self.include_timestamp,
        }
