"""Tests for pipeline stage timing history and prediction (US-81-010)."""

import json
import pytest
from pathlib import Path

from src.pipeline_history import (
    append_stage_timing,
    estimate_duration,
    load_history,
    save_history,
    MAX_RUNS,
)


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Provide a temporary project directory."""
    return tmp_path


class TestLoadSaveHistory:
    """Tests for load/save round-trip and trimming."""

    def test_empty_history_when_no_file(self, project_dir: Path):
        assert load_history(project_dir) == {}

    def test_round_trip(self, project_dir: Path):
        history = {"MATCH": [{"duration": 5.0, "items_processed": 10, "throughput": 2.0}]}
        save_history(project_dir, history)
        loaded = load_history(project_dir)
        assert loaded == history

    def test_trim_to_max_runs(self, project_dir: Path):
        runs = [{"duration": float(i), "items_processed": i, "throughput": 1.0}
                for i in range(MAX_RUNS + 5)]
        history = {"CAPTION": runs}
        save_history(project_dir, history)
        loaded = load_history(project_dir)
        assert len(loaded["CAPTION"]) == MAX_RUNS
        # Should keep the most recent entries
        assert loaded["CAPTION"][-1]["duration"] == float(MAX_RUNS + 4)

    def test_corrupt_file_returns_empty(self, project_dir: Path):
        path = project_dir / "pipeline_history.json"
        path.write_text("NOT JSON", encoding="utf-8")
        assert load_history(project_dir) == {}


class TestAppendStageTiming:
    """Tests for appending individual stage records."""

    def test_append_creates_file(self, project_dir: Path):
        append_stage_timing(project_dir, "ANALYZE", 3.5, 1, 0.29)
        history = load_history(project_dir)
        assert "ANALYZE" in history
        assert len(history["ANALYZE"]) == 1
        assert history["ANALYZE"][0]["duration"] == 3.5

    def test_append_multiple_stages(self, project_dir: Path):
        append_stage_timing(project_dir, "MATCH", 30.0, 10, 0.333)
        append_stage_timing(project_dir, "CAPTION", 12.0, 50, 4.167)
        history = load_history(project_dir)
        assert len(history["MATCH"]) == 1
        assert len(history["CAPTION"]) == 1


class TestEstimateDuration:
    """Tests for duration prediction from historical data."""

    def test_no_history_returns_none(self, project_dir: Path):
        assert estimate_duration(project_dir, "MATCH", 10) is None

    def test_estimate_scales_by_items(self, project_dir: Path):
        """
        Acceptance criterion: 3 historical runs averaging 10 items in 30s
        (throughput = 10/30 ≈ 0.3333 items/sec).
        A new run with 20 items should estimate ~60 seconds.
        """
        for _ in range(3):
            append_stage_timing(
                project_dir, "MATCH",
                duration=30.0, items_processed=10,
                throughput=10.0 / 30.0,  # 0.3333 items/sec
            )
        est = estimate_duration(project_dir, "MATCH", items_count=20)
        assert est is not None
        # 20 items / 0.3333 items/sec ≈ 60.0 seconds
        assert abs(est - 60.0) < 1.0, f"Expected ~60s, got {est}"

    def test_estimate_uses_last_3_runs(self, project_dir: Path):
        """Only the 3 most recent runs should affect the estimate."""
        # Old runs with slow throughput (should be ignored)
        for _ in range(5):
            append_stage_timing(project_dir, "CAPTION", 100.0, 10, 0.1)
        # Recent 3 runs with fast throughput: 10 items in 10s = 1.0 items/sec
        for _ in range(3):
            append_stage_timing(project_dir, "CAPTION", 10.0, 10, 1.0)

        est = estimate_duration(project_dir, "CAPTION", items_count=20)
        assert est is not None
        # 20 items / 1.0 items/sec = 20 seconds
        assert abs(est - 20.0) < 1.0

    def test_estimate_falls_back_to_avg_duration(self, project_dir: Path):
        """When items_count is 0, fall back to average duration."""
        append_stage_timing(project_dir, "ANALYZE", 5.0, 0, 0.0)
        append_stage_timing(project_dir, "ANALYZE", 7.0, 0, 0.0)
        append_stage_timing(project_dir, "ANALYZE", 6.0, 0, 0.0)
        est = estimate_duration(project_dir, "ANALYZE", items_count=0)
        assert est is not None
        assert abs(est - 6.0) < 0.5  # avg of 5, 7, 6 = 6.0

    def test_graceful_with_no_matching_stage(self, project_dir: Path):
        """Asking about a stage not in history returns None."""
        append_stage_timing(project_dir, "MATCH", 10.0, 5, 0.5)
        assert estimate_duration(project_dir, "OUTPUT", items_count=10) is None
