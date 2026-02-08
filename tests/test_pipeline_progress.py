"""Tests for ProgressReporter (US-81-004).

Verifies ETA computation, throttled writes, stage lifecycle,
and progress.json output.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline_progress import ProgressReporter, _MIN_WRITE_INTERVAL


@pytest.fixture
def tmp_project(tmp_path):
    """Return a temporary project directory."""
    return tmp_path / "project"


class TestProgressReporterETA:
    """Verify ETA estimation based on throughput."""

    def test_eta_basic_computation(self, tmp_project):
        """ETA = remaining_items / (completed / elapsed)."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=100)

        # Simulate 50 items completed in 10 seconds
        reporter._current.items_completed = 50
        reporter._current.start_time = time.time() - 10.0

        snap = reporter.get_snapshot()
        # throughput = 50/10 = 5 items/s  →  remaining = 50  →  ETA = 10s
        assert snap["estimated_remaining_seconds"] is not None
        assert abs(snap["estimated_remaining_seconds"] - 10.0) < 1.0

    def test_eta_none_when_no_items_completed(self, tmp_project):
        """ETA should be None when no items have been processed yet."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=100)

        snap = reporter.get_snapshot()
        assert snap["estimated_remaining_seconds"] is None

    def test_eta_zero_when_all_done(self, tmp_project):
        """ETA should be 0 when all items are completed."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        reporter._current.items_completed = 10
        reporter._current.start_time = time.time() - 5.0

        snap = reporter.get_snapshot()
        assert snap["estimated_remaining_seconds"] == 0.0

    def test_eta_none_when_total_zero(self, tmp_project):
        """ETA should be None when total is 0 (unknown batch size)."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=0)

        reporter._current.items_completed = 5
        reporter._current.start_time = time.time() - 5.0

        snap = reporter.get_snapshot()
        assert snap["estimated_remaining_seconds"] is None


class TestProgressReporterSnapshot:
    """Verify snapshot dict contents."""

    def test_snapshot_fields_present(self, tmp_project):
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("DOWNLOAD_SEGMENTS", total_items=50)
        reporter.update(completed=10, failed=2)

        snap = reporter.get_snapshot()
        assert snap["current_stage"] == "DOWNLOAD_SEGMENTS"
        assert snap["items_total"] == 50
        assert snap["items_completed"] == 10
        assert snap["items_failed"] == 2
        assert snap["elapsed_seconds"] >= 0
        assert "pipeline_elapsed_seconds" in snap
        assert "completed_stages" in snap

    def test_snapshot_after_finish_stage(self, tmp_project):
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("ANALYZE")
        reporter.finish_stage()

        snap = reporter.get_snapshot()
        assert snap["current_stage"] is None
        assert "ANALYZE" in snap["completed_stages"]

    def test_update_total_mid_stage(self, tmp_project):
        """Verify total can be updated mid-stage."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("CAPTION", total_items=0)
        reporter.update(total=200)

        snap = reporter.get_snapshot()
        assert snap["items_total"] == 200


class TestProgressReporterFileIO:
    """Verify progress.json writing."""

    def test_writes_progress_json_on_start(self, tmp_project):
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        assert reporter.progress_path.exists()
        data = json.loads(reporter.progress_path.read_text(encoding="utf-8"))
        assert data["current_stage"] == "TEST"
        assert data["items_total"] == 10

    def test_throttled_writes(self, tmp_project):
        """Non-forced writes respect the minimum interval."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=100)

        # First update writes (force=True from start_stage)
        mtime1 = reporter.progress_path.stat().st_mtime

        # Rapid updates should be throttled
        reporter.update(completed=1)
        reporter.update(completed=1)
        reporter.update(completed=1)

        # File should still have old mtime (throttled)
        mtime2 = reporter.progress_path.stat().st_mtime
        # Within the throttle window, the file shouldn't have been rewritten
        # (unless the test is extremely slow)
        assert mtime2 == mtime1 or (time.time() - mtime1) >= _MIN_WRITE_INTERVAL

    def test_finish_forces_write(self, tmp_project):
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)
        reporter.update(completed=5)
        reporter.finish_stage()

        data = json.loads(reporter.progress_path.read_text(encoding="utf-8"))
        assert data["current_stage"] is None
        assert "TEST" in data["completed_stages"]

    def test_write_creates_directory(self, tmp_project):
        """progress.json is created even if project dir doesn't exist yet."""
        deep_dir = tmp_project / "sub" / "deep"
        reporter = ProgressReporter(deep_dir)
        reporter.start_stage("TEST")

        assert (deep_dir / "progress.json").exists()


class TestProgressReporterPipeline:
    """Verify pipeline lifecycle."""

    def test_full_pipeline_lifecycle(self, tmp_project):
        reporter = ProgressReporter(tmp_project)

        reporter.start_stage("ANALYZE", total_items=1)
        reporter.update(completed=1)
        reporter.finish_stage()

        reporter.start_stage("DOWNLOAD_SEGMENTS", total_items=50)
        reporter.update(completed=25)

        snap = reporter.get_snapshot()
        assert snap["current_stage"] == "DOWNLOAD_SEGMENTS"
        assert snap["items_completed"] == 25
        assert "ANALYZE" in snap["completed_stages"]

        reporter.update(completed=25)
        reporter.finish_stage()

        reporter.finish_pipeline()

        snap = reporter.get_snapshot()
        assert snap["current_stage"] is None
        assert "ANALYZE" in snap["completed_stages"]
        assert "DOWNLOAD_SEGMENTS" in snap["completed_stages"]

    def test_update_without_start_is_noop(self, tmp_project):
        """Calling update without start_stage shouldn't crash."""
        reporter = ProgressReporter(tmp_project)
        reporter.update(completed=5)  # Should not raise
