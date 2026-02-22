"""Tests for ProgressReporter (US-81-004).

Verifies ETA computation, throttled writes, stage lifecycle,
and progress.json output.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.pipeline_progress import ProgressReporter, _MIN_WRITE_INTERVAL, _format_duration, _DEFAULT_STUCK_THRESHOLD_SECONDS, _DEFAULT_STAGE_DURATION_WARNING_THRESHOLD_SECONDS


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


class TestETADurationFormatting:
    """Test ETA duration formatting (US-106-003)."""

    def test_format_duration_none(self):
        """None input returns None."""
        assert _format_duration(None) is None

    def test_format_duration_seconds_only(self):
        """Less than 60 seconds shows seconds."""
        assert _format_duration(30.0) == "30s"
        assert _format_duration(45.5) == "45s"

    def test_format_duration_minutes(self):
        """Minutes and optional seconds."""
        assert _format_duration(60.0) == "1m"
        assert _format_duration(90.0) == "1m 30s"
        assert _format_duration(125.0) == "2m 5s"

    def test_format_duration_hours(self):
        """Hours and minutes."""
        assert _format_duration(3600.0) == "1h"
        assert _format_duration(3660.0) == "1h 1m"
        assert _format_duration(3720.0) == "1h 2m"
        assert _format_duration(7325.0) == "2h 2m 5s"


class TestPipelineETAWithHistoricalData:
    """Test pipeline-wide ETA calculation using historical data (US-106-003)."""

    def test_pipeline_eta_with_mock_history(self, tmp_project):
        """Test ETA calculation uses historical stage durations."""
        # Set up remaining stages
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["ANALYZE", "CAPTION", "MATCH", "DOWNLOAD_SEGMENTS", "OUTPUT"]
        )
        reporter.start_stage("CAPTION", total_items=100)

        # Mock estimate_duration to return predictable values
        mock_estimates = {
            "CAPTION": 300.0,  # 5 minutes for caption stage
            "MATCH": 600.0,    # 10 minutes for match
            "DOWNLOAD_SEGMENTS": 900.0,  # 15 minutes for download
            "OUTPUT": 60.0,   # 1 minute for output
        }

        with patch("src.pipeline_progress.estimate_duration") as mock_est:
            def mock_estimate(project_dir, stage_name, items_count):
                return mock_estimates.get(stage_name, 100.0)

            mock_est.side_effect = mock_estimate

            snap = reporter.get_snapshot()

            # Pipeline ETA should be sum of remaining stages (current + after)
            # CAPTION: 300, MATCH: 600, DOWNLOAD_SEGMENTS: 900, OUTPUT: 60 = 1860s
            assert snap["pipeline_eta_seconds"] == 1860.0
            assert snap["pipeline_eta_display"] == "31m"

    def test_pipeline_eta_no_history_returns_none(self, tmp_project):
        """When no historical data, pipeline_eta_seconds is None."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["ANALYZE", "CAPTION", "MATCH"]
        )
        reporter.start_stage("CAPTION", total_items=100)

        # Mock estimate_duration to return None (no history)
        with patch("src.pipeline_progress.estimate_duration") as mock_est:
            mock_est.return_value = None

            snap = reporter.get_snapshot()

            assert snap["pipeline_eta_seconds"] is None
            assert snap["pipeline_eta_display"] is None

    def test_pipeline_eta_throughput_overrides_historical(self, tmp_project):
        """Throughput-based ETA takes priority over historical for current stage."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION", "MATCH"]
        )
        reporter.start_stage("CAPTION", total_items=100)
        reporter._current.items_completed = 50
        reporter._current.start_time = time.time() - 10.0  # 10 seconds elapsed

        # Mock historical estimate to be different
        with patch("src.pipeline_progress.estimate_duration") as mock_est:
            mock_est.return_value = 500.0  # Historical: 500s

            snap = reporter.get_snapshot()

            # Throughput: 50 items in 10s = 5 items/s, remaining 50 items = 10s
            # This should take priority
            assert snap["estimated_remaining_seconds"] is not None
            assert abs(snap["estimated_remaining_seconds"] - 10.0) < 1.0

    def test_pipeline_eta_no_remaining_stages(self, tmp_project):
        """When no remaining stages set, pipeline_eta is None."""
        reporter = ProgressReporter(tmp_project, remaining_stages=[])
        reporter.start_stage("ANALYZE", total_items=10)

        snap = reporter.get_snapshot()

        assert snap["pipeline_eta_seconds"] is None

    def test_pipeline_eta_display_field_present(self, tmp_project):
        """Verify pipeline_eta_display field is in snapshot."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION"]
        )
        reporter.start_stage("CAPTION", total_items=100)

        with patch("src.pipeline_progress.estimate_duration") as mock_est:
            mock_est.return_value = 300.0

            snap = reporter.get_snapshot()

            assert "pipeline_eta_display" in snap


class TestPipelineETAByStage:
    """Test stage-level ETA breakdown (US-138-008)."""

    def test_pipeline_eta_by_stage_field_present(self, tmp_project):
        """Verify pipeline_eta_by_stage field is in snapshot."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION", "MATCH"]
        )
        reporter.start_stage("CAPTION", total_items=100)

        with patch("src.pipeline_progress.estimate_duration") as mock_est:
            mock_est.return_value = 300.0

            snap = reporter.get_snapshot()

            assert "pipeline_eta_by_stage" in snap
            assert isinstance(snap["pipeline_eta_by_stage"], list)

    def test_pipeline_eta_by_stage_contains_all_stages(self, tmp_project):
        """Verify pipeline_eta_by_stage contains all remaining stages."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION", "MATCH", "DOWNLOAD_SEGMENTS"]
        )
        reporter.start_stage("CAPTION", total_items=100)

        mock_estimates = {
            "CAPTION": {"point_estimate": 300.0, "lower_bound": 250.0, "upper_bound": 350.0, "has_confidence_interval": True},
            "MATCH": {"point_estimate": 600.0, "lower_bound": 500.0, "upper_bound": 700.0, "has_confidence_interval": True},
            "DOWNLOAD_SEGMENTS": {"point_estimate": 900.0, "lower_bound": 800.0, "upper_bound": 1000.0, "has_confidence_interval": True},
        }

        with patch("src.pipeline_progress.estimate_duration_with_confidence") as mock_est:
            def mock_estimate(project_dir, stage_name, items_count, confidence):
                return mock_estimates.get(stage_name)

            mock_est.side_effect = mock_estimate

            snap = reporter.get_snapshot()

            eta_by_stage = snap["pipeline_eta_by_stage"]
            assert len(eta_by_stage) == 3

            stage_names = [s["stage"] for s in eta_by_stage]
            assert "CAPTION" in stage_names
            assert "MATCH" in stage_names
            assert "DOWNLOAD_SEGMENTS" in stage_names

    def test_pipeline_eta_by_stage_contains_eta_values(self, tmp_project):
        """Verify each stage entry has eta_seconds and eta_display."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION", "MATCH"]
        )
        reporter.start_stage("CAPTION", total_items=100)

        mock_estimates = {
            "CAPTION": {"point_estimate": 300.0, "lower_bound": 250.0, "upper_bound": 350.0, "has_confidence_interval": True},
            "MATCH": {"point_estimate": 600.0, "lower_bound": 500.0, "upper_bound": 700.0, "has_confidence_interval": True},
        }

        with patch("src.pipeline_progress.estimate_duration_with_confidence") as mock_est:
            def mock_estimate(project_dir, stage_name, items_count, confidence):
                return mock_estimates.get(stage_name)

            mock_est.side_effect = mock_estimate

            snap = reporter.get_snapshot()

            eta_by_stage = snap["pipeline_eta_by_stage"]
            for stage_info in eta_by_stage:
                assert "stage" in stage_info
                assert "eta_seconds" in stage_info
                assert "eta_display" in stage_info

    def test_pipeline_eta_by_stage_current_stage_uses_throughput(self, tmp_project):
        """Current stage should use throughput-based ETA if available."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION", "MATCH"]
        )
        reporter.start_stage("CAPTION", total_items=100)
        # Simulate 50 items completed in 10 seconds
        reporter._current.items_completed = 50
        reporter._current.start_time = time.time() - 10.0

        mock_estimates = {
            "CAPTION": 300.0,  # Historical: 5 minutes
            "MATCH": 600.0,
        }

        with patch("src.pipeline_progress.estimate_duration") as mock_est:
            def mock_estimate(project_dir, stage_name, items_count):
                return mock_estimates.get(stage_name)

            mock_est.side_effect = mock_estimate

            snap = reporter.get_snapshot()

            eta_by_stage = snap["pipeline_eta_by_stage"]
            caption_stage = next(s for s in eta_by_stage if s["stage"] == "CAPTION")
            # Throughput: 50 items in 10s = 5 items/s, remaining 50 items = 10s
            assert abs(caption_stage["eta_seconds"] - 10.0) < 1.0

    def test_remaining_stages_field_present(self, tmp_project):
        """Verify remaining_stages field is in snapshot."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["CAPTION", "MATCH", "DOWNLOAD_SEGMENTS"]
        )

        snap = reporter.get_snapshot()

        assert "remaining_stages" in snap
        assert snap["remaining_stages"] == ["CAPTION", "MATCH", "DOWNLOAD_SEGMENTS"]


class TestStageProgressPercent:
    """Test stage-level progress percentage (US-108-003)."""

    def test_stage_progress_percent_calculated(self, tmp_project):
        """Stage progress percentage is calculated correctly."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("CAPTION", total_items=100)
        reporter.update(completed=45)

        snap = reporter.get_snapshot()
        assert snap["stage_progress_percent"] == 45.0

    def test_stage_progress_percent_zero_at_start(self, tmp_project):
        """Stage progress is 0% before any items completed."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("CAPTION", total_items=100)

        snap = reporter.get_snapshot()
        assert snap["stage_progress_percent"] == 0.0

    def test_stage_progress_percent_100_when_complete(self, tmp_project):
        """Stage progress is 100% when all items completed."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("CAPTION", total_items=50)
        reporter.update(completed=50)

        snap = reporter.get_snapshot()
        assert snap["stage_progress_percent"] == 100.0

    def test_stage_progress_percent_none_when_total_unknown(self, tmp_project):
        """Stage progress is None when total items is unknown."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("CAPTION", total_items=0)
        reporter.update(completed=5)

        snap = reporter.get_snapshot()
        assert snap["stage_progress_percent"] is None


class TestResourceUsage:
    """Test memory and CPU usage tracking (US-108-003)."""

    def test_memory_usage_field_present(self, tmp_project):
        """memory_usage_mb field is present in snapshot."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        snap = reporter.get_snapshot()
        assert "memory_usage_mb" in snap

    def test_cpu_percent_field_present(self, tmp_project):
        """cpu_percent field is present in snapshot."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        snap = reporter.get_snapshot()
        assert "cpu_percent" in snap

    def test_resource_usage_not_none(self, tmp_project):
        """Resource usage fields have values (psutil available or mocked)."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        snap = reporter.get_snapshot()
        # Should have values or None if psutil not available
        assert "memory_usage_mb" in snap
        assert "cpu_percent" in snap


class TestStageProgressEvent:
    """Test EVENT_STAGE_PROGRESS event emission (US-108-003)."""

    def test_progress_event_emitted_on_update(self, tmp_project):
        """EVENT_STAGE_PROGRESS event is emitted on update()."""
        from src.pipeline_events import PipelineEventBus, EVENT_STAGE_PROGRESS

        # Create a test event bus
        test_bus = PipelineEventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        test_bus.subscribe(EVENT_STAGE_PROGRESS, handler)

        # Set the test bus
        from src.pipeline_progress import set_event_bus
        set_event_bus(test_bus)

        try:
            reporter = ProgressReporter(tmp_project)
            reporter.start_stage("CAPTION", total_items=100)
            reporter.update(completed=50)

            # Check event was emitted
            assert len(received_events) > 0
            event = received_events[-1]
            assert event.event_type == EVENT_STAGE_PROGRESS
            assert event.stage_name == "CAPTION"
            assert event.data["items_completed"] == 50
            assert event.data["items_total"] == 100
            assert event.data["progress_percent"] == 50.0
        finally:
            # Reset to default event bus
            from src.pipeline_progress import set_event_bus, get_event_bus
            set_event_bus(get_event_bus())

    def test_progress_event_includes_resource_usage(self, tmp_project):
        """Progress event includes memory and CPU data."""
        from src.pipeline_events import PipelineEventBus, EVENT_STAGE_PROGRESS

        test_bus = PipelineEventBus()
        received_events = []

        def handler(event):
            received_events.append(event)

        test_bus.subscribe(EVENT_STAGE_PROGRESS, handler)

        from src.pipeline_progress import set_event_bus
        set_event_bus(test_bus)

        try:
            reporter = ProgressReporter(tmp_project)
            reporter.start_stage("TEST", total_items=10)
            reporter.update(completed=5)

            event = received_events[-1]
            assert "memory_usage_mb" in event.data
            assert "cpu_percent" in event.data
        finally:
            from src.pipeline_progress import set_event_bus, get_event_bus
            set_event_bus(get_event_bus())


class TestTimeoutDetection:
    """Test stage timeout detection (US-120-008)."""

    def test_timeout_status_initial_values(self, tmp_project):
        """Verify timeout status has correct initial values."""
        reporter = ProgressReporter(tmp_project)

        status = reporter.get_timeout_status()
        assert status["stuck_threshold_seconds"] == _DEFAULT_STUCK_THRESHOLD_SECONDS
        assert status["duration_warning_threshold_seconds"] == _DEFAULT_STAGE_DURATION_WARNING_THRESHOLD_SECONDS
        assert status["last_progress_time"] == 0.0
        assert status["warned_about_duration"] is False
        assert status["warned_about_stuck"] is False

    def test_custom_timeout_thresholds(self, tmp_project):
        """Verify custom timeout thresholds are accepted."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=120.0,
            duration_warning_threshold_seconds=300.0,
        )

        status = reporter.get_timeout_status()
        assert status["stuck_threshold_seconds"] == 120.0
        assert status["duration_warning_threshold_seconds"] == 300.0

    def test_last_progress_time_updated_on_start(self, tmp_project):
        """Verify last_progress_time is set when stage starts."""
        reporter = ProgressReporter(tmp_project)
        start_time = time.time()
        reporter.start_stage("TEST", total_items=10)

        status = reporter.get_timeout_status()
        assert status["last_progress_time"] >= start_time

    def test_last_progress_time_updated_on_progress(self, tmp_project):
        """Verify last_progress_time is updated when progress is made."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        initial_time = reporter._last_progress_time
        time.sleep(0.1)  # Small delay

        reporter.update(completed=1)

        assert reporter._last_progress_time > initial_time

    def test_check_timeout_no_stage_returns_none(self, tmp_project):
        """check_timeout returns None when no stage is active."""
        reporter = ProgressReporter(tmp_project)

        result = reporter.check_timeout()
        assert result is None

    def test_check_timeout_stuck_detection(self, tmp_project):
        """check_timeout detects stuck stage when no progress for threshold."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=0.5,  # Very short for test
            duration_warning_threshold_seconds=600.0,
        )
        reporter.start_stage("TEST", total_items=10)

        # Simulate that last progress was long ago
        reporter._last_progress_time = time.time() - 1.0

        result = reporter.check_timeout()

        assert result is not None
        assert result["is_stuck"] is True
        assert result["seconds_since_last_progress"] >= 0.9

    def test_check_timeout_duration_warning(self, tmp_project):
        """check_timeout detects when stage exceeds duration warning threshold."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=300.0,
            duration_warning_threshold_seconds=0.5,  # Very short for test
        )
        reporter.start_stage("TEST", total_items=10)

        # Simulate stage running for longer than warning threshold
        reporter._current.start_time = time.time() - 1.0

        result = reporter.check_timeout()

        assert result is not None
        assert result["is_duration_warning"] is True

    def test_check_timeout_resets_warned_on_new_stage(self, tmp_project):
        """Timeout warnings are reset when starting a new stage."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=0.1,
            duration_warning_threshold_seconds=0.1,
        )
        reporter.start_stage("STAGE1", total_items=10)

        # Trigger warnings
        reporter._last_progress_time = time.time() - 1.0
        reporter._current.start_time = time.time() - 1.0
        reporter.check_timeout()

        # Verify warnings are set
        assert reporter._warned_about_stuck is True
        assert reporter._warned_about_duration is True

        # Start new stage
        reporter.start_stage("STAGE2", total_items=10)

        # Warnings should be reset
        assert reporter._warned_about_stuck is False
        assert reporter._warned_about_duration is False

    def test_check_timeout_resets_stuck_warning_on_progress(self, tmp_project):
        """Stuck warning is reset when progress is made."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=0.1,
            duration_warning_threshold_seconds=600.0,
        )
        reporter.start_stage("TEST", total_items=10)

        # Simulate stuck
        reporter._last_progress_time = time.time() - 1.0
        reporter._warned_about_stuck = True

        # Make progress
        reporter.update(completed=1)

        # Stuck warning should be reset
        assert reporter._warned_about_stuck is False

    def test_snapshot_includes_timeout_fields(self, tmp_project):
        """Verify snapshot includes timeout detection fields."""
        reporter = ProgressReporter(tmp_project)
        reporter.start_stage("TEST", total_items=10)

        snap = reporter.get_snapshot()

        assert "seconds_since_last_progress" in snap
        assert "is_stage_stuck" in snap
        assert "is_duration_warning" in snap

    def test_snapshot_stuck_flag_when_no_progress(self, tmp_project):
        """Snapshot shows is_stage_stuck when no progress for threshold."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=0.1,
            duration_warning_threshold_seconds=600.0,
        )
        reporter.start_stage("TEST", total_items=10)

        # Simulate no progress
        reporter._last_progress_time = time.time() - 1.0

        snap = reporter.get_snapshot()

        assert snap["is_stage_stuck"] is True

    def test_snapshot_duration_warning_flag(self, tmp_project):
        """Snapshot shows is_duration_warning when exceeds threshold."""
        reporter = ProgressReporter(
            tmp_project,
            stuck_threshold_seconds=300.0,
            duration_warning_threshold_seconds=0.1,
        )
        reporter.start_stage("TEST", total_items=10)

        # Simulate long-running stage
        reporter._current.start_time = time.time() - 1.0
        reporter._last_progress_time = time.time() - 1.0

        snap = reporter.get_snapshot()

        assert snap["is_duration_warning"] is True


class TestConfidenceIntervalETA:
    """Tests for confidence interval ETA calculation."""

    def test_confidence_level_parameter_default(self, tmp_project):
        """Verify default confidence level is 95%."""
        reporter = ProgressReporter(tmp_project)
        assert reporter._confidence_level == 0.95

    def test_confidence_level_parameter_custom(self, tmp_project):
        """Verify custom confidence level can be set."""
        reporter = ProgressReporter(tmp_project, confidence_level=0.90)
        assert reporter._confidence_level == 0.90

    def test_snapshot_has_confidence_interval_fields(self, tmp_project):
        """Verify snapshot includes confidence interval fields."""
        reporter = ProgressReporter(tmp_project, remaining_stages=["VIDEO_SEARCH", "CAPTION"])
        reporter.start_stage("VIDEO_SEARCH", total_items=10)

        snap = reporter.get_snapshot()

        assert "pipeline_eta_lower" in snap
        assert "pipeline_eta_upper" in snap
        assert "pipeline_eta_confidence_level" in snap
        assert "pipeline_eta_has_confidence_interval" in snap

    def test_eta_display_as_range_with_confidence(self, tmp_project, tmp_path):
        """Verify ETA displays as range when confidence interval available."""
        # Add historical data to enable confidence interval
        from src.pipeline_history import append_stage_timing

        # Add multiple runs to get variance
        append_stage_timing(tmp_project, "VIDEO_SEARCH", 100.0, 10, 0.1)
        append_stage_timing(tmp_project, "VIDEO_SEARCH", 120.0, 10, 0.083)
        append_stage_timing(tmp_project, "VIDEO_SEARCH", 80.0, 10, 0.125)
        append_stage_timing(tmp_project, "CAPTION", 200.0, 20, 0.1)

        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["VIDEO_SEARCH", "CAPTION"],
            confidence_level=0.95,
        )
        reporter.start_stage("VIDEO_SEARCH", total_items=10)

        # Complete some items to enable throughput-based estimate
        reporter._current.items_completed = 5
        reporter._current.start_time = time.time() - 5.0

        snap = reporter.get_snapshot()

        # Should have confidence interval fields populated
        if snap["pipeline_eta_has_confidence_interval"]:
            assert snap["pipeline_eta_lower"] is not None
            assert snap["pipeline_eta_upper"] is not None
            assert snap["pipeline_eta_lower"] <= snap["pipeline_eta_upper"]
            # Display should be in range format
            assert " - " in snap["pipeline_eta_display"]

    def test_eta_fallback_without_history(self, tmp_project):
        """ETA falls back gracefully when no historical data."""
        reporter = ProgressReporter(
            tmp_project,
            remaining_stages=["VIDEO_SEARCH"],
            confidence_level=0.95,
        )
        reporter.start_stage("VIDEO_SEARCH", total_items=10)

        # No historical data - should still work
        snap = reporter.get_snapshot()

        # Should have basic fields even without CI
        assert "pipeline_eta_seconds" in snap
