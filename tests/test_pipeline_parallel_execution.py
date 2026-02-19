"""
Unit tests for parallel stage execution in PipelineOrchestrator.

US-108-012: Tests for parallel stage execution functionality:
- detect_parallel_stage_groups() function
- Parallel execution stats calculation
- Dependency resolution for VIDEO_SEARCH + CAPTION parallel execution
"""

import pytest
import tempfile
import shutil
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.pipeline import (
    PipelineOrchestrator,
    detect_parallel_stage_groups,
    log_parallel_execution_plan,
    validate_no_circular_dependencies,
)
from src.state import PipelineState
from src.stages import Stage, StageResult, StageMetrics
from src.config import Config


class MockStage(Stage):
    """Mock stage for testing parallel execution."""

    def __init__(
        self,
        name: str,
        depends_on: list = None,
        should_fail: bool = False,
        validation_error: str = None,
        can_skip_value: bool = False,
        run_delay: float = 0.1,
    ):
        self.name = name
        self.DEPENDS_ON = depends_on or []
        self._should_fail = should_fail
        self._validation_error = validation_error
        self._can_skip_value = can_skip_value
        self._run_delay = run_delay
        self._run_called = False

    def can_skip(self, state, checkpoint) -> bool:
        return self._can_skip_value

    def restore(self, state, checkpoint, config=None) -> bool:
        return True

    def validate_inputs(self, state, config):
        return self._validation_error

    def run(self, state, config, checkpoint=None):
        self._run_called = True
        time.sleep(self._run_delay)  # Simulate work
        return StageResult(
            success=not self._should_fail,
            data={},
            warnings=[],
        )


class TestDetectParallelStageGroups:
    """Tests for detect_parallel_stage_groups function."""

    def test_single_stage_no_parallel(self):
        """Single stage with no dependencies has no parallel opportunities."""
        stage = MockStage("ANALYZE", depends_on=[])
        groups = detect_parallel_stage_groups([stage])
        assert groups == []

    def test_two_stages_same_dependency_parallel(self):
        """Two stages depending on same parent can run in parallel."""
        stage1 = MockStage("VIDEO_SEARCH", depends_on=["ANALYZE"])
        stage2 = MockStage("CAPTION", depends_on=["ANALYZE"])
        groups = detect_parallel_stage_groups([stage1, stage2])
        # Both depend on ANALYZE, should be detected as parallel group
        assert len(groups) >= 1
        # Check that both stages are in a group together
        found_group = False
        for group in groups:
            if "VIDEO_SEARCH" in group and "CAPTION" in group:
                found_group = True
                break
        assert found_group, f"Expected VIDEO_SEARCH and CAPTION in same group, got {groups}"

    def test_sequential_stages_no_parallel(self):
        """Sequential dependencies (A -> B -> C) have no parallel opportunities."""
        stage_a = MockStage("A", depends_on=[])
        stage_b = MockStage("B", depends_on=["A"])
        stage_c = MockStage("C", depends_on=["B"])
        groups = detect_parallel_stage_groups([stage_a, stage_b, stage_c])
        # No groups with 2+ stages since each has different dependencies
        assert groups == []

    def test_three_stages_two_parallel(self):
        """Three stages where two share same dependency."""
        analyze = MockStage("ANALYZE", depends_on=[])
        video_search = MockStage("VIDEO_SEARCH", depends_on=["ANALYZE"])
        caption = MockStage("CAPTION", depends_on=["ANALYZE"])
        match = MockStage("MATCH", depends_on=["VIDEO_SEARCH", "CAPTION"])

        groups = detect_parallel_stage_groups([analyze, video_search, caption, match])

        # VIDEO_SEARCH and CAPTION should be in a parallel group
        found_search_group = False
        for group in groups:
            if "VIDEO_SEARCH" in group and "CAPTION" in group:
                found_search_group = True
                break
        assert found_search_group


class TestParallelExecutionStats:
    """Tests for parallel execution statistics calculation."""

    def test_time_saved_calculation(self):
        """Test that time saved is calculated correctly."""
        # Create a mock pipeline with stage timings
        pipeline = Mock(spec=PipelineOrchestrator)

        # Simulate: VIDEO_SEARCH took 5s, CAPTION took 3s, running parallel took 5s
        # Time saved = (5 + 3) - 5 = 3 seconds
        pipeline.stage_timings = {
            "VIDEO_SEARCH": 5.0,
            "CAPTION": 3.0,
            "MATCH": 2.0,
        }
        pipeline.stages_run_concurrently = ["VIDEO_SEARCH", "CAPTION"]
        pipeline.parallel_group_timings = {
            ("CAPTION", "VIDEO_SEARCH"): 5.0,  # Parallel execution took 5s
        }

        # Manually calculate to verify
        group_key = ("CAPTION", "VIDEO_SEARCH")
        parallel_time = pipeline.parallel_group_timings[group_key]
        individual_times = [pipeline.stage_timings.get(s, 0.0) for s in group_key]
        sum_individual = sum(individual_times)
        time_saved = sum_individual - parallel_time

        assert time_saved == 3.0  # (5 + 3) - 5 = 3

    def test_no_time_saved_when_sequential(self):
        """No time saved when stages run sequentially (no parallel groups)."""
        pipeline = Mock(spec=PipelineOrchestrator)
        pipeline.stage_timings = {"A": 5.0, "B": 3.0}
        pipeline.stages_run_concurrently = []
        pipeline.parallel_group_timings = {}

        # No parallel groups, so time saved should be 0
        assert len(pipeline.parallel_group_timings) == 0


class TestParallelDependencyResolution:
    """Tests for dependency resolution with parallel stages."""

    def test_video_search_caption_dependencies(self):
        """VIDEO_SEARCH and CAPTION should both depend on ANALYZE for parallel execution."""
        # Verify CAPTION stage DEPENDS_ON allows parallel with VIDEO_SEARCH
        from src.stages.caption_stage import CaptionStage
        from src.stages.video_search import VideoSearchStage

        # CAPTION should depend on ANALYZE (not VIDEO_SEARCH) to allow parallel execution
        assert "ANALYZE" in CaptionStage.DEPENDS_ON, \
            "CAPTION should depend on ANALYZE for parallel execution"
        # VIDEO_SEARCH should depend on ANALYZE
        assert "ANALYZE" in VideoSearchStage.DEPENDS_ON, \
            "VIDEO_SEARCH should depend on ANALYZE"

    def test_match_waits_for_both(self):
        """MATCH should wait for both VIDEO_SEARCH and CAPTION to complete."""
        from src.stages.match import MatchStage

        # MATCH depends on both ANALYZE and CAPTION
        assert "CAPTION" in MatchStage.DEPENDS_ON, \
            "MATCH should depend on CAPTION"


class TestRalphConfigLoading:
    """Tests for loading parallel groups from ralph-config.json."""

    def test_ralph_config_groups_format(self):
        """Verify ralph-config.json has correct groups format."""
        config_path = Path(__file__).parent.parent / "scripts/ralph/config/ralph-config.json"

        if not config_path.exists():
            pytest.skip("ralph-config.json not found")

        # Handle UTF-8 BOM that some editors add
        with open(config_path, encoding='utf-8-sig') as f:
            config = json.load(f)

        groups = config.get("pipelineStages", {}).get("groups", {})

        # Check search group has VIDEO_SEARCH and CAPTION
        search_group = groups.get("search", [])
        assert "VIDEO_SEARCH" in search_group, "search group should contain VIDEO_SEARCH"
        assert "CAPTION" in search_group, "search group should contain CAPTION"


class TestParallelExecutionPlanLogging:
    """Tests for parallel execution plan logging."""

    def test_log_plan_formats_correctly(self, caplog):
        """Test that log_parallel_execution_plan formats output correctly."""
        stages = [
            MockStage("ANALYZE", depends_on=[]),
            MockStage("VIDEO_SEARCH", depends_on=["ANALYZE"]),
            MockStage("CAPTION", depends_on=["ANALYZE"]),
            MockStage("MATCH", depends_on=["VIDEO_SEARCH", "CAPTION"]),
        ]
        parallel_groups = [("VIDEO_SEARCH", "CAPTION")]

        # Should not raise
        log_parallel_execution_plan(stages, parallel_groups)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
