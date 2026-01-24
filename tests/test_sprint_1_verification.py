"""
Sprint 1 Verification Tests

US-001: Create sprint verification test
Validates that core pipeline components can be imported and instantiated.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPipelineImports:
    """Test that core pipeline modules can be imported."""

    def test_import_stages(self):
        """Test src.stages can be imported."""
        from src import stages
        assert stages is not None

    def test_import_pipeline(self):
        """Test src.pipeline can be imported."""
        from src import pipeline
        assert pipeline is not None

    def test_import_agents(self):
        """Test src.agents can be imported."""
        from src import agents
        assert agents is not None

    def test_import_stage_classes(self):
        """Test individual stage classes can be imported."""
        from src.stages import (
            Stage,
            StageResult,
            register_stage,
            get_stage,
            list_stages,
        )
        assert Stage is not None
        assert StageResult is not None
        assert register_stage is not None
        assert get_stage is not None
        assert list_stages is not None


class TestStageRegistry:
    """Test that all 12 registered stages can be instantiated."""

    @pytest.fixture(autouse=True)
    def import_all_stages(self):
        """Import all stage modules to trigger registration."""
        # Import all stage modules to trigger @register_stage decorators
        from src.stages import analyze  # noqa: F401
        from src.stages import entity_images  # noqa: F401
        from src.stages import entity_videos  # noqa: F401
        from src.stages import download  # noqa: F401
        from src.stages import stock  # noqa: F401
        from src.stages import broll_download  # noqa: F401
        from src.stages import remix  # noqa: F401
        from src.stages import transcribe  # noqa: F401
        from src.stages import scene_detection  # noqa: F401
        from src.stages import match  # noqa: F401
        from src.stages import broll_match  # noqa: F401
        from src.stages import output  # noqa: F401

    def test_all_stages_registered(self):
        """Test that expected stages are registered."""
        from src.stages import list_stages

        registered = set(list_stages())
        expected = {
            "ANALYZE",
            "ENTITY_IMAGES",
            "ENTITY_VIDEOS",
            "DOWNLOAD",
            "STOCK",
            "BROLL_DOWNLOAD",
            "REMIX",
            "TRANSCRIBE",
            "SCENE_DETECTION",
            "MATCH",
            "BROLL_MATCH",
            "OUTPUT",
        }

        # Check that all expected stages are registered
        missing = expected - registered
        assert not missing, f"Missing stages: {missing}"

    def test_stages_can_be_retrieved(self):
        """Test that each registered stage can be retrieved."""
        from src.stages import list_stages, get_stage

        for stage_name in list_stages():
            stage_class = get_stage(stage_name)
            assert stage_class is not None, f"Failed to get stage: {stage_name}"

    def test_stages_can_be_instantiated(self):
        """Test that each registered stage can be instantiated."""
        from src.stages import list_stages, get_stage

        for stage_name in list_stages():
            stage_class = get_stage(stage_name)
            # Instantiate without arguments
            stage = stage_class()
            assert stage is not None, f"Failed to instantiate: {stage_name}"
            assert hasattr(stage, "name"), f"Stage missing 'name' attribute: {stage_name}"
            assert hasattr(stage, "run"), f"Stage missing 'run' method: {stage_name}"


class TestPipelineOrchestrator:
    """Test that PipelineOrchestrator can be created with mock config."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = MagicMock()
        config._config_hash = "test_hash"
        config.pipeline = MagicMock()
        config.healing = MagicMock()
        config.healing.enabled = False
        return config

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        return project_dir

    def test_orchestrator_can_be_created(self, mock_config, temp_project_dir):
        """Test PipelineOrchestrator instantiation."""
        from src.pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            config=mock_config,
            project_dir=temp_project_dir,
        )

        assert orchestrator is not None
        assert orchestrator.config == mock_config
        assert orchestrator.project_dir == temp_project_dir
        assert orchestrator.stages == []

    def test_orchestrator_add_stage(self, mock_config, temp_project_dir):
        """Test add_stage fluent interface."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import get_stage
        # Import to trigger registration
        from src.stages import analyze  # noqa: F401

        orchestrator = PipelineOrchestrator(
            config=mock_config,
            project_dir=temp_project_dir,
        )

        # Get a real stage
        AnalyzeStage = get_stage("ANALYZE")
        assert AnalyzeStage is not None, "ANALYZE stage should be registered after import"
        stage = AnalyzeStage()

        # Add stage and check fluent interface
        result = orchestrator.add_stage(stage)
        assert result is orchestrator, "add_stage should return self"
        assert len(orchestrator.stages) == 1
        assert orchestrator.stages[0] is stage

    def test_orchestrator_has_checkpoint(self, mock_config, temp_project_dir):
        """Test orchestrator initializes checkpoint manager."""
        from src.pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator(
            config=mock_config,
            project_dir=temp_project_dir,
        )

        assert orchestrator.checkpoint is not None
        assert hasattr(orchestrator.checkpoint, "exists")
        assert hasattr(orchestrator.checkpoint, "load")
        assert hasattr(orchestrator.checkpoint, "save")


class TestStageResult:
    """Test StageResult behavior."""

    def test_stage_result_ok(self):
        """Test StageResult.ok() creates successful result."""
        from src.stages import StageResult

        result = StageResult.ok(data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert bool(result) is True

    def test_stage_result_fail(self):
        """Test StageResult.fail() creates failed result."""
        from src.stages import StageResult

        result = StageResult.fail("Error message")
        assert result.success is False
        assert result.error == "Error message"
        assert bool(result) is False

    def test_stage_result_with_warnings(self):
        """Test StageResult with warnings."""
        from src.stages import StageResult

        result = StageResult.ok(data={}, warnings=["warn1", "warn2"])
        assert result.success is True
        assert result.warnings == ["warn1", "warn2"]


class TestSprint1PipelineVerification:
    """
    Sprint 1 Pipeline verification tests.

    US-001: Create sprint 1 pipeline verification test
    Tests that validate the core pipeline stage infrastructure.
    """

    def test_stage_metrics_import(self):
        """Test that StageMetrics can be imported from src.stages."""
        from src.stages import StageMetrics
        assert StageMetrics is not None

        # Verify it's a dataclass with expected fields
        metrics = StageMetrics(
            items_processed=10,
            items_failed=2,
            duration_seconds=5.5
        )
        assert metrics.items_processed == 10
        assert metrics.items_failed == 2
        assert metrics.duration_seconds == 5.5

    def test_stage_metrics_from_dict(self):
        """Test StageMetrics.from_dict() class method."""
        from src.stages import StageMetrics

        data = {
            "items_processed": 15,
            "items_failed": 3,
            "duration_seconds": 12.3
        }
        metrics = StageMetrics.from_dict(data)
        assert metrics.items_processed == 15
        assert metrics.items_failed == 3
        assert metrics.duration_seconds == 12.3

    def test_orchestrator_accepts_on_stage_start_callback(self, tmp_path):
        """Test that PipelineOrchestrator.run() accepts on_stage_start callback."""
        from src.pipeline import PipelineOrchestrator
        from unittest.mock import MagicMock
        import inspect

        # Create mock config
        config = MagicMock()
        config._config_hash = "test"
        config.pipeline = MagicMock()
        config.healing = MagicMock()
        config.healing.enabled = False

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        orchestrator = PipelineOrchestrator(config=config, project_dir=project_dir)

        # Check that run() method accepts on_stage_start parameter
        sig = inspect.signature(orchestrator.run)
        assert "on_stage_start" in sig.parameters, "run() should accept on_stage_start callback"

    def test_orchestrator_accepts_on_stage_complete_callback(self, tmp_path):
        """Test that PipelineOrchestrator.run() accepts on_stage_complete callback."""
        from src.pipeline import PipelineOrchestrator
        from unittest.mock import MagicMock
        import inspect

        # Create mock config
        config = MagicMock()
        config._config_hash = "test"
        config.pipeline = MagicMock()
        config.healing = MagicMock()
        config.healing.enabled = False

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        orchestrator = PipelineOrchestrator(config=config, project_dir=project_dir)

        # Check that run() method accepts on_stage_complete parameter
        sig = inspect.signature(orchestrator.run)
        assert "on_stage_complete" in sig.parameters, "run() should accept on_stage_complete callback"

    def test_stage_callbacks_are_invoked(self, tmp_path):
        """Test that stage callbacks are actually invoked during pipeline run."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import Stage, StageResult, register_stage
        from unittest.mock import MagicMock

        # Create mock config
        config = MagicMock()
        config._config_hash = "test"
        config.pipeline = MagicMock()
        config.healing = MagicMock()
        config.healing.enabled = False

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create a simple test stage
        class TestStage(Stage):
            name = "TEST_CALLBACK_STAGE"
            description = "Test stage for callbacks"

            def run(self, state, config, checkpoint):
                return StageResult.ok({"test": True})

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint, config=None):
                return False

        orchestrator = PipelineOrchestrator(config=config, project_dir=project_dir)
        orchestrator.add_stage(TestStage())

        # Track callback invocations
        start_calls = []
        complete_calls = []

        def on_start(stage_name):
            start_calls.append(stage_name)

        def on_complete(stage_name, result, elapsed):
            complete_calls.append((stage_name, result.success, elapsed))

        # Run pipeline with callbacks
        orchestrator.run(
            resume=False,
            on_stage_start=on_start,
            on_stage_complete=on_complete
        )

        # Verify callbacks were called
        assert "TEST_CALLBACK_STAGE" in start_calls, "on_stage_start should be called"
        assert len(complete_calls) == 1, "on_stage_complete should be called once"
        assert complete_calls[0][0] == "TEST_CALLBACK_STAGE"
        assert complete_calls[0][1] is True  # success
        assert complete_calls[0][2] >= 0  # elapsed time
