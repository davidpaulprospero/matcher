"""
Tests for explicit stage dependency declarations (US-81-006).

Verifies:
- Each stage declares DEPENDS_ON and PRODUCES class attributes
- Pipeline.run() validates dependencies before running stages
- Running a stage without its dependency raises DependencyError
- STAGE_ORDER is consistent with declared dependencies
"""

import logging
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.stages import Stage, StageResult, DependencyError
from src.checkpoint import STAGE_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_all_stage_classes():
    """Import and return all 7 pipeline stage classes."""
    from src.stages.analyze import AnalyzeStage
    from src.stages.video_search import VideoSearchStage
    from src.stages.caption_stage import CaptionStage
    from src.stages.match import MatchStage
    from src.stages.iterative_match import IterativeMatchStage
    from src.stages.download_segments import DownloadVideoSegmentsStage
    from src.stages.output import OutputStage

    return [
        AnalyzeStage,
        VideoSearchStage,
        CaptionStage,
        MatchStage,
        IterativeMatchStage,
        DownloadVideoSegmentsStage,
        OutputStage,
    ]


def _make_minimal_config():
    """Create a minimal Config mock for PipelineOrchestrator."""
    config = MagicMock()
    config._config_hash = "test"
    config.cache.cache_dir = "/tmp/test_cache"
    config.embedding.provider = "sentence-transformers"
    config.pipeline.quality_gates.min_match_coverage = 0.5
    config.freeze = MagicMock()
    return config


# ---------------------------------------------------------------------------
# Test: DEPENDS_ON and PRODUCES exist on all stages
# ---------------------------------------------------------------------------

class TestStageDeclarations:
    """Verify all stage classes declare DEPENDS_ON and PRODUCES."""

    @pytest.mark.parametrize("stage_cls", _get_all_stage_classes(),
                             ids=lambda c: c.__name__)
    def test_depends_on_declared(self, stage_cls):
        """Each stage has a DEPENDS_ON list."""
        assert hasattr(stage_cls, 'DEPENDS_ON'), (
            f"{stage_cls.__name__} missing DEPENDS_ON"
        )
        assert isinstance(stage_cls.DEPENDS_ON, list), (
            f"{stage_cls.__name__}.DEPENDS_ON must be a list"
        )

    @pytest.mark.parametrize("stage_cls", _get_all_stage_classes(),
                             ids=lambda c: c.__name__)
    def test_produces_declared(self, stage_cls):
        """Each stage has a PRODUCES list."""
        assert hasattr(stage_cls, 'PRODUCES'), (
            f"{stage_cls.__name__} missing PRODUCES"
        )
        assert isinstance(stage_cls.PRODUCES, list), (
            f"{stage_cls.__name__}.PRODUCES must be a list"
        )


# ---------------------------------------------------------------------------
# Test: STAGE_ORDER consistency with DEPENDS_ON
# ---------------------------------------------------------------------------

class TestStageOrderConsistency:
    """Verify STAGE_ORDER is consistent with declared dependencies."""

    def test_stage_order_respects_depends_on(self):
        """No stage runs before its declared dependencies in STAGE_ORDER."""
        stage_classes = _get_all_stage_classes()
        name_to_cls = {cls.name: cls for cls in stage_classes if cls.name}

        stage_index = {name: i for i, name in enumerate(STAGE_ORDER)}

        violations = []
        for stage_name in STAGE_ORDER:
            cls = name_to_cls.get(stage_name)
            if cls is None:
                continue
            my_index = stage_index[stage_name]
            for dep in cls.DEPENDS_ON:
                dep_index = stage_index.get(dep)
                if dep_index is None:
                    violations.append(
                        f"{stage_name} depends on '{dep}' which is not in STAGE_ORDER"
                    )
                elif dep_index >= my_index:
                    violations.append(
                        f"{stage_name} (index {my_index}) depends on "
                        f"{dep} (index {dep_index}) which comes later in STAGE_ORDER"
                    )

        assert not violations, (
            "STAGE_ORDER violates declared dependencies:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_all_dependency_names_are_valid_stages(self):
        """All names in DEPENDS_ON refer to stages in STAGE_ORDER."""
        stage_classes = _get_all_stage_classes()
        stage_names = set(STAGE_ORDER)
        invalid = []
        for cls in stage_classes:
            for dep in cls.DEPENDS_ON:
                if dep not in stage_names:
                    invalid.append(f"{cls.name}.DEPENDS_ON references unknown stage '{dep}'")
        assert not invalid, "\n".join(invalid)


# ---------------------------------------------------------------------------
# Test: Pipeline dependency validation
# ---------------------------------------------------------------------------

class TestPipelineDependencyValidation:
    """Verify Pipeline.run() validates dependencies before each stage."""

    def test_match_without_caption_raises_dependency_error(self, tmp_path):
        """Running MATCH without CAPTION completing should fail with DependencyError."""
        from src.pipeline import PipelineOrchestrator
        from src.stages.match import MatchStage

        config = _make_minimal_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        # Add MATCH stage without its CAPTION dependency
        pipeline.add_stage(MatchStage())

        # Run should fail because CAPTION (and ANALYZE) haven't completed
        result = pipeline.run(resume=False)
        assert result is False, "Pipeline should fail when dependencies are unmet"

    def test_match_without_caption_logs_dependency_names(self, tmp_path, caplog):
        """Error message should name the missing dependency."""
        from src.pipeline import PipelineOrchestrator
        from src.stages.match import MatchStage

        config = _make_minimal_config()
        pipeline = PipelineOrchestrator(config, tmp_path)
        pipeline.add_stage(MatchStage())

        with caplog.at_level(logging.ERROR):
            pipeline.run(resume=False)

        error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        combined = " ".join(error_messages)
        # Should mention the missing dependencies by name
        assert "ANALYZE" in combined or "CAPTION" in combined, (
            f"Error should name missing dependencies. Got: {combined}"
        )

    def test_dependencies_satisfied_passes_validation(self, tmp_path):
        """Stages with all dependencies satisfied pass the dependency check."""
        from src.pipeline import PipelineOrchestrator

        config = _make_minimal_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        stage = MagicMock()
        stage.name = "MATCH"
        stage.DEPENDS_ON = ["ANALYZE", "CAPTION"]

        # Should not raise when all deps are satisfied
        pipeline._validate_stage_dependencies(
            stage, completed_stages={"ANALYZE", "VIDEO_SEARCH", "CAPTION"}
        )

    def test_validate_stage_dependencies_method_directly(self, tmp_path):
        """Test the _validate_stage_dependencies method directly."""
        from src.pipeline import PipelineOrchestrator

        config = _make_minimal_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        # Create a stage with dependencies
        stage = MagicMock()
        stage.name = "MATCH"
        stage.DEPENDS_ON = ["ANALYZE", "CAPTION"]

        # Should raise when dependencies are missing
        with pytest.raises(DependencyError) as exc_info:
            pipeline._validate_stage_dependencies(stage, completed_stages=set())

        assert "ANALYZE" in str(exc_info.value)
        assert "CAPTION" in str(exc_info.value)

        # Should pass when dependencies are satisfied
        pipeline._validate_stage_dependencies(
            stage, completed_stages={"ANALYZE", "CAPTION", "VIDEO_SEARCH"}
        )

    def test_checkpoint_restored_stages_count_as_completed(self, tmp_path):
        """Stages restored from checkpoint should satisfy dependencies."""
        from src.pipeline import PipelineOrchestrator

        config = _make_minimal_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        # A stage with no deps that can be skipped
        stage_a = MagicMock(spec=Stage)
        stage_a.name = "ANALYZE"
        stage_a.DEPENDS_ON = []
        stage_a.PRODUCES = ['keywords']
        stage_a.can_skip.return_value = True
        stage_a.restore.return_value = True

        # A stage that depends on A
        stage_b = MagicMock(spec=Stage)
        stage_b.name = "VIDEO_SEARCH"
        stage_b.DEPENDS_ON = ["ANALYZE"]
        stage_b.PRODUCES = ['video_ids']
        stage_b.can_skip.return_value = False
        stage_b.validate_inputs.return_value = None
        stage_b.run.return_value = StageResult.ok({'ids': []})

        pipeline.stages = [stage_a, stage_b]
        pipeline.resume_mode = True

        result = pipeline.run(resume=False)  # resume=False but resume_mode is set
        # stage_b.run should have been called (not blocked by deps)
        assert stage_b.run.called


# ---------------------------------------------------------------------------
# US-138-007: Startup dependency validation tests
# ---------------------------------------------------------------------------

class TestStartupDependencyValidation:
    """Tests for pipeline startup dependency validation (US-138-007)."""

    def test_validate_no_cycles_detects_direct_cycle(self):
        """validate_no_cycles should detect a direct A -> B -> A cycle."""
        from src.stages import validate_no_cycles, build_dependency_graph
        from unittest.mock import patch

        # Temporarily create a cycle by patching the registry
        original_registry = None

        def mock_build_graph():
            return {
                "ANALYZE": [],
                "VIDEO_SEARCH": ["ANALYZE"],
                "CAPTION": ["VIDEO_SEARCH"],
                "MATCH": ["CAPTION"],
                "ITERATIVE_MATCH": ["MATCH"],
                "DOWNLOAD_SEGMENTS": ["ITERATIVE_MATCH"],
                "OUTPUT": ["DOWNLOAD_SEGMENTS", "MATCH"],  # Normal deps
                # Create a cycle: MATCH depends on OUTPUT and OUTPUT depends on MATCH
                "CYCLE_STAGE_A": ["CYCLE_STAGE_B"],
                "CYCLE_STAGE_B": ["CYCLE_STAGE_A"],
            }

        with patch('src.stages.build_dependency_graph', mock_build_graph):
            from src.stages import validate_no_cycles as vnc
            result = vnc()
            assert result is not None
            assert "CYCLE_STAGE_A" in result or "CYCLE_STAGE_B" in result

    def test_validate_dependencies_passes_for_valid_graph(self, tmp_path):
        """_validate_dependencies should pass for a valid dependency graph."""
        from src.pipeline import PipelineOrchestrator
        from src.stages import get_all_stages
        from src.checkpoint import STAGE_ORDER

        config = _make_minimal_config()
        pipeline = PipelineOrchestrator(config, tmp_path)

        # Should not raise - valid graph
        errors = pipeline._validate_dependencies()
        assert len(errors) == 0, f"Expected no errors, got: {errors}"

    def test_validate_dependencies_fails_on_invalid_depends_on(self, tmp_path):
        """_validate_dependencies should fail if DEPENDS_ON references non-STAGE_ORDER."""
        from src.pipeline import PipelineOrchestrator
        from unittest.mock import patch, MagicMock

        config = _make_minimal_config()

        # Create a mock stage with invalid dependency
        mock_stage = MagicMock()
        mock_stage.name = "INVALID_STAGE"
        mock_stage.DEPENDS_ON = ["NONEXISTENT_STAGE"]

        # Patch get_all_stages to return our mock
        # The error should be raised at __init__ time since validation runs there
        with patch('src.pipeline.get_all_stages', return_value={"INVALID_STAGE": mock_stage}):
            with pytest.raises(ValueError) as exc_info:
                PipelineOrchestrator(config, tmp_path)

            assert "NONEXISTENT_STAGE" in str(exc_info.value)
            assert "not in STAGE_ORDER" in str(exc_info.value)

    def test_validate_dependencies_warns_on_unused_registered_stages(self, tmp_path, caplog):
        """_validate_dependencies should warn about registered stages not in pipeline."""
        from src.pipeline import PipelineOrchestrator
        from unittest.mock import patch, MagicMock
        import logging

        config = _make_minimal_config()

        # Create a mock unused stage
        mock_unused = MagicMock()
        mock_unused.name = "UNUSED_STAGE"
        mock_unused.DEPENDS_ON = []

        # Add a real stage to the pipeline
        from src.stages.analyze import AnalyzeStage

        with patch('src.pipeline.get_all_stages', return_value={
            "ANALYZE": AnalyzeStage,
            "UNUSED_STAGE": mock_unused,
        }):
            pipeline = PipelineOrchestrator(config, tmp_path)
            pipeline.stages = [AnalyzeStage()]  # Only ANALYZE in pipeline

            with caplog.at_level(logging.WARNING):
                errors = pipeline._validate_dependencies()

            # Should pass but log a warning
            assert len(errors) == 0

            # Check warning was logged about unused stage
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("UNUSED_STAGE" in msg for msg in warning_messages), \
                f"Expected warning about UNUSED_STAGE, got: {warning_messages}"

    def test_startup_validation_fails_fast_on_cycle(self, tmp_path):
        """Pipeline initialization should fail fast if cycle detected."""
        from src.pipeline import PipelineOrchestrator
        from unittest.mock import patch, MagicMock

        config = _make_minimal_config()

        # Create a mock stage with a cycle
        mock_stage_a = MagicMock()
        mock_stage_a.name = "CYCLE_A"
        mock_stage_a.DEPENDS_ON = ["CYCLE_B"]

        mock_stage_b = MagicMock()
        mock_stage_b.name = "CYCLE_B"
        mock_stage_b.DEPENDS_ON = ["CYCLE_A"]

        # Patch validate_no_cycles to return a cycle
        with patch('src.pipeline.validate_no_cycles', return_value="CYCLE_A -> CYCLE_B -> CYCLE_A"):
            with pytest.raises(ValueError) as exc_info:
                PipelineOrchestrator(config, tmp_path)

            assert "Dependency cycle detected" in str(exc_info.value)

    def test_stage_order_includes_all_depends_on_stages(self):
        """All DEPENDS_ON values should reference stages in STAGE_ORDER."""
        from src.stages import get_all_stages
        from src.checkpoint import STAGE_ORDER

        stage_names_in_order = set(STAGE_ORDER)
        all_stages = get_all_stages()

        errors = []
        for stage_name, stage_cls in all_stages.items():
            deps = getattr(stage_cls, 'DEPENDS_ON', [])
            for dep in deps:
                if dep not in stage_names_in_order:
                    errors.append(f"{stage_name} depends on {dep} not in STAGE_ORDER")

        assert len(errors) == 0, f"Invalid dependencies: {errors}"
