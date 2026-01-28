"""
Tests for src/stages/__init__.py coverage gaps.

Targets:
- Line 47: StageResult.__bool__ returns self.success
- Line 168: list_stages returns registered stage names
"""

import pytest
from src.stages import StageResult, Stage, register_stage, get_stage, list_stages


class TestStageResultBool:
    """Test StageResult.__bool__ method (line 47)."""

    @pytest.mark.fast
    def test_bool_success_true(self):
        """Test StageResult with success=True is truthy."""
        result = StageResult(success=True)
        assert bool(result) is True
        assert result  # Direct boolean context

    @pytest.mark.fast
    def test_bool_success_false(self):
        """Test StageResult with success=False is falsy."""
        result = StageResult(success=False)
        assert bool(result) is False
        assert not result  # Direct boolean context

    @pytest.mark.fast
    def test_bool_ok_result(self):
        """Test StageResult.ok() is truthy."""
        result = StageResult.ok(data={"key": "value"})
        assert result  # Uses __bool__

    @pytest.mark.fast
    def test_bool_fail_result(self):
        """Test StageResult.fail() is falsy."""
        result = StageResult.fail("Error message")
        assert not result  # Uses __bool__

    @pytest.mark.fast
    def test_bool_in_if_statement(self):
        """Test StageResult in if statement."""
        success_result = StageResult.ok()
        fail_result = StageResult.fail("error")

        if success_result:
            passed_success = True
        else:
            passed_success = False

        if fail_result:
            passed_fail = True
        else:
            passed_fail = False

        assert passed_success is True
        assert passed_fail is False


class TestListStages:
    """Test list_stages function (line 168)."""

    @pytest.mark.fast
    def test_list_stages_returns_list(self):
        """Test that list_stages returns a list."""
        result = list_stages()
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_list_stages_contains_strings(self):
        """Test that list_stages contains string names."""
        result = list_stages()
        for name in result:
            assert isinstance(name, str)

    @pytest.mark.fast
    def test_list_stages_matches_registry(self):
        """Test that list_stages returns all registered stages."""
        from src.stages import _stage_registry

        result = list_stages()
        assert set(result) == set(_stage_registry.keys())


class TestStageRegistry:
    """Test stage registry functions."""

    @pytest.mark.fast
    def test_register_stage_decorator(self):
        """Test register_stage decorator adds to registry."""
        from src.stages import _stage_registry

        @register_stage
        class TestStage(Stage):
            name = "test_coverage_stage"
            description = "Test stage for coverage"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        # Verify registration
        assert "test_coverage_stage" in _stage_registry
        assert _stage_registry["test_coverage_stage"] == TestStage

        # Clean up
        del _stage_registry["test_coverage_stage"]

    @pytest.mark.fast
    def test_get_stage_existing(self):
        """Test get_stage returns registered stage."""
        from src.stages import _stage_registry

        # Register a test stage
        @register_stage
        class AnotherTestStage(Stage):
            name = "another_test_stage"
            description = "Another test stage"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        result = get_stage("another_test_stage")
        assert result == AnotherTestStage

        # Clean up
        del _stage_registry["another_test_stage"]

    @pytest.mark.fast
    def test_get_stage_nonexistent(self):
        """Test get_stage returns None for unknown stage."""
        result = get_stage("nonexistent_stage_xyz")
        assert result is None

    @pytest.mark.fast
    def test_register_stage_without_name(self):
        """Test register_stage with no name doesn't add to registry."""
        from src.stages import _stage_registry
        initial_count = len(_stage_registry)

        @register_stage
        class NoNameStage(Stage):
            # name = ""  # Empty name
            description = "Stage without name"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        # Should not be added (name is empty string from base class)
        assert len(_stage_registry) == initial_count


class TestStageResult:
    """Additional StageResult tests."""

    @pytest.mark.fast
    def test_ok_with_warnings(self):
        """Test StageResult.ok() with warnings."""
        result = StageResult.ok(
            data={"output": "value"},
            warnings=["Warning 1", "Warning 2"]
        )

        assert result.success is True
        assert result.data == {"output": "value"}
        assert len(result.warnings) == 2

    @pytest.mark.fast
    def test_fail_with_warnings(self):
        """Test StageResult.fail() with warnings."""
        result = StageResult.fail(
            error="Something went wrong",
            warnings=["Warning before failure"]
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert len(result.warnings) == 1


class TestStageRepr:
    """Test Stage __repr__ method."""

    @pytest.mark.fast
    def test_stage_repr(self):
        """Test Stage string representation."""
        from src.stages import Stage

        class MyTestStage(Stage):
            name = "my_test_stage"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        stage = MyTestStage()
        repr_str = repr(stage)

        assert "MyTestStage" in repr_str
        assert "my_test_stage" in repr_str
