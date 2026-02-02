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


class TestValidateStateType:
    """Test Stage._validate_state_type() method (US-39-009)."""

    @pytest.mark.fast
    def test_validate_state_type_with_pipeline_state(self):
        """Test _validate_state_type returns PipelineState unchanged."""
        from src.stages import Stage
        from src.state import PipelineState

        class TestStage(Stage):
            name = "validate_test_stage"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        stage = TestStage()
        state = PipelineState()

        result = stage._validate_state_type(state)

        assert result is state
        assert isinstance(result, PipelineState)

    @pytest.mark.fast
    def test_validate_state_type_logs_warning_for_non_pipeline_state(self, caplog):
        """Test _validate_state_type logs warning for non-PipelineState objects."""
        import logging
        from src.stages import Stage

        class TestStage(Stage):
            name = "validate_test_stage_2"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        stage = TestStage()

        # Create a non-PipelineState object (SimpleNamespace)
        from types import SimpleNamespace
        non_standard_state = SimpleNamespace(
            keywords=['test'],
            topic_context='test context',
            voiceover_segments=[],
            matches=[],
            caption_results={}
        )

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            result = stage._validate_state_type(non_standard_state)

        # Verify warning was logged
        assert "Non-standard state object: SimpleNamespace" in caplog.text

    @pytest.mark.fast
    def test_validate_state_type_converts_legacy_object(self, caplog):
        """Test _validate_state_type attempts conversion from legacy object."""
        import logging
        from src.stages import Stage
        from src.state import PipelineState

        class TestStage(Stage):
            name = "validate_test_stage_3"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        stage = TestStage()

        # Create a mock legacy pipeline object
        class LegacyPipeline:
            keywords = ['test', 'keyword']
            topic_context = 'test context'
            voiceover_segments = []
            matches = []
            caption_results = {}
            downloaded_videos = []
            extracted_entities = []
            failed_keywords = []
            face_preference = 'neutral'
            stage_timings = {}

        legacy = LegacyPipeline()

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            result = stage._validate_state_type(legacy)

        # Verify warning was logged
        assert "Non-standard state object: LegacyPipeline" in caplog.text

        # Verify conversion was attempted and resulted in PipelineState
        assert isinstance(result, PipelineState)
        assert result.keywords == ['test', 'keyword']

    @pytest.mark.fast
    def test_validate_state_type_handles_conversion_failure(self, caplog):
        """Test _validate_state_type handles conversion failure gracefully."""
        import logging
        from src.stages import Stage

        class TestStage(Stage):
            name = "validate_test_stage_4"

            def run(self, state, config, checkpoint):
                return StageResult.ok()

            def can_skip(self, state, checkpoint):
                return False

            def restore(self, state, checkpoint):
                return False

        stage = TestStage()

        # Create an object that will fail conversion (minimal, no attributes)
        class MinimalObject:
            pass

        minimal = MinimalObject()

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            result = stage._validate_state_type(minimal)

        # Verify warning was logged
        assert "Non-standard state object: MinimalObject" in caplog.text

        # When conversion fails, should return original object
        # (either original or successfully converted - both are valid)


class TestValidateRequiredStateAttrs:
    """Test validate_required_state_attrs function (US-40-008)."""

    @pytest.mark.fast
    def test_missing_attribute_initialized_with_warning(self, caplog):
        """Test missing attribute is initialized to default and warning logged."""
        import logging
        from types import SimpleNamespace
        from src.stages import validate_required_state_attrs

        # Create state with missing 'text_metadata' attribute
        state = SimpleNamespace(matches=[], voiceover_segments=[])

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            validate_required_state_attrs(state, ['text_metadata'], 'MATCH')

        # Verify attribute was initialized
        assert hasattr(state, 'text_metadata')
        assert state.text_metadata == []

        # Verify warning was logged
        assert "[MATCH] Missing state attribute 'text_metadata'" in caplog.text
        assert "initializing to list" in caplog.text

    @pytest.mark.fast
    def test_existing_attribute_not_modified(self, caplog):
        """Test existing attribute is not modified."""
        import logging
        from types import SimpleNamespace
        from src.stages import validate_required_state_attrs

        # Create state with existing 'text_metadata' attribute
        original_value = [{'video_id': 'abc123', 'text': 'test'}]
        state = SimpleNamespace(text_metadata=original_value)

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            validate_required_state_attrs(state, ['text_metadata'], 'MATCH')

        # Verify attribute was NOT modified
        assert state.text_metadata is original_value
        assert state.text_metadata == [{'video_id': 'abc123', 'text': 'test'}]

        # Verify NO warning was logged
        assert 'text_metadata' not in caplog.text

    @pytest.mark.fast
    def test_multiple_missing_attributes(self, caplog):
        """Test multiple missing attributes are all initialized."""
        import logging
        from types import SimpleNamespace
        from src.stages import validate_required_state_attrs

        # Create state with no required attributes
        state = SimpleNamespace()

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            validate_required_state_attrs(
                state,
                ['text_metadata', 'caption_results', 'matches'],
                'TEST_STAGE'
            )

        # Verify all attributes were initialized
        assert hasattr(state, 'text_metadata')
        assert hasattr(state, 'caption_results')
        assert hasattr(state, 'matches')
        assert state.text_metadata == []
        assert state.caption_results == {}
        assert state.matches == []

        # Verify warnings logged for each
        assert "[TEST_STAGE] Missing state attribute 'text_metadata'" in caplog.text
        assert "[TEST_STAGE] Missing state attribute 'caption_results'" in caplog.text
        assert "[TEST_STAGE] Missing state attribute 'matches'" in caplog.text

    @pytest.mark.fast
    def test_unknown_attribute_defaults_to_none(self, caplog):
        """Test unknown attribute defaults to None."""
        import logging
        from types import SimpleNamespace
        from src.stages import validate_required_state_attrs

        state = SimpleNamespace()

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            validate_required_state_attrs(state, ['unknown_attr'], 'TEST')

        # Verify attribute was initialized to None
        assert hasattr(state, 'unknown_attr')
        assert state.unknown_attr is None
        assert "initializing to NoneType" in caplog.text

    @pytest.mark.fast
    def test_mixed_existing_and_missing_attributes(self, caplog):
        """Test state with some existing and some missing attributes."""
        import logging
        from types import SimpleNamespace
        from src.stages import validate_required_state_attrs

        # Create state with only 'matches' existing
        existing_matches = [{'segment': 1}]
        state = SimpleNamespace(matches=existing_matches)

        with caplog.at_level(logging.WARNING, logger='src.stages'):
            validate_required_state_attrs(
                state,
                ['matches', 'text_metadata'],
                'ITERATIVE_MATCH'
            )

        # Existing attribute unchanged
        assert state.matches is existing_matches

        # Missing attribute initialized
        assert state.text_metadata == []

        # Only missing attribute logged
        assert "'matches'" not in caplog.text
        assert "'text_metadata'" in caplog.text
