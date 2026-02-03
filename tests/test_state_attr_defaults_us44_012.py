"""
Tests for US-44-012: _STATE_ATTR_DEFAULTS covers all PipelineState fields.

Verifies that validate_required_state_attrs initializes missing fields to
the correct type (list, dict, str) instead of falling back to None.
"""

import dataclasses
import pytest
from types import SimpleNamespace

from src.stages import validate_required_state_attrs, _STATE_ATTR_DEFAULTS
from src.state import PipelineState


class TestDefaultsMatchPipelineState:
    """Every PipelineState field should have a matching _STATE_ATTR_DEFAULTS entry."""

    @pytest.mark.fast
    def test_all_pipeline_state_fields_have_defaults(self):
        """All PipelineState dataclass fields should be in _STATE_ATTR_DEFAULTS."""
        pipeline_fields = {f.name for f in dataclasses.fields(PipelineState)}
        defaults_keys = set(_STATE_ATTR_DEFAULTS.keys())

        missing = pipeline_fields - defaults_keys
        # Allow 'videos' legacy key in defaults but not required in PipelineState
        assert not missing, (
            f"PipelineState fields missing from _STATE_ATTR_DEFAULTS: {missing}"
        )

    @pytest.mark.fast
    def test_default_types_match_pipeline_state(self):
        """Each _STATE_ATTR_DEFAULTS value type matches PipelineState default_factory."""
        pipeline_defaults = {}
        for f in dataclasses.fields(PipelineState):
            if f.default is not dataclasses.MISSING:
                pipeline_defaults[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                pipeline_defaults[f.name] = f.default_factory()

        for attr, default_val in _STATE_ATTR_DEFAULTS.items():
            if attr not in pipeline_defaults:
                continue  # Legacy key like 'videos'
            expected = pipeline_defaults[attr]
            assert type(default_val) == type(expected), (
                f"_STATE_ATTR_DEFAULTS['{attr}'] is {type(default_val).__name__} "
                f"but PipelineState default is {type(expected).__name__}"
            )


class TestNewDefaultEntries:
    """Verify the specific entries added by US-44-012."""

    @pytest.mark.fast
    @pytest.mark.parametrize("attr,expected_type", [
        ('video_ids', list),
        ('video_search_results', list),
        ('downloaded_segments', list),
        ('output_files', list),
        ('otio_files', list),
        ('search_failed_keywords', list),
        ('extracted_entities', list),
        ('topic_context', str),
        ('face_preference', str),
        ('stage_timings', dict),
    ])
    def test_entry_exists_with_correct_type(self, attr, expected_type):
        """Each new entry exists in _STATE_ATTR_DEFAULTS with the correct type."""
        assert attr in _STATE_ATTR_DEFAULTS, f"'{attr}' missing from _STATE_ATTR_DEFAULTS"
        assert isinstance(_STATE_ATTR_DEFAULTS[attr], expected_type), (
            f"_STATE_ATTR_DEFAULTS['{attr}'] should be {expected_type.__name__}, "
            f"got {type(_STATE_ATTR_DEFAULTS[attr]).__name__}"
        )


class TestValidateInitializesCorrectTypes:
    """validate_required_state_attrs must set the correct type, not None."""

    @pytest.mark.fast
    @pytest.mark.parametrize("attr,expected_type", [
        ('video_ids', list),
        ('video_search_results', list),
        ('downloaded_segments', list),
        ('output_files', list),
        ('search_failed_keywords', list),
        ('extracted_entities', list),
        ('topic_context', str),
        ('face_preference', str),
        ('caption_results', dict),
        ('alternatives', dict),
        ('stage_timings', dict),
        ('matches', list),
        ('voiceover_segments', list),
        ('text_metadata', list),
    ])
    def test_missing_attr_initialized_to_correct_type(self, attr, expected_type):
        """Missing attr is set to the correct default type, not None."""
        state = SimpleNamespace()  # No attributes
        validate_required_state_attrs(state, [attr], 'TEST')
        value = getattr(state, attr)
        assert value is not None, (
            f"'{attr}' was initialized to None instead of {expected_type.__name__}"
        )
        assert isinstance(value, expected_type), (
            f"'{attr}' was initialized to {type(value).__name__}, "
            f"expected {expected_type.__name__}"
        )

    @pytest.mark.fast
    def test_multiple_missing_attrs_all_initialized(self):
        """When multiple attrs are missing, all get correct defaults."""
        state = SimpleNamespace()
        attrs = ['video_ids', 'downloaded_segments', 'output_files', 'caption_results']
        validate_required_state_attrs(state, attrs, 'TEST')

        assert state.video_ids == []
        assert state.downloaded_segments == []
        assert state.output_files == []
        assert state.caption_results == {}

    @pytest.mark.fast
    def test_existing_attr_not_overwritten(self):
        """Attributes that already exist on state are not overwritten."""
        state = SimpleNamespace(video_ids=['abc123', 'def456'])
        validate_required_state_attrs(state, ['video_ids'], 'TEST')
        assert state.video_ids == ['abc123', 'def456']
