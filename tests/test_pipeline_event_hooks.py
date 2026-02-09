"""
Tests for pipeline event hooks system (US-81-012).

Verifies:
- PipelineEvent dataclass fields
- register_hook / emit_event methods
- Multiple callbacks per event type called in registration order
- before_stage, after_stage, on_stage_error, on_pipeline_complete events emitted
- Existing on_stage_start / on_stage_complete callbacks still work (backward compat)
- Hook exceptions don't break pipeline execution
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.pipeline import PipelineOrchestrator, PipelineEvent
from src.state import PipelineState
from src.stages import Stage, StageResult, StageMetrics


class MockStage(Stage):
    """Minimal mock stage for event hook tests."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self._should_fail = should_fail

    def can_skip(self, state, checkpoint):
        return False

    def restore(self, state, checkpoint, config=None):
        return True

    def validate_inputs(self, state, config):
        return None

    def run(self, state, config, checkpoint):
        if self._should_fail:
            return StageResult.fail(f"{self.name} failed")
        return StageResult.ok(data={'stage': self.name})


@pytest.fixture
def pipeline_dir(tmp_path):
    return tmp_path


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.cache.cache_dir = '/tmp/test_cache'
    config.freeze = Mock()
    config._config_hash = 'testhash'
    return config


@pytest.fixture
def pipeline(pipeline_dir, mock_config):
    with patch.object(PipelineOrchestrator, '_validate_runtime_environment', return_value=[]):
        p = PipelineOrchestrator(mock_config, pipeline_dir)
    return p


class TestPipelineEvent:
    """Test PipelineEvent dataclass."""

    def test_event_fields(self):
        event = PipelineEvent(
            event_type='before_stage',
            stage_name='ANALYZE',
            timestamp=1000.0,
            data={'key': 'value'},
            error=None,
        )
        assert event.event_type == 'before_stage'
        assert event.stage_name == 'ANALYZE'
        assert event.timestamp == 1000.0
        assert event.data == {'key': 'value'}
        assert event.error is None

    def test_event_defaults(self):
        event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=2000.0,
        )
        assert event.data == {}
        assert event.error is None

    def test_event_with_error(self):
        event = PipelineEvent(
            event_type='on_stage_error',
            stage_name='DOWNLOAD_SEGMENTS',
            timestamp=3000.0,
            error='Download failed',
        )
        assert event.error == 'Download failed'


class TestRegisterAndEmitHooks:
    """Test register_hook and emit_event methods."""

    def test_register_single_hook(self, pipeline):
        received = []
        pipeline.register_hook('before_stage', lambda e: received.append(e))

        event = PipelineEvent(
            event_type='before_stage',
            stage_name='ANALYZE',
            timestamp=time.time(),
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0] is event

    def test_register_multiple_hooks_same_event(self, pipeline):
        """AC: Multiple callbacks can be registered per event type
        and are called in registration order."""
        call_order = []

        pipeline.register_hook('after_stage', lambda e: call_order.append('first'))
        pipeline.register_hook('after_stage', lambda e: call_order.append('second'))
        pipeline.register_hook('after_stage', lambda e: call_order.append('third'))

        event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=time.time(),
        )
        pipeline.emit_event(event)

        assert call_order == ['first', 'second', 'third']

    def test_hooks_different_event_types_independent(self, pipeline):
        before_calls = []
        after_calls = []

        pipeline.register_hook('before_stage', lambda e: before_calls.append(e.stage_name))
        pipeline.register_hook('after_stage', lambda e: after_calls.append(e.stage_name))

        pipeline.emit_event(PipelineEvent(
            event_type='before_stage', stage_name='A', timestamp=0,
        ))
        pipeline.emit_event(PipelineEvent(
            event_type='after_stage', stage_name='B', timestamp=0,
        ))

        assert before_calls == ['A']
        assert after_calls == ['B']

    def test_hook_exception_does_not_break_pipeline(self, pipeline):
        received = []

        def bad_hook(e):
            raise RuntimeError("Hook crashed")

        pipeline.register_hook('before_stage', bad_hook)
        pipeline.register_hook('before_stage', lambda e: received.append(e.stage_name))

        event = PipelineEvent(
            event_type='before_stage', stage_name='ANALYZE', timestamp=0,
        )
        pipeline.emit_event(event)  # Should not raise

        # Second hook still called despite first crashing
        assert received == ['ANALYZE']

    def test_emit_unknown_event_type_no_error(self, pipeline):
        """Emitting an event with no registered hooks should be a no-op."""
        event = PipelineEvent(
            event_type='nonexistent', stage_name='X', timestamp=0,
        )
        pipeline.emit_event(event)  # Should not raise


class TestEventsDuringPipelineRun:
    """Test that events are properly emitted during pipeline.run()."""

    def test_before_and_after_stage_events(self, pipeline, mock_config):
        """AC: 2 callbacks for after_stage both called with correct PipelineEvent data."""
        stage = MockStage('TEST_STAGE')
        pipeline.add_stage(stage)

        events_received = []
        pipeline.register_hook('before_stage', lambda e: events_received.append(e))
        pipeline.register_hook('after_stage', lambda e: events_received.append(e))

        success = pipeline.run(resume=False)
        assert success is True

        before_events = [e for e in events_received if e.event_type == 'before_stage']
        after_events = [e for e in events_received if e.event_type == 'after_stage']

        assert len(before_events) == 1
        assert before_events[0].stage_name == 'TEST_STAGE'

        assert len(after_events) == 1
        assert after_events[0].stage_name == 'TEST_STAGE'
        assert after_events[0].data.get('elapsed') is not None
        assert after_events[0].data.get('success') is True

    def test_two_callbacks_for_after_stage_both_called(self, pipeline, mock_config):
        """AC: Registering 2 callbacks for after_stage results in both being called
        with correct PipelineEvent data."""
        stage = MockStage('TEST_STAGE')
        pipeline.add_stage(stage)

        callback1_events = []
        callback2_events = []
        pipeline.register_hook('after_stage', lambda e: callback1_events.append(e))
        pipeline.register_hook('after_stage', lambda e: callback2_events.append(e))

        success = pipeline.run(resume=False)
        assert success is True

        assert len(callback1_events) == 1
        assert len(callback2_events) == 1
        # Both received the same event data
        assert callback1_events[0].event_type == 'after_stage'
        assert callback1_events[0].stage_name == 'TEST_STAGE'
        assert callback2_events[0].event_type == 'after_stage'
        assert callback2_events[0].stage_name == 'TEST_STAGE'
        assert callback1_events[0].data.get('elapsed') is not None

    def test_on_stage_error_event(self, pipeline, mock_config):
        """AC: on_stage_error event emitted when a stage fails."""
        stage = MockStage('FAIL_STAGE', should_fail=True)
        pipeline.add_stage(stage)

        error_events = []
        pipeline.register_hook('on_stage_error', lambda e: error_events.append(e))

        success = pipeline.run(resume=False)
        assert success is False

        assert len(error_events) == 1
        assert error_events[0].event_type == 'on_stage_error'
        assert error_events[0].stage_name == 'FAIL_STAGE'
        assert error_events[0].error == 'FAIL_STAGE failed'

    def test_on_pipeline_complete_event(self, pipeline, mock_config):
        """AC: on_pipeline_complete event emitted after successful pipeline run."""
        stage = MockStage('ONLY_STAGE')
        pipeline.add_stage(stage)

        complete_events = []
        pipeline.register_hook('on_pipeline_complete', lambda e: complete_events.append(e))

        success = pipeline.run(resume=False)
        assert success is True

        assert len(complete_events) == 1
        assert complete_events[0].event_type == 'on_pipeline_complete'
        assert complete_events[0].stage_name == ''
        assert 'total_duration' in complete_events[0].data
        assert 'stages_run' in complete_events[0].data

    def test_legacy_callbacks_still_work(self, pipeline, mock_config):
        """AC: Existing on_stage_start and on_stage_complete callbacks still work."""
        stage = MockStage('TEST_STAGE')
        pipeline.add_stage(stage)

        start_calls = []
        complete_calls = []

        success = pipeline.run(
            resume=False,
            on_stage_start=lambda name: start_calls.append(name),
            on_stage_complete=lambda name, result, elapsed: complete_calls.append(name),
        )
        assert success is True
        assert start_calls == ['TEST_STAGE']
        assert complete_calls == ['TEST_STAGE']

    def test_event_hooks_and_legacy_callbacks_both_fire(self, pipeline, mock_config):
        """Both new event hooks and legacy callbacks fire together."""
        stage = MockStage('TEST_STAGE')
        pipeline.add_stage(stage)

        hook_events = []
        legacy_starts = []

        pipeline.register_hook('before_stage', lambda e: hook_events.append(e))

        success = pipeline.run(
            resume=False,
            on_stage_start=lambda name: legacy_starts.append(name),
        )
        assert success is True
        assert len(hook_events) == 1
        assert legacy_starts == ['TEST_STAGE']

    def test_multiple_stages_emit_events_in_order(self, pipeline, mock_config):
        """Events emitted for each stage in correct order."""
        pipeline.add_stage(MockStage('STAGE_A'))
        pipeline.add_stage(MockStage('STAGE_B'))

        all_events = []
        pipeline.register_hook('before_stage', lambda e: all_events.append(('before', e.stage_name)))
        pipeline.register_hook('after_stage', lambda e: all_events.append(('after', e.stage_name)))

        success = pipeline.run(resume=False)
        assert success is True

        assert all_events == [
            ('before', 'STAGE_A'),
            ('after', 'STAGE_A'),
            ('before', 'STAGE_B'),
            ('after', 'STAGE_B'),
        ]

    def test_no_pipeline_complete_on_failure(self, pipeline, mock_config):
        """on_pipeline_complete should NOT fire if pipeline fails."""
        pipeline.add_stage(MockStage('FAIL_STAGE', should_fail=True))

        complete_events = []
        pipeline.register_hook('on_pipeline_complete', lambda e: complete_events.append(e))

        success = pipeline.run(resume=False)
        assert success is False
        assert len(complete_events) == 0
