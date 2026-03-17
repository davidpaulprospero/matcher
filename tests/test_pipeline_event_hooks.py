"""
Tests for pipeline event hooks system (US-81-012) and PipelineEventBus (US-82-011).

Verifies:
- PipelineEvent dataclass fields
- PipelineEventBus: subscribe, emit, get_handlers, ordering, handler error isolation
- register_hook / emit_event methods on PipelineOrchestrator (delegates to event bus)
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
from src.pipeline_events import (
    PipelineEventBus,
    PipelineEvent,
    EVENT_STAGE_START,
    EVENT_STAGE_SKIP,
    EVENT_CHECKPOINT_SAVE,
    EVENT_RESOURCE_WARNING,
    EVENT_STAGE_CHECKPOINT_WRITE,
    EVENT_RESOURCE_THRESHOLD,
    EVENT_ERROR_RECOVERED,
    EVENT_STAGE_TRACE_START,
    EVENT_STAGE_TRACE_END,
    EventTracer,
    EventReplay,
    VerboseTraceHandler,
    TraceAnalyzer,
    get_memory_usage,
    extract_checkpoint_summary,
)
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


class TestPipelineEventBus:
    """Standalone tests for PipelineEventBus (US-82-011)."""

    def test_subscribe_and_emit(self):
        """AC: subscribe() registers handler, emit() calls it."""
        bus = PipelineEventBus()
        received = []
        bus.subscribe('before_stage', lambda e: received.append(e))

        event = PipelineEvent(
            event_type='before_stage', stage_name='ANALYZE', timestamp=1.0,
        )
        bus.emit(event)

        assert len(received) == 1
        assert received[0] is event

    def test_get_handlers_returns_registered(self):
        """AC: get_handlers() returns list of registered handlers."""
        bus = PipelineEventBus()
        h1 = lambda e: None
        h2 = lambda e: None

        bus.subscribe('after_stage', h1)
        bus.subscribe('after_stage', h2)

        handlers = bus.get_handlers('after_stage')
        assert len(handlers) == 2
        assert handlers[0] is h1
        assert handlers[1] is h2

    def test_get_handlers_empty_for_unregistered(self):
        """get_handlers() returns empty list for unregistered event type."""
        bus = PipelineEventBus()
        assert bus.get_handlers('nonexistent') == []

    def test_handlers_called_in_registration_order(self):
        """AC: Handlers called in order of registration."""
        bus = PipelineEventBus()
        call_order = []

        bus.subscribe('before_stage', lambda e: call_order.append('first'))
        bus.subscribe('before_stage', lambda e: call_order.append('second'))
        bus.subscribe('before_stage', lambda e: call_order.append('third'))

        bus.emit(PipelineEvent(
            event_type='before_stage', stage_name='X', timestamp=0,
        ))

        assert call_order == ['first', 'second', 'third']

    def test_handler_error_isolation(self):
        """AC: One failing handler doesn't block others."""
        bus = PipelineEventBus()
        received = []

        def bad_handler(e):
            raise RuntimeError("boom")

        bus.subscribe('after_stage', bad_handler)
        bus.subscribe('after_stage', lambda e: received.append(e.stage_name))
        bus.subscribe('after_stage', lambda e: received.append('third'))

        bus.emit(PipelineEvent(
            event_type='after_stage', stage_name='MATCH', timestamp=0,
        ))

        # Both subsequent handlers ran despite first crashing
        assert received == ['MATCH', 'third']

    def test_emit_no_handlers_is_noop(self):
        """Emitting with no handlers for that type is a no-op."""
        bus = PipelineEventBus()
        bus.emit(PipelineEvent(
            event_type='unknown', stage_name='X', timestamp=0,
        ))  # Should not raise

    def test_different_event_types_independent(self):
        """Handlers for different event types don't interfere."""
        bus = PipelineEventBus()
        before = []
        after = []

        bus.subscribe('before_stage', lambda e: before.append(e.stage_name))
        bus.subscribe('after_stage', lambda e: after.append(e.stage_name))

        bus.emit(PipelineEvent(event_type='before_stage', stage_name='A', timestamp=0))
        bus.emit(PipelineEvent(event_type='after_stage', stage_name='B', timestamp=0))

        assert before == ['A']
        assert after == ['B']

    def test_event_type_routing(self):
        """PipelineEvent.event_type field correctly routes to handlers."""
        bus = PipelineEventBus()
        error_events = []
        complete_events = []

        bus.subscribe('on_stage_error', lambda e: error_events.append(e))
        bus.subscribe('on_pipeline_complete', lambda e: complete_events.append(e))

        bus.emit(PipelineEvent(
            event_type='on_stage_error', stage_name='FAIL', timestamp=0, error='oops',
        ))
        bus.emit(PipelineEvent(
            event_type='on_pipeline_complete', stage_name='', timestamp=0,
        ))

        assert len(error_events) == 1
        assert error_events[0].error == 'oops'
        assert len(complete_events) == 1

    def test_get_handlers_returns_copy(self):
        """get_handlers() returns a copy, not a mutable reference."""
        bus = PipelineEventBus()
        bus.subscribe('before_stage', lambda e: None)

        handlers = bus.get_handlers('before_stage')
        handlers.clear()  # Mutate the returned list

        # Original handlers should be unaffected
        assert len(bus.get_handlers('before_stage')) == 1


class TestOrchestratorEventBusDelegation:
    """Test that PipelineOrchestrator delegates to PipelineEventBus (US-82-011)."""

    def test_orchestrator_has_event_bus(self, pipeline):
        """AC: PipelineOrchestrator uses PipelineEventBus."""
        assert hasattr(pipeline, 'event_bus')
        assert isinstance(pipeline.event_bus, PipelineEventBus)

    def test_register_hook_delegates_to_bus(self, pipeline):
        """register_hook() adds handler to the event bus."""
        handler = lambda e: None
        pipeline.register_hook('before_stage', handler)

        bus_handlers = pipeline.event_bus.get_handlers('before_stage')
        assert len(bus_handlers) == 1
        assert bus_handlers[0] is handler

    def test_emit_event_delegates_to_bus(self, pipeline):
        """emit_event() routes through the event bus."""
        received = []
        pipeline.event_bus.subscribe('after_stage', lambda e: received.append(e))

        event = PipelineEvent(
            event_type='after_stage', stage_name='TEST', timestamp=0,
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0] is event


class TestNewEventTypes:
    """Test new event types added in US-106-006."""

    def test_stage_start_event_constants_defined(self):
        """AC: STAGE_START event constant is defined."""
        assert EVENT_STAGE_START == 'stage_start'

    def test_stage_skip_event_constant_defined(self):
        """AC: STAGE_SKIP event constant is defined."""
        assert EVENT_STAGE_SKIP == 'stage_skip'

    def test_checkpoint_save_event_constant_defined(self):
        """AC: CHECKPOINT_SAVE event constant is defined."""
        assert EVENT_CHECKPOINT_SAVE == 'checkpoint_save'

    def test_resource_warning_event_constant_defined(self):
        """AC: RESOURCE_WARNING event constant is defined."""
        assert EVENT_RESOURCE_WARNING == 'resource_warning'

    def test_stage_start_event_with_estimated_duration(self, pipeline):
        """AC: STAGE_START event includes estimated duration."""
        received = []
        pipeline.register_hook(EVENT_STAGE_START, lambda e: received.append(e))

        # Emit a stage_start event manually to test the data structure
        event = PipelineEvent(
            event_type=EVENT_STAGE_START,
            stage_name='MATCH',
            timestamp=1000.0,
            data={'estimated_duration': 120.5},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'stage_start'
        assert received[0].stage_name == 'MATCH'
        assert received[0].data.get('estimated_duration') == 120.5

    def test_stage_skip_event_with_reason(self, pipeline):
        """AC: STAGE_SKIP event includes skip reason."""
        received = []
        pipeline.register_hook(EVENT_STAGE_SKIP, lambda e: received.append(e))

        event = PipelineEvent(
            event_type=EVENT_STAGE_SKIP,
            stage_name='VIDEO_SEARCH',
            timestamp=1000.0,
            data={'reason': 'checkpoint_resume'},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'stage_skip'
        assert received[0].data.get('reason') == 'checkpoint_resume'

    def test_checkpoint_save_event_with_path(self, pipeline):
        """AC: CHECKPOINT_SAVE event includes checkpoint path."""
        received = []
        pipeline.register_hook(EVENT_CHECKPOINT_SAVE, lambda e: received.append(e))

        event = PipelineEvent(
            event_type=EVENT_CHECKPOINT_SAVE,
            stage_name='ANALYZE',
            timestamp=1000.0,
            data={'checkpoint_path': '/path/to/checkpoint.json', 'partial': False},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'checkpoint_save'
        assert received[0].data.get('checkpoint_path') == '/path/to/checkpoint.json'
        assert received[0].data.get('partial') is False

    def test_resource_warning_event_with_thresholds(self, pipeline):
        """AC: RESOURCE_WARNING event includes resource metrics."""
        received = []
        pipeline.register_hook(EVENT_RESOURCE_WARNING, lambda e: received.append(e))

        event = PipelineEvent(
            event_type=EVENT_RESOURCE_WARNING,
            stage_name='DOWNLOAD_SEGMENTS',
            timestamp=1000.0,
            data={'resource_type': 'cpu', 'current_value': 95.5, 'threshold': 90.0},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'resource_warning'
        assert received[0].data.get('resource_type') == 'cpu'
        assert received[0].data.get('current_value') == 95.5
        assert received[0].data.get('threshold') == 90.0

    def test_after_stage_includes_stage_duration(self, pipeline):
        """AC: after_stage event includes stage_duration field."""
        received = []
        pipeline.register_hook('after_stage', lambda e: received.append(e))

        event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=1000.0,
            data={'elapsed': 45.2, 'stage_duration': 45.2, 'success': True},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert 'stage_duration' in received[0].data
        assert received[0].data['stage_duration'] == 45.2

    def test_all_new_events_subscribable(self, pipeline):
        """AC: All new event types can be subscribed to."""
        events_received = {EVENT_STAGE_START: [], EVENT_STAGE_SKIP: [],
                          EVENT_CHECKPOINT_SAVE: [], EVENT_RESOURCE_WARNING: []}

        pipeline.register_hook(EVENT_STAGE_START, lambda e: events_received[EVENT_STAGE_START].append(e))
        pipeline.register_hook(EVENT_STAGE_SKIP, lambda e: events_received[EVENT_STAGE_SKIP].append(e))
        pipeline.register_hook(EVENT_CHECKPOINT_SAVE, lambda e: events_received[EVENT_CHECKPOINT_SAVE].append(e))
        pipeline.register_hook(EVENT_RESOURCE_WARNING, lambda e: events_received[EVENT_RESOURCE_WARNING].append(e))

        # Emit each event type
        pipeline.emit_event(PipelineEvent(event_type=EVENT_STAGE_START, stage_name='A', timestamp=0))
        pipeline.emit_event(PipelineEvent(event_type=EVENT_STAGE_SKIP, stage_name='B', timestamp=0))
        pipeline.emit_event(PipelineEvent(event_type=EVENT_CHECKPOINT_SAVE, stage_name='C', timestamp=0))
        pipeline.emit_event(PipelineEvent(event_type=EVENT_RESOURCE_WARNING, stage_name='D', timestamp=0))

        assert len(events_received[EVENT_STAGE_START]) == 1
        assert len(events_received[EVENT_STAGE_SKIP]) == 1
        assert len(events_received[EVENT_CHECKPOINT_SAVE]) == 1
        assert len(events_received[EVENT_RESOURCE_WARNING]) == 1


class TestNewEventTypesUS108007:
    """Test new event types added in US-108-007 (Event Tracing)."""

    def test_stage_checkpoint_write_event_constant_defined(self):
        """AC: EVENT_STAGE_CHECKPOINT_WRITE event constant is defined."""
        assert EVENT_STAGE_CHECKPOINT_WRITE == 'stage_checkpoint_write'

    def test_resource_threshold_event_constant_defined(self):
        """AC: EVENT_RESOURCE_THRESHOLD event constant is defined."""
        assert EVENT_RESOURCE_THRESHOLD == 'resource_threshold'

    def test_error_recovered_event_constant_defined(self):
        """AC: EVENT_ERROR_RECOVERED event constant is defined."""
        assert EVENT_ERROR_RECOVERED == 'error_recovered'

    def test_stage_checkpoint_write_event_with_path(self, pipeline):
        """AC: STAGE_CHECKPOINT_WRITE event includes checkpoint path."""
        received = []
        pipeline.register_hook(EVENT_STAGE_CHECKPOINT_WRITE, lambda e: received.append(e))

        event = PipelineEvent(
            event_type=EVENT_STAGE_CHECKPOINT_WRITE,
            stage_name='MATCH',
            timestamp=1000.0,
            data={'checkpoint_path': '/path/to/checkpoint.json', 'stage_data_keys': ['matches']},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'stage_checkpoint_write'
        assert received[0].data.get('checkpoint_path') == '/path/to/checkpoint.json'

    def test_resource_threshold_event_with_metrics(self, pipeline):
        """AC: RESOURCE_THRESHOLD event includes resource metrics."""
        received = []
        pipeline.register_hook(EVENT_RESOURCE_THRESHOLD, lambda e: received.append(e))

        event = PipelineEvent(
            event_type=EVENT_RESOURCE_THRESHOLD,
            stage_name='DOWNLOAD_SEGMENTS',
            timestamp=1000.0,
            data={'resource_type': 'memory', 'current_value': 85.0, 'threshold': 80.0, 'percentage': 106.25},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'resource_threshold'
        assert received[0].data.get('resource_type') == 'memory'
        assert received[0].data.get('current_value') == 85.0
        assert received[0].data.get('threshold') == 80.0

    def test_error_recovered_event_with_details(self, pipeline):
        """AC: ERROR_RECOVERED event includes recovery details."""
        received = []
        pipeline.register_hook(EVENT_ERROR_RECOVERED, lambda e: received.append(e))

        event = PipelineEvent(
            event_type=EVENT_ERROR_RECOVERED,
            stage_name='VIDEO_SEARCH',
            timestamp=1000.0,
            data={'original_error': 'Rate limit exceeded', 'recovery_strategy': 'backoff', 'retry_count': 1},
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'error_recovered'
        assert received[0].data.get('original_error') == 'Rate limit exceeded'
        assert received[0].data.get('recovery_strategy') == 'backoff'

    def test_all_new_events_subscribable(self, pipeline):
        """AC: All new event types can be subscribed to."""
        events_received = {
            EVENT_STAGE_CHECKPOINT_WRITE: [],
            EVENT_RESOURCE_THRESHOLD: [],
            EVENT_ERROR_RECOVERED: [],
        }

        pipeline.register_hook(EVENT_STAGE_CHECKPOINT_WRITE, lambda e: events_received[EVENT_STAGE_CHECKPOINT_WRITE].append(e))
        pipeline.register_hook(EVENT_RESOURCE_THRESHOLD, lambda e: events_received[EVENT_RESOURCE_THRESHOLD].append(e))
        pipeline.register_hook(EVENT_ERROR_RECOVERED, lambda e: events_received[EVENT_ERROR_RECOVERED].append(e))

        # Emit each event type
        pipeline.emit_event(PipelineEvent(event_type=EVENT_STAGE_CHECKPOINT_WRITE, stage_name='A', timestamp=0))
        pipeline.emit_event(PipelineEvent(event_type=EVENT_RESOURCE_THRESHOLD, stage_name='B', timestamp=0))
        pipeline.emit_event(PipelineEvent(event_type=EVENT_ERROR_RECOVERED, stage_name='C', timestamp=0))

        assert len(events_received[EVENT_STAGE_CHECKPOINT_WRITE]) == 1
        assert len(events_received[EVENT_RESOURCE_THRESHOLD]) == 1
        assert len(events_received[EVENT_ERROR_RECOVERED]) == 1


class TestPipelineEventToTraceDict:
    """Test PipelineEvent.to_trace_dict() method."""

    def test_to_trace_dict_includes_timestamp_ms(self):
        """AC: to_trace_dict includes timestamp_ms field."""
        event = PipelineEvent(
            event_type='before_stage',
            stage_name='ANALYZE',
            timestamp=1000.5,
        )
        trace = event.to_trace_dict()
        assert 'timestamp_ms' in trace
        assert trace['timestamp_ms'] == 1000500

    def test_to_trace_dict_includes_stage_name(self):
        """AC: to_trace_dict includes stage_name field."""
        event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=1000.0,
        )
        trace = event.to_trace_dict()
        assert trace['stage_name'] == 'MATCH'

    def test_to_trace_dict_includes_duration_ms(self):
        """AC: to_trace_dict includes duration_ms field from elapsed."""
        event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=1000.0,
            data={'elapsed': 45.5},
        )
        trace = event.to_trace_dict()
        assert 'duration_ms' in trace
        assert trace['duration_ms'] == 45500

    def test_to_trace_dict_includes_metadata_dict(self):
        """AC: to_trace_dict includes metadata_dict field."""
        event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=1000.0,
            data={'elapsed': 10.0, 'matches_found': 42},
        )
        trace = event.to_trace_dict()
        assert 'metadata_dict' in trace
        assert trace['metadata_dict']['elapsed'] == 10.0
        assert trace['metadata_dict']['matches_found'] == 42


class TestEventTracer:
    """Test EventTracer class for JSON-lines event logging."""

    def test_tracer_start_opens_file(self, tmp_path):
        """AC: start() opens the trace file for writing."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        tracer = EventTracer(str(trace_file))
        tracer.start()

        assert tracer._enabled is True
        assert tracer._file is not None
        assert trace_file.exists()

        tracer.stop()

    def test_tracer_stop_closes_file(self, tmp_path):
        """AC: stop() closes the trace file."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        tracer = EventTracer(str(trace_file))
        tracer.start()
        tracer.stop()

        assert tracer._enabled is False
        assert tracer._file is None

    def test_tracer_on_event_writes_json_line(self, tmp_path):
        """AC: on_event() writes event as JSON line to trace file."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        tracer = EventTracer(str(trace_file))
        tracer.start()

        event = PipelineEvent(
            event_type='before_stage',
            stage_name='ANALYZE',
            timestamp=1000.0,
            data={'elapsed': 0.0},
        )
        tracer.on_event(event)
        tracer.stop()

        # Verify the file contains the event
        content = trace_file.read_text()
        assert 'timestamp_ms' in content
        assert 'ANALYZE' in content

    def test_tracer_disabled_does_not_write(self, tmp_path):
        """AC: on_event() does nothing when tracer is not enabled."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        tracer = EventTracer(str(trace_file))
        # Don't start tracer

        event = PipelineEvent(
            event_type='before_stage',
            stage_name='ANALYZE',
            timestamp=1000.0,
        )
        tracer.on_event(event)

        # File should not exist or be empty
        assert not trace_file.exists() or trace_file.read_text() == ''


class TestEventReplay:
    """Test EventReplay utility for analyzing trace files."""

    def test_load_timeline_returns_empty_for_missing_file(self, tmp_path):
        """AC: load_timeline returns empty list for non-existent file."""
        replay = EventReplay(str(tmp_path / 'nonexistent.jsonl'))
        timeline = replay.load_timeline()
        assert timeline == []

    def test_load_timeline_parses_json_lines(self, tmp_path):
        """AC: load_timeline parses events from JSON-lines file."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "A", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 2000, "stage_name": "B", "event_type": "before_stage"}\n'
        )

        replay = EventReplay(str(trace_file))
        timeline = replay.load_timeline()

        assert len(timeline) == 2
        assert timeline[0]['stage_name'] == 'A'
        assert timeline[1]['stage_name'] == 'B'

    def test_analyze_identifies_longest_stages(self, tmp_path):
        """AC: analyze() identifies longest stages by total duration."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "MATCH", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 3000, "stage_name": "MATCH", "event_type": "after_stage"}\n'
            '{"timestamp_ms": 4000, "stage_name": "ANALYZE", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 4500, "stage_name": "ANALYZE", "event_type": "after_stage"}\n'
        )

        replay = EventReplay(str(trace_file))
        analysis = replay.analyze()

        assert len(analysis['longest_stages']) == 2
        # MATCH took 2000ms, ANALYZE took 500ms
        assert analysis['longest_stages'][0]['stage'] == 'MATCH'
        assert analysis['longest_stages'][0]['total_duration_ms'] == 2000

    def test_analyze_identifies_errors(self, tmp_path):
        """AC: analyze() identifies most common errors."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "A", "event_type": "on_stage_error", "error": "Rate limit"}\n'
            '{"timestamp_ms": 2000, "stage_name": "B", "event_type": "on_stage_error", "error": "Rate limit"}\n'
            '{"timestamp_ms": 3000, "stage_name": "C", "event_type": "on_stage_error", "error": "Timeout"}\n'
        )

        replay = EventReplay(str(trace_file))
        analysis = replay.analyze()

        assert len(analysis['most_common_errors']) == 2
        assert analysis['most_common_errors'][0]['error'] == 'Rate limit'
        assert analysis['most_common_errors'][0]['count'] == 2

    def test_analyze_identifies_retry_patterns(self, tmp_path):
        """AC: analyze() identifies retry patterns from error_recovered events."""
        trace_file = tmp_path / 'pipeline_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "SEARCH", "event_type": "error_recovered"}\n'
            '{"timestamp_ms": 2000, "stage_name": "SEARCH", "event_type": "error_recovered"}\n'
        )

        replay = EventReplay(str(trace_file))
        analysis = replay.analyze()

        assert analysis['retry_patterns'][0]['error_recovered_count'] == 2


class TestTraceEventTypesUS120004:
    """Test new trace event types added in US-120-004 (Detailed Pipeline Stage Tracing)."""

    def test_stage_trace_start_event_constant_defined(self):
        """AC: EVENT_STAGE_TRACE_START event constant is defined."""
        from src.pipeline_events import EVENT_STAGE_TRACE_START
        assert EVENT_STAGE_TRACE_START == 'stage_trace_start'

    def test_stage_trace_end_event_constant_defined(self):
        """AC: EVENT_STAGE_TRACE_END event constant is defined."""
        from src.pipeline_events import EVENT_STAGE_TRACE_END
        assert EVENT_STAGE_TRACE_END == 'stage_trace_end'

    def test_stage_trace_start_event_with_context(self, pipeline):
        """AC: STAGE_TRACE_START event includes context data."""
        received = []
        pipeline.register_hook('stage_trace_start', lambda e: received.append(e))

        event = PipelineEvent(
            event_type='stage_trace_start',
            stage_name='MATCH',
            timestamp=1000.0,
            data={
                'checkpoint_state': {'last_completed_stage': 'VIDEO_SEARCH'},
                'memory_rss_mb': 512.5,
                'match_count': 10,
            },
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'stage_trace_start'
        assert received[0].data.get('checkpoint_state') == {'last_completed_stage': 'VIDEO_SEARCH'}
        assert received[0].data.get('memory_rss_mb') == 512.5
        assert received[0].data.get('match_count') == 10

    def test_stage_trace_end_event_with_context(self, pipeline):
        """AC: STAGE_TRACE_END event includes duration and final context."""
        received = []
        pipeline.register_hook('stage_trace_end', lambda e: received.append(e))

        event = PipelineEvent(
            event_type='stage_trace_end',
            stage_name='MATCH',
            timestamp=1200.0,
            data={
                'duration_ms': 200000,
                'memory_rss_mb': 768.0,
                'match_count': 25,
                'success': True,
            },
        )
        pipeline.emit_event(event)

        assert len(received) == 1
        assert received[0].event_type == 'stage_trace_end'
        assert received[0].data.get('duration_ms') == 200000
        assert received[0].data.get('match_count') == 25


class TestGetMemoryUsage:
    """Test get_memory_usage function."""

    def test_get_memory_usage_returns_dict(self):
        """AC: get_memory_usage returns a dictionary with memory metrics."""
        from src.pipeline_events import get_memory_usage
        mem = get_memory_usage()

        assert isinstance(mem, dict)
        assert 'rss_mb' in mem
        assert 'vms_mb' in mem
        assert 'percent_used' in mem

    def test_get_memory_usage_values_are_numeric(self):
        """AC: get_memory_usage returns numeric values."""
        from src.pipeline_events import get_memory_usage
        mem = get_memory_usage()

        assert isinstance(mem['rss_mb'], (int, float))
        assert isinstance(mem['vms_mb'], (int, float))
        assert isinstance(mem['percent_used'], (int, float))


class TestExtractCheckpointSummary:
    """Test extract_checkpoint_summary function."""

    def test_extract_checkpoint_summary_empty(self):
        """AC: extract_checkpoint_summary handles empty checkpoint."""
        from src.pipeline_events import extract_checkpoint_summary

        result = extract_checkpoint_summary({})
        # Empty dict is falsy, so has_checkpoint is False
        assert result['has_checkpoint'] is False

    def test_extract_checkpoint_summary_none(self):
        """AC: extract_checkpoint_summary handles None."""
        from src.pipeline_events import extract_checkpoint_summary

        result = extract_checkpoint_summary(None)
        assert result['has_checkpoint'] is False

    def test_extract_checkpoint_summary_with_data(self):
        """AC: extract_checkpoint_summary extracts key counts."""
        from src.pipeline_events import extract_checkpoint_summary

        checkpoint = {
            'version': '4.0.0',
            'last_completed_stage': 'MATCH',
            'video_search': {'videos': [1, 2, 3]},
            'match': {'matches': [1, 2, 3, 4, 5]},
        }

        result = extract_checkpoint_summary(checkpoint)

        assert result['has_checkpoint'] is True
        assert result['version'] == '4.0.0'
        assert result['last_completed_stage'] == 'MATCH'
        assert result['match_count'] == 5


class TestTraceAnalyzer:
    """Test TraceAnalyzer class."""

    def test_trace_analyzer_init(self):
        """AC: TraceAnalyzer initializes with file path."""
        from src.pipeline_events import TraceAnalyzer

        analyzer = TraceAnalyzer('/tmp/test_trace.jsonl')
        assert analyzer.trace_file_path == '/tmp/test_trace.jsonl'
        assert analyzer._timeline == []

    def test_trace_analyzer_load_empty_file(self):
        """AC: TraceAnalyzer.load returns empty list for non-existent file."""
        from src.pipeline_events import TraceAnalyzer

        analyzer = TraceAnalyzer('/tmp/nonexistent_trace.jsonl')
        timeline = analyzer.load()

        assert timeline == []

    def test_trace_analyzer_load_parses_events(self, tmp_path):
        """AC: TraceAnalyzer.load parses JSON-lines events."""
        from src.pipeline_events import TraceAnalyzer

        trace_file = tmp_path / 'test_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "A", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 2000, "stage_name": "A", "event_type": "after_stage"}\n'
        )

        analyzer = TraceAnalyzer(str(trace_file))
        timeline = analyzer.load()

        assert len(timeline) == 2
        assert timeline[0]['stage_name'] == 'A'

    def test_trace_analyzer_get_stage_durations(self, tmp_path):
        """AC: TraceAnalyzer.get_stage_durations calculates duration stats."""
        from src.pipeline_events import TraceAnalyzer

        trace_file = tmp_path / 'test_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "MATCH", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 3000, "stage_name": "MATCH", "event_type": "after_stage"}\n'
        )

        analyzer = TraceAnalyzer(str(trace_file))
        durations = analyzer.get_stage_durations()

        assert 'MATCH' in durations
        assert durations['MATCH']['count'] == 1
        assert durations['MATCH']['total_ms'] == 2000
        assert durations['MATCH']['avg_ms'] == 2000

    def test_trace_analyzer_generate_report(self, tmp_path):
        """AC: TraceAnalyzer.generate_report returns comprehensive report."""
        from src.pipeline_events import TraceAnalyzer

        trace_file = tmp_path / 'test_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "ANALYZE", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 1500, "stage_name": "ANALYZE", "event_type": "after_stage"}\n'
            '{"timestamp_ms": 2000, "stage_name": "MATCH", "event_type": "before_stage"}\n'
            '{"timestamp_ms": 4000, "stage_name": "MATCH", "event_type": "after_stage"}\n'
        )

        analyzer = TraceAnalyzer(str(trace_file))
        report = analyzer.generate_report()

        assert 'stage_durations' in report
        assert 'total_duration_ms' in report
        assert 'stage_summary' in report
        # Total should be roughly 4000 - 1000 = 3000ms
        assert report['total_duration_ms'] >= 2900

    def test_trace_analyzer_get_resource_timeline(self, tmp_path):
        """AC: TraceAnalyzer.get_resource_timeline extracts memory data."""
        from src.pipeline_events import TraceAnalyzer

        trace_file = tmp_path / 'test_trace.jsonl'
        trace_file.write_text(
            '{"timestamp_ms": 1000, "stage_name": "MATCH", "event_type": "after_stage", '
            '"metadata_dict": {"memory_rss_mb": 512.5, "memory_percent": 25.0}}\n'
        )

        analyzer = TraceAnalyzer(str(trace_file))
        resources = analyzer.get_resource_timeline()

        assert len(resources) == 1
        assert resources[0]['memory_rss_mb'] == 512.5


class TestVerboseTraceHandler:
    """Test VerboseTraceHandler class."""

    def test_verbose_trace_handler_init(self):
        """AC: VerboseTraceHandler initializes with include_memory flag."""
        from src.pipeline_events import VerboseTraceHandler

        handler = VerboseTraceHandler(include_memory=True)
        assert handler.include_memory is True

        handler_no_mem = VerboseTraceHandler(include_memory=False)
        assert handler_no_mem.include_memory is False

    def test_verbose_trace_handler_on_before_stage(self, caplog):
        """AC: VerboseTraceHandler outputs trace on before_stage."""
        from src.pipeline_events import VerboseTraceHandler
        import logging

        handler = VerboseTraceHandler(include_memory=False)

        event = PipelineEvent(
            event_type='before_stage',
            stage_name='MATCH',
            timestamp=1000.0,
        )

        with caplog.at_level(logging.INFO):
            handler.on_trace_event(event)

        assert any('Stage START: MATCH' in record.message for record in caplog.records)

    def test_verbose_trace_handler_on_after_stage(self, caplog):
        """AC: VerboseTraceHandler outputs trace on after_stage."""
        from src.pipeline_events import VerboseTraceHandler
        import logging

        handler = VerboseTraceHandler(include_memory=False)

        # First trigger a before_stage to track start time
        before_event = PipelineEvent(
            event_type='before_stage',
            stage_name='MATCH',
            timestamp=1000.0,
        )
        handler.on_trace_event(before_event)

        # Then after_stage
        after_event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=1050.0,
            data={'elapsed': 50.0, 'success': True, 'match_count': 10},
        )

        with caplog.at_level(logging.INFO):
            handler.on_trace_event(after_event)

        assert any('Stage END: MATCH' in record.message for record in caplog.records)
        assert any('Duration: 50.0' in record.message for record in caplog.records)

    def test_verbose_trace_handler_context_output(self, caplog):
        """AC: VerboseTraceHandler outputs context like match_count."""
        from src.pipeline_events import VerboseTraceHandler
        import logging

        handler = VerboseTraceHandler(include_memory=False)

        # First trigger a before_stage
        before_event = PipelineEvent(
            event_type='before_stage',
            stage_name='MATCH',
            timestamp=1000.0,
        )
        handler.on_trace_event(before_event)

        # Then after_stage with context
        after_event = PipelineEvent(
            event_type='after_stage',
            stage_name='MATCH',
            timestamp=1050.0,
            data={'elapsed': 50.0, 'success': True, 'match_count': 42},
        )

        with caplog.at_level(logging.INFO):
            handler.on_trace_event(after_event)

        assert any('matches=42' in record.message for record in caplog.records)


# US-125-010: Webhook notification tests
class TestWebhookConfig:
    """Test WebhookConfig dataclass."""

    def test_webhook_config_defaults(self):
        """AC: WebhookConfig has sensible defaults."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig()
        assert config.enabled is False
        assert config.default_url == ""
        assert config.timeout_seconds == 10.0
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 30.0
        assert config.include_full_data is True
        assert config.include_errors is True

    def test_webhook_config_validation_timeout(self):
        """AC: WebhookConfig validates timeout >= 0.1."""
        from src.config.sections.infrastructure import WebhookConfig

        with pytest.raises(ValueError, match="timeout_seconds must be >= 0.1"):
            WebhookConfig(timeout_seconds=0.0)

    def test_webhook_config_validation_max_retries(self):
        """AC: WebhookConfig validates max_retries >= 0."""
        from src.config.sections.infrastructure import WebhookConfig

        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            WebhookConfig(max_retries=-1)

    def test_webhook_config_validation_delay_order(self):
        """AC: WebhookConfig validates retry_max_delay >= retry_base_delay."""
        from src.config.sections.infrastructure import WebhookConfig

        with pytest.raises(ValueError, match="retry_max_delay.*must be >= retry_base_delay"):
            WebhookConfig(retry_base_delay=5.0, retry_max_delay=2.0)

    def test_webhook_config_event_urls(self):
        """AC: WebhookConfig supports event-specific URLs."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            default_url="https://default.example.com/webhook",
            event_urls={
                "after_stage": "https://example.com/after_stage",
                "on_pipeline_complete": "https://example.com/complete",
            },
        )
        assert config.get_url_for_event("after_stage") == "https://example.com/after_stage"
        assert config.get_url_for_event("on_pipeline_complete") == "https://example.com/complete"
        # Falls back to default for unknown events
        assert config.get_url_for_event("unknown") == "https://default.example.com/webhook"

    def test_webhook_config_get_url_no_default(self):
        """AC: get_url_for_event returns None when no URLs configured."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(enabled=True)
        assert config.get_url_for_event("after_stage") is None

    def test_webhook_config_headers(self):
        """AC: WebhookConfig supports custom headers."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            headers={"Authorization": "Bearer token123", "X-Custom": "value"},
        )
        assert config.headers["Authorization"] == "Bearer token123"
        assert config.headers["X-Custom"] == "value"


class TestWebhookSender:
    """Test WebhookSender class."""

    def test_webhook_sender_init(self):
        """AC: WebhookSender initializes with config."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender

        config = WebhookConfig(enabled=True, default_url="https://example.com/webhook")
        sender = WebhookSender(config)

        assert sender.config is config
        assert sender._retry_counts == {}

    def test_webhook_sender_on_event_no_url(self):
        """AC: WebhookSender skips events with no URL configured."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(enabled=True)  # No URL configured
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
            data={"success": True},
        )
        # Should not raise
        sender.on_event(event)

    def test_webhook_sender_on_event_disabled(self):
        """AC: WebhookSender skips events when disabled."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(enabled=False)
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
        )
        # Should not raise
        sender.on_event(event)

    def test_webhook_sender_build_payload_full_data(self):
        """AC: WebhookSender builds correct payload with full data."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            include_full_data=True,
            include_errors=True,
        )
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
            data={"success": True, "match_count": 10, "elapsed": 45.5},
            error=None,
        )

        payload = sender._build_payload(event)

        assert payload["event_type"] == "after_stage"
        assert payload["stage_name"] == "MATCH"
        assert payload["timestamp_unix"] == 1000.0
        assert "timestamp" in payload  # ISO format
        assert payload["data"]["success"] is True
        assert payload["data"]["match_count"] == 10

    def test_webhook_sender_build_payload_minimal(self):
        """AC: WebhookSender builds minimal payload when configured."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            include_full_data=False,
            include_errors=True,
        )
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
            data={"success": True, "match_count": 10, "elapsed": 45.5, "extra_data": "ignored"},
            error="Some error",
        )

        payload = sender._build_payload(event)

        # Should only include key fields when include_full_data=False
        assert "data" not in payload
        assert payload.get("success") is True
        assert payload.get("match_count") == 10
        # Error should still be included
        assert payload.get("error") == "Some error"

    def test_webhook_sender_build_payload_no_error(self):
        """AC: WebhookSender excludes error field when not present."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            include_errors=True,
        )
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
            error=None,
        )

        payload = sender._build_payload(event)

        # Error should not be in payload when None
        assert "error" not in payload

    @patch('src.pipeline_events.urllib.request.urlopen')
    def test_webhook_sender_sends_success(self, mock_urlopen):
        """AC: WebhookSender successfully sends webhook."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        # Mock successful response - must properly support context manager protocol
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            max_retries=0,  # No retries for this test
        )
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
            data={"success": True},
        )

        result = sender._send_with_retry("https://example.com/webhook", sender._build_payload(event), "after_stage")

        assert result is True
        mock_urlopen.assert_called_once()

    @patch('src.pipeline_events.urllib.request.urlopen')
    def test_webhook_sender_retry_on_failure(self, mock_urlopen):
        """AC: WebhookSender retries on failure."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent
        import urllib.error

        # Mock failed response then success - must properly support context manager
        mock_response_success = MagicMock()
        mock_response_success.status = 200
        mock_response_success.__enter__ = Mock(return_value=mock_response_success)
        mock_response_success.__exit__ = Mock(return_value=False)
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("url", 500, "Server Error", {}, None),
            mock_response_success,
        ]

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            max_retries=1,
            retry_base_delay=0.01,  # Fast retries for test
            retry_max_delay=0.1,
        )
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
        )

        # Should retry once and succeed
        with patch('time.sleep'):  # Don't actually sleep
            result = sender._send_with_retry("https://example.com/webhook", sender._build_payload(event), "after_stage")

        assert result is True
        assert mock_urlopen.call_count == 2

    @patch('src.pipeline_events.urllib.request.urlopen')
    def test_webhook_sender_all_retries_fail(self, mock_urlopen):
        """AC: WebhookSender returns False when all retries exhausted."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent
        import urllib.error

        # All requests fail
        mock_urlopen.side_effect = urllib.error.HTTPError("url", 500, "Server Error", {}, None)

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            max_retries=2,
            retry_base_delay=0.01,
            retry_max_delay=0.1,
        )
        sender = WebhookSender(config)

        event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
        )

        with patch('time.sleep'):  # Don't actually sleep
            result = sender._send_with_retry("https://example.com/webhook", sender._build_payload(event), "after_stage")

        assert result is False
        assert mock_urlopen.call_count == 3  # Initial + 2 retries

    def test_webhook_sender_get_url_event_specific(self):
        """AC: WebhookSender uses event-specific URL when configured."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            enabled=True,
            default_url="https://default.example.com/webhook",
            event_urls={
                "after_stage": "https://specific.example.com/after_stage",
            },
        )

        url = config.get_url_for_event("after_stage")
        assert url == "https://specific.example.com/after_stage"

    def test_webhook_sender_get_url_fallback_to_default(self):
        """AC: WebhookSender falls back to default URL for unknown events."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            enabled=True,
            default_url="https://default.example.com/webhook",
            event_urls={
                "after_stage": "https://specific.example.com/after_stage",
            },
        )

        # Unknown event type should use default
        url = config.get_url_for_event("unknown_event")
        assert url == "https://default.example.com/webhook"


class TestWebhookStageFiltering:
    """Test stage and event type filtering for webhooks (US-138-012)."""

    def test_stage_filter_config_empty_allows_all(self):
        """AC: Empty stage_filter allows all stages."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(enabled=True, default_url="https://example.com/webhook")

        assert config.should_send_event("after_stage", "MATCH") is True
        assert config.should_send_event("after_stage", "VIDEO_SEARCH") is True
        assert config.should_send_event("before_stage", "ANALYZE") is True

    def test_stage_filter_config_restricts_stages(self):
        """AC: stage_filter restricts webhooks to specified stages."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            stage_filter=["MATCH", "VIDEO_SEARCH"],
        )

        # Filtered stages should be allowed
        assert config.should_send_event("after_stage", "MATCH") is True
        assert config.should_send_event("after_stage", "VIDEO_SEARCH") is True

        # Non-filtered stages should be blocked
        assert config.should_send_event("after_stage", "ANALYZE") is False
        assert config.should_send_event("after_stage", "CAPTION") is False

    def test_event_type_filter_empty_allows_all(self):
        """AC: Empty event_type_filter allows all event types."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(enabled=True, default_url="https://example.com/webhook")

        assert config.should_send_event("after_stage", "MATCH") is True
        assert config.should_send_event("before_stage", "MATCH") is True
        assert config.should_send_event("on_pipeline_complete", "") is True

    def test_event_type_filter_restricts_events(self):
        """AC: event_type_filter restricts webhooks to specified event types."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            event_type_filter=["after_stage", "on_pipeline_complete"],
        )

        # Filtered events should be allowed
        assert config.should_send_event("after_stage", "MATCH") is True
        assert config.should_send_event("on_pipeline_complete", "") is True

        # Non-filtered events should be blocked
        assert config.should_send_event("before_stage", "MATCH") is False
        assert config.should_send_event("on_stage_error", "MATCH") is False

    def test_combined_filters_both_must_match(self):
        """AC: Both filters must match for event to be sent."""
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            stage_filter=["MATCH"],
            event_type_filter=["after_stage"],
        )

        # Both match
        assert config.should_send_event("after_stage", "MATCH") is True

        # Stage matches but event doesn't
        assert config.should_send_event("before_stage", "MATCH") is False

        # Event matches but stage doesn't
        assert config.should_send_event("after_stage", "VIDEO_SEARCH") is False

    def test_webhook_sender_respects_stage_filter(self):
        """AC: WebhookSender skips events for filtered stages."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            stage_filter=["MATCH"],  # Only MATCH stage
        )
        sender = WebhookSender(config)

        # MATCH stage should be sent
        match_event = PipelineEvent(
            event_type="after_stage",
            stage_name="MATCH",
            timestamp=1000.0,
            data={"success": True},
        )

        with patch.object(sender, '_send_with_retry', return_value=True) as mock_send:
            sender.on_event(match_event)
            mock_send.assert_called_once()

        # VIDEO_SEARCH stage should be filtered
        video_event = PipelineEvent(
            event_type="after_stage",
            stage_name="VIDEO_SEARCH",
            timestamp=1000.0,
            data={"success": True},
        )

        with patch.object(sender, '_send_with_retry') as mock_send:
            sender.on_event(video_event)
            mock_send.assert_not_called()

    def test_webhook_sender_respects_event_type_filter(self):
        """AC: WebhookSender skips events for filtered event types."""
        from src.config.sections.infrastructure import WebhookConfig
        from src.pipeline_events import WebhookSender, PipelineEvent

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            event_type_filter=["on_pipeline_complete"],  # Only completion events
        )
        sender = WebhookSender(config)

        # on_pipeline_complete should be sent
        complete_event = PipelineEvent(
            event_type="on_pipeline_complete",
            stage_name="",
            timestamp=1000.0,
            data={"success": True},
        )

        with patch.object(sender, '_send_with_retry', return_value=True) as mock_send:
            sender.on_event(complete_event)
            mock_send.assert_called_once()

        # before_stage should be filtered
        before_event = PipelineEvent(
            event_type="before_stage",
            stage_name="MATCH",
            timestamp=1000.0,
        )

        with patch.object(sender, '_send_with_retry') as mock_send:
            sender.on_event(before_event)
            mock_send.assert_not_called()


class TestPipelineWebhookIntegration:
    """Test webhook integration in PipelineOrchestrator."""

    def test_pipeline_webhook_enabled(self, pipeline_dir, mock_config):
        """AC: Pipeline initializes webhook when enabled in config."""
        from src.config.sections.infrastructure import WebhookConfig
        from unittest.mock import MagicMock

        # Configure webhook in pipeline config
        mock_pipeline_config = MagicMock()
        mock_pipeline_config.webhook = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
        )
        mock_config.pipeline = mock_pipeline_config

        with patch.object(PipelineOrchestrator, '_validate_runtime_environment', return_value=[]):
            pipeline = PipelineOrchestrator(mock_config, pipeline_dir)

        # Webhook sender should be initialized
        assert pipeline._webhook_sender is not None
        # Should be subscribed to event bus
        handlers = pipeline.event_bus.get_handlers('after_stage')
        assert any(h == pipeline._webhook_sender.on_event for h in handlers)

    def test_pipeline_webhook_disabled(self, pipeline_dir, mock_config):
        """AC: Pipeline skips webhook when disabled in config."""
        from src.config.sections.infrastructure import WebhookConfig
        from unittest.mock import MagicMock

        # Configure webhook as disabled
        mock_pipeline_config = MagicMock()
        mock_pipeline_config.webhook = WebhookConfig(enabled=False)
        mock_config.pipeline = mock_pipeline_config

        with patch.object(PipelineOrchestrator, '_validate_runtime_environment', return_value=[]):
            pipeline = PipelineOrchestrator(mock_config, pipeline_dir)

        # Webhook sender should not be initialized
        assert pipeline._webhook_sender is None
