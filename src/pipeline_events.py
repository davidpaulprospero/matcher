"""
Pipeline Event Bus

Decoupled event system for pipeline stage lifecycle notifications.
External subscribers (logging, metrics, monitoring) can register handlers
without modifying the orchestrator.

Usage:
    bus = PipelineEventBus()
    bus.subscribe('before_stage', my_handler)
    bus.emit(PipelineEvent(event_type='before_stage', stage_name='ANALYZE', timestamp=time.time()))
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Event hook callback type: receives a PipelineEvent
EventCallback = Callable[['PipelineEvent'], None]


@dataclass
class PipelineEvent:
    """Structured event emitted during pipeline lifecycle.

    Attributes:
        event_type: One of 'before_stage', 'after_stage', 'on_stage_error', 'on_pipeline_complete'
        stage_name: Name of the stage (empty string for pipeline-level events)
        timestamp: Unix timestamp when the event was created
        data: Arbitrary data dict (e.g., elapsed time, metrics)
        error: Optional error string if the event relates to a failure
    """
    event_type: str
    stage_name: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PipelineEventBus:
    """Decoupled event bus for pipeline stage lifecycle notifications.

    Manages event subscription and emission. Handlers registered via
    subscribe() are called in order of registration. A failing handler
    does not block subsequent handlers from executing.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventCallback]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventCallback) -> None:
        """Register a handler for a pipeline event type.

        Multiple handlers can be registered per event type. They are called
        in registration order when the event is emitted.

        Args:
            event_type: One of 'before_stage', 'after_stage', 'on_stage_error',
                        'on_pipeline_complete'
            handler: Function that receives a PipelineEvent
        """
        self._handlers[event_type].append(handler)

    def emit(self, event: PipelineEvent) -> None:
        """Emit a pipeline event, calling all registered handlers in order.

        Handler exceptions are logged but do not interrupt pipeline execution
        or prevent other handlers from being called.

        Args:
            event: The PipelineEvent to emit
        """
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.warning(
                    f"Event handler failed for {event.event_type}/{event.stage_name}: {e}"
                )

    def get_handlers(self, event_type: str) -> List[EventCallback]:
        """Return the list of handlers registered for a given event type.

        Args:
            event_type: The event type to look up

        Returns:
            List of registered handler callbacks (may be empty)
        """
        return list(self._handlers.get(event_type, []))
